"""
parallel_mcs_mapper.py — Parallel reaction atom-mapping using MCSReactionMapper.

Provides ``ParallelMCSReactionMapper``, a ``ReactionMapper`` subclass whose
``map_reactions`` spawns a pool of worker processes each with its own
``MCSReactionMapper`` instance, and ``map_reactions_parallel_mcs``, a
module-level convenience wrapper around the same pool logic.
"""

import multiprocessing as mp
import os
from typing import List, Optional, Union

from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper
from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.types import ReactionInput, ReactionMapperResult
from agave_chem.utils.logging_config import disable_library_logging

# ── Worker globals ───────────────────────────────────────────────────────────

_mcs_mapper: Optional[MCSReactionMapper] = None
_min_radius: int = 1
_min_radius_to_anchor_new_mapping: int = 3


def _init_worker(min_radius: int, min_radius_to_anchor_new_mapping: int) -> None:
    """
    Initialize the per-worker MCSReactionMapper instance.

    Called exactly once per worker process by ``multiprocessing.Pool``.
    Stores the mapper and mapping parameters in module-level globals so they
    are reused across all tasks handled by this worker.

    Args:
        min_radius (int): Smallest bond-radius to consider.
        min_radius_to_anchor_new_mapping (int): Below this radius,
            environments are only matched when they already contain at
            least one mapped atom.
    """
    global _mcs_mapper, _min_radius, _min_radius_to_anchor_new_mapping
    disable_library_logging()
    _mcs_mapper = MCSReactionMapper(mapper_name="mcs_parallel_worker")
    _min_radius = min_radius
    _min_radius_to_anchor_new_mapping = min_radius_to_anchor_new_mapping


def _map_one(rxn: str) -> ReactionMapperResult:
    """
    Atom-map a single reaction SMILES string using the worker-local mapper.

    Uses the module-level ``_mcs_mapper`` initialised by ``_init_worker``.

    Args:
        rxn (str): Reaction SMILES string to map.

    Returns:
        ReactionMapperResult: MCS-based mapping result. If the input is
        invalid or no valid mapping can be produced, a default empty result
        is returned.
    """
    return _mcs_mapper.map_reaction(  # type: ignore[union-attr]
        rxn,
        min_radius=_min_radius,
        min_radius_to_anchor_new_mapping=_min_radius_to_anchor_new_mapping,
    )


class ParallelMCSReactionMapper(ReactionMapper):
    """
    A ReactionMapper subclass that maps reactions in parallel using a pool
    of worker processes, each with its own MCSReactionMapper instance.

    ``map_reactions`` distributes work across a ``multiprocessing.Pool`` and
    returns results in input order. ``map_reaction`` falls back to a lazily
    initialised in-process mapper for single-reaction calls.
    """

    def __init__(
        self,
        mapper_name: str,
        mapper_weight: float = 3,
        workers: Optional[int] = None,
        chunksize: int = 50,
        min_radius: int = 1,
        min_radius_to_anchor_new_mapping: int = 3,
    ) -> None:
        """
        Initialize the ParallelMCSReactionMapper.

        Args:
            mapper_name (str): Unique name for this mapper instance.
            mapper_weight (float): Weight used for mapper selection (0–1000).
            workers (Optional[int]): Number of worker processes. Defaults to
                ``min(os.cpu_count() or 1, 16)``.
            chunksize (int): Number of reactions sent to each worker per chunk.
            min_radius (int): Smallest bond-radius to consider.
            min_radius_to_anchor_new_mapping (int): Below this radius,
                environments are only matched when they already contain at
                least one mapped atom.
        """
        super().__init__("mcs", mapper_name, mapper_weight)
        self._workers = workers or min(os.cpu_count() or 1, 16)
        self._chunksize = chunksize
        self._min_radius = min_radius
        self._min_radius_to_anchor_new_mapping = min_radius_to_anchor_new_mapping
        self._inner_mapper: Optional[MCSReactionMapper] = None

    def _get_inner_mapper(self) -> MCSReactionMapper:
        """
        Return the lazily initialised in-process MCSReactionMapper.

        The mapper is created on first access and reused for subsequent
        ``map_reaction`` calls on this instance.

        Returns:
            MCSReactionMapper: The initialised single-process mapper.
        """
        if self._inner_mapper is None:
            self._inner_mapper = MCSReactionMapper(f"{self._mapper_name}_inner")
        return self._inner_mapper

    def map_reaction(
        self, reaction_smiles: Union[str, ReactionInput]
    ) -> ReactionMapperResult:
        """
        Atom-map a single reaction SMILES string in-process.

        Delegates to a lazily initialised ``MCSReactionMapper`` stored on
        this instance. No parallelism is applied; prefer ``map_reactions``
        for bulk workloads.

        Args:
            reaction_smiles (Union[str, ReactionInput]): Reaction SMILES string
                or ``ReactionInput`` to map.

        Returns:
            ReactionMapperResult: MCS-based mapping result. If the input is
                invalid or no valid mapping can be produced, a default empty
                result is returned.
        """
        return self._get_inner_mapper().map_reaction(
            reaction_smiles,
            min_radius=self._min_radius,
            min_radius_to_anchor_new_mapping=self._min_radius_to_anchor_new_mapping,
        )

    def map_reactions(
        self, reaction_smiles_list: Union[List[str], List[ReactionInput]]
    ) -> List[ReactionMapperResult]:
        """
        Atom-map a list of reaction SMILES strings in parallel.

        Spawns a ``multiprocessing.Pool`` of worker processes, each with its
        own initialised ``MCSReactionMapper``. Results are returned in the
        same order as ``reaction_smiles_list``.

        Args:
            reaction_smiles_list (Union[List[str], List[ReactionInput]]): Reaction
                SMILES strings or ``ReactionInput`` objects to map.

        Returns:
            List[ReactionMapperResult]: MCS-based mapping results in the same
                order as ``reaction_smiles_list``. Reactions that fail to map
                have an empty string for ``selected_mapping``.
        """
        smiles_list = [self._get_smiles(item) for item in reaction_smiles_list]
        with mp.Pool(
            processes=self._workers,
            initializer=_init_worker,
            initargs=(self._min_radius, self._min_radius_to_anchor_new_mapping),
        ) as pool:
            return list(pool.imap(_map_one, smiles_list, chunksize=self._chunksize))


def map_reactions_parallel_mcs(
    reaction_smiles: List[str],
    workers: Optional[int] = None,
    chunksize: int = 50,
    min_radius: int = 1,
    min_radius_to_anchor_new_mapping: int = 3,
) -> List[ReactionMapperResult]:
    """
    Atom-map a list of reaction SMILES strings in parallel using MCSReactionMapper.

    Spawns a pool of worker processes, each with its own initialised
    ``MCSReactionMapper`` instance. Results are returned in the same order
    as ``reaction_smiles``.

    Args:
        reaction_smiles (List[str]): Reaction SMILES strings to map.
        workers (Optional[int]): Number of worker processes. Defaults to
            ``min(os.cpu_count() or 1, 16)``.
        chunksize (int): Number of reactions sent to each worker per chunk.
        min_radius (int): Smallest bond-radius to consider.
        min_radius_to_anchor_new_mapping (int): Below this radius,
            environments are only matched when they already contain at
            least one mapped atom.

    Returns:
        List[ReactionMapperResult]: MCS-based mapping results in the same
            order as ``reaction_smiles``. Reactions that fail to map have an
            empty string for ``selected_mapping``.
    """
    mapper = ParallelMCSReactionMapper(
        "mcs_parallel_convenience",
        workers=workers,
        chunksize=chunksize,
        min_radius=min_radius,
        min_radius_to_anchor_new_mapping=min_radius_to_anchor_new_mapping,
    )
    return mapper.map_reactions(reaction_smiles)
