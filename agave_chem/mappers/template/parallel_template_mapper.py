"""
parallel_template_mapper.py — Parallel reaction atom-mapping using TemplateReactionMapper.

Provides ``ParallelTemplateReactionMapper``, a ``ReactionMapper`` subclass whose
``map_reactions`` spawns a pool of worker processes each with its own initialised
``TemplateReactionMapper`` instance, and ``map_reactions_parallel_template``, a
module-level convenience wrapper around the same pool logic.
"""

import multiprocessing as mp
from typing import List, Optional

from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.template.template_mapper import TemplateReactionMapper
from agave_chem.mappers.types import ReactionMapperResult
from agave_chem.utils.logging_config import disable_library_logging

# ── Worker globals ───────────────────────────────────────────────────────────

_template_mapper: Optional[TemplateReactionMapper] = None
_apply_multiple_smirks: bool = True
_num_smirks_to_apply: int = 2


def _init_worker(apply_multiple_smirks: bool, num_smirks_to_apply: int) -> None:
    """
    Initialize the per-worker TemplateReactionMapper instance.

    Called exactly once per worker process by ``multiprocessing.Pool``.
    Stores the mapper and mapping parameters in module-level globals so they
    are reused across all tasks handled by this worker.

    Args:
        apply_multiple_smirks (bool): Whether to apply multiple SMIRKS patterns
            to the same reaction.
        num_smirks_to_apply (int): Number of SMIRKS patterns to apply per
            reaction.
    """
    global _template_mapper, _apply_multiple_smirks, _num_smirks_to_apply
    disable_library_logging()
    _template_mapper = TemplateReactionMapper("template_parallel_worker")
    _apply_multiple_smirks = apply_multiple_smirks
    _num_smirks_to_apply = num_smirks_to_apply


def _map_one(rxn: str) -> ReactionMapperResult:
    """
    Atom-map a single reaction SMILES string using the worker-local mapper.

    Uses the module-level ``_template_mapper`` initialised by ``_init_worker``.
    Returns the full ``ReactionMapperResult`` so that ``classification_info``
    and ``possible_mappings`` are preserved across process boundaries.

    Args:
        rxn (str): Reaction SMILES string to map.

    Returns:
        ReactionMapperResult: Template-based mapping result, including
        classification metadata. If the input is invalid or no valid mapping
        can be produced, a default empty result is returned.
    """
    return _template_mapper.map_reaction(  # type: ignore[union-attr]
        rxn, _apply_multiple_smirks, _num_smirks_to_apply
    )


class ParallelTemplateReactionMapper(ReactionMapper):
    """
    A ReactionMapper subclass that maps reactions in parallel using a pool of
    worker processes, each with its own initialised TemplateReactionMapper instance.

    ``map_reactions`` distributes work across a ``multiprocessing.Pool`` and
    returns results in input order. ``map_reaction`` falls back to a lazily
    initialised in-process mapper for single-reaction calls.
    """

    def __init__(
        self,
        mapper_name: str,
        mapper_weight: float = 3,
        workers: int = 8,
        chunksize: int = 50,
        apply_multiple_smirks: bool = True,
        num_smirks_to_apply: int = 2,
    ) -> None:
        """
        Initialize the ParallelTemplateReactionMapper.

        Args:
            mapper_name (str): Unique name for this mapper instance.
            mapper_weight (float): Weight used for mapper selection (0–1000).
            workers (int): Number of worker processes in the pool.
            chunksize (int): Number of reactions sent to each worker per chunk.
            apply_multiple_smirks (bool): Whether to apply multiple SMIRKS
                patterns to the same reaction.
            num_smirks_to_apply (int): Number of SMIRKS patterns to apply per
                reaction.
        """
        super().__init__("template", mapper_name, mapper_weight)
        self._workers = workers
        self._chunksize = chunksize
        self._apply_multiple_smirks = apply_multiple_smirks
        self._num_smirks_to_apply = num_smirks_to_apply
        self._inner_mapper: Optional[TemplateReactionMapper] = None

    def _get_inner_mapper(self) -> TemplateReactionMapper:
        """
        Return the lazily initialised in-process TemplateReactionMapper.

        The mapper is created on first access and reused for subsequent
        ``map_reaction`` calls on this instance.

        Returns:
            TemplateReactionMapper: The initialised single-process mapper.
        """
        if self._inner_mapper is None:
            self._inner_mapper = TemplateReactionMapper(f"{self._mapper_name}_inner")
        return self._inner_mapper

    def map_reaction(self, reaction_smiles: str) -> ReactionMapperResult:
        """
        Atom-map a single reaction SMILES string in-process.

        Delegates to a lazily initialised ``TemplateReactionMapper`` stored on
        this instance. No parallelism is applied; prefer ``map_reactions`` for
        bulk workloads.

        Args:
            reaction_smiles (str): Reaction SMILES string to map.

        Returns:
            ReactionMapperResult: Template-based mapping result. If the input
                is invalid or no valid mapping can be produced, a default empty
                result is returned.
        """
        return self._get_inner_mapper().map_reaction(
            reaction_smiles,
            self._apply_multiple_smirks,
            self._num_smirks_to_apply,
        )

    def map_reactions(
        self, reaction_smiles_list: List[str]
    ) -> List[ReactionMapperResult]:
        """
        Atom-map a list of reaction SMILES strings in parallel.

        Spawns a ``multiprocessing.Pool`` of worker processes, each with its
        own initialised ``TemplateReactionMapper``. Results are returned in the
        same order as ``reaction_smiles_list``.

        Args:
            reaction_smiles_list (List[str]): Reaction SMILES strings to map.

        Returns:
            List[ReactionMapperResult]: Template-based mapping results in the
                same order as ``reaction_smiles_list``. Reactions that fail to
                map have an empty string for ``selected_mapping``.
        """
        with mp.Pool(
            processes=self._workers,
            initializer=_init_worker,
            initargs=(self._apply_multiple_smirks, self._num_smirks_to_apply),
        ) as pool:
            return list(
                pool.imap(_map_one, reaction_smiles_list, chunksize=self._chunksize)
            )


def map_reactions_parallel_template(
    reaction_smiles: List[str],
    workers: int = 8,
    chunksize: int = 50,
    apply_multiple_smirks: bool = True,
    num_smirks_to_apply: int = 2,
) -> List[ReactionMapperResult]:
    """
    Atom-map a list of reaction SMILES strings in parallel using TemplateReactionMapper.

    Spawns a pool of worker processes, each with its own initialised
    ``TemplateReactionMapper`` instance. Results are returned in the same order
    as ``reaction_smiles``. Only the template-based mapping is returned; no MCS
    fallback is applied for reactions that fail to map.

    Args:
        reaction_smiles (List[str]): Reaction SMILES strings to map.
        workers (int): Number of worker processes.
        chunksize (int): Number of reactions sent to each worker per chunk.
        apply_multiple_smirks (bool): Whether to apply multiple SMIRKS patterns
            to the same reaction.
        num_smirks_to_apply (int): Number of SMIRKS patterns to apply per
            reaction.

    Returns:
        List[ReactionMapperResult]: Template-based mapping results in the same
        order as ``reaction_smiles``, including ``classification_info`` and
        ``possible_mappings``. Reactions that fail to map have an empty string
        for ``selected_mapping``.
    """
    with mp.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(apply_multiple_smirks, num_smirks_to_apply),
    ) as pool:
        return list(pool.imap(_map_one, reaction_smiles, chunksize=chunksize))
