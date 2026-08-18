from functools import lru_cache
from typing import Callable, List, Optional, Tuple, Union

from agave_chem.mappers.identical_fragments.identical_fragment_mapper import (
    IdenticalFragmentMapper,
)
from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.types import AgaveChemMapperResult, ReactionMapperResult
from agave_chem.utils.logging_config import logger


def _validate_and_normalize_input(
    reaction_list: Union[str, List[str]],
    mappers_list: Optional[List[ReactionMapper]],
    batch_size: int,
) -> Tuple[List[str], List[ReactionMapper], int]:
    """
    Validate and normalize inputs for reaction mapping.

    Converts a single reaction string into a one-element list, deduplicates
    reactions while preserving order, and validates all arguments.

    Args:
        reaction_list (Union[str, List[str]]): A single reaction SMILES string or
            a list of reaction SMILES strings.
        mappers_list (Optional[List[ReactionMapper]]): A list of ReactionMapper
            instances, or None.
        batch_size (int): Number of reactions to process per batch.

    Returns:
        Tuple[List[str], List[ReactionMapper], int]: The normalized reaction list
            (deduplicated, order-preserved), the mappers list, and the batch size.

    Raises:
        ValueError: If reaction_list is empty, contains non-strings, or if
            mappers_list is empty or has duplicate mapper names.
        TypeError: If a mapper is not a ReactionMapper instance, or if
            batch_size is not an integer.
        ValueError: If batch_size is not between 1 and 1000.
    """
    if isinstance(reaction_list, str):
        reaction_list = [reaction_list]

    if not isinstance(reaction_list, list):
        raise TypeError(
            "Invalid input: reaction_list must be a string or a non-empty list of strings."
        )
    if len(reaction_list) == 0:
        raise ValueError(
            "Invalid input: reaction_list must be a string or a non-empty list of strings."
        )
    for reaction in reaction_list:
        if not isinstance(reaction, str):
            raise TypeError(
                "Invalid input: reaction_list must be a string or a non-empty list of strings."
            )

    if len(reaction_list) != len(set(reaction_list)):
        logger.warning("Removing duplicate reactions from reaction_list.")
        reaction_list = list(dict.fromkeys(reaction_list))

    if not isinstance(mappers_list, list) or len(mappers_list) == 0:
        raise ValueError(
            "Invalid input: mappers_list must be a non-empty list of ReactionMapper instances."
        )

    seen_mappers: List[str] = []
    for mapper in mappers_list:
        if not isinstance(mapper, ReactionMapper):
            raise TypeError(
                f"Invalid mapper: {mapper} is not an instance of ReactionMapper."
            )
        if mapper.mapper_name in seen_mappers:
            raise ValueError(f"Duplicate mapper name: {mapper.mapper_name}.")
        seen_mappers.append(mapper.mapper_name)

    if not isinstance(batch_size, int):
        raise TypeError("Invalid input: batch_size must be an integer.")
    if batch_size <= 0 or batch_size > 1000:
        raise ValueError("Invalid input: batch_size must be an integer between 1-1000.")

    return reaction_list, mappers_list, batch_size


@lru_cache(maxsize=1)
def _get_default_mappers() -> Tuple[ReactionMapper, ...]:
    """
    Create and cache the default set of reaction mappers.

    Returns a tuple of ReactionMapper instances (neural + template) used when
    no explicit mappers_list is provided to ``map_reactions``.  The result is
    cached via ``lru_cache`` so heavy model-loading happens only once.

    Returns:
        Tuple[ReactionMapper, ...]: A tuple containing the default
            NeuralReactionMapper and TemplateReactionMapper instances.
    """
    from agave_chem.mappers.neural.neural_mapper import NeuralReactionMapper
    from agave_chem.mappers.template.template_mapper import TemplateReactionMapper

    return (
        NeuralReactionMapper(mapper_name="neural_mapper", mapper_weight=1),
        TemplateReactionMapper("expert_default"),
    )


def _resolve_identical_fragments(
    mapper_results: List[ReactionMapperResult],
    identical_fragments_mapping_list: List[List[Tuple[str, str]]],
    identical_fragment_mapper: IdenticalFragmentMapper,
    base_index: int,
    all_mapper_results_by_reaction: List[List[ReactionMapperResult]],
) -> None:
    """
    Resolve identical-fragment mappings and append results to the per-reaction list.

    For each mapper result in a batch, re-adds previously stripped identical
    fragments to the selected mapping (if both exist), then appends the result
    to the appropriate position in ``all_mapper_results_by_reaction``.

    Args:
        mapper_results (List[ReactionMapperResult]): Results from the mapper for
            the current batch.
        identical_fragments_mapping_list (List[List[Tuple[str, str]]]):
            Per-reaction lists of ``(reactant_smiles, product_smiles)`` pairs
            produced by ``create_identical_fragments_mapping_list``.
        identical_fragment_mapper (IdenticalFragmentMapper): The mapper instance
            used to resolve fragments.
        base_index (int): The index in ``all_mapper_results_by_reaction``
            corresponding to the first reaction in this batch.
        all_mapper_results_by_reaction (List[List[ReactionMapperResult]]): The
            accumulator list to append resolved results into.  Modified in place.
    """
    for j, (reaction, identical_fragments) in enumerate(
        zip(mapper_results, identical_fragments_mapping_list)
    ):
        if reaction.selected_mapping and identical_fragments:
            reaction.selected_mapping = (
                identical_fragment_mapper.resolve_identical_fragments_mapping_list(
                    [reaction.selected_mapping],
                    [identical_fragments],
                )[0]
            )
        all_mapper_results_by_reaction[base_index + j].append(reaction)


def map_reactions_using_mappers(
    reaction_list: Union[str, List[str]],
    mappers_list: List[ReactionMapper],
    batch_size: int,
) -> List[AgaveChemMapperResult]:
    """
    Run multiple reaction mappers over a list of reactions with batch processing.

    For each mapper, reactions are processed in batches of ``batch_size``.
    Identical fragments are stripped before mapping and re-added afterward.
    The final mapping for each reaction is the last non-empty result across all
    mappers (in order).

    Args:
        reaction_list (Union[str, List[str]]): A single reaction SMILES string or
            a list of reaction SMILES strings.
        mappers_list (List[ReactionMapper]): A list of ReactionMapper instances
            to run.  Must be non-empty with unique mapper names.
        batch_size (int): Number of reactions to process per batch (1-1000).

    Returns:
        List[AgaveChemMapperResult]: One result per input reaction, in the same
        order as the input.  Each result contains the final mapping, the original
        reaction SMILES, and per-mapper results.

    Raises:
        ValueError: If reaction_list is empty or contains non-strings, if
            mappers_list is empty or has duplicate mapper names, or if
            batch_size is out of range.
        TypeError: If a mapper is not a ReactionMapper instance, or if
            batch_size is not an integer.
    """
    reaction_list, mappers_list, batch_size = _validate_and_normalize_input(
        reaction_list, mappers_list, batch_size
    )

    all_mapper_results_by_reaction: List[List[ReactionMapperResult]] = [
        [] for _ in reaction_list
    ]
    identical_fragment_mapper = IdenticalFragmentMapper("identical_fragment_helper")
    for mapper in mappers_list:
        for i in range(0, len(reaction_list), batch_size):
            chunk = reaction_list[i : i + batch_size]
            new_rxns, identical_fragments_mapping_list = (
                identical_fragment_mapper.create_identical_fragments_mapping_list(chunk)
            )
            out = mapper.map_reactions(new_rxns)
            _resolve_identical_fragments(
                out,
                identical_fragments_mapping_list,
                identical_fragment_mapper,
                i,
                all_mapper_results_by_reaction,
            )

    results: List[AgaveChemMapperResult] = []
    for original_reaction, mapper_results in zip(
        reaction_list, all_mapper_results_by_reaction
    ):
        final_mapping = ""
        for mapper_result in reversed(mapper_results):
            if mapper_result.selected_mapping:
                final_mapping = mapper_result.selected_mapping
                break

        results.append(
            AgaveChemMapperResult(
                final_mapping=final_mapping,
                original_reaction=original_reaction,
                mapper_results=mapper_results,
            )
        )

    return results


def map_reactions(
    reaction_list: Union[str, List[str]],
    mappers_list: Optional[List[ReactionMapper]] = None,
    mapping_selection_mode: Union[str, Callable] = "weighted",
    batch_size: int = 500,
) -> List[AgaveChemMapperResult]:
    """
    Map atom-to-atom correspondences for a list of reaction SMILES strings.

    This is the main public entry point for agave_chem.  When no mappers are
    provided, a default set (neural + template) is lazily loaded and cached.
    Multiple mappers are run in sequence; the final mapping for each reaction is
    the last non-empty result across all mappers.

    Args:
        reaction_list (Union[str, List[str]]): A single reaction SMILES string or
            a list of reaction SMILES strings.  Duplicates are removed with a
            warning, preserving first-occurrence order.
        mappers_list (Optional[List[ReactionMapper]]): A list of ReactionMapper
            instances.  If None or empty, default mappers are used.
        mapping_selection_mode (Union[str, Callable]): Strategy for selecting the
            final mapping across mappers.  Currently validated but not yet
            implemented; reserved for future use.
        batch_size (int): Number of reactions to process per batch (1-1000).
            Defaults to 500.

    Returns:
        List[AgaveChemMapperResult]: One result per input reaction, in the same
        order as the (deduplicated) input.  Each result contains the final
        mapping, the original reaction SMILES, and per-mapper results.

    Raises:
        ValueError: If reaction_list is empty or contains non-strings, if
            mappers_list is empty or has duplicate mapper names, if
            mapping_selection_mode is not a string or callable, or if batch_size
            is out of range.
        TypeError: If a mapper is not a ReactionMapper instance, if
            mapping_selection_mode is not a string or callable, or if batch_size
            is not an integer.
    """
    if not mappers_list:
        mappers_list = list(_get_default_mappers())

    if not isinstance(mapping_selection_mode, str) and not callable(
        mapping_selection_mode
    ):
        raise TypeError(
            "Invalid input: mapping_selection_mode must be a string or function."
        )

    return map_reactions_using_mappers(reaction_list, mappers_list, batch_size)
