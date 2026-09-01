from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from agave_chem.mappers.identical_fragments.identical_fragment_mapper import (
    IdenticalFragmentMapper,
)
from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper
from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.types import (
    AgaveChemMapperResult,
    ReactionInput,
    ReactionMapperResult,
)
from agave_chem.utils.logging_config import logger
from agave_chem.utils.reaction_balancing import (
    compute_unmapped_product_atom_islands,
    determine_one_to_one_correspondence,
)


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
        TemplateReactionMapper("template_default"),
    )


def _prepare_reaction_inputs(
    reaction_list: List[str],
    identical_fragment_mapper: IdenticalFragmentMapper,
    mcs_mapper: Optional[MCSReactionMapper] = None,
) -> Tuple[List[ReactionInput], List[List[Tuple[str, str]]]]:
    """
    Pre-compute ``ReactionInput`` objects for a batch of reactions.

    Strips identical fragments, optionally runs a conservative MCS mapping,
    computes unmapped product atom islands, and determines the
    ``one_to_one_correspondence`` flag per reaction.

    Args:
        reaction_list (List[str]): Raw reaction SMILES strings.
        identical_fragment_mapper (IdenticalFragmentMapper): Mapper used to
            strip and later re-add identical fragments.
        mcs_mapper (Optional[MCSReactionMapper]): If provided, used to compute
            a partial MCS mapping for each reaction.  If None, MCS and island
            detection are skipped.

    Returns:
        Tuple[List[ReactionInput], List[List[Tuple[str, str]]]]:
            - List of ``ReactionInput`` objects, one per input reaction.
            - Per-reaction lists of identical-fragment mapping pairs for
              later re-addition.
    """
    stripped_rxns, identical_fragments_mapping_list = (
        identical_fragment_mapper.create_identical_fragments_mapping_list(
            reaction_list
        )
    )

    reaction_inputs: List[ReactionInput] = []
    for original_smiles, stripped_smiles, identical_fragments in zip(
        reaction_list, stripped_rxns, identical_fragments_mapping_list
    ):
        mcs_mapped_smiles: Optional[str] = None
        islands: Dict[int, Set[int]] = {}

        if mcs_mapper is not None:
            mcs_result = mcs_mapper.map_reaction(stripped_smiles)
            if mcs_result.selected_mapping:
                mcs_mapped_smiles = mcs_result.selected_mapping
                try:
                    islands = compute_unmapped_product_atom_islands(
                        mcs_mapped_smiles.split(">>")[1]
                    )
                except ValueError:
                    islands = {}

        o2o = determine_one_to_one_correspondence(stripped_smiles, islands)

        reaction_inputs.append(
            ReactionInput(
                original_smiles=original_smiles,
                stripped_smiles=stripped_smiles,
                identical_fragments=identical_fragments,
                mcs_mapped_smiles=mcs_mapped_smiles,
                unmapped_product_atom_islands=islands,
                one_to_one_correspondence=o2o,
            )
        )

    return reaction_inputs, identical_fragments_mapping_list


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


def _extract_classification_summary(
    mapper_results: List[ReactionMapperResult],
    final_mapping: str,
) -> Tuple[str, str, Dict[str, List[Dict[str, Any]]]]:
    """
    Extract classification summary fields from template mapper results.

    Scans all mapper results for non-empty ``classification_info`` and extracts
    the entry matching ``final_mapping``.  Builds pipe-delimited summary strings
    from the unique ``class_str`` values and unique ``rxno_id`` values across all
    matching templates.

    Args:
        mapper_results (List[ReactionMapperResult]): Per-mapper results for a
            single reaction.
        final_mapping (str): The final selected mapped SMILES to look up in
            each mapper result's ``classification_info``.

    Returns:
        Tuple[str, str, Dict[str, List[Dict[str, Any]]]]:
            - Pipe-delimited string of unique class_str values (e.g. ``"1.1.1|2.5.1"``).
            - Pipe-delimited string of unique RXNO IDs (e.g. ``"RXNO:0000335|RXNO:0000357"``).
            - The full ``classification_info`` dict for ``final_mapping`` (empty if no match).
    """
    classification_info: Dict[str, List[Dict[str, Any]]] = {}
    for mr in mapper_results:
        if mr.classification_info and final_mapping in mr.classification_info:
            classification_info = {final_mapping: mr.classification_info[final_mapping]}
            break

    if not classification_info:
        return "", "", {}

    templates = classification_info.get(final_mapping, [])
    seen_class_strs: List[str] = []
    seen_rxno_ids: List[str] = []
    seen_class_set: Set[str] = set()
    seen_rxno_set: Set[str] = set()

    for entry in templates:
        cs = entry.get("class_str", "")
        if cs and cs not in seen_class_set:
            seen_class_set.add(cs)
            seen_class_strs.append(cs)
        for rxno in entry.get("rxno_classification", []):
            if isinstance(rxno, dict):
                rid = rxno.get("rxno_id", "")
            else:
                rid = str(rxno)
            if rid and rid not in seen_rxno_set:
                seen_rxno_set.add(rid)
                seen_rxno_ids.append(rid)

    return "|".join(seen_class_strs), "|".join(seen_rxno_ids), classification_info


def _extract_confidence(
    mapper_results: List[ReactionMapperResult],
) -> Optional[float]:
    """
    Extract the confidence score from the neural mapper's result.

    Scans all mapper results for the one whose ``mapping_type`` is ``"neural"``
    and returns its ``mapping_score``.  The neural mapper stores its confidence
    (product of per-atom assignment probabilities) in ``mapping_score``.

    Args:
        mapper_results (List[ReactionMapperResult]): Per-mapper results for a
            single reaction.

    Returns:
        Optional[float]: The neural mapper's confidence score, or ``None`` when
        no neural mapper result is present or its ``mapping_score`` is ``None``.
    """
    for mr in mapper_results:
        if mr.mapping_type == "neural" and mr.mapping_score is not None:
            score = mr.mapping_score
            if isinstance(score, (int, float)):
                return float(score)
    return None


def map_reactions_using_mappers(
    reaction_list: Union[str, List[str]],
    mappers_list: List[ReactionMapper],
    batch_size: int,
    return_detailed_mapper_info: bool = False,
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
        return_detailed_mapper_info (bool): If True, populate
            ``AgaveChemMapperResult.mapper_results`` with per-mapper
            ``ReactionMapperResult`` objects (including template classification
            metadata).  If False (default), ``mapper_results`` is left empty to
            reduce result size for callers that only need ``final_mapping``.

    Returns:
        List[AgaveChemMapperResult]: One result per input reaction, in the same
            order as the input.  Each result always contains ``final_mapping``,
            ``original_reaction``, ``confidence``, ``class_str``,
            ``rxno_classifications``, and ``classification_info``.
            ``confidence`` is populated from the neural mapper's
            ``mapping_score`` when a neural mapper is present; ``None``
            otherwise.  ``class_str``, ``rxno_classifications``, and
            ``classification_info`` are populated when a template mapper matches
            ``final_mapping``; empty otherwise.  When
            ``return_detailed_mapper_info`` is True, ``mapper_results`` also
            contains per-mapper ``ReactionMapperResult`` objects.
    """
    reaction_list, mappers_list, batch_size = _validate_and_normalize_input(
        reaction_list, mappers_list, batch_size
    )

    all_mapper_results_by_reaction: List[List[ReactionMapperResult]] = [
        [] for _ in reaction_list
    ]
    identical_fragment_mapper = IdenticalFragmentMapper("identical_fragment_helper")

    needs_mcs = any(
        mapper._mapper_type in ("neural", "template") for mapper in mappers_list
    )
    mcs_mapper = MCSReactionMapper("mcs_orchestrator", 0) if needs_mcs else None

    # Pre-compute ReactionInput objects once per batch (not per mapper)
    # to avoid duplicate MCS and identical-fragment work.
    batched_inputs: List[Tuple[List[ReactionInput], List[List[Tuple[str, str]]], int]] = []
    for i in range(0, len(reaction_list), batch_size):
        chunk = reaction_list[i : i + batch_size]
        reaction_inputs, identical_fragments_mapping_list = (
            _prepare_reaction_inputs(chunk, identical_fragment_mapper, mcs_mapper)
        )
        batched_inputs.append((reaction_inputs, identical_fragments_mapping_list, i))

    for mapper in mappers_list:
        for reaction_inputs, identical_fragments_mapping_list, i in batched_inputs:
            out = mapper.map_reactions(reaction_inputs)
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

        result_kwargs: dict = {
            "final_mapping": final_mapping,
            "original_reaction": original_reaction,
        }
        class_str, rxno_classifications, classification_info = (
            _extract_classification_summary(mapper_results, final_mapping)
        )
        if class_str:
            result_kwargs["class_str"] = class_str
        if rxno_classifications:
            result_kwargs["rxno_classifications"] = rxno_classifications
        if classification_info:
            result_kwargs["classification_info"] = classification_info
        confidence = _extract_confidence(mapper_results)
        if confidence is not None:
            result_kwargs["confidence"] = confidence
        if return_detailed_mapper_info:
            result_kwargs["mapper_results"] = mapper_results
        results.append(AgaveChemMapperResult(**result_kwargs))

    return results


def map_reactions(
    reaction_list: Union[str, List[str]],
    mappers_list: Optional[List[ReactionMapper]] = None,
    mapping_selection_mode: Union[str, Callable] = "weighted",
    batch_size: int = 500,
    return_detailed_mapper_info: bool = False,
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
        return_detailed_mapper_info (bool): If True, populate
            ``AgaveChemMapperResult.mapper_results`` with per-mapper
            ``ReactionMapperResult`` objects (including template classification
            metadata).  If False (default), ``mapper_results`` is left empty to
            reduce result size for callers that only need ``final_mapping``.

    Returns:
        List[AgaveChemMapperResult]: One result per input reaction, in the same
            order as the (deduplicated) input.  Each result always contains
            ``final_mapping``, ``original_reaction``, ``confidence``,
            ``class_str``, ``rxno_classifications``, and ``classification_info``.
            ``confidence`` is populated from the neural mapper's
            ``mapping_score`` when a neural mapper is present; ``None``
            otherwise.  ``class_str``, ``rxno_classifications``, and
            ``classification_info`` are populated when a template mapper matches
            ``final_mapping``; empty otherwise.  When
            ``return_detailed_mapper_info`` is True, ``mapper_results`` also
            contains per-mapper ``ReactionMapperResult`` objects.

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

    return map_reactions_using_mappers(
        reaction_list,
        mappers_list,
        batch_size,
        return_detailed_mapper_info=return_detailed_mapper_info,
    )
