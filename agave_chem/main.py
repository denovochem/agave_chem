from typing import List, TypedDict

from agave_chem.mappers.identical_fragments.identical_fragment_mapper import (
    IdenticalFragmentMapper,
)
from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper
from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.template.template_mapper import TemplateReactionMapper
from agave_chem.mappers.types import ReactionMapperResult
from agave_chem.utils.logging_config import logger


class AgaveChemMapperResult(TypedDict):
    final_mapping: str
    original_reaction: str
    mapper_results: List[ReactionMapperResult]


def map_reactions_using_mappers(
    reaction_list: List[str],
    mappers_list: List[ReactionMapper],
    batch_size: int,
) -> List[AgaveChemMapperResult]:
    """ """
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
            for j, (reaction, identical_fragments) in enumerate(
                zip(out, identical_fragments_mapping_list)
            ):
                if reaction["selected_mapping"] and identical_fragments:
                    reaction["selected_mapping"] = (
                        identical_fragment_mapper.resolve_identical_fragments_mapping_list(
                            [reaction["selected_mapping"]],
                            [identical_fragments],
                        )[0]
                    )

                all_mapper_results_by_reaction[i + j].append(reaction)

    results: List[AgaveChemMapperResult] = []
    for original_reaction, mapper_results in zip(
        reaction_list, all_mapper_results_by_reaction
    ):
        final_mapping = ""
        for mapper_result in reversed(mapper_results):
            if mapper_result["selected_mapping"]:
                final_mapping = mapper_result["selected_mapping"]
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
    reaction_list: List[str],
    mappers_list: List[ReactionMapper] = [],
    mapping_selection_mode: str = "weighted",
    batch_size: int = 500,
) -> List[AgaveChemMapperResult]:
    """ """
    if not mappers_list:
        mappers_list = [
            MCSReactionMapper("mcs_default"),
            TemplateReactionMapper("expert_default"),
        ]

    if isinstance(reaction_list, str):
        reaction_list = [reaction_list]

    if not isinstance(reaction_list, list):
        raise ValueError(
            "Invalid input: reaction_list must be a string or a non-empty list of strings."
        )
    if isinstance(reaction_list, list):
        if len(reaction_list) == 0:
            raise ValueError(
                "Invalid input: reaction_list must be a string or a non-empty list of strings."
            )
        for reaction in reaction_list:
            if not isinstance(reaction, str):
                raise ValueError(
                    "Invalid input: reaction_list must be a string or a non-empty list of strings."
                )
    if len(reaction_list) != len(set(reaction_list)):
        logger.warning("Removing duplicate reactions from reaction_list.")
        reaction_list = list(set(reaction_list))

    if not isinstance(mappers_list, list) or len(mappers_list) == 0:
        raise ValueError(
            "Invalid input: mappers_list must be a non-empty list of ReactionMapper instances."
        )

    seen_mappers = []
    for mapper in mappers_list:
        if not isinstance(mapper, ReactionMapper):
            raise ValueError(
                f"Invalid mapper: {mapper} is not an instance of ReactionMapper."
            )
        if mapper.mapper_name in seen_mappers:
            raise ValueError(f"Duplicate mapper name: {mapper.mapper_name}.")
        seen_mappers.append(mapper.mapper_name)

    if not isinstance(mapping_selection_mode, str) and not callable(
        mapping_selection_mode
    ):
        raise ValueError(
            "Invalid input: mapping_selection_mode must be a string or function."
        )

    if not isinstance(batch_size, int):
        raise TypeError("Invalid input: batch_size must be an integer.")
    if batch_size <= 0 or batch_size > 1000:
        raise ValueError("Invalid input: batch_size must be an integer between 1-1000.")

    mapper_results = map_reactions_using_mappers(
        reaction_list, mappers_list, batch_size
    )

    if not mapper_results:
        raise ValueError("Invalid input: batch_size must be an integer between 1-1000.")

    return mapper_results
