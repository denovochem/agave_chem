from typing import Dict, List

from agave_chem.mappers.identical_fragments.identical_fragment_mapper import (
    create_identical_fragments_mapping_list,
    resolve_identical_fragments_mapping_dict,
)
from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper
from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.template.template_mapper import ExpertReactionMapper
from agave_chem.utils.logging_config import logger


def map_reactions_using_mappers(
    reaction_list: List[str],
    mappers_list: List[ReactionMapper],
    batch_size: int,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """ """
    mappers_out_dict = {}
    for mapper in mappers_list:
        for i in range(0, len(reaction_list), batch_size):
            chunk = reaction_list[i : i + batch_size]
            new_rxns, identical_fragments_mapping_list = (
                create_identical_fragments_mapping_list(chunk)
            )
            out = mapper.map_reactions(new_rxns)
            for reaction, identical_fragments in zip(
                out, identical_fragments_mapping_list
            ):
                if not reaction["mapping"] or not identical_fragments:
                    final_mapping = reaction["mapping"]
                else:
                    final_mapping = resolve_identical_fragments_mapping_dict(
                        [reaction["mapping"]], [identical_fragments]
                    )
                reaction["mapping"] = final_mapping
        mappers_out_dict[mapper.mapper_name] = {
            "out": out,
        }
    return mappers_out_dict


def map_reactions(
    reaction_list: List[str],
    mappers_list: List[ReactionMapper] = [],
    mapping_selection_mode: str = "weighted",
    batch_size: int = 500,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """ """
    if not mappers_list:
        mappers_list = [
            MCSReactionMapper("mcs_default"),
            ExpertReactionMapper("expert_default"),
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

    mappers_out_dict = map_reactions_using_mappers(
        reaction_list, mappers_list, batch_size
    )

    if not mappers_out_dict:
        raise ValueError("Invalid input: batch_size must be an integer between 1-1000.")

    mappers_out_dict = map_reactions_using_mappers(
        reaction_list, mappers_list, batch_size
    )

    return mappers_out_dict
