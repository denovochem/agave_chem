from typing import List, Dict
from rdkit import Chem
from rdchiral import main as rdc

from agave_chem.utils.logging_config import logger


def expand_first_bracketed_list(input_string: str) -> List[str]:
    """
    Finds the first bracketed section containing a field with commas
    (e.g., [C;H1,H2;+0:6]) and expands that specific field.

    It iterates through brackets until it finds one needing expansion, expands
    only the *first* comma-separated field within that bracket, and returns
    all possible strings resulting from that single expansion step.

    Args:
        input_string: The string to process.

    Returns:
        A list of strings, where each string represents one expansion
        of the first found comma-separated field within a bracket.
        If no such field is found in any bracket, returns a list containing
        just the original string.
    """

    bracket_matches = []
    open_bracket_count, close_bracket_count = 0, 0
    start_index, end_index = 0, 0
    for i, char in enumerate(input_string):
        if char == "[":
            if open_bracket_count == 0:
                start_index = i
            open_bracket_count += 1

        if char == "]":
            end_index = i + 1
            close_bracket_count += 1

        if open_bracket_count == close_bracket_count and open_bracket_count != 0:
            bracket_matches.append(
                [input_string[start_index:end_index], start_index, end_index]
            )
            open_bracket_count = 0
            close_bracket_count = 0

    for original_bracket_with_brackets, start_index, end_index in bracket_matches:
        original_bracket_content = original_bracket_with_brackets[1:-1]
        has_map = False
        if len(original_bracket_content.split(":")) > 1:
            sub_str = original_bracket_content.split(":")[-1]
            if sub_str.isdigit() or (sub_str.startswith("-") and sub_str[1:].isdigit()):
                has_map = True
                map_num = original_bracket_content.split(":")[-1]
        if has_map:
            original_bracket_content_without_map = ":".join(
                original_bracket_content.split(":")[:-1]
            )
        else:
            original_bracket_content_without_map = original_bracket_content

        fields = original_bracket_content_without_map.split(";")

        field_to_expand_index = -1
        alternatives = []
        for i, field in enumerate(fields):
            if "," in field:
                field_to_expand_index = i
                alternatives = field.split(",")
                break

        if field_to_expand_index != -1:
            results = []
            for alt in alternatives:
                new_fields = (
                    fields[:field_to_expand_index]
                    + [alt]
                    + fields[field_to_expand_index + 1 :]
                )
                new_bracket_content = ";".join(new_fields)
                if has_map:
                    new_bracket_content += ":" + map_num
                new_string = (
                    input_string[:start_index]
                    + "["
                    + new_bracket_content
                    + "]"
                    + input_string[end_index:]
                )
                results.append(new_string)

            return results

    return [input_string]


def expand_all_recursively(input_string: str) -> List[str]:
    """
    Recursively applies _expand_first_bracketed_list to a SMIRKS string until
    no more expansions based on comma-separated fields within brackets are
    possible. It collects all final, fully expanded combinations.

    Args:
        input_string: The SMIRKS string to start the expansion from.

    Returns:
        A list of all fully expanded SMIRKS strings.
    """

    expanded_list = expand_first_bracketed_list(input_string)

    if len(expanded_list) == 1 and expanded_list[0] == input_string:
        return expanded_list

    else:
        all_final_strings = []
        for next_string in expanded_list:
            all_final_strings.extend(expand_all_recursively(next_string))

        return all_final_strings


def initialize_template_data(named_reactions: Dict) -> List:
    """
    Initialize reaction template data by processing SMIRKS patterns from named reactions.

    This function takes a list of named reactions, reverses their SMIRKS patterns,
    expands them recursively, and creates RDChiral reaction objects for further processing.

    Args:
        named_reactions (list): List of dictionaries containing named reaction information.
                                Each dictionary should have a 'smirks' key.

    Returns:
        list: List of lists, where each inner list contains:
                [0] - List of product SMARTS molecules
                [1] - List of reactant SMARTS molecules
                [2] - RDChiral reaction object
                [3] - Parent SMIRKS
                [4] - Child SMIRKS
    """
    all_smirks = {}
    for reaction in named_reactions:
        smirks_list = []
        smirks = (
            reaction["smirks"].split(">>")[1] + ">>" + reaction["smirks"].split(">>")[0]
        )

        if len(expand_all_recursively(smirks)) < 100:
            smirks_list.extend(expand_all_recursively(smirks))

        if smirks in all_smirks:
            existing_patters = all_smirks[smirks]
            existing_patters.extend(smirks_list)
            all_smirks[smirks] = sorted(list(set(existing_patters)))
        else:
            all_smirks[smirks] = smirks_list

    rdc_info = []
    for k, v in all_smirks.items():
        for smirk in v:
            products_smarts = [
                Chem.MolFromSmarts(ele) for ele in smirk.split(">>")[0].split(".")
            ]
            reactants_smarts = [
                Chem.MolFromSmarts(ele) for ele in smirk.split(">>")[1].split(".")
            ]

            try:
                rdc_rxn = rdc.rdchiralReaction(smirk)
            except Exception as e:
                logger.warning(f"Error converting smirks to rdchiral reaction: {e}")
                continue

            rdc_info.append([products_smarts, reactants_smarts, rdc_rxn, k, smirk])

    return rdc_info
