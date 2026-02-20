from typing import Dict, List, Optional, Tuple

from rdchiral import main as rdc
from rdkit import Chem

from agave_chem.utils.logging_config import logger


def has_top_level_comma(s: str) -> bool:
    """Check if string has a comma at top level (not inside [] or ())."""
    depth = 0
    for char in s:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            return True
    return False


def split_top_level(s: str, delimiter: str) -> List[str]:
    """Split string by delimiter only at top level (not inside [] or ())."""
    result = []
    current = []
    depth = 0

    for char in s:
        if char in "([":
            depth += 1
            current.append(char)
        elif char in ")]":
            depth -= 1
            current.append(char)
        elif char == delimiter and depth == 0:
            result.append("".join(current))
            current = []
        else:
            current.append(char)

    result.append("".join(current))
    return result


def find_innermost_bracket_with_comma(s: str) -> Optional[Tuple[int, int]]:
    """
    Find the innermost (first-closing) bracket that contains a top-level comma.
    Returns (start_index, end_index) or None if not found.
    """
    bracket_stack = []

    for i, char in enumerate(s):
        if char == "[":
            bracket_stack.append(i)
        elif char == "]":
            if bracket_stack:
                start = bracket_stack.pop()
                content = s[start + 1 : i]
                if has_top_level_comma(content):
                    return (start, i + 1)

    return None


def expand_first_bracketed_list(input_string: str) -> List[str]:
    """
    Finds the innermost bracket containing a comma and expands it.
    Preserves other semicolon-separated fields in the bracket.
    """
    match = find_innermost_bracket_with_comma(input_string)

    if match is None:
        return [input_string]

    start, end = match
    bracket_content = input_string[start + 1 : end - 1]

    # Handle atom map number (e.g., ":3" at the end)
    map_suffix = ""
    if ":" in bracket_content:
        parts = bracket_content.rsplit(":", 1)
        potential_map = parts[1]
        if potential_map.lstrip("-").isdigit():
            map_suffix = ":" + potential_map
            bracket_content = parts[0]

    # Split by top-level semicolons into fields
    fields = split_top_level(bracket_content, ";")

    # Find the first field with a top-level comma
    field_to_expand_index = -1
    for i, field in enumerate(fields):
        if has_top_level_comma(field):
            field_to_expand_index = i
            break

    if field_to_expand_index == -1:
        return [input_string]

    # Split the field by comma to get alternatives
    alternatives = split_top_level(fields[field_to_expand_index], ",")

    results = []
    for alt in alternatives:
        # Reconstruct fields with this alternative
        new_fields = (
            fields[:field_to_expand_index] + [alt] + fields[field_to_expand_index + 1 :]
        )
        new_bracket_content = ";".join(new_fields)
        if map_suffix:
            new_bracket_content += map_suffix
        new_string = (
            input_string[:start] + "[" + new_bracket_content + "]" + input_string[end:]
        )
        results.append(new_string)

    return results


def expand_all_brackets(input_string: str) -> List[str]:
    """
    Recursively expands all brackets with commas until none remain.
    """
    results = [input_string]

    while True:
        new_results = []
        any_expanded = False

        for s in results:
            expanded = expand_first_bracketed_list(s)
            if expanded != [s]:
                any_expanded = True
            new_results.extend(expanded)

        results = new_results

        if not any_expanded:
            break

    return results


def verify_validity_of_template(template: str) -> bool:
    """
    Verify the validity of a template by checking for:

    - Duplicate atom mapping in reactants and products
    - Mapped reactant atoms not present in products
    - Mapped product atoms not present in reactants
    - Atomic transmutation in the template

    Args:
        template (str): The template to verify, in the format "reactant_smarts>>product_smarts"

    Returns:
        bool: True if the template is valid, False otherwise
    """
    reactant_smarts = template.split(">>")[0]
    product_smarts = template.split(">>")[1]
    reactant_mols = [
        Chem.MolFromSmarts(smarts) for smarts in reactant_smarts.split(".")
    ]
    product_mols = [Chem.MolFromSmarts(smarts) for smarts in product_smarts.split(".")]

    product_atom_maps_and_elements = {}
    for mol in product_mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                continue
            if atom.GetAtomMapNum() in product_atom_maps_and_elements:
                logger.warning(f"Duplicate atom mapping in product: {template}")
            product_atom_maps_and_elements[atom.GetAtomMapNum()] = atom.GetSymbol()

    seen_product_atoms = list(product_atom_maps_and_elements.keys())
    reactant_atom_maps_and_elements = {}
    for mol in reactant_mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                continue
            if atom.GetAtomMapNum() in reactant_atom_maps_and_elements:
                logger.warning(f"Duplicate atom mapping in reactant: {template}")
            if atom.GetAtomMapNum() not in product_atom_maps_and_elements:
                logger.warning(
                    f"Mapped reactant atom(s) not present in product: {template}"
                )
                return False
            if product_atom_maps_and_elements[atom.GetAtomMapNum()] != atom.GetSymbol():
                logger.warning(f"Atomic transmutation in template: {template}")
                return False
            reactant_atom_maps_and_elements[atom.GetAtomMapNum()] = atom.GetSymbol()
            seen_product_atoms.remove(atom.GetAtomMapNum())

    if len(seen_product_atoms) != 0:
        logger.warning(f"Mapped product atom(s) not present in reactant: {template}")
        return False

    return True


def initialize_template_data(named_reactions: List[Dict[str, str]]) -> List:
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
    all_smirks: Dict[str, List[str]] = {}
    for reaction in named_reactions:
        smirks_list = []
        smirks = (
            reaction["smirks"].split(">>")[1] + ">>" + reaction["smirks"].split(">>")[0]
        )

        if len(expand_all_brackets(smirks)) < 100:
            smirks_list.extend(expand_all_brackets(smirks))

        if smirks in all_smirks:
            existing_patterns = all_smirks[smirks]
            existing_patterns.extend(smirks_list)
            all_smirks[smirks] = sorted(list(set(existing_patterns)))
        else:
            all_smirks[smirks] = sorted(list(set(smirks_list)))

    rdc_info = []
    for original_smirk, expanded_smirk_list in all_smirks.items():
        for smirk in expanded_smirk_list:
            products_smarts = [
                Chem.MolFromSmarts(smarts) for smarts in smirk.split(">>")[0].split(".")
            ]

            if None in products_smarts:
                continue

            reactants_smarts = [
                Chem.MolFromSmarts(smarts) for smarts in smirk.split(">>")[1].split(".")
            ]

            if None in reactants_smarts:
                continue

            try:
                rdc_rxn = rdc.rdchiralReaction(smirk)
            except Exception as e:
                logger.warning(f"Error converting smirks to rdchiral reaction: {e}")
                continue

            if not verify_validity_of_template(smirk):
                continue

            rdc_info.append(
                [products_smarts, reactants_smarts, rdc_rxn, original_smirk, smirk]
            )

    return rdc_info
