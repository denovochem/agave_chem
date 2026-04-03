"""
Utility functions for molecular operations.

This module provides helper functions for working with RDKit molecules,
including parsing, sanitization, and atom property access.
"""

import random
from typing import Dict

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from agave_chem.utils.logging_config import logger

tautomer_enumerator = rdMolStandardize.TautomerEnumerator()


def canonicalize_smiles(
    smiles: str,
    isomeric: bool = True,
    remove_mapping: bool = True,
    canonicalize_tautomer: bool = True,
    throw_error_on_failure: bool = False,
) -> str:
    """
    Converts SMILES strings to their canonical form using RDKit.

    Takes a SMILES string (potentially containing multiple fragments separated by periods),
    splits it into fragments, sorts them, and converts each to its canonical form. Handles
    atom mapping and isomeric SMILES options.

    Args:
        smiles (str): The input SMILES string to canonicalize
        isomeric (bool): Whether to retain isomeric information. Defaults to True
        remove_mapping (bool): Whether to remove atom mapping numbers. Defaults to True
        canonicalize_tautomer (bool): Whether to use the canonical tautomer. Defaults to True
        throw_error_on_failure (bool): Whether to throw an error if canonicalization fails. Defaults to False

    Returns:
        str: The canonicalized SMILES string. If conversion fails, returns the input string
            unchanged.
    """
    try:
        x = smiles.split(".")
        x = sorted(x)
        frags = []
        for i in x:
            m = Chem.MolFromSmiles(i)
            if canonicalize_tautomer:
                m = tautomer_enumerator.Canonicalize(m)
            if remove_mapping:
                [a.SetAtomMapNum(0) for a in m.GetAtoms()]
            canonical_smiles_string = str(
                Chem.MolToSmiles(m, canonical=True, isomericSmiles=isomeric)
            )
            frags.append(canonical_smiles_string)
        canonical_smiles_string = ".".join(i for i in sorted(frags))
        return canonical_smiles_string
    except Exception as e:
        logger.warning(f"Could not canonicalize {smiles}: {e}")
        if throw_error_on_failure:
            raise e
        return smiles


def randomize_smiles(
    smiles: str,
    isomeric: bool = True,
    shuffle_order: bool = True,
    remove_mapping: bool = True,
    randomize_tautomer: bool = False,
    throw_error_on_failure: bool = False,
) -> str:
    try:
        x = smiles.split(".")
        if shuffle_order:
            random.shuffle(x)
        frags = []
        for i in x:
            m = Chem.MolFromSmiles(i)
            if randomize_tautomer:
                tautomers = tautomer_enumerator.Enumerate(m)
                m = random.choice(tautomers)
            if remove_mapping:
                [a.SetAtomMapNum(0) for a in m.GetAtoms()]
            new_atom_order = list(range(m.GetNumAtoms()))
            random.shuffle(new_atom_order)
            random_mol = Chem.RenumberAtoms(m, newOrder=new_atom_order)
            random_smiles_string = str(
                Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=isomeric)
            )
            frags.append(random_smiles_string)
        random_smiles_string = ".".join(i for i in frags)
        return random_smiles_string
    except Exception as e:
        logger.warning(f"Could not randomize {smiles}: {e}")
        if throw_error_on_failure:
            raise e
        return smiles


def canonicalize_reaction_smiles(
    rxn_smiles: str,
    isomeric: bool = True,
    remove_mapping: bool = True,
    canonicalize_tautomer: bool = False,
    return_canonicalized_atom_mapping: bool = False,
    throw_error_on_failure: bool = False,
) -> str:
    """
    Canonicalizes a reaction SMILES string using RDKit.

    Takes a reaction SMILES string (potentially containing multiple fragments separated by periods),
    splits it into fragments, sorts them, and converts each to its canonical form. Handles
    atom mapping and isomeric SMILES options.

    Args:
        rxn_smiles (str): The input reaction SMILES string to canonicalize
        isomeric (bool): Whether to retain isomeric information. Defaults to True
        remove_mapping (bool): Whether to remove atom mapping numbers. Defaults to True
        canonicalize_tautomer (bool): Whether to use the canonical tautomer. Defaults to False

    Returns:
        str: The canonicalized reaction SMILES string. If conversion fails, returns the input string
            unchanged.
    """
    try:
        split_roles = rxn_smiles.split(">>")
        reaction_list = []
        for x in split_roles:
            role_list = []
            if x == "":
                continue
            y = x.split(".")
            for z in y:
                canonical_smiles = canonicalize_smiles(
                    z,
                    isomeric=isomeric,
                    remove_mapping=remove_mapping,
                    canonicalize_tautomer=canonicalize_tautomer,
                )
                role_list.append(canonical_smiles)

            role_list = sorted(role_list)
            role_list = [ele for ele in role_list if ele != ""]
            reaction_list.append(role_list)

        canonical_rxn_components = [".".join(role_list) for role_list in reaction_list]
        canonical_rxn = ">>".join(canonical_rxn_components)
        if return_canonicalized_atom_mapping:
            canonical_rxn = canonicalize_atom_mapping(canonical_rxn)
        return canonical_rxn
    except Exception as e:
        logger.warning(f"Could not canonicalize {rxn_smiles}: {e}")
        if throw_error_on_failure:
            raise e
        return rxn_smiles


def randomize_reaction_smiles(
    rxn_smiles: str,
    isomeric: bool = True,
    remove_mapping: bool = True,
    shuffle_order: bool = True,
    randomize_tautomer: bool = False,
    randomize_atom_mapping: bool = False,
    throw_error_on_failure: bool = False,
) -> str:
    try:
        split_roles = rxn_smiles.split(">>")
        if len(split_roles) != 2:
            raise ValueError(f"Invalid reaction SMILES: {rxn_smiles}")
        reactants_list = []
        products_list = []
        for reactant in split_roles[0].split("."):
            reactants_list.append(
                randomize_smiles(
                    reactant,
                    isomeric=isomeric,
                    remove_mapping=remove_mapping,
                    randomize_tautomer=randomize_tautomer,
                )
            )
        for product in split_roles[1].split("."):
            products_list.append(
                randomize_smiles(
                    product,
                    isomeric=isomeric,
                    remove_mapping=remove_mapping,
                    randomize_tautomer=randomize_tautomer,
                )
            )
        if shuffle_order:
            random.shuffle(reactants_list)
            random.shuffle(products_list)
        randomized_rxn = ">>".join([".".join(reactants_list), ".".join(products_list)])
        return randomized_rxn
    except Exception as e:
        logger.warning(f"Could not randomize {rxn_smiles}: {e}")
        if throw_error_on_failure:
            raise e
        return rxn_smiles


def remove_reaction_smiles_atom_mapping(rxn_smiles: str) -> str:
    split_roles = rxn_smiles.split(">>")
    reactants = split_roles[0].split(".")
    products = split_roles[1].split(".")
    unmapped_reactants = []
    for reactant in reactants:
        reactant_mol = Chem.MolFromSmiles(reactant)
        [a.SetAtomMapNum(0) for a in reactant_mol.GetAtoms()]
        reactant = Chem.MolToSmiles(
            reactant_mol, canonical=False, doRandom=False, isomericSmiles=True
        )
        unmapped_reactants.append(reactant)
    unmapped_products = []
    for product in products:
        product_mol = Chem.MolFromSmiles(product)
        [a.SetAtomMapNum(0) for a in product_mol.GetAtoms()]
        product = Chem.MolToSmiles(
            product_mol, canonical=False, doRandom=False, isomericSmiles=True
        )
        unmapped_products.append(product)
    return ">>".join([".".join(unmapped_reactants), ".".join(unmapped_products)])


def get_atom_map_to_canonical_idx(mapped_smiles: str) -> Dict[int, int]:
    """
    Given an atom-mapped SMILES, returns a mapping from original atom map numbers
    to atom indices in the canonical SMILES (1-indexed).

    Args:
        mapped_smiles: SMILES string with atom map numbers (e.g., "[CH3:1][OH:2]")

    Returns:
        dict: Dictionary mapping original atom map numbers to canonical atom indices (1-indexed)
    """
    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {mapped_smiles}")

    orig_idx_to_map_num = {}
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num > 0:
            orig_idx_to_map_num[atom.GetIdx()] = map_num
        atom.SetAtomMapNum(0)

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

    mol_canon = Chem.MolFromSmiles(canonical_smiles)

    ranks_orig = Chem.CanonicalRankAtoms(mol, breakTies=True, includeChirality=True)
    ranks_canon = Chem.CanonicalRankAtoms(
        mol_canon, breakTies=True, includeChirality=True
    )

    rank_to_canon_idx = {rank: idx for idx, rank in enumerate(ranks_canon)}

    map_num_to_canon_idx = {}
    for orig_idx, map_num in orig_idx_to_map_num.items():
        orig_rank = ranks_orig[orig_idx]
        canon_idx = rank_to_canon_idx[orig_rank]
        map_num_to_canon_idx[map_num] = canon_idx + 1

    return map_num_to_canon_idx


def canonicalize_atom_mapping(reaction_smiles: str) -> str:
    """
    Canonicalizes a reaction SMILES string with respect to atom mapping.

    Takes a reaction SMILES string, splits it into reactant and product molecules,
    and then reassigns the atom map numbers in the reactant molecules to match the
    canonical atom order in the product molecules.

    Args:
        reaction_smiles (str): The input reaction SMILES string to canonicalize

    Returns:
        str: The canonicalized reaction SMILES string.
    """
    reactant_mols = []
    for reactant_smarts in reaction_smiles.split(">>")[0].split("."):
        reactant_mols.append(Chem.MolFromSmiles(reactant_smarts))

    product_mols = []
    product_mol_mapping_dicts = []
    for product_smarts in reaction_smiles.split(">>")[1].split("."):
        product_mols.append(Chem.MolFromSmiles(product_smarts))
        product_mol_mapping_dicts.append(get_atom_map_to_canonical_idx(product_smarts))

    # Build mapping: old_map_num -> new_map_num (based on canonical product order)
    old_to_new_map = {}
    new_map_num = 1

    for product_mol, product_mol_mapping_dict in zip(
        product_mols, product_mol_mapping_dicts
    ):
        # Collect mapped atoms with their canonical indices
        mapped_atoms = []
        for atom in product_mol.GetAtoms():
            old_map = atom.GetAtomMapNum()
            if old_map > 0 and old_map in product_mol_mapping_dict:
                canon_idx = product_mol_mapping_dict[old_map]
                mapped_atoms.append((canon_idx, atom, old_map))

        # Sort by canonical index to assign new map numbers in canonical order
        mapped_atoms.sort(key=lambda x: x[0])

        for canon_idx, atom, old_map in mapped_atoms:
            # Only assign new mapping if not already assigned (handles duplicate map nums)
            if old_map not in old_to_new_map:
                old_to_new_map[old_map] = new_map_num
                new_map_num += 1
            atom.SetAtomMapNum(old_to_new_map[old_map])

        # Unmapped atoms keep map num 0 (already the case, no action needed)

    # Update reactant mappings
    for reactant_mol in reactant_mols:
        for atom in reactant_mol.GetAtoms():
            old_map = atom.GetAtomMapNum()
            if old_map in old_to_new_map:
                atom.SetAtomMapNum(old_to_new_map[old_map])
            else:
                if old_map > 0:
                    logger.info(
                        f"Reactant atom {atom.GetIdx()} is mapped but has no corresponding product atom"
                    )
                atom.SetAtomMapNum(0)

    canonicalized_rxn = (
        ".".join(
            sorted([Chem.MolToSmiles(mol, canonical=True) for mol in reactant_mols])
        )
        + ">>"
        + ".".join(
            sorted([Chem.MolToSmiles(mol, canonical=True) for mol in product_mols])
        )
    )

    return canonicalized_rxn
