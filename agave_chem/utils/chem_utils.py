"""
Utility functions for molecular operations.

This module provides helper functions for working with RDKit molecules,
including parsing, sanitization, and atom property access.
"""

import random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from agave_chem.utils.constants import BOND_ENERGIES
from agave_chem.utils.logging_config import logger

tautomer_enumerator = rdMolStandardize.TautomerEnumerator()


def canonicalize_smiles(
    smiles: str,
    isomeric: bool = True,
    remove_mapping: bool = True,
    canonicalize_tautomer: bool = True,
) -> str:
    """
    Converts SMILES strings to their canonical form using RDKit.

    Takes a SMILES string (potentially containing multiple fragments separated by periods),
    splits it into fragments, sorts them, and converts each to its canonical form. Handles
    atom mapping and isomeric SMILES options.

    Args:
        smiles (str): The input SMILES string to canonicalize
        isomeric (bool): Whether to retain isomeric information. Defaults to True
        canonicalize_tautomer (bool): Whether to use the canonical tautomer. Defaults to True
        remove_mapping (bool): Whether to remove atom mapping numbers. Defaults to True

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
        return smiles


def randomize_smiles(
    smiles: str,
    isomeric: bool = True,
    shuffle_order: bool = True,
    remove_mapping: bool = True,
) -> str:
    try:
        x = smiles.split(".")
        if shuffle_order:
            random.shuffle(x)
        frags = []
        for i in x:
            m = Chem.MolFromSmiles(i)
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
        return smiles


def canonicalize_reaction_smiles(
    rxn_smiles: str,
    isomeric: bool = True,
    remove_mapping: bool = True,
    canonicalize_tautomer: bool = False,
    return_canonicalized_atom_mapping: bool = False,
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
        return rxn_smiles


def randomize_reaction_smiles(
    smiles: str,
    isomeric: bool = True,
    shuffle_order: bool = True,
) -> str:
    try:
        split_roles = smiles.split(">>")
        if len(split_roles) != 2:
            raise ValueError(f"Invalid reaction SMILES: {smiles}")
        reactants_list = []
        products_list = []
        for reactant in split_roles[0].split("."):
            reactants_list.append(randomize_smiles(reactant, isomeric=isomeric))
        for product in split_roles[1].split("."):
            products_list.append(randomize_smiles(product, isomeric=isomeric))
        if shuffle_order:
            random.shuffle(reactants_list)
            random.shuffle(products_list)
        randomized_rxn = ">>".join([".".join(reactants_list), ".".join(products_list)])
        return randomized_rxn
    except Exception as e:
        logger.warning(f"Could not randomize {smiles}: {e}")
        return smiles


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
        ".".join([Chem.MolToSmiles(mol, canonical=True) for mol in reactant_mols])
        + ">>"
        + ".".join([Chem.MolToSmiles(mol, canonical=True) for mol in product_mols])
    )

    return canonicalized_rxn


def parse_reaction_smiles(
    reaction_smiles: str,
) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
    """
    Parse a reaction SMILES into reactants and products.

    Args:
        reaction_smiles: Reaction SMILES in format "reactants>>products"

    Returns:
        Tuple of (reactants, products) as lists of RDKit Mol objects

    Raises:
        ValueError: If the reaction SMILES cannot be parsed
    """
    parts = reaction_smiles.split(">>")

    if len(parts) != 2:
        raise ValueError(
            f"Invalid reaction SMILES format: expected 2 parts separated by '>>', got {len(parts)}"
        )

    reactant_smiles, product_smiles = parts

    def parse_molecules(smiles_str: str) -> List[Chem.Mol]:
        """Parse dot-separated SMILES into list of molecules."""
        if not smiles_str.strip():
            return []

        molecules = []
        for smi in smiles_str.split("."):
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                logger.warning(f"Could not parse SMILES: {smi}")
                continue
            molecules.append(mol)
        return molecules

    reactants = parse_molecules(reactant_smiles)
    products = parse_molecules(product_smiles)

    if not reactants:
        raise ValueError("No valid reactants found in reaction SMILES")
    if not products:
        raise ValueError("No valid products found in reaction SMILES")

    return reactants, products


def sanitize_molecule(mol: Chem.Mol, add_hs: bool = False) -> Optional[Chem.Mol]:
    """
    Sanitize a molecule and optionally add hydrogens.

    Args:
        mol: RDKit molecule object
        add_hs: Whether to add explicit hydrogens

    Returns:
        Sanitized molecule or None if sanitization fails
    """
    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        if add_hs:
            mol_copy = Chem.AddHs(mol_copy)
        return mol_copy
    except Exception as e:
        logger.warning(f"Sanitization failed: {e}")
        return None


def get_atom_features(atom: Chem.Atom) -> Dict:
    """
    Extract features from an atom for comparison purposes.

    Args:
        atom: RDKit Atom object

    Returns:
        Dictionary of atom features
    """
    return {
        "atomic_num": atom.GetAtomicNum(),
        "symbol": atom.GetSymbol(),
        "formal_charge": atom.GetFormalCharge(),
        "num_hs": atom.GetTotalNumHs(),
        "hybridization": str(atom.GetHybridization()),
        "is_aromatic": atom.GetIsAromatic(),
        "is_in_ring": atom.IsInRing(),
        "degree": atom.GetDegree(),
        "isotope": atom.GetIsotope(),
        "chiral_tag": str(atom.GetChiralTag()),
    }


def get_bond_energy(atom1_symbol: str, atom2_symbol: str, bond_order: float) -> float:
    """
    Get the bond dissociation energy for a bond.

    Args:
        atom1_symbol: Symbol of first atom
        atom2_symbol: Symbol of second atom
        bond_order: Bond order (1.0, 1.5, 2.0, 3.0)

    Returns:
        Estimated bond energy in kcal/mol
    """
    # Normalize the order of atoms for lookup
    key1 = (atom1_symbol, atom2_symbol, bond_order)
    key2 = (atom2_symbol, atom1_symbol, bond_order)

    if key1 in BOND_ENERGIES:
        return BOND_ENERGIES[key1]
    elif key2 in BOND_ENERGIES:
        return BOND_ENERGIES[key2]
    else:
        # Default estimate based on average single bond energy
        logger.debug(
            f"No bond energy data for {atom1_symbol}-{atom2_symbol} (order {bond_order}), using default"
        )
        return 80.0 * bond_order


def remove_atom_mapping(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove atom mapping numbers from a molecule.

    Args:
        mol: RDKit molecule with atom mapping

    Returns:
        New molecule without atom mapping numbers
    """
    mol_copy = Chem.Mol(mol)
    for atom in mol_copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol_copy


def apply_atom_mapping(mol: Chem.Mol, mapping: Dict[int, int]) -> Chem.Mol:
    """
    Apply atom mapping numbers to a molecule.

    Args:
        mol: RDKit molecule
        mapping: Dictionary mapping atom indices to map numbers

    Returns:
        New molecule with atom mapping numbers
    """
    mol_copy = Chem.Mol(mol)
    for atom_idx, map_num in mapping.items():
        if atom_idx < mol_copy.GetNumAtoms():
            mol_copy.GetAtomWithIdx(atom_idx).SetAtomMapNum(map_num)
    return mol_copy


def get_bond_dict(mol: Chem.Mol) -> Dict[FrozenSet[int], Tuple[float, bool]]:
    """
    Get dictionary of bonds in a molecule.

    Args:
        mol: RDKit molecule

    Returns:
        Dictionary mapping frozenset of atom indices to (bond_order, is_aromatic)
    """
    bonds = {}
    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        bond_order = bond.GetBondTypeAsDouble()
        is_aromatic = bond.GetIsAromatic()
        bonds[frozenset([atom1_idx, atom2_idx])] = (bond_order, is_aromatic)
    return bonds


def get_ring_info(mol: Chem.Mol) -> List[Set[int]]:
    """
    Get information about rings in a molecule.

    Args:
        mol: RDKit molecule

    Returns:
        List of sets, each containing atom indices in a ring
    """
    ring_info = mol.GetRingInfo()
    return [set(ring) for ring in ring_info.AtomRings()]


def validate_rxn_mapping(rxn_smiles: str) -> bool:
    reactant_mols = []
    for reactant_smarts in rxn_smiles.split(">>")[0].split("."):
        reactant_mols.append(Chem.MolFromSmiles(reactant_smarts))
    product_mols = []
    for product_smarts in rxn_smiles.split(">>")[1].split("."):
        product_mols.append(Chem.MolFromSmiles(product_smarts))

    num_product_atoms = sum([mol.GetNumAtoms() for mol in product_mols])
    num_reactant_atoms = sum([mol.GetNumAtoms() for mol in reactant_mols])
    if num_product_atoms > num_reactant_atoms:
        print("Incorrect number of atoms")
        return

    num_atoms_of_each_type_product = {}
    for product_mol in product_mols:
        for atom in product_mol.GetAtoms():
            if atom.GetAtomicNum() not in num_atoms_of_each_type_product:
                num_atoms_of_each_type_product[atom.GetAtomicNum()] = 1
            else:
                num_atoms_of_each_type_product[atom.GetAtomicNum()] += 1

    num_atoms_of_each_type_reactant = {}
    for reactant_mol in reactant_mols:
        for atom in reactant_mol.GetAtoms():
            if atom.GetAtomicNum() not in num_atoms_of_each_type_reactant:
                num_atoms_of_each_type_reactant[atom.GetAtomicNum()] = 1
            else:
                num_atoms_of_each_type_reactant[atom.GetAtomicNum()] += 1

    for k, v in num_atoms_of_each_type_product.items():
        if num_atoms_of_each_type_reactant[k] < v:
            print(f"More atoms of atomic num {k} in products than reactants")
            return

    product_mol_atoms = {}
    for product_mol in product_mols:
        for atom in product_mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                raise ValueError("Unmapped product atom")
            product_mol_atoms[atom.GetAtomMapNum()] = atom

    reactant_atom_map_nums = []
    for reactant_mol in reactant_mols:
        for atom in reactant_mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                continue
            if atom.GetAtomMapNum() not in product_mol_atoms:
                raise ValueError(
                    f"Mapped reactant atom {atom.GetAtomMapNum()} not found in products"
                )
            if (
                atom.GetAtomicNum()
                != product_mol_atoms[atom.GetAtomMapNum()].GetAtomicNum()
            ):
                raise ValueError(
                    f"Mapped reactant atom {atom.GetAtomMapNum()} has different atomic number"
                )
            reactant_atom_map_nums.append(atom.GetAtomMapNum())

    if set(reactant_atom_map_nums) != set(product_mol_atoms.keys()):
        raise ValueError("Incorrect atom mapping nums")

    return True


def sanitize_rxn_string(
    rxn_smiles: str, canonicalize: bool = True, remove_duplicate_fragments: bool = False
) -> str:
    """
    Sanitize the input reaction SMILES string by parsing it into reactants and products
    and checking that the constituent molecules are standardized.

    Standardization:
    1. Ensuring each fragment can be rounded-tripped through RDKit
    2. Removing mapping numbers
    3. Remove duplicate fragments
    4. Make sure ">>" is in the string, only once
    5. Removing isotopes
    6. Canonicalizing SMILES strings
    7. Isomerizing SMILES strings

    Args:
        rxn_smiles (str): Reaction SMILES string

    Returns:
        str: Sanitized reaction SMILES string
    """
    if ">>" not in rxn_smiles:
        raise ValueError("Invalid reaction SMILES string")

    reactants_str = rxn_smiles.split(">>")[0]
    products_str = rxn_smiles.split(">>")[1]

    if remove_duplicate_fragments:
        reactants_strs = list(set(reactants_str.split(".")))
        products_strs = list(set(products_str.split(".")))
    else:
        reactants_strs = reactants_str.split(".")
        products_strs = products_str.split(".")

    reactants_mols = [Chem.MolFromSmiles(reactant) for reactant in reactants_strs]
    products_mols = [Chem.MolFromSmiles(product) for product in products_strs]

    if None in reactants_mols or None in products_mols:
        raise ValueError("Invalid SMILES in reaction SMILES string")

    standardized_reactants_str = "".join(
        [
            Chem.MolToSmiles(reactant, canonical=canonicalize, isomericSmiles=True)
            for reactant in reactants_mols
        ]
    )
    standardized_products_str = "".join(
        [
            Chem.MolToSmiles(product, canonical=canonicalize, isomericSmiles=True)
            for product in products_mols
        ]
    )
    standardized_rxn_smiles = (
        standardized_reactants_str + ">>" + standardized_products_str
    )

    return standardized_rxn_smiles
