"""
Utility functions for molecular operations.

This module provides helper functions for working with RDKit molecules,
including parsing, sanitization, and atom property access.
"""

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


def get_atom_map_to_canonical_idx(mapped_smiles: str) -> Dict[int, int]:
    """
    Given an atom-mapped SMILES, returns the canonical unmapped SMILES and a mapping
    from original atom map numbers to atom indices in the canonical SMILES.

    Args:
        mapped_smiles: SMILES string with atom map numbers (e.g., "[CH3:1][OH:2]")

    Returns:
        dict: Dictionary mapping original atom map numbers to canonical atom indices
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

    reactant_mapping_dict = {}
    mapping_offset = 0
    for product_mol, product_mol_mapping_dict in zip(
        product_mols, product_mol_mapping_dicts
    ):
        num_atoms_mapped = 0
        for atom in product_mol.GetAtoms():
            if atom.GetAtomMapNum() in product_mol_mapping_dict:
                reactant_mapping_dict[atom.GetAtomMapNum()] = (
                    product_mol_mapping_dict[atom.GetIdx() + 1] + mapping_offset
                )
                num_atoms_mapped += 1
                atom.SetAtomMapNum(
                    mapping_offset + product_mol_mapping_dict[atom.GetIdx() + 1]
                )
        mapping_offset += num_atoms_mapped

    for reactant_mol in reactant_mols:
        for atom in reactant_mol.GetAtoms():
            if atom.GetAtomMapNum() in reactant_mapping_dict:
                atom.SetAtomMapNum(reactant_mapping_dict[atom.GetAtomMapNum()])
            else:
                if atom.GetAtomMapNum() > 0:
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
