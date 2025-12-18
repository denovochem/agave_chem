from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

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
            if x != "":
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

        canonical_rxn = [".".join(role_list) for role_list in reaction_list]
        canonical_rxn = ">>".join(canonical_rxn)
        return canonical_rxn
    except Exception as e:
        logger.warning(f"Could not canonicalize {rxn_smiles}: {e}")
        return rxn_smiles


def canonicalize_atom_mapping(reaction_smiles: str) -> str:
    """ """

    reactant_mols = []
    for reactant in reaction_smiles.split(">>")[0].split("."):
        reactant_mols.append(Chem.MolFromSmiles(reactant))
    product_mols = []
    for product in reaction_smiles.split(">>")[1].split("."):
        product_mols.append(Chem.MolFromSmiles(product))

    atom_map_dict = {}
    next_map_num = 1

    for product_smiles in reaction_smiles.split(">>")[1].split("."):
        mol = Chem.MolFromSmiles(product_smiles)
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0 and map_num not in atom_map_dict:
                atom_map_dict[map_num] = next_map_num
                next_map_num += 1

    for reactant_smiles in reaction_smiles.split(">>")[0].split("."):
        mol = Chem.MolFromSmiles(reactant_smiles)
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0 and map_num not in atom_map_dict:
                atom_map_dict[map_num] = next_map_num
                next_map_num += 1

    for product_mol in product_mols:
        product_mol_copy = Chem.MolFromSmiles(Chem.MolToSmiles(product_mol))
        exact_match = False
        product_smiles = Chem.MolToSmiles(product_mol_copy)
        [atom.SetAtomMapNum(0) for atom in product_mol_copy.GetAtoms()]
        product_smiles_no_mapping = Chem.MolToSmiles(product_mol_copy)
        for i, reactant_mol in enumerate(reactant_mols):
            if not exact_match:
                reactant_smiles = Chem.MolToSmiles(reactant_mol)
                if reactant_smiles == product_smiles_no_mapping:
                    reactant_mols[i] = Chem.MolFromSmiles(product_smiles)
                    exact_match = True

    reactant_atom_symbol_freq_dict = {}
    for reactant_mol in reactant_mols:
        seen_canonical_ranks = []
        for reactant_atom, canonical_ranking in zip(
            reactant_mol.GetAtoms(),
            Chem.CanonicalRankAtoms(reactant_mol, breakTies=False),
        ):
            if canonical_ranking not in seen_canonical_ranks:
                if reactant_atom.GetSymbol() not in reactant_atom_symbol_freq_dict:
                    reactant_atom_symbol_freq_dict[reactant_atom.GetSymbol()] = 1
                    seen_canonical_ranks.append(canonical_ranking)
                else:
                    freq = reactant_atom_symbol_freq_dict[reactant_atom.GetSymbol()] + 1
                    reactant_atom_symbol_freq_dict[reactant_atom.GetSymbol()] = freq
                    seen_canonical_ranks.append(canonical_ranking)

    reactant_atom_single_occurance_dict = {
        k: v for k, v in reactant_atom_symbol_freq_dict.items() if v == 1
    }

    mapped_product_idx = []
    for product_mol in product_mols:
        for product_atom in product_mol.GetAtoms():
            for reactant_mol in reactant_mols:
                for reactant_atom in reactant_mol.GetAtoms():
                    if product_atom.GetSymbol() == reactant_atom.GetSymbol():
                        if (
                            product_atom.GetSymbol()
                            in reactant_atom_single_occurance_dict
                        ):
                            if (
                                reactant_atom.GetAtomMapNum() == 0
                                or reactant_atom.GetAtomMapNum() >= 900
                            ):
                                if (
                                    product_atom.GetAtomMapNum() != 0
                                    and product_atom.GetAtomMapNum()
                                    not in mapped_product_idx
                                ):
                                    reactant_atom.SetAtomMapNum(
                                        product_atom.GetAtomMapNum()
                                    )
                                    mapped_product_idx.append(
                                        product_atom.GetAtomMapNum()
                                    )

    product_atoms = {}
    mapped_product_atoms = []
    for product_mol in product_mols:
        for atom in product_mol.GetAtoms():
            product_atoms[atom.GetAtomMapNum()] = atom.GetSymbol()
            if atom.GetAtomMapNum() != 0:
                mapped_product_atoms.append(atom.GetAtomMapNum())
            else:
                print("Error mapping: Unmapped product atoms")
                return ""

    reactant_atoms = []
    mapped_reactant_atoms = []
    for reactant_mol in reactant_mols:
        for atom in reactant_mol.GetAtoms():
            reactant_atoms.append(atom.GetAtomMapNum())
            if atom.GetAtomMapNum() != 0:
                mapped_reactant_atoms.append(atom.GetAtomMapNum())

    if len(mapped_product_atoms) != len(set(mapped_product_atoms)):
        print("Error mapping: Duplicate product atoms")
        return ""
    if len(mapped_reactant_atoms) != len(set(mapped_reactant_atoms)):
        print("Error mapping: Duplicate reactant atoms")
        return ""

    seen_reactant_atoms = []
    for reactant_mol in reactant_mols:
        for atom in reactant_mol.GetAtoms():
            if atom.GetAtomMapNum() not in product_atoms:
                atom.SetAtomMapNum(0)
            else:
                if atom.GetSymbol() != product_atoms[atom.GetAtomMapNum()]:
                    print("Error mapping: Atomic transmutation!")
                    return ""
                seen_reactant_atoms.append(atom.GetAtomMapNum())

    if set(seen_reactant_atoms) != set(product_atoms):
        print(
            "Error mapping: Mapped product atoms but not corresponding reactant atoms"
        )
        return ""

    reactants_smiles = sorted(
        [Chem.MolToSmiles(mol, canonical=True) for mol in reactant_mols if mol != ""]
    )
    products_smiles = sorted(
        [Chem.MolToSmiles(mol, canonical=True) for mol in product_mols if mol != ""]
    )

    canonicalized_rxn = ".".join(reactants_smiles) + ">>" + ".".join(products_smiles)
    return canonicalized_rxn
