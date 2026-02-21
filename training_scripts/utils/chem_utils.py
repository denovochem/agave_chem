import random

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

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
    except Exception:
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
    except Exception:
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
        return canonical_rxn
    except Exception:
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
    except Exception:
        return smiles
