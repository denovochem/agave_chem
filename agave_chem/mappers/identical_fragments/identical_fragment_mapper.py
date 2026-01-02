from rdkit import Chem
from typing import List, Dict
from agave_chem.utils.logging_config import logger
from agave_chem.utils.chem_utils import canonicalize_smiles


def atom_map_identical_fragments(reaction_smiles: str):
    """
    Atom map identical fragments in a reaction SMILES string.

    Args:
        reaction_smiles (str): A reaction SMILES string.

    Returns:
        tuple: A tuple containing a list of mapped identical fragments and a mapped reaction SMILES string.
    """
    reactants = reaction_smiles.split(">>")[0]
    products = reaction_smiles.split(">>")[1]

    reactants_smiles_list = reactants.split(".")
    products_smiles_list = products.split(".")

    reactants_smiles_list_mapping_dict = {
        canonicalize_smiles(reactant): reactant for reactant in reactants_smiles_list
    }

    canonicalized_reactants_smiles_list = [
        canonicalize_smiles(smiles) for smiles in reactants_smiles_list
    ]
    canonicalized_products_smiles_list = [
        canonicalize_smiles(smiles) for smiles in products_smiles_list
    ]

    atom_mapped_identical_reactants_products = []
    atom_map_num = 500
    for canonicalized_reactant in canonicalized_reactants_smiles_list:
        reactant = reactants_smiles_list_mapping_dict[canonicalized_reactant]
        if canonicalized_reactant in canonicalized_products_smiles_list:
            reactants_smiles_list.remove(reactant)
            products_smiles_list.remove(reactant)
            reactant_mol = Chem.MolFromSmiles(canonicalized_reactant)
            for atom in reactant_mol.GetAtoms():
                atom.SetAtomMapNum(atom_map_num)
                atom_map_num += 1
            mapped_reactant = Chem.MolToSmiles(reactant_mol)
            atom_mapped_identical_reactants_products.append(mapped_reactant)
    return (
        atom_mapped_identical_reactants_products,
        ".".join(reactants_smiles_list) + ">>" + ".".join(products_smiles_list),
    )


def add_identical_fragments_to_mapping(
    mapped_reaction_smiles: str, atom_mapped_identical_reactants_products: List[str]
):
    """
    Add identical fragments to a mapping.

    Args:
        mapped_reaction_smiles (str): A mapped reaction SMILES string.
        atom_mapped_identical_reactants_products (List[str]): A list of identical fragments mapping lists.

    Returns:
        str: A mapped reaction SMILES string with identical fragments added.
    """
    reactants = mapped_reaction_smiles.split(">>")[0]
    products = mapped_reaction_smiles.split(">>")[1]

    reactants_smiles_list = reactants.split(".")
    products_smiles_list = products.split(".")

    for identical_fragment in atom_mapped_identical_reactants_products:
        reactants_smiles_list.append(identical_fragment)
        products_smiles_list.append(identical_fragment)

    mapped_reactants = ".".join(reactants_smiles_list)
    mapped_products = ".".join(products_smiles_list)

    return mapped_reactants + ">>" + mapped_products


def create_identical_fragments_mapping_list(reaction_smiles_list: List[str]):
    """
    Create a list of mapped reaction SMILES strings and a list of identical fragments mapping lists.

    Args:
        reaction_smiles_list (List[str]): A list of reaction SMILES strings.

    Returns:
        List[str]: A list of mapped reaction SMILES strings.
        List[Dict[str, str]]: A list of identical fragments mapping lists.
    """
    new_rxns = []
    identical_fragments_mapping_list = []
    for reaction_smiles in reaction_smiles_list:
        atom_mapped_identical_fragments, new_rxn = atom_map_identical_fragments(
            reaction_smiles
        )
        identical_fragments_mapping_list.append(atom_mapped_identical_fragments)
        new_rxns.append(new_rxn)
    return new_rxns, identical_fragments_mapping_list


def resolve_identical_fragments_mapping_dict(
    mapped_reaction_smiles_list: List[str],
    identical_fragments_mapping_list: List[Dict[str, str]],
):
    """
    Resolve a list of mapped reaction SMILES strings and a list of identical fragments mapping lists into a list of final reaction SMILES strings.

    Args:
        mapped_reaction_smiles_list (List[str]): A list of mapped reaction SMILES strings.
        identical_fragments_mapping_list (List[Dict[str, str]]): A list of identical fragments mapping lists.

    Returns:
        List[str]: A list of final reaction SMILES strings.
    """
    final_reactions = []
    for mapped_reaction_smiles, identical_fragments_mapping in zip(
        mapped_reaction_smiles_list, identical_fragments_mapping_list
    ):
        final_reactions.append(
            add_identical_fragments_to_mapping(
                mapped_reaction_smiles, identical_fragments_mapping
            )
        )
    return final_reactions
