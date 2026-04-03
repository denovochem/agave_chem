from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from rdkit import Chem

from agave_chem.mappers.types import ReactionMapperResult
from agave_chem.utils.chem_utils import canonicalize_atom_mapping
from agave_chem.utils.logging_config import logger


class ReactionMapper(ABC):
    """
    Abstract base class for mapping chemical reactions.

    Subclasses must implement the `map_reactions` method.
    """

    def __init__(self, mapper_type: str, mapper_name: str, mapper_weight: float):
        if not isinstance(mapper_type, str):
            raise TypeError("Invalid input: mapper_type must be a string.")
        self._mapper_type: str = mapper_type
        if not isinstance(mapper_name, str):
            raise TypeError("Invalid input: mapper_name must be a string.")
        self._mapper_name: str = mapper_name
        if not isinstance(mapper_weight, (int, float)):
            raise TypeError(
                "Invalid input: mapper_weight must be a number between 0-1000."
            )
        if mapper_weight < 0 or mapper_weight > 1000:
            raise ValueError(
                "Invalid input: mapper_weight must be a number between 0-1000."
            )
        self._mapper_weight: float = float(mapper_weight)

        self._default_mapping_dict = ReactionMapperResult(
            original_smiles="",
            selected_mapping="",
            possible_mappings={},
            mapping_type=self._mapper_type,
            mapping_score=None,
            additional_info=[{}],
        )

    @property
    def mapper_name(self) -> str:
        """Return mapper_name."""
        return self._mapper_name

    @property
    def mapper_weight(self) -> float:
        """Return mapper_weight."""
        return self._mapper_weight

    def _return_default_mapping_dict(
        self, original_smiles: str
    ) -> ReactionMapperResult:
        """Return a default mapping dictionary."""
        default_mapping_dict = self._default_mapping_dict
        default_mapping_dict["original_smiles"] = original_smiles

        return default_mapping_dict

    def _split_reaction_components(self, reaction_smiles: str) -> Tuple[str, str]:
        """
        Splits a reaction SMILES string into reactants and products.

        Args:
            reaction_smiles (str): A reaction SMILES string in the format "reactants>>products"

        Returns:
            tuple: A tuple containing the reactants and products as strings
        """
        parts = reaction_smiles.strip().split(">>")
        reactants = parts[0]
        products = parts[1]
        return reactants, products

    def _reaction_smiles_valid(self, reaction_smiles: str) -> bool:
        """
        Checks if the reaction SMILES string is valid.

        Args:
            reaction_smiles (str): The reaction SMILES string to check

        Returns:
            bool: True if the reaction SMILES string is valid, False otherwise
        """
        if reaction_smiles.count(">>") != 1:
            return False
        for ele in reaction_smiles.split(">>"):
            if len(ele) <= 0:
                return False
        reactant_strs, product_strs = self._split_reaction_components(reaction_smiles)
        if not reactant_strs or not product_strs:
            return False
        reactant_mols = [
            Chem.MolFromSmiles(reactant_str)
            for reactant_str in reactant_strs.split(".")
        ]
        if None in reactant_mols:
            return False
        product_mols = [
            Chem.MolFromSmiles(product_str) for product_str in product_strs.split(".")
        ]
        if None in product_mols:
            return False
        return True

    def _remove_duplicate_fragments(self, smiles: str) -> str:
        """
        Removes duplicate fragments from a SMILES string.

        Args:
            smiles (str): A SMILES string

        Returns:
            str: A SMILES string with duplicate fragments removed
        """
        reactants_strs, products_strs = self._split_reaction_components(smiles)
        reactants_strs_list = reactants_strs.split(".")
        products_strs_list = products_strs.split(".")
        unique_reactants_strs_list = list(set(reactants_strs_list))
        unique_products_strs_list = list(set(products_strs_list))
        return (
            ".".join(unique_reactants_strs_list)
            + ">>"
            + ".".join(unique_products_strs_list)
        )

    def _remove_existing_mapping(self, rxn_smiles: str) -> str:
        """
        Removes existing atom mapping from a SMILES string.

        Args:
            rxn_smiles (str): A reaction SMILES string

        Returns:
            str: A SMILES string with existing atom mapping removed
        """
        reactants_strs, products_strs = self._split_reaction_components(rxn_smiles)
        reactant_mols = [
            Chem.MolFromSmiles(reactant_str)
            for reactant_str in reactants_strs.split(".")
        ]
        for reactant_mol in reactant_mols:
            if reactant_mol is not None:
                for atom in reactant_mol.GetAtoms():
                    atom.SetAtomMapNum(0)
        product_mols = [
            Chem.MolFromSmiles(product_str) for product_str in products_strs.split(".")
        ]
        for product_mol in product_mols:
            if product_mol is not None:
                for atom in product_mol.GetAtoms():
                    atom.SetAtomMapNum(0)
        reactants_strs = ".".join(
            [
                Chem.MolToSmiles(reactant)
                for reactant in reactant_mols
                if reactant is not None
            ]
        )
        products_strs = ".".join(
            [
                Chem.MolToSmiles(product)
                for product in product_mols
                if product is not None
            ]
        )
        return reactants_strs + ">>" + products_strs

    @abstractmethod
    def map_reaction(self, reaction_smiles: str) -> ReactionMapperResult:
        pass

    @abstractmethod
    def map_reactions(
        self, reaction_smiles_list: List[str]
    ) -> List[ReactionMapperResult]:
        pass

    def _verify_validity_of_mapping(
        self, reaction_smiles: str, expect_full_mapping: bool = True
    ) -> bool:
        """
        Verifies the validity of a mapped reaction SMILES string.

        Args:
            reaction_smiles (str): A mapped reaction SMILES string
            expect_full_mapping (bool): Whether to expect a full mapping

        Returns:
            bool: True if the mapping is valid, False otherwise
        """
        reactants_smiles = reaction_smiles.split(">>")[0]
        products_smiles = reaction_smiles.split(">>")[1]
        reactants_mols = [
            Chem.MolFromSmiles(reactant) for reactant in reactants_smiles.split(".")
        ]
        if None in reactants_mols:
            return False
        products_mols = [
            Chem.MolFromSmiles(product) for product in products_smiles.split(".")
        ]
        if None in products_mols:
            return False

        product_atom_maps_and_elements = {}
        for mol in products_mols:
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() == 0:
                    if expect_full_mapping:
                        logger.warning("Unmapped atom in product")
                        return False
                    else:
                        continue
                if atom.GetAtomMapNum() in product_atom_maps_and_elements:
                    logger.warning("Duplicate atom mapping in product")
                    return False
                product_atom_maps_and_elements[atom.GetAtomMapNum()] = atom.GetSymbol()

        seen_product_atoms = list(product_atom_maps_and_elements.keys())
        reactant_atom_maps_and_elements = {}
        for mol in reactants_mols:
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() == 0:
                    continue
                if atom.GetAtomMapNum() in reactant_atom_maps_and_elements:
                    logger.warning("Duplicate atom mapping in reactant")
                    return False
                if atom.GetAtomMapNum() not in product_atom_maps_and_elements:
                    logger.warning("Mapped reactant atom(s) not present in product")
                    return False
                if (
                    product_atom_maps_and_elements[atom.GetAtomMapNum()]
                    != atom.GetSymbol()
                ):
                    logger.warning("Atomic transmutation in reaction")
                    return False
                reactant_atom_maps_and_elements[atom.GetAtomMapNum()] = atom.GetSymbol()
                if atom.GetAtomMapNum() in seen_product_atoms:
                    seen_product_atoms.remove(atom.GetAtomMapNum())
                else:
                    logger.warning("Mapped reactant atom(s) not present in product")
                    return False

        if len(seen_product_atoms) != 0:
            logger.warning("Mapped product atom(s) not present in reactants")
            return False

        return True

    def sanitize_molecule(
        self, mol: Chem.Mol, add_hs: bool = False
    ) -> Optional[Chem.Mol]:
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

    def sanitize_rxn_string(
        self,
        reaction_smiles: str,
        expect_full_mapping: bool = True,
        canonicalize: bool = True,
        remove_mapping: bool = False,
    ) -> str:
        """
        Sanitize a reaction SMILES string.

        Args:
            reaction_smiles: Reaction SMILES string
            expect_full_mapping: Whether to expect full atom mapping
            canonicalize: Whether to canonicalize the reaction SMILES
            remove_mapping: Whether to remove existing atom mapping

        Returns:
            Sanitized reaction SMILES string
        """

        if not self._verify_validity_of_mapping(reaction_smiles, expect_full_mapping):
            raise ValueError("Reaction SMILES string is not valid")

        if remove_mapping:
            reaction_smiles = self._remove_existing_mapping(reaction_smiles)

        if canonicalize:
            reaction_smiles = canonicalize_atom_mapping(reaction_smiles)

        return reaction_smiles
