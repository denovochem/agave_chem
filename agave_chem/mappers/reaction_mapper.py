from abc import ABC, abstractmethod
from typing import List, Tuple


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

    @property
    def mapper_name(self) -> str:
        """Return mapper_name."""
        return self._mapper_name

    @property
    def mapper_weight(self) -> float:
        """Return mapper_weight."""
        return self._mapper_weight

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
        return True

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

    @abstractmethod
    def map_reaction(self, reaction_smiles: str):
        pass

    @abstractmethod
    def map_reactions(self, reaction_smiles_list: List[str]):
        pass
