from abc import ABC, abstractmethod
from typing import List, Dict, Tuple


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

    @abstractmethod
    def map_reactions(
        self, reaction_list: List[str]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Map chemical reaction SMILES.

        Args:
            reaction_list: List of reaction SMILES.

        Returns:
            Tuple of:
                - XXX
                - XXX
        """
        pass
