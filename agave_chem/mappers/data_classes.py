"""
Data classes for atom-to-atom mapping.

This module contains all the data structures used throughout the atom mapping
process, including atom mappings, bond changes, and scoring metrics.
"""

from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field


class MappingAlgorithm(Enum):
    """Enumeration of available mapping algorithms."""

    MIN = auto()  # Minimize the number of bond changes
    MAX = auto()  # Maximize the common substructure
    MIXTURE = auto()  # Hybrid approach combining MIN and MAX


class BondChangeType(Enum):
    """Types of bond changes in a reaction."""

    FORMED = auto()
    BROKEN = auto()
    ORDER_CHANGE = auto()


class AtomMapping(BaseModel):
    """
    Represents a mapping between a reactant atom and a product atom.

    Args:
        reactant_mol_idx (int): Index of the molecule in reactants list.
        reactant_atom_idx (int): Atom index within the reactant molecule.
        product_mol_idx (int): Index of the molecule in products list.
        product_atom_idx (int): Atom index within the product molecule.
    """

    model_config = ConfigDict(frozen=True)

    reactant_mol_idx: int
    reactant_atom_idx: int
    product_mol_idx: int
    product_atom_idx: int

    def __repr__(self) -> str:
        return (
            f"AtomMapping(R{self.reactant_mol_idx}:{self.reactant_atom_idx} -> "
            f"P{self.product_mol_idx}:{self.product_atom_idx})"
        )


class BondChange(BaseModel):
    """
    Represents a bond change during a reaction.

    Args:
        atom1_map (int): Atom map number of first atom.
        atom2_map (int): Atom map number of second atom.
        change_type (BondChangeType): Type of bond change.
        old_order (Optional[float]): Bond order before reaction (None if formed).
        new_order (Optional[float]): Bond order after reaction (None if broken).
        energy_cost (float): Estimated energy cost of this bond change.
    """

    model_config = ConfigDict(frozen=True)

    atom1_map: int
    atom2_map: int
    change_type: BondChangeType
    old_order: Optional[float] = None
    new_order: Optional[float] = None
    energy_cost: float = 0.0

    def __repr__(self) -> str:
        return (
            f"BondChange({self.atom1_map}-{self.atom2_map}, "
            f"{self.change_type.name}, {self.old_order}->{self.new_order})"
        )


class MappingScore(BaseModel):
    """
    Scoring metrics for evaluating a mapping solution.

    These metrics are based on the RDTool paper and are used to select
    the best mapping among multiple candidates.

    Args:
        bond_energy_cost (float): Total energy of bonds formed/broken.
        num_bond_changes (int): Number of bonds that change.
        num_bonds_formed (int): Number of new bonds formed.
        num_bonds_broken (int): Number of bonds broken.
        num_fragments (int): Number of molecular fragments affected.
        stereo_changes (int): Number of stereochemistry changes.
        similarity_score (float): Tanimoto similarity of mapped atoms.
        ring_changes (int): Number of ring opening/closing events.
    """

    bond_energy_cost: float = 0.0
    num_bond_changes: int = 0
    num_bonds_formed: int = 0
    num_bonds_broken: int = 0
    num_fragments: int = 0
    stereo_changes: int = 0
    similarity_score: float = 0.0
    ring_changes: int = 0

    def total_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate weighted total score (lower is better).

        Args:
            weights (Optional[Dict[str, float]]): Optional dictionary of metric weights.

        Returns:
            float: Weighted total score.
        """
        if weights is None:
            weights = {
                "bond_energy_cost": 1.0,
                "num_bond_changes": 10.0,
                "num_bonds_formed": 5.0,
                "num_bonds_broken": 5.0,
                "num_fragments": 20.0,
                "stereo_changes": 15.0,
                "similarity_score": -50.0,  # Negative because higher is better
                "ring_changes": 25.0,
            }

        score = 0.0
        for metric, weight in weights.items():
            score += weight * getattr(self, metric, 0)
        return score


class MappingResult(BaseModel):
    """
    Complete result of an atom-to-atom mapping.

    Args:
        atom_mappings (FrozenSet[AtomMapping]): Set of atom mappings between reactants and products.
        bond_changes (List[BondChange]): List of bond changes in the reaction.
        score (MappingScore): Scoring metrics for this mapping.
        algorithm_used (MappingAlgorithm): Which algorithm produced this mapping.
        mapped_smiles (str): SMILES with atom mapping numbers.
        reaction_center (Set[int]): Atom indices involved in the reaction center.
    """

    atom_mappings: FrozenSet[AtomMapping]
    bond_changes: List[BondChange]
    score: MappingScore
    algorithm_used: MappingAlgorithm
    mapped_smiles: str = ""
    reaction_center: Set[int] = Field(default_factory=set)

    def get_mapping_dict(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """
        Get mapping as dictionary from (mol_idx, atom_idx) -> (mol_idx, atom_idx).

        Returns:
            Dict[Tuple[int, int], Tuple[int, int]]: Dictionary mapping reactant atoms to product atoms.
        """
        return {
            (m.reactant_mol_idx, m.reactant_atom_idx): (
                m.product_mol_idx,
                m.product_atom_idx,
            )
            for m in self.atom_mappings
        }


class ReactionComponents(BaseModel):
    """
    Parsed components of a chemical reaction.

    Args:
        reactants (List[Any]): List of reactant molecules (RDKit Mol objects).
        products (List[Any]): List of product molecules (RDKit Mol objects).
        original_smiles (str): Original reaction SMILES.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reactants: List[Any]
    products: List[Any]
    original_smiles: str = ""

    @property
    def num_reactant_atoms(self) -> int:
        """Total number of atoms in reactants."""
        return sum(mol.GetNumAtoms() for mol in self.reactants)

    @property
    def num_product_atoms(self) -> int:
        """Total number of atoms in products."""
        return sum(mol.GetNumAtoms() for mol in self.products)
