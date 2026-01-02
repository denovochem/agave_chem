"""
Mapping algorithm implementations.

This module implements the MIN, MAX, and MIXTURE algorithms
for atom-to-atom mapping, similar to those in RDTool.
"""

from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from rdkit import Chem
from abc import ABC, abstractmethod

from agave_chem.mappers.data_classes import (
    AtomMapping,
    MappingAlgorithm,
    MappingResult,
    MappingScore,
    ReactionComponents,
)
from agave_chem.mappers.mcs.mcs_mapper import MCSMapper, MultiMolMCSMapper
from agave_chem.scoring import MappingScorer
from agave_chem.utils.logging_config import logger


class BaseMappingAlgorithm(ABC):
    """
    Abstract base class for mapping algorithms.

    All mapping algorithms should inherit from this class
    and implement the map_reaction method.
    """

    def __init__(self, scorer: Optional[MappingScorer] = None):
        """
        Initialize the algorithm.

        Args:
            scorer: Optional scorer for evaluating mappings
        """
        self.scorer = scorer or MappingScorer()

    @abstractmethod
    def map_reaction(self, reaction: ReactionComponents) -> List[MappingResult]:
        """
        Generate atom mappings for a reaction.

        Args:
            reaction: Parsed reaction components

        Returns:
            List of possible mapping results, sorted by score
        """
        pass

    @property
    @abstractmethod
    def algorithm_type(self) -> MappingAlgorithm:
        """Return the algorithm type."""
        pass


class MinAlgorithm(BaseMappingAlgorithm):
    """
    MIN algorithm: Minimize bond changes.

    This algorithm focuses on finding mappings that minimize
    the total number of bonds formed and broken during the reaction.
    It prioritizes chemical conservatism.
    """

    def __init__(
        self, scorer: Optional[MappingScorer] = None, max_candidates: int = 100
    ):
        super().__init__(scorer)
        self.max_candidates = max_candidates
        self.mcs_mapper = MultiMolMCSMapper()

    @property
    def algorithm_type(self) -> MappingAlgorithm:
        return MappingAlgorithm.MIN

    def map_reaction(self, reaction: ReactionComponents) -> List[MappingResult]:
        """
        Generate mappings minimizing bond changes.

        Args:
            reaction: Parsed reaction components

        Returns:
            List of mapping results sorted by number of bond changes
        """
        # Generate candidate mappings using MCS
        base_mapping = self.mcs_mapper.map_reaction(
            reaction.reactants, reaction.products
        )

        if not base_mapping:
            logger.warning("No base mapping found")
            return []

        # Generate variations and score them
        candidates = self._generate_mapping_variations(reaction, base_mapping)

        # Score all candidates
        results = []
        for mapping in candidates:
            bond_changes = self.scorer.compute_bond_changes(
                reaction.reactants, reaction.products, mapping
            )
            score = self.scorer.score_mapping(
                reaction.reactants, reaction.products, mapping, bond_changes
            )

            results.append(
                MappingResult(
                    atom_mappings=mapping,
                    bond_changes=bond_changes,
                    score=score,
                    algorithm_used=self.algorithm_type,
                )
            )

        # Sort by number of bond changes (MIN objective)
        results.sort(key=lambda r: r.score.num_bond_changes)

        return results

    def _generate_mapping_variations(
        self, reaction: ReactionComponents, base_mapping: FrozenSet[AtomMapping]
    ) -> List[FrozenSet[AtomMapping]]:
        """
        Generate variations of a base mapping.

        Tries swapping equivalent atoms to find lower-cost alternatives.

        Args:
            reaction: Reaction components
            base_mapping: Starting mapping

        Returns:
            List of mapping variations including the original
        """
        variations = [base_mapping]

        # Find equivalent atoms that could be swapped
        mapping_list = list(base_mapping)

        # Group mappings by molecule pair
        mol_pair_mappings: Dict[Tuple[int, int], List[AtomMapping]] = {}
        for m in mapping_list:
            key = (m.reactant_mol_idx, m.product_mol_idx)
            if key not in mol_pair_mappings:
                mol_pair_mappings[key] = []
            mol_pair_mappings[key].append(m)

        # For each molecule pair, try swapping equivalent atoms
        for (r_idx, p_idx), mappings in mol_pair_mappings.items():
            reactant = reaction.reactants[r_idx]
            product = reaction.products[p_idx]

            # Find swappable atom pairs (same element, similar environment)
            swappable = self._find_swappable_pairs(reactant, product, mappings)

            for swap in swappable[:10]:  # Limit swaps to avoid explosion
                new_mapping_list = list(base_mapping)

                # Apply swap
                m1, m2 = swap
                idx1 = next(i for i, m in enumerate(new_mapping_list) if m == m1)
                idx2 = next(i for i, m in enumerate(new_mapping_list) if m == m2)

                # Swap product atoms
                new_m1 = AtomMapping(
                    m1.reactant_mol_idx,
                    m1.reactant_atom_idx,
                    m2.product_mol_idx,
                    m2.product_atom_idx,
                )
                new_m2 = AtomMapping(
                    m2.reactant_mol_idx,
                    m2.reactant_atom_idx,
                    m1.product_mol_idx,
                    m1.product_atom_idx,
                )

                new_mapping_list[idx1] = new_m1
                new_mapping_list[idx2] = new_m2

                variations.append(frozenset(new_mapping_list))

                if len(variations) >= self.max_candidates:
                    return variations

        return variations

    def _find_swappable_pairs(
        self, reactant: Chem.Mol, product: Chem.Mol, mappings: List[AtomMapping]
    ) -> List[Tuple[AtomMapping, AtomMapping]]:
        """Find pairs of mappings that could be swapped."""
        swappable = []

        for i, m1 in enumerate(mappings):
            for m2 in mappings[i + 1 :]:
                # Check if product atoms are same element
                atom1 = product.GetAtomWithIdx(m1.product_atom_idx)
                atom2 = product.GetAtomWithIdx(m2.product_atom_idx)

                if atom1.GetAtomicNum() == atom2.GetAtomicNum():
                    # Also check reactant atoms
                    r_atom1 = reactant.GetAtomWithIdx(m1.reactant_atom_idx)
                    r_atom2 = reactant.GetAtomWithIdx(m2.reactant_atom_idx)

                    if r_atom1.GetAtomicNum() == r_atom2.GetAtomicNum():
                        swappable.append((m1, m2))

        return swappable


class MaxAlgorithm(BaseMappingAlgorithm):
    """
    MAX algorithm: Maximize common substructure.

    This algorithm focuses on finding the largest possible
    common substructure between reactants and products,
    prioritizing structural preservation.
    """

    def __init__(self, scorer: Optional[MappingScorer] = None, timeout: int = 120):
        super().__init__(scorer)
        self.timeout = timeout
        self.mcs_mapper = MCSMapper(timeout=timeout, complete_rings_only=True)

    @property
    def algorithm_type(self) -> MappingAlgorithm:
        return MappingAlgorithm.MAX

    def map_reaction(self, reaction: ReactionComponents) -> List[MappingResult]:
        """
        Generate mappings maximizing common substructure.

        Args:
            reaction: Parsed reaction components

        Returns:
            List of mapping results sorted by MCS size (descending)
        """
        multi_mapper = MultiMolMCSMapper(self.mcs_mapper)

        # Find all possible MCS-based mappings
        all_mappings = self._find_all_mcs_mappings(reaction, multi_mapper)

        if not all_mappings:
            logger.warning("No MCS mappings found")
            return []

        # Score and create results
        results = []
        for mapping, mcs_size in all_mappings:
            bond_changes = self.scorer.compute_bond_changes(
                reaction.reactants, reaction.products, mapping
            )
            score = self.scorer.score_mapping(
                reaction.reactants, reaction.products, mapping, bond_changes
            )

            results.append(
                MappingResult(
                    atom_mappings=mapping,
                    bond_changes=bond_changes,
                    score=score,
                    algorithm_used=self.algorithm_type,
                )
            )

        # Sort by MCS coverage (similarity score), highest first
        results.sort(key=lambda r: -r.score.similarity_score)

        return results

    def _find_all_mcs_mappings(
        self,
        reaction: ReactionComponents,
        mapper: MultiMolMCSMapper,
        max_mappings: int = 50,
    ) -> List[Tuple[FrozenSet[AtomMapping], int]]:
        """Find all MCS-based mappings with their sizes."""
        results = []

        # Get molecule pairings
        pairings = mapper.find_best_molecule_pairing(
            reaction.reactants, reaction.products
        )

        # For each pairing, get all MCS mappings
        for r_idx, p_idx, _ in pairings:
            reactant = reaction.reactants[r_idx]
            product = reaction.products[p_idx]

            mcs_mappings = self.mcs_mapper.find_all_mcs_mappings(
                reactant, product, max_matches=20
            )

            for atom_mapping_dict in mcs_mappings:
                # Convert to AtomMapping objects
                mappings = frozenset(
                    AtomMapping(r_idx, r_atom, p_idx, p_atom)
                    for r_atom, p_atom in atom_mapping_dict.items()
                )

                mcs_size = len(atom_mapping_dict)
                results.append((mappings, mcs_size))

                if len(results) >= max_mappings:
                    break

        return results


class MixtureAlgorithm(BaseMappingAlgorithm):
    """
    MIXTURE algorithm: Hybrid approach.

    This algorithm combines aspects of both MIN and MAX algorithms,
    using a two-phase approach:
    1. First, find large common substructures (MAX-like)
    2. Then, optimize the remaining mappings for minimal bond changes (MIN-like)
    """

    def __init__(
        self,
        scorer: Optional[MappingScorer] = None,
        min_weight: float = 0.5,
        max_weight: float = 0.5,
    ):
        super().__init__(scorer)
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.min_algo = MinAlgorithm(scorer)
        self.max_algo = MaxAlgorithm(scorer)

    @property
    def algorithm_type(self) -> MappingAlgorithm:
        return MappingAlgorithm.MIXTURE

    def map_reaction(self, reaction: ReactionComponents) -> List[MappingResult]:
        """
        Generate mappings using hybrid approach.

        Args:
            reaction: Parsed reaction components

        Returns:
            List of mapping results with hybrid scoring
        """
        # Get results from both algorithms
        min_results = self.min_algo.map_reaction(reaction)
        max_results = self.max_algo.map_reaction(reaction)

        # Combine and re-score
        all_mappings: Set[FrozenSet[AtomMapping]] = set()

        for result in min_results[:20]:
            all_mappings.add(result.atom_mappings)

        for result in max_results[:20]:
            all_mappings.add(result.atom_mappings)

        # Score with hybrid metric
        results = []
        for mapping in all_mappings:
            bond_changes = self.scorer.compute_bond_changes(
                reaction.reactants, reaction.products, mapping
            )
            score = self.scorer.score_mapping(
                reaction.reactants, reaction.products, mapping, bond_changes
            )

            results.append(
                MappingResult(
                    atom_mappings=mapping,
                    bond_changes=bond_changes,
                    score=score,
                    algorithm_used=self.algorithm_type,
                )
            )

        # Sort by combined score
        results.sort(key=lambda r: self._hybrid_score(r.score))

        return results

    def _hybrid_score(self, score: MappingScore) -> float:
        """Calculate hybrid score combining MIN and MAX objectives."""
        min_component = score.num_bond_changes
        max_component = -score.similarity_score * 100  # Negate for minimization

        return self.min_weight * min_component + self.max_weight * max_component
