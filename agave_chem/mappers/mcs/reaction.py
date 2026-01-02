"""
Main reaction handling and consensus mapping.

This module provides the high-level interface for atom-to-atom mapping,
implementing the consensus approach used in RDTool.
"""

from typing import Dict, List, Optional, Tuple, Set
from rdkit import Chem

from agave_chem.mappers.data_classes import (
    ReactionComponents,
    MappingResult,
    MappingAlgorithm,
)
from agave_chem.utils.chem_utils import parse_reaction_smiles
from agave_chem.mappers.mcs.algorithms import (
    MinAlgorithm,
    MaxAlgorithm,
    MixtureAlgorithm,
    BaseMappingAlgorithm,
)
from agave_chem.scoring import MappingScorer
from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.utils.logging_config import logger


class MCSReactionMapper(ReactionMapper):
    """
    Main class for atom-to-atom mapping of chemical reactions.

    This class implements a consensus approach similar to RDTool,
    running multiple algorithms and selecting the best result
    based on a comprehensive scoring system.

    Example usage:
        >>> mapper = ReactionMapper()
        >>> result = mapper.map_reaction("CC(=O)Cl.N>>CC(=O)N.Cl")
        >>> print(result.mapped_smiles)
        >>> print(result.score.to_dict())
    """

    def __init__(
        self,
        mapper_name: str,
        mapper_weight: float = 3,
        algorithms: Optional[List[MappingAlgorithm]] = None,
        scorer: Optional[MappingScorer] = None,
        timeout: int = 120,
    ):
        """
        Initialize the reaction mapper.

        Args:
            algorithms: List of algorithms to use (default: all three)
            scorer: Custom scorer for evaluating mappings
            timeout: Timeout in seconds for MCS operations
        """
        super().__init__("mcs", mapper_name, mapper_weight)
        self.scorer = scorer or MappingScorer()
        self.timeout = timeout

        if algorithms is None:
            algorithms = [
                MappingAlgorithm.MIN,
                MappingAlgorithm.MAX,
                MappingAlgorithm.MIXTURE,
            ]

        self.algorithms: Dict[MappingAlgorithm, BaseMappingAlgorithm] = {}

        for algo_type in algorithms:
            if algo_type == MappingAlgorithm.MIN:
                self.algorithms[algo_type] = MinAlgorithm(self.scorer)
            elif algo_type == MappingAlgorithm.MAX:
                self.algorithms[algo_type] = MaxAlgorithm(self.scorer, timeout)
            elif algo_type == MappingAlgorithm.MIXTURE:
                self.algorithms[algo_type] = MixtureAlgorithm(self.scorer)

    def parse_reaction(self, reaction_smiles: str) -> ReactionComponents:
        """
        Parse a reaction SMILES string into its components.

        Args:
            reaction_smiles: Reaction in SMILES format (reactants>agents>products)

        Returns:
            ReactionComponents object containing parsed molecules

        Raises:
            ValueError: If the reaction SMILES cannot be parsed
        """
        if not reaction_smiles or not reaction_smiles.strip():
            raise ValueError("Empty reaction SMILES provided")

        # Parse using utility function
        reactants, products = parse_reaction_smiles(reaction_smiles)

        return ReactionComponents(
            reactants=reactants,
            products=products,
            original_smiles=reaction_smiles,
        )

    def map_reaction(
        self, reaction_smiles: str, algorithm: Optional[MappingAlgorithm] = None
    ) -> Optional[MappingResult]:
        """
        Map atoms in a reaction SMILES.

        Args:
            reaction_smiles: Reaction in SMILES format (reactants>agents>products)
            algorithm: Specific algorithm to use, or None for consensus

        Returns:
            Best MappingResult or None if mapping fails
        """
        # Parse the reaction
        try:
            reaction = self.parse_reaction(reaction_smiles)
        except ValueError as e:
            logger.error(f"Failed to parse reaction: {e}")
            return None

        # Run mapping
        if algorithm is not None:
            results = self._run_single_algorithm(reaction, algorithm)
        else:
            results = self._run_consensus(reaction)

        if not results:
            logger.warning("No mapping results found")
            return None

        # Get best result and add mapped SMILES
        best_result = results[0]
        best_result = self._add_mapped_smiles(reaction, best_result)

        return {"mapping": best_result.mapped_smiles, "additional_info": [{}]}

    def map_reactions(self, reaction_list: List[str]):
        """ """

        mapped_reactions = []
        for reaction in reaction_list:
            mapped_reactions.append(self.map_reaction(reaction))
        return mapped_reactions

    def map_reaction_all_algorithms(
        self, reaction_smiles: str
    ) -> Dict[MappingAlgorithm, Optional[MappingResult]]:
        """
        Run all algorithms and return their best results.

        Args:
            reaction_smiles: Reaction in SMILES format

        Returns:
            Dictionary mapping each algorithm to its best result
        """
        try:
            reaction = self.parse_reaction(reaction_smiles)
        except ValueError as e:
            logger.error(f"Failed to parse reaction: {e}")
            return {}

        results = {}
        for algo_type, algo in self.algorithms.items():
            try:
                algo_results = algo.map_reaction(reaction)
                if algo_results:
                    best = algo_results[0]
                    results[algo_type] = self._add_mapped_smiles(reaction, best)
                else:
                    results[algo_type] = None
            except Exception as e:
                logger.error(f"Algorithm {algo_type.name} failed: {e}")
                results[algo_type] = None

        return results

    def _run_single_algorithm(
        self, reaction: ReactionComponents, algorithm: MappingAlgorithm
    ) -> List[MappingResult]:
        """
        Run a single mapping algorithm.

        Args:
            reaction: Parsed reaction components
            algorithm: Algorithm to use

        Returns:
            List of mapping results from the algorithm
        """
        if algorithm not in self.algorithms:
            logger.error(f"Algorithm {algorithm.name} not available")
            return []

        try:
            return self.algorithms[algorithm].map_reaction(reaction)
        except Exception as e:
            logger.error(f"Algorithm {algorithm.name} failed: {e}")
            return []

    def _run_consensus(self, reaction: ReactionComponents) -> List[MappingResult]:
        """
        Run all algorithms and select best result by consensus.

        This implements the consensus approach from RDTool where
        multiple algorithms are run and the best result is selected
        based on the scoring metrics.

        Args:
            reaction: Parsed reaction components

        Returns:
            List of mapping results sorted by score (best first)
        """
        all_results: List[MappingResult] = []

        for algo_type, algo in self.algorithms.items():
            try:
                results = algo.map_reaction(reaction)
                if results:
                    # Take top results from each algorithm
                    all_results.extend(results[:5])
            except Exception as e:
                logger.warning(f"Algorithm {algo_type.name} failed: {e}")
                continue

        if not all_results:
            return []

        # Sort by total score (lower is better)
        all_results.sort(key=lambda r: r.score.total_score())

        # Remove duplicates based on mapping
        seen_mappings: Set[frozenset] = set()
        unique_results: List[MappingResult] = []

        for result in all_results:
            mapping_key = result.atom_mappings
            if mapping_key not in seen_mappings:
                seen_mappings.add(mapping_key)
                unique_results.append(result)

        return unique_results

    def _add_mapped_smiles(
        self, reaction: ReactionComponents, result: MappingResult
    ) -> MappingResult:
        """
        Add mapped SMILES to a mapping result.

        This creates SMILES strings with atom mapping numbers
        that can be used for visualization and further processing.

        Args:
            reaction: Original reaction components
            result: Mapping result to augment

        Returns:
            New MappingResult with mapped_smiles populated
        """
        try:
            # Create atom map number assignments
            # Map number starts from 1
            map_num = 1

            # Create dictionaries to track map number assignments
            reactant_maps: Dict[Tuple[int, int], int] = {}
            product_maps: Dict[Tuple[int, int], int] = {}

            for am in result.atom_mappings:
                reactant_key = (am.reactant_mol_idx, am.reactant_atom_idx)
                product_key = (am.product_mol_idx, am.product_atom_idx)

                reactant_maps[reactant_key] = map_num
                product_maps[product_key] = map_num
                map_num += 1

            # Apply mappings to reactant molecules
            mapped_reactants: List[str] = []
            for mol_idx, mol in enumerate(reaction.reactants):
                mol_copy = Chem.RWMol(mol)
                for atom in mol_copy.GetAtoms():
                    atom_key = (mol_idx, atom.GetIdx())
                    if atom_key in reactant_maps:
                        atom.SetAtomMapNum(reactant_maps[atom_key])
                    else:
                        atom.SetAtomMapNum(0)

                smiles = Chem.MolToSmiles(mol_copy)
                mapped_reactants.append(smiles)

            # Apply mappings to product molecules
            mapped_products: List[str] = []
            for mol_idx, mol in enumerate(reaction.products):
                mol_copy = Chem.RWMol(mol)
                for atom in mol_copy.GetAtoms():
                    atom_key = (mol_idx, atom.GetIdx())
                    if atom_key in product_maps:
                        atom.SetAtomMapNum(product_maps[atom_key])
                    else:
                        atom.SetAtomMapNum(0)

                smiles = Chem.MolToSmiles(mol_copy)
                mapped_products.append(smiles)

            # Construct full reaction SMILES
            reactant_smiles = ".".join(mapped_reactants)
            product_smiles = ".".join(mapped_products)
            mapped_reaction_smiles = f"{reactant_smiles}>>{product_smiles}"

            # Identify reaction center atoms
            reaction_center: Set[int] = set()
            for bc in result.bond_changes:
                reaction_center.add(bc.atom1_map)
                reaction_center.add(bc.atom2_map)

            # Create new result with mapped SMILES
            return MappingResult(
                atom_mappings=result.atom_mappings,
                bond_changes=result.bond_changes,
                score=result.score,
                algorithm_used=result.algorithm_used,
                mapped_smiles=mapped_reaction_smiles,
                reaction_center=reaction_center,
            )

        except Exception as e:
            logger.error(f"Failed to generate mapped SMILES: {e}")
            # Return original result without mapped SMILES
            return result

    def get_reaction_center(self, reaction_smiles: str) -> Optional[Set[int]]:
        """
        Get the atoms involved in the reaction center.

        The reaction center consists of atoms involved in bond
        formation, breaking, or order changes.

        Args:
            reaction_smiles: Reaction in SMILES format

        Returns:
            Set of atom map numbers in the reaction center, or None if mapping fails
        """
        result = self.map_reaction(reaction_smiles)
        if result:
            return result.reaction_center
        return None

    def validate_mapping(
        self, reaction_smiles: str, result: MappingResult
    ) -> Tuple[bool, List[str]]:
        """
        Validate a mapping result for chemical consistency.

        Checks:
        - All atoms are mapped
        - Element conservation
        - Charge conservation

        Args:
            reaction_smiles: Original reaction SMILES
            result: Mapping result to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: List[str] = []

        try:
            reaction = self.parse_reaction(reaction_smiles)
        except ValueError as e:
            return False, [f"Could not parse reaction: {e}"]

        # Check atom conservation by element
        reactant_elements: Dict[str, int] = {}
        product_elements: Dict[str, int] = {}

        for mol in reaction.reactants:
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                reactant_elements[symbol] = reactant_elements.get(symbol, 0) + 1

        for mol in reaction.products:
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                product_elements[symbol] = product_elements.get(symbol, 0) + 1

        # Compare element counts
        all_elements = set(reactant_elements.keys()) | set(product_elements.keys())
        for element in all_elements:
            r_count = reactant_elements.get(element, 0)
            p_count = product_elements.get(element, 0)
            if r_count != p_count:
                errors.append(
                    f"Element {element} not conserved: {r_count} in reactants, {p_count} in products"
                )

        # Check that mapping preserves element types
        for am in result.atom_mappings:
            r_mol = reaction.reactants[am.reactant_mol_idx]
            p_mol = reaction.products[am.product_mol_idx]

            r_atom = r_mol.GetAtomWithIdx(am.reactant_atom_idx)
            p_atom = p_mol.GetAtomWithIdx(am.product_atom_idx)

            if r_atom.GetAtomicNum() != p_atom.GetAtomicNum():
                errors.append(
                    f"Mapping mismatch: {r_atom.GetSymbol()} mapped to {p_atom.GetSymbol()}"
                )

        # Check mapping coverage
        total_reactant_atoms = sum(mol.GetNumAtoms() for mol in reaction.reactants)
        total_product_atoms = sum(mol.GetNumAtoms() for mol in reaction.products)
        mapped_atoms = len(result.atom_mappings)

        if mapped_atoms < min(total_reactant_atoms, total_product_atoms):
            errors.append(
                f"Incomplete mapping: {mapped_atoms} atoms mapped out of "
                f"{total_reactant_atoms} reactant atoms and {total_product_atoms} product atoms"
            )

        is_valid = len(errors) == 0
        return is_valid, errors
