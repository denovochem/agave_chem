"""
Scoring functions for evaluating atom mappings.

This module provides comprehensive scoring metrics for atom-to-atom
mappings based on the criteria used in RDTool.
"""

from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from rdkit import Chem

from agave_chem.mappers.data_classes import (
    AtomMapping,
    MappingScore,
    BondChange,
    BondChangeType,
)
from agave_chem.utils.chem_utils import get_bond_energy, get_ring_info
from agave_chem.utils.logging_config import logger


class MappingScorer:
    """
    Comprehensive scorer for atom-to-atom mappings.

    This class computes all the metrics used to evaluate and compare
    different mapping solutions, following the approach in RDTool.

    Metrics include:
    - Bond energy cost (formation/breaking)
    - Number of bond changes
    - Number of fragments affected
    - Stereochemistry changes
    - Ring opening/closing events
    - Overall similarity score
    """

    def __init__(
        self,
        energy_penalty_weight: float = 1.0,
        bond_change_weight: float = 10.0,
        fragment_weight: float = 20.0,
        stereo_weight: float = 15.0,
        ring_weight: float = 25.0,
    ):
        """
        Initialize the scorer with custom weights.

        Args:
            energy_penalty_weight: Weight for bond energy cost
            bond_change_weight: Weight for number of bond changes
            fragment_weight: Weight for fragment changes
            stereo_weight: Weight for stereo changes
            ring_weight: Weight for ring changes
        """
        self.weights = {
            "bond_energy_cost": energy_penalty_weight,
            "num_bond_changes": bond_change_weight,
            "num_fragments": fragment_weight,
            "stereo_changes": stereo_weight,
            "ring_changes": ring_weight,
        }

    def score_mapping(
        self,
        reactants: List[Chem.Mol],
        products: List[Chem.Mol],
        mapping: FrozenSet[AtomMapping],
        bond_changes: Optional[List[BondChange]] = None,
    ) -> MappingScore:
        """
        Compute comprehensive score for a mapping.

        Args:
            reactants: List of reactant molecules
            products: List of product molecules
            mapping: Set of atom mappings
            bond_changes: Pre-computed bond changes (optional)

        Returns:
            MappingScore object with all metrics
        """
        if bond_changes is None:
            bond_changes = self.compute_bond_changes(reactants, products, mapping)

        # Count bond changes by type
        num_formed = sum(
            1 for bc in bond_changes if bc.change_type == BondChangeType.FORMED
        )
        num_broken = sum(
            1 for bc in bond_changes if bc.change_type == BondChangeType.BROKEN
        )
        num_order_changes = sum(
            1 for bc in bond_changes if bc.change_type == BondChangeType.ORDER_CHANGE
        )

        # Calculate total energy cost
        energy_cost = sum(bc.energy_cost for bc in bond_changes)

        # Calculate fragment changes
        num_fragments = self._count_fragment_changes(reactants, products, mapping)

        # Calculate stereo changes
        stereo_changes = self._count_stereo_changes(reactants, products, mapping)

        # Calculate ring changes
        ring_changes = self._count_ring_changes(
            reactants, products, mapping, bond_changes
        )

        # Calculate similarity score
        similarity = self._calculate_similarity(reactants, products, mapping)

        return MappingScore(
            bond_energy_cost=energy_cost,
            num_bond_changes=num_formed + num_broken + num_order_changes,
            num_bonds_formed=num_formed,
            num_bonds_broken=num_broken,
            num_fragments=num_fragments,
            stereo_changes=stereo_changes,
            similarity_score=similarity,
            ring_changes=ring_changes,
        )

    def compute_bond_changes(
        self,
        reactants: List[Chem.Mol],
        products: List[Chem.Mol],
        mapping: FrozenSet[AtomMapping],
    ) -> List[BondChange]:
        """
        Compute all bond changes in the reaction.

        Args:
            reactants: List of reactant molecules
            products: List of product molecules
            mapping: Set of atom mappings

        Returns:
            List of BondChange objects
        """
        changes = []

        # Create mapping lookup: atom_map_num -> (mol_type, mol_idx, atom_idx)
        # First, assign temporary map numbers
        map_num_counter = 1
        reactant_to_map: Dict[Tuple[int, int], int] = {}
        product_to_map: Dict[Tuple[int, int], int] = {}

        for am in mapping:
            reactant_to_map[(am.reactant_mol_idx, am.reactant_atom_idx)] = (
                map_num_counter
            )
            product_to_map[(am.product_mol_idx, am.product_atom_idx)] = map_num_counter
            map_num_counter += 1

        # Build bond sets for reactants (using map numbers)
        reactant_bonds: Dict[FrozenSet[int], Tuple[float, str, str]] = {}
        for mol_idx, mol in enumerate(reactants):
            for bond in mol.GetBonds():
                atom1_key = (mol_idx, bond.GetBeginAtomIdx())
                atom2_key = (mol_idx, bond.GetEndAtomIdx())

                if atom1_key in reactant_to_map and atom2_key in reactant_to_map:
                    map1 = reactant_to_map[atom1_key]
                    map2 = reactant_to_map[atom2_key]
                    bond_order = bond.GetBondTypeAsDouble()

                    atom1_sym = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
                    atom2_sym = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()

                    reactant_bonds[frozenset([map1, map2])] = (
                        bond_order,
                        atom1_sym,
                        atom2_sym,
                    )

        # Build bond sets for products (using map numbers)
        product_bonds: Dict[FrozenSet[int], Tuple[float, str, str]] = {}
        for mol_idx, mol in enumerate(products):
            for bond in mol.GetBonds():
                atom1_key = (mol_idx, bond.GetBeginAtomIdx())
                atom2_key = (mol_idx, bond.GetEndAtomIdx())

                if atom1_key in product_to_map and atom2_key in product_to_map:
                    map1 = product_to_map[atom1_key]
                    map2 = product_to_map[atom2_key]
                    bond_order = bond.GetBondTypeAsDouble()

                    atom1_sym = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
                    atom2_sym = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()

                    product_bonds[frozenset([map1, map2])] = (
                        bond_order,
                        atom1_sym,
                        atom2_sym,
                    )

        # Find broken bonds (in reactants but not products)
        for bond_key, (order, sym1, sym2) in reactant_bonds.items():
            if bond_key not in product_bonds:
                map1, map2 = sorted(bond_key)
                energy = get_bond_energy(sym1, sym2, order)
                changes.append(
                    BondChange(
                        atom1_map=map1,
                        atom2_map=map2,
                        change_type=BondChangeType.BROKEN,
                        old_order=order,
                        new_order=None,
                        energy_cost=energy,
                    )
                )

        # Find formed bonds (in products but not reactants)
        for bond_key, (order, sym1, sym2) in product_bonds.items():
            if bond_key not in reactant_bonds:
                map1, map2 = sorted(bond_key)
                energy = get_bond_energy(sym1, sym2, order)
                changes.append(
                    BondChange(
                        atom1_map=map1,
                        atom2_map=map2,
                        change_type=BondChangeType.FORMED,
                        old_order=None,
                        new_order=order,
                        energy_cost=energy,
                    )
                )

        # Find order changes (in both but different order)
        for bond_key in reactant_bonds.keys() & product_bonds.keys():
            r_order, r_sym1, r_sym2 = reactant_bonds[bond_key]
            p_order, _, _ = product_bonds[bond_key]

            if abs(r_order - p_order) > 0.1:
                map1, map2 = sorted(bond_key)
                # Energy change for order modification
                old_energy = get_bond_energy(r_sym1, r_sym2, r_order)
                new_energy = get_bond_energy(r_sym1, r_sym2, p_order)
                changes.append(
                    BondChange(
                        atom1_map=map1,
                        atom2_map=map2,
                        change_type=BondChangeType.ORDER_CHANGE,
                        old_order=r_order,
                        new_order=p_order,
                        energy_cost=abs(new_energy - old_energy),
                    )
                )

        return changes

    def _count_fragment_changes(
        self,
        reactants: List[Chem.Mol],
        products: List[Chem.Mol],
        mapping: FrozenSet[AtomMapping],
    ) -> int:
        """Count the number of molecular fragments that change."""
        # Simple heuristic: count difference in number of molecules
        return abs(len(reactants) - len(products))

    def _count_stereo_changes(
        self,
        reactants: List[Chem.Mol],
        products: List[Chem.Mol],
        mapping: FrozenSet[AtomMapping],
    ) -> int:
        """Count stereochemistry changes in the reaction."""
        changes = 0

        # Build lookup for mapping
        mapping_lookup: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for am in mapping:
            mapping_lookup[(am.reactant_mol_idx, am.reactant_atom_idx)] = (
                am.product_mol_idx,
                am.product_atom_idx,
            )

        # Check each mapped atom for stereo changes
        for am in mapping:
            r_mol = reactants[am.reactant_mol_idx]
            p_mol = products[am.product_mol_idx]

            r_atom = r_mol.GetAtomWithIdx(am.reactant_atom_idx)
            p_atom = p_mol.GetAtomWithIdx(am.product_atom_idx)

            # Check chiral tag
            r_chiral = r_atom.GetChiralTag()
            p_chiral = p_atom.GetChiralTag()

            if r_chiral != p_chiral:
                changes += 1

        return changes

    def _count_ring_changes(
        self,
        reactants: List[Chem.Mol],
        products: List[Chem.Mol],
        mapping: FrozenSet[AtomMapping],
        bond_changes: List[BondChange],
    ) -> int:
        """Count ring opening and closing events."""
        changes = 0

        # Create map number to molecule info lookup
        map_to_reactant: Dict[int, Tuple[int, int]] = {}
        map_to_product: Dict[int, Tuple[int, int]] = {}

        map_num = 1
        for am in mapping:
            map_to_reactant[map_num] = (am.reactant_mol_idx, am.reactant_atom_idx)
            map_to_product[map_num] = (am.product_mol_idx, am.product_atom_idx)
            map_num += 1

        # Get ring info for all molecules
        reactant_rings: List[Set[int]] = []
        for mol in reactants:
            reactant_rings.extend(get_ring_info(mol))

        product_rings: List[Set[int]] = []
        for mol in products:
            product_rings.extend(get_ring_info(mol))

        # Check each bond change for ring involvement
        for bc in bond_changes:
            if bc.atom1_map in map_to_reactant and bc.atom2_map in map_to_reactant:
                r_info1 = map_to_reactant.get(bc.atom1_map)
                r_info2 = map_to_reactant.get(bc.atom2_map)

                if r_info1 and r_info2:
                    r_mol_idx1, r_atom_idx1 = r_info1
                    r_mol_idx2, r_atom_idx2 = r_info2

                    # Check if both atoms were in same ring
                    if r_mol_idx1 == r_mol_idx2:
                        mol = reactants[r_mol_idx1]
                        atom1 = mol.GetAtomWithIdx(r_atom_idx1)
                        atom2 = mol.GetAtomWithIdx(r_atom_idx2)

                        if atom1.IsInRing() and atom2.IsInRing():
                            if bc.change_type == BondChangeType.BROKEN:
                                changes += 1  # Ring opening
                            elif bc.change_type == BondChangeType.FORMED:
                                changes += 1  # Ring closing

        return changes

    def _calculate_similarity(
        self,
        reactants: List[Chem.Mol],
        products: List[Chem.Mol],
        mapping: FrozenSet[AtomMapping],
    ) -> float:
        """
        Calculate overall similarity based on the mapping.

        Returns fraction of atoms that are successfully mapped.
        """
        total_reactant_atoms = sum(mol.GetNumAtoms() for mol in reactants)
        total_product_atoms = sum(mol.GetNumAtoms() for mol in products)

        if total_reactant_atoms == 0 or total_product_atoms == 0:
            return 0.0

        mapped_atoms = len(mapping)

        # Similarity as fraction of atoms mapped
        reactant_coverage = mapped_atoms / total_reactant_atoms
        product_coverage = mapped_atoms / total_product_atoms

        return (reactant_coverage + product_coverage) / 2
