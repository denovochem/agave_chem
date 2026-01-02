"""
Maximum Common Substructure based atom-to-atom mapping.

This module provides the core MCS-based functionality for finding
atom correspondences between reactants and products.
"""

from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from rdkit import Chem
from rdkit.Chem import rdFMCS
from dataclasses import dataclass

from agave_chem.mappers.data_classes import AtomMapping
from agave_chem.utils.logging_config import logger


@dataclass
class MCSResult:
    """Result of a Maximum Common Substructure search."""

    smarts: str
    num_atoms: int
    num_bonds: int
    query_match: Tuple[int, ...]
    target_match: Tuple[int, ...]


class MCSMapper:
    """
    Atom mapper using Maximum Common Substructure search.

    This class implements MCS-based atom-to-atom mapping similar to
    the approach used in RDTool. It finds the largest common substructure
    between reactant and product molecules and uses this to establish
    atom correspondences.

    Attributes:
        timeout: Maximum time in seconds for MCS search
        ring_matches_ring_only: Whether ring atoms only match ring atoms
        complete_rings_only: Whether to require complete ring matches
        match_valences: Whether to require matching valences
        match_chiral_tag: Whether to consider stereochemistry
    """

    def __init__(
        self,
        timeout: int = 60,
        ring_matches_ring_only: bool = True,
        complete_rings_only: bool = True,
        match_valences: bool = False,
        match_chiral_tag: bool = False,
    ):
        """
        Initialize the MCS mapper.

        Args:
            timeout: Maximum seconds for MCS search
            ring_matches_ring_only: Ring atoms only match ring atoms
            complete_rings_only: Require complete ring matches
            match_valences: Require matching valences
            match_chiral_tag: Consider stereochemistry in matching
        """
        self.timeout = timeout
        self.ring_matches_ring_only = ring_matches_ring_only
        self.complete_rings_only = complete_rings_only
        self.match_valences = match_valences
        self.match_chiral_tag = match_chiral_tag

    def find_mcs(
        self, mol1: Chem.Mol, mol2: Chem.Mol, atom_compare: str = "elements"
    ) -> Optional[MCSResult]:
        """
        Find Maximum Common Substructure between two molecules.

        Args:
            mol1: First molecule
            mol2: Second molecule
            atom_compare: Comparison mode ('elements', 'isotopes', 'any')

        Returns:
            MCSResult object or None if no MCS found
        """
        # Configure atom comparison
        if atom_compare == "elements":
            atom_compare_param = rdFMCS.AtomCompare.CompareElements
        elif atom_compare == "isotopes":
            atom_compare_param = rdFMCS.AtomCompare.CompareIsotopes
        else:
            atom_compare_param = rdFMCS.AtomCompare.CompareAny

        # Configure bond comparison
        bond_compare = rdFMCS.BondCompare.CompareOrder

        try:
            mcs_result = rdFMCS.FindMCS(
                [mol1, mol2],
                timeout=self.timeout,
                atomCompare=atom_compare_param,
                bondCompare=bond_compare,
                ringMatchesRingOnly=self.ring_matches_ring_only,
                completeRingsOnly=self.complete_rings_only,
                matchValences=self.match_valences,
                matchChiralTag=self.match_chiral_tag,
                maximizeBonds=True,
            )
        except Exception as e:
            logger.error(f"MCS search failed: {e}")
            return None

        if mcs_result.canceled:
            logger.warning("MCS search timed out")

        if not mcs_result.smartsString:
            return None

        # Get the MCS as a query molecule
        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        if mcs_mol is None:
            return None

        # Find matches in both molecules
        matches1 = mol1.GetSubstructMatches(mcs_mol)
        matches2 = mol2.GetSubstructMatches(mcs_mol)

        if not matches1 or not matches2:
            return None

        return MCSResult(
            smarts=mcs_result.smartsString,
            num_atoms=mcs_result.numAtoms,
            num_bonds=mcs_result.numBonds,
            query_match=matches1[0],
            target_match=matches2[0],
        )

    def find_all_mcs_mappings(
        self, mol1: Chem.Mol, mol2: Chem.Mol, max_matches: int = 100
    ) -> List[Dict[int, int]]:
        """
        Find all possible atom mappings based on MCS.

        Args:
            mol1: First molecule (reactant)
            mol2: Second molecule (product)
            max_matches: Maximum number of mappings to return

        Returns:
            List of dictionaries mapping mol1 atom indices to mol2 atom indices
        """
        mcs_result = self.find_mcs(mol1, mol2)
        if mcs_result is None:
            return []

        # Get all substructure matches
        mcs_mol = Chem.MolFromSmarts(mcs_result.smarts)
        if mcs_mol is None:
            return []

        matches1 = mol1.GetSubstructMatches(mcs_mol, maxMatches=max_matches)
        matches2 = mol2.GetSubstructMatches(mcs_mol, maxMatches=max_matches)

        mappings = []
        for match1 in matches1:
            for match2 in matches2:
                # Create mapping from this pair of matches
                mapping = {match1[i]: match2[i] for i in range(len(match1))}
                mappings.append(mapping)

                if len(mappings) >= max_matches:
                    return mappings

        return mappings

    def extend_mapping_greedy(
        self, mol1: Chem.Mol, mol2: Chem.Mol, initial_mapping: Dict[int, int]
    ) -> Dict[int, int]:
        """
        Extend a partial mapping greedily to cover unmapped atoms.

        Args:
            mol1: First molecule
            mol2: Second molecule
            initial_mapping: Starting atom mapping

        Returns:
            Extended atom mapping
        """
        mapping = dict(initial_mapping)
        used_in_mol2 = set(mapping.values())

        # Get unmapped atoms
        unmapped1 = [i for i in range(mol1.GetNumAtoms()) if i not in mapping]
        unmapped2 = [i for i in range(mol2.GetNumAtoms()) if i not in used_in_mol2]

        # Score-based greedy extension
        for atom1_idx in unmapped1:
            atom1 = mol1.GetAtomWithIdx(atom1_idx)
            best_score = -float("inf")
            best_match = None

            for atom2_idx in unmapped2:
                if atom2_idx in used_in_mol2:
                    continue

                atom2 = mol2.GetAtomWithIdx(atom2_idx)

                # Must be same element
                if atom1.GetAtomicNum() != atom2.GetAtomicNum():
                    continue

                # Calculate compatibility score
                score = self._atom_compatibility_score(
                    mol1, atom1_idx, mol2, atom2_idx, mapping
                )

                if score > best_score:
                    best_score = score
                    best_match = atom2_idx

            if best_match is not None:
                mapping[atom1_idx] = best_match
                used_in_mol2.add(best_match)

        return mapping

    def _atom_compatibility_score(
        self,
        mol1: Chem.Mol,
        idx1: int,
        mol2: Chem.Mol,
        idx2: int,
        current_mapping: Dict[int, int],
    ) -> float:
        """
        Calculate compatibility score for mapping two atoms.

        Higher scores indicate better compatibility.

        Args:
            mol1: First molecule
            idx1: Atom index in mol1
            mol2: Second molecule
            idx2: Atom index in mol2
            current_mapping: Current partial mapping

        Returns:
            Compatibility score
        """
        atom1 = mol1.GetAtomWithIdx(idx1)
        atom2 = mol2.GetAtomWithIdx(idx2)
        score = 0.0

        # Base score for same element (already checked in caller)
        score += 10.0

        # Bonus for matching formal charge
        if atom1.GetFormalCharge() == atom2.GetFormalCharge():
            score += 5.0

        # Bonus for matching aromaticity
        if atom1.GetIsAromatic() == atom2.GetIsAromatic():
            score += 3.0

        # Bonus for matching ring membership
        if atom1.IsInRing() == atom2.IsInRing():
            score += 2.0

        # Bonus for each mapped neighbor that is also adjacent
        neighbors1 = {n.GetIdx() for n in atom1.GetNeighbors()}
        neighbors2 = {n.GetIdx() for n in atom2.GetNeighbors()}

        for n1 in neighbors1:
            if n1 in current_mapping:
                if current_mapping[n1] in neighbors2:
                    score += 15.0  # Strong bonus for maintaining connectivity

        # Penalty for mismatched degree
        score -= abs(atom1.GetDegree() - atom2.GetDegree())

        # Penalty for mismatched hydrogen count
        score -= abs(atom1.GetTotalNumHs() - atom2.GetTotalNumHs()) * 0.5

        return score


class MultiMolMCSMapper:
    """
    MCS-based mapper for reactions with multiple reactant/product molecules.

    This class handles the complexity of mapping atoms across multiple
    reactant and product molecules in a reaction.
    """

    def __init__(self, mcs_mapper: Optional[MCSMapper] = None):
        """
        Initialize with an optional custom MCS mapper.

        Args:
            mcs_mapper: MCS mapper instance to use, or None for default
        """
        self.mcs_mapper = mcs_mapper or MCSMapper()

    def find_best_molecule_pairing(
        self, reactants: List[Chem.Mol], products: List[Chem.Mol]
    ) -> List[Tuple[int, int, float]]:
        """
        Find optimal pairing of reactant and product molecules.

        Uses MCS size to score pairings and find which reactant
        molecules correspond to which product molecules.

        Args:
            reactants: List of reactant molecules
            products: List of product molecules

        Returns:
            List of (reactant_idx, product_idx, mcs_score) tuples
        """
        scores = []

        for r_idx, reactant in enumerate(reactants):
            for p_idx, product in enumerate(products):
                mcs = self.mcs_mapper.find_mcs(reactant, product)
                if mcs:
                    # Score based on fraction of atoms in MCS
                    r_fraction = mcs.num_atoms / reactant.GetNumAtoms()
                    p_fraction = mcs.num_atoms / product.GetNumAtoms()
                    score = (r_fraction + p_fraction) / 2
                else:
                    score = 0.0

                scores.append((r_idx, p_idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: -x[2])

        # Greedy assignment
        used_reactants: Set[int] = set()
        used_products: Set[int] = set()
        pairings = []

        for r_idx, p_idx, score in scores:
            if r_idx not in used_reactants and p_idx not in used_products:
                if score > 0:
                    pairings.append((r_idx, p_idx, score))
                    used_reactants.add(r_idx)
                    used_products.add(p_idx)

        return pairings

    def map_reaction(
        self, reactants: List[Chem.Mol], products: List[Chem.Mol]
    ) -> FrozenSet[AtomMapping]:
        """
        Map all atoms in a reaction.

        Args:
            reactants: List of reactant molecules
            products: List of product molecules

        Returns:
            Frozen set of atom mappings
        """
        all_mappings: List[AtomMapping] = []

        # Find molecule pairings
        pairings = self.find_best_molecule_pairing(reactants, products)

        for r_idx, p_idx, _ in pairings:
            reactant = reactants[r_idx]
            product = products[p_idx]

            # Find MCS-based mappings
            atom_mappings = self.mcs_mapper.find_all_mcs_mappings(
                reactant, product, max_matches=10
            )

            if atom_mappings:
                # Use first mapping and extend it
                base_mapping = atom_mappings[0]
                extended = self.mcs_mapper.extend_mapping_greedy(
                    reactant, product, base_mapping
                )

                for r_atom_idx, p_atom_idx in extended.items():
                    all_mappings.append(
                        AtomMapping(
                            reactant_mol_idx=r_idx,
                            reactant_atom_idx=r_atom_idx,
                            product_mol_idx=p_idx,
                            product_atom_idx=p_atom_idx,
                        )
                    )

        return frozenset(all_mappings)
