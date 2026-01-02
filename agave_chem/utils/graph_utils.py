"""
Graph-based utilities for molecular matching.

This module provides graph theory based functions for molecular
substructure matching and isomorphism detection.
"""

from typing import Dict, List, Optional, Set, Tuple, Generator
from collections import defaultdict
from rdkit import Chem

from agave_chem.utils.logging_config import logger


class MolecularGraph:
    """
    Graph representation of a molecule for matching purposes.

    This class provides a graph-theoretic view of a molecule,
    enabling efficient subgraph isomorphism and matching operations.
    """

    def __init__(self, mol: Chem.Mol, mol_idx: int = 0):
        """
        Initialize molecular graph from RDKit molecule.

        Args:
            mol: RDKit molecule object
            mol_idx: Index identifier for this molecule
        """
        self.mol = mol
        self.mol_idx = mol_idx
        self.num_atoms = mol.GetNumAtoms()

        # Build adjacency list
        self.adjacency: Dict[int, Set[int]] = defaultdict(set)
        self.bond_info: Dict[Tuple[int, int], Dict] = {}

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            self.adjacency[i].add(j)
            self.adjacency[j].add(i)

            bond_key = (min(i, j), max(i, j))
            self.bond_info[bond_key] = {
                "order": bond.GetBondTypeAsDouble(),
                "is_aromatic": bond.GetIsAromatic(),
                "stereo": str(bond.GetStereo()),
            }

        # Cache atom properties
        self.atom_props: Dict[int, Dict] = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            self.atom_props[idx] = {
                "atomic_num": atom.GetAtomicNum(),
                "symbol": atom.GetSymbol(),
                "formal_charge": atom.GetFormalCharge(),
                "degree": atom.GetDegree(),
                "is_aromatic": atom.GetIsAromatic(),
                "is_in_ring": atom.IsInRing(),
                "num_hs": atom.GetTotalNumHs(),
            }

    def get_neighbors(self, atom_idx: int) -> Set[int]:
        """Get neighboring atom indices."""
        return self.adjacency[atom_idx]

    def get_bond(self, atom1_idx: int, atom2_idx: int) -> Optional[Dict]:
        """Get bond information between two atoms."""
        key = (min(atom1_idx, atom2_idx), max(atom1_idx, atom2_idx))
        return self.bond_info.get(key)

    def atom_matches(
        self,
        idx1: int,
        other: "MolecularGraph",
        idx2: int,
        match_charge: bool = True,
        match_aromaticity: bool = True,
    ) -> bool:
        """
        Check if two atoms are compatible for mapping.

        Args:
            idx1: Atom index in this graph
            other: Other molecular graph
            idx2: Atom index in other graph
            match_charge: Whether formal charges must match
            match_aromaticity: Whether aromaticity must match

        Returns:
            True if atoms are compatible
        """
        props1 = self.atom_props[idx1]
        props2 = other.atom_props[idx2]

        # Must have same element
        if props1["atomic_num"] != props2["atomic_num"]:
            return False

        # Optional: match formal charge
        if match_charge and props1["formal_charge"] != props2["formal_charge"]:
            return False

        # Optional: match aromaticity
        if match_aromaticity and props1["is_aromatic"] != props2["is_aromatic"]:
            return False

        return True

    def get_atom_signature(self, atom_idx: int, radius: int = 2) -> str:
        """
        Get a canonical signature for an atom based on its neighborhood.

        Args:
            atom_idx: Index of the atom
            radius: Neighborhood radius to consider

        Returns:
            String signature for the atom
        """
        props = self.atom_props[atom_idx]
        base = f"{props['symbol']}{props['formal_charge']}"

        if radius == 0:
            return base

        # Get sorted neighbor signatures
        neighbor_sigs = []
        for neighbor_idx in sorted(self.adjacency[atom_idx]):
            bond = self.get_bond(atom_idx, neighbor_idx)
            bond_char = self._bond_order_to_char(bond["order"] if bond else 1.0)
            neighbor_sig = self.get_atom_signature(neighbor_idx, radius - 1)
            neighbor_sigs.append(f"{bond_char}{neighbor_sig}")

        return f"{base}[{''.join(sorted(neighbor_sigs))}]"

    @staticmethod
    def _bond_order_to_char(order: float) -> str:
        """Convert bond order to character representation."""
        if order == 1.0:
            return "-"
        elif order == 2.0:
            return "="
        elif order == 3.0:
            return "#"
        elif order == 1.5:
            return ":"
        else:
            return "~"


def find_clique_mappings(
    graph1: MolecularGraph,
    graph2: MolecularGraph,
    initial_mapping: Optional[Dict[int, int]] = None,
    max_mappings: int = 1000,
) -> Generator[Dict[int, int], None, None]:
    """
    Find all maximal clique-based mappings between two molecular graphs.

    Uses a backtracking algorithm to find all valid atom mappings
    that preserve graph connectivity.

    Args:
        graph1: First molecular graph (typically reactant)
        graph2: Second molecular graph (typically product)
        initial_mapping: Optional starting mapping to extend
        max_mappings: Maximum number of mappings to generate

    Yields:
        Dictionaries mapping atom indices from graph1 to graph2
    """
    mapping = dict(initial_mapping) if initial_mapping else {}
    used_in_graph2: Set[int] = set(mapping.values())
    count = 0

    def is_consistent(atom1: int, atom2: int) -> bool:
        """Check if adding this mapping maintains consistency."""
        # Check atom compatibility
        if not graph1.atom_matches(atom1, graph2, atom2):
            return False

        # Check that all mapped neighbors have consistent mappings
        for neighbor1 in graph1.get_neighbors(atom1):
            if neighbor1 in mapping:
                neighbor2 = mapping[neighbor1]
                # The mapped neighbor must be adjacent to atom2
                if neighbor2 not in graph2.get_neighbors(atom2):
                    return False
                # Bond types should be compatible
                bond1 = graph1.get_bond(atom1, neighbor1)
                bond2 = graph2.get_bond(atom2, neighbor2)
                if bond1 and bond2:
                    if abs(bond1["order"] - bond2["order"]) > 0.1:
                        return False

        return True

    def backtrack(unmapped1: List[int]) -> Generator[Dict[int, int], None, None]:
        """Backtracking search for valid mappings."""
        nonlocal count

        if count >= max_mappings:
            return

        if not unmapped1:
            count += 1
            yield dict(mapping)
            return

        # Select next atom to map (prefer atoms with mapped neighbors)
        unmapped1 = sorted(
            unmapped1, key=lambda a: -len(graph1.get_neighbors(a) & set(mapping.keys()))
        )
        atom1 = unmapped1[0]
        remaining = unmapped1[1:]

        # Try all possible mappings for this atom
        for atom2 in range(graph2.num_atoms):
            if atom2 in used_in_graph2:
                continue

            if is_consistent(atom1, atom2):
                mapping[atom1] = atom2
                used_in_graph2.add(atom2)

                yield from backtrack(remaining)

                del mapping[atom1]
                used_in_graph2.remove(atom2)

        # Also try not mapping this atom (for partial mappings)
        yield from backtrack(remaining)

    unmapped = [i for i in range(graph1.num_atoms) if i not in mapping]
    yield from backtrack(unmapped)


def compute_graph_similarity(graph1: MolecularGraph, graph2: MolecularGraph) -> float:
    """
    Compute similarity score between two molecular graphs.

    Args:
        graph1: First molecular graph
        graph2: Second molecular graph

    Returns:
        Similarity score between 0 and 1
    """
    # Use atom signature-based comparison
    sigs1 = defaultdict(int)
    sigs2 = defaultdict(int)

    for i in range(graph1.num_atoms):
        sig = graph1.get_atom_signature(i, radius=2)
        sigs1[sig] += 1

    for i in range(graph2.num_atoms):
        sig = graph2.get_atom_signature(i, radius=2)
        sigs2[sig] += 1

    # Compute Tanimoto similarity on signature counts
    all_sigs = set(sigs1.keys()) | set(sigs2.keys())
    if not all_sigs:
        return 0.0

    intersection = sum(min(sigs1.get(s, 0), sigs2.get(s, 0)) for s in all_sigs)
    union = sum(max(sigs1.get(s, 0), sigs2.get(s, 0)) for s in all_sigs)

    return intersection / union if union > 0 else 0.0
