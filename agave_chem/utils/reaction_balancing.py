"""Utilities for detecting reactions that require non-one-to-one atom mapping."""

from collections import Counter, deque
from typing import Dict, Set

from rdkit import Chem


def detect_atom_count_imbalance(reaction_smiles: str) -> bool:
    """
    Check whether any element has more atoms on the product side than the reactant side.

    Parses the reaction SMILES, counts heavy atoms per element on each side,
    and returns ``True`` if any element's product count exceeds its reactant
    count.  This indicates that a one-to-one atom correspondence is impossible
    and the reaction likely needs balancing (e.g. duplicate reactant fragments).

    Args:
        reaction_smiles (str): Reaction SMILES of the form
            ``"reactants>>products"``.

    Returns:
        bool: ``True`` if any element has more atoms in the products than in
        the reactants, ``False`` otherwise.  Returns ``False`` if the SMILES
        cannot be parsed.
    """
    parts = reaction_smiles.strip().split(">>")
    if len(parts) != 2:
        return False

    reactants_str, products_str = parts
    reactant_counts: Counter[str] = Counter()
    product_counts: Counter[str] = Counter()

    for frag in reactants_str.split("."):
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            return False
        for atom in mol.GetAtoms():
            reactant_counts[atom.GetSymbol()] += 1

    for frag in products_str.split("."):
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            return False
        for atom in mol.GetAtoms():
            product_counts[atom.GetSymbol()] += 1

    for element, product_count in product_counts.items():
        if product_count > reactant_counts.get(element, 0):
            return True

    return False


def compute_unmapped_product_atom_islands(product_smiles: str) -> Dict[int, Set[int]]:
    """
    Find connected components ("islands") of unmapped atoms in a product SMILES.

    Performs a BFS over atoms with atom map number 0, grouping them into
    connected components based on molecular bonds.  Each island represents
    a contiguous region of unmapped atoms.

    Args:
        product_smiles (str): Product SMILES string to analyze.

    Returns:
        Dict[int, Set[int]]: Mapping from island index (0..N-1) to a set of
        RDKit atom indices belonging to that connected component, considering
        only atoms with atom map number equal to 0.

    Raises:
        ValueError: If the SMILES cannot be parsed into an RDKit molecule.
    """
    mol = Chem.MolFromSmiles(product_smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {product_smiles}")

    unmapped = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 0}

    visited: Set[int] = set()
    islands: Dict[int, Set[int]] = {}

    for idx in unmapped:
        if idx in visited:
            continue

        island: Set[int] = set()
        queue = deque([idx])
        visited.add(idx)

        while queue:
            current = queue.popleft()
            island.add(current)

            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in unmapped and neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)

        islands[len(islands)] = island

    return islands


def determine_one_to_one_correspondence(
    reaction_smiles: str,
    unmapped_product_atom_islands: Dict[int, Set[int]],
) -> bool:
    """
    Determine whether a reaction should use one-to-one atom correspondence.

    Returns ``False`` (meaning the reaction needs balancing) if either:
    - Any element has more atoms in the products than in the reactants, or
    - There is more than one island of unmapped atoms in the products after
      a conservative partial MCS mapping.

    Args:
        reaction_smiles (str): Reaction SMILES of the form
            ``"reactants>>products"``.
        unmapped_product_atom_islands (Dict[int, Set[int]]): Connected
            components of unmapped product atoms from a partial MCS mapping.
            Empty if MCS was not run or mapped everything.

    Returns:
        bool: ``True`` if one-to-one correspondence should be used (the
        reaction appears balanced), ``False`` if the reaction likely needs
        balancing.
    """
    if detect_atom_count_imbalance(reaction_smiles):
        return False

    return len(unmapped_product_atom_islands) <= 1
