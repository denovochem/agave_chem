import itertools

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

_TAUTOMER_ENUMERATOR: rdMolStandardize.TautomerEnumerator = (
    rdMolStandardize.TautomerEnumerator()
)

_RESONANCE_SWAP_PATTERNS: list[tuple[Chem.Mol, int, int]] = [
    (Chem.MolFromSmarts("[O:1]=[N+][O-:2]"), 1, 2),
    (Chem.MolFromSmarts("[O:1]=C[O-:2]"), 1, 2),
    (Chem.MolFromSmarts("[O:1]=C[OH:2]"), 1, 2),
    (Chem.MolFromSmarts("[O:1]=[S](=O)[O-:2]"), 1, 2),
]


def rxn_to_mapping_graph(rxn_smiles: str) -> nx.Graph:
    """
    Convert a mapped reaction SMILES into a NetworkX graph containing reactant/product atom nodes, bond edges, and atom-mapping edges.

    Args:
        rxn_smiles (str): Reaction SMILES (or SMIRKS-like) string that can be parsed by RDKit via
            `AllChem.ReactionFromSmarts(..., useSmiles=True)`.

    Returns:
        nx.Graph: An undirected graph where:
            - Nodes are labeled as `(side, frag_i, atom_i)` with `side` in `{"R", "P"}` and include attributes:
                `Z`, `side`, `charge`, `aromatic`, `in_ring`, `hydrogen_count`, `degree`, `chiral_tag`
                (CIP chirality code: ``'R'``, ``'S'``, or ``''`` for achiral / unassigned atoms).
            - Edges include:
                - Bond edges within each fragment with `kind="bond"` and `order` as the bond order (integer) and `stereo` as the bond stereo.
                - Mapping edges between reactant and product atoms with the same atom-map number, with `kind="map"`.

    Raises:
        Exception: Propagates RDKit parsing/usage errors if `rxn_smiles` cannot be parsed into a reaction or if
            reaction templates cannot be accessed.

    Note:
        Reactant atoms whose atom-map numbers do not appear anywhere on the product side have their atom-map
        numbers cleared (set to 0) before mapping edges are added. This mutates the underlying RDKit reactant
        template atoms created for this reaction object.
    """
    rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)  # type: ignore
    G = nx.Graph()

    def add_side(mols, side_label):
        atom_nodes = {}  # (frag_i, atom_i) -> node_id
        for frag_i, mol in enumerate(mols):
            # RDKit reaction templates may not have an initialized property cache.
            # Some atom properties (e.g., total H count) require this to be computed.
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
            # AssignStereochemistry (new CIP labeler) requires hybridization to be
            # set; UpdatePropertyCache does not set it.  Run the two lightweight
            # sanitization steps that the CIP labeler depends on, ignoring any
            # errors (e.g. kekulization failures on partial template structures).
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
                catchErrors=True,
            )
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            for atom in mol.GetAtoms():
                a_i = atom.GetIdx()
                node_id = (side_label, frag_i, a_i)
                atom_nodes[(frag_i, a_i)] = node_id
                G.add_node(
                    node_id,
                    Z=atom.GetAtomicNum(),
                    side=side_label,
                    charge=atom.GetFormalCharge(),
                    aromatic=atom.GetIsAromatic(),
                    in_ring=atom.IsInRing(),
                    hydrogen_count=atom.GetTotalNumHs(),
                    degree=atom.GetDegree(),
                    chiral_tag=atom.GetProp("_CIPCode")
                    if atom.HasProp("_CIPCode")
                    else "",
                )
            for bond in mol.GetBonds():
                a = (side_label, frag_i, bond.GetBeginAtomIdx())
                b = (side_label, frag_i, bond.GetEndAtomIdx())
                G.add_edge(
                    a,
                    b,
                    kind="bond",
                    order=int(bond.GetBondTypeAsDouble()),
                    stereo=int(bond.GetStereo()),
                )
        return mols

    r_mols = [rxn.GetReactantTemplate(i) for i in range(rxn.GetNumReactantTemplates())]
    p_mols = [rxn.GetProductTemplate(i) for i in range(rxn.GetNumProductTemplates())]

    add_side(r_mols, "R")
    add_side(p_mols, "P")

    # Add mapping edges based on atom-map numbers *within this reaction*
    p_map = {}
    product_map_nums = set()
    for frag_i, mol in enumerate(p_mols):
        for atom in mol.GetAtoms():
            m = atom.GetAtomMapNum()
            product_map_nums.add(m)
            if m:
                p_map[m] = ("P", frag_i, atom.GetIdx())

    r_map = {}
    for frag_i, mol in enumerate(r_mols):
        for atom in mol.GetAtoms():
            m = atom.GetAtomMapNum()
            if m not in product_map_nums:
                atom.SetAtomMapNum(0)
            if m:
                r_map[m] = ("R", frag_i, atom.GetIdx())

    for m, r_node in r_map.items():
        p_node = p_map.get(m)
        if p_node is not None:
            G.add_edge(r_node, p_node, kind="map")

    return G


def _get_query_atom_idx(pattern: Chem.Mol, map_num: int) -> int:
    """
    Return the 0-indexed atom index within *pattern* of the atom carrying *map_num*.

    Args:
        pattern (Chem.Mol): A compiled SMARTS pattern.
        map_num (int): The query atom-map number to look up.

    Returns:
        int: Index of the atom in *pattern* whose atom-map number equals *map_num*.

    Raises:
        ValueError: If no atom in *pattern* carries *map_num*.
    """
    for atom in pattern.GetAtoms():
        if atom.GetAtomMapNum() == map_num:
            return atom.GetIdx()
    raise ValueError(f"No atom with map number {map_num} found in pattern")


def _mol_frags_from_rxn(
    rxn_smiles: str,
) -> tuple[list[Chem.RWMol], list[Chem.RWMol]]:
    """
    Parse a reaction SMILES and return its reactant and product fragment mols with property caches initialized.

    Args:
        rxn_smiles (str): Reaction SMILES parsable via AllChem.ReactionFromSmarts(..., useSmiles=True).

    Returns:
        Tuple[List[Chem.RWMol], List[Chem.RWMol]]:
            - First list: reactant fragment mols with UpdatePropertyCache applied.
            - Second list: product fragment mols with UpdatePropertyCache applied.
    """
    rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)  # type: ignore
    r_mols: list[Chem.RWMol] = []
    for i in range(rxn.GetNumReactantTemplates()):
        mol = Chem.RWMol(rxn.GetReactantTemplate(i))
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
        r_mols.append(mol)
    p_mols: list[Chem.RWMol] = []
    for i in range(rxn.GetNumProductTemplates()):
        mol = Chem.RWMol(rxn.GetProductTemplate(i))
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
        p_mols.append(mol)
    return r_mols, p_mols


def _mols_to_rxn_smiles(r_mols: list[Chem.Mol], p_mols: list[Chem.Mol]) -> str:
    """
    Reconstruct a reaction SMILES string from lists of reactant and product mols.

    Args:
        r_mols (List[Chem.Mol]): Reactant fragment mols.
        p_mols (List[Chem.Mol]): Product fragment mols.

    Returns:
        str: Reaction SMILES of the form "reactants>>products" where multiple
            fragments on each side are joined with ".".
    """
    r_smi = ".".join(Chem.MolToSmiles(m) for m in r_mols)
    p_smi = ".".join(Chem.MolToSmiles(m) for m in p_mols)
    return f"{r_smi}>>{p_smi}"


def normalize_rxn_atom_maps(rxn_smiles: str) -> str:
    """
    Return a reaction SMILES with reactant atom-map numbers cleared for any atom
    whose map number does not appear on the product side.

    Atom-map numbers that appear only in the reactants (i.e. on leaving-group atoms
    that are absent from the products) carry no structural information about the bond
    changes and can cause spurious mismatches during reaction comparison.  This
    function sets those map numbers to 0 so the SMILES is in a canonical,
    comparison-ready form.

    Args:
        rxn_smiles (str): Reaction SMILES parsable via
            ``AllChem.ReactionFromSmarts(..., useSmiles=True)``.

    Returns:
        str: Reaction SMILES where every reactant atom whose original map number is
            absent from all product atoms has had its map number set to 0.  Product
            atom-map numbers are left unchanged.
    """
    r_mols, p_mols = _mol_frags_from_rxn(rxn_smiles)

    product_map_nums: set[int] = set()
    for mol in p_mols:
        for atom in mol.GetAtoms():
            m = atom.GetAtomMapNum()
            if m:
                product_map_nums.add(m)

    for mol in r_mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() not in product_map_nums:
                atom.SetAtomMapNum(0)

    return _mols_to_rxn_smiles(r_mols, p_mols)


def _enumerate_tautomers_for_mol(mol: Chem.RWMol) -> list[Chem.Mol]:
    """
    Return all tautomeric forms of *mol*, falling back to the input if enumeration fails or yields nothing.

    Args:
        mol (Chem.RWMol): Molecule to enumerate tautomers for.

    Returns:
        List[Chem.Mol]: List of tautomeric forms. Never empty.
    """
    try:
        tauts = list(_TAUTOMER_ENUMERATOR.Enumerate(mol))
    except Exception:
        tauts = []
    return tauts if tauts else [mol]


def _enumerate_tautomeric_rxn_smiles(rxn_smiles: str) -> list[str]:
    """
    Generate all tautomeric variants of a reaction by independently enumerating tautomers for each fragment.

    Args:
        rxn_smiles (str): Reaction SMILES parsable by RDKit.

    Returns:
        List[str]: All reaction SMILES variants arising from the Cartesian product of
            tautomeric forms across all reactant and product fragments. Always includes
            at least one entry.
    """
    r_mols, p_mols = _mol_frags_from_rxn(rxn_smiles)
    r_taut_lists = [_enumerate_tautomers_for_mol(m) for m in r_mols]
    p_taut_lists = [_enumerate_tautomers_for_mol(m) for m in p_mols]
    variants: list[str] = []
    for r_combo in itertools.product(*r_taut_lists):
        for p_combo in itertools.product(*p_taut_lists):
            variants.append(_mols_to_rxn_smiles(list(r_combo), list(p_combo)))
    return variants


def _enumerate_resonance_swap_variants(rxn_smiles: str) -> list[str]:
    """
    Generate all variants of a reaction by swapping atom-map numbers between resonance-equivalent atom pairs.

    For each SMARTS pattern in `_RESONANCE_SWAP_PATTERNS`, substructure matches are
    found in every reactant and product fragment. A matched pair is added as a swap
    candidate only when at least one atom in the pair carries a non-zero atom-map
    number. All 2^n combinations (n = number of candidates) are returned.

    Args:
        rxn_smiles (str): Reaction SMILES parsable by RDKit.

    Returns:
        List[str]: All reaction SMILES variants, including the original (all-swaps-off
            combination). Returns a single-element list containing the original if no
            mapped swap candidates are found.
    """
    r_mols, p_mols = _mol_frags_from_rxn(rxn_smiles)
    all_mols: list[Chem.RWMol] = r_mols + p_mols
    n_r = len(r_mols)

    swap_candidates: list[tuple[int, int, int]] = []
    for mol_idx, mol in enumerate(all_mols):
        for pattern, qmap1, qmap2 in _RESONANCE_SWAP_PATTERNS:
            q_idx1 = _get_query_atom_idx(pattern, qmap1)
            q_idx2 = _get_query_atom_idx(pattern, qmap2)
            for match in mol.GetSubstructMatches(pattern):
                a_idx = match[q_idx1]
                b_idx = match[q_idx2]
                map_a = mol.GetAtomWithIdx(a_idx).GetAtomMapNum()
                map_b = mol.GetAtomWithIdx(b_idx).GetAtomMapNum()
                if map_a or map_b:
                    swap_candidates.append((mol_idx, a_idx, b_idx))

    if not swap_candidates:
        return [rxn_smiles]

    variants: list[str] = []
    for mask in itertools.product([False, True], repeat=len(swap_candidates)):
        mols_copy = [Chem.RWMol(m) for m in all_mols]
        for do_swap, (mol_idx, a_idx, b_idx) in zip(mask, swap_candidates):
            if do_swap:
                mol = mols_copy[mol_idx]
                map_a = mol.GetAtomWithIdx(a_idx).GetAtomMapNum()
                map_b = mol.GetAtomWithIdx(b_idx).GetAtomMapNum()
                mol.GetAtomWithIdx(a_idx).SetAtomMapNum(map_b)
                mol.GetAtomWithIdx(b_idx).SetAtomMapNum(map_a)
        variants.append(_mols_to_rxn_smiles(mols_copy[:n_r], mols_copy[n_r:]))
    return variants


def mapping_equivalent(
    rxn1: str,
    rxn2: str,
    consider_tautomers: bool = False,
    consider_resonance_swaps: bool = False,
) -> bool:
    """
    Determine whether two mapped reaction strings are equivalent under atom mapping,
    optionally considering tautomeric and resonance-symmetric variants.

    rxn1 is always compared as a single fixed graph. A fast-path direct comparison is
    attempted first. If that fails and either flag is True, rxn2 variants are generated
    (tautomers via Cartesian product across fragments, resonance-swap variants via
    `_RESONANCE_SWAP_PATTERNS`) and each is compared against rxn1.

    Args:
        rxn1 (str): Reference mapped reaction SMILES parsable by RDKit.
        rxn2 (str): Query mapped reaction SMILES to compare against rxn1.
        consider_tautomers (bool): If True, enumerate tautomeric forms of each rxn2
            fragment and include all Cartesian-product combinations as candidates.
        consider_resonance_swaps (bool): If True, generate variants of rxn2 by swapping
            atom-map numbers between resonance-equivalent atom pairs (defined by
            `_RESONANCE_SWAP_PATTERNS`) when at least one atom in the pair is mapped.

    Returns:
        bool: True if rxn1 is graph-isomorphic to rxn2 or to any generated variant of
            rxn2; otherwise False.

    Raises:
        Exception: Propagates exceptions from `rxn_to_mapping_graph` if rxn1 cannot
            be parsed. Parsing failures for individual rxn2 variants are silently
            skipped.

    Note:
        Node matching requires equality of: atomic number, side (reactant/product),
        formal charge, aromaticity, ring membership, total hydrogen count, degree, and
        CIP chirality code (``''`` for achiral/unassigned, ``'R'``, or ``'S'``). Edge
        matching requires equality of kind (bond vs map), bond order, and bond stereo.
        Variants are generated only for rxn2; rxn1 is compared as-is.  Both
        inputs are normalized via `normalize_rxn_atom_maps` before comparison,
        so reactant-only atom-map numbers do not affect the result.
    """
    # rxn1 = normalize_rxn_atom_maps(rxn1)
    # rxn2 = normalize_rxn_atom_maps(rxn2)
    G1 = rxn_to_mapping_graph(rxn1)

    def node_match(a, b):
        return (
            a["Z"] == b["Z"]
            and a["side"] == b["side"]
            and a["charge"] == b["charge"]
            and a["aromatic"] == b["aromatic"]
            and a["in_ring"] == b["in_ring"]
            and a["hydrogen_count"] == b["hydrogen_count"]
            and a["degree"] == b["degree"]
            and a["chiral_tag"] == b["chiral_tag"]
        )

    def edge_match(a, b):
        if a["kind"] != b["kind"]:
            return False
        if a["kind"] == "bond":
            return a.get("order") == b.get("order") and a.get("stereo", 0) == b.get(
                "stereo", 0
            )
        return True  # kind == "map"

    try:
        G2 = rxn_to_mapping_graph(rxn2)
        if nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=edge_match):
            return True
    except Exception:
        pass

    if not consider_tautomers and not consider_resonance_swaps:
        return False

    taut_rxns = _enumerate_tautomeric_rxn_smiles(rxn2) if consider_tautomers else [rxn2]

    candidate_smiles: list[str] = []
    for taut_rxn in taut_rxns:
        if consider_resonance_swaps:
            candidate_smiles.extend(_enumerate_resonance_swap_variants(taut_rxn))
        else:
            candidate_smiles.append(taut_rxn)

    for smi in candidate_smiles:
        try:
            G2 = rxn_to_mapping_graph(smi)
        except Exception:
            continue
        if nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=edge_match):
            return True
    return False
