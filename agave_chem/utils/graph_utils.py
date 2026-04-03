import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem


def rxn_to_mapping_graph(rxn_smiles: str) -> nx.Graph:
    """
    Convert a mapped reaction SMILES into a NetworkX graph containing reactant/product atom nodes, bond edges, and atom-mapping edges.

    Args:
        rxn_smiles (str): Reaction SMILES (or SMIRKS-like) string that can be parsed by RDKit via
            `AllChem.ReactionFromSmarts(..., useSmiles=True)`.

    Returns:
        nx.Graph: An undirected graph where:
            - Nodes are labeled as `(side, frag_i, atom_i)` with `side` in `{"R", "P"}` and include attributes:
                `Z`, `side`, `charge`, `aromatic`, `in_ring`, `hydrogen_count`, `degree`, `chiral_tag`.
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
                    chiral_tag=int(atom.GetChiralTag()),
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


def mapping_equivalent(rxn1: str, rxn2: str) -> bool:
    """
    Determine whether two mapped reaction strings are equivalent under atom mapping, up to relabeling.

    This function converts each reaction string into a mapping graph via `rxn_to_mapping_graph` and
    returns True if the two graphs are isomorphic when matching a fixed set of atom (node) attributes
    and bond/mapping (edge) attributes.

    Args:
        rxn1 (str): First mapped reaction SMILES (or SMIRKS-like) string parsable by RDKit.
        rxn2 (str): Second mapped reaction SMILES (or SMIRKS-like) string parsable by RDKit.

    Returns:
        bool: True if the reactions are graph-isomorphic under the matching criteria; otherwise False.

    Raises:
        Exception: Propagates exceptions from `rxn_to_mapping_graph` (e.g., if either reaction string
            cannot be parsed by RDKit).

    Note:
        Node matching requires equality of: atomic number, side (reactant/product), formal charge,
        aromaticity, ring membership, total hydrogen count, degree, and chiral tag. Edge matching
        requires equality of `kind` (bond vs map), and for bond edges also requires equality of bond
        order and bond stereo.
    """
    G1 = rxn_to_mapping_graph(rxn1)
    G2 = rxn_to_mapping_graph(rxn2)

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
        # must match bond-vs-map edges; bonds also match order
        if a["kind"] != b["kind"]:
            return False
        if a["kind"] == "bond":
            return a.get("order") == b.get("order") and a.get("stereo", 0) == b.get(
                "stereo", 0
            )
        return True  # kind == "map"

    return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=edge_match)
