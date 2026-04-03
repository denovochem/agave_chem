import networkx as nx
from rdkit.Chem import AllChem


def rxn_to_mapping_graph(rxn_smiles: str) -> nx.Graph:
    rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
    G = nx.Graph()

    def add_side(mols, side_label):
        atom_nodes = {}  # (frag_i, atom_i) -> node_id
        for frag_i, mol in enumerate(mols):
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
                G.add_edge(a, b, kind="bond", order=int(bond.GetBondTypeAsDouble()))
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
            return a["order"] == b["order"]
        return True  # kind == "map"

    return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=edge_match)
