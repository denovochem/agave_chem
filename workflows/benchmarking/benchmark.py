import time

import networkx as nx
from rdkit.Chem import AllChem

from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper
from agave_chem.mappers.neural.neural_mapper import NeuralReactionMapper
from agave_chem.mappers.template.template_mapper import TemplateReactionMapper
from agave_chem.utils.chem_utils import canonicalize_reaction_smiles

mcs_mapper = MCSReactionMapper(mapper_name="mcs", mapper_weight=1)
neural_mapper = NeuralReactionMapper(mapper_name="neural_mapper", mapper_weight=1)
expert_mapper = TemplateReactionMapper("expert_default")


def rxn_to_mapping_graph(rxn_smiles: str) -> nx.Graph:
    # RDKit parses reaction SMILES via ReactionFromSmarts(useSmiles=True)
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
                    # frag=frag_i,
                    # you can add more invariants if you want:
                    charge=atom.GetFormalCharge(),
                    aromatic=atom.GetIsAromatic(),
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
            # and a["frag"] == b["frag"]
            and a["charge"] == b["charge"]
            and a["aromatic"] == b["aromatic"]
        )

    def edge_match(a, b):
        # must match bond-vs-map edges; bonds also match order
        if a["kind"] != b["kind"]:
            return False
        if a["kind"] == "bond":
            return a["order"] == b["order"]
        return True  # kind == "map"

    return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=edge_match)


def are_reactions_identical(rxn1, rxn2):
    canonical_rxn1 = canonicalize_reaction_smiles(rxn1, remove_mapping=False)
    canonical_rxn2 = canonicalize_reaction_smiles(rxn2, remove_mapping=False)
    return canonical_rxn1 == canonical_rxn2


gold_reactions = []
with open(
    "/home/csnbritt/projects/denovochem_projects/agave_chem/scripts/benchmarks/gold_reactions_filtered.txt",
    "r",
) as file:
    for line in file:
        gold_reactions.append(line.strip())

mapped_count = 0
identical_count = 0
total_start = time.time()
incorrect_smirks = []
correct_smirks = []

for i, gold_reaction in enumerate(gold_reactions):
    identical = None

    rxn_start = time.time()
    reactants = gold_reaction.split(">>")[0]
    products = gold_reaction.split(">>")[1]
    products_list = products.split(".")
    sorted_products_list = sorted(products_list, key=len)
    biggest_product = sorted_products_list[-1]
    better_reaction = canonicalize_reaction_smiles(
        reactants + ">>" + biggest_product,
        remove_mapping=False,
        canonicalize_tautomer=True,
    )
    unmapped_better_reaction = canonicalize_reaction_smiles(
        better_reaction, remove_mapping=True
    )
    try:
        out = expert_mapper.map_reaction(unmapped_better_reaction)
    except Exception:
        print("oops")
    if time.time() - rxn_start > 5:
        print("SLOW:", time.time() - rxn_start)
    print(i, mapped_count, identical_count, identical)
print(time.time() - total_start)
