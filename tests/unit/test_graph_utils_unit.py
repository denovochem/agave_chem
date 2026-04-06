import networkx as nx

from agave_chem.utils.graph_utils import mapping_equivalent, rxn_to_mapping_graph


def _count_edges_by_kind(G: nx.Graph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        kind = data.get("kind")
        assert kind is not None
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def test_rxn_to_mapping_graph_builds_expected_nodes_and_edges_for_identity_reaction():
    rxn = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
    G = rxn_to_mapping_graph(rxn)

    # 3 atoms on each side
    assert G.number_of_nodes() == 6

    # 2 bonds on each side (ethanol backbone), plus 3 mapping edges
    counts = _count_edges_by_kind(G)
    assert counts["bond"] == 4
    assert counts["map"] == 3

    # Validate node attributes exist and sides are only R/P
    for node, data in G.nodes(data=True):
        assert node[0] in {"R", "P"}
        assert data["side"] in {"R", "P"}
        assert data["side"] == node[0]

        for key in [
            "Z",
            "charge",
            "aromatic",
            "in_ring",
            "hydrogen_count",
            "degree",
            "chiral_tag",
        ]:
            assert key in data

    # Validate bond edges have order/stereo and map edges do not require them
    for _, _, data in G.edges(data=True):
        if data["kind"] == "bond":
            assert "order" in data
            assert "stereo" in data
        else:
            assert data["kind"] == "map"


def test_rxn_to_mapping_graph_does_not_add_map_edges_when_product_has_no_mapping_nums():
    # Reactant has an atom-map number, but the product does not.
    rxn = "[CH3:1]Cl>>Cl"
    G = rxn_to_mapping_graph(rxn)

    counts = _count_edges_by_kind(G)
    assert counts.get("map", 0) == 0


def test_mapping_equivalent_true_for_same_reaction_with_relabelled_atom_maps():
    rxn1 = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
    rxn2 = "[CH3:10][CH2:20][OH:30]>>[CH3:10][CH2:20][OH:30]"

    assert mapping_equivalent(rxn1, rxn2) is True


def test_mapping_equivalent_false_when_mapping_connects_different_atom_types():
    # Product mapping swaps methyl and methylene relative to reactant.
    rxn_good = "[CH3:1][CH2:2]O>>[CH3:1][CH2:2]O"
    rxn_bad = "[CH3:1][CH2:2]O>>[CH3:2][CH2:1]O"

    assert mapping_equivalent(rxn_good, rxn_bad) is False
