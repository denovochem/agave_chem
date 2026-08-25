"""Regression tests verifying that the replicated graph-comparison code in
``workflows/benchmarking/_bench_utils.py`` stays in sync with the original
implementations in ``agave_chem/utils/graph_utils.py``.

If either copy is modified without updating the other, these tests will fail.
"""

import sys
from pathlib import Path

import networkx as nx
import pytest

# _bench_utils.py lives outside the agave_chem package under workflows/.
# Add its parent to sys.path so we can import it directly.
_bench_dir = (
    Path(__file__).resolve().parent.parent.parent / "workflows" / "benchmarking"
)
sys.path.insert(0, str(_bench_dir))

from _bench_utils import (
    mapping_equivalent as bench_mapping_equivalent,
)
from _bench_utils import (
    normalize_rxn_atom_maps as bench_normalize_rxn_atom_maps,
)
from _bench_utils import (
    rxn_to_mapping_graph as bench_rxn_to_mapping_graph,
)

from agave_chem.utils.graph_utils import (
    mapping_equivalent as orig_mapping_equivalent,
)
from agave_chem.utils.graph_utils import (
    normalize_rxn_atom_maps as orig_normalize_rxn_atom_maps,
)
from agave_chem.utils.graph_utils import (
    rxn_to_mapping_graph as orig_rxn_to_mapping_graph,
)

# ---------------------------------------------------------------------------
# Test reactions covering various equivalence scenarios
# ---------------------------------------------------------------------------

_TEST_REACTIONS = [
    "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]",
    "[CH3:10][CH2:20][OH:30]>>[CH3:10][CH2:20][OH:30]",
    "[nH:1]1ccn[n:2]1>>[nH:1]1ccn[n:2]1",
    "n1cc[nH:2][n:1]1>>n1cc[nH:2][n:1]1",
    "[CH2:1]c1ccc([N+](=[O:2])[O-])cc1>>[CH2:1]c1ccc([N+](=O)[O-])cc1",
    "[CH2:1]c1ccc([N+](=O)[O-:2])cc1>>[CH2:1]c1ccc([N+](=O)[O-:2])cc1",
    "[NH2:1]C=N>>[NH2:1]C=N",
    "NC=[NH:1]>>NC=[NH:1]",
    "[CH3:1][CH2:2]Cl>>[CH3:1]Br",
    "[CH3:1][C@@H:2](F)Cl>>[CH3:1][C@@H:2](F)Cl",
    "[CH3:1][C@H:2](Cl)F>>[CH3:1][C@H:2](Cl)F",
    "[CH3:1][C@H:2](F)Cl>>[CH3:1][C@H:2](F)Cl",
]


# ---------------------------------------------------------------------------
# rxn_to_mapping_graph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rxn", _TEST_REACTIONS)
def test_rxn_to_mapping_graph_produces_isomorphic_graphs(rxn: str):
    """The replicated and original rxn_to_mapping_graph must produce isomorphic graphs."""
    G_orig = orig_rxn_to_mapping_graph(rxn)
    G_bench = bench_rxn_to_mapping_graph(rxn)

    def node_match(a, b):
        return a == b

    def edge_match(a, b):
        return a == b

    assert nx.is_isomorphic(
        G_orig, G_bench, node_match=node_match, edge_match=edge_match
    )


@pytest.mark.parametrize("rxn", _TEST_REACTIONS)
def test_rxn_to_mapping_graph_identical_node_attributes(rxn: str):
    """Every node in the replicated graph must have identical attributes to the original."""
    G_orig = orig_rxn_to_mapping_graph(rxn)
    G_bench = bench_rxn_to_mapping_graph(rxn)

    orig_attrs = {n: dict(G_orig.nodes[n]) for n in G_orig.nodes}
    bench_attrs = {n: dict(G_bench.nodes[n]) for n in G_bench.nodes}

    assert len(orig_attrs) == len(bench_attrs)
    for node_id, orig_val in orig_attrs.items():
        assert orig_val == bench_attrs[node_id], (
            f"Node {node_id} attributes differ: {orig_val} vs {bench_attrs[node_id]}"
        )


# ---------------------------------------------------------------------------
# normalize_rxn_atom_maps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rxn", _TEST_REACTIONS)
def test_normalize_rxn_atom_maps_produces_identical_results(rxn: str):
    """The replicated and original normalize_rxn_atom_maps must produce identical strings."""
    assert orig_normalize_rxn_atom_maps(rxn) == bench_normalize_rxn_atom_maps(rxn)


# ---------------------------------------------------------------------------
# mapping_equivalent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rxn1, rxn2, consider_tautomers, consider_resonance_swaps",
    [
        # Same reaction, relabelled maps
        (
            "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]",
            "[CH3:10][CH2:20][OH:30]>>[CH3:10][CH2:20][OH:30]",
            True,
            True,
        ),
        # Different atom types mapped
        (
            "[CH3:1][CH2:2]O>>[CH3:1][CH2:2]O",
            "[CH3:1][CH2:2]O>>[CH3:2][CH2:1]O",
            False,
            False,
        ),
        # Tautomeric equivalence
        (
            "[nH:1]1ccn[n:2]1>>[nH:1]1ccn[n:2]1",
            "n1cc[nH:2][n:1]1>>n1cc[nH:2][n:1]1",
            True,
            False,
        ),
        # Resonance swap equivalence
        (
            "[CH2:1]c1ccc([N+](=[O:2])[O-])cc1>>[CH2:1]c1ccc([N+](=O)[O-])cc1",
            "[CH2:1]c1ccc([N+]([O-])=[O:2])cc1>>[CH2:1]c1ccc([N+](=O)[O-])cc1",
            False,
            True,
        ),
        # Chirality same
        (
            "[CH3:1][C@@H:2](F)Cl>>[CH3:1][C@@H:2](F)Cl",
            "[CH3:1][C@H:2](Cl)F>>[CH3:1][C@H:2](Cl)F",
            False,
            False,
        ),
        # Chirality different (enantiomers)
        (
            "[CH3:1][C@@H:2](F)Cl>>[CH3:1][C@@H:2](F)Cl",
            "[CH3:1][C@H:2](F)Cl>>[CH3:1][C@H:2](F)Cl",
            False,
            False,
        ),
    ],
)
def test_mapping_equivalent_identical_results(
    rxn1: str,
    rxn2: str,
    consider_tautomers: bool,
    consider_resonance_swaps: bool,
):
    """The replicated and original mapping_equivalent must return identical booleans."""
    result_orig = orig_mapping_equivalent(
        rxn1,
        rxn2,
        consider_tautomers=consider_tautomers,
        consider_resonance_swaps=consider_resonance_swaps,
    )
    result_bench = bench_mapping_equivalent(
        rxn1,
        rxn2,
        consider_tautomers=consider_tautomers,
        consider_resonance_swaps=consider_resonance_swaps,
    )
    assert result_orig == result_bench, (
        f"mapping_equivalent({rxn1!r}, {rxn2!r}, "
        f"consider_tautomers={consider_tautomers}, "
        f"consider_resonance_swaps={consider_resonance_swaps}) "
        f"differs: orig={result_orig}, bench={result_bench}"
    )
