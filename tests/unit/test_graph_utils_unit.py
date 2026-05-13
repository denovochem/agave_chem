import networkx as nx
import pytest

from agave_chem.utils.graph_utils import (
    _enumerate_resonance_swap_variants,
    _enumerate_tautomeric_rxn_smiles,
    mapping_equivalent,
    normalize_rxn_atom_maps,
    rxn_to_mapping_graph,
)


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

    assert (
        mapping_equivalent(
            rxn1, rxn2, consider_tautomers=True, consider_resonance_swaps=True
        )
        is True
    )


def test_mapping_equivalent_false_when_mapping_connects_different_atom_types():
    # Product mapping swaps methyl and methylene relative to reactant.
    rxn_good = "[CH3:1][CH2:2]O>>[CH3:1][CH2:2]O"
    rxn_bad = "[CH3:1][CH2:2]O>>[CH3:2][CH2:1]O"

    assert (
        mapping_equivalent(
            rxn_good, rxn_bad, consider_tautomers=False, consider_resonance_swaps=False
        )
        is False
    )


# ---------------------------------------------------------------------------
# Tautomer equivalence
# ---------------------------------------------------------------------------


def test_mapping_equivalent_true_for_tautomeric_pyrazole_nitrogen_mapping():
    # 1H-pyrazole: rxn1 maps the N-H nitrogen; rxn2 maps the other N.
    # Both are equivalent due to tautomerism.
    rxn1 = "[nH:1]1ccn[n:2]1>>[nH:1]1ccn[n:2]1"
    rxn2 = "n1cc[nH:2][n:1]1>>n1cc[nH:2][n:1]1"

    assert (
        mapping_equivalent(
            rxn1, rxn2, consider_tautomers=True, consider_resonance_swaps=False
        )
        is True
    )


def test_mapping_equivalent_false_for_truly_distinct_atoms_in_ring():
    # Map the nitrogen vs a carbon in the ring — not tautomerically equivalent.
    rxn1 = "[nH:1]1ccnn1>>[nH:1]1ccnn1"
    rxn2 = "[nH]1[c:1]cnn1>>[nH]1[c:1]cnn1"

    assert (
        mapping_equivalent(
            rxn1, rxn2, consider_tautomers=True, consider_resonance_swaps=True
        )
        is False
    )


def test_enumerate_tautomeric_rxn_smiles_includes_original():
    rxn = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
    variants = _enumerate_tautomeric_rxn_smiles(rxn)

    assert len(variants) >= 1


def test_enumerate_tautomeric_rxn_smiles_produces_multiple_forms_for_tautomerisable_mol():
    # Acetylacetone (keto-enol) as both reactant and product — should yield > 1 variant.
    rxn = "[CH3:1]C(=O)CC(=O)[CH3:2]>>[CH3:1]C(=O)CC(=O)[CH3:2]"
    variants = _enumerate_tautomeric_rxn_smiles(rxn)

    assert len(variants) > 1


# ---------------------------------------------------------------------------
# Resonance-swap equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rxn1, rxn2",
    [
        (
            "[CH2:1]c1ccc([N+](=[O:2])[O-])cc1>>[CH2:1]c1ccc([N+](=O)[O-])cc1",
            "[CH2:1]c1ccc([N+]([O-])=[O:2])cc1>>[CH2:1]c1ccc([N+](=O)[O-])cc1",
        ),
        # (
        #     "OC(=[O:1])c1ccccc1>>OC(=[O:1])c1ccccc1",
        #     "[O:1]C(=O)c1ccccc1>>[O:1]C(=O)c1ccccc1",
        # ),
    ],
    ids=["nitro_oxygen_swap"],
)
def test_mapping_equivalent_true_for_resonance_symmetric_oxygen(rxn1, rxn2):
    assert (
        mapping_equivalent(
            rxn1, rxn2, consider_tautomers=False, consider_resonance_swaps=True
        )
        is True
    )


def test_enumerate_resonance_swap_variants_returns_original_when_no_mapped_resonance_atoms():
    # No mapped atoms in the nitro group → no swap candidates → single variant.
    rxn = "[CH2:1]c1ccc([N+](=O)[O-])cc1>>[CH2:1]c1ccc([N+](=O)[O-])cc1"
    variants = _enumerate_resonance_swap_variants(rxn)

    assert len(variants) == 1


def test_enumerate_resonance_swap_variants_returns_two_variants_for_one_mapped_nitro_oxygen():
    # One nitro oxygen is mapped → one swap candidate → 2 variants (swap on / off).
    rxn = "[CH2:1]c1ccc([N+](=[O:2])[O-])cc1>>[CH2:1]c1ccc([N+](=O)[O-])cc1"
    variants = _enumerate_resonance_swap_variants(rxn)

    assert len(variants) == 2


# ---------------------------------------------------------------------------
# consider_tautomers / consider_resonance_swaps flags
# ---------------------------------------------------------------------------


def test_mapping_equivalent_flags_both_false_falls_back_to_direct_compare_only():
    # Identical reactions → direct compare succeeds regardless of flags.
    rxn = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
    assert (
        mapping_equivalent(
            rxn, rxn, consider_tautomers=False, consider_resonance_swaps=False
        )
        is True
    )


def test_mapping_equivalent_tautomers_false_misses_tautomeric_equivalence():
    # Formamidine tautomers: the mapped atom switches between NH2 (H=2, degree=1)
    # and =NH (H=1, degree=2), so the direct graph comparison genuinely fails.
    # Only tautomer enumeration finds the match.
    rxn1 = "[NH2:1]C=N>>[NH2:1]C=N"
    rxn2 = "NC=[NH:1]>>NC=[NH:1]"
    assert (
        mapping_equivalent(
            rxn1, rxn2, consider_tautomers=False, consider_resonance_swaps=False
        )
        is False
    )


def test_mapping_equivalent_resonance_swaps_false_misses_nitro_swap():
    # rxn1 maps the neutral =O (charge=0) on both sides; rxn2 maps the charged
    # [O-] (charge=-1) on both sides.  The map edge differs in node charge so the
    # direct comparison fails.  Only the resonance-swap pattern converts rxn2.
    rxn1 = "[CH2:1]c1ccc([N+](=[O:2])[O-])cc1>>[CH2:1]c1ccc([N+](=[O:2])[O-])cc1"
    rxn2 = "[CH2:1]c1ccc([N+](=O)[O-:2])cc1>>[CH2:1]c1ccc([N+](=O)[O-:2])cc1"
    assert (
        mapping_equivalent(
            rxn1, rxn2, consider_tautomers=False, consider_resonance_swaps=False
        )
        is False
    )


# ---------------------------------------------------------------------------
# normalize_rxn_atom_maps
# ---------------------------------------------------------------------------


def test_normalize_rxn_atom_maps_clears_reactant_only_map_numbers():
    # Map 2 appears only in the reactant (leaving group) — should be zeroed.
    rxn = "[CH3:1][CH2:2]Cl>>[CH3:1]Br"
    normalized = normalize_rxn_atom_maps(rxn)
    from rdkit.Chem import AllChem

    rxn_obj = AllChem.ReactionFromSmarts(normalized, useSmiles=True)
    r_mol = rxn_obj.GetReactantTemplate(0)
    map_nums = {atom.GetAtomMapNum() for atom in r_mol.GetAtoms()}
    assert 2 not in map_nums
    assert 1 in map_nums


def test_normalize_rxn_atom_maps_preserves_shared_map_numbers():
    # Map 1 appears on both sides — must not be cleared.
    rxn = "[CH3:1][OH:2]>>[CH3:1]Br"
    normalized = normalize_rxn_atom_maps(rxn)
    from rdkit.Chem import AllChem

    rxn_obj = AllChem.ReactionFromSmarts(normalized, useSmiles=True)
    r_mol = rxn_obj.GetReactantTemplate(0)
    map_nums = {atom.GetAtomMapNum() for atom in r_mol.GetAtoms()}
    assert 1 in map_nums
    assert 2 not in map_nums


def test_normalize_rxn_atom_maps_no_change_when_all_maps_are_shared():
    # All mapped reactant atoms appear in the product — nothing should change.
    rxn = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
    normalized = normalize_rxn_atom_maps(rxn)
    from rdkit.Chem import AllChem

    rxn_obj = AllChem.ReactionFromSmarts(normalized, useSmiles=True)
    r_mol = rxn_obj.GetReactantTemplate(0)
    map_nums = {
        atom.GetAtomMapNum() for atom in r_mol.GetAtoms() if atom.GetAtomMapNum()
    }
    assert map_nums == {1, 2, 3}


# ---------------------------------------------------------------------------
# Chirality invariance (CIP code vs raw chiral tag)
# ---------------------------------------------------------------------------


def test_mapping_equivalent_true_for_same_chirality_different_smiles_atom_order():
    # The same chiral center (mapped C:2) written with substituents in a different
    # SMILES branch order.  The raw CW/CCW chiral tag differs between the two
    # representations, but the CIP code is the same in both, so the reactions
    # must be recognised as equivalent.
    rxn1 = "[CH3:1][C@@H:2](F)Cl>>[CH3:1][C@@H:2](F)Cl"
    rxn2 = "[CH3:1][C@H:2](Cl)F>>[CH3:1][C@H:2](Cl)F"
    assert mapping_equivalent(rxn1, rxn2) is True


def test_mapping_equivalent_false_for_enantiomeric_reactions():
    # Same SMILES atom order, only @ vs @@ differs → opposite CIP codes → not equivalent.
    rxn1 = "[CH3:1][C@@H:2](F)Cl>>[CH3:1][C@@H:2](F)Cl"
    rxn2 = "[CH3:1][C@H:2](F)Cl>>[CH3:1][C@H:2](F)Cl"
    assert mapping_equivalent(rxn1, rxn2) is False
