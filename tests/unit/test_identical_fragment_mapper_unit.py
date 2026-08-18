import re

from rdkit import Chem

from agave_chem.mappers.identical_fragments.identical_fragment_mapper import (
    IdenticalFragmentMapper,
)


def _get_atom_map_nums(smiles: str) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return [a.GetAtomMapNum() for a in mol.GetAtoms()]


def _split_rxn(rxn_smiles: str) -> tuple[list[str], list[str]]:
    reactants, products = rxn_smiles.split(">>")
    reactant_frags = [f for f in reactants.split(".") if f]
    product_frags = [f for f in products.split(".") if f]
    return reactant_frags, product_frags


def test_atom_map_identical_fragments_strips_and_maps_fragment():
    mapper = IdenticalFragmentMapper(mapper_name="test")

    # 'Cl' is identical on both sides and should be removed from both sides,
    # returned as atom-mapped fragment.
    mapped_frags, remaining_rxn = mapper._atom_map_identical_fragments("CCC.Cl>>CC.Cl")

    assert remaining_rxn == "CCC>>CC"
    assert len(mapped_frags) == 1

    # mapped_frags[0] is a (reactant_smiles, product_smiles) tuple.
    # For truly identical fragments, both elements are the same.
    assert mapped_frags[0][0] == mapped_frags[0][1]

    # Should have at least one atom map number (starts at 500 in implementation).
    nums = _get_atom_map_nums(mapped_frags[0][0])
    assert all(n >= 500 for n in nums)
    assert _get_atom_map_nums(mapped_frags[0][0]) == _get_atom_map_nums(
        mapped_frags[0][1]
    )


def test_create_and_resolve_identical_fragments_mapping_list_roundtrip():
    mapper = IdenticalFragmentMapper(mapper_name="test")

    rxns = [
        "CCC.Cl>>CC.Cl",
        "CO>>CO",  # entire molecule identical
        "CC>>CO",  # no identical fragments
        "CC.Cl>>CC.Cl",
    ]

    stripped_rxns, mapping_list = mapper.create_identical_fragments_mapping_list(rxns)

    assert stripped_rxns[0] == "CCC>>CC"
    assert stripped_rxns[1] == ">>"  # both sides removed
    assert stripped_rxns[2] == "CC>>CO"
    assert stripped_rxns[3] == ">>"

    resolved = mapper.resolve_identical_fragments_mapping_list(
        stripped_rxns, mapping_list
    )
    assert len(resolved) == 4

    # Ensure identical fragments are restored to both sides.
    r0_react, r0_prod = _split_rxn(resolved[0])
    assert len(r0_react) == 2
    assert len(r0_prod) == 2

    # One fragment should be mapped and present on both sides.
    mapped_candidates = [f for f in r0_react if re.search(r":\d+\]", f)]
    assert len(mapped_candidates) == 1
    assert mapped_candidates[0] in r0_prod


def test_map_reaction_invalid_smiles_returns_default():
    mapper = IdenticalFragmentMapper(mapper_name="test")

    # invalid because reaction_smiles.count('>>') != 1
    res = mapper.map_reaction("CC")

    assert res.original_smiles == "CC"
    assert res.selected_mapping == ""
    assert res.mapping_type == "identical_fragment"


def test_map_reaction_no_identical_fragment_returns_input_unchanged():
    mapper = IdenticalFragmentMapper(mapper_name="test")

    rxn = "CC>>CO"
    res = mapper.map_reaction(rxn)

    assert res.original_smiles == rxn
    assert res.selected_mapping == rxn


def test_map_reaction_with_identical_fragment_adds_atom_mapping():
    mapper = IdenticalFragmentMapper(mapper_name="test")

    rxn = "CCC.Cl>>CC.Cl"
    res = mapper.map_reaction(rxn)

    assert res.original_smiles == rxn

    mapped = res.selected_mapping
    reactants, products = _split_rxn(mapped)

    # Both sides should contain 2 fragments, including a mapped fragment.
    assert len(reactants) == 2
    assert len(products) == 2

    mapped_reactants = [f for f in reactants if re.search(r":\d+\]", f)]
    mapped_products = [f for f in products if re.search(r":\d+\]", f)]

    assert len(mapped_reactants) == 1
    assert len(mapped_products) == 1
    assert mapped_reactants[0] == mapped_products[0]


def test_atom_map_identical_fragments_stoichiometry_mismatch():
    """Regression: fragments with unequal counts on both sides should only
    be paired up to the minimum count."""
    mapper = IdenticalFragmentMapper(mapper_name="test")

    # 3 [K+] reactants, 1 [K+] product
    rxn = "F.[K+].[K+].[K+].CC>>CCC.[K+]"
    mapped_frags, remaining_rxn = mapper._atom_map_identical_fragments(rxn)

    # Only 1 copy should be paired and mapped.
    assert len(mapped_frags) == 1
    assert _get_atom_map_nums(mapped_frags[0][0]) == [500]
    assert _get_atom_map_nums(mapped_frags[0][1]) == [500]

    # Remaining reaction should have 2 unmapped [K+] on reactants, 0 on products.
    r_frags, p_frags = _split_rxn(remaining_rxn)
    assert r_frags.count("[K+]") == 2
    assert "[K+]" not in p_frags


def test_atom_map_charge_different_pair_n_vs_nh4():
    """N on reactants and [NH4+] on products should be detected as identical."""
    mapper = IdenticalFragmentMapper(mapper_name="test")

    mapped_frags, remaining_rxn = mapper._atom_map_identical_fragments(
        "CCC.N>>CCO.[NH4+]"
    )

    # CCC and CCO are not identical, so only N/[NH4+] should match
    assert len(mapped_frags) == 1
    # The mapped forms should differ (N vs [NH4+])
    assert mapped_frags[0][0] != mapped_frags[0][1]
    # Both should have atom map numbers
    r_nums = _get_atom_map_nums(mapped_frags[0][0])
    p_nums = _get_atom_map_nums(mapped_frags[0][1])
    assert all(n >= 500 for n in r_nums)
    assert all(n >= 500 for n in p_nums)
    # Corresponding atoms should have the same map numbers
    assert r_nums == p_nums
    # Remaining should have the non-identical fragments on both sides
    r_frags, p_frags = _split_rxn(remaining_rxn)
    assert r_frags == ["CCC"]
    assert p_frags == ["CCO"]


def test_atom_map_charge_different_pair_phosphoric_acid():
    """O=P(O)(O)O vs O=P([O-])(O)O should match with correct atom correspondence."""
    mapper = IdenticalFragmentMapper(mapper_name="test")

    mapped_frags, _remaining_rxn = mapper._atom_map_identical_fragments(
        "CCC.O=P(O)(O)O>>CCO.O=P([O-])(O)O"
    )

    assert len(mapped_frags) == 1
    r_nums = _get_atom_map_nums(mapped_frags[0][0])
    p_nums = _get_atom_map_nums(mapped_frags[0][1])
    assert len(r_nums) == len(p_nums)
    assert r_nums == p_nums
    assert all(n >= 500 for n in r_nums)


def test_atom_map_charge_different_anion_to_neutral():
    """[O-] on one side and OH on the other should match (net charge difference)."""
    mapper = IdenticalFragmentMapper(mapper_name="test")

    mapped_frags, _remaining_rxn = mapper._atom_map_identical_fragments(
        "CCC.[O-]C>>CCO.OC"
    )

    assert len(mapped_frags) == 1
    assert mapped_frags[0][0] != mapped_frags[0][1]
    r_nums = _get_atom_map_nums(mapped_frags[0][0])
    p_nums = _get_atom_map_nums(mapped_frags[0][1])
    assert len(r_nums) == len(p_nums)
    assert r_nums == p_nums


def test_atom_map_zwitterion_not_matched():
    """Zwitterions with net zero charge are not neutralized by Uncharger, so
    a zwitterion and its protonated form should NOT be detected as identical.

    This is a known limitation of the Uncharger-based approach: it only
    neutralizes net charge, not individual formal charges within a molecule.
    """
    mapper = IdenticalFragmentMapper(mapper_name="test")

    mapped_frags, remaining_rxn = mapper._atom_map_identical_fragments(
        "CCC.CC[N+](C)(C)CCCC(=O)[O-]>>CCO.CC[N+](C)(C)CCCC(=O)O"
    )

    # The zwitterion (net 0) and the protonated form (net +1) should NOT match
    # because the Uncharger only acts on net charge.
    assert len(mapped_frags) == 0
    r_frags, p_frags = _split_rxn(remaining_rxn)
    assert "CC[N+](C)(C)CCCC(=O)[O-]" in r_frags
    assert "CC[N+](C)(C)CCCC(=O)O" in p_frags


def test_atom_map_canonical_matching_different_smiles_order():
    """Same molecule written with different SMILES ordering should match."""
    mapper = IdenticalFragmentMapper(mapper_name="test")

    # O=C(O)C and CC(=O)O are the same molecule (acetic acid)
    mapped_frags, remaining_rxn = mapper._atom_map_identical_fragments(
        "CCC.O=C(O)C>>CC.CC(=O)O"
    )

    assert len(mapped_frags) == 1
    # For identical fragments (same canonical form), both elements should match
    assert mapped_frags[0][0] == mapped_frags[0][1]
    r_frags, p_frags = _split_rxn(remaining_rxn)
    assert r_frags == ["CCC"]
    assert p_frags == ["CC"]


def test_atom_map_duplicate_identical_fragments():
    """Multiple copies of the same fragment should all be paired."""
    mapper = IdenticalFragmentMapper(mapper_name="test")

    # 2 Cl on each side should produce 2 pairs
    mapped_frags, remaining_rxn = mapper._atom_map_identical_fragments(
        "CCC.Cl.Cl>>CC.Cl.Cl"
    )

    assert len(mapped_frags) == 2
    r_frags, p_frags = _split_rxn(remaining_rxn)
    assert r_frags == ["CCC"]
    assert p_frags == ["CC"]


def test_atom_map_duplicate_charge_different_fragments():
    """Multiple charge-different pairs should all be matched."""
    mapper = IdenticalFragmentMapper(mapper_name="test")

    # 2 N on reactants, 2 [NH4+] on products (CCC/CCO are not identical)
    mapped_frags, remaining_rxn = mapper._atom_map_identical_fragments(
        "CCC.N.N>>CCO.[NH4+].[NH4+]"
    )

    assert len(mapped_frags) == 2
    # Both pairs should have differing reactant/product forms
    for pair in mapped_frags:
        assert pair[0] != pair[1]
    r_frags, p_frags = _split_rxn(remaining_rxn)
    assert r_frags == ["CCC"]
    assert p_frags == ["CCO"]


def test_atom_map_mixed_identical_and_charge_different():
    """A reaction with both exact-match and charge-different fragments."""
    mapper = IdenticalFragmentMapper(mapper_name="test")

    # Cl is exact match, N/[NH4+] is charge-different (CCC/CCO not identical)
    mapped_frags, remaining_rxn = mapper._atom_map_identical_fragments(
        "CCC.Cl.N>>CCO.Cl.[NH4+]"
    )

    assert len(mapped_frags) == 2
    identical_pairs = [p for p in mapped_frags if p[0] == p[1]]
    charge_pairs = [p for p in mapped_frags if p[0] != p[1]]
    assert len(identical_pairs) == 1
    assert len(charge_pairs) == 1

    r_frags, p_frags = _split_rxn(remaining_rxn)
    assert r_frags == ["CCC"]
    assert p_frags == ["CCO"]


def test_resolve_charge_different_pair_preserves_charge():
    """After resolve, reactant side has neutral form, product side has charged form."""
    mapper = IdenticalFragmentMapper(mapper_name="test")

    rxn = "CCC.N>>CCO.[NH4+]"
    stripped_rxns, mapping_list = mapper.create_identical_fragments_mapping_list([rxn])
    resolved = mapper.resolve_identical_fragments_mapping_list(
        stripped_rxns, mapping_list
    )

    r_frags, p_frags = _split_rxn(resolved[0])
    # Both sides should have 2 fragments (non-identical + mapped identical)
    assert len(r_frags) == 2
    assert len(p_frags) == 2
    # Reactant side should have a mapped N (not [NH4+])
    r_mapped = [f for f in r_frags if re.search(r":\d+\]", f)]
    assert len(r_mapped) == 1
    assert "[NH4+]" not in r_mapped[0]
    # Product side should have a mapped [NH4+]
    p_mapped = [f for f in p_frags if re.search(r":\d+\]", f)]
    assert len(p_mapped) == 1
    assert "[NH4+" in p_mapped[0]
    # The mapped forms should differ
    assert r_mapped[0] != p_mapped[0]


def test_map_reaction_with_charge_different_fragment():
    """End-to-end test through map_reaction with a charge-different fragment."""
    mapper = IdenticalFragmentMapper(mapper_name="test")

    rxn = "CCC.N>>CCO.[NH4+]"
    res = mapper.map_reaction(rxn)

    assert res.original_smiles == rxn
    assert res.selected_mapping != ""

    reactants, products = _split_rxn(res.selected_mapping)
    assert len(reactants) == 2
    assert len(products) == 2

    mapped_reactants = [f for f in reactants if re.search(r":\d+\]", f)]
    mapped_products = [f for f in products if re.search(r":\d+\]", f)]
    assert len(mapped_reactants) == 1
    assert len(mapped_products) == 1
    # Reactant has N, product has [NH4+]
    assert "[NH4+]" not in mapped_reactants[0]
    assert "[NH4+" in mapped_products[0]
