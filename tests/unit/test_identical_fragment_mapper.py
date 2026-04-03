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

    # Should have at least one atom map number (starts at 500 in implementation).
    nums = _get_atom_map_nums(mapped_frags[0])
    assert all(n >= 500 for n in nums)


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

    assert res["original_smiles"] == "CC"
    assert res["selected_mapping"] == ""
    assert res["mapping_type"] == "identical_fragment"


def test_map_reaction_no_identical_fragment_returns_input_unchanged():
    mapper = IdenticalFragmentMapper(mapper_name="test")

    rxn = "CC>>CO"
    res = mapper.map_reaction(rxn)

    assert res["original_smiles"] == rxn
    assert res["selected_mapping"] == rxn


def test_map_reaction_with_identical_fragment_adds_atom_mapping():
    mapper = IdenticalFragmentMapper(mapper_name="test")

    rxn = "CCC.Cl>>CC.Cl"
    res = mapper.map_reaction(rxn)

    assert res["original_smiles"] == rxn

    mapped = res["selected_mapping"]
    reactants, products = _split_rxn(mapped)

    # Both sides should contain 2 fragments, including a mapped fragment.
    assert len(reactants) == 2
    assert len(products) == 2

    mapped_reactants = [f for f in reactants if re.search(r":\d+\]", f)]
    mapped_products = [f for f in products if re.search(r":\d+\]", f)]

    assert len(mapped_reactants) == 1
    assert len(mapped_products) == 1
    assert mapped_reactants[0] == mapped_products[0]
