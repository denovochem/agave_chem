from rdkit import Chem

from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper


def _split_rxn(rxn_smiles: str) -> tuple[list[str], list[str]]:
    reactants, products = rxn_smiles.split(">>")
    reactant_frags = [f for f in reactants.split(".") if f]
    product_frags = [f for f in products.split(".") if f]
    return reactant_frags, product_frags


def _get_atom_map_nums(smiles: str) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return [a.GetAtomMapNum() for a in mol.GetAtoms()]


def test_map_reaction_invalid_smiles_returns_default():
    mapper = MCSReactionMapper(mapper_name="test")

    # invalid because reaction_smiles.count('>>') != 1
    res = mapper.map_reaction("CC")

    assert res["original_smiles"] == "CC"
    assert res["selected_mapping"] == ""
    assert res["mapping_type"] == "mcs"


def test_map_reaction_identity_reaction_produces_valid_mapping():
    mapper = MCSReactionMapper(mapper_name="test")

    rxn = "CCCCCO>>CCCCCO"
    res = mapper.map_reaction(rxn)

    assert res["original_smiles"] == rxn
    mapped = res["selected_mapping"]
    assert mapped.count(">>") == 1

    reactants, products = _split_rxn(mapped)
    assert len(reactants) == 1
    assert len(products) == 1

    # Should be parseable by RDKit.
    assert Chem.MolFromSmiles(reactants[0]) is not None
    assert Chem.MolFromSmiles(products[0]) is not None

    # Expect at least one atom mapping number to have been assigned.
    prod_nums = _get_atom_map_nums(products[0])
    assert any(n != 0 for n in prod_nums)

    # Basic sanity: non-zero map numbers in the product should be unique.
    prod_nonzero = [n for n in prod_nums if n != 0]
    assert len(prod_nonzero) == len(set(prod_nonzero))


def test_map_reaction_is_deterministic_for_same_input():
    mapper = MCSReactionMapper(mapper_name="test")

    rxn = "CCCCCO.O>>CCCCCO.O"
    res1 = mapper.map_reaction(rxn)
    res2 = mapper.map_reaction(rxn)

    assert res1["original_smiles"] == rxn
    assert res2["original_smiles"] == rxn
    assert res1["selected_mapping"] == res2["selected_mapping"]


def test_map_reactions_returns_results_in_same_order():
    mapper = MCSReactionMapper(mapper_name="test")

    rxns = [
        "CCCCCO>>CCCCCO",
        "CC",  # invalid
        "CCCCCO.O>>CCCCCO.O",
    ]

    results = mapper.map_reactions(rxns)

    assert len(results) == len(rxns)

    assert results[0]["original_smiles"] == rxns[0]
    assert results[1]["original_smiles"] == rxns[1]
    assert results[2]["original_smiles"] == rxns[2]
