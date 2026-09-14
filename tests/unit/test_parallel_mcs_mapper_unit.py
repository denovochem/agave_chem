"""Unit tests for ParallelMCSReactionMapper."""

import pytest
from rdkit import Chem

from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper
from agave_chem.mappers.mcs.parallel_mcs_mapper import ParallelMCSReactionMapper


def _split_rxn(rxn_smiles: str) -> tuple[list[str], list[str]]:
    reactants, products = rxn_smiles.split(">>")
    reactant_frags = [f for f in reactants.split(".") if f]
    product_frags = [f for f in products.split(".") if f]
    return reactant_frags, product_frags


def _get_atom_map_nums(smiles: str) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return [a.GetAtomMapNum() for a in mol.GetAtoms()]


TEST_REACTIONS = [
    "CCCCCO>>CCCCCO",
    "CCCCCO.O>>CCCCCO.O",
    "F.Nc1nc(Br)c(Br)cc1Br.O=N[O-].[Na+].c1ccncc1>>Fc1nc(Br)c(Br)cc1Br",
]

INVALID_REACTION = "CC"


def test_map_reaction_single_returns_valid_result():
    """map_reaction delegates to a lazy in-process MCSReactionMapper."""
    mapper = ParallelMCSReactionMapper("test_mcs_par", workers=2)

    rxn = "CCCCCO>>CCCCCO"
    res = mapper.map_reaction(rxn)

    assert res.original_smiles == rxn
    assert res.selected_mapping
    assert res.mapping_type == "mcs"


def test_map_reaction_invalid_smiles_returns_default():
    """Invalid SMILES via map_reaction returns a default empty result."""
    mapper = ParallelMCSReactionMapper("test_mcs_par", workers=2)

    res = mapper.map_reaction(INVALID_REACTION)

    assert res.original_smiles == INVALID_REACTION
    assert res.selected_mapping == ""
    assert res.mapping_type == "mcs"


@pytest.mark.parametrize("rxn", TEST_REACTIONS)
def test_map_reactions_parallel_returns_valid_result(rxn: str):
    """Each parallel-mapped reaction produces a non-empty mapping for valid input."""
    mapper = ParallelMCSReactionMapper("test_mcs_par", workers=2, chunksize=1)

    results = mapper.map_reactions([rxn])

    assert len(results) == 1
    assert results[0].original_smiles == rxn
    assert results[0].selected_mapping


def test_map_reactions_parallel_returns_same_order_as_input():
    """Results are returned in the same order as the input list."""
    mapper = ParallelMCSReactionMapper("test_mcs_par", workers=2, chunksize=1)

    rxns = [*TEST_REACTIONS, INVALID_REACTION]
    results = mapper.map_reactions(rxns)

    assert len(results) == len(rxns)
    for i, rxn in enumerate(rxns):
        assert results[i].original_smiles == rxn


def test_map_reactions_parallel_matches_serial_results():
    """Parallel output matches serial MCSReactionMapper output for the same inputs."""
    serial_mapper = MCSReactionMapper(mapper_name="test_serial")
    parallel_mapper = ParallelMCSReactionMapper("test_mcs_par", workers=2, chunksize=1)

    serial_results = serial_mapper.map_reactions(TEST_REACTIONS)
    parallel_results = parallel_mapper.map_reactions(TEST_REACTIONS)

    assert len(serial_results) == len(parallel_results)
    for s, p in zip(serial_results, parallel_results):
        assert s.selected_mapping == p.selected_mapping


def test_map_reactions_parallel_handles_invalid_smiles():
    """Invalid reactions don't crash workers and return default results."""
    mapper = ParallelMCSReactionMapper("test_mcs_par", workers=2, chunksize=1)

    rxns = [INVALID_REACTION, *TEST_REACTIONS, INVALID_REACTION]
    results = mapper.map_reactions(rxns)

    assert len(results) == len(rxns)
    assert results[0].selected_mapping == ""
    assert results[-1].selected_mapping == ""
    for i in range(1, len(rxns) - 1):
        assert results[i].selected_mapping


def test_map_reactions_parallel_empty_list():
    """Empty input list returns empty results list."""
    mapper = ParallelMCSReactionMapper("test_mcs_par", workers=2)

    results = mapper.map_reactions([])

    assert results == []


def test_map_reactions_parallel_single_reaction():
    """Degenerate case: one reaction still works with the pool."""
    mapper = ParallelMCSReactionMapper("test_mcs_par", workers=2, chunksize=1)

    rxn = TEST_REACTIONS[0]
    results = mapper.map_reactions([rxn])

    assert len(results) == 1
    assert results[0].original_smiles == rxn
    assert results[0].selected_mapping


def test_map_reactions_parallel_atom_mapping_correctness():
    """Verify that atom map numbers are correctly assigned in parallel output."""
    mapper = ParallelMCSReactionMapper("test_mcs_par", workers=2, chunksize=1)

    rxn = "F.Nc1nc(Br)c(Br)cc1Br.O=N[O-].[Na+].c1ccncc1>>Fc1nc(Br)c(Br)cc1Br"
    results = mapper.map_reactions([rxn])

    mapped = results[0].selected_mapping
    assert mapped

    _reactants, products = _split_rxn(mapped)
    assert len(products) == 1

    prod_mol = Chem.MolFromSmiles(products[0])
    assert prod_mol is not None

    br_atoms = [a for a in prod_mol.GetAtoms() if a.GetSymbol() == "Br"]
    assert br_atoms
    assert all(a.GetAtomMapNum() != 0 for a in br_atoms)
