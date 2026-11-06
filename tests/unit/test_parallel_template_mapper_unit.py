"""Unit tests for ParallelTemplateReactionMapper."""

import pytest

from agave_chem.mappers.template.parallel_template_mapper import (
    ParallelTemplateReactionMapper,
)
from agave_chem.mappers.template.template_mapper import TemplateReactionMapper
from agave_chem.mappers.types import ReactionInput

# A reaction that the template mapper can successfully map.
VALID_REACTION = (
    "[CH2:0]([Cl])[c:1]1ccccc1.[CH2:2]([O:3][H])>>[CH2:0]([O:3])[c:1]1ccccc1.[Cl][H]"
)

# A reaction that will not match any template.
NO_MATCH_REACTION = "CC>>CCO"

INVALID_REACTION = "invalid_smiles"

TEST_REACTIONS = [VALID_REACTION, NO_MATCH_REACTION]


def test_map_reaction_single_returns_valid_result():
    """map_reaction delegates to a lazy in-process TemplateReactionMapper."""
    mapper = ParallelTemplateReactionMapper("test_template_par", workers=2)

    res = mapper.map_reaction(VALID_REACTION)

    assert res.original_smiles == VALID_REACTION
    assert res.mapping_type == "template"


def test_map_reaction_invalid_smiles_returns_default():
    """Invalid SMILES via map_reaction returns a default empty result."""
    mapper = ParallelTemplateReactionMapper("test_template_par", workers=2)

    res = mapper.map_reaction(INVALID_REACTION)

    assert res.original_smiles == INVALID_REACTION
    assert res.selected_mapping == ""


@pytest.mark.parametrize("rxn", TEST_REACTIONS)
def test_map_reactions_parallel_returns_result(rxn: str):
    """Each parallel-mapped reaction produces a result."""
    mapper = ParallelTemplateReactionMapper("test_template_par", workers=2, chunksize=1)

    results = mapper.map_reactions([rxn])

    assert len(results) == 1
    assert results[0].original_smiles == rxn


def test_map_reactions_parallel_returns_same_order_as_input():
    """Results are returned in the same order as the input list."""
    mapper = ParallelTemplateReactionMapper("test_template_par", workers=2, chunksize=1)

    rxns = [*TEST_REACTIONS, INVALID_REACTION]
    results = mapper.map_reactions(rxns)

    assert len(results) == len(rxns)
    for i, rxn in enumerate(rxns):
        assert results[i].original_smiles == rxn


def test_map_reactions_parallel_matches_serial_results():
    """Parallel output matches serial TemplateReactionMapper output for the same inputs."""
    serial_mapper = TemplateReactionMapper("test_serial")
    parallel_mapper = ParallelTemplateReactionMapper(
        "test_template_par", workers=2, chunksize=1
    )

    serial_results = serial_mapper.map_reactions(TEST_REACTIONS)
    parallel_results = parallel_mapper.map_reactions(TEST_REACTIONS)

    assert len(serial_results) == len(parallel_results)
    for s, p in zip(serial_results, parallel_results):
        assert s.selected_mapping == p.selected_mapping


def test_map_reactions_parallel_handles_invalid_smiles():
    """Invalid reactions don't crash workers and return default results."""
    mapper = ParallelTemplateReactionMapper("test_template_par", workers=2, chunksize=1)

    rxns = [INVALID_REACTION, VALID_REACTION, INVALID_REACTION]
    results = mapper.map_reactions(rxns)

    assert len(results) == len(rxns)
    assert results[0].selected_mapping == ""
    assert results[-1].selected_mapping == ""


def test_map_reactions_parallel_empty_list():
    """Empty input list returns empty results list."""
    mapper = ParallelTemplateReactionMapper("test_template_par", workers=2)

    results = mapper.map_reactions([])

    assert results == []


def test_map_reactions_parallel_single_reaction():
    """Degenerate case: one reaction still works with the pool."""
    mapper = ParallelTemplateReactionMapper("test_template_par", workers=2, chunksize=1)

    results = mapper.map_reactions([VALID_REACTION])

    assert len(results) == 1
    assert results[0].original_smiles == VALID_REACTION


def test_map_reactions_parallel_with_reaction_input():
    """ReactionInput objects are passed through to workers, preserving MCS data."""
    mapper = ParallelTemplateReactionMapper("test_template_par", workers=2, chunksize=1)

    rxn_input = ReactionInput(
        original_smiles=VALID_REACTION,
        stripped_smiles=VALID_REACTION,
        identical_fragments=[],
        mcs_mapped_smiles=None,
        unmapped_product_atom_islands={},
        one_to_one_correspondence=True,
    )

    results = mapper.map_reactions([rxn_input])

    assert len(results) == 1
    assert results[0].original_smiles == VALID_REACTION


def test_map_reactions_parallel_classification_info_preserved():
    """Classification metadata survives pickling across process boundaries."""
    mapper = ParallelTemplateReactionMapper("test_template_par", workers=2, chunksize=1)

    results = mapper.map_reactions([VALID_REACTION])

    assert len(results) == 1
    if results[0].selected_mapping:
        assert results[0].classification_info
        selected_key = results[0].selected_mapping
        assert selected_key in results[0].classification_info
