from unittest.mock import patch

import numpy as np
import pytest

from agave_chem.mappers.neural.neural_mapper import NeuralReactionMapper


@pytest.fixture
def mapper():
    """Create a NeuralReactionMapper with mocked model loading."""
    with patch(
        "agave_chem.mappers.neural.neural_mapper.load_neural_albert_model"
    ) as mock_load:
        mock_load.return_value = None
        mapper = NeuralReactionMapper(mapper_name="test")
        return mapper


@pytest.fixture
def custom_mapper():
    """Create a NeuralReactionMapper with custom config and mocked model loading."""
    with patch(
        "agave_chem.mappers.neural.neural_mapper.load_neural_albert_model"
    ) as mock_load:
        mock_load.return_value = None
        mapper = NeuralReactionMapper(
            mapper_name="test",
            layer=5,
            head=3,
            adjacent_atom_multiplier=20,
            identical_adjacent_atom_multiplier=5,
            used_atom_divisor=2,
            sequence_max_length=256,
        )
        return mapper


class TestConstructorDefaults:
    """Verify that constructor stores default config values as properties."""

    def test_default_layer(self, mapper):
        assert mapper._layer == 11

    def test_default_head(self, mapper):
        assert mapper._head == 7

    def test_default_adjacent_atom_multiplier(self, mapper):
        assert mapper._adjacent_atom_multiplier == 10

    def test_default_identical_adjacent_atom_multiplier(self, mapper):
        assert mapper._identical_adjacent_atom_multiplier == 10

    def test_default_used_atom_divisor(self, mapper):
        assert mapper._used_atom_divisor == 10

    def test_default_sequence_max_length(self, mapper):
        assert mapper._sequence_max_length == 1024

    def test_mapper_type(self, mapper):
        assert mapper._mapper_type == "neural"


class TestConstructorCustomValues:
    """Verify that constructor stores custom config values as properties."""

    def test_custom_layer(self, custom_mapper):
        assert custom_mapper._layer == 5

    def test_custom_head(self, custom_mapper):
        assert custom_mapper._head == 3

    def test_custom_adjacent_atom_multiplier(self, custom_mapper):
        assert custom_mapper._adjacent_atom_multiplier == 20

    def test_custom_identical_adjacent_atom_multiplier(self, custom_mapper):
        assert custom_mapper._identical_adjacent_atom_multiplier == 5

    def test_custom_used_atom_divisor(self, custom_mapper):
        assert custom_mapper._used_atom_divisor == 2

    def test_custom_sequence_max_length(self, custom_mapper):
        assert custom_mapper._sequence_max_length == 256


class TestMethodSignatures:
    """Verify that method signatures no longer accept moved parameters."""

    def test_map_reaction_does_not_accept_layer(self, mapper):
        with pytest.raises(TypeError):
            mapper.map_reaction("CC>>CC", layer=5)

    def test_map_reaction_does_not_accept_head(self, mapper):
        with pytest.raises(TypeError):
            mapper.map_reaction("CC>>CC", head=3)

    def test_map_reaction_does_not_accept_sequence_max_length(self, mapper):
        with pytest.raises(TypeError):
            mapper.map_reaction("CC>>CC", sequence_max_length=256)

    def test_map_reaction_does_not_accept_adjacent_atom_multiplier(self, mapper):
        with pytest.raises(TypeError):
            mapper.map_reaction("CC>>CC", adjacent_atom_multiplier=20)

    def test_map_reaction_does_not_accept_used_atom_divisor(self, mapper):
        with pytest.raises(TypeError):
            mapper.map_reaction("CC>>CC", used_atom_divisor=2)

    def test_map_reactions_does_not_accept_layer(self, mapper):
        with pytest.raises(TypeError):
            mapper.map_reactions(["CC>>CC"], layer=5)

    def test_map_reactions_does_not_accept_head(self, mapper):
        with pytest.raises(TypeError):
            mapper.map_reactions(["CC>>CC"], head=3)

    def test_map_reactions_does_not_accept_sequence_max_length(self, mapper):
        with pytest.raises(TypeError):
            mapper.map_reactions(["CC>>CC"], sequence_max_length=256)

    def test_map_reactions_does_not_accept_adjacent_atom_multiplier(self, mapper):
        with pytest.raises(TypeError):
            mapper.map_reactions(["CC>>CC"], adjacent_atom_multiplier=20)

    def test_map_reactions_does_not_accept_used_atom_divisor(self, mapper):
        with pytest.raises(TypeError):
            mapper.map_reactions(["CC>>CC"], used_atom_divisor=2)


class TestMapReactionInvalidInput:
    """Verify that invalid input returns a default result."""

    def test_invalid_smiles_returns_default(self, mapper):
        res = mapper.map_reaction("CC")
        assert res.original_smiles == ""
        assert res.selected_mapping == ""
        assert res.mapping_type == "neural"

    def test_invalid_smiles_in_list_returns_default(self, mapper):
        results = mapper.map_reactions(["CC"])
        assert len(results) == 1
        assert results[0].selected_mapping == ""
        assert results[0].mapping_type == "neural"


class TestResultOrderPreservation:
    """Verify that internal length-sorting does not affect output order."""

    @pytest.fixture(autouse=True)
    def _mock_inference(self, mapper):
        """Mock inference and mapping to avoid needing the real model."""
        from agave_chem.mappers.reaction_mapper import ReactionMapperResult

        def _fake_batch(texts, **kwargs):
            return [(np.zeros((2, 2)), ["C", "C"])] * len(texts)

        def _fake_map(rxn_smiles, **kwargs):
            result = ReactionMapperResult(
                original_smiles=rxn_smiles,
                selected_mapping=rxn_smiles,
                possible_mappings={},
                mapping_type="neural",
                mapping_score=1.0,
                additional_info=[{}],
            )
            return result, None

        with (
            patch.object(
                mapper, "_get_attention_matrices_batch", side_effect=_fake_batch
            ),
            patch.object(mapper, "_map_from_attention", side_effect=_fake_map),
        ):
            yield

    def test_results_match_input_order_with_mixed_lengths(self, mapper):
        rxns = [
            "CC(=O)Oc1ccccc1OC(C)=O.O=[N+]([O-])O>>O=[N+]([O-])c1cc(O)c(O)c([N+](=O)[O-])c1",
            "CC>>CC",
            "CCCCCO>>CCCCCO",
        ]
        results = mapper.map_reactions(rxns)
        assert len(results) == len(rxns)
        assert results[0].original_smiles == rxns[0]
        assert results[1].original_smiles == rxns[1]
        assert results[2].original_smiles == rxns[2]

    def test_results_match_input_order_with_invalid_interspersed(self, mapper):
        rxns = [
            "CC>>CC",
            "CC",
            "CCCCCO>>CCCCCO",
            "invalid",
            "CCO>>CCO",
        ]
        results = mapper.map_reactions(rxns)
        assert len(results) == len(rxns)
        assert results[0].original_smiles == rxns[0]
        assert results[1].original_smiles == ""
        assert results[1].selected_mapping == ""
        assert results[2].original_smiles == rxns[2]
        assert results[3].selected_mapping == ""
        assert results[4].original_smiles == rxns[4]

    def test_single_reaction_returns_single_result(self, mapper):
        results = mapper.map_reactions(["CC>>CC"])
        assert len(results) == 1
