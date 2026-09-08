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
            adjacent_atom_multiplier=20,
            identical_adjacent_atom_multiplier=5,
            used_atom_divisor=2,
            sequence_max_length=256,
        )
        return mapper


class TestConstructorDefaults:
    """Verify that constructor stores default config values as properties."""

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


# ---------------------------------------------------------------------------
# _apply_noisy_or
# ---------------------------------------------------------------------------


class TestApplyNoisyOr:
    """Tests for the _apply_noisy_or symmetry aggregation method."""

    def test_noisy_or_two_atoms_both_confident(self, mapper):
        """Two atoms both at 0.99 → noisy-OR ≈ 0.9901."""
        attn = np.array([[0.99, 0.01], [0.01, 0.99]])
        sym = {0: [1], 1: [0]}
        result = mapper._apply_noisy_or(attn, sym, axis=1)
        # Column 0: 1 - (1-0.99)(1-0.01) = 1 - 0.0099 = 0.9901
        assert result[0, 0] == pytest.approx(0.9901, abs=1e-4)
        assert result[1, 0] == pytest.approx(0.9901, abs=1e-4)

    def test_noisy_or_two_atoms_one_confident(self, mapper):
        """One atom at 0.99, other at 0.01 → noisy-OR ≈ 0.99."""
        attn = np.array([[0.99, 0.01], [0.99, 0.01]])
        sym = {0: [1], 1: [0]}
        result = mapper._apply_noisy_or(attn, sym, axis=1)
        expected = 1 - (1 - 0.99) * (1 - 0.01)
        assert result[0, 0] == pytest.approx(expected, abs=1e-6)
        assert result[1, 0] == pytest.approx(expected, abs=1e-6)

    def test_noisy_or_two_atoms_both_moderate(self, mapper):
        """Two atoms both at 0.50 → noisy-OR = 0.75."""
        attn = np.array([[0.50, 0.50], [0.50, 0.50]])
        sym = {0: [1], 1: [0]}
        result = mapper._apply_noisy_or(attn, sym, axis=1)
        expected = 1 - (1 - 0.50) * (1 - 0.50)
        assert result[0, 0] == pytest.approx(expected, abs=1e-6)
        assert result[1, 0] == pytest.approx(expected, abs=1e-6)

    def test_noisy_or_does_not_exceed_one(self, mapper):
        """Two atoms both at 1.0 → noisy-OR = 1.0 (never exceeds 1)."""
        attn = np.array([[1.0, 1.0], [1.0, 1.0]])
        sym = {0: [1], 1: [0]}
        result = mapper._apply_noisy_or(attn, sym, axis=1)
        assert result.max() <= 1.0
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_noisy_or_no_symmetry(self, mapper):
        """Empty symmetric_indices → unchanged."""
        attn = np.array([[0.3, 0.7], [0.6, 0.4]])
        result = mapper._apply_noisy_or(attn, {}, axis=1)
        np.testing.assert_array_equal(result, attn)

    def test_noisy_or_axis_0(self, mapper):
        """Axis 0 combines rows (product atoms)."""
        attn = np.array([[0.50, 0.30], [0.50, 0.30]])
        sym = {0: [1], 1: [0]}
        result = mapper._apply_noisy_or(attn, sym, axis=0)
        # Row 0 and 1 both become: 1 - (1-0.5)(1-0.5) = 0.75 for col 0
        # and 1 - (1-0.3)(1-0.3) = 0.51 for col 1
        assert result[0, 0] == pytest.approx(0.75, abs=1e-6)
        assert result[0, 1] == pytest.approx(0.51, abs=1e-6)
        assert result[1, 0] == pytest.approx(0.75, abs=1e-6)
        assert result[1, 1] == pytest.approx(0.51, abs=1e-6)

    def test_noisy_or_preserves_non_symmetric_atoms(self, mapper):
        """Atoms not in any symmetric group are unchanged."""
        attn = np.array([[0.50, 0.50, 0.20], [0.50, 0.50, 0.20], [0.10, 0.10, 0.80]])
        sym = {0: [1], 1: [0]}
        result = mapper._apply_noisy_or(attn, sym, axis=1)
        # Column 2 (atom index 2) is not in any symmetric group
        assert result[0, 2] == pytest.approx(0.20, abs=1e-6)
        assert result[2, 2] == pytest.approx(0.80, abs=1e-6)

    def test_noisy_or_three_atom_group(self, mapper):
        """Three symmetric atoms at 0.5 each → noisy-OR = 1 - 0.5^3 = 0.875."""
        attn = np.array([[0.5, 0.5, 0.5]])
        sym = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        result = mapper._apply_noisy_or(attn, sym, axis=1)
        expected = 1 - (1 - 0.5) ** 3
        assert result[0, 0] == pytest.approx(expected, abs=1e-6)
        assert result[0, 1] == pytest.approx(expected, abs=1e-6)
        assert result[0, 2] == pytest.approx(expected, abs=1e-6)

    def test_noisy_or_does_not_modify_input(self, mapper):
        """The input array is not modified."""
        attn = np.array([[0.50, 0.50], [0.50, 0.50]])
        original = attn.copy()
        sym = {0: [1], 1: [0]}
        mapper._apply_noisy_or(attn, sym, axis=1)
        np.testing.assert_array_equal(attn, original)


# ---------------------------------------------------------------------------
# _symmetry_aware_confidence
# ---------------------------------------------------------------------------


class TestSymmetryAwareConfidence:
    """Tests for the _symmetry_aware_confidence method."""

    def test_no_symmetry_returns_p2r_unchanged(self, mapper):
        """No symmetric atoms → result equals p2r (r2p is not used)."""
        p2r = np.array([[0.8, 0.2], [0.3, 0.7]])
        r2p = np.array([[0.6, 0.4], [0.1, 0.9]])
        result = mapper._symmetry_aware_confidence(p2r, r2p, {}, {})
        np.testing.assert_array_almost_equal(result, p2r)

    def test_reactant_symmetry_only(self, mapper):
        """Only reactant symmetry: opposite-side sum on p2r axis=1."""
        p2r = np.array([[0.50, 0.50], [0.30, 0.30]])
        r2p = np.array([[0.99, 0.01], [0.99, 0.01]])  # r2p should be ignored
        r_sym = {0: [1], 1: [0]}
        p_sym = {}
        result = mapper._symmetry_aware_confidence(p2r, r2p, r_sym, p_sym)
        # p2r: sum axis=1 → [[1.0, 1.0], [0.6, 0.6]], clamp → same, noisy-OR p_sym(empty) → same
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert result[0, 1] == pytest.approx(1.0, abs=1e-6)
        assert result[1, 0] == pytest.approx(0.60, abs=1e-6)
        assert result[1, 1] == pytest.approx(0.60, abs=1e-6)

    def test_product_symmetry_only(self, mapper):
        """Only product symmetry: noisy-OR on p2r axis=0."""
        p2r = np.array([[0.50, 0.30], [0.50, 0.30]])
        r2p = np.array([[0.99, 0.01], [0.99, 0.01]])  # r2p should be ignored
        r_sym = {}
        p_sym = {0: [1], 1: [0]}
        result = mapper._symmetry_aware_confidence(p2r, r2p, r_sym, p_sym)
        # p2r: sum r_sym(empty) → no change, noisy-OR p_sym axis=0:
        #   col 0: 1-(1-0.5)(1-0.5)=0.75, col 1: 1-(1-0.3)(1-0.3)=0.51
        assert result[0, 0] == pytest.approx(0.75, abs=1e-6)
        assert result[0, 1] == pytest.approx(0.51, abs=1e-6)
        assert result[1, 0] == pytest.approx(0.75, abs=1e-6)
        assert result[1, 1] == pytest.approx(0.51, abs=1e-6)

    def test_both_sides_symmetry(self, mapper):
        """Both reactant and product symmetry present."""
        p2r = np.array([[0.50, 0.50], [0.50, 0.50]])
        r2p = np.array([[0.99, 0.01], [0.99, 0.01]])  # r2p should be ignored
        r_sym = {0: [1], 1: [0]}
        p_sym = {0: [1], 1: [0]}
        result = mapper._symmetry_aware_confidence(p2r, r2p, r_sym, p_sym)
        # p2r: sum axis=1 → 1.0 everywhere, then noisy-OR axis=0 → 1.0
        assert result.max() <= 1.0
        np.testing.assert_array_almost_equal(result, np.ones((2, 2)))

    def test_values_in_zero_one(self, mapper):
        """All output values are in [0, 1]."""
        p2r = np.array([[0.99, 0.01], [0.50, 0.50], [0.30, 0.70]])
        r2p = np.array([[0.80, 0.20], [0.60, 0.40], [0.10, 0.90]])
        r_sym = {0: [1], 1: [0]}
        p_sym = {0: [2], 2: [0]}
        result = mapper._symmetry_aware_confidence(p2r, r2p, r_sym, p_sym)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_r2p_does_not_affect_result(self, mapper):
        """r2p values should not influence the confidence matrix."""
        p2r = np.array([[0.50, 0.50], [0.30, 0.30]])
        r_sym = {0: [1], 1: [0]}
        p_sym = {}
        r2p_a = np.array([[0.99, 0.01], [0.99, 0.01]])
        r2p_b = np.array([[0.01, 0.99], [0.01, 0.99]])
        result_a = mapper._symmetry_aware_confidence(p2r, r2p_a, r_sym, p_sym)
        result_b = mapper._symmetry_aware_confidence(p2r, r2p_b, r_sym, p_sym)
        np.testing.assert_array_equal(result_a, result_b)

    def test_one_to_one_flag_does_not_affect_result(self, mapper):
        """one_to_one_correspondence flag no longer changes behavior (p2r-only)."""
        p2r = np.array([[0.50, 0.50], [0.30, 0.30]])
        r2p = np.array([[0.99, 0.01], [0.99, 0.01]])
        r_sym = {0: [1], 1: [0]}
        p_sym = {}
        result_true = mapper._symmetry_aware_confidence(
            p2r, r2p, r_sym, p_sym, one_to_one_correspondence=True
        )
        result_false = mapper._symmetry_aware_confidence(
            p2r, r2p, r_sym, p_sym, one_to_one_correspondence=False
        )
        np.testing.assert_array_equal(result_true, result_false)
