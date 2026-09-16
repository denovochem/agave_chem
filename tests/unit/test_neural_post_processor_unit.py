from unittest.mock import patch

import numpy as np
import pytest

from agave_chem.mappers.neural.post_processor import NeuralPostProcessor, _init_worker


@pytest.fixture
def pp():
    """Create a default NeuralPostProcessor."""
    return NeuralPostProcessor()


@pytest.fixture
def custom_pp():
    """Create a NeuralPostProcessor with custom config."""
    return NeuralPostProcessor(
        adjacent_atom_multiplier=20,
        identical_adjacent_atom_multiplier=5,
        used_atom_divisor=2,
        sequence_max_length=256,
        mapper_type="neural",
    )


class TestConstructor:
    """Verify constructor stores config values."""

    def test_defaults(self, pp):
        assert pp._adjacent_atom_multiplier == 10
        assert pp._identical_adjacent_atom_multiplier == 10
        assert pp._used_atom_divisor == 10
        assert pp._sequence_max_length == 1024
        assert pp._mapper_type == "neural"

    def test_custom(self, custom_pp):
        assert custom_pp._adjacent_atom_multiplier == 20
        assert custom_pp._identical_adjacent_atom_multiplier == 5
        assert custom_pp._used_atom_divisor == 2
        assert custom_pp._sequence_max_length == 256


class TestEncodeAtom:
    """Verify _encode_atom produces correct feature vectors."""

    def test_methanol_carbon(self, pp):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CO")
        atom = mol.GetAtomWithIdx(0)
        encoding = pp._encode_atom(atom)
        assert encoding == [6, 0, 0, 0, 3, 1]

    def test_aromatic_carbon(self, pp):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("c1ccccc1")
        atom = mol.GetAtomWithIdx(0)
        encoding = pp._encode_atom(atom)
        assert encoding[2] == 1  # aromatic
        assert encoding[3] == 1  # in ring


class TestGetReactantsProductsDict:
    """Verify token parsing into reactant/product dictionaries."""

    def test_simple_reaction(self, pp):
        tokens = ["C", "C", ">>", "C", "C"]
        info = pp.get_reactants_products_dict(tokens)
        assert info["reactants_start_index"] == 0
        assert info["reactants_end_index"] == 1
        assert info["products_start_index"] == 3
        assert info["products_end_index"] == 4
        assert 0 in info["reactants_dict"]
        assert 1 in info["reactants_dict"]
        assert 3 in info["products_dict"]
        assert 4 in info["products_dict"]
        assert 2 in info["non_atom_tokens"]


class TestMaskAttnMatrix:
    """Verify attention masking zeroes out invalid cross-attentions."""

    def test_diagonal_probs_are_zero(self, pp):
        tokens = ["C", "C", ">>", "C", "C"]
        info = pp.get_reactants_products_dict(tokens)
        attn = np.zeros((5, 5))
        probs, _ = pp.mask_attn_matrix(attn, info)
        # Reactant-reactant block should be zero
        assert probs[0, 1] == 0.0
        assert probs[1, 0] == 0.0
        # Product-product block should be zero
        assert probs[3, 4] == 0.0
        assert probs[4, 3] == 0.0


class TestGetAlignedAttnScores:
    """Verify cross-attention slicing and transposition."""

    def test_shapes(self, pp):
        out = np.ones((10, 10))
        p2r, r2p = pp.get_aligned_attn_scores(out, 0, 3, 5)
        assert p2r.shape == (5, 4)
        assert r2p.shape == (5, 4)


class TestRemoveNonAtomRowsAndColumns:
    """Verify non-atom token removal from attention matrix."""

    def test_removes_non_atom_tokens(self, pp):
        tokens = ["C", "C", ">>", "C", "C"]
        info = pp.get_reactants_products_dict(tokens)
        attn = np.ones((2, 2))
        result = pp.remove_non_atom_rows_and_columns(attn, info)
        assert result.shape == (2, 2)


class TestGetDuplicateIndices:
    """Verify duplicate detection across sublists."""

    def test_no_duplicates(self, pp):
        result = pp.get_duplicate_indices([[1, 2, 3]])
        assert result == {}

    def test_with_duplicates(self, pp):
        result = pp.get_duplicate_indices([[1, 1, 2]])
        assert 0 in result
        assert 1 in result
        assert result[0] == [1]
        assert result[1] == [0]

    def test_multiple_sublists(self, pp):
        result = pp.get_duplicate_indices([[1, 1], [2, 2]])
        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert 3 in result


class TestApplyNoisyOr:
    """Tests for _apply_noisy_or."""

    def test_two_atoms_both_confident(self, pp):
        attn = np.array([[0.99, 0.01], [0.01, 0.99]])
        sym = {0: [1], 1: [0]}
        result = pp._apply_noisy_or(attn, sym, axis=1)
        expected = 1 - (1 - 0.99) * (1 - 0.01)
        assert result[0, 0] == pytest.approx(expected, abs=1e-4)

    def test_no_symmetry_unchanged(self, pp):
        attn = np.array([[0.3, 0.7], [0.6, 0.4]])
        result = pp._apply_noisy_or(attn, {}, axis=1)
        np.testing.assert_array_equal(result, attn)

    def test_does_not_modify_input(self, pp):
        attn = np.array([[0.50, 0.50], [0.50, 0.50]])
        original = attn.copy()
        pp._apply_noisy_or(attn, {0: [1], 1: [0]}, axis=1)
        np.testing.assert_array_equal(attn, original)


class TestSymmetryAwareConfidence:
    """Tests for _symmetry_aware_confidence."""

    def test_no_symmetry_returns_p2r(self, pp):
        p2r = np.array([[0.8, 0.2], [0.3, 0.7]])
        r2p = np.array([[0.6, 0.4], [0.1, 0.9]])
        result = pp._symmetry_aware_confidence(p2r, r2p, {}, {})
        np.testing.assert_array_almost_equal(result, p2r)

    def test_r2p_ignored(self, pp):
        p2r = np.array([[0.50, 0.50], [0.30, 0.30]])
        r_sym = {0: [1], 1: [0]}
        p_sym = {}
        r2p_a = np.array([[0.99, 0.01], [0.99, 0.01]])
        r2p_b = np.array([[0.01, 0.99], [0.01, 0.99]])
        result_a = pp._symmetry_aware_confidence(p2r, r2p_a, r_sym, p_sym)
        result_b = pp._symmetry_aware_confidence(p2r, r2p_b, r_sym, p_sym)
        np.testing.assert_array_equal(result_a, result_b)


class TestMapFromAttention:
    """Tests for map_from_attention edge cases."""

    def test_unknown_token_returns_empty(self, pp):
        result, expanded = pp.map_from_attention(
            rxn_smiles="CC>>CC",
            attn=np.zeros((5, 5)),
            tokens=["[UNK]", "C", ">>", "C", "C"],
        )
        assert result.selected_mapping == ""
        assert expanded is None

    def test_no_reaction_symbol_returns_empty(self, pp):
        result, _ = pp.map_from_attention(
            rxn_smiles="CC",
            attn=np.zeros((2, 2)),
            tokens=["C", "C"],
        )
        assert result.selected_mapping == ""

    def test_sequence_too_long_returns_empty(self, pp):
        result, _ = pp.map_from_attention(
            rxn_smiles="CC>>CC",
            attn=np.zeros((2, 2)),
            tokens=["C"] * 1100,
        )
        assert result.selected_mapping == ""


class TestPostProcessBatch:
    """Tests for post_process_batch serial and parallel execution."""

    def test_empty_batch(self, pp):
        results = pp.post_process_batch([], num_processes=1)
        assert results == []

    def test_serial_returns_correct_count(self, pp):
        from agave_chem.mappers.reaction_mapper import ReactionMapperResult

        def _fake_map(rxn_smiles, **kwargs):
            return (
                ReactionMapperResult(
                    original_smiles=rxn_smiles,
                    selected_mapping=rxn_smiles,
                    possible_mappings={},
                    mapping_type="neural",
                    mapping_score=1.0,
                    additional_info=[{}],
                ),
                None,
            )

        tasks = [
            ("CC>>CC", np.zeros((2, 2)), ["C", "C"], True, True, True),
            ("CCC>>CCC", np.zeros((2, 2)), ["C", "C"], True, True, True),
        ]
        with patch.object(pp, "map_from_attention", side_effect=_fake_map):
            results = pp.post_process_batch(tasks, num_processes=1)
        assert len(results) == 2
        assert results[0][0].original_smiles == "CC>>CC"
        assert results[1][0].original_smiles == "CCC>>CCC"

    def test_serial_matches_parallel(self, pp):
        """Serial and parallel post-processing produce identical results.

        Uses real ``map_from_attention`` with tokens that lack a reaction
        separator, so both paths return empty results — verifying that
        parallel execution preserves order and structure.
        """
        tasks = [
            ("CC>>CC", np.zeros((2, 2)), ["C", "C"], True, True, True),
            ("CCC>>CCC", np.zeros((2, 2)), ["C", "C"], True, True, True),
            ("CCCC>>CCCC", np.zeros((2, 2)), ["C", "C"], True, True, True),
            ("CCCCO>>CCCCO", np.zeros((2, 2)), ["C", "C"], True, True, True),
        ]

        serial_results = pp.post_process_batch(tasks, num_processes=1)
        parallel_results = pp.post_process_batch(tasks, num_processes=2)

        assert len(serial_results) == len(parallel_results)
        for s, p in zip(serial_results, parallel_results):
            assert s[0].original_smiles == p[0].original_smiles
            assert s[0].selected_mapping == p[0].selected_mapping

    def test_pool_parameter_reuses_provided_pool(self, pp):
        """When a pool is provided, it is used and returns results in order.

        Uses real ``map_from_attention`` with tokens that lack a reaction
        separator, so results are empty — verifying that the pool path
        preserves order and structure.
        """
        import multiprocessing as mp

        tasks = [
            ("CC>>CC", np.zeros((2, 2)), ["C", "C"], True, True, True),
            ("CCC>>CCC", np.zeros((2, 2)), ["C", "C"], True, True, True),
        ]

        pool = mp.Pool(
            processes=2,
            initializer=_init_worker,
            initargs=(
                pp._adjacent_atom_multiplier,
                pp._identical_adjacent_atom_multiplier,
                pp._used_atom_divisor,
                pp._sequence_max_length,
                pp._mapper_type,
            ),
        )
        try:
            results = pp.post_process_batch(tasks, pool=pool)
            assert len(results) == 2
        finally:
            pool.close()
            pool.join()

    def test_pool_parameter_matches_serial(self, pp):
        """Results from a provided pool match serial results."""
        import multiprocessing as mp

        tasks = [
            ("CC>>CC", np.zeros((2, 2)), ["C", "C"], True, True, True),
            ("CCC>>CCC", np.zeros((2, 2)), ["C", "C"], True, True, True),
            ("CCCC>>CCCC", np.zeros((2, 2)), ["C", "C"], True, True, True),
        ]

        serial_results = pp.post_process_batch(tasks, num_processes=1)

        pool = mp.Pool(
            processes=2,
            initializer=_init_worker,
            initargs=(
                pp._adjacent_atom_multiplier,
                pp._identical_adjacent_atom_multiplier,
                pp._used_atom_divisor,
                pp._sequence_max_length,
                pp._mapper_type,
            ),
        )
        try:
            pool_results = pp.post_process_batch(tasks, pool=pool)
        finally:
            pool.close()
            pool.join()

        assert len(serial_results) == len(pool_results)
        for s, p in zip(serial_results, pool_results):
            assert s[0].original_smiles == p[0].original_smiles
            assert s[0].selected_mapping == p[0].selected_mapping
