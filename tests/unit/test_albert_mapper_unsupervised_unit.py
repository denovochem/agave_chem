"""Unit tests for workflows/model_training_scripts/albert_mapper_unuspervised_training.py."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from rdkit import Chem

# Ensure the workflows directory is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_ROOT = REPO_ROOT / "workflows"
for p in [str(REPO_ROOT), str(WORKFLOWS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_training_scripts.albert_mapper_unuspervised_training import (
    AlbertTrainer,
    MLMConfig,
    MLMDataset,
    ModelConfig,
    SpanMLMConfig,
    TrainingConfig,
    _build_atom_token_map,
    _get_plausible_replacement,
    _parse_reaction_molecules,
    _select_graph_neighborhood,
    _unwrap_model,
    apply_mlm_masking,
    apply_span_mlm_masking,
    build_albert_model,
    keep_original_tokens,
    preprocess_token,
    replace_with_mask,
    replace_with_random_token,
    resolve_protected_token_ids,
)
from transformers import AlbertConfig, AlbertForMaskedLM

from agave_chem.mappers.neural.constants import (
    smiles_token_to_id_dict,
)
from agave_chem.mappers.neural.tokenizer import CustomTokenizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    """Create a CustomTokenizer for use in tests."""
    return CustomTokenizer(smiles_token_to_id_dict)


@pytest.fixture
def simple_reaction():
    """A simple mapped reaction SMILES string."""
    return "[C:1][C:2][O:3]>>[C:1][C:2][O:3]"


@pytest.fixture
def unmapped_reaction():
    """A simple unmapped reaction SMILES string."""
    return "CCO>>CCO"


# ---------------------------------------------------------------------------
# preprocess_token
# ---------------------------------------------------------------------------


class TestPreprocessToken:
    """Tests for preprocess_token."""

    def test_mask_replacement(self):
        """When rand < mask_token_prob, token is replaced with mask_token_id."""
        mlm_config = MLMConfig()
        with patch(
            "model_training_scripts.albert_mapper_unuspervised_training.random.random",
            return_value=0.0,
        ):
            new_id, modified = preprocess_token(
                token_id=5, mask_token_id=99, vocab_size=100, mlm_config=mlm_config
            )
        assert new_id == 99
        assert modified is True

    def test_random_replacement(self):
        """When mask_token_prob <= rand < mask_token_prob + random_token_prob, random token."""
        mlm_config = MLMConfig()
        with (
            patch(
                "model_training_scripts.albert_mapper_unuspervised_training.random.random",
                return_value=0.85,
            ),
            patch(
                "model_training_scripts.albert_mapper_unuspervised_training.random.randint",
                return_value=42,
            ),
        ):
            new_id, modified = preprocess_token(
                token_id=5, mask_token_id=99, vocab_size=100, mlm_config=mlm_config
            )
            assert new_id == 42
            assert modified is True

    def test_keep_original(self):
        """When rand >= mask_token_prob + random_token_prob, keep original."""
        mlm_config = MLMConfig()
        with patch(
            "model_training_scripts.albert_mapper_unuspervised_training.random.random",
            return_value=0.95,
        ):
            new_id, modified = preprocess_token(
                token_id=5, mask_token_id=99, vocab_size=100, mlm_config=mlm_config
            )
        assert new_id == 5
        assert modified is False

    def test_custom_probs_mask(self):
        """Custom probabilities: all weight on mask."""
        mlm_config = MLMConfig(
            mask_token_prob=1.0, random_token_prob=0.0, keep_token_prob=0.0
        )
        with patch(
            "model_training_scripts.albert_mapper_unuspervised_training.random.random",
            return_value=0.5,
        ):
            new_id, modified = preprocess_token(
                token_id=5, mask_token_id=99, vocab_size=100, mlm_config=mlm_config
            )
        assert new_id == 99
        assert modified is True

    def test_custom_probs_keep(self):
        """Custom probabilities: all weight on keep."""
        mlm_config = MLMConfig(
            mask_token_prob=0.0, random_token_prob=0.0, keep_token_prob=1.0
        )
        with patch(
            "model_training_scripts.albert_mapper_unuspervised_training.random.random",
            return_value=0.5,
        ):
            new_id, modified = preprocess_token(
                token_id=5, mask_token_id=99, vocab_size=100, mlm_config=mlm_config
            )
        assert new_id == 5
        assert modified is False


# ---------------------------------------------------------------------------
# apply_mlm_masking
# ---------------------------------------------------------------------------


class TestApplyMlmMasking:
    """Tests for apply_mlm_masking."""

    def test_returns_correct_lengths(self, tokenizer):
        """Output lists have same length as input."""
        input_ids = [0, 5, 10, 15, 20, 2]
        mlm_config = MLMConfig(mlm_probability=0.5)
        masked, labels = apply_mlm_masking(input_ids, tokenizer, mlm_config)
        assert len(masked) == len(input_ids)
        assert len(labels) == len(input_ids)

    def test_special_tokens_not_masked(self, tokenizer):
        """Special token IDs are never masked."""
        special_ids = set(tokenizer.all_special_ids)
        non_special = [tid for tid in range(100) if tid not in special_ids][:20]
        input_ids = list(special_ids) + non_special
        mlm_config = MLMConfig(mlm_probability=1.0)
        masked, labels = apply_mlm_masking(
            input_ids, tokenizer, mlm_config, special_token_ids=special_ids
        )
        for i, tid in enumerate(input_ids):
            if tid in special_ids:
                assert masked[i] == tid
                assert labels[i] == -100

    def test_labels_are_original_for_masked(self, tokenizer):
        """Labels at masked positions contain the original token ID."""
        input_ids = [5, 10, 15, 20, 25]
        mlm_config = MLMConfig(
            mlm_probability=1.0,
            mask_token_prob=1.0,
            random_token_prob=0.0,
            keep_token_prob=0.0,
        )
        masked, labels = apply_mlm_masking(
            input_ids, tokenizer, mlm_config, special_token_ids=set()
        )
        for i in range(len(input_ids)):
            if masked[i] != input_ids[i]:
                assert labels[i] == input_ids[i]

    def test_non_masked_labels_are_negative_100(self, tokenizer):
        """Labels at non-masked positions are -100."""
        input_ids = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        mlm_config = MLMConfig(mlm_probability=0.1)
        _, labels = apply_mlm_masking(
            input_ids, tokenizer, mlm_config, special_token_ids=set()
        )
        non_masked_count = sum(1 for l in labels if l == -100)
        assert non_masked_count > 0

    def test_empty_input(self, tokenizer):
        """Empty input list returns empty outputs."""
        masked, labels = apply_mlm_masking([], tokenizer, MLMConfig())
        assert masked == []
        assert labels == []

    def test_all_special_tokens(self, tokenizer):
        """Input with only special tokens results in no masking."""
        special_ids = set(tokenizer.all_special_ids)
        input_ids = list(special_ids)[:5]
        mlm_config = MLMConfig(mlm_probability=0.5)
        masked, labels = apply_mlm_masking(input_ids, tokenizer, mlm_config)
        assert masked == input_ids
        assert all(l == -100 for l in labels)

    def test_input_not_mutated(self, tokenizer):
        """The original input_ids list is not modified."""
        input_ids = [5, 10, 15, 20, 25]
        original = input_ids.copy()
        mlm_config = MLMConfig(mlm_probability=0.5)
        apply_mlm_masking(input_ids, tokenizer, mlm_config, special_token_ids=set())
        assert input_ids == original


# ---------------------------------------------------------------------------
# replace_with_mask
# ---------------------------------------------------------------------------


class TestReplaceWithMask:
    """Tests for replace_with_mask."""

    def test_all_masked_replaced_with_mask_token(self, tokenizer):
        """With mlm_probability=1.0, all non-special tokens become mask token."""
        input_ids = [5, 10, 15, 20, 25]
        mlm_config = MLMConfig(mlm_probability=1.0)
        masked, labels = replace_with_mask(
            input_ids, tokenizer, mlm_config, special_token_ids=set()
        )
        for i in range(len(input_ids)):
            assert masked[i] == tokenizer.mask_token_id
            assert labels[i] == input_ids[i]

    def test_no_masking_with_zero_prob(self, tokenizer):
        """With mlm_probability=0.0, at least 1 is still masked (max(1, ...))."""
        input_ids = [5, 10, 15, 20, 25]
        mlm_config = MLMConfig(mlm_probability=0.0)
        masked, _ = replace_with_mask(
            input_ids, tokenizer, mlm_config, special_token_ids=set()
        )
        masked_count = sum(
            1 for i in range(len(input_ids)) if masked[i] == tokenizer.mask_token_id
        )
        assert masked_count >= 1

    def test_empty_input(self, tokenizer):
        """Empty input returns empty outputs."""
        masked, labels = replace_with_mask([], tokenizer, MLMConfig())
        assert masked == []
        assert labels == []


# ---------------------------------------------------------------------------
# replace_with_random_token
# ---------------------------------------------------------------------------


class TestReplaceWithRandomToken:
    """Tests for replace_with_random_token."""

    def test_replaced_tokens_in_vocab_range(self, tokenizer):
        """Random replacements are within [0, vocab_size)."""
        input_ids = [5, 10, 15, 20, 25]
        mlm_config = MLMConfig(mlm_probability=1.0)
        noised, labels = replace_with_random_token(
            input_ids, tokenizer, mlm_config, special_token_ids=set()
        )
        for i in range(len(input_ids)):
            assert 0 <= noised[i] < tokenizer.vocab_size
            assert labels[i] == input_ids[i]

    def test_empty_input(self, tokenizer):
        """Empty input returns empty outputs."""
        noised, labels = replace_with_random_token([], tokenizer, MLMConfig())
        assert noised == []
        assert labels == []


# ---------------------------------------------------------------------------
# keep_original_tokens
# ---------------------------------------------------------------------------


class TestKeepOriginalTokens:
    """Tests for keep_original_tokens."""

    def test_input_unchanged(self, tokenizer):
        """Input IDs are not modified."""
        input_ids = [5, 10, 15, 20, 25]
        mlm_config = MLMConfig(mlm_probability=1.0)
        unchanged, _ = keep_original_tokens(
            input_ids, tokenizer, mlm_config, special_token_ids=set()
        )
        assert unchanged == input_ids

    def test_labels_set_for_selected(self, tokenizer):
        """Labels are set to original IDs for selected positions."""
        input_ids = [5, 10, 15, 20, 25]
        mlm_config = MLMConfig(mlm_probability=1.0)
        _, labels = keep_original_tokens(
            input_ids, tokenizer, mlm_config, special_token_ids=set()
        )
        masked_count = sum(1 for l in labels if l != -100)
        assert masked_count >= 1

    def test_empty_input(self, tokenizer):
        """Empty input returns empty outputs."""
        unchanged, labels = keep_original_tokens([], tokenizer, MLMConfig())
        assert unchanged == []
        assert labels == []


# ---------------------------------------------------------------------------
# resolve_protected_token_ids
# ---------------------------------------------------------------------------


class TestResolveProtectedTokenIds:
    """Tests for resolve_protected_token_ids."""

    def test_none_returns_empty(self, tokenizer):
        """None input returns empty set."""
        assert resolve_protected_token_ids(tokenizer, None) == set()

    def test_empty_set_returns_empty(self, tokenizer):
        """Empty set input returns empty set."""
        assert resolve_protected_token_ids(tokenizer, set()) == set()

    def test_known_tokens_resolved(self, tokenizer):
        """Tokens present in vocab are resolved to their IDs."""
        vocab = tokenizer.get_vocab()
        token = next(iter(vocab))
        result = resolve_protected_token_ids(tokenizer, {token})
        assert vocab[token] in result


# ---------------------------------------------------------------------------
# _parse_reaction_molecules
# ---------------------------------------------------------------------------


class TestParseReactionMolecules:
    """Tests for _parse_reaction_molecules."""

    def test_simple_reaction(self):
        """A simple A>>B reaction parses into two molecules."""
        smiles_list, mols = _parse_reaction_molecules("CCO>>CCO")
        assert smiles_list == ["CCO", "CCO"]
        assert len(mols) == 2
        assert all(m is not None for m in mols)

    def test_multi_fragment_reaction(self):
        """Multi-fragment reaction splits on dots."""
        smiles_list, mols = _parse_reaction_molecules("CC.O>>CC.O")
        assert smiles_list == ["CC", "O", "CC", "O"]
        assert len(mols) == 4

    def test_invalid_smiles_returns_none_mol(self):
        """Invalid SMILES fragment yields None in the mol list."""
        smiles_list, mols = _parse_reaction_molecules("CCO>>invalid_smiles_xyz")
        assert smiles_list == ["CCO", "invalid_smiles_xyz"]
        assert mols[0] is not None
        assert mols[1] is None

    def test_no_reaction_arrow(self):
        """Missing '>>' returns empty lists."""
        smiles_list, mols = _parse_reaction_molecules("CCO")
        assert smiles_list == []
        assert mols == []

    def test_empty_string(self):
        """Empty string returns empty lists."""
        smiles_list, mols = _parse_reaction_molecules("")
        assert smiles_list == []
        assert mols == []

    def test_extra_arrows(self):
        """Multiple '>>' returns empty lists (len(parts) != 2)."""
        smiles_list, mols = _parse_reaction_molecules("CCO>>CCO>>CCO")
        assert smiles_list == []
        assert mols == []


# ---------------------------------------------------------------------------
# _build_atom_token_map
# ---------------------------------------------------------------------------


class TestBuildAtomTokenMap:
    """Tests for _build_atom_token_map."""

    def test_returns_three_part_tuple(self, tokenizer):
        """Returns (token_to_mol_atom, mol_atom_to_token, atom_token_positions)."""
        text = "CCO>>CCO"
        encoding = tokenizer(
            text, max_length=64, padding=False, truncation=True, return_tensors=None
        )
        input_ids = encoding["input_ids"]
        t2ma, ma2t, positions = _build_atom_token_map(input_ids)
        assert isinstance(t2ma, dict)
        assert isinstance(ma2t, dict)
        assert isinstance(positions, list)

    def test_atom_positions_are_subset_of_input(self, tokenizer):
        """All atom positions are valid indices into input_ids."""
        text = "CCO>>CCO"
        encoding = tokenizer(
            text, max_length=64, padding=False, truncation=True, return_tensors=None
        )
        input_ids = encoding["input_ids"]
        _, _, positions = _build_atom_token_map(input_ids)
        for pos in positions:
            assert 0 <= pos < len(input_ids)

    def test_empty_input(self):
        """Empty input returns empty structures."""
        t2ma, ma2t, positions = _build_atom_token_map([])
        assert t2ma == {}
        assert ma2t == {}
        assert positions == []

    def test_bidirectional_mapping_consistency(self, tokenizer):
        """token_to_mol_atom and mol_atom_to_token are inverses."""
        text = "CCO>>CCO"
        encoding = tokenizer(
            text, max_length=64, padding=False, truncation=True, return_tensors=None
        )
        input_ids = encoding["input_ids"]
        t2ma, ma2t, _ = _build_atom_token_map(input_ids)
        for token_pos, (mol_id, atom_idx) in t2ma.items():
            assert ma2t[(mol_id, atom_idx)] == token_pos


# ---------------------------------------------------------------------------
# _select_graph_neighborhood
# ---------------------------------------------------------------------------


class TestSelectGraphNeighborhood:
    """Tests for _select_graph_neighborhood."""

    def test_single_atom_molecule(self):
        """Single-atom molecule returns just the seed."""
        mol = Chem.MolFromSmiles("C")
        result = _select_graph_neighborhood(mol, 0, 3)
        assert result == {0}

    def test_span_size_one(self):
        """Span size 1 returns only the seed atom."""
        mol = Chem.MolFromSmiles("CCCC")
        result = _select_graph_neighborhood(mol, 1, 1)
        assert result == {1}

    def test_span_covers_neighborhood(self):
        """Span size >= num_atoms returns all atoms reachable from seed."""
        mol = Chem.MolFromSmiles("CCCC")
        result = _select_graph_neighborhood(mol, 0, 10)
        assert result == {0, 1, 2, 3}

    def test_seed_out_of_range(self):
        """Seed index >= num_atoms returns {seed_idx}."""
        mol = Chem.MolFromSmiles("CC")
        result = _select_graph_neighborhood(mol, 10, 3)
        assert result == {10}

    def test_result_size_capped(self):
        """Result set size does not exceed span_size."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        result = _select_graph_neighborhood(mol, 0, 3)
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# _get_plausible_replacement
# ---------------------------------------------------------------------------


class TestGetPlausibleReplacement:
    """Tests for _get_plausible_replacement."""

    def test_known_atom_returns_valid_id(self, tokenizer):
        """A known atom token returns a valid vocab ID."""
        result = _get_plausible_replacement("C", tokenizer)
        assert result in tokenizer.get_vocab().values()

    def test_unknown_token_returns_mask_id(self, tokenizer):
        """An unknown token falls back to mask_token_id."""
        result = _get_plausible_replacement("not_a_real_token", tokenizer)
        assert result == tokenizer.mask_token_id

    def test_halogen_substitution(self, tokenizer):
        """Halogen tokens should return a different halogen or valid ID."""
        result = _get_plausible_replacement("F", tokenizer)
        vocab = tokenizer.get_vocab()
        assert result in vocab.values()

    def test_aromatic_carbon_substitution(self, tokenizer):
        """Aromatic carbon returns a valid replacement."""
        result = _get_plausible_replacement("c", tokenizer)
        vocab = tokenizer.get_vocab()
        assert result in vocab.values()

    def test_cached_vocab_matches_uncached(self, tokenizer):
        """Passing a cached vocab dict produces the same result as calling get_vocab()."""
        vocab = tokenizer.get_vocab()
        # Test with a known atom token
        result_cached = _get_plausible_replacement("C", tokenizer, vocab=vocab)
        result_uncached = _get_plausible_replacement("C", tokenizer)
        assert result_cached in vocab.values()
        assert result_uncached in vocab.values()

    def test_cached_vocab_unknown_token_returns_mask_id(self, tokenizer):
        """Unknown token with cached vocab falls back to mask_token_id."""
        vocab = tokenizer.get_vocab()
        result = _get_plausible_replacement("not_a_real_token", tokenizer, vocab=vocab)
        assert result == tokenizer.mask_token_id


# ---------------------------------------------------------------------------
# apply_span_mlm_masking
# ---------------------------------------------------------------------------


class TestApplySpanMlmMasking:
    """Tests for apply_span_mlm_masking."""

    def test_returns_correct_lengths(self, tokenizer):
        """Output lists have same length as input."""
        text = "CCO>>CCO"
        encoding = tokenizer(
            text, max_length=64, padding=False, truncation=True, return_tensors=None
        )
        input_ids = encoding["input_ids"]
        config = SpanMLMConfig(mlm_probability=0.5)
        masked, labels = apply_span_mlm_masking(
            input_ids,
            tokenizer,
            config,
            text,
            special_token_ids=set(tokenizer.all_special_ids),
        )
        assert len(masked) == len(input_ids)
        assert len(labels) == len(input_ids)

    def test_labels_negative_100_for_non_masked(self, tokenizer):
        """Non-masked positions have label -100."""
        text = "CCO>>CCO"
        encoding = tokenizer(
            text, max_length=64, padding=False, truncation=True, return_tensors=None
        )
        input_ids = encoding["input_ids"]
        config = SpanMLMConfig(mlm_probability=0.1)
        _, labels = apply_span_mlm_masking(
            input_ids,
            tokenizer,
            config,
            text,
            special_token_ids=set(tokenizer.all_special_ids),
        )
        assert any(l == -100 for l in labels)

    def test_empty_input(self, tokenizer):
        """Empty input returns empty outputs."""
        config = SpanMLMConfig()
        masked, labels = apply_span_mlm_masking(
            [], tokenizer, config, "", special_token_ids=set()
        )
        assert masked == []
        assert labels == []

    def test_no_atom_tokens(self, tokenizer):
        """Input with no atom tokens returns unchanged."""
        # Use only special tokens
        input_ids = list(tokenizer.all_special_ids)[:3]
        config = SpanMLMConfig()
        masked, labels = apply_span_mlm_masking(
            input_ids,
            tokenizer,
            config,
            ">>",
            special_token_ids=set(tokenizer.all_special_ids),
        )
        assert masked == input_ids
        assert all(l == -100 for l in labels)

    def test_input_not_mutated(self, tokenizer):
        """Original input_ids list is not modified."""
        text = "CCO>>CCO"
        encoding = tokenizer(
            text, max_length=64, padding=False, truncation=True, return_tensors=None
        )
        input_ids = encoding["input_ids"]
        original = input_ids.copy()
        config = SpanMLMConfig(mlm_probability=0.5)
        apply_span_mlm_masking(
            input_ids,
            tokenizer,
            config,
            text,
            special_token_ids=set(tokenizer.all_special_ids),
        )
        assert input_ids == original


# ---------------------------------------------------------------------------
# build_albert_model
# ---------------------------------------------------------------------------


class TestBuildAlbertModel:
    """Tests for build_albert_model."""

    def test_returns_albert_for_masked_lm(self):
        """build_albert_model returns an AlbertForMaskedLM instance."""
        config = ModelConfig(
            vocab_size=100,
            embedding_size=16,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
        )
        model = build_albert_model(config)
        assert isinstance(model, AlbertForMaskedLM)

    def test_model_has_parameters(self):
        """Built model has trainable parameters."""
        config = ModelConfig(
            vocab_size=100,
            embedding_size=16,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
        )
        model = build_albert_model(config)
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0


# ---------------------------------------------------------------------------
# MLMDataset
# ---------------------------------------------------------------------------


class TestMLMDataset:
    """Tests for MLMDataset."""

    def test_len_matches_texts(self, tokenizer):
        """__len__ returns the number of texts."""
        texts = ["CCO>>CCO", "CC>>CC"]
        dataset = MLMDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=64,
            use_random_smiles=True,
            use_canonical_smiles=False,
        )
        assert len(dataset) == len(texts)

    def test_getitem_returns_dict_with_required_keys(self, tokenizer):
        """__getitem__ returns dict with required tensor keys."""
        texts = ["CCO>>CCO"]
        dataset = MLMDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=64,
            use_random_smiles=True,
            use_canonical_smiles=False,
        )
        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "token_type_ids" in sample
        assert "labels" in sample

    def test_getitem_tensor_shapes(self, tokenizer):
        """All returned tensors have shape (max_length,)."""
        texts = ["CCO>>CCO"]
        max_length = 32
        dataset = MLMDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=max_length,
            use_random_smiles=True,
            use_canonical_smiles=False,
        )
        sample = dataset[0]
        for key in ["input_ids", "attention_mask", "token_type_ids", "labels"]:
            assert sample[key].shape == (max_length,)

    def test_invalid_masking_mode_raises(self, tokenizer):
        """Invalid masking_mode raises ValueError."""
        with pytest.raises(ValueError, match="masking_mode"):
            MLMDataset(
                texts=["CCO>>CCO"],
                tokenizer=tokenizer,
                mlm_config=MLMConfig(),
                masking_mode="invalid",
            )

    def test_empty_texts(self, tokenizer):
        """Empty texts list creates a dataset with len 0."""
        dataset = MLMDataset(
            texts=[],
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=32,
            use_random_smiles=True,
            use_canonical_smiles=False,
        )
        assert len(dataset) == 0

    def test_decode_sample_returns_dict(self, tokenizer):
        """decode_sample returns a dict with expected keys."""
        texts = ["CCO>>CCO"]
        dataset = MLMDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=64,
            use_random_smiles=True,
            use_canonical_smiles=False,
        )
        result = dataset.decode_sample(0, print_output=False)
        assert "original" in result
        assert "masked" in result

    def test_decode_sample_span_mode_uses_span_masking(self, tokenizer):
        """decode_sample uses span masking when masking_mode is 'span'."""
        texts = ["CCO>>CCO"]
        dataset = MLMDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=64,
            use_random_smiles=True,
            use_canonical_smiles=False,
            masking_mode="span",
            span_mlm_config=SpanMLMConfig(),
        )
        with (
            patch(
                "model_training_scripts.albert_mapper_unuspervised_training.apply_span_mlm_masking"
            ) as mock_span,
            patch(
                "model_training_scripts.albert_mapper_unuspervised_training.apply_mlm_masking"
            ) as mock_random,
        ):
            mock_span.return_value = ([0], [-100])
            dataset.decode_sample(0, print_output=False)
            mock_span.assert_called_once()
            mock_random.assert_not_called()

    def test_decode_sample_random_mode_uses_random_masking(self, tokenizer):
        """decode_sample uses random masking when masking_mode is 'random'."""
        texts = ["CCO>>CCO"]
        dataset = MLMDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=64,
            use_random_smiles=True,
            use_canonical_smiles=False,
            masking_mode="random",
        )
        with (
            patch(
                "model_training_scripts.albert_mapper_unuspervised_training.apply_span_mlm_masking"
            ) as mock_span,
            patch(
                "model_training_scripts.albert_mapper_unuspervised_training.apply_mlm_masking"
            ) as mock_random,
        ):
            mock_random.return_value = ([0], [-100])
            dataset.decode_sample(0, print_output=False)
            mock_random.assert_called_once()
            mock_span.assert_not_called()


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    """Tests for TrainingConfig dead field removal and new fields."""

    def test_fp16_field_removed(self):
        """fp16 field is no longer a model field."""
        assert "fp16" not in TrainingConfig.model_fields

    def test_save_steps_field_removed(self):
        """save_steps field is no longer a model field."""
        assert "save_steps" not in TrainingConfig.model_fields

    def test_save_best_model_default_true(self):
        """save_best_model defaults to True."""
        config = TrainingConfig()
        assert config.save_best_model is True

    def test_early_stopping_patience_default_zero(self):
        """early_stopping_patience defaults to 0 (disabled)."""
        config = TrainingConfig()
        assert config.early_stopping_patience == 0

    def test_early_stopping_min_delta_default_zero(self):
        """early_stopping_min_delta defaults to 0.0."""
        config = TrainingConfig()
        assert config.early_stopping_min_delta == 0.0

    def test_negative_early_stopping_patience_rejected(self):
        """Negative early_stopping_patience raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            TrainingConfig(early_stopping_patience=-1)

    def test_negative_early_stopping_min_delta_rejected(self):
        """Negative early_stopping_min_delta raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            TrainingConfig(early_stopping_min_delta=-0.01)

    def test_use_amp_default_false(self):
        """use_amp defaults to False."""
        assert TrainingConfig().use_amp is False

    def test_amp_dtype_default_float16(self):
        """amp_dtype defaults to bfloat16."""
        assert TrainingConfig().amp_dtype == "bfloat16"

    def test_amp_dtype_bf16(self):
        """amp_dtype accepts bfloat16."""
        config = TrainingConfig(amp_dtype="bfloat16")
        assert config.amp_dtype == "bfloat16"

    def test_amp_dtype_fp16(self):
        """amp_dtype accepts float16."""
        config = TrainingConfig(amp_dtype="float16")
        assert config.amp_dtype == "float16"

    def test_gradient_accumulation_steps_default_one(self):
        """gradient_accumulation_steps defaults to 1."""
        assert TrainingConfig().gradient_accumulation_steps == 1

    def test_gradient_accumulation_steps_zero_rejected(self):
        """gradient_accumulation_steps=0 raises ValueError."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            TrainingConfig(gradient_accumulation_steps=0)

    def test_compile_model_default_false(self):
        """compile_model defaults to False."""
        assert TrainingConfig().compile_model is False

    def test_deterministic_default_true(self):
        """deterministic defaults to True."""
        assert TrainingConfig().deterministic is True


class TestUnwrapModel:
    """Tests for _unwrap_model helper."""

    def test_unwrap_non_compiled_returns_same(self):
        """_unwrap_model returns the same module when not compiled."""
        from torch import nn

        model = nn.Linear(10, 10)
        assert _unwrap_model(model) is model

    def test_unwrap_compiled_returns_orig(self):
        """_unwrap_model returns _orig_mod when the model is compiled."""
        from torch import nn

        original = nn.Linear(10, 10)
        # Simulate the wrapper that torch.compile creates
        compiled = type("FakeCompiled", (nn.Module,), {})()
        compiled._orig_mod = original
        assert _unwrap_model(compiled) is original


# ---------------------------------------------------------------------------
# AlbertTrainer checkpoint resume
# ---------------------------------------------------------------------------


class TestAlbertTrainerCheckpoint:
    """Tests for AlbertTrainer checkpoint loading."""

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock DataLoader with __len__."""
        from unittest.mock import MagicMock

        dl = MagicMock()
        dl.__len__ = MagicMock(return_value=10)
        return dl

    @pytest.fixture
    def tiny_model(self):
        """Create a tiny AlbertForMaskedLM for testing."""
        config = AlbertConfig(
            vocab_size=100,
            embedding_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=2,
            num_hidden_groups=1,
            max_position_embeddings=32,
            type_vocab_size=2,
        )
        return AlbertForMaskedLM(config)

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for AlbertTrainer."""
        from unittest.mock import MagicMock

        tok = MagicMock()
        tok.save_pretrained = MagicMock(return_value=None)
        return tok

    def test_load_checkpoint_restores_epoch(
        self, tiny_model, mock_dataloader, mock_tokenizer, tmp_path
    ):
        """_load_checkpoint sets start_epoch to saved epoch + 1."""
        import torch as _torch

        training_config = TrainingConfig(output_dir=str(tmp_path), num_epochs=5)
        trainer = AlbertTrainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            training_config=training_config,
            tokenizer=mock_tokenizer,
        )

        ckpt_path = tmp_path / "ckpt.pt"
        _torch.save(
            {
                "epoch": 3,
                "model_state_dict": tiny_model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": trainer.scheduler.state_dict(),
                "best_val_loss": 0.5,
            },
            str(ckpt_path),
        )

        trainer._load_checkpoint(str(ckpt_path))
        assert trainer.start_epoch == 4
        assert trainer.best_val_loss == 0.5

    def test_load_checkpoint_file_not_found(
        self, tiny_model, mock_dataloader, mock_tokenizer, tmp_path
    ):
        """_load_checkpoint raises FileNotFoundError for missing file."""
        training_config = TrainingConfig(output_dir=str(tmp_path))
        trainer = AlbertTrainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            training_config=training_config,
            tokenizer=mock_tokenizer,
        )
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            trainer._load_checkpoint("/nonexistent/checkpoint.pt")


# ---------------------------------------------------------------------------
# AlbertTrainer early stopping logic
# ---------------------------------------------------------------------------


class TestAlbertTrainerEarlyStopping:
    """Tests for AlbertTrainer early stopping and best-model saving logic."""

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock DataLoader with __len__."""
        from unittest.mock import MagicMock

        dl = MagicMock()
        dl.__len__ = MagicMock(return_value=10)
        return dl

    @pytest.fixture
    def tiny_model(self):
        """Create a tiny AlbertForMaskedLM for testing."""
        config = AlbertConfig(
            vocab_size=100,
            embedding_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=2,
            num_hidden_groups=1,
            max_position_embeddings=32,
            type_vocab_size=2,
        )
        return AlbertForMaskedLM(config)

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for AlbertTrainer."""
        from unittest.mock import MagicMock

        tok = MagicMock()
        tok.save_pretrained = MagicMock(return_value=None)
        return tok

    def test_early_stopping_triggers(
        self, tiny_model, mock_dataloader, mock_tokenizer, tmp_path, monkeypatch
    ):
        """Training stops when patience is exceeded."""

        training_config = TrainingConfig(
            output_dir=str(tmp_path),
            num_epochs=10,
            early_stopping_patience=2,
            save_best_model=False,
        )
        trainer = AlbertTrainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            training_config=training_config,
            tokenizer=mock_tokenizer,
        )

        monkeypatch.setattr(trainer, "train_epoch", lambda epoch: 1.0)
        monkeypatch.setattr(trainer, "evaluate", lambda: 2.0)
        monkeypatch.setattr(tiny_model, "save_pretrained", lambda path: None)
        monkeypatch.setattr(mock_tokenizer, "save_pretrained", lambda path: None)

        trainer.train()
        assert trainer.epochs_without_improvement >= 2

    def test_best_model_saved_on_improvement(
        self, tiny_model, mock_dataloader, mock_tokenizer, tmp_path, monkeypatch
    ):
        """Best model is saved when validation loss improves."""

        training_config = TrainingConfig(
            output_dir=str(tmp_path),
            num_epochs=1,
            save_best_model=True,
        )
        trainer = AlbertTrainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            training_config=training_config,
            tokenizer=mock_tokenizer,
        )

        monkeypatch.setattr(trainer, "train_epoch", lambda epoch: 1.0)
        monkeypatch.setattr(trainer, "evaluate", lambda: 0.5)
        monkeypatch.setattr(tiny_model, "save_pretrained", lambda path: None)
        monkeypatch.setattr(mock_tokenizer, "save_pretrained", lambda path: None)

        trainer.train()
        assert (tmp_path / "best_model.pt").exists()

    def test_best_model_not_saved_when_disabled(
        self, tiny_model, mock_dataloader, mock_tokenizer, tmp_path, monkeypatch
    ):
        """Best model file is not created when save_best_model=False."""
        training_config = TrainingConfig(
            output_dir=str(tmp_path),
            num_epochs=1,
            save_best_model=False,
        )
        trainer = AlbertTrainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            training_config=training_config,
            tokenizer=mock_tokenizer,
        )

        monkeypatch.setattr(trainer, "train_epoch", lambda epoch: 1.0)
        monkeypatch.setattr(trainer, "evaluate", lambda: 0.5)
        monkeypatch.setattr(tiny_model, "save_pretrained", lambda path: None)
        monkeypatch.setattr(mock_tokenizer, "save_pretrained", lambda path: None)

        trainer.train()
        assert not (tmp_path / "best_model.pt").exists()

    def test_improvement_tracked_without_save_best_model(
        self, tiny_model, mock_dataloader, mock_tokenizer, tmp_path, monkeypatch
    ):
        """Early stopping counter resets on improvement even when save_best_model=False."""
        training_config = TrainingConfig(
            output_dir=str(tmp_path),
            num_epochs=3,
            save_best_model=False,
            early_stopping_patience=5,
        )
        trainer = AlbertTrainer(
            model=tiny_model,
            train_dataloader=mock_dataloader,
            training_config=training_config,
            tokenizer=mock_tokenizer,
        )

        call_count = [0]
        val_losses = [2.0, 1.0, 0.5]

        def mock_evaluate():
            loss = val_losses[call_count[0]]
            call_count[0] += 1
            return loss

        monkeypatch.setattr(trainer, "train_epoch", lambda epoch: 1.0)
        monkeypatch.setattr(trainer, "evaluate", mock_evaluate)
        monkeypatch.setattr(tiny_model, "save_pretrained", lambda path: None)
        monkeypatch.setattr(mock_tokenizer, "save_pretrained", lambda path: None)

        trainer.train()
        assert trainer.epochs_without_improvement == 0
        assert trainer.best_val_loss == 0.5
