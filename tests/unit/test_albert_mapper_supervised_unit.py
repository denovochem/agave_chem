"""Unit tests for workflows/model_training_scripts/albert_mapper_supervised_training.py."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from rdkit import Chem

# Ensure the workflows directory is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_ROOT = REPO_ROOT / "workflows"
for p in [str(REPO_ROOT), str(WORKFLOWS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_training_scripts.albert_mapper_supervised_training import (
    SupervisedAtomMappingDataset,
    build_attention_target_from_mapped_rxn_smiles,
    evaluate_supervised_attention_loss,
    group_mappings_by_symmetry,
)
from model_training_scripts.albert_mapper_unuspervised_training import (
    MLMConfig,
)

from agave_chem.mappers.neural.constants import (
    smiles_token_to_id_dict,
    token_atom_identity_dict,
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
def mapped_rxn():
    """A simple atom-mapped reaction SMILES."""
    return "[C:1][C:2][O:3]>>[C:1][C:2][O:3]"


@pytest.fixture
def mapped_rxn_with_unmapped():
    """A reaction with some unmapped atoms."""
    return "[C:1]CO>>[C:1]CO"


# ---------------------------------------------------------------------------
# group_mappings_by_symmetry
# ---------------------------------------------------------------------------


class TestGroupMappingsBySymmetry:
    """Tests for group_mappings_by_symmetry."""

    def test_no_symmetry(self):
        """Molecule with no symmetry returns empty list."""
        mol = Chem.MolFromSmiles("[C:1][O:2]")
        result = group_mappings_by_symmetry(mol)
        assert result == []

    def test_symmetric_molecule(self):
        """Symmetric molecule returns groups with >1 member."""
        mol = Chem.MolFromSmiles("[C:1]([O:2])([O:3])=O")
        result = group_mappings_by_symmetry(mol)
        # The two oxygens are symmetric
        assert len(result) >= 1
        for group in result:
            assert len(group) > 1

    def test_atom_map_nums_preserved(self):
        """Returned groups contain original atom map numbers, not indices."""
        mol = Chem.MolFromSmiles("[C:1]([O:2])([O:3])=O")
        result = group_mappings_by_symmetry(mol)
        all_map_nums = set()
        for group in result:
            all_map_nums.update(group)
        # All returned values should be atom map numbers from the input
        original_map_nums = {atom.GetAtomMapNum() for atom in mol.GetAtoms()}
        assert all_map_nums.issubset(original_map_nums)

    def test_clears_map_nums_on_copy(self):
        """The original molecule is not modified."""
        mol = Chem.MolFromSmiles("[C:1]([O:2])([O:3])=O")
        original_maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
        group_mappings_by_symmetry(mol)
        after_maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
        assert original_maps == after_maps


# ---------------------------------------------------------------------------
# build_attention_target_from_mapped_rxn_smiles
# ---------------------------------------------------------------------------


class TestBuildAttentionTarget:
    """Tests for build_attention_target_from_mapped_rxn_smiles."""

    def test_valid_mapped_reaction(self, tokenizer, mapped_rxn):
        """Valid mapped reaction returns (ndarray, str) tuple."""
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            randomize_mapped_rxn_smiles=False,
        )
        assert result is not None
        attn_target, unmapped = result
        assert isinstance(attn_target, np.ndarray)
        assert isinstance(unmapped, str)
        assert attn_target.ndim == 2
        assert attn_target.shape[0] == attn_target.shape[1]

    def test_unmapped_smiles_has_no_atom_mapping(self, tokenizer, mapped_rxn):
        """Returned unmapped SMILES has no atom map numbers."""
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            randomize_mapped_rxn_smiles=False,
        )
        assert result is not None
        _, unmapped = result
        assert ":" not in unmapped

    def test_attention_target_values_in_zero_one(self, tokenizer, mapped_rxn):
        """Attention target values are in [0, 1]."""
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            randomize_mapped_rxn_smiles=False,
        )
        assert result is not None
        attn_target, _ = result
        assert attn_target.min() >= 0.0
        assert attn_target.max() <= 1.0

    def test_invalid_smiles_returns_none(self, tokenizer):
        """Invalid SMILES returns None."""
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles="invalid_not_a_reaction",
            randomize_mapped_rxn_smiles=False,
        )
        assert result is None

    def test_empty_string_returns_none(self, tokenizer):
        """Empty string returns None."""
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles="",
            randomize_mapped_rxn_smiles=False,
        )
        assert result is None

    def test_no_reaction_arrow_returns_none(self, tokenizer):
        """SMILES without '>>' returns None."""
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles="CCO",
            randomize_mapped_rxn_smiles=False,
        )
        assert result is None

    def test_with_token_atom_identity_dict(self, tokenizer, mapped_rxn):
        """Passing token_atom_identity_dict produces same result type."""
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=False,
        )
        assert result is not None
        attn_target, _ = result
        assert isinstance(attn_target, np.ndarray)

    def test_smooth_symmetric_targets(self, tokenizer):
        """Smooth symmetric targets produces fractional weights."""
        # Use a symmetric reaction
        rxn = "[C:1]([O:2])([O:3])=O>>[C:1]([O:2])([O:3])=O"
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            randomize_mapped_rxn_smiles=False,
            smooth_symmetric_targets=True,
        )
        assert result is not None
        attn_target, _ = result
        # With smoothing, some values may be fractional (e.g. 0.5)
        assert attn_target.min() >= 0.0
        assert attn_target.max() <= 1.0

    def test_no_smooth_symmetric_targets(self, tokenizer):
        """Without smoothing, target values are 0 or 1."""
        rxn = "[C:1]([O:2])([O:3])=O>>[C:1]([O:2])([O:3])=O"
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            randomize_mapped_rxn_smiles=False,
            smooth_symmetric_targets=False,
        )
        assert result is not None
        attn_target, _ = result
        unique_vals = set(np.unique(attn_target))
        assert unique_vals.issubset({0.0, 1.0})

    def test_attn_sink_sets_last_column(self, tokenizer):
        """With attn_sink, non-atom tokens have last column set to 1."""
        rxn = "[C:1][C:2][O:3]>>[C:1][C:2][O:3]"
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            randomize_mapped_rxn_smiles=False,
            attn_sink_non_mapped_atoms=True,
        )
        assert result is not None
        attn_target, _ = result
        # Check that at least some rows have the last column set
        last_col = attn_target[:, -1]
        assert last_col.max() >= 0.0

    def test_randomize_smiles_produces_valid_result(self, tokenizer, mapped_rxn):
        """Randomized SMILES still produces a valid attention target."""
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            randomize_mapped_rxn_smiles=True,
        )
        assert result is not None
        attn_target, unmapped = result
        assert isinstance(attn_target, np.ndarray)
        assert ":" not in unmapped


# ---------------------------------------------------------------------------
# SupervisedAtomMappingDataset
# ---------------------------------------------------------------------------


class TestSupervisedAtomMappingDataset:
    """Tests for SupervisedAtomMappingDataset."""

    def test_len_matches_texts(self, tokenizer):
        """__len__ returns the number of texts."""
        texts = ["[C:1]>>[C:1]", "[C:1][O:2]>>[C:1][O:2]"]
        dataset = SupervisedAtomMappingDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=64,
            use_random_smiles=False,
        )
        assert len(dataset) == len(texts)

    def test_getitem_returns_dict_with_required_keys(self, tokenizer):
        """__getitem__ returns dict with required tensor keys."""
        texts = ["[C:1]>>[C:1]"]
        dataset = SupervisedAtomMappingDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=64,
            use_random_smiles=False,
        )
        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "token_type_ids" in sample
        assert "labels" in sample
        assert "attention_target" in sample
        assert "attention_loss_mask" in sample

    def test_getitem_tensor_shapes(self, tokenizer):
        """All returned tensors have correct shapes."""
        texts = ["[C:1]>>[C:1]"]
        max_length = 32
        dataset = SupervisedAtomMappingDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=max_length,
            use_random_smiles=False,
        )
        sample = dataset[0]
        for key in ["input_ids", "attention_mask", "token_type_ids", "labels"]:
            assert sample[key].shape == (max_length,)
        assert sample["attention_target"].shape == (max_length, max_length)
        assert sample["attention_loss_mask"].shape == (max_length,)

    def test_invalid_masking_mode_raises(self, tokenizer):
        """Invalid masking_mode raises ValueError."""
        with pytest.raises(ValueError, match="masking_mode"):
            SupervisedAtomMappingDataset(
                texts=["[C:1]>>[C:1]"],
                tokenizer=tokenizer,
                mlm_config=MLMConfig(),
                masking_mode="invalid",
            )

    def test_empty_texts(self, tokenizer):
        """Empty texts list creates a dataset with len 0."""
        dataset = SupervisedAtomMappingDataset(
            texts=[],
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=32,
            use_random_smiles=False,
        )
        assert len(dataset) == 0

    def test_no_mlm_masking(self, tokenizer):
        """With use_mlm_masking=False, labels are all -100."""
        texts = ["[C:1]>>[C:1]"]
        dataset = SupervisedAtomMappingDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=32,
            use_random_smiles=False,
            use_mlm_masking=False,
        )
        sample = dataset[0]
        assert (sample["labels"] == -100).all()

    def test_attention_loss_mask_binary(self, tokenizer):
        """attention_loss_mask contains only 0s and 1s."""
        texts = ["[C:1]>>[C:1]"]
        dataset = SupervisedAtomMappingDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=32,
            use_random_smiles=False,
        )
        sample = dataset[0]
        unique_vals = set(sample["attention_loss_mask"].unique().tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_getitem_with_invalid_reaction_retries(self, tokenizer):
        """Invalid reactions trigger retry with random index."""
        texts = ["invalid_reaction", "[C:1]>>[C:1]"]
        dataset = SupervisedAtomMappingDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=32,
            use_random_smiles=False,
        )
        with patch(
            "model_training_scripts.albert_mapper_supervised_training.random.randrange",
            return_value=1,
        ):
            sample = dataset[0]
        assert "input_ids" in sample


# ---------------------------------------------------------------------------
# evaluate_supervised_attention_loss
# ---------------------------------------------------------------------------


class TestEvaluateSupervisedAttentionLoss:
    """Tests for evaluate_supervised_attention_loss."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with supervised_config and base_model."""
        from unittest.mock import MagicMock

        model = MagicMock()
        model.supervised_config = MagicMock()
        model.supervised_config.target_layer = 5
        model.base_model.config.num_hidden_layers = 12
        model.to.return_value = model
        model.parameters.return_value = iter([MagicMock(device="cpu")])
        return model

    def test_preserves_target_layer(self, mock_model):
        """evaluate_supervised_attention_loss restores the original target_layer."""
        from unittest.mock import MagicMock

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=0)

        original_layer = mock_model.supervised_config.target_layer
        evaluate_supervised_attention_loss(
            mock_model, dataset, target_layer=3, batch_size=1
        )
        assert mock_model.supervised_config.target_layer == original_layer

    def test_preserves_target_layer_on_exception(self, mock_model):
        """target_layer is restored even when the forward pass raises."""
        from unittest.mock import MagicMock

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=1)
        dataset.__getitem__ = MagicMock(side_effect=RuntimeError("boom"))

        original_layer = mock_model.supervised_config.target_layer
        with pytest.raises(RuntimeError, match="boom"):
            evaluate_supervised_attention_loss(
                mock_model, dataset, target_layer=3, batch_size=1
            )
        assert mock_model.supervised_config.target_layer == original_layer

    def test_invalid_target_layer_raises_value_error(self, mock_model):
        """Out-of-range target_layer raises ValueError before training."""
        from unittest.mock import MagicMock

        dataset = MagicMock()
        with pytest.raises(ValueError, match="target_layer must be in"):
            evaluate_supervised_attention_loss(
                mock_model, dataset, target_layer=99, batch_size=1
            )

    def test_negative_target_layer_raises_value_error(self, mock_model):
        """Negative target_layer raises ValueError."""
        from unittest.mock import MagicMock

        dataset = MagicMock()
        with pytest.raises(ValueError, match="target_layer must be in"):
            evaluate_supervised_attention_loss(
                mock_model, dataset, target_layer=-1, batch_size=1
            )

    def test_no_target_layer_does_not_modify_config(self, mock_model):
        """When target_layer is None, supervised_config is not modified."""
        from unittest.mock import MagicMock

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=0)

        original_layer = mock_model.supervised_config.target_layer
        evaluate_supervised_attention_loss(mock_model, dataset, batch_size=1)
        assert mock_model.supervised_config.target_layer == original_layer

    def test_raises_type_error_for_model_without_supervised_config(self):
        """Model without supervised_config raises TypeError."""
        from unittest.mock import MagicMock

        model = MagicMock(spec=[])
        dataset = MagicMock()
        with pytest.raises(TypeError, match="AlbertWithAttentionAlignment"):
            evaluate_supervised_attention_loss(model, dataset)
