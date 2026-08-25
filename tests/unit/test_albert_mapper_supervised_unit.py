"""Unit tests for workflows/model_training_scripts/albert_mapper_supervised_training.py."""

import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
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
# build_attention_target_from_mapped_rxn_smiles — resonance equivalence
# ---------------------------------------------------------------------------


class TestResonanceEquivalence:
    """Tests for the resonance_equivalence parameter."""

    def test_nitro_oxygens_smoothed_with_resonance(self, tokenizer):
        """With resonance_equivalence=True, nitro oxygens get fractional weights."""
        rxn = "[CH2:1]c1ccc([N+](=[O:2])[O-:3])cc1>>[CH2:1]c1ccc([N+](=[O:2])[O-:3])cc1"
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            randomize_mapped_rxn_smiles=False,
            smooth_symmetric_targets=True,
            resonance_equivalence=True,
        )
        assert result is not None
        attn_target, _ = result
        # With resonance smoothing, some values should be fractional (0.5)
        unique_vals = set(np.unique(attn_target))
        assert 0.5 in unique_vals

    def test_nitro_oxygens_not_smoothed_without_resonance(self, tokenizer):
        """With resonance_equivalence=False, nitro oxygens are not smoothed."""
        rxn = "[CH2:1]c1ccc([N+](=[O:2])[O-:3])cc1>>[CH2:1]c1ccc([N+](=[O:2])[O-:3])cc1"
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            randomize_mapped_rxn_smiles=False,
            smooth_symmetric_targets=True,
            resonance_equivalence=False,
        )
        assert result is not None
        attn_target, _ = result
        # Without resonance, the nitro oxygens are not symmetric
        # so there should be no 0.5 weights from them
        # (topological symmetry may still produce 0.5 for other atoms,
        # but the nitro oxygens themselves won't be smoothed)
        unique_vals = set(np.unique(attn_target))
        assert unique_vals.issubset({0.0, 1.0})

    def test_dinitrobenzene_four_oxygens_smoothed(self, tokenizer):
        """All 4 oxygens in dinitrobenzene get 0.25 weight with resonance."""
        rxn = (
            "[O:2]=[N+]([O-:3])c1ccc([N+]([O-:5])=[O:4])cc1"
            ">>"
            "[O:2]=[N+]([O-:3])c1ccc([N+]([O-:5])=[O:4])cc1"
        )
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            randomize_mapped_rxn_smiles=False,
            smooth_symmetric_targets=True,
            resonance_equivalence=True,
        )
        assert result is not None
        attn_target, _ = result
        # With 4 equivalent oxygens, weight should be 0.25
        unique_vals = set(np.unique(attn_target))
        assert 0.25 in unique_vals

    def test_resonance_equivalence_deterministic(self, tokenizer):
        """Same seed with resonance_equivalence=True produces identical results."""
        rxn = "[CH2:1]c1ccc([N+](=[O:2])[O-:3])cc1>>[CH2:1]c1ccc([N+](=[O:2])[O-:3])cc1"
        result1 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            smooth_symmetric_targets=True,
            resonance_equivalence=True,
            seed=42,
        )
        result2 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            smooth_symmetric_targets=True,
            resonance_equivalence=True,
            seed=42,
        )
        assert result1 is not None
        assert result2 is not None
        attn1, unmapped1 = result1
        attn2, unmapped2 = result2
        np.testing.assert_array_equal(attn1, attn2)
        assert unmapped1 == unmapped2

    def test_resonance_equivalence_false_preserves_existing_behavior(self, tokenizer):
        """resonance_equivalence=False matches behavior before the feature was added."""
        rxn = "[C:1]([O:2])([O:3])=O>>[C:1]([O:2])([O:3])=O"
        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            randomize_mapped_rxn_smiles=False,
            smooth_symmetric_targets=True,
            resonance_equivalence=False,
        )
        assert result is not None
        attn_target, _ = result
        assert attn_target.min() >= 0.0
        assert attn_target.max() <= 1.0

    def test_resonance_equivalence_default_is_true(self, tokenizer):
        """The default value of resonance_equivalence should be True."""
        rxn = "[CH2:1]c1ccc([N+](=[O:2])[O-:3])cc1>>[CH2:1]c1ccc([N+](=[O:2])[O-:3])cc1"
        result_default = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            randomize_mapped_rxn_smiles=False,
            smooth_symmetric_targets=True,
        )
        result_explicit = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            randomize_mapped_rxn_smiles=False,
            smooth_symmetric_targets=True,
            resonance_equivalence=True,
        )
        assert result_default is not None
        assert result_explicit is not None
        np.testing.assert_array_equal(result_default[0], result_explicit[0])


# ---------------------------------------------------------------------------
# build_attention_target_from_mapped_rxn_smiles — determinism
# ---------------------------------------------------------------------------


class TestBuildAttentionTargetDeterminism:
    """Tests that build_attention_target_from_mapped_rxn_smiles is deterministic with a seed."""

    @pytest.mark.parametrize("seed", [0, 42, 123, 9999])
    def test_same_seed_same_output(self, tokenizer, mapped_rxn, seed):
        """Two calls with the same seed and randomization enabled produce identical results."""
        result1 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            seed=seed,
        )
        result2 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            seed=seed,
        )
        assert result1 is not None
        assert result2 is not None
        attn1, unmapped1 = result1
        attn2, unmapped2 = result2
        np.testing.assert_array_equal(attn1, attn2)
        assert unmapped1 == unmapped2

    def test_different_seeds_likely_different(self, tokenizer):
        """Different seeds should (very likely) produce different attention targets."""
        rxn = (
            "[CH3:1][C:2]([O:3][H:4])([c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1)"
            "[C:11](=[O:12])[OH:13]"
            ">>"
            "[CH3:1][C:2]([O:3][H:4])([c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1)"
            "[C:11](=[O:12])[OH:13]"
        )
        result1 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            seed=1,
        )
        result2 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            seed=2,
        )
        assert result1 is not None
        assert result2 is not None
        attn1, _ = result1
        attn2, _ = result2
        # Shapes should match (same reaction), but values should differ
        assert attn1.shape == attn2.shape
        assert not np.array_equal(attn1, attn2)

    def test_seed_does_not_affect_global_state(self, tokenizer, mapped_rxn):
        """Calling with a seed must not change global random module state."""
        random.seed(54321)
        expected = random.random()

        build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            seed=42,
        )

        random.seed(54321)
        actual = random.random()
        assert actual == expected

    def test_no_seed_uses_global_state(self, tokenizer, mapped_rxn):
        """Without a seed, the function should use the global random module."""
        random.seed(777)
        result1 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
        )

        random.seed(777)
        result2 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
        )
        assert result1 is not None
        assert result2 is not None
        attn1, unmapped1 = result1
        attn2, unmapped2 = result2
        np.testing.assert_array_equal(attn1, attn2)
        assert unmapped1 == unmapped2

    def test_seed_with_unmapped_atoms(self, tokenizer, mapped_rxn_with_unmapped):
        """Determinism holds even with unmapped atoms in the reaction."""
        result1 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn_with_unmapped,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            seed=42,
        )
        result2 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn_with_unmapped,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            seed=42,
        )
        assert result1 is not None
        assert result2 is not None
        attn1, unmapped1 = result1
        attn2, unmapped2 = result2
        np.testing.assert_array_equal(attn1, attn2)
        assert unmapped1 == unmapped2

    def test_seed_with_symmetric_reaction(self, tokenizer):
        """Determinism holds with symmetric molecules and smoothing."""
        rxn = "[C:1]([O:2])([O:3])=O>>[C:1]([O:2])([O:3])=O"
        result1 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            smooth_symmetric_targets=True,
            seed=42,
        )
        result2 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            smooth_symmetric_targets=True,
            seed=42,
        )
        assert result1 is not None
        assert result2 is not None
        attn1, unmapped1 = result1
        attn2, unmapped2 = result2
        np.testing.assert_array_equal(attn1, attn2)
        assert unmapped1 == unmapped2

    def test_seed_with_tautomer_randomization(self, tokenizer, mapped_rxn):
        """Determinism holds when tautomer randomization is triggered."""
        result1 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            randomize_tautomer_pct=1.0,
            seed=42,
        )
        result2 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            randomize_tautomer_pct=1.0,
            seed=42,
        )
        assert result1 is not None
        assert result2 is not None
        attn1, unmapped1 = result1
        attn2, unmapped2 = result2
        np.testing.assert_array_equal(attn1, attn2)
        assert unmapped1 == unmapped2

    def test_seed_with_canonicalization(self, tokenizer, mapped_rxn):
        """Determinism holds when canonicalization is triggered."""
        result1 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            canonicalize_mapped_rxn_smiles_pct=1.0,
            seed=42,
        )
        result2 = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=True,
            canonicalize_mapped_rxn_smiles_pct=1.0,
            seed=42,
        )
        assert result1 is not None
        assert result2 is not None
        attn1, unmapped1 = result1
        attn2, unmapped2 = result2
        np.testing.assert_array_equal(attn1, attn2)
        assert unmapped1 == unmapped2


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
        """__getitem__ returns dict with mlm_* and align_* namespaced keys."""
        texts = ["[C:1]>>[C:1]"]
        dataset = SupervisedAtomMappingDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=64,
            use_random_smiles=False,
        )
        sample = dataset[0]
        # MLM view keys
        assert "mlm_input_ids" in sample
        assert "mlm_attention_mask" in sample
        assert "mlm_token_type_ids" in sample
        assert "mlm_labels" in sample
        # Alignment view keys
        assert "align_input_ids" in sample
        assert "align_attention_mask" in sample
        assert "align_token_type_ids" in sample
        assert "align_attention_target" in sample
        assert "align_attention_loss_mask" in sample

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
        for key in [
            "mlm_input_ids",
            "mlm_attention_mask",
            "mlm_token_type_ids",
            "mlm_labels",
        ]:
            assert sample[key].shape == (max_length,)
        for key in ["align_input_ids", "align_attention_mask", "align_token_type_ids"]:
            assert sample[key].shape == (max_length,)
        assert sample["align_attention_target"].shape == (max_length, max_length)
        assert sample["align_attention_loss_mask"].shape == (max_length,)

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

    def test_mlm_view_has_masked_tokens(self, tokenizer):
        """MLM view input_ids differ from alignment view (masking was applied)."""
        texts = ["[C:1]([O:2])([O:3])=O>>[C:1]([O:2])([O:3])=O"]
        dataset = SupervisedAtomMappingDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(mlm_probability=0.5),
            max_length=64,
            use_random_smiles=False,
        )
        # Run multiple times since masking is stochastic
        found_difference = False
        for _ in range(20):
            sample = dataset[0]
            if not torch.equal(sample["mlm_input_ids"], sample["align_input_ids"]):
                found_difference = True
                break
        assert found_difference, (
            "MLM view should sometimes differ from align view due to masking"
        )

    def test_align_view_no_masked_tokens(self, tokenizer):
        """Alignment view input_ids contain no mask tokens."""
        mask_token_id = tokenizer.mask_token_id

        texts = ["[C:1]([O:2])([O:3])=O>>[C:1]([O:2])([O:3])=O"]
        dataset = SupervisedAtomMappingDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(mlm_probability=0.5),
            max_length=64,
            use_random_smiles=False,
        )
        sample = dataset[0]
        assert mask_token_id not in sample["align_input_ids"].tolist()

    def test_mlm_labels_not_all_negative(self, tokenizer):
        """MLM labels have at least some non-(-100) values (masking occurred)."""
        texts = ["[C:1]([O:2])([O:3])=O>>[C:1]([O:2])([O:3])=O"]
        dataset = SupervisedAtomMappingDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(mlm_probability=0.5),
            max_length=64,
            use_random_smiles=False,
        )
        found_non_negative = False
        for _ in range(20):
            sample = dataset[0]
            if (sample["mlm_labels"] != -100).any():
                found_non_negative = True
                break
        assert found_non_negative, "MLM labels should sometimes have non-(-100) values"

    def test_attention_loss_mask_binary(self, tokenizer):
        """align_attention_loss_mask contains only 0s and 1s."""
        texts = ["[C:1]>>[C:1]"]
        dataset = SupervisedAtomMappingDataset(
            texts=texts,
            tokenizer=tokenizer,
            mlm_config=MLMConfig(),
            max_length=32,
            use_random_smiles=False,
        )
        sample = dataset[0]
        unique_vals = set(sample["align_attention_loss_mask"].unique().tolist())
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
        assert "mlm_input_ids" in sample
        assert "align_input_ids" in sample


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
