"""Unit tests for pydantic config models in workflows/model_training_scripts."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Ensure the workflows directory is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_ROOT = REPO_ROOT / "workflows"
for p in [str(REPO_ROOT), str(WORKFLOWS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_training_scripts.albert_mapper_unuspervised_training import (
    MLMConfig,
    ModelConfig,
    SpanMLMConfig,
    TrainingConfig,
)


class TestModelConfig:
    """Tests for the ModelConfig pydantic model."""

    def test_default_values(self):
        config = ModelConfig()
        assert config.vocab_size == 1024
        assert config.embedding_size == 128
        assert config.hidden_size == 256
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 8
        assert config.intermediate_size == 512
        assert config.hidden_act == "gelu_new"
        assert config.hidden_dropout_prob == 0.1
        assert config.max_position_embeddings == 512
        assert config.num_hidden_groups == 1
        assert config.inner_group_num == 1

    def test_valid_custom_values(self):
        config = ModelConfig(
            vocab_size=2048,
            hidden_size=512,
            hidden_act="relu",
        )
        assert config.vocab_size == 2048
        assert config.hidden_size == 512
        assert config.hidden_act == "relu"

    def test_zero_vocab_size_raises(self):
        with pytest.raises(ValidationError):
            ModelConfig(vocab_size=0)

    def test_negative_hidden_size_raises(self):
        with pytest.raises(ValidationError):
            ModelConfig(hidden_size=-1)

    def test_invalid_hidden_act_raises(self):
        with pytest.raises(ValidationError):
            ModelConfig(hidden_act="invalid")  # type: ignore[arg-type]

    def test_negative_dropout_raises(self):
        with pytest.raises(ValidationError):
            ModelConfig(hidden_dropout_prob=-0.1)

    def test_valid_hidden_act_values(self):
        for act in ["gelu", "gelu_new", "relu", "silu", "tanh"]:
            config = ModelConfig(hidden_act=act)
            assert config.hidden_act == act


class TestTrainingConfig:
    """Tests for the TrainingConfig pydantic model."""

    def test_default_values(self):
        config = TrainingConfig()
        assert config.learning_rate == 2e-4
        assert config.weight_decay == 0.001
        assert config.adam_epsilon == 1e-8
        assert config.max_grad_norm == 1.0
        assert config.num_epochs == 3
        assert config.batch_size == 32
        assert config.warmup_steps == 10000
        assert config.save_steps == 1000
        assert config.logging_steps == 100
        assert config.output_dir == "./albert_output"
        assert config.seed == 42
        assert config.fp16 is False

    def test_valid_custom_values(self):
        config = TrainingConfig(
            learning_rate=1e-3,
            batch_size=64,
            num_epochs=10,
        )
        assert config.learning_rate == 1e-3
        assert config.batch_size == 64
        assert config.num_epochs == 10

    def test_negative_learning_rate_raises(self):
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=-1e-4)

    def test_negative_weight_decay_raises(self):
        with pytest.raises(ValidationError):
            TrainingConfig(weight_decay=-0.01)

    def test_negative_batch_size_raises(self):
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=-1)

    def test_zero_learning_rate_allowed(self):
        config = TrainingConfig(learning_rate=0.0)
        assert config.learning_rate == 0.0

    def test_zero_batch_size_allowed(self):
        config = TrainingConfig(batch_size=0)
        assert config.batch_size == 0


class TestMLMConfig:
    """Tests for the MLMConfig pydantic model."""

    def test_default_values(self):
        config = MLMConfig()
        assert config.mlm_probability == 0.15
        assert config.mask_token_prob == 0.80
        assert config.random_token_prob == 0.10
        assert config.keep_token_prob == 0.10

    def test_valid_custom_values(self):
        config = MLMConfig(
            mlm_probability=0.20,
            mask_token_prob=0.70,
            random_token_prob=0.15,
            keep_token_prob=0.15,
        )
        assert config.mlm_probability == 0.20
        assert config.mask_token_prob == 0.70

    def test_probability_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            MLMConfig(mlm_probability=1.5)

    def test_negative_probability_raises(self):
        with pytest.raises(ValidationError):
            MLMConfig(mask_token_prob=-0.1)

    def test_probabilities_not_summing_to_one_raises(self):
        with pytest.raises(ValidationError):
            MLMConfig(
                mask_token_prob=0.50,
                random_token_prob=0.10,
                keep_token_prob=0.10,
            )

    def test_probabilities_summing_to_one_allowed(self):
        config = MLMConfig(
            mask_token_prob=0.60,
            random_token_prob=0.30,
            keep_token_prob=0.10,
        )
        assert config.mask_token_prob == 0.60


class TestSpanMLMConfig:
    """Tests for the SpanMLMConfig pydantic model."""

    def test_default_values(self):
        config = SpanMLMConfig()
        assert config.mlm_probability == 0.15
        assert config.mask_token_prob == 0.70
        assert config.plausible_replace_prob == 0.20
        assert config.keep_token_prob == 0.10

    def test_default_span_size_weights(self):
        config = SpanMLMConfig()
        assert config.span_size_weights == {
            1: 0.3,
            2: 0.25,
            3: 0.2,
            4: 0.15,
            5: 0.1,
        }

    def test_custom_span_size_weights(self):
        weights = {1: 0.5, 2: 0.3, 3: 0.2}
        config = SpanMLMConfig(span_size_weights=weights)
        assert config.span_size_weights == weights

    def test_probability_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            SpanMLMConfig(mlm_probability=2.0)

    def test_probabilities_not_summing_to_one_raises(self):
        with pytest.raises(ValidationError):
            SpanMLMConfig(
                mask_token_prob=0.50,
                plausible_replace_prob=0.10,
                keep_token_prob=0.10,
            )

    def test_probabilities_summing_to_one_allowed(self):
        config = SpanMLMConfig(
            mask_token_prob=0.60,
            plausible_replace_prob=0.25,
            keep_token_prob=0.15,
        )
        assert config.mask_token_prob == 0.60
