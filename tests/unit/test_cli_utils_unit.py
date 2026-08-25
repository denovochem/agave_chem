"""Unit tests for workflows/model_training_scripts/cli_utils.py."""

import sys
from pathlib import Path

import pytest

# Ensure the workflows directory is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_ROOT = REPO_ROOT / "workflows"
for p in [str(REPO_ROOT), str(WORKFLOWS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_training_scripts.cli_utils import (
    load_config,
    read_lines,
    seed_worker,
    split_data,
)

# ---------------------------------------------------------------------------
# read_lines
# ---------------------------------------------------------------------------


class TestReadLines:
    """Tests for read_lines."""

    def test_reads_non_empty_lines(self, tmp_path):
        """Non-empty lines are read and stripped."""
        f = tmp_path / "rxns.txt"
        f.write_text("CCO>>CCO\nCC>>CC\n")
        result = read_lines(str(f))
        assert result == ["CCO>>CCO", "CC>>CC"]

    def test_skips_empty_lines(self, tmp_path):
        """Empty lines are skipped."""
        f = tmp_path / "rxns.txt"
        f.write_text("CCO>>CCO\n\n  \nCC>>CC\n")
        result = read_lines(str(f))
        assert result == ["CCO>>CCO", "CC>>CC"]

    def test_empty_file(self, tmp_path):
        """Empty file returns empty list."""
        f = tmp_path / "rxns.txt"
        f.write_text("")
        assert read_lines(str(f)) == []


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------


class TestSplitData:
    """Tests for split_data."""

    def test_basic_split(self):
        """Data is split at the correct percentage."""
        rxns = [f"rxn_{i}" for i in range(100)]
        train, val = split_data(rxns, train_pct=0.8, shuffle=False, seed=42)
        assert len(train) == 80
        assert len(val) == 20

    def test_shuffle_changes_order(self):
        """Shuffling with different seeds produces different orders."""
        rxns = [f"rxn_{i}" for i in range(100)]
        train1, _ = split_data(rxns, train_pct=0.5, shuffle=True, seed=42)
        train2, _ = split_data(rxns, train_pct=0.5, shuffle=True, seed=99)
        assert train1 != train2

    def test_no_shuffle_preserves_order(self):
        """Without shuffle, original order is preserved."""
        rxns = [f"rxn_{i}" for i in range(10)]
        train, val = split_data(rxns, train_pct=0.5, shuffle=False, seed=42)
        assert train == rxns[:5]
        assert val == rxns[5:]

    def test_invalid_train_pct_zero(self):
        """train_pct=0 raises ValueError."""
        with pytest.raises(ValueError, match="train_pct"):
            split_data(["a"], train_pct=0.0, shuffle=False, seed=42)

    def test_invalid_train_pct_one(self):
        """train_pct=1.0 raises ValueError."""
        with pytest.raises(ValueError, match="train_pct"):
            split_data(["a"], train_pct=1.0, shuffle=False, seed=42)

    def test_invalid_train_pct_negative(self):
        """Negative train_pct raises ValueError."""
        with pytest.raises(ValueError, match="train_pct"):
            split_data(["a"], train_pct=-0.5, shuffle=False, seed=42)

    def test_empty_input(self):
        """Empty input returns empty lists."""
        train, val = split_data([], train_pct=0.8, shuffle=False, seed=42)
        assert train == []
        assert val == []

    def test_input_not_mutated(self):
        """Original list is not mutated by split_data."""
        rxns = [f"rxn_{i}" for i in range(10)]
        original = list(rxns)
        split_data(rxns, train_pct=0.5, shuffle=True, seed=42)
        assert rxns == original


# ---------------------------------------------------------------------------
# seed_worker
# ---------------------------------------------------------------------------


class TestSeedWorker:
    """Tests for seed_worker."""

    def test_sets_random_state_deterministically(self):
        """Calling seed_worker with the same torch seed produces the same random state."""
        import random as _random

        import numpy as _np
        import torch as _torch

        _torch.manual_seed(42)
        seed_worker(0)
        state1 = _random.getstate()[1][:5]
        np_state1 = _np.random.get_state()[1][:5]

        _torch.manual_seed(42)
        seed_worker(0)
        state2 = _random.getstate()[1][:5]
        np_state2 = _np.random.get_state()[1][:5]

        assert state1 == state2
        assert list(np_state1) == list(np_state2)

    def test_different_torch_seeds_produce_different_states(self):
        """Different torch seeds produce different random states."""
        import random as _random

        import torch as _torch

        _torch.manual_seed(42)
        seed_worker(0)
        state1 = _random.getstate()[1][:5]

        _torch.manual_seed(99)
        seed_worker(0)
        state2 = _random.getstate()[1][:5]

        assert state1 != state2


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for load_config."""

    def test_load_json_config(self, tmp_path):
        """JSON config files are loaded correctly."""
        import json

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"training": {"num_epochs": 5}}))
        result = load_config(str(config_path))
        assert result == {"training": {"num_epochs": 5}}

    def test_load_yaml_config(self, tmp_path):
        """YAML config files are loaded correctly."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("training:\n  num_epochs: 5\n")
        result = load_config(str(config_path))
        assert result == {"training": {"num_epochs": 5}}

    def test_file_not_found(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/to/config.json")

    def test_unsupported_extension(self, tmp_path):
        """Unsupported file extension raises ValueError."""
        config_path = tmp_path / "config.txt"
        config_path.write_text("some content")
        with pytest.raises(ValueError, match="Unsupported config file extension"):
            load_config(str(config_path))
