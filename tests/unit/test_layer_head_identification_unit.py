"""Unit tests for workflows/model_training_scripts/layer_head_identification.py."""

import sys
from pathlib import Path

import pytest
import torch

# Ensure the workflows directory is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_ROOT = REPO_ROOT / "workflows"
for p in [str(REPO_ROOT), str(WORKFLOWS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_training_scripts.layer_head_identification import (
    _expand_range,
    _parse_int_list,
    _parse_protected_tokens,
    _resolve_device,
    build_arg_parser,
)

# ---------------------------------------------------------------------------
# _parse_int_list
# ---------------------------------------------------------------------------


class TestParseIntList:
    """Tests for _parse_int_list."""

    def test_none_returns_none(self):
        """None input returns None."""
        assert _parse_int_list(None) is None

    def test_valid_list(self):
        """Valid sequence is converted to list of ints."""
        assert _parse_int_list([1, 2, 3]) == [1, 2, 3]

    def test_string_values_converted(self):
        """String values are converted to int."""
        assert _parse_int_list(["1", "2", "3"]) == [1, 2, 3]

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert _parse_int_list([]) == []


# ---------------------------------------------------------------------------
# _expand_range
# ---------------------------------------------------------------------------


class TestExpandRange:
    """Tests for _expand_range."""

    def test_none_returns_none(self):
        """None input returns None."""
        assert _expand_range(None) is None

    def test_valid_range(self):
        """Valid (start, end) produces inclusive range."""
        assert _expand_range([2, 5]) == [2, 3, 4, 5]

    def test_single_element_range(self):
        """Range where start == end returns single element."""
        assert _expand_range([3, 3]) == [3]

    def test_wrong_length_raises(self):
        """Sequence with != 2 elements raises ValueError."""
        with pytest.raises(ValueError, match="Range"):
            _expand_range([1, 2, 3])

    def test_end_before_start_raises(self):
        """end < start raises ValueError."""
        with pytest.raises(ValueError, match="end"):
            _expand_range([5, 2])


# ---------------------------------------------------------------------------
# _resolve_device
# ---------------------------------------------------------------------------


class TestResolveDevice:
    """Tests for _resolve_device."""

    def test_cpu(self):
        """'cpu' returns cpu device."""
        assert _resolve_device("cpu") == torch.device("cpu")

    def test_auto_returns_torch_device(self):
        """'auto' returns a valid torch.device."""
        dev = _resolve_device("auto")
        assert isinstance(dev, torch.device)
        assert dev.type in ("cpu", "cuda")


# ---------------------------------------------------------------------------
# _parse_protected_tokens
# ---------------------------------------------------------------------------


class TestParseProtectedTokens:
    """Tests for _parse_protected_tokens."""

    def test_returns_set(self):
        """Returns a set."""
        result = _parse_protected_tokens(["^", "$", "."])
        assert isinstance(result, set)
        assert result == {"^", "$", "."}

    def test_empty_input(self):
        """Empty sequence returns empty set."""
        assert _parse_protected_tokens([]) == set()

    def test_duplicates_removed(self):
        """Duplicate values are deduplicated."""
        result = _parse_protected_tokens(["^", "^", "$"])
        assert result == {"^", "$"}


# ---------------------------------------------------------------------------
# build_arg_parser
# ---------------------------------------------------------------------------


class TestBuildArgParser:
    """Tests for build_arg_parser."""

    def test_parser_has_required_args(self):
        """Parser includes required arguments."""
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--pretrained-model-path",
                "/tmp/model",
                "--training-data-file",
                "data.txt",
            ]
        )
        assert args.pretrained_model_path == "/tmp/model"
        assert args.training_data_file == "data.txt"

    def test_default_values(self):
        """Parser sets expected defaults."""
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--pretrained-model-path",
                "/tmp/model",
                "--training-data-file",
                "data.txt",
            ]
        )
        assert args.train_pct == 0.99
        assert args.shuffle is False
        assert args.seed == 42
        assert args.max_length == 256
        assert args.layer_range == [8, 11]
        assert args.head_range == [0, 7]
        assert args.protected_tokens == ["^", "$", ".", ">>"]
        assert args.device == "auto"

    def test_missing_required_arg_raises(self):
        """Missing required argument raises SystemExit."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--training-data-file", "data.txt"])

    def test_explicit_layers(self):
        """--layers overrides layer_range."""
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--pretrained-model-path",
                "/tmp/model",
                "--training-data-file",
                "data.txt",
                "--layers",
                "5",
                "7",
                "9",
            ]
        )
        assert args.layers == [5, 7, 9]

    def test_explicit_heads(self):
        """--heads overrides head_range."""
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--pretrained-model-path",
                "/tmp/model",
                "--training-data-file",
                "data.txt",
                "--heads",
                "0",
                "3",
            ]
        )
        assert args.heads == [0, 3]
