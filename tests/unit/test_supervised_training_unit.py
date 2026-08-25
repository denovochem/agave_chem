"""Unit tests for workflows/model_training_scripts/supervised_training.py."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the workflows directory is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_ROOT = REPO_ROOT / "workflows"
for p in [str(REPO_ROOT), str(WORKFLOWS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_training_scripts.supervised_training import (
    _filter_valid_rxns,
    build_arg_parser,
    main,
)

from agave_chem.mappers.neural.constants import smiles_token_to_id_dict
from agave_chem.mappers.neural.tokenizer import CustomTokenizer

# ---------------------------------------------------------------------------
# _filter_valid_rxns
# ---------------------------------------------------------------------------


class TestFilterValidRxns:
    """Tests for _filter_valid_rxns."""

    @pytest.fixture
    def tokenizer(self):
        return CustomTokenizer(smiles_token_to_id_dict)

    def test_filters_invalid_reactions(self, tokenizer):
        """Invalid reactions are removed."""
        rxns = ["[C:1]>>[C:1]", "not_a_reaction", "[C:1][O:2]>>[C:1][O:2]"]
        result = _filter_valid_rxns(tokenizer, rxns, progress_every=0)
        assert len(result) == 2
        assert "not_a_reaction" not in result

    def test_empty_input(self, tokenizer):
        """Empty input returns empty list."""
        result = _filter_valid_rxns(tokenizer, [], progress_every=0)
        assert result == []

    def test_all_valid(self, tokenizer):
        """All valid reactions are kept."""
        rxns = ["[C:1]>>[C:1]", "[C:1][O:2]>>[C:1][O:2]"]
        result = _filter_valid_rxns(tokenizer, rxns, progress_every=0)
        assert len(result) == 2

    def test_all_invalid(self, tokenizer):
        """All invalid reactions are filtered out."""
        rxns = ["not_a_reaction", "also_not_valid"]
        result = _filter_valid_rxns(tokenizer, rxns, progress_every=0)
        assert result == []


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
                "--save-dir",
                "/tmp/out",
            ]
        )
        assert args.pretrained_model_path == "/tmp/model"
        assert args.training_data_file == "data.txt"
        assert args.save_dir == "/tmp/out"

    def test_default_values(self):
        """Parser sets expected defaults."""
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--pretrained-model-path",
                "/tmp/model",
                "--training-data-file",
                "data.txt",
                "--save-dir",
                "/tmp/out",
            ]
        )
        assert args.target_layer == 9
        assert args.num_epochs == 30
        assert args.batch_size == 64
        assert args.warmup_steps == 10000
        assert args.logging_steps == 100
        assert args.seed == 42
        assert args.train_pct == 0.99
        assert args.shuffle is False
        assert args.skip_filtering is False
        assert args.progress_every == 10000

    def test_missing_required_arg_raises(self):
        """Missing required argument raises SystemExit."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--save-dir", "/tmp/out"])

    def test_skip_filtering_flag(self):
        """--skip-filtering sets skip_filtering to True."""
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--pretrained-model-path",
                "/tmp/model",
                "--training-data-file",
                "data.txt",
                "--save-dir",
                "/tmp/out",
                "--skip-filtering",
            ]
        )
        assert args.skip_filtering is True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main."""

    def test_main_with_skip_filtering(self, tmp_path):
        """main runs with --skip-filtering and calls main_supervised."""
        data_file = tmp_path / "rxns.txt"
        data_file.write_text("[C:1]>>[C:1]\n[C:1][O:2]>>[C:1][O:2]\n")
        save_dir = tmp_path / "output"

        with patch(
            "model_training_scripts.supervised_training.main_supervised"
        ) as mock_main:
            result = main(
                [
                    "--pretrained-model-path",
                    "/tmp/model",
                    "--training-data-file",
                    str(data_file),
                    "--save-dir",
                    str(save_dir),
                    "--skip-filtering",
                ]
            )
        assert result == 0
        mock_main.assert_called_once()

    def test_main_with_filtering(self, tmp_path):
        """main runs with filtering and calls main_supervised."""
        data_file = tmp_path / "rxns.txt"
        data_file.write_text("[C:1]>>[C:1]\n")
        save_dir = tmp_path / "output"

        with patch(
            "model_training_scripts.supervised_training.main_supervised"
        ) as mock_main:
            result = main(
                [
                    "--pretrained-model-path",
                    "/tmp/model",
                    "--training-data-file",
                    str(data_file),
                    "--save-dir",
                    str(save_dir),
                ]
            )
        assert result == 0
        mock_main.assert_called_once()
