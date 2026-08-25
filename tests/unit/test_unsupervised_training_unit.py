"""Unit tests for workflows/model_training_scripts/unsupervised_training.py."""

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

from model_training_scripts.unsupervised_training import (
    _read_and_canonicalize_rxns,
    build_arg_parser,
    main_cli,
)

# ---------------------------------------------------------------------------
# build_arg_parser
# ---------------------------------------------------------------------------


class TestBuildArgParser:
    """Tests for build_arg_parser."""

    def test_parser_has_required_args(self):
        """Parser includes required arguments."""
        parser = build_arg_parser()
        args = parser.parse_args(
            ["--training-data-file", "data.txt", "--save-dir", "/tmp/out"]
        )
        assert args.training_data_file == "data.txt"
        assert args.save_dir == "/tmp/out"

    def test_default_values(self):
        """Parser sets expected defaults."""
        parser = build_arg_parser()
        args = parser.parse_args(
            ["--training-data-file", "data.txt", "--save-dir", "/tmp/out"]
        )
        assert args.num_epochs == 20
        assert args.batch_size == 64
        assert args.warmup_steps == 10000
        assert args.logging_steps == 100
        assert args.train_pct == 0.99
        assert args.seed == 42
        assert args.no_shuffle is False
        assert args.no_deduplicate is False
        assert args.no_replace_tilde is False
        assert args.no_isomeric is False
        assert args.no_remove_mapping is False
        assert args.canonicalize_tautomer is False
        assert args.canonicalize_atom_mapping is False

    def test_missing_required_arg_raises(self):
        """Missing required argument raises SystemExit."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--save-dir", "/tmp/out"])

    def test_no_shuffle_flag(self):
        """--no-shuffle sets no_shuffle to True."""
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--training-data-file",
                "data.txt",
                "--save-dir",
                "/tmp/out",
                "--no-shuffle",
            ]
        )
        assert args.no_shuffle is True

    def test_no_deduplicate_flag(self):
        """--no-deduplicate sets no_deduplicate to True."""
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--training-data-file",
                "data.txt",
                "--save-dir",
                "/tmp/out",
                "--no-deduplicate",
            ]
        )
        assert args.no_deduplicate is True


# ---------------------------------------------------------------------------
# _read_and_canonicalize_rxns
# ---------------------------------------------------------------------------


class TestReadAndCanonicalizeRxns:
    """Tests for _read_and_canonicalize_rxns."""

    def test_reads_valid_smiles(self, tmp_path):
        """Valid SMILES lines are read and canonicalized."""
        data_file = tmp_path / "rxns.txt"
        data_file.write_text("CCO>>CCO\nCC>>CC\n")
        result = _read_and_canonicalize_rxns(
            path=str(data_file),
            replace_tilde=False,
            progress_every=0,
            isomeric=True,
            remove_mapping=False,
            canonicalize_tautomer=False,
            canonicalize_atom_mapping_flag=False,
        )
        assert len(result) == 2
        assert all(">>" in r for r in result)

    def test_skips_empty_lines(self, tmp_path):
        """Empty lines are skipped."""
        data_file = tmp_path / "rxns.txt"
        data_file.write_text("CCO>>CCO\n\n\nCC>>CC\n")
        result = _read_and_canonicalize_rxns(
            path=str(data_file),
            replace_tilde=False,
            progress_every=0,
            isomeric=True,
            remove_mapping=False,
            canonicalize_tautomer=False,
            canonicalize_atom_mapping_flag=False,
        )
        assert len(result) == 2

    def test_invalid_smiles_preserved(self, tmp_path):
        """Invalid SMILES are not skipped — canonicalize_reaction_smiles
        logs a warning and returns the input unchanged rather than raising."""
        data_file = tmp_path / "rxns.txt"
        data_file.write_text("CCO>>CCO\nnot_a_smiles\nCC>>CC\n")
        result = _read_and_canonicalize_rxns(
            path=str(data_file),
            replace_tilde=False,
            progress_every=0,
            isomeric=True,
            remove_mapping=False,
            canonicalize_tautomer=False,
            canonicalize_atom_mapping_flag=False,
        )
        # All 3 lines are returned (invalid SMILES is kept as-is)
        assert len(result) == 3

    def test_replace_tilde(self, tmp_path):
        """Tilde characters are replaced with dots when replace_tilde=True."""
        data_file = tmp_path / "rxns.txt"
        data_file.write_text("C~C>>C~C\n")
        result = _read_and_canonicalize_rxns(
            path=str(data_file),
            replace_tilde=True,
            progress_every=0,
            isomeric=True,
            remove_mapping=False,
            canonicalize_tautomer=False,
            canonicalize_atom_mapping_flag=False,
        )
        assert len(result) == 1
        assert "~" not in result[0]

    def test_empty_file(self, tmp_path):
        """Empty file returns empty list."""
        data_file = tmp_path / "rxns.txt"
        data_file.write_text("")
        result = _read_and_canonicalize_rxns(
            path=str(data_file),
            replace_tilde=False,
            progress_every=0,
            isomeric=True,
            remove_mapping=False,
            canonicalize_tautomer=False,
            canonicalize_atom_mapping_flag=False,
        )
        assert result == []


# ---------------------------------------------------------------------------
# main_cli
# ---------------------------------------------------------------------------


class TestMainCli:
    """Tests for main_cli."""

    def test_main_cli_runs(self, tmp_path):
        """main_cli processes data and calls main."""
        data_file = tmp_path / "rxns.txt"
        data_file.write_text("CCO>>CCO\nCC>>CC\n")
        save_dir = tmp_path / "output"

        with patch("model_training_scripts.unsupervised_training.main") as mock_main:
            result = main_cli(
                [
                    "--training-data-file",
                    str(data_file),
                    "--save-dir",
                    str(save_dir),
                ]
            )
        assert result == 0
        mock_main.assert_called_once()
        call_kwargs = mock_main.call_args
        assert len(call_kwargs.kwargs["train_texts"]) > 0
        assert len(call_kwargs.kwargs["val_texts"]) > 0

    def test_main_cli_no_deduplicate(self, tmp_path):
        """--no-deduplicate preserves duplicates."""
        data_file = tmp_path / "rxns.txt"
        data_file.write_text("CCO>>CCO\nCCO>>CCO\n")
        save_dir = tmp_path / "output"

        with patch("model_training_scripts.unsupervised_training.main") as mock_main:
            main_cli(
                [
                    "--training-data-file",
                    str(data_file),
                    "--save-dir",
                    str(save_dir),
                    "--no-deduplicate",
                ]
            )
        call_kwargs = mock_main.call_args
        total = len(call_kwargs.kwargs["train_texts"]) + len(
            call_kwargs.kwargs["val_texts"]
        )
        assert total == 2

    def test_main_cli_deduplicate(self, tmp_path):
        """Default behavior deduplicates reactions."""
        data_file = tmp_path / "rxns.txt"
        data_file.write_text("CCO>>CCO\nCCO>>CCO\n")
        save_dir = tmp_path / "output"

        with patch("model_training_scripts.unsupervised_training.main") as mock_main:
            main_cli(
                [
                    "--training-data-file",
                    str(data_file),
                    "--save-dir",
                    str(save_dir),
                ]
            )
        call_kwargs = mock_main.call_args
        total = len(call_kwargs.kwargs["train_texts"]) + len(
            call_kwargs.kwargs["val_texts"]
        )
        assert total == 1
