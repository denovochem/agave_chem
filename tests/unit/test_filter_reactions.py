"""Unit tests for filter_reactions.py.

Tests the filtering and sampling logic without requiring the full
agave_chem stack — only rdkit and the is_fully_mapped helper are exercised.
The canonicalize_reaction_smiles call is mocked to avoid loading agave_chem.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Import helper — load filter_reactions from the workflows directory
# ---------------------------------------------------------------------------


def _load_filter_reactions():
    """Import filter_reactions module from workflows/compare_mappers/."""
    workflow_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "workflows"
        / "compare_mappers"
    )
    if str(workflow_dir) not in sys.path:
        sys.path.insert(0, str(workflow_dir))
    return importlib.import_module("filter_reactions")


filter_reactions = _load_filter_reactions()
is_fully_mapped = filter_reactions.is_fully_mapped


# ---------------------------------------------------------------------------
# is_fully_mapped
# ---------------------------------------------------------------------------


class TestIsFullyMapped:
    """Tests for is_fully_mapped."""

    def test_fully_mapped_reaction(self):
        rxn = "[C:1][C:2]>>[C:1][C:2]"
        assert is_fully_mapped(rxn) is True

    def test_partially_mapped_reaction(self):
        rxn = "[C:1][C:2]>>[C:1]C"
        assert is_fully_mapped(rxn) is False

    def test_unmapped_reaction(self):
        rxn = "CC>>CC"
        assert is_fully_mapped(rxn) is False

    def test_multi_product_fully_mapped(self):
        rxn = "[C:1][C:2]>>[C:1][C:2].[O:3]"
        assert is_fully_mapped(rxn) is True

    def test_multi_product_partially_mapped(self):
        rxn = "[C:1][C:2]>>[C:1][C:2].O"
        assert is_fully_mapped(rxn) is False

    def test_invalid_smiles_returns_true(self):
        rxn = "invalid>>smiles"
        assert is_fully_mapped(rxn) is True

    def test_no_reaction_arrow_returns_true(self):
        rxn = "CC"
        assert is_fully_mapped(rxn) is True


# ---------------------------------------------------------------------------
# main() — integration with temp files and mocked canonicalization
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_reaction_file(tmp_path):
    """Create a temp file with a mix of mapped, partially-mapped, and unmapped reactions."""
    reactions = [
        "[C:1][C:2]>>[C:1][C:2]",       # fully mapped
        "[C:1][C:2]>>[C:1]C",            # partially mapped
        "CC>>CC",                         # unmapped (not fully mapped)
        "[O:1]>>[O:1].[C:2]>>[C:2]",     # invalid (two >>), treated as fully mapped
        "CCO>>CCO",                       # unmapped (not fully mapped)
    ]
    rxn_file = tmp_path / "reactions.txt"
    rxn_file.write_text("\n".join(reactions) + "\n")
    return rxn_file


@pytest.fixture
def mock_canonicalize():
    """Mock canonicalize_reaction_smiles to return input unchanged (stripped of mapping)."""
    def _strip(rxn, remove_mapping=True):
        # Simple mock: just remove atom map numbers
        return rxn.replace("[C:1]", "[C]").replace("[C:2]", "[C]").replace("[O:1]", "[O]")

    with patch.object(filter_reactions, "canonicalize_reaction_smiles", side_effect=_strip):
        yield


class TestFilterReactionsMain:
    """Tests for filter_reactions.main()."""

    def test_require_partial_filters_correctly(
        self, tmp_reaction_file, tmp_path, mock_canonicalize
    ):
        result = tmp_path / "out"
        import sys
        old_argv = sys.argv
        sys.argv = [
            "filter_reactions.py",
            "--input", str(tmp_reaction_file),
            "--output-dir", str(result),
            "--require-partial",
            "--no-random",
            "--limit", "100",
        ]
        try:
            filter_reactions.main()
        finally:
            sys.argv = old_argv

        raw = (result / "raw_reactions.txt").read_text().splitlines()
        unmapped = (result / "unmapped_reactions.txt").read_text().splitlines()

        # Should keep only not-fully-mapped reactions (3 out of 5):
        # [C:1][C:2]>>[C:1]C (partially mapped), CC>>CC (unmapped), CCO>>CCO (unmapped)
        assert len(raw) == 3
        assert "[C:1][C:2]>>[C:1]C" in raw
        assert "CC>>CC" in raw
        assert "CCO>>CCO" in raw
        assert len(unmapped) == 3

    def test_no_require_partial_keeps_all(
        self, tmp_reaction_file, tmp_path, mock_canonicalize
    ):
        result = tmp_path / "out"
        import sys
        old_argv = sys.argv
        sys.argv = [
            "filter_reactions.py",
            "--input", str(tmp_reaction_file),
            "--output-dir", str(result),
            "--no-random",
            "--limit", "100",
        ]
        try:
            filter_reactions.main()
        finally:
            sys.argv = old_argv

        raw = (result / "raw_reactions.txt").read_text().splitlines()
        assert len(raw) == 5

    def test_random_sampling_reproducible(
        self, tmp_reaction_file, tmp_path, mock_canonicalize
    ):
        result1 = tmp_path / "out1"
        result2 = tmp_path / "out2"

        import sys
        old_argv = sys.argv

        for out_dir in [result1, result2]:
            sys.argv = [
                "filter_reactions.py",
                "--input", str(tmp_reaction_file),
                "--output-dir", str(out_dir),
                "--seed", "42",
                "--limit", "3",
            ]
            try:
                filter_reactions.main()
            finally:
                sys.argv = old_argv

        raw1 = (result1 / "raw_reactions.txt").read_text().splitlines()
        raw2 = (result2 / "raw_reactions.txt").read_text().splitlines()

        assert len(raw1) == 3
        assert raw1 == raw2  # same seed → same sample

    def test_random_sampling_different_seeds(
        self, tmp_reaction_file, tmp_path, mock_canonicalize
    ):
        # Need more reactions to make different seeds likely produce different samples
        reactions = [f"R{i}>>P{i}" for i in range(100)]
        rxn_file = tmp_path / "big_reactions.txt"
        rxn_file.write_text("\n".join(reactions) + "\n")

        result1 = tmp_path / "out1"
        result2 = tmp_path / "out2"

        import sys
        old_argv = sys.argv

        for seed, out_dir in [(42, result1), (99, result2)]:
            sys.argv = [
                "filter_reactions.py",
                "--input", str(rxn_file),
                "--output-dir", str(out_dir),
                "--seed", str(seed),
                "--limit", "10",
            ]
            try:
                filter_reactions.main()
            finally:
                sys.argv = old_argv

        raw1 = (result1 / "raw_reactions.txt").read_text().splitlines()
        raw2 = (result2 / "raw_reactions.txt").read_text().splitlines()

        assert len(raw1) == 10
        assert len(raw2) == 10
        assert raw1 != raw2  # different seeds → different samples (high probability)

    def test_limit_caps_sample_size(
        self, tmp_reaction_file, tmp_path, mock_canonicalize
    ):
        result = tmp_path / "out"
        import sys
        old_argv = sys.argv
        sys.argv = [
            "filter_reactions.py",
            "--input", str(tmp_reaction_file),
            "--output-dir", str(result),
            "--no-random",
            "--limit", "2",
        ]
        try:
            filter_reactions.main()
        finally:
            sys.argv = old_argv

        raw = (result / "raw_reactions.txt").read_text().splitlines()
        assert len(raw) == 2
