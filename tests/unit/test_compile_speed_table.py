"""Unit tests for compile_speed_table.py.

Tests JSON loading, CSV writing, and LaTeX output generation using
temporary directories with synthetic timing files.
"""

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import helper — load compile_speed_table from the workflows directory
# ---------------------------------------------------------------------------


def _load_compile_speed_table():
    """Import compile_speed_table module from workflows/compare_mappers/."""
    workflow_dir = (
        Path(__file__).resolve().parent.parent.parent / "workflows" / "compare_mappers"
    )
    if str(workflow_dir) not in sys.path:
        sys.path.insert(0, str(workflow_dir))
    return importlib.import_module("compile_speed_table")


cst = _load_compile_speed_table()
load_timing_files = cst.load_timing_files
write_csv = cst.write_csv
print_latex_table = cst.print_latex_table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_timing(tool: str, batch_size: int, ms: float, n: int = 10000) -> dict:
    """Create a synthetic timing dict matching the expected JSON schema."""
    return {
        "tool": tool,
        "batch_size": batch_size,
        "num_reactions": n,
        "total_time_s": round(ms * n / 1000, 2),
        "ms_per_rxn": ms,
    }


@pytest.fixture
def timing_dir(tmp_path):
    """Create a temp directory with several speed_*.json files."""
    files = {
        "speed_rxnmapper_bs1.json": _make_timing("rxnmapper", 1, 11.5),
        "speed_rxnmapper_bs32.json": _make_timing("rxnmapper", 32, 11.5),
        "speed_rxnmapper_v2_bs1.json": _make_timing("rxnmapper_v2", 1, 11.2),
        "speed_rxnmapper_v2_bs32.json": _make_timing("rxnmapper_v2", 32, 11.2),
        "speed_graphormer_bs1.json": _make_timing("graphormer_mapper", 1, 11.1),
        "speed_localmapper_bs1.json": _make_timing("localmapper", 1, 11.2),
        "speed_agavechem_neural_bs1.json": _make_timing("agavechem_neural", 1, 10.9),
        "speed_agavechem_neural_bs32.json": _make_timing("agavechem_neural", 32, 10.9),
        "speed_agavechem_pipeline_bs32.json": _make_timing(
            "agavechem_pipeline", 32, 10.8
        ),
    }
    for name, data in files.items():
        (tmp_path / name).write_text(json.dumps(data, indent=2) + "\n")
    return tmp_path


@pytest.fixture
def empty_dir(tmp_path):
    """Return an empty temporary directory."""
    return tmp_path


# ---------------------------------------------------------------------------
# load_timing_files
# ---------------------------------------------------------------------------


class TestLoadTimingFiles:
    """Tests for load_timing_files."""

    def test_loads_all_files(self, timing_dir):
        timings = load_timing_files(timing_dir)
        assert len(timings) == 9

    def test_keys_are_tool_batch_tuples(self, timing_dir):
        timings = load_timing_files(timing_dir)
        assert ("rxnmapper", 1) in timings
        assert ("rxnmapper", 32) in timings
        assert ("agavechem_pipeline", 32) in timings

    def test_values_contain_ms_per_rxn(self, timing_dir):
        timings = load_timing_files(timing_dir)
        assert timings[("rxnmapper", 1)]["ms_per_rxn"] == 11.5

    def test_empty_dir_returns_empty(self, empty_dir):
        timings = load_timing_files(empty_dir)
        assert timings == {}

    def test_malformed_json_skipped(self, timing_dir):
        # Overwrite one file with invalid JSON
        (timing_dir / "speed_rxnmapper_bs1.json").write_text("{invalid json")
        timings = load_timing_files(timing_dir)
        assert ("rxnmapper", 1) not in timings
        assert len(timings) == 8

    def test_missing_key_skipped(self, timing_dir):
        # Overwrite one file with missing 'tool' key
        bad = {"batch_size": 1, "ms_per_rxn": 10.0}
        (timing_dir / "speed_localmapper_bs1.json").write_text(json.dumps(bad))
        timings = load_timing_files(timing_dir)
        assert ("localmapper", 1) not in timings
        assert len(timings) == 8

    def test_non_speed_files_ignored(self, timing_dir):
        # Add a non-speed JSON file
        (timing_dir / "other.json").write_text(
            json.dumps({"tool": "x", "batch_size": 1})
        )
        timings = load_timing_files(timing_dir)
        assert len(timings) == 9  # unchanged


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------


class TestWriteCSV:
    """Tests for write_csv."""

    def test_csv_has_correct_columns(self, timing_dir, tmp_path):
        timings = load_timing_files(timing_dir)
        csv_path = tmp_path / "out.csv"
        write_csv(timings, csv_path)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "tool",
                "batch_size",
                "num_reactions",
                "total_time_s",
                "ms_per_rxn",
            ]

    def test_csv_has_all_rows(self, timing_dir, tmp_path):
        timings = load_timing_files(timing_dir)
        csv_path = tmp_path / "out.csv"
        write_csv(timings, csv_path)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 9

    def test_csv_row_values(self, timing_dir, tmp_path):
        timings = load_timing_files(timing_dir)
        csv_path = tmp_path / "out.csv"
        write_csv(timings, csv_path)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = {r["tool"] + "_" + r["batch_size"]: r for r in reader}

        rxn_row = rows["rxnmapper_1"]
        assert rxn_row["ms_per_rxn"] == "11.5"
        assert rxn_row["num_reactions"] == "10000"


# ---------------------------------------------------------------------------
# print_latex_table
# ---------------------------------------------------------------------------


class TestPrintLatexTable:
    """Tests for print_latex_table."""

    def test_all_rows_printed(self, timing_dir, capsys):
        timings = load_timing_files(timing_dir)
        print_latex_table(timings)
        captured = capsys.readouterr()
        # Should have 9 data rows + 1 comment line
        lines = [l for l in captured.out.strip().split("\n") if l.strip()]
        assert len(lines) == 10  # 1 comment + 9 rows

    def test_missing_tool_shows_na(self, empty_dir, capsys):
        timings = load_timing_files(empty_dir)
        print_latex_table(timings)
        captured = capsys.readouterr()
        assert "N/A" in captured.out

    def test_pipeline_row_is_bolded(self, timing_dir, capsys):
        timings = load_timing_files(timing_dir)
        print_latex_table(timings)
        captured = capsys.readouterr()
        assert "\\textbf{" in captured.out

    def test_cite_keys_present(self, timing_dir, capsys):
        timings = load_timing_files(timing_dir)
        print_latex_table(timings)
        captured = capsys.readouterr()
        assert "\\cite{Schwaller2021}" in captured.out
        assert "\\cite{Nugmanov2022}" in captured.out
