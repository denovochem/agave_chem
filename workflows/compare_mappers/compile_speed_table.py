"""Compile per-tool timing JSON files into a speed comparison table.

Reads ``speed_*.json`` files from the script directory (or a specified
directory), aggregates them into a CSV, and optionally prints LaTeX table
rows matching the format of ``tab:mapping_speed`` in the manuscript.

Each timing JSON file must contain at least:
    tool (str), batch_size (int), num_reactions (int),
    total_time_s (float), ms_per_rxn (float)

Output:
  - CSV:  speed_results.csv with columns tool, batch_size, num_reactions,
          total_time_s, ms_per_rxn
  - LaTeX table rows (stdout) when --latex is passed
"""

import argparse
import csv
import json
from pathlib import Path

# Ordered rows matching tab:mapping_speed in the manuscript.
# (tool_name, batch_size, latex_label, cite_key)
_TABLE_ROWS: list[tuple[str, int | None, str, str]] = [
    ("rxnmapper", 1, "RXNMapper~(batch size 1)", "Schwaller2021"),
    ("rxnmapper", 32, "RXNMapper~(batch size 32)", "Schwaller2021"),
    ("rxnmapper_v2", 1, "RXNMapper~v2~(batch size 1)", "Schwaller2021rxnmapperv2"),
    ("rxnmapper_v2", 32, "RXNMapper~v2~(batch size 32)", "Schwaller2021rxnmapperv2"),
    ("graphormer_mapper", 1, "GraphormerMapper~(batch size 1)", "Nugmanov2022"),
    ("localmapper", 1, "LocalMapper~(batch size 1)", "Chen2024"),
    ("agavechem_neural", 1, "AgaveChem~(neural only, batch size 1)", ""),
    ("agavechem_neural", 32, "AgaveChem~(neural only, batch size 32)", ""),
    (
        "agavechem_pipeline",
        32,
        r"AgaveChem~(\texttt{map\_reactions()}, batch size 32)",
        "",
    ),
]


def load_timing_files(input_dir: Path) -> dict[tuple[str, int], dict]:
    """Load all speed_*.json files from a directory.

    Args:
        input_dir (Path): Directory containing speed_*.json files.

    Returns:
        dict[tuple[str, int], dict]: Mapping from (tool, batch_size) to the
            timing dict.  Later files with the same key overwrite earlier ones.
    """
    timings: dict[tuple[str, int], dict] = {}
    for json_path in sorted(input_dir.glob("speed_*.json")):
        try:
            data = json.loads(json_path.read_text())
            key = (data["tool"], data["batch_size"])
            timings[key] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARNING: Could not parse {json_path}: {e}")
    return timings


def write_csv(timings: dict[tuple[str, int], dict], output_path: Path) -> None:
    """Write timing results to a CSV file.

    Args:
        timings (dict[tuple[str, int], dict]): Mapping from (tool, batch_size)
            to timing data.
        output_path (Path): Path to write the CSV file.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tool", "batch_size", "num_reactions", "total_time_s", "ms_per_rxn"],
        )
        writer.writeheader()
        for (tool, batch_size), data in sorted(timings.items()):
            writer.writerow(
                {
                    "tool": tool,
                    "batch_size": batch_size,
                    "num_reactions": data.get("num_reactions", ""),
                    "total_time_s": data.get("total_time_s", ""),
                    "ms_per_rxn": data.get("ms_per_rxn", ""),
                }
            )
    print(f"Wrote CSV to {output_path}")


def print_latex_table(timings: dict[tuple[str, int], dict]) -> None:
    """Print LaTeX table rows matching tab:mapping_speed format.

    Args:
        timings (dict[tuple[str, int], dict]): Mapping from (tool, batch_size)
            to timing data.
    """
    print("% --- LaTeX table rows for tab:mapping_speed ---")
    for tool, batch_size, label, cite in _TABLE_ROWS:
        key = (tool, batch_size) if batch_size is not None else (tool, 1)
        data = timings.get(key)
        if data is not None:
            ms = data.get("ms_per_rxn", "N/A")
            value_str = f"{ms:.2f}" if isinstance(ms, (int, float)) else "N/A"
        else:
            value_str = "N/A"

        cite_str = f"\\cite{{{cite}}}" if cite else ""
        # Bold the AgaveChem pipeline row (best result)
        if tool == "agavechem_pipeline":
            value_str = f"\\textbf{{{value_str}}}"
        print(f"    {label:50s} {cite_str:40s} & {value_str:>10s} \\\\")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile timing JSON files into a speed comparison table"
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing speed_*.json files",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "speed_results.csv"),
        help="Path to write the output CSV",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print LaTeX table rows to stdout",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    timings = load_timing_files(input_dir)

    if not timings:
        print(f"No speed_*.json files found in {input_dir}")
        return

    print(f"Loaded {len(timings)} timing files:")
    for (tool, batch_size), data in sorted(timings.items()):
        print(
            f"  {tool} (bs={batch_size}): "
            f"{data.get('ms_per_rxn', 'N/A')} ms/rxn, "
            f"{data.get('num_reactions', 'N/A')} reactions"
        )

    write_csv(timings, Path(args.output))

    if args.latex:
        print()
        print_latex_table(timings)


if __name__ == "__main__":
    main()
