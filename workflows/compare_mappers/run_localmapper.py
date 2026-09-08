"""Run LocalMapper on unmapped reactions in batches.

Reads unmapped_reactions.txt, runs LocalMapper in batches, and writes
mapped reaction SMILES to localmapper_mapped.txt (one per line, empty
string on failure).  Optionally saves a timing JSON file for use by
``compile_speed_table.py``.

Requires: localmapper (pip install localmapper), torch, dgl, dgllife, rdkit
"""

import argparse
import json
import time
from pathlib import Path

from localmapper import localmapper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LocalMapper on unmapped reactions"
    )
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "unmapped_reactions.txt"),
        help="Path to unmapped reaction SMILES file",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "localmapper_mapped.txt"),
        help="Path to write mapped reactions",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--timing-output",
        default=None,
        help="Path to save timing results as JSON",
    )
    args = parser.parse_args()

    reactions = Path(args.input).read_text().splitlines()
    reactions = [r for r in reactions if r.strip()]
    print(f"Loaded {len(reactions)} unmapped reactions from {args.input}")

    mapper = localmapper()

    mapped_results: list[str] = []
    total_start = time.time()

    for batch_start in range(0, len(reactions), args.batch_size):
        batch = reactions[batch_start : batch_start + args.batch_size]
        t0 = time.time()
        try:
            results = mapper.get_atom_map(batch)
        except Exception as e:
            print(f"  Batch {batch_start} failed: {e}")
            results = [""] * len(batch)
        batch_time = time.time() - t0

        if isinstance(results, str):
            results = [results]

        for res in results:
            mapped_results.append(res if res else "")

        done = min(batch_start + args.batch_size, len(reactions))
        elapsed = time.time() - total_start
        print(
            f"  {done}/{len(reactions)} | batch_time={batch_time:.2f}s | "
            f"elapsed={elapsed:.1f}s"
        )

    total_elapsed = time.time() - total_start
    print(f"\nDone in {total_elapsed:.1f}s")

    Path(args.output).write_text("\n".join(mapped_results) + "\n")
    print(f"Wrote {len(mapped_results)} mapped reactions to {args.output}")

    if args.timing_output:
        timing = {
            "tool": "localmapper",
            "batch_size": args.batch_size,
            "num_reactions": len(reactions),
            "total_time_s": round(total_elapsed, 2),
            "ms_per_rxn": round(total_elapsed / len(reactions) * 1000, 2)
            if reactions
            else 0.0,
        }
        Path(args.timing_output).write_text(json.dumps(timing, indent=2) + "\n")
        print(f"Timing saved to {args.timing_output}")


if __name__ == "__main__":
    main()
