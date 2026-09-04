"""Run RXNMapper v2 on unmapped reactions in batches.

Reads unmapped_reactions.txt, runs RXNMapper v2 in batches, and writes
mapped reaction SMILES to rxnmapper_v2_mapped.txt (one per line, empty
string on failure).

Requires: rxnmapper v2 (git+https://github.com/yvsgrndjn/RXNMapper_v2.git)
"""

import argparse
import json
import time
from pathlib import Path

from rxnmapper import RXNMapper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RXNMapper v2 on unmapped reactions"
    )
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "unmapped_reactions.txt"),
        help="Path to unmapped reaction SMILES file",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "rxnmapper_v2_mapped.txt"),
        help="Path to write mapped reactions",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--timing-output",
        default=None,
        help="Path to save timing results as JSON",
    )
    args = parser.parse_args()

    reactions = Path(args.input).read_text().splitlines()
    reactions = [r for r in reactions if r.strip()]
    print(f"Loaded {len(reactions)} unmapped reactions from {args.input}")

    mapper = RXNMapper()

    mapped_results: list[str] = []
    total_start = time.time()

    for batch_start in range(0, len(reactions), args.batch_size):
        batch = reactions[batch_start : batch_start + args.batch_size]
        t0 = time.time()
        try:
            results = mapper.get_attention_guided_atom_maps(batch)
        except Exception as e:
            print(f"  Batch {batch_start} failed: {e}")
            results = [{"mapped_rxn": ""}] * len(batch)
        batch_time = time.time() - t0

        for res in results:
            mapped_results.append(res.get("mapped_rxn", ""))

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
            "tool": "rxnmapper_v2",
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
