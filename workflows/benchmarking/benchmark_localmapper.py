"""Standalone benchmark script for LocalMapper.

Run inside an environment that has localmapper installed along with its
dependencies (torch, dgl, dgllife, rdkit).
Env var AGAVE_BENCH_DIR must point to the benchmarking directory
(or pass --gold-reactions explicitly).
"""

import argparse
import json
import os
import time
from pathlib import Path

from _bench_utils import mappings_equivalent, strip_mapping
from localmapper import localmapper

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LocalMapper on gold reactions"
    )
    bench_dir = os.environ.get("AGAVE_BENCH_DIR", str(Path(__file__).parent))
    default_gold = str(Path(bench_dir) / "gold_reactions_filtered.txt")
    parser.add_argument("--gold-reactions", default=default_gold)
    parser.add_argument("--output", default=None, help="Path to save mapped reactions")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--limit", type=int, default=None, help="Max reactions to process"
    )
    args = parser.parse_args()

    gold_reactions = Path(args.gold_reactions).read_text().splitlines()
    gold_reactions = [r for r in gold_reactions if r.strip()]
    if args.limit:
        gold_reactions = gold_reactions[: args.limit]

    mapper = localmapper()

    mapped_results = []
    correct = 0
    failed = 0
    incorrect = 0
    total_time = 0.0

    unmapped = [strip_mapping(r) for r in gold_reactions]

    print(f"Benchmarking LocalMapper on {len(gold_reactions)} reactions...")
    total_start = time.time()

    for batch_start in range(0, len(unmapped), args.batch_size):
        batch = unmapped[batch_start : batch_start + args.batch_size]
        gold_batch = gold_reactions[batch_start : batch_start + args.batch_size]
        t0 = time.time()
        try:
            results = mapper.get_atom_map(batch)
        except Exception as e:
            print(f"  Batch {batch_start} failed: {e}")
            results = [""] * len(batch)
        batch_time = time.time() - t0
        total_time += batch_time

        if isinstance(results, str):
            results = [results]

        for pred_rxn, gold in zip(results, gold_batch):
            mapped_results.append(pred_rxn if pred_rxn else "")
            if not pred_rxn:
                failed += 1
            elif mappings_equivalent(gold, pred_rxn):
                correct += 1
            else:
                incorrect += 1

        done = min(batch_start + args.batch_size, len(unmapped))
        print(
            f"  {done}/{len(unmapped)} | correct={correct} | incorrect={incorrect} | "
            f"failed={failed} | batch_time={batch_time:.2f}s"
        )

    total_elapsed = time.time() - total_start
    total = len(gold_reactions)
    summary = {
        "tool": "localmapper",
        "total": total,
        "correct": correct,
        "failed": failed,
        "incorrect": incorrect,
        "accuracy": correct / (total - failed) if (total - failed) > 0 else 0.0,
        "pct_correct": round(100.0 * correct / total, 2) if total > 0 else 0.0,
        "total_time_s": round(total_elapsed, 2),
        "avg_time_per_rxn_s": round(total_elapsed / total, 4) if total > 0 else 0.0,
    }
    print("\n=== LocalMapper benchmark results ===")
    print(json.dumps(summary, indent=2))

    if args.output:
        Path(args.output).write_text("\n".join(mapped_results) + "\n")
        print(f"Mapped reactions saved to {args.output}")


if __name__ == "__main__":
    main()
