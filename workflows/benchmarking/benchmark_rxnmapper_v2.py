"""Standalone benchmark script for RXNMapper v2.

RXNMapper v2 (https://github.com/yvsgrndjn/RXNMapper_v2) is an updated,
retrained version of the original RXNMapper.  It uses the same Python API
(``RXNMapper.get_attention_guided_atom_maps``) but ships a new default model
(alberta-uspto-2800k, head 3, layer 10).

Run inside a venv that has rxnmapper-v2 installed
(git+https://github.com/yvsgrndjn/RXNMapper_v2.git).
Env var AGAVE_BENCH_DIR must point to the benchmarking directory
(or pass --gold-reactions explicitly).
"""

import argparse
import json
import os
import time
from pathlib import Path

from _bench_utils import mappings_equivalent, strip_mapping
from rxnmapper import RXNMapper

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RXNMapper v2 on gold reactions"
    )
    bench_dir = os.environ.get("AGAVE_BENCH_DIR", str(Path(__file__).parent))
    default_gold = str(Path(bench_dir) / "gold_reactions_filtered.txt")
    parser.add_argument("--gold-reactions", default=default_gold)
    parser.add_argument("--output", default=None, help="Path to save mapped reactions")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit", type=int, default=None, help="Max reactions to process"
    )
    args = parser.parse_args()

    gold_reactions = Path(args.gold_reactions).read_text().splitlines()
    gold_reactions = [r for r in gold_reactions if r.strip()]
    if args.limit:
        gold_reactions = gold_reactions[: args.limit]

    mapper = RXNMapper()

    mapped_results = []
    correct = 0
    failed = 0
    incorrect = 0
    total_time = 0.0

    unmapped = [strip_mapping(r) for r in gold_reactions]

    print(f"Benchmarking rxnmapper_v2 on {len(gold_reactions)} reactions...")
    total_start = time.time()

    for batch_start in range(0, len(unmapped), args.batch_size):
        batch = unmapped[batch_start : batch_start + args.batch_size]
        gold_batch = gold_reactions[batch_start : batch_start + args.batch_size]
        t0 = time.time()
        try:
            results = mapper.get_attention_guided_atom_maps(batch)
        except Exception as e:
            print(f"  Batch {batch_start} failed: {e}")
            results = [{"mapped_rxn": "", "confidence": 0.0}] * len(batch)
        batch_time = time.time() - t0
        total_time += batch_time

        for i, (res, gold) in enumerate(zip(results, gold_batch)):
            pred_rxn = res.get("mapped_rxn", "")
            mapped_results.append(pred_rxn)
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
        "tool": "rxnmapper_v2",
        "total": total,
        "correct": correct,
        "failed": failed,
        "incorrect": incorrect,
        "accuracy": correct / (total - failed) if (total - failed) > 0 else 0.0,
        "pct_correct": round(100.0 * correct / total, 2) if total > 0 else 0.0,
        "total_time_s": round(total_elapsed, 2),
        "avg_time_per_rxn_s": round(total_elapsed / total, 4) if total > 0 else 0.0,
    }
    print("\n=== rxnmapper_v2 benchmark results ===")
    print(json.dumps(summary, indent=2))

    if args.output:
        Path(args.output).write_text("\n".join(mapped_results) + "\n")
        print(f"Mapped reactions saved to {args.output}")


if __name__ == "__main__":
    main()
