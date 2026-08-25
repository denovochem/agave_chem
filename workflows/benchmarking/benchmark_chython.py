"""Standalone benchmark script for chython-rxnmap (GraphFormer mapper).

Run inside a venv that has chython-rxnmap and chython[mapping] installed.
Env var AGAVE_BENCH_DIR must point to the benchmarking directory
(or pass --gold-reactions explicitly).
"""

import argparse
import json
import os
import time
from pathlib import Path

from _bench_utils import mappings_equivalent, strip_mapping
from chython import smiles as chython_smiles


def _map_with_chython(rxn_smiles: str) -> str:
    """Map a single reaction using chython's attention_mapping and return mapped SMILES."""
    r = chython_smiles(rxn_smiles)
    r.attention_mapping()
    return format(r, "m")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark chython-rxnmap on gold reactions"
    )
    bench_dir = os.environ.get("AGAVE_BENCH_DIR", str(Path(__file__).parent))
    default_gold = str(Path(bench_dir) / "gold_reactions_filtered.txt")
    parser.add_argument("--gold-reactions", default=default_gold)
    parser.add_argument("--output", default=None, help="Path to save mapped reactions")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max reactions to process"
    )
    args = parser.parse_args()

    gold_reactions = Path(args.gold_reactions).read_text().splitlines()
    gold_reactions = [r for r in gold_reactions if r.strip()]
    if args.limit:
        gold_reactions = gold_reactions[: args.limit]

    unmapped = [strip_mapping(r) for r in gold_reactions]

    mapped_results = []
    correct = 0
    failed = 0
    incorrect = 0

    print(f"Benchmarking chython-rxnmap on {len(gold_reactions)} reactions...")
    total_start = time.time()
    log_interval = max(1, len(gold_reactions) // 20)

    for i, (rxn, gold) in enumerate(zip(unmapped, gold_reactions)):
        try:
            pred_rxn = _map_with_chython(rxn)
        except Exception as e:
            pred_rxn = ""
            failed += 1
            print(f"  [{i}] failed: {e}")
        else:
            if not pred_rxn:
                failed += 1
            elif mappings_equivalent(gold, pred_rxn):
                correct += 1
            else:
                incorrect += 1

        mapped_results.append(pred_rxn)

        if (i + 1) % log_interval == 0:
            elapsed = time.time() - total_start
            print(
                f"  {i + 1}/{len(gold_reactions)} | correct={correct} | incorrect={incorrect} | "
                f"failed={failed} | elapsed={elapsed:.1f}s"
            )

    total_elapsed = time.time() - total_start
    total = len(gold_reactions)
    summary = {
        "tool": "chython-rxnmap",
        "total": total,
        "correct": correct,
        "failed": failed,
        "incorrect": incorrect,
        "accuracy": correct / (total - failed) if (total - failed) > 0 else 0.0,
        "pct_correct": round(100.0 * correct / total, 2) if total > 0 else 0.0,
        "total_time_s": round(total_elapsed, 2),
        "avg_time_per_rxn_s": round(total_elapsed / total, 4) if total > 0 else 0.0,
    }
    print("\n=== chython-rxnmap benchmark results ===")
    print(json.dumps(summary, indent=2))

    if args.output:
        Path(args.output).write_text("\n".join(mapped_results) + "\n")
        print(f"Mapped reactions saved to {args.output}")


if __name__ == "__main__":
    main()
