"""Standalone benchmark script for agave_chem mappers.

Run inside a venv that has agave_chem installed
(pip install git+https://github.com/denovochem/agave_chem.git).
Env var AGAVE_BENCH_DIR must point to the benchmarking directory
(or pass --gold-reactions explicitly).

Supports two modes:
  - Per-mapper: benchmark individual mappers (neural, template) by calling
    ``map_reactions`` on the full reaction list.
  - Pipeline:   benchmark the full ``map_reactions`` pipeline (identical-fragment
                handling, multi-mapper fallback) by passing --mapper pipeline.
"""

import argparse
import json
import os
import time
from pathlib import Path

from _bench_utils import mappings_equivalent, strip_mapping

from agave_chem.utils.logging_config import disable_library_logging

disable_library_logging()

from agave_chem.main import map_reactions
from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper
from agave_chem.mappers.neural.neural_mapper import NeuralReactionMapper
from agave_chem.mappers.template.template_mapper import TemplateReactionMapper

_MAPPER_CHOICES = ["neural", "template", "pipeline"]

_MAPPER_LABELS = {
    "mcs": "agave_chem/mcs",
    "neural": "agave_chem/neural",
    "template": "agave_chem/template",
    "pipeline": "agave_chem/pipeline",
}


def _build_mapper(name: str, num_processes: int = 8):
    if name == "mcs":
        if num_processes > 1:
            from agave_chem.mappers.mcs.parallel_mcs_mapper import (
                ParallelMCSReactionMapper,
            )

            return ParallelMCSReactionMapper("mcs_parallel", workers=num_processes)
        return MCSReactionMapper(mapper_name="mcs", mapper_weight=1)
    if name == "neural":
        return NeuralReactionMapper(
            mapper_name="neural_mapper", mapper_weight=1, num_processes=num_processes
        )
    if name == "template":
        if num_processes > 1:
            from agave_chem.mappers.template.parallel_template_mapper import (
                ParallelTemplateReactionMapper,
            )

            return ParallelTemplateReactionMapper(
                "template_parallel", workers=num_processes
            )
        return TemplateReactionMapper("template_default")
    if name == "pipeline":
        raise ValueError(
            "pipeline mode does not use a single mapper; use _benchmark_pipeline"
        )
    raise ValueError(f"Unknown mapper: {name}")


def _evaluate(
    label: str,
    unmapped: list[str],
    gold_reactions: list[str],
    results: list,
    output_prefix: str | None,
    mapper_name: str,
    dump_errors: bool = False,
    total_elapsed: float = 0.0,
) -> dict:
    """Evaluate mapping results against gold standard and print summary."""
    mapped_results: list[str] = []
    error_records: list[str] = []
    correct = 0
    failed = 0
    incorrect = 0

    for i, (result, gold) in enumerate(zip(results, gold_reactions)):
        pred_rxn = getattr(result, "final_mapping", None) or getattr(
            result, "selected_mapping", None
        )
        pred_rxn = pred_rxn or ""
        mapped_results.append(pred_rxn)
        if not pred_rxn:
            failed += 1
            error_records.append(f"{i}\tFAILED\t{unmapped[i]}\t{gold}\t")
        elif mappings_equivalent(gold, pred_rxn):
            correct += 1
        else:
            incorrect += 1
            error_records.append(f"{i}\tINCORRECT\t{unmapped[i]}\t{gold}\t{pred_rxn}")

    total = len(gold_reactions)
    summary = {
        "tool": label,
        "total": total,
        "correct": correct,
        "failed": failed,
        "incorrect": incorrect,
        "accuracy": correct / (total - failed) if (total - failed) > 0 else 0.0,
        "pct_correct": round(100.0 * correct / total, 2) if total > 0 else 0.0,
        "total_time_s": round(total_elapsed, 2),
        "avg_time_per_rxn_s": round(total_elapsed / total, 4) if total > 0 else 0.0,
    }
    print(f"\n=== {label} benchmark results ===")
    print(json.dumps(summary, indent=2))

    if output_prefix:
        out_path = Path(f"{output_prefix}_{mapper_name}.txt")
        out_path.write_text("\n".join(mapped_results) + "\n")
        print(f"Mapped reactions saved to {out_path}")

    if dump_errors and error_records:
        err_path = Path(f"{output_prefix or 'benchmark'}_{mapper_name}_errors.tsv")
        err_path.write_text(
            "index\tstatus\tunmapped\tgold\tpredicted\n"
            + "\n".join(error_records)
            + "\n"
        )
        print(f"Error details saved to {err_path}")

    return summary


def _benchmark_one(
    mapper_name: str,
    gold_reactions: list[str],
    unmapped: list[str],
    output_prefix: str | None,
    num_processes: int = 8,
    dump_errors: bool = False,
) -> dict:
    mapper = _build_mapper(mapper_name, num_processes=num_processes)
    label = _MAPPER_LABELS[mapper_name]

    print(f"\nBenchmarking {label} on {len(gold_reactions)} reactions...")
    total_start = time.time()
    results = mapper.map_reactions(unmapped)
    total_elapsed = time.time() - total_start
    print(f"  Completed in {total_elapsed:.1f}s")

    return _evaluate(
        label=label,
        unmapped=unmapped,
        gold_reactions=gold_reactions,
        results=results,
        output_prefix=output_prefix,
        mapper_name=mapper_name,
        dump_errors=dump_errors,
        total_elapsed=total_elapsed,
    )


def _benchmark_pipeline(
    gold_reactions: list[str],
    unmapped: list[str],
    output_prefix: str | None,
    num_processes: int = 8,
    dump_errors: bool = False,
) -> dict:
    """
    Benchmark the full agave_chem map_reactions pipeline.

    Calls ``agave_chem.main.map_reactions`` with default mappers and batch
    processing, identical-fragment handling, and multi-mapper fallback.  This
    reflects the real-world usage pattern rather than testing individual
    mappers in isolation.

    Args:
        gold_reactions (list[str]): Gold-standard mapped reaction SMILES.
        unmapped (list[str]): Unmapped reaction SMILES to feed to the pipeline.
        output_prefix (str | None): If set, save mapped reactions to
            ``{output_prefix}_pipeline.txt``.
        num_processes (int): Number of parallel processes (default: 8).
        dump_errors (bool): If True, write error details to a TSV file.

    Returns:
        dict: Summary statistics with the same keys as ``_benchmark_one``.
    """
    label = _MAPPER_LABELS["pipeline"]

    print(f"\nBenchmarking {label} on {len(gold_reactions)} reactions...")
    total_start = time.time()
    results = map_reactions(unmapped, num_processes=num_processes)
    total_elapsed = time.time() - total_start
    print(f"  Completed in {total_elapsed:.1f}s")

    return _evaluate(
        label=label,
        unmapped=unmapped,
        gold_reactions=gold_reactions,
        results=results,
        output_prefix=output_prefix,
        mapper_name="pipeline",
        dump_errors=dump_errors,
        total_elapsed=total_elapsed,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark agave_chem mappers on gold reactions"
    )
    bench_dir = os.environ.get("AGAVE_BENCH_DIR", str(Path(__file__).parent))
    default_gold = str(Path(bench_dir) / "gold_reactions_filtered.txt")
    parser.add_argument("--gold-reactions", default=default_gold)
    parser.add_argument(
        "--mapper",
        nargs="+",
        choices=_MAPPER_CHOICES,
        default=_MAPPER_CHOICES,
        help="Which mapper(s) to benchmark (default: all)",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="File prefix for saving results, e.g. 'agave_chem' saves agave_chem_mcs.txt etc.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max reactions to process"
    )
    parser.add_argument(
        "--dump-errors",
        action="store_true",
        help="Write incorrect/failed reactions to a TSV file",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of parallel processes for MCS and template mapping (default: 8)",
    )
    args = parser.parse_args()

    gold_reactions = Path(args.gold_reactions).read_text().splitlines()
    gold_reactions = [r for r in gold_reactions if r.strip()]
    if args.limit:
        gold_reactions = gold_reactions[: args.limit]

    unmapped = [strip_mapping(r) for r in gold_reactions]

    all_summaries = []
    for mapper_name in args.mapper:
        if mapper_name == "pipeline":
            summary = _benchmark_pipeline(
                gold_reactions=gold_reactions,
                unmapped=unmapped,
                output_prefix=args.output_prefix,
                num_processes=args.num_processes,
                dump_errors=args.dump_errors,
            )
        else:
            summary = _benchmark_one(
                mapper_name=mapper_name,
                gold_reactions=gold_reactions,
                unmapped=unmapped,
                output_prefix=args.output_prefix,
                num_processes=args.num_processes,
                dump_errors=args.dump_errors,
            )
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        print("\n=== agave_chem combined summary ===")
        print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
