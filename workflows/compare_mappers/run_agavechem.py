"""Run AgaveChem mapper on unmapped reactions in batches.

Reads unmapped_reactions.txt, runs either the AgaveChem NeuralReactionMapper
alone or the full ``map_reactions`` pipeline (neural + template fallback) in
batches, and writes mapped reaction SMILES to agavechem_mapped.txt (one per
line, empty string on failure).  Optionally saves a timing JSON file for use
by ``compile_speed_table.py``.

Modes:
  --mode neural    Run NeuralReactionMapper only (default).
  --mode pipeline  Run the full agave_chem.map_reactions pipeline (neural +
                   template fallback, identical-fragment handling).

Requires: agave_chem (pip install git+https://github.com/denovochem/agave_chem.git)
"""

import argparse
import json
import time
from pathlib import Path

from agave_chem.utils.logging_config import disable_library_logging

disable_library_logging()

from agave_chem.mappers.neural.neural_mapper import NeuralReactionMapper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AgaveChem neural mapper on unmapped reactions"
    )
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "unmapped_reactions.txt"),
        help="Path to unmapped reaction SMILES file",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "agavechem_mapped.txt"),
        help="Path to write mapped reactions",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=["neural", "pipeline"],
        default="neural",
        help="Mapping mode: 'neural' for NeuralReactionMapper only, "
        "'pipeline' for full map_reactions pipeline (default: neural)",
    )
    parser.add_argument(
        "--timing-output",
        default=None,
        help="Path to save timing results as JSON",
    )
    args = parser.parse_args()

    reactions = Path(args.input).read_text().splitlines()
    reactions = [r for r in reactions if r.strip()]
    print(f"Loaded {len(reactions)} unmapped reactions from {args.input}")
    print(f"Mode: {args.mode}, batch size: {args.batch_size}")

    if args.mode == "pipeline":
        from agave_chem.main import map_reactions as pipeline_map_reactions

        mapped_results: list[str] = []
        total_start = time.time()

        for batch_start in range(0, len(reactions), args.batch_size):
            batch = reactions[batch_start : batch_start + args.batch_size]
            t0 = time.time()
            try:
                results = pipeline_map_reactions(batch, batch_size=args.batch_size)
            except Exception as e:
                print(f"  Batch {batch_start} failed: {e}")
                results = []
            batch_time = time.time() - t0

            for result in results:
                mapped_results.append(
                    result.final_mapping if result.final_mapping else ""
                )

            done = min(batch_start + args.batch_size, len(reactions))
            elapsed = time.time() - total_start
            print(
                f"  {done}/{len(reactions)} | batch_time={batch_time:.2f}s | "
                f"elapsed={elapsed:.1f}s"
            )

        total_elapsed = time.time() - total_start
    else:
        mapper = NeuralReactionMapper(mapper_name="neural_mapper", mapper_weight=1)

        mapped_results = []
        total_start = time.time()

        for batch_start in range(0, len(reactions), args.batch_size):
            batch = reactions[batch_start : batch_start + args.batch_size]
            t0 = time.time()
            try:
                results = mapper.map_reactions(batch, batch_size=args.batch_size)
            except Exception as e:
                print(f"  Batch {batch_start} failed: {e}")
                results = []
            batch_time = time.time() - t0

            for result in results:
                mapped_results.append(
                    result.selected_mapping if result.selected_mapping else ""
                )

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
        tool_name = (
            "agavechem_neural" if args.mode == "neural" else "agavechem_pipeline"
        )
        timing = {
            "tool": tool_name,
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
