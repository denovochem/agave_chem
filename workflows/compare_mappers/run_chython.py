"""Run GraphormerMapper (chython-rxnmap) on unmapped reactions.

Reads unmapped_reactions.txt, runs chython's attention_mapping one reaction
at a time, and writes mapped reaction SMILES to chython_mapped.txt (one per
line, empty string on failure).  Optionally saves a timing JSON file for use
by ``compile_speed_table.py``.

Requires: chython-rxnmap, chython[mapping], rdkit
"""

import argparse
import json
import time
from pathlib import Path

from chython import smiles as chython_smiles


def _map_with_chython(rxn_smiles: str) -> str:
    """Map a single reaction using chython's attention_mapping and return mapped SMILES."""
    r = chython_smiles(rxn_smiles)
    r.attention_mapping()
    return format(r, "m")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GraphormerMapper (chython-rxnmap) on unmapped reactions"
    )
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "unmapped_reactions.txt"),
        help="Path to unmapped reaction SMILES file",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "chython_mapped.txt"),
        help="Path to write mapped reactions",
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

    mapped_results: list[str] = []
    total_start = time.time()
    log_interval = max(1, len(reactions) // 20)

    for i, rxn in enumerate(reactions):
        try:
            mapped_rxn = _map_with_chython(rxn)
        except Exception as e:
            print(f"  [{i}] failed: {e}")
            mapped_rxn = ""

        mapped_results.append(mapped_rxn)

        if (i + 1) % log_interval == 0:
            elapsed = time.time() - total_start
            print(f"  {i + 1}/{len(reactions)} | elapsed={elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\nDone in {total_elapsed:.1f}s")

    Path(args.output).write_text("\n".join(mapped_results) + "\n")
    print(f"Wrote {len(mapped_results)} mapped reactions to {args.output}")

    if args.timing_output:
        timing = {
            "tool": "graphormer_mapper",
            "batch_size": 1,
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
