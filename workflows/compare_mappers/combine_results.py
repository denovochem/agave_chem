"""Combine mapper outputs and extract rdchiral_plus templates into a CSV.

Reads:
  - raw_reactions.txt        (original partially-mapped SMILES)
  - rxnmapper_v2_mapped.txt  (RXNMapper v2 mapped SMILES)
  - agavechem_mapped.txt     (AgaveChem neural mapper mapped SMILES)

Extracts a retrosynthetic template from each mapped reaction using
rdchiral_plus ``extract_from_reaction_smiles`` and writes a CSV with
columns:
  raw_reaction_smiles, rxnmapper_v2_mapped, rxnmapper_v2_template,
  agavechem_mapped, agavechem_template

Requires: rdchiral-plus, rdcanon, rdkit
"""

import argparse
import csv
import time
from pathlib import Path

from rdcanon import canon_reaction_smarts
from rdchiral import extract_from_reaction_smiles


def extract_template(mapped_rxn: str) -> str:
    """Extract and canonicalize a retrosynthetic template from a mapped reaction.

    Uses rdchiral_plus ``extract_from_reaction_smiles`` for extraction and
    rdcanon ``canon_reaction_smarts`` for canonicalization of the resulting
    reaction SMARTS.

    Args:
        mapped_rxn (str): Atom-mapped reaction SMILES string.

    Returns:
        str: Canonicalized reaction SMARTS template, or empty string if the
        input is empty or extraction/canonicalization fails.
    """
    if not mapped_rxn or not mapped_rxn.strip():
        return ""
    try:
        result = extract_from_reaction_smiles(mapped_rxn)
        smarts = result.get("reaction_smarts", "") or ""
        if not smarts:
            return ""
        return canon_reaction_smarts(smarts, mapping=True)
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine mapper outputs and extract templates into CSV"
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing raw_reactions.txt and mapped output files",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "mapper_comparison.csv"),
        help="Path to write the output CSV",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    raw_reactions = Path(input_dir / "raw_reactions.txt").read_text().splitlines()

    rxnmapper_path = input_dir / "rxnmapper_v2_mapped.txt"
    if rxnmapper_path.exists():
        rxnmapper_mapped = rxnmapper_path.read_text().splitlines()
    else:
        print(f"WARNING: {rxnmapper_path} not found — filling with empty strings")
        rxnmapper_mapped = [""] * len(raw_reactions)

    agavechem_path = input_dir / "agavechem_mapped.txt"
    if agavechem_path.exists():
        agavechem_mapped = agavechem_path.read_text().splitlines()
    else:
        print(f"WARNING: {agavechem_path} not found — filling with empty strings")
        agavechem_mapped = [""] * len(raw_reactions)

    n = len(raw_reactions)
    print(f"Raw reactions:       {n}")
    print(f"RXNMapper v2 mapped: {len(rxnmapper_mapped)}")
    print(f"AgaveChem mapped:    {len(agavechem_mapped)}")

    if len(rxnmapper_mapped) < n or len(agavechem_mapped) < n:
        print(
            "WARNING: mapped file lengths differ from raw — padding with empty strings"
        )

    total_start = time.time()
    rows = []
    for i in range(n):
        rxn_mapped = rxnmapper_mapped[i] if i < len(rxnmapper_mapped) else ""
        agave_mapped = agavechem_mapped[i] if i < len(agavechem_mapped) else ""

        rxn_template = extract_template(rxn_mapped)
        agave_template = extract_template(agave_mapped)

        rows.append(
            {
                "raw_reaction_smiles": raw_reactions[i],
                "rxnmapper_v2_mapped": rxn_mapped,
                "rxnmapper_v2_template": rxn_template,
                "agavechem_mapped": agave_mapped,
                "agavechem_template": agave_template,
            }
        )

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - total_start
            print(f"  Processed {i + 1}/{n} | elapsed={elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\nTemplate extraction done in {total_elapsed:.1f}s")

    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "raw_reaction_smiles",
                "rxnmapper_v2_mapped",
                "rxnmapper_v2_template",
                "agavechem_mapped",
                "agavechem_template",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
