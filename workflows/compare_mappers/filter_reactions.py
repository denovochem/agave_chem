"""Filter reactions and strip atom mapping.

Reads reactions from a mapped SMILES file, optionally filters to only
partially-mapped reactions (where at least one product atom has atom map
number 0), randomly samples a subset, canonicalizes and strips all atom
mapping, and writes:
  - raw_reactions.txt   (original SMILES, one per line)
  - unmapped_reactions.txt (canonicalized, mapping-stripped SMILES, one per line)

By default, reactions are randomly sampled (seeded for reproducibility).
Use ``--require-partial`` to keep only partially-mapped reactions, matching
the original behaviour used by the Elo/comparison pipeline.

Requires: agave_chem (for canonicalize_reaction_smiles), rdkit
"""

import argparse
import random
from pathlib import Path

from rdkit import Chem

from agave_chem.utils.chem_utils import canonicalize_reaction_smiles


def is_fully_mapped(rxn_smiles: str) -> bool:
    """Check if every product atom has a non-zero atom map number."""
    parts = rxn_smiles.split(">>")
    if len(parts) != 2:
        return True
    for smi in parts[1].split("."):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter reactions and strip atom mapping"
    )
    parser.add_argument(
        "--input",
        default=str(
            Path(__file__).resolve().parent.parent
            / "data"
            / "mapped_smiles_08_21_26.txt"
        ),
        help="Path to mapped reaction SMILES file",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory to write output files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Max reactions to keep (default 10000)",
    )
    parser.add_argument(
        "--require-partial",
        action="store_true",
        help="Keep only partially-mapped reactions (at least one product atom "
        "has atom map number 0).  Default: accept all reactions.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        default=True,
        help="Randomly sample reactions instead of taking the first N (default: True)",
    )
    parser.add_argument(
        "--no-random",
        dest="random",
        action="store_false",
        help="Take the first N reactions in file order instead of random sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default 42)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw_reactions.txt"
    unmapped_path = output_dir / "unmapped_reactions.txt"

    print(f"Reading reactions from {args.input}...")
    with open(args.input) as f:
        all_lines = f.readlines()

    all_rxns = [line.strip() for line in all_lines if line.strip()]
    print(f"Total reactions in file: {len(all_rxns)}")

    if args.require_partial:
        filtered = [rxn for rxn in all_rxns if not is_fully_mapped(rxn)]
        print(f"Partially-mapped reactions: {len(filtered)}")
    else:
        filtered = all_rxns
        print(f"Using all reactions (no partial-mapping filter): {len(filtered)}")

    if args.random:
        rng = random.Random(args.seed)
        if len(filtered) > args.limit:
            sampled = rng.sample(filtered, args.limit)
        else:
            sampled = filtered
        print(f"Randomly sampled {len(sampled)} reactions (seed={args.seed})")
    else:
        sampled = filtered[: args.limit]
        print(f"Selected first {len(sampled)} reactions")

    raw_reactions = []
    unmapped_reactions = []
    for rxn in sampled:
        raw_reactions.append(rxn)
        unmapped_reactions.append(
            canonicalize_reaction_smiles(rxn, remove_mapping=True)
        )

    raw_path.write_text("\n".join(raw_reactions) + "\n")
    unmapped_path.write_text("\n".join(unmapped_reactions) + "\n")

    print(f"Wrote {len(raw_reactions)} raw reactions to {raw_path}")
    print(f"Wrote {len(unmapped_reactions)} unmapped reactions to {unmapped_path}")


if __name__ == "__main__":
    main()
