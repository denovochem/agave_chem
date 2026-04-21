import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from rdkit import RDLogger

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.albert_mapper_training import TrainingConfig, main

from agave_chem.utils.chem_utils import canonicalize_reaction_smiles  # noqa: E402

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run unsupervised ALBERT mapper training."
    )

    parser.add_argument(
        "--training-data-file",
        required=True,
        help="Text file with one reaction SMILES per line.",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Directory to write checkpoints/logs.",
    )

    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--logging-steps", type=int, default=100)

    parser.add_argument("--train-pct", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="If set, do not shuffle reactions before splitting.",
    )

    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="If set, do not deduplicate reactions after canonicalization.",
    )
    parser.add_argument(
        "--no-replace-tilde",
        action="store_true",
        help='If set, do not replace "~" with "." before canonicalization.',
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print an index update every N lines during preprocessing (0 disables).",
    )

    parser.add_argument(
        "--no-isomeric",
        action="store_true",
        help="If set, do not retain isomeric information during canonicalization.",
    )
    parser.add_argument(
        "--no-remove-mapping",
        action="store_true",
        help="If set, keep atom-map numbers during canonicalization.",
    )
    parser.add_argument(
        "--canonicalize-tautomer",
        action="store_true",
        help="If set, canonicalize each fragment to its canonical tautomer.",
    )
    parser.add_argument(
        "--canonicalize-atom-mapping",
        action="store_true",
        help="If set, reassign atom map numbers to canonical order after canonicalization.",
    )

    return parser


def _read_and_canonicalize_rxns(
    path: str,
    replace_tilde: bool,
    progress_every: int,
    isomeric: bool,
    remove_mapping: bool,
    canonicalize_tautomer: bool,
    canonicalize_atom_mapping_flag: bool,
) -> List[str]:
    rxns: List[str] = []
    with open(path, "r") as handle:
        for i, line in enumerate(handle):
            if progress_every > 0 and i % progress_every == 0:
                print(i)
            s = line.strip()
            if not s:
                continue
            if replace_tilde:
                s = s.replace("~", ".")
            try:
                rxns.append(
                    canonicalize_reaction_smiles(
                        s,
                        isomeric=isomeric,
                        remove_mapping=remove_mapping,
                        canonicalize_tautomer=canonicalize_tautomer,
                        return_canonicalized_atom_mapping=canonicalize_atom_mapping_flag,
                    )
                )
            except Exception:
                print(f"Cannot canonicalize {i}")
                continue
    return rxns


def main_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)

    rxns = _read_and_canonicalize_rxns(
        path=args.training_data_file,
        replace_tilde=not args.no_replace_tilde,
        progress_every=args.progress_every,
        isomeric=not args.no_isomeric,
        remove_mapping=not args.no_remove_mapping,
        canonicalize_tautomer=args.canonicalize_tautomer,
        canonicalize_atom_mapping_flag=args.canonicalize_atom_mapping,
    )

    if not args.no_deduplicate:
        rxns = list(set(rxns))

    if not args.no_shuffle:
        random.seed(args.seed)
        random.shuffle(rxns)

    split_idx = int(len(rxns) * args.train_pct)
    rxns_train = rxns[:split_idx]
    rxns_val = rxns[split_idx:]

    training_config = TrainingConfig(
        output_dir=args.save_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
    )

    main(
        train_texts=rxns_train,
        val_texts=rxns_val,
        training_config=training_config,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
