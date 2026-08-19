import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from loguru import logger
from rdkit import RDLogger

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
WORKFLOWS_ROOT = BASE_DIR.parent

for _p in [str(REPO_ROOT), str(WORKFLOWS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_training_scripts.albert_mapper_unuspervised_training import (
    TrainingConfig,
    main,
)
from model_training_scripts.cli_utils import load_config, split_data

from agave_chem.utils.chem_utils import canonicalize_reaction_smiles

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the unsupervised training CLI.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
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
        "--max-length",
        type=int,
        default=384,
        help="Maximum sequence length for padding/truncation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--masking-mode",
        type=str,
        default="span",
        choices=["random", "span"],
        help="MLM masking strategy.",
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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML or JSON config file with a 'training' section. Overrides CLI equivalents.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable automatic mixed precision (AMP) for training.",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Precision dtype for AMP. 'bfloat16' is recommended for Ampere+ GPUs (no GradScaler needed). 'float16' uses GradScaler.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before optimizer step.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Wrap the model with torch.compile for potential speedup.",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Disable deterministic cuDNN behavior for potential speedup.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: ERROR).",
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
    """
    Read reaction SMILES from a file and canonicalize each line.

    Reads lines from the given file, optionally replaces ``~`` with ``.``,
    and canonicalizes each reaction SMILES using
    ``canonicalize_reaction_smiles``. Lines that fail canonicalization
    are skipped with a printed warning.

    Args:
        path (str): Path to the input text file (one reaction per line).
        replace_tilde (bool): If True, replace ``~`` with ``.`` before
            canonicalization.
        progress_every (int): Print an index update every N lines.
            Set to 0 to disable.
        isomeric (bool): If True, retain isomeric information during
            canonicalization.
        remove_mapping (bool): If True, remove atom-map numbers during
            canonicalization.
        canonicalize_tautomer (bool): If True, canonicalize each fragment
            to its canonical tautomer.
        canonicalize_atom_mapping_flag (bool): If True, reassign atom map
            numbers to canonical order after canonicalization.

    Returns:
        List[str]: A list of canonicalized reaction SMILES strings.
    """
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
    """
    Run unsupervised ALBERT MLM pre-training from the CLI.

    Reads and canonicalizes reaction SMILES, optionally deduplicates and
    shuffles, splits into train/validation sets, and delegates to
    ``main`` from ``albert_mapper_unuspervised_training`` for the actual
    training.

    Args:
        argv (Optional[Sequence[str]]): Command-line arguments. If None,
            ``sys.argv`` is used.

    Returns:
        int: Exit code (0 on success).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

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

    rxns_train, rxns_val = split_data(
        rxns=rxns,
        train_pct=args.train_pct,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )

    training_config = TrainingConfig(
        output_dir=args.save_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        compile_model=args.compile_model,
        deterministic=not args.no_deterministic,
    )

    if args.config:
        config = load_config(args.config)
        if "training" in config:
            training_config = TrainingConfig(
                **{**config["training"], "output_dir": args.save_dir}
            )

    main(
        train_texts=rxns_train,
        val_texts=rxns_val,
        training_config=training_config,
        max_length=args.max_length,
        num_workers=args.num_workers,
        masking_mode=args.masking_mode,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
