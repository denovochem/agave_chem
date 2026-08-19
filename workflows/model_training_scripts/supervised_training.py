import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
WORKFLOWS_ROOT = BASE_DIR.parent

for _p in [str(REPO_ROOT), str(WORKFLOWS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_training_scripts.albert_mapper_supervised_training import (
    SupervisedConfig,
    build_attention_target_from_mapped_rxn_smiles,
    main_supervised,
)
from model_training_scripts.albert_mapper_unuspervised_training import (
    CustomTokenizer,
    TrainingConfig,
)
from model_training_scripts.cli_utils import load_config, read_lines, split_data

from agave_chem.mappers.neural.constants import smiles_token_to_id_dict


def _filter_valid_rxns(
    tokenizer: CustomTokenizer,
    rxns: Sequence[str],
    progress_every: int,
) -> List[str]:
    """
    Filter reactions by attempting to build attention targets.

    Reactions that successfully produce an attention target matrix via
    ``build_attention_target_from_mapped_rxn_smiles`` are kept; those that
    fail (return None or raise) are discarded. A summary of the filtering
    results is logged at INFO level.

    Args:
        tokenizer (CustomTokenizer): Tokenizer for SMILES processing.
        rxns (Sequence[str]): Sequence of mapped reaction SMILES strings.
        progress_every (int): Print an index update every N examples.
            Set to 0 to disable progress output.

    Returns:
        List[str]: A list of reactions that successfully produced
        attention targets.
    """
    filtered: List[str] = []
    for i, rxn in enumerate(rxns):
        if progress_every > 0 and i % progress_every == 0:
            print(i)
        result = build_attention_target_from_mapped_rxn_smiles(tokenizer, rxn)
        if result is not None:
            filtered.append(rxn)
        else:
            logger.debug(f"Filtered out reaction at index {i}")
    total = len(rxns)
    kept = len(filtered)
    logger.info(
        f"Filtering complete: kept {kept}/{total} reactions "
        f"({total - kept} filtered out)"
    )
    return filtered


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the supervised training CLI.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run supervised ALBERT mapping training."
    )

    parser.add_argument(
        "--pretrained-model-path",
        required=True,
        help="Path to a HuggingFace-compatible ALBERT checkpoint directory.",
    )
    parser.add_argument(
        "--training-data-file",
        required=True,
        help="Text file with one mapped reaction SMILES per line.",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Directory to write checkpoints/logs.",
    )

    parser.add_argument("--target-layer", type=int, default=9)

    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-pct", type=float, default=0.99)
    parser.add_argument("--shuffle", action="store_true")

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
        "--progress-every",
        type=int,
        default=10000,
        help="Print an index update every N examples during filtering (0 disables).",
    )
    parser.add_argument(
        "--skip-filtering",
        action="store_true",
        help="If set, do not pre-filter reactions by attempting target construction.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML or JSON config file with 'training' and/or 'supervised' sections. Overrides CLI equivalents.",
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run supervised ALBERT attention alignment training from the CLI.

    Loads mapped reaction SMILES, optionally filters invalid reactions,
    splits into train/validation sets, and delegates to
    ``main_supervised`` for the actual training.

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
    tokenizer = CustomTokenizer(smiles_token_to_id_dict)

    rxns = read_lines(args.training_data_file)
    rxns_train, rxns_val = split_data(
        rxns=rxns,
        train_pct=args.train_pct,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    if args.skip_filtering:
        rxns_train_filtered = list(rxns_train)
        rxns_val_filtered = list(rxns_val)
    else:
        rxns_train_filtered = _filter_valid_rxns(
            tokenizer=tokenizer,
            rxns=rxns_train,
            progress_every=args.progress_every,
        )
        rxns_val_filtered = _filter_valid_rxns(
            tokenizer=tokenizer,
            rxns=rxns_val,
            progress_every=args.progress_every,
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
        supervised_config = SupervisedConfig(**config.get("supervised", {}))
    else:
        supervised_config = SupervisedConfig()
    supervised_config.target_layer = args.target_layer

    main_supervised(
        train_texts=rxns_train_filtered,
        val_texts=rxns_val_filtered,
        training_config=training_config,
        supervised_config=supervised_config,
        pretrained_model_path=args.pretrained_model_path,
        max_length=args.max_length,
        num_workers=args.num_workers,
        masking_mode=args.masking_mode,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
