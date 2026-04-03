import argparse
import os
import pickle
import random
from typing import List, Optional, Sequence, Tuple

from training_scripts.albert_mapper_supervised_training import (
    SupervisedConfig,
    build_attention_target_from_mapped_rxn_smiles,
    main_supervised,
)
from training_scripts.albert_mapper_training import CustomTokenizer, TrainingConfig

from agave_chem.mappers.neural.constants import smiles_token_to_id_dict


def _load_rxns_from_pickle(path: str, mapping_key: str) -> List[str]:
    with open(path, "rb") as handle:
        raw_data = pickle.load(handle)

    rxns: List[str] = []
    for item in raw_data:
        if isinstance(item, dict):
            rxns.append(item[mapping_key])
        else:
            raise TypeError(
                "Expected pickled data to be a sequence of dicts; got non-dict element"
            )
    return rxns


def _split_data(
    rxns: List[str],
    train_pct: float,
    shuffle: bool,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if not (0.0 < train_pct < 1.0):
        raise ValueError("train_pct must be between 0 and 1 (exclusive)")

    rxns_local = list(rxns)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rxns_local)

    split_idx = int(len(rxns_local) * train_pct)
    return rxns_local[:split_idx], rxns_local[split_idx:]


def _filter_valid_rxns(
    tokenizer: CustomTokenizer,
    rxns: Sequence[str],
    progress_every: int,
) -> List[str]:
    filtered: List[str] = []
    for i, rxn in enumerate(rxns):
        if progress_every > 0 and i % progress_every == 0:
            print(i)
        try:
            build_attention_target_from_mapped_rxn_smiles(tokenizer, rxn)
            filtered.append(rxn)
        except Exception:
            print(i)
    return filtered


def build_arg_parser() -> argparse.ArgumentParser:
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
        help="Pickle file containing a sequence of dicts with mapped reactions.",
    )
    parser.add_argument(
        "--mapping-key",
        default="selected_mapping",
        help='Key in each pickle dict to extract the mapped reaction string (default: "selected_mapping").',
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Directory to write checkpoints/logs.",
    )

    parser.add_argument("--target-layer", type=int, default=9)
    parser.add_argument("--target-head", type=int, default=5)

    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-pct", type=float, default=0.95)
    parser.add_argument("--shuffle", action="store_true")

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

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer = CustomTokenizer(smiles_token_to_id_dict)

    rxns = _load_rxns_from_pickle(args.training_data_file, mapping_key=args.mapping_key)
    rxns_train, rxns_val = _split_data(
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
    )

    supervised_config = SupervisedConfig()
    supervised_config.target_head = args.target_head
    supervised_config.target_layer = args.target_layer

    main_supervised(
        train_texts=rxns_train_filtered,
        val_texts=rxns_val_filtered,
        training_config=training_config,
        supervised_config=supervised_config,
        pretrained_model_path=args.pretrained_model_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
