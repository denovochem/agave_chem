import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import torch
from transformers import AlbertForMaskedLM

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agave_chem.mappers.neural.constants import smiles_token_to_id_dict  # noqa: E402
from agave_chem.mappers.neural.neural_mapper import (  # noqa: E402
    AlbertWithAttentionAlignment,
)
from model_training.albert_mapper_supervised_training import (  # noqa: E402
    SupervisedAtomMappingDataset,
    SupervisedConfig,
    evaluate_supervised_attention_loss,
)
from model_training.albert_mapper_unuspervised_training import (  # noqa: E402
    CustomTokenizer,
    MLMConfig,
)


def _parse_int_list(values: Optional[Sequence[int]]) -> Optional[List[int]]:
    if values is None:
        return None
    return [int(v) for v in values]


def _expand_range(r: Optional[Sequence[int]]) -> Optional[List[int]]:
    if r is None:
        return None
    if len(r) != 2:
        raise ValueError("Range must be specified as two integers: start end")
    start, end = int(r[0]), int(r[1])
    if end < start:
        raise ValueError("Range end must be >= start")
    return list(range(start, end + 1))


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _parse_protected_tokens(values: Sequence[str]) -> Set[str]:
    return set(values)


def _read_lines(path: str) -> List[str]:
    rxns: List[str] = []
    with open(path, "r") as handle:
        for line in handle:
            s = line.strip()
            if s:
                rxns.append(s)
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate supervised attention alignment loss across ALBERT layers/heads.",
    )

    parser.add_argument(
        "--pretrained-model-path",
        required=True,
        help="Path to a HuggingFace-compatible ALBERT checkpoint directory.",
    )
    parser.add_argument(
        "--training-data-file",
        required=True,
        help="Text file with one reaction per line.",
    )

    parser.add_argument("--train-pct", type=float, default=0.99)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-length", type=int, default=256)

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Explicit layer indices to evaluate.",
    )
    parser.add_argument(
        "--layer-range",
        nargs=2,
        type=int,
        default=[8, 11],
        metavar=("START", "END"),
        help="Inclusive range of layer indices to evaluate (used if --layers not provided).",
    )

    parser.add_argument(
        "--heads",
        nargs="+",
        type=int,
        default=None,
        help="Explicit head indices to evaluate.",
    )
    parser.add_argument(
        "--head-range",
        nargs=2,
        type=int,
        default=[0, 7],
        metavar=("START", "END"),
        help="Inclusive range of head indices to evaluate (used if --heads not provided).",
    )

    parser.add_argument(
        "--protected-tokens",
        nargs="+",
        default=["^", "$", ".", ">>"],
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device string for torch (e.g. "cpu", "cuda", "cuda:0", or "auto").',
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    layer_list = _parse_int_list(args.layers)
    if layer_list is None:
        layer_list = _expand_range(args.layer_range)

    head_list = _parse_int_list(args.heads)
    if head_list is None:
        head_list = _expand_range(args.head_range)

    if layer_list is None or head_list is None:
        raise ValueError("Layers/heads could not be resolved")

    device = _resolve_device(args.device)

    tokenizer = CustomTokenizer(smiles_token_to_id_dict)
    rxns = _read_lines(args.training_data_file)
    _, rxns_val = _split_data(
        rxns=rxns,
        train_pct=args.train_pct,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    mlm_config = MLMConfig()
    val_dataset = SupervisedAtomMappingDataset(
        texts=rxns_val,
        tokenizer=tokenizer,
        mlm_config=mlm_config,
        protected_tokens=_parse_protected_tokens(args.protected_tokens),
        max_length=args.max_length,
        use_random_smiles=False,
    )

    base_model = AlbertForMaskedLM.from_pretrained(args.pretrained_model_path)
    supervised_config = SupervisedConfig(
        target_layer=0,
        target_head=0,
        multitask=False,
    )
    model = AlbertWithAttentionAlignment(
        base_model=base_model,
        supervised_config=supervised_config,
        max_length=args.max_length,
    )
    model.to(device)

    best_layer_head_combo = (0, 0)
    best_loss = 1e10

    for layer_num in layer_list:
        for head_num in head_list:
            layer_head_combo_loss = evaluate_supervised_attention_loss(
                model,
                val_dataset,
                device=device,
                target_layer=layer_num,
                target_head=head_num,
            )
            print(f"layer={layer_num} head={head_num} loss={layer_head_combo_loss}")
            if layer_head_combo_loss < best_loss:
                best_loss = layer_head_combo_loss
                best_layer_head_combo = (layer_num, head_num)

    print(f"best_loss={best_loss} best_layer_head={best_layer_head_combo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
