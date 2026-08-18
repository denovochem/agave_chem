import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set

import torch
from transformers import AlbertForMaskedLM

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
WORKFLOWS_ROOT = BASE_DIR.parent

for _p in [str(REPO_ROOT), str(WORKFLOWS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_training_scripts.albert_mapper_supervised_training import (
    SupervisedAtomMappingDataset,
    SupervisedConfig,
    evaluate_supervised_attention_loss,
)
from model_training_scripts.albert_mapper_unuspervised_training import (
    CustomTokenizer,
    MLMConfig,
)
from model_training_scripts.cli_utils import read_lines, split_data

from agave_chem.mappers.neural.constants import smiles_token_to_id_dict
from agave_chem.mappers.neural.model import (
    AlbertWithAttentionAlignment,
)


def _parse_int_list(values: Optional[Sequence[int]]) -> Optional[List[int]]:
    """
    Convert a sequence of values to a list of integers.

    Args:
        values (Optional[Sequence[int]]): A sequence of integers, or None.

    Returns:
        Optional[List[int]]: A list of integers, or None if input is None.
    """
    if values is None:
        return None
    return [int(v) for v in values]


def _expand_range(r: Optional[Sequence[int]]) -> Optional[List[int]]:
    """
    Expand a ``(start, end)`` pair into an inclusive integer range.

    Args:
        r (Optional[Sequence[int]]): A sequence of two integers
            ``(start, end)``, or None.

    Returns:
        Optional[List[int]]: A list ``[start, start+1, ..., end]``, or
        None if input is None.

    Raises:
        ValueError: If the sequence does not have exactly two elements,
            or if ``end < start``.
    """
    if r is None:
        return None
    if len(r) != 2:
        raise ValueError("Range must be specified as two integers: start end")
    start, end = int(r[0]), int(r[1])
    if end < start:
        raise ValueError("Range end must be >= start")
    return list(range(start, end + 1))


def _resolve_device(device: str) -> torch.device:
    """
    Resolve a device string into a ``torch.device``.

    Args:
        device (str): A device string such as ``"cpu"``, ``"cuda"``,
            ``"cuda:0"``, or ``"auto"``. ``"auto"`` selects CUDA if
            available, otherwise CPU.

    Returns:
        torch.device: The resolved device.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _parse_protected_tokens(values: Sequence[str]) -> Set[str]:
    """
    Convert a sequence of token strings into a set.

    Args:
        values (Sequence[str]): Token strings to protect from masking.

    Returns:
        Set[str]: A set of protected token strings.
    """
    return set(values)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the layer/head identification CLI.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
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
    """
    Run layer/head identification evaluation.

    Loads a pretrained ALBERT model, creates a supervised attention
    alignment wrapper, and evaluates the attention alignment loss across
    all specified layer and head combinations. Prints the best
    (lowest-loss) combination at the end.

    Args:
        argv (Optional[Sequence[str]]): Command-line arguments. If None,
            ``sys.argv`` is used.

    Returns:
        int: Exit code (0 on success).
    """
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
    rxns = read_lines(args.training_data_file)
    _, rxns_val = split_data(
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
        multitask=False,
    )
    model = AlbertWithAttentionAlignment(
        base_model=base_model,
        supervised_config=supervised_config,
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
            )
            print(f"layer={layer_num} head={head_num} loss={layer_head_combo_loss}")
            if layer_head_combo_loss < best_loss:
                best_loss = layer_head_combo_loss
                best_layer_head_combo = (layer_num, head_num)

    print(f"best_loss={best_loss} best_layer_head={best_layer_head_combo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
