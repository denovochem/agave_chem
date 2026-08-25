"""
Verify that build_attention_target_from_mapped_rxn_smiles produces the same
outputs as the golden datasets. Used as a regression check during refactoring.

Usage:
    uv run python verify_golden_datasets.py
    uv run python verify_golden_datasets.py --dataset golden_canonical.pkl
    uv run python verify_golden_datasets.py --dataset golden_seeded_random.pkl --verbose
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
WORKFLOWS_ROOT = REPO_ROOT / "workflows"
for p in [str(REPO_ROOT), str(WORKFLOWS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_training_scripts.albert_mapper_supervised_training import (
    build_attention_target_from_mapped_rxn_smiles,
)

from agave_chem.mappers.neural.constants import (
    smiles_token_to_id_dict,
    token_atom_identity_dict,
)
from agave_chem.mappers.neural.tokenizer import CustomTokenizer

GOLDEN_FILES = [
    "golden_canonical.pkl",
    "golden_seeded_random.pkl",
    "golden_seeded_tautomer.pkl",
    "golden_seeded_canonicalize.pkl",
]


def verify_dataset(
    pkl_path: Path, tokenizer, verbose: bool = False
) -> tuple[int, int, int]:
    """
    Verify a single golden dataset.

    Args:
        pkl_path (Path): Path to the golden pickle file.
        tokenizer: Tokenizer instance for re-running the function.
        verbose (bool): If True, print details for each mismatch.

    Returns:
        Tuple of (n_total, n_passed, n_failed).
    """
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)

    n_total = len(records)
    n_passed = 0
    n_failed = 0

    for i, record in enumerate(records):
        mapped_rxn_smiles = record["mapped_rxn_smiles"]
        expected_attn = record["attn_target"]
        expected_unmapped = record["unmapped_rxn_smiles"]
        params = record["params"]

        result = build_attention_target_from_mapped_rxn_smiles(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn_smiles,
            token_atom_identity_dict=token_atom_identity_dict,
            **params,
        )

        if result is None:
            n_failed += 1
            if verbose:
                print(f"  [FAIL] Record {i}: function returned None")
                print(f"         SMILES: {mapped_rxn_smiles[:80]}...")
            continue

        actual_attn, actual_unmapped = result

        attn_match = np.array_equal(expected_attn, actual_attn)
        unmapped_match = expected_unmapped == actual_unmapped

        if attn_match and unmapped_match:
            n_passed += 1
        else:
            n_failed += 1
            if verbose:
                print(f"  [FAIL] Record {i}:")
                print(f"         SMILES: {mapped_rxn_smiles[:80]}...")
                if not attn_match:
                    diff_idx = np.argwhere(expected_attn != actual_attn)
                    print(
                        f"         attn_target mismatch: {len(diff_idx)} cells differ"
                    )
                    if len(diff_idx) <= 5:
                        for r, c in diff_idx[:5]:
                            print(
                                f"           [{r},{c}] expected={expected_attn[r, c]}, "
                                f"actual={actual_attn[r, c]}"
                            )
                    print(f"         expected shape: {expected_attn.shape}")
                    print(f"         actual shape:   {actual_attn.shape}")
                if not unmapped_match:
                    print("         unmapped mismatch:")
                    print(f"           expected: {expected_unmapped[:80]}...")
                    print(f"           actual:   {actual_unmapped[:80]}...")

    return n_total, n_passed, n_failed


def main():
    parser = argparse.ArgumentParser(
        description="Verify golden datasets against current implementation."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset file to verify (e.g. golden_canonical.pkl). "
        "If not provided, all golden datasets are verified.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print details for each mismatch.",
    )
    args = parser.parse_args()

    tokenizer = CustomTokenizer(smiles_token_to_id_dict)

    if args.dataset:
        files = [args.dataset]
    else:
        files = GOLDEN_FILES

    all_passed = True

    for fname in files:
        pkl_path = REPO_ROOT / fname
        if not pkl_path.exists():
            print(f"SKIP: {fname} not found at {pkl_path}")
            continue

        print(f"\nVerifying {fname} ...")
        n_total, n_passed, n_failed = verify_dataset(
            pkl_path, tokenizer, verbose=args.verbose
        )

        status = "PASS" if n_failed == 0 else "FAIL"
        print(f"  {status}: {n_passed}/{n_total} passed, {n_failed} failed")

        if n_failed > 0:
            all_passed = False
            if not args.verbose:
                print("  (run with --verbose for mismatch details)")

    print()
    if all_passed:
        print("All datasets verified successfully.")
        sys.exit(0)
    else:
        print("Some datasets failed verification!")
        sys.exit(1)


if __name__ == "__main__":
    main()
