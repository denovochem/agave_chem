import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
WORKFLOWS_ROOT = BASE_DIR.parent

for _p in [str(REPO_ROOT), str(WORKFLOWS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def read_lines(path: str) -> List[str]:
    """
    Read non-empty lines from a text file.

    Args:
        path (str): Path to the input text file.

    Returns:
        List[str]: A list of stripped, non-empty lines.
    """
    lines: List[str] = []
    with open(path, "r") as handle:
        for line in handle:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def split_data(
    rxns: List[str],
    train_pct: float,
    shuffle: bool,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Split a list of reactions into train and validation subsets.

    Args:
        rxns (List[str]): List of reaction SMILES strings.
        train_pct (float): Fraction of data to use for training (0-1,
            exclusive).
        shuffle (bool): If True, shuffle the data before splitting.
        seed (int): Random seed for shuffling.

    Returns:
        Tuple[List[str], List[str]]: A tuple of ``(train_rxns, val_rxns)``.

    Raises:
        ValueError: If ``train_pct`` is not between 0 and 1 (exclusive).
    """
    if not (0.0 < train_pct < 1.0):
        raise ValueError("train_pct must be between 0 and 1 (exclusive)")

    rxns_local = list(rxns)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rxns_local)

    split_idx = int(len(rxns_local) * train_pct)
    return rxns_local[:split_idx], rxns_local[split_idx:]


def seed_worker(worker_id: int) -> None:
    """
    Initialize a DataLoader worker's random state for reproducibility.

    Each worker derives its seed from ``torch.initial_seed()``, which
    incorporates the base seed set in the main process. This ensures
    per-worker RNG diversity while remaining deterministic across runs
    given the same base seed.

    Args:
        worker_id (int): The worker ID assigned by PyTorch DataLoader.
            Not used directly; the seed is derived entirely from
            ``torch.initial_seed()``.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML or JSON config file into a dictionary.

    Args:
        path (str): Path to the config file. Supported extensions are
            ``.yaml``, ``.yml``, and ``.json``.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        with open(p, "r") as f:
            return yaml.safe_load(f)
    elif suffix == ".json":
        with open(p, "r") as f:
            return json.load(f)
    else:
        raise ValueError(
            f"Unsupported config file extension: '{suffix}'. "
            f"Supported: .yaml, .yml, .json"
        )
