import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator
from rdkit import Chem
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AlbertConfig,
    AlbertForMaskedLM,
    AlbertTokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training_scripts.cli_utils import seed_worker

from agave_chem.mappers.neural.constants import (
    smiles_id_to_token_dict,
    smiles_token_to_id_dict,
    token_atom_identity_dict,
)
from agave_chem.mappers.neural.tokenizer import CustomTokenizer
from agave_chem.utils.chem_utils import (
    canonicalize_reaction_smiles,
    randomize_reaction_smiles,
)

# ============================================================
# Configuration
# ============================================================


class ModelConfig(BaseModel):
    """
    Pydantic configuration for the ALBERT model architecture.

    All fields map directly to ``transformers.AlbertConfig`` parameters.
    Validators ensure integer fields are positive and float fields are
    non-negative.

    Args:
        vocab_size (int): Size of the SMILES vocabulary.
        embedding_size (int): Dimension of token embeddings.
        hidden_size (int): Hidden layer dimension.
        num_hidden_layers (int): Number of transformer layers.
        num_attention_heads (int): Number of attention heads per layer.
        intermediate_size (int): Feed-forward intermediate dimension.
        hidden_act (Literal["gelu", "gelu_new", "relu", "silu", "tanh"]):
            Activation function for hidden layers.
        hidden_dropout_prob (float): Dropout probability for hidden layers.
        attention_probs_dropout_prob (float): Dropout probability for
            attention probabilities.
        max_position_embeddings (int): Maximum sequence length the model
            can accept.
        type_vocab_size (int): Size of the token-type vocabulary.
        initializer_range (float): Standard deviation for weight
            initialization.
        layer_norm_eps (float): Epsilon for layer normalization.
        classifier_dropout_prob (float): Dropout probability for the
            classifier head.
        num_hidden_groups (int): Number of hidden parameter-sharing groups
            (ALBERT cross-layer parameter sharing).
        inner_group_num (int): Number of inner groups within each hidden
            group.
    """

    vocab_size: int = 1024
    embedding_size: int = 128
    hidden_size: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    intermediate_size: int = 512
    hidden_act: Literal["gelu", "gelu_new", "relu", "silu", "tanh"] = "gelu_new"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    classifier_dropout_prob: float = 0.1
    num_hidden_groups: int = 1
    inner_group_num: int = 1

    @field_validator(
        "vocab_size",
        "embedding_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "max_position_embeddings",
        "type_vocab_size",
        "num_hidden_groups",
        "inner_group_num",
    )
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @field_validator(
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
        "classifier_dropout_prob",
        "initializer_range",
        "layer_norm_eps",
    )
    @classmethod
    def _non_negative_float(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return v


class TrainingConfig(BaseModel):
    """
    Pydantic configuration for the MLM training loop.

    Controls optimizer hyperparameters, scheduling, checkpointing, and
    logging cadence.

    Args:
        learning_rate (float): Peak learning rate for AdamW.
        weight_decay (float): Weight decay coefficient applied to
            non-bias / non-LayerNorm parameters.
        adam_epsilon (float): Epsilon for the AdamW optimizer.
        max_grad_norm (float): Maximum gradient norm for gradient clipping.
        num_epochs (int): Number of training epochs to run.
        batch_size (int): Training batch size.
        warmup_steps (int): Number of linear warmup steps before the
            learning rate decays.
        logging_steps (int): Number of steps between printed loss logs.
        output_dir (str): Directory for saving model checkpoints.
        seed (int): Random seed for reproducibility (Python ``random``,
            ``numpy``, and ``torch``).
        save_best_model (bool): If True, save the best model (by validation
            loss) to ``{output_dir}/best_model.pt``.
        early_stopping_patience (int): Number of epochs without improvement
            before stopping. 0 disables early stopping.
        early_stopping_min_delta (float): Minimum validation loss improvement
            to count as an improvement.
    """

    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    batch_size: int = 32
    warmup_steps: int = 10000
    logging_steps: int = 100
    output_dir: str = "./albert_output"
    seed: int = 42
    save_best_model: bool = True
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

    @field_validator(
        "learning_rate",
        "weight_decay",
        "adam_epsilon",
        "max_grad_norm",
        "early_stopping_min_delta",
    )
    @classmethod
    def _non_negative_float(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @field_validator(
        "num_epochs",
        "batch_size",
        "warmup_steps",
        "logging_steps",
        "seed",
        "early_stopping_patience",
    )
    @classmethod
    def _non_negative_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be non-negative")
        return v


class MLMConfig(BaseModel):
    """
    Pydantic configuration for standard BERT-style MLM masking.

    Controls the fraction of tokens selected for masking and the
    replacement strategy applied to each selected token.

    Args:
        mlm_probability (float): Probability of selecting a token for masking.
        mask_token_prob (float): Probability of replacing a selected token
            with ``[MASK]``.
        random_token_prob (float): Probability of replacing a selected
            token with a random vocabulary token.
        keep_token_prob (float): Probability of keeping the original token
            unchanged (but still using it as a prediction target).

    Note:
        ``mask_token_prob + random_token_prob + keep_token_prob`` must sum
        to approximately 1.0.  This is validated at construction time.
    """

    mlm_probability: float = 0.15
    mask_token_prob: float = 0.80
    random_token_prob: float = 0.10
    keep_token_prob: float = 0.10

    @field_validator(
        "mlm_probability",
        "mask_token_prob",
        "random_token_prob",
        "keep_token_prob",
    )
    @classmethod
    def _probability_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def _check_prob_sum(self) -> "MLMConfig":
        total = self.mask_token_prob + self.random_token_prob + self.keep_token_prob
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"mask_token_prob + random_token_prob + keep_token_prob "
                f"must sum to ~1.0, got {total}"
            )
        return self


class SpanMLMConfig(BaseModel):
    """
    Pydantic configuration for graph-aware span-based MLM masking.

    Instead of randomly selecting individual tokens, this strategy selects
    contiguous neighborhoods of atoms on the molecular graph for masking.
    Only atom tokens are eligible; structural tokens (bonds, parentheses,
    ring numbers, etc.) are never masked.

    Args:
        mlm_probability (float): Probability of selecting an atom token for
            masking (applied as a binomial draw over all atom tokens).
        span_size_weights (Dict[int, float]): Weights for sampling span
            sizes. Keys are span sizes (number of atoms); values are
            relative weights. Defaults to ``{1: 0.3, 2: 0.25, 3: 0.2,
            4: 0.15, 5: 0.1}``.
        mask_token_prob (float): Probability of replacing a selected token
            with ``[MASK]``.
        plausible_replace_prob (float): Probability of replacing a selected
            token with a chemically plausible substitute (bioisosteric or
            same-element variant).
        keep_token_prob (float): Probability of keeping the original token
            unchanged (but still using it as a prediction target).

    Note:
        ``mask_token_prob + plausible_replace_prob + keep_token_prob`` must
        sum to approximately 1.0.  This is validated at construction time.
    """

    mlm_probability: float = 0.15
    span_size_weights: Dict[int, float] = Field(
        default_factory=lambda: {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
    )
    mask_token_prob: float = 0.70
    plausible_replace_prob: float = 0.20
    keep_token_prob: float = 0.10

    @field_validator(
        "mlm_probability",
        "mask_token_prob",
        "plausible_replace_prob",
        "keep_token_prob",
    )
    @classmethod
    def _probability_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def _check_prob_sum(self) -> "SpanMLMConfig":
        total = (
            self.mask_token_prob + self.plausible_replace_prob + self.keep_token_prob
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"mask_token_prob + plausible_replace_prob + keep_token_prob "
                f"must sum to ~1.0, got {total}"
            )
        return self


# ============================================================
# Data Preprocessing
# ============================================================


def preprocess_token(
    token_id: int,
    mask_token_id: int,
    vocab_size: int,
    mlm_config: MLMConfig,
) -> Tuple[int, bool]:
    """
    Apply the BERT/ALBERT masking strategy to a single token.

    Given a token that has been selected for masking, applies one of three
    transformations based on the probabilities in ``mlm_config``:
        - Replace with ``[MASK]`` (controlled by ``mask_token_prob``)
        - Replace with a random vocabulary token (``random_token_prob``)
        - Keep the original token unchanged (``keep_token_prob``)

    Args:
        token_id (int): The original token ID to be masked.
        mask_token_id (int): The integer ID of the ``[MASK]`` token.
        vocab_size (int): Size of the vocabulary (for random token sampling).
        mlm_config (MLMConfig): MLM configuration containing the
            replacement probabilities.

    Returns:
        Tuple[int, bool]: A tuple of ``(new_token_id, was_modified)`` where
        ``was_modified`` is ``True`` if the token was changed (masked or
        replaced with a random token), and ``False`` if kept unchanged.
    """
    rand = random.random()

    # 80% of the time, replace with [MASK]
    if rand < mlm_config.mask_token_prob:
        return mask_token_id, True

    # 10% of the time, replace with a random token
    elif rand < mlm_config.mask_token_prob + mlm_config.random_token_prob:
        random_token_id = random.randint(0, vocab_size - 1)
        return random_token_id, True

    # 10% of the time, keep the original token
    else:
        return token_id, False


def apply_mlm_masking(
    input_ids: List[int],
    tokenizer: AlbertTokenizer | CustomTokenizer | PreTrainedTokenizer,
    mlm_config: MLMConfig,
    special_token_ids: Set[int] | None = None,
) -> Tuple[List[int], List[int]]:
    """
    Apply standard BERT-style MLM masking to a sequence of token IDs.

    Randomly selects ``mlm_config.mlm_probability`` of non-special tokens
    for masking, then applies the 80/10/10 masking strategy via
    :func:`preprocess_token`.

    Args:
        input_ids (List[int]): List of input token IDs.
        tokenizer (AlbertTokenizer | CustomTokenizer | PreTrainedTokenizer):
            The tokenizer (used for ``mask_token_id`` and ``vocab_size``).
        mlm_config (MLMConfig): MLM configuration containing masking
            probabilities.
        special_token_ids (Set[int] | None): Set of token IDs to skip
            (never masked). Defaults to ``tokenizer.all_special_ids``.

    Returns:
        Tuple[List[int], List[int]]:
            - ``masked_input_ids``: Token IDs after masking is applied.
            - ``labels``: ``-100`` for non-masked positions, original token
              ID for masked positions.
    """
    if special_token_ids is None:
        special_token_ids = set(tokenizer.all_special_ids)

    masked_input_ids = input_ids.copy()
    labels = [-100] * len(input_ids)

    # Collect eligible token indices (non-special tokens)
    eligible_indices = [
        i for i, token_id in enumerate(input_ids) if token_id not in special_token_ids
    ]

    # Randomly select 15% of eligible tokens to mask
    num_to_mask = max(1, int(len(eligible_indices) * mlm_config.mlm_probability))
    indices_to_mask = random.sample(
        eligible_indices, min(num_to_mask, len(eligible_indices))
    )

    for idx in indices_to_mask:
        original_token_id = input_ids[idx]
        new_token_id, _ = preprocess_token(
            token_id=original_token_id,
            mask_token_id=tokenizer.mask_token_id,
            vocab_size=tokenizer.vocab_size,
            mlm_config=mlm_config,
        )
        masked_input_ids[idx] = new_token_id
        labels[idx] = original_token_id

    return masked_input_ids, labels


def replace_with_mask(
    input_ids: List[int],
    tokenizer: AlbertTokenizer,
    mlm_config: MLMConfig,
    special_token_ids: Set[int] | None = None,
) -> Tuple[List[int], List[int]]:
    """
    Replace selected tokens only with the ``[MASK]`` token (no random/keep).

    A simplified masking strategy that always replaces selected tokens with
    ``[MASK]``, useful for inference or analysis where the 80/10/10 split
    is not desired.

    Args:
        input_ids (List[int]): List of input token IDs.
        tokenizer (AlbertTokenizer): The tokenizer (used for
            ``mask_token_id``).
        mlm_config (MLMConfig): MLM configuration (only ``mlm_probability``
            is used to determine the fraction of tokens to select).
        special_token_ids (Set[int] | None): Set of token IDs to skip.
            Defaults to ``tokenizer.all_special_ids``.

    Returns:
        Tuple[List[int], List[int]]:
            - ``masked_input_ids``: Token IDs after masking.
            - ``labels``: ``-100`` for non-masked positions, original token
              ID for masked positions.
    """
    if special_token_ids is None:
        special_token_ids = set(tokenizer.all_special_ids)

    masked_input_ids = input_ids.copy()
    labels = [-100] * len(input_ids)

    eligible_indices = [
        i for i, token_id in enumerate(input_ids) if token_id not in special_token_ids
    ]

    num_to_mask = max(1, int(len(eligible_indices) * mlm_config.mlm_probability))
    indices_to_mask = random.sample(
        eligible_indices, min(num_to_mask, len(eligible_indices))
    )

    for idx in indices_to_mask:
        labels[idx] = input_ids[idx]
        masked_input_ids[idx] = tokenizer.mask_token_id

    return masked_input_ids, labels


def replace_with_random_token(
    input_ids: List[int],
    tokenizer: AlbertTokenizer,
    mlm_config: MLMConfig,
    special_token_ids: Set[int] | None = None,
) -> Tuple[List[int], List[int]]:
    """
    Replace selected tokens only with random vocabulary tokens (no mask/keep).

    Useful for studying the effect of random token replacement in isolation.

    Args:
        input_ids (List[int]): List of input token IDs.
        tokenizer (AlbertTokenizer): The tokenizer (used for
            ``vocab_size``).
        mlm_config (MLMConfig): MLM configuration (only ``mlm_probability``
            is used to determine the fraction of tokens to select).
        special_token_ids (Set[int] | None): Set of token IDs to skip.
            Defaults to ``tokenizer.all_special_ids``.

    Returns:
        Tuple[List[int], List[int]]:
            - ``noised_input_ids``: Token IDs with random replacements.
            - ``labels``: ``-100`` for non-selected positions, original
              token ID for selected positions.
    """
    if special_token_ids is None:
        special_token_ids = set(tokenizer.all_special_ids)

    noised_input_ids = input_ids.copy()
    labels = [-100] * len(input_ids)

    eligible_indices = [
        i for i, token_id in enumerate(input_ids) if token_id not in special_token_ids
    ]

    num_to_replace = max(1, int(len(eligible_indices) * mlm_config.mlm_probability))
    indices_to_replace = random.sample(
        eligible_indices, min(num_to_replace, len(eligible_indices))
    )

    for idx in indices_to_replace:
        labels[idx] = input_ids[idx]
        noised_input_ids[idx] = random.randint(0, tokenizer.vocab_size - 1)

    return noised_input_ids, labels


def keep_original_tokens(
    input_ids: List[int],
    tokenizer: AlbertTokenizer,
    mlm_config: MLMConfig,
    special_token_ids: Set[int] | None = None,
) -> Tuple[List[int], List[int]]:
    """
    Keep selected tokens unchanged but mark them as prediction targets.

    Implements the 'keep original' strategy from the BERT/ALBERT paper in
    isolation: selected tokens are not modified in the input, but their
    original IDs are set as labels so the model still predicts them.

    Args:
        input_ids (List[int]): List of input token IDs.
        tokenizer (AlbertTokenizer): The tokenizer (unused except for
            type compatibility).
        mlm_config (MLMConfig): MLM configuration (only ``mlm_probability``
            is used to determine the fraction of tokens to select).
        special_token_ids (Set[int] | None): Set of token IDs to skip.
            Defaults to ``tokenizer.all_special_ids``.

    Returns:
        Tuple[List[int], List[int]]:
            - ``unchanged_input_ids``: The original input token IDs
              (unchanged).
            - ``labels``: ``-100`` for non-selected positions, original
              token ID for selected positions.
    """
    if special_token_ids is None:
        special_token_ids = set(tokenizer.all_special_ids)

    unchanged_input_ids = input_ids.copy()
    labels = [-100] * len(input_ids)

    eligible_indices = [
        i for i, token_id in enumerate(input_ids) if token_id not in special_token_ids
    ]

    num_to_keep = max(1, int(len(eligible_indices) * mlm_config.mlm_probability))
    indices_to_keep = random.sample(
        eligible_indices, min(num_to_keep, len(eligible_indices))
    )

    for idx in indices_to_keep:
        labels[idx] = input_ids[idx]
        # Input stays the same, but we track it as a prediction target

    return unchanged_input_ids, labels


def resolve_protected_token_ids(
    tokenizer: PreTrainedTokenizer,
    protected_tokens: Set[str] | None,
) -> Set[int]:
    """
    Resolve a set of token strings into their corresponding integer IDs.

    Args:
        tokenizer:         The tokenizer to use for conversion.
        protected_tokens:  A set of token strings that should never be masked
                           or used as prediction targets. For example:
                               {".", ",", "!", "?"}
                           These are IN ADDITION to the tokenizer's built-in
                           special tokens ([CLS], [SEP], [PAD], etc.), which
                           are always protected.

    Returns:
        A set of integer token IDs that should be protected.
    """
    if not protected_tokens:
        return set()

    resolved_ids = set()
    vocab = tokenizer.get_vocab()

    for token in protected_tokens:
        if token in vocab:
            resolved_ids.add(vocab[token])
        else:
            # Try to tokenize it and protect all resulting sub-tokens
            sub_ids = tokenizer.encode(token, add_special_tokens=False)
            if sub_ids:
                resolved_ids.update(sub_ids)
                print(
                    f"Protected token '{token}' was not found directly in vocab. "
                    f"Protecting its sub-token IDs instead: {sub_ids}"
                )
            else:
                print(f"Protected token '{token}' could not be resolved. Skipping.")

    return resolved_ids


# ============================================================
# Graph-Aware Span Masking
# ============================================================

# Chemically plausible substitution groups for the XX% "replace with
# plausible token" strategy.  Tokens within a group are bioisosteric
# replacements of each other.  When a masked atom does not belong to
# any explicit group, we fall back to a random same-element token.
PLAUSIBLE_SUBSTITUTION_GROUPS: List[Set[str]] = [
    # --- Cross-element bioisosteric replacements ---
    # Aromatic ring atoms
    {"c", "n", "o", "s"},
    # Aliphatic heavy atoms
    {"C", "N", "O", "S"},
    # Halogens
    {"F", "Cl", "Br", "I"},
    # --- Carbon stereochemistry variants ---
    {"[C@H]", "[C@@H]", "C"},
    {"[C@]", "[C@@]"},
    # --- Nitrogen variants (aromaticity / charge / protonation) ---
    {"n", "[nH]", "[n+]", "[nH+]", "[n-]"},
    {"N", "[N+]", "[NH+]", "[NH2+]", "[NH3+]", "[N-]"},
    {"[N@@+]", "[N@+]", "[N+]"},
    # --- Oxygen variants (charge / protonation) ---
    {"O", "[O-]", "[OH-]", "[OH+]"},
    # --- Sulfur variants (charge / stereo) ---
    {"S", "[S-]", "[S+]", "[SH-]", "[SH+]"},
    {"s", "[s+]"},
    {"[S@]", "[S@@]", "S"},
    # --- Phosphorus variants (stereo / charge) ---
    {"P", "p", "[PH]", "[P+]", "[P-]"},
    {"[P@]", "[P@@]", "P"},
    # --- Silicon variants (stereo) ---
    {"[Si]", "[SiH]", "[Si@]", "[Si@@]"},
    # --- Boron variants ---
    {"B", "b", "[B-]", "[BH-]"},
]

# Build lookup: each token maps to the *union* of all groups it belongs to.
_SUBSTITUTION_LOOKUP: Dict[str, Set[str]] = {}
for _group in PLAUSIBLE_SUBSTITUTION_GROUPS:
    for _token in _group:
        if _token not in _SUBSTITUTION_LOOKUP:
            _SUBSTITUTION_LOOKUP[_token] = set()
        _SUBSTITUTION_LOOKUP[_token] |= _group

# Pre-compute the set of token IDs that represent atoms (atomic_num > 0).
_ATOM_TOKEN_IDS: Set[int] = {
    smiles_token_to_id_dict[tok]
    for tok, anum in token_atom_identity_dict.items()
    if anum > 0 and tok in smiles_token_to_id_dict
}


def _parse_reaction_molecules(
    reaction_smiles: str,
) -> Tuple[List[str], List[Optional[Chem.Mol]]]:
    """
    Parse a reaction SMILES into per-molecule SMILES and RDKit Mol objects.

    Molecules are ordered: reactant_0, reactant_1, …, product_0, product_1, …
    (matching the left-to-right token order in the SMILES string).

    Args:
        reaction_smiles (str): A reaction SMILES of the form
            ``"reactants>>products"`` where each side may contain
            multiple molecules separated by ``"."``.

    Returns:
        Tuple[List[str], List[Optional[Chem.Mol]]]:
            - A list of individual molecule SMILES strings.
            - A parallel list of RDKit Mol objects (``None`` when
              parsing fails for a fragment).
    """
    parts = reaction_smiles.split(">>")
    if len(parts) != 2:
        return [], []

    all_smiles: List[str] = []
    for part in parts:
        for mol_smi in part.split("."):
            stripped = mol_smi.strip()
            if stripped:
                all_smiles.append(stripped)

    mol_objects: List[Optional[Chem.Mol]] = []
    for smi in all_smiles:
        mol = Chem.MolFromSmiles(smi)
        mol_objects.append(mol)  # None on parse failure

    return all_smiles, mol_objects


def _build_atom_token_map(
    input_ids: List[int],
) -> Tuple[Dict[int, Tuple[int, int]], Dict[Tuple[int, int], int], List[int]]:
    """
    Build bidirectional mappings between token positions and (mol_id, atom_idx).

    Walks the token list and uses ``">>"`` / ``"."`` tokens to track
    molecule boundaries. Atom tokens are identified via
    ``token_atom_identity_dict`` (atomic number > 0).

    Args:
        input_ids (List[int]): The full tokenized sequence (including
            special / padding tokens).

    Returns:
        Tuple[Dict[int, Tuple[int, int]], Dict[Tuple[int, int], int], List[int]]:
            - token_to_mol_atom: ``{token_pos: (mol_id, atom_idx)}``
            - mol_atom_to_token: ``{(mol_id, atom_idx): token_pos}``
            - atom_token_positions: flat list of token positions that
              represent atoms (in sequence order).
    """
    token_to_mol_atom: Dict[int, Tuple[int, int]] = {}
    mol_atom_to_token: Dict[Tuple[int, int], int] = {}
    atom_token_positions: List[int] = []

    mol_id = 0
    atom_idx = 0

    rxn_token_id = smiles_token_to_id_dict.get(">>", -1)
    dot_token_id = smiles_token_to_id_dict.get(".", -1)

    for pos, token_id in enumerate(input_ids):
        # Molecule / side separators reset the atom counter.
        if token_id == rxn_token_id:
            mol_id += 1
            atom_idx = 0
            continue
        if token_id == dot_token_id:
            mol_id += 1
            atom_idx = 0
            continue

        if token_id in _ATOM_TOKEN_IDS:
            token_to_mol_atom[pos] = (mol_id, atom_idx)
            mol_atom_to_token[(mol_id, atom_idx)] = pos
            atom_token_positions.append(pos)
            atom_idx += 1

    return token_to_mol_atom, mol_atom_to_token, atom_token_positions


def _select_graph_neighborhood(
    mol: Chem.Mol,
    seed_atom_idx: int,
    span_size: int,
) -> Set[int]:
    """
    BFS from a seed atom to collect up to ``span_size`` neighboring atoms.

    The BFS frontier is shuffled at each step so that the selected
    neighborhood is stochastic (not always the same canonical order).

    Args:
        mol (Chem.Mol): An RDKit Mol object.
        seed_atom_idx (int): The 0-based index of the seed atom.
        span_size (int): Maximum number of atoms to include.

    Returns:
        Set[int]: Atom indices in the selected neighborhood.
    """
    if seed_atom_idx >= mol.GetNumAtoms():
        return {seed_atom_idx}

    visited: Set[int] = {seed_atom_idx}
    queue: deque[int] = deque([seed_atom_idx])

    while len(visited) < span_size and queue:
        current = queue.popleft()
        neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()]
        random.shuffle(neighbors)
        for nidx in neighbors:
            if nidx not in visited:
                visited.add(nidx)
                if len(visited) >= span_size:
                    break
                queue.append(nidx)

    return visited


def _get_plausible_replacement(
    token: str,
    tokenizer: PreTrainedTokenizer,
) -> int:
    """
    Return a chemically plausible replacement token ID for an atom token.

    The lookup proceeds in three stages:
        1. If the token belongs to one or more
           ``PLAUSIBLE_SUBSTITUTION_GROUPS`` entries, sample uniformly
           from the union of all those groups (excluding itself).  This
           covers bioisosteric cross-element swaps, stereochem flips,
           and charge / protonation variants.
        2. Otherwise, sample from all tokens that share the same atomic
           number (same element, different charge / stereo / H-count).
        3. If neither yields a candidate, fall back to ``[MASK]``.

    Args:
        token (str): The original atom token string.
        tokenizer (PreTrainedTokenizer): The tokenizer (used to
            resolve token strings → IDs).

    Returns:
        int: The token ID of the replacement.
    """
    vocab = tokenizer.get_vocab()

    # Stage 1: bioisosteric substitution group
    group = _SUBSTITUTION_LOOKUP.get(token)
    if group:
        candidates = [t for t in group if t != token and t in vocab]
        if candidates:
            return vocab[random.choice(candidates)]

    # Stage 2: same-element token (same atomic number)
    atomic_num = token_atom_identity_dict.get(token, 0)
    if atomic_num > 0:
        same_element = [
            t
            for t, a in token_atom_identity_dict.items()
            if a == atomic_num and t != token and t in vocab
        ]
        if same_element:
            return vocab[random.choice(same_element)]

    # Stage 3: fallback
    return tokenizer.mask_token_id


def apply_span_mlm_masking(
    input_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    span_mlm_config: SpanMLMConfig,
    reaction_smiles: str,
    special_token_ids: Set[int] | None = None,
) -> Tuple[List[int], List[int]]:
    """
    Apply graph-aware span masking to a tokenized reaction SMILES.

    Selects contiguous neighborhoods on the molecular graph until the
    masking budget (``span_mlm_config.mlm_probability`` of atom tokens)
    is filled. For each masked position the replacement strategy is:

        - mask_token_prob %  →  ``[MASK]``
        - plausible_replace_prob %  →  chemically plausible substitute
        - keep_token_prob %  →  keep original token unchanged

    Only atom tokens (those with a non-zero atomic number in
    ``token_atom_identity_dict``) are eligible for masking. If RDKit
    cannot parse one or more molecules, those atoms are still eligible
    but are selected individually (no graph spanning).

    Args:
        input_ids (List[int]): The tokenized sequence.
        tokenizer (PreTrainedTokenizer): The tokenizer instance.
        span_mlm_config (SpanMLMConfig): Span masking hyper-parameters.
        reaction_smiles (str): The raw reaction SMILES *before*
            tokenizer preprocessing (e.g. ``"CCO.c1ccccc1>>c1ccccc1"``).
        special_token_ids (Set[int] | None): Token IDs that must never
            be masked. Defaults to the tokenizer's built-in special IDs.

    Returns:
        Tuple[List[int], List[int]]:
            - masked_input_ids: token IDs after masking.
            - labels: ``-100`` for non-masked positions, original token
              ID for masked positions.
    """
    masked_input_ids = input_ids.copy()
    labels = [-100] * len(input_ids)

    # --- Build atom ↔ token mappings ---
    token_to_mol_atom, mol_atom_to_token, atom_token_positions = _build_atom_token_map(
        input_ids
    )

    if not atom_token_positions:
        return masked_input_ids, labels

    # --- Parse molecules for graph structure ---
    _, mol_objects = _parse_reaction_molecules(reaction_smiles)

    # --- Determine masking budget (stochastic) ---
    num_to_mask = max(
        1,
        np.random.binomial(len(atom_token_positions), span_mlm_config.mlm_probability),
    )

    # --- Select spans until budget is filled ---
    selected_positions: Set[int] = set()
    max_attempts = num_to_mask * 3  # avoid infinite loops on tiny molecules

    for _ in range(max_attempts):
        if len(selected_positions) >= num_to_mask:
            break

        # Pick a random atom token as the span seed
        seed_pos = random.choice(atom_token_positions)
        mol_id, atom_idx = token_to_mol_atom[seed_pos]

        mol = mol_objects[mol_id] if mol_id < len(mol_objects) else None

        if mol is None or mol.GetNumAtoms() == 0:
            # Cannot do graph-aware spanning; select this single atom.
            selected_positions.add(seed_pos)
            continue

        assert span_mlm_config.span_size_weights is not None
        sizes, weights = zip(*span_mlm_config.span_size_weights.items())
        span_size = random.choices(sizes, weights=weights, k=1)[0]
        neighborhood = _select_graph_neighborhood(mol, atom_idx, span_size)

        for neighbor_atom_idx in neighborhood:
            key = (mol_id, neighbor_atom_idx)
            if key in mol_atom_to_token:
                selected_positions.add(mol_atom_to_token[key])

    # Trim to budget if we overshot
    selected_list = list(selected_positions)
    if len(selected_list) > num_to_mask:
        selected_list = random.sample(selected_list, num_to_mask)

    # --- Apply mask_token_prob / plausible_replace_prob / keep_token_prob masking strategy ---
    for pos in selected_list:
        original_token_id = input_ids[pos]
        labels[pos] = original_token_id

        rand = random.random()
        if rand < span_mlm_config.mask_token_prob:
            # mask_token_prob %: replace with [MASK]
            masked_input_ids[pos] = tokenizer.mask_token_id
        elif rand < (
            span_mlm_config.mask_token_prob + span_mlm_config.plausible_replace_prob
        ):
            # plausible_replace_prob %: replace with a chemically plausible token
            original_token = smiles_id_to_token_dict.get(original_token_id, "")
            replacement_id = _get_plausible_replacement(original_token, tokenizer)
            masked_input_ids[pos] = replacement_id
        # else: keep_token_prob %: keep original token unchanged

    return masked_input_ids, labels


# ============================================================
# Dataset
# ============================================================


class MLMDataset(Dataset):
    """
    PyTorch Dataset for Masked Language Modeling training on reaction SMILES.

    Tokenizes raw reaction SMILES strings and applies MLM masking on the fly
    during ``__getitem__``. Supports two SMILES augmentation modes (random
    or canonical) and two masking strategies (standard random or graph-aware
    span masking).

    Args:
        texts (List[str]): List of reaction SMILES strings.
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding SMILES.
        mlm_config (MLMConfig): MLM masking configuration.
        max_length (int): Maximum sequence length for padding/truncation.
        use_random_smiles (bool): If True, apply random SMILES augmentation
            during ``__getitem__``.
        use_canonical_smiles (bool): If True, canonicalize SMILES during
            ``__getitem__``. Mutually exclusive with ``use_random_smiles``.
        randomize_tautomer_pct (float): Probability of applying tautomer
            randomization (only used when ``use_random_smiles`` is True).
        canonicalize_mapped_rxn_smiles_pct (float): Probability of
            canonicalizing instead of randomizing (only used when
            ``use_random_smiles`` is True).
        protected_tokens (Set[str] | None): Token strings that should never
            be masked, in addition to the tokenizer's special tokens.
        masking_mode (str): Either ``"random"`` for standard MLM masking or
            ``"span"`` for graph-aware span masking.
        span_mlm_config (SpanMLMConfig | None): Configuration for span
            masking. Required if ``masking_mode`` is ``"span"``.

    Raises:
        ValueError: If ``masking_mode`` is not ``"random"`` or ``"span"``,
            or if ``use_random_smiles`` and ``use_canonical_smiles`` are
            both True or both False.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        mlm_config: MLMConfig,
        max_length: int = 256,
        use_random_smiles=True,
        use_canonical_smiles=False,
        randomize_tautomer_pct: float = 0.10,
        canonicalize_mapped_rxn_smiles_pct: float = 0.05,
        protected_tokens: Set[str] | None = None,
        masking_mode: str = "span",
        span_mlm_config: SpanMLMConfig | None = None,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.mlm_config = mlm_config
        self.max_length = max_length

        if masking_mode not in ("random", "span"):
            raise ValueError(
                f"masking_mode must be 'random' or 'span', got '{masking_mode}'"
            )
        self.masking_mode = masking_mode
        self.span_mlm_config = span_mlm_config or SpanMLMConfig()

        if use_canonical_smiles and use_random_smiles:
            raise ValueError(
                "use_canonical_smiles and use_random_smiles cannot both be True"
            )
        if not use_canonical_smiles and not use_random_smiles:
            raise ValueError(
                "use_canonical_smiles and use_random_smiles cannot both be False"
            )
        self._use_canonical_smiles = False
        if use_canonical_smiles:
            self._use_canonical_smiles = True
        self._use_random_smiles = False
        if use_random_smiles:
            self._use_random_smiles = True

        self.randomize_tautomer_pct = randomize_tautomer_pct
        self.canonicalize_mapped_rxn_smiles_pct = canonicalize_mapped_rxn_smiles_pct

        # Build the set of protected token IDs (see section 3)
        special_token_ids = set(tokenizer.all_special_ids)
        protected_token_ids = resolve_protected_token_ids(tokenizer, protected_tokens)
        self.protected_token_ids = special_token_ids | protected_token_ids

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a single training sample with MLM masking applied.

        Applies SMILES augmentation (random or canonical) if configured,
        tokenizes the result, applies the configured masking strategy,
        and returns a dict of tensors suitable for model forward pass.

        Args:
            idx (int): Index into the dataset.

        Returns:
            Dict[str, torch.Tensor]: A dict with keys ``"input_ids"``,
            ``"attention_mask"``, ``"token_type_ids"``, and ``"labels"``,
            each a ``torch.long`` tensor of shape ``(max_length,)``.
        """
        text = self.texts[idx]

        if self._use_random_smiles:
            if random.random() > self.canonicalize_mapped_rxn_smiles_pct:
                if random.random() > self.randomize_tautomer_pct:
                    text = randomize_reaction_smiles(
                        text,
                        remove_mapping=False,
                        randomize_tautomer=False,
                    )
                else:
                    text = randomize_reaction_smiles(
                        text, remove_mapping=False, randomize_tautomer=True
                    )
            else:
                text = canonicalize_reaction_smiles(
                    text, remove_mapping=False, canonicalize_tautomer=True
                )

        if self._use_canonical_smiles:
            text = canonicalize_reaction_smiles(text)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding.get("token_type_ids", [0] * len(input_ids))

        if self.masking_mode == "span":
            masked_input_ids, labels = apply_span_mlm_masking(
                input_ids=input_ids,
                tokenizer=self.tokenizer,
                span_mlm_config=self.span_mlm_config,
                reaction_smiles=text,
                special_token_ids=self.protected_token_ids,
            )
        else:
            masked_input_ids, labels = apply_mlm_masking(
                input_ids=input_ids,
                tokenizer=self.tokenizer,
                mlm_config=self.mlm_config,
                special_token_ids=self.protected_token_ids,
            )

        return {
            "input_ids": torch.tensor(masked_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def decode_sample(
        self, idx: int, print_output: bool = True
    ) -> Dict[str, str | List[str]]:
        """
        Decode a sample from the dataset back to human-readable text.

        Shows three views of the sample:
            - original:  The original text before masking.
            - masked:    The text after masking (what the model sees).
            - labels:    Only the tokens selected for prediction, everything
                         else shown as '_'.
            - diff:      Token-level diff showing original → masked for
                         each masked position.

        Note:
            This method always uses :func:`apply_mlm_masking` regardless of
            ``self.masking_mode``, so the masking shown may differ from what
            ``__getitem__`` produces when ``masking_mode`` is ``"span"``.

        Args:
            idx (int): The index of the sample to decode.
            print_output (bool): Whether to print the decoded output to
                stdout.

        Returns:
            Dict[str, str | List[str]]: A dict with keys ``"original"``,
            ``"masked"``, ``"labels"``, and ``"diff"``.
        """
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        input_ids = encoding["input_ids"]

        # Apply masking with the same logic as __getitem__
        masked_input_ids, labels = apply_mlm_masking(
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            mlm_config=self.mlm_config,
            special_token_ids=self.protected_token_ids,
        )

        original_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        masked_text = self.tokenizer.decode(masked_input_ids, skip_special_tokens=False)

        # Build the labels view: show original token where label != -100, else '_'
        label_tokens = [
            self.tokenizer.convert_ids_to_tokens(label_id) if label_id != -100 else "_"
            for label_id in labels
        ]
        label_text = self.tokenizer.convert_tokens_to_string(
            [t for t in label_tokens if t != "_"]
        )

        # Align original vs masked token by token for a diff-style view
        original_tokens = [
            self.tokenizer.convert_ids_to_tokens(tid) for tid in input_ids
        ]
        masked_tokens = [
            self.tokenizer.convert_ids_to_tokens(tid) for tid in masked_input_ids
        ]

        diff_lines = []
        for orig, mask, label in zip(original_tokens, masked_tokens, labels):
            if label != -100:
                diff_lines.append(f"  [{orig}] -> [{mask}]  (label: {orig})")

        result = {
            "original": original_text,
            "masked": masked_text,
            "labels": label_text,
            "diff": "\n".join(diff_lines),
        }

        if print_output:
            print("=" * 60)
            print(f"Sample index : {idx}")
            print(f"Original     : {result['original']}")
            print(f"Masked       : {result['masked']}")
            print(f"Label tokens : {result['labels']}")
            print("-" * 60)
            print("Token-level diff (only masked positions):")
            print(result["diff"])
            print("=" * 60)

        return result


# ============================================================
# Model Builder
# ============================================================


def build_albert_model(model_config: ModelConfig) -> AlbertForMaskedLM:
    """
    Build an ALBERT model from scratch using the given configuration.

    Constructs a ``transformers.AlbertConfig`` from ``model_config`` and
    instantiates an ``AlbertForMaskedLM`` model. Prints the total trainable
    parameter count.

    Args:
        model_config (ModelConfig): The model architecture configuration.

    Returns:
        AlbertForMaskedLM: An ALBERT model for Masked Language Modeling.
    """
    config = AlbertConfig(
        vocab_size=model_config.vocab_size,
        embedding_size=model_config.embedding_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
        hidden_act=model_config.hidden_act,
        hidden_dropout_prob=model_config.hidden_dropout_prob,
        attention_probs_dropout_prob=model_config.attention_probs_dropout_prob,
        max_position_embeddings=model_config.max_position_embeddings,
        type_vocab_size=model_config.type_vocab_size,
        initializer_range=model_config.initializer_range,
        layer_norm_eps=model_config.layer_norm_eps,
        classifier_dropout_prob=model_config.classifier_dropout_prob,
        num_hidden_groups=model_config.num_hidden_groups,
        inner_group_num=model_config.inner_group_num,
    )

    model = AlbertForMaskedLM(config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built with {total_params:,} trainable parameters.")
    return model


# ============================================================
# Trainer
# ============================================================


class AlbertTrainer:
    """
    Trainer for unsupervised ALBERT Masked Language Modeling pre-training.

    Manages the training loop, evaluation, checkpointing, and logging for
    MLM pre-training. Uses AdamW with linear warmup scheduling and gradient
    clipping.

    Args:
        model (AlbertForMaskedLM): The ALBERT model to train.
        train_dataloader (DataLoader): DataLoader for training batches.
        training_config (TrainingConfig): Training hyperparameters.
        val_dataloader (DataLoader | None): DataLoader for validation
            batches. If None, validation is skipped.
        device (torch.device | None): Device to train on. Defaults to CUDA
            if available, otherwise CPU.
        resume_from_checkpoint (str | None): Path to a ``.pt`` checkpoint
            file to resume training from. Restores model, optimizer, and
            scheduler state dicts, and resumes from the next epoch.
    """

    def __init__(
        self,
        model: AlbertForMaskedLM,
        train_dataloader: DataLoader,
        training_config: TrainingConfig,
        val_dataloader: DataLoader | None = None,
        device: torch.device | None = None,
        resume_from_checkpoint: str | None = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.training_config = training_config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch.nn.Module.to(self.model, self.device)

        self._setup_optimizer_and_scheduler()
        self._set_seed(training_config.seed)

        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        if resume_from_checkpoint is not None:
            self._load_checkpoint(resume_from_checkpoint)

    def _set_seed(self, seed: int) -> None:
        """
        Set random seeds across Python, NumPy, and PyTorch for reproducibility.

        Also enables deterministic cuDNN behavior to ensure reproducible
        results across runs with the same seed.

        Args:
            seed (int): The random seed to use.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model, optimizer, and scheduler state from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the ``.pt`` checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            KeyError: If the checkpoint is missing required state dicts.
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1

        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]

        logger.info(
            f"Resumed from checkpoint: {checkpoint_path} "
            f"(starting at epoch {self.start_epoch})"
        )

    def _setup_optimizer_and_scheduler(self) -> None:
        """
        Set up the AdamW optimizer with parameter-specific weight decay and
        a linear warmup learning-rate scheduler.

        Biases and LayerNorm weights receive zero weight decay. The total
        number of training steps is computed from the dataloader length and
        the number of epochs.
        """
        # Separate weight decay for biases and layer norms
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.training_config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.training_config.learning_rate,
            eps=self.training_config.adam_epsilon,
        )

        total_steps = len(self.train_dataloader) * self.training_config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=total_steps,
        )

    def train_epoch(self, epoch: int) -> float:
        """
        Run one training epoch and return the average loss.

        Iterates over the training dataloader, computes MLM loss, performs
        backward pass, gradient clipping, optimizer step, and scheduler
        step. Logs loss and learning rate at ``logging_steps`` intervals.

        Args:
            epoch (int): The 1-indexed epoch number (for logging only).

        Returns:
            float: The average MLM loss across all batches in the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)

        for step, batch in enumerate(self.train_dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                labels=batch["labels"],
            )

            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.training_config.max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if (step + 1) % self.training_config.logging_steps == 0:
                step_loss = loss.item()
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch} | Step {step + 1}/{num_batches} "
                    f"| Loss: {step_loss:.4f} | LR: {lr:.2e}"
                )

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Run evaluation on the validation set and return the average loss.

        Returns:
            float: The average MLM loss across all validation batches.
            Returns 0.0 if no validation dataloader is configured.
        """
        if self.val_dataloader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0

        for batch in self.val_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                labels=batch["labels"],
            )
            total_loss += outputs.loss.item()

        return total_loss / len(self.val_dataloader)

    def train(self) -> None:
        """
        Run the full training loop across all epochs.

        For each epoch: trains, evaluates, logs metrics, and saves a
        checkpoint (both HuggingFace ``save_pretrained`` format and a raw
        ``.pt`` file with model/optimizer/scheduler state dicts). If
        ``save_best_model`` is enabled in the training config, the best
        model (by validation loss) is saved to ``{output_dir}/best_model.pt``.
        If ``early_stopping_patience`` is set, training stops when validation
        loss has not improved for the specified number of epochs.

        Training resumes from ``self.start_epoch`` if a checkpoint was loaded.
        """
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Epochs: {self.training_config.num_epochs}")
        logger.info(f"Batch size: {self.training_config.batch_size}")

        Path(self.training_config.output_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(self.start_epoch, self.training_config.num_epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            logger.info(
                f"Epoch {epoch} complete | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {time.time() - start_time:.2f}s"
            )

            save_path = f"{self.training_config.output_dir}/checkpoint-epoch-{epoch}"

            self.model.save_pretrained(save_path)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": self.best_val_loss,
                },
                f"{save_path}.pt",
            )
            logger.info(f"Checkpoint saved to {save_path} (+ {save_path}.pt)")

            # Best-model saving and early stopping tracking
            improved = (
                val_loss
                < self.best_val_loss - self.training_config.early_stopping_min_delta
            )
            if improved:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                if self.training_config.save_best_model:
                    best_path = f"{self.training_config.output_dir}/best_model.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "val_loss": val_loss,
                        },
                        best_path,
                    )
                    logger.info(
                        f"New best model saved to {best_path} (val_loss: {val_loss:.4f})"
                    )
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if (
                self.training_config.early_stopping_patience > 0
                and self.epochs_without_improvement
                >= self.training_config.early_stopping_patience
            ):
                logger.warning(
                    f"Early stopping triggered after {self.epochs_without_improvement} "
                    f"epochs without improvement"
                )
                break


# ============================================================
# Main
# ============================================================


def main(
    train_texts: List[str],
    val_texts: List[str],
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    mlm_config: Optional[MLMConfig] = None,
    max_length: int = 384,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    protected_tokens: Optional[Set[str]] = None,
    masking_mode: str = "span",
    span_mlm_config: Optional[SpanMLMConfig] = None,
    resume_from_checkpoint: str | None = None,
):
    """
    Run unsupervised ALBERT MLM pre-training end-to-end.

    Creates the tokenizer, datasets, dataloaders, model, and trainer, then
    launches training. Uses graph-aware span masking by default with
    ``mlm_probability=0.20`` and ``max_length=384``.

    Args:
        train_texts (List[str]): List of reaction SMILES for training.
        val_texts (List[str]): List of reaction SMILES for validation.
        model_config (Optional[ModelConfig]): Model architecture config.
            Defaults to ``ModelConfig()`` if None.
        training_config (Optional[TrainingConfig]): Training hyperparameters.
            Defaults to ``TrainingConfig()`` if None.
        mlm_config (Optional[MLMConfig]): MLM masking config. Defaults to
            ``MLMConfig()`` if None.
        max_length (int): Maximum sequence length for padding/truncation.
        num_workers (int): Number of DataLoader worker processes.
        prefetch_factor (int): Number of batches prefetched per worker.
        protected_tokens (Optional[Set[str]]): Token strings that should
            never be masked. Defaults to ``{"^", "$", ".", ">>"}`` if None.
        masking_mode (str): Either ``"random"`` or ``"span"``.
        span_mlm_config (Optional[SpanMLMConfig]): Span masking
            configuration. Defaults to
            ``SpanMLMConfig(mlm_probability=0.20, ...)`` if None.
        resume_from_checkpoint (str | None): Path to a ``.pt`` checkpoint
            file to resume training from.
    """
    # --- Tokenizer ---
    tokenizer = CustomTokenizer(smiles_token_to_id_dict)

    # --- Configure everything ---
    if not model_config:
        model_config = ModelConfig()

    if not training_config:
        training_config = TrainingConfig()

    if not mlm_config:
        mlm_config = MLMConfig()

    if protected_tokens is None:
        protected_tokens = {"^", "$", ".", ">>"}

    if span_mlm_config is None:
        span_mlm_config = SpanMLMConfig(
            mlm_probability=0.20,
            span_size_weights={1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1},
        )

    # --- Datasets and Dataloaders ---
    train_dataset = MLMDataset(
        train_texts,
        tokenizer,
        mlm_config,
        protected_tokens=protected_tokens,
        max_length=max_length,
        masking_mode=masking_mode,
        span_mlm_config=span_mlm_config,
    )
    val_dataset = MLMDataset(
        val_texts,
        tokenizer,
        mlm_config,
        protected_tokens=protected_tokens,
        max_length=max_length,
        masking_mode=masking_mode,
        span_mlm_config=span_mlm_config,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )

    # --- Build model ---
    model = build_albert_model(model_config)

    # --- Train ---
    trainer = AlbertTrainer(
        model=model,
        train_dataloader=train_dataloader,
        training_config=training_config,
        val_dataloader=val_dataloader,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    trainer.train()
