import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
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

from agave_chem.mappers.neural.constants import (  # noqa: E402
    smiles_id_to_token_dict,
    smiles_token_to_id_dict,
    token_atom_identity_dict,
)
from agave_chem.mappers.neural.tokenizer import CustomTokenizer  # noqa: E402
from agave_chem.utils.chem_utils import (  # noqa: E402
    canonicalize_reaction_smiles,
    randomize_reaction_smiles,
)

# ============================================================
# Configuration
# ============================================================


@dataclass
class ModelConfig:
    """Configuration for the ALBERT model architecture."""

    vocab_size: int = 1024
    embedding_size: int = 128
    hidden_size: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    intermediate_size: int = 512
    hidden_act: str = "gelu_new"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    classifier_dropout_prob: float = 0.1
    num_hidden_groups: int = 1
    inner_group_num: int = 1


@dataclass
class TrainingConfig:
    """Configuration for the training process."""

    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    batch_size: int = 32
    warmup_steps: int = 10000
    save_steps: int = 1000  # This currently does nothing
    logging_steps: int = 100
    output_dir: str = "./albert_output"
    seed: int = 42
    fp16: bool = False


@dataclass
class MLMConfig:
    """Configuration for Masked Language Modeling preprocessing."""

    mlm_probability: float = 0.15
    mask_token_prob: float = 0.80  # 80% replace with [MASK]
    random_token_prob: float = 0.10  # 10% replace with random token
    keep_token_prob: float = 0.10  # 10% keep original token


@dataclass
class SpanMLMConfig:
    """
    Configuration for graph-aware span-based Masked Language Modeling.

    Instead of randomly selecting individual tokens, this strategy selects
    contiguous neighborhoods of atoms on the molecular graph for masking.
    Only atom tokens are eligible; structural tokens (bonds, parentheses,
    ring numbers, etc.) are never masked.
    """

    mlm_probability: float = 0.15
    span_size_weights: Dict[int, float] | None = None
    mask_token_prob: float = 0.70
    plausible_replace_prob: float = 0.20
    keep_token_prob: float = 0.10

    def __post_init__(self) -> None:
        if self.span_size_weights is None:
            self.span_size_weights = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}


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
    Preprocess a single token using the BERT/ALBERT masking strategy.

    For each selected token, apply one of three transformations:
        - Replace with [MASK] token (80%)
        - Replace with a random token (10%)
        - Keep the original token (10%)

    Args:
        token_id:       The original token ID.
        mask_token_id:  The ID of the [MASK] token.
        vocab_size:     The size of the vocabulary.
        mlm_config:     The MLM configuration.

    Returns:
        A tuple of (new_token_id, was_modified).
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
    Apply MLM masking to a sequence of token IDs.

    Randomly selects 15% of tokens to be masked, then applies the
    80/10/10 masking strategy from the BERT/ALBERT paper.

    Args:
        input_ids:          List of input token IDs.
        tokenizer:          The ALBERT tokenizer.
        mlm_config:         The MLM configuration.
        special_token_ids:  Set of special token IDs to skip.

    Returns:
        A tuple of:
            - masked_input_ids: The masked input token IDs.
            - labels:           The labels for the MLM task (-100 for
                                non-masked tokens, original ID for masked).
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
    Replace selected tokens ONLY with the [MASK] token (no random/keep).

    A simplified version of the masking strategy that always replaces
    selected tokens with [MASK], useful for inference or analysis.

    Args:
        input_ids:          List of input token IDs.
        tokenizer:          The ALBERT tokenizer.
        mlm_config:         The MLM configuration.
        special_token_ids:  Set of special token IDs to skip.

    Returns:
        A tuple of:
            - masked_input_ids: The masked input token IDs.
            - labels:           The labels for the MLM task.
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
    Replace selected tokens ONLY with random tokens (no mask/keep).

    Useful for studying the effect of random token replacement in isolation.

    Args:
        input_ids:          List of input token IDs.
        tokenizer:          The ALBERT tokenizer.
        mlm_config:         The MLM configuration.
        special_token_ids:  Set of special token IDs to skip.

    Returns:
        A tuple of:
            - noised_input_ids: The token IDs with random replacements.
            - labels:           The labels for the MLM task.
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

    The 10% 'keep original' strategy from the BERT/ALBERT paper, useful
    for studying the effect of unchanged token prediction in isolation.

    Args:
        input_ids:          List of input token IDs.
        tokenizer:          The ALBERT tokenizer.
        mlm_config:         The MLM configuration.
        special_token_ids:  Set of special token IDs to skip.

    Returns:
        A tuple of:
            - input_ids: The unchanged input token IDs.
            - labels:    The labels for the MLM task.
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
    Dataset for Masked Language Modeling training.
    Tokenizes raw text and applies MLM masking on the fly.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        mlm_config: MLMConfig,
        max_length: int = 256,
        use_random_smiles=True,
        use_canonical_smiles=False,
        protected_tokens: Set[str] | None = None,
        masking_mode: str = "random",
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

        # Build the set of protected token IDs (see section 3)
        special_token_ids = set(tokenizer.all_special_ids)
        protected_token_ids = resolve_protected_token_ids(tokenizer, protected_tokens)
        self.protected_token_ids = special_token_ids | protected_token_ids

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        if self._use_random_smiles:
            text = randomize_reaction_smiles(text)

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

        Args:
            idx:          The index of the sample to decode.
            print_output: Whether to print the decoded output.

        Returns:
            A dict with 'original', 'masked', and 'labels' strings.
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

    Args:
        model_config: The model architecture configuration.

    Returns:
        An ALBERT model for Masked Language Modeling.
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
    """Trainer class for ALBERT MLM pre-training."""

    def __init__(
        self,
        model: AlbertForMaskedLM,
        train_dataloader: DataLoader,
        training_config: TrainingConfig,
        val_dataloader: DataLoader | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.training_config = training_config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self._setup_optimizer_and_scheduler()
        self._set_seed(training_config.seed)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_optimizer_and_scheduler(self) -> None:
        """Set up AdamW optimizer and linear warmup scheduler."""
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
        """Run one training epoch and return the average loss."""
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
                print(
                    f"Epoch {epoch} | Step {step + 1}/{num_batches} "
                    f"| Loss: {step_loss:.4f} | LR: {lr:.2e}"
                )

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation and return the average validation loss."""
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
        """Run the full training loop."""
        print(f"Starting training on device: {self.device}")
        print(f"Epochs: {self.training_config.num_epochs}")
        print(f"Batch size: {self.training_config.batch_size}")

        Path(self.training_config.output_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.training_config.num_epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            print(
                f"Epoch {epoch} complete | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
                f"Time: {time.time() - start_time:.2f} seconds"
            )

            save_path = f"{self.training_config.output_dir}/checkpoint-epoch-{epoch}"

            self.model.save_pretrained(save_path)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                f"{save_path}.pt",
            )
            print(f"Model saved to {save_path} (+ {save_path}.pt)")


# ============================================================
# Main
# ============================================================


def main(
    train_texts: List[str],
    val_texts: List[str],
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    mlm_config: Optional[MLMConfig] = None,
):
    # --- Tokenizer ---
    tokenizer = CustomTokenizer(smiles_token_to_id_dict)

    # --- Configure everything ---
    if not model_config:
        model_config = ModelConfig()

    if not training_config:
        training_config = TrainingConfig()

    if not mlm_config:
        mlm_config = MLMConfig()

    # --- Datasets and Dataloaders ---
    train_dataset = MLMDataset(
        train_texts,
        tokenizer,
        mlm_config,
        protected_tokens={"^", "$", ".", ">>"},
        max_length=256,
    )
    val_dataset = MLMDataset(
        val_texts,
        tokenizer,
        mlm_config,
        protected_tokens={"^", "$", ".", ">>"},
        max_length=256,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=torch.cuda.is_available(),
    )

    # --- Build model ---
    model = build_albert_model(model_config)

    # --- Train ---
    trainer = AlbertTrainer(
        model=model,
        train_dataloader=train_dataloader,
        training_config=training_config,
        val_dataloader=val_dataloader,
    )
    trainer.train()
