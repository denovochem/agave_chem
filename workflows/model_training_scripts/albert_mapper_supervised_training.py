import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AlbertForMaskedLM,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training_scripts.albert_mapper_unuspervised_training import (
    MLMConfig,
    ModelConfig,
    SpanMLMConfig,
    TrainingConfig,
    _unwrap_model,
    apply_mlm_masking,
    apply_span_mlm_masking,
    build_albert_model,
    resolve_protected_token_ids,
)
from model_training_scripts.attention_target_builder import (
    apply_attention_sink,
    assign_temp_atom_maps,
    augment_mapped_smiles,
    build_index_attn_dict,
    build_smoothed_attn_target,
    classify_tokens,
    group_mappings_by_symmetry,
)
from model_training_scripts.cli_utils import seed_worker

from agave_chem.mappers.neural.constants import (
    smiles_token_to_id_dict,
    token_atom_identity_dict,
)
from agave_chem.mappers.neural.model import (
    AlbertWithAttentionAlignment,
    SupervisedConfig,
)
from agave_chem.mappers.neural.tokenizer import CustomTokenizer
from agave_chem.utils.chem_utils import (
    remove_reaction_smiles_atom_mapping,
)

# ============================================================
# Supervised Utils
# ============================================================


def build_attention_target_from_mapped_rxn_smiles(
    tokenizer: PreTrainedTokenizer,
    mapped_rxn_smiles: str,
    token_atom_identity_dict: Optional[Dict[str, int]] = None,
    randomize_mapped_rxn_smiles: bool = True,
    randomize_tautomer_pct: float = 0.10,
    canonicalize_mapped_rxn_smiles_pct: float = 0.05,
    canonicalize_only: bool = False,
    attn_sink_non_mapped_atoms: bool = True,
    smooth_symmetric_targets: bool = True,
    resonance_equivalence: bool = True,
    seed: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, str]]:
    """
    Build an attention alignment target matrix from a mapped reaction SMILES.

    Wrapper around :func:`_build_attention_target_from_mapped_rxn_smiles_impl`
    that catches all exceptions and returns ``None`` on failure. This allows
    callers to filter out invalid reactions without handling exceptions
    individually. Failures are logged at WARNING level with the exception
    type and message.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer for SMILES processing.
        mapped_rxn_smiles (str): Atom-mapped reaction SMILES (e.g.
            ``"[C:1]([O:2])=[O:3]>>[C:1]([O:2])[O:3]"``).
        token_atom_identity_dict (Optional[Dict[str, int]]): Mapping from
            token strings to atomic numbers. Used to identify which tokens
            represent atoms vs. structural elements.
        randomize_mapped_rxn_smiles (bool): If True, apply random SMILES
            augmentation to the mapped reaction.
        randomize_tautomer_pct (float): Probability of tautomer randomization
            during SMILES augmentation.
        canonicalize_mapped_rxn_smiles_pct (float): Probability of
            canonicalization during SMILES augmentation.
        canonicalize_only (bool): If True, always canonicalize the SMILES
            instead of randomizing. Intended for validation to match
            inference behavior.
        attn_sink_non_mapped_atoms (bool): If True, non-atom tokens and
            unmapped reactant atoms attend to the sink position (last token).
        smooth_symmetric_targets (bool): If True, spread attention target
            weight uniformly across symmetry-equivalent atoms.
        resonance_equivalence (bool): If True, merge resonance-equivalent
            atom pairs (e.g. nitro group oxygens) into symmetry groups so
            that attention targets are smoothed across them. Defaults to
            True.
        seed (Optional[int]): If provided, seeds ``random`` before the
            augmentation block for deterministic output. When ``None``,
            behavior is unchanged (uses global random state).

    Returns:
        Optional[Tuple[np.ndarray, str]]: A tuple of
            ``(attention_target, unmapped_rxn_smiles)`` where
            ``attention_target`` is an ``(N, N)`` float32 array and
            ``unmapped_rxn_smiles`` is the reaction SMILES with atom mapping
            removed. Returns ``None`` if processing fails.
    """
    try:
        return _build_attention_target_from_mapped_rxn_smiles_impl(
            tokenizer=tokenizer,
            mapped_rxn_smiles=mapped_rxn_smiles,
            token_atom_identity_dict=token_atom_identity_dict,
            randomize_mapped_rxn_smiles=randomize_mapped_rxn_smiles,
            randomize_tautomer_pct=randomize_tautomer_pct,
            canonicalize_mapped_rxn_smiles_pct=canonicalize_mapped_rxn_smiles_pct,
            canonicalize_only=canonicalize_only,
            attn_sink_non_mapped_atoms=attn_sink_non_mapped_atoms,
            smooth_symmetric_targets=smooth_symmetric_targets,
            resonance_equivalence=resonance_equivalence,
            seed=seed,
        )
    except Exception as e:
        logger.warning(
            f"Failed to build attention target for: "
            f"{mapped_rxn_smiles[:100]}... | {type(e).__name__}: {e}"
        )
        return None


def _randomize_and_unmap_rxn_smiles(
    mapped_rxn_smiles: str,
    randomize_mapped_rxn_smiles: bool = True,
    randomize_tautomer_pct: float = 0.10,
    canonicalize_mapped_rxn_smiles_pct: float = 0.05,
    canonicalize_only: bool = False,
    seed: Optional[int] = None,
) -> Optional[str]:
    """
    Randomize a mapped reaction SMILES and return the unmapped version.

    Lightweight alternative to ``build_attention_target_from_mapped_rxn_smiles``
    that performs only SMILES augmentation and atom-map removal, skipping the
    expensive attention target construction (token classification, symmetry
    smoothing, matrix building). Intended for the MLM view where only the
    unmapped text is needed.

    Args:
        mapped_rxn_smiles (str): Atom-mapped reaction SMILES.
        randomize_mapped_rxn_smiles (bool): If True, apply random SMILES
            augmentation to the mapped reaction.
        randomize_tautomer_pct (float): Probability of tautomer randomization
            during SMILES augmentation.
        canonicalize_mapped_rxn_smiles_pct (float): Probability of
            canonicalization during augmentation.
        canonicalize_only (bool): If True, always canonicalize the SMILES
            instead of randomizing. Intended for validation to match
            inference behavior.
        seed (Optional[int]): If provided, seeds ``random`` before the
            augmentation block for deterministic output.

    Returns:
        Optional[str]: The unmapped reaction SMILES with atom mapping removed,
        or ``None`` if processing fails.
    """
    try:
        new_mapped_rxn_smiles = augment_mapped_smiles(
            mapped_rxn_smiles,
            randomize_mapped_rxn_smiles=randomize_mapped_rxn_smiles,
            randomize_tautomer_pct=randomize_tautomer_pct,
            canonicalize_mapped_rxn_smiles_pct=canonicalize_mapped_rxn_smiles_pct,
            canonicalize_only=canonicalize_only,
            seed=seed,
        )
        return remove_reaction_smiles_atom_mapping(new_mapped_rxn_smiles)
    except Exception as e:
        logger.warning(
            f"Failed to randomize/unmap: "
            f"{mapped_rxn_smiles[:100]}... | {type(e).__name__}: {e}"
        )
        return None


def _build_attention_target_from_mapped_rxn_smiles_impl(
    tokenizer: PreTrainedTokenizer,
    mapped_rxn_smiles: str,
    token_atom_identity_dict: Optional[Dict[str, int]] = None,
    randomize_mapped_rxn_smiles: bool = True,
    randomize_tautomer_pct: float = 0.10,
    canonicalize_mapped_rxn_smiles_pct: float = 0.05,
    canonicalize_only: bool = False,
    attn_sink_non_mapped_atoms: bool = True,
    smooth_symmetric_targets: bool = True,
    resonance_equivalence: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    """
    Implementation for building an attention alignment target matrix.

    Processes a mapped reaction SMILES through several phases:
        1. Assigns temporary atom map numbers (600+ for reactants, 800+ for
           products) to unmapped atoms.
        2. Identifies symmetry groups in reactants and products, optionally
           merging resonance-equivalent atom pairs.
        3. Tokenizes both the mapped and unmapped reaction SMILES.
        4. Matches atom-mapped tokens between reactants and products by
           atom map number (only pairs appearing exactly twice are kept).
        5. Builds an ``(N, N)`` attention target matrix where
           ``attn_target[src, dst] = 1.0`` means token ``src`` should
           attend to token ``dst``.
        6. If ``smooth_symmetric_targets`` is True, spreads the target
           weight uniformly across symmetry-equivalent atoms.
        7. If ``attn_sink_non_mapped_atoms`` is True, sets the last column
           (sink position) to 1.0 for non-atom tokens and unmapped atoms.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer for SMILES processing.
        mapped_rxn_smiles (str): Atom-mapped reaction SMILES.
        token_atom_identity_dict (Optional[Dict[str, int]]): Mapping from
            token strings to atomic numbers.
        randomize_mapped_rxn_smiles (bool): If True, apply random SMILES
            augmentation.
        randomize_tautomer_pct (float): Probability of tautomer randomization.
        canonicalize_mapped_rxn_smiles_pct (float): Probability of
            canonicalization during augmentation.
        canonicalize_only (bool): If True, always canonicalize the SMILES
            instead of randomizing. Intended for validation to match
            inference behavior.
        attn_sink_non_mapped_atoms (bool): If True, route non-atom and
            unmapped atom attention to the sink position.
        smooth_symmetric_targets (bool): If True, spread attention across
            symmetry-equivalent atoms.
        resonance_equivalence (bool): If True, merge resonance-equivalent
            atom pairs into symmetry groups so that attention targets are
            smoothed across them. Defaults to True.
        seed (Optional[int]): If provided, seeds ``random`` before the
            augmentation block so that all stochastic operations (the
            ``random.random()`` branching decisions and the internal
            ``random.shuffle``/``random.choice`` calls inside
            ``randomize_reaction_smiles``) are fully deterministic.
            When ``None``, behavior is unchanged (uses global random state).

    Returns:
        Tuple[np.ndarray, str]: A tuple of ``(attention_target,
        unmapped_rxn_smiles)`` where ``attention_target`` is an
        ``(N, N)`` float32 array and ``unmapped_rxn_smiles`` is the
        reaction SMILES with atom mapping removed.
    """

    if token_atom_identity_dict is None:
        token_atom_identity_dict = {}

    # Phase 1: Assign temporary atom maps and compute symmetry
    temp = assign_temp_atom_maps(
        mapped_rxn_smiles, resonance_equivalence=resonance_equivalence
    )

    # Phase 2: Augment (randomize / canonicalize) the mapped SMILES
    new_mapped_rxn_smiles = augment_mapped_smiles(
        temp.new_mapped_rxn_smiles,
        randomize_mapped_rxn_smiles=randomize_mapped_rxn_smiles,
        randomize_tautomer_pct=randomize_tautomer_pct,
        canonicalize_mapped_rxn_smiles_pct=canonicalize_mapped_rxn_smiles_pct,
        canonicalize_only=canonicalize_only,
        seed=seed,
    )

    # Phase 3: Tokenize
    tokens = tokenizer.preprocess_sentence_reaction_smiles(
        new_mapped_rxn_smiles
    ).split()

    unmapped_rxn_smiles = remove_reaction_smiles_atom_mapping(new_mapped_rxn_smiles)

    unmapped_tokens = tokenizer.preprocess_sentence_reaction_smiles(
        unmapped_rxn_smiles
    ).split()

    # Phase 4: Classify tokens and build matching dict
    classified = classify_tokens(
        tokens=tokens,
        unmapped_tokens=unmapped_tokens,
        token_atom_identity_dict=token_atom_identity_dict,
        symmetric_atom_token_indices_to_not_sink=temp.symmetric_atom_token_indices_to_not_sink,
        all_product_atoms_mapped=temp.all_product_atoms_mapped,
        atom_map_nums_to_sink_atomic_num_not_in_product=temp.atom_map_nums_to_sink_atomic_num_not_in_product,
    )

    # Phase 5: Build bidirectional attention index mapping
    index_attn_dict = build_index_attn_dict(classified.matching_tokens_dict)

    n = len(tokens)

    # Phase 6: Build attention target matrix
    if smooth_symmetric_targets:
        attn_target = build_smoothed_attn_target(
            index_attn_dict=index_attn_dict,
            token_index_to_mapnum=classified.token_index_to_mapnum,
            tokens=tokens,
            reactant_symmetry_groups=temp.reactant_symmetry_groups,
            product_symmetry_groups=temp.product_symmetry_groups,
            n=n,
        )
    else:
        attn_target = np.zeros((n, n), dtype=np.float32)
        for src, dst in index_attn_dict.items():
            attn_target[src, dst] = 1.0

    # Phase 7: Apply attention sink for non-atom and unmapped tokens
    if not attn_sink_non_mapped_atoms:
        return attn_target, unmapped_rxn_smiles

    attn_target = apply_attention_sink(
        attn_target=attn_target,
        non_atom_token_indices=classified.non_atom_token_indices,
        atom_token_indices_to_sink=classified.atom_token_indices_to_sink,
    )

    return attn_target, unmapped_rxn_smiles


_MAX_GETITEM_RETRIES = 10


# ============================================================
# Supervised Dataset
# ============================================================


class SupervisedAtomMappingDataset(Dataset):
    """
    PyTorch Dataset for supervised attention alignment training with dual views.

    Each sample is a mapped reaction SMILES. On ``__getitem__``, the dataset
    builds attention target matrices from the atom mapping **twice** with
    independent SMILES randomizations, producing two views:

    - **MLM view** (``mlm_*`` keys): The unmapped SMILES is tokenized and
      MLM masking is applied. This view is used only for the MLM loss.
    - **Alignment view** (``align_*`` keys): The unmapped SMILES is
      tokenized without masking. This view is used only for the attention
      alignment loss, ensuring the model learns alignment on the same input
      distribution it will see at inference time (no masks).

    If a sample fails to process (e.g. invalid SMILES), a random replacement
    index is tried, up to ``_MAX_GETITEM_RETRIES`` times. Each retry is logged
    at DEBUG level; exhaustion of all retries is logged at WARNING level.

    Args:
        texts (List[str]): List of atom-mapped reaction SMILES strings.
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding SMILES.
        mlm_config (MLMConfig): MLM masking configuration.
        max_length (int): Maximum sequence length for padding/truncation.
        use_random_smiles (bool): If True, apply random SMILES augmentation
            when building attention targets. Mutually exclusive with
            ``use_canonical_smiles``. When both are False, SMILES are
            passed through unchanged (no augmentation).
        use_canonical_smiles (bool): If True, always canonicalize SMILES
            instead of randomizing. Intended for validation to match
            inference behavior. Mutually exclusive with
            ``use_random_smiles``.
        canonicalize_mapped_rxn_smiles_pct (float): Probability of
            canonicalizing instead of randomizing (only used when
            ``use_random_smiles`` is True).
        protected_tokens (Set[str] | None): Token strings that should never
            be masked.
        smooth_symmetric_targets (bool): If True, spread attention target
            weight across symmetry-equivalent atoms.
        resonance_equivalence (bool): If True, merge resonance-equivalent
            atom pairs (e.g. nitro group oxygens) into symmetry groups so
            that attention targets are smoothed across them. Defaults to
            True.
        masking_mode (str): Either ``"random"`` or ``"span"``.
        span_mlm_config (SpanMLMConfig | None): Span masking configuration.
            Required if ``masking_mode`` is ``"span"``.

    Raises:
        ValueError: If ``masking_mode`` is not ``"random"`` or ``"span"``,
            or if ``use_random_smiles`` and ``use_canonical_smiles`` are
            both True.
        RuntimeError: If a valid sample cannot be loaded after
            ``_MAX_GETITEM_RETRIES`` attempts.
    """

    def __init__(
        self,
        texts: List[str],  # mapped reaction SMILES
        tokenizer: PreTrainedTokenizer,
        mlm_config: MLMConfig,
        max_length: int = 256,
        use_random_smiles: bool = True,
        use_canonical_smiles: bool = False,
        canonicalize_mapped_rxn_smiles_pct: float = 0.05,
        protected_tokens: Set[str] | None = None,
        smooth_symmetric_targets: bool = True,
        resonance_equivalence: bool = True,
        masking_mode: str = "span",
        span_mlm_config: SpanMLMConfig | None = None,
    ):
        if use_random_smiles and use_canonical_smiles:
            raise ValueError(
                "use_random_smiles and use_canonical_smiles cannot both be True"
            )

        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.mlm_config = mlm_config
        self.max_length = max_length
        self.use_random_smiles = use_random_smiles
        self.use_canonical_smiles = use_canonical_smiles
        self.canonicalize_mapped_rxn_smiles_pct = canonicalize_mapped_rxn_smiles_pct
        self.smooth_symmetric_targets = smooth_symmetric_targets
        self.resonance_equivalence = resonance_equivalence

        if masking_mode not in ("random", "span"):
            raise ValueError(
                f"masking_mode must be 'random' or 'span', got '{masking_mode}'"
            )
        self.masking_mode = masking_mode
        self.span_mlm_config = span_mlm_config or SpanMLMConfig()

        special_token_ids = set(tokenizer.all_special_ids)
        protected_token_ids = resolve_protected_token_ids(tokenizer, protected_tokens)
        self.protected_token_ids = special_token_ids | protected_token_ids

        # Cache the vocabulary once to avoid repeated dict copies in
        # _get_plausible_replacement during span masking.
        self._vocab_cache = tokenizer.get_vocab()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a single training sample with dual MLM and alignment views.

        Calls ``build_attention_target_from_mapped_rxn_smiles`` twice with
        independent randomizations to produce:
        - An MLM view (masked input + labels) under ``mlm_*`` keys.
        - An alignment view (unmasked input + attention target) under
          ``align_*`` keys.

        If processing fails, retries with a random replacement index up to
        ``_MAX_GETITEM_RETRIES`` times.

        Args:
            idx (int): Index into the dataset.

        Returns:
            Dict[str, torch.Tensor]: A dict with keys ``mlm_input_ids``,
            ``mlm_attention_mask``, ``mlm_token_type_ids``, ``mlm_labels``,
            ``align_input_ids``, ``align_attention_mask``,
            ``align_token_type_ids``, ``align_attention_target``, and
            ``align_attention_loss_mask``.

        Raises:
            RuntimeError: If no valid sample can be loaded after
                ``_MAX_GETITEM_RETRIES`` attempts.
        """
        for _attempt in range(_MAX_GETITEM_RETRIES):
            try:
                mapped_text = self.texts[idx]

                # MLM view: lightweight randomization + unmapping (no attn target)
                mlm_unmapped_text = _randomize_and_unmap_rxn_smiles(
                    mapped_rxn_smiles=mapped_text,
                    randomize_mapped_rxn_smiles=self.use_random_smiles,
                    canonicalize_mapped_rxn_smiles_pct=self.canonicalize_mapped_rxn_smiles_pct,
                    canonicalize_only=self.use_canonical_smiles,
                )
                if mlm_unmapped_text is None:
                    logger.debug(
                        f"MLM view at index {idx} returned None, "
                        f"retrying with random index"
                    )
                    idx = random.randrange(len(self.texts))
                    continue

                # Alignment view: full attention target construction
                align_result = build_attention_target_from_mapped_rxn_smiles(
                    tokenizer=self.tokenizer,
                    mapped_rxn_smiles=mapped_text,
                    token_atom_identity_dict=token_atom_identity_dict,
                    randomize_mapped_rxn_smiles=self.use_random_smiles,
                    canonicalize_mapped_rxn_smiles_pct=self.canonicalize_mapped_rxn_smiles_pct,
                    canonicalize_only=self.use_canonical_smiles,
                    smooth_symmetric_targets=self.smooth_symmetric_targets,
                    resonance_equivalence=self.resonance_equivalence,
                )
                if align_result is None:
                    logger.debug(
                        f"Alignment view at index {idx} returned None, "
                        f"retrying with random index"
                    )
                    idx = random.randrange(len(self.texts))
                    continue

                align_attn_target, align_unmapped_text = align_result
            except Exception as e:
                logger.debug(
                    f"Sample at index {idx} failed: {type(e).__name__}: {e}, retrying"
                )
                idx = random.randrange(len(self.texts))
                continue

            try:
                mlm_sample = self._build_mlm_sample(mlm_unmapped_text)
                align_sample = self._build_alignment_sample(
                    align_attn_target, align_unmapped_text
                )
                return {**mlm_sample, **align_sample}
            except Exception as e:
                logger.debug(
                    f"_build_sample failed for index {idx}: "
                    f"{type(e).__name__}: {e}, retrying"
                )
                idx = random.randrange(len(self.texts))
                continue

        logger.warning(
            f"Failed to load a valid sample after {_MAX_GETITEM_RETRIES} attempts"
        )
        raise RuntimeError(
            f"Failed to load a valid sample after {_MAX_GETITEM_RETRIES} attempts"
        )

    def _build_mlm_sample(self, unmapped_text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize and build the MLM view (masked input + labels).

        Tokenizes the unmapped reaction SMILES, applies MLM masking (span or
        random depending on ``masking_mode``), and pads to ``max_length``.

        Args:
            unmapped_text (str): Reaction SMILES with atom mapping removed.

        Returns:
            Dict[str, torch.Tensor]: A dict with keys ``mlm_input_ids``,
            ``mlm_attention_mask``, ``mlm_token_type_ids``, and
            ``mlm_labels``.
        """
        encoding = self.tokenizer(
            unmapped_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=False,
            return_tensors=None,
        )

        input_ids = encoding["input_ids"]
        if len(input_ids) > self.max_length:
            raise ValueError(
                f"Tokenized sequence length {len(input_ids)} exceeds "
                f"max_length {self.max_length}"
            )
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding.get("token_type_ids", [0] * len(input_ids))

        if self.masking_mode == "span":
            masked_input_ids, labels = apply_span_mlm_masking(
                input_ids=input_ids,
                tokenizer=self.tokenizer,
                span_mlm_config=self.span_mlm_config,
                reaction_smiles=unmapped_text,
                special_token_ids=self.protected_token_ids,
                vocab=self._vocab_cache,
            )
        else:
            masked_input_ids, labels = apply_mlm_masking(
                input_ids=input_ids,
                tokenizer=self.tokenizer,
                mlm_config=self.mlm_config,
                special_token_ids=self.protected_token_ids,
            )

        return {
            "mlm_input_ids": torch.tensor(masked_input_ids, dtype=torch.long),
            "mlm_attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "mlm_token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "mlm_labels": torch.tensor(labels, dtype=torch.long),
        }

    def _build_alignment_sample(
        self, attention_target: np.ndarray, unmapped_text: str
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize and build the alignment view (unmasked input + attention target).

        Tokenizes the unmapped reaction SMILES without any MLM masking, pads
        the attention target to ``max_length × max_length``, and computes the
        attention loss mask (1 for rows with non-zero attention target, 0
        otherwise).

        Args:
            attention_target (np.ndarray): ``(N, N)`` attention target array.
            unmapped_text (str): Reaction SMILES with atom mapping removed.

        Returns:
            Dict[str, torch.Tensor]: A dict with keys ``align_input_ids``,
            ``align_attention_mask``, ``align_token_type_ids``,
            ``align_attention_target``, and ``align_attention_loss_mask``.
        """
        encoding = self.tokenizer(
            unmapped_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=False,
            return_tensors=None,
        )

        input_ids = encoding["input_ids"]
        if len(input_ids) > self.max_length:
            raise ValueError(
                f"Tokenized sequence length {len(input_ids)} exceeds "
                f"max_length {self.max_length}"
            )
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding.get("token_type_ids", [0] * len(input_ids))

        padded_attention_target = np.zeros(
            (self.max_length, self.max_length), dtype=np.float32
        )

        orig_len = min(attention_target.shape[0], self.max_length)
        orig_width = min(attention_target.shape[1], self.max_length)
        padded_attention_target[:orig_len, :orig_width] = attention_target[
            :orig_len, :orig_width
        ]

        attention_loss_mask = (padded_attention_target.sum(axis=1) > 0).astype(
            np.float32
        )

        return {
            "align_input_ids": torch.tensor(input_ids, dtype=torch.long),
            "align_attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "align_token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "align_attention_target": torch.tensor(
                padded_attention_target, dtype=torch.float32
            ),
            "align_attention_loss_mask": torch.tensor(
                attention_loss_mask, dtype=torch.float32
            ),
        }


# ============================================================
# Supervised Trainer
# ============================================================


class SupervisedAlbertTrainer:
    """
    Trainer for supervised attention alignment with dual forward passes.

    Manages the training loop, evaluation, checkpointing, and logging for
    supervised attention alignment. Each batch produces two independent
    forward passes through the shared encoder:

    - **MLM pass**: Masked input → MLM loss only.
    - **Alignment pass**: Unmasked input → attention alignment loss only.

    Both losses' gradients are accumulated before the optimizer step,
    ensuring the alignment task learns on the same unmasked input
    distribution it will see at inference time. When
    ``supervised_config.multitask`` is False, only the alignment pass runs.

    Supports gradient checkpointing (via ``training_config.use_gradient_checkpointing``),
    mixed precision (AMP), gradient accumulation, and ``torch.compile``.
    Uses AdamW with linear warmup scheduling and gradient clipping.

    Args:
        model (AlbertWithAttentionAlignment): The supervised ALBERT model
            with attention alignment head.
        train_dataloader (DataLoader): DataLoader for training batches.
            Each batch must contain ``mlm_*`` and ``align_*`` namespaced
            keys produced by ``SupervisedAtomMappingDataset``.
        training_config (TrainingConfig): Training hyperparameters.
        supervised_config (SupervisedConfig): Supervised attention
            alignment configuration (target layer, loss weight, etc.).
        tokenizer (PreTrainedTokenizer): Tokenizer used during training.
            Saved alongside model checkpoints via ``save_pretrained``.
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
        model: AlbertWithAttentionAlignment,
        train_dataloader: DataLoader,
        training_config: TrainingConfig,
        supervised_config: SupervisedConfig,
        tokenizer: PreTrainedTokenizer,
        val_dataloader: DataLoader | None = None,
        device: torch.device | None = None,
        resume_from_checkpoint: str | None = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.training_config = training_config
        self.supervised_config = supervised_config
        self.tokenizer = tokenizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Mixed precision setup
        self.use_amp = training_config.use_amp and self.device.type == "cuda"
        self.amp_dtype = (
            torch.float16 if training_config.amp_dtype == "float16" else torch.bfloat16
        )
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler: Optional[torch.amp.GradScaler] = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # Gradient checkpointing
        if training_config.use_gradient_checkpointing:
            unwrapped = _unwrap_model(self.model)
            unwrapped.base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on base model")

        # torch.compile
        if training_config.compile_model:
            logger.info("Compiling model with torch.compile (first epoch may be slow)")
            self.model = torch.compile(self.model)  # type: ignore[assignment]

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

        Also configures deterministic cuDNN behavior based on
        ``training_config.deterministic``.

        Args:
            seed (int): The random seed to use.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = self.training_config.deterministic
        torch.backends.cudnn.benchmark = not self.training_config.deterministic
        torch.set_float32_matmul_precision("high")

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
        _unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
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

        Biases and LayerNorm weights receive zero weight decay. Only
        parameters with ``requires_grad=True`` are included.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.training_config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.training_config.learning_rate,
            eps=self.training_config.adam_epsilon,
            fused=self.device.type == "cuda",
        )

        total_steps = (
            len(self.train_dataloader) * self.training_config.num_epochs
        ) // self.training_config.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=total_steps,
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one training epoch with dual forward passes and return average losses.

        For each batch, performs two sequential forward passes through the
        shared encoder:
        1. **MLM pass** on masked input (``mlm_*`` keys) → MLM loss.
        2. **Alignment pass** on unmasked input (``align_*`` keys) →
           attention alignment loss.

        Both losses are scaled by ``1 / grad_accum`` and backpropagated
        independently, accumulating gradients in the shared encoder
        parameters. Gradient clipping, optimizer step, and scheduler step
        occur at the gradient accumulation cadence. When
        ``supervised_config.multitask`` is False, only the alignment pass
        runs and MLM loss is reported as 0.

        Args:
            epoch (int): The 1-indexed epoch number (for logging only).

        Returns:
            Dict[str, float]: A dict with keys ``"loss"``, ``"mlm_loss"``,
            and ``"attention_loss"``, each the average across all batches.
        """
        self.model.train()
        total_loss = torch.zeros(1, device=self.device)
        total_mlm_loss = torch.zeros(1, device=self.device)
        total_attention_loss = torch.zeros(1, device=self.device)
        num_batches = len(self.train_dataloader)
        grad_accum = self.training_config.gradient_accumulation_steps
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # --- MLM forward pass (masked input) ---
            step_mlm_loss = torch.tensor(0.0, device=self.device)
            if self.supervised_config.multitask:
                with torch.amp.autocast(
                    "cuda",
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    mlm_outputs = self.model(
                        input_ids=batch["mlm_input_ids"],
                        attention_mask=batch["mlm_attention_mask"],
                        token_type_ids=batch["mlm_token_type_ids"],
                        labels=batch["mlm_labels"],
                        attention_target=None,
                    )

                mlm_loss = mlm_outputs["mlm_loss"] / grad_accum

                if self.scaler is not None:
                    self.scaler.scale(mlm_loss).backward()
                else:
                    mlm_loss.backward()

                step_mlm_loss = mlm_outputs["mlm_loss"].detach()
                total_mlm_loss += step_mlm_loss

            # --- Alignment forward pass (unmasked input) ---
            with torch.amp.autocast(
                "cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                align_outputs = self.model(
                    input_ids=batch["align_input_ids"],
                    attention_mask=batch["align_attention_mask"],
                    token_type_ids=batch["align_token_type_ids"],
                    labels=None,
                    attention_target=batch["align_attention_target"],
                    attention_loss_mask=batch["align_attention_loss_mask"],
                )

            attn_loss = align_outputs["attention_loss"] / grad_accum

            if self.scaler is not None:
                self.scaler.scale(attn_loss).backward()
            else:
                attn_loss.backward()

            step_attn_loss = align_outputs["attention_loss"].detach()
            total_attention_loss += step_attn_loss
            total_loss += step_mlm_loss + step_attn_loss

            if (step + 1) % grad_accum == 0 or step == num_batches - 1:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.training_config.max_grad_norm
                )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            if (step + 1) % self.training_config.logging_steps == 0:
                step_mlm_loss_val = step_mlm_loss.item()
                step_attention_loss_val = step_attn_loss.item()
                step_loss = step_mlm_loss_val + step_attention_loss_val
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch} | Step {step + 1}/{num_batches} "
                    f"| Loss: {step_loss:.4f} | MLM: {step_mlm_loss_val:.4f} "
                    f"| Attn: {step_attention_loss_val:.4f} | LR: {lr:.2e}"
                )

        return {
            "loss": total_loss.item() / num_batches,
            "mlm_loss": total_mlm_loss.item() / num_batches,
            "attention_loss": total_attention_loss.item() / num_batches,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the validation set with dual forward passes.

        Mirrors the training loop: for each batch, runs an MLM pass on
        masked input (when ``multitask`` is True) and an alignment pass on
        unmasked input, accumulating both losses for reporting.

        Returns:
            Dict[str, float]: A dict with keys ``"loss"``, ``"mlm_loss"``,
            and ``"attention_loss"``, each the average across all validation
            batches. Returns zeros if no validation dataloader is configured.
        """
        if self.val_dataloader is None:
            return {"loss": 0.0, "mlm_loss": 0.0, "attention_loss": 0.0}

        self.model.eval()
        total_loss = torch.zeros(1, device=self.device)
        total_mlm_loss = torch.zeros(1, device=self.device)
        total_attention_loss = torch.zeros(1, device=self.device)

        num_batches = len(self.val_dataloader)
        for step, batch in enumerate(self.val_dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            step_mlm_loss = torch.tensor(0.0, device=self.device)

            # --- MLM forward pass (masked input) ---
            if self.supervised_config.multitask:
                with torch.amp.autocast(
                    "cuda",
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    mlm_outputs = self.model(
                        input_ids=batch["mlm_input_ids"],
                        attention_mask=batch["mlm_attention_mask"],
                        token_type_ids=batch["mlm_token_type_ids"],
                        labels=batch["mlm_labels"],
                        attention_target=None,
                    )
                step_mlm_loss = mlm_outputs["mlm_loss"].detach()
                total_mlm_loss += step_mlm_loss

            # --- Alignment forward pass (unmasked input) ---
            with torch.amp.autocast(
                "cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                align_outputs = self.model(
                    input_ids=batch["align_input_ids"],
                    attention_mask=batch["align_attention_mask"],
                    token_type_ids=batch["align_token_type_ids"],
                    labels=None,
                    attention_target=batch["align_attention_target"],
                    attention_loss_mask=batch["align_attention_loss_mask"],
                )

            step_attn_loss = align_outputs["attention_loss"].detach()
            total_attention_loss += step_attn_loss
            total_loss += step_mlm_loss + step_attn_loss

            if (step + 1) % self.training_config.logging_steps == 0:
                step_mlm_loss_val = step_mlm_loss.item()
                step_attention_loss_val = step_attn_loss.item()
                step_loss = step_mlm_loss_val + step_attention_loss_val
                logger.info(
                    f"Val | Step {step + 1}/{num_batches} "
                    f"| Loss: {step_loss:.4f} | MLM: {step_mlm_loss_val:.4f} "
                    f"| Attn: {step_attention_loss_val:.4f}"
                )

        return {
            "loss": total_loss.item() / num_batches,
            "mlm_loss": total_mlm_loss.item() / num_batches,
            "attention_loss": total_attention_loss.item() / num_batches,
        }

    def train(self) -> None:
        """
        Run the full training loop across all epochs.

        For each epoch: trains, evaluates, logs metrics, and saves a
        checkpoint (raw ``.pt`` file with model/optimizer/scheduler state
        dicts and the base model in HuggingFace ``save_pretrained`` format).
        If ``save_best_model`` is enabled in the training config, the best
        model (by validation loss) is saved to ``{output_dir}/best_model.pt``.
        If ``early_stopping_patience`` is set, training stops when validation
        loss has not improved for the specified number of epochs.

        Training resumes from ``self.start_epoch`` if a checkpoint was loaded.
        """
        mode = "multitask" if self.supervised_config.multitask else "supervised-only"
        logger.info(f"Starting {mode} training on device: {self.device}")
        logger.info(f"Target layer: {self.supervised_config.target_layer}")
        logger.info(f"Epochs: {self.training_config.num_epochs}")

        Path(self.training_config.output_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(self.start_epoch, self.training_config.num_epochs + 1):
            start_time = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            logger.info(
                f"Epoch {epoch} complete | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Time: {time.time() - start_time:.2f}s"
            )

            # Save checkpoint
            save_path = (
                f"{self.training_config.output_dir}/supervised-checkpoint-epoch-{epoch}"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": _unwrap_model(self.model).state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "best_val_loss": self.best_val_loss,
                },
                f"{save_path}.pt",
            )

            # Also save the base model for inference
            _unwrap_model(self.model).base_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Checkpoint saved to {save_path}")

            # Best-model saving and early stopping tracking
            val_loss = val_metrics["loss"]
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
                            "model_state_dict": _unwrap_model(self.model).state_dict(),
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

    @torch.no_grad()
    def predict_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get predicted attention alignment probabilities for inference.

        Runs a forward pass and applies softmax to the attention logits
        to produce normalized attention probabilities.

        Args:
            input_ids (torch.Tensor): Token IDs of shape ``(B, S)``.
            attention_mask (torch.Tensor): Attention mask of shape ``(B, S)``.
            token_type_ids (torch.Tensor): Token type IDs of shape ``(B, S)``.

        Returns:
            torch.Tensor: Softmax-normalized attention probabilities of
            shape ``(B, S, S)``.
        """
        self.model.eval()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        attention_logits = outputs["attention_logits"]  # (B, S, S)
        attention_probs = torch.softmax(attention_logits, dim=-1)

        return attention_probs


@torch.no_grad()
def evaluate_supervised_attention_loss(
    model,
    dataloader_or_dataset: DataLoader | Dataset,
    device: torch.device | None = None,
    target_layer: int | None = None,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> float:
    """
    Evaluate supervised attention alignment loss only (no MLM).

    Supports both DataLoader and raw Dataset inputs. If a Dataset is given,
    a DataLoader is created with the specified batch size and worker settings.

    If ``target_layer`` is provided, the original
    ``model.supervised_config.target_layer`` is saved and restored after
    evaluation, so the caller's model is not permanently modified.

    Args:
        model: An ``AlbertWithAttentionAlignment`` instance.
        dataloader_or_dataset (DataLoader | Dataset): Either a DataLoader
            yielding batches, or a Dataset yielding single examples.
        device (torch.device | None): Device to run eval on. If None, uses
            the model's parameter device.
        target_layer (int | None): Optional override for
            ``supervised_config.target_layer`` during eval.
        batch_size (int): Batch size to use if a Dataset is provided.
        num_workers (int): DataLoader ``num_workers`` if a Dataset is provided.
        pin_memory (bool | None): DataLoader ``pin_memory`` if a Dataset is
            provided. If None, uses ``torch.cuda.is_available()``.

    Returns:
        float: Mean attention alignment loss across all batches. Returns
        0.0 if no batches were processed.

    Raises:
        TypeError: If the model does not have a ``supervised_config``
            attribute (i.e. is not an ``AlbertWithAttentionAlignment``).
        RuntimeError: If the model forward pass does not return
            ``'attention_loss'``.
        ValueError: If ``target_layer`` is out of range for the model's
            attention tensors (must be in
            ``[0, num_hidden_layers - 1]``).
    """
    if not hasattr(model, "supervised_config"):
        raise TypeError(
            "evaluate_supervised_attention_loss requires an AlbertWithAttentionAlignment model "
            "(the supervised wrapper). You passed a model without 'supervised_config' "
            "(likely an AlbertForMaskedLM)."
        )

    if target_layer is not None:
        num_layers = model.base_model.config.num_hidden_layers
        if not (0 <= target_layer < num_layers):
            raise ValueError(
                f"target_layer must be in [0, {num_layers - 1}], got {target_layer}"
            )

    original_target_layer = model.supervised_config.target_layer
    try:
        if target_layer is not None:
            model.supervised_config.target_layer = target_layer

        model = model.to(device)
        eval_device = device if device is not None else next(model.parameters()).device

        if isinstance(dataloader_or_dataset, DataLoader):
            dl = dataloader_or_dataset
        else:
            if pin_memory is None:
                pin_memory = torch.cuda.is_available()
            dl = DataLoader(
                dataloader_or_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        model.eval()
        total_attention_loss = 0.0
        num_batches = 0

        for batch in dl:
            batch = {k: v.to(eval_device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["align_input_ids"],
                attention_mask=batch["align_attention_mask"],
                token_type_ids=batch["align_token_type_ids"],
                labels=None,
                attention_target=batch["align_attention_target"],
                attention_loss_mask=batch["align_attention_loss_mask"],
            )

            if "attention_loss" not in outputs:
                raise RuntimeError(
                    "Model forward did not return 'attention_loss'. "
                    "Ensure 'attention_target' is provided and the model supports supervised attention."
                )

            total_attention_loss += float(outputs["attention_loss"].item())
            num_batches += 1

        return 0.0 if num_batches == 0 else total_attention_loss / num_batches

    finally:
        model.supervised_config.target_layer = original_target_layer


# ============================================================
# Example Usage for Supervised Training
# ============================================================


def main_supervised(
    train_texts: List[str],  # mapped reaction SMILES
    val_texts: List[str],  # mapped reaction SMILES
    pretrained_model_path: str | None = None,
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    mlm_config: Optional[MLMConfig] = None,
    supervised_config: Optional[SupervisedConfig] = None,
    max_length: int = 384,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    protected_tokens: Optional[Set[str]] = None,
    masking_mode: str = "span",
    span_mlm_config: Optional[SpanMLMConfig] = None,
    canonicalize_mapped_rxn_smiles_pct: float = 0.05,
    resume_from_checkpoint: str | None = None,
    log_level: str = "INFO",
):
    """
    Run supervised attention alignment training end-to-end.

    Creates the tokenizer, datasets, dataloaders, model (from scratch or
    loaded from a pretrained MLM checkpoint), and trainer, then launches
    training. The dataset produces dual views per sample: a masked MLM view
    and an unmasked alignment view, each with independent SMILES
    randomization. Uses graph-aware span masking by default with
    ``mlm_probability=0.20`` and ``max_length=384``.

    Args:
        train_texts (List[str]): List of atom-mapped reaction SMILES for
            training.
        val_texts (List[str]): List of atom-mapped reaction SMILES for
            validation.
        pretrained_model_path (str | None): Path to a pretrained MLM model
            directory. If provided, the base model is loaded from there;
            otherwise a new model is built from scratch.
        model_config (Optional[ModelConfig]): Model architecture config.
            Defaults to ``ModelConfig()`` if None. Ignored if
            ``pretrained_model_path`` is provided.
        training_config (Optional[TrainingConfig]): Training hyperparameters.
            Defaults to ``TrainingConfig()`` if None.
        mlm_config (Optional[MLMConfig]): MLM masking config. Defaults to
            ``MLMConfig()`` if None.
        supervised_config (Optional[SupervisedConfig]): Supervised attention
            alignment config. Defaults to ``SupervisedConfig()`` if None.
        max_length (int): Maximum sequence length for padding/truncation.
        num_workers (int): Number of DataLoader worker processes.
        prefetch_factor (int): Number of batches prefetched per worker.
        protected_tokens (Optional[Set[str]]): Token strings that should
            never be masked. Defaults to ``{"^", "$", ".", ">>"}`` if None.
        masking_mode (str): Either ``"random"`` or ``"span"``.
        span_mlm_config (Optional[SpanMLMConfig]): Span masking
            configuration. Defaults to
            ``SpanMLMConfig(mlm_probability=0.20, ...)`` if None.
        canonicalize_mapped_rxn_smiles_pct (float): Probability of
            canonicalizing instead of randomizing SMILES during training
            augmentation. Validation always canonicalizes to match inference
            behavior. Defaults to 0.05.
        resume_from_checkpoint (str | None): Path to a ``.pt`` checkpoint
            file to resume training from.
        log_level (str): Loguru level for progress messages. Defaults to
            ``"INFO"`` so training progress is visible in all contexts
            (CLI, notebooks, scripts).

    Returns:
        SupervisedAlbertTrainer: The trainer instance after training completes.
    """
    # --- Logging (ensure progress is visible regardless of entry point) ---
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # --- Tokenizer ---
    tokenizer = CustomTokenizer(smiles_token_to_id_dict)

    # --- Configure everything ---
    if not model_config:
        model_config = ModelConfig()
    if not training_config:
        training_config = TrainingConfig()
    if not mlm_config:
        mlm_config = MLMConfig()
    if not supervised_config:
        supervised_config = SupervisedConfig()

    if protected_tokens is None:
        protected_tokens = {"^", "$", ".", ">>"}

    if span_mlm_config is None:
        span_mlm_config = SpanMLMConfig(
            mlm_probability=0.20,
            span_size_weights={1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1},
        )

    # --- Build or load base model ---
    if pretrained_model_path:
        base_model = AlbertForMaskedLM.from_pretrained(pretrained_model_path)
        logger.info(f"Loaded pretrained model from {pretrained_model_path}")
    else:
        base_model = build_albert_model(model_config)
        logger.info("Built new model from scratch")

    # --- Wrap with attention alignment head ---
    model = AlbertWithAttentionAlignment(
        base_model=base_model,
        supervised_config=supervised_config,
    )

    # --- Datasets ---
    # Training: random SMILES augmentation with configurable canonicalization pct
    train_dataset = SupervisedAtomMappingDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        mlm_config=mlm_config,
        protected_tokens=protected_tokens,
        max_length=max_length,
        use_random_smiles=True,
        canonicalize_mapped_rxn_smiles_pct=canonicalize_mapped_rxn_smiles_pct,
        masking_mode=masking_mode,
        span_mlm_config=span_mlm_config,
    )
    # Validation: always canonicalize to match inference behavior
    val_dataset = SupervisedAtomMappingDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        mlm_config=mlm_config,
        protected_tokens=protected_tokens,
        max_length=max_length,
        use_random_smiles=False,
        use_canonical_smiles=True,
        masking_mode=masking_mode,
        span_mlm_config=span_mlm_config,
    )

    # --- Dataloaders ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        drop_last=True,
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

    # --- Train ---
    trainer = SupervisedAlbertTrainer(
        model=model,
        train_dataloader=train_dataloader,
        training_config=training_config,
        supervised_config=supervised_config,
        tokenizer=tokenizer,
        val_dataloader=val_dataloader,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    trainer.train()

    return trainer
