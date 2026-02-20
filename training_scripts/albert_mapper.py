import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AlbertConfig,
    AlbertForMaskedLM,
    AlbertTokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from utils.chem_utils import (
    canonicalize_reaction_smiles,
    randomize_reaction_smiles,
)
from utils.constants import smiles_token_to_id_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    num_attention_heads: int = 12
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
    save_steps: int = 1000
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


class CustomTokenizer(PreTrainedTokenizer):
    """
    A tokenizer built from a user-supplied token -> integer mapping.
    Supports both character-level and multi-character tokens (e.g. Br, Cl, >>).

    Tokenization strategy:
        - If the text contains spaces, it is assumed to be a pre-tokenized
          string (e.g. output of preprocess_sentence_template) and will be
          split on whitespace. Spaces themselves are not tokens.
        - If the text contains no spaces, it is tokenized character by
          character, with unknown characters replaced by unk_token.

    Args:
        token_to_id: A dictionary mapping tokens (str) to integer IDs.
        id_to_token: A dictionary mapping integer IDs to tokens (str).
                     If not provided, it will be derived from token_to_id.
        unk_token:   The unknown token string. Must exist in token_to_id.
        pad_token:   The padding token string. Must exist in token_to_id.
        mask_token:  The mask token string. Must exist in token_to_id.
        cls_token:   The CLS token string. Must exist in token_to_id.
        sep_token:   The SEP token string. Must exist in token_to_id.
    """

    def __init__(
        self,
        token_to_id: Dict[str, int],
        id_to_token: Dict[int, str] | None = None,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        use_random_smiles=True,
        use_canonical_smiles=False,
        **kwargs,
    ):
        special_tokens = {
            "unk_token": unk_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "cls_token": cls_token,
            "sep_token": sep_token,
        }
        for name, token in special_tokens.items():
            if token not in token_to_id:
                raise ValueError(
                    f"Special token '{token}' ({name}) not found in token_to_id. "
                    f"Please include it in your vocabulary."
                )

        self._token_to_id = token_to_id
        self._id_to_token = id_to_token or {v: k for k, v in token_to_id.items()}

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

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sep_token=sep_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def get_vocab(self) -> Dict[str, int]:
        return self._token_to_id.copy()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string.

        If the string contains spaces (i.e. it has been preprocessed by
        preprocess_sentence_template), split on whitespace and treat each
        element as a token. Spaces are discarded and are never tokens.

        If the string contains no spaces, fall back to character-level
        tokenization for raw/unprocessed input.

        Unknown tokens are replaced with unk_token.
        """

        if self._use_random_smiles:
            text = randomize_reaction_smiles(text)

        if self._use_canonical_smiles:
            text = canonicalize_reaction_smiles(text)

        text = self.preprocess_sentence_reaction_smiles(text)

        if " " in text:
            # Pre-tokenized path: split on whitespace, discard empty strings
            raw_tokens = text.split()
        else:
            # Character-level fallback for raw strings
            raw_tokens = list(text)

        return [
            token if token in self._token_to_id else self.unk_token
            for token in raw_tokens
        ]

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Rejoin tokens back into a readable string.
        Tokens are joined with a space so the output mirrors the
        preprocessed SMILES format.
        """
        return " ".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: List[int] | None = None
    ) -> List[int]:
        """
        Do not add any special tokens (CLS, SEP) around the sequence.
        Overrides the default BERT-style behaviour of prepending [CLS]
        and appending [SEP].
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: List[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Returns a mask of 0s only, since we add no special tokens.
        """
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * (len(token_ids_0) + len(token_ids_1))

    def preprocess_sentence_template(self, template_smarts: str) -> str:
        REGEXPS_RXN = {
            "2_ring_nums": re.compile(r"(%\d{2})"),
            "rxn_symbol": re.compile(r"(>>)"),
            "brcl": re.compile(r"(Li|Na|Mg|Si|Ca|Cu|Ag|Pb|Br|Cl|>>)"),
        }

        REGEXP_ORDER_RXN = ["2_ring_nums", "brcl", "rxn_symbol"]

        def split_by(template_smarts, regexps):
            if not regexps:
                return list(template_smarts)
            regexp = REGEXPS_RXN[regexps[0]]
            splitted = regexp.split(template_smarts)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(template_smarts, REGEXP_ORDER_RXN)
        string = ""
        for ele in tokens:
            string += ele + " "
        string = "^ " + string + "$"
        return string

    def preprocess_sentence_reaction_smiles(self, reaction_smiles: str) -> str:
        REGEXPS_RXN = {
            "brackets": re.compile(r"(\[[^\]]*\])"),
            "2_ring_nums": re.compile(r"(%\d{2})"),
            "rxn_symbol": re.compile(r"(>>)"),
            "brcl": re.compile(r"(Li|Na|Mg|Si|Ca|Cu|Ag|Pb|Br|Cl|>>)"),
        }

        REGEXP_ORDER_RXN = ["brackets", "2_ring_nums", "brcl", "rxn_symbol"]

        def split_by(reaction_smiles, regexps):
            if not regexps:
                return list(reaction_smiles)
            regexp = REGEXPS_RXN[regexps[0]]
            splitted = regexp.split(reaction_smiles)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(reaction_smiles, REGEXP_ORDER_RXN)
        string = ""
        for ele in tokens:
            string += ele + " "
        string = "^ " + string + "$"
        return string

    def decode_id_list(self, ids: List[int]) -> str:
        """Convenience method to decode a list of IDs back to a string."""
        tokens = [self._convert_id_to_token(i) for i in ids]
        return self.convert_tokens_to_string(tokens)


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
    tokenizer: AlbertTokenizer,
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
                logger.warning(
                    f"Protected token '{token}' was not found directly in vocab. "
                    f"Protecting its sub-token IDs instead: {sub_ids}"
                )
            else:
                logger.warning(
                    f"Protected token '{token}' could not be resolved. Skipping."
                )

    return resolved_ids


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
        protected_tokens: Set[str] | None = None,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.mlm_config = mlm_config
        self.max_length = max_length

        # Build the set of protected token IDs (see section 3)
        special_token_ids = set(tokenizer.all_special_ids)
        protected_token_ids = resolve_protected_token_ids(tokenizer, protected_tokens)
        self.protected_token_ids = special_token_ids | protected_token_ids

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

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

    def decode_sample(self, idx: int, print_output: bool = True) -> Dict[str, str]:
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
    logger.info(f"Model built with {total_params:,} trainable parameters.")
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
                avg_loss = total_loss / (step + 1)
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch} | Step {step + 1}/{num_batches} "
                    f"| Loss: {avg_loss:.4f} | LR: {lr:.2e}"
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
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Epochs: {self.training_config.num_epochs}")
        logger.info(f"Batch size: {self.training_config.batch_size}")

        for epoch in range(1, self.training_config.num_epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            logger.info(
                f"Epoch {epoch} complete | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
                f"Time: {time.time() - start_time:.2f} seconds"
            )

            self.model.save_pretrained(
                f"{self.training_config.output_dir}/checkpoint-epoch-{epoch}"
            )
            logger.info(f"Model saved to {self.training_config.output_dir}")


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
        num_workers=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=1,
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


# if __name__ == "__main__":
#     main()
