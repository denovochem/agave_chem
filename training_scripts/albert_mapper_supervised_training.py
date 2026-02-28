import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AlbertForMaskedLM,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from training_scripts.albert_mapper import (
    CustomTokenizer,
    MLMConfig,
    ModelConfig,
    TrainingConfig,
    apply_mlm_masking,
    build_albert_model,
    resolve_protected_token_ids,
)

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent

sys.path.append(str(PARENT_DIR))

from agave_chem.mappers.neural.constants import smiles_token_to_id_dict  # noqa: E402
from agave_chem.utils.chem_utils import (  # noqa: E402
    canonicalize_reaction_smiles,
    randomize_reaction_smiles,
)

# ============================================================
# Supervised Attention Alignment Configuration
# ============================================================


@dataclass
class SupervisedConfig:
    """Configuration for supervised attention alignment training."""

    target_layer: int = 5  # Which layer's attention to supervise
    target_head: int = 3  # Which head's attention to supervise

    # Loss weighting for multitask learning
    mlm_loss_weight: float = 1.0
    attention_loss_weight: float = 1.0

    # Training mode
    multitask: bool = True  # If False, only attention alignment loss
    freeze_base_model: bool = False  # If True, only train the attention head

    # Dense layer config
    use_residual: bool = True  # Initialize with identity for residual learning


# ============================================================
# Supervised Dataset
# ============================================================


class SupervisedAtomMappingDataset(Dataset):
    """
    Dataset for supervised attention alignment training.

    Each sample includes:
        - Standard MLM inputs (input_ids, attention_mask, etc.)
        - attention_target: N x N matrix where row i has a 1 at column j
          if token i should attend to token j
        - attention_mask_target: 1D mask where 0 means ignore that row's loss
    """

    def __init__(
        self,
        texts: List[str],
        attention_targets: List[np.ndarray],  # List of N x N matrices
        tokenizer: PreTrainedTokenizer,
        mlm_config: MLMConfig,
        max_length: int = 256,
        use_random_smiles: bool = True,
        use_canonical_smiles: bool = False,
        protected_tokens: Set[str] | None = None,
    ):
        assert len(texts) == len(attention_targets), (
            "texts and attention_targets must have same length"
        )

        self.texts = texts
        self.attention_targets = attention_targets
        self.tokenizer = tokenizer
        self.mlm_config = mlm_config
        self.max_length = max_length

        if use_canonical_smiles and use_random_smiles:
            raise ValueError(
                "use_canonical_smiles and use_random_smiles cannot both be True"
            )
        if not use_canonical_smiles and not use_random_smiles:
            raise ValueError(
                "use_canonical_smiles and use_random_smiles cannot both be False"
            )

        self._use_canonical_smiles = use_canonical_smiles
        self._use_random_smiles = use_random_smiles

        special_token_ids = set(tokenizer.all_special_ids)
        protected_token_ids = resolve_protected_token_ids(tokenizer, protected_tokens)
        self.protected_token_ids = special_token_ids | protected_token_ids

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        attention_target = self.attention_targets[idx]

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

        masked_input_ids, labels = apply_mlm_masking(
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            mlm_config=self.mlm_config,
            special_token_ids=self.protected_token_ids,
        )

        # Pad/truncate attention target to max_length x max_length
        padded_attention_target = np.zeros(
            (self.max_length, self.max_length), dtype=np.float32
        )

        orig_len = min(attention_target.shape[0], self.max_length)
        orig_width = min(attention_target.shape[1], self.max_length)
        padded_attention_target[:orig_len, :orig_width] = attention_target[
            :orig_len, :orig_width
        ]

        # Create mask for which rows have valid targets (row sum > 0)
        attention_loss_mask = (padded_attention_target.sum(axis=1) > 0).astype(
            np.float32
        )

        return {
            "input_ids": torch.tensor(masked_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_target": torch.tensor(
                padded_attention_target, dtype=torch.float32
            ),
            "attention_loss_mask": torch.tensor(
                attention_loss_mask, dtype=torch.float32
            ),
        }


# ============================================================
# Model with Supervised Attention Head
# ============================================================


class AttentionAlignmentHead(torch.nn.Module):
    """
    A learnable layer that refines attention scores for atom mapping.

    Initialized as identity (with optional residual connection) so that
    initial behavior matches the pre-trained attention patterns.
    """

    def __init__(self, seq_length: int, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual

        # Linear transformation on attention scores
        # Input: (batch, seq_len, seq_len) -> Output: (batch, seq_len, seq_len)
        self.dense = torch.nn.Linear(seq_length, seq_length, bias=True)

        # Initialize as identity for residual learning
        torch.nn.init.eye_(self.dense.weight)
        torch.nn.init.zeros_(self.dense.bias)

    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_scores: (batch, seq_len, seq_len) attention weights

        Returns:
            Refined attention scores of same shape
        """
        # attention_scores: (B, S, S)
        output = self.dense(attention_scores)  # (B, S, S)

        if self.use_residual:
            output = output + attention_scores

        return output


class AlbertWithAttentionAlignment(torch.nn.Module):
    """
    Wrapper around AlbertForMaskedLM that adds supervised attention alignment.

    Extracts attention from a specific layer/head, passes through a learnable
    dense layer, and computes both MLM loss and attention alignment loss.
    """

    def __init__(
        self,
        base_model: AlbertForMaskedLM,
        supervised_config: SupervisedConfig,
        max_length: int = 256,
    ):
        super().__init__()
        self.base_model = base_model
        self.supervised_config = supervised_config

        # Enable attention output
        self.base_model.config.output_attentions = True

        # Attention alignment head
        self.attention_head = AttentionAlignmentHead(
            seq_length=max_length,
            use_residual=supervised_config.use_residual,
        )

        # Optionally freeze base model
        if supervised_config.freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_target: torch.Tensor | None = None,
        attention_loss_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional MLM and attention alignment losses.

        Returns:
            Dict with keys:
                - loss: Combined loss (or just one if single-task)
                - mlm_loss: MLM loss (if labels provided)
                - attention_loss: Attention alignment loss (if targets provided)
                - attention_logits: Predicted attention (B, S, S)
        """
        # Forward through base model with attention output
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_attentions=True,
        )

        result = {}

        # MLM loss
        mlm_loss = outputs.loss if labels is not None else None
        if mlm_loss is not None:
            result["mlm_loss"] = mlm_loss

        # Extract attention from target layer/head
        # attentions is tuple of (batch, num_heads, seq_len, seq_len) per layer
        attentions = outputs.attentions
        target_layer = self.supervised_config.target_layer
        target_head = self.supervised_config.target_head

        # Get attention from specified layer and head
        layer_attention = attentions[target_layer]  # (B, H, S, S)
        head_attention = layer_attention[:, target_head, :, :]  # (B, S, S)

        # Pass through learnable head
        attention_logits = self.attention_head(head_attention)  # (B, S, S)
        result["attention_logits"] = attention_logits

        # Compute attention alignment loss
        attention_loss = None
        if attention_target is not None:
            attention_loss = self._compute_attention_loss(
                attention_logits, attention_target, attention_loss_mask
            )
            result["attention_loss"] = attention_loss

        # Combine losses
        total_loss = torch.tensor(0.0, device=input_ids.device)

        if self.supervised_config.multitask and mlm_loss is not None:
            total_loss = total_loss + self.supervised_config.mlm_loss_weight * mlm_loss

        if attention_loss is not None:
            total_loss = (
                total_loss
                + self.supervised_config.attention_loss_weight * attention_loss
            )

        result["loss"] = total_loss

        return result

    def _compute_attention_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for attention alignment.

        Args:
            logits: (B, S, S) predicted attention scores
            targets: (B, S, S) one-hot target attention
            mask: (B, S) binary mask, 1 where loss should be computed

        Returns:
            Scalar loss value
        """
        batch_size, seq_len, _ = logits.shape

        # Convert one-hot targets to class indices
        # targets: (B, S, S) -> target_indices: (B, S)
        target_indices = targets.argmax(dim=-1)  # (B, S)

        # Reshape for cross entropy: (B*S, S) and (B*S,)
        logits_flat = logits.view(-1, seq_len)  # (B*S, S)
        targets_flat = target_indices.view(-1)  # (B*S,)

        # Compute per-position cross entropy
        loss_per_position = torch.nn.functional.cross_entropy(
            logits_flat, targets_flat, reduction="none"
        )  # (B*S,)

        # Apply mask
        if mask is not None:
            mask_flat = mask.view(-1)  # (B*S,)
            loss_per_position = loss_per_position * mask_flat

            # Average over valid positions only
            num_valid = mask_flat.sum()
            if num_valid > 0:
                loss = loss_per_position.sum() / num_valid
            else:
                loss = loss_per_position.sum() * 0  # Zero loss if no valid positions
        else:
            loss = loss_per_position.mean()

        return loss


# ============================================================
# Supervised Trainer
# ============================================================


class SupervisedAlbertTrainer:
    """
    Trainer for supervised attention alignment with optional multitask MLM.
    """

    def __init__(
        self,
        model: AlbertWithAttentionAlignment,
        train_dataloader: DataLoader,
        training_config: TrainingConfig,
        supervised_config: SupervisedConfig,
        val_dataloader: DataLoader | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.training_config = training_config
        self.supervised_config = supervised_config
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
        )

        total_steps = len(self.train_dataloader) * self.training_config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=total_steps,
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch and return average losses."""
        self.model.train()
        total_loss = 0.0
        total_mlm_loss = 0.0
        total_attention_loss = 0.0
        num_batches = len(self.train_dataloader)

        for step, batch in enumerate(self.train_dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                labels=batch["labels"] if self.supervised_config.multitask else None,
                attention_target=batch["attention_target"],
                attention_loss_mask=batch["attention_loss_mask"],
            )

            loss = outputs["loss"]
            loss.backward()
            total_loss += loss.item()

            if "mlm_loss" in outputs:
                total_mlm_loss += outputs["mlm_loss"].item()
            if "attention_loss" in outputs:
                total_attention_loss += outputs["attention_loss"].item()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.training_config.max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if (step + 1) % self.training_config.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                avg_mlm = total_mlm_loss / (step + 1)
                avg_attn = total_attention_loss / (step + 1)
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch} | Step {step + 1}/{num_batches} "
                    f"| Loss: {avg_loss:.4f} | MLM: {avg_mlm:.4f} "
                    f"| Attn: {avg_attn:.4f} | LR: {lr:.2e}"
                )

        return {
            "loss": total_loss / num_batches,
            "mlm_loss": total_mlm_loss / num_batches,
            "attention_loss": total_attention_loss / num_batches,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation and return average losses."""
        if self.val_dataloader is None:
            return {"loss": 0.0, "mlm_loss": 0.0, "attention_loss": 0.0}

        self.model.eval()
        total_loss = 0.0
        total_mlm_loss = 0.0
        total_attention_loss = 0.0

        for batch in self.val_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                labels=batch["labels"] if self.supervised_config.multitask else None,
                attention_target=batch["attention_target"],
                attention_loss_mask=batch["attention_loss_mask"],
            )

            total_loss += outputs["loss"].item()
            if "mlm_loss" in outputs:
                total_mlm_loss += outputs["mlm_loss"].item()
            if "attention_loss" in outputs:
                total_attention_loss += outputs["attention_loss"].item()

        num_batches = len(self.val_dataloader)
        return {
            "loss": total_loss / num_batches,
            "mlm_loss": total_mlm_loss / num_batches,
            "attention_loss": total_attention_loss / num_batches,
        }

    def train(self) -> None:
        """Run the full training loop."""
        mode = "multitask" if self.supervised_config.multitask else "supervised-only"
        print(f"Starting {mode} training on device: {self.device}")
        print(
            f"Target layer: {self.supervised_config.target_layer}, "
            f"Target head: {self.supervised_config.target_head}"
        )
        print(f"Epochs: {self.training_config.num_epochs}")

        for epoch in range(1, self.training_config.num_epochs + 1):
            start_time = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            print(
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
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                f"{save_path}.pt",
            )

            # Also save the base model for inference
            self.model.base_model.save_pretrained(save_path)
            print(f"Checkpoint saved to {save_path}")

    @torch.no_grad()
    def predict_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get predicted attention alignment for inference.

        Returns:
            attention_probs: (B, S, S) softmax-normalized attention
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


# ============================================================
# Example Usage for Supervised Training
# ============================================================


def main_supervised(
    train_texts: List[str],
    train_attention_targets: List[np.ndarray],
    val_texts: List[str],
    val_attention_targets: List[np.ndarray],
    pretrained_model_path: str | None = None,
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    mlm_config: Optional[MLMConfig] = None,
    supervised_config: Optional[SupervisedConfig] = None,
):
    """
    Run supervised attention alignment training.

    Args:
        train_texts: List of reaction SMILES for training
        train_attention_targets: List of N x N numpy arrays with attention targets
        val_texts: List of reaction SMILES for validation
        val_attention_targets: List of N x N numpy arrays with attention targets
        pretrained_model_path: Path to pretrained MLM model (optional, for fine-tuning)
        model_config: Model architecture config
        training_config: Training hyperparameters
        mlm_config: MLM masking config
        supervised_config: Supervised attention alignment config
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
    if not supervised_config:
        supervised_config = SupervisedConfig()

    # --- Build or load base model ---
    if pretrained_model_path:
        base_model = AlbertForMaskedLM.from_pretrained(pretrained_model_path)
        print(f"Loaded pretrained model from {pretrained_model_path}")
    else:
        base_model = build_albert_model(model_config)
        print("Built new model from scratch")

    # --- Wrap with attention alignment head ---
    model = AlbertWithAttentionAlignment(
        base_model=base_model,
        supervised_config=supervised_config,
        max_length=256,
    )

    # --- Datasets ---
    train_dataset = SupervisedAtomMappingDataset(
        texts=train_texts,
        attention_targets=train_attention_targets,
        tokenizer=tokenizer,
        mlm_config=mlm_config,
        protected_tokens={"^", "$", ".", ">>"},
        max_length=256,
    )
    val_dataset = SupervisedAtomMappingDataset(
        texts=val_texts,
        attention_targets=val_attention_targets,
        tokenizer=tokenizer,
        mlm_config=mlm_config,
        protected_tokens={"^", "$", ".", ">>"},
        max_length=256,
    )

    # --- Dataloaders ---
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

    # --- Train ---
    trainer = SupervisedAlbertTrainer(
        model=model,
        train_dataloader=train_dataloader,
        training_config=training_config,
        supervised_config=supervised_config,
        val_dataloader=val_dataloader,
    )
    trainer.train()

    return trainer
