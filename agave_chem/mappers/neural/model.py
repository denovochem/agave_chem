from typing import Dict

import torch
from pydantic import BaseModel, field_validator
from transformers import AlbertForMaskedLM


class SupervisedConfig(BaseModel):
    """
    Configuration for supervised attention alignment training.

    Args:
        target_layer (int): Which layer's attention to supervise.
        mlm_loss_weight (float): Loss weight for MLM task in multitask learning.
        attention_loss_weight (float): Loss weight for attention alignment task.
        multitask (bool): If False, only attention alignment loss is used.
        freeze_base_model (bool): If True, only train the attention head.
        bottleneck_size (int): BilinearMappingHead bottleneck size.
        use_attention_residual (bool): Whether BilinearMappingHead uses attention
            residual.
    """

    target_layer: int = 10
    mlm_loss_weight: float = 1.0
    attention_loss_weight: float = 1.0
    multitask: bool = True
    freeze_base_model: bool = False
    bottleneck_size: int = 64
    use_attention_residual: bool = True

    @field_validator("target_layer")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @field_validator("bottleneck_size")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v


class BilinearMappingHead(torch.nn.Module):
    """
    Sequence-length-invariant mapping head that learns task-specific Q/K
    projections from transformer hidden states.

    Computes mapping scores as a scaled bilinear product of projected hidden
    states, optionally combined with a learned weighted fusion of multiple
    attention heads as a residual signal.

    Args:
        hidden_size (int): Dimensionality of the transformer hidden states.
        bottleneck_size (int): Dimensionality of the learned Q/K projections.
        num_attention_heads (int): Number of attention heads in the transformer
            (used for multi-head fusion when use_attention_residual is True).
        use_attention_residual (bool): If True, combine bilinear scores with a
            learned weighted average of all attention heads at the target layer.

    Note:
        When use_attention_residual is True, the bilinear contribution is gated
        by a learned scalar initialized to 0.0. This means the model starts
        from the pre-trained multi-head-fused attention behavior and gradually
        learns corrections via the bilinear pathway during supervised training.
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        num_attention_heads: int = 8,
        use_attention_residual: bool = True,
    ):
        super().__init__()
        self.use_attention_residual = use_attention_residual
        self.scale = bottleneck_size**-0.5

        self.query_proj = torch.nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.key_proj = torch.nn.Linear(hidden_size, bottleneck_size, bias=False)

        if use_attention_residual:
            # Learnable weights over all heads (uniform after softmax at init)
            self.head_weights = torch.nn.Parameter(torch.zeros(num_attention_heads))
            # Gate for bilinear scores; starts at 0 so initial behaviour
            # matches the fused pre-trained attention.
            self.bilinear_gate = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_head_attentions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute mapping logits from hidden states and optional multi-head attention.

        Args:
            hidden_states (torch.Tensor): Hidden states from the target
                transformer layer, shape (B, S, hidden_size).
            all_head_attentions (torch.Tensor | None): Attention weights from
                all heads at the target layer, shape (B, H, S, S). Required
                when use_attention_residual is True.

        Returns:
            torch.Tensor: Mapping logits of shape (B, S, S).
        """
        q = self.query_proj(hidden_states)  # (B, S, bottleneck)
        k = self.key_proj(hidden_states)  # (B, S, bottleneck)
        bilinear_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, S, S)

        if self.use_attention_residual and all_head_attentions is not None:
            weights = torch.softmax(self.head_weights, dim=0)  # (H,)
            fused_attn = torch.einsum(
                "bhij,h->bij", all_head_attentions, weights
            )  # (B, S, S)
            # Convert to log-probability space
            log_fused_attn = torch.log(fused_attn.clamp(min=1e-9))
            mapping_logits = log_fused_attn + self.bilinear_gate * bilinear_scores
        else:
            mapping_logits = bilinear_scores

        return mapping_logits


class AlbertWithAttentionAlignment(torch.nn.Module):
    """
    Wrapper around AlbertForMaskedLM that adds supervised attention alignment.

    Extracts hidden states and attention from a specific layer, passes through
    a BilinearMappingHead, and computes both MLM loss and attention alignment
    loss.
    """

    def __init__(
        self,
        base_model: AlbertForMaskedLM,
        supervised_config: SupervisedConfig,
    ):
        super().__init__()
        self.base_model = base_model
        self.supervised_config = supervised_config

        if getattr(self.base_model.config, "_attn_implementation", None) != "eager":
            self.base_model.config._attn_implementation = "eager"

        # Enable attention and hidden state output
        self.base_model.config.output_attentions = True
        self.base_model.config.output_hidden_states = True

        # Mapping head
        self.bilinear_head = BilinearMappingHead(
            hidden_size=self.base_model.config.hidden_size,
            bottleneck_size=supervised_config.bottleneck_size,
            num_attention_heads=self.base_model.config.num_attention_heads,
            use_attention_residual=supervised_config.use_attention_residual,
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
        # Forward through base model with attention and hidden state output
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
        )

        result = {}

        # MLM loss
        mlm_loss = outputs.loss if labels is not None else None
        if mlm_loss is not None:
            result["mlm_loss"] = mlm_loss

        # Compute mapping logits via BilinearMappingHead
        target_layer = self.supervised_config.target_layer
        attentions = outputs.attentions

        # Use output of target layer (richer than input, includes attn+FFN)
        hidden_states = outputs.hidden_states[target_layer + 1]  # (B, S, H)
        layer_attention = attentions[target_layer]  # (B, num_heads, S, S)
        attention_logits = self.bilinear_head(
            hidden_states, layer_attention
        )  # (B, S, S)

        result["attention_logits"] = attention_logits

        # Compute attention alignment loss
        attention_loss = None
        if attention_target is not None:
            attention_loss = self._compute_attention_loss(
                attention_logits, attention_target, attention_loss_mask, attention_mask
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
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute token-level KL divergence loss for attention alignment.

        KL(p || q) = H(p, q) - H(p), where p is the target distribution and q
        is the predicted distribution. Compared to soft cross-entropy H(p, q),
        this subtracts the (constant w.r.t. parameters) target entropy H(p) so
        that the loss floor is 0 for every position, including symmetric atoms
        whose targets are uniform over k positions (where H(p) = log(k)).
        Gradients w.r.t. model parameters are identical to soft cross-entropy.

        Padding columns are excluded from the softmax normalization via the
        attention_mask so that the loss is invariant to MAX_LENGTH / padding.

        Args:
            logits: (B, S, S) predicted attention scores
            targets: (B, S, S) target attention distributions (one-hot or soft)
            mask: (B, S) binary mask, 1 where loss should be computed
            attention_mask: (B, S) token mask (1 = real, 0 = padding). When
                provided, padding columns are set to -inf before log_softmax.

        Returns:
            Scalar loss value
        """
        if attention_mask is not None:
            # Broadcast (B, S) → (B, 1, S) to mask padding columns in every row
            col_pad_mask = (attention_mask == 0).unsqueeze(1)  # (B, 1, S)
            logits = logits.masked_fill(col_pad_mask, float("-inf"))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, S, S)

        # Diagnostic: detect inf/nan before loss computation
        if torch.isinf(logits).any() or torch.isnan(logits).any():
            import logging

            logging.getLogger(__name__).warning(
                f"_compute_attention_loss: logits contain inf/nan. "
                f"inf_count={torch.isinf(logits).sum().item()}, "
                f"nan_count={torch.isnan(logits).sum().item()}, "
                f"logits_min={logits[~torch.isinf(logits) & ~torch.isnan(logits)].min().item() if (~torch.isinf(logits) & ~torch.isnan(logits)).any() else 'all_inf_nan'}, "
                f"logits_max={logits[~torch.isinf(logits) & ~torch.isnan(logits)].max().item() if (~torch.isinf(logits) & ~torch.isnan(logits)).any() else 'all_inf_nan'}"
            )
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            import logging

            logging.getLogger(__name__).warning(
                f"_compute_attention_loss: log_probs contain inf/nan. "
                f"inf_count={torch.isinf(log_probs).sum().item()}, "
                f"nan_count={torch.isnan(log_probs).sum().item()}"
            )
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            import logging

            logging.getLogger(__name__).warning(
                f"_compute_attention_loss: targets contain inf/nan. "
                f"inf_count={torch.isinf(targets).sum().item()}, "
                f"nan_count={torch.isnan(targets).sum().item()}, "
                f"targets_min={targets[~torch.isinf(targets) & ~torch.isnan(targets)].min().item() if (~torch.isinf(targets) & ~torch.isnan(targets)).any() else 'all_inf_nan'}, "
                f"targets_max={targets[~torch.isinf(targets) & ~torch.isnan(targets)].max().item() if (~torch.isinf(targets) & ~torch.isnan(targets)).any() else 'all_inf_nan'}"
            )

        # Guard against 0 * (-inf) = NaN at masked padding columns
        cross_entropy = -(
            torch.where(targets > 0, targets * log_probs, torch.zeros_like(targets))
        ).sum(dim=-1)  # (B, S)
        target_log = torch.where(
            targets > 0, torch.log(targets), torch.zeros_like(targets)
        )
        target_entropy = -(targets * target_log).sum(dim=-1)  # (B, S)
        loss_per_position = (cross_entropy - target_entropy).view(-1)  # (B*S,)

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

    @torch.no_grad()
    def predict_attention_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

        target_layer = self.supervised_config.target_layer
        attentions = outputs.attentions

        hidden_states = outputs.hidden_states[target_layer + 1]
        layer_attention = attentions[target_layer]
        attention_logits = self.bilinear_head(hidden_states, layer_attention)

        attention_probs = torch.softmax(attention_logits, dim=-1)  # (B,S,S)
        return attention_probs
