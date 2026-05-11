from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import torch
from rdkit import Chem
from transformers import AlbertForMaskedLM

from agave_chem.mappers.neural.constants import (
    smiles_token_to_id_dict,
    token_atom_identity_dict,
)
from agave_chem.mappers.neural.tokenizer import CustomTokenizer
from agave_chem.mappers.reaction_mapper import ReactionMapper, ReactionMapperResult
from agave_chem.utils.logging_config import logger


class StringInfoDict(TypedDict):
    reactants_dict: Dict[int, str]
    products_dict: Dict[int, str]
    reactants_start_index: int
    reactants_end_index: int
    products_start_index: int
    products_end_index: int
    atom_tokens_dict: Dict[int, List[int]]
    non_atom_tokens: List[int]


@dataclass
class SupervisedConfig:
    """Configuration for supervised attention alignment training."""

    target_layer: int = 11  # Which layer's attention to supervise
    target_head: int = 7  # Which head's attention to supervise

    # Loss weighting for multitask learning
    mlm_loss_weight: float = 1.0
    attention_loss_weight: float = 1.0

    # Training mode
    multitask: bool = True  # If False, only attention alignment loss
    freeze_base_model: bool = False  # If True, only train the attention head

    # Dense layer config
    use_residual: bool = True  # Initialize with identity for residual learning

    # Head type: "attention" for AttentionAlignmentHead, "bilinear" for BilinearMappingHead
    head_type: str = "bilinear"

    # BilinearMappingHead config (only used when head_type == "bilinear")
    bottleneck_size: int = 64
    use_attention_residual: bool = True


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

    Extracts attention from a specific layer/head, passes through a learnable
    dense layer, and computes both MLM loss and attention alignment loss.
    """

    def __init__(
        self,
        base_model: AlbertForMaskedLM,
        supervised_config: SupervisedConfig,
        max_length: int = 512,
    ):
        super().__init__()
        self.base_model = base_model
        self.supervised_config = supervised_config

        if getattr(self.base_model.config, "_attn_implementation", None) != "eager":
            self.base_model.config._attn_implementation = "eager"

        # Enable attention output
        self.base_model.config.output_attentions = True

        # Mapping head
        if supervised_config.head_type == "bilinear":
            self.base_model.config.output_hidden_states = True
            self.bilinear_head = BilinearMappingHead(
                hidden_size=self.base_model.config.hidden_size,
                bottleneck_size=supervised_config.bottleneck_size,
                num_attention_heads=self.base_model.config.num_attention_heads,
                use_attention_residual=supervised_config.use_attention_residual,
            )
        else:
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
        use_bilinear = self.supervised_config.head_type == "bilinear"
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_attentions=True,
            output_hidden_states=use_bilinear,
        )

        result = {}

        # MLM loss
        mlm_loss = outputs.loss if labels is not None else None
        if mlm_loss is not None:
            result["mlm_loss"] = mlm_loss

        # Compute mapping logits via the appropriate head
        target_layer = self.supervised_config.target_layer
        attentions = outputs.attentions

        if use_bilinear:
            # Use output of target layer (richer than input, includes attn+FFN)
            hidden_states = outputs.hidden_states[target_layer + 1]  # (B, S, H)
            layer_attention = attentions[target_layer]  # (B, num_heads, S, S)
            attention_logits = self.bilinear_head(
                hidden_states, layer_attention
            )  # (B, S, S)
        else:
            target_head = self.supervised_config.target_head
            layer_attention = attentions[target_layer]  # (B, H, S, S)
            head_attention = layer_attention[:, target_head, :, :]  # (B, S, S)
            attention_logits = self.attention_head(head_attention)  # (B, S, S)

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
        use_bilinear = self.supervised_config.head_type == "bilinear"
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=use_bilinear,
            return_dict=True,
        )

        target_layer = self.supervised_config.target_layer
        attentions = outputs.attentions

        if use_bilinear:
            hidden_states = outputs.hidden_states[target_layer + 1]
            layer_attention = attentions[target_layer]
            attention_logits = self.bilinear_head(hidden_states, layer_attention)
        else:
            layer_attention = attentions[target_layer]  # (B,H,S,S)
            head_attention = layer_attention[
                :, self.supervised_config.target_head, :, :
            ]  # (B,S,S)
            attention_logits = self.attention_head(head_attention)  # (B,S,S)

        attention_probs = torch.softmax(attention_logits, dim=-1)  # (B,S,S)
        return attention_probs


def load_neural_albert_model(
    checkpoint_dir: str,
    device: torch.device,
    use_supervised: bool,
    max_length: int = 512,
    supervised_config: SupervisedConfig | None = None,
) -> torch.nn.Module:
    checkpoint_dir = str(checkpoint_dir)
    base_model = AlbertForMaskedLM.from_pretrained(
        checkpoint_dir,
        attn_implementation="eager",
    ).to(device)

    if not use_supervised:
        return base_model

    if supervised_config is None:
        supervised_config = SupervisedConfig()

    wrapper = AlbertWithAttentionAlignment(
        base_model=base_model,
        supervised_config=supervised_config,
        max_length=max_length,
    ).to(device)

    pt_path = str(Path(checkpoint_dir) / "supervised_albert_model.pt")
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    wrapper.load_state_dict(ckpt["model_state_dict"], strict=True)
    wrapper.eval()
    return wrapper


class NeuralReactionMapper(ReactionMapper):
    """
    Neural network-based reaction atom-mapping
    """

    def __init__(
        self,
        mapper_name: str,
        mapper_weight: float = 3,
        checkpoint_path: Optional[str] = None,
        use_supervised: bool = True,
        supervised_config: SupervisedConfig | None = None,
        sequence_max_length: int = 512,
    ):
        """
        Initialize the NeuralReactionMapper instance.

        Args:
            mapper_name (str): The name of the mapper.
            mapper_weight (float): The weight of the mapper.
            checkpoint_path (Optional[str]): The path to the checkpoint file.
        """

        super().__init__("neural", mapper_name, mapper_weight)

        if not checkpoint_path:
            checkpoint_path = str(
                files("agave_chem.datafiles.models").joinpath("supervised_albert_model")
            )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._sequence_max_length = sequence_max_length
        self._use_supervised = use_supervised
        self._supervised_config = supervised_config or SupervisedConfig()

        self._model = load_neural_albert_model(
            checkpoint_dir=checkpoint_path,
            device=self._device,
            use_supervised=use_supervised,
            max_length=sequence_max_length,
            supervised_config=self._supervised_config,
        )

        self._tokenizer = CustomTokenizer(smiles_token_to_id_dict)

    def _encode_atom(self, atom: Chem.Atom) -> List[int]:
        """
        Encode an RDKit Atom object into a list of integers.

        The encoding is as follows:
        - z: The atomic number of the atom.
        - chg: The formal charge of the atom.
        - arom: 1 if the atom is aromatic, 0 otherwise.
        - ring: 1 if the atom is in a ring, 0 otherwise.
        - h: The total number of hydrogen atoms bonded to the atom.
        - d: The degree of the atom.

        Args:
            atom (Chem.Atom): The RDKit Atom object to encode.

        Returns:
            List[int]: A list of integers encoding the atom.
        """
        z = atom.GetAtomicNum()
        chg = atom.GetFormalCharge()
        arom = 1 if atom.GetIsAromatic() else 0
        ring = 1 if atom.IsInRing() else 0
        h = atom.GetTotalNumHs()
        d = atom.GetDegree()
        return [z, chg, arom, ring, h, d]

    def get_attention_matrix_for_head(
        self,
        text: str,
        layer: int,
        head: int,
        max_length: int = 512,
        trim_padding: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Returns the attention matrix for a given layer/head for a single input string.

        Args:
            text: input reaction SMILES string (raw is fine; CustomTokenizer preprocesses)
            layer: 0-based layer index
            head: 0-based head index
            max_length: tokenization length (should match training, e.g. 256)
            trim_padding: if True, slices matrix down to non-pad tokens only

        Returns:
            attn: Tensor of shape (seq_len, seq_len) (trimmed if requested)
            tokens: list[str] tokens aligned to attn axes (trimmed if requested)
        """
        self._model.eval()

        enc = self._tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        token_type_ids = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))
        token_type_ids = token_type_ids.to(self._device)

        with torch.no_grad():
            if self._use_supervised:
                # self._model is AlbertWithAttentionAlignment
                attn_probs = self._model.predict_attention_probs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )  # (B,S,S)
                attn = attn_probs[0].detach().cpu()  # (S,S)
            else:
                # self._model is AlbertForMaskedLM
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    return_dict=True,
                )
                attentions = outputs.attentions  # tuple[num_layers] of (B,H,S,S)

                if layer < 0 or layer >= len(attentions):
                    raise ValueError(
                        f"layer must be in [0, {len(attentions) - 1}], got {layer}"
                    )

                num_heads = attentions[layer].shape[1]
                if head < 0 or head >= num_heads:
                    raise ValueError(
                        f"head must be in [0, {num_heads - 1}], got {head}"
                    )

                attn = attentions[layer][0, head].detach().cpu()  # (S,S)

        # Tokens for inspection/plotting
        token_ids = enc["input_ids"][0].tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(token_ids)

        if trim_padding:
            real_len = int(enc["attention_mask"][0].sum().item())
            attn = attn[:real_len, :real_len]
            tokens = tokens[:real_len]

        # IMPORTANT: keep downstream behavior identical by returning log-attn
        return torch.log(attn).numpy(), tokens

    def get_reactants_products_dict(
        self,
        tokens: List[str],
    ) -> StringInfoDict:
        """
        Extracts reactants and products from a list of tokens in a reaction SMILES string.

        Args:
            tokens: A list of tokens in a reaction SMILES string.

        Returns:
            A tuple containing:
                reactants_dict: A dictionary where the keys are token indices and the values are the corresponding token strings.
                products_dict: A dictionary where the keys are token indices and the values are the corresponding token strings.
                atom_tokens_dict: A dictionary where the keys are atom identities and the values are lists of token indices.
                non_atom_tokens: A list of token indices that correspond to non-atom tokens.
                reactants_start_index: The index of the first reactant token.
                reactants_end_index: The index of the last reactant token.
                products_start_index: The index of the first product token.
                products_end_index: The index of the last product token.
        """
        reactants_dict: Dict[int, str] = {}
        products_dict: Dict[int, str] = {}
        atom_tokens_dict: Dict[int, List[int]] = {}
        non_atom_tokens: List[int] = []

        found_reaction_symbol = False
        for i, token in enumerate(tokens):
            if token == ">>":
                found_reaction_symbol = True
                non_atom_tokens.append(i)
                continue
            if token_atom_identity_dict.get(token, 0) == 0:
                non_atom_tokens.append(i)
            else:
                if token_atom_identity_dict.get(token, 0) not in atom_tokens_dict:
                    atom_tokens_dict[token_atom_identity_dict.get(token, 0)] = [i]
                else:
                    atom_tokens_dict[token_atom_identity_dict.get(token, 0)].append(i)
            if found_reaction_symbol:
                products_dict[i] = token
            else:
                reactants_dict[i] = token

        string_info_dict: StringInfoDict = {
            "reactants_dict": reactants_dict,
            "products_dict": products_dict,
            "reactants_start_index": 0,
            "reactants_end_index": max(reactants_dict.keys()),
            "products_start_index": min(products_dict.keys()),
            "products_end_index": max(products_dict.keys()),
            "atom_tokens_dict": atom_tokens_dict,
            "non_atom_tokens": non_atom_tokens,
        }

        return string_info_dict

    def mask_attn_matrix(
        self,
        attn: np.ndarray,
        string_info_dict: StringInfoDict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Masks the attention matrix to set the attention probability for certain tokens to 0.

        Args:
            attn: The attention matrix to be masked.
            reactants_start_index: The index of the first reactant token.
            reactants_end_index: The index of the last reactant token.
            products_start_index: The index of the first product token.
            products_end_index: The index of the last product token.
            non_atom_tokens: A list of indices of non-atom tokens.
            atom_tokens_dict: A dictionary mapping atom numbers to a list of token indices.

        Returns:
            The masked attention matrix.
        """
        attn[
            string_info_dict["reactants_start_index"] : string_info_dict[
                "products_start_index"
            ]
            - 1,
            string_info_dict["reactants_start_index"] : string_info_dict[
                "products_start_index"
            ]
            - 1,
        ] = -1e6  # Set attention logits for reactant tokens to other reactant tokens to very small value
        attn[
            string_info_dict["products_start_index"] : string_info_dict[
                "products_end_index"
            ]
            + 1,
            string_info_dict["products_start_index"] : string_info_dict[
                "products_end_index"
            ]
            + 1,
        ] = -1e6  # Set attention logits for product tokens to other product tokens to very small value
        for i in string_info_dict[
            "non_atom_tokens"
        ][
            :-1
        ]:  # Set attention logits for reactant or product tokens to non-atom tokens to very small value
            attn[i] = -1e6
            attn[:, i] = -1e6

        for token_indices in string_info_dict[
            "atom_tokens_dict"
        ].values():  # Set attention logits for reactant and product tokens of different atom numbers to very small value
            idx = np.asarray(token_indices, dtype=np.int64)
            last = attn.shape[0] - 1
            idx = idx[idx != last]  # protect last row/column from mask

            diff_atom_mask = np.ones(attn.shape[1], dtype=bool)
            diff_atom_mask[idx] = False
            diff_atom_mask[last] = False  # protect last row/column from mask

            attn[np.ix_(idx, diff_atom_mask)] = -1e6
            attn[np.ix_(diff_atom_mask, idx)] = -1e6

        row_max = np.max(attn, axis=1, keepdims=True)  # max per row
        exp_logits = np.exp(attn - row_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        probs[
            string_info_dict["reactants_start_index"] : string_info_dict[
                "products_start_index"
            ]
            - 1,
            string_info_dict["reactants_start_index"] : string_info_dict[
                "products_start_index"
            ]
            - 1,
        ] = 0  # Set attention probability for reactant tokens to other reactant tokens to 0
        probs[
            string_info_dict["products_start_index"] : string_info_dict[
                "products_end_index"
            ]
            + 1,
            string_info_dict["products_start_index"] : string_info_dict[
                "products_end_index"
            ]
            + 1,
        ] = 0  # Set attention probability for product tokens to other product tokens to 0
        for i in string_info_dict[
            "non_atom_tokens"
        ][
            :-1
        ]:  # Set attention probability for reactant or product tokens to non-atom tokens to 0
            probs[i] = 0
            probs[:, i] = 0

        for token_indices in string_info_dict[
            "atom_tokens_dict"
        ].values():  # Set attention probability for reactant and product tokens of different atom numbers to 0
            idx = np.asarray(token_indices, dtype=np.int64)

            diff_atom_mask = np.ones(probs.shape[1], dtype=bool)
            diff_atom_mask[idx] = False
            probs[np.ix_(idx, diff_atom_mask)] = 0
            probs[np.ix_(diff_atom_mask, idx)] = 0

        return probs, exp_logits

    def average_attn_scores(
        self,
        out: np.ndarray,
        reactants_start_index: int,
        reactants_end_index: int,
        products_start_index: int,
    ) -> np.ndarray:
        """
        Compute the average attention scores between reactants and products.

        Args:
            out (np.ndarray): The attention matrix.
            reactants_start_index (int): The index of the first reactant token.
            reactants_end_index (int): The index of the last reactant token.
            products_start_index (int): The index of the first product token.

        Returns:
            np.ndarray: The average attention scores between reactants and products.
        """
        reactants_to_products_attn = out[
            products_start_index:,
            reactants_start_index : reactants_end_index + 1,
        ]  # reactants to products attention
        products_to_reactants_attn = out[
            reactants_start_index : reactants_end_index + 1, products_start_index:
        ].T  # products to reactants attention, transposed so indices align
        # avg_attn = reactants_to_products_attn
        return reactants_to_products_attn, products_to_reactants_attn

    def remove_non_atom_rows_and_columns(
        self, attn: np.ndarray, string_info_dict: StringInfoDict
    ) -> np.ndarray:
        """
        Remove non-atom tokens from attention matrix.

        Args:
            attn (np.ndarray): The attention matrix.
            string_info_dict (StringInfoDict): A dictionary containing information about the tokens in the reaction SMILES string.

        Returns:
            np.ndarray: The attention matrix with non-atom tokens removed.
        """
        reactants_non_atom_tokens = [
            ele
            for ele in string_info_dict["non_atom_tokens"]
            if ele <= string_info_dict["reactants_end_index"]
        ]  # Get non-atom tokens in reactants
        products_non_atom_tokens = [
            ele - string_info_dict["products_start_index"]
            for ele in string_info_dict["non_atom_tokens"]
            if ele >= string_info_dict["products_start_index"]
        ]  # Get non-atom tokens in products with offset of products start index

        idx = np.asarray(reactants_non_atom_tokens, dtype=int)
        attn = np.delete(attn, idx, axis=1)

        idx = np.asarray(products_non_atom_tokens, dtype=int)
        attn = np.delete(attn, idx, axis=0)

        return attn

    def get_duplicate_indices(
        self, list_of_lists: List[List[int]]
    ) -> Dict[int, List[int]]:
        """
        Find indices of duplicate values across a list of sublists using globally
        offset indices.

        For each element, returns a mapping to all other elements in the same
        sublist that share the same value. Elements without duplicates are omitted.

        Args:
            list_of_lists (List[List[int]]): A list of sublists, where each sublist
                contains integer values (e.g., canonical atom ranks per molecule).

        Returns:
            Dict[int, List[int]]: A dictionary mapping each globally-offset index
                to a list of other globally-offset indices within the same sublist
                that share the same value. Only indices with at least one duplicate
                are included.
        """
        result = {}
        offset = 0

        for sublist in list_of_lists:
            # Group flattened indices by value within this sublist
            value_to_indices = defaultdict(list)
            for i, val in enumerate(sublist):
                value_to_indices[val].append(offset + i)

            # For each item, map to OTHER items with same value in the same sublist
            for i, val in enumerate(sublist):
                flat_idx = offset + i
                others = [idx for idx in value_to_indices[val] if idx != flat_idx]
                if others:  # only include entries that actually have duplicates
                    result[flat_idx] = others

            offset += len(sublist)

        return result

    def _build_atom_dict(
        self, mols: List[Chem.Mol]
    ) -> Tuple[Dict[int, Chem.Atom], Dict[int, List[Tuple[int, List[int]]]]]:
        """
        Build a global atom dictionary and neighbor dictionary from a list of molecules.

        Iterates over each molecule in order, assigning globally unique atom indices
        that are contiguous across all molecules. Neighbors are stored as
        (global_atom_index, atom_feature_vector) pairs.

        Args:
            mols (List[Chem.Mol]): A list of RDKit molecule objects to process.

        Returns:
            Tuple[Dict[int, Chem.Atom], Dict[int, List[Tuple[int, List[int]]]]]:
                - First dict: Maps global atom index to its RDKit Atom object.
                - Second dict: Maps global atom index to a list of
                  (global_neighbor_index, encoded_neighbor) tuples for each
                  neighboring atom.
        """
        atom_dict: Dict[int, Chem.Atom] = {}
        atom_dict_neighbors: Dict[int, List[Tuple[int, List[int]]]] = {}
        global_atom_num = 0
        for mol in mols:
            mol_atom_dict: Dict[int, Chem.Atom] = {}
            mol_idx_to_atom_num: Dict[int, int] = {}
            for atom in mol.GetAtoms():
                mol_atom_dict[global_atom_num] = atom
                mol_idx_to_atom_num[atom.GetIdx()] = global_atom_num
                global_atom_num += 1
            for atom_num, atom in mol_atom_dict.items():
                atom_dict_neighbors[atom_num] = [
                    (
                        mol_idx_to_atom_num[neighbor.GetIdx()],
                        self._encode_atom(neighbor),
                    )
                    for neighbor in atom.GetNeighbors()
                ]
            atom_dict.update(mol_atom_dict)
        return atom_dict, atom_dict_neighbors

    def _get_symmetric_atom_indices(self, mols: List[Chem.Mol]) -> Dict[int, List[int]]:
        """
        Identify sets of topologically equivalent atoms across a list of molecules.

        All molecules are combined into a single disconnected graph via
        Chem.CombineMols before ranking, so that canonical ranks are assigned
        globally. This means two atoms are considered symmetric if they are
        topologically equivalent either within the same molecule (intra-molecular
        symmetry, e.g., ortho carbons in benzene) or across identical fragment
        molecules (inter-molecular symmetry, e.g., corresponding atoms in two
        identical benzaldehyde reactants).

        Args:
            mols (List[Chem.Mol]): A list of RDKit molecule objects.

        Returns:
            Dict[int, List[int]]: A mapping from each globally-offset atom index
                to a list of other globally-offset atom indices that are
                topologically equivalent. Only atoms with at least one symmetric
                partner are included.
        """
        if not mols:
            return {}
        combined = mols[0]
        for mol in mols[1:]:
            combined = Chem.CombineMols(combined, mol)
        ranks = list(Chem.rdmolfiles.CanonicalRankAtoms(combined, breakTies=False))
        return self.get_duplicate_indices([ranks])

    def _apply_symmetric_attention(
        self,
        attn: np.ndarray,
        symmetric_indices: Dict[int, List[int]],
        axis: int,
    ) -> np.ndarray:
        """
        Sum attention scores for topologically equivalent (symmetric) atoms.

        For each group of symmetric atoms, replaces each member's attention slice
        (row or column) with the summed values across the group. This prevents
        symmetric atoms from receiving artificially low attention scores caused by
        probability mass being split equally among equivalent positions.

        Sums are computed from the input array before any modifications are applied,
        ensuring groups do not double-count one another.

        Args:
            attn (np.ndarray): Attention matrix of shape
                (n_product_atoms, n_reactant_atoms).
            symmetric_indices (Dict[int, List[int]]): Output of
                _get_symmetric_atom_indices — maps each atom index to its
                symmetric partners.
            axis (int): Axis along which to aggregate. Use 1 for reactant atoms
                (columns) and 0 for product atoms (rows).

        Returns:
            np.ndarray: A copy of attn with symmetric atom slices replaced by
                their group sum. The original array is not modified.
        """
        identical_groups: List[Tuple[int, ...]] = list(
            {tuple(sorted([k] + v)) for k, v in symmetric_indices.items()}
        )

        result = attn.copy()
        new_val_mapping: Dict[int, np.ndarray] = {}
        for group in identical_groups:
            idx = list(group)
            if axis == 1:
                summed = np.sum(attn[:, idx], axis=1)
                for i in idx:
                    new_val_mapping[i] = summed
            else:
                summed = np.sum(attn[idx, :], axis=0)
                for i in idx:
                    new_val_mapping[i] = summed

        for i, val in new_val_mapping.items():
            if axis == 1:
                result[:, i] = val
            else:
                result[i, :] = val

        return result

    def assign_atom_maps(
        self,
        rxn_smiles: str,
        attn: np.ndarray,
        one_to_one_correspondence: bool = False,
        adjacent_atom_multiplier: float = 30,
        identical_adjacent_atom_multiplier: float = 10,
        reactants_atom_idx_to_orig_mapping: Optional[Dict[int, int]] = None,
        products_atom_idx_to_orig_mapping: Optional[Dict[int, int]] = None,
    ) -> Tuple[str, float]:
        """
        Assign atom-to-atom map numbers to a reaction SMILES using a pre-computed
        attention matrix.

        Handles symmetric atoms in both reactants and products by summing their
        attention contributions, preventing artificially low confidence scores
        caused by equivalent atoms splitting probability mass.

        Args:
            rxn_smiles (str): Unmapped reaction SMILES string of the form
                "reactants>>products".
            attn (np.ndarray): Attention matrix of shape
                (n_product_atoms, n_reactant_atoms).
            one_to_one_correspondence (bool): If True, enforces a one-to-one
                assignment using greedy selection of the global attention maximum.
                If False, assigns each product atom independently to its
                highest-attention reactant atom.
            adjacent_atom_multiplier (float): Multiplier applied to attention
                scores of atoms neighboring an already-mapped pair.
            identical_adjacent_atom_multiplier (float): Additional multiplier
                applied when a neighboring pair shares the same atom encoding.
            reactants_atom_idx_to_orig_mapping (Optional[Dict[int, int]]): Maps
                global reactant atom indices to existing atom map numbers, used
                to anchor partially pre-mapped reactions.
            products_atom_idx_to_orig_mapping (Optional[Dict[int, int]]): Maps
                global product atom indices to existing atom map numbers, used
                to anchor partially pre-mapped reactions.

        Returns:
            Tuple[str, float]:
                - Mapped reaction SMILES string with atom map numbers assigned.
                - Confidence score computed as the product of per-atom assignment
                  probabilities.
        """
        if not reactants_atom_idx_to_orig_mapping:
            reactants_atom_idx_to_orig_mapping = {}
        if not products_atom_idx_to_orig_mapping:
            products_atom_idx_to_orig_mapping = {}

        reactants_str, products_str = self._split_reaction_components(rxn_smiles)
        reactants_mols = [
            Chem.MolFromSmiles(reactant) for reactant in reactants_str.split(".")
        ]
        products_mols = [
            Chem.MolFromSmiles(product) for product in products_str.split(".")
        ]

        reactants_atom_dict, reactants_atom_dict_neighbors = self._build_atom_dict(
            reactants_mols
        )
        products_atom_dict, products_atom_dict_neighbors = self._build_atom_dict(
            products_mols
        )

        products_orig_mapping_to_idx = {
            value: key
            for key, value in products_atom_idx_to_orig_mapping.items()
            if value != 0
        }
        reactants_orig_mapping_to_idx = {
            value: key
            for key, value in reactants_atom_idx_to_orig_mapping.items()
            if value != 0
        }

        (reactants_to_products_attn, products_to_reactants_attn) = attn
        attn = (reactants_to_products_attn + products_to_reactants_attn) / 2

        orig_reactants_to_products_attn = reactants_to_products_attn.copy()
        orig_products_to_reactants_attn = products_to_reactants_attn.copy()

        reactants_symmetric_indices = self._get_symmetric_atom_indices(reactants_mols)
        products_symmetric_indices = self._get_symmetric_atom_indices(products_mols)
        # print(orig_attn)
        orig_reactants_to_products_attn = self._apply_symmetric_attention(
            orig_reactants_to_products_attn, reactants_symmetric_indices, axis=1
        )
        orig_products_to_reactants_attn = self._apply_symmetric_attention(
            orig_products_to_reactants_attn, products_symmetric_indices, axis=0
        )

        # print(orig_reactants_to_products_attn)
        # print("~~~~~~~~~~~~~~~~~~")
        # print(orig_products_to_reactants_attn)

        orig_attn = (
            orig_reactants_to_products_attn + orig_products_to_reactants_attn
        ) / 2

        # print("~~~~~~~~~~~~~~~~~~")
        # print(orig_attn)

        assignment_probs = []
        if one_to_one_correspondence:
            for map_num in range(attn.shape[0]):
                if products_orig_mapping_to_idx.get(map_num + 1, 0):
                    row_highest_attn = products_orig_mapping_to_idx[map_num + 1]
                    col_highest_attn = reactants_orig_mapping_to_idx[map_num + 1]
                    products_atom_dict[row_highest_attn].SetAtomMapNum(map_num + 1)
                    reactants_atom_dict[col_highest_attn].SetAtomMapNum(map_num + 1)
                    attn[row_highest_attn] = 0
                    attn[:, col_highest_attn] = 0
                    assignment_probs.append(1.0)
                else:
                    highest_attn_score = attn.max()
                    highest_attn_score_indices = np.where(attn == highest_attn_score)
                    row_highest_attn = highest_attn_score_indices[0][0]
                    col_highest_attn = highest_attn_score_indices[1][0]
                    products_atom_dict[row_highest_attn].SetAtomMapNum(map_num + 1)
                    reactants_atom_dict[col_highest_attn].SetAtomMapNum(map_num + 1)
                    attn[row_highest_attn] = 0
                    attn[:, col_highest_attn] = 0
                    assignment_probs.append(
                        orig_attn[row_highest_attn, col_highest_attn]
                    )

                for (
                    product_atom_idx,
                    product_atom_env,
                ) in products_atom_dict_neighbors[row_highest_attn]:
                    for (
                        reactant_atom_idx,
                        reactant_atom_env,
                    ) in reactants_atom_dict_neighbors[col_highest_attn]:
                        if product_atom_env == reactant_atom_env:
                            attn[product_atom_idx, reactant_atom_idx] *= (
                                adjacent_atom_multiplier
                                * identical_adjacent_atom_multiplier
                            )
                        else:
                            attn[product_atom_idx, reactant_atom_idx] *= (
                                adjacent_atom_multiplier
                            )

        else:
            for map_num, row in enumerate(attn):
                # get partial mapping working for now
                # if products_orig_mapping_to_idx.get(map_num + 1, 0):
                #     row_highest_attn = products_orig_mapping_to_idx[map_num + 1]
                #     col_highest_attn = reactants_orig_mapping_to_idx[map_num + 1]
                #     products_atom_dict[row_highest_attn].SetAtomMapNum(map_num + 1)
                #     reactants_atom_dict[col_highest_attn].SetAtomMapNum(map_num + 1)
                #     print(map_num + 1, row_highest_attn, col_highest_attn)
                # else:
                highest_attn_score = row.max()
                highest_attn_indices = int(np.where(row == highest_attn_score)[0][0])
                products_atom_dict[map_num].SetAtomMapNum(map_num + 1)
                reactants_atom_dict[highest_attn_indices].SetAtomMapNum(map_num + 1)
                assignment_probs.append(orig_attn[map_num, highest_attn_indices])

        mapped_reactants_str = ".".join(
            [Chem.MolToSmiles(reactant, canonical=False) for reactant in reactants_mols]
        )
        mapped_products_str = ".".join(
            [Chem.MolToSmiles(product, canonical=False) for product in products_mols]
        )
        mapped_rxn_smiles = mapped_reactants_str + ">>" + mapped_products_str

        confidence = float(np.prod(assignment_probs))

        return mapped_rxn_smiles, confidence

    def get_data_from_partially_mapped_smiles(self, rxn_smiles):
        reactants_str, products_str = self._split_reaction_components(rxn_smiles)
        reactants_mols = [
            Chem.MolFromSmiles(reactant) for reactant in reactants_str.split(".")
        ]
        products_mols = [
            Chem.MolFromSmiles(product) for product in products_str.split(".")
        ]

        reactants_atom_idx_to_orig_mapping = {}
        reactants_atom_dict = {}
        reactants_atom_dict_neighbors = {}
        reactant_atom_num = 0
        for mol in reactants_mols:
            for atom in mol.GetAtoms():
                reactants_atom_dict[reactant_atom_num] = atom
                reactants_atom_idx_to_orig_mapping[reactant_atom_num] = (
                    atom.GetAtomMapNum()
                )
                reactants_atom_dict_neighbors[reactant_atom_num] = [
                    neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                ]
                atom.SetAtomMapNum(0)
                reactant_atom_num += 1

        products_atom_idx_to_orig_mapping = {}
        products_atom_dict = {}
        products_atom_dict_neighbors = {}
        product_atom_num = 0
        for mol in products_mols:
            for atom in mol.GetAtoms():
                products_atom_dict[product_atom_num] = atom
                products_atom_idx_to_orig_mapping[product_atom_num] = (
                    atom.GetAtomMapNum()
                )
                products_atom_dict_neighbors[product_atom_num] = [
                    neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                ]
                atom.SetAtomMapNum(0)
                product_atom_num += 1

        unmapped_reactants_strings = [
            Chem.MolToSmiles(reactant, canonical=False) for reactant in reactants_mols
        ]

        unmapped_products_strings = [
            Chem.MolToSmiles(product, canonical=False) for product in products_mols
        ]

        unmapped_rxn = (
            ".".join(unmapped_reactants_strings)
            + ">>"
            + ".".join(unmapped_products_strings)
        )

        return (
            unmapped_rxn,
            reactants_atom_idx_to_orig_mapping,
            products_atom_idx_to_orig_mapping,
        )

    def map_reaction(
        self,
        rxn_smiles: str,
        layer: int = 11,
        head: int = 7,
        sequence_max_length: int = 512,
        adjacent_atom_multiplier: float = 10,
        identical_adjacent_atom_multiplier: float = 10,
        one_to_one_correspondence: bool = False,
        start_from_partial_map: bool = False,
    ) -> ReactionMapperResult:
        """
        Maps a reaction SMILES string using a pre-trained Albert model.

        Args:
            rxn_smiles (str): A reaction SMILES string.
            layer (int): 0-based layer index to use for attention.
            head (int): 0-based head index to use for attention.

        Returns:
            str: A mapped reaction SMILES string with atom map numbers assigned.
        """
        default_mapping_dict = ReactionMapperResult(
            original_smiles="",
            selected_mapping="",
            possible_mappings={},
            mapping_type=self._mapper_type,
            mapping_score=None,
            additional_info=[{}],
        )
        if not self._reaction_smiles_valid(rxn_smiles):
            return default_mapping_dict

        reactants_atom_idx_to_orig_mapping = None
        products_atom_idx_to_orig_mapping = None
        if start_from_partial_map:
            (
                rxn_smiles,
                reactants_atom_idx_to_orig_mapping,
                products_atom_idx_to_orig_mapping,
            ) = self.get_data_from_partially_mapped_smiles(rxn_smiles)

        attn, tokens = self.get_attention_matrix_for_head(
            text=rxn_smiles,
            layer=layer,
            head=head,
            max_length=sequence_max_length,
            trim_padding=True,
        )

        if "[UNK]" in tokens:
            logger.warning("Unknown token in sequence")
            return default_mapping_dict

        if ">>" not in tokens:
            logger.warning("Sequence too long")

            return default_mapping_dict

        if len(tokens) >= sequence_max_length:
            logger.warning("Sequence too long")
            return default_mapping_dict

        # print(attn)
        string_info_dict = self.get_reactants_products_dict(tokens)
        attn_probs, _ = self.mask_attn_matrix(attn, string_info_dict)
        # print(attn_probs)
        reactants_to_products_attn, products_to_reactants_attn = (
            self.average_attn_scores(
                attn_probs,
                string_info_dict["reactants_start_index"],
                string_info_dict["reactants_end_index"],
                string_info_dict["products_start_index"],
            )
        )
        # print(attn)

        reactants_to_products_attn = self.remove_non_atom_rows_and_columns(
            reactants_to_products_attn, string_info_dict
        )
        products_to_reactants_attn = self.remove_non_atom_rows_and_columns(
            products_to_reactants_attn, string_info_dict
        )

        # print(attn)

        mapped_rxn_smiles, confidence = self.assign_atom_maps(
            rxn_smiles,
            (reactants_to_products_attn, products_to_reactants_attn),
            one_to_one_correspondence=one_to_one_correspondence,
            adjacent_atom_multiplier=adjacent_atom_multiplier,
            identical_adjacent_atom_multiplier=identical_adjacent_atom_multiplier,
            reactants_atom_idx_to_orig_mapping=reactants_atom_idx_to_orig_mapping,
            products_atom_idx_to_orig_mapping=products_atom_idx_to_orig_mapping,
        )

        if not self._verify_validity_of_mapping(mapped_rxn_smiles):
            return default_mapping_dict

        return ReactionMapperResult(
            original_smiles=rxn_smiles,
            selected_mapping=mapped_rxn_smiles,
            possible_mappings={},
            mapping_type=self._mapper_type,
            mapping_score=confidence,
            additional_info=[{}],
        )

    def map_reactions(self, reaction_list: List[str]) -> List[ReactionMapperResult]:
        """
        Map a list of reaction SMILES strings using the neural mapper.

        Args:
            reaction_list (List[str]): A list of unmapped reaction SMILES strings.

        Returns:
            List[ReactionMapperResult]: A list of mapping results, one per input
                reaction. Failed mappings return a result with an empty
                selected_mapping.
        """
        mapped_reactions = []
        for reaction in reaction_list:
            mapped_reactions.append(self.map_reaction(reaction))
        return mapped_reactions
