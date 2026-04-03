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

    target_layer: int = 9  # Which layer's attention to supervise
    target_head: int = 5  # Which head's attention to supervise

    # Loss weighting for multitask learning
    mlm_loss_weight: float = 1.0
    attention_loss_weight: float = 1.0

    # Training mode
    multitask: bool = True  # If False, only attention alignment loss
    freeze_base_model: bool = False  # If True, only train the attention head

    # Dense layer config
    use_residual: bool = True  # Initialize with identity for residual learning


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

        if getattr(self.base_model.config, "_attn_implementation", None) != "eager":
            self.base_model.config._attn_implementation = "eager"

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
            return_dict=True,
        )

        attentions = outputs.attentions
        layer_attention = attentions[self.supervised_config.target_layer]  # (B,H,S,S)
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
    max_length: int = 256,
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
        sequence_max_length: int = 256,
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
        max_length: int = 256,
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
        return torch.log(attn)[1:-1, 1:-1].numpy(), tokens

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
        for i, token in enumerate(tokens[1:-1]):
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
        ]:  # Set attention logits for reactant or product tokens to non-atom tokens to very small value
            attn[i] = -1e6
            attn[:, i] = -1e6
        for token_indices in string_info_dict[
            "atom_tokens_dict"
        ].values():  # Set attention logits for reactant and product tokens of different atom numbers to very small value
            idx = np.asarray(token_indices, dtype=np.int64)

            diff_atom_mask = np.ones(attn.shape[1], dtype=bool)
            diff_atom_mask[idx] = False
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
        avg_attn = (reactants_to_products_attn + products_to_reactants_attn) / 2
        return avg_attn

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

    def get_duplicate_indices(self, list_of_lists: list) -> Dict[int, List[int]]:
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
        """ """
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

        reactants_atom_dict = {}
        reactants_atom_dict_neighbors = {}
        reactant_atom_num = 0
        for mol in reactants_mols:
            mol_reactants_atom_dict = {}
            mol_idx_to_atom_num = {}
            for atom in mol.GetAtoms():
                mol_reactants_atom_dict[reactant_atom_num] = atom
                mol_idx_to_atom_num[atom.GetIdx()] = reactant_atom_num
                reactant_atom_num += 1
            for atom_reactant_atom_num, atom in mol_reactants_atom_dict.items():
                reactants_atom_dict_neighbors[atom_reactant_atom_num] = [
                    (
                        mol_idx_to_atom_num[neighbor.GetIdx()],
                        self._encode_atom(neighbor),
                    )
                    for neighbor in atom.GetNeighbors()
                ]
            reactants_atom_dict.update(mol_reactants_atom_dict)

        products_atom_dict = {}
        products_atom_dict_neighbors = {}
        product_atom_num = 0
        for mol in products_mols:
            mol_products_atom_dict = {}
            mol_idx_to_atom_num = {}
            for atom in mol.GetAtoms():
                mol_products_atom_dict[product_atom_num] = atom
                mol_idx_to_atom_num[atom.GetIdx()] = product_atom_num
                product_atom_num += 1
            for atom_product_atom_num, atom in mol_products_atom_dict.items():
                products_atom_dict_neighbors[atom_product_atom_num] = [
                    (
                        mol_idx_to_atom_num[neighbor.GetIdx()],
                        self._encode_atom(neighbor),
                    )
                    for neighbor in atom.GetNeighbors()
                ]
            products_atom_dict.update(mol_products_atom_dict)

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

        orig_attn = attn.copy()

        reactants_mols_canonical = [
            list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
            for mol in reactants_mols
        ]
        products_mols_canonical = [
            list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
            for mol in products_mols
        ]

        reactants_symmetric_atom_indices = self.get_duplicate_indices(
            reactants_mols_canonical
        )
        # products_symmetric_atom_indices = self.get_duplicate_indices(products_mols_canonical)

        reactants_identical_atoms_indices = []
        for k, v in reactants_symmetric_atom_indices.items():
            reactants_identical_atoms_indices.append(tuple(sorted([k] + v)))

        reactants_identical_atoms_indices = list(set(reactants_identical_atoms_indices))

        symmetric_atom_new_val_mapping = {}
        for symmetric_atoms in reactants_identical_atoms_indices:
            summed_vals = np.sum(
                orig_attn[
                    :,
                    symmetric_atoms,
                ],
                axis=1,
            )
            for val in symmetric_atoms:
                symmetric_atom_new_val_mapping[val] = summed_vals

        for k, v in symmetric_atom_new_val_mapping.items():
            orig_attn[:, k] = v

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

                # Update neighbors
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
        layer: int = 9,
        head: int = 0,
        sequence_max_length: int = 256,
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

        string_info_dict = self.get_reactants_products_dict(tokens)
        attn_probs, _ = self.mask_attn_matrix(attn, string_info_dict)
        attn = self.average_attn_scores(
            attn_probs,
            string_info_dict["reactants_start_index"],
            string_info_dict["reactants_end_index"],
            string_info_dict["products_start_index"],
        )

        attn = self.remove_non_atom_rows_and_columns(attn, string_info_dict)

        mapped_rxn_smiles, confidence = self.assign_atom_maps(
            rxn_smiles,
            attn,
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
        """ """
        mapped_reactions = []
        for reaction in reaction_list:
            mapped_reactions.append(self.map_reaction(reaction))
        return mapped_reactions


## things to check:
# attention averaging works? Set as variable
# run mapping on large number of reactions, make sure it's consistent (all product atoms mapped, atom map numbers unique, atoms map to reactant atoms of same atom number, etc.)
# neighborhood multiplier? More sophisticated neighborhood multiplier
