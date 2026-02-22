from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import torch
from rdkit import Chem
from transformers import AlbertForMaskedLM

from agave_chem.mappers.neural.constants import (
    smiles_token_to_id_dict,
    token_atom_identity_dict,
)
from agave_chem.mappers.neural.tokenizer import CustomTokenizer
from agave_chem.mappers.reaction_mapper import ReactionMapper


class StringInfoDict(TypedDict):
    reactants_dict: Dict[int, str]
    products_dict: Dict[int, str]
    reactants_start_index: int
    reactants_end_index: int
    products_start_index: int
    products_end_index: int
    atom_tokens_dict: Dict[int, List[int]]
    non_atom_tokens: List[int]


class NeuralReactionMapper(ReactionMapper):
    """ """

    def __init__(
        self,
        mapper_name: str,
        mapper_weight: float = 3,
        checkpoint_path: Optional[str] = None,
    ):
        """ """

        super().__init__("neural", mapper_name, mapper_weight)

        if not checkpoint_path:
            checkpoint_path = str(
                files("agave_chem.datafiles").joinpath("albert_model")
            )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AlbertForMaskedLM.from_pretrained(
            checkpoint_path,
            attn_implementation="eager",
        ).to(self._device)

        self._model = model

        self._tokenizer = CustomTokenizer(smiles_token_to_id_dict)

    def get_attention_matrix_for_head(
        self,
        text: str,
        layer: int,
        head: int,
        max_length: int = 256,
        trim_padding: bool = True,
    ) -> Tuple[torch.Tensor, List[str]]:
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

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
            )

        attentions = outputs.attentions  # tuple[num_layers] of (B, H, S, S)

        if layer < 0 or layer >= len(attentions):
            raise ValueError(
                f"layer must be in [0, {len(attentions) - 1}], got {layer}"
            )

        num_heads = attentions[layer].shape[1]
        if head < 0 or head >= num_heads:
            raise ValueError(f"head must be in [0, {num_heads - 1}], got {head}")

        attn = attentions[layer][0, head].detach().cpu()  # (S, S)

        # Tokens for inspection/plotting
        token_ids = enc["input_ids"][0].tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(token_ids)

        if trim_padding:
            # attention_mask is 1 for non-pad, 0 for pad
            real_len = int(enc["attention_mask"][0].sum().item())
            attn = attn[:real_len, :real_len]
            tokens = tokens[:real_len]

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
    ) -> np.ndarray:
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
        ] = (
            -1e6
        )  # Set attention probability for reactant tokens to other reactant tokens to 0
        attn[
            string_info_dict["products_start_index"] : string_info_dict[
                "products_end_index"
            ]
            + 1,
            string_info_dict["products_start_index"] : string_info_dict[
                "products_end_index"
            ]
            + 1,
        ] = (
            -1e6
        )  # Set attention probability for product tokens to other product tokens to 0
        for i in string_info_dict[
            "non_atom_tokens"
        ]:  # Set attention probability for reactant or product tokens to non-atom tokens to 0
            attn[i] = -1e6
            attn[:, i] = -1e6
        for token_indices in string_info_dict[
            "atom_tokens_dict"
        ].values():  # Set attention probability for reactant and product tokens of different atom numbers to 0
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

        return probs

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
        products_to_reactants_attn = (
            out[
                reactants_start_index : reactants_end_index + 1, products_start_index:
            ].T
        )  # products to reactants attention, transposed so the two have the same shape
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

    def assign_atom_maps(
        self, rxn_smiles: str, attn: np.ndarray, one_to_one_correspondence: bool = False
    ) -> str:
        """
        Assign atom maps to a reaction SMILES string based on the attention matrix.

        Args:
            rxn_smiles (str): A reaction SMILES string.
            attn (np.ndarray): The attention matrix.
            one_to_one_correspondence (bool): Whether to use one-to-one correspondence for atom mapping.

        Returns:
            str: The mapped reaction SMILES string.
        """
        reactants_str = rxn_smiles.split(">>")[0]
        products_str = rxn_smiles.split(">>")[1]
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
            for atom in mol.GetAtoms():
                reactants_atom_dict[reactant_atom_num] = atom
                reactants_atom_dict_neighbors[reactant_atom_num] = [
                    neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                ]
                reactant_atom_num += 1

        products_atom_dict = {}
        products_atom_dict_neighbors = {}
        product_atom_num = 0
        for mol in products_mols:
            for atom in mol.GetAtoms():
                products_atom_dict[product_atom_num] = atom
                products_atom_dict_neighbors[product_atom_num] = [
                    neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                ]
                product_atom_num += 1

        if one_to_one_correspondence:
            for map_num in range(attn.shape[0]):
                highest_attn_score = attn.max()
                highest_attn_score_indices = np.where(attn == highest_attn_score)
                row_highest_attn = highest_attn_score_indices[0][0]
                col_highest_attn = highest_attn_score_indices[1][0]
                products_atom_dict[row_highest_attn].SetAtomMapNum(map_num + 1)
                reactants_atom_dict[col_highest_attn].SetAtomMapNum(map_num + 1)
                attn[row_highest_attn] = 0
                attn[:, col_highest_attn] = 0

                # Update neighbors
                for product_atom_idx in products_atom_dict_neighbors[row_highest_attn]:
                    for reactant_atom_idx in reactants_atom_dict_neighbors[
                        col_highest_attn
                    ]:
                        attn[product_atom_idx, reactant_atom_idx] *= 1

        else:
            for map_num, row in enumerate(attn):
                highest_attn_score = row.max()
                highest_attn_indices = int(np.where(row == highest_attn_score)[0][0])
                products_atom_dict[map_num].SetAtomMapNum(map_num + 1)
                reactants_atom_dict[highest_attn_indices].SetAtomMapNum(map_num + 1)

        mapped_reactants_str = ".".join(
            [Chem.MolToSmiles(reactant) for reactant in reactants_mols]
        )
        mapped_products_str = ".".join(
            [Chem.MolToSmiles(product) for product in products_mols]
        )
        mapped_rxn_smiles = mapped_reactants_str + ">>" + mapped_products_str

        return mapped_rxn_smiles

    def map_reaction(self, rxn_smiles: str, layer: int, head: int) -> Any:
        """
        Maps a reaction SMILES string using a pre-trained Albert model.

        Args:
            rxn_smiles (str): A reaction SMILES string.
            layer (int): 0-based layer index to use for attention.
            head (int): 0-based head index to use for attention.

        Returns:
            str: A mapped reaction SMILES string with atom map numbers assigned.
        """
        attn, tokens = self.get_attention_matrix_for_head(
            text=rxn_smiles,
            layer=layer,
            head=head,
            max_length=256,
            trim_padding=True,
        )

        if ">>" not in tokens:
            print("Sequence too long")

            return ""

        if len(tokens) == 256:
            print("Sequence too long")
            return ""

        string_info_dict = self.get_reactants_products_dict(tokens)
        attn = self.mask_attn_matrix(attn, string_info_dict)
        attn = self.average_attn_scores(
            attn,
            string_info_dict["reactants_start_index"],
            string_info_dict["reactants_end_index"],
            string_info_dict["products_start_index"],
        )

        attn = self.remove_non_atom_rows_and_columns(attn, string_info_dict)

        mapped_rxn_smiles = self.assign_atom_maps(rxn_smiles, attn)
        return mapped_rxn_smiles

    def map_reactions(self, reaction_list: List[str]) -> List[str]:
        """ """
        mapped_reactions = []
        for reaction in reaction_list:
            mapped_reactions.append(self.map_reaction(reaction))
        return mapped_reactions

    def map_reactions_parallel(self, reaction_list: List[str]) -> List[str]:
        return None
