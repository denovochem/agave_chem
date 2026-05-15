from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, cast

import numpy as np
import torch
from rdkit import Chem
from transformers import AlbertForMaskedLM

from agave_chem.mappers.neural.constants import (
    smiles_token_to_id_dict,
    token_atom_identity_dict,
)
from agave_chem.mappers.neural.model import (
    AlbertWithAttentionAlignment,
    SupervisedConfig,
)
from agave_chem.mappers.neural.tokenizer import CustomTokenizer
from agave_chem.mappers.reaction_mapper import ReactionMapper, ReactionMapperResult
from agave_chem.utils.chem_utils import canonicalize_reaction_smiles
from agave_chem.utils.logging_config import logger
from agave_chem.utils.symmetry_classes import get_symmetry_class_from_mol


def placeholder():
    get_symmetry_class_from_mol()
    canonicalize_reaction_smiles()


class StringInfoDict(TypedDict):
    reactants_dict: Dict[int, str]
    products_dict: Dict[int, str]
    reactants_start_index: int
    reactants_end_index: int
    products_start_index: int
    products_end_index: int
    atom_tokens_dict: Dict[int, List[int]]
    non_atom_tokens: List[int]


def load_neural_albert_model(
    checkpoint_dir: str,
    device: torch.device,
    use_supervised: bool,
    max_length: int = 512,
    supervised_config: SupervisedConfig | None = None,
) -> AlbertForMaskedLM | AlbertWithAttentionAlignment:
    checkpoint_dir = str(checkpoint_dir)
    base_model = cast(
        AlbertForMaskedLM,
        AlbertForMaskedLM.from_pretrained(
            checkpoint_dir,
            attn_implementation="eager",
        ),
    )
    torch.nn.Module.to(base_model, device)

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
            if isinstance(self._model, AlbertWithAttentionAlignment):
                attn_probs = self._model.predict_attention_probs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )  # (B,S,S)
                attn = attn_probs[0].detach().cpu()  # (S,S)
            else:
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

    def get_aligned_attn_scores(
        self,
        out: np.ndarray,
        reactants_start_index: int,
        reactants_end_index: int,
        products_start_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and align cross-attention scores between reactant and product tokens.

        Slices the full attention matrix `out` to obtain two cross-attention
        sub-matrices: one representing how each product token attends to reactant
        tokens, and one representing how each reactant token attends to product
        tokens. The latter is transposed so that both returned arrays share the
        same index orientation (rows = product tokens, columns = reactant tokens).

        Args:
            out (np.ndarray): Square attention probability matrix of shape
                ``(sequence_length, sequence_length)``, where entry ``[i, j]``
                is the attention weight from token ``i`` to token ``j``.
            reactants_start_index (int): Index of the first reactant atom token
                in the sequence.
            reactants_end_index (int): Index of the last reactant atom token
                in the sequence (inclusive).
            products_start_index (int): Index of the first product token in the
                sequence; all tokens from this index onward are product tokens.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - **products_to_reactants_attn** (np.ndarray): Sub-matrix of shape
                  ``(n_product_tokens, n_reactant_tokens)`` giving the attention
                  weights from each product token to each reactant token.
                - **reactants_to_products_attn** (np.ndarray): Transposed sub-matrix
                  of shape ``(n_product_tokens, n_reactant_tokens)`` giving the
                  attention weights from each reactant token to each product token,
                  transposed so that rows correspond to product tokens and columns
                  to reactant tokens, matching the orientation of
                  ``products_to_reactants_attn``.
        """
        products_to_reactants_attn = out[
            products_start_index:,
            reactants_start_index : reactants_end_index + 1,
        ]  # products to reactants attention
        reactants_to_products_attn = out[
            reactants_start_index : reactants_end_index + 1, products_start_index:
        ].T  # reactants to products attention, transposed so indices align
        return products_to_reactants_attn, reactants_to_products_attn

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
        ranks = []
        seen_smiles_and_symmetry_classes: Dict[str, List[int]] = {}
        for i, mol in enumerate(mols):
            if Chem.MolToSmiles(mol) in seen_smiles_and_symmetry_classes:
                mol_symmetry_classes = seen_smiles_and_symmetry_classes[
                    Chem.MolToSmiles(mol)
                ]
            else:
                mol_symmetry_classes = get_symmetry_class_from_mol(mol)
                mol_symmetry_classes = [
                    ele + (i + 1) * 1000 for ele in mol_symmetry_classes
                ]
                seen_smiles_and_symmetry_classes[Chem.MolToSmiles(mol)] = (
                    mol_symmetry_classes
                )

            ranks.extend(mol_symmetry_classes)
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
        aligned_attn_scores: Tuple[np.ndarray, np.ndarray],
        one_to_one_correspondence: bool = True,
        adjacent_atom_multiplier: float = 30,
        identical_adjacent_atom_multiplier: float = 10,
        used_atom_divisor: float = 10,
        reactants_atom_idx_to_orig_mapping: Optional[Dict[int, int]] = None,
        products_atom_idx_to_orig_mapping: Optional[Dict[int, int]] = None,
    ) -> Tuple[str, float, Dict[str, int]]:
        """
        Assign atom-to-atom map numbers to a reaction SMILES using a pre-computed
        attention matrix.

        Handles symmetric atoms in both reactants and products by summing their
        attention contributions, preventing artificially low confidence scores
        caused by equivalent atoms splitting probability mass.

        Args:
            rxn_smiles (str): Unmapped reaction SMILES string of the form
                "reactants>>products".
            aligned_attn_scores (Tuple[np.ndarray, np.ndarray]): Tuple of attention matrices
                of shape (n_product_atoms, n_reactant_atoms).
            one_to_one_correspondence (bool): If True, enforces a one-to-one
                assignment using greedy selection of the global attention maximum.
                If False, assigns each product atom independently to its
                highest-attention reactant atom.
            adjacent_atom_multiplier (float): Multiplier applied to attention
                scores of atoms neighboring an already-mapped pair.
            identical_adjacent_atom_multiplier (float): Additional multiplier
                applied when a neighboring pair shares the same atom encoding.
            used_atom_divisor (float): Divisor applied to attention scores
                of reactant atoms that are already mapped if one_to_one_correspondence
                is False
            reactants_atom_idx_to_orig_mapping (Optional[Dict[int, int]]): Maps
                global reactant atom indices to existing atom map numbers, used
                to anchor partially pre-mapped reactions.
            products_atom_idx_to_orig_mapping (Optional[Dict[int, int]]): Maps
                global product atom indices to existing atom map numbers, used
                to anchor partially pre-mapped reactions.

        Returns:
            Tuple[str, float, Dict[str, int]]:
                - Mapped reaction SMILES string with atom map numbers assigned.
                - Confidence score computed as the product of per-atom assignment
                  probabilities.
                - Dictionary mapping oversubscribed reactant SMILES (atom maps
                  removed) to the maximum number of times any atom in that fragment
                  was assigned to multiple product atoms. Empty when
                  one_to_one_correspondence is True or when no oversubscription occurs.
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

        (reactants_to_products_attn, products_to_reactants_attn) = aligned_attn_scores

        orig_reactants_to_products_attn = reactants_to_products_attn.copy()
        orig_products_to_reactants_attn = products_to_reactants_attn.copy()

        reactants_symmetric_indices = self._get_symmetric_atom_indices(reactants_mols)
        products_symmetric_indices = self._get_symmetric_atom_indices(products_mols)

        orig_products_to_reactants_attn = self._apply_symmetric_attention(
            orig_products_to_reactants_attn, reactants_symmetric_indices, axis=1
        )
        orig_reactants_to_products_attn = self._apply_symmetric_attention(
            orig_reactants_to_products_attn, products_symmetric_indices, axis=0
        )

        ## If not a one-to-one correspondence, multiple product atoms could map to the same
        ## reactant atom. That reactant atom signal would be split, producing inaccurate
        ## assignment probabilities. So we use only product to reactant attention
        if one_to_one_correspondence:
            orig_attn = (
                orig_reactants_to_products_attn.copy()
                + orig_products_to_reactants_attn.copy()
            ) / 2
            attn = (
                reactants_to_products_attn.copy() + products_to_reactants_attn.copy()
            ) / 2
        else:
            orig_attn = orig_products_to_reactants_attn.copy()
            attn = products_to_reactants_attn.copy()

        assignment_probs = []
        for map_num in range(attn.shape[0]):
            if products_orig_mapping_to_idx.get(map_num + 1, 0):
                row_highest_attn = products_orig_mapping_to_idx[map_num + 1]
                col_highest_attn = reactants_orig_mapping_to_idx[map_num + 1]

                if reactants_atom_dict[col_highest_attn].GetAtomMapNum():
                    if not reactants_atom_dict[col_highest_attn].HasProp(
                        "oversubscribed_count"
                    ):
                        reactants_atom_dict[col_highest_attn].SetIntProp(
                            "oversubscribed_count", 1
                        )
                    else:
                        oversubscribed_count = reactants_atom_dict[
                            col_highest_attn
                        ].GetIntProp("oversubscribed_count")
                        reactants_atom_dict[col_highest_attn].SetIntProp(
                            "oversubscribed_count", oversubscribed_count + 1
                        )

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

                if reactants_atom_dict[col_highest_attn].GetAtomMapNum():
                    if not reactants_atom_dict[col_highest_attn].HasProp(
                        "oversubscribed_count"
                    ):
                        reactants_atom_dict[col_highest_attn].SetIntProp(
                            "oversubscribed_count", 1
                        )
                    else:
                        oversubscribed_count = reactants_atom_dict[
                            col_highest_attn
                        ].GetIntProp("oversubscribed_count")
                        reactants_atom_dict[col_highest_attn].SetIntProp(
                            "oversubscribed_count", oversubscribed_count + 1
                        )

                products_atom_dict[row_highest_attn].SetAtomMapNum(map_num + 1)
                reactants_atom_dict[col_highest_attn].SetAtomMapNum(map_num + 1)

                if one_to_one_correspondence:
                    attn[row_highest_attn] = 0
                    attn[:, col_highest_attn] = 0
                else:
                    attn[row_highest_attn] = 0
                    attn[:, col_highest_attn] /= used_atom_divisor

                assignment_probs.append(orig_attn[row_highest_attn, col_highest_attn])

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

        mapped_reactants_str = ".".join(
            [Chem.MolToSmiles(reactant, canonical=False) for reactant in reactants_mols]
        )
        mapped_products_str = ".".join(
            [Chem.MolToSmiles(product, canonical=False) for product in products_mols]
        )
        mapped_rxn_smiles = mapped_reactants_str + ">>" + mapped_products_str

        confidence = float(np.prod(assignment_probs))

        if one_to_one_correspondence:
            return mapped_rxn_smiles, confidence, {}

        oversubscribed_dict = {}
        for reactant in reactants_mols:
            max_oversubscribed_count = 0
            for reactant_atom in reactant.GetAtoms():
                if not reactant_atom.HasProp("oversubscribed_count"):
                    continue
                oversubscribed_count = reactant_atom.GetIntProp("oversubscribed_count")
                if oversubscribed_count > max_oversubscribed_count:
                    max_oversubscribed_count = oversubscribed_count
            if max_oversubscribed_count == 0:
                continue
            [atom.SetAtomMapNum(0) for atom in reactant.GetAtoms()]
            oversubscribed_dict[Chem.MolToSmiles(reactant)] = max_oversubscribed_count

        return mapped_rxn_smiles, confidence, oversubscribed_dict

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

    def _get_attention_matrices_batch(
        self,
        texts: List[str],
        layer: int = 11,
        head: int = 7,
        max_length: int = 512,
    ) -> List[Tuple[np.ndarray, List[str]]]:
        """
        Run batched neural network inference and return log-attention matrices for a
        list of reaction SMILES strings.

        Tokenizes all inputs together in a single padded batch, executes one forward
        pass, then trims each result to its non-padding length before applying the
        logarithm.

        Args:
            texts (List[str]): Reaction SMILES strings to encode. Must be non-empty.
            layer (int): 0-based layer index. Only used when the underlying model is
                the base AlbertForMaskedLM; ignored for AlbertWithAttentionAlignment.
            head (int): 0-based head index. Only used when the underlying model is the
                base AlbertForMaskedLM; ignored for AlbertWithAttentionAlignment.
            max_length (int): Maximum tokenization length. Must match the value used
                during training.

        Returns:
            List[Tuple[np.ndarray, List[str]]]: One entry per input string, each
                containing:
                    - Log-attention matrix of shape (real_seq_len, real_seq_len) as a
                      numpy array, with padding tokens stripped.
                    - List of token strings aligned to the attention matrix axes.
        """
        self._model.eval()

        enc = self._tokenizer(
            texts,
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
            if isinstance(self._model, AlbertWithAttentionAlignment):
                attn_probs = self._model.predict_attention_probs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )  # (B, S, S)
                attn_batch = attn_probs.detach().cpu()  # (B, S, S)
            else:
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
                    raise ValueError(
                        f"head must be in [0, {num_heads - 1}], got {head}"
                    )
                attn_batch = attentions[layer][:, head].detach().cpu()  # (B, S, S)

        results: List[Tuple[np.ndarray, List[str]]] = []
        for i in range(len(texts)):
            real_len = int(enc["attention_mask"][i].sum().item())
            attn_i = attn_batch[i, :real_len, :real_len]
            tokens_i = self._tokenizer.convert_ids_to_tokens(
                enc["input_ids"][i].tolist()
            )[:real_len]
            results.append((torch.log(attn_i).numpy(), tokens_i))

        return results

    def _map_from_attention(
        self,
        rxn_smiles: str,
        attn: np.ndarray,
        tokens: List[str],
        sequence_max_length: int = 512,
        adjacent_atom_multiplier: float = 10,
        identical_adjacent_atom_multiplier: float = 10,
        one_to_one_correspondence: bool = True,
        # canonicalize_reaction_smiles: bool = True,
        reactants_atom_idx_to_orig_mapping: Optional[Dict[int, int]] = None,
        products_atom_idx_to_orig_mapping: Optional[Dict[int, int]] = None,
    ) -> Tuple[ReactionMapperResult, Optional[str]]:
        """
        Assign atom mappings from a pre-computed log-attention matrix and token list.

        Performs all post-inference processing: token validation, attention masking,
        cross-attention score alignment, non-atom row/column removal, and atom map
        assignment. When one_to_one_correspondence is False and oversubscribed reactant
        atoms are detected, the expanded reaction SMILES (with extra reactant copies)
        is returned as the second element for a downstream retry pass.

        Args:
            rxn_smiles (str): An unmapped reaction SMILES string.
            attn (np.ndarray): Log-attention matrix of shape (seq_len, seq_len) as
                returned by _get_attention_matrices_batch or get_attention_matrix_for_head.
            tokens (List[str]): Token strings aligned to the attention matrix axes.
            sequence_max_length (int): Maximum allowed sequence length; sequences at or
                above this length are treated as failures.
            adjacent_atom_multiplier (float): Multiplier applied to attention scores of
                atoms neighboring an already-mapped pair.
            identical_adjacent_atom_multiplier (float): Additional multiplier applied
                when a neighboring pair shares the same atom encoding.
            one_to_one_correspondence (bool): If True, enforces greedy one-to-one
                assignment; if False, each product atom independently picks its best
                reactant atom.
            reactants_atom_idx_to_orig_mapping (Optional[Dict[int, int]]): Existing
                reactant atom map numbers to anchor partial mappings.
            products_atom_idx_to_orig_mapping (Optional[Dict[int, int]]): Existing
                product atom map numbers to anchor partial mappings.

        Returns:
            Tuple[ReactionMapperResult, Optional[str]]:
                - Mapping result. On failure (unknown tokens, sequence too long, or
                  invalid mapping), returns a result with an empty selected_mapping.
                - Expanded reaction SMILES with extra copies of oversubscribed reactant
                  fragments appended, or None if no oversubscription was detected. Only
                  non-None when one_to_one_correspondence is False and at least one
                  reactant atom was assigned to more than one product atom.
        """
        default_mapping_dict = ReactionMapperResult(
            original_smiles="",
            selected_mapping="",
            possible_mappings={},
            mapping_type=self._mapper_type,
            mapping_score=None,
            additional_info=[{}],
        )

        if "[UNK]" in tokens:
            logger.warning("Unknown token in sequence")
            return default_mapping_dict, None

        if ">>" not in tokens:
            logger.warning("Sequence too long")
            return default_mapping_dict, None

        if len(tokens) >= sequence_max_length:
            logger.warning("Sequence too long")
            return default_mapping_dict, None

        string_info_dict = self.get_reactants_products_dict(tokens)
        attn_probs, _ = self.mask_attn_matrix(attn, string_info_dict)

        products_to_reactants_attn, reactants_to_products_attn = (
            self.get_aligned_attn_scores(
                attn_probs,
                string_info_dict["reactants_start_index"],
                string_info_dict["reactants_end_index"],
                string_info_dict["products_start_index"],
            )
        )

        reactants_to_products_attn = self.remove_non_atom_rows_and_columns(
            reactants_to_products_attn, string_info_dict
        )
        products_to_reactants_attn = self.remove_non_atom_rows_and_columns(
            products_to_reactants_attn, string_info_dict
        )

        mapped_rxn_smiles, confidence, oversubscribed_dict = self.assign_atom_maps(
            rxn_smiles,
            (reactants_to_products_attn, products_to_reactants_attn),
            one_to_one_correspondence=one_to_one_correspondence,
            adjacent_atom_multiplier=adjacent_atom_multiplier,
            identical_adjacent_atom_multiplier=identical_adjacent_atom_multiplier,
            reactants_atom_idx_to_orig_mapping=reactants_atom_idx_to_orig_mapping,
            products_atom_idx_to_orig_mapping=products_atom_idx_to_orig_mapping,
        )

        expanded_rxn_smiles: Optional[str] = None
        if oversubscribed_dict:
            orig_reactants, orig_products = rxn_smiles.split(">>")
            new_reactants_list: List[str] = []
            for reactant, num_oversubscribed in oversubscribed_dict.items():
                new_reactants_list.extend([reactant] * num_oversubscribed)
            expanded_rxn_smiles = (
                orig_reactants
                + "."
                + ".".join(new_reactants_list)
                + ">>"
                + orig_products
            )

        if not self._verify_validity_of_mapping(mapped_rxn_smiles):
            return default_mapping_dict, expanded_rxn_smiles

        return ReactionMapperResult(
            original_smiles=rxn_smiles,
            selected_mapping=mapped_rxn_smiles,
            possible_mappings={},
            mapping_type=self._mapper_type,
            mapping_score=confidence,
            additional_info=[{}],
        ), expanded_rxn_smiles

    def _strip_unmapped_reactant_fragments(
        self,
        mapped_rxn_smiles: str,
        orig_rxn_smiles: str,
    ) -> str:
        """
        Remove unused extra reactant fragments from an oversubscription-expanded mapped reaction.

        Uses fragment counts rather than positional indices, so the result is
        independent of fragment ordering or SMILES canonicalization. For each
        fragment type (identified by canonical SMILES with atom maps stripped),
        the original count from orig_rxn_smiles is tracked in a counter. When
        processing the mapped reactants, each fragment first tries to consume an
        original slot; if one exists it is kept unconditionally (preserving
        legitimate spectators). Once all original slots for a given type are
        consumed, remaining copies are treated as extra and are kept only if at
        least one of their atoms carries a non-zero atom map number.

        Args:
            mapped_rxn_smiles (str): Mapped reaction SMILES from the second-pass
                retry, containing the original reactants plus any extra appended
                copies.
            orig_rxn_smiles (str): The pre-expansion reaction SMILES, used to
                determine the original fragment counts.

        Returns:
            str: The reaction SMILES with unused (fully unmapped) extra reactant
                fragments removed. Returns mapped_rxn_smiles unchanged if either
                reactants side cannot be parsed.
        """
        orig_reactants_str, _ = self._split_reaction_components(orig_rxn_smiles)
        mapped_reactants_str, products_str = self._split_reaction_components(
            mapped_rxn_smiles
        )
        orig_mol = Chem.MolFromSmiles(orig_reactants_str)
        mapped_mol = Chem.MolFromSmiles(mapped_reactants_str)
        if orig_mol is None or mapped_mol is None:
            return mapped_rxn_smiles

        def _canonical_key(frag: Chem.Mol) -> str:
            rw = Chem.RWMol(frag)
            for atom in rw.GetAtoms():
                atom.SetAtomMapNum(0)
            return Chem.MolToSmiles(rw)

        orig_counts: Dict[str, int] = defaultdict(int)
        for frag in Chem.GetMolFrags(orig_mol, asMols=True):
            orig_counts[_canonical_key(frag)] += 1

        kept_frags: List[str] = []
        for frag in Chem.GetMolFrags(mapped_mol, asMols=True):
            key = _canonical_key(frag)
            if orig_counts[key] > 0:
                orig_counts[key] -= 1
                kept_frags.append(Chem.MolToSmiles(frag, canonical=False))
            elif any(atom.GetAtomMapNum() != 0 for atom in frag.GetAtoms()):
                kept_frags.append(Chem.MolToSmiles(frag, canonical=False))

        return ".".join(kept_frags) + ">>" + products_str

    def map_reaction(
        self,
        rxn_smiles: str,
        layer: int = 11,
        head: int = 7,
        sequence_max_length: int = 512,
        adjacent_atom_multiplier: float = 10,
        identical_adjacent_atom_multiplier: float = 10,
        one_to_one_correspondence: bool = True,
        start_from_partial_map: bool = False,
    ) -> ReactionMapperResult:
        """
        Map a single reaction SMILES string using the neural mapper.

        Convenience wrapper around map_reactions for single-reaction use.

        Args:
            rxn_smiles (str): A reaction SMILES string.
            layer (int): 0-based layer index to use for attention.
            head (int): 0-based head index to use for attention.
            sequence_max_length (int): Maximum allowed sequence length.
            adjacent_atom_multiplier (float): Multiplier for adjacent atom attention scores.
            identical_adjacent_atom_multiplier (float): Additional multiplier when
                neighboring atom encodings match.
            one_to_one_correspondence (bool): If True, enforces greedy one-to-one assignment.
            start_from_partial_map (bool): If True, extracts and preserves existing atom
                map numbers from the input SMILES before remapping.

        Returns:
            ReactionMapperResult: Mapping result. On failure returns a result with an
                empty selected_mapping.
        """
        return self.map_reactions(
            [rxn_smiles],
            layer=layer,
            head=head,
            sequence_max_length=sequence_max_length,
            adjacent_atom_multiplier=adjacent_atom_multiplier,
            identical_adjacent_atom_multiplier=identical_adjacent_atom_multiplier,
            one_to_one_correspondence=one_to_one_correspondence,
            start_from_partial_map=start_from_partial_map,
        )[0]

    def map_reactions(
        self,
        reaction_list: List[str],
        layer: int = 11,
        head: int = 7,
        sequence_max_length: int = 512,
        adjacent_atom_multiplier: float = 10,
        identical_adjacent_atom_multiplier: float = 10,
        one_to_one_correspondence: bool = True,
        start_from_partial_map: bool = False,
        batch_size: int = 32,
    ) -> List[ReactionMapperResult]:
        """
        Map a list of reaction SMILES strings using batched neural network inference.

        Tokenizes and runs the model in batches of batch_size for efficiency, then
        assigns atom mappings for each reaction individually from the resulting
        attention matrices.

        Args:
            reaction_list (List[str]): A list of unmapped reaction SMILES strings.
            layer (int): 0-based layer index. Only used for the base AlbertForMaskedLM;
                ignored for AlbertWithAttentionAlignment.
            head (int): 0-based head index. Only used for the base AlbertForMaskedLM;
                ignored for AlbertWithAttentionAlignment.
            sequence_max_length (int): Maximum tokenization length.
            adjacent_atom_multiplier (float): Multiplier for adjacent atom attention scores.
            identical_adjacent_atom_multiplier (float): Additional multiplier when
                neighboring atom encodings match.
            one_to_one_correspondence (bool): If True, enforces greedy one-to-one assignment.
            start_from_partial_map (bool): If True, extracts and preserves existing atom
                map numbers before remapping.
            batch_size (int): Number of reactions to process in a single forward pass.

        Returns:
            List[ReactionMapperResult]: A list of mapping results, one per input
                reaction. Failed mappings return a result with an empty
                selected_mapping. When one_to_one_correspondence is False and
                oversubscribed reactant atoms are detected, a second batched
                inference pass is run on expanded reactions (extra reactant copies
                appended) with one_to_one_correspondence=True; successful retry
                results replace the first-pass results.
        """
        results: List[ReactionMapperResult] = [
            ReactionMapperResult(
                original_smiles="",
                selected_mapping="",
                possible_mappings={},
                mapping_type=self._mapper_type,
                mapping_score=None,
                additional_info=[{}],
            )
            for _ in reaction_list
        ]

        # Preprocess: validate and optionally strip existing partial maps
        prepared: List[
            Optional[Tuple[str, Optional[Dict[int, int]], Optional[Dict[int, int]]]]
        ] = []
        for rxn_smiles in reaction_list:
            rxn_smiles = canonicalize_reaction_smiles(rxn_smiles)
            if not self._reaction_smiles_valid(rxn_smiles):
                prepared.append(None)
                continue
            reactants_atom_idx_to_orig_mapping = None
            products_atom_idx_to_orig_mapping = None
            if start_from_partial_map:
                (
                    rxn_smiles,
                    reactants_atom_idx_to_orig_mapping,
                    products_atom_idx_to_orig_mapping,
                ) = self.get_data_from_partially_mapped_smiles(rxn_smiles)
            prepared.append(
                (
                    rxn_smiles,
                    reactants_atom_idx_to_orig_mapping,
                    products_atom_idx_to_orig_mapping,
                )
            )

        valid_pairs = [(i, p) for i, p in enumerate(prepared) if p is not None]
        valid_smiles = [p[0] for _, p in valid_pairs]

        if not valid_smiles:
            return results

        # Batched neural network inference
        attn_tokens_list: List[Tuple[np.ndarray, List[str]]] = []
        for batch_start in range(0, len(valid_smiles), batch_size):
            batch = valid_smiles[batch_start : batch_start + batch_size]
            attn_tokens_list.extend(
                self._get_attention_matrices_batch(
                    texts=batch,
                    layer=layer,
                    head=head,
                    max_length=sequence_max_length,
                )
            )

        # Assign atom maps per reaction from pre-computed attention matrices
        oversubscribed_cases: List[Tuple[int, str, str]] = []
        for local_idx, (
            orig_idx,
            (rxn_smiles, reactants_map, products_map),
        ) in enumerate(valid_pairs):
            attn, tokens = attn_tokens_list[local_idx]
            result, expanded_rxn_smiles = self._map_from_attention(
                rxn_smiles=rxn_smiles,
                attn=attn,
                tokens=tokens,
                sequence_max_length=sequence_max_length,
                adjacent_atom_multiplier=adjacent_atom_multiplier,
                identical_adjacent_atom_multiplier=identical_adjacent_atom_multiplier,
                one_to_one_correspondence=one_to_one_correspondence,
                reactants_atom_idx_to_orig_mapping=reactants_map,
                products_atom_idx_to_orig_mapping=products_map,
            )
            results[orig_idx] = result
            if expanded_rxn_smiles is not None:
                expanded_rxn_smiles = canonicalize_reaction_smiles(expanded_rxn_smiles)
                oversubscribed_cases.append((orig_idx, rxn_smiles, expanded_rxn_smiles))

        if not oversubscribed_cases:
            return results

        # Second pass: batch-map expanded reactions with one_to_one_correspondence=True
        expanded_smiles = [expanded for _, _, expanded in oversubscribed_cases]
        expanded_attn_tokens: List[Tuple[np.ndarray, List[str]]] = []
        for batch_start in range(0, len(expanded_smiles), batch_size):
            batch = expanded_smiles[batch_start : batch_start + batch_size]
            expanded_attn_tokens.extend(
                self._get_attention_matrices_batch(
                    texts=batch,
                    layer=layer,
                    head=head,
                    max_length=sequence_max_length,
                )
            )

        for local_idx, (orig_idx, orig_rxn_smiles, expanded_rxn) in enumerate(
            oversubscribed_cases
        ):
            attn, tokens = expanded_attn_tokens[local_idx]
            retry_result, _ = self._map_from_attention(
                rxn_smiles=expanded_rxn,
                attn=attn,
                tokens=tokens,
                sequence_max_length=sequence_max_length,
                adjacent_atom_multiplier=adjacent_atom_multiplier,
                identical_adjacent_atom_multiplier=identical_adjacent_atom_multiplier,
                one_to_one_correspondence=True,
            )
            if retry_result["selected_mapping"]:
                retry_result["original_smiles"] = orig_rxn_smiles
                retry_result["selected_mapping"] = (
                    self._strip_unmapped_reactant_fragments(
                        retry_result["selected_mapping"],
                        orig_rxn_smiles,
                    )
                )
                results[orig_idx] = retry_result

        return results
