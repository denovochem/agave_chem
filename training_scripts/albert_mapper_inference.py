from typing import Dict, List, Tuple, TypedDict

import numpy as np
import torch
from albert_mapper import CustomTokenizer
from rdkit import Chem
from transformers import AlbertForMaskedLM
from utils.constants import smiles_token_to_id_dict, token_atom_identity_dict


class StringInfoDict(TypedDict):
    reactants_dict: Dict[int, str]
    products_dict: Dict[int, str]
    reactants_start_index: int
    reactants_end_index: int
    products_start_index: int
    products_end_index: int
    atom_tokens_dict: Dict[int, List[int]]
    non_atom_tokens: List[int]


def sanitize_input_rxn_string(rxn_smiles: str) -> str:
    """
    Sanitize the input reaction SMILES string by parsing it into reactants and products
    and checking that the constituent molecules are standardized.

    Standardization:
    1. Ensuring each fragment can be rounded-tripped through RDKit
    2. Removing mapping numbers
    3. Remove duplicate fragments
    4. Make sure ">>" is in the string, only once
    5. Removing isotopes
    6. Canonicalizing SMILES strings
    7. Isomerizing SMILES strings

    Args:
        rxn_smiles (str): Reaction SMILES string

    Returns:
        str: Sanitized reaction SMILES string
    """
    reactants_str = rxn_smiles.split(">>")[0]
    products_str = rxn_smiles.split(">>")[1]

    reactants_strs = list(set(reactants_str.split(".")))
    products_strs = list(set(products_str.split(".")))

    reactants_mols = [Chem.MolFromSmiles(reactant) for reactant in reactants_strs]
    products_mols = [Chem.MolFromSmiles(product) for product in products_strs]

    reactants_mols = [ele for ele in reactants_mols if ele is not None]
    products_mols = [ele for ele in products_mols if ele is not None]

    standardized_reactants_str = "".join(
        [
            Chem.MolToSmiles(reactant, canonicalSmiles=True, isomericSmiles=True)
            for reactant in reactants_mols
        ]
    )
    standardized_products_str = "".join(
        [
            Chem.MolToSmiles(product, canonicalSmiles=True, isomericSmiles=True)
            for product in products_mols
        ]
    )
    standardized_rxn_smiles = (
        standardized_reactants_str + ">>" + standardized_products_str
    )

    return standardized_rxn_smiles


def get_attention_matrix_for_head(
    checkpoint_dir: str,
    text: str,
    layer: int,
    head: int,
    max_length: int = 256,
    trim_padding: bool = True,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Returns the attention matrix for a given layer/head for a single input string.

    Args:
        checkpoint_dir: path to save_pretrained() checkpoint folder
        text: input reaction SMILES string (raw is fine; CustomTokenizer preprocesses)
        layer: 0-based layer index
        head: 0-based head index
        max_length: tokenization length (should match training, e.g. 256)
        trim_padding: if True, slices matrix down to non-pad tokens only

    Returns:
        attn: Tensor of shape (seq_len, seq_len) (trimmed if requested)
        tokens: list[str] tokens aligned to attn axes (trimmed if requested)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CustomTokenizer(smiles_token_to_id_dict)
    model = AlbertForMaskedLM.from_pretrained(
        checkpoint_dir,
        attn_implementation="eager",
    ).to(device)

    model.eval()

    enc = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )

    attentions = outputs.attentions  # tuple[num_layers] of (B, H, S, S)

    if layer < 0 or layer >= len(attentions):
        raise ValueError(f"layer must be in [0, {len(attentions) - 1}], got {layer}")

    num_heads = attentions[layer].shape[1]
    if head < 0 or head >= num_heads:
        raise ValueError(f"head must be in [0, {num_heads - 1}], got {head}")

    attn = attentions[layer][0, head].detach().cpu()  # (S, S)

    # Tokens for inspection/plotting
    token_ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    if trim_padding:
        # attention_mask is 1 for non-pad, 0 for pad
        real_len = int(enc["attention_mask"][0].sum().item())
        attn = attn[:real_len, :real_len]
        tokens = tokens[:real_len]

    return torch.log(attn)[1:-1, 1:-1].numpy(), tokens


def get_reactants_products_dict(
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
        col_mask = np.ones(attn.shape[1], dtype=bool)
        col_mask[idx] = False
        attn[np.ix_(idx, col_mask)] = -1e6

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

    return probs


def average_attn_scores(
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
    ].T  # products to reactants attention, transposed so the two have the same shape
    avg_attn = (reactants_to_products_attn + products_to_reactants_attn) / 2
    return avg_attn


def assign_atom_maps(
    rxn_smiles: str, attn: np.ndarray, one_to_one_correspondence: bool = False
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
    products_mols = [Chem.MolFromSmiles(product) for product in products_str.split(".")]

    reactants_atom_dict = {}
    reactant_atom_num = 0
    for mol in reactants_mols:
        for atom in mol.GetAtoms():
            reactants_atom_dict[reactant_atom_num] = atom
            reactant_atom_num += 1

    products_atom_dict = {}
    product_atom_num = 0
    for mol in products_mols:
        for atom in mol.GetAtoms():
            products_atom_dict[product_atom_num] = atom
            product_atom_num += 1

    # Remove rows and columns that are all 0 - non-atom tokens
    nonzero_rows_mask = ~np.all(attn == 0, axis=1)
    attn = attn[nonzero_rows_mask, :]

    nonzero_cols_mask = ~np.all(attn == 0, axis=0)
    attn = attn[:, nonzero_cols_mask]

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


def map_reaction(rxn_smiles: str, checkpoint_dir: str, layer: int, head: int) -> str:
    """
    Maps a reaction SMILES string using a pre-trained Albert model.

    Args:
        rxn_smiles (str): A reaction SMILES string.
        checkpoint_dir (str): Path to the pre-trained Albert model checkpoint folder.
        layer (int): 0-based layer index to use for attention.
        head (int): 0-based head index to use for attention.

    Returns:
        str: A mapped reaction SMILES string with atom map numbers assigned.
    """
    attn, toks = get_attention_matrix_for_head(
        checkpoint_dir=checkpoint_dir,
        text=rxn_smiles,
        layer=layer,
        head=head,
        max_length=256,
        trim_padding=True,
    )
    string_info_dict = get_reactants_products_dict(toks)
    attn = mask_attn_matrix(attn, string_info_dict)
    attn = average_attn_scores(
        attn,
        string_info_dict["reactants_start_index"],
        string_info_dict["reactants_end_index"],
        string_info_dict["products_start_index"],
    )

    mapped_rxn_smiles = assign_atom_maps(rxn_smiles, attn)
    return mapped_rxn_smiles


## things to check:
# attention averaging works? Set as variable
# run mapping on large number of reactions, make sure it's consistent (all product atoms mapped, atom map numbers unique, atoms map to reactant atoms of same atom number, etc.)
# neighborhood multiplier?
