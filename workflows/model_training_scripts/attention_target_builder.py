"""
Helper functions for building attention alignment target matrices.

This module contains the decomposed building blocks used by
``_build_attention_target_from_mapped_rxn_smiles_impl`` in
``albert_mapper_supervised_training.py``. Each function implements one
phase of the pipeline:

    1. :func:`assign_temp_atom_maps` – assign temporary atom map numbers
       to unmapped atoms and compute symmetry groups.
    2. :func:`augment_mapped_smiles` – apply random/canonical SMILES
       augmentation.
    3. :func:`classify_tokens` – classify tokens as atom / non-atom /
       to-sink and build the atom-map matching dictionary.
    4. :func:`build_index_attn_dict` – filter matched pairs and build the
       bidirectional attention index mapping.
    5. :func:`build_smoothed_attn_target` – spread attention weights
       across symmetry-equivalent atoms.
    6. :func:`apply_attention_sink` – route non-atom and unmapped atom
       attention to the sink position (last column).
"""

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem

from agave_chem.utils.chem_utils import (
    canonicalize_reaction_smiles,
    randomize_reaction_smiles,
)

# ============================================================
# Data classes for structured returns
# ============================================================


@dataclass
class TempAtomMapResult:
    """
    Result of :func:`assign_temp_atom_maps`.

    Attributes:
        reactant_mol (Chem.Mol): Reactant molecule with temp atom maps assigned.
        product_mol (Chem.Mol): Product molecule with temp atom maps assigned.
        reactant_symmetry_groups (List[List[int]]): Symmetry groups for reactants.
        product_symmetry_groups (List[List[int]]): Symmetry groups for products.
        symmetric_atom_token_indices_to_not_sink (List[int]): Atom map numbers
            in symmetric reactant groups that contain at least one mapped atom
            (mapnum < 600). These should not be sent to the sink.
        all_product_atoms_mapped (bool): True if every product atom had an
            original (non-zero) atom map number.
        atom_map_nums_to_sink_atomic_num_not_in_product (List[int]): Temp map
            numbers for unmapped reactant atoms whose atomic number does not
            appear among unmapped product atoms.
        new_mapped_rxn_smiles (str): The reaction SMILES after temp map
            assignment, before augmentation.
    """

    reactant_mol: Chem.Mol
    product_mol: Chem.Mol
    reactant_symmetry_groups: List[List[int]]
    product_symmetry_groups: List[List[int]]
    symmetric_atom_token_indices_to_not_sink: List[int]
    all_product_atoms_mapped: bool
    atom_map_nums_to_sink_atomic_num_not_in_product: List[int]
    new_mapped_rxn_smiles: str


@dataclass
class TokenClassificationResult:
    """
    Result of :func:`classify_tokens`.

    Attributes:
        matching_tokens_dict (Dict[str, List[int]]): Mapping from atom-map
            suffix strings (e.g. ``":12]"``) to lists of token indices.
            Only includes map numbers that appear exactly twice.
        token_index_to_mapnum (Dict[int, int]): Mapping from token index to
            atom map number.
        non_atom_token_indices (List[int]): Token indices classified as
            non-atom (structural tokens, excluding ``^`` and ``$``).
        atom_token_indices_to_sink (List[int]): Token indices for unmapped
            atoms that should be routed to the attention sink.
    """

    matching_tokens_dict: Dict[str, List[int]] = field(default_factory=dict)
    token_index_to_mapnum: Dict[int, int] = field(default_factory=dict)
    non_atom_token_indices: List[int] = field(default_factory=list)
    atom_token_indices_to_sink: List[int] = field(default_factory=list)


# ============================================================
# Phase 1: Assign temporary atom maps
# ============================================================


def assign_temp_atom_maps(mapped_rxn_smiles: str) -> TempAtomMapResult:
    """
    Assign temporary atom map numbers to unmapped atoms and compute symmetry groups.

    Assigns map numbers starting at 800 for unmapped product atoms and 600 for
    unmapped reactant atoms. Computes symmetry groups for both sides and
    identifies which symmetric reactant groups contain at least one originally
    mapped atom (those should not be sent to the sink).

    Args:
        mapped_rxn_smiles (str): Atom-mapped reaction SMILES with ``>>``
            separator.

    Returns:
        TempAtomMapResult: Structured result containing mutated mol objects,
            symmetry groups, sink classification data, and the re-serialized
            mapped reaction SMILES.
    """
    all_product_atoms_mapped = True
    seen_product_atom_nums_unmapped: List[int] = []
    product_str = mapped_rxn_smiles.split(">>")[1]
    product_mol = Chem.MolFromSmiles(product_str)
    unmapped_product_atom_map_num = 800
    for atom in product_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            all_product_atoms_mapped = False
            atom.SetAtomMapNum(unmapped_product_atom_map_num)
            unmapped_product_atom_map_num += 1
            seen_product_atom_nums_unmapped.append(atom.GetAtomicNum())

    atom_map_nums_to_sink_atomic_num_not_in_product: List[int] = []
    reactant_str = mapped_rxn_smiles.split(">>")[0]
    reactant_mol = Chem.MolFromSmiles(reactant_str)
    unmapped_reactant_atom_map_num = 600
    for atom in reactant_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(unmapped_reactant_atom_map_num)
            if atom.GetAtomicNum() not in seen_product_atom_nums_unmapped:
                atom_map_nums_to_sink_atomic_num_not_in_product.append(
                    unmapped_reactant_atom_map_num
                )
            unmapped_reactant_atom_map_num += 1

    from model_training_scripts.albert_mapper_supervised_training import (
        group_mappings_by_symmetry,
    )

    reactant_symmetry_groups = group_mappings_by_symmetry(reactant_mol)
    product_symmetry_groups = group_mappings_by_symmetry(product_mol)

    symmetric_atom_token_indices_to_not_sink: List[int] = []
    for reactant_symmetry_group in reactant_symmetry_groups:
        if any(x < 600 for x in reactant_symmetry_group):
            symmetric_atom_token_indices_to_not_sink.extend(reactant_symmetry_group)

    new_mapped_rxn_smiles = (
        Chem.MolToSmiles(reactant_mol) + ">>" + Chem.MolToSmiles(product_mol)
    )

    return TempAtomMapResult(
        reactant_mol=reactant_mol,
        product_mol=product_mol,
        reactant_symmetry_groups=reactant_symmetry_groups,
        product_symmetry_groups=product_symmetry_groups,
        symmetric_atom_token_indices_to_not_sink=symmetric_atom_token_indices_to_not_sink,
        all_product_atoms_mapped=all_product_atoms_mapped,
        atom_map_nums_to_sink_atomic_num_not_in_product=atom_map_nums_to_sink_atomic_num_not_in_product,
        new_mapped_rxn_smiles=new_mapped_rxn_smiles,
    )


# ============================================================
# Phase 2: Augment mapped SMILES
# ============================================================


def augment_mapped_smiles(
    mapped_rxn_smiles: str,
    randomize_mapped_rxn_smiles: bool = True,
    randomize_tautomer_pct: float = 0.10,
    canonicalize_mapped_rxn_smiles_pct: float = 0.05,
    seed: Optional[int] = None,
) -> str:
    """
    Apply random or canonical SMILES augmentation to a mapped reaction SMILES.

    When ``randomize_mapped_rxn_smiles`` is True, uses a seeded RNG to decide
    between canonicalization, randomization without tautomer shuffling, and
    randomization with tautomer shuffling. When False, returns the input
    unchanged.

    Args:
        mapped_rxn_smiles (str): Atom-mapped reaction SMILES.
        randomize_mapped_rxn_smiles (bool): If True, apply augmentation.
        randomize_tautomer_pct (float): Probability of tautomer randomization.
        canonicalize_mapped_rxn_smiles_pct (float): Probability of
            canonicalization.
        seed (Optional[int]): Seed for deterministic augmentation. When
            ``None``, uses global random state.

    Returns:
        str: The augmented (or unchanged) reaction SMILES.
    """
    if not randomize_mapped_rxn_smiles:
        return mapped_rxn_smiles

    rng = random.Random(seed) if seed is not None else random
    if rng.random() > canonicalize_mapped_rxn_smiles_pct:
        if rng.random() > randomize_tautomer_pct:
            return randomize_reaction_smiles(
                mapped_rxn_smiles,
                remove_mapping=False,
                randomize_tautomer=False,
                seed=seed,
            )
        else:
            return randomize_reaction_smiles(
                mapped_rxn_smiles,
                remove_mapping=False,
                randomize_tautomer=True,
                seed=seed,
            )
    else:
        return canonicalize_reaction_smiles(
            mapped_rxn_smiles, remove_mapping=False, canonicalize_tautomer=True
        )


# ============================================================
# Phase 3: Classify tokens
# ============================================================

_PATTERN = r":(\d+)\]$"


def classify_tokens(
    tokens: List[str],
    unmapped_tokens: List[str],
    token_atom_identity_dict: Dict[str, int],
    symmetric_atom_token_indices_to_not_sink: List[int],
    all_product_atoms_mapped: bool,
    atom_map_nums_to_sink_atomic_num_not_in_product: List[int],
) -> TokenClassificationResult:
    """
    Classify tokens and build the atom-map matching dictionary.

    Iterates over paired (mapped, unmapped) token lists to:
        - Identify non-atom tokens (structural tokens, excluding ``^``/``$``).
        - Extract atom map numbers from mapped tokens via regex.
        - Build ``matching_tokens_dict`` grouping token indices by map suffix.
        - Build ``token_index_to_mapnum`` mapping token indices to map numbers.
        - Identify unmapped atom tokens that should be routed to the sink.

    After iteration, filters ``matching_tokens_dict`` to keep only map numbers
    that appear exactly twice (once in reactants, once in products).

    Args:
        tokens (List[str]): Token list from the mapped (possibly augmented)
            reaction SMILES.
        unmapped_tokens (List[str]): Token list from the unmapped reaction
            SMILES.
        token_atom_identity_dict (Dict[str, int]): Mapping from token strings
            to atomic numbers (0 = non-atom).
        symmetric_atom_token_indices_to_not_sink (List[int]): Atom map numbers
            in symmetric reactant groups that should not be sent to the sink.
        all_product_atoms_mapped (bool): Whether all product atoms were
            originally mapped.
        atom_map_nums_to_sink_atomic_num_not_in_product (List[int]): Temp map
            numbers for unmapped reactant atoms not present in unmapped
            products.

    Returns:
        TokenClassificationResult: Structured result with matching dict, mapnum
            mapping, and token classification lists.
    """
    matching_tokens_dict: Dict[str, List[int]] = {}
    token_index_to_mapnum: Dict[int, int] = {}
    non_atom_token_indices: List[int] = []
    atom_token_indices_to_sink: List[int] = []

    for i, [token, unmapped_token] in enumerate(zip(tokens, unmapped_tokens)):
        if token_atom_identity_dict.get(unmapped_token) == 0 and unmapped_token not in [
            "^",
            "$",
        ]:
            non_atom_token_indices.append(i)
        m = re.search(_PATTERN, token)
        if not m:
            continue
        key = m.group()  # e.g. ":12]"
        matching_tokens_dict.setdefault(key, []).append(i)
        mapnum = int(m.group()[1:-1])
        token_index_to_mapnum[i] = mapnum
        if (
            mapnum >= 600
            and mapnum <= 800
            and mapnum not in symmetric_atom_token_indices_to_not_sink
            and all_product_atoms_mapped
        ) or (
            mapnum in atom_map_nums_to_sink_atomic_num_not_in_product
            and mapnum not in symmetric_atom_token_indices_to_not_sink
        ):
            atom_token_indices_to_sink.append(i)

    # keep only map nums that appear exactly twice (once reactant, once product)
    matching_tokens_dict = {
        k: v for k, v in matching_tokens_dict.items() if len(v) == 2
    }

    return TokenClassificationResult(
        matching_tokens_dict=matching_tokens_dict,
        token_index_to_mapnum=token_index_to_mapnum,
        non_atom_token_indices=non_atom_token_indices,
        atom_token_indices_to_sink=atom_token_indices_to_sink,
    )


# ============================================================
# Phase 4: Build index_attn_dict
# ============================================================


def build_index_attn_dict(matching_tokens_dict: Dict[str, List[int]]) -> Dict[int, int]:
    """
    Build a bidirectional attention index mapping from matched token pairs.

    Filters to pairs appearing exactly twice, then creates a bidirectional
    mapping ``index_attn_dict[a] = b`` and ``index_attn_dict[b] = a``.
    If a token index already appears in the mapping (duplicate), that pair
    is skipped.

    Args:
        matching_tokens_dict (Dict[str, List[int]]): Mapping from atom-map
            suffix strings to lists of exactly two token indices.

    Returns:
        Dict[int, int]: Bidirectional mapping between paired token indices.
    """
    index_attn_dict: Dict[int, int] = {}
    for a, b in matching_tokens_dict.values():
        if a in index_attn_dict or b in index_attn_dict:
            continue
        index_attn_dict[a] = b
        index_attn_dict[b] = a
    return index_attn_dict


# ============================================================
# Phase 5: Build smoothed attention target
# ============================================================


def build_smoothed_attn_target(
    index_attn_dict: Dict[int, int],
    token_index_to_mapnum: Dict[int, int],
    tokens: List[str],
    reactant_symmetry_groups: List[List[int]],
    product_symmetry_groups: List[List[int]],
    n: int,
) -> np.ndarray:
    """
    Build an attention target matrix with symmetry-smoothed weights.

    Constructs symmetry lookups for reactants and products, propagates
    attention targets to unmapped symmetric atoms, and spreads attention
    weight uniformly across symmetry-equivalent atoms on the destination side.

    The input ``index_attn_dict`` is copied internally so the caller's dict
    is not modified.

    Args:
        index_attn_dict (Dict[int, int]): Bidirectional token-index pairing.
        token_index_to_mapnum (Dict[int, int]): Mapping from token index to
            atom map number.
        tokens (List[str]): Full token list for the reaction SMILES.
        reactant_symmetry_groups (List[List[int]]): Symmetry groups for
            reactants.
        product_symmetry_groups (List[List[int]]): Symmetry groups for
            products.
        n (int): Total number of tokens (matrix dimension).

    Returns:
        np.ndarray: ``(n, n)`` float32 attention target matrix with
            symmetry-smoothed weights.
    """
    # Work on a copy so the caller's dict is not mutated
    index_attn_dict = dict(index_attn_dict)

    arrow_index = tokens.index(">>")

    reactant_sym_lookup: Dict[int, List[int]] = {}
    for group in reactant_symmetry_groups:
        for mn in group:
            reactant_sym_lookup[mn] = group

    product_sym_lookup: Dict[int, List[int]] = {}
    for group in product_symmetry_groups:
        for mn in group:
            product_sym_lookup[mn] = group

    # Map numbers for ALL atom tokens, split by side
    mapnum_to_reactant_token: Dict[int, int] = {}
    mapnum_to_product_token: Dict[int, int] = {}
    for idx, mn in token_index_to_mapnum.items():
        if idx < arrow_index:
            mapnum_to_reactant_token[mn] = idx
        else:
            mapnum_to_product_token[mn] = idx

    # Give unmapped symmetric atoms the same target as their matched counterpart
    for group in reactant_symmetry_groups:
        ref_dst = None
        for mn in group:
            tok = mapnum_to_reactant_token.get(mn)
            if tok is not None and tok in index_attn_dict:
                ref_dst = index_attn_dict[tok]
                break
        if ref_dst is None:
            continue
        for mn in group:
            tok = mapnum_to_reactant_token.get(mn)
            if tok is not None and tok not in index_attn_dict:
                index_attn_dict[tok] = ref_dst

    for group in product_symmetry_groups:
        ref_dst = None
        for mn in group:
            tok = mapnum_to_product_token.get(mn)
            if tok is not None and tok in index_attn_dict:
                ref_dst = index_attn_dict[tok]
                break
        if ref_dst is None:
            continue
        for mn in group:
            tok = mapnum_to_product_token.get(mn)
            if tok is not None and tok not in index_attn_dict:
                index_attn_dict[tok] = ref_dst

    attn_target = np.zeros((n, n), dtype=np.float32)

    for src, dst in index_attn_dict.items():
        dst_mapnum = token_index_to_mapnum.get(dst)
        if dst_mapnum is None:
            attn_target[src, dst] = 1.0
            continue

        if src < arrow_index:
            # Reactant row: spread over symmetric product atoms
            sym_group = product_sym_lookup.get(dst_mapnum, [dst_mapnum])
            sym_indices = [
                mapnum_to_product_token[m]
                for m in sym_group
                if m in mapnum_to_product_token
            ]
        else:
            # Product row: spread over symmetric reactant atoms
            sym_group = reactant_sym_lookup.get(dst_mapnum, [dst_mapnum])
            sym_indices = [
                mapnum_to_reactant_token[m]
                for m in sym_group
                if m in mapnum_to_reactant_token
            ]

        if not sym_indices:
            sym_indices = [dst]
        weight = 1.0 / len(sym_indices)
        for sym_idx in sym_indices:
            attn_target[src, sym_idx] = weight

    return attn_target


# ============================================================
# Phase 6: Apply attention sink
# ============================================================


def apply_attention_sink(
    attn_target: np.ndarray,
    non_atom_token_indices: List[int],
    atom_token_indices_to_sink: List[int],
) -> np.ndarray:
    """
    Route non-atom and unmapped atom attention to the sink position (last column).

    Sets ``attn_target[i, -1] = 1.0`` for every index ``i`` in
    ``non_atom_token_indices`` or ``atom_token_indices_to_sink``. The matrix
    is modified in place and also returned for convenience.

    Args:
        attn_target (np.ndarray): ``(N, N)`` attention target matrix.
        non_atom_token_indices (List[int]): Token indices classified as
            non-atom.
        atom_token_indices_to_sink (List[int]): Token indices for unmapped
            atoms that should be routed to the sink.

    Returns:
        np.ndarray: The same ``attn_target`` matrix, modified in place.
    """
    for i, row in enumerate(attn_target):
        if i in non_atom_token_indices:
            row[-1] = 1
            continue
        if i in atom_token_indices_to_sink:
            row[-1] = 1
            continue
    return attn_target
