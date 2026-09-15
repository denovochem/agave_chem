"""
post_processor.py — CPU post-processing for neural reaction atom-mapping.

Provides ``NeuralPostProcessor``, a ``ReactionMapper`` subclass that holds all
CPU-bound post-inference logic (attention masking, atom assignment, symmetry
correction, oversubscription handling) extracted from ``NeuralReactionMapper``.
This separation allows the post-processing to be parallelised across worker
processes via ``post_process_batch`` while the GPU-bound inference remains in
the parent mapper.
"""

import multiprocessing as mp
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, TypedDict, Union

import numpy as np
from rdkit import Chem

from agave_chem.mappers.neural.constants import token_atom_identity_dict
from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.types import ReactionInput, ReactionMapperResult
from agave_chem.utils.logging_config import disable_library_logging, logger
from agave_chem.utils.reaction_balancing import (
    compute_unmapped_product_atom_islands,
    determine_one_to_one_correspondence,
)
from agave_chem.utils.symmetry_classes import get_symmetry_class_from_mol


class StringInfoDict(TypedDict):
    reactants_dict: Dict[int, str]
    products_dict: Dict[int, str]
    reactants_start_index: int
    reactants_end_index: int
    products_start_index: int
    products_end_index: int
    atom_tokens_dict: Dict[int, List[int]]
    non_atom_tokens: List[int]


# ── Worker globals ───────────────────────────────────────────────────────────

_post_processor: "Optional[NeuralPostProcessor]" = None


def _init_worker(
    adjacent_atom_multiplier: float,
    identical_adjacent_atom_multiplier: float,
    used_atom_divisor: float,
    sequence_max_length: int,
    mapper_type: str,
) -> None:
    """
    Initialize the per-worker NeuralPostProcessor instance.

    Called exactly once per worker process by ``multiprocessing.Pool``.
    Stores the post-processor in a module-level global so it is reused
    across all tasks handled by this worker.

    Args:
        adjacent_atom_multiplier (float): Multiplier applied to attention
            scores of atoms neighboring an already-mapped pair.
        identical_adjacent_atom_multiplier (float): Additional multiplier
            applied when a neighboring pair shares the same atom encoding.
        used_atom_divisor (float): Divisor applied to attention scores of
            reactant atoms that are already mapped when
            one_to_one_correspondence is False.
        sequence_max_length (int): Maximum tokenization length for the model.
        mapper_type (str): Mapper type string for result metadata.
    """
    global _post_processor
    disable_library_logging()
    _post_processor = NeuralPostProcessor(
        adjacent_atom_multiplier=adjacent_atom_multiplier,
        identical_adjacent_atom_multiplier=identical_adjacent_atom_multiplier,
        used_atom_divisor=used_atom_divisor,
        sequence_max_length=sequence_max_length,
        mapper_type=mapper_type,
    )


# Task type: (rxn_smiles, attn, tokens, o2o_flag, tautomer_sym, transform_sym)
_PostProcessTask = Tuple[
    str,  # rxn_smiles
    np.ndarray,  # attn matrix
    List[str],  # tokens
    bool,  # one_to_one_correspondence
    bool,  # consider_tautomer_symmetry
    bool,  # consider_transform_symmetry
]


def _post_process_one(
    task: _PostProcessTask,
) -> Tuple[ReactionMapperResult, Optional[str]]:
    """
    Post-process a single reaction's attention matrix into a mapping result.

    Uses the module-level ``_post_processor`` initialised by ``_init_worker``.

    Args:
        task (_PostProcessTask): Tuple of (rxn_smiles, attn, tokens,
            one_to_one_correspondence, consider_tautomer_symmetry,
            consider_transform_symmetry).

    Returns:
        Tuple[ReactionMapperResult, Optional[str]]: Mapping result and
            expanded reaction SMILES (or None if no oversubscription).
    """
    rxn_smiles, attn, tokens, o2o, taut_sym, transform_sym = task
    return _post_processor.map_from_attention(  # type: ignore[union-attr]
        rxn_smiles=rxn_smiles,
        attn=attn,
        tokens=tokens,
        one_to_one_correspondence=o2o,
        consider_tautomer_symmetry=taut_sym,
        consider_transform_symmetry=transform_sym,
    )


class NeuralPostProcessor(ReactionMapper):
    """
    CPU-bound post-inference processing for neural reaction atom-mapping.

    Holds all attention-masking, atom-assignment, symmetry-correction, and
    oversubscription-handling logic extracted from ``NeuralReactionMapper``.
    This class has no dependency on torch or the neural model, making it
    picklable for ``multiprocessing``.

    Use ``post_process_batch`` to process a batch of attention matrices
    in serial or parallel, or ``map_from_attention`` for a single reaction.
    """

    def __init__(
        self,
        adjacent_atom_multiplier: float = 10,
        identical_adjacent_atom_multiplier: float = 10,
        used_atom_divisor: float = 10,
        sequence_max_length: int = 1024,
        mapper_type: str = "neural",
    ) -> None:
        """
        Initialize the NeuralPostProcessor instance.

        Args:
            adjacent_atom_multiplier (float): Multiplier applied to attention
                scores of atoms neighboring an already-mapped pair.
            identical_adjacent_atom_multiplier (float): Additional multiplier
                applied when a neighboring pair shares the same atom encoding.
            used_atom_divisor (float): Divisor applied to attention scores of
                reactant atoms that are already mapped when
                one_to_one_correspondence is False. Lower values increase the
                likelihood of detecting oversubscription.
            sequence_max_length (int): Maximum tokenization length for the
                model.
            mapper_type (str): Mapper type string used in result metadata.
        """
        super().__init__(mapper_type, "neural_post_processor", 0)
        self._adjacent_atom_multiplier = adjacent_atom_multiplier
        self._identical_adjacent_atom_multiplier = identical_adjacent_atom_multiplier
        self._used_atom_divisor = used_atom_divisor
        self._sequence_max_length = sequence_max_length

    def map_reaction(self, reaction_smiles: Union[str, object]) -> ReactionMapperResult:
        """
        Not supported — use ``map_from_attention`` or ``post_process_batch``.
        """
        raise NotImplementedError(
            "NeuralPostProcessor does not map reactions directly; "
            "use map_from_attention or post_process_batch instead."
        )

    def map_reactions(
        self, reaction_smiles_list: Union[List[str], List[ReactionInput]]
    ) -> List[ReactionMapperResult]:
        """
        Not supported — use ``post_process_batch`` instead.
        """
        raise NotImplementedError(
            "NeuralPostProcessor does not map reactions directly; "
            "use post_process_batch instead."
        )

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

    def _get_symmetric_atom_indices(
        self,
        mols: List[Chem.Mol],
        consider_tautomer_symmetry: bool = True,
        consider_transform_symmetry: bool = True,
    ) -> Dict[int, List[int]]:
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
            consider_tautomer_symmetry (bool): If True, atoms that interconvert via
                tautomerism are treated as symmetrically equivalent.
            consider_transform_symmetry (bool): If True, apply functional group
                normalization transforms before computing symmetry classes.

        Returns:
            Dict[int, List[int]]: A mapping from each globally-offset atom index
                to a list of other globally-offset atom indices that are
                topologically equivalent. Only atoms with at least one symmetric
                partner are included.
        """
        ranks = []
        seen_smiles_and_symmetry_classes: Dict[str, List[int]] = {}
        for i, mol in enumerate(mols):
            mol_smiles = Chem.MolToSmiles(mol)
            if mol_smiles in seen_smiles_and_symmetry_classes:
                mol_symmetry_classes = seen_smiles_and_symmetry_classes[mol_smiles]
            else:
                raw_classes = get_symmetry_class_from_mol(
                    mol,
                    consider_tautomers=consider_tautomer_symmetry,
                    consider_transforms=consider_transform_symmetry,
                )
                mol_symmetry_classes = [ele + (i + 1) * 1000 for ele in raw_classes]
                seen_smiles_and_symmetry_classes[mol_smiles] = mol_symmetry_classes

            ranks.extend(mol_symmetry_classes)
        result = self.get_duplicate_indices([ranks])
        return result

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

    def _apply_noisy_or(
        self,
        attn: np.ndarray,
        symmetric_indices: Dict[int, List[int]],
        axis: int,
    ) -> np.ndarray:
        """
        Combine attention scores for symmetric atoms using noisy-OR.

        For each group of symmetric atoms, replaces each member's attention
        slice (row or column) with the noisy-OR combination across the group:
        ``1 - prod(1 - p_i)``. This treats each symmetric atom's attention as
        an independent opinion about the same event, combining them so that
        multiple moderate confidences yield a higher combined confidence
        without reaching certainty unless at least one source is certain.

        Values are computed from the input array before any modifications are
        applied, ensuring groups do not influence one another.

        Args:
            attn (np.ndarray): Attention matrix of shape
                (n_product_atoms, n_reactant_atoms).
            symmetric_indices (Dict[int, List[int]]): Output of
                _get_symmetric_atom_indices — maps each atom index to its
                symmetric partners.
            axis (int): Axis along which to aggregate. Use 1 for reactant
                atoms (columns) and 0 for product atoms (rows).

        Returns:
            np.ndarray: A copy of attn with symmetric atom slices replaced by
                their group noisy-OR combination. The original array is not
                modified.
        """
        identical_groups: List[Tuple[int, ...]] = list(
            {tuple(sorted([k] + v)) for k, v in symmetric_indices.items()}
        )

        result = attn.copy()
        new_val_mapping: Dict[int, np.ndarray] = {}
        for group in identical_groups:
            idx = list(group)
            if axis == 1:
                combined = 1.0 - np.prod(1.0 - attn[:, idx], axis=1)
                for i in idx:
                    new_val_mapping[i] = combined
            else:
                combined = 1.0 - np.prod(1.0 - attn[idx, :], axis=0)
                for i in idx:
                    new_val_mapping[i] = combined

        for i, val in new_val_mapping.items():
            if axis == 1:
                result[:, i] = val
            else:
                result[i, :] = val

        return result

    def _symmetry_aware_confidence(
        self,
        p2r: np.ndarray,
        r2p: np.ndarray,
        r_sym: Dict[int, List[int]],
        p_sym: Dict[int, List[int]],
        one_to_one_correspondence: bool = True,
    ) -> np.ndarray:
        """
        Compute symmetry-corrected confidence matrix from products-to-reactants attention.

        Uses only the products-to-reactants (p2r) attention matrix for
        confidence scoring, applying two types of symmetry correction:

        1. **Opposite-side sum** (via _apply_symmetric_attention): When a
           product atom could map to any of several symmetric reactant atoms,
           the attention is split across them. Summing recovers the total
           probability that the product atom maps to *some* member of the
           symmetric reactant group.

        2. **Same-side noisy-OR** (via _apply_noisy_or): When several
           symmetric product atoms each have an opinion about the same
           reactant atom, their confidences are combined using noisy-OR
           (``1 - prod(1 - p_i)``). This treats each symmetric product atom
           as an independent observer and correctly combines moderate
           confidences into a higher combined confidence.

        The r2p matrix is accepted for interface compatibility but not used.
        Empirically, including r2p in the confidence calculation dilutes
        accuracy because the reverse-direction noisy-OR introduces different
        values than the forward-direction sum, and r2p is more susceptible
        to noise on complex molecules.

        Args:
            p2r (np.ndarray): Products-to-reactants attention matrix of shape
                (n_product_atoms, n_reactant_atoms).
            r2p (np.ndarray): Reactants-to-products attention matrix of shape
                (n_product_atoms, n_reactant_atoms). Not used; kept for
                interface compatibility.
            r_sym (Dict[int, List[int]]): Symmetric atom indices for
                reactants.
            p_sym (Dict[int, List[int]]): Symmetric atom indices for
                products.
            one_to_one_correspondence (bool): Kept for interface
                compatibility; does not affect the result since only p2r is
                used regardless.

        Returns:
            np.ndarray: Symmetry-corrected confidence matrix of shape
                (n_product_atoms, n_reactant_atoms) with values in [0, 1].
        """
        p2r_corrected = self._apply_symmetric_attention(p2r, r_sym, axis=1)
        p2r_corrected = np.clip(p2r_corrected, 0.0, 1.0)
        p2r_corrected = self._apply_noisy_or(p2r_corrected, p_sym, axis=0)
        return p2r_corrected

    def assign_atom_maps(
        self,
        rxn_smiles: str,
        aligned_attn_scores: Tuple[np.ndarray, np.ndarray],
        one_to_one_correspondence: bool = True,
        reactants_atom_idx_to_orig_mapping: Optional[Dict[int, int]] = None,
        products_atom_idx_to_orig_mapping: Optional[Dict[int, int]] = None,
        consider_tautomer_symmetry: bool = True,
        consider_transform_symmetry: bool = True,
    ) -> Tuple[str, float, Dict[str, int]]:
        """
        Assign atom-to-atom map numbers to a reaction SMILES using a pre-computed
        attention matrix.

        Handles symmetric atoms in both reactants and products via
        ``_symmetry_aware_confidence``, which applies opposite-side sum to
        recover split probability mass and same-side noisy-OR to combine
        independent opinions from symmetric source atoms. This prevents
        artificially low confidence scores caused by equivalent atoms
        splitting probability mass.

        Greedy assignment uses the raw averaged attention (without symmetry
        correction) to select the best individual atom pair, while confidence
        scores are read from the symmetry-corrected matrix.

        Uses the scoring heuristic parameters (adjacent_atom_multiplier,
        identical_adjacent_atom_multiplier, used_atom_divisor) configured on
        the NeuralPostProcessor instance at construction time.

        Args:
            rxn_smiles (str): Unmapped reaction SMILES string of the form
                "reactants>>products".
            aligned_attn_scores (Tuple[np.ndarray, np.ndarray]): Tuple of attention matrices
                of shape (n_product_atoms, n_reactant_atoms).
            one_to_one_correspondence (bool): If True, enforces a one-to-one
                assignment using greedy selection of the global attention maximum.
                If False, assigns each product atom independently to its
                highest-attention reactant atom.
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

        reactants_symmetric_indices = self._get_symmetric_atom_indices(
            reactants_mols,
            consider_tautomer_symmetry=consider_tautomer_symmetry,
            consider_transform_symmetry=consider_transform_symmetry,
        )
        products_symmetric_indices = self._get_symmetric_atom_indices(
            products_mols,
            consider_tautomer_symmetry=consider_tautomer_symmetry,
            consider_transform_symmetry=consider_transform_symmetry,
        )

        orig_attn = self._symmetry_aware_confidence(
            orig_products_to_reactants_attn,
            orig_reactants_to_products_attn,
            reactants_symmetric_indices,
            products_symmetric_indices,
            one_to_one_correspondence=one_to_one_correspondence,
        )

        if one_to_one_correspondence:
            attn = (
                reactants_to_products_attn.copy() + products_to_reactants_attn.copy()
            ) / 2
        else:
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

                conf_val = orig_attn[row_highest_attn, col_highest_attn]

                if one_to_one_correspondence:
                    attn[row_highest_attn] = 0
                    attn[:, col_highest_attn] = 0
                else:
                    attn[row_highest_attn] = 0
                    attn[:, col_highest_attn] /= self._used_atom_divisor

                assignment_probs.append(conf_val)

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
                            self._adjacent_atom_multiplier
                            * self._identical_adjacent_atom_multiplier
                        )
                    else:
                        attn[product_atom_idx, reactant_atom_idx] *= (
                            self._adjacent_atom_multiplier
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

        oversubscribed_dict: Dict[str, int] = {}
        for reactant in reactants_mols:
            max_oversubscribed_count = 0
            for reactant_atom in reactant.GetAtoms():
                if not reactant_atom.HasProp("oversubscribed_count"):
                    continue
                oversubscribed_count = reactant_atom.GetIntProp("oversubscribed_count")
                max_oversubscribed_count = max(
                    max_oversubscribed_count, oversubscribed_count
                )
            if max_oversubscribed_count == 0:
                continue
            for atom in reactant.GetAtoms():
                atom.SetAtomMapNum(0)
            reactant_smiles = Chem.MolToSmiles(reactant)
            oversubscribed_dict[reactant_smiles] = (
                oversubscribed_dict.get(reactant_smiles, 0) + max_oversubscribed_count
            )

        return mapped_rxn_smiles, confidence, oversubscribed_dict

    def map_from_attention(
        self,
        rxn_smiles: str,
        attn: np.ndarray,
        tokens: List[str],
        one_to_one_correspondence: bool = True,
        reactants_atom_idx_to_orig_mapping: Optional[Dict[int, int]] = None,
        products_atom_idx_to_orig_mapping: Optional[Dict[int, int]] = None,
        consider_tautomer_symmetry: bool = True,
        consider_transform_symmetry: bool = True,
    ) -> Tuple[ReactionMapperResult, Optional[str]]:
        """
        Assign atom mappings from a pre-computed log-attention matrix and token list.

        Performs all post-inference processing: token validation, attention masking,
        cross-attention score alignment, non-atom row/column removal, and atom map
        assignment. When one_to_one_correspondence is False and oversubscribed reactant
        atoms are detected, the expanded reaction SMILES (with extra reactant copies)
        is returned as the second element for a downstream retry pass.

        Uses the scoring heuristic parameters and sequence_max_length configured
        on the NeuralPostProcessor instance at construction time.

        Args:
            rxn_smiles (str): An unmapped reaction SMILES string.
            attn (np.ndarray): Log-attention matrix of shape (seq_len, seq_len) as
                returned by _get_attention_matrices_batch or get_attention_matrix_for_head.
            tokens (List[str]): Token strings aligned to the attention matrix axes.
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

        if len(tokens) >= self._sequence_max_length:
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
            reactants_atom_idx_to_orig_mapping=reactants_atom_idx_to_orig_mapping,
            products_atom_idx_to_orig_mapping=products_atom_idx_to_orig_mapping,
            consider_tautomer_symmetry=consider_tautomer_symmetry,
            consider_transform_symmetry=consider_transform_symmetry,
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

    def strip_unmapped_reactant_fragments(
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
        consumed, remaining copies are treated as extra and are kept only if
        at least one of their atoms carries a non-zero atom map number.

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

    def compute_o2o_from_mcs_result(
        self,
        rxn_smiles: str,
        mcs_result: ReactionMapperResult,
    ) -> bool:
        """
        Determine ``one_to_one_correspondence`` from an MCS mapping result.

        Computes unmapped product atom islands from the MCS-mapped SMILES and
        calls :func:`determine_one_to_one_correspondence`.

        Args:
            rxn_smiles (str): The original unmapped reaction SMILES.
            mcs_result (ReactionMapperResult): MCS mapping result. If the
                mapping is empty, no islands are computed and the default
                o2o determination is applied.

        Returns:
            bool: The resolved ``one_to_one_correspondence`` flag.
        """
        islands: Dict[int, Set[int]] = {}
        if mcs_result.selected_mapping:
            try:
                islands = compute_unmapped_product_atom_islands(
                    mcs_result.selected_mapping.split(">>")[1]
                )
            except ValueError:
                islands = {}

        return determine_one_to_one_correspondence(rxn_smiles, islands)

    def create_worker_pool(self, num_processes: int) -> mp.Pool:
        """
        Create a reusable ``multiprocessing.Pool`` for parallel post-processing.

        The pool is initialized with worker processes that each hold a
        ``NeuralPostProcessor`` instance configured with the same scoring
        heuristics as this instance. The caller is responsible for closing
        the pool when done (e.g., via a ``with`` statement or explicit
        ``pool.close()`` / ``pool.join()``).

        Args:
            num_processes (int): Number of worker processes.

        Returns:
            mp.Pool: A multiprocessing pool ready for use with
                ``post_process_batch``.
        """
        return mp.Pool(
            processes=num_processes,
            initializer=_init_worker,
            initargs=(
                self._adjacent_atom_multiplier,
                self._identical_adjacent_atom_multiplier,
                self._used_atom_divisor,
                self._sequence_max_length,
                self._mapper_type,
            ),
        )

    def post_process_batch(
        self,
        tasks: List[_PostProcessTask],
        num_processes: int = 1,
        pool: Optional[mp.Pool] = None,
    ) -> List[Tuple[ReactionMapperResult, Optional[str]]]:
        """
        Post-process a batch of attention matrices into mapping results.

        When ``num_processes`` is 1, runs serially in-process using this
        instance. When ``num_processes > 1``, distributes work across a
        ``multiprocessing.Pool`` of worker processes, each with its own
        ``NeuralPostProcessor`` configured with the same scoring heuristics.
        Results are returned in the same order as ``tasks``.

        Args:
            tasks (List[_PostProcessTask]): One task per reaction, each
                containing (rxn_smiles, attn, tokens, one_to_one_correspondence,
                consider_tautomer_symmetry, consider_transform_symmetry).
            num_processes (int): Number of worker processes for parallel
                post-processing. When 1, runs serially in-process.
            pool (Optional[mp.Pool]): A pre-existing pool to reuse. When
                provided, the pool is used instead of creating a new one and
                ``num_processes`` is ignored. The caller is responsible for
                closing the pool.

        Returns:
            List[Tuple[ReactionMapperResult, Optional[str]]]: Mapping results
                and expanded SMILES, in the same order as ``tasks``.
        """
        if num_processes <= 1 and pool is None:
            return [
                self.map_from_attention(
                    rxn_smiles=rxn,
                    attn=attn,
                    tokens=tokens,
                    one_to_one_correspondence=o2o,
                    consider_tautomer_symmetry=taut_sym,
                    consider_transform_symmetry=transform_sym,
                )
                for rxn, attn, tokens, o2o, taut_sym, transform_sym in tasks
            ]

        if pool is not None:
            return list(pool.map(_post_process_one, tasks, chunksize=1))

        with mp.Pool(
            processes=num_processes,
            initializer=_init_worker,
            initargs=(
                self._adjacent_atom_multiplier,
                self._identical_adjacent_atom_multiplier,
                self._used_atom_divisor,
                self._sequence_max_length,
                self._mapper_type,
            ),
        ) as p:
            return list(p.map(_post_process_one, tasks, chunksize=1))
