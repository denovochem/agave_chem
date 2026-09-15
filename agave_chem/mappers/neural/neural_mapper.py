from importlib.resources import files
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, cast

import numpy as np
import torch
from transformers import AlbertForMaskedLM

from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper
from agave_chem.mappers.neural.constants import smiles_token_to_id_dict
from agave_chem.mappers.neural.model import (
    AlbertWithAttentionAlignment,
    SupervisedConfig,
)
from agave_chem.mappers.neural.post_processor import NeuralPostProcessor
from agave_chem.mappers.neural.tokenizer import CustomTokenizer
from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.types import ReactionInput, ReactionMapperResult
from agave_chem.utils.chem_utils import canonicalize_reaction_smiles


def load_neural_albert_model(
    checkpoint_dir: str,
    device: torch.device,
    supervised_config: SupervisedConfig | None = None,
) -> AlbertWithAttentionAlignment:
    """
    Load the supervised ALBERT model from a checkpoint directory.

    Args:
        checkpoint_dir (str): Path to the directory containing the base
            AlbertForMaskedLM weights and the supervised checkpoint file
            ``supervised_albert_model.pt``.
        device (torch.device): The device to load the model onto.
        supervised_config (SupervisedConfig | None): Configuration for the
            supervised model. If None, a default SupervisedConfig is used.

    Returns:
        AlbertWithAttentionAlignment: The loaded supervised model in eval mode.
    """
    checkpoint_dir = str(checkpoint_dir)
    base_model = AlbertForMaskedLM.from_pretrained(
        checkpoint_dir,
        attn_implementation="eager",
    )
    torch.nn.Module.to(base_model, device)

    if supervised_config is None:
        supervised_config = SupervisedConfig()

    wrapper = AlbertWithAttentionAlignment(
        base_model=base_model,
        supervised_config=supervised_config,
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
        supervised_config: SupervisedConfig | None = None,
        sequence_max_length: int = 1024,
        adjacent_atom_multiplier: float = 10,
        identical_adjacent_atom_multiplier: float = 10,
        used_atom_divisor: float = 10,
        num_processes: int = 1,
        inference_batch_size: int = 32,
    ):
        """
        Initialize the NeuralReactionMapper instance.

        Args:
            mapper_name (str): The name of the mapper.
            mapper_weight (float): The weight of the mapper.
            checkpoint_path (Optional[str]): The path to the checkpoint file.
            supervised_config (SupervisedConfig | None): Configuration for the
                supervised model. If None, a default SupervisedConfig is used.
            sequence_max_length (int): Maximum tokenization length for the model.
                Defaults to 1024.
            adjacent_atom_multiplier (float): Multiplier applied to attention
                scores of atoms neighboring an already-mapped pair.
            identical_adjacent_atom_multiplier (float): Additional multiplier
                applied when a neighboring pair shares the same atom encoding.
            used_atom_divisor (float): Divisor applied to attention scores of
                reactant atoms that are already mapped when
                one_to_one_correspondence is False. Lower values increase the
                likelihood of detecting oversubscription.
            num_processes (int): Number of worker processes for parallel CPU
                post-processing and MCS pre-processing. When 1 (default), both
                run serially. When > 1, MCS pre-processing is parallelized via
                ``ParallelMCSReactionMapper`` and CPU post-processing is
                parallelized via ``NeuralPostProcessor.post_process_batch``.
            inference_batch_size (int): Number of reactions per GPU forward pass.
                Can be overridden per-call via ``map_reactions``. Defaults to 32.
        """

        super().__init__("neural", mapper_name, mapper_weight)

        if not checkpoint_path:
            checkpoint_path = str(
                files("agave_chem.datafiles.models").joinpath("supervised_albert_model")
            )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._sequence_max_length = sequence_max_length
        self._supervised_config = supervised_config or SupervisedConfig()
        self._adjacent_atom_multiplier = adjacent_atom_multiplier
        self._identical_adjacent_atom_multiplier = identical_adjacent_atom_multiplier
        self._used_atom_divisor = used_atom_divisor

        self._model = load_neural_albert_model(
            checkpoint_dir=checkpoint_path,
            device=self._device,
            supervised_config=self._supervised_config,
        )

        self._tokenizer = CustomTokenizer(smiles_token_to_id_dict)
        self._mcs_mapper: Optional[MCSReactionMapper] = None
        self._num_processes = num_processes
        self._inference_batch_size = inference_batch_size
        self._post_processor = NeuralPostProcessor(
            adjacent_atom_multiplier=adjacent_atom_multiplier,
            identical_adjacent_atom_multiplier=identical_adjacent_atom_multiplier,
            used_atom_divisor=used_atom_divisor,
            sequence_max_length=sequence_max_length,
            mapper_type=self._mapper_type,
        )

    def _get_attention_matrices_batch(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> List[Tuple[np.ndarray, List[str]]]:
        """
        Run batched neural network inference and return log-attention matrices for a
        list of reaction SMILES strings.

        Tokenizes all inputs together in a single batch with dynamic padding (padded
        to the longest sequence in the batch rather than max_length), executes one
        forward pass, then trims each result to its non-padding length before applying
        the logarithm.

        Args:
            texts (List[str]): Reaction SMILES strings to encode. Must be non-empty.
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
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)
        token_type_ids = torch.zeros_like(input_ids)

        with torch.no_grad():
            attn_probs = self._model.predict_attention_probs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )  # (B, S, S)
            attn_batch = attn_probs.detach().cpu()  # (B, S, S)

        results: List[Tuple[np.ndarray, List[str]]] = []
        for i in range(len(texts)):
            real_len = int(enc["attention_mask"][i].sum().item())
            attn_i = attn_batch[i, :real_len, :real_len]
            tokens_i = self._tokenizer.convert_ids_to_tokens(
                enc["input_ids"][i].tolist()
            )[:real_len]
            results.append((torch.log(attn_i).numpy(), tokens_i))

        return results

    def _resolve_one_to_one_correspondence(
        self,
        rxn_smiles: str,
        one_to_one_correspondence: Union[bool, Literal["auto"]],
        reaction_input: Optional[ReactionInput] = None,
    ) -> bool:
        """
        Resolve the ``one_to_one_correspondence`` flag for a single reaction.

        When the flag is ``"auto"``, the method uses pre-computed data from
        ``reaction_input`` if available, or lazily instantiates an
        ``MCSReactionMapper`` to compute the MCS mapping and unmapped atom
        islands, then delegates to
        :meth:`NeuralPostProcessor.compute_o2o_from_mcs_result`.

        Args:
            rxn_smiles (str): The reaction SMILES to evaluate.
            one_to_one_correspondence (Union[bool, Literal["auto"]]): The
                correspondence mode.  ``True`` or ``False`` are passed through
                unchanged; ``"auto"`` triggers detection.
            reaction_input (Optional[ReactionInput]): Pre-computed reaction
                data including MCS results and the determined flag.  If
                provided and the mode is ``"auto"``, the
                ``one_to_one_correspondence`` field is used directly.

        Returns:
            bool: The resolved ``one_to_one_correspondence`` flag.
        """
        if one_to_one_correspondence != "auto":
            return one_to_one_correspondence

        if reaction_input is not None:
            return reaction_input.one_to_one_correspondence

        if self._mcs_mapper is None:
            self._mcs_mapper = MCSReactionMapper(
                mapper_name="mcs_auto",
                mapper_weight=0,
            )

        mcs_result = self._mcs_mapper.map_reaction(rxn_smiles)
        return self._post_processor.compute_o2o_from_mcs_result(rxn_smiles, mcs_result)

    def map_reaction(
        self,
        rxn_smiles: Union[str, ReactionInput],
        one_to_one_correspondence: Union[bool, Literal["auto"]] = "auto",
        consider_tautomer_symmetry: bool = True,
        consider_transform_symmetry: bool = True,
    ) -> ReactionMapperResult:
        """
        Map a single reaction SMILES string using the neural mapper.

        Convenience wrapper around map_reactions for single-reaction use.

        Args:
            rxn_smiles (Union[str, ReactionInput]): A reaction SMILES string or
                a ``ReactionInput`` with pre-computed data.
            one_to_one_correspondence (Union[bool, Literal["auto"]]): If True,
                enforces greedy one-to-one assignment.  If False, allows
                oversubscription for reaction balancing.  If ``"auto"`` (the
                default), the flag is determined per-reaction using atom-count
                imbalance and MCS-based island detection.

        Returns:
            ReactionMapperResult: Mapping result. On failure returns a result with an
                empty selected_mapping.
        """
        return self.map_reactions(
            cast(Union[List[str], List[ReactionInput]], [rxn_smiles]),
            one_to_one_correspondence=one_to_one_correspondence,
            consider_tautomer_symmetry=consider_tautomer_symmetry,
            consider_transform_symmetry=consider_transform_symmetry,
        )[0]

    def _run_inference_and_post_process(
        self,
        smiles_list: List[str],
        o2o_flags: List[bool],
        inference_batch_size: int,
        num_processes: int,
        consider_tautomer_symmetry: bool,
        consider_transform_symmetry: bool,
    ) -> Tuple[List[ReactionMapperResult], List[Optional[str]]]:
        """
        Run batched GPU inference and parallel CPU post-processing.

        GPU inference processes ``inference_batch_size`` reactions per forward
        pass. After each GPU batch, CPU post-processing (attention alignment,
        atom assignment) is applied to the batch's results. When
        ``num_processes > 1``, post-processing is parallelized across a
        ``multiprocessing.Pool`` of worker processes, each with its own
        ``NeuralPostProcessor`` instance.

        Pre-tokenizes inputs, sorts by sequence length to minimize padding
        waste, batches through the model, and calls
        ``NeuralPostProcessor.post_process_batch`` for each batch. Results and
        expanded SMILES are returned in the same order as the input.

        Args:
            smiles_list (List[str]): Reaction SMILES strings to process.
            o2o_flags (List[bool]): Per-reaction ``one_to_one_correspondence``
                flags, aligned with ``smiles_list``.
            inference_batch_size (int): Number of reactions per GPU forward pass.
            num_processes (int): Number of worker processes for CPU
                post-processing. When 1, runs serially in-process.
            consider_tautomer_symmetry (bool): If True, consider tautomer
                symmetry during post-processing.
            consider_transform_symmetry (bool): If True, apply functional
                group normalization transforms during post-processing.

        Returns:
            Tuple[List[ReactionMapperResult], List[Optional[str]]]:
                - List of mapping results, one per input, in the same order.
                - List of expanded reaction SMILES (or None for reactions
                  without oversubscription), in the same order.
        """
        n = len(smiles_list)
        results: List[ReactionMapperResult] = [
            ReactionMapperResult(
                original_smiles="",
                selected_mapping="",
                possible_mappings={},
                mapping_type=self._mapper_type,
                mapping_score=None,
                additional_info=[{}],
            )
            for _ in range(n)
        ]
        expanded_list: List[Optional[str]] = [None] * n

        if n == 0:
            return results, expanded_list

        # Pre-tokenize (without padding) to get sequence lengths for sorting.
        # This is cheap (regex tokenization) compared to the model forward pass.
        pre_enc = self._tokenizer(
            smiles_list,
            max_length=self._sequence_max_length,
            truncation=True,
            padding=False,
        )
        lengths = [len(ids) for ids in pre_enc["input_ids"]]
        sorted_order = sorted(range(n), key=lambda i: lengths[i])

        for batch_start in range(0, n, inference_batch_size):
            batch_indices = sorted_order[
                batch_start : batch_start + inference_batch_size
            ]
            batch_smiles = [smiles_list[i] for i in batch_indices]
            attn_tokens_list = self._get_attention_matrices_batch(
                texts=batch_smiles,
                max_length=self._sequence_max_length,
            )

            tasks = [
                (
                    smiles_list[orig_idx],
                    attn_tokens_list[local_idx][0],
                    attn_tokens_list[local_idx][1],
                    o2o_flags[orig_idx],
                    consider_tautomer_symmetry,
                    consider_transform_symmetry,
                )
                for local_idx, orig_idx in enumerate(batch_indices)
            ]

            batch_results = self._post_processor.post_process_batch(
                tasks=tasks,
                num_processes=num_processes,
            )

            for local_idx, orig_idx in enumerate(batch_indices):
                result, expanded_rxn = batch_results[local_idx]
                results[orig_idx] = result
                if expanded_rxn is not None:
                    expanded_list[orig_idx] = canonicalize_reaction_smiles(expanded_rxn)

        return results, expanded_list

    def map_reactions(
        self,
        reaction_list: Union[List[str], List[ReactionInput]],
        one_to_one_correspondence: Union[bool, Literal["auto"]] = "auto",
        inference_batch_size: Optional[int] = None,
        num_processes: Optional[int] = None,
        consider_tautomer_symmetry: bool = True,
        consider_transform_symmetry: bool = True,
    ) -> List[ReactionMapperResult]:
        """
        Map a list of reaction SMILES strings using batched neural network inference.

        GPU inference and CPU post-processing are decoupled: the GPU processes
        ``inference_batch_size`` reactions per forward pass, then CPU
        post-processing (attention alignment, atom assignment) runs on each
        batch's results. When ``num_processes > 1``, CPU post-processing is
        parallelized across worker processes, each with its own
        ``NeuralPostProcessor`` instance.

        Reactions are pre-tokenized and sorted by sequence length before
        batching to minimize padding waste. Each batch's attention matrices
        are consumed for atom mapping immediately after inference and
        discarded before the next batch, avoiding the need to hold all
        matrices in memory simultaneously.

        Uses the scoring heuristics and sequence_max_length configured
        on the NeuralReactionMapper instance at construction time.

        Args:
            reaction_list (Union[List[str], List[ReactionInput]]): A list of
                unmapped reaction SMILES strings or ``ReactionInput`` objects.
            one_to_one_correspondence (Union[bool, Literal["auto"]]): If True,
                enforces greedy one-to-one assignment for all reactions.  If
                False, allows oversubscription for all reactions.  If
                ``"auto"`` (the default), the flag is determined per-reaction
                using atom-count imbalance and MCS-based island detection.
                When ``ReactionInput`` objects are provided, their
                pre-computed ``one_to_one_correspondence`` field is used.
            inference_batch_size (Optional[int]): Number of reactions per GPU
                forward pass. If None, uses the value set at construction time
                (default 32).
            num_processes (Optional[int]): Number of worker processes for
                parallel CPU post-processing and MCS pre-processing. If None,
                uses the value set at construction time. When 1, post-processing
                runs serially in-process.
            consider_tautomer_symmetry (bool): If True, atoms that interconvert via
                tautomerism are treated as symmetrically equivalent during
                post-processing. Disabling this reduces CPU-bound tautomer
                enumeration at the cost of less accurate symmetry handling.
            consider_transform_symmetry (bool): If True, apply functional group
                normalization transforms before computing symmetry classes.
                Disabling this skips the normalization step for speed.

        Note:
            When ``one_to_one_correspondence`` is ``"auto"`` and raw SMILES
            strings (not ``ReactionInput``) are provided, MCS pre-processing is
            run to determine the o2o flag per reaction. When ``num_processes``
            is greater than 1, both MCS pre-processing and CPU post-processing
            are parallelized across worker processes.

        Returns:
            List[ReactionMapperResult]: A list of mapping results, one per input
                reaction. Failed mappings return a result with an empty
                selected_mapping. When one_to_one_correspondence is False and
                oversubscribed reactant atoms are detected, a second batched
                inference pass is run on expanded reactions (extra reactant copies
                appended) with one_to_one_correspondence=True; successful retry
                results replace the first-pass results.
        """
        eff_batch_size = inference_batch_size or self._inference_batch_size
        eff_num_processes = (
            num_processes if num_processes is not None else self._num_processes
        )

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

        # Preprocess: validate and resolve one_to_one_correspondence per reaction.
        # Reactions needing MCS (o2o=="auto", no ReactionInput) are collected
        # and batched through MCS in one pass rather than one at a time.
        prepared: List[Optional[Tuple[str, bool]]] = []
        mcs_needed: List[Tuple[int, str]] = []  # (prepared_index, rxn_smiles)

        for item in reaction_list:
            reaction_input: Optional[ReactionInput] = None
            if isinstance(item, ReactionInput):
                reaction_input = item
                rxn_smiles = item.stripped_smiles
            else:
                rxn_smiles = item

            rxn_smiles = canonicalize_reaction_smiles(rxn_smiles)
            if not self._reaction_smiles_valid(rxn_smiles):
                prepared.append(None)
                continue

            if one_to_one_correspondence != "auto":
                prepared.append((rxn_smiles, one_to_one_correspondence))
            elif reaction_input is not None:
                prepared.append((rxn_smiles, reaction_input.one_to_one_correspondence))
            else:
                mcs_needed.append((len(prepared), rxn_smiles))
                prepared.append(None)

        # Batch MCS resolution for reactions that need it
        if mcs_needed:
            smiles_for_mcs = [s for _, s in mcs_needed]
            if eff_num_processes > 1:
                from agave_chem.mappers.mcs.parallel_mcs_mapper import (
                    ParallelMCSReactionMapper,
                )

                parallel_mcs = ParallelMCSReactionMapper(
                    "mcs_neural_auto", workers=eff_num_processes
                )
                mcs_results = parallel_mcs.map_reactions(smiles_for_mcs)
            else:
                if self._mcs_mapper is None:
                    self._mcs_mapper = MCSReactionMapper(
                        mapper_name="mcs_auto",
                        mapper_weight=0,
                    )
                mcs_results = [self._mcs_mapper.map_reaction(s) for s in smiles_for_mcs]

            assert len(mcs_results) == len(mcs_needed), (
                f"MCS returned {len(mcs_results)} results for "
                f"{len(mcs_needed)} reactions"
            )
            for (prep_idx, rxn_smiles), mcs_result in zip(mcs_needed, mcs_results):
                resolved_o2o = self._post_processor.compute_o2o_from_mcs_result(
                    rxn_smiles, mcs_result
                )
                prepared[prep_idx] = (rxn_smiles, resolved_o2o)

        valid_indices = [i for i, p in enumerate(prepared) if p is not None]
        valid_smiles = [cast(Tuple[str, bool], prepared[i])[0] for i in valid_indices]
        valid_o2o = [cast(Tuple[str, bool], prepared[i])[1] for i in valid_indices]

        if not valid_smiles:
            return results

        # First pass: batched inference + post-processing
        pass_results, expanded_list = self._run_inference_and_post_process(
            smiles_list=valid_smiles,
            o2o_flags=valid_o2o,
            inference_batch_size=eff_batch_size,
            num_processes=eff_num_processes,
            consider_tautomer_symmetry=consider_tautomer_symmetry,
            consider_transform_symmetry=consider_transform_symmetry,
        )

        # Place first-pass results and collect oversubscription cases
        oversubscribed_cases: List[Tuple[int, str, str]] = []
        for local_idx, orig_idx in enumerate(valid_indices):
            results[orig_idx] = pass_results[local_idx]
            expanded = expanded_list[local_idx]
            if expanded is not None:
                oversubscribed_cases.append(
                    (orig_idx, valid_smiles[local_idx], expanded)
                )

        if not oversubscribed_cases:
            return results

        # Second pass: retry oversubscribed reactions with one_to_one_correspondence=True
        expanded_smiles = [expanded for _, _, expanded in oversubscribed_cases]
        retry_o2o = [True] * len(expanded_smiles)

        retry_results, _ = self._run_inference_and_post_process(
            smiles_list=expanded_smiles,
            o2o_flags=retry_o2o,
            inference_batch_size=eff_batch_size,
            num_processes=eff_num_processes,
            consider_tautomer_symmetry=consider_tautomer_symmetry,
            consider_transform_symmetry=consider_transform_symmetry,
        )

        # Merge retry results back, stripping unused reactant fragments
        for i, (orig_idx, orig_rxn_smiles, _) in enumerate(oversubscribed_cases):
            retry_result = retry_results[i]
            if retry_result.selected_mapping:
                retry_result.original_smiles = orig_rxn_smiles
                retry_result.selected_mapping = (
                    self._post_processor.strip_unmapped_reactant_fragments(
                        retry_result.selected_mapping,
                        orig_rxn_smiles,
                    )
                )
                results[orig_idx] = retry_result

        return results
