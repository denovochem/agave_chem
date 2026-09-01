from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

from pydantic import BaseModel, Field
from rdchiral import main as rdc
from rdkit import Chem, DataStructs


def _default_additional_info() -> List[Dict[str, Any]]:
    return [{}]


class ReactionData(TypedDict):
    products_mols: List[Chem.Mol]
    reactants_mols: List[Chem.Mol]
    rdc_products: Any
    tautomers_reactants: Dict[str, List[str]]
    fragment_count_reactants: Dict[str, int]
    unmapped_product_atom_islands: Dict[int, Set[int]]
    product_mol_fps: List[DataStructs.ExplicitBitVect]
    reactant_mol_fps: List[DataStructs.ExplicitBitVect]


class InitializedSmirksPattern(TypedDict):
    name: str
    superclass_id: str
    class_id: str
    subclass_id: str
    subsubclass_id: str
    class_str: str
    products_smarts: List[Chem.Mol]
    reactants_smarts: List[Chem.Mol]
    products_fps: List[DataStructs.ExplicitBitVect]
    reactants_fps: List[DataStructs.ExplicitBitVect]
    rdc_rxn: rdc.rdchiralReaction
    parent_smirks: str
    child_smirks: str
    template_name: str
    priority: Tuple[int, int]
    rxno_classification: List[Dict[str, str]]


class AppliedSmirkData(TypedDict):
    outcome_unmapped_smiles: str
    outcome_mapped_smiles: str
    outcome_atom_map_indices: List[int]
    applied_smirk: InitializedSmirksPattern
    outcome_to_island_id: int | None
    num_smirks_applied: int


class SmirksPattern(BaseModel):
    """
    Represents a SMIRKS pattern with optional classification metadata.

    Args:
        name (str): Human-readable name for the SMIRKS pattern.
        smirks (str): The SMIRKS string defining the reaction template.
        superclass_id (Optional[int]): Optional superclass identifier.
        class_id (Optional[int]): Optional class identifier.
        subclass_id (Optional[int]): Optional subclass identifier.
        subsubclass_id (Optional[int]): Optional sub-subclass identifier.
        rxno_classification (List[Dict[str, str]]): Optional list of RXNO
            classification entries, each containing an ``rxno_id`` key.
    """

    name: str
    smirks: str
    superclass_id: Optional[int] = None
    class_id: Optional[int] = None
    subclass_id: Optional[int] = None
    subsubclass_id: Optional[int] = None
    rxno_classification: List[Dict[str, str]] = Field(default_factory=list)


class ReactionMapperResult(BaseModel):
    """
    Result of a single reaction mapping operation.

    Args:
        original_smiles (str): The original unmapped reaction SMILES.
        selected_mapping (str): The selected atom-mapped reaction SMILES, or empty string on failure.
        possible_mappings (Dict[str, List[str]]): Mapping from mapped SMILES to list of template names.
        ranked_mappings (List[str]): Ordered list of mapped SMILES strings in
            preference order (best first).  Populated by the template mapper
            when multiple valid mappings exist; contains a single element when
            only one mapping is found; empty for non-template mappers or on
            failure.
        mapping_type (str): The type of mapper that produced this result (e.g. "mcs", "template", "neural").
        mapping_score (Any): Optional score or scoring object for the selected mapping.
        additional_info (List[Dict[str, Any]]): Additional metadata about the mapping.
        classification_info (Dict[str, List[Dict[str, Any]]]): Per-mapping reaction
            classification metadata, keyed by mapped SMILES string (same keys as
            ``possible_mappings``).  Each value is a list of dicts, one per matching
            template, containing ``template_name``, ``class_str``, ``class_id``,
            ``subclass_id``, ``subsubclass_id``, ``superclass_id``,
            ``superclass_name``, ``superclass_description``, ``class_name``,
            ``class_description``, ``subclass_name``, ``subclass_description``,
            ``subsubclass_name``, ``subsubclass_description``, and
            ``rxno_classification`` (a list of RXNO classification dicts, each
            containing ``rxno_id``, ``rxno_label``, and ``rxno_definition``).
            Only populated by the template mapper; other mappers leave this empty.
    """

    original_smiles: str = ""
    selected_mapping: str = ""
    possible_mappings: Dict[str, List[str]] = Field(default_factory=dict)
    ranked_mappings: List[str] = Field(default_factory=list)
    mapping_type: str = ""
    mapping_score: Any = None
    additional_info: List[Dict[str, Any]] = Field(
        default_factory=_default_additional_info
    )
    classification_info: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


class AgaveChemMapperResult(BaseModel):
    """
    Aggregated result from running multiple mappers on a single reaction.

    Args:
        final_mapping (str): The final selected atom-mapped reaction SMILES.
        original_reaction (str): The original unmapped reaction SMILES.
        mapper_results (List[ReactionMapperResult]): Results from each individual
            mapper.  Only populated when the caller passes
            ``return_detailed_mapper_info=True`` to ``map_reactions`` or
            ``map_reactions_using_mappers``; otherwise left empty to reduce
            result size.
        confidence (Optional[float]): Confidence score from the neural mapper
            for the ``final_mapping``, computed as the product of per-atom
            assignment probabilities.  ``None`` when no neural mapper is used
            or the neural mapper fails to produce a mapping.
        ranked_mappings (List[str]): Ordered list of mapped SMILES strings in
            preference order (best first), propagated from the template
            mapper's ``ReactionMapperResult.ranked_mappings``.  Empty when no
            template mapper is used or the template mapper produces no
            mappings.
        class_str (str): Pipe-delimited string of unique classification paths
            (e.g. ``"1.1.1|2.5.1|2.5.3"``) for all templates matching the
            ``final_mapping``.  Empty when no template mapper is used or no
            template matches ``final_mapping``.
        rxno_classifications (str): Pipe-delimited string of unique RXNO IDs
            (e.g. ``"RXNO:0000335|RXNO:0000357"``) for all templates matching
            the ``final_mapping``.  Empty when no template mapper is used or no
            template matches ``final_mapping``.
        classification_info (Dict[str, List[Dict[str, Any]]]): Per-mapping
            classification metadata, keyed by mapped SMILES string.  Each value
            is a list of dicts (one per matching template) containing
            ``template_name``, ``class_str``, ``class_id``, ``subclass_id``,
            ``subsubclass_id``, ``superclass_id``, ``superclass_name``,
            ``superclass_description``, ``class_name``, ``class_description``,
            ``subclass_name``, ``subclass_description``, ``subsubclass_name``,
            ``subsubclass_description``, and ``rxno_classification``
            (a list of RXNO classification dicts with ``rxno_id``,
            ``rxno_label``, and ``rxno_definition``).  Always populated when a
            template mapper matches ``final_mapping``; empty otherwise.
    """

    final_mapping: str = ""
    original_reaction: str = ""
    mapper_results: List[ReactionMapperResult] = Field(default_factory=list)
    confidence: Optional[float] = None
    ranked_mappings: List[str] = Field(default_factory=list)
    class_str: str = ""
    rxno_classifications: str = ""
    classification_info: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
