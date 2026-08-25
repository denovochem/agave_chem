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
    """

    name: str
    smirks: str
    superclass_id: Optional[int] = None
    class_id: Optional[int] = None
    subclass_id: Optional[int] = None


class ReactionMapperResult(BaseModel):
    """
    Result of a single reaction mapping operation.

    Args:
        original_smiles (str): The original unmapped reaction SMILES.
        selected_mapping (str): The selected atom-mapped reaction SMILES, or empty string on failure.
        possible_mappings (Dict[str, List[str]]): Mapping from mapped SMILES to list of template names.
        mapping_type (str): The type of mapper that produced this result (e.g. "mcs", "template", "neural").
        mapping_score (Any): Optional score or scoring object for the selected mapping.
        additional_info (List[Dict[str, Any]]): Additional metadata about the mapping.
    """

    original_smiles: str = ""
    selected_mapping: str = ""
    possible_mappings: Dict[str, List[str]] = Field(default_factory=dict)
    mapping_type: str = ""
    mapping_score: Any = None
    additional_info: List[Dict[str, Any]] = Field(
        default_factory=_default_additional_info
    )


class AgaveChemMapperResult(BaseModel):
    """
    Aggregated result from running multiple mappers on a single reaction.

    Args:
        final_mapping (str): The final selected atom-mapped reaction SMILES.
        original_reaction (str): The original unmapped reaction SMILES.
        mapper_results (List[ReactionMapperResult]): Results from each individual mapper.
    """

    final_mapping: str = ""
    original_reaction: str = ""
    mapper_results: List[ReactionMapperResult] = Field(default_factory=list)
