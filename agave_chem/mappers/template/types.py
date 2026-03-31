from typing import Any, Dict, List, Optional, Set, TypedDict

from rdchiral import main as rdc
from rdkit import Chem


class ReactionData(TypedDict):
    products_mols: List[Chem.Mol]
    reactants_mols: List[Chem.Mol]
    rdc_products: Any
    tautomers_reactants: Dict[str, List[str]]
    fragment_count_reactants: Dict[str, int]
    unmapped_product_atom_islands: Dict[int, Set[int]]


class InitializedSmirksPattern(TypedDict):
    products_smarts: List[Chem.Mol]
    reactants_smarts: List[Chem.Mol]
    rdc_rxn: rdc.rdchiralReaction
    parent_smirks: str
    child_smirks: str


class AppliedSmirkData(TypedDict):
    outcome_unmapped_smiles: str
    outcome_mapped_smiles: str
    outcome_atom_map_indices: List[int]
    applied_smirk: InitializedSmirksPattern
    outcome_to_island_id: int | None
    num_smirks_applied: int


class SmirksPattern(TypedDict):
    name: str
    smirks: str
    superclass_id: Optional[int]
    class_id: Optional[int]
    subclass_id: Optional[int]
