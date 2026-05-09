import json
from collections import deque
from importlib.resources import files
from typing import Dict, List, Optional, Set, Tuple

from rdchiral import main as rdc
from rdkit import Chem, DataStructs
from rdkit.Chem import PatternFingerprint
from rdkit.Chem.MolStandardize import rdMolStandardize

from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper
from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.types import (
    AppliedSmirkData,
    InitializedSmirksPattern,
    ReactionData,
    ReactionMapperResult,
    SmirksPattern,
)
from agave_chem.utils.chem_utils import (
    canonicalize_atom_mapping,
    canonicalize_reaction_smiles,
    canonicalize_smiles,
)
from agave_chem.utils.logging_config import logger


class TemplateReactionMapper(ReactionMapper):
    """
    Expert template reaction classification and atom-mapping
    """

    def __init__(
        self,
        mapper_name: str,
        mapper_weight: float = 3,
        custom_smirks_patterns: List[SmirksPattern] | None = None,
        use_default_smirks_patterns: bool = True,
        max_transforms: int = 1000,
        max_tautomers: int = 1000,
        use_mcs_mapping: bool = True,
    ):
        """
        Initialize the TemplateMapper instance.

        Args:
            custom_smirks_patterns (List[Dict]): A list of dictionaries containing
                custom SMIRKS patterns. Each dictionary should have a 'name' key,
                a 'smirks' key, and a 'superclass_id' key.
            use_default_smirks_patterns (bool): Whether to use the default SMIRKS
                patterns.
        """

        super().__init__("template", mapper_name, mapper_weight)

        if custom_smirks_patterns is not None:
            if not isinstance(custom_smirks_patterns, list):
                raise TypeError(
                    "Invalid input: custom_smirks_patterns must be a list of dictionaries."
                )
            for pattern in custom_smirks_patterns:
                if set(pattern.keys()) != set(["name", "smirks", "superclass_id"]):
                    raise TypeError(
                        "Invalid input: each dictionary in custom_smirks_patterns must have 'name', 'smirks', and 'superclass_id' keys."
                    )
                for key, value in pattern.items():
                    if key == "superclass_id":
                        if value is not None and not isinstance(value, int):
                            raise TypeError(
                                "Invalid input: 'superclass_id' value must be an integer or None."
                            )
                    else:
                        if not isinstance(value, str):
                            raise TypeError(
                                "Invalid input: 'name' and 'smirks' values must be strings."
                            )

        self._custom_smirks_patterns = custom_smirks_patterns
        self._use_default_smirks_patterns = use_default_smirks_patterns

        smirks_patterns_file = files("agave_chem.datafiles.smirks_patterns").joinpath(
            "smirks_patterns_with_children.json"
        )
        with smirks_patterns_file.open("r") as f:
            self._uninitialized_smirks_patterns = json.load(f)
        self._initialized_smirks_patterns: Optional[List[InitializedSmirksPattern]] = (
            None
        )

        self._tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
        self._tautomer_enumerator.SetMaxTransforms(max_transforms)
        self._tautomer_enumerator.SetMaxTautomers(max_tautomers)

        self._mcs_mapper = None
        if use_mcs_mapping:
            self._mcs_mapper = MCSReactionMapper(
                mapper_name="mcs_for_template", mapper_weight=1
            )

    def _initialize_smirks_patterns(self) -> None:
        """Initialize SMIRKS patterns."""
        if self._initialized_smirks_patterns is not None:
            return

        if self._use_default_smirks_patterns and self._custom_smirks_patterns is None:
            smirks_patterns = self._uninitialized_smirks_patterns
        elif self._custom_smirks_patterns and not self._use_default_smirks_patterns:
            smirks_patterns = self._custom_smirks_patterns
        elif self._custom_smirks_patterns and self._use_default_smirks_patterns:
            smirks_patterns = (
                self._custom_smirks_patterns + self._uninitialized_smirks_patterns
            )
        else:
            raise TypeError(
                "Attempting to initialize AgaveChem with no SMIRKS patterns"
            )

        initialized_smirks_patterns: List[InitializedSmirksPattern] = []
        for pattern in smirks_patterns:
            pattern_priority = pattern.get(
                "priority", {"priority_class": None, "priority": None}
            )

            pattern_priority_tuple = (
                pattern_priority.get("priority_class", 0),
                pattern_priority.get("priority", 0),
            )

            if None in pattern_priority_tuple:
                pattern_priority_tuple = (0, 0)

            for child_pattern in pattern.get("child_smirks", []):
                reactants_smarts, products_smarts, rdc_rxn = (
                    self._initialize_template_data_from_child_patterns(child_pattern)
                )
                if (
                    reactants_smarts is None
                    or products_smarts is None
                    or rdc_rxn is None
                ):
                    continue

                initialized_smirks_patterns.append(
                    InitializedSmirksPattern(
                        name=str(pattern.get("name", "")),
                        superclass_id=str(pattern.get("superclass_id", "")),
                        class_id=str(pattern.get("class_id", "")),
                        subclass_id=str(pattern.get("subclass_id", "")),
                        class_str=str(pattern.get("class_str", "")),
                        products_smarts=products_smarts,
                        reactants_smarts=reactants_smarts,
                        products_fps=[
                            PatternFingerprint(frag) for frag in products_smarts
                        ],
                        reactants_fps=[
                            PatternFingerprint(frag) for frag in reactants_smarts
                        ],
                        rdc_rxn=rdc_rxn,
                        parent_smirks=str(pattern.get("smirks", "")),
                        child_smirks=str(child_pattern),
                        template_name=str(pattern.get("name", "")),
                        priority=pattern_priority_tuple,
                    )
                )

        self._initialized_smirks_patterns = initialized_smirks_patterns

        return

    def _initialize_template_data_from_child_patterns(
        self,
        child_smirks: str,
    ) -> Tuple[
        Optional[List[Chem.Mol]],
        Optional[List[Chem.Mol]],
        Optional[rdc.rdchiralReaction],
    ]:
        """
        Initialize template data from child SMIRKS pattern.

        Args:
            child_smirks (str): Child SMIRKS pattern.

        Returns:
            Tuple[Optional[List[Chem.Mol]], Optional[List[Chem.Mol]], Optional[rdc.rdchiralReaction]]:
            Tuple of reactants SMARTS, products SMARTS, and rdchiral reaction.
        """
        products_smarts = [
            Chem.MolFromSmarts(smarts)
            for smarts in child_smirks.split(">>")[0].split(".")
        ]

        if None in products_smarts:
            return None, None, None

        reactants_smarts = [
            Chem.MolFromSmarts(smarts)
            for smarts in child_smirks.split(">>")[1].split(".")
        ]

        if None in reactants_smarts:
            return None, None, None

        try:
            rdc_rxn = rdc.rdchiralReaction(child_smirks)
        except Exception as e:
            logger.warning(f"Error converting smirks to rdchiral reaction: {e}")
            return None, None, None

        return reactants_smarts, products_smarts, rdc_rxn

    def _prepare_reaction_data(
        self,
        reactants_str: str,
        products_str: str,
        unmapped_product_atom_islands: Optional[Dict[int, Set[int]]] = None,
    ) -> ReactionData:
        """
        Prepare reaction mapping inputs from reactant and product SMILES strings.

        Args:
            reactants_str (str): Reactants SMILES string.
            products_str (str): Products SMILES string.
            unmapped_product_atom_islands (Optional[Dict[int, Set[int]]]): Product atom-island SMILES
                strings that are intentionally left unmapped. Defaults to None.

        Returns:
            ReactionData: Mapping input data containing RDKit molecule objects for
            reactants/products, an RDChiral reactants object derived from
            `products_str`, a tautomer SMILES dictionary, a fragment count dictionary,
            and the normalized `unmapped_product_atom_islands` list.
        """

        if unmapped_product_atom_islands is None:
            unmapped_product_atom_islands = {}

        product_mols = [
            Chem.MolFromSmiles(product_str) for product_str in products_str.split(".")
        ]
        reactant_mols = [
            Chem.MolFromSmiles(reactant_str)
            for reactant_str in reactants_str.split(".")
        ]

        return ReactionData(
            products_mols=product_mols,
            reactants_mols=reactant_mols,
            rdc_products=rdc.rdchiralReactants(products_str),
            tautomers_reactants=self._enumerate_tautomer_smiles(reactants_str),
            fragment_count_reactants=self._get_fragment_count_dict(reactants_str),
            unmapped_product_atom_islands=unmapped_product_atom_islands,
            product_mol_fps=[PatternFingerprint(mol) for mol in product_mols],
            reactant_mol_fps=[PatternFingerprint(mol) for mol in reactant_mols],
        )

    def _enumerate_tautomer_smiles(self, smiles: str) -> Dict[str, List[str]]:
        """
        Enumerate tautomer SMILES strings for a given SMILES string.

        Args:
            smiles (str): A SMILES string representing one or more molecular fragments.

        Returns:
            Dict[str, List[str]]: A dictionary where keys are the fragments of the input
                SMILES string and values are lists of the enumerated tautomer SMILES strings
                for each fragment.

        """
        enumerated_smiles_dict: Dict[str, List[str]] = {}
        for fragment_str in smiles.split("."):
            mol = Chem.MolFromSmiles(fragment_str)

            if mol is None:
                enumerated_smiles_dict[fragment_str] = []
                continue

            enumerated_fragment_mols = list(self._tautomer_enumerator.Enumerate(mol))
            enumerated_fragment_smiles = [
                Chem.MolToSmiles(frag_mol) for frag_mol in enumerated_fragment_mols
            ]
            enumerated_fragment_smiles.append(fragment_str)
            canonicalized_enumerated_fragment_smiles = [
                canonicalize_smiles(frag_smiles, canonicalize_tautomer=False)
                for frag_smiles in enumerated_fragment_smiles
                if frag_smiles
            ]
            enumerated_smiles_dict[fragment_str] = list(
                set(canonicalized_enumerated_fragment_smiles)
            )
        return enumerated_smiles_dict

    def _get_fragment_count_dict(self, smiles: str) -> Dict[str, int]:
        """
        Build a fragment-count mapping from a dot-delimited SMILES string.

        Args:
            smiles (str): A SMILES string where disconnected fragments are delimited by `.`.

        Returns:
            Dict[str, int]: A mapping of each fragment string to the number of times it
            appears in `smiles`.
        """
        fragment_count_dict = {}
        for fragment_str in smiles.split("."):
            canonical_fragment_str = Chem.MolToSmiles(Chem.MolFromSmiles(fragment_str))
            if canonical_fragment_str is None:
                continue
            if canonical_fragment_str not in fragment_count_dict:
                fragment_count_dict[canonical_fragment_str] = 1
            else:
                fragment_count_dict[canonical_fragment_str] += 1

        return fragment_count_dict

    def _apply_templates_and_collect_outcomes(
        self, reaction_smiles_data: ReactionData, apply_multiple_smirks: bool = False
    ) -> Dict[str, List[InitializedSmirksPattern]]:
        """
        Apply template SMIRKS patterns to a reaction and collect mapped outcomes.

        Args:
            reaction_smiles_data (ReactionData): Reaction data containing reactants,
                products, and precomputed helper structures used for template
                application.
            apply_multiple_smirks (bool): Whether to apply multiple SMIRKS patterns to the same reaction.

        Returns:
            Dict[str, List[InitializedSmirksPattern]]: A mapping from mapped outcome SMILES to a list of
            applied SMIRKS patterns.
        """
        mapped_outcomes_smirks_dict: Dict[str, List[InitializedSmirksPattern]] = {}

        atom_mapped_product = self._generate_mapped_product_smiles(reaction_smiles_data)
        outcomes_and_applied_smirks = self._apply_templates(
            reaction_smiles_data, apply_multiple_smirks=apply_multiple_smirks
        )

        successfully_processed_reactants: Dict[str, str] = {}
        unsuccessfully_processed_reactants = []
        for outcome_and_applied_smirk in outcomes_and_applied_smirks:
            outcome_mapped_smiles = outcome_and_applied_smirk.get(
                "outcome_mapped_smiles"
            )
            if outcome_mapped_smiles is None:
                continue
            outcome_applied_smirk = outcome_and_applied_smirk.get("applied_smirk")
            if outcome_applied_smirk is None:
                continue

            if outcome_mapped_smiles in unsuccessfully_processed_reactants:
                continue

            if outcome_mapped_smiles in successfully_processed_reactants:
                good_reaction_smiles = successfully_processed_reactants[
                    outcome_mapped_smiles
                ]

                mapped_outcomes_smirks_dict[good_reaction_smiles].extend(
                    [outcome_applied_smirk]
                )
                continue

            outcome_reaction_smiles_dict = self._build_reaction_smiles_from_outcome(
                outcome_and_applied_smirk,
                reaction_smiles_data,
                atom_mapped_product,
            )

            if not outcome_reaction_smiles_dict:
                unsuccessfully_processed_reactants.append(outcome_mapped_smiles)

            for mapped_smiles, smirks_list in outcome_reaction_smiles_dict.items():
                if mapped_smiles not in mapped_outcomes_smirks_dict:
                    mapped_outcomes_smirks_dict[mapped_smiles] = smirks_list
                    successfully_processed_reactants[outcome_mapped_smiles] = (
                        mapped_smiles
                    )
                else:
                    mapped_outcomes_smirks_dict[mapped_smiles].extend(smirks_list)
                    successfully_processed_reactants[outcome_mapped_smiles] = (
                        mapped_smiles
                    )

        return mapped_outcomes_smirks_dict

    def _generate_mapped_product_smiles(
        self, reaction_smiles_data: ReactionData
    ) -> str:
        """
        Generate a mapped product SMILES string from reaction reactants.

        Args:
            reaction_smiles_data (ReactionData): Reaction data containing
                `rdc_products`, whose `reactants` molecule is annotated with
                atom-map numbers derived from the reactant index-to-map mapping.

        Returns:
            str: A SMILES string for the reactants molecule with atom-map numbers
            applied.
        """

        rdc_products = reaction_smiles_data["rdc_products"]

        # confusing rdchiral nomenclature - difference between retro and forward perspective
        rdc_products_mol = rdc_products.reactants

        for atom in rdc_products_mol.GetAtoms():
            atom.SetAtomMapNum(rdc_products.idx_to_mapnum(atom.GetIdx()))

        mapped_product = Chem.MolToSmiles(rdc_products_mol)

        return mapped_product

    def _fragment_fits_some_island(
        self,
        product_mols: List[Chem.Mol],
        unmapped_product_atom_islands: Dict[int, Set[int]],
        products_smarts_fragment: Chem.Mol,
    ) -> bool:
        """
        Check whether a SMARTS fragment has any overlap with an unmapped island.
        We can't do a full subset check because the SMARTS for the template
        may be overspecified, and include atoms that aren't actually changing in the
        reaction, and thus *are* mapped with the MCS mapper.

        Args:
            product_mols (List[Chem.Mol]): List of product mols.
            unmapped_product_atom_islands (Dict[int, Set[int]]): Dictionary mapping island IDs to sets of atom indices.
            products_smarts_fragment (Chem.Mol): SMARTS query fragment used to find
                substructure matches in the product molecules.

        Returns:
            bool: True if any substructure match has any overlap with an unmapped island;
            otherwise False.
        """
        for mol in product_mols:
            matches = mol.GetSubstructMatches(products_smarts_fragment)
            matches_set = [set(match) for match in matches]
            for match_set in matches_set:
                for island in unmapped_product_atom_islands.values():
                    if match_set & island:
                        return True
        return False

    def _matching_island_ids(
        self,
        unmapped_product_atom_islands_for_rdchiral: Dict[int, Set[int]],
        outcome_atom_map_indices: List[int],
    ) -> List[int]:
        """
        Find island IDs that contain any atom map indices of a given outcome.
        Can't do a full subset check for similar reasons as in
        _fragment_fits_some_island.

        Args:
            unmapped_product_atom_islands_for_rdchiral (Dict[int, Set[int]]): A dictionary
                mapping island IDs to sets of atom map indices for unmapped atoms in
                the product molecules (1-based indexing for rdchiral).
            outcome_atom_map_indices (List[int]): A list of atom map indices for
                a reaction outcome.

        Returns:
            List[int]: A list of island IDs where any atom map indices of the
                outcome are contained.
        """
        return [
            island_id
            for island_id, island in unmapped_product_atom_islands_for_rdchiral.items()
            if set(outcome_atom_map_indices) & island
        ]

    def _passes_fingerprint_screen(
        self,
        products_fps: List[DataStructs.ExplicitBitVect],
        reactants_fps: List[DataStructs.ExplicitBitVect],
        product_mol_fps: List[DataStructs.ExplicitBitVect],
        reactant_mol_fps: List[DataStructs.ExplicitBitVect],
    ) -> bool:
        """
        Check whether a template passes the fingerprint pre-screen against reaction molecule fingerprints.

        Fast bit-level check that eliminates templates whose required structural bits are
        absent from the reaction molecule fingerprints before running the more expensive
        substructure search.

        Args:
            products_fps (List[DataStructs.ExplicitBitVect]): Pattern fingerprints for the template product fragments.
            reactants_fps (List[DataStructs.ExplicitBitVect]): Pattern fingerprints for the template reactant fragments.
            product_mol_fps (List[DataStructs.ExplicitBitVect]): Pattern fingerprints for the reaction product molecules.
            reactant_mol_fps (List[DataStructs.ExplicitBitVect]): Pattern fingerprints for the reaction reactant molecules.

        Returns:
            bool: True if every template fragment fingerprint is subsumed by at least one
            reaction molecule fingerprint for both products and reactants; False otherwise.
        """
        if not all(
            any(
                DataStructs.AllProbeBitsMatch(q_fp, mol_fp)
                for mol_fp in product_mol_fps
            )
            for q_fp in products_fps
        ):
            return False

        if not all(
            any(
                DataStructs.AllProbeBitsMatch(q_fp, mol_fp)
                for mol_fp in reactant_mol_fps
            )
            for q_fp in reactants_fps
        ):
            return False

        return True

    def _passes_substructure_check(
        self,
        products_smarts: List[Chem.Mol],
        reactants_smarts: List[Chem.Mol],
        product_mols: List[Chem.Mol],
        reactant_mols: List[Chem.Mol],
        unmapped_product_atom_islands: Dict[int, Set[int]],
        has_islands: bool,
    ) -> bool:
        """
        Check whether a template's SMARTS fragments match the reaction molecules via substructure search.

        When unmapped product atom islands are present, the product check uses
        _fragment_fits_some_island to ensure each template fragment overlaps at least one
        unmapped island, rather than performing a plain substructure match.

        Args:
            products_smarts (List[Chem.Mol]): Parsed SMARTS fragments for the template products.
            reactants_smarts (List[Chem.Mol]): Parsed SMARTS fragments for the template reactants.
            product_mols (List[Chem.Mol]): RDKit molecule objects for the reaction products.
            reactant_mols (List[Chem.Mol]): RDKit molecule objects for the reaction reactants.
            unmapped_product_atom_islands (Dict[int, Set[int]]): Mapping from island ID to sets
                of unmapped atom indices in the product molecules (0-based RDKit indexing).
            has_islands (bool): Whether any unmapped product atom islands exist.

        Returns:
            bool: True if all template fragments match the respective reaction molecules;
            False otherwise.
        """
        # When islands exist, _fragment_fits_some_island subsumes the plain
        # substruct-match check (empty matches → no island overlap → False),
        # avoiding a second round of substructure searches.
        if has_islands:
            product_check_passes = all(
                self._fragment_fits_some_island(
                    product_mols, unmapped_product_atom_islands, frag
                )
                for frag in products_smarts
            )
        else:
            product_check_passes = all(
                any(mol.HasSubstructMatch(frag) for mol in product_mols)
                for frag in products_smarts
            )

        if not product_check_passes:
            return False

        return all(
            any(mol.HasSubstructMatch(frag) for mol in reactant_mols)
            for frag in reactants_smarts
        )

    def _collect_outcomes_for_template(
        self,
        template: InitializedSmirksPattern,
        rdc_products: rdc.rdchiralReactants,
        has_islands: bool,
        unmapped_product_atom_islands_for_rdchiral: Dict[int, Set[int]],
        num_smirks_applied: int,
    ) -> List[AppliedSmirkData]:
        """
        Run rdchiral on a single template and collect all valid mapped outcomes.

        Args:
            template (InitializedSmirksPattern): The initialized SMIRKS template to apply.
            rdc_products (rdc.rdchiralReactants): The rdchiral reactants object derived from
                the reaction product SMILES.
            has_islands (bool): Whether any unmapped product atom islands exist.
            unmapped_product_atom_islands_for_rdchiral (Dict[int, Set[int]]): Mapping from
                island ID to sets of atom map indices for unmapped product atoms
                (1-based rdchiral indexing).
            num_smirks_applied (int): Number of SMIRKS patterns already applied in the
                current mapping chain; used to set num_smirks_applied on each outcome.

        Returns:
            List[AppliedSmirkData]: A list of outcome data for each valid template application.
            Returns an empty list if rdchiral raises an exception or no valid outcomes are found.
        """
        rdc_rxn = template["rdc_rxn"]
        outcomes: List[AppliedSmirkData] = []
        try:
            _, outcomes_dict = rdc.rdchiralRun(
                rdc_rxn, rdc_products, return_mapped=True
            )

            for k, v in outcomes_dict.items():
                matching_ids = []
                if has_islands:
                    matching_ids = self._matching_island_ids(
                        unmapped_product_atom_islands_for_rdchiral, v[1]
                    )

                if not matching_ids:
                    continue

                for island_id in matching_ids:
                    outcomes.append(
                        AppliedSmirkData(
                            outcome_unmapped_smiles=k,
                            outcome_mapped_smiles=v[0],
                            outcome_atom_map_indices=v[1],
                            applied_smirk=template,
                            outcome_to_island_id=island_id,
                            num_smirks_applied=num_smirks_applied + 1,
                        )
                    )

                    ## TODO: Check if not Chem.MolFromSmiles(k) - identify bad templates

        except Exception as e:
            logger.warning(f"Error applying templates: {e}")

        return outcomes

    def _apply_templates(
        self,
        reaction_smiles_data: ReactionData,
        num_smirks_applied: int = 0,
        apply_multiple_smirks: bool = False,
        num_smirks_to_apply: int = 1,
    ) -> List[AppliedSmirkData]:
        """
        Apply all initialized SMIRKS templates to a reaction and collect mapped outcomes.

        Each template is screened with a fast fingerprint pre-check, then a substructure
        check, and finally run through rdchiral via _collect_outcomes_for_template. Outcomes
        are collected only when they overlap at least one unmapped product atom island (if
        any islands exist).

        Args:
            reaction_smiles_data (ReactionData): Reaction data containing molecule objects,
                rdchiral reactants, fingerprints, and unmapped product atom islands.
            num_smirks_applied (int): Number of SMIRKS patterns already applied in the
                current mapping chain.
            apply_multiple_smirks (bool): Whether to recursively apply multiple SMIRKS
                patterns (not yet implemented).
            num_smirks_to_apply (int): Maximum number of SMIRKS patterns to apply
                (not yet used).

        Returns:
            List[AppliedSmirkData]: A list of outcome data for all valid template applications.

        Raises:
            ValueError: If SMIRKS patterns were not initialized before calling this method.
        """
        product_mols = reaction_smiles_data["products_mols"]
        reactant_mols = reaction_smiles_data["reactants_mols"]
        rdc_products = reaction_smiles_data["rdc_products"]
        unmapped_product_atom_islands = reaction_smiles_data[
            "unmapped_product_atom_islands"
        ]
        product_mol_fps = reaction_smiles_data["product_mol_fps"]
        reactant_mol_fps = reaction_smiles_data["reactant_mol_fps"]

        # rdchiral uses 1-based indexing, but rdkit uses 0-based indexing
        # so we need another dictionary specifically for rdchiral
        unmapped_product_atom_islands_for_rdchiral = {
            key: {ele + 1 for ele in value}
            for key, value in unmapped_product_atom_islands.items()
        }

        has_islands = bool(unmapped_product_atom_islands)

        if self._initialized_smirks_patterns is None:
            raise ValueError("SMIRKS patterns were not initialized correctly.")

        outcomes_and_applied_smirks = []

        for template in self._initialized_smirks_patterns:
            ## TODO: Check for tautomer matches?

            if not self._passes_fingerprint_screen(
                template["products_fps"],
                template["reactants_fps"],
                product_mol_fps,
                reactant_mol_fps,
            ):
                continue

            if not self._passes_substructure_check(
                template["products_smarts"],
                template["reactants_smarts"],
                product_mols,
                reactant_mols,
                unmapped_product_atom_islands,
                has_islands,
            ):
                continue

            outcomes_and_applied_smirks.extend(
                self._collect_outcomes_for_template(
                    template,
                    rdc_products,
                    has_islands,
                    unmapped_product_atom_islands_for_rdchiral,
                    num_smirks_applied,
                )
            )

        if not apply_multiple_smirks:
            return outcomes_and_applied_smirks

        ## TODO: Implement recursive application of SMIRKS for multiple applications

        return outcomes_and_applied_smirks

    def _remove_spectator_mappings(self, smiles: str) -> str:
        """
        Remove spectator atom-mapping numbers from a SMILES string.

        This clears atom map numbers greater than or equal to 900 (used for
        spectator fragments) by setting them to 0 for each valid SMILES
        fragment.

        Args:
            smiles (str): A SMILES string, potentially containing multiple
                fragments separated by `.`.

        Returns:
            str: A SMILES string with spectator atom mappings removed.
        """
        smiles_fragments = smiles.split(".")
        mol_fragments = []
        for smiles_fragment in smiles_fragments:
            mol_fragment = Chem.MolFromSmiles(smiles_fragment)
            if mol_fragment is None:
                continue
            for atom in mol_fragment.GetAtoms():
                if atom.GetAtomMapNum() >= 900:
                    atom.SetAtomMapNum(0)
            mol_fragments.append(mol_fragment)
        return ".".join(
            [Chem.MolToSmiles(mol_fragment) for mol_fragment in mol_fragments]
        )

    def _build_reaction_smiles_from_outcome(
        self,
        outcome_and_applied_smirk: AppliedSmirkData,
        reaction_smiles_data: ReactionData,
        atom_mapped_product: str,
    ) -> Dict[str, List[InitializedSmirksPattern]]:
        """
        Process a single SMIRKS application outcome to construct a finalized reaction SMILES.

        Removes spectator mappings from the mapped outcome, identifies and handles missing
        fragments by matching against original reactant tautomers, determines spectator
        molecules that should be added back, and assembles the final reaction SMILES string.

        Args:
            outcome_and_applied_smirk (AppliedSmirkData): Data containing the mapped outcome
                SMILES and the SMIRKS pattern that was applied.
            reaction_smiles_data (ReactionData): Reaction data containing tautomer
                dictionaries and fragment count information for the original reactants.
            atom_mapped_product (str): Atom-mapped product SMILES string.

        Returns:
            Dict[str, List[InitializedSmirksPattern]]:
                A dictionary mapping the finalized reaction SMILES (reactants + spectators >> product)
                to a list containing the applied SMIRKS pattern. Returns an empty dict if
                processing fails (e.g., missing fragments cannot be resolved).
        """
        mapped_outcome = outcome_and_applied_smirk.get("outcome_mapped_smiles")
        applied_smirk = outcome_and_applied_smirk.get("applied_smirk")

        if mapped_outcome is None or applied_smirk is None:
            return {}

        tautomers_reactants = reaction_smiles_data["tautomers_reactants"]
        fragment_count_dict = reaction_smiles_data["fragment_count_reactants"]

        mapped_outcome = self._remove_spectator_mappings(mapped_outcome)

        missing_fragments, found_fragments = self._find_missing_fragments(
            mapped_outcome, tautomers_reactants
        )

        if len(missing_fragments) != 0:
            fragment_mapped_dict = self._validate_and_map_missing_fragments(
                missing_fragments,
                found_fragments,
                tautomers_reactants,
            )

            if len(fragment_mapped_dict) != len(missing_fragments):
                return {}

            for k, v in fragment_mapped_dict.items():
                if k not in mapped_outcome:
                    return {}
                mapped_outcome = mapped_outcome.replace(k, v)

        unmapped_canonical_smiles_for_mapped_smiles = [
            canonicalize_smiles(fragment_smiles, canonicalize_tautomer=False)
            for fragment_smiles in mapped_outcome.split(".")
        ]

        spectators = []
        for fragment, fragment_count in fragment_count_dict.items():
            num_occurrences_mapped = unmapped_canonical_smiles_for_mapped_smiles.count(
                fragment
            )
            missing_fragment_count = fragment_count - num_occurrences_mapped
            if missing_fragment_count > 0:
                spectators.extend([fragment] * missing_fragment_count)

        reactants = mapped_outcome.split(".")
        reactants_and_spectators = reactants + spectators

        finalized_reaction_smiles = (
            ".".join(reactants_and_spectators) + ">>" + atom_mapped_product
        )

        return {finalized_reaction_smiles: [applied_smirk]}

    def _find_missing_fragments(
        self, mapped_outcome: str, unmapped_reactants: Dict[str, List[str]]
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Identify fragments from a mapped outcome that are missing from unmapped reactants.

        Compares each fragment in the mapped outcome (after canonicalization) against
        the flattened list of unmapped reactant fragments to determine which fragments
        are present in the original reactants and which are newly formed or missing.

        Args:
            mapped_outcome (str): SMILES string of the mapped reaction outcome with
                atom mapping numbers, with fragments separated by dots.
            unmapped_reactants (Dict[str, List[str]]): Dictionary mapping original
                reactant SMILES to lists of enumerated tautomer SMILES for unmapped reactants.

        Returns:
            Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
                - First list: Tuples of (canonical_unmapped_fragment, original_mapped_fragment)
                    for fragments not found in the unmapped reactants (missing fragments).
                - Second list: Tuples of (canonical_unmapped_fragment, original_mapped_fragment)
                    for fragments found in the unmapped reactants (found fragments).
        """
        missing_fragments = []
        found_fragments = []
        reactant_fragments = list(unmapped_reactants.values())
        flattened_reactant_fragments = [
            item for sublist in reactant_fragments for item in sublist
        ]

        for mapped_fragment in mapped_outcome.split("."):
            unmapped_fragment = canonicalize_smiles(
                mapped_fragment, canonicalize_tautomer=False
            )
            if unmapped_fragment not in flattened_reactant_fragments:
                missing_fragments.append((unmapped_fragment, mapped_fragment))
            else:
                found_fragments.append((unmapped_fragment, mapped_fragment))

        return missing_fragments, found_fragments

    def _validate_and_map_missing_fragments(
        self,
        missing_fragments: List[Tuple[str, str]],
        found_fragments: List[Tuple[str, str]],
        unmapped_reactants: Dict[str, List[str]],
    ) -> Dict[str, str]:
        """
        Identify and map missing reaction fragments to unmapped reactants.

        Validates that missing fragments are substructures of unmapped reactants, then
        attempts to map each missing fragment to its corresponding location in the
        unmapped reactant molecules by transferring atom mapping information.

        Args:
            missing_fragments (List[Tuple[str, str]]): List of tuples containing
                (unmapped_fragment, mapped_fragment) for fragments not found in reactants.
            found_fragments (List[Tuple[str, str]]): List of tuples containing
                (unmapped_fragment, mapped_fragment) for fragments already mapped to reactants.
            unmapped_reactants (Dict[str, List[str]]): Dictionary mapping original reactant SMILES
                to lists of enumerated tautomer SMILES for unmapped reactants.

        Returns:
            Dict[str, str]: Mapping from mapped fragment SMILES to their corresponding
                identified unmapped fragment SMILES with atom mapping transferred.
                Returns an empty dict if fragments cannot be mapped or if multiple
                possible mappings exist for any fragment.

        Note:
            If multiple possible fragments are identified for any reaction SMARTS
            substructure, a warning is logged and an empty dict is returned to
            indicate ambiguous mapping.
        """
        all_fragments_substructs = self._are_fragments_substructures(
            missing_fragments, found_fragments, unmapped_reactants
        )
        if not all_fragments_substructs:
            return {}

        fragment_mapped_dict = self._identify_and_map_fragments(
            missing_fragments,
            found_fragments,
            unmapped_reactants,
        )

        filtered_fragment_mapped_dict = {}
        for k, v in fragment_mapped_dict.items():
            if len(v) > 1:
                logger.warning(
                    "Multiple possible fragments identified for reaction SMARTS substructure"
                )
                return {}
            filtered_fragment_mapped_dict[k] = v[0]

        return filtered_fragment_mapped_dict

    def _are_fragments_substructures(
        self,
        missing_fragments: List[Tuple[str, str]],
        found_fragments: List[Tuple[str, str]],
        unmapped_reactants: Dict[str, List[str]],
    ) -> bool:
        """
        Check if missing fragments with wildcards are substructures of unmapped reactants.

        For each missing fragment that contains a wildcard ("*"), this method checks
        whether the fragment pattern matches as a substructure within any of the
        unmapped reactant fragments. This is used to validate whether missing fragments
        can be accounted for as parts of larger reactant molecules.

        Args:
            missing_fragments (List[Tuple[str, str]]): List of tuples containing
                (fragment_smarts, mapped_fragment) for fragments not yet mapped to reactants.
            found_fragments (List[Tuple[str, str]]): List of tuples containing
                (fragment_smarts, mapped_fragment) for fragments already mapped to reactants.
            unmapped_reactants (Dict[str, List[str]]): Dictionary mapping original reactant
                SMILES to lists of tautomer SMILES that have not been mapped to fragments.

        Returns:
            bool: True if all missing fragments with wildcards are found as substructures
                within unmapped reactants; False otherwise. Returns True if no wildcards
                are present in missing fragments.

        Note:
            Fragments without wildcards ("*") are skipped from substructure matching.
            If any SMARTS parsing fails, the method returns False immediately.
        """
        unmapped_found_fragments = [
            unmapped_fragment[0] for unmapped_fragment in found_fragments
        ]
        for fragment_str, _ in missing_fragments:
            if "*" not in fragment_str:
                continue

            query_mol = Chem.MolFromSmarts(fragment_str)
            if not query_mol:
                return False

            found_match = False
            for reactant_group in unmapped_reactants.values():
                for reactant_fragment_str in reactant_group:
                    if reactant_fragment_str in unmapped_found_fragments:
                        continue
                    reactant_mol = Chem.MolFromSmarts(reactant_fragment_str)
                    if not reactant_mol:
                        return False
                    reactant_mol.UpdatePropertyCache()
                    if reactant_mol.HasSubstructMatch(query_mol):
                        found_match = True
                        break

                if found_match:
                    break

            if not found_match:
                return False

        return True

    def _identify_and_map_fragments(
        self,
        missing_fragments: List[Tuple[str, str]],
        found_fragments: List[Tuple[str, str]],
        unmapped_reactants: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """ """
        unmapped_found_fragments = [ele[0] for ele in found_fragments]
        fragment_mapped_dict = {}
        for _, mapped_reactant_fragment in missing_fragments:
            fragment_found = False
            for orig_tautomer, tautomer_list in unmapped_reactants.items():
                for tautomer in tautomer_list:
                    if tautomer in unmapped_found_fragments:
                        continue

                    out = self._transfer_mapping(mapped_reactant_fragment, tautomer)

                    if not out:
                        continue

                    # TODO: Is this even needed if we just take all possible fragments in _validate_and_map_missing_fragments?
                    if len(tautomer_list) > 1:
                        mapped_enumerated_tautomers = list(
                            self._tautomer_enumerator.Enumerate(Chem.MolFromSmiles(out))
                        )
                        for mapped_enumerated_tautomer in mapped_enumerated_tautomers:
                            unmapped_tautomer_copy = Chem.Mol(
                                mapped_enumerated_tautomer
                            )
                            [
                                atom.SetAtomMapNum(0)
                                for atom in unmapped_tautomer_copy.GetAtoms()
                            ]
                            if Chem.MolToSmiles(
                                unmapped_tautomer_copy
                            ) == Chem.MolToSmiles(Chem.MolFromSmiles(orig_tautomer)):
                                out = Chem.MolToSmiles(mapped_enumerated_tautomer)
                                break

                    fragment_found = True

                    if mapped_reactant_fragment not in fragment_mapped_dict:
                        fragment_mapped_dict[mapped_reactant_fragment] = [out]
                    else:
                        existing_mapped_fragments = fragment_mapped_dict[
                            mapped_reactant_fragment
                        ]
                        existing_mapped_fragments.append(out)
                        fragment_mapped_dict[mapped_reactant_fragment] = sorted(
                            list(set(existing_mapped_fragments))
                        )

            if not fragment_found:
                return {}

        return fragment_mapped_dict

    def _transfer_mapping(
        self, mapped_substructure_smarts: str, full_molecule_smiles: str
    ) -> str | None:
        """
        Transfer atom mapping numbers from a query mol to a full molecule.

        The query mol is matched against the molecule, and if a unique symmetric match is found,
        the atom map numbers from the query mol are transferred to the corresponding atoms
        in the molecule. Map numbers in the range [1, 899] are transferred; map numbers
        >= 900 (spectator atoms) and map number 0 (unmapped) are ignored.

        Args:
            mapped_substructure_smarts (str): A SMARTS string representing the substructure
                query mol with atom map numbers to transfer.
            full_molecule_smiles (str): A SMILES string representing the full molecule
                to receive the atom map numbers.

        Returns:
            str | None: The mapped SMILES string with transferred atom map numbers, or None
                if the query mol cannot be parsed, the molecule cannot be parsed, no substructure
                match is found, or multiple non-symmetric matches are found.

        Raises:
            None: This method does not raise exceptions; it returns None on any error.
        """
        query_mol = Chem.MolFromSmarts(mapped_substructure_smarts)
        if not query_mol:
            return None

        mol = Chem.MolFromSmiles(full_molecule_smiles)
        if not mol:
            return None

        match_indices = mol.GetSubstructMatches(query_mol)

        if not match_indices:
            return None

        symmetry_class = {
            k: v
            for k, v in enumerate(
                list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
            )
        }

        symmetric = True
        for match_1 in match_indices:
            for match_2 in match_indices:
                for ele1, ele2 in zip(match_1, match_2):
                    if symmetry_class[ele1] != symmetry_class[ele2]:
                        symmetric = False
                        break

        if len(match_indices) != 1 and not symmetric:
            return None

        match_indices = match_indices[0]

        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() != 0:
                atom.SetAtomMapNum(0)

        for query_mol_atom in query_mol.GetAtoms():
            map_num = query_mol_atom.GetAtomMapNum()

            if map_num > 0 and map_num < 900:
                query_mol_atom_idx = query_mol_atom.GetIdx()
                mol_idx = match_indices[query_mol_atom_idx]
                mol_atom = mol.GetAtomWithIdx(mol_idx)
                mol_atom.SetAtomMapNum(map_num)

        mapped_smiles_output = Chem.MolToSmiles(mol)
        return mapped_smiles_output

    def _get_unmapped_product_atom_islands(self, smiles: str) -> Dict[int, Set[int]]:
        """
        Find connected components ("islands") of unmapped atoms in a product SMILES.

        Args:
            smiles (str): Product SMILES string to analyze.

        Returns:
            Dict[int, Set[int]]: Mapping from island index (0..N-1) to a set of RDKit
            atom indices belonging to that connected component, considering only atoms
            with atom map number equal to 0.

        Raises:
            ValueError: If the SMILES cannot be parsed into an RDKit molecule.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")

        unmapped = {
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 0
        }

        visited = set()
        islands: Dict[int, Set[int]] = {}

        for idx in unmapped:
            if idx in visited:
                continue

            island: Set[int] = set()
            queue = deque([idx])
            visited.add(idx)

            while queue:
                current = queue.popleft()
                island.add(current)

                for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in unmapped and neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)

            islands[len(islands)] = island

        return islands

    def _select_preferred_mapping(
        self, possible_outcomes: Dict[str, List[InitializedSmirksPattern]]
    ) -> str:
        """
        Select the preferred mapping from a list of mappings.

        Args:
            possible_outcomes (Dict[str, List[InitializedSmirksPattern]]): Dictionary of possible mappings.

        Returns:
            str: The selected preferred mapping.
        """
        selected_mapping = ""
        max_num_mapped_product_atoms = 0
        highest_priority_class = (0, 0)
        for canonicalized_mapping, possible_mappings in possible_outcomes.items():
            for possible_mapping in possible_mappings:
                mapping_priority = possible_mapping.get("priority", (0, 0))
                if highest_priority_class[0] != 0 and mapping_priority[0] == 0:
                    continue
                if (
                    mapping_priority[0] != 0
                    and mapping_priority[0] > highest_priority_class[0]
                ):
                    highest_priority_class = mapping_priority
                    selected_mapping = canonicalized_mapping
                    continue
                elif (
                    mapping_priority[0] != 0
                    and mapping_priority[0] == highest_priority_class[0]
                    and mapping_priority[1] > highest_priority_class[1]
                ):
                    highest_priority_class = mapping_priority
                    selected_mapping = canonicalized_mapping
                    continue

                mapping_num_fragments = len(possible_mapping["reactants_smarts"])
                mapping_num_atoms = mapping_num_fragments
                for mol in possible_mapping["reactants_smarts"]:
                    mapping_num_atoms += len(mol.GetAtoms())
                for mol in possible_mapping["products_smarts"]:
                    mapping_num_atoms += len(mol.GetAtoms())
                if max_num_mapped_product_atoms < mapping_num_atoms:
                    selected_mapping = canonicalized_mapping
                    max_num_mapped_product_atoms = mapping_num_atoms
        return selected_mapping

    def _filter_and_deduplicate_outcomes(
        self,
        mapped_outcomes_smirks_dict: Dict[str, List[InitializedSmirksPattern]],
        canonicalized_input_smiles: str,
        original_smiles: str,
    ) -> Optional[ReactionMapperResult]:
        """
        Filter, validate, and deduplicate mapped reaction outcomes.

        Args:
            mapped_outcomes_smirks_dict (Dict[str, InitializedSmirksPattern]): Map of
                candidate mapped reaction SMIRKS/SMILES strings to their originating
                initialized pattern metadata.
            canonicalized_input_smiles (str): Canonicalized representation of the
                input reaction (used to discard outcomes that do not match).
            original_smiles (str): Original (non-canonicalized) input reaction string
                to return in the result.

        Returns:
            Optional[ReactionMapperResult]: A single selected mapping and associated
                candidate metadata if exactly one valid, matching outcome remains;
                otherwise, returns None.
        """
        deduplicated_mapped_outcomes: Dict[str, List[InitializedSmirksPattern]] = {}
        for (
            candidate_reaction_smiles,
            applied_patterns,
        ) in mapped_outcomes_smirks_dict.items():
            if not candidate_reaction_smiles:
                continue
            canonicalized_candidate_reaction_smiles = canonicalize_atom_mapping(
                canonicalize_reaction_smiles(
                    candidate_reaction_smiles,
                    canonicalize_tautomer=False,
                    remove_mapping=False,
                )
            )
            if not canonicalized_candidate_reaction_smiles:
                continue
            if (
                canonicalize_reaction_smiles(
                    canonicalized_candidate_reaction_smiles, canonicalize_tautomer=False
                )
                != canonicalized_input_smiles
            ):
                continue
            if not self._verify_validity_of_mapping(
                canonicalized_candidate_reaction_smiles
            ):
                continue

            if (
                canonicalized_candidate_reaction_smiles
                not in deduplicated_mapped_outcomes
            ):
                deduplicated_mapped_outcomes[
                    canonicalized_candidate_reaction_smiles
                ] = applied_patterns
            else:
                deduplicated_mapped_outcomes[
                    canonicalized_candidate_reaction_smiles
                ].extend(applied_patterns)

        if len(deduplicated_mapped_outcomes) == 0:
            return None
        if len(deduplicated_mapped_outcomes) > 1:
            logger.warning("Multiple possible mappings")
            selected_mapping = self._select_preferred_mapping(
                deduplicated_mapped_outcomes
            )
        else:
            selected_mapping = list(deduplicated_mapped_outcomes.keys())[0]

        return ReactionMapperResult(
            original_smiles=original_smiles,
            selected_mapping=selected_mapping,
            possible_mappings=deduplicated_mapped_outcomes,
            mapping_type=self._mapper_type,
            mapping_score=None,
            additional_info=[],
        )

    def map_reaction_with_mcs_optimization(
        self, reaction_smiles: str, apply_multiple_smirks: bool = False
    ) -> Tuple[ReactionMapperResult, ReactionMapperResult]:
        """
        Map a reaction SMILES string using template-based atom mapping with optimization
        that uses MCS to identify probable reaction center.

        Args:
            reaction_smiles (str): Reaction SMILES to map.
            apply_multiple_smirks (bool): Whether to apply multiple SMIRKS patterns to the same reaction.

        Returns:
            Tuple[ReactionMapperResult, ReactionMapperResult]: A tuple containing the template-based
                mapping result and the MCS mapping result.
        """
        self._initialize_smirks_patterns()

        if not self._reaction_smiles_valid(reaction_smiles):
            return (
                self._return_default_mapping_dict(reaction_smiles),
                self._return_default_mapping_dict(reaction_smiles),
            )

        canonicalized_reaction_smiles = canonicalize_reaction_smiles(
            reaction_smiles, canonicalize_tautomer=False
        )

        unmapped_product_atom_islands = {}
        if self._mcs_mapper is not None:
            mcs_result = self._mcs_mapper.map_reaction(canonicalized_reaction_smiles)

            if mcs_result["selected_mapping"] != "":
                unmapped_product_atom_islands = self._get_unmapped_product_atom_islands(
                    mcs_result["selected_mapping"].split(">>")[1]
                )

        reactants_str, products_str = self._split_reaction_components(
            canonicalized_reaction_smiles
        )

        reaction_data = self._prepare_reaction_data(
            reactants_str, products_str, unmapped_product_atom_islands
        )

        mapped_outcomes_smirks_dict = self._apply_templates_and_collect_outcomes(
            reaction_data, apply_multiple_smirks=apply_multiple_smirks
        )

        result = self._filter_and_deduplicate_outcomes(
            mapped_outcomes_smirks_dict, canonicalized_reaction_smiles, reaction_smiles
        )
        if not result:
            return self._return_default_mapping_dict(reaction_smiles), mcs_result

        return result, mcs_result

    def map_reaction(
        self, reaction_smiles: str, return_mcs_result: bool = False
    ) -> ReactionMapperResult:
        """
        Map a reaction SMILES string using template-based atom mapping.

        This is a convenience method that calls map_reaction_with_mcs_optimization and returns only the main mapping result.

        Args:
            reaction_smiles (str): Reaction SMILES to map.
            return_mcs_result (bool): Whether to return the MCS mapping result.

        Returns:
            ReactionMapperResult: A mapping result containing the selected mapping and
            related metadata. If the input is invalid or no unique valid mapping can be
            produced, an "empty" default result is returned.
        """
        result, _ = self.map_reaction_with_mcs_optimization(reaction_smiles)
        return result

    def map_reactions(
        self,
        reaction_list: List[str],
    ) -> List[ReactionMapperResult]:
        """
        Map a list of reaction SMILES strings using this mapper.

        Args:
            reaction_list (List[str]): Reaction SMILES strings to map.

        Returns:
            List[ReactionMapperResult]: Mapping results in the same order as the input.
        """

        mapped_reactions = []
        for reaction in reaction_list:
            mapped_reactions.append(self.map_reaction(reaction))
        return mapped_reactions
