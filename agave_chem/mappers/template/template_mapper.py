import json
from collections import deque
from importlib.resources import files
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

from rdchiral import main as rdc
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper
from agave_chem.mappers.reaction_mapper import ReactionMapper, ReactionMapperResult
from agave_chem.mappers.template.template_initialization import (
    InitializedSmirksPattern,
    SmirksPattern,
    initialize_template_data,
)
from agave_chem.utils.chem_utils import (
    canonicalize_atom_mapping,
    canonicalize_reaction_smiles,
    canonicalize_smiles,
)
from agave_chem.utils.logging_config import logger


class ReactionData(TypedDict):
    products_mols: List[Chem.Mol]
    reactants_mols: List[Chem.Mol]
    rdc_products: Any
    tautomers_reactants: Dict[str, List[str]]
    fragment_count_reactants: Dict[str, int]
    unmapped_product_atom_islands: Dict[int, Set[int]]


class AppliedSmirkData(TypedDict):
    outcome_unmapped_smiles: str
    outcome_mapped_smiles: str
    outcome_atom_map_indices: List[int]
    applied_smirk: InitializedSmirksPattern
    outcome_to_island_id: int | None
    num_smirks_applied: int


class ExpertReactionMapper(ReactionMapper):
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

        super().__init__("expert", mapper_name, mapper_weight)

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

        SMIRKS_PATTERNS_FILE = files("agave_chem.datafiles").joinpath(
            "smirks_patterns.json"
        )
        default_smirks_patterns = []
        with SMIRKS_PATTERNS_FILE.open("r") as f:
            default_smirks_patterns = json.load(f)

        self._smirks_patterns: List[SmirksPattern] = []
        if use_default_smirks_patterns and custom_smirks_patterns is None:
            self._smirks_patterns = default_smirks_patterns
        elif custom_smirks_patterns and not use_default_smirks_patterns:
            self._smirks_patterns = custom_smirks_patterns
        elif custom_smirks_patterns and use_default_smirks_patterns:
            self._smirks_patterns = custom_smirks_patterns + default_smirks_patterns
        else:
            raise TypeError(
                "Attempting to initialize AgaveChem with no SMIRKS patterns"
            )

        self._smirks_name_dictionary = {
            pattern["smirks"]: {
                "name": pattern["name"],
                "superclass": pattern["superclass_id"],
            }
            for pattern in self._smirks_patterns
        }
        self._initialized_smirks_patterns: List[InitializedSmirksPattern] = (
            initialize_template_data(self._smirks_patterns)
        )

        self._tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
        self._tautomer_enumerator.SetMaxTransforms(max_transforms)
        self._tautomer_enumerator.SetMaxTautomers(max_tautomers)

        self._mcs_mapper = None
        if use_mcs_mapping:
            self._mcs_mapper = MCSReactionMapper(
                mapper_name="mcs_for_template", mapper_weight=1
            )

    def _validate_smirks_patterns(self, smirks_patterns: List[SmirksPattern]) -> None:
        """Validates SMIRKS patterns."""
        pass

    def _prepare_reaction_data(
        self,
        reactants_str: str,
        products_str: str,
        unmapped_product_atom_islands: Optional[Dict[int, Set[int]]] = None,
    ) -> ReactionData:
        """Prepare reaction mapping inputs from reactant and product SMILES strings.

        Args:
            reactants_str (str): Reactants SMILES string.
            products_str (str): Products SMILES string.
            unmapped_product_atom_islands (Optional[List[str]]): Product atom-island SMILES
                strings that are intentionally left unmapped. Defaults to None.

        Returns:
            ReactionData: Mapping input data containing RDKit molecule objects for
            reactants/products, an RDChiral reactants object derived from
            `products_str`, a tautomer SMILES dictionary, a fragment count dictionary,
            and the normalized `unmapped_product_atom_islands` list.
        """

        if unmapped_product_atom_islands is None:
            unmapped_product_atom_islands = {}

        return ReactionData(
            products_mols=[
                Chem.MolFromSmiles(product_str)
                for product_str in products_str.split(".")
            ],
            reactants_mols=[
                Chem.MolFromSmiles(reactant_str)
                for reactant_str in reactants_str.split(".")
            ],
            rdc_products=rdc.rdchiralReactants(products_str),
            tautomers_reactants=self._enumerate_tautomer_smiles(reactants_str),
            fragment_count_reactants=self._get_fragment_count_dict(reactants_str),
            unmapped_product_atom_islands=unmapped_product_atom_islands,
        )

    def _enumerate_tautomer_smiles(self, smiles: str) -> Dict[str, List[str]]:
        """Enumerate tautomer SMILES strings for a given SMILES string.

        Args:
            smiles (str): A SMILES string representing one or more molecular fragments.

        Returns:
            Dict[str, List[str]]: A dictionary where keys are the fragments of the input
                SMILES string and values are lists of the enumerated tautomer SMILES strings
                for each fragment.

        """
        enumerated_smiles_dict = {}
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
                canonicalize_smiles(frag_smiles)
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
            if fragment_str not in fragment_count_dict:
                fragment_count_dict[fragment_str] = 1
            else:
                fragment_count_dict[fragment_str] += 1

        return fragment_count_dict

    def _process_templates(
        self, reaction_smiles_data: ReactionData
    ) -> Dict[str, InitializedSmirksPattern]:
        """
        Apply template SMIRKS patterns to a reaction and collect mapped outcomes.

        Args:
            reaction_smiles_data (ReactionData): Reaction data containing reactants,
                products, and precomputed helper structures used for template
                application.

        Returns:
            Dict[str, List]: A mapping from mapped outcome SMILES to a list of
            applied SMIRKS patterns.
        """
        mapped_outcomes_smirks_dict: Dict[str, InitializedSmirksPattern] = {}

        atom_mapped_product = self._get_mapped_product(reaction_smiles_data)
        outcomes_and_applied_smirks = self._apply_templates(reaction_smiles_data)

        for outcome_and_applied_smirk in outcomes_and_applied_smirks:
            result = self._process_single_outcome(
                outcome_and_applied_smirk,
                reaction_smiles_data,
                atom_mapped_product,
            )
            mapped_outcomes_smirks_dict.update(result)

        return mapped_outcomes_smirks_dict

    def _get_mapped_product(self, reaction_smiles_data: ReactionData) -> str:
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

        rdc_products_mol = (
            rdc_products.reactants
        )  # rdchiral calls the product "reactants" in the RDC object
        for atom in rdc_products_mol.GetAtoms():
            atom.SetAtomMapNum(rdc_products.idx_to_mapnum(atom.GetIdx()))
        mapped_product = Chem.MolToSmiles(rdc_products_mol)

        return mapped_product

    def _apply_templates(
        self,
        reaction_smiles_data: ReactionData,
        num_smirks_applied: int = 0,
        apply_multiple_smirks: bool = True,
        num_smirks_to_apply: int = 1,
    ) -> List[AppliedSmirkData]:
        """ """
        product_mols = reaction_smiles_data["products_mols"]
        reactant_mols = reaction_smiles_data["reactants_mols"]
        rdc_products = reaction_smiles_data["rdc_products"]
        unmapped_product_atom_islands = reaction_smiles_data[
            "unmapped_product_atom_islands"
        ]

        # rdchiral uses 1-based indexing, but rdkit uses 0-based indexing
        # so we need another dictionary specifically for rdchiral
        unmapped_product_atom_islands_for_rdchiral = {
            key: {ele + 1 for ele in value}
            for key, value in unmapped_product_atom_islands.items()
        }

        outcomes_and_applied_smirks = []

        for template in self._initialized_smirks_patterns:
            products_smarts = template["products_smarts"]
            reactant_smarts = template["reactant_smarts"]
            rdc_rxn = template["rdc_rxn"]

            product_mol_has_substruct_match = all(
                any(
                    product_mol.HasSubstructMatch(products_smarts_fragment)
                    for product_mol in product_mols
                )
                for products_smarts_fragment in products_smarts
            )

            if not product_mol_has_substruct_match:
                continue

            reactant_mol_has_substruct_match = all(
                any(
                    reactant_mol.HasSubstructMatch(reactant_smarts_fragment)
                    for reactant_mol in reactant_mols
                )
                for reactant_smarts_fragment in reactant_smarts
            )

            if not reactant_mol_has_substruct_match:
                continue

            if not unmapped_product_atom_islands:
                template_applies_to_unmapped_product_atoms = True
            else:

                def _fragment_fits_some_island(
                    products_smarts_fragment: Chem.Mol,
                ) -> bool:
                    """
                    Check whether a SMARTS fragment has any overlap with an unmapped island.
                    We can't do a full subset check because the SMARTS for the template
                    may be overspecified, and include atoms that aren't actually changing in the
                    reaction, and thus *are* mapped with the MCS mapper.

                    Args:
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

                template_applies_to_unmapped_product_atoms = all(
                    _fragment_fits_some_island(products_smarts_fragment)
                    for products_smarts_fragment in products_smarts
                )

            if not template_applies_to_unmapped_product_atoms:
                continue

            try:
                _, outcomes_dict = rdc.rdchiralRun(
                    rdc_rxn, rdc_products, return_mapped=True
                )

                if unmapped_product_atom_islands:

                    def _matching_island_ids(v: Tuple[str, List[int]]) -> List[int]:
                        """
                        Find island IDs that contain any atom map indices of a given outcome.
                        Can't do a full subset check for similar reasons as in
                        _fragment_fits_some_island.

                        Args:
                            v (Tuple[str, List[int]]): A tuple where the second element is a list
                                of atom map indices for a reaction outcome.

                        Returns:
                            List[int]: A list of island IDs where any atom map indices of the
                                outcome are contained.
                        """
                        return [
                            island_id
                            for island_id, island in unmapped_product_atom_islands_for_rdchiral.items()
                            if set(v[1]) & island
                        ]

                    for k, v in outcomes_dict.items():
                        matching_island_ids = _matching_island_ids(v)
                        if not matching_island_ids:
                            continue

                        for matching_island_id in matching_island_ids:
                            outcomes_and_applied_smirks.append(
                                AppliedSmirkData(
                                    outcome_unmapped_smiles=k,
                                    outcome_mapped_smiles=v[0],
                                    outcome_atom_map_indices=v[1],
                                    applied_smirk=template,
                                    outcome_to_island_id=matching_island_id,
                                    num_smirks_applied=num_smirks_applied + 1,
                                )
                            )

                else:
                    for k, v in outcomes_dict.items():
                        outcomes_and_applied_smirks.append(
                            AppliedSmirkData(
                                outcome_unmapped_smiles=k,
                                outcome_mapped_smiles=v[0],
                                outcome_atom_map_indices=v[1],
                                applied_smirk=template,
                                outcome_to_island_id=None,
                                num_smirks_applied=num_smirks_applied + 1,
                            )
                        )

            except Exception as e:
                logger.warning(f"Error applying templates: {e}")
                pass

        if not apply_multiple_smirks:
            return outcomes_and_applied_smirks

        # for outcome_and_applied_smirk in outcomes_and_applied_smirks:
        #     if outcome_and_applied_smirk.num_smirks_applied >= num_smirks_to_apply:
        #         continue
        #     recursively_applied_smirks_and_outcomes = self._apply_templates(
        #         reaction_smiles_data=outcome_and_applied_smirk,
        #         num_smirks_applied=outcome_and_applied_smirk.num_smirks_applied,
        #         apply_multiple_smirks=apply_multiple_smirks,
        #         num_smirks_to_apply=num_smirks_to_apply,
        #     )

        #     outcomes_and_applied_smirks.extend(recursively_applied_smirks_and_outcomes)

        return outcomes_and_applied_smirks

    def _remove_spectator_mappings(self, smiles: str) -> str:
        """Remove spectator atom-mapping numbers from a SMILES string.

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

    def _process_single_outcome(
        self,
        outcome_and_applied_smirk: AppliedSmirkData,
        reaction_smiles_data: ReactionData,
        atom_mapped_product: str,
    ) -> Dict[str, InitializedSmirksPattern]:
        """ """
        mapped_outcome = outcome_and_applied_smirk.get("outcome_mapped_smiles")
        applied_smirk = outcome_and_applied_smirk.get("applied_smirk")

        tautomers_reactants = reaction_smiles_data["tautomers_reactants"]
        fragment_count_dict = reaction_smiles_data["fragment_count_reactants"]

        mapped_outcome = self._remove_spectator_mappings(mapped_outcome)

        missing_fragments, found_fragments = self._find_missing_fragments(
            mapped_outcome, tautomers_reactants
        )

        if len(missing_fragments) != 0:
            fragment_mapped_dict = self._handle_missing_fragments(
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
            canonicalize_smiles(ele) for ele in mapped_outcome.split(".")
        ]

        spectators = []
        for ele, ele_count in fragment_count_dict.items():
            canonicalized_ele = canonicalize_smiles(ele)
            num_occurrences_mapped = unmapped_canonical_smiles_for_mapped_smiles.count(
                canonicalized_ele
            )
            dif_num_occurrences = ele_count - num_occurrences_mapped
            if dif_num_occurrences > 0:
                spectators.extend([canonicalized_ele] * dif_num_occurrences)

        reactants = mapped_outcome.split(".")
        reactants_and_spectators = reactants + spectators

        finalized_reaction_smiles = (
            ".".join(reactants_and_spectators) + ">>" + atom_mapped_product
        )

        return {finalized_reaction_smiles: applied_smirk}

    def _find_missing_fragments(
        self, mapped_outcome: str, unmapped_reactants: Dict[str, List[str]]
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """ """
        missing_fragments = []
        found_fragments = []
        reactant_fragments = list(unmapped_reactants.values())
        flattened_reactant_fragments = [
            item for sublist in reactant_fragments for item in sublist
        ]

        for mapped_fragment in mapped_outcome.split("."):
            unmapped_fragment = canonicalize_smiles(mapped_fragment)
            if unmapped_fragment not in flattened_reactant_fragments:
                missing_fragments.append((unmapped_fragment, mapped_fragment))
            else:
                found_fragments.append((unmapped_fragment, mapped_fragment))

        return missing_fragments, found_fragments

    def _handle_missing_fragments(
        self,
        missing_fragments: List[Tuple[str, str]],
        found_fragments: List[Tuple[str, str]],
        unmapped_reactants: Dict[str, List[str]],
    ) -> Dict[str, str]:
        """ """
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
        """ """
        unmapped_found_fragments = [ele[0] for ele in found_fragments]
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
            for _, tautomer_list in unmapped_reactants.items():
                for tautomer in tautomer_list:
                    if tautomer in unmapped_found_fragments:
                        continue

                    out = self._transfer_mapping(mapped_reactant_fragment, tautomer)

                    if not out:
                        continue

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
        Transfers atom map numbers from a mapped SMARTS substructure
        to a full molecule SMILES, leaving atoms corresponding to '*' unmapped.

        Args:
            mapped_substructure_smarts (str): SMARTS string of the substructure
                                               with atom map numbers. Wildcards (*)
                                               are expected for connection points
                                               and should not have map numbers.
            full_molecule_smiles (str): SMILES string of the complete, unmapped molecule.

        Returns:
            str: The SMILES string of the full molecule with map numbers transferred
                 from the substructure match, or None if an error occurs (e.g.,
                 parsing failed, substructure not found).
        """
        pattern = Chem.MolFromSmarts(mapped_substructure_smarts)
        if not pattern:
            return None

        mol = Chem.MolFromSmiles(full_molecule_smiles)
        if not mol:
            return None

        match_indices = mol.GetSubstructMatches(pattern)

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

        if not match_indices:
            return None

        if len(match_indices) != 1 and not symmetric:
            return None

        match_indices = match_indices[0]

        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() != 0:
                atom.SetAtomMapNum(0)

        for pattern_atom in pattern.GetAtoms():
            map_num = pattern_atom.GetAtomMapNum()

            if map_num > 0 and map_num < 900:
                pattern_idx = pattern_atom.GetIdx()
                mol_idx = match_indices[pattern_idx]
                mol_atom = mol.GetAtomWithIdx(mol_idx)
                mol_atom.SetAtomMapNum(map_num)

        mapped_smiles_output = Chem.MolToSmiles(mol)
        return mapped_smiles_output

    def _get_unmapped_product_atom_islands(self, smiles: str) -> Dict[int, Set[int]]:
        """ """
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
                    n_idx = neighbor.GetIdx()
                    if n_idx in unmapped and n_idx not in visited:
                        visited.add(n_idx)
                        queue.append(n_idx)

            islands_idx = len(islands)
            islands[islands_idx] = island

        return islands

    def map_reaction(self, reaction_smiles: str) -> ReactionMapperResult:
        """ """
        default_mapping_dict: ReactionMapperResult = {
            "mapping": "",
            "additional_info": [{}],
        }
        if not self._reaction_smiles_valid(reaction_smiles):
            return default_mapping_dict

        canonicalized_reaction_smiles = canonicalize_reaction_smiles(
            reaction_smiles, canonicalize_tautomer=True
        )

        unmapped_product_atom_islands = {}
        if self._mcs_mapper is not None:
            mcs_result = self._mcs_mapper.map_reaction(canonicalized_reaction_smiles)
            # print("MCS RESULT:", mcs_result)
            # print("~~~~~~~~~~~~~~~~~~~~~")

            unmapped_product_atom_islands = self._get_unmapped_product_atom_islands(
                mcs_result["mapping"].split(">>")[1]
            )

        reactants_str, products_str = self._split_reaction_components(
            canonicalized_reaction_smiles
        )

        reaction_data = self._prepare_reaction_data(
            reactants_str, products_str, unmapped_product_atom_islands
        )

        mapped_outcomes_smirks_dict = self._process_templates(
            reaction_data,
        )

        mapped_outcomes = [
            canonicalize_atom_mapping(
                canonicalize_reaction_smiles(
                    ele, canonicalize_tautomer=False, remove_mapping=False
                )
            )
            for ele in list(set(list(mapped_outcomes_smirks_dict.keys())))
        ]

        deduplicated_mapped_outcomes = list(
            set([ele for ele in mapped_outcomes if ele != ""])
        )

        possible_mappings = list(
            set(
                [
                    ele
                    for ele in deduplicated_mapped_outcomes
                    if canonicalize_reaction_smiles(ele, canonicalize_tautomer=False)
                    == canonicalized_reaction_smiles
                ]
            )
        )

        possible_mappings = [
            ele
            for ele in possible_mappings
            if self._verify_validity_of_mapping(possible_mappings[0])
        ]

        if len(possible_mappings) > 1:
            logger.warning("Multiple possible mappings")
            ## add tie breaker - more specified SMARTS?
            ## also order the applied_smirks_names by most specific smarts

            return default_mapping_dict
        if len(possible_mappings) == 0:
            return default_mapping_dict

        applied_smirks_names = []
        for initialized_smirk in list(mapped_outcomes_smirks_dict.values()):
            parent_smirk = initialized_smirk["parent_smirks"]
            applied_smirk_forward = (
                parent_smirk.split(">>")[1] + ">>" + parent_smirk.split(">>")[0]
            )
            applied_smirks_names.append(
                self._smirks_name_dictionary[applied_smirk_forward]
            )

        return {
            "mapping": possible_mappings[0],
            "additional_info": applied_smirks_names,
        }

    def map_reactions(self, reaction_list: List[str]) -> List[ReactionMapperResult]:
        """ """

        mapped_reactions = []
        for reaction in reaction_list:
            mapped_reactions.append(self.map_reaction(reaction))
        return mapped_reactions
