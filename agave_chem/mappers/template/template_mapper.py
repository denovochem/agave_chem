import json
from importlib.resources import files
from typing import Any, Dict, List, Tuple, TypedDict

from rdchiral import main as rdc
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from agave_chem.mappers.reaction_mapper import ReactionMapper, ReactionMapperResult
from agave_chem.mappers.template.template_initialization import (
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
    products: List[Chem.Mol]
    reactants: List[Chem.Mol]
    rdchiral_reactants: Any
    tautomer_smiles_dict: Dict[str, List[str]]
    fragment_count_dict: Dict[str, int]


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
        self._initialized_smirks_patterns = initialize_template_data(
            self._smirks_patterns
        )

        self._tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
        self._tautomer_enumerator.SetMaxTransforms(max_transforms)
        self._tautomer_enumerator.SetMaxTautomers(max_tautomers)

    def _validate_smirks_patterns(self, smirks_patterns: List[SmirksPattern]) -> None:
        """Validates SMIRKS patterns."""
        pass

    def _prepare_reaction_data(
        self, reactants_str: str, products_str: str
    ) -> ReactionData:
        """
        Prepares reaction data for reaction mapping.

        Args:
            reactants_str (str): Reactants SMILES string
            products_str (str): Products SMILES string

        Returns:
            List: A list containing the reactants and products as RDKit Mol objects, the RDChiral reaction object, a dictionary of enumerated tautomer SMILES strings, and a dictionary of fragment counts.
        """

        return {
            "products": [
                Chem.MolFromSmiles(product_str)
                for product_str in products_str.split(".")
            ],
            "reactants": [
                Chem.MolFromSmiles(reactant_str)
                for reactant_str in reactants_str.split(".")
            ],
            "rdchiral_reactants": rdc.rdchiralReactants(products_str),
            "tautomer_smiles_dict": self._enumerate_tautomer_smiles(reactants_str),
            "fragment_count_dict": self._get_fragment_count_dict(reactants_str),
        }

    def _enumerate_tautomer_smiles(self, smiles: str) -> Dict[str, List[str]]:
        """

        Enumerate tautomer SMILES strings for a given SMILES string.

        Args:
            smiles (str): A SMILES string

        Returns:
            Dict[str, List]: A dictionary where the keys are the fragments of the input SMILES string and the values are lists of the enumerated tautomer SMILES strings.

        """
        enumerated_smiles_dict = {}
        for fragment in smiles.split("."):
            mol = Chem.MolFromSmiles(fragment)

            if mol is None:
                enumerated_smiles_dict[fragment] = []
                continue

            enumerated_fragment_mols = list(self._tautomer_enumerator.Enumerate(mol))
            enumerated_fragment_smiles = [
                Chem.MolToSmiles(frag_mol) for frag_mol in enumerated_fragment_mols
            ]
            enumerated_fragment_smiles.append(fragment)
            canonicalized_enumerated_fragment_smiles = [
                canonicalize_smiles(frag_smiles)
                for frag_smiles in enumerated_fragment_smiles
                if frag_smiles
            ]
            enumerated_smiles_dict[fragment] = list(
                set(canonicalized_enumerated_fragment_smiles)
            )

        return enumerated_smiles_dict

    def _get_fragment_count_dict(self, smiles: str) -> Dict[str, int]:
        """

        Returns a dictionary where the keys are the fragments of the input SMILES string and the values are the counts of each fragment.

        Args:
            smiles (str): A SMILES string

        Returns:
            Dict[str, int]: A dictionary where the keys are the fragments of the input SMILES string and the values are the counts of each fragment.

        """
        fragment_count_dict = {}
        for fragment in smiles.split("."):
            if fragment not in fragment_count_dict:
                fragment_count_dict[fragment] = 1
            else:
                fragment_count_dict[fragment] += 1

        return fragment_count_dict

    def _process_templates(self, reaction_smiles_data: ReactionData) -> Dict[str, List]:
        """
        Process templates for a given reaction.

        Args:
            reaction_smiles_data (List): A list containing the reactants and products as RDKit Mol objects, the RDChiral reaction object, a dictionary of enumerated tautomer SMILES strings, and a dictionary of fragment counts.

        Returns:
            dict: A dictionary where the keys are the outcomes of the reaction and the values are lists of the applied SMIRKS patterns.

        """
        mapped_outcomes_smirks_dict: Dict[str, List] = {}

        atom_mapped_product = self._get_mapped_product(reaction_smiles_data)
        outcomes_and_applied_smirks = self._apply_templates(reaction_smiles_data)

        for outcome_and_applied_smirk in outcomes_and_applied_smirks:
            mapped_outcomes_smirks_dict = self._process_single_outcome(
                outcome_and_applied_smirk[0],
                outcome_and_applied_smirk[1],
                reaction_smiles_data["tautomer_smiles_dict"],
                reaction_smiles_data["fragment_count_dict"],
                atom_mapped_product,
                mapped_outcomes_smirks_dict,
            )

        return mapped_outcomes_smirks_dict

    def _get_mapped_product(self, reaction_smiles_data: ReactionData) -> str:
        """Get the mapped product SMILES string for a given reaction.

        Args:
            reaction_smiles_data (List): A list containing the reactants and products as RDKit Mol objects, the RDChiral reaction object, a dictionary of enumerated tautomer SMILES strings, and a dictionary of fragment counts.

        Returns:
            str: The mapped product SMILES string.

        """
        rdc_reactants = reaction_smiles_data["rdchiral_reactants"]

        rdc_mol = rdc_reactants.reactants
        for atom in rdc_mol.GetAtoms():
            atom.SetAtomMapNum(rdc_reactants.idx_to_mapnum(atom.GetIdx()))
        mapped_product = Chem.MolToSmiles(rdc_mol)

        return mapped_product

    def _apply_templates(self, reaction_smiles_data: ReactionData) -> List[List[Any]]:
        """ """
        product_mols = reaction_smiles_data["products"]
        reactant_mols = reaction_smiles_data["reactants"]
        rdc_reactants = reaction_smiles_data["rdchiral_reactants"]

        outcomes_and_applied_smirks = []

        for template in self._initialized_smirks_patterns:
            products_smarts = template[0]
            reactant_smarts = template[1]
            rdc_rxn = template[2]

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

            try:
                outcomes = rdc.rdchiralRun(rdc_rxn, rdc_reactants, return_mapped=True)
                outcomes_and_applied_smirks.append([outcomes, template])
            except Exception as e:
                logger.warning(f"Error applying templates: {e}")
                pass

        return outcomes_and_applied_smirks

    def _remove_spectator_mappings(self, smiles: str) -> str:
        """ """
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
        rdc_outcome: Tuple[List[str], Dict[str, List[str]]],
        applied_smirk: List[str],
        unmapped_reactants_tautomers_dict: Dict[str, List[str]],
        fragment_count_dict: Dict[str, int],
        atom_mapped_product: str,
        mapped_outcomes_smirks_dict: Dict[str, List],
    ) -> Dict[str, List[Any]]:
        reactants_list, atom_mapped_reactants_dict = rdc_outcome

        for reactant in reactants_list:
            if reactant not in atom_mapped_reactants_dict:
                continue

            mapped_outcome = self._remove_spectator_mappings(
                atom_mapped_reactants_dict[reactant][0]
            )

            missing_fragments, found_fragments = self._find_missing_fragments(
                mapped_outcome, unmapped_reactants_tautomers_dict
            )

            if len(missing_fragments) != 0:
                fragment_mapped_dict = self._handle_missing_fragments(
                    missing_fragments,
                    found_fragments,
                    unmapped_reactants_tautomers_dict,
                    mapped_outcomes_smirks_dict,
                )

                if len(fragment_mapped_dict) != len(missing_fragments):
                    continue

                fragment_not_found = False
                for k, v in fragment_mapped_dict.items():
                    if k not in mapped_outcome:
                        fragment_not_found = True
                        break
                    mapped_outcome = mapped_outcome.replace(k, v)
                if fragment_not_found:
                    continue

            unmapped_canonical_smiles_for_mapped_smiles = [
                canonicalize_smiles(ele) for ele in mapped_outcome.split(".")
            ]

            spectators = []
            for ele, ele_count in fragment_count_dict.items():
                canonicalized_ele = canonicalize_smiles(ele)
                num_occurrences_mapped = (
                    unmapped_canonical_smiles_for_mapped_smiles.count(canonicalized_ele)
                )
                dif_num_occurrences = ele_count - num_occurrences_mapped
                if dif_num_occurrences > 0:
                    spectators.extend([canonicalized_ele] * dif_num_occurrences)

            reactants = mapped_outcome.split(".")
            reactants_and_spectators = reactants + spectators

            finalized_reaction_smiles = (
                ".".join(reactants_and_spectators) + ">>" + atom_mapped_product
            )

            mapped_outcomes_smirks_dict[finalized_reaction_smiles] = applied_smirk

        return mapped_outcomes_smirks_dict

    def _find_missing_fragments(
        self, mapped_outcome: str, unmapped_reactants: Dict[str, List[str]]
    ) -> Tuple[List, List]:
        missing_fragments = []
        found_fragments = []
        reactant_fragments = list(unmapped_reactants.values())
        flattened_reactant_fragments = [
            item for sublist in reactant_fragments for item in sublist
        ]

        for mapped_fragment in mapped_outcome.split("."):
            unmapped_fragment = canonicalize_smiles(mapped_fragment)
            if unmapped_fragment not in flattened_reactant_fragments:
                missing_fragments.append([unmapped_fragment, mapped_fragment])
            else:
                found_fragments.append([unmapped_fragment, mapped_fragment])

        return missing_fragments, found_fragments

    def _handle_missing_fragments(
        self,
        missing_fragments: List,
        found_fragments: List,
        unmapped_reactants: Dict[str, List[str]],
        mapped_outcomes_smirks_dict: Dict[str, List],
    ) -> Dict[str, str]:
        all_fragments_substructs = self._are_fragments_substructures(
            missing_fragments, found_fragments, unmapped_reactants
        )
        if not all_fragments_substructs:
            return mapped_outcomes_smirks_dict

        fragment_mapped_dict = self._identify_and_map_fragments(
            missing_fragments,
            found_fragments,
            unmapped_reactants,
        )

        for k, v in fragment_mapped_dict.items():
            if len(v) > 1:
                logger.warning(
                    "Multiple possible fragments identified for reaction SMARTS substructure"
                )
                return mapped_outcomes_smirks_dict
            fragment_mapped_dict[k] = v[0]

        return fragment_mapped_dict

    def _are_fragments_substructures(
        self,
        missing_fragments: List,
        found_fragments: List,
        unmapped_reactants: Dict[str, List[str]],
    ) -> bool:
        unmapped_found_fragments = [ele[0] for ele in found_fragments]
        for fragment_str, _ in missing_fragments:
            if "*" not in fragment_str:
                continue

            query_mol = Chem.MolFromSmarts(fragment_str)
            if not query_mol:
                return False

            found_match = False
            found_matches = []
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
                        found_matches.append(reactant_fragment_str)

            if not found_match:
                return False

        return True

    def _identify_and_map_fragments(
        self,
        missing_fragments: List,
        found_fragments: List,
        unmapped_reactants: Dict[str, List[str]],
    ) -> Dict[Any, List[str]]:
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

    def map_reaction(self, reaction_smiles: str) -> ReactionMapperResult:
        """
        Maps atoms between reactants and products in a chemical reaction.

        This function takes a reaction SMILES string and attempts to create a mapping between
        atoms in the reactants and products using a library of named reactions. It processes
        the reaction using RDKit and RDChiral, assigns stereochemistry, and generates atom mappings.

        Args:
            reaction_smiles (str): A SMILES string representing a chemical reaction in the format
                "reactants>>products"

        Returns:
            dict: XXX.

        Example:
            >>> mapper = ReactionMapper()
            >>> mapped = mapper.map_reaction("CC(=O)O.CN>>CC(=O)NC")
            '[CH3:1][C:2](=[O:3])[OH:4].[NH2:5][CH3:6]>>[CH3:1][NH:2][C:3]([CH3:5])=[O:6]'
        """
        default_mapping_dict: ReactionMapperResult = {
            "mapping": "",
            "additional_info": [{}],
        }
        if not self._reaction_smiles_valid(reaction_smiles):
            return default_mapping_dict

        canonicalized_reaction_smiles = canonicalize_reaction_smiles(
            reaction_smiles, canonicalize_tautomer=True
        )
        reactants_str, products_str = self._split_reaction_components(
            canonicalized_reaction_smiles
        )

        reaction_data = self._prepare_reaction_data(reactants_str, products_str)

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
        for applied_smirk_data in list(mapped_outcomes_smirks_dict.values()):
            applied_smirk = applied_smirk_data[-2]
            applied_smirk_forward = (
                applied_smirk.split(">>")[1] + ">>" + applied_smirk.split(">>")[0]
            )
            applied_smirks_names.append(
                self._smirks_name_dictionary[applied_smirk_forward]
            )

        return {
            "mapping": possible_mappings[0],
            "additional_info": applied_smirks_names,
        }

    def map_reactions(self, reaction_list: List[str]) -> List[ReactionMapperResult]:
        """
        Map a list of reaction strings.

        Args:
            reaction_list: List of reaction representations to map. Each entry is
                passed to :meth:`map_reaction`.

        Returns:
            A list of :class:`ReactionMapperResult` objects, one per input reaction,
            in the same order.
        """

        mapped_reactions = []
        for reaction in reaction_list:
            mapped_reactions.append(self.map_reaction(reaction))
        return mapped_reactions
