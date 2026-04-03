from typing import List, Tuple

from rdkit import Chem

from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.types import ReactionMapperResult
from agave_chem.utils.chem_utils import canonicalize_smiles
from agave_chem.utils.logging_config import logger


class IdenticalFragmentMapper(ReactionMapper):
    """
    Reaction mapper that identifies and atom-maps fragments appearing
    identically on both sides of a reaction.
    """

    def __init__(self, mapper_name: str, mapper_weight: float = 1):
        super().__init__("identical_fragment", mapper_name, mapper_weight)

    def _atom_map_identical_fragments(
        self, reaction_smiles: str
    ) -> Tuple[List[str], str]:
        """
        Atom map identical fragments in a reaction SMILES string.

        Args:
            reaction_smiles (str): A reaction SMILES string.

        Returns:
            Tuple[List[str], str]:
                - First element: A list of mapped identical fragment SMILES.
                - Second element: The remaining reaction SMILES with identical
                  fragments removed from both sides.
        """
        reactants, products = self._split_reaction_components(reaction_smiles)

        reactants_smiles_list = reactants.split(".")
        products_smiles_list = products.split(".")

        reactants_smiles_list_mapping_dict = {
            canonicalize_smiles(reactant): reactant
            for reactant in reactants_smiles_list
        }

        canonicalized_reactants_smiles_list = [
            canonicalize_smiles(smiles) for smiles in reactants_smiles_list
        ]
        canonicalized_products_smiles_list = [
            canonicalize_smiles(smiles) for smiles in products_smiles_list
        ]

        atom_mapped_identical_reactants_products = []
        atom_map_num = 500
        for canonicalized_reactant in canonicalized_reactants_smiles_list:
            reactant = reactants_smiles_list_mapping_dict[canonicalized_reactant]
            if canonicalized_reactant in canonicalized_products_smiles_list:
                reactants_smiles_list.remove(reactant)
                products_smiles_list.remove(reactant)
                reactant_mol = Chem.MolFromSmiles(canonicalized_reactant)
                for atom in reactant_mol.GetAtoms():
                    atom.SetAtomMapNum(atom_map_num)
                    atom_map_num += 1
                mapped_reactant = Chem.MolToSmiles(reactant_mol)
                atom_mapped_identical_reactants_products.append(mapped_reactant)
        return (
            atom_mapped_identical_reactants_products,
            ".".join(reactants_smiles_list) + ">>" + ".".join(products_smiles_list),
        )

    def _add_identical_fragments_to_mapping(
        self,
        mapped_reaction_smiles: str,
        atom_mapped_identical_reactants_products: List[str],
    ) -> str:
        """
        Add identical fragments back to a mapped reaction SMILES string.

        Args:
            mapped_reaction_smiles (str): A mapped reaction SMILES string.
            atom_mapped_identical_reactants_products (List[str]): A list of
                atom-mapped identical fragment SMILES to append to both sides.

        Returns:
            str: A mapped reaction SMILES string with identical fragments added.
        """
        reactants, products = self._split_reaction_components(mapped_reaction_smiles)

        reactants_smiles_list = reactants.split(".")
        products_smiles_list = products.split(".")

        for identical_fragment in atom_mapped_identical_reactants_products:
            reactants_smiles_list.append(identical_fragment)
            products_smiles_list.append(identical_fragment)

        mapped_reactants = ".".join(reactants_smiles_list)
        mapped_products = ".".join(products_smiles_list)

        return mapped_reactants + ">>" + mapped_products

    def create_identical_fragments_mapping_list(
        self,
        reaction_smiles_list: List[str],
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Strip identical fragments from a list of reactions for downstream mapping.

        Args:
            reaction_smiles_list (List[str]): A list of reaction SMILES strings.

        Returns:
            Tuple[List[str], List[List[str]]]:
                - First element: Reaction SMILES with identical fragments removed.
                - Second element: Per-reaction lists of atom-mapped identical
                  fragment SMILES to be re-added after downstream mapping.
        """
        new_rxns = []
        identical_fragments_mapping_list = []
        for reaction_smiles in reaction_smiles_list:
            atom_mapped_identical_fragments, new_rxn = (
                self._atom_map_identical_fragments(reaction_smiles)
            )
            identical_fragments_mapping_list.append(atom_mapped_identical_fragments)
            new_rxns.append(new_rxn)
        return new_rxns, identical_fragments_mapping_list

    def resolve_identical_fragments_mapping_list(
        self,
        mapped_reaction_smiles_list: List[str],
        identical_fragments_mapping_list: List[List[str]],
    ) -> List[str]:
        """
        Re-add identical fragments to a list of already-mapped reaction SMILES.

        Args:
            mapped_reaction_smiles_list (List[str]): Mapped reaction SMILES strings
                (from a downstream mapper).
            identical_fragments_mapping_list (List[List[str]]): Per-reaction lists of
                atom-mapped identical fragment SMILES (produced by
                ``create_identical_fragments_mapping_list``).

        Returns:
            List[str]: Final reaction SMILES strings with identical fragments restored.
        """
        final_reactions = []
        for mapped_reaction_smiles, identical_fragments_mapping in zip(
            mapped_reaction_smiles_list, identical_fragments_mapping_list
        ):
            final_reactions.append(
                self._add_identical_fragments_to_mapping(
                    mapped_reaction_smiles, identical_fragments_mapping
                )
            )
        return final_reactions

    def map_reaction(self, reaction_smiles: str) -> ReactionMapperResult:
        """
        Map a single reaction by atom-mapping its identical fragments.

        Args:
            reaction_smiles (str): A reaction SMILES string.

        Returns:
            ReactionMapperResult: Mapping result. If the input is invalid, an empty
                default result is returned.
        """
        if not self._reaction_smiles_valid(reaction_smiles):
            return self._return_default_mapping_dict(reaction_smiles)

        atom_mapped_fragments, remaining_rxn = self._atom_map_identical_fragments(
            reaction_smiles
        )

        if atom_mapped_fragments:
            mapped_reaction_smiles = self._add_identical_fragments_to_mapping(
                remaining_rxn, atom_mapped_fragments
            )
        else:
            mapped_reaction_smiles = reaction_smiles

        if not self._verify_validity_of_mapping(
            mapped_reaction_smiles, expect_full_mapping=False
        ):
            logger.warning("Invalid mapping")
            return self._return_default_mapping_dict(reaction_smiles)

        return ReactionMapperResult(
            original_smiles=reaction_smiles,
            selected_mapping=mapped_reaction_smiles,
            possible_mappings={},
            mapping_type=self._mapper_type,
            mapping_score=None,
            additional_info=[{}],
        )

    def map_reactions(self, reaction_list: List[str]) -> List[ReactionMapperResult]:
        """
        Map a list of reaction SMILES strings using the identical-fragment mapper.

        Args:
            reaction_list (List[str]): List of reaction SMILES strings to map.

        Returns:
            List[ReactionMapperResult]: The mapping results in the same order as the
                input reactions.
        """
        mapped_reactions: List[ReactionMapperResult] = []
        for reaction in reaction_list:
            mapped_reactions.append(self.map_reaction(reaction))
        return mapped_reactions
