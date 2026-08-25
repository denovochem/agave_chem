from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.types import ReactionMapperResult
from agave_chem.utils.chem_utils import canonicalize_smiles
from agave_chem.utils.logging_config import logger


class IdenticalFragmentMapper(ReactionMapper):
    """
    Reaction mapper that identifies and atom-maps fragments appearing
    identically on both sides of a reaction.

    Fragments that differ only by charge state (e.g. ``N`` vs ``[NH4+]``)
    are detected as identical by comparing charge-neutral canonical SMILES.
    """

    def __init__(self, mapper_name: str, mapper_weight: float = 1):
        super().__init__("identical_fragment", mapper_name, mapper_weight)

        self.uncharger = rdMolStandardize.Uncharger()

    def _charge_neutral_canonicalization(self, smiles: str) -> str:
        """
        Return the charge-neutral canonical SMILES for a fragment.

        Uncharges the molecule using ``rdMolStandardize.Uncharger``, then
        canonicalizes without tautomer enumeration.

        Args:
            smiles (str): A SMILES string.

        Returns:
            str: Charge-neutral canonical SMILES, or the original string on failure.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            uncharged_mol = self.uncharger.uncharge(mol)
            uncharged_smiles = Chem.MolToSmiles(uncharged_mol)
            return canonicalize_smiles(uncharged_smiles, canonicalize_tautomer=False)
        except Exception as e:
            logger.warning(f"Could not charge-neutralize {smiles}: {e}")
            return smiles

    def _atom_map_identical_fragments(
        self, reaction_smiles: str
    ) -> Tuple[List[Tuple[str, str]], str]:
        """
        Atom map identical fragments in a reaction SMILES string.

        Fragments are compared using charge-neutral canonical SMILES (without
        tautomer canonicalization) so that fragments differing only by charge
        state (e.g. ``N`` vs ``[NH4+]``) are detected as identical.

        Args:
            reaction_smiles (str): A reaction SMILES string.

        Returns:
            Tuple[List[Tuple[str, str]], str]:
                - First element: A list of ``(reactant_smiles, product_smiles)``
                  pairs for each matched identical fragment. For truly identical
                  fragments both elements are the same string; for charge-different
                  pairs they differ.
                - Second element: The remaining reaction SMILES with identical
                  fragments removed from both sides.
        """
        reactants, products = self._split_reaction_components(reaction_smiles)

        reactants_smiles_list = reactants.split(".")
        products_smiles_list = products.split(".")

        # Build lists of (canonical_key, original_smiles) to allow duplicates
        reactant_keys = [
            (self._charge_neutral_canonicalization(r), r) for r in reactants_smiles_list
        ]
        product_keys = [
            (self._charge_neutral_canonicalization(p), p) for p in products_smiles_list
        ]

        atom_mapped_identical_reactants_products = []
        atom_map_num = 500

        matched_reactant_indices: set[int] = set()
        matched_product_indices: set[int] = set()

        for ri, (r_key, reactant_orig_smiles) in enumerate(reactant_keys):
            if ri in matched_reactant_indices:
                continue
            for pi, (p_key, product_orig_smiles) in enumerate(product_keys):
                if pi in matched_product_indices:
                    continue
                if r_key == p_key:
                    r_mol = Chem.MolFromSmiles(reactant_orig_smiles)
                    p_mol = Chem.MolFromSmiles(product_orig_smiles)
                    if r_mol is None or p_mol is None:
                        continue
                    r_neut = self.uncharger.uncharge(r_mol)
                    p_neut = self.uncharger.uncharge(p_mol)

                    match = p_neut.GetSubstructMatch(r_neut)
                    if not match:
                        continue

                    for i, atom in enumerate(r_mol.GetAtoms()):
                        atom.SetAtomMapNum(atom_map_num + i)
                        p_mol.GetAtomWithIdx(match[i]).SetAtomMapNum(atom_map_num + i)
                    atom_map_num += r_mol.GetNumAtoms()

                    mapped_r = Chem.MolToSmiles(r_mol)
                    mapped_p = Chem.MolToSmiles(p_mol)
                    atom_mapped_identical_reactants_products.append(
                        (mapped_r, mapped_p)
                    )
                    matched_reactant_indices.add(ri)
                    matched_product_indices.add(pi)
                    break

        remaining_reactants = [
            r
            for ri, r in enumerate(reactants_smiles_list)
            if ri not in matched_reactant_indices
        ]
        remaining_products = [
            p
            for pi, p in enumerate(products_smiles_list)
            if pi not in matched_product_indices
        ]

        return (
            atom_mapped_identical_reactants_products,
            ".".join(remaining_reactants) + ">>" + ".".join(remaining_products),
        )

    def _add_identical_fragments_to_mapping(
        self,
        mapped_reaction_smiles: str,
        atom_mapped_identical_reactants_products: List[Tuple[str, str]],
    ) -> str:
        """
        Add identical fragments back to a mapped reaction SMILES string.

        Args:
            mapped_reaction_smiles (str): A mapped reaction SMILES string.
            atom_mapped_identical_reactants_products (List[Tuple[str, str]]): A
                list of ``(reactant_smiles, product_smiles)`` pairs to append to
                the reactant and product sides respectively.

        Returns:
            str: A mapped reaction SMILES string with identical fragments added.
        """
        reactants, products = self._split_reaction_components(mapped_reaction_smiles)

        reactants_smiles_list = reactants.split(".")
        products_smiles_list = products.split(".")

        for identical_fragments in atom_mapped_identical_reactants_products:
            reactants_smiles_list.append(identical_fragments[0])
            products_smiles_list.append(identical_fragments[1])

        mapped_reactants = ".".join(reactants_smiles_list)
        mapped_products = ".".join(products_smiles_list)

        return mapped_reactants + ">>" + mapped_products

    def create_identical_fragments_mapping_list(
        self,
        reaction_smiles_list: List[str],
    ) -> Tuple[List[str], List[List[Tuple[str, str]]]]:
        """
        Strip identical fragments from a list of reactions for downstream mapping.

        Args:
            reaction_smiles_list (List[str]): A list of reaction SMILES strings.

        Returns:
            Tuple[List[str], List[List[Tuple[str, str]]]]:
                - First element: Reaction SMILES with identical fragments removed.
                - Second element: Per-reaction lists of ``(reactant_smiles,
                  product_smiles)`` pairs to be re-added after downstream mapping.
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
        identical_fragments_mapping_list: List[List[Tuple[str, str]]],
    ) -> List[str]:
        """
        Re-add identical fragments to a list of already-mapped reaction SMILES.

        Args:
            mapped_reaction_smiles_list (List[str]): Mapped reaction SMILES strings
                (from a downstream mapper).
            identical_fragments_mapping_list (List[List[Tuple[str, str]]]):
                Per-reaction lists of ``(reactant_smiles, product_smiles)`` pairs
                (produced by ``create_identical_fragments_mapping_list``).

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
