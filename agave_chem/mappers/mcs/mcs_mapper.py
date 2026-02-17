from typing import Any, Dict, List, Tuple

from rdkit import Chem

from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.utils.chem_utils import (
    canonicalize_reaction_smiles,
)


class MCSReactionMapper(ReactionMapper):
    """
    MCS reaction classification and atom-mapping
    """

    def __init__(self, mapper_name: str, mapper_weight: float = 3):
        super().__init__("mcs", mapper_name, mapper_weight)

    def _get_atoms_in_radius(
        self, mol: Chem.Mol, atom: Chem.Atom, radius: int
    ) -> List[list]:
        bond_ids = Chem.rdmolops.FindAtomEnvironmentOfRadiusN(
            mol,
            radius=radius,
            rootedAtAtom=atom.GetIdx(),
        )

        if not bond_ids:
            return []

        # Build fragment from the bond environment and get parent->sub atom mapping
        amap: Dict[int, int] = {}  # parent_atom_idx -> submol_atom_idx
        submol = Chem.PathToSubmol(mol, bond_ids, atomMap=amap)

        root_parent = atom.GetIdx()
        root_sub = amap[root_parent]

        # Root the fragment canonicalization using atom-map numbers
        # (do this on the submol only, so you don't mutate the parent mol)
        for a in submol.GetAtoms():
            a.SetAtomMapNum(0)
        submol.GetAtomWithIdx(root_sub).SetAtomMapNum(1)

        # Canonical ranks WITHIN the fragment
        # If your RDKit doesn't accept includeAtomMaps, tell me the exact signature error text.
        ranks = list(
            Chem.CanonicalRankAtoms(submol, breakTies=True, includeAtomMaps=True)
        )

        # Invert mapping: submol_atom_idx -> parent_atom_idx
        inv_amap = {sub_i: parent_i for parent_i, sub_i in amap.items()}

        # Canonical bond ordering inside the fragment
        bond_items = []
        for sb in submol.GetBonds():
            sa1, sa2 = sb.GetBeginAtomIdx(), sb.GetEndAtomIdx()
            r1, r2 = ranks[sa1], ranks[sa2]
            lo, hi = (r1, r2) if r1 <= r2 else (r2, r1)

            key = (
                lo,
                hi,
                int(sb.GetBondTypeAsDouble()),
                int(sb.GetIsAromatic()),
                int(sb.IsInRing()),
            )
            bond_items.append((key, sa1, sa2))

        bond_items.sort(key=lambda x: x[0])

        encoded_environment = []
        for _, sa1, sa2 in bond_items:
            # Choose a stable direction using fragment ranks
            if ranks[sa1] <= ranks[sa2]:
                s_begin, s_end = sa1, sa2
            else:
                s_begin, s_end = sa2, sa1

            p_begin = inv_amap[s_begin]
            p_end = inv_amap[s_end]

            pbond = mol.GetBondBetweenAtoms(p_begin, p_end)
            pbidx = pbond.GetIdx()

            encoded_environment.append(
                [
                    self._encode_atom(mol, p_begin),
                    self._encode_bond(mol, pbidx),
                    self._encode_atom(mol, p_end),
                ]
            )

        return encoded_environment

    def _encode_atom(self, mol: Chem.Mol, idx: int) -> List[int]:
        a = mol.GetAtomWithIdx(idx)
        z = a.GetAtomicNum()
        chg = a.GetFormalCharge()
        arom = 1 if a.GetIsAromatic() else 0
        ring = 1 if a.IsInRing() else 0
        h = a.GetTotalNumHs()
        atom_map_num = a.GetAtomMapNum()
        return [z, chg, arom, ring, h, atom_map_num, idx]

    def _encode_bond(self, mol: Chem.Mol, idx: int) -> List[int]:
        b = mol.GetBondWithIdx(idx)
        # getBondDir??
        return [int(b.GetBondTypeAsDouble() * 10)]

    def _assign_mapping(self, mol: Chem.Mol, atom_idx: int, atom_map_num: int) -> None:
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(atom_map_num)
        return None

    def _remove_atom_idx_for_comparison(
        self, atom_map_list: List[List[List[int]]]
    ) -> List[List[List[int]]]:
        new_list = []
        for init_list in atom_map_list:
            atom_bond_list = []
            for sub_list in init_list:
                if len(sub_list) == 7:
                    atom_bond_list.append(sub_list[:-1])
                else:
                    atom_bond_list.append(sub_list)

            new_list.append(atom_bond_list)
        return new_list

    def _get_num_matching_atom_envs(
        self,
        reactant_encoding_dict: Dict[str, List[List[List[int]]]],
        product_encoding_dict: Dict[str, List[List[List[int]]]],
        radius: int,
    ) -> List[List[str]]:
        matches = []

        reactant_encoding_dict_no_idx = {}
        for k, v in reactant_encoding_dict.items():
            if str(radius) != k.split("_")[-1]:
                continue
            if not v:
                continue
            reactant_encoding_dict_no_idx[k] = self._remove_atom_idx_for_comparison(v)

        product_encoding_dict_no_idx = {}
        for k, v in product_encoding_dict.items():
            if str(radius) != k.split("_")[-1]:
                continue
            if not v:
                continue
            product_encoding_dict_no_idx[k] = self._remove_atom_idx_for_comparison(v)

        for k1, v1 in reactant_encoding_dict_no_idx.items():
            for k2, v2 in product_encoding_dict_no_idx.items():
                if v1 == v2:
                    matches.append([k1, k2])

        return matches

    def _assign_atom_mapping(
        self,
        matches: List[List[str]],
        reactants_mols: List[Chem.Mol],
        products_mols: List[Chem.Mol],
        reactant_encoding_dict: Dict[str, List[List[List[int]]]],
        product_encoding_dict: Dict[str, List[List[List[int]]]],
        atom_map_num: int,
    ) -> Tuple[int, Dict[str, List[List[List[int]]]], Dict[str, List[List[List[int]]]]]:
        reactant_mol_idx = int(matches[0][0].split("_")[0])
        reactant_atom_idx = int(matches[0][0].split("_")[1])
        if (
            reactants_mols[reactant_mol_idx]
            .GetAtomWithIdx(reactant_atom_idx)
            .GetAtomMapNum()
            != 0
        ):
            return atom_map_num, reactant_encoding_dict, product_encoding_dict
        self._assign_mapping(
            reactants_mols[reactant_mol_idx], reactant_atom_idx, atom_map_num
        )

        product_mol_idx = int(matches[0][1].split("_")[0])
        product_atom_idx = int(matches[0][1].split("_")[1])
        if (
            products_mols[product_mol_idx]
            .GetAtomWithIdx(product_atom_idx)
            .GetAtomMapNum()
            != 0
        ):
            return atom_map_num, reactant_encoding_dict, product_encoding_dict
        self._assign_mapping(
            products_mols[product_mol_idx], product_atom_idx, atom_map_num
        )

        keys_to_delete = []
        for k1, v1 in reactant_encoding_dict.items():
            if int(k1.split("_")[0]) != reactant_mol_idx:
                continue
            if int(k1.split("_")[1]) == reactant_atom_idx:
                keys_to_delete.append(k1)
            for sub_v1 in v1:
                for sub_sub_v1 in sub_v1:
                    if len(sub_sub_v1) == 7:
                        if sub_sub_v1[-1] == reactant_atom_idx:
                            sub_sub_v1[-2] = 1
        for key in keys_to_delete:
            del reactant_encoding_dict[key]

        keys_to_delete = []
        for k1, v1 in product_encoding_dict.items():
            if int(k1.split("_")[0]) != product_mol_idx:
                continue
            if int(k1.split("_")[1]) == product_atom_idx:
                keys_to_delete.append(k1)
            for sub_v1 in v1:
                for sub_sub_v1 in sub_v1:
                    if len(sub_sub_v1) == 7:
                        if sub_sub_v1[-1] == product_atom_idx:
                            sub_sub_v1[-2] = 1
        for key in keys_to_delete:
            del product_encoding_dict[key]

        return atom_map_num + 1, reactant_encoding_dict, product_encoding_dict

    def map_reaction(self, reaction_smiles: str, min_radius: int = 1) -> Dict[str, Any]:
        default_mapping_dict = {"mapping": "", "additional_info": [{}]}
        if not self._reaction_smiles_valid(reaction_smiles):
            return default_mapping_dict

        canonicalized_reaction_smiles = canonicalize_reaction_smiles(
            reaction_smiles, canonicalize_tautomer=True
        )
        reactants, products = self._split_reaction_components(
            canonicalized_reaction_smiles
        )

        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]

        largest_num_atoms = max(
            [len(mol.GetAtoms()) for mol in reactant_mols + product_mols]
        )

        reactant_encoding_dict = {}
        product_encoding_dict = {}

        for radius in range(min_radius, largest_num_atoms):
            for i, reactant_mol in enumerate(reactant_mols):
                if len(reactant_mol.GetAtoms()) < radius:
                    continue
                for atom in reactant_mol.GetAtoms():
                    reactant_encoding_dict[
                        str(i) + "_" + str(atom.GetIdx()) + "_" + str(radius)
                    ] = self._get_atoms_in_radius(reactant_mol, atom, radius)

            for i, product_mol in enumerate(product_mols):
                if len(product_mol.GetAtoms()) < radius:
                    continue
                for atom in product_mol.GetAtoms():
                    product_encoding_dict[
                        str(i) + "_" + str(atom.GetIdx()) + "_" + str(radius)
                    ] = self._get_atoms_in_radius(product_mol, atom, radius)

            matches = self._get_num_matching_atom_envs(
                reactant_encoding_dict,
                product_encoding_dict,
                radius,
            )
            if len(matches) == 1:
                final_radius = radius
                break
            if len(matches) == 0:
                final_radius = radius - 1
                break

        atom_map_num = 1
        for radius in range(final_radius, 0, -1):
            matches = self._get_num_matching_atom_envs(
                reactant_encoding_dict,
                product_encoding_dict,
                radius,
            )
            num_matches = len(matches)
            for _ in range(num_matches):
                atom_map_num, reactant_encoding_dict, product_encoding_dict = (
                    self._assign_atom_mapping(
                        [matches[0]],
                        reactant_mols,
                        product_mols,
                        reactant_encoding_dict,
                        product_encoding_dict,
                        atom_map_num,
                    )
                )
                matches = self._get_num_matching_atom_envs(
                    reactant_encoding_dict,
                    product_encoding_dict,
                    radius,
                )
                if len(matches) == 0:
                    break

        mapped_reactant_smiles = ".".join(
            [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in reactant_mols]
        )
        mapped_product_smiles = ".".join(
            [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in product_mols]
        )
        mapped_reaction_smiles = mapped_reactant_smiles + ">>" + mapped_product_smiles

        # mapped_reaction_smiles = canonicalize_atom_mapping(
        #     canonicalize_reaction_smiles(
        #         mapped_reaction_smiles, canonicalize_tautomer=True, remove_mapping=False
        #     )
        # )

        return {"mapping": mapped_reaction_smiles, "additional_info": [{}]}

    def map_reactions(self, reaction_list: List[str]) -> List[Dict[str, Any]]:
        """ """

        mapped_reactions = []
        for reaction in reaction_list:
            mapped_reactions.append(self.map_reaction(reaction))
        return mapped_reactions
