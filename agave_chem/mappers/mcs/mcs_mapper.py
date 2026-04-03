from typing import Dict, List, Tuple

from rdkit import Chem

from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.types import ReactionMapperResult
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
        """
        Encode the bond environment within a radius of a root atom.

        Args:
            mol (Chem.Mol): RDKit molecule containing the environment.
            atom (Chem.Atom): Root atom in `mol` to build the environment around.
            radius (int): Bond-radius to use when extracting the local environment.

        Returns:
            List[List[List[str | int]]]: A deterministic, sorted encoding of the
            radius-limited bond environment. Each element is a 3-item list of
            `[encoded_atom_begin, encoded_bond, encoded_atom_end]`. Returns an
            empty list when no bonds are found within the given radius.
        """
        bond_ids = Chem.rdmolops.FindAtomEnvironmentOfRadiusN(
            mol,
            radius=radius,
            rootedAtAtom=atom.GetIdx(),
        )

        if not bond_ids:
            return []

        amap: Dict[int, int] = {}
        submol = Chem.PathToSubmol(mol, bond_ids, atomMap=amap)

        root_parent = atom.GetIdx()
        root_sub = amap[root_parent]

        for a in submol.GetAtoms():
            a.SetAtomMapNum(0)
        submol.GetAtomWithIdx(root_sub).SetAtomMapNum(1)

        ranks = list(
            Chem.CanonicalRankAtoms(submol, breakTies=True, includeAtomMaps=True)
        )

        inv_amap = {sub_i: parent_i for parent_i, sub_i in amap.items()}

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

    def _encode_atom(self, mol: Chem.Mol, idx: int) -> List[str | int]:
        """
        Encode an RDKit Atom object into a list of integers.

        The encoding is as follows:
        - encoding_type: Atom encoding.
        - z: The atomic number of the atom.
        - chg: The formal charge of the atom.
        - arom: 1 if the atom is aromatic, 0 otherwise.
        - ring: 1 if the atom is in a ring, 0 otherwise.
        - h: The total number of hydrogen atoms bonded to the atom.
        - d: The degree of the atom.
        - chiral_type: The chiral type of the atom.
        - atom_map_num: The atom map number of the atom.
        - idx: The index of the atom in the molecule.

        Args:
            mol (Chem.Mol): The RDKit Molecule object containing the atom.
            idx (int): The index of the atom in the molecule.

        Returns:
            List[int]: A list of integers encoding the atom.
        """
        a = mol.GetAtomWithIdx(idx)
        z = a.GetAtomicNum()
        chg = a.GetFormalCharge()
        arom = 1 if a.GetIsAromatic() else 0
        ring = 1 if a.IsInRing() else 0
        h = a.GetTotalNumHs()
        d = a.GetDegree()
        atom_map_num = a.GetAtomMapNum()
        chiral_type = int(a.GetChiralTag())
        return [
            "atom_encoding",
            z,
            chg,
            arom,
            ring,
            h,
            d,
            chiral_type,
            atom_map_num,
            idx,
        ]

    def _encode_bond(self, mol: Chem.Mol, idx: int) -> List[str | int]:
        """
        Encode an RDKit Bond object into a list of integers.

        The encoding is as follows:
        - encoding_type: Bond encoding.
        - bond_order: The bond order of the bond multiplied by 10.
        - stereo: The stereochemistry of the bond.

        Args:
            mol (Chem.Mol): The RDKit Molecule object containing the bond.
            idx (int): The index of the bond in the molecule.

        Returns:
            List[int]: A list of integers encoding the bond.
        """
        b = mol.GetBondWithIdx(idx)
        stereo = b.GetStereo()
        return ["bond_encoding", int(b.GetBondTypeAsDouble() * 10), int(stereo)]

    def _assign_mapping(self, mol: Chem.Mol, atom_idx: int, atom_map_num: int) -> None:
        """Assign an atom-mapping number to a specific atom in a molecule.

        Args:
            mol (Chem.Mol): Molecule whose atom will be updated.
            atom_idx (int): Index of the atom in `mol` to update.
            atom_map_num (int): Atom-mapping number to assign to the atom.

        Returns:
            None: This method updates `mol` in place.
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(atom_map_num)
        return None

    def _remove_atom_idx_for_comparison(
        self, atom_map_list: List[List[List[int]]]
    ) -> List[List[List[int]]]:
        """Remove the trailing atom index from atom-encoding entries for equality checks.

        Args:
            atom_map_list (List[List[List[int]]]): Nested list structure describing a
                mapping in terms of atom/bond encodings, where sub-lists whose first
                element equals ``"atom_encoding"`` have a trailing atom index that
                should be ignored for comparison.

        Returns:
            List[List[List[int]]]: A new nested list with the last element removed
                from any ``["atom_encoding", ...]`` sub-list, leaving all other
                sub-lists unchanged.
        """
        new_list = []
        for init_list in atom_map_list:
            atom_bond_list = []
            for sub_list in init_list:
                if sub_list[0] == "atom_encoding":
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
        min_radius_to_anchor_new_mapping: int = 2,
    ) -> List[List[str]]:
        """ """
        matches = []
        reactant_encoding_dict_no_idx = {}
        for k, v in reactant_encoding_dict.items():
            if str(radius) != k.split("_")[-1]:
                continue
            if not v:
                continue
            if radius < min_radius_to_anchor_new_mapping:
                mapping_nums = [
                    subsub[-2]
                    for sub in v
                    for subsub in sub
                    if subsub[0] == "atom_encoding"
                ]
                if list(set(mapping_nums)) == [0]:
                    continue
            reactant_encoding_dict_no_idx[k] = self._remove_atom_idx_for_comparison(v)

        product_encoding_dict_no_idx = {}
        for k, v in product_encoding_dict.items():
            if str(radius) != k.split("_")[-1]:
                continue
            if not v:
                continue
            if radius < min_radius_to_anchor_new_mapping:
                mapping_nums = [
                    subsub[-2]
                    for sub in v
                    for subsub in sub
                    if subsub[0] == "atom_encoding"
                ]
                if list(set(mapping_nums)) == [0]:
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
        """ """
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
                    if sub_sub_v1[0] == "atom_encoding":
                        if sub_sub_v1[-1] == reactant_atom_idx:
                            sub_sub_v1[-2] = atom_map_num
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
                    if sub_sub_v1[0] == "atom_encoding":
                        if sub_sub_v1[-1] == product_atom_idx:
                            sub_sub_v1[-2] = atom_map_num
        for key in keys_to_delete:
            del product_encoding_dict[key]

        return atom_map_num + 1, reactant_encoding_dict, product_encoding_dict

    def map_reaction(
        self,
        reaction_smiles: str,
        min_radius: int = 1,
        min_radius_to_anchor_new_mapping: int = 3,
    ) -> ReactionMapperResult:
        """ """
        default_mapping_dict = ReactionMapperResult(
            original_smiles="",
            selected_mapping="",
            possible_mappings={},
            mapping_type=self._mapper_type,
            mapping_score=None,
            additional_info=[{}],
        )
        if not self._reaction_smiles_valid(reaction_smiles):
            return default_mapping_dict

        canonicalized_reaction_smiles = canonicalize_reaction_smiles(
            reaction_smiles, canonicalize_tautomer=False
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

            # For the purpose of finding final radius and populating encoding dictionaries,
            # we don't care if radius < min_radius_to_anchor_new_mapping
            matches = self._get_num_matching_atom_envs(
                reactant_encoding_dict,
                product_encoding_dict,
                radius,
                min_radius_to_anchor_new_mapping=0,
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
            [
                Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False)
                for mol in reactant_mols
            ]
        )
        mapped_product_smiles = ".".join(
            [
                Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False)
                for mol in product_mols
            ]
        )
        mapped_reaction_smiles = mapped_reactant_smiles + ">>" + mapped_product_smiles

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
        Map a list of reaction SMILES strings using the MCS mapper.

        Args:
            reaction_list (List[str]): List of reaction SMILES strings to map.

        Returns:
            List[ReactionMapperResult]: The mapping results in the same order as the
                input reactions.
        """

        mapped_reactions = []
        for reaction in reaction_list:
            mapped_reactions.append(self.map_reaction(reaction))
        return mapped_reactions
