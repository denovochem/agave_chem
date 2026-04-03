from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.types import ReactionMapperResult
from agave_chem.utils.chem_utils import (
    canonicalize_reaction_smiles,
)
from agave_chem.utils.logging_config import logger

AtomProps = Tuple[int, ...]
"""Immutable atom properties: (z, chg, arom, ring, h, d, chiral_type)."""

BondProps = Tuple[int, int]
"""Immutable bond properties: (bond_order_x10, stereo)."""

EnvKey = Tuple[int, int, int]
"""Environment key: (mol_idx, atom_idx, radius)."""

Skeleton = List[Tuple[int, int, int]]
"""Canonical bond-environment skeleton: [(begin_atom_idx, bond_idx, end_atom_idx), ...]."""

Fingerprint = Tuple[Tuple[AtomProps, BondProps, AtomProps], ...]
"""Hashable, index-free encoding of a bond environment used for equality matching."""


class MolPropertyCache:
    """
    Pre-computed, immutable atom and bond properties for a single molecule,
    plus mutable atom-mapping numbers updated during the mapping phase.

    Atom and bond properties are extracted once at construction time so that
    downstream code can look them up by index in O(1) instead of repeatedly
    querying the RDKit molecule.

    Args:
        mol (Chem.Mol): RDKit molecule to extract properties from.
    """

    __slots__ = ("atom_props", "bond_props", "atom_map_nums")

    def __init__(self, mol: Chem.Mol) -> None:
        self.atom_props: List[AtomProps] = []
        for a in mol.GetAtoms():
            self.atom_props.append(
                (
                    a.GetAtomicNum(),
                    a.GetFormalCharge(),
                    1 if a.GetIsAromatic() else 0,
                    1 if a.IsInRing() else 0,
                    a.GetTotalNumHs(),
                    a.GetDegree(),
                    int(a.GetChiralTag()),
                )
            )
        self.bond_props: List[BondProps] = []
        for b in mol.GetBonds():
            self.bond_props.append(
                (
                    int(b.GetBondTypeAsDouble() * 10),
                    int(b.GetStereo()),
                )
            )
        self.atom_map_nums: List[int] = [0] * mol.GetNumAtoms()


class MCSReactionMapper(ReactionMapper):
    """
    MCS reaction classification and atom-mapping.
    """

    def __init__(self, mapper_name: str, mapper_weight: float = 3):
        super().__init__("mcs", mapper_name, mapper_weight)

    @staticmethod
    def _compute_skeleton(mol: Chem.Mol, atom_idx: int, radius: int) -> Skeleton:
        """
        Compute the canonical bond-environment skeleton for an atom at a given
        radius.

        The skeleton records *which* parent-molecule atoms and bonds participate
        in the environment and in what canonical order, but does **not** encode
        any chemical properties.  Properties are looked up later from a
        ``MolPropertyCache`` when a full ``Fingerprint`` is needed.

        Args:
            mol (Chem.Mol): RDKit molecule.
            atom_idx (int): Index of the root atom in *mol*.
            radius (int): Bond-radius for environment extraction.

        Returns:
            Skeleton: Canonically ordered list of
                ``(begin_atom_idx, bond_idx, end_atom_idx)`` referencing
                parent-molecule indices.  Empty list when no bonds are found
                within the given radius.
        """
        bond_ids = Chem.rdmolops.FindAtomEnvironmentOfRadiusN(
            mol,
            radius=radius,
            rootedAtAtom=atom_idx,
        )
        if not bond_ids:
            return []

        amap: Dict[int, int] = {}
        submol = Chem.PathToSubmol(mol, bond_ids, atomMap=amap)

        root_sub = amap[atom_idx]
        for a in submol.GetAtoms():
            a.SetAtomMapNum(0)
        submol.GetAtomWithIdx(root_sub).SetAtomMapNum(1)

        ranks = list(
            Chem.CanonicalRankAtoms(submol, breakTies=True, includeAtomMaps=True)
        )

        inv_amap = {sub_i: parent_i for parent_i, sub_i in amap.items()}

        bond_items: List[Tuple[Tuple[int, ...], int, int]] = []
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

        skeleton: Skeleton = []
        for _, sa1, sa2 in bond_items:
            if ranks[sa1] <= ranks[sa2]:
                s_begin, s_end = sa1, sa2
            else:
                s_begin, s_end = sa2, sa1

            p_begin = inv_amap[s_begin]
            p_end = inv_amap[s_end]
            pbond = mol.GetBondBetweenAtoms(p_begin, p_end)
            skeleton.append((p_begin, pbond.GetIdx(), p_end))

        return skeleton

    @staticmethod
    def _compute_fingerprint(
        cache: MolPropertyCache, skeleton: Skeleton
    ) -> Fingerprint:
        """
        Build a hashable fingerprint from a skeleton and current properties.

        The fingerprint includes atom properties and the current atom-map number
        but **excludes** atom indices, so two environments that are chemically
        identical (modulo index) produce the same fingerprint.

        Args:
            cache (MolPropertyCache): Pre-computed properties for the molecule
                that owns the skeleton.
            skeleton (Skeleton): Canonical list of
                ``(begin_atom_idx, bond_idx, end_atom_idx)``.

        Returns:
            Fingerprint: Hashable nested tuple suitable for equality / hash-based
                matching.
        """
        parts: List[Tuple[AtomProps, BondProps, AtomProps]] = []
        for begin_idx, bond_idx, end_idx in skeleton:
            parts.append(
                (
                    cache.atom_props[begin_idx] + (cache.atom_map_nums[begin_idx],),
                    cache.bond_props[bond_idx],
                    cache.atom_props[end_idx] + (cache.atom_map_nums[end_idx],),
                )
            )
        return tuple(parts)

    def _build_skeletons_at_radius(
        self,
        mols: List[Chem.Mol],
        caches: List[MolPropertyCache],
        radius: int,
        skeletons: Dict[EnvKey, Skeleton],
        fingerprints: Dict[EnvKey, Fingerprint],
        atom_to_entries: Dict[Tuple[int, int], Set[EnvKey]],
    ) -> None:
        """
        Compute skeletons and fingerprints for every atom in *mols* at a single
        radius and append them to the provided dicts.

        Args:
            mols (List[Chem.Mol]): List of molecules.
            caches (List[MolPropertyCache]): Property caches parallel to *mols*.
            radius (int): Bond-radius to compute.
            skeletons (Dict[EnvKey, Skeleton]): Skeleton dict to extend in place.
            fingerprints (Dict[EnvKey, Fingerprint]): Fingerprint dict to extend
                in place.
            atom_to_entries (Dict[Tuple[int, int], Set[EnvKey]]): Reverse index
                mapping ``(mol_idx, atom_idx)`` to the set of ``EnvKey`` values
                whose skeletons reference that atom.  Extended in place.

        Returns:
            None: All three dicts are mutated in place.
        """
        for mol_idx, mol in enumerate(mols):
            if mol.GetNumAtoms() < radius:
                continue
            cache = caches[mol_idx]
            for atom in mol.GetAtoms():
                aidx = atom.GetIdx()
                key: EnvKey = (mol_idx, aidx, radius)
                skel = self._compute_skeleton(mol, aidx, radius)
                skeletons[key] = skel
                fingerprints[key] = self._compute_fingerprint(cache, skel)
                for begin_idx, _, end_idx in skel:
                    atom_to_entries[(mol_idx, begin_idx)].add(key)
                    atom_to_entries[(mol_idx, end_idx)].add(key)

    @staticmethod
    def _find_matches(
        r_fps: Dict[EnvKey, Fingerprint],
        p_fps: Dict[EnvKey, Fingerprint],
        radius: int,
        min_radius_to_anchor_new_mapping: int = 2,
    ) -> List[Tuple[EnvKey, EnvKey]]:
        """
        Find reactant–product environment pairs with identical fingerprints at a
        given radius.

        Uses hash-based grouping for O(R + P) complexity instead of the
        previous O(R × P) pairwise comparison.

        Args:
            r_fps (Dict[EnvKey, Fingerprint]): Reactant fingerprints.
            p_fps (Dict[EnvKey, Fingerprint]): Product fingerprints.
            radius (int): Radius to match at.
            min_radius_to_anchor_new_mapping (int): Below this radius, only
                match environments that already contain at least one mapped
                atom (i.e. at least one atom-map number ≠ 0).

        Returns:
            List[Tuple[EnvKey, EnvKey]]: Pairs of
                ``(reactant_key, product_key)`` whose fingerprints are equal.
        """
        need_mapped_check = radius < min_radius_to_anchor_new_mapping

        def _has_mapped_atom(fp: Fingerprint) -> bool:
            for atom_begin, _, atom_end in fp:
                if atom_begin[-1] != 0 or atom_end[-1] != 0:
                    return True
            return False

        # Build reactant index: fingerprint → [keys]
        fp_to_r_keys: Dict[Fingerprint, List[EnvKey]] = defaultdict(list)
        for key, fp in r_fps.items():
            if key[2] != radius:
                continue
            if not fp:
                continue
            if need_mapped_check and not _has_mapped_atom(fp):
                continue
            fp_to_r_keys[fp].append(key)

        # Match product fingerprints against the reactant index
        matches: List[Tuple[EnvKey, EnvKey]] = []
        for key, fp in p_fps.items():
            if key[2] != radius:
                continue
            if not fp:
                continue
            if need_mapped_check and not _has_mapped_atom(fp):
                continue
            if fp in fp_to_r_keys:
                for rkey in fp_to_r_keys[fp]:
                    matches.append((rkey, key))

        return matches

    def _build_skeletons_and_find_radius(
        self,
        reactant_mols: List[Chem.Mol],
        product_mols: List[Chem.Mol],
        r_caches: List[MolPropertyCache],
        p_caches: List[MolPropertyCache],
        min_radius: int,
        max_radius: int,
    ) -> Tuple[
        Dict[EnvKey, Skeleton],
        Dict[EnvKey, Skeleton],
        Dict[EnvKey, Fingerprint],
        Dict[EnvKey, Fingerprint],
        Dict[Tuple[int, int], Set[EnvKey]],
        Dict[Tuple[int, int], Set[EnvKey]],
        int,
    ]:
        """
        Incrementally build skeletons/fingerprints radius-by-radius and
        determine the optimal (final) radius for atom-mapping.

        Stops early as soon as a unique match is found or matches disappear.

        Args:
            reactant_mols (List[Chem.Mol]): Reactant molecules.
            product_mols (List[Chem.Mol]): Product molecules.
            r_caches (List[MolPropertyCache]): Reactant property caches.
            p_caches (List[MolPropertyCache]): Product property caches.
            min_radius (int): Minimum bond-radius (inclusive).
            max_radius (int): Maximum bond-radius (exclusive).

        Returns:
            Tuple containing:
                - Reactant skeletons dict.
                - Product skeletons dict.
                - Reactant fingerprints dict.
                - Product fingerprints dict.
                - Reactant atom-to-entries reverse index.
                - Product atom-to-entries reverse index.
                - The final radius selected for the mapping phase.
        """
        r_skeletons: Dict[EnvKey, Skeleton] = {}
        p_skeletons: Dict[EnvKey, Skeleton] = {}
        r_fps: Dict[EnvKey, Fingerprint] = {}
        p_fps: Dict[EnvKey, Fingerprint] = {}
        r_a2e: Dict[Tuple[int, int], Set[EnvKey]] = defaultdict(set)
        p_a2e: Dict[Tuple[int, int], Set[EnvKey]] = defaultdict(set)

        final_radius = min_radius
        for radius in range(min_radius, max_radius):
            final_radius = radius

            self._build_skeletons_at_radius(
                reactant_mols,
                r_caches,
                radius,
                r_skeletons,
                r_fps,
                r_a2e,
            )
            self._build_skeletons_at_radius(
                product_mols,
                p_caches,
                radius,
                p_skeletons,
                p_fps,
                p_a2e,
            )

            matches = self._find_matches(
                r_fps,
                p_fps,
                radius,
                min_radius_to_anchor_new_mapping=0,
            )

            if len(matches) == 1:
                break
            if len(matches) == 0:
                final_radius = radius - 1
                break

        return r_skeletons, p_skeletons, r_fps, p_fps, r_a2e, p_a2e, final_radius

    def _recompute_affected_fingerprints(
        self,
        mol_idx: int,
        atom_idx: int,
        skeletons: Dict[EnvKey, Skeleton],
        fps: Dict[EnvKey, Fingerprint],
        caches: List[MolPropertyCache],
        atom_to_entries: Dict[Tuple[int, int], Set[EnvKey]],
    ) -> None:
        """
        Recompute fingerprints for every skeleton entry that references a
        given atom.

        Called after updating an atom's map number so that fingerprints
        reflect the new mapping state.

        Args:
            mol_idx (int): Index of the molecule containing the updated atom.
            atom_idx (int): Index of the updated atom within the molecule.
            skeletons (Dict[EnvKey, Skeleton]): All precomputed skeletons.
            fps (Dict[EnvKey, Fingerprint]): Fingerprint dict to update in
                place.
            caches (List[MolPropertyCache]): Property caches indexed by
                *mol_idx*.
            atom_to_entries (Dict[Tuple[int, int], Set[EnvKey]]): Reverse
                index.

        Returns:
            None: *fps* is mutated in place.
        """
        affected = atom_to_entries.get((mol_idx, atom_idx), set())
        for key in affected:
            if key in skeletons:
                fps[key] = self._compute_fingerprint(
                    caches[key[0]],
                    skeletons[key],
                )

    def _assign_single_mapping(
        self,
        r_key: EnvKey,
        p_key: EnvKey,
        reactant_mols: List[Chem.Mol],
        product_mols: List[Chem.Mol],
        r_caches: List[MolPropertyCache],
        p_caches: List[MolPropertyCache],
        r_skeletons: Dict[EnvKey, Skeleton],
        p_skeletons: Dict[EnvKey, Skeleton],
        r_fps: Dict[EnvKey, Fingerprint],
        p_fps: Dict[EnvKey, Fingerprint],
        r_atom_to_entries: Dict[Tuple[int, int], Set[EnvKey]],
        p_atom_to_entries: Dict[Tuple[int, int], Set[EnvKey]],
        atom_map_num: int,
    ) -> int:
        """
        Assign an atom-mapping number to one matched reactant–product atom
        pair.

        Updates molecule objects, property caches, and fingerprints in place.
        Deletes skeleton / fingerprint entries whose root is the newly mapped
        atom (at all radii) so they cannot match again.

        Args:
            r_key (EnvKey): Reactant environment key ``(mol_idx, atom_idx, radius)``.
            p_key (EnvKey): Product environment key ``(mol_idx, atom_idx, radius)``.
            reactant_mols (List[Chem.Mol]): Reactant molecule objects.
            product_mols (List[Chem.Mol]): Product molecule objects.
            r_caches (List[MolPropertyCache]): Reactant property caches.
            p_caches (List[MolPropertyCache]): Product property caches.
            r_skeletons (Dict[EnvKey, Skeleton]): Reactant skeletons (mutated).
            p_skeletons (Dict[EnvKey, Skeleton]): Product skeletons (mutated).
            r_fps (Dict[EnvKey, Fingerprint]): Reactant fingerprints (mutated).
            p_fps (Dict[EnvKey, Fingerprint]): Product fingerprints (mutated).
            r_atom_to_entries (Dict[Tuple[int, int], Set[EnvKey]]): Reactant
                reverse index.
            p_atom_to_entries (Dict[Tuple[int, int], Set[EnvKey]]): Product
                reverse index.
            atom_map_num (int): The atom-map number to assign.

        Returns:
            int: The next ``atom_map_num`` to use — incremented by 1 when the
                assignment succeeds, unchanged otherwise.
        """
        r_mol_idx, r_atom_idx, _ = r_key
        p_mol_idx, p_atom_idx, _ = p_key

        # Skip if the reactant root atom is already mapped
        if reactant_mols[r_mol_idx].GetAtomWithIdx(r_atom_idx).GetAtomMapNum() != 0:
            return atom_map_num

        # Assign on reactant mol
        reactant_mols[r_mol_idx].GetAtomWithIdx(r_atom_idx).SetAtomMapNum(atom_map_num)

        # Skip if the product root atom is already mapped
        # (reactant mol was already modified — matches original behaviour)
        if product_mols[p_mol_idx].GetAtomWithIdx(p_atom_idx).GetAtomMapNum() != 0:
            return atom_map_num

        # Assign on product mol
        product_mols[p_mol_idx].GetAtomWithIdx(p_atom_idx).SetAtomMapNum(atom_map_num)

        # Update caches
        r_caches[r_mol_idx].atom_map_nums[r_atom_idx] = atom_map_num
        p_caches[p_mol_idx].atom_map_nums[p_atom_idx] = atom_map_num

        # Delete entries whose root IS the newly mapped atom (all radii)
        for k in [k for k in r_skeletons if k[0] == r_mol_idx and k[1] == r_atom_idx]:
            del r_skeletons[k]
            r_fps.pop(k, None)

        for k in [k for k in p_skeletons if k[0] == p_mol_idx and k[1] == p_atom_idx]:
            del p_skeletons[k]
            p_fps.pop(k, None)

        # Recompute fingerprints that reference the mapped atoms
        self._recompute_affected_fingerprints(
            r_mol_idx,
            r_atom_idx,
            r_skeletons,
            r_fps,
            r_caches,
            r_atom_to_entries,
        )
        self._recompute_affected_fingerprints(
            p_mol_idx,
            p_atom_idx,
            p_skeletons,
            p_fps,
            p_caches,
            p_atom_to_entries,
        )

        return atom_map_num + 1

    def _assign_atom_map_nums(
        self,
        reactant_mols: List[Chem.Mol],
        product_mols: List[Chem.Mol],
        r_caches: List[MolPropertyCache],
        p_caches: List[MolPropertyCache],
        r_skeletons: Dict[EnvKey, Skeleton],
        p_skeletons: Dict[EnvKey, Skeleton],
        r_fps: Dict[EnvKey, Fingerprint],
        p_fps: Dict[EnvKey, Fingerprint],
        r_atom_to_entries: Dict[Tuple[int, int], Set[EnvKey]],
        p_atom_to_entries: Dict[Tuple[int, int], Set[EnvKey]],
        final_radius: int,
        min_radius_to_anchor_new_mapping: int,
    ) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
        """
        Walk from *final_radius* down to 1, assigning atom-map numbers to
        matched reactant–product pairs at each level.

        Args:
            reactant_mols (List[Chem.Mol]): Reactant molecule objects.
            product_mols (List[Chem.Mol]): Product molecule objects.
            r_caches (List[MolPropertyCache]): Reactant property caches.
            p_caches (List[MolPropertyCache]): Product property caches.
            r_skeletons (Dict[EnvKey, Skeleton]): Reactant skeletons.
            p_skeletons (Dict[EnvKey, Skeleton]): Product skeletons.
            r_fps (Dict[EnvKey, Fingerprint]): Reactant fingerprints.
            p_fps (Dict[EnvKey, Fingerprint]): Product fingerprints.
            r_atom_to_entries (Dict[Tuple[int, int], Set[EnvKey]]): Reactant
                reverse index.
            p_atom_to_entries (Dict[Tuple[int, int], Set[EnvKey]]): Product
                reverse index.
            final_radius (int): Largest radius to start assigning from.
            min_radius_to_anchor_new_mapping (int): Passed through to
                ``_find_matches``.

        Returns:
            Tuple[List[Chem.Mol], List[Chem.Mol]]: The (mutated) reactant and
                product molecule lists with atom-map numbers assigned.
        """
        atom_map_num = 1
        for radius in range(final_radius, 0, -1):
            matches = self._find_matches(
                r_fps,
                p_fps,
                radius,
                min_radius_to_anchor_new_mapping=min_radius_to_anchor_new_mapping,
            )
            num_matches = len(matches)
            for _ in range(num_matches):
                atom_map_num = self._assign_single_mapping(
                    matches[0][0],
                    matches[0][1],
                    reactant_mols,
                    product_mols,
                    r_caches,
                    p_caches,
                    r_skeletons,
                    p_skeletons,
                    r_fps,
                    p_fps,
                    r_atom_to_entries,
                    p_atom_to_entries,
                    atom_map_num,
                )
                matches = self._find_matches(
                    r_fps,
                    p_fps,
                    radius,
                )
                if not matches:
                    break

        return reactant_mols, product_mols

    def map_reaction(
        self,
        reaction_smiles: str,
        min_radius: int = 1,
        min_radius_to_anchor_new_mapping: int = 3,
        max_radius: Optional[int] = None,
    ) -> ReactionMapperResult:
        """
        Atom-map a single reaction SMILES using MCS-based environment matching.

        Args:
            reaction_smiles (str): Reaction SMILES of the form
                ``"reactants>>products"``.
            min_radius (int): Smallest bond-radius to consider.
            min_radius_to_anchor_new_mapping (int): Below this radius,
                environments are only matched when they already contain at
                least one mapped atom.
            max_radius (Optional[int]): Largest bond-radius to search.
                Defaults to the size of the largest molecule.

        Returns:
            ReactionMapperResult: Mapping result containing the original and
                mapped SMILES.  Falls back to the default (empty) result on
                invalid input or failed mapping.
        """
        if not self._reaction_smiles_valid(reaction_smiles):
            return self._return_default_mapping_dict(reaction_smiles)

        canonicalized_reaction_smiles = canonicalize_reaction_smiles(
            reaction_smiles,
            canonicalize_tautomer=False,
        )
        reactants_str, products_str = self._split_reaction_components(
            canonicalized_reaction_smiles,
        )

        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_str.split(".")]
        if None in reactant_mols:
            logger.warning(f"Failed to parse reactant SMILES: {reactants_str}")
            return self._return_default_mapping_dict(reaction_smiles)
        product_mols = [Chem.MolFromSmiles(p) for p in products_str.split(".")]
        if None in product_mols:
            logger.warning(f"Failed to parse product SMILES: {products_str}")
            return self._return_default_mapping_dict(reaction_smiles)

        if not max_radius:
            max_radius = max(mol.GetNumAtoms() for mol in reactant_mols + product_mols)

        # Pre-compute property caches
        r_caches = [MolPropertyCache(m) for m in reactant_mols]
        p_caches = [MolPropertyCache(m) for m in product_mols]

        # Build skeletons / fingerprints incrementally and find optimal radius
        (
            r_skeletons,
            p_skeletons,
            r_fps,
            p_fps,
            r_a2e,
            p_a2e,
            final_radius,
        ) = self._build_skeletons_and_find_radius(
            reactant_mols,
            product_mols,
            r_caches,
            p_caches,
            min_radius,
            max_radius,
        )

        # Assign atom-map numbers from final_radius down to 1
        reactant_mols, product_mols = self._assign_atom_map_nums(
            reactant_mols,
            product_mols,
            r_caches,
            p_caches,
            r_skeletons,
            p_skeletons,
            r_fps,
            p_fps,
            r_a2e,
            p_a2e,
            final_radius,
            min_radius_to_anchor_new_mapping,
        )

        mapped_reactant_smiles = ".".join(
            Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False)
            for mol in reactant_mols
        )
        mapped_product_smiles = ".".join(
            Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False)
            for mol in product_mols
        )
        mapped_reaction_smiles = mapped_reactant_smiles + ">>" + mapped_product_smiles

        if not self._verify_validity_of_mapping(
            mapped_reaction_smiles,
            expect_full_mapping=False,
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

    def map_reactions(
        self,
        reaction_list: List[str],
        min_radius: int = 1,
        min_radius_to_anchor_new_mapping: int = 3,
    ) -> List[ReactionMapperResult]:
        """
        Map a list of reaction SMILES strings using the MCS mapper.

        Args:
            reaction_list (List[str]): List of reaction SMILES strings to map.
            min_radius (int): Smallest bond-radius to consider.
            min_radius_to_anchor_new_mapping (int): Below this radius,
                environments are only matched when they already contain at
                least one mapped atom.

        Returns:
            List[ReactionMapperResult]: The mapping results in the same order
                as the input reactions.
        """
        mapped_reactions = []
        for reaction in reaction_list:
            mapped_reactions.append(self.map_reaction(reaction))
        return mapped_reactions
