from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.types import ReactionMapperResult
from agave_chem.utils.chem_utils import (
    canonicalize_reaction_smiles,
)
from agave_chem.utils.logging_config import logger

AtomProps = Tuple[int, ...]
"""Immutable atom properties: (z, chg, arom, ring, h, d)."""

BondProps = Tuple[int, int]
"""Immutable bond properties: (bond_order_x10, stereo)."""

EnvKey = Tuple[int, int, int]
"""Environment key: (mol_idx, atom_idx, radius)."""

Skeleton = List[Tuple[int, int, int, int, int]]
"""Canonical bond-environment skeleton: [(begin_atom_idx, bond_idx, end_atom_idx, begin_dist, end_dist), ...]."""

Fingerprint = Tuple[Tuple[int, AtomProps, BondProps, int, AtomProps], ...]
"""Hashable, index-free encoding of a bond environment used for equality matching.
Each entry: (begin_dist, begin_atom_props, bond_props, end_dist, end_atom_props)."""


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
        self._uncharger = rdMolStandardize.Uncharger()
        self._tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

    def _normalize_mol(self, mol: Chem.Mol) -> Chem.Mol:
        """Neutralize charges and canonicalize tautomer for matching purposes."""
        mol = Chem.RWMol(mol)  # work on a copy
        mol = self._uncharger.uncharge(mol)
        mol = self._tautomer_enumerator.Canonicalize(mol)
        return mol

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
                ``(begin_atom_idx, bond_idx, end_atom_idx, begin_dist, end_dist)``
                referencing parent-molecule indices and BFS distances from the
                root atom.  Empty list when no bonds are found within the given
                radius.
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

        # BFS distances from root in the submol
        dist: Dict[int, int] = {root_sub: 0}
        queue = deque([root_sub])
        while queue:
            cur = queue.popleft()
            for nb in submol.GetAtomWithIdx(cur).GetNeighbors():
                nidx = nb.GetIdx()
                if nidx not in dist:
                    dist[nidx] = dist[cur] + 1
                    queue.append(nidx)

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
            skeleton.append(
                (p_begin, pbond.GetIdx(), p_end, dist[s_begin], dist[s_end])
            )

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
        parts: List[Tuple[int, AtomProps, BondProps, int, AtomProps]] = []
        for begin_idx, bond_idx, end_idx, begin_dist, end_dist in skeleton:
            parts.append(
                (
                    begin_dist,
                    cache.atom_props[begin_idx] + (cache.atom_map_nums[begin_idx],),
                    cache.bond_props[bond_idx],
                    end_dist,
                    cache.atom_props[end_idx] + (cache.atom_map_nums[end_idx],),
                )
            )
        return tuple(sorted(parts))

    def _build_skeletons_at_radius(
        self,
        mols: List[Chem.Mol],
        caches: List[MolPropertyCache],
        radius: int,
        skeletons: Dict[EnvKey, Skeleton],
        fingerprints: Dict[EnvKey, Fingerprint],
        atom_to_entries: Dict[Tuple[int, int], Set[EnvKey]],
        skip_atoms: Optional[Set[Tuple[int, int]]] = None,
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
            skip_atoms (Optional[Set[Tuple[int, int]]]): Atoms to skip, as
                ``(mol_idx, atom_idx)`` pairs.  Populated by the caller with
                atoms whose environment at a previous radius had no cross-side
                fingerprint match.

        Returns:
            None: All three dicts are mutated in place.
        """
        for mol_idx, mol in enumerate(mols):
            if mol.GetNumAtoms() < radius:
                continue
            cache = caches[mol_idx]
            for atom in mol.GetAtoms():
                aidx = atom.GetIdx()
                if skip_atoms is not None and (mol_idx, aidx) in skip_atoms:
                    continue
                key: EnvKey = (mol_idx, aidx, radius)
                skel = self._compute_skeleton(mol, aidx, radius)
                skeletons[key] = skel
                fingerprints[key] = self._compute_fingerprint(cache, skel)
                for begin_idx, _, end_idx, _, _ in skel:
                    atom_to_entries[(mol_idx, begin_idx)].add(key)
                    atom_to_entries[(mol_idx, end_idx)].add(key)

    @staticmethod
    def _find_matches(
        r_fps: Dict[EnvKey, Fingerprint],
        p_fps: Dict[EnvKey, Fingerprint],
        radius: int,
        min_radius_to_anchor_new_mapping: int = 3,
        require_anchor: bool = False,
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
            require_anchor (bool): If True, only match environments that
                contain at least one already-mapped atom, regardless of
                radius.  Used during the extend phase to force anchored-only
                matching at every radius level.

        Returns:
            List[Tuple[EnvKey, EnvKey]]: Pairs of
                ``(reactant_key, product_key)`` whose fingerprints are equal.
        """
        need_mapped_check = (
            require_anchor or radius < min_radius_to_anchor_new_mapping - 1
        )

        def _has_mapped_atom(fp: Fingerprint) -> bool:
            for entry in fp:
                # entry: (begin_dist, begin_atom_props, bond_props, end_dist, end_atom_props)
                atom_begin = entry[1]
                atom_end = entry[4]
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

        # Atoms whose environment at a previous radius had no cross-side
        # fingerprint match and can be safely skipped at all subsequent radii.
        r_skip: Set[Tuple[int, int]] = set()
        p_skip: Set[Tuple[int, int]] = set()

        final_radius = min_radius
        for radius in range(min_radius, max_radius):
            final_radius = radius

            # Products first, then reactants.
            self._build_skeletons_at_radius(
                product_mols,
                p_caches,
                radius,
                p_skeletons,
                p_fps,
                p_a2e,
                skip_atoms=p_skip,
            )
            self._build_skeletons_at_radius(
                reactant_mols,
                r_caches,
                radius,
                r_skeletons,
                r_fps,
                r_a2e,
                skip_atoms=r_skip,
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

            # Prune atoms whose fingerprint at this radius had no match on
            # the other side — they cannot match at any higher radius either.
            p_fp_set: Set[Fingerprint] = set()
            r_fp_set: Set[Fingerprint] = set()
            for key, fp in p_fps.items():
                if key[2] == radius and fp:
                    p_fp_set.add(fp)
            for key, fp in r_fps.items():
                if key[2] == radius and fp:
                    r_fp_set.add(fp)

            for key, fp in r_fps.items():
                if key[2] == radius and (not fp or fp not in p_fp_set):
                    r_skip.add((key[0], key[1]))
            for key, fp in p_fps.items():
                if key[2] == radius and (not fp or fp not in r_fp_set):
                    p_skip.add((key[0], key[1]))

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
        product_atom_source: Dict[Tuple[int, int], int],
        atom_map_num: int,
    ) -> int:
        """
        Assign an atom-mapping number to one matched reactant–product atom
        pair.

        Updates molecule objects, property caches, and fingerprints in place.
        Deletes skeleton / fingerprint entries whose root is the newly mapped
        atom (at all radii) so they cannot match again.

        Also enforces adjacency consistency: a mapping is rejected if the
        product atom has an already-mapped neighbor whose mapping originated
        from a different reactant molecule.

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
            product_atom_source (Dict[Tuple[int, int], int]): Mapping from
                ``(product_mol_idx, product_atom_idx)`` to the reactant
                molecule index that the atom was mapped from.  Updated in
                place on successful assignment.
            atom_map_num (int): The atom-map number to assign.

        Returns:
            int: The next ``atom_map_num`` to use — incremented by 1 when the
                assignment succeeds, unchanged otherwise.
        """
        r_mol_idx, r_atom_idx, _ = r_key
        p_mol_idx, p_atom_idx, _ = p_key

        # Skip if either root atom is already mapped. Mapping must be assigned
        # atomically to both sides to avoid creating a one-sided mapping state
        # that can prevent downstream anchored matching.
        if reactant_mols[r_mol_idx].GetAtomWithIdx(r_atom_idx).GetAtomMapNum() != 0:
            return atom_map_num
        if product_mols[p_mol_idx].GetAtomWithIdx(p_atom_idx).GetAtomMapNum() != 0:
            return atom_map_num

        # Adjacency consistency: reject if any already-mapped neighbor of the
        # product atom was mapped from a different reactant molecule.
        p_atom = product_mols[p_mol_idx].GetAtomWithIdx(p_atom_idx)
        for nb in p_atom.GetNeighbors():
            nb_key = (p_mol_idx, nb.GetIdx())
            if (
                nb_key in product_atom_source
                and product_atom_source[nb_key] != r_mol_idx
            ):
                return atom_map_num

        # Assign on both molecules
        reactant_mols[r_mol_idx].GetAtomWithIdx(r_atom_idx).SetAtomMapNum(atom_map_num)
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

        product_atom_source[(p_mol_idx, p_atom_idx)] = r_mol_idx

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
        Assign atom-map numbers using an anchor-extend strategy.

        The algorithm alternates between two phases:

        1. **Extend phase** — scan all radii (high to low), only matching
           environments that contain at least one already-mapped atom.  Repeat
           the full sweep until no more anchored matches are found.
        2. **New-anchor phase** — find one unanchored match at the highest
           feasible radius (>= *min_radius_to_anchor_new_mapping*) and assign
           it.  Return to the extend phase.

        This ensures that each anchor site is fully propagated before a new
        one is created, preventing gaps caused by split-source mapping.

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
        product_atom_source: Dict[Tuple[int, int], int] = {}

        def _try_assign_first_viable(
            matches: List[Tuple[EnvKey, EnvKey]],
        ) -> bool:
            """
            Attempt to assign the first viable match from *matches*.

            Args:
                matches (List[Tuple[EnvKey, EnvKey]]): Candidate
                    reactant-product environment pairs.

            Returns:
                bool: True if a mapping was successfully assigned.
            """
            nonlocal atom_map_num
            for r_key, p_key in matches:
                prev = atom_map_num
                atom_map_num = self._assign_single_mapping(
                    r_key,
                    p_key,
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
                    product_atom_source,
                    atom_map_num,
                )
                if atom_map_num > prev:
                    return True
            return False

        while True:
            # --- Extend phase: propagate from existing anchors ---
            extended = True
            while extended:
                extended = False
                for radius in range(final_radius, 0, -1):
                    any_at_radius = True
                    while any_at_radius:
                        any_at_radius = False
                        matches = self._find_matches(
                            r_fps,
                            p_fps,
                            radius,
                            min_radius_to_anchor_new_mapping=min_radius_to_anchor_new_mapping,
                            require_anchor=True,
                        )
                        if matches and _try_assign_first_viable(matches):
                            any_at_radius = True
                            extended = True

            # --- New-anchor phase: create one new unanchored mapping ---
            new_anchor = False
            for radius in range(final_radius, 0, -1):
                matches = self._find_matches(
                    r_fps,
                    p_fps,
                    radius,
                    min_radius_to_anchor_new_mapping=min_radius_to_anchor_new_mapping,
                    require_anchor=False,
                )
                if matches and _try_assign_first_viable(matches):
                    new_anchor = True
                    break

            if not new_anchor:
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
            mapped_reactions.append(
                self.map_reaction(
                    reaction,
                    min_radius=min_radius,
                    min_radius_to_anchor_new_mapping=min_radius_to_anchor_new_mapping,
                )
            )
        return mapped_reactions
