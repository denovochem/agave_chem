"""Unit tests for agave_chem/utils/symmetry_classes.py."""

from rdkit import Chem

from agave_chem.utils.symmetry_classes import (
    get_symmetry_class_from_mol,
    get_symmetry_class_from_smiles,
    resolve_symmtery_class_for_tautomers,
)

# ---------------------------------------------------------------------------
# resolve_symmtery_class_for_tautomers — fallback on KekulizeException
# ---------------------------------------------------------------------------


def _make_unkekulizable_mol() -> Chem.Mol:
    """Build a 3-membered aromatic ring with no hydrogens.

    Each carbon needs valence 4 from two ring bonds, requiring both bonds
    to be double — impossible in a 3-membered ring.  This triggers a
    KekulizeException when the tautomer enumerator tries to kekulize.
    """
    mol = Chem.RWMol()
    for _ in range(3):
        a = Chem.Atom(6)
        a.SetIsAromatic(True)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        mol.AddAtom(a)
    for i in range(3):
        mol.AddBond(i, (i + 1) % 3, Chem.BondType.AROMATIC)
    result = mol.GetMol()
    result.UpdatePropertyCache(strict=False)
    return result


class TestResolveSymmteryClassForTautomersFallback:
    """Tests that resolve_symmtery_class_for_tautomers falls back gracefully."""

    def test_unkekulizable_mol_returns_valid_ranks(self):
        """A molecule that can't be kekulized should return non-tautomer ranks."""
        mol = _make_unkekulizable_mol()
        ranks = resolve_symmtery_class_for_tautomers(mol)
        assert isinstance(ranks, list)
        assert len(ranks) == mol.GetNumAtoms()
        assert all(isinstance(r, int) for r in ranks)

    def test_unkekulizable_mol_ranks_match_non_tautomer(self):
        """Fallback ranks should match plain CanonicalRankAtoms output."""
        mol = _make_unkekulizable_mol()
        ranks = resolve_symmtery_class_for_tautomers(mol)
        expected = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        assert ranks == expected

    def test_normal_mol_still_works(self):
        """Normal molecules should still get tautomer-aware ranks."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        ranks = resolve_symmtery_class_for_tautomers(mol)
        assert isinstance(ranks, list)
        assert len(ranks) == 6


# ---------------------------------------------------------------------------
# get_symmetry_class_from_mol — integration with fallback
# ---------------------------------------------------------------------------


class TestGetSymmetryClassFromMolFallback:
    """Tests that get_symmetry_class_from_mol handles unkekulizable molecules."""

    def test_unkekulizable_with_tautomers(self):
        """consider_tautomers=True should not crash on unkekulizable molecules."""
        mol = _make_unkekulizable_mol()
        ranks = get_symmetry_class_from_mol(mol, consider_tautomers=True)
        assert isinstance(ranks, list)
        assert len(ranks) == mol.GetNumAtoms()

    def test_unkekulizable_without_tautomers(self):
        """consider_tautomers=False should also work (no tautomer enumeration)."""
        mol = _make_unkekulizable_mol()
        ranks = get_symmetry_class_from_mol(mol, consider_tautomers=False)
        assert isinstance(ranks, list)
        assert len(ranks) == mol.GetNumAtoms()


# ---------------------------------------------------------------------------
# get_symmetry_class_from_smiles — basic sanity
# ---------------------------------------------------------------------------


class TestGetSymmetryClassFromSmiles:
    """Basic sanity tests for get_symmetry_class_from_smiles."""

    def test_benzene_returns_six_ranks(self):
        ranks = get_symmetry_class_from_smiles("c1ccccc1")
        assert len(ranks) == 6

    def test_all_carbons_symmetric_in_benzene(self):
        """All carbons in benzene should share the same symmetry class."""
        ranks = get_symmetry_class_from_smiles("c1ccccc1", consider_tautomers=False)
        assert len(set(ranks)) == 1
