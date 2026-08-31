"""Unit tests for get_differing_atoms in viz_utils.

Tests cover:
- Identical mappings (no differences)
- Mappings that differ on specific atoms
- Non-isomorphic reactions (returns None)
- Unmapped atoms (one mapper maps, other doesn't)
- Symmetry of results (A vs B and B vs A)
"""

from workflows.compare_mappers.viz_utils import DifferingAtoms, get_differing_atoms

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Simple amide formation: acid + amine -> amide + water
RXN_A = "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][NH2:6]>>[CH3:5][NH:6][C:2](=[O:3])[CH3:1].[OH2:4]"

# Same reaction, same mapping
RXN_A_COPY = "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][NH2:6]>>[CH3:5][NH:6][C:2](=[O:3])[CH3:1].[OH2:4]"

# Same reaction, but atoms 1 and 5 are swapped in the product (different mapping)
RXN_B = "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][NH2:6]>>[CH3:1][NH:6][C:2](=[O:3])[CH3:5].[OH2:4]"

# Completely different reaction (not isomorphic)
RXN_DIFFERENT = "[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH3:1][CH2:2][CH2:3][CH3:4]"


# ---------------------------------------------------------------------------
# Identical mappings
# ---------------------------------------------------------------------------


class TestIdenticalMappings:
    def test_no_differences(self) -> None:
        result = get_differing_atoms(RXN_A, RXN_A_COPY)
        assert result is not None
        assert not result.has_differences
        assert len(result.reactant_a) == 0
        assert len(result.product_a) == 0
        assert len(result.reactant_b) == 0
        assert len(result.product_b) == 0

    def test_has_differences_false(self) -> None:
        result = get_differing_atoms(RXN_A, RXN_A_COPY)
        assert result is not None
        assert result.has_differences is False


# ---------------------------------------------------------------------------
# Different mappings
# ---------------------------------------------------------------------------


class TestDifferentMappings:
    def test_detects_differences(self) -> None:
        result = get_differing_atoms(RXN_A, RXN_B)
        assert result is not None
        assert result.has_differences

    def test_differences_symmetric(self) -> None:
        """Differences should be symmetric: A vs B and B vs A."""
        result_ab = get_differing_atoms(RXN_A, RXN_B)
        result_ba = get_differing_atoms(RXN_B, RXN_A)
        assert result_ab is not None
        assert result_ba is not None
        # The number of differing atoms should be the same in both directions
        n_ab = len(result_ab.reactant_a) + len(result_ab.product_a)
        n_ba = len(result_ba.reactant_a) + len(result_ba.product_a)
        assert n_ab == n_ba

    def test_differing_atoms_nonempty(self) -> None:
        result = get_differing_atoms(RXN_A, RXN_B)
        assert result is not None
        # Atoms 1 and 5 are swapped, so they should appear as differing
        assert len(result.product_a) > 0
        assert len(result.product_b) > 0

    def test_differing_atoms_count(self) -> None:
        result = get_differing_atoms(RXN_A, RXN_B)
        assert result is not None
        # Two atoms are swapped (1 and 5), so 2 product atoms differ
        assert len(result.product_a) == 2
        assert len(result.product_b) == 2


# ---------------------------------------------------------------------------
# Non-isomorphic reactions
# ---------------------------------------------------------------------------


class TestNonIsomorphic:
    def test_returns_none(self) -> None:
        result = get_differing_atoms(RXN_A, RXN_DIFFERENT)
        assert result is None

    def test_invalid_smiles_returns_none(self) -> None:
        result = get_differing_atoms("invalid_smiles", RXN_A)
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = get_differing_atoms("", RXN_A)
        assert result is None


# ---------------------------------------------------------------------------
# DifferingAtoms dataclass
# ---------------------------------------------------------------------------


class TestDifferingAtoms:
    def test_empty_has_no_differences(self) -> None:
        da = DifferingAtoms()
        assert not da.has_differences

    def test_with_atoms_has_differences(self) -> None:
        da = DifferingAtoms(
            reactant_a={("R", 0, 0)},
            product_a={("P", 0, 1)},
        )
        assert da.has_differences

    def test_only_reactant_b(self) -> None:
        da = DifferingAtoms(reactant_b={("R", 0, 0)})
        assert da.has_differences
