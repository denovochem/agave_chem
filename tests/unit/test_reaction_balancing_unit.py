"""Unit tests for agave_chem.utils.reaction_balancing."""

import pytest

from agave_chem.utils.reaction_balancing import (
    compute_unmapped_product_atom_islands,
    detect_atom_count_imbalance,
    determine_one_to_one_correspondence,
)


class TestDetectAtomCountImbalance:
    """Tests for detect_atom_count_imbalance."""

    @pytest.mark.parametrize(
        "reaction_smiles",
        [
            "CCO>>CCO",
            "CC(=O)O.O>>CC(=O)O.O",
            "c1ccccc1.O>>c1ccccc1.O",
        ],
    )
    def test_balanced_reactions_return_false(self, reaction_smiles: str) -> None:
        assert detect_atom_count_imbalance(reaction_smiles) is False

    @pytest.mark.parametrize(
        "reaction_smiles",
        [
            "CCO>>CCO.CCO",
            "C>>CC",
            "N.N>>N.N.N",
        ],
    )
    def test_imbalanced_reactions_return_true(self, reaction_smiles: str) -> None:
        assert detect_atom_count_imbalance(reaction_smiles) is True

    def test_invalid_smiles_returns_false(self) -> None:
        assert detect_atom_count_imbalance("invalid>>smiles") is False

    def test_no_reaction_arrow_returns_false(self) -> None:
        assert detect_atom_count_imbalance("CCO") is False

    def test_empty_string_returns_false(self) -> None:
        assert detect_atom_count_imbalance("") is False

    def test_different_elements_imbalance(self) -> None:
        assert detect_atom_count_imbalance("CCO>>CCN") is True


class TestComputeUnmappedProductAtomIslands:
    """Tests for compute_unmapped_product_atom_islands."""

    def test_all_mapped_returns_empty(self) -> None:
        smiles = "[CH3:1][CH2:2][OH:3]"
        result = compute_unmapped_product_atom_islands(smiles)
        assert result == {}

    def test_single_unmapped_island(self) -> None:
        smiles = "[CH3:1][CH2:2]O"
        result = compute_unmapped_product_atom_islands(smiles)
        assert len(result) == 1
        assert 0 in result
        assert result[0] == {2}

    def test_multiple_unmapped_islands(self) -> None:
        smiles = "[CH3:1]O.[NH2:2]O"
        result = compute_unmapped_product_atom_islands(smiles)
        assert len(result) == 2

    def test_all_unmapped_single_molecule(self) -> None:
        smiles = "CCO"
        result = compute_unmapped_product_atom_islands(smiles)
        assert len(result) == 1
        assert result[0] == {0, 1, 2}

    def test_all_unmapped_two_fragments(self) -> None:
        smiles = "CCO.N"
        result = compute_unmapped_product_atom_islands(smiles)
        assert len(result) == 2

    def test_invalid_smiles_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Could not parse SMILES"):
            compute_unmapped_product_atom_islands("not_a_smiles")

    def test_empty_string_returns_empty(self) -> None:
        result = compute_unmapped_product_atom_islands("")
        assert result == {}


class TestDetermineOneToOneCorrespondence:
    """Tests for determine_one_to_one_correspondence."""

    def test_balanced_no_islands_returns_true(self) -> None:
        assert determine_one_to_one_correspondence("CCO>>CCO", {}) is True

    def test_atom_imbalance_returns_false(self) -> None:
        assert determine_one_to_one_correspondence("CCO>>CCO.CCO", {}) is False

    def test_single_island_returns_true(self) -> None:
        islands = {0: {0, 1}}
        assert determine_one_to_one_correspondence("CCO>>CCO", islands) is True

    def test_multiple_islands_returns_false(self) -> None:
        islands = {0: {0}, 1: {1}}
        assert determine_one_to_one_correspondence("CCO>>CCO", islands) is False

    def test_atom_imbalance_overrides_islands(self) -> None:
        islands = {0: {0}}
        assert (
            determine_one_to_one_correspondence("CCO>>CCO.CCO", islands) is False
        )

    def test_empty_islands_balanced_returns_true(self) -> None:
        assert determine_one_to_one_correspondence("CC(=O)O.O>>CC(=O)O.O", {}) is True
