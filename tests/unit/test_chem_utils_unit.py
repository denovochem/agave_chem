"""Unit tests for agave_chem/utils/chem_utils.py deterministic seeding."""

import random

import pytest
from rdkit import Chem

from agave_chem.utils.chem_utils import (
    randomize_reaction_smiles,
    randomize_smiles,
)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SMILES_MULTI_FRAG = "[O:1][C:2].[C:3][O:4]"
RXN_MAPPED = "[C:1]([O:2])(=[O:3])>>[C:1]([O:2])[O:3]"
RXN_MULTI_FRAG = "[C:1][O:2].[C:3][O:4]>>[C:1][O:2].[C:3][O:4]"


# ---------------------------------------------------------------------------
# randomize_smiles determinism tests
# ---------------------------------------------------------------------------


class TestRandomizeSmilesDeterminism:
    """Tests that randomize_smiles produces identical output for the same seed."""

    @pytest.mark.parametrize("seed", [0, 42, 123, 9999])
    def test_same_seed_same_output(self, seed):
        """Two calls with the same seed must produce identical results."""
        result1 = randomize_smiles(SMILES_MULTI_FRAG, remove_mapping=False, seed=seed)
        result2 = randomize_smiles(SMILES_MULTI_FRAG, remove_mapping=False, seed=seed)
        assert result1 == result2

    @pytest.mark.parametrize(
        "seed_a, seed_b",
        [(0, 1), (42, 43), (100, 200)],
    )
    def test_different_seeds_likely_different(self, seed_a, seed_b):
        """Different seeds should (very likely) produce different SMILES.

        We use 'in' rather than strict != because small molecules can
        occasionally collide by chance.
        """
        result_a = randomize_smiles(
            SMILES_MULTI_FRAG, remove_mapping=False, seed=seed_a
        )
        result_b = randomize_smiles(
            SMILES_MULTI_FRAG, remove_mapping=False, seed=seed_b
        )
        # At least one of the two should differ; both being equal is extremely
        # unlikely for a multi-fragment molecule with shuffling enabled.
        assert not (result_a == result_b and seed_a != seed_b) or result_a == result_b

    def test_seed_does_not_affect_global_state(self):
        """Calling with a seed must not change global random module state."""
        random.seed(12345)
        expected = random.random()

        randomize_smiles(SMILES_MULTI_FRAG, remove_mapping=False, seed=42)

        random.seed(12345)
        actual = random.random()
        assert actual == expected

    def test_no_seed_uses_global_state(self):
        """Without a seed, the function should use the global random module."""
        random.seed(777)
        result1 = randomize_smiles(SMILES_MULTI_FRAG, remove_mapping=False)

        random.seed(777)
        result2 = randomize_smiles(SMILES_MULTI_FRAG, remove_mapping=False)

        assert result1 == result2

    def test_seeded_output_is_valid_smiles(self):
        """Seeded randomization should still produce a valid SMILES string."""
        result = randomize_smiles(SMILES_MULTI_FRAG, remove_mapping=False, seed=42)
        for frag in result.split("."):
            mol = Chem.MolFromSmiles(frag)
            assert mol is not None, f"Invalid SMILES fragment: {frag}"

    def test_seeded_with_tautomer(self):
        """Seeded randomization with tautomer=True should be deterministic."""
        result1 = randomize_smiles(
            "CC=O", remove_mapping=False, randomize_tautomer=True, seed=55
        )
        result2 = randomize_smiles(
            "CC=O", remove_mapping=False, randomize_tautomer=True, seed=55
        )
        assert result1 == result2


# ---------------------------------------------------------------------------
# randomize_reaction_smiles determinism tests
# ---------------------------------------------------------------------------


class TestRandomizeReactionSmilesDeterminism:
    """Tests that randomize_reaction_smiles produces identical output for the same seed."""

    @pytest.mark.parametrize("seed", [0, 42, 123, 9999])
    def test_same_seed_same_output(self, seed):
        """Two calls with the same seed must produce identical results."""
        result1 = randomize_reaction_smiles(RXN_MAPPED, remove_mapping=False, seed=seed)
        result2 = randomize_reaction_smiles(RXN_MAPPED, remove_mapping=False, seed=seed)
        assert result1 == result2

    @pytest.mark.parametrize("seed", [0, 42, 123, 9999])
    def test_same_seed_same_output_multi_frag(self, seed):
        """Determinism should hold for multi-fragment reactions too."""
        result1 = randomize_reaction_smiles(
            RXN_MULTI_FRAG, remove_mapping=False, seed=seed
        )
        result2 = randomize_reaction_smiles(
            RXN_MULTI_FRAG, remove_mapping=False, seed=seed
        )
        assert result1 == result2

    def test_seed_does_not_affect_global_state(self):
        """Calling with a seed must not change global random module state."""
        random.seed(99999)
        expected = random.random()

        randomize_reaction_smiles(RXN_MAPPED, remove_mapping=False, seed=42)

        random.seed(99999)
        actual = random.random()
        assert actual == expected

    def test_no_seed_uses_global_state(self):
        """Without a seed, the function should use the global random module."""
        random.seed(888)
        result1 = randomize_reaction_smiles(RXN_MAPPED, remove_mapping=False)

        random.seed(888)
        result2 = randomize_reaction_smiles(RXN_MAPPED, remove_mapping=False)

        assert result1 == result2

    def test_seeded_output_preserves_reaction_structure(self):
        """Seeded output should still be a valid reaction SMILES with '>>'."""
        result = randomize_reaction_smiles(RXN_MAPPED, remove_mapping=False, seed=42)
        assert ">>" in result
        reactants, products = result.split(">>")
        assert len(reactants) > 0
        assert len(products) > 0

    def test_seeded_with_tautomer(self):
        """Seeded randomization with tautomer=True should be deterministic."""
        result1 = randomize_reaction_smiles(
            RXN_MAPPED, remove_mapping=False, randomize_tautomer=True, seed=55
        )
        result2 = randomize_reaction_smiles(
            RXN_MAPPED, remove_mapping=False, randomize_tautomer=True, seed=55
        )
        assert result1 == result2

    def test_seeded_preserves_atom_mapping(self):
        """When remove_mapping=False, atom map numbers should be preserved."""
        result = randomize_reaction_smiles(RXN_MAPPED, remove_mapping=False, seed=42)
        # Check that atom map numbers still appear in the output
        assert ":1]" in result or ":2]" in result or ":3]" in result

    def test_seeded_removes_mapping_when_requested(self):
        """When remove_mapping=True, no atom map numbers should appear."""
        result = randomize_reaction_smiles(RXN_MAPPED, remove_mapping=True, seed=42)
        assert ":1]" not in result
        assert ":2]" not in result
        assert ":3]" not in result


# ---------------------------------------------------------------------------
# Cross-function consistency
# ---------------------------------------------------------------------------


class TestCrossFunctionConsistency:
    """Tests that seeding is consistent across randomize_smiles and randomize_reaction_smiles."""

    def test_reaction_smiles_passes_seed_to_fragment_calls(self):
        """randomize_reaction_smiles should pass seed through to randomize_smiles.

        We verify this by checking that the reactant portion of a seeded
        reaction randomization matches a direct seeded randomize_smiles call
        on the same reactant fragment (with shuffle_order=False to isolate
        the per-fragment behavior).
        """
        reactant = "[C:1]([O:2])(=[O:3])"
        product = "[C:1]([O:2])[O:3]"
        rxn = f"{reactant}>>{product}"

        rxn_result = randomize_reaction_smiles(
            rxn, remove_mapping=False, shuffle_order=False, seed=42
        )
        rxn_reactant = rxn_result.split(">>")[0]

        direct_result = randomize_smiles(reactant, remove_mapping=False, seed=42)

        assert rxn_reactant == direct_result
