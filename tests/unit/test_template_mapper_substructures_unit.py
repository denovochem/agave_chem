"""Unit tests for TemplateReactionMapper._are_fragments_substructures.

Regression tests for hypervalent atom handling (e.g., P with valence 5).
"""

from agave_chem.mappers.template.template_mapper import TemplateReactionMapper


def _make_mapper() -> TemplateReactionMapper:
    """Build a TemplateReactionMapper without loading SMIRKS patterns."""
    return TemplateReactionMapper(
        mapper_name="test",
        use_default_smirks_patterns=False,
        use_mcs_mapping=False,
    )


class TestAreFragmentsSubstructuresHypervalent:
    """Tests that _are_fragments_substructures handles hypervalent atoms."""

    def test_hypervalent_phosphorus_does_not_crash(self):
        """Reactant with P(=O)(Cl)(Cl)Cl (valence 5) should not raise.

        Before the fix, UpdatePropertyCache() with strict=True would raise
        AtomValenceException for phosphorus with valence 5.
        """
        mapper = _make_mapper()

        # Missing fragment with a wildcard — triggers substructure matching
        missing_fragments = [("*CC", "[*:1]CC")]

        # Found fragments (empty — none found yet)
        found_fragments: list[tuple[str, str]] = []

        # Reactant containing hypervalent phosphorus (P with valence 5)
        unmapped_reactants = {"O=P(Cl)(Cl)Cl": ["O=P(Cl)(Cl)Cl"]}

        # Should not raise AtomValenceException
        result = mapper._are_fragments_substructures(
            missing_fragments, found_fragments, unmapped_reactants
        )
        assert isinstance(result, bool)

    def test_no_wildcard_fragments_returns_true(self):
        """Missing fragments without wildcards should return True (skipped)."""
        mapper = _make_mapper()

        missing_fragments = [("CC", "[C:1][C:2]")]
        found_fragments: list[tuple[str, str]] = []
        unmapped_reactants = {"CC": ["CC"]}

        result = mapper._are_fragments_substructures(
            missing_fragments, found_fragments, unmapped_reactants
        )
        assert result is True

    def test_wildcard_match_succeeds(self):
        """A wildcard fragment that is a substructure should return True."""
        mapper = _make_mapper()

        missing_fragments = [("*C", "[*:1]C")]
        found_fragments: list[tuple[str, str]] = []
        unmapped_reactants = {"CC": ["CC"]}

        result = mapper._are_fragments_substructures(
            missing_fragments, found_fragments, unmapped_reactants
        )
        assert result is True

    def test_wildcard_no_match_returns_false(self):
        """A wildcard fragment that is not a substructure should return False."""
        mapper = _make_mapper()

        missing_fragments = [("*N", "[*:1]N")]
        found_fragments: list[tuple[str, str]] = []
        unmapped_reactants = {"CC": ["CC"]}

        result = mapper._are_fragments_substructures(
            missing_fragments, found_fragments, unmapped_reactants
        )
        assert result is False
