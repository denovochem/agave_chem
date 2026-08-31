"""Unit tests for reaction classification info in the template mapper."""

from __future__ import annotations

import pytest

from agave_chem.mappers.template.template_mapper import (
    TemplateReactionMapper,
    _make_composite_smirks_pattern,
)
from agave_chem.mappers.types import InitializedSmirksPattern, ReactionMapperResult


def _make_pattern(
    name: str = "Test Pattern",
    superclass_id: str = "1",
    class_id: str = "0",
    subclass_id: str = "0",
    subsubclass_id: str = "",
    rxno_classification: list[dict[str, str]] | None = None,
    priority: tuple[int, int] = (0, 0),
) -> InitializedSmirksPattern:
    """Build a minimal InitializedSmirksPattern for testing."""
    from rdchiral import main as rdc

    return InitializedSmirksPattern(
        name=name,
        superclass_id=superclass_id,
        class_id=class_id,
        subclass_id=subclass_id,
        subsubclass_id=subsubclass_id,
        class_str=f"{superclass_id}.{class_id}.{subclass_id}",
        products_smarts=[],
        reactants_smarts=[],
        products_fps=[],
        reactants_fps=[],
        rdc_rxn=rdc.rdchiralReaction("[C:1]>>[C:1]"),
        parent_smirks="[C:1]>>[C:1]",
        child_smirks="[C:1]>>[C:1]",
        template_name=name,
        priority=priority,
        rxno_classification=rxno_classification or [],
    )


# ---------------------------------------------------------------------------
# _make_composite_smirks_pattern
# ---------------------------------------------------------------------------


class TestMakeCompositeSmirksPattern:
    """Tests for _make_composite_smirks_pattern with classification fields."""

    def test_subsubclass_id_joined_unique(self):
        p1 = _make_pattern(subsubclass_id="1")
        p2 = _make_pattern(subsubclass_id="2")
        p3 = _make_pattern(subsubclass_id="1")
        composite = _make_composite_smirks_pattern(
            [p1, p2, p3], "[C:1]>>[C:1]", p1["rdc_rxn"]
        )
        assert composite["subsubclass_id"] == "1 + 2"

    def test_subsubclass_id_empty_when_all_empty(self):
        p1 = _make_pattern(subsubclass_id="")
        p2 = _make_pattern(subsubclass_id="")
        composite = _make_composite_smirks_pattern(
            [p1, p2], "[C:1]>>[C:1]", p1["rdc_rxn"]
        )
        assert composite["subsubclass_id"] == ""

    def test_rxno_classification_flattened_deduplicated(self):
        p1 = _make_pattern(
            rxno_classification=[
                {"rxno_id": "RXNO:0000001", "rxno_label": "", "rxno_definition": ""},
                {"rxno_id": "RXNO:0000002", "rxno_label": "", "rxno_definition": ""},
            ]
        )
        p2 = _make_pattern(
            rxno_classification=[
                {"rxno_id": "RXNO:0000002", "rxno_label": "", "rxno_definition": ""},
                {"rxno_id": "RXNO:0000003", "rxno_label": "", "rxno_definition": ""},
            ]
        )
        composite = _make_composite_smirks_pattern(
            [p1, p2], "[C:1]>>[C:1]", p1["rdc_rxn"]
        )
        assert [r["rxno_id"] for r in composite["rxno_classification"]] == [
            "RXNO:0000001",
            "RXNO:0000002",
            "RXNO:0000003",
        ]

    def test_rxno_classification_empty_when_all_empty(self):
        p1 = _make_pattern(rxno_classification=[])
        p2 = _make_pattern(rxno_classification=[])
        composite = _make_composite_smirks_pattern(
            [p1, p2], "[C:1]>>[C:1]", p1["rdc_rxn"]
        )
        assert composite["rxno_classification"] == []

    def test_rxno_classification_preserves_order(self):
        p1 = _make_pattern(
            rxno_classification=[
                {"rxno_id": "RXNO:0000003", "rxno_label": "", "rxno_definition": ""},
                {"rxno_id": "RXNO:0000001", "rxno_label": "", "rxno_definition": ""},
            ]
        )
        p2 = _make_pattern(
            rxno_classification=[
                {"rxno_id": "RXNO:0000002", "rxno_label": "", "rxno_definition": ""},
            ]
        )
        composite = _make_composite_smirks_pattern(
            [p1, p2], "[C:1]>>[C:1]", p1["rdc_rxn"]
        )
        assert [r["rxno_id"] for r in composite["rxno_classification"]] == [
            "RXNO:0000003",
            "RXNO:0000001",
            "RXNO:0000002",
        ]


# ---------------------------------------------------------------------------
# _initialize_smirks_patterns
# ---------------------------------------------------------------------------


class TestInitializeSmirksPatterns:
    """Tests for classification field population during SMIRKS initialization."""

    @pytest.fixture
    def mapper(self) -> TemplateReactionMapper:
        return TemplateReactionMapper(mapper_name="test_init")

    def test_subsubclass_id_not_none_string(self, mapper):
        """subsubclass_id should be '' when JSON value is null, not 'None'."""
        mapper._initialize_smirks_patterns()
        patterns = mapper._initialized_smirks_patterns
        assert patterns is not None
        for p in patterns:
            assert p["subsubclass_id"] != "None"
            assert isinstance(p["subsubclass_id"], str)

    def test_rxno_classification_is_list_of_dicts(self, mapper):
        """rxno_classification should be a list of dicts with rxno_id keys."""
        mapper._initialize_smirks_patterns()
        patterns = mapper._initialized_smirks_patterns
        assert patterns is not None
        for p in patterns:
            assert isinstance(p["rxno_classification"], list)
            for rxno in p["rxno_classification"]:
                assert isinstance(rxno, dict)
                assert "rxno_id" in rxno
                assert rxno["rxno_id"].startswith("RXNO:") or rxno["rxno_id"] == ""

    def test_all_fields_present(self, mapper):
        """All InitializedSmirksPattern fields should be present after init."""
        mapper._initialize_smirks_patterns()
        patterns = mapper._initialized_smirks_patterns
        assert patterns is not None
        assert len(patterns) > 0
        required_keys = {
            "name",
            "superclass_id",
            "class_id",
            "subclass_id",
            "subsubclass_id",
            "class_str",
            "template_name",
            "rxno_classification",
        }
        for p in patterns:
            assert required_keys.issubset(p.keys())


# ---------------------------------------------------------------------------
# map_reaction classification_info
# ---------------------------------------------------------------------------


class TestMapReactionClassificationInfo:
    """Tests for classification_info in map_reaction results."""

    @pytest.fixture
    def mapper(self) -> TemplateReactionMapper:
        return TemplateReactionMapper(mapper_name="test_classify")

    def test_failed_mapping_has_empty_classification_info(self, mapper):
        """When no template matches, classification_info should be empty dict."""
        result = mapper.map_reaction("CC>>CCO")
        assert isinstance(result, ReactionMapperResult)
        assert result.classification_info == {}

    def test_invalid_reaction_has_empty_classification_info(self, mapper):
        """Invalid reaction SMILES should produce empty classification_info."""
        result = mapper.map_reaction("invalid_smiles")
        assert result.classification_info == {}

    def test_successful_mapping_has_classification_info(self, mapper):
        """A successfully mapped reaction should have non-empty classification_info."""
        result = mapper.map_reaction(
            "[CH2:0]([Cl])[c:1]1ccccc1.[CH2:2]([O:3][H])>>[CH2:0]([O:3])[c:1]1ccccc1.[Cl][H]"
        )
        if result.selected_mapping:
            assert result.classification_info
            selected_key = result.selected_mapping
            assert selected_key in result.classification_info
            entries = result.classification_info[selected_key]
            assert len(entries) > 0
            entry = entries[0]
            assert "template_name" in entry
            assert "class_str" in entry
            assert "class_id" in entry
            assert "subclass_id" in entry
            assert "subsubclass_id" in entry
            assert "superclass_id" in entry
            assert "rxno_classification" in entry

    def test_classification_info_keys_match_possible_mappings(self, mapper):
        """classification_info keys should match possible_mappings keys."""
        result = mapper.map_reaction(
            "[CH2:0]([Cl])[c:1]1ccccc1.[CH2:2]([O:3][H])>>[CH2:0]([O:3])[c:1]1ccccc1.[Cl][H]"
        )
        if result.selected_mapping:
            assert set(result.classification_info.keys()) == set(
                result.possible_mappings.keys()
            )

    def test_classification_info_subsubclass_not_none_string(self, mapper):
        """subsubclass_id in classification_info should never be 'None'."""
        result = mapper.map_reaction(
            "[CH2:0]([Cl])[c:1]1ccccc1.[CH2:2]([O:3][H])>>[CH2:0]([O:3])[c:1]1ccccc1.[Cl][H]"
        )
        if result.selected_mapping:
            for entries in result.classification_info.values():
                for entry in entries:
                    assert entry["subsubclass_id"] != "None"

    def test_rxno_classification_is_list_of_dicts_in_result(self, mapper):
        """rxno_classification in classification_info should be list of dicts."""
        result = mapper.map_reaction(
            "[CH2:0]([Cl])[c:1]1ccccc1.[CH2:2]([O:3][H])>>[CH2:0]([O:3])[c:1]1ccccc1.[Cl][H]"
        )
        if result.selected_mapping:
            for entries in result.classification_info.values():
                for entry in entries:
                    assert isinstance(entry["rxno_classification"], list)
                    for rxno in entry["rxno_classification"]:
                        assert isinstance(rxno, dict)
                        assert "rxno_id" in rxno


# ---------------------------------------------------------------------------
# Non-template mappers: classification_info defaults to {}
# ---------------------------------------------------------------------------


class TestNonTemplateMappersClassificationInfo:
    """Verify that non-template mappers leave classification_info empty."""

    def test_reaction_mapper_result_default_classification_info(self):
        """Default ReactionMapperResult should have empty classification_info."""
        result = ReactionMapperResult(
            original_smiles="CC>>CC",
            selected_mapping="",
            mapping_type="mcs",
        )
        assert result.classification_info == {}

    def test_reaction_mapper_result_with_custom_classification_info(self):
        """ReactionMapperResult should accept classification_info."""
        result = ReactionMapperResult(
            original_smiles="CC>>CC",
            selected_mapping="[C:1]>>[C:1]",
            mapping_type="template",
            classification_info={
                "[C:1]>>[C:1]": [
                    {
                        "template_name": "Test",
                        "class_id": "0",
                        "subclass_id": "0",
                        "subsubclass_id": "",
                        "superclass_id": "1",
                        "rxno_classification": [
                            {
                                "rxno_id": "RXNO:0000001",
                                "rxno_label": "",
                                "rxno_definition": "",
                            }
                        ],
                    }
                ]
            },
        )
        assert len(result.classification_info) == 1
        assert result.classification_info["[C:1]>>[C:1]"][0]["template_name"] == "Test"
