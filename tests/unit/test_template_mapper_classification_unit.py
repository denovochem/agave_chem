"""Unit tests for reaction classification info in the template mapper."""

from __future__ import annotations

import pytest

from agave_chem.mappers.template.template_mapper import (
    TemplateReactionMapper,
    _build_class_hierarchy,
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

    def test_lookup_class_names_returns_names_for_known_ids(self, mapper):
        """_lookup_class_names should return non-empty names for valid hierarchy IDs."""
        result = mapper._lookup_class_names("1", "1", "1", "")
        assert isinstance(result["superclass_name"], str)
        assert isinstance(result["class_name"], str)
        assert isinstance(result["subclass_name"], str)

    def test_lookup_class_names_returns_empty_for_unknown_ids(self, mapper):
        """_lookup_class_names should return empty strings for unknown hierarchy IDs."""
        result = mapper._lookup_class_names("99", "99", "99", "99")
        assert result["superclass_name"] == ""
        assert result["class_name"] == ""
        assert result["subclass_name"] == ""
        assert result["subsubclass_name"] == ""

    def test_lookup_class_names_falls_back_when_class_has_no_subclasses(self, mapper):
        """_lookup_class_names should still return superclass and class names
        when a class has no subclasses but the smirks pattern has subclass_id='0'."""
        result = mapper._lookup_class_names("8", "0", "0", "")
        assert result["superclass_name"] == "Heterocycle Formation"
        assert result["class_name"] == "Unspecified"
        # No subclasses defined for class 0, so these stay empty
        assert result["subclass_name"] == ""
        assert result["subsubclass_name"] == ""


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
            assert "superclass_name" in entry
            assert "superclass_description" in entry
            assert "class_name" in entry
            assert "class_description" in entry
            assert "subclass_name" in entry
            assert "subclass_description" in entry
            assert "subsubclass_name" in entry
            assert "subsubclass_description" in entry
            assert "rxno_classification" in entry

    def test_class_names_are_non_empty_for_known_reaction(self, mapper):
        """classification_info entries should have non-empty class names for known reactions."""
        result = mapper.map_reaction(
            "[CH2:0]([Cl])[c:1]1ccccc1.[CH2:2]([O:3][H])>>[CH2:0]([O:3])[c:1]1ccccc1.[Cl][H]"
        )
        if result.selected_mapping:
            for entries in result.classification_info.values():
                for entry in entries:
                    assert isinstance(entry["superclass_name"], str)
                    assert isinstance(entry["class_name"], str)
                    assert isinstance(entry["subclass_name"], str)
                    assert isinstance(entry["superclass_description"], str)
                    assert isinstance(entry["class_description"], str)
                    assert isinstance(entry["subclass_description"], str)

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

    def test_classification_info_templates_ordered_by_priority(self, mapper):
        """Templates within each mapping should be ordered by priority (highest first)."""
        result = mapper.map_reaction(
            "[CH2:0]([Cl])[c:1]1ccccc1.[CH2:2]([O:3][H])>>[CH2:0]([O:3])[c:1]1ccccc1.[Cl][H]"
        )
        if result.selected_mapping:
            for entries in result.classification_info.values():
                priorities = [
                    (e.get("class_id", ""), e.get("subclass_id", "")) for e in entries
                ]
                # Verify non-increasing order (sorted descending)
                for i in range(len(priorities) - 1):
                    assert priorities[i] >= priorities[i + 1]


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


# ---------------------------------------------------------------------------
# _build_class_hierarchy
# ---------------------------------------------------------------------------


class TestBuildClassHierarchy:
    """Tests for _build_class_hierarchy helper."""

    def test_hierarchy_returns_nested_tree(self):
        """Hierarchy should nest classes under superclasses with children dicts."""
        data = {
            "superclasses": [
                {
                    "id": 1,
                    "name": "Heteroatom Alkylation",
                    "description": "Heteroatom alkylation reactions",
                    "classes": [
                        {
                            "id": 1,
                            "name": "N-alkylation",
                            "description": "N-alkylation reactions",
                            "subclasses": [
                                {
                                    "id": 1,
                                    "name": "Reductive amination",
                                    "description": "Reductive amination",
                                    "subsubclasses": [],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        tree = _build_class_hierarchy(data)
        assert "1" in tree
        assert tree["1"]["name"] == "Heteroatom Alkylation"
        assert tree["1"]["description"] == "Heteroatom alkylation reactions"
        assert "1" in tree["1"]["children"]
        c_node = tree["1"]["children"]["1"]
        assert c_node["name"] == "N-alkylation"
        assert c_node["description"] == "N-alkylation reactions"
        assert "1" in c_node["children"]
        sub_node = c_node["children"]["1"]
        assert sub_node["name"] == "Reductive amination"
        assert sub_node["description"] == "Reductive amination"
        assert sub_node["children"] == {}

    def test_hierarchy_with_subsubclasses(self):
        """Hierarchy should nest subsubclasses under subclasses."""
        data = {
            "superclasses": [
                {
                    "id": 2,
                    "name": "Acylation",
                    "description": "Acylation reactions",
                    "classes": [
                        {
                            "id": 5,
                            "name": "Amide formation",
                            "description": "Amide bond formation",
                            "subclasses": [
                                {
                                    "id": 1,
                                    "name": "Carbodiimide coupling",
                                    "description": "Carbodiimide-mediated coupling",
                                    "subsubclasses": [
                                        {
                                            "id": 1,
                                            "name": "EDC coupling",
                                            "description": "EDC/HOBt coupling",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        tree = _build_class_hierarchy(data)
        subsub_node = tree["2"]["children"]["5"]["children"]["1"]["children"]["1"]
        assert subsub_node["name"] == "EDC coupling"
        assert subsub_node["description"] == "EDC/HOBt coupling"

    def test_hierarchy_empty_data(self):
        """Hierarchy should return empty dict for empty input."""
        tree = _build_class_hierarchy({})
        assert tree == {}

    def test_hierarchy_no_superclasses(self):
        """Hierarchy should return empty dict when no superclasses are present."""
        tree = _build_class_hierarchy({"superclasses": []})
        assert tree == {}


# ---------------------------------------------------------------------------
# _sort_patterns_by_priority
# ---------------------------------------------------------------------------


def _make_pattern_with_priority(
    name: str = "Test Pattern",
    priority: tuple[int, int] = (0, 0),
) -> InitializedSmirksPattern:
    """Build a minimal InitializedSmirksPattern with a priority for sorting tests."""
    from rdchiral import main as rdc

    return InitializedSmirksPattern(
        name=name,
        superclass_id="1",
        class_id="0",
        subclass_id="0",
        subsubclass_id="",
        class_str="1.0.0",
        products_smarts=[],
        reactants_smarts=[],
        products_fps=[],
        reactants_fps=[],
        rdc_rxn=rdc.rdchiralReaction("[C:1]>>[C:1]"),
        parent_smirks="[C:1]>>[C:1]",
        child_smirks="[C:1]>>[C:1]",
        template_name=name,
        priority=priority,
        rxno_classification=[],
    )


class TestSortPatternsByPriority:
    """Tests for TemplateReactionMapper._sort_patterns_by_priority."""

    def test_empty_list_returns_empty(self):
        result = TemplateReactionMapper._sort_patterns_by_priority([])
        assert result == []

    def test_single_pattern_unchanged(self):
        p = _make_pattern_with_priority(name="A", priority=(1, 0))
        result = TemplateReactionMapper._sort_patterns_by_priority([p])
        assert len(result) == 1
        assert result[0]["template_name"] == "A"

    def test_sorted_by_priority_class_descending(self):
        p_low = _make_pattern_with_priority(name="Low", priority=(0, 0))
        p_high = _make_pattern_with_priority(name="High", priority=(3, 5))
        result = TemplateReactionMapper._sort_patterns_by_priority([p_low, p_high])
        assert result[0]["template_name"] == "High"
        assert result[1]["template_name"] == "Low"

    def test_priority_subclass_tiebreaker(self):
        p_pri_1 = _make_pattern_with_priority(name="Pri1", priority=(1, 1))
        p_pri_5 = _make_pattern_with_priority(name="Pri5", priority=(1, 5))
        result = TemplateReactionMapper._sort_patterns_by_priority([p_pri_1, p_pri_5])
        assert result[0]["template_name"] == "Pri5"
        assert result[1]["template_name"] == "Pri1"

    def test_equal_priority_preserves_original_order(self):
        p_a = _make_pattern_with_priority(name="A", priority=(0, 0))
        p_b = _make_pattern_with_priority(name="B", priority=(0, 0))
        result = TemplateReactionMapper._sort_patterns_by_priority([p_a, p_b])
        assert result[0]["template_name"] == "A"
        assert result[1]["template_name"] == "B"

    def test_three_patterns_ordered_correctly(self):
        p_generic = _make_pattern_with_priority(name="Generic", priority=(0, 0))
        p_specific = _make_pattern_with_priority(name="Specific", priority=(3, 5))
        p_mid = _make_pattern_with_priority(name="Mid", priority=(1, 0))
        result = TemplateReactionMapper._sort_patterns_by_priority(
            [p_generic, p_specific, p_mid]
        )
        assert [r["template_name"] for r in result] == ["Specific", "Mid", "Generic"]

    def test_returns_new_list(self):
        p = _make_pattern_with_priority(name="A", priority=(0, 0))
        original = [p]
        result = TemplateReactionMapper._sort_patterns_by_priority(original)
        assert result is not original

    def test_atom_count_tiebreaker_when_priority_equal(self):
        """When priorities are equal, more atoms should rank higher."""
        from rdkit import Chem

        p_few = InitializedSmirksPattern(
            name="Few",
            superclass_id="1",
            class_id="0",
            subclass_id="0",
            subsubclass_id="",
            class_str="1.0.0",
            products_smarts=[Chem.MolFromSmarts("[C:1]")],
            reactants_smarts=[Chem.MolFromSmarts("[C:1]")],
            products_fps=[],
            reactants_fps=[],
            rdc_rxn=None,  # type: ignore[arg-type]
            parent_smirks="[C:1]>>[C:1]",
            child_smirks="[C:1]>>[C:1]",
            template_name="Few",
            priority=(0, 0),
            rxno_classification=[],
        )
        p_many = InitializedSmirksPattern(
            name="Many",
            superclass_id="1",
            class_id="0",
            subclass_id="0",
            subsubclass_id="",
            class_str="1.0.0",
            products_smarts=[Chem.MolFromSmarts("[C:1][N:2]")],
            reactants_smarts=[Chem.MolFromSmarts("[C:1]"), Chem.MolFromSmarts("[N:2]")],
            products_fps=[],
            reactants_fps=[],
            rdc_rxn=None,  # type: ignore[arg-type]
            parent_smirks="[C:1].[N:2]>>[C:1][N:2]",
            child_smirks="[C:1].[N:2]>>[C:1][N:2]",
            template_name="Many",
            priority=(0, 0),
            rxno_classification=[],
        )
        result = TemplateReactionMapper._sort_patterns_by_priority([p_few, p_many])
        assert result[0]["template_name"] == "Many"
        assert result[1]["template_name"] == "Few"

    def test_priority_takes_precedence_over_atom_count(self):
        """Higher priority with fewer atoms should rank above lower priority with more atoms."""
        from rdkit import Chem

        p_low_many = InitializedSmirksPattern(
            name="LowMany",
            superclass_id="1",
            class_id="0",
            subclass_id="0",
            subsubclass_id="",
            class_str="1.0.0",
            products_smarts=[Chem.MolFromSmarts("[C:1][N:2][O:3]")],
            reactants_smarts=[
                Chem.MolFromSmarts("[C:1]"),
                Chem.MolFromSmarts("[N:2]"),
                Chem.MolFromSmarts("[O:3]"),
            ],
            products_fps=[],
            reactants_fps=[],
            rdc_rxn=None,  # type: ignore[arg-type]
            parent_smirks="[C:1].[N:2].[O:3]>>[C:1][N:2][O:3]",
            child_smirks="[C:1].[N:2].[O:3]>>[C:1][N:2][O:3]",
            template_name="LowMany",
            priority=(0, 0),
            rxno_classification=[],
        )
        p_high_few = _make_pattern_with_priority(name="HighFew", priority=(1, 0))
        result = TemplateReactionMapper._sort_patterns_by_priority(
            [p_low_many, p_high_few]
        )
        assert result[0]["template_name"] == "HighFew"
        assert result[1]["template_name"] == "LowMany"
