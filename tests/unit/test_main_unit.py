"""Unit tests for agave_chem.main validation and orchestration logic."""

from unittest.mock import patch

import pytest

from agave_chem.main import (
    _validate_and_normalize_input,
    map_reactions,
    map_reactions_using_mappers,
)
from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.types import ReactionMapperResult


class _PassThroughIdenticalFragmentMapper:
    """Drop-in replacement for IdenticalFragmentMapper that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def create_identical_fragments_mapping_list(self, reaction_smiles_list):
        return list(reaction_smiles_list), [[] for _ in reaction_smiles_list]

    def resolve_identical_fragments_mapping_list(
        self, mapped_reaction_smiles_list, identical_fragments_mapping_list
    ):
        return list(mapped_reaction_smiles_list)


class _StubMapper(ReactionMapper):
    """Minimal ReactionMapper subclass for testing."""

    def __init__(
        self,
        mapper_name: str = "stub",
        mapper_type: str = "stub",
        mapper_weight: float = 1.0,
        mappings: list[str] | None = None,
        classification_info: dict[str, list[dict]] | None = None,
        mapping_score: float | None = None,
        ranked_mappings: list[str] | None = None,
    ):
        super().__init__(mapper_type, mapper_name, mapper_weight)
        self._mappings = mappings or []
        self._classification_info = classification_info or {}
        self._mapping_score = mapping_score
        self._ranked_mappings = ranked_mappings or []

    def map_reaction(self, reaction_smiles: str) -> ReactionMapperResult:
        return self.map_reactions([reaction_smiles])[0]

    def map_reactions(
        self, reaction_smiles_list: list[str]
    ) -> list[ReactionMapperResult]:
        results: list[ReactionMapperResult] = []
        for i, rxn in enumerate(reaction_smiles_list):
            mapping = self._mappings[i] if i < len(self._mappings) else rxn
            results.append(
                ReactionMapperResult(
                    original_smiles=rxn,
                    selected_mapping=mapping,
                    mapping_type=self._mapper_type,
                    mapping_score=self._mapping_score,
                    classification_info=self._classification_info,
                    ranked_mappings=self._ranked_mappings,
                )
            )
        return results


# ---------------------------------------------------------------------------
# _validate_and_normalize_input
# ---------------------------------------------------------------------------


class TestValidateAndNormalizeInput:
    """Tests for the _validate_and_normalize_input helper."""

    @pytest.fixture
    def valid_mappers(self) -> list[ReactionMapper]:
        return [_StubMapper(mapper_name="a"), _StubMapper(mapper_name="b")]

    def test_string_input_normalized_to_single_element_list(self, valid_mappers):
        rxns, mappers, batch = _validate_and_normalize_input(
            "CC>>CC", valid_mappers, 100
        )
        assert rxns == ["CC>>CC"]
        assert mappers == valid_mappers
        assert batch == 100

    def test_list_input_passes_through(self, valid_mappers):
        rxns, _, _ = _validate_and_normalize_input(
            ["CC>>CC", "CCO>>CCO"], valid_mappers, 100
        )
        assert rxns == ["CC>>CC", "CCO>>CCO"]

    def test_empty_list_raises(self, valid_mappers):
        with pytest.raises(ValueError, match="non-empty list of strings"):
            _validate_and_normalize_input([], valid_mappers, 100)

    def test_non_string_element_raises(self, valid_mappers):
        with pytest.raises(TypeError, match="non-empty list of strings"):
            _validate_and_normalize_input(["CC>>CC", 42], valid_mappers, 100)

    def test_non_list_non_string_input_raises(self, valid_mappers):
        with pytest.raises(TypeError, match="non-empty list of strings"):
            _validate_and_normalize_input(42, valid_mappers, 100)

    def test_duplicates_removed_order_preserved(self, valid_mappers):
        rxns, _, _ = _validate_and_normalize_input(
            ["CC>>CC", "CCO>>CCO", "CC>>CC", "CCC>>CCC"],
            valid_mappers,
            100,
        )
        assert rxns == ["CC>>CC", "CCO>>CCO", "CCC>>CCC"]

    def test_empty_mappers_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list of ReactionMapper"):
            _validate_and_normalize_input(["CC>>CC"], [], 100)

    def test_none_mappers_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list of ReactionMapper"):
            _validate_and_normalize_input(["CC>>CC"], None, 100)

    def test_non_reaction_mapper_instance_raises(self):
        with pytest.raises(TypeError, match="not an instance of ReactionMapper"):
            _validate_and_normalize_input(["CC>>CC"], ["not_a_mapper"], 100)

    def test_duplicate_mapper_names_raises(self):
        mappers = [_StubMapper(mapper_name="dup"), _StubMapper(mapper_name="dup")]
        with pytest.raises(ValueError, match="Duplicate mapper name: dup"):
            _validate_and_normalize_input(["CC>>CC"], mappers, 100)

    def test_non_int_batch_size_raises_type_error(self, valid_mappers):
        with pytest.raises(TypeError, match="batch_size must be an integer"):
            _validate_and_normalize_input(["CC>>CC"], valid_mappers, "100")

    def test_batch_size_zero_raises(self, valid_mappers):
        with pytest.raises(ValueError, match="between 1-1000"):
            _validate_and_normalize_input(["CC>>CC"], valid_mappers, 0)

    def test_batch_size_negative_raises(self, valid_mappers):
        with pytest.raises(ValueError, match="between 1-1000"):
            _validate_and_normalize_input(["CC>>CC"], valid_mappers, -1)

    def test_batch_size_over_1000_raises(self, valid_mappers):
        with pytest.raises(ValueError, match="between 1-1000"):
            _validate_and_normalize_input(["CC>>CC"], valid_mappers, 1001)

    def test_batch_size_boundary_values_accepted(self, valid_mappers):
        for bs in (1, 1000):
            _, _, batch = _validate_and_normalize_input(["CC>>CC"], valid_mappers, bs)
            assert batch == bs


# ---------------------------------------------------------------------------
# map_reactions_using_mappers
# ---------------------------------------------------------------------------


class TestMapReactionsUsingMappers:
    """Tests for the map_reactions_using_mappers function."""

    @pytest.fixture(autouse=True)
    def _mock_identical_fragment_mapper(self):
        """Replace IdenticalFragmentMapper with a pass-through to avoid SMILES parsing."""
        with patch(
            "agave_chem.main.IdenticalFragmentMapper",
            _PassThroughIdenticalFragmentMapper,
        ):
            yield

    def test_validation_runs_on_invalid_input(self):
        with pytest.raises(ValueError, match="non-empty list of strings"):
            map_reactions_using_mappers([], [_StubMapper()], 100)

    def test_results_preserve_input_order(self):
        rxns = ["CC>>CC", "CCO>>CCO", "CCC>>CCC"]
        mapper = _StubMapper(mapper_name="stub")
        results = map_reactions_using_mappers(rxns, [mapper], 100)
        assert len(results) == 3
        assert results[0].original_reaction == "CC>>CC"
        assert results[1].original_reaction == "CCO>>CCO"
        assert results[2].original_reaction == "CCC>>CCC"

    def test_final_mapping_is_last_non_empty(self):
        rxns = ["CC>>CC"]
        mapper_empty = _StubMapper(mapper_name="empty", mappings=[""])
        mapper_filled = _StubMapper(
            mapper_name="filled", mappings=["[C:1][C:2]>>[C:1][C:2]"]
        )
        results = map_reactions_using_mappers(rxns, [mapper_empty, mapper_filled], 100)
        assert results[0].final_mapping == "[C:1][C:2]>>[C:1][C:2]"

    def test_final_mapping_empty_when_all_mappers_fail(self):
        rxns = ["CC>>CC"]
        mapper = _StubMapper(mapper_name="fail", mappings=[""])
        results = map_reactions_using_mappers(rxns, [mapper], 100)
        assert results[0].final_mapping == ""

    def test_mapper_results_collected_per_reaction(self):
        rxns = ["CC>>CC"]
        m1 = _StubMapper(mapper_name="m1", mappings=["map1"])
        m2 = _StubMapper(mapper_name="m2", mappings=["map2"])
        results = map_reactions_using_mappers(
            rxns, [m1, m2], 100, return_detailed_mapper_info=True
        )
        assert len(results[0].mapper_results) == 2
        assert results[0].mapper_results[0].selected_mapping == "map1"
        assert results[0].mapper_results[1].selected_mapping == "map2"

    def test_mapper_results_empty_by_default(self):
        rxns = ["CC>>CC"]
        m1 = _StubMapper(mapper_name="m1", mappings=["map1"])
        m2 = _StubMapper(mapper_name="m2", mappings=["map2"])
        results = map_reactions_using_mappers(rxns, [m1, m2], 100)
        assert results[0].mapper_results == []
        assert results[0].final_mapping == "map2"

    def test_mapper_results_empty_when_return_detailed_false(self):
        rxns = ["CC>>CC"]
        m1 = _StubMapper(mapper_name="m1", mappings=["map1"])
        results = map_reactions_using_mappers(
            rxns, [m1], 100, return_detailed_mapper_info=False
        )
        assert results[0].mapper_results == []
        assert results[0].final_mapping == "map1"

    def test_batching_with_multiple_batches(self):
        rxns = [f"R{i}>>P{i}" for i in range(5)]
        mapper = _StubMapper(mapper_name="stub")
        results = map_reactions_using_mappers(rxns, [mapper], 2)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.original_reaction == f"R{i}>>P{i}"

    def test_string_input_accepted(self):
        mapper = _StubMapper(mapper_name="stub")
        results = map_reactions_using_mappers("CC>>CC", [mapper], 100)
        assert len(results) == 1
        assert results[0].original_reaction == "CC>>CC"


# ---------------------------------------------------------------------------
# map_reactions
# ---------------------------------------------------------------------------


class TestMapReactions:
    """Tests for the public map_reactions entry point."""

    @pytest.fixture(autouse=True)
    def _mock_identical_fragment_mapper(self):
        """Replace IdenticalFragmentMapper with a pass-through to avoid SMILES parsing."""
        with patch(
            "agave_chem.main.IdenticalFragmentMapper",
            _PassThroughIdenticalFragmentMapper,
        ):
            yield

    def test_string_input_accepted(self):
        with patch(
            "agave_chem.main._get_default_mappers",
            return_value=(_StubMapper(mapper_name="stub"),),
        ):
            results = map_reactions("CC>>CC")
        assert len(results) == 1
        assert results[0].original_reaction == "CC>>CC"

    def test_duplicate_reactions_deduplicated_order_preserved(self):
        with patch(
            "agave_chem.main._get_default_mappers",
            return_value=(_StubMapper(mapper_name="stub"),),
        ):
            results = map_reactions(["CC>>CC", "CCO>>CCO", "CC>>CC", "CCC>>CCC"])
        assert len(results) == 3
        assert results[0].original_reaction == "CC>>CC"
        assert results[1].original_reaction == "CCO>>CCO"
        assert results[2].original_reaction == "CCC>>CCC"

    def test_empty_list_raises(self):
        with (
            patch(
                "agave_chem.main._get_default_mappers",
                return_value=(_StubMapper(mapper_name="stub"),),
            ),
            pytest.raises(ValueError, match="non-empty list of strings"),
        ):
            map_reactions([])

    def test_invalid_mapping_selection_mode_raises(self):
        with pytest.raises(TypeError, match="mapping_selection_mode"):
            map_reactions(["CC>>CC"], mapping_selection_mode=42)

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="between 1-1000"):
            map_reactions(["CC>>CC"], batch_size=0)

    def test_custom_mappers_used_when_provided(self):
        mapper = _StubMapper(mapper_name="custom", mappings=["custom_map"])
        results = map_reactions(["CC>>CC"], mappers_list=[mapper])
        assert len(results) == 1
        assert results[0].final_mapping == "custom_map"

    def test_callable_mapping_selection_mode_accepted(self):
        mapper = _StubMapper(mapper_name="stub")
        results = map_reactions(
            ["CC>>CC"],
            mappers_list=[mapper],
            mapping_selection_mode=lambda x: x,
        )
        assert len(results) == 1

    def test_detailed_mapper_info_false_by_default(self):
        mapper = _StubMapper(mapper_name="stub", mappings=["map1"])
        results = map_reactions(["CC>>CC"], mappers_list=[mapper])
        assert results[0].mapper_results == []

    def test_detailed_mapper_info_true_populates_results(self):
        mapper = _StubMapper(mapper_name="stub", mappings=["map1"])
        results = map_reactions(
            ["CC>>CC"],
            mappers_list=[mapper],
            return_detailed_mapper_info=True,
        )
        assert len(results[0].mapper_results) == 1
        assert results[0].mapper_results[0].selected_mapping == "map1"

    def test_classification_fields_populated_by_default(self):
        mapping = "[C:1]>>[C:1]"
        classification_info = {
            mapping: [
                {
                    "template_name": "Amide coupling",
                    "class_str": "2.5.1",
                    "class_id": "5",
                    "subclass_id": "1",
                    "subsubclass_id": "",
                    "superclass_id": "2",
                    "rxno_classification": [
                        {
                            "rxno_id": "RXNO:0000357",
                            "rxno_label": "Amide formation",
                            "rxno_definition": "Formation of an amide bond.",
                        }
                    ],
                }
            ]
        }
        mapper = _StubMapper(
            mapper_name="template",
            mappings=[mapping],
            classification_info=classification_info,
        )
        results = map_reactions(["CC>>CC"], mappers_list=[mapper])
        assert results[0].class_str == "2.5.1"
        assert results[0].rxno_classifications == "RXNO:0000357"
        assert mapping in results[0].classification_info
        assert results[0].mapper_results == []

    def test_classification_fields_empty_without_template_mapper(self):
        mapper = _StubMapper(mapper_name="stub", mappings=["map1"])
        results = map_reactions(["CC>>CC"], mappers_list=[mapper])
        assert results[0].class_str == ""
        assert results[0].rxno_classifications == ""
        assert results[0].classification_info == {}

    def test_classification_fields_empty_when_final_mapping_empty(self):
        mapper = _StubMapper(
            mapper_name="template",
            mappings=[""],
            classification_info={"some_mapping": []},
        )
        results = map_reactions(["CC>>CC"], mappers_list=[mapper])
        assert results[0].final_mapping == ""
        assert results[0].class_str == ""
        assert results[0].rxno_classifications == ""
        assert results[0].classification_info == {}

    def test_classification_fields_with_multiple_templates(self):
        mapping = "[C:1]>>[C:1]"
        classification_info = {
            mapping: [
                {
                    "template_name": "Reductive amination",
                    "class_str": "1.1.1",
                    "class_id": "1",
                    "subclass_id": "1",
                    "subsubclass_id": "",
                    "superclass_id": "1",
                    "rxno_classification": [
                        {
                            "rxno_id": "RXNO:0000335",
                            "rxno_label": "",
                            "rxno_definition": "",
                        }
                    ],
                },
                {
                    "template_name": "Amide coupling",
                    "class_str": "2.5.1",
                    "class_id": "5",
                    "subclass_id": "1",
                    "subsubclass_id": "",
                    "superclass_id": "2",
                    "rxno_classification": [
                        {
                            "rxno_id": "RXNO:0000357",
                            "rxno_label": "",
                            "rxno_definition": "",
                        }
                    ],
                },
                {
                    "template_name": "Ester aminolysis",
                    "class_str": "2.5.2",
                    "class_id": "5",
                    "subclass_id": "2",
                    "subsubclass_id": "",
                    "superclass_id": "2",
                    "rxno_classification": [
                        {
                            "rxno_id": "RXNO:0000357",
                            "rxno_label": "",
                            "rxno_definition": "",
                        }
                    ],
                },
            ]
        }
        mapper = _StubMapper(
            mapper_name="template",
            mappings=[mapping],
            classification_info=classification_info,
        )
        results = map_reactions(["CC>>CC"], mappers_list=[mapper])
        assert results[0].class_str == "1.1.1|2.5.1|2.5.2"
        assert results[0].rxno_classifications == "RXNO:0000335|RXNO:0000357"
        assert len(results[0].classification_info[mapping]) == 3

    def test_classification_fields_populated_without_detailed_mapper_info(self):
        mapping = "[C:1]>>[C:1]"
        classification_info = {
            mapping: [
                {
                    "template_name": "Schotten-Baumann",
                    "class_str": "2.1.1",
                    "class_id": "1",
                    "subclass_id": "1",
                    "subsubclass_id": "",
                    "superclass_id": "2",
                    "rxno_classification": [
                        {
                            "rxno_id": "RXNO:0000165",
                            "rxno_label": "",
                            "rxno_definition": "",
                        }
                    ],
                }
            ]
        }
        mapper = _StubMapper(
            mapper_name="template",
            mappings=[mapping],
            classification_info=classification_info,
        )
        results = map_reactions(
            ["CC>>CC"],
            mappers_list=[mapper],
            return_detailed_mapper_info=False,
        )
        assert results[0].class_str == "2.1.1"
        assert results[0].rxno_classifications == "RXNO:0000165"
        assert results[0].mapper_results == []

    def test_confidence_populated_from_neural_mapper(self):
        mapping = "[C:1]>>[C:1]"
        neural = _StubMapper(
            mapper_name="neural",
            mapper_type="neural",
            mappings=[mapping],
            mapping_score=0.95,
        )
        results = map_reactions(["CC>>CC"], mappers_list=[neural])
        assert results[0].confidence == 0.95

    def test_confidence_none_without_neural_mapper(self):
        mapping = "[C:1]>>[C:1]"
        template = _StubMapper(
            mapper_name="template",
            mapper_type="template",
            mappings=[mapping],
        )
        results = map_reactions(["CC>>CC"], mappers_list=[template])
        assert results[0].confidence is None

    def test_confidence_none_when_neural_mapping_score_is_none(self):
        mapping = "[C:1]>>[C:1]"
        neural = _StubMapper(
            mapper_name="neural",
            mapper_type="neural",
            mappings=[mapping],
            mapping_score=None,
        )
        results = map_reactions(["CC>>CC"], mappers_list=[neural])
        assert results[0].confidence is None

    def test_confidence_populated_with_multiple_mappers(self):
        mapping = "[C:1]>>[C:1]"
        template = _StubMapper(
            mapper_name="template",
            mapper_type="template",
            mappings=[mapping],
        )
        neural = _StubMapper(
            mapper_name="neural",
            mapper_type="neural",
            mappings=[mapping],
            mapping_score=0.87,
        )
        results = map_reactions(["CC>>CC"], mappers_list=[template, neural])
        assert results[0].confidence == 0.87

    def test_ranked_mappings_populated_from_template_mapper(self):
        mapping = "[C:1]>>[C:1]"
        alt_mapping = "[C:2]>>[C:2]"
        template = _StubMapper(
            mapper_name="template",
            mapper_type="template",
            mappings=[mapping],
            ranked_mappings=[mapping, alt_mapping],
        )
        results = map_reactions(["CC>>CC"], mappers_list=[template])
        assert results[0].ranked_mappings == [mapping, alt_mapping]

    def test_ranked_mappings_empty_without_template_mapper(self):
        mapping = "[C:1]>>[C:1]"
        neural = _StubMapper(
            mapper_name="neural",
            mapper_type="neural",
            mappings=[mapping],
            mapping_score=0.9,
        )
        results = map_reactions(["CC>>CC"], mappers_list=[neural])
        assert results[0].ranked_mappings == []

    def test_ranked_mappings_empty_when_template_has_no_ranked_mappings(self):
        mapping = "[C:1]>>[C:1]"
        template = _StubMapper(
            mapper_name="template",
            mapper_type="template",
            mappings=[mapping],
        )
        results = map_reactions(["CC>>CC"], mappers_list=[template])
        assert results[0].ranked_mappings == []

    def test_ranked_mappings_populated_with_multiple_mappers(self):
        mapping = "[C:1]>>[C:1]"
        alt_mapping = "[C:2]>>[C:2]"
        template = _StubMapper(
            mapper_name="template",
            mapper_type="template",
            mappings=[mapping],
            ranked_mappings=[mapping, alt_mapping],
        )
        neural = _StubMapper(
            mapper_name="neural",
            mapper_type="neural",
            mappings=[mapping],
            mapping_score=0.87,
        )
        results = map_reactions(["CC>>CC"], mappers_list=[template, neural])
        assert results[0].ranked_mappings == [mapping, alt_mapping]
        assert results[0].confidence == 0.87
