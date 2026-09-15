"""Unit tests for agave_chem.main validation and orchestration logic."""

from unittest.mock import patch

import pytest

from agave_chem.main import (
    _validate_and_normalize_input,
    map_reactions,
    map_reactions_using_mappers,
)
from agave_chem.mappers.reaction_mapper import ReactionMapper
from agave_chem.mappers.types import ReactionInput, ReactionMapperResult


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
    ):
        super().__init__(mapper_type, mapper_name, mapper_weight)
        self._mappings = mappings or []
        self._classification_info = classification_info or {}
        self._mapping_score = mapping_score

    def map_reaction(self, reaction_smiles: str) -> ReactionMapperResult:
        return self.map_reactions([reaction_smiles])[0]

    def map_reactions(
        self, reaction_smiles_list: list[str] | list[ReactionInput]
    ) -> list[ReactionMapperResult]:
        results: list[ReactionMapperResult] = []
        for i, rxn in enumerate(reaction_smiles_list):
            original = rxn.stripped_smiles if isinstance(rxn, ReactionInput) else rxn
            mapping = self._mappings[i] if i < len(self._mappings) else original
            results.append(
                ReactionMapperResult(
                    original_smiles=original,
                    selected_mapping=mapping,
                    mapping_type=self._mapper_type,
                    mapping_score=self._mapping_score,
                    classification_info=self._classification_info,
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
        rxns, mappers = _validate_and_normalize_input("CC>>CC", valid_mappers)
        assert rxns == ["CC>>CC"]
        assert mappers == valid_mappers

    def test_list_input_passes_through(self, valid_mappers):
        rxns, _ = _validate_and_normalize_input(["CC>>CC", "CCO>>CCO"], valid_mappers)
        assert rxns == ["CC>>CC", "CCO>>CCO"]

    def test_empty_list_raises(self, valid_mappers):
        with pytest.raises(ValueError, match="non-empty list of strings"):
            _validate_and_normalize_input([], valid_mappers)

    def test_non_string_element_raises(self, valid_mappers):
        with pytest.raises(TypeError, match="non-empty list of strings"):
            _validate_and_normalize_input(["CC>>CC", 42], valid_mappers)

    def test_non_list_non_string_input_raises(self, valid_mappers):
        with pytest.raises(TypeError, match="non-empty list of strings"):
            _validate_and_normalize_input(42, valid_mappers)

    def test_duplicates_removed_order_preserved(self, valid_mappers):
        rxns, _ = _validate_and_normalize_input(
            ["CC>>CC", "CCO>>CCO", "CC>>CC", "CCC>>CCC"],
            valid_mappers,
        )
        assert rxns == ["CC>>CC", "CCO>>CCO", "CCC>>CCC"]

    def test_empty_mappers_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list of ReactionMapper"):
            _validate_and_normalize_input(["CC>>CC"], [])

    def test_none_mappers_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list of ReactionMapper"):
            _validate_and_normalize_input(["CC>>CC"], None)

    def test_non_reaction_mapper_instance_raises(self):
        with pytest.raises(TypeError, match="not an instance of ReactionMapper"):
            _validate_and_normalize_input(["CC>>CC"], ["not_a_mapper"])

    def test_duplicate_mapper_names_raises(self):
        mappers = [_StubMapper(mapper_name="dup"), _StubMapper(mapper_name="dup")]
        with pytest.raises(ValueError, match="Duplicate mapper name: dup"):
            _validate_and_normalize_input(["CC>>CC"], mappers)


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
            map_reactions_using_mappers([], [_StubMapper()])

    def test_results_preserve_input_order(self):
        rxns = ["CC>>CC", "CCO>>CCO", "CCC>>CCC"]
        mapper = _StubMapper(mapper_name="stub")
        results = map_reactions_using_mappers(rxns, [mapper])
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
        results = map_reactions_using_mappers(rxns, [mapper_empty, mapper_filled])
        assert results[0].final_mapping == "[C:1][C:2]>>[C:1][C:2]"

    def test_final_mapping_empty_when_all_mappers_fail(self):
        rxns = ["CC>>CC"]
        mapper = _StubMapper(mapper_name="fail", mappings=[""])
        results = map_reactions_using_mappers(rxns, [mapper])
        assert results[0].final_mapping == ""

    def test_mapper_results_collected_per_reaction(self):
        rxns = ["CC>>CC"]
        m1 = _StubMapper(mapper_name="m1", mappings=["map1"])
        m2 = _StubMapper(mapper_name="m2", mappings=["map2"])
        results = map_reactions_using_mappers(
            rxns, [m1, m2], return_detailed_mapper_info=True
        )
        assert len(results[0].mapper_results) == 2
        assert results[0].mapper_results[0].selected_mapping == "map1"
        assert results[0].mapper_results[1].selected_mapping == "map2"

    def test_mapper_results_empty_by_default(self):
        rxns = ["CC>>CC"]
        m1 = _StubMapper(mapper_name="m1", mappings=["map1"])
        m2 = _StubMapper(mapper_name="m2", mappings=["map2"])
        results = map_reactions_using_mappers(rxns, [m1, m2])
        assert results[0].mapper_results == []
        assert results[0].final_mapping == "map2"

    def test_mapper_results_empty_when_return_detailed_false(self):
        rxns = ["CC>>CC"]
        m1 = _StubMapper(mapper_name="m1", mappings=["map1"])
        results = map_reactions_using_mappers(
            rxns, [m1], return_detailed_mapper_info=False
        )
        assert results[0].mapper_results == []
        assert results[0].final_mapping == "map1"

    def test_string_input_accepted(self):
        mapper = _StubMapper(mapper_name="stub")
        results = map_reactions_using_mappers("CC>>CC", [mapper])
        assert len(results) == 1
        assert results[0].original_reaction == "CC>>CC"


class TestMcsInDetailedMapperInfo:
    """Tests for MCS result injection into mapper_results."""

    @pytest.fixture(autouse=True)
    def _mock_identical_fragment_mapper(self):
        """Replace IdenticalFragmentMapper with a pass-through to avoid SMILES parsing."""
        with patch(
            "agave_chem.main.IdenticalFragmentMapper",
            _PassThroughIdenticalFragmentMapper,
        ):
            yield

    def test_mcs_included_when_detailed_info_true(self):
        """MCS result is prepended to mapper_results when return_detailed_mapper_info=True
        and MCS was run as pre-processing (neural mapper present, no explicit MCS mapper)."""
        mcs_result = ReactionMapperResult(
            original_smiles="CC>>CC",
            selected_mapping="[C:1]>>[C:1]",
            mapping_type="mcs",
        )
        neural = _StubMapper(
            mapper_name="neural", mapper_type="neural", mappings=["[C:1]>>[C:1]"]
        )
        with patch("agave_chem.main._prepare_reaction_inputs") as mock_prep:
            mock_prep.return_value = (
                [
                    ReactionInput(
                        stripped_smiles="CC>>CC", one_to_one_correspondence=True
                    )
                ],
                [[]],
                [mcs_result],
            )
            results = map_reactions_using_mappers(
                ["CC>>CC"], [neural], return_detailed_mapper_info=True
            )
        assert len(results[0].mapper_results) == 2
        assert results[0].mapper_results[0].mapping_type == "mcs"
        assert results[0].mapper_results[0].selected_mapping == "[C:1]>>[C:1]"
        assert results[0].mapper_results[1].mapping_type == "neural"

    def test_mcs_not_included_when_detailed_info_false(self):
        """MCS result is not included when return_detailed_mapper_info=False."""
        mcs_result = ReactionMapperResult(
            original_smiles="CC>>CC",
            selected_mapping="[C:1]>>[C:1]",
            mapping_type="mcs",
        )
        neural = _StubMapper(
            mapper_name="neural", mapper_type="neural", mappings=["[C:1]>>[C:1]"]
        )
        with patch("agave_chem.main._prepare_reaction_inputs") as mock_prep:
            mock_prep.return_value = (
                [
                    ReactionInput(
                        stripped_smiles="CC>>CC", one_to_one_correspondence=True
                    )
                ],
                [[]],
                [mcs_result],
            )
            results = map_reactions_using_mappers(["CC>>CC"], [neural])
        assert results[0].mapper_results == []

    def test_mcs_not_included_when_explicit_mcs_mapper(self):
        """MCS result is not prepended when user explicitly passes an MCS mapper,
        since that result is already in mapper_results."""
        mcs_result = ReactionMapperResult(
            original_smiles="CC>>CC",
            selected_mapping="[C:1]>>[C:1]",
            mapping_type="mcs",
        )
        mcs_mapper = _StubMapper(
            mapper_name="mcs_explicit", mapper_type="mcs", mappings=["[C:1]>>[C:1]"]
        )
        with patch("agave_chem.main._prepare_reaction_inputs") as mock_prep:
            mock_prep.return_value = (
                [
                    ReactionInput(
                        stripped_smiles="CC>>CC", one_to_one_correspondence=True
                    )
                ],
                [[]],
                [mcs_result],
            )
            results = map_reactions_using_mappers(
                ["CC>>CC"], [mcs_mapper], return_detailed_mapper_info=True
            )
        # Only the explicit MCS mapper result, no prepended pre-processing MCS
        assert len(results[0].mapper_results) == 1
        assert results[0].mapper_results[0].mapping_type == "mcs"
        assert results[0].mapper_results[0].selected_mapping == "[C:1]>>[C:1]"

    def test_mcs_not_included_when_mcs_not_run(self):
        """No MCS result when needs_mcs is False (no neural/template mappers)."""
        stub = _StubMapper(mapper_name="stub", mapper_type="stub", mappings=["map1"])
        with patch("agave_chem.main._prepare_reaction_inputs") as mock_prep:
            mock_prep.return_value = (
                [
                    ReactionInput(
                        stripped_smiles="CC>>CC", one_to_one_correspondence=True
                    )
                ],
                [[]],
                [None],
            )
            results = map_reactions_using_mappers(
                ["CC>>CC"], [stub], return_detailed_mapper_info=True
            )
        assert len(results[0].mapper_results) == 1
        assert results[0].mapper_results[0].mapping_type == "stub"

    def test_mcs_does_not_affect_final_mapping(self):
        """final_mapping stays empty when neural/template both return empty,
        even though MCS has a non-empty mapping."""
        mcs_result = ReactionMapperResult(
            original_smiles="CC>>CC",
            selected_mapping="[C:1]>>[C:1]",
            mapping_type="mcs",
        )
        neural = _StubMapper(mapper_name="neural", mapper_type="neural", mappings=[""])
        with patch("agave_chem.main._prepare_reaction_inputs") as mock_prep:
            mock_prep.return_value = (
                [
                    ReactionInput(
                        stripped_smiles="CC>>CC", one_to_one_correspondence=True
                    )
                ],
                [[]],
                [mcs_result],
            )
            results = map_reactions_using_mappers(
                ["CC>>CC"], [neural], return_detailed_mapper_info=True
            )
        assert results[0].final_mapping == ""
        # MCS is still in mapper_results for informational purposes
        assert len(results[0].mapper_results) == 2
        assert results[0].mapper_results[0].mapping_type == "mcs"
        assert results[0].mapper_results[0].selected_mapping == "[C:1]>>[C:1]"

    def test_mcs_result_with_empty_mapping_still_included(self):
        """MCS result with empty selected_mapping is still included in mapper_results."""
        mcs_result = ReactionMapperResult(
            original_smiles="CC>>CC",
            selected_mapping="",
            mapping_type="mcs",
        )
        neural = _StubMapper(
            mapper_name="neural", mapper_type="neural", mappings=["[C:1]>>[C:1]"]
        )
        with patch("agave_chem.main._prepare_reaction_inputs") as mock_prep:
            mock_prep.return_value = (
                [
                    ReactionInput(
                        stripped_smiles="CC>>CC", one_to_one_correspondence=True
                    )
                ],
                [[]],
                [mcs_result],
            )
            results = map_reactions_using_mappers(
                ["CC>>CC"], [neural], return_detailed_mapper_info=True
            )
        assert len(results[0].mapper_results) == 2
        assert results[0].mapper_results[0].mapping_type == "mcs"
        assert results[0].mapper_results[0].selected_mapping == ""


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


# ---------------------------------------------------------------------------
# num_processes parameter
# ---------------------------------------------------------------------------


class TestNumProcesses:
    """Tests for the num_processes parameter in map_reactions."""

    @pytest.fixture(autouse=True)
    def _mock_identical_fragment_mapper(self):
        """Replace IdenticalFragmentMapper with a pass-through to avoid SMILES parsing."""
        with patch(
            "agave_chem.main.IdenticalFragmentMapper",
            _PassThroughIdenticalFragmentMapper,
        ):
            yield

    def test_num_processes_default_is_one(self):
        """Default num_processes is 1 (serial)."""
        mapper = _StubMapper(mapper_name="stub", mappings=["map1"])
        results = map_reactions(["CC>>CC"], mappers_list=[mapper])
        assert len(results) == 1
        assert results[0].final_mapping == "map1"

    def test_num_processes_one_with_explicit_mappers(self):
        """num_processes=1 with explicit mappers runs serially."""
        mapper = _StubMapper(mapper_name="stub", mappings=["map1"])
        results = map_reactions(["CC>>CC"], mappers_list=[mapper], num_processes=1)
        assert len(results) == 1
        assert results[0].final_mapping == "map1"

    def test_num_processes_passed_to_map_reactions_using_mappers(self):
        """num_processes is forwarded to map_reactions_using_mappers."""
        mapper = _StubMapper(mapper_name="stub", mappings=["map1"])
        with patch(
            "agave_chem.main.map_reactions_using_mappers",
            wraps=map_reactions_using_mappers,
        ) as mock:
            map_reactions(["CC>>CC"], mappers_list=[mapper], num_processes=4)
        assert mock.call_count == 1
        assert mock.call_args.kwargs["num_processes"] == 4

    def test_num_processes_greater_than_one_selects_parallel_default_mappers(self):
        """When num_processes > 1 and mappers_list is None, parallel default mappers are used."""
        with (
            patch("agave_chem.main._get_default_mappers_parallel") as mock_parallel,
            patch("agave_chem.main._get_default_mappers") as mock_serial,
        ):
            mock_parallel.return_value = (
                _StubMapper(mapper_name="neural", mapper_type="neural"),
                _StubMapper(mapper_name="template", mapper_type="template"),
            )
            map_reactions(["CC>>CC"], num_processes=4)
        assert mock_parallel.call_count == 1
        assert mock_parallel.call_args.args[0] == 4
        assert mock_serial.call_count == 0

    def test_num_processes_one_selects_serial_default_mappers(self):
        """When num_processes=1 and mappers_list is None, serial default mappers are used."""
        with (
            patch("agave_chem.main._get_default_mappers") as mock_serial,
            patch("agave_chem.main._get_default_mappers_parallel") as mock_parallel,
        ):
            mock_serial.return_value = (
                _StubMapper(mapper_name="neural", mapper_type="neural"),
                _StubMapper(mapper_name="template", mapper_type="template"),
            )
            map_reactions(["CC>>CC"])
        assert mock_serial.call_count == 1
        assert mock_parallel.call_count == 0

    def test_num_processes_zero_raises_value_error(self):
        """num_processes=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_processes"):
            map_reactions(["CC>>CC"], num_processes=0)

    def test_num_processes_negative_raises_value_error(self):
        """num_processes=-1 raises ValueError."""
        with pytest.raises(ValueError, match="num_processes"):
            map_reactions(["CC>>CC"], num_processes=-1)

    def test_num_processes_parallel_mcs_preprocessing_matches_serial(self):
        """Parallel MCS pre-processing (num_processes=2) produces same results as serial."""
        from agave_chem.mappers.mcs.mcs_mapper import MCSReactionMapper

        rxns = [
            "CCCCCO>>CCCCCO",
            "CCCCCO.O>>CCCCCO.O",
        ]

        # Serial
        serial_results = map_reactions(
            rxns,
            mappers_list=[MCSReactionMapper("test_mcs")],
            num_processes=1,
        )

        # Parallel (exercises real _prepare_reaction_inputs parallel MCS path)
        parallel_results = map_reactions(
            rxns,
            mappers_list=[MCSReactionMapper("test_mcs_par")],
            num_processes=2,
        )

        assert len(serial_results) == len(parallel_results)
        for s, p in zip(serial_results, parallel_results):
            assert s.final_mapping == p.final_mapping
