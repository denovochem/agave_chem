"""Unit tests for pydantic data models in agave_chem."""

import pytest
from pydantic import ValidationError

from agave_chem.mappers.data_classes import (
    AtomMapping,
    BondChange,
    BondChangeType,
    MappingScore,
)
from agave_chem.mappers.neural.model import SupervisedConfig
from agave_chem.mappers.types import (
    AgaveChemMapperResult,
    ReactionMapperResult,
    SmirksPattern,
)


class TestSmirksPattern:
    """Tests for the SmirksPattern pydantic model."""

    def test_valid_construction(self):
        pattern = SmirksPattern(name="esterification", smirks="[C:1][O:2]>>[C:1][O:2]")
        assert pattern.name == "esterification"
        assert pattern.smirks == "[C:1][O:2]>>[C:1][O:2]"
        assert pattern.superclass_id is None
        assert pattern.class_id is None
        assert pattern.subclass_id is None

    def test_with_optional_ids(self):
        pattern = SmirksPattern(
            name="test",
            smirks=">>",
            superclass_id=1,
            class_id=2,
            subclass_id=3,
        )
        assert pattern.superclass_id == 1
        assert pattern.class_id == 2
        assert pattern.subclass_id == 3

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            SmirksPattern(smirks=">>")  # type: ignore[call-arg]

    def test_wrong_type_raises(self):
        with pytest.raises(ValidationError):
            SmirksPattern(name=123, smirks=">>")  # type: ignore[arg-type]


class TestReactionMapperResult:
    """Tests for the ReactionMapperResult pydantic model."""

    def test_default_values(self):
        result = ReactionMapperResult(mapping_type="test")
        assert result.original_smiles == ""
        assert result.selected_mapping == ""
        assert result.possible_mappings == {}
        assert result.mapping_type == "test"
        assert result.mapping_score is None
        assert result.additional_info == [{}]

    def test_full_construction(self):
        result = ReactionMapperResult(
            original_smiles="CC>>CC",
            selected_mapping="[C:1][C:2]>>[C:1][C:2]",
            possible_mappings={"[C:1][C:2]>>[C:1][C:2]": ["alkylation"]},
            mapping_type="template",
            mapping_score=0.95,
            additional_info=[{"key": "value"}],
        )
        assert result.original_smiles == "CC>>CC"
        assert result.selected_mapping == "[C:1][C:2]>>[C:1][C:2]"
        assert result.possible_mappings == {"[C:1][C:2]>>[C:1][C:2]": ["alkylation"]}
        assert result.mapping_score == 0.95

    def test_model_copy_preserves_original(self):
        result = ReactionMapperResult(mapping_type="test")
        copied = result.model_copy(update={"original_smiles": "CC>>CC"})
        assert result.original_smiles == ""
        assert copied.original_smiles == "CC>>CC"

    def test_is_picklable(self):
        import pickle

        result = ReactionMapperResult(
            original_smiles="CC>>CC",
            selected_mapping="[C:1]>>[C:1]",
            possible_mappings={"[C:1]>>[C:1]": ["test"]},
            mapping_type="template",
        )
        pickled = pickle.dumps(result)
        unpickled = pickle.loads(pickled)
        assert unpickled.original_smiles == "CC>>CC"
        assert unpickled.selected_mapping == "[C:1]>>[C:1]"
        assert unpickled.possible_mappings == {"[C:1]>>[C:1]": ["test"]}


class TestAgaveChemMapperResult:
    """Tests for the AgaveChemMapperResult pydantic model."""

    def test_default_values(self):
        result = AgaveChemMapperResult()
        assert result.final_mapping == ""
        assert result.original_reaction == ""
        assert result.mapper_results == []

    def test_with_mapper_results(self):
        mapper_result = ReactionMapperResult(mapping_type="mcs")
        result = AgaveChemMapperResult(
            final_mapping="[C:1]>>[C:1]",
            original_reaction="C>>C",
            mapper_results=[mapper_result],
        )
        assert result.final_mapping == "[C:1]>>[C:1]"
        assert len(result.mapper_results) == 1
        assert result.mapper_results[0].mapping_type == "mcs"


class TestAtomMapping:
    """Tests for the AtomMapping pydantic model."""

    def test_valid_construction(self):
        mapping = AtomMapping(
            reactant_mol_idx=0,
            reactant_atom_idx=3,
            product_mol_idx=1,
            product_atom_idx=5,
        )
        assert mapping.reactant_mol_idx == 0
        assert mapping.reactant_atom_idx == 3
        assert mapping.product_mol_idx == 1
        assert mapping.product_atom_idx == 5

    def test_frozen(self):
        mapping = AtomMapping(
            reactant_mol_idx=0,
            reactant_atom_idx=3,
            product_mol_idx=1,
            product_atom_idx=5,
        )
        with pytest.raises(ValidationError):
            mapping.reactant_atom_idx = 99  # type: ignore[misc]

    def test_repr(self):
        mapping = AtomMapping(
            reactant_mol_idx=0,
            reactant_atom_idx=1,
            product_mol_idx=2,
            product_atom_idx=3,
        )
        assert "AtomMapping" in repr(mapping)

    def test_hashable(self):
        mapping = AtomMapping(
            reactant_mol_idx=0,
            reactant_atom_idx=1,
            product_mol_idx=2,
            product_atom_idx=3,
        )
        assert hash(mapping) is not None


class TestBondChange:
    """Tests for the BondChange pydantic model."""

    def test_valid_construction(self):
        change = BondChange(
            atom1_map=1,
            atom2_map=2,
            change_type=BondChangeType.FORMED,
            new_order=1.0,
        )
        assert change.atom1_map == 1
        assert change.atom2_map == 2
        assert change.change_type == BondChangeType.FORMED
        assert change.old_order is None
        assert change.new_order == 1.0
        assert change.energy_cost == 0.0

    def test_frozen(self):
        change = BondChange(
            atom1_map=1,
            atom2_map=2,
            change_type=BondChangeType.BROKEN,
        )
        with pytest.raises(ValidationError):
            change.atom1_map = 99  # type: ignore[misc]

    def test_repr(self):
        change = BondChange(
            atom1_map=1,
            atom2_map=2,
            change_type=BondChangeType.FORMED,
        )
        assert "BondChange" in repr(change)


class TestMappingScore:
    """Tests for the MappingScore pydantic model."""

    def test_default_values(self):
        score = MappingScore()
        assert score.bond_energy_cost == 0.0
        assert score.num_bond_changes == 0
        assert score.similarity_score == 0.0

    def test_total_score_default_weights(self):
        score = MappingScore(
            bond_energy_cost=10.0,
            num_bond_changes=2,
            similarity_score=0.5,
        )
        result = score.total_score()
        assert result == 10.0 * 1.0 + 2 * 10.0 + 0.5 * (-50.0)

    def test_total_score_custom_weights(self):
        score = MappingScore(num_bond_changes=3)
        result = score.total_score(weights={"num_bond_changes": 100.0})
        assert result == 300.0

    def test_model_dump(self):
        score = MappingScore(bond_energy_cost=5.0, num_bonds_formed=2)
        dumped = score.model_dump()
        assert dumped["bond_energy_cost"] == 5.0
        assert dumped["num_bonds_formed"] == 2
        assert "total_score" not in dumped


class TestSupervisedConfig:
    """Tests for the SupervisedConfig pydantic model."""

    def test_default_values(self):
        config = SupervisedConfig()
        assert config.target_layer == 11
        assert config.bottleneck_size == 64

    def test_valid_construction(self):
        config = SupervisedConfig(
            target_layer=5,
        )
        assert config.target_layer == 5

    def test_negative_layer_raises(self):
        with pytest.raises(ValidationError):
            SupervisedConfig(target_layer=-1)

    def test_zero_bottleneck_size_raises(self):
        with pytest.raises(ValidationError):
            SupervisedConfig(bottleneck_size=0)

    def test_mutable(self):
        config = SupervisedConfig()
        config.target_layer = 3
        assert config.target_layer == 3
