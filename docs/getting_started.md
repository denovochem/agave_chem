# Getting Started

This guide covers all four AgaveChem mappers in depth, the `map_reactions` pipeline function, working with mapper results, scoring mappings, and advanced configuration options.

---

## Overview of the mapping pipeline

AgaveChem provides four composable mappers that can be used independently or combined into a pipeline. In the default pipeline, mappers are applied in order of increasing complexity:

1. **Identical fragment mapper** — maps spectator molecules and structurally unchanged fragments before any other mapper is invoked
2. **MCS mapper** — assigns atom-map numbers to atoms whose local chemical environment is preserved across the reaction, yielding conservative partial maps
3. **Expert template mapper** — applies a curated library of reaction SMIRKS templates to classify and fully map known reaction classes
4. **Neural mapper** — an ALBERT-based model trained on labeled USPTO reactions, used for complete mapping at inference time

Each mapper operates on unmapped reaction SMILES and returns a `ReactionMapperResult` dictionary. A higher-priority mapper in the pipeline overrides the output of a lower-priority one.

---

## The `map_reactions` function

The easiest way to map a batch of reactions is the top-level `map_reactions` function, which runs the MCS and template mappers by default:

```python
from agave_chem import map_reactions

reactions = [
    "CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl",
    "OCC(=O)OCCCO.Cl>>ClCC(=O)OCCCO",
]
results = map_reactions(reactions)

for r in results:
    print(r["original_reaction"])
    print(r["final_mapping"])
    print()
```

### Parameters

- `reaction_list` (`List[str]`) — list of unmapped reaction SMILES strings; a single string is also accepted
- `mappers_list` (`List[ReactionMapper]`) — list of mapper instances to apply in order; defaults to `[MCSReactionMapper("mcs_default"), TemplateReactionMapper("expert_default")]`
- `batch_size` (`int`) — number of reactions per processing batch (default: `500`, max: `1000`)

### Using the neural mapper in the pipeline

To use the neural mapper, pass it explicitly in `mappers_list`. Place it last so it acts as the highest-priority mapper:

```python
from agave_chem import map_reactions, MCSReactionMapper, NeuralReactionMapper

results = map_reactions(
    reactions,
    mappers_list=[
        MCSReactionMapper("mcs"),
        NeuralReactionMapper("neural"),
    ],
)
```

---

## Working with mapper results

Every mapper returns a `ReactionMapperResult` dictionary with the following keys:

| Key | Type | Description |
| --- | --- | --- |
| `original_smiles` | `str` | The unmapped input reaction SMILES |
| `selected_mapping` | `str` | The chosen mapped reaction SMILES (empty string if mapping failed) |
| `possible_mappings` | `Dict[str, List]` | All candidate mappings produced before selection |
| `mapping_type` | `str` | Which mapper produced the result (`"mcs"`, `"template"`, `"neural"`, etc.) |
| `mapping_score` | `Any` | Optional score attached by the mapper (may be `None`) |
| `additional_info` | `List[Dict]` | Mapper-specific metadata |

The `map_reactions` function returns `AgaveChemMapperResult` dictionaries with an additional `final_mapping` key (the best mapping across all mappers) and `mapper_results` (the per-mapper results list).

```python
result = results[0]
print(result["final_mapping"])        # best mapping from the pipeline
print(result["original_reaction"])    # original unmapped SMILES
print(result["mapper_results"][0]["mapping_type"])  # which mapper produced it
```

---

## Identical fragment mapper

The identical fragment mapper handles fragments that appear structurally unchanged on both sides of the reaction. It is invoked automatically within `map_reactions`, but can also be used standalone:

```python
from agave_chem import IdenticalFragmentMapper

mapper = IdenticalFragmentMapper("my_ifm")
result = mapper.map_reaction("CC(=O)O.[Na+].[Cl-]>>CC(=O)[O-].[Na+].[Cl-]")
print(result["selected_mapping"])
```

Identical fragments are assigned map numbers from a reserved range (starting at 500) to avoid collisions with downstream mappers.

---

## MCS mapper

The MCS mapper assigns atom-map numbers to atoms whose local chemical environment is preserved across the reaction. It uses a bond-radius environment fingerprinting scheme and is efficient even for large, multi-fragment reactions.

```python
from agave_chem import MCSReactionMapper

mapper = MCSReactionMapper("my_mcs")
result = mapper.map_reaction("CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl")
print(result["selected_mapping"])
```

### Key parameters for `map_reaction`

- `min_radius` (`int`) — smallest bond-radius to consider (default: `1`)
- `min_radius_to_anchor_new_mapping` (`int`) — below this radius, environments are only matched when they already contain at least one mapped atom; controls how close to the reactive center new anchor atoms can be seeded (default: `3`)
- `max_radius` (`Optional[int]`) — largest bond-radius to search; defaults to the size of the largest molecule

### Partial mapping

The MCS mapper produces partial maps for reactions where the reactive center cannot be unambiguously resolved. The `selected_mapping` will contain atom-map numbers only for atoms whose environment was confidently matched:

```python
result = mapper.map_reaction("c1ccccc1Br.B(O)(O)c1ccccc1>>c1ccc(-c2ccccc2)cc1")
# Atoms in the biaryl core will be mapped; the coupling site atoms may not be
print(result["selected_mapping"])
```

---

## Expert template mapper

The expert template mapper applies a curated library of reaction SMIRKS templates to classify and fully map reactions that match known reaction classes.

```python
from agave_chem import TemplateReactionMapper

mapper = TemplateReactionMapper("my_template")
result = mapper.map_reaction("CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl")
print(result["selected_mapping"])
print(result["possible_mappings"])   # all candidate mapped SMILES and their templates
```

### Constructor parameters

- `mapper_name` (`str`) — unique name for this mapper instance
- `mapper_weight` (`float`) — priority weight in a multi-mapper pipeline (default: `3`)
- `custom_smirks_patterns` (`List[Dict] | None`) — list of user-supplied SMIRKS patterns; each dict must have `"name"`, `"smirks"`, and `"superclass_id"` keys
- `use_default_smirks_patterns` (`bool`) — whether to include the built-in template library (default: `True`)
- `max_transforms` (`int`) — maximum number of tautomer transforms (default: `1000`)
- `max_tautomers` (`int`) — maximum number of tautomers to enumerate (default: `1000`)
- `use_mcs_mapping` (`bool`) — whether to use MCS internally to focus template matching on the probable reaction center (default: `True`)

### Using custom SMIRKS patterns

You can extend or replace the built-in template library with your own reaction SMIRKS:

```python
from agave_chem import TemplateReactionMapper

custom_patterns = [
    {
        "name": "My custom acylation",
        "smirks": "[C:1](=[O:2])[Cl:3].[N:4]>>[C:1](=[O:2])[N:4]",
        "superclass_id": None,
    }
]

mapper = TemplateReactionMapper(
    "custom_template",
    custom_smirks_patterns=custom_patterns,
    use_default_smirks_patterns=True,   # combine with built-in patterns
)
result = mapper.map_reaction("CC(=O)Cl.NC>>CC(=O)NC")
print(result["selected_mapping"])
```

---

## Neural mapper

### Overview

The neural mapper uses a supervised ALBERT-based model trained on labeled USPTO reactions. It is the recommended mapper for general-purpose atom mapping.

```python
from agave_chem import NeuralReactionMapper

mapper = NeuralReactionMapper("my_neural")
result = mapper.map_reaction("CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl")
print(result["selected_mapping"])
```

### Neural mapper parameters

- `mapper_name` (`str`) — unique name for this mapper instance
- `mapper_weight` (`float`) — priority weight in a multi-mapper pipeline (default: `3`)
- `checkpoint_path` (`Optional[str]`) — path to a custom model checkpoint directory; defaults to the bundled pre-trained model
- `use_supervised` (`bool`) — whether to use the supervised (fine-tuned) model head (default: `True`)
- `sequence_max_length` (`int`) — maximum token sequence length (default: `512`)

### Mapping a batch

```python
from agave_chem import NeuralReactionMapper

mapper = NeuralReactionMapper("neural_batch")
results = mapper.map_reactions([
    "CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl",
    "OCC(=O)OCCCO.Cl>>ClCC(=O)OCCCO",
])
for r in results:
    print(r["selected_mapping"])
```

---

## Mapping scorer

`MappingScorer` evaluates the quality of an atom-mapped reaction SMILES using a set of chemically motivated metrics.

```python
from agave_chem import MappingScorer

scorer = MappingScorer()
score = scorer.score_mapping(
    "[CH3:1][C:2](=[O:3])[OH:4].[HO:5][CH2:6][CH3:7]>>[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH3:7]"
)
print(score)
```

### Scoring metrics

| Metric | Description |
| --- | --- |
| `bond_energy_cost` | Summed bond dissociation energy cost of all bond changes |
| `num_bond_changes` | Total number of bonds formed or broken |
| `num_fragments` | Number of disconnected fragments affected |
| `stereo_changes` | Number of stereocenters whose configuration changes |
| `ring_changes` | Number of ring opening or closing events |

### Custom weights

Each metric weight can be adjusted at construction time:

```python
scorer = MappingScorer(
    energy_penalty_weight=1.0,
    bond_change_weight=10.0,
    fragment_weight=20.0,
    stereo_weight=15.0,
    ring_weight=25.0,
)
```

---

## Composing mappers manually

Mappers can be composed into a custom pipeline by passing a `mappers_list` to `map_reactions`. The list is applied in order; the last mapper whose result is non-empty is used as the `final_mapping`:

```python
from agave_chem import (
    map_reactions,
    MCSReactionMapper,
    TemplateReactionMapper,
    NeuralReactionMapper,
)

results = map_reactions(
    reactions,
    mappers_list=[
        MCSReactionMapper("mcs", mapper_weight=1),
        TemplateReactionMapper("template", mapper_weight=2),
        NeuralReactionMapper("neural", mapper_weight=3),
    ],
    batch_size=200,
)
```

Note that mapper names within a pipeline must be unique.

---

## Sanitizing and validating mapped reactions

All mapper classes expose shared utility methods via the `ReactionMapper` base class:

```python
from agave_chem import MCSReactionMapper

mapper = MCSReactionMapper("util")

# Sanitize a mapped reaction SMILES (validates, optionally canonicalizes)
clean = mapper.sanitize_rxn_string(
    "[CH3:1][C:2](=[O:3])[OH:4].[HO:5][CH2:6]>>[CH3:1][C:2](=[O:3])[O:5][CH2:6]",
    expect_full_mapping=True,
    canonicalize=True,
    remove_mapping=False,
)
print(clean)

# Sanitize a molecule
from rdkit import Chem
mol = Chem.MolFromSmiles("CC(=O)O")
clean_mol = mapper.sanitize_molecule(mol, add_hs=False)
```
