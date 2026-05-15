# AgaveChem
[![PyPI version](https://badge.fury.io/py/agave-chem.svg)](https://badge.fury.io/py/agave-chem)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://gitHub.com/denovochem/agave_chem/graphs/commit-activity)
[![License](https://img.shields.io/github/license/denovochem/agave_chem)](https://github.com/denovochem/agave_chem/blob/main/LICENSE)
[![Run Tests](https://img.shields.io/github/actions/workflow/status/denovochem/agave_chem/tests.yml?logo=github&logoColor=%23ffffff&label=tests)](https://github.com/denovochem/agave_chem/actions/workflows/tests.yml)
[![Build Docs](https://img.shields.io/github/actions/workflow/status/denovochem/agave_chem/docs.yml?logo=github&logoColor=%23ffffff&label=docs)](https://denovochem.github.io/agave_chem/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/denovochem/agave_chem/blob/main/examples/example_notebook.ipynb)

An open-source Python library for atom-to-atom mapping (AAM) of chemical reactions. AgaveChem provides four composable mappers—from deterministic graph-based methods to a supervised neural mapper—that can be used individually or combined into a pipeline.

| library | per-reaction mapping accuracy (%) | per-atom mapping accuracy (%) |
| --- | :---: | :---: |
| RXNMapper | XXX | XXX |
| GraphormerMapper | XXX | XXX |
| LocalMapper | XXX | XXX |
| AgaveChem | XXX | XXX |

## Neural mapper

- **Supervised ALBERT-based mapper**: Trained in two phases—unsupervised masked language model (MLM) pre-training followed by supervised fine-tuning with a direct attention alignment objective against generated ground truth maps
- **Template and MCS-derived supervision**: Ground truth is generated automatically from ~0.97M filtered USPTO reactions; the deterministic pipeline fully maps ~60% of reactions and covers ~90% of product atoms
- **State-of-the-art accuracy**: Outperforms existing neural mappers on held-out benchmark reactions

## Identical fragment mapper

- **Spectator molecule handling**: Fragments appearing structurally unchanged on both sides of the reaction (counter-ions, solvents, spectator reagents) are detected and atom-mapped before any other mapper is invoked
- **Collision-free numbering**: Pre-assigned atoms use a reserved numbering range to avoid conflicts with downstream mappers

## MCS mapper

- **Environment fingerprint matching**: Identifies invariant atoms using a bond-radius fingerprinting scheme rather than solving the full NP-hard MCS problem, enabling efficient partial mapping
- **Configurable radius**: A `min_radius_to_anchor_new_mapping` parameter controls how close to the reactive center mapping extends, yielding conservative partial maps that avoid incorrectly assigning atoms near bond-breaking events
- **Tautomer and charge normalization**: Molecules are normalized prior to fingerprinting to prevent spurious mismatches from charge variants or tautomers
- **Anchor-extend strategy**: Alternates between propagating mappings from already-assigned anchor atoms and seeding new anchors, ensuring consistent multi-fragment mapping

## Expert template mapper

- **Curated SMIRKS library**: Reaction SMIRKS templates sourced from ReactionFlash, [Rxn-INSIGHT](https://github.com/mrodobbe/Rxn-INSIGHT), and manual curation are applied to classify and map reactions
- **Hierarchical templates**: Templates are organized with parent-child priority relationships for fine-grained reaction class coverage
- **Custom template support**: User-supplied SMIRKS patterns can supplement or replace the built-in library via `custom_smirks_patterns`
- **MCS-guided focus**: Uses the MCS mapper internally to identify probable reaction centers, improving template matching efficiency

## Mapping scorer

- **Multi-metric evaluation**: Scores atom-mapped reactions across bond energy cost, bond change count, fragment changes, stereochemistry changes, and ring opening/closing events
- **Configurable weights**: Each scoring component carries an adjustable weight for custom ranking strategies

## Requirements

- Python (version >= 3.10)
- RDKit
- [rdchiral-plus](https://github.com/denovochem/rdchiral_plus)
- PyTorch
- Transformers (Hugging Face)

## Installation

Install AgaveChem from PyPi:

```bash
pip install agave_chem
```

Or install AgaveChem with pip directly from this repo:

```bash
pip install git+https://github.com/denovochem/agave_chem.git
```

Or clone and install locally:

```bash
git clone https://github.com/denovochem/agave_chem.git
cd agave_chem
pip install .
```

## Basic usage

### Neural mapper (recommended for general use)

```python
from agave_chem import NeuralReactionMapper

mapper = NeuralReactionMapper("my_mapper")
result = mapper.map_reaction("CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl")
print(result["selected_mapping"])
```

### MCS mapper (fast, deterministic, partial mapping)

```python
from agave_chem import MCSReactionMapper

mapper = MCSReactionMapper("my_mcs_mapper")
result = mapper.map_reaction("CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl")
print(result["selected_mapping"])
```

### Expert template mapper (interpretable, mechanistically grounded)

```python
from agave_chem import TemplateReactionMapper

mapper = TemplateReactionMapper("my_template_mapper")
result = mapper.map_reaction("CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl")
print(result["selected_mapping"])
```

### Mapping a batch of reactions through the full pipeline

```python
from agave_chem import map_reactions

reactions = [
    "CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl",
    "OCC(=O)OCCCO.Cl>>ClCC(=O)OCCCO",
]
results = map_reactions(reactions)
for r in results:
    print(r["final_mapping"])
```

### Scoring a mapping

```python
from agave_chem import MappingScorer

scorer = MappingScorer()
score = scorer.score_mapping("[CH3:1][C:2](=[O:3])[Cl:4]>>[CH3:1][C:2](=[O:3])[OH:5]")
print(score)
```

## Documentation

Full documentation is available at the [AgaveChem documentation site](https://denovochem.github.io/agave_chem/).

## Contributing

- Feature ideas and bug reports are welcome on the [Issue Tracker](https://github.com/denovochem/agave_chem/issues).
- Fork the [source code](https://github.com/denovochem/agave_chem) on GitHub, make changes and file a pull request.

## License

AgaveChem is licensed under the [MIT license](https://github.com/denovochem/agave_chem/blob/main/LICENSE).

## References

- [RXNMapper: Schwaller et al., *Science Advances*, 2021](https://www.science.org/doi/10.1126/sciadv.abe4166)
- [LocalMapper: Chen et al., *Nat. Commun.*, 2024](https://www.nature.com/articles/s41467-024-46364-y)
- [GraphormerMapper: Nugmanov et al., *ChemRxiv*, 2022](https://doi.org/10.26434/chemrxiv-2022-bn5nt)
- [Rxn-INSIGHT: Probst et al.](https://github.com/mrodobbe/Rxn-INSIGHT)
- [rdchiral: Coley et al., *J. Chem. Inf. Model.*, 2019](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00286)
- [rdchiral_plus](https://github.com/denovochem/rdchiral_plus)
- [Lowe USPTO dataset](https://doi.org/10.17863/CAM.16293)
- [Benchmarking study: Lin et al., *ChemRxiv*, 2020](https://doi.org/10.26434/chemrxiv.13012679.v1)
