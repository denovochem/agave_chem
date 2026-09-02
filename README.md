# AgaveChem
[![PyPI version](https://badge.fury.io/py/agave-chem.svg)](https://badge.fury.io/py/agave-chem)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://gitHub.com/denovochem/agave_chem/graphs/commit-activity)
[![License](https://img.shields.io/github/license/denovochem/agave_chem)](https://github.com/denovochem/agave_chem/blob/main/LICENSE)
[![Run Tests](https://img.shields.io/github/actions/workflow/status/denovochem/agave_chem/tests.yml?logo=github&logoColor=%23ffffff&label=tests)](https://github.com/denovochem/agave_chem/actions/workflows/tests.yml)
[![Build Docs](https://img.shields.io/github/actions/workflow/status/denovochem/agave_chem/docs.yml?logo=github&logoColor=%23ffffff&label=docs)](https://denovochem.github.io/agave_chem/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/denovochem/agave_chem/blob/main/examples/example_notebook.ipynb)

![AgaveChem](docs/images/agave_chem_image.png)

> **Try the web demo:** [denovochem.com/demos/reaction-atom-mapper](https://denovochem.com/demos/reaction-atom-mapper)

An open-source Python library for atom-to-atom mapping (AAM) of chemical reactions. The default interface `map_reactions` for extracting atom-mapped reaction SMILES achieves state-of-the-art accuracy on the 1,758 reaction [golden dataset benchmark](https://www.nature.com/articles/s41467-024-46364-y). AgaveChem can also be used to classify reactions and is capable of mapping and automatically balancing unbalanced reactions.

AgaveChem is comprised of four mappers:

- **Template mapper**: Reaction SMIRKS templates sourced from [ReactionFlash](https://apps.apple.com/us/app/reactionflash/id432080813), [Rxn-INSIGHT](https://github.com/mrodobbe/Rxn-INSIGHT), and manual curation are applied to classify and map reactions into a scheme inspired by [Carey et al.](https://pubs.rsc.org/ob/article-abstract/4/12/2337/234845/Analysis-of-the-reactions-used-for-the-preparation?redirectedFrom=fulltext), as well as [RXNO](https://www.ebi.ac.uk/ols4/ontologies/rxno).

- **MCS-like mapper**: Fingerprint based mapper that generates conservative partial maps some radius away from detected reaction centers.

- **Identical fragment mapper**: Fragments appearing structurally unchanged on both sides of the reaction (counter-ions, solvents, spectator reagents) are detected and atom-mapped before any other mapper is invoked.

- **Neural mapper**: An ALBERT model trained in two phases - unsupervised masked language model (MLM) pre-training followed by supervised fine-tuning with a direct attention alignment objective against generated "ground truth" maps from the other three mappers. The supervised training data for the second phase is generated automatically from ~0.97M filtered Lowe USPTO reactions; the other three mappers fully map ~63% of reactions and ~90% of all product atoms in this dataset.

These mappers can be used individually, or called as a pipeline using `map_reactions()`.




| Mapper | Per-reaction mapping accuracy |
| --- | :---: |
| [RXNMapper](https://www.science.org/doi/10.1126/sciadv.abe4166) | 87.09% |
| [RXNMapperv2](https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15005247/v1) | 89.59% |
| [GraphormerMapper](https://pubs.acs.org/jcisd8/article-abstract/62/14/3307/850123/Bidirectional-Graphormer-for-Reactivity?redirectedFrom=fulltext) | 89.76% |
| [LocalMapper](https://www.nature.com/articles/s41467-024-46364-y) | 89.59% |
| AgaveChem (neural only) | 91.87% |
| AgaveChem (using `map_reactions()`) | 92.72% |

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

### Mapping a batch of reactions through the full pipeline

```python
from agave_chem import map_reactions

reactions = [
    "CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl",
    "OCC(=O)OCCCO.Cl>>ClCC(=O)OCCCO",
]
results = map_reactions(reactions)
for r in results:
    print(r.final_mapping)
```

### Neural mapper

```python
from agave_chem import NeuralReactionMapper

mapper = NeuralReactionMapper("neural_mapper")
result = mapper.map_reaction("CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl")
print(result.selected_mapping)
```

### MCS-like mapper

```python
from agave_chem import MCSReactionMapper

mapper = MCSReactionMapper("mcs_mapper")
result = mapper.map_reaction("CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl")
print(result.selected_mapping)
```

### Template mapper

```python
from agave_chem import TemplateReactionMapper

mapper = TemplateReactionMapper("template_mapper")
result = mapper.map_reaction("CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl")
print(result.selected_mapping)
```

### Handling unbalanced reactions

The neural mapper is capable of mapping unbalanced reactions and returning balanced mapped reactions when one_to_one_correspondence is set to "auto" or False. When one_to_one_correspondence is set to "auto", the neural mapper uses heuristics to automatically determine for each reaction whether one_to_one_correspondence should be True or False.

```python
from rdkit import Chem
from rdkit.Chem import rdChemReactions

rxn = "c1c(O)cc(O)cc1O.O=[N+]([O-])O>>c(O)1c([N+](=O)[O-])c(O)c([N+](=O)[O-])c(O)c1[N+](=O)[O-]"
rdChemReactions.ReactionFromSmarts(rxn, useSmiles=True)
```

![Unbalanced reaction](docs/images/tnt_unbalanced_reaction.png)

```python
from agave_chem import NeuralReactionMapper

mapper = NeuralReactionMapper("neural_mapper")
result = mapper.map_reaction(rxn, one_to_one_correspondence=False)
rdChemReactions.ReactionFromSmarts(result.selected_mapping, useSmiles=True)
```

![Balanced mapped reaction](docs/images/tnt_balanced_reaction_mapped.png)


## Documentation

Documentation is a work in progress available [here](https://denovochem.github.io/agave_chem/).

## Contributing

- Feature ideas and bug reports are welcome on the [Issue Tracker](https://github.com/denovochem/agave_chem/issues).
- Fork the [source code](https://github.com/denovochem/agave_chem) on GitHub, make changes and file a pull request.

## License

AgaveChem is licensed under the [MIT license](https://github.com/denovochem/agave_chem/blob/main/LICENSE).

## References

- [RXNMapper: Schwaller et al., *Science Advances*, 2021](https://www.science.org/doi/10.1126/sciadv.abe4166)
- [RXNMapperv2: Grandjean et al., *ChemRxiv*, 2026](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15005247/v1)
- [LocalMapper: Chen et al., *Nat. Commun.*, 2024](https://www.nature.com/articles/s41467-024-46364-y)
- [GraphormerMapper: Nugmanov et al., *ChemRxiv*, 2022](https://doi.org/10.26434/chemrxiv-2022-bn5nt)
- [Rxn-INSIGHT: Probst et al.](https://github.com/mrodobbe/Rxn-INSIGHT)
- [rdchiral: Coley et al., *J. Chem. Inf. Model.*, 2019](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00286)
- [rdchiral_plus](https://github.com/denovochem/rdchiral_plus)
- [Lowe USPTO dataset](https://doi.org/10.17863/CAM.16293)
- [Benchmarking study: Lin et al., *ChemRxiv*, 2020](https://doi.org/10.26434/chemrxiv.13012679.v1)
- [ReactionFlash](https://apps.apple.com/us/app/reactionflash/id432080813)
