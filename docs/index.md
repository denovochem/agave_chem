# AgaveChem

An open-source Python library for atom-to-atom mapping (AAM) of chemical reactions. AgaveChem provides four composable mappers—from deterministic graph-based methods to a supervised neural mapper—that can be used individually or combined into a pipeline.

The primary contribution is a supervised ALBERT-based neural mapper trained without any per-reaction manual annotation. Ground truth atom maps are generated automatically by composing an expert template mapper and an MCS mapper over a filtered subset of the Lowe USPTO dataset (~0.97M reactions), yielding a labeled training corpus orders of magnitude larger than what direct annotation can provide.

## Requirements

- Python (version >= 3.10)
- RDKit
- [rdchiral-plus](https://github.com/denovochem/rdchiral_plus)
- PyTorch
- Transformers (Hugging Face)

## Installation

Install AgaveChem with pip directly from this repo:

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
