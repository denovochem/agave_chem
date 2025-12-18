# AgaveChem
[![PyPI Version](https://img.shields.io/pypi/v/PubChemPy?logo=python&logoColor=%23ffffff)](https://pypi.python.org/pypi/PubChemPy)
[![License](https://img.shields.io/pypi/l/PubChemPy)](https://github.com/denovochem/agave_chem/blob/main/LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/denovochem/agave_chem/tests.yml?logo=github&logoColor=%23ffffff&label=tests)](https://github.com/denovochem/agave_chem/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/readthedocs/pubchempy?logo=readthedocs&logoColor=%23ffffff)](https://denovochem.github.io/agave_chem/)

Atom-mapping and reaction classification.

## Install

Install directly from this github repo:

```console
pip install git+https://github.com/denovochem/agave_chem.git
```

## Usage

### Load AgaveChemMapper and atom-map a reaction:
```python
from agave_chem import AgaveChemMapper

mapper = AgaveChemMapper()
mapped_reaction = mapper.map_reaction("CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl")

print(mapped_reaction)
```


## Documentation

Additional documentation coming soon.

## Reference

Initial reaction SMIRKS came from https://github.com/mrodobbe/Rxn-INSIGHT
