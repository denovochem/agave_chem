# AgaveChem

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
