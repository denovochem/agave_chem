Install AgaveChem with pip directly from this repo:

```console
pip install git+https://github.com/denovochem/agave_chem.git
```

## Basic Usage
Load AgaveChemMapper and atom-map a reaction:
```pycon
from agave_chem import AgaveChemMapper

mapper = AgaveChemMapper()
mapped_reaction = mapper.map_reaction("CC(Cl)(Cl)OC(C)(Cl)Cl.CC(=O)C(=O)O>>CC(=O)C(=O)Cl")

print(mapped_reaction)
```