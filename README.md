# AgaveChem
[![PyPI Version](https://img.shields.io/pypi/v/PubChemPy?logo=python&logoColor=%23ffffff)](https://pypi.python.org/pypi/PubChemPy)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://gitHub.com/denovochem/agave_chem/graphs/commit-activity)
[![License](https://img.shields.io/pypi/l/PubChemPy)](https://github.com/denovochem/agave_chem/blob/main/LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/denovochem/agave_chem/tests.yml?logo=github&logoColor=%23ffffff&label=tests)](https://github.com/denovochem/agave_chem/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/readthedocs/pubchempy?logo=readthedocs&logoColor=%23ffffff)](https://denovochem.github.io/agave_chem/)

This library is used for atom-mapping and reaction classification.

## Installation

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

## Documentation

Full documentation is availible [here](https://denovochem.github.io/agave_chem/)

## Contributing

- Feature ideas and bug reports are welcome on the Issue Tracker.
- Fork the [source code](https://github.com/denovochem/agave_chem) on GitHub, make changes and file a pull request.

## License

placeholder_name is licensed under the [MIT license](https://github.com/denovochem/agave_chem/blob/main/LICENSE).

## Reference

Initial reaction SMIRKS came from [Rxn-INSIGHT](https://github.com/mrodobbe/Rxn-INSIGHT)
