# Installation

AgaveChem supports Python versions >= 3.10.

## Requirements

- **Python** (version >= 3.10)
- **RDKit** — cheminformatics toolkit
- **rdchiral-plus** — improved template extraction and application library (installed automatically)
- **PyTorch** — required for the neural mapper; a CPU-only build is sufficient for inference
- **Transformers** (Hugging Face) — required for the neural mapper

All dependencies are declared in `requirements.txt` and are installed automatically by pip.

## Installation Option #1: Use pip (recommended)

Install AgaveChem with pip directly from the GitHub repo:

```bash
pip install git+https://github.com/denovochem/agave_chem.git
```

## Installation Option #2: Clone the repository

Install the latest version of AgaveChem from GitHub. The version on GitHub is not guaranteed to be stable but may include new features not available via other install options.

```bash
git clone https://github.com/denovochem/agave_chem.git
cd agave_chem
pip install .
```

## Verifying the installation

After installation, verify that AgaveChem is importable and the neural mapper loads correctly:

```python
from agave_chem import MCSReactionMapper

mapper = MCSReactionMapper("test")
result = mapper.map_reaction("CC(=O)O.OCC>>CC(=O)OCC")
print(result["selected_mapping"])
```

The neural mapper (`NeuralReactionMapper`) will download model weights on first use if they are not already present.
