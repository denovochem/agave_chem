#!/usr/bin/env bash
# Orchestrate the mapper comparison pipeline.
#
# Each script runs in its own venv because rxnmapper v2 and agave_chem
# have conflicting dependency requirements.  Edit the VENV paths below
# to match your setup.
#
# Usage:
#   bash run_all.sh [--limit N]
#
# Steps:
#   1. filter_reactions.py   — filter partially-mapped, strip mapping  (rdkit venv)
#   2. run_rxnmapper_v2.py    — map with RXNMapper v2                   (rxnmapper v2 venv)
#   3. run_agavechem.py       — map with AgaveChem neural mapper        (agave_chem venv)
#   4. combine_results.py     — extract templates, write CSV            (rdchiral_plus venv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Venv paths — EDIT THESE ──────────────────────────────────────────────
AGAVE_VENV="/home/csnbritt/projects/denovochem_projects/agave_chem/.venv"
RXNMAPPER_V2_VENV="/home/csnbritt/projects/denovochem_projects/agave_chem/.venv-rxnmapper-v2"
# The filter and combine scripts need rdkit + rdchiral_plus — the agave_chem
# venv has both, so we reuse it.
# ─────────────────────────────────────────────────────────────────────────

# Pass through --limit and other args
LIMIT_ARGS="$*"

echo "=== Step 1: Filter partially-mapped reactions ==="
"${AGAVE_VENV}/bin/python" "${SCRIPT_DIR}/filter_reactions.py" --require-partial --no-random ${LIMIT_ARGS}

echo ""
echo "=== Step 2: Run RXNMapper v2 ==="
"${RXNMAPPER_V2_VENV}/bin/python" "${SCRIPT_DIR}/run_rxnmapper_v2.py"

echo ""
echo "=== Step 3: Run AgaveChem neural mapper ==="
"${AGAVE_VENV}/bin/python" "${SCRIPT_DIR}/run_agavechem.py"

echo ""
echo "=== Step 4: Combine results and extract templates ==="
"${AGAVE_VENV}/bin/python" "${SCRIPT_DIR}/combine_results.py"

echo ""
echo "=== Done! CSV written to ${SCRIPT_DIR}/mapper_comparison.csv ==="
