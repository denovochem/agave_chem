#!/usr/bin/env bash
# Orchestrate the speed benchmark pipeline.
#
# Runs each atom-mapping tool on 10k randomly-sampled reactions (mapping
# stripped) and records per-reaction timing.  Each script runs in its own
# venv because the tools have conflicting dependency requirements.
# Edit the VENV paths below to match your setup.
#
# Usage:
#   bash run_speed_benchmark.sh [--limit N]
#
# Steps:
#   1. filter_reactions.py    — random sample, strip mapping     (agave_chem venv)
#   2. run_rxnmapper.py       — RXNMapper v1, bs=1 and bs=32      (rxnmapper venv)
#   3. run_rxnmapper_v2.py    — RXNMapper v2, bs=1 and bs=32      (rxnmapper v2 venv)
#   4. run_chython.py         — GraphormerMapper, bs=1            (chython venv)
#   5. run_localmapper.py     — LocalMapper, bs=1                 (localmapper conda)
#   6. run_agavechem.py       — AgaveChem neural + pipeline       (agave_chem venv)
#   7. compile_speed_table.py — aggregate timing → CSV + LaTeX    (agave_chem venv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Venv paths — EDIT THESE ──────────────────────────────────────────────
AGAVE_VENV="/home/csnbritt/projects/denovochem_projects/agave_chem/.venv"
RXNMAPPER_VENV="/home/csnbritt/projects/denovochem_projects/agave_chem/.venv-rxnmapper"
RXNMAPPER_V2_VENV="/home/csnbritt/projects/denovochem_projects/agave_chem/.venv-rxnmapper-v2"
CHYTHON_VENV="/home/csnbritt/projects/denovochem_projects/agave_chem/.venv-chython"
LOCALMAPPER_CONDA="/home/csnbritt/projects/denovochem_projects/agave_chem/.conda-localmapper"
# ─────────────────────────────────────────────────────────────────────────

# Pass through --limit and other args to filter_reactions.py
LIMIT_ARGS="$*"

echo "=== Step 1: Filter and strip reactions (random sample) ==="
"${AGAVE_VENV}/bin/python" "${SCRIPT_DIR}/filter_reactions.py" ${LIMIT_ARGS}

echo ""
echo "=== Step 2: Run RXNMapper v1 (batch size 1) ==="
"${RXNMAPPER_VENV}/bin/python" "${SCRIPT_DIR}/run_rxnmapper.py" \
    --batch-size 1 \
    --timing-output "${SCRIPT_DIR}/speed_rxnmapper_bs1.json"

echo ""
echo "=== Step 2b: Run RXNMapper v1 (batch size 32) ==="
"${RXNMAPPER_VENV}/bin/python" "${SCRIPT_DIR}/run_rxnmapper.py" \
    --batch-size 32 \
    --timing-output "${SCRIPT_DIR}/speed_rxnmapper_bs32.json"

echo ""
echo "=== Step 3: Run RXNMapper v2 (batch size 1) ==="
"${RXNMAPPER_V2_VENV}/bin/python" "${SCRIPT_DIR}/run_rxnmapper_v2.py" \
    --batch-size 1 \
    --timing-output "${SCRIPT_DIR}/speed_rxnmapper_v2_bs1.json"

echo ""
echo "=== Step 3b: Run RXNMapper v2 (batch size 32) ==="
"${RXNMAPPER_V2_VENV}/bin/python" "${SCRIPT_DIR}/run_rxnmapper_v2.py" \
    --batch-size 32 \
    --timing-output "${SCRIPT_DIR}/speed_rxnmapper_v2_bs32.json"

echo ""
echo "=== Step 4: Run GraphormerMapper (chython, batch size 1) ==="
"${CHYTHON_VENV}/bin/python" "${SCRIPT_DIR}/run_chython.py" \
    --timing-output "${SCRIPT_DIR}/speed_graphormer_bs1.json"

echo ""
echo "=== Step 5: Run LocalMapper (batch size 1) ==="
"${LOCALMAPPER_CONDA}/bin/python" "${SCRIPT_DIR}/run_localmapper.py" \
    --batch-size 1 \
    --timing-output "${SCRIPT_DIR}/speed_localmapper_bs1.json"

echo ""
echo "=== Step 6: Run AgaveChem neural (batch size 1) ==="
"${AGAVE_VENV}/bin/python" "${SCRIPT_DIR}/run_agavechem.py" \
    --mode neural --batch-size 1 \
    --timing-output "${SCRIPT_DIR}/speed_agavechem_neural_bs1.json"

echo ""
echo "=== Step 6b: Run AgaveChem neural (batch size 32) ==="
"${AGAVE_VENV}/bin/python" "${SCRIPT_DIR}/run_agavechem.py" \
    --mode neural --batch-size 32 \
    --timing-output "${SCRIPT_DIR}/speed_agavechem_neural_bs32.json"

echo ""
echo "=== Step 6c: Run AgaveChem pipeline (batch size 32) ==="
"${AGAVE_VENV}/bin/python" "${SCRIPT_DIR}/run_agavechem.py" \
    --mode pipeline --batch-size 32 \
    --timing-output "${SCRIPT_DIR}/speed_agavechem_pipeline_bs32.json"

echo ""
echo "=== Step 7: Compile speed table ==="
"${AGAVE_VENV}/bin/python" "${SCRIPT_DIR}/compile_speed_table.py" --latex

echo ""
echo "=== Done! CSV written to ${SCRIPT_DIR}/speed_results.csv ==="
