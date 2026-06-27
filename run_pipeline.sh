#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# NSMoR — Master Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════
#
# Deterministic, end-to-end execution of the full NSMoR
# experimental DAG: ETL → Train → Analysis → Figures.
#
# Usage:
#   bash run_pipeline.sh
#   EPOCHS=200 DT_MS=8 bash run_pipeline.sh
#
# Environment overrides (with defaults):
#   RAW_DIR         — path to raw CSV data           (data/raw)
#   OUTPUT_DIR      — root output directory           (results)
#   RUN_DIR         — checkpoint / run directory      (runs/default)
#   EPOCHS          — training epochs                 (100)
#   BATCH_SIZE      — training batch size             (32)
#   LR              — learning rate                   (0.001)
#   DT_MS           — frame interval in ms            (10)
#   CONFIG          — YAML config file                (config/default.yaml)
#   PYTHON          — Python interpreter               (python)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colour codes ─────────────────────────────────────────────
BOLD="\033[1m"
GREEN="\033[32m"
CYAN="\033[36m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

stage_header() {
    echo ""
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}${GREEN}  $1${RESET}"
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}"
    echo ""
}

stage_done() {
    echo -e "\n${GREEN}✔ $1 complete.${RESET}\n"
}

stage_fail() {
    echo -e "\n${RED}✘ $1 FAILED (exit code $?).${RESET}\n" >&2
    exit 1
}

# ── Configurable variables ───────────────────────────────────
RAW_DIR="${RAW_DIR:-data/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
RUN_DIR="${RUN_DIR:-runs/default}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
DT_MS="${DT_MS:-10}"
CONFIG="${CONFIG:-config/default.yaml}"
PYTHON="${PYTHON:-python}"

BEST_MODEL="${RUN_DIR}/best_model.pth"

# ── Preflight checks ────────────────────────────────────────
echo -e "${BOLD}NSMoR Pipeline Orchestrator${RESET}"
echo -e "  Python  : ${PYTHON}"
echo -e "  Config  : ${CONFIG}"
echo -e "  Run Dir : ${RUN_DIR}"
echo -e "  Epochs  : ${EPOCHS}"
echo -e "  Output  : ${OUTPUT_DIR}"
echo ""

if ! command -v "${PYTHON}" &>/dev/null; then
    echo -e "${RED}ERROR: Python interpreter '${PYTHON}' not found.${RESET}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${RUN_DIR}"

# ═════════════════════════════════════════════════════════════
# Phase A — ETL: Data Preparation
# ═════════════════════════════════════════════════════════════
stage_header "Phase A — ETL: Data Preparation"

"${PYTHON}" scripts/prepare_data.py \
    --raw_dir "${RAW_DIR}" \
    --output_dir data/processed \
    --dt_ms "${DT_MS}" \
    || stage_fail "Phase A"

stage_done "Phase A"

# ═════════════════════════════════════════════════════════════
# Phase B — Training
# ═════════════════════════════════════════════════════════════
stage_header "Phase B — Training Engine"

"${PYTHON}" scripts/train.py \
    --config "${CONFIG}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --output_dir "${RUN_DIR}" \
    || stage_fail "Phase B"

stage_done "Phase B"

# ═════════════════════════════════════════════════════════════
# Phase C — Dynamics & Manifold Analysis
# ═════════════════════════════════════════════════════════════
stage_header "Phase C — Dynamics & Manifold Analysis"

"${PYTHON}" scripts/analyze_dynamics.py \
    --checkpoint "${BEST_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    || stage_fail "Phase C"

stage_done "Phase C"

# ═════════════════════════════════════════════════════════════
# Phase D — In-Silico Lesion (Virtual Ablation)
# ═════════════════════════════════════════════════════════════
stage_header "Phase D — In-Silico Lesion Analysis"

"${PYTHON}" scripts/simulate_lesion.py \
    --checkpoint "${BEST_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    || stage_fail "Phase D"

stage_done "Phase D"

# ═════════════════════════════════════════════════════════════
# Phase E — Jacobian Eigenvalue Spectrum
# ═════════════════════════════════════════════════════════════
stage_header "Phase E — Jacobian Eigenvalue Spectrum"

"${PYTHON}" scripts/analyze_jacobian.py \
    --checkpoint "${BEST_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    || stage_fail "Phase E"

stage_done "Phase E"

# ═════════════════════════════════════════════════════════════
# Phase F — Multisensory Integration Window
# ═════════════════════════════════════════════════════════════
stage_header "Phase F — Multisensory Integration Window"

"${PYTHON}" scripts/analyze_integration.py \
    --checkpoint "${BEST_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    || stage_fail "Phase F"

stage_done "Phase F"

# ═════════════════════════════════════════════════════════════
# Phase G — Psychophysics & Bayesian Reliability
# ═════════════════════════════════════════════════════════════
stage_header "Phase G — Psychophysics & Bayesian Reliability"

"${PYTHON}" scripts/simulate_psychophysics.py \
    --checkpoint "${BEST_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    || stage_fail "Phase G"

stage_done "Phase G"

# ═════════════════════════════════════════════════════════════
# Phase H — Autoregressive Closed-Loop Generation
# ═════════════════════════════════════════════════════════════
stage_header "Phase H — Autoregressive Closed-Loop Generation"

"${PYTHON}" scripts/simulate_autoregressive.py \
    --checkpoint "${BEST_MODEL}" \
    --output_dir "${OUTPUT_DIR}/sim_session" \
    --dt_ms "${DT_MS}" \
    || stage_fail "Phase H"

stage_done "Phase H"

# ═════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  ✔ Pipeline complete. All outputs in ${OUTPUT_DIR}/${RESET}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${RESET}"
echo ""
echo "  Figures:"
echo "    ${OUTPUT_DIR}/dynamics_manifold.png"
echo "    ${OUTPUT_DIR}/lesion_comparison.png"
echo "    ${OUTPUT_DIR}/jacobian_spectrum.png"
echo "    ${OUTPUT_DIR}/integration_window.png"
echo "    ${OUTPUT_DIR}/bayesian_reliability.png"
echo ""
echo "  Data:"
echo "    ${OUTPUT_DIR}/lesion_statistics.csv"
echo "    ${OUTPUT_DIR}/jacobian_stats.csv"
echo "    ${OUTPUT_DIR}/integration_summary.json"
echo "    ${OUTPUT_DIR}/psychophysics_summary.json"
echo ""
echo "  Synthetic Session:"
echo "    ${OUTPUT_DIR}/sim_session/events.csv"
echo "    ${OUTPUT_DIR}/sim_session/kinematics.csv"
echo ""
