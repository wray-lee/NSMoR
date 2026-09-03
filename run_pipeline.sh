#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# NSMoR — Master Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════
#
# Deterministic, end-to-end execution of the full NSMoR
# experimental DAG: ETL → Train → 6 Analyses → Closed-Loop Sim.
#
# Usage:
#   bash run_pipeline.sh
#   EPOCHS=300 PHASE1_EPOCHS=150 bash run_pipeline.sh
#   DRY_RUN=1 bash run_pipeline.sh          # print the plan, run nothing
#
# Environment overrides.  An EMPTY training override means "let CONFIG
# decide" — the YAML is the single source of truth for hyperparameters,
# so the runner can no longer silently contradict config/default.yaml.
#
#   RAW_DIR        raw CSV root                  (data/raw)
#   DATASET        processed dataset path        (data/processed/nsmor_dataset.pt)
#   OUTPUT_DIR     analysis output root          (results)
#   RUN_DIR        checkpoint / run directory    (runs/default)
#   CONFIG         YAML config                   (config/default.yaml)
#   SEED           ETL + psychophysics seed      (42)
#   DT_MS          frame interval in ms          (CONFIG model.dt_ms)
#   EPOCHS         TOTAL training epochs         (CONFIG training.num_epochs)
#   PHASE1_EPOCHS  Hybrid-Funnel phase-1 epochs  (unset = single-phase)
#   BATCH_SIZE     training batch size           (CONFIG)
#   LR             learning rate                 (CONFIG)
#   PYTHON         interpreter                   (python)
#   DRY_RUN        1 = print planned commands    (0)
#
# Every stage is gated twice: the process must exit 0 AND its declared
# artefacts must exist and be non-empty.  A stage that "succeeds" without
# producing its figure/CSV/JSON fails the pipeline.
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# Run from the repository root regardless of the caller's cwd; every
# path below is repo-relative.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Only 2 of the 11 entry scripts bootstrap sys.path themselves, so the
# remaining 9 need `nsmor` importable.  Exporting the repo root makes the
# pipeline runnable from a bare clone without `pip install -e .`, while
# leaving an existing installation to take precedence is unnecessary —
# the checkout is the code under test.
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Colour codes (disabled when stdout is not a terminal so that
#    logs and the CLI contract test parse cleanly) ──────────────
if [[ -t 1 ]]; then
    BOLD="\033[1m"; GREEN="\033[32m"; CYAN="\033[36m"
    YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
else
    BOLD=""; GREEN=""; CYAN=""; YELLOW=""; RED=""; RESET=""
fi

# ── Configuration ────────────────────────────────────────────
RAW_DIR="${RAW_DIR:-data/raw}"
DATASET="${DATASET:-data/processed/nsmor_dataset.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
RUN_DIR="${RUN_DIR:-runs/default}"
CONFIG="${CONFIG:-config/default.yaml}"
SEED="${SEED:-42}"
DT_MS="${DT_MS:-}"
EPOCHS="${EPOCHS:-}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"

BEST_MODEL="${RUN_DIR}/best_model.pth"
SIM_DIR="${OUTPUT_DIR}/sim_session"
MANIFEST="${OUTPUT_DIR}/pipeline_manifest.json"
COMMAND_LOG="${OUTPUT_DIR}/pipeline_commands.log"

# ── Helpers ──────────────────────────────────────────────────
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

abort() {
    echo -e "${RED}✘ $1${RESET}" >&2
    exit 1
}

# Run one pipeline stage.  In DRY_RUN mode the resolved argv is printed
# on a single "PLAN " line (parsed by tests/test_cli_contract.py) and
# nothing is executed.
run_py() {
    local label="$1"; shift
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'PLAN %s' "${PYTHON}"
        printf ' %s' "$@"
        printf '\n'
        return 0
    fi
    printf '%s' "${PYTHON}" >>"${COMMAND_LOG}"
    printf ' %q' "$@" >>"${COMMAND_LOG}"
    printf '\n' >>"${COMMAND_LOG}"
    local rc=0
    "${PYTHON}" "$@" || rc=$?
    if (( rc != 0 )); then
        abort "${label} FAILED (exit code ${rc})."
    fi
}

# Fail the stage when a declared artefact is missing or empty.
require_outputs() {
    local label="$1"; shift
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    local missing=0
    local f
    for f in "$@"; do
        if [[ ! -f "${f}" || ! -s "${f}" ]]; then
            echo -e "${RED}  ✘ missing or empty artefact: ${f}${RESET}" >&2
            missing=1
        fi
    done
    if (( missing != 0 )); then
        abort "${label}: declared artefacts were not produced."
    fi
}

# model.dt_ms from the YAML config (empty when unavailable).
read_config_dt_ms() {
    "${PYTHON}" - "${CONFIG}" 2>/dev/null <<'PY'
import sys
try:
    import yaml
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    dt = (cfg.get("model") or {}).get("dt_ms", None)
    if dt is not None:
        print(float(dt))
except Exception:
    pass
PY
}

# ── Preflight ────────────────────────────────────────────────
command -v "${PYTHON}" >/dev/null 2>&1 \
    || abort "Python interpreter '${PYTHON}' not found."
[[ -f "${CONFIG}" ]] || abort "Config file not found: ${CONFIG}"

STAGE_SCRIPTS=(
    scripts/prepare_data.py
    scripts/train.py
    scripts/analyze_dynamics.py
    scripts/simulate_lesion.py
    scripts/analyze_jacobian.py
    scripts/analyze_integration.py
    scripts/simulate_psychophysics.py
    scripts/analyze_gating.py
    scripts/simulate_autoregressive.py
)
for s in "${STAGE_SCRIPTS[@]}"; do
    [[ -f "${s}" ]] || abort "Missing pipeline script: ${s}"
done

# Fail here rather than half-way through Phase A: without an importable
# `nsmor` every stage dies on ModuleNotFoundError.
"${PYTHON}" -c 'import nsmor' >/dev/null 2>&1 \
    || abort "Cannot import 'nsmor' with PYTHONPATH=${PYTHONPATH}. Run 'make install' or check the interpreter."

if [[ "${DRY_RUN}" != "1" ]]; then
    [[ -d "${RAW_DIR}" ]] || abort "Raw data directory not found: ${RAW_DIR}"
fi

# The analyses convert frame indices to physical time with DT_MS; if it
# disagrees with the dt_ms the model was discretised at, every latency
# and time constant in the figures is wrong.  Derive it from CONFIG, and
# refuse an explicit override that contradicts the model.
CONFIG_DT_MS="$(read_config_dt_ms)" || CONFIG_DT_MS=""
if [[ -z "${DT_MS}" ]]; then
    DT_MS="${CONFIG_DT_MS:-10.0}"
elif [[ -n "${CONFIG_DT_MS}" ]]; then
    if ! awk -v a="${DT_MS}" -v b="${CONFIG_DT_MS}" \
        'BEGIN { exit (a + 0 == b + 0) ? 0 : 1 }'; then
        abort "DT_MS=${DT_MS} contradicts ${CONFIG} model.dt_ms=${CONFIG_DT_MS}; physical time in every analysis would be inconsistent with the trained model."
    fi
fi

# ── Training argv (empty override = keep the YAML value) ──────
TRAIN_ARGS=(
    scripts/train.py
    --config "${CONFIG}"
    --dataset "${DATASET}"
    --output_dir "${RUN_DIR}"
)
if [[ -n "${EPOCHS}" ]]; then
    TRAIN_ARGS+=(--epochs "${EPOCHS}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
    TRAIN_ARGS+=(--batch_size "${BATCH_SIZE}")
fi
if [[ -n "${LR}" ]]; then
    TRAIN_ARGS+=(--lr "${LR}")
fi
if [[ -n "${PHASE1_EPOCHS}" ]]; then
    # train.py treats --epochs as the TOTAL budget and derives
    # phase2 = epochs - phase1_epochs, so a phase-1 request without an
    # explicit total silently yields a zero-epoch phase 2.
    [[ -n "${EPOCHS}" ]] || abort "PHASE1_EPOCHS=${PHASE1_EPOCHS} requires EPOCHS (total epochs) to be set as well."
    if (( PHASE1_EPOCHS >= EPOCHS )); then
        abort "PHASE1_EPOCHS=${PHASE1_EPOCHS} must be < EPOCHS=${EPOCHS}; phase 2 (BioDecisionCore) would get 0 epochs."
    fi
    TRAIN_ARGS+=(--phase1_epochs "${PHASE1_EPOCHS}")
fi

# ── Banner ───────────────────────────────────────────────────
echo -e "${BOLD}NSMoR Pipeline Orchestrator${RESET}"
echo "  Python   : ${PYTHON}"
echo "  Config   : ${CONFIG}"
echo "  Raw dir  : ${RAW_DIR}"
echo "  Dataset  : ${DATASET}"
echo "  Run dir  : ${RUN_DIR}"
echo "  Output   : ${OUTPUT_DIR}"
echo "  dt_ms    : ${DT_MS}"
echo "  Seed     : ${SEED}"
echo "  Epochs   : ${EPOCHS:-<config>}  (phase1: ${PHASE1_EPOCHS:-<single-phase>})"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo -e "  ${YELLOW}DRY_RUN=1 — printing the plan, executing nothing.${RESET}"
fi
echo ""

if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${OUTPUT_DIR}" "${RUN_DIR}" "$(dirname "${DATASET}")"
    : >"${COMMAND_LOG}"

    export GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    export GIT_DIRTY=false
    if ! git diff --quiet HEAD 2>/dev/null; then GIT_DIRTY=true; fi
    export GIT_DIRTY
    export PY_VERSION="$("${PYTHON}" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unknown)"
    export STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    export CONFIG RAW_DIR DATASET RUN_DIR OUTPUT_DIR SEED DT_MS
    export EPOCHS PHASE1_EPOCHS BATCH_SIZE LR COMMAND_LOG

    "${PYTHON}" - "${MANIFEST}" <<'PYMANIFEST'
import json, os, sys
manifest = {
    "started_utc": os.environ.get("STARTED_UTC", ""),
    "git_sha": os.environ.get("GIT_SHA", "unknown"),
    "git_dirty": os.environ.get("GIT_DIRTY", "false") == "true",
    "python_version": os.environ.get("PY_VERSION", "unknown"),
    "config": os.environ.get("CONFIG", ""),
    "raw_dir": os.environ.get("RAW_DIR", ""),
    "dataset": os.environ.get("DATASET", ""),
    "run_dir": os.environ.get("RUN_DIR", ""),
    "output_dir": os.environ.get("OUTPUT_DIR", ""),
    "seed": int(os.environ.get("SEED") or 0),
    "dt_ms": float(os.environ.get("DT_MS") or 0),
    "epochs": os.environ.get("EPOCHS", "config"),
    "phase1_epochs": os.environ.get("PHASE1_EPOCHS", "none"),
    "batch_size": os.environ.get("BATCH_SIZE", "config"),
    "learning_rate": os.environ.get("LR", "config"),
    "command_log": os.environ.get("COMMAND_LOG", ""),
}
with open(sys.argv[1], "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PYMANIFEST
fi

# ═════════════════════════════════════════════════════════════
# Phase A — ETL: raw sessions → tensor dataset
# ═════════════════════════════════════════════════════════════
stage_header "Phase A — ETL: Data Preparation"
run_py "Phase A" scripts/prepare_data.py \
    --raw_dir "${RAW_DIR}" \
    --output "${DATASET}" \
    --dt_ms "${DT_MS}" \
    --seed "${SEED}"
require_outputs "Phase A" "${DATASET}"
stage_done "Phase A"

# ═════════════════════════════════════════════════════════════
# Phase B — Training (Hybrid Funnel when PHASE1_EPOCHS is set)
# ═════════════════════════════════════════════════════════════
stage_header "Phase B — Training Engine"
run_py "Phase B" "${TRAIN_ARGS[@]}"
require_outputs "Phase B" "${BEST_MODEL}"
stage_done "Phase B"

# ═════════════════════════════════════════════════════════════
# Phase C — Mechanism / manifold analysis
# ═════════════════════════════════════════════════════════════
stage_header "Phase C — Dynamics & Manifold Analysis"
run_py "Phase C" scripts/analyze_dynamics.py \
    --checkpoint "${BEST_MODEL}" \
    --dataset "${DATASET}" \
    --output "${OUTPUT_DIR}/mechanism_analysis.png"
require_outputs "Phase C" "${OUTPUT_DIR}/mechanism_analysis.png"
stage_done "Phase C"

# ═════════════════════════════════════════════════════════════
# Phase D — In-silico lesion (virtual ablation)
# ═════════════════════════════════════════════════════════════
stage_header "Phase D — In-Silico Lesion Analysis"
run_py "Phase D" scripts/simulate_lesion.py \
    --checkpoint "${BEST_MODEL}" \
    --dataset "${DATASET}" \
    --output "${OUTPUT_DIR}/ablation_kinematics.png" \
    --stats_output "${OUTPUT_DIR}/lesion_statistics.csv" \
    --dt_ms "${DT_MS}"
require_outputs "Phase D" \
    "${OUTPUT_DIR}/ablation_kinematics.png" \
    "${OUTPUT_DIR}/lesion_statistics.csv"
stage_done "Phase D"

# ═════════════════════════════════════════════════════════════
# Phase E — GRU recurrence Jacobian eigenvalue spectrum
#   GRU-only on purpose: only d h_{t+1}/d h_t is a state-transition
#   matrix whose eigenvalues carry stability meaning.  --full_system
#   returns input sensitivity (H x F), which has no eigenvalues.
# ═════════════════════════════════════════════════════════════
stage_header "Phase E — Jacobian Eigenvalue Spectrum (GRU recurrence)"
run_py "Phase E" scripts/analyze_jacobian.py \
    --checkpoint "${BEST_MODEL}" \
    --dataset "${DATASET}" \
    --output "${OUTPUT_DIR}/jacobian_spectrum.png" \
    --dt_ms "${DT_MS}" \
    --backend jax
require_outputs "Phase E" \
    "${OUTPUT_DIR}/jacobian_spectrum.png" \
    "${OUTPUT_DIR}/jacobian_spectrum.json"
stage_done "Phase E"

# ═════════════════════════════════════════════════════════════
# Phase F — Multisensory integration window
# ═════════════════════════════════════════════════════════════
stage_header "Phase F — Multisensory Integration Window"
run_py "Phase F" scripts/analyze_integration.py \
    --checkpoint "${BEST_MODEL}" \
    --dataset "${DATASET}" \
    --output "${OUTPUT_DIR}/integration_window.png" \
    --summary "${OUTPUT_DIR}/integration_summary.json" \
    --dt_ms "${DT_MS}"
require_outputs "Phase F" \
    "${OUTPUT_DIR}/integration_window.png" \
    "${OUTPUT_DIR}/integration_summary.json"
stage_done "Phase F"

# ═════════════════════════════════════════════════════════════
# Phase G — Visual-channel noise sensitivity (psychophysics)
# ═════════════════════════════════════════════════════════════
stage_header "Phase G — Psychophysics: Visual Noise Sensitivity"
run_py "Phase G" scripts/simulate_psychophysics.py \
    --checkpoint "${BEST_MODEL}" \
    --dataset "${DATASET}" \
    --raw_dir "${RAW_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}"
require_outputs "Phase G" \
    "${OUTPUT_DIR}/bayesian_reliability.png" \
    "${OUTPUT_DIR}/psychophysics_summary.json"
stage_done "Phase G"

# ═════════════════════════════════════════════════════════════
# Phase H — Unsupervised gating-strategy clustering
#   Previously missing from the runner while `make analyze` and the
#   README both counted it as one of the six analyses.
# ═════════════════════════════════════════════════════════════
stage_header "Phase H — Gating Strategy Clustering"
run_py "Phase H" scripts/analyze_gating.py \
    --checkpoint "${BEST_MODEL}" \
    --dataset "${DATASET}" \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}"
require_outputs "Phase H" \
    "${OUTPUT_DIR}/gating_cluster_summary.json" \
    "${OUTPUT_DIR}/gating_cluster_statistics.csv"
stage_done "Phase H"

# ═════════════════════════════════════════════════════════════
# Phase I — Autoregressive closed-loop generation
# ═════════════════════════════════════════════════════════════
stage_header "Phase I — Autoregressive Closed-Loop Generation"
run_py "Phase I" scripts/simulate_autoregressive.py \
    --checkpoint "${BEST_MODEL}" \
    --output_dir "${SIM_DIR}" \
    --dt_ms "${DT_MS}"
require_outputs "Phase I" \
    "${SIM_DIR}/events.csv" \
    "${SIM_DIR}/kinematics.csv"
stage_done "Phase I"

# ═════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════
if [[ "${DRY_RUN}" == "1" ]]; then
    echo -e "${YELLOW}Dry run finished — no stage was executed.${RESET}"
    exit 0
fi

FINISHED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${PYTHON}" - "${MANIFEST}" "${FINISHED_UTC}" <<'PY' || true
import json
import sys

path, finished = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as fh:
    manifest = json.load(fh)
manifest["finished_utc"] = finished
manifest["status"] = "complete"
with open(path, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2)
PY

echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  ✔ Pipeline complete — every stage verified.${RESET}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${RESET}"
echo ""
echo "  Figures:"
echo "    ${OUTPUT_DIR}/mechanism_analysis.png"
echo "    ${OUTPUT_DIR}/ablation_kinematics.png"
echo "    ${OUTPUT_DIR}/jacobian_spectrum.png"
echo "    ${OUTPUT_DIR}/integration_window.png"
echo "    ${OUTPUT_DIR}/bayesian_reliability.png"
echo ""
echo "  Data:"
echo "    ${OUTPUT_DIR}/lesion_statistics.csv"
echo "    ${OUTPUT_DIR}/jacobian_spectrum.json"
echo "    ${OUTPUT_DIR}/integration_summary.json"
echo "    ${OUTPUT_DIR}/psychophysics_summary.json"
echo "    ${OUTPUT_DIR}/gating_cluster_summary.json"
echo "    ${OUTPUT_DIR}/gating_cluster_statistics.csv"
echo ""
echo "  Synthetic session:"
echo "    ${SIM_DIR}/events.csv"
echo "    ${SIM_DIR}/kinematics.csv"
echo ""
echo "  Provenance:"
echo "    ${MANIFEST}"
echo "    ${COMMAND_LOG}"
echo ""
