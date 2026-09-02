# ═══════════════════════════════════════════════════════════════
# NSMoR — Makefile
# ═══════════════════════════════════════════════════════════════
#
# Standardised entry points for the NSMoR research pipeline.
#
#   make install    — install package + dev dependencies
#   make test       — run full test suite
#   make data       — run ETL pipeline
#   make train      — run training engine
#   make analyze    — run all 5 analysis scripts
#   make pipeline   — execute full end-to-end pipeline
#   make clean      — remove caches and build artefacts
# ═══════════════════════════════════════════════════════════════

PYTHON   ?= python
RAW	 ?= data/raw
DATA	 ?= data/processed/nsmor_dataset.pt
PRE_EPOCHS ?= 150
# Public pipeline override; PRE_EPOCHS remains a backward-compatible alias.
PHASE1_EPOCHS ?= $(PRE_EPOCHS)
EPOCHS   ?= 300
CONFIG   ?= config/default.yaml
RUN_DIR  ?= runs/default
OUTPUT   ?= results
SEED     ?= 42
BATCH_SIZE ?=
LR       ?=
BEST     := $(RUN_DIR)/best_model.pth

# Frame interval used by the ETL and by every analysis that converts
# frames to physical time.  Derived from the config so it cannot drift
# from the dt_ms the model was discretised at (run_pipeline.sh aborts on
# a mismatch).  Falls back to 10 ms if PyYAML is unavailable.
DT_MS ?= $(shell $(PYTHON) -c 'import yaml; cfg = yaml.safe_load(open("$(CONFIG)")) or {}; print(float((cfg.get("model") or {}).get("dt_ms", 10.0)))' 2>/dev/null || echo 10.0)

# Most entry scripts import `nsmor` without bootstrapping sys.path, so
# every target below works from a bare clone, installed or not.
export PYTHONPATH := $(CURDIR)$(if $(PYTHONPATH),:$(PYTHONPATH),)

.PHONY: install test data train analyze pipeline clean help

# ── Default target ───────────────────────────────────────────
help: ## Show available targets
	@echo ""
	@echo "  NSMoR — Available Targets"
	@echo "  ─────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Installation ─────────────────────────────────────────────
install: ## Install package in editable mode with dev deps
	$(PYTHON) -m pip install -e ".[dev]"
	@echo "✔ Installed."

# ── Testing ──────────────────────────────────────────────────
test: ## Run full test suite with verbose output
	$(PYTHON) -m pytest tests/ -v

modeltest:
	$(PYTHON) scripts/train.py --config $(CONFIG) --dataset $(DATA) --epochs 1 --phase1_epochs 1 --output_dir $(RUN_DIR)/test
# ── Pre Data loading ─────────────────────────────────────────
load:
	$(PYTHON) scripts/pre_load_data.py $(RAW)
	$(PYTHON) scripts/pre_load_adapt.py $(RAW)

# ── Data Preparation ─────────────────────────────────────────
data: ## Run ETL pipeline (prepare_data.py)
	$(PYTHON) scripts/prepare_data.py --raw_dir $(RAW) --output $(DATA) --dt_ms $(DT_MS) --seed $(SEED)

# ── Training ─────────────────────────────────────────────────
pretrain:
	$(PYTHON) scripts/train.py --config $(CONFIG) --dataset $(DATA) --phase1_epochs $(PRE_EPOCHS) --output_dir $(RUN_DIR)
posttrain:
	$(PYTHON) scripts/train.py --config $(CONFIG) --dataset $(DATA) --epochs $(EPOCHS) --output_dir $(RUN_DIR)

train: ## Run training engine (train.py)
	$(PYTHON) scripts/train.py --config $(CONFIG) --dataset $(DATA) --epochs $(EPOCHS) --phase1_epochs $(PRE_EPOCHS) --output_dir $(RUN_DIR)

# ── Analysis (all 6 analysis scripts) ──────────────────────────
analyze: dynamics lesion jacobian integration psychophysics cluster ## Run all analysis scripts

dynamics: $(BEST) ## Run dynamics & manifold analysis
	$(PYTHON) scripts/analyze_dynamics.py --checkpoint $(BEST) --dataset $(DATA) --output $(OUTPUT)/mechanism_analysis.png

lesion: $(BEST) ## Run in-silico lesion analysis
	$(PYTHON) scripts/simulate_lesion.py --checkpoint $(BEST) --dataset $(DATA) --output $(OUTPUT)/ablation_kinematics.png --stats_output $(OUTPUT)/lesion_statistics.csv --dt_ms $(DT_MS)

jacobian: $(BEST) ## Run Jacobian eigenvalue spectrum
	$(PYTHON) scripts/analyze_jacobian.py --checkpoint $(BEST) --dataset $(DATA) --output $(OUTPUT)/jacobian_spectrum.png --dt_ms $(DT_MS)

integration: $(BEST) ## Run multisensory integration window
	$(PYTHON) scripts/analyze_integration.py --checkpoint $(BEST) --dataset $(DATA) --output $(OUTPUT)/integration_window.png --summary $(OUTPUT)/integration_summary.json --dt_ms $(DT_MS)

psychophysics: $(BEST) ## Run Bayesian reliability analysis
	$(PYTHON) scripts/simulate_psychophysics.py --checkpoint $(BEST) --dataset $(DATA) --raw_dir $(RAW) --output_dir $(OUTPUT) --seed $(SEED)

cluster: $(BEST) ## Run unsupervised gating strategy clustering
	$(PYTHON) scripts/analyze_gating.py --checkpoint $(BEST) --dataset $(DATA) --config $(CONFIG) --output_dir $(OUTPUT)

# ── Autoregressive Generation ────────────────────────────────
generate: $(BEST) ## Run autoregressive closed-loop generation
	$(PYTHON) scripts/simulate_autoregressive.py --checkpoint $(BEST) --output_dir $(OUTPUT)/sim_session --dt_ms $(DT_MS)

# ── Full Pipeline ────────────────────────────────────────────
# Forwards the Makefile variables so `make pipeline` and `make train` +
# `make analyze` cannot drift apart (the runner defaults to single-phase
# training; the documented Hybrid Funnel is two-phase).
pipeline: ## Execute full end-to-end experimental pipeline
	PYTHON="$(PYTHON)" CONFIG="$(CONFIG)" RAW_DIR="$(RAW)" DATASET="$(DATA)" \
	RUN_DIR="$(RUN_DIR)" OUTPUT_DIR="$(OUTPUT)" DT_MS="$(DT_MS)" \
	EPOCHS="$(EPOCHS)" PHASE1_EPOCHS="$(PHASE1_EPOCHS)" \
	BATCH_SIZE="$(BATCH_SIZE)" LR="$(LR)" SEED="$(SEED)" \
	bash run_pipeline.sh

# ── Cleanup ──────────────────────────────────────────────────
clean: ## Remove caches, build artefacts, and old runs
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/
	@echo "✔ Caches removed."
	@echo "  To also remove runs/ and results/, run: make distclean"

distclean: clean ## Remove runs/ and results/ as well
	rm -rf runs/ results/
	@echo "✔ Full clean (runs/ + results/ removed)."
