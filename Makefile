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
EPOCHS   ?= 150
CONFIG   ?= config/default.yaml
RUN_DIR  ?= runs/default
OUTPUT   ?= results
BEST     := $(RUN_DIR)/best_model.pth

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
	$(PYTHON) scripts/train.py --config $(CONFIG) --epochs 1 --output_dir $(RUN_DIR)/test
# ── Pre Data loading ─────────────────────────────────────────
load:
	$(PYTHON) scripts/pre_load_data.py $(RAW)
	$(PYTHON) scripts/pre_load_adapt.py $(RAW)

# ── Data Preparation ─────────────────────────────────────────
data: ## Run ETL pipeline (prepare_data.py)
	$(PYTHON) scripts/prepare_data.py --raw_dir $(RAW) --output $(DATA)

# ── Training ─────────────────────────────────────────────────
train: ## Run training engine (train.py)
	$(PYTHON) scripts/train.py --config $(CONFIG) --epochs $(EPOCHS) --output_dir $(RUN_DIR)

# ── Analysis (all 5 scripts) ─────────────────────────────────
analyze: dynamics lesion jacobian integration psychophysics ## Run all analysis scripts

dynamics: $(BEST) ## Run dynamics & manifold analysis
	$(PYTHON) scripts/analyze_dynamics.py --checkpoint $(BEST) --output $(OUTPUT)/mechanism_analysis.png

lesion: $(BEST) ## Run in-silico lesion analysis
	$(PYTHON) scripts/simulate_lesion.py --checkpoint $(BEST) --output $(OUTPUT)/ablation_kinematics.png

jacobian: $(BEST) ## Run Jacobian eigenvalue spectrum
	$(PYTHON) scripts/analyze_jacobian.py --checkpoint $(BEST) --output $(OUTPUT)/jacobian_spectrum.png

integration: $(BEST) ## Run multisensory integration window
	$(PYTHON) scripts/analyze_integration.py --checkpoint $(BEST) --output $(OUTPUT)/integration_window.png

psychophysics: $(BEST) ## Run Bayesian reliability analysis
	$(PYTHON) scripts/simulate_psychophysics.py --checkpoint $(BEST) --output_dir $(OUTPUT)

# ── Autoregressive Generation ────────────────────────────────
generate: $(BEST) ## Run autoregressive closed-loop generation
	$(PYTHON) scripts/simulate_autoregressive.py --checkpoint $(BEST) --output_dir $(OUTPUT)/sim_session

# ── Full Pipeline ────────────────────────────────────────────
pipeline: ## Execute full end-to-end experimental pipeline
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
