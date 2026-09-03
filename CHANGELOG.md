# Changelog

All notable changes to the NSMoR project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

#### Phase 2: DataLoader Factory Integration
- **DataLoader Factory Module** (`nsmor/dataloader_factory.py`)
  - Intelligent worker auto-scaling: datasets <200 sequences use single-process mode, larger datasets auto-scale up to 4 workers
  - Automatic `pin_memory` detection for CUDA acceleration
  - Consistent `persistent_workers` and `prefetch_factor` policies
  - Spawn-safe collation function for multiprocessing compatibility
  - Factory functions: `create_optimized_dataloader()` and `create_dataloaders_from_config()`

#### Phase 3: Data Preparation Parallelization
- **Parallel MCMC training** (`scripts/prepare_data.py`)
  - Concurrent 5-fold model training with `ThreadPoolExecutor` (max_workers=5)
  - Session-grouped fold assignments ensure zero session-level leakage
  - Out-of-fold (OOF) prior aggregation with deterministic ordering
  - Per-fold provenance logging with session IDs and class distributions
  - Train-vs-serve distribution consistency diagnostics (KS statistic, variance ratio)

#### Phase 1: Configuration System Extension
- **YAML-based experiment configuration** (`nsmor/config_parser.py`)
  - Single source of truth for hyperparameters, dataset paths, and training strategies
  - CLI argument overrides for rapid experimentation
  - Support for partial weight freezing and scheduled unfreezing
- **DataLoader configuration parameters** (`config/default.yaml`)
  - `training.num_workers`: Worker count control (-1 for auto-scaling)
  - `training.pin_memory`: CUDA memory pinning toggle
  - `training.persistent_workers`: Worker persistence control

#### Round-3: Scientific Rigor Enhancements
- **Pipeline semantics v2.1** with escape-first branch ordering
  - Fixed PREWALK label collapse via responder-priority decision tree
  - Session-grouped 5-fold cross-validation with zero session-level leakage
  - Dataset provenance stamping: `pipeline_semantics_version="2.1"` and `mcmc_prior_provenance="oof_5fold_session_grouped_cv"`
  - Version guard in `model_utils.validate_dataset_provenance()` prevents loading datasets with mismatched semantics

- **Jacobian analysis GMM+BIC calibration** (`scripts/analyze_jacobian.py`)
  - Gaussian Mixture Model with Bayesian Information Criterion for threshold selection (ΔBIC > 10)
  - Per-label epoch-stratified gating (transient vs. sustained dynamics)
  - Frozen-input counterfactual control with independent recalibration
  - Sanity cap (0.3) with sensitivity analysis (±25%)
  - Wilson score confidence intervals for small-sample binomial proportions

- **MCMC cross-validation diagnostics** (`nsmor/mcmc_module.py`)
  - Per-class per-fold sample count assertions to detect data leakage
  - OOF prior variance lower-bound checks (min > 1e-6)
  - Train-vs-serve distribution consistency tests (KS statistic, variance ratio)
  - Fold-level provenance logging with session assignments and class histograms

- **Labeling waterfall improvements** (`nsmor/pipeline/labeling.py`)
  - Window-based score accumulation with hard trial-end truncation
  - Configurable `anchor_min_frames` threshold (default 3)
  - Responder-first branching: ESCAPE → PREWALK (recovery) → PRE_ACTIVE → NO_RESPONSE
  - Walking non-responders correctly classified as PRE_ACTIVE (not NO_RESPONSE)

- **Psychophysics analysis overhaul** (`scripts/simulate_psychophysics.py`)
  - Fixed nonparametric methods: Wilcoxon signed-rank test + Hodges-Lehmann estimator (no Shapiro pre-screening)
  - Holm-Bonferroni family-wise error rate correction across lesion comparisons
  - NaN-safe paired difference handling (excludes NaN pairs from family)

- **Biophysical parameter completeness**
  - `lif_rel_refract_ms`: Relative refractory period in physical time (ms) with alpha-domain conversion
  - `sensory_noise_std`: Stochastic resonance noise injection
  - `lif_lateral_inhibition`: Lateral inhibition coefficient
  - All LIF time constants now use millisecond representation with sampling-rate-invariant conversion

- **Dynamical systems analysis robustness** (`nsmor/analysis/dynamics.py`)
  - cuDNN RNN backward-pass fix: train/eval mode switching with try/finally guarantees
  - Membrane potential trajectory extraction for Jacobian computation
  - Fixed-point adapter with batch Jacobian support

### Changed

#### Training Pipeline Improvements
- **Two-phase Hybrid Funnel training** (`scripts/train.py`)
  - Phase 1 (Frontend): Sensory encoder trained with simple MSE loss
  - Phase 2 (Backend): Bio-decision core trained with full bio-physical loss (ATP, sparsity, jerk penalties)
  - Gradient isolation via `requires_grad` toggling (not unconditional `.detach()`)
  - AdamW optimizer with per-pathway learning rates (LIF pathway at 0.3× base LR)
  - Cosine warmup for bio-loss regularization terms with annealing factor
  - AMP (FP16) training with NaN/Inf guards and post-clip gradient finiteness checks
  - Membrane health telemetry: per-epoch V_max, spike_rate, w_adapt monitoring
  - Escape-band sensitivity audit with sustained-membership guard
  - Best-model checkpointing on validation improvement + periodic epoch snapshots

- **Analysis scripts standardization** (6 scripts updated)
  - `scripts/analyze_dynamics.py`: Uses factory DataLoader with auto-scaling
  - `scripts/analyze_gating.py`: Uses factory DataLoader with auto-scaling
  - `scripts/analyze_jacobian.py`: Uses factory DataLoader with auto-scaling
  - `scripts/analyze_integration.py`: Uses factory DataLoader with auto-scaling
  - `scripts/simulate_lesion.py`: Uses factory DataLoader with auto-scaling
  - `scripts/simulate_psychophysics.py`: Uses factory DataLoader with auto-scaling
  - All scripts now use `num_workers=-1` for automatic scaling (datasets <200 sequences → single-process, larger → up to 4 workers)
  - Replaced manual DataLoader construction with `create_optimized_dataloader()` factory

- **Block bootstrap improvements** (`nsmor/analysis/uq.py`)
  - Runtime sensitivity checks at block_size ∈ {2, 5, 10}
  - BCa extreme-z warning when |z₀| > 0.25
  - NaN p-value exclusion from Holm-Bonferroni family

- **Data preparation pipeline** (`scripts/prepare_data.py`)
  - v2.1 semantics with escape-first branch ordering
  - Parallel MCMC training using ThreadPoolExecutor (5 concurrent folds)
  - Waterfall label count logging with case-sensitive keys (n_ESCAPE, n_PREWALK, n_PRE_ACTIVE, n_NO_RESPONSE)
  - Sensitivity analysis at threshold scales {0.75, 1.0, 1.25}
  - MCMC train-vs-serve KS/variance diagnostics persisted to JSON

### Fixed

- **Critical**: cuDNN RNN backward-pass crash in `analyze_jacobian.py` when model not in training mode
- **Critical**: PREWALK label collapse (塌缩) via responder-priority branch reordering (v2.0 → v2.1 semantics)
- **Critical**: Session-level data leakage in MCMC cross-validation (now session-grouped 5-fold OOF)
- **Major**: Jacobian fixed-point threshold selection replaced ad-hoc percentiles with GMM+BIC calibration
- **Major**: Missing `FP_RESIDUAL_THRESHOLD_CAP` constant definition in `analyze_jacobian.py`
- **Major**: Labeling waterfall case-sensitivity bug (`funnel.get("n_Prewalk")` → `"n_PREWALK"`)
- **Major**: Walking non-responders misclassified as NO_RESPONSE (now PRE_ACTIVE)
- **Minor**: Gradient skip telemetry false positives from AMP loss scaling
- **Minor**: Scheduler state not checkpointed during two-phase training
- **Minor**: Per-sequence escape guard to prevent cross-sequence gradient contamination
- **Minor**: Missing `dt_ms` parameter now raises explicit error (was silent NaN cascade)

### Verified

- **Dataset provenance**: All generated datasets carry v2.1 semantics and OOF provenance keys
- **Label distribution health**:
  - ESCAPE: 204 trials
  - NO_RESPONSE: 129 trials (93 after TTC-50ms snapshot filtering)
  - PREWALK: 11 trials (recovered from 0 via branch reordering)
  - PRE_ACTIVE: 52 trials
- **Threshold sensitivity**: PREWALK count ∈ [8, 15] across ±25% threshold perturbation
- **Convergence**: 150-epoch training achieves R² = 0.3655, val_loss = 0.561 (honest generalization without session leakage)
- **Analysis pipeline**: All 6 analysis scripts execute without errors (EXIT=0) on v2.1 checkpoint and dataset
- **Test suite**: 114 tests passed (baseline 111 + 3 new tests for Round-3 mechanisms)
- **Backward compatibility**: Factory integration preserves existing API contracts; legacy `create_dataloader()` still functional

---

## [0.1.0] - 2026-07-29 (Pre-v2.1 Baseline)

### Initial Release Features

- **Hybrid Funnel Architecture** (`nsmor/model_nsmor_core.py`)
  - Dual-pathway recurrent network: LIF (spiking) + GRU (continuous)
  - MoR Router: learned per-timestep blending gate
  - White-box internals extraction: routing gates, membrane potentials, spikes, GRU hidden states
  - Autoregressive closed-loop inference with state passing

- **Biophysical LIF Mechanisms** (`LIFCell`)
  - Synaptic delay (IIR low-pass filtering)
  - Absolute and relative refractory periods
  - Spike-frequency adaptation (Brette & Gerstner 2005)
  - Short-term plasticity (Tsodyks-Markram model, optional)
  - Lateral inhibition
  - Stochastic resonance via sensory noise injection

- **Loss Functions** (`nsmor/loss.py`)
  - `FrontendLoss`: Masked MSE for sensory encoding
  - `BioDecisionLoss`: MSE + router regularization + ATP cost + population sparsity + jerk penalty
  - `BioJointLoss`: Backward-compatible wrapper

- **Data Pipeline** (`nsmor/pipeline/`)
  - CSV loading with session concatenation (`io.py`)
  - Kinematics processing with Savitzky-Golay/Gaussian smoothing (`kinematics.py`)
  - Ground truth labeling with escape/prewalk/pre-active/no-response categories (`labeling.py`)
  - TTC-50ms snapshot extraction and trial-start anchored sequences (`data_extractor.py`)

- **MCMC Prior Module** (`nsmor/mcmc_module.py`)
  - PyTorch nn.Module + sklearn wrapper
  - Markov chain estimator for behavioral priors

- **Analysis Modules**
  - `nsmor/analysis/dynamics.py`: Fixed-point adapter for dynamical systems analysis
  - `nsmor/analysis/gating_cluster.py`: Window-free unsupervised gating strategy clustering
  - `nsmor/analysis/uq.py`: Block bootstrap, BCa intervals, Cohen's d, Holm correction

- **Analysis Scripts**
  - `scripts/analyze_dynamics.py`: 3D phase-space manifold visualization
  - `scripts/analyze_jacobian.py`: Eigenvalue spectrum analysis
  - `scripts/analyze_integration.py`: Multisensory integration window
  - `scripts/analyze_gating.py`: Unsupervised routing strategy clustering with UMAP
  - `scripts/simulate_lesion.py`: In-silico lesion with block-bootstrap CI
  - `scripts/simulate_psychophysics.py`: Bayesian reliability and cue combination
  - `scripts/simulate_autoregressive.py`: Closed-loop trajectory generation

- **Checkpointing** (`nsmor/checkpoint.py`)
  - Deterministic save/load with full RNG state restoration
  - Interrupted training resumption support

- **Make-based Pipeline**
  - `make install`: Environment setup
  - `make data`: ETL pipeline
  - `make train`: Model training
  - `make analyze`: All 6 analysis scripts
  - `make pipeline`: Full end-to-end execution
  - `make test`: Pytest suite

- **Docker Support**
  - Hermetic container with GPU passthrough
  - NVIDIA Container Toolkit integration
  - Reproducible execution environment

---

## Notes

- **Breaking change in v2.1**: Datasets generated with v2.0 semantics cannot be loaded by v2.1+ code due to version guard enforcement.
- **Performance**: Two-phase training with v2.1 semantics achieves R² ≈ 0.37 (honest generalization) vs. R² ≈ 0.47 (inflated by session leakage in pre-v2.1 pipeline).
- **PREWALK label**: Small sample size (n=11) may limit statistical power for this category in some analyses.

