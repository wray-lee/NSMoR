# Product Requirements Document (PRD): NSMoR — Neural Sensori-Motor Response Model

**Status**: Baseline & Active Specification  
**Target Path**: `PRD.md`  
**Version**: 2.1.0  
**Author**: Antigravity / NSMoR Computational Neuroscience Team  

---

## 1. Title & Overview

**NSMoR (Neural Sensori-Motor Response)** is a bio-plausible, white-box computational neuroscience framework designed to model cricket multi-sensory integration and escape decision dynamics. By combining a transient Leaky Integrate-and-Fire (LIF) spiking pathway with a continuous Gated Recurrent Unit (GRU) pathway managed by a learned Causal Inference Router (Mixture-of-Recursions), NSMoR accounts for how animals rapidly weigh conflicting visual (looming) and mechanosensory (wind) cues. Operating under a gradient-isolated **Hybrid Funnel** training strategy (Phase 1 sensory frontend training; Phase 2 biophysical core training), NSMoR exposes all internal neural states for low-dimensional manifold visualization, fixed-point Jacobian spectrum extraction, virtual in-silico lesions, and unsupervised routing strategy clustering.

---

## 2. Problem Statement & User Value

### Problem Statement
In computational neuroscience and animal behavior modeling, traditional neural networks suffer from a fundamental trade-off:
1. **Black-box Deep Learning (e.g. standard LSTMs/RNNs)**: Achieves high behavioral prediction accuracy but lacks biological realism, rendering internal states uninterpretable to neuroscientists.
2. **Hand-tuned Biophysical Models (e.g. Hodgkin-Huxley networks)**: Fully interpretable, but computationally expensive, difficult to optimize over large empirical datasets, and prone to poor generalization across variable behavioral regimes.
3. **Data Leakage & Inconsistent Pipeline Semantics**: Unchecked temporal leakage (e.g. using future trajectory frames or session-wide priors) inflates performance metrics ($R^2 > 0.45$) while breaking true out-of-session generalization.

### User Value
- **White-Box Dynamical Interpretability**: All internal variables (routing gates $g(t)$, membrane potentials $V_{\text{LIF}}(t)$, spike events, GRU hidden vectors $h_{\text{GRU}}(t)$) are exposed for dynamical systems and manifold analysis.
- **Honest Generalization (Pipeline Semantics v2.1)**: Session-grouped 5-fold Out-Of-Fold (OOF) MCMC prior estimation guarantees zero data leakage across sessions.
- **Publication-Grade Statistical & Biophysical Rigor**: Every analysis module includes block bootstrap confidence intervals, Cohen's $d$ effect sizes, Holm-Bonferroni FWER corrections, and GMM+BIC calibrated threshold selection.
- **Automated AI Harness Governance**: Governed by a 5-layer AI harness (`AGENTS.md`, `CLAUDE.md`, `BOUNDARY.md` matrix, `HARNESS.md`, `.claude/`), enabling double-blind review of code modifications prior to release.

---

## 3. Requirements & Scope

### In-Scope (P0 / P1 / P2)

#### Functional Requirements
- **[P0] Dual-Pathway MoR Core (`model_nsmor_core.py`)**:
  - `SensoryEncoder`: Maps 4D sensory inputs (visual angle, wind state, velocity, acceleration) to hidden representation space $H=64$.
  - `LIFCell`: Implements biophysical spiking kinetics including absolute/relative refractory periods, ATP metabolic consumption, adaptation, and lateral inhibition.
  - `GRUUnit`: Handles continuous temporal integration across variable-length sequences.
  - `MoRRouter`: Learned per-step blending gate producing softmax routing probabilities $[g_{\text{LIF}}, g_{\text{GRU}}]$.
  - `DirectionHead`: LayerNorm $\to$ ReLU $\to$ Linear final decoder predicting target kinematic response (velocity).
- **[P0] Hybrid Funnel Two-Phase Training (`scripts/train.py`, `loss.py`)**:
  - *Phase 1 (`FrontendLoss`)*: Train dendritic frontend using masked MSE.
  - *Phase 2 (`BioDecisionLoss`)*: Freeze frontend via `requires_grad=False` and train decision core using joint loss:
    $$\mathcal{L}_{\text{Phase2}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{reg}}\mathcal{L}_{\text{router}} + \lambda_{\text{energy}}\mathcal{L}_{\text{ATP}} + \lambda_{\text{sparse}}\mathcal{L}_{\text{sparse}} + \lambda_{\text{jerk}}\mathcal{L}_{\text{jerk}}$$
- **[P0] v2.1 Data Ingestion & Feature Pipeline (`pipeline/`, `data_extractor.py`)**:
  - Process raw CSVs into 8D per-frame feature tensors $[B, T, 8]$ and 5D MCMC snapshots.
  - Session-grouped 5-fold cross-validation OOF prior estimation to eliminate session leakage.
- **[P1] Publication Interpretability Suite (`scripts/analyze_*.py`)**:
  - 6 mandatory analysis entry points: `dynamics`, `jacobian`, `integration`, `psychophysics`, `lesion`, `cluster`.
- **[P1] Hermetic Execution & Reproducibility**:
  - WSL Zsh environment integration, Docker containerization (`docker compose run --rm nsmor pipeline`), and `Makefile` entry points.

#### Non-Functional Requirements
- **[P0] Mathematical & Shape Assertions**: Every `forward()` method must enforce runtime tensor shape assertions (`assert tensor.shape == (B, T, H)`).
- **[P0] Numerical Overflow Interception**: 100% protection against `NaN`/`Inf` during loss computation and gradient clipping.
- **[P1] Test Coverage**: Maintain passing status across the complete pytest suite ($\ge 114$ tests).

### Out-of-Scope
- Direct closed-loop motor control of physical hardware (strictly simulated in-silico generation).
- Single-cell patch-clamp voltage modeling beyond point-neuron LIF abstractions.

---

## 4. Technical Design & Architecture

### System Data Flow & Pipeline Semantics v2.1

```
Raw CSV Datasets (Kinematics & Events)
          │
          ▼
load_and_concat_sessions() ──> pd.DataFrame
          │
          ▼
assign_ground_truth_labels() ──> [ESCAPE, PREWALK, PRE_ACTIVE, NO_RESPONSE]
          │                     (v2.1 escape-first branch ordering)
          ▼
train_mcmc() ──> Session-Grouped 5-Fold OOF Priors (No session leakage)
          │
          ▼
create_dataloader() ──> PyTorch DataLoader yielding (X_batch [B,T,8], Y_batch [B,T])
          │
          ▼
NSMoRCore (Hybrid Funnel Model)
   ├─ Phase 1: FrontendEncoder (MSE loss on sensory features)
   └─ Phase 2: BioDecisionCore (LIF + GRU + MoRRouter + DirectionHead)
          │
          ▼
Mechanistic Interpretation Engine (Dynamics, Jacobians, Lesions, Clusters)
```

### Feature Contracts

#### 1. Per-Frame Feature Layout ($D=8$)
$$\mathbf{x}(t) = [v_{\text{vis}}(t), \text{wind}(t), v_{\text{kine}}(t-1), a_{\text{kine}}(t-1), P_{\text{escape}}, P_{\text{prewalk}}, P_{\text{pre\_active}}, P_{\text{no\_response}}]^\top$$

#### 2. MCMC Snapshot Vector ($D=5$)
$$\mathbf{s}_{\text{TTC-50ms}} = [v_{\text{vis}}, l/v \text{ ratio}, \text{wind\_state}, \bar{v}_{\text{bg}(200\text{ms})}, a_{\text{max}(200\text{ms})}]^\top$$

---

## 5. Invariants & Non-Negotiable Constraints

1. **Frozen Mathematical Core**: `nsmor/model_nsmor_core.py` and `nsmor/loss.py` are strictly frozen. Modifications require explicit user override.
2. **Git Author Identity**: Commits MUST be created under `wray-lee <i@wray7.top>` (verified GitHub primary email).
3. **Sampling-Rate Invariance**: Physical time constants (`lif_tau_syn`, `lif_rel_refract_ms`) must be rescaled automatically via $\alpha = \exp(-\Delta t / \tau)$.
4. **Statistical Standard**: Multi-condition evaluations must calculate Cohen's $d$, use Holm-Bonferroni FWER corrections, and report Wilcoxon+Hodges-Lehmann metrics.

---

## 6. Acceptance Criteria

- [x] **Core Architecture**: Hybrid Funnel two-phase model implemented and verified with shape assertions.
- [x] **Data Pipeline**: v2.1 semantics validated with `pipeline_semantics_version="2.1"` and session-grouped 5-fold OOF priors.
- [x] **Generalization Performance**: Model achieves honest validation $R^2 \approx 0.37 \pm 0.03$ without session leakage.
- [x] **Analysis Pipeline**: All 6 analysis scripts (`make analyze`) run end-to-end and generate 300 DPI publication figures in `results/`.
- [x] **Test Infrastructure**: `pytest tests/ -v` passes 100% of 114 test cases.
- [x] **Harness Governance**: 5-layer harness framework operational (`AGENTS.md`, `CLAUDE.md`, path `BOUNDARY.md` files, `HARNESS.md`, `.claude/`).

---

## 7. Risks & Mitigation

| Risk Area | Risk Description | Mitigation Strategy |
| :--- | :--- | :--- |
| **Data Leakage** | Session overlap in prior estimation inflates $R^2$. | `validate_dataset_provenance()` guard rejects pre-v2.1 dataset artifacts. |
| **Numerical Instability** | Non-differentiable spiking boundaries generate `NaN` gradients. | AMP FP16/FP32 master weights, loss scaling, and post-clip gradient finiteness checks. |
| **Multi-Agent Deadlocks** | Developer and Reviewer loop endlessly on hyperparameter tweaks. | Orchestrator watchdog imposes max iteration limit ($N=10$) with user intervention hooks. |
