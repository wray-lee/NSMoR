# NSMoR — Hybrid Funnel Architecture

**Bio-inspired Multi-sensory Object Recognition** for cricket neural modelling.

NSMoR implements a **Mixture-of-Recursions (MoR)** dual-pathway recurrent
network with a **Hybrid Funnel** training strategy that separates sensory
encoding from bio-physical decision-making via gradient isolation.

> **Hybrid Funnel** — two-phase training: Phase 1 fits the sensory frontend
> with simple MSE; Phase 2 freezes the frontend and trains the bio-decision
> core with ATP / sparsity / jerk penalties.

---

## Project Structure

```
nsmor/
├── config.py                 # Frozen dataclasses: thresholds, dimensions, windows
├── config_parser.py          # YAML experiment configuration
├── model_nsmor_core.py       # Hybrid Funnel: FrontendEncoder + BioDecisionCore
│   ├── FrontendEncoder       #   Dendritic filtering + SensoryEncoder (Phase 1)
│   ├── BioDecisionCore       #   LIF + GRU + Router + DirectionHead (Phase 2)
│   ├── LIFCell               #   Leaky Integrate-and-Fire spiking neuron
│   ├── GRUUnit               #   Packed-sequence GRU pathway
│   ├── MoRRouter             #   Learned per-step LIF/GRU blending gate
│   └── DirectionHead         #   Final decoder: LayerNorm → ReLU → Linear
├── loss.py                   # Hybrid Funnel losses
│   ├── FrontendLoss          #   Phase 1: masked MSE only
│   ├── BioDecisionLoss       #   Phase 2: MSE + router reg + ATP + sparsity + jerk
│   └── BioJointLoss          #   Backward-compatible wrapper
├── analysis/
│   ├── dynamics.py           # FixedPointAdapter for dynamical systems analysis
│   ├── gating_cluster.py     # Window-free unsupervised gating strategy clustering
│   └── uq.py                 # Uncertainty quantification: bootstrap CI, Cohen's d
├── analysis/
│   ├── dynamics.py           # FixedPointAdapter for dynamical systems analysis
│   ├── gating_cluster.py     # Window-free unsupervised gating strategy clustering
│   └── uq.py                 # Uncertainty quantification: bootstrap CI, Cohen's d
├── checkpoint.py             # Deterministic save/load with full RNG state
├── model_utils.py            # Canonical model loading from checkpoints
├── pipeline/
│   ├── io.py                 # CSV loading, session concatenation, per-trial extraction
│   ├── kinematics.py         # Savitzky-Golay / Gaussian smoothing, velocity / accel
│   └── labeling.py           # Ground truth: Pre_Active, Startle, Walk, NoResponse
├── data_extractor.py         # TTC-50ms snapshot + Trial-Start anchored sequences
├── mcmc_module.py            # PyTorch nn.Module + sklearn wrapper + Markov estimator
└── nsmor_dataloader.py       # PyTorch Dataset + DataLoader with shape assertions

scripts/
├── train.py                  # Two-phase training engine (--phase1_epochs)
├── analyze_dynamics.py       # Phase-space manifold and gate dynamics
├── analyze_jacobian.py       # Jacobian eigenvalue spectrum
├── analyze_integration.py    # Multisensory integration window
├── analyze_gating.py         # Unsupervised gating strategy clustering (NEW)
├── simulate_lesion.py        # In-silico lesion analysis
├── simulate_psychophysics.py # Bayesian reliability analysis
└── simulate_autoregressive.py # Closed-loop autoregressive generation
```

---

## Analysis Pipeline

NSMoR provides 6 analysis modules for mechanistic interpretation:

| Command | Output | Description |
|---------|--------|-------------|
| `make dynamics` | `mechanism_analysis.png` | 3D phase-space manifold, routing gates |
| `make jacobian` | `jacobian_spectrum.png` | Jacobian eigenvalue analysis |
| `make integration` | `integration_window.png` | Multisensory integration window |
| `make psychophysics` | `reliability_*.png` | Bayesian reliability analysis |
| `make lesion` | `ablation_kinematics.png` | In-silico lesion effects |
| `make cluster` | `gating_*.png, *.json` | **Unsupervised gating clustering** |

Run all analyses:
```bash
make analyze  # Runs all 6 analysis scripts
```

### Gating Cluster Analysis (Window-Free)

The `cluster` analysis performs unsupervised clustering of MoR routing strategies:

- **16-dim fingerprint**: mean, std, max, min, dominant fraction, entropy for LIF/GRU gates
- **Pearson correlation** with NaN guard
- **Silhouette-based k selection** (k ∈ {2,3,4,5})
- **UMAP visualization** with true labels and predicted clusters
- **Outputs**: 5 PNG figures + JSON summary + CSV statistics at 300 DPI

```bash
make cluster  # Requires trained model at runs/default/best_model.pth
```

---

## Data Flow

```
Raw CSVs (kinematics + events)
    ↓
load_and_concat_sessions()        →  pd.DataFrame
    ↓
extract_trial_data()              →  Dict per trial
    ↓
assign_ground_truth_labels()      →  Pre_Active / Startle / Walk / NoResponse
    ↓
extract_mcmc_snapshot()           →  5-D vector at TTC − 50 ms
extract_trial_sequence()          →  (X_seq, Y_seq) anchored at Trial Start
    ↓
train_mcmc()                      →  Cross-Entropy trained MCMCPriorGenerator
    ↓
create_dataloader()               →  DataLoader yielding (X_batch, Y_batch)
    X: (batch, seq_len, 8)
    Y: (batch, seq_len)
```

---

## Quick Start

```python
from nsmor.pipeline.io import load_and_concat_sessions, extract_trial_data
from nsmor.pipeline.labeling import assign_ground_truth_labels
from nsmor.data_extractor import build_snapshot_dataset, build_sequence_dataset
from nsmor.mcmc_module import train_mcmc
from nsmor.nsmor_dataloader import create_dataloader

# 1. Load data
data = load_and_concat_sessions(
    kinematics_paths=["data/session_0/kinematics.csv"],
    events_paths=["data/session_0/events.csv"],
)

# 2. Extract trials and assign labels
trials = [extract_trial_data(data, "session_0", t) for t in range(n_trials)]
labeled = assign_ground_truth_labels(trials)

# 3. Build datasets
snapshots, labels = build_snapshot_dataset(labeled)
sequences = build_sequence_dataset(labeled)

# 4. Train MCMC
model = train_mcmc(snapshots, labels)

# 5. Create DataLoader
priors = model.predict_proba(snapshots)
loader = create_dataloader(sequences, mcmc_priors=priors, batch_size=32)

# 6. Train downstream model
for X_batch, Y_batch in loader:
    # X_batch: (batch, seq_len, 8)
    # Y_batch: (batch, seq_len)
    ...
```

---

## CSV Format

### Kinematics CSV

| Column       | Type  | Description                    |
| ------------ | ----- | ------------------------------ |
| session_id   | str   | Session identifier             |
| trial_id     | int   | Trial number within session    |
| time_ms      | float | Timestamp in milliseconds      |
| x_pos        | float | X position (cm)                |
| y_pos        | float | Y position (cm)                |
| heading      | float | Heading angle (degrees)        |
| velocity     | float | Velocity (cm/s)                |
| acceleration | float | Acceleration (cm/s²)           |
| visual_angle | float | Looming visual angle (degrees) |
| wind_state   | int   | Wind stimulus (0 or 1)         |
| l_v_ratio    | float | Looming l/v ratio              |

### Events CSV

| Column      | Type  | Description            |
| ----------- | ----- | ---------------------- |
| session_id  | str   | Session identifier     |
| trial_id    | int   | Trial number           |
| time_ms     | float | Event timestamp (ms)   |
| event_type  | str   | Event type (see below) |
| event_value | float | Event value            |

Event types: `trial_start`, `stimulus_onset`, `wind_onset`, `response_detected`, `trial_end`

---

## Per-Frame Feature Layout (dim = 8)

| Index | Symbol        | Description                         |
| ----- | ------------- | ----------------------------------- |
| 0     | v_vis(t)      | Real-time visual angle (degrees)    |
| 1     | wind(t)       | Wind stimulus state (0 / 1)         |
| 2     | v_kine(t-1)   | Previous-frame velocity (cm/s)      |
| 3     | a_kine(t-1)   | Previous-frame acceleration (cm/s²) |
| 4     | P_startle     | MCMC prior: P(Startle)              |
| 5     | P_walk        | MCMC prior: P(Walk)                 |
| 6     | P_pre_active  | MCMC prior: P(Pre_Active)           |
| 7     | P_no_response | MCMC prior: P(NoResponse)           |

---

## MCMC Snapshot Features (dim = 5)

| Index | Name                | Description                            |              |                    |
| ----- | ------------------- | -------------------------------------- | ------------ | ------------------ |
| 0     | visual_angle        | Instantaneous visual angle at TTC-50ms |              |                    |
| 1     | looming_velocity    | l/v ratio at TTC-50ms                  |              |                    |
| 2     | wind_state          | Wind stimulus state (0 / 1)            | ------------ | ------------------ |
| 3     | avg_velocity_bg     | Mean                                   | velocity     | in preceding 200ms |
| 4     | max_acceleration_bg | Max                                    | acceleration | in preceding 200ms |

---

## Extensibility

All functions accept configuration objects with sensible defaults.
To support experimental variants (e.g., a 5.7 s silent baseline for
pure-wind trials), instantiate a custom config:

```python
from nsmor.config import TimeWindowConfig

wind_config = TimeWindowConfig(baseline_duration_ms=5700.0)
# Pass wind_config to extraction functions
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.24
- Pandas ≥ 2.0
- PyTorch ≥ 2.0
- scikit-learn ≥ 1.3
- SciPy ≥ 1.10
- tqdm
- matplotlib

---

## Engineering & Architecture Capabilities

### Hybrid Funnel Architecture (Frontend → .detach() → Backend)

NSMoR implements a **two-stage** architecture with gradient isolation:

```
X_batch [B,T,8] ──┬── Sensory_X [B,T,4] ─→ FrontendEncoder ─→ e_sensory [B,T,H]
                   │                                                    │
                   │                                          requires_grad toggle
                   │                                                    │
                   │                                         BioDecisionCore
                   │    MCMC_Prior [B,T,4] ─────────────────────→      │
                   │                                                    │
                   │                                   ┌── LIF Path  ─→ out_lif  [B,T,H]
                   │                                   ├── GRU Path  ─→ out_gru  [B,T,H]
                   │                                   ├── Router    ─→ gates    [B,T,2]
                   │                                   └── Integrate ─→ y_pred   [B,T]
```

| Stage | Module           | Class             | Phase 1 | Phase 2 |
| ----- | ---------------- | ----------------- | ------- | ------- |
| 1     | Dendritic Filter | (in FrontendEncoder) | trainable | frozen |
| 1     | Sensory Encoder  | `SensoryEncoder`  | trainable | frozen |
| 2     | LIF Pathway      | `LIFCell`         | frozen    | trainable |
| 2     | GRU Pathway      | `GRUUnit`         | frozen    | trainable |
| 2     | Causal Gate      | `MoRRouter`       | frozen    | trainable |
| 2     | Decoder          | `DirectionHead`   | frozen    | trainable |

**Gradient isolation** is achieved via `requires_grad` toggling — not
unconditional `.detach()` — so Phase 1 MSE gradients flow through the
frozen backend to reach the trainable frontend.

```python
from nsmor.model_nsmor_core import NSMoRCore

model = NSMoRCore(
    sensory_dim=4,
    mcmc_dim=4,
    hidden_dim=64,
    num_gru_layers=1,
    dropout=0.1,
    lif_alpha=0.9,
    lif_threshold=1.0,
    lif_beta=0.5,
)

# Sub-modules accessible as before (backward compatible)
model.sensory_encoder   # → model.frontend.sensory_encoder
model.lif_cell          # → model.backend.lif_cell
model.router            # → model.backend.router
```

### White-Box Weight/Activation Extraction

The `forward()` method supports `return_internals=True` for dynamical systems analysis (Manifold/Jacobian analysis):

```python
predictions, internals = model(X_batch, lengths, return_internals=True)

# Access internal states for analysis
routing_gates = internals["routing_gates"]      # (B, T, 2) — per-step blending weights
lif_potentials = internals["lif_potentials"]    # (B, T, H) — membrane potentials
lif_spikes = internals["lif_spikes"]            # (B, T, H) — spike events
gru_hidden = internals["gru_hidden"]            # (B, T, H) — GRU hidden states
```

### Autoregressive Closed-Loop Inference

The `forward()` method also supports state passing for autoregressive generation.
Pass a `states` dict to carry LIF membrane potentials and GRU hidden states
across time steps:

```python
states = None
for t in range(T):
    X_t = X_batch[:, t:t+1, :]  # (B, 1, 8)
    y_pred, internals, states = model(
        X_t, lengths=torch.tensor([1]), return_internals=True, states=states,
    )
    # y_pred is the predicted velocity — feed back as kinematic input
```

### Targeted Partial Weight Freezing

Freeze specific pathways for fine-tuning experiments:

```python
model = NSMoRCore()

# Freeze only the LIF pathway and causal gate
model.freeze_modules(["lif_cell", "router"])

# Freeze everything except GRU (GRU receives gradients)
model.freeze_modules([
    "sensory_encoder", "lif_cell", "router", "direction_head",
])
```

Valid module names: `sensory_encoder`, `lif_cell`, `gru_unit`, `router`, `direction_head`

### Deterministic State Checkpointing

Robust save/load for interrupted training with full state restoration:

```python
from nsmor.checkpoint import save_checkpoint, load_checkpoint

# Save checkpoint
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    loss=loss,
    config=config.to_dict(),
    path="runs/experiment_01/checkpoint_epoch_50.pt",
    scheduler=scheduler,  # optional
)

# Load and resume
checkpoint = load_checkpoint(
    path="runs/experiment_01/checkpoint_epoch_50.pt",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
)
# RNG states are restored for deterministic resumption
```

Checkpoint contents:

- `model_state_dict` — full model parameters and buffers
- `optimizer_state_dict` — optimizer momentum/variance buffers
- `scheduler_state_dict` — LR scheduler state (optional)
- `epoch` — current epoch index
- `loss` — loss value at save time
- `rng_state` — `torch.get_rng_state()` for CPU determinism
- `cuda_rng_state` — `torch.cuda.get_rng_state_all()` for GPU determinism
- `config` — parsed experiment configuration

### YAML-Based Flexible Dataset & Config Management

Single source of truth for all hyperparameters, dataset paths, and fine-tuning strategies:

```yaml
# config/base.yaml
model:
    hidden_dim: 64
    lif_alpha: 0.9

training:
    learning_rate: 0.001
    batch_size: 32
    num_epochs: 100

data:
    train_kinematics:
        - data/session_0/kinematics.csv
        - data/session_1/kinematics.csv
    train_events:
        - data/session_0/events.csv
        - data/session_1/events.csv

finetune:
    freeze_modules: ["lif_cell", "router"]
    unfreeze_after_epoch: -1
```

CLI overrides for rapid experimentation:

```bash
python train.py --config config/base.yaml --lr 5e-4 --freeze lif_cell router
python train.py --config config/base.yaml --hidden-dim 128 --epochs 200
```

Dynamic dataset combination for mixed experimental conditions:

```python
from nsmor.nsmor_dataloader import combine_datasets, create_dataloader_from_config

# Combine pure wind baseline with looming datasets
wind_seqs = build_sequence_dataset(wind_trials)
looming_seqs = build_sequence_dataset(looming_trials)
combined = combine_datasets(wind_seqs, looming_seqs)

loader = create_dataloader_from_config(
    config=cfg,
    sequences=combined,
    mcmc_priors=priors,
    split="train",
)
```

---

## Training & Analysis Components

### Hybrid Funnel Loss Functions (`nsmor.loss`)

Separate losses for each training phase:

**Phase 1 — `FrontendLoss`** (simple MSE):

```python
from nsmor.loss import FrontendLoss

criterion = FrontendLoss(reduction="mean")
loss = criterion(y_pred, y_true, lengths)
```

**Phase 2 — `BioDecisionLoss`** (MSE + bio penalties):

```python
from nsmor.loss import BioDecisionLoss

criterion = BioDecisionLoss(reduction="mean", target_rate=0.05)
loss = criterion(
    y_pred, y_true, lengths,
    g_gru=g_gru,             # (B, T, 1) — from routing_gates[:, :, 1:2]
    lambda_reg=0.01,
    lif_spikes=lif_spikes,   # (B, T, H) — from internals["lif_spikes"]
    lambda_energy=1e-3,      # ATP metabolic cost
    lambda_sparse=1e-2,      # Population sparsity L1
    lambda_jerk=1e-3,        # Temporal coherence (jerk penalty)
    annealing_factor=warmup, # Cosine warmup scaling
)
```

**Backward-compatible wrapper — `BioJointLoss`** (delegates to `BioDecisionLoss`):

```python
from nsmor.loss import BioJointLoss

criterion = BioJointLoss(reduction="mean")
loss = criterion(y_pred, y_true, lengths, g_gru, lambda_reg=0.01)
```

| Loss Component           | Formula | Phase |
| ------------------------ | ------- | ----- |
| Masked MSE               | $\frac{1}{N}\sum(y_{\text{pred}} - y_{\text{true}})^2 \cdot \text{mask}$ | Both |
| Router Regularization    | $\lambda_{\text{reg}} \cdot \frac{1}{N}\sum g_{\text{gru}} \cdot \text{mask}$ | 2 |
| ATP Metabolic Cost       | $\lambda_{\text{energy}} \cdot \bar{r}_{\text{spike}}$ | 2 |
| Population Sparsity (L1) | $\lambda_{\text{sparse}} \cdot \sqrt{H} \cdot |\hat{p} - p_{\text{target}}|$ | 2 |
| Temporal Coherence       | $\lambda_{\text{jerk}} \cdot \text{mean}(\text{jerk}^2)$ | 2 |

### Main Training Engine (`scripts/train.py`)

Full training pipeline with **two-phase** (Hybrid Funnel) or single-phase mode:

```bash
# Single-phase (backward compatible — all parameters trainable)
python scripts/train.py --config config/default.yaml --epochs 100

# Two-phase: Phase 1 (30 epochs, frontend MSE) → Phase 2 (70 epochs, bio loss)
python scripts/train.py --config config/default.yaml --epochs 100 --phase1_epochs 30

# Skip Phase 1, start directly with Phase 2 (bio loss from epoch 0)
python scripts/train.py --config config/default.yaml --epochs 100 --phase1_epochs 0
```

**Two-phase training schedule:**

| Epochs | Phase | Trainable | Loss | Optimizer |
| ------ | ----- | --------- | ---- | --------- |
| 0 … `phase1_epochs-1` | 1 | FrontendEncoder | FrontendLoss (MSE) | AdamW (single LR) |
| `phase1_epochs` … end | 2 | BioDecisionCore | BioDecisionLoss (full bio) | AdamW (LIF 0.3× LR) |

Features:

- YAML config + CLI overrides via `config_parser`
- Two-phase training with automatic phase transition (`--phase1_epochs`)
- AdamW optimizer with per-pathway learning rates (LIF 0.3× base LR)
- AMP (FP16 forward/backward, FP32 master weights) for RTX 5060 Ti
- Cosine warmup for bio-loss regularization terms
- NaN/Inf loss guard and post-clip gradient finiteness check
- Membrane health monitoring (V_max, spike_rate, w_adapt per epoch)
- Best-model checkpoint (`best_model.pth`) on validation improvement
- Periodic checkpoints (`epoch_X.pth`) at configurable intervals
- Automatic unfreezing at scheduled epoch

### Dynamical Systems Adapter (`nsmor.analysis.dynamics`)

Adapter for interfacing GRU states with external fixed-point analysis libraries:

```python
from nsmor.analysis.dynamics import FixedPointAdapter

adapter = FixedPointAdapter(model)

# Extract un-padded GRU trajectories
trajectories = adapter.extract_gru_states(dataloader)
# trajectories[i] has shape (T_i, H)

# Compute Jacobian at a specific hidden state
h_t = torch.randn(H, requires_grad=True)
x_t = sensory_encoder(sensory_input)  # (H,)
J = adapter.compute_jacobian_at_state(h_t, x_t)  # (H, H)
eigenvalues = torch.linalg.eigvals(J)

# Batch Jacobian computation
J_batch = adapter.compute_jacobian_batch(h_states, x_inputs)  # (N, H, H)
```

**State Extraction:** Runs the dataset through the model in eval mode, collects `internals["gru_hidden"]`, and un-pads into flat trajectories.

**Jacobian Interface:** Computes $\frac{\partial h_{t+1}}{\partial h_t}$ via PyTorch autograd for fixed-point analysis.

---

## Execution & Reproducibility

NSMoR uses industrial-grade DevOps standards for biological simulations.
Every figure in the paper can be reproduced from a fresh clone with a single command.

### Quick Start

```bash
# 1. Clone and install
git clone https://github.com/<your-org>/nsmor.git
cd nsmor
make install

# 2. Run the full experimental pipeline (ETL → Train → 5 Analyses)
make pipeline
```

### Individual Stages

| Command              | Description                                      |
| -------------------- | ------------------------------------------------ |
| `make load`          | Preload raw CSVs                                 |
| `make data`          | ETL: raw CSVs → processed PyTorch dataset        |
| `make train`         | Train NSMoR model (100 epochs by default)        |
| `make analyze`       | Run all 5 analysis scripts sequentially          |
| `make dynamics`      | Dynamics & manifold visualisation                |
| `make lesion`        | In-silico lesion (virtual ablation)              |
| `make jacobian`      | Jacobian eigenvalue spectrum                     |
| `make integration`   | Multisensory integration window                  |
| `make psychophysics` | Bayesian reliability & cue combination           |
| `make generate`      | Autoregressive closed-loop trajectory generation |
| `make test`          | Run full test suite                              |
| `make clean`         | Remove caches and build artefacts                |

### Configuration

All hyperparameters are centralised in `config/default.yaml` and can be overridden via environment variables:

```bash
EPOCHS=200 LR=0.0005 make train
CONFIG=config/fast.yaml make pipeline
```

### Output Figures

After `make pipeline`, all publication-ready figures (300 DPI, Lancet/Cell aesthetic) are in `results/`:

| File                       | Analysis                        |
| -------------------------- | ------------------------------- |
| `dynamics_manifold.png`    | Neural state-space trajectories |
| `lesion_comparison.png`    | Virtual ablation comparison     |
| `jacobian_spectrum.png`    | Eigenvalue complex plane        |
| `integration_window.png`   | Chronometric + vigor curves     |
| `bayesian_reliability.png` | Psychometric + gate modulation  |

### Data Outputs

| File                         | Format                  |
| ---------------------------- | ----------------------- |
| `lesion_statistics.csv`      | CSV                     |
| `jacobian_stats.csv`         | CSV                     |
| `integration_summary.json`   | JSON                    |
| `psychophysics_summary.json` | JSON                    |
| `sim_session/events.csv`     | CSV (cercus-compatible) |
| `sim_session/kinematics.csv` | CSV (cercus-compatible) |

### Docker & CI/CD

NSMoR provides a hermetic Docker container with GPU passthrough to eliminate
all host-OS dependencies. A reviewer can reproduce the entire paper without
installing PyTorch, CUDA, or Python locally.

#### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Compose V2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (GPU only)

#### Reproduce the Paper from a Fresh Clone

```bash
git clone https://github.com/<your-org>/nsmor.git
cd nsmor
docker compose run --rm nsmor pipeline     # ETL → Train → 5 Analyses
```

All figures and data outputs persist in the host `results/` directory via bind mounts.

#### Containerised Targets

| Command                                  | Description                           |
| ---------------------------------------- | ------------------------------------- |
| `docker compose run --rm nsmor pipeline` | Full end-to-end experimental pipeline |
| `docker compose run --rm nsmor test`     | Pytest suite                          |
| `docker compose run --rm nsmor train`    | Training engine only                  |
| `docker compose run --rm nsmor analyze`  | All 5 analysis scripts                |
| `docker compose run --rm nsmor bash`     | Interactive shell inside container    |

#### CI/CD

Every push and pull request to `main` triggers the GitHub Actions pipeline:
`checkout` → `make install` (Python 3.10) → `make test`.
The pipeline enforces deterministic verification of all PyTorch shape assertions
and padding masks before merge.
