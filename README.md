# BioMoR — Data Pipeline & MCMC Module

**Bio-inspired Multi-sensory Object Recognition** for cricket neural modelling.

This module provides the deterministic data preprocessing pipeline and
the MCMC (Markov Chain Monte Carlo) probabilistic prior generator.
It bridges raw CSV outputs (events + kinematics) into structured
PyTorch DataLoaders for the downstream continuous model.

> **Phase 1 only** — data quality, shape verification, and MCMC probability.
> No RNN / GRU network is included in this phase.

---

## Project Structure

```
biomor/
├── config.py                 # Frozen dataclasses: thresholds, dimensions, windows
├── pipeline/
│   ├── io.py                 # CSV loading, session concatenation, per-trial extraction
│   ├── kinematics.py         # Savitzky-Golay / Gaussian smoothing, velocity / accel
│   └── labeling.py           # Ground truth: Pre_Active, Startle, Walk, NoResponse
├── data_extractor.py         # TTC-50ms snapshot + Trial-Start anchored sequences
├── mcmc_module.py            # PyTorch nn.Module + sklearn wrapper + Markov estimator
└── biomor_dataloader.py       # PyTorch Dataset + DataLoader with shape assertions
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
from biomor.pipeline.io import load_and_concat_sessions, extract_trial_data
from biomor.pipeline.labeling import assign_ground_truth_labels
from biomor.data_extractor import build_snapshot_dataset, build_sequence_dataset
from biomor.mcmc_module import train_mcmc
from biomor.biomor_dataloader import create_dataloader

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

| Column         | Type   | Description                      |
|----------------|--------|----------------------------------|
| session_id     | str    | Session identifier               |
| trial_id       | int    | Trial number within session      |
| time_ms        | float  | Timestamp in milliseconds        |
| x_pos          | float  | X position (cm)                  |
| y_pos          | float  | Y position (cm)                  |
| heading        | float  | Heading angle (degrees)          |
| velocity       | float  | Velocity (cm/s)                  |
| acceleration   | float  | Acceleration (cm/s²)             |
| visual_angle   | float  | Looming visual angle (degrees)   |
| wind_state     | int    | Wind stimulus (0 or 1)           |
| l_v_ratio      | float  | Looming l/v ratio                |

### Events CSV

| Column      | Type   | Description                      |
|-------------|--------|----------------------------------|
| session_id  | str    | Session identifier               |
| trial_id    | int    | Trial number                     |
| time_ms     | float  | Event timestamp (ms)             |
| event_type  | str    | Event type (see below)           |
| event_value | float  | Event value                      |

Event types: `trial_start`, `stimulus_onset`, `wind_onset`, `response_detected`, `trial_end`

---

## Per-Frame Feature Layout (dim = 8)

| Index | Symbol          | Description                              |
|-------|-----------------|------------------------------------------|
| 0     | v_vis(t)        | Real-time visual angle (degrees)         |
| 1     | wind(t)         | Wind stimulus state (0 / 1)              |
| 2     | v_kine(t-1)     | Previous-frame velocity (cm/s)           |
| 3     | a_kine(t-1)     | Previous-frame acceleration (cm/s²)      |
| 4     | P_startle       | MCMC prior: P(Startle)                   |
| 5     | P_walk          | MCMC prior: P(Walk)                      |
| 6     | P_pre_active    | MCMC prior: P(Pre_Active)                |
| 7     | P_no_response   | MCMC prior: P(NoResponse)                |

---

## MCMC Snapshot Features (dim = 5)

| Index | Name                | Description                              |
|-------|---------------------|------------------------------------------|
| 0     | visual_angle        | Instantaneous visual angle at TTC-50ms   |
| 1     | looming_velocity    | l/v ratio at TTC-50ms                    |
| 2     | wind_state          | Wind stimulus state (0 / 1)              |
| 3     | avg_velocity_bg     | Mean |velocity| in preceding 200ms       |
| 4     | max_acceleration_bg | Max |acceleration| in preceding 200ms    |

---

## Extensibility

All functions accept configuration objects with sensible defaults.
To support experimental variants (e.g., a 5.7 s silent baseline for
pure-wind trials), instantiate a custom config:

```python
from biomor.config import TimeWindowConfig

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

---

## Engineering & Architecture Capabilities

### Modular Architecture (LIF + GRU + Causal Gate)

BioMoR implements a **Mixture-of-Recursions (MoR)** architecture with fully decoupled sub-modules:

| Module | Class | Purpose |
|--------|-------|---------|
| Sensory Encoder | `SensoryEncoder` | Maps raw 4-D sensory features to hidden representation |
| LIF Pathway | `LIFCell` | Leaky Integrate-and-Fire spiking neuron for fast, event-driven transients |
| GRU Pathway | `GRUUnit` | Standard GRU for smooth, continuous temporal integration |
| Causal Gate | `MoRRouter` | Learned routing network blending LIF/GRU outputs per timestep |
| Decoder | `DirectionHead` | Final output layer for behavior/direction prediction |

```python
from biomor.model_biomor_core import BioMoRCore

model = BioMoRCore(
    sensory_dim=4,
    mcmc_dim=4,
    hidden_dim=64,
    num_gru_layers=1,
    dropout=0.1,
    lif_alpha=0.9,
    lif_threshold=1.0,
    lif_beta=0.5,
)
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

### Targeted Partial Weight Freezing

Freeze specific pathways for fine-tuning experiments:

```python
model = BioMoRCore()

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
from biomor.checkpoint import save_checkpoint, load_checkpoint

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
from biomor.biomor_dataloader import combine_datasets, create_dataloader_from_config

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

### Bio-Constrained Loss Function (`biomor.loss`)

Custom `nn.Module` that combines masked MSE with biological router regularization:

```python
from biomor.loss import BioJointLoss

criterion = BioJointLoss(reduction="mean")
loss = criterion(
    y_pred=predictions,      # (B, T)
    y_true=targets,          # (B, T)
    lengths=lengths,         # (B,)
    g_gru=g_gru,             # (B, T, 1) — from internals["routing_gates"][:, :, 1:2]
    lambda_reg=0.01,
)
```

**Masked MSE:** Computes MSE only over valid (non-padded) time-steps using the mask `torch.arange(T) < lengths.unsqueeze(1)`.

**Router Regularization:** Prevents the MoR Router from collapsing onto the higher-capacity GRU pathway:

$$\mathcal{L} = \text{MaskedMSE}(y_{\text{pred}}, y_{\text{true}}) + \lambda \cdot \frac{1}{N} \sum_{b,t} g_{\text{gru}}(b,t) \cdot \text{mask}(b,t)$$

### Main Training Engine (`scripts/train.py`)

Full training pipeline with validation and checkpointing:

```bash
python scripts/train.py --config config/default.yaml
python scripts/train.py --config config/default.yaml --lr 5e-4 --epochs 200
```

Features:
- YAML config + CLI overrides via `config_parser`
- AdamW optimizer with gradient clipping
- `return_internals=True` for router gate extraction during training
- Best-model checkpoint (`best_model.pth`) on validation improvement
- Periodic checkpoints (`epoch_X.pth`) at configurable intervals
- Automatic unfreezing at scheduled epoch

### Dynamical Systems Adapter (`biomor.analysis.dynamics`)

Adapter for interfacing GRU states with external fixed-point analysis libraries:

```python
from biomor.analysis.dynamics import FixedPointAdapter

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

BioMoR uses industrial-grade DevOps standards for biological simulations.
Every figure in the paper can be reproduced from a fresh clone with a single command.

### Quick Start

```bash
# 1. Clone and install
git clone https://github.com/<your-org>/biomor.git
cd biomor
make install

# 2. Run the full experimental pipeline (ETL → Train → 5 Analyses)
make pipeline
```

### Individual Stages

| Command              | Description                                    |
|----------------------|------------------------------------------------|
| `make data`          | ETL: raw CSVs → processed PyTorch dataset      |
| `make train`         | Train BioMoR model (100 epochs by default)     |
| `make analyze`       | Run all 5 analysis scripts sequentially        |
| `make dynamics`      | Dynamics & manifold visualisation              |
| `make lesion`        | In-silico lesion (virtual ablation)            |
| `make jacobian`      | Jacobian eigenvalue spectrum                   |
| `make integration`   | Multisensory integration window                |
| `make psychophysics` | Bayesian reliability & cue combination         |
| `make test`          | Run full test suite                            |
| `make clean`         | Remove caches and build artefacts              |

### Configuration

All hyperparameters are centralised in `config/default.yaml` and can be overridden via environment variables:

```bash
EPOCHS=200 LR=0.0005 make train
CONFIG=config/fast.yaml make pipeline
```

### Output Figures

After `make pipeline`, all publication-ready figures (300 DPI, Lancet/Cell aesthetic) are in `results/`:

| File                          | Analysis                        |
|-------------------------------|---------------------------------|
| `dynamics_manifold.png`       | Neural state-space trajectories |
| `lesion_comparison.png`       | Virtual ablation comparison     |
| `jacobian_spectrum.png`       | Eigenvalue complex plane        |
| `integration_window.png`      | Chronometric + vigor curves     |
| `bayesian_reliability.png`    | Psychometric + gate modulation  |

### Data Outputs

| File                            | Format |
|---------------------------------|--------|
| `lesion_statistics.csv`         | CSV    |
| `jacobian_stats.csv`            | CSV    |
| `integration_summary.json`      | JSON   |
| `psychophysics_summary.json`    | JSON   |

### Docker & CI/CD

BioMoR provides a hermetic Docker container with GPU passthrough to eliminate
all host-OS dependencies. A reviewer can reproduce the entire paper without
installing PyTorch, CUDA, or Python locally.

#### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Compose V2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (GPU only)

#### Reproduce the Paper from a Fresh Clone

```bash
git clone https://github.com/<your-org>/biomor.git
cd biomor
docker compose run --rm biomor pipeline     # ETL → Train → 5 Analyses
```

All figures and data outputs persist in the host `results/` directory via bind mounts.

#### Containerised Targets

| Command                                    | Description                          |
|--------------------------------------------|--------------------------------------|
| `docker compose run --rm biomor pipeline`  | Full end-to-end experimental pipeline|
| `docker compose run --rm biomor test`      | Pytest suite                         |
| `docker compose run --rm biomor train`     | Training engine only                 |
| `docker compose run --rm biomor analyze`   | All 5 analysis scripts               |
| `docker compose run --rm biomor bash`      | Interactive shell inside container   |

#### CI/CD

Every push and pull request to `main` triggers the GitHub Actions pipeline:
`checkout` → `make install` (Python 3.10) → `make test`.
The pipeline enforces deterministic verification of all PyTorch shape assertions
and padding masks before merge.
