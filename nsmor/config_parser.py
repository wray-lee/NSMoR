"""
YAML experiment configuration for NSMoR.

Provides :class:`ExperimentConfig` as the single source of truth for
all hyperparameters, dataset paths, and fine-tuning strategies.
Config can be loaded from a YAML file and/or overridden programmatically.

Example
-------
Python::

    from nsmor.config_parser import ExperimentConfig
    cfg = ExperimentConfig.from_yaml("config/base.yaml")
    print(cfg.model.hidden_dim)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ═══════════════════════════════════════════════════════════════
# Nested config dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    sensory_dim: int = 4
    mcmc_dim: int = 4
    hidden_dim: int = 64
    num_gru_layers: int = 1
    dropout: float = 0.1

    # Physical sampling interval (ms).  ALL time constants below are
    # declared in PHYSICAL TIME (ms) and converted internally via
    #   alpha = exp(-dt_ms / tau_ms)
    # so that changing the acquisition rate never silently rescales the
    # biophysics (Reviewer Round-1 BLOCKER-1).  The default matches
    # TimeWindowConfig.frame_interval_ms (100 Hz imaging).
    dt_ms: float = 10.0

    lif_alpha: float = 0.9
    lif_threshold: float = 1.0
    lif_beta: float = 0.5
    # Refractory periods & synaptic dynamics (Hodgkin & Huxley 1952)
    # Round-3 fix (Reviewer B MAJOR-1): refractory periods are declared
    # in PHYSICAL time (ms) like every other tau_* parameter, closing
    # the frame-unit loophole that BLOCKER-1 closed for the synaptic
    # filters.  The relative refractory default (20 ms) matches the
    # tens-of-ms threshold recovery measured in insect giant
    # interneurons (Bean 2007 reports 20-50% threshold elevation over
    # 10-50 ms).  Conversion to steps happens inside LIFCell via the
    # explicitly provided dt_ms.
    lif_abs_refract_ms: float = 0.0  # absolute refractory period (ms; 0=disabled)
    lif_rel_refract_ms: float = 20.0  # relative refractory decay length (ms; 0=disabled)
    lif_tau_syn: float = 0.0         # synaptic time constant (ms; 0=disabled)
    lif_v_rest: float = 0.0          # resting membrane potential (0=disabled)
    lif_v_reset: Optional[float] = None  # fixed reset potential (None=v_rest, standard AdEx)

    # Spike-frequency adaptation (AdEx model, Brette & Gerstner 2005)
    lif_tau_w: float = 0.0       # adaptation time constant (ms; 0=disabled)
    lif_b_adapt: float = 0.0     # spike-triggered adaptation increment (0=disabled)

    # Short-Term Plasticity (Tsodyks-Markram model)
    # Ref: Tsodyks, Pawelzik & Markram 1998, Neural Computation.
    # When lif_tau_fac=0 AND lif_tau_rec=0, STP is fully disabled
    # (backward compatible: no extra parameters, no extra state).
    lif_tau_fac: float = 0.0     # facilitation time constant (ms; 0=disabled)
    lif_tau_rec: float = 0.0     # recovery (depression) time constant (ms; 0=disabled)
    lif_U_stp_init: float = 0.5  # baseline utilization (U in TM model; only used when STP enabled)

    # Lateral inhibition (Ritzmann & Camhi 1978)
    # Inhibitory interneuron pool strength. 0 disables.
    lif_lateral_inhibition: float = 0.0
    # Spike-history EMA window for lateral inhibition (ms). Round-3
    # (Reviewer B MINOR-6): promoted from a hard-coded 50 ms fallback.
    # Within the 20-100 ms window of feedforward inhibition in cricket
    # cercal pathways.
    lif_inhib_tau_ms: float = 50.0

    # Dendritic compartmentalization (London & Hausser 2005)
    # Time constant for visual-input dendritic IIR filter (ms). 0 disables.
    lif_dendritic_tau: float = 0.0

    # Neuromodulatory gain on GRU pathway (Rillich & Stevenson 2011)
    # Octopamine-like arousal scaling. 0 disables.
    gru_neuromod_gain: float = 0.0

    # Neural noise injection (Douglass et al. 1993)
    # Gaussian noise std during training. 0 disables.
    sensory_noise_std: float = 0.0

    # Truncated BPTT for LIF pathway (Williams & Zipser 1989)
    # Detach LIF state every N timesteps to cap gradient path length.
    # 0 disables (full BPTT — risky for long sequences).
    lif_tbptt_steps: int = 64


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 100
    batch_size: int = 32
    grad_clip_norm: float = 1.0
    log_interval: int = 10
    checkpoint_interval: int = 10
    early_stopping_patience: int = 0  # 0 = disabled
    random_seed: int = 42
    max_seq_len: Optional[int] = 1000  # crop sequences longer than this (cuDNN compat)

    lr_warmup_epochs: int = 0
    """Linear LR warmup epoch count.  During the first ``lr_warmup_epochs``
    epochs the effective learning rate is ramped from 0 to :attr:`learning_rate`.
    Prevents the cold-start overshoot that destabilises the coupled LIF+GRU
    gradient flow.  0 disables (backward compatible)."""

    # ── Target normalization for stable, interpretable convergence ──
    normalize_targets: bool = False
    """When ``True``, regress on the *mean-centered, variance-scaled* velocity
    target instead of the raw cm/s signal.  The raw velocity is a heavy-tailed
    signal (p99.99 ≈ 81 cm/s but max ≈ 13e6) where a handful of extreme frames
    dominate the masked MSE and produce the per-epoch loss noise.  Revealing
    the bulk (≈99.99% of frames, |y| < 100 cm/s) to the network::

        y_norm = (y - y_train_mean) / y_train_std

    gives a well-conditioned homoscedastic regression target.  Predictions are
    rescaled back to cm/s for reporting.  The normalization statistics are
    computed from the *training* split only (no validation leakage).  Statistics
    are transparently logged and cached for rescoring.  ``False`` preserves the
    original raw-velocity objective (backward compatible)."""

    target_clip_cm_s: float = 0.0
    """Clip the magnitude of the velocity target (in cm/s) before computing
    loss-based metrics.  ``0.0`` disables clipping (backward compatible).

    This is a *robust regression* safeguard: the raw velocity is
    heavy-tailed (p99.99 ≈ 81 cm/s, but max ≈ 1.3e7 cm/s), driven by a small
    number of tracking-artifact frames near trial onset (~index 1360+).
    A single such frame contributes ``(1.3e7)**2 ≈ 1.7e14`` to the batch MSE,
    completely swamping the resting/escape signal and producing the large
    epoch-to-epoch loss swings.  Clamping the target to a physiologically
    defensible cap (cricket escape vigour is tens of cm/s; a generous cap of
    ``100`` cm/s keeps every real escape response while removing the artifacts)
    keeps the loss landscape well-conditioned.  Predictions are clamped with
    the same value in ``compute_metrics`` so reported error stays physical.

    Note: clipping acts on ``y_true`` *before* the loss (and symmetrically on
    ``y_pred`` only for the reported metrics), so it is a pure training-target
    pre-processing — the frozen loss and model remain untouched.

    WARNING (statistical coherence): when ``normalize_targets`` is enabled but
    ``target_clip_cm_s`` is left at ``0.0`` (disabled), the loss standardises
    *every* frame — including the untrimmed ``~1e7 cm/s`` artifacts — by the
    small bulk std computed in :func:`~scripts.train.compute_target_stats`.
    Those outliers become O(1e5–1e6) sigma and dominate the standardized MSE,
    *amplifying* the very heavy-tail domination normalization was meant to
    suppress.  We therefore strongly discourage enabling normalization without
    a non-zero clip; enable both together for a statistically coherent target."""

    # ── DataLoader parallelism ──────────────────────────────────
    num_workers: int = -1
    """Worker processes for train/val/test DataLoaders.

    ``-1`` (default) auto-scales to ``min(4, os.cpu_count())`` inside
    :mod:`~nsmor.dataloader_factory`; ``0`` runs single-process (deterministic
    and debuggable, no worker overhead).  Windows spawns workers via ``spawn``
    so any worker-visible object must be import-top-level — the factory keeps
    collate closures module-level for exactly this reason."""

    pin_memory: bool = True
    """Page-lock host memory before H2D transfer. Only meaningful on CUDA;
    the factory silently disables it on CPU-only builds."""

    persistent_workers: bool = True
    """Keep worker processes alive across epochs (avoids ~seconds of
    re-spawn per epoch on large datasets). Requires ``num_workers > 0``;
    PyTorch raises if enabled with ``num_workers == 0``, so the factory
    coerces it to ``False`` whenever the resolved worker count is 0."""

    prefetch_factor: int = 2
    """Batches pre-loaded per worker ahead of the consumer. Ignored (and
    must not be forwarded) when ``num_workers == 0``; guarded by the factory."""

    escape_band_cm_s: float = 10.0
    """Absolute velocity-magnitude threshold (cm/s) defining the *high-velocity
    band* for the escape-signal audit in :func:`~scripts.train.compute_metrics`.

    This is a **magnitude heuristic**, NOT a stimulus-conditioned or
    per-trial-baseline-subtracted definition of escape: any frame with
    ``|y| >= escape_band_cm_s`` is grouped into the band.  Cricket intermittent
    locomotion routinely reaches ~20–100 cm/s during exploration, and noise /
    tracking rigs can exceed the band, so the band includes GO-fast and noisy
    frames — not only wind-triggered escapes.  To *condition* the audit on the
    wind stimulus (per-trial baseline subtraction, onset-aligned windows) is
    out of scope here and remains future work.

    Default ``10.0`` cm/s.  It is a *parameter* — pass a different value at the
    ``compute_metrics`` call site (or tune it) to sweep sensitivity; a single
    fixed value does not prove the model learned escapes, only that its error is
    lower in this velocity band."""


@dataclass
class DataPaths:
    """
    Dataset split paths.

    Each field is a list of file paths, allowing multiple CSVs to be
    concatenated for that split.
    """
    train_kinematics: List[str] = field(default_factory=list)
    train_events: List[str] = field(default_factory=list)
    val_kinematics: List[str] = field(default_factory=list)
    val_events: List[str] = field(default_factory=list)
    test_kinematics: List[str] = field(default_factory=list)
    test_events: List[str] = field(default_factory=list)


@dataclass
class FineTuneConfig:
    """Targeted freezing / fine-tuning strategy."""
    freeze_modules: List[str] = field(default_factory=list)
    """List of sub-module names to freeze.  See
    :meth:`~nsmor.model_nsmor_core.NSMoRCore.freeze_modules`."""

    unfreeze_after_epoch: int = -1
    """If >= 0, unfreeze all modules at this epoch for full fine-tuning."""


@dataclass
class CheckpointConfig:
    """Checkpoint and output paths."""
    output_dir: str = "runs/default"
    resume_from: Optional[str] = None
    """Path to a checkpoint file to resume from."""


@dataclass(frozen=True)
class ClusterGatingConfig:
    """
    Configuration for unsupervised gating strategy clustering.

    Window-free by design. NSMoR is Trial-Start anchored. TTC-50ms is only
    for MCMC prior 5-D snapshot. Baseline 5700ms is variant for pure-wind
    via TimeWindowConfig, not universal. Manual windows like [-5700:-500]
    inject human bias and break unsupervised claim. Clustering is
    unsupervised (silhouette selects k without labels); k=4 matches
    labeling.py cardinality; k=3 merged is for biological interpretation
    only and defined as Startle->Escape, Walk+Pre_Active->PreWalk,
    NoResponse->NoResponse. Pearson NaN guarded to 0.0 when std==0.
    """
    n_clusters: int = 4
    """Target number of clusters for k=4 analysis (matches labeling.py)."""

    n_clusters_range: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    """Range of k values to evaluate via silhouette score."""

    random_state: int = 42
    """Deterministic seed for all RNG operations."""

    use_umap: bool = True
    """Whether to compute UMAP embeddings for visualization."""

    fingerprint_dim: int = 16
    """Dimensionality of trial-level gate fingerprint vectors."""

    entropy_bins: int = 20
    """Number of histogram bins for entropy calculation in [0, 1]."""

    interp_length: int = 200
    """Target length for trajectory interpolation (visualization only)."""

    # No window field — window-free by design


@dataclass
class LossConfig:
    """BioJointLoss hyperparameters and regularization schedule."""
    reduction: str = "mean"
    """MSE reduction mode: 'mean' or 'sum'."""
    target_rate: float = 0.05
    """Target mean firing rate for population sparsity L1 loss."""
    lambda_reg: float = 0.2
    """Router regularization weight.

    NOT warmup-scaled: ``lambda_reg`` is passed at full strength from
    epoch 0.  Only ``lambda_energy``, ``lambda_sparse``, and
    ``lambda_jerk`` are ramped by the cosine warmup schedule.  This
    ensures the MoR router receives anti-collapse pressure before the
    GRU pathway monopolises the hidden state during the bio-loss
    warmup window.

    Diagnostic band: 0.1--0.3 prevents gate collapse (g_lif ~ 0.13
    everywhere) while keeping the MSE-dominated loss well-conditioned.
    """
    lambda_energy: float = 0.0
    """ATP metabolic cost weight (Attwell & Laughlin 2001). 0 disables."""
    lambda_sparse: float = 0.0
    """Population sparsity L1 weight (Olshausen & Field 1996). 0 disables."""
    lambda_jerk: float = 0.0
    """Temporal coherence (jerk penalty) weight (Gabbiani et al. 1999). 0 disables."""
    lambda_routing_aux: float = 0.0
    """Auxiliary routing differentiation weight (Ticket #15).

    Penalizes gate overlap between stimulus conditions (pure-wind vs
    visual-present trials). Encourages the router to learn condition-
    specific routing: high g_lif for wind transients, high g_gru for
    smooth looming. Hinge loss with margin=0.2.

    Warmup-scaled like ``lambda_energy/sparse/jerk``, ramping from 0
    over the first ``warmup_epochs``. Default 0.0 (disabled).
    """
    jerk_threshold: float = 0.1
    """Threshold for sudden-change jerk mask (unused when mask=None)."""
    warmup_epochs: int = 0
    """Cosine warmup epoch count for ``lambda_energy``, ``lambda_sparse``,
    and ``lambda_jerk``.  These three bio-loss terms are ramped from 0
    to their configured value over this window.

    ``lambda_reg`` is explicitly EXCLUDED from warmup scaling so the MoR
    router receives anti-collapse pressure from epoch 0."""


# ═══════════════════════════════════════════════════════════════
# Top-level config
# ═══════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """
    Top-level experiment configuration.

    Composed of nested dataclasses for each concern:

    * :attr:`model` — architecture hyperparameters
    * :attr:`training` — optimizer / loop settings
    * :attr:`data` — dataset split paths
    * :attr:`finetune` — freezing strategy
    * :attr:`checkpoint` — output / resume paths
    * :attr:`cluster_gating` — gating strategy clustering

    Construction
    ------------
    ::

        # From YAML
        cfg = ExperimentConfig.from_yaml("config/base.yaml")

        # From dict (e.g. parsed YAML)
        cfg = ExperimentConfig.from_dict(raw_dict)

        # Programmatic override
        cfg = ExperimentConfig()
        cfg.training.learning_rate = 5e-4
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataPaths = field(default_factory=DataPaths)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    cluster_gating: ClusterGatingConfig = field(default_factory=ClusterGatingConfig)

    # ── Constructors ─────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> ExperimentConfig:
        """
        Load configuration from a YAML file.

        Missing keys fall back to dataclass defaults.

        Args:
            path: Path to the YAML file.

        Returns:
            A fully populated :class:`ExperimentConfig`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> ExperimentConfig:
        """
        Construct from a plain dictionary.

        Nested dicts are mapped to the corresponding dataclass.
        Unknown top-level keys are silently ignored so that YAML
        files can contain comments / metadata without breaking
        the parser.
        """
        cfg = cls()

        if "model" in raw:
            cfg.model = _update_dataclass(cfg.model, raw["model"])
        if "training" in raw:
            cfg.training = _update_dataclass(cfg.training, raw["training"])
        if "loss" in raw:
            cfg.loss = _update_dataclass(cfg.loss, raw["loss"])
        if "data" in raw:
            cfg.data = _update_dataclass(cfg.data, raw["data"])
        if "finetune" in raw:
            cfg.finetune = _update_dataclass(cfg.finetune, raw["finetune"])
        if "checkpoint" in raw:
            cfg.checkpoint = _update_dataclass(cfg.checkpoint, raw["checkpoint"])
        if "cluster_gating" in raw:
            cfg.cluster_gating = _update_dataclass(cfg.cluster_gating, raw["cluster_gating"])

        return cfg

    # ── Serialisation ────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary (for YAML / JSON serialisation)."""
        from dataclasses import asdict
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Write this config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        return path



# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════

def _update_dataclass(instance: Any, updates: Dict[str, Any]) -> Any:
    """
    Return a shallow copy of *instance* with fields set from *updates*.

    Unknown keys in *updates* are silently ignored.
    """
    from dataclasses import fields as dc_fields

    cls = type(instance)
    kwargs: Dict[str, Any] = {}
    valid_names = {f.name for f in dc_fields(cls)}

    for key, value in updates.items():
        if key in valid_names:
            kwargs[key] = value

    return cls(**kwargs)


