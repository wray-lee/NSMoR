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
    lif_alpha: float = 0.9
    lif_threshold: float = 1.0
    lif_beta: float = 0.5
    # Refractory periods & synaptic dynamics (Hodgkin & Huxley 1952)
    lif_abs_refract_steps: int = 0   # absolute refractory period (timesteps; 0=disabled)
    lif_rel_refract_steps: int = 0   # relative refractory decay length (timesteps; 0=disabled)
    lif_tau_syn: float = 0.0         # synaptic time constant (dt units; 0=disabled)
    lif_v_rest: float = 0.0          # resting membrane potential (0=disabled)
    lif_v_reset: Optional[float] = None  # fixed reset potential (None=v_rest, standard AdEx)

    # Spike-frequency adaptation (AdEx model, Brette & Gerstner 2005)
    lif_tau_w: float = 0.0       # adaptation time constant (dt units; 0=disabled)
    lif_b_adapt: float = 0.0     # spike-triggered adaptation increment (0=disabled)

    # Short-Term Plasticity (Tsodyks-Markram model)
    # Ref: Tsodyks, Pawelzik & Markram 1998, Neural Computation.
    # When lif_tau_fac=0 AND lif_tau_rec=0, STP is fully disabled
    # (backward compatible: no extra parameters, no extra state).
    lif_tau_fac: float = 0.0     # facilitation time constant (dt units; 0=disabled)
    lif_tau_rec: float = 0.0     # recovery (depression) time constant (dt units; 0=disabled)
    lif_U_stp_init: float = 0.5  # baseline utilization (U in TM model; only used when STP enabled)

    # Lateral inhibition (Ritzmann & Camhi 1978)
    # Inhibitory interneuron pool strength. 0 disables.
    lif_lateral_inhibition: float = 0.0

    # Dendritic compartmentalization (London & Hausser 2005)
    # Time constant for visual-input dendritic IIR filter. 0 disables.
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
    random_seed: int = 42
    max_seq_len: Optional[int] = 1000  # crop sequences longer than this (cuDNN compat)


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


@dataclass
class LossConfig:
    """BioJointLoss hyperparameters and regularization schedule."""
    reduction: str = "mean"
    """MSE reduction mode: 'mean' or 'sum'."""
    target_rate: float = 0.05
    """Target mean firing rate for population sparsity L1 loss."""
    lambda_reg: float = 0.01
    """Router regularization weight (NOT warmup-scaled)."""
    lambda_energy: float = 0.0
    """ATP metabolic cost weight (Attwell & Laughlin 2001). 0 disables."""
    lambda_sparse: float = 0.0
    """Population sparsity L1 weight (Olshausen & Field 1996). 0 disables."""
    lambda_jerk: float = 0.0
    """Temporal coherence (jerk penalty) weight (Gabbiani et al. 1999). 0 disables."""
    jerk_threshold: float = 0.1
    """Threshold for sudden-change jerk mask (unused when mask=None)."""
    warmup_epochs: int = 0
    """Linear warmup epoch count for lambda_energy, lambda_sparse, lambda_jerk."""


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


