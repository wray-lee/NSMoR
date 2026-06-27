"""
YAML + argparse experiment configuration for NSMoR.

Provides :class:`ExperimentConfig` as the single source of truth for
all hyperparameters, dataset paths, and fine-tuning strategies.
Config can be loaded from a YAML file and/or overridden via CLI
arguments using :func:`parse_args`.

Example
-------
CLI::

    python train.py --config config/base.yaml --lr 5e-4 --freeze lif_cell router

Python::

    from nsmor.config_parser import ExperimentConfig
    cfg = ExperimentConfig.from_yaml("config/base.yaml")
    print(cfg.model.hidden_dim)
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

    # ── CLI override ─────────────────────────────────────────

    def apply_cli_overrides(self, args: argparse.Namespace) -> None:
        """
        Apply command-line overrides on top of the loaded config.

        Supported overrides:

        * ``--lr`` → ``training.learning_rate``
        * ``--epochs`` → ``training.num_epochs``
        * ``--batch-size`` → ``training.batch_size``
        * ``--hidden-dim`` → ``model.hidden_dim``
        * ``--freeze`` → ``finetune.freeze_modules``
        * ``--resume`` → ``checkpoint.resume_from``
        * ``--output-dir`` → ``checkpoint.output_dir``
        """
        if hasattr(args, "lr") and args.lr is not None:
            self.training.learning_rate = args.lr
        if hasattr(args, "epochs") and args.epochs is not None:
            self.training.num_epochs = args.epochs
        if hasattr(args, "batch_size") and args.batch_size is not None:
            self.training.batch_size = args.batch_size
        if hasattr(args, "hidden_dim") and args.hidden_dim is not None:
            self.model.hidden_dim = args.hidden_dim
        if hasattr(args, "freeze") and args.freeze is not None:
            self.finetune.freeze_modules = args.freeze
        if hasattr(args, "resume") and args.resume is not None:
            self.checkpoint.resume_from = args.resume
        if hasattr(args, "output_dir") and args.output_dir is not None:
            self.checkpoint.output_dir = args.output_dir


# ═══════════════════════════════════════════════════════════════
# argparse integration
# ═══════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the default :class:`argparse.ArgumentParser` for NSMoR
    training scripts.

    Returns:
        Parser with ``--config`` and all CLI override flags.
    """
    parser = argparse.ArgumentParser(
        description="NSMoR training configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=None,
        help="Override hidden dimension.",
    )
    parser.add_argument(
        "--freeze", nargs="+", default=None, metavar="MODULE",
        help="Sub-modules to freeze (e.g. lif_cell router).",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> ExperimentConfig:
    """
    Parse CLI arguments and return a fully resolved :class:`ExperimentConfig`.

    If ``--config`` is given, YAML is loaded first, then CLI flags
    override individual values.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Resolved experiment configuration.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.config is not None:
        cfg = ExperimentConfig.from_yaml(args.config)
    else:
        cfg = ExperimentConfig()

    cfg.apply_cli_overrides(args)
    return cfg


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


