"""
JAX Experiment Configuration for NSMoR.

Reuses and extends the project's single-source-of-truth YAML schema
defined in ``nsmor.config_parser.ExperimentConfig``, adding JAX-specific
runtime options (XLA compilation, batch vmap, gradient accumulation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from nsmor.config_parser import (
    CheckpointConfig,
    DataPaths,
    ExperimentConfig,
    FineTuneConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)


@dataclass
class JAXRuntimeConfig:
    """JAX/XLA runtime configuration flags."""
    enable_jit: bool = True
    grad_accum_steps: int = 1
    platform: str = "auto"          # "auto", "gpu", "tpu", "cpu"
    preallocate_mem: bool = False   # XLA_PYTHON_CLIENT_PREALLOCATE
    mem_fraction: float = 0.8       # XLA_PYTHON_CLIENT_MEM_FRACTION


@dataclass
class JAXExperimentConfig:
    """Experiment configuration with JAX-specific execution parameters."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataPaths = field(default_factory=DataPaths)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    jax_runtime: JAXRuntimeConfig = field(default_factory=JAXRuntimeConfig)

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> JAXExperimentConfig:
        """Load configuration from YAML file, overlaying overrides."""
        base_cfg = ExperimentConfig.from_yaml(path)
        instance = cls(
            model=base_cfg.model,
            training=base_cfg.training,
            loss=base_cfg.loss,
            data=base_cfg.data,
            finetune=base_cfg.finetune,
            checkpoint=base_cfg.checkpoint,
            jax_runtime=JAXRuntimeConfig(),
        )
        if overrides:
            for k, v in overrides.items():
                if hasattr(instance.jax_runtime, k):
                    setattr(instance.jax_runtime, k, v)
                elif hasattr(instance.training, k):
                    setattr(instance.training, k, v)
                elif hasattr(instance.model, k):
                    setattr(instance.model, k, v)
        return instance

    @classmethod
    def from_experiment_config(
        cls,
        cfg: ExperimentConfig,
        jax_runtime: Optional[JAXRuntimeConfig] = None,
    ) -> JAXExperimentConfig:
        """Wrap an existing ExperimentConfig into JAXExperimentConfig."""
        return cls(
            model=cfg.model,
            training=cfg.training,
            loss=cfg.loss,
            data=cfg.data,
            finetune=cfg.finetune,
            checkpoint=cfg.checkpoint,
            jax_runtime=jax_runtime or JAXRuntimeConfig(),
        )


def load_config(
    path: Union[str, Path] = "config/default.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> JAXExperimentConfig:
    """Convenience helper to load YAML configuration for JAX pipelines."""
    return JAXExperimentConfig.from_yaml(path, overrides=overrides)
