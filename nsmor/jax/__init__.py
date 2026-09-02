"""
NSMoR JAX/XLA Optimization Package.

Provides high-performance JAX/Flax implementations of the NSMoR
dual-pathway recurrent architecture, supporting accelerated training,
bidirectional PyTorch checkpoint compatibility, and scalable data ingestion.
"""

from __future__ import annotations

from nsmor.jax.config import JAXExperimentConfig, JAXRuntimeConfig, load_config
from nsmor.jax.dataloader import JAXDataLoader, JAXDataset, load_nsmor_dataset
from nsmor.jax.model import (
    DirectionHeadJAX,
    MoRRouterJAX,
    NSMoRModel,
    SensoryEncoderJAX,
    load_from_torch_state_dict,
    to_torch_state_dict,
)
from nsmor.jax.train import JAXTrainState, build_optimizer, compute_bio_joint_loss, train_jax

__all__ = [
    "JAXExperimentConfig",
    "JAXRuntimeConfig",
    "load_config",
    "JAXDataLoader",
    "JAXDataset",
    "load_nsmor_dataset",
    "NSMoRModel",
    "SensoryEncoderJAX",
    "MoRRouterJAX",
    "DirectionHeadJAX",
    "load_from_torch_state_dict",
    "to_torch_state_dict",
    "JAXTrainState",
    "build_optimizer",
    "compute_bio_joint_loss",
    "train_jax",
]
