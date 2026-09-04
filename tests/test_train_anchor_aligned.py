"""Tests for anchor-aligned dataloader construction in train.py."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION, FeatureConfig
from nsmor.config_parser import ExperimentConfig
from scripts.train import build_dataloaders


def test_eager_dataloader_wires_anchor_frames(tmp_path: Path):
    """Eager dataloaders must receive anchor_frames for anchor-aligned cropping."""
    dataset_path = tmp_path / "dataset.pt"
    n_trials = 4
    session_ids = [
        "0.500cricket_001_session_1",
        "0.500cricket_001_session_2",
        "0.501cricket_002_session_1",
        "0.501cricket_002_session_2",
    ]
    X_seqs = [np.zeros((500, 8), dtype=np.float32) for _ in range(n_trials)]
    # Put a wind pulse at frame 200 in trial 0
    X_seqs[0][200:250, 1] = 1.0
    # Put a looming ramp peaking at frame 300 in trial 1
    X_seqs[1][100:301, 0] = np.linspace(0.0, 90.0, 201)

    Y_seqs = [np.zeros(500, dtype=np.float32) for _ in range(n_trials)]
    mcmc_priors = np.full((n_trials, 4), 0.25, dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    lengths = np.array([500] * n_trials, dtype=np.int64)

    torch.save(
        {
            "X_seqs": X_seqs,
            "Y_seqs": Y_seqs,
            "mcmc_priors": mcmc_priors,
            "labels": labels,
            "lengths": lengths,
            "session_ids": np.array(session_ids, dtype=object),
            "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
            "mcmc_prior_provenance": "oof_2fold_animal_grouped_cv",
            "feature_config": FeatureConfig(),
        },
        dataset_path,
    )

    config = ExperimentConfig()
    config.training.batch_size = 2
    config.training.max_seq_len = 300

    train_loader, val_loader = build_dataloaders(
        config,
        dataset_path=str(dataset_path),
        val_split=0.5,
        use_lazy_loading=False,
    )

    assert train_loader.dataset.anchor_frames is not None
    assert len(train_loader.dataset.anchor_frames) == len(train_loader.dataset)
    assert val_loader.dataset.anchor_frames is not None
    assert len(val_loader.dataset.anchor_frames) == len(val_loader.dataset)
