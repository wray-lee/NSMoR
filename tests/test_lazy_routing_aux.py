"""Tests for lazy loading routing aux support."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION, FeatureConfig
from nsmor.config_parser import ExperimentConfig
from scripts.train import build_dataloaders


def test_lazy_dataloader_attaches_routing_aux_mask(tmp_path: Path):
    """When lambda_routing_aux > 0, lazy dataloader must yield 4-tuples with wind_only_mask."""
    metadata_path = tmp_path / "metadata.pt"
    n_trials = 4
    # 2 animals, 2 sessions each
    session_ids = [
        "0.500cricket_001_session_1",
        "0.500cricket_001_session_2",
        "0.501cricket_002_session_1",
        "0.501cricket_002_session_2",
    ]

    sess_dir = tmp_path / "session_dummy"
    sess_dir.mkdir(parents=True, exist_ok=True)
    kin_file = sess_dir / "kinematics.csv"
    evt_file = sess_dir / "events.csv"

    import pandas as pd
    kin_rows = []
    evt_rows = []
    for s_id in session_ids:
        for t in range(100):
            kin_rows.append({
                "time_ms": t * 10.0,
                "x_pos": 0.0,
                "y_pos": 0.0,
                "heading": 0.0,
                "velocity": 0.0,
                "acceleration": 0.0,
                "visual_angle": 0.0,
                "wind_state": 0.0,
                "l_v_ratio": 0.0,
                "session_id": s_id,
                "trial_id": 0,
            })
        evt_rows.append({
            "session_id": s_id,
            "trial_id": 0,
            "time_ms": 0.0,
            "event_type": "stimulus_onset",
            "event_value": "",
        })
    kin_df = pd.DataFrame(kin_rows)
    kin_df.to_csv(kin_file, index=False)

    evt_df = pd.DataFrame(evt_rows)
    evt_df.to_csv(evt_file, index=False)

    trial_specs = []
    for i in range(n_trials):
        trial_specs.append({
            "session_id": session_ids[i],
            "session_dir": str(sess_dir),
            "kinematics_file": kin_file.name,
            "events_file": evt_file.name,
            "trial_id": 0,
            "n_frames": 100,
            "trial_start_ms": 0.0,
            "stimulus_onset_ms": 0.0,
            "anchor_ms": 0.0,
            "anchor_frame": 0,
            "anchor_rule": "stimulus_onset",
            "label": "ESCAPE",
            "is_pure_wind": (i % 2 == 0),
        })

    mcmc_priors = torch.full((n_trials, 4), 0.25, dtype=torch.float32)
    torch.save(
        {
            "trial_specs": trial_specs,
            "mcmc_priors": mcmc_priors,
            "session_ids": session_ids,
            "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
            "n_trials": n_trials,
            "label_encoder": {"ESCAPE": 0},
            "feature_config": FeatureConfig(),
            "snapshot_anchor_rules": ["stimulus_onset"] * n_trials,
            "n_sessions": 4,
        },
        metadata_path,
    )

    config = ExperimentConfig()
    config.training.batch_size = 2
    config.loss.lambda_routing_aux = 0.1

    train_loader, val_loader = build_dataloaders(
        config,
        dataset_path=str(metadata_path),
        val_split=0.5,
        use_lazy_loading=True,
    )

    assert hasattr(train_loader.dataset, "is_pure_wind")
    assert train_loader.dataset.is_pure_wind is not None

    batch = next(iter(train_loader))
    # Must be 4-tuple (X, Y, lengths, wind_only_mask)
    assert len(batch) == 4
    x_b, y_b, lengths, mask = batch
    assert mask.dtype == torch.bool
    assert mask.shape == (2,)
