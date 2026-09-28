"""Tests for lazy loading routing aux support."""
from __future__ import annotations

from pathlib import Path
import hashlib
import pickle
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
    # 2 recording prefixes, 2 sessions each
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

    kin_digest = hashlib.sha256(kin_file.read_bytes()).hexdigest()
    evt_digest = hashlib.sha256(evt_file.read_bytes()).hexdigest()

    trial_specs = []
    for i in range(n_trials):
        trial_specs.append({
            "session_id": session_ids[i],
            "session_dir": str(sess_dir),
            "kinematics_file": kin_file.name,
            "events_file": evt_file.name,
            "kinematics_sha256": kin_digest,
            "events_sha256": evt_digest,
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
            "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
            "animal_identity_status": "unverified",
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


def test_lazy_constructor_rejects_generated_metadata_reducer(tmp_path: Path) -> None:
    """Generated stage02 metadata must be restricted before constructor execution."""
    from nsmor.lazy_dataloader import NSMoRLazyDataset

    metadata = tmp_path / "metadata.pt"
    marker = tmp_path / "reducer-executed"

    class Unexpected:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",)

    torch.save({
        "trial_specs": [{"session_id": "recording_session_1"}],
        "mcmc_priors": torch.full((1, 4), 0.25),
        "unexpected": Unexpected(),
    }, metadata)
    try:
        with pytest.raises((ValueError, pickle.UnpicklingError), match="global"):
            NSMoRLazyDataset(str(metadata))
    finally:
        assert not marker.exists(), "lazy constructor executed a generated reducer"


@pytest.mark.parametrize("dt_ms", [None, 4.006, 10.0])
def test_lazy_constructor_preserves_generated_metadata_shapes_and_cadence(
    tmp_path: Path, dt_ms: float | None,
) -> None:
    from nsmor.config import TimeWindowConfig
    from nsmor.lazy_dataloader import NSMoRLazyDataset

    metadata = tmp_path / "metadata.pt"
    specs = [
        {"session_id": "recordingA_session_1", "label": "ESCAPE"},
        {"session_id": "recordingB_session_1", "label": "NO_RESPONSE"},
    ]
    priors = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    assert priors.shape == (2, 4)
    features = FeatureConfig()
    torch.save({
        "trial_specs": specs, "mcmc_priors": priors,
        "feature_config": features,
        "time_window_config": TimeWindowConfig(frame_interval_ms=4.006),
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
            "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
            "animal_identity_status": "unverified",
        "dt_ms": 4.006,
    }, metadata)
    dataset = NSMoRLazyDataset(str(metadata), max_seq_len=32, pre_anchor_frames=8,
                              feature_config=features, dt_ms=dt_ms)
    assert len(dataset) == 2 and dataset.trial_specs == specs
    assert dataset.get_session_id(0) == "recordingA_session_1"
    assert dataset.get_label(1) == "NO_RESPONSE"
    assert dataset.mcmc_priors.shape == (2, 4)
    assert dataset.mcmc_priors.dtype == torch.float64
    assert torch.equal(dataset.mcmc_priors, priors)
    assert dataset.max_seq_len == 32 and dataset.pre_anchor_frames == 8
    assert dataset.feature_config == features
    assert dataset.dt_ms is dt_ms
