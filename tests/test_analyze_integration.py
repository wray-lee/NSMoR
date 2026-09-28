"""Tests for scripts/analyze_integration.py condition grouping and dataset loading."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION, FeatureConfig
from scripts.analyze_integration import (
    classify_wind_condition,
    group_trials_by_condition,
    load_dataset,
)


def _make_mock_dataset(tmp_path: Path, n_trials: int = 6) -> Path:
    """Create a synthetic dataset with known conditions."""
    feature_config = FeatureConfig()
    n_frames = 1000

    X_seqs = []
    Y_seqs = []
    conditions = []
    for i in range(n_trials):
        x = np.zeros((n_frames, 8), dtype=np.float32)
        # Columns 4:8 remain 0 as placeholders for _fill_priors
        y = np.zeros(n_frames, dtype=np.float32)

        if i % 3 == 0:
            # multisensory: visual looming + wind
            x[100:600, 0] = np.linspace(2.0, 90.0, 500)
            x[300:400, 1] = 1.0
            conditions.append("multisensory")
        elif i % 3 == 1:
            # visual_only: visual looming, no wind
            x[100:600, 0] = np.linspace(2.0, 90.0, 500)
            conditions.append("visual_only")
        else:
            # wind_only: wind, no visual
            x[300:400, 1] = 1.0
            conditions.append("wind_only")

        X_seqs.append(x)
        Y_seqs.append(y)

    ds_path = tmp_path / "nsmor_dataset.pt"
    torch.save(
        {
            "X_seqs": X_seqs,
            "Y_seqs": Y_seqs,
            "labels": np.zeros(n_trials, dtype=np.int64),
            "lengths": np.full(n_trials, n_frames, dtype=np.int64),
            "mcmc_priors": np.full((n_trials, 4), 0.25, dtype=np.float32),
            "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
            "session_ids": [f"0.500cricket_001_session_1" for _ in range(n_trials)],
            "stimulus_conditions": conditions,
            "feature_config": feature_config,
            "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        },
        ds_path,
    )
    return ds_path


def test_load_dataset_without_raw_dir(tmp_path: Path):
    """load_dataset must load conditions from dataset and NOT search data/raw."""
    ds_path = _make_mock_dataset(tmp_path, n_trials=6)
    # The parent of parent of ds_path has no "raw" directory
    loader, labels, lengths, X_seqs, trial_info_list = load_dataset(
        ds_path, batch_size=2, max_seq_len=500
    )
    assert len(trial_info_list) == 6
    # Each trial_info must reflect the stimulus conditions
    types = [info.get("type") or info.get("condition") for info in trial_info_list]
    assert types == [
        "multisensory",
        "visual_only",
        "wind_only",
        "multisensory",
        "visual_only",
        "wind_only",
    ]


def test_group_trials_by_condition_uses_stimulus_conditions(tmp_path: Path):
    """group_trials_by_condition must classify all conditions accurately."""
    ds_path = _make_mock_dataset(tmp_path, n_trials=6)
    loader, labels, lengths, X_seqs, trial_info_list = load_dataset(
        ds_path, batch_size=2, max_seq_len=500
    )
    groups = group_trials_by_condition(X_seqs, trial_info_list)
    assert "visual_only" in groups
    assert "wind_only" in groups
    assert groups["visual_only"] == [1, 4]
    assert groups["wind_only"] == [2, 5]

@pytest.mark.parametrize("dt_ms", [4.0, 10.0])
def test_phase_f_tonic_latency_gate_and_signed_peak_compatibility(tmp_path: Path, dt_ms: float):
    """Flat magnitude has vigor but no measurable timing; signed peaks remain timed."""
    import json

    from scripts.analyze_integration import (
        compute_condition_statistics, create_integration_figure,
        export_integration_summary, extract_predicted_metrics,
    )

    anchor = int(2000 / dt_ms)
    length = int(6000 / dt_ms)
    for level in (5.0, 0.0):
        stats = compute_condition_statistics(extract_predicted_metrics(
            [np.full(length, level, dtype=np.float32)], [0],
            dt_ms=dt_ms, anchor_frames=[anchor],
        ))
        assert stats["peak_velocity"] == {"mean": level, "sem": 0.0, "n": 1}
        assert stats["latency"] == {"mean": None, "sem": None, "n": 0}
        summary_path = tmp_path / f"{level}.json"
        figure_path = tmp_path / f"{level}.png"
        export_integration_summary({"visual_only": stats}, summary_path, dt_ms=dt_ms)
        summary = json.loads(summary_path.read_text(encoding="utf-8"),
                             parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        assert summary["status"] == "unavailable"
        assert summary["baseline_reference"] is None
        assert summary["conditions"]["visual_only"]["latency_to_peak_ms"] == stats["latency"]
        assert summary["conditions"]["visual_only"]["peak_velocity_cms"] == stats["peak_velocity"]
        with pytest.raises(ValueError, match="no measurable peak latencies"):
            create_integration_figure({"visual_only": stats}, figure_path)
        assert not figure_path.exists()

    pre = np.zeros(length, dtype=np.float32)
    post = pre.copy()
    pre[anchor - int(500 / dt_ms)] = -30.0
    post[anchor + int(500 / dt_ms)] = 25.0
    metrics = extract_predicted_metrics(
        [pre, post], [0, 1], dt_ms=dt_ms, anchor_frames=[anchor, anchor],
    )
    assert metrics == {"peak_velocities": [30.0, 25.0], "latencies": [-500.0, 500.0]}
    stats = compute_condition_statistics(metrics)
    assert stats["latency"] == {"mean": 0.0, "sem": 500.0, "n": 2}
    assert stats["peak_velocity"]["n"] == 2
    summary_path = tmp_path / "response.json"
    figure_path = tmp_path / "response.png"
    export_integration_summary({"visual_only": stats}, summary_path, dt_ms=dt_ms)
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "ok"
    create_integration_figure({"visual_only": stats}, figure_path)
    assert figure_path.stat().st_size > 0