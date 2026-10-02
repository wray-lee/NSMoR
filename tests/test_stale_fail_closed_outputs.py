"""Stale artefacts must not survive a fail-closed branch (T3).

A run that cannot produce a figure/summary must remove the previous
run's file at the same path, otherwise a stale artefact masquerades as
this run's output while the fresh status JSON says otherwise.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch


def test_stale_integration_figure_removed_before_fail_closed(tmp_path) -> None:
    from scripts.analyze_integration import (
        compute_condition_statistics,
        create_integration_figure,
        extract_predicted_metrics,
    )

    figure = tmp_path / "integration.png"
    figure.write_bytes(b"\x89PNG stale")
    stats = compute_condition_statistics(extract_predicted_metrics(
        [np.zeros(100, dtype=np.float32)], [0], dt_ms=4.0, anchor_frames=[50],
    ))
    with pytest.raises(ValueError, match="no measurable peak latencies"):
        create_integration_figure({"visual_only": stats}, figure)
    assert not figure.exists()


def test_stale_psychophysics_outputs_removed_on_not_applicable(
        monkeypatch, tmp_path) -> None:
    from scripts import simulate_psychophysics as phase_g

    class UnusedModel:
        dt_ms = 4.0

        def __call__(self, X, lengths, return_internals=False):
            raise AssertionError("not_applicable must return before inference")

    X = torch.zeros(2, 12, 8)
    X[:, :, 0] = 20.0
    lengths = torch.full((2,), 12, dtype=torch.int64)
    meta = phase_g.TrialMeta(
        val_indices=[0, 1], target_ttc_ms=[-225.0, -261.0],
        stimulus_conditions=["multisensory", "multisensory"],
        anchor_frames=[2, 2], stim_onset_frame=2, pre_anchor_frames=2,
    )
    data = phase_g.ValidationData(X, torch.zeros(2, 12), lengths, meta)
    monkeypatch.setattr(phase_g, "load_checkpoint", lambda *_a: UnusedModel())
    monkeypatch.setattr(phase_g, "load_validation_data", lambda *_a, **_k: data)

    out = tmp_path / "results"
    out.mkdir()
    (out / "bayesian_reliability.png").write_bytes(b"\x89PNG stale")
    (out / "psychophysics_summary.json").write_text('{"stale": true}', encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "simulate_psychophysics.py", "--dataset", "synthetic.pt",
        "--output_dir", str(out),
    ])
    phase_g.main()

    assert not (out / "bayesian_reliability.png").exists()
    assert not (out / "psychophysics_summary.json").exists()
    status = json.loads((out / "bayesian_reliability.json").read_text(encoding="utf-8"))
    assert status["status"] == "not_applicable" and status["n_ttc0"] == 0
