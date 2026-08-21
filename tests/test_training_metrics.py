"""Regression tests for the training-stability escape-signal metrics.

Exercises ``scripts/train.compute_metrics`` (the best-model evaluation path)
directly, with a lightweight stand-in model and a tiny DataLoader, so that:

1. The function returns a dict containing all nine documented keys
   (mse/rmse/mae/r2 + escape_band_cm_s/n_escape_frames/escape_rmse/
   resting_rmse/escape_ratio) — regressing the historical NameError where
   ``metrics["..."]`` was written against an uninitialised dict.
2. Escape membership is classified on the *unclipped* ground truth, so a
   real escape transient flattened to the ``±target_clip_cm_s`` boundary is
   still counted as escape, while the reported RMSE lives in clipped space
   (a documented, bounded proxy).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_mod", _SCRIPTS / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeModel:
    """Minimal stand-in for NSMoRCore: eval() + return_internals forward."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self._evals = 0

    def eval(self):
        self._evals += 1
        return self

    def __call__(self, x: torch.Tensor, lengths, return_internals=False):
        # Real NSMoRCore: DirectionHead squeezes the final channel dim, so
        # y_pred is (B, T).  x is (B, T, feat); reduce over the channel dim.
        internals = {"routing_gates": torch.zeros(x.size(0), x.size(1), 2),
                     "lif_spikes": torch.zeros(x.size(0), x.size(1), 2)}
        pred = x.mean(dim=-1) * self.scale       # (B, T)
        return pred, internals


def _tiny_loader(y_values: np.ndarray) -> torch.utils.data.DataLoader:
    # Single sequence of length L: x is (1, L, 1) with the target on its sole
    # channel, y is (1, L) (matching the real collate's (B, T) target).
    y_arr = np.asarray(y_values, dtype=np.float32)
    L = y_arr.size
    y = torch.as_tensor(y_arr).view(1, L)          # (B=1, T=L)
    x = y.unsqueeze(-1).clone()                    # (B=1, T=L, feat=1)
    lengths = torch.full((1,), L, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(x, y, lengths)
    return torch.utils.data.DataLoader(ds, batch_size=1)


@pytest.fixture(scope="module")
def compute_metrics():
    return _load_train_module().compute_metrics


def test_compute_metrics_returns_nine_key_dict(compute_metrics):
    y = np.array([0.0, 0.0, 30.0, 0.0, 150.0, -200.0, 5.0, 0.0])
    loader = _tiny_loader(y)
    m = compute_metrics(
        _FakeModel(scale=1.0), loader, torch.device("cpu"),
        target_mean=0.0, target_std=1.0, target_clip_cm_s=100.0,
        escape_band_cm_s=10.0,
    )
    expect = {"mse", "rmse", "mae", "r2", "escape_band_cm_s",
              "n_escape_frames", "escape_rmse", "resting_rmse", "escape_ratio"}
    assert set(m.keys()) == expect, f"missing/extra keys: {set(m.keys()) ^ expect}"


def test_escape_classified_on_unclipped_y_true(compute_metrics):
    # Frames 150 and -200 real escape transients that the ±100 clip flattens
    # to the boundary; they must STILL be counted as escape (raw membership).
    y = np.array([0.0, 0.0, 30.0, 0.0, 150.0, -200.0, 5.0, 0.0])
    loader = _tiny_loader(y)
    m = compute_metrics(
        _FakeModel(scale=1.0), loader, torch.device("cpu"),
        target_mean=0.0, target_std=1.0, target_clip_cm_s=100.0,
        escape_band_cm_s=10.0,
    )
    # raw |y| >= 10 frames: {30, 150, -200} -> 3
    assert int(m["n_escape_frames"]) == 3
    # perfect prediction (scale=1.0) on clipped values -> near-zero escape_rmse
    assert m["escape_rmse"] < 1e-3


def test_normalized_rescale_then_clip_round_trips(compute_metrics):
    # When normalization (target_mean/std) was used at train time, the model
    # emits PREDICTIONS IN NORMALIZED SPACE; compute_metrics rescales them
    # back to cm/s (pred * std + mean) before comparing against the raw cm/s
    # target.  A perfect normalized-space predictor (pred == (y-mean)/std)
    # must therefore recover rmse ≈ 0 in physical units after that rescale.
    y = np.array([0.0, 0.0, 10.0, 0.0, 60.0, -40.0, 0.0, 0.0])
    normalize = lambda v: (v - 5.0) / 2.0   # (y - mean)/std  (mean=5, std=2)
    y_norm = normalize(y)

    # x carries the NORMALIZED target on its channel dim; the fake head's
    # mean(dim=-1) reproduces it exactly.
    yt = torch.as_tensor(y_norm, dtype=torch.float32).view(1, 8)
    x = yt.unsqueeze(-1).clone()             # (1, 8, 1)
    y_raw = torch.as_tensor(y, dtype=torch.float32).view(1, 8)
    lengths = torch.full((1,), 8, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(x, y_raw, lengths)
    loader = torch.utils.data.DataLoader(ds, batch_size=1)

    m = compute_metrics(
        _FakeModel(scale=1.0), loader, torch.device("cpu"),
        target_mean=5.0, target_std=2.0, target_clip_cm_s=100.0,
        escape_band_cm_s=10.0,
    )
    # after *2+5 the normalized predictions match raw y exactly -> rmse ~ 0
    assert m["rmse"] < 1e-3


def test_zero_escape_frames_handled_without_error(compute_metrics):
    # All-resting (no |y|>=10) -> escape_rmse must be NaN (documented) and
    # the dict still returns all nine keys (resting_rmse present).
    y = np.zeros(8)
    loader = _tiny_loader(y)
    m = compute_metrics(
        _FakeModel(scale=1.0), loader, torch.device("cpu"),
        target_mean=0.0, target_std=1.0, target_clip_cm_s=100.0,
        escape_band_cm_s=10.0,
    )
    assert int(m["n_escape_frames"]) == 0
    assert np.isnan(m["escape_rmse"])
    assert 0.0 <= m["escape_ratio"] <= 1.0
    assert "resting_rmse" in m