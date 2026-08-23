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
    # Frames 150 and -200 are a real escape transient (two consecutive frames)
    # that the ±100 clip flattens to the boundary; they must STILL be counted as
    # escape (raw membership).  The isolated single 30-cm/s frame is below the
    # sustained-run guard (min 2 consecutive over-band frames) and is excluded,
    # matching the artifact/escape decoupling.
    y = np.array([0.0, 0.0, 30.0, 0.0, 150.0, -200.0, 5.0, 0.0])
    loader = _tiny_loader(y)
    m = compute_metrics(
        _FakeModel(scale=1.0), loader, torch.device("cpu"),
        target_mean=0.0, target_std=1.0, target_clip_cm_s=100.0,
        escape_band_cm_s=10.0,
    )
    # raw |y| >= 10 frames: {30(iso single→dropped), 150, -200(consecutive run)}
    # -> sustained escape frames = {150, -200} => 2
    assert int(m["n_escape_frames"]) == 2
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


def test_escape_above_clip_is_not_masked(compute_metrics):
    # CORE AUDIT FIX: a model that under-predicts a large escape must show a
    # LARGE escape_rmse, NOT ~0 (which the old clipped-space metric would mask).
    # y_true has a 150 cm/s escape (two consecutive frames so it survives the
    # sustained-run guard); target_clip_cm_s=100 would flatten each to the
    # boundary.  A model predicting a flat clip value (90) everywhere must
    # register a big raw error on that escape.
    y = np.array([0.0, 0.0, 0.0, 150.0, 150.0, 0.0, 0.0, 0.0])
    yt = torch.as_tensor(y, dtype=torch.float32).view(1, 8)
    # model predicts 90.0 everywhere (i.e. x carries 90 on its channel)
    x90 = torch.full((1, 8, 1), 90.0, dtype=torch.float32)
    lengths = torch.full((1,), 8, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(x90, yt, lengths)
    loader = torch.utils.data.DataLoader(ds, batch_size=1)

    m = compute_metrics(
        _FakeModel(scale=1.0), loader, torch.device("cpu"),
        target_mean=0.0, target_std=1.0, target_clip_cm_s=100.0,
        escape_band_cm_s=10.0,
    )
    # the 150 escape is above the 100 clip; a flat-90 model errs by |150-90|=60
    assert int(m["n_escape_frames"]) == 2
    assert m["escape_rmse"] > 50.0, f"escape above clip was masked: {m['escape_rmse']:.2f}"


def test_isolated_artifact_spike_excluded_from_escape(compute_metrics):
    # SUSTAINED-MEMBERSHIP GUARD: a single isolated ~1e7 cm/s tracking-artifact
    # frame (which the training-target clip removes, but which would otherwise
    # land in the raw escape band) must NOT count as escape.  This decouples the
    # audit from the very artifact the clip suppresses — otherwise escape_rmse
    # would be mechanically inflated to O(1e7) by clip-handled artifacts, and
    # "escape_rmse >> resting_rmse" would be a clip artifact, not evidence.
    y = np.array([0.0, 0.0, 1.3e7, 0.0, 0.0, 0.0, 0.0, 0.0])  # one isolated spike
    loader = _tiny_loader(y)
    m = compute_metrics(
        _FakeModel(scale=1.0), loader, torch.device("cpu"),
        target_mean=0.0, target_std=1.0, target_clip_cm_s=100.0,
        escape_band_cm_s=10.0,
    )
    assert int(m["n_escape_frames"]) == 0, (
        "isolated artifact spike should be excluded by the sustained-run guard"
    )
    # and a real sustained escape must still be caught
    y2 = np.array([0.0, 0.0, 150.0, 160.0, 0.0, 0.0, 0.0, 0.0])
    m2 = compute_metrics(
        _FakeModel(scale=1.0), _tiny_loader(y2), torch.device("cpu"),
        target_mean=0.0, target_std=1.0, target_clip_cm_s=100.0,
        escape_band_cm_s=10.0,
    )
    assert int(m2["n_escape_frames"]) == 2


def test_sustained_run_helper():
    mod = _load_train_module()
    over = np.array([False, False, True, False, True, True, False, False])
    assert mod._sustained_run(over, min_run=2).tolist() == \
        [False, False, False, False, True, True, False, False]
    # isolated single frame dropped
    assert mod._sustained_run(np.array([False, True, False]), min_run=2).tolist() == \
        [False, False, False]
    # trailing run reaching the end kept
    assert mod._sustained_run(np.array([True, True, False, True]), min_run=2).tolist() == \
        [True, True, False, False]
    # min_run=1 is identity
    assert mod._sustained_run(np.array([False, True, False]), min_run=1).tolist() == \
        [False, True, False]


def test_escape_runs_do_not_span_sequence_boundaries(compute_metrics):
    # ROUND-3 BLOCKER regression: the sustained-run guard must be applied
    # PER SEQUENCE.  Here seq1 ends with a single over-band frame (150) and
    # seq2 starts with one (-150); concatenated naively they form a spurious
    # 2-frame "run" across the trial boundary and would be miscounted as
    # escape.  Per-sequence masking keeps n_escape_frames == 0.
    y_seq1 = np.array([0.0, 0.0, 150.0])          # trailing isolated spike
    y_seq2 = np.array([-150.0, 0.0, 0.0])         # leading isolated spike
    xs, ys, ls = [], [], []
    for y in (y_seq1, y_seq2):
        yt = torch.as_tensor(y, dtype=torch.float32)
        xs.append(yt.unsqueeze(-1))               # (L, 1)
        ys.append(yt)                             # (L,)
        ls.append(len(y))
    x = torch.stack(xs)                           # (B=2, L=3, 1)
    y = torch.stack(ys)                           # (B=2, 3)
    lengths = torch.tensor(ls, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(x, y, lengths)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)

    m = compute_metrics(
        _FakeModel(scale=1.0), loader, torch.device("cpu"),
        target_mean=0.0, target_std=1.0, target_clip_cm_s=100.0,
        escape_band_cm_s=10.0,
    )
    assert int(m["n_escape_frames"]) == 0, (
        "cross-sequence run bridged two trials: "
        f"n_escape_frames={m['n_escape_frames']}"
    )


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


def test_lr_warmup_scale_linear_and_idempotent():
    # Regression (round-2 review): LR warmup must ramp linearly from 0→1 over
    # the warmup window and return to 1 past it, honouring the backward-compat
    # default (lr_warmup_epochs=0 ⇒ identity scale, no drift).
    mod = _load_train_module()
    compute = mod.compute_lr_warmup_scale
    assert compute(0, 4) == pytest.approx(0.25)
    assert compute(1, 4) == pytest.approx(0.5)
    assert compute(2, 4) == pytest.approx(0.75)
    assert compute(3, 4) == pytest.approx(1.0)
    assert compute(5, 4) == 1.0          # past window → full LR
    assert compute(0, 0) == 1.0          # warmup disabled → identity (backward compat)


def test_apply_lr_warmup_requires_base_lr_groups():
    # Regression (round-2 review): a param group WITHOUT ``base_lr`` (the
    # shape the phase-1 frontend optimizer formerly used) must raise a clear
    # error — guarding the two-phase × warmup combination — while a group that
    # DOES carry ``base_lr`` is scaled without error.
    mod = _load_train_module()
    import torch as _t
    p = _t.nn.Parameter(_t.zeros(2))

    # single group missing base_lr → ValueError (this was the crash path)
    bad = _t.optim.AdamW([{"params": [p], "lr": 1e-3}])
    try:
        mod.apply_lr_warmup(bad, epoch=0, lr_warmup_epochs=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for group without base_lr")

    # group carrying base_lr → LR correctly ramped, no error
    good = _t.optim.AdamW([{"params": [p], "lr": 1e-3, "base_lr": 1e-3}])
    mod.apply_lr_warmup(good, epoch=0, lr_warmup_epochs=4)
    assert good.param_groups[0]["lr"] == pytest.approx(2.5e-4)  # base_lr * 0.25


def test_scheduler_released_only_after_warmup():
    # Regression (round-2 review): with warmup active the cosine step is held
    # until the warmup window has fully elapsed (budget not consumed by the
    # warmup override); with warmup disabled the step is unconditional.
    mod = _load_train_module()

    class _Sched:
        def __init__(self):
            self.steps = 0
        def step(self):
            self.steps += 1

    s = _Sched()
    mod._maybe_step_scheduler(
        scheduler=s,
        lr_warmup_epochs=4,
        warmup_epoch=3,          # last warmup epoch → NOT yet released
    )
    assert s.steps == 0
    mod._maybe_step_scheduler(
        scheduler=s,
        lr_warmup_epochs=4,
        warmup_epoch=4,          # window fully elapsed → released
    )
    assert s.steps == 1
    mod._maybe_step_scheduler(
        scheduler=s,
        lr_warmup_epochs=0,      # disabled → unconditional
        warmup_epoch=0,
    )
    assert s.steps == 2