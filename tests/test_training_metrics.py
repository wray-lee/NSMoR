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


def _tiny_metadata_loader(y_values: np.ndarray) -> torch.utils.data.DataLoader:
    """Return a four-item batch like ``collate_with_metadata``."""
    y_arr = np.asarray(y_values, dtype=np.float32)
    L = y_arr.size
    y = torch.as_tensor(y_arr).view(1, L)
    x = y.unsqueeze(-1).clone()
    lengths = torch.full((1,), L, dtype=torch.long)
    is_pure_wind = torch.tensor([True], dtype=torch.bool)
    ds = torch.utils.data.TensorDataset(x, y, lengths, is_pure_wind)
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


def test_compute_metrics_accepts_metadata_batch(compute_metrics):
    """Best-model scoring must accept the metadata DataLoader contract."""
    y = np.array([0.0, 0.0, 20.0, 30.0])
    m = compute_metrics(
        _FakeModel(scale=1.0), _tiny_metadata_loader(y), torch.device("cpu"),
        target_mean=0.0, target_std=1.0, target_clip_cm_s=0.0,
        escape_band_cm_s=10.0,
    )

    assert m["mse"] == pytest.approx(0.0)
    assert m["n_escape_frames"] == 2.0


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


def test_sweep_escape_sensitivity():
    # Regression (round-5 review): the band x min_run sensitivity sweep must
    # (a) apply _sustained_run PER SEQUENCE (no cross-trial run bridging),
    # (b) count events as contiguous kept runs per sequence, and (c) return
    # NaN escape_rmse when a config admits no frames.
    mod = _load_train_module()
    t1 = np.array([0.0, 30.0, 40.0, 0.0])       # sustained 2-frame run @30-40
    t2 = np.array([150.0, 0.0, 0.0, 1e7])       # tail-run + isolated artifact spike
    pred = [t1.copy(), t2.copy()]               # perfect predictions

    rows = mod.sweep_escape_sensitivity(
        [t1, t2], pred, bands_cm_s=[10.0], min_runs=(1, 2),
    )
    by = {(r["band_cm_s"], r["min_run"]): r for r in rows}
    assert set(by.keys()) == {(10.0, 1), (10.0, 2)}

    r1 = by[(10.0, 1)]
    # min_run=1: {30,40} + {150} + artifact 1e7 all admitted; runs do NOT
    # bridge the sequence boundary ({40},{150} are separate events).
    assert r1["n_escape_frames"] == 4
    assert r1["n_escape_events"] == 3
    assert r1["escape_rmse"] == pytest.approx(0.0)

    r2 = by[(10.0, 2)]
    # min_run=2: artifact spike and lone 150 dropped; only {30,40} survives.
    assert r2["n_escape_frames"] == 2
    assert r2["n_escape_events"] == 1
    assert r2["escape_ratio"] == pytest.approx(2 / 8)

    # No-frame config: band above every value -> NaN rmse, zero counts.
    rows_hi = mod.sweep_escape_sensitivity(
        [t1], [t1.copy()], bands_cm_s=[1e9], min_runs=(2,),
    )
    assert rows_hi[0]["n_escape_frames"] == 0
    assert np.isnan(rows_hi[0]["escape_rmse"])



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


# ═══════════════════════════════════════════════════════════════
# Resume x two-phase regression (round-6 review blocker)
# ═══════════════════════════════════════════════════════════════

def _make_synthetic_dataset(path: Path, n_seqs: int = 6, T: int = 12) -> None:
    """Write a minimal ``nsmor_dataset.pt``-shaped file."""
    from nsmor.config import PIPELINE_SEMANTICS_VERSION
    rng = np.random.RandomState(0)
    X_seqs = [rng.randn(T, 8).astype(np.float32) for _ in range(n_seqs)]
    Y_seqs = [rng.randn(T).astype(np.float32) * 5 for _ in range(n_seqs)]
    priors = np.abs(rng.randn(n_seqs, 4).astype(np.float32)) + 0.1
    priors /= priors.sum(axis=1, keepdims=True)
    dataset = {
        "X_seqs": X_seqs,
        "Y_seqs": Y_seqs,
        "mcmc_priors": priors,
        "labels": np.zeros(n_seqs, dtype=np.int64),
        "lengths": np.full(n_seqs, T, dtype=np.int64),
        # Round-3 provenance guard: loaders reject unstamped artifacts.
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_5fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
        "session_ids": [f"recording{i}_session_1" for i in range(n_seqs)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, path)


def _nested_prefix_metadata(session_ids, train_idx, val_idx):
    from nsmor.pipeline.grouping import animal_keys_of

    prefixes = animal_keys_of(session_ids)
    train_prefixes = sorted(set(prefixes[train_idx].tolist()))
    val_prefixes = sorted(set(prefixes[val_idx].tolist()))
    return {
        "recording_prefix_keys": prefixes,
        "train_recording_prefixes": train_prefixes,
        "val_recording_prefixes": val_prefixes,
        "n_train_recording_prefixes": len(train_prefixes),
        "n_val_recording_prefixes": len(val_prefixes),
        "n_inner_folds": 5,
        "animal_identity_status": "unverified",
    }


def _tiny_config(mod, output_dir):
    cfg = mod.ExperimentConfig()
    cfg.model.hidden_dim = 8
    cfg.model.num_gru_layers = 1
    cfg.training.num_epochs = 2
    cfg.training.batch_size = 2
    cfg.training.checkpoint_interval = 1
    cfg.training.log_interval = 1
    cfg.training.lr_warmup_epochs = 0
    cfg.checkpoint.output_dir = str(output_dir)
    return cfg


def test_post_training_sweep_accepts_metadata_and_legacy_batches(tmp_path, monkeypatch):
    """Export a 12-cell sweep from best Phase 2 weights after one tiny epoch."""
    import csv
    import json
    from nsmor.nsmor_dataloader import collate_with_metadata
    from nsmor.pipeline.grouping import grouped_train_val_split

    mod = _load_train_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod, "_SWEEP_BANDS", None)
    monkeypatch.setattr(mod, "_VAL_SPLIT", 0.5)
    ds_path = tmp_path / "synthetic.pt"
    _make_synthetic_dataset(ds_path)
    dataset = torch.load(ds_path, weights_only=False)
    dataset["session_ids"] = [f"animal{i}_session_1" for i in range(6)]
    for i in range(6):
        n = 6 + i
        dataset["X_seqs"][i] = dataset["X_seqs"][i][:n]
        dataset["Y_seqs"][i] = np.array([0, 0, 60, 60, 60] + [0] * (n - 5), dtype=np.float32)
        dataset["lengths"][i] = n
        assert dataset["X_seqs"][i].shape == (n, 8)
        assert dataset["Y_seqs"][i].shape == (n,)
    torch.save(dataset, ds_path)
    _, val_idx = grouped_train_val_split(dataset["session_ids"], 6, val_split=0.5,
                                        random_seed=42)
    n_escape = 3 * len(val_idx)  # One three-frame raw 60 cm/s run per trial.
    escape_ratio = n_escape / sum(int(dataset["lengths"][i]) for i in val_idx)

    real_build_model = mod.build_model
    observations = []

    def observed_model(config):
        model = real_build_model(config)

        def observe_forward(module, inputs, output):
            best_path = Path(config.checkpoint.output_dir) / "best_model.pth"
            if not best_path.exists():  # Initial training/validation precede selection.
                return
            pred, _ = output
            assert pred.shape == inputs[0].shape[:2]
            assert not module.training and not torch.is_grad_enabled()
            assert not pred.requires_grad
            assert any(p.requires_grad for p in module.backend.parameters())
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            for name, param in module.named_parameters():
                assert torch.equal(param.detach().cpu(), best["model_state_dict"][name])
            observations.append(True)

        model.register_forward_hook(observe_forward)
        return model

    monkeypatch.setattr(mod, "build_model", observed_model)
    real_build_dataloaders = mod.build_dataloaders

    def validation_loader(*args, **kwargs):
        train_loader, val_loader = real_build_dataloaders(*args, **kwargs)
        if width == 3:
            # Existing collate adapter: same padded data, no condition mask.
            val_loader.collate_fn = collate_with_metadata
        batch = next(iter(val_loader))
        assert isinstance(batch, tuple) and len(batch) == width
        assert batch[0].shape[:2] == batch[1].shape
        assert batch[2].shape == (batch[0].size(0),)
        if width == 4:
            assert batch[3].dtype == torch.bool and batch[3].shape == batch[2].shape
        return train_loader, val_loader

    bands = [5.0, 10.0, 20.0, 50.0]
    monkeypatch.setattr(mod, "_SWEEP_BANDS", bands)
    exports, headlines = [], []
    for width in (4, 3):
        with monkeypatch.context() as patch:
            patch.setattr(mod, "build_dataloaders", validation_loader)
            output_dir = tmp_path / f"batch{width}"
            cfg = _tiny_config(mod, output_dir)
            cfg.training.num_epochs = 1
            cfg.training.random_seed = 42
            cfg.training.num_workers = 0
            cfg.training.normalize_targets = True
            cfg.training.target_clip_cm_s = 100.0
            result = mod.train(cfg, phase1_epochs=0, dataset_path=str(ds_path))

        assert len(result["history"]["train_loss"]) == len(result["history"]["val_loss"]) == 1
        assert result["eval_provenance"] == "best"
        final = torch.load(output_dir / "final_model.pth", map_location="cpu", weights_only=False)
        metrics = json.loads((output_dir / "metrics.json").read_text())
        best = torch.load(output_dir / "best_model.pth", map_location="cpu", weights_only=False)
        periodic = torch.load(output_dir / "epoch_1.pth", map_location="cpu", weights_only=False)
        for record in (best, periodic, final, metrics, result, result["metrics"]):
            assert record["is_nested_cv"] is False
            assert record["nested_prior_artifact_sha256"] == ""
            assert record["validation_scope"] == "diagnostic_global_oof"
        with (output_dir / "escape_sensitivity.csv").open(newline="") as f:
            rows = [{key: float(value) for key, value in row.items()}
                    for row in csv.DictReader(f)]
        assert len(rows) == 12
        assert {(row["band_cm_s"], row["min_run"]) for row in rows} == {
            (band, run) for band in bands for run in (1, 2, 3)
        }
        assert all(np.isfinite(list(row.values())).all() for row in rows)
        for row in rows:
            assert row["n_escape_frames"] == n_escape
            assert row["n_escape_events"] == len(val_idx)
            assert row["escape_ratio"] == pytest.approx(escape_ratio)
        exports.append(rows)
        headlines.append(metrics)
    assert observations
    assert exports[0] == exports[1]
    assert headlines[0] == headlines[1]


def test_nested_prior_normalization_uses_loaded_split_after_valid_same_path_swap(tmp_path, monkeypatch):
    """A held-out animal in sidecar A cannot enter train target statistics via B."""
    import hashlib
    from nsmor.pipeline.grouping import grouped_train_val_split
    from nsmor.pipeline.nested_prior import load_nested_prior_split

    mod = _load_train_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ds_path = tmp_path / "synthetic.pt"
    _make_synthetic_dataset(ds_path)
    dataset = torch.load(ds_path, weights_only=False)
    dataset["session_ids"] = [f"animal{i}_session_1" for i in range(6)]
    dataset["Y_seqs"] = [np.full(12, value, dtype=np.float32)
                         for value in (90.0, 1.0, 2.0, 3.0, 4.0, 5.0)]
    torch.save(dataset, ds_path)
    source_sha = hashlib.sha256(ds_path.read_bytes()).hexdigest()
    priors = np.tile([0.4, 0.3, 0.2, 0.1], (6, 1))
    artifact_path = tmp_path / "nested.pt"
    artifacts = []
    for seed in (42, 43):
        train_idx, val_idx = grouped_train_val_split(
            dataset["session_ids"], 6, val_split=0.2, random_seed=seed,
        )
        sidecar = {
            "nested_priors": priors, "train_priors": priors[train_idx],
            "val_priors": priors[val_idx], "train_indices": train_idx,
            "val_indices": val_idx, "is_nested_cv": True,
            "source_fingerprint": source_sha,
            "pipeline_semantics_version": dataset["pipeline_semantics_version"],
            "split_seed": seed, "val_split": 0.2,
            "mcmc_prior_provenance": f"nested_outer_seed{seed}_inner_5fold_recording_prefix_grouped",
            **_nested_prefix_metadata(dataset["session_ids"], train_idx, val_idx),
        }
        path = artifact_path if seed == 42 else tmp_path / "replacement.pt"
        torch.save(sidecar, path)
        loaded_train, loaded_val, _, info = load_nested_prior_split(
            path, ds_path, 6, dataset["session_ids"],
            feature_config=dataset.get("feature_config", mod.DEFAULT_FEATURE),
        )
        np.testing.assert_array_equal(loaded_train, train_idx)
        np.testing.assert_array_equal(loaded_val, val_idx)
        assert info["nested_prior_artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts.append((train_idx, path))

    np.testing.assert_array_equal(artifacts[0][0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(artifacts[1][0], [0, 1, 2, 3, 4])
    digest_a = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    bytes_b = artifacts[1][1].read_bytes()
    assert hashlib.sha256(bytes_b).hexdigest() != digest_a

    cfg = _tiny_config(mod, tmp_path / "run")
    cfg.training.num_epochs = 1
    cfg.training.num_workers = 0
    cfg.training.normalize_targets = True
    cfg.training.target_clip_cm_s = 100.0
    mean_a, _, idx_a = mod.compute_target_stats(
        str(ds_path), cfg, nested_prior_artifact=str(artifact_path),
    )
    np.testing.assert_array_equal(idx_a, artifacts[0][0])
    assert mean_a == pytest.approx(3.0)
    build = mod.build_dataloaders

    def swap_after_load(*args, **kwargs):
        train_loader, val_loader = build(*args, **kwargs)
        np.testing.assert_array_equal(train_loader.dataset.source_indices, idx_a)
        artifact_path.write_bytes(bytes_b)
        return train_loader, val_loader

    monkeypatch.setattr(mod, "build_dataloaders", swap_after_load)
    result = mod.train(cfg, dataset_path=str(ds_path),
                       nested_prior_artifact=str(artifact_path))
    mean_b, _, idx_b = mod.compute_target_stats(
        str(ds_path), cfg, nested_prior_artifact=str(artifact_path),
    )
    np.testing.assert_array_equal(idx_b, artifacts[1][0])
    assert mean_b == pytest.approx(20.0)
    assert mean_b != pytest.approx(mean_a)
    saved = torch.load(tmp_path / "run" / "best_model.pth", map_location="cpu",
                       weights_only=False)
    assert saved["target_mean"] == pytest.approx(mean_a)
    final = torch.load(tmp_path / "run" / "final_model.pth", map_location="cpu",
                       weights_only=False)
    assert final["target_mean"] == pytest.approx(mean_a)
    assert saved["nested_split_seed"] == 42
    assert saved["nested_prior_artifact_sha256"] == digest_a
    assert result["nested_prior_artifact_sha256"] == digest_a


def test_nested_prior_digest_uses_loaded_bytes_when_sidecar_changes_before_recording(tmp_path, monkeypatch):
    """The training matrix and checkpoint digest must come from the same read."""
    import hashlib
    from nsmor.pipeline.grouping import grouped_train_val_split
    from nsmor.analysis.analysis_priors import load_analysis_priors
    from nsmor.analysis.prediction_units import load_model_from_checkpoint

    mod = _load_train_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ds_path = tmp_path / "synthetic.pt"
    _make_synthetic_dataset(ds_path)
    dataset = torch.load(ds_path, weights_only=False)
    sessions = [f"animal{i}_session_1" for i in range(6)]
    dataset["session_ids"] = sessions
    torch.save(dataset, ds_path)
    train_idx, val_idx = grouped_train_val_split(sessions, 6, val_split=0.2, random_seed=42)
    original = np.tile([0.4, 0.3, 0.2, 0.1], (6, 1))
    replacement = np.tile([0.1, 0.2, 0.3, 0.4], (6, 1))
    artifact = {
        "nested_priors": original, "train_priors": original[train_idx],
        "val_priors": original[val_idx], "train_indices": train_idx,
        "val_indices": val_idx, "is_nested_cv": True,
        "source_fingerprint": hashlib.sha256(ds_path.read_bytes()).hexdigest(),
        "pipeline_semantics_version": dataset["pipeline_semantics_version"],
        "split_seed": 42, "val_split": 0.2,
        "mcmc_prior_provenance": "nested_outer_seed42_inner_5fold_recording_prefix_grouped",
        **_nested_prefix_metadata(sessions, train_idx, val_idx),
    }
    artifact_path = tmp_path / "nested.pt"
    torch.save(artifact, artifact_path)
    loaded_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    build = mod.build_dataloaders

    def replace_after_load(*args, **kwargs):
        train_loader, val_loader = build(*args, **kwargs)
        np.testing.assert_allclose(train_loader.dataset.sequences[0][0][0, 4:8], original[train_idx[0]])
        changed = dict(artifact, nested_priors=replacement,
                       train_priors=replacement[train_idx], val_priors=replacement[val_idx])
        torch.save(changed, artifact_path)
        return train_loader, val_loader

    monkeypatch.setattr(mod, "build_dataloaders", replace_after_load)
    output_dir = tmp_path / "run"
    cfg = _tiny_config(mod, output_dir)
    cfg.training.num_epochs = 1
    cfg.training.num_workers = 0
    result = mod.train(cfg, dataset_path=str(ds_path), nested_prior_artifact=str(artifact_path))
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() != loaded_digest
    saved = torch.load(output_dir / "best_model.pth", map_location="cpu", weights_only=False)
    assert result["nested_prior_artifact_sha256"] == saved["nested_prior_artifact_sha256"] == loaded_digest
    model = load_model_from_checkpoint(output_dir / "best_model.pth", torch.device("cpu"))
    with pytest.raises(ValueError, match="Nested artifact SHA-256 mismatch"):
        load_analysis_priors(dataset, ds_path, artifact_path, model,
                             loaded_source_fingerprint=artifact["source_fingerprint"])


def test_nested_prior_content_digest_binds_checkpoints_and_analysis_inputs(tmp_path, monkeypatch):
    """A valid replacement at the same path must differ from recorded content."""
    import hashlib
    import json
    from nsmor.pipeline.grouping import grouped_train_val_split
    from nsmor.analysis.analysis_priors import load_analysis_priors
    from nsmor.analysis.prediction_units import load_model_from_checkpoint as load_analysis_model

    mod = _load_train_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ds_path = tmp_path / "synthetic.pt"
    _make_synthetic_dataset(ds_path)
    dataset = torch.load(ds_path, weights_only=False)
    session_ids = [f"animal{i}_session_1" for i in range(6)]
    dataset["session_ids"] = session_ids
    torch.save(dataset, ds_path)
    source_digest = hashlib.sha256(ds_path.read_bytes()).hexdigest()
    train_idx, val_idx = grouped_train_val_split(
        session_ids, 6, val_split=0.2, random_seed=42,
    )
    priors = np.tile([0.4, 0.3, 0.2, 0.1], (6, 1))
    assert priors.shape == (6, 4)
    artifact = {
        "nested_priors": priors,
        "train_priors": priors[train_idx],
        "val_priors": priors[val_idx],
        "train_indices": train_idx,
        "val_indices": val_idx,
        "is_nested_cv": True,
        "source_fingerprint": source_digest,
        "pipeline_semantics_version": dataset["pipeline_semantics_version"],
        "split_seed": 42,
        "val_split": 0.2,
        "mcmc_prior_provenance": "nested_outer_seed42_inner_5fold_recording_prefix_grouped",
        **_nested_prefix_metadata(session_ids, train_idx, val_idx),
    }
    artifact_path = tmp_path / "nested.pt"
    torch.save(artifact, artifact_path)
    original_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    output_dir = tmp_path / "run"
    cfg = _tiny_config(mod, output_dir)
    cfg.training.num_epochs = 1
    cfg.training.num_workers = 0
    result = mod.train(
        cfg, dataset_path=str(ds_path), nested_prior_artifact=str(artifact_path),
        require_nested_validation=True,
    )
    checkpoints = [
        torch.load(output_dir / name, map_location="cpu", weights_only=False)
        for name in ("best_model.pth", "epoch_1.pth", "final_model.pth")
    ]
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert np.isfinite(checkpoints[0]["val_loss"])
    for record in (*checkpoints, metrics, result):
        assert record["nested_prior_artifact_sha256"] == original_digest
        assert record["nested_prior_artifact"] == str(artifact_path)
        assert record["nested_prior_fingerprint"] == source_digest
        assert record["validation_scope"] == "nested_outer_validation"

    analysis_model = load_analysis_model(output_dir / "best_model.pth", torch.device("cpu"))
    analysis_priors, analysis_val_idx = load_analysis_priors(
        dataset, ds_path, artifact_path, analysis_model,
        loaded_source_fingerprint=source_digest,
    )
    np.testing.assert_allclose(analysis_priors, priors)
    np.testing.assert_array_equal(analysis_val_idx, val_idx)

    replacement = np.tile([0.1, 0.2, 0.3, 0.4], (6, 1))
    assert replacement.shape == (6, 4)
    artifact.update(nested_priors=replacement, train_priors=replacement[train_idx],
                    val_priors=replacement[val_idx])
    torch.save(artifact, artifact_path)
    changed_train, changed_val, changed_priors, info = mod.load_nested_prior_split(
        artifact_path, ds_path, 6, session_ids,
    )
    np.testing.assert_array_equal(changed_train, train_idx)
    np.testing.assert_array_equal(changed_val, val_idx)
    np.testing.assert_allclose(changed_priors, replacement)
    assert not np.allclose(changed_priors, priors)
    # Path/source/split still match; the real analysis boundary must now
    # reject these valid inputs on the saved content digest alone.
    saved = checkpoints[0]
    assert info["nested_prior_artifact"] == saved["nested_prior_artifact"]
    assert info["nested_prior_fingerprint"] == saved["nested_prior_fingerprint"]
    current_digest = mod.compute_source_fingerprint(artifact_path)
    assert current_digest == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert current_digest != saved["nested_prior_artifact_sha256"]
    with pytest.raises(ValueError, match="Nested artifact SHA-256 mismatch"):
        load_analysis_priors(dataset, ds_path, artifact_path, analysis_model,
                             loaded_source_fingerprint=source_digest)

    resume_cfg = _tiny_config(mod, tmp_path / "blocked_resume")
    resume_cfg.training.num_epochs = 2
    resume_cfg.training.num_workers = 0
    resume_cfg.checkpoint.resume_from = str(output_dir / "epoch_1.pth")
    with pytest.raises(ValueError, match="nested prior artifact SHA-256 mismatch"):
        mod.train(resume_cfg, dataset_path=str(ds_path),
                  nested_prior_artifact=str(artifact_path))

    # Resume weights with the current digest are valid; an older companion
    # best checkpoint at the identical sidecar path still must be rejected.
    current_dir = tmp_path / "current_content"
    current_cfg = _tiny_config(mod, current_dir)
    current_cfg.training.num_epochs = 1
    current_cfg.training.num_workers = 0
    mod.train(current_cfg, dataset_path=str(ds_path),
              nested_prior_artifact=str(artifact_path))
    current_resume = torch.load(current_dir / "epoch_1.pth", map_location="cpu",
                                weights_only=False)
    assert current_resume["nested_prior_artifact_sha256"] == current_digest
    torch.save(saved, current_dir / "best_model.pth")
    candidate_cfg = _tiny_config(mod, tmp_path / "blocked_best")
    candidate_cfg.training.num_epochs = 2
    candidate_cfg.training.num_workers = 0
    candidate_cfg.checkpoint.resume_from = str(current_dir / "epoch_1.pth")
    with pytest.raises(ValueError, match="mismatched lineage key nested_prior_artifact_sha256"):
        mod.train(candidate_cfg, dataset_path=str(ds_path),
                  nested_prior_artifact=str(artifact_path))


def test_resume_past_phase_boundary_restores_state(tmp_path, monkeypatch):
    # ROUND-6 BLOCKER regression: resuming at/past the phase-1->2 boundary
    # must mirror the uninterrupted trajectory exactly — an uninterrupted
    # run builds a FRESH phase-2 optimizer at the boundary (phase-1 Adam
    # moments deliberately discarded), so a resumed run must restore ONLY
    # model weights and let the in-loop transition build that fresh
    # optimizer.  Previously the resume path either crashed (1-group
    # checkpoint loaded into a pre-built 2-group optimizer) or restored
    # state into an optimizer that was then discarded.
    mod = _load_train_module()
    ds_path = tmp_path / "nsmor_dataset.pt"
    _make_synthetic_dataset(ds_path)
    monkeypatch.setattr(mod, "_DATASET_PATH", str(ds_path))

    # Run 1: two-phase, phase1_epochs=1 → epoch 0 runs in phase 1 and saves
    # a periodic checkpoint at end of epoch 0 (still in phase 1).
    run1 = tmp_path / "run1"
    cfg = _tiny_config(mod, run1)
    mod.train(cfg, phase1_epochs=1)

    ckpt_path = run1 / "epoch_1.pth"
    assert ckpt_path.exists()
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Run 2: resume from the epoch-1 checkpoint — start_epoch=1 >=
    # phase1_epochs=1, so training resumes directly at the boundary.
    run2 = tmp_path / "run2"
    cfg2 = _tiny_config(mod, run2)
    cfg2.checkpoint.resume_from = str(ckpt_path)
    summary = mod.train(cfg2, phase1_epochs=1)
    assert summary is not None

    # The final checkpoint of run2 must carry a PHASE-2 optimizer shape
    # (two param groups named non_lif/lif — the in-loop transition ran).
    final_path = (run2 / "best_model.pth") if (run2 / "best_model.pth").exists() \
        else (run2 / "epoch_2.pth")
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    groups = final["optimizer_state_dict"]["param_groups"]
    names = [g.get("name") for g in groups]
    assert names == ["non_lif", "lif"], (
        f"resume did not land in the phase-2 optimizer; groups={names}"
    )
    # Model weights were actually restored before continuing: the resumed
    # run's best val loss must be finite and the run must have completed
    # its remaining epochs (summary produced).
    import math as _math
    assert _math.isfinite(summary.get("best_val_loss", float("nan")))


def test_resume_within_phase2_preserves_optimizer_state(tmp_path, monkeypatch):
    mod = _load_train_module()
    ds_path = tmp_path / "nsmor_dataset.pt"
    _make_synthetic_dataset(ds_path)
    monkeypatch.setattr(mod, "_DATASET_PATH", str(ds_path))
    run1 = tmp_path / "run1"
    cfg = _tiny_config(mod, run1)
    cfg.training.num_epochs = 3
    mod.train(cfg, phase1_epochs=1)
    ckpt_path = run1 / "epoch_2.pth"
    assert ckpt_path.exists()
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_names = [g.get("name") for g in saved["optimizer_state_dict"]["param_groups"]]
    assert saved_names == ["non_lif", "lif"], saved_names
    saved_steps = sorted(int(s["step"]) for s in saved["optimizer_state_dict"]["state"].values())
    run2 = tmp_path / "run2"
    cfg2 = _tiny_config(mod, run2)
    cfg2.training.num_epochs = 4
    cfg2.checkpoint.resume_from = str(ckpt_path)
    summary = mod.train(cfg2, phase1_epochs=1)
    assert summary is not None
    final_path = (run2 / "best_model.pth") if (run2 / "best_model.pth").exists() else (run2 / "epoch_4.pth")
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    final_names = [g.get("name") for g in final["optimizer_state_dict"]["param_groups"]]
    assert final_names == ["non_lif", "lif"], final_names
    final_steps = sorted(int(s["step"]) for s in final["optimizer_state_dict"]["state"].values())
    assert len(final_steps) == len(saved_steps), "param mismatch"
    assert min(final_steps) >= max(saved_steps), "phase-2 optimizer reset on resume"
    import math as _math
    assert _math.isfinite(summary.get("best_val_loss", float("nan")))


def test_resume_within_phase_preserves_optimizer_state(tmp_path, monkeypatch):
    # Complement to the boundary test: resuming WITHIN phase 1 must restore
    # the Adam moments and scheduler progress into the SAME optimizer (the
    # round-5 fix's original guarantee).  A fresh optimizer would restart
    # step counters at 0.
    mod = _load_train_module()
    ds_path = tmp_path / "nsmor_dataset.pt"
    _make_synthetic_dataset(ds_path)
    monkeypatch.setattr(mod, "_DATASET_PATH", str(ds_path))

    # Run 1 with num_epochs=2 so the epoch-0 checkpoint is mid-phase.
    run1 = tmp_path / "run1"
    cfg = _tiny_config(mod, run1)
    cfg.training.num_epochs = 2
    cfg.training.checkpoint_interval = 1
    mod.train(cfg, phase1_epochs=5)   # stays in phase 1 throughout

    ckpt_path = run1 / "epoch_1.pth"
    assert ckpt_path.exists()
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_steps = sorted(
        int(s["step"]) for s in saved["optimizer_state_dict"]["state"].values()
    )
    assert all(s >= 1 for s in saved_steps)

    # Run 2: resume within phase 1 (start_epoch=1 < phase1_epochs=5).
    run2 = tmp_path / "run2"
    cfg2 = _tiny_config(mod, run2)
    cfg2.training.num_epochs = 2
    cfg2.checkpoint.resume_from = str(ckpt_path)
    mod.train(cfg2, phase1_epochs=5)

    final_path = (run2 / "best_model.pth") if (run2 / "best_model.pth").exists() \
        else (run2 / "epoch_2.pth")
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    groups = final["optimizer_state_dict"]["param_groups"]
    # Single-group frontend optimizer preserved (not replaced by anything).
    assert len(groups) == 1
    steps = sorted(
        int(s["step"]) for s in final["optimizer_state_dict"]["state"].values()
    )
    # Restored moments survived: step counters continued from the saved
    # values (>= saved max), not restarted from scratch.
    assert len(steps) == len(saved_steps)
    assert min(steps) >= max(saved_steps), (
        f"optimizer state was reset on resume: saved={saved_steps} final={steps}"
    )


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

def _nested_source_replacement_case(tmp_path):
    """Two valid sources share row/group layout but differ in X, Y and labels."""
    import hashlib
    from nsmor.pipeline.grouping import grouped_train_val_split

    dataset_path = tmp_path / "source.pt"
    _make_synthetic_dataset(dataset_path)
    source_a = torch.load(dataset_path, weights_only=False)
    source_a["session_ids"] = [f"animal{i}_session_1" for i in range(6)]
    source_a["Y_seqs"] = [np.linspace(-3.0, 7.0, 12, dtype=np.float32) + i
                          for i in range(6)]
    torch.save(source_a, dataset_path)
    bytes_a = dataset_path.read_bytes()
    source_b = dict(source_a,
                    X_seqs=[values + 2.0 for values in source_a["X_seqs"]],
                    Y_seqs=[values + 30.0 for values in source_a["Y_seqs"]],
                    labels=np.ones(6, dtype=np.int64))
    replacement_path = tmp_path / "replacement.pt"
    torch.save(source_b, replacement_path)
    bytes_b = replacement_path.read_bytes()
    digest_a = hashlib.sha256(bytes_a).hexdigest()
    digest_b = hashlib.sha256(bytes_b).hexdigest()
    assert digest_a != digest_b
    assert not np.array_equal(source_a["Y_seqs"], source_b["Y_seqs"])
    train_idx, val_idx = grouped_train_val_split(
        source_a["session_ids"], 6, val_split=0.2, random_seed=42,
    )
    priors = np.tile([0.4, 0.3, 0.2, 0.1], (6, 1))
    artifact_path = tmp_path / "nested.pt"
    sidecar = {
        "nested_priors": priors, "train_priors": priors[train_idx],
        "val_priors": priors[val_idx], "train_indices": train_idx,
        "val_indices": val_idx, "source_fingerprint": digest_a,
        "is_nested_cv": True,
        "pipeline_semantics_version": source_a["pipeline_semantics_version"],
        "split_seed": 42, "val_split": 0.2,
        "mcmc_prior_provenance": "nested_outer_seed42_inner_5fold_recording_prefix_grouped",
        **_nested_prefix_metadata(source_a["session_ids"], train_idx, val_idx),
    }
    torch.save(sidecar, artifact_path)
    return (dataset_path, artifact_path, source_a, source_b, bytes_a, bytes_b,
            digest_a, digest_b, train_idx, sidecar)


def _replace_loaded_source(monkeypatch, dataset_path, bytes_a, bytes_b):
    """Intercept both legacy path loads and the repaired BytesIO production load."""
    import io

    real_load = torch.load
    reads = []

    def replace_after_deserialization(source, *args, **kwargs):
        loaded = real_load(source, *args, **kwargs)
        is_dataset = (source.getvalue() == bytes_a if isinstance(source, io.BytesIO)
                      else isinstance(source, (str, Path))
                      and Path(source).resolve() == dataset_path.resolve())
        if is_dataset and not reads:
            reads.append(loaded)
            dataset_path.write_bytes(bytes_b)
        return loaded

    monkeypatch.setattr(torch, "load", replace_after_deserialization)
    return reads


@pytest.mark.parametrize("dt_ms", [4.0, 10.0])
@pytest.mark.parametrize("entry", ["dataloaders", "target_stats", "train"])
def test_source_dataset_replacement_rejected(tmp_path, monkeypatch, dt_ms, entry):
    """Loaded A must never authorize B's valid sidecar or reach an optimizer."""
    import hashlib

    mod = _load_train_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    (dataset_path, artifact_path, source_a, source_b, bytes_a, bytes_b,
     digest_a, digest_b, train_idx, sidecar) = _nested_source_replacement_case(tmp_path)
    sidecar["source_fingerprint"] = digest_b
    torch.save(sidecar, artifact_path)
    # Confirm B and its sidecar are valid together before setting up the race.
    dataset_path.write_bytes(bytes_b)
    mod.load_nested_prior_split(artifact_path, dataset_path, 6, source_b["session_ids"])
    dataset_path.write_bytes(bytes_a)
    reads = _replace_loaded_source(monkeypatch, dataset_path, bytes_a, bytes_b)
    output_dir = tmp_path / "run"
    cfg = _tiny_config(mod, output_dir)
    cfg.model.dt_ms = dt_ms
    cfg.training.num_epochs = 1
    cfg.training.num_workers = 0
    cfg.training.normalize_targets = True
    cfg.training.target_clip_cm_s = 80.0
    cfg.loss.lambda_routing_aux = 0.0

    def forbidden_epoch(**kwargs):
        raise AssertionError("Mixed source objects reached the optimizer")

    monkeypatch.setattr(mod, "train_one_epoch", forbidden_epoch)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        if entry == "dataloaders":
            mod.build_dataloaders(cfg, str(dataset_path), nested_prior_artifact=str(artifact_path))
        elif entry == "target_stats":
            mod.compute_target_stats(str(dataset_path), cfg, nested_prior_artifact=str(artifact_path))
        else:
            mod.train(cfg, dataset_path=str(dataset_path), nested_prior_artifact=str(artifact_path))
    assert len(reads) == 1
    np.testing.assert_array_equal(reads[0]["Y_seqs"], source_a["Y_seqs"])
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == digest_b != digest_a
    assert not list(output_dir.glob("*.pth"))


@pytest.mark.parametrize("dt_ms", [4.0, 10.0])
def test_source_snapshot_normalization_and_checkpoint_lineage_stay_honest(
        tmp_path, monkeypatch, dt_ms):
    """A's accepted snapshot keeps A's fitted units and digest despite path B."""
    import hashlib
    import json
    from scripts.analyze_dynamics import load_dataset
    from nsmor.analysis.prediction_units import load_model_from_checkpoint

    mod = _load_train_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    (dataset_path, artifact_path, source_a, source_b, bytes_a, bytes_b,
     digest_a, digest_b, train_idx, sidecar) = _nested_source_replacement_case(tmp_path)
    output_dir = tmp_path / "run"
    cfg = _tiny_config(mod, output_dir)
    cfg.model.dt_ms = dt_ms
    cfg.training.num_epochs = 1
    cfg.training.num_workers = 0
    cfg.training.normalize_targets = True
    cfg.training.target_clip_cm_s = 80.0
    cfg.loss.lambda_routing_aux = 0.0
    mean_a, std_a, _ = mod._fit_target_stats(
        np.concatenate([source_a["Y_seqs"][i] for i in train_idx]).astype(np.float64),
        cfg, train_idx,
    )
    mean_b, std_b, _ = mod._fit_target_stats(
        np.concatenate([source_b["Y_seqs"][i] for i in train_idx]).astype(np.float64),
        cfg, train_idx,
    )
    assert not np.isclose(mean_a, mean_b)
    reads = _replace_loaded_source(monkeypatch, dataset_path, bytes_a, bytes_b)
    result = mod.train(cfg, dataset_path=str(dataset_path), nested_prior_artifact=str(artifact_path))
    checkpoints = [torch.load(output_dir / name, map_location="cpu", weights_only=False)
                   for name in ("best_model.pth", "epoch_1.pth", "final_model.pth")]
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert len(reads) == 1
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == digest_b
    artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    for record in (*checkpoints, metrics, result):
        assert record["nested_prior_fingerprint"] == digest_a != digest_b
        assert record["nested_prior_artifact_sha256"] == artifact_digest
    for checkpoint in checkpoints:
        assert np.isclose(checkpoint["target_mean"], mean_a)
        assert np.isclose(checkpoint["target_std"], std_a)
    assert all(np.isfinite(record["val_loss"]) for record in checkpoints)
    model = load_model_from_checkpoint(output_dir / "best_model.pth", torch.device("cpu"))
    assert np.isclose(model.target_mean, mean_a) and np.isclose(model.target_std, std_a)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_dataset(dataset_path, max_seq_len=None, checkpoint_model=model)


def test_nonnested_train_uses_loaded_targets_and_source_digest_after_same_path_swap(
        tmp_path, monkeypatch):
    """A's loader must set normalization and checkpoint identity after valid B replaces its path."""
    import hashlib
    import json

    mod = _load_train_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dataset_path = tmp_path / "dataset.pt"
    _make_synthetic_dataset(dataset_path)
    source_a = torch.load(dataset_path, weights_only=False)
    source_b = dict(source_a, Y_seqs=[y + 30.0 for y in source_a["Y_seqs"]],
                    mcmc_priors=np.roll(source_a["mcmc_priors"], 1, axis=1))
    bytes_a = dataset_path.read_bytes()
    replacement = tmp_path / "replacement.pt"
    torch.save(source_b, replacement)
    bytes_b = replacement.read_bytes()
    digest_a = hashlib.sha256(bytes_a).hexdigest()
    assert digest_a != hashlib.sha256(bytes_b).hexdigest()
    cfg = _tiny_config(mod, tmp_path / "run")
    cfg.training.num_epochs = 1
    cfg.training.num_workers = 0
    cfg.training.normalize_targets = True
    cfg.training.target_clip_cm_s = 80.0
    cfg.loss.lambda_routing_aux = 0.0
    build = mod.build_dataloaders
    expected = []

    def swap_after_loader(*args, **kwargs):
        train_loader, val_loader = build(*args, **kwargs)
        selected = train_loader.dataset
        targets = [sequence[1] for sequence in selected.sequences]
        indices = np.asarray(selected.source_indices, dtype=np.int64)
        expected.extend(mod._fit_target_stats(
            np.concatenate(targets).astype(np.float64), cfg, indices,
        )[:2])
        dataset_path.write_bytes(bytes_b)
        return train_loader, val_loader

    monkeypatch.setattr(mod, "build_dataloaders", swap_after_loader)
    result = mod.train(cfg, dataset_path=str(dataset_path))
    assert len(expected) == 2
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() != digest_a
    for name in ("best_model.pth", "epoch_1.pth", "final_model.pth"):
        saved = torch.load(tmp_path / "run" / name, weights_only=False)
        assert saved["target_mean"] == pytest.approx(expected[0])
        assert saved["target_std"] == pytest.approx(expected[1])
        assert saved["dataset_source_sha256"] == digest_a
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert metrics["dataset_source_sha256"] == digest_a
    assert result["dataset_source_sha256"] == digest_a

def test_scored_training_requires_nested_artifact_and_cli_diagnostic_opt_in(tmp_path, monkeypatch):
    """Scored entry points reject global OOF; deliberate diagnostics label every output."""
    import json

    mod = _load_train_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dataset_path = tmp_path / "synthetic.pt"
    _make_synthetic_dataset(dataset_path)
    output_dir = tmp_path / "run"
    cfg = _tiny_config(mod, output_dir)
    cfg.training.num_epochs = 1
    cfg.training.num_workers = 0
    config_path = cfg.to_yaml(tmp_path / "config.yaml")
    with pytest.raises(ValueError, match="nested.*artifact"):
        mod.train(cfg, dataset_path=str(dataset_path), require_nested_validation=True)
    argv = ["--config", str(config_path), "--dataset", str(dataset_path)]
    with pytest.raises(ValueError, match="nested.*artifact"):
        mod.main(argv)
    assert not output_dir.exists()

    mod.main([*argv, "--diagnostic_only"])
    result = json.loads((output_dir / "train.log").read_text())
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert result["validation_scope"] == metrics["validation_scope"] == "diagnostic_global_oof"
    assert result["metrics"]["validation_scope"] == "diagnostic_global_oof"


@pytest.mark.parametrize("lineage", ["invalid_version", "missing_tag", "modern", "historical"])
def test_lazy_training_metadata_requires_shared_provenance_gate(tmp_path, lineage):
    """Lazy rows and prior lineage come from one validated captured metadata object."""
    import hashlib
    from nsmor.config import PIPELINE_SEMANTICS_VERSION

    mod = _load_train_module()
    sessions = [f"recording{i}_session_1" for i in range(4)]
    metadata = {
        "trial_specs": [{"session_id": session} for session in sessions],
        "mcmc_priors": torch.full((4, 4), 0.25),
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
    }
    if lineage == "invalid_version":
        metadata["pipeline_semantics_version"] = "1.0-invalid"
    elif lineage == "missing_tag":
        metadata.pop("mcmc_prior_provenance")
        metadata.pop("animal_identity_status")
    elif lineage == "historical":
        metadata["mcmc_prior_provenance"] = "oof_2fold_animal_grouped_cv"
        metadata["animal_identity_status"] = "historical_unknown"
    path = tmp_path / "metadata.pt"
    torch.save(metadata, path)
    cfg = _tiny_config(mod, tmp_path / "unused")
    cfg.loss.lambda_routing_aux = 0.0
    if lineage in ("invalid_version", "missing_tag"):
        with pytest.raises(RuntimeError, match="semantics|mcmc_prior_provenance"):
            mod.build_dataloaders(cfg, str(path), use_lazy_loading=True)
    else:
        train_loader, val_loader = mod.build_dataloaders(cfg, str(path), use_lazy_loading=True)
        assert len(train_loader.dataset) + len(val_loader.dataset) == 4
        assert train_loader.dataset.prior_lineage == (
            metadata["mcmc_prior_provenance"], metadata["animal_identity_status"])
        assert train_loader.dataset.dataset_source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert train_loader.dataset.dataset.trial_specs == metadata["trial_specs"]


@pytest.mark.parametrize("session_ids", [None, ["recording_session_1"], [""] * 6])
@pytest.mark.parametrize("entry", ["dataloaders", "target_stats"])
def test_modern_training_rejects_unusable_recording_prefix_identities(tmp_path, session_ids, entry):
    mod = _load_train_module()
    path = tmp_path / "synthetic.pt"
    _make_synthetic_dataset(path)
    dataset = torch.load(path, weights_only=False)
    if session_ids is None:
        dataset.pop("session_ids")
    else:
        dataset["session_ids"] = session_ids
    torch.save(dataset, path)
    cfg = _tiny_config(mod, tmp_path / "unused")
    cfg.training.normalize_targets = True
    cfg.training.num_workers = 0
    with pytest.raises((ValueError, RuntimeError), match="session_ids"):
        if entry == "dataloaders":
            mod.build_dataloaders(cfg, str(path))
        else:
            mod.compute_target_stats(str(path), cfg)


@pytest.mark.parametrize("binding", [None, "legacy_resume_unbound", "unbound_lazy"])
@pytest.mark.parametrize("checkpoint_kind", ["resume", "companion_best", "direct_best"])
def test_modern_digestless_resume_cannot_downgrade_to_historical_binding(
        tmp_path, monkeypatch, binding, checkpoint_kind):
    """Removing SHA from modern resume/best bytes must never enable a legacy fallback."""
    mod = _load_train_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "synthetic.pt"
    _make_synthetic_dataset(path)
    source_dir = tmp_path / "source"
    cfg = _tiny_config(mod, source_dir)
    cfg.training.num_epochs = 1
    cfg.training.num_workers = 0
    result = mod.train(cfg, dataset_path=str(path))
    assert result["animal_identity_status"] == "unverified"
    assert result["validation_scope"] == "diagnostic_global_oof"
    attacked = source_dir / ("epoch_1.pth" if checkpoint_kind == "resume" else "best_model.pth")
    checkpoint = torch.load(attacked, weights_only=False)
    checkpoint.pop("dataset_source_sha256")
    if binding is not None:
        checkpoint["dataset_source_binding"] = binding
    torch.save(checkpoint, attacked)
    resumed = _tiny_config(mod, tmp_path / "rejected")
    resumed.training.num_workers = 0
    resume_name = "best_model.pth" if checkpoint_kind == "direct_best" else "epoch_1.pth"
    resumed.checkpoint.resume_from = str(source_dir / resume_name)
    with pytest.raises(ValueError, match="modern.*dataset_source_sha256"):
        mod.train(resumed, dataset_path=str(path))
    assert not (tmp_path / "rejected" / "final_model.pth").exists()
