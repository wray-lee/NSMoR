"""Regression tests for scripts/train.py checkpoint and data-split behaviour.

Covers:
- Atomic-write semantics (no partial file left on interrupted save).
- A completed run always yields a loadable best checkpoint.
- Non-finite val loss is surfaced rather than silently swallowed.
- Target-stats split matches the dataloader split (animal-disjoint).
- Deployment provenance fields in all checkpoint types.

All tests use tiny synthetic data to run in <10s on CPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import signal
import threading
import time
from pathlib import Path
from typing import Dict, Optional
from unittest import mock

import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION, FeatureConfig
from nsmor.pipeline.grouping import animal_of as _animal_of
from nsmor.pipeline.nested_prior import load_artifact_bytes


def _checkpoint_shas(*paths: Path) -> tuple[str, ...]:
    return tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)

# ── Fixtures ────────────────────────────────────────────────────────

_HIDDEN = 16
_SENSORY = 4
_MCMC = 4
_N_TRAIN = 8
_N_VAL = 2
_SEQ_LEN = 50
_N_SESSIONS = 5  # animals in the fixture; at least 2 so val is non-empty


def _make_synthetic_dataset(tmp_path: Path) -> Path:
    """Create a minimal synthetic dataset that mirrors the real schema."""
    rng = np.random.RandomState(0)
    n_total = _N_TRAIN + _N_VAL
    # X_seqs needs 8 columns: 4 physical + 4 MCMC prior slots.
    # NSMoRDataset._fill_priors writes mcmc_priors into X[:, 4:8].
    _X_DIM = 8  # per_frame_total_dim from FeatureConfig
    X_seqs = [rng.randn(_SEQ_LEN, _X_DIM).astype(np.float32) for _ in range(n_total)]
    # Y_seqs is 1-D per frame (scalar velocity target), matching the
    # NSMoRDataset assertion Y.shape == (seq_len,).
    Y_seqs = [rng.randn(_SEQ_LEN).astype(np.float32) for _ in range(n_total)]
    labels = np.zeros(n_total, dtype=np.int64)
    lengths = np.full(n_total, _SEQ_LEN, dtype=np.int64)
    # Generate valid probability simplex: softmax of random logits,
    # matching the real MCMC pipeline which always outputs rows summing to 1.
    _raw_logits = rng.randn(n_total, _MCMC).astype(np.float32)
    _exp = np.exp(_raw_logits - _raw_logits.max(axis=1, keepdims=True))
    mcmc_priors = (_exp / _exp.sum(axis=1, keepdims=True)).astype(np.float32)
    # Session ids mirror the real corpus shape:
    # ``<mass>cricket_001_<date>_<time>_session_<N>``.  The ``_session_N``
    # suffix splits ONE recording of ONE animal into blocks, so each
    # animal here owns two sessions.  This matters: with flat ids like
    # ``sess_0`` an animal is indistinguishable from a session, and a
    # session-grouped split passes an animal-disjointness assertion
    # vacuously.  Real ids make the distinction observable.
    session_ids = [
        f"0.{500 + (i % _N_SESSIONS)}cricket_001_20260101_00000"
        f"{i % _N_SESSIONS}_session_{1 + (i // _N_SESSIONS)}"
        for i in range(n_total)
    ]
    assert len(set(session_ids)) == n_total, (
        "fixture must give every trial a distinct session id"
    )
    assert len({_animal_of(s) for s in session_ids}) == _N_SESSIONS, (
        f"fixture must yield {_N_SESSIONS} animals across "
        f"{n_total} sessions, so animal != session"
    )

    dataset = {
        "X_seqs": X_seqs,
        "Y_seqs": Y_seqs,
        "labels": labels,
        "lengths": lengths,
        "mcmc_priors": mcmc_priors,
        "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
        "session_ids": session_ids,
        "feature_config": FeatureConfig(),
        # Track the constant, never a literal: a semantics bump is a
        # deliberate scientific event, and hardcoding the version here
        # turns every legitimate bump into a suite-wide failure that says
        # nothing about the code under test.
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
    }
    path = tmp_path / "test_dataset.pt"
    torch.save(dataset, path)
    return path


def _make_config(
    tmp_path: Path,
    *,
    epochs: int = 1,
    warmup_epochs: int = 20,
    normalize_targets: bool = False,
) -> "ExperimentConfig":
    """Build a minimal ExperimentConfig for testing."""
    from nsmor.config_parser import ExperimentConfig

    config = ExperimentConfig()
    config.model.sensory_dim = _SENSORY
    config.model.mcmc_dim = _MCMC
    config.model.hidden_dim = _HIDDEN
    config.model.num_gru_layers = 1
    config.model.dropout = 0.0
    config.training.num_epochs = epochs
    config.training.batch_size = max(_N_TRAIN, 4)
    config.training.max_seq_len = _SEQ_LEN
    config.training.random_seed = 42
    config.training.normalize_targets = normalize_targets
    config.training.target_clip_cm_s = 0
    config.training.lr_warmup_epochs = 0
    config.training.checkpoint_interval = 999  # no periodic ckpt
    config.loss.warmup_epochs = warmup_epochs
    config.checkpoint.output_dir = str(tmp_path / "run")
    config.checkpoint.resume_from = None
    return config


# ═════════════════════════════════════════════════════════════
# Test 1: completed run always produces a loadable best checkpoint
# ═════════════════════════════════════════════════════════════

def test_best_checkpoint_always_written(tmp_path: Path) -> None:
    """A 1-epoch run with warmup_epochs=20 (>> epochs) must still produce
    best_model.pth with a finite val_loss."""
    from scripts.train import train

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=1, warmup_epochs=20)

    results = train(config, lambda_reg=0.01, dataset_path=str(ds_path))

    best_path = Path(config.checkpoint.output_dir) / "best_model.pth"
    assert best_path.exists(), "best_model.pth was not written"

    ckpt = load_artifact_bytes(best_path.read_bytes())
    assert "model_state_dict" in ckpt
    assert "val_loss" in ckpt
    assert np.isfinite(ckpt["val_loss"]), (
        f"val_loss in checkpoint is not finite: {ckpt['val_loss']}"
    )

    assert np.isfinite(results["best_val_loss"]), (
        f"best_val_loss in results is not finite: {results['best_val_loss']}"
    )
    assert results["metrics"], "metrics dict is empty"
    assert "mse" in results["metrics"]


# ═════════════════════════════════════════════════════════════
# Test 2: atomic-write — no partial file left on interrupted save
# ═════════════════════════════════════════════════════════════

@pytest.mark.parametrize("val_loss", [float("nan"), float("inf"), -float("inf")])
def test_fresh_train_rejects_existing_best_before_model_or_output_writes(
    tmp_path: Path, val_loss: float,
) -> None:
    """Fresh training cannot select stale weights when validation is invalid."""
    from scripts.train import build_model, train

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    output_dir = Path(config.checkpoint.output_dir)
    output_dir.mkdir()
    best_path = output_dir / "best_model.pth"
    stale_model = build_model(config)
    stale_weights = {
        key: value.detach().clone() for key, value in stale_model.state_dict().items()
    }
    torch.save(
        {"model_state_dict": stale_weights, "val_loss": 0.125, "epoch": 8},
        best_path,
    )
    sentinel_bytes = best_path.read_bytes()
    metrics_path = output_dir / "metrics.json"
    metrics_sentinel = b"previous run's metrics\n"
    metrics_path.write_bytes(metrics_sentinel)

    with (
        mock.patch("scripts.train.build_model") as model_builder,
        mock.patch("scripts.train.train_one_epoch") as epoch_runner,
        mock.patch("scripts.train.validate", return_value=val_loss) as validator,
        mock.patch("scripts.train._atomic_save_checkpoint") as saver,
    ):
        with pytest.raises(FileExistsError, match="Pre-existing checkpoint") as error:
            train(config, lambda_reg=0.01, dataset_path=str(ds_path))
        assert str(best_path) in str(error.value)
        model_builder.assert_not_called()
        epoch_runner.assert_not_called()
        validator.assert_not_called()
        saver.assert_not_called()

    assert set(output_dir.iterdir()) == {best_path, metrics_path}
    assert best_path.read_bytes() == sentinel_bytes
    assert metrics_path.read_bytes() == metrics_sentinel
    saved_weights = load_artifact_bytes(best_path.read_bytes())["model_state_dict"]
    assert saved_weights.keys() == stale_weights.keys()
    for key, expected in stale_weights.items():
        assert torch.equal(saved_weights[key], expected), f"stale weight {key} was changed"


def test_atomic_save_no_partial_file(tmp_path: Path) -> None:
    """If torch.save raises mid-write, the TARGET path must not exist
    (only the .tmp file may be left)."""
    from scripts.train import _atomic_save_checkpoint
    from nsmor.model_nsmor_core import NSMoRCore

    model = NSMoRCore(
        sensory_dim=_SENSORY, mcmc_dim=_MCMC, hidden_dim=_HIDDEN,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = tmp_path / "ckpt.pth"

    # Normal save works
    _atomic_save_checkpoint(
        model=model, optimizer=optimizer, epoch=0, loss=1.0,
        config={}, path=target,
    )
    assert target.exists()
    target.unlink()

    # Interrupted save: patch torch.save to raise after partial write
    _original_save = torch.save

    def _failing_save(obj, f, *args, **kwargs):
        # Write a partial file (simulating crash mid-write)
        Path(f).write_bytes(b"PARTIAL")
        raise OSError("Simulated disk failure")

    with mock.patch("nsmor.checkpoint.torch.save", side_effect=_failing_save):
        with pytest.raises(OSError, match="Simulated disk failure"):
            _atomic_save_checkpoint(
                model=model, optimizer=optimizer, epoch=0, loss=1.0,
                config={}, path=target,
            )

    # The TARGET path must not exist — only the .tmp may be left
    assert not target.exists(), (
        "Atomic save left a partial file at the target path"
    )


# ═════════════════════════════════════════════════════════════
# Test 3: non-finite val loss is surfaced, not silently swallowed
# ═════════════════════════════════════════════════════════════

@pytest.mark.parametrize("val_loss", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_val_loss_handled(tmp_path: Path, val_loss: float) -> None:
    """When val_loss is NaN/Inf, the run must still complete and:
    - best_val_loss should remain inf (no best checkpoint from bad loss)
    - final_model.pth should exist as fallback
    - metrics should be computed from the final fallback checkpoint
    """
    from scripts.train import train, validate

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=1, warmup_epochs=0)

    # No non-finite validation value may select a best checkpoint, including -Inf.
    with mock.patch("scripts.train.validate", return_value=val_loss):
        results = train(config, lambda_reg=0.01, dataset_path=str(ds_path))

    output_dir = Path(config.checkpoint.output_dir)
    best_path = output_dir / "best_model.pth"
    final_path = output_dir / "final_model.pth"

    # best_model.pth must not be written for any non-finite validation loss.
    assert not best_path.exists(), (
        "best_model.pth should not be written when val_loss is non-finite"
    )
    # final_model.pth should exist as fallback
    assert final_path.exists(), "final_model.pth must always be written"
    # Metrics should still be populated (from final_fallback)
    assert results.get("eval_provenance") == "final_fallback"
    assert results["metrics"], (
        "metrics should be computed from final_model fallback"
    )


# ═════════════════════════════════════════════════════════════
# Test 4: target-stats split matches the dataloader split
# ═════════════════════════════════════════════════════════════

def test_target_stats_split_matches_dataloader(tmp_path: Path) -> None:
    """``build_dataloaders`` and ``compute_target_stats`` must agree on the
    grouped split at ANIMAL granularity (no data leakage).

    Both real functions are called and nothing is replicated locally.  The
    previous version of this test compared two identically-seeded local
    copies of the split logic to each other, so it could not fail no matter
    what the production functions did.

    Granularity is animal, not session: ``_session_N`` blocks belong to one
    animal, so a session-disjoint split still lets an animal straddle the
    two sides and share its baseline locomotor statistics with validation.
    Asserting at session granularity passes under that bug.
    """
    from scripts.train import (
        build_dataloaders,
        compute_target_stats,
        _VAL_SPLIT,
    )

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=1, normalize_targets=True)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    session_arr = np.asarray(dataset["session_ids"])
    animal_arr = np.array([_animal_of(s) for s in session_arr], dtype=object)
    all_sessions = set(np.unique(session_arr).tolist())

    def _sessions_of(loader) -> set:
        """Read each sequence's recorded source row, then its session id."""
        ds = loader.dataset
        assert len(ds.source_indices) == len(ds.sequences), (
            "source_indices must stay aligned with sequences"
        )
        return {session_arr[i] for i in ds.source_indices}

    def _animals_of(loader) -> set:
        ds = loader.dataset
        return {animal_arr[i] for i in ds.source_indices}

    # Real call 1 — the loaders define the split.
    train_loader, val_loader = build_dataloaders(
        config, dataset_path=str(ds_path), val_split=_VAL_SPLIT,
    )
    assert train_loader is not None and val_loader is not None
    train_sessions_build = _sessions_of(train_loader)
    val_sessions_build = _sessions_of(val_loader)

    # Real call 2 — the returned train indices define the train sessions.
    _mean, _std, train_indices_stats = compute_target_stats(
        str(ds_path), config, val_split=_VAL_SPLIT,
    )
    assert train_indices_stats.size > 0, (
        "compute_target_stats returned no train indices"
    )
    train_sessions_stats = set(session_arr[train_indices_stats].tolist())
    val_sessions_stats = all_sessions - train_sessions_stats

    # build_dataloaders must not put one session on both sides.
    assert not (train_sessions_build & val_sessions_build), (
        "build_dataloaders leaked sessions across the split: "
        f"{sorted(train_sessions_build & val_sessions_build)}"
    )
    # ...and, strictly stronger, not one ANIMAL on both sides.  This is the
    # assertion that fails under a session-grouped split.
    train_animals_build = _animals_of(train_loader)
    val_animals_build = _animals_of(val_loader)
    assert not (train_animals_build & val_animals_build), (
        "build_dataloaders leaked ANIMALS across the split (its "
        "_session_N blocks landed on opposite sides): "
        f"{sorted(train_animals_build & val_animals_build)}"
    )
    # compute_target_stats must be animal-disjoint from val too, or the
    # normalization statistics carry validation animals' locomotor scale.
    train_animals_stats = set(animal_arr[train_indices_stats].tolist())
    assert not (train_animals_stats & val_animals_build), (
        "compute_target_stats fit statistics on validation animals: "
        f"{sorted(train_animals_stats & val_animals_build)}"
    )
    # The two independent code paths must agree, session for session.
    assert train_sessions_build == train_sessions_stats, (
        f"train sessions differ: build={sorted(train_sessions_build)} "
        f"vs stats={sorted(train_sessions_stats)}"
    )
    assert val_sessions_build == val_sessions_stats, (
        f"val sessions differ: build={sorted(val_sessions_build)} "
        f"vs stats={sorted(val_sessions_stats)}"
    )


# ═════════════════════════════════════════════════════════════
# Test 5: multi-epoch warmup still produces best checkpoint
# ═════════════════════════════════════════════════════════════

def test_warmup_longer_than_epochs(tmp_path: Path) -> None:
    """Even when warmup_epochs (20) far exceeds total epochs (2),
    best_model.pth must still be written with a finite val_loss."""
    from scripts.train import train

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=2, warmup_epochs=20)

    results = train(config, lambda_reg=0.01, dataset_path=str(ds_path))

    best_path = Path(config.checkpoint.output_dir) / "best_model.pth"
    assert best_path.exists(), "best_model.pth not written with warmup > epochs"
    assert np.isfinite(results["best_val_loss"])
    assert results["metrics"], "metrics dict empty"


# ═════════════════════════════════════════════════════════════
# Test 6: deployment provenance fields in all checkpoint types
# ═════════════════════════════════════════════════════════════

def _assert_provenance_fields(
    ckpt: dict,
    ckpt_label: str,
    *,
    expected_phase: int,
    expected_dataset_path: str,
) -> None:
    """Assert the five deployment provenance fields exist and have
    correct types/values in a loaded checkpoint dict."""
    # target_mean: float
    assert "target_mean" in ckpt, (
        f"{ckpt_label}: missing 'target_mean'"
    )
    assert isinstance(ckpt["target_mean"], float), (
        f"{ckpt_label}: target_mean should be float, got {type(ckpt['target_mean'])}"
    )

    # target_std: float
    assert "target_std" in ckpt, (
        f"{ckpt_label}: missing 'target_std'"
    )
    assert isinstance(ckpt["target_std"], float), (
        f"{ckpt_label}: target_std should be float, got {type(ckpt['target_std'])}"
    )

    # target_clip_cm_s: float
    assert "target_clip_cm_s" in ckpt, (
        f"{ckpt_label}: missing 'target_clip_cm_s'"
    )
    assert isinstance(ckpt["target_clip_cm_s"], float), (
        f"{ckpt_label}: target_clip_cm_s should be float, got {type(ckpt['target_clip_cm_s'])}"
    )

    # training_phase: int in {0, 1, 2}
    assert "training_phase" in ckpt, (
        f"{ckpt_label}: missing 'training_phase'"
    )
    assert isinstance(ckpt["training_phase"], int), (
        f"{ckpt_label}: training_phase should be int, got {type(ckpt['training_phase'])}"
    )
    assert ckpt["training_phase"] in {0, 1, 2}, (
        f"{ckpt_label}: training_phase should be in {{0,1,2}}, got {ckpt['training_phase']}"
    )
    assert ckpt["training_phase"] == expected_phase, (
        f"{ckpt_label}: training_phase={ckpt['training_phase']} != expected {expected_phase}"
    )

    # dataset_path: str
    assert "dataset_path" in ckpt, (
        f"{ckpt_label}: missing 'dataset_path'"
    )
    assert isinstance(ckpt["dataset_path"], str), (
        f"{ckpt_label}: dataset_path should be str, got {type(ckpt['dataset_path'])}"
    )
    assert ckpt["dataset_path"] == expected_dataset_path, (
        f"{ckpt_label}: dataset_path={ckpt['dataset_path']!r} != expected {expected_dataset_path!r}"
    )


def test_provenance_single_phase(tmp_path: Path) -> None:
    """Single-phase 1-epoch run: best_model.pth and final_model.pth
    must carry all five deployment provenance fields with
    training_phase=0 and correct target stats."""
    from scripts.train import train

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    ds_path_str = str(ds_path)

    results = train(config, lambda_reg=0.01, dataset_path=ds_path_str)

    output_dir = Path(config.checkpoint.output_dir)

    # Both best and final must exist for a healthy single-phase run
    for name in ("best_model.pth", "final_model.pth"):
        ckpt_path = output_dir / name
        assert ckpt_path.exists(), f"{name} not found"
        ckpt = load_artifact_bytes(ckpt_path.read_bytes())
        _assert_provenance_fields(
            ckpt,
            name,
            expected_phase=0,
            expected_dataset_path=ds_path_str,
        )

        # target_mean and target_std must match compute_target_stats
        # For normalize_targets=False, they should be (0.0, 1.0)
        assert ckpt["target_mean"] == 0.0, (
            f"{name}: target_mean should be 0.0 (normalization disabled)"
        )
        assert ckpt["target_std"] == 1.0, (
            f"{name}: target_std should be 1.0 (normalization disabled)"
        )
        assert ckpt["target_clip_cm_s"] == 0.0, (
            f"{name}: target_clip_cm_s should be 0.0 (clipping disabled)"
        )


def test_provenance_two_phase(tmp_path: Path) -> None:
    """Two-phase run: final_model.pth must carry training_phase=2,
    periodic checkpoint in phase 1 must carry training_phase=1."""
    from scripts.train import train

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=3, warmup_epochs=0)
    # Enable periodic checkpoint every epoch so we get a phase-1 ckpt
    config.training.checkpoint_interval = 1
    ds_path_str = str(ds_path)

    results = train(
        config, lambda_reg=0.01,
        phase1_epochs=1,  # 1 epoch phase 1, 2 epochs phase 2
        dataset_path=ds_path_str,
    )

    output_dir = Path(config.checkpoint.output_dir)

    # Phase 1 periodic checkpoint (epoch 1)
    epoch1_ckpt_path = output_dir / "epoch_1.pth"
    assert epoch1_ckpt_path.exists(), "epoch_1.pth not found for phase-1 check"
    epoch1_ckpt = load_artifact_bytes(epoch1_ckpt_path.read_bytes())
    _assert_provenance_fields(
        epoch1_ckpt,
        "epoch_1.pth",
        expected_phase=1,
        expected_dataset_path=ds_path_str,
    )

    # Final checkpoint must be phase 2
    final_path = output_dir / "final_model.pth"
    assert final_path.exists(), "final_model.pth not found"
    final_ckpt = load_artifact_bytes(final_path.read_bytes())
    _assert_provenance_fields(
        final_ckpt,
        "final_model.pth",
        expected_phase=2,
        expected_dataset_path=ds_path_str,
    )


def test_two_phase_patience_and_best_checkpoint_are_phase_local(tmp_path: Path) -> None:
    """Phase 1 must finish, and phase 2 must choose its own best model."""
    from scripts.train import train

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=6, warmup_epochs=0)
    config.training.early_stopping_patience = 1

    with mock.patch("scripts.train.validate", side_effect=[0.1, 0.2, 0.3, 4.0, 5.0]):
        results = train(
            config, lambda_reg=0.01, phase1_epochs=3, dataset_path=str(ds_path),
        )

    assert results["history"]["val_loss"] == [0.1, 0.2, 0.3, 4.0, 5.0]
    assert results["best_val_loss"] == pytest.approx(4.0)
    assert results["eval_provenance"] == "best"
    best = load_artifact_bytes((Path(config.checkpoint.output_dir) / "best_model.pth").read_bytes())
    assert (best["training_phase"], best["epoch"], best["val_loss"]) == (2, 3, 4.0)


def test_resume_phase1_boundary_ignores_later_phase2_best(tmp_path: Path) -> None:
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=6, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    source.training.early_stopping_patience = 1
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.1, 0.2, 0.3, 4.0, 5.0]),
    ):
        train(source, phase1_epochs=3, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_3.pth"
    assert load_artifact_bytes(periodic.read_bytes())["best_val_loss"] == pytest.approx(0.1)
    assert load_artifact_bytes((source_dir / "best_model.pth").read_bytes())["training_phase"] == 2

    resumed = _make_config(tmp_path, epochs=7, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "resume_boundary")
    resumed.checkpoint.resume_from = str(periodic)
    resumed.training.early_stopping_patience = 1
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[6.0, 7.0]),
    ):
        result = train(resumed, phase1_epochs=3, dataset_path=str(ds),
                       trusted_historical_checkpoint_sha256=_checkpoint_shas(periodic))

    # Epoch-boundary recovery (Task #25): the loss curve is a property of the
    # WHOLE run.  The uninterrupted trajectory appends across the phase-1→2
    # boundary, so the resumed run must too; only the early-stop counter and
    # best_val_loss are phase-local.  Phase-1 history is therefore preserved.
    assert result["history"]["val_loss"] == [0.1, 0.2, 0.3, 6.0, 7.0]
    best = load_artifact_bytes((Path(resumed.checkpoint.output_dir) / "best_model.pth").read_bytes())
    assert (best["training_phase"], best["epoch"], best["val_loss"]) == (2, 3, 6.0)

def test_resume_phase2_discards_phase1_companion(tmp_path: Path) -> None:
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=3, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.1, 5.0, 4.0]),
    ):
        train(source, phase1_epochs=1, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_3.pth"
    assert load_artifact_bytes(periodic.read_bytes())["best_val_loss"] == pytest.approx(4.0)
    torch.save(load_artifact_bytes((source_dir / "epoch_1.pth").read_bytes()), source_dir / "best_model.pth")

    resumed = _make_config(tmp_path, epochs=4, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "resume_phase2")
    resumed.checkpoint.resume_from = str(periodic)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", return_value=6.0),
    ):
        result = train(resumed, phase1_epochs=1, dataset_path=str(ds),
                       trusted_historical_checkpoint_sha256=_checkpoint_shas(periodic))

    best = load_artifact_bytes((Path(resumed.checkpoint.output_dir) / "best_model.pth").read_bytes())
    assert (best["training_phase"], best["epoch"], best["val_loss"]) == (2, 2, 4.0)
    assert result["best_val_loss"] == pytest.approx(4.0)

def test_resume_phase2_preserves_same_phase_best(tmp_path: Path) -> None:
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=3, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.1, 4.0, 5.0]),
    ):
        train(source, phase1_epochs=1, dataset_path=str(ds))
    source_dir = Path(source.checkpoint.output_dir)
    old_best = load_artifact_bytes((source_dir / "best_model.pth").read_bytes())
    assert (old_best["training_phase"], old_best["val_loss"]) == (2, 4.0)

    resumed = _make_config(tmp_path, epochs=4, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "resume_phase2_valid")
    resumed.checkpoint.resume_from = str(source_dir / "epoch_3.pth")
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", return_value=6.0),
    ):
        result = train(resumed, phase1_epochs=1, dataset_path=str(ds),
                       trusted_historical_checkpoint_sha256=_checkpoint_shas(
                           source_dir / "epoch_3.pth", source_dir / "best_model.pth"))

    best = load_artifact_bytes((Path(resumed.checkpoint.output_dir) / "best_model.pth").read_bytes())
    assert (best["training_phase"], best["epoch"], best["val_loss"]) == (2, 1, 4.0)
    assert all(torch.equal(v, best["model_state_dict"][k]) for k, v in old_best["model_state_dict"].items())
    assert result["best_val_loss"] == pytest.approx(4.0)

    # A claimed same-phase best must have matching weights in at least one candidate.
    claimed = load_artifact_bytes((source_dir / "epoch_3.pth").read_bytes())
    claimed["best_val_loss"] = 3.0
    altered = source_dir / "epoch_3_claimed.pth"
    torch.save(claimed, altered)
    worse = dict(old_best)
    worse["val_loss"] = worse["best_val_loss"] = 5.0
    destination = tmp_path / "resume_phase2_invalid"
    destination.mkdir()
    torch.save(worse, destination / "best_model.pth")
    resumed.checkpoint.output_dir = str(destination)
    resumed.checkpoint.resume_from = str(altered)
    with pytest.raises(ValueError, match="same-phase best weights"):
        train(resumed, phase1_epochs=1, dataset_path=str(ds),
              trusted_historical_checkpoint_sha256=_checkpoint_shas(
                  altered, source_dir / "best_model.pth", destination / "best_model.pth"))
    assert load_artifact_bytes((destination / "best_model.pth").read_bytes())["val_loss"] == 5.0

def test_early_stopped_final_epoch_resumes_at_next_epoch(tmp_path: Path) -> None:
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=6, warmup_epochs=0)
    source.training.early_stopping_patience = 1
    with mock.patch("scripts.train.validate", side_effect=[0.1, 0.2, 0.3, 4.0, 5.0]):
        result = train(source, phase1_epochs=3, dataset_path=str(ds))

    assert len(result["history"]["val_loss"]) == 5
    final_path = Path(source.checkpoint.output_dir) / "final_model.pth"
    final = load_artifact_bytes(final_path.read_bytes())
    assert (final["training_phase"], final["epoch"]) == (2, 4)

    resumed = _make_config(tmp_path, epochs=6, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "resume_final")
    resumed.checkpoint.resume_from = str(final_path)
    executed = []
    restored = []

    def run_epoch(**kwargs):
        executed.append(kwargs["epoch"])
        restored.append(kwargs["optimizer"].state_dict())
        return 1.0, {}

    with (
        mock.patch("scripts.train.train_one_epoch", side_effect=run_epoch),
        mock.patch("scripts.train.validate", return_value=3.0),
    ):
        train(resumed, phase1_epochs=3, dataset_path=str(ds),
              trusted_historical_checkpoint_sha256=_checkpoint_shas(
                  final_path, Path(source.checkpoint.output_dir) / "best_model.pth"))

    assert executed == [5]
    prior_state = final["optimizer_state_dict"]
    first_key = next(iter(prior_state["state"]))
    assert torch.equal(restored[0]["state"][first_key]["exp_avg"].cpu(), prior_state["state"][first_key]["exp_avg"])
    assert [g["lr"] for g in restored[0]["param_groups"]] == [g["lr"] for g in prior_state["param_groups"]]
    resumed_final = load_artifact_bytes((Path(resumed.checkpoint.output_dir) / "final_model.pth").read_bytes())
    assert resumed_final["epoch"] == 5
    assert resumed_final["scheduler_state_dict"]["last_epoch"] == final["scheduler_state_dict"]["last_epoch"] + 1

def test_resume_earlier_phase1_uses_own_certified_best(tmp_path: Path) -> None:
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=4, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.1, 0.2, 0.3, 4.0]),
    ):
        train(source, phase1_epochs=3, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_1.pth"
    assert load_artifact_bytes((source_dir / "best_model.pth").read_bytes())["training_phase"] == 2
    resumed = _make_config(tmp_path, epochs=3, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "resume_phase1_early")
    resumed.checkpoint.resume_from = str(periodic)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.2, 0.3]),
    ):
        result = train(resumed, phase1_epochs=3, dataset_path=str(ds),
                       trusted_historical_checkpoint_sha256=_checkpoint_shas(periodic))

    best = load_artifact_bytes((Path(resumed.checkpoint.output_dir) / "best_model.pth").read_bytes())
    assert (best["training_phase"], best["epoch"], best["val_loss"]) == (1, 0, 0.1)
    assert result["best_val_loss"] == pytest.approx(0.1)

def test_resume_earlier_phase1_refuses_missing_same_phase_best(tmp_path: Path) -> None:
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=4, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.1, 0.2, 0.3, 4.0]),
    ):
        train(source, phase1_epochs=3, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_2.pth"
    saved = load_artifact_bytes(periodic.read_bytes())
    assert (saved["training_phase"], saved["val_loss"], saved["best_val_loss"]) == (1, 0.2, 0.1)
    assert load_artifact_bytes((source_dir / "best_model.pth").read_bytes())["training_phase"] == 2
    resumed = _make_config(tmp_path, epochs=3, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "resume_phase1_missing")
    resumed.checkpoint.resume_from = str(periodic)
    with pytest.raises(ValueError, match="absent historical best model"):
        train(resumed, phase1_epochs=3, dataset_path=str(ds),
              trusted_historical_checkpoint_sha256=_checkpoint_shas(periodic))

def test_provenance_with_normalization(tmp_path: Path) -> None:
    """When target normalization is enabled, the provenance fields
    must carry the actual computed target_mean/target_std (not the
    identity defaults)."""
    from scripts.train import train, compute_target_stats, _VAL_SPLIT

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(
        tmp_path, epochs=1, warmup_epochs=0, normalize_targets=True,
    )
    # Need a nonzero clip to avoid the coherence warning path
    config.training.target_clip_cm_s = 100.0
    ds_path_str = str(ds_path)

    # Pre-compute expected stats so we can cross-check
    expected_mean, expected_std, _ = compute_target_stats(
        ds_path_str, config, val_split=_VAL_SPLIT,
    )

    results = train(config, lambda_reg=0.01, dataset_path=ds_path_str)

    output_dir = Path(config.checkpoint.output_dir)
    best_path = output_dir / "best_model.pth"
    assert best_path.exists(), "best_model.pth not found"
    ckpt = load_artifact_bytes(best_path.read_bytes())

    _assert_provenance_fields(
        ckpt,
        "best_model.pth (normalized)",
        expected_phase=0,
        expected_dataset_path=ds_path_str,
    )

    # target_mean and target_std must match compute_target_stats output
    assert ckpt["target_mean"] == pytest.approx(expected_mean, abs=1e-6), (
        f"target_mean mismatch: {ckpt['target_mean']} vs {expected_mean}"
    )
    assert ckpt["target_std"] == pytest.approx(expected_std, abs=1e-6), (
        f"target_std mismatch: {ckpt['target_std']} vs {expected_std}"
    )
    assert ckpt["target_clip_cm_s"] == 100.0


def test_provenance_mcmc_prior_train_serve_consistency(
    tmp_path: Path,
) -> None:
    """When mcmc_prior_train_serve_consistency is in dataset, it must
    be recorded in checkpoint provenance and metrics.json."""
    from scripts.train import train

    ds_path = _make_synthetic_dataset(tmp_path)
    data = load_artifact_bytes(ds_path.read_bytes())
    dummy_consistency = {
        "argmax_agreement": 0.95,
        "mean_total_variation_distance": 0.05,
        "n_trials": 10,
    }
    data["mcmc_prior_train_serve_consistency"] = dummy_consistency
    torch.save(data, ds_path)

    config = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    ds_path_str = str(ds_path)
    train(config, lambda_reg=0.01, dataset_path=ds_path_str)

    output_dir = Path(config.checkpoint.output_dir)
    ckpt = load_artifact_bytes((output_dir / "best_model.pth").read_bytes())
    assert "mcmc_prior_train_serve_consistency" in ckpt
    assert ckpt["mcmc_prior_train_serve_consistency"] == dummy_consistency

    with open(output_dir / "metrics.json") as f:
        metrics = json.load(f)
    assert "mcmc_prior_train_serve_consistency" in metrics
    assert metrics["mcmc_prior_train_serve_consistency"] == dummy_consistency


# ═════════════════════════════════════════════════════════════
# Test 7: the architecture PERMITS ADR-0005 gradient isolation
# ═════════════════════════════════════════════════════════════

def test_architecture_permits_gradient_isolation() -> None:
    """The frontend/backend split admits ADR-0005 gradient isolation.

    This test sets ``requires_grad`` by hand, so it does NOT verify that
    ``train()`` applies the freeze schedule — that is
    ``test_train_enforces_phase_freeze_schedule``'s job.  What it does
    verify is the architectural precondition that makes the schedule
    possible at all: frontend and backend share no parameters and no
    hidden gradient path links them, so freezing one really does leave
    its parameters at ``None`` grad.

    Phase 1: frontend trainable, backend frozen -> all backend
    parameters have None grad after forward+backward.
    Phase 2: frontend frozen, backend trainable -> all frontend
    parameters have None grad after forward+backward.
    """
    from nsmor.model_nsmor_core import NSMoRCore
    from nsmor.loss import FrontendLoss, BioDecisionLoss

    device = torch.device("cpu")
    model = NSMoRCore(
        sensory_dim=_SENSORY, mcmc_dim=_MCMC, hidden_dim=_HIDDEN,
    ).to(device)

    B, T = 2, 10
    x = torch.randn(B, T, _SENSORY + _MCMC, device=device)
    y = torch.randn(B, T, device=device)
    lengths = torch.tensor([T, T], dtype=torch.long, device=device)

    # ── Phase 1: Train frontend only, freeze backend ──
    for p in model.backend.parameters():
        p.requires_grad = False
    for p in model.frontend.parameters():
        p.requires_grad = True

    model.zero_grad(set_to_none=True)
    frontend_criterion = FrontendLoss()
    y_pred, internals = model(x, lengths, return_internals=True)
    loss1 = frontend_criterion(y_pred=y_pred, y_true=y, lengths=lengths)
    loss1.backward()

    # Backend parameters must have None grad
    backend_grads_p1 = [p.grad for p in model.backend.parameters()]
    assert all(g is None for g in backend_grads_p1), (
        "Phase 1: Found non-None gradient on frozen backend parameter"
    )
    # Frontend parameters must have received gradients
    frontend_grads_p1 = [p.grad for p in model.frontend.parameters() if p.grad is not None]
    assert len(frontend_grads_p1) > 0, (
        "Phase 1: Expected non-empty gradients on trainable frontend parameters"
    )

    # ── Phase 2: Train backend only, freeze frontend ──
    model.zero_grad(set_to_none=True)
    for p in model.frontend.parameters():
        p.requires_grad = False
    for p in model.backend.parameters():
        p.requires_grad = True

    backend_criterion = BioDecisionLoss()
    y_pred, internals = model(x, lengths, return_internals=True)
    g_gru = internals["routing_gates"][:, :, 1:2]
    lif_spikes = internals["lif_spikes"]
    loss2 = backend_criterion(
        y_pred=y_pred,
        y_true=y,
        lengths=lengths,
        g_gru=g_gru,
        lambda_reg=0.01,
        lif_spikes=lif_spikes,
        lambda_energy=0.0,
        lambda_sparse=0.0,
        lambda_jerk=0.0,
        annealing_factor=1.0,
    )
    loss2.backward()

    # Frontend parameters must have None grad
    frontend_grads_p2 = [p.grad for p in model.frontend.parameters()]
    assert all(g is None for g in frontend_grads_p2), (
        "Phase 2: Found non-None gradient on frozen frontend parameter"
    )
    # Backend parameters must have received gradients
    backend_grads_p2 = [p.grad for p in model.backend.parameters() if p.grad is not None]
    assert len(backend_grads_p2) > 0, (
        "Phase 2: Expected non-empty gradients on trainable backend parameters"
    )


# ═════════════════════════════════════════════════════════════
# Test 8: Warmup factor restart behavior
# ═════════════════════════════════════════════════════════════

def test_warmup_factor_restart() -> None:
    """Warmup factor starts near 0.0 at epoch 0 and reaches 1.0 at boundary.

    Guarantees:
    - compute_warmup_factor(0, W) < 0.1 for W >= 5 (smooth S-curve restart)
    - compute_warmup_factor(W, W) == 1.0 (post-warmup returns full scale)
    - compute_warmup_factor(0, 0) == 1.0 (warmup disabled returns 1.0)
    """
    from scripts.train import compute_warmup_factor, compute_lr_warmup_scale

    # Check warmup factor at epoch 0 for various warmup epochs W >= 5
    for W in [5, 10, 20, 50]:
        val = compute_warmup_factor(0, W)
        assert val < 0.1, (
            f"compute_warmup_factor(0, {W}) = {val} >= 0.1; expected smooth start near 0.0"
        )

    # Post-warmup reaches 1.0
    for W in [2, 5, 10, 20]:
        assert compute_warmup_factor(W, W) == 1.0
        assert compute_warmup_factor(W + 5, W) == 1.0

    # Warmup disabled (W=0) is always 1.0
    assert compute_warmup_factor(0, 0) == 1.0

    # Also test LR warmup scale restart at epoch 0
    for W in [20, 50]:
        assert compute_lr_warmup_scale(0, W) < 0.1


# ═════════════════════════════════════════════════════════════
# Test 9: sweep_escape_sensitivity Cartesian row validation
# ═════════════════════════════════════════════════════════════

def test_sweep_escape_sensitivity() -> None:
    """sweep_escape_sensitivity returns Cartesian product of bands x min_runs
    with all required keys and valid numeric fields."""
    from scripts.train import sweep_escape_sensitivity

    rng = np.random.RandomState(42)
    t1 = np.array([0.0, 15.0, 25.0, 0.0, 5.0], dtype=np.float32)
    t2 = np.array([30.0, 35.0, 0.0, 0.0, 0.0], dtype=np.float32)
    p1 = t1 + rng.normal(0, 0.5, size=t1.shape).astype(np.float32)
    p2 = t2 + rng.normal(0, 0.5, size=t2.shape).astype(np.float32)

    bands = [10.0, 20.0]
    min_runs = [1, 2, 3]

    rows = sweep_escape_sensitivity(
        all_true=[t1, t2],
        all_pred=[p1, p2],
        bands_cm_s=bands,
        min_runs=min_runs,
    )

    expected_row_count = len(bands) * len(min_runs)
    assert len(rows) == expected_row_count, (
        f"Expected {expected_row_count} rows, got {len(rows)}"
    )

    expected_keys = {
        "band_cm_s",
        "min_run",
        "n_escape_frames",
        "n_escape_events",
        "escape_rmse",
        "resting_rmse",
        "escape_ratio",
    }

    seen_pairs = set()
    for row in rows:
        assert set(row.keys()) == expected_keys, (
            f"Row keys mismatch: {set(row.keys()) ^ expected_keys}"
        )
        assert isinstance(row["band_cm_s"], (int, float))
        assert isinstance(row["min_run"], (int, np.integer))
        assert isinstance(row["n_escape_frames"], (int, np.integer))
        assert isinstance(row["n_escape_events"], (int, np.integer))
        assert isinstance(row["escape_ratio"], float)
        assert 0.0 <= row["escape_ratio"] <= 1.0

        pair = (row["band_cm_s"], row["min_run"])
        assert pair not in seen_pairs, f"Duplicate (band, min_run) pair: {pair}"
        seen_pairs.add(pair)

    assert seen_pairs == {(b, r) for b in bands for r in min_runs}


# ═════════════════════════════════════════════════════════════
# Test 10: split cross-verification dry
# ═════════════════════════════════════════════════════════════

def test_split_cross_verification_dry(tmp_path: Path) -> None:
    """Cross-verify that build_dataloaders and compute_target_stats extract the
    EXACT same train indices under the same configuration."""
    from scripts.train import (
        build_dataloaders,
        compute_target_stats,
        _VAL_SPLIT,
    )

    ds_path = _make_synthetic_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    n_total = len(dataset["X_seqs"])

    config = _make_config(tmp_path, epochs=1, normalize_targets=True)

    # 1. Call build_dataloaders and read the split provenance it recorded.
    train_loader, val_loader = build_dataloaders(
        config, dataset_path=str(ds_path), val_split=_VAL_SPLIT,
    )
    assert train_loader is not None

    train_ds = train_loader.dataset
    assert len(train_ds.source_indices) == len(train_ds.sequences), (
        "source_indices must stay aligned with sequences"
    )
    train_indices_from_loader = np.array(train_ds.source_indices)
    assert train_indices_from_loader.size > 0
    assert np.all(train_indices_from_loader < n_total)
    # Provenance must be a permutation-free subset, never a duplicate row.
    assert len(set(train_ds.source_indices)) == len(train_ds.source_indices)

    # 2. Call compute_target_stats which now returns train indices directly
    target_mean, target_std, train_indices_from_stats = compute_target_stats(
        str(ds_path), config, val_split=_VAL_SPLIT,
    )

    # 3. Assert EXACT identity
    np.testing.assert_array_equal(
        np.sort(train_indices_from_loader),
        np.sort(train_indices_from_stats),
        err_msg="build_dataloaders and compute_target_stats used different train indices",
    )
    assert set(train_indices_from_loader) == set(train_indices_from_stats)


# ═════════════════════════════════════════════════════════════
# Test 11: resume within phase 1
# ═════════════════════════════════════════════════════════════

def test_resume_within_phase1(tmp_path: Path) -> None:
    """Save at epoch 5 (phase1_epochs=10), resume, and verify run completes."""
    from scripts.train import train

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=5, warmup_epochs=0)
    config.training.checkpoint_interval = 5

    # Initial 5 epochs (epochs 0..4 in Phase 1)
    results_init = train(
        config, lambda_reg=0.01, phase1_epochs=10, dataset_path=str(ds_path),
    )

    output_dir = Path(config.checkpoint.output_dir)
    ckpt_path = output_dir / "epoch_5.pth"
    assert ckpt_path.exists(), "epoch_5.pth was not saved"

    ckpt = load_artifact_bytes(ckpt_path.read_bytes())
    assert ckpt["epoch"] == 4  # 0-indexed, 5th epoch completed
    assert ckpt["training_phase"] == 1

    # Resume from epoch 5 to epoch 7
    resume_output_dir = tmp_path / "resume_phase1_run"
    config_resume = _make_config(tmp_path, epochs=7, warmup_epochs=0)
    config_resume.checkpoint.output_dir = str(resume_output_dir)
    config_resume.checkpoint.resume_from = str(ckpt_path)

    results_resume = train(
        config_resume,
        lambda_reg=0.01,
        phase1_epochs=10,
        dataset_path=str(ds_path),
        trusted_historical_checkpoint_sha256=_checkpoint_shas(ckpt_path, output_dir / "best_model.pth"),
    )

    assert np.isfinite(results_resume["best_val_loss"])
    assert "final_train_loss" in results_resume
    final_path = resume_output_dir / "final_model.pth"
    assert final_path.exists(), "final_model.pth was not written after resume"

    final_ckpt = load_artifact_bytes(final_path.read_bytes())
    assert final_ckpt["epoch"] == 6  # 0-indexed, 7 total epochs


# ═════════════════════════════════════════════════════════════
# Test 12: resume boundary landing
# ═════════════════════════════════════════════════════════════

def test_resume_boundary_landing(tmp_path: Path) -> None:
    """Save at epoch 10 (phase1_epochs=10 boundary), resume, and verify Phase 2
    optimizer has 2 parameter groups."""
    from scripts.train import train, train_one_epoch

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=10, warmup_epochs=0)
    config.training.checkpoint_interval = 10

    # Initial 10 epochs (epochs 0..9 in Phase 1)
    results_init = train(
        config, lambda_reg=0.01, phase1_epochs=10, dataset_path=str(ds_path),
    )

    output_dir = Path(config.checkpoint.output_dir)
    ckpt_path = output_dir / "epoch_10.pth"
    assert ckpt_path.exists(), "epoch_10.pth was not saved"

    ckpt = load_artifact_bytes(ckpt_path.read_bytes())
    assert ckpt["epoch"] == 9  # 0-indexed, 10th epoch
    assert ckpt["training_phase"] == 1
    assert len(ckpt["optimizer_state_dict"]["param_groups"]) == 1

    # Resume from epoch 10 to epoch 12 (crossing into Phase 2)
    resume_output_dir = tmp_path / "resume_boundary_run"
    config_resume = _make_config(tmp_path, epochs=12, warmup_epochs=0)
    config_resume.checkpoint.output_dir = str(resume_output_dir)
    config_resume.checkpoint.resume_from = str(ckpt_path)

    observed_optimizers = []
    orig_train_one_epoch = train_one_epoch

    def _spy_train_one_epoch(*args, **kwargs):
        opt = kwargs.get("optimizer") if "optimizer" in kwargs else args[3]
        observed_optimizers.append(opt)
        return orig_train_one_epoch(*args, **kwargs)

    with mock.patch("scripts.train.train_one_epoch", side_effect=_spy_train_one_epoch):
        results_resume = train(
            config_resume,
            lambda_reg=0.01,
            phase1_epochs=10,
            dataset_path=str(ds_path),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(ckpt_path),
        )

    # Verify that Phase 2 optimizer created upon landing has 2 param groups
    assert len(observed_optimizers) > 0
    phase2_opt = observed_optimizers[0]
    assert len(phase2_opt.param_groups) == 2, (
        f"Expected 2 param groups in Phase 2 optimizer, got {len(phase2_opt.param_groups)}"
    )
    group_names = [g.get("name") for g in phase2_opt.param_groups]
    assert group_names == ["non_lif", "lif"]

    # Check final checkpoint
    final_path = resume_output_dir / "final_model.pth"
    assert final_path.exists()
    final_ckpt = load_artifact_bytes(final_path.read_bytes())
    assert len(final_ckpt["optimizer_state_dict"]["param_groups"]) == 2
    assert final_ckpt["training_phase"] == 2


# ═════════════════════════════════════════════════════════════
# Test 13: resume mid-phase 2
# ═════════════════════════════════════════════════════════════

def test_resume_mid_phase2(tmp_path: Path) -> None:
    """Save at epoch 15 (mid-Phase 2), resume, and verify 2 param groups and
    momentum buffers are restored."""
    from scripts.train import train, train_one_epoch

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=15, warmup_epochs=0)
    config.training.checkpoint_interval = 15

    # Initial 15 epochs (Phase 1: 0..9, Phase 2: 10..14)
    results_init = train(
        config, lambda_reg=0.01, phase1_epochs=10, dataset_path=str(ds_path),
    )

    output_dir = Path(config.checkpoint.output_dir)
    ckpt_path = output_dir / "epoch_15.pth"
    assert ckpt_path.exists(), "epoch_15.pth was not saved"

    ckpt = load_artifact_bytes(ckpt_path.read_bytes())
    assert ckpt["epoch"] == 14  # 0-indexed, 15th epoch
    assert ckpt["training_phase"] == 2
    assert len(ckpt["optimizer_state_dict"]["param_groups"]) == 2

    ckpt_opt_state = ckpt["optimizer_state_dict"]["state"]
    assert len(ckpt_opt_state) > 0, "No optimizer state saved in epoch_15.pth"
    has_exp_avg = any(
        "exp_avg" in s and s["exp_avg"].norm() > 0
        for s in ckpt_opt_state.values()
    )
    assert has_exp_avg, "No momentum buffer (exp_avg) in checkpoint"

    # Resume from epoch 15 to epoch 17
    resume_output_dir = tmp_path / "resume_mid_phase2_run"
    config_resume = _make_config(tmp_path, epochs=17, warmup_epochs=0)
    config_resume.checkpoint.output_dir = str(resume_output_dir)
    config_resume.checkpoint.resume_from = str(ckpt_path)

    restored_opt_before_step = None
    orig_train_one_epoch = train_one_epoch

    def _spy_train_one_epoch(*args, **kwargs):
        nonlocal restored_opt_before_step
        opt = kwargs.get("optimizer") if "optimizer" in kwargs else args[3]
        if restored_opt_before_step is None:
            restored_opt_before_step = {
                "num_groups": len(opt.param_groups),
                "group_names": [g.get("name") for g in opt.param_groups],
                "state_len": len(opt.state),
                "has_exp_avg": any("exp_avg" in s for s in opt.state.values()),
                "has_exp_avg_sq": any("exp_avg_sq" in s for s in opt.state.values()),
            }
        return orig_train_one_epoch(*args, **kwargs)

    with mock.patch("scripts.train.train_one_epoch", side_effect=_spy_train_one_epoch):
        results_resume = train(
            config_resume,
            lambda_reg=0.01,
            phase1_epochs=10,
            dataset_path=str(ds_path),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(ckpt_path, output_dir / "best_model.pth"),
        )

    assert restored_opt_before_step is not None
    assert restored_opt_before_step["num_groups"] == 2
    assert restored_opt_before_step["group_names"] == ["non_lif", "lif"]
    assert restored_opt_before_step["state_len"] > 0
    assert restored_opt_before_step["has_exp_avg"]
    assert restored_opt_before_step["has_exp_avg_sq"]

    assert np.isfinite(results_resume["best_val_loss"])
    final_path = resume_output_dir / "final_model.pth"
    assert final_path.exists()
    final_ckpt = load_artifact_bytes(final_path.read_bytes())
    assert final_ckpt["epoch"] == 16  # 0-indexed, 17 total epochs
    assert final_ckpt["training_phase"] == 2
    assert len(final_ckpt["optimizer_state_dict"]["param_groups"]) == 2


# ═════════════════════════════════════════════════════════════
# Test 11: train() itself enforces the ADR-0005 freeze schedule
# ═════════════════════════════════════════════════════════════

def test_train_enforces_phase_freeze_schedule(tmp_path: Path) -> None:
    """ADR-0005 enforcement point: ``train()`` — not the caller — must
    freeze the backend during Phase 1 and the frontend during Phase 2.

    ``test_architecture_permits_gradient_isolation`` toggles
    ``requires_grad`` by hand, so it proves the architecture *permits*
    isolation but would still pass if ``train()`` stopped freezing
    anything.  This test observes the real per-epoch parameter state
    inside ``train()``.
    """
    from scripts.train import train, train_one_epoch

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=4, warmup_epochs=0)

    observed = []
    orig_train_one_epoch = train_one_epoch

    def _spy_train_one_epoch(*args, **kwargs):
        mdl = kwargs["model"] if "model" in kwargs else args[0]
        opt = kwargs["optimizer"] if "optimizer" in kwargs else args[3]
        observed.append({
            "frontend_trainable": [
                p.requires_grad for p in mdl.frontend.parameters()
            ],
            "backend_trainable": [
                p.requires_grad for p in mdl.backend.parameters()
            ],
            "n_groups": len(opt.param_groups),
        })
        return orig_train_one_epoch(*args, **kwargs)

    with mock.patch(
        "scripts.train.train_one_epoch", side_effect=_spy_train_one_epoch,
    ):
        train(
            config, lambda_reg=0.01, phase1_epochs=2,
            dataset_path=str(ds_path),
        )

    assert len(observed) == 4, (
        f"expected 4 trained epochs, observed {len(observed)}"
    )

    # Epochs 0-1 → Phase 1: backend frozen, frontend trainable, 1 group.
    for epoch_idx in (0, 1):
        state = observed[epoch_idx]
        assert state["backend_trainable"], "backend has no parameters"
        assert not any(state["backend_trainable"]), (
            f"epoch {epoch_idx}: Phase 1 must freeze every backend parameter"
        )
        assert all(state["frontend_trainable"]), (
            f"epoch {epoch_idx}: Phase 1 must train every frontend parameter"
        )
        assert state["n_groups"] == 1, (
            f"epoch {epoch_idx}: Phase 1 optimizer must have 1 param group"
        )

    # Epochs 2-3 → Phase 2: frontend frozen, backend trainable, 2 groups.
    for epoch_idx in (2, 3):
        state = observed[epoch_idx]
        assert state["frontend_trainable"], "frontend has no parameters"
        assert not any(state["frontend_trainable"]), (
            f"epoch {epoch_idx}: Phase 2 must freeze every frontend parameter"
        )
        assert all(state["backend_trainable"]), (
            f"epoch {epoch_idx}: Phase 2 must train every backend parameter"
        )
        assert state["n_groups"] == 2, (
            f"epoch {epoch_idx}: Phase 2 optimizer must have 2 param groups"
        )


def test_resume_into_new_output_dir_preserves_best_model(tmp_path: Path) -> None:
    """Resuming into a fresh output_dir must preserve genuine historical best model
    and evaluate as 'best' even if resumed epochs show higher val_loss (no improvement)."""
    from scripts.train import train

    ds_path = _make_synthetic_dataset(tmp_path)
    run1_dir = tmp_path / "run1"
    config1 = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    config1.checkpoint.output_dir = str(run1_dir)
    config1.training.checkpoint_interval = 1

    res1 = train(config1, lambda_reg=0.01, dataset_path=str(ds_path))
    best1_val = res1["best_val_loss"]
    assert (run1_dir / "best_model.pth").exists()
    assert (run1_dir / "epoch_1.pth").exists()

    # Resume from epoch_1 into run2_dir with higher val_loss simulation
    run2_dir = tmp_path / "run2"
    config2 = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    config2.checkpoint.output_dir = str(run2_dir)
    config2.checkpoint.resume_from = str(run1_dir / "epoch_1.pth")

    # Patch validate to return worse val_loss (best1_val + 50.0) so epoch 2 does NOT improve
    with mock.patch("scripts.train.validate", return_value=best1_val + 50.0):
        res2 = train(config2, lambda_reg=0.01, dataset_path=str(ds_path),
                     trusted_historical_checkpoint_sha256=_checkpoint_shas(
                         run1_dir / "epoch_1.pth", run1_dir / "best_model.pth"))

    assert (run2_dir / "best_model.pth").exists(), (
        "best_model.pth must be preserved in fresh output_dir from genuine historical best"
    )
    assert res2["eval_provenance"] == "best", (
        f"Expected eval_provenance='best', got {res2['eval_provenance']}"
    )
    assert res2["best_val_loss"] == pytest.approx(best1_val)


def test_resume_never_falls_back_to_training_loss_as_best_val_loss(tmp_path: Path) -> None:
    """A checkpoint with training loss only (no validation loss metadata) must NOT
    use train loss as best_val_loss (must initialize best_val_loss to inf)."""
    from scripts.train import train

    ds_path = _make_synthetic_dataset(tmp_path)
    run1_dir = tmp_path / "run1_trainonly"
    config1 = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    config1.checkpoint.output_dir = str(run1_dir)
    config1.training.checkpoint_interval = 1

    train(config1, lambda_reg=0.01, dataset_path=str(ds_path))
    epoch1_ckpt = run1_dir / "epoch_1.pth"
    assert epoch1_ckpt.exists()

    # Strip validation loss keys from checkpoint so it only carries train loss in ckpt['loss']
    ckpt_data = load_artifact_bytes(epoch1_ckpt.read_bytes())
    ckpt_data.pop("val_loss", None)
    ckpt_data.pop("best_val_loss", None)
    ckpt_data["loss"] = 0.05  # train loss is very small (0.05)
    torch.save(ckpt_data, epoch1_ckpt)

    # Resume into run2
    run2_dir = tmp_path / "run2_res"
    config2 = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    config2.checkpoint.output_dir = str(run2_dir)
    config2.checkpoint.resume_from = str(epoch1_ckpt)

    # If train loss was wrongly used, val_loss ~1.0 > 0.05 would not save best_model.pth.
    # With correct inf initialization, the epoch 2 val_loss (~1.0) must be saved as best!
    res2 = train(config2, lambda_reg=0.01, dataset_path=str(ds_path),
                 trusted_historical_checkpoint_sha256=_checkpoint_shas(
                     epoch1_ckpt, run1_dir / "best_model.pth"))
    assert (run2_dir / "best_model.pth").exists()
    assert res2["eval_provenance"] == "best"
    assert res2["best_val_loss"] > 0.05  # Recorded genuine validation loss, not 0.05 train loss


def test_nonnested_resume_and_best_reject_changed_bytes_at_same_path(tmp_path: Path) -> None:
    """Checkpoint and companion best selection bind the actual nonnested source bytes."""
    import hashlib
    from scripts.train import train

    ds_path = _make_synthetic_dataset(tmp_path)
    cfg_a = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    cfg_a.training.checkpoint_interval = 1
    train(cfg_a, dataset_path=str(ds_path))
    run_a = Path(cfg_a.checkpoint.output_dir)
    digest_a = hashlib.sha256(ds_path.read_bytes()).hexdigest()
    source_b = load_artifact_bytes(ds_path.read_bytes())
    source_b["Y_seqs"] = [y + 4.0 for y in source_b["Y_seqs"]]
    source_b["mcmc_priors"] = np.roll(source_b["mcmc_priors"], 1, axis=1)
    torch.save(source_b, ds_path)
    digest_b = hashlib.sha256(ds_path.read_bytes()).hexdigest()
    assert digest_a != digest_b
    for source_name in ("epoch_1.pth", "best_model.pth"):
        resume = _make_config(tmp_path, epochs=2, warmup_epochs=0)
        resume.checkpoint.output_dir = str(tmp_path / f"blocked_{source_name}")
        resume.checkpoint.resume_from = str(run_a / source_name)
        with pytest.raises(ValueError, match="dataset_source_sha256 mismatch"):
            train(resume, dataset_path=str(ds_path),
                  trusted_historical_checkpoint_sha256=_checkpoint_shas(run_a / source_name))

    cfg_b = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    cfg_b.training.checkpoint_interval = 1
    cfg_b.checkpoint.output_dir = str(tmp_path / "run_b")
    train(cfg_b, dataset_path=str(ds_path))
    run_b = Path(cfg_b.checkpoint.output_dir)
    assert load_artifact_bytes((run_b / "epoch_1.pth").read_bytes())["dataset_source_sha256"] == digest_b
    # A same-path old best can have finite, plausible validation metadata.
    # It still cannot rank against B's current checkpoints.
    torch.save(load_artifact_bytes((run_a / "best_model.pth").read_bytes()), run_b / "best_model.pth")
    resume = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    resume.checkpoint.output_dir = str(tmp_path / "blocked_companion")
    resume.checkpoint.resume_from = str(run_b / "epoch_1.pth")
    with pytest.raises(ValueError, match="dataset_source_sha256 mismatch"):
        train(resume, dataset_path=str(ds_path),
              trusted_historical_checkpoint_sha256=_checkpoint_shas(
                  run_b / "epoch_1.pth", run_b / "best_model.pth"))

def test_legacy_digestless_resume_keeps_historical_source_unbound(tmp_path: Path) -> None:
    """A real old checkpoint remains usable, with its unknown training source visible."""
    import hashlib
    from nsmor.analysis.prediction_units import load_model_from_checkpoint
    from scripts.analyze_dynamics import load_dataset
    from scripts.train import train

    ds_path = _make_synthetic_dataset(tmp_path)
    first = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    first.training.checkpoint_interval = 1
    train(first, dataset_path=str(ds_path))
    run_dir = Path(first.checkpoint.output_dir)
    for name in ("epoch_1.pth", "best_model.pth"):
        path = run_dir / name
        old = load_artifact_bytes(path.read_bytes())
        del old["dataset_source_sha256"]
        torch.save(old, path)
    resumed = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    resumed.training.checkpoint_interval = 1
    resumed.checkpoint.output_dir = str(tmp_path / "resumed")
    resumed.checkpoint.resume_from = str(run_dir / "epoch_1.pth")
    train(resumed, dataset_path=str(ds_path),
          trusted_historical_checkpoint_sha256=_checkpoint_shas(
              run_dir / "epoch_1.pth", run_dir / "best_model.pth"))
    final = Path(resumed.checkpoint.output_dir) / "final_model.pth"
    saved = load_artifact_bytes(final.read_bytes())
    assert saved["dataset_source_sha256"] == hashlib.sha256(ds_path.read_bytes()).hexdigest()
    assert saved["dataset_source_binding"] == "legacy_resume_unbound"
    metrics = json.loads((Path(resumed.checkpoint.output_dir) / "metrics.json").read_text())
    assert metrics["dataset_source_binding"] == "legacy_resume_unbound"
    model = load_model_from_checkpoint(final, torch.device("cpu"))
    load_dataset(ds_path, max_seq_len=None, checkpoint_model=model,
                 trusted_historical_checkpoint_sha256=_checkpoint_shas(final)[0])
    assert model.analysis_dataset_binding == "legacy_resume_unbound"

@pytest.mark.parametrize("bad_dt", [
    True, False, float("nan"), float("inf"), -float("inf"), 0., -4.,
    "4", None, [4.], torch.tensor(4.), np.array([4.]),
])
@pytest.mark.parametrize("entry", ["reconstruct", "build"])
def test_invalid_model_clock_rejected_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_dt: object, entry: str,
) -> None:
    """Serialized and active config clocks must not reach the frozen constructor."""
    from nsmor import model_utils
    from nsmor.analysis.prediction_units import load_model_from_checkpoint
    from scripts import train

    def forbidden_constructor(**kwargs: object) -> None:
        pytest.fail("Invalid dt_ms reached the frozen NSMoRCore constructor")

    if entry == "build":
        config = _make_config(tmp_path)
        config.model.dt_ms = bad_dt
        monkeypatch.setattr(train, "NSMoRCore", forbidden_constructor)
        with pytest.raises(ValueError, match="dt_ms.*finite positive"):
            train.build_model(config)
    else:
        checkpoint = tmp_path / "invalid-clock.pth"
        torch.save({
            "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
            "config": {"model": {"dt_ms": bad_dt, "hidden_dim": 4}},
            "model_state_dict": {},
        }, checkpoint)
        monkeypatch.setattr(model_utils, "NSMoRCore", forbidden_constructor)
        with pytest.raises(ValueError, match="dt_ms.*finite positive"):
            load_model_from_checkpoint(checkpoint, torch.device("cpu"))


@pytest.mark.parametrize("captured", [False, True])
def test_restore_clock_mismatch_is_atomic(
    tmp_path: Path, captured: bool,
) -> None:
    """An 8ms checkpoint must not change any 4ms weights, buffers, or RNG."""
    from nsmor.checkpoint import save_checkpoint
    from scripts.train import load_checkpoint
    from nsmor.model_nsmor_core import NSMoRCore

    saved = NSMoRCore(hidden_dim=4, dt_ms=8., lif_tau_syn=12.)
    active = NSMoRCore(hidden_dim=4, dt_ms=4., lif_tau_syn=12.)
    optimizer = torch.optim.AdamW(active.parameters(), lr=0.7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7)
    checkpoint = tmp_path / "dt8.pth"
    save_checkpoint(saved, torch.optim.AdamW(saved.parameters()), 3, 1.,
                    {"model": {"dt_ms": 8.}}, checkpoint)
    payload = checkpoint.read_bytes() if captured else None
    if captured:
        checkpoint.write_bytes(b"pathname replaced after capture")
    before = {key: value.clone() for key, value in active.state_dict().items()}
    before_optimizer = optimizer.state_dict()
    before_scheduler = scheduler.state_dict()
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="dt_ms.*conflicts|clock.*mismatch"):
        load_checkpoint(checkpoint, active, optimizer, scheduler,
                        map_location="cpu", payload=payload)
    assert active.dt_ms == 4.
    for key, expected in before.items():
        assert torch.equal(active.state_dict()[key], expected), key
    assert optimizer.state_dict() == before_optimizer
    assert scheduler.state_dict() == before_scheduler
    assert torch.equal(torch.get_rng_state(), before_rng)


@pytest.mark.parametrize("bad_dt", [True, np.bool_(True), float("inf"), None])
def test_invalid_active_restore_clock_is_atomic(
    tmp_path: Path, bad_dt: object,
) -> None:
    from scripts.train import load_checkpoint
    from nsmor.model_nsmor_core import NSMoRCore

    active = NSMoRCore(hidden_dim=4, dt_ms=4.)
    before = {key: value.clone() for key, value in active.state_dict().items()}
    active.dt_ms = bad_dt
    path = tmp_path / "valid-saved-clock.pth"
    torch.save({
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "model_state_dict": {key: value + 1 for key, value in before.items()},
        "config": {"model": {"dt_ms": 4.}},
    }, path)
    with pytest.raises(ValueError, match="dt_ms.*finite positive"):
        load_checkpoint(path, active, map_location="cpu")
    for key, expected in before.items():
        assert torch.equal(active.state_dict()[key], expected), key


@pytest.mark.parametrize("bad_dt", [True, float("inf"), "4", [4.], None])
def test_restore_invalid_clock_is_atomic(tmp_path: Path, bad_dt: object) -> None:
    from scripts.train import load_checkpoint
    from nsmor.model_nsmor_core import NSMoRCore

    active = NSMoRCore(hidden_dim=4, dt_ms=4.)
    before = {key: value.clone() for key, value in active.state_dict().items()}
    path = tmp_path / "invalid.pth"
    torch.save({
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "model_state_dict": {key: value + 1 for key, value in before.items()},
        "config": {"model": {"dt_ms": bad_dt}},
    }, path)
    with pytest.raises(ValueError, match="dt_ms.*finite positive"):
        load_checkpoint(path, active, map_location="cpu")
    for key, expected in before.items():
        assert torch.equal(active.state_dict()[key], expected), key


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("dt_ms", [4., 8.])
def test_valid_clock_restores_real_model_and_legacy_metadata(
    tmp_path: Path, modern: bool, dt_ms: float,
) -> None:
    from nsmor.checkpoint import save_checkpoint
    from scripts.train import load_checkpoint
    from nsmor.analysis.prediction_units import load_model_from_checkpoint
    from nsmor.model_nsmor_core import NSMoRCore

    saved = NSMoRCore(hidden_dim=4, dt_ms=dt_ms, lif_tau_syn=12.)
    path = tmp_path / "valid.pth"
    save_checkpoint(saved, torch.optim.AdamW(saved.parameters()), 3, 1.,
                    {"model": {"hidden_dim": 4, "dt_ms": dt_ms, "lif_tau_syn": 12.}},
                    path)
    if modern:
        from nsmor.pipeline.nested_prior import load_artifact_bytes
        state = load_artifact_bytes(path.read_bytes(), map_location="cpu")
        state.update(dataset_source_sha256="a" * 64, animal_identity_status="unverified",
                     mcmc_prior_provenance="oof_5fold_recording_prefix_grouped_cv")
        torch.save(state, path)
    restored = NSMoRCore(hidden_dim=4, dt_ms=dt_ms, lif_tau_syn=12.)
    load_checkpoint(path, restored, map_location="cpu")
    reconstructed = load_model_from_checkpoint(path, torch.device("cpu"))
    for model in (restored, reconstructed):
        assert model.dt_ms == dt_ms
        for key, expected in saved.state_dict().items():
            assert torch.equal(model.state_dict()[key], expected), key


@pytest.mark.parametrize("entry", ["restore", "reconstruct"])
def test_modern_checkpoint_missing_clock_fails_closed(
    tmp_path: Path, entry: str,
) -> None:
    from scripts.train import load_checkpoint
    from nsmor.analysis.prediction_units import load_model_from_checkpoint
    from nsmor.model_nsmor_core import NSMoRCore

    active = NSMoRCore(hidden_dim=4, dt_ms=4.)
    path = tmp_path / "missing-clock.pth"
    torch.save({
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "config": {"model": {"hidden_dim": 4}},
        "model_state_dict": active.state_dict(),
        "dataset_source_sha256": "a" * 64, "animal_identity_status": "unverified",
    }, path)
    with pytest.raises(ValueError, match="dt_ms"):
        if entry == "restore":
            load_checkpoint(path, active, map_location="cpu")
        else:
            load_model_from_checkpoint(path, torch.device("cpu"))


@pytest.mark.parametrize("filename", ["resume.pth", "best_model.pth", "final_model.pth"])
@pytest.mark.parametrize("dt_ms", [4., 8.])
def test_training_restore_uses_guarded_bytes_after_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str, dt_ms: float,
) -> None:
    """Resume/final evaluation must restore the snapshot the clock guard saw."""
    from nsmor.checkpoint import save_checkpoint
    from nsmor.model_nsmor_core import NSMoRCore
    from scripts import train as trainer

    saved = NSMoRCore(hidden_dim=4, dt_ms=dt_ms)
    active = NSMoRCore(hidden_dim=4, dt_ms=dt_ms)
    path = tmp_path / filename
    save_checkpoint(saved, torch.optim.AdamW(saved.parameters()), 0, 1.,
                    {"model": {"dt_ms": dt_ms}}, path)
    payload = path.read_bytes()
    real_restore = trainer._canonical_load_checkpoint

    def replace_then_restore(*args: object, **kwargs: object) -> dict:
        assert kwargs["payload"] == payload
        path.write_bytes(b"replaced after clock validation")
        return real_restore(*args, **kwargs)

    monkeypatch.setattr(trainer, "_canonical_load_checkpoint", replace_then_restore)
    trainer.load_checkpoint(path, active, map_location="cpu")
    for key, expected in saved.state_dict().items():
        assert torch.equal(active.state_dict()[key], expected), key


@pytest.mark.parametrize("candidate", ["best_model.pth", "final_model.pth"])
@pytest.mark.parametrize("violation", ["clock", "lineage"])
def test_final_evaluation_rejects_replaced_clock_before_state_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate: str,
    violation: str,
) -> None:
    """Reach the real final-best call without executing training or plotting."""
    from nsmor.pipeline.nested_prior import load_artifact_bytes
    from scripts import train as trainer
    from tests.test_pipeline_resampling import clock_dataset

    dataset = tmp_path / "dataset.pt"
    torch.save(clock_dataset(), dataset)
    config = _make_config(tmp_path)
    config.training.num_workers = 0
    captured: dict = {}

    def fake_epoch(**kwargs: object) -> tuple[float, dict]:
        captured.update(model=kwargs["model"], optimizer=kwargs["optimizer"])
        return 1., {}

    def replace_clock(
        history: dict, output_dir: Path, history_start_epoch: int = 0,
    ) -> Path:
        path = output_dir / candidate
        state = load_artifact_bytes(path.read_bytes(), map_location="cpu")
        if violation == "clock":
            state["config"]["model"]["dt_ms"] = 8.
        else:
            state["dataset_source_sha256"] = "f" * 64
        state["model_state_dict"] = {
            key: value + 1 for key, value in state["model_state_dict"].items()
        }
        torch.save(state, path)
        captured["weights"] = {
            key: value.clone() for key, value in captured["model"].state_dict().items()
        }
        captured["optimizer_state"] = captured["optimizer"].state_dict()
        captured["rng"] = torch.get_rng_state().clone()
        return output_dir / "unused.png"

    monkeypatch.setattr(trainer, "train_one_epoch", fake_epoch)
    monkeypatch.setattr(trainer, "_maybe_step_scheduler", lambda *_: None)
    monkeypatch.setattr(trainer, "validate", lambda **_: (
        1. if candidate == "best_model.pth" else float("nan")
    ))
    monkeypatch.setattr(trainer, "plot_loss_curve", replace_clock)
    with mock.patch.object(trainer, "_canonical_load_checkpoint") as restore:
        with mock.patch.object(trainer, "compute_metrics") as metrics:
            error = "dt_ms.*conflicts" if violation == "clock" else "dataset_source_sha256"
            with pytest.raises(ValueError, match=error):
                trainer.train(config, dataset_path=str(dataset))
            restore.assert_not_called()
            metrics.assert_not_called()
    for key, expected in captured["weights"].items():
        assert torch.equal(captured["model"].state_dict()[key], expected), key
    assert captured["optimizer"].state_dict() == captured["optimizer_state"]
    assert torch.equal(torch.get_rng_state(), captured["rng"])


@pytest.mark.parametrize("phase1_epochs", [None, 2, 1, 0])
@pytest.mark.parametrize("candidate", [
    "resume", "source_best", "companion_best", "destination_best",
])
def test_train_preflight_rejects_checkpoint_clock_without_running_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate: str,
    phase1_epochs: int | None,
) -> None:
    """Every candidate clock is checked before restoring any resume state."""
    from nsmor.checkpoint import save_checkpoint
    from nsmor.pipeline.nested_prior import load_artifact_bytes
    from scripts import train as trainer
    from tests.test_pipeline_resampling import clock_dataset

    dataset_path = tmp_path / "dataset.pt"
    torch.save(clock_dataset(), dataset_path)
    config = _make_config(tmp_path, epochs=3, warmup_epochs=0)
    config.model.dt_ms = 4.
    config.training.num_workers = 0
    source = tmp_path / "source"
    source.mkdir()
    resume = source / ("best_model.pth" if candidate == "source_best" else "epoch_1.pth")
    saved = trainer.build_model(config)
    save_checkpoint(saved, trainer.build_optimizer(saved, config), 0, 1.,
                    config.to_dict(), resume)
    state = load_artifact_bytes(resume.read_bytes(), map_location="cpu")
    state.update(
        dataset_path=str(dataset_path),
        dataset_source_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        is_nested_cv=False, validation_scope="diagnostic_global_oof",
        animal_identity_status="unverified",
        mcmc_prior_provenance="oof_5fold_recording_prefix_grouped_cv",
        val_loss=1., best_val_loss=1.,
    )
    if phase1_epochs is not None:
        state["training_phase"] = 1 if phase1_epochs > 0 else 2
    torch.save(state, resume)
    config.checkpoint.resume_from = str(resume)

    bad_path = None
    if candidate in ("resume", "source_best"):
        state["config"]["model"]["dt_ms"] = 8.
        torch.save(state, resume)
        bad_path = resume
    elif candidate == "companion_best":
        bad_state = dict(state, config={"model": {"dt_ms": 8.}})
        torch.save(bad_state, source / "best_model.pth")
        bad_path = source / "best_model.pth"
    elif candidate == "destination_best":
        output = Path(config.checkpoint.output_dir)
        output.mkdir()
        bad_state = dict(state, config={"model": {"dt_ms": 8.}})
        torch.save(bad_state, output / "best_model.pth")
        bad_path = output / "best_model.pth"

    def forbidden_epoch(*args: object, **kwargs: object) -> None:
        pytest.fail("Clock preflight reached an epoch or evaluation")

    from copy import deepcopy

    captured: dict = {}
    real_build = trainer.build_model
    real_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    real_clock_guard = trainer.validate_checkpoint_clock
    real_restore = trainer._canonical_load_checkpoint
    restores: list[bool] = []

    def capture_model(cfg: object) -> torch.nn.Module:
        model = real_build(cfg)
        captured["model"] = model
        return model

    def capture_scheduler(*args: object, **kwargs: object) -> object:
        scheduler = real_scheduler(*args, **kwargs)
        captured["scheduler"] = scheduler
        return scheduler

    def snapshot_before_guard(*args: object, **kwargs: object) -> None:
        if "weights" not in captured:
            model = captured["model"]
            scheduler = captured["scheduler"]
            captured["weights"] = {
                key: value.clone() for key, value in model.state_dict().items()
            }
            captured["freeze"] = [p.requires_grad for p in model.parameters()]
            captured["optimizer_state"] = deepcopy(scheduler.optimizer.state_dict())
            captured["scheduler_state"] = deepcopy(scheduler.state_dict())
            captured["rng"] = torch.get_rng_state().clone()
        real_clock_guard(*args, **kwargs)

    def record_restore(*args: object, **kwargs: object) -> dict:
        restores.append(True)
        return real_restore(*args, **kwargs)

    monkeypatch.setattr(trainer, "build_model", capture_model)
    monkeypatch.setattr(torch.optim.lr_scheduler, "CosineAnnealingLR", capture_scheduler)
    monkeypatch.setattr(trainer, "validate_checkpoint_clock", snapshot_before_guard)
    monkeypatch.setattr(trainer, "_canonical_load_checkpoint", record_restore)
    monkeypatch.setattr(trainer, "train_one_epoch", forbidden_epoch)
    monkeypatch.setattr(trainer, "validate", forbidden_epoch)
    monkeypatch.setattr(trainer, "compute_metrics", forbidden_epoch)
    with pytest.raises(ValueError, match="dt_ms.*conflicts"):
        trainer.train(
            config, dataset_path=str(dataset_path), phase1_epochs=phase1_epochs,
        )
    assert restores == []
    for key, expected in captured["weights"].items():
        assert torch.equal(captured["model"].state_dict()[key], expected), key
    assert [p.requires_grad for p in captured["model"].parameters()] == captured["freeze"]
    scheduler = captured["scheduler"]
    assert scheduler.optimizer.state_dict() == captured["optimizer_state"]
    assert scheduler.state_dict() == captured["scheduler_state"]
    assert torch.equal(torch.get_rng_state(), captured["rng"])
    assert bad_path is not None and bad_path.exists()
    assert not (Path(config.checkpoint.output_dir) / "final_model.pth").exists()


def test_clockless_generic_legacy_checkpoint_remains_loadable(tmp_path: Path) -> None:
    from nsmor.checkpoint import load_checkpoint

    saved, active = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    path = tmp_path / "generic-legacy.pth"
    torch.save({
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "model_state_dict": saved.state_dict(),
    }, path)
    load_checkpoint(path, active, map_location="cpu")
    for key, expected in saved.state_dict().items():
        assert torch.equal(active.state_dict()[key], expected)


class _GeneratedArtifactReducer:
    def __init__(self, marker: Path) -> None:
        self.marker = str(marker)

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({self.marker!r}).write_text('executed')",)


def test_load_checkpoint_rejects_generated_reducer_without_side_effect(tmp_path: Path) -> None:
    """A mutable generated checkpoint must never execute its pickle reducer."""
    from nsmor.checkpoint import load_checkpoint

    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "malicious.pth"
    marker = tmp_path / "reducer-executed"
    torch.save({
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "model_state_dict": model.state_dict(),
        "unexpected": _GeneratedArtifactReducer(marker),
    }, checkpoint)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    try:
        with pytest.raises((ValueError, pickle.UnpicklingError)):
            load_checkpoint(checkpoint, model, map_location="cpu")
    finally:
        assert not marker.exists(), "generated checkpoint executed its reducer"
    for key, expected in before.items():
        assert torch.equal(model.state_dict()[key], expected)


def test_atomic_promotion_rejects_replaced_generated_checkpoint(tmp_path: Path, monkeypatch) -> None:
    """A replaced temp checkpoint cannot execute or overwrite the prior target."""
    from scripts.train import _atomic_save_checkpoint

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    target = tmp_path / "checkpoint.pth"
    torch.save({"previous": True}, target)
    previous = target.read_bytes()
    marker = tmp_path / "reducer-executed"
    real_save = torch.save

    def replace_after_save(state, destination, *args, **kwargs):
        real_save(state, destination, *args, **kwargs)
        if Path(destination) == target.with_suffix(".pth.tmp") and "unexpected" not in state:
            real_save(dict(state, unexpected=_GeneratedArtifactReducer(marker)), destination)

    monkeypatch.setattr(torch, "save", replace_after_save)
    try:
        with pytest.raises((ValueError, pickle.UnpicklingError)):
            _atomic_save_checkpoint(
                model=model, optimizer=optimizer, epoch=0, loss=1.0,
                config={}, path=target, target_mean=0.0,
            )
    finally:
        assert not marker.exists(), "atomic promotion executed a generated reducer"
    assert target.read_bytes() == previous


@pytest.mark.parametrize("artifact", ["resume", "companion_best", "destination_best"])
def test_train_rejects_generated_checkpoint_reducers_before_resume(
    tmp_path: Path, artifact: str,
) -> None:
    """Every resume/best peek rejects a mutable generated reducer before training."""
    from nsmor.pipeline.nested_prior import compute_source_fingerprint
    from scripts.train import build_model, train

    dataset = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    source = tmp_path / "source"
    source.mkdir()
    resume = source / "epoch_1.pth"
    state = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "model_state_dict": build_model(config).state_dict(),
        "config": config.to_dict(),
        "epoch": 0, "loss": 1.0, "val_loss": 1.0, "best_val_loss": 1.0,
        "dataset_path": str(dataset),
        "dataset_source_sha256": compute_source_fingerprint(dataset),
        "is_nested_cv": False,
        "validation_scope": "diagnostic_global_oof",
    }
    torch.save(state, resume)
    config.checkpoint.resume_from = str(resume)
    marker = tmp_path / "reducer-executed"
    if artifact == "resume":
        attacked = resume
    elif artifact == "companion_best":
        attacked = source / "best_model.pth"
    else:
        output = Path(config.checkpoint.output_dir)
        output.mkdir()
        attacked = output / "best_model.pth"
    torch.save(dict(state, unexpected=_GeneratedArtifactReducer(marker)), attacked)
    captured = attacked.read_bytes()
    try:
        with pytest.raises((ValueError, pickle.UnpicklingError), match="global"):
            train(config, dataset_path=str(dataset),
                  trusted_historical_checkpoint_sha256=_checkpoint_shas(resume))
    finally:
        assert not marker.exists(), f"{artifact} executed its generated reducer"
    assert attacked.read_bytes() == captured


@pytest.mark.parametrize("artifact", ["resume", "companion_best", "destination_best"])
def test_resume_uses_captured_checkpoint_after_same_path_swap(
    tmp_path: Path, monkeypatch, artifact: str,
) -> None:
    """Lineage inspection, restored tensors and best promotion share captured bytes."""
    import io
    from nsmor.pipeline.nested_prior import compute_source_fingerprint, load_artifact_bytes
    from scripts.train import build_model, train

    dataset = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    config.training.learning_rate = 0.0
    source = tmp_path / "source"
    source.mkdir()
    resume = source / "epoch_1.pth"
    companion = source / "best_model.pth"
    expected = {key: value.clone() for key, value in build_model(config).state_dict().items()}
    state = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "model_state_dict": expected, "config": config.to_dict(),
        "epoch": 0, "loss": 0.0, "val_loss": 0.0, "best_val_loss": 0.0,
        "dataset_path": str(dataset),
        "dataset_source_sha256": compute_source_fingerprint(dataset),
        "is_nested_cv": False,
        "validation_scope": "diagnostic_global_oof",
    }
    torch.save(state, resume)
    torch.save(state, companion)
    config.checkpoint.resume_from = str(resume)
    if artifact == "destination_best":
        destination = Path(config.checkpoint.output_dir)
        destination.mkdir()
        attacked = destination / "best_model.pth"
        torch.save(dict(state, fixture_role="destination"), attacked)
    else:
        attacked = resume if artifact == "resume" else companion
    captured = attacked.read_bytes()
    expected_best = captured if artifact == "destination_best" else companion.read_bytes()
    replacement = dict(state, dataset_source_sha256="f" * 64,
                       model_state_dict={key: value + 0.5 for key, value in expected.items()})
    buffer = io.BytesIO()
    torch.save(replacement, buffer)
    replacement_bytes = buffer.getvalue()
    trusted = _checkpoint_shas(resume, companion, *([attacked] if artifact == "destination_best" else []))
    real_load = torch.load
    swaps = []

    def replace_after_decode(stream, *args, **kwargs):
        loaded = real_load(stream, *args, **kwargs)
        if isinstance(stream, io.BytesIO) and stream.getvalue() == captured and not swaps:
            attacked.write_bytes(replacement_bytes)
            swaps.append(True)
        return loaded

    monkeypatch.setattr(torch, "load", replace_after_decode)
    train(config, dataset_path=str(dataset),
          trusted_historical_checkpoint_sha256=trusted)
    assert swaps == [True]
    output = Path(config.checkpoint.output_dir)
    assert (output / "best_model.pth").read_bytes() == expected_best
    final = load_artifact_bytes((output / "final_model.pth").read_bytes(), map_location="cpu")
    assert final["dataset_source_sha256"] == state["dataset_source_sha256"]
    for key, value in expected.items():
        assert torch.equal(final["model_state_dict"][key], value), key


def test_lazy_metadata_snapshot_ignores_generated_reducer_after_read(tmp_path: Path, monkeypatch) -> None:
    """One captured metadata object supplies rows, conditions and lineage after a path swap."""
    import io
    from scripts.train import build_dataloaders

    config = _make_config(tmp_path)
    metadata = tmp_path / "metadata.pt"
    state = {
        "trial_specs": [{"session_id": f"recording{i}_session_1"} for i in range(5)],
        "mcmc_priors": torch.full((5, 4), 0.25),
        "stimulus_conditions": ["visual_only"] * 5,
        "dt_ms": 4.006,
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
    }
    torch.save(state, metadata)
    captured = metadata.read_bytes()
    marker = tmp_path / "reducer-executed"
    real_load = torch.load
    replaced = []

    def replace_after_initial_metadata_read(stream, *args, **kwargs):
        loaded = real_load(stream, *args, **kwargs)
        if isinstance(stream, io.BytesIO) and stream.getvalue() == captured and not replaced:
            torch.save(dict(state, unexpected=_GeneratedArtifactReducer(marker)), metadata)
            replaced.append(True)
        return loaded

    monkeypatch.setattr(torch, "load", replace_after_initial_metadata_read)
    try:
        train_loader, val_loader = build_dataloaders(
            config, dataset_path=str(metadata), use_lazy_loading=True
        )
        assert train_loader.dataset.prior_lineage == (
            "oof_2fold_recording_prefix_grouped_cv", "unverified"
        )
        assert train_loader.dataset.dataset.trial_specs == state["trial_specs"]
        assert len(train_loader.dataset) + len(val_loader.dataset) == 5
        assert replaced == [True]
        from nsmor.pipeline.nested_prior import load_artifact_bytes
        with pytest.raises((ValueError, pickle.UnpicklingError), match="global"):
            load_artifact_bytes(metadata.read_bytes())
    finally:
        assert not marker.exists(), "replaced metadata executed its reducer"


@pytest.mark.parametrize("dt_ms", [4.006, 10.0])
@pytest.mark.parametrize("condition_key", ["is_pure_wind", "stimulus_conditions"])
def test_lazy_metadata_fallback_preserves_conditions_and_physical_cadence(
    tmp_path: Path, dt_ms: float, condition_key: str,
) -> None:
    from scripts.train import build_dataloaders

    config = _make_config(tmp_path)
    config.model.dt_ms = dt_ms
    flags = np.asarray([True, False, True, False, True], dtype=bool)
    metadata = tmp_path / "metadata.pt"
    state = {
        "trial_specs": [{"session_id": f"recording{i}_session_1"} for i in range(5)],
        "mcmc_priors": torch.full((5, 4), 0.25),
        "dt_ms": dt_ms,
        "feature_config": FeatureConfig(),
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
    }
    state[condition_key] = (flags if condition_key == "is_pure_wind"
                            else ["wind_only" if flag else "visual_only" for flag in flags])
    torch.save(state, metadata)
    loaders = build_dataloaders(config, dataset_path=str(metadata), use_lazy_loading=True)
    for loader in loaders:
        assert loader is not None
        subset = loader.dataset
        np.testing.assert_array_equal(subset.is_pure_wind, flags[subset.indices])
        assert subset.dataset.dt_ms == dt_ms
        assert config.model.dt_ms == dt_ms


def test_checkpoint_safe_decode_preserves_config_dtypes_and_deterministic_state(tmp_path: Path) -> None:
    """Restricted loading keeps allowed config objects, NumPy dtypes and resume state."""
    from nsmor.checkpoint import load_checkpoint, save_checkpoint
    from nsmor.config import TimeWindowConfig

    model = torch.nn.Linear(2, 1).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    inputs = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    outputs = model(inputs)
    assert outputs.shape == (1, 1)
    outputs.square().sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    dtypes = {
        "bool": [True, False], "int8": [-1, 2], "int16": [-1, 2],
        "int32": [-1, 2], "int64": [-1, 2], "uint8": [1, 2],
        "uint16": [1, 2], "uint32": [1, 2], "uint64": [1, 2],
        "float16": [1.25, -2.5], "float32": [1.25, -2.5],
        "float64": [1.25, -2.5], "complex64": [1+2j, 3-4j],
        "complex128": [1+2j, 3-4j], "str": ["visual", "wind"],
        "bytes": [b"visual", b"wind"], "object": ["wind", 4.006],
    }
    arrays = {name: np.asarray(values, dtype=name) for name, values in dtypes.items()}
    config = {
        "model": {"dt_ms": 4.006, "hidden_dim": 16},
        "feature_config": FeatureConfig(),
        "time_window_config": TimeWindowConfig(frame_interval_ms=4.006),
        "arrays": arrays,
    }
    checkpoint = tmp_path / "checkpoint.pth"
    save_checkpoint(model, optimizer, epoch=3, loss=1.25, config=config,
                    path=checkpoint, scheduler=scheduler, train_loss=1.5, val_loss=1.25)
    saved_model = {key: value.clone() for key, value in model.state_dict().items()}
    saved_optimizer = optimizer.state_dict()
    saved_scheduler = scheduler.state_dict()
    saved_rng = torch.get_rng_state().clone()
    torch.rand(7)
    restored_model = torch.nn.Linear(2, 1).double()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.9)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=7)
    mapped = []

    def map_to_cpu(storage, location):
        mapped.append(location)
        return storage.cpu()

    loaded = load_checkpoint(checkpoint, restored_model, restored_optimizer,
                             restored_scheduler, map_location=map_to_cpu)
    assert mapped and set(mapped) == {"cpu"}
    assert loaded["epoch"] == 3 and loaded["loss"] == 1.25
    assert loaded["train_loss"] == 1.5 and loaded["val_loss"] == 1.25
    assert loaded["config"]["model"] == {"dt_ms": 4.006, "hidden_dim": 16}
    assert loaded["config"]["feature_config"] == FeatureConfig()
    assert loaded["config"]["time_window_config"] == TimeWindowConfig(frame_interval_ms=4.006)
    for name, expected in arrays.items():
        actual = loaded["config"]["arrays"][name]
        assert actual.shape == (2,) and actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    for key, expected in saved_model.items():
        assert restored_model.state_dict()[key].shape == expected.shape
        assert restored_model.state_dict()[key].dtype == torch.float64
        assert torch.equal(restored_model.state_dict()[key], expected)
    actual_optimizer = restored_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == saved_optimizer["param_groups"]
    for parameter, expected in saved_optimizer["state"].items():
        for key, value in expected.items():
            assert torch.equal(actual_optimizer["state"][parameter][key], value)
    assert restored_scheduler.state_dict() == saved_scheduler
    assert torch.equal(torch.get_rng_state(), saved_rng)


@pytest.mark.parametrize("version", [None, "0.0"])
def test_checkpoint_safe_decode_preserves_semantics_guard(tmp_path: Path, version: Optional[str]) -> None:
    from nsmor.checkpoint import load_checkpoint

    model = torch.nn.Linear(2, 1)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = {"model_state_dict": {key: value + 1.0 for key, value in before.items()}}
    if version is not None:
        state["pipeline_semantics_version"] = version
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match="semantics|pipeline_semantics_version"):
        load_checkpoint(checkpoint, model, map_location=torch.device("cpu"))
    for key, expected in before.items():
        assert torch.equal(model.state_dict()[key], expected)
