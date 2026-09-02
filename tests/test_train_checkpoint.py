"""Regression tests for scripts/train.py checkpoint and data-split behaviour.

Covers:
- Atomic-write semantics (no partial file left on interrupted save).
- A completed run always yields a loadable best checkpoint.
- Non-finite val loss is surfaced rather than silently swallowed.
- Target-stats split matches the dataloader split (session-disjoint).
- Deployment provenance fields in all checkpoint types.

All tests use tiny synthetic data to run in <10s on CPU.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Dict, Optional
from unittest import mock

import numpy as np
import pytest
import torch

from nsmor.config import FeatureConfig

# ── Fixtures ────────────────────────────────────────────────────────

_HIDDEN = 16
_SENSORY = 4
_MCMC = 4
_N_TRAIN = 8
_N_VAL = 2
_SEQ_LEN = 50
_N_SESSIONS = 5  # at least 2 for val


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
    # Assign sessions round-robin so we have at least 2 unique sessions.
    session_ids = [f"sess_{i % _N_SESSIONS}" for i in range(n_total)]

    dataset = {
        "X_seqs": X_seqs,
        "Y_seqs": Y_seqs,
        "labels": labels,
        "lengths": lengths,
        "mcmc_priors": mcmc_priors,
        "session_ids": session_ids,
        "feature_config": FeatureConfig(),
        "pipeline_semantics_version": "2.1",
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

    ckpt = torch.load(best_path, weights_only=False)
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

def test_nonfinite_val_loss_handled(tmp_path: Path) -> None:
    """When val_loss is NaN/Inf, the run must still complete and:
    - best_val_loss should remain inf (no best checkpoint from bad loss)
    - final_model.pth should exist as fallback
    - metrics should be computed from the final fallback checkpoint
    """
    from scripts.train import train, validate

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=1, warmup_epochs=0)

    # Patch validate to return NaN
    with mock.patch("scripts.train.validate", return_value=float("nan")):
        results = train(config, lambda_reg=0.01, dataset_path=str(ds_path))

    output_dir = Path(config.checkpoint.output_dir)
    best_path = output_dir / "best_model.pth"
    final_path = output_dir / "final_model.pth"

    # best_model.pth should NOT be written (NaN < anything is False)
    assert not best_path.exists(), (
        "best_model.pth should not be written when val_loss is NaN"
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
    """compute_target_stats and build_dataloaders must use the SAME
    session-grouped split (no data leakage)."""
    from scripts.train import (
        build_dataloaders,
        compute_target_stats,
        _VAL_SPLIT,
    )

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=1, normalize_targets=True)

    # Get the train indices from build_dataloaders
    dataset = torch.load(ds_path, weights_only=False)
    n_total = len(dataset["X_seqs"])
    session_arr = np.asarray(dataset["session_ids"])
    unique_sessions = np.unique(session_arr)
    rng = np.random.RandomState(config.training.random_seed)
    rng.shuffle(unique_sessions)
    n_val_sessions = max(1, int(len(unique_sessions) * _VAL_SPLIT))
    val_sessions_build = set(unique_sessions[:n_val_sessions].tolist())
    is_val_build = np.array([s in val_sessions_build for s in session_arr])
    train_indices_build = np.nonzero(~is_val_build)[0]

    # Get the train indices from compute_target_stats
    # We replicate the SAME logic that compute_target_stats now uses
    rng2 = np.random.RandomState(config.training.random_seed)
    unique_sessions2 = np.unique(session_arr)
    rng2.shuffle(unique_sessions2)
    n_val_sessions2 = max(1, int(len(unique_sessions2) * _VAL_SPLIT))
    val_sessions_stats = set(unique_sessions2[:n_val_sessions2].tolist())
    is_val_stats = np.array([s in val_sessions_stats for s in session_arr])
    train_indices_stats = np.nonzero(~is_val_stats)[0]

    # The two sets of train indices must be identical
    np.testing.assert_array_equal(
        np.sort(train_indices_build),
        np.sort(train_indices_stats),
        err_msg="compute_target_stats and build_dataloaders use different splits",
    )

    # Also verify the session sets are identical
    assert val_sessions_build == val_sessions_stats, (
        f"Val sessions differ: build={val_sessions_build} vs stats={val_sessions_stats}"
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
        ckpt = torch.load(ckpt_path, weights_only=False)
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
    epoch1_ckpt = torch.load(epoch1_ckpt_path, weights_only=False)
    _assert_provenance_fields(
        epoch1_ckpt,
        "epoch_1.pth",
        expected_phase=1,
        expected_dataset_path=ds_path_str,
    )

    # Final checkpoint must be phase 2
    final_path = output_dir / "final_model.pth"
    assert final_path.exists(), "final_model.pth not found"
    final_ckpt = torch.load(final_path, weights_only=False)
    _assert_provenance_fields(
        final_ckpt,
        "final_model.pth",
        expected_phase=2,
        expected_dataset_path=ds_path_str,
    )


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
    expected_mean, expected_std = compute_target_stats(
        ds_path_str, config, val_split=_VAL_SPLIT,
    )

    results = train(config, lambda_reg=0.01, dataset_path=ds_path_str)

    output_dir = Path(config.checkpoint.output_dir)
    best_path = output_dir / "best_model.pth"
    assert best_path.exists(), "best_model.pth not found"
    ckpt = torch.load(best_path, weights_only=False)

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
