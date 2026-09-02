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
    """``build_dataloaders`` and ``compute_target_stats`` must agree on the
    session-grouped split at SESSION granularity (no data leakage).

    Both real functions are called and nothing is replicated locally.  The
    previous version of this test compared two identically-seeded local
    copies of the split logic to each other, so it could not fail no matter
    what the production functions did.
    """
    from scripts.train import (
        build_dataloaders,
        compute_target_stats,
        _VAL_SPLIT,
    )

    ds_path = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=1, normalize_targets=True)
    dataset = torch.load(ds_path, weights_only=False)
    Y_ref = dataset["Y_seqs"]
    session_arr = np.asarray(dataset["session_ids"])
    all_sessions = set(np.unique(session_arr).tolist())

    def _sessions_of(loader) -> set:
        """Map a loader's sequences back to their source session ids."""
        found = set()
        for seq in loader.dataset.sequences:
            matches = [
                i for i in range(len(Y_ref))
                if np.array_equal(seq[1], Y_ref[i])
            ]
            assert len(matches) == 1, (
                f"expected a unique Y_seq match, got {len(matches)}"
            )
            found.add(session_arr[matches[0]])
        return found

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
    expected_mean, expected_std, _ = compute_target_stats(
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


# ═════════════════════════════════════════════════════════════
# Test 7: Gradient-isolation invariant across two-phase training
# ═════════════════════════════════════════════════════════════

def test_gradient_isolation_phase1_and_phase2() -> None:
    """ADR-0005 gradient isolation invariant across two-phase training.

    Phase 1: frontend is trainable, backend is frozen -> all backend
    parameters have None grad after forward+backward.
    Phase 2: frontend is frozen, backend is trainable -> all frontend
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
    dataset = torch.load(ds_path, weights_only=False)
    n_total = len(dataset["X_seqs"])

    config = _make_config(tmp_path, epochs=1, normalize_targets=True)

    # 1. Call build_dataloaders and extract actual train indices
    train_loader, val_loader = build_dataloaders(
        config, dataset_path=str(ds_path), val_split=_VAL_SPLIT,
    )
    assert train_loader is not None

    train_indices_from_loader = []
    for seq in train_loader.dataset.sequences:
        y_seq = seq[1]
        matched = [
            i for i in range(n_total)
            if np.array_equal(y_seq, dataset["Y_seqs"][i])
        ]
        assert len(matched) == 1
        train_indices_from_loader.append(matched[0])
    train_indices_from_loader = np.array(train_indices_from_loader)

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

    ckpt = torch.load(ckpt_path, weights_only=False)
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
    )

    assert np.isfinite(results_resume["best_val_loss"])
    assert "final_train_loss" in results_resume
    final_path = resume_output_dir / "final_model.pth"
    assert final_path.exists(), "final_model.pth was not written after resume"

    final_ckpt = torch.load(final_path, weights_only=False)
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

    ckpt = torch.load(ckpt_path, weights_only=False)
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
    final_ckpt = torch.load(final_path, weights_only=False)
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

    ckpt = torch.load(ckpt_path, weights_only=False)
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
    final_ckpt = torch.load(final_path, weights_only=False)
    assert final_ckpt["epoch"] == 16  # 0-indexed, 17 total epochs
    assert final_ckpt["training_phase"] == 2
    assert len(final_ckpt["optimizer_state_dict"]["param_groups"]) == 2


# ═════════════════════════════════════════════════════════════
# Test 11: train() itself enforces the ADR-0005 freeze schedule
# ═════════════════════════════════════════════════════════════

def test_train_enforces_phase_freeze_schedule(tmp_path: Path) -> None:
    """ADR-0005 enforcement point: ``train()`` — not the caller — must
    freeze the backend during Phase 1 and the frontend during Phase 2.

    ``test_gradient_isolation_phase1_and_phase2`` toggles ``requires_grad``
    by hand, so it proves the architecture *permits* isolation but would
    still pass if ``train()`` stopped freezing anything.  This test
    observes the real per-epoch parameter state inside ``train()``.
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



