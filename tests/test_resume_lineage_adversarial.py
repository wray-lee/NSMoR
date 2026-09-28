"""Focused adversarial negative tests for checkpoint resume lineage and historical best preservation.

Must verify that current/fixed code fails closed on:
1. Renamed periodic checkpoint as best_model.pth (must NOT trust filename or use train loss).
2. Missing provenance keys in resume checkpoint (fail closed).
3. Mismatched nested_prior_artifact path in resume checkpoint (fail closed).
4. Wrong destination best_model.pth in output_dir (fail closed).
5. Absent companion historical best_model.pth when resuming periodic checkpoint into fresh output_dir (fail closed).
6. When companion best_model.pth IS available, preserved weights and score in fresh output_dir
   match the historical best, NOT the periodic resume checkpoint weights.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION, FeatureConfig
from nsmor.config_parser import ExperimentConfig
from nsmor.pipeline.grouping import animal_of as _animal_of
from nsmor.pipeline.nested_prior import compute_source_fingerprint
from scripts.train import train

_HIDDEN = 16
_N_ANIMALS = 6
_TRIALS_PER_ANIMAL = 4
_N_TOTAL = _N_ANIMALS * _TRIALS_PER_ANIMAL
_SEQ_LEN = 30
_MCMC = 4
_FEAT = FeatureConfig()


def _make_dataset(tmp_path: Path, seed: int = 42) -> Path:
    rng = np.random.RandomState(seed)
    X_seqs = [rng.randn(_SEQ_LEN, _FEAT.per_frame_total_dim).astype(np.float32) for _ in range(_N_TOTAL)]
    for X in X_seqs:
        X[:, 4:] = 0.0
    Y_seqs = [rng.randn(_SEQ_LEN).astype(np.float32) for _ in range(_N_TOTAL)]
    labels = np.array([i % _MCMC for i in range(_N_TOTAL)], dtype=np.int64)
    lengths = np.full(_N_TOTAL, _SEQ_LEN, dtype=np.int64)
    session_ids = [
        f"0.{500 + (i // _TRIALS_PER_ANIMAL)}cricket_001_20260101_0000{i // _TRIALS_PER_ANIMAL}_session_{1 + (i % _TRIALS_PER_ANIMAL)}"
        for i in range(_N_TOTAL)
    ]
    snapshots = rng.randn(_N_TOTAL, _FEAT.snapshot_dim).astype(np.float64)
    logits = rng.randn(_N_TOTAL, _MCMC).astype(np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    priors = exp / exp.sum(axis=1, keepdims=True)

    ds = {
        "X_seqs": X_seqs,
        "Y_seqs": Y_seqs,
        "labels": labels,
        "lengths": lengths,
        "mcmc_priors": priors,
        "mcmc_prior_provenance": "oof_5fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
        "session_ids": session_ids,
        "snapshots": snapshots,
        "feature_config": _FEAT,
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
    }
    p = tmp_path / "dataset.pt"
    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.save(ds, p)
    return p


def _make_nested_artifact(ds_path: Path, split_seed: int = 3, val_split: float = 0.25) -> Path:
    from nsmor.pipeline.grouping import grouped_train_val_split

    ds = torch.load(ds_path, weights_only=False)
    session_ids = ds["session_ids"]
    n_total = len(session_ids)
    train_idx, val_idx = grouped_train_val_split(session_ids, n_total, val_split=val_split, random_seed=split_seed)
    rng = np.random.RandomState(split_seed)
    logits = rng.randn(n_total, _MCMC)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    priors = exp / exp.sum(axis=1, keepdims=True)

    prefixes = np.array([_animal_of(s) for s in session_ids], dtype=object)
    train_prefixes = sorted(set(prefixes[train_idx].tolist()))
    val_prefixes = sorted(set(prefixes[val_idx].tolist()))
    art = {
        "recording_prefix_keys": prefixes,
        "train_recording_prefixes": train_prefixes,
        "val_recording_prefixes": val_prefixes,
        "n_train_recording_prefixes": len(train_prefixes),
        "n_val_recording_prefixes": len(val_prefixes),
        "n_inner_folds": 5,
        "animal_identity_status": "unverified",
        "nested_priors": priors,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "is_nested_cv": True,
        "source_fingerprint": compute_source_fingerprint(ds_path),
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "split_seed": split_seed,
        "val_split": val_split,
        "mcmc_prior_provenance": f"nested_outer_seed{split_seed}_inner_5fold_recording_prefix_grouped",
    }
    out = ds_path.parent / f"nested_seed{split_seed}.pt"
    torch.save(art, out)
    return out


def _make_config(out_dir: Path, epochs: int = 1, resume: str | None = None) -> ExperimentConfig:
    c = ExperimentConfig()
    c.model.hidden_dim = 8
    c.model.num_gru_layers = 1
    c.model.dropout = 0.0
    c.training.num_epochs = epochs
    c.training.batch_size = 16
    c.training.max_seq_len = 24
    c.training.num_workers = 0
    c.training.normalize_targets = False
    c.training.target_clip_cm_s = 0.0
    c.training.lr_warmup_epochs = 0
    c.training.checkpoint_interval = 1
    c.loss.warmup_epochs = 0
    c.loss.lambda_routing_aux = 0.0
    c.checkpoint.output_dir = str(out_dir)
    c.checkpoint.resume_from = resume
    return c


def test_resume_renamed_periodic_checkpoint_does_not_trust_loss(tmp_path: Path):
    """Renaming periodic checkpoint to best_model.pth must NOT cause train_loss to become best_val_loss."""
    ds_path = _make_dataset(tmp_path)
    art_path = _make_nested_artifact(ds_path, split_seed=3)

    c1 = _make_config(tmp_path / "run1", epochs=1)
    train(c1, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))
    epoch1_ckpt = tmp_path / "run1" / "epoch_1.pth"

    # Strip val_loss metadata, leave only train loss in ckpt['loss'] = 0.01
    ckpt_data = torch.load(epoch1_ckpt, weights_only=False)
    ckpt_data.pop("val_loss", None)
    ckpt_data.pop("best_val_loss", None)
    ckpt_data["loss"] = 0.01  # small train loss
    renamed_best = tmp_path / "run1" / "fake_best_model.pth"
    torch.save(ckpt_data, renamed_best)

    # Resume into fresh dir from renamed_best (where companion best is removed)
    (tmp_path / "run1" / "best_model.pth").unlink(missing_ok=True)
    c2 = _make_config(tmp_path / "run2", epochs=2, resume=str(renamed_best))
    # Must either reject absent validation metadata fail-closed or initialize best_val_loss to inf
    # (NEVER adopting 0.01 train loss!)
    try:
        res2 = train(c2, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))
        assert res2["best_val_loss"] > 0.01, "Train loss was adopted as best_val_loss!"
    except ValueError as e:
        assert "absent" in str(e).lower() or "fail closed" in str(e).lower()

    # Exact trusted name spoofing: file literally named 'best_model.pth' lacking finite validation loss
    exact_renamed_dir = tmp_path / "exact_renamed"
    exact_renamed_dir.mkdir()
    exact_best = exact_renamed_dir / "best_model.pth"
    torch.save(ckpt_data, exact_best)
    c_exact = _make_config(tmp_path / "run_exact", epochs=2, resume=str(exact_best))
    with pytest.raises(ValueError, match="validation|fail closed|best"):
        train(c_exact, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))


def test_resume_missing_provenance_keys_refused(tmp_path: Path):
    """Resume checkpoint missing mandatory nested provenance keys must fail closed."""
    ds_path = _make_dataset(tmp_path)
    art_path = _make_nested_artifact(ds_path, split_seed=3)

    c1 = _make_config(tmp_path / "run1", epochs=1)
    train(c1, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))
    ckpt_path = tmp_path / "run1" / "epoch_1.pth"

    ckpt_data = torch.load(ckpt_path, weights_only=False)
    ckpt_data.pop("nested_prior_artifact", None)  # delete mandatory key
    torch.save(ckpt_data, ckpt_path)

    c2 = _make_config(tmp_path / "run2", epochs=2, resume=str(ckpt_path))
    with pytest.raises(ValueError, match="missing mandatory nested provenance|fail closed"):
        train(c2, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))


def test_resume_different_artifact_path_refused(tmp_path: Path):
    """Resume checkpoint with different nested_prior_artifact path must fail closed."""
    ds_path = _make_dataset(tmp_path)
    art1_path = _make_nested_artifact(ds_path, split_seed=3)
    art2_path = tmp_path / "other_artifact.pt"
    import shutil
    shutil.copy2(art1_path, art2_path)

    c1 = _make_config(tmp_path / "run1", epochs=1)
    train(c1, dataset_path=str(ds_path), nested_prior_artifact=str(art1_path))
    ckpt_path = tmp_path / "run1" / "epoch_1.pth"

    c2 = _make_config(tmp_path / "run2", epochs=2, resume=str(ckpt_path))
    with pytest.raises(ValueError, match="nested prior artifact mismatch|fail closed"):
        train(c2, dataset_path=str(ds_path), nested_prior_artifact=str(art2_path))


def test_resume_wrong_destination_best_model_refused(tmp_path: Path):
    """Resuming into an output_dir that already contains a best_model.pth with wrong lineage must fail closed."""
    ds_path = _make_dataset(tmp_path)
    art_path = _make_nested_artifact(ds_path, split_seed=3)

    c1 = _make_config(tmp_path / "run1", epochs=1)
    train(c1, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))
    ckpt_path = tmp_path / "run1" / "epoch_1.pth"

    # Pre-populate run2 with a corrupt / mismatched destination best_model.pth
    run2_dir = tmp_path / "run2"
    run2_dir.mkdir(parents=True)
    wrong_dest_best = run2_dir / "best_model.pth"
    wrong_data = torch.load(tmp_path / "run1" / "best_model.pth", weights_only=False)
    wrong_data["nested_split_seed"] = 999  # wrong split seed in destination
    torch.save(wrong_data, wrong_dest_best)

    c2 = _make_config(run2_dir, epochs=2, resume=str(ckpt_path))
    with pytest.raises(ValueError, match="Existing destination|mismatched lineage|fail closed"):
        train(c2, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))


def test_resume_periodic_absent_historical_best_refused(tmp_path: Path):
    """Resuming periodic checkpoint into fresh directory without companion best_model.pth must fail closed."""
    ds_path = _make_dataset(tmp_path)
    art_path = _make_nested_artifact(ds_path, split_seed=3)

    c1 = _make_config(tmp_path / "run1", epochs=1)
    train(c1, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))
    epoch1_ckpt = tmp_path / "run1" / "epoch_1.pth"

    # Delete companion best_model.pth from run1
    (tmp_path / "run1" / "best_model.pth").unlink(missing_ok=True)

    c2 = _make_config(tmp_path / "run2", epochs=2, resume=str(epoch1_ckpt))
    with pytest.raises(ValueError, match="absent historical best|unavailable|fail closed"):
        train(c2, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))


def test_resume_preserves_actual_best_weights_not_periodic_weights(tmp_path: Path):
    """When best and periodic weights differ, fresh output_dir must receive the ACTUAL best weights,
    not the periodic resume checkpoint weights."""
    ds_path = _make_dataset(tmp_path)
    art_path = _make_nested_artifact(ds_path, split_seed=3)

    run1_dir = tmp_path / "run1"
    c1 = _make_config(run1_dir, epochs=2)
    # Train 2 epochs: force epoch 1 to be best (e.g. val_loss 0.5), epoch 2 to be worse (e.g. val_loss 0.8)
    val_losses = [0.5, 0.8]
    val_idx = 0

    def _mock_validate(*args, **kwargs):
        nonlocal val_idx
        v = val_losses[min(val_idx, len(val_losses) - 1)]
        val_idx += 1
        return v

    with mock.patch("scripts.train.validate", side_effect=_mock_validate):
        train(c1, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))

    best1 = torch.load(run1_dir / "best_model.pth", weights_only=False)
    epoch2 = torch.load(run1_dir / "epoch_2.pth", weights_only=False)

    # Prove weights differ
    p_best = next(iter(best1["model_state_dict"].values()))
    p_epoch2 = next(iter(epoch2["model_state_dict"].values()))
    assert not torch.equal(p_best, p_epoch2), "Test precondition failed: best and epoch_2 weights must differ"
    assert best1["best_val_loss"] == pytest.approx(0.5)

    # Resume from epoch_2.pth into fresh run2_dir where epoch 3 val_loss is worse (0.9)
    run2_dir = tmp_path / "run2"
    c2 = _make_config(run2_dir, epochs=3, resume=str(run1_dir / "epoch_2.pth"))

    with mock.patch("scripts.train.validate", return_value=0.9):
        res2 = train(c2, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))

    # Output best_model.pth must match historical best (0.5) across ALL state tensors, NOT epoch_2
    res2_best = torch.load(run2_dir / "best_model.pth", weights_only=False)
    mismatched_best = [
        k for k, v in best1["model_state_dict"].items()
        if not torch.equal(v, res2_best["model_state_dict"][k])
    ]
    assert not mismatched_best, (
        f"Preserved weights do NOT match historical best: {len(mismatched_best)} tensors differ"
    )
    diff_from_periodic = [
        k for k, v in epoch2["model_state_dict"].items()
        if not torch.equal(v, res2_best["model_state_dict"][k])
    ]
    assert len(diff_from_periodic) == len(best1["model_state_dict"]), (
        "Preserved weights incorrectly match periodic checkpoint tensors!"
    )
    assert res2["best_val_loss"] == pytest.approx(0.5)
    assert res2["eval_provenance"] == "best"


def test_resume_destination_periodic_weights_refused(tmp_path: Path) -> None:
    """Existing destination best_model.pth with periodic (non-best) weights must fail closed."""
    ds_path = _make_dataset(tmp_path)
    art_path = _make_nested_artifact(ds_path, split_seed=3)

    run1_dir = tmp_path / "run1"
    c1 = _make_config(run1_dir, epochs=2)
    with mock.patch("scripts.train.validate", side_effect=[0.5, 0.8]):
        train(c1, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))

    epoch2 = torch.load(run1_dir / "epoch_2.pth", weights_only=False)
    assert epoch2["best_val_loss"] == pytest.approx(0.5)
    assert epoch2["val_loss"] == pytest.approx(0.8)

    # Pre-populate run2 with epoch2 (non-best weights) as best_model.pth
    run2_dir = tmp_path / "run2"
    run2_dir.mkdir(parents=True)
    torch.save(epoch2, run2_dir / "best_model.pth")

    c2 = _make_config(run2_dir, epochs=3, resume=str(run1_dir / "epoch_2.pth"))
    with pytest.raises(ValueError, match="not a certified best model|differs from best_val_loss|fail closed"):
        train(c2, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))


def test_resume_destination_nonfinite_metrics_refused(tmp_path: Path) -> None:
    """Existing destination best_model.pth with NaN validation metrics must fail closed."""
    ds_path = _make_dataset(tmp_path)
    art_path = _make_nested_artifact(ds_path, split_seed=3)

    run1_dir = tmp_path / "run1"
    c1 = _make_config(run1_dir, epochs=1)
    train(c1, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))

    best1 = torch.load(run1_dir / "best_model.pth", weights_only=False)
    best1["val_loss"] = float("nan")
    best1["best_val_loss"] = float("nan")

    run2_dir = tmp_path / "run2"
    run2_dir.mkdir(parents=True)
    torch.save(best1, run2_dir / "best_model.pth")

    c2 = _make_config(run2_dir, epochs=2, resume=str(run1_dir / "epoch_1.pth"))
    with pytest.raises(ValueError, match="lacks finite OWN validation loss|fail closed"):
        train(c2, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))


def test_resume_no_remaining_epochs_refused(tmp_path: Path) -> None:
    """Resuming when target num_epochs <= start_epoch must fail closed."""
    ds_path = _make_dataset(tmp_path)
    art_path = _make_nested_artifact(ds_path, split_seed=3)

    run1_dir = tmp_path / "run1"
    c1 = _make_config(run1_dir, epochs=2)
    train(c1, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))

    run2_dir = tmp_path / "run2"
    # Target epochs=2, but epoch_2.pth resumes at start_epoch=2 (0 remaining epochs)
    c2 = _make_config(run2_dir, epochs=2, resume=str(run1_dir / "epoch_2.pth"))
    with pytest.raises(ValueError, match="no remaining epochs to train|fail closed"):
        train(c2, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))


def test_resume_destination_worse_reconciled_with_better_source(tmp_path: Path) -> None:
    """A genuine-but-worse destination best_model.pth must be replaced by a superior companion source best model."""
    ds_path = _make_dataset(tmp_path)
    art_path = _make_nested_artifact(ds_path, split_seed=3)

    run1_dir = tmp_path / "run1"
    c1 = _make_config(run1_dir, epochs=2)
    with mock.patch("scripts.train.validate", side_effect=[0.5, 0.8]):
        train(c1, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))

    best1 = torch.load(run1_dir / "best_model.pth", weights_only=False)
    assert best1["best_val_loss"] == pytest.approx(0.5)

    # Pre-populate run2 with a genuine best_model.pth having worse val_loss (1.2)
    run2_dir = tmp_path / "run2"
    run2_dir.mkdir(parents=True)
    worse_best = torch.load(run1_dir / "best_model.pth", weights_only=False)
    worse_best["val_loss"] = 1.2
    worse_best["best_val_loss"] = 1.2
    torch.save(worse_best, run2_dir / "best_model.pth")

    # Resume from epoch_2 in run1 (claimed best 0.5) into run2
    c2 = _make_config(run2_dir, epochs=3, resume=str(run1_dir / "epoch_2.pth"))
    with mock.patch("scripts.train.validate", return_value=0.9):
        res2 = train(c2, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))

    assert res2["best_val_loss"] == pytest.approx(0.5)
    selected = torch.load(run2_dir / "best_model.pth", weights_only=False)
    mismatched = [
        k for k, v in best1["model_state_dict"].items()
        if not torch.equal(v, selected["model_state_dict"][k])
    ]
    assert not mismatched, f"Expected superior source weights to replace worse destination; {len(mismatched)} differ"
    assert selected["val_loss"] == pytest.approx(0.5)


def test_resume_destination_worse_refuses_silent_degradation_without_source(tmp_path: Path) -> None:
    """When destination is worse than claimed historical best and source companion is missing, refuse fail closed."""
    ds_path = _make_dataset(tmp_path)
    art_path = _make_nested_artifact(ds_path, split_seed=3)

    run1_dir = tmp_path / "run1"
    c1 = _make_config(run1_dir, epochs=2)
    with mock.patch("scripts.train.validate", side_effect=[0.5, 0.8]):
        train(c1, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))

    # Pre-populate run2 with a worse best_model.pth
    run2_dir = tmp_path / "run2"
    run2_dir.mkdir(parents=True)
    worse_best = torch.load(run1_dir / "best_model.pth", weights_only=False)
    worse_best["val_loss"] = 1.2
    worse_best["best_val_loss"] = 1.2
    torch.save(worse_best, run2_dir / "best_model.pth")

    # Remove companion best from run1 so 0.5 is unavailable
    (run1_dir / "best_model.pth").unlink()

    c2 = _make_config(run2_dir, epochs=3, resume=str(run1_dir / "epoch_2.pth"))
    with pytest.raises(ValueError, match="unavailable|degrade|fail closed"):
        train(c2, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))


def test_resume_non_nested_mismatched_dataset_refused(tmp_path: Path) -> None:
    """Non-nested checkpoint resumed with mismatched dataset_path must fail closed."""
    ds_path1 = _make_dataset(tmp_path / "ds1")
    ds_path2 = _make_dataset(tmp_path / "ds2")

    run1_dir = tmp_path / "run1"
    c1 = _make_config(run1_dir, epochs=1)
    train(c1, dataset_path=str(ds_path1))

    run2_dir = tmp_path / "run2"
    c2 = _make_config(run2_dir, epochs=2, resume=str(run1_dir / "epoch_1.pth"))
    with pytest.raises(ValueError, match="mismatched dataset_path|fail closed"):
        train(c2, dataset_path=str(ds_path2))


def test_resume_nan_val_split_refused(tmp_path: Path) -> None:
    """Checkpoint with NaN nested_val_split must fail closed."""
    ds_path = _make_dataset(tmp_path)
    art_path = _make_nested_artifact(ds_path, split_seed=3)

    run1_dir = tmp_path / "run1"
    c1 = _make_config(run1_dir, epochs=1)
    train(c1, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))

    ckpt = torch.load(run1_dir / "epoch_1.pth", weights_only=False)
    ckpt["nested_val_split"] = float("nan")
    corrupt_path = run1_dir / "corrupt_val_split.pth"
    torch.save(ckpt, corrupt_path)

    c2 = _make_config(tmp_path / "run2", epochs=2, resume=str(corrupt_path))
    with pytest.raises(ValueError, match="mismatched or non-finite nested val_split|fail closed"):
        train(c2, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))


def test_resume_fractional_split_seed_refused(tmp_path: Path) -> None:
    """Checkpoint with fractional nested_split_seed (e.g. 3.5) must fail closed."""
    ds_path = _make_dataset(tmp_path)
    art_path = _make_nested_artifact(ds_path, split_seed=3)

    run1_dir = tmp_path / "run1"
    c1 = _make_config(run1_dir, epochs=1)
    train(c1, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))

    ckpt = torch.load(run1_dir / "epoch_1.pth", weights_only=False)
    ckpt["nested_split_seed"] = 3.5
    corrupt_path = run1_dir / "fractional_seed.pth"
    torch.save(ckpt, corrupt_path)

    c2 = _make_config(tmp_path / "run2", epochs=2, resume=str(corrupt_path))
    with pytest.raises(ValueError, match="fractional nested split_seed|fail closed"):
        train(c2, dataset_path=str(ds_path), nested_prior_artifact=str(art_path))





def test_checkpoint_prior_lineage_refuses_identity_upgrade(tmp_path: Path):
    from scripts.train import _check_checkpoint_prior_lineage, _PROVENANCE_KEYS

    assert "animal_identity_status" in _PROVENANCE_KEYS
    modern = {
        "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
        "dataset_source_sha256": "a" * 64,
        "validation_scope": "diagnostic_global_oof",
    }
    _check_checkpoint_prior_lineage(modern, modern, tmp_path / "new.pth")
    with pytest.raises(ValueError, match="animal_identity_status"):
        _check_checkpoint_prior_lineage(
            dict(modern, animal_identity_status="verified"), modern, tmp_path / "forged.pth"
        )
    with pytest.raises(ValueError, match="animal_identity_status"):
        _check_checkpoint_prior_lineage(
            {"mcmc_prior_provenance": modern["mcmc_prior_provenance"]},
            modern, tmp_path / "unstamped.pth",
        )
    historical = {
        "mcmc_prior_provenance": "global_oof_animal_grouped_cv",
        "animal_identity_status": "historical_unknown",
    }
    def check_historical(record, expected, name):
        path = tmp_path / name
        torch.save(record, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _check_checkpoint_prior_lineage(
            record, expected, path, checkpoint_sha256=digest,
            trusted_historical_checkpoint_sha256=digest,
        )

    check_historical(
        {"mcmc_prior_provenance": historical["mcmc_prior_provenance"]},
        historical, "old.pth",
    )
    legacy_oof = {
        "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
        "animal_identity_status": "historical_unknown",
        "is_nested_cv": False,
    }
    check_historical({}, legacy_oof, "pre-lineage.pth")
    check_historical(
        {"mcmc_prior_provenance": "global_oof_animal_grouped_cv"},
        legacy_oof, "v4-provisional.pth",
    )
    with pytest.raises(ValueError, match="animal_identity_status"):
        _check_checkpoint_prior_lineage(
            {"animal_identity_status": "verified"}, legacy_oof, tmp_path / "forged-old.pth"
        )
