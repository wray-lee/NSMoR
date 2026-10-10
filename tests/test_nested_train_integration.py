"""Regression tests for the opt-in nested-prior seam in ``scripts/train.py``.

Covers:
- Exact persisted train/val indices (no recomputed split) when the
  nested artifact is supplied.
- Nested priors land in feature channels 4-7 of the actual train dataset.
- Target remains continuous velocity (no classification target).
- Fail-closed refusals: wrong fingerprint, malformed indices, animal
  overlap, bad priors, version mismatch.
- Default legacy behavior (no artifact) is unchanged.
- Target-statistic split matches the nested train split.

All tests use tiny synthetic canonical datasets and crafted artifacts.
No real corpus ETL or training is performed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION, FeatureConfig
from nsmor.pipeline.grouping import animal_of as _animal_of
from nsmor.pipeline.nested_prior import (
    compute_source_fingerprint,
    load_artifact_bytes,
    load_nested_prior_split,
)

# ── Fixtures ────────────────────────────────────────────────────────

_HIDDEN = 16
_N_ANIMALS = 6
_TRIALS_PER_ANIMAL = 4
_N_TOTAL = _N_ANIMALS * _TRIALS_PER_ANIMAL
_SEQ_LEN = 30
_MCMC = 4
_FEAT = FeatureConfig()


def _make_canonical_dataset(tmp_path: Path, seed: int = 7) -> Path:
    """Create a synthetic canonical dataset with animal-grouped session ids.

    Mirrors the real schema closely enough for ``validate_dataset_provenance``
    and ``build_dataloaders``: X is (T, 8) with zeroed prior slots (the
    dataloader fills 4-7), Y is continuous 1-D velocity, priors are a
    probability simplex, and every animal owns multiple sessions.
    """
    rng = np.random.RandomState(seed)
    tmp_path.mkdir(parents=True, exist_ok=True)
    X_seqs = [
        rng.randn(_SEQ_LEN, _FEAT.per_frame_total_dim).astype(np.float32)
        for _ in range(_N_TOTAL)
    ]
    # Prior slots must NOT already be a simplex, or _fill_priors refuses.
    for X in X_seqs:
        X[:, 4:] = 0.0
    Y_seqs = [rng.randn(_SEQ_LEN).astype(np.float32) for _ in range(_N_TOTAL)]
    labels = np.array([i % _MCMC for i in range(_N_TOTAL)], dtype=np.int64)
    lengths = np.full(_N_TOTAL, _SEQ_LEN, dtype=np.int64)
    logits = rng.randn(_N_TOTAL, _MCMC).astype(np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    mcmc_priors = exp / exp.sum(axis=1, keepdims=True)
    # Distinct sessions per trial; ``animal_of`` strips ``_session_N`` so
    # each animal here owns ``_TRIALS_PER_ANIMAL`` sessions.
    session_ids = [
        f"0.{500 + (i // _TRIALS_PER_ANIMAL)}cricket_001_20260101_0000"
        f"{i // _TRIALS_PER_ANIMAL}_session_{1 + (i % _TRIALS_PER_ANIMAL)}"
        for i in range(_N_TOTAL)
    ]
    assert len({_animal_of(s) for s in session_ids}) == _N_ANIMALS
    snapshots = rng.randn(_N_TOTAL, _FEAT.snapshot_dim).astype(np.float64)
    dataset = {
        "X_seqs": X_seqs,
        "Y_seqs": Y_seqs,
        "labels": labels,
        "lengths": lengths,
        "mcmc_priors": mcmc_priors,
        "mcmc_prior_provenance": "oof_5fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
        "session_ids": session_ids,
        "snapshots": snapshots,
        "feature_config": _FEAT,
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
    }
    path = tmp_path / "nsmor_dataset.pt"
    torch.save(dataset, path)
    return path


def _make_config(tmp_path: Path, *, normalize_targets: bool = False) -> Any:
    from nsmor.config_parser import ExperimentConfig

    config = ExperimentConfig()
    config.model.sensory_dim = 4
    config.model.mcmc_dim = _MCMC
    config.model.hidden_dim = _HIDDEN
    config.model.num_gru_layers = 1
    config.model.dropout = 0.0
    config.training.num_epochs = 1
    config.training.batch_size = 8
    config.training.max_seq_len = _SEQ_LEN
    config.training.random_seed = 42
    config.training.normalize_targets = normalize_targets
    config.training.target_clip_cm_s = 0
    config.training.lr_warmup_epochs = 0
    config.training.checkpoint_interval = 999
    config.loss.warmup_epochs = 0
    config.checkpoint.output_dir = str(tmp_path / "run")
    config.checkpoint.resume_from = None
    return config


def _animal_disjoint_split(
    session_ids: list, seed: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic animal-grouped split: first 2/3 of animals -> train."""
    animals = sorted({_animal_of(s) for s in session_ids})
    n_val = max(1, len(animals) // 3)
    rng = np.random.RandomState(seed)
    order = list(animals)
    rng.shuffle(order)
    val_animals = set(order[:n_val])
    train_idx = np.array(
        [i for i, s in enumerate(session_ids) if _animal_of(s) not in val_animals],
        dtype=np.int64,
    )
    val_idx = np.array(
        [i for i, s in enumerate(session_ids) if _animal_of(s) in val_animals],
        dtype=np.int64,
    )
    return train_idx, val_idx


def _make_nested_artifact(
    dataset_path: Path,
    dataset: Dict[str, Any],
    *,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    nested_priors: np.ndarray | None = None,
    fingerprint: str | None = None,
    is_nested_cv: bool = True,
    pipeline_version: str | None = None,
    include_sidecars: bool = True,
    split_seed: int = 3,
    val_split: float = 0.2,
) -> Dict[str, Any]:
    """Craft a nested artifact bound to ``dataset_path`` (unless overridden).

    Default split comes from the real ``grouped_train_val_split`` with the
    same ``(split_seed, val_split)`` recorded in the sidecar, so the
    loader's split-metadata honesty check (recompute-and-compare) passes
    for honest artifacts.  Tests that pass a hand-crafted partition are
    crafting *dishonest* metadata on purpose (refusal tests).
    """
    from nsmor.pipeline.grouping import grouped_train_val_split

    session_ids = list(dataset["session_ids"])
    n_total = len(session_ids)
    if train_idx is None or val_idx is None:
        train_idx, val_idx = grouped_train_val_split(
            session_ids, n_total, val_split=val_split, random_seed=split_seed
        )
    if nested_priors is None:
        rng = np.random.RandomState(11)
        logits = rng.randn(n_total, _MCMC)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        nested_priors = exp / exp.sum(axis=1, keepdims=True)
    if fingerprint is None:
        fingerprint = compute_source_fingerprint(dataset_path)
    art: Dict[str, Any] = {
        "nested_priors": nested_priors,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "is_nested_cv": is_nested_cv,
        "source_fingerprint": fingerprint,
        "pipeline_semantics_version": (
            pipeline_version
            if pipeline_version is not None
            else PIPELINE_SEMANTICS_VERSION
        ),
        "group_keys": np.array([_animal_of(s) for s in session_ids], dtype=object),
        "train_animals": sorted(
            {_animal_of(session_ids[i]) for i in train_idx if 0 <= int(i) < n_total}
        ),
        "val_animals": sorted(
            {_animal_of(session_ids[i]) for i in val_idx if 0 <= int(i) < n_total}
        ),
        "mcmc_prior_provenance": (
            f"nested_outer_seed{split_seed}_inner_5fold_recording_prefix_grouped"
        ),
        "split_seed": split_seed,
        "val_split": val_split,
    }
    art.update({
        "recording_prefix_keys": art["group_keys"].copy(),
        "train_recording_prefixes": list(art["train_animals"]),
        "val_recording_prefixes": list(art["val_animals"]),
        "n_train_recording_prefixes": len(art["train_animals"]),
        "n_val_recording_prefixes": len(art["val_animals"]),
        "n_inner_folds": 5,
        "animal_identity_status": "unverified",
    })
    if include_sidecars:
        art["train_priors"] = nested_priors[train_idx]
        art["val_priors"] = nested_priors[val_idx]
    return art


def _save_artifact(tmp_path: Path, art: Dict[str, Any], name: str = "nested.pt") -> Path:
    path = tmp_path / name
    torch.save(art, path)
    return path


# ═════════════════════════════════════════════════════════════
# Happy path: exact splits + feature channels + target stats
# ═════════════════════════════════════════════════════════════


def test_nested_artifact_exact_splits_and_feature_channels(tmp_path: Path) -> None:
    """build_dataloaders must use the persisted indices verbatim and write
    the nested priors into feature channels 4-7 of every train frame."""
    from scripts.train import build_dataloaders

    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset)
    art_path = _save_artifact(tmp_path, art)
    train_idx = np.asarray(art["train_indices"], dtype=np.int64)
    val_idx = np.asarray(art["val_indices"], dtype=np.int64)

    config = _make_config(tmp_path)
    train_loader, val_loader = build_dataloaders(
        config,
        dataset_path=str(ds_path),
        val_split=0.5,  # deliberately wrong; artifact must win
        use_lazy_loading=False,
        nested_prior_artifact=str(art_path),
    )
    assert train_loader is not None and val_loader is not None

    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    assert sorted(train_ds.source_indices) == sorted(train_idx.tolist())
    assert sorted(val_ds.source_indices) == sorted(val_idx.tolist())
    # Exact order as persisted (not re-sorted, not reshuffled).
    assert list(train_ds.source_indices) == train_idx.tolist()
    assert list(val_ds.source_indices) == val_idx.tolist()

    nested_priors = np.asarray(art["nested_priors"], dtype=np.float64)

    def _unpack(item: Any) -> Tuple[Any, Any]:
        """Dataset returns (idx, X, Y) when condition metadata is present."""
        if len(item) == 3:
            return item[1], item[2]
        return item[0], item[1]

    for i, src in enumerate(train_ds.source_indices):
        X, Y = _unpack(train_ds[i])
        # Feature channels 4-7 carry the nested prior of the SOURCE trial.
        np.testing.assert_allclose(
            X[:, 4:].detach().cpu().numpy(),
            np.broadcast_to(nested_priors[src], X[:, 4:].shape),
            atol=1e-6,
            err_msg=f"feature channels 4-7 wrong for source index {src}",
        )
        # Target stays continuous per-frame velocity, not class labels.
        assert Y.dim() == 1
        assert Y.shape == (_SEQ_LEN,)
        assert Y.dtype == torch.float32 or Y.dtype == torch.float64
        assert not np.allclose(Y.detach().cpu().numpy(), np.round(Y.detach().cpu().numpy()))

    for i, src in enumerate(val_ds.source_indices):
        X, _Y = _unpack(val_ds[i])
        np.testing.assert_allclose(
            X[:, 4:].detach().cpu().numpy(),
            np.broadcast_to(nested_priors[src], X[:, 4:].shape),
            atol=1e-6,
        )


def test_target_stats_use_nested_train_indices(tmp_path: Path) -> None:
    """compute_target_stats must fit on exactly the nested train indices."""
    from scripts.train import compute_target_stats

    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset)
    art_path = _save_artifact(tmp_path, art)
    config = _make_config(tmp_path, normalize_targets=True)

    mean, std, train_indices = compute_target_stats(
        str(ds_path),
        config,
        val_split=0.5,
        nested_prior_artifact=str(art_path),
    )
    expected = np.asarray(art["train_indices"], dtype=np.int64)
    assert np.array_equal(np.asarray(train_indices), expected)
    assert np.isfinite(mean) and std > 0.0


def _tiny_mcmc_config() -> Any:
    """Tiny MCMC training config so the end-to-end test stays fast."""
    from nsmor.config import MCMCTrainingConfig

    return MCMCTrainingConfig(num_epochs=5)


def test_end_to_end_generate_nested_priors_then_train_seam(tmp_path: Path) -> None:
    """Artifact produced by generate_nested_priors feeds build_dataloaders."""
    from scripts.evaluate_nested_prior import generate_nested_priors
    from scripts.train import build_dataloaders

    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    result = generate_nested_priors(
        dataset,
        split_seed=3,
        val_split=0.34,
        n_inner_folds=2,
        mcmc_config=_tiny_mcmc_config(),
        source_dataset_path=ds_path,
        verbose=False,
    )
    art_path = _save_artifact(tmp_path, result, name="nested_split_seed3.pt")

    config = _make_config(tmp_path)
    train_loader, val_loader = build_dataloaders(
        config,
        dataset_path=str(ds_path),
        val_split=0.2,
        nested_prior_artifact=str(art_path),
    )
    assert train_loader is not None and val_loader is not None
    train_idx = np.asarray(result["train_indices"], dtype=np.int64)
    assert list(train_loader.dataset.source_indices) == train_idx.tolist()
    assert sorted(
        list(train_loader.dataset.source_indices)
        + list(val_loader.dataset.source_indices)
    ) == list(range(_N_TOTAL))


# ═════════════════════════════════════════════════════════════
# Fail-closed refusals
# ═════════════════════════════════════════════════════════════


def test_wrong_fingerprint_refused(tmp_path: Path) -> None:
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset, fingerprint="0" * 64)
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


# ── Split-metadata honesty (Reviewer B-3) ──────────────────────


def test_lying_split_metadata_refused(tmp_path: Path) -> None:
    """A sidecar whose (split_seed, val_split) do not reproduce the
    persisted partition must be refused — metadata must not be echoed
    as truth (Reviewer B-3: split_seed=999, val_split=0.99 accepted)."""
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset, split_seed=3, val_split=0.2)
    # Persisted partition comes from (3, 0.2); the sidecar now claims
    # a completely different protocol.
    art["split_seed"] = 999
    art["val_split"] = 0.99
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="split_seed"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


def test_lying_realized_counts_refused(tmp_path: Path) -> None:
    """A sidecar claiming an unrealizable realized trial fraction /
    split sizes must be refused (counts are recomputed from indices)."""
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset, split_seed=3, val_split=0.2)
    art["val_trial_fraction"] = 0.99  # persisted indices say otherwise
    art["n_train"] = 1
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="val_trial_fraction|n_train"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


def test_honest_artifact_reports_realized_fraction(tmp_path: Path) -> None:
    """Honest sidecars load and report realized split sizes from indices,
    distinguishing the animal-level val_split request from the realized
    trial fraction."""
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset, split_seed=3, val_split=0.2)
    art_path = _save_artifact(tmp_path, art)
    _tr, _va, _p, info = load_nested_prior_split(
        art_path, ds_path, _N_TOTAL, dataset["session_ids"]
    )
    assert info["val_split"] == 0.2
    n_val = int(np.asarray(art["val_indices"]).size)
    n_total = _N_TOTAL
    assert abs(info["val_trial_fraction"] - n_val / n_total) < 1e-9
    assert info["n_train"] + info["n_val"] == n_total
    assert info["is_nested_cv"] is True
    assert len(info["nested_prior_fingerprint"]) == 64


# ── Non-finite target fail-closed (Reviewer B-1) ───────────────


def test_nonfinite_targets_refused(tmp_path: Path) -> None:
    """A NaN anywhere in Y_seqs must raise at loader/stats boundaries —
    silent NaN mean/std is total corruption (Reviewer B-1)."""
    from scripts.train import assert_finite_targets

    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    Y = [np.asarray(y).copy() for y in dataset["Y_seqs"]]
    Y[3] = Y[3].copy()
    Y[3][2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        assert_finite_targets(Y)
    assert_finite_targets(dataset["Y_seqs"])  # clean data still passes


# ── Checkpoint provenance (Reviewer B-2 / A-1) ─────────────────


def test_nested_provenance_in_checkpoint_and_metrics(tmp_path: Path) -> None:
    """Nested runs must stamp nested_prior_artifact/fingerprint/is_nested_cv
    into every checkpoint and metrics.json (Reviewer B-2)."""
    from scripts.train import train

    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset)
    art_path = _save_artifact(tmp_path, art)
    config = _make_config(tmp_path)

    results = train(
        config,
        lambda_reg=0.01,
        dataset_path=str(ds_path),
        nested_prior_artifact=str(art_path),
    )
    assert results["is_nested_cv"] is True
    assert results["validation_scope"] == "nested_outer_validation"
    assert results["nested_prior_artifact"] == str(art_path)
    assert len(results["nested_prior_fingerprint"]) == 64
    assert results["mcmc_prior_provenance"].startswith("nested_outer_seed")

    out = Path(config.checkpoint.output_dir)
    ckpt = load_artifact_bytes((out / "final_model.pth").read_bytes())
    for key in (
        "nested_prior_artifact",
        "nested_prior_fingerprint",
        "is_nested_cv",
        "mcmc_prior_provenance",
    ):
        assert key in ckpt, f"checkpoint missing provenance key {key}"
    assert ckpt["is_nested_cv"] is True
    assert ckpt["nested_prior_artifact"] == str(art_path)

    import json

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["is_nested_cv"] is True
    assert metrics["nested_prior_artifact"] == str(art_path)
    assert len(metrics["nested_prior_fingerprint"]) == 64
    expected_lineage = {
        "nested_prior_artifact": str(art_path),
        "nested_prior_fingerprint": compute_source_fingerprint(ds_path),
        "is_nested_cv": True,
        "mcmc_prior_provenance": art["mcmc_prior_provenance"],
        "nested_split_seed": art["split_seed"],
        "nested_val_split": art["val_split"],
    }
    best_ckpt = load_artifact_bytes((out / "best_model.pth").read_bytes())
    for record in (results, ckpt, best_ckpt, metrics):
        for key, expected in expected_lineage.items():
            assert record[key] == expected, f"lineage mismatch for {key}"


def test_legacy_run_marks_not_nested(tmp_path: Path) -> None:
    """Without an artifact, checkpoints/results must explicitly say
    is_nested_cv=False — absence of the seam is itself auditable."""
    from scripts.train import train

    ds_path = _make_canonical_dataset(tmp_path)
    config = _make_config(tmp_path)
    results = train(config, lambda_reg=0.01, dataset_path=str(ds_path))
    assert results["is_nested_cv"] is False
    assert results["nested_prior_artifact"] == ""
    assert results["mcmc_prior_provenance"] == "oof_5fold_recording_prefix_grouped_cv"
    assert results["animal_identity_status"] == "unverified"
    assert results["validation_scope"] == "diagnostic_global_oof"
    ckpt = load_artifact_bytes(
        (Path(config.checkpoint.output_dir) / "final_model.pth").read_bytes()
    )
    assert ckpt["is_nested_cv"] is False
    assert "nested_prior_fingerprint" in ckpt


def test_empty_fingerprint_refused(tmp_path: Path) -> None:
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset, fingerprint="")
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="64-char SHA-256"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


def test_fingerprint_of_different_dataset_refused(tmp_path: Path) -> None:
    """Artifact bound to dataset A must be refused for dataset B."""
    ds_a = _make_canonical_dataset(tmp_path / "a", seed=7)
    ds_b_dir = tmp_path / "b"
    ds_b_dir.mkdir()
    ds_b = _make_canonical_dataset(ds_b_dir, seed=8)
    dataset_a = load_artifact_bytes(ds_a.read_bytes())
    art = _make_nested_artifact(ds_a, dataset_a)
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_nested_prior_split(art_path, ds_b, _N_TOTAL, dataset_a["session_ids"])


def test_overlapping_indices_refused(tmp_path: Path) -> None:
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    train_idx, val_idx = _animal_disjoint_split(list(dataset["session_ids"]))
    bad_val = np.concatenate([val_idx, train_idx[:1]])
    bad_train = train_idx[1:]
    art = _make_nested_artifact(
        ds_path, dataset, train_idx=bad_train, val_idx=bad_val, include_sidecars=False
    )
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


def test_out_of_range_indices_refused(tmp_path: Path) -> None:
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    train_idx, val_idx = _animal_disjoint_split(list(dataset["session_ids"]))
    bad_train = train_idx.copy()
    bad_train[0] = _N_TOTAL + 5
    art = _make_nested_artifact(
        ds_path, dataset, train_idx=bad_train, val_idx=val_idx, include_sidecars=False
    )
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="out-of-range"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


def test_non_partition_indices_refused(tmp_path: Path) -> None:
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    train_idx, val_idx = _animal_disjoint_split(list(dataset["session_ids"]))
    art = _make_nested_artifact(
        ds_path,
        dataset,
        train_idx=train_idx[:-1],
        val_idx=val_idx,
        include_sidecars=False,
    )
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="exact partition"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


def test_animal_overlap_refused(tmp_path: Path) -> None:
    """A partition that still leaks an animal across sides must be refused."""
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    session_ids = list(dataset["session_ids"])
    train_idx, val_idx = _animal_disjoint_split(session_ids)
    # Guarantee a leak: move ONE trial of a val animal into train and one
    # train trial into val, preserving the exact partition size while
    # putting that val animal on both sides.
    moved_val = int(val_idx[0])
    moved_train = int(train_idx[0])
    train_idx = np.concatenate([train_idx[1:], [moved_val]]).astype(np.int64)
    val_idx = np.concatenate([val_idx[1:], [moved_train]]).astype(np.int64)
    train_animals = {_animal_of(session_ids[i]) for i in train_idx}
    val_animals = {_animal_of(session_ids[i]) for i in val_idx}
    assert train_animals & val_animals, "fixture must produce an animal leak"

    art = _make_nested_artifact(
        ds_path, dataset, train_idx=train_idx, val_idx=val_idx, include_sidecars=False
    )
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="leaks"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, session_ids)


def test_bad_prior_dim_refused(tmp_path: Path) -> None:
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    rng = np.random.RandomState(0)
    art = _make_nested_artifact(
        ds_path, dataset, nested_priors=rng.rand(_N_TOTAL, 3), include_sidecars=False
    )
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="shape"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


def test_non_simplex_priors_refused(tmp_path: Path) -> None:
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    bad = np.full((_N_TOTAL, _MCMC), 0.5)
    art = _make_nested_artifact(
        ds_path, dataset, nested_priors=bad, include_sidecars=False
    )
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="sum to 1"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


def test_negative_priors_refused(tmp_path: Path) -> None:
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    bad = np.full((_N_TOTAL, _MCMC), 0.25)
    bad[0, 0] = -0.5
    bad[0, 1] = 1.25
    art = _make_nested_artifact(
        ds_path, dataset, nested_priors=bad, include_sidecars=False
    )
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="negative"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


def test_pipeline_version_mismatch_refused(tmp_path: Path) -> None:
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(
        ds_path, dataset, pipeline_version="0.0.0-not-a-version"
    )
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="pipeline_semantics_version"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


def test_not_nested_flag_refused(tmp_path: Path) -> None:
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset, is_nested_cv=False)
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="is_nested_cv"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


def test_missing_sidecar_misalignment_refused(tmp_path: Path) -> None:
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset)
    # Keep rows a valid simplex so only the alignment check can fire.
    n_train = int(np.asarray(art["train_indices"]).size)
    rng = np.random.RandomState(99)
    logits = rng.randn(n_train, _MCMC)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    art["train_priors"] = exp / exp.sum(axis=1, keepdims=True)
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="aligned"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


def test_missing_session_ids_refused(tmp_path: Path) -> None:
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset)
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="session_ids"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, None)


def test_missing_required_key_refused(tmp_path: Path) -> None:
    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset)
    del art["val_indices"]
    art_path = _save_artifact(tmp_path, art)
    with pytest.raises(ValueError, match="missing required key"):
        load_nested_prior_split(art_path, ds_path, _N_TOTAL, dataset["session_ids"])


def test_build_dataloaders_refuses_bad_artifact(tmp_path: Path) -> None:
    """build_dataloaders must surface the fail-closed error, not fall back."""
    from scripts.train import build_dataloaders

    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset, fingerprint="1" * 64)
    art_path = _save_artifact(tmp_path, art)
    config = _make_config(tmp_path)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        build_dataloaders(
            config,
            dataset_path=str(ds_path),
            nested_prior_artifact=str(art_path),
        )


# ═════════════════════════════════════════════════════════════
# Legacy default unchanged
# ═════════════════════════════════════════════════════════════


def test_legacy_default_unchanged(tmp_path: Path) -> None:
    """Without an artifact, the recomputed animal-grouped split and the
    dataset's global priors must still be used (and the artifact cannot
    be silently ignored when malformed)."""
    from scripts.train import build_dataloaders
    from nsmor.pipeline.grouping import grouped_train_val_split

    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    config = _make_config(tmp_path)

    train_loader, val_loader = build_dataloaders(
        config, dataset_path=str(ds_path), val_split=0.25
    )
    assert train_loader is not None and val_loader is not None

    train_idx, val_idx = grouped_train_val_split(
        dataset["session_ids"],
        _N_TOTAL,
        val_split=0.25,
        random_seed=config.training.random_seed,
    )
    assert sorted(train_loader.dataset.source_indices) == sorted(train_idx.tolist())
    assert sorted(val_loader.dataset.source_indices) == sorted(val_idx.tolist())

    global_priors = np.asarray(dataset["mcmc_priors"], dtype=np.float64)
    item = train_loader.dataset[0]
    X0 = item[1] if len(item) == 3 else item[0]
    src = train_loader.dataset.source_indices[0]
    np.testing.assert_allclose(
        X0[:, 4:].detach().cpu().numpy(),
        np.broadcast_to(global_priors[src], X0[:, 4:].shape),
        atol=1e-6,
    )


def test_lazy_mode_with_nested_artifact_fail_closed(tmp_path: Path) -> None:
    from scripts.train import build_dataloaders

    ds_path = _make_canonical_dataset(tmp_path)
    config = _make_config(tmp_path)
    with pytest.raises(ValueError, match="fail closed|ETL mode"):
        build_dataloaders(
            config,
            dataset_path=str(ds_path),
            use_lazy_loading=True,
            nested_prior_artifact=str(tmp_path / "anything.pt"),
        )


def test_cli_accepts_nested_prior_artifact_flag() -> None:
    from scripts.train import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--nested_prior_artifact", "/tmp/nested.pt"])
    assert args.nested_prior_artifact == "/tmp/nested.pt"
    args_default = parser.parse_args([])
    assert args_default.nested_prior_artifact is None


# ═════════════════════════════════════════════════════════════
# Fail-closed resume validation for nested mode
# ═════════════════════════════════════════════════════════════


def test_resume_missing_checkpoint_refused(tmp_path: Path) -> None:
    """Missing resume checkpoint file must fail closed."""
    from scripts.train import train

    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset)
    art_path = _save_artifact(tmp_path, art)
    config = _make_config(tmp_path)
    config.checkpoint.resume_from = str(tmp_path / "nonexistent.pth")

    with pytest.raises(FileNotFoundError, match="Checkpoint file to resume from not found"):
        train(
            config,
            dataset_path=str(ds_path),
            nested_prior_artifact=str(art_path),
        )


def test_resume_mode_mismatch_refused(tmp_path: Path) -> None:
    """Resuming nested run from non-nested checkpoint (or vice versa) must fail closed."""
    from scripts.train import train

    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset)
    art_path = _save_artifact(tmp_path, art)

    # 1. Run 1 epoch legacy non-nested
    config_legacy = _make_config(tmp_path / "legacy")
    config_legacy.training.num_epochs = 1
    train(config_legacy, dataset_path=str(ds_path))
    legacy_ckpt = Path(config_legacy.checkpoint.output_dir) / "final_model.pth"

    # 2. Try resuming with nested_prior_artifact from legacy checkpoint -> Fail closed
    config_nested = _make_config(tmp_path / "nested_fail")
    config_nested.training.num_epochs = 2
    config_nested.checkpoint.resume_from = str(legacy_ckpt)

    with pytest.raises(ValueError, match="Resume nested mode mismatch"):
        train(
            config_nested,
            dataset_path=str(ds_path),
            nested_prior_artifact=str(art_path),
        )

    # 3. Try resuming WITHOUT nested_prior_artifact from nested checkpoint -> Fail closed
    config_nested2 = _make_config(tmp_path / "nested2")
    config_nested2.training.num_epochs = 1
    train(
        config_nested2,
        dataset_path=str(ds_path),
        nested_prior_artifact=str(art_path),
    )
    nested_ckpt = Path(config_nested2.checkpoint.output_dir) / "final_model.pth"

    config_omitted = _make_config(tmp_path / "omitted")
    config_omitted.training.num_epochs = 2
    config_omitted.checkpoint.resume_from = str(nested_ckpt)

    with pytest.raises(ValueError, match="Resume nested mode mismatch"):
        train(
            config_omitted,
            dataset_path=str(ds_path),
            nested_prior_artifact=None,
        )


def test_resume_mismatched_artifact_refused(tmp_path: Path) -> None:
    """Resuming with a different nested artifact than the one recorded in checkpoint must fail closed."""
    from scripts.train import train

    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art1 = _make_nested_artifact(ds_path, dataset, split_seed=3)
    art1_path = _save_artifact(tmp_path, art1, name="art1.pt")

    # Run epoch 1 on art1
    config1 = _make_config(tmp_path / "run1")
    config1.training.num_epochs = 1
    train(
        config1,
        dataset_path=str(ds_path),
        nested_prior_artifact=str(art1_path),
    )
    ckpt = Path(config1.checkpoint.output_dir) / "final_model.pth"

    # Art2 on different seed
    art2 = _make_nested_artifact(ds_path, dataset, split_seed=4)
    art2_path = _save_artifact(tmp_path, art2, name="art2.pt")

    config2 = _make_config(tmp_path / "run2")
    config2.training.num_epochs = 2
    config2.checkpoint.resume_from = str(ckpt)

    with pytest.raises(ValueError, match="Resume nested prior artifact mismatch"):
        train(
            config2,
            dataset_path=str(ds_path),
            nested_prior_artifact=str(art2_path),
        )


def test_resume_changed_dataset_or_fingerprint_refused(tmp_path: Path) -> None:
    """Resuming with altered dataset / source fingerprint must fail closed."""
    from scripts.train import train

    ds1_path = _make_canonical_dataset(tmp_path / "ds1", seed=7)
    dataset1 = load_artifact_bytes(ds1_path.read_bytes())
    art1 = _make_nested_artifact(ds1_path, dataset1, split_seed=3)
    art1_path = _save_artifact(tmp_path, art1, name="art1.pt")

    config1 = _make_config(tmp_path / "run1")
    config1.training.num_epochs = 1
    train(
        config1,
        dataset_path=str(ds1_path),
        nested_prior_artifact=str(art1_path),
    )
    ckpt = Path(config1.checkpoint.output_dir) / "final_model.pth"

    # Tamper with fingerprint in checkpoint or pass an artifact with changed source
    ds2_path = _make_canonical_dataset(tmp_path / "ds2", seed=8)
    dataset2 = load_artifact_bytes(ds2_path.read_bytes())
    art2 = _make_nested_artifact(ds2_path, dataset2, split_seed=3)
    art2_path = _save_artifact(tmp_path, art2, name="art1.pt")  # same filename!

    config2 = _make_config(tmp_path / "run2")
    config2.training.num_epochs = 2
    config2.checkpoint.resume_from = str(ckpt)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        train(
            config2,
            dataset_path=str(ds2_path),
            nested_prior_artifact=str(art2_path),
        )


def test_resume_changed_split_refused(tmp_path: Path) -> None:
    """Resuming when split_seed or val_split differs must fail closed."""
    from scripts.train import train

    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset, split_seed=3, val_split=0.2)
    art_path = _save_artifact(tmp_path, art)

    config1 = _make_config(tmp_path / "run1")
    config1.training.num_epochs = 1
    train(
        config1,
        dataset_path=str(ds_path),
        nested_prior_artifact=str(art_path),
    )
    ckpt_path = Path(config1.checkpoint.output_dir) / "final_model.pth"

    # Mutate checkpoint's recorded split_seed
    ckpt_data = load_artifact_bytes(ckpt_path.read_bytes())
    ckpt_data["nested_split_seed"] = 999
    torch.save(ckpt_data, ckpt_path)

    config2 = _make_config(tmp_path / "run2")
    config2.training.num_epochs = 2
    config2.checkpoint.resume_from = str(ckpt_path)

    with pytest.raises(ValueError, match="Resume nested split_seed mismatch"):
        train(
            config2,
            dataset_path=str(ds_path),
            nested_prior_artifact=str(art_path),
        )


def test_resume_preserves_best_validation_score_and_checkpoint(tmp_path: Path) -> None:
    """Resuming must inherit best_val_loss from the checkpoint and only overwrite best_model.pth if truly improved."""
    from scripts.train import train

    ds_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(ds_path.read_bytes())
    art = _make_nested_artifact(ds_path, dataset, split_seed=3, val_split=0.2)
    art_path = _save_artifact(tmp_path, art)

    # Initial 1-epoch run
    config1 = _make_config(tmp_path / "run1")
    config1.training.num_epochs = 1
    res1 = train(
        config1,
        dataset_path=str(ds_path),
        nested_prior_artifact=str(art_path),
    )
    best_v1 = res1["best_val_loss"]
    assert math.isfinite(best_v1)
    best_ckpt_1 = Path(config1.checkpoint.output_dir) / "best_model.pth"
    assert best_ckpt_1.exists()
    # The resume source is the segment's last complete-epoch checkpoint
    # (final_model.pth); best_model.pth is NOT a resume source and is refused
    # by _require_resume_source_is_newest_complete_epoch.
    final_ckpt_1 = Path(config1.checkpoint.output_dir) / "final_model.pth"
    assert final_ckpt_1.exists()

    # Artificially set best_val_loss in the best AND the last-epoch checkpoint
    # to a very low value (e.g. 0.001) to test that resume doesn't overwrite it
    # when epoch 2 val_loss is higher.  Both are stamped so the reconciliation
    # (which certifies the companion best against the resumed source) sees a
    # consistent historical best.
    for path in (best_ckpt_1, final_ckpt_1):
        ckpt1_data = load_artifact_bytes(path.read_bytes())
        ckpt1_data["loss"] = 0.001
        ckpt1_data["val_loss"] = 0.001
        ckpt1_data["best_val_loss"] = 0.001
        torch.save(ckpt1_data, path)

    # Resume for epoch 2
    config2 = _make_config(tmp_path / "run2")
    config2.training.num_epochs = 2
    config2.checkpoint.resume_from = str(final_ckpt_1)

    res2 = train(
        config2,
        dataset_path=str(ds_path),
        nested_prior_artifact=str(art_path),
    )
    # Best val loss should be preserved as 0.001 because the new epoch loss (~0.5 - 1.0) is worse
    assert res2["best_val_loss"] == pytest.approx(0.001)
