"""Test nested prior evaluation tool and snapshot persistence."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from nsmor.config import (
    DEFAULT_FEATURE,
    DEFAULT_MCMC_TRAINING,
    FeatureConfig,
    MCMCTrainingConfig,
    PIPELINE_SEMANTICS_VERSION,
)
from nsmor.pipeline.nested_prior import load_artifact_bytes, load_nested_prior_split
from scripts.evaluate_nested_prior import (
    compute_source_fingerprint,
    generate_nested_priors,
    validate_nested_dataset,
)


def _create_synthetic_dataset(
    n_animals: int = 12,
    trials_per_animal: int = 5,
    n_classes: int = 4,
    random_seed: int = 42,
    include_multi_session: bool = False,
) -> dict:
    """Create a synthetic dataset with canonical snapshots, labels, and session IDs."""
    rng = np.random.RandomState(random_seed)
    n_trials = n_animals * trials_per_animal
    n_frames = 100
    feature_config = FeatureConfig(num_classes=n_classes)

    X_seqs = [
        rng.randn(n_frames, feature_config.per_frame_total_dim).astype(np.float32)
        for _ in range(n_trials)
    ]
    Y_seqs = [
        rng.randn(n_frames).astype(np.float32) for _ in range(n_trials)
    ]
    # Canonical 5-D snapshots
    snapshots = rng.randn(n_trials, feature_config.snapshot_dim).astype(np.float64)

    # Balanced class labels across animals
    labels = np.array([i % n_classes for i in range(n_trials)], dtype=np.int64)

    # Global priors (for testing immunity)
    global_priors = rng.dirichlet(np.ones(n_classes), size=n_trials).astype(np.float64)

    session_ids = []
    trial_ids = []
    for animal_idx in range(n_animals):
        animal_name = f"0.{500 + animal_idx}cricket_{animal_idx:03d}"
        for t_idx in range(trials_per_animal):
            if include_multi_session:
                sess_num = (t_idx % 2) + 1
                session_ids.append(f"{animal_name}_session_{sess_num}")
            else:
                session_ids.append(f"{animal_name}_session_1")
            trial_ids.append(t_idx)

    return {
        "X_seqs": X_seqs,
        "Y_seqs": Y_seqs,
        "snapshots": snapshots,
        "mcmc_snapshots": snapshots,
        "labels": labels,
        "lengths": np.array([n_frames] * n_trials, dtype=np.int64),
        "mcmc_priors": global_priors,
        "session_ids": session_ids,
        "trial_ids": np.array(trial_ids, dtype=np.int64),
        "feature_config": feature_config,
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
    }


def _write_ds_file(tmp_path: Path, ds: dict, name: str = "ds.pt") -> Path:
    """Persist a synthetic dataset so generate_nested_priors can fingerprint it.

    The generator fails closed without a real source file (an artifact
    with an empty fingerprint can never be loaded), so every call site
    must have one on disk.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    ds_path = tmp_path / name
    torch.save(ds, ds_path)
    return ds_path


# ── Test 1: CLI Help ──────────────────────────────────────────


def test_evaluate_nested_prior_cli_help():
    """CLI --help must describe nested prior generation and not claim NSMoR regression evaluation."""
    res = subprocess.run(
        [sys.executable, "scripts/evaluate_nested_prior.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "--dataset" in res.stdout
    assert "--output_dir" in res.stdout
    assert "--split_seed" in res.stdout
    # Must not claim NSMoR regression training/evaluation is implemented
    stdout_lower = res.stdout.lower()
    assert "leak-free" in stdout_lower or "nested" in stdout_lower
    assert "separately" in stdout_lower or "not" in stdout_lower
    assert "recording-prefix" in stdout_lower
    assert "unverified" in stdout_lower


# ── Test 2: Validation Label Leakage Immunity ─────────────────


def test_val_label_leakage_immunity(tmp_path: Path):
    """Perturbing ONLY validation labels must yield identical outer split, inner folds, and priors."""
    ds = _create_synthetic_dataset(n_animals=12, trials_per_animal=5, random_seed=42)
    ds_path = _write_ds_file(tmp_path, ds)
    mcmc_cfg = MCMCTrainingConfig(num_epochs=10, random_seed=42)

    # 1. Run baseline nested prior generation
    res1 = generate_nested_priors(
        dataset=ds,
        split_seed=42,
        val_split=0.2,
        n_inner_folds=5,
        mcmc_config=mcmc_cfg,
        source_dataset_path=ds_path,
        verbose=False,
    )

    val_indices = res1["val_indices"]
    train_indices = res1["train_indices"]
    assert len(val_indices) > 0
    assert len(train_indices) > 0

    # 2. Perturb ONLY validation labels and persist to an authoritative file
    ds_perturbed = dict(ds)
    perturbed_labels = ds["labels"].copy()
    perturbed_labels[val_indices] = (perturbed_labels[val_indices] + 1) % 4
    ds_perturbed["labels"] = perturbed_labels
    ds_perturbed_path = _write_ds_file(tmp_path, ds_perturbed, "ds_perturbed.pt")

    res2 = generate_nested_priors(
        dataset=ds_perturbed,
        split_seed=42,
        val_split=0.2,
        n_inner_folds=5,
        mcmc_config=mcmc_cfg,
        source_dataset_path=ds_perturbed_path,
        verbose=False,
    )

    # Assert identical split
    np.testing.assert_array_equal(res1["train_indices"], res2["train_indices"])
    np.testing.assert_array_equal(res1["val_indices"], res2["val_indices"])
    assert res1["n_inner_folds"] == res2["n_inner_folds"]

    # Assert bit-identical / numerically identical priors
    np.testing.assert_allclose(res1["train_priors"], res2["train_priors"], atol=1e-6)
    np.testing.assert_allclose(res1["val_priors"], res2["val_priors"], atol=1e-6)
    np.testing.assert_allclose(res1["nested_priors"], res2["nested_priors"], atol=1e-6)


# ── Test 3: Global Priors Immunity ────────────────────────────


def test_global_priors_immunity(tmp_path: Path):
    """Perturbing supplied global priors must not change generated nested prior outputs."""
    ds = _create_synthetic_dataset(n_animals=12, trials_per_animal=5, random_seed=42)
    ds_path = _write_ds_file(tmp_path, ds)
    mcmc_cfg = MCMCTrainingConfig(num_epochs=10, random_seed=42)

    res1 = generate_nested_priors(
        dataset=ds,
        split_seed=42,
        val_split=0.2,
        n_inner_folds=5,
        mcmc_config=mcmc_cfg,
        source_dataset_path=ds_path,
        verbose=False,
    )

    # Corrupt global priors completely (e.g. all 0.25 uniform or random noise)
    ds_corrupted = dict(ds)
    ds_corrupted["mcmc_priors"] = np.full_like(ds["mcmc_priors"], 0.25)

    res2 = generate_nested_priors(
        dataset=ds_corrupted,
        split_seed=42,
        val_split=0.2,
        n_inner_folds=5,
        mcmc_config=mcmc_cfg,
        source_dataset_path=ds_path,
        verbose=False,
    )

    np.testing.assert_allclose(res1["nested_priors"], res2["nested_priors"], atol=1e-6)
    np.testing.assert_allclose(res1["train_priors"], res2["train_priors"], atol=1e-6)
    np.testing.assert_allclose(res1["val_priors"], res2["val_priors"], atol=1e-6)


# ── Test 3b: In-Memory / File Mismatch Refusal ─────────────────


def test_in_memory_file_mismatch_refused(tmp_path: Path):
    """When in-memory dataset labels disagree with the authoritative file on disk,
    generate_nested_priors must fail closed with an in-memory/file mismatch error."""
    ds = _create_synthetic_dataset(n_animals=12, trials_per_animal=5, random_seed=42)
    ds_path = _write_ds_file(tmp_path, ds)
    mcmc_cfg = MCMCTrainingConfig(num_epochs=5, random_seed=42)

    ds_tampered = dict(ds)
    tampered_labels = ds["labels"].copy()
    tampered_labels[0] = (tampered_labels[0] + 1) % 4
    ds_tampered["labels"] = tampered_labels

    with pytest.raises(ValueError, match="in-memory/file mismatch"):
        generate_nested_priors(
            dataset=ds_tampered,
            split_seed=42,
            val_split=0.2,
            n_inner_folds=3,
            mcmc_config=mcmc_cfg,
            source_dataset_path=ds_path,
            verbose=False,
        )


# ── Test 4: Recording-prefix disjointness ───────────────────────


def test_shared_recording_prefix_sessions_disjointness(tmp_path: Path, caplog):
    """Session suffixes sharing a prefix stay together; legacy counts count prefixes."""
    ds = _create_synthetic_dataset(
        n_animals=10, trials_per_animal=6, include_multi_session=True, random_seed=42,
    )
    ds_path = _write_ds_file(tmp_path, ds)
    mcmc_cfg = MCMCTrainingConfig(num_epochs=5, random_seed=42)

    res = generate_nested_priors(
        dataset=ds,
        split_seed=42,
        val_split=0.2,
        n_inner_folds=3,
        mcmc_config=mcmc_cfg,
        source_dataset_path=ds_path,
        verbose=False,
    )

    train_idx = res["train_indices"]
    val_idx = res["val_indices"]
    group_keys = res["group_keys"]

    # Outer split disjointness
    train_prefixes = set(group_keys[train_idx])
    val_prefixes = set(group_keys[val_idx])
    assert not (train_prefixes & val_prefixes)
    assert res["train_recording_prefixes"] == sorted(train_prefixes)
    assert res["val_recording_prefixes"] == sorted(val_prefixes)
    assert res["animal_identity_status"] == "unverified"
    assert res["mcmc_prior_provenance"].endswith("recording_prefix_grouped")
    assert res["train_animals"] == res["train_recording_prefixes"]
    assert res["val_animals"] == res["val_recording_prefixes"]
    assert res["n_train_animals"] == len(train_prefixes)
    assert res["n_val_animals"] == len(val_prefixes)

    artifact_path = tmp_path / "nested.pt"
    torch.save(res, artifact_path)
    with caplog.at_level("INFO", logger="nsmor.pipeline.nested_prior"):
        loaded_train, loaded_val, _, info = load_nested_prior_split(
            artifact_path, ds_path, len(ds["labels"]), ds["session_ids"],
        )
    np.testing.assert_array_equal(loaded_train, train_idx)
    np.testing.assert_array_equal(loaded_val, val_idx)
    assert info["n_train_recording_prefixes"] == len(train_prefixes)
    assert info["n_val_recording_prefixes"] == len(val_prefixes)
    assert info["animal_identity_status"] == "unverified"
    forged = dict(res, animal_identity_status="verified")
    torch.save(forged, artifact_path)
    with pytest.raises(ValueError, match="animal_identity_status"):
        load_nested_prior_split(artifact_path, ds_path, len(ds["labels"]), ds["session_ids"])
    assert info["n_train_animals"] == len(train_prefixes)
    assert info["n_val_animals"] == len(val_prefixes)
    assert f"Recording prefixes: {len(train_prefixes)} train, {len(val_prefixes)} val" in caplog.text
    assert "animal identities across prefixes unverified" in caplog.text

    # Inner fold diagnostics check
    for diag in res["inner_fold_diagnostics"]:
        assert diag["n_train_sessions"] > 0
        assert diag["n_oof_sessions"] > 0


# ── Test 5: Train-Only Fold Resolution ─────────────────────────


def test_train_only_fold_resolution(tmp_path: Path):
    """Inner fold count must be resolved from outer-train coverage only."""
    ds = _create_synthetic_dataset(n_animals=8, trials_per_animal=4, random_seed=42)
    ds_path = _write_ds_file(tmp_path, ds)
    mcmc_cfg = MCMCTrainingConfig(num_epochs=5, random_seed=42)

    res = generate_nested_priors(
        dataset=ds,
        split_seed=42,
        val_split=0.25,
        n_inner_folds=5,
        mcmc_config=mcmc_cfg,
        source_dataset_path=ds_path,
        verbose=False,
    )
    assert 2 <= res["n_inner_folds"] <= 5
    # Train animals count must be >= n_inner_folds
    assert len(res["train_animals"]) >= res["n_inner_folds"]


# ── Test 6: Fail Closed on Insufficient Groups / Classes ───────


def test_fail_closed_insufficient_groups(tmp_path: Path):
    """Fail closed when outer train has insufficient group/class coverage (no sample fallback)."""
    ds = _create_synthetic_dataset(n_animals=4, trials_per_animal=4, random_seed=42)
    # Force class 3 to appear in only 1 animal
    group_keys = np.array([f"animal_{i // 4}" for i in range(16)])
    ds["session_ids"] = group_keys.tolist()
    labels = np.array([0, 1, 2, 0] * 3 + [0, 1, 2, 3], dtype=np.int64)
    ds["labels"] = labels
    ds_path = _write_ds_file(tmp_path, ds)

    mcmc_cfg = MCMCTrainingConfig(num_epochs=5, random_seed=42)

    # Class 3 only appears in 1 animal -> resolve_group_folds or coverage check must raise ValueError
    with pytest.raises(ValueError, match="occupy fewer than 2 groups|coverage|missing"):
        generate_nested_priors(
            dataset=ds,
            split_seed=42,
            val_split=0.25,
            n_inner_folds=3,
            mcmc_config=mcmc_cfg,
            source_dataset_path=ds_path,
            verbose=False,
        )


# ── Test 7: Fail Closed on Missing/Invalid Inputs ──────────────


def test_fail_closed_missing_or_corrupt_inputs():
    """Fail closed on missing animal IDs, invalid snapshots, wrong shapes, or nonfinite values."""
    ds = _create_synthetic_dataset(n_animals=6, trials_per_animal=4, random_seed=42)

    # 1. Missing session_ids
    bad_ds = dict(ds)
    del bad_ds["session_ids"]
    with pytest.raises(ValueError, match="session_ids"):
        validate_nested_dataset(bad_ds, DEFAULT_FEATURE)

    # 2. Corrupt / empty session_ids
    bad_ds = dict(ds)
    bad_ds["session_ids"] = [""] * len(ds["labels"])
    with pytest.raises(ValueError, match="invalid or empty"):
        validate_nested_dataset(bad_ds, DEFAULT_FEATURE)

    # 3. Missing canonical snapshots
    bad_ds = dict(ds)
    del bad_ds["snapshots"]
    del bad_ds["mcmc_snapshots"]
    with pytest.raises(ValueError, match="canonical 'snapshots'"):
        validate_nested_dataset(bad_ds, DEFAULT_FEATURE)

    # 4. Non-finite snapshots (NaN)
    bad_ds = dict(ds)
    nan_snaps = ds["snapshots"].copy()
    nan_snaps[0, 0] = np.nan
    bad_ds["snapshots"] = nan_snaps
    with pytest.raises(ValueError, match="non-finite"):
        validate_nested_dataset(bad_ds, DEFAULT_FEATURE)

    # 5. Wrong snapshot shape
    bad_ds = dict(ds)
    bad_ds["snapshots"] = ds["snapshots"][:, :3]  # 3 features instead of 5
    with pytest.raises(ValueError, match="shape"):
        validate_nested_dataset(bad_ds, DEFAULT_FEATURE)

    # 6. Invalid pipeline semantics version
    bad_ds = dict(ds)
    bad_ds["pipeline_semantics_version"] = "1.0-legacy"
    with pytest.raises(ValueError, match="pipeline_semantics_version"):
        validate_nested_dataset(bad_ds, DEFAULT_FEATURE)


# ── Test 8: CLI Generation and Overwrite Protection ───────────


def test_cli_generation_and_overwrite_protection(tmp_path: Path):
    """CLI must generate valid nested priors, enforce honest provenance, and protect against accidental overwrite."""
    ds = _create_synthetic_dataset(n_animals=10, trials_per_animal=4, random_seed=42)
    ds_path = tmp_path / "ds.pt"
    torch.save(ds, ds_path)

    out_dir = tmp_path / "nested_out"

    # 1. First run: generate output
    cmd = [
        sys.executable,
        "scripts/evaluate_nested_prior.py",
        "--dataset",
        str(ds_path),
        "--output_dir",
        str(out_dir),
        "--split_seed",
        "77",
        "--epochs",
        "5",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed: {res.stderr}\n{res.stdout}"
    assert "Recording prefixes:" in res.stderr
    assert "animal identities across prefixes unverified" in res.stderr

    out_file = out_dir / "nested_split_seed77.pt"
    assert out_file.exists()

    result = load_artifact_bytes(out_file.read_bytes())
    assert "nested_priors" in result
    assert "train_priors" in result
    assert "val_priors" in result
    assert "train_indices" in result
    assert "val_indices" in result

    # Check finite, nonnegative, row sums == 1
    priors = result["nested_priors"]
    assert priors.shape == (len(ds["labels"]), 4)
    assert np.isfinite(priors).all()
    assert (priors >= 0.0).all()
    np.testing.assert_allclose(priors.sum(axis=1), 1.0, atol=1e-4)

    # Honest split-specific provenance (must NOT impersonate global OOF)
    prov = result["mcmc_prior_provenance"]
    assert "nested" in prov
    assert not prov.startswith("oof_5fold_animal_grouped_cv")

    # Source fingerprint must be non-empty SHA-256
    fingerprint = result.get("source_fingerprint", "")
    assert len(fingerprint) == 64

    # 2. Second run without --overwrite: must fail closed to prevent accidental overwrite
    res_overwrite_fail = subprocess.run(cmd, capture_output=True, text=True)
    assert res_overwrite_fail.returncode != 0
    assert "already exists" in res_overwrite_fail.stderr

    # 3. Third run with --overwrite: succeeds
    res_overwrite_ok = subprocess.run(cmd + ["--overwrite"], capture_output=True, text=True)
    assert res_overwrite_ok.returncode == 0


# ── Test 9: Snapshot Persistence and Strict Sequence Failure ──


@pytest.mark.parametrize("sequence_error", [False, True], ids=["valid", "corrupt"])
def test_snapshot_persistence_or_sequence_error_fails_closed(tmp_path: Path, sequence_error: bool):
    """Persist the full valid cohort; a corrupt sequence aborts with trial identity."""
    from scripts import prepare_data
    from tests.test_prior_grouping import _write_corpus

    raw_dir = tmp_path / "raw"
    _write_corpus(raw_dir, n_animals=4, blocks_per_animal=2)
    out_path = tmp_path / "prepared_ds.pt"
    orig_extract = prepare_data.extract_trial_sequence
    call_count = 0

    def _extract(trial_data, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if sequence_error and call_count == 3:
            raise ValueError("Simulated corrupted sequence")
        return orig_extract(trial_data, *args, **kwargs)

    with (
        mock.patch.object(prepare_data, "extract_trial_sequence", side_effect=_extract) as sequence_spy,
        mock.patch.object(prepare_data, "train_mcmc_cross_fitted",
                          wraps=prepare_data.train_mcmc_cross_fitted) as prior_spy,
        mock.patch.object(prepare_data.torch, "save", wraps=torch.save) as save_spy,
    ):
        if sequence_error:
            with pytest.raises(ValueError, match="Simulated corrupted sequence") as error:
                prepare_data.prepare_dataset(raw_dir, out_path, random_seed=42)
            assert sequence_spy.call_count == 3
            failed_trial = sequence_spy.call_args.args[0]
            assert failed_trial["trial_id"] == 2
            assert f"session={failed_trial['session_id']!r}" in str(error.value)
            assert "trial=2" in str(error.value)
            assert isinstance(error.value.__cause__, ValueError)
            prior_spy.assert_called_once()
            assert prior_spy.call_args.args[0].shape == (32, 5)
            save_spy.assert_not_called()
            assert not out_path.exists()
            return
        prepare_data.prepare_dataset(raw_dir, out_path, random_seed=42)
        save_spy.assert_called_once()

    saved = load_artifact_bytes(out_path.read_bytes())
    n_seqs = len(saved["X_seqs"])
    assert n_seqs == sequence_spy.call_count == 32
    expected_ids = [(directory.name, trial_id)
                    for directory in sorted(raw_dir.iterdir()) for trial_id in range(4)]
    assert list(zip(saved["session_ids"], saved["trial_ids"])) == expected_ids
    assert len(saved["snapshots"]) == n_seqs
    assert len(saved["mcmc_snapshots"]) == n_seqs
    assert len(saved["mcmc_priors"]) == n_seqs
    assert len(saved["labels"]) == n_seqs
    assert saved["snapshots"].shape == (n_seqs, 5)
    assert np.isfinite(saved["snapshots"]).all()
    prior_spy.assert_called_once()
    np.testing.assert_array_equal(saved["snapshots"], prior_spy.call_args.args[0])
    np.testing.assert_array_equal(saved["mcmc_snapshots"], saved["snapshots"])
    np.testing.assert_array_equal(saved["labels"], prior_spy.call_args.args[1])

def test_authoritative_byte_snapshot_survives_same_path_replacement(tmp_path, monkeypatch):
    """A producer must stamp the exact A snapshot it used when the path becomes B."""
    import hashlib
    import io
    from scripts import evaluate_nested_prior as producer
    from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint

    source_path = tmp_path / "source.pt"
    source_a = _create_synthetic_dataset(n_animals=12, trials_per_animal=5, random_seed=42)
    torch.save(source_a, source_path)
    bytes_a = source_path.read_bytes()
    source_b = dict(source_a, snapshots=source_a["snapshots"] + 100.0)
    replacement_path = tmp_path / "replacement.pt"
    torch.save(source_b, replacement_path)
    bytes_b = replacement_path.read_bytes()
    digest_a = hashlib.sha256(bytes_a).hexdigest()
    digest_b = hashlib.sha256(bytes_b).hexdigest()
    assert digest_a != digest_b
    assert np.array_equal(source_a["labels"], source_b["labels"])
    assert np.array_equal(source_a["session_ids"], source_b["session_ids"])
    real_load = torch.load
    reads = []

    def replace_after_deserialization(source, *args, **kwargs):
        loaded = real_load(source, *args, **kwargs)
        is_dataset = (source.getvalue() == bytes_a if isinstance(source, io.BytesIO)
                      else isinstance(source, (str, Path))
                      and Path(source).resolve() == source_path.resolve())
        if is_dataset and not reads:
            reads.append(loaded)
            source_path.write_bytes(bytes_b)
        return loaded

    monkeypatch.setattr(torch, "load", replace_after_deserialization)
    observed_train = []
    observed_val = []

    class FakeFold:
        def predict_proba(self, snapshots):
            observed_val.append(np.asarray(snapshots).copy())
            return np.tile([0.4, 0.3, 0.2, 0.1], (len(snapshots), 1))

    def synthetic_fold(*args, **kwargs):
        observed_train.append(np.asarray(kwargs["snapshots"]).copy())
        return (np.tile([0.4, 0.3, 0.2, 0.1], (len(kwargs["snapshots"]), 1)),
                [FakeFold()], [])

    monkeypatch.setattr(producer, "train_mcmc_cross_fitted", synthetic_fold)
    result = producer.generate_nested_priors(
        dataset=None, source_dataset_path=source_path, n_inner_folds=3,
        val_split=0.2, verbose=False,
    )
    assert len(reads) == len(observed_train) == len(observed_val) == 1
    assert result["source_fingerprint"] == digest_a != digest_b
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == digest_b
    np.testing.assert_array_equal(observed_train[0], source_a["snapshots"][result["train_indices"]])
    np.testing.assert_array_equal(observed_val[0], source_a["snapshots"][result["val_indices"]])
    assert not np.array_equal(observed_train[0], source_b["snapshots"][result["train_indices"]])
    assert result["nested_priors"].shape == (len(source_a["labels"]), 4)
    # Read A again through the public helper: its digest describes the same
    # deserialized snapshot, independent of any subsequent path contents.
    source_path.write_bytes(bytes_a)
    reads.clear()
    loaded, fingerprint = load_dataset_with_fingerprint(source_path)
    assert fingerprint == digest_a != digest_b
    np.testing.assert_array_equal(loaded["snapshots"], source_a["snapshots"])
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == digest_b


def test_generator_in_memory_comparison_still_rejects_mismatched_snapshots(tmp_path):
    source = _create_synthetic_dataset(n_animals=12, trials_per_animal=5, random_seed=42)
    source_path = _write_ds_file(tmp_path, source)
    mismatched = dict(source, snapshots=source["snapshots"] + 100.0)
    with pytest.raises(ValueError, match="in-memory/file mismatch"):
        generate_nested_priors(dataset=mismatched, source_dataset_path=source_path, verbose=False)


@pytest.mark.parametrize("folds", [0, 1, 2, 5])
@pytest.mark.parametrize("grouping,status", [
    ("recording_prefix", "unverified"), ("animal", "historical_unknown"),
])
def test_prior_lineage_contract_oof_folds(tmp_path, folds, grouping, status):
    """Public dataset provenance rejects methods that cannot be cross-fitted."""
    from nsmor.model_utils import validate_dataset_provenance

    dataset = _create_synthetic_dataset(n_animals=6, trials_per_animal=4)
    dataset["mcmc_prior_provenance"] = f"oof_{folds}fold_{grouping}_grouped_cv"
    dataset["animal_identity_status"] = status
    if folds < 2:
        with pytest.raises(RuntimeError, match="at least 2 folds"):
            validate_dataset_provenance(dataset, tmp_path / "synthetic.pt")
    else:
        assert validate_dataset_provenance(dataset, tmp_path / "synthetic.pt") == status

def _prior_lineage_contract_inputs(tmp_path):
    """Persist synthetic source bytes and a modern sidecar without fitting models."""
    from nsmor.pipeline.grouping import animal_keys_of, grouped_train_val_split

    dataset = _create_synthetic_dataset(n_animals=6, trials_per_animal=4)
    dataset.update(mcmc_prior_provenance="oof_2fold_recording_prefix_grouped_cv",
                   animal_identity_status="unverified")
    source = _write_ds_file(tmp_path, dataset)
    sessions = dataset["session_ids"]
    prefixes = animal_keys_of(sessions)
    train, val = grouped_train_val_split(sessions, len(sessions), 0.25, 3)
    train_prefixes = sorted(set(prefixes[train].tolist()))
    val_prefixes = sorted(set(prefixes[val].tolist()))
    artifact = {
        "nested_priors": np.full((len(sessions), 4), 0.25),
        "train_indices": train, "val_indices": val, "is_nested_cv": True,
        "source_fingerprint": compute_source_fingerprint(source),
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "split_seed": 3, "val_split": 0.25, "n_inner_folds": 2,
        "mcmc_prior_provenance": "nested_outer_seed3_inner_2fold_recording_prefix_grouped",
        "animal_identity_status": "unverified",
        "recording_prefix_keys": prefixes.tolist(),
        "train_recording_prefixes": train_prefixes,
        "val_recording_prefixes": val_prefixes,
        "n_train_recording_prefixes": len(train_prefixes),
        "n_val_recording_prefixes": len(val_prefixes),
    }
    return source, sessions, artifact


@pytest.mark.parametrize("field,value", [
    ("mcmc_prior_provenance", "nested_outer_seed999_inner_2fold_recording_prefix_grouped"),
    ("mcmc_prior_provenance", "nested_outer_seed3_inner_3fold_recording_prefix_grouped"),
    ("mcmc_prior_provenance", "nested_outer_seed3_inner_0fold_recording_prefix_grouped"),
    ("mcmc_prior_provenance", "nested_outer_seed3_inner_1fold_recording_prefix_grouped"),
    ("n_inner_folds", None), ("n_inner_folds", True), ("n_inner_folds", 2.0),
    ("n_inner_folds", 0), ("n_inner_folds", 1),
])
def test_prior_lineage_contract_nested_method(tmp_path, field, value):
    """The public loader must reject tag/structured-method contradictions."""
    source, sessions, artifact = _prior_lineage_contract_inputs(tmp_path)
    artifact[field] = value
    if value is None:
        del artifact[field]
    sidecar = tmp_path / "nested.pt"
    torch.save(artifact, sidecar)
    with pytest.raises(ValueError, match="at least 2 folds|n_inner_folds|split_seed"):
        load_nested_prior_split(sidecar, source, len(sessions), sessions)


def test_prior_lineage_contract_modern_roundtrip(tmp_path):
    source, sessions, artifact = _prior_lineage_contract_inputs(tmp_path)
    sidecar = tmp_path / "nested.pt"
    torch.save(artifact, sidecar)
    train, val, priors, info = load_nested_prior_split(sidecar, source, len(sessions), sessions)
    np.testing.assert_array_equal(train, artifact["train_indices"])
    np.testing.assert_array_equal(val, artifact["val_indices"])
    np.testing.assert_array_equal(priors, artifact["nested_priors"])
    assert info["animal_identity_status"] == "unverified"
    assert info["mcmc_prior_provenance"] == "nested_outer_seed3_inner_2fold_recording_prefix_grouped"
    assert info["split_seed"] == 3 and info["n_inner_folds"] == 2

_PREFIX_LINEAGE_FIELDS = (
    "recording_prefix_keys", "train_recording_prefixes", "val_recording_prefixes",
    "n_train_recording_prefixes", "n_val_recording_prefixes",
)


@pytest.mark.parametrize("strip_prefix_fields", [False, True])
@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize("structured_folds", [False, True])
@pytest.mark.parametrize("status", [None, "historical_unknown"])
def test_prior_lineage_contract_historical_requires_trusted_bytes(
        tmp_path, strip_prefix_fields, trusted, structured_folds, status):
    """An old label never authorizes compatibility; an external byte pin does."""
    source, sessions, artifact = _prior_lineage_contract_inputs(tmp_path)
    artifact["mcmc_prior_provenance"] = "nested_outer_seed3_inner_2fold_animal_grouped"
    if status is None:
        del artifact["animal_identity_status"]  # Original historical producer spelling.
    else:
        artifact["animal_identity_status"] = status
    if not structured_folds:
        del artifact["n_inner_folds"]  # Original sidecars embedded folds only in the tag.
    if strip_prefix_fields:
        for field in _PREFIX_LINEAGE_FIELDS:
            del artifact[field]
    sidecar = tmp_path / "historical.pt"
    torch.save(artifact, sidecar)
    if not trusted:
        with pytest.raises(ValueError, match="trusted_historical_artifact_sha256"):
            load_nested_prior_split(sidecar, source, len(sessions), sessions)
        return
    digest = compute_source_fingerprint(sidecar)
    train, val, priors, info = load_nested_prior_split(
        sidecar, source, len(sessions), sessions,
        trusted_historical_artifact_sha256=digest,
    )
    assert info["animal_identity_status"] == "historical_unknown"
    assert info["nested_prior_artifact_sha256"] == digest
    assert info["split_seed"] == 3 and info["n_inner_folds"] == 2
    np.testing.assert_array_equal(train, artifact["train_indices"])
    np.testing.assert_array_equal(val, artifact["val_indices"])
    np.testing.assert_array_equal(priors, artifact["nested_priors"])


@pytest.mark.parametrize("missing_field", _PREFIX_LINEAGE_FIELDS)
def test_prior_lineage_contract_modern_requires_prefix_evidence(tmp_path, missing_field):
    source, sessions, artifact = _prior_lineage_contract_inputs(tmp_path)
    del artifact[missing_field]
    sidecar = tmp_path / "modern.pt"
    torch.save(artifact, sidecar)
    with pytest.raises(ValueError, match=f"missing {missing_field}"):
        load_nested_prior_split(sidecar, source, len(sessions), sessions)


def test_prior_lineage_contract_modern_cannot_use_historical_trust(tmp_path):
    source, sessions, artifact = _prior_lineage_contract_inputs(tmp_path)
    for field in _PREFIX_LINEAGE_FIELDS:
        del artifact[field]
    sidecar = tmp_path / "modern.pt"
    torch.save(artifact, sidecar)
    with pytest.raises(ValueError, match="historical"):
        load_nested_prior_split(
            sidecar, source, len(sessions), sessions,
            trusted_historical_artifact_sha256=compute_source_fingerprint(sidecar),
        )


def test_prior_lineage_contract_historical_pin_rejects_changed_bytes(tmp_path):
    source, sessions, artifact = _prior_lineage_contract_inputs(tmp_path)
    artifact.update(mcmc_prior_provenance="nested_outer_seed3_inner_2fold_animal_grouped",
                    animal_identity_status="historical_unknown")
    sidecar = tmp_path / "historical.pt"
    torch.save(artifact, sidecar)
    trusted_digest = compute_source_fingerprint(sidecar)
    # Same path and valid probabilities, different bytes than the trusted reference.
    artifact["nested_priors"][:] = [0.1, 0.2, 0.3, 0.4]
    torch.save(artifact, sidecar)
    with pytest.raises(ValueError, match="historical.*SHA-256"):
        load_nested_prior_split(
            sidecar, source, len(sessions), sessions,
            trusted_historical_artifact_sha256=trusted_digest,
        )


@pytest.mark.parametrize("tag", [
    "nested_outer_seed999_inner_2fold_animal_grouped",
    "nested_outer_seed3_inner_3fold_animal_grouped",
    "nested_outer_seed3_inner_0fold_animal_grouped",
    "nested_outer_seed3_inner_1fold_animal_grouped",
])
def test_prior_lineage_contract_historical_trust_keeps_method_checks(tmp_path, tag):
    source, sessions, artifact = _prior_lineage_contract_inputs(tmp_path)
    artifact.update(mcmc_prior_provenance=tag, animal_identity_status="historical_unknown")
    sidecar = tmp_path / "historical.pt"
    torch.save(artifact, sidecar)
    with pytest.raises(ValueError, match="at least 2 folds|split_seed/n_inner_folds"):
        load_nested_prior_split(
            sidecar, source, len(sessions), sessions,
            trusted_historical_artifact_sha256=compute_source_fingerprint(sidecar),
        )

def test_prior_lineage_contract_modern_requires_structured_folds(tmp_path):
    source, sessions, artifact = _prior_lineage_contract_inputs(tmp_path)
    del artifact["n_inner_folds"]
    sidecar = tmp_path / "modern.pt"
    torch.save(artifact, sidecar)
    with pytest.raises(ValueError, match="n_inner_folds"):
        load_nested_prior_split(sidecar, source, len(sessions), sessions)