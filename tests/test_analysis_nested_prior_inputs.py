"""Checkpoint-bound nested prior inputs for every C-H analysis entry point."""

import hashlib
import importlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from nsmor.config import FeatureConfig, PIPELINE_SEMANTICS_VERSION
from nsmor.model_nsmor_core import NSMoRCore
from nsmor.pipeline.grouping import grouped_train_val_split, animal_keys_of
from nsmor.pipeline.nested_prior import (
    compute_source_fingerprint, load_artifact_bytes, load_dataset_with_fingerprint,
    load_nested_prior_split,
)
from nsmor.analysis.prediction_units import load_model_from_checkpoint, prediction_to_physical


@pytest.fixture(params=[4.0, 10.0])
def nested_inputs(tmp_path, request):
    n = 8
    etl = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    nested = np.tile([0.0, 1.0, 0.0, 0.0], (n, 1))
    nested[1::2] = [0.0, 0.0, 1.0, 0.0]
    assert etl.shape == nested.shape == (n, 4)
    assert np.all(np.any(etl != nested, axis=1))
    sessions = [f"animal_{i // 2}_session_{i % 2 + 1}" for i in range(n)]
    dataset = {
        "X_seqs": [np.zeros((6, 8), dtype=np.float32) for _ in range(n)],
        "Y_seqs": [np.zeros(6, dtype=np.float32) for _ in range(n)],
        "labels": np.zeros(n, dtype=np.int64),
        "lengths": np.full(n, 6, dtype=np.int64),
        "anchor_frames": [2] * n,
        "mcmc_priors": etl,
        "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
        "session_ids": sessions,
        "feature_config": FeatureConfig(),
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
    }
    dataset_path = tmp_path / "dataset.pt"
    torch.save(dataset, dataset_path)
    fingerprint = compute_source_fingerprint(dataset_path)
    train_idx, val_idx = grouped_train_val_split(sessions, n, val_split=0.25, random_seed=9)

    def save_artifact(path, matrix, seed, fraction):
        train, val = grouped_train_val_split(sessions, n, val_split=fraction, random_seed=seed)
        prefixes = animal_keys_of(sessions)
        torch.save({
            "nested_priors": matrix, "train_indices": train, "val_indices": val,
            "recording_prefix_keys": prefixes,
            "train_recording_prefixes": sorted(set(prefixes[train].tolist())),
            "val_recording_prefixes": sorted(set(prefixes[val].tolist())),
            "n_train_recording_prefixes": len(set(prefixes[train].tolist())),
            "n_val_recording_prefixes": len(set(prefixes[val].tolist())),
            "train_priors": matrix[train], "val_priors": matrix[val],
            "source_fingerprint": fingerprint, "is_nested_cv": True,
            "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
            "split_seed": seed, "val_split": fraction, "n_inner_folds": 2,
            "mcmc_prior_provenance": f"nested_outer_seed{seed}_inner_2fold_recording_prefix_grouped",
            "animal_identity_status": "unverified",
        }, path)

    artifact = tmp_path / "nested.pt"
    save_artifact(artifact, nested, 9, 0.25)
    wrong_artifact = tmp_path / "other_valid_nested.pt"
    save_artifact(wrong_artifact, np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)), 19, 0.5)
    checkpoint = tmp_path / "best_model.pth"
    model = NSMoRCore(hidden_dim=4, dt_ms=request.param, dropout=0.0).eval()
    payload = {
        "model_state_dict": model.state_dict(),
        "config": {"model": {"hidden_dim": 4, "dt_ms": request.param, "dropout": 0.0}},
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "dataset_path": str(dataset_path), "is_nested_cv": True,
        "nested_prior_artifact": str(artifact),
        "nested_prior_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "nested_prior_fingerprint": fingerprint,
        "dataset_source_sha256": fingerprint,
        "nested_split_seed": 9, "nested_val_split": 0.25,
        "validation_scope": "nested_outer_validation",
        "mcmc_prior_provenance": "nested_outer_seed9_inner_2fold_recording_prefix_grouped",
            "animal_identity_status": "unverified",
    }
    torch.save(payload, checkpoint)
    assert len(val_idx) > 0 and len(train_idx) + len(val_idx) == n
    return dataset_path, artifact, wrong_artifact, checkpoint, dataset, nested, val_idx, payload


def phase_inputs(phase, checkpoint, dataset, artifact, model):
    """Use each producer's public loader and the production checkpoint path."""
    if phase == "G":
        from scripts.simulate_psychophysics import load_validation_data
        return load_validation_data(
            torch.device("cpu"), dataset_path=str(dataset), max_seq_len=None,
            nested_prior_artifact=artifact, checkpoint_model=model,
            val_split=0.75, random_seed=999,
        )
    if phase == "H":
        from scripts.analyze_gating import load_model_and_dataset
        return load_model_and_dataset(
            checkpoint, dataset, max_seq_len=None, nested_prior_artifact=artifact,
        )
    from scripts import analyze_dynamics, simulate_lesion, analyze_jacobian, analyze_integration
    module = {"C": analyze_dynamics, "D": simulate_lesion,
              "E": analyze_jacobian, "F": analyze_integration}[phase]
    return module.load_dataset(
        dataset, max_seq_len=None, nested_prior_artifact=artifact,
        checkpoint_model=model,
    )


@pytest.mark.parametrize("scope", [None, "diagnostic_global_oof"])
def test_modern_nested_rejects_missing_or_retagged_scope(nested_inputs, tmp_path, scope):
    dataset, _, _, _, _, _, _, payload = nested_inputs
    altered = dict(payload)
    if scope is None:
        altered.pop("validation_scope")
    else:
        altered["validation_scope"] = scope
    checkpoint = tmp_path / "wrong_nested_scope.pth"
    torch.save(altered, checkpoint)
    model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
    with pytest.raises(ValueError, match="validation_scope must be nested_outer_validation"):
        phase_inputs("C", checkpoint, dataset, None, model)


@pytest.mark.parametrize("scope", [None, "nested_outer_validation"])
def test_modern_nonnested_rejects_missing_or_retagged_scope(nested_inputs, tmp_path, scope):
    dataset, _, _, _, source, _, _, payload = nested_inputs
    source = dict(source, mcmc_prior_provenance="oof_5fold_recording_prefix_grouped_cv",
                  animal_identity_status="unverified")
    torch.save(source, dataset)
    altered = dict(payload, is_nested_cv=False, nested_prior_artifact="",
                   nested_prior_artifact_sha256="", nested_prior_fingerprint="",
                   nested_split_seed=-1, nested_val_split=-1.0,
                   mcmc_prior_provenance=source["mcmc_prior_provenance"],
                   animal_identity_status="unverified",
                   dataset_source_sha256=compute_source_fingerprint(dataset))
    if scope is None:
        altered.pop("validation_scope")
    else:
        altered["validation_scope"] = scope
    checkpoint = tmp_path / "wrong_diagnostic_scope.pth"
    torch.save(altered, checkpoint)
    model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
    with pytest.raises(ValueError, match="validation_scope must be diagnostic_global_oof"):
        phase_inputs("C", checkpoint, dataset, None, model)


@pytest.mark.parametrize("phase", list("CDEFGH"))
def test_each_analysis_loader_requests_zero_workers(phase, nested_inputs, monkeypatch):
    """Every analysis loader must run single-process.

    The full corpus is already resident in RAM (``NSMoRDataset.__init__``
    deep-copies every sequence array), so ``num_workers=-1`` auto-scaling to
    4 forkserver workers only replicates that memory into each child.  That
    replication drove the capped production run past its cgroup limit.  This
    pins the request at each producer's public loader seam.
    """
    import multiprocessing as mp

    import nsmor.dataloader_factory as factory

    dataset, _, _, checkpoint, _, _, _, _ = nested_inputs
    model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))

    requested: list = []
    real = factory.create_optimized_dataloader

    def spy(ds, **kwargs):
        requested.append(kwargs.get("num_workers"))
        return real(ds, **kwargs)

    name = {"C": "analyze_dynamics", "D": "simulate_lesion",
            "E": "analyze_jacobian", "F": "analyze_integration",
            "G": "simulate_psychophysics", "H": "analyze_gating"}[phase]
    module = importlib.import_module(f"scripts.{name}")
    # psychophysics imports the factory inside the function; the rest bind it
    # at module scope.  Patch both so the seam is observed either way.
    monkeypatch.setattr(factory, "create_optimized_dataloader", spy)
    if hasattr(module, "create_optimized_dataloader"):
        monkeypatch.setattr(module, "create_optimized_dataloader", spy)

    before = set(mp.active_children())
    phase_inputs(phase, checkpoint, dataset, None, model)

    assert requested == [0], f"{phase} requested num_workers={requested}"
    assert set(mp.active_children()) == before, f"{phase} forked worker processes"


@pytest.mark.parametrize("phase", list("CDEFGH"))
def test_each_analysis_model_receives_checkpoint_nested_prior(phase, nested_inputs):
    dataset, artifact, wrong, checkpoint, source, nested, val_idx, _ = nested_inputs
    model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
    result = phase_inputs(phase, checkpoint, dataset, None, model)  # Packaged CLI omits the option.
    if phase == "G":
        assert result.meta.val_indices == val_idx.tolist()  # Saved outer split, not caller's seed.
        row = int(val_idx[0])
        X = result.X[0]
    else:
        if phase == "H":
            model, loader = result[:2]
        else:
            loader = result[0]
        row = 1
        X = loader.dataset[row][-2]  # Works with both metadata and legacy tuples.
    assert X.shape == (6, 8)
    assert np.all(np.isfinite(X.numpy()))
    np.testing.assert_allclose(X[:, 4:8].numpy(), np.tile(nested[row], (6, 1)))
    assert not np.array_equal(X[0, 4:8].numpy(), source["mcmc_priors"][row])

    observed = []
    handle = model.backend.register_forward_pre_hook(
        lambda _backend, args: observed.append(args[1].detach().cpu().clone())
    )
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            prediction = model(X.unsqueeze(0).to(device), torch.tensor([len(X)], device=device))
        assert prediction.shape == (1, 6)
    finally:
        handle.remove()
    assert len(observed) == 1 and observed[0].shape == (1, 6, 4)
    np.testing.assert_allclose(observed[0][0].numpy(), np.tile(nested[row], (6, 1)))

    phase_inputs(phase, checkpoint, dataset, artifact, model)  # Explicit matching override works.
    with pytest.raises(ValueError, match="artifact path does not match checkpoint"):
        phase_inputs(phase, checkpoint, dataset, wrong, model)


def test_nested_checkpoint_checks_split_source_and_legacy_fallback(nested_inputs, tmp_path):
    from scripts.analyze_dynamics import load_dataset

    dataset, artifact, _, checkpoint, source, nested, _, payload = nested_inputs
    model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
    mismatch = dict(payload, nested_split_seed=123)
    wrong_ckpt = tmp_path / "wrong_seed.pth"
    torch.save(mismatch, wrong_ckpt)
    wrong_model = load_model_from_checkpoint(wrong_ckpt, torch.device("cpu"))
    with pytest.raises(ValueError, match="split seed"):
        load_dataset(dataset, max_seq_len=None, nested_prior_artifact=artifact,
                     checkpoint_model=wrong_model)

    legacy_ckpt = tmp_path / "legacy.pth"
    torch.save(dict(payload, is_nested_cv=False, nested_prior_artifact="",
                    nested_prior_artifact_sha256="", nested_prior_fingerprint="", nested_split_seed=-1,
                    nested_val_split=-1.0,
                    mcmc_prior_provenance="global_oof_animal_grouped_cv",
                    animal_identity_status="historical_unknown",
                    validation_scope="diagnostic_global_oof"), legacy_ckpt)
    legacy_model = load_model_from_checkpoint(legacy_ckpt, torch.device("cpu"))
    with pytest.raises(ValueError, match="trusted_historical_checkpoint_sha256"):
        load_dataset(dataset, max_seq_len=None, checkpoint_model=legacy_model)
    from scripts.simulate_psychophysics import load_validation_data

    captured_sha = hashlib.sha256(legacy_ckpt.read_bytes()).hexdigest()
    validation = load_validation_data(
        torch.device("cpu"), dataset_path=str(dataset), max_seq_len=None,
        checkpoint_model=legacy_model, trusted_historical_checkpoint_sha256=captured_sha,
    )
    row = validation.meta.val_indices[0]
    np.testing.assert_allclose(validation.X[0, 0, 4:8].numpy(), source["mcmc_priors"][row])
    with pytest.raises(ValueError, match="Non-nested checkpoint"):
        load_validation_data(
            torch.device("cpu"), dataset_path=str(dataset), max_seq_len=None,
            nested_prior_artifact=artifact, checkpoint_model=legacy_model,
            trusted_historical_checkpoint_sha256=captured_sha,
        )

    source["labels"][0] = 1
    torch.save(source, dataset)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_dataset(dataset, max_seq_len=None, nested_prior_artifact=artifact,
                     checkpoint_model=model)


def test_analysis_rejects_valid_swap_during_load_even_if_path_hashes_match(
        nested_inputs, tmp_path, monkeypatch):
    """The matrix deserialized between path checks must match checkpoint-bound bytes."""
    from nsmor.analysis import analysis_priors

    dataset_path, artifact_path, _, checkpoint, dataset, original, _, payload = nested_inputs
    model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
    bound_digest = payload["nested_prior_artifact_sha256"]
    original_bytes = artifact_path.read_bytes()
    saved = torch.load(artifact_path, weights_only=False)
    replacement = np.tile([0.0, 0.0, 0.0, 1.0], (len(original), 1))
    changed = dict(saved, nested_priors=replacement,
                   train_priors=replacement[saved["train_indices"]],
                   val_priors=replacement[saved["val_indices"]])
    other_path = tmp_path / "valid_swap.pt"
    torch.save(changed, other_path)
    swapped_bytes = other_path.read_bytes()
    assert hashlib.sha256(swapped_bytes).hexdigest() != bound_digest
    loaded = []
    real_load = analysis_priors.load_nested_prior_split

    def swapped_load(*args, **kwargs):
        artifact_path.write_bytes(swapped_bytes)
        try:
            result = real_load(*args, **kwargs)
            loaded.append(result[2])
            return result
        finally:
            artifact_path.write_bytes(original_bytes)

    monkeypatch.setattr(analysis_priors, "load_nested_prior_split", swapped_load)
    with pytest.raises(ValueError, match="Nested artifact SHA-256 mismatch"):
        analysis_priors.load_analysis_priors(
            dataset, dataset_path, artifact_path, model,
            qc_sealed_nested_prior_sha256=bound_digest,
            loaded_source_fingerprint=payload["nested_prior_fingerprint"],
        )
    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0], replacement)
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == bound_digest


def test_same_path_valid_artifact_replacement_refused(nested_inputs, tmp_path):
    from scripts.analyze_dynamics import load_dataset

    dataset, artifact, _, checkpoint, _, _, _, payload = nested_inputs
    model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
    original_hash = payload["nested_prior_artifact_sha256"]
    original_sidecar = torch.load(artifact, weights_only=False)
    legacy_ckpt = tmp_path / "v4_nested.pth"
    torch.save({k: v for k, v in payload.items() if k != "nested_prior_artifact_sha256"}, legacy_ckpt)
    v4_model = load_model_from_checkpoint(legacy_ckpt, torch.device("cpu"))
    with pytest.raises(ValueError, match="lacks embedded artifact SHA-256"):
        load_dataset(dataset, max_seq_len=None, checkpoint_model=v4_model)
    with pytest.raises(ValueError, match="lacks embedded artifact SHA-256"):
        load_dataset(dataset, max_seq_len=None, checkpoint_model=v4_model,
                     qc_sealed_nested_prior_sha256=original_hash)
    replacement = np.tile([0.0, 0.0, 0.0, 1.0], (8, 1))
    changed = dict(original_sidecar, nested_priors=replacement,
                   train_priors=replacement[original_sidecar["train_indices"]],
                   val_priors=replacement[original_sidecar["val_indices"]])
    torch.save(changed, artifact)  # Same path, split, source fingerprint, provenance.
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() != original_hash
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        load_dataset(dataset, max_seq_len=None, nested_prior_artifact=artifact,
                     checkpoint_model=model)

    with pytest.raises(ValueError, match="lacks embedded artifact SHA-256"):
        load_dataset(dataset, max_seq_len=None, checkpoint_model=v4_model,
                     qc_sealed_nested_prior_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())


def test_legacy_nested_shared_seam_rejects_compatible_replacement(nested_inputs, tmp_path):
    from nsmor.analysis.analysis_priors import load_analysis_priors

    dataset_path, artifact, _, _, dataset, original, _, payload = nested_inputs
    saved = torch.load(artifact, weights_only=False)
    replacement = np.tile([0.0, 0.0, 0.0, 1.0], (len(original), 1))
    torch.save(dict(saved, nested_priors=replacement,
                    train_priors=replacement[saved["train_indices"]],
                    val_priors=replacement[saved["val_indices"]]), artifact)
    forged = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert forged != payload["nested_prior_artifact_sha256"]
    # The replacement remains a valid split/source sidecar; its current digest proves no training history.
    _, _, accepted_matrix, _ = load_nested_prior_split(
        artifact, dataset_path, len(original), dataset["session_ids"])
    np.testing.assert_allclose(accepted_matrix, replacement)
    legacy = tmp_path / "legacy_nested.pth"
    torch.save({k: v for k, v in payload.items() if k != "nested_prior_artifact_sha256"}, legacy)
    model = load_model_from_checkpoint(legacy, torch.device("cpu"))
    with pytest.raises(ValueError, match="lacks embedded artifact SHA-256"):
        load_analysis_priors(dataset, dataset_path, model=model,
                             qc_sealed_nested_prior_sha256=forged,
                             loaded_source_fingerprint=payload["nested_prior_fingerprint"])


def test_legacy_nested_cli_digest_does_not_authenticate_checkpoint(nested_inputs, tmp_path, monkeypatch):
    from scripts import analyze_dynamics

    dataset_path, artifact, _, _, _, _, _, payload = nested_inputs
    legacy = tmp_path / "legacy_nested.pth"
    torch.save({k: v for k, v in payload.items() if k != "nested_prior_artifact_sha256"}, legacy)
    supplied = hashlib.sha256(artifact.read_bytes()).hexdigest()
    forwarded = []

    def check_loader(**kwargs):
        forwarded.append(kwargs["qc_sealed_nested_prior_sha256"])
        model = load_model_from_checkpoint(kwargs["checkpoint_path"], torch.device("cpu"))
        analyze_dynamics.load_dataset(kwargs["dataset_path"], max_seq_len=None,
                                      checkpoint_model=model,
                                      qc_sealed_nested_prior_sha256=kwargs["qc_sealed_nested_prior_sha256"])

    monkeypatch.setattr(analyze_dynamics, "run_analysis", check_loader)
    with pytest.raises(ValueError, match="lacks embedded artifact SHA-256"):
        analyze_dynamics.main(["--checkpoint", str(legacy), "--dataset", str(dataset_path),
                               "--qc_sealed_nested_prior_sha256", supplied,
                               "--output", str(tmp_path / "out.png")])
    assert forwarded == [supplied]


@pytest.mark.parametrize("swap_after_first_read", [False, True])
def test_checkpoint_snapshot_binds_weights_units_and_downstream_priors(
        nested_inputs, tmp_path, monkeypatch, swap_after_first_read):
    """Two individually valid checkpoints must never produce a mixed analysis model."""
    from nsmor.analysis.analysis_priors import load_analysis_priors

    dataset_path, artifact_a, artifact_b, checkpoint_path, dataset, priors_a, val_a, base = nested_inputs
    dt_a = base["config"]["model"]["dt_ms"]
    dt_b = 10.0 if dt_a == 4.0 else 4.0
    checkpoint_a = dict(base, config={**base["config"], "training": {"normalize_targets": True},
                                     "finetune": {"freeze_modules": ["sensory_encoder"]}},
                        target_mean=5.0, target_std=2.0, target_clip_cm_s=80.0)
    torch.save(checkpoint_a, checkpoint_path)
    checkpoint_b = dict(
        checkpoint_a,
        model_state_dict=NSMoRCore(hidden_dim=6, dt_ms=dt_b, dropout=0.0).state_dict(),
        config={"model": {"hidden_dim": 6, "dt_ms": dt_b, "dropout": 0.0},
                "training": {"normalize_targets": True},
                "finetune": {"freeze_modules": []}},
        nested_prior_artifact=str(artifact_b),
        nested_prior_artifact_sha256=hashlib.sha256(artifact_b.read_bytes()).hexdigest(),
        nested_split_seed=19, nested_val_split=0.5,
        mcmc_prior_provenance="nested_outer_seed19_inner_2fold_recording_prefix_grouped",
        target_mean=-7.0, target_std=3.0, target_clip_cm_s=120.0,
    )
    replacement_path = tmp_path / "valid_checkpoint_b.pth"
    torch.save(checkpoint_b, replacement_path)
    replacement_bytes = replacement_path.read_bytes()

    # The replacement and its sidecar work when loaded together.
    model_b = load_model_from_checkpoint(replacement_path, torch.device("cpu"))
    priors_b, val_b = load_analysis_priors(dataset, dataset_path, model=model_b,
                                           loaded_source_fingerprint=base["nested_prior_fingerprint"])
    np.testing.assert_allclose(priors_b, np.tile([0.0, 0.0, 0.0, 1.0], (len(priors_a), 1)))
    np.testing.assert_array_equal(val_b, torch.load(artifact_b, weights_only=False)["val_indices"])
    assert model_b.hidden_dim == 6 and model_b.dt_ms == dt_b
    assert all(p.requires_grad for p in model_b.frontend.sensory_encoder.parameters())
    assert not model_b.training
    assert model_b.target_mean == -7.0 and model_b.target_std == 3.0
    for key, weight in checkpoint_b["model_state_dict"].items():
        torch.testing.assert_close(model_b.state_dict()[key], weight, rtol=0, atol=0)

    real_read_bytes = Path.read_bytes
    checkpoint_reads = []

    def swap_after_read(path):
        payload = real_read_bytes(path)
        if path == checkpoint_path:
            checkpoint_reads.append(path)
            if len(checkpoint_reads) == 1 and swap_after_first_read:
                replacement_path.replace(checkpoint_path)  # Same path, valid B bytes after A was read.
        return payload

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", swap_after_read)
        model_a = load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))

    if swap_after_first_read:
        assert checkpoint_path.read_bytes() == replacement_bytes
    assert model_a.hidden_dim == 4 and model_a.dt_ms == dt_a
    assert all(not p.requires_grad for p in model_a.frontend.sensory_encoder.parameters())
    assert not model_a.training
    for key, weight in checkpoint_a["model_state_dict"].items():
        torch.testing.assert_close(model_a.state_dict()[key], weight, rtol=0, atol=0)
    assert model_a.analysis_checkpoint_lineage["nested_prior_artifact"] == str(artifact_a)
    assert model_a.analysis_checkpoint_lineage["nested_prior_artifact_sha256"] == checkpoint_a[
        "nested_prior_artifact_sha256"]
    for consumer in (model_a, model_a.backend):
        assert consumer.target_mean == 5.0 and consumer.target_std == 2.0
        assert consumer.target_clip_cm_s == 80.0
    torch.testing.assert_close(prediction_to_physical(torch.tensor([[0.0, 1.0]]), model_a),
                               torch.tensor([[5.0, 7.0]]))
    priors, val_indices = load_analysis_priors(dataset, dataset_path, model=model_a,
                                               loaded_source_fingerprint=base["nested_prior_fingerprint"])
    np.testing.assert_allclose(priors, priors_a)
    np.testing.assert_array_equal(val_indices, val_a)
    with pytest.raises(ValueError, match="artifact path does not match checkpoint"):
        load_analysis_priors(dataset, dataset_path, artifact_b, model_a,
                             loaded_source_fingerprint=base["nested_prior_fingerprint"])
    assert len(checkpoint_reads) == 1  # No second checkpoint path read for config or weights.


@pytest.mark.parametrize("missing", ["pipeline_semantics_version", "dt_ms"])
def test_preloaded_checkpoint_keeps_canonical_provenance_guards(nested_inputs, missing):
    from nsmor.model_utils import load_model_from_checkpoint as canonical_load

    _, _, _, path, _, _, _, base = nested_inputs
    payload = dict(base)
    if missing == "pipeline_semantics_version":
        payload.pop(missing)
        expected_error, message = RuntimeError, "pipeline_semantics_version"
    else:
        config = {**base["config"], "model": dict(base["config"]["model"])}
        config["model"].pop("dt_ms")
        payload["config"] = config
        expected_error, message = ValueError, "lacks explicit 'dt_ms'"
    with pytest.raises(expected_error, match=message):
        canonical_load(path, torch.device("cpu"), checkpoint_payload=payload)


@pytest.mark.parametrize("phase, script, runner", [
    ("C", "analyze_dynamics", "run_analysis"),
    ("D", "simulate_lesion", "run_lesion_experiment"),
    ("E", "analyze_jacobian", "run_jacobian_analysis"),
    ("F", "analyze_integration", "run_integration_analysis"),
    ("G", "simulate_psychophysics", None),
    ("H", "analyze_gating", "run_analysis"),
])
def test_packaged_cli_omits_sidecar_and_uses_checkpoint_path(
        monkeypatch, tmp_path, nested_inputs, phase, script, runner):
    dataset, _, _, checkpoint, _, nested, val_idx, _ = nested_inputs
    module = importlib.import_module(f"scripts.{script}")
    observed = []
    if phase == "G":
        original_loader = module.load_validation_data
        model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
        monkeypatch.setattr(module, "load_checkpoint", lambda *a: model)

        def inspect_validation(*args, **kwargs):
            assert kwargs["nested_prior_artifact"] is None
            assert kwargs["qc_sealed_nested_prior_sha256"] is None
            data = original_loader(*args, **kwargs)
            assert data.meta.val_indices == val_idx.tolist()
            np.testing.assert_allclose(data.X[0, 0, 4:8].cpu().numpy(), nested[int(val_idx[0])])
            observed.append(True)
            return data

        monkeypatch.setattr(module, "load_validation_data", inspect_validation)
        monkeypatch.setattr(module, "find_multisensory_ttc0", lambda X, *a, **k:
                            module.TTC0Selection(torch.zeros(X.shape[0], dtype=torch.bool,
                                                             device=X.device), 0, 0, "not_applicable"))
        monkeypatch.setattr(sys, "argv", [script, "--checkpoint", str(checkpoint),
                                           "--dataset", str(dataset), "--output_dir", str(tmp_path)])
        module.main()
        assert json.loads((tmp_path / "bayesian_reliability.json").read_text())["status"] == "not_applicable"
    else:
        def inspect_entry(**kwargs):
            assert kwargs["nested_prior_artifact"] is None
            assert kwargs["qc_sealed_nested_prior_sha256"] is None
            model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
            loaded = phase_inputs(phase, checkpoint, dataset, None, model)
            loader = loaded[1] if phase == "H" else loaded[0]
            np.testing.assert_allclose(loader.dataset[1][-2][0, 4:8].numpy(), nested[1])
            observed.append(True)

        monkeypatch.setattr(module, runner, inspect_entry)
        output_flag = "--output_dir" if phase == "H" else "--output"
        module.main(["--checkpoint", str(checkpoint), "--dataset", str(dataset),
                     output_flag, str(tmp_path / "out")])
    assert observed == [True]


@pytest.mark.parametrize("phase", list("CDEFGH"))
def test_each_analysis_rejects_source_dataset_replacement(
        phase, nested_inputs, tmp_path, monkeypatch):
    """All production loaders reject A objects paired with B's valid lineage."""
    import io
    from nsmor.pipeline.nested_prior import load_nested_prior_split

    (dataset_path, artifact_path, _, checkpoint_path, source_a,
     priors, val_idx, payload) = nested_inputs
    bytes_a = dataset_path.read_bytes()
    digest_a = hashlib.sha256(bytes_a).hexdigest()
    source_b = dict(source_a,
                    X_seqs=[values + 2.0 for values in source_a["X_seqs"]],
                    Y_seqs=[values + 30.0 for values in source_a["Y_seqs"]],
                    labels=np.ones(len(priors), dtype=np.int64))
    replacement_path = tmp_path / "replacement_source.pt"
    torch.save(source_b, replacement_path)
    bytes_b = replacement_path.read_bytes()
    digest_b = hashlib.sha256(bytes_b).hexdigest()
    assert digest_a != digest_b
    assert not np.array_equal(source_a["Y_seqs"], source_b["Y_seqs"])
    sidecar = torch.load(artifact_path, weights_only=False)
    sidecar["source_fingerprint"] = digest_b
    torch.save(sidecar, artifact_path)
    torch.save(dict(payload, nested_prior_fingerprint=digest_b, dataset_source_sha256=digest_b,
                    nested_prior_artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest()),
               checkpoint_path)
    model = load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))
    assert model.analysis_checkpoint_lineage["nested_prior_fingerprint"] == digest_b
    # Establish that B's sidecar/checkpoint form a valid production input.
    dataset_path.write_bytes(bytes_b)
    load_nested_prior_split(artifact_path, dataset_path, len(priors), source_b["session_ids"])
    phase_inputs(phase, checkpoint_path, dataset_path, None, model)
    dataset_path.write_bytes(bytes_a)
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
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        phase_inputs(phase, checkpoint_path, dataset_path, None, model)
    assert len(reads) == 1
    np.testing.assert_array_equal(reads[0]["Y_seqs"], source_a["Y_seqs"])
    np.testing.assert_array_equal(reads[0]["labels"], source_a["labels"])
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == digest_b


def test_nested_direct_analysis_refuses_missing_loaded_digest(nested_inputs):
    """A naked in-memory object cannot authorize a later path-only source hash."""
    from nsmor.analysis.analysis_priors import load_analysis_priors

    dataset_path, artifact, _, checkpoint, dataset, _, _, _ = nested_inputs
    model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
    with pytest.raises(ValueError, match="loaded source fingerprint"):
        load_analysis_priors(dataset, dataset_path, artifact, model)


@pytest.mark.parametrize("phase", list("CDEFGH"))
def test_nonnested_analysis_rejects_valid_same_path_source_swap(phase, nested_inputs, tmp_path):
    """The C-H production loaders cannot use B targets/priors with A's weights."""
    (dataset_path, artifact, _, checkpoint_path, source_a,
     _, _, payload) = nested_inputs
    source_a = dict(source_a, mcmc_prior_provenance="oof_5fold_recording_prefix_grouped_cv",
                    animal_identity_status="unverified")
    torch.save(source_a, dataset_path)
    digest_a = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    nonnested = dict(payload, is_nested_cv=False, nested_prior_artifact="",
                     nested_prior_artifact_sha256="", nested_prior_fingerprint="",
                     nested_split_seed=-1, nested_val_split=-1.0,
                     mcmc_prior_provenance=source_a["mcmc_prior_provenance"],
                     animal_identity_status="unverified",
                     validation_scope="diagnostic_global_oof",
                     dataset_source_sha256=digest_a)
    torch.save(nonnested, checkpoint_path)
    model = load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))
    result = phase_inputs(phase, checkpoint_path, dataset_path, None, model)
    assert (result[0] if phase == "H" else model).analysis_dataset_binding == "sha256_verified"
    source_b = dict(source_a, Y_seqs=[y + 30.0 for y in source_a["Y_seqs"]],
                    mcmc_priors=np.roll(source_a["mcmc_priors"], 1, axis=1))
    replacement = tmp_path / "valid_b.pt"
    torch.save(source_b, replacement)
    dataset_path.write_bytes(replacement.read_bytes())
    assert compute_source_fingerprint(dataset_path) != digest_a
    assert not np.array_equal(source_b["mcmc_priors"], source_a["mcmc_priors"])
    with pytest.raises(ValueError, match="dataset_source_sha256 mismatch"):
        phase_inputs(phase, checkpoint_path, dataset_path, None, model)


def test_genuine_legacy_nonnested_analysis_reports_unbound_status(
        nested_inputs, caplog, monkeypatch, tmp_path):
    from scripts import simulate_psychophysics as phase_g

    dataset_path, _, _, checkpoint_path, _, _, _, payload = nested_inputs
    legacy = dict(payload, is_nested_cv=False, nested_prior_artifact="",
                  nested_prior_artifact_sha256="", nested_prior_fingerprint="",
                  nested_split_seed=-1, nested_val_split=-1.0,
                  mcmc_prior_provenance="global_oof_animal_grouped_cv",
                  animal_identity_status="historical_unknown",
                  validation_scope="diagnostic_global_oof")
    legacy.pop("dataset_source_sha256")
    assert "dataset_source_sha256" not in legacy
    torch.save(legacy, checkpoint_path)
    model = load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))
    from scripts.simulate_psychophysics import load_validation_data

    captured_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="trusted_historical_checkpoint_sha256"):
        load_validation_data(torch.device("cpu"), dataset_path=str(dataset_path),
                             max_seq_len=None, checkpoint_model=model)
    with pytest.raises(ValueError, match="trusted_historical_checkpoint_sha256"):
        load_validation_data(torch.device("cpu"), dataset_path=str(dataset_path),
                             max_seq_len=None, checkpoint_model=model,
                             trusted_historical_checkpoint_sha256="0" * 64)
    load_validation_data(torch.device("cpu"), dataset_path=str(dataset_path),
                         max_seq_len=None, checkpoint_model=model,
                         trusted_historical_checkpoint_sha256=captured_sha)
    assert model.analysis_dataset_binding == "legacy_unbound"
    assert "legacy unbound dataset content" in caplog.text

    retagged = tmp_path / "retagged_historical.pth"
    torch.save(dict(legacy, animal_identity_status="unverified"), retagged)
    with pytest.raises(ValueError, match="animal_identity_status"):
        load_model_from_checkpoint(retagged, torch.device("cpu"))

    actual_loader = phase_g.load_validation_data
    forwarded = []

    def inspect_validation(*args, **kwargs):
        forwarded.append(kwargs["trusted_historical_checkpoint_sha256"])
        return actual_loader(*args, **kwargs)

    monkeypatch.setattr(phase_g, "load_validation_data", inspect_validation)
    monkeypatch.setattr(phase_g, "load_checkpoint", lambda *_args: model)
    monkeypatch.setattr(phase_g, "find_multisensory_ttc0", lambda X, *a, **k:
                        phase_g.TTC0Selection(torch.zeros(X.shape[0], dtype=torch.bool,
                                                           device=X.device), 0, 0, "not_applicable"))
    monkeypatch.setattr(sys, "argv", ["simulate_psychophysics", "--checkpoint", str(checkpoint_path),
                                       "--dataset", str(dataset_path), "--output_dir", str(tmp_path),
                                       "--trusted_historical_checkpoint_sha256", captured_sha])
    phase_g.main()
    assert forwarded == [captured_sha]
    assert json.loads((tmp_path / "bayesian_reliability.json").read_text())["status"] == "not_applicable"

@pytest.mark.parametrize("consumer", ["dataset", "evaluate", "train_dataset", "nested_split", "train_nested"])
def test_generated_artifact_consumers_reject_reducer_before_side_effect(
        consumer, nested_inputs, tmp_path):
    """Stage 03/04 consumers must refuse replaced pickle before any reducer runs."""
    from nsmor.config_parser import ExperimentConfig
    from scripts.evaluate_nested_prior import generate_nested_priors
    from scripts.train import build_dataloaders

    dataset_path, artifact_path, _, _, dataset, priors, _, _ = nested_inputs
    marker = tmp_path / "pickle_executed.txt"

    class Reducer:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",)

    is_prior = consumer in ("nested_split", "train_nested")
    target = artifact_path if is_prior else dataset_path
    payload = load_dataset_with_fingerprint(target)[0]
    torch.save(dict(payload, malicious_metadata=Reducer()), target)
    try:
        with pytest.raises(ValueError, match="Unexpected serialized global"):
            if consumer == "dataset":
                load_dataset_with_fingerprint(dataset_path)
            elif consumer == "evaluate":
                generate_nested_priors(dataset=None, source_dataset_path=dataset_path, verbose=False)
            elif consumer == "nested_split":
                load_nested_prior_split(artifact_path, dataset_path, len(priors), dataset["session_ids"])
            else:
                build_dataloaders(ExperimentConfig(), dataset_path=str(dataset_path),
                                 nested_prior_artifact=str(artifact_path) if is_prior else None)
    finally:
        assert not marker.exists(), "Serialized reducer ran before artifact rejection"


@pytest.mark.parametrize("dtype", [
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "uint64", "float16", "float32", "float64", "complex64", "complex128",
    "str", "bytes", "object",
])
def test_restricted_artifact_preserves_known_configs_numpy_and_mcmc(tmp_path, dtype):
    from nsmor.config import TimeWindowConfig
    from nsmor.mcmc_module import MCMCPriorGenerator

    values = np.array(["prefix_a", "prefix_b"] if dtype in ("str", "bytes", "object")
                      else [0, 1], dtype=dtype)
    model = MCMCPriorGenerator()
    window = TimeWindowConfig(frame_interval_ms=4.0)
    artifact = {"feature_config": FeatureConfig(), "time_window_config": window,
                "values": values, "mcmc_model": model, "tensor": torch.tensor([3.0, 7.0])}
    artifact_path = tmp_path / "generated.pt"
    torch.save(artifact, artifact_path)
    captured = artifact_path.read_bytes()
    previous = set(torch.serialization.get_safe_globals())
    locations = []

    def map_storage(storage, location):
        locations.append(location)
        return storage

    loaded = load_artifact_bytes(captured, map_storage)
    assert set(torch.serialization.get_safe_globals()) == previous
    assert locations and set(locations) == {"cpu"}
    assert loaded["feature_config"] == artifact["feature_config"]
    assert loaded["time_window_config"] == window
    assert loaded["values"].dtype == values.dtype
    np.testing.assert_array_equal(loaded["values"], values)
    assert isinstance(loaded["mcmc_model"], MCMCPriorGenerator)
    assert isinstance(loaded["mcmc_model"].classifier, torch.nn.Linear)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded["mcmc_model"].state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(loaded["tensor"], torch.tensor([3.0, 7.0]), rtol=0, atol=0)
    # assert_close can lazily register DTensor globals in newer PyTorch versions.
    previous = set(torch.serialization.get_safe_globals())
    via_path, digest = load_dataset_with_fingerprint(artifact_path, map_location="cpu")
    assert digest == hashlib.sha256(captured).hexdigest()
    np.testing.assert_array_equal(via_path["values"], values)
    assert set(torch.serialization.get_safe_globals()) == previous


def test_restricted_artifact_refuses_ambient_pickle_global(tmp_path):
    marker = tmp_path / "ambient_pickle_executed.txt"

    class Reducer:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",)

    stream = io.BytesIO()
    torch.save({"metadata": Reducer()}, stream)
    with torch.serialization.safe_globals([eval, (eval, "__builtin__.eval")]):
        previous = set(torch.serialization.get_safe_globals())
        try:
            with pytest.raises(ValueError, match="Unexpected serialized global"):
                load_artifact_bytes(stream.getvalue(), map_location="cpu")
        finally:
            assert not marker.exists()
            assert set(torch.serialization.get_safe_globals()) == previous


@pytest.mark.parametrize("consumer", ["dataset", "nested_split"])
def test_restricted_artifact_keeps_captured_bytes_when_path_changes_after_scan(
        consumer, nested_inputs, tmp_path, monkeypatch):
    dataset_path, artifact_path, _, _, dataset, priors, _, _ = nested_inputs
    target = dataset_path if consumer == "dataset" else artifact_path
    captured = target.read_bytes()
    original, digest = load_dataset_with_fingerprint(target)
    marker = tmp_path / "swapped_pickle_executed.txt"

    class Reducer:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",)

    stream = io.BytesIO()
    torch.save(dict(original, malicious_metadata=Reducer()), stream)
    replacement = stream.getvalue()
    assert hashlib.sha256(replacement).hexdigest() != digest
    scan = torch.serialization.get_unsafe_globals_in_checkpoint
    scans = []

    def swap_after_scan(source):
        result = scan(source)
        if source.getvalue() == captured:
            scans.append(True)
            target.write_bytes(replacement)
        return result

    monkeypatch.setattr(torch.serialization, "get_unsafe_globals_in_checkpoint", swap_after_scan)
    if consumer == "dataset":
        loaded, loaded_digest = load_dataset_with_fingerprint(target)
        assert loaded_digest == digest
        np.testing.assert_array_equal(loaded["Y_seqs"], original["Y_seqs"])
    else:
        _, _, loaded, info = load_nested_prior_split(
            target, dataset_path, len(priors), dataset["session_ids"])
        assert info["nested_prior_artifact_sha256"] == digest
        np.testing.assert_array_equal(loaded, original["nested_priors"])
    assert scans == [True]
    assert target.read_bytes() == replacement
    with pytest.raises(ValueError, match="Unexpected serialized global"):
        load_dataset_with_fingerprint(target)
    assert not marker.exists()


def test_analysis_identity_status_cannot_be_upgraded(nested_inputs, tmp_path):
    from nsmor.analysis.analysis_priors import load_analysis_priors

    dataset_path, artifact, _, checkpoint, dataset, _, _, payload = nested_inputs
    model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
    load_analysis_priors(
        dataset, dataset_path, model=model,
        loaded_source_fingerprint=payload["nested_prior_fingerprint"],
    )
    assert model.analysis_animal_identity_status == "unverified"

    forged = tmp_path / "forged_identity.pth"
    torch.save(dict(payload, animal_identity_status="verified"), forged)
    with pytest.raises(ValueError, match="animal_identity_status"):
        load_model_from_checkpoint(forged, torch.device("cpu"))


def test_psychophysics_direct_cli_nested_best_checkpoint_is_descriptive(nested_inputs, tmp_path):
    """Real CLI, loader, model, and JSON writers; synthetic checkpoint-bound outer rows."""
    dataset, artifact_path, _, checkpoint, source, _, val_idx, payload = nested_inputs
    source["stimulus_conditions"] = ["multisensory"] * len(source["X_seqs"])
    for row in source["X_seqs"]:
        row[:, 0] = 20.0
        row[:, 1] = 1.0

    for ttc, name in ((0.0, "present"), (-225.0, "absent")):
        source["target_ttc_ms"] = [ttc] * len(source["X_seqs"])
        torch.save(source, dataset)
        fingerprint = compute_source_fingerprint(dataset)
        artifact = load_artifact_bytes(artifact_path.read_bytes())
        artifact["source_fingerprint"] = fingerprint
        torch.save(artifact, artifact_path)
        payload.update(
            nested_prior_fingerprint=fingerprint,
            dataset_source_sha256=fingerprint,
            nested_prior_artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        )
        torch.save(payload, checkpoint)

        out = tmp_path / name
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "scripts/simulate_psychophysics.py"),
             "--checkpoint", str(checkpoint), "--dataset", str(dataset),
             "--raw_dir", str(tmp_path), "--output_dir", str(out),
             "--pre_anchor_frames", "2", "--max_seq_len", "6", "--noise_levels", "0", "5"],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        status = json.loads((out / "bayesian_reliability.json").read_text(encoding="utf-8"))
        outputs = [status]
        if ttc == 0.0:
            assert status["status"] in ("ok", "unavailable_latency")
            assert status["n_ttc0"] == len(val_idx)
            outputs.append(json.loads((out / "psychophysics_summary.json").read_text(encoding="utf-8")))
        else:
            assert status["status"] == "not_applicable" and status["n_ttc0"] == 0
            assert not (out / "psychophysics_summary.json").exists()
        for data in outputs:
            assert data["validation_scope"] == "nested_outer_validation"
            assert data["sample_scope"] == "outer_validation"
            assert data["holdout_eligible"] is False
            assert "Checkpoint-selected" in data["validation_note"]
            assert "descriptive" in data["validation_note"]
            assert "ineligible as independent holdout" in data["validation_note"]
