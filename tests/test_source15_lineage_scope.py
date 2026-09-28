"""SOURCE15 lineage and validation-scope counterexamples (synthetic only)."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nsmor.analysis.analysis_priors import load_analysis_priors
from nsmor.config import PIPELINE_SEMANTICS_VERSION
from nsmor.config_parser import ExperimentConfig
from nsmor.model_utils import validate_dataset_provenance
from nsmor.pipeline.grouping import animal_keys_of
from scripts import simulate_psychophysics as psych, train as trainer


def _dataset():
    sessions = [f"recording_{i // 2}_session_{i % 2 + 1}" for i in range(8)]
    return {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
        "X_seqs": [np.zeros((4, 8), dtype=np.float32) for _ in sessions],
        "Y_seqs": [np.full(4, i, dtype=np.float32) for i in range(8)],
        "lengths": np.full(8, 4),
        "labels": np.zeros(8, dtype=np.int64),
        "mcmc_priors": np.full((8, 4), .25, dtype=np.float32),
        "anchor_frames": [1] * 8,
        "stimulus_conditions": ["wind_only"] * 8,
        "is_pure_wind": np.ones(8, dtype=bool),
        "trial_specs": [{"session_id": s} for s in sessions],
        "session_ids": sessions,
    }


def test_invalid_specs_only_cannot_enter_eager_stats_or_psychophysics(tmp_path, monkeypatch):
    ds = _dataset()
    ds.pop("session_ids")
    ds["trial_specs"][0]["session_id"] = ""
    source = tmp_path / "synthetic.pt"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(trainer, "load_dataset_with_fingerprint", lambda *_: (ds, "a" * 64))
    monkeypatch.setattr(psych, "load_dataset_with_fingerprint", lambda *a, **k: (ds, "a" * 64))
    cfg = ExperimentConfig()
    cfg.training.normalize_targets = True
    with pytest.raises(RuntimeError, match="session_ids"):
        validate_dataset_provenance(ds, source)
    with pytest.raises(RuntimeError, match="session_ids"):
        trainer.build_dataloaders(cfg, dataset_path=str(source))
    with pytest.raises(RuntimeError, match="session_ids"):
        trainer.compute_target_stats(str(source), cfg)
    with pytest.raises(RuntimeError, match="session_ids"):
        psych.load_validation_data(torch.device("cpu"), dataset_path=str(source))


@pytest.mark.parametrize("specs_only", [False, True])
def test_eager_loader_and_statistics_keep_prefixes_disjoint(tmp_path, monkeypatch, specs_only):
    ds = _dataset()
    sessions = list(ds["session_ids"])
    if specs_only:
        ds.pop("session_ids")
    source = tmp_path / "synthetic.pt"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(trainer, "load_dataset_with_fingerprint", lambda *_: (ds, "a" * 64))
    monkeypatch.setattr(trainer, "create_dataloaders_from_config", lambda _cfg, train_dataset, val_dataset: (
        torch.utils.data.DataLoader(train_dataset, batch_size=8),
        torch.utils.data.DataLoader(val_dataset, batch_size=8), None,
    ))
    cfg = ExperimentConfig()
    cfg.training.normalize_targets = True
    cfg.loss.lambda_routing_aux = 0.0
    train_loader, val_loader = trainer.build_dataloaders(cfg, dataset_path=str(source), val_split=.5)
    train_idx = train_loader.dataset.source_indices
    val_idx = val_loader.dataset.source_indices
    keys = animal_keys_of(sessions)
    assert set(keys[train_idx]).isdisjoint(keys[val_idx])
    _, _, stat_idx = trainer.compute_target_stats(str(source), cfg, val_split=.5)
    np.testing.assert_array_equal(stat_idx, train_idx)
    if specs_only:
        monkeypatch.setattr(psych, "load_dataset_with_fingerprint", lambda *a, **k: (ds, "a" * 64))
        validation = psych.load_validation_data(
            torch.device("cpu"), dataset_path=str(source), val_split=.5,
            random_seed=cfg.training.random_seed, max_seq_len=4, pre_anchor_frames=1,
        )
        assert validation.meta.val_indices == list(val_idx)
        assert validation.meta.session_ids == [sessions[i] for i in val_idx]


def _model(ds, digest, *, nested=False, scope="diagnostic_global_oof", checkpoint_sha=None):
    lineage = {
        "is_nested_cv": nested,
        "mcmc_prior_provenance": ds["mcmc_prior_provenance"],
        "animal_identity_status": ds["animal_identity_status"],
    }
    if scope is not None:
        lineage["validation_scope"] = scope
    if digest is not None:
        lineage["dataset_source_sha256"] = digest
        lineage["dataset_source_binding"] = "sha256_bound"
    return SimpleNamespace(analysis_checkpoint_lineage=lineage,
                           analysis_checkpoint_sha256=checkpoint_sha)


@pytest.mark.parametrize("scope", [None, "nested_outer_validation", "bogus"])
def test_nonnested_modern_scope_missing_or_contradictory_is_rejected(tmp_path, scope):
    ds = _dataset()
    digest = hashlib.sha256(b"modern dataset").hexdigest()
    model = _model(ds, digest, scope=scope)
    with pytest.raises(ValueError, match="validation_scope"):
        load_analysis_priors(ds, tmp_path / "synthetic.pt", model=model,
                             loaded_source_fingerprint=digest)


def test_diagnostic_scope_is_exposed_to_all_analysis_callers(tmp_path):
    ds = _dataset()
    digest = hashlib.sha256(b"modern dataset").hexdigest()
    model = _model(ds, digest)
    priors, indices = load_analysis_priors(ds, tmp_path / "synthetic.pt", model=model,
                                          loaded_source_fingerprint=digest)
    assert priors is ds["mcmc_priors"] and indices is None
    assert model.analysis_validation_scope == "diagnostic_global_oof"
    assert model.analysis_dataset_binding == "sha256_verified"


def test_historical_retag_cannot_downgrade_without_external_checkpoint_sha(tmp_path):
    ds = _dataset()
    dataset_path = tmp_path / "synthetic.pt"
    torch.save(ds, dataset_path)
    original_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    original = _model(ds, original_digest).analysis_checkpoint_lineage
    original_path = tmp_path / "original.pth"
    torch.save(original, original_path)
    independent_pin = hashlib.sha256(original_path.read_bytes()).hexdigest()

    # Both files are rewritten as historical and the checkpoint loses its source binding.
    ds.update(mcmc_prior_provenance="oof_2fold_animal_grouped_cv",
              animal_identity_status="historical_unknown")
    torch.save(ds, dataset_path)
    retagged_source_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    changed = dict(original, mcmc_prior_provenance=ds["mcmc_prior_provenance"],
                   animal_identity_status="historical_unknown")
    for field in ("dataset_source_sha256", "dataset_source_binding", "validation_scope"):
        changed.pop(field)
    ckpt_path = tmp_path / "retagged.pth"
    torch.save(changed, ckpt_path)
    changed_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert independent_pin != changed_sha
    model = SimpleNamespace(analysis_checkpoint_lineage=changed,
                            analysis_checkpoint_sha256=changed_sha)
    for trusted in (None, independent_pin):
        with pytest.raises(ValueError, match="trusted_historical_checkpoint_sha256"):
            load_analysis_priors(ds, dataset_path, model=model,
                                 loaded_source_fingerprint=retagged_source_digest,
                                 trusted_historical_checkpoint_sha256=trusted)
        with pytest.raises(ValueError, match="trusted_historical_checkpoint_sha256"):
            trainer._check_checkpoint_prior_lineage(
                changed, changed, ckpt_path, trusted_historical_checkpoint_sha256=trusted,
                checkpoint_sha256=changed_sha)
        with pytest.raises(ValueError, match="trusted_historical_checkpoint_sha256"):
            trainer._validate_best_checkpoint(
                dict(changed, val_loss=.1), ckpt_path, changed,
                trusted_historical_checkpoint_sha256=trusted,
                checkpoint_sha256=changed_sha)


def test_independently_pinned_historical_checkpoint_remains_explicit(tmp_path):
    ds = _dataset()
    ds.update(mcmc_prior_provenance="oof_2fold_animal_grouped_cv",
              animal_identity_status="historical_unknown")
    source_sha = hashlib.sha256(b"independently pinned historical dataset").hexdigest()
    checkpoint_sha = "e6d2c5da8adedefd321cd5dfbb461b6ba9a3343d802d1b3c79d97903d40d4e37"
    assert hashlib.sha256(b"independently pinned historical checkpoint").hexdigest() == checkpoint_sha
    model = _model(ds, None, scope=None, checkpoint_sha=checkpoint_sha)
    priors, indices = load_analysis_priors(
        ds, tmp_path / "historical.pt", model=model, loaded_source_fingerprint=source_sha,
        trusted_historical_checkpoint_sha256=checkpoint_sha)
    assert priors is ds["mcmc_priors"] and indices is None
    assert model.analysis_dataset_binding == "legacy_unbound"
    assert model.analysis_validation_scope == "historical_unscoped"
    ckpt = dict(model.analysis_checkpoint_lineage, val_loss=.1)
    active_lineage = dict(ckpt, dataset_source_binding="legacy_resume_unbound")
    trainer._check_checkpoint_prior_lineage(
        ckpt, ckpt, tmp_path / "historical.pth",
        trusted_historical_checkpoint_sha256=checkpoint_sha,
        checkpoint_sha256=checkpoint_sha)
    assert trainer._validate_best_checkpoint(
        ckpt, tmp_path / "historical.pth", active_lineage,
        trusted_historical_checkpoint_sha256=checkpoint_sha,
        checkpoint_sha256=checkpoint_sha) == .1


def test_nested_scope_checked_and_exposed(tmp_path, monkeypatch):
    from nsmor.analysis import analysis_priors as mod
    ds = _dataset()
    digest = hashlib.sha256(b"nested dataset").hexdigest()
    artifact = tmp_path / "nested.pt"
    artifact.write_bytes(b"synthetic nested artifact")
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    model = _model(ds, digest, nested=True, scope="nested_outer_validation")
    model.analysis_checkpoint_lineage["mcmc_prior_provenance"] = "nested_outer_seed42_inner_2fold_recording_prefix_grouped"
    model.analysis_checkpoint_lineage.update(
        dataset_path=str(tmp_path / "synthetic.pt"), nested_prior_artifact=str(artifact),
        nested_prior_artifact_sha256=artifact_sha, nested_prior_fingerprint=digest,
        nested_split_seed=42, nested_val_split=.5)
    info = dict(nested_prior_artifact=str(artifact), nested_prior_artifact_sha256=artifact_sha,
                nested_prior_fingerprint=digest, split_seed=42, val_split=.5,
                mcmc_prior_provenance=model.analysis_checkpoint_lineage["mcmc_prior_provenance"],
                animal_identity_status="unverified", n_train=4, n_val=4)
    monkeypatch.setattr(mod, "load_nested_prior_split", lambda *a, **kw: (
        np.arange(4), np.arange(4, 8), ds["mcmc_priors"], info))
    priors, val_idx = load_analysis_priors(ds, tmp_path / "synthetic.pt", artifact, model,
                                          loaded_source_fingerprint=digest)
    assert priors.shape == (8, 4)
    assert val_idx.tolist() == [4, 5, 6, 7]
    assert model.analysis_validation_scope == "nested_outer_validation"
    del model.analysis_checkpoint_lineage["validation_scope"]
    with pytest.raises(ValueError, match="validation_scope"):
        load_analysis_priors(ds, tmp_path / "synthetic.pt", artifact, model,
                             loaded_source_fingerprint=digest)


def test_checkpoint_loader_captures_bytes_and_validation_scope(tmp_path, monkeypatch):
    from nsmor.analysis import prediction_units

    ds = _dataset()
    payload = _model(ds, "a" * 64).analysis_checkpoint_lineage
    payload["config"] = {"training": {"normalize_targets": False}}
    checkpoint_path = tmp_path / "modern.pth"
    torch.save(payload, checkpoint_path)
    backend = SimpleNamespace(register_forward_hook=lambda hook: None)
    model = SimpleNamespace(backend=backend)
    monkeypatch.setattr(prediction_units, "_canonical_load_model", lambda *args: model)
    loaded = prediction_units.load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))
    assert loaded is model
    assert loaded.analysis_checkpoint_sha256 == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert loaded.analysis_checkpoint_lineage["validation_scope"] == "diagnostic_global_oof"


@pytest.mark.parametrize("scope", [None, "nested_outer_validation", "bogus"])
def test_modern_resume_and_best_reject_missing_or_conflicting_scope(tmp_path, scope):
    ds = _dataset()
    expected = _model(ds, "a" * 64).analysis_checkpoint_lineage
    candidate = dict(expected, val_loss=.1)
    if scope is None:
        candidate.pop("validation_scope")
    else:
        candidate["validation_scope"] = scope
    with pytest.raises(ValueError, match="validation_scope"):
        trainer._check_checkpoint_prior_lineage(candidate, expected, tmp_path / "resume.pth")
    with pytest.raises(ValueError, match="validation_scope"):
        trainer._validate_best_checkpoint(candidate, tmp_path / "best.pth", expected)
    good = dict(expected, val_loss=.1)
    assert trainer._validate_best_checkpoint(good, tmp_path / "best.pth", expected) == .1


def test_historical_resume_requires_pin_and_accepts_saved_checkpoint_bytes(tmp_path, monkeypatch):
    from tests.test_train_checkpoint import _make_config, _make_synthetic_dataset

    source = _make_synthetic_dataset(tmp_path)
    from nsmor.pipeline.nested_prior import load_artifact_bytes
    historical = load_artifact_bytes(source.read_bytes(), map_location="cpu")
    historical["mcmc_prior_provenance"] = "oof_5fold_animal_grouped_cv"
    historical["animal_identity_status"] = "historical_unknown"
    torch.save(historical, source)
    initial = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    monkeypatch.setattr(trainer, "train_one_epoch", lambda *a, **k: (1.0, {}))
    monkeypatch.setattr(trainer, "validate", lambda *a, **k: .1)
    trainer.train(initial, dataset_path=str(source))
    best = Path(initial.checkpoint.output_dir) / "best_model.pth"
    checkpoint_sha = hashlib.sha256(best.read_bytes()).hexdigest()
    resumed = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "resumed")
    resumed.checkpoint.resume_from = str(best)
    with pytest.raises(ValueError, match="trusted_historical_checkpoint_sha256"):
        trainer.train(resumed, dataset_path=str(source))
    result = trainer.train(resumed, dataset_path=str(source),
                           trusted_historical_checkpoint_sha256=checkpoint_sha)
    assert result["validation_scope"] == "diagnostic_global_oof"
    assert result["best_val_loss"] == .1
