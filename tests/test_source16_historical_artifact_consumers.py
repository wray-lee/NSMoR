"""Synthetic bytes through real training and analysis consumers; no corpus or inference."""

import hashlib
import importlib
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from nsmor.analysis.analysis_priors import load_analysis_priors
from nsmor.pipeline.nested_prior import load_artifact_bytes
from tests.test_nested_train_integration import (
    _make_canonical_dataset, _make_config, _make_nested_artifact, _save_artifact,
)


class _Loaded(Exception):
    """Stop after the real shared prior loader accepts the captured bytes."""


@pytest.fixture
def sources(tmp_path):
    dataset_path = _make_canonical_dataset(tmp_path)
    dataset = load_artifact_bytes(dataset_path.read_bytes())
    dataset.update(mcmc_prior_provenance="oof_5fold_animal_grouped_cv",
                   animal_identity_status="historical_unknown")
    torch.save(dataset, dataset_path)
    artifact = _make_nested_artifact(dataset_path, dataset)
    artifact.update(mcmc_prior_provenance="nested_outer_seed3_inner_5fold_animal_grouped",
                    animal_identity_status="historical_unknown")
    sidecar = _save_artifact(tmp_path, artifact)
    artifact_pin = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    lineage = {
        "is_nested_cv": True, "nested_prior_artifact": str(sidecar),
        "nested_prior_artifact_sha256": artifact_pin,
        "nested_prior_fingerprint": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dataset_path": str(dataset_path), "nested_split_seed": artifact["split_seed"],
        "nested_val_split": artifact["val_split"],
        "mcmc_prior_provenance": artifact["mcmc_prior_provenance"],
        "animal_identity_status": artifact["animal_identity_status"],
        "validation_scope": "nested_outer_validation",
    }
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(lineage, checkpoint)
    checkpoint_pin = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_pin != artifact_pin
    return dataset_path, sidecar, checkpoint, artifact_pin, checkpoint_pin


@pytest.mark.parametrize("script", [
    "analyze_dynamics", "analyze_jacobian", "analyze_integration",
    "simulate_lesion", "analyze_gating", "simulate_psychophysics",
])
def test_analysis_cli_separate_pins(sources, tmp_path, monkeypatch, script):
    dataset, sidecar, checkpoint, artifact_pin, checkpoint_pin = sources
    module = importlib.import_module("scripts." + script)

    def model_loader(path, _device):
        payload = Path(path).read_bytes()
        return SimpleNamespace(analysis_checkpoint_lineage=load_artifact_bytes(payload),
                               analysis_checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
                               dt_ms=1.0)

    loader_name = ("_shared_load_model" if script == "analyze_gating" else
                   "load_checkpoint" if script == "simulate_psychophysics" else
                   "load_model_from_checkpoint")
    monkeypatch.setattr(module, loader_name, model_loader)
    accepted = []

    def stop_after_real_loader(*args, **kwargs):
        priors, val = load_analysis_priors(*args, **kwargs)
        accepted.append((priors.shape, len(val)))
        raise _Loaded

    monkeypatch.setattr(module, "load_analysis_priors", stop_after_real_loader)
    argv = ["--checkpoint", str(checkpoint), "--dataset", str(dataset),
            "--nested_prior_artifact", str(sidecar)]
    if script == "simulate_psychophysics":
        argv += ["--output_dir", str(tmp_path / "out")]

    def run(extra):
        if script == "simulate_psychophysics":
            monkeypatch.setattr(sys, "argv", [script] + argv + extra)
            module.main()
        else:
            module.main(argv + extra)

    checkpoint_flag = ["--trusted_historical_checkpoint_sha256", checkpoint_pin]
    artifact_flag = ["--trusted_historical_artifact_sha256", artifact_pin]
    with pytest.raises(ValueError, match="trusted_historical_checkpoint_sha256"):
        run(artifact_flag)
    with pytest.raises(ValueError, match="trusted_historical_artifact_sha256"):
        run(checkpoint_flag)
    with pytest.raises(ValueError, match="trusted_historical_artifact_sha256"):
        run(checkpoint_flag + ["--trusted_historical_artifact_sha256", checkpoint_pin])
    with pytest.raises(ValueError, match="trusted_historical_checkpoint_sha256"):
        run(["--trusted_historical_checkpoint_sha256", artifact_pin] + artifact_flag)
    assert accepted == []
    with pytest.raises(_Loaded):
        run(checkpoint_flag + artifact_flag)
    assert accepted == [((24, 4), len(load_artifact_bytes(sidecar.read_bytes())["val_indices"]))]

    # A modern sidecar never accepts the historical compatibility pin.
    data = load_artifact_bytes(dataset.read_bytes())
    data.update(mcmc_prior_provenance="oof_5fold_recording_prefix_grouped_cv",
                animal_identity_status="unverified")
    torch.save(data, dataset)
    modern = load_artifact_bytes(sidecar.read_bytes())
    modern.update(mcmc_prior_provenance="nested_outer_seed3_inner_5fold_recording_prefix_grouped",
                  animal_identity_status="unverified",
                  source_fingerprint=hashlib.sha256(dataset.read_bytes()).hexdigest())
    torch.save(modern, sidecar)
    lineage = load_artifact_bytes(checkpoint.read_bytes())
    lineage.update(mcmc_prior_provenance=modern["mcmc_prior_provenance"],
                   animal_identity_status="unverified",
                   nested_prior_artifact_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                   nested_prior_fingerprint=modern["source_fingerprint"],
                   dataset_source_sha256=modern["source_fingerprint"])
    torch.save(lineage, checkpoint)
    with pytest.raises(ValueError, match="only valid for historical"):
        run(["--trusted_historical_artifact_sha256",
             hashlib.sha256(sidecar.read_bytes()).hexdigest()])
    assert len(accepted) == 1


def test_training_loaders_and_cli_pin(sources, tmp_path, monkeypatch):
    from scripts import train as training

    dataset, sidecar, _, artifact_pin, checkpoint_pin = sources
    config = _make_config(tmp_path, normalize_targets=True)
    config.training.target_clip_cm_s = 100.0
    for consumer in (
        lambda pin: training.build_dataloaders(config, dataset_path=str(dataset),
            nested_prior_artifact=str(sidecar), trusted_historical_artifact_sha256=pin),
        lambda pin: training.compute_target_stats(str(dataset), config,
            nested_prior_artifact=str(sidecar), trusted_historical_artifact_sha256=pin),
    ):
        with pytest.raises(ValueError, match="trusted_historical_artifact_sha256"):
            consumer(None)
        with pytest.raises(ValueError, match="trusted_historical_artifact_sha256"):
            consumer(checkpoint_pin)
        assert consumer(artifact_pin) is not None

    def fake_train(*_args, **kwargs):
        assert kwargs["trusted_historical_artifact_sha256"] == artifact_pin
        assert kwargs["nested_prior_artifact"] == str(sidecar)
        raise _Loaded

    monkeypatch.setattr(training, "train", fake_train)
    with pytest.raises(_Loaded):
        training.main(["--dataset", str(dataset), "--nested_prior_artifact", str(sidecar),
                       "--trusted_historical_artifact_sha256", artifact_pin])



def test_train_accepts_historical_artifact_with_external_pin(sources, tmp_path):
    from scripts.train import train

    dataset, sidecar, _, artifact_pin, _ = sources
    config = _make_config(tmp_path)
    result = train(config, dataset_path=str(dataset), nested_prior_artifact=str(sidecar),
                   trusted_historical_artifact_sha256=artifact_pin)
    assert result["best_val_loss"] >= 0
    checkpoint = load_artifact_bytes((Path(config.checkpoint.output_dir) / "best_model.pth").read_bytes())
    assert checkpoint["nested_prior_artifact_sha256"] == artifact_pin
    assert checkpoint["animal_identity_status"] == "historical_unknown"


def test_modern_consumers_reject_historical_artifact_pin(tmp_path, monkeypatch):
    from scripts import train as training

    dataset = _make_canonical_dataset(tmp_path)
    data = load_artifact_bytes(dataset.read_bytes())
    artifact = _make_nested_artifact(dataset, data)
    sidecar = _save_artifact(tmp_path, artifact)
    pin = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    config = _make_config(tmp_path, normalize_targets=True)
    with pytest.raises(ValueError, match="only valid for historical"):
        training.build_dataloaders(config, dataset_path=str(dataset),
            nested_prior_artifact=str(sidecar), trusted_historical_artifact_sha256=pin)
    with pytest.raises(ValueError, match="only valid for historical"):
        training.compute_target_stats(str(dataset), config,
            nested_prior_artifact=str(sidecar), trusted_historical_artifact_sha256=pin)
    model = SimpleNamespace(analysis_checkpoint_lineage={
        "is_nested_cv": True, "nested_prior_artifact": str(sidecar),
        "nested_prior_artifact_sha256": pin,
        "nested_prior_fingerprint": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "dataset_path": str(dataset), "nested_split_seed": artifact["split_seed"],
        "nested_val_split": artifact["val_split"],
        "mcmc_prior_provenance": artifact["mcmc_prior_provenance"],
        "animal_identity_status": "unverified", "validation_scope": "nested_outer_validation",
        "dataset_source_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    })
    with pytest.raises(ValueError, match="only valid for historical"):
        load_analysis_priors(data, dataset, model=model,
            loaded_source_fingerprint=hashlib.sha256(dataset.read_bytes()).hexdigest(),
            trusted_historical_artifact_sha256=pin)
