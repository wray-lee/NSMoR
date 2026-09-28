"""CLI authorization for synthetic historical checkpoint bytes (no corpus reads)."""

import hashlib
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION
from nsmor.pipeline.nested_prior import load_artifact_bytes


class _Loaded(Exception):
    """Stop after the real priors and population loaders, before inference."""


@pytest.mark.parametrize("script", [
    "analyze_dynamics", "analyze_jacobian", "analyze_integration",
    "simulate_lesion", "analyze_gating",
])
def test_historical_pin_cli_and_modern_retag(tmp_path, monkeypatch, script):
    module = importlib.import_module("scripts." + script)
    dataset_path = tmp_path / "synthetic.pt"
    checkpoint_path = tmp_path / "synthetic.pth"
    data = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_2fold_animal_grouped_cv",
        "animal_identity_status": "historical_unknown",
        "X_seqs": [np.zeros((6, 8), dtype=np.float32) for _ in range(2)],
        "Y_seqs": [np.zeros(6, dtype=np.float32) for _ in range(2)],
        "lengths": np.full(2, 6),
        "labels": np.array([0, 1]),
        "mcmc_priors": np.full((2, 4), .25, dtype=np.float32),
        "anchor_frames": [2, 2],
        "session_ids": ["recording_0_session_1", "recording_1_session_1"],
    }
    torch.save(data, dataset_path)
    torch.save({"mcmc_prior_provenance": "global_oof_animal_grouped_cv",
                "animal_identity_status": "historical_unknown"}, checkpoint_path)
    historical_pin = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    def synthetic_model(path, _device):
        checkpoint_bytes = path.read_bytes()
        return SimpleNamespace(
            analysis_checkpoint_lineage=load_artifact_bytes(checkpoint_bytes),
            analysis_checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
            dt_ms=1.0,
        )

    model_loader = "_shared_load_model" if script == "analyze_gating" else "load_model_from_checkpoint"
    monkeypatch.setattr(module, model_loader, synthetic_model)
    populations = []

    def stop_after_priors(bio_dataset, **_kwargs):
        populations.append(bio_dataset.analysis_population)
        raise _Loaded

    monkeypatch.setattr(module, "create_optimized_dataloader", stop_after_priors)
    argv = ["--checkpoint", str(checkpoint_path), "--dataset", str(dataset_path)]
    with pytest.raises(ValueError, match="trusted_historical_checkpoint_sha256"):
        module.main(argv)
    assert not populations

    with pytest.raises(_Loaded):
        module.main(argv + ["--trusted_historical_checkpoint_sha256", historical_pin])
    assert populations[-1]["checkpoint_validation_scope"] == "historical_unscoped"
    assert populations[-1]["selection"] == "whole_corpus"
    assert populations[-1]["outer_val_indices"] is None

    data.update(mcmc_prior_provenance="oof_2fold_recording_prefix_grouped_cv",
                animal_identity_status="unverified")
    torch.save(data, dataset_path)
    modern = {
        "is_nested_cv": False,
        "mcmc_prior_provenance": data["mcmc_prior_provenance"],
        "animal_identity_status": "unverified",
        "validation_scope": "diagnostic_global_oof",
        "dataset_source_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dataset_source_binding": "sha256_bound",
    }
    torch.save(modern, checkpoint_path)
    independent_modern_pin = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    with pytest.raises(_Loaded):
        module.main(argv)
    assert populations[-1]["checkpoint_validation_scope"] == "diagnostic_global_oof"

    # A modern checkpoint retagged as unbound historical changes its bytes.
    data.update(mcmc_prior_provenance="oof_2fold_animal_grouped_cv",
                animal_identity_status="historical_unknown")
    torch.save(data, dataset_path)
    retagged = dict(modern, mcmc_prior_provenance=data["mcmc_prior_provenance"],
                    animal_identity_status="historical_unknown")
    for key in ("dataset_source_sha256", "dataset_source_binding", "validation_scope"):
        retagged.pop(key)
    torch.save(retagged, checkpoint_path)
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != independent_modern_pin
    accepted = len(populations)
    with pytest.raises(ValueError, match="trusted_historical_checkpoint_sha256"):
        module.main(argv + ["--trusted_historical_checkpoint_sha256", independent_modern_pin])
    assert len(populations) == accepted
