"""Nonnested analysis must reconcile prior tags before verifying dataset binding."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nsmor.analysis.analysis_priors import load_analysis_priors
from nsmor.config import PIPELINE_SEMANTICS_VERSION


@pytest.mark.parametrize("grouping,checkpoint_tag,accepted", [
    ("recording_prefix", "oof_2fold_recording_prefix_grouped_cv", True),
    ("recording_prefix", "oof_5fold_recording_prefix_grouped_cv", False),
    ("animal", "oof_2fold_animal_grouped_cv", True),
    ("animal", "oof_5fold_animal_grouped_cv", False),
    ("animal", None, True),
    ("animal", "global_oof_animal_grouped_cv", True),
    ("recording_prefix", None, False),
    ("recording_prefix", "global_oof_animal_grouped_cv", False),
])
def test_nonnested_analysis_prior_lineage(tmp_path, grouping, checkpoint_tag, accepted):
    status = "unverified" if grouping == "recording_prefix" else "historical_unknown"
    dataset = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": f"oof_2fold_{grouping}_grouped_cv",
        "animal_identity_status": status,
        "mcmc_priors": np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        "X_seqs": [np.zeros((1, 8)) for _ in range(2)],
        "session_ids": ["syntheticA_session_1", "syntheticB_session_1"],
    }
    assert dataset["mcmc_priors"].shape == (2, 4)
    dataset_path = tmp_path / "synthetic.pt"
    torch.save(dataset, dataset_path)
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    lineage = {"is_nested_cv": False, "dataset_source_sha256": digest,
               "dataset_source_binding": "sha256_bound"}
    if status == "unverified":
        lineage["validation_scope"] = "diagnostic_global_oof"
    if checkpoint_tag is not None:
        lineage["mcmc_prior_provenance"] = checkpoint_tag
    if checkpoint_tag not in (None, "global_oof_animal_grouped_cv"):
        lineage["animal_identity_status"] = status
    checkpoint_path = tmp_path / "synthetic.pth"
    torch.save(lineage, checkpoint_path)
    model = SimpleNamespace(
        analysis_checkpoint_lineage=lineage,
        analysis_checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    )
    kwargs = {"model": model, "loaded_source_fingerprint": digest}
    if not accepted:
        with pytest.raises(ValueError, match="mcmc_prior_provenance|animal_identity_status"):
            load_analysis_priors(dataset, dataset_path, **kwargs)
        assert not hasattr(model, "analysis_dataset_binding")
        return
    if status == "historical_unknown":
        kwargs["trusted_historical_checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    priors, val_indices = load_analysis_priors(dataset, dataset_path, **kwargs)
    assert priors is dataset["mcmc_priors"]
    assert val_indices is None
    assert model.analysis_dataset_binding == "sha256_verified"
    assert model.analysis_animal_identity_status == status
    assert model.analysis_validation_scope == (
        "diagnostic_global_oof" if status == "unverified" else "historical_unscoped"
    )


@pytest.mark.parametrize("grouping,with_digest,binding,expected_binding", [
    ("recording_prefix", False, None, None),
    ("recording_prefix", False, "legacy_unbound", None),
    ("recording_prefix", False, "unbound_lazy", None),
    ("recording_prefix", False, "sha256_bound", None),
    ("recording_prefix", True, "legacy_resume_unbound", None),
    ("recording_prefix", True, None, "sha256_verified"),
    ("animal", False, None, "legacy_unbound"),
    ("animal", False, "unbound_lazy", "unbound_lazy"),
    ("animal", True, "legacy_resume_unbound", "legacy_resume_unbound"),
])
def test_nonnested_analysis_requires_modern_content_binding(
        tmp_path, grouping, with_digest, binding, expected_binding):
    status = "unverified" if grouping == "recording_prefix" else "historical_unknown"
    dataset = {"pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
               "mcmc_prior_provenance": f"oof_2fold_{grouping}_grouped_cv",
               "animal_identity_status": status, "mcmc_priors": np.full((2, 4), 0.25),
               "X_seqs": [np.zeros((1, 8)) for _ in range(2)],
               "session_ids": ["syntheticA_session_1", "syntheticB_session_1"]}
    assert dataset["mcmc_priors"].shape == (2, 4)
    dataset_path = tmp_path / "synthetic.pt"
    torch.save(dataset, dataset_path)
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    lineage = {"is_nested_cv": False, "dataset_path": str(tmp_path / "synthetic.pt"),
               "mcmc_prior_provenance": dataset["mcmc_prior_provenance"],
               "animal_identity_status": status}
    if status == "unverified":
        lineage["validation_scope"] = "diagnostic_global_oof"
    if with_digest:
        lineage["dataset_source_sha256"] = digest
    if binding is not None:
        lineage["dataset_source_binding"] = binding
    checkpoint_path = tmp_path / "synthetic.pth"
    torch.save(lineage, checkpoint_path)
    model = SimpleNamespace(
        analysis_checkpoint_lineage=lineage,
        analysis_checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    )
    kwargs = {"model": model, "loaded_source_fingerprint": digest}
    if expected_binding is None:
        with pytest.raises(ValueError, match="dataset_source_sha256|dataset_source_binding"):
            load_analysis_priors(dataset, dataset_path, **kwargs)
        assert getattr(model, "analysis_dataset_binding", None) != "sha256_verified"
        return
    if status == "historical_unknown":
        kwargs["trusted_historical_checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    priors, val_indices = load_analysis_priors(dataset, dataset_path, **kwargs)
    assert priors is dataset["mcmc_priors"] and val_indices is None
    assert model.analysis_dataset_binding == expected_binding
    assert model.analysis_animal_identity_status == status
    assert model.analysis_validation_scope == (
        "diagnostic_global_oof" if status == "unverified" else "historical_unscoped"
    )
