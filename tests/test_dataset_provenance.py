"""Tests for dataset provenance validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from nsmor.config import PIPELINE_SEMANTICS_VERSION
from nsmor.model_utils import validate_dataset_provenance


def test_validate_dataset_provenance_rejects_missing_prior_provenance(tmp_path: Path):
    """Corpora without mcmc_prior_provenance must be rejected."""
    ds = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        # Missing mcmc_prior_provenance
    }
    with pytest.raises(RuntimeError, match="mcmc_prior_provenance"):
        validate_dataset_provenance(ds, tmp_path / "dummy.pt")


def test_validate_dataset_provenance_rejects_session_grouped_prior_provenance(tmp_path: Path):
    """Session-grouped or invalid format prior provenance must be rejected."""
    ds = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "session_grouped_5fold",
    }
    with pytest.raises(RuntimeError, match="mcmc_prior_provenance"):
        validate_dataset_provenance(ds, tmp_path / "dummy.pt")


def test_validate_dataset_provenance_accepts_historical_prefix_prior(tmp_path: Path):
    """Historical animal-named artifacts load with unknown identity."""
    ds = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
    }
    assert validate_dataset_provenance(ds, tmp_path / "dummy.pt") == "historical_unknown"


def test_prefix_disjointness_does_not_verify_animal_independence(tmp_path: Path):
    from nsmor.pipeline.grouping import animal_keys_of, check_group_disjoint
    import numpy as np

    # One stipulated animal recorded under two unrelated day prefixes.
    sessions = ["subjectA_day1_session_1", "subjectA_day2_session_1"]
    keys = animal_keys_of(sessions)
    check_group_disjoint(np.array([0]), np.array([1]), keys)
    assert keys.tolist() == ["subjectA_day1", "subjectA_day2"]
    ds = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
        "session_ids": sessions,
    }
    assert validate_dataset_provenance(ds, tmp_path / "synthetic.pt") == "unverified"
    ds["animal_identity_status"] = "verified"
    with pytest.raises(RuntimeError, match="animal_identity_status"):
        validate_dataset_provenance(ds, tmp_path / "synthetic.pt")


def test_historical_animal_named_provenance_is_unknown(tmp_path: Path):
    ds = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_2fold_animal_grouped_cv",
    }
    assert validate_dataset_provenance(ds, tmp_path / "historical.pt") == "historical_unknown"
    ds["animal_identity_status"] = "unverified"
    with pytest.raises(RuntimeError, match="animal_identity_status"):
        validate_dataset_provenance(ds, tmp_path / "historical.pt")


def test_new_prefix_lineage_requires_explicit_identity_status(tmp_path: Path):
    ds = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
    }
    with pytest.raises(RuntimeError, match="animal_identity_status"):
        validate_dataset_provenance(ds, tmp_path / "unstamped.pt")


def test_historical_gate_keyword_is_a_prefix_gate(tmp_path: Path):
    ds = {"pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
          "mcmc_prior_provenance": "oof_2fold_animal_grouped_cv"}
    assert validate_dataset_provenance(
        ds, tmp_path / "old.pt", require_animal_grouped_priors=True
    ) == "historical_unknown"


@pytest.mark.parametrize("sessions", [None, ["A_session_1"], ["A_session_1", ""], ["A_session_1", None], ["A_session_1", "_session_1"]])
def test_modern_prior_requires_aligned_usable_recording_ids(tmp_path: Path, sessions):
    ds = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
        "X_seqs": [None, None],
    }
    if sessions is not None:
        ds["session_ids"] = sessions
    with pytest.raises(RuntimeError, match="session_ids|recording-prefix"):
        validate_dataset_provenance(ds, tmp_path / "missing-recording.pt")


@pytest.mark.parametrize("field,value", [
    ("pipeline_semantics_version", "invalid"),
    ("mcmc_prior_provenance", None),
])
def test_lazy_reader_applies_shared_provenance_gate(tmp_path: Path, field, value):
    import torch
    from nsmor.lazy_dataloader import NSMoRLazyDataset

    metadata = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
        "trial_specs": [{"session_id": "A_session_1"}, {"session_id": "B_session_1"}],
        "mcmc_priors": torch.full((2, 4), 0.25),
    }
    metadata[field] = value
    with pytest.raises(RuntimeError, match="pipeline semantics|mcmc_prior_provenance"):
        NSMoRLazyDataset(str(tmp_path / "captured.pt"), metadata=metadata)