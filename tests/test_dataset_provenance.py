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


def test_validate_dataset_provenance_accepts_valid_animal_grouped_prior(tmp_path: Path):
    """Animal-grouped cross-fitted prior provenance must be accepted."""
    ds = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
    }
    # Should not raise
    validate_dataset_provenance(ds, tmp_path / "dummy.pt")
