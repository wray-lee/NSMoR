"""Analysis-time priors from the checkpoint's validated nested split."""

import hashlib
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from nsmor.config import DEFAULT_FEATURE
from nsmor.pipeline.nested_prior import load_nested_prior_split
from nsmor.pipeline.grouping import prior_identity_status
from nsmor.model_utils import (
    require_trusted_historical_checkpoint_sha256, resolve_dataset_session_ids,
    validate_dataset_provenance,
)

logger = logging.getLogger(__name__)


def describe_analysis_population(
    n_corpus: int,
    val_indices: Optional[Sequence[int]],
    analyzed_indices: Sequence[int],
    validation_scope: Optional[str] = None,
) -> Dict[str, object]:
    """Record exactly which source rows a descriptive analysis includes."""
    analyzed = list(analyzed_indices)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           for i in analyzed):
        raise ValueError("Analyzed row indices must be integers")
    analyzed = [int(i) for i in analyzed]
    if (len(set(analyzed)) != len(analyzed)
            or any(i < 0 or i >= n_corpus for i in analyzed)):
        raise ValueError("Analyzed row indices must be unique corpus indices")
    val = None if val_indices is None else list(val_indices)
    if val is not None:
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
               for i in val):
            raise ValueError("Outer-validation indices must be integers")
        val = [int(i) for i in val]
    if val is not None and (len(set(val)) != len(val)
                            or any(i < 0 or i >= n_corpus for i in val)):
        raise ValueError("Outer-validation indices must be unique corpus indices")
    selected = set(analyzed)
    heldout = set(val) if val is not None else set()
    train = set(range(n_corpus)) - heldout
    n_train = len(train) if val is not None else None
    n_val = len(val) if val is not None else None
    analyzed_train = len(selected & train) if val is not None else None
    analyzed_val = len(selected & heldout) if val is not None else None
    selection = (
        "whole_corpus" if len(selected) == n_corpus else
        "outer_validation_only" if val is not None and selected == heldout else
        "outer_train_only" if val is not None and selected == train else
        "mixed_subset"
    )
    return {
        "selection": selection,
        "checkpoint_validation_scope": validation_scope,
        # Checkpoint selection and early stopping used validation rows; no untouched holdout.
        "evidence_scope": "in_sample_descriptive_only",
        "holdout_evidence": False,
        "outer_split": "persisted" if val is not None else "unavailable",
        "n_corpus": n_corpus, "n_train": n_train, "n_val": n_val,
        "n_analyzed": len(analyzed),
        "n_analyzed_train": analyzed_train, "n_analyzed_val": analyzed_val,
        "train_coverage": analyzed_train / n_train if n_train else None,
        "val_coverage": analyzed_val / n_val if n_val else None,
        "outer_val_indices": val,
    }


def population_for_output(dataloader: object, n_rows: int) -> Dict[str, object]:
    """Use loader provenance; injected legacy loaders declare unknown split."""
    population = getattr(getattr(dataloader, "dataset", None), "analysis_population", None)
    return population if population is not None else describe_analysis_population(
        n_rows, None, range(n_rows),
    )


def load_analysis_priors(dataset, dataset_path, nested_prior_artifact=None, model=None,
                         qc_sealed_nested_prior_sha256=None, *, loaded_source_fingerprint=None,
                         trusted_historical_checkpoint_sha256=None,
                         trusted_historical_artifact_sha256=None):
    """Bind nested row-aligned priors to the checkpoint's embedded artifact digest."""
    lineage = getattr(model, "analysis_checkpoint_lineage", None)
    if lineage is None and model is None and nested_prior_artifact is None and qc_sealed_nested_prior_sha256 is None:
        return dataset.get("mcmc_priors"), None  # Direct legacy dataset loader.
    if not isinstance(lineage, dict):
        raise ValueError("Analysis requires loaded checkpoint provenance")
    recorded = lineage.get("is_nested_cv")
    if recorded is None:
        if lineage.get("nested_prior_artifact") or lineage.get("nested_prior_fingerprint") or lineage.get("nested_prior_artifact_sha256"):
            raise ValueError("Incomplete checkpoint nested-prior provenance")
        recorded = False  # Genuine pre-nested legacy checkpoint.
    if type(recorded) is not bool:
        raise ValueError("Checkpoint is_nested_cv must be boolean")
    nested = recorded
    dataset_status = validate_dataset_provenance(dataset, Path(dataset_path))
    checkpoint_tag = lineage.get("mcmc_prior_provenance")
    if not nested and checkpoint_tag in (None, "global_oof_animal_grouped_cv"):
        if lineage.get("animal_identity_status") not in (None, "historical_unknown"):
            raise ValueError("Historical checkpoint cannot certify animal identity")
        checkpoint_status = "historical_unknown"
    else:
        checkpoint_status = prior_identity_status(
            checkpoint_tag, lineage.get("animal_identity_status"), nested=nested
        )
    def certify_scope():
        scope = lineage.get("validation_scope")
        expected_scope = "nested_outer_validation" if nested else "diagnostic_global_oof"
        if scope != expected_scope:
            if not (scope is None and checkpoint_status == "historical_unknown" and not nested):
                raise ValueError(f"Checkpoint validation_scope must be {expected_scope}; got {scope!r}")
            scope = "historical_unscoped"
        model.analysis_validation_scope = scope

    if not nested:
        if nested_prior_artifact is not None or qc_sealed_nested_prior_sha256 is not None:
            raise ValueError("Non-nested checkpoint cannot use nested prior artifact or QC digest")
        if (lineage.get("nested_prior_artifact") or lineage.get("nested_prior_fingerprint")
                or lineage.get("nested_prior_artifact_sha256")
                or lineage.get("nested_split_seed") not in (None, -1)
                or lineage.get("nested_val_split") not in (None, -1.0)):
            raise ValueError("Non-nested checkpoint carries conflicting nested-prior lineage")
        if dataset_status != checkpoint_status:
            raise ValueError("Dataset and checkpoint animal_identity_status mismatch")
        # Pre-nested checkpoints omitted the fold tag or used this umbrella tag.
        historical_alias = (
            dataset_status == checkpoint_status == "historical_unknown"
            and checkpoint_tag in (None, "global_oof_animal_grouped_cv")
        )
        if checkpoint_tag != dataset.get("mcmc_prior_provenance") and not historical_alias:
            raise ValueError("Dataset and checkpoint mcmc_prior_provenance mismatch")
        if checkpoint_status != "historical_unknown":
            if "dataset_source_sha256" not in lineage:
                raise ValueError("Modern checkpoint requires dataset_source_sha256")
            if lineage.get("dataset_source_binding", "sha256_bound") != "sha256_bound":
                raise ValueError("Modern checkpoint requires sha256_bound dataset_source_binding")
        saved_dataset = lineage.get("dataset_path")
        if saved_dataset and Path(saved_dataset).resolve() != Path(dataset_path).resolve():
            raise ValueError("Dataset path does not match checkpoint provenance")
        model.analysis_dataset_binding = "pending_sha256"
        if "dataset_source_sha256" in lineage:
            digest = lineage["dataset_source_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid checkpoint dataset_source_sha256")
            if loaded_source_fingerprint is None or digest != loaded_source_fingerprint:
                raise ValueError("Analysis dataset_source_sha256 mismatch with checkpoint")
            binding = lineage.get("dataset_source_binding", "sha256_bound")
            if binding not in ("sha256_bound", "legacy_resume_unbound"):
                raise ValueError("Invalid checkpoint dataset_source_binding")
            model.analysis_dataset_binding = (
                "sha256_verified" if binding == "sha256_bound" else "legacy_resume_unbound"
            )
        else:
            binding = lineage.get("dataset_source_binding", "legacy_unbound")
            if binding not in ("legacy_unbound", "unbound_lazy"):
                raise ValueError("Checkpoint binding status lacks dataset_source_sha256")
            model.analysis_dataset_binding = binding
        if checkpoint_status == "historical_unknown":
            require_trusted_historical_checkpoint_sha256(
                getattr(model, "analysis_checkpoint_sha256", None),
                trusted_historical_checkpoint_sha256,
            )
        certify_scope()
        model.analysis_animal_identity_status = checkpoint_status
        if model.analysis_dataset_binding != "sha256_verified":
            status = model.analysis_dataset_binding
            logger.warning("Analysis uses %s: %s dataset content; historical weights are not content-bound",
                           status, "legacy unbound" if status == "legacy_unbound" else "unbound")
        return dataset.get("mcmc_priors"), None
    if checkpoint_status == "historical_unknown":
        require_trusted_historical_checkpoint_sha256(
            getattr(model, "analysis_checkpoint_sha256", None),
            trusted_historical_checkpoint_sha256,
        )
    if loaded_source_fingerprint is None:
        raise ValueError("Nested analysis requires a loaded source fingerprint")
    expected_path = lineage.get("nested_prior_artifact")
    saved_dataset = lineage.get("dataset_path")
    if not expected_path or not saved_dataset:
        raise ValueError("Nested checkpoint missing artifact or dataset path provenance")
    artifact_path = Path(nested_prior_artifact or expected_path)
    if Path(expected_path).resolve() != artifact_path.resolve():
        raise ValueError("Nested artifact path does not match checkpoint provenance")
    if Path(saved_dataset).resolve() != Path(dataset_path).resolve():
        raise ValueError("Dataset path does not match checkpoint provenance")
    saved_digest = lineage.get("nested_prior_artifact_sha256")
    if saved_digest is None:
        raise ValueError("Legacy nested checkpoint lacks embedded artifact SHA-256; caller digest cannot authenticate checkpoint lineage")
    if qc_sealed_nested_prior_sha256 is not None and saved_digest != qc_sealed_nested_prior_sha256:
        raise ValueError("Supplied artifact digest conflicts with checkpoint digest")
    digest = saved_digest
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Nested artifact SHA-256 must be a 64-character lowercase hex digest")
    def verify_artifact_bytes():
        with artifact_path.open("rb") as stream:
            actual = hashlib.sha256(stream.read()).hexdigest()
        if actual != digest:
            raise ValueError("Nested artifact SHA-256 mismatch with checkpoint")
    verify_artifact_bytes()
    _, val_indices, priors, info = load_nested_prior_split(
        artifact_path, Path(dataset_path), len(dataset["X_seqs"]),
        resolve_dataset_session_ids(dataset),
        feature_config=dataset.get("feature_config", DEFAULT_FEATURE),
        loaded_source_fingerprint=loaded_source_fingerprint,
        trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
    )
    if info["nested_prior_artifact_sha256"] != digest:
        raise ValueError("Nested artifact SHA-256 mismatch with checkpoint")
    verify_artifact_bytes()  # Also refuse a file replaced during validation.
    seed = lineage.get("nested_split_seed")
    fraction = lineage.get("nested_val_split")
    if type(seed) is not int or seed != info["split_seed"]:
        raise ValueError("Nested split seed does not match checkpoint provenance")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(float(fraction)) or not math.isclose(float(fraction), info["val_split"], abs_tol=1e-6):
        raise ValueError("Nested validation fraction does not match checkpoint provenance")
    if lineage.get("nested_prior_fingerprint") != info["nested_prior_fingerprint"]:
        raise ValueError("Nested source fingerprint does not match checkpoint provenance")
    if lineage.get("mcmc_prior_provenance") != info["mcmc_prior_provenance"]:
        raise ValueError("Nested prior provenance does not match checkpoint provenance")
    if checkpoint_status != info["animal_identity_status"]:
        raise ValueError("Nested animal_identity_status does not match checkpoint provenance")
    assert priors.shape == (len(dataset["X_seqs"]), dataset.get("feature_config", DEFAULT_FEATURE).mcmc_dim)
    certify_scope()
    model.analysis_animal_identity_status = checkpoint_status
    logger.info(
        "Analysis priors: nested artifact=%s dataset SHA-256=%s train=%d val=%d provenance=%s",
        info["nested_prior_artifact"], info["nested_prior_fingerprint"],
        info["n_train"], info["n_val"], info["mcmc_prior_provenance"],
    )
    return priors, val_indices
