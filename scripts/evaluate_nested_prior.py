#!/usr/bin/env python3
"""
Nested Cross-Validation Prior Generation Tool.

Generates nested cross-validation MCMC prior features:
1. Outer split keeps session-name recording prefixes disjoint across train/val.
2. Inner grouped OOF priors fit only on outer-train recording prefixes.
3. Fold ensemble trained only on outer-train predicts priors for outer-val.
4. Produces an aligned (N, 4) prior feature matrix for downstream use.
Prefixes do not verify independent animal identities across recordings.

NOTE: This tool generates nested MCMC prior features only; downstream NSMoR
recurrent model regression training and evaluation is performed separately.

Usage:
    python scripts/evaluate_nested_prior.py --dataset data/processed/nsmor_dataset.pt
    python scripts/evaluate_nested_prior.py --dataset data/processed/nsmor_dataset.pt --split_seed 42 --epochs 200
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from nsmor.config import (
    DEFAULT_FEATURE,
    DEFAULT_MCMC_TRAINING,
    FeatureConfig,
    Label,
    MCMCTrainingConfig,
    PIPELINE_SEMANTICS_VERSION,
)
from nsmor.mcmc_module import MCMCPriorGenerator, train_mcmc_cross_fitted
from nsmor.pipeline.grouping import (
    animal_keys_of,
    check_group_disjoint,
    grouped_train_val_split,
    resolve_group_folds,
)
# Single canonical fingerprint implementation (str- and Path-safe, fails
# closed on unusable paths).  The previous local copy accepted Path only,
# so a str path raised AttributeError mid-generation, and returned ""
# for missing files, letting artifacts be emitted with an unverifiable
# source.  One contract, one implementation.
from nsmor.pipeline.nested_prior import compute_source_fingerprint, load_dataset_with_fingerprint

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. Fingerprint Helper (re-exported from nsmor.pipeline.nested_prior)
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# 2. Input Validation (Fail Closed)
# ═══════════════════════════════════════════════════════════════


def validate_nested_dataset(
    dataset: Dict[str, Any],
    feature_config: FeatureConfig,
) -> Tuple[np.ndarray, np.ndarray, Sequence[str], np.ndarray]:
    """Validate dataset inputs and fail closed on missing/invalid fields.

    Args:
        dataset: Loaded dataset dictionary.
        feature_config: Feature dimension configuration.

    Returns:
        Tuple of ``(snapshots, labels, session_ids, group_keys)``.

    Raises:
        ValueError: If any validation check fails.
    """
    if not isinstance(dataset, dict):
        raise ValueError(f"Dataset must be a dictionary, got {type(dataset).__name__}")

    # ── Pipeline semantics version guard ──
    psv = dataset.get("pipeline_semantics_version")
    if psv is None:
        raise ValueError(
            "Dataset lacks 'pipeline_semantics_version'. "
            "Legacy unversioned artifacts cannot be used for nested CV."
        )
    if str(psv) != str(PIPELINE_SEMANTICS_VERSION):
        raise ValueError(
            f"Dataset pipeline_semantics_version {psv!r} != "
            f"expected {PIPELINE_SEMANTICS_VERSION!r}."
        )

    # ── Canonical snapshots guard ──
    snapshots = dataset.get("snapshots")
    if snapshots is None:
        snapshots = dataset.get("mcmc_snapshots")
    if snapshots is None:
        raise ValueError(
            "Dataset lacks canonical 'snapshots' (or 'mcmc_snapshots'). "
            "Legacy artifacts without persisted canonical snapshots cannot be used "
            "for nested CV evaluation; re-run scripts/prepare_data.py to regenerate."
        )
    if isinstance(snapshots, torch.Tensor):
        snapshots = snapshots.cpu().numpy()
    snapshots_arr = np.asarray(snapshots, dtype=np.float64)

    if snapshots_arr.ndim != 2 or snapshots_arr.shape[1] != feature_config.snapshot_dim:
        raise ValueError(
            f"Canonical snapshots shape {snapshots_arr.shape} is invalid; "
            f"expected (N, {feature_config.snapshot_dim})."
        )
    if not np.isfinite(snapshots_arr).all():
        raise ValueError("Canonical snapshots contain non-finite values (NaN or Inf).")

    n_trials = snapshots_arr.shape[0]
    if n_trials == 0:
        raise ValueError("Dataset has 0 trials; cannot perform nested CV.")

    # ── Labels guard ──
    labels = dataset.get("labels")
    if labels is None:
        raise ValueError("Dataset lacks 'labels'.")
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    labels_arr = np.asarray(labels, dtype=np.int64)

    if labels_arr.ndim != 1 or len(labels_arr) != n_trials:
        raise ValueError(
            f"Labels shape {labels_arr.shape} does not match snapshots length {n_trials}."
        )
    if not np.isfinite(labels_arr).all():
        raise ValueError("Labels contain non-finite values.")
    if (labels_arr < 0).any() or (labels_arr >= feature_config.num_classes).any():
        raise ValueError(
            f"Labels contain values outside valid range [0, {feature_config.num_classes - 1}]."
        )

    # ── Session IDs / recording-prefix guard ──
    session_ids = dataset.get("session_ids")
    if session_ids is None:
        raise ValueError(
            "Dataset lacks 'session_ids'. Recording prefixes are required "
            "for grouped nested CV; sample-split fallback is forbidden."
        )
    if len(session_ids) != n_trials:
        raise ValueError(
            f"session_ids length {len(session_ids)} != labels count {n_trials}."
        )
    for idx, s in enumerate(session_ids):
        if not isinstance(s, (str, np.str_)) or not str(s).strip():
            raise ValueError(
                f"session_ids[{idx}] has invalid or empty session ID: {s!r}"
            )

    group_keys = animal_keys_of(session_ids)
    assert group_keys.shape == (n_trials,), (
        f"Expected group_keys shape ({n_trials},), got {group_keys.shape}"
    )

    unique_prefixes = np.unique(group_keys)
    if len(unique_prefixes) < 2:
        raise ValueError(
            f"Dataset has only {len(unique_prefixes)} recording prefix(es); "
            "at least 2 are required for a prefix-disjoint outer train/val split."
        )

    return snapshots_arr, labels_arr, session_ids, group_keys


# ═══════════════════════════════════════════════════════════════
# 3. Core Nested CV Generation
# ═══════════════════════════════════════════════════════════════


def generate_nested_priors(
    dataset: Optional[Dict[str, Any]],
    split_seed: int = 42,
    val_split: float = 0.2,
    n_inner_folds: int = 5,
    mcmc_config: Optional[MCMCTrainingConfig] = None,
    feature_config: Optional[FeatureConfig] = None,
    source_dataset_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Generate nested priors with recording-prefix-disjoint folds.

    Prefixes come from session names, not verified animal identifiers.

    Protocol:
    1. Outer split by recording prefix (no prefix crosses train/val).
    2. Inner StratifiedGroupKFold CV fitted only on outer-train prefixes to generate
       train out-of-fold (OOF) priors.
    3. Fold models trained on outer-train partitions predict probabilities for outer-val,
       averaged into an ensemble prior vector per validation trial.
    4. Combines train OOF and val ensemble priors into an aligned (N, 4) matrix.

    Validation labels are NEVER used in fitting, fold resolution, normalization,
    or feature preprocessing.

    Args:
        dataset: Optional in-memory dataset to compare with the authoritative
            source snapshot. The CLI passes None and uses the captured source.
        split_seed: Random seed for outer recording-prefix split.
        val_split: Target outer validation fraction of *recording prefixes* (0-1).
            Grouped split granularity means the realized *trial* fraction
            generally differs; see ``val_trial_fraction`` in the output.
        n_inner_folds: Maximum inner cross-validation folds.
        mcmc_config: Hyperparameters for MCMC training.
        feature_config: Feature dimension configuration.
        source_dataset_path: Path to the source dataset file.  Required:
            the artifact's provenance fingerprint is computed from this
            file and the generator fails closed if it cannot be
            fingerprinted — an artifact with an empty fingerprint can
            never be loaded, so it must never be written.
        verbose: Log intermediate progress.

    Returns:
        Dictionary containing generated nested priors, split indices, provenance,
        diagnostics, and fingerprint. Legacy ``train_animals``/``val_animals``
        and ``n_train_animals``/``n_val_animals`` contain recording prefixes,
        not verified animal identities.
    """
    mc = mcmc_config if mcmc_config is not None else DEFAULT_MCMC_TRAINING

    # One authoritative byte snapshot supplies both the source fingerprint
    # and every input used to generate priors. The supplied in-memory dataset
    # remains an independent consistency check below.
    authoritative_dataset, fingerprint = load_dataset_with_fingerprint(
        source_dataset_path, restore_provenance=False,
    )
    if not isinstance(authoritative_dataset, dict):
        raise ValueError(
            f"Authoritative dataset file must contain a dict, got {type(authoritative_dataset).__name__}"
        )

    fc = feature_config if feature_config is not None else authoritative_dataset.get("feature_config", DEFAULT_FEATURE)

    if dataset is not None and isinstance(dataset, dict):
        if "labels" in dataset and "labels" in authoritative_dataset:
            in_lbl = np.asarray(dataset["labels"])
            auth_lbl = np.asarray(authoritative_dataset["labels"])
            if in_lbl.shape != auth_lbl.shape or not np.array_equal(in_lbl, auth_lbl):
                raise ValueError(
                    f"In-memory labels disagree with authoritative source dataset at "
                    f"{source_dataset_path}; refuse to generate priors from inconsistent "
                    "in-memory labels (in-memory/file mismatch)."
                )
        if "session_ids" in dataset and "session_ids" in authoritative_dataset:
            in_sess = dataset["session_ids"]
            auth_sess = authoritative_dataset["session_ids"]
            if len(in_sess) != len(auth_sess) or [str(s) for s in in_sess] != [str(s) for s in auth_sess]:
                raise ValueError(
                    f"In-memory session_ids disagree with authoritative source dataset at "
                    f"{source_dataset_path}; refuse to generate priors from inconsistent "
                    "in-memory data (in-memory/file mismatch)."
                )
        in_snaps = dataset.get("snapshots") if dataset.get("snapshots") is not None else dataset.get("mcmc_snapshots")
        auth_snaps = authoritative_dataset.get("snapshots") if authoritative_dataset.get("snapshots") is not None else authoritative_dataset.get("mcmc_snapshots")
        if in_snaps is not None and auth_snaps is not None:
            in_snaps_arr = np.asarray(in_snaps)
            auth_snaps_arr = np.asarray(auth_snaps)
            if in_snaps_arr.shape != auth_snaps_arr.shape or not np.array_equal(in_snaps_arr, auth_snaps_arr):
                raise ValueError(
                    f"In-memory snapshots disagree with authoritative source dataset at "
                    f"{source_dataset_path}; refuse to generate priors from inconsistent "
                    "in-memory data (in-memory/file mismatch)."
                )

    # 1. Validate inputs (bound strictly to authoritative dataset)
    snapshots, labels, session_ids, group_keys = validate_nested_dataset(authoritative_dataset, fc)
    n_total = len(labels)

    # 2. Outer recording-prefix-grouped train/val split
    train_idx, val_idx = grouped_train_val_split(
        session_ids, n_total, val_split=val_split, random_seed=split_seed,
    )
    assert len(train_idx) > 0, "Outer split yielded empty train partition"
    assert len(val_idx) > 0, "Outer split yielded empty val partition"
    check_group_disjoint(train_idx, val_idx, group_keys)

    # 3. Outer-train slice
    train_snapshots = snapshots[train_idx]
    train_labels = labels[train_idx]
    train_groups = group_keys[train_idx]

    assert train_snapshots.shape == (len(train_idx), fc.snapshot_dim), (
        f"train_snapshots shape {train_snapshots.shape} != ({len(train_idx)}, {fc.snapshot_dim})"
    )
    assert train_labels.shape == (len(train_idx),), (
        f"train_labels shape {train_labels.shape} != ({len(train_idx)},)"
    )
    assert train_groups.shape == (len(train_idx),), (
        f"train_groups shape {train_groups.shape} != ({len(train_idx)},)"
    )

    # Check class coverage in outer-train against the CANONICAL class
    # vocabulary (range(fc.num_classes)), never against ``np.unique(labels)``.
    # The latter includes validation labels, so the pass/fail of this gate
    # — and therefore the produced priors — would depend on held-out
    # labels (val-label disclosure).  The prior simplex spans the full
    # canonical vocabulary (class_order), so every canonical class must
    # be observable in outer-train for the inner MCMC model to be
    # identifiable, regardless of what the val prefixes contain.
    required_classes = set(range(int(fc.num_classes)))
    train_classes = set(int(c) for c in np.unique(train_labels).tolist())
    missing_classes = required_classes - train_classes
    if missing_classes:
        raise ValueError(
            f"Outer-train split is missing class(es): {sorted(missing_classes)} "
            f"of the canonical vocabulary of {fc.num_classes} class(es). "
            "Insufficient training class coverage; cannot train inner MCMC model. "
            "Coverage is checked against the canonical class vocabulary only — "
            "validation labels are never consulted."
        )

    # Check recording-prefix count in outer-train
    train_unique_prefixes = np.unique(train_groups)
    if len(train_unique_prefixes) < 2:
        raise ValueError(
            f"Outer-train has only {len(train_unique_prefixes)} recording prefix(es); "
            "cannot perform inner cross-fitting."
        )

    # 4. Resolve inner folds using outer-train coverage ONLY
    actual_n_inner_folds = resolve_group_folds(
        train_labels, train_groups, max_folds=n_inner_folds,
    )

    # 5. Inner cross-fitting on outer-train ONLY
    train_oof_priors, inner_fold_models, inner_fold_diagnostics = train_mcmc_cross_fitted(
        snapshots=train_snapshots,
        labels=train_labels,
        config=mc,
        feature_config=fc,
        n_folds=actual_n_inner_folds,
        groups=train_groups,
        verbose=verbose,
    )

    assert train_oof_priors.shape == (len(train_idx), fc.mcmc_dim), (
        f"train_oof_priors shape {train_oof_priors.shape} != ({len(train_idx)}, {fc.mcmc_dim})"
    )
    assert np.isfinite(train_oof_priors).all(), "Train OOF priors contain non-finite values"
    assert (train_oof_priors >= 0.0).all(), "Train OOF priors contain negative values"
    assert abs(train_oof_priors.sum(axis=1) - 1.0).max() < 1e-4, "Train OOF prior rows must sum to 1"

    # 6. Outer-val priors via ensemble of outer-train fold models
    val_snapshots = snapshots[val_idx]
    val_priors = np.zeros((len(val_idx), fc.mcmc_dim), dtype=np.float64)
    for fm in inner_fold_models:
        val_priors += fm.predict_proba(val_snapshots)
    val_priors /= len(inner_fold_models)
    val_priors = np.clip(val_priors, 1e-12, 1.0)
    val_priors /= val_priors.sum(axis=1, keepdims=True)

    assert val_priors.shape == (len(val_idx), fc.mcmc_dim), (
        f"val_priors shape {val_priors.shape} != ({len(val_idx)}, {fc.mcmc_dim})"
    )
    assert np.isfinite(val_priors).all(), "Val ensemble priors contain non-finite values"
    assert (val_priors >= 0.0).all(), "Val ensemble priors contain negative values"
    assert abs(val_priors.sum(axis=1) - 1.0).max() < 1e-4, "Val ensemble prior rows must sum to 1"

    # 7. Assemble full (N, 4) nested priors
    nested_priors = np.zeros((n_total, fc.mcmc_dim), dtype=np.float64)
    nested_priors[train_idx] = train_oof_priors
    nested_priors[val_idx] = val_priors

    assert nested_priors.shape == (n_total, fc.mcmc_dim), (
        f"nested_priors shape {nested_priors.shape} != ({n_total}, {fc.mcmc_dim})"
    )
    assert np.isfinite(nested_priors).all(), "Nested priors contain non-finite values"
    assert (nested_priors >= 0.0).all(), "Nested priors contain negative values"
    assert abs(nested_priors.sum(axis=1) - 1.0).max() < 1e-4, "Nested prior rows must sum to 1"

    # Realized split sizes/fractions — recorded alongside the *requested*
    # recording-prefix-level val_split so no consumer has to overclaim an exact
    # trial fraction the grouped split cannot honour.  (``fingerprint``
    # was computed and verified non-empty at step 0.)
    n_train = int(len(train_idx))
    n_val = int(len(val_idx))
    val_trial_fraction = float(n_val) / float(n_total)

    train_prefixes = sorted(set(train_groups.tolist()))
    val_prefixes = sorted(set(group_keys[val_idx].tolist()))
    # Legacy animal-named aliases remain prefix lists/counts.
    return {
        "nested_priors": nested_priors,
        "train_priors": train_oof_priors,
        "val_priors": val_priors,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "group_keys": group_keys,
        "recording_prefix_keys": group_keys,
        "train_recording_prefixes": train_prefixes,
        "val_recording_prefixes": val_prefixes,
        "train_animals": train_prefixes,
        "val_animals": val_prefixes,
        "source_dataset": str(source_dataset_path),
        "source_fingerprint": fingerprint,
        "split_seed": int(split_seed),
        "val_split": float(val_split),
        "val_trial_fraction": val_trial_fraction,
        "n_train": n_train,
        "n_val": n_val,
        "n_train_recording_prefixes": len(train_prefixes),
        "n_val_recording_prefixes": len(val_prefixes),
        "n_train_animals": len(train_prefixes),
        "n_val_animals": len(val_prefixes),
        "n_inner_folds": int(actual_n_inner_folds),
        "requested_n_inner_folds": int(n_inner_folds),
        "class_order": [label.name for label in Label],
        "mcmc_prior_provenance": (
            f"nested_outer_seed{split_seed}_inner_{actual_n_inner_folds}fold_recording_prefix_grouped"
        ),
        "is_nested_cv": True,
        "animal_identity_status": "unverified",
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "inner_fold_diagnostics": inner_fold_diagnostics,
    }


# ═══════════════════════════════════════════════════════════════
# 4. CLI Entry Point
# ═══════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for nested prior CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate nested cross-validation MCMC priors "
            "(outer recording-prefix split, inner OOF fitting on outer-train). "
            "Independent animal identity across prefixes is unverified. "
            "Note: This tool produces nested MCMC prior features; downstream NSMoR "
            "regression model training is performed separately."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Input processed dataset path (.pt).",
    )
    parser.add_argument(
        "--output_dir",
        default="results/nested_prior",
        type=str,
        help="Output directory.",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Outer recording-prefix split random seed.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Outer validation fraction (recording prefixes).",
    )
    parser.add_argument(
        "--n_inner_folds",
        type=int,
        default=5,
        help="Maximum inner CV folds on outer-train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="MCMC training epochs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for evaluate_nested_prior."""
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"nested_split_seed{args.split_seed}.pt"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists at {output_path}. Pass --overwrite to overwrite."
        )

    logger.info("=== Nested Prior Evaluation Tool ===")
    logger.info("Dataset: %s", dataset_path)
    logger.info("Recording-prefix split seed: %d, requested val_split: %.2f", args.split_seed, args.val_split)
    logger.info("Requested inner folds: %d, epochs: %d", args.n_inner_folds, args.epochs)

    mcmc_config = dataclasses.replace(
        DEFAULT_MCMC_TRAINING,
        num_epochs=args.epochs,
        random_seed=args.split_seed,
    )

    result = generate_nested_priors(
        dataset=None,  # generate_nested_priors reads one authoritative byte snapshot.
        split_seed=args.split_seed,
        val_split=args.val_split,
        n_inner_folds=args.n_inner_folds,
        mcmc_config=mcmc_config,
        source_dataset_path=dataset_path,
        verbose=True,
    )

    torch.save(result, output_path)
    logger.info("Saved nested priors to: %s", output_path)
    logger.info(
        "Recording prefixes: %d train, %d val (legacy n_train_animals/n_val_animals; "
        "animal identities across prefixes unverified).",
        result["n_train_animals"], result["n_val_animals"],
    )
    logger.info(
        "Nested prior generation complete: %d train, %d val trials. "
        "Provenance: %s",
        len(result["train_indices"]),
        len(result["val_indices"]),
        result["mcmc_prior_provenance"],
    )


if __name__ == "__main__":
    main()
