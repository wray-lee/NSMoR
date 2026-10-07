"""Fixed-family evaluation and descriptive statistics for NSMoR optimization.

Pure evaluation routines implementing the accepted protocol in
``docs/model-optimization-protocol-20261004.md``:
  * Full-frame (t >= 0) and aligned (t >= 1) pooled regression metrics.
  * 12 sustained high-velocity band/run cells + rest complements.
  * Independent trial-local persistence (y_true[t-1]) and zero comparators.
  * Exact target hash and trial/frame identity validation (fail-closed).
  * Assembly and Holm-Bonferroni reduction of the frozen 78-slot inferential family
    via ``nsmor.analysis.model_comparison``.
  * Contiguous flat NPZ prediction cache serialization (allow_pickle=False).

Scope: Read-only post-processing of predictions and ground-truth targets.
No model training, loss modification, or biological-pathway inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from nsmor.analysis.model_comparison import (
    ALIGNED_OVERALL_SCOPE,
    CANDIDATE_IDS,
    COMPARATOR_IDS,
    DECLARED_FAMILY_SIZE,
    SCOPE_IDS,
    SENSITIVITY_THRESHOLDS_CM_S,
    SUSTAINED_MIN_RUN_LENGTHS,
    declared_family_slots,
    holm_correct_declared_family,
    paired_mse_comparison,
)
from scripts.train import _sustained_run

__all__ = [
    "ALIGNED_OVERALL_SCOPE",
    "CANDIDATE_IDS",
    "COMPARATOR_IDS",
    "DECLARED_FAMILY_SIZE",
    "SCOPE_IDS",
    "SENSITIVITY_THRESHOLDS_CM_S",
    "SUSTAINED_MIN_RUN_LENGTHS",
    "assemble_declared_family",
    "compute_aligned_grid_metrics",
    "compute_regression_metrics",
    "compute_skill_ratio",
    "hash_target_sequences",
    "load_prediction_cache",
    "save_prediction_cache",
    "validate_target_binding",
]

_EMPTY_REASON = (
    "no eligible frames in this partition on the scored scope; "
    "statistics are unavailable (not zero)"
)
_ZERO_VAR_REASON = "zero_target_variance"


def _as_finite_1d_float(arr: Any, name: str) -> np.ndarray:
    """Coerce to 1-D float64 array, rejecting non-1-D or non-finite entries."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite entries")
    return a


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Compute standard regression metrics (SSE, MSE, RMSE, MAE, R-squared).

    Args:
        y_true: Ground truth 1-D array.
        y_pred: Predicted 1-D array, same shape.

    Returns:
        Dictionary with n_frames, sse, mse, rmse, mae, r2, reason, r2_reason.
    """
    yt = _as_finite_1d_float(y_true, "y_true")
    yp = _as_finite_1d_float(y_pred, "y_pred")
    assert yt.shape == yp.shape, f"Shape mismatch: {yt.shape} != {yp.shape}"
    n = int(yt.size)
    if n == 0:
        return {
            "n_frames": 0,
            "sse": None,
            "mse": None,
            "rmse": None,
            "mae": None,
            "r2": None,
            "reason": _EMPTY_REASON,
            "r2_reason": _EMPTY_REASON,
        }
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            err = yp - yt
            sse = float(np.sum(err ** 2))
            mse = sse / n
            var = float(np.var(yt))
            mae = float(np.mean(np.abs(err)))
        except FloatingPointError as e:
            raise FloatingPointError(
                f"Numerical overflow in regression metrics: {e}"
            ) from e

    r2: Optional[float]
    r2_reason: Optional[str]
    if var > 0.0 and math.isfinite(var):
        with np.errstate(over="ignore", invalid="ignore"):
            raw_r2 = 1.0 - mse / var
        if math.isfinite(raw_r2):
            r2 = float(raw_r2)
            r2_reason = None
        else:
            r2 = None
            r2_reason = "numerical_instability"
    else:
        r2 = None
        r2_reason = _ZERO_VAR_REASON

    return {
        "n_frames": n,
        "sse": sse,
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": mae,
        "r2": r2,
        "reason": None,
        "r2_reason": r2_reason,
    }


def compute_skill_ratio(
    mse_model: Optional[float], mse_base: Optional[float]
) -> Dict[str, Any]:
    """Compute 1.0 - mse_model / mse_base with explicit unavailable reason.

    Returns:
        Dict with keys 'value' (Optional[float]) and 'reason' (Optional[str]).
    """
    if mse_model is None:
        return {"value": None, "reason": "missing_model_mse"}
    if mse_base is None:
        return {"value": None, "reason": "missing_comparator_mse"}
    if not math.isfinite(mse_model) or not math.isfinite(mse_base):
        return {"value": None, "reason": "non_finite_mse"}
    if mse_base == 0.0:
        return {"value": None, "reason": "zero_comparator_mse"}
    if mse_base < 0.0:
        return {"value": None, "reason": "negative_comparator_mse"}

    with np.errstate(over="ignore", invalid="ignore"):
        ratio = 1.0 - mse_model / mse_base
    if not math.isfinite(ratio):
        return {"value": None, "reason": "numerical_instability"}

    return {"value": float(ratio), "reason": None}


def hash_target_sequences(
    y_true_seqs: Sequence[np.ndarray],
) -> Dict[str, Any]:
    """Calculate cryptographic hashes of full and aligned target sequences.

    Args:
        y_true_seqs: Sequence of 1-D target arrays in canonical order.

    Returns:
        Dict with n_trials, n_full_frames, n_eligible_frames,
        full_target_sha256, and eligible_target_sha256.
    """
    true_arrays = [
        _as_finite_1d_float(t, f"y_true[{i}]")
        for i, t in enumerate(y_true_seqs)
    ]
    full_chunks: List[np.ndarray] = []
    elig_chunks: List[np.ndarray] = []
    for t in true_arrays:
        full_chunks.append(t)
        if t.size >= 2:
            elig_chunks.append(t[1:])

    full_cat = (
        np.concatenate(full_chunks)
        if full_chunks
        else np.empty(0, dtype=np.float64)
    )
    elig_cat = (
        np.concatenate(elig_chunks)
        if elig_chunks
        else np.empty(0, dtype=np.float64)
    )

    h_full = hashlib.sha256(full_cat.astype("<f8").tobytes()).hexdigest()
    h_elig = hashlib.sha256(elig_cat.astype("<f8").tobytes()).hexdigest()
    return {
        "n_trials": len(true_arrays),
        "n_full_frames": int(full_cat.size),
        "n_eligible_frames": int(elig_cat.size),
        "full_target_sha256": h_full,
        "eligible_target_sha256": h_elig,
    }


def validate_target_binding(
    y_true_seqs: Sequence[np.ndarray],
    expected: Mapping[str, Any],
) -> None:
    """Validate target sequences against expected counts and hashes. Fail closed."""
    if not expected:
        raise ValueError("expected target binding mapping must not be empty")

    mandatory_keys = (
        "n_trials",
        "n_full_frames",
        "n_eligible_frames",
        "full_target_sha256",
        "eligible_target_sha256",
    )
    for key in mandatory_keys:
        if key not in expected:
            raise ValueError(f"Missing mandatory binding key: {key}")

    hashes = hash_target_sequences(y_true_seqs)
    for key in mandatory_keys:
        exp_val = expected[key]
        act_val = hashes[key]
        if act_val != exp_val:
            raise ValueError(
                f"Target binding mismatch for {key}: "
                f"expected {exp_val}, got {act_val}"
            )


def compute_aligned_grid_metrics(
    y_true_seqs: Sequence[np.ndarray],
    y_pred_seqs: Sequence[np.ndarray],
    *,
    trial_ids: Sequence[str],
    prefix_ids: Sequence[str],
    thresholds: Sequence[float] = SENSITIVITY_THRESHOLDS_CM_S,
    min_runs: Sequence[int] = SUSTAINED_MIN_RUN_LENGTHS,
) -> Dict[str, Any]:
    """Compute pooled and per-trial metrics over full-frame and 13 aligned scopes.

    Sustained run masks are computed on full trials before slicing t >= 1.
    Persistence is strictly trial-local: y_true[t-1] within each trial.
    Full-frame and aligned partitions have separate counts and metrics.

    Args:
        y_true_seqs: Per-trial ground-truth targets.
        y_pred_seqs: Per-trial model predictions.
        trial_ids: Per-trial unique non-empty string IDs.
        prefix_ids: Per-trial recording prefix string IDs.
        thresholds: Velocity threshold magnitudes in cm/s.
        min_runs: Minimum consecutive frames for sustained runs.

    Returns:
        Structured dictionary containing pooled metrics, per-cell sweeps,
        and per-trial vectors for each scope.
    """
    n_trials = len(y_true_seqs)
    if len(y_pred_seqs) != n_trials:
        raise ValueError(
            f"Trial count mismatch: {n_trials} true vs {len(y_pred_seqs)} pred"
        )
    if len(trial_ids) != n_trials or len(prefix_ids) != n_trials:
        raise ValueError("trial_ids and prefix_ids must match trial count")

    if any(not tid for tid in trial_ids):
        raise ValueError("trial_ids must contain non-empty strings")
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError(
            f"trial_ids must be unique, got {len(set(trial_ids))} "
            f"distinct for {n_trials} trials"
        )

    true_arrays = [
        _as_finite_1d_float(t, f"y_true[{i}]")
        for i, t in enumerate(y_true_seqs)
    ]
    pred_arrays = [
        _as_finite_1d_float(p, f"y_pred[{i}]")
        for i, p in enumerate(y_pred_seqs)
    ]
    for i, (t, p) in enumerate(zip(true_arrays, pred_arrays)):
        if t.shape != p.shape:
            raise ValueError(
                f"Trial {i} shape mismatch: true {t.shape} != pred {p.shape}"
            )

    # 1. Full-frame pooled metrics (t >= 0)
    full_true = (
        np.concatenate(true_arrays)
        if true_arrays
        else np.empty(0, dtype=np.float64)
    )
    full_pred = (
        np.concatenate(pred_arrays)
        if pred_arrays
        else np.empty(0, dtype=np.float64)
    )
    full_zero = np.zeros_like(full_true)
    raw_full = {
        "model": compute_regression_metrics(full_true, full_pred),
        "zero": compute_regression_metrics(full_true, full_zero),
    }
    raw_full["skill_vs_zero"] = compute_skill_ratio(
        raw_full["model"]["mse"], raw_full["zero"]["mse"]
    )

    # 2. Aligned overall (t >= 1)
    elig_true_list = [t[1:] for t in true_arrays if t.size >= 2]
    elig_pred_list = [p[1:] for p in pred_arrays if p.size >= 2]
    elig_prev_list = [t[:-1] for t in true_arrays if t.size >= 2]

    elig_true = (
        np.concatenate(elig_true_list)
        if elig_true_list
        else np.empty(0, dtype=np.float64)
    )
    elig_pred = (
        np.concatenate(elig_pred_list)
        if elig_pred_list
        else np.empty(0, dtype=np.float64)
    )
    elig_prev = (
        np.concatenate(elig_prev_list)
        if elig_prev_list
        else np.empty(0, dtype=np.float64)
    )
    elig_zero = np.zeros_like(elig_true)

    aligned_overall = {
        "model": compute_regression_metrics(elig_true, elig_pred),
        "persistence": compute_regression_metrics(elig_true, elig_prev),
        "zero": compute_regression_metrics(elig_true, elig_zero),
    }
    aligned_overall["skill_vs_persistence"] = compute_skill_ratio(
        aligned_overall["model"]["mse"], aligned_overall["persistence"]["mse"]
    )
    aligned_overall["skill_vs_zero"] = compute_skill_ratio(
        aligned_overall["model"]["mse"], aligned_overall["zero"]["mse"]
    )

    # 3. 12-cell sustained sweep + per-trial vectors
    scope_data: Dict[str, Dict[str, Any]] = {}
    scope_data[ALIGNED_OVERALL_SCOPE] = {
        "trial_ids": [],
        "prefix_ids": [],
        "frame_counts": [],
        "model_mse": [],
        "persistence_mse": [],
        "zero_mse": [],
    }
    for i, (t, p) in enumerate(zip(true_arrays, pred_arrays)):
        if t.size >= 2:
            yt_elig = t[1:]
            yp_elig = p[1:]
            yprev_elig = t[:-1]
            n_f = int(yt_elig.size)
            scope_data[ALIGNED_OVERALL_SCOPE]["trial_ids"].append(trial_ids[i])
            scope_data[ALIGNED_OVERALL_SCOPE]["prefix_ids"].append(prefix_ids[i])
            scope_data[ALIGNED_OVERALL_SCOPE]["frame_counts"].append(n_f)
            scope_data[ALIGNED_OVERALL_SCOPE]["model_mse"].append(
                float(np.mean((yp_elig - yt_elig) ** 2))
            )
            scope_data[ALIGNED_OVERALL_SCOPE]["persistence_mse"].append(
                float(np.mean((yprev_elig - yt_elig) ** 2))
            )
            scope_data[ALIGNED_OVERALL_SCOPE]["zero_mse"].append(
                float(np.mean(yt_elig ** 2))
            )
    scope_data[ALIGNED_OVERALL_SCOPE]["per_trial_partitions"] = {
        "trial_ids": list(trial_ids),
        "prefix_ids": list(prefix_ids),
        "full_frame": {
            "total_frames": [int(t.size) for t in true_arrays],
        },
        "aligned": {
            "total_frames": [int(max(0, t.size - 1)) for t in true_arrays],
        },
    }

    cells_summary: List[Dict[str, Any]] = []

    for th in thresholds:
        for mr in min_runs:
            cell_key = f"band{int(th)}_run{mr}"
            scope_data[cell_key] = {
                "trial_ids": [],
                "prefix_ids": [],
                "frame_counts": [],
                "model_mse": [],
                "persistence_mse": [],
                "zero_mse": [],
            }

            # Full-frame accumulators (t >= 0)
            full_esc_true: List[np.ndarray] = []
            full_esc_pred: List[np.ndarray] = []
            full_rest_true: List[np.ndarray] = []
            full_rest_pred: List[np.ndarray] = []
            full_esc_trial_set: set[str] = set()
            full_rest_trial_set: set[str] = set()
            full_esc_prefix_set: set[str] = set()
            full_rest_prefix_set: set[str] = set()
            per_trial_full_escape_frames: List[int] = [0] * len(true_arrays)
            per_trial_full_rest_frames: List[int] = [0] * len(true_arrays)

            # Aligned accumulators (t >= 1)
            elig_esc_true: List[np.ndarray] = []
            elig_esc_pred: List[np.ndarray] = []
            elig_esc_prev: List[np.ndarray] = []
            elig_rest_true: List[np.ndarray] = []
            elig_rest_pred: List[np.ndarray] = []
            elig_rest_prev: List[np.ndarray] = []
            aligned_esc_trial_set: set[str] = set()
            aligned_rest_trial_set: set[str] = set()
            aligned_esc_prefix_set: set[str] = set()
            aligned_rest_prefix_set: set[str] = set()
            per_trial_aligned_escape_frames: List[int] = [0] * len(true_arrays)
            per_trial_aligned_rest_frames: List[int] = [0] * len(true_arrays)

            for i, (yt, yp) in enumerate(zip(true_arrays, pred_arrays)):
                # Mask computed on full trial BEFORE slicing t >= 1
                full_mask = np.asarray(
                    _sustained_run(np.abs(yt) >= float(th), min_run=int(mr)),
                    dtype=bool,
                )
                assert full_mask.shape == yt.shape, "mask shape mismatch"

                # Full-frame accumulation
                n_full_esc = int(np.sum(full_mask))
                n_full_rest = int(yt.size) - n_full_esc
                per_trial_full_escape_frames[i] = n_full_esc
                per_trial_full_rest_frames[i] = n_full_rest
                if n_full_esc > 0:
                    full_esc_trial_set.add(trial_ids[i])
                    full_esc_prefix_set.add(prefix_ids[i])
                    full_esc_true.append(yt[full_mask])
                    full_esc_pred.append(yp[full_mask])

                full_rest_mask = ~full_mask
                if np.any(full_rest_mask):
                    full_rest_trial_set.add(trial_ids[i])
                    full_rest_prefix_set.add(prefix_ids[i])
                    full_rest_true.append(yt[full_rest_mask])
                    full_rest_pred.append(yp[full_rest_mask])

                # Aligned accumulation (requires t >= 2)
                if yt.size < 2:
                    continue

                mask_elig = full_mask[1:]
                yt_e = yt[1:]
                yp_e = yp[1:]
                yprev_e = yt[:-1]

                n_esc_trial = int(np.sum(mask_elig))
                n_rest_trial = int(yt_e.size) - n_esc_trial
                per_trial_aligned_escape_frames[i] = n_esc_trial
                per_trial_aligned_rest_frames[i] = n_rest_trial
                if n_esc_trial > 0:
                    aligned_esc_trial_set.add(trial_ids[i])
                    aligned_esc_prefix_set.add(prefix_ids[i])
                    elig_esc_true.append(yt_e[mask_elig])
                    elig_esc_pred.append(yp_e[mask_elig])
                    elig_esc_prev.append(yprev_e[mask_elig])
                    scope_data[cell_key]["trial_ids"].append(trial_ids[i])
                    scope_data[cell_key]["prefix_ids"].append(prefix_ids[i])
                    scope_data[cell_key]["frame_counts"].append(n_esc_trial)
                    scope_data[cell_key]["model_mse"].append(
                        float(np.mean((yp_e[mask_elig] - yt_e[mask_elig]) ** 2))
                    )
                    scope_data[cell_key]["persistence_mse"].append(
                        float(
                            np.mean((yprev_e[mask_elig] - yt_e[mask_elig]) ** 2)
                        )
                    )
                    scope_data[cell_key]["zero_mse"].append(
                        float(np.mean(yt_e[mask_elig] ** 2))
                    )

                mask_rest = ~mask_elig
                if np.any(mask_rest):
                    aligned_rest_trial_set.add(trial_ids[i])
                    aligned_rest_prefix_set.add(prefix_ids[i])
                    elig_rest_true.append(yt_e[mask_rest])
                    elig_rest_pred.append(yp_e[mask_rest])
                    elig_rest_prev.append(yprev_e[mask_rest])

            # Concatenate full-frame partitions
            f_esc_t = (
                np.concatenate(full_esc_true)
                if full_esc_true
                else np.empty(0, dtype=np.float64)
            )
            f_esc_p = (
                np.concatenate(full_esc_pred)
                if full_esc_pred
                else np.empty(0, dtype=np.float64)
            )
            f_esc_z = np.zeros_like(f_esc_t)
            f_rest_t = (
                np.concatenate(full_rest_true)
                if full_rest_true
                else np.empty(0, dtype=np.float64)
            )
            f_rest_p = (
                np.concatenate(full_rest_pred)
                if full_rest_pred
                else np.empty(0, dtype=np.float64)
            )
            f_rest_z = np.zeros_like(f_rest_t)

            # Concatenate aligned partitions
            e_esc_t = (
                np.concatenate(elig_esc_true)
                if elig_esc_true
                else np.empty(0, dtype=np.float64)
            )
            e_esc_p = (
                np.concatenate(elig_esc_pred)
                if elig_esc_pred
                else np.empty(0, dtype=np.float64)
            )
            e_esc_prev = (
                np.concatenate(elig_esc_prev)
                if elig_esc_prev
                else np.empty(0, dtype=np.float64)
            )
            e_esc_z = np.zeros_like(e_esc_t)

            e_rest_t = (
                np.concatenate(elig_rest_true)
                if elig_rest_true
                else np.empty(0, dtype=np.float64)
            )
            e_rest_p = (
                np.concatenate(elig_rest_pred)
                if elig_rest_pred
                else np.empty(0, dtype=np.float64)
            )
            e_rest_prev = (
                np.concatenate(elig_rest_prev)
                if elig_rest_prev
                else np.empty(0, dtype=np.float64)
            )
            e_rest_z = np.zeros_like(e_rest_t)

            # Reconcile per-trial counts with aggregate totals
            assert sum(per_trial_full_escape_frames) == int(f_esc_t.size)
            assert sum(per_trial_full_rest_frames) == int(f_rest_t.size)
            assert sum(per_trial_aligned_escape_frames) == int(e_esc_t.size)
            assert sum(per_trial_aligned_rest_frames) == int(e_rest_t.size)

            # Store per-trial partition frame counts in scope_data
            scope_data[cell_key]["per_trial_partitions"] = {
                "trial_ids": list(trial_ids),
                "prefix_ids": list(prefix_ids),
                "full_frame": {
                    "escape_frames": per_trial_full_escape_frames,
                    "rest_frames": per_trial_full_rest_frames,
                },
                "aligned": {
                    "escape_frames": per_trial_aligned_escape_frames,
                    "rest_frames": per_trial_aligned_rest_frames,
                },
            }

            cell_summary = {
                "scope_id": cell_key,
                "threshold_cm_s": float(th),
                "min_run": int(mr),
                "counts": {
                    "full_frame": {
                        "n_trials_with_escape": len(full_esc_trial_set),
                        "n_trials_with_rest": len(full_rest_trial_set),
                        "n_prefixes_with_escape": len(full_esc_prefix_set),
                        "n_prefixes_with_rest": len(full_rest_prefix_set),
                        "n_escape_frames": int(f_esc_t.size),
                        "n_rest_frames": int(f_rest_t.size),
                        "total_frames": int(f_esc_t.size + f_rest_t.size),
                        "per_trial_escape_frames": per_trial_full_escape_frames,
                        "per_trial_rest_frames": per_trial_full_rest_frames,
                    },
                    "aligned": {
                        "n_trials_with_escape": len(aligned_esc_trial_set),
                        "n_trials_with_rest": len(aligned_rest_trial_set),
                        "n_prefixes_with_escape": len(aligned_esc_prefix_set),
                        "n_prefixes_with_rest": len(aligned_rest_prefix_set),
                        "n_escape_frames": int(e_esc_t.size),
                        "n_rest_frames": int(e_rest_t.size),
                        "total_frames": int(e_esc_t.size + e_rest_t.size),
                        "per_trial_escape_frames": per_trial_aligned_escape_frames,
                        "per_trial_rest_frames": per_trial_aligned_rest_frames,
                    },
                },
                "per_trial_partitions": {
                    "trial_ids": list(trial_ids),
                    "prefix_ids": list(prefix_ids),
                    "full_frame": {
                        "escape_frames": per_trial_full_escape_frames,
                        "rest_frames": per_trial_full_rest_frames,
                    },
                    "aligned": {
                        "escape_frames": per_trial_aligned_escape_frames,
                        "rest_frames": per_trial_aligned_rest_frames,
                    },
                },
                "full_frame": {
                    "escape": {
                        "model": compute_regression_metrics(f_esc_t, f_esc_p),
                        "zero": compute_regression_metrics(f_esc_t, f_esc_z),
                    },
                    "rest": {
                        "model": compute_regression_metrics(f_rest_t, f_rest_p),
                        "zero": compute_regression_metrics(f_rest_t, f_rest_z),
                    },
                },
                "aligned": {
                    "escape": {
                        "model": compute_regression_metrics(e_esc_t, e_esc_p),
                        "persistence": compute_regression_metrics(
                            e_esc_t, e_esc_prev
                        ),
                        "zero": compute_regression_metrics(e_esc_t, e_esc_z),
                    },
                    "rest": {
                        "model": compute_regression_metrics(e_rest_t, e_rest_p),
                        "persistence": compute_regression_metrics(
                            e_rest_t, e_rest_prev
                        ),
                        "zero": compute_regression_metrics(e_rest_t, e_rest_z),
                    },
                },
            }
            # Attach skill metrics
            cell_summary["full_frame"]["escape"]["skill_vs_zero"] = (
                compute_skill_ratio(
                    cell_summary["full_frame"]["escape"]["model"]["mse"],
                    cell_summary["full_frame"]["escape"]["zero"]["mse"],
                )
            )
            cell_summary["full_frame"]["rest"]["skill_vs_zero"] = (
                compute_skill_ratio(
                    cell_summary["full_frame"]["rest"]["model"]["mse"],
                    cell_summary["full_frame"]["rest"]["zero"]["mse"],
                )
            )
            cell_summary["aligned"]["escape"]["skill_vs_persistence"] = (
                compute_skill_ratio(
                    cell_summary["aligned"]["escape"]["model"]["mse"],
                    cell_summary["aligned"]["escape"]["persistence"]["mse"],
                )
            )
            cell_summary["aligned"]["escape"]["skill_vs_zero"] = (
                compute_skill_ratio(
                    cell_summary["aligned"]["escape"]["model"]["mse"],
                    cell_summary["aligned"]["escape"]["zero"]["mse"],
                )
            )
            cell_summary["aligned"]["rest"]["skill_vs_persistence"] = (
                compute_skill_ratio(
                    cell_summary["aligned"]["rest"]["model"]["mse"],
                    cell_summary["aligned"]["rest"]["persistence"]["mse"],
                )
            )
            cell_summary["aligned"]["rest"]["skill_vs_zero"] = (
                compute_skill_ratio(
                    cell_summary["aligned"]["rest"]["model"]["mse"],
                    cell_summary["aligned"]["rest"]["zero"]["mse"],
                )
            )

            cells_summary.append(cell_summary)

    return {
        "raw_full_frame": raw_full,
        "aligned_overall": aligned_overall,
        "sustained_cells": cells_summary,
        "scope_data": scope_data,
    }


def assemble_declared_family(
    candidate_scopes: Mapping[str, Mapping[str, Any]],
    comparator_scopes: Mapping[str, Mapping[str, Any]],
    *,
    exchangeability_asserted: bool = False,
) -> Dict[str, Any]:
    """Assemble the frozen 78-slot inferential family and compute statistics.

    Args:
        candidate_scopes: Mapping of candidate_id -> scope_data.
        comparator_scopes: Mapping of comparator_id -> scope_data.
            Keys must include "head_only_baseline", "persistence", "zero".
        exchangeability_asserted: Must be a genuine bool explicitly passed.

    Returns:
        Dict with slot results, Holm-Bonferroni correction, and metadata.
    """
    if type(exchangeability_asserted) is not bool:
        raise ValueError(
            "exchangeability_asserted must be a genuine bool, got "
            f"{type(exchangeability_asserted).__name__}"
        )

    for cmp_id in COMPARATOR_IDS:
        if cmp_id not in comparator_scopes:
            raise ValueError(f"Missing declared comparator: {cmp_id}")

    slots = declared_family_slots()
    p_values: Dict[str, Optional[float]] = {}
    family_slots_data: Dict[str, Any] = {}
    present_candidates = [cid for cid in CANDIDATE_IDS if cid in candidate_scopes]

    for slot_key in slots:
        cand_id, comp_id, scope_id = slot_key.split("|")
        if cand_id not in candidate_scopes:
            p_values[slot_key] = None
            family_slots_data[slot_key] = {
                "available": False,
                "reason": "missing_candidate_arm",
                "counts": None,
                "mean_delta": None,
                "cohens_dz": {"value": None, "reason": "missing_candidate_arm"},
                "cluster_statistic": None,
                "sign_flip": {
                    "p_value": None,
                    "reason": "missing_candidate_arm",
                    "method": "unavailable",
                    "n_sign_patterns": 0,
                },
            }
            continue

        cand_data = candidate_scopes[cand_id][scope_id]
        comp_data = comparator_scopes[comp_id][scope_id]

        c_trials = cand_data["trial_ids"]
        cmp_trials = comp_data["trial_ids"]
        c_prefixes = cand_data["prefix_ids"]
        cmp_prefixes = comp_data["prefix_ids"]
        c_counts = cand_data["frame_counts"]
        cmp_counts = comp_data["frame_counts"]

        # Validate trial, prefix, and frame alignment on eligible scope
        if c_trials != cmp_trials:
            raise ValueError(
                f"Slot {slot_key} trial ID mismatch: "
                f"{len(c_trials)} != {len(cmp_trials)}"
            )
        if c_prefixes != cmp_prefixes:
            raise ValueError(
                f"Slot {slot_key} prefix ID mismatch: "
                f"{len(c_prefixes)} != {len(cmp_prefixes)}"
            )
        if c_counts != cmp_counts:
            raise ValueError(
                f"Slot {slot_key} frame counts mismatch: "
                f"{len(c_counts)} != {len(cmp_counts)}"
            )

        n_trials = len(c_trials)
        if n_trials < 2 or sum(c_counts) == 0:
            # Slot is unavailable (not enough trials or zero frames)
            p_values[slot_key] = None
            family_slots_data[slot_key] = {
                "available": False,
                "reason": (
                    "fewer_than_two_trials" if n_trials < 2 else "zero_eligible_frames"
                ),
                "counts": {
                    "trials": n_trials,
                    "frames": sum(c_counts),
                    "prefixes": len(set(c_prefixes)),
                },
                "mean_delta": None,
                "cohens_dz": {"value": None, "reason": "slot_unavailable"},
                "cluster_statistic": None,
                "sign_flip": {
                    "p_value": None,
                    "reason": "slot_unavailable",
                    "method": "unavailable",
                    "n_sign_patterns": 0,
                },
            }
            continue

        comp_result = paired_mse_comparison(
            candidate_mse=cand_data["model_mse"],
            comparator_mse=comp_data["mse"]
            if "mse" in comp_data
            else (
                comp_data["persistence_mse"]
                if comp_id == "persistence"
                else (
                    comp_data["zero_mse"]
                    if comp_id == "zero"
                    else comp_data["model_mse"]
                )
            ),
            trial_ids=c_trials,
            prefix_ids=c_prefixes,
            frame_counts=c_counts,
            exchangeability_asserted=exchangeability_asserted,
        )
        p_val = comp_result["sign_flip"]["p_value"]
        p_values[slot_key] = p_val
        family_slots_data[slot_key] = {
            "available": True,
            "reason": None,
            **comp_result,
        }

    holm_result = holm_correct_declared_family(p_values)

    return {
        "family_size": DECLARED_FAMILY_SIZE,
        "both_arms_present": len(present_candidates) == len(CANDIDATE_IDS),
        "present_candidates": present_candidates,
        "exchangeability_asserted": exchangeability_asserted,
        "holm_bonferroni": holm_result,
        "slots": family_slots_data,
    }


def _cache_paths(base_path: str | Path) -> Tuple[Path, Path]:
    """Derive NPZ and JSON paths safely without suffix truncation."""
    s = str(base_path)
    if s.endswith(".npz"):
        npz_p = Path(s)
        json_p = Path(s[:-4] + ".json")
    elif s.endswith(".json"):
        npz_p = Path(s[:-5] + ".npz")
        json_p = Path(s)
    else:
        npz_p = Path(f"{s}.npz")
        json_p = Path(f"{s}.json")
    return npz_p, json_p


def save_prediction_cache(
    path: str | Path,
    y_pred_seqs: Sequence[np.ndarray],
    metadata: Mapping[str, Any],
    *,
    allow_overwrite: bool = False,
) -> None:
    """Save flat contiguous predictions and metadata without object arrays.

    Uses ``np.savez_compressed`` with numeric arrays.
    Refuses overwriting existing cache unless ``allow_overwrite=True``.
    """
    npz_path, json_path = _cache_paths(path)
    if not allow_overwrite:
        if npz_path.exists():
            raise FileExistsError(f"Prediction cache NPZ exists: {npz_path}")
        if json_path.exists():
            raise FileExistsError(f"Prediction cache JSON exists: {json_path}")

    pred_arrays = [
        _as_finite_1d_float(arr, f"y_pred[{i}]")
        for i, arr in enumerate(y_pred_seqs)
    ]
    lengths = np.array([int(arr.size) for arr in pred_arrays], dtype=np.int64)
    flat_preds = (
        np.concatenate(pred_arrays)
        if pred_arrays
        else np.empty(0, dtype=np.float64)
    )

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        flat_preds=flat_preds.astype(np.float64),
        lengths=lengths,
    )
    json_path.write_text(
        json.dumps(dict(metadata), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_prediction_cache(
    path: str | Path,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Load flat contiguous predictions and metadata with allow_pickle=False.

    Decodes lengths using Python integers to prevent uint64 arithmetic overflow.
    Validates 1-D finite numeric arrays, bounds each segment, and verifies
    exact reconstruction against flat_preds.size.
    """
    npz_path, json_path = _cache_paths(path)

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing cache NPZ file: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing cache JSON metadata file: {json_path}")

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Cache metadata must be a JSON object")

    with np.load(npz_path, allow_pickle=False) as data:
        files = list(data.files)
        if set(files) != {"flat_preds", "lengths"}:
            raise ValueError(
                f"NPZ must contain exactly ['flat_preds', 'lengths'], got {files}"
            )
        flat_preds = data["flat_preds"]
        lengths = data["lengths"]

    if flat_preds.dtype.kind not in ("f", "i"):
        raise ValueError(
            f"flat_preds must be numeric float/int, got dtype {flat_preds.dtype}"
        )
    if flat_preds.ndim != 1:
        raise ValueError(f"flat_preds must be 1-D, got shape {flat_preds.shape}")
    if not np.all(np.isfinite(flat_preds)):
        raise ValueError("flat_preds contains non-finite entries")

    if lengths.ndim != 1:
        raise ValueError(f"lengths must be 1-D, got shape {lengths.shape}")
    if lengths.dtype.kind not in ("i", "u") or lengths.dtype == bool:
        raise ValueError(
            f"lengths must be non-boolean integer dtype (i or u), got {lengths.dtype}"
        )

    flat_n = int(flat_preds.size)
    py_lengths: List[int] = []
    total_len = 0
    for i, l in enumerate(lengths):
        try:
            val = int(l)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Length at index {i} is not a valid integer: {l}"
            ) from exc
        if val < 0:
            raise ValueError(
                f"lengths contains negative value at index {i}: {val}"
            )
        if val > flat_n:
            raise ValueError(
                f"length at index {i} ({val}) exceeds flat_preds size ({flat_n})"
            )
        total_len += val
        py_lengths.append(val)

    if total_len != flat_n:
        raise ValueError(
            f"Sum of lengths ({total_len}) does not match flat_preds size "
            f"({flat_n})"
        )

    seqs: List[np.ndarray] = []
    idx = 0
    for l in py_lengths:
        chunk = flat_preds[idx : idx + l]
        if chunk.size != l:
            raise ValueError(
                f"Slice size mismatch: expected {l}, got {chunk.size}"
            )
        seqs.append(chunk.astype(np.float64, copy=True))
        idx += l

    return seqs, metadata
