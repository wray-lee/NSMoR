"""
NSMoR In-Silico Lesion (Virtual Ablation) Experiment — Phase 7.

Runs a deterministic ablation experiment comparing the behavioral
(kinematic) output of the Intact model versus Lesioned models:
  - Condition 1 (Intact): Natural routing
  - Condition 2 (LIF-Lesioned): Forces all routing through GRU pathway
  - Condition 3 (GRU-Lesioned): Forces all routing through LIF pathway

Generates a Lancet/Cell-quality publication figure demonstrating
behavioral collapse when specific pathways are lesioned.

Output: ``results/ablation_kinematics.png`` at 300 DPI.

Usage
-----
CLI::

    python scripts/simulate_lesion.py --checkpoint runs/default/best_model.pth
    python scripts/simulate_lesion.py --checkpoint runs/default/best_model.pth --dataset data/processed/nsmor_dataset.pt --target_class 0
"""

from __future__ import annotations

import argparse
import sys
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Direct CLI execution must use this checkout, including its prior/lineage helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from nsmor.nsmor_dataloader import NSMoRDataset
from nsmor.dataloader_factory import create_optimized_dataloader
from nsmor.checkpoint import load_checkpoint
from nsmor.config import DEFAULT_FEATURE, Label
from nsmor.model_nsmor_core import NSMoRCore
from nsmor.analysis.analysis_priors import (
    describe_analysis_population, load_analysis_priors, population_for_output,
)
from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint
from nsmor.analysis.prediction_units import resolve_dt_ms
from nsmor.analysis.prediction_units import load_model_from_checkpoint as _shared_load_model
from nsmor.model_utils import validate_dataset_provenance

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Lancet / Cell Publication Style Constants
# ═══════════════════════════════════════════════════════════════

# ── High-contrast color mapping ──
GROUND_TRUTH_COLOR: str = "#C92A2A"   # Lancet Crimson Red
PREDICTED_COLOR: str = "#495057"       # Strong Slate Gray

# ── Lesion condition names ──
CONDITION_NAMES: Dict[str, str] = {
    "intact": "Intact Model",
    "lif_lesioned": "LIF-Lesioned (g_lif=0)",
    "gru_lesioned": "GRU-Lesioned (g_gru=0)",
}

# ── Lesion gate overrides ──
LESION_OVERRIDES: Dict[str, Optional[Dict[str, float]]] = {
    "intact": None,
    "lif_lesioned": {"g_lif": 0.0, "g_gru": 1.0},
    "gru_lesioned": {"g_lif": 1.0, "g_gru": 0.0},
}

# ── Typography ─────────────────────────────────────────────────
FONT_FAMILY: str = "Arial"
FONT_SIZE_AXIS_TITLE: int = 12
FONT_SIZE_TICK: int = 10
FONT_SIZE_LEGEND: int = 9
FONT_SIZE_PANEL_LABEL: int = 14

# ── Figure properties ─────────────────────────────────────────
DPI: int = 300
FIG_WIDTH_INCHES: float = 10.0
FIG_HEIGHT_INCHES: float = 12.0
BACKGROUND_COLOR: str = "#FFFFFF"
AXIS_COLOR: str = "#212529"  # Solid dark charcoal

# ── Plot properties ───────────────────────────────────────────
LINE_WIDTH_GT: float = 2.5
LINE_WIDTH_PRED: float = 2.0
DASHED_LINESTYLE: str = "--"


# ═══════════════════════════════════════════════════════════════
# 1.  Model Loading
# ═══════════════════════════════════════════════════════════════

def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> NSMoRCore:
    """Load trained NSMoRCore from checkpoint.

    Delegates to the shared :func:`nsmor.analysis.prediction_units.load_model_from_checkpoint`
    which guarantees all biophysical parameters are restored.
    """
    return _shared_load_model(checkpoint_path, device)


# ═══════════════════════════════════════════════════════════════
# 2.  Dataset Loading
# ═══════════════════════════════════════════════════════════════

def load_dataset(
    dataset_path: Path,
    batch_size: int = 32,
    max_seq_len: Optional[int] = 2400,
    pre_anchor_frames: int = 1200,
    nested_prior_artifact: Optional[Path] = None,
    qc_sealed_nested_prior_sha256: Optional[str] = None,
    checkpoint_model: Optional[NSMoRCore] = None,
    trusted_historical_checkpoint_sha256: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
) -> Tuple[torch.utils.data.DataLoader, np.ndarray, List[int]]:
    """
    Load the preprocessed dataset and create a DataLoader.

    Args:
        dataset_path: Path to ``nsmor_dataset.pt``.
        batch_size: Batch size for the DataLoader.
        max_seq_len: Maximum sequence length for cropping.
        pre_anchor_frames: Baseline frames before anchor.

    Returns:
        ``(dataloader, labels, cropped_lengths_list)`` tuple. The dataset carries
        ``cropped_reference_frames`` and ``analysis_reference_trials`` in loader
        order, without changing the tuple or the batches.

    Raises:
        FileNotFoundError: If dataset file does not exist.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info("Loading dataset from %s", dataset_path)
    dataset, loaded_source_fingerprint = load_dataset_with_fingerprint(
        dataset_path, restore_provenance=False,
        expected_dt_ms=(
            resolve_dt_ms(checkpoint_model) if checkpoint_model is not None else None
        ),
    )

    # Round-2 CRITICAL-A: refuse pre-2.0 datasets (leaked priors, np.max labels)
    validate_dataset_provenance(dataset, Path(dataset_path))

    X_seqs = dataset["X_seqs"]
    Y_seqs = dataset["Y_seqs"]
    mcmc_priors, val_indices = load_analysis_priors(
        dataset, dataset_path, nested_prior_artifact, checkpoint_model, qc_sealed_nested_prior_sha256,
        loaded_source_fingerprint=loaded_source_fingerprint,
        trusted_historical_checkpoint_sha256=trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
    )
    labels = dataset["labels"]
    lengths = dataset["lengths"]

    n_total = len(X_seqs)
    logger.info("Loaded %d sequences.", n_total)
    session_ids = dataset.get("session_ids")
    trial_ids = dataset.get("trial_ids")
    if trial_ids is None and ("trial_ids" in dataset or "stimulus_conditions" in dataset):
        raise ValueError("modern dataset requires source trial_ids aligned with all trials")
    if session_ids is not None and (
        not isinstance(session_ids, (list, tuple, np.ndarray))
        or isinstance(session_ids, np.ndarray) and session_ids.ndim != 1
        or len(session_ids) != n_total
    ):
        raise ValueError("session_ids must be a one-dimensional sequence aligned with all trials")
    if trial_ids is not None:
        if (not isinstance(trial_ids, (list, tuple, np.ndarray))
                or isinstance(trial_ids, np.ndarray) and trial_ids.ndim != 1
                or len(trial_ids) != n_total
                or any(isinstance(value, (bool, np.bool_))
                       or not isinstance(value, (int, np.integer)) for value in trial_ids)):
            raise ValueError("source trial_ids must be one-dimensional, aligned integer IDs")
        if session_ids is None or any(not isinstance(value, (str, np.str_)) or not value.strip()
                                      for value in session_ids):
            raise ValueError("source trial_ids require aligned, nonblank string session_ids")
        if len(set(zip(session_ids, trial_ids))) != n_total:
            raise ValueError("duplicate (session_id, source trial_id) in dataset")

    anchor_frames = dataset.get("anchor_frames")
    reference_source = "recorded_dataset_anchor"
    if anchor_frames is None:
        from nsmor.pipeline.conditions import derive_anchor_frames
        anchor_frames = derive_anchor_frames(X_seqs, lengths)
        reference_source = "derived_physical_channel_anchor"
        logger.info(
            "Derived %d anchor frames from physical channels.",
            len(anchor_frames),
        )

    # Build sequence list
    sequences = [
        (X_seqs[i], Y_seqs[i], int(labels[i]))
        for i in range(n_total)
    ]

    # Create dataset and dataloader
    feature_config = dataset.get("feature_config", DEFAULT_FEATURE)
    bio_dataset = NSMoRDataset(
        sequences=sequences,
        mcmc_priors=mcmc_priors,
        feature_config=feature_config,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        anchor_frames=anchor_frames,
    )

    bio_dataset.analysis_population = describe_analysis_population(
        n_total, val_indices, bio_dataset.source_indices,
        getattr(checkpoint_model, "analysis_validation_scope", None),
    )
    dataloader = create_optimized_dataloader(
        bio_dataset,
        batch_size=batch_size,
        shuffle=False,  # Preserve ordering for label matching
        num_workers=0,  # Whole corpus is already resident; workers only replicate it
    )

    # Use the same crop resolver as NSMoRDataset.__getitem__, including
    # the end-clamped start. A recorded anchor is not necessarily onset.
    from nsmor.pipeline.conditions import resolve_anchor_crop
    from nsmor.pipeline.grouping import animal_keys_of

    if len(anchor_frames) != n_total or len(labels) != n_total or len(lengths) != n_total:
        raise ValueError("anchor_frames must be aligned with dataset trials")
    if max_seq_len is not None and max_seq_len < 1:
        raise ValueError("max_seq_len must be positive or None")
    # Reuse the canonical suffix stripping only to identify recording prefixes;
    # these keys do not establish independent animals for statistical inference.
    groups = animal_keys_of(session_ids) if session_ids is not None else [None] * n_total
    rules = next((dataset[key] for key in ("anchor_rules", "anchor_rule")
                  if key in dataset and isinstance(dataset[key], (list, tuple, np.ndarray))), None)
    if rules is not None and len(rules) != n_total:
        raise ValueError("per-trial anchor rules must be aligned with dataset trials")
    reference_trials = []
    for i, anchor in enumerate(anchor_frames):
        start, end = resolve_anchor_crop(len(X_seqs[i]), anchor, max_seq_len, pre_anchor_frames)
        original = int(anchor) if anchor is not None and anchor >= 0 else None
        rule = str(rules[i]) if rules is not None and rules[i] is not None else None
        if reference_source == "derived_physical_channel_anchor" or (rule is None and original == 0):
            physical = np.asarray(X_seqs[i])[:int(lengths[i])]
            if np.any(physical[:, 1] > 0.5):
                if reference_source == "derived_physical_channel_anchor":
                    rule = "first_wind_channel_gt_0.5"
            elif np.any(np.abs(physical[:, 0]) > 1e-4):
                if reference_source == "derived_physical_channel_anchor":
                    rule = "peak_visual_angle_proxy"
            else:
                rule = "no_stimulus_frame_zero_fallback"
        reference_status = (
            "recorded_reference_rule_available" if rule is not None else "recorded_reference_rule_unavailable"
        ) if reference_source == "recorded_dataset_anchor" else {
            "first_wind_channel_gt_0.5": "derived_wind_threshold_reference",
            "peak_visual_angle_proxy": "derived_peak_visual_proxy_reference",
            "no_stimulus_frame_zero_fallback": "unavailable_no_stimulus_reference",
        }[rule]
        cropped_reference = (original - start if original is not None and start <= original < end
                             else None)
        if cropped_reference is None:
            reference_status = "unavailable_reference_outside_crop"
        if rule == "no_stimulus_frame_zero_fallback":
            reference_status = "unavailable_no_stimulus_reference"
        if reference_status == "unavailable_no_stimulus_reference":
            cropped_reference = None  # A no-stimulus fallback is not an observed event.
        session = session_ids[i] if session_ids is not None else None
        valid_session = isinstance(session, (str, np.str_)) and bool(session.strip())
        reference_trials.append({
            "trial_id": i, "row_index": i,
            "source_trial_id": int(trial_ids[i]) if trial_ids is not None else None,
            "label": int(labels[i]),
            "original_reference_frame": original,
            "crop_start_frame": start, "crop_end_frame": end,
            "cropped_length": end - start,
            "cropped_reference_frame": cropped_reference,
            "reference_source": reference_source, "anchor_rule": rule,
            "reference_status": reference_status,
            "session_id": str(session) if valid_session else None,
            "recording_prefix": str(groups[i]) if valid_session else None,
        })
    bio_dataset.analysis_reference_trials = reference_trials
    bio_dataset.cropped_reference_frames = [trial["cropped_reference_frame"] for trial in reference_trials]
    lengths_list = [trial["cropped_length"] for trial in reference_trials]
    return dataloader, labels, lengths_list


# ═══════════════════════════════════════════════════════════════
# 3.  Ablation Experiment Runner
# ═══════════════════════════════════════════════════════════════

def run_ablation_condition(
    model: NSMoRCore,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    override_gates: Optional[Dict[str, float]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Run the model under a single lesion condition.

    Args:
        model: Trained NSMoRCore model.
        dataloader: DataLoader yielding (X, Y, lengths) tuples.
        device: Computation device.
        override_gates: Gate override dict (None for intact).

    Returns:
        ``(y_preds, y_trues, trial_labels)`` where:
        - ``y_preds``: List of arrays, each (T_i,)
        - ``y_trues``: List of arrays, each (T_i,)
        - ``trial_labels``: List of label values
    """
    y_preds: List[np.ndarray] = []
    y_trues: List[np.ndarray] = []
    trial_labels: List[int] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            X_batch, Y_batch, lengths = batch
            X_batch = X_batch.to(device).contiguous()
            lengths = lengths.to(device).contiguous()

            # Forward pass with override
            Y_pred = model(X_batch, lengths, override_gates=override_gates)

            B, T = Y_pred.shape

            for i in range(B):
                length_i = int(lengths[i].item())

                # Extract valid (unpadded) predictions and targets
                y_pred_i = Y_pred[i, :length_i].cpu().numpy()  # (T_i,)
                y_true_i = Y_batch[i, :length_i].cpu().numpy()  # (T_i,)

                # Shape assertions
                assert y_pred_i.shape == (length_i,), (
                    f"y_pred shape {y_pred_i.shape} != ({length_i},)"
                )
                assert y_true_i.shape == (length_i,), (
                    f"y_true shape {y_true_i.shape} != ({length_i},)"
                )

                y_preds.append(y_pred_i)
                y_trues.append(y_true_i)

                # Get label
                global_idx = len(trial_labels)
                if global_idx < len(dataloader.dataset):
                    _, _, label_val = dataloader.dataset.sequences[global_idx]
                    trial_labels.append(int(label_val))
                else:
                    trial_labels.append(-1)  # Unknown

    return y_preds, y_trues, trial_labels


def run_full_ablation(
    model: NSMoRCore,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[int]]]:
    """
    Run the full ablation experiment (all three conditions).

    Args:
        model: Trained NSMoRCore model.
        dataloader: DataLoader yielding (X, Y, lengths) tuples.
        device: Computation device.

    Returns:
        Dictionary mapping condition name to
        ``(y_preds, y_trues, trial_labels)`` tuples.
    """
    results: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[int]]] = {}

    for condition_name, override in LESION_OVERRIDES.items():
        logger.info("Running condition: %s", CONDITION_NAMES[condition_name])

        if override is not None:
            logger.info("  Override gates: %s", override)
        else:
            logger.info("  Using natural routing (no override)")

        y_preds, y_trues, trial_labels = run_ablation_condition(
            model=model,
            dataloader=dataloader,
            device=device,
            override_gates=override,
        )

        results[condition_name] = (y_preds, y_trues, trial_labels)

        # Log summary statistics
        n_trials = len(y_preds)
        mean_pred_velocity = np.mean([np.mean(np.abs(yp)) for yp in y_preds])
        mean_true_velocity = np.mean([np.mean(np.abs(yt)) for yt in y_trues])
        logger.info(
            "  Trials: %d, Mean |predicted| velocity: %.3f cm/s, "
            "Mean |true| velocity: %.3f cm/s",
            n_trials, mean_pred_velocity, mean_true_velocity,
        )

    return results


# ═══════════════════════════════════════════════════════════════
# 4.  Class-Specific Trajectory Averaging
# ═══════════════════════════════════════════════════════════════

def _resolve_reference_frames(
    n_trials: int, stim_onset_frame: int,
    reference_frames: Optional[Sequence[Optional[int]]],
) -> List[Optional[int]]:
    """Resolve per-trial references; retain the legacy scalar for direct callers."""
    frames = list(reference_frames) if reference_frames is not None else [stim_onset_frame] * n_trials
    if len(frames) != n_trials:
        raise ValueError("reference_frames must be aligned with all trials")
    if any(frame is not None and (isinstance(frame, (bool, np.bool_))
           or not isinstance(frame, (int, np.integer)) or frame < 0) for frame in frames):
        raise ValueError("reference frames must be nonnegative integers or None (unavailable)")
    return frames


def _eligible_post_reference(
    pred: np.ndarray, true: np.ndarray, frame: Optional[int],
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Return one observed finite trial and its finite MSE, or no measurement.

    The figure, scalar metrics, CSV and schema-2 support use the same trial
    eligibility. A nonfinite sample anywhere in the observed window excludes
    that trial, including when the figure displays only its first time bins.
    """
    if frame is None:
        return None
    n = min(len(pred), len(true))
    if n <= frame:
        return None
    post_pred = np.asarray(pred[frame:n], dtype=np.float64)
    post_true = np.asarray(true[frame:n], dtype=np.float64)
    assert post_pred.shape == post_true.shape == (n - frame,)
    if not np.isfinite(post_pred).all() or not np.isfinite(post_true).all():
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(np.mean((post_pred - post_true) ** 2))
    if not np.isfinite(mse):
        return None
    return post_pred, post_true, mse


def _observed_peak(post_true: np.ndarray, dt_ms: float) -> Tuple[float, Optional[float]]:
    """Observed target magnitude and measurable post-reference peak timing."""
    absolute = np.abs(post_true)
    assert absolute.ndim == 1 and absolute.size > 0 and np.isfinite(absolute).all()
    maximum = float(np.max(absolute))
    # ponytail: same absolute-range floor as Phase F/G; calibrate prominence if drift matters.
    latency = None if maximum - float(np.min(absolute)) < 1e-6 else float(np.argmax(absolute) * dt_ms)
    return maximum, latency

def average_trajectories_by_class(
    y_preds: List[np.ndarray],
    y_trues: List[np.ndarray],
    trial_labels: List[int],
    target_class: int,
    dt_ms: float = 10.0,
    max_time_ms: float = 5000.0,
    stim_onset_frame: int = 1200,
    reference_frames: Optional[Sequence[Optional[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average velocity trajectories across trials of a specific class.

    Aligns trajectories relative to each trial reference (t=0) and averages
    them.  Each time bin averages only trials observed at that time.

    Args:
        y_preds: List of predicted velocity arrays.
        y_trues: List of ground truth velocity arrays.
        trial_labels: List of label values.
        target_class: Label value to filter by.
        dt_ms: Frame interval in milliseconds.
        max_time_ms: Maximum analysis window in ms.
        stim_onset_frame: Legacy scalar reference index (used without a vector).
        reference_frames: Optional cropped reference per trial, aligned with
            all trials. None entries have no supported reference.

    Returns:
        ``(time_ms, mean_pred, mean_true)`` arrays, all (n_frames,).
    """
    # Filter by target class
    class_indices = [i for i, l in enumerate(trial_labels) if l == target_class]

    if not class_indices:
        raise ValueError(f"No trials found for class {target_class}")

    logger.info(
        "Found %d trials for class %d (%s).",
        len(class_indices), target_class,
        Label(target_class).name if target_class in [e.value for e in Label] else "Unknown",
    )

    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be finite and positive")
    window_frames = int(max_time_ms / dt_ms)
    frames = _resolve_reference_frames(len(trial_labels), stim_onset_frame, reference_frames)
    observed = []
    for i in class_indices:
        eligible = _eligible_post_reference(y_preds[i], y_trues[i], frames[i])
        if eligible is not None:
            observed.append(eligible[:2])
    n_frames = min(window_frames, max(
        (min(len(pred), len(true)) for pred, true in observed), default=0
    ))
    if n_frames <= 0:
        raise ValueError(f"No observed post-reference samples for class {target_class}")

    pred_sum = np.zeros(n_frames)
    true_sum = np.zeros(n_frames)
    support = np.zeros(n_frames, dtype=np.int64)
    for post_pred, post_true in observed:
        n_copy = min(len(post_pred), len(post_true), n_frames)
        pred_sum[:n_copy] += post_pred[:n_copy]
        true_sum[:n_copy] += post_true[:n_copy]
        support[:n_copy] += 1
    assert support.shape == (n_frames,) and np.all(support > 0)
    time_ms = np.arange(n_frames) * dt_ms
    return time_ms, pred_sum / support, true_sum / support


# ═══════════════════════════════════════════════════════════════
# 4b.  Scalar Metrics Extraction for Statistical Analysis
# ═══════════════════════════════════════════════════════════════

def extract_scalar_metrics(
    y_preds: List[np.ndarray],
    y_trues: List[np.ndarray],
    trial_labels: List[int],
    target_class: int,
    dt_ms: float = 10.0,
    stim_onset_frame: int = 1200,
    reference_frames: Optional[Sequence[Optional[int]]] = None,
) -> Dict[str, Optional[float]]:
    """
    Extract scalar metrics from velocity trajectories for a given class.

    Computes:
        - **Peak Velocity (V_max):** Maximum absolute velocity in the
          observed post-reference window.
        - **Latency to Peak (T_max):** Time (ms) relative to the trial
          reference when V_max is reached.
        - **Mean MSE:** Mean squared error between predicted and true
          velocity over observed post-reference frames.

    Args:
        y_preds: List of predicted velocity arrays, each (T_i,).
        y_trues: List of ground truth velocity arrays, each (T_i,).
        trial_labels: List of label values for each trial.
        target_class: Label value to filter by.
        dt_ms: Frame interval in milliseconds.
        stim_onset_frame: Legacy scalar reference index (default 1200).
        reference_frames: Optional per-trial cropped reference indices.

    Returns:
        Dictionary with keys:
        - ``"Peak_Velocity_cms"``: Max absolute velocity (cm/s).
        - ``"Latency_to_Peak_ms"``: Time of target peak relative to reference (ms).
        - ``"Mean_MSE"``: Mean squared error over observed post-reference frames.

    Raises:
        ValueError: If no trials found for target_class.
    """
    # ── Filter by target class ────────────────────────────────
    class_indices = [i for i, l in enumerate(trial_labels) if l == target_class]

    if not class_indices:
        raise ValueError(
            f"No trials found for class {target_class} "
            f"({Label(target_class).name if target_class in [e.value for e in Label] else 'Unknown'})"
        )

    logger.info(
        "Extracting metrics for %d trials of class %d (%s).",
        len(class_indices), target_class,
        Label(target_class).name if target_class in [e.value for e in Label] else "Unknown",
    )

    # ── Collect post-reference velocity arrays ─────────────────
    peak_velocities: List[float] = []
    latencies: List[float] = []
    mse_values: List[float] = []

    frames = _resolve_reference_frames(len(trial_labels), stim_onset_frame, reference_frames)
    for trial_idx in class_indices:
        eligible = _eligible_post_reference(y_preds[trial_idx], y_trues[trial_idx], frames[trial_idx])
        if eligible is None:
            continue
        _, post_true, mse = eligible

        # ── Peak Velocity (V_max): maximum absolute velocity ──
        v_max, t_max = _observed_peak(post_true, dt_ms)

        # ── Mean MSE ──────────────────────────────────────────
        peak_velocities.append(v_max)
        if t_max is not None:
            latencies.append(t_max)
        mse_values.append(mse)

    if not peak_velocities:
        raise ValueError(f"No valid post-reference data for class {target_class}")

    # ── Aggregate across trials ───────────────────────────────
    metrics = {
        "Peak_Velocity_cms": float(np.mean(peak_velocities)),
        "Latency_to_Peak_ms": float(np.mean(latencies)) if latencies else None,
        "Mean_MSE": float(np.mean(mse_values)),
    }

    logger.info(
        "  Metrics: V_max=%.3f cm/s, T_max=%s ms, MSE=%.4f",
        metrics["Peak_Velocity_cms"],
        metrics["Latency_to_Peak_ms"],
        metrics["Mean_MSE"],
    )

    return metrics


class _NoValidLesionMeasurements(ValueError):
    """The requested CSV has no eligible post-reference trial rows."""


def export_lesion_statistics_csv(
    results: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[int]]],
    output_path: Path,
    target_classes: List[int],
    dt_ms: float = 10.0,
    stim_onset_frame: int = 1200,
    reference_frames: Optional[Sequence[Optional[int]]] = None,
) -> None:
    """
    Export descriptive per-trial lesion measurements to the legacy CSV.

    MSE uses restored, unclipped physical predictions against raw physical
    targets. It is not the symmetrically clipped train.compute_metrics score.
    Peak/latency columns describe the observed target relative to the trial
    reference. Rows from shared recordings are not independent animal samples.

    Generates a CSV with columns:
    ``Class, Condition, Trial_ID, Peak_Velocity_cms, Latency_to_Peak_ms, MSE``
    ``Trial_ID`` is the zero-based dataset row index, joining to the existing
    sidecar ``trials[row_index]``; raw identity is (session_id, source_trial_id).

    Args:
        results: Dictionary from :func:`run_full_ablation` mapping
            condition names to ``(y_preds, y_trues, trial_labels)``.
        output_path: Path to save the CSV file.
        target_classes: List of label values to analyze.
        dt_ms: Frame interval in milliseconds.
        stim_onset_frame: Legacy scalar reference index.
        reference_frames: Optional per-trial cropped reference indices.

    Raises:
        ValueError: If no valid statistics could be computed.
    """
    logger.info("=" * 60)
    logger.info("Exporting lesion statistics to CSV...")
    logger.info("=" * 60)

    # Keep the six-column measurement interface. Group identities, observed
    # support, and inferential limitations are recorded in the existing sidecar.
    csv_rows: List[Dict[str, str]] = []

    for target_class in target_classes:
        try:
            class_name = Label(target_class).name
        except ValueError:
            class_name = f"Class_{target_class}"

        for condition_name, (y_preds, y_trues, trial_labels) in results.items():
            frames = _resolve_reference_frames(len(trial_labels), stim_onset_frame, reference_frames)
            class_indices = [i for i, l in enumerate(trial_labels) if l == target_class]
            if not class_indices:
                continue

            for trial_idx in class_indices:
                eligible = _eligible_post_reference(y_preds[trial_idx], y_trues[trial_idx],
                                                    frames[trial_idx])
                if eligible is None:
                    continue
                _, post_true, mse = eligible
                v_max, t_max = _observed_peak(post_true, dt_ms)


                csv_rows.append({
                    "Class": class_name,
                    "Condition": CONDITION_NAMES[condition_name],
                    "Trial_ID": str(trial_idx),
                    "Peak_Velocity_cms": f"{v_max:.4f}",
                    "Latency_to_Peak_ms": f"{t_max:.2f}" if t_max is not None else "",
                    "MSE": f"{mse:.6f}",
                })

    if not csv_rows:
        # The selected output belongs to this run. An older CSV must not pass
        # an existence/size check after an all-ineligible run.
        output_path.unlink(missing_ok=True)
        raise _NoValidLesionMeasurements("No valid statistics could be computed for any class/condition.")

    # ── Write CSV ─────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Class", "Condition", "Trial_ID", "Peak_Velocity_cms", "Latency_to_Peak_ms", "MSE"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info("Saved %d rows to %s", len(csv_rows), output_path)

    # ── Log summary (first 10 rows) ──────────────────────────
    logger.info("-" * 80)
    logger.info("Per-trial data: %d rows exported (showing first 10):", len(csv_rows))
    for row in csv_rows[:10]:
        logger.info(
            "  %s | %s | trial=%s | V_max=%s | T_max=%s | MSE=%s",
            row["Class"], row["Condition"], row["Trial_ID"],
            row["Peak_Velocity_cms"], row["Latency_to_Peak_ms"], row["MSE"],
        )
    logger.info("-" * 80)


# ═══════════════════════════════════════════════════════════════
# 5.  Lancet/Cell Publication Figure
# ═══════════════════════════════════════════════════════════════

def setup_lancet_style() -> None:
    """Configure matplotlib for Lancet/Cell publication aesthetics."""
    plt.rcParams.update({
        # ── Font ──
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE_TICK,
        "axes.titlesize": FONT_SIZE_AXIS_TITLE,
        "axes.labelsize": FONT_SIZE_AXIS_TITLE,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,

        # ── Axes ──
        "axes.linewidth": 1.5,
        "axes.edgecolor": AXIS_COLOR,
        "axes.labelcolor": AXIS_COLOR,
        "xtick.color": AXIS_COLOR,
        "ytick.color": AXIS_COLOR,

        # ── Grid ──
        "axes.grid": False,
        "grid.alpha": 0.15,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,

        # ── Figure ──
        "figure.facecolor": BACKGROUND_COLOR,
        "savefig.facecolor": BACKGROUND_COLOR,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,

        # ── Legend ──
        "legend.frameon": True,
        "legend.facecolor": BACKGROUND_COLOR,
        "legend.edgecolor": AXIS_COLOR,
        "legend.framealpha": 1.0,
    })


def create_ablation_figure(
    results: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[int]]],
    target_class: int,
    output_path: Path,
    dt_ms: float = 10.0,
    stim_onset_frame: int = 1200,
    reference_frames: Optional[Sequence[Optional[int]]] = None,
    analysis_population: Optional[Dict[str, object]] = None,
) -> None:
    """
    Create the Lancet/Cell ablation comparison figure.

    Layout: subplots(3, 1) — Intact, LIF-Lesioned, GRU-Lesioned

    Args:
        results: Dictionary from :func:`run_full_ablation`.
        target_class: Label value to plot.
        output_path: Path to save the figure.
        dt_ms: Frame interval in milliseconds.
    """
    setup_lancet_style()

    # Get class name
    try:
        class_name = Label(target_class).name.replace("_", " ").title()
    except ValueError:
        class_name = f"Class {target_class}"

    # ── Create figure with [3, 1] layout ──
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))

    # Panel labels
    panel_labels = ["A", "B", "C"]

    for idx, (condition_name, (y_preds, y_trues, trial_labels)) in enumerate(results.items()):
        ax = axes[idx]

        logger.info("Plotting condition: %s", CONDITION_NAMES[condition_name])

        # Average trajectories by class
        try:
            time_ms, mean_pred, mean_true = average_trajectories_by_class(
                y_preds=y_preds,
                y_trues=y_trues,
                trial_labels=trial_labels,
                target_class=target_class,
                dt_ms=dt_ms,
                stim_onset_frame=stim_onset_frame,
                reference_frames=reference_frames,
            )
        except ValueError as e:
            logger.warning("Skipping condition %s: %s", condition_name, e)
            ax.text(
                0.5, 0.5, f"No data for {class_name}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_SIZE_AXIS_TITLE, color=AXIS_COLOR,
            )
            continue

        # Plot Ground Truth (Lancet Crimson Red, solid)
        ax.plot(
            time_ms, mean_true,
            color=GROUND_TRUTH_COLOR,
            linewidth=LINE_WIDTH_GT,
            solid_capstyle="round",
            label="Ground Truth",
        )

        # Plot Predicted (Strong Slate Gray, dashed)
        ax.plot(
            time_ms, mean_pred,
            color=PREDICTED_COLOR,
            linewidth=LINE_WIDTH_PRED,
            linestyle=DASHED_LINESTYLE,
            solid_capstyle="round",
            label="Predicted",
        )

        # ── Trial reference vertical line ──
        ax.axvline(
            x=0.0,
            color=AXIS_COLOR,
            linewidth=1.0,
            linestyle=":",
            alpha=0.5,
        )

        # ── Axes styling ──
        ax.set_xlabel(
            "Time relative to trial reference (ms)",
            fontsize=FONT_SIZE_AXIS_TITLE,
            color=AXIS_COLOR,
        )
        ax.set_ylabel(
            "Velocity (cm/s)",
            fontsize=FONT_SIZE_AXIS_TITLE,
            color=AXIS_COLOR,
        )

        # Tick formatting
        ax.tick_params(axis="both", colors=AXIS_COLOR, width=1.5)

        # Spine styling
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOR)
            spine.set_linewidth(1.5)

        # Grid: ultra-faint major grid lines
        ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)

        # Legend
        ax.legend(
            loc="upper left",
            fontsize=FONT_SIZE_LEGEND,
            frameon=True,
            facecolor=BACKGROUND_COLOR,
            edgecolor=AXIS_COLOR,
            framealpha=1.0,
        )

        # Panel label and condition title
        ax.set_title(
            f"{panel_labels[idx]}  {CONDITION_NAMES[condition_name]}",
            fontsize=FONT_SIZE_PANEL_LABEL,
            fontweight="bold",
            color=AXIS_COLOR,
            loc="left",
        )

    # ── Suptitle ──
    fig.suptitle(
        f"In-Silico Lesion Analysis — {class_name} Trials ("
        + (analysis_population["selection"].replace("_", " ") + "; "
           + analysis_population["evidence_scope"].replace("_", " ")
           if analysis_population is not None else "descriptive population unavailable") + ")",
        fontsize=14,
        fontweight="bold",
        color=AXIS_COLOR,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Save ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    logger.info("Saved ablation figure to %s (%d DPI)", output_path, DPI)

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 6.  Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════

def run_lesion_experiment(
    checkpoint_path: Path,
    dataset_path: Path,
    output_path: Path,
    stats_output_path: Optional[Path] = None,
    target_class: int = 0,
    target_classes: Optional[List[int]] = None,
    batch_size: int = 32,
    max_seq_len: Optional[int] = 2400,
    pre_anchor_frames: int = 1200,
    dt_ms: Optional[float] = None,
    stim_onset_frame: Optional[int] = None,
    nested_prior_artifact: Optional[Path] = None,
    qc_sealed_nested_prior_sha256: Optional[str] = None,
    trusted_historical_checkpoint_sha256: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
) -> None:
    """
    Run the full in-silico lesion experiment.

    Args:
        checkpoint_path: Path to the trained model checkpoint.
        dataset_path: Path to the preprocessed dataset.
        output_path: Path to save the ablation figure.
        stats_output_path: Path to save the statistics CSV. If None,
            defaults to ``results/lesion_statistics.csv``.
        target_class: Label value to plot in the figure (default 0 = ESCAPE).
        target_classes: List of label values to include in statistics CSV.
            If None, defaults to all classes [0, 1, 2, 3].
        batch_size: Batch size for data loading.
        max_seq_len: Maximum sequence length for cropping.
        pre_anchor_frames: Baseline frames before anchor.
        dt_ms: None inherits saved model.dt_ms; an explicit interval must match.
        stim_onset_frame: Optional scalar override in cropped coordinates.
            None uses each recorded/derived dataset anchor as a reference;
            loaders without metadata retain the legacy scalar fallback of 1200.
            A dataset anchor need not represent physical stimulus onset.
    """
    logger.info("=" * 60)
    logger.info("NSMoR In-Silico Lesion Experiment (Phase 7)")
    logger.info("=" * 60)

    # ── Default paths ─────────────────────────────────────────
    if stats_output_path is None:
        stats_output_path = Path("results/lesion_statistics.csv")
    if target_classes is None:
        target_classes = [e.value for e in Label]  # All 4 classes

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────
    model = load_model_from_checkpoint(checkpoint_path, device)
    dt_ms = resolve_dt_ms(model, dt_ms)

    # ── Load dataset ──────────────────────────────────────────
    dataloader, labels, lengths_list = load_dataset(
        dataset_path,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        nested_prior_artifact=nested_prior_artifact,
        qc_sealed_nested_prior_sha256=qc_sealed_nested_prior_sha256,
        checkpoint_model=model,
        trusted_historical_checkpoint_sha256=trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
    )

    reference_override = stim_onset_frame is not None
    analysis_dataset = getattr(dataloader, "dataset", None)
    reference_frames = getattr(analysis_dataset, "cropped_reference_frames", None)
    # Explicit scalar overrides remain supported; default CLI uses recorded references.
    if stim_onset_frame is not None:
        reference_frames = None
    else:
        stim_onset_frame = 1200  # Legacy loaders without reference metadata.

    logger.info("Lesion MSE compares unclipped physical predictions and raw targets; "
                "it differs from clipped training/evaluation scoring.")

    # ── Run ablation ──────────────────────────────────────────
    results = run_full_ablation(model, dataloader, device)

    # Recording prefixes collapse session blocks, but do not prove independent
    # animal identity. Report descriptive group means/effects; no trial t-tests
    # or acquisition-order bootstrap can supply that missing evidence.
    from nsmor.analysis.uq import cohens_d

    frames = _resolve_reference_frames(len(labels), stim_onset_frame, reference_frames)
    recorded_trials = getattr(analysis_dataset, "analysis_reference_trials", None)
    reference_trials = [dict(trial) for trial in recorded_trials] if recorded_trials is not None else [
        {"trial_id": i, "row_index": i, "source_trial_id": None,
          "label": int(label), "original_reference_frame": None,
         "crop_start_frame": None, "crop_end_frame": None,
         "cropped_length": int(lengths_list[i]), "cropped_reference_frame": None,
         "reference_source": "unavailable", "anchor_rule": None,
         "reference_status": "unavailable_loader_metadata",
         "session_id": None, "recording_prefix": None}
        for i, label in enumerate(labels)
    ]
    if len(reference_trials) != len(labels):
        raise ValueError("analysis reference metadata must align with all trials")
    for i, trial in enumerate(reference_trials):
        trial["analysis_reference_frame"] = frames[i]
        trial["analysis_reference_source"] = (
            "explicit_scalar_override" if reference_override else
            trial["reference_source"] if reference_frames is not None else "legacy_scalar_fallback"
        )
        trial["post_reference_frames"] = {}
        trial["post_reference_status"] = {}

    population = population_for_output(dataloader, len(labels))
    unavailable_p = "unavailable_unverified_animal_identity_and_independence"
    summary = {
        "schema_version": 2, "analysis_status": "descriptive_only",
        "analysis_population": population,
        "csv_trial_id_semantics": "zero_based_dataset_row_index",
        "summary_class": int(target_class), "dt_ms": float(dt_ms),
        "mse_units": "(cm/s)^2", "mse_window": "observed_post_reference",
        "reference_semantics": "trial_reference_not_verified_stimulus_onset",
        "ci_status": unavailable_p, "p_status": unavailable_p,
        "trials": reference_trials, "conditions": {}, "comparisons": {},
    }
    condition_scores = {}
    for name, (y_preds, y_trues, trial_labels) in results.items():
        assert len(y_preds) == len(y_trues) == len(trial_labels) == len(reference_trials)
        if any(int(label) != int(reference_trials[i]["label"])
               for i, label in enumerate(trial_labels)):
            raise ValueError("lesion labels do not align with trial reference metadata")
        scores = {}
        for i, (pred, true) in enumerate(zip(y_preds, y_trues)):
            frame = frames[i]
            n_post = max(0, min(len(pred), len(true)) - frame) if frame is not None else 0
            trial = reference_trials[i]
            trial["post_reference_frames"][name] = n_post
            trial["post_reference_status"][name] = (
                "missing_reference" if frame is None else
                "observed" if n_post else "no_observed_post_reference"
            )
            if not n_post:
                continue
            eligible = _eligible_post_reference(pred, true, frame)
            if eligible is None:
                trial["post_reference_status"][name] = "nonfinite_observations"
                continue
            _, _, mse = eligible
            scores[i] = mse
        condition_scores[name] = scores
        target_scores = {i: mse for i, mse in scores.items() if int(trial_labels[i]) == target_class}
        grouped = {}
        for i, mse in target_scores.items():
            grouped.setdefault(reference_trials[i]["recording_prefix"], []).append(mse)
        group_means = [float(np.mean(values)) for values in grouped.values()] if None not in grouped else []
        summary["conditions"][name] = {
            "n_finite_trials": len(target_scores), "n_recording_prefixes": len(group_means),
            "mean_trial_mse": float(np.mean(list(target_scores.values()))) if target_scores else None,
            "mean_recording_prefix_mse": float(np.mean(group_means)) if group_means else None,
            "status": "descriptive_only" if target_scores else "unavailable_no_valid_post_reference",
        }

    # Exact, finite scores from the full observed window; the six-column CSV
    # rounds the same measurement to six decimal places.
    for i, trial in enumerate(reference_trials):
        trial["mse_by_condition"] = {name: scores.get(i) for name, scores in condition_scores.items()}

    intact = {i: mse for i, mse in condition_scores.get("intact", {}).items()
              if int(reference_trials[i]["label"]) == target_class}
    for name, all_scores in condition_scores.items():
        if name == "intact":
            continue
        scores = {i: mse for i, mse in all_scores.items()
                  if int(reference_trials[i]["label"]) == target_class}
        paired_ids = sorted(scores.keys() & intact.keys())
        grouped = {}
        for i in paired_ids:
            grouped.setdefault(reference_trials[i]["recording_prefix"], []).append(i)
        if None in grouped:
            grouped = {}  # Incomplete identities cannot be silently dropped.
        paired_groups = [
            {"recording_prefix": key, "n_paired_trials": len(ids),
             "intact_mean_mse": float(np.mean([intact[i] for i in ids])),
             "lesioned_mean_mse": float(np.mean([scores[i] for i in ids]))}
            for key, ids in sorted(grouped.items())
        ]
        intact_means = np.array([group["intact_mean_mse"] for group in paired_groups])
        lesion_means = np.array([group["lesioned_mean_mse"] for group in paired_groups])
        assert intact_means.shape == lesion_means.shape == (len(paired_groups),)
        effect = None
        if not paired_ids:
            effect_status = "unavailable_no_paired_post_reference"
        elif not paired_groups:
            effect_status = "unavailable_recording_group_metadata"
        elif len(paired_groups) < 2:
            effect_status = "unavailable_fewer_than_two_recording_prefixes"
        else:
            d = cohens_d(lesion_means, intact_means, paired=True)
            effect = d if np.isfinite(d) else None
            effect_status = "estimable_descriptive_only" if effect is not None else (
                "undefined_zero_variance_paired_differences"
                if np.std(lesion_means - intact_means, ddof=1) == 0 else "undefined_nonfinite_paired_variance"
            )
        comparison = {
            "n_paired_trials": len(paired_ids), "n_paired_recording_prefixes": len(paired_groups),
            "paired_groups": paired_groups,
            "delta_mse": float(np.mean(lesion_means - intact_means)) if paired_groups else None,
            "effect_size": effect, "effect_size_measure": "paired_cohens_dz",
            "effect_size_unit": "paired_recording_prefix_means", "effect_size_status": effect_status,
            "p_value": None, "p_adjusted": None, "significant": None, "p_status": unavailable_p,
        }
        summary["comparisons"][name] = comparison
        logger.info("%s vs Intact: recording-prefix descriptive d=%s (%s); p/adjusted p unavailable: %s",
                    CONDITION_NAMES[name], effect, effect_status, unavailable_p)

    # ── Export statistics CSV ──────────────────────────────────
    export_error = None
    try:
        export_lesion_statistics_csv(
            results=results,
            output_path=stats_output_path,
            target_classes=target_classes,
            dt_ms=dt_ms,
            stim_onset_frame=stim_onset_frame,
            reference_frames=reference_frames,
        )
    except _NoValidLesionMeasurements as exc:
        export_error = exc
        summary["analysis_status"] = "unavailable_no_valid_measurements"
        logger.error("Could not export statistics CSV: %s", exc)

    # Reuse the existing sidecar path; schema 2 describes groups/support rather
    # than asserting that acquisition-order block CIs establish independence.
    sens_path = stats_output_path.with_suffix(".block_sensitivity.json")
    sens_path.parent.mkdir(parents=True, exist_ok=True)
    with sens_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    logger.info("Descriptive statistics and trial references saved to %s", sens_path)
    if export_error is not None:
        raise export_error

    # ── Create figure ─────────────────────────────────────────
    create_ablation_figure(
        results=results,
        target_class=target_class,
        output_path=output_path,
        dt_ms=dt_ms,
        stim_onset_frame=stim_onset_frame,
        reference_frames=reference_frames,
        analysis_population=population,
    )

    logger.info("=" * 60)
    logger.info("Lesion experiment complete!")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
# 7.  CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="NSMoR In-Silico Lesion (Virtual Ablation) Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/nsmor_dataset.pt",
        help="Path to preprocessed dataset.",
    )
    parser.add_argument(
        "--nested_prior_artifact",
        type=str,
        default=None,
        help="Optional validated nested prior artifact used by the checkpoint.",
    )
    parser.add_argument(
        "--qc_sealed_nested_prior_sha256", type=str, default=None,
        help="Optional external QC-sealed SHA-256 to check the nested checkpoint's embedded artifact digest; cannot authenticate a checkpoint missing that digest.",
    )
    parser.add_argument(
        "--trusted_historical_checkpoint_sha256", type=str, default=None,
        help="Independently pinned SHA-256 of historical checkpoint bytes (required for historical analysis).",
    )
    parser.add_argument(
        "--trusted_historical_artifact_sha256", type=str, default=None,
        help="Independently pinned SHA-256 of historical nested artifact bytes.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ablation_kinematics.png",
        help="Output path for ablation figure.",
    )
    parser.add_argument(
        "--stats_output",
        type=str,
        default="results/lesion_statistics.csv",
        help="Output path for lesion statistics CSV.",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=0,
        help="Label value to plot in figure (0=ESCAPE, 1=WALK, 2=PRE_ACTIVE, 3=NO_RESPONSE).",
    )
    parser.add_argument(
        "--target_classes",
        type=int,
        nargs="+",
        default=None,
        help="Label values to include in statistics CSV (default: all classes).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for data loading.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2400,
        help="Crop sequences longer than this (cuDNN compatibility). 0 = disable.",
    )
    parser.add_argument(
        "--pre_anchor_frames",
        type=int,
        default=1200,
        help="Number of frames before anchor to include in anchor-aligned crop.",
    )
    parser.add_argument(
        "--dt_ms",
        type=float,
        default=None,
        help="Frame interval in ms (default: saved model.dt_ms; explicit value must match).",
    )
    parser.add_argument(
        "--stim_onset_frame",
        type=int,
        default=None,
        help="Override all cropped trial reference indices with one scalar (legacy fallback: 1200). "
             "By default use each recorded/derived dataset anchor, which need not be stimulus onset.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    pre_anchor_frames = getattr(args, "pre_anchor_frames", 1200)
    run_lesion_experiment(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        nested_prior_artifact=Path(args.nested_prior_artifact) if args.nested_prior_artifact else None,
        qc_sealed_nested_prior_sha256=args.qc_sealed_nested_prior_sha256,
        trusted_historical_checkpoint_sha256=args.trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=args.trusted_historical_artifact_sha256,
        output_path=Path(args.output),
        stats_output_path=Path(args.stats_output),
        target_class=args.target_class,
        target_classes=args.target_classes,
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        dt_ms=args.dt_ms,
        stim_onset_frame=args.stim_onset_frame,
    )


if __name__ == "__main__":
    main()
