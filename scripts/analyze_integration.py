"""
NSMoR Multisensory Integration Window Analysis — Phase 9.

Generates chronometric curves showing how mechanical wind timing
modulates the visual looming response. Groups model predictions by
experimental condition (Wind delay relative to TTC) and plots the
Multisensory Integration Window.

Key Analysis
------------
- **Condition Grouping:** Trials are grouped by wind onset time
  relative to Visual TTC (ΔT in ms). Conditions include:
  * Visual-Only (no wind)
  * Wind-Only (no visual stimulus, pure-wind with 570-frame prepend)
  * Multisensory: Wind at TTC-373ms, TTC-119ms, TTC 0ms, etc.

- **Metrics:** For each condition, extracts:
  * Peak Velocity (V_max): Maximum absolute predicted velocity post-stimulus
  * Latency to Peak (T_max): Time of V_max relative to stimulus onset

Output
------
- ``results/integration_window.png``: Dual-panel chronometric/vigor curves
- ``results/integration_summary.json``: Statistical summary

Usage
-----
CLI::

    python scripts/analyze_integration.py --checkpoint runs/default/best_model.pth
    python scripts/analyze_integration.py --checkpoint runs/default/best_model.pth --dataset data/processed/nsmor_dataset.pt

"""

from __future__ import annotations

import argparse
import sys
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Direct CLI execution must use this checkout, including its prior/lineage helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
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
from nsmor.pipeline.conditions import resolve_anchor_crop
from nsmor.pipeline.events import load_declared_events_index

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
# STRICTLY REJECT pale, desaturated, or pastel palettes
PRIMARY_LINE_COLOR: str = "#1C7ED6"     # Cell Cobalt Blue — main data
BASELINE_COLOR: str = "#495057"         # Strong Slate Gray — reference lines
AXIS_COLOR: str = "#212529"            # Solid dark charcoal — axes
BACKGROUND_COLOR: str = "#FFFFFF"       # Clean white background
ERROR_BAR_COLOR: str = "#1C7ED6"        # Match line color

# ── Condition display names ──
CONDITION_DISPLAY_NAMES: Dict[str, str] = {
    "visual_only": "Visual-Only",
    "wind_only": "Wind-Only",
    "no_stimulus": "No-Stimulus",
    "multisensory_ttc_-373ms": "Wind at TTC−373ms",
    "multisensory_ttc_-119ms": "Wind at TTC−119ms",
    "multisensory_ttc_0ms": "Wind at TTC",
    "multisensory_ttc_+200ms": "Wind at TTC+200ms",
    "multisensory_other": "Other Multisensory",
}


def format_ttc_condition_name(ttc_val: float) -> str:
    """Return canonical dynamic group name for a declared TTC value (ms)."""
    if abs(float(ttc_val) - 0.0) < 1e-3:
        return "multisensory_ttc_0ms"
    return f"multisensory_ttc_{float(ttc_val):+.0f}ms"


def parse_ttc_condition_delta(cond: str) -> Optional[float]:
    """Parse delta_t_ms from a dynamic condition name, or None if not TTC-bearing.

    Handles both legacy hardcoded names and dynamic names such as
    ``multisensory_ttc_-225ms`` / ``multisensory_ttc_+200ms``.
    """
    prefix = "multisensory_ttc_"
    if not cond.startswith(prefix) or not cond.endswith("ms"):
        return None
    suffix = cond[len(prefix):-2]
    try:
        return float(suffix)
    except ValueError:
        return None

# ── Typography ─────────────────────────────────────────────────
FONT_FAMILY: str = "Arial"
FONT_SIZE_AXIS_TITLE: int = 12
FONT_SIZE_TICK: int = 10
FONT_SIZE_LEGEND: int = 9
FONT_SIZE_PANEL_LABEL: int = 14

# ── Figure properties ─────────────────────────────────────────
DPI: int = 300
FIG_WIDTH_INCHES: float = 12.0
FIG_HEIGHT_INCHES: float = 5.0

# ── Plot properties ───────────────────────────────────────────
LINE_WIDTH: float = 2.5
MARKER_SIZE: float = 8.0
ERROR_BAR_CAPSIZE: float = 4.0
BASELINE_LINESTYLE: str = "--"
BASELINE_LINEWIDTH: float = 1.5


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
    raw_dir: Optional[Path] = None,
    nested_prior_artifact: Optional[Path] = None,
    qc_sealed_nested_prior_sha256: Optional[str] = None,
    checkpoint_model: Optional[NSMoRCore] = None,
    trusted_historical_checkpoint_sha256: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
) -> Tuple[torch.utils.data.DataLoader, np.ndarray, List[int], np.ndarray, List[Dict]]:
    """
    Load the preprocessed dataset and create a DataLoader.

    Args:
        dataset_path: Path to ``nsmor_dataset.pt``.
        batch_size: Batch size for the DataLoader.
        max_seq_len: Maximum sequence length for cropping.
        pre_anchor_frames: Number of baseline frames before anchor to retain.
        raw_dir: Optional directory with declared raw events for condition lookup.

    Returns:
        ``(dataloader, labels, lengths_list, X_seqs, trial_info_list)`` tuple.
        X_seqs is the raw feature array for wind onset detection.
        trial_info_list contains trial type info from events.

    Raises:
        FileNotFoundError: If dataset file does not exist.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info("Loading dataset from %s", dataset_path)
    dataset, loaded_source_fingerprint = load_dataset_with_fingerprint(dataset_path)

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

    anchor_frames = dataset.get("anchor_frames")
    if anchor_frames is None:
        from nsmor.pipeline.conditions import derive_anchor_frames
        anchor_frames = derive_anchor_frames(X_seqs, lengths)
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
        shuffle=False,  # Preserve ordering for condition matching
        num_workers=-1,  # Auto-scale based on dataset size
    )

    lengths_list = [int(l) for l in lengths]

    # Load stimulus condition metadata from dataset or derive from channels
    stored_cond = dataset.get("stimulus_conditions")
    if stored_cond is not None:
        stimulus_conditions = [str(c) for c in stored_cond]
    else:
        from nsmor.pipeline.conditions import derive_stimulus_metadata
        derived_cond, _ = derive_stimulus_metadata(X_seqs, lengths)
        stimulus_conditions = [str(c) for c in derived_cond]

    session_ids = dataset.get("session_ids")
    trial_ids = dataset.get("trial_ids")
    target_ttc_ms = dataset.get("target_ttc_ms")

    # Keyed raw lookup is required when the dataset target_ttc_ms key is
    # absent OR present-but-null/partial and raw_dir is explicitly provided.
    need_raw_lookup = False
    if target_ttc_ms is None:
        need_raw_lookup = raw_dir is not None
    else:
        need_raw_lookup = raw_dir is not None and any(
            t is None or (isinstance(t, float) and not np.isfinite(t))
            for t in target_ttc_ms
        )

    declared_events = {}
    if need_raw_lookup and raw_dir is not None and Path(raw_dir).exists():
        declared_events = load_declared_events_index(raw_dir)

    trial_info_list = []
    for i, cond in enumerate(stimulus_conditions):
        sess_id = (
            str(session_ids[i])
            if session_ids is not None and i < len(session_ids)
            else None
        )
        t_id = (
            int(trial_ids[i])
            if trial_ids is not None and i < len(trial_ids)
            else None
        )
        ttc = None
        lv = None
        ttc_resolved = False
        if target_ttc_ms is not None and i < len(target_ttc_ms):
            raw_t = target_ttc_ms[i]
            if raw_t is not None and not (isinstance(raw_t, float) and math.isnan(raw_t)):
                try:
                    val = float(raw_t)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Corrupt target_ttc_ms {raw_t!r} at index {i}: {e}"
                    ) from e
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(
                        f"Non-finite target_ttc_ms {raw_t!r} at index {i}"
                    )
                ttc = val
                ttc_resolved = True

        is_multisensory = cond in ("multisensory", "looming_wind")

        if not ttc_resolved and raw_dir is not None:
            # Disambiguate single-trial session if trial_ids key was missing in dataset
            if sess_id is not None and t_id is None:
                matching_keys = [k for k in declared_events if k[0] == sess_id]
                if len(matching_keys) == 1:
                    t_id = matching_keys[0][1]

            # Explicit raw_dir: fail closed on missing identity or keys.
            if sess_id is None or t_id is None:
                raise ValueError(
                    f"missing session_ids/trial_ids at index {i} for keyed "
                    "raw lookup; failing closed."
                )
            if (sess_id, t_id) in declared_events:
                ev = declared_events[(sess_id, t_id)]
                raw_t = ev.get("target_ttc_ms")
                if raw_t is not None and np.isfinite(raw_t):
                    ttc = float(raw_t)
                    ttc_resolved = True
                lv = ev.get("lv_ratio_ms")
                if is_multisensory and not ttc_resolved:
                    raise ValueError(
                        f"Unresolved required target_ttc_ms for multisensory "
                        f"trial ({sess_id}, {t_id}); failing closed."
                    )
                # Legitimate unimodal null stays null.
            elif is_multisensory:
                raise ValueError(
                    f"Unresolved declared events key ({sess_id}, {t_id}); "
                    "failing closed without inferred TTC."
                )

        a_frame = anchor_frames[i] if i < len(anchor_frames) else pre_anchor_frames
        trial_info_list.append({
            "type": cond,
            "condition": cond,
            "session_id": sess_id,
            "trial_id": t_id,
            "target_ttc_ms": ttc,
            "lv_ratio_ms": lv,
            "anchor_frame": a_frame,
        })

    return dataloader, labels, lengths_list, X_seqs, trial_info_list


def _load_trial_info_from_events(raw_dir: Path) -> List[Dict]:
    """
    Load trial type info from events CSV files using keyed parser.

    Returns list of dicts with keys: type, target_ttc_ms, lv_ratio_ms
    """
    events_index = load_declared_events_index(raw_dir)
    return list(events_index.values())


# ═══════════════════════════════════════════════════════════════
# 3.  Condition Grouping Logic
# ═══════════════════════════════════════════════════════════════

def detect_wind_onset_frame(
    x_seq: np.ndarray,
    stim_onset_frame: int = 1200,
    wind_threshold: float = 0.5,
) -> Optional[int]:
    """
    Detect the first frame where wind stimulus is active.

    Args:
        x_seq: Single trial feature array, shape ``(T_i, 8)``.
            Index 1 is the wind state feature ``wind(t)``.
        stim_onset_frame: Frame index of visual stimulus onset / anchor.
        wind_threshold: Threshold to detect wind activation.

    Returns:
        Frame index of wind onset relative to sequence start,
        or ``None`` if no wind detected.
    """
    assert x_seq.ndim == 2 and x_seq.shape[1] == 8, (
        f"x_seq must be (T, 8), got {x_seq.shape}"
    )

    # Wind feature is at index 1
    wind_feature = x_seq[:, 1]
    active_frames = np.where(wind_feature > wind_threshold)[0]

    if len(active_frames) == 0:
        return None  # No wind detected

    return int(active_frames[0])


def classify_wind_condition(
    x_seq: np.ndarray,
    stim_onset_frame: int = 1200,
    dt_ms: float = 4.0,
    wind_threshold: float = 0.5,
) -> Tuple[str, Optional[float]]:
    """
    Classify the experimental condition based on wind onset timing.

    Args:
        x_seq: Single trial feature array, shape ``(T_i, 8)``.
        stim_onset_frame: Frame index of visual stimulus onset / anchor.
        dt_ms: Frame interval in milliseconds.
        wind_threshold: Threshold for wind detection.

    Returns:
        ``(condition_name, delta_t_ms)`` tuple where:
        - ``condition_name``: String identifier for the condition.
        - ``delta_t_ms``: Wind onset time relative to TTC (ms).
          Negative = wind before TTC, Positive = wind after TTC.
          ``None`` for visual-only or wind-only conditions.
    """
    visual_feature = x_seq[:, 0]
    has_visual = bool(np.any(np.abs(visual_feature) > 1e-4))

    wind_onset_frame = detect_wind_onset_frame(
        x_seq, stim_onset_frame, wind_threshold
    )
    has_wind = wind_onset_frame is not None

    if has_visual and not has_wind:
        return "visual_only", None

    if has_wind and not has_visual:
        return "wind_only", None

    if has_visual and has_wind:
        # Kinematics-only fallback: NEVER assign a declared TTC0 (or any
        # specific TTC bin) from wind-onset timing. Declared metadata owns
        # TTC group names; this path stays explicit as multisensory_other.
        delta_t_ms = float((wind_onset_frame - stim_onset_frame) * dt_ms)
        return "multisensory_other", delta_t_ms

    return "no_stimulus", None


def group_trials_by_condition(
    X_seqs: Sequence[np.ndarray],
    trial_info_list: List[Dict],
    stim_onset_frame: int = 1200,
    dt_ms: float = 4.0,
) -> Dict[str, List[int]]:
    """
    Group trial indices by their experimental condition.

    Uses trial info from events files for classification when available,
    falls back to kinematics-based detection.

    Args:
        X_seqs: List of feature arrays, each ``(T_i, 8)``.
        trial_info_list: List of trial info dicts from events.
        stim_onset_frame: Frame index of visual stimulus onset.
        dt_ms: Frame interval in milliseconds.

    Returns:
        Dictionary mapping condition name to list of trial indices.
    """
    condition_groups: Dict[str, List[int]] = {}

    for i, x_seq in enumerate(X_seqs):
        # Try to use trial info from events
        if i < len(trial_info_list):
            info = trial_info_list[i]
            trial_type = info.get('type', 'unknown')
            target_ttc_ms = info.get('target_ttc_ms')

            if trial_type in ('baseline_visual', 'visual_only'):
                condition = 'visual_only'
            elif trial_type in ('baseline_wind', 'wind_only'):
                condition = 'wind_only'
            elif trial_type in ('no_stimulus', 'silent'):
                condition = 'no_stimulus'
            elif trial_type in ('looming_wind', 'multisensory'):
                if target_ttc_ms is not None and np.isfinite(target_ttc_ms):
                    ttc_val = float(target_ttc_ms)
                    # Dynamic group name from the declared value itself —
                    # every observed TTC (e.g. -225/-261/-308) gets its own
                    # group; no hardcoded expected-condition list.
                    condition = format_ttc_condition_name(ttc_val)
                else:
                    condition = 'multisensory_other'
            else:
                # Fall back to kinematics-based detection
                condition, _ = classify_wind_condition(
                    x_seq, stim_onset_frame, dt_ms
                )
        else:
            # Fall back to kinematics-based detection
            condition, _ = classify_wind_condition(
                x_seq, stim_onset_frame, dt_ms
            )

        if condition not in condition_groups:
            condition_groups[condition] = []
        condition_groups[condition].append(i)

    # Log summary
    logger.info("Condition grouping summary:")
    for cond, indices in sorted(condition_groups.items()):
        logger.info("  %-35s: %d trials", cond, len(indices))

    return condition_groups


# ═══════════════════════════════════════════════════════════════
# 4.  Metric Extraction per Condition
# ═══════════════════════════════════════════════════════════════

def extract_predicted_metrics(
    y_preds: List[np.ndarray],
    trial_indices: List[int],
    dt_ms: float = 4.0,
    stim_onset_frame: int = 1200,
    search_window_frames: Optional[int] = None,
    pre_stim_window_frames: Optional[int] = None,
    anchor_frames: Optional[Sequence[int]] = None,
    reference_frames: Optional[Sequence[int]] = None,
) -> Dict[str, List[float]]:
    """
    Extract scalar metrics from predicted velocities for a set of trials.

    Computes per-trial:
        - **Peak Velocity (V_max):** Maximum absolute predicted velocity.
        - **Latency to Peak (T_max):** Time (ms) relative to the analysis
          reference frame when V_max is reached.

    The analysis reference is the *collision reference* for visual-containing
    conditions (visual collision), which differs from the crop anchor (which
    may be wind onset).  Pass ``reference_frames`` to override the crop
    anchor for latency computation; ``anchor_frames`` is used only for the
    search window when ``reference_frames`` is absent.

    Searches in physiologically relevant window around stimulus onset:
    [ref - pre_stim_window : ref + search_window] to capture both
    pre-collision escapes (biologically valid) and post-stimulus responses,
    while avoiding spurious peaks from distant baseline drift.

    Args:
        y_preds: List of predicted velocity arrays, each (T_i,).
        trial_indices: Indices of trials to analyze.
        dt_ms: Frame interval in milliseconds.
        stim_onset_frame: Fallback frame index of stimulus anchor.
        search_window_frames: Post-stimulus frames (default +4s from dt_ms).
        pre_stim_window_frames: Pre-stimulus frames (default -2s from dt_ms).
        anchor_frames: Optional list of per-trial crop anchor frame indices.
        reference_frames: Optional per-trial collision/analysis reference
            frames for latency.  When provided these override anchor_frames
            for the latency origin (search window still follows these refs).

    Returns:
        Dictionary with keys:
        - ``"peak_velocities"``: List of V_max values (cm/s).
        - ``"latencies"``: List of T_max values (ms).
    """
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be finite and positive")
    if search_window_frames is None:
        search_window_frames = math.ceil(4000.0 / dt_ms)
    if pre_stim_window_frames is None:
        pre_stim_window_frames = math.floor(2000.0 / dt_ms)

    peak_velocities: List[float] = []
    latencies: List[float] = []

    for trial_idx in trial_indices:
        y_pred = y_preds[trial_idx]

        if len(y_pred) == 0:
            continue

        if reference_frames is not None and trial_idx < len(reference_frames):
            anchor = int(reference_frames[trial_idx])
        elif anchor_frames is not None and trial_idx < len(anchor_frames):
            anchor = int(anchor_frames[trial_idx])
        else:
            anchor = stim_onset_frame

        # Search in window around stimulus: [onset - pre_window : onset + post_window]
        search_start = max(0, anchor - pre_stim_window_frames)
        search_end = min(len(y_pred), anchor + search_window_frames)

        if search_start >= len(y_pred) or search_end <= search_start:
            # No valid search window
            continue

        abs_velocity_window = np.abs(y_pred[search_start:search_end])
        if not np.all(np.isfinite(abs_velocity_window)):
            peak_velocities.append(float("nan"))
            latencies.append(float("nan"))
            continue

        v_max = float(np.max(abs_velocity_window))
        if v_max - float(np.min(abs_velocity_window)) < 1e-6:
            # ponytail: range rejects tonic output; calibrate prominence if drift yields false peaks.
            peak_velocities.append(v_max if v_max >= 1e-6 else 0.0)
            latencies.append(float("nan"))
            continue

        peak_in_window = int(np.argmax(abs_velocity_window))
        peak_frame = search_start + peak_in_window

        t_max = float((peak_frame - anchor) * dt_ms)

        peak_velocities.append(v_max)
        latencies.append(t_max)

    return {
        "peak_velocities": peak_velocities,
        "latencies": latencies,
    }


def compute_condition_statistics(
    metrics: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and SEM for each metric.

    Args:
        metrics: Dictionary from :func:`extract_predicted_metrics`.

    Returns:
        Dictionary with keys ``"latency"`` and ``"peak_velocity"``,
        each containing ``{"mean": float, "sem": float, "n": int}``.
    """
    stats: Dict[str, Dict[str, float]] = {}

    for metric_name, values in metrics.items():
        arr = np.array(values, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        n = len(valid)
        mean = float(np.mean(valid)) if n > 0 else None
        sem = float(np.std(valid, ddof=1) / np.sqrt(n)) if n > 1 else 0.0 if n else None

        # Map metric names
        if metric_name == "latencies":
            key = "latency"
        elif metric_name == "peak_velocities":
            key = "peak_velocity"
        else:
            key = metric_name

        stats[key] = {"mean": mean, "sem": sem, "n": n}

    return stats


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


def create_integration_figure(
    condition_stats: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    """
    Create the Lancet/Cell integration window figure.

    Layout: subplots(1, 2) — Chronometric Curve, Vigor Curve

    Args:
        condition_stats: Dictionary mapping condition name to
            statistics dict (from :func:`compute_condition_statistics`).
        output_path: Path to save the figure.
    """
    setup_lancet_style()

    # ── Prepare data for plotting ─────────────────────────────
    # Every observed condition is included. X coordinates for
    # multisensory_ttc_* groups are derived from the dynamic group name
    # itself (e.g. -225/-261/-308), not a hardcoded expected-condition
    # list. Conditions without a parseable TTC stay explicit (logged
    # skip reason) and are never given a fabricated x coordinate.
    observed = list(condition_stats.keys())
    skipped_conds: Dict[str, str] = {}

    x_values: List[float] = []
    latency_means: List[float] = []
    latency_sems: List[float] = []
    velocity_means: List[float] = []
    velocity_sems: List[float] = []
    cond_labels: List[str] = []

    for cond in observed:
        stats = condition_stats[cond]
        latency = stats.get("latency", {"mean": 0.0, "sem": 0.0, "n": 0})
        velocity = stats.get("peak_velocity", {"mean": 0.0, "sem": 0.0, "n": 0})
        if latency["n"] == 0:
            skipped_conds[cond] = "latency n==0"
            logger.warning("No data for condition '%s', skipping.", cond)
            continue

        if cond == "visual_only":
            # Visual-Only is the documented reference point at x=0 (no wind).
            delta_t = 0.0
        else:
            parsed = parse_ttc_condition_delta(cond)
            if parsed is None:
                skipped_conds[cond] = "no parseable delta_t from condition name"
                logger.warning(
                    "Condition %r has no parseable delta_t; omitted from "
                    "chronometric x-axis (explicit skip, not dropped silently).",
                    cond,
                )
                continue
            delta_t = float(parsed)

        x_values.append(delta_t)
        latency_means.append(latency["mean"])
        latency_sems.append(latency["sem"])
        velocity_means.append(velocity["mean"])
        velocity_sems.append(velocity["sem"])
        cond_labels.append(cond)

    if not x_values:
        raise ValueError("Phase F unavailable: no measurable peak latencies in observed conditions")

    # Convert to numpy arrays
    x_arr = np.array(x_values)
    latency_mean_arr = np.array(latency_means)
    latency_sem_arr = np.array(latency_sems)
    velocity_mean_arr = np.array(velocity_means)
    velocity_sem_arr = np.array(velocity_sems)

    # ── Get visual-only baseline values ───────────────────────
    # No fabricated 0.0 baseline when visual_only is absent: use NaN so
    # downstream isfinite() guards skip the reference line honestly.
    if "visual_only" in condition_stats:
        vis_stats = condition_stats["visual_only"]
        vis_latency = vis_stats.get("latency", {})
        vis_velocity = vis_stats.get("peak_velocity", {})
        vis_latency_baseline = float(vis_latency["mean"]) if vis_latency.get("n", 0) > 0 else float("nan")
        vis_velocity_baseline = float(vis_velocity["mean"]) if vis_velocity.get("n", 0) > 0 else float("nan")
    else:
        vis_latency_baseline = float("nan")
        vis_velocity_baseline = float("nan")
        logger.warning("Visual-only condition not found for baseline reference.")

    # ── Create figure with [1, 2] layout ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))

    panel_labels = ["A", "B"]

    # ══════════════════════════════════════════════════════════
    # Panel A: Chronometric Curve (Latency)
    # ══════════════════════════════════════════════════════════
    ax_a = axes[0]

    # Plot data points with error bars
    ax_a.errorbar(
        x_arr, latency_mean_arr, yerr=latency_sem_arr,
        color=PRIMARY_LINE_COLOR,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        capsize=ERROR_BAR_CAPSIZE,
        capthick=1.5,
        elinewidth=1.5,
        solid_capstyle="round",
        label="Predicted Latency",
        zorder=3,
    )

    # Add horizontal baseline for visual-only
    if np.isfinite(vis_latency_baseline):
        ax_a.axhline(
            y=vis_latency_baseline,
            color=BASELINE_COLOR,
            linewidth=BASELINE_LINEWIDTH,
            linestyle=BASELINE_LINESTYLE,
            label="Visual-Only Baseline",
            zorder=2,
        )

    # Axes styling
    ax_a.set_xlabel(
        r"Wind Onset Time relative to TTC ($\Delta T$ ms)",
        fontsize=FONT_SIZE_AXIS_TITLE,
        color=AXIS_COLOR,
    )
    ax_a.set_ylabel(
        "Latency to Peak Velocity (ms)",
        fontsize=FONT_SIZE_AXIS_TITLE,
        color=AXIS_COLOR,
    )

    # Tick formatting
    ax_a.tick_params(axis="both", colors=AXIS_COLOR, width=1.5)

    # Spine styling
    for spine in ax_a.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(1.5)

    # Grid: ultra-faint major grid lines
    ax_a.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)

    # Legend
    ax_a.legend(
        loc="upper left",
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        facecolor=BACKGROUND_COLOR,
        edgecolor=AXIS_COLOR,
        framealpha=1.0,
    )

    # Panel label
    ax_a.set_title(
        f"{panel_labels[0]}  Chronometric Curve",
        fontsize=FONT_SIZE_PANEL_LABEL,
        fontweight="bold",
        color=AXIS_COLOR,
        loc="left",
    )

    # ══════════════════════════════════════════════════════════
    # Panel B: Vigor Curve (Peak Velocity)
    # ══════════════════════════════════════════════════════════
    ax_b = axes[1]

    # Plot data points with error bars
    ax_b.errorbar(
        x_arr, velocity_mean_arr, yerr=velocity_sem_arr,
        color=PRIMARY_LINE_COLOR,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        capsize=ERROR_BAR_CAPSIZE,
        capthick=1.5,
        elinewidth=1.5,
        solid_capstyle="round",
        label="Predicted Peak Velocity",
        zorder=3,
    )

    # Add horizontal baseline for visual-only
    if vis_velocity_baseline > 0:
        ax_b.axhline(
            y=vis_velocity_baseline,
            color=BASELINE_COLOR,
            linewidth=BASELINE_LINEWIDTH,
            linestyle=BASELINE_LINESTYLE,
            label="Visual-Only Baseline",
            zorder=2,
        )

    # Axes styling
    ax_b.set_xlabel(
        r"Wind Onset Time relative to TTC ($\Delta T$ ms)",
        fontsize=FONT_SIZE_AXIS_TITLE,
        color=AXIS_COLOR,
    )
    ax_b.set_ylabel(
        "Peak Velocity (cm/s)",
        fontsize=FONT_SIZE_AXIS_TITLE,
        color=AXIS_COLOR,
    )

    # Tick formatting
    ax_b.tick_params(axis="both", colors=AXIS_COLOR, width=1.5)

    # Spine styling
    for spine in ax_b.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(1.5)

    # Grid: ultra-faint major grid lines
    ax_b.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)

    # Legend
    ax_b.legend(
        loc="upper left",
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        facecolor=BACKGROUND_COLOR,
        edgecolor=AXIS_COLOR,
        framealpha=1.0,
    )

    # Panel label
    ax_b.set_title(
        f"{panel_labels[1]}  Vigor Curve",
        fontsize=FONT_SIZE_PANEL_LABEL,
        fontweight="bold",
        color=AXIS_COLOR,
        loc="left",
    )

    # ── Suptitle ──────────────────────────────────────────────
    fig = ax_a.get_figure()
    fig.suptitle(
        "Multisensory Integration Window — Whole Corpus, In-Sample Descriptive Predictions",
        fontsize=14,
        fontweight="bold",
        color=AXIS_COLOR,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # ── Save ──────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    logger.info("Saved integration window figure to %s (%d DPI)", output_path, DPI)

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 6.  Statistical Summary Export
# ═══════════════════════════════════════════════════════════════

def export_integration_summary(
    condition_stats: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
    dt_ms: float = 4.0,
    analysis_population: Optional[Dict[str, object]] = None,
) -> None:
    """
    Export integration analysis summary to JSON.

    Args:
        condition_stats: Dictionary mapping condition name to
            statistics dict (from :func:`compute_condition_statistics`).
        output_path: Path to save the JSON file.
        dt_ms: Frame interval in milliseconds (nominal 250 Hz = 4.0 ms).
    """
    # Build summary structure
    summary: Dict[str, Any] = {
        "analysis": "Multisensory Integration Window",
        "phase": 9,
        "dt_ms": dt_ms,
        "analysis_population": analysis_population,
        "status": ("ok" if any(stats.get("latency", {}).get("n", 0) > 0
                             for stats in condition_stats.values()) else "unavailable"),
        "description": (
            "%s chronometric and vigor curves of predicted responses "
            "grouped by observed wind timing; no group difference is inferred."
            % (analysis_population["evidence_scope"].replace("_", " ")
               if analysis_population is not None else "Population-unavailable descriptive"),
        ),
        "inference": {
            "status": "descriptive_only",
            "effect_sizes": None,
            "adjusted_p_values": None,
            "reason": "Grouped trials may share animals; no valid grouped effect test or multiplicity correction was run.",
        },
        "conditions": {},
    }

    # Add condition-specific data. delta_t_ms is derived from dynamic
    # group names (every observed TTC value, e.g. -225/-261/-308);
    # non-TTC conditions stay explicitly None rather than hardcoded
    # expected-condition results.
    for cond_name, stats in condition_stats.items():
        display_name = CONDITION_DISPLAY_NAMES.get(cond_name, cond_name)
        if cond_name == "visual_only":
            delta_t: Optional[float] = None
        else:
            delta_t = parse_ttc_condition_delta(cond_name)

        cond_data: Dict[str, Any] = {
            "display_name": display_name,
            "delta_t_ms": delta_t,
            "latency_to_peak_ms": stats.get("latency", {"mean": None, "sem": None, "n": 0}),
            "peak_velocity_cms": stats.get("peak_velocity", {"mean": None, "sem": None, "n": 0}),
        }

        summary["conditions"][cond_name] = cond_data

    # Add baseline reference
    vis_stats = condition_stats.get("visual_only", {})
    vis_latency = vis_stats.get("latency", {})
    vis_velocity = vis_stats.get("peak_velocity", {})
    summary["baseline_reference"] = (
        {"condition": "visual_only", "latency_mean_ms": vis_latency["mean"],
         "peak_velocity_mean_cms": vis_velocity["mean"] if vis_velocity.get("n", 0) > 0 else None}
        if vis_latency.get("n", 0) > 0 else None
    )

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)

    logger.info("Saved integration summary to %s", output_path)


# ═══════════════════════════════════════════════════════════════
# 7.  Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════

def run_integration_analysis(
    checkpoint_path: Path,
    dataset_path: Path,
    output_path: Path,
    summary_path: Optional[Path] = None,
    batch_size: int = 32,
    max_seq_len: Optional[int] = 2400,
    pre_anchor_frames: int = 1200,
    dt_ms: Optional[float] = None,
    stim_onset_frame: int = 1200,
    raw_dir: Optional[Path] = None,
    nested_prior_artifact: Optional[Path] = None,
    qc_sealed_nested_prior_sha256: Optional[str] = None,
    trusted_historical_checkpoint_sha256: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
) -> None:
    """
    Run the full multisensory integration window analysis.

    Args:
        checkpoint_path: Path to the trained model checkpoint.
        dataset_path: Path to the preprocessed dataset.
        output_path: Path to save the integration window figure.
        summary_path: Path to save the JSON summary. If None,
            defaults to ``results/integration_summary.json``.
        batch_size: Batch size for data loading.
        max_seq_len: Maximum sequence length for cropping.
        pre_anchor_frames: Baseline frames before anchor.
        dt_ms: None inherits saved model.dt_ms; an explicit interval must match.
        stim_onset_frame: Frame index of stimulus onset / anchor.
        raw_dir: Optional path to raw session data for declared events.
    """
    logger.info("=" * 60)
    logger.info("NSMoR Multisensory Integration Window Analysis (Phase 9)")
    logger.info("=" * 60)

    # ── Default paths ─────────────────────────────────────────
    if summary_path is None:
        summary_path = Path("results/integration_summary.json")

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────
    model = load_model_from_checkpoint(checkpoint_path, device)
    dt_ms = resolve_dt_ms(model, dt_ms)

    # ── Load dataset ──────────────────────────────────────────
    dataloader, labels, lengths_list, X_seqs, trial_info_list = load_dataset(
        dataset_path,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        nested_prior_artifact=nested_prior_artifact,
        qc_sealed_nested_prior_sha256=qc_sealed_nested_prior_sha256,
        checkpoint_model=model,
        trusted_historical_checkpoint_sha256=trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
        raw_dir=raw_dir,
    )

    # ── Run model inference ───────────────────────────────────
    logger.info("Running descriptive inference on complete observed corpus (nested priors when provided)...")
    y_preds: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            X_batch, _Y_batch, lengths = batch
            X_batch = X_batch.to(device).contiguous()
            lengths = lengths.to(device).contiguous()

            Y_pred = model(X_batch, lengths)

            B, T = Y_pred.shape

            for i in range(B):
                length_i = int(lengths[i].item())
                y_pred_i = Y_pred[i, :length_i].cpu().numpy()
                y_preds.append(y_pred_i)

    logger.info("Collected predictions for %d trials.", len(y_preds))

    # ── Group trials by condition ─────────────────────────────
    logger.info("-" * 60)
    logger.info("Grouping trials by experimental condition...")
    condition_groups = group_trials_by_condition(
        X_seqs, trial_info_list, stim_onset_frame, dt_ms
    )

    # ── Extract metrics per condition ─────────────────────────
    logger.info("-" * 60)
    logger.info("Extracting predicted metrics per condition...")
    condition_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    raw_anchor_frames = [
        int(info.get("anchor_frame", stim_onset_frame)) for info in trial_info_list
    ]

    # Analysis reference: visual-containing conditions compare relative to
    # the same visual collision; wind-only uses wind onset (separately named).
    # Crop wind anchor != collision reference — these are distinct frames.
    # Map both references into SAVED (post-crop) sequence coordinates so
    # latency matches the scored prediction tensors (Finding A1).
    saved_anchor_frames: List[int] = []
    saved_collision_refs: List[int] = []
    for idx in range(len(X_seqs)):
        x_raw = X_seqs[idx]
        L_raw = len(x_raw)
        raw_anchor = raw_anchor_frames[idx] if idx < len(raw_anchor_frames) else stim_onset_frame
        start, end = resolve_anchor_crop(
            n_frames=L_raw,
            anchor_frame=raw_anchor,
            max_seq_len=max_seq_len,
            pre_anchor_frames=pre_anchor_frames,
        )
        saved_anchor = raw_anchor - start
        saved_anchor = min(max(0, saved_anchor), max(end - start - 1, 0))
        saved_anchor_frames.append(saved_anchor)

        # Visual collision in cropped coordinates
        x_cropped = x_raw[start:end]
        vis = np.abs(np.asarray(x_cropped)[:, 0])
        if vis.size > 0 and np.any(vis > 1e-4):
            collision_ref = int(np.argmax(vis))
        else:
            collision_ref = saved_anchor
        saved_collision_refs.append(collision_ref)

    for cond_name, trial_indices in condition_groups.items():
        logger.info("  Condition: %s (%d trials)", cond_name, len(trial_indices))

        # Visual-containing conditions share the visual-collision reference.
        # Wind-only uses wind onset (crop anchor) and is labelled separately.
        is_visual_containing = cond_name != "wind_only"
        full_refs = saved_collision_refs if is_visual_containing else saved_anchor_frames

        # Extract per-trial metrics
        metrics = extract_predicted_metrics(
            y_preds=y_preds,
            trial_indices=trial_indices,
            dt_ms=dt_ms,
            stim_onset_frame=stim_onset_frame,
            anchor_frames=saved_anchor_frames,
            reference_frames=full_refs if is_visual_containing else None,
        )

        # Compute statistics
        stats = compute_condition_statistics(metrics)
        condition_stats[cond_name] = stats

        # Log summary
        if stats.get("latency", {}).get("n", 0) > 0:
            logger.info(
                "    Latency: %.1f ± %.1f ms (n=%d)",
                stats["latency"]["mean"],
                stats["latency"]["sem"],
                stats["latency"]["n"],
            )
        if stats.get("peak_velocity", {}).get("n", 0) > 0:
            logger.info(
                "    Peak Velocity: %.2f ± %.2f cm/s (n=%d)",
                stats["peak_velocity"]["mean"],
                stats["peak_velocity"]["sem"],
                stats["peak_velocity"]["n"],
            )

    # Persist unavailable status and null estimates before plotting; an all-flat
    # corpus has no measured latency curve and must fail loudly without a figure.
    export_integration_summary(
        condition_stats, summary_path, dt_ms=dt_ms,
        analysis_population=population_for_output(dataloader, len(labels)),
    )
    logger.info("Creating descriptive integration curves; no grouped effect test or adjusted p-values.")
    create_integration_figure(condition_stats, output_path)

    logger.info("=" * 60)
    logger.info("Integration window analysis complete!")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
# 8.  CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="NSMoR Multisensory Integration Window Analysis",
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
        default="results/integration_window.png",
        help="Output path for integration window figure.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="results/integration_summary.json",
        help="Output path for statistical summary JSON.",
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
        default=1200,
        help="Frame index of stimulus onset / anchor (default 1200).",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=None,
        help="Optional directory with declared raw events CSVs for condition lookup.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    pre_anchor_frames = getattr(args, "pre_anchor_frames", 1200)
    raw_dir_path = Path(args.raw_dir) if getattr(args, "raw_dir", None) else None
    run_integration_analysis(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        nested_prior_artifact=Path(args.nested_prior_artifact) if args.nested_prior_artifact else None,
        qc_sealed_nested_prior_sha256=args.qc_sealed_nested_prior_sha256,
        trusted_historical_checkpoint_sha256=args.trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=args.trusted_historical_artifact_sha256,
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        dt_ms=args.dt_ms,
        stim_onset_frame=args.stim_onset_frame,
        raw_dir=raw_dir_path,
    )


if __name__ == "__main__":
    main()
