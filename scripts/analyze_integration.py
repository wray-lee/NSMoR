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
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch

from nsmor.nsmor_dataloader import (
    NSMoRDataset,
    collate_variable_length,
)
from nsmor.checkpoint import load_checkpoint
from nsmor.config import DEFAULT_FEATURE, Label
from nsmor.model_nsmor_core import NSMoRCore
from nsmor.model_utils import load_model_from_checkpoint as _shared_load_model

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
    "multisensory_ttc_-373ms": "Wind at TTC−373ms",
    "multisensory_ttc_-119ms": "Wind at TTC−119ms",
    "multisensory_ttc_0ms": "Wind at TTC",
    "multisensory_ttc_+200ms": "Wind at TTC+200ms",
    "multisensory_other": "Other Multisensory",
}

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

    Delegates to the shared :func:`nsmor.model_utils.load_model_from_checkpoint`
    which guarantees all biophysical parameters are restored.
    """
    return _shared_load_model(checkpoint_path, device)


# ═══════════════════════════════════════════════════════════════
# 2.  Dataset Loading
# ═══════════════════════════════════════════════════════════════

def load_dataset(
    dataset_path: Path,
    batch_size: int = 32,
    max_seq_len: Optional[int] = 1000,
) -> Tuple[torch.utils.data.DataLoader, np.ndarray, List[int], np.ndarray, List[Dict]]:
    """
    Load the preprocessed dataset and create a DataLoader.

    Args:
        dataset_path: Path to ``nsmor_dataset.pt``.
        batch_size: Batch size for the DataLoader.

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
    dataset = torch.load(dataset_path, weights_only=False)

    X_seqs = dataset["X_seqs"]
    Y_seqs = dataset["Y_seqs"]
    mcmc_priors = dataset["mcmc_priors"]
    labels = dataset["labels"]
    lengths = dataset["lengths"]

    n_total = len(X_seqs)
    logger.info("Loaded %d sequences.", n_total)

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
    )

    dataloader = torch.utils.data.DataLoader(
        bio_dataset,
        batch_size=batch_size,
        shuffle=False,  # Preserve ordering for condition matching
        num_workers=0,
        collate_fn=collate_variable_length,
    )

    lengths_list = [int(l) for l in lengths]

    # Load trial info from events files
    trial_info_list = _load_trial_info_from_events(dataset_path.parent.parent / "raw")

    return dataloader, labels, lengths_list, X_seqs, trial_info_list


def _load_trial_info_from_events(raw_dir: Path) -> List[Dict]:
    """
    Load trial type info from events CSV files.

    Returns list of dicts with keys: type, target_ttc_ms, lv_ratio_ms
    """
    import json
    trial_info = []

    events_files = sorted(raw_dir.rglob("*_events.csv"))
    for evt_path in events_files:
        df = pd.read_csv(evt_path)
        for _, row in df.iterrows():
            event_type = str(row.get('event_type', row.get('event_name', '')))
            if event_type == 'trial_start':
                details_str = str(row.get('event_value', row.get('details', '{}')))
                try:
                    details = json.loads(details_str)
                except (json.JSONDecodeError, KeyError, ValueError):
                    details = {}
                trial_info.append({
                    'type': details.get('type', 'unknown'),
                    'target_ttc_ms': details.get('target_ttc_ms'),
                    'lv_ratio_ms': details.get('lv_ratio_ms'),
                })

    return trial_info


# ═══════════════════════════════════════════════════════════════
# 3.  Condition Grouping Logic
# ═══════════════════════════════════════════════════════════════

def detect_wind_onset_frame(
    x_seq: np.ndarray,
    stim_onset_frame: int = 200,
    wind_threshold: float = 0.5,
) -> Optional[int]:
    """
    Detect the first frame where wind stimulus is active.

    Args:
        x_seq: Single trial feature array, shape ``(T_i, 8)``.
            Index 1 is the wind state feature ``wind(t)``.
        stim_onset_frame: Frame index of visual stimulus onset.
        wind_threshold: Threshold to detect wind activation.

    Returns:
        Frame index of wind onset relative to sequence start,
        or ``None`` if no wind detected.

    Note:
        - For pure-wind trials (visual_angle ≡ 0), the sequence has
          570 frames prepended. Wind onset is detected in the original
          portion (after prepend).
        - The 0° visual baseline assumption means visual stimulus starts
          from frame ``stim_onset_frame`` and increases monotonically.
    """
    assert x_seq.ndim == 2 and x_seq.shape[1] == 8, (
        f"x_seq must be (T, 8), got {x_seq.shape}"
    )

    # Wind feature is at index 1
    wind_feature = x_seq[:, 1]

    # Look for first frame after stim_onset where wind > threshold
    post_stim_wind = wind_feature[stim_onset_frame:]

    if len(post_stim_wind) == 0:
        return None

    # Find first active wind frame
    active_frames = np.where(post_stim_wind > wind_threshold)[0]

    if len(active_frames) == 0:
        return None  # No wind detected

    # Return absolute frame index
    wind_onset_frame = stim_onset_frame + int(active_frames[0])
    return wind_onset_frame


def classify_wind_condition(
    x_seq: np.ndarray,
    stim_onset_frame: int = 200,
    dt_ms: float = 10.0,
    wind_threshold: float = 0.5,
) -> Tuple[str, Optional[float]]:
    """
    Classify the experimental condition based on wind onset timing.

    Args:
        x_seq: Single trial feature array, shape ``(T_i, 8)``.
        stim_onset_frame: Frame index of visual stimulus onset.
        dt_ms: Frame interval in milliseconds.
        wind_threshold: Threshold for wind detection.

    Returns:
        ``(condition_name, delta_t_ms)`` tuple where:
        - ``condition_name``: String identifier for the condition.
        - ``delta_t_ms``: Wind onset time relative to TTC (ms).
          Negative = wind before TTC, Positive = wind after TTC.
          ``None`` for visual-only or wind-only conditions.
    """
    # ── Check for visual stimulus ─────────────────────────────
    # Visual angle (index 0) should be non-zero post-stimulus for looming
    visual_feature = x_seq[:, 0]
    post_stim_visual = visual_feature[stim_onset_frame:]
    has_visual = bool(np.any(np.abs(post_stim_visual) > 1e-6))

    # ── Check for wind stimulus ───────────────────────────────
    wind_onset_frame = detect_wind_onset_frame(
        x_seq, stim_onset_frame, wind_threshold
    )
    has_wind = wind_onset_frame is not None

    # ── Classify condition ────────────────────────────────────
    if has_visual and not has_wind:
        return "visual_only", None

    if has_wind and not has_visual:
        return "wind_only", None

    if has_visual and has_wind:
        # Compute wind onset time relative to stimulus onset
        # ΔT = (wind_onset_frame - stim_onset_frame) * dt_ms
        delta_t_ms = float((wind_onset_frame - stim_onset_frame) * dt_ms)

        # Classify into specific multisensory conditions
        # Use tolerance bins for known experimental conditions
        if abs(delta_t_ms - (-373.0)) < 50.0:
            return "multisensory_ttc_-373ms", delta_t_ms
        elif abs(delta_t_ms - (-119.0)) < 50.0:
            return "multisensory_ttc_-119ms", delta_t_ms
        elif abs(delta_t_ms - 0.0) < 50.0:
            return "multisensory_ttc_0ms", delta_t_ms
        elif abs(delta_t_ms - 200.0) < 50.0:
            return "multisensory_ttc_+200ms", delta_t_ms
        else:
            return "multisensory_other", delta_t_ms

    # Fallback: no stimulus detected
    return "visual_only", None


def group_trials_by_condition(
    X_seqs: np.ndarray,
    trial_info_list: List[Dict],
    stim_onset_frame: int = 200,
    dt_ms: float = 10.0,
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

            if trial_type == 'baseline_visual':
                condition = 'visual_only'
            elif trial_type == 'baseline_wind':
                condition = 'wind_only'
            elif trial_type == 'looming_wind' and target_ttc_ms is not None:
                # Classify based on target_ttc_ms
                if abs(target_ttc_ms - (-373)) < 50:
                    condition = 'multisensory_ttc_-373ms'
                elif abs(target_ttc_ms - (-119)) < 50:
                    condition = 'multisensory_ttc_-119ms'
                elif abs(target_ttc_ms) < 50:
                    condition = 'multisensory_ttc_0ms'
                elif abs(target_ttc_ms - 200) < 50:
                    condition = 'multisensory_ttc_+200ms'
                else:
                    condition = f'multisensory_ttc_{target_ttc_ms:+.0f}ms'
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
    dt_ms: float = 10.0,
    stim_onset_frame: int = 200,
) -> Dict[str, List[float]]:
    """
    Extract scalar metrics from predicted velocities for a set of trials.

    Computes per-trial:
        - **Peak Velocity (V_max):** Maximum absolute predicted velocity
          in the post-stimulus window.
        - **Latency to Peak (T_max):** Time (ms) relative to stimulus
          onset when V_max is reached.

    Args:
        y_preds: List of predicted velocity arrays, each (T_i,).
        trial_indices: Indices of trials to analyze.
        dt_ms: Frame interval in milliseconds.
        stim_onset_frame: Frame index of stimulus onset.

    Returns:
        Dictionary with keys:
        - ``"peak_velocities"``: List of V_max values (cm/s).
        - ``"latencies"``: List of T_max values (ms).
    """
    peak_velocities: List[float] = []
    latencies: List[float] = []

    for trial_idx in trial_indices:
        y_pred = y_preds[trial_idx]

        # Ensure valid post-stimulus data
        if len(y_pred) <= stim_onset_frame:
            logger.debug(
                "Trial %d too short (len=%d) for stim_onset_frame=%d, skipping.",
                trial_idx, len(y_pred), stim_onset_frame,
            )
            continue

        # Extract post-stimulus predicted velocity
        post_pred = y_pred[stim_onset_frame:]

        if len(post_pred) == 0:
            continue

        # Peak Velocity: maximum absolute predicted velocity
        abs_velocity = np.abs(post_pred)
        v_max = float(np.max(abs_velocity))

        # Latency to Peak: time of V_max relative to stimulus
        peak_frame = int(np.argmax(abs_velocity))
        t_max = float(peak_frame * dt_ms)

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
        if not values:
            stats[metric_name] = {"mean": 0.0, "sem": 0.0, "n": 0}
            continue

        arr = np.array(values)
        n = len(arr)
        mean = float(np.mean(arr))
        sem = float(np.std(arr) / np.sqrt(n)) if n > 1 else 0.0

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
    # Filter conditions that have multisensory data with valid delta_t
    # We need conditions with known wind onset times
    plot_conditions = [
        "visual_only",
        "multisensory_ttc_-373ms",
        "multisensory_ttc_-119ms",
        "multisensory_ttc_0ms",
        "multisensory_ttc_+200ms",
    ]

    # X-axis values: Wind onset time relative to TTC (ms)
    delta_t_mapping: Dict[str, float] = {
        "visual_only": 0.0,  # Reference point (no wind)
        "multisensory_ttc_-373ms": -373.0,
        "multisensory_ttc_-119ms": -119.0,
        "multisensory_ttc_0ms": 0.0,
        "multisensory_ttc_+200ms": 200.0,
    }

    # Collect plot data
    x_values: List[float] = []
    latency_means: List[float] = []
    latency_sems: List[float] = []
    velocity_means: List[float] = []
    velocity_sems: List[float] = []

    for cond in plot_conditions:
        if cond not in condition_stats:
            logger.warning("Condition '%s' not found in data, skipping.", cond)
            continue

        stats = condition_stats[cond]

        # Get latency stats
        latency = stats.get("latency", {"mean": 0.0, "sem": 0.0, "n": 0})
        velocity = stats.get("peak_velocity", {"mean": 0.0, "sem": 0.0, "n": 0})

        if latency["n"] == 0:
            logger.warning("No data for condition '%s', skipping.", cond)
            continue

        delta_t = delta_t_mapping.get(cond, 0.0)
        x_values.append(delta_t)
        latency_means.append(latency["mean"])
        latency_sems.append(latency["sem"])
        velocity_means.append(velocity["mean"])
        velocity_sems.append(velocity["sem"])

    if not x_values:
        raise ValueError("No valid conditions found for plotting.")

    # Convert to numpy arrays
    x_arr = np.array(x_values)
    latency_mean_arr = np.array(latency_means)
    latency_sem_arr = np.array(latency_sems)
    velocity_mean_arr = np.array(velocity_means)
    velocity_sem_arr = np.array(velocity_sems)

    # ── Get visual-only baseline values ───────────────────────
    if "visual_only" in condition_stats:
        vis_stats = condition_stats["visual_only"]
        vis_latency_baseline = vis_stats.get("latency", {}).get("mean", 0.0)
        vis_velocity_baseline = vis_stats.get("peak_velocity", {}).get("mean", 0.0)
    else:
        vis_latency_baseline = 0.0
        vis_velocity_baseline = 0.0
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
    if vis_latency_baseline > 0:
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
        "Multisensory Integration Window — Predicted Kinematics",
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
) -> None:
    """
    Export integration analysis summary to JSON.

    Args:
        condition_stats: Dictionary mapping condition name to
            statistics dict (from :func:`compute_condition_statistics`).
        output_path: Path to save the JSON file.
    """
    # Build summary structure
    summary: Dict[str, Any] = {
        "analysis": "Multisensory Integration Window",
        "phase": 9,
        "description": (
            "Chronometric and vigor curves showing wind timing "
            "modulation of visual looming response predictions."
        ),
        "conditions": {},
    }

    # Add condition-specific data
    delta_t_mapping: Dict[str, Optional[float]] = {
        "visual_only": None,
        "wind_only": None,
        "multisensory_ttc_-373ms": -373.0,
        "multisensory_ttc_-119ms": -119.0,
        "multisensory_ttc_0ms": 0.0,
        "multisensory_ttc_+200ms": 200.0,
        "multisensory_other": None,
    }

    for cond_name, stats in condition_stats.items():
        display_name = CONDITION_DISPLAY_NAMES.get(cond_name, cond_name)
        delta_t = delta_t_mapping.get(cond_name)

        cond_data: Dict[str, Any] = {
            "display_name": display_name,
            "delta_t_ms": delta_t,
            "latency_to_peak_ms": stats.get("latency", {"mean": 0.0, "sem": 0.0, "n": 0}),
            "peak_velocity_cms": stats.get("peak_velocity", {"mean": 0.0, "sem": 0.0, "n": 0}),
        }

        summary["conditions"][cond_name] = cond_data

    # Add baseline reference
    if "visual_only" in condition_stats:
        vis_stats = condition_stats["visual_only"]
        summary["baseline_reference"] = {
            "condition": "visual_only",
            "latency_mean_ms": vis_stats.get("latency", {}).get("mean", 0.0),
            "peak_velocity_mean_cms": vis_stats.get("peak_velocity", {}).get("mean", 0.0),
        }

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

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
    max_seq_len: Optional[int] = 1000,
    dt_ms: float = 10.0,
    stim_onset_frame: int = 200,
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
        dt_ms: Frame interval in milliseconds.
        stim_onset_frame: Frame index of stimulus onset.
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

    # ── Load dataset ──────────────────────────────────────────
    dataloader, labels, lengths_list, X_seqs, trial_info_list = load_dataset(
        dataset_path, batch_size=batch_size, max_seq_len=max_seq_len,
    )

    # ── Run model inference ───────────────────────────────────
    logger.info("Running model inference on validation set...")
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

    for cond_name, trial_indices in condition_groups.items():
        logger.info("  Condition: %s (%d trials)", cond_name, len(trial_indices))

        # Extract per-trial metrics
        metrics = extract_predicted_metrics(
            y_preds=y_preds,
            trial_indices=trial_indices,
            dt_ms=dt_ms,
            stim_onset_frame=stim_onset_frame,
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

    # ── Create figure ─────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Creating integration window figure...")
    create_integration_figure(
        condition_stats=condition_stats,
        output_path=output_path,
    )

    # ── Export summary ────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Exporting statistical summary...")
    export_integration_summary(
        condition_stats=condition_stats,
        output_path=summary_path,
    )

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
        default=1000,
        help="Crop sequences longer than this (cuDNN compatibility). 0 = disable.",
    )
    parser.add_argument(
        "--dt_ms",
        type=float,
        default=10.0,
        help="Frame interval in milliseconds.",
    )
    parser.add_argument(
        "--stim_onset_frame",
        type=int,
        default=200,
        help="Frame index of stimulus onset (default 200 for 2s baseline at 10ms).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    run_integration_analysis(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
        dt_ms=args.dt_ms,
        stim_onset_frame=args.stim_onset_frame,
    )


if __name__ == "__main__":
    main()
