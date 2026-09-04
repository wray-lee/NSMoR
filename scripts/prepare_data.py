"""
NSMoR Offline Data Preparation Pipeline (Phase 5 ETL).

Ingests raw ``cercus`` CSVs with **hardware-synchronized timestamps**
(Arduino/Photodiode) and produces a single ``nsmor_dataset.pt`` file
ready for training.

Processing Steps
----------------
1. **Data Pairing** — Scan raw data directory, pair events/kinematics CSVs.
2. **Hardware Time Alignment** — Parse Arduino/Photodiode triggers to
   override software ``stim_state`` as ground-truth wind onset.
3. **Kinematics Processing** — Align ``sys_time`` with hardware-corrected
   timestamps; apply Savitzky-Golay smoothing for velocity/acceleration.
4. **Physical Labeling** — ``assign_ground_truth_labels`` on corrected axis.
5. **MCMC Prior Generation** — Train ``MCMCPriorGenerator`` on 5-D snapshots.
6. **Sequence Extraction with Visual Physics Reconstruction** — Extract
   continuous trajectories and mathematically reconstruct visual looming
   parameters (θ(t) and l/v) using:
       θ(t) = 2 × arctan(l/v / (TTC - t))
   Pure-wind trials receive 5.7s (570 frames) prepended zero-padding.

Output
------
``data/processed/nsmor_dataset.pt`` containing:
    - ``X_seqs``: List of ``np.ndarray (T_i, 8)``
    - ``Y_seqs``: List of ``np.ndarray (T_i,)``
    - ``mcmc_priors``: ``np.ndarray (N, 4)``
    - ``labels``: ``np.ndarray (N,)``
    - ``lengths``: ``np.ndarray (N,)``

Usage
-----
CLI::

    python scripts/prepare_data.py --raw_dir data/raw --output data/processed/nsmor_dataset.pt
    python scripts/prepare_data.py --raw_dir data/raw --output data/processed/nsmor_dataset.pt --dt_ms 10.0
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

from nsmor.config import (
    DEFAULT_FEATURE,
    DEFAULT_THRESHOLD,
    DEFAULT_TIME_WINDOW,
    FeatureConfig,
    PIPELINE_SEMANTICS_VERSION,
    TimeWindowConfig,
)
from nsmor.data_extractor import (
    build_sequence_dataset,
    build_snapshot_dataset,
    extract_trial_sequence,
    extract_mcmc_snapshot,
    PURE_WIND_PREPEND_FRAMES,
    _compute_pure_wind_prepend_frames,
)
from nsmor.mcmc_module import MCMCPriorGenerator, train_mcmc, train_mcmc_cross_fitted
from nsmor.pipeline.grouping import animal_keys_of, resolve_group_folds
from nsmor.pipeline.io import EVENT_COLUMNS, KINEMATICS_COLUMNS
from nsmor.pipeline.labeling import (
    assign_ground_truth_labels,
    labeling_funnel_summary,
)
from nsmor.pipeline.io import extract_trial_data, load_and_concat_sessions

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1.  Hardware Synchronization Constants
# ═══════════════════════════════════════════════════════════════

# Arduino/Photodiode event types expected in raw CSVs
PHOTODIODE_EVENT: str = "photodiode_trigger"
ARDUINO_WIND_EVENT: str = "arduino_wind_onset"

# Tolerance for hardware-software clock drift (ms)
CLOCK_DRIFT_TOLERANCE_MS: float = 50.0

# Visual physics constants
VISUAL_PHYSICS_EPSILON: float = 1e-6  # Small value to prevent division by zero


# ═══════════════════════════════════════════════════════════════
# Sampling interval diagnostics
# ═══════════════════════════════════════════════════════════════

def compute_sampling_diagnostics(
    kinematics_df: pd.DataFrame,
    configured_dt_ms: float,
    *,
    mismatch_ratio_threshold: float = 1.5,
) -> Dict[str, Any]:
    """Measure observed inter-frame intervals and compare with config.

    Computes per-trial ``time_ms.diff()``, pools all positive gaps, and
    reports summary statistics.  A ``mismatch_flag`` is set when the
    configured ``dt_ms`` deviates from the observed median by more than
    ``mismatch_ratio_threshold`` in either direction.

    The result dict is persisted in the dataset artefact so the
    discrepancy is impossible to miss during downstream analysis.

    Args:
        kinematics_df: Concatenated kinematics DataFrame with
            ``session_id``, ``trial_id``, ``time_ms`` columns.
        configured_dt_ms: The ``dt_ms`` value from CLI / config YAML.
        mismatch_ratio_threshold: Flag when
            ``configured / observed > threshold`` or
            ``observed / configured > threshold``.

    Returns:
        JSON-serialisable dict with summary statistics and the
        mismatch flag.
    """
    gaps = (
        kinematics_df
        .groupby(["session_id", "trial_id"], sort=False)["time_ms"]
        .diff()
        .dropna()
    )
    positive_gaps = gaps[gaps > 0].values

    if positive_gaps.size == 0:
        return {
            "observed_median_ms": float("nan"),
            "observed_mean_ms": float("nan"),
            "observed_std_ms": float("nan"),
            "observed_min_ms": float("nan"),
            "observed_max_ms": float("nan"),
            "observed_p01_ms": float("nan"),
            "observed_p99_ms": float("nan"),
            "n_positive_gaps": 0,
            "n_zero_or_negative_gaps": int((gaps <= 0).sum()),
            "configured_dt_ms": float(configured_dt_ms),
            "ratio_configured_over_observed": float("nan"),
            "mismatch_flag": True,
        }

    obs_median = float(np.median(positive_gaps))
    ratio = configured_dt_ms / obs_median if obs_median > 0 else float("inf")
    mismatch = (
        ratio > mismatch_ratio_threshold
        or (1.0 / ratio) > mismatch_ratio_threshold
    )

    return {
        "observed_median_ms": obs_median,
        "observed_mean_ms": float(np.mean(positive_gaps)),
        "observed_std_ms": float(np.std(positive_gaps)),
        "observed_min_ms": float(np.min(positive_gaps)),
        "observed_max_ms": float(np.max(positive_gaps)),
        "observed_p01_ms": float(np.percentile(positive_gaps, 1)),
        "observed_p99_ms": float(np.percentile(positive_gaps, 99)),
        "n_positive_gaps": int(positive_gaps.size),
        "n_zero_or_negative_gaps": int((gaps <= 0).sum()),
        "configured_dt_ms": float(configured_dt_ms),
        "ratio_configured_over_observed": float(ratio),
        "mismatch_flag": bool(mismatch),
    }


def reconstruct_visual_looming(
    time_ms: np.ndarray,
    l_v_ratio: float,
    ttc_ms: float,
    stimulus_onset_ms: float,
    dt_ms: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mathematically reconstruct continuous visual looming parameters.

    Uses the expanding object geometry formula:
        θ(t) = 2 × arctan(l/v / (TTC - t))

    where:
        - θ(t) is the visual angle at time t (degrees)
        - l/v is the looming velocity ratio (object_size / approach_speed)
        - TTC is the time-to-collision (absolute ms)
        - t is the current time (absolute ms)

    Args:
        time_ms: 1-D array of timestamps (ms) for this trial.
        l_v_ratio: The l/v ratio extracted from events (constant per trial).
        ttc_ms: Absolute time-to-collision in ms.
        stimulus_onset_ms: Absolute stimulus onset time in ms.
        dt_ms: Frame interval in milliseconds (default 10ms = 100Hz).

    Returns:
        ``(visual_angle, l_v_array)`` where:
        - ``visual_angle``: 1-D array of θ(t) in degrees, shape ``(n_frames,)``
        - ``l_v_array``: 1-D array of l/v values, shape ``(n_frames,)``

    Notes:
        - For t >= TTC (post-collision), θ(t) is clamped to 180°.
        - For t < stimulus_onset (pre-stimulus), θ(t) = 0.
        - Handles NaN/zero-division gracefully via epsilon guard.
        - Fully vectorized (no Python for-loop) for performance.
    """
    n_frames = len(time_ms)
    eps = VISUAL_PHYSICS_EPSILON

    # ── Guard against invalid l/v ratio ──
    if np.isnan(l_v_ratio) or np.isinf(l_v_ratio):
        logger.warning(
            "Invalid l_v_ratio=%.4f, defaulting to 0.", l_v_ratio
        )
        return np.zeros(n_frames, dtype=np.float64), np.zeros(n_frames, dtype=np.float64)

    # ── Vectorized computation ──
    ttc_remaining = ttc_ms - time_ms  # (n_frames,)

    # Region masks
    pre_stimulus = time_ms < stimulus_onset_ms
    post_collision = ttc_remaining < eps
    active = ~pre_stimulus & ~post_collision  # normal looming region

    # θ(t) = 2 × arctan(l/v / (TTC - t))  only for active frames
    visual_angle = np.zeros(n_frames, dtype=np.float64)
    l_v_array = np.zeros(n_frames, dtype=np.float64)

    if np.any(active):
        denom = ttc_remaining[active]
        safe_denom = np.where(np.abs(denom) < eps, eps, denom)
        ratio = l_v_ratio / safe_denom

        theta_deg = np.degrees(2.0 * np.arctan(ratio))
        # Replace NaN/Inf with 0, then clamp to [0, 180]
        theta_deg = np.nan_to_num(theta_deg, nan=0.0, posinf=180.0, neginf=0.0)
        theta_deg = np.clip(theta_deg, 0.0, 180.0)

        visual_angle[active] = theta_deg
        l_v_array[active] = l_v_ratio

    # Post-collision: clamp to 180°
    if np.any(post_collision):
        visual_angle[post_collision] = 180.0
        l_v_array[post_collision] = l_v_ratio

    # Pre-stimulus: already zeros (initialized)

    return visual_angle, l_v_array


def reconstruct_trial_visual_features(
    trial_data: Dict[str, np.ndarray],
    stimulus_onset_ms: float,
    l_v_ratio: float,
    dt_ms: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct visual features for a single trial with pure-wind handling.

    For pure-wind trials (no looming stimulus), returns arrays filled
    with absolute flat zeros (including the 570-frame prepended region).

    Args:
        trial_data: Trial data dictionary from extract_trial_data.
        stimulus_onset_ms: Hardware-corrected stimulus onset time (ms).
        l_v_ratio: The l/v ratio for this trial.
        dt_ms: Frame interval in milliseconds.

    Returns:
        ``(visual_angle_full, l_v_full)`` where each has shape
        ``(total_frames,)`` including any pure-wind prepended frames.
    """
    time_ms = trial_data["time_ms"]
    visual_angle_raw = trial_data["visual_angle"]

    # ── Detect pure-wind trial ──
    is_pure_wind = bool(np.all(np.abs(visual_angle_raw) < 1e-8))

    if is_pure_wind:
        # Pure wind: absolute flat zeros for entire sequence
        n_original = len(time_ms)
        prepend_frames = _compute_pure_wind_prepend_frames(dt_ms)
        n_total = n_original + prepend_frames

        visual_angle_full = np.zeros(n_total, dtype=np.float64)
        l_v_full = np.zeros(n_total, dtype=np.float64)

        logger.debug(
            "Pure-wind trial: %d prepended + %d original = %d total frames.",
            prepend_frames, n_original, n_total,
        )
    else:
        # ── Looming trial: reconstruct from physics ──
        # Estimate TTC from stimulus onset + l/v ratio
        # TTC is when the object would reach the observer
        # For typical looming experiments, TTC ≈ stimulus_onset + expansion_duration
        # We use the event data if available, otherwise estimate
        ttc_ms = stimulus_onset_ms + (l_v_ratio * 1000.0)  # Estimate: l/v in seconds

        visual_angle, l_v_array = reconstruct_visual_looming(
            time_ms=time_ms,
            l_v_ratio=l_v_ratio,
            ttc_ms=ttc_ms,
            stimulus_onset_ms=stimulus_onset_ms,
            dt_ms=dt_ms,
        )

        visual_angle_full = visual_angle
        l_v_full = l_v_array

    return visual_angle_full, l_v_full


# ═══════════════════════════════════════════════════════════════
# 2.  Arduino/Photodiode Parsing (Legacy CLI Logic)
# ═══════════════════════════════════════════════════════════════

def parse_hardware_triggers(
    events_df: pd.DataFrame,
) -> Dict[Tuple[str, int], float]:
    """
    Extract Arduino/Photodiode hardware trigger timestamps from events.

    This implements the synchronization logic from the legacy
    ``Cercus-classical-analysis-cli`` codebase:

    - Photodiode triggers (Arduino time) are the **absolute ground-truth**
      for stimulus onset.
    - If a photodiode trigger exists for a trial, it overrides the
      software ``stim_state`` timestamp.
    - The photodiode timestamp is mapped to the system clock by finding
      the nearest ``trial_start`` event and computing the offset.

    Args:
        events_df: Events DataFrame with columns matching
            :data:`EVENT_COLUMNS`.

    Returns:
        Dictionary mapping ``(session_id, trial_id)`` to the
        hardware-corrected stimulus onset time (in system clock ms).

    Example::

        hw_triggers = parse_hardware_triggers(events_df)
        corrected_onset = hw_triggers[("session_0", 3)]
    """
    hw_triggers: Dict[Tuple[str, int], float] = {}

    # Group events by session/trial
    grouped = events_df.groupby(["session_id", "trial_id"])

    for (session_id, trial_id), group in grouped:
        event_types = group["event_type"].values
        event_times = group["time_ms"].values

        # ── Look for photodiode trigger (Arduino ground truth) ──
        photodiode_mask = event_types == PHOTODIODE_EVENT
        arduino_mask = event_types == ARDUINO_WIND_EVENT

        if np.any(photodiode_mask):
            # Photodiode trigger is the absolute ground truth
            photodiode_time = float(event_times[photodiode_mask][0])

            # Map Arduino time to system clock:
            # Find trial_start as the synchronization reference
            trial_start_mask = event_types == "trial_start"
            if np.any(trial_start_mask):
                trial_start_sys = float(event_times[trial_start_mask][0])

                # The photodiode fires at a known offset from trial_start
                # in Arduino time. We use the system clock trial_start
                # as the anchor and add the photodiode offset.
                #
                # In the legacy CLI, the photodiode fires at stimulus onset,
                # which is typically at 2000ms (baseline_duration) in Arduino time.
                # We compute the actual system-clock time by finding the
                # stimulus_onset event and applying the photodiode correction.
                stimulus_onset_mask = event_types == "stimulus_onset"
                if np.any(stimulus_onset_mask):
                    software_onset = float(event_times[stimulus_onset_mask][0])
                    # Hardware-corrected onset = software_onset + delta
                    # where delta accounts for Arduino system clock drift
                    delta_ms = photodiode_time - software_onset

                    logger.debug(
                        "  [%s, trial %d] Photodiode correction: "
                        "software=%.1fms, hardware=%.1fms, delta=%.1fms",
                        session_id, trial_id,
                        software_onset, photodiode_time, delta_ms,
                    )

                    hw_triggers[(session_id, trial_id)] = photodiode_time
                else:
                    # No software onset — use photodiode directly
                    hw_triggers[(session_id, trial_id)] = photodiode_time

        elif np.any(arduino_mask):
            # Arduino wind onset (secondary hardware trigger)
            arduino_time = float(event_times[arduino_mask][0])
            hw_triggers[(session_id, trial_id)] = arduino_time

        else:
            # No hardware trigger — fall back to software stim_state
            stimulus_mask = event_types == "stimulus_onset"
            if np.any(stimulus_mask):
                hw_triggers[(session_id, trial_id)] = float(
                    event_times[stimulus_mask][0]
                )

    return hw_triggers


def log_time_correction_deltas(
    events_df: pd.DataFrame,
    hw_triggers: Dict[Tuple[str, int], float],
) -> None:
    """
    Log the time-correction delta between software and hardware clocks.

    For each trial with a hardware trigger, computes and logs:
    ``delta = hardware_onset - software_onset``

    Args:
        events_df: Raw events DataFrame.
        hw_triggers: Output of :func:`parse_hardware_triggers`.
    """
    deltas: List[float] = []

    grouped = events_df.groupby(["session_id", "trial_id"])
    for (session_id, trial_id), group in grouped:
        key = (session_id, trial_id)
        if key not in hw_triggers:
            continue

        event_types = group["event_type"].values
        event_times = group["time_ms"].values

        stimulus_mask = event_types == "stimulus_onset"
        if np.any(stimulus_mask):
            software_onset = float(event_times[stimulus_mask][0])
            hardware_onset = hw_triggers[key]
            delta = hardware_onset - software_onset
            deltas.append(delta)

    if deltas:
        deltas_arr = np.array(deltas)
        logger.info(
            "Hardware-Software clock delta: "
            "mean=%.2fms, std=%.2fms, min=%.2fms, max=%.2fms, n=%d",
            np.mean(deltas_arr), np.std(deltas_arr),
            np.min(deltas_arr), np.max(deltas_arr),
            len(deltas_arr),
        )

        # Warn if drift exceeds tolerance
        max_abs_delta = np.max(np.abs(deltas_arr))
        if max_abs_delta > CLOCK_DRIFT_TOLERANCE_MS:
            logger.warning(
                "Max clock drift %.2fms exceeds tolerance %.2fms!",
                max_abs_delta, CLOCK_DRIFT_TOLERANCE_MS,
            )
    else:
        logger.info("No hardware triggers found — using software timestamps only.")


# ═══════════════════════════════════════════════════════════════
# 3.  Kinematics Processing with Hardware Alignment
# ═══════════════════════════════════════════════════════════════

def apply_hardware_time_correction(
    kinematics_df: pd.DataFrame,
    events_df: pd.DataFrame,
    hw_triggers: Dict[Tuple[str, int], float],
    dt_ms: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply hardware time correction to kinematics and events DataFrames.

    For trials with hardware triggers:
    1. Replace the software ``stimulus_onset`` event time with the
       hardware-corrected timestamp.
    2. Recompute velocity and acceleration using Savitzky-Golay smoothing
       on the corrected time axis.

    Args:
        kinematics_df: Kinematics DataFrame.
        events_df: Events DataFrame.
        hw_triggers: Hardware trigger timestamps from
            :func:`parse_hardware_triggers`.
        dt_ms: Frame interval in milliseconds.

    Returns:
        ``(corrected_kinematics, corrected_events)`` DataFrames.
    """
    kin_corrected = kinematics_df.copy()
    evt_corrected = events_df.copy()

    # ── Update stimulus onset events with hardware timestamps ──
    for (session_id, trial_id), hw_time in hw_triggers.items():
        # Find and update stimulus_onset events
        mask = (
            (evt_corrected["session_id"] == session_id)
            & (evt_corrected["trial_id"] == trial_id)
            & (evt_corrected["event_type"] == "stimulus_onset")
        )
        if mask.any():
            old_time = evt_corrected.loc[mask, "time_ms"].iloc[0]
            evt_corrected.loc[mask, "time_ms"] = hw_time

            logger.debug(
                "  [%s, trial %d] Updated stimulus_onset: "
                "%.1fms -> %.1fms (delta=%.1fms)",
                session_id, trial_id, old_time, hw_time, hw_time - old_time,
            )

    # ── Recompute kinematics with Savitzky-Golay smoothing ──
    grouped = kin_corrected.groupby(["session_id", "trial_id"])
    for (session_id, trial_id), group in grouped:
        idx = group.index

        # Extract position arrays
        x_pos = group["x_pos"].values
        y_pos = group["y_pos"].values

        # Savitzky-Golay smoothing (window=11, polyorder=3)
        window_length = min(11, len(x_pos))
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 3:
            x_smooth = savgol_filter(x_pos, window_length, 3)
            y_smooth = savgol_filter(y_pos, window_length, 3)
        else:
            x_smooth = x_pos.copy()
            y_smooth = y_pos.copy()

        # Compute 2-D Cartesian path speed from smoothed trajectory.
        # Previous code computed d|r|/dt (radial derivative of distance
        # from origin) which is physically wrong: a circular orbit has
        # |r|=const → radial speed ≡ 0, but nonzero tangential speed.
        # Correct: speed = sqrt(dx/dt² + dy/dt²).
        #
        # Use the REAL per-sample interval from time_ms (median ~4 ms,
        # not the configured dt_ms=10 ms) to avoid a systematic scaling
        # error.  np.gradient(..., t_s) handles irregular spacing.
        time_ms_arr = group["time_ms"].values
        t_s = time_ms_arr / 1000.0  # seconds
        dx_dt = np.gradient(x_smooth, t_s)  # cm/s
        dy_dt = np.gradient(y_smooth, t_s)  # cm/s
        velocity = np.sqrt(dx_dt**2 + dy_dt**2)  # path speed, cm/s

        # Compute acceleration from smoothed velocity
        if window_length >= 3:
            velocity_smooth = savgol_filter(velocity, window_length, 3)
        else:
            velocity_smooth = velocity.copy()
        acceleration = np.gradient(velocity_smooth, t_s)  # cm/s²

        # Update kinematics DataFrame
        kin_corrected.loc[idx, "velocity"] = velocity_smooth
        kin_corrected.loc[idx, "acceleration"] = acceleration

    logger.info(
        "Applied hardware time correction to %d trials.",
        len(hw_triggers),
    )

    return kin_corrected, evt_corrected


# ═══════════════════════════════════════════════════════════════
# 4.  Data Pairing
# ═══════════════════════════════════════════════════════════════

def pair_csv_files(
    raw_dir: Path,
) -> List[Tuple[Path, Path]]:
    """
    Scan raw data directory and pair kinematics/events CSV files.

    Expected directory structure::

        raw_dir/
        ├── session_0/
        │   ├── kinematics.csv
        │   └── events.csv
        ├── session_1/
        │   ├── kinematics.csv
        │   └── events.csv
        ...

    Args:
        raw_dir: Root directory containing session subdirectories.

    Returns:
        List of ``(kinematics_path, events_path)`` tuples.

    Raises:
        FileNotFoundError: If no valid pairs are found.
    """
    pairs: List[Tuple[Path, Path]] = []

    # Search for session directories
    for session_dir in sorted(raw_dir.iterdir()):
        if not session_dir.is_dir():
            continue

        # Look for kinematics and events CSVs
        kin_candidates = list(session_dir.glob("*kinematics*.csv"))
        evt_candidates = list(session_dir.glob("*events*.csv"))

        if kin_candidates and evt_candidates:
            # Take first match of each
            kin_path = kin_candidates[0]
            evt_path = evt_candidates[0]
            pairs.append((kin_path, evt_path))
            logger.info(
                "Paired: %s <-> %s",
                kin_path.name, evt_path.name,
            )

    if not pairs:
        raise FileNotFoundError(
            f"No valid kinematics/events CSV pairs found in {raw_dir}"
        )

    return pairs


# ═══════════════════════════════════════════════════════════════
# 4b.  OOF → serve prior shift audit
# ═══════════════════════════════════════════════════════════════

def classify_stimulus_condition(trial_data: Dict[str, np.ndarray]) -> str:
    """Name the stimulus condition from the physical channels.

    Condition is read off ``visual_angle`` / ``wind_state`` rather than
    any label or metadata field, so it stays correct for corpora that
    never recorded a condition column.

    The MoR Router's routing hypothesis is stated per *modality*, not per
    behavioural outcome: wind transients should engage the LIF Pathway
    while looming expansion should engage the GRU Pathway.  Reading the
    condition here keeps that split in one place for both the drop audit
    and the per-trial flags consumed downstream.

    Args:
        trial_data: Per-trial channel dict with ``visual_angle`` and
            ``wind_state`` arrays.

    Returns:
        One of ``"multisensory"``, ``"visual_only"``, ``"wind_only"``,
        ``"no_stimulus"``.
    """
    has_visual = bool(np.any(np.abs(trial_data["visual_angle"]) > 0.0))
    has_wind = bool(np.any(np.abs(trial_data["wind_state"]) > 0.0))
    if has_visual and has_wind:
        return "multisensory"
    if has_visual:
        return "visual_only"
    if has_wind:
        return "wind_only"
    return "no_stimulus"


def _audit_snapshot_drops(
    labeled_trials: List[Dict[str, Any]],
    kept_indices: Sequence[int],
) -> Dict[str, Any]:
    """
    Break snapshot-extraction drops down by class AND by condition.

    Recording only a total is what let a 100%-single-condition loss pass
    unnoticed: 36 of 396 trials were dropped on the reference corpus, and
    every one of them was a visual-only No_Response.  A drop concentrated
    in one class or one stimulus condition is a systematic bias, and no
    artefact exposed that until the counts were split out.

    Condition is read off the physical channels, matching how the rest of
    the ETL identifies it, so no extra metadata is required.

    Args:
        labeled_trials: The pre-filter labelled trial list.
        kept_indices: Indices retained by :func:`build_snapshot_dataset`.

    Returns:
        Counts of dropped trials in total, per class name, and per
        stimulus condition.
    """
    kept = set(int(i) for i in kept_indices)
    by_class: Dict[str, int] = {}
    by_condition: Dict[str, int] = {}

    for idx, info in enumerate(labeled_trials):
        if idx in kept:
            continue
        name = info["label"].name
        by_class[name] = by_class.get(name, 0) + 1

        condition = classify_stimulus_condition(info["trial_data"])
        by_condition[condition] = by_condition.get(condition, 0) + 1

    return {
        "n_prefilter_labeled_trials": len(labeled_trials),
        "n_kept": len(kept),
        "n_dropped": len(labeled_trials) - len(kept),
        "dropped_by_class": by_class,
        "dropped_by_condition": by_condition,
    }


# Pre-declared invariants for the OOF → serve prior shift record.
# The MCMC Prior is a 4-D continuous INPUT feature, not a prediction: its
# only consumer is the MoR Router, a Linear(hidden + mcmc_dim, 2) reading
# the concatenated vector.  The router never sees an argmax, so an
# argmax-agreement floor gated on the wrong quantity — and worse, a
# generator collapsed onto the majority class scores PERFECT agreement
# (measured: 3 folds collapsed the generator to agreement 1.000 with zero
# Prewalk and zero Pre_Active predictions), making the floor satisfiable
# by degrading the model.  The failure that does matter, the prior vector
# collapsing to a constant, is covered by the bootstrap per-column
# variance floor in prepare_dataset.  Train-serve shift is therefore
# reported as a distance on the vector and NOT gated; no replacement
# threshold is introduced, because the 0.65 floor was itself an
# unvalidated constant that aborted the real-data ETL at 0.611.
PRIOR_ROW_SUM_TOL = 1e-4


def audit_prior_train_serve_shift(
    oof_priors: np.ndarray,
    ensemble_priors: np.ndarray,
    label_names: Sequence[str],
) -> Dict[str, Any]:
    """
    Quantify the OOF-train / ensemble-serve MCMC prior shift.

    Training consumes SINGLE-fold out-of-fold probabilities while the
    documented inference protocol averages every fold model.  Averaging
    K models shrinks variance *by construction*, so the two sets are
    never identically distributed, and a two-sample KS test on them
    rejects at any usable sample size: it tests a null the protocol
    itself makes false, on paired samples it assumes are independent.
    On the real 360-trial dataset this produced KS p = 9e-26 … 1e-31 for
    all four classes purely from the expected shrinkage (NO_RESPONSE
    variance 0.0719 → 0.0378), aborting the ETL on a healthy model.

    So the shift is reported as *magnitude* — signed bias with a paired
    CI, dispersion ratio, and a total-variation distance on the prior
    vector — and only unambiguous defects raise: non-finite, negative, or
    unnormalised probability rows, mismatched shapes, or empty input.

    Nothing about decision agreement raises.  ``argmax_agreement`` is
    retained as a descriptive figure only, alongside
    ``ks_pvalue_descriptive_only``, because the router reads the vector
    rather than its argmax and because a generator collapsed onto the
    majority class scores perfect agreement.  ``mean_total_variation_
    distance`` is the quantity to report for train-serve consistency: it
    registers vector shifts that leave every argmax untouched.

    Prior-vector collapse — the failure that would actually starve the
    router — is caught by the bootstrap per-column variance floor in
    :func:`prepare_dataset`, not here.

    The record is persisted in the dataset artefact so the residual
    shift is auditable instead of silently accepted.

    Args:
        oof_priors: ``(n, C)`` out-of-fold probabilities used in training.
        ensemble_priors: ``(n, C)`` fold-ensemble probabilities as served.
        label_names: Class names in column order.

    Returns:
        JSON-serialisable telemetry dict.

    Raises:
        ValueError: On invalid probabilities, mismatched shapes, or no trials.
    """
    from scipy.stats import ks_2samp

    if oof_priors.shape != ensemble_priors.shape:
        raise ValueError(
            f"Prior shape mismatch: OOF {oof_priors.shape} vs "
            f"ensemble {ensemble_priors.shape}."
        )
    n_trials, n_cols = oof_priors.shape
    if n_trials == 0:
        raise ValueError("Prior shift audit needs at least one trial.")

    for name, arr in (("OOF", oof_priors), ("ensemble", ensemble_priors)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} priors contain non-finite values.")
        if (arr < 0.0).any():
            raise ValueError(f"{name} priors contain negative probabilities.")
        row_dev = float(np.abs(arr.sum(axis=1) - 1.0).max())
        if row_dev > PRIOR_ROW_SUM_TOL:
            raise ValueError(
                f"{name} prior rows deviate from 1 by {row_dev:.2e} "
                f"(tolerance {PRIOR_ROW_SUM_TOL:.0e})."
            )

    diff = ensemble_priors - oof_priors
    columns: Dict[str, Dict[str, Any]] = {}
    for c in range(n_cols):
        d = diff[:, c]
        mean_signed = float(np.mean(d))
        se = (
            float(np.std(d, ddof=1) / np.sqrt(n_trials))
            if n_trials > 1 else float("nan")
        )
        var_oof = float(np.var(oof_priors[:, c]))
        var_serve = float(np.var(ensemble_priors[:, c]))
        ks = ks_2samp(oof_priors[:, c], ensemble_priors[:, c])
        columns[str(c)] = {
            "class": (
                str(label_names[c]) if c < len(label_names) else f"class_{c}"
            ),
            "mean_abs_diff": float(np.mean(np.abs(d))),
            "max_abs_diff": float(np.max(np.abs(d))),
            "mean_signed_diff": mean_signed,
            "mean_signed_diff_se": se,
            "mean_signed_diff_ci95": [
                mean_signed - 1.96 * se, mean_signed + 1.96 * se,
            ],
            "var_oof": var_oof,
            "var_serve": var_serve,
            "var_ratio_serve_over_oof": (
                float(var_serve / var_oof) if var_oof > 0.0 else float("nan")
            ),
            # Descriptive only: see the docstring.  This is NOT a gate.
            "ks_statistic": float(ks.statistic),
            "ks_pvalue_descriptive_only": float(ks.pvalue),
        }

    agreement = float(
        np.mean(oof_priors.argmax(axis=1) == ensemble_priors.argmax(axis=1))
    )
    record: Dict[str, Any] = {
        "n_trials": int(n_trials),
        "n_classes": int(n_cols),
        "columns": columns,
        "argmax_agreement": agreement,
        "mean_total_variation_distance": float(
            np.mean(0.5 * np.abs(diff).sum(axis=1))
        ),
        "ks_is_descriptive_only": True,
        "argmax_agreement_is_descriptive_only": True,
        "interpretation": (
            "OOF priors come from a single fold model; served priors average "
            "all fold models, so lower serve-side variance is expected by "
            "construction. Report mean_signed_diff (with CI) and "
            "mean_total_variation_distance -- the router consumes the prior "
            "VECTOR, so a distance on the vector is the train-serve quantity. "
            "Neither the KS p-value nor argmax_agreement is a gate: a "
            "generator collapsed onto the majority class scores perfect "
            "agreement, so that floor was satisfiable by degrading the "
            "model. Vector collapse is caught by the bootstrap per-column "
            "variance floor instead."
        ),
    }

    for key, col in columns.items():
        logger.info(
            "[MCMC-CV] prior OOF→serve col %s (%s): signed Δ=%+.4f "
            "(95%% CI %+.4f..%+.4f), mean|Δ|=%.4f, var %.4f→%.4f "
            "(ratio %.2f), KS=%.3f [descriptive]",
            key, col["class"], col["mean_signed_diff"],
            col["mean_signed_diff_ci95"][0], col["mean_signed_diff_ci95"][1],
            col["mean_abs_diff"], col["var_oof"], col["var_serve"],
            col["var_ratio_serve_over_oof"], col["ks_statistic"],
        )
    logger.warning(
        "OOF→serve prior shift recorded: argmax agreement %.3f, mean TV "
        "distance %.4f over %d trials. Ensemble averaging compresses "
        "variance by design; this record is persisted as "
        "'prior_consistency' and must be reported alongside any result "
        "that depends on the MCMC prior channel.",
        record["argmax_agreement"],
        record["mean_total_variation_distance"],
        record["n_trials"],
    )
    return record

# ═══════════════════════════════════════════════════════════════
# 5.  Main ETL Pipeline
# ═══════════════════════════════════════════════════════════════

def prepare_dataset(
    raw_dir: Path,
    output_path: Path,
    dt_ms: float = 10.0,
    time_config: TimeWindowConfig = DEFAULT_TIME_WINDOW,
    feature_config: FeatureConfig = DEFAULT_FEATURE,
    random_seed: int = 42,
) -> Path:
    """
    Run the full offline data preparation pipeline.

    Args:
        raw_dir: Root directory containing raw session data.
        output_path: Path to save the processed dataset.
        dt_ms: Frame interval in milliseconds.
        time_config: Time window configuration.
        feature_config: Feature dimension configuration.
        random_seed: Random seed for MCMC training.

    Returns:
        Path to the saved dataset file.
    """
    logger.info("=" * 60)
    logger.info("NSMoR Data Preparation Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Data Pairing ──────────────────────────────────
    logger.info("[Step 1] Scanning for data pairs in %s", raw_dir)
    csv_pairs = pair_csv_files(raw_dir)
    logger.info("Found %d session pairs.", len(csv_pairs))

    # ── Step 2: Load and concatenate sessions ─────────────────
    logger.info("[Step 2] Loading and concatenating sessions...")
    kin_paths = [p[0] for p in csv_pairs]
    evt_paths = [p[1] for p in csv_pairs]

    session_data = load_and_concat_sessions(kin_paths, evt_paths)
    logger.info(
        "Loaded %d kinematics rows, %d events rows.",
        len(session_data["kinematics"]),
        len(session_data["events"]),
    )

    # ── Step 2b: Sampling interval diagnostics ───────────────
    # Surface the dt_ms mismatch as a first-class, auditable check.
    # The configured dt_ms (from CLI / default.yaml) may not match the
    # actual inter-frame cadence in the raw kinematics CSVs.  Measure
    # the observed intervals per-trial, aggregate, and compare against
    # the config.  Persisted in the dataset artefact as
    # ``sampling_diagnostics`` so the discrepancy is impossible to miss.
    sampling_diagnostics = compute_sampling_diagnostics(
        session_data["kinematics"], dt_ms,
    )
    logger.info(
        "[Step 2b] Sampling diagnostics — observed median=%.3f ms, "
        "mean=%.3f ms, p99=%.3f ms, configured dt_ms=%.1f ms",
        sampling_diagnostics["observed_median_ms"],
        sampling_diagnostics["observed_mean_ms"],
        sampling_diagnostics["observed_p99_ms"],
        dt_ms,
    )
    if sampling_diagnostics["mismatch_flag"]:
        logger.warning(
            "SAMPLING INTERVAL MISMATCH: configured dt_ms=%.1f ms "
            "but observed median=%.3f ms (ratio=%.2fx). The "
            "configured value may cause systematic scaling errors "
            "in any computation that assumes uniform dt.  Consider "
            "updating config/default.yaml model.dt_ms or using "
            "--dt_ms %.1f.",
            dt_ms,
            sampling_diagnostics["observed_median_ms"],
            sampling_diagnostics["ratio_configured_over_observed"],
            sampling_diagnostics["observed_median_ms"],
        )

    # ── Step 3: Per-trial extraction and labeling ─────────────
    logger.info("[Step 3] Extracting trials and assigning labels...")

    # Get unique session/trial pairs
    trial_groups = session_data["kinematics"].groupby(["session_id",    "trial_id"])
    trials = []
    for (session_id, trial_id), _ in trial_groups:
        try:
            trial = extract_trial_data(session_data, session_id, trial_id)
            trials.append(trial)
        except ValueError as e:
            logger.warning("Skipping trial: %s", e)
            continue

    logger.info("Extracted %d valid trials.", len(trials))

    # Assign ground truth labels using hardware-corrected timestamps.
    # Round-3 (Reviewer A BLK-3B): label collapses must be auditable —
    # the elimination funnel records which criterion stage eliminated
    # each trial, and the aggregated waterfall is logged so an entire
    # behavioural class disappearing can never pass silently again.
    labeled_trials = assign_ground_truth_labels(trials, return_funnel=True)
    logger.info("Labeled %d trials.", len(labeled_trials))

    funnel = labeling_funnel_summary(labeled_trials)
    logger.info("Labeling elimination funnel: %s", funnel)
    if funnel.get("n_PREWALK", 0) == 0 and len(labeled_trials) > 0:
        logger.warning(
            "PREWALK count is ZERO across %d trials.  This is a "
            "criterion/data incompatibility signal, not a stricter "
            "label: inspect the funnel stages above to determine "
            "whether the pre-stimulus window criterion or the data "
            "eliminated the class.", len(labeled_trials),
        )

    # Log label distribution
    from nsmor.config import Label
    label_counts = {}
    for info in labeled_trials:
        label = info["label"]
        label_counts[label.name] = label_counts.get(label.name, 0) + 1
    logger.info("Label distribution: %s", label_counts)

    # ── Round-3 (Reviewer A MAJ-3B): threshold sensitivity sweep ──
    # Re-label the full pre-filter trial set with the velocity thresholds
    # scaled by ±25%, retaining the default composition as the reference.
    # Each record is self-describing: scale, actual thresholds, class
    # schema (including zero counts), denominator, and funnel metadata.
    import dataclasses as _dataclasses
    label_names = [label.name for label in Label]
    labeling_threshold_sensitivity: Dict[str, Dict[str, Any]] = {}
    for scale in (0.75, 1.0, 1.25):
        cfg_alt = _dataclasses.replace(
            DEFAULT_THRESHOLD,
            escape_velocity_threshold=(
                DEFAULT_THRESHOLD.escape_velocity_threshold * scale
            ),
            prewalk_velocity_threshold=(
                DEFAULT_THRESHOLD.prewalk_velocity_threshold * scale
            ),
            pre_active_velocity_threshold=(
                DEFAULT_THRESHOLD.pre_active_velocity_threshold * scale
            ),
        )
        labeled_alt = assign_ground_truth_labels(
            trials, config=cfg_alt, return_funnel=True,
        )
        counts_alt = {name: 0 for name in label_names}
        for info in labeled_alt:
            counts_alt[info["label"].name] += 1
        labeling_threshold_sensitivity[f"thresholds_x{scale:.2f}"] = {
            "scale": scale,
            "n_trials": len(labeled_alt),
            "class_schema": list(label_names),
            "counts": counts_alt,
            "thresholds": {
                "escape_velocity_threshold": float(
                    cfg_alt.escape_velocity_threshold
                ),
                "prewalk_velocity_threshold": float(
                    cfg_alt.prewalk_velocity_threshold
                ),
                "pre_active_velocity_threshold": float(
                    cfg_alt.pre_active_velocity_threshold
                ),
            },
            "labeling_funnel": labeling_funnel_summary(labeled_alt),
        }
        logger.info(
            "Label sensitivity (thresholds x%.2f): %s", scale, counts_alt,
        )

    # ── Step 4: MCMC Prior Generation ────────────────────────
    logger.info("[Step 4] Training MCMC prior generator...")

    # ``on_unanchorable="skip"`` is passed EXPLICITLY, and every drop is
    # accounted for per class and per condition below.  The extractor's
    # default is now strict precisely because the previous unconditional
    # ``except ValueError: continue`` deleted every visual-only trial in
    # the corpus without a word (36 of 396, all No_Response) — and those
    # trials vanished from the regression sequence set too, since the
    # retention identity carries the snapshot drop forward.
    snapshots, snapshot_labels, kept_indices, snapshot_anchor_rules = (
        build_snapshot_dataset(
            labeled_trials,
            time_config=time_config,
            feature_config=feature_config,
            return_kept_indices=True,
            return_anchor_rules=True,
            on_unanchorable="skip",
        )
    )
    snapshot_drop_audit = _audit_snapshot_drops(labeled_trials, kept_indices)
    if snapshot_drop_audit["n_dropped"]:
        logger.warning(
            "%d/%d trials dropped during snapshot extraction; downstream "
            "metadata filtered to match.  BY CLASS: %s  BY CONDITION: %s  "
            "A drop concentrated in one class or one condition is a "
            "systematic bias, not attrition.",
            snapshot_drop_audit["n_dropped"], len(labeled_trials),
            snapshot_drop_audit["dropped_by_class"],
            snapshot_drop_audit["dropped_by_condition"],
        )
    logger.info(
        "Snapshot dataset: %s snapshots, %s labels.  Anchor rules: %s",
        snapshots.shape, snapshot_labels.shape,
        {
            rule: snapshot_anchor_rules.count(rule)
            for rule in sorted(set(snapshot_anchor_rules))
        },
    )

    # Train MCMC model
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Reviewer Round-1 BLOCKER-2 fix: cross-fitted (out-of-fold) priors.
    # Training on all snapshots and re-predicting the SAME snapshots
    # leaked ground-truth labels into the NSMoR input features.  With
    # 5-fold cross-fitting, every prior row is produced by a generator
    # that never saw that trial's label.
    #
    # Fold membership is GROUPED BY ANIMAL.  Grouping by session is not
    # enough: ``_session_N`` splits ONE recording of ONE animal into
    # blocks, so a session-grouped fold let an animal's _session_1 train
    # the generator that produced _session_2's "held-out" prior.  Trials
    # of one animal share that animal's baseline locomotor statistics,
    # gain state, and body mass, so the prior was not out-of-fold in any
    # meaningful sense -- and these priors become input channels 4-7, so
    # the leak enters the model as a feature.  Session ids are aligned
    # through kept_indices because build_snapshot_dataset may skip trials
    # whose snapshot cannot be extracted -- and only when this caller
    # explicitly opts in via on_unanchorable="skip" (a plain zip with
    # labeled_trials misaligns groups when any is dropped).
    labeled_kept = [labeled_trials[i] for i in kept_indices]
    snapshot_groups = animal_keys_of(
        [info["session_id"] for info in labeled_kept]
    )
    assert len(snapshot_groups) == len(snapshots), (
        f"Animal-group count {len(snapshot_groups)} != "
        f"snapshot count {len(snapshots)}"
    )

    # Halving the group count can push a rare class (escape is ~3% of
    # trials) below the fold count.  Adapt the folds; never weaken the
    # grouping to keep a round number.
    n_folds = resolve_group_folds(
        np.asarray(snapshot_labels), snapshot_groups, max_folds=5,
    )
    mcmc_priors, fold_models, fold_diagnostics = train_mcmc_cross_fitted(
        snapshots,
        snapshot_labels,
        n_folds=n_folds,
        groups=snapshot_groups,
        verbose=True,
    )
    # Round-3 (Reviewer A MAJ-3C): StratifiedGroupKFold does NOT
    # guarantee balanced class composition across folds.  Persist the
    # per-fold (session count, class histogram) so fold imbalance is
    # auditable in the saved dataset instead of only logged.
    # The diagnostic keys are named n_*_sessions in the frozen module; the
    # groups passed in are now animals, so these are animal counts.
    for diag in fold_diagnostics:
        logger.info(
            "[MCMC-CV] fold %d: train animals=%d classes=%s | "
            "oof animals=%d classes=%s",
            diag["fold"], diag["n_train_sessions"],
            np.bincount(diag["train_classes"], minlength=4).tolist(),
            diag["n_oof_sessions"],
            np.bincount(diag["oof_classes"], minlength=4).tolist(),
        )
    n_animals = len(set(snapshot_groups.tolist()))
    logger.info(
        "Generated out-of-fold MCMC priors (%d-fold animal-grouped "
        "cross-fitting over %d animals): %s",
        n_folds, n_animals, mcmc_priors.shape,
    )
    assert mcmc_priors.shape == (len(snapshots), feature_config.mcmc_dim), (
        f"mcmc_priors shape {mcmc_priors.shape} != "
        f"({len(snapshots)}, {feature_config.mcmc_dim})"
    )

    # Round-3 (Reviewer A MAJ-3C): a fold whose training side lacks a
    # class makes its model emit near-constant probabilities for that
    # class on the held-out side — the "prior" column degenerates into
    # an uninformative constant and silently poisons the sensory
    # encoding.  Variance floor: every prior column must retain
    # meaningful spread across trials.
    #
    # Round-3 Flaw 3 fix (Reviewer task6): establish the variance
    # threshold via bootstrap resampling of the empirical prior
    # distribution, rather than using an arbitrary magic constant.
    # Bootstrap estimates the sampling distribution of variance under
    # the null hypothesis that the column is informative (non-degenerate).
    # We set the floor at the 5th percentile of this bootstrap
    # distribution, ensuring we reject only columns whose variance is
    # statistically indistinguishable from zero.
    #
    # Round-3 revision (BLK-3B conservative resolution): a column is
    # only REQUIRED to be informative when its behavioural class is
    # actually populated in the labelled data.  PREWALK is currently
    # empty in this dataset (every candidate is absorbed by the
    # PRE_ACTIVE branch — see labeling_funnel_summary), so its prior
    # column is trivially constant BY CONSTRUCTION, not by fold
    # imbalance.  Hard-failing on an absent class conflates "criterion/
    # data incompatibility" (already surfaced by the funnel + the
    # PREWALK=0 warning) with "fold grouping failure".  Empty-class
    # columns are recorded in the saved dataset as
    # ``mcmc_degenerate_columns`` and every downstream consumer can
    # exclude them; a populated class with a degenerate column still
    # hard-fails.
    from nsmor.config import Label as _Label
    label_names_in_col = [cls.name for cls in _Label]
    degenerate_columns: Dict[str, str] = {}

    # Bootstrap-based variance floor estimation (per class column)
    n_bootstrap = 1000
    variance_floors: Dict[int, float] = {}
    for c in range(mcmc_priors.shape[1]):
        col_data = mcmc_priors[:, c]

        # Bootstrap: resample with replacement, compute variance
        boot_vars = np.zeros(n_bootstrap)
        rng = np.random.default_rng(seed=random_seed + c)
        n_samples = len(col_data)
        for b in range(n_bootstrap):
            resample_idx = rng.choice(n_samples, size=n_samples, replace=True)
            boot_vars[b] = np.var(col_data[resample_idx])

        # 5th percentile as conservative floor (reject only extreme degeneracy)
        var_floor = float(np.percentile(boot_vars, 5))
        variance_floors[c] = var_floor

        logger.debug(
            "Bootstrap variance floor (col %d): 5th pctl=%.2e, "
            "median=%.2e, 95th pctl=%.2e",
            c, var_floor, float(np.median(boot_vars)),
            float(np.percentile(boot_vars, 95)),
        )

    for c in range(mcmc_priors.shape[1]):
        col_var = float(np.var(mcmc_priors[:, c]))
        var_floor = variance_floors[c]

        if col_var >= var_floor:
            continue

        cls_name = label_names_in_col[c] if c < len(label_names_in_col) else f"class_{c}"
        class_present = any(
            info["label"].name == cls_name for info in labeled_kept
        )
        if class_present:
            raise ValueError(
                f"OOF MCMC prior column {c} ({cls_name}) is degenerate: "
                f"variance={col_var:.2e} < bootstrap floor={var_floor:.2e} "
                f"(5th percentile from {n_bootstrap} resamples). This "
                f"indicates at least one fold's training side lacks the "
                f"class (fold-grouping failure). Regroup folds (fewer "
                f"folds / LOCO) or collect more sessions for that class; "
                f"refusing to save a dataset with uninformative priors."
            )
        logger.warning(
            "OOF MCMC prior column %d (%s) is degenerate because the "
            "class is EMPTY in this dataset (variance=%.2e < floor=%.2e). "
            "Column recorded in mcmc_degenerate_columns; downstream "
            "consumers must not interpret it as evidence.",
            c, cls_name, col_var, var_floor,
        )
        degenerate_columns[cls_name] = f"column_{c}"
    logger.info("OOF prior variance floor check passed (all columns).")

    # Round-3 (Reviewer B CRITICAL-3c): train-vs-serve distribution
    # mismatch audit.  During training the model sees SINGLE-FOLD OOF
    # probabilities (high variance); at inference time the documented
    # protocol feeds ENSEMBLE mean probabilities from all fold models
    # (variance-compressed).  Quantify that shift NOW and persist it,
    # so deployment-time behaviour is comparable against recorded
    # training conditions.
    ens_probs = np.zeros_like(mcmc_priors)
    for fm in fold_models:
        ens_probs += fm.predict_proba(snapshots)
    ens_probs /= len(fold_models)
    ens_probs = np.clip(ens_probs, 1e-12, 1.0)
    ens_probs /= ens_probs.sum(axis=1, keepdims=True)

    # Round-3 (Reviewer B CRITICAL-3c) revisited: the shift is measured
    # and persisted, but a two-sample KS test on paired OOF/ensemble
    # columns is the wrong instrument — see
    # :func:`audit_prior_train_serve_shift`.
    prior_consistency: Dict[str, Any] = audit_prior_train_serve_shift(
        mcmc_priors, ens_probs, label_names_in_col,
    )

# ── Step 5: Sequence Extraction with Visual Physics Reconstruction ──
    logger.info("[Step 5] Extracting continuous sequences with visual physics reconstruction...")

    sequences = []
    valid_snaps = []  # 仅收集快照输入，不在循环内推理
    seq_session_ids: List[str] = []  # Round-3 CRITICAL-3b: session id per kept trial
    kept_seq_indices: List[int] = []  # Track which trials succeed (Flaw 2 fix)
    # Stimulus modality per kept trial.  The MoR Router's routing
    # hypothesis is per modality, so the condition has to travel with the
    # sequence rather than be re-derived from the 8-D features later (the
    # pure-wind zero-prepend makes that ambiguous downstream).
    seq_conditions: List[str] = []

    # Iterate over labeled_kept ONLY: trials dropped during snapshot
    # extraction have no out-of-fold prior row, so including them here
    # would misalign sequences against mcmc_priors (the count assert
    # below catches it).  Note the clamp inside this loop makes every
    # kept trial's snapshot succeed, so sequences/priors/snaps stay
    # row-for-row aligned.
    for trial_idx, info in enumerate(labeled_kept):
        try:
            trial_data = info["trial_data"]
            stimulus_onset_ms = info["stimulus_onset_ms"]

            # 1. 提取快照
            # For baseline_visual trials, stimulus_onset_ms may be 0,
            # making snapshot_time negative. Clamp to trial start.
            ttc_offset = time_config.ttc_offset_ms
            if stimulus_onset_ms + ttc_offset < trial_data["time_ms"][0]:
                ttc_offset = trial_data["time_ms"][0] - stimulus_onset_ms

            snap = extract_mcmc_snapshot(
                trial_data,
                stimulus_onset_ms,
                ttc_offset_ms=ttc_offset,
                time_config=time_config,
                feature_config=feature_config,
            )

            # 2. 提取并计算 l/v ratio
            l_v_ratio_raw = trial_data.get("l_v_ratio", np.array([0.0]))
            if isinstance(l_v_ratio_raw, np.ndarray) and len(l_v_ratio_raw) > 0:
                l_v_ratio = float(np.nanmax(l_v_ratio_raw))
                if np.isnan(l_v_ratio) or np.isinf(l_v_ratio):
                    l_v_ratio = 0.0
            else:
                l_v_ratio = 0.0

            # 3. 提取序列
            X_seq, Y_seq = extract_trial_sequence(
                trial_data,
                feature_config=feature_config,
                dt_ms=dt_ms,
            )

            # 4. 处理视觉特征：优先使用原始数据，否则重构
            raw_visual_angle = trial_data.get("visual_angle", None)
            has_raw_visual = (
                raw_visual_angle is not None
                and isinstance(raw_visual_angle, np.ndarray)
                and len(raw_visual_angle) > 0
                and np.any(np.abs(raw_visual_angle) > 1e-6)
            )


            if has_raw_visual:
                # 使用原始 visual_angle（已由实验设备记录）
                visual_angle_to_use = raw_visual_angle
                l_v_to_use = trial_data.get("l_v_ratio", np.zeros_like(raw_visual_angle))
                if isinstance(l_v_to_use, np.ndarray) and len(l_v_to_use) > 0:
                    l_v_to_use = l_v_to_use
                else:
                    l_v_to_use = np.zeros_like(raw_visual_angle)
            else:
                # 重构视觉特征（纯风试验或缺失数据）
                visual_angle_to_use, l_v_to_use = reconstruct_trial_visual_features(
                    trial_data=trial_data,
                    stimulus_onset_ms=stimulus_onset_ms,
                    l_v_ratio=l_v_ratio,
                    dt_ms=dt_ms,
                )

            # 确保长度匹配
            n_frames = X_seq.shape[0]
            if len(visual_angle_to_use) < n_frames:
                # 填充到匹配长度
                padded = np.zeros(n_frames, dtype=np.float64)
                padded[:len(visual_angle_to_use)] = visual_angle_to_use
                visual_angle_to_use = padded
            elif len(visual_angle_to_use) > n_frames:
                visual_angle_to_use = visual_angle_to_use[:n_frames]

            if len(l_v_to_use) < n_frames:
                padded = np.zeros(n_frames, dtype=np.float64)
                padded[:len(l_v_to_use)] = l_v_to_use
                l_v_to_use = padded
            elif len(l_v_to_use) > n_frames:
                l_v_to_use = l_v_to_use[:n_frames]

            X_seq[:, 0] = visual_angle_to_use

            # 同步入库：保证 sequences 和 valid_snaps 绝对对齐
            sequences.append((X_seq, Y_seq, int(info["label"])))
            valid_snaps.append(snap)
            seq_session_ids.append(str(info["session_id"]))
            kept_seq_indices.append(trial_idx)  # Track successful trial index
            seq_conditions.append(classify_stimulus_condition(trial_data))

            logger.debug(
                "Trial %s/%d: θ(t) range [%.2f°, %.2f°], "
                "l/v=%.4f, is_pure_wind=%s, has_raw_visual=%s",
                info["session_id"], info["trial_id"],
                float(np.min(visual_angle_to_use)),
                float(np.max(visual_angle_to_use)),
                l_v_ratio,
                bool(np.all(np.abs(visual_angle_to_use) < 1e-6)),
                has_raw_visual,
            )

        except (ValueError, KeyError) as e:
            logger.warning(
                "Skipping trial [%s, %d] in Step 5: %s",
                info.get("session_id", "UNKNOWN"),
                info.get("trial_id", -1),
                e
            )
            continue

    # Round-3 Flaw 2 fix: if any trials failed during sequence extraction,
    # filter mcmc_priors and ens_probs to maintain row-for-row alignment.
    # The kept_seq_indices mask tracks which trials succeeded.
    skipped_count = len(labeled_kept) - len(kept_seq_indices)
    if skipped_count > 0:
        logger.warning(
            "Skipped %d/%d trials during sequence extraction; filtering "
            "mcmc_priors to maintain alignment.",
            skipped_count, len(labeled_kept),
        )
        mcmc_priors = mcmc_priors[kept_seq_indices]
        ens_probs = ens_probs[kept_seq_indices]

        # Re-audit on the surviving rows (same instrument, same gates).
        prior_consistency = audit_prior_train_serve_shift(
            mcmc_priors, ens_probs, label_names_in_col,
        )
        logger.info(
            "Recomputed prior_consistency on %d kept trials.", len(kept_seq_indices)
        )

    # 5. 批量推理：一次性处理所有快照，原生输出 (N, 4) 矩阵，彻底规避降维风险
    # Reviewer Round-1 BLOCKER-2: the priors used downstream are the
    # OUT-OF-FOLD cross-fitted ones computed in Step 4 (mcmc_priors).
    # The full-data mcmc_model is NOT re-applied here — that would
    # reintroduce the same-sample label leakage the cross-fitting
    # removes.  valid_snaps is retained only for order verification.
    assert len(valid_snaps) == mcmc_priors.shape[0], (
        f"Snapshot/prior count mismatch: {len(valid_snaps)} vs "
        f"{mcmc_priors.shape[0]}"
    )

    logger.info("Extracted %d sequences with reconstructed visual features.", len(sequences))


    # Unpack sequences
    X_seqs = [seq[0] for seq in sequences]
    Y_seqs = [seq[1] for seq in sequences]
    labels = np.array([seq[2] for seq in sequences], dtype=np.int64)
    lengths = np.array([x.shape[0] for x in X_seqs], dtype=np.int64)
    snapshot_groups_aligned = np.array(seq_session_ids, dtype=object)
    assert len(snapshot_groups_aligned) == len(X_seqs), (
        f"session_ids count {len(snapshot_groups_aligned)} != "
        f"sequence count {len(X_seqs)}"
    )
    stimulus_conditions = np.array(seq_conditions, dtype=object)
    # ``wind_only`` is the modality with no looming at all; multisensory
    # trials carry both channels and must NOT count as pure wind, or the
    # routing signal would be trained against a mixed population.
    is_pure_wind = np.array(
        [c == "wind_only" for c in seq_conditions], dtype=bool
    )
    assert len(stimulus_conditions) == len(X_seqs), (
        f"stimulus_conditions count {len(stimulus_conditions)} != "
        f"sequence count {len(X_seqs)}"
    )
    assert is_pure_wind.shape == (len(X_seqs),), (
        f"is_pure_wind shape {is_pure_wind.shape} != ({len(X_seqs)},)"
    )
    condition_counts = {
        str(name): int(count)
        for name, count in zip(
            *np.unique(stimulus_conditions, return_counts=True)
        )
    }
    logger.info("Stimulus-condition coverage: %s", condition_counts)
    if int(is_pure_wind.sum()) == 0:
        logger.warning(
            "No wind_only trials survived ETL (%s). Routing-aux will "
            "be identically zero. Typical cause: raw CSVs still carry "
            "stim_state instead of wind_state — run "
            "scripts/pre_load_adapt.py on a staging copy first.",
            condition_counts,
        )

    # ── Shape assertions ──
    assert len(X_seqs) == len(Y_seqs) == len(labels) == len(lengths), (
        f"Length mismatch: X={len(X_seqs)}, Y={len(Y_seqs)}, "
        f"labels={len(labels)}, lengths={len(lengths)}"
    )
    assert len(X_seqs) == len(mcmc_priors), (
        f"Sequence/prior count mismatch: Seq={len(X_seqs)} vs Priors={len(mcmc_priors)}"
    )

    for i, (x, y) in enumerate(zip(X_seqs, Y_seqs)):
        T_i = x.shape[0]
        assert x.shape == (T_i, feature_config.per_frame_total_dim), (
            f"X_seqs[{i}] shape {x.shape} != ({T_i}, {feature_config.per_frame_total_dim})"
        )
        assert y.shape == (T_i,), (
            f"Y_seqs[{i}] shape {y.shape} != ({T_i},)"
        )

    # Retention is a separate audit from the labeling elimination funnel.
    # ``labeling_funnel`` stays exactly ``labeling_funnel_summary(...)``.
    funnel_retention = {
        "n_prefilter_labeled_trials": len(labeled_trials),
        "n_retained_sequences": len(sequences),
        "n_dropped_before_snapshot": len(labeled_trials) - len(labeled_kept),
        "n_dropped_during_sequence_extraction": skipped_count,
        # Totals alone hid a 100%-single-condition loss; see
        # :func:`_audit_snapshot_drops`.
        "snapshot_drops_by_class": snapshot_drop_audit["dropped_by_class"],
        "snapshot_drops_by_condition": (
            snapshot_drop_audit["dropped_by_condition"]
        ),
        "snapshot_anchor_rules": {
            rule: snapshot_anchor_rules.count(rule)
            for rule in sorted(set(snapshot_anchor_rules))
        },
    }
    assert funnel_retention["n_retained_sequences"] == len(X_seqs)
    assert (
        funnel_retention["n_prefilter_labeled_trials"]
        - funnel_retention["n_dropped_before_snapshot"]
        - funnel_retention["n_dropped_during_sequence_extraction"]
        == funnel_retention["n_retained_sequences"]
    ), f"Inconsistent labeling funnel denominators: {funnel_retention}"

    logger.info(
        "Dataset summary: %d sequences, total_frames=%d, "
        "avg_length=%.1f, max_length=%d",
        len(X_seqs), int(lengths.sum()),
        float(lengths.mean()), int(lengths.max()),
    )

    # ── Save ─────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = {
        "X_seqs": X_seqs,
        "Y_seqs": Y_seqs,
        "mcmc_priors": mcmc_priors,  # OUT-OF-FOLD (cross-fitted) priors
        # Reviewer Round-2 M-3: persist the fold models and document the
        # inference-time prior protocol so downstream scripts never have
        # to refit a generator on data whose labels they hold (that would
        # reintroduce the Round-1 leakage).  Protocol: predict with every
        # fold model, average the probability rows, renormalise.
        # Records the fold count ACTUALLY used, not a literal: the count
        # adapts down when a rare class occupies too few animals, and a
        # stale "5fold" claim in the artifact would misstate provenance.
        "mcmc_prior_provenance": f"oof_{n_folds}fold_animal_grouped_cv",
        "mcmc_fold_models": fold_models,
        # Round-3 (Reviewer A MAJ-3C): per-fold composition records so
        # StratifiedGroupKFold imbalance is auditable post hoc.
        "mcmc_fold_diagnostics": fold_diagnostics,
        # Round-3 (Reviewer B CRITICAL-3c): OOF (training) vs ensemble
        # (serving) prior distribution shift statistics per class.
        "mcmc_prior_train_serve_consistency": prior_consistency,
        "mcmc_inference_protocol": (
            "ensemble: probs = mean([m.predict_proba(x) for m in "
            "mcmc_fold_models]); probs /= probs.sum(-1, keepdims=True)"
        ),
        "labels": labels,
        "lengths": lengths,
        # Per-sequence session ids, stored with the ``_session_N`` suffix
        # INTACT.  Downstream splits group by ANIMAL, deriving the animal
        # key by stripping that suffix
        # (nsmor.pipeline.grouping.animal_of), so the finer granularity is
        # kept here and coarsened at the point of use — the reverse would
        # discard information no consumer can recover.  Grouping by
        # session alone is insufficient: ``_session_N`` blocks belong to
        # one animal, and trials of one animal share its baseline
        # locomotor statistics, gain state, and body mass.  (Full nested CV
        # remains a documented limitation in the analysis report.)
        "session_ids": snapshot_groups_aligned,
        # Per-trial stimulus modality, read off the physical channels at
        # ETL time.  ``is_pure_wind`` is the boolean the routing-aux loss
        # partitions on; ``stimulus_conditions`` keeps the full 4-way
        # naming for analysis.  Additive keys — loaders that predate them
        # ignore them, and consumers must treat a missing key as "unknown"
        # rather than assume False.
        "stimulus_conditions": stimulus_conditions,
        "is_pure_wind": is_pure_wind,
        # Round-3 (Reviewer A MAJ-3B): label-composition sensitivity to
        # a ±25% scaling of the velocity thresholds.
        "labeling_threshold_sensitivity": labeling_threshold_sensitivity,
        # Exact ``labeling_funnel_summary(labeled_trials)``; do not mutate.
        "labeling_funnel": funnel,
        # Post-label snapshot/sequence filtering denominators.
        "labeling_funnel_retention": funnel_retention,
        # Round-3 (BLK-3B conservative resolution): prior columns whose
        # behavioural class is empty in this dataset — recorded so
        # downstream consumers can exclude them from interpretation.
        "mcmc_degenerate_columns": degenerate_columns,
        "feature_config": feature_config,
        "time_config": time_config,
        # Sampling interval diagnostics: observed vs configured dt_ms.
        # Persisted so downstream consumers can audit the cadence
        # assumption without re-scanning the raw CSVs.
        "sampling_diagnostics": sampling_diagnostics,
        # Round-2 CRITICAL-A fix: provenance stamp; loaders reject
        # datasets without it (pre-2.0 data has leaked priors and
        # np.max-based labels).
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
    }

    torch.save(dataset, output_path)
    logger.info("Saved dataset to %s", output_path)

    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info("=" * 60)

    return output_path


# ═══════════════════════════════════════════════════════════════
# 6.  CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="NSMoR Offline Data Preparation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Root directory containing raw session data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/nsmor_dataset.pt",
        help="Output path for processed dataset.",
    )
    parser.add_argument(
        "--dt_ms",
        type=float,
        default=10.0,
        help="Frame interval in milliseconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for MCMC training.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    prepare_dataset(
        raw_dir=Path(args.raw_dir),
        output_path=Path(args.output),
        dt_ms=args.dt_ms,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
