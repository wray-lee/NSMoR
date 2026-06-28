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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

from nsmor.config import DEFAULT_FEATURE, DEFAULT_TIME_WINDOW, FeatureConfig, TimeWindowConfig
from nsmor.data_extractor import (
    build_sequence_dataset,
    build_snapshot_dataset,
    extract_trial_sequence,
    PURE_WIND_PREPEND_FRAMES,
)
from nsmor.mcmc_module import MCMCPriorGenerator, train_mcmc
from nsmor.pipeline.io import EVENT_COLUMNS, KINEMATICS_COLUMNS
from nsmor.pipeline.labeling import assign_ground_truth_labels
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
# Visual Looming Physics Reconstructor
# ═══════════════════════════════════════════════════════════════

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
    """
    n_frames = len(time_ms)
    visual_angle = np.zeros(n_frames, dtype=np.float64)
    l_v_array = np.zeros(n_frames, dtype=np.float64)

    # ── Guard against invalid l/v ratio ──
    if np.isnan(l_v_ratio) or np.isinf(l_v_ratio):
        logger.warning(
            "Invalid l_v_ratio=%.4f, defaulting to 0.", l_v_ratio
        )
        return visual_angle, l_v_array

    # ── Compute θ(t) for each frame ──
    for i, t in enumerate(time_ms):
        # Pre-stimulus: no visual stimulus yet
        if t < stimulus_onset_ms:
            visual_angle[i] = 0.0
            l_v_array[i] = 0.0
            continue

        # Time-to-collision remaining (can be negative post-TTC)
        ttc_remaining = ttc_ms - t

        # Post-collision: clamp to maximum visual angle
        if ttc_remaining < VISUAL_PHYSICS_EPSILON:
            visual_angle[i] = 180.0  # Maximum visual angle post-collision
            l_v_array[i] = l_v_ratio
            continue

        # ── Main formula: θ(t) = 2 × arctan(l/v / (TTC - t)) ──
        # Guard against division by zero
        denominator = ttc_remaining
        if abs(denominator) < VISUAL_PHYSICS_EPSILON:
            visual_angle[i] = 180.0
            l_v_array[i] = l_v_ratio
            continue

        # Compute the ratio inside arctan
        ratio = l_v_ratio / denominator

        # Guard against NaN from arctan
        if np.isnan(ratio) or np.isinf(ratio):
            visual_angle[i] = 0.0
            l_v_array[i] = 0.0
            continue

        # Compute visual angle in radians, then convert to degrees
        theta_rad = 2.0 * np.arctan(ratio)
        theta_deg = np.degrees(theta_rad)

        # ── Sanity checks ──
        if np.isnan(theta_deg) or np.isinf(theta_deg):
            logger.warning(
                "NaN/Inf in θ(t) at frame %d (t=%.1fms, TTC=%.1fms), "
                "clamping to 0.",
                i, t, ttc_ms,
            )
            theta_deg = 0.0

        # Clamp to valid range [0, 180] degrees
        theta_deg = float(np.clip(theta_deg, 0.0, 180.0))

        visual_angle[i] = theta_deg
        l_v_array[i] = l_v_ratio

    # ── Shape assertions ──
    assert visual_angle.shape == (n_frames,), (
        f"visual_angle shape {visual_angle.shape} != ({n_frames},)"
    )
    assert l_v_array.shape == (n_frames,), (
        f"l_v_array shape {l_v_array.shape} != ({n_frames},)"
    )

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
        n_total = n_original + PURE_WIND_PREPEND_FRAMES

        visual_angle_full = np.zeros(n_total, dtype=np.float64)
        l_v_full = np.zeros(n_total, dtype=np.float64)

        logger.debug(
            "Pure-wind trial: %d prepended + %d original = %d total frames.",
            PURE_WIND_PREPEND_FRAMES, n_original, n_total,
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

        # Compute velocity from smoothed position
        dt_s = dt_ms / 1000.0
        velocity = np.gradient(np.sqrt(x_smooth**2 + y_smooth**2), dt_s)

        # Compute acceleration from velocity
        if window_length >= 3:
            velocity_smooth = savgol_filter(velocity, window_length, 3)
        else:
            velocity_smooth = velocity.copy()
        acceleration = np.gradient(velocity_smooth, dt_s)

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

    # Assign ground truth labels using hardware-corrected timestamps
    labeled_trials = assign_ground_truth_labels(trials)
    logger.info("Labeled %d trials.", len(labeled_trials))

    # Log label distribution
    from nsmor.config import Label
    label_counts = {}
    for info in labeled_trials:
        label = info["label"]
        label_counts[label.name] = label_counts.get(label.name, 0) + 1
    logger.info("Label distribution: %s", label_counts)

    # ── Step 4: MCMC Prior Generation ────────────────────────
    logger.info("[Step 4] Training MCMC prior generator...")

    snapshots, snapshot_labels = build_snapshot_dataset(
        labeled_trials,
        time_config=time_config,
        feature_config=feature_config,
    )
    logger.info(
        "Snapshot dataset: %s snapshots, %s labels.",
        snapshots.shape, snapshot_labels.shape,
    )

    # Train MCMC model
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    mcmc_model = train_mcmc(
        snapshots,
        snapshot_labels,
        verbose=True,
    )
    logger.info("MCMC model trained.")

    # Generate priors for all trials
    mcmc_priors = mcmc_model.predict_proba(snapshots)
    assert mcmc_priors.shape == (len(snapshots), feature_config.mcmc_dim), (
        f"mcmc_priors shape {mcmc_priors.shape} != "
        f"({len(snapshots)}, {feature_config.mcmc_dim})"
    )
    logger.info("Generated MCMC priors: %s", mcmc_priors.shape)

    # ── Step 5: Sequence Extraction with Visual Physics Reconstruction ──
    logger.info("[Step 5] Extracting continuous sequences with visual physics reconstruction...")

    sequences = []
    valid_indices = []
    for info in labeled_trials:
        try:
            trial_data = info["trial_data"]
            stimulus_onset_ms = info["stimulus_onset_ms"]

            # Extract l/v ratio from trial data (use median of l_v_ratio array)
            l_v_ratio_raw = trial_data.get("l_v_ratio", np.array([0.0]))
            if isinstance(l_v_ratio_raw, np.ndarray) and len(l_v_ratio_raw) > 0:
                # Use the maximum l/v ratio during stimulus period
                l_v_ratio = float(np.nanmax(l_v_ratio_raw))
                if np.isnan(l_v_ratio) or np.isinf(l_v_ratio):
                    l_v_ratio = 0.0
            else:
                l_v_ratio = 0.0

            # Reconstruct visual features
            visual_angle_recon, l_v_recon = reconstruct_trial_visual_features(
                trial_data=trial_data,
                stimulus_onset_ms=stimulus_onset_ms,
                l_v_ratio=l_v_ratio,
                dt_ms=dt_ms,
            )

            # Extract sequence with the original function
            X_seq, Y_seq = extract_trial_sequence(
                trial_data,
                feature_config=feature_config,
            )

            # ── Inject reconstructed visual features into X_seq ──
            # X_seq[:, 0] = v_vis(t) — reconstructed visual angle
            # X_seq[:, 1] = wind(t)  — keep original wind state
            n_frames = X_seq.shape[0]
            assert len(visual_angle_recon) == n_frames, (
                f"Visual angle length {len(visual_angle_recon)} != "
                f"X_seq frames {n_frames}"
            )

            # Overwrite visual angle column with reconstructed values
            X_seq[:, 0] = visual_angle_recon

            # Optionally store l/v ratio in a metadata dict (not in X_seq)
            # The l/v ratio is already embedded in the visual angle computation

            sequences.append((X_seq, Y_seq, int(info["label"])))

            logger.debug(
                "Trial %s/%d: reconstructed θ(t) range [%.2f°, %.2f°], "
                "l/v=%.4f, is_pure_wind=%s",
                info["session_id"], info["trial_id"],
                float(np.min(visual_angle_recon)),
                float(np.max(visual_angle_recon)),
                l_v_ratio,
                bool(np.all(np.abs(visual_angle_recon) < 1e-8)),
            )
            valid_indices.append(i)
            sequences.append((X_seq, Y_seq, int(info["label"])))

        except (ValueError, KeyError) as e:
            logger.warning("Skipping trial: %s", e)
            continue

    logger.info("Extracted %d sequences with reconstructed visual features.", len(sequences))
    mcmc_priors = mcmc_priors[valid_indices]

    # Unpack sequences
    X_seqs = [seq[0] for seq in sequences]
    Y_seqs = [seq[1] for seq in sequences]
    labels = np.array([seq[2] for seq in sequences], dtype=np.int64)
    lengths = np.array([x.shape[0] for x in X_seqs], dtype=np.int64)

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
        "mcmc_priors": mcmc_priors,
        "labels": labels,
        "lengths": lengths,
        "feature_config": feature_config,
        "time_config": time_config,
        "hw_triggers": hw_triggers,
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
