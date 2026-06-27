"""
Data extraction — MCMC snapshots and Trial-Start anchored sequences.

Extracts strictly constrained time-slices from trial data for
downstream MCMC prior generation and continuous modelling.

Snapshot (5-D, at TTC + offset)
    [visual_angle, looming_velocity, wind_state,
     avg_velocity_bg, max_acceleration_bg]

Sequence (per frame, anchored at Trial Start)
    [v_vis(t), wind(t), v_kine(t-1), a_kine(t-1), P_startle,
     P_walk, P_pre_active, P_no_response]   →  8-D

Pure-Wind baseline alignment
----------------------------
If a trial is a **Pure Wind** stimulus (visual_angle array is entirely
flat / zero), a 5.7-second zero-matrix (570 frames at 100 Hz) is
prepended to the front of the physical features and target vector.
This preserves the temporal alignment with looming trials whose
sequences already include the 2-second baseline plus stimulus period.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from nsmor.config import (
    DEFAULT_FEATURE,
    DEFAULT_TIME_WINDOW,
    FeatureConfig,
    TimeWindowConfig,
)

# Pure-wind prepended baseline: 5.7 s × 100 Hz = 570 frames
PURE_WIND_PREPEND_FRAMES: int = 570


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _find_nearest_index(time_ms: np.ndarray, target_ms: float) -> int:
    """Return the index of the element closest to *target_ms*."""
    return int(np.argmin(np.abs(time_ms - target_ms)))


def _extract_background_features(
    velocity: np.ndarray,
    acceleration: np.ndarray,
    time_ms: np.ndarray,
    snapshot_time_ms: float,
    window_ms: float,
) -> Tuple[float, float]:
    """
    Mean |velocity| and max |acceleration| in the window
    ``[snapshot_time_ms - window_ms, snapshot_time_ms)``.
    """
    window_start = snapshot_time_ms - window_ms
    mask = (time_ms >= window_start) & (time_ms < snapshot_time_ms)

    if not np.any(mask):
        return 0.0, 0.0

    avg_velocity = float(np.mean(np.abs(velocity[mask])))
    max_acceleration = float(np.max(np.abs(acceleration[mask])))
    return avg_velocity, max_acceleration


def _is_pure_wind(visual_angle: np.ndarray, atol: float = 1e-8) -> bool:
    """
    Detect a Pure Wind stimulus trial.

    A trial is *pure wind* if the ``visual_angle`` array is entirely
    flat (constant) **and** effectively zero — i.e. no looming visual
    stimulus was presented.

    Args:
        visual_angle: 1-D array of visual angles across all frames.
        atol: Absolute tolerance for the "all zero" check.

    Returns:
        ``True`` if every element of *visual_angle* is within *atol* of 0.
    """
    return bool(np.all(np.abs(visual_angle) < atol))


# ──────────────────────────────────────────────────────────────
# MCMC snapshot extraction (5-D)
# ──────────────────────────────────────────────────────────────

def extract_mcmc_snapshot(
    trial_data: Dict[str, np.ndarray],
    stimulus_onset_ms: float,
    ttc_offset_ms: float = -50.0,
    time_config: TimeWindowConfig = DEFAULT_TIME_WINDOW,
    feature_config: FeatureConfig = DEFAULT_FEATURE,
) -> np.ndarray:
    """
    Extract a 5-D MCMC snapshot at TTC + *ttc_offset_ms*.

    Features
    --------
    [0] visual_angle        — instantaneous looming visual angle (deg)
    [1] looming_velocity    — l / v ratio at snapshot time
    [2] wind_state          — wind stimulus state (0 or 1)
    [3] avg_velocity_bg     — mean |velocity| in preceding 200 ms
    [4] max_acceleration_bg — max |acceleration| in preceding 200 ms

    Args:
        trial_data: From :func:`pipeline.io.extract_trial_data`.
        stimulus_onset_ms: Absolute time of stimulus onset (TTC reference).
        ttc_offset_ms: Offset from TTC for snapshot (default −50 ms).
        time_config: Time window config.
        feature_config: Feature dimension config.

    Returns:
        1-D array, shape ``(5,)``.

    Raises:
        ValueError: If the snapshot time precedes the first frame.
    """
    snapshot_time_ms = stimulus_onset_ms + ttc_offset_ms
    time_ms = trial_data["time_ms"]

    if snapshot_time_ms < time_ms[0]:
        raise ValueError(
            f"Snapshot time {snapshot_time_ms:.1f} ms is before trial "
            f"start {time_ms[0]:.1f} ms."
        )

    idx = _find_nearest_index(time_ms, snapshot_time_ms)

    visual_angle = float(trial_data["visual_angle"][idx])
    looming_velocity = float(trial_data["l_v_ratio"][idx])
    wind_state = float(trial_data["wind_state"][idx])

    avg_velocity, max_acceleration = _extract_background_features(
        trial_data["velocity"],
        trial_data["acceleration"],
        time_ms,
        snapshot_time_ms,
        window_ms=time_config.background_window_ms,
    )

    snapshot = np.array(
        [visual_angle, looming_velocity, wind_state,
         avg_velocity, max_acceleration],
        dtype=np.float64,
    )

    assert snapshot.shape == (feature_config.snapshot_dim,), (
        f"Snapshot shape mismatch: expected ({feature_config.snapshot_dim},), "
        f"got {snapshot.shape}"
    )
    return snapshot


# ──────────────────────────────────────────────────────────────
# Trial-Start anchored continuous sequence
# ──────────────────────────────────────────────────────────────

def extract_trial_sequence(
    trial_data: Dict[str, np.ndarray],
    feature_config: FeatureConfig = DEFAULT_FEATURE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a Trial-Start anchored continuous sequence.

    The sequence origin is **Trial Start** (the exact moment of the
    2-second absolute static baseline), *not* wind onset or TTC.

    **Pure-Wind baseline alignment:**
    If the trial has no looming visual stimulus (``visual_angle`` is
    all zeros), a 5.7-second zero-matrix (570 frames at 100 Hz) is
    prepended to the physical features and target vector so that the
    temporal structure matches looming trials.

    Per-frame feature layout (8-D)
    ------------------------------
    [0] v_vis(t)        — real-time visual angle
    [1] wind(t)         — real-time wind state (0 / 1)
    [2] v_kine(t-1)     — physical velocity from the **previous** frame
    [3] a_kine(t-1)     — physical acceleration from the **previous** frame
    [4] P_startle       ┐
    [5] P_walk          │ MCMC prior placeholder (filled later by
    [6] P_pre_active    │ the DataLoader from pre-computed priors)
    [7] P_no_response   ┘

    Target Y_t = continuous velocity at time *t*.

    Args:
        trial_data: From :func:`pipeline.io.extract_trial_data`.
        feature_config: Feature dimension config.

    Returns:
        ``(X_seq, Y_seq)`` where
        X_seq has shape ``(seq_len, 8)`` and
        Y_seq has shape ``(seq_len,)``.
    """
    time_ms = trial_data["time_ms"]
    n_frames = len(time_ms)

    visual_angle = trial_data["visual_angle"]
    wind_state = trial_data["wind_state"]
    velocity = trial_data["velocity"]
    acceleration = trial_data["acceleration"]

    # ── Physical features (n_frames, 4) ──
    physical = np.zeros(
        (n_frames, feature_config.per_frame_physical_dim), dtype=np.float64,
    )
    physical[:, 0] = visual_angle     # v_vis(t)
    physical[:, 1] = wind_state       # wind(t)
    # v_kine(t-1) and a_kine(t-1): shift by one frame
    physical[1:, 2] = velocity[:-1]
    physical[1:, 3] = acceleration[:-1]
    # Frame 0 has no predecessor → already zero

    # ── Pure-Wind baseline alignment ──
    # If no looming stimulus was presented, prepend 5.7 s of zeros
    # so that the temporal structure matches looming trials.
    if _is_pure_wind(visual_angle):
        prepend_zeros = np.zeros(
            (PURE_WIND_PREPEND_FRAMES, feature_config.per_frame_physical_dim),
            dtype=np.float64,
        )
        physical = np.concatenate([prepend_zeros, physical], axis=0)

        target_zeros = np.zeros(PURE_WIND_PREPEND_FRAMES, dtype=np.float64)
        Y_seq = np.concatenate([target_zeros, velocity.copy()], axis=0)
    else:
        Y_seq = velocity.copy()

    # ── MCMC placeholder ──
    total_frames = physical.shape[0]
    mcmc_placeholder = np.zeros(
        (total_frames, feature_config.mcmc_dim), dtype=np.float64,
    )

    # ── Concatenate ──
    X_seq = np.concatenate([physical, mcmc_placeholder], axis=1)

    # ── Shape assertions ──
    assert X_seq.shape == (total_frames, feature_config.per_frame_total_dim), (
        f"X_seq shape: expected ({total_frames}, "
        f"{feature_config.per_frame_total_dim}), got {X_seq.shape}"
    )
    assert Y_seq.shape == (total_frames,), (
        f"Y_seq shape: expected ({total_frames},), got {Y_seq.shape}"
    )
    return X_seq, Y_seq


# ──────────────────────────────────────────────────────────────
# Batch builders
# ──────────────────────────────────────────────────────────────

def build_snapshot_dataset(
    labeled_trials: List[Dict],
    time_config: TimeWindowConfig = DEFAULT_TIME_WINDOW,
    feature_config: FeatureConfig = DEFAULT_FEATURE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the full MCMC snapshot matrix and label vector.

    Args:
        labeled_trials: Output of :func:`labeling.assign_ground_truth_labels`.
        time_config: Time window configuration.
        feature_config: Feature dimension configuration.

    Returns:
        ``(snapshots, labels)`` where
        snapshots has shape ``(n_valid, 5)`` and
        labels has shape ``(n_valid,)``.
    """
    snapshots: List[np.ndarray] = []
    labels: List[int] = []

    for info in labeled_trials:
        try:
            snap = extract_mcmc_snapshot(
                info["trial_data"],
                info["stimulus_onset_ms"],
                ttc_offset_ms=time_config.ttc_offset_ms,
                time_config=time_config,
                feature_config=feature_config,
            )
            snapshots.append(snap)
            labels.append(int(info["label"]))
        except ValueError:
            continue

    if not snapshots:
        raise ValueError("No valid snapshots could be extracted.")

    snapshots_arr = np.stack(snapshots, axis=0)
    labels_arr = np.array(labels, dtype=np.int64)

    assert snapshots_arr.shape == (len(snapshots), feature_config.snapshot_dim)
    assert labels_arr.shape == (len(snapshots),)
    return snapshots_arr, labels_arr


def build_sequence_dataset(
    labeled_trials: List[Dict],
    feature_config: FeatureConfig = DEFAULT_FEATURE,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Build Trial-Start anchored sequences for all valid trials.

    Args:
        labeled_trials: Output of :func:`labeling.assign_ground_truth_labels`.
        feature_config: Feature dimension configuration.

    Returns:
        List of ``(X_seq, Y_seq, label)`` tuples.
    """
    sequences: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for info in labeled_trials:
        try:
            X_seq, Y_seq = extract_trial_sequence(
                info["trial_data"],
                feature_config=feature_config,
            )
            sequences.append((X_seq, Y_seq, int(info["label"])))
        except ValueError:
            continue

    return sequences
