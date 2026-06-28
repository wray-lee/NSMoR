"""
Ground truth labeling and filtering.

Implements standardized criteria for cricket escape behavior:

**Escape**
- Walking speed was < 10 mm/s when the airflow was applied.
- The maximum walking speed was > 50 mm/s for 250-ms periods after the airflow stimulus onset.

**Prewalk**
- The maximum walking speed exceeded 10 mm/s for 1-s periods just before the airflow stimulus onset.
- The maximum walking speed was > 50 mm/s for 250-ms periods after the airflow stimulus onset.

**No Response**
- The maximum walking speed was ≤ 50 mm/s for 250-ms periods after the airflow stimulus onset.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from nsmor.config import DEFAULT_THRESHOLD, Label, ThresholdConfig


# ──────────────────────────────────────────────────────────────
# Event lookup helpers
# ──────────────────────────────────────────────────────────────

def find_event_time(
    event_types: np.ndarray,
    event_times: np.ndarray,
    event_name: str,
) -> Optional[float]:
    """
    Return the timestamp of the first occurrence of *event_name*.

    Args:
        event_types: 1-D array of event type strings.
        event_times: 1-D array of corresponding timestamps (ms).
        event_name: The event type to search for.

    Returns:
        Timestamp in ms, or ``None`` if the event is absent.
    """
    mask = event_types == event_name
    if not np.any(mask):
        return None
    return float(event_times[mask][0])


# ──────────────────────────────────────────────────────────────
# Sustained speed check
# ──────────────────────────────────────────────────────────────

def _check_sustained_speed(
    velocity: np.ndarray,
    time_ms: np.ndarray,
    start_ms: float,
    duration_ms: float,
    threshold: float,
) -> bool:
    """
    Check if velocity remains above threshold for a sustained period.

    Args:
        velocity: 1-D velocity time series (cm/s).
        time_ms: 1-D timestamps (ms).
        start_ms: Start of the check window (ms).
        duration_ms: Duration to check (ms).
        threshold: Velocity threshold (cm/s).

    Returns:
        True if max velocity in window exceeds threshold.
    """
    end_ms = start_ms + duration_ms
    mask = (time_ms >= start_ms) & (time_ms < end_ms)
    if not np.any(mask):
        return False
    return bool(np.max(np.abs(velocity[mask])) > threshold)


# ──────────────────────────────────────────────────────────────
# Response classification
# ──────────────────────────────────────────────────────────────

def classify_response(
    velocity: np.ndarray,
    time_ms: np.ndarray,
    stimulus_onset_ms: float,
    config: ThresholdConfig = DEFAULT_THRESHOLD,
) -> Label:
    """
    Classify the behavioral response after stimulus onset.

    Criteria (from standard cricket escape behavior protocol):
    1. Check post-stimulus: max speed > 50 mm/s for 250ms
       - If NO → No Response
       - If YES → check pre-stimulus
         - If pre-stimulus speed > 10 mm/s for 1s → Prewalk
         - Else → Escape

    Args:
        velocity: 1-D velocity time series (cm/s).
        time_ms: 1-D timestamps (ms).
        stimulus_onset_ms: Time of stimulus onset (ms).
        config: Threshold configuration.

    Returns:
        One of Label.ESCAPE, Label.PREWALK, or Label.NO_RESPONSE.
    """
    # ── Check post-stimulus sustained speed (>50 mm/s for 250ms) ──
    post_stim_250ms = _check_sustained_speed(
        velocity, time_ms,
        start_ms=stimulus_onset_ms,
        duration_ms=config.escape_sustained_ms,
        threshold=config.escape_velocity_threshold,
    )

    if not post_stim_250ms:
        # No escape response
        return Label.NO_RESPONSE

    # ── Post-stimulus escape detected ──
    # Check if pre-stimulus speed > 10 mm/s for 1s (Prewalk)
    pre_stim_start = stimulus_onset_ms - config.prewalk_sustained_ms
    pre_stim_1s = _check_sustained_speed(
        velocity, time_ms,
        start_ms=pre_stim_start,
        duration_ms=config.prewalk_sustained_ms,
        threshold=config.prewalk_velocity_threshold,
    )

    if pre_stim_1s:
        return Label.PREWALK
    else:
        return Label.ESCAPE


# ──────────────────────────────────────────────────────────────
# Batch labeling
# ──────────────────────────────────────────────────────────────

def assign_ground_truth_labels(
    trials: List[Dict[str, np.ndarray]],
    config: ThresholdConfig = DEFAULT_THRESHOLD,
) -> List[Dict]:
    """
    Assign ground truth labels to a list of trials.

    Args:
        trials: List of trial data dictionaries
            (as returned by :func:`pipeline.io.extract_trial_data`).
        config: Threshold configuration.

    Returns:
        List of dicts, each containing:

        - ``session_id``          — str
        - ``trial_id``            — int
        - ``label``               — :class:`Label`
        - ``stimulus_onset_ms``   — float
        - ``trial_data``          — the original trial dict
    """
    labeled: List[Dict] = []

    for trial in trials:
        time_ms: np.ndarray = trial["time_ms"]
        velocity: np.ndarray = trial["velocity"]

        stimulus_onset = find_event_time(
            trial["event_types"], trial["event_times"], "stimulus_onset",
        )
        if stimulus_onset is None:
            # No stimulus event — skip trial
            continue

        # Pre-active check (high spontaneous activity before stimulus)
        if is_pre_active(velocity, time_ms, stimulus_onset, config):
            label = Label.PRE_ACTIVE
        else:
            label = classify_response(velocity, time_ms, stimulus_onset, config)

        labeled.append({
            "session_id": trial["session_id"],
            "trial_id": trial["trial_id"],
            "label": label,
            "stimulus_onset_ms": stimulus_onset,
            "trial_data": trial,
        })

    return labeled


def is_pre_active(
    velocity: np.ndarray,
    time_ms: np.ndarray,
    baseline_end_ms: float,
    config: ThresholdConfig = DEFAULT_THRESHOLD,
) -> bool:
    """
    Check whether a trial has high spontaneous activity during baseline.

    A trial is *pre-active* if the maximum absolute velocity in the
    window ``[trial_start, baseline_end_ms)`` exceeds the configured
    threshold.

    Args:
        velocity: 1-D velocity time series (cm / s).
        time_ms: 1-D timestamps (ms).
        baseline_end_ms: End of the baseline period (ms).
        config: Threshold configuration.

    Returns:
        ``True`` if the trial should be labelled :attr:`Label.PRE_ACTIVE`.
    """
    baseline_mask = time_ms < baseline_end_ms
    if not np.any(baseline_mask):
        return False

    max_baseline_velocity = np.max(np.abs(velocity[baseline_mask]))
    return bool(max_baseline_velocity > config.pre_active_velocity_threshold)
