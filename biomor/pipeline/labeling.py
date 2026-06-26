"""
Ground truth labeling and filtering.

Implements the traditional pipeline for classifying trials into
behavioral categories based on velocity thresholds and event timing.

Label assignment order (first match wins):
  1. Pre_Active  — high spontaneous velocity during baseline
  2. Startle     — peak velocity > threshold within startle latency
  3. Walk        — sustained velocity > threshold within walk latency
  4. NoResponse  — none of the above
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from biomor.config import DEFAULT_THRESHOLD, Label, ThresholdConfig


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
# Pre-active detection
# ──────────────────────────────────────────────────────────────

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

    Priority: Startle > Walk > NoResponse.

    Args:
        velocity: 1-D velocity time series (cm / s).
        time_ms: 1-D timestamps (ms).
        stimulus_onset_ms: Time of stimulus onset (ms).
        config: Threshold configuration.

    Returns:
        One of :attr:`Label.STARTLE`, :attr:`Label.WALK`,
        or :attr:`Label.NO_RESPONSE`.
    """
    post_mask = time_ms > stimulus_onset_ms
    if not np.any(post_mask):
        return Label.NO_RESPONSE

    post_velocity = velocity[post_mask]
    post_time = time_ms[post_mask]

    # ── Startle: high peak velocity within short latency window ──
    startle_end = stimulus_onset_ms + config.startle_latency_max_ms
    startle_mask = post_time <= startle_end
    if np.any(startle_mask):
        if np.max(np.abs(post_velocity[startle_mask])) > config.startle_velocity_threshold:
            return Label.STARTLE

    # ── Walk: sustained velocity within extended window ──
    walk_end = stimulus_onset_ms + config.walk_latency_max_ms
    walk_mask = post_time <= walk_end
    if np.any(walk_mask):
        if np.mean(np.abs(post_velocity[walk_mask])) > config.walk_velocity_threshold:
            return Label.WALK

    return Label.NO_RESPONSE


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
