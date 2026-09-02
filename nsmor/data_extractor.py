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
from nsmor.pipeline.kinematics import mirror_to_right

# Pure-wind prepended baseline: 5.7 s × 100 Hz = 570 frames
PURE_WIND_PREPEND_FRAMES: int = 570


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _find_nearest_index(time_ms: np.ndarray, target_ms: float) -> int:
    """Return the index of the element closest to *target_ms*."""
    return int(np.argmin(np.abs(time_ms - target_ms)))


def resolve_snapshot_anchor(
    trial_data: Dict[str, np.ndarray],
    stimulus_onset_ms: float,
) -> Tuple[float, str]:
    """
    Locate the reference instant the MCMC Snapshot is offset from.

    Two rules, because the two stimulus conditions put their decisive
    moment in different places:

    ``"stimulus_onset"``
        Any trial carrying wind.  ``stimulus_onset_ms`` is the first
        wind-on frame, which for the real corpus already sits close to the
        looming collision (measured median 6714 ms against a collision at
        6873 ms).  Wind is the potent escape trigger, so its arrival is
        the decision-relevant instant, and this rule is left exactly as it
        was — the 288 multisensory and 72 pure-wind Snapshots of the real
        corpus are unchanged by this function's introduction.

    ``"looming_collision"``
        Visual-only trials.  Their looming begins at the ``TrialStart ->
        Looming`` transition, i.e. the same instant as ``trial_start``, so
        ``stimulus_onset_ms`` is 0 and offsetting backwards from it lands
        before the first frame.  Every visual-only trial in the corpus was
        therefore dropped: 36 of 396, all No_Response, none surviving —
        and because the drop happens before sequence extraction, those
        trials disappear from the regression training set as well.  The
        collision is located by the visual-angle peak, which measured
        6873.0 ms (median) against a geometric collision time of 6874.8 ms
        for l/v = 120 ms at a 2 deg initial angle — inside one 250 Hz
        frame — so no stimulus-geometry constant is assumed here.

    Args:
        trial_data: From :func:`pipeline.io.extract_trial_data`.
        stimulus_onset_ms: Absolute time of stimulus onset.

    Returns:
        ``(anchor_ms, anchor_rule)``.  The rule is returned rather than
        inferred so a mixed-anchor dataset stays auditable.
    """
    visual_angle = np.asarray(trial_data["visual_angle"], dtype=np.float64)
    wind_state = np.asarray(trial_data["wind_state"], dtype=np.float64)

    has_looming = bool(np.any(np.abs(visual_angle) > 0.0))
    has_wind = bool(np.any(np.abs(wind_state) > 0.0))

    if has_looming and not has_wind:
        time_ms = np.asarray(trial_data["time_ms"], dtype=np.float64)
        collision_idx = int(np.argmax(visual_angle))
        return float(time_ms[collision_idx]), "looming_collision"

    return float(stimulus_onset_ms), "stimulus_onset"


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
    Extract a 5-D MCMC snapshot at the trial's anchor + *ttc_offset_ms*.

    The anchor is condition-dependent — see :func:`resolve_snapshot_anchor`.
    Trials carrying wind are offset from ``stimulus_onset_ms`` exactly as
    before; visual-only trials are offset from the looming collision,
    because their stimulus onset coincides with trial start and offsetting
    backwards from it fell outside the trial.

    Features
    --------
    [0] visual_angle        — instantaneous looming visual angle (deg)
    [1] looming_velocity    — l / v ratio at snapshot time
    [2] wind_state          — wind stimulus state (0 or 1)
    [3] avg_velocity_bg     — mean |velocity| in preceding 200 ms
    [4] max_acceleration_bg — max |acceleration| in preceding 200 ms

    Args:
        trial_data: From :func:`pipeline.io.extract_trial_data`.
        stimulus_onset_ms: Absolute time of stimulus onset.
        ttc_offset_ms: Offset from the anchor (default −50 ms).
        time_config: Time window config.
        feature_config: Feature dimension config.

    Returns:
        1-D array, shape ``(5,)``.

    Raises:
        ValueError: If the snapshot time precedes the first frame.
    """
    anchor_ms, anchor_rule = resolve_snapshot_anchor(
        trial_data, stimulus_onset_ms,
    )
    snapshot_time_ms = anchor_ms + ttc_offset_ms
    time_ms = trial_data["time_ms"]

    if snapshot_time_ms < time_ms[0]:
        raise ValueError(
            f"Snapshot time {snapshot_time_ms:.1f} ms is before trial "
            f"start {time_ms[0]:.1f} ms "
            f"(anchor={anchor_ms:.1f} ms by rule '{anchor_rule}', "
            f"offset={ttc_offset_ms:.1f} ms)."
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
    return_kept_indices: bool = False,
    return_anchor_rules: bool = False,
    on_unanchorable: str = "raise",
) -> Tuple[np.ndarray, ...]:
    """
    Build the full MCMC snapshot matrix and label vector.

    Args:
        labeled_trials: Output of :func:`labeling.assign_ground_truth_labels`.
        time_config: Time window configuration.
        feature_config: Feature dimension configuration.
        return_kept_indices: If ``True``, additionally return the indices
            into *labeled_trials* of the trials whose snapshot was
            extracted.  Downstream per-trial metadata (e.g. session ids for
            grouped cross-fitting) MUST be filtered with these indices —
            assuming row-for-row alignment with *labeled_trials* silently
            misaligns groups whenever a trial is dropped.
        return_anchor_rules: If ``True``, additionally return the anchor
            rule name per retained trial (see
            :func:`resolve_snapshot_anchor`).  The anchor is
            condition-dependent, so recording which rule applied keeps a
            mixed-anchor dataset auditable instead of leaving it implicit.
        on_unanchorable: ``"raise"`` (default) propagates the ValueError
            from a trial whose snapshot time falls outside the trial;
            ``"skip"`` drops it.  The default is strict because the
            previous unconditional ``except ValueError: continue`` deleted
            an entire experimental condition in silence — every
            visual-only trial in the corpus, 36 of 396, all No_Response.
            A caller that genuinely tolerates drops must say so and report
            what it lost.

    Returns:
        ``(snapshots, labels)``, with ``kept_indices`` appended when
        *return_kept_indices* is set and ``anchor_rules`` appended when
        *return_anchor_rules* is set, in that order.

    Raises:
        ValueError: If *on_unanchorable* is unrecognised, if a trial
            cannot be anchored while *on_unanchorable* is ``"raise"``, or
            if no snapshot could be extracted at all.
    """
    if on_unanchorable not in ("raise", "skip"):
        raise ValueError(
            f"on_unanchorable must be 'raise' or 'skip', got "
            f"{on_unanchorable!r}."
        )

    snapshots: List[np.ndarray] = []
    labels: List[int] = []
    kept_indices: List[int] = []
    anchor_rules: List[str] = []

    for info_idx, info in enumerate(labeled_trials):
        try:
            snap = extract_mcmc_snapshot(
                info["trial_data"],
                info["stimulus_onset_ms"],
                ttc_offset_ms=time_config.ttc_offset_ms,
                time_config=time_config,
                feature_config=feature_config,
            )
        except ValueError as exc:
            if on_unanchorable == "raise":
                raise ValueError(
                    f"labeled_trials[{info_idx}] "
                    f"(session={info.get('session_id')!r}, "
                    f"trial={info.get('trial_id')!r}) could not be "
                    f"anchored: {exc}  Pass on_unanchorable='skip' to "
                    f"tolerate this, and report what was dropped."
                ) from exc
            continue
        _, anchor_rule = resolve_snapshot_anchor(
            info["trial_data"], info["stimulus_onset_ms"],
        )
        snapshots.append(snap)
        labels.append(int(info["label"]))
        kept_indices.append(info_idx)
        anchor_rules.append(anchor_rule)

    if not snapshots:
        raise ValueError("No valid snapshots could be extracted.")

    snapshots_arr = np.stack(snapshots, axis=0)
    labels_arr = np.array(labels, dtype=np.int64)

    assert snapshots_arr.shape == (len(snapshots), feature_config.snapshot_dim)
    assert labels_arr.shape == (len(snapshots),)

    out: List[object] = [snapshots_arr, labels_arr]
    if return_kept_indices:
        out.append(kept_indices)
    if return_anchor_rules:
        out.append(anchor_rules)
    return tuple(out)  # type: ignore[return-value]


def build_sequence_dataset(
    labeled_trials: List[Dict],
    feature_config: FeatureConfig = DEFAULT_FEATURE,
    unify_wind_sides: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Build Trial-Start anchored sequences for all valid trials.

    Args:
        labeled_trials: Output of :func:`labeling.assign_ground_truth_labels`.
        feature_config: Feature dimension configuration.
        unify_wind_sides: Mirror left-wind trials into the right-wind frame.
            When ``True``, left-wind trials are reflected via
            :func:`pipeline.kinematics.mirror_to_right` into a canonical
            right frame so that lateralized escape is pooled. Scalar speed
            target is invariant; ``wind_side_original`` / ``wind_side_mirrored``
            provenance is preserved in the source trial dict for downstream
            stratification.

            **Statistical caveat (must be honoured downstream):** pooling
            left+right doubles nominal *n* but does not double independent
            subjects — trials from the same animal are paired. Any
            inferential test on pooled ``X_seq/Y_seq`` must stratify or
            model ``wind_side_original`` (mixed-effects / stratified
            permutation) and report effect size (Cohen's d) with
            FDR/Bonferroni correction; do not treat pooled trials as i.i.d.
            for naive t-tests. Train/test splits must apply the same
            ``unify_wind_sides`` flag on both sides to avoid leakage; log
            the flag in experiment metadata.

    Returns:
        List of ``(X_seq, Y_seq, label)`` tuples.
    """
    sequences: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for info in labeled_trials:
        try:
            trial_data = info["trial_data"]
            if unify_wind_sides:
                trial_data = mirror_to_right(trial_data)
                # Preserve provenance for downstream stratification (no API break)
                info["_wind_side_original"] = trial_data.get("wind_side_original", "unknown")
                info["_wind_side_mirrored"] = trial_data.get("wind_side_mirrored", False)
            X_seq, Y_seq = extract_trial_sequence(
                trial_data,
                feature_config=feature_config,
            )
            sequences.append((X_seq, Y_seq, int(info["label"])))
        except ValueError:
            continue

    return sequences
