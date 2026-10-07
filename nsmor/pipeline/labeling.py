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
# Sustained-window coverage (Round-4)
# ──────────────────────────────────────────────────────────────

SUSTAINED_MIN_COVERAGE: float = 0.5
"""Minimum fraction of the required sustained duration that must be
spanned by recorded frames for the anchored window to be evaluable.

Round-4 fix: the previous sustained check divided by whatever frames
happened to remain, so a trial whose recording ended a few frames after
response initiation certified a 250-ms response from 3 frames
(2000/2004/2008 ms).  A window that covers less than half of the
required duration is missing/truncated evidence and must be unavailable,
not a positive response."""


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

    Round-4 fix: a matched anchor whose timestamp is non-finite
    (NaN/Inf) is corrupt chronology, not a valid event.  It is reported
    as unavailable (``None``) so every caller fails closed instead of
    computing comparisons against NaN/Inf (which silently produced
    ``NO_RESPONSE``/``PRE_ACTIVE`` labels).

    Args:
        event_types: 1-D array of event type strings.
        event_times: 1-D array of corresponding timestamps (ms).
        event_name: The event type to search for.

    Returns:
        Timestamp in ms, or ``None`` if the event is absent or its
        timestamp is non-finite.
    """
    mask = event_types == event_name
    if not np.any(mask):
        return None
    timestamp = float(event_times[mask][0])
    if not np.isfinite(timestamp):
        return None
    return timestamp


def has_stimulus_anchor(
    event_types: np.ndarray,
    event_times: np.ndarray,
) -> bool:
    """Return whether a finite ``stimulus_onset`` anchor is present.

    Round-4 fail-closed helper for eligibility ledgers: a trial with no
    ``stimulus_onset`` event, or with one whose timestamp is non-finite
    (NaN/Inf), has no valid chronology and must be recorded as
    unavailable rather than fabricated into a behavioural class.

    Args:
        event_types: 1-D array of event type strings.
        event_times: 1-D array of corresponding timestamps (ms).

    Returns:
        ``True`` only when a finite ``stimulus_onset`` timestamp exists.
    """
    return find_event_time(event_types, event_times, "stimulus_onset") is not None


# ──────────────────────────────────────────────────────────────
# Sustained speed check
# ──────────────────────────────────────────────────────────────

def _check_sustained_speed(
    velocity: np.ndarray,
    time_ms: np.ndarray,
    start_ms: float,
    duration_ms: float,
    threshold: float,
    min_fraction: float = 0.5,
    max_latency_ms: float = 200.0,
    hard_end_ms: Optional[float] = None,
    anchor_min_frames: int = 2,
    min_coverage: float = SUSTAINED_MIN_COVERAGE,
) -> bool:
    """
    Check if velocity remains above threshold for a sustained period.

    Round-1 fix (Reviewer B MINOR-A): the previous implementation
    flagged the window on a single instantaneous sample
    (``np.max(...) > threshold``), so one spike frame classified a
    trial as ESCAPE — the "sustained" semantics were not implemented.
    The corrected criterion anchors on **response initiation**: let
    t0 be the first frame in ``[start_ms, start_ms + max_latency_ms)``
    whose |velocity| exceeds *threshold* (escape latency in insects is
    ~50-100 ms; anchoring on stimulus onset instead would confound
    latency with the sustained-duration requirement).  The response
    counts as sustained if at least ``min_fraction`` of the
    ``duration_ms`` window starting at t0 is above threshold.
    ``min_fraction = 0.5`` tolerates brief sub-threshold dips (sensor
    dropouts, stride pauses) while still demanding genuine sustained
    locomotion.

    Round-2 fix (Reviewer A MAJOR-C): two residual contamination paths
    are closed.

    1. **Anchor continuity.**  A single-frame sensor glitch could serve
       as the anchor and drag the window over frames that have nothing
       to do with a locomotor bout.  The anchor must therefore be the
       start of at least ``anchor_min_frames`` consecutive above-thresh
       old frames (2 at dt=10 ms — a 20 ms commitment consistent with
       the tens-of-ms threshold recovery of insect giant-interneuron
       pathways, far below any behavioural timescale).
    2. **Hard end boundary (``hard_end_ms``).**  For PREWALK's
       pre-stimulus check, an anchor near the search-window end let the
       ``[t0, t0 + duration)`` window extend PAST stimulus onset, so
       post-stimulus escape-running frames counted toward "pre-stimulus
       walking" and flipped genuine ESCAPE trials to PREWALK (verified
       empirically by Reviewer A with a constructed counterexample).
       Passing ``hard_end_ms=stimulus_onset_ms`` truncates every
       evaluation window there; combined with requiring the anchor to
       lie strictly before onset, only truly pre-stimulus frames can
       satisfy the criterion.

    Args:
        velocity: 1-D velocity time series (cm/s).
        time_ms: 1-D timestamps (ms).
        start_ms: Start of the search window (ms), e.g. stimulus onset.
        duration_ms: Sustained duration to verify (ms).
        threshold: Velocity threshold (cm/s).
        min_fraction: Minimum fraction of frames above threshold
            within the anchored window.
        max_latency_ms: Maximum delay between *start_ms* and response
            initiation for the anchor to be valid.
        hard_end_ms: Optional exclusive upper bound (ms); all windows
            (search and evaluation) are truncated to end here.  Used to
            keep pre-stimulus checks strictly pre-stimulus.
        anchor_min_frames: Minimum consecutive above-threshold frames
            required to establish the response anchor (Round-2 MAJOR-C).
        min_coverage: Minimum fraction of *duration_ms* that must be
            spanned by recorded frames for an anchored window to be
            evaluable (Round-4).  A window covering less than this is
            missing/truncated evidence, so the response is unavailable
            rather than certified from whatever few frames remain.

    Returns:
        True if an above-threshold response initiates within
        *max_latency_ms* and remains above threshold for at least
        ``min_fraction`` of the subsequent *duration_ms* (evaluated
        entirely before *hard_end_ms* when given).

    Round-4 fix: the previous version anchored on the FIRST qualifying
    frame and divided by whatever frames remained, so a recording that
    ended 3 frames after response initiation (e.g. onset 2000 ms, frames
    at 2000/2004/2008 ms) certified a 250-ms sustained response from 8 ms
    of evidence.  Two root causes are repaired here:

    1. **Coverage.**  The anchored window must span at least
       *min_coverage* of *duration_ms*; a truncated/missing tail makes
       the response unavailable.
    2. **Anchor search.**  When the first qualifying anchor's window is
       not fully covered, later anchors WITHIN the latency bound are
       still searched, so a genuine response at 190 ms latency is not
       discarded because a short burst at 0 ms failed the coverage test.
       Anchor selection and window boundaries stay strict: only frames
       before *hard_end_ms* are eligible, and the window still ends at
       the earliest of ``anchor_t + duration_ms`` and *hard_end_ms*.
    """
    abs_v = np.abs(velocity)
    if abs_v.size == 0 or time_ms.size == 0 or duration_ms <= 0.0:
        return False

    distinct_times = np.unique(time_ms)
    if distinct_times.size < 2 or not np.all(np.isfinite(distinct_times)):
        return False
    cadence = float(np.median(np.diff(distinct_times)))
    if not np.isfinite(cadence) or cadence <= 0.0:
        return False

    search_end = start_ms + max_latency_ms
    if hard_end_ms is not None:
        search_end = min(search_end, hard_end_ms)
        if search_end <= start_ms:
            return False
    search_mask = (time_ms >= start_ms) & (time_ms < search_end)
    above_idx = np.nonzero(search_mask & (abs_v > threshold))[0]
    if above_idx.size == 0:
        return False

    nominal_frames = round(duration_ms / cadence) + 1

    # Anchor at response initiation, demanding minimal continuity so a
    # single-frame glitch cannot define the window (Reviewer A MAJOR-C).
    # Round-3 (Reviewer B MINOR-4): anchor_min_frames is configuration-
    # owned, not a function-local magic number.
    # Round-4: iterate over candidate anchors and evaluate the sustained
    # window at each.  A candidate whose window is truncated (the trial
    # ends before the response is demonstrated) does not commit the
    # trial; a later qualifying anchor within the latency bound is still
    # tested, so an early short burst cannot suppress a genuine response.
    for i in above_idx:
        run = 1
        j = i + 1
        while (
            j < abs_v.size
            and (hard_end_ms is None or time_ms[j] < hard_end_ms)
            and abs_v[j] > threshold
            and time_ms[j] - time_ms[i] < max_latency_ms
            and (time_ms[j] - time_ms[j - 1] <= cadence * 1.5 + 1e-9)
        ):
            run += 1
            j += 1
        if run < anchor_min_frames:
            continue

        # A candidate whose evaluation window ends before the required
        # duration (either by hard_end_ms or by the recorded timeline
        # ceasing prematurely) cannot certify a sustained event.
        anchor_t = time_ms[i]
        end_ms = anchor_t + duration_ms
        if hard_end_ms is not None and end_ms > hard_end_ms:
            continue
        win_mask = (time_ms >= anchor_t) & (time_ms < end_ms)
        n_frames = int(win_mask.sum())
        if n_frames < anchor_min_frames:
            continue

        # Coverage guard: the anchored window must span at least
        # *min_coverage* of the required duration.  The nominal frame
        # count follows from the trial's own cadence over the full
        # timeline, never from the candidate window itself.
        if nominal_frames < 1 or (n_frames / nominal_frames) < min_coverage:
            continue

        times_win = np.sort(time_ms[win_mask])
        evidence_span_ms = float(times_win[-1] - anchor_t)
        if (evidence_span_ms / duration_ms) < min_coverage:
            continue

        # Endpoint guard: the recorded evaluation window must not end
        # before the required duration has elapsed.
        if (end_ms - times_win[-1]) > cadence * 1.5 + 1e-9:
            continue

        # Temporal-gap guard: a window whose recorded frames leave
        # a hole larger than the nominal cadence is not contiguous evidence.
        if np.any(np.diff(times_win) > cadence * 1.5 + 1e-9):
            continue

        above = int(np.sum(abs_v[win_mask] > threshold))
        if (above / n_frames) >= min_fraction:
            return True
    return False


# ──────────────────────────────────────────────────────────────
# Pre-stimulus walking check (Round-3 MAJ-3A)
# ──────────────────────────────────────────────────────────────

def check_prewalk_window(
    velocity: np.ndarray,
    time_ms: np.ndarray,
    stimulus_onset_ms: float,
    window_ms: float,
    threshold: float,
    min_fraction: float,
    min_coverage: float = 0.5,
) -> bool:
    """
    Verify ongoing locomotion in the whole pre-stimulus window.

    Round-3 fix (Reviewer A MAJ-3A / BLK-3B): the previous PREWALK
    criterion reused the POST-stimulus latency-anchored machinery —
    it searched for a response anchor inside
    ``[onset-1000, onset-800)`` and anchored the 1 s evaluation window
    there.  That semantics made the classification depend on the
    animal's motion state in exactly the 200 ms before onset: an animal
    already walking since ``onset-700`` had no anchor in the search
    band and was classified as NOT prewalking.  The more of the
    pre-stimulus window an animal was walking, the less likely a
    PREWALK label — inverted from the behavioural definition and the
    direct mechanism behind the PREWALK=0 collapse observed after the
    Round-2 re-run.

    The corrected criterion is the plain window-fraction reading of
    "pre-stimulus speed > 10 mm/s for 1 s": at least *min_fraction* of
    the frames in the ENTIRE ``[onset - window_ms, onset)`` interval
    must exceed *threshold*.  No anchoring, no latency band — the 1 s
    requirement describes how MUCH of the last second was locomotion,
    not WHEN it started.

    Args:
        velocity: 1-D velocity time series (cm/s).
        time_ms: 1-D timestamps (ms).
        stimulus_onset_ms: Time of stimulus onset (ms); also the
            exclusive upper bound of the evaluation window.
        window_ms: Length of the pre-stimulus window (ms), i.e. the
            nominal ``prewalk_sustained_ms``.
        threshold: Velocity threshold (cm/s).
        min_fraction: Required fraction of above-threshold frames.
        min_coverage: Minimum fraction of the nominal window that must
            be populated by recorded frames; sparser baselines fail
            conservatively (cannot demonstrate sustained walking).
            Round-4: the nominal frame count is derived from the trial's
            own cadence (median gap over the full recorded timeline),
            never from the candidate window itself, and a temporal gap
            larger than the cadence invalidates the window as
            non-contiguous evidence.

    Returns:
        True if the trial shows sustained pre-stimulus locomotion.
    """
    abs_v = np.abs(velocity)
    win_start_ms = stimulus_onset_ms - window_ms
    win_mask = (time_ms >= win_start_ms) & (time_ms < stimulus_onset_ms)
    n_frames = int(win_mask.sum())

    # Coverage guard: a nearly empty window cannot support the claim.
    # The nominal frame count follows from the TRIAL's own cadence,
    # estimated from the FULL recorded timeline rather than from the
    # candidate window itself.  Deriving cadence from the window let two
    # high-speed samples 900 ms apart masquerade as a dense cadence and
    # fabricate sufficient coverage (Round-4).  When the trial's cadence
    # cannot be established (fewer than two distinct timestamps) the
    # window is unsupported evidence and fails conservatively.
    if n_frames < 2:
        return False
    distinct_times = np.unique(time_ms)
    if distinct_times.size < 2 or not np.all(np.isfinite(distinct_times)):
        return False
    cadence = float(np.median(np.diff(distinct_times)))
    if not np.isfinite(cadence) or cadence <= 0.0:
        return False
    nominal_frames = round(window_ms / cadence) + 1
    if nominal_frames < 1 or (n_frames / nominal_frames) < min_coverage:
        return False

    # Temporal-gap guard (Round-4): a window whose recorded frames leave
    # a hole larger than the nominal cadence is not contiguous evidence.
    # The above-threshold fraction would then be computed over
    # disconnected islands and cannot demonstrate sustained locomotion.
    times_win = np.sort(time_ms[win_mask])
    if np.any(np.diff(times_win) > cadence * 1.5 + 1e-9):
        return False

    above = int(np.sum(abs_v[win_mask] > threshold))
    return (above / n_frames) >= min_fraction


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

    Criteria (from standard cricket escape behavior protocol,
    responder-first branch order — Round-3 BLK-3B root-cause fix):
    1. Check post-stimulus: sustained speed > 50 mm/s for 250ms
       (latency-anchored)
       - If NO → No Response
       - If YES → check pre-stimulus
         - If pre-stimulus speed > 10 mm/s for ≥50% of the last 1s →
           Prewalk
         - Else → Escape

    PRE_ACTIVE is deliberately NOT returned here: it is a baseline-state
    attribute of NON-responders, decided in
    :func:`assign_ground_truth_labels` (the previous pre-active-first
    order structurally absorbed every walking animal into PRE_ACTIVE —
    see the Round-3 BLK-3B comment there).

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
        min_fraction=config.sustained_min_fraction,
        max_latency_ms=config.response_max_latency_ms,
        anchor_min_frames=config.sustained_anchor_min_frames,
    )

    if not post_stim_250ms:
        # No escape response
        return Label.NO_RESPONSE

    # ── Post-stimulus escape detected ──
    # Check if pre-stimulus speed > 10 mm/s for 1s (Prewalk).
    # Round-3 fix (Reviewer A MAJ-3A): the latency-anchored machinery is
    # REPLACED by a whole-window fraction criterion over
    # ``[onset - 1000, onset)`` — the previous design searched for the
    # anchor in the last 200 ms before onset, so animals that started
    # walking earlier were systematically denied the PREWALK label (the
    # mechanism behind the Round-2 PREWALK=0 collapse).  The window is
    # hard-bounded at stimulus onset, so only truly pre-stimulus frames
    # can satisfy the criterion.
    pre_stim_1s = check_prewalk_window(
        velocity, time_ms,
        stimulus_onset_ms=stimulus_onset_ms,
        window_ms=config.prewalk_sustained_ms,
        threshold=config.prewalk_velocity_threshold,
        min_fraction=config.prewalk_min_fraction,
        min_coverage=config.prewalk_min_coverage,
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
    return_funnel: bool = False,
) -> List[Dict]:
    """
    Assign ground truth labels to a list of trials.

    Args:
        trials: List of trial data dictionaries
            (as returned by :func:`pipeline.io.extract_trial_data`).
        config: Threshold configuration.
        return_funnel: When True, each returned dict additionally
            carries a ``"funnel"`` entry recording which criterion
            stages the trial passed/failed.  Round-3 fix (Reviewer A
            BLK-3B): after the Round-2 re-run produced PREWALK=0 with
            no diagnostic trail, label collapses must be auditable —
            the funnel makes every rejection stage countable so a
            criterion that silently eliminates an entire behavioural
            class cannot pass unnoticed again.

    Returns:
        List of dicts, each containing:

        - ``session_id``          — str
        - ``trial_id``            — int
        - ``label``               — :class:`Label`
        - ``stimulus_onset_ms``   — float
        - ``trial_data``          — the original trial dict
        - ``funnel``              — dict of stage booleans
          (only when *return_funnel*)
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

        funnel: Dict[str, bool] = {}

        # ── Round-3 fix (Reviewer A BLK-3B, root cause): responder-first
        # branch order. ──
        # The previous order ran the PRE_ACTIVE baseline check FIRST.
        # "Walking at some point during baseline" is a strict superset of
        # "walking in the last second before onset" (the PREWALK
        # criterion), so every candidate PREWALK animal was structurally
        # absorbed by the PRE_ACTIVE branch before the stimulus-locked
        # response could be evaluated — 104/189 of the old PRE_ACTIVE
        # trials satisfy the post-stimulus sustained-speed criterion.
        # GI-mediated wind escape is DEFINED by a short-latency,
        # stimulus-locked burst (~50-100 ms latency; GI→TTM/CoLa), and
        # crickets mounting wind escape while walking are documented;
        # baseline activity cannot abolish that response.  The correct
        # semantics therefore test the STIMULUS-LOCKED response first and
        # split it by pre-stimulus state (PREWALK vs ESCAPE), reserving
        # PRE_ACTIVE for non-responders with high spontaneous activity.

        # Branch A: stimulus-locked escape response?
        post_stim = _check_sustained_speed(
            velocity, time_ms,
            start_ms=stimulus_onset,
            duration_ms=config.escape_sustained_ms,
            threshold=config.escape_velocity_threshold,
            min_fraction=config.sustained_min_fraction,
            max_latency_ms=config.response_max_latency_ms,
            anchor_min_frames=config.sustained_anchor_min_frames,
        )
        funnel["post_stim_sustained"] = bool(post_stim)

        if post_stim:
            # Responder: split by pre-stimulus locomotion.  The whole-
            # window fraction criterion over ``[onset -
            # prewalk_sustained_ms, onset)`` (Round-3 MAJ-3A) is
            # hard-bounded at stimulus onset, so only truly pre-stimulus
            # frames can satisfy it.
            pre_stim = check_prewalk_window(
                velocity, time_ms,
                stimulus_onset_ms=stimulus_onset,
                window_ms=config.prewalk_sustained_ms,
                threshold=config.prewalk_velocity_threshold,
                min_fraction=config.prewalk_min_fraction,
                min_coverage=config.prewalk_min_coverage,
            )
            funnel["pre_stim_window_fraction"] = bool(pre_stim)
            label = Label.PREWALK if pre_stim else Label.ESCAPE
        else:
            # Non-responder: was the animal spontaneously active?
            pre_active = is_pre_active(velocity, time_ms, stimulus_onset, config)
            funnel["pre_active_baseline"] = bool(pre_active)
            label = Label.PRE_ACTIVE if pre_active else Label.NO_RESPONSE

        entry: Dict = {
            "session_id": trial["session_id"],
            "trial_id": trial["trial_id"],
            "label": label,
            "stimulus_onset_ms": stimulus_onset,
            "trial_data": trial,
        }
        if return_funnel:
            entry["funnel"] = funnel
        labeled.append(entry)

    return labeled


def labeling_funnel_summary(labeled: List[Dict]) -> Dict[str, int]:
    """
    Aggregate per-trial funnel records into class-level stage counts.

    Round-3 fix (Reviewer A BLK-3B): the caller (prepare_data) logs this
    waterfall so the number of trials eliminated at each labelling
    criterion is reported whenever labels are (re)generated.

    Args:
        labeled: Output of :func:`assign_ground_truth_labels` called
            with ``return_funnel=True``.

    Returns:
        Dict mapping stage name to the number of trials FAILING it,
        plus per-label totals.
    """
    from collections import Counter

    label_counts: Counter = Counter()
    stage_failures: Counter = Counter()
    n_with_funnel = 0
    for info in labeled:
        label_counts[info["label"].name] += 1
        funnel = info.get("funnel")
        if not funnel:
            continue
        n_with_funnel += 1
        for stage, passed in funnel.items():
            if not passed:
                stage_failures[f"failed_{stage}"] += 1

    summary: Dict[str, int] = {
        f"n_{name}": int(cnt) for name, cnt in sorted(label_counts.items())
    }
    summary.update({stage: int(cnt) for stage, cnt in sorted(stage_failures.items())})
    summary["n_trials"] = len(labeled)
    return summary


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
