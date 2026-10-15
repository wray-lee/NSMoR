"""Stimulus condition derivation from physical input channels.

The four conditions are a function of two physical channels only -- visual
angle at feature index 0 and wind state at index 1 -- never of behavioural
labels or session names:

===============  ======================================
condition        channels
===============  ======================================
multisensory     visual present AND wind present
visual_only      visual present, wind absent
wind_only        wind present, visual absent
no_stimulus      neither present
===============  ======================================

``is_pure_wind`` is exactly ``condition == "wind_only"``.  The distinction
matters: a ``no_stimulus`` trial also has a silent visual channel, so any
test of the form "visual is silent" folds it into the pure-wind group and
contaminates every per-condition statistic derived from it.

This lives in the installed package rather than in a CLI script because
three entry points need it -- ``scripts/train.py``,
``scripts/analyze_gating.py`` and ``scripts/make_subset_dataset.py``.  A
cross-script import works under pytest (repo root on ``sys.path``) but
breaks ``python scripts/<name>.py``, since ``pyproject.toml`` installs
``nsmor*`` and not ``scripts``.

The module also owns the SHARED stimulus-event vocabulary introduced for
the phase-3 wind-lagging paradigms (ADR 0011, PIPELINE_SEMANTICS_VERSION
2.3): one wind-presence threshold, the first-stimulus-event reference
rule (``min(wind onset, collision)``), the signed per-trial SOA
``collision - wind onset`` (positive = wind first), its declared
paradigm bucket, and the variant key derived from
``(target_ttc_ms, SOA bucket)``.  Every consumer -- the staging adapter,
the ETL and the behavior analysis -- reads these definitions from here so
the condition stamp, the anchor and the SOA can never disagree.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "WIND_PRESENCE_THRESHOLD",
    "INIT_DEG_DEFAULT",
    "looming_ttc_ms",
    "SOA_BUCKET_MS",
    "SOA_HOMOGENEITY_TOLERANCE_FRAMES",
    "derive_stimulus_metadata",
    "derive_anchor_frames",
    "resolve_anchor_crop",
    "wind_onset_frame",
    "visual_collision_frame",
    "first_stimulus_frame",
    "stimulus_soa_ms",
    "soa_bucket_ms",
    "stimulus_variant_key",
    "stimulus_reference_event",
]

# Feature-axis indices of the two physical stimulus channels.
_VISUAL_ANGLE_IDX = 0
_WIND_STATE_IDX = 1

WIND_PRESENCE_THRESHOLD: float = 0.5
"""Single declared wind-presence threshold (channel value strictly above).

One threshold guards the physical wind channel everywhere: the condition
stamp (:func:`derive_stimulus_metadata`), the anchor rule
(:func:`derive_anchor_frames`), the staging adapter
(``scripts/pre_load_adapt.py``) and the behavior analysis
(``scripts/analyze_behavior.py``).  Before this unification the condition
stamp used ``|wind| > 0`` while the anchor/onset detection used
``> 0.5``; a corpus coding wind as e.g. 0.4 was stamped ``multisensory``
while its anchor fell back and its wind onset was never found, feeding a
garbage SOA.  The predicate is strictly positive (``wind > threshold``,
no absolute value): presence and onset then agree by construction.
"""

SOA_BUCKET_MS: float = 50.0
"""Declared granularity (ms) of the multisensory SOA-bucket variant key.

The bucket separates paradigms that share a ``target_ttc_ms`` but differ
in wind timing (the phase-3 wind-at-80-deg / wind-at-collision /
wind-200-ms-after-collision paradigms).  50 ms is far below the spacing
between the existing corpus's paradigm SOAs (356/288/244/208 ms) and the
new SOAs (0 / -200 ms), and far above one model frame; within-bucket
homogeneity is enforced separately by the SOA-homogeneity audit
(:data:`SOA_HOMOGENEITY_TOLERANCE_FRAMES`).
"""

SOA_HOMOGENEITY_TOLERANCE_FRAMES: float = 1.0
"""Default tolerance (frames) for the within-variant SOA-spread audit.

A variant is one ``(target_ttc_ms, SOA bucket)`` cell.  The audit fails
closed when the spread (max - min) of the per-trial signed SOA inside a
cell exceeds ``tolerance_frames * dt_ms``: a cell whose trials do not
share one wind->collision delay cannot supply one Miller-bound shift.
The declared tolerance is a per-run parameter
(``analyze_behavior.py --soa_tolerance_frames``); corpora with known
wind-delivery jitter (the phase-2 corpus measures 6-7 frames) must
declare a larger tolerance explicitly rather than silently pooling.
"""


def _valid_slice(x_seq: np.ndarray, length: Optional[int]) -> np.ndarray:
    """Return the ``(L, C)`` valid prefix of a per-trial feature array."""
    valid_len = int(length) if length is not None else len(x_seq)
    return np.asarray(x_seq)[:valid_len]


INIT_DEG_DEFAULT: float = 2.0
"""Declared initial visual angle (deg) of the looming paradigm.

Shared by the staging adapter's visual reconstruction, its injected
collision clock and the visual-only snapshot anchor geometry.
"""


def looming_ttc_ms(
    lv_ratio_ms: float, init_deg: float = INIT_DEG_DEFAULT
) -> float:
    """Time-to-collision (ms) from looming onset, from the l/v geometry.

    Mirrors the experiment code (``paradigm.py``):
    ``t_col = (lv_ratio_ms / 1000) / tan(radians(init_deg / 2))``.

    Args:
        lv_ratio_ms: Object size / approach speed in ms; must be > 0.
        init_deg: Initial visual angle in degrees.

    Returns:
        Collision time in ms relative to the looming onset.

    Raises:
        ValueError: If ``lv_ratio_ms`` is not a positive finite number or
            the geometry is degenerate.
    """
    import math

    if not (isinstance(lv_ratio_ms, (int, float))
            and math.isfinite(float(lv_ratio_ms))
            and float(lv_ratio_ms) > 0.0):
        raise ValueError(
            f"looming_ttc_ms requires a positive finite lv_ratio_ms, got "
            f"{lv_ratio_ms!r} (fail closed)"
        )
    tan_val = math.tan(math.radians(float(init_deg) / 2.0))
    if not (math.isfinite(tan_val) and tan_val > 0.0):
        raise ValueError(
            f"degenerate looming geometry: tan(init_deg/2)={tan_val!r} for "
            f"init_deg={init_deg!r} (fail closed)"
        )
    return float(lv_ratio_ms) / tan_val


def wind_onset_frame(
    x_seq: np.ndarray, length: Optional[int] = None
) -> int:
    """First frame whose wind channel exceeds the declared threshold.

    Args:
        x_seq: ``(T, C)`` trial feature array with wind at index 1.
        length: Optional valid (unpadded) frame count.

    Returns:
        Frame index, or ``-1`` when the wind channel never activates.
    """
    wind = _valid_slice(x_seq, length)[:, _WIND_STATE_IDX]
    idx = np.where(wind > WIND_PRESENCE_THRESHOLD)[0]
    return int(idx[0]) if idx.size else -1


def visual_collision_frame(
    x_seq: np.ndarray, length: Optional[int] = None
) -> int:
    """Frame of the visual collision (peak visual angle) for a trial.

    Args:
        x_seq: ``(T, C)`` trial feature array with visual angle at index 0.
        length: Optional valid (unpadded) frame count.

    Returns:
        The ``argmax`` frame of the visual channel when the channel is
        present (any ``|angle| > 1e-4``), else ``-1``.
    """
    vis = _valid_slice(x_seq, length)[:, _VISUAL_ANGLE_IDX]
    if not bool(np.any(np.abs(vis) > 1e-4)):
        return -1
    return int(np.argmax(vis))


def first_stimulus_frame(
    x_seq: np.ndarray, length: Optional[int] = None
) -> int:
    """First stimulus event of a trial: ``min(wind onset, collision)``.

    The reference event for the label clock and for the behavior
    analysis's D1 trigger/gate/search (ADR 0011).  For the wind-first
    corpora this is the wind onset, exactly as before; for wind-lagging
    paradigms (wind at or after the collision) it is the collision.

    Args:
        x_seq: ``(T, C)`` trial feature array.
        length: Optional valid (unpadded) frame count.

    Returns:
        The earliest present event frame, or ``-1`` when neither channel
        is present.
    """
    candidates = [
        frame for frame in (
            wind_onset_frame(x_seq, length),
            visual_collision_frame(x_seq, length),
        )
        if frame >= 0
    ]
    return int(min(candidates)) if candidates else -1


def stimulus_soa_ms(
    x_seq: np.ndarray, length: Optional[int], dt_ms: float
) -> float:
    """Signed per-trial wind->collision SOA in ms from the channels.

    ``SOA = (collision_frame - wind_onset_frame) * dt_ms``; positive
    means the wind comes first.  ``NaN`` when either event is absent.

    Args:
        x_seq: ``(T, C)`` trial feature array.
        length: Optional valid (unpadded) frame count.
        dt_ms: Frame interval in ms.

    Returns:
        Signed SOA in ms, or ``NaN`` when either channel is silent.
    """
    w_on = wind_onset_frame(x_seq, length)
    v_col = visual_collision_frame(x_seq, length)
    if w_on < 0 or v_col < 0:
        return float("nan")
    return float((v_col - w_on) * dt_ms)


def soa_bucket_ms(soa_ms: float, bucket_ms: float = SOA_BUCKET_MS) -> float:
    """Round a signed SOA (ms) onto the declared paradigm bucket grid."""
    if not np.isfinite(soa_ms):
        return float("nan")
    return float(np.round(float(soa_ms) / float(bucket_ms)) * float(bucket_ms))


def stimulus_variant_key(
    condition: str,
    ttc_ms: float,
    soa_ms: float,
    bucket_ms: float = SOA_BUCKET_MS,
) -> str:
    """Paradigm/variant key for a trial: ``(target_ttc_ms, SOA bucket)``.

    Non-multisensory trials are keyed by their condition name.  A
    multisensory trial is keyed by its declared ``target_ttc_ms`` and the
    bucket of its signed SOA, so two paradigms that share a TTC but
    differ in wind timing can never merge into one variant.  Undeclared
    (non-finite) TTC or SOA is keyed ``unknown`` rather than guessed;
    the analysis entry point fails fast on undeclared multisensory TTC.
    """
    if condition != "multisensory":
        return str(condition)
    ttc_token = (
        str(int(round(float(ttc_ms))))
        if (ttc_ms is not None and np.isfinite(float(ttc_ms)))
        else "unknown"
    )
    soa_token = (
        str(int(round(soa_bucket_ms(soa_ms, bucket_ms))))
        if np.isfinite(soa_ms) else "unknown"
    )
    return f"multisensory_ttc{ttc_token}_soa{soa_token}"


def stimulus_reference_event(condition: str, soa_ms: float) -> str:
    """The per-trial reference event the label clock is anchored to.

    ``wind_onset`` for wind-only trials and for multisensory trials whose
    wind comes first (SOA >= 0); ``visual_collision`` for visual-only
    trials and for wind-lagging multisensory trials (SOA < 0);
    ``none`` for trials with no stimulus channel.
    """
    if condition == "multisensory":
        if not np.isfinite(soa_ms):
            return "none"
        return "wind_onset" if soa_ms >= 0.0 else "visual_collision"
    if condition == "wind_only":
        return "wind_onset"
    if condition == "visual_only":
        return "visual_collision"
    return "none"


def derive_stimulus_metadata(
    x_seqs: Sequence[np.ndarray],
    lengths: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive auditable condition metadata from physical input channels.

    Mirrors ``scripts.prepare_data.classify_stimulus_condition`` so legacy
    processed artifacts -- which predate the explicit condition stamp --
    are usable by the routing-aux path without guessing from behavioural
    labels or session names.

    Args:
        x_seqs: Per-trial feature arrays with visual angle at index 0 and
            wind state at index 1.
        lengths: Valid (unpadded) frame count for each trial.

    Returns:
        ``(stimulus_conditions, is_pure_wind)`` aligned 1:1 with
        ``x_seqs``; ``stimulus_conditions`` holds the condition names and
        ``is_pure_wind`` is the ``wind_only`` indicator.

    Raises:
        ValueError: If the two sequences disagree in length, or a trial's
            declared length exceeds its array.
    """
    if len(x_seqs) != len(lengths):
        raise ValueError(
            f"x_seqs/lengths mismatch: {len(x_seqs)} != {len(lengths)}"
        )

    conditions: List[str] = []
    for index, (x_seq, length) in enumerate(zip(x_seqs, lengths)):
        valid_length = int(length)
        if valid_length < 1 or valid_length > len(x_seq):
            raise ValueError(
                f"trial {index}: invalid length {valid_length} for "
                f"sequence length {len(x_seq)}"
            )
        physical = np.asarray(x_seq)[:valid_length, :2]
        has_visual = bool(np.any(np.abs(physical[:, _VISUAL_ANGLE_IDX]) > 0.0))
        # Unified wind-presence predicate (strictly above the declared
        # threshold, no absolute value) so the condition stamp and the
        # anchor rule can never disagree on whether wind is present.
        has_wind = bool(
            np.any(physical[:, _WIND_STATE_IDX] > WIND_PRESENCE_THRESHOLD)
        )
        if has_visual and has_wind:
            conditions.append("multisensory")
        elif has_visual:
            conditions.append("visual_only")
        elif has_wind:
            conditions.append("wind_only")
        else:
            conditions.append("no_stimulus")

    stimulus_conditions = np.asarray(conditions, dtype=object)
    is_pure_wind = stimulus_conditions == "wind_only"
    return stimulus_conditions, is_pure_wind


def derive_anchor_frames(
    x_seqs: Sequence[np.ndarray],
    lengths: Sequence[int] | None = None,
) -> List[int]:
    """Derive the FIRST STIMULUS EVENT anchor frame for each sequence.

    Uses physical channels (visual angle at index 0, wind state at index 1):
    - Trials with both channels present (multisensory), including
      wind-lagging paradigms, anchor at ``min(wind onset, collision)``.
    - Wind-only trials anchor at the first wind activation frame.
    - Visual-only trials anchor at the peak visual angle (looming
      collision) frame.
    - No-stimulus trials (neither present) anchor at 0.

    For every wind-first corpus (wind onset strictly before the
    collision) this equals the historical "first wind frame" rule
    exactly; it changes only for wind-lagging multisensory trials, whose
    reference event is the collision (ADR 0011).  This is the same
    first-stimulus-event rule the staging adapter injects as the
    ``stimulus_onset`` label clock, so the anchor, the label clock and
    the behavior analysis's D1 trigger agree by construction.

    Args:
        x_seqs: List of (T, 8) trial feature arrays.
        lengths: Optional valid (unpadded) sequence lengths.

    Returns:
        List of integer anchor frame indices, aligned 1:1 with x_seqs.
    """
    if lengths is not None and len(x_seqs) != len(lengths):
        raise ValueError(
            f"x_seqs/lengths mismatch: {len(x_seqs)} != {len(lengths)}"
        )

    anchors: List[int] = []
    for idx, x_seq in enumerate(x_seqs):
        valid_len = int(lengths[idx]) if lengths is not None else len(x_seq)
        first = first_stimulus_frame(np.asarray(x_seq), valid_len)
        anchors.append(first if first >= 0 else 0)
    return anchors


def resolve_anchor_crop(
    n_frames: int,
    anchor_frame: Optional[int],
    max_seq_len: Optional[int],
    pre_anchor_frames: int,
) -> Tuple[int, int]:
    """Return the ``[start, end)`` crop window applied to a trial sequence.

    Single source of truth for the anchor-aligned crop arithmetic shared by
    ``NSMoRDataset.__getitem__``, ``NSMoRLazyDataset.__getitem__`` and
    ``scripts.simulate_psychophysics.load_validation_data``.  Downstream
    analysis must map pre-crop metadata anchors into saved-sequence
    coordinates with the same window, otherwise gate/latency extraction
    silently reads a different time base than the tensors it scores.

    Args:
        n_frames: Full (pre-crop) sequence length.
        anchor_frame: Stimulus anchor frame index in pre-crop coordinates.
            ``None`` or negative selects the whole sequence (no crop).
        max_seq_len: Maximum sequence length; ``None`` disables cropping.
        pre_anchor_frames: Frames before the anchor to retain (baseline).

    Returns:
        ``(start, end)`` with ``end - start <= max_seq_len`` and
        ``0 <= start <= end <= n_frames``.  Returns ``(0, n_frames)`` when
        cropping is not required or not possible.
    """
    if max_seq_len is None or n_frames <= max_seq_len:
        return 0, n_frames
    if anchor_frame is None or anchor_frame < 0:
        return 0, n_frames
    start = max(0, int(anchor_frame) - int(pre_anchor_frames))
    end = min(n_frames, start + int(max_seq_len))
    # Adjust start if end clamped (keeps window size consistent)
    if end - start < int(max_seq_len):
        start = max(0, end - int(max_seq_len))
    return start, end
