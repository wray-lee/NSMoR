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
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["derive_stimulus_metadata", "derive_anchor_frames", "resolve_anchor_crop"]

# Feature-axis indices of the two physical stimulus channels.
_VISUAL_ANGLE_IDX = 0
_WIND_STATE_IDX = 1


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
        has_wind = bool(np.any(np.abs(physical[:, _WIND_STATE_IDX]) > 0.0))
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
    """Derive stimulus/collision anchor frame index for each sequence.

    Uses physical channels (visual angle at index 0, wind state at index 1):
    - For trials with wind activity (channel 1 > 0.5), anchor is the first
      wind activation frame.
    - For visual-only trials (wind absent, visual present), anchor is the peak
      visual angle (looming collision) frame.
    - For no-stimulus trials (neither present), anchor defaults to 0.

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
        vis = np.asarray(x_seq)[:valid_len, _VISUAL_ANGLE_IDX]
        wind = np.asarray(x_seq)[:valid_len, _WIND_STATE_IDX]

        has_wind = bool(np.any(wind > 0.5))
        has_vis = bool(np.any(np.abs(vis) > 1e-4))

        if has_wind:
            wind_active = np.where(wind > 0.5)[0]
            anchor = int(wind_active[0]) if len(wind_active) > 0 else 0
        elif has_vis:
            anchor = int(np.argmax(vis))
        else:
            anchor = 0
        anchors.append(anchor)
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
