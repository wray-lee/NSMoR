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

from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["derive_stimulus_metadata"]

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
