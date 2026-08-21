"""
Data I/O — loading and concatenating experimental sessions.

Defines the expected CSV column schemas and provides functions for loading
raw experimental data into pandas DataFrames and per-trial dictionaries.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# Expected CSV column schemas
# ──────────────────────────────────────────────────────────────

KINEMATICS_COLUMNS: List[str] = [
    "session_id",
    "trial_id",
    "time_ms",
    "x_pos",
    "y_pos",
    "heading",
    "velocity",
    "acceleration",
    "visual_angle",
    "wind_state",
    "l_v_ratio",
]

EVENT_COLUMNS: List[str] = [
    "session_id",
    "trial_id",
    "time_ms",
    "event_type",
    "event_value",
]


def _parse_event_value(value: Any) -> Dict[str, Any]:
    """Parse JSON-like event metadata without failing legacy scalar values.

    Handles both strict JSON (double quotes) and Python literal dicts
    (single quotes, e.g. ``{'wind_side': 'left'}``) that appear in legacy
    cercus exports. Falls back to ``{}`` for bare scalars / NaN.
    """
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    raw = str(value).strip()
    if not raw:
        return {}
    # Try strict JSON first (double-quoted)
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    # Fallback for single-quoted Python literals: "{'wind_side': 'left'}"
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalise_side(value: Any) -> str:
    """Return a canonical wind side or ``unknown`` for ambiguous metadata."""
    side = str(value).strip().lower() if value is not None else ""
    if side in {"left", "l"}:
        return "left"
    if side in {"right", "r"}:
        return "right"
    return "unknown"


def _trial_wind_side(
    events: pd.DataFrame, session_id: str, trial_id: int,
) -> str:
    """Resolve side from wind_onset, then trial_start metadata."""
    trial = events[
        (events["session_id"] == session_id) & (events["trial_id"] == trial_id)
    ]
    for event_type in ("wind_onset", "wind_onset_event"):
        values = trial.loc[trial["event_type"] == event_type, "event_value"]
        parsed = [_parse_event_value(value) for value in values]
        sides = {_normalise_side(item.get("wind_side", item.get("side"))) for item in parsed}
        sides.discard("unknown")
        if len(sides) == 1:
            return sides.pop()
        if len(sides) > 1:
            return "unknown"

    starts = trial.loc[trial["event_type"] == "trial_start", "event_value"]
    sides = {
        _normalise_side(
            item.get("wind_dir", item.get("wind_side", item.get("screen_side")))
        )
        for item in (_parse_event_value(value) for value in starts)
    }
    sides.discard("unknown")
    return sides.pop() if len(sides) == 1 else "unknown"


# ──────────────────────────────────────────────────────────────
# Single-file loaders
# ──────────────────────────────────────────────────────────────

def load_kinematics_csv(
    path: Union[str, Path],
    artifact_velocity_cm_s: float = 1000.0,
) -> pd.DataFrame:
    """
    Load a single kinematics CSV file.

    Validates that all required columns are present, sanitizes ``wind_state``
    to binary ``{0, 1}``, and zeros the phantom velocity/acceleration spike on
    a trial's first frame (``time_ms==0``) when it exceeds
    ``artifact_velocity_cm_s``.

    Real data contains rare single-sample spikes at ``time_ms==0`` (values
    12, 297, 602, 752, 852) that are legacy encoding artifacts — only ``1``
    is a true wind-on signal. Separately, the first frame of each trial
    carries a ``10^6``-cm/s-scale velocity/acceleration phantom spike from
    the raw sensor cross-trial position jump (see the fixture block below);
    both are zeroed here.

    Args:
        path: File path to the kinematics CSV.
        artifact_velocity_cm_s: Single-frame velocity magnitude above which
            a trial-first frame's velocity is treated as a cross-trial
            sensor-jump artifact (zeroed).  Real escape onsets are ~10^1-10^2
            cm/s; the sensor spike is 10^3+ cm/s, so ``1000`` is a safe
            three-order-of-magnitude separation.  ``float("inf")`` disables.

    Returns:
        DataFrame with columns matching :data:`KINEMATICS_COLUMNS`.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(path)
    missing = set(KINEMATICS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df[KINEMATICS_COLUMNS].copy()
    df["wind_state"] = pd.to_numeric(df["wind_state"], errors="coerce").fillna(0).eq(1).astype(np.int64)

    # ── Trial-boundary velocity/acceleration sanitization ──
    # NSMoR audit 2026-08-21: in the adapted cercus CSVs the first frame
    # of each trial (time_ms==0) can carry phantom velocity/acceleration
    # spikes of 10^6-10^7 cm/s. Root cause: the raw sensor dx/dy at the
    # reset frame encode the cross-trial arena-position jump (positions
    # restart at (0,0) but dx/dy still encode the displacement from the
    # previous trial's last sample), and velocity = step_dist/dt_clamped
    # yields the spur.  A blanket zero of *every* first frame, however,
    # would also erase legitimate wind-onset escape kinematics when the
    # first frame genuinely begins a fast evasive jump (finite sampling
    # cadence, nonzero dt).  We therefore zero a first frame ONLY when it
    # is an artifact-scale spike: an apparent single-frame velocity above the
    # physical plausibility bound ``artifact_velocity_cm_s``.  Solo escape
    # jumps are ~10^1-10^2 cm/s; the sensor-jump artifact is 10^3+ cm/s, so
    # a 3-order magnitude gap makes the bound safe.  Frames below the bound
    # keep their true onset value; only the artifact-scale spike is nulled
    # and the per-trial acceleration is recomputed from the cleaned velocity
    # (removing both the frame-0 velocity spike and its diff-echo in frame 1).
    if len(df) > 0:
        first_frame = df.groupby(["session_id", "trial_id"], sort=False)["time_ms"].transform("min")
        is_trial_first = df["time_ms"] == first_frame
        is_artifact = is_trial_first & (df["velocity"].abs() >= artifact_velocity_cm_s)
        df.loc[is_artifact, "velocity"] = 0.0
        # Recompute acceleration as per-trial diff of the cleaned velocity;
        # artifact first frames are 0 by construction, others keep theirs.
        df["acceleration"] = df.groupby(["session_id", "trial_id"], sort=False)["velocity"].transform(
            lambda v: v.diff().fillna(0.0)
        )
    return df


def load_events_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single events CSV file.

    Validates that all expected columns are present.

    Args:
        path: File path to the events CSV.

    Returns:
        DataFrame with columns matching :data:`EVENT_COLUMNS`.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(path)
    missing = set(EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[EVENT_COLUMNS]


# ──────────────────────────────────────────────────────────────
# Multi-session loader
# ──────────────────────────────────────────────────────────────

def load_and_concat_sessions(
    kinematics_paths: List[Union[str, Path]],
    events_paths: List[Union[str, Path]],
) -> Dict[str, pd.DataFrame]:
    """
    Load and concatenate multiple experimental sessions.

    Each session may span one kinematics CSV and one events CSV.
    All sessions are concatenated row-wise into two DataFrames.

    Args:
        kinematics_paths: List of paths to kinematics CSV files.
        events_paths: List of paths to events CSV files.

    Returns:
        ``{"kinematics": DataFrame, "events": DataFrame}``
    """
    kin_dfs = [load_kinematics_csv(p) for p in kinematics_paths]
    evt_dfs = [load_events_csv(p) for p in events_paths]

    return {
        "kinematics": pd.concat(kin_dfs, ignore_index=True),
        "events": pd.concat(evt_dfs, ignore_index=True),
    }


# ──────────────────────────────────────────────────────────────
# Per-trial extraction
# ──────────────────────────────────────────────────────────────

def extract_trial_data(
    session_data: Dict[str, pd.DataFrame],
    session_id: str,
    trial_id: int,
) -> Dict[str, np.ndarray]:
    """
    Extract all data for a single trial as a flat dictionary of arrays.

    Args:
        session_data: Output of :func:`load_and_concat_sessions`.
        session_id: Session identifier string.
        trial_id: Trial identifier integer.

    Returns:
        Dictionary with the following keys (all np.ndarray unless noted):

        - ``time_ms``          — float64, sorted ascending
        - ``x_pos``            — float64
        - ``y_pos``            — float64
        - ``heading``          — float64
        - ``velocity``         — float64 (cm / s)
        - ``acceleration``     — float64 (cm / s²)
        - ``visual_angle``     — float64 (degrees)
        - ``wind_state``       — float64 (0 or 1)
        - ``l_v_ratio``        — float64
        - ``event_times``      — float64, sorted ascending
        - ``event_types``      — object (str)
        - ``session_id``       — str (scalar)
        - ``trial_id``         — int (scalar)

    Raises:
        ValueError: If no matching rows are found.
    """
    kin = session_data["kinematics"]
    mask_kin = (kin["session_id"] == session_id) & (kin["trial_id"] == trial_id)
    kin_trial = kin.loc[mask_kin].sort_values("time_ms")

    if kin_trial.empty:
        raise ValueError(
            f"No kinematics data for session={session_id!r}, trial={trial_id}"
        )

    evt = session_data["events"]
    mask_evt = (evt["session_id"] == session_id) & (evt["trial_id"] == trial_id)
    evt_trial = evt.loc[mask_evt].sort_values("time_ms")
    wind_side = _trial_wind_side(evt, session_id, trial_id)

    time_ms_arr = kin_trial["time_ms"].to_numpy(dtype=np.float64)
    # Shape / monotonicity assertions (engineering rigor)
    assert time_ms_arr.ndim == 1, f"time_ms ndim {time_ms_arr.ndim} !=1"
    assert time_ms_arr.shape[0] > 0, "time_ms empty"
    assert np.all(np.diff(time_ms_arr) >= 0), "time_ms not sorted ascending"
    T = time_ms_arr.shape[0]
    for _col in ("x_pos", "y_pos", "heading", "velocity", "acceleration", "visual_angle", "wind_state", "l_v_ratio"):
        arr = kin_trial[_col].to_numpy(dtype=np.float64)
        assert arr.shape == (T,), f"{_col} shape {arr.shape} != ({T},)"

    return {
        "time_ms": time_ms_arr,
        "x_pos": kin_trial["x_pos"].to_numpy(dtype=np.float64),
        "y_pos": kin_trial["y_pos"].to_numpy(dtype=np.float64),
        "heading": kin_trial["heading"].to_numpy(dtype=np.float64),
        "velocity": kin_trial["velocity"].to_numpy(dtype=np.float64),
        "acceleration": kin_trial["acceleration"].to_numpy(dtype=np.float64),
        "visual_angle": kin_trial["visual_angle"].to_numpy(dtype=np.float64),
        "wind_state": kin_trial["wind_state"].to_numpy(dtype=np.float64),
        "l_v_ratio": kin_trial["l_v_ratio"].to_numpy(dtype=np.float64),
        "event_times": evt_trial["time_ms"].to_numpy(dtype=np.float64),
        "event_types": evt_trial["event_type"].to_numpy(),
        "event_values": evt_trial["event_value"].to_numpy(),
        "wind_side_original": wind_side,
        "wind_side_unified": wind_side,
        "session_id": session_id,
        "trial_id": trial_id,
    }
