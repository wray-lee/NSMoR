"""
Data I/O — loading and concatenating experimental sessions.

Defines the expected CSV column schemas and provides functions for loading
raw experimental data into pandas DataFrames and per-trial dictionaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

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


# ──────────────────────────────────────────────────────────────
# Single-file loaders
# ──────────────────────────────────────────────────────────────

def load_kinematics_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single kinematics CSV file.

    Validates that all expected columns are present.

    Args:
        path: File path to the kinematics CSV.

    Returns:
        DataFrame with columns matching :data:`KINEMATICS_COLUMNS`.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(path)
    missing = set(KINEMATICS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[KINEMATICS_COLUMNS]


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

    return {
        "time_ms": kin_trial["time_ms"].to_numpy(dtype=np.float64),
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
        "session_id": session_id,
        "trial_id": trial_id,
    }
