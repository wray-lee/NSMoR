"""Declared experimental event parsing and trial identity resolution.

Provides keyed event lookup by stable (session_id, trial_id) across both
raw schema (flat filename stem session ID; event_name, timestamp, session_num,
trial_in_session, global_trial_id, details) and transformed staging schema
(session_id, trial_id, time_ms, event_type, event_value).
"""
from __future__ import annotations

import ast
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "parse_event_details",
    "parse_declared_ttc",
    "parse_events_file",
    "load_declared_events_index",
]


def parse_declared_ttc(raw: Any, session_id: str, trial_id: int) -> Optional[float]:
    """Parse declared target_ttc_ms strictly fail-closed.

    Distinguishes legitimate null from corrupt nonnull/nonfinite values:
    - None or empty/null literal -> None (legitimate null)
    - Bool (True/False or string 'true'/'false') -> raises ValueError (False must not become 0.0)
    - Finite float/int -> float
    - Non-finite (NaN, Inf, -Inf) -> raises ValueError
    - Corrupt non-null unparseable string -> raises ValueError
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(
            f"Boolean target_ttc_ms {raw!r} in trial [{session_id}, {trial_id}] is invalid; "
            "refusing boolean coercion to 0/1 (fail closed)."
        )
    if isinstance(raw, (int, float, np.integer, np.floating)):
        val = float(raw)
        if math.isnan(val) or math.isinf(val):
            raise ValueError(
                f"Non-finite target_ttc_ms {raw} in trial [{session_id}, {trial_id}] (fail closed)"
            )
        return val
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    if s.lower() in ("true", "false"):
        raise ValueError(
            f"Boolean string target_ttc_ms {raw!r} in trial [{session_id}, {trial_id}] is invalid; "
            "refusing boolean coercion to 0/1 (fail closed)."
        )
    try:
        val = float(s)
    except (ValueError, TypeError):
        raise ValueError(
            f"Corrupt target_ttc_ms (unparseable) {raw!r} in trial [{session_id}, {trial_id}] (fail closed)"
        )
    if math.isnan(val) or math.isinf(val):
        raise ValueError(
            f"Non-finite target_ttc_ms {raw!r} in trial [{session_id}, {trial_id}] (fail closed)"
        )
    return val


def parse_event_details(value: Any) -> Dict[str, Any]:
    """Parse JSON or Python-dict literal event details safely.

    Handles dict, JSON strings, Python literal dict strings, and missing values.
    Returns empty dict on legitimate null/missing values or numeric event indicators (e.g. 1/0).
    Raises ValueError on malformed non-empty details (fail-closed).
    """
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, bool):
        raise ValueError(
            f"Malformed event details: boolean value {value!r} is not a valid dictionary (fail closed)"
        )
    if isinstance(value, (int, float, np.integer, np.floating)):
        if math.isnan(float(value)):
            return {}
        # Numeric event flag (e.g. 1 / 0) carried in event_value column
        return {}
    raw = str(value).strip()
    if not raw or raw.lower() in ("null", "none", "nan"):
        return {}
    try:
        fval = float(raw)
        if not math.isnan(fval) and not math.isinf(fval):
            return {}
    except (ValueError, TypeError):
        pass
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError, TypeError, MemoryError):
        pass
    raise ValueError(f"Malformed non-empty event details string: {raw!r} (fail closed)")


def _extract_session_id_from_path(path: Path) -> str:
    """Extract session_id from filename stem.

    Example:
        '0.513cricket_001_20260707_193143_session_1_events.csv'
        -> '0.513cricket_001_20260707_193143_session_1'
    """
    stem = path.stem
    if stem.endswith("_events"):
        return stem[:-7]
    return stem


def parse_events_file(
    path: Union[str, Path],
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Parse an events CSV file into a keyed dictionary by (session_id, trial_id).

    Supports:
    - Raw schema: event_name, timestamp, session_num, trial_in_session, global_trial_id, details
      with session_id derived from filename stem.
    - Transformed schema: session_id, trial_id, time_ms, event_type, event_value.

    Prioritizes global_trial_id in raw schema, registering trial_in_session as alias
    when distinct and non-conflicting.
    Rejects duplicate conflicting trial_start entries.

    Returns:
        Dict mapping (session_id, trial_id) -> {
            "session_id": str,
            "trial_id": int,
            "type": str,
            "target_ttc_ms": Optional[float],
            "lv_ratio_ms": Optional[float],
            "wind_dir": Optional[str],
            "details": Dict[str, Any],
        }
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Events file not found: {file_path}")

    # Handle utf-8-sig BOM if present
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    if df.empty:
        return {}

    # Identify columns
    cols = set(df.columns)
    is_transformed = "session_id" in cols and "trial_id" in cols and "event_type" in cols

    evt_name_col = "event_type" if "event_type" in cols else ("event_name" if "event_name" in cols else None)
    if evt_name_col is None:
        raise ValueError(f"Neither 'event_type' nor 'event_name' found in {file_path}")

    details_col = "event_value" if "event_value" in cols else ("details" if "details" in cols else None)
    if details_col is None:
        raise ValueError(f"Neither 'event_value' nor 'details' found in {file_path}")

    fallback_session_id = _extract_session_id_from_path(file_path)

    # Filter to trial_start rows
    trial_start_mask = df[evt_name_col].astype(str) == "trial_start"
    ts_df = df.loc[trial_start_mask]

    trials_map: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def _register(key: Tuple[str, int], entry_dict: Dict[str, Any]) -> None:
        if key in trials_map:
            prev = trials_map[key]
            # Any declared detail can affect the scientific interpretation.
            if prev != entry_dict:
                raise ValueError(
                    f"Conflicting duplicate trial_start for session {key[0]}, trial {key[1]}: "
                    f"existing {prev} vs new {entry_dict}"
                )
            # Identical duplicate: keep existing
            return
        trials_map[key] = entry_dict

    for _, row in ts_df.iterrows():
        alias_trial_id: Optional[int] = None
        if is_transformed:
            sess_id = str(row["session_id"])
            trial_id = int(row["trial_id"])
        else:
            sess_id = str(row.get("session_id", fallback_session_id))
            # In raw schema, global_trial_id is prioritized
            if "global_trial_id" in cols and pd.notna(row["global_trial_id"]):
                trial_id = int(row["global_trial_id"])
                if "trial_in_session" in cols and pd.notna(row["trial_in_session"]):
                    alias_trial_id = int(row["trial_in_session"])
            elif "trial_id" in cols and pd.notna(row["trial_id"]):
                trial_id = int(row["trial_id"])
            elif "trial_in_session" in cols and pd.notna(row["trial_in_session"]):
                trial_id = int(row["trial_in_session"])
            else:
                raise ValueError(f"No trial identifier column found in {file_path}")

        details = parse_event_details(row[details_col])
        trial_type = str(details.get("type", "unknown"))

        raw_ttc = details.get("target_ttc_ms")
        target_ttc_ms = parse_declared_ttc(raw_ttc, sess_id, trial_id)

        raw_lv = details.get("lv_ratio_ms")
        lv_ratio_ms: Optional[float] = None
        if raw_lv is not None and not (isinstance(raw_lv, float) and np.isnan(raw_lv)):
            try:
                val_lv = float(raw_lv)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Corrupt lv_ratio_ms {raw_lv!r} in events for session {sess_id}, "
                    f"trial {trial_id}: {e}"
                ) from e
            if math.isnan(val_lv) or math.isinf(val_lv):
                raise ValueError(
                    f"Non-finite lv_ratio_ms {val_lv} in events for session {sess_id}, "
                    f"trial {trial_id}"
                )
            lv_ratio_ms = val_lv

        wind_dir = details.get("wind_dir", details.get("wind_side", "none"))

        entry = {
            "session_id": sess_id,
            "trial_id": trial_id,
            "type": trial_type,
            "target_ttc_ms": target_ttc_ms,
            "lv_ratio_ms": lv_ratio_ms,
            "wind_dir": str(wind_dir) if wind_dir is not None else "none",
            "details": details,
        }

        _register((sess_id, trial_id), entry)

    return trials_map


def load_declared_events_index(
    raw_dir: Union[str, Path],
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Scan raw directory and build index of all declared trial events.

    Supports flat directories (e.g. D:/Data/train/all) or nested session dirs.
    Merges all events files into a single (session_id, trial_id) map.
    Fails closed if conflicting duplicates exist across files.
    """
    root = Path(raw_dir)
    if not root.exists():
        return {}

    # Find all event CSV candidates
    event_files = sorted(root.rglob("*events*.csv"))
    if not event_files:
        event_files = sorted(root.glob("*events*.csv"))

    merged: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for evt_path in event_files:
        try:
            file_trials = parse_events_file(evt_path)
        except Exception as e:
            logger.warning("Failed parsing %s: %s", evt_path, e)
            raise
        for key, entry in file_trials.items():
            if key in merged:
                prev = merged[key]
                if prev != entry:
                    raise ValueError(
                        f"Conflicting duplicate trial_start across files for {key}: "
                        f"existing {prev} vs new {entry} from {evt_path.name}"
                    )
                continue
            merged[key] = entry

    return merged
