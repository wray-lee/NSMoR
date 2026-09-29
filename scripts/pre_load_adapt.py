"""Cercus sensor to canonical NSMoR dataset adaptation script.

Converts archived legacy or raw-sensor session CSVs into the canonical NSMoR schema.

Modes:
  * ``--output_dir``: non-destructive staging into nested session subdirectories
    consumable by ``prepare_data.pair_csv_files``. The destination must be absent
    or an existing empty directory. Source is strictly read-only.
  * Legacy in-place: active whenever ``--output_dir`` is omitted (``--in_place``
    is an explicit confirmation of that same behavior). In-place execution is
    strictly prohibited on the authoritative raw dataset
    (``/mnt/d/Data/train/all`` and aliases).

Hardcoded protected-path strings are a backstop only; universal source/target
separation (``validate_output_paths`` / ``adapt_session_pair`` samefile and
ancestor identity checks) is the primary safety mechanism and applies to both
the batch and pair public APIs.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import stat as stat_mod
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

KIN_TARGET: List[str] = [
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
EVT_TARGET: List[str] = [
    "session_id",
    "trial_id",
    "time_ms",
    "event_type",
    "event_value",
]

_PROTECTED_RAW_PATHS: List[Path] = [
    Path("/mnt/d/Data/train/all"),
]
if sys.platform == "win32":
    _PROTECTED_RAW_PATHS.extend([
        Path(r"D:\Data\train\all"),
        Path("D:/Data/train/all"),
    ])


def is_protected_raw_path(path: Union[str, Path, None]) -> bool:
    """Check if path resolves or aliases to authoritative read-only raw data.

    Guards against accidental in-place mutation of the authoritative raw
    dataset (/mnt/d/Data/train/all).
    """
    if path is None:
        return False
    path_str = str(path).replace("\\", "/").lower().rstrip("/")
    _PROTECTED_PREFIXES = (
        "d:/data/train/all",
        "/mnt/d/data/train/all",
    )
    for prefix in _PROTECTED_PREFIXES:
        if path_str == prefix or path_str.startswith(prefix + "/"):
            return True
    try:
        p = Path(path).resolve()
    except Exception:
        return False

    for prot in _PROTECTED_RAW_PATHS:
        try:
            if not prot.is_absolute():
                continue
            pr = prot.resolve()
        except Exception:
            continue
        if p == pr or pr in p.parents:
            return True
        if pr.exists():
            if p.exists():
                try:
                    if os.path.samefile(p, pr):
                        return True
                except (FileNotFoundError, ValueError, OSError):
                    pass
            for parent in p.parents:
                if parent.exists():
                    try:
                        if os.path.samefile(parent, pr):
                            return True
                    except (FileNotFoundError, ValueError, OSError):
                        pass
    return False


def _validate_session_id(session_id: str) -> str:
    """Validate session_id is a safe single filesystem path component.

    Rejects path separators (POSIX and Windows), drive/UNC/absolute forms,
    '.'/'..', NUL bytes, and empty names. Legitimate periods inside names
    (e.g. decimal mass prefixes like '0.513cricket_001') are allowed.

    Args:
        session_id: Candidate identifier string.

    Returns:
        The validated session_id unchanged.

    Raises:
        ValueError: If session_id is not a safe single path component.
    """
    if not isinstance(session_id, str) or session_id == '':
        raise ValueError('session_id must be a non-empty string')
    if chr(0) in session_id:
        raise ValueError('session_id must not contain NUL bytes')
    _BS = chr(92)  # backslash
    if '/' in session_id or _BS in session_id:
        raise ValueError(
            f'session_id must be a single path component (no separators): {session_id!r}'
        )
    if session_id in ('.', '..'):
        raise ValueError(f"session_id must not be '.' or '..': {session_id!r}")
    if Path(session_id).is_absolute():
        raise ValueError(f'session_id must not be an absolute path: {session_id!r}')
    if len(session_id) >= 2 and session_id[1] == ':':
        raise ValueError(
            f'session_id must not carry a drive prefix: {session_id!r}'
        )
    return session_id


def _samefile_or_fail(a: Union[str, Path], b: Union[str, Path]) -> bool:
    """Return True iff a and b are the same filesystem object.

    Fails CLOSED on any error that prevents establishing identity.
    FileNotFoundError alone is tolerated (path vanished after an existence
    check), consistent with exclusive-creation safety elsewhere.
    """
    try:
        return os.path.samefile(a, b)
    except FileNotFoundError:
        return False
    except ValueError as exc:
        raise ValueError(
            f'Cannot establish filesystem identity between {a!r} and {b!r}: {exc}'
        ) from exc
    except OSError as exc:
        raise ValueError(
            f'Cannot establish filesystem identity between {a!r} and {b!r} '
            f'(failing closed): {exc}'
        ) from exc


def _reject_symlink_components(path: Union[str, Path], *, kind: str = 'path') -> None:
    """Reject if any existing component of path (ancestors or leaf) is a symlink.

    Args:
        path: Destination path whose components must not include symlinks.
        kind: Label used in the error message.

    Raises:
        FileExistsError: If any existing component is a symlink.
    """
    p = Path(path)
    components: List[Path] = []
    cur: Path = p
    while True:
        components.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    for comp in reversed(components):
        if comp.is_symlink():
            raise FileExistsError(
                f'Target {kind} component cannot be a symlink: {comp!r}'
            )


def _publish_no_replace(tmp: Path, target: Path, published: List[Tuple[Path, int, int]]) -> None:
    """Publish tmp to target with genuine no-replace semantics via hard link.

    Records (target, st_dev, st_ino) in published so rollback can verify
    ownership before unlinking. Refuses any overwrite fallback.
    """
    _reject_symlink_components(target.parent, kind='output session directory')
    try:
        os.link(tmp, target)
    except FileExistsError as exc:
        raise FileExistsError(
            f'Target file already exists: {target!r}. Refusing to overwrite.'
        ) from exc
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise OSError(
                f'Cannot publish {tmp!r} to {target!r}: cross-device hard link. '
                'Refusing overwrite fallback.'
            ) from exc
        raise
    try:
        st = os.stat(tmp)
    except OSError as exc:
        # Do not bare-unlink target here: another process may have replaced target.
        raise OSError(
            f'Cannot stat published temporary {tmp!r} for ownership record: {exc}'
        ) from exc
    published.append((target, st.st_dev, st.st_ino))
    try:
        tmp.unlink()
    except OSError:
        pass


def _rollback_owned(
    owned_tmps: List[Tuple[Path, int, int]],
    published: List[Tuple[Path, int, int]],
) -> None:
    """Delete only positively owned temporaries and published outputs.

    A temporary or published target is removed only when lstat still matches the inode
    recorded at creation/publish time and it is not a symlink (i.e. another writer
    has not replaced it).
    """
    for tmp, dev, ino in owned_tmps:
        try:
            st = os.lstat(tmp)
        except OSError:
            continue
        if stat_mod.S_ISLNK(st.st_mode):
            continue
        if st.st_dev != dev or st.st_ino != ino:
            continue
        try:
            os.unlink(tmp)
        except OSError:
            pass
    for target, dev, ino in published:
        try:
            st = os.lstat(target)
        except OSError:
            continue
        if stat_mod.S_ISLNK(st.st_mode):
            continue
        if st.st_dev != dev or st.st_ino != ino:
            continue
        try:
            os.unlink(target)
        except OSError:
            pass


def _atomic_replace(df: pd.DataFrame, target: Path) -> None:
    """Safely replace target in-place without mutating other hard link aliases."""
    fd, tmp_name = tempfile.mkstemp(
        prefix='.' + target.name + '.',
        suffix='.tmp',
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as fh:
            df.to_csv(fh, index=False)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(tmp_path, 0o644)
        except OSError:
            pass
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def reconstruct_visual_angle(
    time_ms: np.ndarray,
    lv_ratio_ms: Optional[float],
    init_deg: float = 2.0,
) -> np.ndarray:
    """Reconstruct visual looming angle θ(t) = 2 × arctan(lv / (TTC - t)).

    From the experiment code (paradigm.py):
        lv_s = lv_ratio_ms / 1000.0
        init_rad = math.radians(init_deg / 2)
        t_col = lv_s / math.tan(init_rad)  # TTC in seconds
        theta = math.degrees(2 * math.atan(lv_s / delta))  # delta = t_col - elapsed_s

    Args:
        time_ms: Array of timestamps in ms (relative to looming onset).
        lv_ratio_ms: l/v ratio in ms (object size / approach speed).
        init_deg: Initial visual angle in degrees (default 2.0).

    Returns:
        Array of visual angles in degrees.
    """
    visual_angle = np.zeros_like(time_ms, dtype=np.float64)
    # Pre-onset baseline: the object sits at init_deg before looming starts.
    visual_angle[time_ms < 0] = init_deg

    if lv_ratio_ms is None or lv_ratio_ms <= 0:
        return visual_angle

    # Compute TTC from experiment parameters (matching paradigm.py)
    lv_s = lv_ratio_ms / 1000.0
    init_rad = math.radians(init_deg / 2)
    tan_val = math.tan(init_rad)
    if tan_val <= 0:
        return visual_angle
    t_col_s = lv_s / tan_val  # TTC in seconds from looming onset
    ttc_ms = t_col_s * 1000.0  # Convert to ms

    # Compute visual angle for all time points where t < TTC
    # time_ms is relative to looming onset (0 = start of looming)
    mask = (time_ms >= 0) & (time_ms < ttc_ms)
    t_s = time_ms[mask] / 1000.0  # Convert to seconds

    # θ(t) = 2 × arctan(lv_s / (t_col - t))  — continuous through TTC.
    # No 0.001 s delta floor: the floor capped pre-TTC theta at
    # 2*arctan(lv_s/0.001) and then the post-TTC hold slammed to 179°,
    # producing up to an 89° single-frame step for small lv (v5 NEW-2 /
    # V5B-F2).  arctan2 handles the t→TTC⁻ limit (delta→0 ⇒ θ→180°)
    # without division-by-zero.
    delta = t_col_s - t_s
    visual_angle[mask] = np.degrees(2.0 * np.arctan2(lv_s, delta))

    # Post-closest-approach hold at the clipped continuous limit (179°).
    # A receding object must never subtend 0 deg (v4 A3 / B F5); holding
    # at the pre-TTC limit keeps the input manifold smooth (v5 NEW-2).
    visual_angle[time_ms >= ttc_ms] = 179.0

    # Clamp to max_deg
    visual_angle = np.clip(visual_angle, 0.0, 179.0)

    return visual_angle


def _normalize_rel_ms(
    abs_ms: pd.Series,
    trial_ids: pd.Series,
    time_origin_abs_ms: Optional[Dict[int, float]],
) -> pd.Series:
    """Normalize absolute-ms timestamps to per-trial relative milliseconds.

    Single shared seam for raw-schema event time normalization (used by both
    :func:`parse_trial_events` and the step-4 events rewrite in
    :func:`adapt_session_pair`) so ``stimulus_onset`` and ``phase_transition``
    times cannot desynchronize.

    When *time_origin_abs_ms* is provided, every ``trial_id`` present in
    *trial_ids* must exist in the map — **fail closed** rather than silently
    mixing a kinematics-clock origin with an event-minimum origin in one
    session (v4 review A2).  Otherwise the per-trial chronological minimum
    of *abs_ms* is the origin (same definition as the kinematics fallback).

    Args:
        abs_ms: Absolute timestamps in milliseconds.
        trial_ids: Trial identifier per row (aligned with *abs_ms*).
        time_origin_abs_ms: Optional mapping trial_id -> absolute-ms origin.

    Returns:
        Series of per-trial relative milliseconds (float64).

    Raises:
        ValueError: If *time_origin_abs_ms* is given but any event trial_id
            is absent from the map.
    """
    if time_origin_abs_ms is not None:
        mapped = trial_ids.map(time_origin_abs_ms)
        if mapped.isna().any():
            orphans = sorted({int(t) for t in trial_ids[mapped.isna()].unique()})
            raise ValueError(
                f"Events reference trial_id(s) {orphans} with no kinematics "
                f"rows; refusing to mix per-trial time origins in one session."
            )
        return abs_ms.astype(np.float64) - mapped.astype(np.float64)
    origin = abs_ms.astype(np.float64).groupby(trial_ids).transform('min')
    return abs_ms.astype(np.float64) - origin


def _first_per_trial(
    df: pd.DataFrame,
    tid_col: str,
    time_col: str,
) -> pd.DataFrame:
    """Return the chronologically first row per trial (stable sort).

    Out-of-order event rows must not let the last file-order writer win
    when selecting the Looming / TTC / wind onset (v4 review B F4).
    """
    if df.empty:
        return df
    ordered = df.sort_values(time_col, kind='mergesort')
    return ordered.drop_duplicates(tid_col, keep='first')


def parse_trial_events(
    evt_path: Union[str, Path],
    time_origin_abs_ms: Optional[Dict[int, float]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Parse events CSV to extract trial types and stimulus parameters.

    Schema-aware timestamp normalization (the shared seam with canonical
    event processing in :func:`adapt_session_pair`):

    * **Raw schema** (``timestamp`` column, seconds): values are converted
      to per-trial relative milliseconds.  When *time_origin_abs_ms* is
      provided (per-trial absolute-ms origin, typically the chronological
      first kinematics sample), that origin is subtracted so event times
      share the kinematics trial zero.  Trials absent from the map fail
      closed (no silent clock mixing).  Otherwise the fallback origin is
      ``min(ts*1000)`` per trial.
    * **Canonical schema** (``time_ms`` column, already milliseconds):
      values are returned as-is (no double scaling).

    Phase-selection semantics: when multiple phase_transition rows target
    the same phase, the **chronologically first** (by normalized time) wins.
    An explicit ``looming_onset`` event row is honored as a fallback source
    when no ``phase_transition -> Looming`` exists.

    Args:
        evt_path: Path to events CSV.
        time_origin_abs_ms: Optional mapping trial_id -> absolute time in ms
            of the shared per-trial origin (kinematics chronological first
            sample).  When given, raw-schema times are normalized against
            this origin instead of the event-file minimum.

    Returns:
        Dict mapping trial_id -> {
            'type': str,  # 'baseline_wind', 'baseline_visual', 'looming_wind'
            'lv_ratio_ms': float or None,
            'target_ttc_ms': float or None,
            'wind_dir': str or None,
            'stimulus_onset_ms': float or None,  # per-trial relative ms
            'looming_onset_ms': float or None,   # per-trial relative ms
            'ttc_ms': float or None,             # per-trial relative ms
        }

    Raises:
        ValueError: If the events schema lacks a recognizable event-name
            column, trial-id column, or timestamp column; or if
            *time_origin_abs_ms* is given and any event trial_id is missing
            from the map.
    """
    df_e = pd.read_csv(evt_path)
    trial_info: Dict[int, Dict[str, Any]] = {}

    # Normalize column names (fail closed on unrecognized schemas).
    evt_col = 'event_type' if 'event_type' in df_e.columns else (
        'event_name' if 'event_name' in df_e.columns else None)
    tid_col = 'trial_id' if 'trial_id' in df_e.columns else (
        'global_trial_id' if 'global_trial_id' in df_e.columns else None)
    val_col = 'event_value' if 'event_value' in df_e.columns else (
        'details' if 'details' in df_e.columns else None)
    ts_col = 'time_ms' if 'time_ms' in df_e.columns else (
        'timestamp' if 'timestamp' in df_e.columns else None)
    if evt_col is None or tid_col is None or ts_col is None:
        raise ValueError(
            f"Unrecognized events schema: columns={list(df_e.columns)}. "
            f"Need event_type|event_name, trial_id|global_trial_id, and "
            f"time_ms|timestamp."
        )
    if val_col is None:
        df_e['_empty_val'] = ''
        val_col = '_empty_val'

    # Shared timestamp normalization seam: raw seconds → per-trial relative ms.
    # Canonical already-ms values pass through unchanged.
    if ts_col == 'timestamp':
        df_e['_abs_ms'] = df_e[ts_col].astype(np.float64) * 1000.0
        df_e['_norm_ms'] = _normalize_rel_ms(
            df_e['_abs_ms'], df_e[tid_col], time_origin_abs_ms,
        )
        time_val_col = '_norm_ms'
    else:
        time_val_col = ts_col

    # ── trial_start events: extract trial metadata ──
    ts_mask = df_e[evt_col] == 'trial_start'
    ts_df = df_e.loc[ts_mask]
    if not ts_df.empty:
        for _tid, _grp in ts_df.groupby(tid_col):
            if len(_grp) > 1:
                warnings.warn(
                    f"Trial {_tid} has {len(_grp)} trial_start rows; "
                    f"using chronologically first, discarding {len(_grp) - 1} duplicate(s).",
                    stacklevel=2,
                )
        ts_ordered = _first_per_trial(ts_df, tid_col, time_val_col)
        for _, row in ts_ordered.iterrows():
            trial_id = int(row[tid_col])
            try:
                details = json.loads(str(row[val_col]))
            except Exception:
                details = {}

            raw_ttc = details.get('target_ttc_ms')
            target_ttc_ms: Optional[float] = None
            if raw_ttc is not None and not (isinstance(raw_ttc, float) and np.isnan(raw_ttc)):
                raw_s = str(raw_ttc).strip()
                if raw_s not in ('', 'null', 'None', 'nan', 'NaN'):
                    try:
                        val = float(raw_ttc)
                        if math.isnan(val):
                            target_ttc_ms = None
                        elif math.isinf(val):
                            raise ValueError(
                                f"Trial {trial_id} declared non-finite target_ttc_ms: {raw_ttc!r}"
                            )
                        else:
                            target_ttc_ms = val
                    except (ValueError, TypeError) as exc:
                        if "non-finite" in str(exc):
                            raise
                        raise ValueError(
                            f"Trial {trial_id} declared unparseable target_ttc_ms: {raw_ttc!r}"
                        ) from exc

            raw_lv = details.get('lv_ratio_ms')
            lv_ratio_ms: Optional[float] = None
            if raw_lv is not None and not (isinstance(raw_lv, float) and np.isnan(raw_lv)):
                raw_s = str(raw_lv).strip()
                if raw_s not in ('', 'null', 'None', 'nan', 'NaN'):
                    try:
                        val = float(raw_lv)
                        if math.isnan(val):
                            lv_ratio_ms = None
                        elif math.isinf(val):
                            raise ValueError(
                                f"Trial {trial_id} declared non-finite lv_ratio_ms: {raw_lv!r}"
                            )
                        else:
                            lv_ratio_ms = val
                    except (ValueError, TypeError) as exc:
                        if "non-finite" in str(exc):
                            raise
                        raise ValueError(
                            f"Trial {trial_id} declared unparseable lv_ratio_ms: {raw_lv!r}"
                        ) from exc

            trial_info[trial_id] = {
                'type': details.get('type', 'unknown'),
                'lv_ratio_ms': lv_ratio_ms,
                'target_ttc_ms': target_ttc_ms,
                'wind_dir': details.get('wind_dir', 'none'),
                'stimulus_onset_ms': None,
                'looming_onset_ms': None,
                'ttc_ms': None,
            }

    # ── phase_transition events: vectorized processing ──
    pt_mask = df_e[evt_col] == 'phase_transition'
    if trial_info and pt_mask.any():
        pt_df = df_e.loc[pt_mask, [tid_col, val_col, time_val_col]].copy()
        # Only process transitions for known trials
        pt_df = pt_df[pt_df[tid_col].isin(trial_info)]

        # Parse details JSON in batch
        parsed = pt_df[val_col].apply(lambda s: json.loads(str(s)) if pd.notna(s) else {})
        pt_df['from_phase'] = parsed.apply(lambda d: d.get('from_phase', ''))
        pt_df['to_phase'] = parsed.apply(lambda d: d.get('to_phase', ''))

        # Looming onset: to_phase == 'Looming' (chronologically first).
        # Warn on duplicate Looming transitions per trial (v5 NEW-3):
        # silently dropping extra rows would mutate event chronology
        # without a counted diagnostic.
        looming = pt_df[pt_df['to_phase'] == 'Looming']
        for _tid, _grp in looming.groupby(tid_col):
            if len(_grp) > 1:
                warnings.warn(
                    f"Trial {_tid} has {len(_grp)} Looming phase_transition "
                    f"rows; using chronologically first, discarding "
                    f"{len(_grp) - 1} duplicate(s).",
                    stacklevel=2,
                )
        for _, r in _first_per_trial(looming, tid_col, time_val_col).iterrows():
            trial_info[int(r[tid_col])]['looming_onset_ms'] = float(r[time_val_col])

        # TTC: to_phase == 'Collision_TTC0' (chronologically first)
        ttc = pt_df[pt_df['to_phase'] == 'Collision_TTC0']
        for _, r in _first_per_trial(ttc, tid_col, time_val_col).iterrows():
            trial_info[int(r[tid_col])]['ttc_ms'] = float(r[time_val_col])

        # Wind onset: from_phase == 'Baseline' -> to_phase == 'PostStimulus'
        wind = pt_df[(pt_df['from_phase'] == 'Baseline') & (pt_df['to_phase'] == 'PostStimulus')]
        for _, r in _first_per_trial(wind, tid_col, time_val_col).iterrows():
            trial_info[int(r[tid_col])]['stimulus_onset_ms'] = float(r[time_val_col])

    # ── Fallback: explicit 'looming_onset' event rows (alternative schema) ──
    lo_mask = df_e[evt_col] == 'looming_onset'
    if trial_info and lo_mask.any():
        lo_df = df_e.loc[lo_mask, [tid_col, time_val_col]].copy()
        lo_df = lo_df[lo_df[tid_col].isin(trial_info)]
        for _, r in _first_per_trial(lo_df, tid_col, time_val_col).iterrows():
            tid = int(r[tid_col])
            if trial_info[tid]['looming_onset_ms'] is None:
                trial_info[tid]['looming_onset_ms'] = float(r[time_val_col])

    return trial_info


def extract_file_stem(path: Union[str, Path]) -> Tuple[str, str]:
    """Extract (session_stem, role) from a CSV path.

    Enforces strict NSMoR session filename endings:
      '<stem>_kinematics.csv' -> (stem, 'kinematics')
      '<stem>_events.csv'     -> (stem, 'events')
    Or legacy bare filenames in dedicated session subdirectories:
      'kinematics.csv'        -> ('', 'kinematics')
      'events.csv'            -> ('', 'events')

    Args:
        path: Path to CSV file.

    Returns:
        Tuple of (session_stem, role), where role is 'kinematics' or 'events'.

    Raises:
        ValueError: If file is not a valid CSV or fails the naming contract.
    """
    path = Path(path)
    name = path.name
    if not name.lower().endswith(".csv"):
        raise ValueError(f"File is not a CSV: '{path}'")

    stem = name[:-4]  # strip '.csv'
    stem_lower = stem.lower()

    if stem_lower.endswith("_kinematics"):
        return stem[:-len("_kinematics")], "kinematics"
    if stem_lower == "kinematics":
        return "", "kinematics"

    if stem_lower.endswith("_events"):
        return stem[:-len("_events")], "events"
    if stem_lower == "events":
        return "", "events"

    raise ValueError(
        f"File does not follow NSMoR session naming contract "
        f"('<stem>_kinematics.csv' or '<stem>_events.csv'): '{path.name}'"
    )


def derive_session_id(
    kin_path: Union[str, Path],
    evt_path: Optional[Union[str, Path]] = None,
    root_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Derive canonical session identifier from file stems or parent folder.

    Requires kinematics and events filenames to share an identical stem prefix.
    Legacy bare filenames ('kinematics.csv' / 'events.csv') are accepted ONLY
    when placed in a dedicated session subfolder (not flat in root_dir).

    Args:
        kin_path: Path to kinematics CSV.
        evt_path: Optional path to events CSV.
        root_dir: Optional root directory for flat vs nested validation.

    Returns:
        Derived session ID string.

    Raises:
        ValueError: If stems conflict, filenames violate contract, or session
            ID cannot be unambiguously determined.
    """
    k_path = Path(kin_path)
    e_path = Path(evt_path) if evt_path is not None else None
    r_dir = Path(root_dir) if root_dir is not None else None

    k_stem, k_role = extract_file_stem(k_path)
    if k_role != "kinematics":
        raise ValueError(f"Expected kinematics CSV, got: '{k_path.name}'")

    if e_path is not None:
        e_stem, e_role = extract_file_stem(e_path)
        if e_role != "events":
            raise ValueError(f"Expected events CSV, got: '{e_path.name}'")
        if k_stem != e_stem:
            raise ValueError(
                f"Session stem mismatch between kinematics ('{k_stem}') "
                f"and events ('{e_stem}'): '{k_path.name}' vs '{e_path.name}'"
            )
        candidate = k_stem
    else:
        candidate = k_stem

    if candidate:
        return candidate

    # Legacy bare filename fallback ('kinematics.csv' / 'events.csv')
    parent = k_path.parent
    if r_dir is not None and parent.resolve() == r_dir.resolve():
        raise ValueError(
            f"Cannot derive session ID from bare filename '{k_path.name}' "
            f"in flat directory '{r_dir}'. A session prefix or session folder is required."
        )
    if e_path is not None and parent.resolve() != e_path.parent.resolve():
        raise ValueError(
            f"Parent directory mismatch for bare filenames: "
            f"'{k_path}' vs '{e_path}'"
        )
    if not parent.name:
        raise ValueError(f"Cannot determine session ID from path: '{k_path}'")
    return parent.name


def validate_output_paths(
    raw_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """Validate that output_dir is disjoint and distinct from raw_dir.

    Args:
        raw_dir: Source raw data directory.
        output_dir: Destination staging directory.

    Returns:
        Tuple of (resolved_raw_dir, resolved_output_dir).

    Raises:
        ValueError: If output_dir equals or nests within raw_dir (or vice versa).
    """
    raw_resolved = Path(raw_dir).resolve()
    out_resolved = Path(output_dir).resolve()

    if raw_resolved == out_resolved:
        raise ValueError(
            f"output_dir ({output_dir} -> {out_resolved}) cannot be identical "
            f"to raw_dir ({raw_dir} -> {raw_resolved})."
        )
    if raw_resolved in out_resolved.parents:
        raise ValueError(
            f"output_dir ({output_dir} -> {out_resolved}) cannot be inside "
            f"raw_dir ({raw_dir} -> {raw_resolved})."
        )
    if out_resolved in raw_resolved.parents:
        raise ValueError(
            f"raw_dir ({raw_dir} -> {raw_resolved}) cannot be inside "
            f"output_dir ({output_dir} -> {out_resolved})."
        )

    if raw_resolved.exists() and out_resolved.exists():
        if _samefile_or_fail(raw_resolved, out_resolved):
            raise ValueError(
                f"output_dir ({output_dir} -> {out_resolved}) resolves to the same "
                f"file/directory as raw_dir ({raw_dir} -> {raw_resolved})."
            )

    # Check parents for case-insensitive / aliased filesystems (even if output_dir does not exist yet)
    if raw_resolved.exists():
        for p in out_resolved.parents:
            if p.exists() and _samefile_or_fail(raw_resolved, p):
                raise ValueError(
                    f"output_dir ({output_dir} -> {out_resolved}) cannot be inside "
                    f"raw_dir ({raw_dir} -> {raw_resolved})."
                )

    if out_resolved.exists():
        for p in raw_resolved.parents:
            if p.exists() and _samefile_or_fail(out_resolved, p):
                raise ValueError(
                    f"raw_dir ({raw_dir} -> {raw_resolved}) cannot be inside "
                    f"output_dir ({output_dir} -> {out_resolved})."
                )

    return raw_resolved, out_resolved


def find_session_pairs(
    raw_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> List[Tuple[str, Path, Path]]:
    """Scan raw directory and return deterministically matched session pairs.

    Handles both flat directories (stems matching) and nested session subdirectories.
    Guards against scanning inside output_dir if specified.

    Args:
        raw_dir: Root directory containing raw session files.
        output_dir: Optional output directory to exclude from scanning.

    Returns:
        List of ``(session_id, kinematics_path, events_path)`` sorted by session_id.

    Raises:
        FileNotFoundError: If raw_dir does not exist or no valid pairs found.
        ValueError: If orphan files or duplicate session IDs are encountered.
    """
    base_path = Path(raw_dir).resolve()
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist or is not a directory: {raw_dir}")

    out_resolved = Path(output_dir).resolve() if output_dir is not None else None

    # Find all CSV files, excluding output_dir if within or overlapping
    all_csvs: List[Path] = []
    for p in base_path.rglob("*.csv"):
        if out_resolved is not None:
            try:
                p_res = p.resolve()
                if out_resolved in p_res.parents or p_res == out_resolved:
                    continue
            except Exception:
                pass
        all_csvs.append(p)

    kin_map: Dict[str, Path] = {}
    evt_map: Dict[str, Path] = {}

    for p in all_csvs:
        try:
            stem, role = extract_file_stem(p)
        except ValueError:
            # Skip non-conforming CSV files (e.g. metadata/summary)
            continue

        if role == "kinematics":
            sid = derive_session_id(p, root_dir=base_path)
            if sid in kin_map:
                raise ValueError(
                    f"Duplicate kinematics file for session '{sid}': "
                    f"'{kin_map[sid]}' and '{p}'"
                )
            kin_map[sid] = p
        elif role == "events":
            sid = stem if stem else (p.parent.name if p.parent.resolve() != base_path else "")
            if not sid:
                raise ValueError(
                    f"Cannot derive session ID from bare events filename '{p.name}' in '{base_path}'"
                )
            if sid in evt_map:
                raise ValueError(
                    f"Duplicate events file for session '{sid}': "
                    f"'{evt_map[sid]}' and '{p}'"
                )
            evt_map[sid] = p

    kin_sessions = set(kin_map.keys())
    evt_sessions = set(evt_map.keys())

    unpaired_kin = sorted(kin_sessions - evt_sessions)
    unpaired_evt = sorted(evt_sessions - kin_sessions)

    if unpaired_kin or unpaired_evt:
        error_parts = []
        if unpaired_kin:
            error_parts.append(
                f"Unpaired kinematics ({len(unpaired_kin)}): {unpaired_kin[:5]}"
                + ("..." if len(unpaired_kin) > 5 else "")
            )
        if unpaired_evt:
            error_parts.append(
                f"Unpaired events ({len(unpaired_evt)}): {unpaired_evt[:5]}"
                + ("..." if len(unpaired_evt) > 5 else "")
            )
        raise ValueError(f"Unpaired session files found in '{raw_dir}': {'; '.join(error_parts)}")

    if not kin_sessions:
        raise FileNotFoundError(f"No valid kinematics/events pairs found in '{raw_dir}'")

    matched_pairs: List[Tuple[str, Path, Path]] = [
        (sid, kin_map[sid], evt_map[sid])
        for sid in sorted(kin_sessions)
    ]
    return matched_pairs


def _experimental_trial_clock(
    grp: pd.DataFrame, tolerance_ms: float, diagnostic: Dict[str, Any],
) -> np.ndarray:
    """Estimate host-arrival association, not acquisition or stimulus latency.

    Centered ordinary least squares uses every paired row, with no outlier
    removal or repair. The caller's maximum residual tolerance is an operational
    criterion, not a biological precision guarantee. Leave-one-out residuals
    prevent a high-leverage endpoint from validating its own fitted location.
    """
    n = len(grp)
    diagnostic.update(n_samples=n, status="rejected")
    if n == 1:
        host = pd.to_numeric(grp["sys_time"], errors="coerce").to_numpy(
            dtype=np.float64,
        ) * 1000.0
        if not np.isfinite(host).all():
            raise ValueError("Single-row host time must be finite.")
        diagnostic.update(
            status="host_only_convention", method="single_row_host_only",
            derivative_interpretation="undefined; stored zero is a convention",
            scientific_acceptance="unresolved",
        )
        return host
    if n < 4:
        raise ValueError("At least four paired rows are required for an affine fit.")
    tokens = grp["ard_time"].astype(str)
    lexical = tokens.str.fullmatch(r"[0-9]{1,10}")
    ticks = pd.to_numeric(tokens, errors="coerce").to_numpy(dtype=np.float64)
    valid = lexical.to_numpy() & np.isfinite(ticks) & (ticks <= 2**32 - 1)
    if not valid.all():
        diagnostic["invalid_source_rows"] = grp.index[~valid].tolist()
        raise ValueError("Invalid uint32 tick token; no trimming or reconstruction.")
    host = pd.to_numeric(grp["sys_time"], errors="coerce").to_numpy(
        dtype=np.float64,
    ) * 1000.0
    if not np.isfinite(host).all() or np.any(np.diff(host) < 0):
        raise ValueError("Host times must be finite and nondecreasing in file order.")
    if np.any(np.diff(ticks) <= 0):
        raise ValueError("Hardware ticks must increase; resets/wraps are unresolved.")
    diagnostic["duplicate_host_intervals"] = int(np.sum(np.diff(host) == 0))
    x = ticks - ticks.mean()
    y_mean = float(host.mean())
    ss = float(x @ x)
    slope = float(x @ (host - y_mean) / ss)
    fitted = y_mean + slope * x
    residual = host - fitted
    leverage = 1.0 / n + x * x / ss
    if not np.isfinite(slope) or slope <= 0 or np.any(leverage >= 1):
        raise ValueError("Nonpositive/nonfinite slope or unidentifiable mapping.")
    loo = residual / (1.0 - leverage)
    diagnostic.update(
        slope=slope, intercept_ms=float(y_mean - slope * ticks.mean()),
        reference_tick=float(ticks.mean()), reference_host_ms=y_mean,
        rmse_ms=float(np.sqrt(np.mean(residual**2))),
        max_abs_residual_ms=float(np.max(np.abs(residual))),
        max_abs_leave_one_out_residual_ms=float(np.max(np.abs(loo))),
        worst_source_row=int(grp.index[int(np.argmax(np.abs(loo)))]),
    )
    if not np.isfinite(loo).all() or np.max(np.abs(loo)) > tolerance_ms:
        raise ValueError(
            "Clock association exceeds explicit operational residual tolerance; "
            "retain raw trial for timing review, not automatic reconstruction."
        )
    if not np.isfinite(fitted).all() or np.any(np.diff(fitted) <= 0):
        raise ValueError("Mapped times are not finite and strictly increasing.")
    diagnostic["status"] = "accepted_by_operational_tolerance"
    assert fitted.shape == (n,)
    return fitted


def _suspect_prefixes(
    path: Union[str, Path], audit: Dict[str, Any], frame: pd.DataFrame,
) -> Dict[str, int]:
    """Validate externally specified suspect rows, not a corruption diagnosis."""
    content = Path(path).read_bytes()
    manifest = json.loads(content)
    assumption = "manually specified suspect prefix/model assumption"
    if (not isinstance(manifest, dict)
            or manifest.get("schema") != "experimental_suspect_prefix_v1"
            or manifest.get("assumption") != assumption
            or not isinstance(manifest.get("sessions"), list)):
        raise ValueError("Invalid suspect-prefix manifest schema/assumption.")
    audit["prefix_manifest_sha256"] = hashlib.sha256(content).hexdigest()
    audit["prefix_manifest_path"] = str(Path(path).resolve())
    sessions = manifest["sessions"]
    if any(not isinstance(s, dict) for s in sessions):
        raise ValueError("Invalid suspect-prefix session entry.")
    names = [s.get("session_id") for s in sessions]
    if any(not isinstance(s, str) for s in names) or len(set(names)) != len(names):
        raise ValueError("Missing or duplicate suspect-prefix session identity.")
    matches = [s for s in sessions if s["session_id"] == audit["session_id"]]
    if not matches:
        return {}  # Unlisted sessions still undergo all-row validation.
    session = matches[0]
    for key in ("kinematics_sha256", "events_sha256"):
        if session.get(key) != audit[key]:
            raise ValueError(f"Suspect-prefix manifest {key} mismatch.")
    trials = session.get("trials")
    if not isinstance(trials, list):
        raise ValueError("Invalid suspect-prefix trials.")
    groups = {str(t): g for t, g in frame.groupby("global_trial_id", sort=False)}
    prefixes: Dict[str, int] = {}
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("Invalid suspect-prefix trial entry.")
        tid, rows = trial.get("trial_id"), trial.get("rows")
        if not isinstance(tid, str) or tid not in groups or tid in prefixes:
            raise ValueError("Unknown or duplicate suspect-prefix trial identity.")
        if not isinstance(rows, list) or not rows or len(rows) >= len(groups[tid]):
            raise ValueError("Suspect prefix must leave an independently fitted suffix.")
        expected = [
            {"source_row_index": int(i), "raw_sys_time": r["sys_time"],
             "raw_ard_time": r["ard_time"]}
            for i, r in groups[tid].iloc[:len(rows)].iterrows()
        ]
        if rows != expected or any(
            type(r.get("source_row_index")) is not int for r in rows
        ):
            raise ValueError("Suspect-prefix row index or lexical token mismatch.")
        prefixes[tid] = len(rows)
    audit["suspect_prefix_assumption"] = assumption
    audit["suspect_prefix_entries"] = trials
    audit["method"] = (
        "centered_OLS_suffix_leave_one_out_gate_with_declared_prefix_cadence"
    )
    return prefixes


def _prefix_clock_candidate(
    grp: pd.DataFrame, prefix: int, tolerance_ms: float,
    diagnostic: Dict[str, Any], events: pd.DataFrame,
) -> np.ndarray:
    """Extrapolate a declared prefix; retain host-boundary sensitivity separately."""
    diagnostic.update(
        status="rejected", n_samples=len(grp), prefix_rows=prefix,
        assumption="manually specified suspect prefix/model assumption",
        model="constant cadence: median mapped suffix interval",
        scientific_acceptance="unresolved",
        derivative_interpretation=(
            "Estimated prefix and boundary derivatives are model-dependent; "
            "finite values do not establish first-interval reliability."
        ),
        suffix_fit={},
    )
    suffix = grp.iloc[prefix:]
    if len(suffix) < 4:
        raise ValueError("Suspect prefix requires at least four suffix pairs.")
    fitted = _experimental_trial_clock(suffix, tolerance_ms, diagnostic["suffix_fit"])
    cadence = float(np.median(np.diff(fitted)))
    mapped = np.concatenate((fitted[0] - cadence * np.arange(prefix, 0, -1), fitted))
    if not np.isfinite(mapped).all() or np.any(np.diff(mapped) <= 0):
        raise ValueError("Prefix estimates must be finite and strictly increasing.")
    host = pd.to_numeric(grp["sys_time"], errors="coerce").to_numpy(
        dtype=np.float64,
    ) * 1000.0
    if not np.isfinite(host).all() or np.any(np.diff(host) < 0):
        raise ValueError("Prefix host times must be finite and nondecreasing.")
    alternative = np.concatenate((host[:prefix], fitted))
    host_valid = bool(np.all(np.diff(alternative) > 0))
    # Rotation does not change the dx/dy norm; use the adapter's cm conversion.
    distance = np.hypot(grp["dx"].to_numpy(dtype=float),
                        grp["dy"].to_numpy(dtype=float)) / 10.0
    distance[0] = 0.0
    if not np.isfinite(distance).all():
        raise ValueError("Cannot describe derivatives for nonfinite displacements.")
    count = min(prefix + 2, len(grp))  # Boundary derivative plus acceleration echo.

    def describe(axis: np.ndarray, valid: bool) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": "strictly_increasing" if valid else "unresolved_nonpositive_interval",
            "absolute_ms": axis[:count].tolist(),
            "source_row_indices": grp.index[:count].tolist(),
        }
        if valid:
            dt = np.r_[1.0, np.diff(axis) / 1000.0]
            velocity = distance / dt
            acceleration = np.r_[0.0, np.diff(velocity)] / dt
            if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
                raise ValueError("Nonfinite candidate derivatives.")
            result.update(
                velocity_cm_s=velocity[:count].tolist(),
                acceleration_cm_s2=acceleration[:count].tolist(),
            )
        return result

    sensitivity: Dict[str, Any] = {
        "scope": "descriptive only; no scientific acceptance limits established",
        "cadence_ms": cadence,
        "cadence": describe(mapped, True),
        "host_boundary": describe(alternative, host_valid),
        "events": [],
    }
    for source_index, event in events.iterrows():
        time = float(event["timestamp"]) * 1000.0
        if not math.isfinite(time):
            raise ValueError("Cannot assess a nonfinite host event timestamp.")
        sensitivity["events"].append({
            "source_event_row_index": int(source_index),
            "cadence_relative_ms": time - float(mapped[0]),
            "host_boundary_relative_ms": time - float(alternative[0]),
            "cadence_in_recorded_window": bool(mapped[0] <= time <= mapped[-1]),
            "host_boundary_in_recorded_window": (
                bool(alternative[0] <= time <= alternative[-1]) if host_valid else None
            ),
        })
    diagnostic.update(sensitivity=sensitivity, status="accepted_by_operational_tolerance")
    assert mapped.shape == (len(grp),)
    return mapped


def _clock_source_hash(path: Path) -> str:
    """Bind diagnostics to exact input bytes without loading a whole CSV."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_clock_failure(path: Path, audit: Dict[str, Any]) -> None:
    """Persist rejected mapping diagnostics without publishing canonical CSVs."""
    _reject_symlink_components(path.parent, kind="clock diagnostic directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent, kind="clock diagnostic directory")
    # Exclusive creation also rejects existing files and symlink destinations.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def adapt_session_pair(
    kin_path: Union[str, Path],
    evt_path: Union[str, Path],
    *,
    session_id: Optional[str] = None,
    output_session_dir: Optional[Union[str, Path]] = None,
    in_place: bool = False,
    experimental_clock_residual_ms: Optional[float] = None,
    experimental_clock_prefix_manifest: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Adapt a single paired session (kinematics + events) to canonical NSMoR schema.

    Preserves exact physical calibration, true dt acceleration derivatives, phantom spike
    zeroing at trial boundaries, and visual angle reconstruction.

    Strictly protects source files from accidental overwrite when output_session_dir is used.

    Args:
        kin_path: Source kinematics CSV path (read-only unless in_place=True).
        evt_path: Source events CSV path (read-only unless in_place=True).
        session_id: Optional explicit session identifier string.
        output_session_dir: Optional target directory to write adapted CSVs.
        in_place: If True and output_session_dir is None, rewrite source CSVs in place.
        experimental_clock_residual_ms: Explicit opt-in to per-trial OLS clock
            association. Positive finite maximum leave-one-out residual in ms;
            an operational tolerance, NOT physical synchronization accuracy.
            Requires non-destructive staging. No implicit repairs or frame deletion.
        experimental_clock_prefix_manifest: Optional hash/token/index-bound JSON
            declaring suspect leading rows. Uses constant median suffix cadence
            with separate host-boundary sensitivity; scientific acceptance unresolved.

    Returns:
        Tuple of (df_kinematics_adapted, df_events_adapted).

    Raises:
        FileExistsError: If target files already exist in output_session_dir.
        ValueError: If inputs are invalid, conflicting, or target aliases source.
        FileNotFoundError: If source kinematics or events CSV does not exist.
    """
    if (experimental_clock_prefix_manifest is not None
            and experimental_clock_residual_ms is None):
        raise ValueError("Suspect-prefix manifest requires explicit clock tolerance.")
    if experimental_clock_residual_ms is not None:
        if output_session_dir is None or in_place:
            raise ValueError("Experimental clock mapping requires staged output.")
        if (not math.isfinite(experimental_clock_residual_ms)
                or experimental_clock_residual_ms <= 0):
            raise ValueError("Experimental clock residual tolerance must be positive.")
    if output_session_dir is not None and in_place:
        raise ValueError("Cannot specify both output_session_dir and in_place=True.")

    if in_place and (is_protected_raw_path(kin_path) or is_protected_raw_path(evt_path)):
        raise PermissionError(
            "In-place adaptation is strictly prohibited on authoritative raw source files: "
            f"'{kin_path}', '{evt_path}'. Specify output_session_dir to stage non-destructively."
        )

    k_path = Path(kin_path)
    e_path = Path(evt_path)

    if not k_path.is_file():
        raise FileNotFoundError(f"Kinematics source file does not exist: '{k_path}'")
    if not e_path.is_file():
        raise FileNotFoundError(f"Events source file does not exist: '{e_path}'")

    derived_sid = derive_session_id(k_path, e_path)
    if session_id is None:
        session_id = derived_sid
    else:
        session_id = _validate_session_id(session_id)
        if session_id != derived_sid:
            raise ValueError(
                f"session_id {session_id!r} does not match the source stem "
                f"contract (expected {derived_sid!r} derived from "
                f"'{k_path.name}' / '{e_path.name}')."
            )

    # ── Source/Target path safety and collision pre-checks ──
    target_kin: Optional[Path] = None
    target_evt: Optional[Path] = None
    if output_session_dir is not None:
        out_dir = Path(output_session_dir)
        _reject_symlink_components(out_dir, kind='output session directory')
        k_res = k_path.resolve()
        e_res = e_path.resolve()
        out_res = out_dir.resolve()

        if out_res == k_res.parent or out_res == e_res.parent:
            raise ValueError(
                f"output_session_dir '{out_res}' cannot be identical to source directory '{k_res.parent}'."
            )
        if k_res.parent in out_res.parents or e_res.parent in out_res.parents:
            raise ValueError(
                f"output_session_dir '{out_res}' cannot be nested inside source directory '{k_res.parent}'."
            )
        if out_res in k_res.parents or out_res in e_res.parents:
            raise ValueError(
                f"Source directory '{k_res.parent}' cannot be nested inside output_session_dir '{out_res}'."
            )

        # Samefile check for out_res vs source parent (handles DrvFs casing/aliases)
        if out_res.exists() and k_res.parent.exists() and _samefile_or_fail(out_res, k_res.parent):
            raise ValueError(
                f"output_session_dir '{out_res}' cannot be identical to source directory '{k_res.parent}'."
            )

        if k_res.parent.exists():
            for p in out_res.parents:
                if p.exists() and _samefile_or_fail(p, k_res.parent):
                    raise ValueError(
                        f"output_session_dir '{out_res}' cannot be nested inside source directory '{k_res.parent}'."
                    )

        if out_res.exists():
            for p in k_res.parent.parents:
                if p.exists() and _samefile_or_fail(p, out_res):
                    raise ValueError(
                        f"Source directory '{k_res.parent}' cannot be nested inside output_session_dir '{out_res}'."
                    )

        target_kin = out_dir / f"{session_id}_kinematics.csv"
        target_evt = out_dir / f"{session_id}_events.csv"

        # Final containment: targets must sit directly inside out_res.
        for _tgt in (target_kin, target_evt):
            if _tgt.parent != out_dir:
                raise ValueError(
                    f"Target path {_tgt!r} escapes output session directory {out_dir!r}."
                )

        if target_kin.resolve() == k_res or target_evt.resolve() == e_res:
            raise ValueError("Target path resolves to source path! Refusing to overwrite source.")

        if target_kin.exists() and _samefile_or_fail(target_kin, k_res):
            raise ValueError("Target path resolves to source path! Refusing to overwrite source.")
        if target_evt.exists() and _samefile_or_fail(target_evt, e_res):
            raise ValueError("Target path resolves to source path! Refusing to overwrite source.")

        if target_kin.exists() or target_kin.is_symlink():
            raise FileExistsError(
                f"Target kinematics file already exists: '{target_kin}'. "
                "Refusing to overwrite."
            )
        if target_evt.exists() or target_evt.is_symlink():
            raise FileExistsError(
                f"Target events file already exists: '{target_evt}'. "
                "Refusing to overwrite."
            )

    # ── 1. Process Kinematics ──
    clock_audit: Optional[Dict[str, Any]] = None
    clock_target: Optional[Path] = None
    if experimental_clock_residual_ms is not None:
        assert target_kin is not None
        clock_target = target_kin.parent / f"{session_id}_timebase_audit.json"
        if clock_target.exists() or clock_target.is_symlink():
            raise FileExistsError(f"Clock audit already exists: {clock_target}")
        clock_audit = {
            "schema": "experimental_clock_association_v1",
            "status": "rejected", "session_id": session_id,
            "method": "centered_OLS_all_pairs_leave_one_out_residual_gate",
            "operational_tolerance_ms": experimental_clock_residual_ms,
            "limitation": "Arrival-clock association; not physical latency bounds.",
            "kinematics_sha256": _clock_source_hash(k_path),
            "events_sha256": _clock_source_hash(e_path),
            "kinematics_path": str(k_path.resolve()),
            "events_path": str(e_path.resolve()), "trials": [],
        }
        # Preserve lexical tokens; do not normalize malformed device values.
        df_k = pd.read_csv(k_path, dtype={"ard_time": str, "sys_time": str},
                           keep_default_na=False)
    else:
        df_k = pd.read_csv(k_path)
    stim_starts: List[Dict[str, Any]] = []

    is_raw = "sys_time" in df_k.columns and "x_pos" not in df_k.columns
    if "sys_time" in df_k.columns and "x_pos" in df_k.columns:
        raise ValueError(
            "Ambiguous mixed kinematics schema: contains both raw 'sys_time' "
            "and canonical 'x_pos'. Refusing to adapt unvalidated mixed format."
        )

    if clock_audit is not None:
        assert clock_target is not None
        try:
            if not is_raw or "ard_time" not in df_k:
                raise ValueError("Experimental mapping requires raw paired clocks.")
            event_schema = pd.read_csv(e_path, nrows=0).columns
            if "timestamp" not in event_schema or "time_ms" in event_schema:
                raise ValueError("Experimental mapping requires host timestamp events.")
            ids = df_k["global_trial_id"]
            if ids.isna().any() or (ids.astype(str).str.strip() == "").any():
                raise ValueError("Missing trial identity in raw timing rows.")
            runs = ids[ids.ne(ids.shift())]
            if runs.duplicated().any():
                raise ValueError("Repeated noncontiguous trial IDs need timing review.")
            df_k["raw_sys_time"] = df_k["sys_time"]
            df_k["raw_ard_time"] = df_k["ard_time"]
            df_k["source_row_index"] = np.arange(len(df_k))
            df_k["time_source"] = "experimental_affine_estimate"
            prefixes = (
                _suspect_prefixes(experimental_clock_prefix_manifest, clock_audit, df_k)
                if experimental_clock_prefix_manifest is not None else {}
            )
            raw_events = pd.read_csv(e_path) if prefixes else None
            mapped = np.empty(len(df_k), dtype=np.float64)
            for tid, grp in df_k.groupby("global_trial_id", sort=False):
                diag: Dict[str, Any] = {"trial_id": str(tid)}
                clock_audit["trials"].append(diag)
                prefix = prefixes.get(str(tid), 0)
                if prefix:
                    assert raw_events is not None
                    event_id = ("global_trial_id" if "global_trial_id" in raw_events
                                else "trial_id")
                    trial_events = raw_events.loc[raw_events[event_id] == tid]
                    mapped[grp.index] = _prefix_clock_candidate(
                        grp, prefix, experimental_clock_residual_ms, diag, trial_events,
                    )
                    df_k.loc[grp.index[:prefix], "time_source"] = (
                        "experimental_prefix_cadence_estimate"
                    )
                else:
                    mapped[grp.index] = _experimental_trial_clock(
                        grp, experimental_clock_residual_ms, diag,
                    )
                    if len(grp) == 1:
                        df_k.loc[grp.index, "time_source"] = "single_row_host_only"
            df_k["abs_time"] = mapped
            clock_audit["scientific_acceptance"] = "unresolved"
            clock_audit["status"] = "accepted_by_operational_tolerance"
        except ValueError as exc:
            clock_audit["error"] = str(exc)
            _write_clock_failure(clock_target, clock_audit)
            raise ValueError(
                f"Experimental clock mapping rejected; diagnostic: {clock_target}: {exc}"
            ) from exc

    if is_raw:
        df_k["session_id"] = session_id
        df_k["trial_id"] = df_k["global_trial_id"]

        if clock_audit is None:
            df_k["abs_time"] = df_k["sys_time"] * 1000.0
        # Fail closed on unsorted per-trial rows: heading/position cumsum and
        # velocity/acceleration diffs require chronological order. File-order
        # first() would silently move the shared origin (v4 review A1 / B F1).
        for _tid, _grp in df_k.groupby("trial_id", sort=False):
            _t = _grp["abs_time"].to_numpy()
            if _t.size >= 2 and np.any(np.diff(_t) < 0.0):
                raise ValueError(
                    f"Kinematics rows for trial {_tid} are not chronologically "
                    f"ordered (abs_time decreases in file order). Refusing to "
                    f"compute derivatives or derive a trial origin from a "
                    f"non-monotonic time base."
                )
        # Origin := chronological minimum (same definition as the events
        # fallback in _normalize_rel_ms), never file-order first.
        df_k["time_ms"] = df_k.groupby("trial_id")["abs_time"].transform(
            lambda x: x - x.min()
        )

        # Per-trial heading and position integration:
        # Cumulative heading and Cartesian position are tracked per trial so trials
        # (even if interleaved or contiguous in a session) do not leak trajectory,
        # rotation, or displacement into each other.
        heading_out = np.zeros(len(df_k), dtype=np.float64)
        x_pos_out = np.zeros(len(df_k), dtype=np.float64)
        y_pos_out = np.zeros(len(df_k), dtype=np.float64)
        for _, grp in df_k.groupby("trial_id", sort=False):
            idx = grp.index
            h = np.cumsum(np.degrees(grp["dz"].to_numpy() / 30.0))
            rad = np.radians(h)
            dx = grp["dx"].to_numpy()
            dy = grp["dy"].to_numpy()
            dx_glob = dx * np.cos(rad) - dy * np.sin(rad)
            dy_glob = dx * np.sin(rad) + dy * np.cos(rad)
            heading_out[idx] = h
            x_pos_out[idx] = np.cumsum(dx_glob) / 10.0
            y_pos_out[idx] = np.cumsum(dy_glob) / 10.0

        df_k["heading"] = heading_out
        df_k["x_pos"] = x_pos_out
        df_k["y_pos"] = y_pos_out

        per_trial_group = df_k.groupby("trial_id", sort=False)
        dx_sq = per_trial_group["x_pos"].transform(
            lambda x: np.concatenate([[0.0], np.diff(x) ** 2])
        )
        dy_sq = per_trial_group["y_pos"].transform(
            lambda y: np.concatenate([[0.0], np.diff(y) ** 2])
        )
        step_dist_mm = np.sqrt(dx_sq.to_numpy() + dy_sq.to_numpy()) * 10.0
        dt_s = per_trial_group["abs_time"].transform(
            lambda x: x.diff().fillna(0) / 1000.0
        )
        if clock_audit is None:
            dt_s = dt_s.clip(lower=0.001)
        else:
            # Only the first-row convention needs a denominator; later intervals
            # use the estimated clock unchanged, including sub-millisecond gaps.
            dt_s = dt_s.mask(per_trial_group.cumcount() == 0, 1.0)
        df_k["velocity"] = (step_dist_mm / dt_s.to_numpy()) / 10.0

        acc_out = np.zeros(len(df_k), dtype=np.float64)
        for _, grp in df_k.groupby("trial_id", sort=False):
            idx = grp.index
            v = grp["velocity"].to_numpy()
            t = grp["abs_time"].to_numpy()  # ms
            dv = np.concatenate([[0.0], np.diff(v)])
            dt_ms_arr = np.concatenate([[0.0], np.diff(t)])
            pos_gaps = dt_ms_arr[dt_ms_arr > 0]
            floor_ms = float(np.min(pos_gaps)) if pos_gaps.size > 0 else 1.0
            dt_safe = np.where(dt_ms_arr > 0, dt_ms_arr, floor_ms)
            acc = dv / (dt_safe / 1000.0)
            acc[0] = 0.0
            acc_out[idx] = acc
        df_k["acceleration"] = acc_out

        df_k["visual_angle"] = 0.0
        df_k["l_v_ratio"] = 0.0
        if "stim_state" in df_k.columns:
            df_k["wind_state"] = (
                pd.to_numeric(df_k["stim_state"], errors="coerce")
                .fillna(0)
                .eq(1)
                .astype(int)
            )
        else:
            df_k["wind_state"] = 0
    else:
        df_k["session_id"] = session_id
        if "wind_state" in df_k.columns:
            df_k["wind_state"] = (
                pd.to_numeric(df_k["wind_state"], errors="coerce")
                .fillna(0)
                .eq(1)
                .astype(int)
            )

    # Per-trial kinematics absolute-time origin (chronological first sample).
    # Canonical event/stimulus times share this zero so a trial_start that
    # precedes the first kinematics sample cannot skew stimulus onsets onto
    # a different clock.  min() == first() on sorted rows; min() is the
    # shared definition with _normalize_rel_ms.
    kin_origin_abs_ms: Dict[int, float] = {}
    if "abs_time" in df_k.columns:
        kin_origin_abs_ms = {
            int(k): float(v)
            for k, v in df_k.groupby("trial_id")["abs_time"].min().items()
        }

    # ── 2. Parse trial events (kin-origin normalized for raw schema) ──
    trial_info = parse_trial_events(
        e_path,
        time_origin_abs_ms=kin_origin_abs_ms or None,
    )

    # ── 3. Reconstruct visual angle per trial ──
    tid_series = df_k["trial_id"]
    unique_tids = tid_series.unique()

    for tid in unique_tids:
        trial_id = int(tid)
        info = trial_info.get(trial_id, {})
        trial_type = info.get('type', 'unknown')
        lv_ratio_ms = info.get('lv_ratio_ms')

        has_visual = trial_type in ('baseline_visual', 'looming_wind')
        has_wind = trial_type in ('baseline_wind', 'looming_wind')
        known_type = trial_type in ('baseline_visual', 'baseline_wind', 'looming_wind')

        trial_mask = tid_series == tid
        time_vals = df_k.loc[trial_mask, "time_ms"].values

        if has_visual and lv_ratio_ms and lv_ratio_ms > 0:
            onset_kin = info.get('looming_onset_ms')
            if onset_kin is None:
                # Missing Looming: do NOT fabricate a visual stimulus whose
                # expansion starts at t=0, and do NOT publish a flat init_deg
                # baseline (downstream argmax of a constant array == index 0
                # is a false collision anchor).  Publish zeros so has_looming
                # stays False (v4 review A4 / B F2 / B F5).
                warnings.warn(
                    f"[{session_id}/trial {tid}] visual trial (type="
                    f"{trial_type!r}, lv_ratio_ms={lv_ratio_ms}) has no "
                    f"Looming phase_transition; refusing to fabricate a "
                    f"visual stimulus or stimulus_onset.",
                    stacklevel=2,
                )
                df_k.loc[trial_mask, "visual_angle"] = 0.0
                df_k.loc[trial_mask, "l_v_ratio"] = 0.0
            else:
                t_rel = time_vals - float(onset_kin)
                visual_angle = reconstruct_visual_angle(
                    t_rel, lv_ratio_ms, init_deg=2.0,
                )
                # Data-quality guard: reject any non-expanding (flat) curve
                # regardless of its baseline value.  A flat curve — whether
                # at init_deg (pre-onset only) or at the 179° post-TTC
                # plateau (all-post-TTC window) — yields argmax == 0 in the
                # frozen downstream anchor and is a false collision
                # (v5 NEW-1 / V5B-F1).  Peak-to-peak variation is the
                # correct expansion test; mere exceedance of init_deg is not.
                if float(np.ptp(visual_angle)) < 1e-9:
                    warnings.warn(
                        f"[{session_id}/trial {tid}] reconstructed visual_angle "
                        f"never expands within the recorded window "
                        f"(looming_onset_ms={float(onset_kin):.3f}, "
                        f"window=[{time_vals[0]:.3f},{time_vals[-1]:.3f}]); "
                        f"publishing zeros to avoid a false collision anchor.",
                        stacklevel=2,
                    )
                    df_k.loc[trial_mask, "visual_angle"] = 0.0
                    df_k.loc[trial_mask, "l_v_ratio"] = 0.0
                else:
                    df_k.loc[trial_mask, "visual_angle"] = visual_angle
                    df_k.loc[trial_mask, "l_v_ratio"] = lv_ratio_ms
        else:
            df_k.loc[trial_mask, "visual_angle"] = 0.0
            df_k.loc[trial_mask, "l_v_ratio"] = 0.0

        if known_type and not has_wind:
            df_k.loc[trial_mask, "wind_state"] = 0
        elif not known_type:
            has_wind = bool((df_k.loc[trial_mask, "wind_state"] == 1).any())

        if has_wind:
            wind_mask = trial_mask & (df_k["wind_state"] == 1)
            if wind_mask.any():
                # Chronology-safe first wind onset (never file-order first).
                # The raw branch fails closed on unsorted rows; canonical has
                # no such guard, so sort here so the earliest wind sample
                # wins regardless of row order (v5 V5B-F3).
                wind_times = df_k.loc[wind_mask, "time_ms"].sort_values(
                    kind='mergesort',
                )
                stim_time = wind_times.iloc[0]
                stim_starts.append({
                    "session_id": session_id,
                    "trial_id": tid,
                    "time_ms": stim_time,
                    "event_type": "stimulus_onset",
                    "event_value": '{"source": "kinematics_injected"}',
                })
        elif has_visual:
            parsed_onset = info.get('looming_onset_ms')
            if parsed_onset is not None:
                # Keep valid negative kin-relative onsets negative (v4 B F3).
                # A normal delayed looming (onset > 0) is NOT an anomaly and
                # must not warn (v4 A5).
                onset_val = float(parsed_onset)
                stim_starts.append({
                    "session_id": session_id,
                    "trial_id": tid,
                    "time_ms": onset_val,
                    "event_type": "stimulus_onset",
                    "event_value": '{"source": "kinematics_injected"}',
                })
            # Missing Looming: do NOT fabricate stimulus_onset at 0.0
            # (v4 review A4 / B F2).

    # ── 4. Process Events ──
    df_e = pd.read_csv(e_path)
    # Schema normalization: accept event_type|event_name × time_ms|timestamp
    # without silent NaN corruption or KeyError (v4 review B F6).
    if "event_type" not in df_e.columns:
        if "event_name" not in df_e.columns:
            raise ValueError(
                f"Events schema must provide 'event_type' or 'event_name': "
                f"columns={list(df_e.columns)}"
            )
        df_e["event_type"] = df_e["event_name"]
    if "event_value" not in df_e.columns:
        df_e["event_value"] = (
            df_e["details"].fillna("") if "details" in df_e.columns else ""
        )
    if "trial_id" not in df_e.columns:
        if "global_trial_id" not in df_e.columns:
            raise ValueError(
                f"Events schema must provide 'trial_id' or 'global_trial_id': "
                f"columns={list(df_e.columns)}"
            )
        df_e["trial_id"] = df_e["global_trial_id"]
    df_e["session_id"] = session_id

    if "time_ms" in df_e.columns:
        # Canonical already-ms: pass through unchanged (no double scaling).
        pass
    elif "timestamp" in df_e.columns:
        df_e["abs_time"] = df_e["timestamp"].astype(np.float64) * 1000.0
        df_e["time_ms"] = _normalize_rel_ms(
            df_e["abs_time"], df_e["trial_id"], kin_origin_abs_ms or None,
        )
    else:
        raise ValueError(
            f"Events schema must provide 'time_ms' or 'timestamp': "
            f"columns={list(df_e.columns)}"
        )

    if stim_starts:
        df_stim = pd.DataFrame(stim_starts)
        # Per-trial replacement: only replace source stimulus_onset rows for
        # trials that produced a computed onset.  A trial with missing Looming
        # (no computed onset) must keep its source event (v5 secondary A).
        computed_tids = set(int(t) for t in df_stim["trial_id"])
        replace_mask = (
            (df_e["event_type"] == "stimulus_onset")
            & (df_e["trial_id"].isin(computed_tids))
        )
        if replace_mask.any():
            df_e = df_e[~replace_mask]
        df_e = pd.concat([df_e, df_stim], ignore_index=True)

    kin_columns = KIN_TARGET + ([
        "raw_sys_time", "raw_ard_time", "source_row_index", "time_source",
    ] if clock_audit is not None else [])
    df_k_out = df_k[kin_columns].copy()
    df_e_out = df_e[EVT_TARGET].copy()

    # Rigid shape and schema assertions
    assert df_k_out.shape[1] == len(kin_columns), (
        f"Kinematics column mismatch: {df_k_out.columns.tolist()}"
    )
    assert df_e_out.shape[1] == len(EVT_TARGET), (
        f"Events column mismatch: {df_e_out.columns.tolist()}"
    )
    assert len(df_k_out) > 0, "Kinematics data cannot be empty"
    assert len(df_e_out) > 0, "Events data cannot be empty"

    # ── 5. Write Outputs if Requested ──
    if output_session_dir is not None and target_kin is not None and target_evt is not None:
        _reject_symlink_components(target_kin.parent, kind='output session directory')
        target_kin.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(target_kin.parent, kind='output session directory')
        owned_tmps: List[Tuple[Path, int, int]] = []
        published: List[Tuple[Path, int, int]] = []

        def _write_exclusive(df: pd.DataFrame, target: Path) -> Path:
            fd, tmp_name = tempfile.mkstemp(
                prefix='.' + target.name + '.',
                suffix='.tmp',
                dir=str(target.parent),
            )
            tmp_path = Path(tmp_name)
            try:
                st = os.fstat(fd)
                owned_tmps.append((tmp_path, st.st_dev, st.st_ino))
                with os.fdopen(fd, 'w', newline='', encoding='utf-8') as fh:
                    df.to_csv(fh, index=False)
                    fh.flush()
                    os.fsync(fh.fileno())
            except Exception:
                raise
            try:
                os.chmod(tmp_path, 0o644)
            except OSError:
                pass
            return tmp_path

        try:
            tmp_k = _write_exclusive(df_k_out, target_kin)
            tmp_e = _write_exclusive(df_e_out, target_evt)
            if clock_audit is not None:
                assert clock_target is not None
                clock_audit["staged_kinematics_sha256"] = _clock_source_hash(tmp_k)
                clock_audit["staged_events_sha256"] = _clock_source_hash(tmp_e)
                fd, name = tempfile.mkstemp(dir=str(target_kin.parent))
                tmp_audit = Path(name)
                st = os.fstat(fd)
                owned_tmps.append((tmp_audit, st.st_dev, st.st_ino))
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(clock_audit, stream, indent=2, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                _publish_no_replace(tmp_audit, clock_target, published)
            _publish_no_replace(tmp_k, target_kin, published)
            _publish_no_replace(tmp_e, target_evt, published)
        except Exception:
            _rollback_owned(owned_tmps, published)
            raise
    elif in_place:
        if is_protected_raw_path(k_path) or is_protected_raw_path(e_path):
            raise PermissionError(
                "In-place adaptation is strictly prohibited on authoritative raw source files: "
                f"'{k_path}', '{e_path}'. Specify output_session_dir to stage non-destructively."
            )
        _atomic_replace(df_k_out, k_path)
        _atomic_replace(df_e_out, e_path)

    return df_k_out, df_e_out


def adapt_cercus_to_nsmor(
    raw_dir: Union[str, Path] = "data/raw",
    output_dir: Optional[Union[str, Path]] = None,
    *,
    single_pair: Optional[Tuple[Union[str, Path], Union[str, Path]]] = None,
    session_id: Optional[str] = None,
    experimental_clock_residual_ms: Optional[float] = None,
    experimental_clock_prefix_manifest: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Convert archived or flat cercus session CSVs to canonical NSMoR schema.

    Args:
        raw_dir: Root directory of session CSVs (flat or nested).
        output_dir: Optional absent or empty destination directory for staging.
        single_pair: Optional tuple of (kinematics_path, events_path) for bounded runs.
        session_id: Optional explicit session ID override for single_pair mode.
        experimental_clock_residual_ms: Opt-in operational leave-one-out clock
            residual tolerance in ms; requires staging. See adapt_session_pair.

    Returns:
        Dict with summary report: 'adapted_sessions', 'count_kinematics', 'count_events'.

    Raises:
        PermissionError: If in-place execution (output_dir=None) targets the
            authoritative raw dataset. Checked before any directory scan.
        ValueError: If output_dir conflicts with raw_dir or single_pair inputs.
        FileExistsError: If output_dir is not absent or an empty directory, or
            any staged target already exists (preflight).
        FileNotFoundError: If raw_dir/pair sources are missing or unpaired.
    """
    if (experimental_clock_prefix_manifest is not None
            and experimental_clock_residual_ms is None):
        raise ValueError("Suspect-prefix manifest requires explicit clock tolerance.")
    if experimental_clock_residual_ms is not None and output_dir is None:
        raise ValueError("Experimental clock mapping requires staged output.")
    # ── Safety gate BEFORE any filesystem scan or source read ──
    if output_dir is None:
        if is_protected_raw_path(raw_dir):
            raise PermissionError(
                f"In-place adaptation is strictly prohibited on authoritative data directory: '{raw_dir}'. "
                "Specify an explicit output_dir to stage adapted data non-destructively."
            )
        if single_pair is not None and (
            is_protected_raw_path(single_pair[0]) or is_protected_raw_path(single_pair[1])
        ):
            raise PermissionError(
                "In-place adaptation is strictly prohibited on authoritative raw source files: "
                f"'{single_pair[0]}', '{single_pair[1]}'. Specify output_dir to stage non-destructively."
            )

    raw_resolved: Optional[Path] = None
    out_resolved: Optional[Path] = None
    if output_dir is not None:
        _reject_symlink_components(Path(output_dir), kind='output directory')
        raw_resolved, out_resolved = validate_output_paths(raw_dir, output_dir)

    if single_pair is not None:
        kin_p = Path(single_pair[0])
        evt_p = Path(single_pair[1])
        derived_sid = derive_session_id(kin_p, evt_p)
        if session_id is None:
            sid = derived_sid
        else:
            sid = _validate_session_id(session_id)
            if sid != derived_sid:
                raise ValueError(
                    f"session_id {sid!r} does not match the source stem "
                    f"contract (expected {derived_sid!r})."
                )
        pairs = [(sid, kin_p, evt_p)]
    else:
        pairs = find_session_pairs(raw_dir, output_dir=out_resolved or output_dir)

    # Every sid must be a single safe path component before any join/write.
    for sid, _, _ in pairs:
        _validate_session_id(sid)

    count_k = 0
    count_e = 0
    adapted_sessions: List[str] = []

    if output_dir is not None:
        assert out_resolved is not None
        _reject_symlink_components(out_resolved, kind='output directory')
        # Collision pre-check across ALL pairs before any writing begins
        for sid, _, _ in pairs:
            s_dir = out_resolved / sid
            _reject_symlink_components(s_dir, kind='target session directory')
            t_k = s_dir / f"{sid}_kinematics.csv"
            t_e = s_dir / f"{sid}_events.csv"
            if t_k.exists() or t_k.is_symlink():
                raise FileExistsError(
                    f"Pre-existing target kinematics file already exists: '{t_k}'. "
                    "Aborting before writing any files."
                )
            if t_e.exists() or t_e.is_symlink():
                raise FileExistsError(
                    f"Pre-existing target events file already exists: '{t_e}'. "
                    "Aborting before writing any files."
                )

        # Existing staging roots must be empty: unrelated old sessions would be
        # consumed alongside this run by downstream recursive CSV pairing.
        try:
            output_stat = out_resolved.lstat()
        except FileNotFoundError:
            pass
        else:
            if not stat_mod.S_ISDIR(output_stat.st_mode):
                raise FileExistsError(
                    f"output_dir already exists and is not a directory: '{out_resolved}'."
                )
            if next(out_resolved.iterdir(), None) is not None:
                raise FileExistsError(
                    f"output_dir already exists and is not empty: '{out_resolved}'. "
                    "Use an absent or empty staging directory."
                )

        for sid, kin_p, evt_p in pairs:
            s_dir = out_resolved / sid
            adapt_session_pair(
                kin_p,
                evt_p,
                session_id=sid,
                output_session_dir=s_dir,
                in_place=False,
                experimental_clock_residual_ms=experimental_clock_residual_ms,
                experimental_clock_prefix_manifest=experimental_clock_prefix_manifest,
            )
            count_k += 1
            count_e += 1
            adapted_sessions.append(sid)
    else:
        # Legacy in-place conversion
        if is_protected_raw_path(raw_dir):
            raise PermissionError(
                f"In-place adaptation is strictly prohibited on authoritative data directory: '{raw_dir}'. "
                "Specify an explicit --output_dir to stage adapted data non-destructively."
            )
        for sid, kin_p, evt_p in pairs:
            adapt_session_pair(
                kin_p,
                evt_p,
                session_id=sid,
                in_place=True,
            )
            count_k += 1
            count_e += 1
            adapted_sessions.append(sid)

    print(
        f"Absolute alignment complete: adapted {count_k} sessions "
        f"({count_k} kinematics, {count_e} events)."
    )
    return {
        "adapted_sessions": adapted_sessions,
        "count_kinematics": count_k,
        "count_events": count_e,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert archived legacy-schema session CSVs to the canonical "
            "NSMoR schema. Supports non-destructive output-directory mode via "
            "--output_dir, or legacy in-place rewriting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "raw_dir",
        nargs="?",
        default="data/raw",
        help=(
            "Root directory of raw session CSVs (flat or nested). NOTE: "
            "when --output_dir is omitted, conversion rewrites CSVs IN PLACE "
            "(legacy default, always active unless --output_dir is given; "
            "strictly prohibited on authoritative raw data paths like /mnt/d/Data/train/all). "
            "When --output_dir is provided, raw_dir is strictly read-only."
        ),
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default=None,
        help=(
            "Optional target directory to write adapted canonical session "
            "subdirectories. Avoids mutating source data."
        ),
    )
    parser.add_argument(
        "--in_place",
        "--in-place",
        action="store_true",
        default=False,
        help=(
            "Explicit confirmation of legacy in-place mutation of input files. "
            "In-place rewriting is the active default whenever --output_dir is omitted; "
            "this flag documents that intent and silences the in-place warning. "
            "Strictly prohibited on authoritative raw data paths."
        ),
    )
    parser.add_argument(
        "--kinematics",
        "-k",
        default=None,
        help="Optional single kinematics CSV path to adapt.",
    )
    parser.add_argument(
        "--events",
        "-e",
        default=None,
        help="Optional single events CSV path to adapt.",
    )
    parser.add_argument(
        "--session_id",
        "-s",
        default=None,
        help="Optional explicit session ID override for single pair mode.",
    )
    parser.add_argument(
        "--experimental-clock-residual-ms", type=float, default=None,
        help=("Opt-in per-trial affine arrival-clock association; requires output "
              "staging and explicit max leave-one-out residual in ms. This is an "
              "operational tolerance, not physical timing accuracy. No implicit repairs."),
    )
    parser.add_argument(
        "--experimental-clock-prefix-manifest", default=None,
        help=("Hash-bound manually specified suspect-prefix model assumptions. "
              "Opt-in constant-cadence extrapolation with descriptive sensitivity; "
              "requires explicit clock residual tolerance. Not scientific acceptance."),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.output_dir is not None and args.in_place:
        raise ValueError("Cannot specify both --output_dir and --in_place.")
    if args.output_dir is None and not args.in_place:
        # Legacy in-place is the active default when --output_dir is omitted.
        # Keep CLI contract (bare `pre_load_adapt.py <raw_dir>`), but do not be silent.
        print(
            "WARNING: --output_dir not provided; legacy IN-PLACE rewrite of source CSVs "
            "is active (pass --in_place to confirm explicitly, or --output_dir to stage "
            "non-destructively). Authoritative raw paths are still prohibited.",
            file=sys.stderr,
        )
    if args.kinematics or args.events:
        if not (args.kinematics and args.events):
            raise ValueError(
                "Both --kinematics and --events must be provided for single-pair adaptation."
            )
        adapt_cercus_to_nsmor(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            single_pair=(args.kinematics, args.events),
            session_id=args.session_id,
            experimental_clock_residual_ms=args.experimental_clock_residual_ms,
            experimental_clock_prefix_manifest=args.experimental_clock_prefix_manifest,
        )
    else:
        adapt_cercus_to_nsmor(
            raw_dir=args.raw_dir, output_dir=args.output_dir,
            experimental_clock_residual_ms=args.experimental_clock_residual_ms,
            experimental_clock_prefix_manifest=args.experimental_clock_prefix_manifest,
        )
