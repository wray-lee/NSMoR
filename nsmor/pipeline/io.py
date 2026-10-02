"""
Data I/O — loading and concatenating experimental sessions.

Defines the expected CSV column schemas and provides functions for loading
raw experimental data into pandas DataFrames and per-trial dictionaries.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple, Union

import numpy as np
import pandas as pd
import torch

from nsmor.pipeline.resampling import (
    resample_trial_for_model,
    resolve_model_anchor_frame,
    validate_lazy_artifact_clock,
    validate_source_frame_count,
)

logger = logging.getLogger(__name__)

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


class _SourceKinematics(pd.DataFrame):
    """Keep CSV row boundaries across concat without changing public columns."""

    _metadata = ["_source_lengths"]

    @property
    def _constructor(self):
        return _SourceKinematics

    def __finalize__(self, other, method=None, **kwargs):
        super().__finalize__(other, method=method, **kwargs)
        if method == "concat":
            lengths = []
            for frame in other.objs:
                parts = getattr(frame, "_source_lengths", ())
                lengths.extend(parts if parts and sum(parts) == len(frame) else (len(frame),))
            self._source_lengths = tuple(lengths)
        return self


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
    *,
    source_path: Union[str, Path, None] = None,
    audit_sidecar: Dict[str, Any] | None = None,
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
        path: File path or binary stream to the kinematics CSV.
        artifact_velocity_cm_s: Single-frame velocity magnitude above which
            a trial-first frame's velocity is treated as a cross-trial
            sensor-jump artifact (zeroed).  Real escape onsets are ~10^1-10^2
            cm/s; the sensor spike is 10^3+ cm/s, so ``1000`` is a safe
            three-order-of-magnitude separation.  ``float("inf")`` disables.
        source_path: Origin path of the bytes in *path*.  Required only when
            *path* is a stream (e.g. a snapshot) and the CSV carries
            clock-provenance columns, since the sibling ``*_timebase_audit.json``
            and ``*_events.csv`` are resolved relative to this origin.  A bare
            stream cannot bind them and fails closed.
        audit_sidecar: Pre-parsed audit JSON bound at snapshot-capture time.
            When supplied, its SHA-256 (``audit_sha256``) and canonical
            ``audit_path`` must match the live sibling audit file; otherwise
            the read fails closed.  This prevents a captured CSV byte snapshot
            from being re-paired with a post-capture audit mutation, because
            the provenance reference embeds the captured audit bytes.

    Returns:
        DataFrame with columns matching :data:`KINEMATICS_COLUMNS`.

    Raises:
        ValueError: If required columns are missing, if clock-provenance
            columns are present but no origin path is available, or if a
            bound ``audit_sidecar`` no longer matches the live audit file.
    """
    df = pd.read_csv(path, converters={"raw_sys_time": str, "raw_ard_time": str})
    missing = set(KINEMATICS_COLUMNS) - set(df.columns)
    if missing:
        raw_markers = {"sys_time", "stim_state"}
        if raw_markers <= set(df.columns):
            raise ValueError(
                f"{path} is still in the raw sensor schema "
                f"(sys_time/stim_state), missing {sorted(missing)}. "
                "Run scripts/pre_load_adapt.py on a staging copy first so "
                "stim_state becomes wind_state; skipping that step collapses "
                "wind-only trials to no_stimulus."
            )
        raise ValueError(f"Missing columns in {path}: {missing}")
    clock_columns = ["raw_sys_time", "raw_ard_time", "source_row_index", "time_source"]
    has_clock = any(c in df for c in clock_columns)
    if has_clock and not all(c in df for c in clock_columns):
        raise ValueError(f"Incomplete clock provenance columns in {path}")
    df = df[KINEMATICS_COLUMNS + (clock_columns if has_clock else [])].copy()
    if has_clock:
        if source_path is not None:
            source = Path(source_path)
        else:
            try:
                source = Path(path)
            except TypeError as exc:
                raise ValueError(
                    "Clock-provenance columns require source_path=<origin CSV "
                    "path> to resolve the sibling audit/events files; a bare "
                    f"stream cannot bind them: {path!r}"
                ) from exc

        audit_path = source.with_name(
            source.name.removesuffix("_kinematics.csv") + "_timebase_audit.json"
        )
        if audit_sidecar is not None and audit_sidecar.get("audit_absent"):
            raise ValueError(
                f"Clock audit absent at snapshot capture: {audit_path}; the "
                "captured CSV cannot attach later provenance"
            )
        audit_bytes = audit_path.read_bytes()
        if audit_sidecar is not None:
            bound_path = Path(audit_sidecar["audit_path"])
            if bound_path.resolve() != audit_path.resolve():
                raise ValueError(
                    f"Captured audit sidecar is bound to {bound_path}, not "
                    f"the live sibling {audit_path}"
                )
            if hashlib.sha256(audit_bytes).hexdigest() != audit_sidecar["audit_sha256"]:
                raise ValueError(
                    f"Clock audit changed after snapshot capture: {audit_path}"
                )
            audit = dict(audit_sidecar["audit"])
        else:
            audit = json.loads(audit_bytes)
        if audit.get("status") != "accepted_by_operational_tolerance":
            raise ValueError(f"Clock audit is not accepted: {audit_path}")
        for candidate, key in (
            (source, "staged_kinematics_sha256"),
            (source.with_name(source.name.replace("_kinematics.csv", "_events.csv")),
             "staged_events_sha256"),
        ):
            digest = hashlib.sha256()
            with candidate.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != audit.get(key):
                raise ValueError(f"Clock provenance hash mismatch: {candidate}")
        if not df["time_source"].isin({
            "experimental_affine_estimate", "experimental_prefix_cadence_estimate",
            "single_row_host_only",
        }).all():
            raise ValueError(f"Unknown clock estimate source in {path}")
        # A bound reference survives concat/slicing without confusing raw row
        # indices with the later resampled/cropped tensor coordinates.
        df["clock_audit_reference"] = json.dumps({
            "audit_path": str(audit_path.resolve()),
            "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "kinematics_sha256": audit["kinematics_sha256"],
            "events_sha256": audit["events_sha256"],
            "method": audit["method"],
            "operational_tolerance_ms": audit["operational_tolerance_ms"],
            "prefix_manifest_sha256": audit.get("prefix_manifest_sha256"),
            "scientific_acceptance": audit.get("scientific_acceptance", "unresolved"),
            "coordinate_scope": "source_CSV_rows_not_resampled_tensor_frames",
        }, sort_keys=True)
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
        # Recompute acceleration as the true time derivative dv/dt
        # (cm/s²) using the ACTUAL per-sample interval from time_ms.
        # Why recompute ALL rows (not just artifact rows): a zeroed
        # artifact velocity at frame 0 changes the diff at frame 1
        # (the echo spike), so downstream frames are affected too.
        # The cleanest invariant is: acceleration is ALWAYS the
        # physical dv/dt of the cleaned velocity column.
        acc_out = np.zeros(len(df), dtype=np.float64)
        for _, grp in df.groupby(["session_id", "trial_id"], sort=False):
            idx = grp.index
            v = grp["velocity"].values
            t = grp["time_ms"].values
            dv = np.concatenate([[0.0], np.diff(v)])           # cm/s
            dt_ms_arr = np.concatenate([[0.0], np.diff(t)])    # ms
            pos_gaps = dt_ms_arr[dt_ms_arr > 0]
            floor_ms = float(np.min(pos_gaps)) if pos_gaps.size > 0 else 1.0
            dt_safe = np.where(dt_ms_arr > 0, dt_ms_arr, floor_ms)
            acc = dv / (dt_safe / 1000.0)  # cm/s²
            acc[0] = 0.0  # first frame: no prior sample
            acc_out[idx] = acc
        df["acceleration"] = acc_out
    loaded = _SourceKinematics(df)
    loaded._source_lengths = (len(loaded),)
    return loaded


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
        ValueError: If rows are missing or trial timestamps/declarations are ambiguous.
    """
    kin = session_data["kinematics"]
    mask_kin = (kin["session_id"] == session_id) & (kin["trial_id"] == trial_id)
    kin_trial = kin.loc[mask_kin]
    if kin_trial.empty:
        raise ValueError(
            f"No kinematics data for session={session_id!r}, trial={trial_id}"
        )

    identity = f"session={session_id!r}, trial={trial_id!r}"
    times = kin_trial["time_ms"].to_numpy(dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(np.sort(times)) <= 0):
        raise ValueError(f"Ambiguous {identity}: repeated or non-finite time_ms")
    lengths = getattr(kin, "_source_lengths", ())
    if lengths and sum(lengths) == len(kin):
        end = 0
        previous_end = -np.inf
        for length in lengths:
            part = kin.iloc[end:end + length]
            end += length
            part_times = part.loc[
                (part["session_id"] == session_id) & (part["trial_id"] == trial_id),
                "time_ms",
            ]
            if part_times.empty:
                continue  # Event-only continuation has no kinematics range.
            first, last = float(part_times.min()), float(part_times.max())
            if first <= previous_end:
                raise ValueError(
                    f"Ambiguous {identity}: overlapping or unordered CSV pair time_ms ranges"
                )
            previous_end = last

    evt = session_data["events"]
    mask_evt = (evt["session_id"] == session_id) & (evt["trial_id"] == trial_id)
    evt_trial = evt.loc[mask_evt]
    if evt_trial["event_type"].eq("trial_start").sum() > 1:
        raise ValueError(f"Ambiguous {identity}: repeated trial_start declarations")
    kin_trial = kin_trial.sort_values("time_ms")
    evt_trial = evt_trial.sort_values("time_ms")
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

    clock_provenance = []
    if "clock_audit_reference" in kin_trial:
        for reference, rows in kin_trial.groupby("clock_audit_reference", sort=False):
            entry = json.loads(reference)
            entry["source_row_indices"] = rows["source_row_index"].astype(int).tolist()
            entry["raw_sys_time"] = rows["raw_sys_time"].astype(str).tolist()
            entry["raw_ard_time"] = rows["raw_ard_time"].astype(str).tolist()
            entry["time_source"] = rows["time_source"].tolist()
            clock_provenance.append(entry)

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
        **({"clock_provenance": clock_provenance} if clock_provenance else {}),
    }


# ──────────────────────────────────────────────────────────────
# Clock-aware bounded-memory snapshot loading
# ──────────────────────────────────────────────────────────────

def _csv_sha256(path: Path, snapshot: Any = None) -> str:
    """Root ``csv_sha256``; deferred import breaks the io <-> loader cycle."""
    from nsmor.lazy_dataloader import csv_sha256

    return csv_sha256(path, snapshot)


def load_csv_snapshot(
    path: Union[str, Path],
    loader: Callable[..., pd.DataFrame],
    expected: Union[str, None] = None,
    *,
    source_path: Union[str, Path, None] = None,
) -> Tuple[pd.DataFrame, str]:
    """Parse the same bounded-memory byte snapshot whose SHA-256 is recorded.

    Editable, clock-aware counterpart to ``nsmor.lazy_dataloader.load_csv_snapshot``.
    Unlike the frozen root helper, it can bind a snapshot *stream* to its origin
    path so a clock-stamped kinematics CSV resolves its sibling
    ``*_timebase_audit.json`` / ``*_events.csv`` relative to the real origin
    rather than the pathless ``SpooledTemporaryFile``. The two are numerically
    equivalent: the digest is of the source bytes, and the pre/post digest
    checks bracket parsing exactly as before.

    Args:
        path: Source CSV path whose bytes are hashed and copied.
        loader: Callable ``loader(stream[, source_path=...])``. The
            ``source_path`` kwarg is forwarded only when supplied.
        expected: Recorded SHA-256; a mismatch fails closed.
        source_path: Origin path of the snapshot bytes. When omitted, no
            ``source_path`` kwarg is forwarded (legacy contract). Must be the
            actual origin, never ``snapshot.name``.

    Returns:
        ``(table, digest)`` where ``digest`` is the SHA-256 of ``path``'s bytes.

    Raises:
        ValueError: If ``expected`` is supplied and does not match, or if the
            source bytes change between the pre- and post-parse digests.
    """
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b") as snapshot:
        digest = _csv_sha256(path, snapshot)
        if expected is not None and digest != expected:
            raise ValueError(f"Source CSV changed: {path}; regenerate metadata from raw CSVs")
        snapshot.seek(0)
        table = loader(
            snapshot,
            **({"source_path": source_path} if source_path is not None else {}),
        )
    if _csv_sha256(path) != digest:
        raise ValueError(f"Source CSV changed: {path}; regenerate metadata from raw CSVs")
    return table, digest


def source_paths(source: Dict) -> Tuple[Path, Path]:
    """Root ``source_paths``; deferred import breaks the io <-> loader cycle."""
    from nsmor.lazy_dataloader import source_paths as _root_source_paths

    return _root_source_paths(source)


def source_digests(source: Dict) -> Tuple[str, str]:
    """Root ``source_digests``; deferred import breaks the io <-> loader cycle."""
    from nsmor.lazy_dataloader import source_digests as _root_source_digests

    return _root_source_digests(source)


def verify_source_pair(source: Dict) -> None:
    """Root ``verify_source_pair``; deferred import breaks the io <-> loader cycle."""
    from nsmor.lazy_dataloader import verify_source_pair as _root_verify_source_pair

    _root_verify_source_pair(source)


def _audit_path_for(kin_path: Path) -> Path:
    """Sibling ``*_timebase_audit.json`` for a kinematics CSV path."""
    return kin_path.with_name(
        kin_path.name.removesuffix("_kinematics.csv") + "_timebase_audit.json"
    )


def _capture_audit_sidecar(kin_path: Path) -> Dict[str, Any]:
    """Bind a kinematics CSV's sibling audit JSON at capture time.

    Reads the audit bytes once and records their SHA-256 alongside the parsed
    document, so a later :func:`load_kinematics_csv` can fail closed if the
    audit file changes after the CSV snapshot was captured. When the sibling
    audit does not exist at capture, an explicit ``{"audit_absent": True}``
    marker is returned instead of ``None``: absence is itself provenance, and a
    captured clock-stamped CSV whose audit was absent must fail closed rather
    than attach a later-invented audit. (Non-clock pairs ignore the marker,
    preserving legacy behaviour.)
    """
    audit_path = _audit_path_for(kin_path)
    if not audit_path.exists():
        return {"audit_absent": True}
    audit_bytes = audit_path.read_bytes()
    return {
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "audit": json.loads(audit_bytes),
    }


class ClockAwareLazyDataset:
    """Clock-aware replay of a metadata's trials from captured CSV bytes.

    Thin, boundary-safe wrapper over the frozen root
    :class:`nsmor.lazy_dataloader.NSMoRLazyDataset`. It composes — never
    copies or monkeypatches — the root class, and swaps only the two source
    reads that must bind a snapshot stream to its origin path:

    * the cold path routes through :func:`load_csv_snapshot` (this module),
    * the ``captured_sources`` path hands the origin path to
      :func:`load_kinematics_csv`.

    Everything else (feature layout, pure-wind prepend, anchor cropping,
    MCMC prior tiling, digest rechecks) is inherited unchanged from the frozen
    loader, so a clock-stamped pair replays byte-identically to the eager path.

    The wrapper is picklable: the frozen inner dataset is the sole pickled
    state, and the per-instance ``_load_session`` override is rebound on
    unpickle. ``captured_sources`` also binds each kinematics CSV's sibling
    clock audit bytes, so a captured CSV snapshot can never be re-paired with a
    post-capture audit mutation.
    """

    def __init__(self, metadata_path: str, **kwargs: Any) -> None:
        from nsmor.lazy_dataloader import NSMoRLazyDataset

        self._metadata_path = metadata_path
        # Acquire exactly one metadata snapshot and validate *that* object, then
        # hand it to the frozen inner loader through its existing ``metadata=``
        # seam.  Reading the path here and again inside ``_resolve_lazy_clock``
        # would let a path swapped between the two reads be validated as one
        # revision while a different revision is served.
        metadata = kwargs.pop("metadata", None)
        if metadata is None:
            from nsmor.pipeline.nested_prior import load_artifact_bytes

            metadata = load_artifact_bytes(Path(metadata_path).read_bytes())
        self._metadata = metadata
        self._inner = NSMoRLazyDataset(metadata_path, metadata=metadata, **kwargs)
        self._source_audits: Dict[Path, Dict[str, Any]] | None = None
        # Redirect the frozen loader's per-trial source read to the clock-aware
        # override below, per instance (no global monkeypatching).
        self._inner._load_session = self._load_session
        # The frozen ``__getitem__`` compares reconstructed frames against
        # ``spec['n_frames']``, which counts *source* frames.  When the trial is
        # resampled onto the model grid the built length is ``model_n`` (+ the
        # synthetic pure-wind prepend), so the item build is owned here and
        # delegates the feature math to the frozen ``_build_sequence``.
        self._inner._build_sequence = self._build_sequence
        self._lazy_clock_contract: Dict[str, Any] | None = None
        self._lazy_clock_dt_ms: float | None = None
        self._last_grid_record: Dict[str, Any] | None = None
        self._resolve_lazy_clock()

    def _resolve_lazy_clock(self) -> None:
        """Validate the metadata's lazy model-clock contract, if declared.

        Absence keeps the legacy source-cadence behavior (with an explicit
        unverified-clock warning); a present-but-invalid (including present-but-
        null) contract, or a per-spec flag contradicting the top-level
        declaration, fails closed via the shared validator.
        """
        metadata = self._metadata
        dt_ms = validate_lazy_artifact_clock(metadata, self._inner.dt_ms)
        if dt_ms is None:
            logger.warning(
                "Lazy metadata declares no model-clock contract; source cadence "
                "is unverified and no model-grid resampling is applied."
            )
            self._lazy_clock_contract = None
            self._lazy_clock_dt_ms = None
        else:
            self._lazy_clock_contract = dict(metadata["lazy_model_clock_contract"])
            self._lazy_clock_dt_ms = dt_ms

    @property
    def uses_model_grid(self) -> bool:
        """Whether a declared lazy model-clock contract resamples trials."""
        return self._lazy_clock_dt_ms is not None

    def __getattr__(self, name: str) -> Any:
        # ``_inner`` is set in ``__init__``; guard the unpickle window (and any
        # other pre-init access) so attribute probes cannot recurse forever.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle the frozen inner dataset plus the small resolved clock state.

        The instance-bound overrides cannot be pickled; the resolved contract
        and dt are stored so the rebuilt item math is identical after unpickle.
        """
        return {
            "_inner": self._inner,
            "_metadata_path": self._metadata_path,
            "_lazy_clock_contract": self._lazy_clock_contract,
            "_lazy_clock_dt_ms": self._lazy_clock_dt_ms,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._metadata = None
        self._source_audits = None
        self._last_grid_record = None
        # The bound methods cannot be pickled; rebind them to this wrapper.
        self._inner._load_session = self._load_session
        self._inner._build_sequence = self._build_sequence

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Any:
        # A declared model clock builds the item here: the frozen check compares
        # the built length against ``spec['n_frames']`` (source frames), which is
        # intentionally different from the model-grid length.  Legacy metadata
        # keeps the frozen item path byte-identical.
        if self._lazy_clock_dt_ms is None:
            return self._inner[idx]
        inner = self._inner
        spec = inner.trial_specs[idx]
        trial_data = extract_trial_data(
            self._load_session(spec),
            session_id=spec["session_id"],
            trial_id=spec["trial_id"],
        )
        X_seq, Y_seq = self._build_sequence(trial_data, idx)
        # Reuse the grid record built during the single resample above rather
        # than resampling the trial a second time for the anchor.
        anchor_frame = (
            None if self._last_grid_record is None
            else resolve_model_anchor_frame(self._last_grid_record)
        )
        actual_length = X_seq.shape[0]
        if inner.max_seq_len is not None and actual_length > inner.max_seq_len:
            from nsmor.pipeline.conditions import resolve_anchor_crop

            start, end = resolve_anchor_crop(
                n_frames=actual_length,
                anchor_frame=anchor_frame,
                max_seq_len=inner.max_seq_len,
                pre_anchor_frames=inner.pre_anchor_frames,
            )
            X_seq = X_seq[start:end]
            Y_seq = Y_seq[start:end]
            actual_length = X_seq.shape[0]
        return (
            torch.from_numpy(X_seq).float(),
            torch.from_numpy(Y_seq).float(),
            actual_length,
        )

    @contextmanager
    def captured_sources(self) -> Iterator[List[Dict[str, str]]]:
        """Bind private, pathless copies of every contributing CSV's bytes.

        Mirrors the frozen root context manager (digest binding, conflict
        detection, cache reset, revision report) but retains the origin path
        per snapshot so a clock-stamped kinematics snapshot can still resolve
        its sibling audit/events files during ``__getitem__``.
        """
        inner = self._inner
        with ExitStack() as stack:
            snapshots: Dict[Path, Tuple[Any, str]] = {}
            audits: Dict[Path, Dict[str, Any]] = {}
            revision: List[Dict[str, str]] = []
            seen = set()
            for spec in inner.trial_specs:
                for source in spec.get("source_pairs", [spec]):
                    paths = source_paths(source)
                    digests = source_digests(source)
                    if (*paths, *digests) in seen:
                        continue
                    for path, expected in zip(paths, digests):
                        if path in snapshots:
                            if snapshots[path][1] != expected:
                                raise ValueError(f"Conflicting SHA-256 for source CSV: {path}")
                            continue
                        snapshot = stack.enter_context(
                            tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")
                        )
                        if _csv_sha256(path, snapshot) != expected or _csv_sha256(path) != expected:
                            raise ValueError(
                                f"Source CSV changed: {path}; regenerate metadata from raw CSVs"
                            )
                        snapshots[path] = (snapshot, expected)
                    audits[paths[0]] = _capture_audit_sidecar(paths[0])
                    seen.add((*paths, *digests))
                    revision.append({key: source[key] for key in (
                        "session_dir", "kinematics_file", "events_file",
                        "kinematics_sha256", "events_sha256",
                    )})
            inner._session_cache.clear()
            inner._cache_keys.clear()
            inner._source_snapshots = snapshots
            self._source_audits = audits
            try:
                yield revision
            finally:
                inner._source_snapshots = None
                self._source_audits = None
                inner._session_cache.clear()
                inner._cache_keys.clear()

    def _load_session(self, spec: Dict) -> Dict[str, pd.DataFrame]:
        """Replay a trial's source pairs, binding snapshot streams to origins."""
        inner = self._inner
        sources = spec["source_pairs"] if "source_pairs" in spec else [spec]
        if not sources:
            raise ValueError("Trial metadata has no source pairs")

        kin_parts: List[pd.DataFrame] = []
        evt_parts: List[pd.DataFrame] = []
        for source in sources:
            kin_path, evt_path = source_paths(source)
            kin_digest, evt_digest = source_digests(source)
            cache_key = (kin_path, evt_path, kin_digest, evt_digest)
            if cache_key not in inner._session_cache:
                if inner._source_snapshots is None:
                    pair = {
                        "kinematics": load_csv_snapshot(
                            kin_path, load_kinematics_csv, kin_digest,
                            source_path=kin_path,
                        )[0],
                        "events": load_csv_snapshot(
                            evt_path, load_events_csv, evt_digest,
                        )[0],
                    }
                    verify_source_pair(source)
                else:
                    kin_snapshot = inner._source_snapshots[kin_path][0]
                    evt_snapshot = inner._source_snapshots[evt_path][0]
                    kin_snapshot.seek(0)
                    evt_snapshot.seek(0)
                    bound_audit = (self._source_audits or {}).get(kin_path)
                    pair = {
                        "kinematics": load_kinematics_csv(
                            kin_snapshot,
                            source_path=kin_path,
                            audit_sidecar=bound_audit,
                        ),
                        "events": load_events_csv(evt_snapshot),
                    }
                if len(inner._session_cache) >= inner._cache_size:
                    evict_key = inner._cache_keys.pop(0)
                    inner._session_cache.pop(evict_key)
                inner._session_cache[cache_key] = pair
                inner._cache_keys.append(cache_key)
            else:
                # The converter reuses its private captured revision; training checks live paths.
                if inner._source_snapshots is None:
                    verify_source_pair(source)
                inner._cache_keys.remove(cache_key)
                inner._cache_keys.append(cache_key)
                pair = inner._session_cache[cache_key]
            kin_parts.append(pair["kinematics"])
            evt_parts.append(pair["events"])

        if len(sources) == 1:
            return pair
        return {
            "kinematics": pd.concat(kin_parts, ignore_index=True),
            "events": pd.concat(evt_parts, ignore_index=True),
        }

    def _prepend_frames(self, spec: Dict) -> int:
        """Synthetic leading-zero count for a pure-wind trial.

        On the declared model grid the prepend is computed from the *model* dt
        (5.7 s / dt_ms), never the source cadence.  Legacy metadata without a
        contract keeps its recorded source-cadence prepend.
        """
        if not bool(spec.get("is_pure_wind")):
            return 0
        if self._lazy_clock_dt_ms is not None:
            from nsmor.data_extractor import _compute_pure_wind_prepend_frames

            return _compute_pure_wind_prepend_frames(self._lazy_clock_dt_ms)
        return int(spec.get("pure_wind_prepended_frames", 0) or 0)

    def _build_sequence(
        self, trial_data: Dict[str, np.ndarray], idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample the source trial onto the model grid, then build features.

        Delegates the feature assembly to the frozen
        ``NSMoRLazyDataset._build_sequence`` (called on the class, so the
        per-instance binding does not recurse); only the pure-wind prepend is
        forced to the model-grid count.  A trial with no declared contract is
        passed through unchanged, preserving legacy byte-identical behavior.

        The eager grid/anchor record for this trial is cached on
        ``self._last_grid_record`` so a single caller can reuse one resample for
        both the features and the anchor, instead of resampling twice.
        """
        spec = self._inner.trial_specs[idx]
        self._last_grid_record = None
        if self._lazy_clock_dt_ms is not None:
            # Source completeness is checked on raw source frames, before any
            # resampling: a partial/missing contributing pair must fail here.
            validate_source_frame_count(spec, len(trial_data["time_ms"]))
            source_trial = trial_data
            trial_data = resample_trial_for_model(trial_data, self._lazy_clock_dt_ms)
            self._last_grid_record = self._grid_record(spec, source_trial, trial_data)
        previous = spec.get("pure_wind_prepended_frames")
        spec["pure_wind_prepended_frames"] = self._prepend_frames(spec)
        try:
            return type(self._inner)._build_sequence(self._inner, trial_data, idx)
        finally:
            if previous is None:
                spec.pop("pure_wind_prepended_frames", None)
            else:
                spec["pure_wind_prepended_frames"] = previous

    def _grid_record(
        self, spec: Dict, source_trial: Dict[str, np.ndarray],
        model_trial: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Build the eager grid/anchor record from a source and resampled trial.

        Single builder shared by :meth:`_build_sequence` and
        :meth:`model_grid_provenance`, so the on-demand item and the converter's
        published artifact carry the exact same contract.
        """
        from nsmor.data_extractor import resolve_snapshot_anchor

        anchor_ms, anchor_rule = resolve_snapshot_anchor(
            source_trial, float(spec["stimulus_onset_ms"]),
        )
        record = dict(model_trial["model_grid_provenance"])
        record["synthetic_prepend_frames"] = self._prepend_frames(spec)
        record["source_anchor_ms"] = float(anchor_ms)
        record["source_anchor_rule"] = anchor_rule
        resolve_model_anchor_frame(record)  # fail closed on an unmappable anchor
        return record

    def model_grid_provenance(
        self, spec: Dict, trial_data: Dict[str, np.ndarray],
    ) -> Dict[str, Any] | None:
        """Full eager grid/anchor record for a resampled trial, or ``None``.

        Reuses the shared :func:`resample_trial_for_model` record and the shared
        :func:`resolve_snapshot_anchor` source anchor, so the converter's eager
        artifact carries the exact contract the restricted loader validates.
        Source completeness is re-validated here (before resampling) so the
        converter refuses a partial trial even if it never called ``__getitem__``.
        """
        if self._lazy_clock_dt_ms is None:
            return None
        validate_source_frame_count(spec, len(trial_data["time_ms"]))
        model_trial = resample_trial_for_model(trial_data, self._lazy_clock_dt_ms)
        return self._grid_record(spec, trial_data, model_trial)

    def model_anchor_frame(
        self, spec: Dict, trial_data: Dict[str, np.ndarray],
    ) -> int | None:
        """Map the source anchor onto the model grid, or ``None`` for legacy."""
        record = self.model_grid_provenance(spec, trial_data)
        return None if record is None else resolve_model_anchor_frame(record)
