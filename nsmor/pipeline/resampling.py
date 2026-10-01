"""Explicit causal model-grid estimates; source observations remain unchanged."""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np

_CHANNELS = (
    "x_pos", "y_pos", "heading", "velocity", "acceleration",
    "visual_angle", "wind_state", "l_v_ratio",
)


def validate_dt_ms(value: Any, *, name: str = "dt_ms") -> float:
    """Require a finite, positive, nonboolean real scalar clock."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive interval")
    try:
        interval = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive interval") from exc
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError(f"{name} must be a finite positive interval")
    return interval


def validate_checkpoint_clock(
    checkpoint: Mapping[str, Any],
    active_dt_ms: float | None = None,
    *,
    require_dt_ms: bool = True,
) -> float | None:
    """Validate serialized cadence before construction or any state restoration."""
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("Invalid checkpoint config: dt_ms required")
    model_config = config.get("model", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("Invalid checkpoint model config: dt_ms required")
    active = (
        validate_dt_ms(active_dt_ms, name="active dt_ms")
        if active_dt_ms is not None else None
    )
    if "dt_ms" not in model_config:
        if require_dt_ms:
            raise ValueError("Checkpoint config lacks explicit dt_ms")
        return None
    saved = validate_dt_ms(model_config["dt_ms"], name="checkpoint dt_ms")
    if active is not None and not math.isclose(
        saved, active, rel_tol=1e-9, abs_tol=0.0,
    ):
        raise ValueError(
            f"Checkpoint dt_ms={saved} conflicts with active dt_ms={active}"
        )
    return saved


def resample_trial_for_model(
    trial_data: dict[str, Any], dt_ms: float,
) -> dict[str, Any]:
    """Hold the last source sample on a uniform grid inside source support.

    No future source sample contributes to a model frame. This matters for
    both stimulus onset and the velocity/acceleration inputs shifted by one
    model step downstream. Held values are estimates, not extra observations.
    Labels and MCMC snapshots must be computed from the original trial.
    Events and source-row clock provenance retain their original coordinates.
    """
    if isinstance(dt_ms, (bool, np.bool_)) or not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be positive and finite")
    times = np.asarray(trial_data["time_ms"], dtype=np.float64)
    if (times.ndim != 1 or not times.size or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Source time_ms must be finite and strictly increasing")
    arrays = {}
    for name in _CHANNELS:
        values = np.asarray(trial_data[name], dtype=np.float64)
        if values.shape != times.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid source channel {name}: shape/finite check")
        arrays[name] = values
    # Subtraction can undercount an aligned endpoint; filter the extra candidate.
    count = int(np.floor((times[-1] - times[0]) / dt_ms)) + 2
    grid = times[0] + np.arange(count, dtype=np.float64) * dt_ms
    grid = grid[grid <= times[-1]]
    if not grid.size or np.any(np.diff(grid) <= 0):
        raise ValueError("Model grid is empty or not representable at this origin")
    indices = np.searchsorted(times, grid, side="right") - 1
    assert indices.shape == grid.shape
    assert np.all((indices >= 0) & (indices < times.size))
    assert np.all(times[indices] <= grid)
    # Every active stimulus interval must reach a distinct held-grid run.
    # Otherwise discretization silently changes the condition or pulse count.
    # Abort rather than fabricate a sample beyond source support or drop a trial.
    for name in ("visual_angle", "wind_state"):
        active = np.abs(arrays[name]) > 0.0
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
        represented = (
            np.searchsorted(indices, ends) > np.searchsorted(indices, starts)
        )
        if not represented.all():
            raise ValueError(
                f"Model grid loses {int((~represented).sum())} active {name} "
                "interval(s); trial unavailable on this grid (no trials dropped)"
            )
        held_active = active[indices]
        held_starts = np.flatnonzero(
            held_active & ~np.r_[False, held_active[:-1]]
        )
        if held_starts.size != starts.size:
            raise ValueError(
                f"Model grid merges active {name} intervals; trial unavailable "
                "on this grid (no trials dropped)"
            )
    result = dict(trial_data)
    result["time_ms"] = grid
    for name, values in arrays.items():
        result[name] = values[indices].copy()
        assert result[name].shape == grid.shape
    result["model_grid_provenance"] = {
        "method": "causal_previous_source_sample_hold",
        "interpretation": "model_grid_estimates_not_observations",
        "source_n": int(times.size), "model_n": int(grid.size),
        "dt_ms": float(dt_ms), "origin_ms": float(grid[0]),
        "source_end_ms": float(times[-1]), "model_end_ms": float(grid[-1]),
        "omitted_tail_ms": float(times[-1] - grid[-1]),
        "max_source_sample_age_ms": float(np.max(grid - times[indices])),
        "singleton": bool(times.size == 1),
        "source_coordinate_scope": "source_CSV_rows_not_model_grid_frames",
        "synthetic_prepend_frames": 0,
    }
    return result


def resolve_model_anchor_frame(provenance: dict[str, Any]) -> int:
    """Map the authoritative source event to its first supported model frame.

    Source timestamps, not held-channel peaks, define the anchor. Include the
    actual sequence prepend without extrapolating past recorded support.
    """
    numeric = (
        "source_anchor_ms", "origin_ms", "source_end_ms", "model_end_ms", "dt_ms",
    )
    integers = ("source_n", "model_n", "synthetic_prepend_frames")
    required = (*numeric, *integers, "source_anchor_rule")
    if not isinstance(provenance, dict) or any(
        name not in provenance for name in required
    ):
        raise ValueError("Incomplete source anchor/model grid provenance")
    for name in numeric:
        value = provenance[name]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)):
            raise ValueError(f"source anchor/model grid {name} must be finite numeric")
    for name in integers:
        value = provenance[name]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))):
            raise ValueError(f"source anchor/model grid {name} must be integer")
    if (not isinstance(provenance["source_anchor_rule"], str)
            or provenance["source_anchor_rule"] not in (
                "looming_collision", "stimulus_onset",
            )):
        raise ValueError("Invalid source anchor rule")
    anchor_ms = float(provenance["source_anchor_ms"])
    origin = float(provenance["origin_ms"])
    end = float(provenance["source_end_ms"])
    dt_ms = float(provenance["dt_ms"])
    count = int(provenance["model_n"])
    prepend = int(provenance["synthetic_prepend_frames"])
    if (not np.isfinite([anchor_ms, origin, end, dt_ms]).all()
            or dt_ms <= 0 or count < 1 or prepend < 0 or provenance["source_n"] < 1
            or not origin <= anchor_ms <= end):
        raise ValueError(f"Invalid source anchor/model grid: {anchor_ms!r}")
    grid = origin + np.arange(count, dtype=np.float64) * dt_ms
    assert grid.shape == (count,)
    if (grid[-1] > end or grid[-1] != provenance["model_end_ms"]
            or np.any(np.diff(grid) <= 0)):
        raise ValueError("Model grid exceeds or disagrees with recorded support")
    index = int(np.searchsorted(grid, anchor_ms, side="left"))
    if index == count:
        raise ValueError(
            f"source anchor {anchor_ms:.17g} ms has no model frame at or after "
            f"it within support (model ends {grid[-1]:.17g} ms)"
        )
    return index + prepend
