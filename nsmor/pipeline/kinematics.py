"""
Kinematics processing: smoothing, velocity / acceleration computation.

Provides Savitzky-Golay and Gaussian kernel smoothing for raw position data,
with utilities for computing derived kinematic quantities.

Coordinate convention (fixed for mirroring):
    - allocentric frame: +Y = forward (anterior), +X = right (lateral),
      heading = yaw angle in degrees, 0 = +Y, CCW positive (standard
      mathematical rotation). This matches pre_load_adapt.py:
        heading = cumsum(degrees(dz / 30))
        x_pos = cumsum(dx*cos - dy*sin)/10, y_pos = cumsum(dx*sin + dy*cos)/10
    - Mirroring across the sagittal (Y) plane: x -> -x, heading -> -heading
      (mod 360), dz -> -dz, dx -> -dx, vel_x -> -vel_x. Scalar speed
      ``velocity = sqrt(dx^2+dy^2)/dt`` is rotation-invariant, so Y target
      is identity under reflection. Fields y_pos, visual_angle, l_v_ratio
      are sagittal-invariant and intentionally not mirrored.
    - Biological/energetic invariance: cercal GI afferents and downstream
      giant-fiber conduction are bilaterally symmetric; ATP cost of spiking
      scales with scalar spike-count (velocity magnitude), not signed
      lateral direction, and synaptic delay is symmetric across midline. No
      lateralized metabolic or delay asymmetry is introduced by reflection.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


WIND_SIDES = {"left", "right"}


def _copy_trial(trial: Dict[str, Any]) -> Dict[str, Any]:
    """Return a defensive deep copy of a trial dictionary."""
    # deepcopy ensures object-dtype arrays (event_values) do not share refs
    return deepcopy(trial)


def mirror_to_right(trial: Dict[str, Any]) -> Dict[str, Any]:
    """Reflect a left-wind trial into the canonical right-wind frame.

    The processed NSMoR target ``velocity`` is scalar speed magnitude, so it
    is invariant under reflection. The global x coordinate is lateral; its
    sign and the corresponding heading angle are reflected. Raw directional
    ``dx`` and ``vel_x`` fields are reflected when present. Raw ``dz`` is yaw
    increment and is reflected with heading. ``y_pos``, ``visual_angle``,
    ``l_v_ratio`` are sagittal-invariant and left unchanged. Provenance
    (``wind_side_mirrored``) is the sole idempotency source; repeated calls
    are no-ops.

    Biological basis: cricket cercal wind afferents are bilateral; escape
    direction is lateralized (mirror-symmetric). Pooling left/right by
    reflection preserves the biomechanical symmetry while keeping scalar
    vigor (and thus spike-count ATP cost and axonal conduction delay,
    both symmetric across midline) invariant.

    Args:
        trial: Per-trial dictionary from ``extract_trial_data``.

    Returns:
        A copied trial in the canonical right-wind frame.
    """
    mirrored = _copy_trial(trial)
    # Defensive shape assertions for core kinematics (1-D time series)
    if "time_ms" in mirrored:
        assert isinstance(mirrored["time_ms"], np.ndarray), "time_ms must be ndarray"
        assert mirrored["time_ms"].ndim == 1, f"time_ms ndim {mirrored['time_ms'].ndim} !=1"
        T = mirrored["time_ms"].shape[0]
        for key in ("x_pos", "y_pos", "heading", "velocity", "acceleration"):
            if key in mirrored and isinstance(mirrored[key], np.ndarray):
                assert mirrored[key].shape == (T,), f"{key} shape {mirrored[key].shape} != ({T},)"
    original_side = str(
        mirrored.get("wind_side_original", mirrored.get("wind_side", "unknown"))
    ).lower()
    mirrored["wind_side_original"] = original_side

    # Idempotency: wind_side_mirrored is the single source of truth
    if mirrored.get("wind_side_mirrored") is True:
        mirrored["wind_side_unified"] = "right"
        return mirrored
    if mirrored.get("wind_side_mirrored") is False:
        # Already processed as non-mirrored; preserve unified value
        return mirrored
    # Legacy path: no provenance yet. If already unified right with
    # original right, mark as non-mirrored and return.
    if mirrored.get("wind_side_unified") == "right" and original_side == "right":
        mirrored["wind_side_mirrored"] = False
        return mirrored
    # If unified==right but original==left with no provenance, treat as
    # not-yet-mirrored (dirty metadata) and fall through to flip.

    if original_side == "left":
        for field in ("x_pos", "dx", "dz", "vel_x"):
            if field in mirrored:
                arr = np.asarray(mirrored[field])
                assert arr.ndim == 1, f"{field} ndim {arr.ndim} !=1"
                if "T" in locals():
                    assert arr.shape[0] == T, f"{field} len {arr.shape[0]} != T={T}"
                mirrored[field] = -arr
        if "heading" in mirrored:
            h = np.asarray(mirrored["heading"])
            assert h.ndim == 1, f"heading ndim {h.ndim} !=1"
            if "T" in locals():
                assert h.shape[0] == T, f"heading len {h.shape[0]} != T={T}"
            mirrored["heading"] = (-h) % 360.0
        mirrored["wind_side_unified"] = "right"
        mirrored["wind_side_mirrored"] = True
    else:
        mirrored["wind_side_unified"] = (
            "right" if original_side == "right" else original_side
        )
        mirrored["wind_side_mirrored"] = False

    return mirrored


def demirror_prediction(
    y_pred_unified: np.ndarray, original_side: str,
) -> np.ndarray:
    """Map unified scalar-speed predictions to the original-side view.

    ``Y`` is speed magnitude (derived from Euclidean displacement), not a
    signed lateral component. Reflection therefore leaves prediction values
    unchanged; the original side is preserved separately for absolute-side
    grouping in analyses.

    Args:
        y_pred_unified: Predicted scalar speed in the canonical frame,
            shape (T,) or (N, T).
        original_side: Original wind side (``left``, ``right``, or unknown).

    Returns:
        A copy of the prediction in the original experimental frame.
    """
    arr = np.asarray(y_pred_unified)
    assert arr.ndim in (1, 2), f"y_pred ndim {arr.ndim} not in (1,2)"
    del original_side
    return arr.copy()


def smooth_kinematics(
    data: np.ndarray,
    method: Literal["savgol", "gaussian"] = "savgol",
    window_length: int = 11,
    polyorder: int = 3,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Apply smoothing to a 1-D kinematics time series.

    Args:
        data: 1-D array of position / velocity values.
        method: ``"savgol"`` for Savitzky-Golay or ``"gaussian"`` for
            Gaussian kernel smoothing.
        window_length: Window length for Savitzky-Golay (must be odd;
            incremented by 1 if even).
        polyorder: Polynomial order for Savitzky-Golay.
        sigma: Standard deviation (in samples) for the Gaussian kernel.

    Returns:
        Smoothed array of the same shape as *data*.

    Raises:
        ValueError: If *data* is not 1-D or *method* is unknown.
    """
    if data.ndim != 1:
        raise ValueError(f"Expected 1-D array, got {data.ndim}-D.")

    if method == "savgol":
        if window_length % 2 == 0:
            window_length += 1
        return savgol_filter(data, window_length, polyorder)
    elif method == "gaussian":
        return gaussian_filter1d(data, sigma)
    else:
        raise ValueError(f"Unknown smoothing method: {method!r}")


def compute_velocity(
    position: np.ndarray,
    dt_ms: float = 10.0,
    smooth: bool = True,
    **smooth_kwargs: object,
) -> np.ndarray:
    """
    Compute instantaneous velocity from a position time series.

    Args:
        position: 1-D array of positions (cm).
        dt_ms: Frame interval in milliseconds.
        smooth: Whether to smooth *position* before differentiation.
        **smooth_kwargs: Forwarded to :func:`smooth_kinematics`.

    Returns:
        1-D array of velocities (cm / s), same length as *position*.
    """
    dt_s = dt_ms / 1000.0

    if smooth:
        position = smooth_kinematics(position, **smooth_kwargs)

    return np.gradient(position, dt_s)


def compute_acceleration(
    velocity: np.ndarray,
    dt_ms: float = 10.0,
    smooth: bool = True,
    **smooth_kwargs: object,
) -> np.ndarray:
    """
    Compute instantaneous acceleration from a velocity time series.

    Args:
        velocity: 1-D array of velocities (cm / s).
        dt_ms: Frame interval in milliseconds.
        smooth: Whether to smooth *velocity* before differentiation.
        **smooth_kwargs: Forwarded to :func:`smooth_kinematics`.

    Returns:
        1-D array of accelerations (cm / s²), same length as *velocity*.
    """
    dt_s = dt_ms / 1000.0

    if smooth:
        velocity = smooth_kinematics(velocity, **smooth_kwargs)

    return np.gradient(velocity, dt_s)


def compute_kinematics(
    position: np.ndarray,
    dt_ms: float = 10.0,
    smooth_method: Literal["savgol", "gaussian"] = "savgol",
    smooth_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity **and** acceleration from a position time series.

    Convenience wrapper that calls :func:`compute_velocity` and
    :func:`compute_acceleration` in sequence.

    Args:
        position: 1-D array of positions (cm).
        dt_ms: Frame interval in milliseconds.
        smooth_method: Smoothing method to use for both passes.
        smooth_kwargs: Additional smoothing parameters.

    Returns:
        ``(velocity, acceleration)`` — both 1-D arrays.
    """
    kwargs = smooth_kwargs or {}
    velocity = compute_velocity(
        position, dt_ms, smooth=True, method=smooth_method, **kwargs,
    )
    acceleration = compute_acceleration(
        velocity, dt_ms, smooth=True, method=smooth_method, **kwargs,
    )
    return velocity, acceleration
