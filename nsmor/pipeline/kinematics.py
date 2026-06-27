"""
Kinematics processing: smoothing, velocity / acceleration computation.

Provides Savitzky-Golay and Gaussian kernel smoothing for raw position data,
with utilities for computing derived kinematic quantities.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


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
