"""
Centralized configuration for the NSMoR data pipeline.

All physical constants, thresholds, and dimensional parameters
are defined here as frozen dataclasses for immutability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Tuple


# ──────────────────────────────────────────────────────────────
# Label definitions
# ──────────────────────────────────────────────────────────────

class Label(IntEnum):
    """Discrete behavioral labels for cricket trials."""
    STARTLE = 0
    WALK = 1
    PRE_ACTIVE = 2
    NO_RESPONSE = 3


# ──────────────────────────────────────────────────────────────
# Time window configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TimeWindowConfig:
    """
    Physical time boundaries for data extraction.

    All times in milliseconds.  Trial Start is the moment of
    the 2-second absolute static baseline onset.

    To support experimental variants (e.g. a 5.7 s silent baseline
    for pure-wind trials), instantiate a new config:
        TimeWindowConfig(baseline_duration_ms=5700.0)
    """
    baseline_duration_ms: float = 2000.0
    """Duration of the static baseline period (Trial Start → Stimulus onset)."""

    ttc_offset_ms: float = -50.0
    """Offset from TTC for snapshot extraction (negative = before TTC)."""

    background_window_ms: float = 200.0
    """Lookback window for background kinematics features."""

    frame_interval_ms: float = 10.0
    """Expected frame interval in ms (100 Hz sampling rate)."""

    min_baseline_duration_ms: float = 500.0
    """Minimum baseline duration required for a valid trial."""


# ──────────────────────────────────────────────────────────────
# Threshold configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThresholdConfig:
    """
    Velocity and latency thresholds for behavioral classification.

    These thresholds define the boundaries between behavioral categories.
    """
    startle_velocity_threshold: float = 5.0
    """Peak velocity (cm/s) above which a response is classified as Startle."""

    walk_velocity_threshold: float = 1.0
    """Sustained velocity (cm/s) above which a response is classified as Walk."""

    pre_active_velocity_threshold: float = 0.5
    """Velocity (cm/s) during baseline that indicates spontaneous activity."""

    startle_latency_max_ms: float = 500.0
    """Maximum time (ms) after stimulus onset for a Startle response."""

    walk_latency_max_ms: float = 2000.0
    """Maximum time (ms) after stimulus onset for a Walk response."""


# ──────────────────────────────────────────────────────────────
# Feature dimension configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureConfig:
    """
    Dimensional constants for feature vectors.

    These define the exact shape of all tensors in the pipeline.
    """
    # --- MCMC snapshot features (5D) ---
    snapshot_visual_angle_dim: int = 1
    snapshot_looming_velocity_dim: int = 1
    snapshot_wind_state_dim: int = 1
    snapshot_avg_velocity_dim: int = 1
    snapshot_max_acceleration_dim: int = 1
    snapshot_dim: int = 5
    """Total snapshot feature dimension:
    [visual_angle, looming_velocity, wind_state, avg_velocity_bg, max_acceleration_bg]."""

    # --- Per-frame physical features (4D) ---
    per_frame_visual_dim: int = 1
    per_frame_wind_dim: int = 1
    per_frame_velocity_dim: int = 1
    per_frame_acceleration_dim: int = 1
    per_frame_physical_dim: int = 4
    """Per-frame physical features:
    [v_vis(t), wind(t), v_kine(t-1), a_kine(t-1)]."""

    # --- MCMC probability vector (4D) ---
    mcmc_dim: int = 4
    """MCMC prior dimension:
    [P_startle, P_walk, P_pre_active, P_no_response]."""

    # --- Total per-frame feature dimension ---
    per_frame_total_dim: int = 8
    """Total per-frame features: physical (4) + MCMC prior (4) = 8."""

    # --- Classification ---
    num_classes: int = 4
    """Number of behavioral classes."""

    label_names: ClassVar[Tuple[str, ...]] = (
        "Startle", "Walk", "Pre_Active", "NoResponse",
    )
    """Human-readable label names in Label enum order."""


# ──────────────────────────────────────────────────────────────
# MCMC training configuration
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MCMCTrainingConfig:
    """Hyperparameters for MCMC prior training."""
    learning_rate: float = 1e-2
    num_epochs: int = 200
    batch_size: int = 32
    convergence_tol: float = 1e-6
    random_seed: int = 42


# ──────────────────────────────────────────────────────────────
# Default singleton instances
# ──────────────────────────────────────────────────────────────

DEFAULT_TIME_WINDOW = TimeWindowConfig()
DEFAULT_THRESHOLD = ThresholdConfig()
DEFAULT_FEATURE = FeatureConfig()
DEFAULT_MCMC_TRAINING = MCMCTrainingConfig()
