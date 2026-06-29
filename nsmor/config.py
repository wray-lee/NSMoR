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
    ESCAPE = 0
    PREWALK = 1
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

    Standardized criteria for cricket escape behavior:
    - **Escape**: Post-stimulus sustained speed > 50 mm/s for 250ms,
      with pre-stimulus speed < 10 mm/s.
    - **Prewalk**: Pre-stimulus speed > 10 mm/s for 1s AND
      post-stimulus sustained speed > 50 mm/s for 250ms.
    - **No Response**: Post-stimulus sustained speed ≤ 50 mm/s for 250ms.
    """
    escape_velocity_threshold: float = 5.0
    """Post-stimulus sustained velocity (cm/s) for escape classification.
    50 mm/s = 5.0 cm/s."""

    escape_sustained_ms: float = 250.0
    """Duration (ms) that velocity must remain above threshold for escape."""

    prewalk_velocity_threshold: float = 1.0
    """Pre-stimulus velocity (cm/s) for prewalk classification.
    10 mm/s = 1.0 cm/s."""

    prewalk_sustained_ms: float = 1000.0
    """Duration (ms) that pre-stimulus velocity must remain above threshold."""

    pre_active_velocity_threshold: float = 0.5
    """Velocity (cm/s) during baseline that indicates spontaneous activity."""


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
    snapshot_dim: int = 5
    """Total snapshot feature dimension:
    [visual_angle, looming_velocity, wind_state, avg_velocity_bg, max_acceleration_bg]."""

    # --- Per-frame physical features (4D) ---
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
        "Escape", "Prewalk", "Pre_Active", "NoResponse",
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
