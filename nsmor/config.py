"""
Centralized configuration for the NSMoR data pipeline.

All physical constants, thresholds, and dimensional parameters
are defined here as frozen dataclasses for immutability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar, Tuple


# ──────────────────────────────────────────────────────────────
# Pipeline provenance (Round-2 CRITICAL-A / m-2)
# ──────────────────────────────────────────────────────────────

PIPELINE_SEMANTICS_VERSION: str = "2.3"
"""
Version tag for the CURRENT scientific semantics of the pipeline.

Bumped to ``2.0`` at Reviewer Round 2 (tau units → physical ms;
anchored sustained criteria; out-of-fold session-grouped priors).

Bumped to ``2.1`` at Reviewer Round 3 because the labelling BRANCH
ORDER changed (Reviewer A BLK-3B root-cause fix): classification now
tests the stimulus-locked escape response FIRST and splits responders
into PREWALK vs ESCAPE by pre-stimulus locomotion, reserving
PRE_ACTIVE for non-responders.  The previous pre-active-first order
structurally absorbed every walking animal into PRE_ACTIVE (the
mechanism behind PREWALK=0).  Every label of every trial can change
under this reordering, so all v2.0 datasets and checkpoints are
scientifically invalid for v2.1 code.

Bumped to ``2.2`` because the SET OF RETAINED TRIALS changed.  The MCMC
Snapshot anchor was ``stimulus_onset_ms − 50 ms`` for every condition,
but visual-only trials begin looming at the ``TrialStart -> Looming``
transition, i.e. the same instant as ``trial_start``, so their anchor
landed before the first frame and every one of them was silently
dropped — 36 of 396 on the reference corpus, all No_Response, none
surviving, and dropped from the regression sequence set as well, not
merely from the prior generator's input.  Visual-only trials are now
anchored 50 ms before the looming collision (located by the visual-angle
peak).  A v2.1 dataset is missing those trials entirely, so its
snapshots, priors, sequences, and target statistics are all computed
over a different population.

Every checkpoint written by :func:`nsmor.checkpoint.save_checkpoint`
carries this key; loaders MUST reject artifacts whose version differs
(or is absent) instead of silently reinterpreting them — except for the
explicitly RETAINED versions (below), which load clearly marked as
legacy, never silently as current.

Bumped to ``2.3`` (phase-3 prerequisites, ADR 0011) because the LABEL
CLOCK of visual-only trials changed.  The injected ``stimulus_onset``
of a visual-only trial is now the looming COLLISION
(``looming_onset + lv_ratio_ms / tan(radians(init_deg/2))``), not the
looming onset: under the 2.2 clock no collision-locked visual escape
could ever be labelled (the response window
``[looming_onset, looming_onset + 200 ms)`` closes seconds before the
collision; on the phase-2 corpus 1 stored ESCAPE vs 8 collision-anchored
responders).  The same change makes the multisensory reference event the
FIRST stimulus event of the trial (``min(wind onset, collision)``), a
no-op for the existing wind-first corpus but the correct clock for
wind-lagging paradigms (wind at / after the collision).  New additive
per-trial keys (``stimulus_soa_ms``, ``stimulus_variant``,
``stimulus_reference_event``) record the signed wind->collision SOA and
the reference event.  Every label of a visual-only trial can change
under this re-clock, so v2.2 and v2.3 label arrays are NOT
interchangeable for visual-only trials.

``2.2`` is RETAINED (see :data:`RETAINED_PIPELINE_SEMANTICS_VERSIONS`):
the frozen phase-2 corpus, its checkpoints and its analyses must keep
loading.  Retained artifacts load READ-ONLY with an explicit warning
that stamps them as legacy; every other version is still refused.
"""

RETAINED_PIPELINE_SEMANTICS_VERSIONS: Tuple[str, ...] = ("2.2",)
"""Version stamps that still load, clearly marked as legacy.

A retained version is accepted by the provenance guards with a warning
that names the version and what changed since; it is never silently
treated as current.  Artifacts stamped with any other version (or no
version at all) remain rejected, because pre-2.0 artifacts carry
leaked priors / frame-unit time constants that are silently
misinterpreted at load time.
"""


def pipeline_version_status(found: Any) -> str:
    """Classify an artifact's ``pipeline_semantics_version`` stamp.

    Single source of truth for every provenance gate (checkpoints,
    datasets, nested-prior artifacts, evaluation scripts).

    Args:
        found: The stamp stored in the artifact (may be None or any type).

    Returns:
        ``"current"`` when the stamp equals
        :data:`PIPELINE_SEMANTICS_VERSION`; ``"retained"`` when it is one
        of :data:`RETAINED_PIPELINE_SEMANTICS_VERSIONS` (loadable, marked
        legacy); ``"missing"`` when absent; ``"rejected"`` otherwise.
    """
    if found is None:
        return "missing"
    text = str(found)
    if text == PIPELINE_SEMANTICS_VERSION:
        return "current"
    if text in RETAINED_PIPELINE_SEMANTICS_VERSIONS:
        return "retained"
    return "rejected"


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

    # Round-2 fix (Reviewer B m-3): sustained-criterion parameters were
    # bare magic numbers inside labeling._check_sustained_speed; they
    # are now configuration-owned so the classification semantics are
    # fully declared and sensitivity-analysable.
    sustained_min_fraction: float = 0.5
    """Minimum fraction of frames above threshold within an anchored
    sustained window.  0.5 tolerates brief sub-threshold dips (sensor
    dropouts, stride pauses) while demanding genuine sustained
    locomotion; sensitivity is covered in tests/test_pipeline.py."""

    response_max_latency_ms: float = 200.0
    """Maximum delay between window start and above-threshold response
    initiation.  Insect escape latency is ~50-100 ms (GI→TTM/CoLa);
    200 ms admits the physiological range while excluding late,
    stimulus-unrelated locomotion.  NOTE: this parameter applies ONLY
    to the POST-stimulus escape check — see Round-3 MAJ-3A below."""

    # Round-3 fix (Reviewer A MAJ-3A): the PREWALK pre-stimulus check
    # previously reused ``response_max_latency_ms`` as an anchor search
    # band, which made "pre-stimulus walking" depend on the animal's
    # motion state in exactly the 200 ms before onset (animals that
    # started walking EARLIER were systematically classified as not
    # prewalking — the mechanism behind the PREWALK=0 collapse).  The
    # pre-stimulus criterion is now a plain WINDOW FRACTION over the
    # whole ``[onset - prewalk_sustained_ms, onset)`` interval with its
    # own configuration-owned parameters.
    prewalk_min_fraction: float = 0.5
    """Minimum fraction of frames above the prewalk velocity threshold
    within the entire pre-stimulus window ``[onset-1000, onset)``.
    Window-fraction semantics (not latency anchoring) match the
    behavioural definition of ongoing locomotion."""

    prewalk_min_coverage: float = 0.5
    """Minimum fraction of the nominal pre-stimulus window that must be
    populated by actual frames before the fraction criterion is
    evaluated.  Trials whose recorded baseline covers less than this
    share of the window cannot support a walking/non-walking decision
    and fail the PREWALK check conservatively."""

    sustained_anchor_min_frames: int = 2
    """Minimum number of consecutive above-threshold frames required to
    establish the response-initiation anchor of the POST-stimulus
    sustained-speed check (Round-2 MAJOR-C).  2 frames = 20 ms at
    dt=10 ms: long enough that a single-frame sensor glitch cannot
    define the window, far below any behavioural timescale."""


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
