#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 13 — Autoregressive Closed-Loop Inference.

Generates synthetic cricket trajectories driven EXCLUSIVELY by stimulus
paradigms, using the trained model's own predictions to step forward
in time.  No real kinematics data is used during generation.

Output CSVs mock the hardware logs from the ``cercus`` experimental
setup for downstream evaluation in ``cercus-classical-analysis-cli``.

Outputs
-------
    results/sim_session/events.csv      — trial events
    results/sim_session/kinematics.csv  — per-frame kinematics
    results/sim_session/simulation_manifest.json — synthetic run provenance

The uniform default / CLI prior is synthetic, not fitted to or verified against
an empirical dataset or nested-prior artifact. Fatigue is an uncalibrated
macro-variable assumption. This CLI sets no seed and restores no checkpoint
RNG state; the manifest does not certify reproducibility.

Usage
-----
CLI::

    python scripts/simulate_autoregressive.py --checkpoint runs/default/best_model.pth
    python scripts/simulate_autoregressive.py --checkpoint runs/default/best_model.pth --paradigms visual_only wind_only

Respects all BOUNDARY.md constraints — never modifies frozen core.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Bootstrap: resolve paths, import project modules
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from nsmor.data_extractor import _compute_pure_wind_prepend_frames  # noqa: E402
from nsmor.model_nsmor_core import NSMoRCore  # noqa: E402
from nsmor.analysis.prediction_units import resolve_dt_ms
from nsmor.analysis.prediction_units import load_model_from_checkpoint as _shared_load_model  # noqa: E402
from nsmor.model_utils import _extract_model_params  # noqa: E402
from nsmor.pipeline.nested_prior import (  # noqa: E402
    compute_source_fingerprint,
    load_artifact_bytes,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1.  Stimulus Paradigm Definitions
# ═══════════════════════════════════════════════════════════════

@dataclass
class StimulusParadigm:
    """Specification for a single stimulus condition."""
    name: str
    description: str
    target_ttc_ms: float           # collision offset from baseline reference
    lv_ratio: float                # l/v ratio (object size / speed)
    has_visual: bool               # whether visual looming is present
    has_wind: bool                 # whether wind step is present
    wind_onset_delta_ms: float = 0.0  # ms relative to TTC (negative = early)
    wind_offset_delta_ms: float = 0.0  # 0 = no offset (sustained)
    total_duration_ms: float = 5000.0  # total trial duration
    baseline_ms: float = 2000.0    # collision reference at target_ttc_ms=0
    visual_lead_ms: float = 1000.0  # sampled approach before collision
    wind_gap_deltas_ms: Optional[Tuple[float, float]] = None  # off interval relative to TTC


# ── The 9 experimental paradigms ──────────────────────────────
PARADIGMS: Dict[str, StimulusParadigm] = {
    "visual_only": StimulusParadigm(
        name="visual_only",
        description="Pure visual looming, no wind",
        target_ttc_ms=0.0,
        lv_ratio=120.0,
        has_visual=True,
        has_wind=False,
    ),
    "wind_only": StimulusParadigm(
        name="wind_only",
        description="Pure wind step, no visual looming",
        target_ttc_ms=0.0,
        lv_ratio=120.0,
        has_visual=False,
        has_wind=True,
        wind_onset_delta_ms=0.0,
    ),
    "sync_ttc_0": StimulusParadigm(
        name="sync_ttc_0",
        description="Synchronous wind + visual at TTC",
        target_ttc_ms=0.0,
        lv_ratio=120.0,
        has_visual=True,
        has_wind=True,
        wind_onset_delta_ms=0.0,
    ),
    "early_wind_ttc_neg373": StimulusParadigm(
        name="early_wind_ttc_neg373",
        description="Wind begins 373ms before visual collision",
        target_ttc_ms=0.0,
        lv_ratio=120.0,
        has_visual=True,
        has_wind=True,
        wind_onset_delta_ms=-373.0,
    ),
    "early_wind_ttc_neg119": StimulusParadigm(
        name="early_wind_ttc_neg119",
        description="Wind begins 119ms before visual collision",
        target_ttc_ms=0.0,
        lv_ratio=120.0,
        has_visual=True,
        has_wind=True,
        wind_onset_delta_ms=-119.0,
    ),
    "late_wind_ttc_plus200": StimulusParadigm(
        name="late_wind_ttc_plus200",
        description="Wind begins 200ms after visual collision",
        target_ttc_ms=0.0,
        lv_ratio=120.0,
        has_visual=True,
        has_wind=True,
        wind_onset_delta_ms=200.0,
    ),
    "strong_looming": StimulusParadigm(
        name="strong_looming",
        description="Fast approach (low l/v = 60)",
        target_ttc_ms=0.0,
        lv_ratio=60.0,
        has_visual=True,
        has_wind=False,
    ),
    "weak_looming": StimulusParadigm(
        name="weak_looming",
        description="Slow approach (high l/v = 240)",
        target_ttc_ms=0.0,
        lv_ratio=240.0,
        has_visual=True,
        has_wind=False,
    ),
    "double_pulse": StimulusParadigm(
        name="double_pulse",
        description="Wind double-pulse: on/off/on around TTC",
        target_ttc_ms=0.0,
        lv_ratio=120.0,
        has_visual=True,
        has_wind=True,
        wind_onset_delta_ms=-200.0,
        wind_offset_delta_ms=100.0,  # second pulse ends at TTC+100ms
        wind_gap_deltas_ms=(-100.0, 0.0),  # off between first and second pulses
    ),
}


# ═══════════════════════════════════════════════════════════════
# 2.  Visual Looming Physics (reused from prepare_data.py)
# ═══════════════════════════════════════════════════════════════

def compute_visual_angle(
    t_ms: float,
    stimulus_onset_ms: float,
    ttc_absolute_ms: float,
    lv_ratio: float,
) -> float:
    """
    Compute looming visual angle at time t.

    θ(t) = 2 × arctan(l_v / (TTC - t))

    Args:
        t_ms: Current time (ms, absolute).
        stimulus_onset_ms: When the visual stimulus begins (ms, absolute).
        ttc_absolute_ms: Time-to-collision (ms, absolute).
        lv_ratio: l/v ratio.

    Returns:
        Visual angle in degrees.
    """
    if t_ms < stimulus_onset_ms:
        return 0.0

    ttc_remaining = ttc_absolute_ms - t_ms
    if ttc_remaining < 1e-6:
        return 180.0

    ratio = lv_ratio / ttc_remaining
    theta_rad = 2.0 * np.arctan(ratio)
    return float(np.clip(np.degrees(theta_rad), 0.0, 180.0))


# ═══════════════════════════════════════════════════════════════
# 3.  Stimulus Paradigm Generator
# ═══════════════════════════════════════════════════════════════

def generate_stimulus_paradigm(
    paradigm: StimulusParadigm,
    dt_ms: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize the physical stimulus time-series for a paradigm.

    Args:
        paradigm: The stimulus paradigm specification.
        dt_ms: Frame interval in milliseconds (default 4.0ms = 250Hz).

    Returns:
        ``(time_ms, v_vis, wind)`` where:
        - ``time_ms``: 1-D array of timestamps (ms), relative to trial start.
        - ``v_vis``: 1-D array of visual angle (degrees) at each frame.
        - ``wind``: 1-D array of wind state (0 or 1) at each frame.
    """
    total_frames = int(paradigm.total_duration_ms / dt_ms)
    time_ms = np.arange(total_frames) * dt_ms

    # ── Absolute timing ─────────────────────────────────────
    ttc_absolute_ms = paradigm.baseline_ms + paradigm.target_ttc_ms
    stimulus_onset_ms = (
        ttc_absolute_ms - paradigm.visual_lead_ms
        if paradigm.has_visual else paradigm.baseline_ms
    )

    # ── Visual channel ──────────────────────────────────────
    v_vis = np.zeros(total_frames, dtype=np.float64)
    if paradigm.has_visual:
        for i in range(total_frames):
            v_vis[i] = compute_visual_angle(
                t_ms=time_ms[i],
                stimulus_onset_ms=stimulus_onset_ms,
                ttc_absolute_ms=ttc_absolute_ms,
                lv_ratio=paradigm.lv_ratio,
            )

    # ── Wind channel ────────────────────────────────────────
    wind = np.zeros(total_frames, dtype=np.float64)
    if paradigm.has_wind:
        wind_onset_ms = ttc_absolute_ms + paradigm.wind_onset_delta_ms
        wind_offset_ms = (
            ttc_absolute_ms + paradigm.wind_offset_delta_ms
            if paradigm.wind_offset_delta_ms != 0
            else paradigm.total_duration_ms  # sustained
        )

        wind[(time_ms >= wind_onset_ms) & (time_ms < wind_offset_ms)] = 1.0
        if paradigm.wind_gap_deltas_ms is not None:
            gap_start, gap_end = paradigm.wind_gap_deltas_ms
            if not (wind_onset_ms < ttc_absolute_ms + gap_start
                    < ttc_absolute_ms + gap_end < wind_offset_ms):
                raise ValueError("Wind gap must lie strictly inside the wind interval")
            wind[(time_ms >= ttc_absolute_ms + gap_start)
                 & (time_ms < ttc_absolute_ms + gap_end)] = 0.0

    # ── Pure-wind prepend (5.7s structural alignment) ──
    if paradigm.has_wind and not paradigm.has_visual:
        prepend_frames = _compute_pure_wind_prepend_frames(dt_ms)
        v_vis = np.concatenate([np.zeros(prepend_frames), v_vis])
        wind = np.concatenate([np.zeros(prepend_frames), wind])
        time_ms = np.concatenate([
            np.arange(prepend_frames) * dt_ms - prepend_frames * dt_ms,
            time_ms,
        ])

    return time_ms, v_vis, wind


# ═══════════════════════════════════════════════════════════════
# 4.  Model Loading
# ═══════════════════════════════════════════════════════════════

def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> NSMoRCore:
    """Load trained NSMoRCore from checkpoint.

    Delegates to the shared :func:`nsmor.analysis.prediction_units.load_model_from_checkpoint`
    which guarantees all biophysical parameters are restored.
    """
    model = _shared_load_model(checkpoint_path, device)
    # Bind the saved config to the bytes actually loaded, not a later replacement.
    payload = checkpoint_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != model.analysis_checkpoint_sha256:
        raise ValueError("Checkpoint changed while capturing simulation provenance")
    model.simulation_checkpoint_config = load_artifact_bytes(
        payload, map_location="cpu",
    ).get("config", {})
    return model


# ═══════════════════════════════════════════════════════════════
# 5.  Fatigue Simulation (Macro-Variable Layer)
# ═══════════════════════════════════════════════════════════════

def apply_fatigue_to_priors_leaky(
    base_prior: np.ndarray,
    current_fatigue: float,
) -> np.ndarray:
    """
    Shift MCMC prior probability mass using the leaky-accumulator fatigue level.

    Probability mass is transferred from ``P_startle`` and ``P_walk``
    into ``P_no_response`` proportionally to ``current_fatigue``,
    simulating sensory desensitization.  The 4-D vector is
    renormalised to sum to 1.0.

    Args:
        base_prior: ``(4,)`` base MCMC prior
            ``[P_startle, P_walk, P_pre_active, P_no_response]``.
        current_fatigue: Leaky-accumulator fatigue level in [0, 1].

    Returns:
        ``(4,)`` fatigue-adjusted prior, renormalised to sum to 1.0.
    """
    # The caller may use the empirical feature layout with a synthetic prior.
    assert base_prior.shape == (4,), f"Expected (4,), got {base_prior.shape}"
    fatigued = base_prior.copy()
    shift_startle = base_prior[0] * current_fatigue
    shift_walk = base_prior[1] * current_fatigue

    fatigued[0] -= shift_startle   # P_startle
    fatigued[1] -= shift_walk      # P_walk
    fatigued[3] += shift_startle + shift_walk  # P_no_response

    # Renormalise to ensure strict unit sum
    total = fatigued.sum()
    if total > 0:
        fatigued /= total

    return fatigued


# ═══════════════════════════════════════════════════════════════
# 6.  Autoregressive Inference Engine
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrialResult:
    """Result of a single autoregressive trial."""
    paradigm_name: str
    time_ms: np.ndarray         # (T,)
    position: np.ndarray        # (T,) cumulative displacement (cm)
    velocity: np.ndarray        # (T,) predicted velocity (cm/s)
    acceleration: np.ndarray    # (T,) derived acceleration (cm/s²)
    v_vis: np.ndarray           # (T,) model input: visual angle (degrees)
    wind: np.ndarray            # (T,) wind state (0/1)
    gate_lif: np.ndarray        # (T,) LIF routing gate
    gate_gru: np.ndarray        # (T,) GRU routing gate
    target_ttc_ms: float
    lv_ratio: float
    dt_ms: float = 4.0
    stimulus_onset_ms: float = 2000.0
    collision_ms: float = 2000.0


def run_autoregressive_trial(
    model: NSMoRCore,
    paradigm: StimulusParadigm,
    mcmc_prior: np.ndarray,
    device: torch.device,
    dt_ms: Optional[float] = None,
    current_fatigue: float = 0.0,
    max_fatigue_penalty: float = 0.0,
) -> TrialResult:
    """
    Run a single autoregressive trial with leaky-accumulator fatigue.

    The model's predicted velocity feeds back as the next timestep's
    kinematic velocity input, creating a self-contained generative
    simulation driven purely by the stimulus paradigm.

    When ``current_fatigue > 0``, two macro-variable layers modulate
    the loop without touching the neural dynamics:

    1. **Sensory desensitization** shifts the MCMC prior toward
       ``P_no_response`` proportionally to ``current_fatigue``.
    2. **Soft-gain velocity scaling** multiplies the raw prediction
       by ``(1 - current_fatigue × max_fatigue_penalty)``,
       preserving smooth derivatives (no hard clipping).

    Args:
        model: Trained NSMoRCore model (eval mode).
        paradigm: Stimulus paradigm specification.
        mcmc_prior: ``(4,)`` base MCMC prior vector.
        device: Computation device.
        dt_ms: None inherits saved model.dt_ms; an explicit interval must match.
        current_fatigue: Leaky-accumulator fatigue level in [0, 1].
        max_fatigue_penalty: Maximum velocity reduction fraction [0, 1].

    Returns:
        TrialResult with all per-frame trajectories.
    """
    dt_ms = resolve_dt_ms(model, dt_ms)

    # ── Generate stimulus ────────────────────────────────────
    time_ms, v_vis, wind = generate_stimulus_paradigm(paradigm, dt_ms)
    T = len(time_ms)
    dt_s = dt_ms / 1000.0

    # ── Apply prior modulation (sensory desensitization) ─────
    fatigue_prior = apply_fatigue_to_priors_leaky(
        mcmc_prior, current_fatigue,
    )

    # ── Pre-compute soft-gain multiplier ─────────────────────
    gain = 1.0 - (current_fatigue * max_fatigue_penalty)

    # ── Initialize state ─────────────────────────────────────
    states: Optional[Dict[str, torch.Tensor]] = None
    v_kine_prev = 0.0
    a_kine_prev = 0.0
    position = 0.0

    # ── Storage ──────────────────────────────────────────────
    positions = np.zeros(T, dtype=np.float64)
    velocities = np.zeros(T, dtype=np.float64)
    accelerations = np.zeros(T, dtype=np.float64)
    gates_lif = np.zeros(T, dtype=np.float64)
    gates_gru = np.zeros(T, dtype=np.float64)

    mcmc_tensor = torch.tensor(fatigue_prior, dtype=torch.float32, device=device)

    # ── Autoregressive loop ──────────────────────────────────
    with torch.no_grad():
        for t in range(T):
            # Construct input tensor X_t: (1, 1, 8)
            sensory = torch.tensor(
                [[[v_vis[t], wind[t], v_kine_prev, a_kine_prev]]],
                dtype=torch.float32,
                device=device,
            )                                               # (1, 1, 4)
            X_t = torch.cat(
                [sensory, mcmc_tensor.unsqueeze(0).unsqueeze(0)],
                dim=-1,
            )                                               # (1, 1, 8)

            lengths_t = torch.tensor([1], dtype=torch.int64, device=device)

            # Forward pass with state tracking
            if states is None:
                # First call: model initializes recurrent states from zeros
                y_pred, internals, states = model(
                    X_t, lengths_t, return_internals=True, states={},
                )
            else:
                # Subsequent calls: pass states for temporal continuity
                # states_out from the model contains ALL recurrent state
                # components: lif_v, lif_i_syn, lif_refract, lif_w_adapt,
                # gru_h, and (when STP enabled) lif_x_resource, lif_u_facil.
                # We propagate states_out directly to avoid discarding any
                # component (Critical Flaw 1 fix).
                y_pred, internals, states = model(
                    X_t, lengths_t, return_internals=True, states=states,
                )

            # The analysis loader restores cm/s before gain, displacement and feedback.
            assert y_pred.shape == (1, 1), f'Per-step velocity shape: {y_pred.shape}'
            v_adjusted = y_pred.item() * gain

            # Derive acceleration from the adjusted velocity
            acceleration = (v_adjusted - v_kine_prev) / dt_s

            # Accumulate displacement
            position += v_adjusted * dt_s

            # Extract routing gates
            gate_vals = internals["routing_gates"][0, 0, :]  # (2,)
            g_lif_val = gate_vals[0].item()
            g_gru_val = gate_vals[1].item()

            # Store
            positions[t] = position
            velocities[t] = v_adjusted
            accelerations[t] = acceleration
            gates_lif[t] = g_lif_val
            gates_gru[t] = g_gru_val

            # Update feedback for next timestep
            v_kine_prev = v_adjusted
            a_kine_prev = acceleration

    return TrialResult(
        paradigm_name=paradigm.name,
        time_ms=time_ms,
        position=positions,
        velocity=velocities,
        acceleration=accelerations,
        v_vis=v_vis,
        wind=wind,
        gate_lif=gates_lif,
        gate_gru=gates_gru,
        target_ttc_ms=paradigm.target_ttc_ms,
        lv_ratio=paradigm.lv_ratio,
        dt_ms=dt_ms,
        stimulus_onset_ms=(paradigm.baseline_ms + paradigm.target_ttc_ms
                           - paradigm.visual_lead_ms if paradigm.has_visual
                           else paradigm.baseline_ms),
        collision_ms=paradigm.baseline_ms + paradigm.target_ttc_ms,
    )


# ═══════════════════════════════════════════════════════════════
# 6.  Hardware-Identical CSV Export
# ═══════════════════════════════════════════════════════════════

def export_events_csv(
    trials: List[TrialResult],
    output_path: Path,
    session_num: int = 0,
) -> None:
    """
    Export trial events in ``cercus``-compatible format.

    Columns: event_name, timestamp, session_num, trial_in_session,
             global_trial_id, details

    Args:
        trials: List of TrialResult objects.
        output_path: Path to write events.csv.
        session_num: Session number for this virtual session.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "event_name", "timestamp", "session_num",
        "trial_in_session", "global_trial_id", "details",
    ]

    rows = []
    for trial_idx, trial in enumerate(trials):
        details_base = json.dumps({
            "target_ttc_ms": trial.target_ttc_ms,
            "type": trial.paradigm_name,
            "lv_ratio": trial.lv_ratio,
        })

        details_collision = json.dumps({
            "target_ttc_ms": trial.target_ttc_ms,
            "phase": "Collision_TTC0",
        })

        T = len(trial.time_ms)
        stimulus_onset_ms = trial.stimulus_onset_ms
        ttc_absolute_ms = trial.collision_ms

        # trial_start
        rows.append({
            "event_name": "trial_start",
            "timestamp": f"{trial.time_ms[0]:.1f}",
            "session_num": str(session_num),
            "trial_in_session": str(trial_idx),
            "global_trial_id": str(trial_idx),
            "details": details_base,
        })

        # stimulus_onset
        rows.append({
            "event_name": "stimulus_onset",
            "timestamp": f"{stimulus_onset_ms:.1f}",
            "session_num": str(session_num),
            "trial_in_session": str(trial_idx),
            "global_trial_id": str(trial_idx),
            "details": details_base,
        })

        # phase_transition (TTC)
        rows.append({
            "event_name": "phase_transition",
            "timestamp": f"{ttc_absolute_ms:.1f}",
            "session_num": str(session_num),
            "trial_in_session": str(trial_idx),
            "global_trial_id": str(trial_idx),
            "details": details_collision,
        })

        # trial_stop
        rows.append({
            "event_name": "trial_stop",
            "timestamp": f"{trial.time_ms[-1]:.1f}",
            "session_num": str(session_num),
            "trial_in_session": str(trial_idx),
            "global_trial_id": str(trial_idx),
            "details": details_base,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported events.csv: %d events → %s", len(rows), output_path)


def export_kinematics_csv(
    trials: List[TrialResult],
    output_path: Path,
    session_num: int = 0,
    dt_ms: Optional[float] = None,
) -> None:
    """
    Export per-frame kinematics in ``cercus``-compatible format.

    Columns: sys_time, dx, dy, dz, stim_state, global_trial_id

    Args:
        trials: List of TrialResult objects.
        output_path: Path to write kinematics.csv.
        session_num: Session number for this virtual session.
        dt_ms: Frame interval in milliseconds. If None, derived from
            trial.dt_ms or trial.time_ms step (fallback 4.0).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["sys_time", "dx", "dy", "dz", "stim_state", "global_trial_id"]

    rows = []
    for trial_idx, trial in enumerate(trials):
        T = len(trial.time_ms)
        effective_dt = (
            dt_ms
            if dt_ms is not None
            else getattr(
                trial,
                "dt_ms",
                (trial.time_ms[1] - trial.time_ms[0]) if T > 1 else 4.0,
            )
        )
        for t in range(T):
            # stim_state: 1 if either visual or wind is active
            stim_active = 1 if (trial.v_vis[t] > 0 or trial.wind[t] > 0) else 0

            rows.append({
                "sys_time": f"{trial.time_ms[t]:.1f}",
                "dx": f"{trial.velocity[t] * (effective_dt / 1000.0):.6f}",  # displacement per frame
                "dy": "0.0",
                "dz": "0.0",
                "stim_state": str(stim_active),
                "global_trial_id": str(trial_idx),
            })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Exported kinematics.csv: %d frames → %s",
        len(rows), output_path,
    )


def export_stimulus_evidence(trials: List[TrialResult], output_dir: Path) -> None:
    """Export actual model inputs and the timing needed to replay their physics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stimuli.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sys_time", "global_trial_id", "visual_angle_deg", "v_vis", "wind"))
        for trial_id, trial in enumerate(trials):
            assert trial.time_ms.shape == trial.v_vis.shape == trial.wind.shape
            for time, angle, wind_state in zip(trial.time_ms, trial.v_vis, trial.wind):
                writer.writerow((format(time, ".17g"), trial_id, format(angle, ".17g"),
                                 format(angle, ".17g"), format(wind_state, ".17g")))

    summaries = []
    for trial_id, trial in enumerate(trials):
        active = np.flatnonzero(trial.wind > 0)
        intervals = np.split(active, np.where(np.diff(active) > 1)[0] + 1) if active.size else []
        summaries.append({
            "global_trial_id": trial_id,
            "type": trial.paradigm_name,
            "lv_ratio": trial.lv_ratio,
            "visual_onset_ms": trial.stimulus_onset_ms if np.any(trial.v_vis) else None,
            "collision_ms": trial.collision_ms,
            "wind_intervals_ms": [[float(trial.time_ms[segment[0]]),
                                   float(trial.time_ms[segment[-1]] + trial.dt_ms)]
                                  for segment in intervals],
            "visual_gain": 1.0,
            "visual_tau_ms": None,
            "visual_input_semantics": "angle_deg",
        })
    with (output_dir / "stimulus_summary.json").open("w", encoding="utf-8") as stream:
        json.dump({"schema_version": 1, "dt_ms": trials[0].dt_ms,
                   "trials": summaries}, stream, indent=2, allow_nan=False)


def export_simulation_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    model: NSMoRCore,
    base_prior: np.ndarray,
    trial_provenance: List[Dict[str, object]],
) -> None:
    """Bind exported bytes to synthetic assumptions, not empirical validation."""
    assert base_prior.shape == (4,), f"Expected (4,), got {base_prior.shape}"
    assert len(trial_provenance) == len(args.paradigms)
    config = model.simulation_checkpoint_config
    manifest = {
        "schema_version": 1,
        "scope": "synthetic_autoregressive",
        "empirical_dataset_bound": False,
        "nested_prior_artifact_bound": False,
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": model.analysis_checkpoint_sha256,
        },
        "checkpoint_config": config,
        "resolved_model_config": _extract_model_params(config.get("model", {})),
        "training_config": config.get("training"),
        "training_config_status": "saved_checkpoint_only; no defaults inferred",
        "dt_ms": args.dt_ms,
        "device": str(next(model.parameters()).device),
        "model_mode": "eval",
        "prediction_units": {
            "units": "cm/s", "target_mean": model.target_mean,
            "target_std": model.target_std,
            "target_clip_cm_s": model.target_clip_cm_s,
            "inference_clipped": False,
        },
        "synthetic_prior": {
            "source": "uniform_default" if args.mcmc_prior is None else "cli_vector",
            "base_vector": base_prior.tolist(),
            "class_order": ["P_startle", "P_walk", "P_pre_active", "P_no_response"],
            "model_input_dtype": "float32",
        },
        "paradigm_order": args.paradigms,
        "paradigm_specs": [asdict(PARADIGMS[name]) for name in args.paradigms],
        "randomness": {
            "cli_seed": None,
            "seed_status": "not_set_by_cli",
            "checkpoint_rng_restored": False,
            "rng_state_captured": False,
            "entropy_status": "uncontrolled_process_rng; no replay guarantee",
            "reproducibility_guaranteed": False,
        },
        "fatigue": {
            "initial_level": 0.0, "cap": 1.0,
            "trial_cost": args.trial_cost,
            "recovery_rate_per_second": args.recovery_rate,
            "iti_seconds": args.iti_seconds,
            "max_fatigue_penalty": args.max_fatigue_penalty,
            "update_equation": (
                "min(previous * exp(-recovery_rate_per_second * iti_seconds) "
                "+ trial_cost, cap)"
            ),
            "update_timing": "before_every_trial_including_first",
            "within_trial_level": "constant",
            "neural_state_between_trials": "reset",
            "rest_semantics": "scalar recovery only; no neural rollout during ITI",
            "prior_modulation": (
                "transfer fatigue fraction of P_startle/P_walk to P_no_response; "
                "renormalize by sum when positive"
            ),
            "velocity_modulation": (
                "physical velocity * (1 - fatigue * max_fatigue_penalty); "
                "adjusted velocity/acceleration feed back into model"
            ),
            "empirically_calibrated": False,
        },
        "session_num": args.session_num,
        "trials": trial_provenance,
        "outputs_sha256": {
            name: compute_source_fingerprint(output_dir / name) for name in (
                "events.csv", "kinematics.csv", "stimuli.csv", "stimulus_summary.json",
            )
        },
    }
    with (output_dir / "simulation_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)


# ═══════════════════════════════════════════════════════════════
# 7.  Summary Statistics
# ═══════════════════════════════════════════════════════════════

def log_trial_summary(
    trials: List[TrialResult],
    dt_ms: Optional[float] = None,
) -> None:
    """Log summary statistics for all trials.

    Args:
        trials: List of TrialResult objects.
        dt_ms: Kept for caller compatibility; recorded timestamps define latency.
    """
    logger.info("=" * 70)
    logger.info("Autoregressive Generation Summary")
    logger.info("=" * 70)
    logger.info("%-30s %10s %12s %12s", "Paradigm", "Frames", "V_peak", "Latency")
    logger.info("-" * 70)

    for trial in trials:
        T = len(trial.time_ms)
        stim_onset_frame = int(np.searchsorted(trial.time_ms, trial.stimulus_onset_ms))

        if stim_onset_frame < T:
            post_stim = trial.velocity[stim_onset_frame:]
            v_peak = float(np.max(np.abs(post_stim)))
            peak_frame = int(np.argmax(np.abs(post_stim)))
            latency_ms = float(trial.time_ms[stim_onset_frame + peak_frame] - trial.stimulus_onset_ms)
        else:
            v_peak = 0.0
            latency_ms = 0.0

        logger.info(
            "%-30s %10d %12.3f %12.1f",
            trial.paradigm_name, T, v_peak, latency_ms,
        )

    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════
# 8.  Main Entry Point
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 13 — Autoregressive Closed-Loop Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "runs", "default", "best_model.pth"),
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--paradigms",
        type=str,
        nargs="+",
        default=list(PARADIGMS.keys()),
        help="Paradigm names to generate (default: all 9).",
    )
    parser.add_argument(
        "--mcmc_prior",
        type=float,
        nargs=4,
        default=None,
        help=("Synthetic prior [P_startle, P_walk, P_pre_active, P_no_response] "
              "(default: uniform 0.25 each; no empirical/nested prior binding)."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "results", "sim_session"),
        help="Output directory for CSVs.",
    )
    parser.add_argument(
        "--dt_ms",
        type=float,
        default=None,
        help="Frame interval in ms (default: saved model.dt_ms; explicit value must match).",
    )
    parser.add_argument(
        "--session_num",
        type=int,
        default=0,
        help="Session number for CSV metadata.",
    )
    parser.add_argument(
        "--trial_cost",
        type=float,
        default=0.15,
        help="Fatigue gained per trial (leaky accumulator increment).",
    )
    parser.add_argument(
        "--recovery_rate",
        type=float,
        default=0.005,
        help="Exponential recovery rate per second of inter-trial rest.",
    )
    parser.add_argument(
        "--max_fatigue_penalty",
        type=float,
        default=0.6,
        help="Maximum velocity reduction fraction (e.g. 0.6 = up to 60%% slower).",
    )
    parser.add_argument(
        "--iti_seconds",
        type=float,
        default=120.0,
        help="Simulated inter-trial interval in seconds (baseline).",
    )
    args = parser.parse_args()

    # ── Validate paradigm names ──────────────────────────────
    for name in args.paradigms:
        if name not in PARADIGMS:
            logger.error("Unknown paradigm '%s'. Available: %s",
                         name, list(PARADIGMS.keys()))
            sys.exit(1)

    # ── Device ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load model ───────────────────────────────────────────
    model = load_model_from_checkpoint(Path(args.checkpoint), device)
    args.dt_ms = resolve_dt_ms(model, args.dt_ms)

    # ── Synthetic prior (legacy CLI name retained) ────────────
    mcmc_prior = np.array(
        [0.25] * 4 if args.mcmc_prior is None else args.mcmc_prior,
        dtype=np.float64,
    )
    logger.info("Synthetic prior (not empirically bound): %s", mcmc_prior)
    logger.info("No CLI seed set; checkpoint RNG is not restored.")

    # ── Log fatigue configuration ─────────────────────────────
    if args.trial_cost > 0.0:
        logger.info(
            "Fatigue enabled: trial_cost=%.3f, recovery_rate=%.4f /s, "
            "max_penalty=%.1f%%, ITI=%.0fs",
            args.trial_cost, args.recovery_rate,
            args.max_fatigue_penalty * 100, args.iti_seconds,
        )
    else:
        logger.info("Fatigue disabled (trial_cost=0).")

    # ── Run generation ───────────────────────────────────────
    logger.info("Generating %d paradigms...", len(args.paradigms))
    trials: List[TrialResult] = []
    trial_provenance: List[Dict[str, object]] = []

    # ── Leaky-accumulator state ──────────────────────────────
    current_fatigue: float = 0.0

    for global_trial_id, paradigm_name in enumerate(args.paradigms):
        paradigm = PARADIGMS[paradigm_name]

        # ── Update leaky accumulator before each trial ───────
        # Recovery from inter-trial rest, then accrue trial cost
        current_fatigue = current_fatigue * np.exp(
            -args.recovery_rate * args.iti_seconds
        ) + args.trial_cost
        current_fatigue = min(current_fatigue, 1.0)

        logger.info(
            "  [trial %d | %s] fatigue=%.3f — %s",
            global_trial_id, paradigm_name, current_fatigue,
            paradigm.description,
        )

        trial = run_autoregressive_trial(
            model=model,
            paradigm=paradigm,
            mcmc_prior=mcmc_prior,
            device=device,
            dt_ms=args.dt_ms,
            current_fatigue=current_fatigue,
            max_fatigue_penalty=args.max_fatigue_penalty,
        )
        trials.append(trial)
        trial_provenance.append({
            "global_trial_id": global_trial_id,
            "paradigm": paradigm_name,
            "fatigue_level": float(current_fatigue),
            "effective_prior": apply_fatigue_to_priors_leaky(
                mcmc_prior, current_fatigue,
            ).tolist(),
            "velocity_gain": float(1.0 - current_fatigue * args.max_fatigue_penalty),
        })

    # ── Summary ──────────────────────────────────────────────
    log_trial_summary(trials, dt_ms=args.dt_ms)

    # ── Export CSVs ──────────────────────────────────────────
    output_dir = Path(args.output_dir)
    export_events_csv(trials, output_dir / "events.csv", args.session_num)
    export_kinematics_csv(trials, output_dir / "kinematics.csv", args.session_num, dt_ms=args.dt_ms)
    export_stimulus_evidence(trials, output_dir)
    export_simulation_manifest(output_dir, args, model, mcmc_prior, trial_provenance)

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
