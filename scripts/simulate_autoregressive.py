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
import json
import logging
import os
import sys
from dataclasses import dataclass, field
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

from nsmor.model_nsmor_core import NSMoRCore  # noqa: E402

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
    target_ttc_ms: float           # ms relative to stimulus onset
    lv_ratio: float                # l/v ratio (object size / speed)
    has_visual: bool               # whether visual looming is present
    has_wind: bool                 # whether wind step is present
    wind_onset_delta_ms: float = 0.0  # ms relative to TTC (negative = early)
    wind_offset_delta_ms: float = 0.0  # 0 = no offset (sustained)
    total_duration_ms: float = 5000.0  # total trial duration
    baseline_ms: float = 2000.0    # pre-stimulus baseline


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
        description="Wind leads visual by 373ms",
        target_ttc_ms=0.0,
        lv_ratio=120.0,
        has_visual=True,
        has_wind=True,
        wind_onset_delta_ms=-373.0,
    ),
    "early_wind_ttc_neg119": StimulusParadigm(
        name="early_wind_ttc_neg119",
        description="Wind leads visual by 119ms",
        target_ttc_ms=0.0,
        lv_ratio=120.0,
        has_visual=True,
        has_wind=True,
        wind_onset_delta_ms=-119.0,
    ),
    "late_wind_ttc_plus200": StimulusParadigm(
        name="late_wind_ttc_plus200",
        description="Visual leads wind by 200ms",
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
        wind_offset_delta_ms=100.0,  # wind off at TTC+100ms
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
    dt_ms: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize the physical stimulus time-series for a paradigm.

    Args:
        paradigm: The stimulus paradigm specification.
        dt_ms: Frame interval in milliseconds (default 10ms = 100Hz).

    Returns:
        ``(time_ms, v_vis, wind)`` where:
        - ``time_ms``: 1-D array of timestamps (ms), relative to trial start.
        - ``v_vis``: 1-D array of visual angle (degrees) at each frame.
        - ``wind``: 1-D array of wind state (0 or 1) at each frame.
    """
    total_frames = int(paradigm.total_duration_ms / dt_ms)
    time_ms = np.arange(total_frames) * dt_ms

    # ── Absolute timing ─────────────────────────────────────
    stimulus_onset_ms = paradigm.baseline_ms
    ttc_absolute_ms = stimulus_onset_ms + paradigm.target_ttc_ms

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

        for i in range(total_frames):
            if wind_onset_ms <= time_ms[i] < wind_offset_ms:
                wind[i] = 1.0

    # ── Pure-wind prepend (570 frames = 5.7s structural alignment) ──
    if paradigm.has_wind and not paradigm.has_visual:
        prepend_frames = 570
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
    """Load trained NSMoRCore from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dict = checkpoint.get("config", {})
    model_config = config_dict.get("model", {})

    model = NSMoRCore(
        sensory_dim=model_config.get("sensory_dim", 4),
        mcmc_dim=model_config.get("mcmc_dim", 4),
        hidden_dim=model_config.get("hidden_dim", 64),
        num_gru_layers=model_config.get("num_gru_layers", 1),
        dropout=model_config.get("dropout", 0.1),
        lif_alpha=model_config.get("lif_alpha", 0.9),
        lif_threshold=model_config.get("lif_threshold", 1.0),
        lif_beta=model_config.get("lif_beta", 0.5),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %s parameters, hidden_dim=%d",
                f"{param_count:,}", model.hidden_dim)
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
    v_vis: np.ndarray           # (T,) visual angle (degrees)
    wind: np.ndarray            # (T,) wind state (0/1)
    gate_lif: np.ndarray        # (T,) LIF routing gate
    gate_gru: np.ndarray        # (T,) GRU routing gate
    target_ttc_ms: float
    lv_ratio: float


def run_autoregressive_trial(
    model: NSMoRCore,
    paradigm: StimulusParadigm,
    mcmc_prior: np.ndarray,
    device: torch.device,
    dt_ms: float = 10.0,
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
        dt_ms: Frame interval in milliseconds.
        current_fatigue: Leaky-accumulator fatigue level in [0, 1].
        max_fatigue_penalty: Maximum velocity reduction fraction [0, 1].

    Returns:
        TrialResult with all per-frame trajectories.
    """
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
                y_pred, internals = model(
                    X_t, lengths_t, return_internals=True,
                )
            else:
                # Subsequent calls: pass states for temporal continuity
                y_pred, internals, states = model(
                    X_t, lengths_t, return_internals=True, states=states,
                )

            # Build states from internals for next step
            states = {
                "lif_v": internals["lif_potentials"][:, -1, :].contiguous(),
                "gru_h": internals["gru_hidden"][:, -1:, :].permute(1, 0, 2).contiguous(),
            }

            # Soft-gain velocity scaling (preserves derivative continuity)
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
        stimulus_onset_ms = 2000.0  # 2s baseline
        ttc_absolute_ms = stimulus_onset_ms + trial.target_ttc_ms

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
) -> None:
    """
    Export per-frame kinematics in ``cercus``-compatible format.

    Columns: sys_time, dx, dy, dz, stim_state, global_trial_id

    Args:
        trials: List of TrialResult objects.
        output_path: Path to write kinematics.csv.
        session_num: Session number for this virtual session.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["sys_time", "dx", "dy", "dz", "stim_state", "global_trial_id"]

    rows = []
    for trial_idx, trial in enumerate(trials):
        T = len(trial.time_ms)
        for t in range(T):
            # stim_state: 1 if either visual or wind is active
            stim_active = 1 if (trial.v_vis[t] > 0 or trial.wind[t] > 0) else 0

            rows.append({
                "sys_time": f"{trial.time_ms[t]:.1f}",
                "dx": f"{trial.velocity[t] * (10.0 / 1000.0):.6f}",  # displacement per frame
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


# ═══════════════════════════════════════════════════════════════
# 7.  Summary Statistics
# ═══════════════════════════════════════════════════════════════

def log_trial_summary(trials: List[TrialResult]) -> None:
    """Log summary statistics for all trials."""
    logger.info("=" * 70)
    logger.info("Autoregressive Generation Summary")
    logger.info("=" * 70)
    logger.info("%-30s %10s %12s %12s", "Paradigm", "Frames", "V_peak", "Latency")
    logger.info("-" * 70)

    for trial in trials:
        T = len(trial.time_ms)
        stim_onset_frame = int(2000.0 / 10.0)  # 2s baseline at 10ms

        if stim_onset_frame < T:
            post_stim = trial.velocity[stim_onset_frame:]
            v_peak = float(np.max(np.abs(post_stim)))
            peak_frame = int(np.argmax(np.abs(post_stim)))
            latency_ms = float(peak_frame * 10.0)
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
        default=[0.25, 0.25, 0.25, 0.25],
        help="MCMC prior [P_startle, P_walk, P_pre_active, P_no_response].",
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
        default=10.0,
        help="Frame interval in milliseconds.",
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

    # ── MCMC prior ───────────────────────────────────────────
    mcmc_prior = np.array(args.mcmc_prior, dtype=np.float64)
    logger.info("MCMC prior: %s", mcmc_prior)

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

    # ── Summary ──────────────────────────────────────────────
    log_trial_summary(trials)

    # ── Export CSVs ──────────────────────────────────────────
    output_dir = Path(args.output_dir)
    export_events_csv(trials, output_dir / "events.csv", args.session_num)
    export_kinematics_csv(trials, output_dir / "kinematics.csv", args.session_num)

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
