#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 10 — In-Silico Psychophysics & Bayesian Reliability Analysis.

Injects graded Gaussian noise into the visual input channel and quantifies
the resulting shifts in MoR routing gates and kinematic latencies.

Key outputs:
    results/bayesian_reliability.png  — Dual-panel Lancet/Cell figure
    results/psychophysics_summary.json — Aggregated statistics

Hypothesis:
    Higher visual noise → delayed/suppressed GRU gating, systematic
    latency shift reflecting Bayesian re-weighting of sensory evidence.

Respects all BOUNDARY.md constraints — never modifies frozen core.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Bootstrap: resolve paths, import project modules
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from nsmor.model_utils import load_model_from_checkpoint  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lancet/Cell colour palette (strict Phase 9 aesthetic)
# ---------------------------------------------------------------------------
# Panel A: monochromatic gradient — dark (σ=0) → light (high σ)
GATE_COLOURS = [
    "#1C7ED6",  # σ=0.0   Cell Cobalt Blue (saturated)
    "#4DABF7",  # σ=5.0   lighter
    "#A5D8FF",  # σ=15.0  pastel
    "#D0EBFF",  # σ=30.0  very faded
]
LATENCY_COLOUR = "#C92A2A"  # Lancet Crimson Red
BASELINE_COLOUR = "#495057"  # Strong Slate Gray
AXIS_COLOUR = "#212529"  # Dark charcoal
BG_COLOUR = "#FFFFFF"
LINEWIDTH = 1.5
DPI = 300


# ===================================================================
# Data Loading (reuse logic from analyze_integration.py)
# ===================================================================

def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load model checkpoint with ALL biophysical parameters.

    Delegates to the shared :func:`nsmor.model_utils.load_model_from_checkpoint`
    which guarantees every biophysical parameter is reconstructed from
    the saved config (refractory periods, synaptic delay, STP, lateral
    inhibition, dendritic compartmentalization, neuromodulatory gain,
    sensory noise).  The original local implementation only forwarded
    8 of 21 parameters, silently using defaults for the rest.
    """
    return load_model_from_checkpoint(Path(ckpt_path), device)


def load_validation_data(device: torch.device, max_seq_len: int = 1000):
    """Load nsmor_dataset.pt and return validation split."""
    dataset_path = os.path.join(_PROJECT_ROOT, "data", "processed", "nsmor_dataset.pt")
    if not os.path.exists(dataset_path):
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    X_seqs = data["X_seqs"]
    Y_seqs = data["Y_seqs"]
    lengths = data["lengths"]
    mcmc_priors = data.get("mcmc_priors", None)

    n_total = len(X_seqs)
    n_val = int(n_total * 0.2)
    split = n_total - n_val

    # Use DataLoader with collate_variable_length for proper padding
    from nsmor.nsmor_dataloader import NSMoRDataset, collate_variable_length
    from nsmor.config import DEFAULT_FEATURE

    sequences = [(X_seqs[i], Y_seqs[i], 0) for i in range(split, n_total)]
    feature_config = data.get("feature_config", DEFAULT_FEATURE)
    val_priors = mcmc_priors[split:] if mcmc_priors is not None else None

    val_dataset = NSMoRDataset(
        sequences=sequences,
        mcmc_priors=val_priors if val_priors is not None else np.ones((len(sequences), 4)) * 0.25,
        feature_config=feature_config,
        max_seq_len=max_seq_len,
    )

    # Create a single batch with all validation data
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_length,
    )

    X_val, Y_val, lengths_val = next(iter(val_loader))

    X_val = X_val.to(device).contiguous()
    Y_val = Y_val.to(device).contiguous()
    lengths_val = lengths_val.to(device).contiguous()

    logger.info("Validation data loaded: %d trials", X_val.shape[0])
    return X_val, Y_val, lengths_val


# ===================================================================
# Condition Filtering & Noise Injection
# ===================================================================

STIM_ONSET_FRAME = 200
NOISE_LEVELS = [0.0, 5.0, 15.0, 30.0]  # σ in degrees


def detect_wind_onset_frame(x_seq: torch.Tensor) -> int | None:
    """Return first frame index where wind(t) > 0.5, or None."""
    wind_channel = x_seq[:, 1]
    indices = (wind_channel > 0.5).nonzero(as_tuple=False)
    if indices.numel() == 0:
        return None
    return int(indices[0].item())


def find_multisensory_ttc0(
    X_seqs: torch.Tensor,
    lengths: torch.Tensor,
    raw_dir: str = "data/raw",
) -> torch.Tensor:
    """Return boolean mask for trials matching multisensory_ttc_0ms condition.

    Uses events files to determine trial type and target_ttc_ms.
    """
    import json
    from pathlib import Path

    B, T, _ = X_seqs.shape
    mask = torch.zeros(B, dtype=torch.bool, device=X_seqs.device)

    # Load trial info from events files
    trial_info = []
    events_files = sorted(Path(raw_dir).rglob("*_events.csv"))
    for evt_path in events_files:
        df = pd.read_csv(evt_path)
        for _, row in df.iterrows():
            event_type = str(row.get('event_type', row.get('event_name', '')))
            if event_type == 'trial_start':
                details_str = str(row.get('event_value', row.get('details', '{}')))
                try:
                    details = json.loads(details_str)
                except (json.JSONDecodeError, KeyError, ValueError):
                    details = {}
                trial_info.append({
                    'type': details.get('type', 'unknown'),
                    'target_ttc_ms': details.get('target_ttc_ms'),
                })

    # Use validation split (last 20%)
    n_total = len(trial_info)
    n_val = min(B, int(n_total * 0.2))
    split = n_total - n_val
    val_info = trial_info[split:split + B]

    # Mark trials with target_ttc_ms ≈ 0
    for i, info in enumerate(val_info):
        if i >= B:
            break
        if info['type'] == 'looming_wind' and info['target_ttc_ms'] is not None:
            if abs(info['target_ttc_ms']) < 50:
                mask[i] = True

    return mask


def inject_visual_noise(
    X_batch: torch.Tensor,
    lengths: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Add N(0, σ²) noise to visual channel (X[:,:,0]).

    Noise is applied only to non-padded frames (respects sequence masks).
    Returns a new tensor (does not mutate the original).
    """
    if sigma == 0.0:
        return X_batch.clone()

    X_noisy = X_batch.clone()
    B, T, _ = X_noisy.shape
    for i in range(B):
        L = int(lengths[i].item())
        noise = torch.randn(L, device=X_noisy.device) * sigma
        X_noisy[i, :L, 0] += noise

    return X_noisy


# ===================================================================
# Metric Extraction
# ===================================================================

def extract_gate_trajectory(internals: dict, lengths: torch.Tensor) -> np.ndarray:
    """
    Extract mean g_gru(t) across trials at each time-step.

    Returns: (T,) numpy array of mean gate probabilities.
    """
    g_gru = internals["routing_gates"][:, :, 1]  # (B, T)
    B, T = g_gru.shape
    mask = torch.arange(T, device=g_gru.device).unsqueeze(0) < lengths.unsqueeze(1)
    g_gru_masked = g_gru * mask.float()
    count = mask.float().sum(dim=0).clamp(min=1)
    return (g_gru_masked.sum(dim=0) / count).cpu().numpy()


def extract_latency_to_peak(
    Y_pred: torch.Tensor, lengths: torch.Tensor, dt_ms: float = 10.0
) -> list[float]:
    """
    Per-trial latency to peak velocity (ms) relative to stimulus onset.

    Returns list of latencies (one per trial).
    """
    latencies = []
    B, T = Y_pred.shape
    for i in range(B):
        L = int(lengths[i].item())
        vel = Y_pred[i, :L]
        peak_frame = torch.argmax(vel.abs()).item()
        latency_ms = max(0.0, (peak_frame - STIM_ONSET_FRAME) * dt_ms)
        latencies.append(latency_ms)
    return latencies


def extract_peak_velocity(Y_pred: torch.Tensor, lengths: torch.Tensor) -> list[float]:
    """Per-trial peak absolute velocity."""
    peaks = []
    B, T = Y_pred.shape
    for i in range(B):
        L = int(lengths[i].item())
        vel = Y_pred[i, :L]
        peaks.append(vel.abs().max().item())
    return peaks


# ===================================================================
# Figure Creation
# ===================================================================

def create_figure(
    gate_trajectories: dict[float, np.ndarray],
    latency_stats: dict,
    T: int,
    dt_ms: float,
    output_path: str,
) -> None:
    """
    Dual-panel Lancet/Cell figure.

    Panel A: Gate modulation by noise level (g_gru vs time).
    Panel B: Psychometric curve (latency vs noise level).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG_COLOUR)

    for ax in axes:
        ax.set_facecolor(BG_COLOUR)
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOUR)
        ax.tick_params(colors=AXIS_COLOUR, labelsize=10)
        ax.xaxis.label.set_color(AXIS_COLOUR)
        ax.yaxis.label.set_color(AXIS_COLOUR)
        ax.title.set_color(AXIS_COLOUR)

    # ---- Panel A: Gate trajectories ----
    ax_a = axes[0]
    time_ms = (np.arange(T) - STIM_ONSET_FRAME) * dt_ms

    for idx, sigma in enumerate(NOISE_LEVELS):
        colour = GATE_COLOURS[idx % len(GATE_COLOURS)]
        style = "-" if sigma == 0.0 else "--"
        lw = LINEWIDTH + 0.3 if sigma == 0.0 else LINEWIDTH
        ax_a.plot(
            time_ms,
            gate_trajectories[sigma],
            color=colour,
            linewidth=lw,
            linestyle=style,
            label=f"σ = {sigma:.0f}°",
            alpha=0.95 if sigma == 0.0 else 0.85,
        )

    ax_a.axvline(0, color=BASELINE_COLOUR, linewidth=0.8, linestyle=":", alpha=0.6)
    ax_a.set_xlabel("Time relative to stimulus onset (ms)", fontsize=11)
    ax_a.set_ylabel("MoR Gate Probability  g_gru(t)", fontsize=11)
    ax_a.set_title("A. Gate Modulation by Visual Noise", fontsize=12, fontweight="bold")
    ax_a.legend(fontsize=9, loc="upper left", framealpha=0.85)
    ax_a.set_ylim(-0.05, 1.05)

    # ---- Panel B: Psychometric curve ----
    ax_b = axes[1]
    sigmas = sorted(latency_stats.keys())
    means = [latency_stats[s]["mean"] for s in sigmas]
    sems = [latency_stats[s]["sem"] for s in sigmas]

    ax_b.errorbar(
        sigmas,
        means,
        yerr=sems,
        color=LATENCY_COLOUR,
        marker="o",
        markersize=7,
        markeredgecolor=LATENCY_COLOUR,
        markerfacecolor="white",
        linewidth=LINEWIDTH,
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
    )

    ax_b.set_xlabel("Visual Noise Level σ (degrees)", fontsize=11)
    ax_b.set_ylabel("Mean Latency to Peak Velocity (ms)", fontsize=11)
    ax_b.set_title("B. Psychometric Curve", fontsize=12, fontweight="bold")
    ax_b.set_xlim(-2, max(sigmas) + 5)

    # Annotate N per point
    for s in sigmas:
        n = latency_stats[s]["n"]
        ax_b.annotate(
            f"n={n}",
            xy=(s, latency_stats[s]["mean"]),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=BASELINE_COLOUR,
        )

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", output_path)


# ===================================================================
# Main Pipeline
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 10 — Bayesian Reliability & Optimal Cue Combination"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "runs", "default", "best_model.pth"),
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--noise_levels",
        type=float,
        nargs="+",
        default=NOISE_LEVELS,
        help="Visual noise σ values in degrees.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "results"),
        help="Directory for output figures and JSON.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1000,
        help="Crop sequences longer than this (cuDNN compatibility). 0 = disable.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Load model & data ---
    model = load_checkpoint(args.checkpoint, device)
    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    X_val, Y_val, lengths_val = load_validation_data(device, max_seq_len=max_seq_len)

    # --- Filter to multisensory_ttc_0ms ---
    ttc0_mask = find_multisensory_ttc0(X_val, lengths_val)
    n_ttc0 = ttc0_mask.sum().item()
    logger.info("multisensory_ttc_0ms trials found: %d / %d", n_ttc0, X_val.shape[0])
    if n_ttc0 == 0:
        logger.error("No multisensory_ttc_0ms trials found. Aborting.")
        sys.exit(1)

    X_ttc0 = X_val[ttc0_mask]
    Y_ttc0 = Y_val[ttc0_mask]
    L_ttc0 = lengths_val[ttc0_mask]

    # --- Run noise sweep ---
    gate_trajectories: dict[float, np.ndarray] = {}
    latency_stats: dict = {}
    T = X_ttc0.shape[1]

    for sigma in args.noise_levels:
        logger.info("--- Noise level σ = %.1f° ---", sigma)

        X_noisy = inject_visual_noise(X_ttc0, L_ttc0, sigma)

        with torch.no_grad():
            Y_pred, internals = model(
                X_noisy, L_ttc0, return_internals=True
            )

        # Gate trajectory
        gate_traj = extract_gate_trajectory(internals, L_ttc0)
        gate_trajectories[sigma] = gate_traj
        logger.info(
            "  g_gru mean (post-stim): %.4f",
            gate_traj[STIM_ONSET_FRAME:].mean(),
        )

        # Latency
        latencies = extract_latency_to_peak(Y_pred, L_ttc0)
        mean_lat = float(np.mean(latencies))
        sem_lat = float(np.std(latencies, ddof=1) / np.sqrt(len(latencies))) if len(latencies) > 1 else 0.0
        latency_stats[sigma] = {
            "mean": mean_lat,
            "sem": sem_lat,
            "n": len(latencies),
            "std": float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0,
        }
        logger.info("  Latency: %.1f ± %.1f ms (n=%d)", mean_lat, sem_lat, len(latencies))

        # Peak velocity
        peaks = extract_peak_velocity(Y_pred, L_ttc0)
        logger.info("  Peak Vel: %.2f cm/s", float(np.mean(peaks)))

    # --- Create figure ---
    fig_path = os.path.join(args.output_dir, "bayesian_reliability.png")
    create_figure(gate_trajectories, latency_stats, T, 10.0, fig_path)

    # --- Export JSON summary ---
    summary = {
        "noise_levels": args.noise_levels,
        "n_ttc0_trials": n_ttc0,
        "stim_onset_frame": STIM_ONSET_FRAME,
        "latency_stats": {
            str(k): v for k, v in latency_stats.items()
        },
        "gate_post_stim_mean": {
            str(sigma): float(gate_trajectories[sigma][STIM_ONSET_FRAME:].mean())
            for sigma in args.noise_levels
        },
    }
    json_path = os.path.join(args.output_dir, "psychophysics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("JSON summary saved → %s", json_path)

    logger.info("Done. All outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
