#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 10 — In-Silico Psychophysics: Routing-Gate Noise Sensitivity.

Injects graded Gaussian noise into the visual input channel and quantifies
the resulting shifts in MoR routing gates and kinematic latencies.

Key outputs:
    results/bayesian_reliability.png  — Dual-panel Lancet/Cell figure
    results/psychophysics_summary.json — Aggregated statistics

Round-1 fix (Reviewer A MAJOR-2): the MCMC prior columns
(X[:, :, 4:7]) are held FIXED across noise levels — only the visual
channel degrades.  The router's response therefore demonstrates
*input-noise sensitivity of the routing gate*, NOT Bayesian cue
re-weighting (which would require the prior to degrade with evidence
reliability).  Claims are scoped accordingly.

Noise definition (SNR): sigma is in DEGREES of visual angle and is
injected ADDITIVELY onto the raw visual-angle channel before sensory
encoding.  The per-trial signal scale is the peak |visual_angle|
excursion; SNR_dB = 20*log10(peak|angle| / sigma) is reported per
condition so the physical meaning of each sigma is explicit.

Hypothesis (scoped):
    Higher visual noise → delayed/suppressed GRU gating and systematic
    latency increase, consistent with reliability-dependent routing.
    This is NECESSARY but not SUFFICIENT evidence for optimal cue
    combination; a causal test would additionally require perturbing
    prior reliability.

Respects all BOUNDARY.md constraints — never modifies frozen core.
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional
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


def load_validation_data(
    device: torch.device,
    max_seq_len: int = 1000,
    dataset_path: Optional[str] = None,
):
    """
    Load ``nsmor_dataset.pt`` and return the validation split.

    ``dataset_path`` defaults to the in-repo processed dataset so existing
    callers keep working, but the pipeline passes the dataset the current
    run produced — otherwise this stage silently scores whatever file is
    left over in ``data/processed`` from an earlier run.

    NOTE (open provenance gap): the split here is the trailing 20% by
    position, which is NOT the session-grouped split train.py uses.  Fix
    that with the trial-key work, not here.
    """
    if dataset_path is None:
        dataset_path = os.path.join(
            _PROJECT_ROOT, "data", "processed", "nsmor_dataset.pt"
        )
    if not os.path.exists(dataset_path):
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    # Round-2 CRITICAL-A: refuse pre-2.0 datasets (leaked priors)
    from nsmor.model_utils import validate_dataset_provenance
    validate_dataset_provenance(data, Path(dataset_path))
    X_seqs = data["X_seqs"]
    Y_seqs = data["Y_seqs"]
    lengths = data["lengths"]
    mcmc_priors = data.get("mcmc_priors", None)

    n_total = len(X_seqs)
    n_val = int(n_total * 0.2)
    split = n_total - n_val

    # Use DataLoader with collate_variable_length for proper padding
    from nsmor.nsmor_dataloader import NSMoRDataset
    from nsmor.dataloader_factory import create_optimized_dataloader
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
    val_loader = create_optimized_dataloader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=-1,  # Auto-scale based on dataset size
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


# Visual-angle channel index in the feature dimension (X[:, :, 0]).
VISUAL_ANGLE_IDX = 0


def inject_visual_noise(
    X_batch: torch.Tensor,
    lengths: torch.Tensor,
    sigma: float,
    seed: int | None = None,
    trial_seed_offset: int = 0,
) -> torch.Tensor:
    """
    Add N(0, σ²) noise to visual channel (X[:,:,0]).

    Noise is applied only to non-padded frames (respects sequence masks).
    Returns a new tensor (does not mutate the original).

    Round-1 note (Reviewer A MAJOR-2): only this channel is degraded;
    the MCMC prior columns are held fixed, so the experiment measures
    gate sensitivity to evidence noise, not cue re-weighting.

    Round-2 fix (Reviewer A MAJOR-D-3 / Reviewer B M-1a): noise is drawn
    from an explicit ``torch.Generator`` — bare ``torch.randn`` left
    results irreproducible.

    Round-3 fix (Reviewer B MAJOR-2): *trial_seed_offset* gives each
    σ level an independent, reproducible noise realisation.  The
    previous design reused ONE seed across all σ levels, so the SAME
    standard-normal draw was scaled by different σ — i.e. every
    condition saw the identical noise realisation, not independent
    ones.  That is not a paired design over independent noisy
    presentations; it is one noise field repeatedly rescaled, and the
    paired tests' error terms did not contain the between-condition
    noise variance they claim to test.  With per-sigma offsets, each
    trial keeps a fixed index across conditions (the pairing unit is
    still the stimulus trial) while the noise realisations differ.
    """
    if sigma == 0.0:
        return X_batch.clone()

    gen = torch.Generator(device=X_batch.device)
    if seed is None:
        gen.seed()
    else:
        # Large coprime stride decorrelates the per-sigma streams.
        gen.manual_seed(int(seed) + 1_000_003 * int(trial_seed_offset))

    X_noisy = X_batch.clone()
    for i in range(X_noisy.shape[0]):
        L = int(lengths[i].item())
        noise = torch.randn(L, device=X_noisy.device, generator=gen) * sigma
        X_noisy[i, :L, VISUAL_ANGLE_IDX] += noise

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

    Trials whose global peak falls at or before stimulus onset (i.e.
    peak occurred during the baseline epoch) are reported as NaN and
    must be excluded from inferential statistics — clamping them to 0
    (previous behaviour) artificially compressed the variance of the
    clean-condition sample and biased the psychometric curve upward.
    """
    latencies = []
    B, T = Y_pred.shape
    for i in range(B):
        L = int(lengths[i].item())
        vel = Y_pred[i, :L]
        peak_frame = torch.argmax(vel.abs()).item()
        if peak_frame <= STIM_ONSET_FRAME:
            # Pre-stimulus peak: no post-stimulus response measurable.
            latencies.append(float("nan"))
            continue
        latency_ms = (peak_frame - STIM_ONSET_FRAME) * dt_ms
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
        description="Phase 10 — In-Silico Psychophysics: "
                    "Routing-Gate Noise Sensitivity "
                    "(priors held fixed; NOT a cue-combination test)"
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
        "--dataset",
        type=str,
        default=None,
        help="Path to the processed dataset (default: "
             "<repo>/data/processed/nsmor_dataset.pt).",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "data", "raw"),
        help="Raw session directory whose events CSVs identify the "
             "multisensory_ttc_0ms condition.",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the visual-noise generator (paired design must be "
             "reproducible; recorded in the JSON summary).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Load model & data ---
    model = load_checkpoint(args.checkpoint, device)
    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    X_val, Y_val, lengths_val = load_validation_data(
        device, max_seq_len=max_seq_len, dataset_path=args.dataset,
    )

    # --- Filter to multisensory_ttc_0ms ---
    ttc0_mask = find_multisensory_ttc0(X_val, lengths_val, raw_dir=args.raw_dir)
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
    latency_arrays: dict[float, np.ndarray] = {}  # per-trial latencies (NaN=excluded)
    snr_by_sigma: dict[float, float] = {}
    derived_seeds: dict[float, Optional[int]] = {}  # Round-3 m-3a
    T = X_ttc0.shape[1]

    for sigma_idx, sigma in enumerate(args.noise_levels):
        # the peak |visual_angle| excursion within the valid length;
        # report the median SNR across trials so each sigma has an
        # explicit physical meaning.  Round-2 fix (Reviewer A MAJOR-D-4):
        # the per-sigma SNR value is persisted in the JSON summary, not
        # only logged.
        peak_excursions = []
        for i in range(X_ttc0.shape[0]):
            L = int(L_ttc0[i].item())
            angle = X_ttc0[i, :L, VISUAL_ANGLE_IDX].cpu().numpy()
            if angle.size:
                peak_excursions.append(float(np.abs(angle).max()))
        sig_scale = float(np.median(peak_excursions)) if peak_excursions else 0.0
        snr_db = (
            20.0 * np.log10(sig_scale / sigma)
            if sigma > 0.0 and sig_scale > 0.0
            else float("inf")
        )
        snr_by_sigma[sigma] = float(snr_db)
        logger.info("--- Noise level σ = %.1f° (median SNR %.1f dB) ---", sigma, snr_db)

        X_noisy = inject_visual_noise(
            X_ttc0, L_ttc0, sigma, seed=args.seed,
            trial_seed_offset=sigma_idx,
        )
        # Round-3 (Reviewer A m-3a): record the ACTUAL derived seed per
        # σ so each condition is independently reproducible.
        derived_seeds[sigma] = (
            int(args.seed) + 1_000_003 * int(sigma_idx)
            if sigma > 0.0 else None
        )

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
        latency_arrays[sigma] = np.asarray(latencies, dtype=np.float64)
        valid = latency_arrays[sigma][~np.isnan(latency_arrays[sigma])]
        mean_lat = float(np.mean(valid)) if valid.size else float("nan")
        sem_lat = (
            float(np.std(valid, ddof=1) / np.sqrt(valid.size))
            if valid.size > 1 else 0.0
        )
        latency_stats[sigma] = {
            "mean": mean_lat,
            "sem": sem_lat,
            "n": int(valid.size),
            "n_excluded_prestim_peak": int(np.isnan(latency_arrays[sigma]).sum()),
            "std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
        }
        # Round-2 fix (Reviewer B M-1c): report the VALID count, not the
        # raw array length (which includes NaN-excluded trials).
        logger.info("  Latency: %.1f ± %.1f ms (n=%d)", mean_lat, sem_lat, int(valid.size))

        # Peak velocity
        peaks = extract_peak_velocity(Y_pred, L_ttc0)
        logger.info("  Peak Vel: %.2f cm/s", float(np.mean(peaks)))

    # --- Inferential statistics (Reviewer Round-1 BLOCKER-3) ---
    # Paired comparisons of per-trial latency against the clean
    # condition (σ=0).  Trials are PAIRED (identical stimulus set at
    # every noise level); noise realisations are independent across σ.
    # Holm-Bonferroni controls the family-wise error rate across the
    # multiple σ levels; the Hodges-Lehmann location shift quantifies
    # effect size, matched to the Wilcoxon signed-rank test (Round-3
    # CRITICAL-4B: single fixed test — no Shapiro pre-test gate).
    #
    # Round-2 fixes:
    # * Reviewer B M-1a: p_values / effect_sizes / corrected are bound
    #   unconditionally so the JSON construction below cannot hit a
    #   NameError when σ=0 is absent from --noise_levels.
    # * Reviewer B M-1b: exclusion is ANCHORED ON THE CLEAN CONDITION —
    #   a trial excluded at σ=0 is excluded from every condition, so all
    #   conditions are tested on the identical stimulus set.  Per-
    #   condition exclusion counts are reported.
    # * Reviewer A MAJOR-D-1/2: NaN p-values and zero-variance paired
    #   differences are REMOVED from the Holm family (and counted),
    #   never imputed as "no effect" — matching the lesion-script
    #   standard adopted in this same PR.
    from nsmor.analysis.uq import holm_bonferroni
    from scipy import stats as sp_stats

    p_values: dict[float, float] = {}
    effect_sizes: dict[float, float] = {}
    corrected: dict = {}

    if 0.0 in latency_arrays:
        baseline = latency_arrays[0.0]
        clean_valid = ~np.isnan(baseline)

        for sigma in args.noise_levels:
            if sigma == 0.0:
                continue
            noisy = latency_arrays[sigma]
            # Identical stimulus set across conditions: anchor on the
            # clean-condition exclusions.
            valid = clean_valid & ~np.isnan(noisy)
            n_pairs = int(valid.sum())
            latency_stats[sigma]["n_pairs_vs_clean"] = n_pairs
            if n_pairs < 3:
                logger.warning(
                    "σ=%.1f°: only %d complete pairs — test skipped.",
                    sigma, n_pairs,
                )
                continue
            b, q = baseline[valid], noisy[valid]

            diffs = q - b

            # Zero-variance paired differences are degenerate samples:
            # exclude from the family and count (Reviewer A MAJOR-D-2).
            sd_diff = float(np.std(diffs, ddof=1))
            if sd_diff <= 1e-12:
                logger.warning(
                    "σ=%.1f°: zero-variance paired differences "
                    "(n=%d) — excluded from test family.",
                    sigma, n_pairs,
                )
                latency_stats[sigma]["n_degenerate_zero_variance"] = n_pairs
                latency_stats[sigma]["test"] = "excluded_zero_variance"
                continue

            # Round-3 fix (Reviewer B CRITICAL-4B): the Shapiro-Wilk
            # pre-test gate inflates type-I error (pre-test dilemma) and
            # mixing Wilcoxon p-values with Cohen's d_z is a paradigm
            # mismatch.  The paired design now uses a SINGLE test:
            # Wilcoxon signed-rank with the Hodges-Lehmann estimate as
            # effect size (median of pairwise averages of the diffs) —
            # distribution-free, no pre-test, effect and test measure
            # the same quantity.  Normality is still REPORTED
            # descriptively (Shapiro-Wilk W logged per condition) so
            # readers can judge approximate symmetry, but it no longer
            # selects the procedure.
            stat, p_raw = sp_stats.wilcoxon(q, b)
            test_name = "Wilcoxon_signed_rank"

            # NaN p-values are numerical pathologies, not null results:
            # exclude from the family and count (Reviewer A MAJOR-D-1).
            if np.isnan(p_raw):
                logger.warning(
                    "σ=%.1f°: %s returned NaN p-value — excluded "
                    "from Holm family.", sigma, test_name,
                )
                latency_stats[sigma]["n_nan_pvalue_excluded"] = 1
                latency_stats[sigma]["test"] = f"{test_name}_nan_pvalue"
                continue

            # Hodges-Lehmann location shift: median over all pairwise
            # averages of the differences — the effect measure matched
            # to the Wilcoxon signed-rank test (Round-3 CRITICAL-4B).
            pairwise_avg = (
                diffs[:, None] + diffs[None, :]
            )[np.triu_indices(n_pairs, k=1)]
            hl = float(np.median(
                np.concatenate([pairwise_avg, diffs])
            )) if n_pairs > 1 else float(np.median(diffs))
            p_values[sigma] = float(p_raw)
            effect_sizes[sigma] = hl
            latency_stats[sigma]["test"] = test_name
            latency_stats[sigma]["hodges_lehmann_ms"] = hl
            # Descriptive normality diagnostic (does NOT gate the test).
            if n_pairs >= 3 and n_pairs <= 5000:
                W_stat, W_p = sp_stats.shapiro(diffs)
                latency_stats[sigma]["shapiro_W"] = float(W_stat)
                latency_stats[sigma]["shapiro_p"] = float(W_p)

        corrected = holm_bonferroni(p_values) if p_values else {}
        logger.info("-" * 60)
        logger.info(
            "Inference: post-stimulus latency vs clean condition "
            "(identical stimulus set anchored on σ=0 exclusions, "
            "Holm-Bonferroni corrected):"
        )
        for sigma in sorted(p_values):
            adj_p, sig = corrected.get(sigma, (1.0, False))
            logger.info(
                "  σ=%.1f° vs σ=0°: HL=%+.3f ms, %s p=%.4f (corrected), %s",
                sigma, effect_sizes[sigma],
                latency_stats[sigma].get("test", "Wilcoxon"),
                adj_p, "*" if sig else "n.s.",
            )
    else:
        logger.warning(
            "σ=0 not among --noise_levels; no paired inference possible."
        )

    # --- Create figure ---
    fig_path = os.path.join(args.output_dir, "bayesian_reliability.png")
    create_figure(gate_trajectories, latency_stats, T, 10.0, fig_path)

    # --- Export JSON summary ---
    summary = {
        "noise_levels": args.noise_levels,
        "n_ttc0_trials": n_ttc0,
        "stim_onset_frame": STIM_ONSET_FRAME,
        # Round-1 (Reviewer A MAJOR-2): explicit scope + SNR semantics
        "scope": (
            "Routing-gate sensitivity to VISUAL-channel noise. "
            "MCMC prior columns held fixed across conditions; this is "
            "NOT a Bayesian cue-combination test. Latencies are "
            "post-stimulus-peak only (pre-stimulus peaks reported as "
            "NaN and excluded)."
        ),
        "snr_definition": (
            "SNR_dB = 20*log10(median peak |visual_angle| / sigma); "
            "sigma is additive noise in degrees on the raw visual-angle "
            "channel before sensory encoding."
        ),
        # Round-2 fix (Reviewer A MAJOR-D-4): per-sigma SNR values are
        # persisted so the summary is statistically self-contained.
        "snr_db_by_sigma": {str(k): v for k, v in snr_by_sigma.items()},
        "noise_seed": args.seed,
        # Round-3 (Reviewer A m-3a): per-σ derived seeds for exact
        # per-condition reproduction.
        "derived_seeds_by_sigma": {
            str(k): v for k, v in derived_seeds.items()
        },
        "noise_realisations": (
            "independent per sigma level (seed + 1000003*sigma_index); "
            "pairing unit is the stimulus trial"
        ),
        "latency_stats": {
            str(k): v for k, v in latency_stats.items()
        },
        "gate_post_stim_mean": {
            str(sigma): float(gate_trajectories[sigma][STIM_ONSET_FRAME:].mean())
            for sigma in args.noise_levels
        },
        "inference": {
            "design": "paired per trial (identical stimulus set anchored "
                      "on σ=0 exclusions), independent noise "
                      "realisations across σ levels",
            "correction": "Holm-Bonferroni step-down (FWER α=0.05)",
            # Round-3 (CRITICAL-4B): single fixed test, no pre-test gate.
            "test": "Wilcoxon signed-rank (paired)",
            "effect_size": "Hodges-Lehmann location shift (ms)",
            "p_values_uncorrected": {str(k): v for k, v in p_values.items()},
            "p_values_holm_corrected": (
                {str(k): v[0] for k, v in corrected.items()} if corrected else {}
            ),
            "effect_sizes_hodges_lehmann_ms": {
                str(k): v for k, v in effect_sizes.items()
            },
        } if 0.0 in latency_arrays else {"design": "σ=0 absent — no paired inference"},
    }
    json_path = os.path.join(args.output_dir, "psychophysics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("JSON summary saved → %s", json_path)

    logger.info("Done. All outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
