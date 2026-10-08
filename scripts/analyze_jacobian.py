"""
NSMoR Jacobian Eigenvalue Spectrum Analysis — Phase 8.

Computes the Jacobian of the GRU at different trial phases (epochs)
and plots the eigenvalues on the complex plane (unit circle).

Target: Label.PREWALK trials (sustained locomotion response) by
default — the "sustained" epoch hypothesis requires a behaviour that
actually sustains locomotion.  ESCAPE trials may still be selected
explicitly via ``--target_class 0``.

Epochs relative to the observed wind onset or supplied visual collision reference:
  1. Early (Baseline):      reference - 1000ms
  2. Transient (Burst):     reference
  3. Sustained (Late Walk): reference + 1000ms

Slow-point search: For each epoch, a ±5-frame window is searched to
find the frame that minimises kinetic energy ||h_{t+1} - h_t||₂.

Round-1 fixes (Reviewer A BLOCKER-2 / Reviewer B MAJOR-1):
  * Every candidate slow point is verified against a quasi-fixed-point
    residual criterion ||GRU(x,h) - h|| < FP_RESIDUAL_THRESHOLD before
    its spectrum is computed; states failing the check are excluded
    and reported.  Eigenvalues at non-stationary points do NOT license
    line-attractor claims.
  * Surviving candidates additionally pass through
    ``FixedPointAdapter.test_attractor_convergence`` (perturbation-
    response verification along top eigendirections).
  * The docstring/CLI default mismatch (PREWALK vs ESCAPE) is removed.
  * Input-dependence vs state-dependence is separated: eigenvalue
    statistics are also reported under a common frozen input (the
    epoch-pooled median e_sensory), so spectral differences between
    epochs reflect state changes under an identical map.

Hypothesis: During the "Sustained" epoch, eigenvalues of VERIFIED
slow points should cluster near the boundary of the unit circle,
consistent with continuous integration / line attractor operation.

Output: ``results/jacobian_spectrum.png`` at 300 DPI.

Usage
-----

CLI::

    python scripts/analyze_jacobian.py --checkpoint runs/default/best_model.pth
    python scripts/analyze_jacobian.py --checkpoint runs/default/best_model.pth --dataset data/processed/nsmor_dataset.pt
"""

from __future__ import annotations

import argparse
import sys
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Direct CLI execution must use this checkout, including its prior/lineage helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from nsmor.analysis.analyze_jacobian_jax import create_jacobian_adapter
from nsmor.analysis.dynamics import FixedPointAdapter, raw_gru_trajectory
from nsmor.nsmor_dataloader import NSMoRDataset
from nsmor.dataloader_factory import create_optimized_dataloader
from nsmor.checkpoint import load_checkpoint
from nsmor.config import DEFAULT_FEATURE, Label
from nsmor.model_nsmor_core import NSMoRCore
from nsmor.analysis.analysis_priors import (
    describe_analysis_population, load_analysis_priors, population_for_output,
)
from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint
from nsmor.analysis.prediction_units import resolve_dt_ms
from nsmor.analysis.prediction_units import load_model_from_checkpoint as _shared_load_model
from nsmor.model_utils import validate_dataset_provenance

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Lancet / Cell Publication Style Constants
# ═══════════════════════════════════════════════════════════════

# ── High-contrast color mapping ──
EIGENVALUE_COLOR: str = "#1C7ED6"       # Cell Cobalt Blue
UNIT_CIRCLE_COLOR: str = "#495057"      # Strong Slate Gray
AXIS_COLOR: str = "#212529"             # Solid dark charcoal
BACKGROUND_COLOR: str = "#FFFFFF"       # Clean white
CMAP_HEATMAP: str = "inferno"           # Perceptually uniform heatmap

# ── Typography ─────────────────────────────────────────────────
FONT_FAMILY: str = "Arial"
FONT_SIZE_AXIS_TITLE: int = 12
FONT_SIZE_TICK: int = 10
FONT_SIZE_LEGEND: int = 9
FONT_SIZE_PANEL_LABEL: int = 14

# ── Figure properties ─────────────────────────────────────────
DPI: int = 300
FIG_WIDTH_INCHES: float = 15.0
FIG_HEIGHT_INCHES: float = 5.0

# ── Plot properties ───────────────────────────────────────────
EIGENVALUE_ALPHA: float = 0.6
EIGENVALUE_SIZE: float = 15.0
UNIT_CIRCLE_LINEWIDTH: float = 1.5
UNIT_CIRCLE_LINESTYLE: str = "--"
HEXBIN_GRIDSIZE: int = 40               # Resolution for hexbin density plot

# ── Epoch definitions ─────────────────────────────────────────
# Time offsets relative to stimulus onset (in ms)
EPOCH_DEFINITIONS: Dict[str, Dict[str, float]] = {
    "early": {
        "offset_ms": -1000.0,
        "label": "Early (Baseline)\nonset − 1000 ms",
    },
    "transient": {
        "offset_ms": 0.0,
        "label": "Transient (Stimulus Burst)\nonset",
    },
    "sustained": {
        "offset_ms": 1000.0,
        "label": "Sustained (Late Walk)\nonset + 1000 ms",
    },
}

# ── Slow-point search ────────────────────────────────────────
SLOW_POINT_RADIUS: int = 5   # ±5 frames around each epoch centre

# ── Quasi-fixed-point residual gate (Round-1 BLOCKER-2) ─────
# Round-3 fix (Reviewer A BLK-3A / B-CRIT-1/2): the Round-2 "Kneedle
# elbow" threshold was replaced by a 1-D two-component Gaussian mixture
# model over the log-residuals with BIC model selection.  The elbow of
# a sorted curve is a purely geometric construct: on the near-uniform
# residual distributions actually observed it lands at an arbitrary
# quantile (empirically ~p50), rejecting half of all candidates —
# including every state in one epoch — while providing no evidence that
# the rejected states are transients rather than slow points.  A
# two-component GMM asks the statistically meaningful question: does
# the residual distribution contain a distinct low-residual
# subpopulation at all?  If YES (BIC prefers 2 components), the gate is
# the posterior probability of the low component; if NO (BIC ties or
# prefers 1 component), there is NO evidence for a slow manifold and
# the analysis aborts loudly instead of silently keeping an arbitrary
# subset.
FP_GMM_RANDOM_SEED: int = 42
# Sanity cap on the GMM-calibrated boundary: a residual above this value
# means per-step state displacement comparable to the tanh-bounded state
# magnitude — no quasi-fixed-point interpretation is possible (Round-3
# Reviewer B CRIT-2: the cap must exist independently of the calibration
# so the gate can never talk itself into accepting transients).
FP_RESIDUAL_THRESHOLD_CAP: float = 0.3

# ── Component support / structure guards (Round-4) ─────────────
# A BIC preference for two components is NECESSARY but not SUFFICIENT
# evidence of two distinct subpopulations.  Two failure modes survive
# the ΔBIC gate on synthetic data:
#   * a skewed UNIMODAL log-residual distribution, where the second
#     component buys likelihood by stretching one tail (ΔBIC ≈ 32–63 on
#     exp(-6 + Exp(0.4/0.5)) with a single fitted density mode), and
#   * a DEGENERATE fit, where one component collapses onto a near-empty
#     point mass (weight 1/129, σ_log ≈ 0.001) and absorbs an outlier.
# The structure of the fitted DENSITY, plus the support of both
# components, must therefore be checked before the boundary is used.
FP_MIN_COMPONENT_SUPPORT: int = 5
"""Minimum effective sample size (``round(weight * n)``) required for
each fitted mixture component.  A component backed by fewer candidates
is a point-mass/outlier artefact, not a subpopulation."""
FP_MIN_COMPONENT_SIGMA_LOG: float = 0.01
"""Minimum fitted log-space component standard deviation.  A narrower
component is a degenerate spike, not a resolvable cluster."""
FP_MIN_DENSITY_MODES: int = 2
"""Minimum number of strict interior local maxima the fitted density
must exhibit before it counts as bimodal."""


def _fitted_density_mode_count(
    mu_lo: float,
    sigma_lo: float,
    weight_lo: float,
    mu_hi: float,
    sigma_hi: float,
    weight_hi: float,
    grid_points: int = 1001,
) -> int:
    """Count interior local maxima of the fitted two-component density.

    Round-4 fix: the gate previously treated ``ΔBIC > 10`` alone as proof
    of bimodality.  On a skewed unimodal log-residual distribution the
    second component can raise the likelihood without creating a second
    mode, and on a degenerate fit the extra component is a point mass.
    Both leave the fitted *density* unimodal.  This evaluates the weighted
    mixture on a closed grid spanning both components and counts strict
    interior local maxima, so "bimodal" is established from the density
    shape rather than from the component count.

    Args:
        mu_lo: Log-space mean of the low-residual component.
        sigma_lo: Log-space standard deviation of the low component.
        weight_lo: Mixing weight of the low component.
        mu_hi: Log-space mean of the high-residual component.
        sigma_hi: Log-space standard deviation of the high component.
        weight_hi: Mixing weight of the high component.
        grid_points: Number of evaluation points on the closed grid.

    Returns:
        Number of strict interior local maxima (0, 1, or 2 for a
        two-component fit).

    Raises:
        ValueError: If any parameter is non-finite or a scale/weight is
            non-positive.
    """
    values = (mu_lo, sigma_lo, weight_lo, mu_hi, sigma_hi, weight_hi)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("non-finite component parameters for mode count")
    if sigma_lo <= 0.0 or sigma_hi <= 0.0 or weight_lo <= 0.0 or weight_hi <= 0.0:
        raise ValueError("invalid component scale or weight for mode count")

    m1, s1 = (float(mu_lo), float(sigma_lo)) if mu_lo <= mu_hi else (float(mu_hi), float(sigma_hi))
    m2, s2 = (float(mu_hi), float(sigma_hi)) if mu_lo <= mu_hi else (float(mu_lo), float(sigma_lo))
    w1 = float(weight_lo) if mu_lo <= mu_hi else float(weight_hi)
    w2 = float(weight_hi) if mu_lo <= mu_hi else float(weight_lo)

    n_comp = max(201, (grid_points // 4) | 1)
    n_mid = max(101, (grid_points // 8) | 1)
    pts = [
        np.linspace(m1 - 4.5 * s1, m1 + 4.5 * s1, n_comp),
        np.linspace(m2 - 4.5 * s2, m2 + 4.5 * s2, n_comp),
    ]
    if m2 > m1:
        pts.append(np.linspace(m1, m2, n_mid))
    raw = np.sort(np.concatenate(pts))
    eps = min(s1, s2) * 1e-4
    grid = raw[np.r_[True, np.diff(raw) > eps]]

    density = (
        w1 / s1 * np.exp(-0.5 * ((grid - m1) / s1) ** 2)
        + w2 / s2 * np.exp(-0.5 * ((grid - m2) / s2) ** 2)
    )
    # Condense consecutive flat plateaus so peak detection is invariant to equal adjacent floats
    condensed = density[np.r_[True, np.diff(density) != 0.0]]
    if len(condensed) < 3:
        return 0
    interior = condensed[1:-1]
    return int(np.count_nonzero(
        (interior > condensed[:-2]) & (interior > condensed[2:])
    ))


def _calibrate_fp_threshold(
    residuals: np.ndarray,
    context: str = "",
) -> Tuple[float, Dict[str, float]]:
    """
    Calibrate the fixed-point residual gate by 2-component GMM +
    BIC model selection on the pooled candidate residuals.

    Args:
        residuals: 1-D array of candidate residuals ||GRU(x,h) - h||.
        context: Epoch label for error/log messages.

    Returns:
        ``(threshold, diagnostics)`` where *threshold* is the residual
        value whose low-component posterior is 0.5 (the Bayes-optimal
        separation boundary), and *diagnostics* reports the full
        distribution (BIC1/BIC2, low-component weight, quantiles) so
        the calibration is auditable in the persisted summary.

    Raises:
        ValueError: If fewer than 4 residuals are supplied, or BIC
            favours ONE component (no distinct quasi-stationary
            subpopulation exists in this state space), or the fitted
            two-component mixture is degenerate (a component with
            insufficient support or near-zero log-variance), or the
            fitted DENSITY is not bimodal, or the fitted boundary
            exceeds ``FP_RESIDUAL_THRESHOLD_CAP``.
    """
    from sklearn.mixture import GaussianMixture

    if residuals.size < 4:
        raise ValueError(
            f"Only {residuals.size} candidate residuals collected ({context}) — "
            "cannot calibrate the fixed-point gate."
        )
    r = np.asarray(residuals, dtype=np.float64)
    if r.ndim != 1 or not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ValueError(f"[{context}] residuals must be finite, positive and 1-D")
    # Log-transform: residuals are strictly positive and typically
    # right-skewed; the mixture structure lives in log space.
    x_log = np.log(r)[:, None]

    gm1 = GaussianMixture(
        n_components=1, covariance_type="full", random_state=FP_GMM_RANDOM_SEED,
    ).fit(x_log)
    gm2 = GaussianMixture(
        n_components=2, covariance_type="full", random_state=FP_GMM_RANDOM_SEED,
    ).fit(x_log)

    bic1, bic2 = float(gm1.bic(x_log)), float(gm2.bic(x_log))
    diag: Dict[str, float] = {
        "bic_1comp": bic1,
        "bic_2comp": bic2,
        "residual_min": float(r.min()),
        "residual_p25": float(np.percentile(r, 25)),
        "residual_median": float(np.median(r)),
        "residual_p75": float(np.percentile(r, 75)),
        "residual_max": float(r.max()),
        "n_candidates": int(r.size),
    }

    # Model selection: the two-component mixture must WIN by a margin,
    # not by noise.  ΔBIC > 10 is strong evidence (Kass & Raftery 1995).
    if bic2 >= bic1 - 10.0:
        raise ValueError(
            f"[{context}] BIC does not support a bimodal residual "
            f"distribution (BIC_1={bic1:.1f}, BIC_2={bic2:.1f}, "
            f"delta={bic1 - bic2:.1f}, need > +10).  There is NO "
            f"evidence for a distinct quasi-fixed-point subpopulation "
            f"in this state space; eigenvalue spectra here cannot "
            f"support stability claims.  Residual range "
            f"[{r.min():.4g}, {r.max():.4g}]."
        )

    means = gm2.means_.ravel()
    lo_idx, hi_idx = int(np.argmin(means)), int(np.argmax(means))
    w_lo = float(gm2.weights_[lo_idx])
    w_hi = float(gm2.weights_[hi_idx])
    diag["low_component_weight"] = w_lo
    diag["high_component_weight"] = w_hi

    # ── Round-4: component support guard ────────────────────────
    # A component backed by a handful of points (or a near-zero-variance
    # spike) is an outlier/point-mass artefact, not a subpopulation.  It
    # must not license a "two-population" gate however large its ΔBIC.
    support_lo = int(round(w_lo * r.size))
    support_hi = int(round(w_hi * r.size))
    sigma_lo_log = float(np.sqrt(gm2.covariances_[lo_idx, 0, 0]))
    sigma_hi_log = float(np.sqrt(gm2.covariances_[hi_idx, 0, 0]))
    diag["low_component_support"] = support_lo
    diag["high_component_support"] = support_hi
    diag["low_sigma_log"] = sigma_lo_log
    diag["high_sigma_log"] = sigma_hi_log
    if (
        support_lo < FP_MIN_COMPONENT_SUPPORT
        or support_hi < FP_MIN_COMPONENT_SUPPORT
        or sigma_lo_log < FP_MIN_COMPONENT_SIGMA_LOG
        or sigma_hi_log < FP_MIN_COMPONENT_SIGMA_LOG
    ):
        raise ValueError(
            f"[{context}] two-component fit is DEGENERATE: a component "
            f"lacks support (support_lo={support_lo}, support_hi={support_hi}, "
            f"need >= {FP_MIN_COMPONENT_SUPPORT}; sigma_log=("
            f"{sigma_lo_log:.4g}, {sigma_hi_log:.4g}), need >= "
            f"{FP_MIN_COMPONENT_SIGMA_LOG}).  One component is a point mass / "
            f"outlier artefact, so there is NO evidence for two distinct "
            f"quasi-fixed-point subpopulations; eigenvalue spectra here cannot "
            f"support stability claims."
        )

    # ── Round-4: density-structure guard ────────────────────────
    # BIC preference for two components is not itself evidence of two
    # modes: a skewed unimodal distribution can absorb a tail-stretching
    # second component (ΔBIC ≈ 32–63 with a single fitted mode).  The
    # fitted DENSITY must actually be bimodal.
    n_modes = _fitted_density_mode_count(
        float(means[lo_idx]), sigma_lo_log, w_lo,
        float(means[hi_idx]), sigma_hi_log, w_hi,
    )
    diag["fitted_density_modes"] = n_modes
    if n_modes < FP_MIN_DENSITY_MODES:
        raise ValueError(
            f"[{context}] BIC prefers two components (delta={bic1 - bic2:.1f}) "
            f"but the FITTED DENSITY has {n_modes} interior mode(s), not "
            f"{FP_MIN_DENSITY_MODES}: the extra component stretched one tail "
            f"of a unimodal distribution rather than separating a distinct "
            f"subpopulation.  BIC preference alone is not evidence of two "
            f"modes; no quasi-fixed-point population can be claimed here."
        )

    # Boundary where the posterior of the LOW component equals 0.5:
    # solve log N(x|mu_lo,s_lo) + log w_lo == log N(x|mu_hi,s_hi) + log w_hi.
    mu_lo, mu_hi = float(means[lo_idx]), float(means[hi_idx])
    s_lo = sigma_lo_log
    s_hi = sigma_hi_log
    boundary_log = _gmm_posterior_half_boundary_log(
        mu_lo, s_lo, w_lo, mu_hi, s_hi, w_hi, context,
    )

    diag.update(
        low_mean_log=mu_lo, high_mean_log=mu_hi,
    )
    threshold = float(np.exp(boundary_log))
    diag["fp_threshold"] = threshold

    if threshold > FP_RESIDUAL_THRESHOLD_CAP:
        raise ValueError(
            f"[{context}] calibrated residual boundary ({threshold:.3f}) "
            f"exceeds the sanity cap {FP_RESIDUAL_THRESHOLD_CAP:.1f} — "
            f"no usable quasi-fixed-point population exists in this "
            f"state space; eigenvalue spectra here cannot support "
            f"stability claims."
        )
    logger.info(
        "[%s] FP gate CALIBRATED by GMM+BIC: threshold=%.4g, "
        "BIC_1=%.1f, BIC_2=%.1f, low-comp weight=%.2f",
        context, threshold, bic1, bic2, w_lo,
    )
    return threshold, diag


def _gmm_posterior_half_boundary_log(
    mu_lo: float,
    sigma_lo: float,
    weight_lo: float,
    mu_hi: float,
    sigma_hi: float,
    weight_hi: float,
    context: str = "",
) -> float:
    """Return the log-residual where the two GMM posteriors are equal.

    The equality is between the *weighted* Gaussian densities.  Keeping this
    calculation separate makes the prior/variance terms auditable and avoids
    silently selecting a boundary for the unweighted component densities.
    """
    values = (mu_lo, sigma_lo, weight_lo, mu_hi, sigma_hi, weight_hi)
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"[{context}] non-finite GMM boundary parameters")
    if sigma_lo <= 0.0 or sigma_hi <= 0.0 or weight_lo <= 0.0 or weight_hi <= 0.0:
        raise ValueError(f"[{context}] invalid GMM boundary scale or weight")
    if mu_lo >= mu_hi:
        raise ValueError(f"[{context}] low GMM mean must precede high mean")

    # log(w_lo N_lo) - log(w_hi N_hi) = 0, expanded as a*x²+b*x+c.
    a = 0.5 / sigma_hi**2 - 0.5 / sigma_lo**2
    b = mu_lo / sigma_lo**2 - mu_hi / sigma_hi**2
    c = (
        0.5 * mu_hi**2 / sigma_hi**2
        - 0.5 * mu_lo**2 / sigma_lo**2
        + np.log(weight_lo / weight_hi)
        + np.log(sigma_hi / sigma_lo)
    )
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            raise ValueError(f"[{context}] GMM components have no unique boundary")
        roots = [-c / b]
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            raise ValueError(
                f"[{context}] GMM components too separated to admit a "
                "posterior boundary — degenerate fit."
            )
        sqrt_discriminant = np.sqrt(discriminant)
        roots = [
            (-b - sqrt_discriminant) / (2.0 * a),
            (-b + sqrt_discriminant) / (2.0 * a),
        ]

    between = [root for root in roots if mu_lo <= root <= mu_hi]
    if not between:
        raise ValueError(
            f"[{context}] no GMM posterior boundary between the component means "
            "— degenerate fit."
        )
    # In the usual two-component fit there is one crossing in this interval.
    # If unequal variances create two, use the crossing nearest the midpoint;
    # it is the boundary between the component means rather than a tail
    # crossing.
    return float(min(between, key=lambda root: abs(root - (mu_lo + mu_hi) / 2.0)))


def _posterior_keep(
    residuals: np.ndarray, threshold: float, diagnostics: Dict[str, Any],
) -> np.ndarray:
    """Accept the fitted low-component posterior, including unequal-width tails."""
    if "low_mean_log" not in diagnostics:
        return residuals < threshold
    x = np.log(residuals)
    lo = (
        np.log(diagnostics["low_component_weight"] / diagnostics["low_sigma_log"])
        - 0.5 * ((x - diagnostics["low_mean_log"])
                 / diagnostics["low_sigma_log"]) ** 2
    )
    hi = (
        np.log(diagnostics["high_component_weight"] / diagnostics["high_sigma_log"])
        - 0.5 * ((x - diagnostics["high_mean_log"])
                 / diagnostics["high_sigma_log"]) ** 2
    )
    return (lo > hi) & (residuals < FP_RESIDUAL_THRESHOLD_CAP)


def _sample_indices(n_states: int, cap: int, seed: int) -> torch.Tensor:
    """Select a capped CPU subset without consuming the caller's RNG."""
    if cap <= 0:
        raise ValueError("max_states_per_epoch must be positive")
    if n_states <= cap:
        return torch.arange(n_states)
    return torch.randperm(
        n_states, generator=torch.Generator().manual_seed(seed),
    )[:cap]


# ═══════════════════════════════════════════════════════════════
# 1.  Model Loading
# ═══════════════════════════════════════════════════════════════

def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> NSMoRCore:
    """Load trained NSMoRCore from checkpoint.

    Delegates to the shared :func:`nsmor.analysis.prediction_units.load_model_from_checkpoint`
    which guarantees all biophysical parameters are restored.
    """
    return _shared_load_model(checkpoint_path, device)


# ═══════════════════════════════════════════════════════════════
# 2.  Dataset Loading
# ═══════════════════════════════════════════════════════════════

def load_dataset(
    dataset_path: Path,
    batch_size: int = 32,
    max_seq_len: Optional[int] = 2400,
    pre_anchor_frames: int = 1200,
    nested_prior_artifact: Optional[Path] = None,
    qc_sealed_nested_prior_sha256: Optional[str] = None,
    checkpoint_model: Optional[NSMoRCore] = None,
    trusted_historical_checkpoint_sha256: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
) -> Tuple[torch.utils.data.DataLoader, np.ndarray, List[int], List[np.ndarray]]:
    """
    Load the preprocessed dataset and create a DataLoader.

    Also returns the raw X_seqs so that stimulus onset can be
    detected dynamically from the sensory channels.

    Args:
        dataset_path: Path to ``nsmor_dataset.pt``.
        batch_size: Batch size for the DataLoader.
        max_seq_len: Maximum sequence length for cropping.
        pre_anchor_frames: Baseline frames before anchor.

    Returns:
        ``(dataloader, labels, lengths_list, X_seqs)`` tuple.

    Raises:
        FileNotFoundError: If dataset file does not exist.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info("Loading dataset from %s", dataset_path)
    dataset, loaded_source_fingerprint = load_dataset_with_fingerprint(
        dataset_path, restore_provenance=False,
        expected_dt_ms=(
            resolve_dt_ms(checkpoint_model) if checkpoint_model is not None else None
        ),
    )

    # Round-2 CRITICAL-A: refuse pre-2.0 datasets (leaked priors, np.max labels)
    validate_dataset_provenance(dataset, Path(dataset_path))

    X_seqs = dataset["X_seqs"]
    Y_seqs = dataset["Y_seqs"]
    mcmc_priors, val_indices = load_analysis_priors(
        dataset, dataset_path, nested_prior_artifact, checkpoint_model, qc_sealed_nested_prior_sha256,
        loaded_source_fingerprint=loaded_source_fingerprint,
        trusted_historical_checkpoint_sha256=trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
    )
    labels = dataset["labels"]
    lengths = dataset["lengths"]

    n_total = len(X_seqs)
    logger.info("Loaded %d sequences.", n_total)

    anchor_frames = dataset.get("anchor_frames")
    if anchor_frames is None:
        from nsmor.pipeline.conditions import derive_anchor_frames
        anchor_frames = derive_anchor_frames(X_seqs, lengths)
        logger.info(
            "Derived %d anchor frames from physical channels.",
            len(anchor_frames),
        )

    # Build sequence list
    sequences = [
        (X_seqs[i], Y_seqs[i], int(labels[i]))
        for i in range(n_total)
    ]

    # Create dataset and dataloader
    feature_config = dataset.get("feature_config", DEFAULT_FEATURE)
    bio_dataset = NSMoRDataset(
        sequences=sequences,
        mcmc_priors=mcmc_priors,
        feature_config=feature_config,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        anchor_frames=anchor_frames,
    )

    bio_dataset.trial_ids = dataset.get("trial_ids")
    bio_dataset.session_ids = dataset.get("session_ids")
    # Producer anchors are model-grid references, not new source observations.
    # For visual-only trials an explicit anchor means collision, not onset.
    from nsmor.pipeline.conditions import resolve_anchor_crop
    detected = detect_stimulus_onset_frames(X_seqs)
    explicit_anchors = dataset.get("anchor_frames")
    analysis_references: List[Optional[int]] = []
    analysis_rules: List[str] = []
    for i, x_seq in enumerate(X_seqs):
        x_array = np.asarray(x_seq)
        valid_length = int(lengths[i])
        wind_indices = np.flatnonzero(x_array[:valid_length, 1] > 0.5)
        visual_present = bool(np.any(np.abs(x_array[:valid_length, 0]) > 1e-6))
        if wind_indices.size:
            raw_reference: Optional[int] = int(wind_indices[0])
            rule = "wind_onset"
        elif visual_present and explicit_anchors is not None:
            raw_reference = explicit_anchors[i]
            rule = "looming_collision" if raw_reference is not None else "unavailable"
        elif visual_present:
            raw_reference = detected[i]
            rule = "visual_onset" if raw_reference is not None else "unavailable"
        else:
            raw_reference = None
            rule = "unavailable"
        if raw_reference is None:
            analysis_references.append(None)
        else:
            raw_reference = _validate_analysis_reference(raw_reference, valid_length, i)
            start, _end = resolve_anchor_crop(
                n_frames=len(x_array), anchor_frame=bio_dataset.anchor_frames[i],
                max_seq_len=bio_dataset.max_seq_len,
                pre_anchor_frames=bio_dataset.pre_anchor_frames,
            )
            mapped_reference = raw_reference - start
            analysis_references.append(mapped_reference)
        analysis_rules.append(rule)
    bio_dataset.analysis_reference_frames = analysis_references
    bio_dataset.analysis_reference_rules = analysis_rules
    bio_dataset.analysis_population = describe_analysis_population(
        n_total, val_indices, bio_dataset.source_indices,
        getattr(checkpoint_model, "analysis_validation_scope", None),
    )
    dataloader = create_optimized_dataloader(
        bio_dataset,
        batch_size=batch_size,
        shuffle=False,  # Preserve ordering for label matching
        num_workers=0,  # Whole corpus is already resident; workers only replicate it
    )

    lengths_list = [int(l) for l in lengths]
    return dataloader, labels, lengths_list, X_seqs


# ═══════════════════════════════════════════════════════════════
# 3.  Dynamic Stimulus Onset Detection
# ═══════════════════════════════════════════════════════════════

def detect_stimulus_onset_frames(
    X_seqs: List[np.ndarray],
    dt_ms: float = 10.0,
    threshold: float = 1e-6,
) -> List[Optional[int]]:
    """
    Detect the stimulus onset frame for each trial from the data.

    Scans the sensory channels (visual angle feature 0, wind state
    feature 1) to find the first frame where *either* channel
    becomes non-zero.  This replaces the hardcoded
    ``stim_onset_frame = 200`` with a data-driven anchor.

    For looming trials the visual angle transitions from 0 to a
    positive value at stimulus onset.  For pure-wind trials the
    wind channel transitions from 0 to 1.  Both are captured.

    Args:
        X_seqs: List of arrays, each ``(T_i, 8)``.
        dt_ms: Frame interval in ms (for logging only).
        threshold: Absolute value below which a channel is
            considered zero.

    Returns:
        List of frame indices (one per sequence).  A sequence without an
        observed visual or wind onset returns ``None`` so downstream analyses
        can fail closed instead of inventing a frame-zero reference.
    """
    onset_frames: List[Optional[int]] = []
    for i, X in enumerate(X_seqs):
        assert X.ndim == 2 and X.shape[1] == 8, f"Trial {i}: {X.shape}"
        wind = np.flatnonzero(X[:, 1] > 0.5)
        visual_active = np.abs(X[:, 0]) > threshold
        if wind.size:
            onset_frames.append(int(wind[0]))
        elif visual_active.size and not visual_active[0] and visual_active.any():
            onset_frames.append(int(np.flatnonzero(visual_active)[0]))
        else:
            # An active visual baseline does not reveal looming onset/collision.
            onset_frames.append(None)
            logger.warning("Trial %d: stimulus reference unavailable.", i)
    return onset_frames


# ═══════════════════════════════════════════════════════════════
# 4.  GRU State Extraction at Specific Epochs
# ═══════════════════════════════════════════════════════════════

def _find_slow_point(
    gru_hidden: torch.Tensor,
    window_centre: int,
    length_i: int,
    radius: int = SLOW_POINT_RADIUS,
) -> Tuple[int, torch.Tensor]:
    """
    Find the slow-point frame within a ±radius window.

    The slow point is the frame *t* that minimises the kinetic
    energy  ‖h_{t+1} − h_t‖₂  within the search window.

    Args:
        gru_hidden: ``(T, H)`` GRU hidden-state trajectory for one trial.
        window_centre: Centre frame index of the search window.
        length_i: True (unpadded) sequence length.
        radius: Search radius in frames.

    Returns:
        ``(slow_frame, h_slow)`` where *slow_frame* is the index
        and *h_slow* is the ``(H,)`` hidden state at that frame.
    """
    lo = max(0, window_centre - radius)
    hi = min(length_i - 2, window_centre + radius)  # need t+1 to exist

    if lo > hi:
        # Degenerate window — fall back to centre clamped to valid range
        frame = max(0, min(window_centre, length_i - 2))
        return frame, gru_hidden[frame]

    # Kinetic energy: ||h_{t+1} - h_t||_2 for t in [lo, hi]
    h_window = gru_hidden[lo:hi + 2]         # (window+1, H)
    diffs = h_window[1:] - h_window[:-1]     # (window, H)
    ke = diffs.norm(dim=1)                   # (window,)

    best_local = int(ke.argmin().item())
    slow_frame = lo + best_local
    return slow_frame, gru_hidden[slow_frame]


def _fixed_point_residual(
    adapter: FixedPointAdapter,
    h_slow: torch.Tensor,
    x_slow: torch.Tensor,
) -> float:
    """
    One-step GRU residual at a candidate slow point.

    Computes ||GRU(x_slow, h_slow) - h_slow||_2 with the cell in eval
    mode (no dropout noise).  Round-1 BLOCKER-2: this is the fixed-point
    verification that was previously missing from the main analysis
    path — eigenvalue spectra are only computed for candidates passing
    the FP_RESIDUAL_THRESHOLD gate.
    """
    gru_cell = adapter._gru_cell
    prev_mode = gru_cell.training
    gru_cell.eval()
    try:
        with torch.no_grad():
            x_seq = x_slow.detach().unsqueeze(0).unsqueeze(0)   # (1, 1, H)
            h_in = h_slow.detach().unsqueeze(0).unsqueeze(0)    # (1, 1, H)
            h_next, _ = gru_cell(x_seq, h_in.permute(1, 0, 2))
            residual = float((h_next.squeeze(0).squeeze(0) - h_slow).norm().item())
    finally:
        gru_cell.train(prev_mode)
    return residual


def _validate_analysis_reference(
    reference: Optional[int],
    n_frames: int,
    trial_index: int,
) -> int:
    """Validate one supplied reference without inventing a fallback."""
    if reference is None or isinstance(reference, (bool, np.bool_)):
        raise ValueError(
            f"Trial {trial_index}: analysis reference unavailable; "
            "cannot select Jacobian epochs."
        )
    if not isinstance(reference, (int, np.integer)):
        raise ValueError(
            f"Trial {trial_index}: analysis reference unavailable; "
            f"expected an integer, got {reference!r}."
        )
    frame = int(reference)
    if frame < 0 or frame >= n_frames:
        raise ValueError(
            f"Trial {trial_index}: analysis reference unavailable; "
            f"frame {frame} is outside [0, {n_frames})."
        )
    return frame


def _reference_in_batch_coordinates(
    reference: Optional[int],
    dataset: Any,
    trial_index: int,
    batch_length: int,
) -> int:
    """Map a raw trial reference through the dataset's shared crop window."""
    analysis_references = getattr(dataset, "analysis_reference_frames", None)
    if analysis_references is not None:
        if trial_index >= len(analysis_references):
            raise ValueError(
                f"Trial {trial_index}: analysis reference unavailable; "
                "reference sidecar is misaligned."
            )
        mapped = analysis_references[trial_index]
        return _validate_analysis_reference(mapped, batch_length, trial_index)

    sequences = getattr(dataset, "sequences", None)
    if sequences is None or trial_index >= len(sequences):
        raise ValueError(
            f"Trial {trial_index}: analysis reference unavailable; "
            "dataset sequence metadata is missing."
        )
    x_raw = sequences[trial_index][0]
    # Minimal test/custom loaders may only supply batch tensors.
    raw_length = int(x_raw.shape[0]) if x_raw is not None else batch_length
    raw_reference = _validate_analysis_reference(reference, raw_length, trial_index)
    from nsmor.pipeline.conditions import resolve_anchor_crop

    anchors = getattr(dataset, "anchor_frames", None)
    anchor = anchors[trial_index] if anchors is not None else None
    start, end = resolve_anchor_crop(
        n_frames=raw_length,
        anchor_frame=anchor,
        max_seq_len=getattr(dataset, "max_seq_len", None),
        pre_anchor_frames=int(getattr(dataset, "pre_anchor_frames", 0)),
    )
    if end - start != batch_length:
        raise ValueError(
            f"Trial {trial_index}: analysis reference unavailable; "
            f"crop metadata length {end - start} != batch length {batch_length}."
        )
    return _validate_analysis_reference(
        raw_reference - start, batch_length, trial_index,
    )


def _epoch_reference_frame(
    dataset: Any,
    trial_index: int,
    x_trial: torch.Tensor,
    onset_frames: Sequence[Optional[int]],
) -> int:
    """Prefer original references; never replace them with a later cropped pulse."""
    length_i = x_trial.shape[0]
    assert x_trial.shape == (length_i, 8), f"Trial {trial_index}: {x_trial.shape}"
    if (getattr(dataset, "analysis_reference_frames", None) is not None
            or len(onset_frames) > 0):
        reference = (onset_frames[trial_index]
                     if trial_index < len(onset_frames) else None)
        return _reference_in_batch_coordinates(
            reference, dataset, trial_index, length_i,
        )
    # Legacy loaders without references may use the original observed wind.
    sequences = getattr(dataset, "sequences", None)
    x_raw = (sequences[trial_index][0]
             if sequences is not None and trial_index < len(sequences) else None)
    wind = (np.asarray(x_raw)[:, 1] if x_raw is not None
            else x_trial[:, 1].detach().cpu().numpy())
    wind_indices = np.flatnonzero(wind > 0.5)
    if wind_indices.size and int(wind_indices[0]) > 0:
        return _reference_in_batch_coordinates(
            int(wind_indices[0]), dataset, trial_index, length_i,
        )
    raise ValueError(f"Trial {trial_index}: analysis reference unavailable")


def _trial_identity(
    dataset: Any, trial_index: int, epoch_name: str, frame: int,
) -> Dict[str, Any]:
    """Return only available provenance for one candidate state."""
    source_indices = getattr(dataset, "source_indices", None)
    row = int(source_indices[trial_index]) if source_indices is not None else trial_index
    identity = {
        "candidate_id": f"{epoch_name}:{row}:{frame}",
        "source_row_index": row, "epoch": epoch_name, "frame": int(frame),
    }
    for sidecar, field in (("trial_ids", "source_trial_id"),
                           ("session_ids", "session_id")):
        values = getattr(dataset, sidecar, None)
        if values is not None and trial_index < len(values):
            value = values[trial_index]
            if value is not None:
                identity[field] = value.item() if isinstance(value, np.generic) else value
    return identity


def _reconstruct_gru_input(
    model: Any,
    X_batch: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the exact per-frame input the GRU cell received.

    For a real :class:`NSMoRCore` the GRU input is produced by the
    ``FrontendEncoder`` (dendritic IIR on the visual channels followed by
    the ``SensoryEncoder``), *not* by ``sensory_encoder`` alone.  Route
    through ``model.frontend`` so the reconstruction matches
    ``forward()`` when dendritic filtering is enabled; legacy probe
    models without a ``.frontend`` fall back to
    ``model.sensory_encoder``.

    ``FrontendEncoder`` keeps a module-level ``_dendritic_state`` cache
    that leaks ACROSS forward calls when dendritic filtering is enabled
    (Round-1 BLOCKER-2 item 3).  It is reset before re-encoding so the
    filter history starts from the sequence start, matching the original
    forward.  The reset is also required for correctness even without
    dendritic filtering, because it restores the documented
    "starts-from-zero at each sequence" contract.

    Args:
        model: Trained model exposing ``sensory_dim`` and either
            ``frontend`` or ``sensory_encoder``.
        X_batch: ``(B, T, D_total)`` padded feature tensor.
        lengths: ``(B,)`` true sequence lengths.

    Returns:
        ``(B, T, H)`` sensory encoding fed to the GRU.

    Raises:
        AssertionError: If the reconstructed shape does not match
            ``(B, T, H)``.
    """
    B, T, _ = X_batch.shape
    sensory_x = X_batch[:, :, :model.sensory_dim]  # (B, T, D_sensory)
    frontend = getattr(model, "frontend", None)
    if frontend is not None:
        if getattr(frontend, "_dendritic_enabled", False):
            frontend._dendritic_state = None
        e_sensory = frontend(sensory_x, lengths)     # (B, T, H)
        # Match the actual GRU input exactly: ``BioDecisionCore`` (and the
        # top-level core) exclude invalid (padded) frames with a selection
        # BEFORE the GRU/LIF/router math, so the per-frame encoding the GRU
        # receives is zero on the padded suffix.  Reconstruct the same
        # selection for a real model (which always has a frontend).
        lengths_i = lengths.to(device=X_batch.device, dtype=torch.int64)
        valid = (
            torch.arange(T, device=X_batch.device).unsqueeze(0)
            < lengths_i.unsqueeze(1)
        )                                            # (B, T) bool
        e_sensory = torch.where(
            valid.unsqueeze(-1), e_sensory, torch.zeros_like(e_sensory),
        )
    else:
        if getattr(model.sensory_encoder, "_dendritic_enabled", False):
            model.sensory_encoder._dendritic_state = None
        e_sensory = model.sensory_encoder(sensory_x)  # (B, T, H)
    H = e_sensory.shape[-1]
    assert e_sensory.shape == (B, T, H), (
        f"Reconstructed e_sensory shape {tuple(e_sensory.shape)} "
        f"!= (B={B}, T={T}, H={H})"
    )
    return e_sensory


def extract_gru_states_at_epochs(
    model: NSMoRCore,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    onset_frames: List[Optional[int]],
    target_class: int = Label.PREWALK.value,
    dt_ms: Optional[float] = None,
    adapter: Optional[FixedPointAdapter] = None,
) -> Dict[str, Any]:
    """
    Extract GRU hidden states at specific trial epochs for a target class.

    For each trial, computes epoch frames relative to the
    **dynamically detected** stimulus onset, then searches a
    ±``SLOW_POINT_RADIUS`` window for the slow point (minimum
    kinetic energy).

    Round-1 BLOCKER-2 fix: every candidate slow point must pass a
    quasi-fixed-point residual check before being accepted.  Candidates
    failing the gate are counted and logged; their spectra would be
    invalid for line-attractor inference.

    Round-2 fix (Reviewer B M-2a): the gate threshold is CALIBRATED
    from the candidate-residual distribution instead of a hand-picked
    constant — see :func:`_calibrate_fp_threshold`.

    Round-3 fix (Reviewer A BLK-3A / B-CRIT-2): the gate is calibrated
    PER EPOCH on each epoch's own residual distribution.  The previous
    pooled-across-epochs threshold was statistically incoherent: if
    epoch A's states are 10x slower than epoch B's, a pooled boundary
    either rejects all of A or accepts transients from B — exactly
    what happened (the Sustained epoch lost every one of its states).
    Each epoch now stands or falls on its OWN distribution; an epoch
    with no bimodal structure is dropped WITH its recorded reason,
    never silently.

    The input passed to the Jacobian adapter is the **full sensory
    encoding** ``e_sensory_t`` (dim H) produced by the model's
    ``frontend`` (dendritic IIR filtering when enabled, then the
    sensory encoder) — the vector the GRU cell actually receives at
    time *t*, reconstructed by :func:`_reconstruct_gru_input`.  For
    legacy probe models without a ``.frontend`` it is the inner
    ``sensory_encoder`` output.  This captures the partial derivative
    ∂h_{t+1}/∂h_t holding the GRU input fixed.

    Args:
        model: Trained NSMoRCore model.
        dataloader: DataLoader yielding (X, Y, lengths) tuples.
        device: Computation device.
        onset_frames: Per-trial stimulus onset frame indices
            (from :func:`detect_stimulus_onset_frames`).
        target_class: Label value to filter by
            (default: PREWALK — sustained locomotion).
        dt_ms: None inherits saved model.dt_ms; an explicit interval must match.
        adapter: FixedPointAdapter providing the GRU cell for the
            residual check.  Created from *model* if omitted.

    Returns:
        Dictionary mapping epoch name to ``(h_states, x_inputs)``
        tuples, plus per-epoch gate diagnostics attached as
        ``epoch name + "_gate_diag"`` keys (Round-3 BLK-3A: the
        calibration evidence is returned alongside the accepted states
        so the caller can persist it).

    Raises:
        ValueError: If no valid states found for ANY epoch after the
            per-epoch gates (each epoch's failure reason is logged).
    """
    logger.info("Extracting GRU states for class %d (%s) at target epochs...",
                target_class, Label(target_class).name)

    dt_ms = resolve_dt_ms(model, dt_ms)

    if adapter is None:
        adapter = FixedPointAdapter(model, device=device)

    # Initialize storage for each epoch
    epoch_states: Dict[str, List[torch.Tensor]] = {
        name: [] for name in EPOCH_DEFINITIONS
    }
    epoch_inputs: Dict[str, List[torch.Tensor]] = {
        name: [] for name in EPOCH_DEFINITIONS
    }
    by_epoch: Dict[str, List[
        Tuple[torch.Tensor, torch.Tensor, float, Dict[str, Any]]
    ]] = {name: [] for name in EPOCH_DEFINITIONS}
    accepted_identities: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in EPOCH_DEFINITIONS
    }

    model.eval()

    with torch.no_grad():
        global_idx = 0  # Tracks position across batches

        for batch_idx, batch in enumerate(dataloader):
            X_batch, _Y_batch, lengths = batch
            X_batch = X_batch.to(device).contiguous()
            lengths = lengths.to(device).contiguous()

            B, T, _ = X_batch.shape

            # Forward pass with internals to get GRU hidden states
            _y_pred, internals = model(X_batch, lengths, return_internals=True)

            # Raw recurrent GRU trajectory (B, T, H): the actual recurrent
            # coordinate, not the routed/post-gain ``gru_hidden`` (r5 R5).
            gru_hidden = raw_gru_trajectory(
                model, internals, context="extract_gru_states_at_epochs",
            )
            H = gru_hidden.shape[2]

            # ── Task 2: Exact input reconstruction ─────────────
            # Rebuild the exact vector the GRU cell received at each
            # frame (see :func:`_reconstruct_gru_input`).
            e_sensory = _reconstruct_gru_input(model, X_batch, lengths)

            for i in range(B):
                if global_idx >= len(dataloader.dataset):
                    break

                # Check if this trial is of the target class
                _, _, label_val = dataloader.dataset.sequences[global_idx]
                if int(label_val) != target_class:
                    global_idx += 1
                    continue

                length_i = int(lengths[i].item())

                onset_frame = _epoch_reference_frame(
                    dataloader.dataset, global_idx, X_batch[i, :length_i],
                    onset_frames,
                )

                # ── Compute epoch centre frames ────────────────
                for epoch_name, epoch_def in EPOCH_DEFINITIONS.items():
                    offset_ms = epoch_def["offset_ms"]
                    frame_offset = int(offset_ms / dt_ms)
                    centre_frame = onset_frame + frame_offset

                    # Bounds check (need at least frame+1 for KE)
                    if centre_frame < 0 or centre_frame >= length_i - 1:
                        logger.debug(
                            "  Trial %d: epoch '%s' centre frame %d "
                            "out of bounds (length=%d), skipping.",
                            global_idx, epoch_name, centre_frame, length_i,
                        )
                        continue

                    # ── Task 3: Slow-point search ──────────────
                    slow_frame, h_slow = _find_slow_point(
                        gru_hidden[i], centre_frame, length_i,
                    )
                    x_slow = e_sensory[i, slow_frame, :]    # (H,)

                    # Shape assertions
                    assert h_slow.shape == (H,), (
                        f"h_slow shape {tuple(h_slow.shape)} != (H={H},)"
                    )
                    assert x_slow.shape == (H,), (
                        f"x_slow shape {tuple(x_slow.shape)} != (H={H},)"
                    )

                    # Collect all candidates before calibrating each epoch's gate.
                    residual = _fixed_point_residual(adapter, h_slow, x_slow)
                    by_epoch[epoch_name].append((
                        h_slow.cpu(), x_slow.cpu(), float(residual),
                        _trial_identity(dataloader.dataset, global_idx,
                                        epoch_name, slow_frame),
                    ))

                global_idx += 1

    # ── Calibrate the fixed-point gate PER EPOCH and filter ──
    # Round-3 (Reviewer A BLK-3A / B-CRIT-2): each epoch's own residual
    # distribution is fitted with a 2-component GMM + BIC; an epoch
    # without bimodal structure is dropped with a recorded reason.  The
    # previous pooled threshold (a) crashed with NameError when zero
    # candidates were collected, and (b) applied one boundary across
    # epochs whose residual scales may differ by orders of magnitude.
    gate_diagnostics: Dict[str, Dict[str, Any]] = {}

    for epoch_name in EPOCH_DEFINITIONS:
        recs = by_epoch[epoch_name]
        if not recs:
            logger.warning(
                "Epoch '%s': no slow-point candidates collected — "
                "dropped (gate reason: no_candidates).", epoch_name,
            )
            gate_diagnostics[epoch_name] = {
                "gate_status": "unavailable", "reason": "no_candidates",
                "n_candidates": 0, "n_accepted": 0, "n_rejected": 0,
                "accepted_candidates": [],
            }
            continue

        res_epoch = np.array([rec[2] for rec in recs])
        try:
            fp_threshold, diag = _calibrate_fp_threshold(
                res_epoch, context=f"epoch {epoch_name}",
            )
        except ValueError as exc:
            # No bimodal structure / degenerate fit: this epoch provides
            # no evidence for quasi-fixed points.  Drop it LOUDLY —
            # its spectra would be invalid for stability claims.
            logger.warning(
                "Epoch '%s': fixed-point gate REJECTED all %d "
                "candidates — %s",
                epoch_name, len(recs), exc,
            )
            gate_diagnostics[epoch_name] = {
                "gate_status": "unavailable", "reason": str(exc),
                "n_candidates": len(recs), "n_accepted": 0,
                "n_rejected": len(recs), "accepted_candidates": [],
            }
            continue

        keep = _posterior_keep(res_epoch, fp_threshold, diag)
        n_rejected = int((~keep).sum())
        logger.info(
            "Epoch '%s': residual gate %.4g — %d accepted, %d rejected.",
            epoch_name, fp_threshold,
            int(keep.sum()), n_rejected,
        )
        # Preserve legacy scalar-cutoff counts, but distinguish this diagnostic
        # from the fitted posterior accepted set (including unequal-width tails).
        diag["acceptance_rule"] = (
            "low_component_posterior > 0.5 and residual < residual_cap"
        )
        diag["residual_cap"] = FP_RESIDUAL_THRESHOLD_CAP
        diag["threshold_sensitivity_status"] = (
            "scalar_cutoff_diagnostic_not_posterior_gate_robustness"
        )
        diag["threshold_sensitivity_interpretation"] = (
            "n_accept_at_75pct/n_accept_at_125pct count residual < scaled "
            "fp_threshold only; they do not perturb the fitted posterior gate."
        )
        for scale in (0.75, 1.25):
            n_alt = int((res_epoch < fp_threshold * scale).sum())
            diag[f"n_accept_at_{int(scale*100)}pct"] = float(n_alt)
        for accepted, (h_slow, x_slow, residual, identity) in zip(keep, recs):
            if accepted:
                epoch_states[epoch_name].append(h_slow)
                epoch_inputs[epoch_name].append(x_slow)
                accepted_identities[epoch_name].append(identity)
        diag["gate_status"] = "ok"
        diag["n_candidates"] = len(recs)
        diag["accepted_candidates"] = accepted_identities[epoch_name]
        diag["n_accepted"] = int(keep.sum())
        diag["n_rejected"] = n_rejected
        gate_diagnostics[epoch_name] = diag

    result: Dict[str, Any] = {}

    for epoch_name in EPOCH_DEFINITIONS:
        if not epoch_states[epoch_name]:
            continue

        h_stack = torch.stack(epoch_states[epoch_name], dim=0)  # (N, H)
        x_stack = torch.stack(epoch_inputs[epoch_name], dim=0)  # (N, H)

        assert h_stack.shape == x_stack.shape, (
            f"Shape mismatch: h_stack {tuple(h_stack.shape)} != "
            f"x_stack {tuple(x_stack.shape)}"
        )

        result[epoch_name] = (h_stack, x_stack)
        logger.info(
            "  Epoch '%s': extracted %d slow-point states (shape=%s)",
            epoch_name, h_stack.shape[0], tuple(h_stack.shape),
        )

    if not result:
        raise ValueError(
            "No valid states found for any epoch after the per-epoch "
            "fixed-point gates.  Gate diagnostics: "
            + "; ".join(
                f"{name}: {len(by_epoch[name])} candidates" for name in EPOCH_DEFINITIONS
            )
        )

    # Round-3 (BLK-3A): attach per-epoch gate diagnostics under
    # dedicated keys so the caller can persist the calibration evidence.
    for epoch_name, diag in gate_diagnostics.items():
        result[f"{epoch_name}__gate_diag"] = diag

    return result


# ═══════════════════════════════════════════════════════════════
# 5.  Jacobian Eigenvalue Computation
# ═══════════════════════════════════════════════════════════════

def compute_eigenvalues_at_epochs(
    adapter: FixedPointAdapter,
    epoch_data: Dict[str, Any],
    device: torch.device,
    max_states_per_epoch: int = 100,
    sampling_seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]],
           Dict[str, Dict[str, Any]]]:
    """
    Compute Jacobian eigenvalues at each epoch.

    Args:
        adapter: FixedPointAdapter instance.
        epoch_data: Dictionary from :func:`extract_gru_states_at_epochs`.
        device: Computation device.
        max_states_per_epoch: Maximum number of states to process per
            epoch (for computational efficiency).

    Returns:
        ``(eigenvalue_results, attractor_stats, frozen_input_stats)``:
        - ``eigenvalue_results`` maps epoch name to complex eigenvalue
          array (N, H) for EVERY epoch whose own-input spectrum was
          computed (accepted states).  It is retained as raw evidence;
          the release gate in :func:`run_jacobian_analysis` is the sole
          authority on which epochs may be published, because an epoch
          present here may still be withheld by the frozen-input control.
        - ``attractor_stats`` maps epoch name to perturbation-response
          verification counts (Round-2 M-2c: persisted so the attractor
          claim can be audited alongside the spectra).
        - ``frozen_input_stats`` maps epoch name to frozen-input control
          status: ``{"status": "ok", ...}`` for epochs verified under the
          common pooled-median map, or ``{"status": "withheld", ...}``
          (with a reason, finite residual diagnostics and pass counts)
          for epochs whose control failed.
    """
    logger.info("Computing Jacobian eigenvalues at each epoch...")

    # ── Round-1 BLOCKER-2: attractor verification of accepted states ──
    # States already passed the quasi-fixed-point residual gate during
    # extraction; here a subset additionally goes through the
    # perturbation-response test (convergence along top eigendirections)
    # so the report can state whether the verified slow points behave
    # as attractors, not merely as stationary candidates.
    #
    # Round-2 fixes (Reviewer B M-2b/MINOR-E):
    # * n_verified/n_tested are counted PER EPOCH (the previous
    #   cumulative totals were logged under per-epoch names — misleading
    #   from the second epoch on).
    # * The verified subset is drawn RANDOMLY (seeded) rather than
    #   taking the first min(10, N) states, which inherited
    #   batch-ordering selection bias.
    attractor_stats: Dict[str, Dict[str, Any]] = {}
    rng_check = np.random.default_rng(sampling_seed)
    # Round-3 (BLK-3A): skip gate-diagnostic entries — epoch_data now
    # carries "<epoch>__gate_diag" keys alongside state tensors.
    state_epochs = {
        name: val for name, val in epoch_data.items()
        if not name.endswith("__gate_diag") and val[0].shape[0] > 0
    }
    if not state_epochs:
        raise ValueError("GRU eigenvalue spectrum unavailable: no accepted states")
    _sample_indices(0, max_states_per_epoch, sampling_seed)
    for epoch_name, (h_states_v, x_inputs_v) in state_epochs.items():
        Nv = h_states_v.shape[0]
        n_check = min(10, Nv)
        check_idx = rng_check.choice(Nv, size=n_check, replace=False)
        ep_verified = 0
        for vi in check_idx:
            is_att, max_res, _mono = adapter.test_attractor_convergence(
                h_states_v[int(vi)].to(device), x_inputs_v[int(vi)].to(device),
            )
            if is_att:
                ep_verified += 1
        gate = epoch_data.get(f"{epoch_name}__gate_diag", {})
        identities = gate.get("accepted_candidates")
        attractor_stats[epoch_name] = {
            "n_tested": int(n_check),
            "n_verified": int(ep_verified),
            "fraction": float(ep_verified / n_check) if n_check else 0.0,
            "sampling_seed": sampling_seed,
            "attractor_state_indices": [int(i) for i in check_idx],
            **{key: gate[key] for key in ("n_candidates", "n_accepted", "n_rejected")
               if key in gate},
        }
        if identities is not None:
            if len(identities) != Nv:
                raise ValueError("Accepted candidate identifiers are misaligned")
            attractor_stats[epoch_name]["attractor_candidates"] = [
                identities[int(i)] for i in check_idx
            ]
        # Round-3 (Reviewer B MINOR-3): n=10 cannot support a bare
        # fraction claim — attach a Wilson score interval (Wilson 1927)
        # so the reported proportion carries its sampling uncertainty.
        if n_check > 0:
            z95 = 1.959963984540054
            p = ep_verified / n_check
            denom = 1.0 + z95**2 / n_check
            centre = (p + z95**2 / (2 * n_check)) / denom
            half = (
                z95 * math.sqrt(p * (1 - p) / n_check
                                + z95**2 / (4 * n_check**2)) / denom
            )
            attractor_stats[epoch_name]["wilson_ci_95"] = [
                float(max(0.0, centre - half)),
                float(min(1.0, centre + half)),
            ]
        logger.info(
            "  Epoch '%s': attractor verification %d/%d passed "
            "(perturbation-response test)%s.",
            epoch_name, ep_verified, n_check,
            f" CI95={attractor_stats[epoch_name]['wilson_ci_95']}"
            if n_check else "",
        )

    eigenvalue_results: Dict[str, np.ndarray] = {}

    for epoch_name, (h_states, x_inputs) in state_epochs.items():
        N = h_states.shape[0]
        H = h_states.shape[1]

        logger.info(
            "  Epoch '%s': %d states (H=%d), processing up to %d...",
            epoch_name, N, H, max_states_per_epoch,
        )

        indices = _sample_indices(N, max_states_per_epoch, sampling_seed)
        h_sub, x_sub = h_states[indices], x_inputs[indices]
        assert h_sub.shape == x_sub.shape == (len(indices), H)
        stats = attractor_stats[epoch_name]
        stats["n_sampled"] = len(indices)
        stats["selected_state_indices"] = indices.tolist()
        identities = epoch_data.get(f"{epoch_name}__gate_diag", {}).get(
            "accepted_candidates",
        )
        if identities is not None:
            stats["selected_candidates"] = [identities[i] for i in indices.tolist()]

        N_sub = h_sub.shape[0]

        # ── Prepare for Jacobian computation ──────────────────
        # Move to device and enable gradients
        h_sub = h_sub.to(device).requires_grad_(True)
        x_sub = x_sub.to(device)

        # ── Compute Jacobians in batches ──────────────────────
        batch_size = 32  # Process in smaller batches for memory
        all_eigenvalues: List[np.ndarray] = []

        for start_idx in range(0, N_sub, batch_size):
            end_idx = min(start_idx + batch_size, N_sub)
            h_batch = h_sub[start_idx:end_idx]
            x_batch = x_sub[start_idx:end_idx]

            # Compute Jacobians for this batch
            J_batch = adapter.compute_jacobian_batch(h_batch, x_batch)  # (B, H, H)
            if not isinstance(J_batch, torch.Tensor):
                J_batch = torch.from_numpy(np.asarray(J_batch)).to(
                    device=h_batch.device, dtype=torch.float32,
                )

            # ── Shape assertion ───────────────────────────────
            assert J_batch.shape == (h_batch.shape[0], H, H), (
                f"J_batch shape {tuple(J_batch.shape)} != "
                f"({h_batch.shape[0]}, {H}, {H})"
            )

            # ── Extract eigenvalues ───────────────────────────
            # Stay on torch.linalg.eigvals: fused JAX jac+eig is slower
            # than jacfwd + PyTorch eig on this GPU (measured 2026-09-03).
            eigvals = torch.linalg.eigvals(J_batch)  # (B, H) complex

            # Shape assertion
            assert eigvals.shape == (h_batch.shape[0], H), (
                f"eigvals shape {tuple(eigvals.shape)} != ({h_batch.shape[0]}, {H})"
            )

            # Move to CPU and convert to numpy
            all_eigenvalues.append(eigvals.cpu().numpy())

        # Concatenate all eigenvalues for this epoch
        epoch_eigvals = np.concatenate(all_eigenvalues, axis=0)  # (N_total, H)

        # ── Shape and type assertions ─────────────────────────
        assert epoch_eigvals.shape[1] == H, (
            f"Eigenvalue dim {epoch_eigvals.shape[1]} != H={H}"
        )
        assert np.iscomplexobj(epoch_eigvals), (
            f"Eigenvalues should be complex, got dtype={epoch_eigvals.dtype}"
        )

        eigenvalue_results[epoch_name] = epoch_eigvals

        # NOTE (release-audit fix): own-input eigenvalue statistics are
        # deliberately NOT logged here.  At this point the frozen-input
        # control has not yet run, so an epoch may still be withheld;
        # logging its magnitudes would leak unverified spectra into
        # run.log even when the control later fails.  Magnitudes are
        # emitted only after the control gate, for controlled epochs.

    # ── Round-1 fix (Reviewer B MAJOR-1): frozen-input control ──
    # Re-evaluate every epoch's states under ONE common input (the
    # epoch-pooled median e_sensory).  Spectral differences surviving
    # this control are attributable to STATE changes, not to
    # input-driven shifts of the map.
    #
    # Round-2 fix (Reviewer B M-2d): under the FROZEN input the states
    # are generally no longer quasi-fixed points of the modified map
    # (the gate was passed under each state's own input).  The control
    # spectra must therefore be re-gated with the residual check under
    # x_frozen; states failing it are excluded and the exclusion is
    # reported.  Without this, frozen-input spectra of transient states
    # would invite exactly the misreading the control was designed to
    # rule out.
    frozen_input_stats: Dict[str, Dict[str, Any]] = {}
    all_x = torch.cat([x for (_h, x) in state_epochs.values()], dim=0)
    x_frozen = all_x.median(dim=0).values.to(device)  # (H,)
    logger.info(
        "Frozen-input control: re-evaluating spectra under the "
        "pooled-median e_sensory (input dependence removed); "
        "states re-gated against the frozen map."
    )
    for epoch_name, (h_states_c, _x_inputs_c) in state_epochs.items():
        N_c = h_states_c.shape[0]

        # Re-check quasi-fixed-point residuals under the frozen map.
        # Round-3 (B-CRIT-2): the frozen map is a DIFFERENT map from
        # the one each state's gate was calibrated on, so its
        # residuals get their OWN GMM+BIC calibration.  The Round-2
        # code referenced a loop-local fp_threshold here that no
        # longer existed after the Round-3 per-epoch refactor — a
        # latent NameError, not a gate.  If the frozen-map residual
        # distribution shows no bimodal structure the spectrum is
        # withheld with the recorded reason.
        res_frozen = []
        for hi in range(N_c):
            res_frozen.append(float(_fixed_point_residual(
                adapter, h_states_c[hi].to(device), x_frozen,
            )))
        res_frozen = np.array(res_frozen)
        try:
            frozen_threshold, frozen_diag = _calibrate_fp_threshold(
                res_frozen,
                context=f"frozen-input {epoch_name}",
            )
        except ValueError as exc:
            logger.warning(
                "    [frozen-input] %s: re-gate failed — %s "
                "(spectrum withheld).",
                epoch_name, exc,
            )
            withheld_entry: Dict[str, Any] = {
                "status": "withheld",
                "withheld_reason": "frozen_map_gate_unavailable",
                "withheld_epochs": [epoch_name],
                "gate_reason": str(exc),
                "n_pass": 0, "n_total": int(N_c),
            }
            # Residual diagnostics are publishable only when finite.  The
            # gate rejected this epoch precisely because the residual
            # array was degenerate — which includes the NaN/Inf case that
            # ``_calibrate_fp_threshold`` refuses (nonfinite residuals are
            # why the frozen map has no usable stationary set).  Summarizing
            # that same array would fabricate NaN/Infinity into the artifact,
            # so the summaries are omitted (published downstream as an
            # explicit null) rather than emitted as nonfinite numbers.
            if np.all(np.isfinite(res_frozen)):
                withheld_entry.update(
                    residual_median=float(np.median(res_frozen)),
                    residual_min=float(res_frozen.min()),
                    residual_max=float(res_frozen.max()),
                )
            frozen_input_stats[epoch_name] = withheld_entry
            continue
        keep_mask = _posterior_keep(res_frozen, frozen_threshold, frozen_diag)
        logger.info(
            "    [frozen-input] %s: %d/%d states pass the residual "
            "gate under the frozen map (median residual %.4f).",
            epoch_name, int(keep_mask.sum()), N_c,
            float(np.median(res_frozen)),
        )

        if keep_mask.sum() < 2:
            logger.warning(
                "    [frozen-input] %s: too few stationary states "
                "under the frozen map — spectrum withheld.",
                epoch_name,
            )
            frozen_input_stats[epoch_name] = {
                "status": "withheld",
                "withheld_reason": "insufficient_stationary_states",
                "withheld_epochs": [epoch_name],
                "n_pass": int(keep_mask.sum()), "n_total": int(N_c),
                "residual_median": float(np.median(res_frozen)),
            }
            continue

        h_keep = h_states_c.to(device)[torch.from_numpy(
            np.nonzero(keep_mask)[0]).to(device)]
        x_rep = x_frozen.unsqueeze(0).expand(h_keep.shape[0], -1)
        J_frozen = adapter.compute_jacobian_batch(h_keep, x_rep)
        if not isinstance(J_frozen, torch.Tensor):
            J_frozen = torch.from_numpy(np.asarray(J_frozen)).to(
                device=h_keep.device, dtype=torch.float32,
            )
        eig_frozen = torch.linalg.eigvals(J_frozen)
        mags_frozen = eig_frozen.abs().flatten()
        frozen_input_stats[epoch_name] = {
            "status": "ok",
            "n_pass": int(keep_mask.sum()),
            "n_total": int(N_c),
            "mag_mean": float(mags_frozen.mean()),
            "mag_max": float(mags_frozen.max()),
            # Round-3 (B-CRIT-2): persist the frozen-map gate
            # threshold so the re-gating is auditable.
            "frozen_gate_threshold": float(frozen_threshold),
        }
        logger.info(
            "    [frozen-input] %s: |λ|_mean=%.4f, |λ|_max=%.4f",
            epoch_name, float(mags_frozen.mean()), float(mags_frozen.max()),
        )

    return eigenvalue_results, attractor_stats, frozen_input_stats


# ═══════════════════════════════════════════════════════════════
# 5b. Full System Jacobian (CF5: LIF + GRU + Router)
# ═══════════════════════════════════════════════════════════════

def compute_full_system_eigenvalues(
    model: NSMoRCore,
    adapter: FixedPointAdapter,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    onset_frames: List[Optional[int]],
    target_class: int = Label.PREWALK.value,
    dt_ms: Optional[float] = None,
    max_states_per_epoch: int = 100,
    sampling_seed: int = 42,
    selection_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Compute surrogate input singular values; retain the legacy function name.

    The rectangular dh_out/dx uses LIF surrogate gradients and zero initial
    recurrent state. These values describe local input sensitivity, not
    recurrent eigenvalues or attractor stability.

    Args:
        model: Trained NSMoRCore.
        adapter: Full-system Jacobian adapter.
        dataloader: Ordered trial loader.
        device: Computation device.
        onset_frames: Raw per-trial references (wind onset or visual collision).
        target_class: Behavioral class to select.
        dt_ms: Saved cadence or a matching explicit override.
        max_states_per_epoch: Positive computation cap.
        sampling_seed: Local sampling seed, independent of global RNG.
        selection_metadata: Optional output dictionary for actual selection/counts.

    Returns:
        Per-epoch singular values shaped (N, min(H, F)).

    Raises:
        ValueError: When references or every candidate spectrum are unavailable.
    """
    dt_ms = resolve_dt_ms(model, dt_ms)
    _sample_indices(0, max_states_per_epoch, sampling_seed)
    candidates: Dict[str, List[Tuple[torch.Tensor, Dict[str, Any]]]] = {
        name: [] for name in EPOCH_DEFINITIONS
    }
    model.eval()
    global_idx = 0
    for X_batch, _Y_batch, lengths in dataloader:
        B, T, F = X_batch.shape
        assert X_batch.shape == (B, T, 8)
        assert lengths.shape == (B,)
        for i in range(B):
            trial_index = global_idx
            global_idx += 1
            _, _, label = dataloader.dataset.sequences[trial_index]
            if int(label) != target_class:
                continue
            length = int(lengths[i])
            reference = _epoch_reference_frame(
                dataloader.dataset, trial_index, X_batch[i, :length], onset_frames,
            )
            for name, definition in EPOCH_DEFINITIONS.items():
                frame = reference + int(definition["offset_ms"] / dt_ms)
                if 0 <= frame < length - 1:
                    x = X_batch[i, frame:frame + 1].unsqueeze(0).detach().cpu()
                    assert x.shape == (1, 1, F)
                    candidates[name].append((
                        x, _trial_identity(dataloader.dataset, trial_index, name, frame),
                    ))

    result: Dict[str, np.ndarray] = {}
    for name, records in candidates.items():
        indices = _sample_indices(len(records), max_states_per_epoch, sampling_seed)
        values: List[np.ndarray] = []
        accepted: List[Dict[str, Any]] = []
        for index in indices.tolist():
            x, identity = records[index]
            try:
                with torch.enable_grad():
                    jacobian, _ = adapter.compute_full_system_jacobian(
                        x.to(device), torch.tensor([1], device=device),
                    )
                assert jacobian.shape == (model.hidden_dim, x.shape[-1])
                singular = torch.linalg.svdvals(jacobian.detach())
                assert singular.shape == (min(model.hidden_dim, x.shape[-1]),)
                if not torch.isfinite(singular).all():
                    raise ValueError("Nonfinite input sensitivity singular values")
                values.append(singular.cpu().numpy())
                accepted.append(identity)
            except Exception as exc:
                logger.warning("Full-system candidate %s failed: %s",
                               identity["candidate_id"], exc)
        if selection_metadata is not None:
            selection_metadata[name] = {
                "sampling_seed": sampling_seed, "n_candidates": len(records),
                "n_sampled": len(indices), "n_accepted": len(values),
                "n_rejected": len(indices) - len(values),
                "n_unsampled": len(records) - len(indices),
                "selected_state_indices": indices.tolist(),
                "selected_candidates": [records[i][1] for i in indices.tolist()],
                "accepted_candidates": accepted,
                "acceptance_rule": "finite_singular_values; not_fixed_point_gated",
            }
        if values:
            result[name] = np.stack(values)
    if not result:
        raise ValueError("Full-system input sensitivity unavailable: no valid spectra")
    return result


# ═══════════════════════════════════════════════════════════════
# 6.  Lancet/Cell Publication Figure
# ═══════════════════════════════════════════════════════════════

def setup_lancet_style() -> None:
    """Configure matplotlib for Lancet/Cell publication aesthetics."""
    plt.rcParams.update({
        # ── Font ──
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE_TICK,
        "axes.titlesize": FONT_SIZE_AXIS_TITLE,
        "axes.labelsize": FONT_SIZE_AXIS_TITLE,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,

        # ── Axes ──
        "axes.linewidth": 1.5,
        "axes.edgecolor": AXIS_COLOR,
        "axes.labelcolor": AXIS_COLOR,
        "xtick.color": AXIS_COLOR,
        "ytick.color": AXIS_COLOR,

        # ── Grid ──
        "axes.grid": False,
        "grid.alpha": 0.15,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,

        # ── Figure ──
        "figure.facecolor": BACKGROUND_COLOR,
        "savefig.facecolor": BACKGROUND_COLOR,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,

        # ── Legend ──
        "legend.frameon": True,
        "legend.facecolor": BACKGROUND_COLOR,
        "legend.edgecolor": AXIS_COLOR,
        "legend.framealpha": 1.0,
    })


def plot_eigenvalue_spectrum(
    eigenvalue_results: Dict[str, np.ndarray],
    output_path: Path,
    analysis_population: Optional[Dict[str, object]] = None,
    withheld_epochs: Optional[Sequence[str]] = None,
) -> None:
    """
    Plot the Jacobian eigenvalue spectrum on the complex plane.

    Creates a ``subplots(1, 3)`` figure with one panel per epoch.
    Each panel uses a **hexbin density heatmap** to handle massive
    point overlap, with the unit circle reference overlaid.

    Args:
        eigenvalue_results: Dictionary from :func:`compute_eigenvalues_at_epochs`,
            containing ONLY the controlled (publishable) epochs.
        output_path: Path to save the figure.
        analysis_population: Descriptive population metadata.
        withheld_epochs: Epochs withheld by the frozen-input control.  Their
            panels state the withholding explicitly; no spectrum is drawn.
    """
    if not any(values.size for values in eigenvalue_results.values()):
        raise ValueError("GRU eigenvalue spectrum unavailable")
    withheld = set(withheld_epochs or ())
    setup_lancet_style()

    # ── Create figure with [1, 3] layout ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))

    # Panel labels
    panel_labels = ["A", "B", "C"]

    for idx, (epoch_name, epoch_def) in enumerate(EPOCH_DEFINITIONS.items()):
        ax = axes[idx]

        if epoch_name in withheld:
            logger.warning(
                "Epoch '%s' withheld (frozen-input control failed); "
                "no spectrum drawn.", epoch_name,
            )
            ax.text(
                0.5, 0.55, "Withheld",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_SIZE_PANEL_LABEL, fontweight="bold",
                color=AXIS_COLOR,
            )
            ax.text(
                0.5, 0.38,
                "frozen-input control failed;\nno verified spectrum",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_SIZE_LEGEND, color=AXIS_COLOR,
            )
            ax.set_title(
                f"{panel_labels[idx]}  "
                f"{epoch_def['label'].replace('onset', 'reference')}",
                fontsize=FONT_SIZE_PANEL_LABEL, fontweight="bold",
                color=AXIS_COLOR, loc="left",
            )
            continue

        if epoch_name not in eigenvalue_results:
            logger.warning("No eigenvalues for epoch '%s', skipping.", epoch_name)
            ax.text(
                0.5, 0.5, "No data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_SIZE_AXIS_TITLE, color=AXIS_COLOR,
            )
            continue

        eigvals = eigenvalue_results[epoch_name]  # (N, H) complex

        # Flatten to 1D for plotting
        eigvals_flat = eigvals.flatten()
        real_parts = np.real(eigvals_flat)
        imag_parts = np.imag(eigvals_flat)

        # ── Draw unit circle ──────────────────────────────────
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(
            np.cos(theta), np.sin(theta),
            color=UNIT_CIRCLE_COLOR,
            linewidth=UNIT_CIRCLE_LINEWIDTH,
            linestyle=UNIT_CIRCLE_LINESTYLE,
            label="Unit circle",
            zorder=4,
        )

        # ── Task 4: Density-based spectrum visualization ──────
        # Use hexbin for density heatmap when many points,
        # fall back to scatter for small datasets.
        n_points = len(eigvals_flat)

        if n_points > 200:
            # Hexbin density heatmap
            hb = ax.hexbin(
                real_parts, imag_parts,
                gridsize=HEXBIN_GRIDSIZE,
                cmap=CMAP_HEATMAP,
                mincnt=1,
                linewidths=0.2,
                edgecolors="face",
                alpha=0.85,
                zorder=3,
            )
            # Colour bar
            cb = fig.colorbar(hb, ax=ax, shrink=0.78, pad=0.02)
            cb.set_label("Count", fontsize=FONT_SIZE_LEGEND, color=AXIS_COLOR)
            cb.ax.tick_params(labelsize=FONT_SIZE_LEGEND - 1, colors=AXIS_COLOR)
            for spine in cb.ax.spines.values():
                spine.set_color(AXIS_COLOR)

            # Legend entry for hexbin (manual proxy)
            from matplotlib.patches import Patch
            hex_proxy = Patch(facecolor=plt.get_cmap(CMAP_HEATMAP)(0.6), label="Density")
            unit_proxy = plt.Line2D([0], [0], color=UNIT_CIRCLE_COLOR,
                                    linewidth=UNIT_CIRCLE_LINEWIDTH,
                                    linestyle=UNIT_CIRCLE_LINESTYLE,
                                    label="Unit circle")
            ax.legend(
                handles=[hex_proxy, unit_proxy],
                loc="upper left",
                fontsize=FONT_SIZE_LEGEND,
                frameon=True,
                facecolor=BACKGROUND_COLOR,
                edgecolor=AXIS_COLOR,
                framealpha=1.0,
            )
        else:
            # Scatter for small datasets
            ax.scatter(
                real_parts, imag_parts,
                color=EIGENVALUE_COLOR,
                s=EIGENVALUE_SIZE,
                alpha=EIGENVALUE_ALPHA,
                edgecolors="white",
                linewidths=0.3,
                label=r"Eigenvalues ($\lambda$)",
                zorder=3,
            )
            ax.legend(
                loc="upper left",
                fontsize=FONT_SIZE_LEGEND,
                frameon=True,
                facecolor=BACKGROUND_COLOR,
                edgecolor=AXIS_COLOR,
                framealpha=1.0,
            )

        # ── Reference lines (real=1, imaginary=0) ─────────────
        ax.axvline(
            x=1.0,
            color=UNIT_CIRCLE_COLOR,
            linewidth=0.8,
            linestyle=":",
            alpha=0.5,
        )
        ax.axhline(
            y=0.0,
            color=UNIT_CIRCLE_COLOR,
            linewidth=0.8,
            linestyle=":",
            alpha=0.5,
        )

        # ── Axes styling ──────────────────────────────────────
        ax.set_xlabel(
            r"Re($\lambda$)",
            fontsize=FONT_SIZE_AXIS_TITLE,
            color=AXIS_COLOR,
        )
        ax.set_ylabel(
            r"Im($\lambda$)",
            fontsize=FONT_SIZE_AXIS_TITLE,
            color=AXIS_COLOR,
        )

        # Set equal aspect ratio for proper circle visualization
        ax.set_aspect("equal", adjustable="datalim")

        # Set axis limits (with some padding)
        max_mag = max(np.max(np.abs(real_parts)), np.max(np.abs(imag_parts)))
        limit = min(max_mag * 1.2, 2.0)  # Cap at 2.0 for readability
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

        # Tick formatting
        ax.tick_params(axis="both", colors=AXIS_COLOR, width=1.5)

        # Spine styling
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOR)
            spine.set_linewidth(1.5)

        # Grid: ultra-faint major grid lines
        ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)

        # Panel label and epoch title
        ax.set_title(
            f"{panel_labels[idx]}  {epoch_def['label'].replace('onset', 'reference')}",
            fontsize=FONT_SIZE_PANEL_LABEL,
            fontweight="bold",
            color=AXIS_COLOR,
            loc="left",
        )

        # ── Add statistics annotation ─────────────────────────
        n_eigenvalues = len(eigvals_flat)
        mean_magnitude = np.mean(np.abs(eigvals_flat))
        pct_near_unity = np.mean(np.abs(np.abs(eigvals_flat) - 1.0) < 0.1) * 100

        stats_text = (
            f"n = {n_eigenvalues}\n"
            f"|λ|$_{{mean}}$ = {mean_magnitude:.3f}\n"
            f"% near |λ|=1: {pct_near_unity:.1f}%"
        )
        ax.text(
            0.97, 0.03, stats_text,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=FONT_SIZE_LEGEND,
            color=AXIS_COLOR,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=BACKGROUND_COLOR,
                edgecolor=UNIT_CIRCLE_COLOR,
                alpha=0.9,
            ),
        )

    # ── Suptitle ──────────────────────────────────────────────
    # Round-1 fix (Reviewer B MAJOR-2): the figure title must state
    # the mathematical scope explicitly.  This figure shows EXACT
    # autograd Jacobian eigenvalues of the GRU pathway only; any
    # full-system surrogate-gradient singular values belong to a
    # separate figure and carry no stability interpretation.
    fig.suptitle(
        "Jacobian Eigenvalue Spectrum — GRU Pathway (exact, "
        "autograd ∂h$_{t+1}$/∂h$_t$); "
        + (analysis_population["selection"].replace("_", " ") + ", "
           + analysis_population["evidence_scope"].replace("_", " ")
           if analysis_population is not None else "descriptive population unavailable"),
        fontsize=14,
        fontweight="bold",
        color=AXIS_COLOR,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # ── Save ──────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.1,
        metadata={"Description": json.dumps({"analysis_population": analysis_population})}
        if analysis_population is not None else None,
    )
    logger.info("Saved Jacobian spectrum figure to %s (%d DPI)", output_path, DPI)

    plt.close(fig)


def plot_withheld_spectrum_notice(
    output_path: Path,
    reason: str,
    analysis_population: Optional[Dict[str, object]] = None,
) -> None:
    """Render an explicit withheld-spectra figure instead of an empty file.

    When the frozen-input control fails, the own-input eigenvalues are NOT a
    publishable spectrum (their map was not verified under a common input),
    so the figure must say so rather than omit the PNG or plot the unverified
    points.  The remaining finite diagnostics live in the JSON sidecar.
    """
    setup_lancet_style()
    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))
    ax.axis("off")
    ax.text(
        0.5, 0.62, "Jacobian spectra WITHHELD",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=FONT_SIZE_PANEL_LABEL, fontweight="bold", color=AXIS_COLOR,
    )
    ax.text(
        0.5, 0.42,
        "The frozen-input control did not verify a quasi-fixed-point\n"
        "population under a common map; own-input eigenvalues are not\n"
        "a publishable spectrum and are intentionally not shown.",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=FONT_SIZE_AXIS_TITLE, color=AXIS_COLOR,
    )
    ax.text(
        0.5, 0.18, f"reason: {reason}",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=FONT_SIZE_LEGEND, color=AXIS_COLOR, wrap=True,
    )
    fig.suptitle(
        "Jacobian Eigenvalue Spectrum — GRU Pathway (frozen-input control failed)",
        fontsize=14, fontweight="bold", color=AXIS_COLOR, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.1,
        metadata={"Description": json.dumps({
            "spectral_quantity": "withheld",
            "stability_interpretation": False,
            "reason": reason,
            "analysis_population": analysis_population,
        })},
    )
    logger.warning(
        "Frozen-input control failed — withheld-spectra notice written to %s.",
        output_path,
    )
    plt.close(fig)


def plot_singular_value_spectrum(
    singular_results: Dict[str, np.ndarray],
    output_path: Path,
    analysis_population: Optional[Dict[str, object]] = None,
) -> None:
    """Plot surrogate input-sensitivity distributions, without a unit circle."""
    if not any(values.size for values in singular_results.values()):
        raise ValueError("Input sensitivity singular values unavailable")
    setup_lancet_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))
    for ax, (name, definition) in zip(axes, EPOCH_DEFINITIONS.items()):
        ax.set_xlabel("Input sensitivity singular value (σ)")
        ax.set_ylabel("Count")
        ax.set_title(definition["label"].replace("onset", "reference"))
        values = singular_results.get(name)
        if values is None or not values.size:
            ax.text(0.5, 0.5, "Unavailable", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        ax.hist(values.ravel(), bins="auto", color=EIGENVALUE_COLOR,
                edgecolor=BACKGROUND_COLOR, linewidth=1.0)
        ax.set_xlim(left=0)
    fig.suptitle(
        "Full-system surrogate input sensitivity — singular values\n"
        "Zero initial recurrent state; no stability interpretation",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path, dpi=DPI, bbox_inches="tight",
        metadata={"Description": json.dumps({
            "spectral_quantity": "surrogate_input_singular_values",
            "stability_interpretation": False,
            "analysis_population": analysis_population,
        })},
    )
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 7.  Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════

def run_jacobian_analysis(
    checkpoint_path: Path,
    dataset_path: Path,
    output_path: Path,
    target_class: int = Label.PREWALK.value,
    batch_size: int = 32,
    max_seq_len: Optional[int] = 2400,
    pre_anchor_frames: int = 1200,
    dt_ms: Optional[float] = None,
    max_states_per_epoch: int = 100,
    full_system: bool = False,
    backend: str = "jax",
    nested_prior_artifact: Optional[Path] = None,
    qc_sealed_nested_prior_sha256: Optional[str] = None,
    trusted_historical_checkpoint_sha256: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
    sampling_seed: int = 42,
) -> None:
    """
    Run the full Jacobian eigenvalue spectrum analysis.

    Args:
        checkpoint_path: Path to the trained model checkpoint.
        dataset_path: Path to the preprocessed dataset.
        output_path: Path to save the spectrum figure.
        target_class: Label value to analyze (default: PREWALK=1,
            sustained locomotion — required by the "Sustained epoch"
            line-attractor hypothesis).
        batch_size: Batch size for data loading.
        max_seq_len: Maximum sequence length for cropping.
        pre_anchor_frames: Baseline frames before anchor.
        dt_ms: None inherits saved model.dt_ms; an explicit interval must match.
        max_states_per_epoch: Maximum states to process per epoch.
        backend: ``"jax"`` uses the measured-faster GRU Jacobian kernel
            (falls back to PyTorch if JAX is missing). ``"torch"``
            forces autograd. Eigenvalues stay on PyTorch either way —
            the fused JAX eig path is slower on this hardware.
    """
    logger.info("=" * 60)
    logger.info("NSMoR Jacobian Eigenvalue Spectrum Analysis (Phase 8)")
    logger.info("=" * 60)

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────
    model = load_model_from_checkpoint(checkpoint_path, device)
    dt_ms = resolve_dt_ms(model, dt_ms)

    # ── Load dataset (returns raw X_seqs for onset detection) ─
    dataloader, labels, lengths_list, X_seqs = load_dataset(
        dataset_path,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        nested_prior_artifact=nested_prior_artifact,
        qc_sealed_nested_prior_sha256=qc_sealed_nested_prior_sha256,
        checkpoint_model=model,
        trusted_historical_checkpoint_sha256=trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
    )

    # ── Task 1: Dynamic stimulus onset detection ──────────────
    onset_frames = detect_stimulus_onset_frames(X_seqs, dt_ms=dt_ms)

    # ── Initialize adapter (JAX Jacobian only if requested) ──
    adapter = create_jacobian_adapter(model, device=device, backend=backend)
    logger.info(
        "Jacobian backend: requested=%s resolved=%s",
        backend,
        type(adapter).__name__,
    )

    population = population_for_output(dataloader, len(labels))
    if full_system:
        selection: Dict[str, Any] = {}
        summary: Dict[str, Any] = {
            "status": "unavailable",
            "spectral_quantity": "surrogate_input_singular_values",
            "stability_interpretation": False,
            "recurrent_state": "zero_initial; not_trajectory_conditioned",
            "analysis_population": population,
            "target_class": int(target_class), "dt_ms": dt_ms,
            "sampling_seed": sampling_seed, "max_states_per_epoch": max_states_per_epoch,
            "analysis_reference_rules": getattr(
                dataloader.dataset, "analysis_reference_rules", None,
            ),
            "spectral_statistics": {}, "selection": selection,
        }
        json_path = output_path.with_suffix(".json")
        try:
            singular_results = compute_full_system_eigenvalues(
                model=model, adapter=adapter, dataloader=dataloader,
                device=device, onset_frames=onset_frames,
                target_class=target_class, dt_ms=dt_ms,
                max_states_per_epoch=max_states_per_epoch,
                sampling_seed=sampling_seed, selection_metadata=selection,
            )
            if not any(values.size for values in singular_results.values()):
                raise ValueError("Full-system input sensitivity unavailable")
        except ValueError as exc:
            summary["reason"] = str(exc)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(summary, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            raise
        summary["status"] = "ok"
        summary["spectral_statistics"] = {
            name: {"singular_mean": float(values.mean()),
                   "singular_max": float(values.max()),
                   "n_states": int(values.shape[0])}
            for name, values in singular_results.items()
        }
        plot_singular_value_spectrum(singular_results, output_path, population)
        json_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8",
        )
        return

    # GRU-only spectra require the actual residual gate; input sensitivity does not.
    epoch_data_full = extract_gru_states_at_epochs(
        model=model, dataloader=dataloader, device=device,
        onset_frames=onset_frames, target_class=target_class,
        dt_ms=dt_ms, adapter=adapter,
    )
    gate_diagnostics = {
        name: val for name, val in epoch_data_full.items()
        if name.endswith("__gate_diag")
    }
    eigenvalue_results, attractor_stats, frozen_input_stats = (
        compute_eigenvalues_at_epochs(
            adapter=adapter, epoch_data=epoch_data_full, device=device,
            max_states_per_epoch=max_states_per_epoch, sampling_seed=sampling_seed,
        )
    )

    # ── Frozen-input control gate (adversarial release audit) ──
    # The control exists to separate state-dependence from input-
    # dependence.  An epoch whose control failed or was withheld has NO
    # verified spectrum under a common map, so its own-input eigenvalues
    # must not appear in the JSON, the figure, or run.log.  This gate
    # therefore filters the published subset to the controlled epochs;
    # the ALL-failed case is an explicit withholding with an empty
    # spectral_statistics, and a MIXED case publishes only the
    # controlled epochs while marking the artifact partial and naming
    # the withheld epochs (without their spectra).
    controlled_epochs = [
        name for name, diag in frozen_input_stats.items()
        if diag.get("status") == "ok"
    ]
    withheld_epochs = [
        name for name in EPOCH_DEFINITIONS
        if frozen_input_stats.get(name, {}).get("status") != "ok"
    ]
    # Withheld epochs are identified by reason and finite residual
    # diagnostics only — never by their spectra.
    withheld_spectra: Dict[str, Dict[str, Any]] = {}
    for name in withheld_epochs:
        diag = frozen_input_stats.get(name, {})
        withheld_spectra[name] = {
            "withheld_reason": diag.get(
                "withheld_reason", "frozen_input_control_unavailable",
            ),
            "n_pass": int(diag.get("n_pass", 0)),
            "n_total": int(diag.get("n_total", 0)),
            "residual_median": diag.get("residual_median"),
        }

    if not controlled_epochs:
        json_path = output_path.with_suffix(".json")
        reason = (
            "frozen_input_control_failed: no epoch produced a "
            "quasi-fixed-point population under the pooled-median "
            "e_sensory map, so own-input eigenvalues are not a "
            "publishable spectrum."
        )
        withheld_summary = {
            "status": "withheld",
            "spectral_quantity": "gru_jacobian_eigenvalues_withheld",
            "spectral_statistics": {},
            "stability_interpretation": False,
            "withheld_reason": reason,
            "controlled_epochs": [],
            "withheld_epochs": withheld_epochs,
            "withheld_spectra": withheld_spectra,
            "analysis_population": population,
            "target_class": int(target_class),
            "sampling_seed": sampling_seed,
            "max_states_per_epoch": max_states_per_epoch,
            "analysis_reference_rules": getattr(
                dataloader.dataset, "analysis_reference_rules", None,
            ),
            "dt_ms": dt_ms,
            "epochs": EPOCH_DEFINITIONS,
            # Honest, finite diagnostics survive the withholding: the
            # per-epoch frozen-map gate outcome (n_pass/n_total and
            # residual range) and the own-input gate calibration remain
            # auditable without publishing a spectrum.
            "frozen_input_control": frozen_input_stats,
            "fp_gate_calibration": gate_diagnostics,
        }
        plot_withheld_spectrum_notice(output_path, reason, population)
        with open(json_path, "w") as f:
            json.dump(withheld_summary, f, indent=2, allow_nan=False)
        logger.warning(
            "Jacobian spectra WITHHELD — frozen-input control passed "
            "0 epochs; own-input spectral_statistics not published. "
            "Summary → %s", json_path,
        )
        logger.info("=" * 60)
        logger.info("Jacobian analysis complete (spectra withheld)!")
        logger.info("=" * 60)
        return

    # Mixed case: some epochs are controlled, others withheld.  Publish
    # only the controlled subset and mark the artifact explicitly.
    partial = bool(withheld_epochs)
    status = "partial" if partial else "ok"

    # ── Log GRU spectral statistics (controlled epochs only) ───
    logger.info(
        "Hypothesis Verification (controlled epochs only):"
        if partial else "Hypothesis Verification:",
    )
    for epoch_name in controlled_epochs:
        eigvals = eigenvalue_results[epoch_name]
        magnitudes = np.abs(eigvals.flatten())
        real_parts = np.real(eigvals.flatten())
        logger.info(
            "  %s: |λ|_mean=%.4f, %%near|λ|=1: %.1f%%, %%nearRe(λ)=1: %.1f%%",
            epoch_name, np.mean(magnitudes),
            np.mean(np.abs(magnitudes - 1.0) < 0.1) * 100,
            np.mean(np.abs(real_parts - 1.0) < 0.1) * 100,
        )
    if partial:
        logger.warning(
            "Epochs %s were WITHHELD (frozen-input control failed); "
            "their spectra are excluded from the figure, JSON and "
            "spectral_statistics.", withheld_epochs,
        )

    # ── Create figure (controlled epochs only) ────────────────
    controlled_spectra = {
        name: eigenvalue_results[name] for name in controlled_epochs
    }
    plot_eigenvalue_spectrum(
        eigenvalue_results=controlled_spectra,
        output_path=output_path,
        analysis_population=population,
        withheld_epochs=withheld_epochs,
    )

    # ── Export JSON summary (Round-2 M-2c) ───────────────────
    # The attractor verification and frozen-input control results are
    # persisted next to the figure so spectral conclusions are auditable
    # against the verification evidence, not just logged.  Only the
    # controlled epochs contribute spectral_statistics.
    json_path = output_path.with_suffix(".json")
    epoch_summary: Dict[str, Dict[str, float]] = {}
    for epoch_name in controlled_epochs:
        eigvals = eigenvalue_results[epoch_name]
        magnitudes = np.abs(eigvals.flatten())
        real_parts = np.real(eigvals.flatten())
        epoch_summary[epoch_name] = {
            "mag_mean": float(np.mean(magnitudes)),
            "mag_max": float(np.max(magnitudes)),
            "pct_near_unit_circle": float(
                np.mean(np.abs(magnitudes - 1.0) < 0.1) * 100
            ),
            "pct_near_real_one": float(
                np.mean(np.abs(real_parts - 1.0) < 0.1) * 100
            ),
            "n_states": int(eigvals.shape[0]),
            "n_verified_attractors": attractor_stats.get(epoch_name, {}).get("n_verified", 0),
            "n_tested_attractors": attractor_stats.get(epoch_name, {}).get("n_tested", 0),
        }
    summary = {
        "status": status,
        "spectral_quantity": "gru_jacobian_eigenvalues",
        "stability_interpretation": True,
        "controlled_epochs": controlled_epochs,
        "withheld_epochs": withheld_epochs,
        "withheld_spectra": withheld_spectra,
        "analysis_population": population,
        "target_class": int(target_class),
        "sampling_seed": sampling_seed,
        "max_states_per_epoch": max_states_per_epoch,
        "analysis_reference_rules": getattr(
            dataloader.dataset, "analysis_reference_rules", None,
        ),
        "dt_ms": dt_ms,
        "epochs": EPOCH_DEFINITIONS,
        "spectral_statistics": epoch_summary,
        "attractor_verification": attractor_stats,
        "frozen_input_control": frozen_input_stats,
        # Round-3 (BLK-3A): per-epoch GMM+BIC gate calibration
        # evidence — threshold, BIC values, residual quantiles and
        # accept/reject counts are persisted so the gate is fully
        # auditable.
        "fp_gate_calibration": gate_diagnostics,
    }
    if partial:
        summary["partial_reason"] = (
            "frozen_input_control_partial: spectra published only for "
            f"controlled epochs {controlled_epochs}; withheld epochs "
            f"{withheld_epochs} carry no verified spectrum under the "
            "common map."
        )
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    logger.info("JSON summary saved → %s", json_path)

    logger.info("=" * 60)
    logger.info(
        "Jacobian analysis complete (partial: controlled subset only)!"
        if partial else "Jacobian analysis complete!"
    )
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
# 8.  CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="NSMoR Jacobian Eigenvalue Spectrum Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/nsmor_dataset.pt",
        help="Path to preprocessed dataset.",
    )
    parser.add_argument(
        "--nested_prior_artifact",
        type=str,
        default=None,
        help="Optional validated nested prior artifact used by the checkpoint.",
    )
    parser.add_argument(
        "--qc_sealed_nested_prior_sha256", type=str, default=None,
        help="Optional external QC-sealed SHA-256 to check the nested checkpoint's embedded artifact digest; cannot authenticate a checkpoint missing that digest.",
    )
    parser.add_argument(
        "--trusted_historical_checkpoint_sha256", type=str, default=None,
        help="Independently pinned SHA-256 of historical checkpoint bytes (required for historical analysis).",
    )
    parser.add_argument(
        "--trusted_historical_artifact_sha256", type=str, default=None,
        help="Independently pinned SHA-256 of historical nested artifact bytes.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/jacobian_spectrum.png",
        help="Output path for Jacobian spectrum figure.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (alternative to --output; "
             "saves as <output_dir>/jacobian_spectrum.png).",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=Label.PREWALK.value,
        help="Label value to analyze (0=ESCAPE, 1=PREWALK, 2=PRE_ACTIVE, "
             "3=NO_RESPONSE).  Default PREWALK: the Sustained-epoch "
             "line-attractor hypothesis requires sustained locomotion.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for data loading.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2400,
        help="Crop sequences longer than this (cuDNN compatibility). 0 = disable.",
    )
    parser.add_argument(
        "--pre_anchor_frames",
        type=int,
        default=1200,
        help="Number of frames before anchor to include in anchor-aligned crop.",
    )
    parser.add_argument(
        "--dt_ms",
        type=float,
        default=None,
        help="Frame interval in ms (default: saved model.dt_ms; explicit value must match).",
    )
    parser.add_argument(
        "--max_states",
        type=int,
        default=100,
        help="Maximum states to process per epoch.",
    )
    parser.add_argument(
        "--full_system",
        action="store_true",
        default=False,
        help="Compute full-system surrogate input sensitivity singular values, "
             "not recurrent eigenvalues or stability.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("jax", "torch"),
        default="jax",
        help="GRU Jacobian kernel. jax is ~20x faster on this GPU for "
             "N=32–100, H=64 (falls back to torch if JAX is missing). "
             "Eigenvalues always use torch.linalg.eigvals.",
    )
    parser.add_argument(
        "--sampling_seed", type=int, default=42,
        help="Local seed for deterministic capped state selection.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve output path: --output_dir takes precedence when provided
    if args.output_dir is not None:
        output_path = Path(args.output_dir) / "jacobian_spectrum.png"
    else:
        output_path = Path(args.output)

    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    pre_anchor_frames = getattr(args, "pre_anchor_frames", 1200)
    run_jacobian_analysis(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        nested_prior_artifact=Path(args.nested_prior_artifact) if args.nested_prior_artifact else None,
        qc_sealed_nested_prior_sha256=args.qc_sealed_nested_prior_sha256,
        trusted_historical_checkpoint_sha256=args.trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=args.trusted_historical_artifact_sha256,
        output_path=output_path,
        target_class=args.target_class,
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        dt_ms=args.dt_ms,
        max_states_per_epoch=args.max_states,
        full_system=args.full_system,
        backend=args.backend,
        sampling_seed=args.sampling_seed,
    )


if __name__ == "__main__":
    main()
