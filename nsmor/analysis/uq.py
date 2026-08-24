"""
Uncertainty Quantification utilities for NSMoR analysis scripts.

Provides bootstrap confidence intervals, effect size computation,
and multiple comparison correction for scientific rigor.

Ref: Efron & Tibshirani 1993, "An Introduction to the Bootstrap".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import logging

import numpy as np
from scipy.stats import norm as _scipy_norm

logger = logging.getLogger(__name__)


def _norm_ppf(p: float) -> float:
    """Standard normal quantile (scipy is a hard project dependency)."""
    return float(_scipy_norm.ppf(p))


def _norm_cdf(z: float) -> float:
    """Standard normal CDF."""
    return float(_scipy_norm.cdf(z))


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: callable = np.mean,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    block_size: Optional[int] = None,
    method: str = "percentile",
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Supports both standard i.i.d. bootstrap and block-bootstrap for
    temporally correlated data (e.g., eigenvalue sequences from smooth
    GRU trajectories).

    When ``block_size`` is ``None``, uses standard bootstrap (i.i.d.
    resampling).  When ``block_size`` is set, uses block-bootstrap:
    resamples contiguous blocks of ``block_size`` observations rather
    than individual points.  This preserves temporal autocorrelation
    structure and produces valid confidence intervals for correlated
    data.

    Block-bootstrap reference:
        Künsch 1989, "The Jackknife and the Bootstrap for General
        Stationary Observations", Annals of Statistics 17(3):1217-1241.

    .. note::

        **Boundary behavior (Reviewer B #3, Round-1 MINOR-C fix):** the
        block resampling is *circular* (Politis & Romano 1992, JASA
        87:130-138): block start indices wrap around the end of the
        series, so every observation has identical marginal resampling
        probability and the truncation boundary bias of a plain
        moving-blocks scheme is eliminated entirely.

    Recommended block sizes for eigenvalue sequences:
        - Membrane time constant tau = -1/ln(alpha) ≈ 9.5 steps
          (for alpha=0.9).  Block_size = 5-10 steps captures the
          autocorrelation decay.
        - For GRU hidden states, block_size = 5-10 is typical.

    Args:
        data: 1-D array of observations.
        statistic_fn: Function to compute the statistic (default: mean).
        n_bootstrap: Number of bootstrap resamples (default: 1000).
        ci_level: Confidence level (default: 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        block_size: Block length for block-bootstrap.  ``None``
            (default) uses standard i.i.d. bootstrap.  Typical: 5-10
            for temporally correlated eigenvalue sequences.
        method: ``"percentile"`` (default) or ``"bca"``.  Round-2 fix
            (Reviewer A MINOR-K): the percentile interval is only
            second-order accurate; the bias-corrected and accelerated
            (BCa) interval corrects for estimator bias and skewness of
            the bootstrap distribution (Efron 1987, JASA 82:171-185)
            and attains better coverage when the bootstrap distribution
            is skewed — as it typically is for MSE statistics.
            BCa requires jackknife recomputation, so it costs O(n)
            extra statistic evaluations.  Only supported for the i.i.d.
            branch; block-bootstrap callers keep percentile intervals.

    Returns:
        ``(point_estimate, ci_lower, ci_upper)`` tuple.

    Raises:
        ValueError: If ``method`` is unknown or ``method="bca"`` is
            combined with ``block_size`` (not defined for circular
            blocks), or if fewer than 3 observations are supplied for
            BCa (acceleration undefined).
    """
    if method not in ("percentile", "bca"):
        raise ValueError(f"Unknown CI method: {method!r}")
    if method == "bca" and block_size is not None:
        raise ValueError(
            "method='bca' is not defined for block-bootstrap; use "
            "method='percentile' with block_size."
        )

    rng = np.random.RandomState(seed)
    point = statistic_fn(data)
    n = len(data)

    boot_stats = np.empty(n_bootstrap)

    if block_size is not None and block_size > 1 and block_size <= n:
        # ── Circular block-bootstrap for correlated data ─────────
        # Politis & Romano 1992: block start indices are drawn from the
        # full circular range [0, n-1]; blocks that run past the end of
        # the series wrap around (index modulo n).  Every observation
        # then has identical marginal resampling probability, removing
        # the truncation boundary bias of the moving-blocks variant.
        n_blocks_needed = int(np.ceil(n / block_size))

        for i in range(n_bootstrap):
            starts = rng.randint(0, n, size=n_blocks_needed)
            indices = np.concatenate([
                (np.arange(s, s + block_size) % n) for s in starts
            ])[:n]
            sample = data[indices]
            boot_stats[i] = statistic_fn(sample)
    else:
        # ── Standard i.i.d. bootstrap ────────────────────────────
        for i in range(n_bootstrap):
            sample = rng.choice(data, size=n, replace=True)
            boot_stats[i] = statistic_fn(sample)

    alpha = 1.0 - ci_level

    if method == "bca":
        # ── BCa correction (Efron 1987) ─────────────────────────
        # Bias correction z0: proportion of bootstrap statistics below
        # the point estimate, mapped through the normal quantile.
        #
        # Round-3 fix (Reviewer A MAJ-3A): a degenerate bootstrap
        # distribution (every replicate equal to the point estimate,
        # e.g. constant data) previously fell through to z0=0 — the
        # interval silently collapsed to the point value and LOOKED
        # like high precision.  A zero-width "95% CI" is misinformation,
        # not conservatism, so degeneracy now raises.
        n_degenerate = int(np.sum(boot_stats == point))
        if n_degenerate == n_bootstrap:
            raise ValueError(
                "BCa is undefined for a degenerate bootstrap "
                f"distribution (all {n_bootstrap} replicates equal the "
                "point estimate).  The CI would collapse to a single "
                "value and masquerade as precision; check whether the "
                "data are constant or the statistic is broken."
            )

        prop_below = float(np.mean(boot_stats < point))
        # Round-3 fix (Reviewer B MINOR-1): prop_below ∈ {0, 1} is the
        # EXTREME bias signal, not "no bias" — clamping z0 to 0 there
        # pretends the estimator is unbiased exactly when it is most
        # biased.  Keep the extreme z0 and warn; the interval widens
        # honestly.
        if prop_below <= 0.0:
            logger.warning(
                "BCa: no bootstrap replicate reached the point estimate "
                "(prop_below=0) — extreme positive bias correction "
                "applied (z0 = -8)."
            )
            z0 = -8.0
        elif prop_below >= 1.0:
            logger.warning(
                "BCa: every bootstrap replicate below the point estimate "
                "(prop_below=1) — extreme negative bias correction "
                "applied (z0 = +8)."
            )
            z0 = 8.0
        else:
            z0 = _norm_ppf(prop_below)

        # Acceleration a: from jackknife leave-one-out estimates,
        # a = sum(tau_i - tau_dot)^3 / (6 * [sum(tau_i - tau_dot)^2]^1.5)
        data_arr = np.asarray(data, dtype=np.float64)
        if n < 3:
            raise ValueError(
                f"BCa requires at least 3 observations, got {n}."
            )
        jack = np.array([
            statistic_fn(np.delete(data_arr, i)) for i in range(n)
        ])
        jack_mean = float(jack.mean())
        d = jack - jack_mean
        denom_sq = float(np.sum(d ** 2))
        if denom_sq <= np.finfo(float).eps:
            # Degenerate jackknife (statistic insensitive to single
            # points): fall back to plain bias-corrected interval.
            a = 0.0
        else:
            a = float(np.sum(d ** 3)) / (
                6.0 * denom_sq ** 1.5
            )

        # Corrected percentiles
        def _bca_pct(p_tail: float) -> float:
            z_alpha = _norm_ppf(p_tail)
            denom = 1.0 - a * (z0 + z_alpha)
            # Round-3 fix (Reviewer B MINOR-1): a denominator crossing
            # zero means the acceleration correction is singular — the
            # BCa transform is undefined, not merely "extreme".  Warn;
            # the sign of the residual denominator decides which side
            # the percentile saturates to (clamped below).
            if abs(denom) < 1e-8:
                logger.warning(
                    "BCa: acceleration correction singular "
                    "(1 - a(z0+z_alpha) = 0 at tail %.3f); interval "
                    "saturates to the nearest bootstrap order statistic.",
                    p_tail,
                )
            adj = z0 + (z0 + z_alpha) / denom
            return float(_norm_cdf(adj))

        pct_lo = _bca_pct(alpha / 2)
        pct_hi = _bca_pct(1 - alpha / 2)
        # Clamp AFTER the singularity warning; saturation to 0/1 maps
        # to the extreme bootstrap order statistics.
        pct_lo = min(max(pct_lo, 0.0), 1.0)
        pct_hi = min(max(pct_hi, 0.0), 1.0)
        ci_lower = float(np.percentile(boot_stats, 100 * pct_lo))
        ci_upper = float(np.percentile(boot_stats, 100 * pct_hi))
    else:
        ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return float(point), ci_lower, ci_upper


def cohens_d(group1: np.ndarray, group2: np.ndarray, paired: bool = False) -> float:
    """
    Compute Cohen's d effect size between two groups.

    For independent samples (default):
        d = (mean1 - mean2) / pooled_std

    For paired samples (CF3 fix — lesion experiments with same subjects):
        d = mean(diff) / sd(diff)
        where diff = group1 - group2 (element-wise)

    Paired d is appropriate when the same trials are measured under
    two conditions (e.g., intact vs lesioned model on same inputs).

    Interpretation (Cohen 1988):
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large

    Args:
        group1: 1-D array of observations from group 1.
        group2: 1-D array of observations from group 2.
        paired: If True, compute paired Cohen's d.

    Returns:
        Cohen's d (signed).
    """
    if paired:
        diff = group1 - group2
        sd_diff = np.std(diff, ddof=1)
        if sd_diff < 1e-12:
            return 0.0
        return float(np.mean(diff) / sd_diff)

    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-12:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, Tuple[float, bool]]:
    """
    Apply Holm-Bonferroni step-down correction for multiple comparisons.

    Controls the family-wise error rate (FWER) at the given alpha level.

    Round-1 fix (Reviewer A MAJOR-1): the previous implementation only
    enforced monotonicity of adjusted p-values and then compared every
    hypothesis against alpha uniformly.  That is *not* the Holm
    procedure: Holm is a step-down test where hypotheses are tested in
    ascending p-order with per-rank thresholds alpha/(m - rank), and
    the first non-rejection stops the procedure — all later (larger-p)
    hypotheses are automatically non-significant regardless of their
    individual adjusted p-values.  This implementation applies that
    linkage explicitly.

    Args:
        p_values: Dict mapping test name to uncorrected p-value.

    Returns:
        Dict mapping test name to ``(adjusted_p, significant)`` tuple,
        where ``adjusted_p`` is the standard Holm-adjusted p-value and
        ``significant`` reflects the full step-down rejection rule.
    """
    alpha = 0.05
    m = len(p_values)
    if m == 0:
        return {}

    # Sort by ascending p-value
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])

    result: Dict[str, Tuple[float, bool]] = {}
    prev_adjusted = 0.0
    rejecting = True  # step-down gate: once False, never reactivates
    for rank, (name, p) in enumerate(sorted_items):
        # Standard Holm-adjusted p: max over j<=rank of (m - j) * p_j,
        # enforced monotonically (Wright 1992).
        adjusted_p = min(1.0, max(prev_adjusted, (m - rank) * p))
        prev_adjusted = adjusted_p

        # Step-down rejection rule: reject rank r iff all ranks < r were
        # rejected AND (m - rank) * p_r <= alpha.
        if rejecting and (m - rank) * p <= alpha:
            significant = True
        else:
            significant = False
            rejecting = False

        result[name] = (adjusted_p, significant)

    return result


def log_pca_variance(pca_explained_var: np.ndarray, logger: callable) -> None:
    """
    Log PCA explained variance ratios with cumulative sums.

    Args:
        pca_explained_var: Array of explained variance ratios from PCA.
        logger: Logging function (e.g., logger.info).
    """
    cumulative = np.cumsum(pca_explained_var)
    for i, (var, cum) in enumerate(zip(pca_explained_var, cumulative)):
        logger(
            "    PC%d: explained variance = %.2f%%, cumulative = %.2f%%",
            i + 1, var * 100, cum * 100,
        )
