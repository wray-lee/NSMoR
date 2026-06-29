"""
Uncertainty Quantification utilities for NSMoR analysis scripts.

Provides bootstrap confidence intervals, effect size computation,
and multiple comparison correction for scientific rigor.

Ref: Efron & Tibshirani 1993, "An Introduction to the Bootstrap".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: callable = np.mean,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    block_size: Optional[int] = None,
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

        **Boundary behavior (Reviewer B #3):** When ``n`` is not evenly
        divisible by ``block_size``, the concatenation of resampled
        blocks is truncated to length ``n``.  This means the final
        ``n % block_size`` observations may have different resampling
        probabilities from the rest, introducing a minor but systematic
        boundary bias.  The standard Künsch (1989) moving-blocks
        bootstrap uses ``n - block_size + 1`` overlapping blocks and
        always samples complete blocks; the circular bootstrap
        (Politis & Romano 1992, JASA 87:130-138) wraps around to
        eliminate boundary effects entirely.  The current implementation
        is a valid but non-standard truncation variant.  For typical
        ``block_size = 5-10`` with eigenvalue sequences of length
        ~50-200, fewer than 20% of observations are affected.  For
        stricter scientific requirements, consider switching to the
        circular bootstrap.

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

    Returns:
        ``(point_estimate, ci_lower, ci_upper)`` tuple.
    """
    rng = np.random.RandomState(seed)
    point = statistic_fn(data)
    n = len(data)

    boot_stats = np.empty(n_bootstrap)

    if block_size is not None and block_size > 1 and block_size <= n:
        # ── Block-bootstrap for correlated data ──────────────────
        # Resample contiguous blocks of `block_size` observations.
        # The number of possible block start indices is n - block_size + 1.
        n_blocks_needed = int(np.ceil(n / block_size))
        max_start = n - block_size  # last valid start index

        for i in range(n_bootstrap):
            # Sample block start indices with replacement
            starts = rng.randint(0, max_start + 1, size=n_blocks_needed)
            # Concatenate blocks and truncate to original length
            indices = np.concatenate([
                np.arange(s, s + block_size) for s in starts
            ])[:n]
            sample = data[indices]
            boot_stats[i] = statistic_fn(sample)
    else:
        # ── Standard i.i.d. bootstrap ────────────────────────────
        for i in range(n_bootstrap):
            sample = rng.choice(data, size=n, replace=True)
            boot_stats[i] = statistic_fn(sample)

    alpha = 1.0 - ci_level
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
    Apply Holm-Bonferroni correction for multiple comparisons.

    Controls the family-wise error rate (FWER) at the given alpha level.

    Args:
        p_values: Dict mapping test name to uncorrected p-value.

    Returns:
        Dict mapping test name to ``(corrected_p, significant)`` tuple,
        where ``significant`` is True if corrected_p < 0.05.
    """
    alpha = 0.05
    m = len(p_values)
    if m == 0:
        return {}

    # Sort by p-value
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])

    result: Dict[str, Tuple[float, bool]] = {}
    prev_adjusted = 0.0
    for rank, (name, p) in enumerate(sorted_items):
        # Holm-Bonferroni: adjusted_p = min(1, (m - rank) * p)
        # CF2 fix: enforce monotonicity — adjusted p-values must be
        # non-decreasing.  Without this, rank 2 could be significant
        # while rank 1 is not, violating the step-down procedure.
        adjusted_p = min(1.0, max(prev_adjusted, (m - rank) * p))
        result[name] = (adjusted_p, adjusted_p < alpha)
        prev_adjusted = adjusted_p

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
