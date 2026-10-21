"""CPU-only synthetic calibration for the Phase-3 confirmatory core.

This module generates synthetic animal-level data only.  It never reads
real animal data, never trains, and never touches a GPU.  It calibrates
the four frozen confirmatory slots implemented in
:mod:`nsmor.analysis.confirmatory_stats`:

* type-I error at the H0 boundary at the planned sample sizes and under
  the pre-registered S0-S8 scenario matrix;
* family-wise error for the fixed four-slot Holm family with shared
  animal-level dependence (S7) and joint power (S8);
* power curves over the pre-registered effect grids;
* the descriptive leave-one-animal-out (LOO) fragility index (not a
  gate; no power-cost or sign-off keys);
* the H2 analysis set: per-seed closed-loop availability, defective-KM
  median undefined, the single 2.5% ensemble quantile, and the
  effective-draw < 40 exclusion.

The confirmatory test is the exact animal-level binomial sign test.
Its operating characteristics depend on ``p_improve = P(d_i < delta)``
and on the tie probability, not on the marginal skewness or variance.
The S1/S2 scenarios therefore generate skewed or heavy-tailed
differences whose **median** is at the boundary; the sign test remains
exact, and the calibration demonstrates that the earlier t-test's
skew sensitivity is gone.

The heavy runner lives in
``.scratch/confirmatory-core-20261019/calibration/run_calibration.py``
and writes the JSON and Markdown report.  This module is imported by the
runner and by the focused tests.

Scenario matrix (spec section 5.3):

==========  =========================================================
S0          Gaussian; arm/model correlation rho=0.5
S1          Lognormal (skew ~1.5); rho=0.5; median-centred
S2          Student t3 heavy tails; rho=0.5
S3          Gaussian; rho=0.8
S4          Gaussian; animal-level variance ratio 4:1
S5          H2 classical refit + three-seed noise + informative missingness
S6          H1/H3 three-seed noise (Gaussian)
S7          joint null for the four-slot family (shared animals)
S8          joint alternative for the four-slot family
==========  =========================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.stats import beta

from nsmor.analysis import confirmatory_stats as cs

SEED_BASE: int = cs.SEED_BASE
ALPHA_SLOT: float = 0.05 / cs.FAMILY_M
ALPHA_FAMILY: float = cs.ALPHA
# Single source of truth for the one-sided 95% Clopper-Pearson gate.
TYPE1_GATE: float = cs.TYPE1_CP95_GATE
assert TYPE1_GATE == ALPHA_SLOT + 0.005
FWER_GATE: float = 0.06

# H2 calibration criterion (user decision 2026-10-19 #2): the frozen
# replicate count, the frozen new seed, and the gate scenarios.  S1 is
# explicitly part of the criterion.
H2_CALIBRATION_R: int = cs.H2_CALIBRATION_R
H2_CALIBRATION_SEED: int = cs.H2_CALIBRATION_SEED
H2_GATE_SCENARIOS: tuple[str, ...] = ("S0", "S1", "S2")
# Replicate count for H2 curve / sensitivity / power cells (the frozen
# 200,000-replicate criterion applies to the three gate cells only).
H2_AUX_R: int = 5_000
# A power below this value at the planned n is recorded as
# ``underpowered_at_target``.  The flag is descriptive and never
# changes n, alpha, the test construction, or the planned analysis.
POWER_TARGET: float = 0.50


def _sim_n(slot, args: tuple[object, ...]) -> int:
    """Infer the synthetic animal count for a slot call."""
    if slot is cs.h1_foundation_persistence:
        first = next(iter(args[0].values()))  # type: ignore[union-attr]
        return int(np.asarray(first).shape[0])
    return int(np.asarray(args[0]).shape[0])


def _sim_call(slot, *args, **kwargs):
    """Call a slot with synthetic animal keys and the v(t) baseline.

    Calibration data are simulated at the animal level, so the keys are
    deterministic fixtures; they never denote real animals.
    """
    kwargs.setdefault(
        "animal_key", [f"sim-{index}" for index in range(_sim_n(slot, args))]
    )
    if slot is cs.h1_foundation_persistence:
        kwargs.setdefault("persistence_baseline", "v_t")
    return slot(*args, **kwargs)


SCENARIOS: tuple[str, ...] = (
    "S0",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
)
# Common MSE offset: the confirmatory statistics are differences, so a
# common offset leaves every p-value unchanged while keeping the arm MSE
# non-negative under heavy-tailed scenarios.
MSE_BASE: float = 100.0


# ---------------------------------------------------------------------------
# Monte Carlo helpers
# ---------------------------------------------------------------------------


def cp_upper(k: int, n: int, level: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound for ``k`` of ``n``."""
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    return float(beta.ppf(level, k + 1, n - k))


def mc_se(p: float, n: int) -> float:
    """Binomial Monte Carlo standard error."""
    return math.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1))


def spawn_seeds(base: int, count: int) -> list[int]:
    """Derive ``count`` reproducible child seeds with SeedSequence.spawn."""
    if count < 1:
        raise ValueError("count must be positive")
    children = np.random.SeedSequence(int(base)).spawn(int(count))
    seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in children
    ]
    assert len(seeds) == count
    return seeds


def equicorr(k: int, rho: float) -> np.ndarray:
    """Equicorrelation matrix with unit diagonal."""
    if k < 1:
        raise ValueError("k must be positive")
    if k > 1 and not -1.0 / (k - 1) <= rho <= 1.0:
        raise ValueError("rho outside the positive-semidefinite range")
    matrix = rho * np.ones((k, k)) + (1.0 - rho) * np.eye(k)
    assert matrix.shape == (k, k)
    return matrix


def lognormal_sigma(skew: float) -> float:
    """Lognormal sigma yielding the requested skewness."""
    if skew <= 0.0:
        return 0.0
    lo, hi = 0.0, 3.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        u = math.exp(mid * mid) - 1.0
        value = (u + 3.0) * math.sqrt(u)
        if value < skew:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _lognormal_sd(sigma: float) -> float:
    """Population SD of ``exp(sigma * Z)``."""
    return math.sqrt(
        (math.exp(sigma * sigma) - 1.0) * math.exp(sigma * sigma)
    )


def skew_noise(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    sigma: float,
    skew: float,
) -> np.ndarray:
    """Zero-median, unit-SD lognormal noise with the requested skewness.

    The raw lognormal has median 1, so subtracting 1 gives a distribution
    with median exactly zero.  This is the relevant centring for a sign
    test: ``P(noise < 0) = 0.5`` even though the mean is positive.
    Standardisation uses the **population** lognormal SD, not the
    realised sample moments.
    """
    if skew <= 0.0:
        return rng.normal(scale=sigma, size=shape)
    s = lognormal_sigma(skew)
    raw = rng.lognormal(mean=0.0, sigma=s, size=shape)
    return sigma * (raw - 1.0) / _lognormal_sd(s)


def t3_noise(
    rng: np.random.Generator, shape: tuple[int, ...], sigma: float
) -> np.ndarray:
    """Zero-median, unit-SD Student t3 noise.

    A t3 variate has variance 3; dividing by ``sqrt(3)`` standardises it
    with the population variance (no per-dataset rescaling).
    """
    z = rng.normal(size=shape)
    chi = rng.chisquare(3, size=shape) / 3.0
    raw = z / np.sqrt(chi)
    return sigma * raw / math.sqrt(3.0)


def animal_scale(n: int, ratio: float = 2.0) -> np.ndarray:
    """Animal-level SD scale with a variance ratio of ``ratio**2``."""
    scale = np.ones(n, dtype=np.float64)
    scale[: n // 2] = ratio
    assert scale.shape == (n,)
    return scale


def _copula_marginal(
    correlated: np.ndarray, scenario: str
) -> np.ndarray:
    """Apply the scenario's marginal to correlated standard normals.

    The transform is monotone, so the Gaussian-copula rank dependence
    (``rho``) is preserved.  Standardisation uses population moments
    (lognormal analytic moments; t3 variance 3), never the realised
    sample moments of the simulated dataset.  S6 is the three-seed
    scenario and keeps a Gaussian marginal; only S1 is lognormal.
    """
    if scenario == "S1":
        a = lognormal_sigma(1.5)
        raw = np.exp(a * correlated)
        return (raw - 1.0) / _lognormal_sd(a)
    if scenario == "S2":
        return stats.t.ppf(stats.norm.cdf(correlated), df=3) / math.sqrt(3.0)
    return correlated


def _equicorrelated_noise(
    rng: np.random.Generator,
    n: int,
    n_sim: int,
    k: int,
    scenario: str,
    sigma: float,
) -> np.ndarray:
    """Return ``(n_sim, n, k)`` equicorrelated noise with the S-marginal."""
    rho = 0.8 if scenario == "S3" else 0.5
    corr = equicorr(k, rho)
    chol = np.linalg.cholesky(corr)
    z = rng.normal(size=(n_sim, n, k))
    correlated = z @ chol.T
    assert correlated.shape == (n_sim, n, k)
    return sigma * _copula_marginal(correlated, scenario)


def _marginal_noise(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    scenario: str,
    sigma: float,
) -> np.ndarray:
    """Zero-median noise with the scenario marginal."""
    if scenario == "S1":
        return skew_noise(rng, shape, sigma, 1.5)
    if scenario == "S2":
        return t3_noise(rng, shape, sigma)
    return rng.normal(scale=sigma, size=shape)


def _quantile_shift(scenario: str, p_improve: float, sigma: float) -> float:
    """Shift that makes ``P(noise - shift < 0) = p_improve``."""
    if not 0.0 < p_improve < 1.0:
        raise ValueError("p_improve must lie strictly between 0 and 1")
    if scenario == "S1":
        a = lognormal_sigma(1.5)
        return sigma * (math.exp(a * stats.norm.ppf(p_improve)) - 1.0) / (
            _lognormal_sd(a)
        )
    if scenario == "S2":
        return sigma * float(stats.t.ppf(p_improve, df=3)) / math.sqrt(3.0)
    return sigma * float(stats.norm.ppf(p_improve))


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def h1_draws(
    rng: np.random.Generator,
    n: int,
    n_sim: int,
    scenario: str,
    p_improve: float = 0.5,
    sigma: float = 10.0,
    n_seed: int = 1,
    p_tie: float = 0.0,
) -> np.ndarray:
    """Return ``(n_sim, n, 4)`` per-animal arm differences.

    The confirmatory arm is index 1 (``R1``).  Its median difference is
    shifted so that ``P(d < 0) = p_improve``; the other arms stay at
    median zero and are descriptive.  ``p_tie`` sets an exact-zero tie
    with that probability (ties count as non-improvements in the sign
    test).
    """
    if n_seed < 1:
        raise ValueError("n_seed must be positive")
    if not 0.0 <= p_tie < 1.0:
        raise ValueError("p_tie must lie in [0, 1)")
    if n_seed == 1:
        noise = _equicorrelated_noise(rng, n, n_sim, 4, scenario, sigma)
        effective_sigma = sigma
    else:
        shared = _equicorrelated_noise(rng, n, n_sim, 4, scenario, sigma)
        seed_noise = _marginal_noise(
            rng, (n_sim, n, 4), scenario, sigma * 0.30
        )
        noise = shared + seed_noise / math.sqrt(n_seed)
        effective_sigma = sigma * math.sqrt(1.0 + 0.09 / n_seed)
    if scenario == "S4":
        scale = animal_scale(n)
        noise = noise * scale[None, :, None]
    else:
        scale = np.ones(n, dtype=np.float64)
    shift = _quantile_shift(scenario, p_improve, effective_sigma) * scale
    noise = noise.copy()
    noise[:, :, 1] -= shift[None, :]
    if p_tie > 0.0:
        ties = rng.random(noise.shape) < p_tie
        noise[ties] = 0.0
    assert noise.shape == (n_sim, n, 4)
    return noise


def h3_draws(
    rng: np.random.Generator,
    n: int,
    n_sim: int,
    scenario: str,
    p_improve: float = 0.5,
    sigma: float = 0.05,
    n_seed: int = 1,
    p_tie: float = 0.0,
) -> np.ndarray:
    """Return ``(n_sim, n)`` per-animal C1-S minus C16 MSE differences."""
    if not 0.0 <= p_tie < 1.0:
        raise ValueError("p_tie must lie in [0, 1)")
    if n_seed == 1:
        draws = _equicorrelated_noise(
            rng, n, n_sim, 1, scenario, sigma
        )[:, :, 0]
        effective_sigma = sigma
    else:
        shared = _equicorrelated_noise(
            rng, n, n_sim, 1, scenario, sigma
        )[:, :, 0]
        seed_noise = _marginal_noise(
            rng, (n_sim, n), scenario, sigma * 0.30
        )
        draws = shared + seed_noise / math.sqrt(n_seed)
        effective_sigma = sigma * math.sqrt(1.0 + 0.09 / n_seed)
    if scenario == "S4":
        scale = animal_scale(n)
        draws = draws * scale[None, :]
    else:
        scale = np.ones(n, dtype=np.float64)
    shift = _quantile_shift(scenario, p_improve, effective_sigma) * scale
    draws = draws - shift[None, :]
    if p_tie > 0.0:
        ties = rng.random(draws.shape) < p_tie
        draws[ties] = 0.0
    assert draws.shape == (n_sim, n)
    return draws


@dataclass(frozen=True)
class H2RealisticDraws:
    """Synthetic H2 analysis-set data.

    Attributes:
        l_obs: ``(n_sim, n)`` observed latency, NaN if undefined.
        l_nsmor_by_seed: ``(n_sim, n, n_seed)`` closed-loop latency per
            seed, NaN if undefined.
        l_nsmor: ``(n_sim, n)`` seed-mean latency, NaN if any seed is
            undefined.
        all_seed_mask: ``(n_sim, n)`` boolean mask from
            :func:`nsmor.analysis.confirmatory_stats.h2_seed_errors`.
        e_ns: ``(n_sim, n)`` mean per-seed absolute closed-loop error,
            NaN if the animal is excluded.
        e_cls: ``(n_sim, n, 4)`` conservative classical errors, NaN if
            undefined.  Model 0 is the binding (best) model.
        e_cls_true: ``(n_sim, n)`` the single 2.5% quantile used to
            construct both ``e_ns`` and model 0's ``e_cls``.
        true_delta_best: ``(n_sim,)`` mean realised
            ``e_NS - e_cls_true`` on the analysis set; the boundary
            diagnostic that prevents a vacuous zero-rejection count.
        median_true_delta_best: ``(n_sim,)`` median of the same.
        effective_draws: ``(n_sim, n, 4)`` effective ensemble draws.
        analysis_mask: ``(n_sim, n)`` analysis-set A mask.
        n_available: ``(n_sim,)`` analysis-set size.
        excluded: aggregate analysis-set exclusion counts (hierarchy).
    """

    l_obs: np.ndarray
    l_nsmor_by_seed: np.ndarray
    l_nsmor: np.ndarray
    all_seed_mask: np.ndarray
    e_ns: np.ndarray
    e_cls: np.ndarray
    e_cls_true: np.ndarray
    true_delta_best: np.ndarray
    median_true_delta_best: np.ndarray
    effective_draws: np.ndarray
    analysis_mask: np.ndarray
    n_available: np.ndarray
    excluded: Mapping[str, int]


def h2_realistic_draws(
    rng: np.random.Generator,
    n: int,
    n_sim: int,
    scenario: str,
    delta_best: float,
    n_ens: int = 200,
    n_seed: int = 3,
    p_obs_defined: float = 0.90,
    p_ns_defined: float = 0.90,
    p_model_ok: float = 0.98,
    sigma_ens: float = 5.0,
    sigma_ens_spread: float = 0.0,
    sigma_ns: float = 6.0,
    sigma_seed: float = 0.0,
    informative_missing: bool = False,
) -> H2RealisticDraws:
    """Simulate the H2 analysis set with per-seed availability.

    The binding (best) classical model has a latent animal-level error
    (mean 20 ms).  Its conservative error is **one** ensemble quantile

    ``e_cls_true,i = Q_0.025(|latent_i + eps_i|)``.

    That same value constructs the closed-loop target
    ``e_ns_target,i = e_cls_true,i + delta_best + noise``.  For each
    training seed an independent availability draw is made; the animal
    enters A only if **all** seeds are defined (the pre-registered
    all-seed rule).  :func:`h2_seed_errors` then averages the per-seed
    absolute errors.  The other three classical models are worse by a
    fixed 20 ms and use the same quantile plus that offset.

    Args:
        rng: Random generator.
        n: Planned number of multisensory test animals.
        n_sim: Number of simulated data sets.
        scenario: S0-S6 scenario identifier.
        delta_best: True ``e_NS - e_cls,best`` at the boundary (15 for
            the null, smaller under the alternative).
        n_ens: Ensemble draws per animal (the real analysis uses 2,000;
            200 is conservative for the quantile noise).
        n_seed: Number of training seeds (three in the real design).
        p_obs_defined: Probability the defective-KM median is defined.
        p_ns_defined: Probability a single seed's closed-loop onset is
            defined; the all-seed probability is ``p_ns_defined**n_seed``.
        p_model_ok: Probability a classical model ensemble is usable.
        sigma_ens: Ensemble prediction noise (ms).
        sigma_ens_spread: Log-scale SD of animal-level ensemble widths
            (0 = equal widths, matching the main calibration).
        sigma_ns: Animal-level closed-loop error noise (ms).
        sigma_seed: Per-seed closed-loop noise SD (0 for the main
            boundary calibration; >0 for the S5 sensitivity).
        informative_missing: When true, larger ``e_NS`` lowers the
            animal-level closed-loop availability probability
            (selection sensitivity).  The dropout is shared across
            seeds, matching a closed loop that fails on the animals with
            the longest latencies.
    """
    if n < 2:
        raise ValueError("n must be at least two")
    if n_ens < 40:
        raise ValueError("n_ens must be at least 40")
    if n_seed < 1:
        raise ValueError("n_seed must be positive")
    if sigma_ens <= 0.0 or sigma_ns <= 0.0:
        raise ValueError("sigma_ens and sigma_ns must be positive")
    if sigma_seed < 0.0:
        raise ValueError("sigma_seed must be non-negative")
    if sigma_ens_spread < 0.0:
        raise ValueError("sigma_ens_spread must be non-negative")
    n_models = len(cs.H2_MODELS)
    l_obs = np.full((n_sim, n), np.nan)
    l_nsmor_by_seed = np.full((n_sim, n, n_seed), np.nan)
    l_nsmor = np.full((n_sim, n), np.nan)
    all_seed_mask = np.zeros((n_sim, n), dtype=bool)
    e_ns = np.full((n_sim, n), np.nan)
    e_cls = np.full((n_sim, n, n_models), np.nan)
    e_cls_true_out = np.full((n_sim, n), np.nan)
    true_delta_best = np.full(n_sim, np.nan)
    median_true_delta_best = np.full(n_sim, np.nan)
    effective = np.zeros((n_sim, n, n_models), dtype=np.int64)
    analysis_mask = np.zeros((n_sim, n), dtype=bool)
    n_available = np.zeros(n_sim, dtype=np.int64)
    excluded = {
        "seed_incomplete": 0,
        "l_obs_undefined": 0,
        "l_nsmor_undefined": 0,
        "classical_undefined": 0,
        "effective_draws_lt_40": 0,
    }
    for r in range(n_sim):
        obs_latent = 100.0 + rng.normal(scale=10.0, size=n)
        obs_ok = rng.random(n) < p_obs_defined
        # Latent animal-level error of the best classical model.
        if scenario == "S1":
            latent = 20.0 + skew_noise(rng, (n,), 8.0, 1.5)
        elif scenario == "S2":
            latent = 20.0 + t3_noise(rng, (n,), 8.0)
        else:
            latent = 20.0 + rng.normal(scale=8.0, size=n)
        if scenario == "S4":
            latent = latent * animal_scale(n)
        latent = np.clip(latent, 1.0, None)
        if sigma_ens_spread > 0.0:
            width = sigma_ens * np.exp(
                sigma_ens_spread * rng.normal(size=n)
            )
            width = np.clip(width, 0.25 * sigma_ens, 4.0 * sigma_ens)
        else:
            width = np.full(n, sigma_ens, dtype=np.float64)
        ens = latent[:, None] + width[:, None] * rng.normal(
            size=(n, n_ens)
        )
        e_cls_true = np.quantile(np.abs(ens), 0.025, axis=1)
        if scenario == "S5":
            e_cls_true = np.clip(
                e_cls_true + rng.normal(scale=2.0, size=n), 0.0, None
            )
        ns_noise = _marginal_noise(rng, (n,), scenario, sigma_ns)
        if scenario == "S5":
            ns_noise = ns_noise + rng.normal(
                scale=2.0 / math.sqrt(n_seed), size=n
            )
        e_ns_target = np.clip(
            e_cls_true + delta_best + ns_noise, 0.0, None
        )
        l_obs[r] = np.where(obs_ok, obs_latent, np.nan)
        # Informative-missingness sensitivity (r9).  The dropout is an
        # animal-level property: the closed loop fails on the animals
        # with the longest latencies, which is what makes a
        # non-inferiority test anti-conservative.  The earlier DGP used
        # a per-seed logistic with centre 30 ms, so at the boundary
        # (e_ns_target ~ 35-40 ms) the all-seed probability collapsed to
        # ~0.02 and the cell was degenerate (n_evaluable = 1/5000).
        # Centring at 40 ms with scale 6 keeps overall availability at
        # or below the non-informative setting while retaining a strong
        # error dependence (at the boundary e_NS ~ 35 ms the probability
        # is ~0.70, so the all-seed availability is ~0.60 versus 0.73).
        if informative_missing:
            animal_ok = rng.random(n) < 1.0 / (
                1.0 + np.exp((e_ns_target - 40.0) / 6.0)
            )
        else:
            animal_ok = np.ones(n, dtype=bool)
        for seed_index in range(n_seed):
            if informative_missing:
                ns_ok = animal_ok & (rng.random(n) < 0.95)
            else:
                ns_ok = rng.random(n) < p_ns_defined
            seed_noise = (
                rng.normal(scale=sigma_seed, size=n)
                if sigma_seed > 0.0
                else np.zeros(n, dtype=np.float64)
            )
            e_seed = e_ns_target + seed_noise
            l_seed = obs_latent + e_seed
            l_seed = np.where(obs_ok & ns_ok, l_seed, np.nan)
            l_nsmor_by_seed[r, :, seed_index] = l_seed
        e_ns_r, seed_mask, _ = cs.h2_seed_errors(
            l_nsmor_by_seed[r], l_obs[r]
        )
        e_ns[r] = e_ns_r
        all_seed_mask[r] = seed_mask
        if np.any(seed_mask):
            l_nsmor[r, seed_mask] = l_nsmor_by_seed[
                r, seed_mask
            ].mean(axis=1)
        for model_index in range(n_models):
            e_true = (
                e_cls_true
                if model_index == 0
                else e_cls_true + 20.0
            )
            model_ok = rng.random(n) < p_model_ok
            n_eff = rng.binomial(n_ens, 0.95, size=n)
            valid = model_ok & (n_eff >= 40)
            effective[r, :, model_index] = n_eff
            e_cls[r, :, model_index] = np.where(valid, e_true, np.nan)
        e_cls_map = {
            model: e_cls[r, :, index]
            for index, model in enumerate(cs.H2_MODELS)
        }
        draws_map = {
            model: effective[r, :, index].astype(np.float64)
            for index, model in enumerate(cs.H2_MODELS)
        }
        analysis_set = cs.build_h2_analysis_set(
            l_obs[r],
            l_nsmor[r],
            e_cls_map,
            all_seed_mask=seed_mask,
            effective_draws_by_model=draws_map,
        )
        analysis_mask[r] = analysis_set.mask
        n_available[r] = analysis_set.n_animal
        for key, value in analysis_set.excluded.items():
            excluded[key] += int(value)
        mask = analysis_set.mask
        if np.any(mask):
            delta = e_ns[r, mask] - e_cls_true[mask]
            true_delta_best[r] = float(np.mean(delta))
            median_true_delta_best[r] = float(np.median(delta))
        e_cls_true_out[r] = e_cls_true
    assert l_obs.shape == (n_sim, n)
    assert l_nsmor_by_seed.shape == (n_sim, n, n_seed)
    assert e_cls.shape == (n_sim, n, n_models)
    assert analysis_mask.shape == (n_sim, n)
    assert n_available.shape == (n_sim,)
    return H2RealisticDraws(
        l_obs=l_obs,
        l_nsmor_by_seed=l_nsmor_by_seed,
        l_nsmor=l_nsmor,
        all_seed_mask=all_seed_mask,
        e_ns=e_ns,
        e_cls=e_cls,
        e_cls_true=e_cls_true_out,
        true_delta_best=true_delta_best,
        median_true_delta_best=median_true_delta_best,
        effective_draws=effective,
        analysis_mask=analysis_mask,
        n_available=n_available,
        excluded=excluded,
    )


# ---------------------------------------------------------------------------
# Calibration routines
# ---------------------------------------------------------------------------


def batch_mc_se(
    rejections: np.ndarray, n_batches: int = 20
) -> float:
    """Batch-means Monte Carlo SE for a rejection indicator vector."""
    values = np.asarray(rejections, dtype=np.float64).ravel()
    if values.size == 0:
        return float("nan")
    n_batches = min(int(n_batches), int(values.size))
    if n_batches < 2:
        return mc_se(float(values.mean()), values.size)
    batches = np.array_split(values, n_batches)
    means = np.array([float(batch.mean()) for batch in batches])
    assert means.shape == (n_batches,)
    return float(means.std(ddof=1) / math.sqrt(n_batches))


def _summarise(p_values: np.ndarray, n_sim: int) -> dict[str, object]:
    """Summarise a vector of p-values at the slot alpha."""
    finite = p_values[np.isfinite(p_values)]
    rejected = finite <= ALPHA_SLOT
    rejections = int(np.count_nonzero(rejected))
    n_eval = int(finite.size)
    type1 = rejections / n_eval if n_eval else float("nan")
    binomial_se = mc_se(type1, n_eval) if n_eval else float("nan")
    batch_se = batch_mc_se(rejected) if n_eval else float("nan")
    return {
        "n_sim": int(n_sim),
        "n_evaluable": n_eval,
        "rejections": rejections,
        "type1": float(type1),
        "type1_cp95_upper": cp_upper(rejections, n_eval)
        if n_eval
        else float("nan"),
        "mc_se": float(batch_se),
        "mc_se_binomial": float(binomial_se),
        "mc_se_method": "batch_means_20",
        "mean_p": float(finite.mean()) if n_eval else float("nan"),
    }


def calibrate_h1_null(
    n: int,
    scenario: str,
    n_sim: int,
    seed: int,
    p_improve: float = 0.5,
    scope: str = "onset",
    p_tie: float = 0.0,
) -> dict[str, object]:
    """Calibrate one H1 scope at the sign-test null boundary."""
    rng = np.random.default_rng(seed)
    n_seed = 3 if scenario == "S6" else 1
    draws = h1_draws(
        rng,
        n,
        n_sim,
        scenario,
        p_improve=p_improve,
        n_seed=n_seed,
        p_tie=p_tie,
    )
    p_values = np.empty(n_sim)
    for r in range(n_sim):
        persistence = np.full(n, MSE_BASE, dtype=np.float64)
        by_arm = {
            arm: np.maximum(persistence + draws[r, :, j], 0.0)
            for j, arm in enumerate(cs.H1_FOUNDATION_ARMS)
        }
        result = _sim_call(cs.h1_foundation_persistence, 
            by_arm, persistence, scope, n_boot=1, compute_ci=False
        )
        p_values[r] = result.p_value
    summary = _summarise(p_values, n_sim)
    summary.update(
        {
            "n": n,
            "scenario": scenario,
            "scope": scope,
            "p_improve": p_improve,
            "p_tie": p_tie,
            "seed": int(seed),
            "seed_base": SEED_BASE,
        }
    )
    return summary


def calibrate_h3_null(
    n: int,
    scenario: str,
    n_sim: int,
    seed: int,
    p_improve: float = 0.5,
    p_tie: float = 0.0,
) -> dict[str, object]:
    """Calibrate H3 at the sign-test null boundary."""
    rng = np.random.default_rng(seed)
    n_seed = 3 if scenario == "S6" else 1
    draws = h3_draws(
        rng,
        n,
        n_sim,
        scenario,
        p_improve=p_improve,
        n_seed=n_seed,
        p_tie=p_tie,
    )
    p_values = np.empty(n_sim)
    for r in range(n_sim):
        c16 = np.full(n, MSE_BASE, dtype=np.float64)
        c1s = np.maximum(c16 + draws[r], 0.0)
        result = _sim_call(cs.h3_camor_derecursion, 
            c1s,
            c16,
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, False, False),
            n_boot=1,
            compute_ci=False,
        )
        p_values[r] = result.p_value
    summary = _summarise(p_values, n_sim)
    summary.update(
        {
            "n": n,
            "scenario": scenario,
            "p_improve": p_improve,
            "p_tie": p_tie,
            "seed": int(seed),
            "seed_base": SEED_BASE,
        }
    )
    return summary


def calibrate_h2_realistic(
    n: int,
    scenario: str,
    n_sim: int,
    seed: int,
    delta_best: float = cs.DELTA_NI_MS,
    informative_missing: bool = False,
    sigma_ens_spread: float = 0.0,
    sigma_seed: float = 0.0,
    p_obs_defined: float = 0.90,
    p_ns_defined: float = 0.90,
    p_model_ok: float = 0.98,
    n_ens: int = 50,
) -> dict[str, object]:
    """Calibrate H2 on the simulated analysis set A.

    ``n_ens`` is the synthetic ensemble size.  The real analysis uses
    2,000 draws; 50 is conservative (more quantile noise, lower power)
    and leaves the sign-test type-I error unchanged.
    """
    rng = np.random.default_rng(seed)
    data = h2_realistic_draws(
        rng,
        n,
        n_sim,
        scenario,
        delta_best=delta_best,
        n_ens=n_ens,
        informative_missing=informative_missing,
        sigma_ens_spread=sigma_ens_spread,
        sigma_seed=sigma_seed,
        p_obs_defined=p_obs_defined,
        p_ns_defined=p_ns_defined,
        p_model_ok=p_model_ok,
    )
    p_values = np.full(n_sim, np.nan)
    for r in range(n_sim):
        mask = data.analysis_mask[r]
        if int(mask.sum()) < cs.N_MIN:
            continue
        e_cls_map = {
            model: data.e_cls[r, :, index]
            for index, model in enumerate(cs.H2_MODELS)
        }
        result = _sim_call(cs.h2_non_inferior_best_classical, 
            data.e_ns[r],
            e_cls_map,
            analysis_set=cs._analysis_set_from_mask(mask),
            n_boot=1,
            compute_ci=False,
        )
        p_values[r] = result.p_value
    summary = _summarise(p_values, n_sim)
    worst_case = np.full(n_sim, np.nan)
    for r in range(n_sim):
        mask = data.analysis_mask[r]
        if int(mask.sum()) < cs.N_MIN:
            continue
        worst_case[r] = cs.h2_worst_case_p(
            data.e_ns[r],
            {
                model: data.e_cls[r, :, index]
                for index, model in enumerate(cs.H2_MODELS)
            },
            analysis_set=cs._analysis_set_from_mask(mask),
        )[0]
    worst = _summarise(worst_case, n_sim)
    summary["worst_case"] = worst
    available = data.n_available >= cs.N_MIN
    true_delta = data.true_delta_best[available]
    median_delta = data.median_true_delta_best[available]
    summary.update(
        {
            "n_planned": n,
            "scenario": scenario,
            "delta_best": delta_best,
            "sigma_ens_spread": sigma_ens_spread,
            "sigma_seed": sigma_seed,
            "informative_missing": informative_missing,
            "seed": int(seed),
            "seed_base": H2_CALIBRATION_SEED,
            "p_obs_defined": p_obs_defined,
            "p_ns_defined": p_ns_defined,
            "p_model_ok": p_model_ok,
            "p_analysis_set_available": float(np.mean(available)),
            "mean_n_available": float(np.mean(data.n_available)),
            "type1_unconditional": (
                float(summary["rejections"]) / float(n_sim)
            ),
            "mean_true_delta_best": float(np.mean(true_delta))
            if true_delta.size
            else float("nan"),
            "median_true_delta_best": float(np.mean(median_delta))
            if median_delta.size
            else float("nan"),
            "sd_true_delta_best": float(np.std(true_delta, ddof=1))
            if true_delta.size > 1
            else float("nan"),
            "mean_abs_true_delta_error": float(
                np.mean(np.abs(true_delta - delta_best))
            )
            if true_delta.size
            else float("nan"),
            "excluded": dict(data.excluded),
        }
    )
    return summary


def calibrate_h1_power(
    n: int,
    scenario: str,
    n_sim: int,
    seed: int,
    p_improve: float,
    scope: str = "onset",
) -> dict[str, object]:
    """Calibrate H1 power for one sign probability."""
    rng = np.random.default_rng(seed)
    n_seed = 3 if scenario == "S6" else 1
    draws = h1_draws(
        rng,
        n,
        n_sim,
        scenario,
        p_improve=p_improve,
        n_seed=n_seed,
    )
    rejected = np.zeros(n_sim, dtype=bool)
    for r in range(n_sim):
        persistence = np.full(n, MSE_BASE, dtype=np.float64)
        by_arm = {
            arm: np.maximum(persistence + draws[r, :, j], 0.0)
            for j, arm in enumerate(cs.H1_FOUNDATION_ARMS)
        }
        result = _sim_call(cs.h1_foundation_persistence, 
            by_arm, persistence, scope, n_boot=1, compute_ci=False
        )
        rejected[r] = result.p_value <= ALPHA_SLOT
    power = float(rejected.mean())
    return {
        "n": n,
        "scenario": scenario,
        "scope": scope,
        "p_improve": p_improve,
        "gaussian_equivalent_skill": float(
            stats.norm.ppf(p_improve) * 10.0 / MSE_BASE
        ),
        "n_sim": n_sim,
        "rejections": int(np.count_nonzero(rejected)),
        "power": power,
        "power_threshold": POWER_TARGET,
        "target_n": int(n),
        "underpowered_at_target": bool(power < POWER_TARGET),
        "mc_se": batch_mc_se(rejected),
        "mc_se_binomial": mc_se(power, n_sim),
        "mc_se_method": "batch_means_20",
        "seed": int(seed),
        "seed_base": SEED_BASE,
    }


def calibrate_h3_power(
    n: int,
    scenario: str,
    n_sim: int,
    seed: int,
    p_improve: float,
) -> dict[str, object]:
    """Calibrate H3 power for one sign probability."""
    rng = np.random.default_rng(seed)
    n_seed = 3 if scenario == "S6" else 1
    draws = h3_draws(
        rng,
        n,
        n_sim,
        scenario,
        p_improve=p_improve,
        n_seed=n_seed,
    )
    rejected = np.zeros(n_sim, dtype=bool)
    for r in range(n_sim):
        c16 = np.full(n, MSE_BASE, dtype=np.float64)
        c1s = np.maximum(c16 + draws[r], 0.0)
        result = _sim_call(cs.h3_camor_derecursion, 
            c1s,
            c16,
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, False, False),
            n_boot=1,
            compute_ci=False,
        )
        rejected[r] = result.p_value <= ALPHA_SLOT
    power = float(rejected.mean())
    return {
        "n": n,
        "scenario": scenario,
        "p_improve": p_improve,
        "gaussian_equivalent_d_z": float(stats.norm.ppf(p_improve)),
        "n_sim": n_sim,
        "rejections": int(np.count_nonzero(rejected)),
        "power": power,
        "power_threshold": POWER_TARGET,
        "target_n": int(n),
        "underpowered_at_target": bool(power < POWER_TARGET),
        "mc_se": batch_mc_se(rejected),
        "mc_se_binomial": mc_se(power, n_sim),
        "mc_se_method": "batch_means_20",
        "seed": int(seed),
        "seed_base": SEED_BASE,
    }


def calibrate_h2_power(
    n: int,
    scenario: str,
    n_sim: int,
    seed: int,
    delta_best: float,
    n_ens: int = 50,
) -> dict[str, object]:
    """Calibrate H2 power on the simulated analysis set A.

    Reports both the conditional power (given ``n_A >= 10``) and the
    unconditional power over all simulated data sets.  The unconditional
    number is the one that reflects the planned design with missingness.
    """
    rng = np.random.default_rng(seed)
    data = h2_realistic_draws(
        rng, n, n_sim, scenario, delta_best=delta_best, n_ens=n_ens
    )
    rejected: list[bool] = []
    true_deltas: list[float] = []
    for r in range(n_sim):
        mask = data.analysis_mask[r]
        if int(mask.sum()) < cs.N_MIN:
            continue
        e_cls_map = {
            model: data.e_cls[r, :, index]
            for index, model in enumerate(cs.H2_MODELS)
        }
        result = _sim_call(cs.h2_non_inferior_best_classical, 
            data.e_ns[r],
            e_cls_map,
            analysis_set=cs._analysis_set_from_mask(mask),
            n_boot=1,
            compute_ci=False,
        )
        rejected.append(bool(result.p_value <= ALPHA_SLOT))
        true_deltas.append(float(data.true_delta_best[r]))
    rejected_array = np.asarray(rejected, dtype=bool)
    evaluable = int(rejected_array.size)
    power = float(rejected_array.mean()) if evaluable else float("nan")
    available = data.n_available >= cs.N_MIN
    return {
        "n_planned": n,
        "scenario": scenario,
        "delta_best": delta_best,
        "n_sim": n_sim,
        "n_evaluable": evaluable,
        "rejections": int(np.count_nonzero(rejected_array)),
        "power_conditional": power,
        "power_unconditional": (
            float(np.count_nonzero(rejected_array)) / float(n_sim)
        ),
        "power_threshold": POWER_TARGET,
        "target_n": int(n),
        "underpowered_at_target": bool(
            (float(np.count_nonzero(rejected_array)) / float(n_sim))
            < POWER_TARGET
        ),
        "underpowered_conditional": bool(power < POWER_TARGET),
        "p_analysis_set_available": float(np.mean(available)),
        "mean_n_available": float(np.mean(data.n_available)),
        "mc_se": batch_mc_se(rejected_array)
        if evaluable
        else float("nan"),
        "mc_se_binomial": mc_se(power, evaluable)
        if evaluable
        else float("nan"),
        "mc_se_method": "batch_means_20",
        "mean_true_delta_best": float(np.mean(true_deltas))
        if true_deltas
        else float("nan"),
        "seed": int(seed),
        "seed_base": H2_CALIBRATION_SEED,
    }


def _joint_latents(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared animal latents with the pre-registered correlations.

    ``h3_load = 0.5 / sqrt(0.5 * 0.7)`` makes ``corr(H3, H1_R1) = 0.5``
    given the H1 arm loading ``sqrt(0.5)`` and scope loading
    ``sqrt(0.7)``.  The H2 latent loads ``sqrt(0.5)`` on the same shared
    factor, giving ``corr(H2, H1_R1) = sqrt(0.5 * 0.7 * 0.5) = 0.418``
    and ``corr(H2, H3) = sqrt(0.5) * h3_load = 0.598``.
    """
    u = rng.normal(size=n)
    scope_onset = math.sqrt(0.7) * u + math.sqrt(0.3) * rng.normal(size=n)
    scope_h5 = math.sqrt(0.7) * u + math.sqrt(0.3) * rng.normal(size=n)
    h1_arm_load = math.sqrt(0.5 * 0.7)
    h3_load = 0.5 / h1_arm_load
    z_h3 = h3_load * u + math.sqrt(1.0 - h3_load * h3_load) * rng.normal(
        size=n
    )
    z_h2 = math.sqrt(0.5) * u + math.sqrt(0.5) * rng.normal(size=n)
    return scope_onset, scope_h5, z_h3, z_h2


def calibrate_family(
    n: int,
    n_sim: int,
    seed: int,
    alternative: bool,
    split_n: int | None = None,
    h2_calibration_pass: bool | None = None,
) -> dict[str, object]:
    """Calibrate the four-slot family with shared animal dependence.

    The active slots enter the family; inactive slots enter Holm with
    ``p=1`` placeholders.  The null is the joint boundary; the
    alternative uses H1 ``p_improve=0.841`` (skill 0.10 at sigma 10),
    H3 ``p_improve=0.691`` (d_z 0.5) and H2 ``Delta=5 ms``.
    """
    active = cs.resolve_active_confirmatory_order(
        split_n, h2_calibration_pass
    )
    rng = np.random.default_rng(seed)
    sigma = 10.0
    sigma_h3 = 0.05
    sigma_h2 = 5.0
    h1_shift = sigma if alternative else 0.0
    h3_shift = 0.5 * sigma_h3 if alternative else 0.0
    h2_delta = 5.0 if alternative else cs.DELTA_NI_MS
    reject_any = np.zeros(n_sim, dtype=bool)
    reject_h1_gate = np.zeros(n_sim, dtype=bool)
    slot_rejections = {
        name: np.zeros(n_sim, dtype=bool) for name in cs.FAMILY_ORDER
    }
    realized_h1: list[np.ndarray] = []
    realized_h5: list[np.ndarray] = []
    realized_h3: list[np.ndarray] = []
    realized_h2: list[np.ndarray] = []
    for r in range(n_sim):
        scope_onset, scope_h5, z_h3, z_h2 = _joint_latents(rng, n)
        arm_onset = rng.normal(size=(n, 4))
        arm_h5 = rng.normal(size=(n, 4))
        d_onset = sigma * (
            math.sqrt(0.5) * scope_onset[:, None]
            + math.sqrt(0.5) * arm_onset
        )
        d_h5 = sigma * (
            math.sqrt(0.5) * scope_h5[:, None]
            + math.sqrt(0.5) * arm_h5
        )
        d_onset[:, 1] -= h1_shift
        d_h5[:, 1] -= h1_shift
        d_h3 = sigma_h3 * z_h3 - h3_shift
        persistence = np.full(n, MSE_BASE, dtype=np.float64)
        results: dict[str, cs.TestResult] = {
            "H1_onset": cs.descriptive_placeholder(
                "H1_onset", n_animal=n
            ),
            "H1_h5": cs.descriptive_placeholder("H1_h5", n_animal=n),
            "H2_NI": cs.descriptive_placeholder("H2_NI", n_animal=n),
            "H3_derecursion": cs.descriptive_placeholder(
                "H3_derecursion", n_animal=n
            ),
        }
        if "H1_onset" in active:
            results["H1_onset"] = _sim_call(cs.h1_foundation_persistence, 
                {
                    arm: np.maximum(persistence + d_onset[:, j], 0.0)
                    for j, arm in enumerate(cs.H1_FOUNDATION_ARMS)
                },
                persistence,
                "onset",
                n_boot=1,
                compute_ci=False,
            )
        if "H1_h5" in active:
            results["H1_h5"] = _sim_call(cs.h1_foundation_persistence, 
                {
                    arm: np.maximum(persistence + d_h5[:, j], 0.0)
                    for j, arm in enumerate(cs.H1_FOUNDATION_ARMS)
                },
                persistence,
                "h5",
                n_boot=1,
                compute_ci=False,
            )
        if "H3_derecursion" in active:
            c16 = np.full(n, MSE_BASE, dtype=np.float64)
            results["H3_derecursion"] = _sim_call(cs.h3_camor_derecursion, 
                np.maximum(c16 + d_h3, 0.0),
                c16,
                budget_matched_by_seed=(True, True, True),
                oom_by_seed=(False, False, False),
                n_boot=1,
                compute_ci=False,
            )
        if "H2_NI" in active:
            e_best = 100.0 + rng.normal(scale=8.0, size=n)
            e_ns = e_best + h2_delta + sigma_h2 * z_h2
            e_cls = {
                model: e_best + (0.0 if j == 0 else 20.0)
                for j, model in enumerate(cs.H2_MODELS)
            }
            results["H2_NI"] = _sim_call(cs.h2_non_inferior_best_classical, 
                e_ns,
                e_cls,
                analysis_set=cs._analysis_set_from_mask(
                    np.ones(n, dtype=bool)
                ),
                n_boot=1,
                compute_ci=False,
            )
        holm = cs.holm_confirmatory_family(
            {name: result.p_value for name, result in results.items()},
            multisensory_test_n=split_n,
            h2_calibration_pass=h2_calibration_pass,
        )
        reject_any[r] = any(holm.significant.values())
        reject_h1_gate[r] = bool(
            holm.significant["H1_onset"] and holm.significant["H1_h5"]
        )
        for name, value in holm.significant.items():
            slot_rejections[name][r] = bool(value)
        realized_h1.append(d_onset[:, 1])
        realized_h5.append(d_h5[:, 1])
        realized_h3.append(d_h3)
        realized_h2.append(z_h2)
    fwer = float(reject_any.mean())
    h1_all = np.concatenate(realized_h1)
    h5_all = np.concatenate(realized_h5)
    h3_all = np.concatenate(realized_h3)
    h2_all = np.concatenate(realized_h2)
    return {
        "n": n,
        "n_sim": n_sim,
        "alternative": alternative,
        "active_order": list(active),
        "seed": int(seed),
        "seed_base": SEED_BASE,
        "fwer_or_power": fwer,
        "cp95_upper": cp_upper(int(np.count_nonzero(reject_any)), n_sim),
        "mc_se": batch_mc_se(reject_any),
        "mc_se_binomial": mc_se(fwer, n_sim),
        "mc_se_method": "batch_means_20",
        "h1_gate_rate": float(reject_h1_gate.mean()),
        "slot_rejection_rates": {
            name: float(value.mean())
            for name, value in slot_rejections.items()
        },
        "h1_arm_correlation": 0.5,
        "h1_scope_correlation": 0.7,
        "h1_scope_realized_correlation": float(
            np.corrcoef(h1_all, h5_all)[0, 1]
        ),
        "h3_h1_correlation": 0.5,
        "h3_h1_realized_correlation": float(
            np.corrcoef(h1_all, h3_all)[0, 1]
        ),
        "h2_h1_correlation": float(
            math.sqrt(0.5 * 0.7 * 0.5)
        ),
        "h2_h1_realized_correlation": float(
            np.corrcoef(h1_all, h2_all)[0, 1]
        ),
        "h2_h3_correlation": float(
            math.sqrt(0.5) * 0.5 / math.sqrt(0.5 * 0.7)
        ),
        "h2_h3_realized_correlation": float(
            np.corrcoef(h3_all, h2_all)[0, 1]
        ),
        "h1_p_improve": 0.8413 if alternative else 0.5,
        "h3_p_improve": 0.6915 if alternative else 0.5,
        "h2_delta_best": h2_delta,
    }


def calibrate_loo_fragility(
    n: int,
    n_sim: int,
    seed: int,
    alternative: bool,
    split_n: int | None = None,
    h2_calibration_pass: bool | None = None,
    max_remove: int = cs.FRAGILITY_MAX_REMOVE,
    h2_delta: float | None = None,
    sigma_h2: float = 5.0,
) -> dict[str, object]:
    """Calibrate the descriptive LOO fragility index (no gate).

    User decision 2026-10-19 #1: leave-one-animal-out robustness is a
    descriptive fragility diagnostic, not a confirmatory gate.  This
    routine therefore reports only flip rates and the minimal flipping
    animal set; it deliberately has **no** power-with-gate /
    power-loss keys and no sign-off.

    Per simulated data set the frozen family is evaluated on the active
    slots, then :func:`nsmor.analysis.confirmatory_stats.loo_fragility_index`
    recomputes each slot under single-animal deletions and under
    simultaneous deletions up to ``max_remove``.  The default is the
    analysis-time constant
    :data:`nsmor.analysis.confirmatory_stats.FRAGILITY_MAX_REMOVE` (3),
    so the calibration and the frozen analysis use the same cap.

    Args:
        n: Animals per simulated data set (planned n).
        n_sim: Simulated data sets.
        seed: Random seed.
        alternative: Joint alternative (target effect) vs joint null.
        split_n: Multisensory test n used by the frozen active-order
            rule; ``None`` fails closed.
        h2_calibration_pass: One-shot H2 calibration verdict; ``None``
            fails closed.
        max_remove: Search cap for the minimal flipping set; defaults to
            :data:`nsmor.analysis.confirmatory_stats.FRAGILITY_MAX_REMOVE`.
        h2_delta: H2 effect used when H2 is active.
        sigma_h2: H2 latent SD.

    Returns:
        JSON-serialisable dictionary with ``fragile_rate_any``,
        ``mean_single_flip_fraction``, ``min_animals_to_flip_counts``
        (keys ``"1"``, ..., ``str(max_remove)``, ``"gt_max"``), and
        descriptive MC errors.  No gate semantics.
    """
    active = cs.resolve_active_confirmatory_order(
        split_n, h2_calibration_pass
    )
    max_remove = int(max_remove)
    if max_remove < 1:
        raise ValueError("max_remove must be at least 1")
    min_counts = {str(k): 0 for k in range(1, max_remove + 1)}
    min_counts["gt_max"] = 0
    if not active:
        return {
            "n": n,
            "n_sim": n_sim,
            "alternative": alternative,
            "active_order": [],
            "max_remove": max_remove,
            "fragile_rate_any": 0.0,
            "mean_single_flip_fraction": 0.0,
            "min_animals_to_flip_counts": min_counts,
            "mc_se": 0.0,
            "n_animal": 0,
            "seed": int(seed),
            "seed_base": SEED_BASE,
            "reason": "no_active_slots",
        }
    if h2_delta is None:
        h2_delta = 5.0 if alternative else cs.DELTA_NI_MS
    rng = np.random.default_rng(seed)
    fragile_any = np.zeros(n_sim, dtype=bool)
    flip_fractions = np.zeros(n_sim, dtype=np.float64)
    sigma = 10.0
    sigma_h3 = 0.05
    h1_shift = sigma if alternative else 0.0
    h3_shift = 0.5 * sigma_h3 if alternative else 0.0
    for r in range(n_sim):
        scope_onset, scope_h5, z_h3, z_h2 = _joint_latents(rng, n)
        arm_onset = rng.normal(size=(n, 4))
        arm_h5 = rng.normal(size=(n, 4))
        d_onset = sigma * (
            math.sqrt(0.5) * scope_onset[:, None]
            + math.sqrt(0.5) * arm_onset
        )
        d_h5 = sigma * (
            math.sqrt(0.5) * scope_h5[:, None]
            + math.sqrt(0.5) * arm_h5
        )
        d_onset[:, 1] -= h1_shift
        d_h5[:, 1] -= h1_shift
        d_h3 = sigma_h3 * z_h3 - h3_shift
        persistence = np.full(n, MSE_BASE, dtype=np.float64)
        by_onset = {
            arm: np.maximum(persistence + d_onset[:, j], 0.0)
            for j, arm in enumerate(cs.H1_FOUNDATION_ARMS)
        }
        by_h5 = {
            arm: np.maximum(persistence + d_h5[:, j], 0.0)
            for j, arm in enumerate(cs.H1_FOUNDATION_ARMS)
        }
        c16 = np.full(n, MSE_BASE, dtype=np.float64)
        c1s = np.maximum(c16 + d_h3, 0.0)
        e_best = 100.0 + rng.normal(scale=8.0, size=n)
        e_ns = e_best + h2_delta + sigma_h2 * z_h2
        e_cls = {
            model: e_best + (0.0 if j == 0 else 20.0)
            for j, model in enumerate(cs.H2_MODELS)
        }
        results: dict[str, cs.TestResult] = {
            "H1_onset": cs.descriptive_placeholder(
                "H1_onset", n_animal=n
            ),
            "H1_h5": cs.descriptive_placeholder("H1_h5", n_animal=n),
            "H2_NI": cs.descriptive_placeholder("H2_NI", n_animal=n),
            "H3_derecursion": cs.descriptive_placeholder(
                "H3_derecursion", n_animal=n
            ),
        }
        if "H1_onset" in active:
            results["H1_onset"] = _sim_call(cs.h1_foundation_persistence, 
                by_onset, persistence, "onset", n_boot=1, compute_ci=False
            )
        if "H1_h5" in active:
            results["H1_h5"] = _sim_call(cs.h1_foundation_persistence, 
                by_h5, persistence, "h5", n_boot=1, compute_ci=False
            )
        if "H2_NI" in active:
            results["H2_NI"] = _sim_call(cs.h2_non_inferior_best_classical, 
                e_ns,
                e_cls,
                analysis_set=cs._analysis_set_from_mask(
                    np.ones(n, dtype=bool)
                ),
                n_boot=1,
                compute_ci=False,
            )
        if "H3_derecursion" in active:
            results["H3_derecursion"] = _sim_call(cs.h3_camor_derecursion, 
                c1s,
                c16,
                budget_matched_by_seed=(True, True, True),
                oom_by_seed=(False, False, False),
                n_boot=1,
                compute_ci=False,
            )

        def recompute(slot: str, keep: np.ndarray) -> float:
            if slot == "H1_onset":
                return _sim_call(cs.h1_foundation_persistence, 
                    {
                        arm: by_onset[arm][keep]
                        for arm in cs.H1_FOUNDATION_ARMS
                    },
                    persistence[keep],
                    "onset",
                    n_boot=1,
                    compute_ci=False,
                ).p_value
            if slot == "H1_h5":
                return _sim_call(cs.h1_foundation_persistence, 
                    {
                        arm: by_h5[arm][keep]
                        for arm in cs.H1_FOUNDATION_ARMS
                    },
                    persistence[keep],
                    "h5",
                    n_boot=1,
                    compute_ci=False,
                ).p_value
            if slot == "H2_NI":
                return _sim_call(cs.h2_non_inferior_best_classical, 
                    e_ns,
                    e_cls,
                    analysis_set=cs._analysis_set_from_mask(keep),
                    n_boot=1,
                    compute_ci=False,
                ).p_value
            return _sim_call(cs.h3_camor_derecursion, 
                c1s[keep],
                c16[keep],
                budget_matched_by_seed=(True, True, True),
                oom_by_seed=(False, False, False),
                n_boot=1,
                compute_ci=False,
            ).p_value

        frag = cs.loo_fragility_index(
            results,
            recompute,
            multisensory_test_n=split_n,
            h2_calibration_pass=h2_calibration_pass,
            max_remove=max_remove,
        )
        slot_rates = [
            float(entry.loo_flip_rate)
            for entry in frag.values()
            if entry.available and entry.loo_flip_rate is not None
        ]
        fragile_any[r] = any(rate > 0.0 for rate in slot_rates)
        flip_fractions[r] = (
            float(np.mean(slot_rates)) if slot_rates else 0.0
        )
        minima = [
            entry.min_animals_to_flip
            for entry in frag.values()
            if entry.available
        ]
        found = [value for value in minima if value is not None]
        if not found:
            min_counts["gt_max"] += 1
        else:
            min_counts[str(min(found))] += 1
    fragile_rate = float(fragile_any.mean())
    return {
        "n": n,
        "n_sim": n_sim,
        "alternative": alternative,
        "active_order": list(active),
        "max_remove": int(max_remove),
        "fragile_rate_any": fragile_rate,
        "mean_single_flip_fraction": float(flip_fractions.mean()),
        "min_animals_to_flip_counts": min_counts,
        "mc_se": batch_mc_se(fragile_any),
        "mc_se_binomial": mc_se(fragile_rate, n_sim),
        "mc_se_method": "batch_means_20",
        "n_animal": n,
        "seed": int(seed),
        "seed_base": SEED_BASE,
        "h2_delta": float(h2_delta),
        "sigma_h2": float(sigma_h2),
        "reason": "",
        "inference": "descriptive",
        "affects_gate": False,
    }


def calibrate_h2_availability(
    n: int,
    n_sim: int,
    seed: int,
    p_obs_defined: float,
    p_ns_defined: float,
    p_model_ok: float,
    n_seed: int = 3,
) -> dict[str, object]:
    """Fast per-seed availability calibration (no ensemble draws).

    An animal enters A iff ``L_obs`` is defined, **all** ``n_seed``
    closed-loop seeds are defined, and all four classical models are
    usable.  This is the exact availability rule used by
    :func:`h2_realistic_draws`; the ensemble effective-draw check is
    omitted because it is effectively always satisfied at
    ``n_ens=200``.
    """
    rng = np.random.default_rng(seed)
    obs_ok = rng.random((n_sim, n)) < p_obs_defined
    ns_ok = np.all(
        rng.random((n_sim, n, n_seed)) < p_ns_defined, axis=2
    )
    model_ok = np.all(
        rng.random((n_sim, n, len(cs.H2_MODELS))) < p_model_ok, axis=2
    )
    mask = obs_ok & ns_ok & model_ok
    n_available = mask.sum(axis=1)
    available = float(np.mean(n_available >= cs.N_MIN))
    return {
        "n_planned": n,
        "n_sim": n_sim,
        "p_obs_defined": p_obs_defined,
        "p_ns_defined": p_ns_defined,
        "p_model_ok": p_model_ok,
        "n_seed": n_seed,
        "p_analysis_set_available": available,
        "mean_n_available": float(np.mean(n_available)),
        "mc_se": mc_se(available, n_sim),
    }


def _gate_verdict(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply the type-I gate to a set of records."""
    failed = [
        {
            "scenario": record["scenario"],
            "n": record.get("n", record.get("n_planned")),
            "type1": record["type1"],
            "cp95_upper": record["type1_cp95_upper"],
        }
        for record in records
        if not (
            math.isfinite(float(record["type1_cp95_upper"]))
            and float(record["type1_cp95_upper"]) <= TYPE1_GATE
        )
    ]
    return {
        "gate": TYPE1_GATE,
        "pass": len(failed) == 0,
        "failures": failed,
    }


def _h2_gate_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    """Select the main H2 gate records (central availability, n=15).

    Sensitivity records (informative missingness, heterogeneous
    ensembles, per-seed noise) and other sample sizes are not part of
    the pre-registered criterion.
    """
    selected: dict[str, Mapping[str, object]] = {}
    for record in records:
        if record.get("slot") != "H2_NI":
            continue
        scenario = str(record.get("scenario"))
        if scenario not in H2_GATE_SCENARIOS:
            continue
        n = record.get("n_planned", record.get("n"))
        try:
            n_value = int(n)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if n_value != cs.N_PLAN_H2:
            continue
        if bool(record.get("informative_missing", False)):
            continue
        if float(record.get("sigma_ens_spread", 0.0)) != 0.0:
            continue
        if float(record.get("sigma_seed", 0.0)) != 0.0:
            continue
        selected[scenario] = record
    return selected


def _as_exact_int(value: object) -> int | None:
    """Return an integer value, rejecting bools and non-integers."""
    if isinstance(value, bool) or not isinstance(
        value, (int, np.integer)
    ):
        return None
    return int(value)


def h2_calibration_verdict(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply the pre-registered H2 calibration criterion.

    User decision 2026-10-19 #2: with the frozen replicate count
    :data:`H2_CALIBRATION_R` and the frozen new seed
    :data:`H2_CALIBRATION_SEED`, the H2 gate passes iff **every** gate
    scenario (``S0``, ``S1``, ``S2``; S1 explicitly included) at the
    planned ``n=15`` has an observed type-I rate whose one-sided 95%
    Clopper-Pearson upper bound is ``<= TYPE1_GATE`` (0.0175).

    The verdict also **verifies the provenance of every gate record**:
    ``n_sim`` must equal :data:`H2_CALIBRATION_R` and ``seed_base`` must
    equal :data:`H2_CALIBRATION_SEED`.  A missing or mismatched field
    fails closed, so an auxiliary run with a smaller replicate count or
    a different seed can never be reported as the one-shot criterion.
    The returned ``r_h2`` and ``seed`` are therefore the *verified*
    values; they are ``None`` when the provenance check fails, rather
    than being echoed unconditionally.

    A missing gate scenario fails closed.  A failed criterion downgrades
    H2 permanently, regardless of the split n; the criterion is run once
    and is never re-seeded or re-run.

    Args:
        records: Calibration records (typically the ``type1`` list).
            Records that are not main H2 gate records are ignored.

    Returns:
        Dictionary with ``gate``, ``pass``, ``failures``,
        ``scenarios``, ``r_h2``, ``seed``, ``n_planned``, ``verified``,
        ``observed_n_sim``, and ``observed_seed_base``.

    Raises:
        ValueError: If ``records`` is empty.
    """
    record_list = list(records)
    if not record_list:
        raise ValueError("h2_calibration_verdict needs at least one record")
    by_scenario = _h2_gate_records(record_list)
    failures: list[dict[str, object]] = []
    observed_n_sim: set[int] = set()
    observed_seed_base: set[int] = set()
    for scenario in H2_GATE_SCENARIOS:
        record = by_scenario.get(scenario)
        if record is None:
            failures.append(
                {"scenario": scenario, "reason": "missing_gate_record"}
            )
            continue
        n_sim = _as_exact_int(record.get("n_sim"))
        seed_base = _as_exact_int(record.get("seed_base"))
        if n_sim is not None:
            observed_n_sim.add(n_sim)
        if seed_base is not None:
            observed_seed_base.add(seed_base)
        if n_sim != H2_CALIBRATION_R:
            failures.append(
                {
                    "scenario": scenario,
                    "reason": "n_sim_mismatch",
                    "n_sim": record.get("n_sim"),
                    "required_n_sim": H2_CALIBRATION_R,
                }
            )
            continue
        if seed_base != H2_CALIBRATION_SEED:
            failures.append(
                {
                    "scenario": scenario,
                    "reason": "seed_base_mismatch",
                    "seed_base": record.get("seed_base"),
                    "required_seed_base": H2_CALIBRATION_SEED,
                }
            )
            continue
        upper = record.get("type1_cp95_upper")
        try:
            upper_value = float(upper)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            failures.append(
                {
                    "scenario": scenario,
                    "reason": "cp_upper_missing",
                    "type1_cp95_upper": upper,
                }
            )
            continue
        if not (math.isfinite(upper_value) and upper_value <= TYPE1_GATE):
            failures.append(
                {
                    "scenario": scenario,
                    "reason": "cp_upper_above_gate",
                    "type1_cp95_upper": upper_value,
                    "gate": TYPE1_GATE,
                    "n_sim": n_sim,
                    "seed_base": seed_base,
                }
            )
    verified = len(failures) == 0
    return {
        "gate": TYPE1_GATE,
        "pass": verified,
        "failures": failures,
        "scenarios": list(H2_GATE_SCENARIOS),
        "r_h2": H2_CALIBRATION_R if verified else None,
        "seed": H2_CALIBRATION_SEED if verified else None,
        "n_planned": cs.N_PLAN_H2,
        "verified": verified,
        "observed_n_sim": sorted(observed_n_sim),
        "observed_seed_base": sorted(observed_seed_base),
        "criterion": (
            "every gate scenario has n_sim == H2_CALIBRATION_R, "
            "seed_base == H2_CALIBRATION_SEED, and "
            "type1_cp95_upper <= TYPE1_CP95_GATE"
        ),
    }


def build_g1_power_table() -> dict[str, object]:
    """Return the G1 24-dataset exact binomial operating characteristics.

    The G1 design gate flags a gross false-positive rate when the
    one-sided 95% Clopper-Pearson lower bound of the observed error rate
    exceeds 0.20.  With ``n_mis=24`` the smallest such count is
    ``k_min=9`` (``L_CP(9/24)=0.2116``).  For each true dataset-level
    false-positive rate ``p`` the probability of crossing the gate is
    ``P(K >= 9)`` with ``K ~ Binomial(24, p)``.

    Returns:
        JSON-serialisable dictionary carrying both the protocol key
        spellings (``true_fpr``/``p_flag_gross_error``) and the legacy
        aliases (``true_dataset_fpr``/``alarm_power_k_ge_9``) so the
        frozen calibration JSON, the protocol table, and the tester
        probe read the same values.
    """
    grid = (
        0.01,
        0.03,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
        0.60,
    )
    rows: list[dict[str, float]] = []
    for p in grid:
        alarm = float(stats.binom.sf(8, 24, p))
        rows.append(
            {
                "true_fpr": p,
                "true_dataset_fpr": p,
                "p_flag_gross_error": alarm,
                "alarm_power_k_ge_9": alarm,
                "expected_false_positive_datasets": 24.0 * p,
            }
        )
    return {
        "status": "computed",
        "reason": (
            "exact Binomial(24, p_true) operating characteristics; "
            "k_min=9 from the one-sided 95% Clopper-Pearson rule"
        ),
        "method": "exact_binomial_operating_characteristics",
        "n_datasets": 24,
        "n_mis": 24,
        "threshold_k_min": 9,
        "k_min": 9,
        "clopper_pearson_lower_at_threshold": float(
            stats.beta.ppf(0.05, 9, 24 - 9 + 1)
        ),
        "null_fpr_empirical": 0.0433,
        "criterion": "P(K >= 9) where K ~ Binomial(24, p_true)",
        "operating_characteristics": rows,
    }


def run_calibration_suite(
    r_sim: int = 20_000,
    r_loo: int = 1_000,
    quick: bool = False,
    h2_split_n: int | None = 13,
    r_h2: int | None = None,
    r_h2_aux: int = H2_AUX_R,
    h2_n_ens: int = 50,
) -> dict[str, object]:
    """Run the pre-registered calibration matrix and return all records.

    r8 (2026-10-20) changes:

    * H1/H3 type-I gate cells and power curves use the frozen split
      ``n = N_PLAN_H1_H3 = 18``; ``n=13``, ``n=16`` (r9 formula grid)
      and ``n=20`` remain as sensitivity points.  Every power record
      carries the explicit ``underpowered_at_target`` flag.
    * The three H2 gate cells (``S0``/``S1``/``S2`` at ``n=15``) use the
      frozen criterion replicates ``r_h2`` (default
      :data:`H2_CALIBRATION_R`, 200,000) and seeds spawned from the
      frozen new seed :data:`H2_CALIBRATION_SEED`.  The criterion is run
      once and is never re-seeded.  Gate records carry ``n_sim`` and
      ``seed_base`` so :func:`h2_calibration_verdict` can verify the
      provenance instead of echoing the constants.
    * H2 curve / sensitivity / power cells use ``r_h2_aux`` (default
      :data:`H2_AUX_R`) with the same H2 seed stream.
    * LOO is a descriptive fragility diagnostic, not a gate: the suite
      reports ``loo_fragility`` records with the same
      :data:`nsmor.analysis.confirmatory_stats.FRAGILITY_MAX_REMOVE`
      cap as the analysis-time index, and no power-loss keys.

    Args:
        r_sim: Replicates per H1/H3 type-I and power record.
        r_loo: Replicates for the descriptive LOO fragility records.
        quick: When true, use small replicate counts for smoke testing.
        h2_split_n: Split-fixed multisensory test-animal count; the
            frozen rule excludes H2 when it is missing or ``< 15``.
        r_h2: Replicates per H2 gate cell.  ``None`` uses the frozen
            :data:`H2_CALIBRATION_R`.
        r_h2_aux: Replicates per H2 curve/sensitivity/power cell.
        h2_n_ens: Synthetic ensemble size for the H2 DGP.

    Returns:
        Dictionary with ``type1``, ``power``, ``family``,
        ``loo_fragility``, ``h2_availability``, ``verdicts``,
        ``active_order``, ``h2_split_feasibility``,
        ``h2_calibration``, and ``g1_power_table``.
    """
    if r_h2 is None:
        r_h2 = H2_CALIBRATION_R
    if quick:
        r_sim = min(r_sim, 300)
        r_h2 = min(r_h2, 300)
        r_h2_aux = min(r_h2_aux, 300)
        r_loo = min(r_loo, 50)
    n_h1 = (10, 13, 15, 16, 18, 20)
    n_h2 = (10, 13, 14, 15, 18)
    n_h3 = (10, 13, 15, 16, 18, 20)
    type1_scenarios = ("S0", "S1", "S2")
    sensitivity_scenarios = ("S3", "S4")
    seed_count = 4_000
    seeds = spawn_seeds(SEED_BASE, seed_count)
    h2_seed_count = 64
    h2_seeds = spawn_seeds(H2_CALIBRATION_SEED, h2_seed_count)
    cursor = 0
    h2_cursor = 0

    def take() -> int:
        nonlocal cursor
        value = seeds[cursor]
        cursor += 1
        return value

    def take_h2() -> int:
        nonlocal h2_cursor
        value = h2_seeds[h2_cursor]
        h2_cursor += 1
        return value

    type1: list[dict[str, object]] = []
    # H1 onset/h5: gate scenarios S0/S1/S2 and S6; n curves; S3/S4
    # sensitivity; tie sensitivity.
    for scope in ("onset", "h5"):
        for scenario in type1_scenarios + ("S6",) + sensitivity_scenarios:
            for n in n_h1:
                type1.append(
                    {
                        "slot": f"H1_{scope}",
                        **calibrate_h1_null(
                            n, scenario, r_sim, take(), scope=scope
                        ),
                    }
                )
        type1.append(
            {
                "slot": f"H1_{scope}",
                **calibrate_h1_null(
                    20,
                    "S1",
                    r_sim,
                    take(),
                    scope=scope,
                    p_tie=0.10,
                ),
                "scenario": "S1_ties_10pct",
            }
        )
    # H3: same structure.
    for scenario in type1_scenarios + ("S6",) + sensitivity_scenarios:
        for n in n_h3:
            type1.append(
                {
                    "slot": "H3_derecursion",
                    **calibrate_h3_null(n, scenario, r_sim, take()),
                }
            )
    type1.append(
        {
            "slot": "H3_derecursion",
            **calibrate_h3_null(20, "S1", r_sim, take(), p_tie=0.10),
            "scenario": "S1_ties_10pct",
        }
    )
    # H2 gate cells: the frozen criterion (R_H2, new seed, S0/S1/S2 at
    # the planned n=15).  Curve/sensitivity cells use the auxiliary
    # replicate count on the same H2 seed stream.
    for scenario in type1_scenarios:
        type1.append(
            {
                "slot": "H2_NI",
                **calibrate_h2_realistic(
                    15, scenario, r_h2, take_h2(), n_ens=h2_n_ens
                ),
            }
        )
    for n in (10, 13, 18):
        type1.append(
            {
                "slot": "H2_NI",
                **calibrate_h2_realistic(
                    n, "S1", r_h2_aux, take_h2(), n_ens=h2_n_ens
                ),
            }
        )
    for scenario in sensitivity_scenarios:
        type1.append(
            {
                "slot": "H2_NI",
                **calibrate_h2_realistic(
                    15, scenario, r_h2_aux, take_h2(), n_ens=h2_n_ens
                ),
            }
        )
    type1.append(
        {
            "slot": "H2_NI",
            **calibrate_h2_realistic(
                15,
                "S5",
                r_h2_aux,
                take_h2(),
                sigma_seed=2.0,
                n_ens=h2_n_ens,
            ),
            "scenario": "S5_seed_noise",
        }
    )
    type1.append(
        {
            "slot": "H2_NI",
            **calibrate_h2_realistic(
                15,
                "S5",
                r_h2_aux,
                take_h2(),
                informative_missing=True,
                n_ens=h2_n_ens,
            ),
            "scenario": "S5_informative_missing",
        }
    )
    type1.append(
        {
            "slot": "H2_NI",
            **calibrate_h2_realistic(
                15,
                "S5",
                r_h2_aux,
                take_h2(),
                sigma_ens_spread=0.5,
                n_ens=h2_n_ens,
            ),
            "scenario": "S5_hetero_ensemble",
        }
    )

    power: list[dict[str, object]] = []
    for scope in ("onset", "h5"):
        for scenario in type1_scenarios:
            for n in (13, cs.N_PLAN_H1_H3, 20):
                for p_improve in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
                    power.append(
                        {
                            "slot": f"H1_{scope}",
                            **calibrate_h1_power(
                                n,
                                scenario,
                                r_sim,
                                take(),
                                p_improve,
                                scope=scope,
                            ),
                        }
                    )
    for scenario in type1_scenarios:
        for n in (13, cs.N_PLAN_H1_H3, 20):
            for p_improve in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
                power.append(
                    {
                        "slot": "H3_derecursion",
                        **calibrate_h3_power(
                            n, scenario, r_sim, take(), p_improve
                        ),
                    }
                )
    for scenario in type1_scenarios:
        for delta_best in (15.0, 9.0, 5.0, 0.0):
            power.append(
                {
                    "slot": "H2_NI",
                    **calibrate_h2_power(
                        15,
                        scenario,
                        r_h2_aux,
                        take_h2(),
                        delta_best,
                        n_ens=h2_n_ens,
                    ),
                }
            )

    gate_records: dict[str, list[dict[str, object]]] = {
        "H1_onset": [
            record
            for record in type1
            if record["slot"] == "H1_onset"
            and record["scenario"] in ("S0", "S1", "S2", "S6")
            and record["n"] == cs.N_PLAN_H1_H3
            and float(record.get("p_tie", 0.0)) == 0.0
        ],
        "H1_h5": [
            record
            for record in type1
            if record["slot"] == "H1_h5"
            and record["scenario"] in ("S0", "S1", "S2", "S6")
            and record["n"] == cs.N_PLAN_H1_H3
            and float(record.get("p_tie", 0.0)) == 0.0
        ],
        "H2_NI": [
            record
            for record in type1
            if record["slot"] == "H2_NI"
            and record["scenario"] in H2_GATE_SCENARIOS
            and record.get("n_planned") == cs.N_PLAN_H2
            and not bool(record.get("informative_missing", False))
            and float(record.get("sigma_ens_spread", 0.0)) == 0.0
            and float(record.get("sigma_seed", 0.0)) == 0.0
        ],
        "H3_derecursion": [
            record
            for record in type1
            if record["slot"] == "H3_derecursion"
            and record["scenario"] in ("S0", "S1", "S2", "S6")
            and record["n"] == cs.N_PLAN_H1_H3
            and float(record.get("p_tie", 0.0)) == 0.0
        ],
    }
    verdicts = {
        "type1": {
            slot: _gate_verdict(records)
            for slot, records in gate_records.items()
        }
    }
    h2_calibration = h2_calibration_verdict(gate_records["H2_NI"])
    verdicts["h2_calibration"] = h2_calibration
    h2_resolved = cs.resolve_active_confirmatory_order(
        h2_split_n, bool(h2_calibration["pass"])
    )
    h2_eligible = "H2_NI" in h2_resolved
    if h2_split_n is None:
        h2_reason = "split_n_missing"
    elif int(h2_split_n) < cs.H2_MIN_TEST_N:
        h2_reason = "expected_split_n_lt_15"
    elif not h2_calibration["pass"]:
        h2_reason = "h2_calibration_failed"
    else:
        h2_reason = ""
    h2_split_feasibility = {
        "h2_split_n": h2_split_n,
        "min_feasible_n": cs.H2_MIN_TEST_N,
        "calibration_pass": bool(h2_calibration["pass"]),
        "calibration_seed": H2_CALIBRATION_SEED,
        "calibration_r": r_h2,
        "eligible": h2_eligible,
        "reason": h2_reason,
        "rule": (
            "resolve_active_confirmatory_order("
            "split_multisensory_test_n, h2_calibration_pass)"
        ),
    }
    active_order = cs.resolve_active_confirmatory_order(
        h2_split_n, bool(h2_calibration["pass"])
    )
    family = [
        {
            "scenario": "S7",
            "alternative": False,
            **calibrate_family(
                n=cs.N_PLAN_H1_H3,
                n_sim=r_sim,
                seed=take(),
                alternative=False,
                split_n=h2_split_n,
                h2_calibration_pass=bool(h2_calibration["pass"]),
            ),
        },
        {
            "scenario": "S8",
            "alternative": True,
            **calibrate_family(
                n=cs.N_PLAN_H1_H3,
                n_sim=r_sim,
                seed=take(),
                alternative=True,
                split_n=h2_split_n,
                h2_calibration_pass=bool(h2_calibration["pass"]),
            ),
        },
    ]
    verdicts["family_fwer"] = {
        "gate": FWER_GATE,
        "pass": bool(family[0]["cp95_upper"] <= FWER_GATE),
        "observed": family[0]["fwer_or_power"],
        "cp95_upper": family[0]["cp95_upper"],
    }
    loo_fragility = [
        {
            "alternative": False,
            **calibrate_loo_fragility(
                n=cs.N_PLAN_H1_H3,
                n_sim=r_loo,
                seed=take(),
                alternative=False,
                split_n=h2_split_n,
                h2_calibration_pass=bool(h2_calibration["pass"]),
            ),
        },
        {
            "alternative": True,
            **calibrate_loo_fragility(
                n=cs.N_PLAN_H1_H3,
                n_sim=r_loo,
                seed=take(),
                alternative=True,
                split_n=h2_split_n,
                h2_calibration_pass=bool(h2_calibration["pass"]),
            ),
        },
    ]
    verdicts["loo_fragility"] = {
        "active_order": list(active_order),
        "descriptive_only": True,
        "affects_gate": False,
        "requires_signoff": False,
        "fragile_rate_any": {
            "null": loo_fragility[0]["fragile_rate_any"],
            "alternative": loo_fragility[1]["fragile_rate_any"],
        },
        "mean_single_flip_fraction": {
            "null": loo_fragility[0]["mean_single_flip_fraction"],
            "alternative": loo_fragility[1]["mean_single_flip_fraction"],
        },
        "note": (
            "LOO fragility is reported for description only; it is not "
            "a gate and has no sign-off (user decision 2026-10-19 #1)"
        ),
    }
    h2_availability: list[dict[str, object]] = []
    assumptions = (
        ("optimistic", 0.95, 0.95, 0.99),
        ("central", 0.90, 0.90, 0.98),
        ("conservative", 0.85, 0.85, 0.95),
    )
    for label, p_obs, p_ns, p_model in assumptions:
        for n in n_h2:
            record = calibrate_h2_availability(
                n, r_sim, take(), p_obs, p_ns, p_model
            )
            record["assumption"] = label
            h2_availability.append(record)
    record = calibrate_h2_availability(
        15, r_sim, take(), 0.90, 0.90, 0.98
    )
    record["assumption"] = "central_n15_check"
    h2_availability.append(record)
    g1_power_table = build_g1_power_table()
    return {
        "schema_version": "5.0",
        "r_sim": r_sim,
        "r_h2": r_h2,
        "r_h2_gate": r_h2,
        "r_h2_aux": r_h2_aux,
        "r_loo": r_loo,
        "seed_base": SEED_BASE,
        "h2_calibration_seed": H2_CALIBRATION_SEED,
        "seed_count": seed_count,
        "type1": type1,
        "power": power,
        "family": family,
        "loo_fragility": loo_fragility,
        "h2_availability": h2_availability,
        "h2_split_feasibility": h2_split_feasibility,
        "h2_calibration": h2_calibration,
        "active_order": list(active_order),
        "g1_power_table": g1_power_table,
        "verdicts": verdicts,
    }
