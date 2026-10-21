"""Pre-registered confirmatory statistics for real-data Phase 3.

This module is the single executable definition of the four confirmatory
slots fixed in the Phase-3 protocol.  The protocol text cites these
functions and this file's SHA-256 instead of restating formulas, so the
implementation is the normative object under review.

Design contract
---------------
* Inferential unit: one ``Cercus animal_key`` per observation.  The
  caller aggregates seeds and trials before calling these functions.
* Primary construction: **exact one-sided binomial sign test** on
  animal-level paired differences.  For differences ``d_i`` and a
  boundary ``delta`` the null is ``P(d_i < delta) <= 0.5`` and the
  alternative is ``P(d_i < delta) > 0.5``.  The p-value is
  ``P(X >= k)`` with ``X ~ Binomial(n, 0.5)`` and ``k = #{d_i < delta}``.
  The test is exact in finite samples and does not assume symmetry,
  normality, or a finite variance.
* Scientific estimand: the **median** paired difference (equivalently,
  the proportion of animals that improve).  This is a deliberate change
  from the earlier animal-level paired t-test, which estimated the mean
  and was not calibrated under skew at the planned n.  Confirmatory
  claims therefore read "a majority of animals improve", not "the mean
  difference is negative".
* Ties (``d_i == delta`` exactly) are counted as **non-improvements**:
  they stay in the denominator but not in ``k``.  This rule is
  conservative and is fixed before unblinding.
* H1: the foundation arms are the newly rerun ``R0-R3`` family.  The
  only confirmatory construction is the **4-arm Bonferroni union** over
  ``{R0, R1, R2, R3}``: ``p = min(1.0, 4.0 * min_a p_a)``.  This tests
  whether *at least one* foundation arm beats persistence on the given
  scope.  The onset and h5 scopes may be satisfied by different winning
  arms (consistent with Phase 2).  The reported statistic and CI are
  those of the selected winner and are therefore explicitly labelled
  winner-conditional.  There is **no single-arm path**: all four arms
  are required, a missing, mis-shaped, negative, or non-finite arm
  raises instead of silently dropping the arm and changing the winner.
  Crucial constructive caveat: ``R1`` is residual-over-persistence
  (``v(t-1) + f``).  If ``R1`` is the winning arm, its advantage is
  explicitly reported as residual skill over the trivial copy, not
  evidence that the autonomous recurrent core beats persistence.  The
  ``unstable_across_seeds`` filter is applied by
  :func:`h1_seed_sign_consistency`, which fixes the winning arm on the
  seed-averaged data before evaluating the per-seed signs.  The effect
  reports ``skill = 1 - mean(MSE_winner) / mean(MSE_persistence)``.
* H2: for each classical model ``m``,
  ``d_{i,m} = e_NS,i - e_cls,i,m - DELTA_NI_MS``.  The slot is an
  intersection-union test over the four pre-specified models:
  ``p = max_m p_m`` (Berger 1982, *JASA* 77:503-513).  This tests
  non-inferiority against *every* pre-specified classical model; the
  "best classical model" is a descriptive label (largest median
  ``d``), never a data-dependent comparator that would invalidate the
  sign test.  The estimand is the **median** paired absolute-error
  difference; the test is a majority-of-animals sign test, never a mean
  test.  :func:`h2_non_inferior_best_classical` requires the
  :class:`AnalysisSet` returned by :func:`build_h2_analysis_set` and
  applies its mask; the NI margin is the frozen module constant and is
  not an overridable argument.  The effect also reports a worst-case
  sensitivity that counts every animal excluded from A as a
  non-improvement; the report uses it as an additional gate for a
  positive H2 claim.
* H3: exact sign test on ``D_i = MSE_{i,C1-S} - MSE_{i,C16}``.  The
  per-seed budget comparison and the K=16 OOM guard are **required**
  arguments and enforced inside :func:`h3_camor_derecursion`; a
  missing, invalid, or NOTMATCHED status fails closed with ``p=1`` and
  cannot be bypassed by omitting a default.  The effect reports every
  per-seed median difference and the seed-sign consistency; the report
  withholds a positive claim when the per-seed signs disagree.
* Degenerate inputs fail closed: ``available=False`` and
  ``p_value=1.0``.  A zero-variance sample is not degenerate for a sign
  test; all-positive differences give ``p=1`` and all-negative
  differences give ``p=2**-n``.
* Holm family: four frozen slots; unavailable or downgraded slots enter
  with ``p=1`` and occupy their position.  Slots outside the resolved
  active order are rejected unless their p-value is exactly ``1.0``.
  The active order is produced by the frozen rule function
  :func:`resolve_active_confirmatory_order`: at unblinding the H2 slot
  is confirmatory only when the PG-02 split SHA fixes at least
  :data:`H2_MIN_TEST_N` multisensory test animals **and** the
  fixed-seed H2 calibration (``H2_CALIBRATION_R`` replicates with
  :data:`H2_CALIBRATION_SEED`) passes its pre-registered criterion in
  every gate scenario including S1.  Missing or invalid split facts
  fail closed to :data:`FAIL_CLOSED_ACTIVE_ORDER` (H2 excluded, p=1).
  The Monte Carlo guard, when p-values are Monte-Carlo estimated,
  compares the *adjusted* p-value against ``alpha`` using the Holm
  multiplier (``m - rank``) times the raw MC standard error, and an
  indeterminate slot is never rejected.
* Leave-one-animal-out (LOO) fragility is **descriptive only** (user
  decision 2026-10-19): :func:`loo_fragility_index` reports, per active
  slot, the single-deletion flip rate and the minimal number of
  simultaneously removed animals that flips the frozen Holm verdict.
  It is not a gate, requires no sign-off, and never changes the
  family decision.
* Descriptive statistics report a point estimate and an animal-cluster
  percentile bootstrap CI only; they never carry a p-value, never
  enter Holm, and are explicitly labelled as having **no multiplicity
  control**.  A descriptive CI cannot support a superiority,
  non-inferiority, or equivalence claim for a slot that did not pass
  its confirmatory gate.

Frozen constants live at module level and are asserted by the tests.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import stats

from nsmor.analysis.uq import bootstrap_ci, holm_bonferroni

# ---------------------------------------------------------------------------
# Frozen constants (protocol v3)
# ---------------------------------------------------------------------------

DELTA_NI_MS: float = 15.0
ALPHA: float = 0.05
FAMILY_M: int = 4
FAMILY_ORDER: tuple[str, ...] = (
    "H1_onset",
    "H1_h5",
    "H2_NI",
    "H3_derecursion",
)
# Fail-closed active order: used whenever the split facts are missing,
# invalid, or the H2 calibration has not passed.  The confirmatory active
# order is NOT hardcoded from a predicted n (user decision 2026-10-19
# #2); it is produced by :func:`resolve_active_confirmatory_order` from
# the split SHA's realised multisensory test n.  A downgraded slot must
# enter Holm with p=1.0 and still occupies its position (m stays 4).
FAIL_CLOSED_ACTIVE_ORDER: tuple[str, ...] = (
    "H1_onset",
    "H1_h5",
    "H3_derecursion",
)
N_MIN: int = 10
# Frozen PG-02 split (2026-10-20, split SHA 380c81de...).  The SHA and
# the realised multisensory test n are binding facts, not caller input:
# `confirmatory_report` accepts a split SHA and refuses to run the frozen
# split with any other multisensory n.  Synthetic strings prefixed with
# `synthetic:` exist only for simulation tests and never denote the
# frozen split.
FROZEN_SPLIT_SHA256: str = (
    "380c81de8232de813624ab119877701e06bc134270827c74d9caee7ad0afde59"
)
FROZEN_MULTISENSORY_TEST_N: int = 14
SYNTHETIC_SPLIT_SHA_PREFIX: str = "synthetic:"
# The h5 persistence baseline is fixed to the current target v(t), not
# the previous position v(t-1).  The caller must declare it; the slot
# rejects any other value before scoring.
PERSISTENCE_BASELINES: tuple[str, ...] = ("v_t",)
# H1/H3 confirmatory population: every test animal with a defined
# scoring window, aggregated across conditions at the animal level
# (multisensory + bv; the frozen split has no wind_only test animal).
# Frozen PG-02 split: 18 test animals = 14 multisensory
# (7 looming_wind strata x 2) + 4 bv; bw = 0.
# The per-stratum formula is n_test = min(half_up(0.2 n),
# max(0, n - min_non_test)) with min_non_test=8 (bw 10).  The earlier
# 6*2 + 4 = 16 derivation treated +200 as a <8-animal stratum; after the
# +200 metadata QC it contributes 2 test animals, so the formula gives
# 18.  n=16 and 13/20 remain in the calibration range as sensitivity
# rows, never as the frozen primary anchor.
N_PLAN_H1_H3: int = 18
N_PLAN_H1_H3_RANGE: tuple[int, ...] = (13, 16, 18, 20)
# H2 is defined on multisensory test animals only: 14 < 15, so the
# frozen rule returns FAIL_CLOSED_ACTIVE_ORDER and H2 is descriptive.
# n=15 is a hypothetical calibration design point, not the planned n.
N_PLAN_H2: int = 15
N_PLAN_H2_RANGE: tuple[int, ...] = (10, 13, 15, 18)
CI_LEVEL: float = 0.95
CI_BOOT: int = 20_000
SEED_BASE: int = 20261019

# H2 activation and calibration gate (user decision 2026-10-19 #2).
# The H2_NI slot is confirmatory only when BOTH conditions hold:
#   (1) the PG-02 split SHA fixes at least H2_MIN_TEST_N multisensory
#       test animals, and
#   (2) the one-shot H2 calibration, run with the frozen replicate count
#       H2_CALIBRATION_R and the frozen seed H2_CALIBRATION_SEED, gives a
#       one-sided 95% Clopper-Pearson upper bound <= TYPE1_CP95_GATE in
#       EVERY gate scenario (S0, S1, S2; S1 explicitly included).
# A failed criterion downgrades H2 permanently, regardless of n.
H2_MIN_TEST_N: int = 15
H2_CALIBRATION_R: int = 200_000
H2_CALIBRATION_SEED: int = 20261219
TYPE1_CP95_GATE: float = 0.0175

# Search cap for the descriptive LOO fragility index.
FRAGILITY_MAX_REMOVE: int = 3

# Effect-dictionary keys derived from p-values.  They are stripped from
# downgraded slots in :func:`confirmatory_report`: a descriptive row must
# never carry a p-derived label.
P_DERIVED_EFFECT_KEYS: tuple[str, ...] = (
    "p_determining_model",
    "model_p_values",
)

# H1 foundation arms are the newly rerun R0-R3 family, not the Phase-3
# C0-C3 family.  All four are required; there is no single-arm
# confirmatory path (r9 finding: the arm argument let a caller drop the
# four-arm Bonferroni multiplier).
H1_FOUNDATION_ARMS: tuple[str, ...] = ("R0", "R1", "R2", "R3")
H2_MODELS: tuple[str, ...] = ("race", "coactivation", "gain", "TWIN")
H1_SCOPES: tuple[str, ...] = ("onset", "h5")
SLOT_BOUNDARIES: Mapping[str, float] = {
    "H1_onset": 0.0,
    "H1_h5": 0.0,
    "H2_NI": DELTA_NI_MS,
    "H3_derecursion": 0.0,
}
TIE_POLICY: str = "non_improvement"

assert ALPHA == 0.05, "holm_bonferroni in nsmor.analysis.uq hardcodes 0.05"
assert len(FAMILY_ORDER) == FAMILY_M
assert len(H1_FOUNDATION_ARMS) == 4
assert len(H2_MODELS) == 4
assert set(SLOT_BOUNDARIES) == set(FAMILY_ORDER)
assert FAIL_CLOSED_ACTIVE_ORDER == tuple(
    slot for slot in FAMILY_ORDER if slot != "H2_NI"
)
assert TYPE1_CP95_GATE == ALPHA / FAMILY_M + 0.005
assert H2_CALIBRATION_SEED != SEED_BASE
assert H2_CALIBRATION_R >= 200_000


# ---------------------------------------------------------------------------
# Frozen active-order rule (user decision 2026-10-19 #2)
# ---------------------------------------------------------------------------


def resolve_active_confirmatory_order(
    multisensory_test_n: int | None,
    h2_calibration_pass: bool | None,
    *,
    split_sha256: str | None = None,
) -> tuple[str, ...]:
    """Resolve the frozen confirmatory active order from the split facts.

    The active order is never hardcoded from a predicted n.  It is a
    deterministic function of exactly two frozen inputs:

    * ``multisensory_test_n``: the number of multisensory test animals
      fixed by the PG-02 split SHA (``None`` before the split exists);
    * ``h2_calibration_pass``: whether the one-shot H2 calibration with
      the frozen replicate count :data:`H2_CALIBRATION_R` and the frozen
      seed :data:`H2_CALIBRATION_SEED` satisfied the pre-registered
      criterion in every gate scenario.

    H2 enters the confirmatory family only when
    ``multisensory_test_n >= H2_MIN_TEST_N`` **and**
    ``h2_calibration_pass`` is exactly ``True``.  Anything else --
    including missing **or invalid** split facts -- fails closed to
    :data:`FAIL_CLOSED_ACTIVE_ORDER`, in which H2 occupies its frozen
    slot with ``p=1.0`` and ``m`` stays 4.  Invalid types are therefore
    not raised: a malformed split fact must never promote H2 and must
    not abort the unblinding report.

    Args:
        multisensory_test_n: Split-fixed multisensory test animal count,
            or ``None`` when the split is not yet frozen.  A bool or a
            non-integer value is treated as an invalid split fact and
            fails closed.
        h2_calibration_pass: H2 calibration verdict, or ``None`` when the
            one-shot calibration has not run.  A non-bool value is
            treated as an invalid verdict and fails closed.
        split_sha256: Optional split SHA.  When supplied, the frozen
            PG-02 SHA binds ``multisensory_test_n`` to 14; a mismatch
            raises.  ``synthetic:`` fixtures are simulations and are not
            checked against the frozen split.

    Returns:
        The active slots in frozen family order:
        :data:`FAMILY_ORDER` when H2 is eligible, otherwise
        :data:`FAIL_CLOSED_ACTIVE_ORDER`.
    """
    if split_sha256 is not None:
        _check_split_facts(split_sha256, multisensory_test_n)
    if multisensory_test_n is not None:
        if isinstance(multisensory_test_n, bool) or not isinstance(
            multisensory_test_n, (int, np.integer)
        ):
            return FAIL_CLOSED_ACTIVE_ORDER
        multisensory_test_n = int(multisensory_test_n)
    if h2_calibration_pass is not None and not isinstance(
        h2_calibration_pass, bool
    ):
        return FAIL_CLOSED_ACTIVE_ORDER
    if multisensory_test_n is None or h2_calibration_pass is None:
        return FAIL_CLOSED_ACTIVE_ORDER
    if multisensory_test_n >= H2_MIN_TEST_N and h2_calibration_pass:
        return FAMILY_ORDER
    return FAIL_CLOSED_ACTIVE_ORDER


# ---------------------------------------------------------------------------
# Legacy constants (retained for backward compatibility only; DO NOT USE)
# ---------------------------------------------------------------------------

B_PRIMARY: int = 100_000
"""Deprecated legacy constant.

Confirmatory p-values are exact discrete probabilities.  Monte Carlo draws
are not used for confirmatory p-values and this constant must never be
passed to Holm family procedures.
"""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    """Outcome of one confirmatory slot.

    Attributes:
        name: Slot identifier, one of :data:`FAMILY_ORDER`.
        statistic: Observed statistic, or ``None`` when unavailable.
            H1/H3: median paired difference.  H2: ``max_m`` median
            ``d_{i,m}``.
        boundary: Null boundary ``delta`` in the statistic's units.
        p_value: Exact one-sided lower-tail p-value.  ``1.0`` when
            unavailable.
        n_animal: Number of animal-level observations used.
        n_boot: Bootstrap draws used for effect-size CIs.
        seed: Random seed used for the effect-size bootstrap.
        available: Whether the slot could be evaluated.
        reason: Empty string when available; fail-closed reason otherwise.
        effect: JSON-serialisable effect-size and diagnostic fields.
        ci_low: Lower 95% percentile bootstrap bound for the statistic.
        ci_high: Upper 95% percentile bootstrap bound for the statistic.
    """

    name: str
    statistic: float | None
    boundary: float
    p_value: float
    n_animal: int
    n_boot: int
    seed: int
    available: bool
    reason: str
    effect: Mapping[str, object] = field(default_factory=dict)
    ci_low: float | None = None
    ci_high: float | None = None

    def __post_init__(self) -> None:
        allowed = set(FAMILY_ORDER) | {"animal_cluster_bootstrap_shift"}
        if self.name not in allowed:
            raise ValueError(f"unknown confirmatory slot: {self.name!r}")
        if not (math.isfinite(self.p_value) and 0.0 <= self.p_value <= 1.0):
            raise ValueError(f"invalid p-value: {self.p_value!r}")
        if self.n_animal < 0:
            raise ValueError("n_animal must be non-negative")
        if self.n_boot < 1:
            raise ValueError("n_boot must be at least 1")
        if self.available and self.statistic is None:
            raise ValueError("available results must carry a statistic")
        if self.available and not math.isfinite(float(self.statistic)):
            raise ValueError("available results must carry a finite statistic")
        if not self.available and self.p_value != 1.0:
            raise ValueError("unavailable results must carry p_value=1.0")
        for bound in (self.ci_low, self.ci_high):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("CI bounds must be finite or None")


@dataclass(frozen=True)
class HolmResult:
    """Holm step-down result for the fixed four-slot family.

    Attributes:
        order: Frozen slot order.
        m: Family size, always :data:`FAMILY_M`.
        alpha: Family-wise alpha, always :data:`ALPHA`.
        raw_p: Input p-values in frozen order.
        adjusted_p: Holm-adjusted p-values in frozen order.
        significant: Step-down rejection flags after the MC guard.
        indeterminate_mc: Slots whose adjusted p is within two adjusted
            Monte Carlo standard errors of ``alpha``; these are not
            rejected.
        mc_se: Adjusted-p Monte Carlo standard errors in frozen order.
    """

    order: tuple[str, ...]
    m: int
    alpha: float
    raw_p: Mapping[str, float]
    adjusted_p: Mapping[str, float]
    significant: Mapping[str, bool]
    indeterminate_mc: Mapping[str, bool]
    mc_se: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisSet:
    """H2 analysis set A selected at unblinding.

    Attributes:
        mask: Boolean ``(n,)`` mask of animals in A.
        n_animal: Number of animals in A.
        excluded: Per-reason exclusion counts.  The categories form a
            hierarchy, so their sum equals ``n - n_animal``.
        available: Whether ``n_animal >= N_MIN``.
        reason: Empty when available; ``out_of_scope`` otherwise.
    """

    mask: np.ndarray
    n_animal: int
    excluded: Mapping[str, int]
    available: bool
    reason: str

    def __post_init__(self) -> None:
        if self.mask.ndim != 1 or self.mask.dtype != np.bool_:
            raise ValueError("mask must be a 1-D boolean array")
        if int(self.mask.sum()) != self.n_animal:
            raise ValueError("n_animal must equal mask.sum()")
        if self.available and self.n_animal < N_MIN:
            raise ValueError("available analysis sets need at least N_MIN animals")
        if sum(int(value) for value in self.excluded.values()) != (
            self.mask.size - self.n_animal
        ):
            raise ValueError("excluded counts must sum to n - n_animal")


@dataclass(frozen=True)
class SlotFragility:
    """Descriptive leave-one-animal-out fragility for one confirmatory slot.

    This is a **descriptive** robustness diagnostic (user decision
    2026-10-19 #1).  It is not a gate: it never changes the pass/fail
    verdict, requires no sign-off, and carries no p-value claim.

    Attributes:
        slot: Confirmatory slot name.
        n_animal: Animals in the slot's baseline analysis.
        baseline_significant: Frozen Holm verdict of the slot under the
            observed p-values.
        available: Whether the index could be computed.
        loo_flip_rate: Fraction of single-animal deletions whose Holm
            verdict for this slot differs from the baseline; ``None``
            when unavailable.
        n_single_deletion_flips: Count behind ``loo_flip_rate``.
        min_animals_to_flip: Smallest number of simultaneously removed
            animals that flips the slot verdict, searched exhaustively
            up to ``max_remove``; ``None`` when no flipping subset was
            found or the index is unavailable.
        censored: True when the minimal flipping set was not found
            within ``max_remove``, or the index is unavailable.
        max_remove: Search cap for the minimal flipping set.
        reason: Empty when available; failure reason otherwise.
        construction: Fixed label ``"descriptive_loo_fragility"``.
    """

    slot: str
    n_animal: int
    baseline_significant: bool
    available: bool
    loo_flip_rate: float | None
    n_single_deletion_flips: int
    min_animals_to_flip: int | None
    censored: bool
    max_remove: int
    reason: str = ""
    construction: str = "descriptive_loo_fragility"

    def __post_init__(self) -> None:
        if self.n_animal < 0:
            raise ValueError("n_animal must be non-negative")
        if self.max_remove < 1:
            raise ValueError("max_remove must be at least 1")
        if self.available:
            if self.loo_flip_rate is None:
                raise ValueError("available fragility must carry a rate")
            if not 0.0 <= float(self.loo_flip_rate) <= 1.0:
                raise ValueError("loo_flip_rate must lie in [0, 1]")
            if self.n_single_deletion_flips < 0:
                raise ValueError("flip count must be non-negative")
            if (
                self.min_animals_to_flip is not None
                and not 1 <= self.min_animals_to_flip <= self.max_remove
            ):
                raise ValueError(
                    "min_animals_to_flip must lie in [1, max_remove]"
                )
        elif self.loo_flip_rate is not None:
            raise ValueError("unavailable fragility must not carry a rate")


@dataclass(frozen=True)
class G1Result:
    """Animal-level permutation test for the G1 existence criterion."""

    rho_observed: float
    p_value: float
    n_animal: int
    n_perm: int
    seed: int
    exists: bool
    available: bool = True
    reason: str = ""


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _as_1d(values: object, name: str) -> np.ndarray:
    """Validate a finite 1-D float array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _as_1d_nan(values: object, name: str) -> np.ndarray:
    """Validate a 1-D float array that may contain NaN but not inf."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} contains infinite values")
    return np.ascontiguousarray(array)


def _analysis_mask(analysis_set: AnalysisSet, n_total: int) -> np.ndarray:
    """Return the validated analysis-set mask for ``n_total`` animals."""
    mask = np.asarray(getattr(analysis_set, "mask", None))
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("analysis_set.mask must be a 1-D boolean array")
    if mask.size != n_total:
        raise ValueError(
            f"analysis_set.mask has length {mask.size}, expected {n_total}"
        )
    return np.ascontiguousarray(mask)


def _check_animal_key(
    animal_key: Sequence[object] | None,
    n_expected: int,
    name: str,
) -> None:
    """Validate the mandatory animal-level key array.

    The inferential unit is one Cercus ``animal_key``.  A missing key
    lets trial- or condition-level arrays enter ``n`` silently; a
    duplicated key weights one animal twice in every animal-level
    statistic.  Both are rejected before any computation, so the slot
    entries enforce the animal-level unit in code rather than by caller
    discipline.
    """
    if animal_key is None:
        raise ValueError(
            f"{name} is required: the inferential unit is one Cercus "
            "animal_key, and the slot rejects trial/condition-level arrays"
        )
    keys = np.asarray(animal_key, dtype=object)
    if keys.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {keys.shape}")
    if keys.shape[0] != n_expected:
        raise ValueError(
            f"{name} has length {keys.shape[0]}, expected {n_expected}"
        )
    if len(set(keys.tolist())) != keys.shape[0]:
        raise ValueError(f"{name} contains duplicate animal keys")


def _check_n_boot(n_boot: int) -> int:
    """Validate the bootstrap draw count."""
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)):
        raise ValueError(f"n_boot must be an integer, got {type(n_boot)}")
    if int(n_boot) < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    return int(n_boot)


def _check_seed(seed: int) -> int:
    """Validate a non-negative integer seed."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer, got {type(seed)}")
    if int(seed) < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return int(seed)


def _check_finite_scalar(value: float, name: str) -> float:
    """Validate a finite scalar."""
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return scalar


def _check_persistence_baseline(persistence_baseline: str) -> str:
    """Validate the declared persistence baseline.

    The h5 scope must use ``v(t) -> v(t + 5*dt)``; the onset scope uses
    the current target ``v(t)``.  ``v(t-1)`` is never a valid baseline.
    A string declaration is the only thing the slot can verify, so the
    value is mandatory and anything other than :data:`PERSISTENCE_BASELINES`
    fails closed.
    """
    if persistence_baseline not in PERSISTENCE_BASELINES:
        raise ValueError(
            "persistence_baseline must be one of "
            f"{PERSISTENCE_BASELINES} (the h5 baseline is v(t), never "
            f"v(t-1)), got {persistence_baseline!r}"
        )
    return persistence_baseline


def _check_split_facts(
    split_sha256: str,
    multisensory_test_n: int | None,
) -> None:
    """Bind reported split facts to the frozen PG-02 split.

    The frozen split SHA fixes the realised multisensory test n (14).
    A caller passing 15 with the frozen SHA would promote H2 into the
    confirmatory family against the pre-registration, so the mismatch
    raises instead of resolving.  Strings prefixed with
    ``synthetic:`` are simulation fixtures and are accepted without a
    frozen-split claim; any other unknown SHA fails closed.
    """
    if not isinstance(split_sha256, str) or not split_sha256:
        raise ValueError("split_sha256 must be a non-empty string")
    if split_sha256 == FROZEN_SPLIT_SHA256:
        if multisensory_test_n != FROZEN_MULTISENSORY_TEST_N:
            raise ValueError(
                "frozen split requires multisensory_test_n="
                f"{FROZEN_MULTISENSORY_TEST_N}, got {multisensory_test_n!r}"
            )
        return
    if split_sha256.startswith(SYNTHETIC_SPLIT_SHA_PREFIX):
        return
    raise ValueError(
        "unknown split: split_sha256 is neither the frozen PG-02 SHA "
        f"{FROZEN_SPLIT_SHA256} nor a synthetic: fixture"
    )


def _cohens_dz_1d(values: np.ndarray) -> float:
    """Paired Cohen's d_z; zero when the sample SD is zero."""
    if values.size < 2:
        return 0.0
    sd = float(values.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(values.mean() / sd)


def _percentile_ci(
    boot: np.ndarray, ci_level: float = CI_LEVEL
) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval; ``(None, None)`` if non-finite."""
    alpha = 1.0 - ci_level
    if not np.all(np.isfinite(boot)):
        return None, None
    low, high = np.percentile(
        boot, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)]
    )
    return float(low), float(high)


def _bootstrap_indices(
    n_animal: int, n_boot: int, seed: int
) -> np.ndarray:
    """Animal-cluster bootstrap indices of shape ``(n_boot, n_animal)``."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_animal, size=(n_boot, n_animal))
    assert idx.shape == (n_boot, n_animal), (
        f"expected {(n_boot, n_animal)}, got {idx.shape}"
    )
    return idx


def _exact_sign_test(
    differences: np.ndarray,
    boundary: float = 0.0,
) -> tuple[float, dict[str, object]]:
    """Exact one-sided binomial sign test.

    The null is ``P(d_i < boundary) <= 0.5``; the alternative is
    ``P(d_i < boundary) > 0.5``.  Ties at the boundary are counted as
    non-improvements.  The p-value is ``P(X >= k)`` with
    ``X ~ Binomial(n, 0.5)`` and ``k`` the number of strict
    improvements.

    Args:
        differences: Finite ``(n,)`` per-animal paired differences.
        boundary: Null boundary ``delta``.

    Returns:
        ``(p_value, diagnostics)``.  ``diagnostics`` contains the sign
        counts, the exact Clopper-Pearson interval for the proportion
        below the boundary, and the median difference.

    Raises:
        ValueError: On empty, non-finite, or non-1-D input.
    """
    array = _as_1d(differences, "differences")
    boundary = _check_finite_scalar(boundary, "boundary")
    shifted = array - boundary
    n = int(shifted.size)
    below = int(np.count_nonzero(shifted < 0.0))
    ties = int(np.count_nonzero(shifted == 0.0))
    if below == 0:
        p_value = 1.0
    else:
        p_value = float(
            stats.binomtest(
                below, n, 0.5, alternative="greater"
            ).pvalue
        )
    interval = stats.binomtest(below, n, 0.5).proportion_ci(
        confidence_level=CI_LEVEL, method="exact"
    )
    diagnostics: dict[str, object] = {
        "n_animal": n,
        "n_below_boundary": below,
        "n_tied_at_boundary": ties,
        "proportion_below_boundary": below / float(n),
        "proportion_ci_low": float(interval.low),
        "proportion_ci_high": float(interval.high),
        "median_difference": float(np.median(array)),
        "boundary": boundary,
        "tie_policy": TIE_POLICY,
        "construction": "exact_one_sided_binomial_sign_test",
        "null": "P(difference < boundary) <= 0.5",
        "alternative": "P(difference < boundary) > 0.5",
    }
    assert 0.0 <= p_value <= 1.0
    return p_value, diagnostics


def _unavailable(
    name: str,
    boundary: float,
    n_animal: int,
    n_boot: int,
    seed: int,
    reason: str,
) -> TestResult:
    """Build a fail-closed result with ``p=1``."""
    return TestResult(
        name=name,
        statistic=None,
        boundary=boundary,
        p_value=1.0,
        n_animal=n_animal,
        n_boot=n_boot,
        seed=seed,
        available=False,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Generic bootstrap-shift primitive (sensitivity utility, not confirmatory)
# ---------------------------------------------------------------------------


def animal_cluster_bootstrap_shift(
    values: np.ndarray,
    boundary: float,
    n_boot: int,
    seed: int,
) -> TestResult:
    """Animal-cluster bootstrap-shift test (sensitivity utility only).

    This primitive is retained for descriptive sensitivity checks.  It is
    **not** used by any of the four confirmatory slots: the calibrated
    primary construction is the exact sign test.  For a single candidate
    it implements ``p = (1 + #{T*_b <= 2*T_obs - boundary}) /
    (n_boot + 1)``; for ``k > 1`` candidates the null shift is applied
    per candidate before the minimum.

    Args:
        values: ``(n,)`` or ``(n, k)`` finite per-animal statistics.
        boundary: Null boundary ``delta``.
        n_boot: Number of animal-cluster bootstrap draws.
        seed: Random seed.

    Returns:
        :class:`TestResult` with
        ``name="animal_cluster_bootstrap_shift"``.  A zero-variance
        candidate returns ``available=False`` with ``p_value=1.0`` and
        ``reason="degenerate_zero_variance"`` (fail-closed; the
        bootstrap-shift p is not valid for a degenerate sample).

    Raises:
        ValueError: If inputs are empty, non-finite, wrongly shaped, or
            ``n_boot < 1``.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(f"values must be 1-D or 2-D, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values contains non-finite values")
    boundary = _check_finite_scalar(boundary, "boundary")
    n_boot = _check_n_boot(n_boot)
    seed = _check_seed(seed)
    n, k = matrix.shape
    if n < 2:
        raise ValueError(
            "animal-cluster bootstrap-shift needs at least two animals"
        )
    if np.any(matrix.std(axis=0, ddof=1) == 0.0):
        return TestResult(
            name="animal_cluster_bootstrap_shift",
            statistic=None,
            boundary=boundary,
            p_value=1.0,
            n_animal=n,
            n_boot=n_boot,
            seed=seed,
            available=False,
            reason="degenerate_zero_variance",
        )
    t_arm = matrix.mean(axis=0)
    t_obs = float(t_arm.min())
    idx = _bootstrap_indices(n, n_boot, seed)
    boot_arm = matrix[idx].mean(axis=1)
    assert boot_arm.shape == (n_boot, k)
    boot_null = (boot_arm - t_arm + boundary).min(axis=1)
    assert boot_null.shape == (n_boot,)
    p_value = (1.0 + float(np.count_nonzero(boot_null <= t_obs))) / (
        n_boot + 1.0
    )
    boot_raw = boot_arm.min(axis=1)
    ci_low, ci_high = _percentile_ci(boot_raw)
    return TestResult(
        name="animal_cluster_bootstrap_shift",
        statistic=t_obs,
        boundary=boundary,
        p_value=p_value,
        n_animal=n,
        n_boot=n_boot,
        seed=seed,
        available=True,
        reason="",
        effect={"n_candidate": int(k)},
        ci_low=ci_low,
        ci_high=ci_high,
    )


# ---------------------------------------------------------------------------
# H1: foundation persistence gate
# ---------------------------------------------------------------------------


def h1_foundation_persistence(
    mse_by_arm: Mapping[str, np.ndarray],
    mse_persistence: np.ndarray,
    scope: str,
    *,
    animal_key: Sequence[object],
    persistence_baseline: str,
    n_boot: int = CI_BOOT,
    seed: int = SEED_BASE,
    compute_ci: bool = True,
) -> TestResult:
    """H1 persistence gate for the onset or h5 scoring scope.

    Evaluates the pre-registered 4-arm Bonferroni union across all
    foundation arms :data:`H1_FOUNDATION_ARMS` (``R0``, ``R1``, ``R2``,
    ``R3``): ``p = min(1.0, 4.0 * min_a p_a)``.  There is no single-arm
    path: the r9 review showed that a caller-chosen arm with
    ``multiplier=1`` could turn a p of 0.00636 into a "confirmatory"
    result while the pre-registered union p was 0.02545, so the degree
    of freedom was removed entirely rather than validated downstream.

    For each candidate arm ``a``, the null is
    ``median(MSE_a - MSE_pers) >= 0`` (at most half of the animals
    improve); the alternative is ``median < 0``.  The p-value is the
    exact one-sided binomial sign test.  Onset and h5 scopes may be
    satisfied by different winning arms (consistent with Phase-2
    design).

    Constructive caveat: ``R1`` is residual-over-persistence
    (``v(t-1) + f``).  If ``R1`` is the winning arm, its advantage
    reflects the skill of the residual head over the trivial copy, and
    is not claimed as evidence that the recurrent core autonomously
    beats persistence.

    Args:
        mse_by_arm: Mapping containing exactly the four foundation arms
            in :data:`H1_FOUNDATION_ARMS`.
        mse_persistence: Finite non-negative ``(n,)`` per-animal
            persistence MSE on the same scope and target transform.
        scope: ``"onset"`` or ``"h5"``.  The h5 caller must use
            persistence ``v(t) -> v(t + 5*dt)``, never ``v(t-1)``.
        animal_key: Mandatory ``(n,)`` Cercus animal keys.  Keys must
            be unique and length matched; this enforces the animal-level
            inferential unit without changing the statistic.
        persistence_baseline: Mandatory declaration of the persistence
            target; only ``"v_t"`` is accepted.  The h5 caller must use
            ``v(t) -> v(t + 5*dt)``, never ``v(t-1)``.
        n_boot: Bootstrap draws for the winner-conditional median CI.
        seed: Random seed for the effect CI.
        compute_ci: When ``False``, skip the bootstrap CI (calibration
            and fragility recomputation only; the p-value is
            unchanged).

    Returns:
        :class:`TestResult` named ``H1_onset`` or ``H1_h5``.  ``n < 10``
        returns ``available=False`` with ``p=1``.

    Raises:
        ValueError: On unknown scope, missing or extra arms, an arm with
            the wrong shape, negative or non-finite MSE, or non-finite
            persistence MSE.  A structural mismatch is never converted
            into a silently unavailable arm: doing so would change the
            winning arm and the winner-conditional CI without raising.
    """
    if scope not in H1_SCOPES:
        raise ValueError(f"scope must be one of {H1_SCOPES}, got {scope!r}")
    persistence_baseline = _check_persistence_baseline(persistence_baseline)
    if set(mse_by_arm) != set(H1_FOUNDATION_ARMS):
        raise ValueError(
            "mse_by_arm must contain exactly "
            f"{H1_FOUNDATION_ARMS}, got {tuple(sorted(mse_by_arm))}"
        )
    persistence = _as_1d(mse_persistence, "mse_persistence")
    if np.any(persistence < 0.0):
        raise ValueError("mse_persistence must be non-negative")
    n = int(persistence.size)
    _check_animal_key(animal_key, n, "animal_key")
    name = f"H1_{scope}"
    boundary = SLOT_BOUNDARIES[name]
    n_boot = _check_n_boot(n_boot)
    seed = _check_seed(seed)
    if n < N_MIN:
        return _unavailable(
            name, boundary, n, n_boot, seed, "out_of_scope (n_lt_10)"
        )

    arm_arrays: dict[str, np.ndarray] = {}
    arm_p: dict[str, float] = {}
    arm_median: dict[str, float] = {}
    arm_below: dict[str, int] = {}
    for candidate in H1_FOUNDATION_ARMS:
        arr = _as_1d(mse_by_arm[candidate], f"mse_by_arm[{candidate!r}]")
        if arr.shape != persistence.shape:
            raise ValueError(
                f"mse_by_arm[{candidate!r}] has shape {arr.shape}, "
                f"expected {persistence.shape}"
            )
        if np.any(arr < 0.0):
            raise ValueError(f"mse_by_arm[{candidate!r}] must be non-negative")
        arm_arrays[candidate] = arr
        diff = arr - persistence
        p_val, _ = _exact_sign_test(diff, boundary)
        arm_p[candidate] = p_val
        arm_median[candidate] = float(np.median(diff))
        arm_below[candidate] = int(np.count_nonzero(diff < boundary))

    p_value = min(1.0, 4.0 * min(arm_p[c] for c in H1_FOUNDATION_ARMS))
    target_arm = min(
        H1_FOUNDATION_ARMS,
        key=lambda c: (
            arm_p[c],
            arm_median[c],
            H1_FOUNDATION_ARMS.index(c),
        ),
    )
    assert target_arm in arm_arrays

    differences = arm_arrays[target_arm] - persistence
    assert differences.shape == (n,)
    d_z = _cohens_dz_1d(differences)
    mean_mse_arm = float(np.mean(arm_arrays[target_arm]))
    mean_mse_persistence = float(np.mean(persistence))
    skill = (
        1.0 - mean_mse_arm / mean_mse_persistence
        if mean_mse_persistence > 0.0
        else None
    )

    ci_low: float | None = None
    ci_high: float | None = None
    if compute_ci:
        idx = _bootstrap_indices(n, n_boot, seed)
        boot = np.median(differences[idx], axis=1)
        assert boot.shape == (n_boot,)
        ci_low, ci_high = _percentile_ci(boot)

    caveat = ""
    if target_arm == "R1":
        caveat = (
            "R1 is residual-over-persistence; its advantage reflects the skill "
            "of the residual head over the trivial copy, not evidence that the "
            "recurrent core autonomously beats persistence"
        )

    effect: dict[str, object] = {
        "confirmatory_arm": target_arm,
        "winning_arm": target_arm,
        "arm_p_values": arm_p,
        "arm_available": {c: True for c in H1_FOUNDATION_ARMS},
        "arm_median_differences": arm_median,
        "arm_n_below_boundary": arm_below,
        "bonferroni_multiplier": 4,
        "descriptive_best_arm": target_arm,
        # The reported statistic and its CI belong to the arm selected
        # by the smallest p-value (the frozen winner); they carry a
        # winner's-curse selection effect that the report must state.
        "winner_conditional": True,
        "winner_selection_rule": "min_arm_p_then_median_then_frozen_order",
        "ci_is_winner_conditional": True,
        "d_z": d_z,
        "skill": skill,
        "mean_mse_winning_arm": mean_mse_arm,
        "mean_mse_persistence": mean_mse_persistence,
        "median_difference": float(np.median(differences)),
        "caveat": caveat,
        "construction": "exact_sign_test_4_arm_bonferroni_union",
        "scientific_estimand": (
            "at least one foundation arm beats persistence "
            "(median paired MSE difference < 0, Bonferroni union)"
        ),
        "selection_rule": "bonferroni_union_over_R0_R1_R2_R3",
        "n_animal": n,
        "n_below_boundary": arm_below[target_arm],
        "proportion_below_boundary": arm_below[target_arm] / float(n),
        "boundary": boundary,
        "tie_policy": TIE_POLICY,
        "persistence_baseline": persistence_baseline,
    }
    return TestResult(
        name=name,
        statistic=float(np.median(differences)),
        boundary=boundary,
        p_value=p_value,
        n_animal=n,
        n_boot=n_boot,
        seed=seed,
        available=True,
        reason="",
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def h1_seed_sign_consistency(
    mse_by_arm_by_seed: Mapping[str, np.ndarray],
    mse_persistence_by_seed: np.ndarray,
    scope: str,
    *,
    animal_key: Sequence[object],
    persistence_baseline: str,
) -> dict[str, object]:
    """Apply the seed-sign filter to the frozen H1 winning arm.

    The H1 four-arm union has no single pre-specified arm, so the arm
    used for the ``unstable_across_seeds`` filter must itself be fixed
    before the filter is evaluated.  This function removes that degree
    of freedom:

    1. average each arm and persistence over the training seeds;
    2. resolve the confirmatory winner with the *same* frozen rule as
       :func:`h1_foundation_persistence` (smallest arm p, then median
       difference, then frozen arm order);
    3. for that one arm, compute the per-seed statistic
       ``median_i (MSE_{i,a}^{(r)} - MSE_{i,pers}^{(r)})``;
    4. call :func:`seed_sign_consistency` with ``reference=0.0``.

    The filter is a pre-registered claim restriction, not a p-value
    change.  It never re-selects the arm and never changes the slot
    p-value.  All four arms are required; a missing or malformed arm
    raises rather than silently changing the winner.

    Args:
        mse_by_arm_by_seed: Mapping containing exactly the four arms in
            :data:`H1_FOUNDATION_ARMS`, each a finite non-negative
            ``(n_seed, n_animal)`` MSE array.
        mse_persistence_by_seed: Finite non-negative
            ``(n_seed, n_animal)`` persistence MSE on the same scope.
        scope: ``"onset"`` or ``"h5"``.
        animal_key: Mandatory ``(n_animal,)`` Cercus animal keys.  Keys
            must be unique and length matched.
        persistence_baseline: Mandatory persistence-target declaration;
            only ``"v_t"`` is accepted (never ``v(t-1)``).

    Returns:
        Dictionary with ``available``, ``consistent``, ``winning_arm``,
        ``per_seed_statistics``, ``n_seed``, ``n_animal``, ``scope``,
        ``reference``, ``selection_rule``, ``construction``, and
        ``reason``.  A slot that cannot be evaluated returns
        ``available=False`` and ``consistent=False``; this helper has
        no p-value of its own.

    Raises:
        ValueError: If ``scope`` is unknown, arms are missing or extra,
            or any array is not a finite non-negative 2-D array with at
            least two seeds.
    """
    if scope not in H1_SCOPES:
        raise ValueError(f"scope must be one of {H1_SCOPES}, got {scope!r}")
    persistence_baseline = _check_persistence_baseline(persistence_baseline)
    if set(mse_by_arm_by_seed) != set(H1_FOUNDATION_ARMS):
        raise ValueError(
            "mse_by_arm_by_seed must contain exactly "
            f"{H1_FOUNDATION_ARMS}, got {tuple(sorted(mse_by_arm_by_seed))}"
        )
    persistence = np.asarray(mse_persistence_by_seed, dtype=np.float64)
    if persistence.ndim != 2:
        raise ValueError(
            "mse_persistence_by_seed must be 2-D (n_seed, n_animal), got "
            f"{persistence.shape}"
        )
    if persistence.shape[0] < 2:
        raise ValueError("seed-sign consistency needs at least two seeds")
    if persistence.size == 0:
        raise ValueError("mse_persistence_by_seed must not be empty")
    if not np.all(np.isfinite(persistence)) or np.any(persistence < 0.0):
        raise ValueError(
            "mse_persistence_by_seed must be finite and non-negative"
        )
    n_seed, n_animal = persistence.shape
    assert persistence.shape == (n_seed, n_animal)
    _check_animal_key(animal_key, n_animal, "animal_key")

    arm_arrays: dict[str, np.ndarray] = {}
    for candidate in H1_FOUNDATION_ARMS:
        array = np.asarray(mse_by_arm_by_seed[candidate], dtype=np.float64)
        if array.shape != persistence.shape:
            raise ValueError(
                f"mse_by_arm_by_seed[{candidate!r}] has shape "
                f"{array.shape}, expected {persistence.shape}"
            )
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(
                f"mse_by_arm_by_seed[{candidate!r}] must be finite and "
                "non-negative"
            )
        arm_arrays[candidate] = array

    seed_mean_by_arm = {
        candidate: array.mean(axis=0)
        for candidate, array in arm_arrays.items()
    }
    seed_mean_persistence = persistence.mean(axis=0)
    confirmatory = h1_foundation_persistence(
        seed_mean_by_arm,
        seed_mean_persistence,
        scope,
        animal_key=animal_key,
        persistence_baseline=persistence_baseline,
        n_boot=1,
        compute_ci=False,
    )
    if not confirmatory.available:
        return {
            "available": False,
            "consistent": False,
            "winning_arm": None,
            "per_seed_statistics": None,
            "n_seed": int(n_seed),
            "n_animal": int(n_animal),
            "scope": scope,
            "reference": 0.0,
            "persistence_baseline": persistence_baseline,
            "selection_rule": "h1_slot_unavailable",
            "construction": (
                "h1_union_seed_consistency_after_frozen_arm_selection"
            ),
            "reason": confirmatory.reason,
        }
    winning_arm = str(confirmatory.effect["confirmatory_arm"])
    assert winning_arm in arm_arrays

    differences = arm_arrays[winning_arm] - persistence
    assert differences.shape == (n_seed, n_animal)
    per_seed = np.median(differences, axis=1)
    assert per_seed.shape == (n_seed,)
    consistent = seed_sign_consistency(per_seed, reference=0.0)
    return {
        "available": True,
        "consistent": bool(consistent),
        "winning_arm": winning_arm,
        "per_seed_statistics": [float(value) for value in per_seed],
        "n_seed": int(n_seed),
        "n_animal": int(n_animal),
        "scope": scope,
        "reference": 0.0,
        "persistence_baseline": persistence_baseline,
        "selection_rule": "frozen_union_winner_min_p_then_median_then_order",
        "construction": (
            "h1_union_seed_consistency_after_frozen_arm_selection"
        ),
        "reason": "",
    }


# ---------------------------------------------------------------------------
# H2: non-inferiority against every pre-specified classical model
# ---------------------------------------------------------------------------


def build_h2_analysis_set(
    l_obs: np.ndarray,
    l_nsmor: np.ndarray,
    abs_err_classical_by_model: Mapping[str, np.ndarray],
    *,
    all_seed_mask: np.ndarray,
    effective_draws_by_model: Mapping[str, np.ndarray],
    min_effective_draws: int = 40,
    n_min: int = N_MIN,
) -> AnalysisSet:
    """Select H2 analysis set A at unblinding.

    An animal enters A iff the all-seed mask is true, ``L_obs`` and
    ``L_NS`` are defined, all four classical absolute errors are
    defined, and every classical ensemble has at least
    ``min_effective_draws`` effective draws.  The mask and the effective
    draw counts are **required** keyword arguments: there is no default
    that silently skips the pre-registered exclusions.

    The exclusion counts form a hierarchy (seed, observed latency,
    NSMoR latency, classical error, effective draws); each excluded
    animal is counted once, so their sum equals ``n - n_animal``.
    Note on hierarchy: ``all_seed_mask`` (derived from ``h2_seed_errors``)
    already excludes any animal where any seed's closed-loop latency is
    undefined.  Therefore, in normal pipeline execution,
    ``l_nsmor_undefined`` is a redundant defensive check that evaluates
    to 0, avoiding audit ambiguity.

    Args:
        l_obs: ``(n,)`` observed defective-KM median latency, NaN if
            ``F(C) < 0.5``.
        l_nsmor: ``(n,)`` seed-mean C0 closed-loop latency; NaN if any
            seed is undefined or the closed loop is unstable.
        abs_err_classical_by_model: Mapping with exactly
            :data:`H2_MODELS` keys, each ``(n,)`` conservative absolute
            error, NaN if the ensemble is undefined.
        all_seed_mask: ``(n,)`` boolean mask from
            :func:`h2_seed_errors`; true iff every training seed has a
            defined closed-loop latency.
        effective_draws_by_model: Mapping with the same keys, each
            ``(n,)`` non-negative integer-valued effective ensemble
            draw count.
        min_effective_draws: Minimum effective draws per model.
        n_min: Minimum analysis-set size; must be at least
            :data:`N_MIN`.

    Returns:
        :class:`AnalysisSet` with the boolean mask and exclusion counts.

    Raises:
        ValueError: On missing/extra models, shape mismatch,
            infinities, non-boolean masks, non-integer draw counts, or
            an inconsistent all-seed mask.
    """
    if n_min < N_MIN:
        raise ValueError(f"n_min must be at least {N_MIN}, got {n_min}")
    if set(abs_err_classical_by_model) != set(H2_MODELS):
        raise ValueError(
            "abs_err_classical_by_model must contain exactly "
            f"{H2_MODELS}, got {tuple(abs_err_classical_by_model)}"
        )
    if set(effective_draws_by_model) != set(H2_MODELS):
        raise ValueError(
            "effective_draws_by_model must contain exactly "
            f"{H2_MODELS}, got {tuple(effective_draws_by_model)}"
        )
    obs = _as_1d_nan(l_obs, "l_obs")
    nsmor = _as_1d_nan(l_nsmor, "l_nsmor")
    if obs.shape != nsmor.shape:
        raise ValueError(
            f"l_obs shape {obs.shape} != l_nsmor shape {nsmor.shape}"
        )
    n = int(obs.size)
    seed_mask = np.asarray(all_seed_mask)
    if seed_mask.dtype != np.bool_ or seed_mask.shape != (n,):
        raise ValueError(
            "all_seed_mask must be a 1-D boolean array of shape "
            f"{(n,)}, got {seed_mask.dtype} {seed_mask.shape}"
        )
    seed_mask = np.ascontiguousarray(seed_mask)
    if np.any(seed_mask & ~np.isfinite(nsmor)):
        raise ValueError(
            "all_seed_mask is true for an animal with undefined l_nsmor; "
            "fail-closed"
        )
    classical: list[np.ndarray] = []
    draws: list[np.ndarray] = []
    for model in H2_MODELS:
        array = _as_1d_nan(
            abs_err_classical_by_model[model],
            f"abs_err_classical_by_model[{model!r}]",
        )
        if array.shape != obs.shape:
            raise ValueError(
                f"model {model!r} has shape {array.shape}, expected "
                f"{obs.shape}"
            )
        finite = np.isfinite(array)
        if np.any(array[finite] < 0.0):
            raise ValueError(
                f"model {model!r} absolute errors must be non-negative"
            )
        classical.append(array)
        draw_array = np.asarray(
            effective_draws_by_model[model], dtype=np.float64
        )
        if draw_array.shape != obs.shape:
            raise ValueError(
                f"effective draws for {model!r} have shape "
                f"{draw_array.shape}, expected {obs.shape}"
            )
        if not np.all(np.isfinite(draw_array)):
            raise ValueError("effective draws must be finite")
        if np.any(draw_array < 0.0):
            raise ValueError("effective draws must be non-negative")
        if np.any(draw_array != np.round(draw_array)):
            raise ValueError("effective draws must be integer-valued")
        draws.append(draw_array)
    cls_matrix = np.column_stack(classical)
    assert cls_matrix.shape == (n, len(H2_MODELS))
    draws_matrix = np.column_stack(draws)
    assert draws_matrix.shape == (n, len(H2_MODELS))

    obs_ok = np.isfinite(obs)
    ns_ok = np.isfinite(nsmor)
    cls_ok = np.all(np.isfinite(cls_matrix), axis=1)
    draws_ok = np.all(
        draws_matrix >= float(min_effective_draws), axis=1
    )
    mask = seed_mask & obs_ok & ns_ok & cls_ok & draws_ok
    assert mask.shape == (n,)
    excluded = {
        "seed_incomplete": int(np.count_nonzero(~seed_mask)),
        "l_obs_undefined": int(
            np.count_nonzero(seed_mask & ~obs_ok)
        ),
        "l_nsmor_undefined": int(
            np.count_nonzero(seed_mask & obs_ok & ~ns_ok)
        ),
        "classical_undefined": int(
            np.count_nonzero(seed_mask & obs_ok & ns_ok & ~cls_ok)
        ),
        "effective_draws_lt_40": int(
            np.count_nonzero(
                seed_mask & obs_ok & ns_ok & cls_ok & ~draws_ok
            )
        ),
    }
    n_animal = int(np.count_nonzero(mask))
    available = n_animal >= int(n_min)
    return AnalysisSet(
        mask=mask,
        n_animal=n_animal,
        excluded=excluded,
        available=available,
        reason="" if available else "out_of_scope (n_lt_10)",
    )


def h2_worst_case_p(
    abs_err_nsmor: np.ndarray,
    abs_err_classical_by_model: Mapping[str, np.ndarray],
    *,
    analysis_set: AnalysisSet,
) -> tuple[float, Mapping[str, float], int]:
    """Worst-case H2 intersection-union p under informative missingness.

    A non-inferiority test is anti-conservative when the animals that
    drop out are systematically the hard ones.  The r9 review showed
    that a finite error on an excluded animal was still counted as an
    improvement, so an animal excluded for ``effective_draws_lt_40``
    could make the worst-case p smaller than the analysis-set p.  The
    frozen rule is therefore mask-based: an animal **outside**
    ``analysis_set.mask`` is a non-improvement regardless of whether its
    error is finite or NaN.  It stays in the denominator and never
    contributes to ``k``.

    Args:
        abs_err_nsmor: ``(n,)`` C0 absolute errors on the full sample.
            Values inside the analysis set must be finite; values
            outside may be finite or NaN.
        abs_err_classical_by_model: Mapping with exactly
            :data:`H2_MODELS` keys, each ``(n,)`` on the same sample.
        analysis_set: Frozen analysis set A from
            :func:`build_h2_analysis_set`; its mask is applied here.

    Returns:
        ``(p_value, per_model_p, n_total)`` where ``p_value`` is
        ``max_m p_m``.  Because every excluded animal is a
        non-improvement, it is never smaller than the analysis-set p.

    Raises:
        ValueError: On missing/extra models, shape mismatch, a negative
            value, a bad mask, or a non-finite value inside A.
    """
    if set(abs_err_classical_by_model) != set(H2_MODELS):
        raise ValueError(
            "abs_err_classical_by_model must contain exactly "
            f"{H2_MODELS}, got {tuple(abs_err_classical_by_model)}"
        )
    nsmor = _as_1d_nan(abs_err_nsmor, "abs_err_nsmor")
    if np.any(nsmor < 0.0):
        raise ValueError("abs_err_nsmor must be non-negative")
    n_total = int(nsmor.size)
    mask = _analysis_mask(analysis_set, n_total)
    if not np.all(np.isfinite(nsmor[mask])):
        raise ValueError(
            "abs_err_nsmor must be finite on the analysis set"
        )
    out: dict[str, float] = {}
    for model in H2_MODELS:
        array = _as_1d_nan(
            abs_err_classical_by_model[model],
            f"abs_err_classical_by_model[{model!r}]",
        )
        if array.shape != nsmor.shape:
            raise ValueError(
                f"model {model!r} has shape {array.shape}, expected "
                f"{nsmor.shape}"
            )
        if np.any(array < 0.0):
            raise ValueError(
                f"model {model!r} errors must be non-negative"
            )
        if not np.all(np.isfinite(array[mask])):
            raise ValueError(
                f"model {model!r} must be finite on the analysis set"
            )
        difference = nsmor - array - DELTA_NI_MS
        # Excluded animals can never count as improvements, even when
        # their error is finite (r9 blocker).
        below = int(np.count_nonzero(mask & (difference < 0.0)))
        excluded = int(np.count_nonzero(~mask))
        assert below <= int(np.count_nonzero(mask))
        assert below + excluded <= n_total
        out[model] = (
            1.0
            if below == 0
            else float(
                stats.binomtest(
                    below, n_total, 0.5, alternative="greater"
                ).pvalue
            )
        )
    return float(max(out.values())), out, n_total


def _analysis_set_from_mask(
    mask: np.ndarray,
    *,
    reason: str = "",
) -> AnalysisSet:
    """Private constructor of an :class:`AnalysisSet` from a mask.

    Unblinding must use :func:`build_h2_analysis_set`, which carries the
    pre-registered exclusion hierarchy.  This private helper exists only
    for synthetic calibration code, which owns its ground-truth mask.
    It is deliberately **not** a public entry point: the r9 review
    showed that a public mask constructor let any caller bypass the
    all-seed rule and the effective-draw exclusion.

    The resulting object exposes the same strict invariants
    (``mask.sum() == n_animal``, available only at ``n >= N_MIN``).

    Args:
        mask: Boolean ``(n,)`` analysis-set mask.
        reason: Reason string; empty when the set is available.

    Returns:
        :class:`AnalysisSet` whose exclusions are counted under
        ``"outside_mask"``.

    Raises:
        ValueError: If ``mask`` is not a 1-D boolean array.
    """
    array = np.asarray(mask)
    if array.ndim != 1 or array.dtype != np.bool_:
        raise ValueError("mask must be a 1-D boolean array")
    n_animal = int(np.count_nonzero(array))
    return AnalysisSet(
        mask=array,
        n_animal=n_animal,
        excluded={"outside_mask": int(array.size - n_animal)},
        available=n_animal >= N_MIN,
        reason="" if n_animal >= N_MIN else (reason or "out_of_scope"),
    )


def h2_seed_errors(
    l_nsmor_by_seed: np.ndarray,
    l_obs: np.ndarray,
    n_seed_required: int | None = None,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, int]]:
    """Aggregate per-seed closed-loop latencies into H2 ``e_NS``.

    Pre-registered rule (protocol v3 section 3.4):

    * each seed's closed-loop PG-10 onset is aggregated to the animal
      with the same defective-KM median estimator used for ``L_obs``;
    * the per-animal error is the mean of the per-seed absolute errors,
      ``e_NS,i = (1/R) sum_r |L_NS,i^(r) - L_obs,i|`` (average the
      errors, not the latencies);
    * the default requires all ``R`` seeds to be defined.  A missing seed
      excludes the animal and is counted; there is no silent fallback to
      the available seeds.  A caller may lower ``n_seed_required`` only
      as a declared sensitivity analysis.

    Exclusion counts form a non-overlapping hierarchy: an animal with a
    missing seed is counted once under ``l_nsmor_seed_undefined`` (or,
    for a lowered requirement, once under ``n_seed_lt_required``);
    ``included_with_missing_seed`` is a separate diagnostic, not an
    exclusion count.

    Args:
        l_nsmor_by_seed: ``(n, R)`` closed-loop latency per animal and
            seed; NaN marks an undefined (unstable/unresolved) seed.
        l_obs: ``(n,)`` observed defective-KM median latency; NaN marks
            ``F(C) < 0.5``.
        n_seed_required: Minimum finite seeds per animal.  ``None``
            requires all ``R`` seeds.

    Returns:
        ``(e_ns, mask, excluded)``: ``e_ns`` is ``(n,)`` with NaN for
        excluded animals, ``mask`` is the ``(n,)`` boolean availability
        mask, and ``excluded`` counts the exclusion reasons plus the
        ``included_with_missing_seed`` diagnostic.

    Raises:
        ValueError: On wrong shapes, infinities, empty input, or an
            invalid ``n_seed_required``.
    """
    seeds = np.asarray(l_nsmor_by_seed, dtype=np.float64)
    if seeds.ndim != 2:
        raise ValueError(
            f"l_nsmor_by_seed must be 2-D, got shape {seeds.shape}"
        )
    if seeds.shape[0] < 1 or seeds.shape[1] < 1:
        raise ValueError("l_nsmor_by_seed must not be empty")
    if np.any(np.isinf(seeds)):
        raise ValueError("l_nsmor_by_seed contains infinite values")
    obs = _as_1d_nan(l_obs, "l_obs")
    if seeds.shape[0] != obs.size:
        raise ValueError(
            f"l_nsmor_by_seed has {seeds.shape[0]} animals, "
            f"l_obs has {obs.size}"
        )
    n_seed = int(seeds.shape[1])
    if n_seed_required is None:
        required = n_seed
    else:
        if isinstance(n_seed_required, bool) or not isinstance(
            n_seed_required, (int, np.integer)
        ):
            raise ValueError("n_seed_required must be an integer")
        required = int(n_seed_required)
        if required < 1 or required > n_seed:
            raise ValueError(
                f"n_seed_required must lie in [1, {n_seed}], got {required}"
            )
    finite = np.isfinite(seeds)
    n_finite = finite.sum(axis=1)
    obs_ok = np.isfinite(obs)
    mask = obs_ok & (n_finite >= required)
    assert mask.shape == obs.shape
    errors = np.abs(seeds - obs[:, None])
    safe = np.where(finite, errors, 0.0)
    e_ns = np.full(obs.size, np.nan, dtype=np.float64)
    e_ns[mask] = safe[mask].sum(axis=1) / n_finite[mask]
    assert e_ns.shape == obs.shape
    excluded = {
        "l_obs_undefined": int(np.count_nonzero(~obs_ok)),
        "l_nsmor_seed_undefined": int(
            np.count_nonzero(
                obs_ok & (n_finite < required) & (n_finite < n_seed)
            )
        ),
        "n_seed_lt_required": int(
            np.count_nonzero(
                obs_ok & (n_finite < required) & (n_finite == n_seed)
            )
        ),
        "included_with_missing_seed": int(
            np.count_nonzero(obs_ok & mask & (n_finite < n_seed))
        ),
    }
    return e_ns, mask, excluded


def h2_non_inferior_best_classical(
    abs_err_nsmor: np.ndarray,
    abs_err_classical_by_model: Mapping[str, np.ndarray],
    *,
    analysis_set: AnalysisSet,
    abs_err_nsmor_by_seed: np.ndarray | None = None,
    abs_err_classical_by_seed: Mapping[str, np.ndarray] | None = None,
    animal_key: Sequence[object],
    n_boot: int = CI_BOOT,
    seed: int = SEED_BASE,
    compute_ci: bool = True,
) -> TestResult:
    """H2 non-inferiority in absolute latency error.

    For each classical model ``m``,
    ``d_{i,m} = e_NS,i - e_cls,i,m - DELTA_NI_MS``.  The per-model null
    is ``P(d_{i,m} < 0) <= 0.5``; the per-model p-value is the exact
    one-sided binomial sign test.  The slot is an intersection-union
    test: ``p = max_m p_m``.  It tests non-inferiority against **every**
    pre-specified classical model, which implies non-inferiority
    against the best one.  The "best classical model" is the model with
    the largest median ``d_{i,m}`` (smallest absolute error); it is a
    descriptive label, not a data-dependent comparator.

    The **estimand is the median** paired absolute-error difference:
    the test is a majority-of-animals sign test, never a mean test.  A
    mean-based reading (60% of animals improving by 20 ms while 40%
    worsen by 100 ms) can fail non-inferiority while the median passes;
    the protocol must not describe the estimand as a mean.

    Analysis set A is mandatory.  ``analysis_set`` is the frozen object
    returned by :func:`build_h2_analysis_set`; its mask is applied here
    so a caller cannot skip the all-seed rule, the observed/NSMoR
    latency requirements, or the effective-draw >= 40 exclusion by
    passing full-sample arrays.  The NI margin is the frozen module
    constant; it is not an argument and cannot be replaced at the API
    layer.

    Worst-case sensitivity (r9): for a positive claim the report
    additionally requires the worst-case p-value computed here, which
    counts **every animal excluded from A as a non-improvement** (they
    stay in the denominator).  Informative missingness in a
    non-inferiority test is anti-conservative: if the closed loop fails
    on the animals with the longest latencies, the claim rests only on
    the easy animals.  The worst-case p is reported in
    ``effect["worst_case_p_value"]`` and is never smaller than the
    analysis-set p.  The mask is passed to
    :func:`h2_worst_case_p`, so a finite error on an excluded animal is
    still a non-improvement.

    Seed-sign filter (U11/C58).  When the per-seed absolute errors are
    supplied, the effect reports the frozen per-seed statistic
    ``max_m median_i(e_NS,i^(r) - e_cls,i,m^(r))`` and applies
    :func:`seed_sign_consistency` with ``reference=DELTA_NI_MS`` (15 ms).
    Using the max over the four pre-specified models mirrors the IUT and
    selects no model after seeing the data.  A missing per-seed array
    yields ``seed_sign_consistent=None``; the report then withholds a
    positive claim instead of assuming stability.

    Args:
        abs_err_nsmor: Finite non-negative ``(n,)`` absolute latency
            errors of C0 from closed-loop PG-10 onset, on the full
            unblinded sample (the mask is applied internally).
        abs_err_classical_by_model: Mapping with exactly the four keys
            :data:`H2_MODELS`, each a finite non-negative ``(n,)``
            conservative absolute error on the same full sample.
        analysis_set: Frozen analysis set A from
            :func:`build_h2_analysis_set`; its mask must cover the
            input arrays.
        abs_err_nsmor_by_seed: Optional finite ``(n_seed, n)`` per-seed
            C0 absolute errors on the full sample, used only by the
            frozen seed-sign filter.  Must be supplied together with
            ``abs_err_classical_by_seed``.
        abs_err_classical_by_seed: Optional mapping with exactly
            :data:`H2_MODELS` keys, each ``(n_seed, n)`` per-seed
            classical absolute error.
        animal_key: Mandatory ``(n,)`` Cercus animal keys for the full
            sample.  Keys must be unique and length matched; this
            enforces the animal-level inferential unit.
        n_boot: Bootstrap draws for the binding-model effect CI.
        seed: Random seed for the effect CI.
        compute_ci: When ``False``, skip the bootstrap CI.

    Returns:
        :class:`TestResult` named ``H2_NI``.  ``n_A < 10`` returns
        ``available=False`` with ``p=1``.

    Raises:
        ValueError: On missing/extra models, shape mismatch, negative
            errors, non-finite values, or a mask that does not match
            the input length.
    """
    if set(abs_err_classical_by_model) != set(H2_MODELS):
        raise ValueError(
            "abs_err_classical_by_model must contain exactly "
            f"{H2_MODELS}, got {tuple(abs_err_classical_by_model)}"
        )
    nsmor_full = _as_1d_nan(abs_err_nsmor, "abs_err_nsmor")
    if np.any(nsmor_full < 0.0):
        raise ValueError("abs_err_nsmor must be non-negative")
    n_total = int(nsmor_full.size)
    _check_animal_key(animal_key, n_total, "animal_key")
    classical_full: list[np.ndarray] = []
    for model in H2_MODELS:
        array = _as_1d_nan(
            abs_err_classical_by_model[model],
            f"abs_err_classical_by_model[{model!r}]",
        )
        if array.shape != nsmor_full.shape:
            raise ValueError(
                f"model {model!r} has shape {array.shape}, expected "
                f"{nsmor_full.shape}"
            )
        if np.any(array < 0.0):
            raise ValueError(
                f"model {model!r} errors must be non-negative"
            )
        classical_full.append(array)
    mask = _analysis_mask(analysis_set, n_total)
    delta_ni_ms = DELTA_NI_MS
    n_boot = _check_n_boot(n_boot)
    seed = _check_seed(seed)
    name = "H2_NI"
    n = int(np.count_nonzero(mask))
    assert n == int(analysis_set.n_animal)
    if n < N_MIN:
        return _unavailable(
            name, delta_ni_ms, n, n_boot, seed, "out_of_scope (n_lt_10)"
        )
    nsmor = nsmor_full[mask]
    assert nsmor.shape == (n,)
    if not np.all(np.isfinite(nsmor)):
        raise ValueError(
            "abs_err_nsmor must be finite on the analysis set"
        )
    for model, array in zip(H2_MODELS, classical_full):
        if not np.all(np.isfinite(array[mask])):
            raise ValueError(
                f"model {model!r} must be finite on the analysis set"
            )
    cls_matrix = np.column_stack([array[mask] for array in classical_full])
    assert cls_matrix.shape == (n, len(H2_MODELS))
    differences = nsmor[:, None] - cls_matrix - delta_ni_ms
    assert differences.shape == (n, len(H2_MODELS))
    p_model = np.empty(len(H2_MODELS), dtype=np.float64)
    medians = np.empty(len(H2_MODELS), dtype=np.float64)
    below = np.empty(len(H2_MODELS), dtype=np.int64)
    for index in range(len(H2_MODELS)):
        p_model[index], diagnostics = _exact_sign_test(
            differences[:, index], 0.0
        )
        medians[index] = float(diagnostics["median_difference"])
        below[index] = int(diagnostics["n_below_boundary"])
    assert p_model.shape == (len(H2_MODELS),)
    p_value = float(np.max(p_model))
    worst_case_p_value, worst_case_by_model, worst_total = h2_worst_case_p(
        nsmor_full,
        {
            model: classical_full[index]
            for index, model in enumerate(H2_MODELS)
        },
        analysis_set=analysis_set,
    )
    assert worst_total == n_total
    # Masking every excluded animal as a non-improvement can only move
    # the binomial upper tail up when the denominator grows.
    assert worst_case_p_value >= p_value - 1e-12, (
        "worst-case p must never be smaller than the analysis-set p: "
        f"{worst_case_p_value} < {p_value}"
    )

    seed_sign_consistent: bool | None = None
    seed_sign_statistics: list[float] | None = None
    seed_sign_n_seed: int | None = None
    seed_sign_reason = "per_seed_arrays_missing"
    if (abs_err_nsmor_by_seed is None) != (
        abs_err_classical_by_seed is None
    ):
        raise ValueError(
            "abs_err_nsmor_by_seed and abs_err_classical_by_seed must "
            "be supplied together"
        )
    if abs_err_nsmor_by_seed is not None:
        nsmor_by_seed = np.asarray(
            abs_err_nsmor_by_seed, dtype=np.float64
        )
        if nsmor_by_seed.ndim != 2 or nsmor_by_seed.shape[1] != n_total:
            raise ValueError(
                "abs_err_nsmor_by_seed must have shape "
                f"(n_seed, {n_total}), got {nsmor_by_seed.shape}"
            )
        n_seed = int(nsmor_by_seed.shape[0])
        if n_seed < 2:
            raise ValueError("the seed-sign filter needs at least 2 seeds")
        if np.any(np.isinf(nsmor_by_seed)):
            raise ValueError("abs_err_nsmor_by_seed contains infinities")
        if np.any(nsmor_by_seed < 0.0):
            raise ValueError("abs_err_nsmor_by_seed must be non-negative")
        if not np.all(np.isfinite(nsmor_by_seed[:, mask])):
            raise ValueError(
                "abs_err_nsmor_by_seed must be finite on the analysis set"
            )
        assert abs_err_classical_by_seed is not None
        if set(abs_err_classical_by_seed) != set(H2_MODELS):
            raise ValueError(
                "abs_err_classical_by_seed must contain exactly "
                f"{H2_MODELS}, got {tuple(abs_err_classical_by_seed)}"
            )
        seed_stats = np.empty(n_seed, dtype=np.float64)
        for seed_index in range(n_seed):
            per_model_medians = np.empty(
                len(H2_MODELS), dtype=np.float64
            )
            for model_index, model in enumerate(H2_MODELS):
                array = np.asarray(
                    abs_err_classical_by_seed[model], dtype=np.float64
                )
                if array.shape != nsmor_by_seed.shape:
                    raise ValueError(
                        f"abs_err_classical_by_seed[{model!r}] must have "
                        f"shape {nsmor_by_seed.shape}, got {array.shape}"
                    )
                if np.any(np.isinf(array)):
                    raise ValueError(
                        f"abs_err_classical_by_seed[{model!r}] contains "
                        "infinities"
                    )
                if np.any(array < 0.0):
                    raise ValueError(
                        f"abs_err_classical_by_seed[{model!r}] must be "
                        "non-negative"
                    )
                if not np.all(np.isfinite(array[:, mask])):
                    raise ValueError(
                        f"abs_err_classical_by_seed[{model!r}] must be "
                        "finite on the analysis set"
                    )
                differences_by_seed = (
                    nsmor_by_seed[seed_index, mask]
                    - array[seed_index, mask]
                )
                per_model_medians[model_index] = float(
                    np.median(differences_by_seed)
                )
            seed_stats[seed_index] = float(
                np.max(per_model_medians)
            )
        assert seed_stats.shape == (n_seed,)
        seed_sign_consistent = seed_sign_consistency(
            seed_stats, reference=DELTA_NI_MS
        )
        seed_sign_statistics = [float(value) for value in seed_stats]
        seed_sign_n_seed = n_seed
        seed_sign_reason = ""

    binding_index = int(np.argmax(medians))
    p_determining_index = int(np.argmax(p_model))
    binding_model = H2_MODELS[binding_index]
    p_determining_model = H2_MODELS[p_determining_index]
    d_z = _cohens_dz_1d(differences[:, binding_index])
    ci_low: float | None = None
    ci_high: float | None = None
    binding_ci_low: float | None = None
    binding_ci_high: float | None = None
    if compute_ci:
        idx = _bootstrap_indices(n, n_boot, seed)
        boot = np.median(differences[idx], axis=1)
        assert boot.shape == (n_boot, len(H2_MODELS))
        boot_max = boot.max(axis=1)
        ci_low, ci_high = _percentile_ci(boot_max)
        binding_ci_low, binding_ci_high = _percentile_ci(
            boot[:, binding_index]
        )
    effect: dict[str, object] = {
        "binding_model": binding_model,
        "binding_model_is_best_classical": True,
        "p_determining_model": p_determining_model,
        "winner_conditional": True,
        "d_z": d_z,
        "median_difference": float(medians[binding_index]),
        "binding_ci_low": binding_ci_low,
        "binding_ci_high": binding_ci_high,
        "model_median_differences": {
            model: float(medians[index])
            for index, model in enumerate(H2_MODELS)
        },
        "model_n_below_boundary": {
            model: int(below[index])
            for index, model in enumerate(H2_MODELS)
        },
        "model_p_values": {
            model: float(p_model[index])
            for index, model in enumerate(H2_MODELS)
        },
        "worst_case_p_value": worst_case_p_value,
        "worst_case_model_p_values": dict(worst_case_by_model),
        "worst_case_n_animal": n_total,
        "worst_case_excluded_as_non_improvement": n_total - n,
        "seed_sign_consistent": seed_sign_consistent,
        "seed_sign_statistics": seed_sign_statistics,
        "seed_sign_n_seed": seed_sign_n_seed,
        "seed_sign_reference": delta_ni_ms,
        "seed_sign_construction": (
            "per_seed_max_over_models_median_difference"
        ),
        "seed_sign_reason": seed_sign_reason,
        "analysis_set_excluded": dict(analysis_set.excluded),
        "analysis_set_reason": analysis_set.reason,
        "delta_ni_ms": delta_ni_ms,
        "construction": "iut_exact_sign_test",
        "direction": "max_over_models",
        "tie_policy": TIE_POLICY,
        "scientific_estimand": (
            "median absolute-latency-error difference; "
            "majority-of-animals non-inferiority"
        ),
    }
    return TestResult(
        name=name,
        statistic=float(np.max(medians)),
        boundary=delta_ni_ms,
        p_value=p_value,
        n_animal=n,
        n_boot=n_boot,
        seed=seed,
        available=True,
        reason="",
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def h2_descriptive_row(
    abs_err_nsmor: np.ndarray,
    abs_err_classical_by_model: Mapping[str, np.ndarray],
    *,
    analysis_set: AnalysisSet,
    n_boot: int = CI_BOOT,
    seed: int = SEED_BASE,
) -> dict[str, object]:
    """Build the descriptive H2 row without computing any p-value.

    When H2 is downgraded (expected split n < 15, or the one-shot
    calibration failed), the protocol still requires a descriptive row
    with the binding classical model and an animal-cluster CI.  Calling
    :func:`h2_non_inferior_best_classical` would compute per-model
    p-values and invite a p-value to leak into a descriptive claim, so
    this function is the only sanctioned path for that row.  It never
    calls the sign test and never returns ``p_value``,
    ``model_p_values``, or ``p_determining_model``.

    The binding model is the model with the largest median
    ``d_{i,m} = e_NS,i - e_cls,i,m - delta_NI`` (smallest absolute
    error).  Its CI is a winner-conditional percentile bootstrap CI and
    is labelled as such.  The row is descriptive: it carries no
    multiplicity control and cannot support a non-inferiority claim.

    Args:
        abs_err_nsmor: Finite non-negative ``(n,)`` C0 absolute latency
            errors from closed-loop PG-10 onset.
        abs_err_classical_by_model: Mapping with exactly
            :data:`H2_MODELS` keys, each a finite non-negative ``(n,)``
            conservative absolute error.
        analysis_set: Frozen analysis set A from
            :func:`build_h2_analysis_set`.
        n_boot: Bootstrap draws for the binding-model median CI.
        seed: Random seed for the bootstrap CI.

    Returns:
        JSON-serialisable dictionary with ``available``, ``n_animal``,
        ``binding_model``, ``median_difference``, ``d_z``,
        ``model_median_differences``, ``ci_low``, ``ci_high``,
        ``n_boot``, ``seed``, ``inference="descriptive"``,
        ``p_value=None``, ``multiplicity_control="none"``, and the
        descriptive claim limit.  ``available=False`` with
        ``reason="out_of_scope (n_lt_10)"`` when ``n < N_MIN``.

    Raises:
        ValueError: On missing/extra models, shape mismatch, negative
            errors, non-finite values, or invalid bootstrap arguments.
    """
    if set(abs_err_classical_by_model) != set(H2_MODELS):
        raise ValueError(
            "abs_err_classical_by_model must contain exactly "
            f"{H2_MODELS}, got {tuple(abs_err_classical_by_model)}"
        )
    nsmor_full = _as_1d_nan(abs_err_nsmor, "abs_err_nsmor")
    if np.any(nsmor_full < 0.0):
        raise ValueError("abs_err_nsmor must be non-negative")
    mask = np.asarray(getattr(analysis_set, "mask", None))
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("analysis_set.mask must be a 1-D boolean array")
    if mask.size != nsmor_full.size:
        raise ValueError(
            f"analysis_set.mask has length {mask.size}, expected "
            f"{nsmor_full.size}"
        )
    nsmor = nsmor_full[mask]
    n = int(nsmor.size)
    if n >= N_MIN and not np.all(np.isfinite(nsmor)):
        raise ValueError("abs_err_nsmor must be finite on the analysis set")
    classical_arrays: list[np.ndarray] = []
    for model in H2_MODELS:
        raw = _as_1d_nan(
            abs_err_classical_by_model[model],
            f"abs_err_classical_by_model[{model!r}]",
        )
        if raw.shape != nsmor_full.shape:
            raise ValueError(
                f"model {model!r} has shape {raw.shape}, expected "
                f"{nsmor_full.shape}"
            )
        if np.any(raw < 0.0):
            raise ValueError(
                f"model {model!r} errors must be non-negative"
            )
        array = raw[mask]
        if n >= N_MIN and not np.all(np.isfinite(array)):
            raise ValueError(
                f"model {model!r} must be finite on the analysis set"
            )
        classical_arrays.append(array)
    n_boot = _check_n_boot(n_boot)
    seed = _check_seed(seed)
    row: dict[str, object] = {
        "available": n >= N_MIN,
        "n_animal": n,
        "binding_model": None,
        "median_difference": None,
        "d_z": None,
        "model_median_differences": None,
        "ci_low": None,
        "ci_high": None,
        "n_boot": n_boot,
        "seed": seed,
        "inference": "descriptive",
        "p_value": None,
        "multiplicity_control": "none",
        "claim_limit": (
            "no superiority/non-inferiority/equivalence claim"
        ),
        "winner_conditional": True,
        "construction": "descriptive_binding_model_no_p",
        "reason": "",
    }
    if n < N_MIN:
        row["reason"] = "out_of_scope (n_lt_10)"
        return row
    cls_matrix = np.column_stack(classical_arrays)
    assert cls_matrix.shape == (n, len(H2_MODELS))
    differences = nsmor[:, None] - cls_matrix - DELTA_NI_MS
    assert differences.shape == (n, len(H2_MODELS))
    medians = np.median(differences, axis=0)
    assert medians.shape == (len(H2_MODELS),)
    binding_index = int(np.argmax(medians))
    binding_model = H2_MODELS[binding_index]
    idx = _bootstrap_indices(n, n_boot, seed)
    boot = np.median(differences[idx, binding_index], axis=1)
    assert boot.shape == (n_boot,)
    ci_low, ci_high = _percentile_ci(boot)
    row.update(
        {
            "binding_model": binding_model,
            "median_difference": float(medians[binding_index]),
            "d_z": _cohens_dz_1d(differences[:, binding_index]),
            "model_median_differences": {
                model: float(medians[index])
                for index, model in enumerate(H2_MODELS)
            },
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )
    return row


# ---------------------------------------------------------------------------
# H3: CAMoR-L2 de-recursion effect
# ---------------------------------------------------------------------------


def _check_bool_flags(
    flags: Sequence[bool] | None,
    name: str,
) -> np.ndarray | None:
    """Validate an optional 1-D boolean flag sequence.

    ``None`` is returned unchanged so callers can distinguish a missing
    status (fail-closed) from a present all-True status.
    """
    if flags is None:
        return None
    if isinstance(flags, (str, bytes)):
        return None
    try:
        values = list(flags)
    except TypeError:
        return None
    if not values:
        return None
    if not all(isinstance(value, (bool, np.bool_)) for value in values):
        return None
    array = np.asarray(values, dtype=np.bool_)
    if array.ndim != 1 or array.size != len(values):
        return None
    return array


def h3_camor_derecursion(
    mse_c1s: np.ndarray,
    mse_c16: np.ndarray,
    *,
    budget_matched_by_seed: Sequence[bool],
    oom_by_seed: Sequence[bool],
    mse_c1s_by_seed: np.ndarray | None = None,
    mse_c16_by_seed: np.ndarray | None = None,
    animal_key: Sequence[object],
    n_boot: int = CI_BOOT,
    seed: int = SEED_BASE,
    compute_ci: bool = True,
) -> TestResult:
    """H3 CAMoR-L2 de-recursion effect (C1-S vs C16).

    ``D_i = MSE_{i,C1-S} - MSE_{i,C16}`` on the common eligible-frame
    intersection.  The null is ``median D_i >= 0``; the alternative is
    ``median D_i < 0`` (a majority of animals improve under stochastic
    depth).  The p-value is the exact one-sided binomial sign test.

    Budget and OOM gate (protocol section 3.5): the H3 slot is
    confirmatory only when every training seed's parameter-budget
    comparison is MATCHED and no seed hit the K=16 OOM guard.  Both
    per-seed status arrays are **required keyword arguments** with no
    default: an optional ``oom_by_seed`` silently became all-``False``
    and let a caller bypass the OOM gate while recording fabricated
    "no OOM" evidence.  A missing or wrong-length status fails closed;
    ``None`` raises.

    Per-seed reporting (U11): when the per-seed MSE arrays are
    supplied, the effect reports each seed's median difference and the
    seed-sign consistency with ``reference=0``.  A positive H3 claim
    additionally requires :func:`confirmatory_report` to see a
    consistent seed sign.

    Args:
        mse_c1s: Finite non-negative ``(n,)`` per-animal MSE for C1-S.
        mse_c16: Finite non-negative ``(n,)`` per-animal MSE for C16 on
            the same eligible frames.
        budget_matched_by_seed: One boolean per training seed; ``True``
            iff the C1-S/C16 Linear-hook budget comparison is MATCHED
            (tolerance 4,128).  Required.
        oom_by_seed: One boolean per training seed; ``True`` iff that
            seed hit the K=16 OOM guard.  Required.
        mse_c1s_by_seed: Optional ``(n_seed, n)`` per-seed C1-S MSE on
            the same animals and frames.
        mse_c16_by_seed: Optional ``(n_seed, n)`` per-seed C16 MSE.
        animal_key: Mandatory ``(n,)`` Cercus animal keys.  Keys must be
            unique and length matched; this enforces the animal-level
            inferential unit.
        n_boot: Bootstrap draws for the effect CI.
        seed: Random seed for the effect CI.
        compute_ci: When ``False``, skip the bootstrap CI.

    Returns:
        :class:`TestResult` named ``H3_derecursion``.  Missing/invalid
        budget or OOM status, any NOTMATCHED seed, any OOM seed, or
        ``n < 10`` returns ``available=False`` with ``p=1``.

    Raises:
        TypeError: If ``budget_matched_by_seed`` or ``oom_by_seed`` is
            omitted.
        ValueError: On shape mismatch, non-finite values, negative MSE,
            or per-seed arrays that do not match ``(n_seed, n)``.
    """
    # Both statuses are required keyword arguments: omitting either
    # raises TypeError at the call boundary.  ``None`` is accepted only
    # so a caller that threads an absent status gets an explicit
    # fail-closed result instead of an all-False default.
    if budget_matched_by_seed is None:
        return _unavailable(
            "H3_derecursion",
            SLOT_BOUNDARIES["H3_derecursion"],
            int(np.asarray(mse_c1s).size) if np.ndim(mse_c1s) else 0,
            _check_n_boot(n_boot),
            _check_seed(seed),
            "budget_status_missing",
        )
    if oom_by_seed is None:
        return _unavailable(
            "H3_derecursion",
            SLOT_BOUNDARIES["H3_derecursion"],
            int(np.asarray(mse_c1s).size) if np.ndim(mse_c1s) else 0,
            _check_n_boot(n_boot),
            _check_seed(seed),
            "oom_status_missing",
        )
    c1s = _as_1d(mse_c1s, "mse_c1s")
    c16 = _as_1d(mse_c16, "mse_c16")
    if c1s.shape != c16.shape:
        raise ValueError(
            f"mse_c1s shape {c1s.shape} != mse_c16 shape {c16.shape}"
        )
    if np.any(c1s < 0.0) or np.any(c16 < 0.0):
        raise ValueError("MSE values must be non-negative")
    n = int(c1s.size)
    _check_animal_key(animal_key, n, "animal_key")
    n_boot = _check_n_boot(n_boot)
    seed = _check_seed(seed)
    boundary = SLOT_BOUNDARIES["H3_derecursion"]
    name = "H3_derecursion"

    budget_flags = _check_bool_flags(
        budget_matched_by_seed, "budget_matched_by_seed"
    )
    if budget_flags is None:
        return _unavailable(
            name, boundary, n, n_boot, seed, "budget_status_missing"
        )
    oom_flags = _check_bool_flags(oom_by_seed, "oom_by_seed")
    if oom_flags is None or oom_flags.size != budget_flags.size:
        return _unavailable(
            name, boundary, n, n_boot, seed, "oom_status_missing"
        )
    if not bool(np.all(budget_flags)):
        bad = [
            int(index)
            for index, value in enumerate(budget_flags)
            if not bool(value)
        ]
        return _unavailable(
            name,
            boundary,
            n,
            n_boot,
            seed,
            f"budget_not_matched (seeds={bad})",
        )
    if bool(np.any(oom_flags)):
        bad = [
            int(index)
            for index, value in enumerate(oom_flags)
            if bool(value)
        ]
        return _unavailable(
            name, boundary, n, n_boot, seed, f"oom_k16 (seeds={bad})"
        )
    assert budget_flags.ndim == 1
    assert oom_flags.shape == budget_flags.shape
    if n < N_MIN:
        return _unavailable(
            name, boundary, n, n_boot, seed, "out_of_scope (n_lt_10)"
        )

    per_seed_medians: list[float] | None = None
    seed_consistent: bool | None = None
    if (mse_c1s_by_seed is None) != (mse_c16_by_seed is None):
        raise ValueError(
            "mse_c1s_by_seed and mse_c16_by_seed must be supplied together"
        )
    if mse_c1s_by_seed is not None and mse_c16_by_seed is not None:
        c1s_seed = np.asarray(mse_c1s_by_seed, dtype=np.float64)
        c16_seed = np.asarray(mse_c16_by_seed, dtype=np.float64)
        if c1s_seed.shape != c16_seed.shape:
            raise ValueError(
                f"per-seed shapes differ: {c1s_seed.shape} vs "
                f"{c16_seed.shape}"
            )
        if c1s_seed.ndim != 2 or c1s_seed.shape[1] != n:
            raise ValueError(
                "per-seed arrays must be (n_seed, n) with n="
                f"{n}, got {c1s_seed.shape}"
            )
        if c1s_seed.shape[0] != budget_flags.size:
            raise ValueError(
                f"per-seed arrays have {c1s_seed.shape[0]} seeds but "
                f"{budget_flags.size} budget flags"
            )
        if not np.all(np.isfinite(c1s_seed)) or not np.all(
            np.isfinite(c16_seed)
        ):
            raise ValueError("per-seed MSE arrays must be finite")
        if np.any(c1s_seed < 0.0) or np.any(c16_seed < 0.0):
            raise ValueError("per-seed MSE values must be non-negative")
        per_seed = np.median(c1s_seed - c16_seed, axis=1)
        assert per_seed.shape == (c1s_seed.shape[0],)
        per_seed_medians = [float(value) for value in per_seed]
        seed_consistent = bool(seed_sign_consistency(per_seed, 0.0))

    differences = c1s - c16
    assert differences.shape == (n,)
    p_value, diagnostics = _exact_sign_test(differences, boundary)
    d_z = _cohens_dz_1d(differences)
    ci_low: float | None = None
    ci_high: float | None = None
    if compute_ci:
        idx = _bootstrap_indices(n, n_boot, seed)
        boot = np.median(differences[idx], axis=1)
        assert boot.shape == (n_boot,)
        ci_low, ci_high = _percentile_ci(boot)
    effect: dict[str, object] = {
        "median_difference": float(np.median(differences)),
        "d_z": d_z,
        "construction": "exact_sign_test",
        "budget_matched": True,
        "n_seed_budget_checked": int(budget_flags.size),
        "budget_matched_by_seed": [
            bool(value) for value in budget_flags
        ],
        "oom_by_seed": [bool(value) for value in oom_flags],
        "per_seed_median_differences": per_seed_medians,
        "seed_sign_consistent": seed_consistent,
        "seed_sign_reference": 0.0,
        "scientific_estimand": (
            "median MSE difference; majority-of-animals improvement"
        ),
    }
    effect.update(diagnostics)
    return TestResult(
        name=name,
        statistic=float(np.median(differences)),
        boundary=boundary,
        p_value=p_value,
        n_animal=n,
        n_boot=n_boot,
        seed=seed,
        available=True,
        reason="",
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def holm_confirmatory_family(
    p_values: Mapping[str, float],
    order: Sequence[str] = FAMILY_ORDER,
    *,
    multisensory_test_n: int | None,
    h2_calibration_pass: bool | None,
    split_sha256: str | None = None,
    n_boot: int | None = None,
    mc_se: Mapping[str, float] | None = None,
) -> HolmResult:
    """Apply the frozen Holm family to the four confirmatory slots.

    The order is part of the pre-registration: keys must match
    :data:`FAMILY_ORDER` exactly, ``m`` is fixed at :data:`FAMILY_M`,
    and ties are broken by the frozen order.  Unavailable slots enter
    with ``p=1`` and occupy their slot.

    Fail-closed rules:

    * The active order is **not** an argument.  It is recomputed by
      :func:`resolve_active_confirmatory_order` from the split SHA's
      realised multisensory test n and the H2 calibration verdict, so a
      caller cannot hand over an arbitrary subset (the r9 review showed
      that passing ``FAMILY_ORDER`` at split n=13 promoted H2 against
      the pre-registration).  ``None`` split facts fail closed to
      :data:`FAIL_CLOSED_ACTIVE_ORDER`.
    * A slot outside ``active_order`` must carry ``p_value == 1.0``.
      Passing a real p-value for a downgraded slot raises instead of
      silently promoting it into the family.
    * If Monte Carlo standard errors are supplied (``mc_se`` or
      ``n_boot``), the adjusted-p MC SE is the Holm multiplier
      ``(m - rank)`` times the raw MC SE of the slot that determines the
      adjusted p.  A slot whose adjusted p lies within two adjusted MC
      SEs of ``alpha`` is marked ``indeterminate_mc`` and is **not**
      rejected.  Exact discrete p-values pass ``mc_se=0`` and have no
      Monte Carlo uncertainty.

    Args:
        p_values: Mapping from slot name to raw p-value.
        order: Frozen slot order; must equal :data:`FAMILY_ORDER`.
        multisensory_test_n: Split-fixed multisensory test animal
            count, or ``None`` before the split is frozen.
        h2_calibration_pass: One-shot H2 calibration verdict, or
            ``None`` when it has not run.
        split_sha256: Optional split SHA; the frozen PG-02 SHA binds the
            multisensory test n to 14 and a mismatch raises.
        n_boot: Optional bootstrap draw count used to derive raw MC SEs
            ``sqrt(p(1-p)/(n_boot+1))`` when ``mc_se`` is not given.
        mc_se: Optional mapping from slot name to raw-p MC SE.

    Returns:
        :class:`HolmResult` with adjusted p-values, rejection flags,
        adjusted MC SEs, and ``indeterminate_mc`` flags.

    Raises:
        ValueError: If the order is changed, keys are missing or extra,
            p-values are invalid, a downgraded slot carries ``p != 1``,
            or the resolved active slots are not part of the family.
    """
    if split_sha256 is not None:
        _check_split_facts(split_sha256, multisensory_test_n)
    order_tuple = tuple(order)
    if order_tuple != FAMILY_ORDER:
        raise ValueError(
            f"Holm order is frozen as {FAMILY_ORDER}, got {order_tuple}"
        )
    if len(order_tuple) != FAMILY_M or len(set(order_tuple)) != FAMILY_M:
        raise ValueError("Holm family must contain four unique slots")
    active_tuple = resolve_active_confirmatory_order(
        multisensory_test_n,
        h2_calibration_pass,
        split_sha256=split_sha256,
    )
    if len(set(active_tuple)) != len(active_tuple):
        raise ValueError("active_order must not contain duplicates")
    if not set(active_tuple).issubset(set(order_tuple)):
        raise ValueError(
            f"active_order must be a subset of {order_tuple}, got "
            f"{active_tuple}"
        )
    if set(p_values) != set(order_tuple):
        raise ValueError(
            f"p_values keys must equal {order_tuple}, got {tuple(p_values)}"
        )
    ordered = {name: float(p_values[name]) for name in order_tuple}
    for name, p in ordered.items():
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(
                f"invalid p-value for {name!r}: {p!r}; must be finite "
                "and within [0, 1]"
            )
    for name in order_tuple:
        if name not in active_tuple and ordered[name] != 1.0:
            raise ValueError(
                f"slot {name!r} is downgraded to descriptive; its p-value "
                f"must be exactly 1.0, got {ordered[name]!r}"
            )

    if mc_se is not None:
        if set(mc_se) != set(order_tuple):
            raise ValueError(
                f"mc_se keys must equal {order_tuple}, got {tuple(mc_se)}"
            )
        raw_mc_se = {name: float(mc_se[name]) for name in order_tuple}
        for name, value in raw_mc_se.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"invalid mc_se for {name!r}: {value!r}"
                )
    elif n_boot is not None:
        n_boot = _check_n_boot(n_boot)
        raw_mc_se = {
            name: math.sqrt(
                ordered[name] * (1.0 - ordered[name]) / (n_boot + 1.0)
            )
            for name in order_tuple
        }
    else:
        raw_mc_se = {name: 0.0 for name in order_tuple}

    raw = holm_bonferroni(ordered)
    adjusted = {name: float(raw[name][0]) for name in order_tuple}
    significant = {name: bool(raw[name][1]) for name in order_tuple}

    # Reconstruct the dominating rank for each slot so the MC SE of the
    # adjusted p uses the correct multiplier.
    sorted_items = sorted(
        order_tuple, key=lambda name: (ordered[name], order_tuple.index(name))
    )
    adjusted_mc_se: dict[str, float] = {}
    indeterminate: dict[str, bool] = {}
    for rank, name in enumerate(sorted_items):
        candidates = sorted_items[: rank + 1]
        best = max(
            candidates,
            key=lambda candidate: (
                (FAMILY_M - sorted_items.index(candidate))
                * ordered[candidate],
                -sorted_items.index(candidate),
            ),
        )
        multiplier = FAMILY_M - sorted_items.index(best)
        adjusted_mc_se[name] = multiplier * raw_mc_se[best]
        # Exact discrete p-values have mc_se = 0 and therefore no Monte
        # Carlo uncertainty; an adjusted p exactly at alpha is
        # determined and must be rejected.  The guard only applies to
        # MC-estimated p.
        indeterminate[name] = (
            adjusted_mc_se[name] > 0.0
            and abs(adjusted[name] - ALPHA) <= 2.0 * adjusted_mc_se[name]
        )
        if indeterminate[name]:
            significant[name] = False

    return HolmResult(
        order=order_tuple,
        m=FAMILY_M,
        alpha=ALPHA,
        raw_p=ordered,
        adjusted_p=adjusted,
        significant=significant,
        indeterminate_mc=indeterminate,
        mc_se=adjusted_mc_se,
    )


def descriptive_placeholder(
    name: str,
    n_animal: int,
    n_boot: int = CI_BOOT,
    seed: int = SEED_BASE,
    reason: str = "descriptive_after_calibration_failure",
) -> TestResult:
    """Build the ``p=1`` placeholder for a downgraded confirmatory slot.

    The slot still occupies its position in the fixed m=4 Holm family,
    so the family remains conservative.  The returned result must never
    be interpreted as negative evidence.

    Args:
        name: Slot name, one of :data:`FAMILY_ORDER`.
        n_animal: Number of animal-level observations available.
        n_boot: Bootstrap draws recorded in the result.
        seed: Random seed recorded in the result.
        reason: Fail-closed reason recorded in the report.

    Returns:
        A :class:`TestResult` with ``available=False`` and ``p_value=1``.

    Raises:
        ValueError: If ``name`` is not a frozen confirmatory slot.
    """
    if name not in FAMILY_ORDER:
        raise ValueError(f"unknown confirmatory slot: {name!r}")
    return _unavailable(
        name,
        SLOT_BOUNDARIES[name],
        n_animal,
        _check_n_boot(n_boot),
        _check_seed(seed),
        reason,
    )


def confirmatory_report(
    results: Sequence[TestResult] | Mapping[str, TestResult],
    *,
    split_sha256: str,
    multisensory_test_n: int | None,
    h2_calibration_pass: bool | None,
    seed_consistency: Mapping[str, Mapping[str, object]] | None = None,
    descriptive_rows: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build a JSON-serialisable report for the four-slot family.

    The active order is **not** a caller argument.  It is recomputed
    here by :func:`resolve_active_confirmatory_order` from the two
    frozen split facts, so an unblinding script cannot promote H2 (or
    drop a slot) by passing an arbitrary subset.  The previous signature
    accepted any ``active_order`` that was a subset of the family; the
    r9 review showed that passing ``FAMILY_ORDER`` at split n=13 put H2
    into the confirmatory family against the pre-registration, enforced
    only by caller discipline.

    Claim levels are no longer "confirmatory for every active slot":

    * ``"confirmatory"`` -- active, available, rejected after Holm, and
      every pre-registered claim restriction is satisfied (H1/H3 seed
      sign consistency; H2 worst-case missingness gate);
    * ``"confirmatory_no_rejection"`` -- active and available but not
      rejected; a genuine negative evaluation;
    * ``"indeterminate_unstable_across_seeds"`` -- rejected after Holm
      but at least one seed is not strictly on the improvement side
      (H1/H3 ``< 0``; H2 ``< 15 ms``), or the seed-sign evidence is
      missing.  Agreement on the wrong side is unstable.  The p-value
      is reported, the positive claim is withheld (protocol U11/C58).
      H2 is not exempt: its slot function computes the frozen 15 ms
      seed-sign statistic, and a missing per-seed array withholds the
      claim;
    * ``"indeterminate_informative_missingness"`` -- H2 was rejected
      after Holm but its worst-case missingness p exceeds the
      ``alpha/m`` gate.  This is not a confirmatory negative: the
      analysis-set result is reported as indeterminate because the
      excluded animals could overturn the positive claim;
    * ``"unavailable"`` -- active but not evaluable (for example H3
      downgraded by the budget or OOM gate).  "Not evaluable" is never
      reported as a confirmatory negative;
    * ``"descriptive"`` -- downgraded slot, no p, no Holm.

    Args:
        results: The four :class:`TestResult` objects, as a sequence or
            a mapping keyed by slot name.
        split_sha256: Split SHA.  The frozen PG-02 SHA binds the
            multisensory test n to 14; any other n raises.  Synthetic
            ``synthetic:`` fixtures are simulation-only.
        multisensory_test_n: Split-fixed multisensory test animal count,
            or ``None`` before the split is frozen.
        h2_calibration_pass: One-shot H2 calibration verdict, or
            ``None`` when it has not run.
        seed_consistency: Optional mapping from slot name to the seed
            evidence.  H1 entries must be the **full** dictionary
            returned by :func:`h1_seed_sign_consistency`; the report
            cross-checks its ``scope`` and ``winning_arm`` against the
            H1 result and rejects a bare boolean.  H3 may carry a
            boolean ``consistent`` entry, but it may not contradict the
            slot's computed ``seed_sign_consistent``.  H2 evidence is
            computed only by :func:`h2_non_inferior_best_classical`; a
            caller entry cannot substitute for missing slot evidence.
            When evidence is absent, a rejected H1/H2/H3 slot is still
            reported but its claim level is
            ``"indeterminate_unstable_across_seeds"`` -- the claim is
            withheld rather than assumed stable.
        descriptive_rows: Optional mapping from a downgraded slot name to
            a descriptive row produced by a no-p constructor (for H2,
            :func:`h2_descriptive_row`).  The row is merged into the
            downgraded slot's ``effect`` and echoed under the report's
            ``descriptive_rows`` key.  p-derived keys are stripped even
            if a caller supplies them.

    Returns:
        Dictionary with ``family`` metadata and per-slot raw/adjusted
        p-values, adjusted MC SEs, rejection flags, availability, effect
        sizes, CIs, ``claim_level`` and ``descriptive_claim_limit``.
        ``family["h1_joint_gate"]`` reports the pre-registered rule that
        H1 passes only when both ``H1_onset`` and ``H1_h5`` are rejected
        after Holm **and** their seed signs agree.

    Raises:
        ValueError: If the four slots are not present exactly once, a
            downgraded slot carries an available result or ``p != 1``, a
            descriptive row names an unknown or active slot, an active
            H3 result lacks the budget evidence, an active H2 result
            carries a boundary other than :data:`DELTA_NI_MS`, or a
            supplied seed-consistency entry has the wrong shape.
    """
    _check_split_facts(split_sha256, multisensory_test_n)
    if isinstance(results, Mapping):
        items = list(results.values())
    else:
        items = list(results)
    names = [result.name for result in items]
    if len(names) != FAMILY_M or len(set(names)) != FAMILY_M:
        raise ValueError(
            f"confirmatory_report needs exactly {FAMILY_M} unique slots, "
            f"got {names}"
        )
    if set(names) != set(FAMILY_ORDER):
        raise ValueError(
            f"confirmatory_report slots must equal {FAMILY_ORDER}, "
            f"got {tuple(names)}"
        )
    by_name = {result.name: result for result in items}
    active_order_tuple = resolve_active_confirmatory_order(
        multisensory_test_n,
        h2_calibration_pass,
        split_sha256=split_sha256,
    )
    for name in FAMILY_ORDER:
        if name not in active_order_tuple:
            if by_name[name].available:
                raise ValueError(
                    f"downgraded slot {name!r} must not carry an available "
                    "confirmatory result"
                )
            if by_name[name].p_value != 1.0:
                raise ValueError(
                    f"downgraded slot {name!r} must carry p_value=1.0"
                )
    row_map: dict[str, Mapping[str, object]] = {}
    if descriptive_rows is not None:
        for name, row in descriptive_rows.items():
            if name not in FAMILY_ORDER:
                raise ValueError(
                    f"descriptive row for unknown slot {name!r}"
                )
            if name in active_order_tuple:
                raise ValueError(
                    f"descriptive row for active confirmatory slot {name!r}"
                )
            row_map[name] = row
    if "H3_derecursion" in active_order_tuple:
        h3_result = by_name["H3_derecursion"]
        if h3_result.available and h3_result.effect.get(
            "budget_matched"
        ) is not True:
            raise ValueError(
                "active H3_derecursion result lacks budget_matched=True; "
                "the budget gate must be evaluated by "
                "h3_camor_derecursion"
            )
    if "H2_NI" in active_order_tuple:
        h2_result = by_name["H2_NI"]
        if h2_result.available and h2_result.boundary != DELTA_NI_MS:
            raise ValueError(
                "active H2_NI result must use the frozen boundary "
                f"{DELTA_NI_MS}, got {h2_result.boundary}"
            )
    consistency_entries: dict[str, Mapping[str, object]] = {}
    if seed_consistency is not None:
        for name, entry in seed_consistency.items():
            if name not in FAMILY_ORDER:
                raise ValueError(
                    f"seed_consistency entry for unknown slot {name!r}"
                )
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"seed_consistency[{name!r}] must be a mapping"
                )
            value = entry.get("consistent")
            if value is not None and not isinstance(value, (bool, np.bool_)):
                raise ValueError(
                    f"seed_consistency[{name!r}]['consistent'] must be bool"
                )
            consistency_entries[name] = entry

    def _slot_consistency(name: str) -> bool | None:
        result = by_name[name]
        entry = consistency_entries.get(name)
        if name in ("H1_onset", "H1_h5"):
            if entry is None:
                return None
            expected_scope = "onset" if name == "H1_onset" else "h5"
            if "winning_arm" not in entry or "scope" not in entry:
                raise ValueError(
                    f"seed_consistency[{name!r}] must be the full output "
                    "of h1_seed_sign_consistency with scope and "
                    "winning_arm; a bare boolean has no provenance"
                )
            if entry["scope"] != expected_scope:
                raise ValueError(
                    f"seed_consistency[{name!r}] scope "
                    f"{entry['scope']!r} != {expected_scope!r}"
                )
            result_arm = result.effect.get("winning_arm")
            winning_arm = entry["winning_arm"]
            if winning_arm is None:
                if entry.get("available") is not False:
                    raise ValueError(
                        f"seed_consistency[{name!r}] has no winning_arm"
                    )
            elif winning_arm != result_arm:
                raise ValueError(
                    f"seed_consistency[{name!r}] winning_arm "
                    f"{winning_arm!r} != slot winning_arm {result_arm!r}"
                )
            value = entry.get("consistent")
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(
                    f"seed_consistency[{name!r}]['consistent'] must be bool"
                )
            return bool(value)
        raw = result.effect.get("seed_sign_consistent")
        if raw is not None and not isinstance(raw, (bool, np.bool_)):
            raise ValueError(
                f"{name!r} effect['seed_sign_consistent'] must be bool "
                "or None"
            )
        computed = None if raw is None else bool(raw)
        if entry is None:
            return computed
        supplied = entry.get("consistent")
        supplied_bool = None if supplied is None else bool(supplied)
        if computed is None:
            if name == "H2_NI":
                raise ValueError(
                    "H2 seed-sign evidence must be computed by "
                    "h2_non_inferior_best_classical with the per-seed "
                    "arrays; a caller entry cannot substitute for it"
                )
            return supplied_bool
        if supplied_bool is not None and supplied_bool != computed:
            raise ValueError(
                f"seed_consistency[{name!r}]={supplied_bool} contradicts "
                f"the slot-computed evidence {computed}"
            )
        return computed

    # The confirmatory p-values are exact discrete sign-test
    # probabilities; they carry no Monte Carlo error.  Do NOT pass
    # n_boot here: doing so would treat the exact p as MC-estimated and
    # open a false-negative window around the Holm threshold.  The MC
    # guard is exercised only by callers that supply genuine MC p.
    holm = holm_confirmatory_family(
        {name: by_name[name].p_value for name in FAMILY_ORDER},
        multisensory_test_n=multisensory_test_n,
        h2_calibration_pass=h2_calibration_pass,
        split_sha256=split_sha256,
        mc_se={name: 0.0 for name in FAMILY_ORDER},
    )
    slots: dict[str, object] = {}
    for name in FAMILY_ORDER:
        result = by_name[name]
        active = name in active_order_tuple
        effect = dict(result.effect)
        positive = bool(active and result.available and holm.significant[name])
        consistency: bool | None = None
        worst_case_gate_failed = False
        if active:
            consistency = _slot_consistency(name)
            if name == "H2_NI" and result.available:
                raw_worst = result.effect.get("worst_case_p_value")
                if raw_worst is None:
                    raise ValueError(
                        "active H2_NI result lacks effect['worst_case_p_value']; "
                        "the informative-missingness gate cannot be skipped "
                        "(r9 blocker)"
                    )
                worst = float(raw_worst)
                if not (math.isfinite(worst) and 0.0 <= worst <= 1.0):
                    raise ValueError(
                        f"invalid H2 worst-case p-value: {raw_worst!r}"
                    )
                effect["worst_case_gate_pass"] = bool(
                    worst <= ALPHA / FAMILY_M
                )
                if positive and not effect["worst_case_gate_pass"]:
                    worst_case_gate_failed = True
        if not active:
            claim_level = "descriptive"
            claim_limit = "no superiority/non-inferiority/equivalence claim"
            if name in row_map:
                effect.update(dict(row_map[name]))
            for key in P_DERIVED_EFFECT_KEYS:
                effect.pop(key, None)
            effect["p_value"] = None
        elif not result.available:
            claim_level = "unavailable"
            claim_limit = (
                "slot not evaluable; this is not a confirmatory negative "
                "and supports no positive or negative claim"
            )
        elif positive and worst_case_gate_failed:
            claim_level = "indeterminate_informative_missingness"
            claim_limit = (
                "informative missingness: the analysis-set result is "
                "positive after Holm but the worst-case p over all "
                "excluded animals exceeds alpha/m; no positive claim"
            )
        elif positive and consistency is not True:
            claim_level = "indeterminate_unstable_across_seeds"
            claim_limit = (
                "at least one seed is not strictly on the improvement "
                "side or seed-sign evidence is missing; no positive "
                "confirmatory claim (protocol U11/C58)"
            )
        elif positive:
            claim_level = "confirmatory"
            claim_limit = ""
        else:
            claim_level = "confirmatory_no_rejection"
            claim_limit = (
                "evaluated and not rejected after Holm; not evidence of "
                "absence at this n"
            )
        slots[name] = {
            "statistic": result.statistic,
            "boundary": result.boundary,
            "p_value": result.p_value,
            "p_adjusted": holm.adjusted_p[name],
            "mc_se": holm.mc_se[name],
            "significant": holm.significant[name],
            "indeterminate_mc": holm.indeterminate_mc[name],
            "available": result.available,
            "reason": result.reason,
            "n_animal": result.n_animal,
            "n_boot": result.n_boot,
            "seed": result.seed,
            "effect": effect,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
            "seed_sign_consistent": consistency,
            "claim_level": claim_level,
            "descriptive_claim_limit": claim_limit,
        }
    h1_slots_ok = all(
        slots[name]["claim_level"] == "confirmatory"
        for name in ("H1_onset", "H1_h5")
    )
    return {
        "family": {
            "order": list(FAMILY_ORDER),
            "active_order": list(active_order_tuple),
            "active_order_rule": (
                "resolve_active_confirmatory_order("
                "split_multisensory_test_n, h2_calibration_pass)"
            ),
            "split_sha256": split_sha256,
            "active_order_inputs": {
                "multisensory_test_n": multisensory_test_n,
                "h2_calibration_pass": h2_calibration_pass,
                "split_sha256": split_sha256,
            },
            "fail_closed_active_order": list(FAIL_CLOSED_ACTIVE_ORDER),
            "downgraded_slots": [
                name
                for name in FAMILY_ORDER
                if name not in active_order_tuple
            ],
            "m": FAMILY_M,
            "alpha": ALPHA,
            "first_threshold": ALPHA / FAMILY_M,
            "p_value_source": "exact_one_sided_binomial_sign_test",
            "statistic_kind": "median_paired_difference",
            "mc_se_rule": "exact p -> mc_se=0; MC guard only for MC p",
            "h1_joint_gate": {
                "pass": h1_slots_ok,
                "rule": (
                    "both_H1_onset_and_H1_h5_rejected_after_holm_with_"
                    "consistent_seed_signs"
                ),
                "slots": ["H1_onset", "H1_h5"],
            },
        },
        "descriptive_rows": {
            name: dict(row) for name, row in row_map.items()
        },
        "slots": slots,
    }


def loo_fragility_index(
    results: Mapping[str, TestResult],
    p_recompute: Callable[[str, np.ndarray], float],
    *,
    multisensory_test_n: int | None,
    h2_calibration_pass: bool | None,
    max_remove: int = FRAGILITY_MAX_REMOVE,
) -> Mapping[str, SlotFragility]:
    """Descriptive LOO fragility index per confirmatory slot (no gate).

    The caller supplies ``p_recompute(slot, keep)`` which must rebuild
    the slot's inputs using only the animals whose ``keep`` entry is
    ``True`` (a boolean mask of shape ``(n_animal,)``) and return the
    slot's p-value with the same frozen method.  ``keep`` never changes
    shape; the callback must not reorder animals.

    For each active slot the index reports:

    * ``loo_flip_rate``: the fraction of single-animal deletions that
      flip the slot's frozen Holm verdict (all other slots keep their
      observed p-values);
    * ``min_animals_to_flip``: the smallest number of simultaneously
      removed animals that flips the slot verdict, searched exhaustively
      up to ``max_remove`` (``None`` and ``censored=True`` when no
      flipping subset exists within the cap).

    The index is descriptive: it does not change any pass/fail verdict,
    requires no sign-off, and downgraded slots are not reported (their
    p=1 placeholder cannot flip).  A per-slot recompute failure or
    invalid p-value fails closed for that slot only
    (``available=False``, ``reason="loo_recompute_failed ..."``).

    The active order is recomputed from the same two frozen split facts
    used by :func:`confirmatory_report`; it is not a caller argument, so
    the fragility index cannot describe a slot the pre-registration
    downgraded.

    Args:
        results: Mapping with exactly :data:`FAMILY_ORDER` keys.
        p_recompute: Callback ``(slot, keep) -> p_value``.
        multisensory_test_n: Split-fixed multisensory test animal
            count, or ``None`` before the split is frozen.
        h2_calibration_pass: One-shot H2 calibration verdict, or
            ``None`` when it has not run.
        max_remove: Search cap for the minimal flipping subset; must be
            at least 1.

    Returns:
        Mapping from active slot name to :class:`SlotFragility`.  The
        empty mapping (not an error) when no slot is active.

    Raises:
        ValueError: If the results keys or ``max_remove`` are
            invalid.
    """
    if set(results) != set(FAMILY_ORDER):
        raise ValueError(
            f"results keys must equal {FAMILY_ORDER}, got {tuple(results)}"
        )
    if (
        isinstance(max_remove, bool)
        or not isinstance(max_remove, (int, np.integer))
        or int(max_remove) < 1
    ):
        raise ValueError(
            f"max_remove must be a positive integer, got {max_remove!r}"
        )
    max_remove = int(max_remove)
    active_tuple = resolve_active_confirmatory_order(
        multisensory_test_n, h2_calibration_pass
    )
    if not active_tuple:
        # The all-downgraded family has no active slot: the index is
        # empty, not a crash.
        return {}

    baseline_p = {
        name: float(results[name].p_value) for name in FAMILY_ORDER
    }
    baseline = holm_confirmatory_family(
        baseline_p,
        multisensory_test_n=multisensory_test_n,
        h2_calibration_pass=h2_calibration_pass,
    )
    per_slot: dict[str, SlotFragility] = {}
    for slot in active_tuple:
        n_slot = int(results[slot].n_animal)
        base_significant = bool(baseline.significant[slot])
        if n_slot < 2:
            per_slot[slot] = SlotFragility(
                slot=slot,
                n_animal=n_slot,
                baseline_significant=base_significant,
                available=False,
                loo_flip_rate=None,
                n_single_deletion_flips=0,
                min_animals_to_flip=None,
                censored=True,
                max_remove=max_remove,
                reason="n_lt_2",
            )
            continue

        def verdict_with(removed: tuple[int, ...]) -> bool:
            """Holm verdict of ``slot`` with ``removed`` animals dropped."""
            keep = np.ones(n_slot, dtype=bool)
            keep[list(removed)] = False
            p_new = float(p_recompute(slot, keep))
            if not (math.isfinite(p_new) and 0.0 <= p_new <= 1.0):
                raise ValueError(
                    f"loo_recompute_invalid_p ({slot}): {p_new!r}"
                )
            candidate = dict(baseline_p)
            candidate[slot] = p_new
            holm_new = holm_confirmatory_family(
                candidate,
                multisensory_test_n=multisensory_test_n,
                h2_calibration_pass=h2_calibration_pass,
            )
            return bool(holm_new.significant[slot])

        flips = 0
        min_flip: int | None = None
        failure = ""
        try:
            for animal_index in range(n_slot):
                if verdict_with((animal_index,)) != base_significant:
                    flips += 1
            for size in range(1, max_remove + 1):
                found = False
                for subset in itertools.combinations(range(n_slot), size):
                    if verdict_with(subset) != base_significant:
                        min_flip = size
                        found = True
                        break
                if found:
                    break
        except ValueError as exc:
            failure = str(exc)
        except Exception as exc:  # fail closed on any recompute error
            failure = f"loo_recompute_failed ({slot}: {exc})"

        if failure:
            per_slot[slot] = SlotFragility(
                slot=slot,
                n_animal=n_slot,
                baseline_significant=base_significant,
                available=False,
                loo_flip_rate=None,
                n_single_deletion_flips=0,
                min_animals_to_flip=None,
                censored=True,
                max_remove=max_remove,
                reason=failure,
            )
            continue
        per_slot[slot] = SlotFragility(
            slot=slot,
            n_animal=n_slot,
            baseline_significant=base_significant,
            available=True,
            loo_flip_rate=flips / float(n_slot),
            n_single_deletion_flips=flips,
            min_animals_to_flip=min_flip,
            censored=min_flip is None,
            max_remove=max_remove,
            reason="",
        )
    return per_slot


# ---------------------------------------------------------------------------
# Pre-registered filters
# ---------------------------------------------------------------------------


def seed_sign_consistency(
    per_seed_statistics: np.ndarray,
    reference: float = 0.0,
) -> bool:
    """Return whether every seed lies on the improvement side.

    A positive confirmatory claim requires **each** training seed's
    statistic to be strictly below the slot boundary.  Agreement on the
    wrong side is not stability: the r11 review showed that three seeds
    can share the same positive median difference while their average
    looks like a strong improvement, and the old same-sign rule still
    labelled the slot ``confirmatory``.  H1 and H3 improve when
    ``statistic - 0 < 0``; H2 is non-inferior when ``statistic - 15 ms
    < 0``.  A statistic equal to the boundary is a tie and counts as a
    non-improvement (conservative).

    Args:
        per_seed_statistics: Finite ``(n_seed,)`` slot statistic per
            training seed.
        reference: Slot boundary; every seed must satisfy
            ``statistic < reference``.

    Returns:
        ``True`` only when all values are strictly below ``reference``.
        An all-zero vector at ``reference=0`` is not an improvement and
        returns ``False``; any seed on or above the boundary returns
        ``False``.
    """
    values = _as_1d(per_seed_statistics, "per_seed_statistics")
    if values.size < 2:
        raise ValueError("need at least two seed statistics")
    reference = _check_finite_scalar(reference, "reference")
    return bool(np.all(values < reference))


def g1_existence_permutation(
    n_response: np.ndarray,
    latency_mean: np.ndarray,
    n_perm: int = 20_000,
    seed: int = SEED_BASE,
    alpha: float = 0.05,
    min_abs_rho: float = 0.3,
) -> G1Result:
    """Animal-level permutation test for the G1 existence criterion.

    ``rho`` is the animal_key-level Spearman correlation between the
    mean response count and the mean latency (one point per animal).
    The null is ``rho = 0``; the p-value is two-sided and uses label
    permutation of the latency values,
    ``p = (1 + #{|rho_perm| >= |rho_obs|}) / (n_perm + 1)``.  The
    median-centred bootstrap p is forbidden.  Constant ranks make rho
    undefined; the result is returned as not evaluable instead of
    crashing the harness.

    Args:
        n_response: Finite ``(n_animal,)`` mean response counts.
        latency_mean: Finite ``(n_animal,)`` mean latencies.
        n_perm: Permutation draws.
        seed: Random seed.
        alpha: Two-sided significance level.
        min_abs_rho: Minimum absolute observed rho for ``exists``.

    Returns:
        :class:`G1Result` with ``exists = p <= alpha and |rho| >=
        min_abs_rho``.  ``available=False`` for constant ranks.

    Raises:
        ValueError: On shape mismatch, non-finite values, fewer than
            three animals, or invalid parameters.
    """
    x = _as_1d(n_response, "n_response")
    y = _as_1d(latency_mean, "latency_mean")
    if x.shape != y.shape:
        raise ValueError(
            f"n_response shape {x.shape} != latency_mean shape {y.shape}"
        )
    if x.size < 3:
        raise ValueError("G1 permutation needs at least three animals")
    n_perm = _check_n_boot(n_perm)
    seed = _check_seed(seed)
    alpha = _check_finite_scalar(alpha, "alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    min_abs_rho = _check_finite_scalar(min_abs_rho, "min_abs_rho")
    rank_x = stats.rankdata(x)
    rank_y = stats.rankdata(y)
    centered_x = rank_x - rank_x.mean()
    centered_y = rank_y - rank_y.mean()
    norm_x = float(np.sqrt(np.sum(centered_x ** 2)))
    norm_y = float(np.sqrt(np.sum(centered_y ** 2)))
    if norm_x == 0.0 or norm_y == 0.0:
        return G1Result(
            rho_observed=float("nan"),
            p_value=1.0,
            n_animal=int(x.size),
            n_perm=n_perm,
            seed=seed,
            exists=False,
            available=False,
            reason="not_evaluable (constant ranks)",
        )
    rho_obs = float(np.sum(centered_x * centered_y) / (norm_x * norm_y))
    rng = np.random.default_rng(seed)
    perm_ranks = rng.permuted(
        np.tile(rank_y, (n_perm, 1)), axis=1
    )
    assert perm_ranks.shape == (n_perm, rank_y.size)
    centered_perm = perm_ranks - rank_y.mean()
    numer = centered_perm @ centered_x
    denom = norm_x * norm_y
    rho_perm = numer / denom
    assert rho_perm.shape == (n_perm,)
    p_value = (
        1.0
        + float(np.count_nonzero(np.abs(rho_perm) >= abs(rho_obs)))
    ) / (n_perm + 1.0)
    exists = bool(p_value <= alpha and abs(rho_obs) >= min_abs_rho)
    return G1Result(
        rho_observed=rho_obs,
        p_value=p_value,
        n_animal=int(x.size),
        n_perm=n_perm,
        seed=seed,
        exists=exists,
        available=True,
        reason="" if exists else "not_detected",
    )


# ---------------------------------------------------------------------------
# Descriptive statistics (no p-values, no Holm, no multiplicity control)
# ---------------------------------------------------------------------------


def animal_cluster_bootstrap_ci(
    values: np.ndarray,
    n_boot: int = CI_BOOT,
    ci_level: float = CI_LEVEL,
    seed: int = SEED_BASE,
) -> tuple[float, float, float]:
    """Animal-cluster percentile bootstrap CI for a one-sample statistic.

    This is the only interval used by descriptive Phase-3 items: no
    p-value, no Holm, no significance wording, and no multiplicity
    control.

    Args:
        values: Finite ``(n,)`` per-animal observations.
        n_boot: Bootstrap draws (protocol default 20,000).
        ci_level: Confidence level (protocol default 0.95).
        seed: Random seed.

    Returns:
        ``(point_estimate, ci_lower, ci_upper)``.

    Raises:
        ValueError: If fewer than two finite observations are supplied.
    """
    array = _as_1d(values, "values")
    if array.size < 2:
        raise ValueError("descriptive CI needs at least two animals")
    n_boot = _check_n_boot(n_boot)
    seed = _check_seed(seed)
    ci_level = _check_finite_scalar(ci_level, "ci_level")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must lie strictly between 0 and 1")
    point, low, high = bootstrap_ci(
        array,
        statistic_fn=np.mean,
        n_bootstrap=n_boot,
        ci_level=ci_level,
        seed=seed,
        method="percentile",
    )
    return float(point), float(low), float(high)


def animal_cluster_bootstrap_ci_two_sample(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_boot: int = CI_BOOT,
    ci_level: float = CI_LEVEL,
    seed: int = SEED_BASE,
) -> tuple[float, float, float]:
    """Two-sample animal-cluster percentile CI for a mean difference.

    Each group is resampled with replacement while preserving its own
    size; the two groups are never pooled or relabelled.

    Args:
        values_a: Finite ``(n_a,)`` per-animal observations for group A.
        values_b: Finite ``(n_b,)`` per-animal observations for group B.
        n_boot: Bootstrap draws.
        ci_level: Confidence level.
        seed: Random seed.

    Returns:
        ``(mean_a - mean_b, ci_lower, ci_upper)``.

    Raises:
        ValueError: If either group has fewer than two observations.
    """
    group_a = _as_1d(values_a, "values_a")
    group_b = _as_1d(values_b, "values_b")
    if group_a.size < 2 or group_b.size < 2:
        raise ValueError("each group needs at least two animals")
    n_boot = _check_n_boot(n_boot)
    seed = _check_seed(seed)
    ci_level = _check_finite_scalar(ci_level, "ci_level")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must lie strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, group_a.size, size=(n_boot, group_a.size))
    idx_b = rng.integers(0, group_b.size, size=(n_boot, group_b.size))
    assert idx_a.shape == (n_boot, group_a.size)
    assert idx_b.shape == (n_boot, group_b.size)
    boot = group_a[idx_a].mean(axis=1) - group_b[idx_b].mean(axis=1)
    assert boot.shape == (n_boot,)
    point = float(group_a.mean() - group_b.mean())
    low, high = _percentile_ci(boot, ci_level=ci_level)
    if low is None or high is None:
        raise ValueError("non-finite bootstrap distribution")
    return point, low, high


def cohens_dz(values: np.ndarray) -> float:
    """Paired Cohen's d_z (mean / sample SD); zero when SD is zero."""
    return _cohens_dz_1d(_as_1d(values, "values"))


def cliffs_delta(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Cliff's delta for two independent samples.

    Positive values mean group A tends to exceed group B.

    Args:
        values_a: Finite ``(n_a,)`` observations for group A.
        values_b: Finite ``(n_b,)`` observations for group B.

    Returns:
        Cliff's delta in ``[-1, 1]``.

    Raises:
        ValueError: If either sample is empty.
    """
    group_a = _as_1d(values_a, "values_a")
    group_b = _as_1d(values_b, "values_b")
    differences = group_a[:, None] - group_b[None, :]
    assert differences.shape == (group_a.size, group_b.size)
    greater = float(np.count_nonzero(differences > 0.0))
    less = float(np.count_nonzero(differences < 0.0))
    return (greater - less) / (group_a.size * group_b.size)


def descriptive_summary(
    point: float | None,
    ci_low: float | None,
    ci_high: float | None,
    n_animal: int,
    n_trial: int,
    effect_size: float | None,
    reason: str = "",
) -> dict[str, object]:
    """Build the standard descriptive report row (never a p-value).

    The row explicitly declares that there is **no multiplicity
    control** and that the interval cannot support a superiority,
    non-inferiority, or equivalence claim for a slot that did not pass
    its confirmatory gate.

    Args:
        point: Point estimate, or ``None`` when unavailable.
        ci_low: Lower 95% animal-cluster bootstrap bound.
        ci_high: Upper 95% animal-cluster bootstrap bound.
        n_animal: Number of animals.
        n_trial: Number of trials.
        effect_size: Cohen's d_z or Cliff's delta.
        reason: Availability reason.

    Returns:
        JSON-serialisable dictionary with ``inference="descriptive"``,
        ``p_value=None``, and ``multiplicity_control="none"``.

    Raises:
        ValueError: On negative counts or inconsistent availability.
    """
    if n_animal < 0 or n_trial < 0:
        raise ValueError("n_animal and n_trial must be non-negative")
    if point is None:
        if ci_low is not None or ci_high is not None:
            raise ValueError("unavailable rows must not carry CI bounds")
    else:
        if not math.isfinite(float(point)):
            raise ValueError("point must be finite or None")
        if ci_low is None or ci_high is None:
            raise ValueError("available rows must carry both CI bounds")
        if not math.isfinite(float(ci_low)) or not math.isfinite(
            float(ci_high)
        ):
            raise ValueError("CI bounds must be finite")
        if float(ci_low) > float(ci_high):
            raise ValueError("ci_low must not exceed ci_high")
    return {
        "inference": "descriptive",
        "p_value": None,
        "multiplicity_control": "none",
        "claim_limit": "no superiority/non-inferiority/equivalence claim",
        "point": None if point is None else float(point),
        "ci_low": None if ci_low is None else float(ci_low),
        "ci_high": None if ci_high is None else float(ci_high),
        "n_animal": int(n_animal),
        "n_trial": int(n_trial),
        "effect_size": None if effect_size is None else float(effect_size),
        "available": point is not None,
        "reason": reason,
    }


__all__ = [
    "ALPHA",
    "AnalysisSet",
    "B_PRIMARY",
    "CI_BOOT",
    "CI_LEVEL",
    "DELTA_NI_MS",
    "FAIL_CLOSED_ACTIVE_ORDER",
    "FAMILY_M",
    "FAMILY_ORDER",
    "FRAGILITY_MAX_REMOVE",
    "G1Result",
    "H1_FOUNDATION_ARMS",
    "H1_SCOPES",
    "H2_CALIBRATION_R",
    "H2_CALIBRATION_SEED",
    "H2_MIN_TEST_N",
    "H2_MODELS",
    "HolmResult",
    "N_MIN",
    "N_PLAN_H1_H3",
    "N_PLAN_H2",
    "N_PLAN_H2_RANGE",
    "P_DERIVED_EFFECT_KEYS",
    "SEED_BASE",
    "SLOT_BOUNDARIES",
    "SlotFragility",
    "TIE_POLICY",
    "TYPE1_CP95_GATE",
    "TestResult",
    "animal_cluster_bootstrap_ci",
    "animal_cluster_bootstrap_ci_two_sample",
    "animal_cluster_bootstrap_shift",
    "build_h2_analysis_set",
    "cliffs_delta",
    "cohens_dz",
    "confirmatory_report",
    "descriptive_placeholder",
    "descriptive_summary",
    "g1_existence_permutation",
    "h1_foundation_persistence",
    "h2_non_inferior_best_classical",
    "h2_seed_errors",
    "h3_camor_derecursion",
    "holm_confirmatory_family",
    "loo_fragility_index",
    "resolve_active_confirmatory_order",
    "seed_sign_consistency",
]
