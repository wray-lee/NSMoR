"""Frozen-family paired comparison statistics for NSMoR candidate scoring.

Engineering descriptive validation only. This module implements the
comparison statistics fixed by ``docs/model-optimization-protocol-20261004.md``
*before* any candidate training: paired per-trial MSE deltas, trial-level
paired Cohen's ``d_z``, a recording-prefix cluster statistic, a two-sided
entire-prefix sign-flip permutation diagnostic, and Holm-Bonferroni
adjustment over the fixed 78-slot declared family.

Scope: the input is per-trial MSE values already computed on identical
eligible frames for candidate and comparator. Frame extraction, masking,
and metric definitions live in the evaluation scripts; this module is a
pure, stateless, side-effect-free reducer so the same numbers can be
reproduced and audited independently.

Conventions
-----------
- ``delta = candidate_mse - comparator_mse`` per trial. Negative is
  better for the candidate.
- Prefix means are unweighted across the trials of each recording prefix;
  the cluster statistic is the unweighted mean of those prefix means.
- Sign flips act on entire prefix means, never on individual frames or
  trials.
- Undefined statistics are reported as ``None`` with an explicit
  ``reason``; they are never coerced to a finite placeholder.

Nothing here is a biological, animal-generalization, or global-optimum
claim. ``animal_identity_status`` is unverified and recording prefixes are
not animals.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import combinations
from numbers import Real
from typing import Any

import numpy as np

from nsmor.analysis.uq import cohens_d, holm_bonferroni

__all__ = [
    "ALIGNED_OVERALL_SCOPE",
    "CANDIDATE_IDS",
    "COMPARATOR_IDS",
    "DECLARED_FAMILY_SIZE",
    "SCOPE_IDS",
    "comparison_to_json",
    "declared_family_slots",
    "holm_correct_declared_family",
    "paired_mse_comparison",
]

#: Sustained high-velocity sensitivity thresholds (cm/s) from the protocol.
SENSITIVITY_THRESHOLDS_CM_S: tuple[int, ...] = (5, 10, 20, 50)
#: Sustained minimum run lengths from the protocol.
SUSTAINED_MIN_RUN_LENGTHS: tuple[int, ...] = (1, 2, 3)
#: Aligned overall scope plus the 12 sustained high-velocity cells = 13.
ALIGNED_OVERALL_SCOPE: str = "aligned_overall"
#: The two fresh candidates declared before training.
CANDIDATE_IDS: tuple[str, ...] = ("k_0.5", "k_1.0")
#: Comparators per candidate: verified head-only, persistence, zero.
COMPARATOR_IDS: tuple[str, ...] = (
    "head_only_baseline",
    "persistence",
    "zero",
)
SCOPE_IDS: tuple[str, ...] = (ALIGNED_OVERALL_SCOPE,) + tuple(
    f"band{band}_run{run}"
    for band in SENSITIVITY_THRESHOLDS_CM_S
    for run in SUSTAINED_MIN_RUN_LENGTHS
)
#: Declared family size: 2 candidates * 3 comparators * 13 scopes.
DECLARED_FAMILY_SIZE: int = (
    len(CANDIDATE_IDS) * len(COMPARATOR_IDS) * len(SCOPE_IDS)
)
assert DECLARED_FAMILY_SIZE == 78, "Declared family must stay frozen at 78"
assert len(SCOPE_IDS) == 13, "Declared scope count must stay frozen at 13"

#: Minimum eligible prefixes required before any permutation p-value.
_MIN_PREFIXES = 6
#: Exact enumeration is used for at most this many prefixes (2**16 = 65536).
_MAX_EXACT_PREFIXES = 16
#: Monte Carlo sign-pattern count above the exact-enumeration threshold.
_MC_PATTERNS = 100000
#: Sign patterns drawn per chunk so the MC branch never allocates the full
#: ``_MC_PATTERNS * n_prefix`` matrix at once.
_MC_CHUNK = 4096
#: Hard ceiling on the number of matrix cells (``rows * n_prefix``) any single
#: Monte Carlo draw may allocate. Above this a finite prefix count is refused
#: explicitly rather than allocating an unbounded matrix.
_MC_MAX_CELLS = 262144
#: Fixed permutation seed required by the protocol.
_DEFAULT_SEED = 42


def declared_family_slots(
    candidate_ids: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Return the frozen 78 comparison slots in a deterministic order.

    Args:
        candidate_ids: Candidate ids to build slots for. Defaults to the
            module-level :data:`CANDIDATE_IDS` (the k-family). The real-data
            preliminary protocol passes ``("A1", "A2")`` so the SAME frozen
            slot algebra and comparator/scope grid is reused verbatim instead
            of being re-implemented downstream. Any candidate set of length
            ``len(CANDIDATE_IDS)`` yields the frozen family size (78).

    Returns:
        Tuple of ``"<candidate>|<comparator>|<scope>"`` slot keys, length
        :data:`DECLARED_FAMILY_SIZE` (78).
    """
    cands = CANDIDATE_IDS if candidate_ids is None else tuple(candidate_ids)
    assert len(cands) == len(CANDIDATE_IDS), (
        f"Declared family requires {len(CANDIDATE_IDS)} candidates, "
        f"got {len(cands)}"
    )
    slots = tuple(
        f"{candidate}|{comparator}|{scope}"
        for candidate in cands
        for comparator in COMPARATOR_IDS
        for scope in SCOPE_IDS
    )
    assert len(slots) == DECLARED_FAMILY_SIZE
    assert len(set(slots)) == DECLARED_FAMILY_SIZE, "Slot keys must be unique"
    return slots


def _finite_vector(
    values: Sequence[float] | np.ndarray,
    name: str,
    *,
    require_nonnegative: bool = False,
) -> np.ndarray:
    """Coerce a 1-D finite float64 array or fail closed.

    Args:
        values: Sequence or array of per-trial values.
        name: Field name used in error messages.
        require_nonnegative: Reject any value below zero (MSE semantics).

    Returns:
        Finite 1-D ``float64`` array.

    Raises:
        ValueError: If empty, not 1-D, contains non-finite entries, or
            (when required) any value is negative.
    """
    arr = np.asarray(values, dtype=np.float64)
    assert arr.ndim == 1, f"{name} must be 1-D, got shape {arr.shape}"
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if require_nonnegative and np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative (MSE semantics)")
    return arr


def _require_bool(value: Any, name: str) -> bool:
    """Return a genuine boolean or fail closed.

    Rejects strings (``"false"`` is truthy in Python), integers, and every
    other non-boolean type, so a permissive truth conversion can never
    silently assert an exchangeability null that the caller did not.

    Args:
        value: Candidate flag.
        name: Parameter name used in error messages.

    Returns:
        The flag as a Python ``bool``.

    Raises:
        ValueError: If ``value`` is not a ``bool`` (or NumPy bool).
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(
        f"{name} must be a genuine bool, got {type(value).__name__}: {value!r}"
    )


def _require_protocol_seed(value: Any) -> int:
    """Return the protocol-fixed Monte Carlo seed or fail closed.

    The permutation seed is fixed at :data:`_DEFAULT_SEED` by the frozen
    protocol; a caller may not silently substitute its own seed.

    Args:
        value: Candidate seed.

    Returns:
        The integer :data:`_DEFAULT_SEED`.

    Raises:
        ValueError: If ``value`` is not an integer equal to
            :data:`_DEFAULT_SEED` (bools and floats are rejected).
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(
            f"seed must be the integer {_DEFAULT_SEED}, got "
            f"{type(value).__name__}: {value!r}"
        )
    if int(value) != _DEFAULT_SEED:
        raise ValueError(
            f"seed must be {_DEFAULT_SEED} (protocol-fixed), got {int(value)}"
        )
    return _DEFAULT_SEED


def _id_vector(
    values: Sequence[str], name: str, *, require_unique: bool
) -> list[str]:
    """Validate a 1-D vector of non-empty source/trial identifiers.

    Args:
        values: Sequence of identifier strings.
        name: Field name used in error messages.
        require_unique: Whether identifiers must be pairwise unique.

    Returns:
        List of validated identifiers.

    Raises:
        ValueError: If empty, non-string, blank, or (when required) not
            unique.
    """
    ids = list(values)
    if not ids:
        raise ValueError(f"{name} must not be empty")
    for index, value in enumerate(ids):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{name}[{index}] must be a non-empty string, got {value!r}"
            )
    if require_unique and len(set(ids)) != len(ids):
        raise ValueError(f"{name} must be pairwise unique")
    return ids


def _frame_counts(
    values: Sequence[int] | np.ndarray, n_trials: int
) -> np.ndarray:
    """Validate per-trial eligible frame counts as positive integers.

    Args:
        values: Per-trial frame counts.
        n_trials: Expected number of trials.

    Returns:
        Integer 1-D array of frame counts.

    Raises:
        ValueError: If not integral, wrong length, or non-positive.
    """
    arr = np.asarray(values)
    assert arr.ndim == 1, f"frame_counts must be 1-D, got shape {arr.shape}"
    if arr.dtype.kind not in "iu":
        raise ValueError("frame_counts must be integral")
    if arr.size != n_trials:
        raise ValueError(
            f"frame_counts length {arr.size} != trial count {n_trials}"
        )
    if np.any(arr < 1):
        raise ValueError("frame_counts must be >= 1 per trial")
    return arr


def _sqrt_fraction(value: Fraction) -> float:
    """Scale-safe float square root of a non-negative :class:`Fraction`.

    The argument is scaled up by a perfect square before taking integer square
    roots, so the result carries far more than the 53 bits float64 keeps and
    neither the numerator nor the denominator ever has to be converted to a
    float first (which would overflow or underflow for extreme magnitudes).

    Args:
        value: Non-negative :class:`fractions.Fraction`.

    Returns:
        ``sqrt(value)`` rounded to float64.
    """
    numerator = value.numerator
    denominator = value.denominator
    if numerator == 0:
        return 0.0
    bits = min(numerator.bit_length(), denominator.bit_length())
    shift = max(0, 128 - bits)
    if shift % 2:
        shift += 1
    return math.isqrt(numerator << shift) / math.isqrt(denominator << shift)


def _exact_dz(delta: np.ndarray) -> tuple[float | None, str | None]:
    """Compute trial-level paired Cohen's ``d_z`` in exact arithmetic.

    The paired statistic ``mean(delta) / std(delta, ddof=1)`` is evaluated from
    the represented per-trial deltas as exact rationals. Both the numerator
    mean and the sample variance come from the same exact reductions, so the
    effect size is order-invariant: pairwise float summation of a cancelling
    delta (e.g. ``[1, 1e16, -1e16, -0.5]``) silently loses the small addends
    and can even flip the sign of ``d_z``. Forming the ratio as
    ``sign(mean) * sqrt(mean**2 / variance)`` keeps it representable for
    extreme finite inputs (``[1e300, 2e300]`` and ``[1e-200, 2e-200]`` both
    give ``3/sqrt(2)``) without a float overflow or underflow in any step.

    Args:
        delta: Finite 1-D float64 array of per-trial candidate-minus-comparator
            deltas.

    Returns:
        ``(value, reason)``; exactly one is ``None``. ``reason`` is one of
        ``"fewer_than_two_trials"``, ``"zero_delta_variance"`` or
        ``"non_finite_effect_size"`` when the value is undefined.
    """
    n_trials = int(delta.size)
    if n_trials < 2:
        return None, "fewer_than_two_trials"
    if not np.all(np.isfinite(delta)):
        return None, "non_finite_effect_size"
    values = [Fraction.from_float(float(value)) for value in delta]
    mean = sum(values, Fraction(0)) / n_trials
    variance = (
        sum(((value - mean) ** 2 for value in values), Fraction(0))
        / (n_trials - 1)
    )
    if variance == 0:
        return None, "zero_delta_variance"
    if mean == 0:
        return 0.0, None
    magnitude = _sqrt_fraction(mean * mean / variance)
    value = -magnitude if mean < 0 else magnitude
    if not math.isfinite(value):
        return None, "non_finite_effect_size"
    return value, None


def _as_exact_ints(values: np.ndarray) -> list[int]:
    """Convert finite float64 prefix means to common-denominator integers.

    The exact sign-flip comparison ``|sum(signs * v)| >= |sum(v)|`` is decided
    in integer arithmetic, so it is exact at mathematical tie boundaries
    instead of relying on a floating-point tolerance. Each value is scaled by
    its exact binary denominator (via :class:`fractions.Fraction`), and the
    common denominator is factored out of the comparison.

    Args:
        values: Finite 1-D float64 array of prefix means.

    Returns:
        Integer list ``num`` with ``values[i] == num[i] / D`` for a single
        positive denominator ``D``; since ``D`` is common it cancels from the
        comparison.
    """
    scaled: list[Fraction] = []
    for value in values:
        f = Fraction.from_float(float(value))
        scaled.append(f)
    denominator = 1
    for f in scaled:
        denominator = denominator * f.denominator // math.gcd(
            denominator, f.denominator
        )
    return [int(f.numerator * (denominator // f.denominator)) for f in scaled]


def _exact_count(nums: list[int]) -> int:
    """Count sign patterns whose statistic reaches the observed magnitude.

    A sign pattern contributes ``sum(signs_i * num_i) = total - 2 * sum_S``
    where ``S`` ranges over the subsets of flipped indices, so enumerating all
    ``2**n_prefix`` subset sums decides ``|stat| >= |total|`` in exact integer
    arithmetic with no floating-point tolerance.

    Args:
        nums: Integer prefix means (common denominator cancelled).

    Returns:
        Number of the ``2**len(nums)`` sign patterns with
        ``|sum| >= |sum(nums)|``.
    """
    total = sum(nums)
    threshold = abs(total)
    sums = [0]
    for value in nums:
        sums += [s + value for s in sums]
    return sum(1 for s in sums if abs(total - 2 * s) >= threshold)


def _exact_mean(values: Sequence[float] | np.ndarray) -> float:
    """Order-invariant exact mean of finite float64 values.

    A plain ``np.mean`` accumulates in float64, so summation order changes the
    result: ``[1e16, 1, -1e16]`` averages to ``0.0`` while
    ``[1e16, -1e16, 1]`` averages to ``1/3``. Every float64 is dyadic, so the
    sum is accumulated exactly as a :class:`fractions.Fraction` and rounded
    once. The represented mean is then the correctly rounded mathematical mean
    and does not depend on trial order, scale, or catastrophic cancellation.

    For finite float64 inputs the exact mean is bounded in magnitude by the
    largest absolute input, so it is always representable; the non-finite guard
    below is defensive and unreachable for valid input.

    Args:
        values: Non-empty sequence of finite float64 values.

    Returns:
        The mean, rounded once to float64.

    Raises:
        ValueError: If ``values`` is empty, contains a non-finite entry, or
            the exact mean is not representable in float64.
    """
    floats = [float(value) for value in values]
    if not floats:
        raise ValueError("mean requires at least one value")
    total = Fraction(0)
    for value in floats:
        if not math.isfinite(value):
            raise ValueError("mean inputs must be finite")
        total += Fraction.from_float(value)
    result = float(total / len(floats))
    if not math.isfinite(result):
        raise ValueError("exact mean is not representable in float64")
    return result


def _sign_flip_cluster_p(
    prefix_means: Mapping[str, float],
    *,
    exchangeability_asserted: bool,
    seed: int,
) -> dict[str, Any]:
    """Two-sided entire-prefix sign-flip p-value for the cluster statistic.

    Sign flips are applied to whole prefix means, and the statistic is
    computed on the max-abs normalized prefix means so it is scale
    invariant. Exact enumeration is used for at most
    :data:`_MAX_EXACT_PREFIXES` prefixes; above that a seed-42 Monte Carlo
    draw of :data:`_MC_PATTERNS` sign patterns with a plus-one correction is
    used. Each pattern is drawn from its own ordinal-indexed stream, so the
    result does not depend on the chunking used to bound memory.

    Args:
        prefix_means: Mapping of recording prefix to its mean trial delta.
        exchangeability_asserted: Genuine ``bool`` asserting the joint
            sign-exchangeability null. When ``False`` the p-value is
            ``None``.
        seed: Monte Carlo seed; must be the protocol-fixed value.

    Returns:
        JSON-safe dict with ``p_value``, ``reason``, ``method``,
        ``n_prefixes``, ``n_sign_patterns``, ``exchangeability_asserted``
        and ``seed``.

    Raises:
        ValueError: If a prefix mean is non-finite, if
            ``exchangeability_asserted`` is not a genuine bool, if ``seed``
            is not the protocol-fixed integer, if the prefix count exceeds
            the Monte Carlo cell budget, or if a permutation statistic is
            not representable in float64.
    """
    asserted = _require_bool(exchangeability_asserted, "exchangeability_asserted")
    seed = _require_protocol_seed(seed)
    values = np.asarray(list(prefix_means.values()), dtype=np.float64)
    assert values.ndim == 1
    if not np.all(np.isfinite(values)):
        raise ValueError("prefix means must be finite")
    n_prefix = int(values.size)
    base: dict[str, Any] = {
        "exchangeability_asserted": asserted,
        "seed": seed,
        "n_prefixes": n_prefix,
    }
    if n_prefix < _MIN_PREFIXES:
        return {
            **base,
            "p_value": None,
            "reason": "fewer_than_six_prefixes",
            "method": None,
            "n_sign_patterns": None,
        }
    if not asserted:
        return {
            **base,
            "p_value": None,
            "reason": "exchangeability_not_asserted",
            "method": None,
            "n_sign_patterns": None,
        }
    # Exact integer representation of the prefix means. The statistic is
    # compared at mathematical tie boundaries, so a tolerance is never used
    # to decide the exact path.
    ints = _as_exact_ints(values)
    total = sum(ints)
    # Exact branch: enumerate every one of the 2**n_prefix sign patterns in
    # integer arithmetic, so ties are decided exactly (no tolerance).
    if n_prefix <= _MAX_EXACT_PREFIXES:
        n_patterns = 1 << n_prefix
        count = _exact_count(ints)
        return {
            **base,
            "p_value": count / float(n_patterns),
            "reason": None,
            "method": "exact",
            "n_sign_patterns": n_patterns,
        }
    # Monte Carlo branch. The cell budget bounds the matrix, and each pattern
    # is drawn from its OWN seed-42 stream indexed by the pattern ordinal, so
    # the result is invariant to how the patterns are chunked.
    if n_prefix > _MC_MAX_CELLS:
        raise ValueError(
            f"n_prefix={n_prefix} exceeds the Monte Carlo cell budget "
            f"({_MC_MAX_CELLS}); refusing an unbounded permutation draw"
        )
    # For prefix counts above the exact limit the observed statistic is still
    # computed in float64; any pattern whose statistic lands within round-off
    # of the boundary is re-decided per pattern in exact integer arithmetic.
    chunk_rows = max(1, min(_MC_CHUNK, _MC_MAX_CELLS // n_prefix))
    max_abs = float(np.max(np.abs(values)))
    if max_abs == 0.0:
        # Every prefix mean is exactly zero: the observed statistic is zero
        # and every sign pattern matches it, so p = 1.
        count = _MC_PATTERNS
    else:
        normalized = values / max_abs
        observed = abs(_exact_mean(normalized))
        tol = 8.0 * n_prefix * float(np.finfo(np.float64).eps)
        count = 0
        pattern = 0
        while pattern < _MC_PATTERNS:
            rows = min(chunk_rows, _MC_PATTERNS - pattern)
            bits = np.empty((rows, n_prefix), dtype=np.int8)
            for r in range(rows):
                bits[r] = np.random.default_rng(
                    [seed, pattern + r]
                ).integers(0, 2, size=n_prefix, dtype=np.int8)
            stats = np.abs((bits * 2.0 - 1.0) @ normalized / n_prefix)
            near = np.abs(stats - observed) <= tol
            count += int(np.count_nonzero(~near & (stats > observed)))
            for r in np.nonzero(near)[0]:
                signed = [int(bits[r, i]) * 2 - 1 for i in range(n_prefix)]
                s = sum(a * b for a, b in zip(signed, ints))
                count += int(abs(s) >= abs(total))
            pattern += rows
    return {
        **base,
        "p_value": (count + 1) / (_MC_PATTERNS + 1),
        "reason": None,
        "method": "monte_carlo",
        "n_sign_patterns": _MC_PATTERNS,
    }


def paired_mse_comparison(
    *,
    candidate_mse: Sequence[float] | np.ndarray,
    comparator_mse: Sequence[float] | np.ndarray,
    trial_ids: Sequence[str],
    prefix_ids: Sequence[str],
    frame_counts: Sequence[int] | np.ndarray,
    exchangeability_asserted: bool = False,
    seed: int = _DEFAULT_SEED,
) -> dict[str, Any]:
    """Reduce paired per-trial MSE into descriptive comparison statistics.

    All trials must be scored on identical eligible frames per condition;
    only the per-trial MSE values and their provenance enter here.

    Args:
        candidate_mse: Finite per-trial candidate MSE, shape ``(N,)``.
        comparator_mse: Finite per-trial comparator MSE, shape ``(N,)``.
        trial_ids: Unique non-empty trial identifiers, length ``N``.
        prefix_ids: Non-empty recording-prefix identifiers, length ``N``.
        frame_counts: Integral per-trial eligible frame counts, length
            ``N``, each ``>= 1``.
        exchangeability_asserted: Genuine ``bool`` asserting the joint
            sign-exchangeability null for the permutation diagnostic.
            Default ``False`` yields a null p-value with an explicit
            reason. A non-bool (e.g. the truthy string ``"false"``) is
            rejected, never coerced.
        seed: Monte Carlo seed; must be the protocol-fixed integer 42.
            Any other value (including bools and floats) is rejected rather
            than silently substituted.

    Returns:
        JSON-safe dict with ``delta_convention``, ``counts``,
        ``mean_delta``, ``cohens_dz``, ``prefix_means``,
        ``cluster_statistic`` and ``sign_flip``.

    Raises:
        ValueError: If inputs are empty, mismatched, non-finite, negative
            (MSE semantics), non-unique where required, if frame counts are
            non-positive, if ``exchangeability_asserted`` is not a genuine
            bool, if ``seed`` deviates from the protocol value, or if any
            derived statistic is not representable in float64.
    """
    candidate = _finite_vector(
        candidate_mse, "candidate_mse", require_nonnegative=True
    )
    comparator = _finite_vector(
        comparator_mse, "comparator_mse", require_nonnegative=True
    )
    n_trials = int(candidate.size)
    assert comparator.shape == (n_trials,), (
        f"comparator_mse shape {comparator.shape} != candidate "
        f"shape {candidate.shape}"
    )
    trials = _id_vector(trial_ids, "trial_ids", require_unique=True)
    prefixes = _id_vector(prefix_ids, "prefix_ids", require_unique=False)
    frames = _frame_counts(frame_counts, n_trials)
    if len(trials) != n_trials:
        raise ValueError(
            f"trial_ids length {len(trials)} != trial count {n_trials}"
        )
    if len(prefixes) != n_trials:
        raise ValueError(
            f"prefix_ids length {len(prefixes)} != trial count {n_trials}"
        )

    delta = candidate - comparator
    if not np.all(np.isfinite(delta)):
        raise ValueError("trial MSE delta is not representable in float64")
    # Reduce exactly and round once: pairwise float summation is order
    # dependent (e.g. [1e16, 1, -1e16] loses the 1), which would make the
    # prefix means, the cluster statistic, and therefore the sign-flip
    # p-value depend on trial order. Exact rational accumulation removes
    # that artifact while keeping the represented float64 contract.
    grouped: dict[str, list[float]] = {}
    for prefix, value in zip(prefixes, delta):
        grouped.setdefault(prefix, []).append(float(value))
    prefix_means = {
        prefix: _exact_mean(grouped[prefix]) for prefix in sorted(grouped)
    }
    cluster = _exact_mean(list(prefix_means.values()))
    mean_delta = _exact_mean(delta)
    dz_value, dz_reason = _exact_dz(delta)
    # Sum as Python ints so a large frame total cannot overflow a NumPy
    # fixed-width accumulator.
    total_frames = sum(int(count) for count in frames)
    return {
        "delta_convention": "candidate_minus_comparator_mse",
        "counts": {
            "trials": n_trials,
            "frames": total_frames,
            "prefixes": len(prefix_means),
        },
        "mean_delta": mean_delta,
        "cohens_dz": {"value": dz_value, "reason": dz_reason},
        "prefix_means": prefix_means,
        "cluster_statistic": cluster,
        "sign_flip": _sign_flip_cluster_p(
            prefix_means,
            exchangeability_asserted=exchangeability_asserted,
            seed=seed,
        ),
    }


def holm_correct_declared_family(
    p_values: Mapping[str, float | None],
    *,
    candidate_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Holm-adjust the frozen 78-slot family, preserving unavailable slots.

    Unavailable slots (``None``) are treated as ``p=1`` for adjustment
    bookkeeping but reported as ``None``. Slots may not be silently dropped
    or added.

    Args:
        p_values: Mapping from :func:`declared_family_slots` keys to a
            finite p-value in ``[0, 1]`` or ``None`` when unavailable.
        candidate_ids: Candidate ids the family was declared for; forwarded to
            :func:`declared_family_slots`. Defaults to the k-family.

    Returns:
        JSON-safe dict with ``family_size``, ``valid_test_count``,
        ``unavailable_slot_count`` and per-slot ``slots`` entries of
        ``available``/``adjusted_p``/``significant``.

    Raises:
        ValueError: If any declared slot is missing, any undeclared slot is
            present, or a supplied p-value is not finite in ``[0, 1]``.
    """
    declared = declared_family_slots(candidate_ids)
    declared_set = set(declared)
    provided = set(p_values)
    missing = declared_set - provided
    extra = provided - declared_set
    if missing:
        raise ValueError(
            f"Declared family is incomplete; missing {len(missing)} slot(s): "
            f"{sorted(missing)[:3]}"
        )
    if extra:
        raise ValueError(
            f"Undeclared slot(s) present: {sorted(extra)[:3]}; the family "
            "must not be enlarged"
        )
    internal: dict[str, float] = {}
    unavailable: set[str] = set()
    for key in declared:
        raw = p_values[key]
        if raw is None:
            internal[key] = 1.0
            unavailable.add(key)
            continue
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ValueError(f"Invalid p-value for {key!r}: {raw!r}")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(
                f"Invalid p-value for {key!r}: {raw!r}; must be finite in "
                "[0, 1] or None"
            )
        internal[key] = value
    adjusted = holm_bonferroni(internal)
    slots: dict[str, dict[str, Any]] = {}
    for key in declared:
        if key in unavailable:
            slots[key] = {
                "available": False,
                "adjusted_p": None,
                "significant": None,
            }
        else:
            adj, significant = adjusted[key]
            slots[key] = {
                "available": True,
                "adjusted_p": float(adj),
                "significant": bool(significant),
            }
    return {
        "family_size": DECLARED_FAMILY_SIZE,
        "valid_test_count": DECLARED_FAMILY_SIZE - len(unavailable),
        "unavailable_slot_count": len(unavailable),
        "slots": slots,
    }


def comparison_to_json(result: Mapping[str, Any], *, indent: int = 2) -> str:
    """Serialize a comparison result, refusing NaN/Inf.

    Args:
        result: JSON-safe comparison dict (or nested mapping).
        indent: Indentation passed to :func:`json.dumps`.

    Returns:
        JSON text with no non-finite numbers.

    Raises:
        ValueError: If the payload contains a non-finite float.
    """
    return json.dumps(result, allow_nan=False, indent=indent, sort_keys=True)
