"""Tests for the frozen-family paired comparison statistics.

Covers the protocol-fixed descriptive/inferential behaviour: prefix-cluster
units, exact and Monte Carlo sign-flip p-values, sign symmetry, Holm
adjustment over the fixed 78-slot family, fail-closed validation, and
effect-size sign/units. All statistics are engineering descriptive
validation only.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from nsmor.analysis.model_comparison import (
    ALIGNED_OVERALL_SCOPE,
    DECLARED_FAMILY_SIZE,
    SCOPE_IDS,
    comparison_to_json,
    declared_family_slots,
    holm_correct_declared_family,
    paired_mse_comparison,
)
from nsmor.analysis.uq import cohens_d


def _call(
    candidate: list[float],
    comparator: list[float],
    prefixes: list[str],
    *,
    frames: list[int] | None = None,
    exchange: bool = False,
    seed: int = 42,
) -> dict:
    n = len(candidate)
    return paired_mse_comparison(
        candidate_mse=candidate,
        comparator_mse=comparator,
        trial_ids=[f"t{i}" for i in range(n)],
        prefix_ids=prefixes,
        frame_counts=frames if frames is not None else [1] * n,
        exchangeability_asserted=exchange,
        seed=seed,
    )


# ── Declared family is frozen ─────────────────────────────────────────


def test_declared_family_is_78_unique_slots() -> None:
    slots = declared_family_slots()
    assert len(slots) == 78 == DECLARED_FAMILY_SIZE
    assert len(set(slots)) == 78
    assert len(SCOPE_IDS) == 13
    assert ALIGNED_OVERALL_SCOPE in SCOPE_IDS


# ── Prefix cluster unit (unequal prefix sizes) ────────────────────────


def test_cluster_statistic_is_prefix_mean_not_trial_mean() -> None:
    result = _call(
        [1.0, 2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        ["A", "B", "B", "B"],
    )
    # Trial mean would be 1.75; unweighted prefix mean is (1 + 2) / 2.
    assert result["mean_delta"] == pytest.approx(1.75)
    assert result["cluster_statistic"] == pytest.approx(1.5)
    assert result["cluster_statistic"] != pytest.approx(result["mean_delta"])
    assert result["prefix_means"] == {"A": pytest.approx(1.0), "B": pytest.approx(2.0)}
    assert result["counts"] == {"trials": 4, "frames": 4, "prefixes": 2}
    assert result["delta_convention"] == "candidate_minus_comparator_mse"


# ── Exact sign-flip enumeration ───────────────────────────────────────


def test_exact_enumeration_known_p_value() -> None:
    prefixes = ["p1", "p2", "p3", "p4", "p5", "p6"]
    result = _call([1.0] * 6, [0.0] * 6, prefixes, exchange=True)
    flip = result["sign_flip"]
    assert flip["method"] == "exact"
    assert flip["n_sign_patterns"] == 64
    assert flip["n_prefixes"] == 6
    # |mean| >= 1 only for the all-plus and all-minus patterns: 2 / 64.
    assert flip["p_value"] == pytest.approx(2.0 / 64.0)
    assert flip["reason"] is None


def test_exact_enumeration_is_two_sided_and_sign_symmetric() -> None:
    prefixes = ["p1", "p2", "p3", "p4", "p5", "p6"]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    pos = _call(values, [0.0] * 6, prefixes, exchange=True)
    # Same per-trial deltas with the opposite sign (candidate zero, comparator
    # values); both MSE vectors stay non-negative.
    neg = _call([0.0] * 6, values, prefixes, exchange=True)
    assert pos["sign_flip"]["p_value"] == pytest.approx(
        neg["sign_flip"]["p_value"]
    )
    assert neg["cluster_statistic"] == pytest.approx(-pos["cluster_statistic"])


# ── Monte Carlo branch: deterministic and plus-one corrected ──────────


def test_monte_carlo_deterministic_and_bounded() -> None:
    prefixes = [f"p{i}" for i in range(20)]
    values = [float(i) for i in range(1, 21)]
    first = _call(values, [0.0] * 20, prefixes, exchange=True, seed=42)
    second = _call(values, [0.0] * 20, prefixes, exchange=True, seed=42)
    assert first["sign_flip"] == second["sign_flip"]
    flip = first["sign_flip"]
    assert flip["method"] == "monte_carlo"
    assert flip["n_sign_patterns"] == 100000
    assert 1.0 / 100001.0 <= flip["p_value"] <= 1.0
    assert flip["p_value"] > 0.0


# ── Null p-value paths ────────────────────────────────────────────────


def test_too_few_prefixes_yields_null_p() -> None:
    result = _call([1.0] * 5, [0.0] * 5, ["a", "a", "b", "b", "c"], exchange=True)
    flip = result["sign_flip"]
    assert flip["p_value"] is None
    assert flip["reason"] == "fewer_than_six_prefixes"
    assert flip["method"] is None


def test_exchangeability_not_asserted_defaults_to_null_p() -> None:
    prefixes = ["p1", "p2", "p3", "p4", "p5", "p6"]
    result = _call([1.0] * 6, [0.0] * 6, prefixes)  # exchange defaults False
    flip = result["sign_flip"]
    assert flip["p_value"] is None
    assert flip["reason"] == "exchangeability_not_asserted"
    assert flip["exchangeability_asserted"] is False


# ── Effect size: sign and units ───────────────────────────────────────


def test_effect_size_sign_and_units() -> None:
    candidate = [1.0, 2.0, 3.0, 4.0]
    comparator = [3.0, 3.0, 3.0, 3.0]
    result = _call(candidate, comparator, ["A", "A", "A", "A"], frames=[5, 5, 5, 5])
    assert result["mean_delta"] == pytest.approx(-0.5)
    assert result["cohens_dz"]["value"] < 0.0  # candidate better -> negative
    assert result["cohens_dz"]["value"] == pytest.approx(
        cohens_d(np.array(candidate), np.array(comparator), paired=True)
    )
    assert result["cohens_dz"]["reason"] is None
    assert result["counts"]["frames"] == 20


def test_zero_variance_delta_gives_null_dz_with_reason() -> None:
    result = _call([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], ["A", "B", "C"])
    assert result["cohens_dz"] == {
        "value": None,
        "reason": "zero_delta_variance",
    }


def test_single_trial_gives_null_dz_with_reason() -> None:
    result = _call([1.0], [0.0], ["A"])
    assert result["cohens_dz"] == {
        "value": None,
        "reason": "fewer_than_two_trials",
    }


# ── Fail-closed validation ────────────────────────────────────────────


def test_empty_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        paired_mse_comparison(
            candidate_mse=[],
            comparator_mse=[],
            trial_ids=[],
            prefix_ids=[],
            frame_counts=[],
        )


def test_non_finite_mse_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _call([1.0, float("nan")], [0.0, 0.0], ["A", "B"])
    with pytest.raises(ValueError, match="non-finite"):
        _call([1.0, float("inf")], [0.0, 0.0], ["A", "B"])


def test_mismatched_lengths_fail_closed() -> None:
    with pytest.raises(AssertionError):
        _call([1.0, 2.0], [0.0], ["A", "B"])
    with pytest.raises(ValueError, match="trial count"):
        _call([1.0, 2.0], [0.0, 0.0], ["A"])


def test_invalid_identifiers_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _call([1.0, 2.0], [0.0, 0.0], ["A", "  "])
    with pytest.raises(ValueError, match="unique"):
        paired_mse_comparison(
            candidate_mse=[1.0, 2.0],
            comparator_mse=[0.0, 0.0],
            trial_ids=["dup", "dup"],
            prefix_ids=["A", "B"],
            frame_counts=[1, 1],
        )


def test_frame_counts_validation() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        _call([1.0, 2.0], [0.0, 0.0], ["A", "B"], frames=[1, 0])
    with pytest.raises(ValueError, match="integral"):
        _call([1.0, 2.0], [0.0, 0.0], ["A", "B"], frames=[1.0, 2.0])


# ── Holm adjustment over the fixed family ─────────────────────────────


def _full_family(fill: float | None, unavailable: tuple[str, ...] = ()) -> dict:
    slots = declared_family_slots()
    return {
        slot: (None if (fill is None or slot in unavailable) else fill)
        for slot in slots
    }


def test_holm_preserves_78_and_nulls_unavailable() -> None:
    slots = declared_family_slots()
    p_values = {slot: 0.01 for slot in slots}
    unavailable = (slots[0], slots[5])
    for slot in unavailable:
        p_values[slot] = None
    result = holm_correct_declared_family(p_values)
    assert result["family_size"] == 78
    assert len(result["slots"]) == 78
    assert result["unavailable_slot_count"] == 2
    assert result["valid_test_count"] == 76
    for slot in unavailable:
        assert result["slots"][slot] == {
            "available": False,
            "adjusted_p": None,
            "significant": None,
        }


def test_unavailable_treated_as_p_one_preserves_adjustment() -> None:
    slots = declared_family_slots()
    all_available = holm_correct_declared_family({slot: 0.01 for slot in slots})
    with_null = holm_correct_declared_family(
        {slot: (None if slot == slots[3] else 0.01) for slot in slots}
    )
    # The single unavailable slot sorts last (p=1) and must not shift the
    # adjusted values of the remaining slots, since m stays 78.
    for slot in slots:
        if slot != slots[3]:
            assert with_null["slots"][slot]["adjusted_p"] == pytest.approx(
                all_available["slots"][slot]["adjusted_p"]
            )


def test_holm_adjusted_p_is_monotone() -> None:
    slots = declared_family_slots()
    p_values = {slot: 0.001 * (i + 1) for i, slot in enumerate(slots)}
    result = holm_correct_declared_family(p_values)
    ordered = sorted(
        (p_values[slot], result["slots"][slot]["adjusted_p"]) for slot in slots
    )
    adjusted = [adj for _, adj in ordered]
    assert adjusted == sorted(adjusted)
    assert all(0.0 <= adj <= 1.0 for adj in adjusted)


def test_holm_rejects_incomplete_or_enlarged_family() -> None:
    slots = declared_family_slots()
    with pytest.raises(ValueError, match="incomplete"):
        holm_correct_declared_family({slot: 0.5 for slot in slots[:-1]})
    with pytest.raises(ValueError, match="Undeclared"):
        holm_correct_declared_family(
            {**{slot: 0.5 for slot in slots}, "extra|slot|x": 0.5}
        )


def test_holm_rejects_invalid_p_values() -> None:
    slots = declared_family_slots()
    bad = {slot: 0.5 for slot in slots}
    bad[slots[0]] = float("nan")
    with pytest.raises(ValueError, match="Invalid p-value"):
        holm_correct_declared_family(bad)


# ── JSON serialization is finite ──────────────────────────────────────


def test_json_output_has_no_non_finite_numbers() -> None:
    result = _call([1.0, 2.0, 3.0], [0.5, 1.5, 2.5], ["A", "B", "C"])
    text = comparison_to_json(result)
    assert "NaN" not in text
    assert "Infinity" not in text
    restored = json.loads(text)
    assert restored["counts"]["trials"] == 3
    assert all(math.isfinite(v) for v in restored["prefix_means"].values())


def test_json_rejects_non_finite_payload() -> None:
    with pytest.raises(ValueError):
        comparison_to_json({"value": float("nan")})


# ── Reviewer B fix 1: exchangeability_asserted must be a genuine bool ──


@pytest.mark.parametrize("bad", ["false", "true", "", 1, 0, None, [False], 0.0])
def test_exchangeability_rejects_non_bool(bad: object) -> None:
    prefixes = ["p1", "p2", "p3", "p4", "p5", "p6"]
    with pytest.raises(ValueError, match="genuine bool"):
        paired_mse_comparison(
            candidate_mse=[1.0] * 6,
            comparator_mse=[0.0] * 6,
            trial_ids=[f"t{i}" for i in range(6)],
            prefix_ids=prefixes,
            frame_counts=[1] * 6,
            exchangeability_asserted=bad,  # type: ignore[arg-type]
        )


def test_exchangeability_accepts_genuine_bool() -> None:
    prefixes = ["p1", "p2", "p3", "p4", "p5", "p6"]
    result = _call([1.0] * 6, [0.0] * 6, prefixes, exchange=np.bool_(True))
    assert result["sign_flip"]["exchangeability_asserted"] is True
    assert result["sign_flip"]["p_value"] is not None


# ── Reviewer B fix 2: protocol seed fixed at 42 ───────────────────────


@pytest.mark.parametrize("bad", [0, 43, 42.0, True, False, "42", None])
def test_seed_must_be_protocol_fixed_42(bad: object) -> None:
    prefixes = ["p1", "p2", "p3", "p4", "p5", "p6"]
    with pytest.raises(ValueError, match="seed must be"):
        paired_mse_comparison(
            candidate_mse=[1.0] * 6,
            comparator_mse=[0.0] * 6,
            trial_ids=[f"t{i}" for i in range(6)],
            prefix_ids=prefixes,
            frame_counts=[1] * 6,
            exchangeability_asserted=True,
            seed=bad,  # type: ignore[arg-type]
        )


def test_seed_42_integer_accepted() -> None:
    prefixes = ["p1", "p2", "p3", "p4", "p5", "p6"]
    result = _call(
        [1.0] * 6, [0.0] * 6, prefixes, exchange=True, seed=np.int64(42)
    )
    assert result["sign_flip"]["seed"] == 42


# ── Reviewer B fix 3: bounded chunked MC, deterministic, same result ──


def test_mc_chunk_invariant_and_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    from nsmor.analysis import model_comparison as mod

    prefixes = [f"p{i}" for i in range(17)]  # > 16 -> MC branch
    values = [float(i) for i in range(1, 18)]
    default = _call(values, [0.0] * 17, prefixes, exchange=True)
    assert default["sign_flip"]["method"] == "monte_carlo"
    assert default["sign_flip"]["n_sign_patterns"] == 100000
    assert default["sign_flip"]["seed"] == 42

    # The p-value must not depend on how the 100000 patterns are chunked.
    for chunk in (1000, 1, 7777):
        monkeypatch.setattr(mod, "_MC_CHUNK", chunk)
        got = _call(values, [0.0] * 17, prefixes, exchange=True)
        assert got["sign_flip"]["p_value"] == default["sign_flip"]["p_value"]
        assert got["sign_flip"] == default["sign_flip"]


def test_mc_matrix_cells_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from nsmor.analysis import model_comparison as mod

    prefixes = [f"p{i}" for i in range(20)]
    values = [float(i) for i in range(1, 21)]
    cells: list[int] = []
    real_empty = mod.np.empty

    def spy_empty(shape: object, *a: object, **k: object) -> np.ndarray:
        if isinstance(shape, tuple) and len(shape) == 2:
            cells.append(int(shape[0]) * int(shape[1]))
        return real_empty(shape, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(mod.np, "empty", spy_empty)
    _call(values, [0.0] * 20, prefixes, exchange=True)
    assert cells
    assert max(cells) <= mod._MC_MAX_CELLS


def test_mc_cell_budget_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from nsmor.analysis import model_comparison as mod

    prefixes = [f"p{i}" for i in range(17)]
    values = [float(i) for i in range(1, 18)]
    monkeypatch.setattr(mod, "_MC_MAX_CELLS", 16)
    with pytest.raises(ValueError, match="cell budget"):
        _call(values, [0.0] * 17, prefixes, exchange=True)


# ── Reviewer B fix 4: extreme finite inputs fail closed, no overflow ──


def test_negative_mse_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _call([1.0, -1.0], [0.0, 0.0], ["A", "B"])
    with pytest.raises(ValueError, match="non-negative"):
        _call([1.0, 2.0], [0.0, -1.0], ["A", "B"])


def test_extreme_finite_prefix_mean_does_not_spuriously_overflow() -> None:
    huge = 1.7e308
    # Pairwise float summation overflowed two huge deltas to +inf; the exact
    # mean of two identical finite values is the value itself, so the reducer
    # must return it rather than failing closed on an avoidable overflow.
    result = _call([huge, huge], [0.0, 0.0], ["A", "A"])
    assert result["prefix_means"]["A"] == pytest.approx(huge)
    assert result["cluster_statistic"] == pytest.approx(huge)
    assert math.isfinite(result["cluster_statistic"])


def test_exact_mean_guard_rejects_non_finite() -> None:
    from nsmor.analysis.model_comparison import _exact_mean

    with pytest.raises(ValueError, match="finite"):
        _exact_mean([1.0, float("inf")])
    with pytest.raises(ValueError, match="finite"):
        _exact_mean([float("nan")])
    with pytest.raises(ValueError, match="at least one"):
        _exact_mean([])


def test_finite_extreme_still_serializes() -> None:
    result = _call([1e300, 1e300, 1e300], [0.0, 0.0, 0.0], ["A", "B", "C"])
    assert math.isfinite(result["mean_delta"])
    assert math.isfinite(result["cluster_statistic"])
    assert "Infinity" not in comparison_to_json(result)


# ── A/B root fix: exact tiny effect, scale invariance ─────────────────


def test_exact_tiny_effect_is_two_of_64_not_one() -> None:
    prefixes = ["p1", "p2", "p3", "p4", "p5", "p6"]
    result = _call([1e-13] * 6, [0.0] * 6, prefixes, exchange=True)
    assert result["sign_flip"]["method"] == "exact"
    assert result["sign_flip"]["p_value"] == pytest.approx(2.0 / 64.0)


def test_exact_p_is_scale_invariant() -> None:
    prefixes = ["p1", "p2", "p3", "p4", "p5", "p6"]
    tiny = _call([1e-13] * 6, [0.0] * 6, prefixes, exchange=True)
    huge = _call([1e13] * 6, [0.0] * 6, prefixes, exchange=True)
    assert tiny["sign_flip"]["p_value"] == pytest.approx(
        huge["sign_flip"]["p_value"]
    )
    assert tiny["sign_flip"]["p_value"] == pytest.approx(2.0 / 64.0)


# ── A root fix: representable d_z for extreme finite deltas ───────────


@pytest.mark.parametrize("base", [1e300, 1e-200])
def test_dz_representable_for_extreme_finite_deltas(base: float) -> None:
    result = _call([base, 2.0 * base], [0.0, 0.0], ["A", "B"])
    assert result["cohens_dz"]["reason"] is None
    assert result["cohens_dz"]["value"] == pytest.approx(3.0 / math.sqrt(2.0))


# ── A/B root fix: frame totals summed as Python ints ──────────────────


def test_huge_frame_counts_do_not_overflow() -> None:
    big = 2 ** 62
    result = _call([1.0, 2.0], [0.0, 0.0], ["A", "B"], frames=[big, big])
    assert result["counts"]["frames"] == 2 ** 63


# ── A/B root fix: exact tie-boundary decisions (no tolerance) ─────────


def _delta_to_mse(delta: list[float]) -> tuple[list[float], list[float]]:
    """Split per-prefix deltas into non-negative candidate/comparator MSE."""
    candidate = [max(d, 0.0) for d in delta]
    comparator = [max(-d, 0.0) for d in delta]
    return candidate, comparator


def _exact_oracle(delta: list[float]) -> float:
    """Independent Fraction/brute-force sign-flip p for the same statistic."""
    fractions = [Fraction.from_float(d) for d in delta]
    n = len(fractions)
    observed = abs(sum(fractions) / n)
    hits = sum(
        1
        for signs in product((1, -1), repeat=n)
        if abs(sum(s * f for s, f in zip(signs, fractions)) / n) >= observed
    )
    return hits / (1 << n)


def test_exact_tie_boundary_matches_oracle_cancellation() -> None:
    delta = [1.0, -1.0, 2.0 ** -46, 2.0 ** -46, 2.0 ** -46, 2.0 ** -46]
    candidate, comparator = _delta_to_mse(delta)
    prefixes = [f"p{i}" for i in range(6)]
    result = _call(candidate, comparator, prefixes, exchange=True)
    assert result["sign_flip"]["p_value"] == pytest.approx(36.0 / 64.0)
    assert result["sign_flip"]["p_value"] == pytest.approx(_exact_oracle(delta))


def test_exact_tiny_unequal_effects_match_oracle() -> None:
    delta = [1.0] + [1e-14] * 5
    candidate, comparator = _delta_to_mse(delta)
    prefixes = [f"p{i}" for i in range(6)]
    result = _call(candidate, comparator, prefixes, exchange=True)
    assert result["sign_flip"]["p_value"] == pytest.approx(2.0 / 64.0)
    assert result["sign_flip"]["p_value"] == pytest.approx(_exact_oracle(delta))


# ── A/B root fix: all-zero prefix means have a defined p=1 ────────────


def test_all_zero_prefix_means_exact_p_is_one() -> None:
    prefixes = [f"p{i}" for i in range(6)]
    result = _call([0.0] * 6, [0.0] * 6, prefixes, exchange=True)
    assert result["sign_flip"]["method"] == "exact"
    assert result["sign_flip"]["n_sign_patterns"] == 64
    assert result["sign_flip"]["p_value"] == pytest.approx(1.0)
    assert result["sign_flip"]["reason"] is None
    assert result["cohens_dz"]["reason"] == "zero_delta_variance"


def test_all_zero_prefix_means_mc_p_is_one() -> None:
    prefixes = [f"p{i}" for i in range(17)]  # > 16 -> MC branch
    result = _call([0.0] * 17, [0.0] * 17, prefixes, exchange=True)
    assert result["sign_flip"]["method"] == "monte_carlo"
    assert result["sign_flip"]["p_value"] == pytest.approx(1.0)


# ── Reviewer A root fix: order-invariant prefix reduction ─────────────


def test_prefix_reduction_is_order_invariant_under_cancellation() -> None:
    # Each prefix holds three trials with deltas {1e16, 1, -1e16}. Pairwise
    # float summation loses the 1 when 1e16 and -1e16 are adjacent, so the
    # order [1e16, 1, -1e16] previously averaged to 0.0 and the reordered
    # [1e16, -1e16, 1] to 1/3, changing the sign-flip p from 1.0 to 2/64.
    prefixes = [f"p{i}" for i in range(6) for _ in range(3)]
    order_a = [1e16, 1.0, -1e16] * 6
    order_b = [1e16, -1e16, 1.0] * 6
    candidate_a, comparator_a = _delta_to_mse(order_a)
    candidate_b, comparator_b = _delta_to_mse(order_b)
    result_a = _call(candidate_a, comparator_a, prefixes, exchange=True)
    result_b = _call(candidate_b, comparator_b, prefixes, exchange=True)
    # The true per-prefix mean is 1/3; order must not change it.
    for means in (result_a["prefix_means"], result_b["prefix_means"]):
        assert all(value == pytest.approx(1.0 / 3.0) for value in means.values())
    assert result_a["cluster_statistic"] == pytest.approx(1.0 / 3.0)
    assert result_b["cluster_statistic"] == pytest.approx(1.0 / 3.0)
    # Six equal nonzero prefix means -> two of 64 sign patterns reach the
    # observed magnitude.
    assert result_a["sign_flip"]["p_value"] == pytest.approx(2.0 / 64.0)
    assert result_a["sign_flip"] == result_b["sign_flip"]


def test_exact_mean_agrees_with_fraction_oracle_for_cancelling_orders() -> None:
    from nsmor.analysis.model_comparison import _exact_mean

    for order in ([1e16, 1.0, -1e16], [1e16, -1e16, 1.0], [-1e16, 1.0, 1e16]):
        exact = Fraction(1, 3)
        assert _exact_mean(order) == pytest.approx(float(exact))
        assert _exact_mean(order) == _exact_mean([1e16, 1.0, -1e16])


def test_prefix_reduction_is_order_invariant_generic() -> None:
    # A generic shuffling regression: permuting trials within prefixes must
    # not change any reported statistic for finite deltas with wide dynamic
    # range.
    rng = np.random.default_rng(7)
    n = 18
    delta = [
        float(rng.choice([-1.0, 1.0]) * 10.0 ** int(rng.integers(-12, 13)))
        for _ in range(n)
    ]
    prefixes = [f"p{i % 6}" for i in range(n)]
    candidate, comparator = _delta_to_mse(delta)
    baseline = _call(candidate, comparator, prefixes, exchange=True)
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(n)
        shuffled_candidate = [candidate[i] for i in perm]
        shuffled_comparator = [comparator[i] for i in perm]
        shuffled_prefixes = [prefixes[i] for i in perm]
        got = _call(
            shuffled_candidate, shuffled_comparator, shuffled_prefixes, exchange=True
        )
        assert got["prefix_means"] == baseline["prefix_means"]
        assert got["cluster_statistic"] == baseline["cluster_statistic"]
        assert got["sign_flip"] == baseline["sign_flip"]


# ── A/B root fix: order-invariant exact Cohen's d_z ───────────────────


def _dz_oracle(delta: list[float]) -> float:
    """Independent Fraction/Decimal paired d_z for the same deltas.

    Encodes ``sign(mean) * sqrt(mean**2 / variance)`` with an independent
    exact-rational numerator and a local 80-digit Decimal for the square
    root. The global Decimal context is never mutated.
    """
    from decimal import Decimal, localcontext

    fractions = [Fraction.from_float(d) for d in delta]
    n = len(fractions)
    mean = sum(fractions, Fraction(0)) / n
    variance = (
        sum(((f - mean) ** 2 for f in fractions), Fraction(0)) / (n - 1)
    )
    # mean**2 / variance as an exact Fraction, then sqrt in a local context.
    ratio = mean * mean / variance
    with localcontext() as ctx:
        ctx.prec = 80
        magnitude = (
            Decimal(ratio.numerator) / Decimal(ratio.denominator)
        ).sqrt()
        # Apply the sign inside the local context: negating after the context
        # exits would use the caller's precision and could raise Inexact or
        # Rounded on the caller's context.
        signed = -magnitude if mean < 0 else magnitude
    return float(signed)


def _assert_dz_matches_oracle(delta: list[float], value: float | None) -> None:
    """Compare a tiny d_z to the oracle with an ULP-scale tolerance.

    ``abs=0`` forbids an absolute slack, so a wrong value (even a wrong sign
    or a near-zero placeholder) cannot pass; the relative tolerance only
    absorbs the final float64 rounding of the oracle.
    """
    assert value is not None
    oracle = _dz_oracle(delta)
    assert math.copysign(1.0, value) == math.copysign(1.0, oracle)
    assert value == pytest.approx(oracle, rel=1e-9, abs=0.0)


def test_dz_oracle_does_not_touch_caller_decimal_context() -> None:
    from decimal import getcontext

    ctx = getcontext()
    before = (ctx.prec, ctx.rounding, dict(ctx.flags))
    _dz_oracle([-1.0, -2.0])  # negative mean exercises the sign operation
    _dz_oracle([1.0, 2.0])
    after = (ctx.prec, ctx.rounding, dict(ctx.flags))
    assert before == after
    assert ctx.prec == before[0]


def test_dz_oracle_matches_fraction_definition() -> None:
    # The oracle must encode sqrt(mean**2/variance), not sqrt(mean/variance):
    # for deltas [1, 2], mean = 1.5 and sample variance = 0.5, so the true
    # d_z = 1.5/sqrt(0.5) = 3/sqrt(2) ~ 2.1213, whereas the wrong formula
    # sqrt(mean/variance) would give sqrt(3) ~ 1.7321.
    assert _dz_oracle([1.0, 2.0]) == pytest.approx(3.0 / math.sqrt(2.0))
    assert _dz_oracle([1.0, 2.0]) != pytest.approx(math.sqrt(3.0))


def test_dz_sign_and_order_invariant_under_cancellation() -> None:
    # deltas [1, 1e16, -1e16, -0.5]: the true mean is +1/8, so d_z is a small
    # POSITIVE number. Pairwise float summation of the scaled deltas lost the
    # small addends and even produced a negative d_z.
    delta = [1.0, 1e16, -1e16, -0.5]
    candidate, comparator = _delta_to_mse(delta)
    prefixes = [f"p{i}" for i in range(4)]
    result = _call(candidate, comparator, prefixes)
    assert result["mean_delta"] == pytest.approx(0.125)
    value = result["cohens_dz"]["value"]
    assert value is not None and value > 0.0
    _assert_dz_matches_oracle(delta, value)
    # Reordering the same trials must not change the effect size.
    reordered = [-1e16, -0.5, 1.0, 1e16]
    shuffled = _call(*_delta_to_mse(reordered), prefixes)
    assert shuffled["cohens_dz"] == result["cohens_dz"]


def test_dz_is_permutation_invariant() -> None:
    from itertools import permutations

    delta = [1.0, 1e16, -1e16, -0.5]
    prefixes = [f"p{i}" for i in range(4)]
    baseline = _call(*_delta_to_mse(delta), prefixes)
    values = {
        _call(*_delta_to_mse([delta[i] for i in perm]), prefixes)["cohens_dz"][
            "value"
        ]
        for perm in permutations(range(4))
    }
    assert len(values) == 1
    only = values.pop()
    assert only == baseline["cohens_dz"]["value"]
    _assert_dz_matches_oracle(delta, only)


def test_dz_exact_mean_sign_matches_six_prefix_cancellation() -> None:
    # Six prefixes, three trials each, deltas {1e16, 1, -1e16}: mean delta 1/3
    # and a tiny positive d_z, not 0.
    delta = [1e16, 1.0, -1e16] * 6
    candidate, comparator = _delta_to_mse(delta)
    prefixes = [f"p{i}" for i in range(6) for _ in range(3)]
    result = _call(candidate, comparator, prefixes)
    assert result["mean_delta"] == pytest.approx(1.0 / 3.0)
    value = result["cohens_dz"]["value"]
    assert value is not None and value > 0.0
    _assert_dz_matches_oracle(delta, value)


def test_dz_oracle_simple_and_negative_cases() -> None:
    # [1, 2] -> mean 1.5, sample sd sqrt(0.5), d_z = 1.5/sqrt(0.5) = 3/sqrt2.
    assert _dz_oracle([1.0, 2.0]) == pytest.approx(3.0 / math.sqrt(2.0))
    # [-1, -2] must be the exact negative of [1, 2].
    assert _dz_oracle([-1.0, -2.0]) == pytest.approx(-3.0 / math.sqrt(2.0))
    # And the production reducer agrees, including the sign.
    plus = _call(*_delta_to_mse([1.0, 2.0]), ["p0", "p1"])
    minus = _call(*_delta_to_mse([-1.0, -2.0]), ["p0", "p1"])
    assert plus["cohens_dz"]["value"] == pytest.approx(3.0 / math.sqrt(2.0))
    assert minus["cohens_dz"]["value"] == pytest.approx(-3.0 / math.sqrt(2.0))
    assert minus["cohens_dz"]["value"] == pytest.approx(
        -plus["cohens_dz"]["value"]
    )


def test_dz_zero_expected_is_not_admitted_by_tolerance() -> None:
    # A cancelling delta with an exactly zero mean must yield d_z = 0 (not a
    # tiny nonzero placeholder), and the oracle must agree at abs=0.
    delta = [1e16, -1e16]
    result = _call(*_delta_to_mse(delta), ["p0", "p1"])
    assert result["cohens_dz"] == {"value": 0.0, "reason": None}
    assert _dz_oracle(delta) == 0.0
    # A genuinely tiny nonzero effect must not be absorbed by an absolute
    # tolerance: abs=0 keeps it pinned to its oracle value. Deltas
    # [1, -1 + 2**-40] have exact mean 2**-41 with unit-scale spread, so
    # d_z ~ 3.4e-13 -- nonzero but far below any 1e-12 absolute slack.
    tiny = [1.0, -1.0 + 2.0 ** -40]
    value = _call(*_delta_to_mse(tiny), ["p0", "p1"])["cohens_dz"]["value"]
    _assert_dz_matches_oracle(tiny, value)
    assert value != 0.0
    assert abs(value) < 1e-12
