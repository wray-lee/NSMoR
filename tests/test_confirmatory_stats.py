"""Tests for the pre-registered Phase-3 confirmatory statistics.

The tests cover seven layers:

1. The exact one-sided binomial sign test: tail direction, tie handling,
   non-degenerate p-values, and the exact Clopper-Pearson interval.
2. The four slot functions: H1 is the R0-R3 four-arm Bonferroni
   union with **no** single-arm path (a missing or mis-shaped arm
   raises), H2 is an intersection-union test over all four
   pre-specified classical models that takes the frozen analysis set
   and reports a worst-case missingness p, H3 is a sign test on C1-S
   minus C16 whose budget and OOM statuses are required keywords, and
   all slots fail closed at ``n < 10``.
3. The H2 analysis set: required all-seed mask and effective draws,
   the exclusion hierarchy (no double counting), and the
   ``< 40`` effective-draw exclusion.
4. The Holm family: frozen order, ``p=1`` placeholders, the exact-p
   ``mc_se=0`` rule, the Monte Carlo guard for genuine MC p-values,
   and the frozen active-order rule
   ``resolve_active_confirmatory_order`` (fail-closed; H2 enters only
   when the split-fixed multisensory test n >= 15 and the fixed-seed
   H2 calibration passes).
5. The descriptive leave-one-animal-out fragility index (user decision
   2026-10-19 #1): it is a descriptor, never a gate, and carries no
   sign-off; the seed-sign filter, G1 constant-rank handling, and the
   descriptive claim limit.
6. Fast simulation calibration at the planned sample sizes: type-I at
   the sign-test boundary under S0/S1/S2, the corrected per-seed H2
   availability, the H2 analysis-set construction, the H2 calibration
   criterion (fixed R, fixed new seed, every gate scenario including
   S1), and a family-wise error smoke test.
7. The r7 descriptive LOO fragility calibration records: flip rates
   and the minimal flipping animal set, with no gate power-loss keys.

The heavy calibration run (R_sim=20,000; H2 gate cells R=200,000) lives
in ``.scratch/confirmatory-core-20261019/calibration/``; this file only
runs fast smoke calibrations.  Every simulation uses a fixed seed.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest
from scipy import stats
from scipy.stats import beta

from nsmor.analysis import confirmatory_calibration as cc
from nsmor.analysis import confirmatory_stats as cs

ARMS = cs.H1_FOUNDATION_ARMS
MODELS = cs.H2_MODELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cp_upper(k: int, n: int, level: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound for k successes out of n."""
    if n <= 0:
        raise ValueError("n must be positive")
    if k >= n:
        return 1.0
    return float(beta.ppf(level, k + 1, n - k))


def _h1_inputs(
    d: np.ndarray, base: float = 100.0
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Convert per-animal arm differences into MSE arrays."""
    assert d.ndim == 2 and d.shape[1] == len(ARMS)
    persistence = np.full(d.shape[0], base, dtype=np.float64)
    by_arm = {
        arm: persistence + d[:, j] for j, arm in enumerate(ARMS)
    }
    return by_arm, persistence


def _h3_inputs(
    d: np.ndarray, base: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    c16 = np.full(d.shape[0], base, dtype=np.float64)
    c1s = c16 + d
    return c1s, c16


def _h2_classical(
    values: np.ndarray,
) -> dict[str, np.ndarray]:
    assert values.ndim == 2 and values.shape[1] == len(MODELS)
    return {
        model: values[:, index].astype(np.float64)
        for index, model in enumerate(MODELS)
    }


def _animal_key(n: int) -> list[str]:
    """Return unique Cercus-style animal keys for a test array."""
    return [f"animal-{index}" for index in range(n)]


def _infer_n(slot, args: tuple[object, ...]) -> int:
    """Infer the animal count from the first slot argument."""
    if slot is cs.h1_foundation_persistence:
        first = next(iter(args[0].values()))  # type: ignore[union-attr]
        return int(np.asarray(first).shape[0])
    if slot is cs.h1_seed_sign_consistency:
        first = next(iter(args[0].values()))  # type: ignore[union-attr]
        return int(np.asarray(first).shape[1])
    return int(np.asarray(args[0]).shape[0])


def _call(slot, *args, **kwargs):
    """Call a confirmatory slot with the mandatory provenance arguments."""
    kwargs.setdefault("animal_key", _animal_key(_infer_n(slot, args)))
    if slot in (
        cs.h1_foundation_persistence,
        cs.h1_seed_sign_consistency,
    ):
        kwargs.setdefault("persistence_baseline", "v_t")
    return slot(*args, **kwargs)


def _build_report(*args, **kwargs):
    """Call confirmatory_report with a synthetic split fixture by default."""
    kwargs.setdefault("split_sha256", "synthetic:test")
    return cs.confirmatory_report(*args, **kwargs)


def _h2_set_inputs(
    n: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
]:
    l_obs = np.linspace(90.0, 110.0, n)
    l_nsmor = l_obs + 5.0
    cls = {
        model: np.full(n, 10.0 + index, dtype=np.float64)
        for index, model in enumerate(MODELS)
    }
    mask = np.ones(n, dtype=bool)
    draws = {
        model: np.full(n, 100.0, dtype=np.float64) for model in MODELS
    }
    return l_obs, l_nsmor, cls, mask, draws


# ---------------------------------------------------------------------------
# Exact sign test
# ---------------------------------------------------------------------------


class TestExactSignTest:
    def test_exact_p_matches_binomial_tail(self) -> None:
        d = np.array([-1.0] * 10 + [1.0] * 2)
        p, info = cs._exact_sign_test(d, 0.0)
        expected = float(
            stats.binomtest(10, 12, 0.5, alternative="greater").pvalue
        )
        assert p == pytest.approx(expected, rel=1e-12)
        assert info["n_below_boundary"] == 10
        assert info["n_tied_at_boundary"] == 0
        assert info["construction"] == (
            "exact_one_sided_binomial_sign_test"
        )

    def test_no_improvements_gives_p_one(self) -> None:
        d = np.array([0.5, 1.0, 2.0])
        p, info = cs._exact_sign_test(d, 0.0)
        assert p == 1.0
        assert info["n_below_boundary"] == 0

    def test_all_improvements_gives_two_to_minus_n(self) -> None:
        d = np.array([-1.0] * 12)
        p, _ = cs._exact_sign_test(d, 0.0)
        assert p == pytest.approx(2.0 ** -12, rel=1e-12)

    def test_ties_are_non_improvements(self) -> None:
        d = np.array([-1.0, -1.0, 0.0, 0.0, 1.0])
        p, info = cs._exact_sign_test(d, 0.0)
        expected = float(
            stats.binomtest(2, 5, 0.5, alternative="greater").pvalue
        )
        assert p == pytest.approx(expected, rel=1e-12)
        assert info["n_tied_at_boundary"] == 2
        assert info["tie_policy"] == "non_improvement"

    def test_boundary_shift(self) -> None:
        d = np.array([13.0, 14.0, 16.0, 17.0])
        p, info = cs._exact_sign_test(d, 15.0)
        assert info["n_below_boundary"] == 2
        assert p == pytest.approx(0.6875, rel=1e-12)

    def test_clopper_pearson_interval(self) -> None:
        d = np.array([-1.0] * 10 + [1.0] * 2)
        _, info = cs._exact_sign_test(d, 0.0)
        interval = stats.binomtest(10, 12).proportion_ci(
            confidence_level=0.95, method="exact"
        )
        assert info["proportion_ci_low"] == pytest.approx(
            float(interval.low)
        )
        assert info["proportion_ci_high"] == pytest.approx(
            float(interval.high)
        )

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError):
            cs._exact_sign_test(np.array([1.0, np.nan]), 0.0)


# ---------------------------------------------------------------------------
# H1: R0-R3 four-arm Bonferroni union (single-arm mode available)
# ---------------------------------------------------------------------------


class TestH1FoundationPersistence:
    def test_no_single_arm_path(self) -> None:
        # r9: the ``arm`` argument let a caller drop the four-arm
        # Bonferroni multiplier (single-arm p=0.00636 vs union 0.02545).
        n = 12
        d = np.column_stack(
            [np.full(n, 3.0), np.full(n, -2.0), np.full(n, 4.0), np.full(n, 5.0)]
        )
        by_arm, persistence = _h1_inputs(d)
        with pytest.raises(TypeError):
            _call(cs.h1_foundation_persistence, 
                by_arm, persistence, "onset", arm="R1"  # type: ignore[call-arg]
            )

    def test_union_multiplier_is_always_four(self) -> None:
        n = 12
        d = np.column_stack(
            [np.full(n, 3.0), np.full(n, -2.0), np.full(n, 4.0), np.full(n, 5.0)]
        )
        result = _call(cs.h1_foundation_persistence, 
            *_h1_inputs(d), "onset", n_boot=1, compute_ci=False
        )
        assert result.effect["bonferroni_multiplier"] == 4
        assert result.effect["winning_arm"] == "R1"
        # 14/17-style improvement: one arm at p=0.00636 must not be
        # promotable to "confirmatory"; the union p is 4x that.
        assert result.p_value == pytest.approx(
            min(1.0, 4.0 * 2.0 ** -12), rel=1e-12
        )

    def test_four_arm_bonferroni_union_default(self) -> None:
        n = 12
        d_r0 = np.column_stack(
            [np.full(n, -2.0), np.full(n, 3.0), np.full(n, 4.0), np.full(n, 5.0)]
        )
        res_r0 = _call(cs.h1_foundation_persistence, 
            *_h1_inputs(d_r0), "onset", n_boot=1, compute_ci=False
        )
        assert res_r0.effect["winning_arm"] == "R0"
        # The union statistic and CI are selected by the smallest arm
        # p-value, so the winner's-curse label must be explicit.
        assert res_r0.effect["winner_conditional"] is True
        assert res_r0.effect["ci_is_winner_conditional"] is True
        assert res_r0.p_value == pytest.approx(
            min(1.0, 4.0 * (2.0 ** -12)), rel=1e-12
        )

        d_r1 = np.column_stack(
            [np.full(n, 3.0), np.full(n, -2.0), np.full(n, 4.0), np.full(n, 5.0)]
        )
        res_r1 = _call(cs.h1_foundation_persistence, 
            *_h1_inputs(d_r1), "onset", n_boot=1, compute_ci=False
        )
        assert res_r1.effect["winning_arm"] == "R1"
        assert res_r1.p_value == pytest.approx(
            min(1.0, 4.0 * (2.0 ** -12)), rel=1e-12
        )

    def test_missing_arm_raises_instead_of_silently_changing_winner(
        self,
    ) -> None:
        # r9: a pipeline misalignment (for example R1 with the wrong
        # animal count) used to mark the arm unavailable, silently
        # changing the winning arm and the winner-conditional CI.
        n = 12
        by_arm, persistence = _h1_inputs(np.zeros((n, 4)))
        by_arm.pop("R1")
        with pytest.raises(ValueError):
            _call(cs.h1_foundation_persistence, 
                by_arm, persistence, "onset", n_boot=1, compute_ci=False
            )

    def test_wrong_shape_raises(self) -> None:
        n = 12
        by_arm, persistence = _h1_inputs(np.zeros((n, 4)))
        by_arm["R2"] = np.zeros(n - 1)
        with pytest.raises(ValueError):
            _call(cs.h1_foundation_persistence, 
                by_arm, persistence, "onset", n_boot=1, compute_ci=False
            )

    def test_negative_or_non_finite_arm_raises(self) -> None:
        n = 12
        by_arm, persistence = _h1_inputs(np.zeros((n, 4)))
        by_arm["R0"] = np.full(n, -1.0)
        with pytest.raises(ValueError):
            _call(cs.h1_foundation_persistence, 
                by_arm, persistence, "onset", n_boot=1, compute_ci=False
            )
        by_arm, persistence = _h1_inputs(np.zeros((n, 4)))
        by_arm["R0"] = np.full(n, np.nan)
        with pytest.raises(ValueError):
            _call(cs.h1_foundation_persistence, 
                by_arm, persistence, "onset", n_boot=1, compute_ci=False
            )

    def test_unknown_scope_raises(self) -> None:
        n = 12
        with pytest.raises(ValueError):
            _call(cs.h1_foundation_persistence, 
                *_h1_inputs(np.zeros((n, 4))),
                "decay",
                n_boot=1,
                compute_ci=False,
            )

    def test_n_below_minimum_is_unavailable(self) -> None:
        n = 9
        d = np.full((n, 4), -1.0)
        result = _call(cs.h1_foundation_persistence, 
            *_h1_inputs(d), "h5", n_boot=1, compute_ci=False
        )
        assert not result.available
        assert result.p_value == 1.0
        assert "n_lt_10" in result.reason

    def test_improvement_direction(self) -> None:
        n = 20
        better = np.column_stack(
            [
                np.zeros(n),
                np.full(n, -10.0),
                np.zeros(n),
                np.zeros(n),
            ]
        )
        worse = np.column_stack(
            [
                np.zeros(n),
                np.full(n, 10.0),
                np.zeros(n),
                np.zeros(n),
            ]
        )
        p_better = _call(cs.h1_foundation_persistence, 
            *_h1_inputs(better), "onset", n_boot=1, compute_ci=False
        )
        p_worse = _call(cs.h1_foundation_persistence, 
            *_h1_inputs(worse), "onset", n_boot=1, compute_ci=False
        )
        assert p_better.p_value == pytest.approx(
            4.0 * (2.0 ** -20), rel=1e-12
        )
        assert p_worse.p_value == 1.0

    def test_skill_is_reported(self) -> None:
        # Protocol section 3.2 requires
        # skill = 1 - mean(MSE_a*) / mean(MSE_pers).
        n = 20
        d = np.column_stack(
            [np.zeros(n), np.full(n, -10.0), np.zeros(n), np.zeros(n)]
        )
        result = _call(cs.h1_foundation_persistence, 
            *_h1_inputs(d, base=100.0), "onset", n_boot=1, compute_ci=False
        )
        assert result.effect["skill"] == pytest.approx(0.10, rel=1e-9)
        assert result.effect["mean_mse_persistence"] == pytest.approx(100.0)

    def test_r1_winner_carries_construction_caveat(self) -> None:
        n = 12
        d = np.column_stack(
            [np.full(n, 3.0), np.full(n, -2.0), np.full(n, 4.0), np.full(n, 5.0)]
        )
        result = _call(cs.h1_foundation_persistence, 
            *_h1_inputs(d), "onset", n_boot=1, compute_ci=False
        )
        assert result.effect["winning_arm"] == "R1"
        assert "residual" in str(result.effect["caveat"])

    def test_effect_is_strict_json_serializable(self) -> None:
        n = 12
        result = _call(cs.h1_foundation_persistence, 
            *_h1_inputs(np.zeros((n, 4))),
            "onset",
            n_boot=1,
            compute_ci=False,
        )
        json.dumps(asdict(result), allow_nan=False)


class TestH1SeedConsistency:
    def _by_seed(
        self, per_seed_arm: dict[str, list[float]], n: int, base: float = 100.0
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        persistence = np.full(
            (len(next(iter(per_seed_arm.values()))), n), base
        )
        by_arm = {
            arm: persistence + np.asarray(
                [np.full(n, value) for value in values], dtype=np.float64
            )
            for arm, values in per_seed_arm.items()
        }
        return by_arm, persistence

    def _four_arms(self, values: dict[str, list[float]]) -> dict[str, list[float]]:
        out = {
            "R0": [0.0] * 3,
            "R1": [0.0] * 3,
            "R2": [0.0] * 3,
            "R3": [0.0] * 3,
        }
        out.update(values)
        return out

    def test_uses_frozen_union_winner_and_is_consistent(self) -> None:
        n = 12
        by_arm, persistence = self._by_seed(
            self._four_arms({"R0": [-2.0, -2.0, -2.0]}), n
        )
        result = _call(cs.h1_seed_sign_consistency, by_arm, persistence, "onset")
        assert result["available"] is True
        assert result["winning_arm"] == "R0"
        assert result["consistent"] is True
        assert result["per_seed_statistics"] == [-2.0, -2.0, -2.0]
        assert result["selection_rule"].startswith("frozen_union_winner")

    def test_mixed_seed_signs_are_inconsistent_for_the_frozen_winner(
        self,
    ) -> None:
        n = 12
        by_arm, persistence = self._by_seed(
            self._four_arms({"R0": [-2.0, -2.0, 2.0]}), n
        )
        result = _call(cs.h1_seed_sign_consistency, by_arm, persistence, "onset")
        assert result["available"] is True
        assert result["winning_arm"] == "R0"
        assert result["consistent"] is False
        assert result["per_seed_statistics"] == [-2.0, -2.0, 2.0]

    def test_missing_arm_raises(self) -> None:
        # r9: a missing arm used to silently change the winner.
        persistence = np.zeros((3, 12))
        with pytest.raises(ValueError):
            _call(cs.h1_seed_sign_consistency, {"R0": persistence}, persistence, "onset")

    def test_shape_mismatch_raises(self) -> None:
        persistence = np.zeros((3, 12))
        by_arm = {"R0": persistence, "R1": persistence, "R2": persistence}
        by_arm["R3"] = np.zeros((3, 11))
        with pytest.raises(ValueError):
            _call(cs.h1_seed_sign_consistency, by_arm, persistence, "onset")

    def test_negative_mse_raises(self) -> None:
        persistence = np.full((3, 12), 100.0)
        by_arm = {
            arm: persistence.copy()
            for arm in cs.H1_FOUNDATION_ARMS
        }
        by_arm["R2"][0, 0] = -1.0
        with pytest.raises(ValueError):
            _call(cs.h1_seed_sign_consistency, by_arm, persistence, "onset")



# ---------------------------------------------------------------------------
# H2 analysis set and seed aggregation
# ---------------------------------------------------------------------------


class TestH2AnalysisSet:
    def test_mask_and_draws_are_required(self) -> None:
        l_obs, l_nsmor, cls, _, _ = _h2_set_inputs(12)
        with pytest.raises(TypeError):
            cs.build_h2_analysis_set(l_obs, l_nsmor, cls)  # type: ignore[call-arg]

    def test_exclusion_hierarchy_sums_to_attrition(self) -> None:
        n = 6
        l_obs, l_nsmor, cls, mask, draws = _h2_set_inputs(n)
        l_obs = l_obs.copy()
        l_nsmor = l_nsmor.copy()
        cls = {model: values.copy() for model, values in cls.items()}
        draws = {model: values.copy() for model, values in draws.items()}
        mask[0] = False  # seed incomplete
        mask[2] = False  # second seed-incomplete animal
        l_obs[1] = np.nan  # observed median undefined
        l_nsmor[1] = 100.0  # keep the mask internally consistent
        cls["race"][3] = np.nan  # classical ensemble undefined
        draws["gain"][4] = 39.0  # effective draws < 40
        analysis = cs.build_h2_analysis_set(
            l_obs,
            l_nsmor,
            cls,
            all_seed_mask=mask,
            effective_draws_by_model=draws,
        )
        assert analysis.n_animal == 1
        assert analysis.excluded["seed_incomplete"] == 2
        assert analysis.excluded["l_obs_undefined"] == 1
        assert analysis.excluded["l_nsmor_undefined"] == 0
        assert analysis.excluded["classical_undefined"] == 1
        assert analysis.excluded["effective_draws_lt_40"] == 1
        assert sum(analysis.excluded.values()) == n - analysis.n_animal
        assert not analysis.available

    def test_effective_draws_are_per_model(self) -> None:
        n = 12
        l_obs, l_nsmor, cls, mask, draws = _h2_set_inputs(n)
        draws = {model: values.copy() for model, values in draws.items()}
        draws["TWIN"][0] = 40.0
        draws["race"][0] = 39.0
        analysis = cs.build_h2_analysis_set(
            l_obs,
            l_nsmor,
            cls,
            all_seed_mask=mask,
            effective_draws_by_model=draws,
        )
        assert not analysis.mask[0]
        assert analysis.excluded["effective_draws_lt_40"] == 1
        assert analysis.n_animal == n - 1

    def test_seed_mask_true_with_undefined_l_nsmor_raises(self) -> None:
        n = 12
        l_obs, l_nsmor, cls, mask, draws = _h2_set_inputs(n)
        l_nsmor = l_nsmor.copy()
        l_nsmor[0] = np.nan
        with pytest.raises(ValueError):
            cs.build_h2_analysis_set(
                l_obs,
                l_nsmor,
                cls,
                all_seed_mask=mask,
                effective_draws_by_model=draws,
            )

    def test_mask_must_be_boolean(self) -> None:
        n = 12
        l_obs, l_nsmor, cls, mask, draws = _h2_set_inputs(n)
        with pytest.raises(ValueError):
            cs.build_h2_analysis_set(
                l_obs,
                l_nsmor,
                cls,
                all_seed_mask=mask.astype(int),
                effective_draws_by_model=draws,
            )

    def test_draws_must_be_integer_valued(self) -> None:
        n = 12
        l_obs, l_nsmor, cls, mask, draws = _h2_set_inputs(n)
        draws = {model: values.copy() for model, values in draws.items()}
        draws["race"][0] = 40.5
        with pytest.raises(ValueError):
            cs.build_h2_analysis_set(
                l_obs,
                l_nsmor,
                cls,
                all_seed_mask=mask,
                effective_draws_by_model=draws,
            )


class TestH2SeedAggregation:
    def test_averages_absolute_errors_not_latencies(self) -> None:
        seeds = np.array(
            [[10.0, 12.0, 14.0], [10.0, np.nan, 12.0], [np.nan] * 3]
        )
        l_obs = np.array([10.0, 10.0, 10.0])
        e_ns, mask, excluded = cs.h2_seed_errors(seeds, l_obs)
        assert e_ns[0] == pytest.approx(2.0)
        assert np.isnan(e_ns[1])
        assert np.isnan(e_ns[2])
        assert mask.tolist() == [True, False, False]
        # The missing-seed animals are counted once, not twice.
        assert excluded["l_nsmor_seed_undefined"] == 2
        assert excluded["n_seed_lt_required"] == 0
        assert excluded["included_with_missing_seed"] == 0
        assert (
            excluded["l_obs_undefined"]
            + excluded["l_nsmor_seed_undefined"]
            + excluded["n_seed_lt_required"]
            == 2
        )

    def test_lowered_requirement_is_a_declared_sensitivity(self) -> None:
        seeds = np.array([[10.0, 12.0, 14.0], [10.0, np.nan, 12.0]])
        l_obs = np.array([10.0, 10.0])
        e_ns, mask, excluded = cs.h2_seed_errors(
            seeds, l_obs, n_seed_required=2
        )
        assert e_ns.tolist() == pytest.approx([2.0, 1.0])
        assert mask.tolist() == [True, True]
        assert excluded["l_nsmor_seed_undefined"] == 0
        assert excluded["n_seed_lt_required"] == 0
        assert excluded["included_with_missing_seed"] == 1

    def test_l_obs_undefined_is_not_double_counted(self) -> None:
        seeds = np.array([[10.0, 12.0], [np.nan, np.nan]])
        l_obs = np.array([np.nan, 10.0])
        _, mask, excluded = cs.h2_seed_errors(seeds, l_obs)
        assert mask.tolist() == [False, False]
        assert excluded["l_obs_undefined"] == 1
        assert excluded["l_nsmor_seed_undefined"] == 1
        assert sum(
            excluded[key]
            for key in (
                "l_obs_undefined",
                "l_nsmor_seed_undefined",
                "n_seed_lt_required",
            )
        ) == 2


# ---------------------------------------------------------------------------
# H2 non-inferiority: IUT over all pre-specified models
# ---------------------------------------------------------------------------


class TestH2NonInferiority:
    def _full_set(self, n: int) -> cs.AnalysisSet:
        return cs._analysis_set_from_mask(np.ones(n, dtype=bool))

    def test_iut_is_max_over_models(self) -> None:
        n = 20
        e_ns = np.full(n, 10.0)
        good = np.full(n, 10.0)  # d = -15
        bad = np.full(n, 0.0)  # d = -5
        cls = _h2_classical(np.column_stack([bad, good, good, good]))
        result = _call(cs.h2_non_inferior_best_classical, 
            e_ns,
            cls,
            analysis_set=self._full_set(n),
            n_boot=1,
            compute_ci=False,
        )
        # All four models are non-inferior, so p is the smallest exact
        # p (all 20 animals below zero).
        assert result.p_value == pytest.approx(2.0 ** -20, rel=1e-12)
        assert result.effect["construction"] == "iut_exact_sign_test"

    def test_one_failing_model_drives_p_to_one(self) -> None:
        n = 20
        good = np.full(n, 10.0)
        failing = np.full(n, 0.0)
        e_ns_fail = np.full(n, 20.0)
        cls = _h2_classical(np.column_stack([failing, good, good, good]))
        result = _call(cs.h2_non_inferior_best_classical, 
            e_ns_fail,
            cls,
            analysis_set=self._full_set(n),
            n_boot=1,
            compute_ci=False,
        )
        # model 0: d = 20 - 0 - 15 = +5 for all animals -> p=1.
        # The other models: d = 20 - 10 - 15 = -5 -> small p.
        assert result.p_value == 1.0
        assert result.effect["p_determining_model"] == "race"
        assert result.effect["model_p_values"]["race"] == 1.0

    def test_estimand_is_median_not_mean(self) -> None:
        # 60% improve by 20 ms, 40% worsen by 100 ms: the mean moves the
        # wrong way but the median improves, and the sign test tracks
        # the median.  The effect must label the estimand as median.
        n = 20
        better = np.full(12, 10.0)
        worse = np.full(8, 130.0)
        e_ns = np.concatenate([better, worse])
        cls = _h2_classical(
            np.column_stack(
                [np.full(n, 10.0)] * 4
            )
        )
        result = _call(cs.h2_non_inferior_best_classical, 
            e_ns,
            cls,
            analysis_set=self._full_set(n),
            n_boot=1,
            compute_ci=False,
        )
        assert "median" in str(result.effect["scientific_estimand"])
        assert "mean" not in str(result.effect["scientific_estimand"])
        d = e_ns - 10.0 - 15.0
        assert float(np.mean(d)) > 0.0
        assert float(np.median(d)) < 0.0
        # The sign test counts 12/20 improvements, not the mean.
        assert result.p_value == pytest.approx(
            float(stats.binomtest(12, 20, 0.5, alternative="greater").pvalue),
            rel=1e-12,
        )

    def test_direction(self) -> None:
        n = 20
        cls_good = _h2_classical(np.full((n, 4), 20.0))
        cls_bad = _h2_classical(np.full((n, 4), 0.0))
        p_good = _call(cs.h2_non_inferior_best_classical, 
            np.full(n, 20.0),
            cls_good,
            analysis_set=self._full_set(n),
            n_boot=1,
            compute_ci=False,
        )
        p_bad = _call(cs.h2_non_inferior_best_classical, 
            np.full(n, 20.0),
            cls_bad,
            analysis_set=self._full_set(n),
            n_boot=1,
            compute_ci=False,
        )
        assert p_good.p_value == pytest.approx(2.0 ** -20, rel=1e-12)
        assert p_bad.p_value == 1.0

    def test_n_below_minimum_is_unavailable(self) -> None:
        n = 9
        result = _call(cs.h2_non_inferior_best_classical, 
            np.full(n, 10.0),
            _h2_classical(np.full((n, 4), 10.0)),
            analysis_set=self._full_set(n),
            n_boot=1,
            compute_ci=False,
        )
        assert not result.available
        assert result.p_value == 1.0
        assert result.boundary == 15.0

    def test_negative_errors_raise(self) -> None:
        n = 12
        cls = _h2_classical(np.full((n, 4), 10.0))
        cls["race"][0] = -1.0
        with pytest.raises(ValueError):
            _call(cs.h2_non_inferior_best_classical, 
                np.full(n, 10.0),
                cls,
                analysis_set=self._full_set(n),
                n_boot=1,
                compute_ci=False,
            )

    def test_analysis_set_is_required(self) -> None:
        # r9: the mask from build_h2_analysis_set used to be optional,
        # so a caller could skip every pre-registered exclusion.
        n = 12
        with pytest.raises(TypeError):
            _call(cs.h2_non_inferior_best_classical, 
                np.full(n, 10.0),
                _h2_classical(np.full((n, 4), 10.0)),
                n_boot=1,
                compute_ci=False,
            )

    def test_analysis_set_mask_is_applied(self) -> None:
        n = 20
        e_ns = np.full(n, 10.0)
        e_ns[10:] = 500.0
        cls = _h2_classical(np.full((n, 4), 10.0))
        mask = np.zeros(n, dtype=bool)
        mask[:10] = True
        result = _call(cs.h2_non_inferior_best_classical, 
            e_ns,
            cls,
            analysis_set=cs._analysis_set_from_mask(mask),
            n_boot=1,
            compute_ci=False,
        )
        assert result.n_animal == 10
        assert result.p_value == pytest.approx(2.0 ** -10, rel=1e-12)
        # The worst-case p keeps the 10 excluded animals as
        # non-improvements, so it cannot be smaller.
        assert result.effect["worst_case_p_value"] >= result.p_value
        assert result.effect["worst_case_n_animal"] == 20
        assert result.effect["worst_case_excluded_as_non_improvement"] == 10

    def test_mask_length_must_match(self) -> None:
        n = 12
        with pytest.raises(ValueError):
            _call(cs.h2_non_inferior_best_classical, 
                np.full(n, 10.0),
                _h2_classical(np.full((n, 4), 10.0)),
                analysis_set=cs._analysis_set_from_mask(
                    np.ones(n + 1, dtype=bool)
                ),
                n_boot=1,
                compute_ci=False,
            )

    def test_nan_outside_the_mask_is_a_non_improvement(self) -> None:
        n = 20
        e_ns = np.full(n, 10.0)
        e_ns[15:] = np.nan
        cls = _h2_classical(np.full((n, 4), 10.0))
        mask = np.ones(n, dtype=bool)
        mask[15:] = False
        result = _call(cs.h2_non_inferior_best_classical, 
            e_ns,
            cls,
            analysis_set=cs._analysis_set_from_mask(mask),
            n_boot=1,
            compute_ci=False,
        )
        assert result.n_animal == 15
        # 15 improvements out of the full 20 (the 5 NaN animals stay in
        # the worst-case denominator as non-improvements).
        assert result.effect["worst_case_p_value"] == pytest.approx(
            float(stats.binomtest(15, 20, 0.5, alternative="greater").pvalue),
            rel=1e-12,
        )
        assert result.effect["worst_case_p_value"] > result.p_value


class TestH2DescriptiveRow:
    def test_row_has_no_p_derived_fields(self) -> None:
        n = 20
        nsmor = np.full(n, 10.0)
        classical = _h2_classical(
            np.column_stack(
                [
                    np.full(n, 10.0),
                    np.full(n, 12.0),
                    np.full(n, 14.0),
                    np.full(n, 16.0),
                ]
            )
        )
        row = cs.h2_descriptive_row(
            nsmor,
            classical,
            analysis_set=cs._analysis_set_from_mask(
                np.ones(n, dtype=bool)
            ),
            n_boot=100,
            seed=7,
        )
        assert row["available"] is True
        assert row["binding_model"] == "race"
        assert row["inference"] == "descriptive"
        assert row["p_value"] is None
        assert row["multiplicity_control"] == "none"
        for banned in ("p_determining_model", "model_p_values"):
            assert banned not in row
        assert row["ci_low"] is not None
        assert row["ci_high"] is not None

    def test_n_below_minimum_is_unavailable(self) -> None:
        n = 9
        row = cs.h2_descriptive_row(
            np.full(n, 10.0),
            _h2_classical(np.full((n, 4), 10.0)),
            analysis_set=cs._analysis_set_from_mask(
                np.ones(n, dtype=bool)
            ),
            n_boot=10,
            seed=7,
        )
        assert row["available"] is False
        assert row["reason"] == "out_of_scope (n_lt_10)"
        assert row["p_value"] is None


# ---------------------------------------------------------------------------
# H3
# ---------------------------------------------------------------------------


class TestH3Derecursion:
    def test_exact_p_and_direction(self) -> None:
        n = 20
        better = np.full(n, -0.01)
        worse = np.full(n, 0.01)
        c16 = np.full(n, 1.0)
        p_better = _call(cs.h3_camor_derecursion, 
            c16 + better,
            c16,
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, False, False),
            n_boot=1,
            compute_ci=False,
        )
        p_worse = _call(cs.h3_camor_derecursion, 
            c16 + worse,
            c16,
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, False, False),
            n_boot=1,
            compute_ci=False,
        )
        assert p_better.p_value == pytest.approx(2.0 ** -20, rel=1e-12)
        assert p_worse.p_value == 1.0

    def test_n_below_minimum_is_unavailable(self) -> None:
        n = 9
        c16 = np.full(n, 1.0)
        result = _call(cs.h3_camor_derecursion, 
            c16 - 0.1,
            c16,
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, False, False),
            n_boot=1,
            compute_ci=False,
        )
        assert not result.available
        assert result.p_value == 1.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            _call(cs.h3_camor_derecursion, 
                np.zeros(10),
                np.zeros(11),
                budget_matched_by_seed=(True, True, True),
                oom_by_seed=(False, False, False),
                n_boot=1,
                compute_ci=False,
            )

    def test_oom_status_is_required(self) -> None:
        # r9: an omitted oom_by_seed used to be silently treated as
        # all-False, bypassing the K=16 OOM gate.
        c16 = np.full(20, 1.0)
        with pytest.raises(TypeError):
            _call(cs.h3_camor_derecursion, 
                c16 - 0.01,
                c16,
                budget_matched_by_seed=(True, True, True),
                n_boot=1,
                compute_ci=False,
            )
        result = _call(cs.h3_camor_derecursion, 
            c16 - 0.01,
            c16,
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=None,  # type: ignore[arg-type]
            n_boot=1,
            compute_ci=False,
        )
        assert result.available is False
        assert result.reason == "oom_status_missing"

    def test_any_oom_seed_downgrades_to_p_one(self) -> None:
        c16 = np.full(20, 1.0)
        result = _call(cs.h3_camor_derecursion, 
            c16 - 0.01,
            c16,
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, True, False),
            n_boot=1,
            compute_ci=False,
        )
        assert result.available is False
        assert result.p_value == 1.0
        assert result.reason.startswith("oom_k16")
        assert "[1]" in result.reason

    def test_missing_budget_status_fails_closed(self) -> None:
        c16 = np.full(20, 1.0)
        result = _call(cs.h3_camor_derecursion, 
            c16 - 0.01,
            c16,
            budget_matched_by_seed=None,  # type: ignore[arg-type]
            oom_by_seed=(False, False, False),
            n_boot=1,
            compute_ci=False,
        )
        assert result.available is False
        assert result.p_value == 1.0
        assert result.reason == "budget_status_missing"

    def test_not_matched_budget_downgrades_to_p_one(self) -> None:
        c16 = np.full(20, 1.0)
        result = _call(cs.h3_camor_derecursion, 
            c16 - 0.01,
            c16,
            budget_matched_by_seed=(True, False, True),
            oom_by_seed=(False, False, False),
            n_boot=1,
            compute_ci=False,
        )
        assert result.available is False
        assert result.p_value == 1.0
        assert result.reason.startswith("budget_not_matched")

    def test_per_seed_statistics_are_reported(self) -> None:
        n = 12
        c16 = np.full(n, 1.0)
        c1s = np.stack(
            [
                np.full(n, 0.99),
                np.full(n, 0.99),
                np.full(n, 0.99),
            ]
        )
        result = _call(cs.h3_camor_derecursion, 
            c1s.mean(axis=0),
            c16,
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, False, False),
            mse_c1s_by_seed=c1s,
            mse_c16_by_seed=np.stack([c16, c16, c16]),
            n_boot=1,
            compute_ci=False,
        )
        assert result.effect["seed_sign_consistent"] is True
        assert result.effect["per_seed_median_differences"] == pytest.approx(
            [-0.01, -0.01, -0.01]
        )
        assert result.effect["budget_matched"] is True
        assert result.effect["oom_by_seed"] == [False, False, False]

    def test_mixed_seed_signs_are_flagged(self) -> None:
        n = 12
        c16 = np.full(n, 1.0)
        c1s = np.stack(
            [
                np.full(n, 0.99),
                np.full(n, 0.99),
                np.full(n, 1.01),
            ]
        )
        result = _call(cs.h3_camor_derecursion, 
            c1s.mean(axis=0),
            c16,
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, False, False),
            mse_c1s_by_seed=c1s,
            mse_c16_by_seed=np.stack([c16, c16, c16]),
            n_boot=1,
            compute_ci=False,
        )
        assert result.effect["seed_sign_consistent"] is False

    def test_per_seed_seed_count_must_match_budget_flags(self) -> None:
        n = 12
        c16 = np.full(n, 1.0)
        with pytest.raises(ValueError):
            _call(cs.h3_camor_derecursion, 
                c16 - 0.01,
                c16,
                budget_matched_by_seed=(True, True),
                oom_by_seed=(False, False),
                mse_c1s_by_seed=np.stack([c16 - 0.01] * 3),
                mse_c16_by_seed=np.stack([c16] * 3),
                n_boot=1,
                compute_ci=False,
            )




# ---------------------------------------------------------------------------
# Holm family
# ---------------------------------------------------------------------------


class TestHolmConfirmatoryFamily:
    def test_frozen_order_and_active_slots(self) -> None:
        p = {name: 1.0 for name in cs.FAMILY_ORDER}
        p["H1_onset"] = 0.01
        holm = cs.holm_confirmatory_family(
            p, multisensory_test_n=13, h2_calibration_pass=True
        )
        assert holm.order == cs.FAMILY_ORDER
        assert holm.adjusted_p["H1_onset"] == pytest.approx(0.04)
        assert holm.significant["H1_onset"] is True
        assert holm.mc_se["H1_onset"] == 0.0
        assert holm.indeterminate_mc["H1_onset"] is False

    def test_exact_p_at_threshold_rejects(self) -> None:
        p = {name: 1.0 for name in cs.FAMILY_ORDER}
        p["H1_onset"] = 0.0125
        holm = cs.holm_confirmatory_family(
            p,
            multisensory_test_n=13,
            h2_calibration_pass=True,
            mc_se={name: 0.0 for name in cs.FAMILY_ORDER},
        )
        assert holm.adjusted_p["H1_onset"] == pytest.approx(0.05)
        assert holm.significant["H1_onset"] is True
        assert holm.indeterminate_mc["H1_onset"] is False

    def test_downgraded_slot_must_have_p_one(self) -> None:
        p = {name: 1.0 for name in cs.FAMILY_ORDER}
        p["H2_NI"] = 0.001
        with pytest.raises(ValueError):
            cs.holm_confirmatory_family(
                p, multisensory_test_n=13, h2_calibration_pass=True
            )

    def test_changing_order_raises(self) -> None:
        p = {name: 1.0 for name in cs.FAMILY_ORDER}
        with pytest.raises(ValueError):
            cs.holm_confirmatory_family(
                p,
                order=tuple(reversed(cs.FAMILY_ORDER)),
                multisensory_test_n=13,
                h2_calibration_pass=True,
            )

    def test_split_facts_are_required(self) -> None:
        # r9: the active order is recomputed, never passed in.
        p = {name: 1.0 for name in cs.FAMILY_ORDER}
        with pytest.raises(TypeError):
            cs.holm_confirmatory_family(p)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            cs.holm_confirmatory_family(
                p,
                active_order=cs.FAMILY_ORDER,  # type: ignore[call-arg]
                multisensory_test_n=13,
                h2_calibration_pass=True,
            )

    def test_h2_cannot_be_promoted_at_split_n_13(self) -> None:
        # Explicit regression for the r9 finding: passing
        # FAMILY_ORDER with split n=13 previously promoted H2.
        p = {name: 1.0 for name in cs.FAMILY_ORDER}
        p["H2_NI"] = 1e-8
        with pytest.raises(ValueError):
            cs.holm_confirmatory_family(
                p, multisensory_test_n=13, h2_calibration_pass=True
            )

    def test_mc_guard_marks_indeterminate(self) -> None:
        p = {name: 1.0 for name in cs.FAMILY_ORDER}
        p["H1_onset"] = 0.0125
        mc = {name: 0.0 for name in cs.FAMILY_ORDER}
        mc["H1_onset"] = 0.01
        holm = cs.holm_confirmatory_family(
            p,
            multisensory_test_n=13,
            h2_calibration_pass=True,
            mc_se=mc,
        )
        # adjusted SE = 4 * 0.01 = 0.04; the adjusted p is exactly at
        # alpha, inside two adjusted SEs -> indeterminate, no rejection.
        assert holm.indeterminate_mc["H1_onset"] is True
        assert holm.significant["H1_onset"] is False

    def test_mc_guard_does_not_block_clear_rejection(self) -> None:
        p = {name: 1.0 for name in cs.FAMILY_ORDER}
        p["H1_onset"] = 0.0001
        mc = {name: 0.0 for name in cs.FAMILY_ORDER}
        mc["H1_onset"] = 1e-6
        holm = cs.holm_confirmatory_family(
            p,
            multisensory_test_n=13,
            h2_calibration_pass=True,
            mc_se=mc,
        )
        assert holm.significant["H1_onset"] is True


class TestDescriptivePlaceholder:
    def test_h2_boundary_is_15(self) -> None:
        result = cs.descriptive_placeholder("H2_NI", n_animal=13)
        assert result.boundary == 15.0
        assert result.p_value == 1.0
        assert not result.available

    def test_unknown_slot_raises(self) -> None:
        with pytest.raises(ValueError):
            cs.descriptive_placeholder("H9", n_animal=13)


class TestConfirmatoryReport:
    def _full_set(self, n: int) -> cs.AnalysisSet:
        return cs._analysis_set_from_mask(np.ones(n, dtype=bool))

    def _results(self) -> dict[str, cs.TestResult]:
        n = 20
        d_better = np.column_stack(
            [
                np.zeros(n),
                np.full(n, -10.0),
                np.zeros(n),
                np.zeros(n),
            ]
        )
        d_worse = np.column_stack(
            [
                np.zeros(n),
                np.full(n, 10.0),
                np.zeros(n),
                np.zeros(n),
            ]
        )
        c16 = np.full(n, 100.0)
        return {
            "H1_onset": _call(cs.h1_foundation_persistence, 
                *_h1_inputs(d_better), "onset", n_boot=1, compute_ci=False
            ),
            "H1_h5": _call(cs.h1_foundation_persistence, 
                *_h1_inputs(d_worse), "h5", n_boot=1, compute_ci=False
            ),
            "H2_NI": cs.descriptive_placeholder("H2_NI", n_animal=n),
            "H3_derecursion": _call(cs.h3_camor_derecursion, 
                c16 - 0.01,
                c16,
                budget_matched_by_seed=(True, True, True),
                oom_by_seed=(False, False, False),
                n_boot=1,
                compute_ci=False,
            ),
        }

    def _report(self, results=None, **kwargs):
        return _build_report(
            self._results() if results is None else results,
            multisensory_test_n=13,
            h2_calibration_pass=True,
            **kwargs,
        )

    def test_report_labels_and_h2_boundary(self) -> None:
        report = self._report()
        assert report["family"]["active_order"] == list(
            cs.FAIL_CLOSED_ACTIVE_ORDER
        )
        assert report["family"]["active_order_rule"] == (
            "resolve_active_confirmatory_order("
            "split_multisensory_test_n, h2_calibration_pass)"
        )
        assert report["family"]["active_order_inputs"] == {
            "multisensory_test_n": 13,
            "h2_calibration_pass": True,
            "split_sha256": "synthetic:test",
        }
        assert report["family"]["p_value_source"] == (
            "exact_one_sided_binomial_sign_test"
        )
        assert report["slots"]["H2_NI"]["boundary"] == 15.0
        assert report["slots"]["H2_NI"]["claim_level"] == "descriptive"
        assert report["slots"]["H2_NI"]["descriptive_claim_limit"] == (
            "no superiority/non-inferiority/equivalence claim"
        )

    def test_active_order_cannot_be_supplied(self) -> None:
        # r9: an unblinding script could once pass FAMILY_ORDER and
        # promote H2 at split n=13.
        with pytest.raises(TypeError):
            _build_report(
                self._results(),
                active_order=cs.FAMILY_ORDER,  # type: ignore[call-arg]
                multisensory_test_n=13,
                h2_calibration_pass=True,
            )
        with pytest.raises(TypeError):
            _build_report(self._results())  # type: ignore[call-arg]

    def test_h2_stays_descriptive_at_split_n_13(self) -> None:
        n = 20
        results = self._results()
        results["H2_NI"] = _call(cs.h2_non_inferior_best_classical, 
            np.full(n, 10.0),
            _h2_classical(np.full((n, 4), 10.0)),
            analysis_set=self._full_set(n),
            n_boot=1,
            compute_ci=False,
        )
        with pytest.raises(ValueError):
            _build_report(
                results,
                multisensory_test_n=13,
                h2_calibration_pass=True,
            )

    def test_downgraded_slot_cannot_be_available(self) -> None:
        n = 20
        results = self._results()
        results["H2_NI"] = _call(cs.h2_non_inferior_best_classical, 
            np.full(n, 10.0),
            _h2_classical(np.full((n, 4), 10.0)),
            analysis_set=self._full_set(n),
            n_boot=1,
            compute_ci=False,
        )
        with pytest.raises(ValueError):
            _build_report(
                results,
                multisensory_test_n=13,
                h2_calibration_pass=True,
            )

    @staticmethod
    def _h1_entry(
        d: np.ndarray, scope: str, *, unstable: bool = False
    ) -> dict[str, object]:
        # r10: only the full h1_seed_sign_consistency output is accepted.
        by_arm, persistence = _h1_inputs(d)
        by_seed = {
            arm: np.tile(values, (3, 1)) for arm, values in by_arm.items()
        }
        persistence_by_seed = np.tile(persistence, (3, 1))
        if unstable:
            # R1's per-seed signs disagree (-40, +5, -20) while its
            # seed mean still improves (-18.33), so the frozen winner
            # stays R1 and only the filter flips.
            by_seed["R1"] = np.vstack(
                [persistence + shift for shift in (-40.0, 5.0, -20.0)]
            )
        return _call(cs.h1_seed_sign_consistency, 
            by_seed, persistence_by_seed, scope
        )

    def test_h1_joint_gate_requires_both_scopes_and_seed_agreement(
        self,
    ) -> None:
        report = self._report()
        assert report["family"]["h1_joint_gate"]["pass"] is False
        assert report["family"]["h1_joint_gate"]["rule"] == (
            "both_H1_onset_and_H1_h5_rejected_after_holm_with_"
            "consistent_seed_signs"
        )
        n = 20
        better = np.column_stack(
            [
                np.zeros(n),
                np.full(n, -10.0),
                np.zeros(n),
                np.zeros(n),
            ]
        )
        results = self._results()
        results["H1_h5"] = _call(cs.h1_foundation_persistence, 
            *_h1_inputs(better), "h5", n_boot=1, compute_ci=False
        )
        report_both = self._report(
            results,
            seed_consistency={
                "H1_onset": self._h1_entry(better, "onset"),
                "H1_h5": self._h1_entry(better, "h5"),
            },
        )
        assert report_both["family"]["h1_joint_gate"]["pass"] is True
        unstable = self._report(
            results,
            seed_consistency={
                "H1_onset": self._h1_entry(
                    better, "onset", unstable=True
                ),
                "H1_h5": self._h1_entry(better, "h5"),
            },
        )
        assert unstable["family"]["h1_joint_gate"]["pass"] is False
        assert unstable["slots"]["H1_onset"]["claim_level"] == (
            "indeterminate_unstable_across_seeds"
        )

    def test_missing_seed_consistency_withholds_the_claim(self) -> None:
        # r9: H1/H3 previously reported claim_level="confirmatory" even
        # when the per-seed signs were never checked.
        report = self._report()
        assert report["slots"]["H1_onset"]["claim_level"] == (
            "indeterminate_unstable_across_seeds"
        )
        # H3 is rejected but carries no per-seed array here, so the
        # positive claim is withheld as well (fail-closed).
        assert report["slots"]["H3_derecursion"]["claim_level"] == (
            "indeterminate_unstable_across_seeds"
        )
        stable = self._report(
            seed_consistency={"H3_derecursion": {"consistent": True}}
        )
        assert stable["slots"]["H3_derecursion"]["claim_level"] == (
            "confirmatory"
        )

    def test_unavailable_active_slot_is_not_a_confirmatory_negative(
        self,
    ) -> None:
        # r9: active H3 downgraded by the budget/OOM gate used to be
        # labelled "confirmatory" with an empty claim limit.
        results = self._results()
        c16 = np.full(20, 100.0)
        results["H3_derecursion"] = _call(cs.h3_camor_derecursion, 
            c16 - 0.01,
            c16,
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, True, False),
            n_boot=1,
            compute_ci=False,
        )
        report = self._report(results)
        slot = report["slots"]["H3_derecursion"]
        assert slot["claim_level"] == "unavailable"
        assert slot["available"] is False
        assert slot["claim_level"] != "confirmatory"
        assert "not a confirmatory negative" in slot["descriptive_claim_limit"]

    def test_active_h3_without_budget_evidence_raises(self) -> None:
        results = self._results()
        results["H3_derecursion"] = cs.TestResult(
            name="H3_derecursion",
            statistic=-0.01,
            boundary=0.0,
            p_value=1e-6,
            n_animal=20,
            n_boot=1,
            seed=1,
            available=True,
            reason="",
            effect={},
        )
        with pytest.raises(ValueError):
            self._report(results)

    def test_active_h2_boundary_must_be_frozen(self) -> None:
        n = 20
        results = self._results()
        results["H2_NI"] = cs.TestResult(
            name="H2_NI",
            statistic=-1.0,
            boundary=10.0,
            p_value=1e-6,
            n_animal=n,
            n_boot=1,
            seed=1,
            available=True,
            reason="",
            effect={"worst_case_p_value": 1e-6},
        )
        with pytest.raises(ValueError):
            _build_report(
                results,
                multisensory_test_n=20,
                h2_calibration_pass=True,
            )

    def test_h2_worst_case_gate_withholds_the_claim(self) -> None:
        n = 20
        results = self._results()
        results["H2_NI"] = cs.TestResult(
            name="H2_NI",
            statistic=-1.0,
            boundary=15.0,
            p_value=1e-6,
            n_animal=n,
            n_boot=1,
            seed=1,
            available=True,
            reason="",
            effect={"worst_case_p_value": 0.4},
        )
        report = _build_report(
            results,
            multisensory_test_n=20,
            h2_calibration_pass=True,
        )
        assert report["slots"]["H2_NI"]["effect"]["worst_case_gate_pass"] is False
        assert report["slots"]["H2_NI"]["claim_level"] == (
            "indeterminate_informative_missingness"
        )

    def test_descriptive_h2_row_is_merged_without_p(self) -> None:
        n = 20
        row = cs.h2_descriptive_row(
            np.full(n, 10.0),
            _h2_classical(np.full((n, 4), 10.0)),
            analysis_set=self._full_set(n),
            n_boot=50,
            seed=3,
        )
        report = self._report(descriptive_rows={"H2_NI": row})
        effect = report["slots"]["H2_NI"]["effect"]
        assert effect["binding_model"] == "race"
        assert effect["p_value"] is None
        assert "p_determining_model" not in effect
        assert "model_p_values" not in effect
        assert report["descriptive_rows"]["H2_NI"]["inference"] == (
            "descriptive"
        )




# ---------------------------------------------------------------------------
# LOO stability and seed-sign filter
# ---------------------------------------------------------------------------


class TestH2WorstCaseMask:
    """r10 regressions: excluded finite errors must never count as gains."""

    def _set(self, mask: np.ndarray) -> cs.AnalysisSet:
        return cs._analysis_set_from_mask(mask)

    def test_finite_excluded_improvers_are_non_improvements(self) -> None:
        # Reviewer A repro: n=12, |A|=10, all 12 improve.  The old code
        # counted the two excluded finite errors as improvements and
        # returned 0.000244; the frozen rule gives binom(10, 12).
        n = 12
        mask = np.zeros(n, dtype=bool)
        mask[:10] = True
        e_ns = np.full(n, 5.0)
        cls = _h2_classical(np.full((n, 4), 40.0))
        worst, _, total = cs.h2_worst_case_p(
            e_ns, cls, analysis_set=self._set(mask)
        )
        assert total == n
        assert worst == pytest.approx(
            float(stats.binomtest(10, n, 0.5, alternative="greater").pvalue),
            rel=1e-12,
        )
        result = _call(cs.h2_non_inferior_best_classical, 
            e_ns,
            cls,
            analysis_set=self._set(mask),
            n_boot=1,
            compute_ci=False,
        )
        assert result.p_value == pytest.approx(2.0**-10, rel=1e-12)
        assert result.effect["worst_case_p_value"] == pytest.approx(
            worst, rel=1e-12
        )
        assert worst >= result.p_value

    def test_reviewer_second_repro_with_draw_exclusions(self) -> None:
        # n=19, |A|=16, 14 of A improve, 3 draw<40 animals excluded
        # with finite errors: worst-case k=14, denominator 19.
        n = 19
        mask = np.zeros(n, dtype=bool)
        mask[:16] = True
        e_ns = np.full(n, 5.0)
        cls = _h2_classical(np.full((n, 4), 40.0))
        e_ns[14:16] = 100.0  # the two non-improving animals inside A
        worst, _, _ = cs.h2_worst_case_p(
            e_ns, cls, analysis_set=self._set(mask)
        )
        expected = float(
            stats.binomtest(14, n, 0.5, alternative="greater").pvalue
        )
        assert expected == pytest.approx(0.0318, abs=5e-4)
        assert worst == pytest.approx(expected, rel=1e-12)

    def test_analysis_set_is_a_required_keyword(self) -> None:
        e_ns = np.full(3, 5.0)
        cls = _h2_classical(np.full((3, 4), 40.0))
        with pytest.raises(TypeError):
            cs.h2_worst_case_p(e_ns, cls)  # type: ignore[call-arg]

    def test_mask_builder_is_private(self) -> None:
        assert not hasattr(cs, "analysis_set_from_mask")
        assert hasattr(cs, "_analysis_set_from_mask")

    def test_report_fails_closed_without_worst_case(self) -> None:
        result = _call(cs.h2_non_inferior_best_classical, 
            np.full(16, 5.0),
            _h2_classical(np.full((16, 4), 40.0)),
            analysis_set=self._set(np.ones(16, dtype=bool)),
            n_boot=1,
            compute_ci=False,
        )
        effect = dict(result.effect)
        del effect["worst_case_p_value"]
        broken = replace(result, effect=effect)
        with pytest.raises(ValueError):
            _h2_family_report(broken)


def _h2_seed_arrays(
    n: int, nsmor_by_seed: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build full per-seed absolute-error arrays for the H2 filter."""
    assert nsmor_by_seed.ndim == 2 and nsmor_by_seed.shape[1] == n
    classical = {
        model: np.full(nsmor_by_seed.shape, 10.0) for model in MODELS
    }
    return nsmor_by_seed, classical


def _h2_result(
    n: int,
    *,
    nsmor_by_seed: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> cs.TestResult:
    """Build an H2 result with or without the frozen per-seed evidence."""
    if mask is None:
        mask = np.ones(n, dtype=bool)
    animal = mask.astype(np.float64) * 10.0 + 10.0
    if nsmor_by_seed is None:
        e_ns = animal
        kwargs: dict[str, object] = {}
    else:
        e_ns = nsmor_by_seed.mean(axis=0)
        by_seed, cls_by_seed = _h2_seed_arrays(n, nsmor_by_seed)
        kwargs = {
            "abs_err_nsmor_by_seed": by_seed,
            "abs_err_classical_by_seed": cls_by_seed,
        }
    return _call(cs.h2_non_inferior_best_classical, 
        e_ns,
        _h2_classical(np.tile(animal, (4, 1)).T),
        analysis_set=cs._analysis_set_from_mask(mask),
        n_boot=1,
        compute_ci=False,
        **kwargs,
    )


def _h2_family_report(h2: cs.TestResult) -> dict[str, object]:
    n = h2.n_animal
    d_better = np.column_stack(
        [np.zeros(n), np.full(n, -10.0), np.zeros(n), np.zeros(n)]
    )
    d_worse = np.column_stack(
        [np.zeros(n), np.full(n, 10.0), np.zeros(n), np.zeros(n)]
    )
    results = {
        "H1_onset": _call(cs.h1_foundation_persistence, 
            *_h1_inputs(d_better), "onset", n_boot=1, compute_ci=False
        ),
        "H1_h5": _call(cs.h1_foundation_persistence, 
            *_h1_inputs(d_worse), "h5", n_boot=1, compute_ci=False
        ),
        "H2_NI": h2,
        "H3_derecursion": _call(cs.h3_camor_derecursion, 
            np.full(n, 99.0),
            np.full(n, 100.0),
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, False, False),
            n_boot=1,
            compute_ci=False,
        ),
    }
    return _build_report(
        results, multisensory_test_n=16, h2_calibration_pass=True
    )


class TestH2SeedSignConsistency:
    """r10: the H2 slot computes the frozen 15 ms seed-sign filter."""

    def test_slot_computes_consistent_signs_relative_to_15ms(self) -> None:
        n = 16
        seeds = np.full((3, n), 10.0)  # raw difference 0 < 15 ms
        result = _h2_result(n, nsmor_by_seed=seeds)
        assert result.effect["seed_sign_consistent"] is True
        assert result.effect["seed_sign_reference"] == cs.DELTA_NI_MS
        assert result.effect["seed_sign_construction"] == (
            "per_seed_max_over_models_median_difference"
        )

    def test_slot_reports_inconsistent_seeds(self) -> None:
        n = 16
        seeds = np.full((3, n), 10.0)
        seeds[1] = 30.0  # raw difference 20 ms > 15 ms
        result = _h2_result(n, nsmor_by_seed=seeds)
        assert result.effect["seed_sign_consistent"] is False

    def test_missing_per_seed_arrays_yield_no_evidence(self) -> None:
        result = _h2_result(16)
        assert result.effect["seed_sign_consistent"] is None
        assert result.effect["seed_sign_reason"] == (
            "per_seed_arrays_missing"
        )

    def test_missing_evidence_withholds_a_positive_claim(self) -> None:
        report = _h2_family_report(_h2_result(16))
        assert report["slots"]["H2_NI"]["claim_level"] == (
            "indeterminate_unstable_across_seeds"
        )

    def test_consistent_evidence_allows_confirmatory(self) -> None:
        seeds = np.full((3, 16), 10.0)
        report = _h2_family_report(
            _h2_result(16, nsmor_by_seed=seeds)
        )
        assert report["slots"]["H2_NI"]["claim_level"] == "confirmatory"

    def test_caller_entry_cannot_override_slot_evidence(self) -> None:
        seeds = np.full((3, 16), 10.0)
        seeds[1] = 30.0
        h2 = _h2_result(16, nsmor_by_seed=seeds)
        assert h2.effect["seed_sign_consistent"] is False
        n = 16
        d_better = np.column_stack(
            [np.zeros(n), np.full(n, -10.0), np.zeros(n), np.zeros(n)]
        )
        d_worse = np.column_stack(
            [np.zeros(n), np.full(n, 10.0), np.zeros(n), np.zeros(n)]
        )
        results = {
            "H1_onset": _call(cs.h1_foundation_persistence, 
                *_h1_inputs(d_better), "onset", n_boot=1, compute_ci=False
            ),
            "H1_h5": _call(cs.h1_foundation_persistence, 
                *_h1_inputs(d_worse), "h5", n_boot=1, compute_ci=False
            ),
            "H2_NI": h2,
            "H3_derecursion": cs.descriptive_placeholder(
                "H3_derecursion", n_animal=n
            ),
        }
        with pytest.raises(ValueError):
            _build_report(
                results,
                multisensory_test_n=16,
                h2_calibration_pass=True,
                seed_consistency={"H2_NI": {"consistent": True}},
            )

    def test_h2_caller_entry_cannot_substitute_missing_evidence(
        self,
    ) -> None:
        h2 = _h2_result(16)
        n = 16
        results = {
            "H1_onset": cs.descriptive_placeholder(
                "H1_onset", n_animal=n
            ),
            "H1_h5": cs.descriptive_placeholder(
                "H1_h5", n_animal=n
            ),
            "H2_NI": h2,
            "H3_derecursion": cs.descriptive_placeholder(
                "H3_derecursion", n_animal=n
            ),
        }
        with pytest.raises(ValueError):
            _build_report(
                results,
                multisensory_test_n=16,
                h2_calibration_pass=True,
                seed_consistency={"H2_NI": {"consistent": True}},
            )


class TestSeedConsistencyReportValidation:
    """r10: H1 entries carry provenance; H3 conflicts raise."""

    def _results(self) -> dict[str, cs.TestResult]:
        n = 20
        d_better = np.column_stack(
            [np.zeros(n), np.full(n, -10.0), np.zeros(n), np.zeros(n)]
        )
        d_worse = np.column_stack(
            [np.zeros(n), np.full(n, 10.0), np.zeros(n), np.zeros(n)]
        )
        return {
            "H1_onset": _call(cs.h1_foundation_persistence, 
                *_h1_inputs(d_better), "onset", n_boot=1, compute_ci=False
            ),
            "H1_h5": _call(cs.h1_foundation_persistence, 
                *_h1_inputs(d_worse), "h5", n_boot=1, compute_ci=False
            ),
            "H2_NI": cs.descriptive_placeholder("H2_NI", n_animal=n),
            "H3_derecursion": _call(cs.h3_camor_derecursion, 
                *_h3_inputs(np.full(n, -1.0)),
                budget_matched_by_seed=(True, True, True),
                oom_by_seed=(False, False, False),
                n_boot=1,
                compute_ci=False,
            ),
        }

    def _h1_entry(self, results, scope: str, winning_arm: str | None = None):
        d = np.column_stack(
            [
                np.zeros(20),
                np.full(20, -10.0 if scope == "onset" else 10.0),
                np.zeros(20),
                np.zeros(20),
            ]
        )
        by_arm, persistence = _h1_inputs(d)
        entry = _call(cs.h1_seed_sign_consistency, 
            {arm: np.tile(values, (3, 1)) for arm, values in by_arm.items()},
            np.tile(persistence, (3, 1)),
            scope,
        )
        if winning_arm is not None:
            entry = dict(entry)
            entry["winning_arm"] = winning_arm
        return entry

    def test_h1_bare_bool_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            _build_report(
                self._results(),
                multisensory_test_n=13,
                h2_calibration_pass=True,
                seed_consistency={"H1_onset": {"consistent": True}},
            )

    def test_h1_provenance_is_cross_checked(self) -> None:
        results = self._results()
        entry = self._h1_entry(results, "onset")
        wrong_arm = "R0" if entry["winning_arm"] != "R0" else "R2"
        with pytest.raises(ValueError):
            _build_report(
                results,
                multisensory_test_n=13,
                h2_calibration_pass=True,
                seed_consistency={
                    "H1_onset": self._h1_entry(
                        results, "onset", winning_arm=wrong_arm
                    )
                },
            )
        with pytest.raises(ValueError):
            _build_report(
                results,
                multisensory_test_n=13,
                h2_calibration_pass=True,
                seed_consistency={"H1_onset": self._h1_entry(results, "h5")},
            )
        report = _build_report(
            results,
            multisensory_test_n=13,
            h2_calibration_pass=True,
            seed_consistency={"H1_onset": entry},
        )
        assert report["slots"]["H1_onset"]["claim_level"] == "confirmatory"

    def test_h3_conflicting_caller_entry_raises(self) -> None:
        n = 20
        results = self._results()
        c16 = np.full((3, n), 100.0)
        c1s = np.full((3, n), 90.0)
        c1s[1] = 110.0  # mixed seed signs
        results["H3_derecursion"] = _call(cs.h3_camor_derecursion, 
            c1s.mean(axis=0),
            c16.mean(axis=0),
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, False, False),
            mse_c1s_by_seed=c1s,
            mse_c16_by_seed=c16,
            n_boot=1,
            compute_ci=False,
        )
        assert results["H3_derecursion"].effect[
            "seed_sign_consistent"
        ] is False
        with pytest.raises(ValueError):
            _build_report(
                results,
                multisensory_test_n=13,
                h2_calibration_pass=True,
                seed_consistency={"H3_derecursion": {"consistent": True}},
            )


class TestH2ClaimLevelIndependent:
    """r10: worst-case failure is its own indeterminate state."""

    def test_worst_case_failure_is_not_confirmatory_no_rejection(self) -> None:
        n = 16
        mask = np.zeros(n, dtype=bool)
        mask[:10] = True
        e_ns = np.full(n, 5.0)
        cls = _h2_classical(np.full((n, 4), 40.0))
        h2 = _call(cs.h2_non_inferior_best_classical, 
            e_ns,
            cls,
            analysis_set=cs._analysis_set_from_mask(mask),
            n_boot=1,
            compute_ci=False,
        )
        assert h2.effect["worst_case_p_value"] > cs.ALPHA / cs.FAMILY_M
        report = _h2_family_report(h2)
        slot = report["slots"]["H2_NI"]
        assert slot["significant"] is True
        assert slot["claim_level"] == (
            "indeterminate_informative_missingness"
        )
        assert slot["effect"]["worst_case_gate_pass"] is False

    def test_worst_case_pass_allows_confirmatory(self) -> None:
        h2 = _h2_result(16)
        report = _h2_family_report(h2)
        assert report["slots"]["H2_NI"]["effect"][
            "worst_case_gate_pass"
        ] is True
        assert report["slots"]["H2_NI"]["claim_level"] == (
            "indeterminate_unstable_across_seeds"
        )


class TestAnimalKeyValidation:
    """r10: slot entries validate the animal-level key array."""

    def test_valid_keys_do_not_change_statistics(self) -> None:
        n = 20
        d = np.column_stack(
            [np.zeros(n), np.full(n, -10.0), np.zeros(n), np.zeros(n)]
        )
        by_arm, persistence = _h1_inputs(d)
        keys = [f"a{i}" for i in range(n)]
        base = _call(cs.h1_foundation_persistence, 
            by_arm, persistence, "onset", n_boot=1, compute_ci=False
        )
        keyed = _call(cs.h1_foundation_persistence, 
            by_arm,
            persistence,
            "onset",
            n_boot=1,
            compute_ci=False,
            animal_key=keys,
        )
        assert keyed.p_value == base.p_value
        e_ns = np.full(n, 5.0)
        cls = _h2_classical(np.full((n, 4), 40.0))
        aset = cs._analysis_set_from_mask(np.ones(n, dtype=bool))
        h2_base = _call(cs.h2_non_inferior_best_classical, 
            e_ns, cls, analysis_set=aset, n_boot=1, compute_ci=False
        )
        h2_keyed = _call(cs.h2_non_inferior_best_classical, 
            e_ns,
            cls,
            analysis_set=aset,
            n_boot=1,
            compute_ci=False,
            animal_key=keys,
        )
        assert h2_keyed.p_value == h2_base.p_value
        h3_base = _call(cs.h3_camor_derecursion, 
            *_h3_inputs(np.full(n, -1.0)),
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, False, False),
            n_boot=1,
            compute_ci=False,
        )
        h3_keyed = _call(cs.h3_camor_derecursion, 
            *_h3_inputs(np.full(n, -1.0)),
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, False, False),
            n_boot=1,
            compute_ci=False,
            animal_key=keys,
        )
        assert h3_keyed.p_value == h3_base.p_value
        entry_base = _call(cs.h1_seed_sign_consistency, 
            {arm: np.tile(v, (3, 1)) for arm, v in by_arm.items()},
            np.tile(persistence, (3, 1)),
            "onset",
        )
        entry_keyed = _call(cs.h1_seed_sign_consistency, 
            {arm: np.tile(v, (3, 1)) for arm, v in by_arm.items()},
            np.tile(persistence, (3, 1)),
            "onset",
            animal_key=keys,
        )
        assert entry_keyed["consistent"] == entry_base["consistent"]

    @pytest.mark.parametrize("entry", ["h1", "h2", "h3", "h1seed"])
    def test_duplicate_or_mismatched_keys_raise(self, entry: str) -> None:
        n = 12
        bad = ["a"] * n
        short = [f"a{i}" for i in range(n - 1)]
        d = np.column_stack(
            [np.zeros(n), np.full(n, -10.0), np.zeros(n), np.zeros(n)]
        )
        by_arm, persistence = _h1_inputs(d)
        if entry == "h1":
            call = lambda keys: _call(cs.h1_foundation_persistence, 
                by_arm,
                persistence,
                "onset",
                n_boot=1,
                compute_ci=False,
                animal_key=keys,
            )
        elif entry == "h2":
            aset = cs._analysis_set_from_mask(np.ones(n, dtype=bool))
            call = lambda keys: _call(cs.h2_non_inferior_best_classical, 
                np.full(n, 5.0),
                _h2_classical(np.full((n, 4), 40.0)),
                analysis_set=aset,
                n_boot=1,
                compute_ci=False,
                animal_key=keys,
            )
        elif entry == "h3":
            call = lambda keys: _call(cs.h3_camor_derecursion, 
                *_h3_inputs(np.full(n, -1.0)),
                budget_matched_by_seed=(True, True, True),
                oom_by_seed=(False, False, False),
                n_boot=1,
                compute_ci=False,
                animal_key=keys,
            )
        else:
            call = lambda keys: _call(cs.h1_seed_sign_consistency, 
                {arm: np.tile(v, (3, 1)) for arm, v in by_arm.items()},
                np.tile(persistence, (3, 1)),
                "onset",
                animal_key=keys,
            )
        with pytest.raises(ValueError):
            call(bad)
        with pytest.raises(ValueError):
            call(short)


class TestActiveOrderRule:
    def test_fail_closed_default_excludes_h2(self) -> None:
        assert cs.FAIL_CLOSED_ACTIVE_ORDER == (
            "H1_onset",
            "H1_h5",
            "H3_derecursion",
        )
        assert "H2_NI" not in cs.FAIL_CLOSED_ACTIVE_ORDER

    def test_h2_enters_only_with_split_n_and_calibration_pass(self) -> None:
        assert cs.resolve_active_confirmatory_order(20, True) == (
            cs.FAMILY_ORDER
        )
        assert cs.resolve_active_confirmatory_order(15, True) == (
            cs.FAMILY_ORDER
        )
        assert cs.resolve_active_confirmatory_order(13, True) == (
            cs.FAIL_CLOSED_ACTIVE_ORDER
        )
        assert cs.resolve_active_confirmatory_order(20, False) == (
            cs.FAIL_CLOSED_ACTIVE_ORDER
        )
        assert cs.resolve_active_confirmatory_order(None, True) == (
            cs.FAIL_CLOSED_ACTIVE_ORDER
        )
        assert cs.resolve_active_confirmatory_order(20, None) == (
            cs.FAIL_CLOSED_ACTIVE_ORDER
        )

    def test_invalid_types_fail_closed(self) -> None:
        # A malformed split fact must never promote H2 and must not
        # abort the unblinding report; it resolves to the fail-closed
        # order instead.
        assert cs.resolve_active_confirmatory_order(
            True, True  # type: ignore[arg-type]
        ) == cs.FAIL_CLOSED_ACTIVE_ORDER
        assert cs.resolve_active_confirmatory_order(
            "20", True  # type: ignore[arg-type]
        ) == cs.FAIL_CLOSED_ACTIVE_ORDER
        assert cs.resolve_active_confirmatory_order(
            20, 1  # type: ignore[arg-type]
        ) == cs.FAIL_CLOSED_ACTIVE_ORDER
        assert cs.resolve_active_confirmatory_order(
            20.0, True  # type: ignore[arg-type]
        ) == cs.FAIL_CLOSED_ACTIVE_ORDER

    def test_h2_calibration_constants_are_frozen(self) -> None:
        assert cs.H2_CALIBRATION_R >= 200_000
        assert cs.H2_CALIBRATION_SEED not in (cs.SEED_BASE, 20261119)
        assert cs.H2_MIN_TEST_N == 15
        assert cs.TYPE1_CP95_GATE == pytest.approx(0.0175)
        assert cs.FRAGILITY_MAX_REMOVE >= 1

    def test_planned_h1_h3_n_matches_frozen_split(self) -> None:
        # The frozen PG-02 split (SHA 380c81de...) has 18 test animals:
        # 7 looming_wind strata x 2 = 14 multisensory + 4 bv; bw is 0.
        # The r9 formula-based 16 (6 x 2 + 4) is retained only as a
        # sensitivity grid point.
        assert cs.N_PLAN_H1_H3 == 18
        assert cs.N_PLAN_H1_H3_RANGE == (13, 16, 18, 20)
        assert cs.N_PLAN_H1_H3 in cs.N_PLAN_H1_H3_RANGE

    def test_holm_downgrades_h2_when_split_n_is_below_15(self) -> None:
        p = {name: 1.0 for name in cs.FAMILY_ORDER}
        p["H2_NI"] = 0.001
        with pytest.raises(ValueError):
            cs.holm_confirmatory_family(
                p, multisensory_test_n=13, h2_calibration_pass=True
            )
        with pytest.raises(ValueError):
            cs.holm_confirmatory_family(
                p, multisensory_test_n=20, h2_calibration_pass=False
            )
        holm = cs.holm_confirmatory_family(
            p, multisensory_test_n=20, h2_calibration_pass=True
        )
        assert holm.significant["H2_NI"] is True


class TestLooFragilityIndex:
    def _baseline(
        self, p_h1: float, n: int = 10
    ) -> tuple[dict[str, cs.TestResult], dict[str, float]]:
        results = {
            "H1_onset": cs.TestResult(
                name="H1_onset",
                statistic=0.0,
                boundary=0.0,
                p_value=p_h1,
                n_animal=n,
                n_boot=1,
                seed=1,
                available=True,
                reason="",
            ),
            "H1_h5": cs.descriptive_placeholder("H1_h5", n_animal=n),
            "H2_NI": cs.descriptive_placeholder("H2_NI", n_animal=n),
            "H3_derecursion": cs.descriptive_placeholder(
                "H3_derecursion", n_animal=n
            ),
        }
        return results, {
            name: result.p_value for name, result in results.items()
        }

    def _index(
        self,
        results: dict[str, cs.TestResult],
        recompute,
        max_remove: int,
    ):
        return cs.loo_fragility_index(
            results,
            recompute,
            multisensory_test_n=13,
            h2_calibration_pass=True,
            max_remove=max_remove,
        )

    def test_single_deletion_flip_rate(self) -> None:
        results, baseline = self._baseline(2.0 ** -10)

        def recompute(slot: str, keep: np.ndarray) -> float:
            assert slot == "H1_onset"
            return 1.0 if not keep[0] else baseline["H1_onset"]

        frag = self._index(results, recompute, 1)
        assert "H1_onset" in frag
        entry = frag["H1_onset"]
        assert entry.baseline_significant is True
        assert entry.available is True
        assert entry.n_single_deletion_flips == 1
        assert entry.loo_flip_rate == pytest.approx(0.1)
        assert entry.min_animals_to_flip == 1
        assert entry.censored is False
        assert entry.construction == "descriptive_loo_fragility"

    def test_min_animals_to_flip_finds_a_pair(self) -> None:
        results, _ = self._baseline(0.012)

        def recompute(slot: str, keep: np.ndarray) -> float:
            assert slot == "H1_onset"
            if not keep[0] and not keep[1]:
                return 1.0
            return 0.012

        frag = self._index(results, recompute, 2)
        entry = frag["H1_onset"]
        assert entry.min_animals_to_flip == 2
        assert entry.n_single_deletion_flips == 0
        assert entry.censored is False

    def test_censored_when_no_small_subset_flips(self) -> None:
        results, _ = self._baseline(0.012)

        def recompute(slot: str, keep: np.ndarray) -> float:
            assert slot == "H1_onset"
            return 0.012 if keep.any() else 1.0

        frag = self._index(results, recompute, 1)
        entry = frag["H1_onset"]
        assert entry.min_animals_to_flip is None
        assert entry.censored is True

    def test_active_order_cannot_be_supplied(self) -> None:
        results, _ = self._baseline(1.0)
        with pytest.raises(TypeError):
            cs.loo_fragility_index(
                results,
                lambda slot, keep: 1.0,
                active_order=("H1_onset",),  # type: ignore[call-arg]
                multisensory_test_n=13,
                h2_calibration_pass=True,
            )

    def test_recompute_failure_is_per_slot(self) -> None:
        results, _ = self._baseline(1.0)

        def recompute(slot: str, keep: np.ndarray) -> float:
            raise RuntimeError("boom")

        frag = self._index(results, recompute, 1)
        entry = frag["H1_onset"]
        assert entry.available is False
        assert entry.reason.startswith("loo_recompute_failed")


class TestSeedSignConsistency:
    def test_h2_reference_is_15(self) -> None:
        assert cs.seed_sign_consistency(
            np.array([-1.0, 2.0, 4.0]), reference=15.0
        )
        assert not cs.seed_sign_consistency(
            np.array([14.0, 16.0, 20.0]), reference=15.0
        )

    def test_zero_vector_is_not_improvement(self) -> None:
        # r11: a tie is not the improvement side; the conservative
        # tie policy withholds the positive claim.
        assert not cs.seed_sign_consistency(np.zeros(3))

    def test_mixed_zero_and_nonzero_is_inconsistent(self) -> None:
        assert not cs.seed_sign_consistency(np.array([0.0, 1.0, 2.0]))


class TestR11ClaimGateHardening:
    """r11 review: the claim gate itself must be correct.

    A positive confirmatory claim requires every training seed to lie
    on the *improvement* side of the slot boundary.  Three seeds that
    agree on the wrong side are unstable, not confirmatory.
    """

    def test_seed_sign_consistency_requires_improvement_side(self) -> None:
        assert cs.seed_sign_consistency(np.array([-2.0, -1.0, -3.0]))
        assert not cs.seed_sign_consistency(np.array([1.0, 1.0, 1.0]))
        assert not cs.seed_sign_consistency(np.array([1.0, -10.0, 1.0]))
        assert not cs.seed_sign_consistency(np.array([0.0, 0.0, 0.0]))
        assert not cs.seed_sign_consistency(
            np.array([16.0, 20.0, 30.0]), reference=cs.DELTA_NI_MS
        )
        assert cs.seed_sign_consistency(
            np.array([14.0, 14.0, 14.0]), reference=cs.DELTA_NI_MS
        )

    def test_rotated_wrong_side_seeds_block_confirmatory_claim(
        self,
    ) -> None:
        # Minimal repro from the r11 review: n=18, each animal has two
        # seeds at D=+1 and one seed at D=-10.  Every seed's median
        # difference is +1 (each seed alone says most animals worsen),
        # yet the seed mean is negative and the sign test is tiny.
        n = 18
        d_by_seed = np.full((3, n), 1.0)
        for seed_index in range(3):
            d_by_seed[seed_index, np.arange(n) % 3 == seed_index] = -10.0
        per_seed_medians = np.median(d_by_seed, axis=1)
        assert np.all(per_seed_medians > 0.0)
        assert not cs.seed_sign_consistency(per_seed_medians)

        d_mean = d_by_seed.mean(axis=0)
        assert np.all(d_mean < 0.0)
        persistence = np.full(n, 100.0)
        by_arm_by_seed = {
            arm: persistence[None, :] + d_by_seed for arm in ARMS
        }
        d_avg = np.column_stack([d_mean for _ in ARMS])
        h1_inputs = _h1_inputs(d_avg)
        animal_key = [f"animal-{index}" for index in range(n)]
        h1_onset = _call(cs.h1_foundation_persistence, 
            *h1_inputs,
            "onset",
            animal_key=animal_key,
            persistence_baseline="v_t",
            n_boot=1,
            compute_ci=False,
        )
        h1_h5 = _call(cs.h1_foundation_persistence, 
            *h1_inputs,
            "h5",
            animal_key=animal_key,
            persistence_baseline="v_t",
            n_boot=1,
            compute_ci=False,
        )
        assert h1_onset.p_value == pytest.approx(2.0 ** -16, rel=1e-12)
        onset_entry = _call(cs.h1_seed_sign_consistency, 
            by_arm_by_seed,
            np.full((3, n), 100.0),
            "onset",
            animal_key=animal_key,
            persistence_baseline="v_t",
        )
        h5_entry = _call(cs.h1_seed_sign_consistency, 
            by_arm_by_seed,
            np.full((3, n), 100.0),
            "h5",
            animal_key=animal_key,
            persistence_baseline="v_t",
        )
        assert onset_entry["consistent"] is False
        assert h5_entry["consistent"] is False
        c16_by_seed = np.full((3, n), 100.0)
        c1s_by_seed = c16_by_seed + d_by_seed
        h3 = _call(cs.h3_camor_derecursion, 
            c1s_by_seed.mean(axis=0),
            c16_by_seed.mean(axis=0),
            budget_matched_by_seed=(True, True, True),
            oom_by_seed=(False, False, False),
            mse_c1s_by_seed=c1s_by_seed,
            mse_c16_by_seed=c16_by_seed,
            animal_key=animal_key,
            n_boot=1,
            compute_ci=False,
        )
        assert h3.p_value == pytest.approx(2.0 ** -18, rel=1e-12)
        assert h3.effect["seed_sign_consistent"] is False
        results = {
            "H1_onset": h1_onset,
            "H1_h5": h1_h5,
            "H2_NI": cs.descriptive_placeholder("H2_NI", n_animal=n),
            "H3_derecursion": h3,
        }
        report = _build_report(
            results,
            multisensory_test_n=18,
            h2_calibration_pass=True,
            split_sha256="synthetic:r11-repro",
            seed_consistency={
                "H1_onset": onset_entry,
                "H1_h5": h5_entry,
            },
        )
        for slot in ("H1_onset", "H1_h5", "H3_derecursion"):
            assert (
                report["slots"][slot]["claim_level"]
                == "indeterminate_unstable_across_seeds"
            )
        assert report["family"]["h1_joint_gate"]["pass"] is False

    def test_confirmatory_slots_require_animal_key(self) -> None:
        for function in (
            cs.h1_foundation_persistence,
            cs.h1_seed_sign_consistency,
            cs.h2_non_inferior_best_classical,
            cs.h3_camor_derecursion,
        ):
            parameter = inspect.signature(function).parameters["animal_key"]
            assert parameter.default is inspect.Parameter.empty, function
        with pytest.raises(ValueError, match="animal_key"):
            cs._check_animal_key(None, 3, "animal_key")

    def test_h1_requires_v_t_persistence_baseline(self) -> None:
        n = 12
        by_arm, persistence = _h1_inputs(np.zeros((n, len(ARMS))))
        animal_key = [f"animal-{index}" for index in range(n)]
        for function in (
            cs.h1_foundation_persistence,
            cs.h1_seed_sign_consistency,
        ):
            parameter = inspect.signature(function).parameters[
                "persistence_baseline"
            ]
            assert parameter.default is inspect.Parameter.empty, function
        with pytest.raises(ValueError, match="v_t"):
            _call(cs.h1_foundation_persistence, 
                by_arm,
                persistence,
                "h5",
                animal_key=animal_key,
                persistence_baseline="v_t_minus_1",
                n_boot=1,
                compute_ci=False,
            )

    def test_report_binds_split_sha_and_planned_n(self) -> None:
        assert cs.FROZEN_SPLIT_SHA256 == (
            "380c81de8232de813624ab119877701e06bc134270827c74d9caee7ad0afde59"
        )
        assert cs.FROZEN_MULTISENSORY_TEST_N == 14
        assert cs.N_PLAN_H1_H3 == 18
        parameter = inspect.signature(
            cs.confirmatory_report
        ).parameters["split_sha256"]
        assert parameter.default is inspect.Parameter.empty
        with pytest.raises(ValueError, match="frozen split"):
            _build_report(
                [],
                multisensory_test_n=15,
                h2_calibration_pass=True,
                split_sha256=cs.FROZEN_SPLIT_SHA256,
            )
        with pytest.raises(ValueError, match="unknown split"):
            _build_report(
                [],
                multisensory_test_n=18,
                h2_calibration_pass=True,
                split_sha256="deadbeef",
            )


# ---------------------------------------------------------------------------
# G1 and descriptive statistics
# ---------------------------------------------------------------------------


class TestG1Existence:
    def test_constant_ranks_are_not_evaluable(self) -> None:
        x = np.arange(10, dtype=np.float64)
        y = np.ones(10, dtype=np.float64)
        result = cs.g1_existence_permutation(x, y, n_perm=100)
        assert result.available is False
        assert result.exists is False
        assert "constant ranks" in result.reason

    def test_strong_monotone_relation_exists(self) -> None:
        x = np.arange(10, dtype=np.float64)
        y = x ** 2
        result = cs.g1_existence_permutation(x, y, n_perm=2_000)
        assert result.available is True
        assert result.rho_observed == pytest.approx(1.0)
        assert result.exists is True

    def test_null_relation_not_detected(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=12)
        y = rng.normal(size=12)
        result = cs.g1_existence_permutation(x, y, n_perm=2_000)
        assert result.available is True
        assert result.exists is False


class TestDescriptiveStatistics:
    def test_summary_declares_no_multiplicity_control(self) -> None:
        row = cs.descriptive_summary(
            point=1.0,
            ci_low=0.5,
            ci_high=1.5,
            n_animal=12,
            n_trial=120,
            effect_size=0.4,
        )
        assert row["inference"] == "descriptive"
        assert row["p_value"] is None
        assert row["multiplicity_control"] == "none"
        assert row["claim_limit"] == (
            "no superiority/non-inferiority/equivalence claim"
        )
        assert row["available"] is True

    def test_unavailable_row_has_no_ci(self) -> None:
        row = cs.descriptive_summary(
            point=None,
            ci_low=None,
            ci_high=None,
            n_animal=0,
            n_trial=0,
            effect_size=None,
            reason="n_lt_10",
        )
        assert row["available"] is False
        assert row["reason"] == "n_lt_10"

    def test_inconsistent_row_raises(self) -> None:
        with pytest.raises(ValueError):
            cs.descriptive_summary(
                point=None,
                ci_low=0.0,
                ci_high=1.0,
                n_animal=12,
                n_trial=120,
                effect_size=0.4,
            )

    def test_bootstrap_ci_is_deterministic(self) -> None:
        values = np.linspace(-1.0, 1.0, 12)
        a = cs.animal_cluster_bootstrap_ci(values, n_boot=500, seed=3)
        b = cs.animal_cluster_bootstrap_ci(values, n_boot=500, seed=3)
        assert a == pytest.approx(b)
        assert a[1] <= a[0] <= a[2]


class TestLegacyBootstrapShift:
    def test_single_candidate_direction(self) -> None:
        values = -2.0 + 0.01 * np.arange(12, dtype=np.float64)
        result = cs.animal_cluster_bootstrap_shift(
            values, boundary=0.0, n_boot=500, seed=1
        )
        assert result.available is True
        assert result.p_value < 0.05

    def test_zero_variance_fails_closed(self) -> None:
        result = cs.animal_cluster_bootstrap_shift(
            np.zeros(12), boundary=0.0, n_boot=100, seed=1
        )
        assert not result.available
        assert result.p_value == 1.0
        assert "degenerate" in result.reason


# ---------------------------------------------------------------------------
# Calibration: scenario generators
# ---------------------------------------------------------------------------


class TestCalibrationGenerators:
    @pytest.mark.parametrize("scenario", ["S0", "S1", "S2", "S4", "S6"])
    def test_median_shift_matches_requested_probability(
        self, scenario: str
    ) -> None:
        rng = np.random.default_rng(11)
        n_seed = 3 if scenario == "S6" else 1
        draws = cc.h1_draws(
            rng,
            n=200,
            n_sim=300,
            scenario=scenario,
            p_improve=0.7,
            n_seed=n_seed,
        )
        assert draws.shape == (300, 200, 4)
        proportion = float(np.mean(draws[:, :, 1] < 0.0))
        assert proportion == pytest.approx(0.7, abs=0.02)
        assert np.median(draws[:, :, 1]) < 0.0

    def test_s4_variance_ratio(self) -> None:
        rng = np.random.default_rng(12)
        draws = cc.h1_draws(
            rng, n=2000, n_sim=1, scenario="S4", p_improve=0.5
        )
        values = draws[0, :, 1]
        ratio = np.var(values[:1000]) / np.var(values[1000:])
        assert ratio == pytest.approx(4.0, rel=0.1)

    def test_s6_seed_mean_variance(self) -> None:
        rng = np.random.default_rng(13)
        draws = cc.h1_draws(
            rng,
            n=2000,
            n_sim=1,
            scenario="S6",
            p_improve=0.5,
            n_seed=3,
        )
        expected = 10.0 * math.sqrt(1.0 + 0.09 / 3.0)
        assert np.std(draws[0, :, 1]) == pytest.approx(expected, rel=0.05)

    def test_h3_draws_shape(self) -> None:
        rng = np.random.default_rng(14)
        draws = cc.h3_draws(
            rng, n=200, n_sim=100, scenario="S1", p_improve=0.6
        )
        assert draws.shape == (100, 200)
        assert float(np.mean(draws < 0.0)) == pytest.approx(0.6, abs=0.02)

    def test_h2_draws_all_seed_rule_and_exclusions(self) -> None:
        rng = np.random.default_rng(15)
        data = cc.h2_realistic_draws(
            rng,
            n=15,
            n_sim=100,
            scenario="S1",
            delta_best=15.0,
        )
        assert data.l_obs.shape == (100, 15)
        assert data.l_nsmor_by_seed.shape == (100, 15, 3)
        assert data.e_cls.shape == (100, 15, 4)
        assert data.analysis_mask.shape == (100, 15)
        # The all-seed mask is a strict subset of the observed-available
        # animals: some animals have 1-2 missing seeds.
        assert data.all_seed_mask.sum() < 100 * 15
        # The analysis set is a subset of the all-seed mask.
        assert np.all(data.analysis_mask <= data.all_seed_mask)
        # The exclusion hierarchy sums to the attrition in every
        # replicate, and the aggregate does too.
        total = 100 * 15
        assert sum(data.excluded.values()) == total - int(
            data.n_available.sum()
        )
        assert np.all(data.n_available <= data.all_seed_mask.sum(axis=1))
        assert np.all(np.isfinite(data.true_delta_best))

    def test_h2_informative_missing_reduces_availability(self) -> None:
        base = cc.h2_realistic_draws(
            np.random.default_rng(16),
            15,
            200,
            "S5",
            15.0,
            informative_missing=False,
        )
        selected = cc.h2_realistic_draws(
            np.random.default_rng(16),
            15,
            200,
            "S5",
            15.0,
            informative_missing=True,
        )
        assert selected.n_available.mean() < base.n_available.mean()


# ---------------------------------------------------------------------------
# Calibration: type-I and availability
# ---------------------------------------------------------------------------


class TestCalibrationTypeOne:
    @pytest.mark.parametrize("scenario", ["S0", "S1", "S2"])
    def test_h1_type_one_is_calibrated(self, scenario: str) -> None:
        record = cc.calibrate_h1_null(
            n=20,
            scenario=scenario,
            n_sim=1_000,
            seed=21,
            scope="onset",
        )
        assert record["n_evaluable"] == 1_000
        assert record["type1_cp95_upper"] <= 0.05
        assert 0.0 < record["mean_p"] < 1.0

    @pytest.mark.parametrize("scenario", ["S0", "S1", "S2"])
    def test_h3_type_one_is_calibrated(self, scenario: str) -> None:
        record = cc.calibrate_h3_null(
            n=20, scenario=scenario, n_sim=1_000, seed=22
        )
        assert record["n_evaluable"] == 1_000
        assert record["type1_cp95_upper"] <= 0.05

    def test_h1_ties_are_conservative(self) -> None:
        record = cc.calibrate_h1_null(
            n=20,
            scenario="S1",
            n_sim=1_000,
            seed=23,
            scope="onset",
            p_tie=0.20,
        )
        assert record["type1_cp95_upper"] <= 0.03

    def test_h1_power_uses_four_arm_bonferroni_threshold(self) -> None:
        # The 4-arm union rejects only when min_a p_a <= 0.0125 / 4.
        # At n=20 that requires k >= 17, not the single-arm k >= 16.
        critical_k = min(
            k
            for k in range(21)
            if 4.0 * stats.binom.sf(k - 1, 20, 0.5) <= cc.ALPHA_SLOT
        )
        assert critical_k == 17
        assert stats.binom.sf(16, 20, 0.85) == pytest.approx(
            0.6477, abs=5e-4
        )
        record = cc.calibrate_h1_power(
            20, "S0", 2_000, 20261019, 0.85, scope="onset"
        )
        # Old single-arm table reported ~0.82; the 4-arm union is ~0.65.
        assert record["power"] == pytest.approx(0.65, abs=0.04)
        assert record["power"] < 0.75

    def test_h2_type_one_and_true_delta(self) -> None:
        record = cc.calibrate_h2_realistic(
            n=15, scenario="S1", n_sim=300, seed=24
        )
        assert record["n_evaluable"] > 0
        assert record["type1_cp95_upper"] <= 0.07
        assert record["mean_true_delta_best"] == pytest.approx(
            15.0, abs=1.5
        )
        assert 0.1 < record["p_analysis_set_available"] < 0.8


class TestH2Availability:
    def test_central_n13_matches_per_seed_rule(self) -> None:
        record = cc.calibrate_h2_availability(
            n=13,
            n_sim=20_000,
            seed=25,
            p_obs_defined=0.90,
            p_ns_defined=0.90,
            p_model_ok=0.98,
        )
        # 0.90**3 * 0.90 * 0.98**4 = 0.6052 per animal; at n=13 the
        # probability of n_A >= 10 is 0.1783, not the old 0.5717.
        assert record["p_analysis_set_available"] == pytest.approx(
            0.1783, abs=0.02
        )
        assert record["mean_n_available"] == pytest.approx(7.87, abs=0.2)

    def test_more_animals_increase_availability(self) -> None:
        low = cc.calibrate_h2_availability(
            10, 4_000, 26, 0.90, 0.90, 0.98
        )
        high = cc.calibrate_h2_availability(
            18, 4_000, 27, 0.90, 0.90, 0.98
        )
        assert high["p_analysis_set_available"] > (
            low["p_analysis_set_available"]
        )


# ---------------------------------------------------------------------------
# Calibration: family and LOO
# ---------------------------------------------------------------------------


class TestCalibrationFamily:
    def test_joint_null_controls_fwer(self) -> None:
        record = cc.calibrate_family(
            n=20,
            n_sim=300,
            seed=31,
            alternative=False,
            split_n=13,
            h2_calibration_pass=True,
        )
        assert record["cp95_upper"] <= 0.06
        assert record["h3_h1_realized_correlation"] == pytest.approx(
            0.5, abs=0.08
        )
        assert record["h2_h1_realized_correlation"] == pytest.approx(
            0.418, abs=0.08
        )

    def test_joint_alternative_has_power(self) -> None:
        record = cc.calibrate_family(
            n=20,
            n_sim=300,
            seed=32,
            alternative=True,
            split_n=16,
            h2_calibration_pass=True,
        )
        assert record["fwer_or_power"] > 0.6
        assert record["slot_rejection_rates"]["H2_NI"] > 0.9

    def test_inactive_slots_use_placeholders(self) -> None:
        record = cc.calibrate_family(
            n=20,
            n_sim=100,
            seed=33,
            alternative=False,
            split_n=13,
            h2_calibration_pass=True,
        )
        assert record["slot_rejection_rates"]["H2_NI"] == 0.0
        assert record["active_order"] == list(cs.FAIL_CLOSED_ACTIVE_ORDER)


class TestH2CalibrationCriterion:
    def _record(self, scenario: str, cp_upper: float) -> dict[str, object]:
        return {
            "slot": "H2_NI",
            "scenario": scenario,
            "n_planned": 15,
            "n_sim": 200_000,
            "seed_base": cs.H2_CALIBRATION_SEED,
            "type1_cp95_upper": cp_upper,
            "informative_missing": False,
            "sigma_ens_spread": 0.0,
            "sigma_seed": 0.0,
        }

    def test_pass_requires_every_scenario_including_s1(self) -> None:
        ok = [
            self._record("S0", 0.014),
            self._record("S1", 0.0175),
            self._record("S2", 0.012),
        ]
        verdict = cc.h2_calibration_verdict(ok)
        assert verdict["pass"] is True
        assert verdict["failures"] == []
        assert verdict["r_h2"] == cs.H2_CALIBRATION_R
        assert verdict["seed"] == cs.H2_CALIBRATION_SEED
        assert verdict["gate"] == cs.TYPE1_CP95_GATE

        bad = [
            self._record("S0", 0.014),
            self._record("S1", 0.0176),
            self._record("S2", 0.012),
        ]
        verdict_bad = cc.h2_calibration_verdict(bad)
        assert verdict_bad["pass"] is False
        assert verdict_bad["failures"][0]["scenario"] == "S1"

    def test_missing_scenario_fails_closed(self) -> None:
        verdict = cc.h2_calibration_verdict(
            [self._record("S0", 0.014), self._record("S2", 0.012)]
        )
        assert verdict["pass"] is False
        assert verdict["failures"][0] == {
            "scenario": "S1",
            "reason": "missing_gate_record",
        }

    def test_sensitivity_records_are_not_part_of_the_criterion(self) -> None:
        records = [
            self._record("S0", 0.014),
            self._record("S1", 0.014),
            self._record("S2", 0.012),
        ]
        noisy = dict(self._record("S5", 0.99))
        noisy["sigma_seed"] = 2.0
        records.append(noisy)
        verdict = cc.h2_calibration_verdict(records)
        assert verdict["pass"] is True
        assert verdict["verified"] is True
        assert verdict["r_h2"] == cs.H2_CALIBRATION_R
        assert verdict["seed"] == cs.H2_CALIBRATION_SEED

    def test_auxiliary_replicate_run_is_not_the_gate(self) -> None:
        # A 5,000-replicate auxiliary run with a different seed stream
        # must never be reported as the one-shot 200,000-replicate
        # criterion.
        records = [
            self._record("S0", 0.014),
            self._record("S1", 0.014),
            self._record("S2", 0.012),
        ]
        for record in records:
            record["n_sim"] = 5_000
            record["seed_base"] = 20261230
        verdict = cc.h2_calibration_verdict(records)
        assert verdict["pass"] is False
        assert verdict["verified"] is False
        assert verdict["r_h2"] is None
        assert verdict["seed"] is None
        assert {failure["reason"] for failure in verdict["failures"]} == {
            "n_sim_mismatch"
        }

    def test_seed_mismatch_fails_closed(self) -> None:
        records = [
            self._record("S0", 0.014),
            self._record("S1", 0.014),
            self._record("S2", 0.012),
        ]
        records[1]["seed_base"] = 20261230
        verdict = cc.h2_calibration_verdict(records)
        assert verdict["pass"] is False
        assert verdict["verified"] is False
        assert verdict["r_h2"] is None
        assert verdict["seed"] is None
        assert verdict["failures"] == [
            {
                "scenario": "S1",
                "reason": "seed_base_mismatch",
                "seed_base": 20261230,
                "required_seed_base": cs.H2_CALIBRATION_SEED,
            }
        ]

    def test_missing_provenance_fails_closed(self) -> None:
        records = [
            self._record("S0", 0.014),
            self._record("S1", 0.014),
            self._record("S2", 0.012),
        ]
        for record in records:
            record.pop("seed_base")
        verdict = cc.h2_calibration_verdict(records)
        assert verdict["pass"] is False
        assert verdict["verified"] is False
        assert verdict["r_h2"] is None
        assert verdict["seed"] is None
        assert all(
            failure["reason"] == "seed_base_mismatch"
            for failure in verdict["failures"]
        )


class TestCalibrationLooFragility:
    def test_records_are_descriptive(self) -> None:
        record = cc.calibrate_loo_fragility(
            n=20,
            n_sim=5,
            seed=41,
            alternative=True,
            split_n=13,
            h2_calibration_pass=True,
            max_remove=2,
        )
        assert 0.0 <= record["fragile_rate_any"] <= 1.0
        assert 0.0 <= record["mean_single_flip_fraction"] <= 1.0
        assert record["max_remove"] == 2
        assert set(record["min_animals_to_flip_counts"]) >= {
            "1",
            "2",
            "gt_max",
        }
        # No gate semantics: the descriptive index never reports a
        # power loss or a signed-off gate.
        for banned in (
            "power_loss",
            "power_with_gate",
            "power_without_gate",
            "requires_signoff",
        ):
            assert banned not in record

    def test_default_search_cap_matches_frozen_constant(self) -> None:
        record = cc.calibrate_loo_fragility(
            n=17,
            n_sim=1,
            seed=44,
            alternative=False,
            split_n=None,
            h2_calibration_pass=None,
        )
        assert record["max_remove"] == cs.FRAGILITY_MAX_REMOVE
        assert set(record["min_animals_to_flip_counts"]) == {
            "1",
            "2",
            "3",
            "gt_max",
        }

    def test_null_and_alternative_are_finite(self) -> None:
        null = cc.calibrate_loo_fragility(
            n=20,
            n_sim=10,
            seed=42,
            alternative=False,
            split_n=13,
            h2_calibration_pass=True,
            max_remove=1,
        )
        assert 0.0 <= null["fragile_rate_any"] <= 1.0

    def test_no_active_slots(self) -> None:
        record = cc.calibrate_loo_fragility(
            n=20,
            n_sim=10,
            seed=43,
            alternative=True,
            split_n=13,
            h2_calibration_pass=True,
            max_remove=1,
        )
        # The frozen rule never returns an empty order, so the fragility
        # index always has at least the three fail-closed slots.
        assert record["active_order"] == list(cs.FAIL_CLOSED_ACTIVE_ORDER)
        assert 0.0 <= record["fragile_rate_any"] <= 1.0


# ---------------------------------------------------------------------------
# Calibration suite smoke
# ---------------------------------------------------------------------------


class TestCalibrationSuite:
    def test_quick_suite_shapes_and_verdicts(self) -> None:
        result = cc.run_calibration_suite(
            r_sim=3,
            r_loo=1,
            quick=True,
            h2_split_n=13,
            r_h2=3,
            r_h2_aux=3,
            h2_n_ens=40,
        )
        assert result["schema_version"] == "5.0"
        feasible = result["h2_split_feasibility"]
        assert feasible["h2_split_n"] == 13
        assert feasible["eligible"] is False
        assert feasible["reason"] == "expected_split_n_lt_15"
        assert feasible["calibration_seed"] == cs.H2_CALIBRATION_SEED
        # The frozen rule alone downgrades H2 at n=13, independent of
        # the finite-replicate type-I gate.
        assert "H2_NI" not in result["active_order"]
        assert result["g1_power_table"]["status"] in (
            "unavailable_pending_recompute",
            "computed",
        )
        assert set(result["verdicts"]["type1"]) == set(cs.FAMILY_ORDER)
        assert "h2_calibration" in result["verdicts"]
        # ``quick=True`` caps the H2 cells far below the frozen
        # 200,000-replicate criterion, so the provenance check must
        # fail closed and must not echo the frozen constants.
        assert result["verdicts"]["h2_calibration"]["verified"] is False
        assert result["verdicts"]["h2_calibration"]["pass"] is False
        assert result["verdicts"]["h2_calibration"]["r_h2"] is None
        assert result["verdicts"]["h2_calibration"]["seed"] is None
        assert len(result["type1"]) > 0
        assert len(result["power"]) > 0
        assert len(result["family"]) == 2
        assert len(result["loo_fragility"]) == 2
        assert "loo" not in result
        assert "loo_counterfactual" not in result
        # r8: power is calibrated at the planned n=17 (n=13/20 remain
        # sensitivity points), and every power record carries the
        # explicit underpowered flag.
        assert any(
            record["n"] == cs.N_PLAN_H1_H3
            for record in result["power"]
            if record["slot"].startswith("H1_")
        )
        assert any(
            record["n"] == cs.N_PLAN_H1_H3
            for record in result["power"]
            if record["slot"] == "H3_derecursion"
        )
        assert all(
            "underpowered_at_target" in record
            for record in result["power"]
        )
        assert result["family"][0]["n"] == cs.N_PLAN_H1_H3
        assert (
            result["loo_fragility"][0]["max_remove"]
            == cs.FRAGILITY_MAX_REMOVE
        )

    def test_g1_operating_characteristics_use_24_datasets(self) -> None:
        result = cc.run_calibration_suite(
            r_sim=2,
            r_loo=1,
            quick=True,
            h2_split_n=13,
            r_h2=2,
            r_h2_aux=2,
            h2_n_ens=40,
        )
        g1 = result["g1_power_table"]
        assert g1["status"] == "computed"
        assert g1["n_datasets"] == 24
        assert g1["threshold_k_min"] == 9
        assert g1["k_min"] == 9
        assert g1["clopper_pearson_lower_at_threshold"] == pytest.approx(
            float(stats.beta.ppf(0.05, 9, 16)), rel=1e-12
        )
        rows = {
            row["true_fpr"]: row for row in g1["operating_characteristics"]
        }
        assert rows[0.35]["p_flag_gross_error"] == pytest.approx(
            float(stats.binom.sf(8, 24, 0.35)), rel=1e-12
        )
        assert rows[0.50]["p_flag_gross_error"] == pytest.approx(
            float(stats.binom.sf(8, 24, 0.50)), rel=1e-12
        )
        assert rows[0.20]["p_flag_gross_error"] == pytest.approx(
            0.0362, abs=5e-5
        )
        assert rows[0.35]["true_dataset_fpr"] == 0.35
        assert rows[0.35]["alarm_power_k_ge_9"] == rows[0.35][
            "p_flag_gross_error"
        ]

    def test_h2_split_15_follows_the_frozen_rule(self) -> None:
        result = cc.run_calibration_suite(
            r_sim=3,
            r_loo=1,
            quick=True,
            h2_split_n=15,
            r_h2=3,
            r_h2_aux=3,
            h2_n_ens=40,
        )
        # At n=15 the split gate no longer downgrades H2; eligibility
        # then follows the frozen rule function and the H2 calibration
        # verdict, and the active order can only contain H2 when both
        # the rule and the slot's own type-I gate pass.
        feasible = result["h2_split_feasibility"]
        calibration_pass = result["verdicts"]["h2_calibration"]["pass"]
        expected_eligible = "H2_NI" in cs.resolve_active_confirmatory_order(
            15, calibration_pass
        )
        assert feasible["eligible"] is expected_eligible
        expected_reason = "" if calibration_pass else "h2_calibration_failed"
        assert feasible["reason"] == expected_reason
        # r9: the active order is the frozen rule's output alone; the
        # type-I gate is a calibration verdict on the slot, not an
        # input to the order rule.
        assert result["active_order"] == list(
            cs.resolve_active_confirmatory_order(15, calibration_pass)
        )
