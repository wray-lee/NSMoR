"""Strict unit tests for the Stage-0 error-decomposition diagnostics.

These tests pin the *contract* of the optimization-decision instrumentation
(:mod:`nsmor.analysis.error_decomposition`) before any model/loss change is
licensed:

1. Magnitude-band decomposition of pooled squared error (artifact-share).
2. Lag-one persistence / zero-predictor skill per band.
3. Input-channel ``v_kine(t-1)`` identity check against ``y_true(t-1)``.
4. Missingness ledger over ``labeling_eligibility`` rows.

All fixtures are synthetic; no model or dataset artifact is required.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

import nsmor.analysis.error_decomposition as ed
from nsmor.analysis.error_decomposition import (
    decompose_squared_error,
    escape_sensitivity_sweep,
    lag_one_identity_check,
    missingness_ledger,
    padding_masking_audit,
    unavailable_trial_comparison,
)


# ═══════════════════════════════════════════════════════════════
# decompose_squared_error
# ═══════════════════════════════════════════════════════════════

class TestDecomposeSquaredError:
    """Band-share decomposition on the t>=1 (persistence-eligible) scope."""

    def _fixture(self):
        # 5 frames; t=0 is dropped (lag-one scope), leaving [5, 2000, -15000, 3].
        y_true = [np.array([0.0, 5.0, 2000.0, -15000.0, 3.0])]
        y_pred = [np.zeros(5)]
        return y_true, y_pred

    def test_frame_scope_is_t_ge_1(self):
        y_true, y_pred = self._fixture()
        out = decompose_squared_error(y_true, y_pred, bands_cm_s=(1e3, 1e4))
        assert out["n_frames"] == 4, out
        assert out["n_trials"] == 1

    def test_pooled_mse_matches_hand_computation(self):
        y_true, y_pred = self._fixture()
        out = decompose_squared_error(y_true, y_pred, bands_cm_s=(1e3, 1e4))
        # SSE = 5^2 + 2000^2 + 15000^2 + 3^2 = 229_000_034 over 4 frames.
        assert out["mse"] == pytest.approx(229_000_034.0 / 4.0)
        assert out["rmse"] == pytest.approx(math.sqrt(229_000_034.0 / 4.0))

    def test_band_sse_share_is_contribution_fraction(self):
        y_true, y_pred = self._fixture()
        out = decompose_squared_error(y_true, y_pred, bands_cm_s=(1e3, 1e4))
        bands = {b["band_cm_s"]: b for b in out["bands"]}

        b1k = bands[1e3]
        assert b1k["n_frames"] == 2
        assert b1k["frame_share"] == pytest.approx(0.5)
        assert b1k["sse"] == pytest.approx(229_000_000.0)
        assert b1k["sse_share"] == pytest.approx(229_000_000.0 / 229_000_034.0)

        b10k = bands[1e4]
        assert b10k["n_frames"] == 1
        assert b10k["sse"] == pytest.approx(225_000_000.0)
        assert b10k["sse_share"] == pytest.approx(225_000_000.0 / 229_000_034.0)

    def test_rest_is_below_lowest_band(self):
        y_true, y_pred = self._fixture()
        out = decompose_squared_error(y_true, y_pred, bands_cm_s=(1e3, 1e4))
        # rest = |y_true| < 1e3 -> frames 5 and 3.
        assert out["rest"]["n_frames"] == 2
        assert out["rest"]["sse"] == pytest.approx(34.0)

    def test_per_band_skill_columns(self):
        y_true, y_pred = self._fixture()
        out = decompose_squared_error(y_true, y_pred, bands_cm_s=(1e3, 1e4))
        b10k = {b["band_cm_s"]: b for b in out["bands"]}[1e4]
        # Single frame in the 1e4 band: true -15000, pred 0.
        assert b10k["baseline_mse"] == pytest.approx((-15000.0 - 2000.0) ** 2)
        assert b10k["zero_mse"] == pytest.approx(15000.0 ** 2)
        assert b10k["skill_vs_zero"] == pytest.approx(1.0 - 15000.0 ** 2 / 15000.0 ** 2)
        assert b10k["skill_vs_persistence"] == pytest.approx(
            1.0 - 15000.0 ** 2 / ((-15000.0 - 2000.0) ** 2)
        )

    def test_skill_vs_zero_and_persistence(self):
        y_true, y_pred = self._fixture()
        out = decompose_squared_error(y_true, y_pred, bands_cm_s=(1e3, 1e4))
        # Zero predictor MSE = mean(y_true^2) over eligible = same 4-frame mean.
        assert out["zero_mse"] == pytest.approx(229_000_034.0 / 4.0)
        # y_pred == 0 everywhere, so skill_vs_zero is exactly 0.
        assert out["skill_vs_zero"] == pytest.approx(0.0, abs=1e-12)
        # Persistence y_hat[t]=y_true[t-1] must be computed within the trial.
        base = np.array([0.0, 5.0, 2000.0, -15000.0])
        base_sse = float(np.sum((np.array([5.0, 2000.0, -15000.0, 3.0]) - base) ** 2))
        assert out["baseline_mse"] == pytest.approx(base_sse / 4.0)
        assert out["skill_vs_persistence"] == pytest.approx(
            1.0 - (229_000_034.0 / 4.0) / (base_sse / 4.0)
        )

    def test_lengths_crop_sequences(self):
        y_true = [np.array([0.0, 5.0, 2000.0, -15000.0, 3.0])]
        y_pred = [np.zeros(5)]
        # Single band (1e3) stays non-empty after the crop: eligible t in
        # {1,2} = [5, 2000] -> 2000 >= 1e3. (A second 1e4 band would be empty
        # post-crop and now fails closed by design.)
        out = decompose_squared_error(
            y_true, y_pred, lengths=[3], bands_cm_s=(1e3,)
        )
        assert out["n_frames"] == 2
        assert out["mse"] == pytest.approx((25.0 + 4_000_000.0) / 2.0)

    def test_rejects_nonfinite_true(self):
        y_true = [np.array([0.0, np.nan, 1.0])]
        y_pred = [np.zeros(3)]
        with pytest.raises(ValueError, match="non-finite"):
            decompose_squared_error(y_true, y_pred)

    def test_rejects_nonfinite_pred(self):
        y_true = [np.array([0.0, 1.0, 2.0])]
        y_pred = [np.array([0.0, np.inf, 0.0])]
        with pytest.raises(ValueError, match="non-finite"):
            decompose_squared_error(y_true, y_pred)

    def test_rejects_multidimensional_sequence(self):
        y_true = [np.zeros((3, 2))]
        y_pred = [np.zeros((3, 2))]
        with pytest.raises(ValueError, match="1-D"):
            decompose_squared_error(y_true, y_pred)

    def test_rejects_misaligned_sequences(self):
        y_true = [np.zeros(3), np.zeros(4)]
        y_pred = [np.zeros(3)]
        with pytest.raises(ValueError, match="aligned"):
            decompose_squared_error(y_true, y_pred)

    def test_rejects_bad_lengths(self):
        y_true = [np.zeros(3)]
        y_pred = [np.zeros(3)]
        with pytest.raises(ValueError):
            decompose_squared_error(y_true, y_pred, lengths=[5])

    # NOTE: the previous ``test_empty_scope_reports_none_with_reason`` pinned
    # a null-with-reason empty table. The Stage-0 rejection review requires an
    # empty band scope to FAIL CLOSED instead; see
    # ``TestFailClosedValidation.test_decompose_empty_scope_fails_closed``.


# ═══════════════════════════════════════════════════════════════
# lag_one_identity_check
# ═══════════════════════════════════════════════════════════════

class TestLagOneIdentityCheck:
    """Channel [2] (v_kine(t-1)) must equal the target's own lag-one frame."""

    def test_exact_identity(self):
        Y = [np.array([0.0, 5.0, 7.0, -3.0])]
        X = [np.zeros((4, 8))]
        # channel 2 holds v_kine(t-1): 0 at t=0, then velocity shifted by one.
        X[0][1:, 2] = Y[0][:-1]
        out = lag_one_identity_check(X, Y, lengths=[4])
        assert out["n_frames_compared"] == 3
        assert out["max_abs_residual"] == pytest.approx(0.0)
        assert out["exact_fraction"] == pytest.approx(1.0)
        assert out["correlation"] == pytest.approx(1.0)

    def test_detects_residual(self):
        Y = [np.array([0.0, 5.0, 7.0])]
        X = [np.zeros((3, 8))]
        X[0][1:, 2] = Y[0][:-1]
        X[0][2, 2] = 100.0  # corrupt one comparison frame
        out = lag_one_identity_check(X, Y, lengths=[3])
        assert out["max_abs_residual"] == pytest.approx(abs(100.0 - 5.0))
        assert out["n_exact"] == 1

    def test_rejects_nonfinite_channel(self):
        Y = [np.array([0.0, 5.0, 7.0])]
        X = [np.zeros((3, 8))]
        X[0][1, 2] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            lag_one_identity_check(X, Y, lengths=[3])

    def test_rejects_bad_channel_dim(self):
        Y = [np.array([0.0, 5.0])]
        X = [np.zeros((2, 2))]  # only 2 channels; channel 2 is out of range
        with pytest.raises(ValueError, match="channel"):
            lag_one_identity_check(X, Y, lengths=[2])


# ═══════════════════════════════════════════════════════════════
# missingness_ledger
# ═══════════════════════════════════════════════════════════════

class TestMissingnessLedger:
    """Ledger over ``labeling_eligibility`` rows."""

    def _rows(self):
        return [
            {"status": "labeled", "label": "ESCAPE",
             "source_channel_condition": "multisensory"},
            {"status": "labeled", "label": "NO_RESPONSE",
             "source_channel_condition": "wind_only"},
            {"status": "labeled", "label": "PREWALK",
             "source_channel_condition": "visual_only"},
            {"status": "unavailable_no_stimulus_anchor", "label": None,
             "source_channel_condition": "no_stimulus"},
            {"status": "unavailable_no_stimulus_anchor", "label": None,
             "source_channel_condition": "no_stimulus"},
        ]

    def test_counts(self):
        out = missingness_ledger(self._rows())
        assert out["n_eligibility_rows"] == 5
        assert out["n_labeled"] == 3
        assert out["n_unavailable"] == 2
        assert out["by_status"]["unavailable_no_stimulus_anchor"] == 2
        assert out["unavailable_by_condition"] == {"no_stimulus": 2}
        assert out["unavailable_by_label"] == {"None": 2}
        assert out["mechanism"] == "unavailable_no_stimulus_anchor"
        # The dropped condition (no_stimulus) never appears among labeled rows.
        # That is an OBSERVATION only: without an independent drop rule it does
        # NOT license a structural/MCAR mechanism or a biological-completeness
        # claim, so the assumption stays unresolved.
        assert out["labeled_conditions"] == ["multisensory", "visual_only", "wind_only"]
        assert out["observed_condition_separation"] is True
        assert out["missingness_assumption"] == "unresolved_requires_mechanism_analysis"
        assert out["biological_completeness"] == "not_established"

    def test_mixed_statuses_flagged(self):
        rows = self._rows() + [
            {"status": "other_drop", "label": None,
             "source_channel_condition": "wind_only"},
        ]
        out = missingness_ledger(rows)
        assert out["n_unavailable"] == 3
        assert out["mechanism"] == "mixed_statuses"
        # A dropped condition that also appears among labeled rows is observed
        # overlap; either way the mechanism is unresolved.
        assert out["observed_condition_separation"] is False
        assert out["missingness_assumption"] == "unresolved_requires_mechanism_analysis"

    def test_no_missingness(self):
        rows = [{"status": "labeled", "label": "ESCAPE",
                 "source_channel_condition": "wind_only"}]
        out = missingness_ledger(rows)
        assert out["n_unavailable"] == 0
        assert out["mechanism"] == "none"
        assert out["observed_condition_separation"] is False
        assert out["missingness_assumption"] == "no_missingness"
        # Completeness is never asserted from the ledger alone.
        assert out["biological_completeness"] == "not_established"

    def test_rejects_malformed_row(self):
        with pytest.raises(ValueError, match="status"):
            missingness_ledger([{"label": "ESCAPE"}])

    def test_rejects_non_mapping_row(self):
        with pytest.raises(ValueError):
            missingness_ledger(["not-a-dict"])


# ═══════════════════════════════════════════════════════════════
# Fail-closed validation (rejected-stage-0 repair)
# ═══════════════════════════════════════════════════════════════

class TestFailClosedValidation:
    """Empty bands and non-integral lengths must raise, never fabricate."""

    def test_decompose_empty_band_fails_closed(self):
        # 1e3 band admits no frame in this fixture: the rejection review
        # requires an explicit ValueError rather than a silent empty row.
        y_true = [np.array([0.0, 5.0, 7.0])]
        y_pred = [np.zeros(3)]
        with pytest.raises(ValueError, match="empty band"):
            decompose_squared_error(y_true, y_pred, bands_cm_s=(1e3,))

    def test_decompose_non_integral_length_rejected(self):
        y_true = [np.array([0.0, 5.0, 7.0])]
        y_pred = [np.zeros(3)]
        # 2.5 must be rejected, not silently truncated to 2.
        with pytest.raises(ValueError, match="non-integral"):
            decompose_squared_error(y_true, y_pred, lengths=[2.5])
        with pytest.raises(ValueError, match="non-integral"):
            decompose_squared_error(y_true, y_pred, lengths=[True])

    def test_decompose_empty_scope_fails_closed(self):
        # Every trial has a single frame -> no eligible frame anywhere. This
        # is an empty band scope and must raise, not emit a null table.
        y_true = [np.array([1.0])]
        y_pred = [np.array([1.0])]
        with pytest.raises(ValueError, match="empty band"):
            decompose_squared_error(y_true, y_pred, bands_cm_s=(1e3,))


class TestUnavailableArtifactBands:
    """A *configured* band that admits no frame is reported, not fabricated.

    The default CLI evaluates a scope whose ``max |y_true|`` (~139.78 cm/s) is
    far below the artifact bands 1000/10000 cm/s. A valid configured threshold
    with zero eligible frames must produce an explicit unavailable row
    (``status``/``reason`` + null statistics), NEVER a fabricated ``mse=0`` and
    NEVER a silent replacement of 1000/10000 with 10/100. An empty threshold
    *specification* (``bands_cm_s=[]`` / ``--bands ""``) still fails closed.
    """

    def _low_magnitude(self):
        # max |y_true| = 139.78 cm/s: both artifact bands are empty.
        y_true = [np.array([0.0, 100.0, 139.78])]
        y_pred = [np.zeros(3)]
        return y_true, y_pred

    def test_configured_empty_band_reports_unavailable_row(self):
        y_true, y_pred = self._low_magnitude()
        out = decompose_squared_error(
            y_true, y_pred, bands_cm_s=(1000.0, 10000.0),
            allow_unavailable=True,
        )
        # Thresholds are preserved verbatim, never substituted.
        bands = {b["band_cm_s"]: b for b in out["bands"]}
        assert set(bands) == {1000.0, 10000.0}
        for row in bands.values():
            assert row["status"] == "unavailable"
            assert isinstance(row["reason"], str) and row["reason"]
            assert row["n_frames"] == 0
            # Null statistics -- never a fabricated zero.
            assert row["mse"] is None
            assert row["rmse"] is None
            assert row["baseline_mse"] is None
            assert row["skill_vs_persistence"] is None
            assert row["zero_mse"] is None
            assert row["skill_vs_zero"] is None
        # The rest partition is still observed on the same scope.
        assert out["rest"]["status"] == "observed"
        assert out["rest"]["n_frames"] == 2
        # The pooled scope is still finite and explicit.
        assert out["n_frames"] == 2
        assert math.isfinite(out["mse"])

    def test_unavailable_band_rows_are_json_serializable(self):
        y_true, y_pred = self._low_magnitude()
        out = decompose_squared_error(
            y_true, y_pred, bands_cm_s=(1000.0, 10000.0),
            allow_unavailable=True,
        )
        text = json.dumps(out, allow_nan=False)
        assert "NaN" not in text and "Infinity" not in text

    def test_mixed_observed_and_unavailable_bands(self):
        y_true = [np.array([0.0, 2000.0, 3.0])]
        y_pred = [np.zeros(3)]
        out = decompose_squared_error(
            y_true, y_pred, bands_cm_s=(1000.0, 10000.0),
            allow_unavailable=True,
        )
        bands = {b["band_cm_s"]: b for b in out["bands"]}
        assert bands[1000.0]["status"] == "observed"
        assert bands[1000.0]["n_frames"] == 1
        assert bands[10000.0]["status"] == "unavailable"
        assert bands[10000.0]["mse"] is None

    def test_default_still_fails_closed_without_opt_in(self):
        y_true, y_pred = self._low_magnitude()
        with pytest.raises(ValueError, match="empty band"):
            decompose_squared_error(y_true, y_pred, bands_cm_s=(1000.0, 10000.0))

    def test_empty_band_spec_fails_closed_even_with_opt_in(self):
        # An empty *specification* is a scope error regardless of opt-in.
        y_true, y_pred = self._low_magnitude()
        with pytest.raises(ValueError, match="band"):
            decompose_squared_error(
                y_true, y_pred, bands_cm_s=(), allow_unavailable=True,
            )
        with pytest.raises(ValueError, match="band"):
            decompose_squared_error(
                y_true, y_pred, bands_cm_s=[], allow_unavailable=True,
            )


class TestEmptyBandPaths:
    """Every public empty-band path raises ValueError (never IndexError)."""

    def _fixture(self):
        y_true = [np.array([0.0, 5.0, 2000.0, 3.0])]
        y_pred = [np.zeros(4)]
        return y_true, y_pred

    def test_decompose_empty_tuple_raises_value_error(self):
        y_true, y_pred = self._fixture()
        with pytest.raises(ValueError, match="band"):
            decompose_squared_error(y_true, y_pred, bands_cm_s=())
        with pytest.raises(ValueError, match="band"):
            decompose_squared_error(y_true, y_pred, bands_cm_s=[])

    def test_escape_empty_band_list_raises_value_error(self):
        y_true, y_pred = self._fixture()
        with pytest.raises(ValueError, match="band"):
            escape_sensitivity_sweep(y_true, y_pred, bands_cm_s=[])
        with pytest.raises(ValueError, match="band"):
            escape_sensitivity_sweep(y_true, y_pred, bands_cm_s=[], min_runs=(1,))

    def test_parse_float_list_rejects_empty_cli_value(self):
        # ``--bands ""`` must fail closed with ValueError, not a later
        # IndexError from bands[0] or a silent zero-band run.
        with pytest.raises(ValueError, match="band"):
            ed._parse_float_list("", "bands")
        with pytest.raises(ValueError, match="band"):
            ed._parse_float_list("   ,  , ", "bands")

    def test_parse_float_list_parses_valid(self):
        assert ed._parse_float_list("1000,10000", "bands") == [1000.0, 10000.0]

    def test_cli_default_bands_are_artifact_bands(self):
        args = ed._build_parser().parse_args(["--output_dir", "unused"])
        assert ed._parse_float_list(args.bands, "bands") == [1000.0, 10000.0]
        assert ed._parse_float_list(args.sweep_bands, "sweep_bands") == [
            5.0, 10.0, 20.0, 50.0,
        ]
        assert ed._parse_float_list(args.min_runs, "min_runs") == [1.0, 2.0, 3.0]


# ═══════════════════════════════════════════════════════════════
# Cumulative (nested) share labelling
# ═══════════════════════════════════════════════════════════════

class TestBandShareScope:
    """Band shares are cumulative/nested, never summed as disjoint."""

    def _fixture(self):
        # |y_true| t>=1: [5, 2000, -15000, 3]; bands 1e3 and 1e4 are nested.
        y_true = [np.array([0.0, 5.0, 2000.0, -15000.0, 3.0])]
        y_pred = [np.zeros(5)]
        return y_true, y_pred

    def test_band_shares_are_cumulative_not_summable(self):
        y_true, y_pred = self._fixture()
        out = decompose_squared_error(y_true, y_pred, bands_cm_s=(1e3, 1e4))
        bands = {b["band_cm_s"]: b for b in out["bands"]}
        # The 1e4 band is a strict subset of the 1e3 band, so the two shares
        # overlap; summing them would double-count and can exceed 1.
        assert bands[1e4]["n_frames"] < bands[1e3]["n_frames"]
        assert bands[1e3]["sse_share"] + bands[1e4]["sse_share"] > 1.0
        for row in bands.values():
            assert row["share_scope"] == ed._CUMULATIVE_SHARE_SCOPE
        assert out["band_share_scope"] == ed._CUMULATIVE_SHARE_SCOPE
        # The scope label explicitly says "not summable".
        assert "not_summable" in ed._CUMULATIVE_SHARE_SCOPE

    def test_rest_is_the_only_disjoint_partition(self):
        y_true, y_pred = self._fixture()
        out = decompose_squared_error(y_true, y_pred, bands_cm_s=(1e3, 1e4))
        # rest (|y| < 1e3) and the 1e3 band partition the pooled scope exactly.
        bands = {b["band_cm_s"]: b for b in out["bands"]}
        assert bands[1e3]["n_frames"] + out["rest"]["n_frames"] == out["n_frames"]
        assert bands[1e3]["sse"] + out["rest"]["sse"] == pytest.approx(
            out["mse"] * out["n_frames"]
        )


# ═══════════════════════════════════════════════════════════════
# Numerical overflow fails closed (finite-but-extreme inputs)
# ═══════════════════════════════════════════════════════════════

class TestNumericalOverflowFailsClosed:
    """Finite inputs whose difference/square/sum overflows must reject."""

    def test_decompose_opposite_extreme_overflow_rejected(self):
        # [0, 1e308] vs [0, -1e308]: the residual 2e308 overflows float64.
        y_true = [np.array([0.0, 1e308])]
        y_pred = [np.array([0.0, -1e308])]
        with pytest.raises(ValueError, match="not representable|overflow"):
            decompose_squared_error(y_true, y_pred, bands_cm_s=(1.0,))

    def test_decompose_cumulative_sum_overflow_rejected(self):
        # GENUINE cumulative-only overflow: each squared error is finite
        # ((1e154)**2 = 1e308 < float64 max), but their SUM (2e308) overflows.
        # This is NOT the square-level overflow case (1e308 would already
        # overflow on squaring), which is covered separately above.
        y_true = [np.array([0.0, 1e154, 1e154])]
        y_pred = [np.zeros(3)]
        with pytest.raises(ValueError, match="overflow|not representable"):
            decompose_squared_error(y_true, y_pred, bands_cm_s=(1.0,))

    def test_skill_tiny_comparator_is_unavailable_not_negative_inf(self):
        # B's repro: target [0, 1e-160], pred [0, 1]. The lag-one comparator
        # MSE is ~1e-320 (subnormal) while the model MSE is ~1; the ratio
        # overflows to inf, so skill_vs_persistence must be None (unavailable),
        # never -inf.
        y_true = [np.array([0.0, 1e-160])]
        y_pred = [np.array([0.0, 1.0])]
        out = decompose_squared_error(
            y_true, y_pred, bands_cm_s=(1e-300,), allow_unavailable=True,
        )
        assert out["skill_vs_persistence"] is None
        assert out["skill_vs_persistence"] != float("-inf")
        # JSON-safe: no inf leaks from the undefined skill.
        text = json.dumps(out, allow_nan=False)
        assert "Infinity" not in text and "NaN" not in text

    def test_skill_tiny_comparator_direct(self):
        # Direct helper contract for the same overflow path.
        assert ed._skill(1.0, 1e-320) is None
        assert ed._skill(1.0, 1e-310) is None
        # A representable ratio still yields a finite skill.
        assert ed._skill(2.0, 4.0) == pytest.approx(0.5)
        assert ed._skill(1.0, 0.0) is None

    def test_lag_one_sign_flip_overflow_rejected(self):
        # channel == +1e308, y_prev == -1e308 -> residual 2e308 overflows.
        Y = [np.array([-1e308, 5.0, 7.0])]
        X = [np.zeros((3, 8))]
        X[0][1, 2] = 1e308
        with pytest.raises(ValueError, match="overflow|not representable"):
            lag_one_identity_check(X, Y, lengths=[3])

    def test_escape_sweep_overflow_rejected(self):
        y_true = [np.array([0.0, 1e308])]
        y_pred = [np.array([0.0, -1e308])]
        with pytest.raises(ValueError, match="overflow|not representable"):
            escape_sensitivity_sweep(
                y_true, y_pred, bands_cm_s=[1.0], min_runs=(1,),
            )



# ═══════════════════════════════════════════════════════════════
# Sustained escape-band sensitivity sweep
# ═══════════════════════════════════════════════════════════════

class TestEscapeSensitivitySweep:
    """Formal sweep with train.py ``_sustained_run`` semantics."""

    def _fixture(self):
        # t1: sustained 2-frame run at 30-40; t2: lone 150 + isolated 1e7.
        t1 = np.array([0.0, 30.0, 40.0, 0.0])
        t2 = np.array([150.0, 0.0, 0.0, 1e7])
        return [t1, t2], [t1.copy(), t2.copy()]

    def test_cartesian_and_per_sequence_runs(self):
        y_true, y_pred = self._fixture()
        out = escape_sensitivity_sweep(
            y_true, y_pred, bands_cm_s=[10.0], min_runs=(1, 2),
        )
        rows = {(r["band_cm_s"], r["min_run"]): r for r in out["rows"]}
        assert set(rows) == {(10.0, 1), (10.0, 2)}
        # min_run=1: {30,40} + {150} + 1e7 = 4 frames, 3 events (full scope).
        assert rows[(10.0, 1)]["n_escape_frames_full"] == 4
        assert rows[(10.0, 1)]["n_escape_events_full"] == 3
        # min_run=2: only the sustained {30,40} run survives.
        assert rows[(10.0, 2)]["n_escape_frames_full"] == 2
        assert rows[(10.0, 2)]["n_escape_events_full"] == 1

    def test_full_vs_aligned_are_separate_scopes(self):
        # The mask is computed on the FULL valid trial, then sliced to t>=1 for
        # the aligned scope; a run that began at t=0 is NOT recomputed/split on
        # the slice. y=[20,20,0], band=10, min_run=2 -> full keeps {20,20}
        # (2 escape frames); aligned (t>=1) keeps only the second 20 (1 frame).
        y_true = [np.array([20.0, 20.0, 0.0])]
        y_pred = [np.zeros(3)]
        out = escape_sensitivity_sweep(
            y_true, y_pred, bands_cm_s=[10.0], min_runs=(2,),
        )
        row = out["rows"][0]
        assert row["status"] == "observed"
        assert row["n_escape_frames_full"] == 2
        assert row["n_escape_frames_aligned"] == 1
        assert row["n_frames_full"] == 3
        assert row["n_frames_aligned"] == 2
        assert out["denominators"]["full"]["n_frames"] == 3
        assert out["denominators"]["aligned"]["n_frames"] == 2

    def test_matches_train_sweep_semantics(self):
        # Parity: the analysis sweep must reproduce train.py's own numbers
        # (same _sustained_run, same per-sequence run counting) rather than
        # an incompatible escape definition. The ``full`` scope is compared
        # because it is exactly train.py's scope.
        from scripts.train import sweep_escape_sensitivity as train_sweep

        y_true, y_pred = self._fixture()
        for bands, runs in (([10.0], [1, 2]), ([100.0], [1])):
            out = escape_sensitivity_sweep(
                y_true, y_pred, bands_cm_s=bands, min_runs=tuple(runs),
            )
            ref = train_sweep(y_true, y_pred, bands_cm_s=bands, min_runs=runs)
            ref_by = {(r["band_cm_s"], r["min_run"]): r for r in ref}
            for row in out["rows"]:
                expected = ref_by[(row["band_cm_s"], row["min_run"])]
                assert row["n_escape_frames_full"] == expected["n_escape_frames"]
                assert row["n_escape_events_full"] == expected["n_escape_events"]
                assert row["escape_rmse_full"] == pytest.approx(
                    expected["escape_rmse"]
                )
                assert row["resting_rmse_full"] == pytest.approx(
                    expected["resting_rmse"]
                )

    def test_singleton_high_trial_aligned_escape_unavailable(self):
        # B's repro: a singleton over-band trial is an escape on the FULL scope
        # but its only t>=1 frame is not, so the aligned escape partition has
        # zero frames on a non-zero denominator and must be explicitly
        # unavailable with a reason (never a fabricated zero).
        y_true = [np.array([20.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0])]
        y_pred = [np.zeros(2), np.zeros(4)]
        out = escape_sensitivity_sweep(
            y_true, y_pred, bands_cm_s=[10.0], min_runs=(1,),
        )
        row = out["rows"][0]
        assert row["full"]["escape_status"] == "observed"
        assert row["full"]["n_frames"] == 6
        assert row["n_escape_frames_full"] == 1
        # Aligned denominator is non-zero, but the aligned band admits no frame.
        assert row["aligned"]["n_frames"] == 4
        assert row["aligned"]["escape_status"] == "unavailable"
        assert isinstance(row["aligned"]["escape_reason"], str)
        assert row["aligned"]["escape_reason"]
        assert row["n_escape_frames_aligned"] == 0
        assert row["escape_rmse_aligned"] is None
        assert row["escape_rmse_full"] is not None
        # Cell is observed because the full scope has an escape.
        assert row["status"] == "observed"

    def test_singleton_only_trial_full_escape_aligned_empty(self):
        # y=[20], band=10, run=1: full scope has one escape frame; aligned
        # (t>=1) has no frames at all -> aligned escape unavailable, no -inf.
        y_true = [np.array([20.0])]
        y_pred = [np.zeros(1)]
        out = escape_sensitivity_sweep(
            y_true, y_pred, bands_cm_s=[10.0], min_runs=(1,),
        )
        row = out["rows"][0]
        assert row["full"]["escape_status"] == "observed"
        assert row["n_escape_frames_full"] == 1
        assert row["aligned"]["escape_status"] == "unavailable"
        assert row["aligned"]["escape_reason"]
        assert row["aligned"]["n_frames"] == 0
        assert row["escape_rmse_aligned"] is None
        assert row["status"] == "observed"

    def test_empty_cell_is_null_row_not_abort(self):
        # An empty (band, min_run) cell must NOT throw and abort the whole grid;
        # it is emitted with null statistics and valid zero counts, so every
        # requested cell is retained.
        y_true, y_pred = self._fixture()
        out = escape_sensitivity_sweep(
            y_true, y_pred, bands_cm_s=[10.0, 1e9], min_runs=(1, 2),
        )
        assert len(out["rows"]) == 4  # full Cartesian grid retained
        rows = {(r["band_cm_s"], r["min_run"]): r for r in out["rows"]}
        empty = rows[(1e9, 2)]
        assert empty["status"] == "unavailable"
        assert isinstance(empty["reason"], str) and empty["reason"]
        assert empty["n_escape_frames_full"] == 0
        assert empty["n_escape_frames_aligned"] == 0
        assert empty["escape_rmse_full"] is None
        assert empty["escape_rmse_aligned"] is None
        # JSON-safe: no NaN/Inf leaks from an empty cell.
        text = json.dumps(out, allow_nan=False)
        assert "NaN" not in text and "Infinity" not in text

    def test_non_integral_min_run_rejected(self):
        y_true, y_pred = self._fixture()
        with pytest.raises(ValueError, match="min_run"):
            escape_sensitivity_sweep(
                y_true, y_pred, bands_cm_s=[10.0], min_runs=(2.5,),
            )
        with pytest.raises(ValueError, match="min_run"):
            escape_sensitivity_sweep(
                y_true, y_pred, bands_cm_s=[10.0], min_runs=(True,),
            )


# ═══════════════════════════════════════════════════════════════
# Padding/masking vs removal audit
# ═══════════════════════════════════════════════════════════════

class TestPaddingMaskingAudit:
    """Padding is masked (retained, unscored); unavailable trials are removed.

    A frame count is only truthful when it is derived from *observed* batch
    tensors. The default call emits ``status="unavailable"`` rather than
    inventing a corpus-wide padding count from dataset lengths.
    """

    def test_default_scope_reports_unavailable(self):
        out = padding_masking_audit(
            batch_true_lengths=[[5, 3]], batch_padded_lengths=[5],
        )
        assert out["status"] == "unavailable"
        assert out["reason"]
        assert out["scored_frames"] is None
        assert out["padding_frames_masked"] is None
        # The removal count is scoped to the full-corpus labeling_eligibility
        # ledger, NOT the observed validation batches.
        assert out["unavailable_trials_removed_scope"] == (
            "full_corpus_labeling_eligibility"
        )

    def test_observed_scope_carries_full_corpus_removal_scope(self):
        out = padding_masking_audit(
            batch_true_lengths=[[5, 3]], batch_padded_lengths=[5],
            unavailable_trials=128, scope="observed_batches",
        )
        assert out["status"] == "observed"
        assert out["unavailable_trials_removed"] == 128
        assert out["unavailable_trials_removed_scope"] == (
            "full_corpus_labeling_eligibility"
        )

    def test_padding_is_masked_not_removed(self):
        # One observed batch: padded tensor length 5, true lengths [5, 3] ->
        # 2 padded frames that are masked, NOT removed.
        out = padding_masking_audit(
            batch_true_lengths=[[5, 3]], batch_padded_lengths=[5],
            scope="observed_batches",
        )
        assert out["status"] == "observed"
        assert out["scored_frames"] == 8
        assert out["padding_frames_masked"] == 2
        assert out["unavailable_trials_removed"] == 0

    def test_unavailable_trials_reported_separately(self):
        out = padding_masking_audit(
            batch_true_lengths=[[5, 3]], batch_padded_lengths=[5],
            unavailable_trials=128, scope="observed_batches",
        )
        # The 128 dropped rows are removed upstream and must never be folded
        # into the padding count (which is masked, not removed).
        assert out["unavailable_trials_removed"] == 128
        assert out["padding_frames_masked"] == 2

    def test_true_length_exceeding_padding_fails_closed(self):
        with pytest.raises(ValueError, match="padding"):
            padding_masking_audit(
                batch_true_lengths=[[6]], batch_padded_lengths=[5],
                scope="observed_batches",
            )

    def test_non_integral_length_rejected(self):
        with pytest.raises(ValueError, match="non-integral"):
            padding_masking_audit(
                batch_true_lengths=[[3.5]], batch_padded_lengths=[5],
                scope="observed_batches",
            )


# ═══════════════════════════════════════════════════════════════
# With-vs-without unavailable-trial comparison
# ═══════════════════════════════════════════════════════════════

class TestUnavailableTrialComparison:
    """Fail closed when the 128 unavailable rows cannot be replayed."""

    def test_fails_closed_when_not_replayable(self):
        out = unavailable_trial_comparison(
            n_unavailable_trials=128, unavailable_trials_replayable=False,
        )
        assert out["status"] == "unavailable"
        assert out["with_unavailable"] is None
        assert out["without_unavailable"] is None
        assert "reason" in out

    def test_computes_delta_when_replayable(self):
        out = unavailable_trial_comparison(
            with_unavailable={"mse": 11.0, "n_frames": 100},
            without_unavailable={"mse": 10.0, "n_frames": 90},
            n_unavailable_trials=2, unavailable_trials_replayable=True,
        )
        assert out["status"] == "computed"
        assert out["delta_mse"] == pytest.approx(1.0)

    def test_replayable_but_missing_block_fails_closed(self):
        with pytest.raises(ValueError, match="both"):
            unavailable_trial_comparison(
                with_unavailable={"mse": 11.0},
                n_unavailable_trials=2, unavailable_trials_replayable=True,
            )

    def test_no_unavailable_is_not_applicable(self):
        out = unavailable_trial_comparison(
            n_unavailable_trials=0, unavailable_trials_replayable=False,
        )
        assert out["status"] == "not_applicable"

    def test_delta_overflow_is_null_with_reason(self):
        # Two finite MSEs of opposite extreme sign: the difference overflows
        # float64, so delta_mse must be None with a reason (never inf), and the
        # blocks are still reported.
        out = unavailable_trial_comparison(
            with_unavailable={"mse": 1e308, "n_frames": 10},
            without_unavailable={"mse": -1e308, "n_frames": 10},
            n_unavailable_trials=2, unavailable_trials_replayable=True,
        )
        assert out["status"] == "computed"
        assert out["delta_mse"] is None
        assert out["reason"]
        assert out["with_unavailable"]["mse"] == 1e308
        # JSON-safe: no inf leaks from the undefined delta.
        text = json.dumps(out, allow_nan=False)
        assert "Infinity" not in text


# ═══════════════════════════════════════════════════════════════
# main() CLI wiring (on-disk JSON), with stubbed model/artifact I/O
# ═══════════════════════════════════════════════════════════════

class _StubModel(torch.nn.Module):
    """Minimal NSMoRCore-shaped module: ``(x, lengths, return_internals)``."""

    def __init__(self, dt_ms: float = 4.0) -> None:
        super().__init__()
        self.dt_ms = dt_ms

    def forward(self, x, lengths, return_internals=True):  # noqa: D102
        return x[..., 1] * 0.0, {}


def _stub_dataset(y_high: float = 20000.0):
    """Two trials (T=5 and T=4); ``y_high`` tunes the max |y_true|.

    The default (20000) admits both default artifact bands; a low value (e.g.
    139.78) makes 1000/10000 admit no frame so the explicit unavailable path is
    exercised end-to-end.
    """
    y0 = np.array([0.0, 60.0, 70.0, 80.0, y_high])
    y1 = np.array([0.0, 5.0, 6.0, 7.0])
    x0 = np.zeros((5, 8))
    x1 = np.zeros((4, 8))
    x0[1:, 2] = y0[:-1]  # channel 2 == v_kine(t-1)
    x1[1:, 2] = y1[:-1]
    priors = np.full((2, 4), 0.25)
    return {
        "X_seqs": [x0, x1],
        "Y_seqs": [y0, y1],
        "lengths": [5, 4],
        "anchor_frames": [2, 2],
        "mcmc_priors": priors,
        "labeling_eligibility": [
            {"status": "labeled", "label": "ESCAPE",
             "source_channel_condition": "multisensory"},
            {"status": "labeled", "label": "NO_RESPONSE",
             "source_channel_condition": "wind_only"},
            {"status": "unavailable_no_stimulus_anchor", "label": None,
             "source_channel_condition": "no_stimulus"},
        ],
    }


def _patch_io(monkeypatch, dataset):
    """Stub the model/dataset/prior I/O used by ``error_decomposition.main``."""
    import nsmor.analysis.analysis_priors as priors_mod
    import nsmor.analysis.prediction_units as pu
    import nsmor.model_utils as model_utils
    import nsmor.pipeline.nested_prior as nested

    monkeypatch.setattr(
        pu, "load_model_from_checkpoint", lambda path, device: _StubModel(),
    )
    monkeypatch.setattr(
        nested, "load_dataset_with_fingerprint",
        lambda *a, **k: (dataset, "stub-fingerprint"),
    )
    monkeypatch.setattr(
        model_utils, "validate_dataset_provenance", lambda *a, **k: None,
    )
    monkeypatch.setattr(
        priors_mod, "load_analysis_priors",
        lambda *a, **k: (dataset["mcmc_priors"], [0, 1]),
    )


class TestMainCliWiring:
    """Exercise the on-disk JSON path, not only the pure helpers."""

    def test_main_writes_truthful_json(self, tmp_path, monkeypatch):
        dataset = _stub_dataset()
        _patch_io(monkeypatch, dataset)

        rc = ed.main(["--output_dir", str(tmp_path)])
        assert rc == 0

        out = json.loads((tmp_path / "error_decomposition.json").read_text())
        # Artifact bands restored (not the rejected 10,100 default).
        assert out["scope"]["bands_cm_s"] == [1000.0, 10000.0]
        assert out["scope"]["escape_sweep_bands_cm_s"] == [5.0, 10.0, 20.0, 50.0]
        assert out["scope"]["escape_sweep_min_runs"] == [1, 2, 3]
        assert out["n_val_trials"] == 2
        assert out["n_frames_scored"] == 9

        # Blocker 1: the configured artifact bands are PRESERVED verbatim and
        # reported as explicitly unavailable on this scope (the stub's max
        # |y_true| = 20000 -> 1e3 observed, 1e4 observed here; the CLI scope is
        # still truthful and never silently substitutes narrower bands).
        bands = {b["band_cm_s"]: b for b in out["artifact_decomposition"]["bands"]}
        assert set(bands) == {1000.0, 10000.0}
        assert all("status" in b for b in bands.values())
        assert all(b["mse"] is not None for b in bands.values())
        # A separate run pins the unavailable path on the real 139.78 cm/s
        # scope; see TestUnavailableArtifactBands.

        # Blocker 2: denominator scopes are explicit and distinct.
        assert out["scope"]["padding_masking_audit_scope"] == (
            "validation_observed_batches"
        )
        assert out["scope"]["missingness_ledger_scope"] == (
            "full_corpus_labeling_eligibility"
        )
        assert out["scope"]["unavailable_trials_removed_scope"] == (
            "full_corpus_labeling_eligibility"
        )

        # Padding audit is bound to OBSERVED validation batches (truthful), and
        # the unavailable rows are reported as removed, never as padding.
        pa = out["padding_masking_audit"]
        assert pa["status"] == "observed"
        assert pa["scored_frames"] == 9
        assert pa["padding_frames_masked"] == 1  # (5-5) + (5-4)
        assert pa["unavailable_trials_removed"] == 1
        assert pa["unavailable_trials_removed_scope"] == (
            "full_corpus_labeling_eligibility"
        )

        # Unavailable comparison stays unavailable (raw trials absent).
        assert out["unavailable_trial_comparison"]["status"] == "unavailable"
        assert out["missingness_ledger"]["n_unavailable"] == 1
        assert out["missingness_ledger"]["n_eligibility_rows"] == 3

        # Full default sweep grid present (12 rows) and unchanged semantics.
        assert len(out["escape_sensitivity"]["rows"]) == 12
        # lag-one identity on channel 2 is exact by construction.
        assert out["lag_one_identity"]["exact_fraction"] == pytest.approx(1.0)

    def test_main_reports_artifact_bands_unavailable_on_low_scope(
        self, tmp_path, monkeypatch,
    ):
        # Reproduces the real default-CLI scope: max |y_true| = 139.78 cm/s, far
        # below the artifact bands 1000/10000 cm/s. The CLI must SUCCEED with an
        # explicit unavailable row for each band (null stats, threshold
        # preserved), never abort, never fabricate a zero, never substitute.
        dataset = _stub_dataset(y_high=139.78)
        _patch_io(monkeypatch, dataset)

        rc = ed.main(["--output_dir", str(tmp_path)])
        assert rc == 0

        out = json.loads((tmp_path / "error_decomposition.json").read_text())
        # Thresholds preserved verbatim.
        assert out["scope"]["bands_cm_s"] == [1000.0, 10000.0]
        bands = {b["band_cm_s"]: b for b in out["artifact_decomposition"]["bands"]}
        assert set(bands) == {1000.0, 10000.0}
        for band, row in bands.items():
            assert row["status"] == "unavailable", band
            assert row["reason"]
            assert row["n_frames"] == 0
            assert row["mse"] is None
            assert row["rmse"] is None
            assert row["baseline_mse"] is None
            assert row["skill_vs_persistence"] is None
        # The low-magnitude bulk is fully accounted for by the rest partition.
        assert out["artifact_decomposition"]["rest"]["n_frames"] == 7
        # Unavailable comparison and full-corpus missingness scope unchanged.
        assert out["unavailable_trial_comparison"]["status"] == "unavailable"
        assert out["scope"]["missingness_ledger_scope"] == (
            "full_corpus_labeling_eligibility"
        )
        assert len(out["escape_sensitivity"]["rows"]) == 12

    def test_main_rejects_non_fresh_output_dir(self, tmp_path, monkeypatch):
        # A populated output_dir must be refused BEFORE the model/dataset I/O,
        # and the prior artifact must be left byte-identical.
        dataset = _stub_dataset()
        _patch_io(monkeypatch, dataset)
        (tmp_path / "error_decomposition.json").write_text("PRIOR_ARTIFACT")
        with pytest.raises(FileExistsError, match="fresh"):
            ed.main(["--output_dir", str(tmp_path)])
        assert (tmp_path / "error_decomposition.json").read_text() == "PRIOR_ARTIFACT"

    def test_main_records_input_sha256_content_binding(self, tmp_path, monkeypatch):
        # The receipt must bind the exact input bytes, not just the paths.
        import hashlib

        dataset = _stub_dataset()
        _patch_io(monkeypatch, dataset)
        ckpt = tmp_path / "ckpt.bin"
        data = tmp_path / "data.bin"
        nest = tmp_path / "nest.bin"
        for p, blob in ((ckpt, b"CKPT"), (data, b"DATA"), (nest, b"NEST")):
            p.write_bytes(blob)
        out_dir = tmp_path / "out"

        rc = ed.main([
            "--output_dir", str(out_dir),
            "--checkpoint", str(ckpt),
            "--dataset", str(data),
            "--nested_prior_artifact", str(nest),
        ])
        assert rc == 0
        out = json.loads((out_dir / "error_decomposition.json").read_text())
        sha = out["scope"]["input_sha256"]
        assert sha[str(ckpt)] == hashlib.sha256(b"CKPT").hexdigest()
        assert sha[str(data)] == hashlib.sha256(b"DATA").hexdigest()
        assert sha[str(nest)] == hashlib.sha256(b"NEST").hexdigest()

    def test_main_fails_closed_on_missing_input(self, tmp_path, monkeypatch):
        dataset = _stub_dataset()
        _patch_io(monkeypatch, dataset)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            ed.main([
                "--output_dir", str(tmp_path / "out"),
                "--checkpoint", str(tmp_path / "absent.bin"),
            ])


class TestAbsentLedgerFailsClosed:
    """A missing labeling_eligibility ledger is UNKNOWN, never removed=0."""

    def test_absent_ledger_is_null_unavailable(self, tmp_path, monkeypatch):
        dataset = _stub_dataset()
        dataset.pop("labeling_eligibility")
        _patch_io(monkeypatch, dataset)

        rc = ed.main(["--output_dir", str(tmp_path)])
        assert rc == 0
        out = json.loads((tmp_path / "error_decomposition.json").read_text())

        ledger = out["missingness_ledger"]
        assert ledger["status"] == "unavailable"
        assert ledger["n_unavailable"] is None  # UNKNOWN, not 0
        assert ledger["n_eligibility_rows"] is None
        assert ledger["missingness_assumption"] == "unavailable"
        assert ledger["biological_completeness"] == "not_established"
        assert ledger["reason"]
        # The removed-trial count must not masquerade as a verified zero.
        pa = out["padding_masking_audit"]
        assert pa["unavailable_trials_removed"] is None
        cmp_ = out["unavailable_trial_comparison"]
        assert cmp_["status"] == "unavailable"
        assert cmp_["n_unavailable_trials"] is None
        assert out["scope"]["missingness_ledger_scope"] == "unavailable_no_ledger"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
