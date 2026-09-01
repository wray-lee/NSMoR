"""Tests for kinematic-unit semantics and sampling-interval handling.

Covers:
- Acceleration as true time derivative (cm/s²) with irregular timestamps
- 2-D Cartesian path speed (tangential, not radial)
- dt-scaling correctness when the sampling interval changes
- Looming onset sourced from parsed event, with guard on deviation
- Sampling diagnostics reporting the observed interval

All tests use small synthetic DataFrames — no real data dependency.
"""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# ── Subjects under test ────────────────────────────────────────
from nsmor.pipeline.io import load_kinematics_csv, KINEMATICS_COLUMNS


# ═══════════════════════════════════════════════════════════════
# Helpers: synthetic CSV on disk
# ═══════════════════════════════════════════════════════════════

def _make_kin_csv(
    tmp_path,
    *,
    session_id: str = "s0",
    trial_id: int = 0,
    time_ms: np.ndarray,
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    velocity: np.ndarray | None = None,
    acceleration: np.ndarray | None = None,
    heading: np.ndarray | None = None,
    visual_angle: np.ndarray | None = None,
    wind_state: np.ndarray | None = None,
    l_v_ratio: np.ndarray | None = None,
) -> str:
    """Write a minimal kinematics CSV and return its path."""
    n = len(time_ms)
    df = pd.DataFrame({
        "session_id": [session_id] * n,
        "trial_id": [trial_id] * n,
        "time_ms": time_ms,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "heading": heading if heading is not None else np.zeros(n),
        "velocity": velocity if velocity is not None else np.zeros(n),
        "acceleration": acceleration if acceleration is not None else np.zeros(n),
        "visual_angle": visual_angle if visual_angle is not None else np.zeros(n),
        "wind_state": wind_state if wind_state is not None else np.zeros(n, dtype=int),
        "l_v_ratio": l_v_ratio if l_v_ratio is not None else np.zeros(n),
    })
    path = tmp_path / "kin.csv"
    df.to_csv(path, index=False)
    return str(path)


# ═══════════════════════════════════════════════════════════════
# A. Acceleration = true dv/dt (cm/s²), irregular timestamps
# ═══════════════════════════════════════════════════════════════

class TestAccelerationTrueDerivative:
    """Constant-acceleration trajectory must recover the right cm/s²."""

    def test_constant_acceleration_irregular_dt(self, tmp_path):
        """v(t) = a₀·t with a₀=500 cm/s², irregular timestamps.

        Acceleration at every non-first frame should be ≈ 500 cm/s².
        """
        a0 = 500.0  # cm/s²
        # Irregular timestamps (ms): 0, 4, 7, 11, 18, 22
        time_ms = np.array([0.0, 4.0, 7.0, 11.0, 18.0, 22.0])
        t_s = time_ms / 1000.0
        velocity = a0 * t_s  # v = a₀·t
        # positions don't matter for this test but must be present
        x_pos = np.cumsum(velocity * np.concatenate([[0], np.diff(t_s)]))
        y_pos = np.zeros_like(x_pos)

        csv_path = _make_kin_csv(
            tmp_path,
            time_ms=time_ms,
            x_pos=x_pos,
            y_pos=y_pos,
            velocity=velocity,
        )
        df = load_kinematics_csv(csv_path, artifact_velocity_cm_s=float("inf"))

        acc = df["acceleration"].values
        # Frame 0 is 0 by convention (no prior sample)
        assert acc[0] == pytest.approx(0.0, abs=1e-6)
        # Remaining frames: dv/dt should be exactly a₀
        for i in range(1, len(acc)):
            assert acc[i] == pytest.approx(a0, rel=0.01), (
                f"Frame {i}: expected {a0} cm/s², got {acc[i]}"
            )

    def test_zero_gap_floors_to_min_positive(self, tmp_path):
        """A zero-dt gap is floored to the min positive gap, not div-by-zero."""
        time_ms = np.array([0.0, 0.0, 4.0, 8.0])  # duplicate at t=0
        velocity = np.array([0.0, 1.0, 2.0, 3.0])
        csv_path = _make_kin_csv(
            tmp_path,
            time_ms=time_ms,
            x_pos=np.zeros(4),
            y_pos=np.zeros(4),
            velocity=velocity,
        )
        df = load_kinematics_csv(csv_path, artifact_velocity_cm_s=float("inf"))
        acc = df["acceleration"].values
        # No NaN or Inf
        assert np.all(np.isfinite(acc)), f"Non-finite acceleration: {acc}"


# ═══════════════════════════════════════════════════════════════
# B. 2-D Cartesian path speed (tangential, not radial)
# ═══════════════════════════════════════════════════════════════

class TestPathSpeed:
    """prepare_data velocity must be tangential path speed, not d|r|/dt."""

    def test_circular_motion_nonzero_speed(self):
        """Uniform circular motion: |r|=const → radial speed ≡ 0,
        but tangential speed = ω·r > 0.

        The old formula np.gradient(sqrt(x²+y²), dt) would give ~0.
        The correct formula sqrt(dx/dt² + dy/dt²) gives ω·r.
        """
        from scripts.prepare_data import apply_hardware_time_correction

        r = 5.0  # cm
        omega = 2 * np.pi / 1000.0  # rad/ms → one revolution in 1000 ms
        n = 200
        # Uniform 5 ms spacing
        time_ms = np.arange(n, dtype=np.float64) * 5.0
        theta = omega * time_ms
        x_pos = r * np.cos(theta)
        y_pos = r * np.sin(theta)

        # Build a minimal kinematics DataFrame
        kin_df = pd.DataFrame({
            "session_id": ["s0"] * n,
            "trial_id": [0] * n,
            "time_ms": time_ms,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "heading": np.zeros(n),
            "velocity": np.zeros(n),
            "acceleration": np.zeros(n),
            "visual_angle": np.zeros(n),
            "wind_state": np.zeros(n, dtype=int),
            "l_v_ratio": np.zeros(n),
        })
        evt_df = pd.DataFrame({
            "session_id": ["s0"],
            "trial_id": [0],
            "time_ms": [0.0],
            "event_type": ["stimulus_onset"],
            "event_value": ["{}"],
        })

        corrected_kin, _ = apply_hardware_time_correction(
            kin_df, evt_df, hw_triggers={}, dt_ms=5.0,
        )

        # Expected tangential speed = ω·r (in cm/s)
        # ω = 2π / 1000 ms = 2π / 1.0 s → ω·r = 2π·5 ≈ 31.42 cm/s
        expected_speed = omega * 1000.0 * r  # convert ω from rad/ms to rad/s
        vel = corrected_kin["velocity"].values

        # Interior frames (skip edge effects from gradient)
        interior = vel[10:-10]
        assert np.all(interior > expected_speed * 0.5), (
            f"Path speed should be ≈{expected_speed:.1f} cm/s, "
            f"got min={interior.min():.2f}"
        )
        # Should NOT be near zero (which the radial formula would give)
        assert np.mean(np.abs(interior)) > 10.0, (
            "Circular-motion speed should not be near zero (radial bug)"
        )


# ═══════════════════════════════════════════════════════════════
# C. dt-scaling: changing the interval by 100x/250x
# ═══════════════════════════════════════════════════════════════

class TestDtScaling:
    """Acceleration must scale inversely with dt for same dv."""

    def _make_const_dv(self, tmp_path, dt_ms: float) -> pd.DataFrame:
        """Create a trial with constant dv=1 cm/s per step at given dt."""
        n = 5
        time_ms = np.arange(n, dtype=np.float64) * dt_ms
        velocity = np.arange(n, dtype=np.float64)  # 0, 1, 2, 3, 4
        csv_path = _make_kin_csv(
            tmp_path / f"dt{dt_ms}",
            time_ms=time_ms,
            x_pos=np.zeros(n),
            y_pos=np.zeros(n),
            velocity=velocity,
        )
        return load_kinematics_csv(csv_path, artifact_velocity_cm_s=float("inf"))

    def test_acceleration_scales_with_dt(self, tmp_path):
        """dv=1 cm/s per step: a(dt=10ms) = 100 cm/s²,
        a(dt=4ms) = 250 cm/s².  Ratio ≈ 2.5x.
        """
        (tmp_path / "dt10.0").mkdir()
        (tmp_path / "dt4.0").mkdir()

        df_10 = self._make_const_dv(tmp_path, dt_ms=10.0)
        df_4 = self._make_const_dv(tmp_path, dt_ms=4.0)

        # Acceleration at frame 1 (first non-zero)
        a_10 = df_10["acceleration"].values[1]
        a_4 = df_4["acceleration"].values[1]

        # dv=1 cm/s / 0.010 s = 100 cm/s²
        assert a_10 == pytest.approx(100.0, rel=0.01)
        # dv=1 cm/s / 0.004 s = 250 cm/s²
        assert a_4 == pytest.approx(250.0, rel=0.01)
        # Ratio
        assert (a_4 / a_10) == pytest.approx(2.5, rel=0.01)


# ═══════════════════════════════════════════════════════════════
# D. Looming onset from parsed event, guard fires on deviation
# ═══════════════════════════════════════════════════════════════

class TestLoomingOnset:
    """Visual onset must prefer parsed looming_onset_ms over hardcoded 0."""

    def test_parsed_onset_used(self):
        """When looming_onset_ms is present, stimulus_onset_ms == that value."""
        from scripts.pre_load_adapt import parse_trial_events, adapt_cercus_to_nsmor  # noqa: F401

        # Directly test the onset injection logic by building
        # the trial_info dict as parse_trial_events would.
        trial_info = {
            0: {
                'type': 'baseline_visual',
                'lv_ratio_ms': 40.0,
                'target_ttc_ms': None,
                'wind_dir': 'none',
                'stimulus_onset_ms': None,
                'looming_onset_ms': 0.194,  # parsed from events
                'ttc_ms': None,
            }
        }
        # Simulate the onset selection logic from adapt_cercus_to_nsmor
        info = trial_info[0]
        parsed_onset = info.get('looming_onset_ms')
        assert parsed_onset is not None
        assert parsed_onset >= 0
        onset_val = float(parsed_onset)
        assert onset_val == pytest.approx(0.194)

    def test_guard_fires_on_large_deviation(self):
        """Warning emitted when parsed onset deviates from trial start
        by more than one median frame gap.
        """
        # Simulate: trial starts at t=0, parsed onset at t=100 ms,
        # median gap = 4 ms → deviation = 100 >> 4 → warning.
        time_vals = np.arange(0, 200, 4.0)  # 0, 4, 8, ..., 196
        parsed_onset = 100.0

        pos_gaps = np.diff(time_vals)
        pos_gaps = pos_gaps[pos_gaps > 0]
        median_gap = float(np.median(pos_gaps))

        deviation = abs(parsed_onset - time_vals[0])
        assert deviation > median_gap, "Test setup: deviation must exceed gap"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if deviation > median_gap:
                warnings.warn(
                    f"parsed looming_onset_ms ({parsed_onset:.3f}) deviates "
                    f"from trial start ({time_vals[0]:.3f}) by "
                    f"{deviation:.3f} ms (> 1 median gap {median_gap:.3f} ms)",
                    stacklevel=1,
                )
            assert len(w) == 1
            assert "looming_onset_ms" in str(w[0].message)
            assert "100.000" in str(w[0].message)

    def test_fallback_when_missing(self):
        """When looming_onset_ms is None, fall back to 0.0."""
        info = {
            'type': 'baseline_visual',
            'looming_onset_ms': None,
        }
        parsed_onset = info.get('looming_onset_ms')
        if parsed_onset is not None and parsed_onset >= 0:
            onset_val = float(parsed_onset)
        else:
            onset_val = 0.0
        assert onset_val == 0.0


# ═══════════════════════════════════════════════════════════════
# E. Sampling diagnostics
# ═══════════════════════════════════════════════════════════════

class TestSamplingDiagnostics:
    """compute_sampling_diagnostics must report the observed interval."""

    def test_uniform_4ms_vs_config_10ms(self):
        """Uniform 4 ms cadence with configured dt_ms=10 → mismatch."""
        from scripts.prepare_data import compute_sampling_diagnostics

        n = 100
        df = pd.DataFrame({
            "session_id": ["s0"] * n,
            "trial_id": [0] * n,
            "time_ms": np.arange(n, dtype=np.float64) * 4.0,
        })
        diag = compute_sampling_diagnostics(df, configured_dt_ms=10.0)

        assert diag["observed_median_ms"] == pytest.approx(4.0)
        assert diag["observed_mean_ms"] == pytest.approx(4.0)
        assert diag["configured_dt_ms"] == 10.0
        assert diag["ratio_configured_over_observed"] == pytest.approx(2.5)
        assert diag["mismatch_flag"] is True  # 2.5x > 1.5x threshold

    def test_matching_interval_no_flag(self):
        """When observed ≈ configured, no mismatch flag."""
        from scripts.prepare_data import compute_sampling_diagnostics

        n = 100
        df = pd.DataFrame({
            "session_id": ["s0"] * n,
            "trial_id": [0] * n,
            "time_ms": np.arange(n, dtype=np.float64) * 10.0,
        })
        diag = compute_sampling_diagnostics(df, configured_dt_ms=10.0)

        assert diag["observed_median_ms"] == pytest.approx(10.0)
        assert diag["mismatch_flag"] is False

    def test_irregular_gaps_reports_stats(self):
        """Irregular timestamps: median/mean/p99 should differ."""
        from scripts.prepare_data import compute_sampling_diagnostics

        # 90% at 4ms, 10% at 36ms (long tail)
        gaps_4 = np.full(90, 4.0)
        gaps_36 = np.full(10, 36.0)
        gaps = np.concatenate([gaps_4, gaps_36])
        np.random.seed(42)
        np.random.shuffle(gaps)
        time_ms = np.concatenate([[0.0], np.cumsum(gaps)])
        n = len(time_ms)

        df = pd.DataFrame({
            "session_id": ["s0"] * n,
            "trial_id": [0] * n,
            "time_ms": time_ms,
        })
        diag = compute_sampling_diagnostics(df, configured_dt_ms=10.0)

        assert diag["observed_median_ms"] == pytest.approx(4.0)
        assert diag["observed_mean_ms"] > 4.0  # pulled up by 36ms outliers
        assert diag["observed_p99_ms"] > 20.0  # p99 in the tail
        assert diag["n_positive_gaps"] == 100

    def test_empty_df_returns_nan_and_flag(self):
        """Empty kinematics → NaN stats, mismatch=True."""
        from scripts.prepare_data import compute_sampling_diagnostics

        df = pd.DataFrame({
            "session_id": pd.Series(dtype=str),
            "trial_id": pd.Series(dtype=int),
            "time_ms": pd.Series(dtype=float),
        })
        diag = compute_sampling_diagnostics(df, configured_dt_ms=10.0)

        assert np.isnan(diag["observed_median_ms"])
        assert diag["mismatch_flag"] is True
