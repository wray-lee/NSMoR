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

import json
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# ── Subjects under test ────────────────────────────────────────
from nsmor.pipeline.io import load_kinematics_csv, KINEMATICS_COLUMNS
from scripts.pre_load_adapt import adapt_cercus_to_nsmor, parse_trial_events


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

    def test_parsed_onset_used(self, tmp_path: Path):
        """When looming_onset_ms is present, parse_trial_events extracts it."""
        evt_path = tmp_path / "test_events.csv"
        evt_df = pd.DataFrame([
            {
                "session_id": "s0",
                "trial_id": 0,
                "time_ms": 0.0,
                "event_type": "trial_start",
                "event_value": json.dumps({
                    "type": "baseline_visual",
                    "lv_ratio_ms": 40.0,
                    "wind_dir": "none",
                }),
            },
            {
                "session_id": "s0",
                "trial_id": 0,
                "time_ms": 137.5,
                "event_type": "phase_transition",
                "event_value": json.dumps({
                    "from_phase": "Baseline",
                    "to_phase": "Looming",
                }),
            },
        ])
        evt_df.to_csv(evt_path, index=False)

        trial_info = parse_trial_events(evt_path)
        assert 0 in trial_info
        assert trial_info[0]["looming_onset_ms"] == pytest.approx(137.5)

    def test_guard_fires_on_large_deviation(self, tmp_path: Path):
        """Warning emitted when parsed onset deviates from trial start
        by more than one median frame gap.
        """
        session_dir = tmp_path / "session_001"
        session_dir.mkdir()
        evt_path = session_dir / "session_001_events.csv"
        kin_path = session_dir / "session_001_kinematics.csv"

        # Events with Looming transition at 100.0 ms (> 1 median gap of 4.0 ms)
        evt_df = pd.DataFrame([
            {
                "session_id": "session_001",
                "trial_id": 0,
                "time_ms": 0.0,
                "event_type": "trial_start",
                "event_value": json.dumps({
                    "type": "baseline_visual",
                    "lv_ratio_ms": 40.0,
                    "wind_dir": "none",
                }),
            },
            {
                "session_id": "session_001",
                "trial_id": 0,
                "time_ms": 100.0,
                "event_type": "phase_transition",
                "event_value": json.dumps({
                    "from_phase": "Baseline",
                    "to_phase": "Looming",
                }),
            },
        ])
        evt_df.to_csv(evt_path, index=False)

        n_samples = 50
        kin_df = pd.DataFrame({
            "session_id": ["session_001"] * n_samples,
            "trial_id": [0] * n_samples,
            "time_ms": np.arange(n_samples, dtype=np.float64) * 4.0,  # 0, 4, 8, ...
            "x_pos": np.zeros(n_samples),
            "y_pos": np.zeros(n_samples),
            "heading": np.zeros(n_samples),
            "velocity": np.zeros(n_samples),
            "acceleration": np.zeros(n_samples),
            "visual_angle": np.zeros(n_samples),
            "wind_state": np.zeros(n_samples, dtype=int),
            "l_v_ratio": np.zeros(n_samples),
        })
        kin_df.to_csv(kin_path, index=False)

        with pytest.warns(UserWarning, match="deviates from trial start"):
            adapt_cercus_to_nsmor(raw_dir=str(tmp_path))

    def test_fallback_when_missing(self, tmp_path: Path):
        """When looming_onset_ms is None, stimulus_onset falls back to 0.0."""
        session_dir = tmp_path / "session_001"
        session_dir.mkdir()
        evt_path = session_dir / "session_001_events.csv"
        kin_path = session_dir / "session_001_kinematics.csv"

        # Events without Looming phase transition
        evt_df = pd.DataFrame([
            {
                "session_id": "session_001",
                "trial_id": 0,
                "time_ms": 0.0,
                "event_type": "trial_start",
                "event_value": json.dumps({
                    "type": "baseline_visual",
                    "lv_ratio_ms": 40.0,
                    "wind_dir": "none",
                }),
            }
        ])
        evt_df.to_csv(evt_path, index=False)

        # 1. parse_trial_events contract: looming_onset_ms is None
        trial_info = parse_trial_events(evt_path)
        assert trial_info[0]["looming_onset_ms"] is None

        # 2. adapt_cercus_to_nsmor fallback contract: stimulus_onset at 0.0
        n_samples = 20
        kin_df = pd.DataFrame({
            "session_id": ["session_001"] * n_samples,
            "trial_id": [0] * n_samples,
            "time_ms": np.arange(n_samples, dtype=np.float64) * 4.0,
            "x_pos": np.zeros(n_samples),
            "y_pos": np.zeros(n_samples),
            "heading": np.zeros(n_samples),
            "velocity": np.zeros(n_samples),
            "acceleration": np.zeros(n_samples),
            "visual_angle": np.zeros(n_samples),
            "wind_state": np.zeros(n_samples, dtype=int),
            "l_v_ratio": np.zeros(n_samples),
        })
        kin_df.to_csv(kin_path, index=False)

        adapt_cercus_to_nsmor(raw_dir=str(tmp_path))

        rewritten_events = pd.read_csv(evt_path)
        stim_onset_events = rewritten_events[rewritten_events["event_type"] == "stimulus_onset"]
        assert len(stim_onset_events) == 1
        assert stim_onset_events.iloc[0]["time_ms"] == pytest.approx(0.0)


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


# ═══════════════════════════════════════════════════════════════
# F. Cross-path consistency: load_kinematics_csv vs
#    apply_hardware_time_correction on same trajectory
# ═══════════════════════════════════════════════════════════════

class TestCrossPathConsistency:
    """Both public kinematics paths agree with one analytical 2-D trajectory.

    The same irregularly sampled trajectory is sent through both public
    loaders.  It is a circular arc around a *displaced* centre with a linearly
    increasing angular speed:

        x(t) = c_x + R cos(theta(t))
        y(t) = c_y + R sin(theta(t))
        theta(t) = theta_0 + omega_0 t + 0.5 alpha t²

    Therefore the independently derived Cartesian path-speed oracle is
    ``|dr/dt| = R (omega_0 + alpha t)`` and its derivative is
    ``d|dr/dt|/dt = R alpha``.  The centre is offset from the origin, so the
    radial derivative ``d|r|/dt`` is a different quantity and cannot satisfy
    the speed oracle.

    Timestamps have positive 12 +/- 3 ms gaps while ``dt_ms=4`` is passed to
    the correction path.  A nominal-dt implementation therefore has a large,
    detectable scale error rather than being hidden by near-uniform jitter.

    Biological basis: a cercal recording can contain frame-drop / USB timing
    jitter, while escape trajectories are genuinely two-dimensional and may
    curve as the animal turns.  Kinematic units remain physical (cm/s and
    cm/s²) by differentiating against the observed hardware timestamps.
    """

    @pytest.fixture()
    def trajectory(self) -> Dict[str, np.ndarray]:
        """Return one non-radial 2-D trajectory and its analytical oracle."""
        n: int = 180
        radius_cm: float = 5.0
        center_x_cm: float = 1.0
        center_y_cm: float = -0.5
        theta_0_rad: float = 0.31
        omega_0_rad_s: float = 0.4
        alpha_rad_s2: float = 0.2

        # Positive, strongly nonuniform gaps: observed median is ~12 ms,
        # whereas the correction path receives a deliberately stale 4 ms
        # nominal configuration.  The 41-frame period avoids repeating a
        # two-point alternation that could look deceptively uniform.
        gaps_ms = 12.0 + 3.0 * np.sin(
            2.0 * np.pi * np.arange(n - 1) / 41.0
        )
        assert np.all(gaps_ms > 0.0), "All synthetic timestamp gaps must be positive"
        time_ms = np.concatenate([[0.0], np.cumsum(gaps_ms)])
        t_s = time_ms / 1000.0

        theta_rad = (
            theta_0_rad
            + omega_0_rad_s * t_s
            + 0.5 * alpha_rad_s2 * t_s**2
        )
        x_pos = center_x_cm + radius_cm * np.cos(theta_rad)
        y_pos = center_y_cm + radius_cm * np.sin(theta_rad)

        # Analytical oracle, derived from d[x(t), y(t)]/dt rather than any
        # implementation under test.
        angular_speed_rad_s = omega_0_rad_s + alpha_rad_s2 * t_s
        expected_velocity = radius_cm * angular_speed_rad_s
        expected_acceleration = np.full(n, radius_cm * alpha_rad_s2)

        unique_gaps = np.unique(np.round(np.diff(time_ms), 8))
        assert len(unique_gaps) >= 7, (
            f"Timestamps must be genuinely irregular, got {len(unique_gaps)} gaps"
        )

        return {
            "n": n,
            "time_ms": time_ms,
            "t_s": t_s,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "velocity": expected_velocity,
            "acceleration": expected_acceleration,
        }

    def test_fixture_separates_real_time_and_path_speed_semantics(
        self, trajectory: Dict[str, np.ndarray],
    ) -> None:
        """Guard that both requested mutants are far outside acceptance."""
        time_ms = trajectory["time_ms"]
        t_s = trajectory["t_s"]
        expected_velocity = trajectory["velocity"]
        gaps_ms = np.diff(time_ms)
        interior = slice(15, -15)

        assert np.all(gaps_ms > 0.0)
        assert np.ptp(gaps_ms) > 5.9
        # The configured nominal interval is deliberately about 3x too short.
        assert np.median(gaps_ms) / 4.0 > 2.9

        # d|r|/dt is analytically distinct from Cartesian path speed because
        # the circle is not centred on the coordinate origin.
        radial_speed = np.abs(
            np.gradient(np.hypot(trajectory["x_pos"], trajectory["y_pos"]), t_s)
        )
        assert np.max(radial_speed[interior] / expected_velocity[interior]) < 0.25

    def test_both_public_paths_match_independent_oracle(
        self, tmp_path: Path, trajectory: Dict[str, np.ndarray],
    ) -> None:
        """Both paths recover physical units from one irregular 2-D trace."""
        from scripts.prepare_data import apply_hardware_time_correction

        n = len(trajectory["time_ms"])
        expected_velocity = trajectory["velocity"]
        expected_acceleration = trajectory["acceleration"]

        # Public path 1: load measured speed and derive dv/dt from timestamps.
        csv_path = _make_kin_csv(
            tmp_path,
            time_ms=trajectory["time_ms"],
            x_pos=trajectory["x_pos"],
            y_pos=trajectory["y_pos"],
            velocity=expected_velocity,
        )
        loaded = load_kinematics_csv(
            csv_path, artifact_velocity_cm_s=float("inf"),
        )

        # Public path 2: derive 2-D speed and acceleration from positions.
        kinematics = pd.DataFrame({
            "session_id": ["s0"] * n,
            "trial_id": [0] * n,
            "time_ms": trajectory["time_ms"],
            "x_pos": trajectory["x_pos"],
            "y_pos": trajectory["y_pos"],
            "heading": np.zeros(n),
            "velocity": np.zeros(n),
            "acceleration": np.zeros(n),
            "visual_angle": np.zeros(n),
            "wind_state": np.zeros(n, dtype=int),
            "l_v_ratio": np.zeros(n),
        })
        events = pd.DataFrame({
            "session_id": ["s0"],
            "trial_id": [0],
            "time_ms": [0.0],
            "event_type": ["stimulus_onset"],
            "event_value": ["{}"],
        })
        corrected, _ = apply_hardware_time_correction(
            kinematics, events, hw_triggers={}, dt_ms=4.0,
        )

        loaded_acceleration = loaded["acceleration"].to_numpy()
        corrected_velocity = corrected["velocity"].to_numpy()
        corrected_acceleration = corrected["acceleration"].to_numpy()
        assert loaded_acceleration.shape == (n,)
        assert corrected_velocity.shape == (n,)
        assert corrected_acceleration.shape == (n,)

        assert loaded_acceleration[0] == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(
            loaded_acceleration[1:],
            expected_acceleration[1:],
            rtol=1e-10,
            atol=1e-10,
        )

        # Exclude SavGol/gradient boundary support; the oracle itself is exact.
        interior = slice(15, -15)
        np.testing.assert_allclose(
            corrected_velocity[interior],
            expected_velocity[interior],
            rtol=0.01,
            atol=0.01,
        )
        np.testing.assert_allclose(
            corrected_acceleration[interior],
            expected_acceleration[interior],
            rtol=0.05,
            atol=0.01,
        )
