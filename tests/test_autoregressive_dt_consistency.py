"""
Tests for physical clock consistency in scripts/simulate_autoregressive.py.

Verifies:
1. Model-driven rollout inherits the checkpoint clock; standalone synthesis defaults to 4.0 ms.
2. Stimulus paradigm generator respects dt_ms (sampling interval, total frames, pure wind prepend).
3. export_kinematics_csv computes dx = velocity * (dt_ms / 1000.0) without hardcoding 10.0 ms.
4. log_trial_summary calculates onset frame and peak latency using passed/trial dt_ms.
5. Zero hardcoded 10.0 ms temporal constants remain in simulate_autoregressive.py.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import logging
import re
from pathlib import Path
from typing import List

import numpy as np
import pytest

from scripts.simulate_autoregressive import (
    PARADIGMS,
    StimulusParadigm,
    TrialResult,
    export_kinematics_csv,
    generate_stimulus_paradigm,
    log_trial_summary,
)


# =========================================================================
# 1. Default dt_ms Parameter Values
# =========================================================================

def test_default_dt_ms_inherits_model_clock_for_rollout():
    """Only model-driven rollout uses None; standalone synthesis retains 4 ms."""
    sig_gen = inspect.signature(generate_stimulus_paradigm)
    assert sig_gen.parameters["dt_ms"].default == 4.0, (
        f"generate_stimulus_paradigm default dt_ms={sig_gen.parameters['dt_ms'].default}, expected 4.0"
    )

    from scripts.simulate_autoregressive import run_autoregressive_trial
    sig_trial = inspect.signature(run_autoregressive_trial)
    assert sig_trial.parameters["dt_ms"].default is None, (
        "run_autoregressive_trial must inherit model.dt_ms when no cadence is requested"
    )

    sig_res = inspect.signature(TrialResult)
    assert sig_res.parameters["dt_ms"].default == 4.0, (
        f"TrialResult default dt_ms={sig_res.parameters['dt_ms'].default}, expected 4.0"
    )


def test_cli_parser_dt_ms_default():
    """Model-driven CLI inherits the saved clock; explicit intervals are validated."""
    import ast
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "simulate_autoregressive.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    actions = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
               and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--dt_ms']
    assert len(actions) == 1
    default, = [kw.value for kw in actions[0].keywords if kw.arg == 'default']
    assert isinstance(default, ast.Constant) and default.value is None


# =========================================================================
# 2. Stimulus Generation Physics and Temporal Resolution
# =========================================================================

def test_generate_stimulus_paradigm_dt_scaling():
    """generate_stimulus_paradigm must generate correct dt_ms timestamps."""
    paradigm = PARADIGMS["visual_only"]

    for dt in [4.0, 2.0, 5.0, 10.0]:
        time_ms, v_vis, wind = generate_stimulus_paradigm(paradigm, dt_ms=dt)
        expected_frames = int(paradigm.total_duration_ms / dt)
        assert len(time_ms) == expected_frames
        assert len(v_vis) == expected_frames
        assert len(wind) == expected_frames
        # Timestamps must advance by exactly dt
        diffs = np.diff(time_ms)
        np.testing.assert_allclose(diffs, dt, rtol=1e-5)


def test_generate_stimulus_pure_wind_prepend_dt_scaling():
    """Pure-wind prepend frames must scale consistently with dt_ms."""
    from nsmor.data_extractor import _compute_pure_wind_prepend_frames

    paradigm = PARADIGMS["wind_only"]
    for dt in [4.0, 2.0, 10.0]:
        time_ms, v_vis, wind = generate_stimulus_paradigm(paradigm, dt_ms=dt)
        prepend = _compute_pure_wind_prepend_frames(dt)
        expected_frames = int(paradigm.total_duration_ms / dt) + prepend
        assert len(time_ms) == expected_frames
        diffs = np.diff(time_ms)
        np.testing.assert_allclose(diffs, dt, rtol=1e-5)


# =========================================================================
# 3. Kinematics Export Displacement (dx) Calculation
# =========================================================================

def _create_mock_trial(
    velocity_val: float = 100.0,
    n_frames: int = 10,
    dt_ms: float = 4.0,
) -> TrialResult:
    """Helper to build a deterministic TrialResult."""
    time_ms = np.arange(n_frames) * dt_ms
    return TrialResult(
        paradigm_name="mock_paradigm",
        time_ms=time_ms,
        position=np.cumsum(np.full(n_frames, velocity_val) * (dt_ms / 1000.0)),
        velocity=np.full(n_frames, velocity_val),
        acceleration=np.zeros(n_frames),
        v_vis=np.zeros(n_frames),
        wind=np.zeros(n_frames),
        gate_lif=np.full(n_frames, 0.5),
        gate_gru=np.full(n_frames, 0.5),
        target_ttc_ms=0.0,
        lv_ratio=120.0,
        dt_ms=dt_ms,
    )


def test_export_kinematics_csv_uses_dt_ms(tmp_path: Path):
    """export_kinematics_csv must calculate dx = v * (dt_ms / 1000.0)."""
    # Test 1: dt_ms = 4.0 ms, velocity = 100.0 cm/s -> dx = 0.400000 cm
    trial_4ms = _create_mock_trial(velocity_val=100.0, n_frames=5, dt_ms=4.0)
    out_csv = tmp_path / "kinematics_4ms.csv"
    export_kinematics_csv([trial_4ms], out_csv)

    with open(out_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 5
    for row in rows:
        assert row["dx"] == "0.400000", f"Expected dx=0.400000 for 4ms dt, got {row['dx']}"

    # Test 2: explicit dt_ms override = 2.0 ms -> dx = 0.200000 cm
    out_csv_2ms = tmp_path / "kinematics_2ms.csv"
    export_kinematics_csv([trial_4ms], out_csv_2ms, dt_ms=2.0)

    with open(out_csv_2ms, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        assert row["dx"] == "0.200000", f"Expected dx=0.200000 for 2ms dt, got {row['dx']}"

    # Test 3: trial with dt_ms=10.0 ms -> dx = 1.000000 cm
    trial_10ms = _create_mock_trial(velocity_val=100.0, n_frames=5, dt_ms=10.0)
    out_csv_10ms = tmp_path / "kinematics_10ms.csv"
    export_kinematics_csv([trial_10ms], out_csv_10ms)

    with open(out_csv_10ms, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        assert row["dx"] == "1.000000", f"Expected dx=1.000000 for 10ms dt, got {row['dx']}"


# =========================================================================
# 4. log_trial_summary Latency and Baseline Calculation
# =========================================================================

def test_log_trial_summary_onset_and_latency(caplog):
    """log_trial_summary must compute onset frame as 2000/dt_ms and latency as peak_frame*dt_ms."""
    # Build trial with 1250 frames at 4ms (5000ms total)
    dt = 4.0
    T = int(5000.0 / dt)  # 1250 frames
    time_ms = np.arange(T) * dt
    velocity = np.zeros(T)

    # 2s baseline = frame 500. Place peak at frame 550 (offset = 50 frames from onset)
    onset_frame = int(2000.0 / dt)  # 500
    peak_offset = 50
    velocity[onset_frame + peak_offset] = 25.0

    trial = TrialResult(
        paradigm_name="latency_test",
        time_ms=time_ms,
        position=np.zeros(T),
        velocity=velocity,
        acceleration=np.zeros(T),
        v_vis=np.zeros(T),
        wind=np.zeros(T),
        gate_lif=np.zeros(T),
        gate_gru=np.zeros(T),
        target_ttc_ms=0.0,
        lv_ratio=120.0,
        dt_ms=dt,
    )

    caplog.clear()
    with caplog.at_level(logging.INFO):
        log_trial_summary([trial], dt_ms=dt)

    # Expected latency = peak_offset * dt = 50 * 4.0 = 200.0 ms (NOT 50 * 10 = 500.0 ms!)
    log_text = caplog.text
    assert "latency_test" in log_text
    assert "200.0" in log_text, f"Expected latency 200.0 ms in log, got: {log_text}"


# =========================================================================
# 5. Code Integrity Audit
# =========================================================================

def test_no_hardcoded_10ms_constants():
    """Verify simulate_autoregressive.py contains no hardcoded 10.0 ms temporal calculations."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "simulate_autoregressive.py"
    content = script_path.read_text(encoding="utf-8")

    # Disallow hardcoded displacement step: (10.0 / 1000.0)
    assert not re.search(r'\(10\.0\s*/\s*1000\.0\)', content), "Found hardcoded (10.0 / 1000.0) in script"

    # Disallow hardcoded onset frame: 2000.0 / 10.0
    assert not re.search(r'2000\.0\s*/\s*10\.0', content), "Found hardcoded 2000.0 / 10.0 in script"

    # Disallow hardcoded latency multiplication: peak_frame * 10.0
    assert not re.search(r'peak_frame\s*\*\s*10\.0', content), "Found hardcoded peak_frame * 10.0 in script"
