"""RED/GREEN tests for event-time correctness in raw-schema adaptation.

Covers the base-inherited time bug where parse_trial_events stored raw
timestamp seconds as looming_onset_ms and the visual branch injected that
value directly into canonical stimulus_onset time_ms.

Contract:
  * Raw schema (``timestamp`` seconds): stimulus_onset time_ms must be
    per-trial relative milliseconds sharing the kinematics coordinate
    origin, not raw seconds or session-absolute milliseconds.
  * Canonical schema (``time_ms`` already ms): no double scaling.
  * Same constant offset on event+kin seconds => same trial-relative
    canonical event times and downstream anchor selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from scripts.pre_load_adapt import (
    adapt_cercus_to_nsmor,
    parse_trial_events,
    reconstruct_visual_angle,
)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _write_raw_pair(
    directory: Path,
    session_id: str,
    *,
    trial_start_s: float,
    looming_s: float,
    kin_first_s: float,
    n_samples: int = 30,
    trial_id: int = 1,
    trial_type: str = "baseline_visual",
    lv_ratio_ms: float = 120.0,
    kin_dt_s: float = 0.004,
    wind_state: int = 0,
    extra_events: List[Dict] | None = None,
) -> Tuple[Path, Path]:
    """Write a raw-schema kinematics + events pair with explicit absolute times.

    Args:
        trial_start_s: Absolute timestamp (seconds) of trial_start event.
        looming_s: Absolute timestamp (seconds) of Looming phase_transition.
        kin_first_s: Absolute sys_time (seconds) of first kinematics sample.
    """
    directory.mkdir(parents=True, exist_ok=True)
    kin_path = directory / f"{session_id}_kinematics.csv"
    evt_path = directory / f"{session_id}_events.csv"

    sys_time = kin_first_s + np.arange(n_samples, dtype=np.float64) * kin_dt_s
    df_k = pd.DataFrame({
        "sys_time": sys_time,
        "ard_time": (sys_time * 1000.0).astype(np.int64),
        "dx": np.full(n_samples, 0.1),
        "dy": np.full(n_samples, 0.05),
        "dz": np.zeros(n_samples),
        "stim_state": np.full(n_samples, wind_state, dtype=np.int64),
        "global_trial_id": np.full(n_samples, trial_id, dtype=np.int64),
    })
    df_k.to_csv(kin_path, index=False)

    details = json.dumps({
        "type": trial_type,
        "target_ttc_ms": None,
        "lv_ratio_ms": lv_ratio_ms,
        "wind_dir": "none",
        "screen_side": "right",
    })
    rows = [
        {
            "event_name": "trial_start",
            "timestamp": trial_start_s,
            "session_num": 1,
            "trial_in_session": 1,
            "global_trial_id": trial_id,
            "details": details,
        },
        {
            "event_name": "phase_transition",
            "timestamp": looming_s,
            "session_num": 1,
            "trial_in_session": 1,
            "global_trial_id": trial_id,
            "details": json.dumps({
                "from_phase": "TrialStart",
                "to_phase": "Looming",
            }),
        },
    ]
    if extra_events:
        rows.extend(extra_events)
    pd.DataFrame(rows).to_csv(evt_path, index=False)
    return kin_path, evt_path


def _get_stimulus_onset(evt_df: pd.DataFrame, trial_id: int) -> float:
    """Return time_ms of the stimulus_onset event for a trial."""
    mask = (evt_df["trial_id"] == trial_id) & (evt_df["event_type"] == "stimulus_onset")
    assert mask.sum() == 1, (
        f"Expected exactly 1 stimulus_onset for trial {trial_id}, "
        f"got {mask.sum()}"
    )
    return float(evt_df.loc[mask, "time_ms"].iloc[0])


def _get_phase_time(evt_df: pd.DataFrame, trial_id: int, to_phase: str) -> float:
    """Return time_ms of the phase_transition with given to_phase."""
    for _, row in evt_df[evt_df["trial_id"] == trial_id].iterrows():
        if row["event_type"] != "phase_transition":
            continue
        try:
            d = json.loads(str(row["event_value"]))
        except Exception:
            continue
        if d.get("to_phase") == to_phase:
            return float(row["time_ms"])
    raise AssertionError(f"No phase_transition to {to_phase!r} for trial {trial_id}")


# ═══════════════════════════════════════════════════════════════
# A. Core RED: raw seconds → per-trial relative ms
# ═══════════════════════════════════════════════════════════════

class TestRawEventTimeNormalization:
    """Raw timestamp seconds must become per-trial relative milliseconds."""

    def test_raw_looming_onset_20ms_not_012(self, tmp_path: Path):
        """trial_start=.100s, Looming=.120s, first kin=.100s → event 20ms.

        The bug injects 0.12 (raw seconds) instead of 20 (ms, per-trial).
        Assert observed output rows, not helper arithmetic.
        """
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_time_test"
        _write_raw_pair(
            raw_dir,
            s1,
            trial_start_s=0.100,
            looming_s=0.120,
            kin_first_s=0.100,
        )
        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        onset = _get_stimulus_onset(evt_out, trial_id=1)
        assert onset == pytest.approx(20.0, abs=1e-6), (
            f"Expected stimulus_onset at 20ms (per-trial relative), got {onset}"
        )
        # Must also agree with the authoritative Looming transition
        looming_t = _get_phase_time(evt_out, trial_id=1, to_phase="Looming")
        assert onset == pytest.approx(looming_t, abs=1e-6), (
            f"stimulus_onset {onset} != Looming transition {looming_t}"
        )

    def test_raw_looming_onset_not_raw_seconds(self, tmp_path: Path):
        """Guard: injected value must NOT be the raw seconds timestamp."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_raw_guard"
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100, looming_s=0.120, kin_first_s=0.100,
        )
        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        onset = _get_stimulus_onset(evt_out, trial_id=1)
        assert onset != pytest.approx(0.120, abs=1e-6), (
            "stimulus_onset still holds raw seconds (0.12)"
        )
        assert onset > 1.0, (
            f"stimulus_onset {onset} looks like seconds, not ms"
        )


# ═══════════════════════════════════════════════════════════════
# B. Offset / multi-session invariance
# ═══════════════════════════════════════════════════════════════

class TestOffsetInvariance:
    """Constant session-absolute offsets must not affect trial-relative times."""

    def test_same_offset_same_relative_times(self, tmp_path: Path):
        """Adding constant C to event+kin seconds → same canonical event times."""
        results = []
        for tag, offset in [("base", 0.0), ("offset", 1000.0)]:
            raw_dir = tmp_path / f"raw_{tag}"
            out_dir = tmp_path / f"out_{tag}"
            s1 = f"session_{tag}"
            _write_raw_pair(
                raw_dir, s1,
                trial_start_s=offset + 0.100,
                looming_s=offset + 0.120,
                kin_first_s=offset + 0.100,
            )
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
            evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
            results.append(_get_stimulus_onset(evt_out, trial_id=1))

        assert results[0] == pytest.approx(results[1], abs=1e-6), (
            f"Offset changed trial-relative onset: {results}"
        )
        assert results[0] == pytest.approx(20.0, abs=1e-6)

    def test_two_sessions_different_origins(self, tmp_path: Path):
        """Two sessions with distinct absolute clock origins → same relative times."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_origin_a"
        s2 = "session_origin_b"
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100, looming_s=0.120, kin_first_s=0.100,
        )
        _write_raw_pair(
            raw_dir, s2,
            trial_start_s=500.100, looming_s=500.120, kin_first_s=500.100,
        )
        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        onset_a = _get_stimulus_onset(
            pd.read_csv(out_dir / s1 / f"{s1}_events.csv"), trial_id=1,
        )
        onset_b = _get_stimulus_onset(
            pd.read_csv(out_dir / s2 / f"{s2}_events.csv"), trial_id=1,
        )
        assert onset_a == pytest.approx(20.0, abs=1e-6)
        assert onset_b == pytest.approx(20.0, abs=1e-6)

    def test_multi_trial_per_trial_relative(self, tmp_path: Path):
        """Multiple trials in one session: each gets its own per-trial origin."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_multi"
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"
        raw_dir.mkdir(parents=True)

        # Trial 1: start=10.000s, looming=10.020s
        # Trial 2: start=20.000s, looming=20.020s
        n = 20
        dt = 0.004
        sys_times = []
        trial_ids = []
        for tid, t0 in [(1, 10.0), (2, 20.0)]:
            for i in range(n):
                sys_times.append(t0 + i * dt)
                trial_ids.append(tid)
        df_k = pd.DataFrame({
            "sys_time": sys_times,
            "ard_time": (np.array(sys_times) * 1000).astype(np.int64),
            "dx": np.full(len(sys_times), 0.1),
            "dy": np.full(len(sys_times), 0.05),
            "dz": np.zeros(len(sys_times)),
            "stim_state": np.zeros(len(sys_times), dtype=np.int64),
            "global_trial_id": trial_ids,
        })
        df_k.to_csv(kin_path, index=False)

        rows = []
        for tid, t0 in [(1, 10.0), (2, 20.0)]:
            details = json.dumps({
                "type": "baseline_visual",
                "target_ttc_ms": None,
                "lv_ratio_ms": 120.0,
                "wind_dir": "none",
                "screen_side": "right",
            })
            rows.append({
                "event_name": "trial_start",
                "timestamp": t0,
                "session_num": 1,
                "trial_in_session": tid,
                "global_trial_id": tid,
                "details": details,
            })
            rows.append({
                "event_name": "phase_transition",
                "timestamp": t0 + 0.020,
                "session_num": 1,
                "trial_in_session": tid,
                "global_trial_id": tid,
                "details": json.dumps({
                    "from_phase": "TrialStart",
                    "to_phase": "Looming",
                }),
            })
        pd.DataFrame(rows).to_csv(evt_path, index=False)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        for tid in (1, 2):
            onset = _get_stimulus_onset(evt_out, trial_id=tid)
            assert onset == pytest.approx(20.0, abs=1e-6), (
                f"Trial {tid}: expected 20ms, got {onset}"
            )


# ═══════════════════════════════════════════════════════════════
# C. Canonical schema: no double conversion
# ═══════════════════════════════════════════════════════════════

class TestCanonicalSchemaNoDoubleConvert:
    """Canonical time_ms input must not be scaled by 1000 again."""

    def test_canonical_looming_onset_preserved(self, tmp_path: Path):
        """Canonical events with time_ms=137.5 → stimulus_onset stays 137.5."""
        session_dir = tmp_path / "session_canon"
        session_dir.mkdir()
        kin_path = session_dir / "session_canon_kinematics.csv"
        evt_path = session_dir / "session_canon_events.csv"

        n = 50
        pd.DataFrame({
            "session_id": ["session_canon"] * n,
            "trial_id": [0] * n,
            "time_ms": np.arange(n, dtype=np.float64) * 4.0,
            "x_pos": np.zeros(n),
            "y_pos": np.zeros(n),
            "heading": np.zeros(n),
            "velocity": np.zeros(n),
            "acceleration": np.zeros(n),
            "visual_angle": np.zeros(n),
            "wind_state": np.zeros(n, dtype=int),
            "l_v_ratio": np.zeros(n),
        }).to_csv(kin_path, index=False)

        pd.DataFrame([
            {
                "session_id": "session_canon",
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
                "session_id": "session_canon",
                "trial_id": 0,
                "time_ms": 137.5,
                "event_type": "phase_transition",
                "event_value": json.dumps({
                    "from_phase": "Baseline",
                    "to_phase": "Looming",
                }),
            },
        ]).to_csv(evt_path, index=False)

        # parse_trial_events: canonical value must be returned as-is
        info = parse_trial_events(evt_path)
        assert info[0]["looming_onset_ms"] == pytest.approx(137.5)

        # Full adapt: injected stimulus_onset must match
        adapt_cercus_to_nsmor(raw_dir=str(tmp_path))
        evt_out = pd.read_csv(evt_path)
        onset = _get_stimulus_onset(evt_out, trial_id=0)
        assert onset == pytest.approx(137.5, abs=1e-6), (
            f"Canonical onset double-converted: {onset}"
        )

    def test_canonical_ms_not_multiplied(self, tmp_path: Path):
        """Guard: canonical value 137.5 must not become 137500."""
        session_dir = tmp_path / "session_canon2"
        session_dir.mkdir()
        n = 20
        pd.DataFrame({
            "session_id": ["session_canon2"] * n,
            "trial_id": [0] * n,
            "time_ms": np.arange(n, dtype=np.float64) * 4.0,
            "x_pos": np.zeros(n),
            "y_pos": np.zeros(n),
            "heading": np.zeros(n),
            "velocity": np.zeros(n),
            "acceleration": np.zeros(n),
            "visual_angle": np.zeros(n),
            "wind_state": np.zeros(n, dtype=int),
            "l_v_ratio": np.zeros(n),
        }).to_csv(session_dir / "session_canon2_kinematics.csv", index=False)
        pd.DataFrame([
            {
                "session_id": "session_canon2",
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
                "session_id": "session_canon2",
                "trial_id": 0,
                "time_ms": 50.0,
                "event_type": "phase_transition",
                "event_value": json.dumps({
                    "from_phase": "Baseline",
                    "to_phase": "Looming",
                }),
            },
        ]).to_csv(session_dir / "session_canon2_events.csv", index=False)

        adapt_cercus_to_nsmor(raw_dir=str(tmp_path))
        evt_out = pd.read_csv(session_dir / "session_canon2_events.csv")
        onset = _get_stimulus_onset(evt_out, trial_id=0)
        assert onset == pytest.approx(50.0, abs=1e-6)
        assert onset < 1000.0, f"Canonical ms double-scaled: {onset}"


# ═══════════════════════════════════════════════════════════════
# D. Alignment: visual / wind / multisensory
# ═══════════════════════════════════════════════════════════════

class TestBranchAlignment:
    """All injection branches produce correct stimulus_onset times."""

    def test_wind_onset_from_kinematics(self, tmp_path: Path):
        """Wind branch: stimulus_onset = first wind_state=1 kinematics time_ms."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_wind"
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"
        raw_dir.mkdir(parents=True)

        n = 30
        sys_time = 5.0 + np.arange(n, dtype=np.float64) * 0.004
        stim = np.zeros(n, dtype=np.int64)
        stim[10:] = 1  # wind starts at sample 10 → t = 5.0 + 10*0.004 = 5.040s
        pd.DataFrame({
            "sys_time": sys_time,
            "ard_time": (sys_time * 1000).astype(np.int64),
            "dx": np.full(n, 0.1),
            "dy": np.full(n, 0.05),
            "dz": np.zeros(n),
            "stim_state": stim,
            "global_trial_id": np.full(n, 1, dtype=np.int64),
        }).to_csv(kin_path, index=False)

        details = json.dumps({
            "type": "baseline_wind",
            "target_ttc_ms": None,
            "lv_ratio_ms": None,
            "wind_dir": "left",
            "screen_side": "none",
        })
        pd.DataFrame([
            {
                "event_name": "trial_start",
                "timestamp": 5.0,
                "session_num": 1,
                "trial_in_session": 1,
                "global_trial_id": 1,
                "details": details,
            },
        ]).to_csv(evt_path, index=False)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        onset = _get_stimulus_onset(evt_out, trial_id=1)
        # First wind sample is at kin time_ms = (5.040-5.000)*1000 = 40.0
        assert onset == pytest.approx(40.0, abs=1e-6), (
            f"Wind onset expected 40ms, got {onset}"
        )

    def test_multisensory_uses_wind_branch(self, tmp_path: Path):
        """looming_wind (multisensory) uses wind kinematics timing."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_multi_stim"
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"
        raw_dir.mkdir(parents=True)

        n = 30
        sys_time = 0.100 + np.arange(n, dtype=np.float64) * 0.004
        stim = np.zeros(n, dtype=np.int64)
        stim[5:] = 1
        pd.DataFrame({
            "sys_time": sys_time,
            "ard_time": (sys_time * 1000).astype(np.int64),
            "dx": np.full(n, 0.1),
            "dy": np.full(n, 0.05),
            "dz": np.zeros(n),
            "stim_state": stim,
            "global_trial_id": np.full(n, 1, dtype=np.int64),
        }).to_csv(kin_path, index=False)

        details = json.dumps({
            "type": "looming_wind",
            "target_ttc_ms": 200.0,
            "lv_ratio_ms": 120.0,
            "wind_dir": "left",
            "screen_side": "right",
        })
        pd.DataFrame([
            {
                "event_name": "trial_start",
                "timestamp": 0.100,
                "session_num": 1,
                "trial_in_session": 1,
                "global_trial_id": 1,
                "details": details,
            },
            {
                "event_name": "phase_transition",
                "timestamp": 0.120,
                "session_num": 1,
                "trial_in_session": 1,
                "global_trial_id": 1,
                "details": json.dumps({
                    "from_phase": "TrialStart",
                    "to_phase": "Looming",
                }),
            },
        ]).to_csv(evt_path, index=False)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        onset = _get_stimulus_onset(evt_out, trial_id=1)
        # Wind branch: first wind sample at kin sample 5 → time_ms = 5*4 = 20ms
        assert onset == pytest.approx(20.0, abs=1e-6), (
            f"Multisensory onset expected 20ms (wind branch), got {onset}"
        )


# ═══════════════════════════════════════════════════════════════
# E. Preservation: details / null TTC / global IDs
# ═══════════════════════════════════════════════════════════════

class TestPreservation:
    """Event details, null TTC, and global trial IDs survive adaptation."""

    def test_details_null_ttc_ids_preserved(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_preserve"
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100, looming_s=0.120, kin_first_s=0.100,
            trial_id=77,
        )
        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        # Global trial ID preserved
        assert set(evt_out["trial_id"].unique()) == {77}

        # trial_start details preserved including null target_ttc_ms
        ts_rows = evt_out[evt_out["event_type"] == "trial_start"]
        assert len(ts_rows) == 1
        d = json.loads(str(ts_rows.iloc[0]["event_value"]))
        assert d["target_ttc_ms"] is None
        assert d["lv_ratio_ms"] == 120.0
        assert d["type"] == "baseline_visual"

    def test_no_duplicate_stimulus_onset(self, tmp_path: Path):
        """Exactly one stimulus_onset per visual trial after adapt."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_nodup"
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100, looming_s=0.120, kin_first_s=0.100,
        )
        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        stim = evt_out[evt_out["event_type"] == "stimulus_onset"]
        assert len(stim) == 1, f"Expected 1 stimulus_onset, got {len(stim)}"


class TestV6PerTrialStimulusOnsetPreservation:
    """Fix 5 (v6): source stimulus_onset survives for trials without computed onset.

    Regression: the old wholesale replacement dropped ALL source
    ``stimulus_onset`` rows whenever ANY trial produced a computed onset.
    A trial with missing Looming (no computed onset) must keep its source
    event; only trials with a computed onset get their source row replaced.
    """

    def test_missing_looming_trial_keeps_source_onset(self, tmp_path: Path):
        """Trial 2 (no Looming) keeps source stimulus_onset when trial 1 computes one."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_per_trial_onset"
        raw_dir.mkdir(parents=True)
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"

        # Two trials: trial 1 has Looming, trial 2 does not.
        rows = []
        for i in range(30):
            rows.append(_raw_kin_row(i, 0.100 + i * 0.004, 1))
        for i in range(30):
            rows.append(_raw_kin_row(i, 0.300 + i * 0.004, 2))
        _write_raw_kin_rows(kin_path, s1, rows)

        details1 = json.dumps({
            "type": "baseline_visual", "target_ttc_ms": None,
            "lv_ratio_ms": 120.0, "wind_dir": "none", "screen_side": "right",
        })
        details2 = json.dumps({
            "type": "baseline_visual", "target_ttc_ms": None,
            "lv_ratio_ms": 120.0, "wind_dir": "none", "screen_side": "right",
        })
        evt_rows = [
            _raw_evt_row("trial_start", 0.100, 1, details=details1),
            _raw_evt_row("trial_start", 0.300, 2, details=details2),
            # Trial 1: Looming transition → computed onset.
            _raw_evt_row(
                "phase_transition", 0.120, 1,
                details=json.dumps({"from_phase": "TrialStart", "to_phase": "Looming"}),
            ),
            # Trial 2: NO Looming, but has a source stimulus_onset event.
            _raw_evt_row(
                "stimulus_onset", 0.310, 2,
                details='{"source": "manual_annotation"}',
            ),
        ]
        pd.DataFrame(evt_rows).to_csv(evt_path, index=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        stim = evt_out[evt_out["event_type"] == "stimulus_onset"]

        # Trial 1: exactly one computed stimulus_onset.
        t1_stim = stim[stim["trial_id"] == 1]
        assert len(t1_stim) == 1, (
            f"Trial 1 must have exactly 1 stimulus_onset, got {len(t1_stim)}"
        )
        # Computed onset: Looming at 0.120s, kin first at 0.100s → 20ms.
        assert float(t1_stim.iloc[0]["time_ms"]) == pytest.approx(20.0, abs=1e-6)

        # Trial 2: source stimulus_onset PRESERVED (Fix 5 regression guard).
        t2_stim = stim[stim["trial_id"] == 2]
        assert len(t2_stim) == 1, (
            f"Trial 2 (missing Looming) must keep its source stimulus_onset, "
            f"got {len(t2_stim)} rows:\n{t2_stim}"
        )
        # Source onset: 0.310s, kin first for trial 2 at 0.300s → 10ms.
        assert float(t2_stim.iloc[0]["time_ms"]) == pytest.approx(10.0, abs=1e-6), (
            f"Trial 2 source onset must be kin-relative 10ms, "
            f"got {float(t2_stim.iloc[0]['time_ms'])}"
        )
        # Missing-Looming warning must still fire for trial 2.
        assert any("Looming" in str(w.message) for w in caught), (
            "missing-Looming warning must fire for trial 2"
        )
        # Total: exactly 2 stimulus_onset rows (one per trial).
        assert len(stim) == 2, f"Expected 2 total stimulus_onset, got {len(stim)}"


# ═══════════════════════════════════════════════════════════════
# F. Source immutability
# ═══════════════════════════════════════════════════════════════

class TestSourceImmutability:
    """Source bytes unchanged under explicit staging output."""

    def test_source_bytes_unchanged(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_immutable"
        kin_path, evt_path = _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100, looming_s=0.120, kin_first_s=0.100,
        )
        h_kin = _sha256(kin_path)
        h_evt = _sha256(evt_path)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        assert _sha256(kin_path) == h_kin, "Kinematics source mutated"
        assert _sha256(evt_path) == h_evt, "Events source mutated"


# ═══════════════════════════════════════════════════════════════
# G. Downstream public-path: extract_trial_data + resolve_snapshot_anchor
# ═══════════════════════════════════════════════════════════════

class TestDownstreamAnchorAlignment:
    """Canonical event time feeds extract_trial_data and anchor resolution."""

    def test_extract_trial_data_visual_event_time(self, tmp_path: Path):
        """extract_trial_data exposes stimulus_onset at 20ms (visual trial)."""
        from nsmor.pipeline.io import extract_trial_data
        from nsmor.pipeline.labeling import find_event_time

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_anchor"
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100, looming_s=0.120, kin_first_s=0.100,
            trial_type="baseline_visual",
        )
        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        kin_df = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")
        evt_df = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        trial = extract_trial_data(
            {"kinematics": kin_df, "events": evt_df}, s1, 1,
        )
        assert trial["time_ms"][0] == pytest.approx(0.0, abs=1e-6)

        onset = find_event_time(
            trial["event_types"], trial["event_times"], "stimulus_onset",
        )
        assert onset is not None
        assert onset == pytest.approx(20.0, abs=1e-6)

    def test_visual_anchor_rule_looming_collision(self, tmp_path: Path):
        """Visual-only trials anchor at looming_collision, not stimulus_onset."""
        from nsmor.pipeline.io import extract_trial_data
        from nsmor.data_extractor import resolve_snapshot_anchor
        from nsmor.pipeline.labeling import find_event_time

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_vis_anchor"
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100, looming_s=0.120, kin_first_s=0.100,
            trial_type="baseline_visual", n_samples=1800,
        )
        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        kin_df = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")
        evt_df = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        trial = extract_trial_data(
            {"kinematics": kin_df, "events": evt_df}, s1, 1,
        )
        onset = find_event_time(
            trial["event_types"], trial["event_times"], "stimulus_onset",
        )
        anchor_ms, rule = resolve_snapshot_anchor(trial, onset)
        assert rule == "looming_collision"
        # The trial clock has enough frames to contain the physical collision.
        assert anchor_ms == pytest.approx(20.0 + 120.0 / math.tan(math.radians(1.0)), abs=4.0)
        assert 0.0 <= anchor_ms <= trial["time_ms"][-1]

    def test_wind_anchor_uses_stimulus_onset(self, tmp_path: Path):
        """Wind trials anchor at stimulus_onset time from kinematics."""
        from nsmor.pipeline.io import extract_trial_data
        from nsmor.data_extractor import resolve_snapshot_anchor
        from nsmor.pipeline.labeling import find_event_time

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_wind_anchor"
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"
        raw_dir.mkdir(parents=True)

        n = 30
        sys_time = 5.0 + np.arange(n, dtype=np.float64) * 0.004
        stim = np.zeros(n, dtype=np.int64)
        stim[10:] = 1
        pd.DataFrame({
            "sys_time": sys_time,
            "ard_time": (sys_time * 1000).astype(np.int64),
            "dx": np.full(n, 0.1),
            "dy": np.full(n, 0.05),
            "dz": np.zeros(n),
            "stim_state": stim,
            "global_trial_id": np.full(n, 1, dtype=np.int64),
        }).to_csv(kin_path, index=False)
        details = json.dumps({
            "type": "baseline_wind",
            "target_ttc_ms": None,
            "lv_ratio_ms": None,
            "wind_dir": "left",
            "screen_side": "none",
        })
        pd.DataFrame([{
            "event_name": "trial_start",
            "timestamp": 5.0,
            "session_num": 1,
            "trial_in_session": 1,
            "global_trial_id": 1,
            "details": details,
        }]).to_csv(evt_path, index=False)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        kin_df = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")
        evt_df = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        trial = extract_trial_data(
            {"kinematics": kin_df, "events": evt_df}, s1, 1,
        )
        onset = find_event_time(
            trial["event_types"], trial["event_times"], "stimulus_onset",
        )
        assert onset == pytest.approx(40.0, abs=1e-6)
        anchor_ms, rule = resolve_snapshot_anchor(trial, onset)
        assert rule == "stimulus_onset"
        assert anchor_ms == pytest.approx(40.0, abs=1e-6)

    def test_offset_invariance_downstream_anchor(self, tmp_path: Path):
        """Offset invariance holds through extract_trial_data + anchor."""
        from nsmor.pipeline.io import extract_trial_data
        from nsmor.data_extractor import resolve_snapshot_anchor
        from nsmor.pipeline.labeling import find_event_time

        results = []
        for tag, offset in [("base", 0.0), ("offset", 500.0)]:
            raw_dir = tmp_path / f"raw_{tag}"
            out_dir = tmp_path / f"out_{tag}"
            s1 = f"session_anchor_{tag}"
            _write_raw_pair(
                raw_dir, s1,
                trial_start_s=offset + 0.100,
                looming_s=offset + 0.120,
                kin_first_s=offset + 0.100,
                trial_type="baseline_visual", n_samples=1800,
            )
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

            kin_df = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")
            evt_df = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
            trial = extract_trial_data(
                {"kinematics": kin_df, "events": evt_df}, s1, 1,
            )
            onset = find_event_time(
                trial["event_types"], trial["event_times"], "stimulus_onset",
            )
            anchor_ms, rule = resolve_snapshot_anchor(trial, onset)
            results.append((onset, anchor_ms, rule))

        (onset_a, anchor_a, rule_a) = results[0]
        (onset_b, anchor_b, rule_b) = results[1]
        assert onset_a == pytest.approx(onset_b, abs=1e-6), (
            f"Onset not offset-invariant: {onset_a} vs {onset_b}"
        )
        assert anchor_a == pytest.approx(anchor_b, abs=1e-6), (
            f"Anchor not offset-invariant: {anchor_a} vs {anchor_b}"
        )
        assert rule_a == rule_b


# ═══════════════════════════════════════════════════════════════
# H. Kin-clock alignment: trial_start precedes first kinematics sample
# ═══════════════════════════════════════════════════════════════

class TestKinClockAlignment:
    """Event/stimulus times must share the kinematics trial zero.

    Reproduces trial_start=0.100s, first kinematics sample=0.104s,
    Looming transition=0.120s.  Expected canonical onset = 16 ms on the
    kinematics clock (0.120 − 0.104), not 20 ms on the event-file clock
    (0.120 − 0.100).
    """

    def test_stimulus_onset_kin_clock_16ms(self, tmp_path: Path):
        """stimulus_onset and Looming transition land at 16 ms (kin clock)."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_kin_clock"
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100,
            looming_s=0.120,
            kin_first_s=0.104,
        )
        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        onset = _get_stimulus_onset(evt_out, trial_id=1)
        assert onset == pytest.approx(16.0, abs=1e-6), (
            f"Expected stimulus_onset at 16ms on kin clock, got {onset}"
        )
        looming_t = _get_phase_time(evt_out, trial_id=1, to_phase="Looming")
        assert looming_t == pytest.approx(16.0, abs=1e-6), (
            f"Looming transition expected 16ms on kin clock, got {looming_t}"
        )

    def test_visual_pre_onset_baseline_and_true_onset(self, tmp_path: Path):
        """Pre-onset frames hold init_deg baseline; curve expands only after
        true Looming onset (16 ms on the kin clock)."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_vis_baseline"
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100,
            looming_s=0.120,
            kin_first_s=0.104,
            trial_type="baseline_visual",
            lv_ratio_ms=40.0,
        )
        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        kin_out = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")

        # Pre-onset samples (t < 16 ms) must hold the 2.0 deg baseline.
        pre = kin_out[kin_out["time_ms"] < 16.0 - 1e-6]
        assert len(pre) > 0, "Need at least one pre-onset sample"
        assert np.allclose(pre["visual_angle"].values, 2.0, atol=1e-9), (
            f"Pre-onset visual_angle must be baseline 2.0, got {pre['visual_angle'].values}"
        )

        # Sample at or nearest true onset (t = 16 ms): theta(0) = init_deg.
        onset_row = kin_out.loc[(kin_out["time_ms"] - 16.0).abs().idxmin()]
        assert onset_row["visual_angle"] == pytest.approx(2.0, abs=0.05), (
            f"Expected ~2.0deg at true onset, got {onset_row['visual_angle']}"
        )

        # At least one post-onset sample must show expansion.
        post = kin_out[kin_out["time_ms"] > 16.0 + 1e-6]
        assert len(post) > 0, "Need at least one post-onset sample"
        assert post["visual_angle"].max() > 2.0 + 1e-6, (
            "Visual angle must expand after true Looming onset"
        )

    def test_public_path_extract_trial_16ms(self, tmp_path: Path):
        """Downstream extract_trial_data sees stimulus_onset at 16 ms."""
        from nsmor.pipeline.io import extract_trial_data
        from nsmor.pipeline.labeling import find_event_time

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_pub_kin_clock"
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100,
            looming_s=0.120,
            kin_first_s=0.104,
            trial_type="baseline_visual",
        )
        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        kin_df = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")
        evt_df = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        trial = extract_trial_data(
            {"kinematics": kin_df, "events": evt_df}, s1, 1,
        )
        assert trial["time_ms"][0] == pytest.approx(0.0, abs=1e-6)
        onset = find_event_time(
            trial["event_types"], trial["event_times"], "stimulus_onset",
        )
        assert onset is not None
        assert onset == pytest.approx(16.0, abs=1e-6), (
            f"extract_trial_data onset expected 16ms, got {onset}"
        )


# ═══════════════════════════════════════════════════════════════
# V4 adversarial regressions (both independent v4 reviewers REJECT)
# ═══════════════════════════════════════════════════════════════

def _hash_tree(root: Path) -> Dict[str, str]:
    """Map every file under *root* to its sha256 (staging-safety oracle)."""
    out: Dict[str, str] = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = _sha256(p)
    return out


def _write_raw_kin_rows(
    kin_path: Path,
    session_id: str,
    rows: List[Dict],
) -> None:
    """Write raw-schema kinematics from explicit per-row dicts."""
    df = pd.DataFrame(rows)
    df.insert(0, "session_id", session_id)
    kin_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(kin_path, index=False)


def _raw_kin_row(
    i: int,
    sys_time: float,
    trial_id: int,
    *,
    dx: float = 0.1,
    dy: float = 0.05,
    dz: float = 0.0,
) -> Dict:
    return {
        "sys_time": sys_time,
        "ard_time": int(round(sys_time * 1000)),
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "stim_state": 0,
        "global_trial_id": trial_id,
    }


def _raw_evt_row(
    name: str,
    ts: float,
    trial_id: int,
    *,
    session_num: int = 1,
    details: str | None = None,
) -> Dict:
    return {
        "event_name": name,
        "timestamp": ts,
        "session_num": session_num,
        "trial_in_session": trial_id,
        "global_trial_id": trial_id,
        "details": details if details is not None else "",
    }


class TestV4ChronologicalOrigin:
    """A1 / B F1: origin must be chronological, never file-order first."""

    def test_unsorted_kinematics_fails_closed(self, tmp_path: Path):
        """Unsorted per-trial abs_time must raise, not silently skew origin."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_unsorted"
        raw_dir.mkdir(parents=True)
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"

        # File order: 0.104, 0.100, 0.108  (NOT chronological)
        rows = [
            _raw_kin_row(0, 0.104, 1),
            _raw_kin_row(1, 0.100, 1),
            _raw_kin_row(2, 0.108, 1),
        ]
        _write_raw_kin_rows(kin_path, s1, rows)

        details = json.dumps({
            "type": "baseline_visual", "target_ttc_ms": None,
            "lv_ratio_ms": 120.0, "wind_dir": "none", "screen_side": "right",
        })
        evt_rows = [
            _raw_evt_row("trial_start", 0.100, 1, details=details),
            _raw_evt_row("phase_transition", 0.120, 1,
                         details=json.dumps({"from_phase": "Baseline",
                                             "to_phase": "Looming"})),
        ]
        pd.DataFrame(evt_rows).to_csv(evt_path, index=False)

        before = _hash_tree(raw_dir)
        with pytest.raises(ValueError, match="chronologically ordered"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        assert not out_dir.exists(), "no output may be published on failure"
        assert _hash_tree(raw_dir) == before, "source must be unchanged on failure"

    def test_sorted_rows_origin_is_chronological_min(self, tmp_path: Path):
        """Sorted rows: origin == min(abs_time), velocity is finite & signed."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_sorted"
        raw_dir.mkdir(parents=True)
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"

        rows = [_raw_kin_row(i, 0.100 + i * 0.004, 1, dx=0.1) for i in range(30)]
        _write_raw_kin_rows(kin_path, s1, rows)
        details = json.dumps({
            "type": "baseline_visual", "target_ttc_ms": None,
            "lv_ratio_ms": 120.0, "wind_dir": "none", "screen_side": "right",
        })
        evt_rows = [
            _raw_evt_row("trial_start", 0.100, 1, details=details),
            _raw_evt_row("phase_transition", 0.120, 1,
                         details=json.dumps({"from_phase": "Baseline",
                                             "to_phase": "Looming"})),
        ]
        pd.DataFrame(evt_rows).to_csv(evt_path, index=False)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        kin_df = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")
        assert kin_df["time_ms"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert np.all(np.diff(kin_df["time_ms"].values) > 0)
        # Derivatives must be finite and physically signed (forward motion).
        vel = kin_df["velocity"].values
        acc = kin_df["acceleration"].values
        assert np.all(np.isfinite(vel)), f"non-finite velocity: {vel}"
        assert np.all(np.isfinite(acc)), f"non-finite acceleration: {acc}"
        assert np.all(vel[1:] > 0), f"forward dx must yield positive v: {vel}"


class TestV4OrphanEvents:
    """A2: events without kin trials must fail closed, never mix clocks."""

    def test_orphan_event_trial_fails_closed(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_orphan"
        raw_dir.mkdir(parents=True)
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"

        # Trial 1 has kinematics; trial 99 exists only in events.
        rows = [_raw_kin_row(i, 0.100 + i * 0.004, 1) for i in range(20)]
        _write_raw_kin_rows(kin_path, s1, rows)

        details = json.dumps({
            "type": "baseline_visual", "target_ttc_ms": None,
            "lv_ratio_ms": 120.0, "wind_dir": "none", "screen_side": "right",
        })
        evt_rows = [
            _raw_evt_row("trial_start", 0.100, 1, details=details),
            _raw_evt_row("trial_start", 0.500, 99, details=details),
        ]
        pd.DataFrame(evt_rows).to_csv(evt_path, index=False)

        before = _hash_tree(raw_dir)
        with pytest.raises(ValueError, match="99"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        assert not out_dir.exists(), "no output may be published on failure"
        assert _hash_tree(raw_dir) == before, "source must be unchanged on failure"


class TestV4MissingLoomingAndOnsetSign:
    """A4 / B F2 / B F3: no fabricated stimulus; negative onset stays negative."""

    def test_missing_looming_raw_no_fabrication(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_no_loom"
        raw_dir.mkdir(parents=True)
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"

        rows = [_raw_kin_row(i, 0.100 + i * 0.004, 1) for i in range(30)]
        _write_raw_kin_rows(kin_path, s1, rows)
        details = json.dumps({
            "type": "baseline_visual", "target_ttc_ms": None,
            "lv_ratio_ms": 120.0, "wind_dir": "none", "screen_side": "right",
        })
        evt_rows = [_raw_evt_row("trial_start", 0.100, 1, details=details)]
        pd.DataFrame(evt_rows).to_csv(evt_path, index=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        evt_df = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        stim = evt_df[evt_df["event_type"] == "stimulus_onset"]
        assert len(stim) == 0, f"no fabricated stimulus_onset, got:\n{stim}"
        kin_df = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")
        assert np.allclose(kin_df["visual_angle"].values, 0.0), (
            "missing Looming must not expand theta from t=0"
        )
        assert any("Looming" in str(w.message) for w in caught)

    def test_negative_onset_stays_negative(self, tmp_path: Path):
        """Looming before kin first sample -> kin-relative onset < 0 (B F3)."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_neg_onset"
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100,
            looming_s=0.100,
            kin_first_s=0.110,
            trial_type="baseline_visual",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        evt_df = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        stim = evt_df[evt_df["event_type"] == "stimulus_onset"]
        assert len(stim) == 1
        onset = float(stim.iloc[0]["time_ms"])
        assert onset == pytest.approx(-10.0, abs=1e-6), (
            f"valid negative onset must stay negative, got {onset}"
        )
        # And the warning channel must not claim a deviation anomaly.
        assert not any("deviates" in str(w.message) for w in caught)


class TestV4ChronologicalFirstLooming:
    """B F4: out-of-order event rows must select the chronological first."""

    def test_ooo_looming_rows_first_wins(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_ooo_loom"
        raw_dir.mkdir(parents=True)
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"

        rows = [_raw_kin_row(i, 0.100 + i * 0.004, 1) for i in range(30)]
        _write_raw_kin_rows(kin_path, s1, rows)
        details = json.dumps({
            "type": "baseline_visual", "target_ttc_ms": None,
            "lv_ratio_ms": 120.0, "wind_dir": "none", "screen_side": "right",
        })
        # Looming at 0.120 s is written AFTER Looming at 0.116 s.
        evt_rows = [
            _raw_evt_row("trial_start", 0.100, 1, details=details),
            _raw_evt_row("phase_transition", 0.120, 1,
                         details=json.dumps({"from_phase": "Baseline",
                                             "to_phase": "Looming"})),
            _raw_evt_row("phase_transition", 0.116, 1,
                         details=json.dumps({"from_phase": "Baseline",
                                             "to_phase": "Looming"})),
        ]
        pd.DataFrame(evt_rows).to_csv(evt_path, index=False)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        evt_df = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        stim = evt_df[evt_df["event_type"] == "stimulus_onset"]
        assert len(stim) == 1
        # kin_first=0.100, chronological-first Looming=0.116 -> 16.0 ms
        assert float(stim.iloc[0]["time_ms"]) == pytest.approx(16.0, abs=1e-6), (
            f"expected chronological-first Looming onset 16 ms, "
            f"got {float(stim.iloc[0]['time_ms'])}"
        )


class TestV4VisualAnglePhysics:
    """A3 / B F5: post-TTC hold; flat baseline is never a collision anchor."""

    def test_post_ttc_holds_max_not_zero(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_post_ttc"
        # lv_ratio_ms=1.0 -> ttc = lv_s / tan(1 deg) ~= 57.3 ms.
        # 40 samples x 4 ms = 160 ms window covers a long post-TTC stretch.
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100,
            looming_s=0.100,
            kin_first_s=0.100,
            n_samples=40,
            trial_type="baseline_visual",
            lv_ratio_ms=1.0,
        )
        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        kin_df = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")
        va = kin_df["visual_angle"].values
        # Must expand somewhere...
        assert va.max() > 2.0 + 1e-6, f"expected expansion, got max={va.max()}"
        # ...and after the peak, values must stay at the hold level, never 0.
        peak_idx = int(np.argmax(va))
        post = va[peak_idx:]
        assert np.all(post > 1.0), (
            f"post-TTC visual_angle collapsed: {post}"
        )
        assert np.all(post >= post[-1] - 1e-6) or np.allclose(post, 179.0), (
            f"post-TTC must hold/clamp, got {post}"
        )

    def test_flat_pre_onset_baseline_not_collision_anchor(self, tmp_path: Path):
        """Onset after the recorded window -> publish zeros, not flat init_deg.

        Frozen data_extractor.resolve_snapshot_anchor treats any nonzero
        visual_angle as 'has_looming' and anchors at argmax(v) == 0, which
        is a false collision for a flat baseline.  The adapter must never
        publish that flat nonzero baseline (B F5; no data_extractor edit).
        """
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_flat_base"
        raw_dir.mkdir(parents=True)
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"

        # 30 samples covering 0.100 .. 0.216 s; Looming at 0.500 s (beyond).
        rows = [_raw_kin_row(i, 0.100 + i * 0.004, 1) for i in range(30)]
        _write_raw_kin_rows(kin_path, s1, rows)
        details = json.dumps({
            "type": "baseline_visual", "target_ttc_ms": None,
            "lv_ratio_ms": 120.0, "wind_dir": "none", "screen_side": "right",
        })
        evt_rows = [
            _raw_evt_row("trial_start", 0.100, 1, details=details),
            _raw_evt_row("phase_transition", 0.500, 1,
                         details=json.dumps({"from_phase": "Baseline",
                                             "to_phase": "Looming"})),
        ]
        pd.DataFrame(evt_rows).to_csv(evt_path, index=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        kin_df = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")
        va = kin_df["visual_angle"].values
        assert np.allclose(va, 0.0), (
            f"flat pre-onset baseline must not be published as init_deg, got {va}"
        )
        assert any("never expands" in str(w.message) for w in caught)

        # Downstream oracle: no false looming_collision anchor.
        from nsmor.pipeline.io import extract_trial_data
        from nsmor.data_extractor import resolve_snapshot_anchor
        from nsmor.pipeline.labeling import find_event_time

        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        trial = extract_trial_data(
            {"kinematics": kin_df, "events": evt_out}, s1, 1,
        )
        onset = find_event_time(
            trial["event_types"], trial["event_times"], "stimulus_onset",
        )
        anchor_ms, rule = resolve_snapshot_anchor(trial, onset)
        assert rule != "looming_collision", (
            f"flat baseline must not be a collision anchor (rule={rule})"
        )


class TestV4AlternativeEventSchemas:
    """B F6: event_type+timestamp and event_name+time_ms must not corrupt."""

    def test_event_type_plus_timestamp(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_et_ts"
        raw_dir.mkdir(parents=True)
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"

        rows = [_raw_kin_row(i, 0.100 + i * 0.004, 1) for i in range(30)]
        _write_raw_kin_rows(kin_path, s1, rows)
        details = json.dumps({
            "type": "baseline_visual", "target_ttc_ms": None,
            "lv_ratio_ms": 120.0, "wind_dir": "none", "screen_side": "right",
        })
        # Alternative schema: event_type + timestamp (NO time_ms, NO event_name)
        evt = pd.DataFrame([
            {"event_type": "trial_start", "timestamp": 0.100,
             "trial_id": 1, "event_value": details},
            {"event_type": "phase_transition", "timestamp": 0.120,
             "trial_id": 1,
             "event_value": json.dumps({"from_phase": "Baseline",
                                        "to_phase": "Looming"})},
        ])
        evt.to_csv(evt_path, index=False)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        assert not evt_out["time_ms"].isna().any(), (
            f"time_ms NaN corruption:\n{evt_out}"
        )
        # trial_start shares kinematics zero; Looming at 0.120 -> 20 ms
        ts0 = evt_out[evt_out["event_type"] == "trial_start"]["time_ms"].iloc[0]
        assert float(ts0) == pytest.approx(0.0, abs=1e-6)
        stim = evt_out[evt_out["event_type"] == "stimulus_onset"]
        assert len(stim) == 1
        assert float(stim.iloc[0]["time_ms"]) == pytest.approx(20.0, abs=1e-6)

    def test_event_name_plus_time_ms(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_en_ms"
        raw_dir.mkdir(parents=True)
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"

        rows = [_raw_kin_row(i, 0.100 + i * 0.004, 1) for i in range(30)]
        _write_raw_kin_rows(kin_path, s1, rows)
        details = json.dumps({
            "type": "baseline_visual", "target_ttc_ms": None,
            "lv_ratio_ms": 120.0, "wind_dir": "none", "screen_side": "right",
        })
        # Alternative schema: event_name + time_ms (NO event_type, NO timestamp)
        evt = pd.DataFrame([
            {"event_name": "trial_start", "time_ms": 0.0,
             "global_trial_id": 1, "details": details},
            {"event_name": "phase_transition", "time_ms": 20.0,
             "global_trial_id": 1,
             "details": json.dumps({"from_phase": "Baseline",
                                    "to_phase": "Looming"})},
        ])
        evt.to_csv(evt_path, index=False)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        assert not evt_out["time_ms"].isna().any(), (
            f"time_ms NaN corruption:\n{evt_out}"
        )
        assert (evt_out["event_type"] == "trial_start").any()
        stim = evt_out[evt_out["event_type"] == "stimulus_onset"]
        assert len(stim) == 1
        assert float(stim.iloc[0]["time_ms"]) == pytest.approx(20.0, abs=1e-6)


class TestV4NoSpuriousWarnings:
    """A5: normal delayed looming must not generate warnings."""

    def test_normal_delayed_looming_silent(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_ok_delay"
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100,
            looming_s=0.120,
            kin_first_s=0.100,
            trial_type="baseline_visual",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        noisy = [w for w in caught if "deviates" in str(w.message)]
        assert not noisy, f"spurious warning on normal delayed looming: {noisy}"


# ═══════════════════════════════════════════════════════════════
# V5 adversarial regressions (both independent v5 reviewers REJECT)
# ═══════════════════════════════════════════════════════════════


class TestV5AllPostTtcFlatMax:
    """NEW-1 / V5B-F1: all-post-TTC flat 179° must not create a false anchor.

    When every sample lands post-TTC the reconstructed curve is a flat 179°
    plateau.  The old guard (``np.any(va > 2.0)``) passed it; downstream
    ``argmax`` of a constant array == index 0 is a false looming_collision
    anchor at trial start.  The guard must test for actual expansion
    (peak-to-peak), not mere exceedance of init_deg.
    """

    def test_all_post_ttc_flat_max_publishes_zeros(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_all_post_ttc"
        # Looming at 0.010 s; kinematics start at 0.100 s → onset_kin = -90 ms.
        # lv=1.0 ms → TTC ≈ 57.3 ms.  All t_rel = [90, 94, …] > TTC → flat 179.
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100,
            looming_s=0.010,
            kin_first_s=0.100,
            n_samples=30,
            trial_type="baseline_visual",
            lv_ratio_ms=1.0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        kin_df = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")
        va = kin_df["visual_angle"].values
        assert np.allclose(va, 0.0), (
            f"all-post-TTC flat 179° must be zeroed, got unique="
            f"{np.unique(va)}"
        )
        assert any("never expands" in str(w.message) for w in caught)

        # Downstream oracle: no false looming_collision anchor.
        from nsmor.pipeline.io import extract_trial_data
        from nsmor.data_extractor import resolve_snapshot_anchor
        from nsmor.pipeline.labeling import find_event_time

        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        trial = extract_trial_data(
            {"kinematics": kin_df, "events": evt_out}, s1, 1,
        )
        onset = find_event_time(
            trial["event_types"], trial["event_times"], "stimulus_onset",
        )
        anchor_ms, rule = resolve_snapshot_anchor(trial, onset)
        assert rule != "looming_collision", (
            f"all-post-TTC flat max must not be a collision anchor (rule={rule})"
        )

    def test_ptp_guard_rejects_any_flat_curve(self):
        """Unit: the guard predicate rejects flat curves at ANY baseline."""
        from scripts.pre_load_adapt import reconstruct_visual_angle

        # All-pre-onset: flat init_deg.
        va_pre = reconstruct_visual_angle(
            np.array([-10.0, -6.0, -2.0]), lv_ratio_ms=120.0,
        )
        assert float(np.ptp(va_pre)) < 1e-9

        # All-post-TTC: flat 179°.
        ttc_ms = (120.0 / 1000.0) / math.tan(math.radians(1.0)) * 1000.0
        va_post = reconstruct_visual_angle(
            np.array([ttc_ms + 10, ttc_ms + 20, ttc_ms + 30]), lv_ratio_ms=120.0,
        )
        assert float(np.ptp(va_post)) < 1e-9
        assert np.allclose(va_post, 179.0), f"expected flat 179°, got {va_post}"


class TestV5TtcBoundaryContinuity:
    """NEW-2 / V5B-F2: no step discontinuity at TTC for any lv.

    The 0.001 s delta floor capped pre-TTC theta at 2*arctan(lv_s/0.001),
    then the post-TTC hold slammed to 179° — up to an 89° single-frame step
    for small lv.  With the floor removed, the pre-TTC curve converges to
    180° (clipped 179°) and the hold is continuous.
    """

    @pytest.mark.parametrize("lv", [1.0, 2.0, 5.0, 20.0, 120.0])
    def test_dense_grid_boundary_jump_small(self, lv: float):
        """Dense 0.01-ms grid straddling TTC: jump must be < 5°."""
        lv_s = lv / 1000.0
        t_col_s = lv_s / math.tan(math.radians(1.0))
        ttc_ms = t_col_s * 1000.0

        # Dense grid from just after onset to just past TTC.
        dt = 0.01  # ms
        time_ms = np.arange(0.0, ttc_ms + 5.0, dt)
        va = reconstruct_visual_angle(time_ms, lv_ratio_ms=lv, init_deg=2.0)

        pre_mask = time_ms < ttc_ms
        post_mask = time_ms >= ttc_ms
        assert pre_mask.any(), "need pre-TTC samples"
        assert post_mask.any(), "need post-TTC samples"

        pre_max = float(va[pre_mask].max())
        post_min = float(va[post_mask].min())
        jump = post_min - pre_max
        assert abs(jump) < 5.0, (
            f"lv={lv}: {jump:.2f}° step at TTC (pre_max={pre_max:.2f}, "
            f"post_min={post_min:.2f})"
        )

    @pytest.mark.parametrize("lv", [1.0, 2.0, 5.0, 20.0, 120.0])
    def test_post_ttc_hold_nonzero(self, lv: float):
        """Post-TTC hold must be nonzero (never collapse to 0)."""
        lv_s = lv / 1000.0
        t_col_s = lv_s / math.tan(math.radians(1.0))
        ttc_ms = t_col_s * 1000.0
        time_ms = np.array([ttc_ms, ttc_ms + 10.0, ttc_ms + 100.0])
        va = reconstruct_visual_angle(time_ms, lv_ratio_ms=lv, init_deg=2.0)
        assert np.all(va > 1.0), f"lv={lv}: post-TTC collapsed to {va}"
        assert np.allclose(va, 179.0), f"lv={lv}: post-TTC hold {va}"


class TestV5CanonicalUnsortedWind:
    """V5B-F3: canonical unsorted wind onset must be chronology-safe.

    The raw branch fails closed on unsorted rows; canonical had no guard and
    took file-order ``.iloc[0]``.  Wind onset must be the chronological first.
    """

    def test_canonical_unsorted_wind_onset_chronological(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_canon_unsorted"
        raw_dir.mkdir(parents=True)
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"

        # Canonical schema: time_ms already relative, x_pos present, no sys_time.
        # File order puts t=200 (wind) BEFORE t=20 (wind).
        n = 30
        t_sorted = np.arange(n, dtype=np.float64) * 4.0  # 0, 4, …, 116
        # File-order permutation: swap first and the row at t=200.
        # Build explicit file-order times: 200, 0, 4, …, 20, …, 116
        time_ms_file = np.concatenate([[200.0], t_sorted])
        wind = np.zeros(len(time_ms_file), dtype=int)
        wind[0] = 1   # t=200 (file first, chronological last)
        wind[6] = 1   # t=20 (chronological first)

        df_k = pd.DataFrame({
            "session_id": s1,
            "trial_id": 1,
            "time_ms": time_ms_file,
            "x_pos": np.zeros(len(time_ms_file)),
            "y_pos": np.zeros(len(time_ms_file)),
            "heading": np.zeros(len(time_ms_file)),
            "velocity": np.zeros(len(time_ms_file)),
            "acceleration": np.zeros(len(time_ms_file)),
            "visual_angle": np.zeros(len(time_ms_file)),
            "wind_state": wind,
            "l_v_ratio": np.zeros(len(time_ms_file)),
        })
        df_k.to_csv(kin_path, index=False)

        details = json.dumps({
            "type": "baseline_wind", "target_ttc_ms": None,
            "lv_ratio_ms": None, "wind_dir": "right", "screen_side": "right",
        })
        evt = pd.DataFrame([
            {"event_type": "trial_start", "time_ms": 0.0,
             "trial_id": 1, "event_value": details},
        ])
        evt.to_csv(evt_path, index=False)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        stim = evt_out[evt_out["event_type"] == "stimulus_onset"]
        assert len(stim) == 1
        onset = float(stim.iloc[0]["time_ms"])
        assert onset == pytest.approx(20.0, abs=1e-6), (
            f"chronological-first wind onset expected 20 ms, got {onset}"
        )


class TestV5DuplicateLooming:
    """NEW-3: duplicate Looming transitions must warn, not silently drop.

    ``_first_per_trial`` keeps the chronologically first (correct semantics)
    but a multiplicity diagnostic must fire under the project's statistical
    rigor standard.
    """

    def test_duplicate_looming_warns_and_keeps_first(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        s1 = "session_dup_loom"
        # Second Looming at 0.200 s (after the first at 0.120 s).
        extra = [
            _raw_evt_row(
                "phase_transition", 0.200, 1,
                details=json.dumps({"from_phase": "Baseline",
                                    "to_phase": "Looming"}),
            ),
        ]
        _write_raw_pair(
            raw_dir, s1,
            trial_start_s=0.100,
            looming_s=0.120,
            kin_first_s=0.100,
            trial_type="baseline_visual",
            lv_ratio_ms=120.0,
            extra_events=extra,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        warn_msgs = [str(w.message) for w in caught]
        assert any("duplicate" in m.lower() for m in warn_msgs), (
            f"expected a duplicate-Looming warning, got: {warn_msgs}"
        )

        # Chronologically first Looming (0.120 s → 20 ms) must win.
        evt_out = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        onset = _get_stimulus_onset(evt_out, trial_id=1)
        assert onset == pytest.approx(20.0, abs=1e-6), (
            f"chronologically-first Looming expected 20 ms, got {onset}"
        )


class TestV7InterleavedTrialIsolation:
    """Interleaved trial rows must not leak coordinates or velocity across trials."""

    def test_interleaved_trials_independent_velocity(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_interleave"
        out_dir = tmp_path / "out_interleave"
        s1 = "0.100cricket_session_1"
        raw_dir.mkdir(parents=True)
        kin_path = raw_dir / f"{s1}_kinematics.csv"
        evt_path = raw_dir / f"{s1}_events.csv"

        rows = []
        for i in range(20):
            for tid, t0 in [(1, 100.0), (2, 200.0)]:
                rows.append(_raw_kin_row(i, t0 + i * 0.004, tid, dx=0.1, dy=0.0, dz=0.0))
        _write_raw_kin_rows(kin_path, s1, rows)

        details = json.dumps({"type": "baseline_wind", "target_ttc_ms": None})
        evt_rows = [
            _raw_evt_row("trial_start", 100.0, 1, details=details),
            _raw_evt_row("trial_start", 200.0, 2, details=details),
        ]
        pd.DataFrame(evt_rows).to_csv(evt_path, index=False)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        kin_out = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")
        v1 = kin_out.loc[(kin_out["trial_id"] == 1) & (kin_out["time_ms"] > 0), "velocity"].to_numpy()
        v2 = kin_out.loc[(kin_out["trial_id"] == 2) & (kin_out["time_ms"] > 0), "velocity"].to_numpy()

        # dx=0.1 mm per 0.004 s -> step_dist_mm = 0.1, velocity = 0.1 / 0.004 / 10 = 2.5 cm/s
        np.testing.assert_allclose(v1, 2.5, atol=1e-8)
        np.testing.assert_allclose(v2, 2.5, atol=1e-8)


class TestV7TtcDeclarationValidation:
    """Declared target_ttc_ms and lv_ratio_ms must be finite or raise ValueError."""

    def test_corrupt_target_ttc_raises(self, tmp_path: Path):
        from scripts.pre_load_adapt import parse_trial_events
        evt_path = tmp_path / "corrupt_ttc.csv"
        pd.DataFrame([
            {
                "trial_id": 1,
                "time_ms": 0.0,
                "event_type": "trial_start",
                "event_value": json.dumps({"type": "baseline_visual", "target_ttc_ms": "inf"}),
            }
        ]).to_csv(evt_path, index=False)

        with pytest.raises(ValueError, match="non-finite|unparseable"):
            parse_trial_events(evt_path)

    def test_corrupt_lv_ratio_raises(self, tmp_path: Path):
        from scripts.pre_load_adapt import parse_trial_events
        evt_path = tmp_path / "corrupt_lv.csv"
        pd.DataFrame([
            {
                "trial_id": 1,
                "time_ms": 0.0,
                "event_type": "trial_start",
                "event_value": json.dumps({"type": "baseline_visual", "lv_ratio_ms": "invalid_number"}),
            }
        ]).to_csv(evt_path, index=False)

        with pytest.raises(ValueError, match="unparseable"):
            parse_trial_events(evt_path)



def test_legacy_visual_first_frame_artifact_does_not_shift_snapshot(tmp_path: Path):
    """Raw adapter omits init_deg; one early angle spike must not define collision."""
    from nsmor.data_extractor import extract_mcmc_snapshot, resolve_snapshot_anchor
    from nsmor.pipeline.io import extract_trial_data
    from nsmor.pipeline.labeling import find_event_time

    raw_dir, out_dir, session = tmp_path / "raw", tmp_path / "adapted", "legacy_visual"
    _write_raw_pair(raw_dir, session, trial_start_s=10.0, looming_s=10.02,
                    kin_first_s=10.0, n_samples=1800)
    adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
    kin = pd.read_csv(out_dir / session / f"{session}_kinematics.csv")
    evt = pd.read_csv(out_dir / session / f"{session}_events.csv")
    details = json.loads(evt.loc[evt.event_type == "trial_start", "event_value"].iloc[0])
    assert "init_deg" not in details  # Actual legacy metadata path.
    def trial_from(frame):
        return extract_trial_data({"kinematics":frame, "events":evt}, session, 1)
    clean = trial_from(kin)
    onset = find_event_time(clean["event_types"], clean["event_times"], "stimulus_onset")
    expected, rule = resolve_snapshot_anchor(clean, onset)
    assert rule == "looming_collision"
    assert expected == pytest.approx(onset + 120.0 / math.tan(math.radians(1)), abs=0.01)
    baseline = extract_mcmc_snapshot(clean, stimulus_onset_ms=onset)
    changed = kin.copy()
    first = changed.index[(changed.visual_angle > 0) & (changed.visual_angle < 179)][0]
    changed.loc[first, "visual_angle"] = 35.0
    corrupted = trial_from(changed)
    actual, rule = resolve_snapshot_anchor(corrupted, onset)
    assert rule == "looming_collision" and actual == pytest.approx(expected, abs=0.01)
    assert np.array_equal(extract_mcmc_snapshot(corrupted, stimulus_onset_ms=onset), baseline)


def _four_degree_visual_trial_in_long_recording(tmp_path: Path):
    from nsmor.pipeline.io import extract_trial_data, load_and_concat_sessions
    from tests.test_pipeline import _make_visual_only_csvs

    # The 2-degree fixture runs past 6874 ms; give it a coherent 4-degree trace.
    session_dir = tmp_path / 'raw' / 'visual_session'
    session_dir.mkdir(parents=True)
    kin, evt = _make_visual_only_csvs(session_dir, n_trials=1, init_deg=2.0)
    frame = pd.read_csv(kin)
    collision_ms = 120.0 / math.tan(math.radians(2.0))
    time_ms = frame['time_ms'].to_numpy()
    frame['visual_angle'] = np.where(
        time_ms < collision_ms,
        np.degrees(2.0 * np.arctan2(120.0, np.maximum(collision_ms - time_ms, 1e-6))),
        179.0,
    )
    frame.to_csv(kin, index=False)
    return extract_trial_data(load_and_concat_sessions([kin], [evt]), 'visual_session', 0)


def test_declared_visual_angle_conflict_fails_snapshot_with_identity(tmp_path: Path):
    """A coherent 4-degree trace cannot be anchored by a declared 2 degrees."""
    from nsmor.data_extractor import build_snapshot_dataset, extract_mcmc_snapshot
    from nsmor.pipeline.labeling import assign_ground_truth_labels

    trial = _four_degree_visual_trial_in_long_recording(tmp_path)
    assert trial['time_ms'][-1] > 6874.795396
    with pytest.raises(ValueError, match='declared.*angle.*trace'):
        extract_mcmc_snapshot(trial, stimulus_onset_ms=0.0)
    labeled = assign_ground_truth_labels([trial])
    assert len(labeled) == 1
    with pytest.raises(ValueError, match=r'labeled_trials\[0\].*visual_session.*0.*declared.*angle.*trace'):
        build_snapshot_dataset(labeled)


def test_declared_visual_angle_tolerates_one_early_artifact(tmp_path: Path):
    from nsmor.data_extractor import extract_mcmc_snapshot, resolve_snapshot_anchor

    trial = _four_degree_visual_trial_in_long_recording(tmp_path)
    trial['event_values'] = trial['event_values'].copy()
    idx = int(np.flatnonzero(trial['event_types'] == 'trial_start')[0])
    details = json.loads(trial['event_values'][idx])
    details['init_deg'] = 4.0
    trial['event_values'][idx] = json.dumps(details)
    expected, rule = resolve_snapshot_anchor(trial, 0.0)
    assert rule == 'looming_collision'
    assert expected == pytest.approx(3436.350394, abs=0.001)
    baseline = extract_mcmc_snapshot(trial, stimulus_onset_ms=0.0)
    trial['visual_angle'] = trial['visual_angle'].copy()
    first = int(np.flatnonzero((trial['visual_angle'] > 0) & (trial['visual_angle'] < 179))[0])
    trial['visual_angle'][first] = 35.0
    actual, rule = resolve_snapshot_anchor(trial, 0.0)
    assert rule == 'looming_collision' and actual == pytest.approx(expected, abs=0.001)
    assert np.array_equal(extract_mcmc_snapshot(trial, stimulus_onset_ms=0.0), baseline)

@pytest.mark.parametrize('producer', ['prepare_data', 'prepare_metadata'])
def test_declared_angle_conflict_aborts_production_snapshot_before_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, producer: str,
):
    """Both producers use the strict Snapshot gate on synthetic canonical CSVs."""
    trial = _four_degree_visual_trial_in_long_recording(tmp_path)
    assert trial['time_ms'][-1] > 6874.795396
    output = tmp_path / 'should_not_exist.pt'
    if producer == 'prepare_data':
        from scripts.prepare_data import prepare_dataset
        run = lambda: prepare_dataset(raw_dir=tmp_path / 'raw', output_path=output)
    else:
        import sys
        from scripts.prepare_metadata import main
        monkeypatch.setattr(sys, 'argv', [
            'prepare_metadata.py', '--raw_dir', str(tmp_path / 'raw'), '--output', str(output),
        ])
        run = main
    with pytest.raises(ValueError, match=r'labeled_trials\[0\].*visual_session.*0.*declared.*angle.*trace'):
        run()
    assert not output.exists()
