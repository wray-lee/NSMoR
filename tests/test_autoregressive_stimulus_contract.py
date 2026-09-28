"""Synthetic physical-input and export contract for the nine I paradigms."""
import csv
import json
import logging

import numpy as np
import pytest

from scripts.simulate_autoregressive import (
    PARADIGMS, TrialResult, export_events_csv, export_kinematics_csv,
    export_stimulus_evidence, generate_stimulus_paradigm, log_trial_summary,
)


@pytest.mark.parametrize("dt_ms", [4.0, 10.0])
def test_nine_declared_paradigms_have_distinct_physical_inputs(dt_ms):
    traces = {}
    for name, paradigm in PARADIGMS.items():
        time, angle, wind = generate_stimulus_paradigm(paradigm, dt_ms)
        assert time.shape == angle.shape == wind.shape
        assert len(time) == int(5000 / dt_ms) + (int(5700 / dt_ms) if name == "wind_only" else 0)
        np.testing.assert_allclose(np.diff(time), dt_ms)
        assert np.isfinite(angle).all() and np.isfinite(wind).all()
        assert set(np.unique(wind)) <= {0.0, 1.0}
        traces[name] = (time, angle, wind)
    assert len({(tuple(angle), tuple(wind)) for _, angle, wind in traces.values()}) == 9

    baseline, strong, weak = (traces[name][1] for name in
                              ("visual_only", "strong_looming", "weak_looming"))
    time = traces["visual_only"][0]
    approach = (time >= 1000) & (time < 2000)
    assert approach.sum() > 1
    assert np.all((strong[approach] > 0) & (strong[approach] < baseline[approach]))
    assert np.all((baseline[approach] < weak[approach]) & (weak[approach] < 180))
    assert np.all(np.diff(baseline[approach]) > 0)
    assert np.all(baseline[time < 1000] == 0) and np.all(baseline[time >= 2000] == 180)

    pulse_time, _, pulse_wind = traces["double_pulse"]
    rises = np.flatnonzero(np.diff(np.r_[0, pulse_wind]) == 1)
    falls = np.flatnonzero(np.diff(np.r_[pulse_wind, 0]) == -1)
    np.testing.assert_array_equal(pulse_time[rises], [1800, 2000])
    np.testing.assert_array_equal(pulse_time[falls] + dt_ms, [1900, 2100])
    for name, onset in (("early_wind_ttc_neg373", 2000 - 373),
                        ("early_wind_ttc_neg119", 2000 - 119),
                        ("sync_ttc_0", 2000), ("late_wind_ttc_plus200", 2200)):
        t, _, wind = traces[name]
        first = t[np.flatnonzero(wind)[0]]
        assert onset <= first < onset + dt_ms


def test_stimulus_evidence_replays_events_and_inputs(tmp_path):
    trials = []
    for paradigm in PARADIGMS.values():
        time, angle, wind = generate_stimulus_paradigm(paradigm)
        zero = np.zeros(len(time))
        trials.append(TrialResult(
            paradigm_name=paradigm.name, time_ms=time, position=zero, velocity=zero,
            acceleration=zero, v_vis=angle, wind=wind, gate_lif=zero, gate_gru=zero,
            target_ttc_ms=paradigm.target_ttc_ms, lv_ratio=paradigm.lv_ratio,
            stimulus_onset_ms=(paradigm.baseline_ms + paradigm.target_ttc_ms - paradigm.visual_lead_ms
                               if paradigm.has_visual else paradigm.baseline_ms),
            collision_ms=paradigm.baseline_ms + paradigm.target_ttc_ms,
        ))
    export_stimulus_evidence(trials, tmp_path)
    export_events_csv(trials, tmp_path / "events.csv")
    export_kinematics_csv(trials, tmp_path / "kinematics.csv")
    summary = json.loads((tmp_path / "stimulus_summary.json").read_text())
    assert summary["schema_version"] == 1 and summary["dt_ms"] == 4.0
    assert [item["type"] for item in summary["trials"]] == list(PARADIGMS)
    with (tmp_path / "stimuli.csv").open(newline="") as stream:
        inputs = list(csv.DictReader(stream))
    with (tmp_path / "kinematics.csv").open(newline="") as stream:
        motion = list(csv.DictReader(stream))
    with (tmp_path / "events.csv").open(newline="") as stream:
        events = list(csv.DictReader(stream))
    assert len(inputs) == len(motion) == sum(len(trial.time_ms) for trial in trials)
    offset = 0
    for idx, (trial, parameters) in enumerate(zip(trials, summary["trials"])):
        frames = inputs[offset:offset + len(trial.time_ms)]
        states = motion[offset:offset + len(trial.time_ms)]
        offset += len(frames)
        np.testing.assert_array_equal([float(row["sys_time"]) for row in frames], trial.time_ms)
        np.testing.assert_array_equal([float(row["visual_angle_deg"]) for row in frames], trial.v_vis)
        np.testing.assert_array_equal([float(row["v_vis"]) for row in frames], trial.v_vis)
        np.testing.assert_array_equal([float(row["wind"]) for row in frames], trial.wind)
        assert all(int(row["global_trial_id"]) == idx for row in frames)
        assert [int(row["stim_state"]) for row in states] == list(
            ((trial.v_vis > 0) | (trial.wind > 0)).astype(int))
        assert parameters["visual_input_semantics"] == "angle_deg"
        assert parameters["visual_gain"] == 1.0 and parameters["visual_tau_ms"] is None
        assert parameters["visual_onset_ms"] == (trial.stimulus_onset_ms if idx != 1 else None)
        assert parameters["collision_ms"] == trial.collision_ms
        if idx != 1:
            active = (trial.time_ms >= parameters["visual_onset_ms"]) & (trial.time_ms < trial.collision_ms)
            expected = np.degrees(2 * np.arctan(trial.lv_ratio / (trial.collision_ms - trial.time_ms[active])))
            np.testing.assert_allclose(trial.v_vis[active], expected, rtol=1e-14)
        ev = events[4 * idx:4 * idx + 4]
        assert [e["event_name"] for e in ev] == [
            "trial_start", "stimulus_onset", "phase_transition", "trial_stop"]
        assert float(ev[1]["timestamp"]) == trial.stimulus_onset_ms
        assert float(ev[2]["timestamp"]) == trial.collision_ms
        rebuilt = np.zeros(len(trial.wind))
        for start, stop in parameters["wind_intervals_ms"]:
            rebuilt[(trial.time_ms >= start) & (trial.time_ms < stop)] = 1.0
        np.testing.assert_array_equal(rebuilt, trial.wind)
    assert summary["trials"][8]["wind_intervals_ms"] == [[1800.0, 1900.0], [2000.0, 2100.0]]


def test_summary_latency_uses_each_trial_onset_including_prepended_wind(caplog):
    trials = []
    for name in ("visual_only", "wind_only"):
        paradigm = PARADIGMS[name]
        time, angle, wind = generate_stimulus_paradigm(paradigm)
        velocity = np.zeros(len(time))
        velocity[np.flatnonzero(time == 0)[0]] = 100.0  # before either stimulus
        velocity[np.flatnonzero(time == (1120 if name == "visual_only" else 2120))[0]] = 20.0
        zero = np.zeros(len(time))
        trials.append(TrialResult(
            name, time, zero, velocity, zero, angle, wind, zero, zero,
            paradigm.target_ttc_ms, paradigm.lv_ratio,
            stimulus_onset_ms=1000.0 if name == "visual_only" else 2000.0,
            collision_ms=2000.0,
        ))
    with caplog.at_level(logging.INFO):
        log_trial_summary(trials)
    for name in ("visual_only", "wind_only"):
        line = next(record.message for record in caplog.records
                    if record.message.startswith(name))
        assert line.split()[-2:] == ["20.000", "120.0"]
