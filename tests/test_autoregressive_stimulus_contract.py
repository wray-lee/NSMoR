"""Synthetic physical-input and export contract for the nine I paradigms."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION
from nsmor.model_nsmor_core import NSMoRCore
from scripts import simulate_autoregressive as ar

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


@pytest.mark.parametrize("dt_ms", [4.0, 10.0])
def test_cli_outputs_bind_synthetic_run_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dt_ms: float,
) -> None:
    """Real rollout/exports record assumptions even when CSV motion is identical."""
    config = {
        "model": {"hidden_dim": 4, "dt_ms": dt_ms, "sensory_noise_std": 0.0},
        "training": {"normalize_targets": True, "seed": 42, "lr": 0.001},
    }
    model = NSMoRCore(**config["model"])
    # Zero weights make all motion identical regardless of prior/fatigue.
    for parameter in model.parameters():
        parameter.data.zero_()
    payload = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "config": config, "model_state_dict": model.state_dict(),
        "target_mean": 0.0, "target_std": 2.0, "target_clip_cm_s": 100.0,
        "rng_state": torch.get_rng_state(),
    }
    checkpoint = tmp_path / "tiny.pth"
    torch.save(payload, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected_seed(*_args: object, **_kwargs: object) -> None:
        pytest.fail("No-seed simulation must not seed or restore checkpoint RNG")

    monkeypatch.setattr(torch, "manual_seed", unexpected_seed)
    monkeypatch.setattr(torch, "set_rng_state", unexpected_seed)
    monkeypatch.setattr(np.random, "seed", unexpected_seed)
    for name in ("visual_only", "double_pulse"):
        monkeypatch.setitem(ar.PARADIGMS, name, replace(
            ar.PARADIGMS[name], total_duration_ms=24.0,
            baseline_ms=12.0, visual_lead_ms=8.0,
            wind_onset_delta_ms=-8.0, wind_offset_delta_ms=8.0,
            wind_gap_deltas_ms=(-4.0, 0.0) if name == "double_pulse" else None,
        ))

    def run(label: str, *options: str) -> tuple[dict, bytes]:
        output = tmp_path / label
        monkeypatch.setattr(sys, "argv", [
            "simulate_autoregressive.py", "--checkpoint", str(checkpoint),
            "--paradigms", "double_pulse", "visual_only",
            "--output_dir", str(output), *options,
        ])
        ar.main()
        manifest_path = output / "simulation_manifest.json"
        assert manifest_path.is_file(), "CLI outputs lack synthetic run provenance"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for filename, digest in manifest["outputs_sha256"].items():
            assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest
        assert set(manifest["outputs_sha256"]) == {
            "events.csv", "kinematics.csv", "stimuli.csv", "stimulus_summary.json",
        }
        return manifest, (output / "kinematics.csv").read_bytes()

    manifest, motion = run("default")
    assert manifest["schema_version"] == 1
    assert manifest["scope"] == "synthetic_autoregressive"
    assert manifest["empirical_dataset_bound"] is False
    assert manifest["nested_prior_artifact_bound"] is False
    assert manifest["checkpoint"] == {
        "path": str(checkpoint.resolve()),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    assert manifest["checkpoint_config"] == config
    assert manifest["training_config"] == config["training"]
    assert manifest["training_config_status"] == "saved_checkpoint_only; no defaults inferred"
    assert manifest["resolved_model_config"]["hidden_dim"] == 4
    assert manifest["resolved_model_config"]["lif_rel_refract_ms"] == 20.0
    assert manifest["dt_ms"] == dt_ms
    assert manifest["prediction_units"] == {
        "units": "cm/s", "target_mean": 0.0, "target_std": 2.0,
        "target_clip_cm_s": 100.0, "inference_clipped": False,
    }
    assert manifest["synthetic_prior"] == {
        "source": "uniform_default", "base_vector": [0.25] * 4,
        "class_order": ["P_startle", "P_walk", "P_pre_active", "P_no_response"],
        "model_input_dtype": "float32",
    }
    assert manifest["paradigm_order"] == ["double_pulse", "visual_only"]
    assert manifest["paradigm_specs"][0]["wind_gap_deltas_ms"] == [-4.0, 0.0]
    assert manifest["randomness"]["cli_seed"] is None
    assert manifest["randomness"]["seed_status"] == "not_set_by_cli"
    assert manifest["randomness"]["checkpoint_rng_restored"] is False
    assert manifest["randomness"]["rng_state_captured"] is False
    assert manifest["randomness"]["reproducibility_guaranteed"] is False
    fatigue = manifest["fatigue"]
    assert fatigue["initial_level"] == 0.0 and fatigue["cap"] == 1.0
    assert fatigue["trial_cost"] == 0.15
    assert fatigue["recovery_rate_per_second"] == 0.005
    assert fatigue["max_fatigue_penalty"] == 0.6
    assert fatigue["iti_seconds"] == 120.0
    assert fatigue["update_timing"] == "before_every_trial_including_first"
    assert fatigue["within_trial_level"] == "constant"
    assert fatigue["neural_state_between_trials"] == "reset"
    assert fatigue["empirically_calibrated"] is False
    first, second = manifest["trials"]
    assert first["global_trial_id"] == 0 and first["paradigm"] == "double_pulse"
    assert first["fatigue_level"] == 0.15
    assert first["velocity_gain"] == 0.91
    np.testing.assert_allclose(first["effective_prior"], [0.2125, 0.2125, 0.25, 0.325])
    assert second["fatigue_level"] == pytest.approx(0.23232174541410394)

    changed, changed_motion = run("prior", "--mcmc_prior", "4", "2", "1", "3")
    assert changed_motion == motion
    assert changed["synthetic_prior"]["source"] == "cli_vector"
    assert changed["synthetic_prior"]["base_vector"] == [4.0, 2.0, 1.0, 3.0]
    np.testing.assert_allclose(changed["trials"][0]["effective_prior"],
                               [0.34, 0.17, 0.1, 0.39])
    assert changed["synthetic_prior"] != manifest["synthetic_prior"]

    changed, changed_motion = run(
        "fatigue", "--trial_cost", "0.5", "--recovery_rate", "0",
        "--max_fatigue_penalty", "0.2", "--iti_seconds", "3",
    )
    assert changed_motion == motion
    assert changed["fatigue"] != fatigue
    assert [trial["fatigue_level"] for trial in changed["trials"]] == [0.5, 1.0]
    assert [trial["velocity_gain"] for trial in changed["trials"]] == [0.9, 0.8]

    changed, changed_motion = run("disabled", "--trial_cost", "0")
    assert changed_motion == motion
    assert changed["fatigue"]["trial_cost"] == 0.0
    assert all(trial["fatigue_level"] == 0.0 for trial in changed["trials"])
    assert all(trial["effective_prior"] == [0.25] * 4 for trial in changed["trials"])

    # Replacement at the same path must still change checkpoint provenance.
    payload["config"]["training"]["lr"] = 0.002
    torch.save(payload, checkpoint)
    changed, changed_motion = run("checkpoint")
    assert changed_motion == motion
    assert changed["checkpoint"]["path"] == manifest["checkpoint"]["path"]
    assert changed["checkpoint"]["sha256"] != manifest["checkpoint"]["sha256"]
    assert changed["checkpoint_config"]["training"]["lr"] == 0.002

    # Same config, different weights: digest still distinguishes the checkpoint.
    next(model.parameters()).data.fill_(0.1)
    torch.save(payload, checkpoint)
    reweighted, reweighted_motion = run("weights")
    assert reweighted_motion == motion
    assert reweighted["checkpoint_config"] == changed["checkpoint_config"]
    assert reweighted["checkpoint"]["sha256"] != changed["checkpoint"]["sha256"]


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
