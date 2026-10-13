"""Synthetic-only tests for the phase-2 secondary battery."""
import csv
import hashlib
import json

import numpy as np
import pytest
import torch
import yaml

from nsmor.analysis import secondary_battery as battery
from nsmor.config import FeatureConfig, PIPELINE_SEMANTICS_VERSION
from nsmor.model_nsmor_core import NSMoRCore
from nsmor.model_utils import _extract_model_params
from nsmor.pipeline.grouping import animal_keys_of, grouped_train_val_split
from scripts.analyze_phase2_secondary import main


def test_window_edges_and_no_responder():
    observed = np.zeros(60)
    observed[5] = 10
    observed[55:] = 10
    masks, response = battery.scoring_windows(observed, 2, 4)
    assert response == 5
    assert np.flatnonzero(masks["onset"]).tolist() == list(range(2, 55))
    assert np.flatnonzero(masks["sustained"]).tolist() == list(range(55, 60))
    assert masks["overall"].sum() == 58
    masks, response = battery.scoring_windows(np.zeros(60), 2, 4)
    assert response is None and not masks["onset"].any() and not masks["sustained"].any()
    assert masks["overall"].sum() == 58
    masks, _ = battery.scoring_windows(observed, None, 4)
    assert not any(m.any() for m in masks.values())


def test_baseline_invalid_censored_and_exact_tie():
    rms = np.zeros(70)
    invalid = battery.departure_timing(rms, 61, 4)
    assert invalid["status"] == "invalid_baseline" and invalid["right_censored"] is None
    assert invalid["baseline_n_frames"] == 62 and invalid["baseline_effective_ms"] == 248
    assert invalid["baseline_requested_ms"] == 250
    censored = battery.departure_timing(rms, 62, 4)
    assert censored["status"] == "right_censored" and censored["latency_ms"] is None
    assert censored["opportunity_end_ms"] == 32 and censored["opportunity_end_frame_exclusive"] == 70
    assert censored["threshold"] == 3e-6
    rms[64] = 1
    event = battery.departure_timing(rms, 62, 4)
    assert event["status"] == "observed" and event["event_frame"] == 64 and event["latency_ms"] == 8
    assert not event["right_censored"]


def test_bouts_holes_and_false_alarm_denominators():
    assert not battery._bout(np.array([True, False, True]))
    assert battery._bout(np.array([False, True, True]))
    counts = battery.prestim_counts(np.array([12., -12., 12.]), np.array([0., 12., 0.]), 3)
    assert counts["prestim_false_alarm_frames"] == 2 and counts["prestim_below_observed_frames"] == 2
    assert counts["prestim_observed_escape_frames"] == 1 and counts["prestim_observed_bout_trials"] == 0
    assert counts["prestim_predicted_bout_trials"] == counts["prestim_strict_false_alarm_bout_trials"] == 0
    assert counts["prestim_below_observed_predicted_bout_trials"] == 0
    # All-trial model bouts and no-observed-bout false alarms have separate denominators.
    counts = battery.prestim_counts(np.array([12., 12.]), np.array([12., 12.]), 2)
    assert counts["prestim_predicted_bout_trials"] == 1
    assert counts["prestim_observed_no_bout_trials"] == counts["prestim_strict_false_alarm_bout_trials"] == 0
    assert sum(counts[k] for k in ("prestim_proxy_exactly_zero_frames", "prestim_proxy_0_to_10_frames", "prestim_proxy_ge10_frames")) == 2


def test_known_signed_residual_moments():
    mse, bias, variance = battery.residual_moments(np.array([1., 1.]), np.array([2., 4.]))
    assert (mse, bias, variance) == (5., -2., 1.)
    assert mse == bias ** 2 + variance


@pytest.fixture
def artifacts(tmp_path):
    n, length = 8, 72
    sessions = [f"prefix_{i // 2}_session_{i % 2 + 1}" for i in range(n)]
    train, val = grouped_train_val_split(sessions, n, val_split=0.25, random_seed=42)
    prefixes = animal_keys_of(sessions)
    source = {
        "X_seqs": [np.tile([i + 1, 0., 7., 9., 0., 0., 0., 0.], (length, 1)).astype(np.float32) for i in range(n)],
        "Y_seqs": [np.r_[np.zeros(62), np.full(10, 12.)].astype(np.float32) for _ in range(n)],
        "labels": np.zeros(n, dtype=np.int64), "lengths": np.full(n, length, dtype=np.int64),
        "anchor_frames": [62] * n, "anchor_rules": ["recorded_visual_reference"] * n,
        "trial_ids": list(range(100, 100 + n)), "session_ids": sessions,
        "mcmc_priors": np.tile([1., 0., 0., 0.], (n, 1)), "feature_config": FeatureConfig(),
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_5fold_recording_prefix_grouped_cv", "animal_identity_status": "unverified",
    }
    dataset = tmp_path / "dataset.pt"
    torch.save(source, dataset)
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    nested = np.tile([0., 1., 0., 0.], (n, 1))
    prior_data = {"nested_priors": nested, "train_indices": train, "val_indices": val,
        "train_priors": nested[train], "val_priors": nested[val], "source_fingerprint": digest,
        "is_nested_cv": True, "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "split_seed": 42, "val_split": 0.25, "n_inner_folds": 2,
        "recording_prefix_keys": prefixes,
        "train_recording_prefixes": sorted(set(prefixes[train].tolist())),
        "val_recording_prefixes": sorted(set(prefixes[val].tolist())),
        "n_train_recording_prefixes": len(set(prefixes[train])), "n_val_recording_prefixes": len(set(prefixes[val])),
        "mcmc_prior_provenance": "nested_outer_seed42_inner_2fold_recording_prefix_grouped", "animal_identity_status": "unverified"}
    prior = tmp_path / "prior.pt"
    torch.save(prior_data, prior)
    repo = battery.Path(battery.__file__).resolve().parents[2]
    config = yaml.safe_load((repo / "config/realdata-phase2-20261010/R2-history-ablated-seed42.yaml").read_text())
    config.pop("phase2")
    config["model"].update(hidden_dim=4, dropout=0.)
    torch.manual_seed(42)
    model = NSMoRCore(**_extract_model_params(config["model"]))
    ckpt_data = {"config": config, "model_state_dict": model.state_dict(),
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION, "zero_input_channels": [2, 3], "selection_metric": "mse",
        "dataset_path": str(dataset), "dataset_source_sha256": digest, "dataset_source_binding": "sha256_bound",
        "is_nested_cv": True, "nested_prior_artifact": str(prior),
        "nested_prior_artifact_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
        "nested_prior_fingerprint": digest, "nested_split_seed": 42, "nested_val_split": 0.25,
        "validation_scope": "nested_outer_validation", "animal_identity_status": "unverified",
        "mcmc_prior_provenance": prior_data["mcmc_prior_provenance"]}
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(ckpt_data, checkpoint)
    return checkpoint, dataset, prior, val.tolist(), source, ckpt_data


def test_cli_exact_validation_rows_actual_r2_input_and_exports(artifacts, tmp_path, monkeypatch):
    checkpoint, dataset, prior, selected, source, _ = artifacts
    real_load = battery.load_model_from_checkpoint
    inputs = []
    def capture(path, device):
        model = real_load(path, device)
        model.register_forward_pre_hook(lambda _, args: inputs.append(args[0].clone()))
        return model
    monkeypatch.setattr(battery, "load_model_from_checkpoint", capture)
    output = tmp_path / "out"
    assert main(["--checkpoint", str(checkpoint), "--arm", "R2", "--dataset", str(dataset),
                 "--nested_prior", str(prior), "--output_dir", str(output)]) == 0
    assert len(inputs) == 3 * len(selected)
    assert [int(inputs[i * 3][0, 0, 0]) - 1 for i in range(len(selected))] == selected
    for xb in inputs:
        assert torch.count_nonzero(xb[:, :, 2:4]) == 0
        assert torch.all(xb[:, :, 4:8] == torch.tensor([0., 1., 0., 0.]))
    assert all(np.all(x[:, 2:4] == [7., 9.]) for x in source["X_seqs"])
    with (output / "trial_level.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["row_index"]) for r in rows] == selected
    assert [int(r["source_trial_id"]) for r in rows] == [source["trial_ids"][i] for i in selected]
    assert all(int(r["cropped_reference_frame"]) == 62 and int(r["onset_frame_count"]) == 10 for r in rows)
    assert all(r["sustained_natural_mse"] == "" for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["population"]["selection"] == "outer_validation_only"
    assert summary["population"]["n_analyzed_train"] == 0
    assert summary["definitions"]["baseline_effective_ms"] == 248
    assert summary["provenance"]["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert summary["provenance"]["nested_prior_sha256"] == hashlib.sha256(prior.read_bytes()).hexdigest()
    assert summary["provenance"]["code_sha256"]["scripts/evaluate_realdata_phase2.py"] == battery.SCORER_SHA256
    assert summary["readout_window_effects"]["onset"]["lif_replaced"]["sign_flip"]["p_value"] is None
    assert summary["prestim_rates"]["false_alarm_frame_rate"]["denominator"] == 62 * len(selected)
    assert summary["analysis_claim"] == "model-internal exploratory"
    for extension in ("pdf", "svg", "png"):
        assert (output / f"secondary_battery.{extension}").stat().st_size > 1000
    with pytest.raises(FileExistsError):
        battery.run_battery(checkpoint, "R2", dataset, prior, output)


@pytest.mark.parametrize("change,reason", [
    ({"validation_scope": "diagnostic_global_oof"}, "nested_outer_validation"),
    ({"animal_identity_status": "verified"}, "unverified"),
    ({"is_nested_cv": False}, "unverified"),
    ({"zero_input_channels": []}, "zero_input_channels"),
    ({"zero_input_channels": [2.1, 3.]}, "integer"),
    ({"dataset_source_sha256": "a" * 64}, "fingerprint"),
    ({"nested_prior_artifact_sha256": "a" * 64}, "SHA-256"),
])
def test_checkpoint_provenance_rejection_before_inference(artifacts, tmp_path, monkeypatch, change, reason):
    checkpoint, dataset, prior, _, _, payload = artifacts
    payload.update(change)
    torch.save(payload, checkpoint)
    monkeypatch.setattr(NSMoRCore, "forward", lambda *_a, **_k: pytest.fail("Inference before validation"))
    with pytest.raises(ValueError, match=reason):
        battery.run_battery(checkpoint, "R2", dataset, prior, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("key,value,reason", [("random_seed", 45, "seed"), ("random_seed", 42., "seed"),
    ("dt_ms", 10., "conflicts"), ("time_consuming_mode", "depth_delay", "incompatible"),
    ("refinement_mode", "adaptive", "refinement_mode")])
def test_incompatible_config_rejection(artifacts, tmp_path, key, value, reason):
    checkpoint, dataset, prior, _, _, payload = artifacts
    section = "training" if key == "random_seed" else "model"
    payload["config"][section][key] = value
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match=reason):
        battery.run_battery(checkpoint, "R2", dataset, prior, tmp_path / "out")


@pytest.mark.parametrize("bad", ["input", "target", "prediction", "internals", "replacement"])
def test_stream_finite_guards(bad):
    class Model(torch.nn.Module):
        def forward(self, xb, lengths, **kwargs):
            pred = torch.zeros(1, 70)
            hidden = torch.zeros(1, 70, 2)
            if bad == "prediction" or (bad == "replacement" and kwargs.get("override_gates")):
                pred[0, 1] = float("nan")
            if bad == "internals":
                hidden[0, 1, 0] = float("inf")
            return pred, {"lif_spikes": hidden, "gru_hidden": hidden}
    xb = torch.zeros(1, 70, 8)
    if bad == "input":
        xb[0, 1, 2] = float("nan")  # Must reject before R2 zeroing could hide it.
    yb = torch.zeros(1, 70)
    if bad == "target":
        yb[0, 1] = float("nan")
    batch = (xb, yb, torch.tensor([70]))
    refs = [{"cropped_length": 70, "cropped_reference_frame": 62}]
    with pytest.raises(ValueError, match="Nonfinite"):
        list(battery.stream_trials(Model(), [batch], refs, "R2", 42, 4))


def test_stream_tie_and_short_sustained_window():
    class Model(torch.nn.Module):
        def forward(self, xb, lengths, **kwargs):
            hidden = torch.zeros(1, 120, 2)
            hidden[:, 64:] = 1
            return torch.zeros(1, 120), {"lif_spikes": hidden, "gru_hidden": hidden}
    yb = torch.zeros(1, 120)
    yb[0, 66] = 10
    yb[0, 116:] = 12  # Four sustained eligible frames, hence unavailable MSE.
    batch = (torch.zeros(1, 120, 8), yb, torch.tensor([120]))
    refs = [{"cropped_length": 120, "cropped_reference_frame": 62}]
    row = list(battery.stream_trials(Model(), [batch], refs, "R2", 42, 4))[0]
    assert row["timing_lif_minus_gru_ms"] == 0 and row["timing_order"] == "exact_zero_tie"
    assert row["sustained_frame_count"] == 4 and row["sustained_lif_replaced_mse"] is None
    assert row["onset_natural_bias_cm_s"] < 0


def test_summary_censor_denominator_and_common_uncensored_pairs():
    class Model(torch.nn.Module):
        def forward(self, xb, lengths, **kwargs):
            mode = int(xb[0, 0, 0])
            lif, gru = torch.zeros(1, 70, 2), torch.zeros(1, 70, 2)
            if mode in (2, 3):
                lif[:, 64:] = 1
            if mode == 2:
                gru[:, 66:] = 1
            return torch.zeros(1, 70), {"lif_spikes": lif, "gru_hidden": gru}
    refs = [{"cropped_length": 70, "cropped_reference_frame": 61 if i == 0 else 62,
             "source_trial_id": i, "session_id": "session", "recording_prefix": "prefix"} for i in range(4)]
    batches = [(torch.full((1, 70, 8), float(i)), torch.zeros(1, 70), torch.tensor([70])) for i in range(4)]
    rows = list(battery.stream_trials(Model(), batches, refs, "R0", 42, 4))
    summary = battery.summarize(rows)
    lif, gru = summary["timing"]["lif"], summary["timing"]["gru"]
    assert lif["eligible_trials"] == gru["eligible_trials"] == 3
    assert lif["invalid_baseline_trials"] == gru["invalid_baseline_trials"] == 1
    assert lif["censor_rate"] == 1 / 3 and gru["censor_rate"] == 2 / 3
    paired = summary["timing"]["common_uncensored_lif_minus_gru_ms"]
    assert paired["n"] == 1 and paired["median"] == -8 and paired["lif_first"] == 1
    assert rows[3]["timing_order"] == "unavailable"
    effect = summary["readout_window_effects"]["overall"]["lif_replaced"]
    assert effect["prefix_means"] == {"prefix": 0.} and effect["sign_flip"]["p_value"] is None
    assert "prefix_paired" not in effect


@pytest.mark.parametrize("arm,seed", [("R0", 43), ("R1", 44), ("R3", 42)])
def test_other_declared_arms_preserve_history(artifacts, tmp_path, monkeypatch, arm, seed):
    checkpoint, dataset, prior, selected, _, payload = artifacts
    payload["config"]["training"]["random_seed"] = seed
    payload["zero_input_channels"] = []
    payload["config"]["model"]["persistence_skip"] = 1. if arm == "R1" else 0.
    if arm == "R3":
        payload["config"]["loss"]["lambda_jerk"] = 0.
    torch.save(payload, checkpoint)
    real_load, inputs = battery.load_model_from_checkpoint, []
    def capture(path, device):
        model = real_load(path, device)
        model.register_forward_pre_hook(lambda _, args: inputs.append(args[0].clone()))
        return model
    monkeypatch.setattr(battery, "load_model_from_checkpoint", capture)
    summary = battery.run_battery(checkpoint, arm, dataset, prior, tmp_path / "out")
    assert summary["arm"] == arm and summary["seed"] == seed
    assert len(inputs) == 3 * len(selected)
    assert all(torch.all(xb[:, :, 2:4] == torch.tensor([7., 9.])) for xb in inputs)


def test_export_failure_is_loud(artifacts, tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    checkpoint, dataset, prior, *_ = artifacts
    def fail(*_a, **_k):
        raise OSError("synthetic export failure")
    monkeypatch.setattr(Figure, "savefig", fail)
    with pytest.raises(OSError, match="export failure"):
        battery.run_battery(checkpoint, "R2", dataset, prior, tmp_path / "out")
    assert not (tmp_path / "out/summary.json").exists()
