"""Aggregate consistency checks and real restricted-load/vector-export smoke test."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.figure_evidence_support import build_support  # noqa: E402


@pytest.fixture
def artifacts():
    # Unequal prefix sizes catch accidental assumptions of 36 trials per prefix.
    prefixes = np.array(["v_train"] * 3 + ["v_val"] * 2 + ["w_train"] * 4
                        + ["m_train"] * 2 + ["m_val"])
    assert prefixes.shape == (12,)
    summary = {
        "meta": {
            "n_trials": 12, "n_recording_prefixes": 5, "n_sessions": 10,
            "counts_by_condition": {"visual_only": 5, "wind_only": 4, "multisensory": 3},
            "prefix_condition_pure": True,
            "animal_identity_status": "unverified -> recording prefix is the working cluster unit",
            "exploratory": True, "in_sample": True,
            "stationary_window_ms": 250.0, "stationary_threshold_cms": 10.0,
            "stationary_threshold_sensitivity_cms": 5.0,
        },
        "prefix_condition": {
            "v_train": "visual_only", "v_val": "visual_only", "w_train": "wind_only",
            "m_train": "multisensory", "m_val": "multisensory",
        },
        "eligibility": {
            "window_ms": 250.0, "threshold_cms": 10.0, "sensitivity_threshold_cms": 5.0,
            "by_condition": {
                "visual_only": {"n": 5, "n_eligible": 4, "n_excluded": 1,
                                "primary_responders": 1, "n_eligible_5cms": 3,
                                "n_excluded_5cms": 2, "elig5_responders": 1,
                                "excluded_fraction": 0.2},
                "wind_only": {"n": 4, "n_eligible": 3, "n_excluded": 1,
                              "primary_responders": 2, "n_eligible_5cms": 2,
                              "n_excluded_5cms": 2, "elig5_responders": 1},
                "multisensory": {"n": 3, "n_eligible": 2, "n_excluded": 1,
                                 "primary_responders": 2, "n_eligible_5cms": 1,
                                 "n_excluded_5cms": 2, "elig5_responders": 1},
            },
        },
    }
    prior = {
        "group_keys": prefixes.copy(), "recording_prefix_keys": prefixes,
        "train_indices": np.array([0, 1, 2, 5, 6, 7, 8, 9, 10]),
        "val_indices": np.array([3, 4, 11]),
        "train_recording_prefixes": ["v_train", "w_train", "m_train"],
        "val_recording_prefixes": ["v_val", "m_val"],
        "source_fingerprint": "a" * 64, "split_seed": 42, "is_nested_cv": True,
        "animal_identity_status": "unverified",
        "n_train": 9, "n_val": 3,
        "n_train_recording_prefixes": 3, "n_val_recording_prefixes": 2,
        "nested_priors": torch.full((12, 4), 0.25),
    }
    return summary, prior


def test_valid_unequal_prefix_counts_and_missing_wind_validation(artifacts):
    report = build_support(*artifacts)
    rows = {r["condition"]: r for r in report["condition_summary"]}
    assert report["totals"] == {
        "n_trials": 12, "n_recording_prefixes": 5, "n_sessions_reported_by_summary": 10,
        "train_trials": 9, "validation_trials": 3,
        "train_recording_prefixes": 3, "validation_recording_prefixes": 2,
    }
    assert report["missing_validation_conditions"] == ["wind_only"]
    assert (rows["visual_only"]["train_trials"], rows["visual_only"]["validation_trials"]) == (3, 2)
    assert (rows["multisensory"]["train_trials"], rows["multisensory"]["validation_trials"]) == (2, 1)
    assert (rows["wind_only"]["train_trials"], rows["wind_only"]["validation_trials"]) == (4, 0)
    assert (rows["wind_only"]["train_recording_prefixes"], rows["wind_only"]["validation_recording_prefixes"]) == (1, 0)
    assert rows["visual_only"]["primary_nonresponders"] == 3
    assert rows["visual_only"]["elig5_nonresponders"] == 2
    assert [r["prefix"] for r in report["recording_prefix_support"]] == ["m_train", "m_val", "v_train", "v_val", "w_train"]
    assert [r["n_trials"] for r in report["recording_prefix_support"]] == [2, 1, 3, 2, 4]
    assert report["scope"]["behavior"] == "EXPLORATORY / IN-SAMPLE"
    assert report["scope"]["model_split"] == "nested_outer_validation"
    assert "no untouched test" in report["scope"]["model_selection"]


def test_missing_condition_map_fails(artifacts):
    summary, prior = artifacts
    del summary["prefix_condition"]["w_train"]
    with pytest.raises(ValueError, match="prefix_condition missing required keys"):
        build_support(summary, prior)


@pytest.mark.parametrize("key,indices,match", [
    ("train_indices", [0, 0, 2, 5, 6, 7, 8, 9, 10], "duplicate"),
    ("val_indices", [0, 4, 11], "overlap"),
    ("val_indices", [3, 4, 12], "out-of-bounds"),
    ("val_indices", [3, 4, -1], "out-of-bounds"),
    ("val_indices", [3, 4], "complete partition"),
    ("val_indices", [[3, 4, 11]], "one-dimensional integer"),
    ("val_indices", [3.0, 4.0, 11.0], "one-dimensional integer"),
])
def test_invalid_split_indices_fail(artifacts, key, indices, match):
    summary, prior = artifacts
    prior[key] = np.asarray(indices)
    with pytest.raises(ValueError, match=match):
        build_support(summary, prior)


def test_prefix_straddling_splits_fails(artifacts):
    summary, prior = artifacts
    prior["train_indices"] = np.array([0, 2, 5, 6, 7, 8, 9, 10])
    prior["val_indices"] = np.array([3, 4, 11, 1])
    with pytest.raises(ValueError, match="prefix straddles"):
        build_support(summary, prior)


@pytest.mark.parametrize("key,value,match", [
    ("train_recording_prefixes", ["v_train", "w_train", "w_train"], "conflict"),
    ("val_recording_prefixes", ["v_val", "w_train"], "conflict"),
    ("n_train", 8, "conflicts with exact"),
    ("n_val_recording_prefixes", 3, "conflicts with exact"),
    ("source_fingerprint", "a" * 63, "64-hex"),
    ("source_fingerprint", "z" * 64, "64-hex"),
    ("animal_identity_status", "verified", "unverified"),
    ("split_seed", 7, "split_seed 42"),
])
def test_prior_metadata_conflicts_fail(artifacts, key, value, match):
    summary, prior = artifacts
    prior[key] = value
    with pytest.raises(ValueError, match=match):
        build_support(summary, prior)


@pytest.mark.parametrize("key,value,match", [
    ("n_trials", 13, "both prior key arrays"),
    ("n_recording_prefixes", 4, "conflicts with prior keys"),
    ("prefix_condition_pure", False, "must be true"),
    ("animal_identity_status", "verified", "unverified"),
    ("in_sample", False, "conflicts with"),
    ("stationary_threshold_cms", 5, "conflicts with eligibility"),
])
def test_summary_metadata_conflicts_fail(artifacts, key, value, match):
    summary, prior = artifacts
    summary["meta"][key] = value
    with pytest.raises(ValueError, match=match):
        build_support(summary, prior)


def test_group_keys_conflict_fails(artifacts):
    summary, prior = artifacts
    prior["group_keys"][0] = "v_val"
    with pytest.raises(ValueError, match="group_keys conflict"):
        build_support(summary, prior)


@pytest.mark.parametrize("changes,match", [
    ({"n": 6}, "trial counts conflict"),
    ({"n_excluded": 0}, "denominators"),
    ({"n_excluded_5cms": 1}, "denominators"),
    ({"primary_responders": 5}, "subset counts"),
    ({"n_eligible_5cms": 5, "n_excluded_5cms": 0}, "subset counts"),
    ({"elig5_responders": 2}, "subset counts"),
    ({"primary_responders": -1}, "nonnegative integer"),
    ({"primary_responders": 1.0}, "nonnegative integer"),
    ({"primary_responders": True}, "nonnegative integer"),
    ({"excluded_fraction": float("nan")}, "excluded_fraction"),
    ({"excluded_fraction": 0.25}, "excluded_fraction"),
    ({"labeling_responders": 5}, "exceeds its denominator"),
    ({"labeling_responders": 2, "labeling_responders_ungated": 1}, "exceed ungated"),
    ({"all_trials_responders_s2": 0}, "exceed all-trials"),
])
def test_incorrect_eligibility_counts_fail(artifacts, changes, match):
    summary, prior = artifacts
    summary["eligibility"]["by_condition"]["visual_only"].update(changes)
    with pytest.raises(ValueError, match=match):
        build_support(summary, prior)


@pytest.mark.parametrize("field,value", [("window_ms", 200), ("threshold_cms", 5), ("sensitivity_threshold_cms", 10)])
def test_changed_gate_thresholds_fail(artifacts, field, value):
    summary, prior = artifacts
    summary["eligibility"][field] = value
    with pytest.raises(ValueError, match="existing D1/5 cm/s gates"):
        build_support(summary, prior)


def test_unsupported_condition_is_not_fabricated(artifacts):
    summary, prior = artifacts
    summary["prefix_condition"]["w_train"] = "ttc_1"
    with pytest.raises(ValueError, match="unsupported condition"):
        build_support(summary, prior)


def test_condition_count_conflict_fails(artifacts):
    summary, prior = artifacts
    summary["meta"]["counts_by_condition"]["visual_only"] = 6
    with pytest.raises(ValueError, match="trial counts conflict"):
        build_support(summary, prior)


def test_zero_eligibility_remains_counts_without_division_by_zero(artifacts):
    summary, prior = artifacts
    summary["eligibility"]["by_condition"]["wind_only"].update({
        "n_eligible": 0, "n_excluded": 4, "primary_responders": 0,
        "n_eligible_5cms": 0, "n_excluded_5cms": 4, "elig5_responders": 0,
    })
    report = build_support(summary, prior)
    wind = next(r for r in report["condition_summary"] if r["condition"] == "wind_only")
    assert wind["primary_nonresponders"] == wind["elig5_nonresponders"] == 0
    json.dumps(report, allow_nan=False)


def _run_cli(summary_path, prior_path, output, tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", MPLCONFIGDIR=str(tmp_path / "mpl"))
    return subprocess.run([
        sys.executable, str(REPO / "scripts/figure_evidence_support.py"),
        "--behavior-summary", str(summary_path), "--nested-prior", str(prior_path),
        "--output-dir", str(output),
    ], env=env, capture_output=True, text=True, timeout=45)


def test_cli_restricted_load_vectors_provenance_and_collision(artifacts, tmp_path):
    summary, prior = artifacts
    summary_path, prior_path = tmp_path / "behavior_summary.json", tmp_path / "nested_split_seed42.pt"
    summary_path.write_text(json.dumps(summary))
    torch.save(prior, prior_path)
    inputs = {p: p.read_bytes() for p in (summary_path, prior_path)}
    output = tmp_path / "support"
    result = _run_cli(summary_path, prior_path, output, tmp_path)
    assert result.returncode == 0, result.stderr
    assert {p.name for p in output.iterdir()} == {
        "recording_prefix_support.csv", "condition_summary.csv", "evidence_support.json",
        "figure_evidence_support.pdf", "figure_evidence_support.svg", "figure_evidence_support.png",
    }
    report = json.loads((output / "evidence_support.json").read_bytes())
    sources = report["provenance"]["source_artifacts"]
    for name, path in (("behavior_summary", summary_path), ("nested_prior", prior_path)):
        assert sources[name]["sha256"] == hashlib.sha256(inputs[path]).hexdigest()
        assert path.read_bytes() == inputs[path]
    assert sources["script"]["sha256"] == hashlib.sha256((REPO / "scripts/figure_evidence_support.py").read_bytes()).hexdigest()
    assert report["provenance"]["behavior_dataset_binding"] == "not_recorded_in_source_summary"
    assert report["provenance"]["prior_source_fingerprint"] == "a" * 64
    assert report["provenance"]["prior_source_fingerprint_status"] == "recorded in prior; corpus not rehashed or loaded"
    assert report["missing_validation_conditions"] == ["wind_only"]
    assert (output / "figure_evidence_support.pdf").read_bytes().startswith(b"%PDF-")
    assert (output / "figure_evidence_support.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg = (output / "figure_evidence_support.svg").read_text()
    assert "<svg" in svg and "<text" in svg
    assert "EXPLORATORY / IN-SAMPLE" in svg and "nested_outer_validation" in svg
    assert "MISSING" in svg and "Behavior dataset binding is absent" in svg
    with (output / "recording_prefix_support.csv").open(newline="") as stream:
        prefix_rows = list(csv.DictReader(stream))
    assert list(prefix_rows[0]) == ["prefix", "condition", "split", "n_trials"]
    assert sum(int(r["n_trials"]) for r in prefix_rows) == 12
    with (output / "condition_summary.csv").open(newline="") as stream:
        condition_rows = {r["condition"]: r for r in csv.DictReader(stream)}
    assert condition_rows["wind_only"]["validation_trials"] == "0"
    assert condition_rows["visual_only"]["n_eligible_5cms"] == "3"
    before = {p: p.read_bytes() for p in output.iterdir()}
    collision = _run_cli(summary_path, prior_path, output, tmp_path)
    assert collision.returncode != 0 and "already exists" in collision.stderr
    assert all(p.read_bytes() == payload for p, payload in before.items())
    assert all(p.read_bytes() == payload for p, payload in inputs.items())


def test_cli_invalid_source_creates_no_output(artifacts, tmp_path):
    summary, prior = artifacts
    del summary["prefix_condition"]["w_train"]
    summary_path, prior_path = tmp_path / "behavior_summary.json", tmp_path / "nested_split_seed42.pt"
    summary_path.write_text(json.dumps(summary))
    torch.save(prior, prior_path)
    output = tmp_path / "support"
    result = _run_cli(summary_path, prior_path, output, tmp_path)
    assert result.returncode != 0 and "prefix_condition missing" in result.stderr
    assert not output.exists()
