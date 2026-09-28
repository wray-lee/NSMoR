"""Story20 mutation contracts against the original loader/Snapshot/prior seams.

The CSV fixtures are the existing pipeline fixtures; assertions check the
public snapshot and prior records. Tests stay immutable in proof copies.
"""
import json

import numpy as np
import pytest

from nsmor.data_extractor import build_snapshot_dataset, extract_mcmc_snapshot, resolve_snapshot_anchor
from nsmor.pipeline.io import extract_trial_data, load_and_concat_sessions
from nsmor.pipeline.labeling import assign_ground_truth_labels
from scripts.prepare_data import _audit_snapshot_drops, audit_prior_train_serve_shift
from tests.test_pipeline import _make_pure_wind_csvs, _make_synthetic_csvs, _make_visual_only_csvs
from tests.test_prior_shift_audit import CLASS_NAMES, _make_controlled_agreement_priors


def _expected_snapshot(trial, anchor_ms):
    """Independent fixture trace sampling at 50 ms before the requested event."""
    time_ms = trial["time_ms"]
    sample_ms = anchor_ms - 50.0
    idx = int(np.argmin(np.abs(time_ms - sample_ms)))
    background = (time_ms >= sample_ms - 200.0) & (time_ms < sample_ms)
    return np.array([
        trial["visual_angle"][idx], trial["l_v_ratio"][idx],
        trial["wind_state"][idx],
        np.mean(np.abs(trial["velocity"][background])),
        np.max(np.abs(trial["acceleration"][background])),
    ], dtype=np.float64)


def test_visual_collision_minus_50ms_full_snapshot(tmp_path):
    kin, evt = _make_visual_only_csvs(tmp_path, n_trials=1)
    data = load_and_concat_sessions([kin], [evt])
    trial = extract_trial_data(data, "visual_session", 0)
    assert np.any(trial["visual_angle"] > 0.0)
    assert not np.any(trial["wind_state"])
    collision_ms = 6874.795395691131
    expected = _expected_snapshot(trial, collision_ms)
    try:
        actual = extract_mcmc_snapshot(trial, stimulus_onset_ms=0.0)
    except ValueError as exc:
        assert False, f"visual-only trial was rejected by Snapshot extraction: {exc}"
    assert actual.shape == (5,)
    assert np.array_equal(actual, expected), "visual-only five features differ at collision minus 50 ms"


def test_visual_peak_mutation_cannot_move_physical_snapshot(tmp_path):
    kin, evt = _make_visual_only_csvs(tmp_path, n_trials=1)
    trial = extract_trial_data(load_and_concat_sessions([kin], [evt]), "visual_session", 0)
    # Worked stimulus geometry: l/v = 120 ms, initial angle = 2 degrees.
    collision_ms = 6874.795395691131
    expected = _expected_snapshot(trial, collision_ms)
    artifact_idx = int(np.argmin(np.abs(trial["time_ms"] - 1000.0)))
    trial["visual_angle"] = trial["visual_angle"].copy()
    trial["visual_angle"][artifact_idx] = 179.999
    actual = extract_mcmc_snapshot(trial, stimulus_onset_ms=0.0)
    assert actual.shape == (5,)
    assert np.array_equal(actual, expected), "angle artifact moved the physical collision snapshot"


@pytest.mark.parametrize("lv_ms, init_deg, onset_ms, collision_ms", [
    (120.0, 2.0, 0.0, 6874.795395691131),
    (80.0, 4.0, 320.0, 2610.9002626332485),
    (180.0, 6.0, -3.0, 3431.604603791078),
])
@pytest.mark.parametrize("declare_angle", [True, False])
def test_trial_geometry_and_looming_clock(tmp_path, lv_ms, init_deg, onset_ms, collision_ms, declare_angle):
    kin, evt = _make_visual_only_csvs(
        tmp_path, n_trials=1, lv_ratio_ms=lv_ms, init_deg=init_deg, looming_onset_ms=onset_ms,
    )
    trial = extract_trial_data(load_and_concat_sessions([kin], [evt]), "visual_session", 0)
    if not declare_angle:
        # Production trial_start declares l/v but omits the converter's initial angle.
        trial["event_values"] = trial["event_values"].copy()
        idx = int(np.flatnonzero(trial["event_types"] == "trial_start")[0])
        details = json.loads(trial["event_values"][idx])
        del details["init_deg"]
        trial["event_values"][idx] = json.dumps(details)
    anchor_ms, rule = resolve_snapshot_anchor(trial, stimulus_onset_ms=1234.0)
    assert rule == "looming_collision"
    assert anchor_ms == pytest.approx(collision_ms, abs=1e-8)
    expected = _expected_snapshot(trial, collision_ms)
    actual = extract_mcmc_snapshot(trial, stimulus_onset_ms=1234.0)
    assert actual.shape == (5,)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("field, value", [
    ("lv_ratio_ms", 0.0), ("lv_ratio_ms", -1.0), ("lv_ratio_ms", float("nan")),
    ("lv_ratio_ms", float("inf")), ("lv_ratio_ms", "bad"), ("lv_ratio_ms", None),
    ("lv_ratio_ms", True), ("init_deg", 0.0), ("init_deg", -2.0),
    ("init_deg", 180.0), ("init_deg", float("nan")), ("init_deg", float("inf")),
    ("init_deg", "bad"), ("init_deg", None), ("init_deg", True),
])
def test_invalid_visual_geometry_fails_loudly_with_trial_identity(tmp_path, field, value):
    kin, evt = _make_visual_only_csvs(tmp_path, n_trials=1)
    trial = extract_trial_data(load_and_concat_sessions([kin], [evt]), "visual_session", 0)
    labeled = assign_ground_truth_labels([trial])
    assert len(labeled) == 1
    trial["event_values"] = trial["event_values"].copy()
    idx = int(np.flatnonzero(trial["event_types"] == "trial_start")[0])
    details = json.loads(trial["event_values"][idx])
    details[field] = value
    trial["event_values"][idx] = json.dumps(details)
    with pytest.raises(ValueError, match=r"labeled_trials\[0\].*visual_session.*0.*could not be anchored"):
        build_snapshot_dataset(labeled)


def test_visual_collision_outside_recording_fails_loudly(tmp_path):
    kin, evt = _make_visual_only_csvs(tmp_path, n_trials=1)
    trial = extract_trial_data(load_and_concat_sessions([kin], [evt]), "visual_session", 0)
    trial = {key: value[:100] if isinstance(value, np.ndarray) and value.shape == trial["time_ms"].shape else value
             for key, value in trial.items()}
    with pytest.raises(ValueError, match="outside recorded trial"):
        extract_mcmc_snapshot(trial, stimulus_onset_ms=0.0)


def test_nonfinite_visual_trace_cannot_become_no_stimulus(tmp_path):
    kin, evt = _make_visual_only_csvs(tmp_path, n_trials=1)
    trial = extract_trial_data(load_and_concat_sessions([kin], [evt]), "visual_session", 0)
    trial["visual_angle"] = np.full_like(trial["visual_angle"], np.nan)
    with pytest.raises(ValueError, match="finite"):
        extract_mcmc_snapshot(trial, stimulus_onset_ms=2000.0)


def test_visual_declared_lv_cannot_disagree_with_empty_trace(tmp_path):
    kin, evt = _make_visual_only_csvs(tmp_path, n_trials=1)
    trial = extract_trial_data(load_and_concat_sessions([kin], [evt]), "visual_session", 0)
    trial["l_v_ratio"] = np.zeros_like(trial["l_v_ratio"])
    with pytest.raises(ValueError, match="l/v trace disagrees"):
        extract_mcmc_snapshot(trial, stimulus_onset_ms=0.0)


def test_ten_ms_grouping_fixture_retains_physical_visual_snapshots(tmp_path):
    from tests.test_prior_grouping import _write_corpus

    _write_corpus(tmp_path, n_animals=1, blocks_per_animal=1)
    kin = next(tmp_path.glob("*/kinematics.csv"))
    data = load_and_concat_sessions([kin], [kin.with_name("events.csv")])
    trials = [extract_trial_data(data, kin.parent.name, trial_id) for trial_id in range(4)]
    labeled = assign_ground_truth_labels(trials)
    snapshots, labels, kept, rules = build_snapshot_dataset(
        labeled, return_kept_indices=True, return_anchor_rules=True,
    )
    # Worked 30-ms l/v and 2-degree initial angle, from a 2000-ms looming onset.
    assert snapshots.shape == (4, 5) and labels.shape == (4,)
    assert kept == [0, 1, 2, 3]
    assert rules == ["looming_collision"] * 4
    assert np.array_equal(snapshots[0], _expected_snapshot(trials[0], 3718.6988489227826))


@pytest.mark.parametrize("kind", ["pure_wind", "multisensory"])
def test_wind_full_snapshot_bit_invariant(tmp_path, kind):
    if kind == "pure_wind":
        kin, evt = _make_pure_wind_csvs(tmp_path)
        trial = extract_trial_data(load_and_concat_sessions([kin], [evt]), "wind_session", 0)
    else:
        kin, evt = _make_synthetic_csvs(tmp_path, n_sessions=1, trials_per_session=1)
        trial = extract_trial_data(load_and_concat_sessions([kin], [evt]), "session_0", 0)
    assert np.any(trial["wind_state"])
    expected = _expected_snapshot(trial, 2000.0)
    try:
        actual = extract_mcmc_snapshot(trial, stimulus_onset_ms=2000.0)
    except ValueError as exc:
        assert False, f"wind-bearing trial was rejected by Snapshot extraction: {exc}"
    assert actual.shape == (5,)
    assert np.array_equal(actual, expected), "wind-bearing five features differ from original anchor"


def test_unanchorable_default_raises_and_opt_in_counts_drop(tmp_path):
    kin, evt = _make_pure_wind_csvs(tmp_path)
    trial = extract_trial_data(load_and_concat_sessions([kin], [evt]), "wind_session", 0)
    labeled = assign_ground_truth_labels([trial])
    assert len(labeled) == 1
    invalid = dict(labeled[0], session_id="unanchorable", trial_id=99, stimulus_onset_ms=20.0)
    record = [invalid, labeled[0]]
    try:
        build_snapshot_dataset(record, return_kept_indices=True)
    except ValueError as exc:
        message = str(exc)
    else:
        assert False, "default extraction silently skipped an unanchorable trial"
    assert "labeled_trials[0]" in message
    assert "unanchorable" in message and "99" in message
    assert "before trial start" in message and "skip" in message
    try:
        snapshots, labels, kept = build_snapshot_dataset(
            record, return_kept_indices=True, on_unanchorable="skip",
        )
    except ValueError as exc:
        assert False, f"explicit skip rejected the valid companion trial: {exc}"
    assert kept == [1]
    assert snapshots.shape == (1, 5) and labels.shape == (1,)
    counts = _audit_snapshot_drops(record, kept)
    assert counts["n_dropped"] == 1
    assert counts["dropped_by_class"] == {invalid["label"].name: 1}
    assert counts["dropped_by_condition"] == {"wind_only": 1}


def test_prior_argmax_0611_stays_descriptive():
    oof, served = _make_controlled_agreement_priors(1000, 611)
    try:
        record = audit_prior_train_serve_shift(oof, served, CLASS_NAMES)
    except ValueError as exc:
        assert False, f"valid 0.611 prior shift was gated: {exc}"
    assert record["n_trials"] == 1000
    assert record["argmax_agreement"] == 0.611
    assert record["argmax_agreement_is_descriptive_only"] is True
    expected = float(np.mean(np.abs(served - oof).sum(axis=1)) / 2.0)
    assert record["mean_total_variation_distance"] == pytest.approx(expected)


def test_prior_vector_tv_stays_descriptive():
    confident = np.tile([0.97, 0.01, 0.01, 0.01], (128, 1))
    diffuse = np.tile([0.40, 0.20, 0.20, 0.20], (128, 1))
    try:
        record = audit_prior_train_serve_shift(confident, diffuse, CLASS_NAMES)
    except ValueError as exc:
        assert False, f"valid vector shift was gated: {exc}"
    assert record["argmax_agreement"] == 1.0
    assert record["mean_total_variation_distance"] == pytest.approx(0.57)
    assert record["ks_is_descriptive_only"] is True

@pytest.mark.parametrize("producer", ["prepare_data.py", "prepare_metadata.py"])
def test_production_snapshot_call_rejects_unanchorable_visual(tmp_path, producer):
    """Run each actual ETL Snapshot call with a valid and an invalid visual trial."""
    import ast
    from pathlib import Path
    from nsmor.config import DEFAULT_FEATURE, DEFAULT_TIME_WINDOW

    kin, evt = _make_visual_only_csvs(tmp_path, n_trials=1)
    trial = extract_trial_data(load_and_concat_sessions([kin], [evt]), "visual_session", 0)
    good = assign_ground_truth_labels([trial])[0]
    bad_trial = dict(trial, l_v_ratio=np.zeros_like(trial["l_v_ratio"]))
    bad = dict(good, trial_data=bad_trial, session_id="visual_bad", trial_id=1)
    src = Path(__file__).resolve().parents[1] / "scripts" / producer
    nodes = [node for node in ast.walk(ast.parse(src.read_text(encoding="utf-8")))
             if isinstance(node, ast.Assign) and any(
                 isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                 and call.func.id == "build_snapshot_dataset" for call in ast.walk(node.value))]
    assert len(nodes) == 1, producer
    scope = {"labeled_trials": [good, bad], "time_config": DEFAULT_TIME_WINDOW,
             "feature_config": DEFAULT_FEATURE, "build_snapshot_dataset": build_snapshot_dataset}
    statement = compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                        str(src), "exec")
    with pytest.raises(ValueError, match=r"visual_bad.*could not be anchored"):
        exec(statement, scope)
