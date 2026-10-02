"""Comprehensive regression tests for downstream analysis contracts.

Validates:
1. Symptom A: ETL anchor_frames describe SAVED arrays after lazy cropping.
2. Symptom B: TTC=0 selection uses declared target_ttc_ms joined by (session_id, trial_id),
   not static crop anchor or physical wind onset heuristic.
3. Symptom C: Early/late anchor reference coordinates in metric extraction; zero/flat/nonfinite
   responses produce NaN rather than fabricated latencies; dt calibration.
4. Differing raw/dataset ordering and skipped events still match exact IDs.
5. Multiple sessions in flat root correctly disambiguated.
6. Nonzero declared TTCs (-373ms, -119ms, +200ms) strictly excluded from TTC0.
7. Gate trajectory alignment with heterogeneous anchors.
8. CLI contracts (--dt_ms inherits checkpoint, --raw_dir).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION, FeatureConfig
from nsmor.pipeline.events import (
    load_declared_events_index,
    parse_events_file,
)
from scripts.analyze_integration import (
    build_parser as build_integration_parser,
    classify_wind_condition,
    compute_condition_statistics,
    extract_predicted_metrics,
    group_trials_by_condition,
)
from scripts.convert_metadata_to_etl import (
    populate_etl_provenance_and_conditions,
    resolve_legacy_pure_wind_prepend_frames,
)
from scripts.simulate_psychophysics import (
    extract_gate_trajectory,
    extract_latency_to_peak,
    extract_peak_velocity,
    find_multisensory_ttc0,
    psychophysics_gate_verdict,
)
from scripts.analyze_integration import (
    parse_ttc_condition_delta,
    format_ttc_condition_name,
)


# =========================================================================
# 1. Symptom A: Anchor Frames in ETL Describe SAVED Arrays
# =========================================================================

def test_repro_symptom_a_uncropped_anchor_in_etl():
    """Symptom A: populate_etl_provenance_and_conditions must update anchor_frames to describe SAVED arrays."""
    raw_anchor = 2000
    crop_start = 800
    saved_anchor = raw_anchor - crop_start  # 1200

    X_saved = np.zeros((2400, 8), dtype=np.float32)
    # Put visual and wind stimulus at saved_anchor
    X_saved[saved_anchor:saved_anchor + 100, 0] = np.linspace(2.0, 90.0, 100)
    X_saved[saved_anchor:saved_anchor + 50, 1] = 1.0

    metadata = {
        "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
        "anchor_frames": [raw_anchor],  # 2000 (uncropped)
        "stimulus_conditions": ["multisensory"],
        "is_pure_wind": np.array([False]),
        "trial_specs": [
            {
                "session_id": "sess_1",
                "trial_id": 1,
                "anchor_frame": raw_anchor,
                "stimulus_condition": "multisensory",
                "is_pure_wind": False,
            }
        ],
    }

    output = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "X_seqs": [X_saved],
        "Y_seqs": [np.zeros(2400, dtype=np.float32)],
    }

    populate_etl_provenance_and_conditions(
        output, metadata, X_seqs=[X_saved], lengths=[2400]
    )

    # In saved X_saved (len 2400), anchor MUST be 1200, NOT 2000!
    assert output["anchor_frames"][0] == saved_anchor, (
        f"Expected saved anchor {saved_anchor}, got {output['anchor_frames'][0]}"
    )


def test_silent_trial_anchor_does_not_leak_raw_coordinate():
    """Silent/no-stimulus trial anchor must not use raw (uncropped) coordinate."""
    # Simulate: 5000-frame trial cropped to 2400 frames starting at raw frame 800.
    # Raw anchor = 2000 (in uncropped coordinates).
    # Saved anchor in cropped coordinates would be 2000 - 800 = 1200.
    # Old buggy code: raw_a=2000 < len(x_seq)=2400 → leaks 2000 (wrong).
    # Fixed code: silent trial → use der_anchor (0), no raw coordinate used.
    raw_anchor = 2000

    X_silent = np.zeros((2400, 8), dtype=np.float32)  # Fully silent: no vis, no wind

    metadata = {
        "anchor_frames": [raw_anchor],  # 2000 (uncropped raw coordinate)
        "stimulus_conditions": ["no_stimulus"],
        "is_pure_wind": np.array([False]),
    }
    output: Dict[str, Any] = {}
    populate_etl_provenance_and_conditions(
        output, metadata, X_seqs=[X_silent], lengths=[2400]
    )

    # Must NOT be 2000 (raw uncropped coordinate)
    assert output["anchor_frames"][0] != raw_anchor, (
        f"Silent trial must not leak raw anchor {raw_anchor} into saved coordinates"
    )
    # Must be 0 (physical derivation default for silent channels)
    assert output["anchor_frames"][0] == 0, (
        f"Silent trial anchor must be 0 (physical default), got {output['anchor_frames'][0]}"
    )


# =========================================================================
# 2. Symptom B: TTC=0 Selection Uses Declared Target, Not Crop Anchor
# =========================================================================

def test_repro_symptom_b_false_ttc0_selection():
    """Symptom B: Fallback wind_onset vs stim_onset_frame=1200 must NOT classify non-TTC0 trials as TTC0."""
    B = 2
    T = 2400
    dt_ms = 4.0
    stim_onset = 1200

    X_seqs = torch.zeros(B, T, 8, dtype=torch.float32)
    # Both trials have wind at 1200:1500 (crop anchor)
    X_seqs[0, 1200:1500, 1] = 1.0
    X_seqs[1, 1200:1500, 1] = 1.0
    # Trial 0 has visual peak at 1300
    X_seqs[0, 1300, 0] = 50.0
    # Trial 1 has visual peak at 1150
    X_seqs[1, 1150, 0] = 50.0

    lengths = torch.tensor([T, T], dtype=torch.int64)

    # With raw_dir missing and no declared events, it must fail closed (raise ValueError)
    with pytest.raises(ValueError, match="No multisensory TTC=0ms"):
        find_multisensory_ttc0(
            X_seqs,
            lengths,
            raw_dir="nonexistent_raw_dir",
            stim_onset_frame=stim_onset,
            dt_ms=dt_ms,
        )


def test_declared_nonzero_ttc_excluded_even_with_anchor_at_1200():
    """Declared nonzero TTCs (-373, -119, +200) are excluded; exact 0 is included."""
    B = 4
    T = 2400
    dt_ms = 4.0

    X_seqs = torch.zeros(B, T, 8, dtype=torch.float32)
    # ALL trials have wind at 1200 (crop anchor) and visual angle present
    for i in range(B):
        X_seqs[i, :, 0] = 20.0
        X_seqs[i, 1200:1300, 1] = 1.0

    lengths = torch.tensor([T, T, T, T], dtype=torch.int64)

    # Declared target TTCs
    target_ttcs = [-373.0, -119.0, 0.0, 200.0]
    conditions = ["multisensory"] * B

    selection = find_multisensory_ttc0(
        X_seqs,
        lengths,
        target_ttc_ms=target_ttcs,
        stimulus_conditions=conditions,
        dt_ms=dt_ms,
    )
    mask = selection.mask

    # Only trial 2 (target_ttc_ms == 0.0) must be True
    assert mask[0].item() is False, "-373ms must not be selected as TTC0"
    assert mask[1].item() is False, "-119ms must not be selected as TTC0"
    assert mask[2].item() is True, "0.0ms must be selected as TTC0"
    assert mask[3].item() is False, "+200ms must not be selected as TTC0"


# =========================================================================
# 3. Symptom C: Per-Trial Anchor Reference & Flat Response Handling
# =========================================================================

def test_repro_symptom_c_early_anchor_metrics_and_flat_response():
    """Symptom C: Early anchor (300) with peak at 350 gives +200ms at 4ms; flat response gives NaN."""
    dt_ms = 4.0
    real_anchor = 300
    peak_frame = 350

    y_pred = np.zeros(2000, dtype=np.float32)
    y_pred[peak_frame] = 50.0

    # With per-trial anchor_frames
    metrics = extract_predicted_metrics(
        y_preds=[y_pred],
        trial_indices=[0],
        dt_ms=dt_ms,
        anchor_frames=[real_anchor],
    )

    expected_latency = (peak_frame - real_anchor) * dt_ms  # (350 - 300) * 4.0 = 200.0 ms
    assert np.isclose(metrics["latencies"][0], expected_latency)
    assert metrics["peak_velocities"][0] == 50.0

    # Flat response: all zeros
    y_flat = np.zeros(2000, dtype=np.float32)
    metrics_flat = extract_predicted_metrics(
        y_preds=[y_flat],
        trial_indices=[0],
        dt_ms=dt_ms,
        anchor_frames=[real_anchor],
    )
    assert np.isnan(metrics_flat["latencies"][0]), "Flat response must produce NaN latency"
    assert metrics_flat["peak_velocities"][0] == 0.0


def test_metrics_calibrated_dt_and_boundary_clipping():
    """Metrics calculation honors non-default dt and handles late-anchor boundary clipping."""
    # Calibrated dt: 2.5 ms
    dt_calibrated = 2.5
    real_anchor = 400
    peak_frame = 450  # 50 frames later = 125.0 ms at 2.5 ms/frame

    y_pred = np.zeros(2400, dtype=np.float32)
    y_pred[peak_frame] = 30.0

    metrics = extract_predicted_metrics(
        y_preds=[y_pred],
        trial_indices=[0],
        dt_ms=dt_calibrated,
        anchor_frames=[real_anchor],
    )
    assert np.isclose(metrics["latencies"][0], 125.0)

    # Late anchor near sequence end (anchor=2300, len=2400)
    # Search window [1800, 2400], peak at 2350
    y_late = np.zeros(2400, dtype=np.float32)
    y_late[2350] = 60.0
    metrics_late = extract_predicted_metrics(
        y_preds=[y_late],
        trial_indices=[0],
        dt_ms=4.0,
        anchor_frames=[2300],
    )
    assert np.isclose(metrics_late["latencies"][0], (2350 - 2300) * 4.0)  # 200.0 ms
    assert metrics_late["peak_velocities"][0] == 60.0

    # Non-finite values in window
    y_nan = np.zeros(1000, dtype=np.float32)
    y_nan[500] = float("nan")
    metrics_nan = extract_predicted_metrics(
        y_preds=[y_nan],
        trial_indices=[0],
        dt_ms=4.0,
        anchor_frames=[500],
    )
    assert np.isnan(metrics_nan["latencies"][0])


def test_compute_condition_statistics_filters_nan():
    """compute_condition_statistics properly filters NaN and reports n_excluded."""
    metrics = {
        "latencies": [100.0, 200.0, float("nan"), 300.0],
        "peak_velocities": [10.0, 20.0, 0.0, 30.0],
    }
    stats = compute_condition_statistics(metrics)
    assert stats["latency"]["n"] == 3
    assert np.isclose(stats["latency"]["mean"], 200.0)


# =========================================================================
# 4. Identity Resolution: Differing Ordering, Skipped Rows, Conflicting Duplicates
# =========================================================================

def test_events_parsing_differing_order_and_skipped_events(tmp_path: Path):
    """Declared events index correctly joins by (session_id, trial_id) despite reordering."""
    session_id = "0.513cricket_001_20260707_193143_session_1"
    evt_file = tmp_path / f"{session_id}_events.csv"

    # Raw schema CSV with scrambled trial order: trial 3, trial 1, trial 2
    # Plus unrelated events: initial_baseline, phase_transition, trial_stop
    content = (
        "event_name,timestamp,session_num,trial_in_session,global_trial_id,details\n"
        'initial_baseline_start,1.0,1,0,36,"{""duration"": 2.0}"\n'
        f'trial_start,10.0,1,3,39,"{{""type"": ""looming_wind"", ""target_ttc_ms"": -119, ""lv_ratio_ms"": 120}}"\n'
        'phase_transition,10.1,1,3,39,"{""from_phase"": ""TrialStart""}"\n'
        f'trial_start,20.0,1,1,37,"{{""type"": ""looming_wind"", ""target_ttc_ms"": 0, ""lv_ratio_ms"": 120}}"\n'
        f'trial_start,30.0,1,2,38,"{{""type"": ""baseline_visual"", ""target_ttc_ms"": null, ""lv_ratio_ms"": 120}}"\n'
        "trial_stop,35.0,1,2,38,\n"
    )
    evt_file.write_text(content, encoding="utf-8")

    index = parse_events_file(evt_file)
    assert len(index) == 3

    # Check trial 1 (global_trial_id = 37)
    t1 = index[(session_id, 37)]
    assert t1["type"] == "looming_wind"
    assert t1["target_ttc_ms"] == 0.0

    # Check trial 2 (global_trial_id = 38)
    t2 = index[(session_id, 38)]
    assert t2["type"] == "baseline_visual"
    assert t2["target_ttc_ms"] is None

    # Check trial 3 (global_trial_id = 39)
    t3 = index[(session_id, 39)]
    assert t3["type"] == "looming_wind"
    assert t3["target_ttc_ms"] == -119.0


def test_events_parsing_rejects_conflicting_duplicates(tmp_path: Path):
    """Conflicting duplicate trial_start rows for the same trial_id must raise ValueError."""
    session_id = "test_sess"
    evt_file = tmp_path / f"{session_id}_events.csv"
    content = (
        "event_name,timestamp,session_num,trial_in_session,global_trial_id,details\n"
        'trial_start,10.0,1,1,1,"{""type"": ""looming_wind"", ""target_ttc_ms"": 0}"\n'
        'trial_start,15.0,1,1,1,"{""type"": ""looming_wind"", ""target_ttc_ms"": -373}"\n'
    )
    evt_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Conflicting duplicate trial_start"):
        parse_events_file(evt_file)


@pytest.mark.parametrize("across_files", [False, True], ids=["within-file", "across-files"])
@pytest.mark.parametrize(
    "changed_details",
    [
        {"lv_ratio_ms": 121},
        {"wind_dir": "left"},
        {"loom_speed": 0.6},
    ],
    ids=["lv-ratio", "wind-direction", "additional-detail"],
)
def test_events_reject_same_key_conflicting_declared_details(
    tmp_path: Path, across_files: bool, changed_details: Dict[str, Any]
):
    """Equal type/TTC cannot mask changes to other trial_start declarations."""
    base = {
        "type": "looming_wind",
        "target_ttc_ms": 0,
        "lv_ratio_ms": 120,
        "wind_dir": "right",
        "loom_speed": 0.5,
    }
    changed = {**base, **changed_details}

    def row(details: Dict[str, Any]) -> str:
        value = json.dumps(details).replace('"', '""')
        return f'session_0,1,0.0,trial_start,"{value}"\n'

    header = "session_id,trial_id,time_ms,event_type,event_value\n"
    first = tmp_path / "a_events.csv"
    first.write_text(header + row(base) + ("" if across_files else row(changed)), encoding="utf-8")
    if across_files:
        (tmp_path / "b_events.csv").write_text(header + row(changed), encoding="utf-8")

    message = "Conflicting duplicate trial_start across files" if across_files else "Conflicting duplicate trial_start"
    with pytest.raises(ValueError, match=message):
        if across_files:
            load_declared_events_index(tmp_path)
        else:
            parse_events_file(first)


@pytest.mark.parametrize("across_files", [False, True], ids=["within-file", "across-files"])
def test_events_accept_identical_declared_duplicates(tmp_path: Path, across_files: bool):
    """Repeated declarations with the same parsed entry remain idempotent."""
    row = 'session_0,1,0.0,trial_start,"{""type"": ""looming_wind"", ""target_ttc_ms"": 0, ""lv_ratio_ms"": 120}"\n'
    header = "session_id,trial_id,time_ms,event_type,event_value\n"
    first = tmp_path / "a_events.csv"
    first.write_text(header + row + ("" if across_files else row), encoding="utf-8")
    if across_files:
        (tmp_path / "b_events.csv").write_text(header + row, encoding="utf-8")
    trials = load_declared_events_index(tmp_path) if across_files else parse_events_file(first)
    assert len(trials) == 1
    assert trials[("session_0", 1)]["lv_ratio_ms"] == 120.0

# =========================================================================
# 5. Multiple Sessions in Flat Root
# =========================================================================

def test_multiple_sessions_in_flat_root(tmp_path: Path):
    """Flat raw directory with multiple session event files joins accurately by session_id."""
    sess1 = "0.513cricket_001_session_1"
    sess2 = "0.513cricket_001_session_2"

    csv1 = tmp_path / f"{sess1}_events.csv"
    csv2 = tmp_path / f"{sess2}_events.csv"

    csv1.write_text(
        "event_name,timestamp,session_num,trial_in_session,global_trial_id,details\n"
        'trial_start,10.0,1,1,1,"{""type"": ""looming_wind"", ""target_ttc_ms"": 0}"\n',
        encoding="utf-8",
    )
    csv2.write_text(
        "event_name,timestamp,session_num,trial_in_session,global_trial_id,details\n"
        'trial_start,10.0,2,1,20,"{""type"": ""looming_wind"", ""target_ttc_ms"": -261}"\n',
        encoding="utf-8",
    )

    merged = load_declared_events_index(tmp_path)
    assert len(merged) == 2
    assert (sess1, 1) in merged
    assert merged[(sess1, 1)]["target_ttc_ms"] == 0.0
    assert (sess2, 20) in merged
    assert merged[(sess2, 20)]["target_ttc_ms"] == -261.0


# =========================================================================
# 6. Gate Trajectory Alignment with Heterogeneous Anchors
# =========================================================================

def test_extract_gate_trajectory_heterogeneous_anchors():
    """Gate trajectory shifts each trial so its anchor aligns to target_anchor."""
    B = 2
    T = 2000
    target_anchor = 1200
    anchor_0 = 1000
    anchor_1 = 1200

    g_gru = torch.zeros(B, T)
    # Trial 0 has gate peak 100 frames after its anchor (frame 1100)
    g_gru[0, anchor_0 + 100] = 1.0
    # Trial 1 has gate peak 100 frames after its anchor (frame 1300)
    g_gru[1, anchor_1 + 100] = 1.0

    lengths = torch.tensor([T, T], dtype=torch.int64)
    internals = {"routing_gates": torch.stack([1.0 - g_gru, g_gru], dim=-1)}

    aligned_mean = extract_gate_trajectory(
        internals,
        lengths,
        anchor_frames=[anchor_0, anchor_1],
        target_anchor=target_anchor,
    )

    # When aligned, both peaks should coincide at target_anchor + 100 = 1300
    assert np.isclose(aligned_mean[target_anchor + 100], 1.0), (
        "Heterogeneous gate peaks did not align at target_anchor + 100"
    )


# =========================================================================
# 7. Integration Grouping & CLI Contract
# =========================================================================

def test_group_trials_by_condition_unknown_ttc_no_inferred_ttc0():
    """Multisensory trials with missing target_ttc_ms map to multisensory_other, not TTC0."""
    X_dummy = [np.zeros((100, 8), dtype=np.float32)]
    trial_info = [{"type": "multisensory", "target_ttc_ms": None}]

    groups = group_trials_by_condition(X_dummy, trial_info)
    assert "multisensory_other" in groups
    assert "multisensory_ttc_0ms" not in groups


def test_legitimate_null_ttc_preserved_for_unimodal(tmp_path: Path):
    """Unimodal trials (baseline_visual, baseline_wind) have legitimate target_ttc_ms=null; preserved as None, not rejected or zero-filled."""
    session_id = "test_unimodal_sess"
    evt_file = tmp_path / f"{session_id}_events.csv"
    content = (
        "event_name,timestamp,session_num,trial_in_session,global_trial_id,details\n"
        'trial_start,10.0,1,1,1,"{""type"": ""baseline_visual"", ""target_ttc_ms"": null, ""lv_ratio_ms"": 120}"\n'
        'trial_start,20.0,1,2,2,"{""type"": ""baseline_wind"", ""target_ttc_ms"": null}"\n'
    )
    evt_file.write_text(content, encoding="utf-8")

    index = parse_events_file(evt_file)
    assert len(index) == 2
    assert index[(session_id, 1)]["target_ttc_ms"] is None
    assert index[(session_id, 1)]["type"] == "baseline_visual"
    assert index[(session_id, 2)]["target_ttc_ms"] is None
    assert index[(session_id, 2)]["type"] == "baseline_wind"


def test_legacy_dataset_lacking_trial_ids_fails_closed():
    """When a dataset lacks both target_ttc_ms and trial_ids, find_multisensory_ttc0 fails closed."""
    B = 3
    T = 2400
    X_seqs = torch.zeros(B, T, 8, dtype=torch.float32)
    lengths = torch.full((B,), T, dtype=torch.int64)

    # Legacy dataset with only session_ids, no trial_ids, no target_ttc_ms
    with pytest.raises(ValueError, match="lacks trial_ids and target_ttc_ms"):
        find_multisensory_ttc0(
            X_seqs,
            lengths,
            session_ids=["s1", "s2", "s3"],
            trial_ids=None,
            target_ttc_ms=None,
        )


def test_analyze_integration_cli_parser_defaults():
    """analyze_integration.py inherits saved cadence and includes --raw_dir."""
    parser = build_integration_parser()
    dt_action = next(a for a in parser._actions if "--dt_ms" in a.option_strings)
    assert dt_action.default is None
    assert parser.parse_args(["--checkpoint", "unused.pth"]).dt_ms is None
    # Verify --raw_dir argument exists and defaults to None (no NameError in main)
    raw_dir_action = next(
        (a for a in parser._actions if "--raw_dir" in a.option_strings), None
    )
    assert raw_dir_action is not None, "--raw_dir argument missing from build_parser()"
    assert raw_dir_action.default is None, (
        f"Expected default raw_dir=None, got {raw_dir_action.default}"
    )


# =========================================================================
# 8. Not-Applicable TTC0 vs Fail-Closed, Stale Artefacts, Dynamic Groups
# =========================================================================

def test_ttc0_valid_empty_declared_subset_not_applicable(tmp_path: Path):
    """Complete valid declared metadata with ZERO TTC=0 -> empty mask + status."""
    B = 3
    T = 2400
    X_seqs = torch.zeros(B, T, 8, dtype=torch.float32)
    X_seqs[:, :, 0] = 20.0
    lengths = torch.full((B,), T, dtype=torch.int64)
    target_ttcs = [-225.0, -261.0, -308.0]
    conditions = ["multisensory"] * B

    selection = find_multisensory_ttc0(
        X_seqs,
        lengths,
        target_ttc_ms=target_ttcs,
        stimulus_conditions=conditions,
        dt_ms=4.0,
    )
    assert selection.status == "not_applicable"
    assert selection.n_ttc0 == 0
    assert selection.n_candidates == B
    assert selection.mask.sum().item() == 0
    assert not selection.mask.any()


def test_ttc0_missing_ids_raise_not_applicable():
    """Missing identity with required raw lookup fails closed (raises)."""
    B = 2
    T = 2400
    X_seqs = torch.zeros(B, T, 8, dtype=torch.float32)
    X_seqs[:, :, 0] = 20.0
    lengths = torch.full((B,), T, dtype=torch.int64)

    with pytest.raises(ValueError):
        find_multisensory_ttc0(
            X_seqs,
            lengths,
            raw_dir="/nonexistent",
            target_ttc_ms=[None, None],
            session_ids=None,
            trial_ids=None,
            stimulus_conditions=["multisensory", "multisensory"],
            dt_ms=4.0,
        )


def test_ttc0_invalid_corrupt_ttc_string_fails_closed(tmp_path: Path):
    """Corrupt numeric TTC string in events raises — never coerced to None."""
    session_id = "sess_corrupt"
    evt_file = tmp_path / f"{session_id}_events.csv"
    content = (
        "event_name,timestamp,session_num,trial_in_session,global_trial_id,details\n"
        'trial_start,10.0,1,1,1,"{""type"": ""multisensory"", ""target_ttc_ms"": ""not-a-number""}"\n'
    )
    evt_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupt target_ttc_ms"):
        parse_events_file(evt_file)


def test_stale_png_summary_cannot_satisfy_gate(tmp_path: Path):
    """Stale PNG/summary with no fresh status artefact never passes the gate."""
    out = tmp_path / "results"
    out.mkdir()
    (out / "bayesian_reliability.png").write_bytes(b"\x89PNG stale")
    (out / "psychophysics_summary.json").write_text('{"stale": true}', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing fresh status artefact"):
        psychophysics_gate_verdict(out)

    # Fresh status written by the CURRENT run is what the gate reads.
    (out / "bayesian_reliability.json").write_text(
        '{"status": "not_applicable", "n_ttc0": 0}', encoding="utf-8",
    )
    assert psychophysics_gate_verdict(out) == "not_applicable"


def test_stale_not_applicable_status_cannot_hide_new_ok_run(tmp_path: Path):
    """If current run is ok, gate demands PNG + summary despite leftover status."""
    out = tmp_path / "results2"
    out.mkdir()
    (out / "bayesian_reliability.json").write_text(
        '{"status": "ok"}', encoding="utf-8",
    )
    # Verdict is ok (not not_applicable) — runner then requires PNG+summary.
    assert psychophysics_gate_verdict(out) == "ok"
    (out / "bayesian_reliability.json").write_text(
        '{"status": "garbled"}', encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown psychophysics status"):
        psychophysics_gate_verdict(out)


# =========================================================================
# 9. Dynamic Condition Groups: -225 / -261 / -308 Not Silently Dropped
# =========================================================================

def test_dynamic_ttc_condition_names_and_deltas():
    """format/parse round-trips dynamic TTC values including -225/-261/-308."""
    assert format_ttc_condition_name(0.0) == "multisensory_ttc_0ms"
    assert format_ttc_condition_name(-225.0) == "multisensory_ttc_-225ms"
    assert parse_ttc_condition_delta("multisensory_ttc_-225ms") == -225.0
    assert parse_ttc_condition_delta("multisensory_ttc_-261ms") == -261.0
    assert parse_ttc_condition_delta("multisensory_ttc_-308ms") == -308.0
    assert parse_ttc_condition_delta("multisensory_ttc_+200ms") == 200.0
    assert parse_ttc_condition_delta("multisensory_ttc_0ms") == 0.0
    assert parse_ttc_condition_delta("visual_only") is None
    assert parse_ttc_condition_delta("wind_only") is None
    assert parse_ttc_condition_delta("multisensory_other") is None


def test_export_summary_includes_dynamic_conditions(tmp_path: Path):
    """export_integration_summary writes every observed condition with real delta_t."""
    from scripts.analyze_integration import export_integration_summary

    condition_stats = {
        "multisensory_ttc_-225ms": {
            "latency": {"mean": 10.0, "sem": 1.0, "n": 3},
            "peak_velocity": {"mean": 50.0, "sem": 2.0, "n": 3},
        },
        "multisensory_ttc_-261ms": {
            "latency": {"mean": 11.0, "sem": 1.0, "n": 3},
            "peak_velocity": {"mean": 51.0, "sem": 2.0, "n": 3},
        },
        "multisensory_ttc_-308ms": {
            "latency": {"mean": 12.0, "sem": 1.0, "n": 3},
            "peak_velocity": {"mean": 52.0, "sem": 2.0, "n": 3},
        },
    }
    out_path = tmp_path / "integration_summary.json"
    export_integration_summary(condition_stats, out_path, dt_ms=4.0)
    with open(out_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)

    for name, delta in (
        ("multisensory_ttc_-225ms", -225.0),
        ("multisensory_ttc_-261ms", -261.0),
        ("multisensory_ttc_-308ms", -308.0),
    ):
        assert name in summary["conditions"], f"{name} missing from summary"
        assert summary["conditions"][name]["delta_t_ms"] == pytest.approx(delta)
    assert summary["dt_ms"] == pytest.approx(4.0)


def test_classify_wind_condition_keeps_no_stimulus_separate():
    """All-zero sequences classify as no_stimulus, never visual_only."""
    x = np.zeros((100, 8), dtype=np.float32)
    cond, delta = classify_wind_condition(x, dt_ms=4.0, stim_onset_frame=0)
    assert cond == "no_stimulus"
    assert delta is None


def test_classify_wind_condition_never_assigns_ttc0_from_wind_onset():
    """Kinematics-only fallback must not map wind onset to declared TTC0."""
    x = np.zeros((100, 8), dtype=np.float32)
    x[:, 0] = 10.0          # visual present
    x[50:60, 1] = 1.0       # wind exactly at stim_onset
    cond, delta = classify_wind_condition(x, dt_ms=4.0, stim_onset_frame=50)
    assert cond != "multisensory_ttc_0ms"
    assert cond == "multisensory_other"
    assert delta is not None


def test_visual_only_not_confused_with_all_zero():
    """Visual-only (visual present, no wind) stays visual_only; zero stays no_stimulus."""
    x_vis = np.zeros((100, 8), dtype=np.float32)
    x_vis[:, 0] = 10.0
    cond, _ = classify_wind_condition(x_vis, dt_ms=4.0, stim_onset_frame=0)
    assert cond == "visual_only"

    x_zero = np.zeros((100, 8), dtype=np.float32)
    cond2, _ = classify_wind_condition(x_zero, dt_ms=4.0, stim_onset_frame=0)
    assert cond2 == "no_stimulus"


# =========================================================================
# 10. Corrupt TTC String, Nondefault dt, Legacy Migration
# =========================================================================

def test_corrupt_ttc_string_never_becomes_none_in_parse():
    """parse_events_file raises on corrupt numeric TTC strings."""
    import tempfile

    session_id = "sess_bad"
    with tempfile.TemporaryDirectory() as td:
        evt_file = Path(td) / f"{session_id}_events.csv"
        content = (
            "event_name,timestamp,session_num,trial_in_session,global_trial_id,details\n"
            'trial_start,10.0,1,1,1,"{""type"": ""multisensory"", ""target_ttc_ms"": ""abc""}"\n'
        )
        evt_file.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=r"Corrupt target_ttc_ms \(unparseable\)"):
            parse_events_file(evt_file)


def test_nondefault_dt_and_pre_anchor_in_latency():
    """extract_latency_to_peak honors explicit stim_onset_frame / anchor_frames."""
    dt_ms = 10.0
    stim_onset = 300
    anchor_frames = [350]
    peak_frame = 400

    Y_pred = torch.zeros(1, 2000, dtype=torch.float32)
    Y_pred[0, peak_frame] = 80.0
    lengths = torch.tensor([2000], dtype=torch.int64)

    lats = extract_latency_to_peak(
        Y_pred,
        lengths,
        dt_ms=dt_ms,
        stim_onset_frame=stim_onset,
        anchor_frames=anchor_frames,
    )
    expected = float((peak_frame - anchor_frames[0]) * dt_ms)
    assert lats[0] == pytest.approx(expected, abs=1e-3)
    assert expected == pytest.approx(500.0)


def test_legacy_pure_wind_prepend_uses_explicit_dt():
    """Legacy pure-wind prepend uses effective dt; unknown dt fails closed."""
    spec = {"is_pure_wind": True, "pure_wind_prepended_frames": 0, "anchor_frame": 10}

    # Explicit dt
    prepend = resolve_legacy_pure_wind_prepend_frames(spec, dt_ms=4.0)
    assert prepend == int(round(5700.0 / 4.0))

    prepend10 = resolve_legacy_pure_wind_prepend_frames(spec, dt_ms=10.0)
    assert prepend10 == int(round(5700.0 / 10.0))
    assert prepend10 != prepend

    # Unknown dt: fail closed (explicit limitation, not hardcoded 4.0)
    with pytest.raises(ValueError, match="dt_ms is unknown"):
        resolve_legacy_pure_wind_prepend_frames(spec, dt_ms=None)

    # Already recorded prepend is reused without dt.
    spec_rec = {"is_pure_wind": True, "pure_wind_prepended_frames": 42}
    assert resolve_legacy_pure_wind_prepend_frames(spec_rec, dt_ms=None) == 42


# =========================================================================
# 11. Runner Dry-Run: raw_dir + dt_ms + Mismatch Guard
# =========================================================================

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_run_pipeline_dry_run_passes_raw_dir_and_dt_ms(tmp_path: Path):
    """DRY_RUN=1 plan shows --raw_dir and --dt_ms on downstream stages."""
    import subprocess

    root = _repo_root()
    script = root / "run_pipeline.sh"
    if not script.is_file():
        pytest.skip("run_pipeline.sh not present")
    env = {
        "HOME": str(tmp_path),
        "OUTPUT_DIR": str(tmp_path / "results"),
        "RUN_DIR": str(tmp_path / "runs"),
        "DATASET": str(tmp_path / "dataset.pt"),
        "CONFIG": str(root / "config" / "default.yaml"),
        "DRY_RUN": "1",
        "RAW_DIR": str(tmp_path / "raw"),
        "DT_MS": "4.0",
    }
    # Merge with process env so bash/python resolve.
    import os
    full_env = {**os.environ, **env}
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=str(root),
        env=full_env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, f"run_pipeline.sh dry-run failed:\n{combined}"
    assert "--raw_dir" in combined, "dry-run plan missing --raw_dir"
    assert "--dt_ms" in combined, "dry-run plan missing --dt_ms"


def test_run_pipeline_dry_run_dt_mismatch_guard(tmp_path: Path):
    """DRY_RUN=1 aborts when DT_MS contradicts config model.dt_ms."""
    import os
    import subprocess

    root = _repo_root()
    script = root / "run_pipeline.sh"
    if not script.is_file():
        pytest.skip("run_pipeline.sh not present")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("model:\n  dt_ms: 4.0\n", encoding="utf-8")

    env = {
        **os.environ,
        "OUTPUT_DIR": str(tmp_path / "results"),
        "RUN_DIR": str(tmp_path / "runs"),
        "DATASET": str(tmp_path / "dataset.pt"),
        "CONFIG": str(cfg_path),
        "DRY_RUN": "1",
        "RAW_DIR": str(tmp_path / "raw"),
        "DT_MS": "10.0",
    }
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode != 0, f"expected dt mismatch abort, got:\n{combined}"
    assert "contradicts" in combined.lower() or "dt_ms" in combined.lower()


# =========================================================================
# 12. MAJOR F1: Gate Alignment Origin for Nondefault pre_anchor_frames
# =========================================================================

def test_gate_alignment_uses_meta_stim_onset_for_nondefault_pre_anchor():
    """Gate trajectory alignment origin must match meta.stim_onset_frame (F1).

    With --pre_anchor_frames=600 the stimulus lands at frame 600 in each
    saved sequence.  ``extract_gate_trajectory`` must be called with
    ``target_anchor=600`` so the aligned peak sits at index 600 and the
    figure time axis ``time_ms = (arange(T) - 600) * dt`` labels it 0 ms.
    The old hardcoded default of 1200 shifted the peak to index 1200
    (+2400 ms) and folded 2400 ms of pre-stimulus samples into
    ``gate_post_stim_mean``.
    """
    dt_ms = 4.0
    pre_anchor = 600  # nondefault --pre_anchor_frames
    T = 2400
    B = 1

    # Gate peak exactly at the anchor (stimulus onset).
    g_gru = torch.zeros(B, T)
    g_gru[0, pre_anchor] = 1.0
    lengths = torch.tensor([T], dtype=torch.int64)
    internals = {"routing_gates": torch.stack([1.0 - g_gru, g_gru], dim=-1)}

    # Correct call: target_anchor = meta.stim_onset_frame = pre_anchor.
    aligned = extract_gate_trajectory(
        internals, lengths,
        anchor_frames=[pre_anchor],
        target_anchor=pre_anchor,
    )

    # Peak must land at index pre_anchor (= 600).
    assert np.isclose(aligned[pre_anchor], 1.0), (
        f"Gate peak expected at index {pre_anchor}, got argmax {int(np.nanargmax(aligned))}"
    )

    # Time axis derived from meta.stim_onset_frame must label peak as 0 ms.
    time_ms = (np.arange(T) - pre_anchor) * dt_ms
    assert time_ms[pre_anchor] == pytest.approx(0.0), (
        f"Stimulus onset must be 0 ms on the time axis, got {time_ms[pre_anchor]}"
    )

    # Post-stim split must start at pre_anchor, not the old hardcoded 1200.
    post_stim = aligned[pre_anchor:]
    pre_stim = aligned[:pre_anchor]
    assert np.nanmax(pre_stim) < 0.5, (
        "Pre-stimulus window must not contain the aligned gate peak"
    )
    assert np.isclose(post_stim[0], 1.0), (
        "Post-stimulus window must start at the gate peak"
    )

    # Demonstrate the old defect: target_anchor=1200 misplaces the peak.
    misaligned = extract_gate_trajectory(
        internals, lengths,
        anchor_frames=[pre_anchor],
        target_anchor=1200,  # old hardcoded default
    )
    assert int(np.nanargmax(misaligned)) == 1200, (
        "Old default target_anchor=1200 shifts peak to 1200 (+2400 ms)"
    )
    # And the post-stim split at meta.stim_onset_frame=600 would include
    # 600 frames of pre-stimulus data (the F1 defect).
    wrong_post = misaligned[pre_anchor:]
    assert not np.isclose(wrong_post[0], 1.0), (
        "Old alignment puts the peak at 1200, not at the start of post-stim"
    )


# =========================================================================
# 13. MAJOR F2: Silent Uncropped OOB Anchor Fails Safe
# =========================================================================

def test_silent_uncropped_oob_meta_anchor_falls_back():
    """Silent uncropped trial with meta anchor >= saved_len must fall back (F2).

    A silent trial (no visual / wind) with saved length 1000 and raw/meta
    anchor 2000 must NOT emit 2000 as the saved-coordinate anchor.  The
    ``populate_etl_provenance_and_conditions`` path must enforce
    ``0 <= anchor < saved_len`` and fall back to the derived anchor (0).
    """
    v_len = 1000
    raw_anchor = 2000  # out of bounds for the saved array

    X_silent = np.zeros((v_len, 8), dtype=np.float32)  # no vis, no wind

    metadata = {
        "anchor_frames": [raw_anchor],
        "stimulus_conditions": ["no_stimulus"],
        "is_pure_wind": np.array([False]),
    }
    output: Dict[str, Any] = {}
    populate_etl_provenance_and_conditions(
        output, metadata, X_seqs=[X_silent], lengths=[v_len]
    )

    saved_anchor = output["anchor_frames"][0]
    assert 0 <= saved_anchor < v_len, (
        f"Saved anchor {saved_anchor} out of bounds for length {v_len}"
    )
    assert saved_anchor != raw_anchor, (
        f"Raw coordinate {raw_anchor} must not leak into saved anchors"
    )
    assert saved_anchor == 0, (
        f"Silent OOB trial must fall back to derived anchor 0, got {saved_anchor}"
    )


# =========================================================================
# 14. MAJOR F3: prepare_metadata Fails Closed on Corrupt TTC
# =========================================================================

def test_prepare_metadata_corrupt_ttc_fails_closed():
    """prepare_metadata must raise on corrupt non-null target_ttc_ms (F3).

    A non-numeric target_ttc_ms string in trial_start details must raise
    ValueError (matching ``parse_events_file`` semantics), never silently
    coerce to None.  Legitimate null for unimodal trials is preserved.
    """
    from scripts.prepare_metadata import parse_declared_ttc_ms

    # Legitimate nulls are preserved.
    assert parse_declared_ttc_ms(None, "s", 1) is None
    assert parse_declared_ttc_ms(float("nan"), "s", 1) is None

    # Valid numeric values parse normally.
    assert parse_declared_ttc_ms(-225, "s", 1) == pytest.approx(-225.0)
    assert parse_declared_ttc_ms("-261.5", "s", 1) == pytest.approx(-261.5)

    # Corrupt non-null strings must raise, never return None.
    with pytest.raises(ValueError, match="Corrupt target_ttc_ms"):
        parse_declared_ttc_ms("abc", "s", 1)
    with pytest.raises(ValueError, match="Corrupt target_ttc_ms"):
        parse_declared_ttc_ms("not_a_number", "s", 42)

    # Non-finite values are corrupt when explicitly provided.
    with pytest.raises(ValueError, match="Non-finite target_ttc_ms"):
        parse_declared_ttc_ms(float("inf"), "s", 1)
    with pytest.raises(ValueError, match="Non-finite target_ttc_ms"):
        parse_declared_ttc_ms("nan", "s", 1)


# =========================================================================
# 15. ROOT-CAUSE: lv_ratio_ms Fail-Closed Parsing (B2 finding 2)
# =========================================================================

def test_parse_declared_lv_ratio_ms_fail_closed():
    """Malformed non-null lv_ratio_ms must raise, never map to None (B2 #2).

    Mirrors the target_ttc_ms contract: legitimate null (None / NaN float)
    is preserved, but a non-numeric or non-finite non-null value is corrupt
    and must fail closed rather than silently becoming "no LV ratio".
    """
    from scripts.prepare_metadata import parse_declared_lv_ratio_ms

    # Legitimate nulls are preserved.
    assert parse_declared_lv_ratio_ms(None, "s", 1) is None
    assert parse_declared_lv_ratio_ms(float("nan"), "s", 1) is None

    # Valid numeric values parse normally.
    assert parse_declared_lv_ratio_ms(120, "s", 1) == pytest.approx(120.0)
    assert parse_declared_lv_ratio_ms("119.5", "s", 1) == pytest.approx(119.5)

    # Corrupt non-null strings must raise, never return None.
    with pytest.raises(ValueError, match="Corrupt lv_ratio_ms"):
        parse_declared_lv_ratio_ms("abc", "s", 1)
    with pytest.raises(ValueError, match="Corrupt lv_ratio_ms"):
        parse_declared_lv_ratio_ms("not_a_number", "s", 42)

    # Non-finite values are corrupt when explicitly provided.
    with pytest.raises(ValueError, match="Non-finite lv_ratio_ms"):
        parse_declared_lv_ratio_ms(float("inf"), "s", 1)
    with pytest.raises(ValueError, match="Non-finite lv_ratio_ms"):
        parse_declared_lv_ratio_ms("nan", "s", 1)


def test_events_parse_corrupt_lv_ratio_fails_closed(tmp_path: Path):
    """parse_events_file must raise on corrupt non-null lv_ratio_ms (B2 #2).

    The events parser previously coerced malformed lv_ratio_ms to None
    silently, laundering corrupt values into "no LV ratio declared".
    """
    from nsmor.pipeline.events import parse_events_file

    events_csv = tmp_path / "session_0_events.csv"
    events_csv.write_text(
        "session_id,trial_id,time_ms,event_type,event_value\n"
        'session_0,0,0.0,trial_start,"{""type"": ""looming_wind"", ""target_ttc_ms"": 0, ""lv_ratio_ms"": ""abc""}"\n'
        "session_0,0,200.0,stimulus_onset,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Corrupt lv_ratio_ms"):
        parse_events_file(events_csv)


@pytest.mark.parametrize("raw_lv", [float("inf"), float("-inf"), "inf", "nan"])
def test_events_parse_nonfinite_lv_ratio_fails_closed(tmp_path: Path, raw_lv: Any):
    """Keep the finite LV guard before declarations can enter the merged index."""
    details = {"type": "looming_wind", "target_ttc_ms": 0, "lv_ratio_ms": raw_lv}
    value = json.dumps(details).replace('"', '""')
    events_csv = tmp_path / "session_0_events.csv"
    events_csv.write_text(
        "session_id,trial_id,time_ms,event_type,event_value\n"
        f'session_0,0,0.0,trial_start,"{value}"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Non-finite lv_ratio_ms"):
        parse_events_file(events_csv)
    with pytest.raises(ValueError, match="Non-finite lv_ratio_ms"):
        load_declared_events_index(tmp_path)

def test_events_parse_legitimate_null_lv_ratio_preserved(tmp_path: Path):
    """Legitimate null lv_ratio_ms stays None (not an error)."""
    from nsmor.pipeline.events import parse_events_file

    events_csv = tmp_path / "session_0_events.csv"
    events_csv.write_text(
        "session_id,trial_id,time_ms,event_type,event_value\n"
        'session_0,0,0.0,trial_start,"{""type"": ""baseline_visual"", ""target_ttc_ms"": null, ""lv_ratio_ms"": null}"\n'
        "session_0,0,200.0,stimulus_onset,0\n",
        encoding="utf-8",
    )
    trials = parse_events_file(events_csv)
    entry = trials[("session_0", 0)]
    assert entry["lv_ratio_ms"] is None
    assert entry["target_ttc_ms"] is None


# =========================================================================
# 16. ROOT-CAUSE: Shared Crop Window (B2 finding 1, unit)
# =========================================================================

def test_resolve_anchor_crop_matches_dataset_math():
    """resolve_anchor_crop must reproduce the NSMoRDataset crop arithmetic.

    B2 scenario: raw_len=3000, raw_anchor=2000, max_seq=2400, pre=600.
    Naive start = 2000-600 = 1400; end = min(3000, 1400+2400) = 3000;
    end-start = 1600 < 2400 triggers the clamp: start = max(0, 3000-2400) = 600.
    Final window [600, 3000), length 2400.  Post-crop anchor = 2000-600 = 1400.
    """
    from nsmor.pipeline.conditions import resolve_anchor_crop

    start, end = resolve_anchor_crop(
        n_frames=3000,
        anchor_frame=2000,
        max_seq_len=2400,
        pre_anchor_frames=600,
    )
    assert start == 600
    assert end == 3000
    assert end - start == 2400

    # Short sequences are not cropped at all.
    s2, e2 = resolve_anchor_crop(
        n_frames=2000, anchor_frame=1200,
        max_seq_len=2400, pre_anchor_frames=600,
    )
    assert (s2, e2) == (0, 2000)

    # None max_seq_len disables cropping.
    s3, e3 = resolve_anchor_crop(
        n_frames=5000, anchor_frame=2000,
        max_seq_len=None, pre_anchor_frames=600,
    )
    assert (s3, e3) == (0, 5000)

    # No anchor -> full sequence.
    s4, e4 = resolve_anchor_crop(
        n_frames=5000, anchor_frame=None,
        max_seq_len=2400, pre_anchor_frames=600,
    )
    assert (s4, e4) == (0, 5000)

    # Anchor smaller than pre_anchor clamps start at 0.
    s5, e5 = resolve_anchor_crop(
        n_frames=5000, anchor_frame=300,
        max_seq_len=2400, pre_anchor_frames=600,
    )
    assert s5 == 0
    assert e5 == 2400


# =========================================================================
# 17. ROOT-CAUSE: Full load_validation_data -> extract_gate_trajectory
# =========================================================================

def _build_synthetic_val_dataset(
    tmp_path: Path,
    *,
    raw_len: int = 3000,
    raw_anchor: int = 2000,
    n_trials: int = 4,
) -> Path:
    """Write a synthetic dataset that passes the provenance gate.

    All trials share the same raw_len/raw_anchor so the crop arithmetic is
    unambiguous.  Physical channels are silent (no vis/wind) so the gate
    trajectory is the only signal under test.
    """
    from nsmor.config import PIPELINE_SEMANTICS_VERSION

    X_seqs = [np.zeros((raw_len, 8), dtype=np.float32) for _ in range(n_trials)]
    Y_seqs = [np.zeros(raw_len, dtype=np.float32) for _ in range(n_trials)]
    lengths = np.full((n_trials,), raw_len, dtype=np.int64)
    mcmc_priors = np.full((n_trials, 4), 0.25, dtype=np.float32)

    data = {
        "X_seqs": X_seqs,
        "Y_seqs": Y_seqs,
        "lengths": lengths,
        "mcmc_priors": mcmc_priors,
        "anchor_frames": [raw_anchor] * n_trials,
        "session_ids": [f"sess_{i % 2}" for i in range(n_trials)],
        "trial_ids": list(range(n_trials)),
        "target_ttc_ms": [0.0] * n_trials,
        "stimulus_conditions": ["multisensory"] * n_trials,
        "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
    }
    path = tmp_path / "synthetic_val_dataset.pt"
    torch.save(data, path)
    return path


def test_load_validation_data_post_crop_anchor_full_path(tmp_path: Path):
    """Full path: load_validation_data must emit post-crop anchors (B2 #1).

    Scenario from the B2 repro: raw_len=3000, raw_anchor=2000, max_seq=2400,
    nondefault pre=600.  The dataset crop yields start=600, so the stimulus
    lands at post-crop frame 1400.  ``meta.anchor_frames`` must therefore be
    1400 (saved coordinates), NOT 2000 (pre-crop).  ``extract_gate_trajectory``
    called with those anchors and ``target_anchor=meta.stim_onset_frame`` must
    place a gate peak that sits exactly at the stimulus in the saved tensor
    at the aligned index.
    """
    raw_len = 3000
    raw_anchor = 2000
    pre_anchor = 600
    max_seq_len = 2400

    ds_path = _build_synthetic_val_dataset(
        tmp_path,
        raw_len=raw_len,
        raw_anchor=raw_anchor,
        n_trials=4,
    )

    from scripts.simulate_psychophysics import (
        load_validation_data,
        extract_gate_trajectory,
    )

    val_data = load_validation_data(
        torch.device("cpu"),
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor,
        dataset_path=str(ds_path),
    )
    X_val, Y_val, lengths_val = val_data
    meta = val_data.meta

    B, T, F = X_val.shape
    assert F == 8, f"Expected feature dim 8, got {F}"

    # Crop math (shared resolve_anchor_crop): start = 600, saved_len = 2400.
    saved_len = int(lengths_val[0].item())
    assert saved_len == max_seq_len, (
        f"Expected saved_len {max_seq_len}, got {saved_len}"
    )
    expected_anchor = raw_anchor - 600  # = 1400
    assert meta.stim_onset_frame == pre_anchor
    for i, a in enumerate(meta.anchor_frames):
        assert a == expected_anchor, (
            f"Trial {i}: meta.anchor_frames must be post-crop "
            f"{expected_anchor}, got {a} (pre-crop {raw_anchor} would be wrong)"
        )

    # Place a gate peak exactly at the stimulus in the SAVED tensor
    # (post-crop anchor = 1400).  extract_gate_trajectory must align it to
    # target_anchor = stim_onset_frame = pre_anchor = 600.
    g_gru = torch.zeros(B, T)
    for i in range(B):
        g_gru[i, expected_anchor] = 1.0
    lengths_t = torch.full((B,), T, dtype=torch.int64)
    internals = {"routing_gates": torch.stack([1.0 - g_gru, g_gru], dim=-1)}

    aligned = extract_gate_trajectory(
        internals,
        lengths_t,
        anchor_frames=meta.anchor_frames,
        target_anchor=meta.stim_onset_frame,
    )

    peak_idx = int(np.nanargmax(aligned))
    assert peak_idx == pre_anchor, (
        f"Aligned gate peak expected at target_anchor={pre_anchor}, "
        f"got {peak_idx} — pre-crop/post-crop coordinate mismatch"
    )
    assert np.isclose(aligned[pre_anchor], 1.0)


def test_load_validation_data_uncropped_anchor_identity(tmp_path: Path):
    """When no cropping occurs (saved_len <= max_seq_len) anchors are unchanged."""
    raw_len = 2000
    raw_anchor = 800
    pre_anchor = 600
    max_seq_len = 2400  # larger than raw_len -> no crop

    ds_path = _build_synthetic_val_dataset(
        tmp_path,
        raw_len=raw_len,
        raw_anchor=raw_anchor,
        n_trials=4,
    )

    from scripts.simulate_psychophysics import load_validation_data

    val_data = load_validation_data(
        torch.device("cpu"),
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor,
        dataset_path=str(ds_path),
    )
    meta = val_data.meta
    for a in meta.anchor_frames:
        assert a == raw_anchor, (
            f"Uncropped trial must keep anchor {raw_anchor}, got {a}"
        )


@pytest.mark.parametrize("dt_ms, anchor, length", [
    (4.0, 1200, 2400),
    (10.0, 500, 1200),
])
def test_physical_peak_search_windows_at_4_and_10_ms(dt_ms, anchor, length):
    """F/G ignore stronger peaks at -2.1/+4.1s, and retain +0.5s."""
    early = anchor - int(2100 / dt_ms)
    in_window = anchor + int(500 / dt_ms)
    late = anchor + int(4100 / dt_ms)
    assert 0 <= early < in_window < late < length
    y_pred = np.zeros(length, dtype=np.float32)
    y_pred[early] = 90.0
    y_pred[in_window] = 25.0
    y_pred[late] = 100.0

    metrics = extract_predicted_metrics(
        y_preds=[y_pred], trial_indices=[0], dt_ms=dt_ms,
        anchor_frames=[anchor], reference_frames=[anchor],
    )
    assert metrics["peak_velocities"] == [25.0]
    assert metrics["latencies"] == [500.0]
    lengths = torch.tensor([length], dtype=torch.int64)
    latency = extract_latency_to_peak(
        torch.from_numpy(y_pred).unsqueeze(0), lengths,
        dt_ms=dt_ms, anchor_frames=[anchor],
    )
    assert latency == [500.0]

    y_pred[in_window] = 0.0
    metrics = extract_predicted_metrics(
        y_preds=[y_pred], trial_indices=[0], dt_ms=dt_ms,
        anchor_frames=[anchor], reference_frames=[anchor],
    )
    assert metrics["peak_velocities"] == [0.0]
    assert np.isnan(metrics["latencies"][0])
    latency = extract_latency_to_peak(
        torch.from_numpy(y_pred).unsqueeze(0), lengths,
        dt_ms=dt_ms, anchor_frames=[anchor],
    )
    assert len(latency) == 1 and np.isnan(latency[0])
    peaks = extract_peak_velocity(torch.from_numpy(y_pred).unsqueeze(0), lengths,
                                  dt_ms=dt_ms, anchor_frames=[anchor])
    assert len(peaks) == 1 and np.isnan(peaks[0])


@pytest.mark.parametrize("dt_ms", [0.0, -4.0, float("nan"), float("inf")])
def test_peak_windows_reject_nonfinite_or_nonpositive_dt(dt_ms):
    with pytest.raises(ValueError, match="dt_ms must be finite and positive"):
        extract_predicted_metrics([np.zeros(4)], [0], dt_ms=dt_ms)
    with pytest.raises(ValueError, match="dt_ms must be finite and positive"):
        extract_latency_to_peak(torch.zeros(1, 4), torch.tensor([4]), dt_ms=dt_ms)


@pytest.mark.parametrize('dt_ms, anchor, length', [(4.0, 1200, 2400), (10.0, 500, 1200)])
def test_precollision_latency_and_matching_peak_window(dt_ms, anchor, length):
    from scripts.simulate_psychophysics import LATENCY_SCOPE

    values = torch.zeros(1, length)
    values[0, anchor - int(500 / dt_ms)] = 50.0  # In-window precollision escape.
    values[0, anchor + int(500 / dt_ms)] = 25.0
    values[0, anchor + int(4100 / dt_ms)] = 100.0  # Outside +4s window.
    lengths = torch.tensor([length], dtype=torch.int64)
    assert extract_latency_to_peak(values, lengths, dt_ms=dt_ms,
                                   anchor_frames=[anchor]) == [-500.0]
    assert extract_peak_velocity(values, lengths, dt_ms=dt_ms,
                                 anchor_frames=[anchor]) == [50.0]
    assert 'negative' in LATENCY_SCOPE and 'pre-collision' in LATENCY_SCOPE


def test_phase_f_group_curves_are_explicitly_descriptive(tmp_path):
    from scripts.analyze_integration import export_integration_summary

    path = tmp_path / "integration.json"
    export_integration_summary({"visual_only": {"latency": {"mean": 12.0, "sem": 1.0, "n": 2}}}, path)
    with path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    assert summary["inference"]["status"] == "descriptive_only"
    assert summary["inference"]["adjusted_p_values"] is None
    assert summary["inference"]["effect_sizes"] is None


@pytest.mark.parametrize("dt_ms, anchor, length, offset", [
    (4.0, 1200, 2400, 125), (10.0, 500, 1200, 50),
])
def test_phase_f_unmeasured_visual_baseline_stays_null_in_figure_and_summary(
        monkeypatch, tmp_path, dt_ms, anchor, length, offset):
    import matplotlib.pyplot as plt
    from scripts.analyze_integration import (
        create_integration_figure, export_integration_summary,
    )

    flat = np.zeros(length, dtype=np.float32)
    response = flat.copy()
    response[anchor + offset] = 25.0
    visual = compute_condition_statistics(extract_predicted_metrics(
        [flat], [0], dt_ms=dt_ms, anchor_frames=[anchor],
    ))
    multisensory = compute_condition_statistics(extract_predicted_metrics(
        [response], [0], dt_ms=dt_ms, anchor_frames=[anchor],
    ))
    assert visual["latency"] == {"mean": None, "sem": None, "n": 0}
    assert visual["peak_velocity"] == {"mean": 0.0, "sem": None, "n": 1}
    assert multisensory["latency"]["mean"] == 500.0
    stats = {"visual_only": visual, "multisensory_ttc_-225ms": multisensory}
    summary_path = tmp_path / "integration.json"
    export_integration_summary(stats, summary_path, dt_ms=dt_ms)
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream, parse_constant=lambda val: (_ for _ in ()).throw(ValueError(val)))
    assert summary["baseline_reference"] is None
    assert summary["conditions"]["visual_only"]["latency_to_peak_ms"] == visual["latency"]
    assert summary["conditions"]["multisensory_ttc_-225ms"]["latency_to_peak_ms"]["mean"] == 500.0
    captured = {}

    def capture(*_args, **_kwargs):
        axes = plt.gcf().axes
        captured["baseline_lines"] = [
            line for axis in axes for line in axis.lines
            if line.get_label() == "Visual-Only Baseline"
        ]
        captured["latency"] = axes[0].lines[0].get_ydata().copy()

    monkeypatch.setattr(plt, "savefig", capture)
    create_integration_figure(stats, tmp_path / "integration.png")
    assert captured["baseline_lines"] == []
    assert np.all(np.isfinite(captured["latency"]))
    assert captured["latency"].tolist() == [500.0]


@pytest.mark.parametrize("dt_ms", [4.0, 10.0])
def test_phase_f_all_flat_corpus_exports_unavailable_before_figure(monkeypatch, tmp_path, dt_ms):
    from scripts import analyze_integration as phase_f

    class FlatModel:
        def __init__(self):
            self.dt_ms = dt_ms

        def __call__(self, X, lengths):
            assert X.shape == (2, 12, 8) and lengths.shape == (2,)
            return torch.zeros((2, 12), device=X.device)

    X = torch.zeros(2, 12, 8)
    lengths = torch.tensor([12, 12], dtype=torch.int64)
    info = [{"anchor_frame": 2}, {"anchor_frame": 2}]
    monkeypatch.setattr(phase_f, "load_model_from_checkpoint", lambda *_: FlatModel())
    monkeypatch.setattr(phase_f, "load_dataset", lambda *a, **k: (
        [(X, torch.zeros(2, 12), lengths)], np.zeros(2, dtype=int),
        [12, 12], [x.numpy() for x in X], info,
    ))
    monkeypatch.setattr(phase_f, "group_trials_by_condition", lambda *a, **k: {
        "visual_only": [0], "multisensory_ttc_-225ms": [1],
    })
    figure = tmp_path / "integration.png"
    summary = tmp_path / "integration.json"
    with pytest.raises(ValueError, match="Phase F unavailable: no measurable peak latencies"):
        phase_f.run_integration_analysis(
            tmp_path / "unused.pth", tmp_path / "unused.pt", figure,
            summary_path=summary, dt_ms=dt_ms, max_seq_len=None,
        )
    assert not figure.exists()
    with summary.open(encoding="utf-8") as stream:
        report = json.load(stream, parse_constant=lambda val: (_ for _ in ()).throw(ValueError(val)))
    assert report["status"] == "unavailable" and report["dt_ms"] == dt_ms
    assert report["baseline_reference"] is None
    assert all(c["latency_to_peak_ms"] == {"mean": None, "sem": None, "n": 0}
               for c in report["conditions"].values())


@pytest.mark.parametrize("dt_ms, anchor, length, offset", [
    (4.0, 1200, 2400, 125), (10.0, 500, 1200, 50),
])
def test_visual_only_never_becomes_a_connected_wind_delta_t_point(
        monkeypatch, tmp_path, dt_ms, anchor, length, offset):
    """Regression: visual-only data must not be coerced to delta_t=0.0.

    Visual-only trials have no wind onset, so they have no wind delta_t.
    Before this guard, ``create_integration_figure`` assigned them x=0.0,
    silently fabricating a wind-at-TTC point and connecting it into the
    wind response curve, contradicting the JSON contract (delta_t_ms=null)
    and the psychophysics artifact (n_ttc0=0). The visual-only group may
    only appear as a separate, disconnected no-wind reference line.
    """
    import matplotlib.pyplot as plt
    from scripts.analyze_integration import create_integration_figure

    response = np.zeros(length, dtype=np.float32)
    response[anchor + offset] = 25.0
    visual = compute_condition_statistics(extract_predicted_metrics(
        [response], [0], dt_ms=dt_ms, anchor_frames=[anchor],
    ))
    multisensory = compute_condition_statistics(extract_predicted_metrics(
        [response], [0], dt_ms=dt_ms, anchor_frames=[anchor],
    ))
    assert visual["latency"]["n"] == 1  # measurable: would be a real x=0 candidate
    assert multisensory["latency"]["mean"] == 500.0
    stats = {"visual_only": visual, "multisensory_ttc_-225ms": multisensory}

    captured = {}

    def capture(*_args, **_kwargs):
        axes = plt.gcf().axes
        captured["x"] = axes[0].lines[0].get_xdata().copy()
        captured["vigor_x"] = axes[1].lines[0].get_xdata().copy()
        captured["ref_lines"] = [
            float(np.asarray(line.get_ydata()).ravel()[0])
            for axis in axes for line in axis.lines
            if line.get_label() == "Visual-Only Baseline"
        ]

    monkeypatch.setattr(plt, "savefig", capture)
    create_integration_figure(stats, tmp_path / "integration.png")

    # The wind response series contains only the genuine wind delta_t (-225).
    assert captured["x"].tolist() == [-225.0]
    assert captured["vigor_x"].tolist() == [-225.0]
    # No fabricated point at delta_t=0 on either wind panel.
    assert 0.0 not in captured["x"].tolist()
    assert 0.0 not in captured["vigor_x"].tolist()
    # Visual-only survives only as a separate reference line, never connected:
    # panel A reference = visual latency, panel B reference = visual peak velocity.
    assert sorted(captured["ref_lines"]) == sorted(
        [visual["latency"]["mean"], visual["peak_velocity"]["mean"]]
    )


@pytest.mark.parametrize("dt_ms, anchor, length, offset", [
    (4.0, 1200, 2400, 125), (10.0, 500, 1200, 50),
])
def test_wind_only_stays_an_honest_skip_not_a_wind_axis_point(
        monkeypatch, tmp_path, dt_ms, anchor, length, offset):
    """Wind-only has wind but no TTC-relative delta_t: explicit skip, no point."""
    import matplotlib.pyplot as plt
    from scripts.analyze_integration import create_integration_figure

    response = np.zeros(length, dtype=np.float32)
    response[anchor + offset] = 25.0
    wind_only = compute_condition_statistics(extract_predicted_metrics(
        [response], [0], dt_ms=dt_ms, anchor_frames=[anchor],
    ))
    multisensory = compute_condition_statistics(extract_predicted_metrics(
        [response], [0], dt_ms=dt_ms, anchor_frames=[anchor],
    ))
    stats = {"wind_only": wind_only, "multisensory_ttc_-225ms": multisensory}

    captured = {}

    def capture(*_args, **_kwargs):
        captured["x"] = plt.gcf().axes[0].lines[0].get_xdata().copy()

    monkeypatch.setattr(plt, "savefig", capture)
    create_integration_figure(stats, tmp_path / "integration.png")
    assert captured["x"].tolist() == [-225.0]


@pytest.mark.parametrize("dt_ms, anchor, length", [
    (4.0, 1200, 2400), (10.0, 500, 1200),
])
def test_wind_response_series_is_sorted_by_delta_t_and_paired(
        monkeypatch, tmp_path, dt_ms, anchor, length):
    """Regression: the connected wind curve must be monotone in delta_t.

    Conditions arrive in dictionary insertion order, which reflects how
    trials were observed, not the physical wind-onset axis. Passing them
    unsorted (e.g. x = [-261, -308, -373, -225]) makes the connected
    response curve self-cross. The plotting seam must sort each genuine
    wind delta_t together with its paired latency/velocity means AND
    errors, while keeping the visual-only record out of the connected
    series as a disconnected baseline only.

    Each wind group carries >1 trial so its SEM is a real nonzero value,
    and the four groups carry distinct latency/velocity SEMs. A bug that
    permuted only the SEM vectors (leaving x and y correctly sorted)
    would be invisible to a single-trial n=1/SEM=0 fixture; here the
    plotted error bars are read back from the real matplotlib
    ErrorbarContainer and checked for both panels.
    """
    import matplotlib.pyplot as plt
    from scripts.analyze_integration import (
        create_integration_figure,
        parse_ttc_condition_delta,
    )

    def trial(latency_ms, velocity):
        response = np.zeros(length, dtype=np.float32)
        response[anchor + int(round(latency_ms / dt_ms))] = velocity
        return response

    def condition(pairs):
        responses = [trial(lat, vel) for lat, vel in pairs]
        return compute_condition_statistics(extract_predicted_metrics(
            responses, list(range(len(responses))),
            dt_ms=dt_ms, anchor_frames=[anchor] * len(responses),
        ))

    # >1 trial per wind group; each group's per-trial spread is chosen so
    # every group gets a distinct nonzero latency/velocity SEM.
    groups = {
        "multisensory_ttc_-261ms": [
            (900.0, 30.0), (1000.0, 31.0), (1300.0, 33.0), (1600.0, 36.0),
        ],
        "multisensory_ttc_-373ms": [
            (400.0, 10.0), (500.0, 12.0), (600.0, 14.0), (700.0, 16.0),
        ],
        "multisensory_ttc_-225ms": [
            (1000.0, 25.0), (1500.0, 26.0), (1600.0, 27.0), (2100.0, 29.0),
        ],
        "multisensory_ttc_-308ms": [
            (200.0, 20.0), (350.0, 23.0), (550.0, 27.0), (800.0, 32.0),
        ],
    }
    # Out-of-order insertion: observed order is not the delta_t order.
    stats = {name: condition(pairs) for name, pairs in groups.items()}
    stats["visual_only"] = condition([(250.0, 12.0), (300.0, 14.0)])

    # The fixture must actually exercise real error bars: every wind group
    # needs n>1, nonzero SEM, and the groups must differ in SEM and mean.
    for name in groups:
        assert stats[name]["latency"]["n"] > 1
        assert stats[name]["latency"]["sem"] > 0.0
        assert stats[name]["peak_velocity"]["sem"] > 0.0
    lat_sems = [stats[n]["latency"]["sem"] for n in groups]
    vel_sems = [stats[n]["peak_velocity"]["sem"] for n in groups]
    assert len(set(round(s, 9) for s in lat_sems)) == len(groups)
    assert len(set(round(s, 9) for s in vel_sems)) == len(groups)

    delta_of = {n: parse_ttc_condition_delta(n) for n in groups}
    order = sorted(groups, key=lambda n: delta_of[n])
    exp_x = [delta_of[n] for n in order]
    exp_lat_y = [stats[n]["latency"]["mean"] for n in order]
    exp_lat_e = [stats[n]["latency"]["sem"] for n in order]
    exp_vel_y = [stats[n]["peak_velocity"]["mean"] for n in order]
    exp_vel_e = [stats[n]["peak_velocity"]["sem"] for n in order]

    captured = {}

    def capture(*_args, **_kwargs):
        axes = plt.gcf().axes

        def series(axis):
            # The wind errorbar series is the panel's ErrorbarContainer;
            # read the plotted data line and the real error-bar segments
            # (lines[2] is the LineCollection of bar segments).
            assert len(axis.containers) == 1, "expected one errorbar series"
            container = axis.containers[0]
            x = np.asarray(container.lines[0].get_xdata(), dtype=float)
            y = np.asarray(container.lines[0].get_ydata(), dtype=float)
            segs = np.asarray(container.lines[2][0].get_segments(), dtype=float)
            yerr = (segs[:, 1, 1] - segs[:, 0, 1]) / 2.0
            return x, y, yerr

        captured["latency"] = series(axes[0])
        captured["velocity"] = series(axes[1])
        captured["ref_lines"] = sorted(
            float(np.asarray(line.get_ydata()).ravel()[0])
            for axis in axes for line in axis.lines
            if line.get_label() == "Visual-Only Baseline"
        )

    monkeypatch.setattr(plt, "savefig", capture)
    create_integration_figure(stats, tmp_path / "integration.png")

    lat_x, lat_y, lat_e = captured["latency"]
    vel_x, vel_y, vel_e = captured["velocity"]

    # Both panels share the genuine, strictly increasing wind delta_t axis.
    assert lat_x.tolist() == vel_x.tolist() == exp_x
    assert np.all(np.diff(lat_x) > 0.0)
    assert np.all(np.diff(vel_x) > 0.0)
    # x/y/yerr stay jointly permuted: sorted x carries the matching paired
    # means and error bars on BOTH panels.
    assert np.allclose(lat_y, exp_lat_y)
    assert np.allclose(vel_y, exp_vel_y)
    assert np.allclose(lat_e, exp_lat_e)
    assert np.allclose(vel_e, exp_vel_e)
    # The plotted error bars are the real, distinct, nonzero SEMs.
    assert np.all(lat_e > 0.0)
    assert np.all(vel_e > 0.0)
    assert len(set(np.round(lat_e, 9))) == len(groups)
    assert len(set(np.round(vel_e, 9))) == len(groups)
    # No fabricated wind-at-TTC point from the visual-only record.
    assert 0.0 not in lat_x.tolist() and 0.0 not in vel_x.tolist()
    # Visual-only survives only as a disconnected baseline reference,
    # drawn from its own latency/peak-velocity statistics (never connected
    # to the wind series). Compare against the actual stats to stay
    # dt-agnostic.
    vis = stats["visual_only"]
    assert captured["ref_lines"] == sorted(
        [vis["latency"]["mean"], vis["peak_velocity"]["mean"]]
    )
