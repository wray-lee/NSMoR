"""
End-to-end pipeline validation with synthetic data.

Generates fake kinematics / events CSVs, runs the full pipeline
(load → label → extract → train MCMC → DataLoader), and asserts
all tensor shapes and invariants.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
import weakref
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nsmor.config import (
    DEFAULT_FEATURE,
    DEFAULT_MCMC_TRAINING,
    DEFAULT_THRESHOLD,
    DEFAULT_TIME_WINDOW,
    FeatureConfig,
    Label,
    MCMCTrainingConfig,
    ThresholdConfig,
    TimeWindowConfig,
)
from nsmor.pipeline.io import (
    _parse_event_value,
    extract_trial_data,
    load_and_concat_sessions,
    load_kinematics_csv,
)
from nsmor.pipeline.kinematics import demirror_prediction, mirror_to_right
from nsmor.pipeline.labeling import (
    assign_ground_truth_labels,
    labeling_funnel_summary,
)
from nsmor.data_extractor import (
    PURE_WIND_PREPEND_FRAMES,
    _compute_pure_wind_prepend_frames,
    build_sequence_dataset,
    build_snapshot_dataset,
    extract_mcmc_snapshot,
    extract_trial_sequence,
)
from nsmor.mcmc_module import (
    MCMCPriorGenerator,
    MCMCPriorSKLearn,
    MarkovTransitionEstimator,
    train_mcmc,
)
from nsmor.nsmor_dataloader import create_dataloader
from nsmor.model_nsmor_core import NSMoR


# ═══════════════════════════════════════════════════════════════
# Synthetic data generators
# ═══════════════════════════════════════════════════════════════

def _make_synthetic_csvs(
    tmp_dir: Path,
    n_sessions: int = 2,
    trials_per_session: int = 10,
    frames_per_trial: int = 400,
    dt_ms: float = 10.0,
) -> tuple[Path, Path]:
    """
    Write synthetic kinematics + events CSVs to *tmp_dir*.

    Trials are labelled deterministically by *trial_id*:
      0-2 → Startle (high velocity spike after stimulus)
      3-5 → Walk (sustained moderate velocity)
      6-7 → Pre_Active (high baseline velocity)
      8-9 → NoResponse (no movement)
    """
    kin_rows = []
    evt_rows = []

    for s in range(n_sessions):
        sid = f"session_{s}"
        for t in range(trials_per_session):
            time_ms = np.arange(frames_per_trial) * dt_ms
            stimulus_onset = 2000.0  # 2 s baseline

            # Baseline velocity: low or high (pre-active)
            if 6 <= t <= 7:
                base_vel = np.random.uniform(0.8, 1.5, size=frames_per_trial)
            else:
                base_vel = np.random.uniform(0.0, 0.2, size=frames_per_trial)

            velocity = base_vel.copy()

            # Post-stimulus response
            stim_idx = int(stimulus_onset / dt_ms)
            if t <= 2:
                # Startle/Escape: sustained high velocity (>5 cm/s).
                # Round-1 labeling fix requires >=50% of the 250 ms
                # post-stimulus window above threshold, so the response
                # must span >= ~13 frames; use 35 frames (350 ms).
                spike_start = stim_idx + np.random.randint(5, 20)
                spike_end = min(spike_start + 35, frames_per_trial)
                velocity[spike_start:spike_end] = np.random.uniform(8.0, 15.0)
            elif 3 <= t <= 5:
                # Walk: sustained moderate velocity
                walk_start = stim_idx + np.random.randint(10, 50)
                walk_end = min(walk_start + 80, frames_per_trial)
                velocity[walk_start:walk_end] = np.random.uniform(1.5, 4.0)

            acceleration = np.gradient(velocity, dt_ms / 1000.0)

            # Visual angle: ramp from 0 after stimulus
            visual_angle = np.zeros(frames_per_trial)
            visual_angle[stim_idx:] = np.linspace(
                5.0, 60.0, frames_per_trial - stim_idx,
            )

            wind_state = np.zeros(frames_per_trial)
            if t % 2 == 0:
                wind_state[stim_idx:] = 1.0

            l_v_ratio = visual_angle * 0.1
            if not np.any(wind_state):
                # One stimulus has one l/v; keep its collision inside this short fixture.
                collision_ms = stimulus_onset + 20.0 / math.tan(math.radians(1.0))
                remaining_ms = np.maximum(collision_ms - time_ms[stim_idx:], 0.0)
                visual_angle[stim_idx:] = np.minimum(np.degrees(2.0 * np.arctan2(20.0, remaining_ms)), 179.0)
                l_v_ratio = np.full(frames_per_trial, 20.0)

            for f in range(frames_per_trial):
                kin_rows.append({
                    "session_id": sid,
                    "trial_id": t,
                    "time_ms": float(time_ms[f]),
                    "x_pos": float(f * 0.01),
                    "y_pos": float(f * 0.005),
                    "heading": 0.0,
                    "velocity": float(velocity[f]),
                    "acceleration": float(acceleration[f]),
                    "visual_angle": float(visual_angle[f]),
                    "wind_state": float(wind_state[f]),
                    "l_v_ratio": float(l_v_ratio[f]),
                })

            # Events
            evt_rows.append({
                "session_id": sid,
                "trial_id": t,
                "time_ms": 0.0,
                "event_type": "trial_start",
                "event_value": 1,
            })
            evt_rows.append({
                "session_id": sid,
                "trial_id": t,
                "time_ms": stimulus_onset,
                "event_type": "stimulus_onset",
                "event_value": 1,
            })

    kin_df = pd.DataFrame(kin_rows)
    evt_df = pd.DataFrame(evt_rows)

    kin_path = tmp_dir / "kinematics.csv"
    evt_path = tmp_dir / "events.csv"
    kin_df.to_csv(kin_path, index=False)
    evt_df.to_csv(evt_path, index=False)

    return kin_path, evt_path


def _make_pure_wind_csvs(
    tmp_dir: Path,
    frames_per_trial: int = 400,
    dt_ms: float = 10.0,
) -> tuple[Path, Path]:
    """
    Write synthetic CSVs for a Pure Wind trial (visual_angle ≡ 0).
    """
    kin_rows = []
    evt_rows = []

    time_ms = np.arange(frames_per_trial) * dt_ms
    stimulus_onset = 2000.0
    stim_idx = int(stimulus_onset / dt_ms)

    velocity = np.random.uniform(0.0, 0.1, size=frames_per_trial)
    acceleration = np.gradient(velocity, dt_ms / 1000.0)
    visual_angle = np.zeros(frames_per_trial)       # ← pure wind: no looming
    wind_state = np.zeros(frames_per_trial)
    wind_state[stim_idx:] = 1.0
    l_v_ratio = np.zeros(frames_per_trial)

    for f in range(frames_per_trial):
        kin_rows.append({
            "session_id": "wind_session",
            "trial_id": 0,
            "time_ms": float(time_ms[f]),
            "x_pos": float(f * 0.01),
            "y_pos": float(f * 0.005),
            "heading": 0.0,
            "velocity": float(velocity[f]),
            "acceleration": float(acceleration[f]),
            "visual_angle": float(visual_angle[f]),
            "wind_state": float(wind_state[f]),
            "l_v_ratio": float(l_v_ratio[f]),
        })

    evt_rows.append({
        "session_id": "wind_session",
        "trial_id": 0,
        "time_ms": 0.0,
        "event_type": "trial_start",
        "event_value": 1,
    })
    evt_rows.append({
        "session_id": "wind_session",
        "trial_id": 0,
        "time_ms": stimulus_onset,
        "event_type": "stimulus_onset",
        "event_value": 1,
    })

    kin_df = pd.DataFrame(kin_rows)
    evt_df = pd.DataFrame(evt_rows)

    kin_path = tmp_dir / "kinematics_wind.csv"
    evt_path = tmp_dir / "events_wind.csv"
    kin_df.to_csv(kin_path, index=False)
    evt_df.to_csv(evt_path, index=False)

    return kin_path, evt_path


def _make_visual_only_csvs(
    tmp_dir: Path,
    n_trials: int = 3,
    dt_ms: float = 4.0,
    lv_ratio_ms: float = 120.0,
    init_deg: float = 2.0,
    looming_onset_ms: float = 0.0,
) -> tuple[Path, Path]:
    """Write visual-only CSVs from declared, trial-specific stimulus geometry."""
    t_col_ms = looming_onset_ms + lv_ratio_ms / math.tan(math.radians(init_deg / 2.0))
    frames_per_trial = int((t_col_ms + 2500.0) / dt_ms)

    kin_rows: list[dict] = []
    evt_rows: list[dict] = []

    for t in range(n_trials):
        time_ms = np.arange(frames_per_trial) * dt_ms
        # theta(t) = 2*atan(lv / (t_col - t)), clamped past collision.
        delta_s = np.clip((t_col_ms - time_ms) / 1000.0, 1e-6, None)
        visual_angle = np.degrees(2.0 * np.arctan((lv_ratio_ms / 1000.0) / delta_s))
        visual_angle[time_ms < looming_onset_ms] = init_deg
        visual_angle[time_ms > t_col_ms] = 179.0

        velocity = np.random.uniform(0.0, 0.15, size=frames_per_trial)
        acceleration = np.gradient(velocity, dt_ms / 1000.0)

        for f in range(frames_per_trial):
            kin_rows.append({
                "session_id": "visual_session",
                "trial_id": t,
                "time_ms": float(time_ms[f]),
                "x_pos": float(f * 0.01),
                "y_pos": float(f * 0.005),
                "heading": 0.0,
                "velocity": float(velocity[f]),
                "acceleration": float(acceleration[f]),
                "visual_angle": float(visual_angle[f]),
                "wind_state": 0.0,          # <- visual only: no wind, ever
                "l_v_ratio": float(lv_ratio_ms),
            })

        for event_type, stamp, details in (
            ("trial_start", 0.0, {"type": "baseline_visual", "lv_ratio_ms": lv_ratio_ms, "init_deg": init_deg}),
            ("phase_transition", looming_onset_ms, {"from_phase": "TrialStart", "to_phase": "Looming"}),
            ("stimulus_onset", looming_onset_ms, {"source": "kinematics_injected"}),
        ):
            evt_rows.append({
                "session_id": "visual_session",
                "trial_id": t,
                "time_ms": stamp,
                "event_type": event_type,
                "event_value": json.dumps(details),
            })

    kin_path = tmp_dir / "kinematics_visual.csv"
    evt_path = tmp_dir / "events_visual.csv"
    pd.DataFrame(kin_rows).to_csv(kin_path, index=False)
    pd.DataFrame(evt_rows).to_csv(evt_path, index=False)
    return kin_path, evt_path


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

def test_session_streaming_matches_corpus_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve trial fields, wind side, ordering and corpus-wide cadence statistics.

    The spies also fail if the loader concatenates every pair or the extractor
    sees the whole kinematics/events corpus for any one trial.
    """
    from scripts import prepare_data as prep
    from nsmor.pipeline.io import EVENT_COLUMNS, KINEMATICS_COLUMNS

    # Directory order and session-id order intentionally disagree. Frames and
    # event rows are interleaved and unsorted, as in an imperfect export.
    for dirname, sid in (("a_directory", "z_session"), ("z_directory", "a_session")):
        directory = tmp_path / dirname
        directory.mkdir()
        kin_rows = []
        for trial_id, frame, time_ms in (
            (2, 1, 8.0), (1, 2, 12.0), (2, 0, 0.0),
            (1, 0, 0.0), (2, 2, 4.0), (1, 1, 4.0),
        ):
            kin_rows.append((
                sid, trial_id, time_ms, float(100 * trial_id + frame),
                float(frame), 0.0, float(frame + trial_id), 0.0,
                float(trial_id), float(trial_id == 2), 120.0,
            ))
        pd.DataFrame(kin_rows, columns=KINEMATICS_COLUMNS).to_csv(
            directory / "kinematics.csv", index=False,
        )
        evt_rows = [
            (sid, 2, 8.0, "stimulus_onset", "1"),
            (sid, 2, 0.0, "trial_start", "{'screen_side': 'right'}"),
            (sid, 2, 4.0, "wind_onset", "{'wind_side': 'left'}"),
        ]
        if sid == "a_session":
            evt_rows += [
                (sid, 1, 0.0, "trial_start", "{'wind_dir': 'right'}"),
                (sid, 1, 4.0, "stimulus_onset", "1"),
            ]
        # z_session/1 intentionally has no events.
        pd.DataFrame(evt_rows, columns=EVENT_COLUMNS).to_csv(
            directory / "events.csv", index=False,
        )

    # A later CSV continues z_session/2 and disagrees about wind side.
    continuation = tmp_path / "m_directory"
    continuation.mkdir()
    pd.DataFrame([
        ("z_session", 2, 16.0, 203.0, 3.0, 0.0, 5.0, 0.0, 2.0, 1.0, 120.0),
    ], columns=KINEMATICS_COLUMNS).to_csv(continuation / "kinematics.csv", index=False)
    pd.DataFrame([
        ("z_session", 2, 16.0, "wind_onset", "{'wind_side': 'right'}"),
        ("z_session", 1, 0.0, "trial_start", "{'wind_dir': 'right'}"),
    ], columns=EVENT_COLUMNS).to_csv(continuation / "events.csv", index=False)

    # A numeric-only event file must keep the corpus-wide event_value dtype.
    numeric = tmp_path / "n_directory"
    numeric.mkdir()
    pd.DataFrame([
        ("b_session", 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 120.0),
        ("b_session", 0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 120.0),
    ], columns=KINEMATICS_COLUMNS).to_csv(numeric / "kinematics.csv", index=False)
    pd.DataFrame([("b_session", 0, 0.0, "trial_start", 1)],
                 columns=EVENT_COLUMNS).to_csv(numeric / "events.csv", index=False)

    pairs = prep.pair_csv_files(tmp_path)
    full = load_and_concat_sessions([p[0] for p in pairs], [p[1] for p in pairs])
    expected = [
        extract_trial_data(full, sid, tid)
        for (sid, tid), _ in full["kinematics"].groupby(["session_id", "trial_id"])
    ]
    expected_diagnostics = prep.compute_sampling_diagnostics(full["kinematics"], 4.0)
    max_trial_rows = max(len(v) for _, v in full["kinematics"].groupby(["session_id", "trial_id"]))
    max_event_rows = max(len(v) for _, v in full["events"].groupby(["session_id", "trial_id"]))
    original_loader = prep.load_and_concat_sessions
    original_extractor = prep.extract_trial_data
    seen = []

    def bounded_loader(kin_paths, evt_paths):
        assert len(kin_paths) == len(evt_paths) == 1
        return original_loader(kin_paths, evt_paths)

    def bounded_extractor(data, sid, tid):
        assert len(data["kinematics"]) <= max_trial_rows
        assert len(data["events"]) <= max_event_rows
        assert set(zip(data["kinematics"]["session_id"], data["kinematics"]["trial_id"])) == {(sid, tid)}
        assert set(zip(data["events"]["session_id"], data["events"]["trial_id"])) <= {(sid, tid)}
        seen.append((sid, tid))
        return original_extractor(data, sid, tid)

    monkeypatch.setattr(prep, "load_and_concat_sessions", bounded_loader)
    monkeypatch.setattr(prep, "extract_trial_data", bounded_extractor)
    actual, diagnostics, n_kin, n_evt = prep._load_trials_and_diagnostics(pairs, 4.0)

    assert n_kin == len(full["kinematics"])
    assert n_evt == len(full["events"])
    assert len(seen) >= len(expected)
    assert len(expected) == len(actual) == 5
    assert diagnostics == expected_diagnostics
    assert [(t["session_id"], t["trial_id"]) for t in actual] == [
        (t["session_id"], t["trial_id"]) for t in expected
    ]
    assert [t["wind_side_original"] for t in actual] == [
        t["wind_side_original"] for t in expected
    ] == ["right", "left", "unknown", "right", "unknown"]
    for old, new in zip(expected, actual):
        assert new.keys() == old.keys()
        for key in old:
            if isinstance(old[key], np.ndarray):
                np.testing.assert_array_equal(new[key], old[key], err_msg=key)
                assert new[key].dtype == old[key].dtype
            else:
                assert new[key] == old[key], key


def _assert_sessionwise_csv_parity(tmp_path: Path, sessions: list[tuple[object, list[object]]]):
    """Compare actual CSV loading with the original corpus extractor."""
    from scripts import prepare_data as prep
    from nsmor.pipeline.io import EVENT_COLUMNS, KINEMATICS_COLUMNS

    pairs = []
    for index, (sid, values) in enumerate(sessions):
        directory = tmp_path / str(index)
        directory.mkdir()
        kin_path, evt_path = directory / "kinematics.csv", directory / "events.csv"
        pd.DataFrame([
            (sid, 1, float(time), float(index + time), 0.0, 0.0,
             float(time), 0.0, 0.0, 0.0, 120.0)
            for time in (0, 4)
        ], columns=KINEMATICS_COLUMNS).to_csv(kin_path, index=False)
        pd.DataFrame([
            (sid, 1, float(i * 4), "photodiode_trigger", value)
            for i, value in enumerate(values)
        ], columns=EVENT_COLUMNS).to_csv(evt_path, index=False)
        pairs.append((kin_path, evt_path))

    full = load_and_concat_sessions([kin for kin, _ in pairs], [evt for _, evt in pairs])
    expected = [
        extract_trial_data(full, sid, tid)
        for (sid, tid), _ in full["kinematics"].groupby(["session_id", "trial_id"])
    ]
    actual, diagnostics, n_kin, n_evt = prep._load_trials_and_diagnostics(pairs, 4.0)
    assert (n_kin, n_evt) == (len(full["kinematics"]), len(full["events"]))
    assert diagnostics == prep.compute_sampling_diagnostics(full["kinematics"], 4.0)
    assert len(actual) == len(expected)
    for old, new in zip(expected, actual):
        assert old.keys() == new.keys()
        for key in old:
            if isinstance(old[key], np.ndarray):
                np.testing.assert_array_equal(new[key], old[key], err_msg=key)
                assert new[key].dtype == old[key].dtype, key
            else:
                assert new[key] == old[key], key
                assert type(new[key]) is type(old[key]), key
    return actual


def test_sessionwise_csv_mixed_session_id_order(tmp_path: Path) -> None:
    trials = _assert_sessionwise_csv_parity(
        tmp_path, [("a_session", [1, 0]), (123, [1, 0])],
    )
    assert [(trial["session_id"], trial["trial_id"]) for trial in trials] == [
        (123, 1), ("a_session", 1),
    ]


@pytest.mark.parametrize(
    "values_per_session",
    [
        ([True, False], [1, 0]),
        ([True, False], [1.5, 0.0]),
        ([], [True, False]),
        ([], [1, 0]),
        ([None, None], [1, 0]),
    ],
    ids=["bool-int", "bool-float", "empty-bool", "empty-int", "all-missing-int"],
)
def test_sessionwise_csv_event_value_promotion(
    tmp_path: Path, values_per_session: tuple[list[object], list[object]],
) -> None:
    _assert_sessionwise_csv_parity(
        tmp_path,
        [("a_session", values_per_session[0]), ("b_session", values_per_session[1])],
    )



def _split_pair(directory: Path, sid: str, tid: int, start: int, count: int,
                dt_ms: float, *, repeated_start: bool = False) -> tuple[Path, Path]:
    from nsmor.pipeline.io import EVENT_COLUMNS, KINEMATICS_COLUMNS

    directory.mkdir(parents=True)
    kin_path, evt_path = directory / "kinematics.csv", directory / "events.csv"
    pd.DataFrame([
        (sid, tid, frame * dt_ms, 0., 0., 0., 0.1, 0., 0., 0., 120.)
        for frame in range(start, start + count)
    ], columns=KINEMATICS_COLUMNS).to_csv(kin_path, index=False)
    pd.DataFrame([
        (sid, tid, 0., "trial_start", "{}")
    ] if repeated_start else [], columns=EVENT_COLUMNS).to_csv(evt_path, index=False)
    return kin_path, evt_path


@pytest.mark.parametrize("dt_ms", [4., 10.])
@pytest.mark.parametrize("offset_frames", [0., 0.5])
@pytest.mark.parametrize("producer", ["metadata", "eager"])
def test_overlapping_split_trial_rejected_before_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    dt_ms: float, offset_frames: float, producer: str,
) -> None:
    from scripts import prepare_data, prepare_metadata
    import sys

    sid, tid = "animalA_session_1", 11
    raw = tmp_path / "raw"
    first = _split_pair(raw / "a", sid, tid, 0, 300, dt_ms, repeated_start=True)
    second = _split_pair(raw / "b", sid, tid, 0, 300, dt_ms)
    if offset_frames:
        frame = pd.read_csv(second[0])
        frame["time_ms"] += offset_frames * dt_ms
        frame.to_csv(second[0], index=False)
    output = tmp_path / "produced.pt"
    if producer == "metadata":
        monkeypatch.setattr(sys, "argv", ["prepare_metadata.py", "--raw_dir", str(raw),
                                          "--output", str(output)])
        run = prepare_metadata.main
    else:
        run = lambda: prepare_data._load_trials_and_diagnostics([first, second], dt_ms)
    with pytest.raises(ValueError, match=r"session=.*animalA_session_1.*trial=11.*(overlap|time_ms)"):
        run()
    assert not output.exists()


def test_repeated_trial_start_and_duplicate_time_rejected(tmp_path: Path) -> None:
    from nsmor.pipeline.io import EVENT_COLUMNS

    sid, tid = "animalA_session_1", 11
    first = _split_pair(tmp_path / "a", sid, tid, 0, 10, 4., repeated_start=True)
    second = _split_pair(tmp_path / "b", sid, tid, 10, 10, 4., repeated_start=True)
    data = load_and_concat_sessions([first[0], second[0]], [first[1], second[1]])
    with pytest.raises(ValueError, match="trial_start"):
        extract_trial_data(data, sid, tid)
    pd.DataFrame(columns=EVENT_COLUMNS).to_csv(second[1], index=False)
    frame = pd.read_csv(second[0])
    frame.loc[1, "time_ms"] = frame.loc[0, "time_ms"]
    frame.to_csv(second[0], index=False)
    data = load_and_concat_sessions([first[0], second[0]], [first[1], second[1]])
    with pytest.raises(ValueError, match="time_ms"):
        extract_trial_data(data, sid, tid)


@pytest.mark.parametrize("dt_ms", [4., 10.])
def test_ordered_split_with_event_only_continuation(tmp_path: Path, dt_ms: float) -> None:
    from nsmor.pipeline.io import EVENT_COLUMNS
    from scripts import prepare_data

    sid, tid = "animalA_session_1", 11
    first = _split_pair(tmp_path / "a", sid, tid, 0, 300, dt_ms)
    second = _split_pair(tmp_path / "b", sid, tid, 300, 100, dt_ms)
    event_only = _split_pair(tmp_path / "c", "other_session", 12, 0, 2, dt_ms)
    pd.DataFrame([(sid, tid, 0., "trial_start", "{}"),
                  (sid, tid, 200 * dt_ms, "stimulus_onset", "")],
                 columns=EVENT_COLUMNS).to_csv(event_only[1], index=False)
    trials, _, _, _ = prepare_data._load_trials_and_diagnostics(
        [first, second, event_only], dt_ms,
    )
    trial = next(t for t in trials if (t["session_id"], t["trial_id"]) == (sid, tid))
    assert trial["time_ms"].shape == (400,)
    np.testing.assert_array_equal(trial["time_ms"], np.arange(400) * dt_ms)
    assert trial["event_types"].tolist() == ["trial_start", "stimulus_onset"]


def test_disjoint_split_in_reverse_pair_order_rejected(tmp_path: Path) -> None:
    sid, tid = "animalA_session_1", 11
    later = _split_pair(tmp_path / "a", sid, tid, 10, 10, 4.)
    earlier = _split_pair(tmp_path / "b", sid, tid, 0, 10, 4.)
    data = load_and_concat_sessions(
        [later[0], earlier[0]], [later[1], earlier[1]],
    )
    with pytest.raises(ValueError, match="unordered CSV pair time_ms ranges"):
        extract_trial_data(data, sid, tid)


def test_converter_refuses_overlapping_replayed_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import prepare_metadata
    from scripts.convert_metadata_to_etl import main as convert_main
    from nsmor.pipeline.io import EVENT_COLUMNS, KINEMATICS_COLUMNS
    import sys

    raw = tmp_path / "raw"
    sid, tid, other_sid = "animalA_session_1", 11, "animalB_session_1"
    first = _split_pair(raw / "a", sid, tid, 0, 300, 4.)
    second = _split_pair(raw / "b", sid, tid, 300, 100, 4.)
    extra = pd.DataFrame([
        (other_sid, 12, frame * 4., 0., 0., 0., 0.2, 0.,
         10. if frame >= 200 else 0., int(frame >= 200), 120.)
        for frame in range(300)
    ], columns=KINEMATICS_COLUMNS)
    pd.concat([pd.read_csv(second[0]), extra], ignore_index=True).to_csv(second[0], index=False)
    pd.DataFrame([
        (sid, tid, 0., "trial_start", "{}"),
        (sid, tid, 800., "stimulus_onset", ""),
    ], columns=EVENT_COLUMNS).to_csv(first[1], index=False)
    pd.DataFrame([
        (other_sid, 12, 0., "trial_start", "{}"),
        (other_sid, 12, 800., "stimulus_onset", ""),
    ], columns=EVENT_COLUMNS).to_csv(second[1], index=False)

    metadata = tmp_path / "metadata.pt"
    etl = tmp_path / "etl.pt"
    monkeypatch.setattr(prepare_metadata, "resolve_group_folds", lambda *args, **kwargs: 2)
    monkeypatch.setattr(prepare_metadata, "train_mcmc_cross_fitted",
                        lambda *args, **kwargs: (np.full((2, 4), 0.25), [], []))
    monkeypatch.setattr(sys, "argv", ["prepare_metadata.py", "--raw_dir", str(raw),
                                      "--output", str(metadata)])
    prepare_metadata.main()
    assert metadata.exists()
    frame = pd.read_csv(second[0])
    frame.loc[frame["session_id"] == sid, "time_ms"] -= 1200.
    frame.to_csv(second[0], index=False)
    # Bind the changed synthetic bytes to isolate the temporal invariant from
    # the independent raw-input digest gate.
    from hashlib import sha256
    saved = torch.load(metadata, weights_only=False)
    for spec in saved["trial_specs"]:
        for source in spec["source_pairs"]:
            if Path(source["session_dir"]) / source["kinematics_file"] == second[0]:
                source["kinematics_sha256"] = sha256(second[0].read_bytes()).hexdigest()
    torch.save(saved, metadata)
    with pytest.raises(ValueError, match="overlapping|time_ms"):
        convert_main(["--input", str(metadata), "--output", str(etl)])
    assert not etl.exists()


class TestPipelineIO:
    """Tests for pipeline.io module."""

    def test_load_and_concat_sessions(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])
        assert "kinematics" in data
        assert "events" in data
        assert len(data["kinematics"]) > 0
        assert len(data["events"]) > 0

    def test_extract_trial_data(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])
        trial = extract_trial_data(data, "session_0", 0)
        assert trial["session_id"] == "session_0"
        assert trial["trial_id"] == 0
        assert len(trial["time_ms"]) == 400
        assert trial["velocity"].dtype == np.float64

    def test_extract_trial_data_normalizes_numpy_groupby_keys(
        self, tmp_path: Path,
    ) -> None:
        """Groupby keys (np.int64 on pandas 2.x) are returned as native scalars.

        Regression for the NumPy 2.x scalar-repr leak: ``prepare_data`` /
        ``prepare_metadata`` iterate a ``groupby(["session_id", "trial_id"])``
        key and pass it straight to :func:`extract_trial_data`, so the dict
        must normalise numpy scalars — otherwise ``trial=np.int64(2)`` leaks
        into error messages and breaks the documented scalar contract.
        """
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])
        for sid, tid in data["kinematics"].groupby(
            ["session_id", "trial_id"]
        ).indices:
            trial = extract_trial_data(data, sid, tid)
            assert isinstance(trial["session_id"], str)
            assert type(trial["trial_id"]) is int, (
                f"trial_id leaked as {type(trial['trial_id'])}"
            )

    def test_sanitize_trial_first_frame_spike(self, tmp_path: Path) -> None:
        """First-frame phantom spike is zeroed; legitimate onsets preserved.

        Regression test for the NSMoR audit finding: adapted cercus CSVs can
        carry a ``10**6``-cm/s velocity spike on a trial's first frame
        (``time_ms==0``), a pure cross-trial kinematics artifact. The artifact
        guard ``artifact_velocity_cm_s`` (default 1000 cm/s) must zero that
        frame's velocity and recompute acceleration from the cleaned velocity
        so no spike survives, while preserving first-frame velocities below
        the plausibility bound (real wind-onset escape onsets).
        """
        # Build two trials; plant one phantom spike on trial 1's first frame.
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        raw = pd.read_csv(kin_path)
        spiked_idx = raw.index[(raw["trial_id"] == 1) & (raw["time_ms"] == 0.0)][0]
        raw.loc[spiked_idx, "velocity"] = 5_278_733.873
        raw.loc[spiked_idx, "acceleration"] = 5_278_733.873
        raw.to_csv(kin_path, index=False)

        df = load_kinematics_csv(kin_path)
        # The spiked trial's first frame must be zeroed...
        spiked = df[(df["trial_id"] == 1) & (df["time_ms"] == 0.0)]
        assert spiked["velocity"].abs().max() < 1.0, (
            f"First-frame spike survived sanitization: "
            f"{spiked[['trial_id', 'time_ms', 'velocity']].to_dict('records')}"
        )
        # ...while non-spiked trials' legitimate small onsets are preserved
        # (not nulled): they must retain their original <1000 cm/s value.
        other = df[(df["trial_id"] != 1) & (df["time_ms"] == 0.0)]
        assert float(other["velocity"].abs().min()) > 0.0, (
            "Non-spiked first frames were blanket-zeroed; legitimate onsets lost."
        )
        assert float(other["velocity"].abs().max()) < 1e3, (
            "Non-spiked first frame exceeds plausibility bound unexpectedly."
        )
        # Recomputed acceleration must also be finite/bounded (no diff-echo).
        assert np.isfinite(df["acceleration"]).all()
        # Physical bound: synthetic escape velocity jumps from ~0.2 to
        # ~15 cm/s in one dt=10 ms frame -> dv/dt ~ 1500 cm/s^2.  The
        # spiked artifact (5 M cm/s in one frame -> 5e8 cm/s^2) is the
        # only thing that should be removed; normal escape accelerations
        # are O(10^3) cm/s^2 and must be preserved.
        assert float(df["acceleration"].abs().max()) < 1e5, (
            f"Acceleration still carries an artifact-scale spike after recompute: "
            f"{df['acceleration'].abs().max()}"
        )

    def test_sanitize_first_frame_preserves_legitimate_onset(
        self, tmp_path: Path,
    ) -> None:
        """A realistic (sub-plausibility-bound) first-frame velocity survives.

        The artifact guard zeroes a trial-first frame ONLY when it exceeds
        the ``artifact_velocity_cm_s`` artifact bound (a cross-trial sensor
        jump).  A legitimate wind-onset escape — e.g. a genuine first-frame
        velocity of tens of cm/s — must be preserved, not blanket-zeroed,
        so real onset kinematics are not erased.
        """
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        raw = pd.read_csv(kin_path)
        first_idx = raw.index[(raw["trial_id"] == 1) & (raw["time_ms"] == 0.0)][0]
        raw.loc[first_idx, "velocity"] = 42.0  # realistic escape onset, cm/s
        raw.loc[first_idx, "acceleration"] = 42.0
        raw.to_csv(kin_path, index=False)

        df = load_kinematics_csv(kin_path)
        first_idx_loaded = df.index[(df["trial_id"] == 1) & (df["time_ms"] == 0.0)][0]
        assert df.loc[first_idx_loaded, "velocity"] == pytest.approx(42.0, abs=1e-6), (
            "Legitimate first-frame onset velocity was incorrectly zeroed."
        )


class TestLabeling:
    """Tests for pipeline.labeling module."""

    def test_assign_labels(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = []
        for t in range(10):
            trials.append(extract_trial_data(data, "session_0", t))

        labeled = assign_ground_truth_labels(trials)
        assert len(labeled) == 10

        labels = [info["label"] for info in labeled]
        # Trials 0-2 should be Escape (startle-like response)
        assert all(l == Label.ESCAPE for l in labels[:3])
        # Trials 6-7 should be Pre_Active
        assert all(l == Label.PRE_ACTIVE for l in labels[6:8])

    def test_responder_first_branch_order(self) -> None:
        """Round-3 BLK-3B regression test (root cause of PREWALK=0).

        A trial with BOTH a stimulus-locked escape burst AND sustained
        pre-stimulus walking must be labelled PREWALK.  Under the old
        pre-active-first order the baseline check ran before the
        response check and structurally absorbed every walking animal
        into PRE_ACTIVE — "walking at some point in baseline" is a
        strict superset of "walking in the last second before onset".
        """
        dt_ms = 10.0
        n_frames = 400
        time_ms = np.arange(n_frames) * dt_ms
        onset_ms = 2000.0
        stim_idx = int(onset_ms / dt_ms)

        # Continuous walking from t=0 through stimulus onset and beyond.
        velocity = np.full(n_frames, 1.5)
        # Stimulus-locked escape burst at ~100 ms latency, 350 ms long.
        velocity[stim_idx + 10:stim_idx + 45] = 8.0

        label = assign_ground_truth_labels([{
            "session_id": "s", "trial_id": 0,
            "time_ms": time_ms, "velocity": velocity,
            "event_types": np.array(["stimulus_onset"]),
            "event_times": np.array([onset_ms]),
        }])[0]["label"]
        assert label == Label.PREWALK, (
            "responder with sustained pre-stimulus walking must be "
            "PREWALK, got %s" % label.name
        )

    def test_walking_non_responder_is_pre_active(self) -> None:
        """Walking animal WITHOUT a stimulus-locked response is PRE_ACTIVE.

        Completes the Round-3 BLK-3B semantics: PRE_ACTIVE is reserved
        for non-responders; a responder that walked is PREWALK/ESCAPE.
        """
        dt_ms = 10.0
        n_frames = 400
        time_ms = np.arange(n_frames) * dt_ms
        onset_ms = 2000.0
        stim_idx = int(onset_ms / dt_ms)

        # Walking everywhere EXCEPT no post-stimulus burst: keep velocity
        # moderate (below escape threshold) after onset.
        velocity = np.full(n_frames, 1.5)
        velocity[stim_idx:] = 0.1  # animal stops at stimulus

        label = assign_ground_truth_labels([{
            "session_id": "s", "trial_id": 0,
            "time_ms": time_ms, "velocity": velocity,
            "event_types": np.array(["stimulus_onset"]),
            "event_times": np.array([onset_ms]),
        }])[0]["label"]
        assert label == Label.PRE_ACTIVE


class TestSnapshotExtraction:
    """Tests for data_extractor snapshot functions."""

    def test_extract_snapshot_shape(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])
        trial = extract_trial_data(data, "session_0", 0)

        snapshot = extract_mcmc_snapshot(trial, stimulus_onset_ms=2000.0)
        assert snapshot.shape == (5,)
        assert snapshot.dtype == np.float64

    def test_build_snapshot_dataset(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(10)]
        labeled = assign_ground_truth_labels(trials)

        snapshots, labels = build_snapshot_dataset(labeled)
        assert snapshots.shape == (10, 5)
        assert labels.shape == (10,)

    def test_visual_only_trial_survives_snapshot_extraction(
        self, tmp_path: Path,
    ) -> None:
        """
        A visual-only trial must produce a Snapshot, not vanish.

        Its stimulus onset is 0 because looming begins at trial start, so
        anchoring at ``stimulus_onset - 50 ms`` lands before frame one and
        the trial used to be swallowed by a bare ``except ValueError:
        continue``.  Measured on ``data/raw``: 36 of 396 trials dropped,
        every one visual-only, every one No_Response, zero visual-only
        trials surviving -- and the drop removes them from the regression
        sequence set too, not merely from the prior generator's input.
        """
        kin_path, evt_path = _make_visual_only_csvs(tmp_path, n_trials=3)
        data = load_and_concat_sessions([kin_path], [evt_path])
        trials = [extract_trial_data(data, "visual_session", t) for t in range(3)]
        labeled = assign_ground_truth_labels(trials)

        snapshots, labels, kept = build_snapshot_dataset(
            labeled, return_kept_indices=True,
        )

        assert len(kept) == 3, (
            "visual-only trials were dropped from the Snapshot dataset; "
            f"kept {len(kept)} of 3"
        )
        assert snapshots.shape == (3, 5)
        assert np.isfinite(snapshots).all()

    def test_visual_only_snapshot_is_anchored_at_the_collision(
        self, tmp_path: Path,
    ) -> None:
        """
        The visual-only anchor sits 50 ms before the looming collision.

        The worked 120 ms / 2 degree stimulus collides at 6874.795 ms.
        Expected feature values are sampled from that physical instant.
        """
        kin_path, evt_path = _make_visual_only_csvs(tmp_path, n_trials=1)
        data = load_and_concat_sessions([kin_path], [evt_path])
        trial = extract_trial_data(data, "visual_session", 0)

        time_ms = trial["time_ms"]
        collision_ms = 6874.795395691131
        expected_idx = int(np.argmin(np.abs(time_ms - (collision_ms - 50.0))))

        snapshot = extract_mcmc_snapshot(trial, stimulus_onset_ms=0.0)

        assert snapshot[0] == pytest.approx(
            float(trial["visual_angle"][expected_idx])
        )
        assert snapshot[2] == 0.0, "visual-only trial must report no wind"

    def test_wind_bearing_snapshot_stays_anchored_at_stimulus_onset(
        self, tmp_path: Path,
    ) -> None:
        """
        Trials carrying wind keep the stimulus-onset anchor, bit for bit.

        The collision fix is deliberately confined to the visual-only case:
        it is the minimal edit to a frozen module, and it leaves the 288
        multisensory plus 72 pure-wind Snapshots of the real corpus
        unchanged.  This test is the guard rail on that promise.
        """
        kin_path, evt_path = _make_pure_wind_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])
        trial = extract_trial_data(data, "wind_session", 0)

        time_ms = trial["time_ms"]
        stimulus_onset_ms = 2000.0
        expected_idx = int(
            np.argmin(np.abs(time_ms - (stimulus_onset_ms - 50.0)))
        )

        snapshot = extract_mcmc_snapshot(
            trial, stimulus_onset_ms=stimulus_onset_ms,
        )

        assert snapshot[0] == pytest.approx(
            float(trial["visual_angle"][expected_idx])
        )
        assert snapshot[2] == pytest.approx(
            float(trial["wind_state"][expected_idx])
        )


class TestSequenceExtraction:
    """Tests for data_extractor sequence functions."""

    def test_extract_sequence_shape(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])
        trial = extract_trial_data(data, "session_0", 0)

        X_seq, Y_seq = extract_trial_sequence(trial)
        assert X_seq.shape == (400, 8)
        assert Y_seq.shape == (400,)

        # First frame: t-1 features should be zero
        assert X_seq[0, 2] == 0.0  # v_kine(-1)
        assert X_seq[0, 3] == 0.0  # a_kine(-1)

    def test_build_sequence_dataset(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(10)]
        labeled = assign_ground_truth_labels(trials)

        sequences = build_sequence_dataset(labeled)
        assert len(sequences) == 10
        X_seq, Y_seq, label = sequences[0]
        assert X_seq.shape[1] == 8

    def test_pure_wind_baseline_prepend(self, tmp_path: Path) -> None:
        """Pure Wind trials get 570 zero-frames prepended."""
        kin_path, evt_path = _make_pure_wind_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])
        trial = extract_trial_data(data, "wind_session", 0)

        X_seq, Y_seq = extract_trial_sequence(trial)
        expected_len = PURE_WIND_PREPEND_FRAMES + 400  # 570 + 400 = 970
        assert X_seq.shape == (expected_len, 8), (
            f"Expected ({expected_len}, 8), got {X_seq.shape}"
        )
        assert Y_seq.shape == (expected_len,)

        # Prepended region should be all zeros
        assert np.all(X_seq[:PURE_WIND_PREPEND_FRAMES, :] == 0.0)
        assert np.all(Y_seq[:PURE_WIND_PREPEND_FRAMES] == 0.0)

        # Original region should have non-zero wind after stimulus
        assert np.any(X_seq[PURE_WIND_PREPEND_FRAMES:, 1] != 0.0)

    def test_pure_wind_baseline_prepend_dynamic_dt(self, tmp_path: Path) -> None:
        """Pure Wind trials dynamically scale prepend frames to dt_ms."""
        # 1. Direct function calculation
        assert _compute_pure_wind_prepend_frames(10.0) == 570
        assert _compute_pure_wind_prepend_frames(4.0) == 1425
        with pytest.raises(ValueError, match="positive"):
            _compute_pure_wind_prepend_frames(-1.0)

        # 2. Sequence extraction with explicit dt_ms
        kin_path, evt_path = _make_pure_wind_csvs(tmp_path, frames_per_trial=600, dt_ms=4.0)
        data = load_and_concat_sessions([kin_path], [evt_path])
        trial = extract_trial_data(data, "wind_session", 0)

        X_seq, Y_seq = extract_trial_sequence(trial, dt_ms=4.0)
        expected_prepend = 1425
        expected_len = expected_prepend + 600
        assert X_seq.shape == (expected_len, 8)
        assert Y_seq.shape == (expected_len,)
        assert np.all(X_seq[:expected_prepend, :] == 0.0)
        assert np.all(Y_seq[:expected_prepend] == 0.0)
        assert np.any(X_seq[expected_prepend:, 1] != 0.0)


class TestMirrorToRight:
    """Tests for wind-side mirroring (NSMoR-pond reviewer #2 gating)."""

    def _left_trial(self, T: int = 20) -> dict:
        t = np.arange(T, dtype=np.float64) * 10.0
        return {
            "time_ms": t,
            "x_pos": np.linspace(1.0, 5.0, T, dtype=np.float64),
            "y_pos": np.linspace(0.0, 2.0, T, dtype=np.float64),
            "heading": np.linspace(0.0, 90.0, T, dtype=np.float64),
            "velocity": np.random.uniform(0, 1, T).astype(np.float64),
            "acceleration": np.zeros(T, dtype=np.float64),
            "visual_angle": np.zeros(T, dtype=np.float64),
            "wind_state": np.ones(T, dtype=np.float64),
            "l_v_ratio": np.zeros(T, dtype=np.float64),
            "event_types": np.array(["wind_onset"], dtype=object),
            "event_values": np.array(["{'wind_side': 'left'}"] * T, dtype=object),
            "wind_side_original": "left",
            "wind_side_unified": "left",
            "session_id": "s0",
            "trial_id": 0,
            "dx": np.ones(T, dtype=np.float64),
            "dz": np.ones(T, dtype=np.float64) * 10.0,
            "vel_x": np.ones(T, dtype=np.float64) * 2.0,
        }

    def _right_trial(self, T: int = 20) -> dict:
        t = self._left_trial(T)
        t["wind_side_original"] = "right"
        t["wind_side_unified"] = "right"
        t["event_values"] = np.array(["{'wind_side': 'right'}"] * T, dtype=object)
        return t

    def test_left_flip_and_heading(self) -> None:
        trial = self._left_trial()
        x_before = trial["x_pos"].copy()
        h_before = trial["heading"].copy()
        mirrored = mirror_to_right(trial)
        # x, dx, dz, vel_x sign flipped
        np.testing.assert_allclose(mirrored["x_pos"], -x_before)
        np.testing.assert_allclose(mirrored["dx"], -trial["dx"])
        np.testing.assert_allclose(mirrored["dz"], -trial["dz"])
        np.testing.assert_allclose(mirrored["vel_x"], -trial["vel_x"])
        # heading = (-h) % 360
        np.testing.assert_allclose(mirrored["heading"], (-h_before) % 360.0)
        # y invariant
        np.testing.assert_allclose(mirrored["y_pos"], trial["y_pos"])
        assert mirrored["wind_side_original"] == "left"
        assert mirrored["wind_side_unified"] == "right"
        assert mirrored["wind_side_mirrored"] is True
        # shape assertions hold
        assert mirrored["x_pos"].shape == mirrored["time_ms"].shape

    def test_right_no_flip(self) -> None:
        trial = self._right_trial()
        x_before = trial["x_pos"].copy()
        mirrored = mirror_to_right(trial)
        np.testing.assert_allclose(mirrored["x_pos"], x_before)
        assert mirrored["wind_side_mirrored"] is False
        assert mirrored["wind_side_unified"] == "right"

    def test_idempotent_left(self) -> None:
        trial = self._left_trial()
        m1 = mirror_to_right(trial)
        m2 = mirror_to_right(m1)
        np.testing.assert_allclose(m2["x_pos"], m1["x_pos"])
        np.testing.assert_allclose(m2["heading"], m1["heading"])
        assert m2["wind_side_mirrored"] is True

    def test_idempotent_right(self) -> None:
        trial = self._right_trial()
        m1 = mirror_to_right(trial)
        m2 = mirror_to_right(m1)
        np.testing.assert_allclose(m2["x_pos"], m1["x_pos"])
        assert m2["wind_side_mirrored"] is False

    def test_missing_optional_fields_no_crash(self) -> None:
        trial = self._left_trial()
        for k in ("dx", "dz", "vel_x"):
            trial.pop(k, None)
        mirrored = mirror_to_right(trial)
        # still flips x_pos/heading
        assert mirrored["wind_side_mirrored"] is True
        assert "dx" not in mirrored

    def test_deepcopy_isolation(self) -> None:
        trial = self._left_trial()
        mirrored = mirror_to_right(trial)
        mirrored["x_pos"][0] = 999.0
        assert trial["x_pos"][0] != 999.0
        # object array isolation
        assert mirrored["event_values"] is not trial["event_values"]

    def test_demirror_identity_scalar_speed(self) -> None:
        pred = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        out_left = demirror_prediction(pred, "left")
        out_right = demirror_prediction(pred, "right")
        np.testing.assert_allclose(out_left, pred)
        np.testing.assert_allclose(out_right, pred)
        assert out_left is not pred  # copy

    def test_demirror_shape_assert(self) -> None:
        pred2d = np.zeros((2, 10), dtype=np.float64)
        out = demirror_prediction(pred2d, "left")
        assert out.shape == (2, 10)
        import pytest as _pytest
        with _pytest.raises(AssertionError):
            demirror_prediction(np.zeros((2, 3, 4)), "left")

    def test_parse_event_value_single_quotes(self) -> None:
        parsed = _parse_event_value("{'wind_side': 'left'}")
        assert parsed.get("wind_side") == "left"
        # strict JSON also works
        parsed2 = _parse_event_value('{"wind_side": "right"}')
        assert parsed2.get("wind_side") == "right"
        # NaN / empty -> {}
        assert _parse_event_value(float("nan")) == {}
        assert _parse_event_value("") == {}

    def test_unknown_side_no_flip(self) -> None:
        trial = self._left_trial()
        trial["wind_side_original"] = "unknown"
        trial["wind_side_unified"] = "unknown"
        mirrored = mirror_to_right(trial)
        assert mirrored["wind_side_mirrored"] is False
        assert mirrored["wind_side_unified"] == "unknown"


class TestMCMCModule:
    """Tests for mcmc_module."""

    def test_pytorch_forward_shape(self) -> None:
        model = MCMCPriorGenerator()
        x = torch.randn(3, 5)
        probs = model(x)
        assert probs.shape == (3, 4)
        assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)

    def test_pytorch_single_sample(self) -> None:
        model = MCMCPriorGenerator()
        x = torch.randn(5)
        probs = model(x)
        assert probs.shape == (4,)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_predict_proba_numpy(self) -> None:
        model = MCMCPriorGenerator()
        x = np.random.randn(5).astype(np.float64)
        probs = model.predict_proba(x)
        assert probs.shape == (4,)
        assert np.allclose(probs.sum(), 1.0, atol=1e-5)

    def test_train_mcmc(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(10)]
        labeled = assign_ground_truth_labels(trials)
        snapshots, labels = build_snapshot_dataset(labeled)

        model = train_mcmc(
            snapshots, labels,
            config=MCMCTrainingConfig(num_epochs=50),
            verbose=False,
        )
        probs = model.predict_proba(snapshots[0])
        assert probs.shape == (4,)
        assert np.allclose(probs.sum(), 1.0, atol=1e-5)

    def test_sklearn_wrapper(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(10)]
        labeled = assign_ground_truth_labels(trials)
        snapshots, labels = build_snapshot_dataset(labeled)

        model = MCMCPriorSKLearn()
        model.fit(snapshots, labels)
        probs = model.predict_proba(snapshots[0])
        n_classes = len(np.unique(labels))
        assert probs.shape[0] == n_classes, (
            f"probs.shape[0]={probs.shape[0]} != n_classes={n_classes}"
        )
        assert np.allclose(probs.sum(), 1.0, atol=1e-5)

    def test_markov_transition(self) -> None:
        estimator = MarkovTransitionEstimator(num_states=4)
        seq = np.array([0, 0, 1, 1, 2, 3, 3, 3])
        estimator.fit([seq])
        assert estimator.transition_matrix is not None
        assert estimator.transition_matrix.shape == (4, 4)
        # Rows should sum to 1
        assert np.allclose(
            estimator.transition_matrix.sum(axis=1), 1.0, atol=1e-10,
        )


class TestNSMoRDataLoader:
    """Tests for nsmor_dataloader."""

    def test_dataloader_with_priors(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(10)]
        labeled = assign_ground_truth_labels(trials)
        snapshots, labels = build_snapshot_dataset(labeled)
        sequences = build_sequence_dataset(labeled)

        # Train MCMC and get priors
        model = train_mcmc(
            snapshots, labels,
            config=MCMCTrainingConfig(num_epochs=50),
            verbose=False,
        )
        priors = model.predict_proba(snapshots)

        loader = create_dataloader(
            sequences, mcmc_priors=priors, batch_size=4,
        )

        for X_batch, Y_batch, lengths in loader:
            bs = X_batch.shape[0]
            seq_len = X_batch.shape[1]
            assert X_batch.shape == (bs, seq_len, 8), (
                f"X_batch shape {X_batch.shape} != (bs, seq_len, 8)"
            )
            assert Y_batch.shape == (bs, seq_len), (
                f"Y_batch shape {Y_batch.shape} != (bs, seq_len)"
            )
            assert lengths.shape == (bs,), (
                f"lengths shape {lengths.shape} != (bs,)"
            )
            assert lengths.dtype == torch.int64
            assert (lengths <= seq_len).all(), (
                f"Some lengths exceed seq_len: {lengths}"
            )

            # MCMC probabilities should sum to 1
            mcmc = X_batch[:, :, 4:8]
            sums = mcmc.sum(dim=2)
            assert torch.allclose(sums, torch.ones(bs, seq_len), atol=1e-4), (
                f"MCMC prob sums: min={sums.min():.6f} max={sums.max():.6f}"
            )
            break  # one batch is enough

    def test_dataloader_requires_priors(self, tmp_path: Path) -> None:
        """NSMoRDataset raises ValueError when mcmc_priors is None."""
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(3)]
        labeled = assign_ground_truth_labels(trials)
        sequences = build_sequence_dataset(labeled)

        with pytest.raises(ValueError, match="mcmc_priors is required"):
            create_dataloader(sequences, mcmc_priors=None, batch_size=2)


class TestNSMoRModel:
    """Tests for the NSMoR Mixture-of-Recursions network."""

    def test_forward_shape(self) -> None:
        B, T, H = 4, 100, 32
        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=H)
        X_batch = torch.randn(B, T, 8)
        lengths = torch.tensor([100, 80, 50, 20], dtype=torch.int64)

        model.eval()
        with torch.no_grad():
            Y_pred = model(X_batch, lengths)

        assert Y_pred.shape == (B, T), (
            f"Y_pred shape {Y_pred.shape} != ({B}, {T})"
        )

    def test_forward_variable_lengths(self) -> None:
        """Different length sequences produce correct shapes."""
        B, T, H = 2, 60, 16
        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=H)
        X_batch = torch.randn(B, T, 8)
        lengths = torch.tensor([60, 30], dtype=torch.int64)

        model.eval()
        with torch.no_grad():
            Y_pred = model(X_batch, lengths)

        assert Y_pred.shape == (B, T)

    def test_forward_single_sample(self) -> None:
        """Batch size 1 works correctly."""
        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=16)
        X = torch.randn(1, 50, 8)
        lengths = torch.tensor([50], dtype=torch.int64)

        model.eval()
        with torch.no_grad():
            Y = model(X, lengths)

        assert Y.shape == (1, 50)

    def test_gradient_flow(self) -> None:
        """Gradients flow through both LIF and GRU paths."""
        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=16)
        X = torch.randn(2, 40, 8, requires_grad=True)
        lengths = torch.tensor([40, 20], dtype=torch.int64)

        Y = model(X, lengths)
        loss = Y.sum()
        loss.backward()

        assert X.grad is not None
        assert X.grad.shape == (2, 40, 8)
        # Valid frames should have non-zero gradient
        assert X.grad[:, :20, :].abs().sum() > 0

    def test_invalid_feature_dim_raises(self) -> None:
        """ValueError when feature dim != 8."""
        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=16)
        X_bad = torch.randn(2, 40, 6)  # wrong dim
        lengths = torch.tensor([40, 20], dtype=torch.int64)

        with pytest.raises(ValueError, match="Expected feature dim 8"):
            model(X_bad, lengths)

    def test_end_to_end_pipeline(self, tmp_path: Path) -> None:
        """Full pipeline: CSV → DataLoader → NSMoR forward."""
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(10)]
        labeled = assign_ground_truth_labels(trials)
        snapshots, labels = build_snapshot_dataset(labeled)
        sequences = build_sequence_dataset(labeled)

        mcmc_model = train_mcmc(
            snapshots, labels,
            config=MCMCTrainingConfig(num_epochs=50),
            verbose=False,
        )
        priors = mcmc_model.predict_proba(snapshots)

        loader = create_dataloader(
            sequences, mcmc_priors=priors, batch_size=4,
        )

        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for X_batch, Y_batch, lengths in loader:
            optimizer.zero_grad()
            Y_pred = model(X_batch, lengths)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            assert loss.isfinite(), f"Loss is not finite: {loss}"
            break  # one step is enough

    def test_end_to_end_with_pure_wind(self, tmp_path: Path) -> None:
        """Pure Wind trials (with 570-frame prepend) flow through model."""
        kin_path, evt_path = _make_pure_wind_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trial = extract_trial_data(data, "wind_session", 0)
        X_seq, Y_seq = extract_trial_sequence(trial)

        # Should be 570 + 400 = 970 frames
        assert X_seq.shape[0] == 970

        # Wrap in a minimal dataset with dummy priors
        priors = np.array([[0.1, 0.1, 0.7, 0.1]], dtype=np.float64)
        sequences = [(X_seq, Y_seq, Label.NO_RESPONSE)]

        loader = create_dataloader(sequences, mcmc_priors=priors, batch_size=1)

        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=16)
        model.eval()
        for X_batch, Y_batch, lengths in loader:
            with torch.no_grad():
                Y_pred = model(X_batch, lengths)
            assert Y_pred.shape == (1, 970)
            break


# Need nn import for the end-to-end test
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════
# Labeling Funnel Persistence & Threshold Sensitivity (T4)
# ═══════════════════════════════════════════════════════════════


def _make_label_audit_csvs(
    raw_dir: Path,
    *,
    n_sessions: int = 5,
    frames_per_trial: int = 400,
    dt_ms: float = 10.0,
) -> None:
    """Write deterministic trials with one near the escape threshold."""
    onset_ms = 2000.0
    stim_idx = int(onset_ms / dt_ms)
    time_ms = np.arange(frames_per_trial, dtype=np.float64) * dt_ms
    profiles = (
        (0.1, 5.5),   # ESCAPE by default, NO_RESPONSE at 1.25x.
        (0.1, 15.0),  # ESCAPE at every audited threshold.
        (1.5, 15.0),  # PREWALK at every audited threshold.
        (0.8, 0.1),   # PRE_ACTIVE at every audited threshold.
        (0.1, 0.1),   # NO_RESPONSE at every audited threshold.
    )

    for session_idx in range(n_sessions):
        # Distinct ANIMALS, one recording block each -- mirroring real ids
        # (``<mass>cricket_001_<date>_<time>_session_<N>``).  The previous
        # name ``audit_session_{i}`` looked like N sessions but was N
        # blocks of ONE animal named "audit": animal grouping strips the
        # ``_session_N`` suffix, so cross-fitting saw a single group and
        # correctly refused to split.
        session_id = f"0.50{session_idx}cricket_001_20260101_00000{session_idx}_session_1"
        session_dir = raw_dir / session_id
        session_dir.mkdir(parents=True)
        kin_rows: list[dict[str, object]] = []
        event_rows: list[dict[str, object]] = []

        for trial_id, (baseline_velocity, response_velocity) in enumerate(
            profiles,
        ):
            velocity = np.full(
                frames_per_trial, baseline_velocity, dtype=np.float64,
            )
            velocity[stim_idx:] = 0.1
            velocity[stim_idx + 5:stim_idx + 45] = response_velocity
            acceleration = np.gradient(velocity, dt_ms / 1000.0)
            visual_angle = np.zeros(frames_per_trial, dtype=np.float64)
            visual_angle[stim_idx:] = np.linspace(
                5.0, 60.0, frames_per_trial - stim_idx,
            )

            for frame_idx in range(frames_per_trial):
                kin_rows.append({
                    "session_id": session_id,
                    "trial_id": trial_id,
                    "time_ms": float(time_ms[frame_idx]),
                    "x_pos": float(frame_idx * 0.01),
                    "y_pos": float(frame_idx * 0.005),
                    "heading": 0.0,
                    "velocity": float(velocity[frame_idx]),
                    "acceleration": float(acceleration[frame_idx]),
                    "visual_angle": float(visual_angle[frame_idx]),
                    "wind_state": float(frame_idx >= stim_idx),
                    "l_v_ratio": float(visual_angle[frame_idx] * 0.1),
                })
            event_rows.extend((
                {
                    "session_id": session_id,
                    "trial_id": trial_id,
                    "time_ms": 0.0,
                    "event_type": "trial_start",
                    "event_value": 1,
                },
                {
                    "session_id": session_id,
                    "trial_id": trial_id,
                    "time_ms": onset_ms,
                    "event_type": "stimulus_onset",
                    "event_value": 1,
                },
            ))

        pd.DataFrame(kin_rows).to_csv(
            session_dir / "kinematics.csv", index=False,
        )
        pd.DataFrame(event_rows).to_csv(
            session_dir / "events.csv", index=False,
        )


def _extract_label_audit_trials(raw_dir: Path) -> list[dict]:
    """Load fixture CSVs independently of ``prepare_dataset`` persistence."""
    kin_paths: list[Path] = []
    evt_paths: list[Path] = []
    for session_dir in sorted(raw_dir.iterdir()):
        if session_dir.is_dir():
            kin_paths.append(session_dir / "kinematics.csv")
            evt_paths.append(session_dir / "events.csv")
    session_data = load_and_concat_sessions(kin_paths, evt_paths)
    trials: list[dict] = []
    for (session_id, trial_id), _ in session_data["kinematics"].groupby(
        ["session_id", "trial_id"],
    ):
        trials.append(extract_trial_data(session_data, session_id, trial_id))
    return trials


def _expected_scaled_thresholds(scale: float) -> dict[str, float]:
    """Independently expected velocity thresholds at a sensitivity scale."""
    return {
        "escape_velocity_threshold": (
            DEFAULT_THRESHOLD.escape_velocity_threshold * scale
        ),
        "prewalk_velocity_threshold": (
            DEFAULT_THRESHOLD.prewalk_velocity_threshold * scale
        ),
        "pre_active_velocity_threshold": (
            DEFAULT_THRESHOLD.pre_active_velocity_threshold * scale
        ),
    }


def _expected_sensitivity_record(
    trials: list[dict],
    scale: float,
) -> dict[str, object]:
    """Independently expected self-describing sensitivity record."""
    class_schema = [label.name for label in Label]
    cfg = dataclasses.replace(
        DEFAULT_THRESHOLD,
        escape_velocity_threshold=(
            DEFAULT_THRESHOLD.escape_velocity_threshold * scale
        ),
        prewalk_velocity_threshold=(
            DEFAULT_THRESHOLD.prewalk_velocity_threshold * scale
        ),
        pre_active_velocity_threshold=(
            DEFAULT_THRESHOLD.pre_active_velocity_threshold * scale
        ),
    )
    labeled = assign_ground_truth_labels(
        trials, config=cfg, return_funnel=True,
    )
    counts = {name: 0 for name in class_schema}
    for info in labeled:
        counts[info["label"].name] += 1
    return {
        "scale": scale,
        "n_trials": len(labeled),
        "class_schema": class_schema,
        "counts": counts,
        "thresholds": _expected_scaled_thresholds(scale),
        "labeling_funnel": labeling_funnel_summary(labeled),
    }


def test_stimulus_condition_keys_in_dataset(tmp_path: Path) -> None:
    """Ticket #13: ETL produces is_pure_wind derived from stimulus_conditions.

    Full end-to-end test is deferred until next data regeneration; this
    verifies the classification logic at least.
    """
    from scripts.prepare_data import classify_stimulus_condition

    # Verify multiple condition types
    examples = [
        ({"visual_angle": np.zeros(10), "wind_state": np.zeros(10)}, "no_stimulus"),
        ({"visual_angle": np.array([0, 5, 10]), "wind_state": np.zeros(3)}, "visual_only"),
        ({"visual_angle": np.zeros(5), "wind_state": np.array([0, 1, 1, 0, 0])}, "wind_only"),
        ({"visual_angle": np.array([0, 5, 10]), "wind_state": np.array([0, 1, 1])}, "multisensory"),
    ]

    for trial_data, expected in examples:
        result = classify_stimulus_condition(trial_data)
        assert result == expected, (
            f"Expected {expected}, got {result} for trial {trial_data}"
        )


def test_stimulus_condition_schema_extension() -> None:
    """Ticket #13: is_pure_wind and stimulus_conditions survive ETL."""
    from scripts.prepare_data import classify_stimulus_condition

    # Mock trial data representing different conditions
    visual_only = {
        "visual_angle": np.array([0.0, 5.0, 10.0]),
        "wind_state": np.array([0.0, 0.0, 0.0]),
    }
    wind_only = {
        "visual_angle": np.array([0.0, 0.0, 0.0]),
        "wind_state": np.array([0.0, 1.0, 1.0]),
    }
    multisensory = {
        "visual_angle": np.array([0.0, 5.0, 10.0]),
        "wind_state": np.array([0.0, 1.0, 1.0]),
    }
    no_stimulus = {
        "visual_angle": np.array([0.0, 0.0, 0.0]),
        "wind_state": np.array([0.0, 0.0, 0.0]),
    }

    assert classify_stimulus_condition(visual_only) == "visual_only"
    assert classify_stimulus_condition(wind_only) == "wind_only"
    assert classify_stimulus_condition(multisensory) == "multisensory"
    assert classify_stimulus_condition(no_stimulus) == "no_stimulus"


def test_label_audit_metadata_round_trips_through_artifact(
    tmp_path: Path,
) -> None:
    """Persist exact funnel summary and self-describing sensitivity records."""
    from scripts.prepare_data import prepare_dataset

    raw_dir = tmp_path / "raw"
    _make_label_audit_csvs(raw_dir)
    independent_trials = _extract_label_audit_trials(raw_dir)
    expected_funnel = labeling_funnel_summary(
        assign_ground_truth_labels(
            independent_trials, return_funnel=True,
        )
    )
    expected_retention = {
        "n_prefilter_labeled_trials": len(independent_trials),
        "n_retained_sequences": len(independent_trials),
        "n_dropped_before_snapshot": 0,
        "n_dropped_during_sequence_extraction": 0,
        # Totals alone once hid a 100%-single-condition loss: 36 of 396
        # trials dropped on the real corpus, every one visual-only and
        # every one No_Response, with nothing in any artefact saying so.
        # The breakdowns are empty here because nothing was dropped, and
        # asserting that explicitly is the point.
        "snapshot_drops_by_class": {},
        "snapshot_drops_by_condition": {},
        "snapshot_anchor_rules": {"stimulus_onset": len(independent_trials)},
    }
    expected_sensitivity = {
        f"thresholds_x{scale:.2f}": _expected_sensitivity_record(
            independent_trials, scale,
        )
        for scale in (0.75, 1.0, 1.25)
    }

    output = tmp_path / "dataset.pt"
    prepare_dataset(raw_dir=raw_dir, output_path=output, random_seed=42)
    dataset = torch.load(output, weights_only=False)

    assert dataset["labeling_funnel"] == expected_funnel
    assert dataset["labeling_funnel_retention"] == expected_retention
    assert dataset["mcmc_priors"].shape == (25, 4)
    assert dataset["mcmc_prior_provenance"] == "oof_5fold_recording_prefix_grouped_cv"
    assert dataset["animal_identity_status"] == "unverified"
    assert len(dataset["mcmc_fold_models"]) == 5
    assert dataset["mcmc_prior_train_serve_consistency"]["n_trials"] == 25
    assert "n_retained_sequences" not in dataset["labeling_funnel"]
    assert dataset["labeling_threshold_sensitivity"] == expected_sensitivity
    assert (
        expected_sensitivity["thresholds_x1.25"]["counts"]
        != expected_sensitivity["thresholds_x1.00"]["counts"]
    )


def test_etl_releases_raw_trials_before_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The save must not retain raw trials alongside all sequence arrays."""
    from scripts import prepare_data as prep
    from nsmor.pipeline.conditions import derive_anchor_frames

    raw_dir = tmp_path / "raw"
    _make_label_audit_csvs(raw_dir, n_sessions=5)
    # Independently construct the causal 4ms grid, then compare the
    # model-facing float32 cast. Source observations remain on their own axis.
    expected = {}
    for trial in _extract_label_audit_trials(raw_dir):
        source_time = trial["time_ms"]
        grid = source_time[0] + np.arange(
            int(np.floor((source_time[-1] - source_time[0]) / 4.0)) + 1
        ) * 4.0
        indices = np.searchsorted(source_time, grid, side="right") - 1
        model_trial = dict(trial)
        for channel in ("visual_angle", "wind_state", "velocity", "acceleration"):
            model_trial[channel] = trial[channel][indices]
        model_trial["time_ms"] = grid
        x64, y64 = extract_trial_sequence(model_trial, dt_ms=4.0)
        expected[(trial["session_id"], trial["trial_id"])] = (
            x64.copy(), y64.copy(), derive_anchor_frames([x64])[0],
        )
    raw_refs: list[weakref.ReferenceType[np.ndarray]] = []
    original_loader = prep._load_trials_and_diagnostics
    original_save = prep.torch.save
    save_calls = 0

    def record_raw_arrays(*args, **kwargs):
        result = original_loader(*args, **kwargs)
        raw_refs.extend(weakref.ref(trial["time_ms"]) for trial in result[0])
        return result

    def check_save(dataset, path, **kwargs):
        nonlocal save_calls
        save_calls += 1
        assert len(raw_refs) == 25
        assert all(ref() is None for ref in raw_refs), (
            "Raw trial arrays remained live when serializing X_seqs/Y_seqs"
        )
        assert len(dataset["X_seqs"]) == len(dataset["Y_seqs"]) == 25
        for key in ("X_seqs", "Y_seqs"):
            assert all(isinstance(seq, torch.Tensor) and seq.dtype == torch.float32
                       and seq.device.type == "cpu" for seq in dataset[key])
        assert kwargs.get("_use_new_zipfile_serialization", True) is True
        return original_save(dataset, path, **kwargs)

    monkeypatch.setattr(prep, "_load_trials_and_diagnostics", record_raw_arrays)
    monkeypatch.setattr(prep.torch, "save", check_save)
    output = tmp_path / "dataset.pt"
    prep.prepare_dataset(raw_dir=raw_dir, output_path=output, random_seed=42)

    assert save_calls == 1
    import hashlib
    from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint

    dataset, fingerprint = load_dataset_with_fingerprint(output)
    assert fingerprint == hashlib.sha256(output.read_bytes()).hexdigest()
    assert len(dataset["X_seqs"]) == 25
    assert dataset["labeling_funnel_retention"]["n_retained_sequences"] == 25
    assert zipfile.is_zipfile(output)
    assert len(dataset["labeling_eligibility"]) == 25
    assert all(item["status"] == "labeled" for item in dataset["labeling_eligibility"])
    assert len(dataset["model_grid_provenance"]) == 25
    assert dataset["model_dt_ms"] == 4.0
    assert len(dataset["anchor_frames"]) == 25
    assert all(isinstance(x, np.ndarray) and x.dtype == np.float32
               for x in dataset["X_seqs"])
    assert all(isinstance(y, np.ndarray) and y.dtype == np.float32
               for y in dataset["Y_seqs"])
    old_sequences = []
    new_sequences = []
    for sid, tid, x, y, anchor, label in zip(
        dataset["session_ids"], dataset["trial_ids"], dataset["X_seqs"],
        dataset["Y_seqs"], dataset["anchor_frames"], dataset["labels"],
    ):
        expected_x, expected_y, expected_anchor = expected[(str(sid), int(tid))]
        np.testing.assert_array_equal(
            x, torch.as_tensor(expected_x, dtype=torch.float32).numpy(),
        )
        np.testing.assert_array_equal(
            y, torch.as_tensor(expected_y, dtype=torch.float32).numpy(),
        )
        assert anchor == expected_anchor
        old_sequences.append((expected_x, expected_y, int(label)))
        new_sequences.append((x, y, int(label)))

    # Exercise the actual DataLoader conversion and prior filling on both
    # representations, comparing every trial at the model-facing boundary.
    from nsmor.nsmor_dataloader import NSMoRDataset
    old_loader = NSMoRDataset(old_sequences, dataset["mcmc_priors"], max_seq_len=None)
    new_loader = NSMoRDataset(new_sequences, dataset["mcmc_priors"], max_seq_len=None)
    for i in range(25):
        old_x, old_y = old_loader[i]
        new_x, new_y = new_loader[i]
        torch.testing.assert_close(new_x, old_x, rtol=0, atol=0)
        torch.testing.assert_close(new_y, old_y, rtol=0, atol=0)


@pytest.mark.parametrize("channel", ["X", "Y"])
def test_etl_rejects_float32_overflow_before_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, channel: str,
) -> None:
    from scripts import prepare_data as prep

    raw_dir = tmp_path / "raw"
    _make_label_audit_csvs(raw_dir, n_sessions=5)
    original_extract = prep.extract_trial_sequence

    def overflow_sequence(*args, **kwargs):
        x, y = original_extract(*args, **kwargs)
        if channel == "X":
            x[0, 2] = 1e40
        else:
            y[0] = 1e40
        return x, y

    monkeypatch.setattr(prep, "extract_trial_sequence", overflow_sequence)
    output = tmp_path / "overflow.pt"
    with pytest.raises(AssertionError, match=f"non-finite {channel}"):
        prep.prepare_dataset(raw_dir=raw_dir, output_path=output, random_seed=42)
    assert not output.exists()


def test_production_rejects_unanchorable_before_save(tmp_path: Path) -> None:
    """An unanchorable trial fails with identity, before a dataset can be saved."""
    from scripts.prepare_data import prepare_dataset

    raw_dir = tmp_path / "raw_with_drops"
    # Use the same fixture generator with >=5 sessions for 5-fold cross-fitting
    _make_label_audit_csvs(raw_dir, n_sessions=5, frames_per_trial=400)

    # Add one session with trials that fail filters:
    # - Trial with insufficient post-stim frames (will fail snapshot gate)
    # - Trial with malformed kinematics (will fail extraction)
    session_id = "drop_session"
    session_dir = raw_dir / session_id
    session_dir.mkdir(parents=True)

    # Short trial: only 50 frames, stimulus at 20 ms -> snapshot at -30 ms < 0 ms
    short_trial_kin = pd.DataFrame({
        "session_id": [session_id] * 50,
        "trial_id": [999] * 50,
        "time_ms": np.arange(50) * 10.0,
        "x_pos": np.linspace(0, 0.5, 50),
        "y_pos": np.linspace(0, 0.5, 50),
        "heading": np.zeros(50),
        "velocity": np.full(50, 5.0),
        "acceleration": np.zeros(50),
        "visual_angle": np.zeros(50),
        "wind_state": np.zeros(50, dtype=int),
        "l_v_ratio": np.zeros(50),
    })
    short_trial_evt = pd.DataFrame({
        "session_id": [session_id, session_id],
        "trial_id": [999, 999],
        "time_ms": [0.0, 20.0],  # stimulus at 20 ms -> snapshot time before trial start
        "event_type": ["trial_start", "stimulus_onset"],
        "event_value": [1, 1],
    })

    # Combine with existing sessions' data (updating session_id).  Must
    # track the id scheme _make_label_audit_csvs writes.
    first_session = "0.500cricket_001_20260101_000000_session_1"
    existing_kin = pd.read_csv(raw_dir / first_session / "kinematics.csv")
    existing_kin["session_id"] = session_id
    combined_kin = pd.concat([existing_kin, short_trial_kin], ignore_index=True)
    combined_kin.to_csv(session_dir / "kinematics.csv", index=False)

    existing_evt = pd.read_csv(raw_dir / first_session / "events.csv")
    existing_evt["session_id"] = session_id
    combined_evt = pd.concat([existing_evt, short_trial_evt], ignore_index=True)
    combined_evt.to_csv(session_dir / "events.csv", index=False)

    output = tmp_path / "dataset_drops.pt"
    with pytest.raises(ValueError, match=r"drop_session.*999.*could not be anchored"):
        prepare_dataset(raw_dir=raw_dir, output_path=output, random_seed=42)
    assert not output.exists()


# ═══════════════════════════════════════════════════════════════
# CLI Override Tests (Flaw #2: regression guard for build_config)
# ═══════════════════════════════════════════════════════════════

class TestCLIOverrides:
    """Verify that build_config correctly applies CLI overrides."""

    def test_freeze_modules_override(self):
        """--freeze lif_cell router sets config.finetune.freeze_modules."""
        from scripts.train import build_config
        config, _, _ = build_config(["--freeze", "lif_cell", "router"])
        assert config.finetune.freeze_modules == ["lif_cell", "router"]

    def test_lr_override(self):
        """--lr 5e-4 sets training.learning_rate."""
        from scripts.train import build_config
        config, _, _ = build_config(["--lr", "5e-4"])
        assert config.training.learning_rate == 5e-4

    def test_epochs_override(self):
        """--epochs 200 sets training.num_epochs."""
        from scripts.train import build_config
        config, _, _ = build_config(["--epochs", "200"])
        assert config.training.num_epochs == 200

    def test_hidden_dim_override(self):
        """--hidden_dim 128 sets model.hidden_dim."""
        from scripts.train import build_config
        config, _, _ = build_config(["--hidden_dim", "128"])
        assert config.model.hidden_dim == 128

    def test_batch_size_override(self):
        """--batch_size 64 sets training.batch_size."""
        from scripts.train import build_config
        config, _, _ = build_config(["--batch_size", "64"])
        assert config.training.batch_size == 64

    def test_output_dir_override(self):
        """--output_dir runs/test sets checkpoint.output_dir."""
        from scripts.train import build_config
        config, _, _ = build_config(["--output_dir", "runs/test"])
        assert config.checkpoint.output_dir == "runs/test"

    def test_lambda_reg(self):
        """--lambda_reg 0.05 returns lambda_reg=0.05."""
        from scripts.train import build_config
        _, lambda_reg, _ = build_config(["--lambda_reg", "0.05"])
        assert lambda_reg == 0.05

    def test_unfreeze_after_epoch_in_config(self):
        """unfreeze_after_epoch field exists in FineTuneConfig."""
        from nsmor.config_parser import FineTuneConfig
        cfg = FineTuneConfig()
        assert hasattr(cfg, "unfreeze_after_epoch")
        assert cfg.unfreeze_after_epoch == -1


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

def test_float32_noninteger_lv_snapshot_and_bad_geometry(tmp_path: Path) -> None:
    kin_path, evt_path = _make_visual_only_csvs(tmp_path, n_trials=1, lv_ratio_ms=120.1)
    kin = pd.read_csv(kin_path)
    stored_lv = float(np.float32(120.1))
    kin["l_v_ratio"] = stored_lv
    kin.to_csv(kin_path, index=False)
    session = load_and_concat_sessions([kin_path], [evt_path])
    trial = extract_trial_data(session, "visual_session", 0)
    assert trial["l_v_ratio"][0] == stored_lv != 120.1

    snapshot = extract_mcmc_snapshot(trial, stimulus_onset_ms=0.0)
    collision = 120.1 / math.tan(math.radians(1.0))
    idx = int(np.argmin(np.abs(trial["time_ms"] - (collision - 50.0))))
    assert snapshot.shape == (5,)
    assert snapshot[0] == pytest.approx(trial["visual_angle"][idx])
    snapshots, _ = build_snapshot_dataset(assign_ground_truth_labels([trial]))
    np.testing.assert_array_equal(snapshots[0], snapshot)

    for bad_lv in (121.0, 0.0, -1.0, float("nan"), float("inf")):
        bad_trial = {**trial, "l_v_ratio": np.full_like(trial["l_v_ratio"], bad_lv)}
        with pytest.raises(ValueError, match="finite|l/v trace disagrees|positive"):
            extract_mcmc_snapshot(bad_trial, stimulus_onset_ms=0.0)


@pytest.mark.parametrize("producer", ["data", "metadata"])
def test_float32_noninteger_lv_survives_both_producers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, producer: str,
) -> None:
    from scripts import prepare_data, prepare_metadata
    import sys

    raw_dir = tmp_path / "raw"
    _make_label_audit_csvs(raw_dir, n_sessions=5)
    session_id = "0.500cricket_001_20260101_000000_session_1"
    session_dir = raw_dir / session_id
    visual_kin, visual_evt = _make_visual_only_csvs(
        tmp_path, n_trials=1, dt_ms=10.0, lv_ratio_ms=120.1,
    )
    kin = pd.read_csv(visual_kin)
    kin["session_id"] = session_id
    kin["trial_id"] = 5
    kin["l_v_ratio"] = float(np.float32(120.1))
    evt = pd.read_csv(visual_evt)
    evt["session_id"] = session_id
    evt["trial_id"] = 5
    kin_path = session_dir / "kinematics.csv"
    evt_path = session_dir / "events.csv"
    pd.concat([pd.read_csv(kin_path), kin], ignore_index=True).to_csv(kin_path, index=False)
    pd.concat([pd.read_csv(evt_path), evt], ignore_index=True).to_csv(evt_path, index=False)

    def run(output: Path) -> None:
        if producer == "data":
            prepare_data.prepare_dataset(raw_dir, output, random_seed=42)
        else:
            monkeypatch.setattr(sys, "argv", [
                "prepare_metadata.py", "--raw_dir", str(raw_dir), "--output", str(output),
            ])
            prepare_metadata.main()

    output = tmp_path / f"{producer}.pt"
    run(output)
    saved = torch.load(output, weights_only=False)
    assert (session_id, 5) in list(zip(saved["session_ids"], saved["trial_ids"]))
    if producer == "metadata":
        assert saved["snapshot_anchor_rules"].count("looming_collision") == 1
    else:
        assert saved["labeling_funnel_retention"]["snapshot_anchor_rules"]["looming_collision"] == 1

    # A 0.9 ms trace conflict is far beyond float32 storage rounding.
    all_kin = pd.read_csv(kin_path)
    all_kin.loc[all_kin["trial_id"] == 5, "l_v_ratio"] = 121.0
    all_kin.to_csv(kin_path, index=False)
    rejected = tmp_path / f"{producer}_conflict.pt"
    with pytest.raises(ValueError, match=r"session=.*trial=5.*l/v trace disagrees"):
        run(rejected)
    assert not rejected.exists()


@pytest.mark.parametrize("producer", ["data", "metadata"])
def test_production_extraction_error_aborts_without_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, producer: str,
) -> None:
    from scripts import prepare_data, prepare_metadata
    import sys

    raw_dir = tmp_path / "raw"
    _make_label_audit_csvs(raw_dir, n_sessions=5)
    session_id = "0.500cricket_001_20260101_000000_session_1"
    module = prepare_data if producer == "data" else prepare_metadata
    original = module.extract_trial_data

    def corrupt_trial(session_data, sid, tid):
        if sid == session_id and tid == 4:
            raise ValueError("corrupt kinematics")
        return original(session_data, sid, tid)

    monkeypatch.setattr(module, "extract_trial_data", corrupt_trial)
    output = tmp_path / f"{producer}.pt"
    if producer == "data":
        run = lambda: prepare_data.prepare_dataset(raw_dir, output)
    else:
        monkeypatch.setattr(sys, "argv", [
            "prepare_metadata.py", "--raw_dir", str(raw_dir), "--output", str(output),
        ])
        run = prepare_metadata.main

    with pytest.raises(ValueError, match=rf"{session_id}.*4.*corrupt kinematics"):
        run()
    assert not output.exists()


def test_production_sequence_error_preserves_prior_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import prepare_data as prep

    raw_dir = tmp_path / "raw"
    _make_label_audit_csvs(raw_dir, n_sessions=2)
    session_id = "0.500cricket_001_20260101_000000_session_1"
    original = prep.extract_trial_sequence

    def corrupt_sequence(trial_data, *args, **kwargs):
        if trial_data["session_id"] == session_id and trial_data["trial_id"] == 4:
            raise ValueError("corrupt sequence")
        return original(trial_data, *args, **kwargs)

    monkeypatch.setattr(prep, "extract_trial_sequence", corrupt_sequence)
    output = tmp_path / "dataset.pt"
    with pytest.raises(ValueError, match=rf"{session_id}.*4.*corrupt sequence"):
        prep.prepare_dataset(raw_dir, output)
    assert not output.exists()


def test_float32_cast_overflow_warning_suppressed_but_guard_still_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A representable but out-of-float32-range value must not raise a
    RuntimeWarning before the finite guard runs; the guard itself still
    rejects the data (fail closed)."""
    import warnings

    from scripts import prepare_data as prep

    raw_dir = tmp_path / "raw"
    _make_label_audit_csvs(raw_dir, n_sessions=5)
    original = prep.extract_trial_sequence

    def overflow_sequence(*args, **kwargs):
        x64, y64 = original(*args, **kwargs)
        x64 = np.array(x64, dtype=np.float64)
        # 1e300 is finite in float64 but overflows float32 to +inf; the
        # cast emits an expected "overflow encountered in cast" warning.
        x64[0, 3] = 1e300
        return x64, y64

    monkeypatch.setattr(prep, "extract_trial_sequence", overflow_sequence)
    output = tmp_path / "dataset.pt"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(AssertionError, match="non-finite X after float32"):
            prep.prepare_dataset(raw_dir, output)
    assert not output.exists()


def test_metadata_valid_trials_keep_prior_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import prepare_metadata
    import sys

    raw_dir = tmp_path / "raw"
    _make_label_audit_csvs(raw_dir, n_sessions=2)
    output = tmp_path / "metadata.pt"
    monkeypatch.setattr(sys, "argv", [
        "prepare_metadata.py", "--raw_dir", str(raw_dir), "--output", str(output),
    ])
    prepare_metadata.main()
    metadata = torch.load(output, weights_only=False)

    expected = [(directory.name, trial_id)
                for directory in sorted(raw_dir.iterdir()) for trial_id in range(5)]
    assert metadata["n_trials"] == 10
    assert list(zip(metadata["session_ids"], metadata["trial_ids"])) == expected
    assert metadata["mcmc_priors"].shape == (10, 4)
    assert torch.isfinite(metadata["mcmc_priors"]).all()
    torch.testing.assert_close(metadata["mcmc_priors"].sum(dim=1), torch.ones(10))
    assert metadata["mcmc_prior_provenance"] == "oof_2fold_recording_prefix_grouped_cv"
    assert metadata["animal_identity_status"] == "unverified"
    assert len(metadata["snapshot_anchor_rules"]) == 10
