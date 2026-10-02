"""Test that prepare_data preserves exact retained trial identity and target_ttc_ms.

Contract requirements (downstream finding A2):
1. Exactly retained trial_ids, session_ids, and target_ttc_ms are persisted in dataset.
2. trial_ids are extracted from source trial_data, NOT reconstructed from row order.
3. Valid trial_ids, session_ids, target_ttc_ms, snapshots, priors, and sequences stay
   strictly aligned row-for-row; Step 5 errors abort without saving an artifact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION, FeatureConfig, TimeWindowConfig
from scripts.prepare_data import prepare_dataset


class _FakeMCMCModel:
    def predict_proba(self, x):
        return np.full((len(x), 4), 0.25, dtype=np.float64)


def test_prepare_data_producer_preserves_retained_identity(tmp_path: Path):
    """prepare_data must persist trial_ids, session_ids, and target_ttc_ms row-for-row aligned."""
    raw_dir = tmp_path / "raw"
    sess_dir = raw_dir / "0.500cricket_001_20260101_000000_session_1"
    sess_dir.mkdir(parents=True)

    dt_ms = 4.0
    n_frames = 300
    times = np.arange(n_frames) * dt_ms

    # Create 3 trials with non-trivial trial_ids: 101, 102, 103
    kin_rows = []
    evt_rows = []
    trial_ids = [101, 102, 103]
    target_ttcs = [0.0, -119.0, None]

    for t_id, ttc in zip(trial_ids, target_ttcs):
        for f, t in enumerate(times):
            kin_rows.append({
                "session_id": sess_dir.name,
                "trial_id": t_id,
                "time_ms": t,
                "x_pos": float(f),
                "y_pos": 0.0,
                "heading": 0.0,
                "velocity": 0.0,
                "acceleration": 0.0,
                "visual_angle": 10.0 if f >= 200 else 0.0,
                "wind_state": 1.0 if f >= 200 else 0.0,
                "l_v_ratio": 120.0,
            })
        det_dict = {"type": "looming_wind"}
        if ttc is not None:
            det_dict["target_ttc_ms"] = ttc
            det_dict["lv_ratio_ms"] = 120.0
        evt_rows.append({
            "session_id": sess_dir.name,
            "trial_id": t_id,
            "time_ms": times[200],
            "event_type": "stimulus_onset",
            "event_value": "",
        })
        evt_rows.append({
            "session_id": sess_dir.name,
            "trial_id": t_id,
            "time_ms": times[0],
            "event_type": "trial_start",
            "event_value": json.dumps(det_dict),
        })

    pd.DataFrame(kin_rows).to_csv(sess_dir / "kinematics.csv", index=False)
    pd.DataFrame(evt_rows).to_csv(sess_dir / "events.csv", index=False)

    out_dataset = tmp_path / "processed" / "nsmor_dataset.pt"

    # Run prepare_dataset with mocked MCMC cross-fitting and fold resolution
    with (
        mock.patch("scripts.prepare_data.resolve_group_folds", return_value=2),
        mock.patch("scripts.prepare_data.train_mcmc_cross_fitted") as mock_mcmc,
    ):
        mock_mcmc.return_value = (
            np.full((3, 4), 0.25, dtype=np.float64),
            [_FakeMCMCModel()],
            [],
        )
        prepare_dataset(
            raw_dir=raw_dir,
            output_path=out_dataset,
            dt_ms=dt_ms,
            random_seed=42,
        )

    assert out_dataset.exists()
    ds = torch.load(out_dataset, weights_only=False)

    assert "trial_ids" in ds, "dataset must persist 'trial_ids'"
    assert "session_ids" in ds, "dataset must persist 'session_ids'"
    assert "target_ttc_ms" in ds, "dataset must persist 'target_ttc_ms'"

    assert np.array_equal(ds["trial_ids"], np.array(trial_ids, dtype=np.int64)), (
        f"trial_ids mismatch: {ds['trial_ids']} vs {trial_ids}"
    )
    assert list(ds["session_ids"]) == [sess_dir.name] * len(trial_ids)
    assert ds["target_ttc_ms"] is not None
    assert ds["target_ttc_ms"][0] == 0.0
    assert ds["target_ttc_ms"][1] == -119.0
    assert np.isnan(ds["target_ttc_ms"][2])


def test_prepare_data_corrupt_ttc_refused(tmp_path: Path):
    """prepare_data must fail closed on corrupt or non-finite target_ttc_ms."""
    raw_dir = tmp_path / "raw_corrupt"
    sess_dir = raw_dir / "0.500cricket_001_20260101_000000_session_1"
    sess_dir.mkdir(parents=True)

    dt_ms = 4.0
    times = np.arange(300) * dt_ms

    kin_rows = []
    for f, t in enumerate(times):
        kin_rows.append({
            "session_id": sess_dir.name,
            "trial_id": 1,
            "time_ms": t,
            "x_pos": float(f),
            "y_pos": 0.0,
            "heading": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "visual_angle": 10.0 if f >= 200 else 0.0,
            "wind_state": 1.0 if f >= 200 else 0.0,
            "l_v_ratio": 120.0,
        })
    evt_rows = [
        {
            "session_id": sess_dir.name,
            "trial_id": 1,
            "time_ms": times[200],
            "event_type": "stimulus_onset",
            "event_value": "",
        },
        {
            "session_id": sess_dir.name,
            "trial_id": 1,
            "time_ms": times[0],
            "event_type": "trial_start",
            "event_value": json.dumps({"type": "looming_wind", "target_ttc_ms": "corrupt_nonfinite"}),
        },
    ]

    pd.DataFrame(kin_rows).to_csv(sess_dir / "kinematics.csv", index=False)
    pd.DataFrame(evt_rows).to_csv(sess_dir / "events.csv", index=False)

    out_dataset = tmp_path / "processed" / "corrupt.pt"

    with (
        mock.patch("scripts.prepare_data.resolve_group_folds", return_value=2),
        mock.patch("scripts.prepare_data.train_mcmc_cross_fitted") as mock_mcmc,
    ):
        mock_mcmc.return_value = (
            np.full((1, 4), 0.25, dtype=np.float64),
            [_FakeMCMCModel()],
            [],
        )
        with pytest.raises(ValueError, match="fail closed|Corrupt|Non-finite"):
            prepare_dataset(
                raw_dir=raw_dir,
                output_path=out_dataset,
                dt_ms=dt_ms,
                random_seed=42,
            )


def test_prepare_data_requires_valid_dt_ms(tmp_path: Path):
    """prepare_data requires positive finite dt_ms (fail closed on unknown cadence)."""
    raw_dir = tmp_path / "raw_dt"
    sess_dir = raw_dir / "0.500cricket_001_20260101_000000_session_1"
    sess_dir.mkdir(parents=True)

    kin_rows = [{
        "session_id": sess_dir.name,
        "trial_id": 1,
        "time_ms": 0.0,
        "x_pos": 0.0,
        "y_pos": 0.0,
        "heading": 0.0,
        "velocity": 0.0,
        "acceleration": 0.0,
        "visual_angle": 0.0,
        "wind_state": 1.0,
        "l_v_ratio": 0.0,
    }]
    evt_rows = [{
        "session_id": sess_dir.name,
        "trial_id": 1,
        "time_ms": 0.0,
        "event_type": "trial_start",
        "event_value": json.dumps({"type": "wind_only"}),
    }]
    pd.DataFrame(kin_rows).to_csv(sess_dir / "kinematics.csv", index=False)
    pd.DataFrame(evt_rows).to_csv(sess_dir / "events.csv", index=False)

    out_dataset = tmp_path / "processed" / "out.pt"

    with pytest.raises(ValueError, match="positive finite dt_ms"):
        prepare_dataset(
            raw_dir=raw_dir,
            output_path=out_dataset,
            dt_ms=None,
            random_seed=42,
        )


@pytest.mark.parametrize("sequence_error", [False, True], ids=["valid", "corrupt"])
def test_prepare_data_sequence_failure_or_valid_alignment(tmp_path: Path, sequence_error: bool):
    """A corrupt sequence aborts before save; valid trial and prior rows stay aligned."""
    raw_dir = tmp_path / "raw_drop"
    sess_dir = raw_dir / "0.500cricket_001_20260101_000000_session_1"
    sess_dir.mkdir(parents=True)

    dt_ms = 4.0
    times = np.arange(300) * dt_ms

    kin_rows = []
    evt_rows = []
    trial_ids = [201, 202]

    for t_id in trial_ids:
        for f, t in enumerate(times):
            kin_rows.append({
                "session_id": sess_dir.name,
                "trial_id": t_id,
                "time_ms": t,
                "x_pos": float(f),
                "y_pos": 0.0,
                "heading": 0.0,
                "velocity": 0.0,
                "acceleration": 0.0,
                "visual_angle": 10.0 if f >= 200 else 0.0,
                "wind_state": 1.0 if f >= 200 else 0.0,
                "l_v_ratio": 120.0,
            })
        evt_rows.append({
            "session_id": sess_dir.name,
            "trial_id": t_id,
            "time_ms": times[200],
            "event_type": "stimulus_onset",
            "event_value": "",
        })
        evt_rows.append({
            "session_id": sess_dir.name,
            "trial_id": t_id,
            "time_ms": times[0],
            "event_type": "trial_start",
            "event_value": json.dumps({"type": "looming_wind", "target_ttc_ms": 0.0 if t_id == 201 else -119.0}),
        })

    pd.DataFrame(kin_rows).to_csv(sess_dir / "kinematics.csv", index=False)
    pd.DataFrame(evt_rows).to_csv(sess_dir / "events.csv", index=False)

    out_dataset = tmp_path / "processed" / "drop_out.pt"

    def _mock_extract(trial_data, **kwargs):
        if sequence_error and trial_data.get("trial_id") == 202:
            raise ValueError("Simulated corrupt kinematics in trial 202")
        from nsmor.data_extractor import extract_trial_sequence as _real_extract
        return _real_extract(trial_data, **kwargs)

    priors = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    with (
        mock.patch("scripts.prepare_data.resolve_group_folds", return_value=2),
        mock.patch("scripts.prepare_data.train_mcmc_cross_fitted") as mock_mcmc,
        mock.patch("scripts.prepare_data.extract_trial_sequence", side_effect=_mock_extract) as sequence_spy,
        mock.patch("scripts.prepare_data.torch.save", wraps=torch.save) as save_spy,
    ):
        mock_mcmc.return_value = (priors, [_FakeMCMCModel()], [])
        if sequence_error:
            with pytest.raises(ValueError, match="Simulated corrupt kinematics in trial 202") as error:
                prepare_dataset(raw_dir, out_dataset, dt_ms=dt_ms, random_seed=42)
            assert sequence_spy.call_count == 2
            failed_trial = sequence_spy.call_args.args[0]
            assert failed_trial["session_id"] == sess_dir.name
            assert failed_trial["trial_id"] == 202
            assert f"session={sess_dir.name!r}" in str(error.value)
            assert "trial=202" in str(error.value)
            assert isinstance(error.value.__cause__, ValueError)
            mock_mcmc.assert_called_once()
            assert mock_mcmc.call_args.args[0].shape == (2, 5)
            save_spy.assert_not_called()
            assert not out_dataset.exists()
            return
        prepare_dataset(raw_dir, out_dataset, dt_ms=dt_ms, random_seed=42)
        save_spy.assert_called_once()

    ds = torch.load(out_dataset, weights_only=False)
    np.testing.assert_array_equal(ds["trial_ids"], trial_ids)
    assert list(ds["session_ids"]) == [sess_dir.name] * 2
    np.testing.assert_array_equal(ds["target_ttc_ms"], [0.0, -119.0])
    assert len(ds["X_seqs"]) == len(ds["Y_seqs"]) == sequence_spy.call_count == 2
    np.testing.assert_array_equal(ds["snapshots"], mock_mcmc.call_args.args[0])
    np.testing.assert_array_equal(ds["mcmc_snapshots"], ds["snapshots"])
    np.testing.assert_array_equal(ds["labels"], mock_mcmc.call_args.args[1])
    np.testing.assert_array_equal(ds["mcmc_priors"], priors)


def test_prepare_data_bool_ttc_refused(tmp_path: Path):
    """prepare_data must refuse boolean target_ttc_ms (fail closed, False != 0ms)."""
    raw_dir = tmp_path / "raw_bool_ttc"
    sess_dir = raw_dir / "0.500cricket_001_20260101_000000_session_1"
    sess_dir.mkdir(parents=True)

    dt_ms = 4.0
    times = np.arange(300) * dt_ms

    kin_rows = []
    for f, t in enumerate(times):
        kin_rows.append({
            "session_id": sess_dir.name,
            "trial_id": 1,
            "time_ms": t,
            "x_pos": float(f),
            "y_pos": 0.0,
            "heading": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "visual_angle": 10.0 if f >= 200 else 0.0,
            "wind_state": 1.0 if f >= 200 else 0.0,
            "l_v_ratio": 120.0,
        })
    evt_rows = [
        {
            "session_id": sess_dir.name,
            "trial_id": 1,
            "time_ms": times[200],
            "event_type": "stimulus_onset",
            "event_value": "",
        },
        {
            "session_id": sess_dir.name,
            "trial_id": 1,
            "time_ms": times[0],
            "event_type": "trial_start",
            "event_value": json.dumps({"type": "looming_wind", "target_ttc_ms": False}),
        },
    ]

    pd.DataFrame(kin_rows).to_csv(sess_dir / "kinematics.csv", index=False)
    pd.DataFrame(evt_rows).to_csv(sess_dir / "events.csv", index=False)

    out_dataset = tmp_path / "processed" / "bool_out.pt"

    with (
        mock.patch("scripts.prepare_data.resolve_group_folds", return_value=2),
        mock.patch("scripts.prepare_data.train_mcmc_cross_fitted") as mock_mcmc,
    ):
        mock_mcmc.return_value = (
            np.full((1, 4), 0.25, dtype=np.float64),
            [_FakeMCMCModel()],
            [],
        )
        with pytest.raises(ValueError, match="Boolean target_ttc_ms|fail closed"):
            prepare_dataset(
                raw_dir=raw_dir,
                output_path=out_dataset,
                dt_ms=dt_ms,
                random_seed=42,
            )


def test_prepare_data_malformed_details_refused(tmp_path: Path):
    """prepare_data must refuse malformed non-empty event_values (fail closed)."""
    raw_dir = tmp_path / "raw_malformed_details"
    sess_dir = raw_dir / "0.500cricket_001_20260101_000000_session_1"
    sess_dir.mkdir(parents=True)

    dt_ms = 4.0
    times = np.arange(300) * dt_ms

    kin_rows = []
    for f, t in enumerate(times):
        kin_rows.append({
            "session_id": sess_dir.name,
            "trial_id": 1,
            "time_ms": t,
            "x_pos": float(f),
            "y_pos": 0.0,
            "heading": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "visual_angle": 10.0 if f >= 200 else 0.0,
            "wind_state": 1.0 if f >= 200 else 0.0,
            "l_v_ratio": 120.0,
        })
    evt_rows = [
        {
            "session_id": sess_dir.name,
            "trial_id": 1,
            "time_ms": times[200],
            "event_type": "stimulus_onset",
            "event_value": "",
        },
        {
            "session_id": sess_dir.name,
            "trial_id": 1,
            "time_ms": times[0],
            "event_type": "trial_start",
            "event_value": "{not a valid json or dict",
        },
    ]

    pd.DataFrame(kin_rows).to_csv(sess_dir / "kinematics.csv", index=False)
    pd.DataFrame(evt_rows).to_csv(sess_dir / "events.csv", index=False)

    out_dataset = tmp_path / "processed" / "malformed_out.pt"

    with (
        mock.patch("scripts.prepare_data.resolve_group_folds", return_value=2),
        mock.patch("scripts.prepare_data.train_mcmc_cross_fitted") as mock_mcmc,
    ):
        mock_mcmc.return_value = (
            np.full((1, 4), 0.25, dtype=np.float64),
            [_FakeMCMCModel()],
            [],
        )
        with pytest.raises(ValueError, match="Malformed non-empty event details"):
            prepare_dataset(
                raw_dir=raw_dir,
                output_path=out_dataset,
                dt_ms=dt_ms,
                random_seed=42,
            )


def test_prepare_data_all_missing_ttc_persists_float_array_shape_n(tmp_path: Path):
    """When all trials have missing TTC, target_ttc_ms is still a float array of shape (N,) with NaNs."""
    raw_dir = tmp_path / "raw_all_missing"
    sess_dir = raw_dir / "0.500cricket_001_20260101_000000_session_1"
    sess_dir.mkdir(parents=True)

    dt_ms = 4.0
    times = np.arange(300) * dt_ms

    kin_rows = []
    evt_rows = []
    for t_id in (1, 2):
        for f, t in enumerate(times):
            kin_rows.append({
                "session_id": sess_dir.name,
                "trial_id": t_id,
                "time_ms": t,
                "x_pos": float(f),
                "y_pos": 0.0,
                "heading": 0.0,
                "velocity": 0.0,
                "acceleration": 0.0,
                "visual_angle": 10.0 if f >= 200 else 0.0,
                "wind_state": 1.0 if f >= 200 else 0.0,
                "l_v_ratio": 120.0,
            })
        evt_rows.append({
            "session_id": sess_dir.name,
            "trial_id": t_id,
            "time_ms": times[200],
            "event_type": "stimulus_onset",
            "event_value": "",
        })
        evt_rows.append({
            "session_id": sess_dir.name,
            "trial_id": t_id,
            "time_ms": times[0],
            "event_type": "trial_start",
            "event_value": json.dumps({"type": "wind_only"}),  # No target_ttc_ms at all
        })

    pd.DataFrame(kin_rows).to_csv(sess_dir / "kinematics.csv", index=False)
    pd.DataFrame(evt_rows).to_csv(sess_dir / "events.csv", index=False)

    out_dataset = tmp_path / "processed" / "all_missing_out.pt"

    with (
        mock.patch("scripts.prepare_data.resolve_group_folds", return_value=2),
        mock.patch("scripts.prepare_data.train_mcmc_cross_fitted") as mock_mcmc,
    ):
        mock_mcmc.return_value = (
            np.full((2, 4), 0.25, dtype=np.float64),
            [_FakeMCMCModel()],
            [],
        )
        prepare_dataset(
            raw_dir=raw_dir,
            output_path=out_dataset,
            dt_ms=dt_ms,
            random_seed=42,
        )

    ds = torch.load(out_dataset, weights_only=False)
    assert "target_ttc_ms" in ds
    assert isinstance(ds["target_ttc_ms"], np.ndarray)
    assert ds["target_ttc_ms"].shape == (2,)
    assert ds["target_ttc_ms"].dtype == np.float64
    assert np.isnan(ds["target_ttc_ms"]).all()


def test_prepare_data_distinguishable_multi_trial_content_and_ttc(tmp_path: Path):
    """Test with distinct non-uniform kinematics traces, trial_ids, and distinct TTC values,
    verifying exact 1-to-1 retention and value binding."""
    raw_dir = tmp_path / "raw_distinguishable"
    sess_dir = raw_dir / "0.500cricket_001_20260101_000000_session_1"
    sess_dir.mkdir(parents=True)

    dt_ms = 4.0
    times = np.arange(300) * dt_ms

    kin_rows = []
    evt_rows = []
    trial_ids = [501, 502, 503]
    ttcs = [-373.0, 0.0, 150.0]

    for i, (t_id, ttc) in enumerate(zip(trial_ids, ttcs)):
        # Highly distinct kinematics per trial (different linear scales)
        for f, t in enumerate(times):
            kin_rows.append({
                "session_id": sess_dir.name,
                "trial_id": t_id,
                "time_ms": t,
                "x_pos": float(f * (i + 1)),
                "y_pos": float(f * 0.1),
                "heading": float(i * 10.0),
                "velocity": float(i * 2.0 + 0.1),
                "acceleration": 0.0,
                "visual_angle": float(10.0 * (i + 1)) if f >= 200 else 0.0,
                "wind_state": 1.0 if f >= 200 else 0.0,
                "l_v_ratio": 120.0,
            })
        evt_rows.append({
            "session_id": sess_dir.name,
            "trial_id": t_id,
            "time_ms": times[200],
            "event_type": "stimulus_onset",
            "event_value": "",
        })
        evt_rows.append({
            "session_id": sess_dir.name,
            "trial_id": t_id,
            "time_ms": times[0],
            "event_type": "trial_start",
            "event_value": json.dumps({"type": "looming_wind", "target_ttc_ms": ttc, "lv_ratio_ms": 120.0}),
        })

    pd.DataFrame(kin_rows).to_csv(sess_dir / "kinematics.csv", index=False)
    pd.DataFrame(evt_rows).to_csv(sess_dir / "events.csv", index=False)

    out_dataset = tmp_path / "processed" / "distinguishable_out.pt"

    with (
        mock.patch("scripts.prepare_data.resolve_group_folds", return_value=2),
        mock.patch("scripts.prepare_data.train_mcmc_cross_fitted") as mock_mcmc,
    ):
        mock_mcmc.return_value = (
            np.full((3, 4), 0.25, dtype=np.float64),
            [_FakeMCMCModel()],
            [],
        )
        prepare_dataset(
            raw_dir=raw_dir,
            output_path=out_dataset,
            dt_ms=dt_ms,
            random_seed=42,
        )

    ds = torch.load(out_dataset, weights_only=False)
    assert np.array_equal(ds["trial_ids"], np.array(trial_ids, dtype=np.int64))
    assert np.allclose(ds["target_ttc_ms"], np.array(ttcs, dtype=np.float64))
    # Verify each trial's X_seq/Y_seq carries distinct trace corresponding to that trial
    for i, t_id in enumerate(trial_ids):
        # Initial velocity was float(i * 2.0 + 0.1)
        expected_vel_start = float(i * 2.0 + 0.1)
        assert np.isclose(ds["Y_seqs"][i][0], expected_vel_start, atol=1e-3)
        # Visual angle at frame 200 was float(10.0 * (i + 1))
        expected_vis = float(10.0 * (i + 1))
        assert np.isclose(ds["X_seqs"][i][200, 0], expected_vis, atol=1e-3)



def _write_declared_trial_pairs(raw_dir: Path, *, missing_frames: bool) -> tuple[str, str]:
    """Two real CSV pairs; the second carries an event-only continuation of the first."""
    sessions = (
        "0.500cricket_001_20260101_000000_session_1",
        "0.500cricket_002_20260101_000000_session_1",
    )
    for sid, tid in zip(sessions, (11, 12)):
        directory = raw_dir / sid
        directory.mkdir(parents=True)
        kin = pd.DataFrame([
            {
                "session_id": sid, "trial_id": tid, "time_ms": frame * 4.0,
                "x_pos": float(frame), "y_pos": 0.0, "heading": 0.0,
                "velocity": 0.0, "acceleration": 0.0,
                "visual_angle": 10.0 if frame >= 200 else 0.0,
                "wind_state": 1.0 if frame >= 200 else 0.0,
                "l_v_ratio": 120.0,
            }
            for frame in range(300)
        ])
        events = [
            {"session_id": sid, "trial_id": tid, "time_ms": 800.0,
             "event_type": "stimulus_onset", "event_value": ""},
        ]
        if sid == sessions[1]:
            events.insert(0, {
                "session_id": sid, "trial_id": tid, "time_ms": 0.0,
                "event_type": "trial_start", "event_value": json.dumps({"type": "looming_wind"}),
            })
        if sid == sessions[0] and missing_frames:
            events.append({
                "session_id": sid, "trial_id": 99, "time_ms": 0.0,
                "event_type": "trial_start", "event_value": json.dumps({"type": "looming_wind"}),
            })
        if sid == sessions[1]:
            events.append({
                "session_id": sessions[0], "trial_id": 11, "time_ms": 0.0,
                "event_type": "trial_start",
                "event_value": json.dumps({"type": "looming_wind", "target_ttc_ms": -119.0}),
            })
        kin.to_csv(directory / "kinematics.csv", index=False)
        pd.DataFrame(events).to_csv(directory / "events.csv", index=False)
    return sessions


@pytest.mark.parametrize("producer", ["data", "metadata"])
@pytest.mark.parametrize("missing_frames", [True, False], ids=["orphan", "continuation"])
def test_declared_trial_kinematics_completeness_at_production_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, producer: str, missing_frames: bool,
) -> None:
    """Fail before prior fit/save on an orphan; keep a valid cross-pair continuation."""
    from scripts import prepare_data, prepare_metadata

    raw_dir = tmp_path / "raw"
    sessions = _write_declared_trial_pairs(raw_dir, missing_frames=missing_frames)
    module = prepare_data if producer == "data" else prepare_metadata
    output = tmp_path / "processed" / f"{producer}.pt"
    monkeypatch.setattr(module, "resolve_group_folds", lambda *args, **kwargs: 2)
    prior_fit = mock.Mock(return_value=(
        np.full((2, 4), 0.25, dtype=np.float64), [_FakeMCMCModel()], [],
    ))
    monkeypatch.setattr(module, "train_mcmc_cross_fitted", prior_fit)

    def run() -> None:
        if producer == "data":
            prepare_data.prepare_dataset(raw_dir, output, dt_ms=4.0)
        else:
            monkeypatch.setattr(sys, "argv", [
                "prepare_metadata.py", "--raw_dir", str(raw_dir), "--output", str(output),
            ])
            prepare_metadata.main()

    if missing_frames:
        with pytest.raises(ValueError, match=rf"{sessions[0]}.*99.*kinematics"):
            run()
        prior_fit.assert_not_called()
        assert not output.exists()
    else:
        run()
        prior_fit.assert_called_once()
        saved = torch.load(output, weights_only=False)
        assert list(zip(saved["session_ids"], saved["trial_ids"])) == [
            (sessions[0], 11), (sessions[1], 12),
        ]
        assert len(saved["mcmc_priors"]) == 2
        assert saved["target_ttc_ms"][0] == -119.0


@pytest.mark.parametrize("end_ms,anchor_ms,condition,expected_frame,origin_ms,dt_ms", [
    (105, 105., "visual_only", None, 0., 4.),
    (104, 104., "visual_only", 26, 0., 4.),
    (110, 105., "visual_only", 27, 0., 4.),
    (110, 83., "wind_only", 1446, 0., 4.),
    (110, 83., "multisensory", 21, 0., 4.),
    (65.1, 65.1, "visual_only", 16, 1.1, 4.),
    (1400.1, 1400.1, "visual_only", 100, 1000.1, 4.),
    (65.1, 65.1, "visual_only", 8, 1.1, 8.),
])
def test_producer_maps_source_anchor_causally(
    tmp_path: Path, end_ms: float, anchor_ms: float,
    condition: str, expected_frame: int | None, origin_ms: float, dt_ms: float,
) -> None:
    """Real CSV → ETL → restricted loader, including peak/onset disagreements."""
    from tests.test_pipeline import _make_label_audit_csvs
    from nsmor.data_extractor import resolve_snapshot_anchor, build_snapshot_dataset
    from nsmor.pipeline.io import load_and_concat_sessions, extract_trial_data
    from nsmor.pipeline.labeling import assign_ground_truth_labels
    from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint
    from nsmor.nsmor_dataloader import NSMoRDataset

    raw = tmp_path / "raw"
    _make_label_audit_csvs(raw, n_sessions=5)
    directory = sorted(raw.iterdir())[0]
    sid, tid = directory.name, 777
    if origin_ms:
        n = 17 if origin_ms == 1.1 else 101
        times = origin_ms + np.arange(n) * 4
    else:
        times = np.arange(0, end_ms + 1, 5, dtype=np.float64)
        if times[-1] != end_ms:
            times = np.r_[times, float(end_ms)]
    visual = np.zeros_like(times)
    lv = 30.
    init_deg = float(np.degrees(2 * np.arctan2(lv, anchor_ms - origin_ms)))
    if condition != "wind_only":
        visual = np.minimum(np.degrees(2 * np.arctan2(
            lv, np.maximum(anchor_ms - times, 0.),
        )), 179.)
        # One early angle artifact must not turn the peak into the collision.
        visual[1] = 180.
    wind = ((times >= anchor_ms).astype(float) if condition != "visual_only"
            else np.zeros_like(times))
    special_kin = pd.DataFrame({
        "session_id": sid, "trial_id": tid, "time_ms": times,
        "x_pos": 0., "y_pos": 0., "heading": 0., "velocity": 0.1,
        "acceleration": 0., "visual_angle": visual, "wind_state": wind,
        "l_v_ratio": lv if condition != "wind_only" else 0.,
    })
    onset = origin_ms if condition == "visual_only" else anchor_ms
    special_evt = pd.DataFrame([
        {"session_id": sid, "trial_id": tid, "time_ms": origin_ms,
         "event_type": "trial_start", "event_value": json.dumps({
             "lv_ratio_ms": lv, "init_deg": init_deg,
         })},
        {"session_id": sid, "trial_id": tid, "time_ms": onset,
         "event_type": "stimulus_onset", "event_value": ""},
    ])
    kin_path, evt_path = directory / "kinematics.csv", directory / "events.csv"
    for path, extra in ((kin_path, special_kin), (evt_path, special_evt)):
        pd.concat([pd.read_csv(path), extra], ignore_index=True).to_csv(path, index=False)
    trial = extract_trial_data(load_and_concat_sessions([kin_path], [evt_path]), sid, tid)
    source_times = trial["time_ms"].copy()
    resolved_ms, rule = resolve_snapshot_anchor(trial, onset)
    assert resolved_ms == pytest.approx(anchor_ms, abs=1e-12)
    expected_snaps, expected_labels = build_snapshot_dataset(assign_ground_truth_labels([trial]))
    output = tmp_path / "dataset.pt"
    if expected_frame is None:
        with pytest.raises(ValueError, match="777.*source anchor.*no model frame"):
            prepare_dataset(raw, output, dt_ms=4.)
        assert not output.exists()
        return

    prepare_dataset(raw, output, dt_ms=dt_ms)
    saved, _ = load_dataset_with_fingerprint(output, expected_dt_ms=dt_ms)
    index = list(saved["trial_ids"]).index(tid)
    assert saved["anchor_frames"][index] == expected_frame
    record = saved["model_grid_provenance"][index]
    assert record["source_anchor_ms"] == resolved_ms
    assert record["source_anchor_rule"] == rule
    assert saved["stimulus_conditions"][index] == condition
    np.testing.assert_array_equal(saved["snapshots"][index], expected_snaps[0])
    assert saved["labels"][index] == expected_labels[0]
    np.testing.assert_array_equal(trial["time_ms"], source_times)
    # No invented visual collision: persisted features are causal source holds.
    grid = source_times[::int(dt_ms / 4)] if origin_ms else np.arange(
        0, end_ms + 1, dt_ms, dtype=np.float64,
    )
    assert record["model_n"] == len(grid)
    assert record["model_end_ms"] == grid[-1]
    assert record["model_end_ms"] <= record["source_end_ms"]
    held_indices = np.searchsorted(source_times, grid, side="right") - 1
    prepend = record["synthetic_prepend_frames"]
    np.testing.assert_array_equal(
        saved["X_seqs"][index][prepend:, 0], trial["visual_angle"][held_indices].astype(np.float32),
    )
    sequence = (saved["X_seqs"][index], saved["Y_seqs"][index], saved["labels"][index])
    loader = NSMoRDataset([sequence], saved["mcmc_priors"][[index]],
                          max_seq_len=8, pre_anchor_frames=2,
                          anchor_frames=[saved["anchor_frames"][index]])
    x, _ = loader[0]
    start = min(expected_frame - 2, len(sequence[0]) - 8)
    np.testing.assert_array_equal(x[:, :4].numpy(), sequence[0][start:start + 8, :4])

    # The producer's sidecars must survive real train/downstream ingestion.
    from nsmor.config_parser import ExperimentConfig
    from scripts import train, analyze_dynamics

    config = ExperimentConfig()
    config.model.dt_ms = dt_ms
    config.training.num_workers = 0
    train_loader, val_loader = train.build_dataloaders(
        config, dataset_path=str(output),
    )
    for split in (train_loader.dataset, val_loader.dataset):
        assert split.anchor_frames == [
            saved["anchor_frames"][i] for i in split.source_indices
        ]
    downstream, _ = analyze_dynamics.load_dataset(
        output, max_seq_len=8, pre_anchor_frames=2,
    )
    assert downstream.dataset.anchor_frames[index] == expected_frame
    downstream_x, _ = downstream.dataset[index]
    np.testing.assert_array_equal(
        downstream_x[:, :4].numpy(), sequence[0][start:start + 8, :4],
    )

    # A stale peak-derived stamp must not pass the shared loader validation.
    saved["anchor_frames"][index] = 2
    bad = tmp_path / "bad-anchor.pt"
    torch.save(saved, bad)
    with pytest.raises(ValueError, match="source anchor.*disagrees"):
        load_dataset_with_fingerprint(bad)

    # Losing both sidecars must not downgrade a new producer to the peak proxy.
    saved.pop("model_grid_provenance")
    saved.pop("anchor_frames")
    stripped = tmp_path / "stripped-clock.pt"
    torch.save(saved, stripped)
    with pytest.raises(ValueError, match="modern.*clock|model grid"):
        load_dataset_with_fingerprint(stripped)
    with pytest.raises(ValueError, match="modern.*clock|model grid"):
        train.build_dataloaders(config, dataset_path=str(stripped))
    with pytest.raises(ValueError, match="modern.*clock|model grid"):
        analyze_dynamics.load_dataset(stripped)
