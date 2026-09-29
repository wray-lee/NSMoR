"""Public staging checks for experimental paired-clock association."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.pre_load_adapt import adapt_cercus_to_nsmor


class _UniformPrior:
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.full((len(x), 4), 0.25)


def _pair(
    root: Path, ticks: list[str], host: list[float]
) -> tuple[Path, Path]:
    root.mkdir()
    kin = root / "clock_kinematics.csv"
    evt = root / "clock_events.csv"
    pd.DataFrame({
        "sys_time": host, "ard_time": ticks,
        "dx": [0.0] + [0.1] * (len(ticks) - 1),
        "dy": 0.0, "dz": 0.0, "stim_state": 0, "global_trial_id": 1,
    }).to_csv(kin, index=False)
    pd.DataFrame({
        "event_name": ["trial_start"], "timestamp": [host[-1]],
        "global_trial_id": [1], "details": ["{}"],
    }).to_csv(evt, index=False)
    return kin, evt


def test_experimental_mapping_estimates_scale_and_persists_provenance(
    tmp_path: Path,
) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    kin, evt = _pair(raw, ["100", "105", "110", "115"],
                     [10.0, 10.006, 10.012, 10.018])
    originals = {p: p.read_bytes() for p in (kin, evt)}
    adapt_cercus_to_nsmor(
        raw, out, experimental_clock_residual_ms=0.001,
    )
    staged = pd.read_csv(out / "clock" / kin.name)
    events = pd.read_csv(out / "clock" / evt.name)
    audit = json.loads((out / "clock" / "clock_timebase_audit.json").read_text())
    np.testing.assert_allclose(staged["time_ms"], [0, 6, 12, 18], atol=1e-8)
    assert events["time_ms"].iloc[0] == pytest.approx(18)
    assert staged["source_row_index"].tolist() == [0, 1, 2, 3]
    assert staged["raw_ard_time"].tolist() == [100, 105, 110, 115]
    assert staged["time_source"].eq("experimental_affine_estimate").all()
    assert audit["status"] == "accepted_by_operational_tolerance"
    assert audit["trials"][0]["slope"] == pytest.approx(1.2)
    assert audit["trials"][0]["n_samples"] == 4
    assert audit["kinematics_sha256"] == hashlib.sha256(originals[kin]).hexdigest()
    for path, content in originals.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize("ticks,host", [
    (["100", "105", "110", "100000"], [10, 10.005, 10.010, 10.015]),
    (["0", "3058", "13063", "13067"],
     [13.966034, 13.966070, 13.971649, 13.976228]),
    (["460402460407", "105", "110", "115"], [10, 10.005, 10.01, 10.015]),
    (["NaN", "105", "110", "115"], [10, 10.005, 10.01, 10.015]),
    (["100.5", "105", "110", "115"], [10, 10.005, 10.01, 10.015]),
])
def test_unsupported_timing_keeps_raw_and_persists_rejection(
    tmp_path: Path, ticks: list[str], host: list[float],
) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    kin, evt = _pair(raw, ticks, host)
    before = {p: p.read_bytes() for p in (kin, evt)}
    with pytest.raises(ValueError, match="Experimental clock mapping rejected"):
        adapt_cercus_to_nsmor(raw, out, experimental_clock_residual_ms=0.1)
    assert not (out / "clock" / kin.name).exists()
    assert not (out / "clock" / evt.name).exists()
    audit = json.loads((out / "clock" / "clock_timebase_audit.json").read_text())
    assert audit["status"] == "rejected"
    assert audit["error"]
    for path, content in before.items():
        assert path.read_bytes() == content


def test_default_staging_keeps_host_axis_and_schema(tmp_path: Path) -> None:
    from scripts.pre_load_adapt import KIN_TARGET

    raw, out = tmp_path / "raw", tmp_path / "out"
    kin, _ = _pair(raw, ["invalid"] * 4, [10, 10.005, 10.005, 10.015])
    adapt_cercus_to_nsmor(raw, out)
    staged = pd.read_csv(out / "clock" / kin.name)
    np.testing.assert_allclose(staged["time_ms"], [0, 5, 5, 15])
    assert list(staged) == KIN_TARGET
    assert not (out / "clock" / "clock_timebase_audit.json").exists()


def test_zero_tick_is_legal_and_small_estimated_intervals_are_not_clipped(
    tmp_path: Path,
) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    kin, _ = _pair(raw, ["0", "1", "2", "3"],
                   [10, 10.0005, 10.0010, 10.0015])
    adapt_cercus_to_nsmor(raw, out, experimental_clock_residual_ms=0.001)
    staged = pd.read_csv(out / "clock" / kin.name)
    np.testing.assert_allclose(staged["time_ms"], [0, 0.5, 1, 1.5], atol=1e-8)
    np.testing.assert_allclose(staged["velocity"], [0, 20, 20, 20], atol=1e-7)


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), float("inf")])
def test_tolerance_must_be_explicit_finite_positive(
    tmp_path: Path, tolerance: float,
) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    kin, _ = _pair(raw, ["100", "105", "110", "115"],
                   [10, 10.005, 10.01, 10.015])
    with pytest.raises(ValueError, match="tolerance must be positive"):
        adapt_cercus_to_nsmor(raw, out, experimental_clock_residual_ms=tolerance)
    assert not (out / "clock" / kin.name).exists()


def _manifest(root: Path, kin: Path, evt: Path, prefix: int) -> Path:
    rows = pd.read_csv(kin, dtype=str, keep_default_na=False)
    path = root / "prefix.json"
    path.write_text(json.dumps({
        "schema": "experimental_suspect_prefix_v1",
        "assumption": "manually specified suspect prefix/model assumption",
        "sessions": [{
            "session_id": "clock",
            "kinematics_sha256": hashlib.sha256(kin.read_bytes()).hexdigest(),
            "events_sha256": hashlib.sha256(evt.read_bytes()).hexdigest(),
            "trials": [{"trial_id": "1", "rows": [
                {"source_row_index": i, "raw_sys_time": rows.iloc[i]["sys_time"],
                 "raw_ard_time": rows.iloc[i]["ard_time"]}
                for i in range(prefix)
            ]}],
        }],
    }), encoding="utf-8")
    return path


def test_manifest_prefix2_estimate_preserves_rows_and_describes_sensitivity(
    tmp_path: Path,
) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    kin, evt = _pair(raw, ["0", "3058", "13063", "13068", "13073", "13078"],
                     [9.997, 9.998, 10, 10.005, 10.01, 10.015])
    events = pd.read_csv(evt)
    events["timestamp"] = 9.993
    events.to_csv(evt, index=False)
    manifest = _manifest(tmp_path, kin, evt, 2)
    originals = {p: p.read_bytes() for p in (kin, evt, manifest)}
    adapt_cercus_to_nsmor(
        raw, out, experimental_clock_residual_ms=0.001,
        experimental_clock_prefix_manifest=manifest,
    )
    staged = pd.read_csv(out / "clock" / kin.name)
    audit = json.loads((out / "clock" / "clock_timebase_audit.json").read_text())
    np.testing.assert_allclose(staged["time_ms"], [0, 5, 10, 15, 20, 25])
    assert staged["source_row_index"].tolist() == list(range(6))
    assert staged["time_source"].tolist() == (
        ["experimental_prefix_cadence_estimate"] * 2
        + ["experimental_affine_estimate"] * 4
    )
    diag = audit["trials"][0]
    assert diag["prefix_rows"] == 2
    assert diag["suffix_fit"]["n_samples"] == 4
    assert diag["scientific_acceptance"] == "unresolved"
    sensitivity = diag["sensitivity"]
    assert sensitivity["host_boundary"]["status"] == "strictly_increasing"
    np.testing.assert_allclose(sensitivity["cadence"]["velocity_cm_s"][:3], [0, 2, 2])
    np.testing.assert_allclose(
        sensitivity["host_boundary"]["velocity_cm_s"][:3], [0, 10, 5],
    )
    np.testing.assert_allclose(
        sensitivity["cadence"]["acceleration_cm_s2"][:3], [0, 400, 0], atol=1e-7,
    )
    np.testing.assert_allclose(
        sensitivity["host_boundary"]["acceleration_cm_s2"][:3],
        [0, 10000, -2500], atol=1e-6,
    )
    staged_events = pd.read_csv(out / "clock" / evt.name)
    assert staged_events["time_ms"].tolist() == pytest.approx([3])
    np.testing.assert_allclose(staged["velocity"].iloc[:4], sensitivity["cadence"]["velocity_cm_s"])
    assert sensitivity["events"][0]["cadence_in_recorded_window"] is True
    assert sensitivity["events"][0]["host_boundary_in_recorded_window"] is False
    assert sensitivity["events"][0]["cadence_relative_ms"] == pytest.approx(3)
    assert sensitivity["events"][0]["host_boundary_relative_ms"] == pytest.approx(-4)
    assert audit["prefix_manifest_sha256"] == hashlib.sha256(originals[manifest]).hexdigest()
    for path, content in originals.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize("mutation", ["hash", "event_hash", "token", "row", "trial"])
def test_prefix_manifest_mismatch_fails_closed(tmp_path: Path, mutation: str) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    kin, evt = _pair(raw, ["0", "100", "105", "110", "115"],
                     [9.995, 10, 10.005, 10.01, 10.015])
    manifest = _manifest(tmp_path, kin, evt, 1)
    content = json.loads(manifest.read_text())
    session = content["sessions"][0]
    if mutation in {"hash", "event_hash"}:
        key = "events_sha256" if mutation == "event_hash" else "kinematics_sha256"
        session[key] = "0" * 64
    elif mutation == "trial":
        session["trials"][0]["trial_id"] = "2"
    else:
        row = session["trials"][0]["rows"][0]
        row["raw_ard_time" if mutation == "token" else "source_row_index"] = (
            "00" if mutation == "token" else 1
        )
    manifest.write_text(json.dumps(content))
    before = {p: p.read_bytes() for p in (kin, evt)}
    with pytest.raises(ValueError, match="Experimental clock mapping rejected"):
        adapt_cercus_to_nsmor(
            raw, out, experimental_clock_residual_ms=0.1,
            experimental_clock_prefix_manifest=manifest,
        )
    assert not (out / "clock" / kin.name).exists()
    assert not (out / "clock" / evt.name).exists()
    audit = json.loads((out / "clock" / "clock_timebase_audit.json").read_text())
    assert audit["status"] == "rejected"
    for path, original in before.items():
        assert path.read_bytes() == original


@pytest.mark.parametrize("suffix", [
    ["100", "105", "110", "100000"],
    ["100", "105", "104", "115"],
    ["100", "105", "NaN", "115"],
])
def test_suspect_prefix_never_widens_to_hide_suffix_anomaly(
    tmp_path: Path, suffix: list[str],
) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    kin, evt = _pair(raw, ["NaN"] + suffix, [9.995, 10, 10.005, 10.01, 10.015])
    manifest = _manifest(tmp_path, kin, evt, 1)
    with pytest.raises(ValueError, match="Experimental clock mapping rejected"):
        adapt_cercus_to_nsmor(
            raw, out, experimental_clock_residual_ms=0.1,
            experimental_clock_prefix_manifest=manifest,
        )
    audit = json.loads((out / "clock" / "clock_timebase_audit.json").read_text())
    assert audit["trials"][0]["prefix_rows"] == 1
    assert audit["trials"][0]["suffix_fit"]["status"] == "rejected"
    assert not (out / "clock" / kin.name).exists()


@pytest.mark.parametrize("prefix_hosts", [[10, 10], [10.001, 10.001]])
def test_nonpositive_host_boundary_remains_unresolved_not_clipped(
    tmp_path: Path, prefix_hosts: list[float],
) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    # Suffix batching can put its OLS start below the first observed host arrival.
    kin, evt = _pair(raw, ["NaN", "", "100", "105", "110", "115"],
                     prefix_hosts + [10.001, 10.005, 10.01, 10.015])
    manifest = _manifest(tmp_path, kin, evt, 2)
    adapt_cercus_to_nsmor(
        raw, out, experimental_clock_residual_ms=2,
        experimental_clock_prefix_manifest=manifest,
    )
    audit = json.loads((out / "clock" / "clock_timebase_audit.json").read_text())
    alternative = audit["trials"][0]["sensitivity"]["host_boundary"]
    assert alternative["status"] == "unresolved_nonpositive_interval"
    assert "velocity_cm_s" not in alternative
    from nsmor.pipeline.io import load_kinematics_csv
    loaded = load_kinematics_csv(out / "clock" / kin.name)
    assert loaded["raw_ard_time"].tolist()[:2] == ["NaN", ""]
    assert (np.diff(loaded["time_ms"]) > 0).all()


def test_single_row_is_host_only_without_fit_or_derivative_claim(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    kin, evt = _pair(raw, ["not-a-tick"], [10])
    adapt_cercus_to_nsmor(raw, out, experimental_clock_residual_ms=0.1)
    staged = pd.read_csv(out / "clock" / kin.name)
    assert staged["time_ms"].tolist() == [0]
    assert staged["velocity"].tolist() == [0]
    assert staged["acceleration"].tolist() == [0]
    assert staged["time_source"].tolist() == ["single_row_host_only"]
    audit = json.loads((out / "clock" / "clock_timebase_audit.json").read_text())
    assert audit["trials"][0]["method"] == "single_row_host_only"
    assert "slope" not in audit["trials"][0]
    from nsmor.pipeline.io import extract_trial_data, load_and_concat_sessions
    extracted = extract_trial_data(load_and_concat_sessions(
        [out / "clock" / kin.name], [out / "clock" / evt.name],
    ), "clock", 1)
    assert extracted["clock_provenance"][0]["time_source"] == ["single_row_host_only"]


@pytest.mark.parametrize("ticks,host", [
    (["100", "105"], [10, 10.005]),
    (["100", "105", "110", "115"], [10, 10, 10, 10]),
    (["100", "105", "110", "115"], [10, 10.005, 10.004, 10.015]),
])
def test_degenerate_affine_mapping_rejects(tmp_path: Path, ticks: list[str], host: list[float]) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    kin, _ = _pair(raw, ticks, host)
    with pytest.raises(ValueError, match="Experimental clock mapping rejected"):
        adapt_cercus_to_nsmor(raw, out, experimental_clock_residual_ms=10)
    assert not (out / "clock" / kin.name).exists()


def test_batched_host_and_strict_etl_guard(tmp_path: Path) -> None:
    from nsmor.pipeline.io import extract_trial_data, load_and_concat_sessions
    raw, out = tmp_path / "raw", tmp_path / "out"
    kin, evt = _pair(raw, ["100", "105", "110", "115"], [10, 10.005, 10.005, 10.015])
    adapt_cercus_to_nsmor(raw, out, experimental_clock_residual_ms=10)
    data = load_and_concat_sessions([out / "clock" / kin.name], [out / "clock" / evt.name])
    trial = extract_trial_data(data, "clock", 1)
    assert (np.diff(trial["time_ms"]) > 0).all()
    data["kinematics"].loc[1, "time_ms"] = data["kinematics"].loc[0, "time_ms"]
    with pytest.raises(ValueError, match="repeated or non-finite time_ms"):
        extract_trial_data(data, "clock", 1)


@pytest.mark.parametrize("prefix", [0, 2])
def test_mapping_provenance_survives_actual_dataset_save_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: int,
) -> None:
    import torch
    from scripts.prepare_data import prepare_dataset

    raw, out = tmp_path / "raw", tmp_path / "out"
    host = (10 + np.arange(300) * 0.004).tolist()
    kin, evt = _pair(raw, [str(100 + i * 5) for i in range(300)], host)
    pd.DataFrame({
        "event_name": ["trial_start", "stimulus_onset"],
        "timestamp": [10.0, 10.8], "global_trial_id": [1, 1],
        "details": ['{"type":"baseline_wind"}', "{}"],
    }).to_csv(evt, index=False)
    frame = pd.read_csv(kin)
    frame["dx"] = 0.0
    frame.loc[200:, "stim_state"] = 1
    if prefix:
        frame.loc[:prefix - 1, "ard_time"] = 0
    frame.to_csv(kin, index=False)
    manifest = _manifest(tmp_path, kin, evt, prefix) if prefix else None
    adapt_cercus_to_nsmor(
        raw, out, experimental_clock_residual_ms=0.001,
        experimental_clock_prefix_manifest=manifest,
    )
    monkeypatch.setattr("scripts.prepare_data.resolve_group_folds", lambda *a, **k: 2)
    monkeypatch.setattr(
        "scripts.prepare_data.train_mcmc_cross_fitted",
        lambda *a, **k: (np.full((1, 4), 0.25), [_UniformPrior()], []),
    )
    destination = tmp_path / "dataset.pt"
    prepare_dataset(out, destination, dt_ms=4.0)
    saved = torch.load(destination, weights_only=False)
    provenance = saved["source_clock_provenance"][0][0]
    assert provenance["source_row_indices"] == list(range(300))
    assert provenance["time_source"] == (
        ["experimental_prefix_cadence_estimate"] * prefix
        + ["experimental_affine_estimate"] * (300 - prefix)
    )
    if manifest:
        assert provenance["prefix_manifest_sha256"] == hashlib.sha256(
            manifest.read_bytes(),
        ).hexdigest()
        assert provenance["scientific_acceptance"] == "unresolved"
    assert provenance["events_sha256"] == hashlib.sha256(evt.read_bytes()).hexdigest()
    assert provenance["raw_ard_time"][0] == ("0" if prefix else "100")
    assert provenance["coordinate_scope"] == "source_CSV_rows_not_resampled_tensor_frames"
    audit_bytes = Path(provenance["audit_path"]).read_bytes()
    assert hashlib.sha256(audit_bytes).hexdigest() == provenance["audit_sha256"]
    assert saved["X_seqs"][0].shape[1] == 8
    # Trial-level source rows remain distinct from baseline padding and crops.
    from nsmor.nsmor_dataloader import NSMoRDataset
    sequence = saved["X_seqs"][0].copy()
    sequence[:, 4:8] = 0  # Supply priors at the public dataloader boundary.
    dataset = NSMoRDataset(
        [(sequence, saved["Y_seqs"][0], 0)], np.full((1, 4), 0.25),
        max_seq_len=32, pre_anchor_frames=8, anchor_frames=[1625],
    )
    cropped, _ = dataset[0]
    assert cropped.shape == (32, 8)
    assert len(provenance["source_row_indices"]) == 300
    assert len(saved["X_seqs"][0]) != 300
    assert dataset.source_indices == [0]
