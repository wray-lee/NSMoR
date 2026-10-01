"""Known-answer causal model-grid and source-integrity checks."""
from __future__ import annotations

from copy import deepcopy

from pathlib import Path

import numpy as np
import pytest
import torch

from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint
from nsmor.pipeline.resampling import (
    resample_trial_for_model, resolve_model_anchor_frame,
)


def trial(times: list[float]) -> dict:
    axis = np.asarray(times, dtype=float)
    result = {name: axis.copy() for name in (
        "x_pos", "y_pos", "heading", "velocity", "acceleration",
        "visual_angle", "wind_state", "l_v_ratio",
    )}
    result.update(time_ms=axis, session_id="s", trial_id=1,
                  event_times=np.array([10.0]),
                  event_types=np.array(["stimulus_onset"]),
                  event_values=np.array([""]),
                  clock_provenance=[{"source_row_indices": [0, 1, 2]}])
    return result


def test_five_to_four_ms_grid_is_causal_and_within_support() -> None:
    source = trial([0, 5, 10, 15, 20, 25])
    source["visual_angle"] = np.array([0, 0, 10, 20, 30, 40.])
    source["wind_state"] = np.array([0, 0, 1, 1, 1, 0.])
    before = deepcopy(source)
    model = resample_trial_for_model(source, 4)
    np.testing.assert_array_equal(model["time_ms"], [0, 4, 8, 12, 16, 20, 24])
    np.testing.assert_array_equal(model["velocity"], [0, 0, 5, 10, 15, 20, 20])
    np.testing.assert_array_equal(model["visual_angle"][:3], [0, 0, 0])
    np.testing.assert_array_equal(model["wind_state"][:3], [0, 0, 0])
    # Previous model frame at 16ms must not use the future source at 20ms.
    assert model["velocity"][4] == 15
    assert model["model_grid_provenance"]["omitted_tail_ms"] == 1
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(source[key], value)
        else:
            assert source[key] == value
    for key in ("event_times", "event_types", "event_values"):
        np.testing.assert_array_equal(model[key], source[key])
    assert model["clock_provenance"] == source["clock_provenance"]
    assert not np.shares_memory(model["velocity"], source["velocity"])


def test_aligned_grid_and_singleton_are_unchanged() -> None:
    for times in ([0, 4, 8], [9]):
        source = trial(times)
        model = resample_trial_for_model(source, 4)
        for key in ("time_ms", "velocity", "heading", "wind_state"):
            np.testing.assert_array_equal(model[key], source[key])
        assert model["model_grid_provenance"]["singleton"] == (len(times) == 1)


@pytest.mark.parametrize("times", [[], [0, 0], [2, 1], [0, np.nan]])
def test_invalid_source_time_rejected(times: list[float]) -> None:
    with pytest.raises(ValueError, match="Source time_ms"):
        resample_trial_for_model(trial(times), 4)


@pytest.mark.parametrize("dt", [0, -1, np.inf, np.nan, True])
def test_invalid_step_rejected(dt: float) -> None:
    with pytest.raises(ValueError, match="dt_ms"):
        resample_trial_for_model(trial([0, 5]), dt)


def test_channel_shape_and_nonfinite_rejected() -> None:
    for value in (np.array([1.]), np.array([0., np.nan])):
        source = trial([0, 5])
        source["velocity"] = value
        with pytest.raises(ValueError, match="velocity"):
            resample_trial_for_model(source, 4)


@pytest.mark.parametrize("channel", ["visual_angle", "wind_state"])
@pytest.mark.parametrize("times,active_rows", [
    ([0, 5, 10], [2]),  # Activation in the unsupported final fractional step.
    ([0, 1, 2, 5, 8], [1, 4]),  # First pulse lost; later pulse survives.
])
def test_grid_rejects_lost_stimulus_intervals(
    channel: str, times: list[float], active_rows: list[int],
) -> None:
    source = trial(times)
    source["visual_angle"][:] = 0
    source["wind_state"][:] = 0
    source[channel][active_rows] = 1
    with pytest.raises(ValueError, match=f"active {channel} interval"):
        resample_trial_for_model(source, 4)
    np.testing.assert_array_equal(source[channel][active_rows], 1)


@pytest.mark.parametrize("channel", ["visual_angle", "wind_state"])
@pytest.mark.parametrize("times,active_rows,held,error", [
    ([0, 4, 5, 6, 8, 12], [1, 3, 4], None, "merges"),
    ([0, 5, 10], [2], None, "loses"),
    ([0, 4, 8, 12, 16], [1, 3], [0, 1, 0, 1, 0], None),
])
def test_grid_preserves_distinct_active_runs(
    channel: str, times: list[float], active_rows: list[int],
    held: list[int] | None, error: str | None,
) -> None:
    source = trial(times)
    source["visual_angle"][:] = 0
    source["wind_state"][:] = 0
    source[channel][active_rows] = 1
    before = source[channel].copy()
    if error is not None:
        with pytest.raises(ValueError, match=f"{error}.*active {channel} interval"):
            resample_trial_for_model(source, 4)
    else:
        model = resample_trial_for_model(source, 4)
        np.testing.assert_array_equal(model[channel], held)
    np.testing.assert_array_equal(source[channel], before)


def test_etl_persists_unavailable_trials_and_model_grid(tmp_path, monkeypatch) -> None:
    from scripts import prepare_data as prep
    from tests.test_pipeline import _make_label_audit_csvs
    import torch

    raw = tmp_path / "raw"
    _make_label_audit_csvs(raw, n_sessions=5)
    original_load = prep._load_trials_and_diagnostics

    def load_with_unavailable(*args, **kwargs):
        trials, diagnostics, n_kin, n_evt = original_load(*args, **kwargs)
        unavailable = deepcopy(trials[0])
        unavailable["trial_id"] = 999
        mask = unavailable["event_types"] != "stimulus_onset"
        for key in ("event_types", "event_times", "event_values"):
            unavailable[key] = unavailable[key][mask]
        trials.append(unavailable)
        return trials, diagnostics, n_kin, n_evt

    monkeypatch.setattr(prep, "_load_trials_and_diagnostics", load_with_unavailable)
    output = tmp_path / "dataset.pt"
    prep.prepare_dataset(raw, output, dt_ms=4.0)
    data = torch.load(output, weights_only=False)
    ledger = data["labeling_eligibility"]
    assert len(ledger) == 26
    assert sum(row["status"] == "labeled" for row in ledger) == 25
    missing = [row for row in ledger if row["trial_id"] == 999]
    assert len(missing) == 1
    assert missing[0]["status"] == "unavailable_no_stimulus_anchor"
    assert missing[0]["label"] is None
    assert 999 not in data["trial_ids"]
    assert len(data["X_seqs"]) == len(data["model_grid_provenance"]) == 25
    assert data["model_dt_ms"] == 4.0
    for x, record in zip(data["X_seqs"], data["model_grid_provenance"]):
        assert len(x) == record["model_n"] + record["synthetic_prepend_frames"]
        assert record["method"] == "causal_previous_source_sample_hold"
        assert record["model_end_ms"] <= record["source_end_ms"]


@pytest.mark.parametrize("origin,n", [(1.1, 17), (1000.1, 101)])
def test_fractional_origin_keeps_aligned_endpoint(origin: float, n: int) -> None:
    times = origin + np.arange(n) * 4
    source = trial(times.tolist())
    model = resample_trial_for_model(source, 4)
    np.testing.assert_array_equal(model["time_ms"], times)
    np.testing.assert_array_equal(model["velocity"], source["velocity"])
    assert model["model_grid_provenance"]["omitted_tail_ms"] == 0


def test_nonaligned_endpoint_never_extends_or_rounds_event_back() -> None:
    times = 1.1 + np.arange(17) * 4
    times[-1] = np.nextafter(times[-1], -np.inf)
    model = resample_trial_for_model(trial(times.tolist()), 4)
    np.testing.assert_array_equal(model["time_ms"], 1.1 + np.arange(16) * 4)
    assert np.all(model["time_ms"] <= times[-1])
    record = dict(model["model_grid_provenance"], source_anchor_ms=times[-1],
                  source_anchor_rule="looming_collision")
    with pytest.raises(ValueError, match="no model frame"):
        resolve_model_anchor_frame(record)


@pytest.mark.parametrize("missing", [
    None, "source_anchor_ms", "source_anchor_rule", "origin_ms", "dt_ms",
    "source_n", "model_n", "source_end_ms", "model_end_ms", "synthetic_prepend_frames",
])
def test_restricted_loader_requires_complete_grid_anchor(
    tmp_path: Path, missing: str | None,
) -> None:
    record = {
        "method": "causal_previous_source_sample_hold",
        "origin_ms": 0., "dt_ms": 4., "source_n": 23, "model_n": 28,
        "source_end_ms": 110., "model_end_ms": 108.,
        "synthetic_prepend_frames": 0, "source_anchor_ms": 105.,
        "source_anchor_rule": "looming_collision",
    }
    if missing is not None:
        del record[missing]
    dataset = {"X_seqs": [np.zeros((28, 8), dtype=np.float32)],
               "Y_seqs": [np.zeros(28, dtype=np.float32)], "lengths": [28],
               "anchor_frames": [2 if missing is None else 27], "model_dt_ms": 4.,
               "model_grid_provenance": [record]}
    path = tmp_path / "grid.pt"
    torch.save(dataset, path)
    with pytest.raises(ValueError, match="source anchor|model grid"):
        load_dataset_with_fingerprint(path)


@pytest.mark.parametrize("field,value", [
    ("source_anchor_ms", True), ("source_anchor_ms", "105"),
    ("source_anchor_ms", np.nan), ("source_anchor_ms", -1.),
    ("source_anchor_ms", 111.), ("source_anchor_ms", 109.),
    ("source_anchor_rule", None), ("source_anchor_rule", "peak"),
    ("origin_ms", False), ("dt_ms", "4"), ("dt_ms", np.inf),
    ("model_n", 28.5), ("model_n", True),
    ("synthetic_prepend_frames", 0.5), ("synthetic_prepend_frames", False),
    ("model_end_ms", np.nan), ("model_end_ms", 104.),
    ("source_end_ms", True), ("source_end_ms", 107.),
    ("anchor_frames", [27.]), ("anchor_frames", [True]),
    ("anchor_frames", []), ("lengths", None), ("lengths", [27]),
    ("lengths", [28.]), ("Y_seqs", []), ("model_grid_provenance", []),
    ("model_grid_provenance", [None]), ("model_grid_provenance", None),
])
def test_restricted_loader_rejects_invalid_grid_anchor(
    tmp_path: Path, field: str, value: object,
) -> None:
    record = {
        "method": "causal_previous_source_sample_hold",
        "origin_ms": 0., "dt_ms": 4., "source_n": 23, "model_n": 28,
        "source_end_ms": 110., "model_end_ms": 108.,
        "synthetic_prepend_frames": 0, "source_anchor_ms": 105.,
        "source_anchor_rule": "looming_collision",
    }
    dataset = {"X_seqs": [np.zeros((28, 8), dtype=np.float32)],
               "Y_seqs": [np.zeros(28, dtype=np.float32)], "lengths": [28],
               "anchor_frames": [27], "model_dt_ms": 4.,
               "model_grid_provenance": [record]}
    if field in dataset:
        dataset[field] = value
    else:
        record[field] = value
    path = tmp_path / "invalid-grid.pt"
    torch.save(dataset, path)
    with pytest.raises(ValueError):
        load_dataset_with_fingerprint(path)


def test_restricted_loader_accepts_source_anchor_and_genuine_legacy(
    tmp_path: Path,
) -> None:
    x = np.zeros((28, 8), dtype=np.float32)
    x[2, 0] = 180.  # Legacy peak differs from the authoritative event.
    dataset = {"X_seqs": [x], "Y_seqs": [np.zeros(28, dtype=np.float32)],
               "lengths": [28], "anchor_frames": [27], "model_dt_ms": 4.,
               "model_grid_provenance": [{
                   "method": "causal_previous_source_sample_hold",
                   "origin_ms": 0., "dt_ms": 4., "source_n": 23, "model_n": 28,
                   "source_end_ms": 110., "model_end_ms": 108.,
                   "synthetic_prepend_frames": 0, "source_anchor_ms": 105.,
                   "source_anchor_rule": "looming_collision",
               }]}
    path = tmp_path / "dataset.pt"
    torch.save(dataset, path)
    loaded, _ = load_dataset_with_fingerprint(path)
    assert loaded["anchor_frames"] == [27]
    np.testing.assert_array_equal(loaded["X_seqs"][0], x)
    del dataset["model_grid_provenance"]
    del dataset["model_dt_ms"]
    dataset["anchor_frames"] = [2]
    torch.save(dataset, path)
    loaded, _ = load_dataset_with_fingerprint(path)
    assert loaded["anchor_frames"] == [2]
    np.testing.assert_array_equal(loaded["X_seqs"][0], x)


def clock_dataset(dt_ms: float = 4.0) -> dict[str, object]:
    """A visual artifact at frame 2 must not replace the source collision."""
    x = np.zeros((28, 8), dtype=np.float32)
    x[2, 0] = 180.
    return {
        "X_seqs": [x.copy() for _ in range(8)],
        "Y_seqs": [np.arange(28, dtype=np.float32) for _ in range(8)],
        "lengths": np.full(8, 28, dtype=np.int64), "labels": [0] * 8,
        "session_ids": [f"recording_{i}_session_1" for i in range(8)],
        "mcmc_priors": np.full((8, 4), 0.25),
        "mcmc_prior_provenance": "oof_5fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
        "pipeline_semantics_version": "2.2", "model_dt_ms": dt_ms,
        "anchor_frames": [int(np.ceil(100. / dt_ms))] * 8,
        "model_grid_provenance": [{
            "origin_ms": 0., "dt_ms": dt_ms, "source_n": 28, "model_n": 28,
            "source_end_ms": 27 * dt_ms, "model_end_ms": 27 * dt_ms,
            "synthetic_prepend_frames": 0, "source_anchor_ms": 100.,
            "source_anchor_rule": "looming_collision",
        } for _ in range(8)],
    }


def test_record_interval_must_match_dataset_declaration(tmp_path: Path) -> None:
    dataset = clock_dataset(10.)
    dataset["model_dt_ms"] = 4.
    path = tmp_path / "inconsistent-dt.pt"
    torch.save(dataset, path)
    with pytest.raises(ValueError, match="dt_ms.*model_dt_ms"):
        load_dataset_with_fingerprint(path)


@pytest.mark.parametrize("dt_ms", [4., 8., 10.])
def test_declared_grid_binds_to_expected_model_clock(
    tmp_path: Path, dt_ms: float,
) -> None:
    path = tmp_path / "clock.pt"
    torch.save(clock_dataset(dt_ms), path)
    before = path.read_bytes()
    loaded, _ = load_dataset_with_fingerprint(path, expected_dt_ms=dt_ms)
    assert loaded["model_dt_ms"] == dt_ms
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="model_dt_ms.*expected_dt_ms"):
        load_dataset_with_fingerprint(path, expected_dt_ms=dt_ms * 2)


@pytest.mark.parametrize("expected", [True, 0., -4., np.nan, np.inf, "4"])
def test_invalid_consumer_clock_rejected(tmp_path: Path, expected: object) -> None:
    path = tmp_path / "clock.pt"
    torch.save(clock_dataset(), path)
    with pytest.raises(ValueError, match="expected_dt_ms"):
        load_dataset_with_fingerprint(path, expected_dt_ms=expected)


@pytest.mark.parametrize("marker", [
    "model_dt_ms", "model_grid_provenance", "labeling_eligibility",
    "source_clock_provenance",
])
def test_modern_marker_cannot_downgrade_to_held_peak(
    tmp_path: Path, marker: str,
) -> None:
    dataset = clock_dataset()
    for key in ("model_dt_ms", "model_grid_provenance", "anchor_frames"):
        if key != marker:
            dataset.pop(key)
    if marker in ("labeling_eligibility", "source_clock_provenance"):
        dataset[marker] = []
    path = tmp_path / "stripped.pt"
    torch.save(dataset, path)
    with pytest.raises(ValueError, match="modern.*clock|model grid"):
        load_dataset_with_fingerprint(path)


@pytest.mark.parametrize("entry", ["restricted_loader", "dynamics", "train"])
@pytest.mark.parametrize("compact", [False, True])
def test_source_clock_only_cannot_fall_back_to_peak(
    tmp_path: Path, entry: str, compact: bool,
) -> None:
    """Source-clock producers cannot lose the grid and use frame-2 geometry."""
    from nsmor.pipeline.clock_storage import pack_clock_provenance
    from nsmor.config_parser import ExperimentConfig
    from scripts import analyze_dynamics, train

    dataset = clock_dataset()
    for key in ("model_dt_ms", "model_grid_provenance", "anchor_frames"):
        dataset.pop(key)
    records = [{
        "source_row_indices": [0, 1], "raw_sys_time": ["0", "0.004"],
        "raw_ard_time": ["100", "104"], "time_source": ["host", "host"],
    }]
    stored = pack_clock_provenance(records) if compact else records
    dataset["source_clock_provenance"] = [stored] * 8
    path = tmp_path / "source-clock-only.pt"
    torch.save(dataset, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="modern.*clock|model grid"):
        if entry == "restricted_loader":
            load_dataset_with_fingerprint(path, expected_dt_ms=4.)
        elif entry == "dynamics":
            analyze_dynamics.load_dataset(path)
        else:
            config = ExperimentConfig()
            config.training.num_workers = 0
            train.build_dataloaders(config, dataset_path=str(path))
    assert path.read_bytes() == before


@pytest.mark.parametrize("entry", ["eager", "lazy", "stats"])
@pytest.mark.parametrize("dt_ms", [8., 10.])
def test_actual_train_ingestion_rejects_consumer_clock_mismatch(
    tmp_path: Path, entry: str, dt_ms: float,
) -> None:
    from nsmor.config_parser import ExperimentConfig
    from scripts import train

    path = tmp_path / "clock.pt"
    torch.save(clock_dataset(dt_ms), path)
    config = ExperimentConfig()
    config.model.dt_ms = 4.
    config.training.normalize_targets = True
    with pytest.raises(ValueError, match="model_dt_ms.*expected_dt_ms"):
        if entry == "stats":
            train.compute_target_stats(str(path), config)
        else:
            train.build_dataloaders(
                config, dataset_path=str(path), use_lazy_loading=entry == "lazy",
            )


def clock_checkpoint(tmp_path: Path, dataset_path: Path, dt_ms: float) -> Path:
    """Use a real content-bound checkpoint, not a stubbed analysis loader."""
    import hashlib
    from nsmor.model_nsmor_core import NSMoRCore

    model = NSMoRCore(hidden_dim=4, dt_ms=dt_ms, dropout=0.)
    checkpoint = tmp_path / "model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"model": {"hidden_dim": 4, "dt_ms": dt_ms, "dropout": 0.}},
        "pipeline_semantics_version": "2.2", "is_nested_cv": False,
        "dataset_source_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dataset_path": str(dataset_path),
        "mcmc_prior_provenance": "oof_5fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
        "validation_scope": "diagnostic_global_oof",
    }, checkpoint)
    return checkpoint


@pytest.mark.parametrize("phase", ["C", "D", "E", "F", "G", "H"])
@pytest.mark.parametrize("dt_ms", [8., 10.])
def test_actual_analysis_ingestion_rejects_checkpoint_clock_mismatch(
    tmp_path: Path, phase: str, dt_ms: float,
) -> None:
    from nsmor.analysis.prediction_units import load_model_from_checkpoint
    from tests.test_analysis_nested_prior_inputs import phase_inputs

    path = tmp_path / "clock.pt"
    torch.save(clock_dataset(dt_ms), path)
    checkpoint = clock_checkpoint(tmp_path, path, 4.)
    model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
    with pytest.raises(ValueError, match="model_dt_ms.*expected_dt_ms"):
        phase_inputs(phase, checkpoint, path, None, model)


@pytest.mark.parametrize("dt_ms", [4., 8.])
def test_real_training_and_analysis_keep_authoritative_anchor(
    tmp_path: Path, dt_ms: float,
) -> None:
    from nsmor.config_parser import ExperimentConfig
    from nsmor.analysis.prediction_units import load_model_from_checkpoint
    from scripts import train, analyze_dynamics

    dataset = clock_dataset(dt_ms)
    # Exercise the exact source collision 105ms -> frame27 at dt4.
    for record in dataset["model_grid_provenance"]:
        record["source_anchor_ms"] = 105.
    anchor = int(np.ceil(105. / dt_ms))
    dataset["anchor_frames"] = [anchor] * 8
    path = tmp_path / "clock.pt"
    torch.save(dataset, path)
    config = ExperimentConfig()
    config.model.dt_ms = dt_ms
    config.training.num_workers = 0
    config.training.max_seq_len = None
    loaders = train.build_dataloaders(config, dataset_path=str(path))
    for loader in loaders:
        assert loader.dataset.anchor_frames == [anchor] * len(loader.dataset)
        x, y = loader.dataset[0][-2:]
        assert x.shape == (28, 8) and y.shape == (28,)
        assert x[2, 0] == 180.
    checkpoint = clock_checkpoint(tmp_path, path, dt_ms)
    model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
    loader, _ = analyze_dynamics.load_dataset(
        path, max_seq_len=None, checkpoint_model=model,
    )
    assert loader.dataset.anchor_frames == [anchor] * 8


def test_legacy_clock_is_unverified_not_fabricated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    dataset = clock_dataset()
    for key in ("model_grid_provenance", "model_dt_ms", "anchor_frames"):
        dataset.pop(key)
    path = tmp_path / "legacy-2.2.pt"
    torch.save(dataset, path)
    loaded, _ = load_dataset_with_fingerprint(path, expected_dt_ms=13.)
    assert "model_dt_ms" not in loaded
    assert "unverified" in caplog.text.lower()
    from nsmor.pipeline.conditions import derive_anchor_frames
    assert derive_anchor_frames(loaded["X_seqs"]) == [2] * 8


@pytest.mark.parametrize("bad_dt", [True, 0., -1., np.nan, np.inf, "4"])
@pytest.mark.parametrize("field", ["model_dt_ms", "record_dt_ms"])
def test_invalid_declared_clock_rejected(
    tmp_path: Path, bad_dt: object, field: str,
) -> None:
    dataset = clock_dataset()
    if field == "model_dt_ms":
        dataset[field] = bad_dt
    else:
        dataset["model_grid_provenance"][0]["dt_ms"] = bad_dt
    path = tmp_path / "bad-clock.pt"
    torch.save(dataset, path)
    with pytest.raises(ValueError, match="dt_ms|model grid"):
        load_dataset_with_fingerprint(path)


def test_one_trial_clock_cannot_disagree_with_other_records(tmp_path: Path) -> None:
    dataset = clock_dataset()
    other = clock_dataset(10.)
    dataset["model_grid_provenance"][1] = other["model_grid_provenance"][1]
    dataset["anchor_frames"][1] = other["anchor_frames"][1]
    path = tmp_path / "mixed-clocks.pt"
    torch.save(dataset, path)
    with pytest.raises(ValueError, match="trial 1.*dt_ms.*model_dt_ms"):
        load_dataset_with_fingerprint(path)


@pytest.mark.parametrize("entry", ["train", "analysis"])
def test_sidecar_loss_fails_before_held_peak_fallback(
    tmp_path: Path, entry: str,
) -> None:
    from nsmor.config_parser import ExperimentConfig
    from scripts import train, analyze_dynamics

    dataset = clock_dataset()
    dataset.pop("model_grid_provenance")
    dataset.pop("anchor_frames")
    path = tmp_path / "stripped.pt"
    torch.save(dataset, path)
    with pytest.raises(ValueError, match="modern.*clock|model grid"):
        if entry == "train":
            train.build_dataloaders(ExperimentConfig(), dataset_path=str(path))
        else:
            analyze_dynamics.load_dataset(path)


@pytest.mark.parametrize("entry", ["train", "analysis"])
def test_legacy_2_2_custom_model_clock_keeps_peak_compatibility(
    tmp_path: Path, entry: str, caplog: pytest.LogCaptureFixture,
) -> None:
    from nsmor.config_parser import ExperimentConfig
    from nsmor.analysis.prediction_units import load_model_from_checkpoint
    from scripts import train, analyze_dynamics

    dataset = clock_dataset()
    for key in ("model_grid_provenance", "model_dt_ms", "anchor_frames"):
        dataset.pop(key)
    path = tmp_path / "legacy.pt"
    torch.save(dataset, path)
    if entry == "train":
        config = ExperimentConfig()
        config.model.dt_ms = 13.
        config.training.num_workers = 0
        loader = train.build_dataloaders(config, dataset_path=str(path))[0]
    else:
        checkpoint = clock_checkpoint(tmp_path, path, 13.)
        model = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
        loader = analyze_dynamics.load_dataset(path, checkpoint_model=model)[0]
    assert loader.dataset.anchor_frames == [2] * len(loader.dataset)
    assert "unverified" in caplog.text.lower()


@pytest.mark.parametrize("entry", ["subset", "evaluate_prior"])
@pytest.mark.parametrize("defect", ["missing_sidecars", "record_mismatch"])
def test_model_free_consumers_reject_broken_clock_contract(
    tmp_path: Path, entry: str, defect: str,
) -> None:
    from scripts.make_subset_dataset import main
    from scripts.evaluate_nested_prior import generate_nested_priors

    dataset = clock_dataset(10.)
    if defect == "missing_sidecars":
        dataset.pop("model_grid_provenance")
        dataset.pop("anchor_frames")
    else:
        dataset["model_dt_ms"] = 4.
    path, output = tmp_path / "clock.pt", tmp_path / "subset.pt"
    torch.save(dataset, path)
    with pytest.raises(ValueError, match="model grid|dt_ms.*model_dt_ms"):
        if entry == "subset":
            main(["--input", str(path), "--output", str(output),
                  "--n_recording_prefixes", "2"])
        else:
            generate_nested_priors(dataset=None, source_dataset_path=path)
    assert not output.exists()


def test_jax_training_ingestion_rejects_consumer_clock_mismatch(
    tmp_path: Path,
) -> None:
    from nsmor.config_parser import ExperimentConfig
    from nsmor.jax.train import train_jax

    path = tmp_path / "clock.pt"
    torch.save(clock_dataset(8.), path)
    config = ExperimentConfig()
    config.model.dt_ms = 4.
    with pytest.raises(ValueError, match="model_dt_ms.*expected_dt_ms"):
        train_jax(config, dataset_path=str(path), output_dir=str(tmp_path / "out"))
    assert not list((tmp_path / "out").iterdir())


def test_jax_loader_rejects_consumer_clock_mismatch(tmp_path: Path) -> None:
    from nsmor.jax.dataloader import load_nsmor_dataset

    path = tmp_path / "clock.pt"
    torch.save(clock_dataset(8.), path)
    with pytest.raises(ValueError, match="model_dt_ms.*expected_dt_ms"):
        load_nsmor_dataset(path, expected_dt_ms=4.)
