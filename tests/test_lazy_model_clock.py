"""T8 — enhanced lazy producer/reader/converter model-clock contract.

The real producer emits ~4.997 ms source cadence while the model grid is
4.0 ms.  The enhanced :class:`nsmor.pipeline.io.ClockAwareLazyDataset`
resamples each trial on demand onto the declared model grid (reusing the
shared :func:`nsmor.pipeline.resampling.resample_trial_for_model`), the
converter emits the full eager grid/anchor contract, and the restricted
loader binds it to the consumer clock.  Source time stays source evidence;
model samples are causal-hold estimates and pure-wind leading zeros are
synthetic.
"""
from __future__ import annotations

import builtins
import copy
import io
import json
import logging
import math
import pickle
import sys
import types
from pathlib import Path, PosixPath
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from nsmor.config import FeatureConfig
from nsmor.data_extractor import _compute_pure_wind_prepend_frames
from nsmor.lazy_dataloader import NSMoRLazyDataset
from nsmor.pipeline.io import (
    ClockAwareLazyDataset,
    _SourceKinematics,
    extract_trial_data,
)
from nsmor.pipeline.nested_prior import (
    load_artifact_bytes,
    load_dataset_with_fingerprint,
)
from nsmor.pipeline.resampling import (
    LAZY_MODEL_CLOCK_SCHEMA,
    build_lazy_model_clock_contract,
    resolve_model_anchor_frame,
    validate_lazy_model_clock_contract,
)

KIN_COLUMNS = [
    "session_id", "trial_id", "time_ms", "x_pos", "y_pos", "heading",
    "velocity", "acceleration", "visual_angle", "wind_state", "l_v_ratio",
]
SOURCE_DT_MS = 4.997
MODEL_DT_MS = 4.0
LV_RATIO_MS = 120.0
# Collision at lv / tan(init/2) = 1716 ms, inside a 600-frame (~3 s) trial.
INIT_DEG = 8.0


class _AmbientOnlyUnlisted:
    """Module-level, picklable, and deliberately absent from the test list."""


class _GeneratedReducer:
    """A picklable reducer outside the list; must never execute on load."""

    def __reduce__(self):
        return eval, ("raise AssertionError('reducer executed')",)


def _write_session(
    raw_dir: Path,
    session_id: str,
    *,
    trial_id: int = 0,
    dt_ms: float = SOURCE_DT_MS,
    n_frames: int = 600,
    velocity: float = 0.1,
    wind_from: int | None = None,
    looming: bool = False,
    subdir: str | None = None,
    start_ms: float = 0.0,
    declare_start: bool = True,
) -> None:
    """Write one session pair: pure-wind, visual-looming, or no-stimulus."""
    directory = raw_dir / (subdir or session_id)
    directory.mkdir(parents=True)
    time_ms = start_ms + np.arange(n_frames, dtype=np.float64) * dt_ms
    if looming:
        # theta(t) = 2*atan(lv / (t_col - t)); collision declared via lv/init.
        t_col_ms = LV_RATIO_MS / math.tan(math.radians(INIT_DEG / 2.0))
        delta_s = np.clip((t_col_ms - time_ms) / 1000.0, 1e-6, None)
        visual_angle = np.degrees(2.0 * np.arctan((LV_RATIO_MS / 1000.0) / delta_s))
        visual_angle[time_ms > t_col_ms] = 179.0
    else:
        visual_angle = np.zeros(n_frames, dtype=np.float64)
    kin = []
    for frame in range(n_frames):
        kin.append({
            "session_id": session_id, "trial_id": trial_id,
            "time_ms": float(time_ms[frame]), "x_pos": 0.0, "y_pos": 0.0,
            "heading": 0.0, "velocity": float(velocity), "acceleration": 0.0,
            "visual_angle": float(visual_angle[frame]),
            "wind_state": float(frame >= wind_from) if wind_from is not None else 0.0,
            "l_v_ratio": LV_RATIO_MS if looming else 120.0,
        })
    if looming:
        start_details = {"type": "baseline_visual", "lv_ratio_ms": LV_RATIO_MS,
                         "init_deg": INIT_DEG}
        events = [
            {"session_id": session_id, "trial_id": trial_id, "time_ms": 0.0,
             "event_type": "trial_start", "event_value": json.dumps(start_details)},
            {"session_id": session_id, "trial_id": trial_id, "time_ms": 0.0,
             "event_type": "phase_transition",
             "event_value": json.dumps({"from_phase": "TrialStart",
                                        "to_phase": "Looming"})},
            {"session_id": session_id, "trial_id": trial_id, "time_ms": 0.0,
             "event_type": "stimulus_onset",
             "event_value": json.dumps({"source": "kinematics_injected"})},
        ]
    else:
        if declare_start:
            events = [{"session_id": session_id, "trial_id": trial_id,
                       "time_ms": float(start_ms), "event_type": "trial_start",
                       "event_value": '{"target_ttc_ms": -119.0}'}]
        else:
            events = [{"session_id": session_id, "trial_id": trial_id,
                       "time_ms": float(start_ms),
                       "event_type": "phase_transition",
                       "event_value": '{"from_phase": "TrialStart", "to_phase": "Tracking"}'}]
        if wind_from is not None:
            events.append({"session_id": session_id, "trial_id": trial_id,
                           "time_ms": wind_from * dt_ms,
                           "event_type": "stimulus_onset", "event_value": ""})
    pd.DataFrame(kin, columns=KIN_COLUMNS).to_csv(
        directory / f"{session_id}_kinematics.csv", index=False)
    pd.DataFrame(events).to_csv(directory / f"{session_id}_events.csv", index=False)


def _produce_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_dir: Path,
) -> Path:
    """Run prepare_metadata over *raw_dir*; return the metadata path."""
    from scripts import prepare_metadata

    metadata_path = tmp_path / "metadata.pt"
    monkeypatch.setattr(prepare_metadata, "resolve_group_folds", lambda *a, **k: 2)
    monkeypatch.setattr(
        prepare_metadata, "train_mcmc_cross_fitted",
        lambda snapshots, *a, **k: (np.full((len(snapshots), 4), 0.25), [], []),
    )
    monkeypatch.setattr(sys, "argv", [
        "prepare_metadata.py", "--raw_dir", str(raw_dir),
        "--output", str(metadata_path),
    ])
    prepare_metadata.main()
    return metadata_path


def _spec(metadata_path: Path, index: int = 0) -> dict:
    metadata = load_artifact_bytes(metadata_path.read_bytes())
    return metadata["trial_specs"][index]


# Restricted allowlist for test-authored, trusted in-memory wrappers only.  It
# is applied under the process-global ``safe_globals`` context (the registry
# ``load_artifact_bytes`` also restores), never as a production allowlist.  The
# warm entries (``_SourceKinematics``, pandas/numpy/pathlib) are present only
# because a *read* wrapper holds live pandas frames; the cold entry set is the
# four classes alone.  See ``_warm_payload_globals``.
_TEST_LOCAL_SAFE_GLOBALS = [
    ClockAwareLazyDataset, NSMoRLazyDataset, FeatureConfig, _SourceKinematics,
    types.SimpleNamespace, PosixPath, builtins.getattr, builtins.slice,
    np.ndarray, np.dtype,
    (np._core if hasattr(np, "_core") else np.core).multiarray._reconstruct,
    pd.DataFrame, pd.Index, pd.RangeIndex, pd.StringDtype, pd.arrays.StringArray,
    pd.core.indexes.base._new_Index, pd.core.internals.managers.BlockManager,
    pd._libs.arrays.__pyx_unpickle_NDArrayBacked,
    pd._libs.internals._unpickle_block,
    np.dtypes.Float64DType, np.dtypes.Int64DType, np.dtypes.ObjectDType,
]


def _warm_payload_globals(obj: object) -> list[str]:
    """Statically scan *obj*'s pickle for unlisted globals (no deserialize)."""
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return torch.serialization.get_unsafe_globals_in_checkpoint(
        io.BytesIO(buffer.getvalue())
    )


def _restricted_roundtrip(obj: object) -> object:
    """Round-trip *obj* through a restricted unpickler under a test-local list.

    The list accepts both the cold class set and the warm payload a worker
    pickles after serving an item (live pandas frames, ``_SourceKinematics``).
    Any global outside ``_TEST_LOCAL_SAFE_GLOBALS`` still fails closed, and the
    ambient registry is restored exactly on both success and failure.
    """
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    previous = torch.serialization.get_safe_globals()
    torch.serialization.clear_safe_globals()
    try:
        with torch.serialization.safe_globals(_TEST_LOCAL_SAFE_GLOBALS):
            return torch.load(io.BytesIO(buffer.getvalue()), weights_only=True)
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(previous)


def test_restricted_roundtrip_rejects_ambient_only_unlisted_class() -> None:
    """The test-local list wins: an ambient-only class must still be refused."""
    previous = torch.serialization.get_safe_globals()
    torch.serialization.add_safe_globals([_AmbientOnlyUnlisted])
    try:
        with pytest.raises(pickle.UnpicklingError):
            _restricted_roundtrip(_AmbientOnlyUnlisted())
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(previous)


def test_restricted_roundtrip_rejects_malicious_reducer() -> None:
    """A reducer outside the list is refused before it can execute."""
    with pytest.raises((pickle.UnpicklingError, ValueError)):
        _restricted_roundtrip(_GeneratedReducer())


def test_restricted_roundtrip_restores_registry_on_success_and_failure() -> None:
    """The ambient registry is restored exactly after both success and error."""
    previous = torch.serialization.get_safe_globals()
    restored = _restricted_roundtrip({"payload": torch.arange(3)})
    assert restored["payload"].shape == (3,)
    assert torch.serialization.get_safe_globals() == previous
    with pytest.raises((pickle.UnpicklingError, ValueError)):
        _restricted_roundtrip(_GeneratedReducer())
    assert torch.serialization.get_safe_globals() == previous


# ── 1. source cadence → model grid, end to end ────────────────────────

def test_enhanced_lazy_resamples_source_cadence_onto_model_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 4.997 ms source trial becomes a 4.0 ms model-grid sequence."""
    raw = tmp_path / "raw"
    _write_session(raw, "visual_session", n_frames=600, looming=True)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    spec = _spec(metadata_path)
    assert spec["n_frames"] == 600  # source evidence, not model grid

    enhanced = ClockAwareLazyDataset(str(metadata_path), dt_ms=MODEL_DT_MS)
    assert enhanced.uses_model_grid is True
    X, Y, length = enhanced[0]
    assert X.shape == (length, 8) and Y.shape == (length,)
    assert length != 600  # resampled, not the source count
    record = enhanced.model_grid_provenance(
        spec, extract_trial_data(enhanced._load_session(spec),
                                 spec["session_id"], spec["trial_id"]))
    assert record["source_n"] == 600 and record["model_n"] == length
    assert record["dt_ms"] == MODEL_DT_MS
    # Causal hold: frame 0 holds source row 0 and no model frame exceeds the
    # observed source support (no extrapolated sample).
    source = extract_trial_data(enhanced._load_session(spec),
                                spec["session_id"], spec["trial_id"])
    np.testing.assert_allclose(X[0, 0].item(), source["visual_angle"][0])
    assert X[:, 0].numpy().max() <= source["visual_angle"].max() + 1e-9


def test_enhanced_lazy_item_length_equals_model_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    _write_session(raw, "visual_session", n_frames=600, looming=True)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    spec = _spec(metadata_path)
    enhanced = ClockAwareLazyDataset(str(metadata_path), dt_ms=MODEL_DT_MS)
    X, _, length = enhanced[0]
    record = enhanced.model_grid_provenance(
        spec, extract_trial_data(enhanced._load_session(spec),
                                 spec["session_id"], spec["trial_id"]))
    # The model grid is denser than the ~4.997 ms source, so it holds more frames.
    assert length == record["model_n"] > 600
    assert X.shape[0] == length


# ── 2. pure-wind prepend is a model-grid quantity ─────────────────────

def test_pure_wind_prepend_uses_model_grid_not_source_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 5.7 s prepend is 1425 frames at 4 ms, never 1141 at 4.997 ms."""
    raw = tmp_path / "raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    spec = _spec(metadata_path)
    assert spec["is_pure_wind"] is True
    assert spec["n_frames"] == 600 and spec["anchor_frame"] == 50  # source coords
    assert spec["lazy_model_clock"] is True

    source_cadence_prepend = _compute_pure_wind_prepend_frames(SOURCE_DT_MS)
    assert source_cadence_prepend == 1141  # the wrong, source-grid value

    enhanced = ClockAwareLazyDataset(str(metadata_path), dt_ms=MODEL_DT_MS)
    X, Y, length = enhanced[0]
    assert enhanced._prepend_frames(spec) == _compute_pure_wind_prepend_frames(4.0) == 1425
    record = enhanced.model_grid_provenance(
        spec, extract_trial_data(enhanced._load_session(spec),
                                 spec["session_id"], spec["trial_id"]))
    assert record["synthetic_prepend_frames"] == 1425
    assert length == record["model_n"] + 1425
    np.testing.assert_allclose(Y[:1425].numpy(), 0.0)  # synthetic leading zeros
    np.testing.assert_allclose(Y[1425:].numpy(), 0.1)  # observed velocity


# ── 3. converter emits the complete eager contract ────────────────────

def test_converter_emits_complete_eager_contract_and_binds_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.convert_metadata_to_etl import main as convert_main

    raw = tmp_path / "raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    _write_session(raw, "visual_session", n_frames=600, looming=True)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    output_path = tmp_path / "etl.pt"
    convert_main(["--input", str(metadata_path), "--output", str(output_path)])

    saved = load_artifact_bytes(output_path.read_bytes())
    assert saved["model_dt_ms"] == MODEL_DT_MS
    records = saved["model_grid_provenance"]
    assert len(records) == len(saved["X_seqs"]) == 2
    for x, record, anchor in zip(saved["X_seqs"], records, saved["anchor_frames"]):
        assert record["method"] == "causal_previous_source_sample_hold"
        assert record["dt_ms"] == MODEL_DT_MS
        assert len(x) == record["model_n"] + record["synthetic_prepend_frames"]
        assert anchor == resolve_model_anchor_frame(record)
        assert 0 <= anchor < len(x)
    # The shared loader accepts the artifact at the consumer clock and refuses
    # to serve it at a different one.
    load_dataset_with_fingerprint(output_path, expected_dt_ms=MODEL_DT_MS)
    with pytest.raises(ValueError, match="model_dt_ms.*expected_dt_ms"):
        load_dataset_with_fingerprint(output_path, expected_dt_ms=SOURCE_DT_MS)


def test_converter_output_rejects_a_cropped_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.convert_metadata_to_etl import main as convert_main

    raw = tmp_path / "raw"
    _write_session(raw, "visual_session", n_frames=600, looming=True)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    output_path = tmp_path / "etl.pt"
    convert_main(["--input", str(metadata_path), "--output", str(output_path)])

    saved = load_artifact_bytes(output_path.read_bytes())
    saved["X_seqs"][0] = saved["X_seqs"][0][: len(saved["X_seqs"][0]) - 1]
    cropped = tmp_path / "cropped.pt"
    torch.save(saved, cropped)
    with pytest.raises(ValueError, match="unpadded|disagrees"):
        load_dataset_with_fingerprint(cropped, expected_dt_ms=MODEL_DT_MS)


def test_converter_saves_full_model_grid_sequence_not_cropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long model-grid sequence exceeds max_seq_len but must be saved full."""
    from scripts.convert_metadata_to_etl import main as convert_main

    raw = tmp_path / "raw"
    # model_n (~1124) + 1425 synthetic prepend > the 2400 default crop length.
    _write_session(raw, "wind_session", n_frames=900, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    output_path = tmp_path / "etl.pt"
    convert_main(["--input", str(metadata_path), "--output", str(output_path)])

    saved = load_artifact_bytes(output_path.read_bytes())
    record = saved["model_grid_provenance"][0]
    assert record["synthetic_prepend_frames"] == 1425
    assert len(saved["X_seqs"][0]) == record["model_n"] + 1425 > 2400
    assert saved["lengths"][0] == len(saved["X_seqs"][0])
    load_dataset_with_fingerprint(output_path, expected_dt_ms=MODEL_DT_MS)


# ── 4. legacy markerless metadata stays unverified ────────────────────

def test_legacy_markerless_metadata_is_unverified_not_stamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = tmp_path / "raw"
    _write_session(raw, "visual_session", n_frames=600, looming=True)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    metadata = load_artifact_bytes(metadata_path.read_bytes())
    metadata.pop("lazy_model_clock_contract")
    # Genuine legacy metadata predates the per-spec flag too; a leftover flag
    # would contradict the now-absent declaration and must fail closed.
    for spec in metadata["trial_specs"]:
        spec.pop("lazy_model_clock", None)
    stripped = tmp_path / "legacy.pt"
    torch.save(metadata, stripped)

    with caplog.at_level(logging.WARNING):
        dataset = ClockAwareLazyDataset(str(stripped))
    assert dataset.uses_model_grid is False
    assert "unverified" in caplog.text.lower()
    assert dataset.model_grid_provenance(
        _spec(stripped), {"stimulus_onset_ms": 0.0}) is None  # no false stamp
    _, _, length = dataset[0]
    assert length == 600  # source cadence preserved


# ── 5. invalid / contradictory contracts fail closed ──────────────────

@pytest.mark.parametrize("contract", [
    {"dt_ms": 4.0, "resampler": "nsmor.pipeline.resampling.resample_trial_for_model"},
    {"schema_version": "wrong", "dt_ms": 4.0,
     "resampler": "nsmor.pipeline.resampling.resample_trial_for_model"},
    {"schema_version": LAZY_MODEL_CLOCK_SCHEMA, "dt_ms": 4.0, "resampler": "nope"},
    {"schema_version": LAZY_MODEL_CLOCK_SCHEMA, "dt_ms": 0.0,
     "resampler": "nsmor.pipeline.resampling.resample_trial_for_model"},
    {"schema_version": LAZY_MODEL_CLOCK_SCHEMA, "dt_ms": True,
     "resampler": "nsmor.pipeline.resampling.resample_trial_for_model"},
    "not-a-mapping",
])
def test_invalid_lazy_contract_fails_closed(contract: object) -> None:
    with pytest.raises(ValueError):
        validate_lazy_model_clock_contract(contract)


def test_contradictory_consumer_clock_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    _write_session(raw, "visual_session", n_frames=600, looming=True)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    with pytest.raises(ValueError, match="conflicts with expected_dt_ms"):
        ClockAwareLazyDataset(str(metadata_path), dt_ms=10.0)


def test_build_contract_rejects_bad_dt() -> None:
    for bad in (0.0, -1.0, np.nan, np.inf, True):
        with pytest.raises(ValueError):
            build_lazy_model_clock_contract(bad)


# ── 6. no cross-trial borrowing ───────────────────────────────────────

def test_single_trial_item_reads_only_its_own_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    # A is pure-wind (visual silent), B looms (wind silent); distinct velocities.
    _write_session(raw, "animalA_session_1", trial_id=11, n_frames=600,
                   velocity=0.1, wind_from=50)
    _write_session(raw, "animalB_session_1", trial_id=12, n_frames=600,
                   velocity=0.2, looming=True)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    enhanced = ClockAwareLazyDataset(str(metadata_path), dt_ms=MODEL_DT_MS)

    seen = []
    original = enhanced._load_session

    def spy(spec):
        seen.append((spec["session_id"], spec["trial_id"]))
        return original(spec)

    enhanced._load_session = spy
    enhanced._inner._load_session = spy
    for index in range(len(enhanced)):
        spec = _spec(metadata_path, index)
        X, Y, _ = enhanced[index]
        assert seen[-1] == (spec["session_id"], spec["trial_id"])
        is_a = spec["session_id"] == "animalA_session_1"
        np.testing.assert_allclose(Y[-1].item(), 0.1 if is_a else 0.2)
        # A carries wind only; B carries visual only -- no cross-trial channel.
        assert (X[:, 1].numpy() > 0.5).any() == is_a
        assert (X[:, 0].numpy() > 0).any() == (not is_a)
    assert len(seen) == len(enhanced)


# ── 7. train lazy seam ────────────────────────────────────────────────

def test_train_lazy_seam_yields_model_grid_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nsmor.config_parser import ExperimentConfig
    from scripts import train

    raw = tmp_path / "raw"
    _write_session(raw, "animalA_session_1", trial_id=11, n_frames=600,
                   looming=True)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    monkeypatch.setattr(
        train, "grouped_train_val_split",
        lambda session_ids, n_total, **kw: (np.arange(n_total), np.array([], dtype=int)),
    )
    config = ExperimentConfig()
    assert config.model.dt_ms == MODEL_DT_MS
    train_loader, _ = train.build_dataloaders(
        config, dataset_path=str(metadata_path), val_split=0.5,
        use_lazy_loading=True,
    )
    X, Y, length = train_loader.dataset.dataset[0]
    assert length == X.shape[0] == Y.shape[0]
    assert length != 600  # model-grid length, not source count
    assert X.shape[1] == 8
    assert np.isfinite(X.numpy()).all() and np.isfinite(Y.numpy()).all()


def test_enhanced_dataset_pickle_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolved model-clock contract survives pickle and replays identically.

    Covers both the cold payload (before any item read) and the warm payload a
    forked worker pickles after it has served items.
    """
    raw = tmp_path / "raw"
    _write_session(raw, "visual_session", n_frames=600, looming=True)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)

    cold = ClockAwareLazyDataset(str(metadata_path), dt_ms=MODEL_DT_MS)
    cold_restored = _restricted_roundtrip(cold)
    assert cold_restored.uses_model_grid is True
    X_ref, Y_ref, length_ref = cold[0]
    X, Y, length = cold_restored[0]
    assert length == length_ref
    np.testing.assert_allclose(X.numpy(), X_ref.numpy())
    np.testing.assert_allclose(Y.numpy(), Y_ref.numpy())

    warm = ClockAwareLazyDataset(str(metadata_path), dt_ms=MODEL_DT_MS)
    X_warm, Y_warm, length_warm = warm[0]
    assert set(_warm_payload_globals(warm)) >= {
        "pandas.DataFrame", "nsmor.pipeline.io._SourceKinematics",
    }
    warm_restored = _restricted_roundtrip(warm)
    assert warm_restored.uses_model_grid is True
    X2, Y2, length2 = warm_restored[0]
    assert length2 == length_warm
    np.testing.assert_allclose(X2.numpy(), X_warm.numpy())
    np.testing.assert_allclose(Y2.numpy(), Y_warm.numpy())


# ── 8. restricted-loader seam binds the declared lazy clock ───────────

def test_restricted_loader_binds_declared_lazy_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tensor-free lazy artifact must not be consumed at a wrong clock."""
    raw = tmp_path / "raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    load_dataset_with_fingerprint(metadata_path, expected_dt_ms=MODEL_DT_MS)
    with pytest.raises(ValueError, match="conflicts with expected_dt_ms"):
        load_dataset_with_fingerprint(metadata_path, expected_dt_ms=10.0)


def test_restricted_loader_rejects_partial_lazy_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    metadata = load_artifact_bytes(metadata_path.read_bytes())
    metadata["lazy_model_clock_contract"].pop("resampler")  # partial declaration
    partial = tmp_path / "partial.pt"
    torch.save(metadata, partial)
    with pytest.raises(ValueError, match="resampler"):
        load_dataset_with_fingerprint(partial, expected_dt_ms=MODEL_DT_MS)


def test_present_null_lazy_contract_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present-but-null contract is not legacy; it must not fall to source cadence."""
    raw = tmp_path / "raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    metadata = load_artifact_bytes(metadata_path.read_bytes())
    metadata["lazy_model_clock_contract"] = None
    nulled = tmp_path / "null.pt"
    torch.save(metadata, nulled)
    with pytest.raises(ValueError, match="mapping"):
        load_dataset_with_fingerprint(nulled)
    with pytest.raises(ValueError, match="mapping"):
        ClockAwareLazyDataset(str(nulled), dt_ms=MODEL_DT_MS)


@pytest.mark.parametrize("flag", [False, "yes", 1])
def test_contradictory_per_spec_flag_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: object,
) -> None:
    """A per-spec clock flag that contradicts the top contract is not ignored."""
    raw = tmp_path / "raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    metadata = load_artifact_bytes(metadata_path.read_bytes())
    metadata["trial_specs"][0]["lazy_model_clock"] = flag
    mixed = tmp_path / "mixed.pt"
    torch.save(metadata, mixed)
    with pytest.raises(ValueError, match="lazy_model_clock"):
        ClockAwareLazyDataset(str(mixed), dt_ms=MODEL_DT_MS)
    with pytest.raises(ValueError, match="lazy_model_clock"):
        load_dataset_with_fingerprint(mixed, expected_dt_ms=MODEL_DT_MS)


def test_missing_source_pair_refused_at_dataset_and_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trial spanning two source pairs is refused if one pair is dropped."""
    from scripts.convert_metadata_to_etl import main as convert_main

    raw = tmp_path / "raw"
    _write_session(raw, "dup", trial_id=0, n_frames=300, wind_from=50,
                   subdir="dup_a")
    _write_session(raw, "dup", trial_id=0, n_frames=100, declare_start=False,
                   subdir="dup_b", start_ms=300 * SOURCE_DT_MS + 5.0)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    metadata = load_artifact_bytes(metadata_path.read_bytes())
    spec = metadata["trial_specs"][0]
    assert spec["n_frames"] == 400 and len(spec["source_pairs"]) == 2

    # Complete metadata reads fine on the model grid.
    full = ClockAwareLazyDataset(str(metadata_path), dt_ms=MODEL_DT_MS)
    X, _, length = full[0]
    assert length == X.shape[0]

    spec["source_pairs"] = spec["source_pairs"][:1]  # drop a contributing pair
    truncated = tmp_path / "truncated.pt"
    torch.save(metadata, truncated)

    dataset = ClockAwareLazyDataset(str(truncated), dt_ms=MODEL_DT_MS)
    with pytest.raises(ValueError, match="source n_frames"):
        dataset[0]

    output_path = tmp_path / "truncated_etl.pt"
    with pytest.raises(ValueError, match="source n_frames"):
        convert_main(["--input", str(truncated), "--output", str(output_path)])
    assert not output_path.exists()  # malformed input never publishes an artifact


# ── 9. shared seam: declared lazy artifact must carry the per-spec marker ──
#
# A declared lazy artifact is resampled by the top-level contract, so a trial
# spec that omits ``lazy_model_clock`` is not "unverified legacy" — it is a
# declaration gap.  Both ingestion seams (restricted loader and enhanced
# constructor) must refuse it, and the source-frame completeness check must be
# driven by the resolved artifact state, never by the redundant per-spec flag.


def _construct(
    seam: str, path: Path, dt_ms: float = MODEL_DT_MS,
) -> ClockAwareLazyDataset:
    """Build the enhanced dataset via the path- or metadata= constructor seam."""
    if seam == "metadata_path":
        return ClockAwareLazyDataset(str(path), dt_ms=dt_ms)
    return ClockAwareLazyDataset(
        str(path), metadata=load_artifact_bytes(path.read_bytes()), dt_ms=dt_ms,
    )


def _two_pair_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, first_frames: int = 300,
    second_frames: int = 100,
) -> tuple[Path, dict]:
    """A declared lazy artifact whose trial 0 spans two contributing source pairs."""
    raw = tmp_path / "raw"
    _write_session(raw, "dup", trial_id=0, n_frames=first_frames, wind_from=50,
                   subdir="dup_a")
    _write_session(raw, "dup", trial_id=0, n_frames=second_frames,
                   declare_start=False, subdir="dup_b",
                   start_ms=first_frames * SOURCE_DT_MS + 5.0)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    metadata = load_artifact_bytes(metadata_path.read_bytes())
    spec = metadata["trial_specs"][0]
    assert spec["n_frames"] == first_frames + second_frames
    assert len(spec["source_pairs"]) == 2
    return metadata_path, metadata


@pytest.mark.parametrize("seam", ["metadata_path", "metadata"])
def test_declared_lazy_artifact_missing_per_spec_marker_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str,
) -> None:
    """Dropping the per-spec marker from a declared artifact is not legacy."""
    metadata_path, metadata = _two_pair_metadata(tmp_path, monkeypatch)
    metadata["trial_specs"][0].pop("lazy_model_clock")
    stripped = tmp_path / "no_marker.pt"
    torch.save(metadata, stripped)
    with pytest.raises(ValueError, match="lazy_model_clock"):
        _construct(seam, stripped)
    with pytest.raises(ValueError, match="lazy_model_clock"):
        load_dataset_with_fingerprint(stripped, expected_dt_ms=MODEL_DT_MS)


@pytest.mark.parametrize("seam", ["metadata_path", "metadata"])
def test_declared_lazy_artifact_missing_marker_and_source_pair_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str,
) -> None:
    """A missing marker must not bypass the source-frame completeness check."""
    metadata_path, metadata = _two_pair_metadata(tmp_path, monkeypatch)
    metadata["trial_specs"][0].pop("lazy_model_clock")
    metadata["trial_specs"][0]["source_pairs"] = (
        metadata["trial_specs"][0]["source_pairs"][:1]
    )
    dropped = tmp_path / "no_marker_pair.pt"
    torch.save(metadata, dropped)
    with pytest.raises(ValueError):
        dataset = _construct(seam, dropped)
        dataset[0]


@pytest.mark.parametrize("bad", [0, -3, True, 2.5, "400", None])
@pytest.mark.parametrize("seam", ["metadata_path", "metadata"])
def test_declared_lazy_artifact_missing_marker_cannot_hide_invalid_n_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str, bad: object,
) -> None:
    """An invalid n_frames is refused even when the redundant flag is absent."""
    metadata_path, metadata = _two_pair_metadata(tmp_path, monkeypatch)
    metadata["trial_specs"][0].pop("lazy_model_clock")
    metadata["trial_specs"][0]["n_frames"] = bad
    tampered = tmp_path / "bad_nframes.pt"
    torch.save(metadata, tampered)
    with pytest.raises(ValueError):
        dataset = _construct(seam, tampered)
        dataset[0]


def test_declared_lazy_missing_marker_refused_by_converter_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.convert_metadata_to_etl import main as convert_main

    metadata_path, metadata = _two_pair_metadata(tmp_path, monkeypatch)
    metadata["trial_specs"][0].pop("lazy_model_clock")
    stripped = tmp_path / "no_marker.pt"
    torch.save(metadata, stripped)
    output_path = tmp_path / "no_marker_etl.pt"
    with pytest.raises(ValueError, match="lazy_model_clock"):
        convert_main(["--input", str(stripped), "--output", str(output_path)])
    assert not output_path.exists()


# ── 10. shared seam: lazy and eager namespaces cannot be mixed ─────────────

_PARTIAL_EAGER_KEYS = (
    "model_dt_ms", "model_grid_provenance", "X_seqs", "Y_seqs", "lengths",
)


@pytest.mark.parametrize("key", _PARTIAL_EAGER_KEYS)
@pytest.mark.parametrize("seam", ["metadata_path", "metadata"])
def test_declared_lazy_artifact_rejects_partial_eager_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str, key: str,
) -> None:
    """One eager-namespace key beside a lazy contract is a mixed declaration.

    ``anchor_frames`` is deliberately not in this set: it is a shared key that
    genuine tensor-free lazy metadata also carries (the producer writes it), so
    its presence alone cannot discriminate a mixed declaration.
    """
    raw = tmp_path / "raw"
    _write_session(raw, "visual_session", n_frames=600, looming=True)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    metadata = load_artifact_bytes(metadata_path.read_bytes())
    if key == "X_seqs":
        metadata[key] = [np.zeros((600, 8), dtype=np.float32)]
    elif key == "Y_seqs":
        metadata[key] = [np.zeros(600, dtype=np.float32)]
    elif key == "lengths":
        metadata[key] = [600]
    elif key == "model_grid_provenance":
        metadata[key] = [{"dt_ms": MODEL_DT_MS}]
    else:
        metadata[key] = MODEL_DT_MS
    mixed = tmp_path / f"partial_{key}.pt"
    torch.save(metadata, mixed)
    with pytest.raises(ValueError):
        _construct(seam, mixed)
    # The other consumer of the same fingerprint must agree.
    with pytest.raises(ValueError):
        load_dataset_with_fingerprint(mixed, expected_dt_ms=MODEL_DT_MS)


def _complete_eager_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, y_value: float = 9.0,
) -> Path:
    """A converter-published complete eager artifact with a divergent final Y."""
    from scripts.convert_metadata_to_etl import main as convert_main

    raw = tmp_path / "raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    output_path = tmp_path / "eager.pt"
    convert_main(["--input", str(metadata_path), "--output", str(output_path)])
    eager = load_artifact_bytes(output_path.read_bytes())
    eager["Y_seqs"][0][-1] = y_value  # divergent from the CSV's observed 0.1
    path = tmp_path / "eager_divergent.pt"
    torch.save(eager, path)
    return path


@pytest.mark.parametrize("seam", ["metadata_path", "metadata"])
def test_declared_lazy_artifact_rejects_complete_eager_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str,
) -> None:
    """A complete eager contract cannot coexist with a lazy contract.

    Both namespaces describe the same artifact at the same fingerprint; if they
    disagree (here the eager final Y is 9.0 while the source CSV is 0.1) neither
    seam may silently prefer one and serve divergent values.
    """
    eager_path = _complete_eager_artifact(tmp_path, monkeypatch, y_value=9.0)
    eager = load_artifact_bytes(eager_path.read_bytes())
    raw = tmp_path / "raw2"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    lazy_meta = load_artifact_bytes(
        _produce_metadata(tmp_path, monkeypatch, raw).read_bytes(),
    )
    mixed = dict(eager)
    mixed.update(lazy_meta)  # lazy contract + trial_specs + provenance
    assert "lazy_model_clock_contract" in mixed and "model_grid_provenance" in mixed
    # ``dict.update`` must retain the *eager* target and tensors (the reviewer
    # suspected it did not): the mutation is the changed eager final Y, and the
    # source CSV still observes 0.1, so the two representations disagree.
    assert mixed["Y_seqs"][0][-1] == 9.0
    assert "X_seqs" in mixed and "lengths" in mixed and "model_dt_ms" in mixed
    assert 9.0 != 0.1  # CSV-observed velocity from _write_session(velocity=0.1)
    mixed_path = tmp_path / "complete_mix.pt"
    torch.save(mixed, mixed_path)
    # Assert the on-disk bytes still carry the divergent eager target.
    reloaded = load_artifact_bytes(mixed_path.read_bytes())
    assert float(reloaded["Y_seqs"][0][-1]) == 9.0
    with pytest.raises(ValueError):
        _construct(seam, mixed_path)
    # Both consumers must reject the same mixed fingerprint.
    with pytest.raises(ValueError):
        load_dataset_with_fingerprint(mixed_path, expected_dt_ms=MODEL_DT_MS)


def test_declared_lazy_complete_eager_mix_refused_by_converter_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.convert_metadata_to_etl import main as convert_main

    eager_path = _complete_eager_artifact(tmp_path, monkeypatch, y_value=9.0)
    eager = load_artifact_bytes(eager_path.read_bytes())
    raw = tmp_path / "raw2"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    lazy_meta = load_artifact_bytes(
        _produce_metadata(tmp_path, monkeypatch, raw).read_bytes(),
    )
    mixed = dict(eager)
    mixed.update(lazy_meta)
    mixed_path = tmp_path / "complete_mix.pt"
    torch.save(mixed, mixed_path)
    output_path = tmp_path / "mixed_etl.pt"
    with pytest.raises(ValueError):
        convert_main(["--input", str(mixed_path), "--output", str(output_path)])
    assert not output_path.exists()


# ── 10b. markerless lazy hybrids: no silent fall to source cadence ────────
#
# A ``trial_specs`` container is written by every lazy producer and by no eager
# producer.  Beside any eager grid marker it is a hybrid whose lazy contract was
# deleted: the frozen inner reader would silently serve source-cadence frames
# while the restricted loader serves the eager tensors, producing two different
# Y values under one fingerprint.  Every seam must resolve a representation
# verdict on the same object, while supported eager-only artifacts (no
# ``trial_specs``) and genuine markerless legacy (no eager marker) stay loadable.


def _markerless_lazy_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict]:
    """A lazy metadata whose contract and per-spec markers were both deleted."""
    raw = tmp_path / "legacy_raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    metadata = load_artifact_bytes(metadata_path.read_bytes())
    metadata.pop("lazy_model_clock_contract")
    for spec in metadata["trial_specs"]:
        spec.pop("lazy_model_clock", None)
    return metadata_path, metadata


def _eager_and_markerless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, dict, Path]:
    """Eager tensors (divergent final Y), markerless lazy specs, and the CSV."""
    from scripts.convert_metadata_to_etl import main as convert_main

    raw = tmp_path / "raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    eager_path = tmp_path / "eager.pt"
    convert_main(["--input", str(metadata_path), "--output", str(eager_path)])
    eager = load_artifact_bytes(eager_path.read_bytes())
    eager["Y_seqs"][0][-1] = 9.0  # diverges from the source CSV's observed 0.1
    lazy_meta = load_artifact_bytes(metadata_path.read_bytes())
    lazy_meta.pop("lazy_model_clock_contract")
    for spec in lazy_meta["trial_specs"]:
        spec.pop("lazy_model_clock", None)
    kinematics = raw / "wind_session" / "wind_session_kinematics.csv"
    return eager, lazy_meta, kinematics


def _assert_hybrid_payload_intact(mixed: dict, kinematics_csv: Path) -> None:
    """Assert the mixed object really carries both divergent representations.

    Raw lazy specs and the complete eager tensor namespace coexist in one object;
    the eager final Y is 9.0 while the source CSV the lazy specs point at still
    observes 0.1 cm/s.  Called immediately before every ingestion seam so no seam
    can pass by silently dropping one representation from the served object.
    """
    assert "trial_specs" in mixed and mixed["trial_specs"]
    assert "lazy_model_clock_contract" not in mixed
    for key in ("X_seqs", "Y_seqs", "lengths", "model_dt_ms",
                "model_grid_provenance", "anchor_frames"):
        assert key in mixed, f"hybrid must still carry the eager key {key!r}"
    assert len(mixed["X_seqs"]) == len(mixed["Y_seqs"]) == len(mixed["lengths"])
    assert float(mixed["Y_seqs"][0][-1]) == 9.0
    source = pd.read_csv(kinematics_csv)
    assert float(source["velocity"].iloc[-1]) == 0.1


@pytest.mark.parametrize("seam", ["metadata_path", "metadata"])
def test_markerless_complete_eager_hybrid_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str,
) -> None:
    """Raw specs + a complete eager contract cannot serve two Y values."""
    eager, lazy_meta, kinematics = _eager_and_markerless(tmp_path, monkeypatch)
    mixed = dict(eager)
    mixed.update(lazy_meta)  # lazy trial_specs beside eager tensors/provenance
    _assert_hybrid_payload_intact(mixed, kinematics)
    mixed_path = tmp_path / "markerless_hybrid.pt"
    torch.save(mixed, mixed_path)
    # The on-disk bytes still carry both representations, not a rewritten one.
    reloaded = load_artifact_bytes(mixed_path.read_bytes())
    _assert_hybrid_payload_intact(reloaded, kinematics)
    with pytest.raises(ValueError):
        _construct(seam, mixed_path)
    # Both consumers of the same fingerprint must agree.
    with pytest.raises(ValueError):
        load_dataset_with_fingerprint(mixed_path, expected_dt_ms=MODEL_DT_MS)


@pytest.mark.parametrize("seam", ["metadata_path", "metadata"])
def test_markerless_hybrid_with_all_four_markers_absent_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str,
) -> None:
    """The reviewers' exact repro: raw specs + complete tensors, no markers.

    The mixed object carries the raw ``trial_specs`` and the complete eager
    ``X_seqs``/``Y_seqs``/``lengths``/``anchor_frames`` while
    ``model_dt_ms``/``model_grid_provenance``/``labeling_eligibility``/
    ``source_clock_provenance`` are all absent.  Eager Y[-1]=9.0 and the lazy
    CSV the specs point at observes 0.1 under one artifact; every seam must
    refuse before serving (both constructors, restricted loader) or publishing
    (converter).
    """
    from scripts.convert_metadata_to_etl import main as convert_main

    eager, lazy_meta, kinematics = _eager_and_markerless(tmp_path, monkeypatch)
    mixed = dict(eager)
    mixed.update(lazy_meta)
    for marker in ("model_dt_ms", "model_grid_provenance",
                   "labeling_eligibility", "source_clock_provenance"):
        mixed.pop(marker, None)
    # Assert the hybrid payload persists right before each seam.
    assert "trial_specs" in mixed and mixed["trial_specs"]
    assert all(k in mixed for k in ("X_seqs", "Y_seqs", "lengths", "anchor_frames"))
    assert all(m not in mixed for m in (
        "model_dt_ms", "model_grid_provenance",
        "labeling_eligibility", "source_clock_provenance"))
    assert len(mixed["X_seqs"]) == len(mixed["Y_seqs"]) == len(mixed["lengths"])
    assert float(mixed["Y_seqs"][0][-1]) == 9.0
    assert float(pd.read_csv(kinematics)["velocity"].iloc[-1]) == 0.1
    mixed_path = tmp_path / "markerless_four.pt"
    torch.save(mixed, mixed_path)
    reloaded = load_artifact_bytes(mixed_path.read_bytes())
    assert float(reloaded["Y_seqs"][0][-1]) == 9.0
    assert "trial_specs" in reloaded
    with pytest.raises(ValueError):
        _construct(seam, mixed_path)
    with pytest.raises(ValueError):
        load_dataset_with_fingerprint(mixed_path, expected_dt_ms=MODEL_DT_MS)
    output_path = tmp_path / "markerless_four_etl.pt"
    with pytest.raises(ValueError):
        convert_main(["--input", str(mixed_path), "--output", str(output_path)])
    assert not output_path.exists()


# ``labeling_eligibility`` / ``source_clock_provenance`` are source-side, non-grid
# ledgers a tensor-free legacy artifact may legitimately carry; alone they are
# not an eager payload (see the ledger control in 10b-iii).  Only the eager grid
# clock and tensor keys mark a hybrid.
@pytest.mark.parametrize("key", ["model_dt_ms", "model_grid_provenance"])
@pytest.mark.parametrize("seam", ["metadata_path", "metadata"])
def test_markerless_partial_eager_hybrid_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str, key: str,
) -> None:
    """A single eager grid marker beside raw specs is an incomplete hybrid."""
    _, lazy_meta = _markerless_lazy_metadata(tmp_path, monkeypatch)
    if key == "model_grid_provenance":
        lazy_meta[key] = [{"dt_ms": MODEL_DT_MS}]
    else:
        lazy_meta[key] = MODEL_DT_MS
    mixed_path = tmp_path / f"partial_{key}.pt"
    torch.save(lazy_meta, mixed_path)
    with pytest.raises(ValueError):
        _construct(seam, mixed_path)
    with pytest.raises(ValueError):
        load_dataset_with_fingerprint(mixed_path, expected_dt_ms=MODEL_DT_MS)


# A lone eager *tensor* key (no grid clock, no provenance) is equally an eager
# namespace beside raw specs: the frozen lazy reader would serve source-cadence
# frames while the restricted loader reads the tensors, two Y values at one
# fingerprint.  It must fail closed before either seam serves or publishes.
@pytest.mark.parametrize("key", ["X_seqs", "Y_seqs", "lengths"])
@pytest.mark.parametrize("seam", ["metadata_path", "metadata"])
def test_markerless_partial_tensor_namespace_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str, key: str,
) -> None:
    """One eager tensor key beside raw specs is an incomplete eager hybrid."""
    _, lazy_meta = _markerless_lazy_metadata(tmp_path, monkeypatch)
    if key == "X_seqs":
        lazy_meta[key] = [np.zeros((600, 8), dtype=np.float32)]
    elif key == "Y_seqs":
        lazy_meta[key] = [np.zeros(600, dtype=np.float32)]
    else:
        lazy_meta[key] = [600]
    mixed_path = tmp_path / f"partial_tensor_{key}.pt"
    torch.save(lazy_meta, mixed_path)
    with pytest.raises(ValueError):
        _construct(seam, mixed_path)
    with pytest.raises(ValueError):
        load_dataset_with_fingerprint(mixed_path, expected_dt_ms=MODEL_DT_MS)


# ── 10b-iii. non-grid ledgers alone are source evidence, not an eager payload ─
#
# ``labeling_eligibility`` / ``source_clock_provenance`` are source-side ledgers:
# a genuine tensor-free legacy artifact may carry them without any eager
# ``X_seqs``/``Y_seqs``/``lengths`` or ``model_dt_ms``/``model_grid_provenance``
# payload.  They must NOT, alone, be classified as an eager model-grid contract
# (that is the reviewers' BLOCKER-2 over-rejection).  Only a real eager tensor
# namespace -- or the grid clock/provenance -- beside raw specs is a hybrid.


def _markerless_legacy_with_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """A tensor-free legacy artifact carrying only source-side ledgers."""
    _, legacy = _markerless_lazy_metadata(tmp_path, monkeypatch)
    legacy["labeling_eligibility"] = [
        {"session_id": s["session_id"], "trial_id": s["trial_id"],
         "status": "labeled", "clock_provenance": None}
        for s in legacy["trial_specs"]
    ]
    legacy["source_clock_provenance"] = [None] * len(legacy["trial_specs"])
    return legacy


@pytest.mark.parametrize("seam", ["metadata_path", "metadata"])
def test_markerless_legacy_with_non_grid_ledgers_stays_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ledgers without eager tensors/model grid stay loadable and unverified."""
    legacy = _markerless_legacy_with_ledgers(tmp_path, monkeypatch)
    assert "labeling_eligibility" in legacy and "source_clock_provenance" in legacy
    assert "X_seqs" not in legacy and "model_dt_ms" not in legacy
    legacy_path = tmp_path / "ledger_legacy.pt"
    torch.save(legacy, legacy_path)
    with caplog.at_level(logging.WARNING):
        dataset = _construct(seam, legacy_path)
    assert dataset.uses_model_grid is False
    assert "unverified" in caplog.text.lower()
    # The shared restricted loader must agree: unverified legacy, not a hybrid.
    loaded, _ = load_dataset_with_fingerprint(legacy_path, expected_dt_ms=MODEL_DT_MS)
    assert "trial_specs" in loaded


def test_eager_artifact_with_inert_lineage_specs_stays_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eager tensors beside inert per-trial lineage stubs are not a hybrid.

    The subset tool slices the source's ``trial_specs`` beside its eager tensors,
    but those stubs carry no lazy source pointer (``session_dir``/``kinematics_file``
    /``events_file``), so the frozen lazy reader could not resolve them.  Such an
    artifact is a supported eager-only dataset and must stay loadable -- unlike a
    hybrid whose specs really point at the source CSVs.
    """
    from scripts.convert_metadata_to_etl import main as convert_main

    raw = tmp_path / "raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    eager_path = tmp_path / "eager.pt"
    convert_main(["--input", str(metadata_path), "--output", str(eager_path)])
    eager = load_artifact_bytes(eager_path.read_bytes())
    assert "trial_specs" not in eager  # the converter publishes eager-only
    eager["trial_specs"] = [
        {"session_id": s, "trial_id": t, "source_pairs": [f"raw_{i}"]}
        for i, (s, t) in enumerate(zip(eager["session_ids"], eager["trial_ids"]))
    ]
    path = tmp_path / "eager_lineage.pt"
    torch.save(eager, path)
    loaded, _ = load_dataset_with_fingerprint(path, expected_dt_ms=MODEL_DT_MS)
    assert "trial_specs" in loaded and "X_seqs" in loaded


# ── 10b-iv. the classifier follows the reader's actual source resolution ───
#
# The frozen readers resolve a spec's raw CSVs through ``spec['source_pairs']``
# when present (``lazy_dataloader.source_paths`` on each nested dict), falling
# back to the spec itself otherwise.  A spec whose only raw pointers live inside
# ``source_pairs`` is therefore a genuine lazy source description, not an inert
# lineage stub: a markerless hybrid whose top-level pointers were dropped must
# still fail closed.  Only a ``source_pairs`` entry that does *not* resolve to a
# raw source dict is inert lineage.


def _nested_source_hybrid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
    partial: bool = False, container: str = "list",
) -> tuple[dict, Path]:
    """Markerless lazy specs whose raw pointers live only inside source_pairs.

    Mirrors a reader-resolvable spec whose top-level source keys were dropped:
    the nested ``source_pairs`` dicts still carry ``session_dir`` /
    ``kinematics_file`` / ``events_file`` (plus digests), so ``source_paths``
    resolves the CSVs.  ``partial`` drops the nested ``events_file`` (a
    malformed pointer); ``container`` selects the source_pairs container type
    the frozen reader iterates.  The eager tensors diverge from the source CSV.
    """
    eager, lazy_meta, kinematics = _eager_and_markerless(tmp_path, monkeypatch)
    for spec in lazy_meta["trial_specs"]:
        for key in ("session_dir", "kinematics_file", "events_file"):
            spec.pop(key, None)
        pairs = [
            {k: v for k, v in pair.items() if not (partial and k == "events_file")}
            for pair in spec["source_pairs"]
        ]
        if container == "tuple":
            spec["source_pairs"] = tuple(pairs)
        elif container == "ndarray":
            spec["source_pairs"] = np.asarray(pairs, dtype=object)
        else:
            spec["source_pairs"] = pairs
    mixed = dict(eager)
    mixed.update(lazy_meta)
    return mixed, kinematics


def _assert_nested_source_hybrid_intact(mixed: dict, kinematics: Path) -> None:
    """The raw pointers survive only inside source_pairs; eager Y diverges."""
    assert "trial_specs" in mixed and mixed["trial_specs"]
    for spec in mixed["trial_specs"]:
        assert not any(
            k in spec for k in ("session_dir", "kinematics_file", "events_file")
        ), "top-level source keys must be gone"
        assert spec["source_pairs"] is not None and len(spec["source_pairs"]) > 0
    assert float(mixed["Y_seqs"][0][-1]) == 9.0
    assert float(pd.read_csv(kinematics)["velocity"].iloc[-1]) == 0.1


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("seam", ["metadata_path", "metadata"])
def test_markerless_nested_source_hybrid_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str, partial: bool,
) -> None:
    """Nested-only raw pointers are a lazy source description, not inert lineage.

    The spec's top-level source keys are gone but ``source_pairs`` still carries
    resolvable (or, when ``partial``, malformed) raw dicts the frozen reader
    resolves.  Every seam must refuse before serving or publishing.
    """
    from scripts.convert_metadata_to_etl import main as convert_main

    mixed, kinematics = _nested_source_hybrid(tmp_path, monkeypatch, partial=partial)
    _assert_nested_source_hybrid_intact(mixed, kinematics)
    mixed_path = tmp_path / f"nested_source_hybrid_{partial}.pt"
    torch.save(mixed, mixed_path)
    _assert_nested_source_hybrid_intact(
        load_artifact_bytes(mixed_path.read_bytes()), kinematics)
    with pytest.raises(ValueError):
        _construct(seam, mixed_path)
    with pytest.raises(ValueError):
        load_dataset_with_fingerprint(mixed_path, expected_dt_ms=MODEL_DT_MS)
    output_path = tmp_path / f"nested_source_etl_{partial}.pt"
    with pytest.raises(ValueError):
        convert_main(["--input", str(mixed_path), "--output", str(output_path)])
    assert not output_path.exists()


@pytest.mark.parametrize("container", ["list", "tuple", "ndarray"])
def test_markerless_nested_source_hybrid_container_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, container: str,
) -> None:
    """Every source_pairs container the frozen reader iterates is resolved."""
    from nsmor.pipeline.resampling import validate_lazy_artifact_clock

    mixed, kinematics = _nested_source_hybrid(
        tmp_path, monkeypatch, container=container)
    _assert_nested_source_hybrid_intact(mixed, kinematics)
    with pytest.raises(ValueError):
        validate_lazy_artifact_clock(mixed, MODEL_DT_MS)


def test_eager_artifact_with_nested_lineage_stubs_stays_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """source_pairs entries that are not raw dicts stay inert, not lazy.

    The subset tool carries per-trial lineage stubs whose ``source_pairs`` hold
    opaque strings; the frozen reader cannot resolve those to CSVs, so the
    artifact is a supported eager-only dataset and must stay loadable.
    """
    from scripts.convert_metadata_to_etl import main as convert_main

    raw = tmp_path / "raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    eager_path = tmp_path / "eager.pt"
    convert_main(["--input", str(metadata_path), "--output", str(eager_path)])
    eager = load_artifact_bytes(eager_path.read_bytes())
    eager["trial_specs"] = [
        {"session_id": s, "trial_id": t, "source_pairs": [f"raw_{i}", f"raw_{i}b"]}
        for i, (s, t) in enumerate(zip(eager["session_ids"], eager["trial_ids"]))
    ]
    path = tmp_path / "eager_nested_lineage.pt"
    torch.save(eager, path)
    loaded, _ = load_dataset_with_fingerprint(path, expected_dt_ms=MODEL_DT_MS)
    assert "trial_specs" in loaded and "X_seqs" in loaded


def test_markerless_legacy_with_anchor_frames_only_stays_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``anchor_frames`` alone is a shared grid-era key, not an eager payload."""
    _, legacy = _markerless_lazy_metadata(tmp_path, monkeypatch)
    legacy["anchor_frames"] = [int(s["anchor_frame"]) for s in legacy["trial_specs"]]
    assert not any(k in legacy for k in ("X_seqs", "Y_seqs", "lengths",
                                         "model_dt_ms", "model_grid_provenance"))
    legacy_path = tmp_path / "anchor_only_legacy.pt"
    torch.save(legacy, legacy_path)
    loaded, _ = load_dataset_with_fingerprint(legacy_path, expected_dt_ms=MODEL_DT_MS)
    assert "trial_specs" in loaded


def test_markerless_hybrid_refused_by_converter_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.convert_metadata_to_etl import main as convert_main

    eager, lazy_meta, kinematics = _eager_and_markerless(tmp_path, monkeypatch)
    mixed = dict(eager)
    mixed.update(lazy_meta)
    _assert_hybrid_payload_intact(mixed, kinematics)
    mixed_path = tmp_path / "markerless_hybrid.pt"
    torch.save(mixed, mixed_path)
    output_path = tmp_path / "markerless_etl.pt"
    with pytest.raises(ValueError):
        convert_main(["--input", str(mixed_path), "--output", str(output_path)])
    assert not output_path.exists()


# ── 10c. non-list trial-spec containers are validated, not skipped ────────


@pytest.mark.parametrize("seam", ["metadata_path", "metadata"])
def test_ndarray_trial_specs_missing_marker_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seam: str,
) -> None:
    """An object-array spec container must not let a missing marker bypass."""
    metadata_path, metadata = _two_pair_metadata(tmp_path, monkeypatch)
    metadata["trial_specs"][0].pop("lazy_model_clock")
    metadata["trial_specs"] = np.asarray(metadata["trial_specs"], dtype=object)
    array_path = tmp_path / "array_specs.pt"
    torch.save(metadata, array_path)
    with pytest.raises(ValueError, match="lazy_model_clock"):
        _construct(seam, array_path)
    with pytest.raises(ValueError, match="lazy_model_clock"):
        load_dataset_with_fingerprint(array_path, expected_dt_ms=MODEL_DT_MS)


def test_ndarray_trial_specs_hybrid_is_refused_by_converter_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.convert_metadata_to_etl import main as convert_main

    _, lazy_meta = _markerless_lazy_metadata(tmp_path, monkeypatch)
    lazy_meta["model_dt_ms"] = MODEL_DT_MS
    lazy_meta["trial_specs"] = np.asarray(lazy_meta["trial_specs"], dtype=object)
    array_path = tmp_path / "array_hybrid.pt"
    torch.save(lazy_meta, array_path)
    output_path = tmp_path / "array_etl.pt"
    with pytest.raises(ValueError):
        convert_main(["--input", str(array_path), "--output", str(output_path)])
    assert not output_path.exists()


@pytest.mark.parametrize("container", ["not-a-spec-container", {"a": 1}, 42])
def test_unrecognised_trial_specs_container_fails_closed(
    container: object,
) -> None:
    """An unrecognised container is refused, never silently skipped."""
    from nsmor.pipeline.resampling import validate_lazy_artifact_clock

    with pytest.raises(ValueError, match="trial_specs"):
        validate_lazy_artifact_clock({"trial_specs": container})


def test_non_mapping_trial_spec_element_fails_closed() -> None:
    from nsmor.pipeline.resampling import validate_lazy_artifact_clock

    with pytest.raises(ValueError, match="mapping"):
        validate_lazy_artifact_clock({"trial_specs": [{"lazy_model_clock": True}, 7]})


# ── 10d. one snapshot: the constructor validates the object it serves ──────


def test_constructor_reads_metadata_path_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path swapped between two reads cannot validate a different revision."""
    raw = tmp_path / "raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)

    served = load_artifact_bytes(metadata_path.read_bytes())
    replacement = copy.deepcopy(served)
    for spec in replacement["trial_specs"]:
        spec["session_id"] = "replacement_session_1"
    replacement["session_ids"] = [
        "replacement_session_1"
    ] * len(replacement["trial_specs"])

    race_path = tmp_path / "race.pt"
    torch.save(served, race_path)
    first, second = io.BytesIO(), io.BytesIO()
    torch.save(served, first)
    torch.save(replacement, second)

    reads: list[int] = []
    original = Path.read_bytes

    def swapped(self: Path) -> bytes:
        if self == race_path:
            reads.append(1)
            return first.getvalue() if len(reads) == 1 else second.getvalue()
        return original(self)

    with patch.object(Path, "read_bytes", swapped):
        dataset = ClockAwareLazyDataset(str(race_path), dt_ms=MODEL_DT_MS)
    assert len(reads) == 1, "constructor must read the metadata path exactly once"
    # The served dataset is the captured object, not a second revision.
    assert dataset.trial_specs[0]["session_id"] == "wind_session"


def test_constructor_rejects_mixed_capture_without_a_second_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid first capture must be rejected, never swapped for a valid one."""
    raw = tmp_path / "raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)

    invalid = load_artifact_bytes(metadata_path.read_bytes())
    invalid["model_dt_ms"] = MODEL_DT_MS  # raw specs + partial eager marker
    valid = load_artifact_bytes(metadata_path.read_bytes())

    race_path = tmp_path / "race.pt"
    torch.save(invalid, race_path)
    first, second = io.BytesIO(), io.BytesIO()
    torch.save(invalid, first)
    torch.save(valid, second)

    reads: list[int] = []
    original = Path.read_bytes

    def swapped(self: Path) -> bytes:
        if self == race_path:
            reads.append(1)
            return first.getvalue() if len(reads) == 1 else second.getvalue()
        return original(self)

    with patch.object(Path, "read_bytes", swapped):
        with pytest.raises(ValueError):
            ClockAwareLazyDataset(str(race_path), dt_ms=MODEL_DT_MS)
    assert len(reads) == 1, "rejecting must not trigger a second metadata read"


# ── 10e. positive controls: supported eager-only and true legacy ──────────


def test_eager_only_artifact_without_trial_specs_still_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eager loader's own artifacts carry no trial_specs and stay supported."""
    from scripts.convert_metadata_to_etl import main as convert_main

    raw = tmp_path / "raw"
    _write_session(raw, "wind_session", n_frames=600, wind_from=50)
    metadata_path = _produce_metadata(tmp_path, monkeypatch, raw)
    output_path = tmp_path / "etl.pt"
    convert_main(["--input", str(metadata_path), "--output", str(output_path)])
    eager = load_artifact_bytes(output_path.read_bytes())
    assert "trial_specs" not in eager
    dataset, _ = load_dataset_with_fingerprint(output_path, expected_dt_ms=MODEL_DT_MS)
    assert len(dataset["X_seqs"]) == 1


def test_true_markerless_legacy_lazy_stays_loadable_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No lazy contract and no eager marker is genuine legacy, not a hybrid."""
    _, legacy = _markerless_lazy_metadata(tmp_path, monkeypatch)
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    with caplog.at_level(logging.WARNING):
        dataset = ClockAwareLazyDataset(str(legacy_path), dt_ms=MODEL_DT_MS)
    assert dataset.uses_model_grid is False
    assert "unverified" in caplog.text.lower()
    loaded, _ = load_dataset_with_fingerprint(legacy_path, expected_dt_ms=MODEL_DT_MS)
    assert "trial_specs" in loaded  # unverified legacy is served, not rejected
