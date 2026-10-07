"""Clock-stamped lazy/ELT snapshot loading (T7).

``load_csv_snapshot`` and ``NSMoRLazyDataset.captured_sources`` parse a
bounded-memory byte snapshot (a ``SpooledTemporaryFile``) rather than a
path.  When a kinematics CSV carries clock-provenance columns, the loader
must resolve its sibling ``*_timebase_audit.json`` and ``*_events.csv``
from the *origin* path, not from the stream.  These tests bind that
origin explicitly and prove snapshot and direct-path parsing of identical
bytes return identical public data.
"""
from __future__ import annotations

import builtins
import hashlib
import io
import json
import pickle
import sys
import types
from pathlib import Path, PosixPath
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from nsmor.config import FeatureConfig
from nsmor.lazy_dataloader import NSMoRLazyDataset
from nsmor.pipeline.io import ClockAwareLazyDataset, _SourceKinematics
from nsmor.pipeline.nested_prior import load_artifact_bytes
from scripts.pre_load_adapt import adapt_cercus_to_nsmor

N_FRAMES = 300
STIMULUS_FRAME = 200


class _AmbientOnlyUnlisted:
    """Module-level, picklable, and deliberately absent from the test list."""


class _GeneratedReducer:
    """A picklable reducer outside the list; must never execute on load."""

    def __reduce__(self):
        return eval, ("raise AssertionError('reducer executed')",)


def _write_raw_pair(raw: Path) -> Tuple[Path, Path]:
    """Write a cercus pair whose host clock and tick counter diverge."""
    raw.mkdir(parents=True, exist_ok=True)
    host = (10.0 + np.arange(N_FRAMES) * 0.004).tolist()
    ticks = [str(100 + i * 5) for i in range(N_FRAMES)]
    kin = raw / "clock_kinematics.csv"
    evt = raw / "clock_events.csv"
    pd.DataFrame({
        "sys_time": host,
        "ard_time": ticks,
        "dx": [0.0] + [0.1] * (N_FRAMES - 1),
        "dy": 0.0,
        "dz": 0.0,
        # No wind and no looming: a no-stimulus trial, so the lazy reader
        # does not prepend a pure-wind baseline (which would change length).
        "stim_state": 0,
        "global_trial_id": 1,
    }).to_csv(kin, index=False)
    pd.DataFrame({
        "event_name": ["trial_start", "stimulus_onset"],
        "timestamp": [10.0, 10.0 + STIMULUS_FRAME * 0.004],
        "global_trial_id": [1, 1],
        "details": ['{"type": "baseline_wind"}', "{}"],
    }).to_csv(evt, index=False)
    return kin, evt


def _staged_clock_pair(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """Return (raw_root, staged_kin, staged_evt) for a clock-stamped pair."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_raw_pair(raw)
    adapt_cercus_to_nsmor(raw, out, experimental_clock_residual_ms=0.001)
    staged = out / "clock"
    return out, staged / "clock_kinematics.csv", staged / "clock_events.csv"


def _produce_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run prepare_metadata over a clock-stamped pair; return (meta, out)."""
    from scripts import prepare_metadata

    out, _, _ = _staged_clock_pair(tmp_path)
    metadata_path = tmp_path / "metadata.pt"
    monkeypatch.setattr(prepare_metadata, "resolve_group_folds", lambda *a, **k: 2)
    monkeypatch.setattr(
        prepare_metadata, "train_mcmc_cross_fitted",
        lambda *a, **k: (np.full((1, 4), 0.25), [], []),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["prepare_metadata.py", "--raw_dir", str(out), "--output", str(metadata_path)],
    )
    prepare_metadata.main()
    return metadata_path, out


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

    A cold wrapper pickles only its four class globals; a wrapper that already
    read an item embeds the live pandas frames and ``_SourceKinematics`` of a
    warm worker.  Both are covered; any global outside the list fails closed.
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


def test_load_csv_snapshot_parses_clock_pair_from_stream(tmp_path: Path) -> None:
    """A clock CSV snapshot parses when its origin path is supplied."""
    from nsmor.pipeline.io import load_csv_snapshot, load_kinematics_csv

    _, kin, _ = _staged_clock_pair(tmp_path)
    direct = load_kinematics_csv(kin)
    table, digest = load_csv_snapshot(kin, load_kinematics_csv, source_path=kin)

    assert digest == hashlib.sha256(kin.read_bytes()).hexdigest()
    pd.testing.assert_frame_equal(table, direct)
    assert table["clock_audit_reference"].tolist() == (
        direct["clock_audit_reference"].tolist()
    )
    assert table["clock_audit_reference"].nunique() == 1


def test_load_csv_snapshot_requires_source_path_for_clock_columns(
    tmp_path: Path,
) -> None:
    """A clock-stamped stream without an origin must fail closed, clearly."""
    from nsmor.pipeline.io import load_csv_snapshot, load_kinematics_csv

    _, kin, _ = _staged_clock_pair(tmp_path)
    with pytest.raises(ValueError, match="source_path"):
        load_csv_snapshot(kin, load_kinematics_csv)


def test_lazy_dataset_reads_clock_pair_cold_and_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NSMoRLazyDataset replays a clock pair both cold and via captured_sources."""
    from nsmor.pipeline.io import ClockAwareLazyDataset

    metadata_path, _ = _produce_metadata(tmp_path, monkeypatch)
    cold = ClockAwareLazyDataset(str(metadata_path))
    X_cold, Y_cold, length_cold = cold[0]
    assert length_cold == N_FRAMES
    assert X_cold.shape == (N_FRAMES, 8)
    assert Y_cold.shape == (N_FRAMES,)

    captured = ClockAwareLazyDataset(str(metadata_path))
    with captured.captured_sources() as revision:
        assert revision and revision[0]["kinematics_sha256"]
        X_cap, Y_cap, length_cap = captured[0]
    assert length_cap == length_cold
    np.testing.assert_allclose(X_cap.numpy(), X_cold.numpy())
    np.testing.assert_allclose(Y_cap.numpy(), Y_cold.numpy())


def test_captured_sources_fails_closed_on_post_capture_audit_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A captured CSV snapshot cannot be re-paired with a mutated audit.

    The kinematics/events bytes are frozen at ``captured_sources`` entry, but
    the sibling ``*_timebase_audit.json`` is a separate live file.  If it is
    rewritten after capture (here keeping ``status`` accepted, so only the
    binding check can catch it), parsing the captured snapshot must fail closed
    rather than pair old CSV bytes with new provenance.
    """
    from nsmor.pipeline.io import ClockAwareLazyDataset

    metadata_path, out = _produce_metadata(tmp_path, monkeypatch)
    audit_path = out / "clock" / "clock_timebase_audit.json"

    dataset = ClockAwareLazyDataset(str(metadata_path))
    with dataset.captured_sources():
        mutated = json.loads(audit_path.read_text())
        mutated["method"] = "MUTATED_AFTER_CAPTURE"
        mutated["scientific_acceptance"] = "tampered"
        audit_path.write_text(json.dumps(mutated, indent=2))
        with pytest.raises(ValueError, match="changed after snapshot capture"):
            dataset[0]


def test_captured_sources_rejects_audit_absent_at_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An audit absent at capture must not be attachable later.

    If the sibling audit is removed *before* ``captured_sources`` entry, the
    capture must record that absence explicitly. A clock-stamped CSV snapshot
    must then be rejected even when a *new* audit file with unchanged CSV
    hashes and accepted status appears inside the capture window: absence at
    capture is itself provenance, and must never degrade to ordinary live
    parsing that attaches post-capture provenance.
    """
    from nsmor.pipeline.io import ClockAwareLazyDataset

    metadata_path, out = _produce_metadata(tmp_path, monkeypatch)
    audit_path = out / "clock" / "clock_timebase_audit.json"
    original = json.loads(audit_path.read_text())

    dataset = ClockAwareLazyDataset(str(metadata_path))
    audit_path.unlink()  # Absent at capture time.
    with dataset.captured_sources():
        invented = dict(original)
        invented["method"] = "INVENTED_AFTER_CAPTURE"
        invented["scientific_acceptance"] = "fabricated"
        audit_path.write_text(json.dumps(invented, indent=2))
        with pytest.raises(ValueError, match="absent at snapshot capture"):
            dataset[0]


def test_captured_sources_repeated_reads_use_bound_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated captured reads reuse the bound revision and recover cold after.

    The first read parses from the frozen snapshot; the second hits the session
    cache and must return byte-identical data without re-reading the live audit.
    Deleting the audit *after* the first read must not affect either captured
    read, and a subsequent cold read (outside the context) must succeed again
    from the live files.
    """
    from nsmor.pipeline.io import ClockAwareLazyDataset

    metadata_path, out = _produce_metadata(tmp_path, monkeypatch)
    audit_path = out / "clock" / "clock_timebase_audit.json"

    dataset = ClockAwareLazyDataset(str(metadata_path))
    with dataset.captured_sources():
        X1, Y1, length1 = dataset[0]
        audit_path.unlink()  # Live audit gone; bound revision must still serve.
        X2, Y2, length2 = dataset[0]
    assert length1 == length2 == N_FRAMES
    np.testing.assert_allclose(X1.numpy(), X2.numpy())
    np.testing.assert_allclose(Y1.numpy(), Y2.numpy())

    # Outside the capture the loader reads live sources again; the deleted
    # audit means a clock-stamped cold read now fails closed.
    cold = ClockAwareLazyDataset(str(metadata_path))
    with pytest.raises((ValueError, FileNotFoundError)):
        cold[0]


def test_clock_aware_dataset_pickle_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper survives a pickle round-trip and still replays a clock pair.

    Covers the cold payload (no item read yet) and the warm payload a worker
    pickles after it has served items — the live pandas frames and
    ``_SourceKinematics`` a forked loader carries.
    """
    metadata_path, _ = _produce_metadata(tmp_path, monkeypatch)

    cold = ClockAwareLazyDataset(str(metadata_path))
    assert set(_warm_payload_globals(cold)) == {
        "builtins.getattr", "nsmor.config.FeatureConfig",
        "nsmor.lazy_dataloader.NSMoRLazyDataset",
        "nsmor.pipeline.io.ClockAwareLazyDataset",
    }
    cold_restored = _restricted_roundtrip(cold)
    X_ref, Y_ref, length_ref = cold[0]
    X, Y, length = cold_restored[0]
    assert len(cold_restored) == len(cold) and length == length_ref
    np.testing.assert_allclose(X.numpy(), X_ref.numpy())
    np.testing.assert_allclose(Y.numpy(), Y_ref.numpy())

    warm = ClockAwareLazyDataset(str(metadata_path))
    X_warm, Y_warm, length_warm = warm[0]
    assert set(_warm_payload_globals(warm)) >= {
        "pandas.DataFrame", "nsmor.pipeline.io._SourceKinematics",
    }
    warm_restored = _restricted_roundtrip(warm)
    X2, Y2, length2 = warm_restored[0]
    assert length2 == length_warm
    np.testing.assert_allclose(X2.numpy(), X_warm.numpy())
    np.testing.assert_allclose(Y2.numpy(), Y_warm.numpy())


def test_prepare_metadata_accepts_clock_stamped_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prepare_metadata binds the clock pair and the lazy reader replays it."""
    from nsmor.pipeline.io import (
        ClockAwareLazyDataset,
        extract_trial_data,
        load_and_concat_sessions,
    )

    metadata_path, out = _produce_metadata(tmp_path, monkeypatch)
    metadata = load_artifact_bytes(metadata_path.read_bytes())
    spec = metadata["trial_specs"][0]
    pair = spec["source_pairs"][0]
    assert pair["kinematics_sha256"] == hashlib.sha256(
        (out / "clock" / "clock_kinematics.csv").read_bytes()
    ).hexdigest()

    # The cold lazy read reconstructs the same clock provenance the direct
    # path exposes for the identical bytes.
    staged_kin = out / "clock" / "clock_kinematics.csv"
    staged_evt = out / "clock" / "clock_events.csv"
    direct = extract_trial_data(
        load_and_concat_sessions([staged_kin], [staged_evt]), "clock", 1,
    )
    dataset = ClockAwareLazyDataset(str(metadata_path))
    X, Y, length = dataset[0]
    assert length == N_FRAMES and X.shape == (N_FRAMES, 8)
    np.testing.assert_allclose(
        Y.numpy(), direct["velocity"], atol=1e-6,
    )
    reference = direct["clock_provenance"][0]
    assert reference["coordinate_scope"] == "source_CSV_rows_not_resampled_tensor_frames"
    assert reference["time_source"] == ["experimental_affine_estimate"] * N_FRAMES


def test_direct_path_loaders_unchanged_without_source_path(tmp_path: Path) -> None:
    """Non-clock and event loaders keep their existing positional contract."""
    from nsmor.pipeline.io import (
        KINEMATICS_COLUMNS,
        load_events_csv,
        load_kinematics_csv,
    )

    _, _, evt = _staged_clock_pair(tmp_path)
    events = load_events_csv(evt)
    assert list(events.columns) == [
        "session_id", "trial_id", "time_ms", "event_type", "event_value",
    ]

    plain = tmp_path / "plain_kinematics.csv"
    pd.DataFrame({
        col: (np.arange(4.0) if col == "time_ms" else np.zeros(4))
        for col in KINEMATICS_COLUMNS
    }).to_csv(plain, index=False)
    frame = load_kinematics_csv(plain)
    assert "clock_audit_reference" not in frame
    assert list(frame.columns) == KINEMATICS_COLUMNS


def test_convert_metadata_to_etl_replays_clock_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production converter replays a clock pair and binds its provenance."""
    from scripts import convert_metadata_to_etl as converter
    from nsmor.pipeline.io import extract_trial_data, load_and_concat_sessions

    metadata_path, out = _produce_metadata(tmp_path, monkeypatch)
    output_path = tmp_path / "etl.pt"
    converter.main(["--input", str(metadata_path), "--output", str(output_path)])

    saved = load_artifact_bytes(output_path.read_bytes())
    assert saved["lengths"] == [N_FRAMES]
    direct = extract_trial_data(
        load_and_concat_sessions(
            [out / "clock" / "clock_kinematics.csv"],
            [out / "clock" / "clock_events.csv"],
        ),
        "clock",
        1,
    )
    np.testing.assert_allclose(saved["Y_seqs"][0], direct["velocity"], atol=1e-6)


def test_prepare_metadata_rejects_tampered_clock_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kinematics byte change is refused by the bound audit hash."""
    from scripts import prepare_metadata

    out, kin, _ = _staged_clock_pair(tmp_path)
    kin.write_bytes(kin.read_bytes() + b"\n")  # Same schema, different bytes.
    metadata_path = tmp_path / "metadata.pt"
    monkeypatch.setattr(prepare_metadata, "resolve_group_folds", lambda *a, **k: 2)
    monkeypatch.setattr(
        prepare_metadata, "train_mcmc_cross_fitted",
        lambda *a, **k: (np.full((1, 4), 0.25), [], []),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["prepare_metadata.py", "--raw_dir", str(out), "--output", str(metadata_path)],
    )
    with pytest.raises(ValueError, match="Clock provenance hash mismatch"):
        prepare_metadata.main()
    assert not metadata_path.exists()


def test_train_lazy_loading_reads_clock_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train --lazy_loading builds a loader that replays the clock pair."""
    from nsmor.config_parser import ExperimentConfig
    from scripts import train

    metadata_path, _ = _produce_metadata(tmp_path, monkeypatch)
    # One session cannot yield a grouped train/val split; the clock replay,
    # not the split arithmetic, is what this test binds.
    monkeypatch.setattr(
        train, "grouped_train_val_split",
        lambda session_ids, n_total, **kw: (np.arange(n_total), np.array([], dtype=int)),
    )
    config = ExperimentConfig()
    train_loader, _ = train.build_dataloaders(
        config, dataset_path=str(metadata_path), val_split=0.5, use_lazy_loading=True,
    )
    assert train_loader is not None
    X, Y, length = train_loader.dataset.dataset[0]
    assert length == N_FRAMES and X.shape == (N_FRAMES, 8)
    assert Y.shape == (N_FRAMES,)
