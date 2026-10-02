"""Actual restricted-loader compatibility for compact clock provenance."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import zlib

import numpy as np
import pytest
import torch

from nsmor.pipeline.clock_storage import (
    pack_clock_provenance,
    restore_clock_provenance,
    unpack_clock_provenance,
)
from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("tensor_storage", [False, True])
def test_clock_storage_round_trip(
    tmp_path: Path, compact: bool, tensor_storage: bool,
) -> None:
    records = [{
        "source_row_indices": [0, 2, 5],
        "raw_sys_time": ["", "NaN", "00001.2300"],
        "raw_ard_time": ["001", "原始", "-0"],
        "time_source": ["host", "estimated", "host"],
        "reference": {"path": "source.csv", "observed": True},
    }]
    stored = pack_clock_provenance(records) if compact else records
    xs = [np.arange(n * 8, dtype=np.float32).reshape(n, 8) for n in (1, 3, 7)]
    ys = [np.arange(n, dtype=np.float32) for n in (1, 3, 7)]
    dataset = {
        "X_seqs": [torch.from_numpy(x) for x in xs] if tensor_storage else xs,
        "Y_seqs": [torch.from_numpy(y) for y in ys] if tensor_storage else ys,
        "lengths": [1, 3, 7], "labels": [0, 1, 2], "anchor_frames": [0, 1, 2],
        "model_dt_ms": 4.,
        "model_grid_provenance": [{
            "origin_ms": 0., "dt_ms": 4., "source_n": n, "model_n": n,
            "source_end_ms": (n - 1) * 4., "model_end_ms": (n - 1) * 4.,
            "synthetic_prepend_frames": 0, "source_anchor_ms": anchor * 4.,
            "source_anchor_rule": "stimulus_onset",
        } for n, anchor in zip((1, 3, 7), (0, 1, 2))],
        "source_clock_provenance": [stored, None, stored],
        "labeling_eligibility": [{"clock_provenance": stored}],
    }
    path = tmp_path / "dataset.pt"
    torch.save(dataset, path)
    loaded, digest = load_dataset_with_fingerprint(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    for key, expected in (("X_seqs", xs), ("Y_seqs", ys)):
        for actual, original in zip(loaded[key], expected):
            assert isinstance(actual, np.ndarray) and actual.dtype == np.float32
            np.testing.assert_array_equal(actual, original)
    actual = loaded["source_clock_provenance"][0]
    assert actual == records
    assert actual is loaded["source_clock_provenance"][2]
    assert actual is loaded["labeling_eligibility"][0]["clock_provenance"]
    assert loaded["source_clock_provenance"][1] is None


@pytest.mark.parametrize("value", [
    {}, {"encoding": "unknown", "payload": b""},
    {"encoding": "clock-provenance-zlib-json-v1", "payload": b"broken"},
    {"encoding": "clock-provenance-zlib-json-v1", "payload": zlib.compress(b"[]")[:-1]},
    {"encoding": "clock-provenance-zlib-json-v1", "payload": zlib.compress(b"[]") + b"tail"},
    {"encoding": "clock-provenance-zlib-json-v1", "payload": zlib.compress(b"{}")},
    {"encoding": "clock-provenance-zlib-json-v1", "payload": zlib.compress(b"[{}]")},
])
def test_invalid_clock_storage_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        unpack_clock_provenance(value)


def test_clock_columns_must_align() -> None:
    value = pack_clock_provenance([{
        "source_row_indices": [1], "raw_sys_time": [],
        "raw_ard_time": ["1"], "time_source": ["host"],
    }])
    with pytest.raises(ValueError, match="align"):
        unpack_clock_provenance(value)


def test_compact_validation_preserves_original_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nsmor.pipeline import clock_storage
    import weakref

    records = [{
        "source_row_indices": [0, 2], "raw_sys_time": ["", "0001.2300"],
        "raw_ard_time": ["原始", "-0"], "time_source": ["host", "estimated"],
        "reference": {"path": "source.csv", "observed": True},
    }]
    first = pack_clock_provenance(records)
    second = pack_clock_provenance(records)
    roots = [first, None, first, second, records]
    eligibility = [{"clock_provenance": first, "eligible": True},
                   {"clock_provenance": second}, {"clock_provenance": records}]
    dataset = {"source_clock_provenance": roots, "labeling_eligibility": eligibility}
    calls: list[int] = []
    released: list[weakref.ReferenceType] = []
    unpack = clock_storage.unpack_clock_provenance

    class DecodedRecords(list):
        pass

    def checked_unpack(value: object) -> object:
        assert all(ref() is None for ref in released)
        decoded = unpack(value)
        if isinstance(value, dict):
            calls.append(id(value))
            decoded = DecodedRecords(decoded)
            released.append(weakref.ref(decoded))
        return decoded

    monkeypatch.setattr(clock_storage, "unpack_clock_provenance", checked_unpack)
    restore_clock_provenance(dataset, materialize=False)
    assert calls == [id(first), id(second)]  # identity, not equal payload bytes
    assert all(ref() is None for ref in released)
    assert dataset["source_clock_provenance"] is roots
    assert dataset["labeling_eligibility"] is eligibility
    assert roots[0] is roots[2] is eligibility[0]["clock_provenance"] is first
    assert roots[3] is eligibility[1]["clock_provenance"] is second
    assert roots[4] is eligibility[2]["clock_provenance"] is records
    assert roots[1] is None and eligibility[0]["eligible"] is True
    assert unpack_clock_provenance(first) == records


def test_loader_compact_opt_in_keeps_snapshot_hash_and_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nsmor.pipeline import nested_prior

    records = [{
        "source_row_indices": [0], "raw_sys_time": ["0001.2300"],
        "raw_ard_time": ["原始"], "time_source": ["host"],
        "reference": {"path": "source.csv"},
    }]
    stored = pack_clock_provenance(records)
    path = tmp_path / "dataset.pt"
    torch.save({
        "X_seqs": [torch.zeros((1, 8))], "Y_seqs": [torch.zeros(1)],
        "lengths": [1], "anchor_frames": [0], "model_dt_ms": 4.,
        "model_grid_provenance": [{
            "origin_ms": 0., "dt_ms": 4., "source_n": 1, "model_n": 1,
            "source_end_ms": 0., "model_end_ms": 0.,
            "synthetic_prepend_frames": 0, "source_anchor_ms": 0.,
            "source_anchor_rule": "stimulus_onset",
        }],
        "source_clock_provenance": [stored, stored],
        "labeling_eligibility": [{"clock_provenance": stored, "eligible": True}],
    }, path)
    captured = path.read_bytes()
    originals: list[dict] = []
    reads: list[Path] = []
    loads: list[bytes] = []
    read_bytes = Path.read_bytes
    load_bytes = nested_prior.load_artifact_bytes

    def read_once(source: Path) -> bytes:
        reads.append(source)
        return read_bytes(source)

    def capture_loaded(payload: bytes, map_location: object = None) -> dict:
        loads.append(payload)
        dataset = load_bytes(payload, map_location=map_location)
        originals.append(dataset["source_clock_provenance"][0])
        return dataset

    monkeypatch.setattr(Path, "read_bytes", read_once)
    monkeypatch.setattr(nested_prior, "load_artifact_bytes", capture_loaded)
    compact, compact_hash = load_dataset_with_fingerprint(
        path, map_location="cpu", restore_provenance=False,
    )
    default, default_hash = load_dataset_with_fingerprint(path, map_location="cpu")
    explicit, explicit_hash = load_dataset_with_fingerprint(
        path, map_location="cpu", restore_provenance=True,
    )
    assert compact_hash == default_hash == explicit_hash
    assert compact_hash == hashlib.sha256(captured).hexdigest()
    assert reads == [path] * 3 and loads == [captured] * 3
    assert read_bytes(path) == captured
    actual = compact["source_clock_provenance"][0]
    assert actual is originals[0] and actual == stored
    assert actual is compact["source_clock_provenance"][1]
    assert actual is compact["labeling_eligibility"][0]["clock_provenance"]
    assert compact["labeling_eligibility"][0]["eligible"] is True
    for restored in (default, explicit):
        actual = restored["source_clock_provenance"][0]
        assert actual == records
        assert actual is restored["source_clock_provenance"][1]
        assert actual is restored["labeling_eligibility"][0]["clock_provenance"]
    np.testing.assert_array_equal(compact["X_seqs"][0], default["X_seqs"][0])


@pytest.mark.parametrize("materialize", [False, True])
@pytest.mark.parametrize("root", ["source", "eligibility"])
@pytest.mark.parametrize("payload", [
    b"broken", zlib.compress(b"[]")[:-1], zlib.compress(b"[]") + b"tail",
    zlib.compress(b"\xff"), zlib.compress(b"["), zlib.compress(b"{}"),
    zlib.compress(b"[1]"), zlib.compress(b"[{}]"),
])
def test_both_modes_reject_invalid_payload_in_either_root(
    materialize: bool, root: str, payload: bytes,
) -> None:
    invalid = {"encoding": "clock-provenance-zlib-json-v1", "payload": payload}
    dataset = {"source_clock_provenance": [pack_clock_provenance([])],
               "labeling_eligibility": []}
    if root == "source":
        dataset["source_clock_provenance"].append(invalid)
    else:
        dataset["labeling_eligibility"].append({"clock_provenance": invalid})
    with pytest.raises(ValueError) as expected:
        unpack_clock_provenance(invalid)
    with pytest.raises(ValueError) as actual:
        restore_clock_provenance(dataset, materialize=materialize)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("materialize", [False, True])
@pytest.mark.parametrize("field, value", [
    ("source_row_indices", []), ("source_row_indices", [True]),
    ("source_row_indices", [-1]), ("source_row_indices", [1.0]),
    ("raw_sys_time", [1.23]), ("raw_ard_time", [None]), ("time_source", [True]),
    ("raw_sys_time", "token"),
])
def test_both_modes_reject_misaligned_or_coerced_raw_columns(
    materialize: bool, field: str, value: object,
) -> None:
    record = {"source_row_indices": [0], "raw_sys_time": ["0001.2300"],
              "raw_ard_time": ["-0"], "time_source": ["host"]}
    record[field] = value
    invalid = pack_clock_provenance([record])
    with pytest.raises(ValueError) as expected:
        unpack_clock_provenance(invalid)
    with pytest.raises(ValueError) as actual:
        restore_clock_provenance(
            {"labeling_eligibility": [{"clock_provenance": invalid}]},
            materialize=materialize,
        )
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("consumer", ["eager", "lazy", "stats", "jax"])
def test_training_consumers_request_compact_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consumer: str,
) -> None:
    from nsmor.config_parser import ExperimentConfig
    from scripts import train

    source = tmp_path / "tiny.pt"
    source.write_bytes(b"not loaded")
    config = ExperimentConfig()
    config.training.normalize_targets = True

    class LoadedCompact(Exception):
        pass

    def compact_load(
        path: Path, *, restore_provenance: bool = True, **kwargs: object,
    ) -> None:
        assert restore_provenance is False
        raise LoadedCompact

    if consumer == "jax":
        pytest.importorskip("jax", exc_type=ImportError)
        pytest.importorskip("flax", exc_type=ImportError)
        pytest.importorskip("optax", exc_type=ImportError)
        from nsmor.jax import dataloader
        monkeypatch.setattr(dataloader, "load_dataset_with_fingerprint", compact_load)
        with pytest.raises(LoadedCompact):
            dataloader.load_nsmor_dataset(source)
    else:
        monkeypatch.setattr(train, "load_dataset_with_fingerprint", compact_load)
        with pytest.raises(LoadedCompact):
            if consumer == "stats":
                train.compute_target_stats(str(source), config)
            else:
                train.build_dataloaders(
                    config, str(source), use_lazy_loading=consumer == "lazy",
                )


def test_nested_prior_generator_requests_compact_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prior generation validates clocks without retaining decoded raw rows."""
    from scripts import evaluate_nested_prior

    source = tmp_path / "tiny.pt"

    class LoadedCompact(Exception):
        pass

    def compact_load(
        path: Path, *, restore_provenance: bool = True, **kwargs: object,
    ) -> None:
        assert path == source
        assert restore_provenance is False
        raise LoadedCompact

    monkeypatch.setattr(
        evaluate_nested_prior, "load_dataset_with_fingerprint", compact_load,
    )
    with pytest.raises(LoadedCompact):
        evaluate_nested_prior.generate_nested_priors(
            dataset=None, source_dataset_path=source, verbose=False,
        )


def test_shared_loader_consumers_keep_clock_provenance_compact() -> None:
    """Only subset production needs restored raw-clock records for slicing."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted(root.glob("scripts/*.py")) + sorted(root.glob("nsmor/jax/*.py"))
    consumers: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Name)
                and node.func.id == "load_dataset_with_fingerprint"
            ):
                continue
            relative = path.relative_to(root).as_posix()
            consumers.add(relative)
            restore = next(
                (kw.value for kw in node.keywords if kw.arg == "restore_provenance"),
                None,
            )
            if relative == "scripts/make_subset_dataset.py":
                # Default restoration is intentional: subset metadata is sliced.
                assert restore is None or (
                    isinstance(restore, ast.Constant) and restore.value is True
                ), f"{relative}:{node.lineno} must restore subset provenance"
            else:
                assert isinstance(restore, ast.Constant) and restore.value is False, (
                    f"{relative}:{node.lineno} must request restore_provenance=False"
                )
    assert "scripts/evaluate_nested_prior.py" in consumers
    assert "scripts/make_subset_dataset.py" in consumers
    assert "nsmor/jax/dataloader.py" in consumers


@pytest.mark.parametrize("restore_provenance", [False, True])
def test_loader_rejects_bad_clock_before_grid_checks(
    tmp_path: Path, restore_provenance: bool,
) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"labeling_eligibility": [{"clock_provenance": {
        "encoding": "clock-provenance-zlib-json-v1", "payload": b"broken",
    }}]}, path)
    with pytest.raises(ValueError, match="Invalid clock provenance payload"):
        load_dataset_with_fingerprint(path, restore_provenance=restore_provenance)


@pytest.mark.parametrize("restore_provenance", [False, True])
def test_loader_opt_in_does_not_allow_pickle_reducers(
    tmp_path: Path, restore_provenance: bool,
) -> None:
    marker = tmp_path / "executed.txt"

    class Reducer:
        def __reduce__(self) -> tuple:
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",)

    path = tmp_path / "bad.pt"
    torch.save({"metadata": Reducer()}, path)
    with pytest.raises(ValueError, match="Unexpected serialized global"):
        load_dataset_with_fingerprint(path, restore_provenance=restore_provenance)
    assert not marker.exists()
