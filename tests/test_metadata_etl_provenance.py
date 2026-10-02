"""
Tests for metadata and ETL dataset provenance validation.

Verifies:
1. scripts/convert_metadata_to_etl.py enriches output dict with:
   - "mcmc_prior_provenance"
   - "anchor_frames"
   - "stimulus_conditions"
   - "is_pure_wind"
2. Converted output passes validate_dataset_provenance() provenance gate.
3. Provenance gate strictly rejects missing or session-grouped provenance.
4. Fallback extraction from trial_specs when top-level fields are omitted.
5. End-to-end CLI integration with synthetic metadata and sessions.
"""
from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from nsmor.config import (
    DEFAULT_FEATURE,
    Label,
    PIPELINE_SEMANTICS_VERSION,
)
from nsmor.model_utils import validate_dataset_provenance
from scripts.convert_metadata_to_etl import (
    main as convert_main,
    populate_etl_provenance_and_conditions,
)


# =========================================================================
# 1. Enrichment Function Tests
# =========================================================================

def test_populate_etl_provenance_and_conditions_direct():
    """populate_etl_provenance_and_conditions must copy all 4 required fields."""
    metadata = {
        "mcmc_prior_provenance": "oof_5fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
        "anchor_frames": [1200, 1205],
        "stimulus_conditions": ["multisensory", "wind_only"],
        "is_pure_wind": np.array([False, True], dtype=bool),
    }

    output: Dict[str, Any] = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "X_seqs": [np.zeros((10, 8)), np.zeros((10, 8))],
        "session_ids": ["recordingA_session_1", "recordingB_session_1"],
        "Y_seqs": [np.zeros(10), np.zeros(10)],
    }

    populate_etl_provenance_and_conditions(output, metadata)

    assert output["mcmc_prior_provenance"] == "oof_5fold_recording_prefix_grouped_cv"
    assert output["animal_identity_status"] == "unverified"
    assert output["anchor_frames"] == [1200, 1205]
    assert output["stimulus_conditions"] == ["multisensory", "wind_only"]
    np.testing.assert_array_equal(output["is_pure_wind"], np.array([False, True], dtype=bool))

    # Must pass provenance gate
    validate_dataset_provenance(output, Path("mock_etl.pt"))


def test_populate_etl_provenance_extracts_from_trial_specs():
    """Fields should be safely extracted from trial_specs if missing at top-level."""
    trial_specs = [
        {
            "session_id": "s1",
            "trial_id": 0,
            "anchor_frame": 1200,
            "stimulus_condition": "multisensory",
            "is_pure_wind": False,
        },
        {
            "session_id": "s1",
            "trial_id": 1,
            "anchor_frame": 1250,
            "stimulus_condition": "wind_only",
            "is_pure_wind": True,
        },
    ]

    metadata = {
        "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
        "trial_specs": trial_specs,
    }

    output: Dict[str, Any] = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
    }

    populate_etl_provenance_and_conditions(output, metadata)

    assert output["mcmc_prior_provenance"] == "oof_5fold_animal_grouped_cv"
    assert output["animal_identity_status"] == "historical_unknown"
    assert output["anchor_frames"] == [1200, 1250]
    assert output["stimulus_conditions"] == ["multisensory", "wind_only"]
    np.testing.assert_array_equal(output["is_pure_wind"], np.array([False, True], dtype=bool))

    validate_dataset_provenance(output, Path("mock_etl.pt"))


def test_provenance_gate_rejects_missing_provenance():
    """Missing mcmc_prior_provenance must be rejected by validate_dataset_provenance."""
    output = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        # Missing mcmc_prior_provenance
    }
    with pytest.raises(RuntimeError, match="mcmc_prior_provenance"):
        validate_dataset_provenance(output, Path("dummy.pt"))


def test_provenance_gate_rejects_session_grouped():
    """Session-grouped provenance must be rejected by validate_dataset_provenance."""
    output = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "session_grouped_5fold",
    }
    with pytest.raises(RuntimeError, match="mcmc_prior_provenance"):
        validate_dataset_provenance(output, Path("dummy.pt"))


# =========================================================================
# 2. prepare_metadata.py Code Audit
# =========================================================================

def test_prepare_metadata_outputs_provenance_and_dt_default():
    """prepare_metadata.py must output prefix provenance and have dt_ms default 4.0."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_metadata.py"
    content = script_path.read_text(encoding="utf-8")

    # Verify mcmc_prior_provenance in metadata dict
    assert '"mcmc_prior_provenance":' in content, (
        "prepare_metadata.py must save 'mcmc_prior_provenance' in metadata"
    )
    assert 'oof_{n_folds}fold_recording_prefix_grouped_cv' in content, (
        "prepare_metadata.py must format recording-prefix OOF provenance"
    )

    # Verify dt_ms CLI default is 4.0
    match = re.search(r'--dt_ms[\s\S]*?default=([0-9.]+)', content)
    assert match is not None, "Could not find --dt_ms in prepare_metadata.py"
    assert float(match.group(1)) == 4.0, f"--dt_ms default is {match.group(1)}, expected 4.0"


# =========================================================================
# 3. End-to-End CLI Pipeline Test
# =========================================================================

def _write_minimal_session_csvs(
    session_dir: Path, n_trials: int = 2, frames_per_trial: int = 100,
    session_id: str = "session_0", wind_onset_ms: float | None = None,
):
    """Create minimal valid kinematics and events CSVs for lazy loader."""
    kin_csv = session_dir / f"{session_id}_kinematics.csv"
    evt_csv = session_dir / f"{session_id}_events.csv"

    kin_header = [
        "session_id", "trial_id", "time_ms", "x_pos", "y_pos",
        "heading", "velocity", "acceleration", "visual_angle", "wind_state", "l_v_ratio"
    ]
    evt_header = ["session_id", "trial_id", "time_ms", "event_type", "event_value"]

    kin_lines = [",".join(kin_header) + "\n"]
    evt_lines = [",".join(evt_header) + "\n"]

    dt_ms = 4.0
    for t in range(n_trials):
        time_ms = np.arange(frames_per_trial) * dt_ms
        for i in range(frames_per_trial):
            wind = float(wind_onset_ms is not None and time_ms[i] >= wind_onset_ms)
            kin_lines.append(f"{session_id},{t},{time_ms[i]:.1f},0.0,0.0,0.0,0.1,0.0,0.0,{wind},120.0\n")

        evt_lines.append(f"{session_id},{t},0.0,trial_start,0\n")
        evt_lines.append(f"{session_id},{t},200.0,stimulus_onset,0\n")
        evt_lines.append(f"{session_id},{t},{time_ms[-1]:.1f},trial_stop,0\n")

    kin_csv.write_text("".join(kin_lines), encoding="utf-8")
    evt_csv.write_text("".join(evt_lines), encoding="utf-8")
    return kin_csv, evt_csv


def _unexpected_reducer():
    raise AssertionError("unsafe metadata reducer executed")


class _UnknownGlobal:
    def __reduce__(self):
        return _unexpected_reducer, ()


def test_converter_rejects_unknown_global_before_reducer(tmp_path: Path):
    metadata_path = tmp_path / "hostile_metadata.pt"
    output_path = tmp_path / "dataset.pt"
    torch.save({"trial_specs": _UnknownGlobal()}, metadata_path)

    with pytest.raises(ValueError, match="Unexpected serialized global"):
        convert_main(["--input", str(metadata_path), "--output", str(output_path)])
    assert not output_path.exists()


def test_actual_producer_metadata_converts_from_direct_cli(tmp_path: Path, monkeypatch):
    from scripts import prepare_metadata

    raw_dir = tmp_path / "raw"
    for name in ("animalA_session_1", "animalB_session_1"):
        session_dir = raw_dir / name
        session_dir.mkdir(parents=True)
        _write_minimal_session_csvs(
            session_dir, n_trials=1, frames_per_trial=100,
            session_id=name, wind_onset_ms=200.0,
        )

    metadata_path = tmp_path / "produced_metadata.pt"
    output_path = tmp_path / "produced_etl.pt"
    monkeypatch.setattr(sys, "argv", [
        "prepare_metadata.py", "--raw_dir", str(raw_dir),
        "--output", str(metadata_path),
    ])
    prepare_metadata.main()

    script = Path(__file__).resolve().parents[1] / "scripts" / "convert_metadata_to_etl.py"
    run = subprocess.run(
        [sys.executable, str(script), "--input", str(metadata_path),
         "--output", str(output_path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr
    converted = torch.load(output_path, weights_only=False)
    validate_dataset_provenance(converted, output_path)
    assert len(converted["X_seqs"]) == 2
    assert converted["mcmc_priors"].shape == (2, DEFAULT_FEATURE.mcmc_dim)
    assert converted["session_ids"] == ["animalA_session_1", "animalB_session_1"]
    assert converted["stimulus_conditions"] == ["wind_only", "wind_only"]
    assert all(0 <= anchor < length for anchor, length in zip(converted["anchor_frames"], converted["lengths"]))


def test_convert_metadata_to_etl_cli_e2e(tmp_path: Path):
    """Run convert_metadata_to_etl CLI and verify resulting file passes provenance gate."""
    session_dir = tmp_path / "session_0"
    session_dir.mkdir(parents=True)
    kin_csv, evt_csv = _write_minimal_session_csvs(session_dir, n_trials=2, frames_per_trial=50)

    trial_specs = [
        {
            "session_id": "session_0",
            "session_dir": str(session_dir),
            "kinematics_file": kin_csv.name,
            "events_file": evt_csv.name,
            "trial_id": 0,
            "n_frames": 50,
            "trial_start_ms": 0.0,
            "stimulus_onset_ms": 200.0,
            "anchor_ms": 200.0,
            "anchor_frame": 25,
            "anchor_rule": "stimulus_onset",
            "label": "NO_RESPONSE",
            "stimulus_condition": "multisensory",
            "is_pure_wind": False,
        },
        {
            "session_id": "session_0",
            "session_dir": str(session_dir),
            "kinematics_file": kin_csv.name,
            "events_file": evt_csv.name,
            "trial_id": 1,
            "n_frames": 50,
            "trial_start_ms": 0.0,
            "stimulus_onset_ms": 200.0,
            "anchor_ms": 200.0,
            "anchor_frame": 25,
            "anchor_rule": "stimulus_onset",
            "label": "NO_RESPONSE",
            "stimulus_condition": "wind_only",
            "is_pure_wind": True,
        },
    ]

    metadata = {
        "trial_specs": trial_specs,
        "mcmc_priors": torch.full((2, DEFAULT_FEATURE.mcmc_dim), 0.25),
        "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "n_trials": 2,
        "label_encoder": {label.name: label.value for label in Label},
        "feature_config": DEFAULT_FEATURE,
        "snapshot_anchor_rules": ["stimulus_onset", "stimulus_onset"],
        "n_sessions": 1,
        "session_ids": ["session_0", "session_0"],
        "anchor_frames": [25, 25],
        "stimulus_conditions": ["multisensory", "wind_only"],
        "is_pure_wind": np.array([False, True], dtype=bool),
        # Explicit frame interval matching the CSV timestamps written by
        # _write_minimal_session_csvs (dt_ms = 4.0).  Required for the
        # legacy pure-wind prepend migration (trial 1 lacks
        # pure_wind_prepended_frames) — no fabricated default.
        "dt_ms": 4.0,
    }

    # The old spec layout still works when its raw sources are explicitly bound.
    import hashlib
    for spec in trial_specs:
        spec["kinematics_sha256"] = hashlib.sha256(kin_csv.read_bytes()).hexdigest()
        spec["events_sha256"] = hashlib.sha256(evt_csv.read_bytes()).hexdigest()

    meta_path = tmp_path / "metadata.pt"
    etl_path = tmp_path / "dataset_etl.pt"
    torch.save(metadata, meta_path)

    # Run converter
    convert_main(["--input", str(meta_path), "--output", str(etl_path)])

    assert etl_path.exists()

    # Load converted dataset
    etl_data = torch.load(etl_path, weights_only=False)

    # 1. Provenance Gate
    validate_dataset_provenance(etl_data, etl_path)

    # 2. Key presence and value integrity
    assert etl_data["mcmc_prior_provenance"] == "oof_5fold_animal_grouped_cv"
    assert etl_data["animal_identity_status"] == "historical_unknown"
    assert "anchor_frames" in etl_data
    assert etl_data["anchor_frames"] == [25, 25]
    assert "stimulus_conditions" in etl_data
    assert etl_data["stimulus_conditions"] == ["multisensory", "wind_only"]
    assert "is_pure_wind" in etl_data
    np.testing.assert_array_equal(etl_data["is_pure_wind"], np.array([False, True], dtype=bool))
    assert len(etl_data["X_seqs"]) == 2
    assert len(etl_data["Y_seqs"]) == 2

# SOURCE10: the actual metadata producer, lazy reader and converter must agree.
def _source10_pair(directory, rows, events):
    import pandas as pd

    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / 'kinematics.csv', index=False)
    pd.DataFrame(events).to_csv(directory / 'events.csv', index=False)


def _source10_frames(session_id, trial_id, start, count, dt_ms, velocity, angle):
    return [dict(
        session_id=session_id, trial_id=trial_id, time_ms=frame * dt_ms,
        x_pos=0.0, y_pos=0.0, heading=0.0, velocity=velocity,
        acceleration=0.0, visual_angle=angle if frame >= 200 else 0.0,
        wind_state=int(frame >= 200), l_v_ratio=120.0,
    ) for frame in range(start, start + count)]


def _source10_event(session_id, trial_id, time_ms, event_type, value=''):
    return dict(session_id=session_id, trial_id=trial_id, time_ms=time_ms,
                event_type=event_type, event_value=value)


def _source10_produce(raw_dir, metadata_path, monkeypatch):
    from scripts import prepare_metadata

    priors = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    monkeypatch.setattr(prepare_metadata, 'resolve_group_folds', lambda *a, **kw: 2)
    monkeypatch.setattr(prepare_metadata, 'train_mcmc_cross_fitted',
                        lambda *a, **kw: (priors, [], []))
    monkeypatch.setattr(sys, 'argv', ['prepare_metadata.py', '--raw_dir', str(raw_dir),
                                      '--output', str(metadata_path)])
    prepare_metadata.main()
    return torch.load(metadata_path, weights_only=False), priors


@pytest.mark.parametrize('dt_ms,lengths,anchor,onset,step', [
    (4.0, [400, 300], 200, 200, 300),   # source cadence == model grid (identity)
    (10.0, [998, 748], 500, 500, 750),  # source cadence 10 ms -> model grid 4 ms
])
def test_split_trial_replays_all_source_pairs_at_both_cadences(
    tmp_path, monkeypatch, dt_ms, lengths, anchor, onset, step,
):
    from nsmor.lazy_dataloader import NSMoRLazyDataset
    from nsmor.pipeline.io import ClockAwareLazyDataset

    raw = tmp_path / 'raw'
    first, second = 'animalA_session_1', 'animalB_session_1'
    _source10_pair(raw / first,
                   _source10_frames(first, 11, 0, 300, dt_ms, 0.1, 10.0),
                   [_source10_event(first, 11, 200 * dt_ms, 'stimulus_onset')])
    _source10_pair(raw / second,
                   _source10_frames(first, 11, 300, 100, dt_ms, 2.1, 10.0)
                   + _source10_frames(second, 12, 0, 300, dt_ms, 0.2, 20.0),
                   [_source10_event(first, 11, 0, 'trial_start',
                                    '{"target_ttc_ms": -119.0}'),
                    _source10_event(second, 12, 0, 'trial_start'),
                    _source10_event(second, 12, 200 * dt_ms, 'stimulus_onset')])
    metadata_path = tmp_path / 'metadata.pt'
    output_path = tmp_path / 'etl.pt'
    metadata, priors = _source10_produce(raw, metadata_path, monkeypatch)
    # Source frames stay source evidence in the tensor-free metadata.
    assert [s['n_frames'] for s in metadata['trial_specs']] == [400, 300]
    assert metadata['target_ttc_ms'] == [-119.0, None]
    assert metadata['lazy_model_clock_contract']['dt_ms'] == 4.0

    # Legacy reader without a clock contract keeps source-cadence behavior.
    lazy = NSMoRLazyDataset(str(metadata_path))
    X, Y, length = lazy[0]
    assert length == 400 and X.shape == (400, 8) and Y.shape == (400,)
    assert [Path(pair['session_dir']).name for pair in metadata['trial_specs'][0]['source_pairs']] == [first, second]
    np.testing.assert_allclose(Y[:300], 0.1)
    np.testing.assert_allclose(Y[300:], 2.1)
    np.testing.assert_allclose(X[:, 4:].numpy(), np.broadcast_to(priors[0], (400, 4)), atol=1e-7)

    # The enhanced reader resamples each trial onto the declared model grid.
    enhanced = ClockAwareLazyDataset(str(metadata_path), dt_ms=4.0)
    Xe, Ye, le = enhanced[0]
    assert le == lengths[0] and Xe.shape == (lengths[0], 8)

    convert_main(['--input', str(metadata_path), '--output', str(output_path)])
    saved = torch.load(output_path, weights_only=False)
    validate_dataset_provenance(saved, output_path)
    assert metadata["animal_identity_status"] == saved["animal_identity_status"] == "unverified"
    assert saved["mcmc_prior_provenance"].endswith("recording_prefix_grouped_cv")
    assert saved['lengths'].tolist() == lengths
    assert list(zip(saved['session_ids'], saved['trial_ids'])) == [(first, 11), (second, 12)]
    assert saved['target_ttc_ms'] == [-119.0, None]
    np.testing.assert_allclose(saved['mcmc_priors'], priors, atol=1e-7)
    # Complete eager model-grid contract on the resampled arrays.
    assert saved['model_dt_ms'] == 4.0
    assert saved['anchor_frames'] == [anchor, anchor]
    assert len(saved['model_grid_provenance']) == 2
    record = saved['model_grid_provenance'][0]
    assert record['method'] == 'causal_previous_source_sample_hold'
    assert record['synthetic_prepend_frames'] == 0
    assert len(saved['X_seqs'][0]) == record['model_n'] + record['synthetic_prepend_frames']
    np.testing.assert_allclose(saved['Y_seqs'][0][:step], 0.1)
    np.testing.assert_allclose(saved['Y_seqs'][0][step:], 2.1)
    np.testing.assert_allclose(saved['X_seqs'][0][onset:, 0], 10.0)
    np.testing.assert_allclose(saved['X_seqs'][0][onset - 1, 0], 0.0)
    np.testing.assert_allclose(saved['X_seqs'][0][step + 1, 2], 2.1)
    from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint
    load_dataset_with_fingerprint(output_path, expected_dt_ms=4.0)
    with pytest.raises(ValueError, match='model_dt_ms.*expected_dt_ms'):
        load_dataset_with_fingerprint(output_path, expected_dt_ms=10.0)

    # A damaged new spec cannot silently convert the producer's 400 frames as 300.
    metadata['trial_specs'][0]['source_pairs'] = metadata['trial_specs'][0]['source_pairs'][:1]
    with pytest.raises(ValueError, match='300 reconstructed frames, expected 400'):
        NSMoRLazyDataset(str(metadata_path), metadata=metadata)[0]

    # The cached continuation must also be bound to its own events bytes.
    _source11_substitute(raw / second / 'events.csv', '-119.0', '-373.0')
    with pytest.raises(ValueError, match='changed.*regenerate metadata'):
        lazy[0]


def _source10_same_names(raw, dt_ms=4.0):
    for session_id, trial_id, velocity, angle, ttc in (
        ('animalA_session_1', 11, 0.1, 10.0, -119.0),
        ('animalB_session_1', 12, 0.2, 20.0, -200.0),
    ):
        _source10_pair(raw / session_id,
                       _source10_frames(session_id, trial_id, 0, 300, dt_ms, velocity, angle),
                       [_source10_event(session_id, trial_id, 0, 'trial_start',
                                        f'{{"target_ttc_ms": {ttc}}}'),
                        _source10_event(session_id, trial_id, 200 * dt_ms, 'stimulus_onset')])


def test_same_bare_filenames_in_distinct_sessions_do_not_share_cache(tmp_path, monkeypatch):
    from nsmor.lazy_dataloader import NSMoRLazyDataset

    raw = tmp_path / 'raw'
    _source10_same_names(raw)
    metadata_path, output_path = tmp_path / 'metadata.pt', tmp_path / 'etl.pt'
    metadata, priors = _source10_produce(raw, metadata_path, monkeypatch)
    assert [s['kinematics_file'] for s in metadata['trial_specs']] == ['kinematics.csv'] * 2
    lazy = NSMoRLazyDataset(str(metadata_path))
    assert [lazy[i][2] for i in range(2)] == [300, 300]
    np.testing.assert_allclose(lazy[0][1], 0.1)
    np.testing.assert_allclose(lazy[1][1], 0.2)

    convert_main(['--input', str(metadata_path), '--output', str(output_path)])
    saved = torch.load(output_path, weights_only=False)
    validate_dataset_provenance(saved, output_path)
    assert saved['lengths'].tolist() == [300, 300]
    assert list(zip(saved['session_ids'], saved['trial_ids'])) == [
        ('animalA_session_1', 11), ('animalB_session_1', 12)]
    assert saved['target_ttc_ms'] == [-119.0, -200.0]
    np.testing.assert_allclose(saved['mcmc_priors'], priors, atol=1e-7)
    np.testing.assert_allclose(saved['Y_seqs'][0], 0.1)
    np.testing.assert_allclose(saved['Y_seqs'][1], 0.2)


def test_converter_uses_one_decoded_snapshot_after_same_size_path_swap(tmp_path, monkeypatch):
    from scripts import convert_metadata_to_etl as converter

    raw = tmp_path / 'raw'
    _source10_same_names(raw)
    # Unique filenames isolate the pathname swap from the cache collision case.
    for directory in raw.iterdir():
        for kind in ('kinematics', 'events'):
            (directory / f'{kind}.csv').rename(directory / f'{directory.name}_{kind}.csv')
    metadata_path, output_path = tmp_path / 'metadata.pt', tmp_path / 'etl.pt'
    metadata, priors = _source10_produce(raw, metadata_path, monkeypatch)
    import io
    original_buffer = io.BytesIO()
    torch.save(metadata, original_buffer)
    original = original_buffer.getvalue()
    metadata['trial_specs'].reverse()
    metadata['mcmc_priors'] = metadata['mcmc_priors'].flip(0)
    for key in ('session_ids', 'trial_ids', 'target_ttc_ms', 'anchor_frames',
                'stimulus_conditions', 'snapshot_anchor_rules'):
        metadata[key].reverse()
    metadata['is_pure_wind'] = metadata['is_pure_wind'][::-1].copy()
    replacement_buffer = io.BytesIO()
    torch.save(metadata, replacement_buffer)
    replacement = replacement_buffer.getvalue()
    assert len(replacement) == len(original)
    metadata_path.write_bytes(original)

    decode = converter.load_artifact_bytes
    def swap_after_decode(data, **kwargs):
        captured = decode(data, **kwargs)
        metadata_path.write_bytes(replacement)
        return captured
    monkeypatch.setattr(converter, 'load_artifact_bytes', swap_after_decode)
    converter.main(['--input', str(metadata_path), '--output', str(output_path)])
    saved = torch.load(output_path, weights_only=False)
    validate_dataset_provenance(saved, output_path)
    assert saved['session_ids'] == ['animalA_session_1', 'animalB_session_1']
    assert saved['trial_ids'] == [11, 12]
    assert saved['target_ttc_ms'] == [-119.0, -200.0]
    np.testing.assert_allclose(saved['mcmc_priors'], priors, atol=1e-7)
    np.testing.assert_allclose(saved['Y_seqs'][0], 0.1)
    np.testing.assert_allclose(saved['Y_seqs'][1], 0.2)

# SOURCE11: bind metadata, lazy reads, and conversion to both exact CSV byte streams.
def _source11_fixture(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    _source10_same_names(raw)
    metadata_path, output_path = tmp_path / 'metadata.pt', tmp_path / 'etl.pt'
    metadata, priors = _source10_produce(raw, metadata_path, monkeypatch)
    return raw, metadata_path, output_path, metadata, priors


def _source11_substitute(path, old, new, swap=False):
    import os

    before = path.read_bytes()
    after = before.replace(old.encode(), new.encode())
    assert len(after) == len(before) and after != before
    if swap:
        replacement = path.with_suffix('.replacement')
        replacement.write_bytes(after)
        os.replace(replacement, path)
    else:
        path.write_bytes(after)


@pytest.mark.parametrize('kind,old,new', [
    ('kinematics', '0.1', '0.9'),
    ('events', '-119.0', '-373.0'),
])
@pytest.mark.parametrize('swap', [False, True])
def test_source11_changed_csv_refused_on_cache_hit_and_conversion(
    tmp_path, monkeypatch, kind, old, new, swap,
):
    import hashlib
    from nsmor.lazy_dataloader import NSMoRLazyDataset

    raw, metadata_path, output_path, metadata, priors = _source11_fixture(tmp_path, monkeypatch)
    source = metadata['trial_specs'][0]['source_pairs'][0]
    for name in ('kinematics', 'events'):
        path = raw / 'animalA_session_1' / f'{name}.csv'
        assert source[f'{name}_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()

    lazy = NSMoRLazyDataset(str(metadata_path))
    X, Y, length = lazy[0]
    assert X.shape == (300, 8) and Y.shape == (300,) and length == 300
    np.testing.assert_allclose(X[:, 4:].numpy(), np.broadcast_to(priors[0], (300, 4)), atol=1e-7)
    assert metadata['target_ttc_ms'][0] == -119.0
    path = raw / 'animalA_session_1' / f'{kind}.csv'
    _source11_substitute(path, old, new, swap)
    with pytest.raises(ValueError, match='changed.*regenerate metadata'):
        lazy[0]  # A cached DataFrame must not hide a raw-file replacement.
    with pytest.raises(ValueError, match='changed.*regenerate metadata'):
        NSMoRLazyDataset(str(metadata_path))[0]  # Cold lazy training path.
    with pytest.raises(ValueError, match='changed.*regenerate metadata'):
        convert_main(['--input', str(metadata_path), '--output', str(output_path)])
    assert not output_path.exists()


@pytest.mark.parametrize('kind,old,new', [
    ('kinematics', '0.1', '0.9'),
    ('events', '-119.0', '-373.0'),
])
def test_source11_mid_conversion_change_refused_at_output_gate(
    tmp_path, monkeypatch, kind, old, new,
):
    from scripts import convert_metadata_to_etl as converter

    raw, metadata_path, output_path, _, _ = _source11_fixture(tmp_path, monkeypatch)
    original_save = torch.save

    def change_source_after_temporary_save(obj, destination):
        original_save(obj, destination)
        _source11_substitute(raw / 'animalA_session_1' / f'{kind}.csv', old, new, swap=True)

    monkeypatch.setattr(converter.torch, 'save', change_source_after_temporary_save)
    with pytest.raises(ValueError, match='changed.*regenerate metadata'):
        converter.main(['--input', str(metadata_path), '--output', str(output_path)])
    assert not output_path.exists()
    assert not list(tmp_path.glob('.etl.pt.*.tmp'))


def test_source11_legacy_metadata_refuses_unbound_replay(tmp_path, monkeypatch):
    from nsmor.lazy_dataloader import NSMoRLazyDataset

    _, metadata_path, output_path, metadata, _ = _source11_fixture(tmp_path, monkeypatch)
    for spec in metadata['trial_specs']:
        for source in spec['source_pairs']:
            source.pop('kinematics_sha256', None)
            source.pop('events_sha256', None)
    torch.save(metadata, metadata_path)
    lazy = NSMoRLazyDataset(str(metadata_path))
    assert len(lazy) == 2  # Legacy metadata remains inspectable.
    with pytest.raises(ValueError, match='missing.*SHA-256.*regenerate metadata'):
        lazy[0]
    with pytest.raises(ValueError, match='missing.*SHA-256.*regenerate metadata'):
        convert_main(['--input', str(metadata_path), '--output', str(output_path)])
    assert not output_path.exists()


def test_source11_producer_refuses_source_changed_during_parse(tmp_path, monkeypatch):
    from scripts import prepare_metadata

    raw = tmp_path / 'raw'
    _source10_same_names(raw)
    metadata_path = tmp_path / 'metadata.pt'
    path = raw / 'animalA_session_1' / 'kinematics.csv'
    original_load = prepare_metadata.load_kinematics_csv
    calls = 0

    def change_after_parse(source, **kwargs):
        nonlocal calls
        table = original_load(source, **kwargs)
        calls += 1
        if calls == 1:
            _source11_substitute(path, '0.1', '0.9')
        return table

    monkeypatch.setattr(prepare_metadata, 'load_kinematics_csv', change_after_parse)
    monkeypatch.setattr(sys, 'argv', ['prepare_metadata.py', '--raw_dir', str(raw),
                                      '--output', str(metadata_path)])
    with pytest.raises(ValueError, match='changed.*regenerate metadata'):
        prepare_metadata.main()
    assert not metadata_path.exists()


# SOURCE12: publication failures retain the previous artifact; ETL binds captured bytes.
@pytest.mark.parametrize('kind,old,new', [
    ('kinematics', '0.1', '0.9'),
    ('events', '-119.0', '-373.0'),
])
def test_source12_metadata_mutation_during_save_preserves_previous(
    tmp_path, monkeypatch, kind, old, new,
):
    from scripts import prepare_metadata

    raw, metadata_path, _, _, _ = _source11_fixture(tmp_path, monkeypatch)
    previous = metadata_path.read_bytes()
    original_save = torch.save

    def change_during_save(obj, destination):
        assert Path(destination) != metadata_path
        original_save(obj, destination)
        _source11_substitute(raw / 'animalA_session_1' / f'{kind}.csv', old, new, swap=True)

    monkeypatch.setattr(prepare_metadata.torch, 'save', change_during_save)
    with pytest.raises(ValueError, match='changed.*regenerate metadata'):
        prepare_metadata.main()
    assert metadata_path.read_bytes() == previous
    assert not list(tmp_path.glob('.metadata.pt.*.tmp'))


def test_source12_failed_metadata_save_preserves_previous(tmp_path, monkeypatch):
    from scripts import prepare_metadata

    _, metadata_path, _, _, _ = _source11_fixture(tmp_path, monkeypatch)
    previous = metadata_path.read_bytes()

    def partial_save(obj, destination):
        assert Path(destination) != metadata_path
        Path(destination).write_bytes(b'incomplete-new-artifact')
        raise OSError('serialization failed')

    monkeypatch.setattr(prepare_metadata.torch, 'save', partial_save)
    with pytest.raises(OSError, match='serialization failed'):
        prepare_metadata.main()
    assert metadata_path.read_bytes() == previous
    assert not list(tmp_path.glob('.metadata.pt.*.tmp'))


def test_source12_etl_path_swap_at_publication_claims_captured_revision(tmp_path, monkeypatch):
    import hashlib
    from scripts import convert_metadata_to_etl as converter
    from nsmor.lazy_dataloader import NSMoRLazyDataset

    raw, metadata_path, output_path, metadata, _ = _source11_fixture(tmp_path, monkeypatch)
    path = raw / 'animalA_session_1' / 'kinematics.csv'
    original = path.read_bytes()
    expected = hashlib.sha256(original).hexdigest()
    real_replace = converter.os.replace
    published = False

    def swap_on_publication(source, destination):
        nonlocal published
        if Path(destination) == output_path:
            replacement = path.with_suffix('.replacement')
            altered = original.replace(b'0.1', b'0.9')
            assert len(altered) == len(original) and altered != original
            replacement.write_bytes(altered)
            real_replace(replacement, path)
            published = True
        return real_replace(source, destination)

    monkeypatch.setattr(converter.os, 'replace', swap_on_publication)
    converter.main(['--input', str(metadata_path), '--output', str(output_path)])
    assert published
    saved = torch.load(output_path, weights_only=False)
    validate_dataset_provenance(saved, output_path)
    np.testing.assert_allclose(saved['Y_seqs'][0], 0.1)
    revision = saved['raw_input_revision']
    assert revision['claim'] == 'SHA-256 of captured CSV bytes; paths are labels, not a live-path guarantee'
    assert revision['source_pairs'][0]['kinematics_sha256'] == expected
    assert metadata['trial_specs'][0]['source_pairs'][0]['kinematics_sha256'] == expected
    assert hashlib.sha256(path.read_bytes()).hexdigest() != expected
    with pytest.raises(ValueError, match='changed.*regenerate metadata'):
        NSMoRLazyDataset(str(metadata_path))[0]


def test_source12_metadata_path_swap_at_publication_claims_captured_revision(
    tmp_path, monkeypatch,
):
    import hashlib
    from scripts import prepare_metadata
    from nsmor.lazy_dataloader import NSMoRLazyDataset

    raw, metadata_path, _, _, _ = _source11_fixture(tmp_path, monkeypatch)
    path = raw / 'animalA_session_1' / 'kinematics.csv'
    original = path.read_bytes()
    expected = hashlib.sha256(original).hexdigest()
    real_replace = prepare_metadata.os.replace
    published = False

    def swap_on_publication(source, destination):
        nonlocal published
        if Path(destination) == metadata_path:
            altered = original.replace(b'0.1', b'0.9')
            assert len(altered) == len(original) and altered != original
            replacement = path.with_suffix('.replacement')
            replacement.write_bytes(altered)
            real_replace(replacement, path)
            published = True
        return real_replace(source, destination)

    monkeypatch.setattr(prepare_metadata.os, 'replace', swap_on_publication)
    prepare_metadata.main()
    assert published
    saved = torch.load(metadata_path, weights_only=False)
    revision = saved['raw_input_revision']
    assert revision['claim'] == 'SHA-256 of captured CSV bytes; paths are labels, not a live-path guarantee'
    assert revision['source_pairs'][0]['kinematics_sha256'] == expected
    assert saved['trial_specs'][0]['source_pairs'][0]['kinematics_sha256'] == expected
    assert hashlib.sha256(path.read_bytes()).hexdigest() != expected
    with pytest.raises(ValueError, match='changed.*regenerate metadata'):
        NSMoRLazyDataset(str(metadata_path))[0]
    assert not list(tmp_path.glob('.metadata.pt.*.tmp'))
