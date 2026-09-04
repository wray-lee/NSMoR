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
        "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
        "anchor_frames": [1200, 1205],
        "stimulus_conditions": ["multisensory", "wind_only"],
        "is_pure_wind": np.array([False, True], dtype=bool),
    }

    output: Dict[str, Any] = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "X_seqs": [np.zeros((10, 8)), np.zeros((10, 8))],
        "Y_seqs": [np.zeros(10), np.zeros(10)],
    }

    populate_etl_provenance_and_conditions(output, metadata)

    assert output["mcmc_prior_provenance"] == "oof_5fold_animal_grouped_cv"
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
    """prepare_metadata.py must output animal-grouped provenance and have dt_ms default 4.0."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_metadata.py"
    content = script_path.read_text(encoding="utf-8")

    # Verify mcmc_prior_provenance in metadata dict
    assert '"mcmc_prior_provenance":' in content, (
        "prepare_metadata.py must save 'mcmc_prior_provenance' in metadata"
    )
    assert 'oof_{n_folds}fold_animal_grouped_cv' in content, (
        "prepare_metadata.py must format provenance as oof_{n_folds}fold_animal_grouped_cv"
    )

    # Verify dt_ms CLI default is 4.0
    match = re.search(r'--dt_ms[\s\S]*?default=([0-9.]+)', content)
    assert match is not None, "Could not find --dt_ms in prepare_metadata.py"
    assert float(match.group(1)) == 4.0, f"--dt_ms default is {match.group(1)}, expected 4.0"


# =========================================================================
# 3. End-to-End CLI Pipeline Test
# =========================================================================

def _write_minimal_session_csvs(session_dir: Path, n_trials: int = 2, frames_per_trial: int = 100):
    """Create minimal valid kinematics and events CSVs for lazy loader."""
    kin_csv = session_dir / "session_kinematics.csv"
    evt_csv = session_dir / "session_events.csv"

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
            kin_lines.append(f"session_0,{t},{time_ms[i]:.1f},0.0,0.0,0.0,0.1,0.0,0.0,0.0,120.0\n")

        evt_lines.append(f"session_0,{t},0.0,trial_start,0\n")
        evt_lines.append(f"session_0,{t},200.0,stimulus_onset,0\n")
        evt_lines.append(f"session_0,{t},{time_ms[-1]:.1f},trial_stop,0\n")

    kin_csv.write_text("".join(kin_lines), encoding="utf-8")
    evt_csv.write_text("".join(evt_lines), encoding="utf-8")
    return kin_csv, evt_csv


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
            "label": "NoResponse",
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
            "label": "NoResponse",
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
    }

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
    assert "anchor_frames" in etl_data
    assert etl_data["anchor_frames"] == [25, 25]
    assert "stimulus_conditions" in etl_data
    assert etl_data["stimulus_conditions"] == ["multisensory", "wind_only"]
    assert "is_pure_wind" in etl_data
    np.testing.assert_array_equal(etl_data["is_pure_wind"], np.array([False, True], dtype=bool))
    assert len(etl_data["X_seqs"]) == 2
    assert len(etl_data["Y_seqs"]) == 2
