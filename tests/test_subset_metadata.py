"""Regression tests for legacy subset stimulus-condition metadata."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path

from scripts.make_subset_dataset import derive_stimulus_metadata, main, subset_dataset


def _trial(visual: float, wind: float) -> np.ndarray:
    """Build a minimal valid 8-feature trial with supplied modalities."""
    x = np.zeros((3, 8), dtype=np.float32)
    x[:, 0] = visual
    x[:, 1] = wind
    return x


def test_derive_stimulus_metadata_uses_physical_channels() -> None:
    """Conditions derive from visual/wind signals, not labels or sessions."""
    x_seqs = [
        _trial(2.0, 1.0),
        _trial(3.0, 0.0),
        _trial(0.0, 1.0),
        _trial(0.0, 0.0),
    ]
    conditions, is_pure_wind = derive_stimulus_metadata(
        x_seqs, np.array([3, 3, 3, 3])
    )

    assert conditions.tolist() == [
        "multisensory",
        "visual_only",
        "wind_only",
        "no_stimulus",
    ]
    assert is_pure_wind.tolist() == [False, False, True, False]


def test_legacy_subset_derives_routing_metadata() -> None:
    """A legacy artifact gains aligned condition metadata after subsetting."""
    x_seqs = [_trial(1.0, 0.0), _trial(0.0, 1.0)]
    data = {
        "X_seqs": x_seqs,
        "Y_seqs": [np.zeros(3, dtype=np.float32) for _ in x_seqs],
        "labels": np.array([0, 1]),
        "lengths": np.array([3, 3]),
        "mcmc_priors": np.full((2, 4), 0.25, dtype=np.float32),
        "session_ids": np.array([
            "cricket_a_session_1",
            "cricket_b_session_1",
        ], dtype=object),
        "pipeline_semantics_version": "2.2",
        "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
    }

    subset, kept, _ = subset_dataset(data, n_animals=2, seed=42)

    assert kept == [0, 1]
    assert subset["stimulus_conditions"].tolist() == [
        "visual_only",
        "wind_only",
    ]
    assert subset["is_pure_wind"].tolist() == [False, True]
    assert subset["animal_identity_status"] == "historical_unknown"


def test_existing_routing_metadata_is_sliced_not_rederived() -> None:
    """Already-audited source metadata stays authoritative after subsetting."""
    data = {
        "X_seqs": [_trial(1.0, 0.0), _trial(0.0, 1.0)],
        "Y_seqs": [np.zeros(3, dtype=np.float32) for _ in range(2)],
        "labels": np.array([0, 1]),
        "lengths": np.array([3, 3]),
        "mcmc_priors": np.full((2, 4), 0.25, dtype=np.float32),
        "session_ids": np.array([
            "cricket_a_session_1",
            "cricket_b_session_1",
        ], dtype=object),
        "stimulus_conditions": np.array(["visual_only", "wind_only"], dtype=object),
        "is_pure_wind": np.array([False, True]),
        "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
    }

    subset, _, _ = subset_dataset(data, n_animals=2, seed=42)

    assert subset["stimulus_conditions"].tolist() == [
        "visual_only",
        "wind_only",
    ]
    assert subset["is_pure_wind"].tolist() == [False, True]
    assert subset["animal_identity_status"] == "historical_unknown"


def _unexpected_reducer(marker: str) -> str:
    Path(marker).write_text("executed", encoding="utf-8")
    return "untrusted"


class _UnknownGlobal:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return _unexpected_reducer, (str(self.marker),)


def test_subset_cli_rejects_reducer_before_side_effect(tmp_path: Path) -> None:
    source = tmp_path / "untrusted.pt"
    output = tmp_path / "subset.pt"
    marker = tmp_path / "reducer-executed"
    torch.save({"unexpected": _UnknownGlobal(marker)}, source)

    try:
        with pytest.raises(ValueError, match="Unexpected serialized global"):
            main(["--input", str(source), "--output", str(output),
                  "--n_animals", "1"])
    finally:
        assert not marker.exists(), "source reducer executed"
    assert not output.exists()


def test_subset_keeps_modern_trial_rows_aligned() -> None:
    from nsmor.config import FeatureConfig, PIPELINE_SEMANTICS_VERSION
    from scripts.evaluate_nested_prior import validate_nested_dataset

    sessions = ["recording_A_session_1", "recording_A_session_2",
                "recording_D_session_1", "recording_B_session_1",
                "recording_D_session_2", "recording_E_session_1",
                "recording_C_session_1", "recording_E_session_2"]
    data = {
        "X_seqs": [_trial(float(i), 0.0) for i in range(8)],
        "Y_seqs": [np.zeros(3, dtype=np.float32) for _ in range(8)],
        "labels": np.array([0, 1, 0, 2, 0, 0, 3, 0]),
        "lengths": np.full(8, 3, dtype=np.int64),
        "mcmc_priors": np.full((8, 4), 0.25),
        "session_ids": sessions,
        "trial_ids": np.array([100, 101, 102, 103, 104, 105, 106, 107]),
        "target_ttc_ms": [100., 110., 120., 130., 140., 150., 160., 170.],
        "snapshots": np.arange(40, dtype=np.float64).reshape(8, 5),
        "mcmc_snapshots": torch.arange(40, dtype=torch.float64).reshape(8, 5),
        "anchor_rules": ["rule_0", "rule_1", "rule_2", "rule_3",
                         "rule_4", "rule_5", "rule_6", "rule_7"],
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
    }

    subset, kept, prefixes = subset_dataset(data, n_animals=3, seed=42)

    assert kept == [0, 1, 3, 6]
    assert prefixes == ["recording_A", "recording_B", "recording_C"]
    assert subset["trial_ids"].tolist() == [100, 101, 103, 106]
    assert subset["target_ttc_ms"] == [100., 110., 130., 160.]
    expected_snapshots = [[0., 1., 2., 3., 4.], [5., 6., 7., 8., 9.],
                          [15., 16., 17., 18., 19.], [30., 31., 32., 33., 34.]]
    assert subset["snapshots"].tolist() == expected_snapshots
    assert subset["mcmc_snapshots"].tolist() == expected_snapshots
    assert subset["labels"].tolist() == [0, 1, 2, 3]
    assert subset["anchor_rules"] == ["rule_0", "rule_1", "rule_3", "rule_6"]
    assert subset["session_ids"] == [sessions[i] for i in (0, 1, 3, 6)]
    assert subset["animal_identity_status"] == "unverified"
    assert validate_nested_dataset(subset, FeatureConfig())[0].tolist() == expected_snapshots


@pytest.mark.parametrize("flag", ["--n_recording_prefixes", "--n_animals"])
def test_cli_reports_prefixes_without_claiming_animal_identity(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, flag: str,
) -> None:
    import hashlib
    from nsmor.config import PIPELINE_SEMANTICS_VERSION
    from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint

    # Both recording days are stipulated to be one animal; prefixes cannot prove that.
    sessions = ["subjectA_day1_session_1", "subjectA_day1_session_2",
                "subjectA_day2_session_1", "subjectA_day2_session_2"]
    data = {
        "X_seqs": [_trial(1.0, 0.0) for _ in sessions],
        "Y_seqs": [np.zeros(3, dtype=np.float32) for _ in sessions],
        "lengths": np.full(4, 3, dtype=np.int64),
        "labels": np.array([0, 1, 2, 3]),
        "mcmc_priors": np.full((4, 4), 0.25),
        "session_ids": sessions,
        "trial_ids": np.array([1, 2, 1, 2]),
        "target_ttc_ms": np.array([110., 120., 130., 140.]),
        "snapshots": np.arange(20, dtype=np.float64).reshape(4, 5),
        "mcmc_snapshots": np.arange(20, dtype=np.float64).reshape(4, 5),
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
    }
    source, output = tmp_path / "source.pt", tmp_path / "subset.pt"
    torch.save(data, source)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    caplog.set_level("INFO")

    main(["--input", str(source), "--output", str(output), flag, "2"])

    saved, _ = load_dataset_with_fingerprint(output)
    assert saved["mcmc_prior_provenance"] == "oof_2fold_recording_prefix_grouped_cv"
    assert saved["animal_identity_status"] == "unverified"
    assert saved["subset_source_sha256"] == source_digest
    assert saved["trial_ids"].tolist() == [1, 2, 1, 2]
    assert saved["target_ttc_ms"].tolist() == [110., 120., 130., 140.]
    assert saved["snapshots"].shape == (4, 5)
    assert saved["mcmc_snapshots"].shape == (4, 5)
    assert "2 recording prefixes" in caplog.text
    assert "animal identity: unverified" in caplog.text
    assert "independent animal identities" in caplog.text
    assert "2 animals" not in caplog.text
