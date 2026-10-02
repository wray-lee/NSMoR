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


def test_subset_keeps_modern_trial_rows_aligned(tmp_path: Path) -> None:
    from nsmor.config import FeatureConfig, PIPELINE_SEMANTICS_VERSION
    from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint
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
        "source_clock_provenance": [
            [{"source_row_indices": [i], "raw_sys_time": [str(i)],
              "raw_ard_time": [str(i)], "time_source": ["sys_time"]}]
            for i in range(8)
        ],
        "model_dt_ms": 4.,
        "model_grid_provenance": [
            {"model_n": i + 3, "synthetic_prepend_frames": 0,
             "source_n": i + 3, "origin_ms": float(i), "dt_ms": 4.,
             "source_end_ms": float(i + 4 * (i + 2)),
             "model_end_ms": float(i + 4 * (i + 2)),
             "source_anchor_ms": float(i + 2),
             "source_anchor_rule": "looming_collision"}
            for i in range(8)
        ],
        "anchor_frames": [1] * 8,
        "labeling_eligibility": [
            {"session_id": sid, "trial_id": 100 + i, "status": "labeled",
             "label": "NO_RESPONSE", "source_rows": i + 3}
            for i, sid in enumerate(sessions)
        ] + [{"session_id": sessions[0], "trial_id": 999,
              "status": "unavailable_no_stimulus_anchor", "label": None},
             {"session_id": sessions[2], "trial_id": 998,
              "status": "unavailable_no_stimulus_anchor", "label": None}],
        "labeling_funnel_retention": {"n_retained_sequences": 8},
        "animal_identity_status": "unverified",
    }
    # Distinct lengths expose grid records accidentally retained in source order.
    data["X_seqs"] = [np.full((i + 3, 8), float(i), dtype=np.float32) for i in range(8)]
    data["Y_seqs"] = [np.zeros(i + 3, dtype=np.float32) for i in range(8)]
    data["lengths"] = np.arange(3, 11, dtype=np.int64)
    from copy import deepcopy
    from nsmor.pipeline.clock_storage import pack_clock_provenance, unpack_clock_provenance
    data["source_clock_provenance"][1] = pack_clock_provenance(data["source_clock_provenance"][1])
    data["source_clock_provenance"][3] = None
    for i, row in enumerate(data["labeling_eligibility"][:8]):
        row["clock_provenance"] = data["source_clock_provenance"][i]
    original_ledger = deepcopy(data["labeling_eligibility"])

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
    assert subset["model_grid_provenance"] == [
        data["model_grid_provenance"][i] for i in (0, 1, 3, 6)
    ]
    assert len(subset["source_clock_provenance"]) == 4
    for j, i in enumerate((0, 1, 3, 6)):
        assert subset["source_clock_provenance"][j] is data["source_clock_provenance"][i]
    assert [row["trial_id"] for row in subset["labeling_eligibility"]] == [
        100, 101, 103, 106, 999, 998,
    ]
    assert subset["labeling_eligibility"][-2:] == original_ledger[-2:]
    assert subset["subset_labeling_accounting"] == {
        "scope": "selected_sequences_and_all_parent_unavailable",
        "n_sequences": 4,
        "n_parent_entries": 10,
        "n_excluded_labeled": 4,
        "n_labeled": 4,
        "n_unavailable": 2,
        "n_entries": 6,
        "parent_summary_keys": ["labeling_funnel_retention"],
    }
    assert data["labeling_eligibility"] == original_ledger
    for j in range(4):
        assert subset["source_clock_provenance"][j] is (
            subset["labeling_eligibility"][j]["clock_provenance"]
        )

    source, output = tmp_path / "source.pt", tmp_path / "subset.pt"
    torch.save(data, source)
    main(["--input", str(source), "--output", str(output),
          "--n_recording_prefixes", "3", "--seed", "42"])
    saved, _ = load_dataset_with_fingerprint(output)
    assert saved["model_grid_provenance"] == subset["model_grid_provenance"]
    assert saved["anchor_frames"] == [1, 1, 1, 1]
    assert saved["lengths"].tolist() == [3, 4, 6, 9]
    assert saved["subset_labeling_accounting"] == subset["subset_labeling_accounting"]
    assert [row["trial_id"] for row in saved["labeling_eligibility"]] == [
        100, 101, 103, 106, 999, 998,
    ]
    assert saved["labeling_eligibility"][-2:] == original_ledger[-2:]
    assert saved["labeling_funnel_retention"] == {"n_retained_sequences": 8}
    assert len(saved["source_clock_provenance"]) == 4
    for j, i in enumerate((0, 1, 3, 6)):
        assert saved["source_clock_provenance"][j] == unpack_clock_provenance(
            data["source_clock_provenance"][i]
        )
        assert saved["source_clock_provenance"][j] is (
            saved["labeling_eligibility"][j]["clock_provenance"]
        )
        np.testing.assert_array_equal(saved["X_seqs"][j], data["X_seqs"][i])
        np.testing.assert_array_equal(saved["Y_seqs"][j], data["Y_seqs"][i])


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
