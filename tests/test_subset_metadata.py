"""Regression tests for legacy subset stimulus-condition metadata."""

from __future__ import annotations

import numpy as np

from scripts.make_subset_dataset import derive_stimulus_metadata, subset_dataset


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
    }

    subset, kept, _ = subset_dataset(data, n_animals=2, seed=42)

    assert kept == [0, 1]
    assert subset["stimulus_conditions"].tolist() == [
        "visual_only",
        "wind_only",
    ]
    assert subset["is_pure_wind"].tolist() == [False, True]


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
    }

    subset, _, _ = subset_dataset(data, n_animals=2, seed=42)

    assert subset["stimulus_conditions"].tolist() == [
        "visual_only",
        "wind_only",
    ]
    assert subset["is_pure_wind"].tolist() == [False, True]
