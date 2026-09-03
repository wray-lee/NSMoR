"""Unit tests for per-condition gate statistics (Ticket #17)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_gating import _compute_condition_gate_stats


def test_compute_condition_stats_with_is_pure_wind():
    """Compute stats when is_pure_wind metadata is present."""
    sequences = [
        {"gate_seq": np.array([[0.8, 0.2], [0.9, 0.1]]), "is_pure_wind": True},
        {"gate_seq": np.array([[0.3, 0.7], [0.4, 0.6]]), "is_pure_wind": False},
        {"gate_seq": np.array([[0.7, 0.3], [0.8, 0.2]]), "is_pure_wind": True},
    ]

    stats = _compute_condition_gate_stats(sequences)

    assert stats is not None
    assert "mean_g_lif_wind" in stats
    assert "mean_g_lif_visual" in stats
    assert "separation" in stats
    assert "cohens_d" in stats

    # Wind trials: (0.8+0.9)/2=0.85, (0.7+0.8)/2=0.75 → mean=0.80
    # Visual trials: (0.3+0.4)/2=0.35
    assert abs(stats["mean_g_lif_wind"] - 0.80) < 0.01
    assert abs(stats["mean_g_lif_visual"] - 0.35) < 0.01
    assert abs(stats["separation"] - 0.45) < 0.01
    assert stats["n_wind_trials"] == 2
    assert stats["n_visual_trials"] == 1


def test_compute_condition_stats_with_stimulus_condition():
    """Compute stats when stimulus_condition string is present."""
    sequences = [
        {"gate_seq": np.array([[0.9, 0.1]]), "stimulus_condition": "wind_only"},
        {"gate_seq": np.array([[0.3, 0.7]]), "stimulus_condition": "visual_only"},
        {"gate_seq": np.array([[0.8, 0.2]]), "stimulus_condition": "wind_only"},
    ]

    stats = _compute_condition_gate_stats(sequences)

    assert stats is not None
    assert abs(stats["mean_g_lif_wind"] - 0.85) < 0.01
    assert abs(stats["mean_g_lif_visual"] - 0.30) < 0.01
    assert stats["n_wind_trials"] == 2
    assert stats["n_visual_trials"] == 1


def test_compute_condition_stats_no_metadata_returns_none():
    """Return None when no metadata is present."""
    sequences = [
        {"gate_seq": np.array([[0.8, 0.2], [0.9, 0.1]])},
        {"gate_seq": np.array([[0.3, 0.7], [0.4, 0.6]])},
    ]

    stats = _compute_condition_gate_stats(sequences)

    assert stats is None


def test_compute_condition_stats_empty_group_returns_none():
    """Return None when one group is empty."""
    sequences = [
        {"gate_seq": np.array([[0.9, 0.1]]), "is_pure_wind": True},
        {"gate_seq": np.array([[0.8, 0.2]]), "is_pure_wind": True},
    ]

    stats = _compute_condition_gate_stats(sequences)

    assert stats is None  # No visual trials


def test_compute_condition_stats_cohens_d_calculation():
    """Cohen's d is computed correctly."""
    # Create sequences with known statistics
    sequences = [
        {"gate_seq": np.array([[1.0, 0.0]]), "is_pure_wind": True},
        {"gate_seq": np.array([[1.0, 0.0]]), "is_pure_wind": True},
        {"gate_seq": np.array([[0.0, 1.0]]), "is_pure_wind": False},
        {"gate_seq": np.array([[0.0, 1.0]]), "is_pure_wind": False},
    ]

    stats = _compute_condition_gate_stats(sequences)

    # Wind: mean=1.0, std=0.0
    # Visual: mean=0.0, std=0.0
    # pooled_std=0.0 → Cohen's d should handle division by zero
    assert stats is not None
    assert stats["cohens_d"] == 0.0  # Handled gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
