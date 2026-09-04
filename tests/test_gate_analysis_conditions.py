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


class TestPureWindFallbackDerivation:
    """The legacy-artifact fallback must not fold no_stimulus into wind.

    ``load_model_and_dataset`` derives ``is_pure_wind`` when a corpus predates
    the condition stamp. An earlier version tested only "visual channel is
    silent", which labels a no_stimulus trial (both channels silent) as
    pure-wind. On ``nsmor_subset_small.pt`` that put 36 no_stimulus trials
    into a wind group whose true size is 0, so every per-condition gate
    statistic derived from it was fabricated.
    """

    @staticmethod
    def _trial(has_visual: bool, has_wind: bool, n_frames: int = 6) -> np.ndarray:
        """Build an (n_frames, 8) trial with the requested physical channels."""
        x = np.zeros((n_frames, 8), dtype=np.float32)
        if has_visual:
            x[2:, 0] = 12.5  # visual_angle
        if has_wind:
            x[3:, 1] = 1.0  # wind_state
        return x

    def test_no_stimulus_is_not_pure_wind(self):
        """A trial with neither channel active is no_stimulus, not wind."""
        from scripts.analyze_gating import derive_stimulus_metadata

        x_seqs = [self._trial(has_visual=False, has_wind=False)]
        conditions, is_pure_wind = derive_stimulus_metadata(x_seqs, [6])

        assert conditions[0] == "no_stimulus"
        assert not bool(is_pure_wind[0]), (
            "no_stimulus trial was labeled pure-wind; the wind group of every "
            "per-condition gate statistic is contaminated"
        )

    def test_wind_without_visual_is_pure_wind(self):
        """Wind present and visual absent is the only pure-wind case."""
        from scripts.analyze_gating import derive_stimulus_metadata

        x_seqs = [self._trial(has_visual=False, has_wind=True)]
        conditions, is_pure_wind = derive_stimulus_metadata(x_seqs, [6])

        assert conditions[0] == "wind_only"
        assert bool(is_pure_wind[0])

    def test_all_four_conditions_separate(self):
        """The fallback reproduces the canonical four-way split."""
        from scripts.analyze_gating import derive_stimulus_metadata

        x_seqs = [
            self._trial(has_visual=False, has_wind=False),  # no_stimulus
            self._trial(has_visual=True, has_wind=False),   # visual_only
            self._trial(has_visual=False, has_wind=True),   # wind_only
            self._trial(has_visual=True, has_wind=True),    # multisensory
        ]
        conditions, is_pure_wind = derive_stimulus_metadata(x_seqs, [6] * 4)

        assert list(conditions) == [
            "no_stimulus", "visual_only", "wind_only", "multisensory",
        ]
        # Exactly one trial is pure-wind: index 2.
        assert is_pure_wind.tolist() == [False, False, True, False]

    def test_fallback_matches_prepare_data_classifier(self):
        """The fallback agrees with the source-of-truth classifier."""
        from scripts.analyze_gating import derive_stimulus_metadata
        from scripts.prepare_data import classify_stimulus_condition

        x_seqs = [
            self._trial(has_visual=v, has_wind=w)
            for v in (False, True)
            for w in (False, True)
        ]
        conditions, _ = derive_stimulus_metadata(x_seqs, [6] * len(x_seqs))

        for x, derived in zip(x_seqs, conditions):
            expected = classify_stimulus_condition(
                {"visual_angle": x[:, 0], "wind_state": x[:, 1]}
            )
            assert derived == expected, (
                f"fallback said {derived!r}, classifier said {expected!r}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
