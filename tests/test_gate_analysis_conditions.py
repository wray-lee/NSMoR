"""Unit tests for per-condition gate statistics (Ticket #17)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.analyze_gating import _compute_condition_gate_stats, build_summary_json
from nsmor.analysis.gating_cluster import ClusterGatingConfig


def test_summary_excludes_unclassified_trials_and_preserves_conditions() -> None:
    """Only explicit visual-present trials enter the routing contrast."""
    sequences = [
        {"stimulus_condition": "wind_only", "is_pure_wind": True},
        {"stimulus_condition": "visual_only", "is_pure_wind": False},
        {"stimulus_condition": "multisensory", "is_pure_wind": False},
        {"stimulus_condition": "no_stimulus", "is_pure_wind": False},
        {"stimulus_condition": "unknown", "is_pure_wind": False},
        {"is_pure_wind": False},
        {"stimulus_condition": "wind_only", "is_pure_wind": False},
    ]
    for seq, value in zip(sequences, [0.8, 0.2, 0.4, 1.0, 1.0, 1.0, 1.0]):
        seq.update(
            gates=np.array([[value, 1.0 - value]]),
            true_4way=0,
            true_3way_merged=0,
        )
    result = {
        "sequences": sequences,
        "k_opt": 2,
        "silhouette_scores": {},
        "stability_scores": {},
        "k_selection_basis": "test",
    }
    stats = build_summary_json(result, ClusterGatingConfig())[
        "condition_specific_routing"
    ]
    assert stats["n_wind_trials"] == 1
    assert stats["n_visual_trials"] == 2
    assert stats["mean_g_lif_visual"] == pytest.approx(0.3)
    assert stats["condition_counts"] == {
        "wind_only": 1, "visual_only": 1, "multisensory": 1, "no_stimulus": 1,
    }
    assert stats["per_condition"]["multisensory"]["mean_g_lif"] == 0.4
    assert stats["excluded_trial_counts"] == {
        "unknown_condition": 1, "missing_condition": 1,
        "conflicting_metadata": 1,
    }


def test_compute_condition_stats_with_is_pure_wind() -> None:
    """A false legacy flag does not establish a visual-present trial."""
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

    assert stats["mean_g_lif_wind"] is None
    assert stats["mean_g_lif_visual"] is None
    assert stats["comparison_available"] is False
    assert stats["n_wind_trials"] == 2
    assert stats["n_visual_trials"] == 0
    assert stats["per_condition"]["wind_only"]["mean_g_lif"] == pytest.approx(0.8)
    assert stats["excluded_trial_counts"] == {"missing_condition": 1}


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
    assert stats["cohens_d"] is None
    assert stats["cohens_d_unavailable_reason"] == "fewer_than_two_trials_per_group"
    json.dumps(stats, allow_nan=False)


def test_compute_condition_stats_no_metadata_returns_none():
    """Return None when no metadata is present."""
    sequences = [
        {"gate_seq": np.array([[0.8, 0.2], [0.9, 0.1]])},
        {"gate_seq": np.array([[0.3, 0.7], [0.4, 0.6]])},
    ]

    stats = _compute_condition_gate_stats(sequences)

    assert stats is None


def test_compute_condition_stats_empty_group_reports_unavailable() -> None:
    """Keep counts when one contrast group is empty."""
    sequences = [
        {"gate_seq": np.array([[0.9, 0.1]]), "is_pure_wind": True},
        {"gate_seq": np.array([[0.8, 0.2]]), "is_pure_wind": True},
    ]

    stats = _compute_condition_gate_stats(sequences)

    assert stats is not None
    assert not stats["comparison_available"]
    assert stats["unavailable_reason"] == "missing_wind_or_visual_present_group"
    assert stats["n_wind_trials"] == 2
    assert stats["n_visual_trials"] == 0


def test_compute_condition_stats_cohens_d_calculation() -> None:
    """Separated constant groups have undefined d, not zero separation."""
    sequences = [
        {"gate_seq": np.array([[1.0, 0.0]]), "stimulus_condition": "wind_only"},
        {"gate_seq": np.array([[1.0, 0.0]]), "stimulus_condition": "wind_only"},
        {"gate_seq": np.array([[0.0, 1.0]]), "stimulus_condition": "visual_only"},
        {"gate_seq": np.array([[0.0, 1.0]]), "stimulus_condition": "visual_only"},
    ]
    stats = _compute_condition_gate_stats(sequences)
    assert stats is not None
    assert stats["separation"] == 1.0
    assert stats["cohens_d"] is None
    assert stats["cohens_d_unavailable_reason"] == "zero_pooled_variance"


def test_cohens_d_uses_sample_size_weighted_sample_variance() -> None:
    """Unequal trial groups use pooled sample variance, not population SDs."""
    sequences = [
        {"gates": np.array([[v, 1 - v]]), "stimulus_condition": condition}
        for condition, values in (
            ("wind_only", [0.6, 1.0]),
            ("visual_only", [0.0, 0.2, 0.4]),
        )
        for v in values
    ]
    stats = _compute_condition_gate_stats(sequences)
    # Hand calculation: means .8/.2; pooled variance (.08 + 2*.04)/3.
    assert stats["cohens_d"] == pytest.approx(2.598076211353316)
    assert stats["cohens_d_unavailable_reason"] is None


@pytest.mark.parametrize(
    "condition, flag",
    [("wind_only", False), ("visual_only", True), ("multisensory", True)],
)
def test_conflicting_conditions_are_reported_not_reassigned(
    condition: str, flag: bool,
) -> None:
    """Neither inconsistent metadata field may silently win classification."""
    stats = _compute_condition_gate_stats([
        {"gates": np.array([[0.5, 0.5]]), "stimulus_condition": condition,
         "is_pure_wind": flag},
    ])
    assert stats["excluded_trial_counts"] == {"conflicting_metadata": 1}
    assert sum(stats["condition_counts"].values()) == 0
    assert not stats["comparison_available"]
    json.dumps(stats, allow_nan=False)


def test_empty_nonfinite_gates_are_reported_without_nan_summary() -> None:
    """Missing numeric support cannot produce a fabricated routing contrast."""
    stats = _compute_condition_gate_stats([
        {"gates": np.empty((0, 2)), "stimulus_condition": "wind_only"},
        {"gates": np.array([[np.nan, 0.5]]),
         "stimulus_condition": "visual_only"},
        {"stimulus_condition": "multisensory"},
    ])
    assert stats["excluded_trial_counts"] == {
        "empty_or_nonfinite_gates": 2, "missing_gates": 1,
    }
    assert not stats["comparison_available"]
    json.dumps(stats, allow_nan=False)


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
