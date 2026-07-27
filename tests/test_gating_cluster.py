"""
Tests for gating_cluster module.

Window-free by design. NSMoR is Trial-Start anchored. TTC-50ms is only
for MCMC prior 5-D snapshot. Baseline 5700ms is variant for pure-wind via
TimeWindowConfig, not universal. Manual windows like [-5700:-500] inject
human bias and break unsupervised claim. Clustering is unsupervised
(silhouette selects k without labels); k=4 matches labeling.py cardinality;
k=3 merged is for biological interpretation only and defined as:
    Startle->Escape, Walk+Pre_Active->PreWalk, NoResponse->NoResponse.
Pearson NaN guarded to 0.0 when std==0.
"""

from __future__ import annotations

import random
import re
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from nsmor.analysis.gating_cluster import (
    ClusterGatingConfig,
    GatingClusterAdapter,
    fingerprint,
    compute_fingerprints,
    cluster,
    interpolate_for_viz,
)
from nsmor.config import Label


class TestGatingClusterConfig:
    """Test ClusterGatingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        cfg = ClusterGatingConfig()

        assert cfg.n_clusters == 4
        assert cfg.n_clusters_range == [2, 3, 4, 5]
        assert cfg.random_state == 42
        assert cfg.use_umap is True
        assert cfg.fingerprint_dim == 16
        assert cfg.entropy_bins == 20
        assert cfg.interp_length == 200

    def test_frozen_dataclass(self) -> None:
        """Test that config is frozen (immutable)."""
        cfg = ClusterGatingConfig()

        with pytest.raises(FrozenInstanceError):
            cfg.n_clusters = 5

    def test_no_window_field(self) -> None:
        """Verify no window-related fields exist in config."""
        cfg = ClusterGatingConfig()
        fields = dir(cfg)

        # No window fields
        window_keywords = ["window", "baseline", "ttc", "5700", "TimeWindowConfig"]
        for keyword in window_keywords:
            assert not any(keyword.lower() in f.lower() for f in fields if not f.startswith("_")), \
                f"Found window-related field: {keyword}"


class TestFingerprint:
    """Test fingerprint computation."""

    def test_fingerprint_shape(self) -> None:
        """Test fingerprint returns correct shape."""
        config = ClusterGatingConfig(fingerprint_dim=16, entropy_bins=20)
        gates = np.random.rand(100, 2).astype(np.float64)

        fp = fingerprint(gates, config)

        assert fp.shape == (16,), f"Expected (16,), got {fp.shape}"

    def test_fingerprint_dimensions_constant(self) -> None:
        """Test fingerprint has correct 16 dimensions."""
        config = ClusterGatingConfig(fingerprint_dim=16, entropy_bins=20)
        gates = np.random.rand(100, 2).astype(np.float64)

        fp = fingerprint(gates, config)

        # [0-5] LIF features
        assert fp[0] >= 0 and fp[0] <= 1  # mean in [0,1]
        assert fp[2] >= 0 and fp[2] <= 1  # max in [0,1]
        assert fp[3] >= 0 and fp[3] <= 1  # min in [0,1]

        # [6-11] GRU features
        assert fp[6] >= 0 and fp[6] <= 1  # mean in [0,1]

        # [12] Pearson correlation
        assert fp[12] >= -1 and fp[12] <= 1

        # [13-14] Normalized times in [0,1]
        assert fp[13] >= 0 and fp[13] <= 1
        assert fp[14] >= 0 and fp[14] <= 1

        # [15] Max gradient >= 0
        assert fp[15] >= 0

    def test_entropy_determinism(self) -> None:
        """Test entropy calculation is deterministic."""
        config = ClusterGatingConfig(entropy_bins=20)

        np.random.seed(42)
        gates1 = np.random.rand(100, 2).astype(np.float64)

        np.random.seed(42)
        gates2 = np.random.rand(100, 2).astype(np.float64)

        fp1 = fingerprint(gates1, config)
        fp2 = fingerprint(gates2, config)

        np.testing.assert_allclose(fp1, fp2)

    def test_pearson_nan_guard(self) -> None:
        """Test Pearson correlation returns 0.0 when std==0."""
        config = ClusterGatingConfig(fingerprint_dim=16)

        # Constant gates - std = 0
        gates = np.ones((50, 2), dtype=np.float64) * 0.5

        fp = fingerprint(gates, config)

        assert fp[12] == 0.0, f"Expected 0.0 for constant gates, got {fp[12]}"

    def test_pearson_nan_guard_single_timestep(self) -> None:
        """Test Pearson correlation handles single timestep."""
        config = ClusterGatingConfig(fingerprint_dim=16)

        # Single timestep
        gates = np.array([[0.5, 0.5]], dtype=np.float64)

        fp = fingerprint(gates, config)

        assert fp[12] == 0.0, f"Expected 0.0 for single timestep, got {fp[12]}"

    def test_pearson_correlated_gates(self) -> None:
        """Test Pearson correlation for anti-correlated gates."""
        config = ClusterGatingConfig(fingerprint_dim=16)

        # Anti-correlated gates
        t = np.linspace(0, 1, 100)
        g_lif = 0.3 + 0.4 * np.sin(2 * np.pi * t)
        g_gru = 1.0 - g_lif
        gates = np.stack([g_lif, g_gru], axis=1).astype(np.float64)

        fp = fingerprint(gates, config)

        # Should be strongly negative
        assert fp[12] < -0.5, f"Expected negative correlation, got {fp[12]}"

    def test_no_window_strings_in_code(self) -> None:
        """Verify no forbidden window strings in non-docstring code."""
        import inspect

        # Get source of fingerprint function
        source = inspect.getsource(fingerprint)

        # Remove docstrings (simple approach)
        source_no_docstring = re.sub(r'""".*?"""', '', source, flags=re.DOTALL)
        source_no_docstring = re.sub(r"'''.*?'''", '', source_no_docstring, flags=re.DOTALL)

        forbidden = ["5700", "TTC", "baseline_duration", "TimeWindowConfig"]
        for term in forbidden:
            assert term not in source_no_docstring, \
                f"Found forbidden term '{term}' in fingerprint function"


class TestComputeFingerprints:
    """Test compute_fingerprints function."""

    def test_batch_fingerprints(self) -> None:
        """Test computing fingerprints for multiple sequences."""
        config = ClusterGatingConfig(fingerprint_dim=16)

        sequences = [
            {"gates": np.random.rand(80, 2).astype(np.float64)},
            {"gates": np.random.rand(100, 2).astype(np.float64)},
            {"gates": np.random.rand(120, 2).astype(np.float64)},
            {"gates": np.random.rand(90, 2).astype(np.float64)},
        ]

        fps = compute_fingerprints(sequences, config)

        assert fps.shape == (4, 16), f"Expected (4, 16), got {fps.shape}"


class TestCluster:
    """Test clustering functionality."""

    def test_determinism(self) -> None:
        """Test clustering is deterministic with same seed."""
        config = ClusterGatingConfig(random_state=42)

        # Create synthetic data with clear clusters
        np.random.seed(42)
        data = np.vstack([
            np.random.randn(20, 16) + [5] * 16,
            np.random.randn(20, 16) + [-5] * 16,
            np.random.randn(20, 16) + [0] * 16,
        ])

        result1 = cluster(data, config)
        result2 = cluster(data, config)

        np.testing.assert_array_equal(result1["labels_k4"], result2["labels_k4"])
        np.testing.assert_array_equal(result1["labels_k3"], result2["labels_k3"])

    def test_silhouette_scores(self) -> None:
        """Test silhouette scores are computed."""
        config = ClusterGatingConfig(random_state=42)

        np.random.seed(42)
        data = np.vstack([
            np.random.randn(20, 16) + [5] * 16,
            np.random.randn(20, 16) + [-5] * 16,
            np.random.randn(20, 16) + [0] * 16,
        ])

        result = cluster(data, config)

        assert "silhouette_scores" in result
        assert len(result["silhouette_scores"]) > 0

        # All scores should be in [-1, 1]
        for k, score in result["silhouette_scores"].items():
            assert -1 <= score <= 1, f"Silhouette score {score} out of range for k={k}"

    def test_k_opt_selection(self) -> None:
        """Test k_opt is in n_clusters_range."""
        config = ClusterGatingConfig(
            n_clusters_range=[2, 3, 4, 5],
            random_state=42,
        )

        np.random.seed(42)
        data = np.vstack([
            np.random.randn(20, 16) + [5] * 16,
            np.random.randn(20, 16) + [-5] * 16,
            np.random.randn(20, 16) + [0] * 16,
        ])

        result = cluster(data, config)

        assert result["k_opt"] in config.n_clusters_range


class TestGatingClusterAdapter:
    """Test GatingClusterAdapter class."""

    def test_map_4way_to_3way(self) -> None:
        """Test 4-way to 3-way mapping."""
        # Startle -> Escape
        assert GatingClusterAdapter._map_4way_to_3way(Label.ESCAPE) == 0

        # Walk -> PreWalk
        assert GatingClusterAdapter._map_4way_to_3way(Label.PREWALK) == 1

        # Pre_Active -> PreWalk
        assert GatingClusterAdapter._map_4way_to_3way(Label.PRE_ACTIVE) == 1

        # NoResponse -> NoResponse
        assert GatingClusterAdapter._map_4way_to_3way(Label.NO_RESPONSE) == 2

    def test_map_4way_to_3way_unknown(self) -> None:
        """Test mapping of unknown label returns -1."""
        assert GatingClusterAdapter._map_4way_to_3way(-1) == -1
        assert GatingClusterAdapter._map_4way_to_3way(99) == -1


class TestInterpolateForViz:
    """Test trajectory interpolation for visualization."""

    def test_interpolation_shape(self) -> None:
        """Test interpolation produces correct shape."""
        gates = np.random.rand(50, 2).astype(np.float64)
        target_length = 200

        interp = interpolate_for_viz(gates, target_length)

        assert interp.shape == (200, 2), f"Expected (200, 2), got {interp.shape}"

    def test_interpolation_identity(self) -> None:
        """Test interpolation with same length returns copy."""
        gates = np.random.rand(100, 2).astype(np.float64)

        interp = interpolate_for_viz(gates, 100)

        assert interp.shape == (100, 2)
        np.testing.assert_allclose(interp, gates, rtol=1e-5)

    def test_interpolation_range(self) -> None:
        """Test interpolated values stay in valid range."""
        gates = np.random.rand(50, 2).astype(np.float64)

        interp = interpolate_for_viz(gates, 200)

        # Interpolated values should be within [min, max] of original
        assert interp.min() >= gates.min() - 1e-10
        assert interp.max() <= gates.max() + 1e-10


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_end_to_end_mock(self) -> None:
        """Test end-to-end with mock data."""
        # Set seeds for determinism
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)

        # Create mock sequences
        n_trials = 20
        sequences = []
        for i in range(n_trials):
            t_len = 80 + np.random.randint(-10, 11)
            gates = np.random.rand(t_len, 2).astype(np.float64)
            # Make some anti-correlated
            if i < 5:
                gates[:, 1] = 1.0 - gates[:, 0]
            sequences.append({
                "trial_id": i,
                "gates": gates,
                "true_4way": i % 4,
                "true_3way_merged": i % 3,
            })

        config = ClusterGatingConfig(random_state=42)

        # Compute fingerprints
        fingerprints = compute_fingerprints(sequences, config)
        assert fingerprints.shape == (n_trials, 16)

        # Cluster
        result = cluster(fingerprints, config)
        assert "labels_k4" in result
        assert "labels_k3" in result
        assert "k_opt" in result

        # Check determinism
        assert len(result["labels_k4"]) == n_trials
        assert len(result["labels_k3"]) == n_trials


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
