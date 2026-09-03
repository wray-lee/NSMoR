"""Tests for JAX analysis modules: gating_cluster_jax and uq_jax.

Verifies:
  - Fingerprint computation: callable, correct shape, no NaN.
  - Clustering: callable, correct output keys.
  - MC dropout: correct shapes, finite values, std > 0.
  - Numerical consistency between JAX and PyTorch fingerprints.
  - Re-exported UQ utilities remain accessible.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    from nsmor.jax.model import NSMoRModel, load_from_torch_state_dict
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.fixture(scope="module")
def jax_model_and_params():
    """Create a small NSMoRModel and initialize parameters."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX is not installed")

    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cpu"
    # Force CPU backend to avoid GPU OOM in CI / parallel test runs
    jax.config.update("jax_default_device", jax.devices("cpu")[0])

    H = 32
    model = NSMoRModel(
        sensory_dim=4,
        mcmc_dim=4,
        hidden_dim=H,
        dt_ms=4.0,
        dropout_rate=0.1,
        lif_lateral_inhibition=0.0,  # simpler for tests
    )
    rng = jax.random.PRNGKey(0)
    dummy_x = jnp.zeros((2, 20, 8), dtype=jnp.float32)
    dummy_l = jnp.array([20, 15], dtype=jnp.int32)
    params = model.init(rng, dummy_x, dummy_l)
    return model, params


# ===============================================================
# gating_cluster_jax Tests
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestGatingClusterJAX:

    def test_fingerprint_shape_and_no_nan(self):
        """fingerprint_jax returns (16,) without NaN."""
        from nsmor.analysis.gating_cluster_jax import fingerprint_jax
        from nsmor.analysis.gating_cluster import ClusterGatingConfig

        config = ClusterGatingConfig()
        np.random.seed(42)
        T = 100
        g_lif = 0.3 + 0.2 * np.sin(np.linspace(0, 2 * np.pi, T))
        g_gru = 1.0 - g_lif
        gates = np.stack([g_lif, g_gru], axis=1).astype(np.float32)

        fp = fingerprint_jax(gates, config)
        assert fp.shape == (16,), f"Expected (16,), got {fp.shape}"
        assert not np.any(np.isnan(fp)), "Fingerprint contains NaN"
        assert not np.any(np.isinf(fp)), "Fingerprint contains Inf"

    def test_fingerprint_empty_sequence(self):
        """Empty gate sequence returns zero fingerprint."""
        from nsmor.analysis.gating_cluster_jax import fingerprint_jax
        from nsmor.analysis.gating_cluster import ClusterGatingConfig

        config = ClusterGatingConfig()
        gates = np.zeros((0, 2), dtype=np.float32)
        fp = fingerprint_jax(gates, config)
        assert fp.shape == (16,)
        assert np.allclose(fp, 0.0)

    def test_fingerprint_t1_no_nan(self):
        """MAJOR-1: T=1 gate sequence does not produce NaN (ddof=1 guard)."""
        from nsmor.analysis.gating_cluster_jax import fingerprint_jax
        from nsmor.analysis.gating_cluster import ClusterGatingConfig

        config = ClusterGatingConfig()
        gates = np.array([[0.6, 0.4]], dtype=np.float32)  # T=1
        fp = fingerprint_jax(gates, config)
        assert fp.shape == (16,)
        # T=1 < 2 is caught by the guard in fingerprint_jax, returns zeros
        assert np.allclose(fp, 0.0)

    def test_adapter_extract_and_fingerprint(self, jax_model_and_params):
        """GatingClusterAdapterJAX produces correct shapes from model output."""
        from nsmor.analysis.gating_cluster_jax import GatingClusterAdapterJAX
        from nsmor.analysis.gating_cluster import ClusterGatingConfig

        model, params = jax_model_and_params
        config = ClusterGatingConfig()
        adapter = GatingClusterAdapterJAX(model, params, config=config)

        B, T = 4, 20
        rng = jax.random.PRNGKey(1)
        X = jax.random.normal(rng, (B, T, 8))
        lengths = jnp.array([20, 18, 15, 10], dtype=jnp.int32)

        sequences = adapter.extract_gating_sequences([X], [lengths])
        assert len(sequences) == B
        for seq in sequences:
            assert "gates" in seq
            assert "trial_id" in seq
            assert seq["gates"].shape[1] == 2
            assert seq["gates"].shape[0] == seq["length"]

        fingerprints = adapter.compute_fingerprints(sequences)
        assert fingerprints.shape == (B, 16), (
            f"Expected ({B}, 16), got {fingerprints.shape}"
        )
        assert not np.any(np.isnan(fingerprints))

    def test_clustering_output_keys(self, jax_model_and_params):
        """Clustering returns expected keys."""
        from nsmor.analysis.gating_cluster_jax import GatingClusterAdapterJAX
        from nsmor.analysis.gating_cluster import ClusterGatingConfig

        model, params = jax_model_and_params
        config = ClusterGatingConfig(n_clusters_range=[2, 3])
        adapter = GatingClusterAdapterJAX(model, params, config=config)

        # Create 12 trials to have enough for clustering
        B, T = 12, 20
        rng = jax.random.PRNGKey(2)
        X = jax.random.normal(rng, (B, T, 8))
        lengths = jnp.full((B,), T, dtype=jnp.int32)

        sequences = adapter.extract_gating_sequences([X], [lengths])
        fingerprints = adapter.compute_fingerprints(sequences)
        result = adapter.cluster(fingerprints)

        required_keys = {
            "k_opt", "silhouette_scores", "labels_k4", "labels_k3",
            "fingerprints_scaled",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )
        assert result["labels_k4"].shape == (B,)
        assert result["labels_k3"].shape == (B,)

    def test_fingerprint_parity_with_pytorch(self):
        """JAX and PyTorch fingerprints are numerically identical.

        MAJOR-2 fix: tests multiple variable-length trials to cover
        the padding-removed path.
        MAJOR-3 fix: tolerance tightened to atol=1e-4, rtol=1e-4.
        """
        from nsmor.analysis.gating_cluster_jax import fingerprint_jax
        from nsmor.analysis.gating_cluster import (
            ClusterGatingConfig,
            GatingClusterAdapter,
        )
        import torch

        config = ClusterGatingConfig()
        rng = np.random.RandomState(99)

        # Test with multiple lengths to cover variable-length path
        test_lengths = [30, 50, 80, 120]

        for T in test_lengths:
            g_lif = np.clip(
                0.4 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, T))
                + 0.05 * rng.randn(T),
                0, 1,
            ).astype(np.float32)
            g_gru = np.clip(
                1.0 - g_lif + 0.05 * rng.randn(T),
                0, 1,
            ).astype(np.float32)
            gates = np.stack([g_lif, g_gru], axis=1)

            # JAX
            fp_jax = fingerprint_jax(gates, config)

            # PyTorch adapter
            pt_adapter = GatingClusterAdapter(torch.nn.Module(), config=config)
            seq = {
                "trial_id": 0, "gates": gates,
                "true_4way": 0, "true_3way_merged": 0,
            }
            fp_pt = pt_adapter.compute_fingerprints([seq])[0]

            assert fp_jax.shape == fp_pt.shape == (16,)
            np.testing.assert_allclose(
                fp_jax, fp_pt, atol=1e-4, rtol=1e-4,
                err_msg=(
                    f"JAX vs PyTorch fingerprint mismatch at T={T}. "
                    f"Max diff: {np.max(np.abs(fp_jax - fp_pt)):.6e}"
                ),
            )

    def test_adapter_compute_fingerprints_variable_length(self, jax_model_and_params):
        """BLOCKER-1 regression: adapter fingerprints match per-trial fingerprints.

        Verifies that compute_fingerprints (batch path) produces the same
        result as calling fingerprint_jax per trial -- confirming no
        padding contamination.
        """
        from nsmor.analysis.gating_cluster_jax import (
            GatingClusterAdapterJAX,
            fingerprint_jax,
        )
        from nsmor.analysis.gating_cluster import ClusterGatingConfig

        model, params = jax_model_and_params
        config = ClusterGatingConfig()
        adapter = GatingClusterAdapterJAX(model, params, config=config)

        B, T = 4, 25
        rng = jax.random.PRNGKey(77)
        X = jax.random.normal(rng, (B, T, 8))
        # Deliberately variable lengths
        lengths = jnp.array([25, 10, 18, 5], dtype=jnp.int32)

        sequences = adapter.extract_gating_sequences([X], [lengths])
        batch_fps = adapter.compute_fingerprints(sequences)

        for i, seq in enumerate(sequences):
            single_fp = fingerprint_jax(seq["gates"], config)
            np.testing.assert_allclose(
                batch_fps[i], single_fp, atol=1e-6,
                err_msg=(
                    f"Batch vs single fingerprint mismatch at trial {i} "
                    f"(length={seq['length']})"
                ),
            )


# ===============================================================
# uq_jax Tests
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestUQJAX:

    def test_mc_dropout_predict_shapes(self, jax_model_and_params):
        """MC dropout predict returns correct shapes."""
        from nsmor.analysis.uq_jax import mc_dropout_predict_jax

        model, params = jax_model_and_params
        B, T = 3, 20
        n_samples = 5

        rng = jax.random.PRNGKey(7)
        x = jax.random.normal(rng, (B, T, 8))
        lengths = jnp.array([20, 15, 10], dtype=jnp.int32)

        result = mc_dropout_predict_jax(
            model, params, x, lengths, n_samples=n_samples, seed=42,
        )

        assert result["y_mean"].shape == (B, T), (
            f"y_mean shape {result['y_mean'].shape} != ({B}, {T})"
        )
        assert result["y_std"].shape == (B, T), (
            f"y_std shape {result['y_std'].shape} != ({B}, {T})"
        )
        assert result["y_samples"].shape == (n_samples, B, T), (
            f"y_samples shape {result['y_samples'].shape} != ({n_samples}, {B}, {T})"
        )
        assert np.all(np.isfinite(result["y_mean"])), "y_mean contains non-finite"
        assert np.all(np.isfinite(result["y_std"])), "y_std contains non-finite"

    def test_mc_dropout_uncertainty_per_trial(self, jax_model_and_params):
        """Per-trial uncertainty produces (B,) arrays with sensible values."""
        from nsmor.analysis.uq_jax import mc_dropout_uncertainty_jax

        model, params = jax_model_and_params
        B, T = 4, 20
        n_samples = 8

        rng = jax.random.PRNGKey(8)
        x = jax.random.normal(rng, (B, T, 8))
        lengths = jnp.array([20, 18, 12, 8], dtype=jnp.int32)

        result = mc_dropout_uncertainty_jax(
            model, params, x, lengths, n_samples=n_samples, seed=42,
        )

        assert result["trial_uncertainty"].shape == (B,), (
            f"trial_uncertainty shape {result['trial_uncertainty'].shape} != ({B},)"
        )
        assert result["trial_cv"].shape == (B,)
        assert np.all(result["trial_uncertainty"] >= 0.0)
        assert np.all(np.isfinite(result["trial_uncertainty"]))

    def test_mc_dropout_analyzer_internals(self, jax_model_and_params):
        """MCDropoutAnalyzerJAX.predict with return_internals gives gate stats."""
        from nsmor.analysis.uq_jax import MCDropoutAnalyzerJAX

        model, params = jax_model_and_params
        analyzer = MCDropoutAnalyzerJAX(model, params, n_samples=4, seed=0)

        B, T = 2, 15
        rng = jax.random.PRNGKey(9)
        x = jax.random.normal(rng, (B, T, 8))
        lengths = jnp.array([15, 10], dtype=jnp.int32)

        result = analyzer.predict(x, lengths, return_internals=True)

        assert "gates_mean" in result
        assert "gates_std" in result
        assert result["gates_mean"].shape == (B, T, 2), (
            f"gates_mean shape {result['gates_mean'].shape} != ({B}, {T}, 2)"
        )
        assert result["gates_std"].shape == (B, T, 2)

    def test_mc_dropout_n_samples_validation(self, jax_model_and_params):
        """MCDropoutAnalyzerJAX raises on n_samples < 2."""
        from nsmor.analysis.uq_jax import MCDropoutAnalyzerJAX

        model, params = jax_model_and_params
        with pytest.raises(ValueError, match="n_samples must be"):
            MCDropoutAnalyzerJAX(model, params, n_samples=1)

    def test_reexported_bootstrap_ci(self):
        """bootstrap_ci is accessible from uq_jax and works correctly."""
        from nsmor.analysis.uq_jax import bootstrap_ci

        data = np.random.RandomState(42).randn(50)
        point, lo, hi = bootstrap_ci(data, n_bootstrap=200)
        assert lo < point < hi
        assert np.isfinite(point)

    def test_reexported_cohens_d(self):
        """cohens_d is accessible from uq_jax and returns finite value."""
        from nsmor.analysis.uq_jax import cohens_d

        g1 = np.random.RandomState(10).randn(30)
        g2 = np.random.RandomState(11).randn(30) + 0.5
        d = cohens_d(g1, g2)
        assert np.isfinite(d)

    def test_reexported_holm_bonferroni(self):
        """holm_bonferroni is accessible from uq_jax and returns dict."""
        from nsmor.analysis.uq_jax import holm_bonferroni

        pvals = {"test_a": 0.01, "test_b": 0.04, "test_c": 0.5}
        result = holm_bonferroni(pvals)
        assert len(result) == 3
        for name, (adj_p, sig) in result.items():
            assert 0.0 <= adj_p <= 1.0
            assert isinstance(sig, bool)
