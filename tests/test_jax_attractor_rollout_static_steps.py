"""Phase E regression for actual JAX attractor perturbation rollouts.

Proposal / DEVELOP progress (2026-09-28): lax.scan requires a static length;
functools.partial makes only K static without changing the GRU or adapter API.
The zero-weight GRU has h_next = 0.5*h, so distinct K values must yield finite,
monotonic trajectories and the correct convergence decision/residual.
Validation: stdlib AST/compile only; runtime is deferred until training release.
A fresh v5 seal and independent A/B review remain required.
"""

import numpy as np
import pytest


def test_actual_jax_attractor_rollout_at_distinct_static_lengths():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import torch

    from nsmor.analysis.dynamics_jax import FixedPointAdapterJAX
    from nsmor.model_nsmor_core import NSMoRCore

    hidden_dim = 4
    model = NSMoRCore(hidden_dim=hidden_dim, dt_ms=4.0, dropout=0.0).cpu().eval()
    with torch.no_grad():
        for parameter in model.backend.gru_unit.gru.parameters():
            parameter.zero_()

    # Scope all JAX allocations/JITs to CPU without changing global defaults.
    with jax.default_device(jax.devices("cpu")[0]):
        adapter = FixedPointAdapterJAX(model, device=torch.device("cpu"), backend="jax")
        assert adapter.use_jax  # A Torch fallback cannot validate this bug.
        h_star = np.zeros(hidden_dim, dtype=np.float32)
        x_fixed = jnp.zeros(hidden_dim, dtype=jnp.float32)
        h_initial = np.array([0.05, -0.02, 0.01, -0.04], dtype=np.float32)
        for K in (2, 8):
            # The same compiled kernel must specialize for each scan length.
            trajectory = np.asarray(
                adapter._rollout_k_steps_jit(jnp.asarray(h_initial), x_fixed, K)
            )
            assert trajectory.shape == (K, hidden_dim)
            assert np.all(np.isfinite(trajectory))
            expected = (0.5 ** np.arange(1, K + 1))[:, None] * h_initial[None, :]
            assert expected.shape == trajectory.shape
            np.testing.assert_allclose(trajectory, expected, rtol=1e-6, atol=1e-9)
            assert np.all(np.diff(np.linalg.norm(trajectory, axis=1)) < 0)

            # Exercise the real fixed-point/Jacobian/eigenvector/scan branch;
            # a positive analytic residual rules out an early zero return.
            is_attractor, max_residual, monotonic = adapter.test_attractor_convergence(
                h_star, x_fixed, perturbation_magnitude=0.05,
                convergence_radius=0.001, K=K, n_directions=hidden_dim,
            )
            assert np.isfinite(max_residual) and max_residual > 0
            assert max_residual == pytest.approx(0.05 * 0.5 ** K, rel=1e-6)
            assert is_attractor == (K == 8)
            assert monotonic
