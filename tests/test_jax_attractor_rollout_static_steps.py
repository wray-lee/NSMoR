"""Phase E regression for actual JAX attractor perturbation rollouts.

Proposal / DEVELOP progress (2026-09-28): lax.scan requires a static length;
functools.partial makes only K static without changing the GRU or adapter API.
The zero-weight GRU has h_next = 0.5*h, so distinct K values must yield finite,
monotonic trajectories and the correct convergence decision/residual.
Validation: stdlib AST/compile only; runtime is deferred until training release.
A fresh v5 seal and independent A/B review remain required.
"""

from __future__ import annotations

from contextlib import nullcontext
import logging

import numpy as np
import pytest
import torch

from nsmor.analysis.dynamics import FixedPointAdapter
from nsmor.analysis.dynamics_jax import FixedPointAdapterJAX
from nsmor.model_nsmor_core import NSMoRCore


def _model_with_zero_point_jacobian(jacobian: torch.Tensor) -> NSMoRCore:
    """Build an actual GRU whose zero-input fixed point has this Jacobian."""
    hidden_dim = jacobian.shape[0]
    assert jacobian.shape == (hidden_dim, hidden_dim)
    model = NSMoRCore(hidden_dim=hidden_dim, dt_ms=4.0, dropout=0.0).cpu().eval()
    gru = model.backend.gru_unit.gru
    with torch.no_grad():
        for parameter in gru.parameters():
            parameter.zero_()
        # r = z = 0.5; at h = x = 0, J = 0.5*I + 0.25*W_hh_n.
        gru.weight_hh_l0[2 * hidden_dim:].copy_(
            4.0 * (jacobian - 0.5 * torch.eye(hidden_dim))
        )
    return model


@pytest.mark.parametrize("backend", ["torch", "jax", "jax-torch"])
@pytest.mark.parametrize(
    ("mode", "expected_attractor"),
    [
        ("unstable_real", False),
        ("unstable_complex", False),
        ("weakly_expanding", False),
        ("stable_real", True),
        ("stable_complex", True),
        ("within_spectral_tolerance", True),
    ],
)
def test_attractor_checks_full_spectrum(
    backend: str,
    mode: str,
    expected_attractor: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Near-unit slow modes must not hide an expanding mode in a 64-D GRU."""
    hidden_dim = 64
    diagonal = torch.full((hidden_dim,), 0.5)
    diagonal[:3] = torch.tensor([0.99, 0.98, 0.97])
    jacobian = torch.diag(diagonal)
    if mode == "unstable_real":
        jacobian[3, 3] = 1.5
    elif mode == "weakly_expanding":
        jacobian[3, 3] = 1.0001
    elif mode == "within_spectral_tolerance":
        jacobian[3, 3] = 1.0 + 5e-7
    elif mode.endswith("complex"):
        imaginary = 1.1 if mode == "unstable_complex" else 0.95
        jacobian[3:5, 3:5] = torch.tensor(
            [[0.3, -imaginary], [imaginary, 0.3]]
        )
    model = _model_with_zero_point_jacobian(jacobian)

    context = nullcontext()
    if backend == "jax":
        jax = pytest.importorskip("jax")
        context = jax.default_device(jax.devices("cpu")[0])
    with context:
        adapter = (
            FixedPointAdapter(model, device=torch.device("cpu"))
            if backend == "torch"
            else FixedPointAdapterJAX(
                model, device=torch.device("cpu"),
                backend="jax" if backend == "jax" else "torch",
            )
        )
        if backend == "jax":
            assert adapter.use_jax  # Do not validate only the fallback.
        h_star = torch.zeros(hidden_dim)
        x_input = torch.zeros(hidden_dim)
        actual_jacobian = adapter.compute_jacobian_at_state(h_star, x_input)
        assert actual_jacobian.shape == (hidden_dim, hidden_dim)
        torch.testing.assert_close(actual_jacobian, jacobian)
        spectrum = torch.linalg.eigvals(actual_jacobian)
        if mode.endswith("complex"):
            assert spectrum.imag.abs().max() > 0.7
        assert bool((spectrum.abs() > 1.0 + 1e-6).any()) == (
            not expected_attractor
        )
        with caplog.at_level(logging.WARNING):
            is_attractor, max_residual, monotonic = adapter.test_attractor_convergence(
                h_star, x_input, perturbation_magnitude=0.01,
                convergence_radius=0.02, K=50, n_directions=3,
            )

    assert is_attractor is expected_attractor
    assert np.isfinite(max_residual)
    # These sampled rollouts converge even for the saddle: they are not proof
    # of stability. A loose radius also admits the weakly expanding rollout.
    assert monotonic
    if expected_attractor:
        assert "unstable eigenmode" not in caplog.text
        assert 0.0 < max_residual < 0.02
    else:
        assert "unstable eigenmode" in caplog.text
        assert "|lambda|=" in caplog.text


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
