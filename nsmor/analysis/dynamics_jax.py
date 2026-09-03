"""
JAX-Accelerated Dynamical Systems Adapter for NSMoR GRU Pathway.

Provides :class:`FixedPointAdapterJAX` for high-throughput fixed-point and
Jacobian analysis using JAX automatic differentiation (forward-mode ``jacfwd``)
and vectorization (``vmap``).

Key Improvements over PyTorch autograd:
  1. Vectorized Jacobian computation: compiles a single XLA kernel that computes
     all (N, H, H) Jacobians in parallel without looping over H dimensions or
     retaining backward graphs.
  2. Integrated eigenspectrum calculation: batches eigenvalue decomposition
     alongside Jacobian evaluation.
  3. Fused attractor convergence rollout: simulates K-step perturbation trajectories
     via ``lax.scan`` and vectorized across perturbation directions.

Falls back gracefully to PyTorch :class:`FixedPointAdapter` if JAX is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import jax
    import jax.numpy as jnp
    import jax.lax as lax
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    lax = None
    JAX_AVAILABLE = False

from nsmor.analysis.dynamics import FixedPointAdapter
from nsmor.model_nsmor_core import NSMoRCore

logger = logging.getLogger(__name__)


def _to_numpy(tensor: Any) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    if JAX_AVAILABLE and isinstance(tensor, jnp.ndarray):
        return np.asarray(tensor)
    return np.asarray(tensor)


def _to_torch(array: Any, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        res = array
    else:
        np_arr = np.array(array, copy=True)
        res = torch.from_numpy(np_arr)
    if device is not None:
        res = res.to(device)
    if dtype is not None and res.dtype != dtype:
        res = res.to(dtype)
    return res


# ===============================================================
# Pure Functional JAX GRU Cell & Jacobian Kernels
# ===============================================================

if JAX_AVAILABLE:

    def _gru_cell_step(
        h: jnp.ndarray,
        x: jnp.ndarray,
        w_ih: jnp.ndarray,
        w_hh: jnp.ndarray,
        b_ih: jnp.ndarray,
        b_hh: jnp.ndarray,
    ) -> jnp.ndarray:
        """Single-step GRU forward pass matching PyTorch nn.GRU conventions."""
        gi = x @ w_ih.T + b_ih
        gh = h @ w_hh.T + b_hh
        gi_r, gi_z, gi_n = jnp.split(gi, 3, axis=-1)
        gh_r, gh_z, gh_n = jnp.split(gh, 3, axis=-1)
        r_gate = jax.nn.sigmoid(gi_r + gh_r)
        z_gate = jax.nn.sigmoid(gi_z + gh_z)
        n_gate = jnp.tanh(gi_n + r_gate * gh_n)
        h_next = (1.0 - z_gate) * n_gate + z_gate * h
        return h_next


    def _build_jacobian_kernels(w_ih: jnp.ndarray, w_hh: jnp.ndarray, b_ih: jnp.ndarray, b_hh: jnp.ndarray):
        """Build and JIT-compile forward-mode Jacobian and eigenvalue kernels."""

        def step_fn(h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            return _gru_cell_step(h, x, w_ih, w_hh, b_ih, b_hh)

        # Forward-mode Jacobian wrt h: (H,) -> (H, H)
        jac_single = jax.jacfwd(step_fn, argnums=0)

        # Vectorize across batch dimension: (N, H), (N, D) -> (N, H, H)
        jac_batch = jax.vmap(jac_single, in_axes=(0, 0))

        @jax.jit
        def compute_jac_batch_jit(h_batch: jnp.ndarray, x_batch: jnp.ndarray) -> jnp.ndarray:
            return jac_batch(h_batch, x_batch)

        @jax.jit
        def compute_jac_and_eigvals_jit(h_batch: jnp.ndarray, x_batch: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            J = jac_batch(h_batch, x_batch)
            eigs = jnp.linalg.eigvals(J)
            return J, eigs

        @jax.jit
        def rollout_k_steps_jit(h_init: jnp.ndarray, x_fixed: jnp.ndarray, K: int) -> jnp.ndarray:
            """Roll out GRU for K steps under constant input x_fixed."""
            def scan_body(h_prev, _):
                h_curr = step_fn(h_prev, x_fixed)
                return h_curr, h_curr

            _, trajectory = lax.scan(scan_body, h_init, None, length=K)
            return trajectory  # (K, H)

        return step_fn, compute_jac_batch_jit, compute_jac_and_eigvals_jit, rollout_k_steps_jit


# ===============================================================
# FixedPointAdapterJAX Class
# ===============================================================

class FixedPointAdapterJAX:
    """
    Adapter for extracting GRU hidden states and computing Jacobians
    accelerated with JAX forward-mode AD and XLA compilation.

    Maintains identical public method signatures and behaviors as
    :class:`FixedPointAdapter` in ``nsmor/analysis/dynamics.py``.
    """

    def __init__(
        self,
        model: NSMoRCore,
        device: Optional[torch.device] = None,
        backend: str = "jax",
    ) -> None:
        """
        Args:
            model: Trained NSMoRCore model instance.
            device: Device to run computations on.
            backend: "jax" for JAX acceleration, "torch" for PyTorch autograd.
        """
        self.model = model
        self.model.eval()

        if device is None:
            device = next(model.parameters()).device
        self.device = device
        self.hidden_dim = model.hidden_dim

        self.use_jax = (backend == "jax") and JAX_AVAILABLE
        self._pytorch_adapter = FixedPointAdapter(model, device=device)

        if self.use_jax:
            gru = model.backend.gru_unit.gru
            self._w_ih = jnp.array(_to_numpy(gru.weight_ih_l0), dtype=jnp.float32)
            self._w_hh = jnp.array(_to_numpy(gru.weight_hh_l0), dtype=jnp.float32)
            self._b_ih = jnp.array(_to_numpy(gru.bias_ih_l0), dtype=jnp.float32)
            self._b_hh = jnp.array(_to_numpy(gru.bias_hh_l0), dtype=jnp.float32)

            (
                self._step_fn,
                self._compute_jac_batch_jit,
                self._compute_jac_and_eigvals_jit,
                self._rollout_k_steps_jit,
            ) = _build_jacobian_kernels(self._w_ih, self._w_hh, self._b_ih, self._b_hh)
        else:
            if backend == "jax" and not JAX_AVAILABLE:
                logger.warning("JAX not installed. Falling back to PyTorch FixedPointAdapter.")
            self._step_fn = None
            self._compute_jac_batch_jit = None
            self._compute_jac_and_eigvals_jit = None
            self._rollout_k_steps_jit = None

    @torch.no_grad()
    def extract_gru_states(self, dataloader: DataLoader) -> List[torch.Tensor]:
        """Extract un-padded GRU hidden-state trajectories from dataloader."""
        return self._pytorch_adapter.extract_gru_states(dataloader)

    def compute_jacobian_at_state(
        self,
        h_t: Union[torch.Tensor, np.ndarray, Any],
        x_t: Union[torch.Tensor, np.ndarray, Any],
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute the Jacobian ∂h_{t+1} / ∂h_t at a single state.

        Args:
            h_t: (H,) or (1, H) hidden state.
            x_t: (D,) or (1, D) input encoding.

        Returns:
            (H, H) Jacobian matrix.
        """
        is_torch = isinstance(h_t, torch.Tensor)
        H = self.hidden_dim

        if not self.use_jax:
            h_t_torch = _to_torch(h_t, device=self.device)
            x_t_torch = _to_torch(x_t, device=self.device)
            return self._pytorch_adapter.compute_jacobian_at_state(h_t_torch, x_t_torch)

        # Format input shapes
        h_arr = _to_numpy(h_t).reshape(1, H)
        x_arr = _to_numpy(x_t).reshape(1, -1)
        assert x_arr.shape[1] == H, f"x_t dim {x_arr.shape[1]} != H={H}"

        h_jax = jnp.array(h_arr, dtype=jnp.float32)
        x_jax = jnp.array(x_arr, dtype=jnp.float32)

        J_jax = self._compute_jac_batch_jit(h_jax, x_jax)[0]
        assert J_jax.shape == (H, H), f"Jacobian shape {J_jax.shape} != ({H}, {H})"

        if is_torch:
            return _to_torch(J_jax, device=self.device)
        return np.array(J_jax, copy=True)

    def compute_jacobian_batch(
        self,
        h_states: Union[torch.Tensor, np.ndarray, Any],
        x_inputs: Union[torch.Tensor, np.ndarray, Any],
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute Jacobians for a batch of hidden states in parallel.

        Args:
            h_states: (N, H) batch of hidden states.
            x_inputs: (N, H) corresponding inputs.

        Returns:
            (N, H, H) Jacobian tensor.
        """
        is_torch = isinstance(h_states, torch.Tensor)
        N, H = h_states.shape
        assert H == self.hidden_dim, f"h_states hidden dim {H} != {self.hidden_dim}"
        assert x_inputs.shape[0] == N, f"x_inputs batch {x_inputs.shape[0]} != {N}"

        if not self.use_jax:
            h_torch = _to_torch(h_states, device=self.device)
            x_torch = _to_torch(x_inputs, device=self.device)
            return self._pytorch_adapter.compute_jacobian_batch(h_torch, x_torch)

        h_jax = jnp.array(_to_numpy(h_states), dtype=jnp.float32)
        x_jax = jnp.array(_to_numpy(x_inputs), dtype=jnp.float32)

        J_batch = self._compute_jac_batch_jit(h_jax, x_jax)
        assert J_batch.shape == (N, H, H), f"J_batch shape {J_batch.shape} != ({N}, {H}, {H})"

        if is_torch:
            return _to_torch(J_batch, device=self.device)
        return np.array(J_batch, copy=True)

    def compute_eigenvalues_batch(
        self,
        h_states: Union[torch.Tensor, np.ndarray, Any],
        x_inputs: Union[torch.Tensor, np.ndarray, Any],
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        """
        Compute Jacobians and eigenvalues for a batch of states in one compiled pass.

        Args:
            h_states: (N, H) batch of states.
            x_inputs: (N, H) batch of inputs.

        Returns:
            (J_batch (N, H, H), eigvals (N, H))
        """
        is_torch = isinstance(h_states, torch.Tensor)
        N, H = h_states.shape

        if not self.use_jax:
            J_batch = self.compute_jacobian_batch(h_states, x_inputs)
            eigvals = torch.linalg.eigvals(J_batch)
            return J_batch, eigvals

        h_jax = jnp.array(_to_numpy(h_states), dtype=jnp.float32)
        x_jax = jnp.array(_to_numpy(x_inputs), dtype=jnp.float32)

        J_jax, eigvals_jax = self._compute_jac_and_eigvals_jit(h_jax, x_jax)

        if is_torch:
            return _to_torch(J_jax, device=self.device), _to_torch(eigvals_jax, device=self.device, dtype=torch.complex64)
        return np.array(J_jax, copy=True), np.array(eigvals_jax, copy=True)

    def test_attractor_convergence(
        self,
        h_star: torch.Tensor,
        x_input: torch.Tensor,
        perturbation_magnitude: float = 0.01,
        convergence_radius: float = 0.01,
        K: int = 50,
        n_directions: int = 3,
    ) -> Tuple[bool, float, bool]:
        """
        Test whether candidate slow point h* converges as an attractor.

        Perturbs along the principal eigenvectors of the Jacobian at h*
        and evaluates trajectory convergence over K steps.

        Args:
            h_star: (H,) candidate fixed point.
            x_input: (H,) or (1, H) input sensory encoding.
            perturbation_magnitude: Radius of initial perturbation.
            convergence_radius: Tolerance threshold for convergence.
            K: Forward steps (calibrated to time constants).
            n_directions: Number of perturbation eigendirections.

        Returns:
            (is_attractor, max_residual, monotonic_convergence)
        """
        if not self.use_jax:
            return self._pytorch_adapter.test_attractor_convergence(
                h_star, x_input, perturbation_magnitude, convergence_radius, K, n_directions
            )

        H = self.hidden_dim
        h_star_np = _to_numpy(h_star).reshape(H)
        x_in_np = _to_numpy(x_input).reshape(H)

        h_star_jax = jnp.array(h_star_np, dtype=jnp.float32)
        x_in_jax = jnp.array(x_in_np, dtype=jnp.float32)

        # 1. Verify h* is a slow/fixed point: ||GRU(h*, x) - h*|| <= convergence_radius
        h_next = self._step_fn(h_star_jax, x_in_jax)
        fixed_point_residual = float(jnp.linalg.norm(h_next - h_star_jax))
        if fixed_point_residual > convergence_radius:
            return False, fixed_point_residual, False

        # 2. Compute Jacobian and principal eigenvectors at h*
        J = self.compute_jacobian_at_state(h_star, x_input)  # (H, H)
        if isinstance(J, torch.Tensor):
            J_np = J.detach().cpu().numpy()
        else:
            J_np = np.asarray(J)

        eigvals, eigvecs = np.linalg.eig(J_np)
        eigval_mags = np.abs(eigvals)
        distance_to_boundary = np.abs(eigval_mags - 1.0)
        n_dirs = min(n_directions, H)
        top_k_indices = np.argsort(distance_to_boundary)[:n_dirs]

        max_residual = 0.0
        all_converged = True
        all_monotonic = True

        for idx in top_k_indices:
            eigvec = eigvecs[:, idx]
            ev = eigvals[idx]

            if abs(ev.imag) > 1e-6:
                directions_to_test = [eigvec.real, eigvec.imag]
            else:
                directions_to_test = [eigvec.real]

            for direction in directions_to_test:
                norm_d = np.linalg.norm(direction)
                if norm_d < 1e-12:
                    continue
                unit_dir = direction / (norm_d + 1e-8)

                # Perturbed initial state
                h_perturbed = h_star_np + perturbation_magnitude * unit_dir
                h_pert_jax = jnp.array(h_perturbed, dtype=jnp.float32)

                # Rollout K steps in JAX via lax.scan
                traj_k = self._rollout_k_steps_jit(h_pert_jax, x_in_jax, K)  # (K, H)
                residuals = np.linalg.norm(np.array(traj_k) - h_star_np, axis=-1)

                final_res = float(residuals[-1])
                max_residual = max(max_residual, final_res)
                if final_res > convergence_radius:
                    all_converged = False

                # Check monotonicity
                diffs = residuals[1:] - residuals[:-1]
                if np.any(diffs > residuals[:-1] * 0.01):
                    all_monotonic = False

        is_attractor = all_converged and (max_residual <= convergence_radius)
        return is_attractor, float(max_residual), all_monotonic

    @property
    def _gru_cell(self) -> nn.GRU:
        """Expose the PyTorch GRU cell for residual checks in analyze_jacobian."""
        return self._pytorch_adapter._gru_cell

    def compute_full_system_jacobian(self, *args: Any, **kwargs: Any) -> Any:
        """Full-system Jacobian is a PyTorch autograd path (LIF surrogate)."""
        return self._pytorch_adapter.compute_full_system_jacobian(*args, **kwargs)
