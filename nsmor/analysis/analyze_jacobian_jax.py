"""
JAX-Optimized Jacobian and Eigenvalue Spectrum Analysis Routines.

Provides batch acceleration functions for:
  - Vectorized Jacobian evaluation across epoch-specific candidate states
  - High-throughput eigenspectrum computation
  - JAX-accelerated quasi-fixed-point filtering
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from nsmor.analysis.dynamics_jax import FixedPointAdapterJAX, JAX_AVAILABLE
from nsmor.analysis.dynamics import FixedPointAdapter
from nsmor.model_nsmor_core import NSMoRCore

logger = logging.getLogger(__name__)


def create_jacobian_adapter(
    model: NSMoRCore,
    device: Optional[torch.device] = None,
    backend: str = "jax",
) -> Union[FixedPointAdapterJAX, FixedPointAdapter]:
    """
    Factory function to create a FixedPointAdapter with requested backend.

    Args:
        model: Trained NSMoRCore model.
        device: Computation device.
        backend: "jax" for JAX acceleration, "torch" for PyTorch autograd.

    Returns:
        Adapter instance.
    """
    if backend == "jax" and JAX_AVAILABLE:
        return FixedPointAdapterJAX(model, device=device, backend="jax")
    return FixedPointAdapter(model, device=device)


def batch_compute_eigenvalues(
    adapter: Union[FixedPointAdapterJAX, FixedPointAdapter],
    h_states: torch.Tensor,
    x_inputs: torch.Tensor,
    batch_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Jacobians and eigenvalues for a set of states in batches.

    Args:
        adapter: Adapter instance (FixedPointAdapterJAX or FixedPointAdapter).
        h_states: (N, H) states tensor.
        x_inputs: (N, H) inputs tensor.
        batch_size: Sub-batch size.

    Returns:
        (jacobians (N, H, H), eigenvalues (N, H))
    """
    N = h_states.shape[0]
    H = h_states.shape[1]
    device = h_states.device

    all_j = []
    all_eig = []

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        h_b = h_states[start_idx:end_idx]
        x_b = x_inputs[start_idx:end_idx]

        # jacfwd is ~20x vs autograd on this GPU; fused JAX eigvals is slower
        # than torch.linalg.eigvals on the same Jacobian (measured 2026-09-03).
        J_b = adapter.compute_jacobian_batch(h_b, x_b)
        if not isinstance(J_b, torch.Tensor):
            J_b = torch.from_numpy(np.asarray(J_b)).to(device)
        eig_b = torch.linalg.eigvals(J_b)

        all_j.append(J_b)
        all_eig.append(eig_b)

    if not all_j:
        return torch.zeros((0, H, H), device=device), torch.zeros((0, H), dtype=torch.complex64, device=device)

    return torch.cat(all_j, dim=0), torch.cat(all_eig, dim=0)
