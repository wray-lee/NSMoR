"""Incremental graph-memory budget for opt-in CAMoR-L2 workloads."""

from __future__ import annotations

import math
from typing import Any, Dict

# Conservative saved-slot coefficients calibrated against synthetic CPU
# saved-tensor/profiler/RSS measurements. K input/output states remain live
# under block checkpointing, so neither estimate is constant in K.
_GRAPH_SLOTS = {False: 14, True: 10}


def require_memory_budget(
    *,
    batch: int,
    timesteps: int,
    hidden_dim: int,
    K: int,
    grad_checkpointing: bool,
    budget_mb: float = 8192.0,
    parameter_bytes: int = 0,
) -> Dict[str, Any]:
    """Reject estimated incremental activation/optimizer memory over budget.

    The estimate includes c*N*K*H*4 graph bytes, six N*H working slots, the
    float64 pmf, and four parameter slots (parameters, gradients, Adam moments).
    Budget units are MiB of incremental memory, excluding the imported process.
    Actual RSS/profiler measurements remain mandatory for a new workload.

    Args:
        batch: Maximum batch size.
        timesteps: Maximum valid sequence length.
        hidden_dim: Recursion block width.
        K: Maximum recursion steps.
        grad_checkpointing: Whether block activations are recomputed.
        budget_mb: Incremental memory budget in MiB.
        parameter_bytes: Model parameter storage in bytes.

    Returns:
        The formula, coefficient, estimate and budget for provenance.

    Raises:
        ValueError: If dimensions/budget are invalid or the estimate is too large.
    """
    for name, value in (("batch", batch), ("timesteps", timesteps),
                        ("hidden_dim", hidden_dim), ("K", K)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive int")
    if not math.isfinite(budget_mb) or budget_mb <= 0:
        raise ValueError("memory budget must be finite and positive")
    slots = _GRAPH_SLOTS[bool(grad_checkpointing)]
    row_bytes = batch * timesteps * hidden_dim * 4
    estimated = (
        slots * K * row_bytes + 6 * row_bytes
        + batch * timesteps * K * 8 + 4 * parameter_bytes
    )
    report = {
        "estimated_bytes": int(estimated),
        "graph_slots_c": slots,
        "budget_mb": float(budget_mb),
        "formula": "c*N*K*H*4 + 6*N*H*4 + N*K*8 + 4*parameter_bytes",
        "grad_checkpointing": bool(grad_checkpointing),
    }
    if estimated > budget_mb * 1024**2:
        raise ValueError(
            f"Time-consuming activation estimate {estimated / 1024**2:.1f} "
            f"MiB exceeds memory budget {budget_mb:.1f} MiB; use K=8 or "
            "--time_consuming_grad_checkpointing, or increase the declared "
            "budget after measurement. No batch/precision changes are made."
        )
    return report
