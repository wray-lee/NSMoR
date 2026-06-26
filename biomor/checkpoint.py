"""
Deterministic checkpoint management for BioMoR training.

Provides :func:`save_checkpoint` and :func:`load_checkpoint` that
persist and restore all state needed for exact training resumption:
model weights, optimizer state, epoch, loss, RNG state, and the
experiment configuration dictionary.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: Dict[str, Any],
    path: Union[str, Path],
    *,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a deterministic training checkpoint.

    The checkpoint dictionary contains:

    * ``model_state_dict`` — full model parameters and buffers
    * ``optimizer_state_dict`` — optimizer momentum / variance buffers
    * ``scheduler_state_dict`` — LR scheduler state (if provided)
    * ``epoch`` — current epoch index (0-based)
    * ``loss`` — loss value at the time of saving
    * ``rng_state`` — ``torch.get_rng_state()`` for deterministic resumption
    * ``cuda_rng_state`` — ``torch.cuda.get_rng_state_all()`` if CUDA is
      available, so GPU-side stochasticity is also restored
    * ``config`` — the parsed experiment configuration dict
    * ``extra_state`` — any additional caller-defined state

    Args:
        model: The model to checkpoint.
        optimizer: The optimizer whose state to persist.
        epoch: Current epoch number.
        loss: Current loss value.
        config: Experiment configuration dictionary.
        path: File path for the checkpoint (typically ``.pt``).
        scheduler: Optional LR scheduler.
        extra_state: Optional dict of additional state to persist.

    Returns:
        The resolved :class:`~pathlib.Path` of the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "rng_state": torch.get_rng_state(),
        "config": config,
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    if extra_state is not None:
        state["extra_state"] = extra_state

    torch.save(state, path)
    return path


# ═══════════════════════════════════════════════════════════════
# Load
# ═══════════════════════════════════════════════════════════════

def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    *,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore all deterministic state.

    Restores:

    * model parameters and buffers
    * optimizer state (if *optimizer* is provided)
    * LR scheduler state (if *scheduler* is provided)
    * ``torch`` RNG state (and CUDA RNG if available)

    Args:
        path: Path to the checkpoint file.
        model: Model whose state to restore.
        optimizer: Optimizer whose state to restore (optional).
        scheduler: LR scheduler whose state to restore (optional).
        map_location: Device mapping for ``torch.load``.

    Returns:
        The full checkpoint dictionary.  The caller can inspect
        ``epoch``, ``loss``, ``config``, and any ``extra_state``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # ── Restore model ──
    model.load_state_dict(checkpoint["model_state_dict"])

    # ── Restore optimizer ──
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # ── Restore scheduler ──
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # ── Restore RNG ──
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])

    if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    return checkpoint


# ═══════════════════════════════════════════════════════════════
# Convenience: resume from directory
# ═══════════════════════════════════════════════════════════════

def find_latest_checkpoint(directory: Union[str, Path]) -> Optional[Path]:
    """
    Find the most recently modified ``.pt`` file in *directory*.

    Returns ``None`` if no checkpoint files exist.
    """
    directory = Path(directory)
    checkpoints = sorted(
        directory.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None
