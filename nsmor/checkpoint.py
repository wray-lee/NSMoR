"""
Deterministic checkpoint management for NSMoR training.

Provides :func:`save_checkpoint` and :func:`load_checkpoint` that
persist and restore all state needed for exact training resumption:
model weights, optimizer state, epoch, loss, RNG state, and the
experiment configuration dictionary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from nsmor.config import PIPELINE_SEMANTICS_VERSION


# ═══════════════════════════════════════════════════════════════
# Provenance guard (Round-2 CRITICAL-A)
# ═══════════════════════════════════════════════════════════════

def _require_pipeline_version(
    found: Optional[str],
    artifact_description: str,
) -> None:
    """
    Reject artifacts produced under a different (or unknown) pipeline
    semantics version.

    A pre-2.0 checkpoint interprets LIF time constants in FRAME units;
    loading it under the ms-semantics code silently runs a different
    biophysical system.  A pre-2.0 dataset carries same-sample-leaked
    MCMC priors and np.max-based labels.  Both failure modes are silent
    at load time, so the only safe behaviour is to refuse the artifact.

    Args:
        found: The version string stored in the artifact (may be None).
        artifact_description: Human-readable identifier for error text.

    Raises:
        RuntimeError: If the version is missing or mismatched.
    """
    if found is None:
        raise RuntimeError(
            f"{artifact_description} has no 'pipeline_semantics_version' "
            f"stamp: it was produced by pre-2.0 code whose scientific "
            f"semantics (tau units, labeling criteria, MCMC prior "
            f"generation) are INCOMPATIBLE with the current pipeline. "
            f"Re-run data preparation / training with the current code "
            f"before using this artifact."
        )
    if str(found) != PIPELINE_SEMANTICS_VERSION:
        raise RuntimeError(
            f"{artifact_description} was produced under pipeline "
            f"semantics v{found}, but the running code implements "
            f"v{PIPELINE_SEMANTICS_VERSION}. Refusing to mix semantics. "
            f"Regenerate the artifact with the current pipeline."
        )


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
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
) -> Path:
    """
    Save a deterministic training checkpoint.

    The checkpoint dictionary contains:

    * ``model_state_dict`` — full model parameters and buffers
    * ``optimizer_state_dict`` — optimizer momentum / variance buffers
    * ``scheduler_state_dict`` — LR scheduler state (if provided)
    * ``epoch`` — current epoch index (0-based)
    * ``loss`` — loss value at the time of saving
    * ``train_loss`` — training loss (if provided)
    * ``val_loss`` — validation loss (if provided)
    * ``rng_state`` — ``torch.get_rng_state()`` for deterministic resumption
    * ``cuda_rng_state`` — ``torch.cuda.get_rng_state_all()`` if CUDA is
      available, so GPU-side stochasticity is also restored
    * ``config`` — the parsed experiment configuration dict

    Args:
        model: The model to checkpoint.
        optimizer: The optimizer whose state to persist.
        epoch: Current epoch number.
        loss: Current loss value (legacy; used for backward compat).
        config: Experiment configuration dictionary.
        path: File path for the checkpoint (typically ``.pt``).
        scheduler: Optional LR scheduler.
        train_loss: Training loss for this epoch (optional).
        val_loss: Validation loss for this epoch (optional).

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
        # Round-2 CRITICAL-A fix: provenance stamp.  Loaders reject
        # checkpoints lacking this key or carrying a different
        # semantics version — a pre-2.0 checkpoint interprets its
        # time constants in FRAME units and would silently run a
        # different biophysical system under the new code.
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
    }

    if train_loss is not None:
        state["train_loss"] = train_loss
    if val_loss is not None:
        state["val_loss"] = val_loss

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()

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
        ``epoch``, ``loss``, and ``config``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # ── Provenance check (Round-2 CRITICAL-A) ──
    _require_pipeline_version(
        checkpoint.get("pipeline_semantics_version"),
        f"checkpoint {path}",
    )

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
        rng_state = checkpoint["rng_state"]
        if not isinstance(rng_state, torch.ByteTensor):
            rng_state = rng_state.cpu().to(torch.uint8)
        torch.set_rng_state(rng_state)

    if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
        cuda_rng_states = checkpoint["cuda_rng_state"]
        cuda_rng_states = [
            s.cpu().to(torch.uint8) if not isinstance(s, torch.ByteTensor) else s
            for s in cuda_rng_states
        ]
        torch.cuda.set_rng_state_all(cuda_rng_states)

    return checkpoint
