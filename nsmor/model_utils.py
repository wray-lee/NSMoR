"""
Shared model utilities for NSMoR scripts.

Provides a single canonical implementation of model loading from
checkpoints, eliminating the 5-way duplication that existed across
``scripts/analyze_dynamics.py``, ``scripts/analyze_jacobian.py``,
``scripts/simulate_lesion.py``, ``scripts/simulate_autoregressive.py``,
and ``scripts/simulate_psychophysics.py``.

All scripts MUST use :func:`load_model_from_checkpoint` from this
module to guarantee that every biophysical parameter is faithfully
reconstructed from the saved config.

CF5 Fix: Parameter defaults are extracted programmatically from
``NSMoRCore.__init__`` via ``inspect.signature``, eliminating the
risk of manual dict drifting out of sync with the constructor.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from nsmor.model_nsmor_core import NSMoRCore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1.  Parameter extraction from NSMoRCore.__init__ (CF5 fix)
# ═══════════════════════════════════════════════════════════════

def _get_param_defaults() -> Dict[str, Any]:
    """
    Programmatically extract all parameter names and default values
    from ``NSMoRCore.__init__`` via ``inspect.signature``.

    This guarantees that the parameter list is ALWAYS in sync with
    the actual constructor.  Adding a new parameter to NSMoRCore
    automatically makes it available here -- no manual dict update
    needed.

    Returns:
        Dict mapping parameter name to its default value.
        Parameters without defaults (i.e., positional-only) are
        excluded (there are none in NSMoRCore currently).
    """
    sig = inspect.signature(NSMoRCore.__init__)
    defaults: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is not inspect.Parameter.empty:
            defaults[name] = param.default
    return defaults


# Cached at module load time (computed once, not on every call)
_NS_MOR_PARAM_DEFAULTS: Dict[str, Any] = _get_param_defaults()


def _extract_model_params(
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract all NSMoRCore constructor parameters from a checkpoint
    config dict, filling in defaults for any missing keys.

    The default values are extracted programmatically from
    ``NSMoRCore.__init__`` via ``inspect.signature`` (CF5 fix),
    ensuring the parameter list never drifts out of sync.

    Args:
        config_dict: The ``"model"`` sub-dict from the checkpoint's
            ``"config"`` entry.  May contain extra keys (ignored)
            or be missing keys (filled with defaults).

    Returns:
        A dict suitable for ``NSMoRCore(**params)``.
    """
    params: Dict[str, Any] = {}
    for key, default in _NS_MOR_PARAM_DEFAULTS.items():
        params[key] = config_dict.get(key, default)
    return params


# ═══════════════════════════════════════════════════════════════
# 2.  Canonical model loader
# ═══════════════════════════════════════════════════════════════

def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> NSMoRCore:
    """
    Load a trained NSMoRCore model from a checkpoint file.

    This is the CANONICAL loading function for all NSMoR analysis
    scripts.  It extracts every biophysical parameter from the
    checkpoint config, including those added after the initial
    release (refractory periods, STP, lateral inhibition, dendritic
    compartmentalization, neuromodulatory gain, sensory noise).

    Args:
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        device: Device to load the model onto.

    Returns:
        Loaded model in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )

    # Extract config from checkpoint
    config_dict = checkpoint.get("config", {})
    model_config = config_dict.get("model", {})

    # Build model with ALL saved parameters (fills defaults for missing)
    params = _extract_model_params(model_config)
    model = NSMoRCore(**params)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded: %s parameters, hidden_dim=%d",
        f"{param_count:,}", model.hidden_dim,
    )

    return model


# ═══════════════════════════════════════════════════════════════
# 3.  Tensor shape validation helper
# ═══════════════════════════════════════════════════════════════

def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    name: str,
) -> None:
    """
    Assert that a tensor has the expected shape.

    Args:
        tensor: The tensor to validate.
        expected_shape: Expected shape tuple.  Use ``-1`` for
            dimensions that can be any value.
        name: Human-readable name for the tensor.

    Raises:
        AssertionError: If the shape does not match.
    """
    actual = tuple(tensor.shape)
    if len(actual) != len(expected_shape):
        raise AssertionError(
            f"{name}: expected {len(expected_shape)}-D tensor with "
            f"shape {expected_shape}, got {len(actual)}-D with shape {actual}"
        )
    for i, (a, e) in enumerate(zip(actual, expected_shape)):
        if e != -1 and a != e:
            raise AssertionError(
                f"{name}: dimension {i} expected {e}, got {actual}. "
                f"Full expected shape: {expected_shape}, actual: {actual}"
            )
