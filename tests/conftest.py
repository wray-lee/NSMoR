"""
Pytest fixtures for NSMoR test suite.

Provides synthetic tensor fixtures for unit testing loss functions,
model forward passes, and analysis tools without requiring real data.

Shape legend
------------
    B  = batch_size (default 4)
    T  = seq_len    (default 100)
    H  = hidden_dim (default 32)
    D  = sensory_dim (default 4)
    M  = mcmc_dim    (default 4)
"""

from __future__ import annotations

import pytest
import torch
from typing import Tuple


# ═══════════════════════════════════════════════════════════════
# Dimension fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def batch_size() -> int:
    """Default batch size."""
    return 4


@pytest.fixture
def seq_len() -> int:
    """Default sequence length."""
    return 100


@pytest.fixture
def hidden_dim() -> int:
    """Default hidden dimension."""
    return 32


@pytest.fixture
def sensory_dim() -> int:
    """Sensory feature dimension."""
    return 4


@pytest.fixture
def mcmc_dim() -> int:
    """MCMC prior dimension."""
    return 4


# ═══════════════════════════════════════════════════════════════
# Tensor fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def x_batch(batch_size: int, seq_len: int) -> torch.Tensor:
    """
    Synthetic input feature tensor.

    Shape: ``(B, T, 8)`` — 4 sensory features + 4 MCMC priors.
    """
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, 8)

    # Ensure MCMC priors sum to 1 (indices 4-7)
    mcmc = x[:, :, 4:8]
    mcmc = torch.softmax(mcmc, dim=-1)
    x[:, :, 4:8] = mcmc

    return x


@pytest.fixture
def y_batch(batch_size: int, seq_len: int) -> torch.Tensor:
    """
    Synthetic ground truth target tensor.

    Shape: ``(B, T)`` — continuous output values.
    """
    torch.manual_seed(43)
    return torch.randn(batch_size, seq_len)


@pytest.fixture
def lengths(batch_size: int, seq_len: int) -> torch.Tensor:
    """
    Synthetic sequence lengths with variable padding.

    Shape: ``(B,)`` — true sequence lengths (not padded).
    """
    # Create variable lengths: [T, T-10, T-25, T-50]
    lengths = torch.tensor(
        [seq_len, seq_len - 10, seq_len - 25, seq_len - 50],
        dtype=torch.int64,
    )
    return lengths


@pytest.fixture
def g_gru(batch_size: int, seq_len: int) -> torch.Tensor:
    """
    Synthetic GRU routing gate values.

    Shape: ``(B, T, 1)`` — values in [0, 1] (from softmax).
    """
    torch.manual_seed(44)
    return torch.rand(batch_size, seq_len, 1)


@pytest.fixture
def model_inputs(
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    lengths: torch.Tensor,
    g_gru: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined model inputs as a tuple.

    Returns:
        ``(x_batch, y_batch, lengths, g_gru)``
    """
    return x_batch, y_batch, lengths, g_gru


# ═══════════════════════════════════════════════════════════════
# Device fixture
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def device() -> torch.device:
    """Default device (GPU(CPU FALLBACK) for deterministic testing)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
# Padding mask fixture
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def padding_mask(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Boolean padding mask derived from lengths.

    Shape: ``(B, T)`` — True for valid positions, False for padding.

    Uses the canonical construction: ``torch.arange(T) < lengths.unsqueeze(1)``
    """
    arange_t = torch.arange(seq_len)
    return arange_t.unsqueeze(0) < lengths.unsqueeze(1)
