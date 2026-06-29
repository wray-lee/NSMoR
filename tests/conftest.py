"""
Shared test fixtures for NSMoR test suite.

Provides deterministic dimension constants and RNG seed protocol
to ensure cross-test consistency.
"""

import pytest
import torch


# ═══════════════════════════════════════════════════════════════
# Shared dimension fixtures (single source of truth)
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def batch_size() -> int:
    return 4

@pytest.fixture
def seq_len() -> int:
    return 100

@pytest.fixture
def hidden_dim() -> int:
    return 32

@pytest.fixture
def sensory_dim() -> int:
    return 4

@pytest.fixture
def mcmc_dim() -> int:
    return 4

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
# Deterministic RNG seed protocol
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _seed_rng():
    """Ensure deterministic RNG state for every test function."""
    torch.manual_seed(42)
    yield
