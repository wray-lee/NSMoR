"""Unit tests for stimulus condition metadata wiring (Ticket #16)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nsmor.config import DEFAULT_FEATURE
from nsmor.nsmor_dataloader import NSMoRDataset, collate_with_metadata


def test_dataset_with_is_pure_wind_returns_3tuple():
    """Dataset with is_pure_wind returns (idx, X, Y) 3-tuple."""
    n_seqs = 4
    seqs = [
        (np.random.randn(100, 8).astype(np.float32), np.random.randn(100).astype(np.float32), 0)
        for _ in range(n_seqs)
    ]
    priors = np.random.rand(n_seqs, 4).astype(np.float32)
    priors /= priors.sum(axis=1, keepdims=True)

    is_pure_wind = np.array([True, False, True, False], dtype=bool)

    dataset = NSMoRDataset(
        sequences=seqs,
        mcmc_priors=priors,
        feature_config=DEFAULT_FEATURE,
        is_pure_wind=is_pure_wind,
    )

    # Should return 3-tuple (idx, X, Y)
    item = dataset[0]
    assert len(item) == 3, f"Expected 3-tuple, got {len(item)}-tuple"
    idx, X, Y = item
    assert idx == 0
    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)


def test_dataset_without_is_pure_wind_returns_2tuple():
    """Dataset without is_pure_wind returns (X, Y) 2-tuple (legacy)."""
    n_seqs = 4
    seqs = [
        (np.random.randn(100, 8).astype(np.float32), np.random.randn(100).astype(np.float32), 0)
        for _ in range(n_seqs)
    ]
    priors = np.random.rand(n_seqs, 4).astype(np.float32)
    priors /= priors.sum(axis=1, keepdims=True)

    dataset = NSMoRDataset(
        sequences=seqs,
        mcmc_priors=priors,
        feature_config=DEFAULT_FEATURE,
        is_pure_wind=None,  # No metadata
    )

    # Should return 2-tuple (X, Y)
    item = dataset[0]
    assert len(item) == 2, f"Expected 2-tuple, got {len(item)}-tuple"
    X, Y = item
    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)


def test_collate_with_metadata_returns_4tuple():
    """collate_with_metadata returns (X, Y, lengths, wind_only_mask) 4-tuple."""
    # Build batch of (idx, X, Y) tuples
    batch = [
        (0, torch.randn(100, 8), torch.randn(100)),
        (2, torch.randn(120, 8), torch.randn(120)),
        (3, torch.randn(90, 8), torch.randn(90)),
    ]
    is_pure_wind = np.array([True, False, True, False], dtype=bool)

    result = collate_with_metadata(batch, is_pure_wind=is_pure_wind)

    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"
    X_batch, Y_batch, lengths, wind_only_mask = result

    # Check shapes
    assert X_batch.shape == (3, 120, 8), f"X_batch shape {X_batch.shape}"
    assert Y_batch.shape == (3, 120), f"Y_batch shape {Y_batch.shape}"
    assert lengths.shape == (3,), f"lengths shape {lengths.shape}"
    assert wind_only_mask.shape == (3,), f"wind_only_mask shape {wind_only_mask.shape}"

    # Check metadata values (indices 0, 2, 3 → wind [True, True, False])
    expected_mask = torch.tensor([True, True, False], dtype=torch.bool)
    assert torch.equal(wind_only_mask, expected_mask), (
        f"wind_only_mask mismatch: {wind_only_mask} != {expected_mask}"
    )


def test_collate_with_metadata_without_metadata_returns_3tuple():
    """collate_with_metadata without metadata returns (X, Y, lengths) 3-tuple."""
    # Build batch of (idx, X, Y) tuples
    batch = [
        (0, torch.randn(100, 8), torch.randn(100)),
        (1, torch.randn(120, 8), torch.randn(120)),
    ]

    result = collate_with_metadata(batch, is_pure_wind=None)

    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
    X_batch, Y_batch, lengths = result

    assert X_batch.shape == (2, 120, 8)
    assert Y_batch.shape == (2, 120)
    assert lengths.shape == (2,)


def test_is_pure_wind_length_mismatch_raises():
    """is_pure_wind length mismatch raises ValueError."""
    n_seqs = 4
    seqs = [
        (np.random.randn(100, 8).astype(np.float32), np.random.randn(100).astype(np.float32), 0)
        for _ in range(n_seqs)
    ]
    priors = np.random.rand(n_seqs, 4).astype(np.float32)
    priors /= priors.sum(axis=1, keepdims=True)

    # Wrong length
    is_pure_wind = np.array([True, False], dtype=bool)

    with pytest.raises(ValueError, match="is_pure_wind length .* does not match"):
        NSMoRDataset(
            sequences=seqs,
            mcmc_priors=priors,
            is_pure_wind=is_pure_wind,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
