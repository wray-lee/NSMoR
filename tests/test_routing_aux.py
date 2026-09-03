"""Unit tests for auxiliary routing loss (gate modality differentiation)."""

from __future__ import annotations

import pytest
import torch

from nsmor.loss import compute_routing_aux_loss


def test_routing_aux_loss_zero_when_separation_exceeds_margin():
    """Loss is zero when mean(g_wind) - mean(g_visual) >= margin."""
    g_lif = torch.tensor([
        [0.9, 0.9, 0.9],  # pure-wind trial
        [0.2, 0.2, 0.2],  # visual trial
    ], dtype=torch.float32)
    lengths = torch.tensor([3, 3], dtype=torch.int64)
    wind_only_mask = torch.tensor([True, False], dtype=torch.bool)

    loss = compute_routing_aux_loss(g_lif, lengths, wind_only_mask, margin=0.2)

    # mean(g_wind)=0.9, mean(g_visual)=0.2, separation=0.7 > 0.2
    assert loss.item() == pytest.approx(0.0), (
        f"Expected zero loss when separation exceeds margin, got {loss.item()}"
    )


def test_routing_aux_loss_positive_when_separation_below_margin():
    """Loss is positive when mean(g_wind) - mean(g_visual) < margin."""
    g_lif = torch.tensor([
        [0.5, 0.5, 0.5],  # pure-wind trial
        [0.4, 0.4, 0.4],  # visual trial
    ], dtype=torch.float32)
    lengths = torch.tensor([3, 3], dtype=torch.int64)
    wind_only_mask = torch.tensor([True, False], dtype=torch.bool)

    loss = compute_routing_aux_loss(g_lif, lengths, wind_only_mask, margin=0.2)

    # mean(g_wind)=0.5, mean(g_visual)=0.4, separation=0.1 < 0.2
    # loss = max(0, 0.2 - 0.1) = 0.1
    assert loss.item() == pytest.approx(0.1, abs=1e-5), (
        f"Expected loss=0.1 when separation=0.1 < margin=0.2, got {loss.item()}"
    )


def test_routing_aux_loss_zero_when_no_wind_trials():
    """Loss is zero when wind_only_mask contains no True values."""
    g_lif = torch.tensor([
        [0.5, 0.5, 0.5],
        [0.4, 0.4, 0.4],
    ], dtype=torch.float32)
    lengths = torch.tensor([3, 3], dtype=torch.int64)
    wind_only_mask = torch.tensor([False, False], dtype=torch.bool)

    loss = compute_routing_aux_loss(g_lif, lengths, wind_only_mask, margin=0.2)

    assert loss.item() == pytest.approx(0.0), (
        "Expected zero loss when wind group is empty"
    )


def test_routing_aux_loss_zero_when_no_visual_trials():
    """Loss is zero when all trials are pure-wind."""
    g_lif = torch.tensor([
        [0.9, 0.9, 0.9],
        [0.8, 0.8, 0.8],
    ], dtype=torch.float32)
    lengths = torch.tensor([3, 3], dtype=torch.int64)
    wind_only_mask = torch.tensor([True, True], dtype=torch.bool)

    loss = compute_routing_aux_loss(g_lif, lengths, wind_only_mask, margin=0.2)

    assert loss.item() == pytest.approx(0.0), (
        "Expected zero loss when visual group is empty"
    )


def test_routing_aux_loss_respects_lengths_mask():
    """Loss uses only valid timesteps per trial."""
    g_lif = torch.tensor([
        [0.9, 0.9, 0.0],  # pure-wind, length=2, last frame padded
        [0.2, 0.2, 1.0],  # visual, length=2, last frame padded
    ], dtype=torch.float32)
    lengths = torch.tensor([2, 2], dtype=torch.int64)
    wind_only_mask = torch.tensor([True, False], dtype=torch.bool)

    loss = compute_routing_aux_loss(g_lif, lengths, wind_only_mask, margin=0.2)

    # mean(g_wind) = mean([0.9, 0.9]) = 0.9
    # mean(g_visual) = mean([0.2, 0.2]) = 0.2
    # separation = 0.7 > 0.2 → loss=0
    assert loss.item() == pytest.approx(0.0), (
        "Padded frames should be masked out"
    )


def test_routing_aux_loss_accepts_3d_g_lif():
    """Loss handles g_lif with shape (B, T, 1)."""
    g_lif = torch.tensor([
        [[0.9], [0.9], [0.9]],
        [[0.2], [0.2], [0.2]],
    ], dtype=torch.float32)
    lengths = torch.tensor([3, 3], dtype=torch.int64)
    wind_only_mask = torch.tensor([True, False], dtype=torch.bool)

    loss = compute_routing_aux_loss(g_lif, lengths, wind_only_mask, margin=0.2)

    # Same as test_routing_aux_loss_zero_when_separation_exceeds_margin
    assert loss.item() == pytest.approx(0.0)


def test_routing_aux_default_margin_is_calibrated() -> None:
    """Default hinge is 0.024 (1.0× pooled std), not the old 0.2."""
    g_lif = torch.tensor(
        [
            [0.50, 0.50, 0.50],
            [0.49, 0.49, 0.49],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([3, 3], dtype=torch.int64)
    wind_only_mask = torch.tensor([True, False], dtype=torch.bool)

    loss = compute_routing_aux_loss(g_lif, lengths, wind_only_mask)

    # separation = 0.01; default margin = 0.024 → hinge = 0.014
    assert loss.item() == pytest.approx(0.014, abs=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
