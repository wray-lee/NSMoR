"""Unit tests for loss_ext.py (Ticket #14 + #18)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nsmor.loss_ext import BioJointLossExt, compute_routing_aux_loss


class TestComputeRoutingAuxLoss:
    """Unit tests for compute_routing_aux_loss."""

    def test_zero_loss_when_separation_exceeds_margin(self) -> None:
        """Loss is zero when mean(g_wind) - mean(g_visual) >= margin."""
        B, T = 4, 100
        lengths = torch.full((B,), T, dtype=torch.long)

        # Wind trials have high g_lif, visual trials have low g_lif
        g_lif = torch.zeros(B, T)
        wind_only_mask = torch.tensor([True, True, False, False])
        g_lif[wind_only_mask, :] = 0.8  # Wind -> LIF dominant
        g_lif[~wind_only_mask, :] = 0.2  # Visual -> GRU dominant

        loss = compute_routing_aux_loss(g_lif, lengths, wind_only_mask, margin=0.2)

        # Separation = 0.8 - 0.2 = 0.6 > 0.2 → loss should be zero
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_when_separation_below_margin(self) -> None:
        """Loss > 0 when mean(g_wind) - mean(g_visual) < margin."""
        B, T = 4, 100
        lengths = torch.full((B,), T, dtype=torch.long)

        # Wind and visual trials have similar g_lif (collapse)
        g_lif = torch.zeros(B, T)
        wind_only_mask = torch.tensor([True, True, False, False])
        g_lif[wind_only_mask, :] = 0.45  # Collapsed gate
        g_lif[~wind_only_mask, :] = 0.40  # Collapsed gate

        loss = compute_routing_aux_loss(g_lif, lengths, wind_only_mask, margin=0.2)

        # Separation = 0.45 - 0.40 = 0.05 < 0.2
        # Hinge = 0.2 - 0.05 = 0.15
        expected = 0.15
        assert loss.item() == pytest.approx(expected, abs=1e-5)

    def test_zero_loss_when_wind_only_mask_all_false(self) -> None:
        """No wind trials → no penalty."""
        B, T = 4, 100
        lengths = torch.full((B,), T, dtype=torch.long)
        g_lif = torch.rand(B, T)
        wind_only_mask = torch.tensor([False, False, False, False])

        loss = compute_routing_aux_loss(g_lif, lengths, wind_only_mask, margin=0.2)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_zero_loss_when_wind_only_mask_all_true(self) -> None:
        """No visual trials → no penalty."""
        B, T = 4, 100
        lengths = torch.full((B,), T, dtype=torch.long)
        g_lif = torch.rand(B, T)
        wind_only_mask = torch.tensor([True, True, True, True])

        loss = compute_routing_aux_loss(g_lif, lengths, wind_only_mask, margin=0.2)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_respects_variable_lengths(self) -> None:
        """Aggregation only considers valid (non-padded) frames."""
        B, T = 2, 100
        lengths = torch.tensor([50, 80], dtype=torch.long)
        wind_only_mask = torch.tensor([True, False])

        g_lif = torch.zeros(B, T)
        # Wind trial (valid 0:50)
        g_lif[0, :50] = 0.9
        g_lif[0, 50:] = 0.0  # Padding (should be ignored)

        # Visual trial (valid 0:80)
        g_lif[1, :80] = 0.3
        g_lif[1, 80:] = 1.0  # Padding (should be ignored)

        loss = compute_routing_aux_loss(g_lif, lengths, wind_only_mask, margin=0.2)

        # Mean(wind) = 0.9, Mean(visual) = 0.3
        # Separation = 0.9 - 0.3 = 0.6 > 0.2 → zero loss
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_accepts_g_lif_shape_BT1(self) -> None:
        """Handles [B, T, 1] shape (router output format)."""
        B, T = 4, 100
        lengths = torch.full((B,), T, dtype=torch.long)
        g_lif = torch.rand(B, T, 1)
        wind_only_mask = torch.tensor([True, True, False, False])

        # Should not raise
        loss = compute_routing_aux_loss(g_lif, lengths, wind_only_mask, margin=0.2)
        assert loss.shape == ()  # Scalar

    def test_shape_assertion_failures(self) -> None:
        """Shape mismatches raise AssertionError."""
        B, T = 4, 100
        lengths = torch.full((B,), T, dtype=torch.long)
        g_lif = torch.rand(B, T)
        wind_only_mask = torch.tensor([True, True, False, False])

        # Mismatched batch size
        with pytest.raises(AssertionError):
            compute_routing_aux_loss(
                g_lif, torch.ones(3, dtype=torch.long) * T, wind_only_mask
            )

        # Mismatched mask size
        with pytest.raises(AssertionError):
            compute_routing_aux_loss(
                g_lif, lengths, torch.tensor([True, False, True])
            )

        # Invalid g_lif shape
        with pytest.raises(AssertionError):
            compute_routing_aux_loss(
                torch.rand(B, T, 2), lengths, wind_only_mask
            )


class TestBioJointLossExt:
    """Integration tests for BioJointLossExt wrapper."""

    def test_fallback_to_base_when_lambda_routing_aux_zero(self) -> None:
        """lambda_routing_aux=0 returns frozen base loss exactly."""
        from nsmor.loss import BioJointLoss

        B, T, H = 4, 100, 32
        y_pred = torch.randn(B, T)
        y_true = torch.randn(B, T)
        lengths = torch.full((B,), T, dtype=torch.long)
        g_gru = torch.rand(B, T, 1)

        base_loss = BioJointLoss()
        ext_loss = BioJointLossExt(base_loss)

        base_out = base_loss(
            y_pred, y_true, lengths, g_gru, lambda_reg=0.2, annealing_factor=1.0
        )
        ext_out = ext_loss(
            y_pred,
            y_true,
            lengths,
            g_gru,
            lambda_reg=0.2,
            lambda_routing_aux=0.0,
            annealing_factor=1.0,
        )

        assert torch.allclose(ext_out, base_out, atol=1e-6)

    def test_adds_routing_aux_when_lambda_positive(self) -> None:
        """lambda_routing_aux > 0 increases total loss."""
        from nsmor.loss import BioJointLoss

        B, T = 4, 100
        y_pred = torch.randn(B, T)
        y_true = torch.randn(B, T)
        lengths = torch.full((B,), T, dtype=torch.long)
        g_gru = torch.rand(B, T, 1)
        g_lif = torch.rand(B, T, 1)
        wind_only_mask = torch.tensor([True, True, False, False])

        # Induce separation gap below margin
        g_lif[wind_only_mask, :, 0] = 0.45
        g_lif[~wind_only_mask, :, 0] = 0.40

        base_loss = BioJointLoss()
        ext_loss = BioJointLossExt(base_loss)

        base_out = base_loss(
            y_pred, y_true, lengths, g_gru, lambda_reg=0.2, annealing_factor=1.0
        )
        ext_out = ext_loss(
            y_pred,
            y_true,
            lengths,
            g_gru,
            g_lif=g_lif,
            lambda_reg=0.2,
            lambda_routing_aux=0.5,
            wind_only_mask=wind_only_mask,
            annealing_factor=1.0,
        )

        assert ext_out.item() > base_out.item()

    def test_raises_when_g_lif_missing(self) -> None:
        """lambda_routing_aux > 0 requires g_lif."""
        from nsmor.loss import BioJointLoss

        B, T = 4, 100
        y_pred = torch.randn(B, T)
        y_true = torch.randn(B, T)
        lengths = torch.full((B,), T, dtype=torch.long)
        g_gru = torch.rand(B, T, 1)
        wind_only_mask = torch.tensor([True, True, False, False])

        base_loss = BioJointLoss()
        ext_loss = BioJointLossExt(base_loss)

        with pytest.raises(ValueError, match="g_lif required"):
            ext_loss(
                y_pred,
                y_true,
                lengths,
                g_gru,
                lambda_routing_aux=0.5,
                wind_only_mask=wind_only_mask,
            )

    def test_raises_when_wind_only_mask_missing(self) -> None:
        """lambda_routing_aux > 0 requires wind_only_mask."""
        from nsmor.loss import BioJointLoss

        B, T = 4, 100
        y_pred = torch.randn(B, T)
        y_true = torch.randn(B, T)
        lengths = torch.full((B,), T, dtype=torch.long)
        g_gru = torch.rand(B, T, 1)
        g_lif = torch.rand(B, T, 1)

        base_loss = BioJointLoss()
        ext_loss = BioJointLossExt(base_loss)

        with pytest.raises(ValueError, match="wind_only_mask required"):
            ext_loss(
                y_pred,
                y_true,
                lengths,
                g_gru,
                g_lif=g_lif,
                lambda_routing_aux=0.5,
            )

    def test_annealing_factor_scales_routing_aux(self) -> None:
        """Warmup annealing scales lambda_routing_aux."""
        from nsmor.loss import BioJointLoss

        B, T = 4, 100
        y_pred = torch.randn(B, T)
        y_true = torch.randn(B, T)
        lengths = torch.full((B,), T, dtype=torch.long)
        g_gru = torch.rand(B, T, 1)
        g_lif = torch.rand(B, T, 1)
        wind_only_mask = torch.tensor([True, True, False, False])

        # Induce positive routing_aux loss
        g_lif[wind_only_mask, :, 0] = 0.45
        g_lif[~wind_only_mask, :, 0] = 0.40

        base_loss = BioJointLoss()
        ext_loss = BioJointLossExt(base_loss)

        # Full annealing (annealing_factor=1.0)
        loss_full = ext_loss(
            y_pred,
            y_true,
            lengths,
            g_gru,
            g_lif=g_lif,
            lambda_reg=0.2,
            lambda_routing_aux=0.5,
            wind_only_mask=wind_only_mask,
            annealing_factor=1.0,
        )

        # Half annealing (annealing_factor=0.5)
        loss_half = ext_loss(
            y_pred,
            y_true,
            lengths,
            g_gru,
            g_lif=g_lif,
            lambda_reg=0.2,
            lambda_routing_aux=0.5,
            wind_only_mask=wind_only_mask,
            annealing_factor=0.5,
        )

        # The difference should be roughly half of the routing_aux component
        # (MSE and lambda_reg are constant)
        assert loss_half < loss_full
