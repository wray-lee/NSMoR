"""
Strict unit tests for BioJointLoss.

Verifies:
1. Shape correctness of forward pass.
2. Padding mask correctness — padded regions do NOT contribute to loss.
3. Gradient isolation — padded regions receive zero gradients.
4. Router regularization effect — higher g_gru produces higher loss.
5. Reduction modes (mean vs sum).

These tests use synthetic fixtures from conftest.py.
"""

from __future__ import annotations

import pytest
import torch

from nsmor.loss import BioJointLoss


# ═══════════════════════════════════════════════════════════════
# Fixtures (inlined from deleted conftest.py)
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
def y_batch(batch_size: int, seq_len: int) -> torch.Tensor:
    """Synthetic ground truth target tensor. Shape: ``(B, T)``."""
    torch.manual_seed(43)
    return torch.randn(batch_size, seq_len)


@pytest.fixture
def lengths(batch_size: int, seq_len: int) -> torch.Tensor:
    """Synthetic sequence lengths with variable padding. Shape: ``(B,)``."""
    return torch.tensor(
        [seq_len, seq_len - 10, seq_len - 25, seq_len - 50],
        dtype=torch.int64,
    )


@pytest.fixture
def g_gru(batch_size: int, seq_len: int) -> torch.Tensor:
    """Synthetic GRU routing gate values. Shape: ``(B, T, 1)``."""
    torch.manual_seed(44)
    return torch.rand(batch_size, seq_len, 1)


# ═══════════════════════════════════════════════════════════════
# Test class
# ═══════════════════════════════════════════════════════════════

class TestBioJointLoss:
    """Strict tests for BioJointLoss padding mask and gradients."""

    # ── Shape tests ───────────────────────────────────────────

    def test_forward_returns_scalar(
        self,
        y_batch: torch.Tensor,
        lengths: torch.Tensor,
        g_gru: torch.Tensor,
    ) -> None:
        """Forward pass returns a scalar loss tensor."""
        criterion = BioJointLoss(reduction="mean")
        y_pred = torch.randn_like(y_batch)

        loss = criterion(y_pred, y_batch, lengths, g_gru, lambda_reg=0.01)

        assert loss.dim() == 0, f"Loss should be scalar, got {loss.dim()}-D"
        assert loss.isfinite(), f"Loss should be finite, got {loss}"

    def test_forward_shape_mismatch_raises(
        self,
        y_batch: torch.Tensor,
        lengths: torch.Tensor,
        g_gru: torch.Tensor,
    ) -> None:
        """Shape mismatch between y_pred and y_true raises AssertionError."""
        criterion = BioJointLoss()
        y_pred_bad = torch.randn(y_batch.shape[0] + 1, y_batch.shape[1])

        with pytest.raises(AssertionError):
            criterion(y_pred_bad, y_batch, lengths, g_gru, lambda_reg=0.01)

    def test_invalid_reduction_raises(self) -> None:
        """Invalid reduction mode raises ValueError."""
        with pytest.raises(ValueError, match="reduction must be"):
            BioJointLoss(reduction="invalid")

    # ── Padding mask tests ────────────────────────────────────

    def test_padded_region_does_not_contribute_to_loss(
        self,
        y_batch: torch.Tensor,
        lengths: torch.Tensor,
        g_gru: torch.Tensor,
    ) -> None:
        """
        CRITICAL: Changing values in padded regions must NOT change the loss.

        This is the core invariant: the mask ``torch.arange(T) < lengths.unsqueeze(1)``
        must exclude all padded positions from the loss computation.
        """
        criterion = BioJointLoss(reduction="mean")

        B, T = y_batch.shape

        # ── Baseline loss with zero predictions ──
        y_pred_zero = torch.zeros(B, T)
        loss_baseline = criterion(y_pred_zero, y_batch, lengths, g_gru, lambda_reg=0.0)

        # ── Modify padded regions with huge errors ──
        y_pred_padded = torch.zeros(B, T)
        y_true_padded = y_batch.clone()

        for i in range(B):
            # Padded region: indices >= lengths[i]
            padded_start = lengths[i].item()
            if padded_start < T:
                y_pred_padded[i, padded_start:] = 1000.0
                y_true_padded[i, padded_start:] = -1000.0

        loss_padded = criterion(y_pred_padded, y_true_padded, lengths, g_gru, lambda_reg=0.0)

        # ── Losses must be identical ──
        assert torch.allclose(loss_baseline, loss_padded, atol=1e-6), (
            f"Padded region contributed to loss! "
            f"baseline={loss_baseline.item():.6f}, "
            f"with_padding_errors={loss_padded.item():.6f}"
        )

    def test_padded_region_zero_gradient(
        self,
        y_batch: torch.Tensor,
        lengths: torch.Tensor,
        g_gru: torch.Tensor,
    ) -> None:
        """
        CRITICAL: Gradients in padded regions must be zero.

        The padding mask must block gradient flow from padded positions
        back to the predictions.
        """
        criterion = BioJointLoss(reduction="mean")
        B, T = y_batch.shape

        # Create predictions with requires_grad
        y_pred = torch.randn(B, T, requires_grad=True)

        loss = criterion(y_pred, y_batch, lengths, g_gru, lambda_reg=0.0)
        loss.backward()

        # Check gradients in padded regions
        assert y_pred.grad is not None, "Gradient should exist"

        for i in range(B):
            padded_start = lengths[i].item()
            if padded_start < T:
                padded_grad = y_pred.grad[i, padded_start:]
                assert torch.allclose(padded_grad, torch.zeros_like(padded_grad)), (
                    f"Sample {i}: padded region [T={padded_start}:] has non-zero gradient "
                    f"(norm={padded_grad.norm().item():.6f})"
                )

    def test_valid_region_has_nonzero_gradient(
        self,
        y_batch: torch.Tensor,
        lengths: torch.Tensor,
        g_gru: torch.Tensor,
    ) -> None:
        """Valid (non-padded) regions must receive gradients."""
        criterion = BioJointLoss(reduction="mean")
        B, T = y_batch.shape

        y_pred = torch.randn(B, T, requires_grad=True)
        loss = criterion(y_pred, y_batch, lengths, g_gru, lambda_reg=0.0)
        loss.backward()

        # At least one valid region should have non-zero gradient
        has_nonzero_grad = False
        for i in range(B):
            valid_len = lengths[i].item()
            valid_grad = y_pred.grad[i, :valid_len]
            if valid_grad.abs().sum() > 0:
                has_nonzero_grad = True
                break

        assert has_nonzero_grad, "No valid region received gradients"

    # ── Mask construction tests ───────────────────────────────

    def test_mask_construction(
        self,
        lengths: torch.Tensor,
        seq_len: int,
    ) -> None:
        """
        Verify the canonical mask construction matches manual expectation.

        mask[i, t] = (t < lengths[i])
        """
        B = lengths.shape[0]
        arange_t = torch.arange(seq_len)
        mask = (arange_t.unsqueeze(0) < lengths.unsqueeze(1))

        for i in range(B):
            valid_len = lengths[i].item()
            # First valid_len positions should be True
            assert mask[i, :valid_len].all(), (
                f"Sample {i}: mask[{i}, :{valid_len}] should be all True"
            )
            # Remaining positions should be False
            if valid_len < seq_len:
                assert not mask[i, valid_len:].any(), (
                    f"Sample {i}: mask[{i}, {valid_len}:] should be all False"
                )

    # ── Regularization tests ──────────────────────────────────

    def test_router_regularization_effect(
        self,
        y_batch: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        """Higher g_gru values produce higher regularization loss."""
        criterion = BioJointLoss(reduction="mean")
        B, T = y_batch.shape
        y_pred = torch.zeros(B, T)  # Zero predictions for clean MSE=0

        # g_gru = 0 (all LIF) → minimal regularization
        g_gru_low = torch.zeros(B, T, 1)
        loss_low = criterion(y_pred, y_batch, lengths, g_gru_low, lambda_reg=0.1)

        # g_gru = 1 (all GRU) → maximal regularization
        g_gru_high = torch.ones(B, T, 1)
        loss_high = criterion(y_pred, y_batch, lengths, g_gru_high, lambda_reg=0.1)

        assert loss_high > loss_low, (
            f"High g_gru should produce higher loss: "
            f"{loss_high.item():.6f} <= {loss_low.item():.6f}"
        )

    def test_lambda_reg_scaling(
        self,
        y_batch: torch.Tensor,
        lengths: torch.Tensor,
        g_gru: torch.Tensor,
    ) -> None:
        """Lambda_reg linearly scales the regularization term."""
        criterion = BioJointLoss(reduction="mean")
        B, T = y_batch.shape
        y_pred = torch.zeros(B, T)

        loss_1 = criterion(y_pred, y_batch, lengths, g_gru, lambda_reg=0.01)
        loss_2 = criterion(y_pred, y_batch, lengths, g_gru, lambda_reg=0.02)

        # The difference should be approximately lambda_reg * mean(g_gru)
        # Since MSE is the same for both, the delta is purely from regularization
        delta = loss_2 - loss_1
        expected_delta = 0.01 * g_gru.mean().item()

        assert torch.allclose(
            delta, torch.tensor(expected_delta), atol=1e-4,
        ), (
            f"Lambda scaling not linear: delta={delta.item():.6f}, "
            f"expected={expected_delta:.6f}"
        )

    # ── Reduction mode tests ──────────────────────────────────

    def test_mean_vs_sum_reduction(
        self,
        y_batch: torch.Tensor,
        lengths: torch.Tensor,
        g_gru: torch.Tensor,
    ) -> None:
        """Mean reduction divides by valid count; sum does not."""
        B, T = y_batch.shape
        y_pred = torch.randn(B, T)

        criterion_mean = BioJointLoss(reduction="mean")
        criterion_sum = BioJointLoss(reduction="sum")

        loss_mean = criterion_mean(y_pred, y_batch, lengths, g_gru, lambda_reg=0.0)
        loss_sum = criterion_sum(y_pred, y_batch, lengths, g_gru, lambda_reg=0.0)

        # Sum should be larger (not divided by N)
        total_valid = lengths.sum().item()
        if total_valid > 1:
            assert loss_sum > loss_mean, (
                f"Sum reduction ({loss_sum.item():.6f}) should be > "
                f"mean reduction ({loss_mean.item():.6f})"
            )

    # ── Edge case tests ───────────────────────────────────────

    def test_single_timestep(
        self,
        g_gru: torch.Tensor,
    ) -> None:
        """Works correctly with seq_len=1."""
        B = 2
        y_pred = torch.randn(B, 1)
        y_true = torch.randn(B, 1)
        lengths = torch.tensor([1, 1], dtype=torch.int64)
        g_gru_single = torch.rand(B, 1, 1)

        criterion = BioJointLoss()
        loss = criterion(y_pred, y_true, lengths, g_gru_single, lambda_reg=0.01)

        assert loss.isfinite()
        assert loss.dim() == 0

    def test_all_same_lengths(
        self,
        batch_size: int,
        seq_len: int,
    ) -> None:
        """Works when all sequences have the same length."""
        y_pred = torch.randn(batch_size, seq_len)
        y_true = torch.randn(batch_size, seq_len)
        lengths = torch.full((batch_size,), seq_len, dtype=torch.int64)
        g_gru = torch.rand(batch_size, seq_len, 1)

        criterion = BioJointLoss()
        loss = criterion(y_pred, y_true, lengths, g_gru, lambda_reg=0.01)

        assert loss.isfinite()
        assert loss.dim() == 0

    def test_zero_lambda_reg(
        self,
        y_batch: torch.Tensor,
        lengths: torch.Tensor,
        g_gru: torch.Tensor,
    ) -> None:
        """With lambda_reg=0, loss is pure masked MSE."""
        criterion = BioJointLoss(reduction="mean")
        B, T = y_batch.shape
        y_pred = torch.randn(B, T)

        loss_with_reg = criterion(y_pred, y_batch, lengths, g_gru, lambda_reg=0.01)
        loss_no_reg = criterion(y_pred, y_batch, lengths, g_gru, lambda_reg=0.0)

        assert loss_no_reg < loss_with_reg, (
            f"Zero lambda should give lower loss: "
            f"{loss_no_reg.item():.6f} >= {loss_with_reg.item():.6f}"
        )


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
