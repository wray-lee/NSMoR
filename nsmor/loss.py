"""
Bio-Constrained Joint Loss for NSMoR training.

Provides :class:`BioJointLoss`, a custom ``nn.Module`` that combines:

1. **Masked MSE** — Mean Squared Error computed only over valid
   (non-padded) time-steps, determined by true sequence lengths.
2. **Biological Router Regularization** — A penalty term that prevents
   the MoR Router from collapsing onto the higher-capacity GRU pathway
   when the LIF pathway is biologically appropriate (e.g., during
   sudden, high-reliability stimuli).

Shape legend
------------
    B  = batch_size
    T  = seq_len (padded)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BioJointLoss(nn.Module):
    """
    Bio-constrained joint loss with masked MSE and conditional router
    regularization.

    The loss is computed as:

    .. math::

        \\mathcal{L} = \\text{MaskedMSE}(y_{\\text{pred}},\\, y_{\\text{true}})
        + \\lambda \\cdot \\frac{1}{N} \\sum_{b,t}
          g_{\\text{gru}}(b,t) \\cdot \\text{mask}(b,t)
          \\cdot \\text{jerk\\_mask}(b,t)

    where :math:`N = \\sum_b \\text{lengths}(b)` is the total number of
    valid time-steps across the batch, and :math:`g_{\\text{gru}}` is the
    GRU routing gate (index 1 of the routing gate vector).

    When a ``jerk_mask`` is provided, the :math:`\\lambda` regularization
    penalty is applied **only** to time-steps where the sensory input's
    absolute first-order derivative exceeds a threshold (i.e., during
    sudden-change / transient intervals).  During steady-state intervals
    the GRU pathway is free to track smoothly without penalty.

    Args:
        reduction: How to aggregate the MSE across valid timesteps.
            ``"mean"`` (default) divides by total valid count.
            ``"sum"`` sums without dividing.

    Example::

        criterion = BioJointLoss()
        # Compute jerk mask from sensory velocity (B, T)
        velocity = sensory[:, :, 2]  # e.g., velocity channel
        jerk_mask = (velocity.diff(dim=1).abs() > threshold).float()
        # Pad to match T (first frame has no diff)
        jerk_mask = torch.cat([torch.zeros(B, 1), jerk_mask], dim=1)

        loss = criterion(
            y_pred=predictions,      # (B, T)
            y_true=targets,          # (B, T)
            lengths=lengths,         # (B,)
            g_gru=g_gru,             # (B, T, 1)
            lambda_reg=0.01,
            jerk_mask=jerk_mask,     # (B, T) — optional
        )
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"reduction must be 'mean' or 'sum', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        lengths: torch.Tensor,
        g_gru: torch.Tensor,
        lambda_reg: float = 0.01,
        jerk_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the bio-constrained joint loss.

        Args:
            y_pred: ``(B, T)`` — predicted output from the model.
            y_true: ``(B, T)`` — ground truth target.
            lengths: ``(B,)`` — true (unpadded) sequence lengths.
            g_gru: ``(B, T, 1)`` — GRU routing gate from model internals.
            lambda_reg: Weight for the router regularization term.
            jerk_mask: ``(B, T)`` — optional sudden-change mask.
                Values are 1.0 where the sensory input's absolute
                first-order derivative exceeds a threshold, 0.0
                otherwise.  When provided, the :math:`\\lambda`
                penalty applies **only** to those transient
                time-steps; steady-state steps receive zero penalty.
                When ``None``, the penalty applies uniformly to all
                valid (non-padded) steps (backward compatible).

        Returns:
            Scalar loss tensor.

        Raises:
            AssertionError: If tensor shapes are inconsistent.
        """
        # ── Shape assertions ──────────────────────────────────
        assert y_pred.dim() == 2, (
            f"y_pred must be 2-D (B, T), got {y_pred.dim()}-D"
        )
        assert y_true.shape == y_pred.shape, (
            f"y_true shape {tuple(y_true.shape)} != y_pred shape {tuple(y_pred.shape)}"
        )
        assert lengths.dim() == 1, (
            f"lengths must be 1-D (B,), got {lengths.dim()}-D"
        )
        assert g_gru.dim() == 3, (
            f"g_gru must be 3-D (B, T, 1), got {g_gru.dim()}-D"
        )
        assert g_gru.shape[2] == 1, (
            f"g_gru last dim must be 1, got {g_gru.shape[2]}"
        )

        B, T = y_pred.shape

        assert lengths.shape == (B,), (
            f"lengths shape {tuple(lengths.shape)} != (B={B},)"
        )
        assert g_gru.shape == (B, T, 1), (
            f"g_gru shape {tuple(g_gru.shape)} != (B={B}, T={T}, 1)"
        )
        if jerk_mask is not None:
            assert jerk_mask.shape == (B, T), (
                f"jerk_mask shape {tuple(jerk_mask.shape)} != (B={B}, T={T})"
            )

        # ── Build padding mask ────────────────────────────────
        # mask[i, t] = 1 if t < lengths[i], else 0
        # Shape: (B, T)
        arange_t = torch.arange(T, device=y_pred.device)        # (T,)
        mask = (arange_t.unsqueeze(0) < lengths.unsqueeze(1))    # (B, T)
        mask = mask.float()                                       # (B, T)

        # ── Masked MSE ────────────────────────────────────────
        squared_errors = (y_pred - y_true) ** 2                  # (B, T)
        masked_errors = squared_errors * mask                     # (B, T)

        if self.reduction == "mean":
            total_valid = mask.sum().clamp(min=1.0)               # scalar
            mse_loss = masked_errors.sum() / total_valid           # scalar
        else:
            mse_loss = masked_errors.sum()                         # scalar

        # ── Router regularization (conditional on jerk_mask) ──
        # Squeeze last dim: (B, T, 1) → (B, T)
        g_gru_sq = g_gru.squeeze(-1)                              # (B, T)

        if jerk_mask is not None:
            # Penalize g_gru only on sudden-change (transient) time-steps
            reg_raw = (g_gru_sq * mask * jerk_mask).sum()          # scalar
            N = (mask * jerk_mask).sum().clamp(min=1.0)            # scalar
        else:
            # Backward compatible: penalize on all valid time-steps
            reg_raw = (g_gru_sq * mask).sum()                      # scalar
            N = mask.sum().clamp(min=1.0)                          # scalar

        reg_loss = lambda_reg * (reg_raw / N)                      # scalar

        # ── Total loss ────────────────────────────────────────
        total_loss = mse_loss + reg_loss                           # scalar

        return total_loss


# ═══════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════

def _test_bio_joint_loss() -> None:
    """
    Verify ``BioJointLoss`` produces correct shapes and masked computation.

    Run::

        python -m nsmor.loss
    """
    print("=" * 60)
    print("BioJointLoss smoke test")
    print("=" * 60)

    B, T = 4, 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_pred = torch.randn(B, T, device=device)
    y_true = torch.randn(B, T, device=device)
    lengths = torch.tensor([50, 40, 25, 10], dtype=torch.int64, device=device)
    g_gru = torch.rand(B, T, 1, device=device)  # (B, T, 1) in [0, 1]

    criterion = BioJointLoss(reduction="mean")

    # ── Forward pass ──
    loss = criterion(y_pred, y_true, lengths, g_gru, lambda_reg=0.01)
    assert loss.dim() == 0, f"Loss should be scalar, got {loss.dim()}-D"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    print(f"  loss (mean reduction): {loss.item():.6f}")

    # ── Sum reduction ──
    criterion_sum = BioJointLoss(reduction="sum")
    loss_sum = criterion_sum(y_pred, y_true, lengths, g_gru, lambda_reg=0.01)
    print(f"  loss (sum reduction):  {loss_sum.item():.6f}")

    # ── Masking correctness ──
    # Create a case where padded positions have huge errors
    y_pred_bad = torch.zeros(B, T, device=device)
    y_true_bad = torch.zeros(B, T, device=device)
    # Fill padded positions with large errors (should be masked out)
    y_pred_bad[3, 40:] = 1000.0  # padded region for sample 3 (length=10)
    y_true_bad[3, 40:] = -1000.0

    loss_masked = criterion(y_pred_bad, y_true_bad, lengths, g_gru, lambda_reg=0.0)
    assert loss_masked.item() == 0.0, (
        f"Loss should be 0 when non-padded predictions are correct, "
        f"got {loss_masked.item()}"
    )
    print("  masking correctness:   OK (padded errors ignored)")

    # ── Regularization effect ──
    g_gru_high = torch.ones(B, T, 1, device=device)  # all GRU
    g_gru_low = torch.zeros(B, T, 1, device=device)   # all LIF

    loss_high_reg = criterion(y_pred, y_true, lengths, g_gru_high, lambda_reg=0.1)
    loss_low_reg = criterion(y_pred, y_true, lengths, g_gru_low, lambda_reg=0.1)
    assert loss_high_reg > loss_low_reg, (
        f"High g_gru should produce higher loss: {loss_high_reg.item():.6f} "
        f"<= {loss_low_reg.item():.6f}"
    )
    print(f"  reg effect (g_gru=1):  {loss_high_reg.item():.6f}")
    print(f"  reg effect (g_gru=0):  {loss_low_reg.item():.6f}")

    # ── Gradient flow ──
    y_pred_grad = torch.randn(B, T, requires_grad=True)
    loss_grad = criterion(y_pred_grad, y_true, lengths, g_gru, lambda_reg=0.01)
    loss_grad.backward()
    assert y_pred_grad.grad is not None, "Gradient should flow to y_pred"
    assert y_pred_grad.grad.abs().sum() > 0, "Gradient should be non-zero"
    print("  gradient flow:         OK")

    print("=" * 60)
    print("All BioJointLoss assertions passed.")
    print("=" * 60)


if __name__ == "__main__":
    _test_bio_joint_loss()
