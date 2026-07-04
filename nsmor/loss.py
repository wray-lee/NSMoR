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

import math
from typing import Optional

import torch
import torch.nn as nn


class FrontendLoss(nn.Module):
    """
    Frontend loss for Phase 1 of Hybrid Funnel training.

    Computes **masked MSE only** — no biological regularization,
    no physics constraints.  This is the "regular fitting" loss
    used to train :class:`~nsmor.model_nsmor_core.FrontendEncoder`
    while :class:`~nsmor.model_nsmor_core.BioDecisionCore` is frozen.

    The gradient path is clean: ``L_MSE -> y_pred -> backend -> e_detached``
    stops at the ``.detach()`` boundary, so only the frontend receives
    parameter updates.

    Args:
        reduction: ``"mean"`` or ``"sum"``.

    Example::

        criterion = FrontendLoss()
        loss = criterion(y_pred, y_true, lengths)
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
    ) -> torch.Tensor:
        """
        Compute masked MSE loss.

        Args:
            y_pred: ``(B, T)``
            y_true: ``(B, T)``
            lengths: ``(B,)`` — true sequence lengths.

        Returns:
            Scalar loss tensor.
        """
        assert y_pred.dim() == 2, f"y_pred must be 2-D (B, T), got {y_pred.dim()}-D"
        assert y_true.shape == y_pred.shape
        B, T = y_pred.shape

        arange_t = torch.arange(T, device=y_pred.device)
        mask = (arange_t.unsqueeze(0) < lengths.unsqueeze(1)).float()

        squared_errors = (y_pred - y_true) ** 2
        masked_errors = squared_errors * mask

        if self.reduction == "mean":
            total_valid = mask.sum().clamp(min=1.0)
            return masked_errors.sum() / total_valid
        return masked_errors.sum()


class BioDecisionLoss(nn.Module):
    """
    Bio-decision loss for Phase 2 of Hybrid Funnel training.

    Combines masked MSE with all biological / physics penalties:

    - **Router regularization** — penalizes GRU routing gate collapse
    - **ATP metabolic cost** — penalizes mean firing rate
    - **Population sparsity** (L1) — pushes firing rate toward target
    - **Temporal coherence** (jerk) — enforces smooth kinematics

    This loss is used to train
    :class:`~nsmor.model_nsmor_core.BioDecisionCore` while
    :class:`~nsmor.model_nsmor_core.FrontendEncoder` is frozen.

    Args:
        reduction: ``"mean"`` or ``"sum"``.
        target_rate: Target mean firing rate for sparsity L1 penalty.

    Example::

        criterion = BioDecisionLoss(target_rate=0.05)
        loss = criterion(
            y_pred, y_true, lengths, g_gru,
            lambda_reg=0.01, lif_spikes=lif_spikes,
            lambda_energy=1e-3, lambda_sparse=1e-2,
        )
    """

    def __init__(
        self,
        reduction: str = "mean",
        target_rate: float = 0.05,
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"reduction must be 'mean' or 'sum', got '{reduction}'"
            )
        self.reduction = reduction
        self.target_rate = target_rate

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        lengths: torch.Tensor,
        g_gru: torch.Tensor,
        lambda_reg: float = 0.01,
        jerk_mask: Optional[torch.Tensor] = None,
        lif_spikes: Optional[torch.Tensor] = None,
        lambda_energy: float = 0.0,
        lambda_sparse: float = 0.0,
        lambda_jerk: float = 0.0,
        annealing_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute bio-decision loss (MSE + physics penalties).

        Args:
            y_pred: ``(B, T)``
            y_true: ``(B, T)``
            lengths: ``(B,)``
            g_gru: ``(B, T, 1)`` — GRU routing gate.
            lambda_reg: Router regularization weight.
            jerk_mask: ``(B, T)`` — optional sudden-change mask.
            lif_spikes: ``(B, T, H)`` — LIF spike tensor.
            lambda_energy: ATP metabolic cost weight.
            lambda_sparse: Population sparsity L1 weight.
            lambda_jerk: Temporal coherence weight.
            annealing_factor: Scaling factor for bio-loss lambdas.

        Returns:
            Scalar loss tensor.
        """
        assert y_pred.dim() == 2
        assert y_true.shape == y_pred.shape
        assert lengths.dim() == 1
        assert g_gru.dim() == 3 and g_gru.shape[2] == 1

        B, T = y_pred.shape

        lambda_energy_eff = lambda_energy * annealing_factor
        lambda_sparse_eff = lambda_sparse * annealing_factor
        lambda_jerk_eff = lambda_jerk * annealing_factor

        arange_t = torch.arange(T, device=y_pred.device)
        mask = (arange_t.unsqueeze(0) < lengths.unsqueeze(1)).float()

        # ── Masked MSE ──
        squared_errors = (y_pred - y_true) ** 2
        masked_errors = squared_errors * mask
        if self.reduction == "mean":
            total_valid = mask.sum().clamp(min=1.0)
            mse_loss = masked_errors.sum() / total_valid
        else:
            mse_loss = masked_errors.sum()

        # ── Router regularization ──
        g_gru_sq = g_gru.squeeze(-1)
        if jerk_mask is not None:
            reg_raw = (g_gru_sq * mask * jerk_mask).sum()
            N = (mask * jerk_mask).sum().clamp(min=1.0)
        else:
            reg_raw = (g_gru_sq * mask).sum()
            N = mask.sum().clamp(min=1.0)
        reg_loss = lambda_reg * (reg_raw / N)

        total_loss = mse_loss + reg_loss

        # ── Spike statistics ──
        p_hat: Optional[torch.Tensor] = None
        if lif_spikes is not None and (lambda_energy_eff > 0 or lambda_sparse_eff > 0):
            mask_3d = mask.unsqueeze(-1)
            valid_spikes = lif_spikes * mask_3d
            spike_count = valid_spikes.sum()
            n_neurons = lif_spikes.shape[2]
            N_total_valid = mask.sum().clamp(min=1.0)
            n_valid_neuron_steps = N_total_valid * n_neurons
            p_hat = spike_count / n_valid_neuron_steps.clamp(min=1.0)

        # ── ATP metabolic energy cost ──
        if p_hat is not None and lambda_energy_eff > 0:
            total_loss = total_loss + lambda_energy_eff * p_hat

        # ── Population sparsity (L1) ──
        if p_hat is not None and lambda_sparse_eff > 0:
            n_neurons = lif_spikes.shape[2]
            sparse_scale = math.sqrt(float(n_neurons))
            p = torch.tensor(self.target_rate, device=p_hat.device)
            total_loss = total_loss + lambda_sparse_eff * sparse_scale * torch.abs(p_hat - p)

        # ── Temporal coherence (jerk) ──
        if lambda_jerk_eff > 0 and T >= 4:
            dy1 = y_pred[:, 1:] - y_pred[:, :-1]
            dy2 = dy1[:, 1:] - dy1[:, :-1]
            dy3 = dy2[:, 1:] - dy2[:, :-1]
            arange_t3 = torch.arange(T - 3, device=y_pred.device)
            length_mask = (arange_t3.unsqueeze(0) + 3 < lengths.unsqueeze(1)).float()
            jerk_sq = (dy3 ** 2) * length_mask
            jerk_count = length_mask.sum().clamp(min=1.0)
            total_loss = total_loss + lambda_jerk_eff * (jerk_sq.sum() / jerk_count)

        return total_loss


class BioJointLoss(nn.Module):
    """
    Bio-constrained joint loss — backward-compatible wrapper.

    Delegates to :class:`FrontendLoss` (MSE) and
    :class:`BioDecisionLoss` (MSE + bio penalties) internally,
    or can be used directly as the original monolithic loss.

    For two-phase training, prefer using :class:`FrontendLoss` and
    :class:`BioDecisionLoss` directly.

    Args:
        reduction: How to aggregate the MSE across valid timesteps.
            ``"mean"`` (default) divides by total valid count.
            ``"sum"`` sums without dividing.
        target_rate: Target mean firing rate for population sparsity
            L1 penalty.  Default 0.05 (5% activation).
    """

    def __init__(
        self,
        reduction: str = "mean",
        target_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.frontend_loss = FrontendLoss(reduction)
        self.bio_loss = BioDecisionLoss(reduction, target_rate)
        self.reduction = reduction
        self.target_rate = target_rate

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        lengths: torch.Tensor,
        g_gru: torch.Tensor,
        lambda_reg: float = 0.01,
        jerk_mask: Optional[torch.Tensor] = None,
        lif_spikes: Optional[torch.Tensor] = None,
        lambda_energy: float = 0.0,
        lambda_sparse: float = 0.0,
        lambda_jerk: float = 0.0,
        annealing_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the bio-constrained joint loss.

        Delegates to :class:`BioDecisionLoss` which contains the full
        MSE + bio penalty computation.

        Args:
            y_pred: ``(B, T)``
            y_true: ``(B, T)``
            lengths: ``(B,)``
            g_gru: ``(B, T, 1)``
            lambda_reg: Router regularization weight.
            jerk_mask: ``(B, T)`` — optional sudden-change mask.
            lif_spikes: ``(B, T, H)`` — optional LIF spike tensor.
            lambda_energy: ATP metabolic cost weight.
            lambda_sparse: Population sparsity L1 weight.
            lambda_jerk: Temporal coherence weight.
            annealing_factor: Scaling factor for bio-loss lambdas.

        Returns:
            Scalar loss tensor.
        """
        return self.bio_loss(
            y_pred=y_pred,
            y_true=y_true,
            lengths=lengths,
            g_gru=g_gru,
            lambda_reg=lambda_reg,
            jerk_mask=jerk_mask,
            lif_spikes=lif_spikes,
            lambda_energy=lambda_energy,
            lambda_sparse=lambda_sparse,
            lambda_jerk=lambda_jerk,
            annealing_factor=annealing_factor,
        )


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

    B, T, H = 4, 50, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_pred = torch.randn(B, T, device=device)
    y_true = torch.randn(B, T, device=device)
    lengths = torch.tensor([50, 40, 25, 10], dtype=torch.int64, device=device)
    g_gru = torch.rand(B, T, 1, device=device)  # (B, T, 1) in [0, 1]
    lif_spikes = torch.rand(B, T, H, device=device).round()  # binary (B, T, H)

    criterion = BioJointLoss(reduction="mean", target_rate=0.05)

    # ── Forward pass (backward compatible: no spikes) ──
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
    y_pred_grad = torch.randn(B, T, device=device, requires_grad=True)
    loss_grad = criterion(y_pred_grad, y_true, lengths, g_gru, lambda_reg=0.01)
    loss_grad.backward()
    assert y_pred_grad.grad is not None, "Gradient should flow to y_pred"
    assert y_pred_grad.grad.abs().sum() > 0, "Gradient should be non-zero"
    print("  gradient flow:         OK")

    # ── ATP metabolic energy cost ──
    # Ref: Attwell & Laughlin 2001
    print("\n  --- ATP energy cost tests ---")
    spikes_dense = torch.ones(B, T, H, device=device)   # 100% firing
    spikes_sparse = torch.zeros(B, T, H, device=device)  # 0% firing

    loss_dense = criterion(
        y_pred, y_true, lengths, g_gru, lambda_reg=0.0,
        lif_spikes=spikes_dense, lambda_energy=0.1,
    )
    loss_sparse = criterion(
        y_pred, y_true, lengths, g_gru, lambda_reg=0.0,
        lif_spikes=spikes_sparse, lambda_energy=0.1,
    )
    assert loss_dense > loss_sparse, (
        f"Dense spikes should cost more: {loss_dense.item():.6f} "
        f"<= {loss_sparse.item():.6f}"
    )
    print(f"  energy (dense):        {loss_dense.item():.6f}")
    print(f"  energy (sparse):       {loss_sparse.item():.6f}")

    # ── Population sparsity L1 ──
    # Ref: Olshausen & Field 1996
    print("\n  --- Population sparsity tests ---")
    # Spike rate at target (5%) should minimize L1 loss
    spikes_target = torch.bernoulli(
        torch.full((B, T, H), 0.05, device=device)
    )
    loss_at_target = criterion(
        y_pred, y_true, lengths, g_gru, lambda_reg=0.0,
        lif_spikes=spikes_target, lambda_sparse=0.1,
    )
    loss_dense_sparsity = criterion(
        y_pred, y_true, lengths, g_gru, lambda_reg=0.0,
        lif_spikes=spikes_dense, lambda_sparse=0.1,
    )
    assert loss_dense_sparsity > loss_at_target, (
        f"100% firing should have higher L1 loss than 5% target: "
        f"{loss_dense_sparsity.item():.6f} <= {loss_at_target.item():.6f}"
    )
    print(f"  sparse L1 (at target): {loss_at_target.item():.6f}")
    print(f"  sparse L1 (dense):     {loss_dense_sparsity.item():.6f}")

    # ── Backward compatibility: lambda_energy=0, lambda_sparse=0 ──
    loss_compat = criterion(y_pred, y_true, lengths, g_gru, lambda_reg=0.01)
    loss_with_spikes = criterion(
        y_pred, y_true, lengths, g_gru, lambda_reg=0.01,
        lif_spikes=lif_spikes, lambda_energy=0.0, lambda_sparse=0.0,
    )
    assert torch.allclose(loss_compat, loss_with_spikes, atol=1e-6), (
        f"Backward compatibility broken: {loss_compat.item():.6f} != "
        f"{loss_with_spikes.item():.6f}"
    )
    print("  backward compat:       OK")

    print("=" * 60)
    print("All BioJointLoss assertions passed.")
    print("=" * 60)


if __name__ == "__main__":
    _test_bio_joint_loss()
