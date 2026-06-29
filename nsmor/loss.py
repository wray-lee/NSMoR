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
    Bio-constrained joint loss with masked MSE, conditional router
    regularization, ATP metabolic cost, and population sparsity.

    The total loss is:

    .. math::

        \\mathcal{L} = \\mathcal{L}_{\\text{MSE}}
        + \\lambda_{\\text{reg}} \\cdot \\mathcal{L}_{\\text{router}}
        + \\lambda_{\\text{energy}} \\cdot \\mathcal{L}_{\\text{ATP}}
        + \\lambda_{\\text{sparse}} \\cdot \\mathcal{L}_{\\text{sparse}}

    **Router regularization** (existing):
        Penalizes GRU routing gate during transient intervals.

    **ATP metabolic cost** (existing):
        ``L_energy = mean(spike_rate)`` — penalizes mean population
        firing rate, modeling the ~10^8 ATP molecules consumed per
        action potential (Attwell & Laughlin 2001, J. Cereb. Blood
        Flow Metab. 21:1133-1145).  The brain consumes ~20% of
        total body ATP for neural signaling; sparse coding is a
        primary energy-efficiency strategy.

    **Population sparsity** (L1 penalty):
        ``L_sparse = |p_hat - p_target|`` — L1 distance between
        the actual mean firing rate and a target sparse rate (~5%).
        Non-zero gradient everywhere except at target: gradient = -1
        when p_hat < target (constant bootstrap signal), +1 when
        p_hat > target.  No dead zones, no gradient explosion.
        Replaces KL divergence which had gradient dead zones at
        boundaries (p_hat=0 and p_hat>0.998).
        Ref: Olshausen & Field 1996, Nature 381:607-609.

    **Temporal coherence** (new):
        ``L_jerk = mean(jerk^2)`` — penalizes the squared third
        temporal derivative (jerk) of the predicted velocity, enforcing
        biologically plausible smooth kinematics.  Real escape movements
        in crickets have constrained jerk profiles due to muscle
        biomechanics and neural smoothing.  The jerk is computed as
        ``d³y/dt³`` using finite differences on valid (non-padded)
        timesteps.
        Ref: Gabbiani et al. 1999, Nature 401:672-676.

    Args:
        reduction: How to aggregate the MSE across valid timesteps.
            ``"mean"`` (default) divides by total valid count.
            ``"sum"`` sums without dividing.
        target_rate: Target mean firing rate for population sparsity
            L1 penalty.  Default 0.05 (5% activation).  Typical range:
            0.01-0.05 for cortical sparse coding.

            **Design note:** ``target_rate`` is stored as a plain
            Python float attribute (NOT a registered buffer).  It is
            read dynamically on each ``forward()`` call, so callers
            can mutate ``criterion.target_rate = 0.1`` between epochs
            to implement annealing or curriculum schedules.  A
            ``register_buffer`` approach would freeze the value at
            construction time, silently breaking this mutation pattern.
            The device-resident tensor ``p`` is constructed on-the-fly
            inside ``forward()`` from the current ``self.target_rate``.

    Example::

        criterion = BioJointLoss(target_rate=0.05)
        loss = criterion(
            y_pred=predictions,      # (B, T)
            y_true=targets,          # (B, T)
            lengths=lengths,         # (B,)
            g_gru=g_gru,             # (B, T, 1)
            lambda_reg=0.01,
            jerk_mask=jerk_mask,     # (B, T) — optional
            lif_spikes=lif_spikes,   # (B, T, H) — optional
            lambda_energy=1e-3,
            lambda_sparse=1e-2,
            lambda_jerk=1e-3,        # temporal coherence weight
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
            lif_spikes: ``(B, T, H)`` — LIF spike tensor from model
                internals.  Required for energy and sparsity losses.
                When ``None``, energy and sparsity losses are zero
                (backward compatible).
            lambda_energy: Weight for ATP metabolic cost term.
                Penalizes mean firing rate to model the ~10^8 ATP
                cost per action potential (Attwell & Laughlin 2001).
                Default 0.0 (disabled, backward compatible).
            lambda_sparse: Weight for population sparsity L1 penalty.
                ``|p_hat - target_rate|`` — encourages firing rate
                near ``self.target_rate``.  Non-zero gradient everywhere.
                Default 0.0 (disabled, backward compatible).
            lambda_jerk: Weight for temporal coherence (jerk penalty).
                Penalizes the squared third temporal derivative of
                predicted velocity, enforcing biologically plausible
                smooth kinematics.  Ref: Gabbiani et al. 1999, Nature.
                Default 0.0 (disabled, backward compatible).
            annealing_factor: Scaling factor for lambda_energy,
                lambda_sparse, and lambda_jerk.  Implements warmup/
                annealing schedule self-contained within the loss
                function, so callers don't need to independently
                compute and multiply the warmup factor.  The effective
                weights are:
                - lambda_energy_eff = lambda_energy * annealing_factor
                - lambda_sparse_eff = lambda_sparse * annealing_factor
                - lambda_jerk_eff = lambda_jerk * annealing_factor
                - lambda_reg is NOT scaled (router regularization is
                  active from epoch 0).
                Default 1.0 (no annealing, backward compatible).

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
        if lif_spikes is not None:
            assert lif_spikes.dim() == 3, (
                f"lif_spikes must be 3-D (B, T, H), got {lif_spikes.dim()}-D"
            )
            H = lif_spikes.shape[2]
            assert lif_spikes.shape == (B, T, H), (
                f"lif_spikes shape {tuple(lif_spikes.shape)} != (B={B}, T={T}, H={H})"
            )

        # ── Apply annealing factor ─────────────────────────────
        # Scales bio-loss lambdas but NOT lambda_reg.
        lambda_energy_eff = lambda_energy * annealing_factor
        lambda_sparse_eff = lambda_sparse * annealing_factor
        lambda_jerk_eff = lambda_jerk * annealing_factor

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

        total_loss = mse_loss + reg_loss                           # scalar

        # ── Spike statistics (shared by energy + sparsity losses) ──
        # Pre-computed once to avoid duplicate forward graph nodes.
        # p_hat = mean firing rate across all neurons and valid timesteps.
        p_hat: Optional[torch.Tensor] = None
        if lif_spikes is not None and (lambda_energy_eff > 0 or lambda_sparse_eff > 0):
            # lif_spikes: (B, T, H) — binary spike tensor
            # mask: (B, T) — expand to (B, T, 1) for broadcasting
            mask_3d = mask.unsqueeze(-1)                           # (B, T, 1)
            valid_spikes = lif_spikes * mask_3d                    # (B, T, H)
            spike_count = valid_spikes.sum()                       # scalar
            n_neurons = lif_spikes.shape[2]                        # H
            N_total_valid = mask.sum().clamp(min=1.0)                # scalar
            n_valid_neuron_steps = N_total_valid * n_neurons        # scalar
            p_hat = spike_count / n_valid_neuron_steps.clamp(min=1.0)  # scalar ∈ [0, 1]

        # ── ATP metabolic energy cost ─────────────────────────
        # Ref: Attwell & Laughlin 2001, J. Cereb. Blood Flow Metab.
        # Each action potential costs ~10^8 ATP molecules.  The brain
        # allocates ~20% of body energy to neural signaling, so
        # minimizing spike frequency is a primary metabolic objective.
        if p_hat is not None and lambda_energy_eff > 0:
            energy_loss = lambda_energy_eff * p_hat                # scalar
            total_loss = total_loss + energy_loss                  # scalar

        # ── Population sparsity (L1 penalty) ─────────────────
        # Ref: Olshausen & Field 1996, Nature 381:607-609.
        # Cortical neurons fire at ~1-5% rate.  We use L1 distance
        # |p_hat - p_target| to push the population firing rate toward
        # a sparse target.
        #
        # L1 properties:
        # - Gradient = sign(p_hat - p_target), bounded [-1, +1]
        # - Non-zero gradient everywhere except at target
        # - No dead zones at boundaries (unlike KL with clamp)
        # - Symmetric penalty for undershoot and overshoot
        # - No Adam moment inflation (gradient magnitude <= 1)
        #
        # Note: gradients flow through p_hat → spike_count → lif_spikes
        # because all operations are tensor-based (no .item() calls).
        if p_hat is not None and lambda_sparse_eff > 0:
            # L1 sparse penalty: |p_hat - p_target|
            # Replaces KL divergence which has gradient dead zones at
            # p_hat=0 (clamp) and p_hat>0.998 (upper boundary).
            #
            # L1 gradient analysis:
            #   d(L1)/d(p_hat) = sign(p_hat - p_target)
            #   At p_hat=0 (silent):     grad = -1 (constant, non-zero)
            #   At p_hat=0.01 (<target): grad = -1
            #   At p_hat=0.05 (=target): grad = 0 (minimum, expected)
            #   At p_hat=0.20 (>target): grad = +1
            #   At p_hat=1.0 (saturated): grad = +1
            #
            # Note: the above gradients are d(L)/d(p_hat).  The actual
            # per-spike gradient is scaled by 1/N_valid_neuron_steps:
            #   d(L)/d(spike[i]) = sign * lambda_sparse / N_valid
            # For N_valid=960 and lambda_sparse=0.1, the per-spike
            # gradient at p_hat=0 is -0.1/960 ≈ -1e-4, not -1.
            # This is important for lambda_sparse tuning: the effective
            # gradient magnitude scales as lambda_sparse / N_valid.
            #
            # Properties:
            # - Non-zero gradient EVERYWHERE except exactly at target
            # - No dead zones at boundaries
            # - No gradient explosion (bounded [-1, +1] at p_hat level)
            # - No Adam moment inflation
            # - Symmetric penalty: undershoot and overshoot penalized equally
            # - Bootstrap: gradient = -1 at p_hat=0, constant signal
            #   to increase firing (surrogate gradient handles propagation)
            p = torch.tensor(self.target_rate, device=p_hat.device)
            sparse_loss = lambda_sparse_eff * torch.abs(p_hat - p)
            total_loss = total_loss + sparse_loss

        # ── Temporal coherence (jerk penalty) ─────────────────
        # Ref: Gabbiani et al. 1999, Nature 401:672-676.
        # Penalizes the squared third temporal derivative (jerk) of the
        # predicted velocity.  Real escape movements have constrained
        # jerk profiles due to muscle biomechanics and neural smoothing.
        # jerk = d^3(y_pred)/dt^3, computed via finite differences.
        if lambda_jerk_eff > 0 and T >= 4:
            # Compute third derivative via finite differences
            # dy1 = y[t] - y[t-1]  (velocity of velocity)
            # dy2 = dy1[t] - dy1[t-1]  (acceleration of velocity)
            # dy3 = dy2[t] - dy2[t-1]  (jerk of velocity)
            dy1 = y_pred[:, 1:] - y_pred[:, :-1]                # (B, T-1)
            dy2 = dy1[:, 1:] - dy1[:, :-1]                      # (B, T-2)
            dy3 = dy2[:, 1:] - dy2[:, :-1]                      # (B, T-3)

            # Mask: only count valid (non-padded) timesteps
            # A timestep t in dy3 is valid if t+3 < length
            arange_t3 = torch.arange(T - 3, device=y_pred.device)  # (T-3,)
            length_mask = (arange_t3.unsqueeze(0) + 3 < lengths.unsqueeze(1))  # (B, T-3)
            length_mask = length_mask.float()

            # Squared jerk, masked
            jerk_sq = (dy3 ** 2) * length_mask                  # (B, T-3)
            jerk_count = length_mask.sum().clamp(min=1.0)
            jerk_loss = lambda_jerk_eff * (jerk_sq.sum() / jerk_count)  # scalar
            total_loss = total_loss + jerk_loss                  # scalar

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
    y_pred_grad = torch.randn(B, T, requires_grad=True)
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
