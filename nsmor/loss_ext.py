"""Extension layer for BioJointLoss — auxiliary routing loss (Ticket #14).

Respects frozen core `loss.py` immutability by wrapping rather than editing.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def compute_routing_aux_loss(
    g_lif: torch.Tensor,
    lengths: torch.Tensor,
    wind_only_mask: torch.Tensor,
    margin: float = 0.024,
) -> torch.Tensor:
    """Auxiliary routing loss: penalize gate overlap between conditions.

    Encourages trial-level differentiation by stimulus modality. The loss
    fires when the gap between pure-wind (which should route to LIF) and
    visual-present (which should route to GRU) is smaller than `margin`.

    Args:
        g_lif: LIF gate from MoRRouter, shape [B, T, 1] or [B, T].
        lengths: True (unpadded) sequence lengths, shape [B].
        wind_only_mask: Boolean mask where True indicates pure-wind trials,
            shape [B]. Derived from `is_pure_wind` dataset key.
        margin: Desired minimum separation (mean(g_lif_wind) -
            mean(g_lif_visual) >= margin). Default 0.024 (1.0× pooled std).

    Returns:
        Scalar tensor. Zero when separation >= margin, positive hinge
        penalty otherwise.

    Shape assertions:
        - g_lif must have shape [B, T, 1] or [B, T]
        - lengths must have shape [B]
        - wind_only_mask must have shape [B]
        - B must match across all three
        - lengths must all be > 0
    """
    # Shape validation
    if g_lif.dim() == 3 and g_lif.size(2) == 1:
        g_lif = g_lif.squeeze(2)  # [B, T, 1] -> [B, T]
    assert g_lif.dim() == 2, f"g_lif must be [B, T] or [B, T, 1], got {g_lif.shape}"
    B, T = g_lif.shape
    assert lengths.shape == (B,), f"lengths shape {lengths.shape} != ({B},)"
    assert wind_only_mask.shape == (B,), (
        f"wind_only_mask shape {wind_only_mask.shape} != ({B},)"
    )
    assert torch.all(lengths > 0), "All lengths must be > 0"
    assert torch.all(lengths <= T), f"lengths must be <= {T}"

    # Aggregate per-trial gates: mean over valid (non-padded) frames
    g_lif_per_trial = torch.zeros(B, device=g_lif.device, dtype=g_lif.dtype)
    for i in range(B):
        valid_frames = lengths[i].item()
        g_lif_per_trial[i] = g_lif[i, :valid_frames].mean()

    # Partition by stimulus condition
    has_wind = wind_only_mask.sum().item()
    has_visual = (~wind_only_mask).sum().item()

    # Degenerate cases: no separation penalty if one condition is absent
    if has_wind == 0 or has_visual == 0:
        return torch.tensor(0.0, device=g_lif.device, dtype=g_lif.dtype)

    g_wind = g_lif_per_trial[wind_only_mask]
    g_visual = g_lif_per_trial[~wind_only_mask]

    # Separation objective: mean(g_wind) - mean(g_visual) >= margin
    # Hinge loss: max(0, margin - separation)
    separation = g_wind.mean() - g_visual.mean()
    loss = torch.clamp(margin - separation, min=0.0)

    return loss


class BioJointLossExt(nn.Module):
    """Extended BioJointLoss with auxiliary routing differentiation.

    Wraps the frozen `nsmor.loss.BioJointLoss` without modification, adding
    the `lambda_routing_aux` term and `wind_only_mask` handling.
    """

    def __init__(self, base_loss: nn.Module):
        """Wrap an existing BioJointLoss instance.

        Args:
            base_loss: Frozen `BioJointLoss` instance from `nsmor.loss`.
        """
        super().__init__()
        self.base_loss = base_loss

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        lengths: torch.Tensor,
        g_gru: torch.Tensor,
        g_lif: Optional[torch.Tensor] = None,
        lif_spikes: Optional[torch.Tensor] = None,
        *,
        lambda_reg: float = 0.2,
        lambda_energy: float = 0.0,
        lambda_sparse: float = 0.0,
        lambda_jerk: float = 0.0,
        lambda_routing_aux: float = 0.0,
        wind_only_mask: Optional[torch.Tensor] = None,
        annealing_factor: float = 1.0,
        routing_aux_margin: float = 0.024,
    ) -> torch.Tensor:
        """Compute joint loss with optional auxiliary routing term.

        Args:
            (same as frozen BioJointLoss, plus:)
            g_lif: LIF gate from MoRRouter, shape [B, T, 1] or [B, T].
                Required if lambda_routing_aux > 0.
            lambda_routing_aux: Weight for the routing auxiliary loss.
                Default 0.0 (disabled).
            wind_only_mask: Boolean mask [B] where True indicates pure-wind
                trials. Required if lambda_routing_aux > 0.
            annealing_factor: Warmup scaling factor applied to bio-losses
                (energy, sparse, jerk, routing_aux). Not applied to MSE or
                `lambda_reg`.

        Returns:
            Scalar loss tensor.
        """
        # Base loss (frozen core)
        base = self.base_loss(
            y_pred=y_pred,
            y_true=y_true,
            lengths=lengths,
            g_gru=g_gru,
            lif_spikes=lif_spikes,
            lambda_reg=lambda_reg,
            lambda_energy=lambda_energy,
            lambda_sparse=lambda_sparse,
            lambda_jerk=lambda_jerk,
            annealing_factor=annealing_factor,
        )

        # Auxiliary routing loss (extension)
        if lambda_routing_aux > 0.0:
            if g_lif is None:
                raise ValueError(
                    "g_lif required when lambda_routing_aux > 0"
                )
            if wind_only_mask is None:
                raise ValueError(
                    "wind_only_mask required when lambda_routing_aux > 0"
                )
            routing_aux = compute_routing_aux_loss(
                g_lif, lengths, wind_only_mask, margin=routing_aux_margin
            )
            return base + lambda_routing_aux * annealing_factor * routing_aux

        return base
