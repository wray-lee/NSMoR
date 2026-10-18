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
from typing import Any, Dict, List, Optional, Tuple

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
        out_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute masked MSE loss.

        Args:
            y_pred: ``(B, T)``
            y_true: ``(B, T)``
            lengths: ``(B,)`` — true sequence lengths.
            out_valid: Optional ``(B, T)`` bool emission mask (time-consuming
                mode).  ``None`` (default) keeps the historical length-only
                mask bitwise unchanged.

        Returns:
            Scalar loss tensor.
        """
        assert y_pred.dim() == 2, f"y_pred must be 2-D (B, T), got {y_pred.dim()}-D"
        assert y_true.shape == y_pred.shape
        B, T = y_pred.shape

        arange_t = torch.arange(T, device=y_pred.device)
        mask = (arange_t.unsqueeze(0) < lengths.unsqueeze(1)).float()
        if out_valid is not None:
            assert out_valid.shape == (B, T), (
                f"out_valid shape {tuple(out_valid.shape)} != (B={B}, T={T})"
            )
            mask = mask * out_valid.to(mask.dtype)

        squared_errors = (y_pred - y_true) ** 2
        masked_errors = squared_errors * mask

        if self.reduction == "mean":
            total_valid = mask.sum().clamp(min=1.0)
            return masked_errors.sum() / total_valid
        return masked_errors.sum()


def stimulus_anchored_expected_delay(
    expected_delay: torch.Tensor,
    anchor_frames: torch.Tensor,
) -> torch.Tensor:
    """Read the model's expected processing delay at each stimulus anchor.

    The primary latency readout (ADR 0012) is the SINGLE anchor frame
    ``d_i = E[delay][i, a_i]``, not a whole-trial average and not the
    post-arbitration ``realized_delay``.  ``expected_delay`` is source-indexed
    and deterministic (it does not depend on the sampled depth or on
    arbitration); ``anchor_frames`` are in the SAVED (post-crop) coordinate
    system produced by
    :func:`~nsmor.pipeline.conditions.resolve_anchor_crop`.

    Args:
        expected_delay: ``(B, T)`` floating per-source expected delay.
        anchor_frames: ``(B,)`` int64 anchor frame indices in saved
            coordinates.

    Returns:
        ``(B,)`` floating anchor-frame expected delays.

    Raises:
        ValueError: On a shape/dtype violation or an out-of-range anchor
            (``a_i < 0`` or ``a_i >= T``).
    """
    if expected_delay.dim() != 2:
        raise ValueError(
            "expected_delay must be (B, T), got "
            f"{tuple(expected_delay.shape)}"
        )
    if not expected_delay.is_floating_point():
        raise ValueError(
            f"expected_delay must be floating, got {expected_delay.dtype}"
        )
    B, T = expected_delay.shape
    assert expected_delay.shape == (B, T)
    if anchor_frames.shape != (B,):
        raise ValueError(
            f"anchor_frames shape {tuple(anchor_frames.shape)} != (B={B},)"
        )
    if anchor_frames.dtype != torch.long:
        raise ValueError(
            f"anchor_frames must be int64, got {anchor_frames.dtype}"
        )
    if anchor_frames.device != expected_delay.device:
        raise ValueError("anchor_frames and expected_delay must share a device")
    if bool(((anchor_frames < 0) | (anchor_frames >= T)).any()):
        raise ValueError(
            f"anchor_frames must lie in [0, {T}); got "
            f"{anchor_frames.tolist()}"
        )
    out = expected_delay.gather(1, anchor_frames.unsqueeze(1)).squeeze(1)
    assert out.shape == (B,)
    return out


def _differentiable_latency_zero(
    expected_delay: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Return a graph-connected zero that cannot be poisoned by padding.

    Prefer a zero multiple of the finite VALID operands (``t < lengths`` and
    finite); if that slice is empty, fall back to a fresh leaf.  Never reduce
    the raw tensor: ``NaN * 0`` in padded frames would otherwise produce a
    nonfinite "zero".
    """
    B, T = expected_delay.shape
    arange_t = torch.arange(T, device=expected_delay.device).unsqueeze(0)
    valid = arange_t < lengths.unsqueeze(1)
    safe = expected_delay[valid]
    safe = safe[torch.isfinite(safe)]
    if safe.numel() > 0:
        return safe.sum() * 0.0
    return expected_delay.new_zeros((), requires_grad=True)


def _latency_contrast_loss(
    *,
    expected_delay: torch.Tensor,
    anchor_frames: torch.Tensor,
    target_frames: torch.Tensor,
    target_mask: torch.Tensor,
    group_id: torch.Tensor,
    prefix_id: torch.Tensor,
    pair_id: Optional[torch.Tensor],
    target_group: int,
    reference_group: int,
    lengths: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Stimulus-anchored, prefix-level latency contrast (ADR 0012).

    The observed latency decomposes as
    ``L_obs = L_fixed(g) + L_proc(g) + eps``.  This loss aligns only the
    CONDITION-RELATIVE difference between the pre-registered target and
    reference groups, so it identifies ``Delta L_obs = Delta L_fixed +
    Delta L_proc`` — never ``Delta L_proc`` alone.  The condition-independence
    of ``L_fixed`` is a declared assumption, not a result.  Processing time
    is supervised by this term, not emergent.

    Aggregation is prefix-level and unweighted: per-prefix means of the
    anchor readout ``d_i`` and of the observed target latency, then the
    unweighted mean over the target / reference prefix sets.  A prefix with
    many trials therefore gets ONE group weight.  Only the single
    pre-registered contrast enters the loss; ``visual_only`` /
    ``no_stimulus`` trials are ignored.  If the optional ``pair_id`` is
    supplied, the paired variant is used (exactly one target and one
    reference prefix per pair; incomplete pairs are skipped).

    Args:
        expected_delay: ``(B, T)`` differentiable per-source expected delay.
        anchor_frames: ``(B,)`` int64 saved-coordinate stimulus anchors.
        target_frames: ``(B,)`` observed latency in frames.
        target_mask: ``(B,)`` bool mask of trials with an observed latency.
        group_id: ``(B,)`` int64 condition id.
        prefix_id: ``(B,)`` int64 recording-prefix id.
        pair_id: Optional ``(B,)`` int64 pair id (paired variant).
        target_group: Pre-registered target condition id.
        reference_group: Pre-registered reference condition id.
        lengths: ``(B,)`` true sequence lengths.

    Returns:
        ``(loss, stats)``: a scalar loss (differentiable zero when the batch
        lacks a group/prefix) and a plain-Python stats dict.

    Raises:
        ValueError: On any shape/dtype/domain violation.  A batch-level
            missing group is NOT an error (it returns a differentiable zero);
            only a malformed request fails closed.
    """
    if not isinstance(expected_delay, torch.Tensor):
        raise ValueError("latency_expected_delay must be a torch.Tensor")
    if expected_delay.dim() != 2:
        raise ValueError(
            "latency_expected_delay shape must be (B, T), got "
            f"{tuple(expected_delay.shape)}"
        )
    if not expected_delay.is_floating_point():
        raise ValueError(
            "latency_expected_delay must be floating, got "
            f"{expected_delay.dtype}"
        )
    B, T = expected_delay.shape
    assert expected_delay.shape == (B, T)
    device = expected_delay.device
    if torch.is_grad_enabled() and not expected_delay.requires_grad:
        raise ValueError(
            "latency_expected_delay must be differentiable (requires_grad); "
            "refusing a detached delay (a no_grad diagnostic call is exempt)."
        )
    if lengths.shape != (B,) or lengths.device != device:
        raise ValueError(
            f"lengths shape/device {tuple(lengths.shape)}/{lengths.device} "
            f"!= (B={B},)/{device}"
        )

    def _require_tensor(
        value: Any, name: str, shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if value.shape != shape:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} != {shape}"
            )
        if value.dtype != dtype:
            raise ValueError(
                f"{name} dtype {value.dtype} != {dtype}"
            )
        if value.device != device:
            raise ValueError(f"{name} must share the expected_delay device")
        return value

    anchor_frames = _require_tensor(
        anchor_frames, "latency_anchor_frames", (B,), torch.long,
    )
    if not isinstance(target_frames, torch.Tensor):
        raise ValueError("latency_target_frames must be a torch.Tensor")
    if target_frames.shape != (B,):
        raise ValueError(
            f"latency_target_frames shape {tuple(target_frames.shape)} "
            f"!= (B={B},)"
        )
    if not target_frames.is_floating_point():
        raise ValueError(
            "latency_target_frames must be floating (frames), got "
            f"{target_frames.dtype}"
        )
    if target_frames.device != device:
        raise ValueError("latency_target_frames must share the device")
    target_mask = _require_tensor(
        target_mask, "latency_target_mask", (B,), torch.bool,
    )
    group_id = _require_tensor(
        group_id, "latency_group_id", (B,), torch.long,
    )
    prefix_id = _require_tensor(
        prefix_id, "latency_prefix_id", (B,), torch.long,
    )
    if bool((group_id < 0).any()):
        raise ValueError("latency_group_id must be >= 0")
    if bool((prefix_id < 0).any()):
        raise ValueError("latency_prefix_id must be >= 0")
    if pair_id is not None:
        pair_id = _require_tensor(
            pair_id, "latency_pair_id", (B,), torch.long,
        )
        if bool((pair_id < 0).any()):
            raise ValueError("latency_pair_id must be >= 0")
    if (
        isinstance(target_group, bool)
        or not isinstance(target_group, int)
        or target_group < 0
    ):
        raise ValueError(
            f"latency_target_group must be a non-bool int >= 0, got "
            f"{target_group!r}"
        )
    if (
        isinstance(reference_group, bool)
        or not isinstance(reference_group, int)
        or reference_group < 0
    ):
        raise ValueError(
            f"latency_reference_group must be a non-bool int >= 0, got "
            f"{reference_group!r}"
        )
    if target_group == reference_group:
        raise ValueError(
            "latency_target_group and latency_reference_group must be distinct"
        )

    if bool(((anchor_frames < 0) | (anchor_frames >= T)).any()):
        raise ValueError(
            f"latency_anchor_frames must lie in [0, {T}); got "
            f"{anchor_frames.tolist()}"
        )
    arange_t = torch.arange(T, device=device).unsqueeze(0)
    valid = arange_t < lengths.unsqueeze(1)
    ed_valid = expected_delay[valid]
    if ed_valid.numel() > 0:
        if not bool(torch.isfinite(ed_valid).all()):
            raise ValueError(
                "latency_expected_delay must be finite on valid frames"
            )
        if bool((ed_valid < 0).any()):
            raise ValueError(
                "latency_expected_delay must be non-negative on valid frames"
            )

    masked_idx = target_mask.nonzero(as_tuple=True)[0]
    n_masked = int(masked_idx.numel())
    stats: Dict[str, Any] = {
        "latency_skipped": 0,
        "skip_reason": None,
        "n_masked_trials": n_masked,
        "n_target_prefixes": 0,
        "n_reference_prefixes": 0,
        "n_target_trials": 0,
        "n_reference_trials": 0,
        "n_complete_pairs": 0,
        "delta_d_frames": None,
        "delta_t_frames": None,
    }

    def _zero(reason: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        stats["latency_skipped"] = 1
        stats["skip_reason"] = reason
        stats["latency_loss"] = 0.0
        return _differentiable_latency_zero(expected_delay, lengths), stats

    if n_masked == 0:
        return _zero("no_masked_trials")
    if bool((lengths[masked_idx] == 0).any()):
        raise ValueError(
            "a masked latency trial has length 0 (no prediction to align)"
        )
    if bool((anchor_frames[masked_idx] >= lengths[masked_idx]).any()):
        raise ValueError(
            "latency_anchor_frames must satisfy a_i < lengths_i for masked "
            "trials"
        )
    if not bool(torch.isfinite(target_frames[masked_idx]).all()):
        raise ValueError(
            "latency_target_frames must be finite where the mask is True"
        )
    if bool((target_frames[masked_idx] < 0).any()):
        raise ValueError(
            "latency_target_frames must be non-negative where the mask is True"
        )

    p_masked = prefix_id[masked_idx]
    g_masked = group_id[masked_idx]
    for prefix in torch.unique(p_masked).tolist():
        groups = torch.unique(g_masked[p_masked == prefix])
        if groups.numel() != 1:
            raise ValueError(
                f"latency_prefix_id={prefix} spans multiple groups "
                f"{groups.tolist()}; a prefix-level contrast is not "
                "identifiable (conditions must be nested in prefixes)."
            )

    in_target = g_masked == target_group
    in_reference = g_masked == reference_group
    stats["n_target_trials"] = int(in_target.sum())
    stats["n_reference_trials"] = int(in_reference.sum())
    if not bool(in_target.any()):
        return _zero("no_target_group")
    if not bool(in_reference.any()):
        return _zero("no_reference_group")
    target_prefixes = torch.unique(p_masked[in_target])
    reference_prefixes = torch.unique(p_masked[in_reference])
    stats["n_target_prefixes"] = int(target_prefixes.numel())
    stats["n_reference_prefixes"] = int(reference_prefixes.numel())
    if target_prefixes.numel() == 0:
        return _zero("no_target_prefix")
    if reference_prefixes.numel() == 0:
        return _zero("no_reference_prefix")

    d = stimulus_anchored_expected_delay(expected_delay, anchor_frames)
    d_masked = d.double()[masked_idx]
    t_masked = target_frames.double()[masked_idx]

    def _prefix_mean(
        values: torch.Tensor, prefixes: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [values[p_masked == p].mean() for p in prefixes.tolist()]
        )

    if pair_id is None:
        delta_d = (
            _prefix_mean(d_masked, target_prefixes).mean()
            - _prefix_mean(d_masked, reference_prefixes).mean()
        )
        delta_t = (
            _prefix_mean(t_masked, target_prefixes).mean()
            - _prefix_mean(t_masked, reference_prefixes).mean()
        )
        loss = (delta_d - delta_t) ** 2
        stats["delta_d_frames"] = float(delta_d.detach())
        stats["delta_t_frames"] = float(delta_t.detach())
    else:
        pair_masked = pair_id[masked_idx]
        complete: List[Tuple[int, int]] = []
        for pair in torch.unique(pair_masked).tolist():
            in_pair = pair_masked == pair
            prefixes_in_pair = torch.unique(p_masked[in_pair])
            target_in_pair = [
                p for p in prefixes_in_pair.tolist()
                if int(torch.unique(
                    g_masked[in_pair & (p_masked == p)]
                ).item()) == target_group
            ]
            reference_in_pair = [
                p for p in prefixes_in_pair.tolist()
                if int(torch.unique(
                    g_masked[in_pair & (p_masked == p)]
                ).item()) == reference_group
            ]
            if len(target_in_pair) == 1 and len(reference_in_pair) == 1:
                complete.append(
                    (target_in_pair[0], reference_in_pair[0])
                )
        stats["n_complete_pairs"] = len(complete)
        if not complete:
            return _zero("no_complete_pairs")

        def _one_prefix_mean(values: torch.Tensor, prefix: int) -> torch.Tensor:
            return values[p_masked == prefix].mean()

        diffs = []
        for target_prefix, reference_prefix in complete:
            diffs.append(
                (
                    _one_prefix_mean(d_masked, target_prefix)
                    - _one_prefix_mean(d_masked, reference_prefix)
                ) - (
                    _one_prefix_mean(t_masked, target_prefix)
                    - _one_prefix_mean(t_masked, reference_prefix)
                )
            )
        loss = torch.stack(diffs).pow(2).mean()

    assert loss.dim() == 0
    stats["latency_loss"] = float(loss.detach())
    return loss, stats


class BioDecisionLoss(nn.Module):
    """
    Bio-decision loss for Phase 2 of Hybrid Funnel training.

    Combines masked MSE with all biological / physics penalties:

    - **Router regularization** — penalizes GRU routing gate collapse
    - **ATP metabolic cost** — penalizes mean firing rate
    - **Population sparsity** (L1) — pushes firing rate toward target
    - **Temporal-coherence smoothness** (``lambda_jerk``) — penalizes the
      squared third finite difference of the scalar velocity prediction.
      This is a *frame-based third-velocity-difference smoothness proxy*,
      NOT physical jerk (physical jerk is the first time-derivative of
      acceleration, equivalently the second derivative of velocity): it
      carries no ``dt`` scaling and is evaluated on the sampled frame grid,
      so the third difference of velocity corresponds to a snap-like fourth
      derivative of position, not jerk.
      The historical parameter name/key and its numerics are preserved
      unchanged for checkpoint/experiment compatibility.

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
        # Last latency-contrast diagnostics (plain Python; never a tensor,
        # never in state_dict).  Overwritten on every forward call.
        self.last_latency_stats: Dict[str, Any] = {
            "latency_loss": 0.0,
            "latency_skipped": 1,
            "skip_reason": "disabled",
        }

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
        lambda_routing_aux: float = 0.0,
        wind_only_mask: Optional[torch.Tensor] = None,
        routing_aux_margin: float = 0.024,
        lambda_compute: float = 0.0,
        refinement_ponder_cost: Optional[torch.Tensor] = None,
        out_valid: Optional[torch.Tensor] = None,
        lambda_latency: float = 0.0,
        latency_expected_delay: Optional[torch.Tensor] = None,
        latency_anchor_frames: Optional[torch.Tensor] = None,
        latency_target_frames: Optional[torch.Tensor] = None,
        latency_target_mask: Optional[torch.Tensor] = None,
        latency_group_id: Optional[torch.Tensor] = None,
        latency_prefix_id: Optional[torch.Tensor] = None,
        latency_pair_id: Optional[torch.Tensor] = None,
        latency_target_group: Optional[int] = None,
        latency_reference_group: Optional[int] = None,
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
            lambda_jerk: Frame-based third-velocity-difference smoothness
                weight (legacy name "jerk"; NOT physical jerk — no dt
                scaling, evaluated on the sampled frame grid).
            annealing_factor: Scaling factor for bio-loss lambdas.
            lambda_routing_aux: Auxiliary routing loss weight for modality differentiation.
            wind_only_mask: ``(B,)`` — boolean, True for pure-wind trials.
            routing_aux_margin: Hinge margin on trial-mean ``g_lif``
                separation. Default 0.024 is 1.0× pooled std of the
                backup-corpus calibration (whole-trial mean, not peri
                top-k). The previous 0.2 exceeded the observed 5–95
                frame-level range.
            lambda_compute: Weight on the adaptive-refinement compute cost
                (ACT ponder surrogate). ``0.0`` (default) is bitwise
                unchanged — the term is skipped entirely. Only supplied when
                refinement is enabled (see ``refinement_ponder_cost``).
            refinement_ponder_cost: Scalar ponder cost from the refinement
                module (``N.detach() + R`` averaged over valid tokens).
                Required (and finite) whenever ``lambda_compute > 0``.
            out_valid: Optional ``(B, T)`` bool mask of frames that received a
                real emission.  ``None`` (default) preserves the historical
                length-only mask bitwise.  When supplied (the time-consuming
                mode's alignment mask) the masked MSE scores ONLY those frames,
                so the zero-filled, never-emitted frames — whose target is a
                real (nonzero) value — cannot inject a constant-bias/truncation
                contaminant into the reported skill (findings R1-B2 / R2-B2).
            lambda_latency: Weight on the stimulus-anchored latency contrast
                (ADR 0012).  ``0.0`` (default) leaves the loss bitwise
                unchanged and does not touch any latency tensor.  A positive
                value requires all latency metadata below and a differentiable
                ``latency_expected_delay``.  NOT warmup-scaled.
            latency_expected_delay: ``(B, T)`` differentiable per-source
                expected delay (``time_consuming_expected_delay``).
            latency_anchor_frames: ``(B,)`` int64 saved-coordinate stimulus
                anchors.
            latency_target_frames: ``(B,)`` observed latency in FRAMES.
            latency_target_mask: ``(B,)`` bool mask of observed latencies.
            latency_group_id: ``(B,)`` int64 condition ids.
            latency_prefix_id: ``(B,)`` int64 recording-prefix ids.  A prefix
                may not span multiple conditions (conditions are nested in
                prefixes).
            latency_pair_id: Optional ``(B,)`` int64 pair ids (paired variant).
            latency_target_group: Pre-registered target condition id.
            latency_reference_group: Pre-registered reference condition id.

        Returns:
            Scalar loss tensor.
        """
        assert y_pred.dim() == 2
        assert y_true.shape == y_pred.shape
        assert lengths.dim() == 1
        assert g_gru.dim() == 3 and g_gru.shape[2] == 1

        B, T = y_pred.shape
        # ADR 0012: validate the ORIGINAL weight UNCONDITIONALLY, before the
        # ``== 0`` fast path, so a boolean / non-finite / negative weight is
        # refused even when the term is disabled (mirrors lambda_compute).
        if (
            isinstance(lambda_latency, bool)
            or not isinstance(lambda_latency, (int, float))
            or not math.isfinite(lambda_latency)
            or lambda_latency < 0.0
        ):
            raise ValueError(
                "lambda_latency must be a finite scalar >= 0, got "
                f"{lambda_latency!r}"
            )
        _latency_active = float(lambda_latency) != 0.0

        lambda_energy_eff = lambda_energy * annealing_factor
        lambda_sparse_eff = lambda_sparse * annealing_factor
        lambda_jerk_eff = lambda_jerk * annealing_factor

        arange_t = torch.arange(T, device=y_pred.device)
        mask = (arange_t.unsqueeze(0) < lengths.unsqueeze(1)).float()
        if out_valid is not None:
            assert out_valid.shape == (B, T), (
                f"out_valid shape {tuple(out_valid.shape)} != (B={B}, T={T})"
            )
            # Length mask AND the emission mask: a frame that was never emitted
            # (zero-filled after the causal shift) is excluded from scoring.
            mask = mask * out_valid.to(mask.dtype)

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

        # ── Temporal-coherence smoothness (frame-based third-difference proxy)
        # NOTE: dy3 is the third finite difference of the scalar velocity on
        # the sampled frame grid (no dt scaling).  This is a smoothness proxy,
        # NOT physical jerk.  The historical lambda_jerk numerics are kept
        # unchanged for experiment/checkpoint compatibility.
        # ──
        if lambda_jerk_eff > 0 and T >= 4:
            dy1 = y_pred[:, 1:] - y_pred[:, :-1]
            dy2 = dy1[:, 1:] - dy1[:, :-1]
            dy3 = dy2[:, 1:] - dy2[:, :-1]
            arange_t3 = torch.arange(T - 3, device=y_pred.device)
            length_mask = (arange_t3.unsqueeze(0) + 3 < lengths.unsqueeze(1)).float()
            if out_valid is not None:
                # Frame-based third difference: ``dy3[:, i]`` spans source
                # frames ``i..i+3``, so the mask has width ``T - 3`` while
                # ``out_valid`` has width ``T``.  A window is scored ONLY if
                # every one of its four contributing frames received a real
                # emission, so a never-emitted (zero-filled) frame cannot
                # change the smoothness contribution at all (findings R1-B1 /
                # R2-m1).  ``None`` (default) keeps the historical length-only
                # mask bitwise unchanged.
                ov = out_valid.to(length_mask.dtype)
                assert ov.shape == (B, T), (
                    f"out_valid shape {tuple(out_valid.shape)} != (B={B}, T={T})"
                )
                fully_emitted = (
                    ov[:, :-3] * ov[:, 1:-2] * ov[:, 2:-1] * ov[:, 3:]
                )
                length_mask = length_mask * fully_emitted
            jerk_sq = (dy3 ** 2) * length_mask
            jerk_count = length_mask.sum().clamp(min=1.0)
            total_loss = total_loss + lambda_jerk_eff * (jerk_sq.sum() / jerk_count)

        # ── Auxiliary routing loss (modality differentiation) ──
        lambda_routing_aux_eff = lambda_routing_aux * annealing_factor
        if lambda_routing_aux_eff > 0.0 and wind_only_mask is not None:
            g_lif = 1.0 - g_gru.squeeze(-1)  # (B, T)
            aux_loss = compute_routing_aux_loss(
                g_lif, lengths, wind_only_mask, margin=routing_aux_margin
            )
            total_loss = total_loss + lambda_routing_aux_eff * aux_loss

        # ── Adaptive-refinement compute cost (architecture v1) ──
        # Opt-in and default-off: lambda_compute=0.0 leaves the loss bitwise
        # unchanged (the term is never touched).  H3 / findings A3, B5: the
        # ORIGINAL weight is validated UNCONDITIONALLY, BEFORE the zero fast
        # path, so a boolean ``False`` (which ``== 0.0`` and previously slipped
        # through) is refused like ``True``.  When a positive weight is
        # requested the ponder cost is MANDATORY and must be a FINITE REAL
        # SCALAR tensor (a non-scalar ponder would make the joint loss
        # vector-valued and break ``loss.backward()``); a missing, nonfinite or
        # non-scalar cost is refused rather than silently dropped.
        if (
            isinstance(lambda_compute, bool)
            or not isinstance(lambda_compute, (int, float))
            or not math.isfinite(lambda_compute)
            or lambda_compute < 0.0
        ):
            raise ValueError(
                "lambda_compute must be a finite scalar >= 0, got "
                f"{lambda_compute!r}"
            )
        if lambda_compute != 0.0:
            if refinement_ponder_cost is None:
                raise ValueError(
                    "lambda_compute > 0 requires refinement_ponder_cost; "
                    "refusing to silently drop the compute term."
                )
            if not isinstance(refinement_ponder_cost, torch.Tensor):
                raise ValueError(
                    "refinement_ponder_cost must be a torch.Tensor scalar, got "
                    f"{type(refinement_ponder_cost).__name__}"
                )
            # R6 / A11: require a 0-d tensor, not merely numel()==1.  A (1,)
            # shaped ponder makes the joint loss (1,)-shaped, violating the
            # "Scalar loss tensor" contract and silently broadcasting
            # downstream.  ``dim()==0`` is the exact scalar contract.
            if refinement_ponder_cost.dim() != 0:
                raise ValueError(
                    "refinement_ponder_cost must be a 0-d scalar tensor, got "
                    f"shape {tuple(refinement_ponder_cost.shape)}"
                )
            if not refinement_ponder_cost.is_floating_point():
                raise ValueError(
                    "refinement_ponder_cost must be a real floating scalar, got "
                    f"dtype {refinement_ponder_cost.dtype}"
                )
            if not bool(torch.isfinite(refinement_ponder_cost).all()):
                raise ValueError(
                    "refinement_ponder_cost is nonfinite; refusing to add a "
                    "nonfinite compute-cost term."
                )
            total_loss = total_loss + lambda_compute * refinement_ponder_cost

        # ── Stimulus-anchored latency contrast (ADR 0012) ──
        # Default-off: ``lambda_latency == 0.0`` returns the loss bitwise
        # unchanged and never reads a latency tensor.  A positive weight
        # requires every metadata tensor and a differentiable expected delay;
        # the contrast is computed in float64 (prefix/group reductions) and
        # cast back to the loss dtype.
        if not _latency_active:
            self.last_latency_stats = {
                "latency_loss": 0.0,
                "latency_skipped": 1,
                "skip_reason": "disabled",
            }
            return total_loss
        missing = [
            name for name, value in (
                ("latency_expected_delay", latency_expected_delay),
                ("latency_anchor_frames", latency_anchor_frames),
                ("latency_target_frames", latency_target_frames),
                ("latency_target_mask", latency_target_mask),
                ("latency_group_id", latency_group_id),
                ("latency_prefix_id", latency_prefix_id),
                ("latency_target_group", latency_target_group),
                ("latency_reference_group", latency_reference_group),
            ) if value is None
        ]
        if missing:
            raise ValueError(
                "lambda_latency > 0 requires " + ", ".join(missing)
                + "; refusing to silently drop the latency term."
            )
        # The expected delay must be aligned frame-for-frame with the
        # prediction; a (B, T') tensor with a different T would silently
        # index a different time base (anchor/valid checks are internal to
        # the passed tensor, so they cannot catch it).
        if not isinstance(latency_expected_delay, torch.Tensor):
            raise ValueError(
                "latency_expected_delay must be a torch.Tensor"
            )
        if latency_expected_delay.shape != y_pred.shape:
            raise ValueError(
                "latency_expected_delay shape "
                f"{tuple(latency_expected_delay.shape)} != y_pred shape "
                f"{tuple(y_pred.shape)}"
            )
        latency_loss, latency_stats = _latency_contrast_loss(
            expected_delay=latency_expected_delay,
            anchor_frames=latency_anchor_frames,
            target_frames=latency_target_frames,
            target_mask=latency_target_mask,
            group_id=latency_group_id,
            prefix_id=latency_prefix_id,
            pair_id=latency_pair_id,
            target_group=latency_target_group,
            reference_group=latency_reference_group,
            lengths=lengths,
        )
        self.last_latency_stats = latency_stats
        total_loss = total_loss + float(lambda_latency) * latency_loss.to(
            total_loss.dtype,
        )

        return total_loss


def compute_routing_aux_loss(
    g_lif: torch.Tensor,
    lengths: torch.Tensor,
    wind_only_mask: torch.Tensor,
    margin: float = 0.024,
) -> torch.Tensor:
    """
    Auxiliary routing loss: penalize gate overlap between pure-wind and visual-present trials.

    Encourages trial-level gate differentiation by pushing pure-wind trials
    toward high `g_lif` (fast LIF pathway) and visual-present trials toward
    low `g_lif` (smooth GRU pathway). Aggregation is the whole-trial
    masked mean: peri-stimulus top-k saturates both groups (~0.514) and
    carries no condition signal.

    Args:
        g_lif: ``(B, T, 1)`` or ``(B, T)`` — LIF routing gate over time.
        lengths: ``(B,)`` — valid timesteps per trial.
        wind_only_mask: ``(B,)`` — boolean mask, True for pure-wind trials.
        margin: Minimum desired separation between group means.
            Default 0.024 = 1.0× pooled std on
            ``nsmor_dataset_full_backup.pt`` (wind 0.485±0.032 vs
            visual 0.439±0.012, Cohen's d=1.87).

    Returns:
        Scalar loss: ``max(0, margin - (mean(g_wind) - mean(g_visual)))``.
        Zero when either group is empty, or separation exceeds margin.
    """
    if g_lif.dim() == 3:
        g_lif = g_lif.squeeze(-1)  # (B, T)
    assert g_lif.dim() == 2, f"g_lif must be (B, T) or (B, T, 1), got {g_lif.shape}"
    assert lengths.dim() == 1
    assert wind_only_mask.dim() == 1
    assert g_lif.shape[0] == lengths.shape[0] == wind_only_mask.shape[0]

    B, T = g_lif.shape
    device = g_lif.device

    # Per-trial mean gate (masked by valid lengths)
    arange_t = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    mask = (arange_t < lengths.unsqueeze(1)).float()  # (B, T)
    masked_gates = g_lif * mask  # (B, T)
    valid_counts = mask.sum(dim=1).clamp(min=1.0)  # (B,)
    g_lif_per_trial = masked_gates.sum(dim=1) / valid_counts  # (B,)

    # Partition by modality
    wind_mask_f = wind_only_mask.float()
    n_wind = wind_mask_f.sum()
    n_visual = (1.0 - wind_mask_f).sum()

    # If either group is empty, no penalty
    if n_wind < 1.0 or n_visual < 1.0:
        return torch.zeros((), device=device)

    g_wind_mean = (g_lif_per_trial * wind_mask_f).sum() / n_wind
    g_visual_mean = (g_lif_per_trial * (1.0 - wind_mask_f)).sum() / n_visual

    # Hinge loss: penalize if separation < margin
    separation = g_wind_mean - g_visual_mean
    loss = torch.clamp(margin - separation, min=0.0)
    return loss


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
        lambda_routing_aux: float = 0.0,
        wind_only_mask: Optional[torch.Tensor] = None,
        routing_aux_margin: float = 0.024,
        lambda_compute: float = 0.0,
        refinement_ponder_cost: Optional[torch.Tensor] = None,
        out_valid: Optional[torch.Tensor] = None,
        lambda_latency: float = 0.0,
        latency_expected_delay: Optional[torch.Tensor] = None,
        latency_anchor_frames: Optional[torch.Tensor] = None,
        latency_target_frames: Optional[torch.Tensor] = None,
        latency_target_mask: Optional[torch.Tensor] = None,
        latency_group_id: Optional[torch.Tensor] = None,
        latency_prefix_id: Optional[torch.Tensor] = None,
        latency_pair_id: Optional[torch.Tensor] = None,
        latency_target_group: Optional[int] = None,
        latency_reference_group: Optional[int] = None,
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
            lambda_jerk: Frame-based third-velocity-difference smoothness
                weight (legacy name "jerk"; NOT physical jerk — no dt
                scaling, evaluated on the sampled frame grid).
            annealing_factor: Scaling factor for bio-loss lambdas.
            lambda_routing_aux: Auxiliary routing loss weight for modality differentiation.
            wind_only_mask: ``(B,)`` — boolean, True for pure-wind trials.
            routing_aux_margin: Hinge margin for the routing-aux term.
            lambda_compute: Adaptive-refinement compute-cost weight (default
                0.0 keeps the loss bitwise unchanged).
            refinement_ponder_cost: Scalar ACT ponder cost; required and
                finite whenever ``lambda_compute > 0``.
            out_valid: Optional ``(B, T)`` bool emission mask (time-consuming
                mode).  ``None`` (default) keeps the historical length-only
                mask bitwise unchanged.
            lambda_latency: Stimulus-anchored latency-contrast weight (ADR
                0012; default ``0.0`` keeps the loss bitwise unchanged).
            latency_expected_delay: ``(B, T)`` differentiable expected delay.
            latency_anchor_frames: ``(B,)`` int64 saved-coordinate anchors.
            latency_target_frames: ``(B,)`` observed latencies in frames.
            latency_target_mask: ``(B,)`` bool observed-latency mask.
            latency_group_id: ``(B,)`` int64 condition ids.
            latency_prefix_id: ``(B,)`` int64 recording-prefix ids.
            latency_pair_id: Optional ``(B,)`` int64 pair ids.
            latency_target_group: Pre-registered target condition id.
            latency_reference_group: Pre-registered reference condition id.

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
            lambda_routing_aux=lambda_routing_aux,
            wind_only_mask=wind_only_mask,
            routing_aux_margin=routing_aux_margin,
            lambda_compute=lambda_compute,
            refinement_ponder_cost=refinement_ponder_cost,
            out_valid=out_valid,
            lambda_latency=lambda_latency,
            latency_expected_delay=latency_expected_delay,
            latency_anchor_frames=latency_anchor_frames,
            latency_target_frames=latency_target_frames,
            latency_target_mask=latency_target_mask,
            latency_group_id=latency_group_id,
            latency_prefix_id=latency_prefix_id,
            latency_pair_id=latency_pair_id,
            latency_target_group=latency_target_group,
            latency_reference_group=latency_reference_group,
        )

    @property
    def last_latency_stats(self) -> Dict[str, Any]:
        """Read-only delegate to the owned :class:`BioDecisionLoss`."""
        return self.bio_loss.last_latency_stats


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
