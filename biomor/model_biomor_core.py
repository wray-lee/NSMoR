"""
BioMoR Core — Mixture-of-Recursions (MoR) neural network.

Implements a dual-pathway recurrent architecture that combines:
  - **Path A (LIF):** A Leaky Integrate-and-Fire spiking neuron for
    fast, event-driven sensory transients.
  - **Path B (GRU):** A standard Gated Recurrent Unit for smooth,
    continuous temporal integration.

A learned routing network (the *MoR Router*) blends the two pathway
outputs at every time-step, conditioned on both the sensory encoding
and the static MCMC prior.

All sub-modules are exposed as named attributes for white-box
introspection (manifold / Jacobian analysis) and targeted freezing.

Shape tracking legend
---------------------
    B  = batch_size
    T  = seq_len        (padded)
    D  = sensory_dim    (4 — visual angle, wind, velocity, acceleration)
    H  = hidden_dim
    M  = mcmc_dim       (4 — prior probability vector)
    L  = 2              (number of recursive pathways)

All tensors are annotated with their shape in comments.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ═══════════════════════════════════════════════════════════════
# 1.  Sensory Encoder
# ═══════════════════════════════════════════════════════════════

class SensoryEncoder(nn.Module):
    """
    Map raw 4-D sensory features to a hidden representation.

    ``Linear(4, hidden_dim)`` + ``LayerNorm`` + ``ReLU``

    Input:  ``(B, T, 4)``
    Output: ``(B, T, H)``
    """

    def __init__(self, sensory_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sensory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, sensory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensory: ``(B, T, 4)``

        Returns:
            ``(B, T, H)``
        """
        return self.net(sensory)


# ═══════════════════════════════════════════════════════════════
# 2.  LIF (Leaky Integrate-and-Fire) RNN Cell
# ═══════════════════════════════════════════════════════════════

class LIFCell(nn.Module):
    """
    Leaky Integrate-and-Fire recurrent neuron.

    Dynamics (per time-step *t*)::

        V[t] = α · V[t-1] + β · W_in · E_sensory[t]     (leaky integration)

        if V[t] > V_threshold:
            spike = V[t]          (fire)
            V[t] -= V_threshold   (soft reset)
        else:
            spike = 0             (silent)

    The output is the *spike* (the membrane potential when it fires,
    zero otherwise), making the pathway sparse and event-driven.

    Input:  ``(B, H)`` at each step
    Output: ``(B, H)`` at each step
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        alpha: float = 0.9,
        v_threshold: float = 1.0,
        beta: float = 0.5,
    ) -> None:
        """
        Args:
            hidden_dim: Dimensionality of the membrane state.
            alpha: Leak factor in (0, 1).  Higher → slower decay.
            v_threshold: Spike threshold.
            beta: Input scaling factor.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.v_threshold = v_threshold
        self.beta = beta

        # Input projection (recurrent weight W_in)
        self.W_in = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(
        self,
        input_t: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advance the LIF state by one time-step.

        Args:
            input_t: ``(B, H)`` — sensory encoding at time *t*.
            state: ``(B, H)`` — membrane potential V[t-1].

        Returns:
            ``(spike, new_state)`` where both are ``(B, H)``.
        """
        # Leak + input integration:  V = α·V_prev + β·W_in·input
        v_new = self.alpha * state + self.beta * self.W_in(input_t)   # (B, H)

        # Spike detection: fires where V > threshold
        spike_mask = (v_new > self.v_threshold).float()               # (B, H)
        spike = spike_mask * v_new                                    # (B, H)

        # Soft reset: subtract threshold where spike occurred
        v_new = v_new - spike_mask * self.v_threshold                 # (B, H)

        return spike, v_new

    def init_state(
        self, batch_size: int, device: torch.device,
    ) -> torch.Tensor:
        """Return a zero membrane potential for the batch."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)


# ═══════════════════════════════════════════════════════════════
# 3.  GRU Unit (packed-sequence wrapper)
# ═══════════════════════════════════════════════════════════════

class GRUUnit(nn.Module):
    """
    GRU pathway with ``pack_padded_sequence`` / ``pad_packed_sequence``.

    Input:  ``(B, T, H)`` — sensory encoding
    Output: ``(B, T, H)`` — recurrent hidden states
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, H)``
            lengths: ``(B,)`` — true sequence lengths.

        Returns:
            ``(B, T, H)``
        """
        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.shape[1],
        )
        return out


# ═══════════════════════════════════════════════════════════════
# 4.  MoR Router (Causal Inference Gate)
# ═══════════════════════════════════════════════════════════════

class MoRRouter(nn.Module):
    """
    Per-time-step routing network that blends LIF and GRU outputs.

    Takes the concatenation of ``[E_sensory(t), MCMC_Prior(t)]`` and
    produces a 2-D routing vector ``[g_lif, g_gru]`` via softmax.

    Input:  ``(B, H + M)`` at each step
    Output: ``(B, 2)`` — soft routing weights
    """

    def __init__(self, hidden_dim: int = 64, mcmc_dim: int = 4) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim + mcmc_dim, 2)

    def forward(
        self,
        e_sensory: torch.Tensor,
        mcmc_prior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            e_sensory: ``(B, H)``
            mcmc_prior: ``(B, M)``

        Returns:
            ``(B, 2)`` — ``[g_lif, g_gru]`` with softmax along dim=-1.
        """
        combined = torch.cat([e_sensory, mcmc_prior], dim=-1)  # (B, H+M)
        logits = self.gate(combined)                            # (B, 2)
        return F.softmax(logits, dim=-1)                        # (B, 2)


# ═══════════════════════════════════════════════════════════════
# 5.  Direction Head (Decoder)
# ═══════════════════════════════════════════════════════════════

class DirectionHead(nn.Module):
    """
    Final decoder: ``LayerNorm → ReLU → Dropout → Linear(H, 1)``.

    Input:  ``(B, T, H)``
    Output: ``(B, T)``
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: ``(B, T, H)``

        Returns:
            ``(B, T)``
        """
        return self.net(h).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# 6.  BioMoR Core Network
# ═══════════════════════════════════════════════════════════════

# Valid sub-module names for freeze_modules()
_FREEZABLE_MODULES = frozenset({
    "sensory_encoder",
    "lif_cell",
    "gru_unit",
    "router",
    "direction_head",
})


class BioMoRCore(nn.Module):
    """
    Mixture-of-Recursions (MoR) — dual-pathway recurrent network.

    Architecture
    ------------
    ::

        X_batch ──┬── Sensory_X [B,T,4] ──→ SensoryEncoder ──→ E_sensory [B,T,H]
                   │                                              ↓
                   │                                    ┌─── LIF Path (step-by-step)
                   │                                    │    → Out_lif  [B,T,H]
                   │                                    │
                   │    MCMC_Prior [B,T,4] ──→ MoR Router ──→ [g_lif, g_gru] [B,T,2]
                   │                                    │
                   │                                    └─── GRU Path (packed)
                   │                                         → Out_gru  [B,T,H]
                   │
                   └── Integration: H = g_lif·Out_lif + g_gru·Out_gru  [B,T,H]
                                              ↓
                                    DirectionHead(H, 1) ──→ Y_pred [B,T]

    White-box interfaces
    --------------------
    * ``return_internals=True`` exposes routing gates, LIF potentials /
      spikes, and GRU hidden states for dynamical-systems analysis.
    * ``freeze_modules(["lif_cell", "router"])`` freezes only the
      specified pathways for targeted fine-tuning.
    """

    def __init__(
        self,
        sensory_dim: int = 4,
        mcmc_dim: int = 4,
        hidden_dim: int = 64,
        num_gru_layers: int = 1,
        dropout: float = 0.1,
        lif_alpha: float = 0.9,
        lif_threshold: float = 1.0,
        lif_beta: float = 0.5,
    ) -> None:
        """
        Args:
            sensory_dim: Dimensionality of raw sensory features (4).
            mcmc_dim: Dimensionality of MCMC prior vector (4).
            hidden_dim: Hidden state dimensionality for both pathways.
            num_gru_layers: Number of stacked GRU layers.
            dropout: Dropout probability in the GRU and decoder.
            lif_alpha: LIF leak factor.
            lif_threshold: LIF spike threshold.
            lif_beta: LIF input scaling.
        """
        super().__init__()
        self.sensory_dim = sensory_dim
        self.mcmc_dim = mcmc_dim
        self.hidden_dim = hidden_dim

        # ── Named sub-modules (white-box) ──
        self.sensory_encoder = SensoryEncoder(sensory_dim, hidden_dim)
        self.lif_cell = LIFCell(hidden_dim, lif_alpha, lif_threshold, lif_beta)
        self.gru_unit = GRUUnit(hidden_dim, num_gru_layers, dropout)
        self.router = MoRRouter(hidden_dim, mcmc_dim)
        self.direction_head = DirectionHead(hidden_dim, dropout)

    # ── Public API ────────────────────────────────────────────

    def forward(
        self,
        X_batch: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_internals: bool = False,
        override_gates: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            X_batch: ``(B, T, 8)`` — padded feature tensor.
            lengths: ``(B,)`` — true (unpadded) sequence lengths.
            return_internals: If ``True``, return a second element
                containing routing gates, LIF potentials, LIF spikes,
                and GRU hidden states for downstream analysis.
            override_gates: Optional dictionary for in-silico lesioning.
                If provided, replaces dynamically computed routing gates
                with hardcoded scalar values.  Keys: ``"g_lif"`` and/or
                ``"g_gru"``.  Example: ``{"g_lif": 0.0, "g_gru": 1.0}``
                silences the LIF pathway.  Default ``None`` uses natural
                routing.

        Returns:
            If ``return_internals=False``:
                ``Y_pred``: ``(B, T)``
            If ``return_internals=True``:
                ``(Y_pred, internals)`` where *internals* is a dict with
                keys ``routing_gates``, ``lif_potentials``, ``lif_spikes``,
                ``gru_hidden``.

        Raises:
            ValueError: If ``X_batch`` feature dim is not 8.
            AssertionError: If tensor shapes are inconsistent.
        """
        expected_dim = self.sensory_dim + self.mcmc_dim
        if X_batch.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected feature dim {expected_dim}, got {X_batch.shape[-1]}"
            )

        B, T, _ = X_batch.shape

        # ── Input shape assertions ──
        assert X_batch.dim() == 3, (
            f"X_batch must be 3-D (B, T, F), got {X_batch.dim()}-D"
        )
        assert lengths.shape == (B,), (
            f"lengths must be (B={B},), got {tuple(lengths.shape)}"
        )
        assert (lengths > 0).all(), "All sequence lengths must be positive"
        assert (lengths <= T).all(), (
            f"lengths must not exceed seq_len={T}, got max={lengths.max().item()}"
        )

        # ──────────────────────────────────────────────────────
        # Step 1: Unpack input
        # ──────────────────────────────────────────────────────
        sensory_x = X_batch[:, :, : self.sensory_dim]        # (B, T, D)
        mcmc_prior = X_batch[:, :, self.sensory_dim :]       # (B, T, M)

        # ── Unpack shape assertions ──
        assert sensory_x.shape == (B, T, self.sensory_dim), (
            f"sensory_x shape {tuple(sensory_x.shape)} != "
            f"(B={B}, T={T}, D={self.sensory_dim})"
        )
        assert mcmc_prior.shape == (B, T, self.mcmc_dim), (
            f"mcmc_prior shape {tuple(mcmc_prior.shape)} != "
            f"(B={B}, T={T}, M={self.mcmc_dim})"
        )

        # ──────────────────────────────────────────────────────
        # Step 2: Encode sensory input
        # ──────────────────────────────────────────────────────
        e_sensory = self.sensory_encoder(sensory_x)           # (B, T, H)

        assert e_sensory.shape == (B, T, self.hidden_dim), (
            f"e_sensory shape {tuple(e_sensory.shape)} != "
            f"(B={B}, T={T}, H={self.hidden_dim})"
        )

        # ──────────────────────────────────────────────────────
        # Step 3: Path A — LIF (step-by-step, respects padding)
        # ──────────────────────────────────────────────────────
        out_lif, lif_potentials, lif_spikes = self._run_lif_path(
            e_sensory, lengths,
        )                                                     # (B, T, H) each

        # ── LIF output shape assertions ──
        assert out_lif.shape == (B, T, self.hidden_dim), (
            f"out_lif shape {tuple(out_lif.shape)} != (B={B}, T={T}, H={self.hidden_dim})"
        )
        assert lif_potentials.shape == (B, T, self.hidden_dim), (
            f"lif_potentials shape {tuple(lif_potentials.shape)} != "
            f"(B={B}, T={T}, H={self.hidden_dim})"
        )
        assert lif_spikes.shape == (B, T, self.hidden_dim), (
            f"lif_spikes shape {tuple(lif_spikes.shape)} != "
            f"(B={B}, T={T}, H={self.hidden_dim})"
        )

        # ──────────────────────────────────────────────────────
        # Step 4: Path B — GRU (packed, efficient)
        # ──────────────────────────────────────────────────────
        out_gru = self.gru_unit(e_sensory, lengths)           # (B, T, H)

        assert out_gru.shape == (B, T, self.hidden_dim), (
            f"out_gru shape {tuple(out_gru.shape)} != (B={B}, T={T}, H={self.hidden_dim})"
        )

        # ──────────────────────────────────────────────────────
        # Step 5: MoR Router — per-step blending weights
        # ──────────────────────────────────────────────────────
        e_flat = e_sensory.reshape(B * T, -1)                 # (B*T, H)
        m_flat = mcmc_prior.reshape(B * T, -1)                # (B*T, M)
        gates = self.router(e_flat, m_flat)                    # (B*T, 2)
        gates = gates.reshape(B, T, 2)                         # (B, T, 2)

        assert gates.shape == (B, T, 2), (
            f"gates shape {tuple(gates.shape)} != (B={B}, T={T}, 2)"
        )

        g_lif = gates[:, :, 0:1]                               # (B, T, 1)
        g_gru = gates[:, :, 1:2]                               # (B, T, 1)

        # ──────────────────────────────────────────────────────
        # Step 5b: In-Silico Lesion Hook (Optogenetic Override)
        # ──────────────────────────────────────────────────────
        if override_gates is not None:
            # Replace dynamically computed gates with hardcoded values
            # This enables virtual ablation experiments (Phase 7)
            if "g_lif" in override_gates:
                g_lif = torch.full_like(g_lif, override_gates["g_lif"])
            if "g_gru" in override_gates:
                g_gru = torch.full_like(g_gru, override_gates["g_gru"])

            # Shape assertions after override
            assert g_lif.shape == (B, T, 1), (
                f"g_lif override shape {tuple(g_lif.shape)} != (B={B}, T={T}, 1)"
            )
            assert g_gru.shape == (B, T, 1), (
                f"g_gru override shape {tuple(g_gru.shape)} != (B={B}, T={T}, 1)"
            )

        # ──────────────────────────────────────────────────────
        # Step 6: Integrate pathways
        # ──────────────────────────────────────────────────────
        h_out = g_lif * out_lif + g_gru * out_gru             # (B, T, H)

        assert h_out.shape == (B, T, self.hidden_dim), (
            f"h_out shape {tuple(h_out.shape)} != (B={B}, T={T}, H={self.hidden_dim})"
        )

        # ──────────────────────────────────────────────────────
        # Step 7: Decode to continuous output
        # ──────────────────────────────────────────────────────
        y_pred = self.direction_head(h_out)                    # (B, T)

        assert y_pred.shape == (B, T), (
            f"y_pred shape {tuple(y_pred.shape)} != (B={B}, T={T})"
        )

        if return_internals:
            # Reconstruct the gates tensor from (possibly overridden) g_lif, g_gru
            effective_gates = torch.cat([g_lif, g_gru], dim=-1)  # (B, T, 2)

            internals: Dict[str, torch.Tensor] = {
                "routing_gates": effective_gates,  # (B, T, 2) — may be overridden
                "natural_gates": gates,            # (B, T, 2) — always natural
                "lif_potentials": lif_potentials,  # (B, T, H)
                "lif_spikes": lif_spikes,          # (B, T, H)
                "gru_hidden": out_gru,             # (B, T, H)
            }
            return y_pred, internals

        return y_pred

    def freeze_modules(self, module_names: List[str]) -> None:
        """
        Freeze parameters of the specified sub-modules.

        Sets ``requires_grad = False`` for all parameters in each
        named sub-module.  This enables targeted fine-tuning strategies
        such as freezing only the visual pathway or only the causal gate.

        Args:
            module_names: List of sub-module names to freeze.
                Valid names: ``sensory_encoder``, ``lif_cell``,
                ``gru_unit``, ``router``, ``direction_head``.

        Raises:
            ValueError: If a name is not a valid sub-module.

        Example::

            model = BioMoRCore()
            # Freeze everything except the GRU pathway
            model.freeze_modules([
                "sensory_encoder", "lif_cell", "router", "direction_head",
            ])
            # Only gru_unit parameters will receive gradients
        """
        for name in module_names:
            if name not in _FREEZABLE_MODULES:
                raise ValueError(
                    f"Unknown module '{name}'. "
                    f"Valid names: {sorted(_FREEZABLE_MODULES)}"
                )
            submodule: nn.Module = getattr(self, name)
            for param in submodule.parameters():
                param.requires_grad = False

    # ── Private pathway runners ──────────────────────────────

    def _run_lif_path(
        self,
        e_sensory: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the LIF cell step-by-step, masking padded positions.

        Args:
            e_sensory: ``(B, T, H)``
            lengths: ``(B,)``

        Returns:
            ``(out_lif, potentials, spikes)`` — each ``(B, T, H)``.
        """
        B, T, H = e_sensory.shape
        device = e_sensory.device

        v_state = self.lif_cell.init_state(B, device)         # (B, H)

        out_lif = torch.zeros(B, T, H, device=device)
        potentials = torch.zeros(B, T, H, device=device)
        spikes = torch.zeros(B, T, H, device=device)

        for t in range(T):
            inp_t = e_sensory[:, t, :]                        # (B, H)
            spike, v_state = self.lif_cell(inp_t, v_state)    # (B, H), (B, H)

            mask = (t < lengths).float().unsqueeze(-1)        # (B, 1)
            out_lif[:, t, :] = spike * mask
            potentials[:, t, :] = v_state * mask
            spikes[:, t, :] = spike * mask

        return out_lif, potentials, spikes


# ═══════════════════════════════════════════════════════════════
# 7.  Backward-compatible alias
# ═══════════════════════════════════════════════════════════════

# Keep the old class name working so existing tests / imports survive.
BioMoR = BioMoRCore


# ═══════════════════════════════════════════════════════════════
# 8.  Forward-pass smoke test
# ═══════════════════════════════════════════════════════════════

def _test_forward_pass() -> None:
    """
    Verify that ``BioMoRCore.forward`` produces the expected output shapes.

    Run this module directly to execute the test::

        python -m biomor.model_biomor_core
    """
    print("=" * 60)
    print("BioMoRCore forward-pass smoke test")
    print("=" * 60)

    B, T, H = 4, 120, 64
    device = torch.device("cpu")

    X_batch = torch.randn(B, T, 8, device=device)
    lengths = torch.tensor([120, 90, 60, 30], dtype=torch.int64, device=device)

    model = BioMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=H,
        num_gru_layers=1, dropout=0.1,
        lif_alpha=0.9, lif_threshold=1.0, lif_beta=0.5,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # ── Plain forward ──
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_batch, lengths)

    assert Y_pred.shape == (B, T)
    print(f"  X_batch shape:  {tuple(X_batch.shape)}  == (B={B}, T={T}, 8)")
    print(f"  Y_pred shape:   {tuple(Y_pred.shape)}  == (B={B}, T={T})")

    # ── Internals forward ──
    with torch.no_grad():
        Y_pred2, internals = model(X_batch, lengths, return_internals=True)

    assert Y_pred2.shape == (B, T)
    assert internals["routing_gates"].shape == (B, T, 2)
    assert internals["lif_potentials"].shape == (B, T, H)
    assert internals["lif_spikes"].shape == (B, T, H)
    assert internals["gru_hidden"].shape == (B, T, H)
    print(f"  routing_gates:  {tuple(internals['routing_gates'].shape)}")
    print(f"  lif_potentials: {tuple(internals['lif_potentials'].shape)}")
    print(f"  lif_spikes:     {tuple(internals['lif_spikes'].shape)}")
    print(f"  gru_hidden:     {tuple(internals['gru_hidden'].shape)}")

    # ── freeze_modules ──
    model.freeze_modules(["lif_cell", "router"])
    for p in model.lif_cell.parameters():
        assert not p.requires_grad
    for p in model.router.parameters():
        assert not p.requires_grad
    for p in model.sensory_encoder.parameters():
        assert p.requires_grad
    print("  freeze_modules: lif_cell + router frozen, encoder trainable")

    # ── Gradient flow (unfrozen) ──
    model2 = BioMoRCore(hidden_dim=H)
    X2 = torch.randn(2, 40, 8, requires_grad=True)
    len2 = torch.tensor([40, 20], dtype=torch.int64)
    Y2 = model2(X2, len2)
    Y2.sum().backward()
    assert X2.grad is not None
    assert X2.grad.abs().sum() > 0
    print("  Gradient flow: OK")

    print("=" * 60)
    print("All forward-pass assertions passed.")
    print("=" * 60)


if __name__ == "__main__":
    _test_forward_pass()
