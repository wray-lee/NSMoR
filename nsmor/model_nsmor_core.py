"""
NSMoR Core — Mixture-of-Recursions (MoR) neural network.

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

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ===============================================================
# 1.  Sensory Encoder
# ===============================================================

class SensoryEncoder(nn.Module):
    """
    Map raw 4-D sensory features to a hidden representation.

    ``Linear(4, hidden_dim)`` + ``LayerNorm`` + ``ReLU``

    Optionally injects Gaussian noise during training to model intrinsic
    neural variability and stochastic resonance (Gap D).
    Ref: Douglass et al. 1993, Nature 365:721-723.

    Input:  ``(B, T, 4)``
    Output: ``(B, T, H)``
    """

    def __init__(
        self,
        sensory_dim: int = 4,
        hidden_dim: int = 64,
        noise_std: float = 0.0,
    ) -> None:
        """
        Args:
            sensory_dim: Input feature dimensionality.
            hidden_dim: Hidden representation dimensionality.
            noise_std: Standard deviation of Gaussian noise injected
                during training.  Models intrinsic neural variability
                and stochastic resonance.  0 disables (backward
                compatible).  Typical: 0.01-0.1.
                Ref: Douglass et al. 1993, Nature 365:721-723.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sensory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.noise_std = noise_std

    def forward(self, sensory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensory: ``(B, T, 4)``

        Returns:
            ``(B, T, H)``
        """
        h = self.net(sensory)
        # Inject noise during training only (stochastic resonance)
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(h) * self.noise_std
            h = h + noise
        return h


# ===============================================================
# 2.  LIF (Leaky Integrate-and-Fire) RNN Cell
# ===============================================================

class LIFCell(nn.Module):
    """
    Leaky Integrate-and-Fire recurrent neuron with biophysical realism.

    Implements five biologically grounded mechanisms beyond the basic LIF:

    **1. Synaptic Delay (IIR Low-Pass Filter)**
        Input current passes through a first-order IIR filter before
        reaching the soma, modeling finite neurotransmitter diffusion
        and receptor binding time.
        Ref: Destexhe, Mainen & Sejnowski 1994, "Synaptic modeling
        of cortical dynamics", Neural Computation.

        ``I_syn[t] = alpha_syn * I_syn[t-1] + (1 - alpha_syn) * input[t]``

    **2. Absolute Refractory Period**
        After a spike, the membrane potential is clamped to ``v_rest``
        for ``abs_refract_steps`` timesteps, modeling Na+ channel
        inactivation (Hodgkin-Huxley h-gate kinetics).
        Ref: Hodgkin & Huxley 1952, J. Physiol. 117:500-544.

    **3. Relative Refractory Period**
        After the absolute period, the effective threshold decays
        exponentially from an elevated value back to baseline,
        modeling K+ delayed-rectifier hyperpolarization and the
        slow Na+ channel recovery from inactivation.
        Ref: Bean 2007, "The action potential in mammalian central
        neurons", Nature Reviews Neuroscience.

    **4. Spike-Frequency Adaptation (AdEx-style)**
        A slow adaptation current ``w`` accumulates on each spike
        and decays exponentially between spikes, modeling the
        combined effect of slow K+ M-current and Ca2+-activated K+ current.
        Ref: Brette & Gerstner 2005, Neural Computation 17:1515-1548.
        Ref: Benda & Herz 2003, Neural Computation 15:2523-2564.

    **5. Short-Term Plasticity (Tsodyks-Markram model)**
        Modulates input current via paired facilitation/depression
        dynamics.  Two state variables track synaptic efficacy:

        - ``x_resource`` (available fraction of neurotransmitter, [0,1])
        - ``u_facil`` (release probability / utilization, [0,1])

        The effective input scaling is ``stp_factor = x_resource * u_facil``.

        Discrete update order (per timestep) -- the critical aspect:
        We MUST decay first, then apply the spike-triggered jump, per the
        correct Tsodyks-Markram discretization.

        1. Inter-step decay (every timestep, regardless of spike):
           ``u_pre = u_old * exp(-dt / tau_fac)``  (facilitation decays)
           ``x_pre = 1 - (1 - x_old) * exp(-dt / tau_rec)``  (resource recovers)

        2. STP modulation applied to input current:
           ``stp_factor = x_pre * u_pre``
           ``I_raw = beta * W_in(input) * stp_factor``

        3. Spike-triggered updates (only when spike fires):
           ``x_new = x_pre - x_pre * u_pre * spike``  (depletion)
           ``u_new = u_pre + U * (1 - u_pre) * spike``  (facilitation)

        4. Non-spike timestep:
           ``x_new = x_pre, u_new = u_pre``  (no change beyond decay)

        The utilization parameter U is a LEARNABLE ``nn.Parameter``,
        constrained to (0, 1) via sigmoid.
        Ref: Tsodyks, Pawelzik & Markram 1998, Neural Computation 10:821-839.
        Ref: Markram et al. 1998, PNAS 95:5323-5328.

        STP is disabled when ``tau_fac=0`` AND ``tau_rec=0`` (default),
        in which case no extra parameters or state variables are added.

    Dynamics (per time-step *t*)::

        [STP decay if enabled]
        u_pre = u_old * exp(-1/tau_fac)
        x_pre = 1 - (1 - x_old) * exp(-1/tau_rec)
        stp_factor = x_pre * u_pre

        I_syn[t] = alpha_syn * I_syn[t-1] + (1 - alpha_syn) * beta * W_in(input[t]) * stp_factor
        theta_eff[t] = v_threshold + delta_theta * exp(-k_rel * refract_counter[t])
        V[t] = alpha * V[t-1] + I_syn[t] - w[t-1]   (clamped to v_rest if in absolute refractory)
        spike = 1 if V[t] > theta_eff[t] else 0      (suppressed if in absolute refractory)
        V[t] -= theta_eff[t] * spike                   (soft reset)
        w[t] = exp(-1/tau_w) * w[t-1] + b * spike     (adaptation update)

        [STP spike-triggered update if enabled]
        x_new = x_pre - x_pre * u_pre * spike          (depletion)
        u_new = u_pre + U * (1 - u_pre) * spike        (facilitation)

    Surrogate gradient trick::

        spike = spike_mask - sigmoid(V - theta_eff).detach() + sigmoid(V - theta_eff)

    Forward  -> ``spike_mask`` (binary 0/1, sigmoid terms cancel).
    Backward -> gradient flows only through ``sigmoid(V - theta_eff)``.

    State tuple (when STP disabled, 6 tensors)::

        (V, I_syn, refract_counter, v_threshold_eff, w_adapt, rel_refract_counter)

    State tuple (when STP enabled, 8 tensors)::

        (V, I_syn, refract_counter, v_threshold_eff, w_adapt,
         rel_refract_counter, x_resource, u_facil)

    For backward compatibility:
    - When ``state`` is a single tensor ``V``, the remaining
      components default to zero (STP defaults: x=1, u=U).
    - When ``state`` is a 4-tuple (legacy), ``w_adapt`` defaults
      to zero (no adaptation).
    - When ``state`` is a 5-tuple (legacy), ``rel_refract_counter``
      defaults to large value (baseline threshold).
    - When ``state`` is a 7-tuple (legacy STP), ``rel_refract_counter``
      defaults to large value.

    Input:  ``(B, H)`` at each step
    Output: ``(B, H)`` at each step
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        alpha: float = 0.9,
        v_threshold: float = 1.0,
        beta: float = 0.5,
        abs_refract_steps: int = 0,
        rel_refract_steps: int = 0,
        tau_syn: float = 0.0,
        v_rest: float = 0.0,
        v_reset: Optional[float] = None,
        tau_w: float = 0.0,
        b_adapt: float = 0.0,
        tau_fac: float = 0.0,
        tau_rec: float = 0.0,
        U_stp_init: float = 0.5,
        lateral_inhibition: float = 0.0,
        dendritic_tau: float = 0.0,
    ) -> None:
        """
        Args:
            hidden_dim: Dimensionality of the membrane state.
            alpha: Leak factor in (0, 1).  Higher -> slower decay.
            v_threshold: Baseline spike threshold.
            beta: Input scaling factor.
            abs_refract_steps: Number of timesteps of absolute
                refractory period after a spike (Na+ channel
                inactivation).  0 disables.
                Ref: Hodgkin & Huxley 1952.
            rel_refract_steps: Characteristic decay length (in
                timesteps) for the relative refractory threshold.
                0 disables.
                Ref: Bean 2007.
            tau_syn: Synaptic time constant in dt units.  Controls
                the IIR low-pass filter on input current.  0 bypasses
                the filter (instantaneous, backward compatible).
                Ref: Destexhe et al. 1994.
            v_rest: Resting membrane potential, used to clamp V
                during absolute refractory period.  Default 0.0
                (backward compatible).
            v_reset: Fixed reset potential after spike (standard AdEx).
                When ``None`` (default), falls back to subtracting
                ``v_thresh_new`` from the membrane (backward-compatible
                soft reset).  When set, uses ``V_reset`` as a fixed
                reset target: ``v_new = v_reset * spike_mask + v_new * (1-spike_mask)``.
                This decouples the reset from the elevated threshold
                during the relative refractory period, matching the
                standard AdEx model (Brette & Gerstner 2005).
                Typical: set to ``v_rest`` for hard reset to resting potential.
            tau_w: Adaptation time constant in dt units.  Controls
                how fast the adaptation current decays between spikes.
                0 disables adaptation (backward compatible).
                Ref: Brette & Gerstner 2005.
            b_adapt: Spike-triggered adaptation increment.
                Added to adaptation current on each spike.
                0 disables adaptation (backward compatible).
                Ref: Benda & Herz 2003.
            tau_fac: Facilitation time constant in dt units.
                Controls decay of utilization (release probability)
                between spikes.  0 disables STP (when combined with
                tau_rec=0).  Typical: 10-100 dt-units.
                Ref: Tsodyks et al. 1998.
            tau_rec: Recovery (depression) time constant in dt units.
                Controls recovery of available neurotransmitter
                resources toward 1.  0 disables STP (when combined
                with tau_fac=0).  Typical: 100-800 dt-units.
                Ref: Tsodyks et al. 1998.
            U_stp_init: Initial baseline utilization (U parameter in
                the Tsodyks-Markram model).  Only used when STP is
                enabled (tau_fac>0 AND tau_rec>0).  Stored as a
                learnable nn.Parameter constrained to (0,1) via sigmoid.
                Typical: 0.2-0.7.  Default 0.5.
                Ref: Markram et al. 1998, PNAS 95:5323-5328.
            lateral_inhibition: Strength of recurrent lateral
                inhibition between hidden units.  Models inhibitory
                interneuron pools (e.g., feedforward inhibition in
                the cricket cercal giant-fiber system).  The
                inhibitory current is computed as
                ``W_inhib @ spike_history``, where ``W_inhib`` is a
                learned weight matrix with zero diagonal (no self-
                inhibition) and negative weights constrained via
                ``-softplus``.  The ``spike_history`` is an
                exponential moving average of recent spikes with
                time constant ``tau_syn`` (reuses existing synaptic
                filter).  0 disables (backward compatible).
                Ref: Ritzmann & Camhi 1978, J. Comp. Physiol.
            dendritic_tau: Time constant for the dendritic low-pass
                filter applied to visual inputs before somatic
                integration.  Models the separate dendritic
                compartmentalization seen in LGI: wind signals arrive
                at cercal dendrites (fast, no filtering) while visual
                signals traverse optic lobe dendrites (slower, with
                temporal smoothing).  When > 0, the first
                ``sensory_dim//2`` input channels (visual) pass
                through an IIR filter with this time constant before
                reaching the soma, while the remaining channels
                (wind/kinematic) bypass it.  0 disables (backward
                compatible).
                Ref: London & Hausser 2005, Annu. Rev. Neurosci.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.v_threshold = v_threshold
        self.beta = beta
        self.abs_refract_steps = abs_refract_steps
        self.rel_refract_steps = rel_refract_steps
        self.tau_syn = tau_syn

        # CF1 fix: Membrane leak and input scaling constraints.
        # alpha in (0, 1): leak factor.  alpha=0 means instantaneous
        # decay; alpha=1 means no leak (pure integration).  Values
        # outside (0, 1) produce unstable dynamics.
        assert 0.0 < alpha < 1.0, (
            f"alpha (membrane leak) must be in (0, 1), got {alpha}"
        )
        # beta > 0: input scaling.  beta=0 means no input reaches
        # the membrane (neuron can never fire from input alone).
        assert beta > 0, (
            f"beta (input scaling) must be > 0, got {beta}"
        )

        # CF2 note: The full AdEx model includes a subthreshold
        # adaptation term: dw/dt = a(V - V_rest) - w/tau_w.
        # Our implementation only has the spike-triggered term:
        # w[t] = exp(-1/tau_w) * w[t-1] + b * spike.
        # The subthreshold term (a parameter) is omitted because:
        # 1. The cricket LGI escape circuit uses Type I excitability
        #    (no subthreshold adaptation), as shown by continuous
        #    frequency-current curves without spike-frequency adaptation
        #    at low rates.  Ref: Gabbiani et al. 1999, Nature 401:672.
        # 2. The spike-triggered term alone captures the observed
        #    spike-frequency adaptation in LGI neurons (Benda & Herz
        #    2003, Neural Computation 15:2523-2564).
        # 3. Adding the subthreshold term would require an additional
        #    parameter and change the resting dynamics.

        # CF3 fix: Parameter guard — v_rest must be below v_threshold.
        # If v_rest >= v_threshold, the neuron would fire spontaneously
        # at every step (membrane at rest already exceeds threshold),
        # making refractory periods and adaptation meaningless.
        assert v_rest < v_threshold, (
            f"v_rest ({v_rest}) must be < v_threshold ({v_threshold}). "
            f"Otherwise the neuron fires spontaneously at rest."
        )
        self.v_rest = v_rest
        # CF1 fix: Use boolean flag to track intent, not value comparison.
        # self.v_reset = v_rest when v_reset is None (backward compat).
        # self._hard_reset = True when user explicitly set v_reset.
        self._hard_reset = v_reset is not None
        self.v_reset = v_reset if v_reset is not None else v_rest
        self.tau_w = tau_w
        self.b_adapt = b_adapt
        self.tau_fac = tau_fac
        self.tau_rec = tau_rec
        self.U_stp_init = U_stp_init

        # Derived constants for relative refractory threshold elevation
        # CF3 fix: Default elevation is 30% of v_threshold, matching
        # Bean 2007 (Nature Reviews Neuroscience) which reports 20-50%
        # threshold elevation during the relative refractory period.
        # Previously _delta_theta = v_threshold (100% elevation) which
        # exceeded biological measurements.
        self._delta_theta = 0.3 * v_threshold
        if rel_refract_steps > 0:
            self._k_rel = 1.0 / rel_refract_steps
        else:
            self._k_rel = 0.0

        # Synaptic filter coefficient: alpha_syn = exp(-1/tau_syn)
        if tau_syn > 0.0:
            self._alpha_syn = torch.tensor(
                torch.exp(torch.tensor(-1.0 / tau_syn)).item()
            )
        else:
            self._alpha_syn = torch.tensor(0.0)

        # Adaptation decay coefficient: alpha_w = exp(-1/tau_w)
        if tau_w > 0.0:
            self._decay_w = torch.tensor(
                torch.exp(torch.tensor(-1.0 / tau_w)).item()
            )
        else:
            self._decay_w = torch.tensor(0.0)

        # Short-Term Plasticity (Tsodyks-Markram)
        # STP is enabled only when BOTH time constants are positive.
        # This ensures backward compatibility: defaults (tau_fac=0, tau_rec=0)
        # produce zero extra parameters and zero extra state.
        self.stp_enabled = (tau_fac > 0.0) and (tau_rec > 0.0)

        if self.stp_enabled:
            # Learnable utilization parameter U, sigmoid-constrained to (0, 1).
            # U_stp_raw is the unconstrained parameter; sigmoid(U_stp_raw) = U.
            U_clamped = max(1e-4, min(1.0 - 1e-4, U_stp_init))
            U_raw_init = math.log(U_clamped / (1.0 - U_clamped))
            self.U_stp_raw = nn.Parameter(
                torch.tensor(U_raw_init, dtype=torch.float32)
            )

            # Pre-compute decay coefficients (scalars, not parameters)
            self._decay_fac = math.exp(-1.0 / tau_fac)
            self._decay_rec = math.exp(-1.0 / tau_rec)

            assert 0.0 < self._decay_fac < 1.0, (
                f"_decay_fac={self._decay_fac} must be in (0, 1) "
                f"for tau_fac={tau_fac} > 0"
            )
            assert 0.0 < self._decay_rec < 1.0, (
                f"_decay_rec={self._decay_rec} must be in (0, 1) "
                f"for tau_rec={tau_rec} > 0"
            )

        # ── Lateral Inhibition (Gap A) ──
        # Ref: Ritzmann & Camhi 1978, J. Comp. Physiol.
        # Models feedforward/feedback inhibition between hidden units
        # via inhibitory interneuron pools.  W_inhib is learned, with
        # negative-only weights (enforced by -softplus) and zero diagonal
        # (no self-inhibition).  The inhibitory current is computed from
        # an exponential moving average of recent spike activity.
        self.lateral_inhibition = lateral_inhibition
        if lateral_inhibition > 0.0:
            # CF1 fix: Only parameterize NON-diagonal elements.
            # The diagonal mask is a registered buffer (not a parameter)
            # so it is never updated by the optimizer.
            # Ref: Ritzmann & Camhi 1978, J. Comp. Physiol.
            self.register_buffer(
                '_inhib_diag_mask',
                (1.0 - torch.eye(hidden_dim, dtype=torch.float32)),
            )
            # Raw weight matrix (unconstrained); actual weights = -softplus(raw) * mask
            self._W_inhib_raw = nn.Parameter(
                torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32)
            )

            # Spike history buffer (exponential moving average)
            # Reuses tau_syn if available, else a fixed 5-step window
            self._inhib_tau = max(tau_syn, 5.0)
            self._decay_inhib = math.exp(-1.0 / self._inhib_tau)

            # CF4 note: _spike_history is NOT registered as a buffer because
            # its shape is (B, H) — batch-dependent and not known at init.
            # It is a volatile computation cache, not a learned parameter.
            # It does NOT persist across save/load (state_dict).
            # This is intentional: the spike history is re-initialized to
            # zeros at the start of each sequence via init_state().
            # Same applies to _dendritic_state in NSMoRCore.

        # ── Dendritic Compartmentalization (Gap B) ──
        # Ref: London & Hausser 2005, Annu. Rev. Neurosci.
        # Models separate dendritic processing for visual vs wind inputs.
        # Visual inputs (first half) pass through IIR filter; wind inputs
        # (second half) reach soma directly.
        self.dendritic_tau = dendritic_tau
        self._dendritic_enabled = dendritic_tau > 0.0
        if self._dendritic_enabled:
            self._alpha_dend = math.exp(-1.0 / dendritic_tau)

        # Input projection
        self.W_in = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(
        self,
        input_t: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Advance the LIF state by one time-step.

        Args:
            input_t: ``(B, H)`` — sensory encoding at time *t*.
            state: Tuple of ``(B, H)`` tensors.
                If ``None``, initializes to defaults (backward compatible).
                For backward compatibility, also accepts:
                - A single ``(B, H)`` tensor (treated as V only).
                - A 4-tuple ``(V, I_syn, refract_counter, v_threshold_eff)``
                  (w_adapt defaults to zero).
                - A 5-tuple ``(V, I_syn, refract_counter, v_threshold_eff, w_adapt)``
                  (STP variables default: x=1, u=U when STP enabled).
                - A 7-tuple with STP state when STP enabled.

        Returns:
            ``(spike, new_state)`` where:
            - spike: ``(B, H)`` — binary spike signal (surrogate gradient)
            - new_state: tuple of 5 ``(B, H)`` tensors (STP disabled)
              or 7 ``(B, H)`` tensors (STP enabled)
        """
        B, H = input_t.shape
        device = input_t.device

        # ── Shape assertion on input ──
        assert input_t.shape == (B, H), (
            f"input_t shape {tuple(input_t.shape)} != (B={B}, H={H})"
        )

        # ── Unpack state (backward compatible, CF3: +rel_refract_counter) ──
        # State tuple sizes (current):
        #   No STP: 6 = (V, I_syn, refract_counter, v_thresh_eff, w_adapt, rel_refract_counter)
        #   STP:    8 = (V, I_syn, refract_counter, v_thresh_eff, w_adapt, rel_refract_counter, x_resource, u_facil)
        # Backward compatible with old sizes:
        #   4-tuple (legacy), 5-tuple (no STP, old), 7-tuple (STP, old)
        # Default: large value so exp(-k_rel * large) ≈ 0 → baseline threshold
        _large_rel = float(10 * max(self.rel_refract_steps, 1))
        rel_refract_counter = torch.full((B, H), _large_rel, device=device)
        if state is None:
            v, i_syn, refract_counter, v_thresh_eff, w_adapt = self.init_state_5(B, device)
            if self.stp_enabled:
                x_resource = torch.ones(B, H, device=device)
                u_facil = torch.full(
                    (B, H),
                    torch.sigmoid(self.U_stp_raw).item(),
                    device=device,
                )
        elif isinstance(state, tuple):
            if self.stp_enabled and len(state) == 8:
                (v, i_syn, refract_counter, v_thresh_eff, w_adapt,
                 rel_refract_counter, x_resource, u_facil) = state
            elif self.stp_enabled and len(state) == 7:
                # Backward compat: old 7-tuple STP (no rel_refract_counter)
                v, i_syn, refract_counter, v_thresh_eff, w_adapt, x_resource, u_facil = state
            elif len(state) == 6:
                # New 6-tuple (no STP, with rel_refract_counter)
                v, i_syn, refract_counter, v_thresh_eff, w_adapt, rel_refract_counter = state
                if self.stp_enabled:
                    x_resource = torch.ones(B, H, device=device)
                    u_facil = torch.full(
                        (B, H),
                        torch.sigmoid(self.U_stp_raw).item(),
                        device=device,
                    )
            elif len(state) == 5:
                v, i_syn, refract_counter, v_thresh_eff, w_adapt = state
                if self.stp_enabled:
                    x_resource = torch.ones(B, H, device=device)
                    u_facil = torch.full(
                        (B, H),
                        torch.sigmoid(self.U_stp_raw).item(),
                        device=device,
                    )
            elif len(state) == 4:
                # Backward compat: legacy 4-tuple, no adaptation
                v, i_syn, refract_counter, v_thresh_eff = state
                w_adapt = torch.zeros(B, H, device=device)
                if self.stp_enabled:
                    x_resource = torch.ones(B, H, device=device)
                    u_facil = torch.full(
                        (B, H),
                        torch.sigmoid(self.U_stp_raw).item(),
                        device=device,
                    )
            else:
                expected = "6, 8" if self.stp_enabled else "4, 5, or 6"
                raise ValueError(
                    f"state tuple must have {expected} elements, got {len(state)}"
                )
        else:
            # Backward compat: single tensor = membrane potential only
            v = state
            i_syn = torch.zeros(B, H, device=device)
            refract_counter = torch.zeros(B, H, device=device)
            v_thresh_eff = torch.full(
                (B, H), self.v_threshold, device=device,
            )
            w_adapt = torch.zeros(B, H, device=device)
            if self.stp_enabled:
                x_resource = torch.ones(B, H, device=device)
                u_facil = torch.full(
                    (B, H),
                    torch.sigmoid(self.U_stp_raw).item(),
                    device=device,
                )

        # ── Assertions: core state dimensions ──
        assert v.shape == (B, H), (
            f"v shape {tuple(v.shape)} != (B={B}, H={H})"
        )
        assert i_syn.shape == (B, H), (
            f"i_syn shape {tuple(i_syn.shape)} != (B={B}, H={H})"
        )
        assert refract_counter.shape == (B, H), (
            f"refract_counter shape {tuple(refract_counter.shape)} != (B={B}, H={H})"
        )
        assert v_thresh_eff.shape == (B, H), (
            f"v_thresh_eff shape {tuple(v_thresh_eff.shape)} != (B={B}, H={H})"
        )
        assert w_adapt.shape == (B, H), (
            f"w_adapt shape {tuple(w_adapt.shape)} != (B={B}, H={H})"
        )

        # ── 1. Short-Term Plasticity: inter-step decay (FIRST) ──
        # Critical: decay must happen BEFORE the spike-triggered jump.
        # This is the correct Tsodyks-Markram discretization.
        # Ref: Tsodyks, Pawelzik & Markram 1998, Neural Computation.
        if self.stp_enabled:
            # STP state assertions
            assert x_resource.shape == (B, H), (
                f"x_resource shape {tuple(x_resource.shape)} != (B={B}, H={H})"
            )
            assert u_facil.shape == (B, H), (
                f"u_facil shape {tuple(u_facil.shape)} != (B={B}, H={H})"
            )

            # Inter-step decay (every timestep, regardless of spike)
            # u_pre = u_old * exp(-dt / tau_fac)  -- facilitation decays
            # x_pre = 1 - (1 - x_old) * exp(-dt / tau_rec)  -- resource recovers
            u_pre = u_facil * self._decay_fac                # (B, H)
            x_pre = 1.0 - (1.0 - x_resource) * self._decay_rec  # (B, H)

            # Clamp to prevent float drift.
            # CF2 fix: use min=1e-6 (not 0.0) consistently with
            # post-spike clamp to avoid gradient dead zone at exact zero.
            u_pre = u_pre.clamp(min=1e-6, max=1.0)           # (B, H)
            x_pre = x_pre.clamp(min=1e-6, max=1.0)           # (B, H)

            # STP modulation factor
            stp_factor = x_pre * u_pre                       # (B, H)
            assert stp_factor.shape == (B, H), (
                f"stp_factor shape {tuple(stp_factor.shape)} != (B={B}, H={H})"
            )
        else:
            # STP disabled: no modulation (backward compatible)
            stp_factor = 1.0  # scalar, broadcasts to (B, H)

        # ── 2. Input projection ──
        # Note: dendritic compartmentalization (CF2 fix) is now applied
        # to raw sensory channels in NSMoRCore._run_lif_path BEFORE
        # SensoryEncoder, preserving modality isolation.
        projected_input = self.W_in(input_t)                  # (B, H)

        # ── 3. Synaptic delay: IIR low-pass filter with STP ──
        # Ref: Destexhe et al. 1994.
        raw_input = self.beta * projected_input * stp_factor  # (B, H)
        alpha_syn = self._alpha_syn.to(device)
        i_syn_new = alpha_syn * i_syn + (1.0 - alpha_syn) * raw_input  # (B, H)

        # ── 4. Absolute refractory: clamp membrane ──
        in_abs_refract = (refract_counter > 0).float()       # (B, H) 0/1

        # ── 5. Relative refractory: elevated threshold ──
        # Ref: Bean 2007, Nature Reviews Neuroscience.
        # Counter semantics: rel_refract_counter counts UP from 0
        # (time since last spike).  At spike: counter = 0 (threshold
        # highest).  Each step without spike: counter += 1 (threshold
        # decays toward baseline).
        #
        # Formula: theta = v_threshold + delta_theta * exp(-k_rel * counter)
        #   counter=0 (just spiked): exp(0)=1.0 → theta = v_threshold + delta_theta
        #   counter=5 (5 steps ago): exp(-1)=0.37 → theta ≈ v_threshold + 0.11
        #   counter→∞ (long ago):    exp→0 → theta → v_threshold (baseline)
        #
        # This matches Bean 2007: threshold highest immediately after spike,
        # decaying exponentially back to baseline.
        if self._k_rel > 0:
            v_thresh_new = self.v_threshold + self._delta_theta * torch.exp(
                -self._k_rel * rel_refract_counter
            )                                                # (B, H)
        else:
            v_thresh_new = torch.full(
                (B, H), self.v_threshold, device=device,
            )                                                # (B, H)

        # ── 6. Membrane integration (clamped in abs refractory) ──
        v_new = self.alpha * v + i_syn_new - w_adapt         # (B, H)
        v_new = v_new * (1.0 - in_abs_refract) + self.v_rest * in_abs_refract  # (B, H)

        # ── 6b. Lateral inhibition (Gap A) ──
        # Ref: Ritzmann & Camhi 1978, J. Comp. Physiol.
        # Subtracts a weighted sum of recent population spike activity
        # from the membrane potential, modeling inhibitory interneuron
        # pools.  W_inhib has negative-only weights (enforced by
        # -softplus) and zero diagonal (no self-inhibition).
        if self.lateral_inhibition > 0.0:
            # Retrieve or initialize spike history (EMA of recent spikes)
            spike_hist = getattr(self, '_spike_history', None)
            if spike_hist is None or spike_hist.shape != (B, H):
                spike_hist = torch.zeros(B, H, device=device)
            # Update spike history (will be finalized after spike detection)
            # For now, use the previous step's spike history
            # CF1 fix: Apply diagonal mask to enforce zero self-inhibition.
            # The mask is a registered buffer, so diagonal elements are
            # permanently zero regardless of gradient updates.
            W_inhib = -F.softplus(self._W_inhib_raw) * self._inhib_diag_mask  # (H, H)
            # Inhibitory current: spike_history @ W_inhib^T
            # (B, H) @ (H, H) -> (B, H)
            inhib_current = spike_hist @ W_inhib.t()           # (B, H) all <= 0
            v_new = v_new + self.lateral_inhibition * inhib_current  # (B, H)
            assert inhib_current.shape == (B, H), (
                f"inhib_current shape {tuple(inhib_current.shape)} != (B={B}, H={H})"
            )

        # ── 7. Spike detection (suppressed in abs refractory) ──
        raw_spike = (v_new > v_thresh_new).float()           # (B, H) binary
        spike_mask = raw_spike * (1.0 - in_abs_refract)      # (B, H) binary
        # Surrogate gradient (straight-through estimator)
        sig = torch.sigmoid(v_new - v_thresh_new)            # (B, H) smooth
        spike = spike_mask - sig.detach() + sig              # (B, H) binary fwd, smooth bwd

        # ── 7b. Update spike history for lateral inhibition ──
        # CF2 note: .detach() implements TBPTT-1 (truncated backpropagation
        # through time with truncation length 1).  The optimizer sees the
        # inhibitory current from the PREVIOUS step's spike history, but
        # gradients do NOT flow through the spike history recurrence.
        # This means the optimizer cannot learn temporal accumulation
        # patterns in inhibition (e.g., "inhibit more after bursts").
        # This is a deliberate trade-off: full BPTT through the EMA
        # would require storing the entire spike history computation graph,
        # increasing memory by O(T).  TBPTT-1 is sufficient for learning
        # the instantaneous inhibitory weight matrix W_inhib.
        # Ref: Williams & Zipser 1989, Neural Computation (truncated BPTT).
        if self.lateral_inhibition > 0.0:
            decay_inhib = self._decay_inhib
            spike_hist_new = decay_inhib * spike_hist + (1.0 - decay_inhib) * spike_mask
            self._spike_history = spike_hist_new.detach()    # TBPTT-1

        # ── 8. Reset ──
        # CF1 fix: Use _hard_reset flag (not value comparison) to decide.
        # When _hard_reset=True, use fixed v_reset (standard AdEx).
        # When _hard_reset=False, use backward-compatible soft reset.
        #
        # CF4 note: The soft reset subtracts v_thresh_new (which may be
        # elevated during relative refractory), NOT a fixed baseline.
        # This is a DELIBERATE design choice, not a standard AdEx behavior.
        # Biological motivation: during the relative refractory period,
        # Na+ channels are partially inactivated and K+ channels are
        # open, so the effective reset is deeper (more hyperpolarized)
        # than at rest.  This models the observed phenomenon where
        # post-spike membrane potential is lower during relative
        # refractory than after a spike at rest.
        # Ref: Bean 2007, Nature Reviews Neuroscience (Fig. 2).
        # To use standard AdEx reset (fixed voltage), set v_reset explicitly.
        if self._hard_reset:
            v_new = v_new * (1.0 - spike_mask) + self.v_reset * spike_mask
        else:
            v_new = v_new - spike_mask * v_thresh_new

        # ── 9. Spike-frequency adaptation update ──
        decay_w = self._decay_w.to(device)
        w_new = decay_w * w_adapt + self.b_adapt * spike_mask  # (B, H)

        # ── 10. STP spike-triggered update (AFTER spike detection) ──
        # Critical: this happens AFTER we know spike_mask.
        # The spike-triggered jump is the second step of the TM discretization.
        if self.stp_enabled:
            U = torch.sigmoid(self.U_stp_raw)                # scalar in (0, 1)

            # Spike-triggered depletion: x loses u_pre * x_pre fraction
            x_new = x_pre - x_pre * u_pre * spike_mask      # (B, H)

            # Spike-triggered facilitation: u jumps toward 1
            u_new = u_pre + U * (1.0 - u_pre) * spike_mask  # (B, H)

            # Clamp to prevent float drift.
            # CF2 fix: use clamp(min, max) which is equivalent to max/min
            # for gradient behavior (both have zero gradient at boundary).
            # The previous torch.max/min with torch.tensor() allocated
            # 4 scalar tensors per step (4T allocations).  clamp() is
            # a single fused op and the eps constant is a Python float
            # (no tensor allocation).
            # Note: clamp gradient at boundary is zero for the clamped
            # dimension, same as max/min.  This is unavoidable with any
            # boundary enforcement.
            x_new = x_new.clamp(min=1e-6, max=1.0)
            u_new = u_new.clamp(min=1e-6, max=1.0)

            # STP state assertions after update
            assert x_new.shape == (B, H), (
                f"x_new shape {tuple(x_new.shape)} != (B={B}, H={H})"
            )
            assert u_new.shape == (B, H), (
                f"u_new shape {tuple(u_new.shape)} != (B={B}, H={H})"
            )

        # ── 11. Update refractory counters ──
        # Absolute refractory counter (only when abs_refract_steps > 0)
        if self.abs_refract_steps > 0:
            refract_new = torch.where(
                spike_mask.bool(),
                torch.tensor(float(self.abs_refract_steps), device=device),
                torch.clamp(refract_counter - 1.0, min=0.0),
            )                                                # (B, H)
        else:
            refract_new = refract_counter                    # (B, H) unchanged

        # Relative refractory counter: reset to 0 on spike (threshold
        # highest), increment each step without spike (threshold decays).
        if self._k_rel > 0:
            rel_refract_new = torch.where(
                spike_mask.bool(),
                torch.tensor(0.0, device=device),
                rel_refract_counter + 1.0,
            )                                                # (B, H)
        else:
            rel_refract_new = rel_refract_counter            # (B, H) unchanged

        # ── 12. Pack state (CF3: includes rel_refract_counter) ──
        if self.stp_enabled:
            new_state = (v_new, i_syn_new, refract_new, v_thresh_new, w_new,
                         rel_refract_new, x_new, u_new)
        else:
            new_state = (v_new, i_syn_new, refract_new, v_thresh_new, w_new,
                         rel_refract_new)

        return spike, new_state

    def init_state_5(
        self, batch_size: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the 5-element core state tuple (no STP, no rel_refract).

        Returns:
            ``(V, I_syn, refract_counter, v_threshold_eff, w_adapt)``
            each ``(B, H)``.
        """
        v = torch.full(
            (batch_size, self.hidden_dim), self.v_rest, device=device,
        )
        i_syn = torch.zeros(batch_size, self.hidden_dim, device=device)
        refract_counter = torch.zeros(batch_size, self.hidden_dim, device=device)
        v_thresh_eff = torch.full(
            (batch_size, self.hidden_dim), self.v_threshold, device=device,
        )
        w_adapt = torch.zeros(batch_size, self.hidden_dim, device=device)
        return v, i_syn, refract_counter, v_thresh_eff, w_adapt

    def init_state_6(
        self, batch_size: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the 6-element core state tuple (with rel_refract_counter, no STP).

        Returns:
            ``(V, I_syn, refract_counter, v_threshold_eff, w_adapt, rel_refract_counter)``
            each ``(B, H)``.
        """
        v, i_syn, refract_counter, v_thresh_eff, w_adapt = self.init_state_5(batch_size, device)
        # Initialize to large value so exp(-k_rel * large) ≈ 0 and
        # threshold starts at baseline (not elevated).  The counter
        # represents "time since last spike" — at init, no spike has
        # occurred, so the effective time is very large.
        _large = float(10 * max(self.rel_refract_steps, 1))
        rel_refract_counter = torch.full(
            (batch_size, self.hidden_dim), _large, device=device,
        )
        return v, i_syn, refract_counter, v_thresh_eff, w_adapt, rel_refract_counter

    def init_state(
        self, batch_size: int, device: torch.device,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Return initial state tuple for the LIF cell.

        Returns 6-tuple when STP is disabled, 8-tuple when STP is enabled.
        (CF3: includes rel_refract_counter at position 5.)
        Also resets cached spike history for lateral inhibition.
        """
        # Reset cached states for new sequence
        # CF2 fix: Reset _dendritic_state to prevent batch N's state
        # from leaking into batch N+1 when batch sizes match.
        if self._dendritic_enabled:
            self._dendritic_state = None
        if self.lateral_inhibition > 0.0:
            self._spike_history = None

        v, i_syn, refract_counter, v_thresh_eff, w_adapt = self.init_state_5(
            batch_size, device,
        )
        # Initialize to large value so threshold starts at baseline
        _large_rel = float(10 * max(self.rel_refract_steps, 1))
        rel_refract_counter = torch.full(
            (batch_size, self.hidden_dim), _large_rel, device=device,
        )
        if self.stp_enabled:
            x_resource = torch.ones(batch_size, self.hidden_dim, device=device)
            u_facil = torch.full(
                (batch_size, self.hidden_dim),
                torch.sigmoid(self.U_stp_raw).item(),
                device=device,
            )
            return (v, i_syn, refract_counter, v_thresh_eff, w_adapt,
                    rel_refract_counter, x_resource, u_facil)
        return v, i_syn, refract_counter, v_thresh_eff, w_adapt, rel_refract_counter


# ===============================================================
# 3.  GRU Unit (packed-sequence wrapper)
# ===============================================================

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
        h0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, H)``
            lengths: ``(B,)`` — true sequence lengths.
            h0: ``(num_layers, B, H)`` — optional initial hidden state.

        Returns:
            ``(B, T, H)``
        """
        x = x.contiguous()
        if h0 is not None:
            h0 = h0.contiguous()

        self.gru.flatten_parameters()

        lengths_cpu = lengths.clamp(min=1).cpu().contiguous()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed, h0)
        out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.shape[1],
        )
        return out


# ===============================================================
# 4.  MoR Router (Causal Inference Gate)
# ===============================================================

class MoRRouter(nn.Module):
    """
    Per-time-step routing network that blends LIF and GRU outputs.

    Input:  ``(B, H + M)`` at each step
    Output: ``(B, 2)`` — independent routing weights in [0, 1]
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
            ``(B, 2)`` — ``[g_lif, g_gru]`` independently in [0, 1].
        """
        combined = torch.cat([e_sensory, mcmc_prior], dim=-1)
        logits = self.gate(combined)
        return torch.sigmoid(logits)


# ===============================================================
# 5.  Direction Head (Decoder)
# ===============================================================

class DirectionHead(nn.Module):
    """
    Final decoder: ``LayerNorm -> ReLU -> Dropout -> Linear(H, 1)``.

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


# ===============================================================
# 6.  NSMoR Core Network
# ===============================================================

_FREEZABLE_MODULES = frozenset({
    "sensory_encoder",
    "lif_cell",
    "gru_unit",
    "router",
    "direction_head",
})


class NSMoRCore(nn.Module):
    """
    Mixture-of-Recursions (MoR) — dual-pathway recurrent network.

    Architecture::

        X_batch --+-- Sensory_X [B,T,4] -> SensoryEncoder -> E_sensory [B,T,H]
                   |                                              |
                   |                                    +--- LIF Path (step-by-step)
                   |                                    |    -> Out_lif  [B,T,H]
                   |                                    |
                   |    MCMC_Prior [B,T,4] -> MoR Router -> [g_lif, g_gru] [B,T,2]
                   |                                    |
                   |                                    +--- GRU Path (packed)
                   |                                         -> Out_gru  [B,T,H]
                   |
                   +-- Integration: H = g_lif*Out_lif + g_gru*Out_gru  [B,T,H]
                                              |
                                    DirectionHead(H, 1) -> Y_pred [B,T]
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
        lif_abs_refract_steps: int = 0,
        lif_rel_refract_steps: int = 0,
        lif_tau_syn: float = 0.0,
        lif_v_rest: float = 0.0,
        lif_v_reset: Optional[float] = None,
        lif_tau_w: float = 0.0,
        lif_b_adapt: float = 0.0,
        lif_tau_fac: float = 0.0,
        lif_tau_rec: float = 0.0,
        lif_U_stp_init: float = 0.5,
        lif_lateral_inhibition: float = 0.0,
        lif_dendritic_tau: float = 0.0,
        gru_neuromod_gain: float = 0.0,
        sensory_noise_std: float = 0.0,
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
            lif_abs_refract_steps: LIF absolute refractory period
                in timesteps. 0 disables.
            lif_rel_refract_steps: LIF relative refractory decay
                length in timesteps. 0 disables.
            lif_tau_syn: LIF synaptic time constant in dt units.
                0 bypasses.
            lif_v_rest: LIF resting membrane potential. Default 0.0.
            lif_v_reset: LIF fixed reset potential after spike (standard
                AdEx).  When ``None`` (default), uses backward-compatible
                soft reset (subtract threshold).  When set, uses hard
                reset to this voltage.  Typical: set to ``lif_v_rest``.
            lif_tau_w: LIF adaptation time constant in dt units.
                0 disables (backward compatible).
            lif_b_adapt: LIF spike-triggered adaptation increment.
                0 disables (backward compatible).
            lif_tau_fac: LIF STP facilitation time constant in dt units.
                0 disables STP (when combined with lif_tau_rec=0).
            lif_tau_rec: LIF STP recovery time constant in dt units.
                0 disables STP (when combined with lif_tau_fac=0).
            lif_U_stp_init: LIF STP baseline utilization. Only used
                when STP is enabled. Default 0.5.
            lif_lateral_inhibition: Strength of recurrent lateral
                inhibition in the LIF pathway.  0 disables (backward
                compatible).  Ref: Ritzmann & Camhi 1978.
            lif_dendritic_tau: Time constant for dendritic low-pass
                filter on visual inputs.  0 disables (backward
                compatible).  Ref: London & Hausser 2005.
            gru_neuromod_gain: Strength of neuromodulatory (octopamine-
                like) gain scaling on the GRU pathway.  When > 0, a
                learnable arousal signal modulates GRU hidden states
                via multiplicative scaling.  The arousal signal is
                computed from MCMC prior entropy (high entropy =
                uncertain stimulus = high arousal).  0 disables
                (backward compatible).
                Ref: Rillich & Stevenson 2011, PLOS ONE.
            sensory_noise_std: Standard deviation of Gaussian noise
                injected into the sensory encoding during training.
                Models intrinsic neural variability and stochastic
                resonance.  0 disables (backward compatible).
                Ref: Douglass et al. 1993, Nature 365:721-723.
        """
        super().__init__()
        self.sensory_dim = sensory_dim
        self.mcmc_dim = mcmc_dim
        self.hidden_dim = hidden_dim

        # Named sub-modules (white-box)
        self.sensory_encoder = SensoryEncoder(sensory_dim, hidden_dim, sensory_noise_std)
        self.lif_cell = LIFCell(
            hidden_dim, lif_alpha, lif_threshold, lif_beta,
            abs_refract_steps=lif_abs_refract_steps,
            rel_refract_steps=lif_rel_refract_steps,
            tau_syn=lif_tau_syn,
            v_rest=lif_v_rest,
            v_reset=lif_v_reset,
            tau_w=lif_tau_w,
            b_adapt=lif_b_adapt,
            tau_fac=lif_tau_fac,
            tau_rec=lif_tau_rec,
            U_stp_init=lif_U_stp_init,
            lateral_inhibition=lif_lateral_inhibition,
            dendritic_tau=lif_dendritic_tau,
        )
        self.gru_unit = GRUUnit(hidden_dim, num_gru_layers, dropout)
        self.router = MoRRouter(hidden_dim, mcmc_dim)
        self.direction_head = DirectionHead(hidden_dim, dropout)

        # ── Neuromodulatory gain for GRU pathway (Gap C) ──
        # Ref: Rillich & Stevenson 2011, PLOS ONE.
        # Octopamine-like arousal modulation: higher MCMC entropy
        # (uncertain stimulus) -> higher arousal -> amplified GRU gain.
        self.gru_neuromod_gain = gru_neuromod_gain
        if gru_neuromod_gain > 0.0:
            # Learnable gain scale: maps entropy scalar to gain multiplier
            self._gain_scale = nn.Parameter(torch.tensor(0.0))
            self._gain_bias = nn.Parameter(torch.tensor(1.0))

    # -- Public API -------------------------------------------------

    def forward(
        self,
        X_batch: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_internals: bool = False,
        override_gates: Optional[Dict[str, float]] = None,
        states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]] | Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            X_batch: ``(B, T, 8)`` — padded feature tensor.
            lengths: ``(B,)`` — true (unpadded) sequence lengths.
            return_internals: If ``True``, return internals dict.
            override_gates: Optional dict for in-silico lesioning.
            states: Optional dict of recurrent states for autoregressive mode.

        Returns:
            If ``return_internals=False`` and ``states=None``:
                ``Y_pred``: ``(B, T)``
            If ``return_internals=True`` and ``states=None``:
                ``(Y_pred, internals)``
            If ``states`` is not None:
                ``(Y_pred, internals, states_out)``
        """
        X_batch = X_batch.contiguous()
        lengths = lengths.contiguous()

        expected_dim = self.sensory_dim + self.mcmc_dim
        if X_batch.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected feature dim {expected_dim}, got {X_batch.shape[-1]}"
            )

        B, T, _ = X_batch.shape
        device = X_batch.device

        # Input shape assertions
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

        # Step 1: Unpack input
        sensory_x = X_batch[:, :, :self.sensory_dim]
        mcmc_prior = X_batch[:, :, self.sensory_dim:]

        assert sensory_x.shape == (B, T, self.sensory_dim), (
            f"sensory_x shape {tuple(sensory_x.shape)} != "
            f"(B={B}, T={T}, D={self.sensory_dim})"
        )
        assert mcmc_prior.shape == (B, T, self.mcmc_dim), (
            f"mcmc_prior shape {tuple(mcmc_prior.shape)} != "
            f"(B={B}, T={T}, M={self.mcmc_dim})"
        )

        # Step 2: Dendritic compartmentalization (Gap B) — CF2 fix
        # Ref: London & Hausser 2005, Annu. Rev. Neurosci.
        # Apply IIR filter to raw visual channels BEFORE encoding.
        # This preserves modality isolation: visual signals are filtered
        # through slow dendrites, wind/kinematic signals bypass directly.
        if self.lif_cell._dendritic_enabled:
            D = self.sensory_dim
            half_d = D // 2
            alpha_dend = self.lif_cell._alpha_dend
            # Retrieve or initialize dendritic state in sensory space
            # CF4 fix: only store visual IIR state (B, half_d), not full (B, D)
            dend_state = getattr(self.lif_cell, '_dendritic_state', None)
            if dend_state is None or dend_state.shape != (B, half_d):
                dend_state = torch.zeros(B, half_d, device=device)
            # Visual channels: IIR filter
            visual_raw = sensory_x[:, :, :half_d]              # (B, T, D//2)
            wind_raw = sensory_x[:, :, half_d:]                # (B, T, D//2)

            # CF6 fix: Use gradient checkpointing to reduce O(T) memory.
            # Process the IIR filter in segments.  Within each segment,
            # torch.utils.checkpoint discards intermediate activations
            # and recomputes them during backward, reducing memory from
            # O(T) to O(sqrt(T)) while preserving correct gradients.
            #
            # CF4 note: alpha_dend is a Python float (from math.exp),
            # not a tensor.  This means it is captured by closure in
            # _iir_segment and is NOT part of the computation graph.
            # If alpha_dend were changed to a learnable nn.Parameter,
            # checkpoint would use the value at graph construction time,
            # not the current value.  The fix would be to pass alpha_dend
            # as an explicit argument to _iir_segment (using *args pattern).
            # Currently this is not an issue because alpha_dend is constant.
            _SEG_LEN = 32  # segment length for checkpointing

            def _iir_segment(
                seg_input: torch.Tensor, seg_state: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                """Process one IIR segment: seg_input (B, seg_len, half_d) -> output."""
                seg_len = seg_input.shape[1]
                seg_out = torch.zeros_like(seg_input)
                s = seg_state
                for t_s in range(seg_len):
                    s = alpha_dend * s + (1.0 - alpha_dend) * seg_input[:, t_s, :]
                    seg_out[:, t_s, :] = s
                return seg_out, s

            # Process in checkpointed segments
            dend_visual = torch.zeros_like(visual_raw)
            for seg_start in range(0, T, _SEG_LEN):
                seg_end = min(seg_start + _SEG_LEN, T)
                seg_input = visual_raw[:, seg_start:seg_end, :]

                if self.training and seg_input.requires_grad:
                    # Checkpoint: discard intermediate activations, recompute in backward
                    seg_out, dend_state = torch.utils.checkpoint.checkpoint(
                        _iir_segment, seg_input, dend_state,
                        use_reentrant=False,
                    )
                else:
                    seg_out, dend_state = _iir_segment(seg_input, dend_state)

                dend_visual[:, seg_start:seg_end, :] = seg_out

            # Cache for next forward call (detach: persistent state
            # is NOT part of the current computation graph)
            self.lif_cell._dendritic_state = dend_state.detach()
            sensory_x = torch.cat([dend_visual, wind_raw], dim=-1)  # (B, T, D)

        # Step 2c: Encode sensory input
        e_sensory = self.sensory_encoder(sensory_x)

        assert e_sensory.shape == (B, T, self.hidden_dim), (
            f"e_sensory shape {tuple(e_sensory.shape)} != "
            f"(B={B}, T={T}, H={self.hidden_dim})"
        )

        # Step 2b: Extract initial states (autoregressive mode, CF3: +rel_refract)
        lif_state0: Optional[Tuple[torch.Tensor, ...]] = None
        gru_h0: Optional[torch.Tensor] = None
        if states is not None:
            lif_v0 = states.get("lif_v", None)
            lif_i_syn0 = states.get("lif_i_syn", None)
            lif_refract0 = states.get("lif_refract", None)
            lif_w_adapt0 = states.get("lif_w_adapt", None)
            lif_rel_refract0 = states.get("lif_rel_refract", None)
            if lif_v0 is not None:
                _zeros = torch.zeros_like(lif_v0)
                # CF1 fix: default rel_refract_counter to large value (not zero).
                # counter=0 means "just spiked" (threshold elevated to max).
                # counter=large means "no recent spikes" (threshold at baseline).
                _large_rel = float(10 * max(self.lif_cell.rel_refract_steps, 1))
                _rel_refract_default = torch.full_like(lif_v0, _large_rel)
                # Build state tuple for LIFCell
                if self.lif_cell.stp_enabled:
                    lif_x_resource0 = states.get("lif_x_resource", None)
                    lif_u_facil0 = states.get("lif_u_facil", None)
                    lif_state0 = (
                        lif_v0,
                        lif_i_syn0 if lif_i_syn0 is not None else _zeros,
                        lif_refract0 if lif_refract0 is not None else _zeros,
                        torch.full_like(lif_v0, self.lif_cell.v_threshold),
                        lif_w_adapt0 if lif_w_adapt0 is not None else _zeros,
                        lif_rel_refract0 if lif_rel_refract0 is not None else _rel_refract_default,
                        lif_x_resource0 if lif_x_resource0 is not None else torch.ones_like(lif_v0),
                        lif_u_facil0 if lif_u_facil0 is not None else torch.full(
                            (B, self.hidden_dim),
                            torch.sigmoid(self.lif_cell.U_stp_raw).item(),
                            device=lif_v0.device,
                        ),
                    )
                else:
                    lif_state0 = (
                        lif_v0,
                        lif_i_syn0 if lif_i_syn0 is not None else _zeros,
                        lif_refract0 if lif_refract0 is not None else _zeros,
                        torch.full_like(lif_v0, self.lif_cell.v_threshold),
                        lif_w_adapt0 if lif_w_adapt0 is not None else _zeros,
                        lif_rel_refract0 if lif_rel_refract0 is not None else _rel_refract_default,
                    )
            gru_h0 = states.get("gru_h", None)

            # Restore dendritic compartment cache when available
            if self.lif_cell._dendritic_enabled:
                dend_state_in = states.get("lif_dendritic_state", None)
                if dend_state_in is not None:
                    self.lif_cell._dendritic_state = dend_state_in

            # Restore lateral inhibition spike history when available
            if self.lif_cell.lateral_inhibition > 0.0:
                spike_hist_in = states.get("lif_spike_history", None)
                if spike_hist_in is not None:
                    self.lif_cell._spike_history = spike_hist_in

        # Step 3: Path A -- LIF (step-by-step, respects padding)
        (out_lif, lif_potentials, lif_spikes, lif_thresholds,
         lif_v_final, lif_i_syn_final, lif_refract_final,
         lif_w_adapt_final, lif_rel_refract_final,
         lif_x_resource_final, lif_u_facil_final) = self._run_lif_path(
            e_sensory, lengths, lif_state0=lif_state0,
        )

        # LIF output shape assertions
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
        assert lif_thresholds.shape == (B, T, self.hidden_dim), (
            f"lif_thresholds shape {tuple(lif_thresholds.shape)} != "
            f"(B={B}, T={T}, H={self.hidden_dim})"
        )
        assert lif_v_final.shape == (B, self.hidden_dim), (
            f"lif_v_final shape {tuple(lif_v_final.shape)} != (B={B}, H={self.hidden_dim})"
        )

        # Step 4: Path B -- GRU (packed, efficient)
        out_gru = self.gru_unit(e_sensory, lengths, h0=gru_h0)

        assert out_gru.shape == (B, T, self.hidden_dim), (
            f"out_gru shape {tuple(out_gru.shape)} != (B={B}, T={T}, H={self.hidden_dim})"
        )

        # Step 4b: Neuromodulatory gain on GRU pathway (Gap C)
        # Ref: Rillich & Stevenson 2011, PLOS ONE.
        # Computes an arousal signal from MCMC prior entropy:
        # high entropy (uncertain stimulus) -> high arousal -> amplified GRU.
        # gain = sigmoid(gain_scale * entropy + gain_bias)
        # out_gru = out_gru * gain
        if self.gru_neuromod_gain > 0.0:
            # MCMC entropy: H = -sum(p * log(p)) per timestep
            # mcmc_prior: (B, T, M)
            mcmc_safe = mcmc_prior.clamp(min=1e-8)            # (B, T, M)
            entropy = -(mcmc_safe * mcmc_safe.log()).sum(dim=-1)  # (B, T)
            assert entropy.shape == (B, T), (
                f"entropy shape {tuple(entropy.shape)} != (B={B}, T={T})"
            )
            # Normalize entropy to [0, 1] range (max entropy = log(M))
            max_entropy = math.log(self.mcmc_dim)
            entropy_norm = entropy / max_entropy               # (B, T) in [0, 1]
            # Gain modulation: gain in (0, 2) centered at 1
            gain = torch.sigmoid(
                self._gain_scale * entropy_norm + self._gain_bias
            ) * 2.0                                            # (B, T) in (0, 2)
            gain = gain.unsqueeze(-1)                          # (B, T, 1)
            out_gru = out_gru * gain                           # (B, T, H)
            assert out_gru.shape == (B, T, self.hidden_dim), (
                f"neuromod out_gru shape {tuple(out_gru.shape)} != (B={B}, T={T}, H={self.hidden_dim})"
            )

        # Step 5: MoR Router -- per-step blending weights
        e_flat = e_sensory.reshape(B * T, -1)
        m_flat = mcmc_prior.reshape(B * T, -1)
        gates = self.router(e_flat, m_flat)
        gates = gates.reshape(B, T, 2)

        assert gates.shape == (B, T, 2), (
            f"gates shape {tuple(gates.shape)} != (B={B}, T={T}, 2)"
        )

        g_lif = gates[:, :, 0:1]
        g_gru = gates[:, :, 1:2]

        # Step 5b: In-Silico Lesion Hook
        if override_gates is not None:
            if "g_lif" in override_gates:
                g_lif = torch.full_like(g_lif, override_gates["g_lif"])
            if "g_gru" in override_gates:
                g_gru = torch.full_like(g_gru, override_gates["g_gru"])

            assert g_lif.shape == (B, T, 1), (
                f"g_lif override shape {tuple(g_lif.shape)} != (B={B}, T={T}, 1)"
            )
            assert g_gru.shape == (B, T, 1), (
                f"g_gru override shape {tuple(g_gru.shape)} != (B={B}, T={T}, 1)"
            )

        # Step 6: Integrate pathways
        h_out = g_lif * out_lif + g_gru * out_gru

        assert h_out.shape == (B, T, self.hidden_dim), (
            f"h_out shape {tuple(h_out.shape)} != (B={B}, T={T}, H={self.hidden_dim})"
        )

        # Step 7: Decode to continuous output
        y_pred = self.direction_head(h_out)

        assert y_pred.shape == (B, T), (
            f"y_pred shape {tuple(y_pred.shape)} != (B={B}, T={T})"
        )

        # Step 8: Build output
        effective_gates = torch.cat([g_lif, g_gru], dim=-1)

        internals: Dict[str, torch.Tensor] = {
            "routing_gates": effective_gates,
            "natural_gates": gates,
            "lif_potentials": lif_potentials,
            "lif_spikes": lif_spikes,
            "lif_thresholds": lif_thresholds,
            "gru_hidden": out_gru,
        }

        # Build updated states for autoregressive mode
        if states is not None:
            states_out: Dict[str, torch.Tensor] = {
                "lif_v": lif_v_final.contiguous(),
                "lif_i_syn": lif_i_syn_final.contiguous(),
                "lif_refract": lif_refract_final.contiguous(),
                "lif_w_adapt": lif_w_adapt_final.contiguous(),
                "lif_rel_refract": lif_rel_refract_final.contiguous(),
                "gru_h": out_gru[:, -1:, :].permute(1, 0, 2).contiguous(),
            }
            # Add STP state when enabled
            if self.lif_cell.stp_enabled:
                states_out["lif_x_resource"] = lif_x_resource_final.contiguous()
                states_out["lif_u_facil"] = lif_u_facil_final.contiguous()
            # Add dendritic compartment state when enabled
            if self.lif_cell._dendritic_enabled:
                dend_state = getattr(self.lif_cell, '_dendritic_state', None)
                if dend_state is not None:
                    states_out["lif_dendritic_state"] = dend_state.contiguous()
            # Add lateral inhibition spike history when enabled
            if self.lif_cell.lateral_inhibition > 0.0:
                spike_hist = getattr(self.lif_cell, '_spike_history', None)
                if spike_hist is not None:
                    states_out["lif_spike_history"] = spike_hist.contiguous()
            return y_pred, internals, states_out

        if return_internals:
            return y_pred, internals

        return y_pred

    def freeze_modules(self, module_names: List[str]) -> None:
        """
        Freeze parameters of the specified sub-modules.

        Args:
            module_names: List of sub-module names to freeze.

        Raises:
            ValueError: If a name is not a valid sub-module.
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

    # -- Private pathway runners ------------------------------------

    def _run_lif_path(
        self,
        e_sensory: torch.Tensor,
        lengths: torch.Tensor,
        lif_state0: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """
        Run the LIF cell step-by-step, masking padded positions.

        Args:
            e_sensory: ``(B, T, H)``
            lengths: ``(B,)``
            lif_state0: Optional initial state tuple.

        Returns:
            ``(out_lif, potentials, spikes, thresholds,
            v_final, i_syn_final, refract_final, w_adapt_final,
            rel_refract_final, x_resource_final, u_facil_final)``
            First four are ``(B, T, H)``; last seven are ``(B, H)``.
            x_resource_final and u_facil_final are ones/zeros when STP disabled.

            ``thresholds[:, t, :]`` records the effective threshold used
            for spike detection at step *t*.  This threshold is computed
            from ``rel_refract_counter`` at step *t-1* (the incoming
            counter state).  Execution order within ``LIFCell.forward``:

            1. ``v_thresh_new`` computed from incoming ``rel_refract_counter``
               (carried from step *t-1*).
            2. Spike detection uses ``v_thresh_new``.
            3. ``rel_refract_new`` computed (counter reset to 0 on spike,
               else incremented).
            4. State packed: ``lif_state[3] = v_thresh_new`` (from step 1,
               NOT from the updated counter).

            ``thresholds[t]`` and ``spikes[t]`` describe the same physical
            moment: both use the threshold from the *t-1* counter state.
            The counter update at step *t* (``rel_refract_new``) does NOT
            affect ``thresholds[t]`` — it produces the threshold at step
            *t+1*.
        """
        B, T, H = e_sensory.shape
        device = e_sensory.device

        if lif_state0 is not None:
            lif_state = lif_state0
        else:
            lif_state = self.lif_cell.init_state(B, device)

        out_lif = torch.zeros(B, T, H, device=device)
        potentials = torch.zeros(B, T, H, device=device)
        spikes = torch.zeros(B, T, H, device=device)
        thresh_over_time = torch.zeros(B, T, H, device=device)

        for t in range(T):
            inp_t = e_sensory[:, t, :]
            spike, lif_state = self.lif_cell(inp_t, lif_state)

            mask = (t < lengths).float().unsqueeze(-1)
            out_lif[:, t, :] = spike * mask
            potentials[:, t, :] = lif_state[0] * mask
            spikes[:, t, :] = spike * mask
            # Record threshold used for spike detection at step t.
            # lif_state[3] = v_thresh_new computed from the INCOMING
            # rel_refract_counter (step t-1 state).  The counter update
            # at step t (rel_refract_new) does NOT affect this value —
            # it will produce the threshold at step t+1.
            thresh_over_time[:, t, :] = lif_state[3] * mask  # v_thresh_eff

        # Extract final state components
        # State layout (CF3):
        #   No STP, 6-tuple: (v, i_syn, refract, v_thresh, w_adapt, rel_refract)
        #   STP,    8-tuple: (v, i_syn, refract, v_thresh, w_adapt, rel_refract, x_resource, u_facil)
        v_final = lif_state[0]
        i_syn_final = lif_state[1]
        refract_final = lif_state[2]
        w_adapt_final = lif_state[4]
        rel_refract_final = lif_state[5]

        # STP state (when enabled, state has 8 elements)
        if self.lif_cell.stp_enabled and len(lif_state) == 8:
            x_resource_final = lif_state[6]
            u_facil_final = lif_state[7]
        else:
            # STP disabled: return dummy tensors for consistent return signature
            x_resource_final = torch.ones(B, H, device=device)
            u_facil_final = torch.zeros(B, H, device=device)

        return (out_lif, potentials, spikes, thresh_over_time,
                v_final, i_syn_final, refract_final, w_adapt_final,
                rel_refract_final, x_resource_final, u_facil_final)


# ===============================================================
# 7.  Backward-compatible alias
# ===============================================================

NSMoR = NSMoRCore


# ===============================================================
# 8.  Forward-pass smoke test
# ===============================================================

def _test_forward_pass() -> None:
    """
    Verify that ``NSMoRCore.forward`` produces the expected output shapes.

    Tests backward-compatible default behavior and all biophysical
    features (refractory periods, synaptic delay, spike-frequency
    adaptation, short-term plasticity).

    Run::

        python -m nsmor.model_nsmor_core
    """
    print("=" * 60)
    print("NSMoRCore forward-pass smoke test")
    print("=" * 60)

    B, T, H = 4, 120, 64
    device = torch.device("cpu")

    X_batch = torch.randn(B, T, 8, device=device)
    lengths = torch.tensor([120, 90, 60, 30], dtype=torch.int64, device=device)

    # 1. Default model (backward compatible)
    print("\n  --- Backward-compatible defaults ---")
    model = NSMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=H,
        num_gru_layers=1, dropout=0.1,
        lif_alpha=0.9, lif_threshold=1.0, lif_beta=0.5,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Verify STP is disabled by default
    assert not model.lif_cell.stp_enabled, "STP should be disabled by default"

    # Plain forward
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_batch, lengths)

    assert Y_pred.shape == (B, T)
    print(f"  X_batch shape:  {tuple(X_batch.shape)}  == (B={B}, T={T}, 8)")
    print(f"  Y_pred shape:   {tuple(Y_pred.shape)}  == (B={B}, T={T})")

    # Internals forward
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

    # freeze_modules
    model.freeze_modules(["lif_cell", "router"])
    for p in model.lif_cell.parameters():
        assert not p.requires_grad
    for p in model.router.parameters():
        assert not p.requires_grad
    for p in model.sensory_encoder.parameters():
        assert p.requires_grad
    print("  freeze_modules: lif_cell + router frozen, encoder trainable")

    # Gradient flow (unfrozen)
    model2 = NSMoRCore(hidden_dim=H)
    X2 = torch.randn(2, 40, 8, requires_grad=True)
    len2 = torch.tensor([40, 20], dtype=torch.int64)
    Y2 = model2(X2, len2)
    Y2.sum().backward()
    assert X2.grad is not None
    assert X2.grad.abs().sum() > 0
    print("  Gradient flow: OK")

    # 2. Biophysical model (refractory + synaptic delay)
    print("\n  --- Biophysical features ---")
    model_bio = NSMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=H,
        num_gru_layers=1, dropout=0.1,
        lif_alpha=0.9, lif_threshold=1.0, lif_beta=0.5,
        lif_abs_refract_steps=2,
        lif_rel_refract_steps=5,
        lif_tau_syn=2.0,
        lif_v_rest=0.0,
    ).to(device)

    param_count_bio = sum(p.numel() for p in model_bio.parameters())
    assert param_count_bio == param_count, (
        f"Biophysical model should have same param count: "
        f"{param_count_bio} != {param_count}"
    )
    print(f"  Bio model params: {param_count_bio:,} (same as default)")

    model_bio.eval()
    with torch.no_grad():
        Y_bio, internals_bio = model_bio(
            X_batch, lengths, return_internals=True,
        )

    assert Y_bio.shape == (B, T)
    assert internals_bio["lif_potentials"].shape == (B, T, H)
    assert internals_bio["lif_spikes"].shape == (B, T, H)
    print(f"  Y_bio shape:    {tuple(Y_bio.shape)}")

    # Gradient flow with biophysics
    X_bio = torch.randn(2, 40, 8, requires_grad=True)
    len_bio = torch.tensor([40, 20], dtype=torch.int64)
    Y_bio_g = model_bio(X_bio, len_bio)
    Y_bio_g.sum().backward()
    assert X_bio.grad is not None
    assert X_bio.grad.abs().sum() > 0
    print("  Bio gradient:   OK")

    # 3. Spike-frequency adaptation (AdEx-style)
    print("\n  --- Spike-frequency adaptation ---")
    model_sfa = NSMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=H,
        num_gru_layers=1, dropout=0.1,
        lif_alpha=0.9, lif_threshold=1.0, lif_beta=0.5,
        lif_tau_w=10.0,
        lif_b_adapt=0.05,
    ).to(device)

    param_count_sfa = sum(p.numel() for p in model_sfa.parameters())
    assert param_count_sfa == param_count, (
        f"SFA model should have same param count: "
        f"{param_count_sfa} != {param_count}"
    )
    print(f"  SFA model params: {param_count_sfa:,} (same as default)")

    model_sfa.eval()
    with torch.no_grad():
        Y_sfa, internals_sfa = model_sfa(
            X_batch, lengths, return_internals=True,
        )

    assert Y_sfa.shape == (B, T)
    assert internals_sfa["lif_spikes"].shape == (B, T, H)
    print(f"  Y_sfa shape:    {tuple(Y_sfa.shape)}")

    # Gradient flow with SFA
    X_sfa = torch.randn(2, 40, 8, requires_grad=True)
    len_sfa = torch.tensor([40, 20], dtype=torch.int64)
    Y_sfa_g = model_sfa(X_sfa, len_sfa)
    Y_sfa_g.sum().backward()
    assert X_sfa.grad is not None
    assert X_sfa.grad.abs().sum() > 0
    print("  SFA gradient:   OK")

    # Verify adaptation effect
    with torch.no_grad():
        constant_input = torch.ones(1, 50, 8) * 0.5
        len_const = torch.tensor([50], dtype=torch.int64)
        _, int_const = model_sfa(constant_input, len_const, return_internals=True)
        spikes_t = int_const["lif_spikes"][0].sum(dim=-1)
        early_rate = spikes_t[:10].mean()
        late_rate = spikes_t[-10:].mean()
        print(f"  early spike rate: {early_rate.item():.4f}")
        print(f"  late spike rate:  {late_rate.item():.4f}")
        print(f"  adaptation suppresses: {late_rate <= early_rate}")

    # 4. Short-Term Plasticity (Tsodyks-Markram)
    print("\n  --- Short-Term Plasticity (Tsodyks-Markram) ---")
    model_stp = NSMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=H,
        num_gru_layers=1, dropout=0.1,
        lif_alpha=0.9, lif_threshold=1.0, lif_beta=0.5,
        lif_tau_fac=20.0,
        lif_tau_rec=200.0,
        lif_U_stp_init=0.5,
    ).to(device)

    # STP should be enabled
    assert model_stp.lif_cell.stp_enabled, "STP should be enabled"

    # STP adds 1 learnable parameter (U_stp_raw)
    param_count_stp = sum(p.numel() for p in model_stp.parameters())
    assert param_count_stp == param_count + 1, (
        f"STP model should have 1 extra param: "
        f"{param_count_stp} != {param_count} + 1"
    )
    print(f"  STP model params: {param_count_stp:,} (base + 1 for U_stp_raw)")

    # Verify U_stp is learnable
    assert model_stp.lif_cell.U_stp_raw.requires_grad, "U_stp_raw must be learnable"
    U_val = torch.sigmoid(model_stp.lif_cell.U_stp_raw).item()
    print(f"  U_stp (sigmoid): {U_val:.4f}")

    # Forward with STP
    model_stp.eval()
    with torch.no_grad():
        Y_stp, internals_stp = model_stp(
            X_batch, lengths, return_internals=True,
        )

    assert Y_stp.shape == (B, T)
    assert internals_stp["lif_spikes"].shape == (B, T, H)
    print(f"  Y_stp shape:    {tuple(Y_stp.shape)}")
    print(f"  lif_spikes:     {tuple(internals_stp['lif_spikes'].shape)}")

    # Gradient flow with STP
    X_stp = torch.randn(2, 40, 8, requires_grad=True)
    len_stp = torch.tensor([40, 20], dtype=torch.int64)
    Y_stp_g = model_stp(X_stp, len_stp)
    Y_stp_g.sum().backward()
    assert X_stp.grad is not None
    assert X_stp.grad.abs().sum() > 0
    # U_stp_raw should also receive gradients
    assert model_stp.lif_cell.U_stp_raw.grad is not None, (
        "U_stp_raw must receive gradients"
    )
    print("  STP gradient:   OK (X + U_stp_raw)")

    # Verify STP state is in init_state (CF3: 8-tuple with rel_refract_counter)
    stp_state = model_stp.lif_cell.init_state(2, device)
    assert len(stp_state) == 8, f"STP state should be 8-tuple, got {len(stp_state)}"
    x_init, u_init = stp_state[6], stp_state[7]
    assert x_init.shape == (2, H)
    assert u_init.shape == (2, H)
    # x should be 1.0 (full resources), u should be U (baseline utilization)
    assert torch.allclose(x_init, torch.ones_like(x_init)), "x_resource should init to 1.0"
    expected_u = torch.sigmoid(model_stp.lif_cell.U_stp_raw).item()
    assert torch.allclose(u_init, torch.full_like(u_init, expected_u), atol=1e-6), (
        f"u_facil should init to U={expected_u:.4f}"
    )
    print(f"  STP init: x=1.0, u={expected_u:.4f} (correct)")

    # Autoregressive state with STP
    print("\n  --- Autoregressive state with STP ---")
    model_ar_stp = NSMoRCore(
        hidden_dim=H,
        lif_tau_fac=20.0,
        lif_tau_rec=200.0,
        lif_U_stp_init=0.5,
    )
    model_ar_stp.eval()

    X_step = torch.randn(1, 1, 8)
    len_step = torch.tensor([1], dtype=torch.int64)

    # First step (no states)
    y1, internals1 = model_ar_stp(X_step, len_step, return_internals=True)

    # Build states from internals
    states = {
        "lif_v": internals1["lif_potentials"][:, -1, :].contiguous(),
        "gru_h": internals1["gru_hidden"][:, -1:, :].permute(1, 0, 2).contiguous(),
    }

    # Second step (with states)
    y2, internals2, states_out = model_ar_stp(
        X_step, len_step, return_internals=True, states=states,
    )
    assert "lif_v" in states_out
    assert "lif_i_syn" in states_out
    assert "lif_refract" in states_out
    assert "lif_w_adapt" in states_out
    assert "lif_rel_refract" in states_out  # CF3
    assert "lif_x_resource" in states_out
    assert "lif_u_facil" in states_out
    assert "gru_h" in states_out
    assert states_out["lif_v"].shape == (1, H)
    assert states_out["lif_x_resource"].shape == (1, H)
    assert states_out["lif_u_facil"].shape == (1, H)
    print(f"  states_out keys: {sorted(states_out.keys())}")
    print(f"  lif_x_resource:  {tuple(states_out['lif_x_resource'].shape)}")
    print(f"  lif_u_facil:     {tuple(states_out['lif_u_facil'].shape)}")

    # Third step
    y3, internals3, states_out3 = model_ar_stp(
        X_step, len_step, return_internals=True, states=states_out,
    )
    assert y3.shape == (1, 1)
    print("  Extended STP state loop: OK")

    # 5. Autoregressive state (backward compat, no STP)
    print("\n  --- Autoregressive state (no STP) ---")
    model_ar = NSMoRCore(
        hidden_dim=H,
        lif_abs_refract_steps=2,
        lif_rel_refract_steps=5,
        lif_tau_syn=2.0,
    )
    model_ar.eval()

    y1b, internals1b = model_ar(X_step, len_step, return_internals=True)
    states_b = {
        "lif_v": internals1b["lif_potentials"][:, -1, :].contiguous(),
        "gru_h": internals1b["gru_hidden"][:, -1:, :].permute(1, 0, 2).contiguous(),
    }
    y2b, internals2b, states_out_b = model_ar(
        X_step, len_step, return_internals=True, states=states_b,
    )
    assert "lif_v" in states_out_b
    assert "lif_x_resource" not in states_out_b  # no STP
    assert "lif_u_facil" not in states_out_b      # no STP
    print("  No-STP autoregressive: OK")

    print("=" * 60)
    print("All forward-pass assertions passed.")
    print("=" * 60)


if __name__ == "__main__":
    _test_forward_pass()
