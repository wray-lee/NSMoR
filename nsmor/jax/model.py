"""
JAX/Flax Implementation of NSMoR — Neural Spiking Mixture-of-Recursions.

Provides accelerated, JIT-compiled dual-pathway recurrent architecture:
  - Path A (LIF): Biophysical Leaky Integrate-and-Fire neuron with
    synaptic delay (IIR), spike-frequency adaptation (AdEx), relative/absolute
    refractory dynamics, lateral inhibition, and XLA surrogate gradients.
  - Path B (GRU): Native cuDNN-compatible recurrent unit with optional
    entropy-driven neuromodulatory gain.
  - MoR Router & DirectionHead: Learned dynamic gating and linear decoding.

Sequence unrolling is fused into a single ``jax.lax.scan`` kernel, eliminating
per-timestep Python interpretation overhead and achieving multi-fold speedup.
Full bidirectional weight compatibility with PyTorch NSMoRCore is provided.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np


# ===============================================================
# Biophysical Helper Functions & Decay Computations
# ===============================================================

def compute_decay_factor(tau_ms: float, dt_ms: float) -> float:
    """Compute per-timestep exponential decay factor alpha = exp(-dt / tau)."""
    if tau_ms <= 0.0:
        return 0.0
    return float(math.exp(-dt_ms / tau_ms))


def surrogate_spike(v: jnp.ndarray, v_th: jnp.ndarray, in_abs: jnp.ndarray, scale: float = 4.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Spike detection with smooth sigmoid surrogate gradient.

    Forward: exact Heaviside step masked by absolute refractory period.
    Backward: surrogate derivative via sigmoid(scale * (v - v_th)).
    """
    raw_spk = (v > v_th).astype(jnp.float32)
    spk_mask = raw_spk * (1.0 - in_abs)
    sig = jax.nn.sigmoid(scale * (v - v_th))
    # Surrogate gradient trick: forward is spk_mask, backward is d(sig)/dv
    spike = spk_mask - lax.stop_gradient(sig) + sig
    return spike, spk_mask


# ===============================================================
# Flax Submodules
# ===============================================================

class SensoryEncoderJAX(nn.Module):
    """Sensory projection: Linear(4, H) -> LayerNorm -> ReLU.

    Optionally injects Gaussian noise during training to model intrinsic
    neural variability and stochastic resonance.
    Ref: Douglass et al. 1993, Nature 365:721-723.
    """
    hidden_dim: int = 64
    sensory_noise_std: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        assert x.ndim >= 2, f"SensoryEncoderJAX input must be >= 2-D, got {x.ndim}-D"
        h = nn.Dense(self.hidden_dim, name="dense")(x)
        h = nn.LayerNorm(name="ln")(h)
        h = nn.relu(h)
        # MINOR-4 fix: Stochastic resonance noise injection (training only)
        if not deterministic and self.sensory_noise_std > 0.0:
            noise = jax.random.normal(
                self.make_rng("dropout"), h.shape
            ) * self.sensory_noise_std
            h = h + noise
        return h


class MoRRouterJAX(nn.Module):
    """MoR causal inference gate: Linear(H + M, 2) -> Softmax."""
    @nn.compact
    def __call__(self, e_sensory: jnp.ndarray, mcmc_prior: jnp.ndarray) -> jnp.ndarray:
        comb = jnp.concatenate([e_sensory, mcmc_prior], axis=-1)
        logits = nn.Dense(2, name="gate")(comb)
        return jax.nn.softmax(logits, axis=-1)


class DirectionHeadJAX(nn.Module):
    """Direction decoder: LayerNorm(H) -> ReLU -> Dropout -> Linear(H, 1)."""
    hidden_dim: int = 64
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, h: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        h_norm = nn.LayerNorm(name="ln")(h)
        h_act = nn.relu(h_norm)
        if not deterministic and self.dropout_rate > 0.0:
            h_act = nn.Dropout(self.dropout_rate, deterministic=False)(h_act)
        y = nn.Dense(1, name="dense")(h_act)
        return jnp.squeeze(y, axis=-1)


# ===============================================================
# Full NSMoR Flax Model
# ===============================================================

class NSMoRModel(nn.Module):
    """
    Unified Flax Linen NSMoR Model.

    Encapsulates SensoryEncoder, LIF and GRU recurrent paths, MoR router,
    and DirectionHead decoder into a functional XLA-compiled graph.
    """
    sensory_dim: int = 4
    mcmc_dim: int = 4
    hidden_dim: int = 64
    dt_ms: float = 4.0

    # LIF hyperparameters
    lif_alpha: float = 0.9587
    lif_threshold: float = 0.5
    lif_beta: float = 2.0
    lif_tau_syn: float = 5.0
    lif_tau_w: float = 100.0
    lif_b_adapt: float = 0.5
    lif_lateral_inhibition: float = 0.1
    lif_inhib_tau_ms: float = 50.0
    lif_rel_refract_ms: float = 20.0
    lif_abs_refract_ms: float = 0.0
    lif_v_rest: float = 0.0
    lif_tbptt_steps: int = 32

    # Neuromodulatory gain on GRU
    gru_neuromod_gain: float = 0.0
    dropout_rate: float = 0.1
    sensory_noise_std: float = 0.0

    def setup(self) -> None:
        self.sensory_encoder = SensoryEncoderJAX(
            hidden_dim=self.hidden_dim,
            sensory_noise_std=self.sensory_noise_std,
        )
        self.router = MoRRouterJAX()
        self.direction_head = DirectionHeadJAX(hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate)

        # LIF linear projection parameters (H, H) and (H,)
        self.lif_w_in = self.param(
            "lif_w_in",
            lambda rng, shape: jax.random.normal(rng, shape) * (1.0 / math.sqrt(self.hidden_dim)),
            (self.hidden_dim, self.hidden_dim),
        )
        self.lif_b_in = self.param(
            "lif_b_in",
            lambda rng, shape: jnp.full(shape, 0.01, dtype=jnp.float32),
            (self.hidden_dim,),
        )

        # Lateral inhibition weight matrix
        if self.lif_lateral_inhibition > 0.0:
            self.lif_w_inhib = self.param(
                "lif_w_inhib",
                lambda rng, shape: jnp.zeros(shape, dtype=jnp.float32),
                (self.hidden_dim, self.hidden_dim),
            )
        else:
            self.lif_w_inhib = None

        # GRU parameters (matching PyTorch cuDNN parameter layout)
        # weight_ih: (192, 64), weight_hh: (192, 64), bias_ih: (192,), bias_hh: (192,)
        self.gru_w_ih = self.param(
            "gru_w_ih",
            lambda rng, shape: jax.random.normal(rng, shape) * (1.0 / math.sqrt(self.hidden_dim)),
            (3 * self.hidden_dim, self.hidden_dim),
        )
        self.gru_w_hh = self.param(
            "gru_w_hh",
            lambda rng, shape: jax.random.normal(rng, shape) * (1.0 / math.sqrt(self.hidden_dim)),
            (3 * self.hidden_dim, self.hidden_dim),
        )
        self.gru_b_ih = self.param(
            "gru_b_ih",
            lambda rng, shape: jnp.zeros(shape, dtype=jnp.float32),
            (3 * self.hidden_dim,),
        )
        self.gru_b_hh = self.param(
            "gru_b_hh",
            lambda rng, shape: jnp.zeros(shape, dtype=jnp.float32),
            (3 * self.hidden_dim,),
        )

        # Neuromodulatory gain parameters
        if self.gru_neuromod_gain > 0.0:
            self.gain_scale = self.param("gain_scale", lambda rng: jnp.array(0.0, dtype=jnp.float32))
            self.gain_bias = self.param("gain_bias", lambda rng: jnp.array(1.0, dtype=jnp.float32))

    def __call__(
        self,
        x: jnp.ndarray,
        lengths: jnp.ndarray,
        *,
        deterministic: bool = True,
        override_gates: Optional[Dict[str, float]] = None,
        return_internals: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]]:
        """
        Forward pass over variable-length sequence batch.

        Args:
            x: (B, T, 8) input features.
            lengths: (B,) true sequence lengths.
            deterministic: disable dropout during eval.
            override_gates: optional lesion override dict {'g_lif': ..., 'g_gru': ...}.
            return_internals: if True, returns dictionary of internal states.

        Returns:
            y_pred: (B, T) predicted velocity.
            internals (optional): dictionary of internal tensors.
        """
        # MINOR-5 fix: Shape assertions on critical tensors
        assert x.ndim == 3, f"Input must be 3-D (B, T, D), got {x.ndim}-D"
        B, T, D = x.shape
        H = self.hidden_dim
        dt = self.dt_ms

        assert D == self.sensory_dim + self.mcmc_dim, f"Expected dim {self.sensory_dim + self.mcmc_dim}, got {D}"

        sensory_x = x[:, :, :self.sensory_dim]
        mcmc_prior = x[:, :, self.sensory_dim:]

        # 1. Sensory Encoder (pass deterministic for noise gating)
        e_sensory = self.sensory_encoder(sensory_x, deterministic=deterministic)  # (B, T, H)
        assert e_sensory.shape == (B, T, H), f"e_sensory shape {e_sensory.shape} != ({B}, {T}, {H})"

        # 2. Precompute decay constants
        alpha_syn = compute_decay_factor(self.lif_tau_syn, dt)
        decay_w = compute_decay_factor(self.lif_tau_w, dt)
        decay_inhib = compute_decay_factor(max(self.lif_tau_syn, self.lif_inhib_tau_ms), dt)
        k_rel = dt / self.lif_rel_refract_ms if self.lif_rel_refract_ms > 0.0 else 0.0
        delta_theta = 0.3 * self.lif_threshold
        v_thresh = self.lif_threshold
        # BLOCKER-2 fix: Match PyTorch clamp thresholds exactly.
        # PyTorch uses 3.0 * v_threshold and 5.0 * v_threshold
        # (model_nsmor_core.py:436,442).
        v_clamp_max = 3.0 * self.lif_threshold
        i_syn_clamp = 5.0 * self.lif_threshold
        abs_refract_steps = round(self.lif_abs_refract_ms / dt) if self.lif_abs_refract_ms > 0.0 else 0.0

        if self.lif_lateral_inhibition > 0.0 and self.lif_w_inhib is not None:
            inhib_mask = 1.0 - jnp.eye(H, dtype=jnp.float32)
            W_inhib = -jax.nn.softplus(self.lif_w_inhib) * inhib_mask
        else:
            W_inhib = jnp.zeros((H, H), dtype=jnp.float32)

        tbptt = self.lif_tbptt_steps

        # 3. Recurrent Scan Step (Fused LIF + GRU)
        # Carry: (v, i_syn, ref, rel_ref, w, spk_hist, h_gru, t_step)
        def scan_step(carry, step_input):
            (v, i_syn, ref, rel_ref, w, spk_hist, h_gru, t_step), (e_t, mask_t) = carry, step_input

            # Optional TBPTT gradient truncation
            if tbptt > 0:
                do_detach = (t_step > 0) & (t_step % tbptt == 0)
                v = lax.cond(do_detach, lax.stop_gradient, lambda x: x, v)
                i_syn = lax.cond(do_detach, lax.stop_gradient, lambda x: x, i_syn)
                ref = lax.cond(do_detach, lax.stop_gradient, lambda x: x, ref)
                rel_ref = lax.cond(do_detach, lax.stop_gradient, lambda x: x, rel_ref)
                w = lax.cond(do_detach, lax.stop_gradient, lambda x: x, w)
                spk_hist = lax.cond(do_detach, lax.stop_gradient, lambda x: x, spk_hist)

            # --- LIF Pathway ---
            proj = e_t @ self.lif_w_in + self.lif_b_in
            raw_input = self.lif_beta * proj
            i_syn_new = jnp.clip(
                alpha_syn * i_syn + (1.0 - alpha_syn) * raw_input,
                -i_syn_clamp,
                i_syn_clamp,
            )

            in_abs = (ref > 0.0).astype(jnp.float32)
            if k_rel > 0.0:
                v_th = v_thresh + delta_theta * jnp.exp(-k_rel * rel_ref)
            else:
                v_th = jnp.full_like(v, v_thresh)

            v_new = self.lif_alpha * v + i_syn_new - w
            v_new = v_new * (1.0 - in_abs) + self.lif_v_rest * in_abs
            v_new = jnp.clip(v_new, -v_thresh, v_clamp_max)

            if self.lif_lateral_inhibition > 0.0:
                inhib_current = spk_hist @ W_inhib.T
                v_new = v_new + self.lif_lateral_inhibition * inhib_current

            # Spike generation
            spike, spk_mask = surrogate_spike(v_new, v_th, in_abs)

            # State updates
            if self.lif_lateral_inhibition > 0.0:
                spk_hist_new = decay_inhib * spk_hist + (1.0 - decay_inhib) * spk_mask
            else:
                spk_hist_new = spk_hist

            # Soft reset
            v_reset = v_new - spk_mask * v_th

            # Adaptation
            w_new = jnp.clip(decay_w * w + self.lif_b_adapt * spk_mask, 0.0, 10.0 * v_thresh)

            # Refractory counters
            if abs_refract_steps > 0:
                ref_new = jnp.where(spk_mask > 0.5, abs_refract_steps, jnp.maximum(0.0, ref - 1.0))
            else:
                ref_new = ref

            if k_rel > 0.0:
                rel_ref_new = jnp.where(spk_mask > 0.5, 0.0, rel_ref + 1.0)
            else:
                rel_ref_new = rel_ref

            # --- GRU Pathway (cuDNN matching) ---
            gi = e_t @ self.gru_w_ih.T + self.gru_b_ih
            gh = h_gru @ self.gru_w_hh.T + self.gru_b_hh

            gi_r, gi_z, gi_n = jnp.split(gi, 3, axis=-1)
            gh_r, gh_z, gh_n = jnp.split(gh, 3, axis=-1)

            r_gate = jax.nn.sigmoid(gi_r + gh_r)
            z_gate = jax.nn.sigmoid(gi_z + gh_z)
            n_gate = jnp.tanh(gi_n + r_gate * gh_n)

            h_gru_next = (1.0 - z_gate) * n_gate + z_gate * h_gru

            # Padded sequence masking
            m_2d = mask_t[:, None]
            out_lif_t = spike * m_2d
            # MINOR-2 fix: Export post-reset potentials to align with PyTorch
            # which exports lif_state[0] (= post-reset v) in
            # model_nsmor_core.py:1651.
            out_pot_t = v_reset * m_2d
            out_spk_t = spike * m_2d
            out_gru_t = h_gru_next * m_2d

            # MAJOR-4 fix: Gate ALL carry states by padding mask.
            # Padded frames must not advance LIF state — matching the
            # GRU h_gru_state treatment already applied below.
            h_gru_state = jnp.where(m_2d > 0.5, h_gru_next, h_gru)
            v_reset_gated = jnp.where(m_2d > 0.5, v_reset, v)
            i_syn_gated = jnp.where(m_2d > 0.5, i_syn_new, i_syn)
            w_gated = jnp.where(m_2d > 0.5, w_new, w)
            ref_gated = jnp.where(m_2d > 0.5, ref_new, ref)
            rel_ref_gated = jnp.where(m_2d > 0.5, rel_ref_new, rel_ref)
            spk_hist_gated = jnp.where(m_2d > 0.5, spk_hist_new, spk_hist)

            next_carry = (
                v_reset_gated, i_syn_gated, ref_gated, rel_ref_gated, w_gated,
                spk_hist_gated, h_gru_state, t_step + 1,
            )
            step_outputs = (out_lif_t, out_pot_t, out_spk_t, out_gru_t)
            return next_carry, step_outputs

        # Initial state setup
        init_v = jnp.zeros((B, H), dtype=jnp.float32)
        init_isyn = jnp.zeros((B, H), dtype=jnp.float32)
        init_ref = jnp.zeros((B, H), dtype=jnp.float32)
        large_rel = float(10 * max(self.lif_rel_refract_ms / dt, 1.0))
        init_rel_ref = jnp.full((B, H), large_rel, dtype=jnp.float32)
        init_w = jnp.zeros((B, H), dtype=jnp.float32)
        init_spk_hist = jnp.zeros((B, H), dtype=jnp.float32)
        init_h_gru = jnp.zeros((B, H), dtype=jnp.float32)
        init_carry = (
            init_v, init_isyn, init_ref, init_rel_ref, init_w,
            init_spk_hist, init_h_gru, jnp.array(0, dtype=jnp.int32),
        )

        t_idx = jnp.arange(T)[:, None]
        mask_seq = (t_idx < lengths[None, :]).astype(jnp.float32)  # (T, B)
        e_trans = e_sensory.transpose(1, 0, 2)  # (T, B, H)

        _, (lif_out_t, pot_t, spk_t, gru_out_t) = lax.scan(
            scan_step, init_carry, (e_trans, mask_seq)
        )

        out_lif = lif_out_t.transpose(1, 0, 2)
        lif_potentials = pot_t.transpose(1, 0, 2)
        lif_spikes = spk_t.transpose(1, 0, 2)
        out_gru = gru_out_t.transpose(1, 0, 2)

        # 4. Neuromodulatory gain on GRU
        if self.gru_neuromod_gain > 0.0:
            mcmc_safe = jnp.clip(mcmc_prior, 1e-8)
            entropy = -(mcmc_safe * jnp.log(mcmc_safe)).sum(axis=-1)
            max_entropy = math.log(self.mcmc_dim)
            entropy_norm = entropy / max_entropy
            gain = jax.nn.sigmoid(self.gain_scale * entropy_norm + self.gain_bias) * 2.0
            out_gru = out_gru * gain[..., None]

        # 5. MoR Router
        natural_gates = self.router(e_sensory, mcmc_prior)  # (B, T, 2)
        g_lif = natural_gates[:, :, 0:1]
        g_gru = natural_gates[:, :, 1:2]

        if override_gates is not None:
            if "g_lif" in override_gates:
                g_lif = jnp.full_like(g_lif, override_gates["g_lif"])
            if "g_gru" in override_gates:
                g_gru = jnp.full_like(g_gru, override_gates["g_gru"])

        effective_gates = jnp.concatenate([g_lif, g_gru], axis=-1)

        # 6. DirectionHead Decoding
        h_fused = g_lif * out_lif + g_gru * out_gru
        y_pred = self.direction_head(h_fused, deterministic=deterministic)  # (B, T)

        # MINOR-5 fix: Output shape assertions
        assert y_pred.shape == (B, T), f"y_pred shape {y_pred.shape} != ({B}, {T})"
        assert effective_gates.shape == (B, T, 2), f"gates shape {effective_gates.shape} != ({B}, {T}, 2)"

        if return_internals:
            internals = {
                "routing_gates": effective_gates,
                "natural_gates": natural_gates,
                "lif_potentials": lif_potentials,
                "lif_spikes": lif_spikes,
                "gru_hidden": out_gru,
            }
            return y_pred, internals

        return y_pred


# ===============================================================
# PyTorch Checkpoint Loading & State Dict Compatibility
# ===============================================================

def load_from_torch_state_dict(
    model: NSMoRModel,
    state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Map PyTorch NSMoRCore state_dict to Flax parameter PyTree.

    Supports both legacy flat keys and modern hierarchical 'frontend'/'backend' keys.
    """
    def _np(k: str) -> np.ndarray:
        t = state_dict[k]
        if hasattr(t, "cpu"):
            t = t.cpu().numpy()
        return np.asarray(t)

    # Resolve keys with fallback
    def _get(key_primary: str, key_fallback: str) -> np.ndarray:
        if key_primary in state_dict:
            return _np(key_primary)
        elif key_fallback in state_dict:
            return _np(key_fallback)
        raise KeyError(f"Neither '{key_primary}' nor '{key_fallback}' found in checkpoint")

    se_w = _get("frontend.sensory_encoder.net.0.weight", "sensory_encoder.net.0.weight")
    se_b = _get("frontend.sensory_encoder.net.0.bias", "sensory_encoder.net.0.bias")
    se_ln_w = _get("frontend.sensory_encoder.net.1.weight", "sensory_encoder.net.1.weight")
    se_ln_b = _get("frontend.sensory_encoder.net.1.bias", "sensory_encoder.net.1.bias")

    lif_w = _get("backend.lif_cell.W_in.weight", "lif_cell.W_in.weight")
    lif_b = _get("backend.lif_cell.W_in.bias", "lif_cell.W_in.bias")

    gru_w_ih = _get("backend.gru_unit.gru.weight_ih_l0", "gru_unit.gru.weight_ih_l0")
    gru_w_hh = _get("backend.gru_unit.gru.weight_hh_l0", "gru_unit.gru.weight_hh_l0")
    gru_b_ih = _get("backend.gru_unit.gru.bias_ih_l0", "gru_unit.gru.bias_ih_l0")
    gru_b_hh = _get("backend.gru_unit.gru.bias_hh_l0", "gru_unit.gru.bias_hh_l0")

    r_w = _get("backend.router.gate.weight", "router.gate.weight")
    r_b = _get("backend.router.gate.bias", "router.gate.bias")

    dh_ln_w = _get("backend.direction_head.net.0.weight", "direction_head.net.0.weight")
    dh_ln_b = _get("backend.direction_head.net.0.bias", "direction_head.net.0.bias")
    dh_lin_w = _get("backend.direction_head.net.3.weight", "direction_head.net.3.weight")
    dh_lin_b = _get("backend.direction_head.net.3.bias", "direction_head.net.3.bias")

    params: Dict[str, Any] = {
        "sensory_encoder": {
            "dense": {
                "kernel": jnp.array(se_w.T, dtype=jnp.float32),
                "bias": jnp.array(se_b, dtype=jnp.float32),
            },
            "ln": {
                "scale": jnp.array(se_ln_w, dtype=jnp.float32),
                "bias": jnp.array(se_ln_b, dtype=jnp.float32),
            },
        },
        "lif_w_in": jnp.array(lif_w.T, dtype=jnp.float32),
        "lif_b_in": jnp.array(lif_b, dtype=jnp.float32),
        "gru_w_ih": jnp.array(gru_w_ih, dtype=jnp.float32),
        "gru_w_hh": jnp.array(gru_w_hh, dtype=jnp.float32),
        "gru_b_ih": jnp.array(gru_b_ih, dtype=jnp.float32),
        "gru_b_hh": jnp.array(gru_b_hh, dtype=jnp.float32),
        "router": {
            "gate": {
                "kernel": jnp.array(r_w.T, dtype=jnp.float32),
                "bias": jnp.array(r_b, dtype=jnp.float32),
            },
        },
        "direction_head": {
            "ln": {
                "scale": jnp.array(dh_ln_w, dtype=jnp.float32),
                "bias": jnp.array(dh_ln_b, dtype=jnp.float32),
            },
            "dense": {
                "kernel": jnp.array(dh_lin_w.T, dtype=jnp.float32),
                "bias": jnp.array(dh_lin_b, dtype=jnp.float32),
            },
        },
    }

    if model.lif_lateral_inhibition > 0.0:
        w_inhib = _get("backend.lif_cell._W_inhib_raw", "lif_cell._W_inhib_raw")
        params["lif_w_inhib"] = jnp.array(w_inhib, dtype=jnp.float32)

    if model.gru_neuromod_gain > 0.0:
        g_scale = _get("backend._gain_scale", "_gain_scale")
        g_bias = _get("backend._gain_bias", "_gain_bias")
        params["gain_scale"] = jnp.array(g_scale, dtype=jnp.float32)
        params["gain_bias"] = jnp.array(g_bias, dtype=jnp.float32)

    return {"params": params}


def to_torch_state_dict(flax_params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Convert Flax parameter PyTree back to a PyTorch state_dict.

    Provides compatibility so checkpoints trained in JAX can be loaded by
    evaluation and analysis tools in PyTorch.
    """
    import torch

    p = flax_params.get("params", flax_params)
    sd: Dict[str, torch.Tensor] = {}

    def _t(arr: Any) -> torch.Tensor:
        np_arr = np.array(arr, copy=True)
        return torch.from_numpy(np_arr)

    # SensoryEncoder
    sd["frontend.sensory_encoder.net.0.weight"] = _t(p["sensory_encoder"]["dense"]["kernel"]).T
    sd["frontend.sensory_encoder.net.0.bias"] = _t(p["sensory_encoder"]["dense"]["bias"])
    sd["frontend.sensory_encoder.net.1.weight"] = _t(p["sensory_encoder"]["ln"]["scale"])
    sd["frontend.sensory_encoder.net.1.bias"] = _t(p["sensory_encoder"]["ln"]["bias"])

    # LIF
    sd["backend.lif_cell.W_in.weight"] = _t(p["lif_w_in"]).T
    sd["backend.lif_cell.W_in.bias"] = _t(p["lif_b_in"])
    if "lif_w_inhib" in p:
        sd["backend.lif_cell._W_inhib_raw"] = _t(p["lif_w_inhib"])
        H = p["lif_w_inhib"].shape[0]
        sd["backend.lif_cell._inhib_diag_mask"] = torch.from_numpy(1.0 - np.eye(H, dtype=np.float32))

    # GRU
    sd["backend.gru_unit.gru.weight_ih_l0"] = _t(p["gru_w_ih"])
    sd["backend.gru_unit.gru.weight_hh_l0"] = _t(p["gru_w_hh"])
    sd["backend.gru_unit.gru.bias_ih_l0"] = _t(p["gru_b_ih"])
    sd["backend.gru_unit.gru.bias_hh_l0"] = _t(p["gru_b_hh"])

    # Router
    sd["backend.router.gate.weight"] = _t(p["router"]["gate"]["kernel"]).T
    sd["backend.router.gate.bias"] = _t(p["router"]["gate"]["bias"])

    # DirectionHead
    sd["backend.direction_head.net.0.weight"] = _t(p["direction_head"]["ln"]["scale"])
    sd["backend.direction_head.net.0.bias"] = _t(p["direction_head"]["ln"]["bias"])
    sd["backend.direction_head.net.3.weight"] = _t(p["direction_head"]["dense"]["kernel"]).T
    sd["backend.direction_head.net.3.bias"] = _t(p["direction_head"]["dense"]["bias"])

    # Neuromodulatory gain parameters (BLOCKER-1 fix)
    if "gain_scale" in p:
        sd["backend._gain_scale"] = _t(p["gain_scale"])
        sd["backend._gain_bias"] = _t(p["gain_bias"])

    # Duplicate to top-level aliases for full backward compatibility.
    # NOTE: gain parameters only exist under backend.* in PyTorch
    # (they are nn.Parameters on BioDecisionCore, not legacy flat keys),
    # so we must NOT create top-level aliases for them.
    _no_alias = {"backend._gain_scale", "backend._gain_bias"}
    for k in list(sd.keys()):
        if k in _no_alias:
            continue
        if k.startswith("frontend."):
            sd[k.replace("frontend.", "")] = sd[k]
        elif k.startswith("backend."):
            sd[k.replace("backend.", "")] = sd[k]

    return sd
