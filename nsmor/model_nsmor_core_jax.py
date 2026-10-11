"""
NSMoR Core — JAX-Optimized Mixture-of-Recursions Network.

Provides high-performance JAX accelerated implementations of the
dual-pathway recurrent architecture:
  - Path A (LIF): Fused step dynamics with absolute/relative refractory,
    synaptic delay, spike-frequency adaptation, STP, and lateral inhibition.
  - Path B (GRU): Vectorized recurrent cell matching PyTorch cuDNN/native GRU.
  - MoR Router & DirectionHead: JIT-compiled gate blending and decoding.

The entire sequence loop is fused via ``jax.lax.scan`` and JIT-compiled
into a single XLA kernel, eliminating Python per-step loop overhead and
yielding 5-20x speedup for long sequences.

Transparent fallback to PyTorch NSMoRCore is provided if JAX is unavailable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    import jax
    import jax.numpy as jnp
    import jax.lax as lax
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    lax = None
    JAX_AVAILABLE = False

from nsmor.model_nsmor_core import (
    NSMoRCore, certify_executed_topology, _validate_np_leaf,
    _require_real_floating_observation, assert_mechanism_tree_consistent,
    refinement_module_present, time_consuming_module_present,
)


def _require_real_floating_obs(array: Any, *, name: str, context: str) -> None:
    """Reject a non-real-floating observation on its ORIGINAL dtype (root G3).

    Shared by the raw-JAX runner (real and fallback paths): a complex
    measurement (whose imaginary part a later ``astype(float32)`` would silently
    discard) or an integer/boolean observation is a scientific/data error, not a
    value to coerce.  Handles torch tensors via the shared Torch guard and
    numpy/JAX arrays via their numpy dtype.
    """
    if isinstance(array, torch.Tensor):
        _require_real_floating_observation(array, name=name, context=context)
        return
    arr = np.asarray(array)
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(
            f"{context}: {name} must have a real floating dtype, got "
            f"{arr.dtype}. Integer/boolean/complex observations are refused "
            "before conversion."
        )


def _to_numpy(tensor: Any) -> np.ndarray:
    """Convert a PyTorch tensor, JAX array, or sequence to a NumPy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    if JAX_AVAILABLE and isinstance(tensor, jnp.ndarray):
        return np.asarray(tensor)
    return np.asarray(tensor)


def _leaf(
    tensor: Any, key: str, expected_shape: Optional[Tuple[int, ...]] = None,
) -> jnp.ndarray:
    """Validate an ORIGINAL parameter leaf, then narrow to a float32 JAX array.

    Shared helper (root 2): rejects an original NaN/Inf or a finite float64
    value (e.g. 1e300) that overflows the destination float32 representation,
    BEFORE it becomes a poisoned JAX leaf.  H6 / findings A8, B3: when
    ``expected_shape`` is given the EXACT canonical shape is required, so a
    malformed leaf (e.g. a LayerNorm weight of shape ``(1,)`` that would
    silently broadcast) is rejected before mapping rather than certified as a
    runnable model.  Applied to EVERY required leaf (encoder, decoder, GRU,
    router, LIF, gated projections, buffers).
    """
    arr = _validate_np_leaf(
        _to_numpy(tensor), key=key, expected_shape=expected_shape,
        dtype=np.float32,
    )
    return jnp.array(arr, dtype=jnp.float32)


def _scalar_leaf(raw: Any, key: str) -> float:
    """Validate an ORIGINAL learned scalar leaf, then return a python float.

    Root G5 / finding B3: a learned scalar (STP utilization ``U_stp_raw``, the
    neuromodulatory ``_gain_scale`` / ``_gain_bias``) is a required learned leaf,
    not disabled metadata.  It must pass the SAME original real-floating /
    exact-scalar-shape / finiteness / float32-representability checks as the
    matrix leaves before it reaches the compiled kernel — otherwise an enabled
    ``U_stp_raw=NaN`` (or an overflowing finite scalar) is accepted and the
    runner exports nonfinite predictions.
    """
    arr = _validate_np_leaf(
        _to_numpy(raw), key=key, expected_shape=(), dtype=np.float32,
    )
    return float(arr)


def _to_torch(array: Any, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert a NumPy or JAX array to a PyTorch tensor."""
    if isinstance(array, torch.Tensor):
        res = array
    else:
        np_arr = np.array(array, copy=True)
        res = torch.from_numpy(np_arr)
    if device is not None:
        res = res.to(device)
    if dtype is not None and res.dtype != dtype:
        res = res.to(dtype)
    return res


# ===============================================================
# Pure Functional JAX Simulation Kernels
# ===============================================================

if JAX_AVAILABLE:

    def _dendritic_filter_step(alpha_dend: float, s: jnp.ndarray, v_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        s_new = alpha_dend * s + (1.0 - alpha_dend) * v_t
        return s_new, s_new

    def _apply_layernorm(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + eps) * weight + bias


    class _JAXCoreParams:
        """Container for flattened JAX parameters of NSMoRCore."""

        def __init__(self, model: NSMoRCore) -> None:
            # r7 R1: certify the ACTUAL executed encoder/readout topology BEFORE
            # copying any parameter.  The fused kernel hard-codes
            # ``Linear->LayerNorm(eps=1e-5)->ReLU`` (or the gated swiglu branch)
            # and a fixed head; a live source with a replaced activation slot,
            # an appended/missing operator, a noncanonical/nonfinite epsilon or a
            # divergent live dropout rate would be silently mapped onto a
            # different computation.  Share the exact certificate with the Flax
            # converter rather than maintaining a separate allowed-type list.
            certify_executed_topology(
                model, context="NSMoRCoreJAX.from_torch",
            )
            # H4 / findings A5, B2: the shared mechanism-tree preflight runs
            # BEFORE any leaf is dereferenced, so an enabled mechanism whose
            # parameters were never created (an inconsistent live flag) fails
            # closed with a descriptive ValueError instead of an AttributeError
            # mid-copy.
            assert_mechanism_tree_consistent(
                model, context="NSMoRCoreJAX.from_torch",
            )

            # 1. Config attributes
            self.sensory_dim = model.sensory_dim
            self.mcmc_dim = model.mcmc_dim
            self.hidden_dim = model.hidden_dim
            self.dt_ms = model.dt_ms
            self.persistence_skip = float(getattr(model, "persistence_skip", 0.0))
            # Exact canonical dimensions (H6): every required leaf is validated
            # against these before narrowing.
            H = int(self.hidden_dim)
            D = int(self.sensory_dim)
            M = int(self.mcmc_dim)

            # 2. Frontend parameters
            fe = model.frontend
            self.dendritic_enabled = fe._dendritic_enabled
            self.alpha_dend = fe._alpha_dend

            se = fe.sensory_encoder
            self.se_activation = getattr(se, "activation", "relu")
            self.se_w = _leaf(se.net[0].weight, "sensory_encoder.net.0.weight", (H, D))
            self.se_b = _leaf(se.net[0].bias, "sensory_encoder.net.0.bias", (H,))
            self.se_ln_w = _leaf(se.net[1].weight, "sensory_encoder.net.1.weight", (H,))
            self.se_ln_b = _leaf(se.net[1].bias, "sensory_encoder.net.1.bias", (H,))
            if self.se_activation == "swiglu":
                self.se_gate_w = _leaf(se.gate_proj.weight, "sensory_encoder.gate_proj.weight", (H, H))
                self.se_gate_b = _leaf(se.gate_proj.bias, "sensory_encoder.gate_proj.bias", (H,))

            # 3. LIF cell parameters
            lif = model.backend.lif_cell
            self.lif_alpha = lif.alpha
            self.lif_beta = lif.beta
            self.lif_v_threshold = lif.v_threshold
            self.lif_v_rest = lif.v_rest
            self.lif_v_reset = lif.v_reset
            self.lif_hard_reset = lif._hard_reset
            self.lif_delta_theta = lif._delta_theta
            self.lif_k_rel = lif._k_rel
            self.lif_alpha_syn = float(lif._alpha_syn)
            self.lif_decay_w = float(lif._decay_w)
            self.lif_b_adapt = lif.b_adapt
            self.lif_abs_refract_steps = float(lif.abs_refract_steps)
            self.lif_rel_refract_steps = float(lif.rel_refract_steps)
            self.lif_v_clamp_max = lif._v_clamp_max
            self.lif_i_syn_clamp = lif._i_syn_clamp

            self.lif_w_in = _leaf(lif.W_in.weight, "lif_cell.W_in.weight", (H, H))
            self.lif_b_in = _leaf(lif.W_in.bias, "lif_cell.W_in.bias", (H,))

            # STP
            self.stp_enabled = lif.stp_enabled
            if self.stp_enabled:
                # root G5: validate the learned utilization leaf (original
                # dtype/shape/finiteness/float32-representability) before use.
                self.U_stp_raw = _scalar_leaf(
                    lif.U_stp_raw, "lif_cell.U_stp_raw",
                )
                self.decay_fac = lif._decay_fac
                self.decay_rec = lif._decay_rec
            else:
                self.U_stp_raw = 0.0
                self.decay_fac = 0.0
                self.decay_rec = 0.0

            # Lateral inhibition
            self.lateral_inhibition = lif.lateral_inhibition
            if self.lateral_inhibition > 0.0:
                self.W_inhib_raw = _leaf(lif._W_inhib_raw, "lif_cell._W_inhib_raw", (H, H))
                self.decay_inhib = float(lif._decay_inhib)
                self.inhib_diag_mask = _leaf(lif._inhib_diag_mask, "lif_cell._inhib_diag_mask", (H, H))
            else:
                self.W_inhib_raw = jnp.zeros((self.hidden_dim, self.hidden_dim))
                self.decay_inhib = 0.0
                self.inhib_diag_mask = 1.0 - jnp.eye(self.hidden_dim)

            # 4. GRU parameters
            gru = model.backend.gru_unit.gru
            self.gru_w_ih = _leaf(gru.weight_ih_l0, "gru.weight_ih_l0", (3 * H, H))
            self.gru_w_hh = _leaf(gru.weight_hh_l0, "gru.weight_hh_l0", (3 * H, H))
            self.gru_b_ih = _leaf(gru.bias_ih_l0, "gru.bias_ih_l0", (3 * H,))
            self.gru_b_hh = _leaf(gru.bias_hh_l0, "gru.bias_hh_l0", (3 * H,))

            # Neuromodulatory gain
            self.gru_neuromod_gain = model.backend.gru_neuromod_gain
            if self.gru_neuromod_gain > 0.0:
                # root G5: both gain parameters are required learned scalar
                # leaves; validate each before conversion (an enabled gain with
                # a NaN/overflowing scalar otherwise exports NaN predictions).
                self.gain_scale = _scalar_leaf(
                    model.backend._gain_scale, "backend._gain_scale",
                )
                self.gain_bias = _scalar_leaf(
                    model.backend._gain_bias, "backend._gain_bias",
                )
            else:
                self.gain_scale = 0.0
                self.gain_bias = 0.0

            # 5. Router parameters
            router = model.backend.router
            self.router_w = _leaf(router.gate.weight, "router.gate.weight", (2, H + M))
            self.router_b = _leaf(router.gate.bias, "router.gate.bias", (2,))

            # 6. DirectionHead parameters
            dh = model.backend.direction_head
            self.dh_activation = getattr(dh, "activation", "relu")
            self.dh_ln_w = _leaf(dh.net[0].weight, "direction_head.net.0.weight", (H,))
            self.dh_ln_b = _leaf(dh.net[0].bias, "direction_head.net.0.bias", (H,))
            if self.dh_activation == "swiglu":
                self.dh_gate_w = _leaf(dh.gate_proj.weight, "direction_head.gate_proj.weight", (H, H))
                self.dh_gate_b = _leaf(dh.gate_proj.bias, "direction_head.gate_proj.bias", (H,))
                self.dh_value_w = _leaf(dh.value_proj.weight, "direction_head.value_proj.weight", (H, H))
                self.dh_value_b = _leaf(dh.value_proj.bias, "direction_head.value_proj.bias", (H,))
                self.dh_out_w = _leaf(dh.out_proj.weight, "direction_head.out_proj.weight", (1, H))
                self.dh_out_b = _leaf(dh.out_proj.bias, "direction_head.out_proj.bias", (1,))
            else:
                self.dh_lin_w = _leaf(dh.net[3].weight, "direction_head.net.3.weight", (1, H))
                self.dh_lin_b = _leaf(dh.net[3].bias, "direction_head.net.3.bias", (1,))

            # r6 R6: every copied effective runtime coefficient must be finite
            # BEFORE it reaches the compiled kernel.  A poisoned (NaN/Inf)
            # source coefficient would otherwise silently produce nonfinite
            # output instead of failing closed.
            _coeffs = {
                "lif_alpha": self.lif_alpha,
                "lif_beta": self.lif_beta,
                "lif_v_threshold": self.lif_v_threshold,
                "lif_delta_theta": self.lif_delta_theta,
                "lif_k_rel": self.lif_k_rel,
                "lif_alpha_syn": self.lif_alpha_syn,
                "lif_decay_w": self.lif_decay_w,
                "lif_b_adapt": self.lif_b_adapt,
                "lif_v_clamp_max": self.lif_v_clamp_max,
                "lif_i_syn_clamp": self.lif_i_syn_clamp,
                "lif_abs_refract_steps": self.lif_abs_refract_steps,
                "lif_rel_refract_steps": self.lif_rel_refract_steps,
                "decay_fac": self.decay_fac,
                "decay_rec": self.decay_rec,
                "decay_inhib": self.decay_inhib,
            }
            _nonfinite = [
                k for k, v in _coeffs.items() if not math.isfinite(float(v))
            ]
            if _nonfinite:
                raise ValueError(
                    "NSMoRCoreJAX: live source has nonfinite effective runtime "
                    f"coefficients ({', '.join(_nonfinite)}); refusing to copy "
                    "a poisoned runtime source."
                )


    def _build_jax_forward_fn(params: _JAXCoreParams):
        """Construct a compiled JAX forward function with fused lax.scan."""

        H = params.hidden_dim
        v_thresh = params.lif_v_threshold
        delta_theta = params.lif_delta_theta
        k_rel = params.lif_k_rel
        alpha_syn = params.lif_alpha_syn
        decay_w = params.lif_decay_w
        b_adapt = params.lif_b_adapt
        v_rest = params.lif_v_rest
        v_reset = params.lif_v_reset
        hard_reset = params.lif_hard_reset
        abs_refract = params.lif_abs_refract_steps
        v_clamp_max = params.lif_v_clamp_max
        i_syn_clamp = params.lif_i_syn_clamp
        lif_alpha = params.lif_alpha
        lif_beta = params.lif_beta

        stp_enabled = params.stp_enabled
        decay_fac = params.decay_fac
        decay_rec = params.decay_rec
        U_scalar = float(jax.nn.sigmoid(params.U_stp_raw)) if stp_enabled else 0.5

        lateral_inhibition = params.lateral_inhibition
        decay_inhib = params.decay_inhib
        if lateral_inhibition > 0.0:
            W_inhib = -jax.nn.softplus(params.W_inhib_raw) * params.inhib_diag_mask
        else:
            W_inhib = jnp.zeros((H, H))

        @jax.jit
        def _forward_jax(
            sensory_x: jnp.ndarray,
            mcmc_prior: jnp.ndarray,
            lengths: jnp.ndarray,
            override_g_lif: float,
            override_g_gru: float,
            do_override_lif: bool,
            do_override_gru: bool,
            init_v: jnp.ndarray,
            init_isyn: jnp.ndarray,
            init_refract: jnp.ndarray,
            init_rel_refract: jnp.ndarray,
            init_w_adapt: jnp.ndarray,
            init_x_res: jnp.ndarray,
            init_u_facil: jnp.ndarray,
            init_spike_hist: jnp.ndarray,
            init_h_gru: jnp.ndarray,
            init_dend: jnp.ndarray,
        ):
            B, T, D = sensory_x.shape

            # 1. Dendritic filtering on the SINGLE visual channel (index 0)
            # if enabled.  Wind/kinematic channels bypass it.  The filter
            # carry is accepted and exported so stateful chunked calls are
            # equivalent to one whole call.
            if params.dendritic_enabled:
                vis_trans = sensory_x[:, :, 0:1].transpose(1, 0, 2)  # (T, B, 1)
                bypass = sensory_x[:, :, 1:]  # (B, T, D-1)
                # r6 R2: an inactive row's dendritic carry must not repopulate
                # its padded frames before the encoder.  Zero the inactive
                # incoming carry for the filter (parity with the Torch
                # FrontendEncoder, which restores it by selection afterward).
                dend_init_safe = jnp.where(
                    (lengths > 0)[:, None], init_dend, 0.0,
                )
                dend_carry, vis_filtered = lax.scan(
                    lambda s, v_t: _dendritic_filter_step(params.alpha_dend, s, v_t),
                    dend_init_safe,
                    vis_trans,
                )
                vis_filtered = vis_filtered.transpose(1, 0, 2)  # (B, T, 1)
                sensory_proc = jnp.concatenate([vis_filtered, bypass], axis=-1)
                # Carry frozen at each sample's true endpoint.  A ZERO-length
                # sample has no valid frame: its carry is an exact no-op that
                # preserves the incoming dendritic carry (never the surrogate
                # frame-0 update).
                end_idx = jnp.clip(lengths - 1, 0, T - 1)          # (B,)
                dend_pick = vis_filtered[jnp.arange(B), end_idx, 0]  # (B,)
                dend_final = jnp.where(
                    lengths > 0, dend_pick, init_dend[:, 0],
                )                                                  # (B,)
            else:
                sensory_proc = sensory_x
                dend_carry = jnp.zeros((B, 1))
                dend_final = jnp.zeros((B,))

            # 2. Sensory Encoder (Linear + LayerNorm + [ReLU | SwiGLU])
            h_se = sensory_proc @ params.se_w.T + params.se_b
            h_se_norm = _apply_layernorm(h_se, params.se_ln_w, params.se_ln_b)
            if params.se_activation == "swiglu":
                gate = h_se_norm @ params.se_gate_w.T + params.se_gate_b
                e_sensory = jax.nn.silu(gate) * h_se_norm  # (B, T, H)
            else:
                e_sensory = jax.nn.relu(h_se_norm)  # (B, T, H)

            # 3. Dual-pathway scan loop (LIF + GRU fused over T)
            def scan_step(carry, step_inputs):
                (v, i_syn, ref, rel_ref, w, x_res, u_fac, spk_hist, h_gru), (e_t, mask_t) = carry, step_inputs
                m_2d = mask_t[:, None]
                act = m_2d > 0.5
                # r6 R2: keep inactive rows OUT of the recurrence arithmetic
                # BEFORE it is evaluated.  A finite-but-huge inactive carry
                # (e.g. 3e38) fed to the GRU/LIF nonlinearity overflows; the
                # discarded branch's ``NaN * 0`` mask would then leak NaN into
                # the output trajectory.  Substitute safe operands for inactive
                # rows for the recurrence ONLY; the true carry is restored by
                # ``jnp.where`` below.  Active rows use their own values.
                (v_o, i_syn_o, ref_o, rel_ref_o, w_o, x_res_o, u_fac_o,
                 spk_hist_o, h_gru_o) = (
                    v, i_syn, ref, rel_ref, w, x_res, u_fac, spk_hist, h_gru)
                v = jnp.where(act, v, 0.0)
                i_syn = jnp.where(act, i_syn, 0.0)
                ref = jnp.where(act, ref, 0.0)
                rel_ref = jnp.where(act, rel_ref, 0.0)
                w = jnp.where(act, w, 0.0)
                x_res = jnp.where(act, x_res, 1.0)
                u_fac = jnp.where(act, u_fac, 0.5)
                spk_hist = jnp.where(act, spk_hist, 0.0)
                h_gru = jnp.where(act, h_gru, 0.0)

                # --- LIF Pathway ---
                # STP decay
                if stp_enabled:
                    u_pre = jnp.clip(u_fac * decay_fac, 1e-6, 1.0)
                    x_pre = jnp.clip(1.0 - (1.0 - x_res) * decay_rec, 1e-6, 1.0)
                    stp_factor = x_pre * u_pre
                else:
                    u_pre = u_fac
                    x_pre = x_res
                    stp_factor = 1.0

                # Input projection and synaptic current
                proj = e_t @ params.lif_w_in.T + params.lif_b_in
                raw_input = lif_beta * proj * stp_factor
                i_syn_new = jnp.clip(
                    alpha_syn * i_syn + (1.0 - alpha_syn) * raw_input,
                    -i_syn_clamp,
                    i_syn_clamp,
                )

                # Absolute refractory mask & relative refractory threshold
                in_abs = (ref > 0).astype(jnp.float32)
                v_th = jnp.where(
                    k_rel > 0,
                    v_thresh + delta_theta * jnp.exp(-k_rel * rel_ref),
                    v_thresh,
                )

                # Membrane potential update
                v_new = lif_alpha * v + i_syn_new - w
                v_new = v_new * (1.0 - in_abs) + v_rest * in_abs
                v_new = jnp.clip(v_new, -v_thresh, v_clamp_max)

                # Lateral inhibition
                if lateral_inhibition > 0.0:
                    inhib_current = spk_hist @ W_inhib.T
                    v_new = v_new + lateral_inhibition * inhib_current

                # Spike detection & surrogate gradient
                raw_spk = (v_new > v_th).astype(jnp.float32)
                spk_mask = raw_spk * (1.0 - in_abs)
                sig = jax.nn.sigmoid(4.0 * (v_new - v_th))
                spike = spk_mask - lax.stop_gradient(sig) + sig

                # Lateral inhibition history update
                if lateral_inhibition > 0.0:
                    spk_hist_new = decay_inhib * spk_hist + (1.0 - decay_inhib) * spk_mask
                else:
                    spk_hist_new = spk_hist

                # Reset
                if hard_reset:
                    v_new = v_new * (1.0 - spk_mask) + v_reset * spk_mask
                else:
                    v_new = v_new - spk_mask * v_th

                # Adaptation
                w_new = jnp.clip(decay_w * w + b_adapt * spk_mask, 0.0, 10.0 * v_thresh)

                # STP update
                if stp_enabled:
                    x_new = jnp.clip(x_pre - x_pre * u_pre * spk_mask, 1e-6, 1.0)
                    u_new = jnp.clip(u_pre + U_scalar * (1.0 - u_pre) * spk_mask, 1e-6, 1.0)
                else:
                    x_new = x_res
                    u_new = u_fac

                # Refractory counters
                ref_new = jnp.where(
                    abs_refract > 0,
                    jnp.where(spk_mask > 0.5, abs_refract, jnp.clip(ref - 1.0, 0.0)),
                    ref,
                )
                rel_ref_new = jnp.where(
                    k_rel > 0,
                    jnp.where(spk_mask > 0.5, 0.0, rel_ref + 1.0),
                    rel_ref,
                )

                # --- GRU Pathway ---
                gi = e_t @ params.gru_w_ih.T + params.gru_b_ih
                gh = h_gru @ params.gru_w_hh.T + params.gru_b_hh
                gi_r, gi_z, gi_n = jnp.split(gi, 3, axis=-1)
                gh_r, gh_z, gh_n = jnp.split(gh, 3, axis=-1)
                r_gate = jax.nn.sigmoid(gi_r + gh_r)
                z_gate = jax.nn.sigmoid(gi_z + gh_z)
                n_gate = jnp.tanh(gi_n + r_gate * gh_n)
                h_gru_next = (1.0 - z_gate) * n_gate + z_gate * h_gru

                # Masking for padded steps
                out_lif = spike * m_2d
                out_pot = v_new * m_2d
                out_spk = spike * m_2d
                out_w = w_new * m_2d
                out_th = v_th * m_2d
                out_gru = h_gru_next * m_2d

                # State propagation: every per-sample recurrent/cache field is
                # frozen once the true endpoint is passed.  A padded frame must
                # not advance the membrane, synaptic current, refractory
                # counters, adaptation, STP, spike history, dendritic carry or
                # GRU hidden state.  The inactive rows are restored to their
                # ORIGINAL incoming values (r6 R2) — the recurrence above ran
                # on safe sanitized operands, so the inactive carry is an exact
                # no-op with no NaN leakage.  This matches PyTorch
                # packed-sequence masking and makes chunked stateful calls
                # equal one whole call.
                v_state = jnp.where(act, v_new, v_o)
                i_syn_state = jnp.where(act, i_syn_new, i_syn_o)
                ref_state = jnp.where(act, ref_new, ref_o)
                rel_ref_state = jnp.where(act, rel_ref_new, rel_ref_o)
                w_state = jnp.where(act, w_new, w_o)
                x_state = jnp.where(act, x_new, x_res_o)
                u_state = jnp.where(act, u_new, u_fac_o)
                spk_hist_state = jnp.where(act, spk_hist_new, spk_hist_o)
                h_gru_state = jnp.where(act, h_gru_next, h_gru_o)

                new_carry = (
                    v_state, i_syn_state, ref_state, rel_ref_state, w_state,
                    x_state, u_state, spk_hist_state, h_gru_state,
                )
                step_outputs = (out_lif, out_pot, out_spk, out_w, out_th, out_gru)
                return new_carry, step_outputs

            # Prepare inputs for scan over dimension 0 (timesteps T)
            t_idx = jnp.arange(T)[:, None]
            mask_seq = (t_idx < lengths[None, :]).astype(jnp.float32)  # (T, B)
            e_trans = e_sensory.transpose(1, 0, 2)  # (T, B, H)

            carry_init = (
                init_v, init_isyn, init_refract, init_rel_refract, init_w_adapt,
                init_x_res, init_u_facil, init_spike_hist, init_h_gru,
            )

            carry_final, (lif_out_t, pot_t, spk_t, w_t, th_t, gru_out_t) = lax.scan(
                scan_step, carry_init, (e_trans, mask_seq)
            )

            out_lif = lif_out_t.transpose(1, 0, 2)
            lif_potentials = pot_t.transpose(1, 0, 2)
            lif_spikes = spk_t.transpose(1, 0, 2)
            lif_w_adapt = w_t.transpose(1, 0, 2)
            lif_thresholds = th_t.transpose(1, 0, 2)
            out_gru = gru_out_t.transpose(1, 0, 2)
            # Raw recurrent trajectory BEFORE the output-only gain (r5 R5).
            gru_hidden_raw = out_gru

            # 4. Neuromodulatory gain on GRU if enabled
            if params.gru_neuromod_gain > 0.0:
                mcmc_safe = jnp.clip(mcmc_prior, 1e-8)
                entropy = -(mcmc_safe * jnp.log(mcmc_safe)).sum(axis=-1)
                max_entropy = math.log(params.mcmc_dim)
                entropy_norm = entropy / max_entropy
                gain = jax.nn.sigmoid(params.gain_scale * entropy_norm + params.gain_bias) * 2.0
                out_gru = out_gru * gain[..., None]

            # 5. MoR Router
            comb = jnp.concatenate([e_sensory, mcmc_prior], axis=-1)
            logits = comb @ params.router_w.T + params.router_b
            natural_gates = jax.nn.softmax(logits, axis=-1)

            g_lif = natural_gates[:, :, 0:1]
            g_gru = natural_gates[:, :, 1:2]

            g_lif = jnp.where(do_override_lif, override_g_lif, g_lif)
            g_gru = jnp.where(do_override_gru, override_g_gru, g_gru)
            effective_gates = jnp.concatenate([g_lif, g_gru], axis=-1)

            # 6. Integration & DirectionHead decode
            h_out = g_lif * out_lif + g_gru * out_gru
            h_norm = _apply_layernorm(h_out, params.dh_ln_w, params.dh_ln_b)
            if params.dh_activation == "swiglu":
                gate = h_norm @ params.dh_gate_w.T + params.dh_gate_b
                value = h_norm @ params.dh_value_w.T + params.dh_value_b
                h_act = jax.nn.silu(gate) * value
                y_pred = (h_act @ params.dh_out_w.T + params.dh_out_b).squeeze(-1)
            else:
                h_relu = jax.nn.relu(h_norm)
                y_pred = (h_relu @ params.dh_lin_w.T + params.dh_lin_b).squeeze(-1)

            if params.persistence_skip != 0.0:
                v_lag = sensory_x[:, :, 2]
                mask_bt = (t_idx < lengths[None, :]).T
                assert v_lag.shape == mask_bt.shape == (B, T)
                y_pred = y_pred + params.persistence_skip * jnp.where(
                    mask_bt, v_lag, 0.0,
                )

            return (
                y_pred, effective_gates, natural_gates,
                lif_potentials, lif_spikes, lif_thresholds,
                lif_w_adapt, out_gru, gru_hidden_raw, carry_final, dend_final,
            )

        return _forward_jax


# ===============================================================
# Main Model Interface: NSMoRCoreJAX
# ===============================================================

class NSMoRCoreJAX:
    """
    JAX-optimized inference and execution wrapper for NSMoRCore.

    Provides identical I/O contracts and shape assertions to NSMoRCore,
    utilizing compiled JAX ``lax.scan`` for maximum throughput.

    Can be constructed directly from an existing PyTorch ``NSMoRCore``
    model via :meth:`from_torch`, or initialized with the same keyword
    arguments.

    Falls back seamlessly to PyTorch NSMoRCore if JAX is unavailable.
    """

    def __init__(
        self,
        pytorch_model: Optional[NSMoRCore] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize NSMoRCoreJAX.

        Args:
            pytorch_model: Optional pre-existing NSMoRCore instance.
            **kwargs: Arguments passed to NSMoRCore if pytorch_model is None.
        """
        if pytorch_model is not None:
            self.torch_model = pytorch_model
        else:
            self.torch_model = NSMoRCore(**kwargs)

        self.sensory_dim = self.torch_model.sensory_dim
        self.mcmc_dim = self.torch_model.mcmc_dim
        self.hidden_dim = self.torch_model.hidden_dim
        self.dt_ms = self.torch_model.dt_ms
        self.persistence_skip = float(
            getattr(self.torch_model, "persistence_skip", 0.0)
        )

        # Fail closed on adaptive latent refinement (architecture v1): the
        # fused JAX kernel does not implement the refinement block.  A model
        # whose ACTUAL executed backend carries a refinement module would
        # silently drop it and return unfaithful predictions, so refuse rather
        # than diverge.  R1: inspect the EXECUTED child (backend.refinement),
        # not the stale ``refinement_mode`` string — a mode mutated to "off"
        # after construction still executes refinement in Torch.
        if refinement_module_present(self.torch_model):
            _ref_mode = getattr(
                getattr(self.torch_model, "backend", None),
                "refinement_mode", "off",
            )
            raise ValueError(
                f"NSMoRCoreJAX does not implement adaptive latent refinement "
                f"(refinement_mode={_ref_mode!r}, module present); the fused "
                f"kernel would silently drop it. Use the complete Torch "
                f"backend or construct with refinement_mode='off'."
            )

        # Fail closed on time-consuming recursion (ADR 0009): the fused kernel
        # does not implement the delayed/accumulated output map, so an enabled
        # module would be silently dropped and the kernel would return
        # undelayed predictions.  Inspect the EXECUTED child, not the string.
        if time_consuming_module_present(self.torch_model):
            _tc_mode = getattr(
                getattr(self.torch_model, "backend", None),
                "time_consuming_mode", "off",
            )
            raise ValueError(
                f"NSMoRCoreJAX does not implement time-consuming recursion "
                f"(time_consuming_mode={_tc_mode!r}, module present); the fused "
                f"kernel would silently drop it. Use the complete Torch "
                f"backend or construct with time_consuming_mode='off'."
            )

        # Submodule aliases matching PyTorch API
        self.sensory_encoder = self.torch_model.sensory_encoder
        self.lif_cell = self.torch_model.lif_cell
        self.gru_unit = self.torch_model.gru_unit
        self.router = self.torch_model.router
        self.direction_head = self.torch_model.direction_head
        self.frontend = self.torch_model.frontend
        self.backend = self.torch_model.backend

        self.use_jax = JAX_AVAILABLE
        if self.use_jax:
            # Fail closed on stacked GRU: the fused kernel maps only GRU
            # layer 0 (weight_ih_l0 ...).  A multi-layer request would
            # silently drop layers 1..N-1, so refuse it rather than diverge.
            # r6 R7: inspect the ACTUAL executed nn.GRU depth, not a stale
            # wrapper ``num_layers`` — a single-layer wrapper surrounding an
            # actual two-layer child would otherwise be accepted and copy only
            # layer 0.  A missing/inconsistent count also fails closed.
            gru = getattr(self.torch_model.backend, "gru_unit", None)
            actual_gru = getattr(gru, "gru", None)
            n_actual = getattr(actual_gru, "num_layers", None)
            n_wrapper = getattr(gru, "num_layers", None)
            if n_actual is None or n_wrapper is None or int(n_actual) != int(n_wrapper):
                raise ValueError(
                    f"NSMoRCoreJAX: GRU depth is inconsistent or unknown "
                    f"(wrapper num_layers={n_wrapper!r}, actual nn.GRU "
                    f"num_layers={n_actual!r}); refusing to map an "
                    f"unverifiable recurrent operator."
                )
            if int(n_actual) > 1:
                raise ValueError(
                    f"NSMoRCoreJAX does not implement stacked GRU "
                    f"(num_gru_layers={int(n_actual)}); only layer 0 is "
                    f"mapped. Use the complete Torch backend or set "
                    f"num_gru_layers=1."
                )
            self._params = _JAXCoreParams(self.torch_model)
            self._forward_jax = _build_jax_forward_fn(self._params)
        else:
            self._params = None
            self._forward_jax = None

    @classmethod
    def from_torch(cls, model: NSMoRCore) -> NSMoRCoreJAX:
        """Create a JAX-accelerated runner from an existing PyTorch model."""
        return cls(pytorch_model=model)

    def eval(self) -> NSMoRCoreJAX:
        """Set model to evaluation mode."""
        self.torch_model.eval()
        return self

    def train(self, mode: bool = True) -> NSMoRCoreJAX:
        """Set model to training mode.

        Note: the fused JAX kernel implements deterministic inference only.  A
        subsequent ``forward`` while the model is in training mode AND a
        stochastic mechanism (sensory noise or decoder dropout) is enabled is
        rejected with a precise error rather than silently ignoring the
        requested stochasticity.
        """
        self.torch_model.train(mode)
        return self

    def _active_stochastic_mechanisms(self) -> List[str]:
        """Return active stochastic mechanisms from the LIVE child modules.

        The fused JAX kernel implements deterministic inference only.  Whether
        a stochastic mechanism is actually active depends on the *executed*
        child modules' current mode and live probability, not on the parent's
        ``training`` flag or any constructor-copied rate (r5 R3).  A parent in
        eval mode with ``direction_head`` (or its live ``Dropout``) left in
        train mode still performs dropout, and ``Dropout.p`` may be mutated
        after runner construction; both must be detected at every call.
        """
        active: List[str] = []
        model = self.torch_model

        # Sensory noise: produced by the ACTUAL executed encoder
        # (``model.frontend.sensory_encoder``); the top-level
        # ``model.sensory_encoder`` is a historical ALIAS that may be stale
        # after the executed child is replaced (r6 R4).  Prefer the canonical
        # executed child; fall back to the alias only when the canonical
        # component genuinely does not exist.
        frontend = getattr(model, "frontend", None)
        encoder = getattr(frontend, "sensory_encoder", None)
        if encoder is None:
            encoder = getattr(model, "sensory_encoder", None)
        if encoder is not None:
            noise = float(getattr(encoder, "noise_std", 0.0) or 0.0)
            if getattr(encoder, "training", False) and noise > 0.0:
                active.append(f"sensory_noise_std={noise}")

        # Decoder dropout: inspect the ACTUAL executed head
        # (``model.backend.direction_head``), then its live Dropout modules
        # (both activation layouts) for live training mode and live p.  The
        # top-level ``model.direction_head`` is an alias and is used only when
        # the canonical executed head is absent.
        backend = getattr(model, "backend", None)
        head = getattr(backend, "direction_head", None)
        if head is None:
            head = getattr(model, "direction_head", None)
        if head is not None:
            for name, module in head.named_modules():
                if isinstance(module, torch.nn.Dropout):
                    if getattr(module, "training", False) and float(module.p) > 0.0:
                        label = f"dropout[{name or 'direction_head'}].p={module.p}"
                        active.append(label)
        return active

    def forward(
        self,
        X_batch: Union[torch.Tensor, np.ndarray, Any],
        lengths: Union[torch.Tensor, np.ndarray, Any],
        *,
        return_internals: bool = False,
        override_gates: Optional[Dict[str, float]] = None,
        states: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]], Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Execute forward pass using JAX acceleration (or fallback).

        Args:
            X_batch: (B, T, 8) input features.
            lengths: (B,) sequence lengths.
            return_internals: If True, returns dictionary of internals.
            override_gates: Optional gating override for in-silico lesions.
            states: Optional recurrent states for autoregressive simulation.

        Returns:
            y_pred or (y_pred, internals) or (y_pred, internals, states_out)
            matching the tensor type of X_batch.
        """
        input_is_torch = isinstance(X_batch, torch.Tensor)
        device = X_batch.device if input_is_torch else None

        # ── Root G3 / finding B4: validate the ORIGINAL observation dtype at
        # this public boundary BEFORE any conversion, for BOTH the real JAX
        # kernel and the Torch fallback (which otherwise casts via
        # ``_to_torch`` without an original-dtype check).  A complex
        # observation with a nonzero imaginary part, or an integer/boolean
        # observation, is refused rather than silently cast.
        _require_real_floating_obs(
            X_batch, name="X_batch", context="NSMoRCoreJAX",
        )

        if self.persistence_skip != 0.0:
            B, T, D_in = X_batch.shape
            assert D_in == self.sensory_dim + self.mcmc_dim
            if self.sensory_dim < 3:
                raise ValueError("Nonzero persistence_skip requires sensory_dim >= 3")
            l_host = np.asarray(_to_numpy(lengths))
            assert l_host.shape == (B,), f"Expected lengths ({B},), got {l_host.shape}"
            if not np.issubdtype(l_host.dtype, np.integer):
                raise ValueError("lengths must have an integer, nonboolean dtype")
            if not np.all((l_host >= 0) & (l_host <= T)):
                raise ValueError(f"lengths must satisfy 0 <= lengths <= T={T}")
            for consumer in (self.torch_model, self.torch_model.backend):
                if (
                    getattr(consumer, "target_mean", 0.0) != 0.0
                    or getattr(consumer, "target_std", 1.0) != 1.0
                    or getattr(consumer, "target_clip_cm_s", 0.0) != 0.0
                ):
                    raise ValueError(
                        "persistence_skip requires physical unnormalized/unclipped "
                        "mode"
                    )

        # Fallback validation must precede its integer cast too.
        # Fallback to PyTorch directly if JAX is unavailable
        if not self.use_jax:
            # Validate the ORIGINAL lengths BEFORE the ``torch.long`` cast:
            # a fractional/boolean length (e.g. 1.75, True) would otherwise be
            # silently truncated by ``_to_torch(..., dtype=torch.long)`` and
            # then accepted by the Torch boundary (r4 R4).
            _fb_B, _fb_T, _ = X_batch.shape
            _l_fb = np.asarray(_to_numpy(lengths))
            if _l_fb.shape != (_fb_B,):
                raise ValueError(
                    f"Expected lengths ({_fb_B},), got {_l_fb.shape}"
                )
            if not np.issubdtype(_l_fb.dtype, np.integer):
                raise ValueError("lengths must have an integer, nonboolean dtype")
            if not np.all((_l_fb >= 0) & (_l_fb <= _fb_T)):
                raise ValueError(
                    f"lengths must satisfy 0 <= lengths <= T={_fb_T}"
                )
            X_torch = _to_torch(X_batch)
            lengths_torch = _to_torch(lengths, dtype=torch.long)
            out = self.torch_model(
                X_torch, lengths_torch,
                return_internals=return_internals,
                override_gates=override_gates,
                states=states,
            )
            return out

        # Explicit deterministic-inference contract for the REAL JAX kernel:
        # it implements neither sensory noise nor decoder dropout, so it must
        # not silently accept an ACTIVE stochastic computation.  Inspect the
        # ACTUAL stochastic child modules at EVERY call (r5 R3): a parent in
        # eval mode with a child left in train mode still performs dropout, and
        # the live ``Dropout.p`` may have changed after runner construction.
        # Neither ``self.torch_model.training`` nor a constructor-copied
        # ``dropout_rate`` is authoritative.
        _active = self._active_stochastic_mechanisms()
        if _active:
            raise ValueError(
                "NSMoRCoreJAX implements deterministic inference only; "
                f"refusing active stochastic computation ({', '.join(_active)}) "
                "which the fused kernel does not implement. Use the Torch "
                "backend for stochastic training, or call eval() / set the "
                "stochastic mechanisms to 0 for deterministic execution."
            )

        # ── Shape verification on inputs ──
        B, T, D_in = X_batch.shape
        expected_dim = self.sensory_dim + self.mcmc_dim
        assert D_in == expected_dim, (
            f"Expected input feature dim {expected_dim}, got {D_in}"
        )
        assert len(lengths) == B, (
            f"lengths batch dim {len(lengths)} != {B}"
        )
        if T == 0:
            raise ValueError(
                "NSMoRCoreJAX.forward requires T >= 1; a zero-time tensor "
                "(T=0) is unsupported (distinct from lengths==0)."
            )
        _l_host = np.asarray(_to_numpy(lengths))
        if not np.issubdtype(_l_host.dtype, np.integer):
            raise ValueError("lengths must have an integer, nonboolean dtype")
        if not np.all((_l_host >= 0) & (_l_host <= T)):
            raise ValueError(f"lengths must satisfy 0 <= lengths <= T={T}")

        # Convert to JAX arrays
        X_np = _to_numpy(X_batch)
        # Sanitize invalid (padded) frames BEFORE any frontend/GRU/LIF math so
        # a finite-overflow or NaN padded suffix cannot poison a recurrent
        # carry.  Nonfinite values in a VALID frame are refused.
        _valid = np.arange(T)[None, :] < _l_host[:, None]
        if not np.isfinite(X_np[_valid]).all():
            raise ValueError(
                "NSMoRCoreJAX received nonfinite values in valid frames; "
                "refusing to sanitize real observations."
            )
        X_np = np.where(_valid[:, :, None], X_np, np.zeros_like(X_np))
        # root 3: the ACTUAL float32 computation representation of the valid
        # (non-padding) frames must be finite.  A finite original float64
        # observation (e.g. 1e300) overflows to +inf when narrowed and would
        # otherwise poison the kernels and return NaN y.  Padding remains a safe
        # no-op (already zeroed above), so this never rejects padded frames.
        _narrow = X_np.astype(np.float32)
        if not np.isfinite(_narrow[_valid]).all():
            raise ValueError(
                "NSMoRCoreJAX received finite valid-frame observations that are "
                "not representable in the float32 computation representation "
                "(overflow on narrowing); refusing to run a poisoned kernel."
            )
        sensory_x = jnp.array(X_np[:, :, :self.sensory_dim], dtype=jnp.float32)
        mcmc_prior = jnp.array(X_np[:, :, self.sensory_dim:], dtype=jnp.float32)
        lengths_arr = jnp.array(_l_host, dtype=jnp.int32)

        # Gate overrides
        override_lif_val = 0.0
        override_gru_val = 0.0
        do_override_lif = False
        do_override_gru = False
        if override_gates is not None:
            if "g_lif" in override_gates:
                override_lif_val = float(override_gates["g_lif"])
                do_override_lif = True
            if "g_gru" in override_gates:
                override_gru_val = float(override_gates["g_gru"])
                do_override_gru = True

        # Initial recurrent states.  Every supplied carry field is restored
        # INDEPENDENTLY: a partial state (e.g. only ``frontend_dendritic_state``
        # or only ``gru_h``) initializes the missing fields canonically instead
        # of silently discarding the supplied ones.  Each supplied field is
        # shape/dtype validated.
        H = self.hidden_dim
        lif_cell = self.torch_model.backend.lif_cell
        _large = float(10 * max(lif_cell.rel_refract_steps, 1))
        _u_val = (
            float(torch.sigmoid(lif_cell.U_stp_raw).item())
            if lif_cell.stp_enabled else 0.5
        )

        def _validate_carry(raw: np.ndarray, key: str) -> np.ndarray:
            """Return *raw* as float32 after checking its ORIGINAL dtype/domain.

            The supplied carry's dtype is inspected BEFORE any numeric
            conversion: a boolean/integer/complex (or otherwise non-real-
            floating) carry is rejected exactly like the Torch boundary, so a
            complex imaginary part is never silently discarded.  Finiteness AND
            the justified physical domain are checked on the ORIGINAL value
            BEFORE narrowing (r7 R3): a float64 fraction such as 1.000000001
            would round to 1.0 and slip past a post-narrowing check, so an
            invalid observation must be rejected on its original representation
            rather than repaired by the cast.  The float32 computation
            representation is then independently required to be finite.
            """
            if not np.issubdtype(raw.dtype, np.floating):
                raise ValueError(
                    f"carry field {key!r} must have a real floating dtype, "
                    f"got {raw.dtype}"
                )
            if not np.isfinite(raw).all():
                raise ValueError(
                    f"carry field {key!r} contains nonfinite values"
                )
            # Physical domain (r6 R3 / r7 R3): non-negative counters/adaptation
            # and fraction/EMA fields in [0, 1], checked on the ORIGINAL value.
            # Reject for parity with the Torch boundary rather than clamping a
            # real observation.  Membrane/GRU coordinates are unconstrained.
            if key in ("lif_refract", "lif_rel_refract", "lif_w_adapt") and bool(
                (raw < 0).any()
            ):
                raise ValueError(
                    f"carry field {key!r} must be non-negative; got a negative "
                    f"value."
                )
            if key in ("lif_x_resource", "lif_u_facil", "lif_spike_history") and (
                bool((raw < 0).any()) or bool((raw > 1.0).any())
            ):
                raise ValueError(
                    f"carry field {key!r} must lie in [0, 1] (a fraction / EMA "
                    f"history value); got a value outside that range."
                )
            arr = raw.astype(np.float32)
            # Representability of the ACTUAL float32 computation representation
            # (r7 R3): a finite original value that overflows when narrowed must
            # not export a nonfinite carry.
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"carry field {key!r} is not representable in the float32 "
                    f"computation representation."
                )
            return arr

        def _restore(key: str, default: np.ndarray) -> jnp.ndarray:
            """Return the supplied field (validated) or the canonical default."""
            if states is None or states.get(key) is None:
                arr = np.asarray(default, dtype=np.float32)
            else:
                arr = _validate_carry(_to_numpy(states[key]), key)
                if arr.shape != np.asarray(default).shape:
                    raise ValueError(
                        f"carry field {key!r} shape {arr.shape} != expected "
                        f"{np.asarray(default).shape}"
                    )
            return jnp.array(arr, dtype=jnp.float32)

        _defaults = {
            "lif_v": np.full((B, H), lif_cell.v_rest, dtype=np.float32),
            "lif_i_syn": np.zeros((B, H), dtype=np.float32),
            "lif_refract": np.zeros((B, H), dtype=np.float32),
            "lif_rel_refract": np.full((B, H), _large, dtype=np.float32),
            "lif_w_adapt": np.zeros((B, H), dtype=np.float32),
            "lif_x_resource": np.ones((B, H), dtype=np.float32),
            "lif_u_facil": np.full((B, H), _u_val, dtype=np.float32),
            "lif_spike_history": np.zeros((B, H), dtype=np.float32),
            "gru_h": np.zeros((B, H), dtype=np.float32),
        }
        init_v = _restore("lif_v", _defaults["lif_v"])
        init_isyn = _restore("lif_i_syn", _defaults["lif_i_syn"])
        init_ref = _restore("lif_refract", _defaults["lif_refract"])
        init_rel_ref = _restore("lif_rel_refract", _defaults["lif_rel_refract"])
        init_w = _restore("lif_w_adapt", _defaults["lif_w_adapt"])
        init_x_res = _restore("lif_x_resource", _defaults["lif_x_resource"])
        init_u_fac = _restore("lif_u_facil", _defaults["lif_u_facil"])
        init_spk_hist = _restore("lif_spike_history", _defaults["lif_spike_history"])

        # GRU carry: accept (B, H) or a stacked (1, B, H) (only 1 layer is
        # supported; stacked GRU is rejected at construction).
        if states is not None and states.get("gru_h") is not None:
            _gru = _validate_carry(_to_numpy(states["gru_h"]), "gru_h")
            if _gru.ndim == 3 and _gru.shape[0] == 1:
                _gru = _gru[0]
            if _gru.shape != (B, H):
                raise ValueError(
                    f"gru_h shape {_gru.shape} != (B={B}, H={H})"
                )
            init_h_gru = jnp.array(_gru, dtype=jnp.float32)
        else:
            init_h_gru = jnp.array(_defaults["gru_h"], dtype=jnp.float32)

        # Dendritic carry key matches the Torch NSMoRCore contract
        # ("frontend_dendritic_state") so a state dict produced by one backend
        # can be resumed by the other without silently dropping the low-pass
        # carry.  Legacy "lif_dendritic_state" is accepted as a fallback for
        # previously persisted raw-JAX states.  A historical (B, 2) width is
        # MIGRATED (visual column 0); any other width/rank is rejected.
        _dend_in = None
        if states is not None:
            _dend_in = states.get(
                "frontend_dendritic_state",
                states.get("lif_dendritic_state", None),
            )
        if _dend_in is None:
            init_dend = jnp.zeros((B, 1), dtype=jnp.float32)
        else:
            # Inspect the ORIGINAL dtype/rank/width BEFORE any cast, so an
            # integer or complex carry is rejected (never silently truncated
            # or stripped of its imaginary part) exactly as the Torch boundary
            # does.  The documented legacy (B, 2) -> visual-column migration
            # and the real-floating conversion are preserved.
            _d = _validate_carry(_to_numpy(_dend_in), "frontend_dendritic_state")
            if _d.ndim != 2 or _d.shape[0] != B:
                raise ValueError(
                    f"frontend_dendritic_state shape {_d.shape} != (B={B}, 1)"
                )
            if _d.shape[1] == 2:
                _d = _d[:, 0:1]
            elif _d.shape[1] != 1:
                raise ValueError(
                    "frontend_dendritic_state must be (B, 1) or legacy "
                    f"(B, 2); got width {_d.shape[1]}"
                )
            init_dend = jnp.array(_d, dtype=jnp.float32)

        # Call compiled JAX kernel
        (
            y_pred_j, effective_gates_j, natural_gates_j,
            lif_potentials_j, lif_spikes_j, lif_thresholds_j,
            lif_w_adapt_j, out_gru_j, gru_hidden_raw_j, carry_final, dend_final,
        ) = self._forward_jax(
            sensory_x, mcmc_prior, lengths_arr,
            override_lif_val, override_gru_val,
            do_override_lif, do_override_gru,
            init_v, init_isyn, init_ref, init_rel_ref, init_w,
            init_x_res, init_u_fac, init_spk_hist, init_h_gru, init_dend,
        )

        # Convert outputs to match caller's type (PyTorch Tensor or JAX array)
        if input_is_torch:
            y_pred = _to_torch(y_pred_j, device=device)
            internals = {
                "routing_gates": _to_torch(effective_gates_j, device=device),
                "natural_gates": _to_torch(natural_gates_j, device=device),
                "lif_potentials": _to_torch(lif_potentials_j, device=device),
                "lif_spikes": _to_torch(lif_spikes_j, device=device),
                "lif_thresholds": _to_torch(lif_thresholds_j, device=device),
                "lif_w_adapt": _to_torch(lif_w_adapt_j, device=device),
                "gru_hidden": _to_torch(out_gru_j, device=device),
                "gru_hidden_raw": _to_torch(gru_hidden_raw_j, device=device),
            }
        else:
            y_pred = y_pred_j
            internals = {
                "routing_gates": effective_gates_j,
                "natural_gates": natural_gates_j,
                "lif_potentials": lif_potentials_j,
                "lif_spikes": lif_spikes_j,
                "lif_thresholds": lif_thresholds_j,
                "lif_w_adapt": lif_w_adapt_j,
                "gru_hidden": out_gru_j,
                "gru_hidden_raw": gru_hidden_raw_j,
            }

        # Assert output shapes
        assert y_pred.shape == (B, T), f"y_pred shape {y_pred.shape} != ({B}, {T})"
        assert internals["routing_gates"].shape == (B, T, 2)
        assert internals["gru_hidden"].shape == (B, T, H)
        assert internals["gru_hidden_raw"].shape == (B, T, H)
        assert internals["lif_spikes"].shape == (B, T, H)

        # Autoregressive state output
        if states is not None:
            (v_fin, i_syn_fin, ref_fin, rel_ref_fin, w_fin,
             x_fin, u_fin, spk_hist_fin, h_gru_fin) = carry_final
            if input_is_torch:
                states_out = {
                    "lif_v": _to_torch(v_fin, device=device),
                    "lif_i_syn": _to_torch(i_syn_fin, device=device),
                    "lif_refract": _to_torch(ref_fin, device=device),
                    "lif_w_adapt": _to_torch(w_fin, device=device),
                    "lif_rel_refract": _to_torch(rel_ref_fin, device=device),
                    "gru_h": _to_torch(h_gru_fin, device=device).unsqueeze(0),
                    "lif_spike_history": _to_torch(spk_hist_fin, device=device),
                }
                if lif_cell.stp_enabled:
                    states_out["lif_x_resource"] = _to_torch(x_fin, device=device)
                    states_out["lif_u_facil"] = _to_torch(u_fin, device=device)
                if self._params.dendritic_enabled:
                    # Match the Torch NSMoRCore key so cross-backend resume
                    # preserves the dendritic carry (see init above).  The
                    # dendritic filter lives on the FRONTEND, so the guard is
                    # the frontend flag, not ``lif_cell._dendritic_enabled``.
                    states_out["frontend_dendritic_state"] = _to_torch(
                        dend_final, device=device,
                    ).unsqueeze(-1)
            else:
                states_out = {
                    "lif_v": v_fin,
                    "lif_i_syn": i_syn_fin,
                    "lif_refract": ref_fin,
                    "lif_w_adapt": w_fin,
                    "lif_rel_refract": rel_ref_fin,
                    "gru_h": h_gru_fin[None, :, :],
                    "lif_spike_history": spk_hist_fin,
                }
                if lif_cell.stp_enabled:
                    states_out["lif_x_resource"] = x_fin
                    states_out["lif_u_facil"] = u_fin
                if self._params.dendritic_enabled:
                    states_out["frontend_dendritic_state"] = dend_final[:, None]
            return y_pred, internals, states_out

        if return_internals:
            return y_pred, internals

        return y_pred

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
