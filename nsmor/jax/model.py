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
Bidirectional parameter mapping uses the PyTorch NSMoRCore weight layout.
Weight layout compatibility does not certify canonical analysis checkpoint
provenance or numerical equivalence. JAX training artifacts are development only.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import torch


# ===============================================================
# Biophysical Helper Functions & Decay Computations
# ===============================================================

def validate_lengths(lengths: Any, batch_size: int, timesteps: int) -> None:
    """Validate nonzero-skip lengths on the host before JIT or integer casts.

    Shape/dtype checks also run under tracing. Direct JIT callers must perform
    this host preflight themselves: tracer values cannot be range-checked here.
    Supported training and analysis wrappers perform the host check.
    """
    assert lengths.shape == (batch_size,), (
        f"Expected lengths ({batch_size},), got {lengths.shape}"
    )
    if not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError("lengths must have an integer, nonboolean dtype")
    if not isinstance(lengths, jax.core.Tracer):
        values = np.asarray(lengths)
        if not np.all((values >= 0) & (values <= timesteps)):
            raise ValueError(f"lengths must satisfy 0 <= lengths <= T={timesteps}")


def validate_input_and_lengths(
    x: Any, lengths: Any, *, context: str,
) -> None:
    """Unconditionally validate a Flax input batch and its lengths.

    Shared host-boundary preflight for every supported analysis/UQ caller
    (the eval wrapper, the MC-dropout analyzer and direct eager
    ``NSMoRModel.apply``).  It validates BEFORE any integer cast or compiled
    call:

      * ``x`` is ``(B, T, D)`` with ``T > 0``;
      * ``lengths`` is ``(B,)`` with a nonboolean integer dtype;
      * every length satisfies ``0 <= lengths <= T`` (``lengths == 0`` is a
        legal zero-length no-op that is preserved; ``T == 0`` is refused);
      * the VALID frames of ``x`` are finite (an invalid padded suffix is
        harmless and remains the caller's responsibility to zero).

    Shape and dtype checks are static and therefore also run under tracing;
    the value/range/finiteness checks need concrete host arrays and are
    skipped for traced values, so a JIT-compiled caller must run this on the
    host first (see the checked-host contract in the callers).

    Args:
        x: Input batch, array-like ``(B, T, D)``.
        lengths: True sequence lengths, array-like ``(B,)``.
        context: Caller label used in error messages.

    Raises:
        ValueError: On a shape, dtype, range or finiteness violation.
    """
    ndim = getattr(x, "ndim", None)
    if ndim is None:
        x = np.asarray(x)
        ndim = x.ndim
    assert ndim == 3, f"{context}: x must be 3-D (B, T, D), got {ndim}-D"
    B, T, _D = x.shape
    # Root G3 / finding B4: validate the ORIGINAL observation dtype BEFORE any
    # integer cast or compiled call.  A complex measurement (whose imaginary
    # part a later float cast would silently discard) or an integer/boolean
    # observation is refused rather than coerced.  The dtype check is static
    # and therefore also runs under tracing; a torch tensor (whose dtype is not
    # a numpy dtype) is validated by its own boundary instead.
    try:
        _x_dtype = np.dtype(x.dtype)
    except TypeError:
        _x_dtype = None
    if _x_dtype is not None and not np.issubdtype(_x_dtype, np.floating):
        raise ValueError(
            f"{context}: x must have a real floating dtype, got {_x_dtype}. "
            "Integer/boolean/complex observations are refused before "
            "conversion."
        )
    if T == 0:
        raise ValueError(
            f"{context}: x requires T >= 1; a zero-time tensor (T=0) is "
            f"unsupported (distinct from lengths==0, a valid no-op)."
        )
    assert lengths.shape == (B,), (
        f"{context}: expected lengths ({B},), got {lengths.shape}"
    )
    if not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError(
            f"{context}: lengths must have an integer, nonboolean dtype"
        )
    if isinstance(x, jax.core.Tracer) or isinstance(lengths, jax.core.Tracer):
        return
    values = np.asarray(lengths)
    if not np.all((values >= 0) & (values <= T)):
        raise ValueError(f"{context}: lengths must satisfy 0 <= lengths <= T={T}")
    valid = np.arange(T)[None, :] < values[:, None]
    x_arr = np.asarray(x)
    if not np.isfinite(x_arr[valid]).all():
        raise ValueError(
            f"{context}: x contains nonfinite values in valid frames; "
            f"refusing to sanitize real observations."
        )
    # H5 / finding A7: the ACTIVE (valid-frame) observations must be finite in
    # the ACTUAL float32 computation representation.  A finite float64 value
    # such as 1e300 overflows to +inf when JAX/Dense narrows to float32 and
    # would return NaN predictions; refuse it at this host preflight.  Only the
    # valid frames are cast (padding is sanitized to zero by the model), so a
    # harmless overflowing padded suffix is never rejected.
    with np.errstate(over="ignore"):
        active_f32 = x_arr[valid].astype(np.float32)
    if not np.isfinite(active_f32).all():
        raise ValueError(
            f"{context}: active observations are not representable in the "
            f"float32 computation representation (overflow on narrowing); "
            f"refusing to run a poisoned kernel."
        )


def _f(value: Any) -> Optional[float]:
    """Return *value* as a rounded float, or ``None`` when it is absent/off."""
    if value is None:
        return None
    return round(float(value), 9)


def _attr(obj: Any, *names: str) -> Any:
    """First present attribute among *names* on *obj*, else ``None``."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _semantic_signature(source: Any) -> Dict[str, Any]:
    """Normalize the *supported* computation settings of *source*.

    Every known setting that changes the Flax-representable computation is
    normalized into a comparable value: the physical grid (``dt_ms``) and the
    LIF dynamics/stochastic/router/gain/truncation/persistence settings.  Only
    attributes that actually exist on the object are included, so a PyTorch
    ``NSMoRCore`` (attributes on the model, its ``backend``/``lif_cell``), a
    Flax ``NSMoRModel`` and a ``ModelConfig`` all normalize consistently.
    Initializer-only constructor values that the runtime does not use are not
    read; learned tensors are never inspected here.
    """
    lif = getattr(source, "lif_cell", None)
    backend = getattr(source, "backend", source)
    out: Dict[str, Any] = {}

    def _put(key: str, value: Any) -> None:
        if value is not None:
            out[key] = value

    # Physical grid.
    _put("dt_ms", _f(getattr(source, "dt_ms", None)))
    # LIF dynamics.
    if lif is not None:
        _put("lif_alpha", _f(_attr(lif, "alpha")))
        _put("lif_threshold", _f(_attr(lif, "v_threshold")))
        _put("lif_beta", _f(_attr(lif, "beta")))
        _put("lif_tau_syn", _f(_attr(lif, "tau_syn")))
        _put("lif_tau_w", _f(_attr(lif, "tau_w")))
        _put("lif_b_adapt", _f(_attr(lif, "b_adapt")))
        _put("lif_v_rest", _f(_attr(lif, "v_rest")))
        _put("lif_abs_refract_ms", _f(_attr(lif, "abs_refract_ms")))
        _put("lif_rel_refract_ms", _f(_attr(lif, "rel_refract_ms")))
        _put("lif_lateral_inhibition", _f(_attr(lif, "lateral_inhibition")))
    else:
        for key in (
            "lif_alpha", "lif_threshold", "lif_beta", "lif_tau_syn",
            "lif_tau_w", "lif_b_adapt", "lif_v_rest", "lif_abs_refract_ms",
            "lif_rel_refract_ms", "lif_lateral_inhibition",
        ):
            _put(key, _f(getattr(source, key, None)))
    # Effective lateral-inhibition history decay (r4 R3).  The actual
    # recurrence uses ``max(tau_syn, inhib_tau_ms)`` in BOTH backends: Torch
    # stores it as ``lif_cell._inhib_tau_ms``, Flax as ``lif_inhib_tau_ms``
    # with ``max(lif_tau_syn, lif_inhib_tau_ms)``.  Normalizing the effective
    # value (not the inert constructor argument) makes an enabled-mechanism
    # mismatch reject before any parameter mapping.  Present only when
    # inhibition is actually enabled, so a disabled mechanism never triggers
    # a spurious mismatch.
    tau_syn_eff = None
    inhib_strength = None
    inhib_tau_eff = None
    if lif is not None:
        tau_syn_eff = _attr(lif, "tau_syn")
        inhib_strength = _attr(lif, "lateral_inhibition")
        inhib_tau_eff = _attr(lif, "_inhib_tau_ms")
    else:
        tau_syn_eff = getattr(source, "lif_tau_syn", None)
        inhib_strength = getattr(source, "lif_lateral_inhibition", None)
        inhib_tau_eff = getattr(source, "lif_inhib_tau_ms", None)
    if (inhib_strength is not None and float(inhib_strength) > 0.0
            and tau_syn_eff is not None and inhib_tau_eff is not None):
        _put("lif_effective_inhib_tau_ms",
             _f(max(float(tau_syn_eff), float(inhib_tau_eff))))
    # Router / truncation / gain / persistence / stochastic / activation.
    # These live on different sub-objects depending on the source kind: a
    # Flax NSMoRModel exposes them as top-level fields, a PyTorch NSMoRCore
    # stores the GRU depth on ``gru_unit``, the TBPTT window and gain on its
    # ``backend``, and the noise/dropout on its encoder/decoder submodules.
    tbptt = _attr(source, "lif_tbptt_steps", "_tbptt_steps")
    if tbptt is None:
        tbptt = _attr(backend, "lif_tbptt_steps", "_tbptt_steps")
    _put("lif_tbptt_steps", tbptt)

    gain = _attr(source, "gru_neuromod_gain")
    if gain is None:
        gain = _attr(backend, "gru_neuromod_gain")
    _put("gru_neuromod_gain", _f(gain))

    _put("persistence_skip", _f(getattr(source, "persistence_skip", None)))
    _put("activation", getattr(source, "activation", None))
    # Adaptive latent refinement (architecture v1): the Flax destination only
    # ever runs "off"; a Torch source with an enabled mode must reject here.
    _put("refinement_mode", getattr(source, "refinement_mode", None))
    # Time-consuming recursion (ADR 0009): the Flax destination only ever runs
    # "off"; a Torch source with an enabled mode must reject here.
    _put("time_consuming_mode", getattr(source, "time_consuming_mode", None))

    noise = _attr(source, "sensory_noise_std")
    if noise is None:
        se = getattr(source, "sensory_encoder", None)
        noise = _attr(se, "noise_std") if se is not None else None
    if noise is None:
        frontend = getattr(source, "frontend", None)
        noise = _attr(getattr(frontend, "sensory_encoder", None), "noise_std")
    _put("sensory_noise_std", _f(noise))

    dropout = _attr(source, "dropout_rate")
    if dropout is None:
        dropout = _attr(source, "dropout")
    if dropout is None:
        dropout = _attr(getattr(source, "direction_head", None), "dropout_rate")
    _put("dropout", _f(dropout))

    n_layers = _attr(source, "num_gru_layers")
    if n_layers is None:
        n_layers = _attr(getattr(source, "gru_unit", None), "num_layers")
    _put("num_gru_layers", n_layers)
    return out


def assert_semantics_match(source: Any, destination: Any, *, context: str) -> None:
    """Reject a known *source* whose supported settings differ from *destination*.

    Only settings present on BOTH signatures are compared, so a config that
    omits a field never triggers a spurious mismatch.  The bare-state_dict API
    (``source=None``) is unaffected and stays parameter-layout-only.
    """
    src = _semantic_signature(source)
    dst = _semantic_signature(destination)
    diffs = {
        key: (src[key], dst[key])
        for key in src.keys() & dst.keys()
        if src[key] != dst[key]
    }
    if diffs:
        detail = ", ".join(f"{k}: source={v[0]} destination={v[1]}"
                           for k, v in sorted(diffs.items()))
        raise ValueError(
            f"{context}: source semantic settings do not match the Flax "
            f"destination ({detail}). Refusing to map parameters across a "
            f"divergent computation — use a matching destination or the "
            f"Torch backend."
        )
    # Effective derived runtime coefficients (r5 R4 / r6 R6): the
    # declaration-level comparison above cannot detect a live Torch model whose
    # *registered nonpersistent* decay buffers (``_alpha_syn``, ``_decay_w``,
    # ``_decay_inhib``) or derived refractory/clamp scalars were mutated away
    # from their canonical values.  Such a source executes materially different
    # dynamics that the Flax destination cannot reproduce (it recomputes them).
    # Fail closed on any inconsistent/nonfinite derived coefficient.
    _assert_effective_runtime_coefficients(source, context=context)
    # Actual executed normalization/activation/dropout topology (r6 R5): the
    # destination hard-codes eps=1e-5 and an activation chosen from a string.
    # A live source with a different LayerNorm eps, a replaced activation
    # module, or a live dropout p the destination cannot reproduce must fail
    # closed rather than silently diverge.
    _assert_live_topology(source, context=context)


def _assert_live_topology(source: Any, *, context: str) -> None:
    """Reject a known live source whose executed norm/activation/dropout topology
    the Flax destination cannot reproduce (r6 R5 / r7 R1).

    Delegates to the single shared exact certificate
    :func:`nsmor.model_nsmor_core.certify_executed_topology`, which inspects the
    ACTUAL executed child modules/slots/semantics (ordered module types,
    LayerNorm epsilon, dropout placement/rate, required projections) rather than
    an allowed-type whitelist.  A Flax ``NSMoRModel``/config has no Torch
    modules and is a no-op.
    """
    from nsmor.model_nsmor_core import certify_executed_topology

    certify_executed_topology(source, context=context)


def _assert_effective_runtime_coefficients(
    source: Any, *, context: str, atol: float = 1e-6,
) -> None:
    """Reject a live Torch source whose derived runtime coefficients are noncanonical.

    Only fires for objects exposing a ``lif_cell`` (a real PyTorch model or its
    ``backend``); a Flax ``NSMoRModel`` or ``ModelConfig`` has no such buffers
    and is a no-op.  Every enabled derived coefficient actually used by the
    recurrent computation is compared against its canonical value with a
    float32 tolerance, so a normally-constructed model never spuriously
    rejects.  Finiteness is checked BEFORE the tolerance comparison: a NaN
    buffer would otherwise bypass ``abs(NaN-expected) > atol`` (r6 R6).
    """
    lif = getattr(source, "lif_cell", None)
    if lif is None:
        backend = getattr(source, "backend", None)
        lif = getattr(backend, "lif_cell", None)
    if lif is None:
        return
    dt_ms = float(getattr(source, "dt_ms", None)
                  or getattr(getattr(source, "backend", None), "dt_ms", 0.0)
                  or 0.0)

    def _canon_decay(tau_ms: float) -> float:
        return 0.0 if tau_ms <= 0.0 else float(math.exp(-dt_ms / tau_ms))

    def _ceil_steps(ms: float) -> int:
        return int(math.ceil(float(ms) / dt_ms)) if ms > 0.0 else 0

    v_thresh = float(getattr(lif, "v_threshold", 0.0) or 0.0)
    tau_syn = float(getattr(lif, "tau_syn", 0.0) or 0.0)
    tau_w = float(getattr(lif, "tau_w", 0.0) or 0.0)
    rel_ms = float(getattr(lif, "rel_refract_ms", 0.0) or 0.0)
    abs_ms = float(getattr(lif, "abs_refract_ms", 0.0) or 0.0)
    rel_steps = _ceil_steps(rel_ms)

    checks: list[tuple[str, Any, float]] = []
    # Synaptic filter: always registered.
    checks.append(("_alpha_syn", getattr(lif, "_alpha_syn", None),
                   _canon_decay(tau_syn)))
    # Adaptation decay: always registered.
    checks.append(("_decay_w", getattr(lif, "_decay_w", None),
                   _canon_decay(tau_w)))
    # Lateral-inhibition decay: present only when inhibition is enabled.
    if float(getattr(lif, "lateral_inhibition", 0.0) or 0.0) > 0.0:
        checks.append(("_decay_inhib", getattr(lif, "_decay_inhib", None),
                       _canon_decay(float(getattr(lif, "_inhib_tau_ms", 0.0) or 0.0))))
    # r6 R6: the full supported derived-coefficient inventory used by the
    # recurrence (refractory threshold, relative-refractory rate, membrane /
    # synaptic clamps, refractory step counts).  These are recomputed by the
    # Flax destination, so a live source whose actual values differ would
    # execute materially different spike/potential dynamics.
    checks.append(("_delta_theta", getattr(lif, "_delta_theta", None),
                   0.3 * v_thresh))
    checks.append(("_k_rel", getattr(lif, "_k_rel", None),
                   (1.0 / rel_steps) if rel_steps > 0 else 0.0))
    checks.append(("_v_clamp_max", getattr(lif, "_v_clamp_max", None),
                   3.0 * v_thresh))
    checks.append(("_i_syn_clamp", getattr(lif, "_i_syn_clamp", None),
                   5.0 * v_thresh))
    checks.append(("abs_refract_steps", getattr(lif, "abs_refract_steps", None),
                   float(_ceil_steps(abs_ms))))
    checks.append(("rel_refract_steps", getattr(lif, "rel_refract_steps", None),
                   float(rel_steps)))

    inconsistent: list[str] = []
    for name, buffer, expected in checks:
        if buffer is None:
            continue
        try:
            actual = float(buffer)
        except (TypeError, ValueError):
            inconsistent.append(f"{name}=<non-scalar>")
            continue
        # r6 R6: finiteness BEFORE the tolerance comparison (NaN bypass).
        if not math.isfinite(actual):
            inconsistent.append(f"{name}={actual!r} (nonfinite)")
            continue
        if abs(actual - expected) > atol:
            inconsistent.append(
                f"{name}={actual:.9g} (canonical={expected:.9g})"
            )
    if inconsistent:
        raise ValueError(
            f"{context}: live source has noncanonical effective recurrence "
            f"coefficients that the Flax destination cannot reproduce "
            f"({'; '.join(inconsistent)}). Refusing to map parameters across "
            f"divergent runtime dynamics."
        )


def compute_decay_factor(tau_ms: float, dt_ms: float) -> float:
    """Compute per-timestep exponential decay factor alpha = exp(-dt / tau)."""
    if tau_ms <= 0.0:
        return 0.0
    return float(math.exp(-dt_ms / tau_ms))


def assert_flax_supported(source: Any, *, context: str) -> None:
    """Fail closed when *source* enables a setting the Flax path cannot honor.

    The Flax ``NSMoRModel`` implements neither stacked GRU, frontend dendritic
    filtering, short-term plasticity (STP) nor an explicit hard reset.  *source*
    may be a PyTorch ``NSMoRCore`` (attributes on the model, its ``backend``,
    ``frontend`` or ``lif_cell``) or a Flax ``NSMoRModel`` config; the enabled
    mechanisms are read from whichever attribute names the object exposes.
    Raises ``ValueError`` naming the unsupported setting(s) BEFORE any
    conversion, mapping, eval, training or UQ so a strict ``--backend jax``
    request can never silently fall back to a divergent computation.

    Args:
        source: Object exposing the relevant config attributes.
        context: Caller label used in the error message.

    Raises:
        ValueError: If any unsupported mechanism is enabled.
    """
    unsupported: list[str] = []

    # Stacked GRU: PyTorch exposes ``gru_unit.num_layers``; the Flax config
    # field is ``num_gru_layers``.
    n_layers = getattr(source, "num_gru_layers", None)
    if n_layers is None:
        n_layers = getattr(getattr(source, "gru_unit", None), "num_layers", 1)
    if n_layers is not None and int(n_layers) > 1:
        unsupported.append(f"num_gru_layers={int(n_layers)}")

    # Frontend dendritic filtering: PyTorch stores ``dendritic_tau`` on the
    # frontend encoder (``lif_dendritic_tau`` in config / the Flax field).
    dendritic = getattr(source, "lif_dendritic_tau", None)
    if dendritic is None:
        frontend = getattr(source, "frontend", None)
        dendritic = getattr(frontend, "dendritic_tau", 0.0) if frontend else 0.0
    if float(dendritic or 0.0) > 0.0:
        unsupported.append("lif_dendritic_tau>0")

    # STP: PyTorch enables it via ``lif_cell.stp_enabled`` (both tau>0); the
    # Flax fields are ``lif_tau_fac`` / ``lif_tau_rec``.
    lif_cell = getattr(source, "lif_cell", None)
    stp_enabled = getattr(lif_cell, "stp_enabled", None)
    if stp_enabled is None:
        stp_enabled = (
            float(getattr(source, "lif_tau_fac", 0.0) or 0.0) > 0.0
            and float(getattr(source, "lif_tau_rec", 0.0) or 0.0) > 0.0
        )
    if stp_enabled:
        unsupported.append("lif_tau_fac>0 / lif_tau_rec>0 (STP)")

    # Hard reset: PyTorch records ``lif_cell._hard_reset``; the Flax field is
    # ``lif_v_reset``.
    hard_reset = getattr(lif_cell, "_hard_reset", None)
    if hard_reset is None:
        hard_reset = getattr(source, "lif_v_reset", None) is not None
    if hard_reset:
        unsupported.append("lif_v_reset set (hard reset)")

    # Adaptive latent refinement (architecture v1): the Flax graph does not
    # implement the refinement block.  R1: inspect the EXECUTED child
    # (``backend.refinement``), not only the mode string — a mode mutated to
    # "off" after construction still executes refinement in Torch.
    from nsmor.model_nsmor_core import refinement_module_present

    if refinement_module_present(source):
        ref_mode = getattr(source, "refinement_mode", None)
        if ref_mode in (None, "off"):
            ref_mode = getattr(
                getattr(source, "backend", None), "refinement_mode", None,
            )
        unsupported.append(f"refinement module present (mode={ref_mode!r})")

    # Time-consuming recursion (ADR 0009): the Flax graph does not implement
    # the delayed/accumulated output map.  Inspect the EXECUTED child, not the
    # mode string, so a mode mutated to "off" after construction is refused.
    from nsmor.model_nsmor_core import time_consuming_module_present

    if time_consuming_module_present(source):
        tc_mode = getattr(source, "time_consuming_mode", None)
        if tc_mode in (None, "off"):
            tc_mode = getattr(
                getattr(source, "backend", None), "time_consuming_mode", None,
            )
        unsupported.append(f"time-consuming module present (mode={tc_mode!r})")

    if unsupported:
        raise ValueError(
            f"{context}: JAX/Flax backend does not implement "
            f"{', '.join(unsupported)}; the Torch backend is complete for "
            f"these settings. Refusing to silently diverge — use "
            f"--backend torch or disable the listed mechanisms."
        )


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
    """Sensory projection: Linear(4, H) -> LayerNorm -> [ReLU | SwiGLU].

    ``activation="swiglu"`` computes the genuine gated activation
    ``SiLU(gate_proj(h)) * h`` (Shazeer 2020).

    Optionally injects Gaussian noise during training to model intrinsic
    neural variability and stochastic resonance.
    Ref: Douglass et al. 1993, Nature 365:721-723.
    """
    hidden_dim: int = 64
    sensory_noise_std: float = 0.0
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        assert x.ndim >= 2, f"SensoryEncoderJAX input must be >= 2-D, got {x.ndim}-D"
        h = nn.Dense(self.hidden_dim, name="dense")(x)
        # Match PyTorch LayerNorm exactly: centered variance and eps=1e-5
        # (Flax defaults to 1e-6 and a fast uncentered variance that diverge
        # on near-constant low-variance inputs).
        h = nn.LayerNorm(
            name="ln", epsilon=1e-5, use_fast_variance=False,
        )(h)
        if self.activation == "swiglu":
            gate = nn.Dense(self.hidden_dim, name="gate")(h)
            h = jax.nn.silu(gate) * h
        else:
            h = nn.relu(h)
        # MINOR-4 fix: Stochastic resonance noise injection (training only)
        if not deterministic and self.sensory_noise_std > 0.0:
            noise = jax.random.normal(
                self.make_rng("dropout"), h.shape
            ) * self.sensory_noise_std
            h = h + noise
        return h


class MoRRouterJAX(nn.Module):
    """MoR representational-routing gate: Linear(H + M, 2) -> Softmax.

    The two weights are coupled (softmax sums to 1); this is not a
    causal-inference estimator.
    """
    @nn.compact
    def __call__(self, e_sensory: jnp.ndarray, mcmc_prior: jnp.ndarray) -> jnp.ndarray:
        assert e_sensory.shape[:-1] == mcmc_prior.shape[:-1], (
            f"Batch/time dims must match: {e_sensory.shape[:-1]} vs {mcmc_prior.shape[:-1]}"
        )
        comb = jnp.concatenate([e_sensory, mcmc_prior], axis=-1)
        logits = nn.Dense(2, name="gate")(comb)
        gates = jax.nn.softmax(logits, axis=-1)
        assert gates.shape == (*e_sensory.shape[:-1], 2), (
            f"Router output shape {gates.shape} != expected {(*e_sensory.shape[:-1], 2)}"
        )
        return gates


class DirectionHeadJAX(nn.Module):
    """Direction decoder: LayerNorm -> [ReLU | SwiGLU] -> Dropout -> Linear.

    ``activation="swiglu"`` computes the genuine gated activation
    ``SiLU(gate_proj(n)) * value_proj(n)`` followed by a ``Linear(H, 1)``
    readout (Shazeer 2020).
    """
    hidden_dim: int = 64
    dropout_rate: float = 0.1
    activation: str = "relu"

    @nn.compact
    def __call__(self, h: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        assert h.ndim >= 2, f"DirectionHeadJAX input must be >= 2-D, got {h.ndim}-D"
        leading_shape = h.shape[:-1]
        # Match PyTorch LayerNorm exactly: centered variance and eps=1e-5.
        h_norm = nn.LayerNorm(
            name="ln", epsilon=1e-5, use_fast_variance=False,
        )(h)
        if self.activation == "swiglu":
            # PyTorch drops the NORMALIZED input BEFORE both gate/value
            # projections (Sequential(LayerNorm, Dropout) then gate/value).
            # Dropout must be applied here, not after the gated product,
            # or the training distribution and MC-dropout uncertainty
            # diverge from PyTorch even with identical parameters.
            n = h_norm
            if not deterministic and self.dropout_rate > 0.0:
                n = nn.Dropout(self.dropout_rate, deterministic=False)(n)
            gate = nn.Dense(self.hidden_dim, name="gate")(n)
            value = nn.Dense(self.hidden_dim, name="value")(n)
            h_act = jax.nn.silu(gate) * value
            y = nn.Dense(1, name="out")(h_act)
        else:
            h_act = nn.relu(h_norm)
            if not deterministic and self.dropout_rate > 0.0:
                h_act = nn.Dropout(self.dropout_rate, deterministic=False)(h_act)
            y = nn.Dense(1, name="dense")(h_act)
        y = jnp.squeeze(y, axis=-1)
        assert y.shape == leading_shape, (
            f"DirectionHead output shape {y.shape} != expected {leading_shape}"
        )
        return y


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
    persistence_skip: float = 0.0
    activation: str = "relu"
    refinement_mode: str = "off"
    time_consuming_mode: str = "off"

    def setup(self) -> None:
        if (
            isinstance(self.persistence_skip, bool)
            or not isinstance(self.persistence_skip, (int, float))
            or not math.isfinite(self.persistence_skip)
            or not 0.0 <= self.persistence_skip <= 1.0
        ):
            raise ValueError("persistence_skip must be a finite scalar in [0, 1]")
        if self.activation not in ("relu", "swiglu"):
            raise ValueError(
                f"activation must be 'relu' or 'swiglu', got {self.activation!r}"
            )
        # Fail closed on adaptive latent refinement (architecture v1): the Flax
        # graph does not implement the refinement block, so an enabled mode
        # must be refused rather than silently dropped (unfaithful predictions).
        if self.refinement_mode not in ("off", None):
            raise ValueError(
                f"NSMoRModel (Flax) does not implement adaptive latent "
                f"refinement (refinement_mode={self.refinement_mode!r}); use "
                f"the complete Torch backend or set refinement_mode='off'."
            )
        # Time-consuming recursion (ADR 0009): the Flax graph does not
        # implement the delayed/accumulated output map; refuse an enabled mode.
        if self.time_consuming_mode not in ("off", None):
            raise ValueError(
                f"NSMoRModel (Flax) does not implement time-consuming "
                f"recursion (time_consuming_mode={self.time_consuming_mode!r}); "
                f"use the complete Torch backend or set "
                f"time_consuming_mode='off'."
            )
        self.sensory_encoder = SensoryEncoderJAX(
            hidden_dim=self.hidden_dim,
            sensory_noise_std=self.sensory_noise_std,
            activation=self.activation,
        )
        self.router = MoRRouterJAX()
        self.direction_head = DirectionHeadJAX(
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
        )

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
        sensory_noise: bool = True,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]]:
        """
        Forward pass over variable-length sequence batch.

        Args:
            x: (B, T, 8) input features.
            lengths: (B,) true sequence lengths.
            deterministic: disable dropout during eval.
            override_gates: optional lesion override dict {'g_lif': ..., 'g_gru': ...}.
            return_internals: if True, returns dictionary of internal states.
            sensory_noise: if False, suppress the intrinsic Gaussian sensory
                noise even when ``deterministic=False``.  This lets a
                dropout-only estimator isolate epistemic (dropout) from
                aleatoric (input-noise) dispersion at the analysis interface
                without disabling training noise globally.  Dropout remains
                governed solely by ``deterministic``.

        Returns:
            y_pred: (B, T) predicted velocity.
            internals (optional): dictionary of internal tensors.
        """
        # MINOR-5 fix: Shape assertions on critical tensors
        assert x.ndim == 3, f"Input must be 3-D (B, T, D), got {x.ndim}-D"
        B, T, D = x.shape
        H = self.hidden_dim
        dt = self.dt_ms
        if T == 0:
            raise ValueError(
                "NSMoRModel.forward requires T >= 1; a zero-time tensor (T=0) "
                "is unsupported (distinct from lengths==0, a valid no-op)."
            )

        assert D == self.sensory_dim + self.mcmc_dim, f"Expected dim {self.sensory_dim + self.mcmc_dim}, got {D}"

        if self.persistence_skip != 0.0 and self.sensory_dim < 3:
            raise ValueError("Nonzero persistence_skip requires sensory_dim >= 3")
        # Unconditional input/length preflight (NOT gated on persistence_skip):
        # direct eager apply must reject fractional/negative/overlong lengths
        # and nonfinite valid frames before the arithmetic below.  Under
        # tracing only the static shape/dtype checks run; the supported
        # wrappers perform the value preflight on the host before compiling.
        validate_input_and_lengths(x, lengths, context="NSMoRModel.__call__")

        # Sanitize invalid (padded) frames BEFORE any computation so a
        # finite-overflow or NaN padded suffix can never poison a recurrent
        # carry or the next prediction.  Selection (jnp.where) is used, not
        # arithmetic masking.  Nonfinite values in a VALID frame are validated
        # on the host by the supported callers (they cannot be checked under
        # jit); the model itself never hides real observations.
        _valid = (jnp.arange(T)[None, :] < lengths[:, None])[..., None]  # (B,T,1)
        x = jnp.where(_valid, x, jnp.zeros_like(x))

        sensory_x = x[:, :, :self.sensory_dim]
        mcmc_prior = x[:, :, self.sensory_dim:]

        # 1. Sensory Encoder (pass deterministic for noise gating).
        # The encoder injects noise only when ``deterministic`` is False, so
        # suppress it whenever the caller asks for eval (``deterministic``)
        # OR explicitly disables input noise (``not sensory_noise``).  Using
        # ``and`` here inverted the flag: ``sensory_noise=False`` would pass
        # ``deterministic=False`` and *inject* the noise it was meant to drop.
        e_sensory = self.sensory_encoder(
            sensory_x, deterministic=deterministic or not sensory_noise,
        )  # (B, T, H)
        assert e_sensory.shape == (B, T, H), f"e_sensory shape {e_sensory.shape} != ({B}, {T}, {H})"

        # 2. Precompute decay constants
        alpha_syn = compute_decay_factor(self.lif_tau_syn, dt)
        decay_w = compute_decay_factor(self.lif_tau_w, dt)
        decay_inhib = compute_decay_factor(max(self.lif_tau_syn, self.lif_inhib_tau_ms), dt)
        # Physical-time refractory parity with PyTorch LIFCell:
        #   abs_refract_steps = ceil(abs_ms / dt)   (sub-frame -> 1 step)
        #   rel_refract_steps = ceil(rel_ms / dt); k_rel = 1 / rel_steps
        # The historical Flax code used round() for absolute and dt/ms for
        # k_rel, which diverges for non-integral grids (dt=4 ms, abs=1 ms
        # must be ONE step, not zero).
        rel_refract_steps = (
            math.ceil(self.lif_rel_refract_ms / dt)
            if self.lif_rel_refract_ms > 0.0 else 0
        )
        k_rel = (1.0 / rel_refract_steps) if rel_refract_steps > 0 else 0.0
        delta_theta = 0.3 * self.lif_threshold
        v_thresh = self.lif_threshold
        # BLOCKER-2 fix: Match PyTorch clamp thresholds exactly.
        # PyTorch uses 3.0 * v_threshold and 5.0 * v_threshold
        # (model_nsmor_core.py:436,442).
        v_clamp_max = 3.0 * self.lif_threshold
        i_syn_clamp = 5.0 * self.lif_threshold
        abs_refract_steps = (
            math.ceil(self.lif_abs_refract_ms / dt)
            if self.lif_abs_refract_ms > 0.0 else 0.0
        )

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

            # Optional TBPTT gradient truncation.  Selective per-sample: only
            # genuinely advancing (active) rows detach at a boundary; a finished
            # row (mask_t == 0) keeps its differentiable carried path so other
            # samples' padding / shorter lengths cannot sever its gradient.
            if tbptt > 0:
                do_detach = (t_step > 0) & (t_step % tbptt == 0)
                act = mask_t[:, None]  # (B, 1) bool
                def _sel(s):
                    return lax.cond(
                        do_detach,
                        lambda z: jnp.where(act, lax.stop_gradient(z), z),
                        lambda z: z,
                        s,
                    )
                v = _sel(v)
                i_syn = _sel(i_syn)
                ref = _sel(ref)
                rel_ref = _sel(rel_ref)
                w = _sel(w)
                spk_hist = _sel(spk_hist)

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
        init_v = jnp.full((B, H), self.lif_v_rest, dtype=jnp.float32)
        init_isyn = jnp.zeros((B, H), dtype=jnp.float32)
        init_ref = jnp.zeros((B, H), dtype=jnp.float32)
        large_rel = float(10 * max(rel_refract_steps, 1))
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

        # Raw recurrent GRU trajectory BEFORE the output-only gain (r5 R5).
        # This is the true recurrent coordinate used by fixed-point/Jacobian
        # analysis; ``out_gru`` below carries the gain for the routed output
        # contract.  When the gain is disabled the two are identical.
        gru_hidden_raw = out_gru

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

        if self.persistence_skip != 0.0:
            v_lag = x[:, :, 2]
            mask_bt = jnp.arange(T)[None, :] < lengths[:, None]
            assert v_lag.shape == mask_bt.shape == (B, T)
            y_pred = y_pred + self.persistence_skip * jnp.where(mask_bt, v_lag, 0.0)

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
                "gru_hidden_raw": gru_hidden_raw,
            }
            assert internals["gru_hidden_raw"].shape == (B, T, self.hidden_dim), (
                f"gru_hidden_raw shape {internals['gru_hidden_raw'].shape} "
                f"!= ({B}, {T}, {self.hidden_dim})"
            )
            return y_pred, internals

        return y_pred


# ===============================================================
# PyTorch Checkpoint Loading & State Dict Compatibility
# ===============================================================

def _flax_dest_keys(model: NSMoRModel) -> set[str]:
    """The exact set of PyTorch state_dict keys the Flax destination represents.

    Includes both the hierarchical ``frontend.``/``backend.`` keys and the
    legacy flat aliases.  Gain parameters exist only under ``backend.*`` (they
    are nn.Parameters on BioDecisionCore, never flat legacy keys).
    """
    swiglu = getattr(model, "activation", "relu") == "swiglu"
    keys = {
        "frontend.sensory_encoder.net.0.weight",
        "frontend.sensory_encoder.net.0.bias",
        "frontend.sensory_encoder.net.1.weight",
        "frontend.sensory_encoder.net.1.bias",
        "backend.lif_cell.W_in.weight",
        "backend.lif_cell.W_in.bias",
        "backend.gru_unit.gru.weight_ih_l0",
        "backend.gru_unit.gru.weight_hh_l0",
        "backend.gru_unit.gru.bias_ih_l0",
        "backend.gru_unit.gru.bias_hh_l0",
        "backend.router.gate.weight",
        "backend.router.gate.bias",
        "backend.direction_head.net.0.weight",
        "backend.direction_head.net.0.bias",
    }
    if swiglu:
        keys |= {
            "frontend.sensory_encoder.gate_proj.weight",
            "frontend.sensory_encoder.gate_proj.bias",
            "backend.direction_head.gate_proj.weight",
            "backend.direction_head.gate_proj.bias",
            "backend.direction_head.value_proj.weight",
            "backend.direction_head.value_proj.bias",
            "backend.direction_head.out_proj.weight",
            "backend.direction_head.out_proj.bias",
        }
    else:
        keys |= {
            "backend.direction_head.net.3.weight",
            "backend.direction_head.net.3.bias",
        }
    if float(getattr(model, "lif_lateral_inhibition", 0.0) or 0.0) > 0.0:
        keys |= {
            "backend.lif_cell._W_inhib_raw",
            "backend.lif_cell._inhib_diag_mask",
        }
    if float(getattr(model, "gru_neuromod_gain", 0.0) or 0.0) > 0.0:
        keys |= {"backend._gain_scale", "backend._gain_bias"}
    flat = {
        k.replace("frontend.", "").replace("backend.", "")
        for k in keys
        if k not in ("backend._gain_scale", "backend._gain_bias")
    }
    return keys | flat


def load_from_torch_state_dict(
    model: NSMoRModel,
    state_dict: Dict[str, Any],
    *,
    source: Any = None,
    context: str = "load_from_torch_state_dict",
) -> Dict[str, Any]:
    """
    Map a PyTorch NSMoRCore state_dict to a Flax parameter PyTree.

    Supports both legacy flat keys and modern hierarchical
    ``frontend.``/``backend.`` keys.  This conversion FAILS CLOSED: any source
    key the Flax destination cannot represent (a stacked GRU layer 1, STP,
    gain/lateral-inhibition the destination does not have, a mismatched
    activation) is rejected BEFORE mapping, and the required destination keys
    and their shapes are verified.  A successful conversion therefore never
    silently drops parameters.

    A bare ``state_dict`` only proves parameter layout — it cannot prove
    non-parameter source settings (dendritic filtering, hard reset, timestep).
    Callers that hold the live model or its config MUST pass it as ``source``
    so those settings are validated too; the two-argument form is an explicitly
    bounded parameter-only API.

    Args:
        model: Destination Flax ``NSMoRModel`` (defines the supported topology).
        state_dict: Source PyTorch state_dict (flat or hierarchical keys).
        source: Optional live PyTorch ``NSMoRCore`` or config object used to
            validate non-parameter settings via :func:`assert_flax_supported`.
        context: Caller label used in error messages.

    Returns:
        ``{"params": {...}}`` Flax parameter PyTree.

    Raises:
        ValueError: On unsupported/mismatched source topology, or unexpected or
            missing keys, or a shape mismatch.
    """
    if source is not None:
        assert_flax_supported(source, context=context)
        assert_semantics_match(source, model, context=context)

    allowed = _flax_dest_keys(model)
    unexpected = sorted(set(state_dict) - allowed)
    if unexpected:
        raise ValueError(
            f"{context}: source state_dict contains keys the Flax destination "
            f"cannot represent (unsupported/mismatched topology): {unexpected}. "
            f"Refusing to silently drop parameters — use the complete Torch "
            f"backend or disable the listed mechanisms."
        )

    H = int(model.hidden_dim)
    D = int(model.sensory_dim)
    M = int(model.mcmc_dim)

    def _np(k: str) -> np.ndarray:
        t = state_dict[k]
        if hasattr(t, "cpu"):
            t = t.cpu().numpy()
        return np.asarray(t)

    # Resolve keys with fallback, verifying the expected shape of EVERY
    # present alias and requiring duplicate representations to agree.  A
    # wrong-width duplicate is rejected (strict Torch loading would also
    # reject it), and a conflicting hierarchical/legacy pair is rejected
    # rather than silently selecting one representation.
    def _get(
        key_primary: str, key_fallback: str, shape: Optional[tuple] = None,
    ) -> np.ndarray:
        present = [k for k in (key_primary, key_fallback) if k in state_dict]
        if not present:
            raise ValueError(
                f"{context}: required key '{key_primary}' (or legacy "
                f"'{key_fallback}') missing from source state_dict."
            )
        arrays = {k: _np(k) for k in present}
        for key, arr in arrays.items():
            # r7 R4: every required original alias must be real-floating and
            # finite BEFORE it is copied.  A NaN/Inf weight (or a complex/object
            # array) would otherwise be mapped into the parameter tree and
            # poison the compiled apply.
            if not np.issubdtype(arr.dtype, np.floating):
                raise ValueError(
                    f"{context}: required key '{key}' must be real-floating, "
                    f"got dtype {arr.dtype}."
                )
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"{context}: required key '{key}' contains nonfinite "
                    f"values; refusing to map a poisoned parameter."
                )
            if shape is not None and tuple(arr.shape) != shape:
                raise ValueError(
                    f"{context}: key '{key}' shape {tuple(arr.shape)} != "
                    f"expected {shape} for the Flax destination."
                )
        if len(present) == 2:
            a, b = arrays[key_primary], arrays[key_fallback]
            if not np.array_equal(a, b):
                raise ValueError(
                    f"{context}: conflicting duplicate aliases '{key_primary}' "
                    f"and '{key_fallback}' carry different values; refusing to "
                    f"silently choose one representation."
                )
        return arrays[present[0]]

    se_w = _get("frontend.sensory_encoder.net.0.weight", "sensory_encoder.net.0.weight", (H, D))
    se_b = _get("frontend.sensory_encoder.net.0.bias", "sensory_encoder.net.0.bias", (H,))
    se_ln_w = _get("frontend.sensory_encoder.net.1.weight", "sensory_encoder.net.1.weight", (H,))
    se_ln_b = _get("frontend.sensory_encoder.net.1.bias", "sensory_encoder.net.1.bias", (H,))

    # Gated (swiglu) activation adds a gate projection with new keys; the
    # default relu branch never reads or writes them, so old strict loads
    # are unaffected.
    swiglu = getattr(model, "activation", "relu") == "swiglu"
    if swiglu:
        se_gate_w = _get(
            "frontend.sensory_encoder.gate_proj.weight",
            "sensory_encoder.gate_proj.weight", (H, H),
        )
        se_gate_b = _get(
            "frontend.sensory_encoder.gate_proj.bias",
            "sensory_encoder.gate_proj.bias", (H,),
        )

    lif_w = _get("backend.lif_cell.W_in.weight", "lif_cell.W_in.weight", (H, H))
    lif_b = _get("backend.lif_cell.W_in.bias", "lif_cell.W_in.bias", (H,))

    gru_w_ih = _get("backend.gru_unit.gru.weight_ih_l0", "gru_unit.gru.weight_ih_l0", (3 * H, H))
    gru_w_hh = _get("backend.gru_unit.gru.weight_hh_l0", "gru_unit.gru.weight_hh_l0", (3 * H, H))
    gru_b_ih = _get("backend.gru_unit.gru.bias_ih_l0", "gru_unit.gru.bias_ih_l0", (3 * H,))
    gru_b_hh = _get("backend.gru_unit.gru.bias_hh_l0", "gru_unit.gru.bias_hh_l0", (3 * H,))

    r_w = _get("backend.router.gate.weight", "router.gate.weight", (2, H + M))
    r_b = _get("backend.router.gate.bias", "router.gate.bias", (2,))

    dh_ln_w = _get("backend.direction_head.net.0.weight", "direction_head.net.0.weight", (H,))
    dh_ln_b = _get("backend.direction_head.net.0.bias", "direction_head.net.0.bias", (H,))
    if swiglu:
        dh_gate_w = _get("backend.direction_head.gate_proj.weight", "direction_head.gate_proj.weight", (H, H))
        dh_gate_b = _get("backend.direction_head.gate_proj.bias", "direction_head.gate_proj.bias", (H,))
        dh_value_w = _get("backend.direction_head.value_proj.weight", "direction_head.value_proj.weight", (H, H))
        dh_value_b = _get("backend.direction_head.value_proj.bias", "direction_head.value_proj.bias", (H,))
        dh_out_w = _get("backend.direction_head.out_proj.weight", "direction_head.out_proj.weight", (1, H))
        dh_out_b = _get("backend.direction_head.out_proj.bias", "direction_head.out_proj.bias", (1,))
    else:
        dh_lin_w = _get("backend.direction_head.net.3.weight", "direction_head.net.3.weight", (1, H))
        dh_lin_b = _get("backend.direction_head.net.3.bias", "direction_head.net.3.bias", (1,))

    sensory_encoder_params: Dict[str, Any] = {
        "dense": {
            "kernel": jnp.array(se_w.T, dtype=jnp.float32),
            "bias": jnp.array(se_b, dtype=jnp.float32),
        },
        "ln": {
            "scale": jnp.array(se_ln_w, dtype=jnp.float32),
            "bias": jnp.array(se_ln_b, dtype=jnp.float32),
        },
    }
    if swiglu:
        sensory_encoder_params["gate"] = {
            "kernel": jnp.array(se_gate_w.T, dtype=jnp.float32),
            "bias": jnp.array(se_gate_b, dtype=jnp.float32),
        }

    direction_head_params: Dict[str, Any] = {
        "ln": {
            "scale": jnp.array(dh_ln_w, dtype=jnp.float32),
            "bias": jnp.array(dh_ln_b, dtype=jnp.float32),
        },
    }
    if swiglu:
        direction_head_params["gate"] = {
            "kernel": jnp.array(dh_gate_w.T, dtype=jnp.float32),
            "bias": jnp.array(dh_gate_b, dtype=jnp.float32),
        }
        direction_head_params["value"] = {
            "kernel": jnp.array(dh_value_w.T, dtype=jnp.float32),
            "bias": jnp.array(dh_value_b, dtype=jnp.float32),
        }
        direction_head_params["out"] = {
            "kernel": jnp.array(dh_out_w.T, dtype=jnp.float32),
            "bias": jnp.array(dh_out_b, dtype=jnp.float32),
        }
    else:
        direction_head_params["dense"] = {
            "kernel": jnp.array(dh_lin_w.T, dtype=jnp.float32),
            "bias": jnp.array(dh_lin_b, dtype=jnp.float32),
        }

    params: Dict[str, Any] = {
        "sensory_encoder": sensory_encoder_params,
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
        "direction_head": direction_head_params,
    }

    if model.lif_lateral_inhibition > 0.0:
        w_inhib = _get(
            "backend.lif_cell._W_inhib_raw", "lif_cell._W_inhib_raw", (H, H),
        )
        # The inhibition-mask buffer is REQUIRED and must be the canonical
        # fixed off-diagonal mask (1 - I) that the Flax path hard-codes.  A
        # missing, malformed or noncanonical mask is rejected rather than
        # silently replaced by a fabricated canonical buffer, which would
        # change the represented mechanism (Torch uses the supplied buffer in
        # its inhibitory-current equation).
        mask = _get(
            "backend.lif_cell._inhib_diag_mask",
            "lif_cell._inhib_diag_mask", (H, H),
        )
        canonical = 1.0 - np.eye(H, dtype=np.float32)
        if not np.array_equal(mask, canonical):
            raise ValueError(
                f"{context}: noncanonical '_inhib_diag_mask' buffer "
                f"(sum={float(np.sum(mask))}, expected {float(H * (H - 1))}); "
                f"the Flax backend supports only the fixed off-diagonal "
                f"inhibition mask. Refusing to substitute a fabricated mask."
            )
        params["lif_w_inhib"] = jnp.array(w_inhib, dtype=jnp.float32)

    if model.gru_neuromod_gain > 0.0:
        g_scale = _get("backend._gain_scale", "_gain_scale", ())
        g_bias = _get("backend._gain_bias", "_gain_bias", ())
        params["gain_scale"] = jnp.array(g_scale, dtype=jnp.float32)
        params["gain_bias"] = jnp.array(g_bias, dtype=jnp.float32)

    # r7 R4: independently validate the DESTINATION representation after every
    # leaf was narrowed to float32.  A finite float64 original (e.g. 1e300) that
    # overflows to +inf under the forced-float32 copy must fail closed rather
    # than return a poisoned parameter tree.
    def _check_dest_finite(tree: Any, path: str) -> None:
        if isinstance(tree, dict):
            for k, v in tree.items():
                _check_dest_finite(v, f"{path}.{k}")
        elif isinstance(tree, (list, tuple)):
            for i, v in enumerate(tree):
                _check_dest_finite(v, f"{path}[{i}]")
        else:
            arr = np.asarray(tree)
            if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                raise ValueError(
                    f"{context}: destination parameter {path!r} is not "
                    f"representable in the float32 Flax computation "
                    f"representation (a finite original overflowed on narrowing); "
                    f"refusing to return a poisoned parameter tree."
                )

    _check_dest_finite(params, "params")

    return {"params": params}


def to_torch_state_dict(flax_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Flax parameter PyTree back to a PyTorch state_dict.

    Converts parameter layouts only. Canonical PyTorch analysis additionally
    requires its checkpoint schema and valid dataset/prior provenance; JAX
    development training artifacts are outside that pipeline.
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
    if "gate" in p["sensory_encoder"]:
        sd["frontend.sensory_encoder.gate_proj.weight"] = _t(p["sensory_encoder"]["gate"]["kernel"]).T
        sd["frontend.sensory_encoder.gate_proj.bias"] = _t(p["sensory_encoder"]["gate"]["bias"])

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
    if "out" in p["direction_head"]:
        sd["backend.direction_head.gate_proj.weight"] = _t(p["direction_head"]["gate"]["kernel"]).T
        sd["backend.direction_head.gate_proj.bias"] = _t(p["direction_head"]["gate"]["bias"])
        sd["backend.direction_head.value_proj.weight"] = _t(p["direction_head"]["value"]["kernel"]).T
        sd["backend.direction_head.value_proj.bias"] = _t(p["direction_head"]["value"]["bias"])
        sd["backend.direction_head.out_proj.weight"] = _t(p["direction_head"]["out"]["kernel"]).T
        sd["backend.direction_head.out_proj.bias"] = _t(p["direction_head"]["out"]["bias"])
    else:
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
