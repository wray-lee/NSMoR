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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Valid activation names for the encoder / decoder projections.
# ``relu`` preserves the historical architecture and state_dict keys;
# ``swiglu`` selects a genuine gated activation
# ``SiLU(W_gate x) * (W_value x)`` (Shazeer 2020).
_VALID_ACTIVATIONS = ("relu", "swiglu")

# Integer (nonboolean) length dtypes accepted at every public boundary.
_INT_LENGTH_DTYPES = (
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
)


def _validate_original_lengths(
    lengths: torch.Tensor,
    batch_size: int,
    timesteps: int,
    *,
    context: str,
) -> torch.Tensor:
    """Validate original length tensors BEFORE any integer cast.

    Every public component boundary shares this preflight so a fractional,
    boolean, negative or overlong length is rejected against the ORIGINAL
    dtype instead of being silently truncated by ``.to(torch.int64)`` (which
    reinterprets ``1.75 -> 1``, ``-0.5 -> 0``, ``True -> 1``).  Returns the
    validated lengths cast to ``int64`` for downstream use.

    Args:
        lengths: Original ``(B,)`` length tensor.
        batch_size: Expected ``B``.
        timesteps: Padded ``T``; lengths must satisfy ``0 <= lengths <= T``.
        context: Caller label used in error messages.

    Raises:
        ValueError: On a shape, dtype or range violation.
    """
    if lengths.shape != (batch_size,):
        raise ValueError(
            f"{context}: lengths shape {tuple(lengths.shape)} != "
            f"(B={batch_size},)"
        )
    if lengths.dtype not in _INT_LENGTH_DTYPES:
        raise ValueError(
            f"{context}: lengths must have an integer, nonboolean dtype, "
            f"got {lengths.dtype}"
        )
    lengths_i = lengths.to(torch.int64)
    if bool((lengths_i < 0).any()) or bool((lengths_i > timesteps).any()):
        raise ValueError(
            f"{context}: lengths must satisfy 0 <= lengths <= T={timesteps}"
        )
    return lengths_i


def _validate_original_carry(
    supplied: torch.Tensor,
    expected_shape: Tuple[int, ...],
    *,
    key: str,
    device: torch.device,
    dtype: torch.dtype,
    domain: Optional[str] = None,
) -> torch.Tensor:
    """Validate an ORIGINAL Torch carry field before cast/device/math.

    Inspects the supplied tensor's ORIGINAL real-floating dtype, exact shape
    and finiteness BEFORE any numeric conversion, so a boolean/integer/complex
    carry (whose imaginary part ``.float()`` would silently discard) or a
    nonfinite carry (e.g. ``lif_v=NaN``) is rejected rather than poisoning the
    recurrence.  Returns the validated field converted to ``dtype``/``device``.

    ``dtype`` MUST be the ACTUAL computation dtype (float32 for the forced-FP32
    LIF/GRU paths), not a default-dtype canonical allocation: a finite float64
    ``1e300`` is representable in float64 but overflows to ``+inf`` when the
    recurrence narrows it, so representability is checked against the real
    computation representation.

    ``domain`` optionally enforces a physically-constrained range (r6 R3):
    ``"nonneg"`` for time-since-spike counters / physical adaptation and
    ``"fraction"`` for STP resource/facilitation and EMA spike-history fields
    in ``[0, 1]``.  Membrane/GRU coordinates are unconstrained.

    Raises:
        ValueError: On a non-tensor, non-real-floating dtype, wrong shape, a
            nonfinite value, non-representability in ``dtype``, or a domain
            violation.
    """
    if not isinstance(supplied, torch.Tensor):
        raise ValueError(
            f"carry field {key!r} must be a torch.Tensor, got "
            f"{type(supplied).__name__}"
        )
    if not supplied.is_floating_point():
        raise ValueError(
            f"carry field {key!r} must have a real floating dtype, got "
            f"{supplied.dtype}"
        )
    if tuple(supplied.shape) != tuple(expected_shape):
        raise ValueError(
            f"carry field {key!r} shape {tuple(supplied.shape)} != "
            f"{tuple(expected_shape)}"
        )
    if not torch.isfinite(supplied).all():
        raise ValueError(f"carry field {key!r} contains nonfinite values")
    if domain is not None:
        _check_carry_domain(supplied, key=key, domain=domain)
    converted = supplied.to(device=device, dtype=dtype)
    # Representability after conversion (r5 R1 / r6 R1): a finite original
    # float64 value such as 1e300 becomes float32 +inf when narrowed.  The
    # original finiteness check above is necessary but NOT sufficient; the
    # ACTUAL computation representation must also be finite, otherwise an
    # ostensibly-safe no-op row exports an infinite carry.
    if not torch.isfinite(converted).all():
        raise ValueError(
            f"carry field {key!r} is not representable in the computation "
            f"dtype {dtype}: finite {supplied.dtype} values overflow to "
            f"nonfinite after conversion."
        )
    return converted


def _check_carry_domain(
    value: torch.Tensor, *, key: str, domain: str,
) -> None:
    """Reject a supplied physical carry outside its justified domain (r6 R3)."""
    if domain == "nonneg":
        if bool((value < 0).any()):
            raise ValueError(
                f"carry field {key!r} must be non-negative (a physical "
                f"time-since-spike counter / adaptation value); got a "
                f"negative value."
            )
    elif domain == "fraction":
        if bool((value < 0).any()) or bool((value > 1.0).any()):
            raise ValueError(
                f"carry field {key!r} must lie in [0, 1] (a fraction / EMA "
                f"history value); got a value outside that range."
            )
    else:  # pragma: no cover - programming error
        raise ValueError(f"unknown carry domain {domain!r} for {key!r}")


def _validate_np_leaf(
    raw: Any, *, key: str, expected_shape: Optional[Tuple[int, ...]] = None,
    dtype: Any = None,
) -> "np.ndarray":
    """Validate an ORIGINAL numpy/torch parameter leaf before numeric narrowing.

    Shared by the raw-JAX parameter copier and the Flax state-dict converter
    (root 2).  Checks, on the ORIGINAL value BEFORE any float32 conversion:
    real-floating dtype, exact shape (when given), finiteness; then independently
    requires the destination representation (``dtype``, float32) to be finite, so
    a finite float64 ``1e300`` that overflows on narrowing fails closed instead
    of producing a poisoned leaf.
    """
    import numpy as _np

    arr = _np.asarray(raw)
    if not _np.issubdtype(arr.dtype, _np.floating):
        raise ValueError(
            f"parameter {key!r} must be real-floating, got dtype {arr.dtype}."
        )
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(
            f"parameter {key!r} shape {tuple(arr.shape)} != expected "
            f"{tuple(expected_shape)}."
        )
    if not _np.isfinite(arr).all():
        raise ValueError(
            f"parameter {key!r} contains nonfinite values; refusing to copy a "
            f"poisoned parameter."
        )
    if dtype is not None:
        narrowed = arr.astype(dtype)
        if not _np.isfinite(narrowed).all():
            raise ValueError(
                f"parameter {key!r} is not representable in the destination "
                f"{_np.dtype(dtype)} representation (a finite original "
                f"overflowed on narrowing); refusing to copy."
            )
        return narrowed
    return arr


def _require_real_floating_observation(
    tensor: Any, *, name: str, context: str,
) -> None:
    """Reject a non-real-floating public observation BEFORE any conversion.

    A public observation boundary must inspect the ORIGINAL dtype: an integer,
    boolean or complex measurement (whose imaginary part a later ``.float()``
    would silently discard) is a scientific/data error, not a value to coerce.
    Shared by the Torch core, the direct backend and the raw-JAX boundary so
    every public seam refuses the same invalid inputs.

    Raises:
        ValueError: When ``tensor`` is not a real-floating torch tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"{context}: {name} must be a torch.Tensor, got "
            f"{type(tensor).__name__}"
        )
    if not tensor.is_floating_point():
        raise ValueError(
            f"{context}: {name} must have a real floating dtype, got "
            f"{tensor.dtype}. Integer/boolean/complex observations are "
            "refused before conversion."
        )


def _validate_refinement_options(
    *, mode: Any, max_steps: Any, eps: Any, update_scale: Any, context: str,
) -> Tuple[str, int, float, float]:
    """Validate ORIGINAL refinement options independent of module creation.

    The refinement-option contract must hold at every public trust boundary,
    including ``refinement_mode == "off"`` where no module is constructed.
    This single validator is shared by :class:`AdaptiveLatentRefinement` and
    :class:`BioDecisionCore` (and mirrors ``ModelConfig.__post_init__``) so a
    bool/NaN/negative/illegal setting is refused even when refinement is
    disabled and the module is absent.  The ORIGINAL values are validated
    before any cast.

    Returns the validated ``(mode, int(max_steps), float(eps),
    float(update_scale))``.

    Raises:
        ValueError: On any malformed option.
    """
    if mode not in ("off", "fixed", "adaptive"):
        raise ValueError(
            f"{context}: refinement_mode must be 'off'/'fixed'/'adaptive', "
            f"got {mode!r}"
        )
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError(
            f"{context}: refinement_max_steps must be an int >= 1, got "
            f"{max_steps!r}"
        )
    if (
        isinstance(eps, bool)
        or not isinstance(eps, (int, float))
        or not math.isfinite(eps)
        or not (0.0 < float(eps) < 1.0)
    ):
        raise ValueError(
            f"{context}: refinement_eps must be in (0, 1), got {eps!r}"
        )
    if (
        isinstance(update_scale, bool)
        or not isinstance(update_scale, (int, float))
        or not math.isfinite(update_scale)
        or update_scale <= 0.0
    ):
        raise ValueError(
            f"{context}: refinement_update_scale must be finite > 0, got "
            f"{update_scale!r}"
        )
    return mode, int(max_steps), float(eps), float(update_scale)


def assert_mechanism_tree_consistent(source: Any, *, context: str) -> None:
    """Fail closed if an ENABLED mechanism lacks its required parameter leaves.

    H4 / findings A5, B2: a live mechanism flag (``gru_neuromod_gain``,
    ``lif_cell.stp_enabled``, ``lif_cell.lateral_inhibition``) may be set
    inconsistently AFTER construction (e.g. a direct attribute mutation),
    leaving the branch active while its parameters were never created.  Every
    such consumer would then raise a bare ``AttributeError`` deep inside the
    math (or, in the raw-JAX converter, mid-dereference).  This SINGLE shared
    preflight inspects the flags against the actual parameter tree and raises a
    descriptive ``ValueError`` BEFORE any leaf is read.  It is used by the
    Torch ``BioDecisionCore.forward`` and by the raw-JAX live converter.

    It is NOT a dynamic setter framework: a constructor-configured mechanism
    (which always creates its leaves) and the disabled default are unaffected.

    Args:
        source: A ``BioDecisionCore``/``NSMoRCore`` (or any object exposing
            ``backend``/``lif_cell``).
        context: Caller label used in the error message.

    Raises:
        ValueError: When an enabled mechanism is missing required parameters.
    """
    backend = getattr(source, "backend", source)
    lif = getattr(backend, "lif_cell", None)
    if lif is None:
        lif = getattr(source, "lif_cell", None)
    missing: List[str] = []
    gain = float(getattr(backend, "gru_neuromod_gain", 0.0) or 0.0)
    if gain > 0.0:
        for name in ("_gain_scale", "_gain_bias"):
            if not hasattr(backend, name):
                missing.append(f"backend.{name}")
    if lif is not None:
        if bool(getattr(lif, "stp_enabled", False)):
            for name in ("U_stp_raw", "_decay_fac", "_decay_rec"):
                if not hasattr(lif, name):
                    missing.append(f"lif_cell.{name}")
        if float(getattr(lif, "lateral_inhibition", 0.0) or 0.0) > 0.0:
            if not hasattr(lif, "_W_inhib_raw"):
                missing.append("lif_cell._W_inhib_raw")
    if missing:
        raise ValueError(
            f"{context}: inconsistent mechanism configuration — enabled "
            f"mechanism(s) are missing required parameters ({', '.join(missing)}). "
            "This indicates a live flag was mutated after construction; "
            "refusing to run a mechanism whose parameters were never created. "
            "Construct the model with the mechanism enabled instead."
        )


def refinement_module_present(source: Any) -> bool:
    """Return True if the EXECUTED backend actually carries a refinement module.

    R1 / findings A1, B1, B11, B16: the Torch core executes refinement whenever
    ``backend.refinement is not None`` (model_nsmor_core.py:2865), regardless of
    the ``refinement_mode`` string.  A model whose mode was mutated to ``"off"``
    AFTER construction still carries the module, so keying a refusal on the
    stale string alone lets the fused JAX kernel silently drop the block.  This
    shared predicate inspects the executed child, not the alias.

    Args:
        source: A ``NSMoRCore``, a ``BioDecisionCore``, or a config object.

    Returns:
        True when a refinement module (or a non-``"off"``/``None`` mode alias)
        is present on the source or its ``backend``.
    """
    backend = getattr(source, "backend", source)
    if getattr(backend, "refinement", None) is not None:
        return True
    for holder in (source, backend):
        mode = getattr(holder, "refinement_mode", None)
        if mode not in (None, "off"):
            return True
    return False


def _refinement_empty_zero(
    h_fused: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    """Return a graph-connected DIFFERENTIABLE zero for an all-empty batch.

    H2 / findings A2, B1: the previous ``h_fused.sum() * 0.0`` reduced the RAW
    input, so NaN padding gave ``NaN * 0 = NaN`` and a finite-overflow padding
    (e.g. 3e38) overflowed the sum to ``inf * 0 = NaN`` — a nonfinite ponder
    cost for a batch that executed NO work.  Select the SAFE (valid) operands
    FIRST, then reduce and multiply by zero: every element is a real zero, so
    the reduction is exactly ``0.0`` while the result stays connected to the
    graph (gradient 0, never leaf-less).  Shared by the fixed and adaptive
    empty branches so both sites cannot diverge.
    """
    safe = torch.where(valid.unsqueeze(-1), h_fused, torch.zeros_like(h_fused))
    return safe.sum() * 0.0


def _canonical_encoder_topology(activation: str) -> Tuple[Tuple[type, ...], bool]:
    """Return the exact ordered ``net`` module types for a canonical encoder.

    Returns ``(module_types, has_gate_proj)``.  ``activation`` must be a
    supported name.
    """
    if activation == "swiglu":
        return (nn.Linear, nn.LayerNorm), True
    return (nn.Linear, nn.LayerNorm, nn.ReLU), False


def _canonical_head_topology(activation: str) -> Tuple[Tuple[type, ...], bool]:
    """Return the exact ordered ``net`` module types for a canonical head."""
    if activation == "swiglu":
        return (nn.LayerNorm, nn.Dropout), True
    return (nn.LayerNorm, nn.ReLU, nn.Dropout, nn.Linear), False


def certify_executed_topology(source: Any, *, context: str) -> None:
    """Certify the ACTUAL executed encoder/readout topology of a live source.

    The known-live converters (Flax ``load_from_torch_state_dict`` and raw
    ``NSMoRCoreJAX``) hard-code an exact ordered computation:

    * ``sensory_encoder`` (relu): ``Linear -> LayerNorm(eps=1e-5) -> ReLU``
      (swiglu): ``Linear -> LayerNorm(eps=1e-5)`` plus ``gate_proj`` and
      ``SiLU(gate_proj(h)) * h``
    * ``direction_head`` (relu): ``LayerNorm(eps=1e-5) -> ReLU -> Dropout(p)
      -> Linear`` (swiglu): ``LayerNorm(eps=1e-5) -> Dropout(p)`` plus
      ``gate_proj``/``value_proj``/``out_proj`` and
      ``SiLU(gate) * value``.

    A source whose executed modules differ (an Identity replacing ReLU, an
    appended/missing operator, a stale child activation, a noncanonical or
    nonfinite LayerNorm epsilon, a live dropout rate the destination cannot
    reproduce) would be silently mapped onto a DIFFERENT computation.  This
    certificate inspects the actual modules/slots/semantics (r7 R1) and rejects
    any divergence.  ``source`` with no executed ``frontend``/``backend`` (a
    Flax model/config) is a no-op.

    Raises:
        ValueError: On any divergence from the canonical executed topology.
    """
    frontend = getattr(source, "frontend", None)
    backend = getattr(source, "backend", None)
    if frontend is None and backend is None:
        return

    se = getattr(frontend, "sensory_encoder", None)
    if se is None:
        se = getattr(source, "sensory_encoder", None)
    dh = getattr(backend, "direction_head", None)
    if dh is None:
        dh = getattr(source, "direction_head", None)

    problems: List[str] = []

    def _check(
        module: Any, name: str, expected: Tuple[type, ...],
        need_projs: Tuple[str, ...], hidden_dim: Optional[int] = None,
    ) -> None:
        if module is None:
            return
        activation = getattr(module, "activation", None)
        if activation not in _VALID_ACTIVATIONS:
            problems.append(f"{name}.activation={activation!r}")
            return
        net = getattr(module, "net", None)
        if not isinstance(net, nn.Sequential):
            problems.append(f"{name}.net is not a Sequential")
            return
        got = [type(sub) for sub in net]
        if got != list(expected):
            problems.append(
                f"{name}.net modules {[t.__name__ for t in got]} != "
                f"{[t.__name__ for t in expected]}"
            )
        # Finite supported LayerNorm epsilon AND true normalized axes/affine
        # semantics at every normalization slot (root 1).  A live LayerNorm
        # with a non-canonical normalized_shape (e.g. ``(2, 4)``) or disabled
        # affine normalizes different axes than the destination's last-axis,
        # affine LayerNorm, silently changing the executed computation.
        for idx, sub in enumerate(net):
            if isinstance(sub, nn.LayerNorm):
                if not math.isfinite(float(sub.eps)) or abs(float(sub.eps) - 1e-5) > 1e-12:
                    problems.append(f"{name}.net[{idx}].eps={sub.eps}")
                if hidden_dim is not None and tuple(sub.normalized_shape) != (int(hidden_dim),):
                    problems.append(
                        f"{name}.net[{idx}].normalized_shape={tuple(sub.normalized_shape)}"
                    )
                if not bool(getattr(sub, "elementwise_affine", True)):
                    problems.append(f"{name}.net[{idx}].elementwise_affine=False")
        # Live dropout rate the destination reproduces from ``dropout_rate``.
        declared = getattr(module, "dropout_rate", None)
        for idx, sub in enumerate(net):
            if isinstance(sub, nn.Dropout):
                if declared is None:
                    problems.append(f"{name}.net[{idx}] dropout without declared rate")
                elif not math.isfinite(float(sub.p)) or abs(
                    float(sub.p) - float(declared)
                ) > 1e-12:
                    problems.append(f"{name}.net[{idx}].p={sub.p} != declared {declared}")
        # Required projection children of a gated (swiglu) branch.
        for proj in need_projs:
            child = getattr(module, proj, None)
            if not isinstance(child, nn.Linear):
                problems.append(f"{name}.{proj} is not a Linear")

    h_dim = getattr(source, "hidden_dim", None)
    se_types, se_gate = _canonical_encoder_topology(
        getattr(se, "activation", "relu") if se is not None else "relu"
    )
    _check(se, "sensory_encoder", se_types, ("gate_proj",) if se_gate else (), h_dim)
    dh_types, dh_gate = _canonical_head_topology(
        getattr(dh, "activation", "relu") if dh is not None else "relu"
    )
    dh_projs = ("gate_proj", "value_proj", "out_proj") if dh_gate else ()
    _check(dh, "direction_head", dh_types, dh_projs, h_dim)

    if problems:
        raise ValueError(
            f"{context}: live source has executed normalization/activation/"
            f"dropout topology the destination cannot reproduce "
            f"({'; '.join(problems)}). Refusing to map parameters across "
            f"divergent runtime computation."
        )


# ===============================================================
# 1.  Sensory Encoder
# ===============================================================

class SensoryEncoder(nn.Module):
    """
    Map raw 4-D sensory features to a hidden representation.

    ``activation="relu"`` (default)::

        Linear(4, hidden_dim) -> LayerNorm -> ReLU

    ``activation="swiglu"``::

        h = LayerNorm(Linear(4, hidden_dim)(x))
        out = SiLU(gate_proj(h)) * h        # genuine gated activation

    where ``gate_proj`` is a second ``Linear(hidden_dim, hidden_dim)``.
    SwiGLU is ``SiLU(W_gate x) * (W_value x)``; a single ``SiLU(W x)`` is
    NOT gated and is not what this branch implements.

    Optionally injects Gaussian noise during training to model intrinsic
    neural variability and stochastic resonance (Gap D).
    Ref: Douglass et al. 1993, Nature 365:721-723.
    Ref: Shazeer 2020, "GLU Variants Improve Transformer" (gated activation).

    Input:  ``(B, T, 4)``
    Output: ``(B, T, H)``
    """

    def __init__(
        self,
        sensory_dim: int = 4,
        hidden_dim: int = 64,
        noise_std: float = 0.0,
        activation: str = "relu",
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
            activation: ``"relu"`` (default, historical) or ``"swiglu"``
                (opt-in gated activation).

        Raises:
            ValueError: If *activation* is not a supported name.
        """
        super().__init__()
        if activation not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_VALID_ACTIVATIONS}, "
                f"got {activation!r}"
            )
        self.activation = activation
        self.hidden_dim = hidden_dim
        if activation == "swiglu":
            # LayerNorm is applied to the value projection, matching the
            # relu branch's post-Linear normalisation point.
            self.net = nn.Sequential(
                nn.Linear(sensory_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
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
        B, T, _ = sensory.shape
        H = self.hidden_dim
        h = self.net(sensory)
        if self.activation == "swiglu":
            h = F.silu(self.gate_proj(h)) * h
        assert h.shape == (B, T, H), (
            f"SensoryEncoder output {tuple(h.shape)} != (B={B}, T={T}, H={H})"
        )
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
        abs_refract_ms: float = 0.0,
        rel_refract_ms: float = 20.0,
        tau_syn: float = 0.0,
        v_rest: float = 0.0,
        v_reset: Optional[float] = None,
        tau_w: float = 0.0,
        b_adapt: float = 0.0,
        tau_fac: float = 0.0,
        tau_rec: float = 0.0,
        U_stp_init: float = 0.5,
        lateral_inhibition: float = 0.0,
        inhib_tau_ms: float = 50.0,
        dendritic_tau: float = 0.0,
        dt_ms: float = 10.0,
    ) -> None:
        """
        Args:
            hidden_dim: Dimensionality of the membrane state.
            alpha: Leak factor in (0, 1).  Higher -> slower decay.
            v_threshold: Baseline spike threshold.
            beta: Input scaling factor.
            abs_refract_ms: Duration of the absolute refractory period
                in PHYSICAL time (ms) — Na+ channel inactivation.
                Converted internally to whole steps via dt_ms (rounded
                up; sub-frame durations round to 0 = disabled).
                0 disables.
                Ref: Hodgkin & Huxley 1952.
            rel_refract_ms: Characteristic decay length of the relative
                refractory threshold elevation in PHYSICAL time (ms).
                Converted internally to steps via
                ``rel_refract_steps = ceil(rel_refract_ms / dt_ms)`` so
                a change of sampling rate can never silently rescale
                the biophysics (Round-3, Reviewer B MAJOR-1).
                0 disables.
                Ref: Bean 2007.
            tau_syn: Synaptic time constant in PHYSICAL time (ms).
                Converted internally to the per-step coefficient
                ``alpha_syn = exp(-dt_ms / tau_syn)``.  0 bypasses
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
                time constant ``max(tau_syn, inhib_tau_ms)`` (Round-3,
                Reviewer B MINOR-6: the 50 ms fallback window is now an
                explicit parameter instead of a magic number).
                0 disables (backward compatible).
                Ref: Ritzmann & Camhi 1978, J. Comp. Physiol.
            inhib_tau_ms: Spike-history EMA window for lateral
                inhibition in PHYSICAL time (ms).  Used when tau_syn is
                shorter than this; the longer of the two governs.
                Default 50 ms — within the 20-100 ms window of
                feedforward inhibition in cricket cercal pathways.
            dendritic_tau: Time constant for the dendritic low-pass
                filter applied to the visual input before somatic
                integration.  Models the separate dendritic
                compartmentalization seen in LGI: wind signals arrive
                at cercal dendrites (fast, no filtering) while visual
                signals traverse optic lobe dendrites (slower, with
                temporal smoothing).  When > 0, the SINGLE visual
                channel (sensory layout index 0) passes through an IIR
                filter with this time constant before reaching the
                soma, while the remaining channels (wind/kinematic)
                bypass it.  0 disables (backward compatible).
                Ref: London & Hausser 2005, Annu. Rev. Neurosci.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.v_threshold = v_threshold
        self.beta = beta

        # Reviewer Round-1 BLOCKER-1: physical time base.
        # All tau_* parameters are declared in milliseconds; the
        # per-step decay coefficients are exp(-dt_ms / tau_ms).  The
        # sampling interval MUST be provided explicitly so that a
        # change of acquisition rate can never silently rescale the
        # biophysics.
        assert dt_ms > 0.0, (
            f"dt_ms (sampling interval) must be > 0, got {dt_ms}"
        )
        self.dt_ms = float(dt_ms)

        # Round-3 fix (Reviewer B MAJOR-1): refractory periods are also
        # declared in PHYSICAL time (ms) and converted to whole steps
        # here — closing the frame-unit loophole that BLOCKER-1 closed
        # for the synaptic filters.  Absolute refractory rounds UP (a
        # 2 ms physiological duration at dt=10 ms rounds to 1 step;
        # sub-frame values round to 0 = disabled).  Relative refractory
        # uses ceil so the declared decay length is never silently
        # shortened by sampling.
        assert abs_refract_ms >= 0.0, (
            f"abs_refract_ms must be >= 0, got {abs_refract_ms}"
        )
        assert rel_refract_ms >= 0.0, (
            f"rel_refract_ms must be >= 0, got {rel_refract_ms}"
        )
        self.abs_refract_steps = int(math.ceil(
            float(abs_refract_ms) / self.dt_ms
        )) if abs_refract_ms > 0.0 else 0
        self.rel_refract_steps = int(math.ceil(
            float(rel_refract_ms) / self.dt_ms
        )) if rel_refract_ms > 0.0 else 0
        # Physical-time records of the declarations (for checkpoints /
        # introspection; the step counts above drive the dynamics).
        self.abs_refract_ms = float(abs_refract_ms)
        self.rel_refract_ms = float(rel_refract_ms)
        self.tau_syn = tau_syn

        def _decay_per_step(tau_ms: float, name: str) -> float:
            """Convert a physical time constant (ms) to a per-step
            decay coefficient exp(-dt_ms/tau_ms).  Returns 0.0 for
            tau_ms <= 0 (mechanism disabled)."""
            if tau_ms <= 0.0:
                return 0.0
            c = math.exp(-self.dt_ms / tau_ms)
            assert 0.0 < c < 1.0, (
                f"{name}: decay coefficient {c} out of (0, 1) for "
                f"tau={tau_ms} ms at dt={self.dt_ms} ms"
            )
            return c

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
        # CF7 fix: Membrane potential upper bound to prevent runaway accumulation.
        # 3x threshold is generous enough to allow natural spike overshoot
        # dynamics but prevents exponential blow-up when alpha*V + I_syn
        # compounds across timesteps.  Without this, V can grow to 100+,
        # saturating the surrogate gradient and causing NaN/Inf in backward.
        self._v_clamp_max = 3.0 * v_threshold
        # CF7 fix: Synaptic current clamp to prevent I_syn accumulation runaway.
        # With alpha_syn near 1.0, I_syn acts as an exponential moving average
        # of raw_input.  If raw_input is consistently positive (from the
        # linear projection W_in which has no output normalization), I_syn can
        # grow without bound.  Clamping at 5x threshold prevents this.
        self._i_syn_clamp = 5.0 * v_threshold
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
        if self.rel_refract_steps > 0:
            self._k_rel = 1.0 / self.rel_refract_steps
        else:
            self._k_rel = 0.0

        # Synaptic filter coefficient: alpha_syn = exp(-dt_ms/tau_syn)
        # persistent=False: these are derived constants, not learnable
        # parameters.  Using persistent=False keeps them OUT of
        # state_dict(), so old checkpoints without these keys load
        # cleanly with load_state_dict(strict=True).
        _alpha_syn_val = _decay_per_step(tau_syn, "tau_syn")
        self.register_buffer(
            '_alpha_syn', torch.tensor(_alpha_syn_val), persistent=False,
        )

        # Adaptation decay coefficient: alpha_w = exp(-dt_ms/tau_w)
        _decay_w_val = _decay_per_step(tau_w, "tau_w")
        self.register_buffer(
            '_decay_w', torch.tensor(_decay_w_val), persistent=False,
        )

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

            # Pre-compute decay coefficients (scalars, not parameters;
            # physical ms -> per-step conversion via _decay_per_step)
            self._decay_fac = _decay_per_step(tau_fac, "tau_fac")
            self._decay_rec = _decay_per_step(tau_rec, "tau_rec")

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
            # Reuses tau_syn if longer, else the explicit inhib_tau_ms
            # fallback window (Round-3, Reviewer B MINOR-6: no magic
            # number).
            self._inhib_tau_ms = max(tau_syn, inhib_tau_ms)
            self._decay_inhib = _decay_per_step(
                self._inhib_tau_ms, "_inhib_tau_ms",
            )

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
        # The single visual channel (sensory index 0) passes through the IIR
        # filter; wind/kinematic channels reach the soma directly.
        self.dendritic_tau = dendritic_tau
        self._dendritic_enabled = dendritic_tau > 0.0
        if self._dendritic_enabled:
            self._alpha_dend = _decay_per_step(dendritic_tau, "dendritic_tau")

        # Input projection
        self.W_in = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # CF8 fix: Small positive bias for baseline excitability.
        # Ensures neurons have a slight tendency to accumulate toward
        # threshold even with zero-mean input, preventing initial silence.
        # 0.01 is small enough to not dominate learned weights (~0.07).
        nn.init.constant_(self.W_in.bias, 0.01)

    def forward(
        self,
        input_t: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        *,
        _checked: bool = False,
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
            _checked: PRIVATE (r8 R5).  When True, the per-step public
                trust-boundary validation is skipped.  Set ONLY by
                ``BioDecisionCore._run_lif_path``, which validates the supplied
                carry ONCE at its public entry and then feeds internally
                produced state through the T-loop.  Default False keeps the
                full validation for every direct/public caller.

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

        # ── Trust boundary: validate the COMPLETE original direct-LIF input
        # and state BEFORE any arithmetic (r7 R2).  This is a documented PUBLIC
        # step API; the top-level dictionary guards do not protect direct
        # callers.  Every supplied field is checked for ORIGINAL real-floating
        # dtype, exact shape, finiteness and its justified physical domain
        # (membrane/synaptic coordinates are intentionally unconstrained;
        # counters/adaptation non-negative; STP fractions and the EMA
        # spike-history cache in [0, 1]).  A negative history would otherwise
        # flip lateral inhibition to excitation, and a nonfinite membrane/input
        # would poison spikes and the exported state.  Conversion is a no-op
        # (the direct cell computes in the supplied dtype), so the original
        # representation is also the computation representation.
        #
        # r8 R5: ``_checked`` skips this host-synchronizing validation for the
        # INTERNAL T-loop only.  ``BioDecisionCore._run_lif_path`` validates the
        # supplied carry once at its public entry and then feeds state it
        # produced itself, so re-validating every step (about 20 device syncs
        # per step) is redundant cost, not extra safety.  Numerics are
        # bitwise unchanged; a direct/public caller keeps the full validation.
        if not _checked:
            def _bound(key: str, val: Any, domain: Optional[str]) -> torch.Tensor:
                return _validate_original_carry(
                    val, (B, H), key=key, device=device,
                    dtype=getattr(val, "dtype", None), domain=domain,
                )

            if not (input_t.is_floating_point() and torch.isfinite(input_t).all()):
                raise ValueError(
                    "LIFCell input_t must be real-floating and finite; got "
                    f"dtype={input_t.dtype}."
                )
            v = _bound("LIFCell.v", v, None)
            i_syn = _bound("LIFCell.i_syn", i_syn, None)
            refract_counter = _bound("LIFCell.refract_counter", refract_counter, "nonneg")
            v_thresh_eff = _bound("LIFCell.v_thresh_eff", v_thresh_eff, None)
            w_adapt = _bound("LIFCell.w_adapt", w_adapt, "nonneg")
            rel_refract_counter = _bound(
                "LIFCell.rel_refract_counter", rel_refract_counter, "nonneg",
            )
            if self.stp_enabled:
                x_resource = _bound("LIFCell.x_resource", x_resource, "fraction")
                u_facil = _bound("LIFCell.u_facil", u_facil, "fraction")
            # The lateral-inhibition spike-history cache lives OUTSIDE
            # ``state``.  A GENUINELY ABSENT cache (None) is lazily initialized
            # below, but a MALFORMED supplied cache (a Tensor of the wrong
            # shape, or a same-shape nonfinite / out-of-domain history) is a
            # poisoned observation and must be REJECTED, not silently reset
            # (root 4).  "Absent" and "malformed" are distinct.
            if self.lateral_inhibition > 0.0:
                _supplied_hist = getattr(self, "_spike_history", None)
                if _supplied_hist is not None:
                    self._spike_history = _validate_original_carry(
                        _supplied_hist, (B, H), key="LIFCell._spike_history",
                        device=device, dtype=getattr(_supplied_hist, "dtype", None),
                        domain="fraction",
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
            # r6 R3: clamp BEFORE the nonfinite check so a bounded supplied
            # STP state cannot introduce nonfinite arithmetic; a supplied
            # nonfinite/out-of-domain STP field is refused earlier at the
            # public trust boundary.
            u_pre = u_pre.clamp(min=1e-6, max=1.0)           # (B, H)
            x_pre = x_pre.clamp(min=1e-6, max=1.0)           # (B, H)
            if not (torch.isfinite(u_pre).all() and torch.isfinite(x_pre).all()):
                raise ValueError(
                    "STP state produced nonfinite facilitation/resource after "
                    "decay; refusing to propagate a poisoned short-term "
                    "plasticity state."
                )

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
        # CF7 fix: Clamp synaptic current to prevent accumulation runaway.
        i_syn_new = i_syn_new.clamp(-self._i_syn_clamp, self._i_syn_clamp)

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
        # CF7 fix: Clamp membrane potential to prevent runaway accumulation.
        # Upper bound = 3x threshold (allows natural spike overshoot but
        # prevents exponential blow-up).  Lower bound = -threshold
        # (allows hyperpolarization but prevents extreme negative values).
        # Without this, V can grow unbounded when alpha*V + I_syn compounds,
        # causing surrogate gradient saturation and NaN/Inf in backward pass.
        v_new = v_new.clamp(-self.v_threshold, self._v_clamp_max)

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
                # root 5: match the actual compute dtype (v_new), not the
                # ambient global default.
                spike_hist = torch.zeros(
                    B, H, device=device, dtype=v_new.dtype,
                )
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
        # CF8 fix: Sharpened sigmoid (sharpness=4.0) raises peak gradient
        # from 0.25 to 1.0 (4x improvement), recovering gradient signal
        # in the LIF pathway.  Fixed constant (not learnable) for:
        # - Biological plausibility (Na+ channel kinetics are fixed)
        # - Backward compatibility (no new nn.Parameter in state_dict)
        # - Numerical stability (no drift/NaN risk)
        # Ref: surrogate gradient sharpness in spiking neural networks.
        _SHARPNESS = 4.0
        # CF10 fix: Force FP32 for surrogate gradient to prevent sigmoid saturation
        # in FP16 (AMP).  In FP16, sigmoid saturates for |x| > ~2.75, creating
        # gradient dead zones.  FP32 extends non-saturated range to |x| > ~8.8.
        _v_diff_fp32 = (v_new - v_thresh_new).float()
        sig = torch.sigmoid(_SHARPNESS * _v_diff_fp32).to(v_new.dtype)  # (B, H) smooth
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
        # CF10 fix: Clamp adaptation current to prevent unbounded growth.
        # With tau_w=100 (decay_w=0.99), w can accumulate to ~13.7 in 32 TBPTT
        # steps, driving membrane to extreme negative values.  Biological
        # adaptation currents are bounded by maximal conductance densities.
        # min=0: adaptation is hyperpolarizing (outward K+ current), never depolarizing.
        # max=10*v_threshold: matches _i_syn_clamp convention, allows strong suppression.
        w_new = w_new.clamp(min=0.0, max=10.0 * self.v_threshold)

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
        # root 5: allocate the canonical state in the model's OWN compute dtype,
        # not the ambient global default.  Under a global torch float64 default a
        # float32 model otherwise creates float64 init state that crashes the
        # first float32 recurrence (double != float).  A float64 model (whose
        # parameter dtype is float64) still gets float64 state.
        _dt = self.W_in.weight.dtype
        v = torch.full(
            (batch_size, self.hidden_dim), self.v_rest, device=device, dtype=_dt,
        )
        i_syn = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=_dt)
        refract_counter = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=_dt)
        v_thresh_eff = torch.full(
            (batch_size, self.hidden_dim), self.v_threshold, device=device, dtype=_dt,
        )
        w_adapt = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=_dt)
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
            dtype=self.W_in.weight.dtype,
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
    Output: ``(B, T, H)`` — recurrent hidden states (default), or
            ``(out (B, T, H), h_n (num_layers, B, H))`` when
            ``return_hidden=True``.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
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
        return_hidden: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``(B, T, H)``
            lengths: ``(B,)`` — true sequence lengths.
            h0: ``(num_layers, B, H)`` — optional initial hidden state.
            return_hidden: If ``True``, also return the raw packed-GRU final
                hidden state ``h_n`` ``(num_layers, B, H)`` — the true
                per-sample endpoint, BEFORE any output-only gain.  The
                default ``False`` preserves the historical output-only
                return type ``(B, T, H)``.

        Returns:
            ``(B, T, H)`` when ``return_hidden=False``; otherwise
            ``(out (B, T, H), h_n (num_layers, B, H))``.
        """
        B, T, H = x.shape
        assert x.shape == (B, T, H)
        # H5 / finding A6: validate the ORIGINAL observation dtype at this
        # direct public boundary BEFORE ``.contiguous()``/finiteness/kernels.
        # A complex/int/bool observation (whose imaginary part a later cast
        # would discard) is refused; the top-level facade guard does not cover
        # direct GRUUnit callers.
        _require_real_floating_observation(x, name="x", context="GRUUnit")
        x = x.contiguous()

        # A zero-time tensor (T == 0) is unsupported and is refused BEFORE any
        # indexing.  This is distinct from ``lengths == 0`` (a zero-length
        # sample in a T > 0 batch), which is a valid exact no-op below.
        if T == 0:
            raise ValueError(
                "GRUUnit.forward requires T >= 1; a zero-time tensor (T=0) is "
                "unsupported (distinct from lengths==0, a valid no-op)."
            )
        lengths_i = _validate_original_lengths(
            lengths, B, T, context="GRUUnit",
        ).to(device=x.device).contiguous()

        if h0 is not None:
            h0 = _validate_original_carry(
                h0, (self.num_layers, B, H), key="h0",
                device=x.device, dtype=x.dtype,
            ).contiguous()

        # Sanitize invalid (padded) frames BEFORE any recurrent math via
        # selection, so a NaN/Inf padded suffix can never poison the GRU
        # output, the packed ``h_n`` carry, or the input/parameter/h0
        # gradients.  A nonfinite value in a VALID frame is refused (a real
        # observation is never silently sanitized).  ``torch.where`` (not an
        # arithmetic multiply) is the exact selection: invalid frames become
        # canonical zeros, which also makes the ``clamp(min=1)`` dummy frame
        # for an empty packed sequence numerically safe.
        valid = (
            torch.arange(T, device=x.device).unsqueeze(0) < lengths_i.unsqueeze(1)
        )                                                    # (B, T) bool
        if not torch.isfinite(x[valid]).all():
            raise ValueError(
                "GRUUnit received nonfinite values in valid frames; refusing "
                "to sanitize real observations."
            )
        x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))

        self.gru.flatten_parameters()

        # r6 R2: a zero-length row's clamped dummy packed frame must NOT
        # evaluate the original inactive ``h0``.  With a finite-but-huge
        # inactive carry (e.g. 1e38) and recurrent weights >1, the dummy
        # recurrence overflows; the value is discarded by the selection below,
        # but the discarded branch's backward computes ``0 * inf = NaN`` and
        # poisons the required identity gradient.  Substitute safe zeros for
        # inactive rows for the recurrence ONLY; the original carry is restored
        # by differentiable selection below (identity derivative 1, finite
        # zero cross/input derivatives).  Active rows use their own h0.
        empty = (lengths_i == 0)                               # (B,)
        h0_safe = h0
        if bool(empty.any()) and h0 is not None:
            h0_safe = torch.where(
                empty.view(1, B, 1), torch.zeros_like(h0), h0,
            )

        lengths_cpu = lengths_i.clamp(min=1).cpu().contiguous()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False,
        )
        packed_out, h_n = self.gru(packed, h0_safe)
        out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.shape[1],
        )
        assert out.shape == (B, T, H), (
            f"GRU output {tuple(out.shape)} != (B={B}, T={T}, H={H})"
        )
        # ``h_n`` is the raw recurrent carry (num_layers, B, H): the true
        # per-sample endpoint, unpadded and pre-gain.  ``pack_padded_sequence``
        # already returns the state at each sample's valid endpoint.
        assert h_n.shape == (self.num_layers, B, H), (
            f"GRU h_n {tuple(h_n.shape)} != (num_layers={self.num_layers}, "
            f"B={B}, H={H})"
        )

        # Zero-length rows: clamping the packed length to 1 makes the GRU
        # advance a garbage frame for ``lengths == 0`` samples.  A zero-length
        # sample has no valid frames, so its recurrent carry must be an EXACT
        # no-op — preserve the incoming ``h_n`` (or the canonical zeros when no
        # state was supplied) and export zeros.  This keeps a finished sample's
        # carry unchanged when a batch is resumed.
        if bool(empty.any()):
            carry0 = h0 if h0 is not None else torch.zeros(
                self.num_layers, B, H, device=x.device, dtype=h_n.dtype,
            )
            h_n = torch.where(empty.view(1, B, 1), carry0, h_n)
            out = torch.where(
                empty.view(B, 1, 1), torch.zeros_like(out), out,
            )

        if not return_hidden:
            return out
        return out, h_n


# ===============================================================
# 4.  MoR Router (Learned Representational-Routing Gate)
# ===============================================================

class MoRRouter(nn.Module):
    """
    Per-time-step routing network that blends LIF and GRU outputs.

    This is a learned representational-routing gate, not a causal-inference
    estimator.  The two weights are *coupled*: a ``softmax`` over the two
    logits makes ``g_lif + g_gru = 1`` per time-step.

    Input:  ``(B, H + M)`` at each step
    Output: ``(B, 2)`` — coupled softmax weights summing to 1
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
            ``(B, 2)`` — ``[g_lif, g_gru]`` as coupled softmax weights
            (each in [0, 1], summing to 1).
        """
        combined = torch.cat([e_sensory, mcmc_prior], dim=-1)
        logits = self.gate(combined)
        return torch.softmax(logits, dim=-1)


# ===============================================================
# 5.  Direction Head (Decoder)
# ===============================================================

class DirectionHead(nn.Module):
    """
    Final decoder.

    ``activation="relu"`` (default)::

        LayerNorm -> ReLU -> Dropout -> Linear(H, 1)

    ``activation="swiglu"``::

        n = Dropout(LayerNorm(h))
        out = Linear(H, 1)(SiLU(gate_proj(n)) * value_proj(n))

    ``gate_proj`` and ``value_proj`` are ``Linear(H, H)``; the readout is a
    separate ``Linear(H, 1)``.  SwiGLU is ``SiLU(W_gate x) * (W_value x)``.

    Input:  ``(B, T, H)``
    Output: ``(B, T)``
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """
        Args:
            hidden_dim: Input hidden dimensionality.
            dropout: Dropout probability (applied in both branches).
            activation: ``"relu"`` (default, historical) or ``"swiglu"``.

        Raises:
            ValueError: If *activation* is not a supported name.
        """
        super().__init__()
        if activation not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_VALID_ACTIVATIONS}, "
                f"got {activation!r}"
            )
        self.activation = activation
        self.hidden_dim = hidden_dim
        # Stable attribute for downstream introspection (e.g. the JAX eval
        # wrapper) so callers need not index into ``net``.
        self.dropout_rate = dropout
        if activation == "swiglu":
            self.net = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
            self.value_proj = nn.Linear(hidden_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, 1)
        else:
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
        B, T, H = h.shape
        if self.activation == "swiglu":
            n = self.net(h)
            assert n.shape == (B, T, H), (
                f"DirectionHead norm {tuple(n.shape)} != (B={B}, T={T}, H={H})"
            )
            y = self.out_proj(F.silu(self.gate_proj(n)) * self.value_proj(n))
        else:
            y = self.net(h)
        y = y.squeeze(-1)
        assert y.shape == (B, T), (
            f"DirectionHead output {tuple(y.shape)} != (B={B}, T={T})"
        )
        return y


# ===============================================================
# 6.  NSMoR Core Network
# ===============================================================

# ===============================================================
# 6a.  Frontend Encoder (Hybrid Funnel — Stage 1)
# ===============================================================

class FrontendEncoder(nn.Module):
    """
    Frontend encoder for the Hybrid Funnel architecture.

    Encapsulates dendritic compartmentalization (IIR filtering on
    visual channels) and :class:`SensoryEncoder` into a single
    front-end module.  The output ``e_sensory`` is detached before
    entering :class:`BioDecisionCore`, severing the gradient path
    between the two stages.

    This separation enables **two-phase training**:

    - **Phase 1:** Train FrontendEncoder with a simple MSE loss
      (regular curve fitting).  BioDecisionCore is frozen.
    - **Phase 2:** Freeze FrontendEncoder.  Train BioDecisionCore
      with physics / ATP / sparsity penalties.  Gradients never
      reach the frontend because of ``.detach()``.

    Input:  ``(B, T, D)`` — raw sensory features
    Output: ``(B, T, H)`` — sensory encoding (on computation graph)
    """

    def __init__(
        self,
        sensory_dim: int = 4,
        hidden_dim: int = 64,
        sensory_noise_std: float = 0.0,
        dendritic_tau: float = 0.0,
        dt_ms: float = 10.0,
        activation: str = "relu",
    ) -> None:
        """
        Args:
            sensory_dim: Input feature dimensionality (4).
            hidden_dim: Hidden representation dimensionality.
            sensory_noise_std: Gaussian noise std for stochastic resonance.
                0 disables.  Ref: Douglass et al. 1993.
            dendritic_tau: Time constant (ms) for dendritic IIR filter on
                visual channels.  Converted to per-step coefficient via
                ``exp(-dt_ms/tau_ms)``.  0 disables.
                Ref: London & Hausser 2005.
            dt_ms: Sampling interval in ms (physical time base for all
                time constants).
            activation: ``"relu"`` (default) or ``"swiglu"`` for the
                sensory encoder.
        """
        super().__init__()
        self.sensory_dim = sensory_dim
        self.sensory_encoder = SensoryEncoder(
            sensory_dim, hidden_dim, sensory_noise_std, activation=activation,
        )

        # ── Dendritic compartmentalization ──
        # Ref: London & Hausser 2005, Annu. Rev. Neurosci.
        # tau in PHYSICAL ms; per-step coefficient exp(-dt_ms/tau_ms)
        # (Reviewer Round-1 BLOCKER-1).
        assert dt_ms > 0.0, f"dt_ms must be > 0, got {dt_ms}"
        self.dt_ms = float(dt_ms)
        self.dendritic_tau = dendritic_tau
        self._dendritic_enabled = dendritic_tau > 0.0
        if self._dendritic_enabled:
            self._alpha_dend = math.exp(-self.dt_ms / dendritic_tau)
            assert 0.0 < self._alpha_dend < 1.0, (
                f"_alpha_dend={self._alpha_dend} out of (0, 1) for "
                f"dendritic_tau={dendritic_tau} ms at dt={self.dt_ms} ms"
            )
        else:
            self._alpha_dend = 0.0

    def forward(
        self,
        sensory_x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode raw sensory features through dendritic filtering
        and the sensory encoder.

        Args:
            sensory_x: ``(B, T, D)`` — raw sensory features.
            lengths: ``(B,)`` — true sequence lengths (for padding mask).

        Returns:
            ``(B, T, H)`` — sensory encoding.
        """
        B, T, _ = sensory_x.shape
        device = sensory_x.device

        # A zero-time tensor is unsupported and refused BEFORE any indexing
        # (distinct from ``lengths == 0``, a valid zero-length no-op sample).
        if T == 0:
            raise ValueError(
                "FrontendEncoder.forward requires T >= 1; a zero-time tensor "
                "(T=0) is unsupported (distinct from lengths==0)."
            )
        lengths_i = _validate_original_lengths(
            lengths, B, T, context="FrontendEncoder",
        ).to(device=device)

        # Reject nonfinite values in VALID frames: a NaN/Inf in a real
        # observation is a scientific/data error, not padding, and must not be
        # silently hidden.  Invalid (padded) frames are sanitized to zero below.
        valid_mask = (
            torch.arange(T, device=device).unsqueeze(0)
            < lengths_i.unsqueeze(1)
        )                                                    # (B, T) bool
        if not torch.isfinite(sensory_x[valid_mask]).all():
            raise ValueError(
                "FrontendEncoder received nonfinite values in valid frames; "
                "refusing to sanitize real observations."
            )
        # Sanitize invalid (padded) frames BEFORE any computation so overflow /
        # NaN in the padded suffix can never propagate into the encoder, the
        # dendritic IIR carry, or downstream state.
        sensory_x = torch.where(
            valid_mask.unsqueeze(-1), sensory_x,
            torch.zeros_like(sensory_x),
        )

        # Dendritic IIR filter on the visual channel (optional)
        if self._dendritic_enabled:
            alpha_dend = self._alpha_dend

            dend_state = getattr(self, '_dendritic_state', None)
            # Round-1 BLOCKER-2 item 3: the caller (NSMoRCore.forward)
            # now resets _dendritic_state to None at the start of every
            # forward call unless it explicitly restores autoregressive
            # state.  The shape check below remains as a secondary
            # guard for direct FrontendEncoder use.
            #
            # Historical (B, 2) carry is MIGRATED, not silently reset: the
            # current contract is a single visual-channel low-pass (B, 1).
            # Column 0 is the visual component; the discarded column was the
            # retired wind-memory filter (wind is no longer filtered).  Any
            # other rank/width/dtype is rejected.
            if dend_state is None:
                dend_state = torch.zeros(B, 1, device=device)
            else:
                if not isinstance(dend_state, torch.Tensor):
                    raise ValueError(
                        "dendritic carry must be a torch.Tensor, got "
                        f"{type(dend_state).__name__}"
                    )
                if dend_state.dim() != 2 or dend_state.shape[0] != B:
                    raise ValueError(
                        "dendritic carry must be (B, 1) [legacy (B, 2) "
                        f"migrated]; got shape {tuple(dend_state.shape)} for B={B}"
                    )
                if dend_state.shape[1] == 2:
                    # Legacy wind-memory width: keep the visual component.
                    dend_state = dend_state[:, 0:1]
                elif dend_state.shape[1] != 1:
                    raise ValueError(
                        "dendritic carry must be (B, 1) or legacy (B, 2); got "
                        f"width {dend_state.shape[1]}"
                    )
                if not dend_state.is_floating_point():
                    raise ValueError("dendritic carry must be floating point")
                if not torch.isfinite(dend_state).all():
                    raise ValueError(
                        "dendritic carry contains nonfinite values; refusing "
                        "to propagate a poisoned low-pass state."
                    )
                dend_state = dend_state.to(device=device, dtype=sensory_x.dtype)
                # Representability after conversion (r5 R1): the ORIGINAL
                # finiteness above is insufficient — a finite float64 carry
                # can overflow to float32 +inf when narrowed.
                if not torch.isfinite(dend_state).all():
                    raise ValueError(
                        "dendritic carry is not representable in the "
                        f"computation dtype {sensory_x.dtype}: finite values "
                        "overflow to nonfinite after conversion."
                    )
            # Snapshot the incoming carry BEFORE the IIR loop advances it, so
            # a zero-length sample can restore its exact no-op value AND keep
            # the identity gradient to the incoming state (r4 R6).  The
            # snapshot is deliberately NOT detached: a ``requires_grad``
            # incoming carry must satisfy d(exported_empty)/d(incoming) = 1.
            incoming = dend_state                               # (B, 1)

            # r6 R2: keep inactive rows OUT of the dendritic IIR before it is
            # evaluated.  A finite-but-huge inactive carry (e.g. 3e38) fed
            # through the IIR repopulates the padded visual frames and reaches
            # the encoder, whose backward through those discarded rows is
            # nonfinite.  Substitute a safe zero carry for inactive rows for
            # the IIR ONLY; the original incoming carry is restored for export
            # by differentiable selection below (identity derivative 1).
            empty_dend = (lengths_i == 0).unsqueeze(-1)         # (B, 1)
            dend_state = torch.where(
                empty_dend, torch.zeros_like(incoming), incoming,
            )

            # The 4-D sensory layout is [visual, wind, v_lag, a_lag]:
            # ONLY channel 0 is the visual channel that traverses the slow
            # optic-lobe dendrite.  Channels 1.. (wind, kinematic history)
            # are fast cercal/somatic signals and MUST bypass the filter.
            visual_raw = sensory_x[:, :, 0:1]        # (B, T, 1)
            bypass_raw = sensory_x[:, :, 1:]         # (B, T, D-1)

            _SEG_LEN = 32

            def _iir_segment(
                seg_input: torch.Tensor, seg_state: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                """Process one IIR segment."""
                seg_len = seg_input.shape[1]
                seg_out = torch.zeros_like(seg_input)
                s = seg_state
                for t_s in range(seg_len):
                    s = alpha_dend * s + (1.0 - alpha_dend) * seg_input[:, t_s, :]
                    seg_out[:, t_s, :] = s
                return seg_out, s

            dend_visual = torch.zeros_like(visual_raw)
            for seg_start in range(0, T, _SEG_LEN):
                seg_end = min(seg_start + _SEG_LEN, T)
                seg_input = visual_raw[:, seg_start:seg_end, :]

                if self.training and seg_input.requires_grad:
                    seg_out, dend_state = torch.utils.checkpoint.checkpoint(
                        _iir_segment, seg_input, dend_state,
                        use_reentrant=False,
                    )
                else:
                    seg_out, dend_state = _iir_segment(seg_input, dend_state)

                dend_visual[:, seg_start:seg_end, :] = seg_out

            # Freeze the exported carry at each sample's TRUE endpoint:
            # padded frames must not advance the dendritic low-pass state
            # that the next autoregressive call resumes from.  Selecting the
            # frame at ``lengths-1`` is exact and, for full-length samples,
            # equals the historical end-of-sequence state.  A ZERO-length
            # sample has no valid frame: it is an exact no-op that preserves
            # the incoming carry (never the surrogate frame-0 update).
            end_idx = (lengths_i - 1).clamp(min=0)              # (B,)
            dend_end = dend_visual[
                torch.arange(B, device=device), end_idx, 0
            ]                                                   # (B,)
            dend_end = dend_end.unsqueeze(-1)                   # (B, 1)
            empty = (lengths_i == 0).unsqueeze(-1)              # (B, 1)
            # Truncate (TBPTT) ONLY the actively advancing rows at their true
            # endpoint; a zero-length row keeps the differentiable identity
            # path to its incoming carry (r4 R6).  A blanket ``.detach()``
            # here severed the empty-row identity gradient (finding A6).
            self._dendritic_state = torch.where(
                empty, incoming, dend_end.detach(),
            )                                                   # (B, 1)
            sensory_x = torch.cat([dend_visual, bypass_raw], dim=-1)

        return self.sensory_encoder(sensory_x)


# ===============================================================
# 6b.  Bio-Decision Core (Hybrid Funnel — Stage 2)
# ===============================================================

class AdaptiveLatentRefinement(nn.Module):
    """Shared adaptive-recursion latent refinement (opt-in, architecture v1).

    A SHARED bounded residual block and a SHARED halt head are applied
    iteratively to the CURRENT post-fusion latent ``u_0 = h_fused`` (after the
    LIF/GRU fusion and before the direction head).  This is representational
    refinement, NOT additional biological time: one external timestep still
    advances the LIF membrane/synaptic/refractory state and the GRU temporal
    state exactly once.  The module never calls ``LIFCell``/``GRUUnit``.

    Modes (``mode``):

    * ``"off"`` — the module is not constructed; historical numerics/keys are
      unchanged.
    * ``"fixed"`` — the shared block is applied exactly ``max_steps`` times to
      every valid row (the halt head is present in the parameter tree but is NOT
      executed).  The refined output is the FINAL step ``u_K``.
    * ``"adaptive"`` — ACT-style adaptive depth.  ``p_k = sigmoid(HaltHead(u_k))``
      is shared across internal depth and external ``t``.  For each valid row the
      first index ``N`` with ``cum_N = sum_{j<=N} p_j >= 1 - eps`` (else
      ``max_steps``) defines the depth.  The output is the FULL PREFIX ACT
      mixture

          out = sum_{j=1..N-1} p_j * u_j  +  R * u_N ,   R = 1 - sum_{j<N} p_j

      with weights nonnegative and summing to one; at forced max depth the
      remainder mass is fully allocated to ``u_N``; ``N == 1`` gives ``out = u_1``.

    Real compute saving: only ACTIVE valid rows (not yet halted) are gathered and
    the shared block/halt head run on that subset each step; halted rows stop
    executing.  This is NOT all-K compute plus weighting.

    Gradient reality / limits (documented, not hidden):

    * The discrete first-crossing index ``N`` has NO pathwise derivative.  The
      trainable ponder surrogate is ``N.detach() + R`` (the ACT estimator): ``N``
      is a detached integer order, only the remainder ``R`` is differentiable.
      This surrogate is NOT actual work / MAC / ATP.  Actual integer depth is
      recorded separately (``refinement_depth``/``refinement_updates``) and is
      detached.  ``K == 1`` or immediate halting can have ZERO halt gradient.
    * The block update is bounded (``scale * tanh(...)``); LayerNorm alone is not
      a stability proof.  Finite gradients/numerics are checked empirically.
    * A negative halt-head bias DECREASES ``p`` and typically INCREASES depth; it
      is NOT used to encourage early halting.  Initialization is checked
      empirically (nondegenerate depth), not asserted.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        mode: str = "off",
        max_steps: int = 4,
        eps: float = 0.01,
        update_scale: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Share the refinement-option contract with the off-mode boundary so
        # the enabled and disabled paths cannot diverge.
        mode, max_steps, eps, update_scale = _validate_refinement_options(
            mode=mode, max_steps=max_steps, eps=eps, update_scale=update_scale,
            context="AdaptiveLatentRefinement",
        )
        # H1 / finding A1: ``mode="off"`` is NOT a valid module state.  This
        # public constructor is the ENABLED refinement module; "off" means the
        # module is absent, and ``BioDecisionCore`` already never constructs it
        # for "off".  Refusing "off" here is the smallest consistent contract:
        # it can never create parameters, consume RNG, or execute refinement
        # under an "off" flag.  The declared option contract is validated FIRST
        # (so a malformed off-mode request still fails closed on the option
        # itself), then "off" is refused.
        if mode == "off":
            raise ValueError(
                "AdaptiveLatentRefinement is the ENABLED refinement module and "
                "does not accept mode='off'; 'off' means the module is absent "
                "(BioDecisionCore builds it only for 'fixed'/'adaptive'). "
                "Construct with mode='fixed' or 'adaptive', or omit the module."
            )
        self.mode = mode
        self.max_steps = int(max_steps)
        self.eps = float(eps)
        self.update_scale = float(update_scale)
        self.hidden_dim = int(hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Shared halt head (used only in adaptive mode).
        self.halt = nn.Linear(hidden_dim, 1)

    def _block(self, u: torch.Tensor) -> torch.Tensor:
        h = self.norm(u)
        h = torch.tanh(self.fc1(h))
        h = self.fc2(h)
        # Bounded residual: |delta| <= update_scale.  LayerNorm alone is not a
        # stability proof; the bounded increment controls per-step drift.
        delta = self.update_scale * torch.tanh(h)
        return u + delta

    def forward(
        self, h_fused: torch.Tensor, valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine ``h_fused`` (B, T, H).

        Returns ``(out, depth, updates, ponder_cost, weights)`` where
        ``depth`` is int ``(B, T)`` (0 for padding), ``updates`` is an int
        scalar (``depth.sum()``), ``ponder_cost`` is a differentiable scalar
        normalized over valid tokens (0 when all-empty), and ``weights`` is
        ``(B, T, K)`` (valid rows sum to 1, padding 0).

        ``weights`` semantics differ by mode because the modes have different
        outputs: ``fixed`` returns the FINAL state ``u_K`` (one-hot mass at
        slot ``K``), while ``adaptive`` returns the ACT full-prefix mixture
        (mass spread over the executed steps).  Both are truthful per-row
        diagnostics, not a shared uniform mean.
        """
        B, T, H = h_fused.shape
        assert h_fused.shape == (B, T, H)
        assert valid.shape == (B, T)
        K = self.max_steps
        device = h_fused.device

        flat = h_fused.reshape(B * T, H)
        valid_flat = valid.reshape(B * T)
        n_valid = int(valid_flat.sum().item())
        updates_scalar = torch.zeros((), dtype=torch.long, device=device)

        if self.mode == "fixed":
            # Real-work contract: gather ONLY the valid rows and run the shared
            # block exactly K times on that subset.  An all-empty batch performs
            # NO block call (no fabricated work) and returns an exact-zero
            # output.  Padded rows are scattered as exact zeros, so they carry
            # no value and receive no parameter/input gradient contribution
            # (root G1 / findings A1, B2).
            out_flat = torch.zeros(B * T, H, device=device, dtype=h_fused.dtype)
            if n_valid > 0:
                idx = valid_flat.nonzero(as_tuple=True)[0]
                u = flat.index_select(0, idx)
                for _ in range(K):
                    u = self._block(u)
                out_flat = out_flat.index_copy(0, idx, u)
            out = out_flat.reshape(B, T, H)
            depth = torch.where(
                valid,
                torch.full((B, T), K, dtype=torch.long, device=device),
                torch.zeros((B, T), dtype=torch.long, device=device),
            )
            # Counter equals the ACTUAL executed rows (n_valid * K), zero for an
            # all-empty batch (no block call).
            updates_scalar = torch.tensor(
                n_valid * K, dtype=torch.long, device=device,
            )
            ponder = (
                # All-empty: a safe, graph-connected differentiable zero (H2).
                _refinement_empty_zero(h_fused, valid)
                if n_valid == 0 else torch.full((), float(K), device=device)
            )
            # Fixed output is the FINAL step u_K, so the truthful per-row weight
            # diagnostic is a one-hot at step K (mass fully on the executed
            # final state), NOT a misleading uniform 1/K mean over every step.
            # Valid rows sum to 1; padding is exactly 0.
            weights = torch.zeros(
                (B, T, K), device=device, dtype=h_fused.dtype,
            )
            weights[..., K - 1] = torch.where(
                valid, torch.ones((B, T), device=device, dtype=h_fused.dtype),
                torch.zeros((B, T), device=device, dtype=h_fused.dtype),
            )
            return out, depth, updates_scalar, ponder, weights

        # ── adaptive ──
        u = flat.clone()
        prefix = torch.zeros(B * T, H, device=device, dtype=h_fused.dtype)
        cum = torch.zeros(B * T, device=device, dtype=h_fused.dtype)
        final_out = torch.zeros(B * T, H, device=device, dtype=h_fused.dtype)
        depth_flat = torch.zeros(B * T, dtype=torch.long, device=device)
        weights_flat = torch.zeros(B * T, K, device=device, dtype=h_fused.dtype)
        halted = ~valid_flat  # invalid/padding rows start "done"
        N_order = torch.zeros(B * T, dtype=torch.long, device=device)
        R_val = torch.zeros(B * T, device=device, dtype=h_fused.dtype)
        total_updates = 0
        for k in range(1, K + 1):
            active = (~halted) & valid_flat
            n_active = int(active.sum().item())
            if n_active == 0:
                break
            idx = active.nonzero(as_tuple=True)[0]
            u_act = self._block(u[idx])
            p_act = torch.sigmoid(self.halt(u_act)).squeeze(-1)  # (n_active,)
            total_updates += n_active
            u = u.index_copy(0, idx, u_act)
            p = torch.zeros(B * T, device=device, dtype=h_fused.dtype)
            p = p.index_copy(0, idx, p_act)
            new_cum = cum + p
            # A row crosses AT this step when its cumulative halt reaches
            # 1 - eps for the first time here.
            just = active & (new_cum >= (1.0 - self.eps)) & (N_order == 0)
            if bool(just.any()):
                j_idx = just.nonzero(as_tuple=True)[0]
                # R = 1 - sum_{j<k} p_j = 1 - cum (cum is the PRE-step prefix).
                R_j = (1.0 - cum)[just]
                # out = sum_{j<k} p_j u_j + R * u_k  (prefix excludes this step).
                final_out = final_out.index_copy(
                    0, j_idx, prefix[just] + R_j.unsqueeze(-1) * u[just],
                )
                depth_flat = depth_flat.index_copy(
                    0, j_idx, torch.full_like(depth_flat[just], k),
                )
                N_order = N_order.index_copy(
                    0, j_idx, torch.full_like(N_order[just], k),
                )
                R_val = R_val.index_copy(0, j_idx, R_j)
            # Rows still active AFTER this step accumulate p_k u_k into the
            # prefix (they have k < N).  At the FINAL step (k == K) we do NOT
            # accumulate: the still-active rows become the never-crossing rows
            # whose remainder is allocated to u_K with R = 1 - sum_{j<K} p_j
            # (the same formula as crossing exactly at K).
            accum = active & ~just
            if k < K and bool(accum.any()):
                prefix = prefix + torch.where(
                    accum.unsqueeze(-1), p.unsqueeze(-1) * u, torch.zeros_like(u),
                )
            # Provisional weight p_k for active rows (final step overwritten by R).
            weights_flat[:, k - 1] = torch.where(active, p, torch.zeros_like(p))
            cum = new_cum
            halted = halted | (new_cum >= (1.0 - self.eps))
        # Rows that never crossed within K: N = K, remainder fully allocated to
        # u_K with R = 1 - sum_{j<K} p_j.  ``cum`` holds sum_{j<=K} p_j, so the
        # final-step halt probability ``p`` must be removed to recover the
        # prefix sum.  (``never`` is nonempty only when the loop ran to K, so
        # ``p`` is the K-th step value.)
        never = valid_flat & (N_order == 0)
        if bool(never.any()):
            n_idx = never.nonzero(as_tuple=True)[0]
            R_j = (1.0 - (cum - p))[never]
            final_out = final_out.index_copy(
                0, n_idx, prefix[never] + R_j.unsqueeze(-1) * u[never],
            )
            depth_flat = depth_flat.index_copy(
                0, n_idx, torch.full_like(depth_flat[never], K),
            )
            N_order = N_order.index_copy(
                0, n_idx, torch.full_like(N_order[never], K),
            )
            R_val = R_val.index_copy(0, n_idx, R_j)
        # The FINAL state weight is R (not p_N).  Put R in the N-th slot and
        # zero every slot beyond N, so valid rows sum to 1.
        for kk in range(1, K + 1):
            beyond = (N_order != 0) & (N_order < kk)
            is_n = N_order == kk
            col = weights_flat[:, kk - 1]
            col = torch.where(is_n, R_val, col)
            col = torch.where(beyond, torch.zeros_like(col), col)
            weights_flat[:, kk - 1] = col
        out = final_out.reshape(B, T, H)
        depth = depth_flat.reshape(B, T)
        updates_scalar = torch.tensor(total_updates, dtype=torch.long, device=device)
        weights = weights_flat.reshape(B, T, K)
        # Differentiable ACT ponder surrogate: N.detach() + R (per valid row).
        if n_valid == 0:
            # All-empty: a safe, graph-connected differentiable zero (H2), so
            # downstream backward never errors on a leaf-less scalar and NaN /
            # overflow padding cannot poison the cost.
            ponder = _refinement_empty_zero(h_fused, valid)
        else:
            N_detached = N_order.to(h_fused.dtype).detach()
            per_row = N_detached + R_val
            valid_flat_f = valid_flat.to(h_fused.dtype)
            ponder = (per_row * valid_flat_f).sum() / valid_flat_f.sum()
        return out, depth, updates_scalar, ponder, weights


class BioDecisionCore(nn.Module):
    """
    Bio-decision core for the Hybrid Funnel architecture.

    Encapsulates the dual-pathway recurrent network (LIF + GRU),
    the MoR Router, pathway integration, and the final decoder.
    This module receives **detached** sensory encodings from
    :class:`FrontendEncoder`, ensuring that physics / ATP / sparsity
    gradients never propagate back to the sensory frontend.

    Architecture::

        e_sensory_detached [B,T,H] -+-> LIF Path  -> out_lif  [B,T,H]
                                     |
            mcmc_prior [B,T,M] ---->-+-> MoR Router -> gates [B,T,2]
                                     |
                                     +-> GRU Path  -> out_gru  [B,T,H]
                                     |
                                     +-> Integration: H = g_lif*Out_lif + g_gru*Out_gru
                                     |
                                     +-> DirectionHead -> Y_pred [B,T]

    Input:  ``e_sensory`` ``(B, T, H)`` — detached sensory encoding
            ``mcmc_prior`` ``(B, T, M)`` — static MCMC prior vector
            ``lengths`` ``(B,)`` — true sequence lengths
    Output: ``y_pred`` ``(B, T)``
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        mcmc_dim: int = 4,
        num_gru_layers: int = 1,
        dropout: float = 0.1,
        lif_alpha: float = 0.9,
        lif_threshold: float = 1.0,
        lif_beta: float = 0.5,
        lif_abs_refract_ms: float = 0.0,
        lif_rel_refract_ms: float = 20.0,
        lif_tau_syn: float = 0.0,
        lif_v_rest: float = 0.0,
        lif_v_reset: Optional[float] = None,
        lif_tau_w: float = 0.0,
        lif_b_adapt: float = 0.0,
        lif_tau_fac: float = 0.0,
        lif_tau_rec: float = 0.0,
        lif_U_stp_init: float = 0.5,
        lif_lateral_inhibition: float = 0.0,
        lif_inhib_tau_ms: float = 50.0,
        gru_neuromod_gain: float = 0.0,
        lif_tbptt_steps: int = 64,
        dt_ms: float = 10.0,
        activation: str = "relu",
        refinement_mode: str = "off",
        refinement_max_steps: int = 4,
        refinement_eps: float = 0.01,
        refinement_update_scale: float = 1.0,
    ) -> None:
        """
        Args:
            hidden_dim: Hidden state dimensionality for both pathways.
            mcmc_dim: Dimensionality of MCMC prior vector (4).
            num_gru_layers: Number of stacked GRU layers.
            dropout: Dropout probability in GRU and decoder.
            lif_alpha: LIF leak factor.
            lif_threshold: LIF spike threshold.
            lif_beta: LIF input scaling.
            lif_abs_refract_ms: LIF absolute refractory period (ms; 0=disabled).
            lif_rel_refract_ms: LIF relative refractory decay length (ms; 0=disabled).
            lif_tau_syn: LIF synaptic time constant (ms; 0=bypasses).
            lif_v_rest: LIF resting membrane potential.
            lif_v_reset: LIF fixed reset potential (None=soft reset).
            lif_tau_w: LIF adaptation time constant (ms; 0=disabled).
            lif_b_adapt: LIF spike-triggered adaptation increment (0=disabled).
            lif_tau_fac: LIF STP facilitation time constant (0=disabled).
            lif_tau_rec: LIF STP recovery time constant (ms; 0=disabled).
            lif_U_stp_init: LIF STP baseline utilization.
            lif_lateral_inhibition: LIF lateral inhibition strength (0=disabled).
            gru_neuromod_gain: Neuromodulatory gain strength on GRU (0=disabled).
            lif_tbptt_steps: Truncated BPTT window for LIF (0=full BPTT).
            dt_ms: Sampling interval in ms (physical time base for all
                LIF time constants).
            activation: ``"relu"`` (default) or ``"swiglu"`` for the
                direction head.
            refinement_mode: ``"off"`` (default; module absent, historical
                numerics unchanged), ``"fixed"`` (shared block applied
                ``refinement_max_steps`` times to every valid row), or
                ``"adaptive"`` (ACT-style learned depth).
            refinement_max_steps: Maximum internal refinement depth K (>= 1).
            refinement_eps: ACT halting threshold; a row halts when its
                cumulative halt probability reaches ``1 - eps``.
            refinement_update_scale: Bound on the per-step residual update
                (``|delta| <= update_scale``).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mcmc_dim = mcmc_dim
        self.dt_ms = float(dt_ms)

        self.lif_cell = LIFCell(
            hidden_dim, lif_alpha, lif_threshold, lif_beta,
            abs_refract_ms=lif_abs_refract_ms,
            rel_refract_ms=lif_rel_refract_ms,
            tau_syn=lif_tau_syn,
            v_rest=lif_v_rest,
            v_reset=lif_v_reset,
            tau_w=lif_tau_w,
            b_adapt=lif_b_adapt,
            tau_fac=lif_tau_fac,
            tau_rec=lif_tau_rec,
            U_stp_init=lif_U_stp_init,
            lateral_inhibition=lif_lateral_inhibition,
            inhib_tau_ms=lif_inhib_tau_ms,
            dendritic_tau=0.0,  # dendritic filtering is in FrontendEncoder
            dt_ms=dt_ms,
        )
        self.gru_unit = GRUUnit(hidden_dim, num_gru_layers, dropout)
        self.router = MoRRouter(hidden_dim, mcmc_dim)
        self.direction_head = DirectionHead(hidden_dim, dropout, activation=activation)

        # ── Opt-in adaptive latent refinement (architecture v1) ──
        # ``off`` leaves the parameter tree / numerics bitwise unchanged; the
        # module is only constructed for ``fixed``/``adaptive``.
        #
        # Root G6 / finding B6: the ORIGINAL refinement options are validated
        # independently of whether a module is constructed, so an ``off`` model
        # still refuses a bool/NaN/negative/illegal setting at the public trust
        # boundary (mirroring ModelConfig).  Validation runs BEFORE the off
        # branch, so ``off`` consumes no RNG and adds no parameters.
        (refinement_mode, refinement_max_steps, refinement_eps,
         refinement_update_scale) = _validate_refinement_options(
            mode=refinement_mode, max_steps=refinement_max_steps,
            eps=refinement_eps, update_scale=refinement_update_scale,
            context="BioDecisionCore",
        )
        self.refinement_mode = refinement_mode
        if refinement_mode == "off":
            self.refinement = None
        else:
            self.refinement = AdaptiveLatentRefinement(
                hidden_dim,
                mode=refinement_mode,
                max_steps=refinement_max_steps,
                eps=refinement_eps,
                update_scale=refinement_update_scale,
                dropout=dropout,
            )

        # ── Neuromodulatory gain for GRU pathway (Gap C) ──
        self.gru_neuromod_gain = gru_neuromod_gain
        if gru_neuromod_gain > 0.0:
            self._gain_scale = nn.Parameter(torch.tensor(0.0))
            self._gain_bias = nn.Parameter(torch.tensor(1.0))

        # ── Truncated BPTT for LIF pathway (CF7 fix) ──
        self._tbptt_steps = lif_tbptt_steps if lif_tbptt_steps > 0 else 0

    def forward(
        self,
        e_sensory: torch.Tensor,
        mcmc_prior: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_internals: bool = False,
        override_gates: Optional[Dict[str, float]] = None,
        states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]] | Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through the bio-decision core.

        Args:
            e_sensory: ``(B, T, H)`` — sensory encoding (should be detached).
            mcmc_prior: ``(B, T, M)`` — MCMC prior vector.
            lengths: ``(B,)`` — true sequence lengths.
            return_internals: If ``True``, return internals dict.
            override_gates: Optional dict for in-silico lesioning.
            states: Optional dict of recurrent states for autoregressive mode.

        Returns:
            Same as :meth:`NSMoRCore.forward`.
        """
        B, T, H = e_sensory.shape
        device = e_sensory.device

        assert e_sensory.shape == (B, T, self.hidden_dim), (
            f"e_sensory shape {tuple(e_sensory.shape)} != "
            f"(B={B}, T={T}, H={self.hidden_dim})"
        )

        # H4 / findings A5, B2: one shared preflight of every enabled mechanism
        # (gain, STP, lateral inhibition) against its required parameter tree,
        # BEFORE any leaf is read.  An inconsistent live flag (mutated after
        # construction) fails closed with a descriptive ValueError instead of a
        # bare AttributeError deep inside the recurrence.
        assert_mechanism_tree_consistent(self, context="BioDecisionCore")

        # A zero-time tensor is refused before indexing (distinct from
        # ``lengths == 0``, a valid zero-length no-op sample).
        if T == 0:
            raise ValueError(
                "BioDecisionCore.forward requires T >= 1; a zero-time tensor "
                "(T=0) is unsupported (distinct from lengths==0)."
            )
        lengths_i = _validate_original_lengths(
            lengths, B, T, context="BioDecisionCore",
        ).to(device=device)
        lengths = lengths_i
        # Sanitize invalid (padded) frames BEFORE any GRU/LIF/router/head math
        # so a finite-overflow or NaN padded suffix can never poison a
        # recurrent carry or the next prediction.  The raw SENSORY channels are
        # already sanitized in the frontend (before the encoder); the MCMC
        # prior reaches the router/GRU directly and is sanitized here.  Nonfinite
        # values in a VALID frame are refused (a real observation is never
        # silently sanitized).  Selection (torch.where) is used, not arithmetic
        # masking.
        valid = (
            torch.arange(T, device=device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )                                                    # (B, T) bool
        # ── Root G3 / findings A3, B4: validate the ORIGINAL observation dtype
        # at this public boundary BEFORE any conversion.  A complex measurement
        # (whose imaginary part the later ``.float()`` would silently discard)
        # or an integer/boolean observation is refused rather than coerced.
        _require_real_floating_observation(
            e_sensory, name="e_sensory", context="BioDecisionCore",
        )
        _require_real_floating_observation(
            mcmc_prior, name="mcmc_prior", context="BioDecisionCore",
        )
        if not torch.isfinite(e_sensory[valid]).all():
            raise ValueError(
                "BioDecisionCore received nonfinite sensory encoding in valid "
                "frames; refusing to sanitize real observations."
            )
        if not torch.isfinite(mcmc_prior[valid]).all():
            raise ValueError(
                "BioDecisionCore received nonfinite MCMC prior in valid "
                "frames; refusing to sanitize real observations."
            )
        mcmc_prior = torch.where(
            valid.unsqueeze(-1), mcmc_prior, torch.zeros_like(mcmc_prior),
        )
        # Sanitize invalid ENCODED frames too.  ``e_sensory`` is supplied
        # directly by a caller (NSMoRCore passes the frontend output, but
        # BioDecisionCore is a public boundary reachable on its own), so a
        # NaN/Inf padded suffix must be excluded BEFORE the LIF/GRU/router/head
        # math rather than only masked afterwards — otherwise it poisons the
        # output, the routing gates and every parameter/input gradient.  The
        # valid-frame finiteness above already refused real nonfinite input.
        e_sensory = torch.where(
            valid.unsqueeze(-1), e_sensory, torch.zeros_like(e_sensory),
        )

        # ── Initial states (autoregressive mode, CF3: +rel_refract) ──
        # Each call initializes the canonical LIF tuple and caches for THIS
        # call, then overlays every independently supplied carry field with
        # validation.  The overlay is deliberately NOT gated on ``lif_v``: a
        # partial state supplying only ``lif_i_syn`` / ``lif_w_adapt`` /
        # ``lif_rel_refract`` / ``lif_spike_history`` / ... must be honored,
        # and a field the caller omits must reset to its canonical default
        # rather than reuse a prior call's cache (spike-history leakage).
        lif_state0: Optional[Tuple[torch.Tensor, ...]] = None
        gru_h0: Optional[torch.Tensor] = None
        if states is not None:
            # Physical-domain map (r6 R3): counters/adaptation are non-negative;
            # STP resource/facilitation fractions lie in [0, 1].  Membrane and
            # GRU coordinates have no arbitrary bound (absent from this map).
            _DOMAINS: Dict[str, str] = {
                "lif_refract": "nonneg",
                "lif_rel_refract": "nonneg",
                "lif_w_adapt": "nonneg",
                "lif_x_resource": "fraction",
                "lif_u_facil": "fraction",
            }

            def _carry(key: str, ref: torch.Tensor) -> torch.Tensor:
                """Return a validated ``(B, H)`` supplied field, else *ref*."""
                supplied = states.get(key, None)
                if supplied is None:
                    return ref
                # Validate the ORIGINAL dtype, shape and finiteness BEFORE any
                # conversion: a boolean/integer/complex carry (whose imaginary
                # part ``.float()`` would silently discard) or a nonfinite
                # carry (e.g. ``lif_v=NaN``) is rejected rather than poisoning
                # the recurrence.  Shared with the raw-JAX boundary.  The dtype
                # is the ACTUAL forced-FP32 computation dtype, not a
                # default-dtype canonical allocation (r6 R1).
                return _validate_original_carry(
                    supplied, tuple(ref.shape), key=key,
                    device=device, dtype=torch.float32,
                    domain=_DOMAINS.get(key),
                )

            # Canonical tuple (also resets the lateral-inhibition spike-history
            # cache).  ``init_state`` leaves ``_dendritic_state`` untouched for
            # this cell (its ``_dendritic_enabled`` is False); the frontend
            # dendritic carry is restored by ``NSMoRCore.forward``.
            _canonical = self.lif_cell.init_state(B, device)
            _keys: Tuple[Optional[str], ...] = (
                "lif_v", "lif_i_syn", "lif_refract", None, "lif_w_adapt",
                "lif_rel_refract",
            )
            if len(_canonical) == 8:  # STP enabled
                _keys = _keys + ("lif_x_resource", "lif_u_facil")
            assert len(_keys) == len(_canonical), (
                f"LIF carry key map {len(_keys)} != canonical state "
                f"{len(_canonical)}"
            )
            lif_state0 = tuple(
                canon if key is None else _carry(key, canon)
                for key, canon in zip(_keys, _canonical)
            )
            # The stacked GRU carry ``(num_layers, B, H)`` is validated on its
            # ORIGINAL dtype/shape/finiteness against the ACTUAL forced-FP32
            # computation dtype before the later ``.float()``.
            _gru_in = states.get("gru_h", None)
            if _gru_in is not None:
                gru_h0 = _validate_original_carry(
                    _gru_in, (self.gru_unit.num_layers, B, self.hidden_dim),
                    key="gru_h", device=device, dtype=torch.float32,
                )

            # Install supplied caches AFTER canonical initialization so an
            # omitted cache stays reset (never a prior-call value).  A
            # malformed cache is rejected, not silently replaced by the
            # canonical default.
            if self.lif_cell._dendritic_enabled:
                dend_state_in = states.get("lif_dendritic_state", None)
                if dend_state_in is not None:
                    self.lif_cell._dendritic_state = _validate_original_carry(
                        dend_state_in, (B, self.hidden_dim),
                        key="lif_dendritic_state", device=device,
                        dtype=torch.float32,
                    )

            if self.lif_cell.lateral_inhibition > 0.0:
                spike_hist_in = states.get("lif_spike_history", None)
                if spike_hist_in is not None:
                    # r6 R3: the spike history is an EMA of binary spikes, so
                    # its justified domain is [0, 1].
                    self.lif_cell._spike_history = _validate_original_carry(
                        spike_hist_in, (B, self.hidden_dim),
                        key="lif_spike_history", device=device,
                        dtype=torch.float32, domain="fraction",
                    )

        # ── Path A: LIF (step-by-step) ──
        (out_lif, lif_potentials, lif_spikes, lif_thresholds,
         lif_v_final, lif_i_syn_final, lif_refract_final,
         lif_w_adapt_final, lif_rel_refract_final,
         lif_x_resource_final, lif_u_facil_final,
         lif_w_adapt_over_time) = self._run_lif_path(
            e_sensory, lengths, lif_state0=lif_state0,
        )

        assert out_lif.shape == (B, T, self.hidden_dim), (
            f"out_lif shape {tuple(out_lif.shape)} != (B={B}, T={T}, H={self.hidden_dim})"
        )

        # ── Non-LIF paths forced to FP32 (residual NaN gradient fix) ──
        # CF10 forced the LIF loop to FP32, eliminating most NaN gradients.
        # The remaining 1-element NaN comes from the GRU, Router softmax,
        # and DirectionHead running under AMP FP16:
        #
        #   1. cuDNN FP16 GRU: sigmoid/tanh saturate for |x|>2.75 in FP16
        #      (vs 8.8 in FP32).  Gate derivatives flush to zero (denorm),
        #      then multiply by large upstream gradients -> NaN.
        #   2. Router softmax: exp(logit) overflows for logit>11.1 in FP16,
        #      producing Inf/Inf=NaN.  Softmax backward with underflowed
        #      softmax_j=0 produces 0*large=NaN.
        #   3. DirectionHead LayerNorm: affine gradient passes through FP16,
        #      accumulating rounding error over H=64 dimensions.
        #
        # FP32 cost is negligible: GRU matmuls on (B, T, H=64) are not
        # the compute-bound bottleneck (that would be large transformer
        # models).  The Router and DirectionHead are element-wise on small
        # tensors.
        _amp_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=_amp_device, enabled=False):
            # Cast inputs to FP32 for the non-LIF computation graph
            e_sensory_f32 = e_sensory.float()
            mcmc_prior_f32 = mcmc_prior.float()
            # ── Root G4 / findings A3, B5: validate the ACTIVE operands against
            # the ACTUAL forced-FP32 computation representation AFTER the safe
            # padding selection (invalid frames are already exact zeros) and
            # BEFORE any numerical kernel.  A finite original float64 value such
            # as 1e300 overflows to +inf on narrowing and would poison the
            # router/GRU and export NaN y and gates.  Padding is already zero,
            # so harmless overflowing padding is never rejected and valid
            # representable controls pass.  No clamp / nan_to_num workaround.
            for _name, _orig, _f32 in (
                ("e_sensory", e_sensory, e_sensory_f32),
                ("mcmc_prior", mcmc_prior, mcmc_prior_f32),
            ):
                if not torch.isfinite(_f32[valid]).all():
                    raise ValueError(
                        f"BioDecisionCore received {_name} whose finite "
                        f"{_orig.dtype} values are not representable in the "
                        f"float32 computation representation (overflow on "
                        f"narrowing) in valid frames; refusing to run a "
                        f"poisoned kernel."
                    )
            # Cast GRU hidden state to FP32 (may be FP16 from autoregressive
            # mode where states were stored under AMP).
            gru_h0_f32 = gru_h0.float() if gru_h0 is not None else None

            # ── Path B: GRU (packed) ──
            # ``gru_h_n`` is the RAW packed-GRU final hidden state
            # (num_layers, B, H) — the true per-sample endpoint, before the
            # output-only neuromodulatory gain and without padding.  It is
            # the correct recurrent carry; exporting the padded/post-gain
            # trajectory (the historical bug) zeroed the carry on padded
            # samples and recycled a gain-scaled value.
            out_gru, gru_h_n = self.gru_unit(
                e_sensory_f32, lengths, h0=gru_h0_f32, return_hidden=True,
            )
            # Raw recurrent GRU trajectory BEFORE the output-only gain (r5 R5).
            # Scientific analysis (fixed-point / Jacobian / slow-point) must use
            # the actual recurrent coordinate, not the gain-scaled routed
            # output; ``out_gru`` below carries the gain for the routed output
            # contract, while ``gru_hidden_raw`` is the true state trajectory.
            gru_hidden_raw = out_gru

            # ── Neuromodulatory gain on GRU (Gap C) ──
            # H4 / findings A5, B2: the shared mechanism-tree preflight (run
            # once before the LIF path) already refused an inconsistent enabled
            # flag, so this branch may read its parameters directly.
            if self.gru_neuromod_gain > 0.0:
                mcmc_safe = mcmc_prior_f32.clamp(min=1e-8)
                entropy = -(mcmc_safe * mcmc_safe.log()).sum(dim=-1)
                max_entropy = math.log(self.mcmc_dim)
                entropy_norm = entropy / max_entropy
                gain = torch.sigmoid(
                    self._gain_scale * entropy_norm + self._gain_bias
                ) * 2.0
                gain = gain.unsqueeze(-1)
                out_gru = out_gru * gain

            # ── MoR Router ──
            e_flat = e_sensory_f32.reshape(B * T, -1)
            m_flat = mcmc_prior_f32.reshape(B * T, -1)
            gates = self.router(e_flat, m_flat)
            gates = gates.reshape(B, T, 2)

            g_lif = gates[:, :, 0:1]
            g_gru = gates[:, :, 1:2]

            # ── In-Silico Lesion Hook ──
            if override_gates is not None:
                if "g_lif" in override_gates:
                    g_lif = torch.full_like(g_lif, override_gates["g_lif"])
                if "g_gru" in override_gates:
                    g_gru = torch.full_like(g_gru, override_gates["g_gru"])

            # ── Integration ──
            h_out = g_lif * out_lif.float() + g_gru * out_gru

            # ── Opt-in adaptive latent refinement (architecture v1) ──
            # Applied to the POST-FUSION latent, BEFORE the direction head.
            # This is representational refinement only: the LIF membrane /
            # synaptic / refractory state and the GRU temporal state each
            # advanced exactly ONCE above (one external timestep = one
            # LIF/GRU advance).  Padding / empty rows never recurse (depth 0).
            if self.refinement is not None:
                (h_refined, refinement_depth, refinement_updates,
                 refinement_ponder_cost, refinement_weights) = self.refinement(
                    h_out, valid,
                )
            else:
                h_refined = h_out
                refinement_depth = torch.zeros(
                    (B, T), dtype=torch.long, device=device,
                )
                refinement_updates = torch.zeros(
                    (), dtype=torch.long, device=device,
                )
                refinement_ponder_cost = torch.zeros((), device=device)
                refinement_weights = h_out.new_zeros((B, T, 0))

            # ── Decode ──
            y_pred = self.direction_head(h_refined)

        assert y_pred.shape == (B, T), (
            f"y_pred shape {tuple(y_pred.shape)} != (B={B}, T={T})"
        )

        # ── Build output ──
        effective_gates = torch.cat([g_lif, g_gru], dim=-1)

        internals: Dict[str, torch.Tensor] = {
            "routing_gates": effective_gates,
            "natural_gates": gates,
            "lif_potentials": lif_potentials,
            "lif_spikes": lif_spikes,
            "lif_thresholds": lif_thresholds,
            "lif_w_adapt": lif_w_adapt_over_time,
            # Existing routed-output contract (post output-only gain).  Kept
            # unchanged for backward compatibility.
            "gru_hidden": out_gru,
            # Additive RAW recurrent GRU trajectory (B, T, H), pre-gain — the
            # actual recurrent coordinate for fixed-point / Jacobian analysis
            # (r5 R5).  When the gain is disabled this equals ``gru_hidden``.
            "gru_hidden_raw": gru_hidden_raw,
        }
        assert internals["gru_hidden_raw"].shape == (B, T, self.hidden_dim), (
            f"gru_hidden_raw shape {tuple(internals['gru_hidden_raw'].shape)} "
            f"!= (B={B}, T={T}, H={self.hidden_dim})"
        )
        # ── Adaptive latent refinement internals (architecture v1) ──
        # Present ONLY when refinement is enabled, so an ``off`` model's
        # internals dict is bitwise/structurally unchanged (frozen additive
        # names, enabled modes only).  When enabled these are always populated
        # (fixed mode reports the executed K, adaptive mode the per-row depth).
        if self.refinement is not None:
            assert h_refined.shape == (B, T, self.hidden_dim), (
                f"refined_hidden shape {tuple(h_refined.shape)} != "
                f"(B={B}, T={T}, H={self.hidden_dim})"
            )
            assert refinement_depth.shape == (B, T), (
                f"refinement_depth shape {tuple(refinement_depth.shape)} != "
                f"(B={B}, T={T})"
            )
            K = self.refinement.max_steps
            assert refinement_weights.shape == (B, T, K), (
                f"refinement_weights shape {tuple(refinement_weights.shape)} != "
                f"(B={B}, T={T}, K={K})"
            )
            internals["refined_hidden"] = h_refined
            internals["refinement_depth"] = refinement_depth
            internals["refinement_updates"] = refinement_updates
            internals["refinement_ponder_cost"] = refinement_ponder_cost
            internals["refinement_weights"] = refinement_weights

        if states is not None:
            states_out: Dict[str, torch.Tensor] = {
                "lif_v": lif_v_final.contiguous(),
                "lif_i_syn": lif_i_syn_final.contiguous(),
                "lif_refract": lif_refract_final.contiguous(),
                "lif_w_adapt": lif_w_adapt_final.contiguous(),
                "lif_rel_refract": lif_rel_refract_final.contiguous(),
                # Raw packed-GRU carry (num_layers, B, H) at each sample's
                # true endpoint, pre-gain and unpadded.  This is the correct
                # recurrent state, not the padded/post-gain trajectory.
                "gru_h": gru_h_n.contiguous(),
            }
            if self.lif_cell.stp_enabled:
                states_out["lif_x_resource"] = lif_x_resource_final.contiguous()
                states_out["lif_u_facil"] = lif_u_facil_final.contiguous()
            if self.lif_cell._dendritic_enabled:
                dend_state = getattr(self.lif_cell, '_dendritic_state', None)
                if dend_state is not None:
                    states_out["lif_dendritic_state"] = dend_state.contiguous()
            if self.lif_cell.lateral_inhibition > 0.0:
                spike_hist = getattr(self.lif_cell, '_spike_history', None)
                if spike_hist is not None:
                    states_out["lif_spike_history"] = spike_hist.contiguous()
            return y_pred, internals, states_out

        if return_internals:
            return y_pred, internals

        return y_pred

    def _run_lif_path(
        self,
        e_sensory: torch.Tensor,
        lengths: torch.Tensor,
        lif_state0: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the LIF cell step-by-step, masking padded positions."""
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
        w_adapt_over_time = torch.zeros(B, T, H, device=device)

        # CF10 fix: Force FP32 for the entire LIF loop to prevent NaN
        # gradients under AMP (FP16).  The LIF cell uses discontinuous
        # spike-and-reset dynamics with surrogate gradients that are
        # numerically sensitive in FP16:
        #
        #   1. The sigmoid surrogate gradient saturates to exactly 0 or 1
        #      in FP16 for |input| > ~2.75, killing gradient signal.
        #   2. The IIR synaptic filter and membrane recurrence amplify
        #      FP16 rounding errors over 32 TBPTT steps.
        #   3. The clamp operations create gradient dead zones that
        #      destabilize Adam's moment estimates in FP16.
        #
        # FP32 cost is negligible: LIF ops are element-wise on (B, H)
        # tensors, not the compute-bound bottleneck (that's the GRU).
        _amp_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=_amp_device, enabled=False):
            # Cast state tensors to FP32 for the loop
            lif_state = tuple(s.float() for s in lif_state)

            for t in range(T):
                inp_t = e_sensory[:, t, :].float()
                # Per-sample validity at this frame: True for t < lengths.
                mask_b = (t < lengths)                       # (B,) bool
                mask_2d = mask_b.unsqueeze(-1)               # (B, 1)
                if self._tbptt_steps > 0 and t > 0 and t % self._tbptt_steps == 0:
                    # Selective TBPTT: detach ONLY the genuinely advancing
                    # (active) rows at a boundary.  A finished row (t >= length)
                    # keeps its differentiable carried path so other samples'
                    # padding / shorter lengths cannot sever its gradient.
                    lif_state = tuple(
                        torch.where(mask_2d, s.detach(), s) for s in lif_state
                    )
                prev_state = lif_state
                _lat_inhib = self.lif_cell.lateral_inhibition > 0.0
                if _lat_inhib:
                    prev_spike_hist = getattr(self.lif_cell, '_spike_history', None)
                    if prev_spike_hist is None:
                        prev_spike_hist = torch.zeros(
                            B, H, device=device, dtype=lif_state[0].dtype,
                        )
                # r5 R2 (corrected): keep inactive rows OUT of the recurrence
                # arithmetic BEFORE it is evaluated by substituting a safe
                # finite operand (zeros) for the LIF step only.  This is
                # defense-in-depth: it bounds every inactive-row operand to a
                # provably-finite value, so no future arithmetic change in the
                # LIF cell can turn an extreme (but validated finite) inactive
                # carry into a discarded-branch nonfinite backward.
                #
                # Honest scope (A9/B9): the ORIGINAL claim that feeding an
                # extreme relative-refractory counter into ``exp(-k_rel *
                # counter)`` produces ``0 * inf = NaN`` in the backward is NOT
                # reproducible.  For every non-negative counter the counters
                # admit, ``exp(-k_rel * counter)`` underflows to exactly 0 with
                # derivative 0, so the discarded branch stays finite; a 5-field
                # x 8-value x config sweep found no poisoning input.  The
                # value-preservation contract (inactive carry restored exactly
                # by the differentiable selection below: identity derivative 1,
                # cross-field derivative 0) is what is actually exercised.
                # Active rows use their own state unchanged.
                safe_state = tuple(
                    torch.where(mask_2d, s, torch.zeros_like(s))
                    for s in lif_state
                )
                # The lateral-inhibition spike-history cache lives OUTSIDE
                # ``lif_state``; substitute a safe zero history for inactive
                # rows for the LIF step only, then restore the original with
                # differentiable selection below (r6 R2).  A finite-but-huge
                # inactive cache (e.g. [1e38, -1e38, ...]) fed to
                # ``spike_hist @ W_inhib`` overflows; its discarded backward
                # would be NaN.
                if _lat_inhib and prev_spike_hist is not None:
                    self.lif_cell._spike_history = torch.where(
                        mask_2d, prev_spike_hist,
                        torch.zeros_like(prev_spike_hist),
                    )
                # r8 R5: ``_checked=True`` skips the per-step public trust
                # boundary -- the supplied carry was already validated once at
                # the BioDecisionCore.forward entry (lif_state0) and every
                # subsequent step consumes state this loop produced.  Numerics
                # are bitwise unchanged; only the redundant host syncs are gone.
                spike, lif_state_new = self.lif_cell(
                    inp_t, safe_state, _checked=True,
                )

                # Freeze EVERY per-sample recurrent/cache field once the true
                # endpoint is passed: a padded frame must not advance the
                # membrane, synaptic current, refractory counters, adaptation,
                # STP, relative-refractory counter or lateral-inhibition
                # spike history.  Selection via ``torch.where`` (not arithmetic
                # multiply) is the exact per-sample freeze: it keeps the
                # valid-frame gradient path unchanged (mask True there) and,
                # unlike ``s*mask``, cannot let a nonfinite padded-frame value
                # survive as ``NaN * 0``.
                lif_state = tuple(
                    torch.where(mask_2d, s, p)
                    for s, p in zip(lif_state_new, prev_state)
                )
                if _lat_inhib:
                    sh_new = getattr(self.lif_cell, '_spike_history', None)
                    if sh_new is not None:
                        # Restore the ORIGINAL inactive history (r6 R2): the
                        # cache was zeroed for the inactive rows before the
                        # LIF step, so select the advancing value only for
                        # active rows and the true previous history otherwise.
                        self.lif_cell._spike_history = torch.where(
                            mask_2d, sh_new, prev_spike_hist,
                        )

                out_lif[:, t, :] = torch.where(
                    mask_2d, spike, torch.zeros_like(spike),
                )
                potentials[:, t, :] = torch.where(
                    mask_2d, lif_state[0], torch.zeros_like(lif_state[0]),
                )
                spikes[:, t, :] = torch.where(
                    mask_2d, spike, torch.zeros_like(spike),
                )
                w_adapt_over_time[:, t, :] = torch.where(
                    mask_2d, lif_state[4], torch.zeros_like(lif_state[4]),
                )
                thresh_over_time[:, t, :] = torch.where(
                    mask_2d, lif_state[3], torch.zeros_like(lif_state[3]),
                )

        v_final = lif_state[0]
        i_syn_final = lif_state[1]
        refract_final = lif_state[2]
        w_adapt_final = lif_state[4]
        rel_refract_final = lif_state[5]

        if self.lif_cell.stp_enabled and len(lif_state) == 8:
            x_resource_final = lif_state[6]
            u_facil_final = lif_state[7]
        else:
            x_resource_final = torch.ones(B, H, device=device)
            u_facil_final = torch.zeros(B, H, device=device)

        return (out_lif, potentials, spikes, thresh_over_time,
                v_final, i_syn_final, refract_final, w_adapt_final,
                rel_refract_final, x_resource_final, u_facil_final,
                w_adapt_over_time)


# ===============================================================
# 6c.  NSMoR Core Network (Hybrid Funnel Composition)
# ===============================================================

_FREEZABLE_MODULES = frozenset({
    "sensory_encoder",
    "lif_cell",
    "gru_unit",
    "router",
    "direction_head",
    "refinement",
})


class NSMoRCore(nn.Module):
    """
    Mixture-of-Recursions (MoR) — Hybrid Funnel architecture.

    Composes :class:`FrontendEncoder` and :class:`BioDecisionCore`
    with a **gradient-severing** ``.detach()`` boundary between them::

        X_batch --+-- Sensory_X [B,T,4]
                   |       |
                   |  FrontendEncoder
                   |       |
                   |   e_sensory [B,T,H] -- .detach() --> e_detached
                   |                                      |
                   |                           BioDecisionCore
                   |       MCMC_Prior [B,T,4] ----->      |
                   |                                      |
                   +-- Integration + Decode -> Y_pred [B,T]

    The ``.detach()`` ensures that Phase-2 physics / ATP / sparsity
    gradients never propagate back to the sensory frontend, enabling
    clean two-phase training.

    Backward-compatible: ``forward()`` signature is unchanged.
    ``sensory_encoder``, ``lif_cell``, etc. remain accessible as
    attributes (delegated to the sub-modules).
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
        lif_abs_refract_ms: float = 0.0,
        lif_rel_refract_ms: float = 20.0,
        lif_tau_syn: float = 0.0,
        lif_v_rest: float = 0.0,
        lif_v_reset: Optional[float] = None,
        lif_tau_w: float = 0.0,
        lif_b_adapt: float = 0.0,
        lif_tau_fac: float = 0.0,
        lif_tau_rec: float = 0.0,
        lif_U_stp_init: float = 0.5,
        lif_lateral_inhibition: float = 0.0,
        lif_inhib_tau_ms: float = 50.0,
        lif_dendritic_tau: float = 0.0,
        gru_neuromod_gain: float = 0.0,
        sensory_noise_std: float = 0.0,
        lif_tbptt_steps: int = 64,
        dt_ms: float = 10.0,
        persistence_skip: float = 0.0,
        activation: str = "relu",
        refinement_mode: str = "off",
        refinement_max_steps: int = 4,
        refinement_eps: float = 0.01,
        refinement_update_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if (
            isinstance(persistence_skip, bool)
            or not isinstance(persistence_skip, (int, float))
            or not math.isfinite(persistence_skip)
            or not 0.0 <= persistence_skip <= 1.0
        ):
            raise ValueError("persistence_skip must be a finite scalar in [0, 1]")
        if activation not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_VALID_ACTIVATIONS}, "
                f"got {activation!r}"
            )
        self.persistence_skip: float = float(persistence_skip)
        self.activation = activation
        self.refinement_mode = refinement_mode
        self.sensory_dim = sensory_dim
        self.mcmc_dim = mcmc_dim
        self.hidden_dim = hidden_dim
        self.dt_ms = float(dt_ms)

        # ── Hybrid Funnel: two-stage composition ──
        self.frontend = FrontendEncoder(
            sensory_dim=sensory_dim,
            hidden_dim=hidden_dim,
            sensory_noise_std=sensory_noise_std,
            dendritic_tau=lif_dendritic_tau,
            dt_ms=dt_ms,
            activation=activation,
        )
        self.backend = BioDecisionCore(
            hidden_dim=hidden_dim,
            mcmc_dim=mcmc_dim,
            num_gru_layers=num_gru_layers,
            dropout=dropout,
            lif_alpha=lif_alpha,
            lif_threshold=lif_threshold,
            lif_beta=lif_beta,
            lif_abs_refract_ms=lif_abs_refract_ms,
            lif_rel_refract_ms=lif_rel_refract_ms,
            lif_tau_syn=lif_tau_syn,
            lif_v_rest=lif_v_rest,
            lif_v_reset=lif_v_reset,
            lif_tau_w=lif_tau_w,
            lif_b_adapt=lif_b_adapt,
            lif_tau_fac=lif_tau_fac,
            lif_tau_rec=lif_tau_rec,
            lif_U_stp_init=lif_U_stp_init,
            lif_lateral_inhibition=lif_lateral_inhibition,
            lif_inhib_tau_ms=lif_inhib_tau_ms,
            gru_neuromod_gain=gru_neuromod_gain,
            lif_tbptt_steps=lif_tbptt_steps,
            dt_ms=dt_ms,
            activation=activation,
            refinement_mode=refinement_mode,
            refinement_max_steps=refinement_max_steps,
            refinement_eps=refinement_eps,
            refinement_update_scale=refinement_update_scale,
        )

        # ── Backward-compatible attribute aliases ──
        self.sensory_encoder = self.frontend.sensory_encoder
        self.lif_cell = self.backend.lif_cell
        self.gru_unit = self.backend.gru_unit
        self.router = self.backend.router
        self.direction_head = self.backend.direction_head

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
        Forward pass (Hybrid Funnel).

        Args:
            X_batch: ``(B, T, 8)`` — padded feature tensor.
            lengths: ``(B,)`` — true (unpadded) sequence lengths.
            return_internals: If ``True``, return internals dict.
            override_gates: Optional dict for in-silico lesioning.
            states: Optional dict of recurrent states for autoregressive mode.

        Returns:
            Same as original ``NSMoRCore.forward``.
        """
        # ── Root G3 / finding B4: validate the ORIGINAL observation dtype at
        # this public boundary BEFORE any conversion (integer/boolean/complex
        # observations are refused rather than silently cast).  Done before
        # ``.contiguous()``/indexing so the check sees the caller's dtype.
        _require_real_floating_observation(
            X_batch, name="X_batch", context="NSMoRCore",
        )
        X_batch = X_batch.contiguous()
        lengths = lengths.contiguous()

        expected_dim = self.sensory_dim + self.mcmc_dim
        if X_batch.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected feature dim {expected_dim}, got {X_batch.shape[-1]}"
            )
        B, T, _ = X_batch.shape
        if T == 0:
            raise ValueError(
                "NSMoRCore.forward requires T >= 1; a zero-time tensor (T=0) "
                "is unsupported (distinct from lengths==0, a valid no-op)."
            )
        lengths = _validate_original_lengths(
            lengths, B, T, context="NSMoRCore",
        )

        if self.persistence_skip != 0.0:
            if self.sensory_dim < 3:
                raise ValueError("Nonzero persistence_skip requires sensory_dim >= 3")
            target_mean = getattr(self, "target_mean", 0.0)
            target_std = getattr(self, "target_std", 1.0)
            target_clip = getattr(self, "target_clip_cm_s", 0.0)
            backend_mean = getattr(self.backend, "target_mean", 0.0)
            backend_std = getattr(self.backend, "target_std", 1.0)
            backend_clip = getattr(self.backend, "target_clip_cm_s", 0.0)
            if (
                target_mean != 0.0
                or target_std != 1.0
                or target_clip != 0.0
                or backend_mean != 0.0
                or backend_std != 1.0
                or backend_clip != 0.0
            ):
                raise ValueError(
                    "persistence_skip requires physical unnormalized/unclipped "
                    f"mode (got k={self.persistence_skip}), with restored "
                    f"target_mean={(target_mean, backend_mean)}, "
                    f"target_std={(target_std, backend_std)}, "
                    f"target_clip_cm_s={(target_clip, backend_clip)}."
                )

        # ── Unpack input ──
        sensory_x = X_batch[:, :, :self.sensory_dim]
        mcmc_prior = X_batch[:, :, self.sensory_dim:]

        # ── Frontend dendritic state handling ────────────────────
        # Round-1 BLOCKER-2 item 3: the module-level
        # ``FrontendEncoder._dendritic_state`` persisted across forward
        # calls, so batch N's end-of-sequence filter state leaked into
        # batch N+1's first frames whenever batch sizes matched (the
        # shape check inside FrontendEncoder.forward only re-initialises
        # on mismatch).  Set it explicitly EVERY call: to the restored
        # value in autoregressive mode, otherwise to None so the IIR
        # starts from zero at each sequence.
        if self.frontend._dendritic_enabled:
            dend_in = (
                states.get("frontend_dendritic_state", None)
                if states is not None else None
            )
            self.frontend._dendritic_state = dend_in

        # ── Stage 1: Frontend encoding ──
        e_sensory = self.frontend(sensory_x, lengths)

        # ── Stage 2: Bio-decision core ──
        # NOTE: No explicit .detach() here.  Gradient isolation between
        # the two stages is achieved via `requires_grad` toggling in the
        # training script:
        #
        #   Phase 1 (train frontend, freeze backend):
        #       backend params have requires_grad=False → backward through
        #       backend operations still reaches e_sensory, but backend
        #       param .grad stays None → only frontend receives updates.
        #
        #   Phase 2 (freeze frontend, train backend):
        #       frontend params have requires_grad=False → e_sensory has
        #       no grad_fn → bio-loss gradients cannot reach frontend.
        #
        #   Single-phase (all trainable):
        #       gradients flow freely through both stages.
        #
        # An explicit .detach() would break Phase 1 by severing the
        # gradient path before it reaches the trainable frontend.
        # Fixed observed-history residual; the recurrent path is unchanged.
        def _apply_skip(y: torch.Tensor) -> torch.Tensor:
            if self.persistence_skip != 0.0:
                v_lag = X_batch[:, :, 2]
                assert v_lag.shape == (B, T), (
                    f"Expected lag ({B}, {T}), got {v_lag.shape}"
                )
                t_idx = torch.arange(T, device=X_batch.device).unsqueeze(0)
                mask = t_idx < lengths.to(
                    device=X_batch.device, dtype=torch.int64,
                ).unsqueeze(1)
                assert mask.shape == (B, T), (
                    f"Expected mask ({B}, {T}), got {mask.shape}"
                )
                y = y + self.persistence_skip * torch.where(mask, v_lag, 0.0)
            assert y.shape == (B, T), f"y shape {tuple(y.shape)} != (B={B}, T={T})"
            return y

        if states is not None:
            y_pred, internals, states_out = self.backend(
                e_sensory, mcmc_prior, lengths,
                return_internals=True,
                override_gates=override_gates,
                states=states,
            )
            y_pred = _apply_skip(y_pred)
            # Include frontend dendritic state in states_out so
            # autoregressive checkpoint/restore preserves it.
            if self.frontend._dendritic_enabled:
                dend = getattr(self.frontend, '_dendritic_state', None)
                if dend is not None:
                    states_out["frontend_dendritic_state"] = dend.contiguous()
            return y_pred, internals, states_out

        if return_internals:
            y_pred, internals = self.backend(
                e_sensory, mcmc_prior, lengths,
                return_internals=True,
                override_gates=override_gates,
            )
            y_pred = _apply_skip(y_pred)
            return y_pred, internals

        y_pred = self.backend(
            e_sensory, mcmc_prior, lengths,
            override_gates=override_gates,
        )
        return _apply_skip(y_pred)

    def freeze_modules(self, module_names: List[str]) -> None:
        """
        Freeze parameters of the specified sub-modules.

        Supports both old-style names (``sensory_encoder``, ``lif_cell``,
        etc.) which are delegated to the appropriate sub-module.

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
            # Route to the correct sub-module
            if name == "sensory_encoder":
                submodule = self.frontend.sensory_encoder
            else:
                submodule = getattr(self.backend, name)
            if submodule is None:
                raise ValueError(
                    f"Cannot freeze '{name}': the module is absent "
                    "(refinement is only built when refinement_mode != 'off')."
                )
            for param in submodule.parameters():
                param.requires_grad = False


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
        lif_abs_refract_ms=2.0,
        lif_rel_refract_ms=50.0,
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
        lif_abs_refract_ms=2.0,
        lif_rel_refract_ms=50.0,
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
