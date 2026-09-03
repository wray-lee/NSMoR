"""Duck-typed JAX eval wrapper for analysis scripts (opt-in).

Measured on RTX 5060 Ti + WSL2 (2026-09-03): full-seq forward B=8 T=2400
is ~2.5x vs PyTorch.  That speedup is **not** a default: Heaviside
``v > v_th`` is chaotic under fp32 drift, so a 0.05% spike-decision
disagreement moves ``argmax(|y|)`` on ~half of real trials (latency
~20% relative).  Mean MSE / mean gates stay within 0.01%; publication
kinematics (peak, latency, LIF PCA) do not.

``states=`` is unsupported (Flax ``NSMoRModel`` has no carry API).
Fingerprint JAX is ~50x slower — not used here.

Scripts keep calling ``model(X, lengths, return_internals=...,
override_gates=...)`` and receive torch tensors on the input device.
Default analysis remains PyTorch; pass ``--backend jax`` only when the
reported statistic is a mean, not a spike-timed extremum.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from nsmor.model_nsmor_core import NSMoRCore

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from nsmor.jax.model import NSMoRModel, load_from_torch_state_dict

    JAX_AVAILABLE = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    NSMoRModel = None  # type: ignore[misc, assignment]
    load_from_torch_state_dict = None  # type: ignore[misc, assignment]
    JAX_AVAILABLE = False


def _to_torch(
    array: Any,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    np_arr = np.asarray(array)
    tensor = torch.from_numpy(np.array(np_arr, copy=True))
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor.to(device)


class JAXEvalWrapper:
    """Flax ``NSMoRModel.apply`` behind the PyTorch analysis call signature."""

    def __init__(
        self,
        jax_model: Any,
        params: Any,
        device: torch.device,
        hidden_dim: int,
        dt_ms: float,
        sensory_dim: int,
        mcmc_dim: int,
    ) -> None:
        self._jax_model = jax_model
        self._params = params
        self.device = device
        self.hidden_dim = hidden_dim
        self.dt_ms = dt_ms
        self.sensory_dim = sensory_dim
        self.mcmc_dim = mcmc_dim
        self._compiled: Dict[Any, Any] = {}

    @classmethod
    def from_torch(cls, model: NSMoRCore, device: Optional[torch.device] = None) -> JAXEvalWrapper:
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not installed")

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        lif = model.lif_cell
        backend = model.backend
        inhib_tau = (
            float(getattr(lif, "_inhib_tau_ms", 50.0))
            if lif.lateral_inhibition > 0.0
            else 50.0
        )
        jax_model = NSMoRModel(
            sensory_dim=model.sensory_dim,
            mcmc_dim=model.mcmc_dim,
            hidden_dim=model.hidden_dim,
            dt_ms=float(model.dt_ms),
            lif_alpha=float(lif.alpha),
            lif_threshold=float(lif.v_threshold),
            lif_beta=float(lif.beta),
            lif_tau_syn=float(lif.tau_syn),
            lif_tau_w=float(lif.tau_w),
            lif_b_adapt=float(lif.b_adapt),
            lif_lateral_inhibition=float(lif.lateral_inhibition),
            lif_inhib_tau_ms=inhib_tau,
            lif_rel_refract_ms=float(lif.rel_refract_ms),
            lif_abs_refract_ms=float(lif.abs_refract_ms),
            lif_v_rest=float(lif.v_rest),
            lif_tbptt_steps=int(backend._tbptt_steps),
            gru_neuromod_gain=float(backend.gru_neuromod_gain),
            # Eval is always deterministic=True, so dropout is inert; copied
            # anyway so the wrapper stays faithful if that ever changes.
            dropout_rate=float(getattr(model.direction_head.net[2], "p", 0.1)),
            sensory_noise_std=float(model.sensory_encoder.noise_std),
        )
        params = load_from_torch_state_dict(jax_model, model.state_dict())
        return cls(
            jax_model=jax_model,
            params=params,
            device=device,
            hidden_dim=model.hidden_dim,
            dt_ms=float(model.dt_ms),
            sensory_dim=model.sensory_dim,
            mcmc_dim=model.mcmc_dim,
        )

    def eval(self) -> JAXEvalWrapper:
        return self

    def parameters(self, recurse: bool = True):  # noqa: ARG002
        yield torch.empty(0, device=self.device)

    def _apply_fn(self, return_internals: bool, override_gates: Optional[Dict[str, float]]):
        key = (
            return_internals,
            tuple(sorted((override_gates or {}).items())),
        )
        cached = self._compiled.get(key)
        if cached is not None:
            return cached

        ov = dict(override_gates) if override_gates else None
        jax_model = self._jax_model

        if return_internals:
            fn = jax.jit(
                lambda p, xx, ll: jax_model.apply(
                    p,
                    xx,
                    ll,
                    deterministic=True,
                    return_internals=True,
                    override_gates=ov,
                )
            )
        else:
            fn = jax.jit(
                lambda p, xx, ll: jax_model.apply(
                    p,
                    xx,
                    ll,
                    deterministic=True,
                    override_gates=ov,
                )
            )
        self._compiled[key] = fn
        return fn

    def __call__(
        self,
        X_batch: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_internals: bool = False,
        override_gates: Optional[Dict[str, float]] = None,
        states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if states is not None:
            raise TypeError(
                "JAXEvalWrapper has no states= carry; use --backend torch "
                "for autoregressive rollout"
            )

        device = X_batch.device
        x_j = jnp.asarray(X_batch.detach().cpu().numpy())
        l_j = jnp.asarray(lengths.detach().cpu().numpy().astype(np.int32))
        apply_fn = self._apply_fn(return_internals, override_gates)
        out = apply_fn(self._params, x_j, l_j)

        if return_internals:
            y_j, internals_j = out
            y_j.block_until_ready()
            y = _to_torch(y_j, device=device, dtype=torch.float32)
            internals: Dict[str, torch.Tensor] = {}
            for key, value in internals_j.items():
                arr = np.asarray(value)
                if np.iscomplexobj(arr):
                    internals[key] = _to_torch(arr, device=device)
                else:
                    internals[key] = _to_torch(arr, device=device, dtype=torch.float32)
            return y, internals

        out.block_until_ready()
        return _to_torch(out, device=device, dtype=torch.float32)


def wrap_eval_model(
    model: NSMoRCore,
    backend: str = "jax",
    device: Optional[torch.device] = None,
) -> Union[NSMoRCore, JAXEvalWrapper]:
    """Return a JAX duck-typed wrapper, or *model* unchanged.

    ``backend='jax'`` falls back to PyTorch if JAX is missing.
    """
    if backend == "torch":
        return model
    if backend != "jax":
        raise ValueError(f"backend must be 'jax' or 'torch', got {backend!r}")
    if not JAX_AVAILABLE:
        logger.warning("JAX not installed; analysis forward stays on PyTorch")
        return model
    logger.info("Analysis forward: JAX (measured ~2x vs PyTorch on T=2400)")
    return JAXEvalWrapper.from_torch(model, device=device)
