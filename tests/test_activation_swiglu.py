"""Focused tests for the opt-in SwiGLU activation and its contracts.

Covers:
  - Default ``relu`` preserves the historical state_dict keys.
  - ``swiglu`` is a genuine gated activation ``SiLU(W_gate x) * (W_value x)``.
  - Config / constructor boundary validation rejects unknown activations.
  - Strict state_dict round-trip for both activations.
  - Finite, non-zero encoder/decoder gradients under ``swiglu``.
  - Padding invariance and causal future-perturbation isolation.
  - Absolute torch<->JAX parity (skips if JAX is absent).

Ref: Shazeer 2020, "GLU Variants Improve Transformer" (gated activation).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from nsmor.config_parser import ModelConfig
from nsmor.model_nsmor_core import DirectionHead, NSMoRCore, SensoryEncoder

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Runs in a fresh interpreter: the platform env vars are set BEFORE the first
# ``jax`` import, so the parity bound is measured on CPU.  Prints one JSON line.
_CPU_PARITY_SCRIPT = r'''
import json, os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import torch
import jax
from nsmor.model_nsmor_core import NSMoRCore
from nsmor.model_nsmor_core_jax import NSMoRCoreJAX
from nsmor.analysis.jax_eval import JAXEvalWrapper

activation = sys.argv[1]
torch.manual_seed(0)
pt = NSMoRCore(hidden_dim=16, dt_ms=4.0, activation=activation,
               dropout=0.0).eval()
x = torch.randn(2, 10, 8)
lengths = torch.tensor([10, 6], dtype=torch.int64)
mask = torch.arange(10).unsqueeze(0) < lengths.unsqueeze(1)
with torch.no_grad():
    y_pt, i_pt = pt(x, lengths, return_internals=True)

raw = NSMoRCoreJAX.from_torch(pt)
with torch.no_grad():
    y_raw, i_raw = raw(x, lengths, return_internals=True)
wrap = JAXEvalWrapper.from_torch(pt, device=torch.device("cpu"))
with torch.no_grad():
    y_fl, i_fl = wrap(x, lengths, return_internals=True)

def _t(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)

res = {
    "platform": jax.default_backend(),
    "raw_y": float((y_pt[mask] - _t(y_raw)[mask]).abs().max()),
    "fl_y": float((y_pt[mask] - _t(y_fl)[mask]).abs().max()),
    "raw_gate": float(
        (i_pt["routing_gates"][mask] - _t(i_raw["routing_gates"])[mask]).abs().max()
    ),
    "fl_gate": float(
        (i_pt["routing_gates"][mask] - _t(i_fl["routing_gates"])[mask]).abs().max()
    ),
}
print("RESULT:" + json.dumps(res))
'''


def _model(activation: str, hidden_dim: int = 16, **kwargs) -> NSMoRCore:
    """Build a small deterministic NSMoRCore for the given activation."""
    torch.manual_seed(0)
    return NSMoRCore(
        hidden_dim=hidden_dim, dt_ms=4.0, activation=activation, **kwargs
    )


class TestActivationContract:
    """State_dict key contract and genuine-gating checks."""

    def test_default_relu_keys_unchanged(self) -> None:
        keys = set(_model("relu").state_dict().keys())
        assert "sensory_encoder.net.0.weight" in keys
        assert "direction_head.net.3.weight" in keys
        assert "sensory_encoder.gate_proj.weight" not in keys
        assert "direction_head.gate_proj.weight" not in keys

    def test_swiglu_adds_new_keys_only(self) -> None:
        keys = set(_model("swiglu").state_dict().keys())
        assert "sensory_encoder.gate_proj.weight" in keys
        assert "direction_head.gate_proj.weight" in keys
        assert "direction_head.value_proj.weight" in keys
        assert "direction_head.out_proj.weight" in keys
        # The ReLU / readout Sequential slots are absent under swiglu.
        assert "sensory_encoder.net.2.weight" not in keys
        assert "direction_head.net.3.weight" not in keys

    def test_swiglu_is_genuinely_gated(self) -> None:
        """Exercise the REAL ``DirectionHead.forward`` (not a re-derived
        formula): it must equal the gated reference and DIFFER from both the
        additive variant and the ungated value-only variant.  A forward made
        additive or ungated would otherwise pass unnoticed (finding A14/B15).
        """
        torch.manual_seed(0)
        head = DirectionHead(hidden_dim=16, dropout=0.0, activation="swiglu").eval()
        assert head.gate_proj.weight.data_ptr() != head.value_proj.weight.data_ptr()
        h = torch.randn(1, 5, 16)
        with torch.no_grad():
            y = head(h)  # the real forward path under test
            n = head.net(h)
            gated = head.out_proj(
                torch.nn.functional.silu(head.gate_proj(n)) * head.value_proj(n)
            ).squeeze(-1)
            additive = head.out_proj(
                torch.nn.functional.silu(head.gate_proj(n)) + head.value_proj(n)
            ).squeeze(-1)
            ungated = head.out_proj(
                torch.nn.functional.silu(head.value_proj(n))
            ).squeeze(-1)
        # The executed forward is the genuine gated product.
        assert torch.allclose(y, gated, atol=1e-6), "forward != gated reference"
        # And it is NOT the additive or the ungated value-only variant.
        assert not torch.allclose(y, additive, atol=1e-4), "forward is additive"
        assert not torch.allclose(y, ungated, atol=1e-4), "forward is ungated"

    def test_config_and_constructor_validation(self) -> None:
        assert ModelConfig().activation == "relu"
        ModelConfig(activation="swiglu")
        for bad in ("silu", "gelu", ""):
            with pytest.raises(ValueError):
                ModelConfig(activation=bad)
        with pytest.raises(ValueError):
            NSMoRCore(hidden_dim=8, activation="nope")
        with pytest.raises(ValueError):
            SensoryEncoder(4, 8, 0.0, activation="nope")
        with pytest.raises(ValueError):
            DirectionHead(8, 0.1, activation="nope")


class TestActivationRuntime:
    """Forward, gradient, round-trip, masking and causality behavior."""

    @pytest.mark.parametrize("activation", ["relu", "swiglu"])
    def test_forward_shapes_and_finite_grads(self, activation: str) -> None:
        model = _model(activation).train()
        x = torch.randn(2, 10, 8, requires_grad=True)
        lengths = torch.tensor([10, 6], dtype=torch.int64)
        y, internals = model(x, lengths, return_internals=True)
        assert y.shape == (2, 10)
        assert internals["routing_gates"].shape == (2, 10, 2)
        assert internals["lif_spikes"].shape == (2, 10, 16)
        y.sum().backward()
        for module in (model.sensory_encoder, model.direction_head):
            for name, param in module.named_parameters():
                assert param.grad is not None, (activation, name)
                assert torch.isfinite(param.grad).all(), (activation, name)
        if activation == "swiglu":
            assert model.sensory_encoder.gate_proj.weight.grad.abs().sum() > 0
            assert model.direction_head.gate_proj.weight.grad.abs().sum() > 0
            assert model.direction_head.value_proj.weight.grad.abs().sum() > 0

    @pytest.mark.parametrize("activation", ["relu", "swiglu"])
    def test_strict_state_dict_roundtrip(self, activation: str) -> None:
        model = _model(activation).eval()
        clone = _model(activation).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        x = torch.randn(2, 8, 8)
        lengths = torch.tensor([8, 5], dtype=torch.int64)
        with torch.no_grad():
            assert torch.equal(model(x, lengths), clone(x, lengths))

    @pytest.mark.parametrize("activation", ["relu", "swiglu"])
    def test_padding_and_causal_isolation(self, activation: str) -> None:
        model = _model(activation, dropout=0.0).eval()
        x = torch.randn(2, 12, 8)
        lengths = torch.tensor([12, 6], dtype=torch.int64)
        with torch.no_grad():
            y1 = model(x, lengths)
            padded = x.clone()
            padded[1, 6:, :] += 10.0
            y2 = model(padded, lengths)
            future = x.clone()
            future[0, 9, :] += 5.0
            y3 = model(future, lengths)
        assert torch.equal(y1[1, :6], y2[1, :6]), "padded region leaked"
        assert torch.equal(y1[0, :9], y3[0, :9]), "non-causal future leak"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestActivationJAXParity:
    """Absolute torch<->JAX parity across supported routes.

    The raw/Flax parity check runs in a SUBPROCESS with ``JAX_PLATFORMS=cpu``
    pinned BEFORE the first ``jax`` import, so the agreement bound is genuinely
    measured on CPU (the module-level ``jax`` import here already initialized a
    GPU backend; an in-process fixture cannot re-pin it).  Both the masked
    prediction ``y`` and the routing gates are compared with an ABSOLUTE bound
    at the CPU agreement level (~1e-7), so a swapped decoder gate/value leaf --
    which perturbs ``y`` by ~0.06 while leaving the gates untouched -- is
    caught (finding B8).
    """

    @pytest.mark.parametrize("activation", ["relu", "swiglu"])
    def test_raw_jax_and_flax_eval_parity(self, activation: str) -> None:
        r = subprocess.run(
            [sys.executable, "-c", _CPU_PARITY_SCRIPT, activation, "1e-6"],
            cwd=str(_REPO_ROOT),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=600,
        )
        assert r.returncode == 0, r.stderr[-3000:]
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT:")]
        assert lines, f"parity subprocess produced no result: {r.stdout[-1000:]}"
        payload = json.loads(lines[-1][len("RESULT:"):])
        assert payload["platform"] == "cpu", payload
        bound = 1e-6
        assert payload["raw_y"] < bound, payload
        assert payload["fl_y"] < bound, payload
        assert payload["raw_gate"] < bound, payload
        assert payload["fl_gate"] < bound, payload

    @pytest.mark.parametrize("activation", ["relu", "swiglu"])
    def test_decoder_module_parity(self, activation: str) -> None:
        import flax.linen as fnn  # noqa: F401  (ensures flax importable)

        from nsmor.jax.model import DirectionHeadJAX

        torch.manual_seed(3)
        torch_dh = DirectionHead(hidden_dim=16, dropout=0.0, activation=activation).eval()
        flax_dh = DirectionHeadJAX(hidden_dim=16, dropout_rate=0.0, activation=activation)
        h = np.random.RandomState(5).randn(2, 7, 16).astype(np.float32)
        params = flax_dh.init(jax.random.PRNGKey(0), jnp.asarray(h), deterministic=True)
        with torch.no_grad():
            params["params"]["ln"] = {
                "scale": jnp.asarray(torch_dh.net[0].weight.numpy()),
                "bias": jnp.asarray(torch_dh.net[0].bias.numpy()),
            }
            if activation == "swiglu":
                params["params"]["gate"] = {
                    "kernel": jnp.asarray(torch_dh.gate_proj.weight.numpy().T),
                    "bias": jnp.asarray(torch_dh.gate_proj.bias.numpy()),
                }
                params["params"]["value"] = {
                    "kernel": jnp.asarray(torch_dh.value_proj.weight.numpy().T),
                    "bias": jnp.asarray(torch_dh.value_proj.bias.numpy()),
                }
                params["params"]["out"] = {
                    "kernel": jnp.asarray(torch_dh.out_proj.weight.numpy().T),
                    "bias": jnp.asarray(torch_dh.out_proj.bias.numpy()),
                }
            else:
                params["params"]["dense"] = {
                    "kernel": jnp.asarray(torch_dh.net[3].weight.numpy().T),
                    "bias": jnp.asarray(torch_dh.net[3].bias.numpy()),
                }
        y_flax = np.asarray(flax_dh.apply(params, jnp.asarray(h), deterministic=True))
        with torch.no_grad():
            y_torch = torch_dh(torch.from_numpy(h.copy())).numpy()
        d = np.abs(y_torch - y_flax).max()
        scale = max(float(np.abs(y_torch).max()), 1e-6)
        assert d < 5e-4 and d < 1e-3 * scale
