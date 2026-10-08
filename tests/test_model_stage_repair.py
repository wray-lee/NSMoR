"""Focused failure regressions for the model-stage correction phase (IDs 1-10).

Each test encodes a concrete reproduction from the two independent reviewer
reports so the specific root cause cannot silently regress.  These are
synthetic, deterministic, in-memory checks; they do not read formal data,
checkpoints, or nested artifacts.

Mapping (reviewer B finding -> regression):

* B1 (ID 1): SwiGLU dropout placement parity (Torch drops the normalized
  input before both projections; Flax must too).
* B2 (ID 2): Flax LayerNorm epsilon / centered-variance parity on
  near-constant inputs.
* B4 (ID 3): padding freezes EVERY biological / cache state, not just outputs.
* B3 (ID 4): raw all-layer GRU ``h_n`` carry before output gain.
* B5 (ID 5): no silent backend mechanism / configuration loss (fail closed).
* (ID 6): absolute + relative refractory physical-time discretization parity.
* (ID 7): dendritic filter touches the visual channel (index 0) only.
* B6 (ID 8): raw-JAX spike-history + dendritic carry across stateful calls.
* B7 (ID 9): sensory noise vs MC-dropout epistemic attribution.
* B8 (ID 10): legacy third-velocity-difference proxy truth / units, with the
  historical ``lambda_jerk`` numerics immutable.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nsmor.loss import BioJointLoss
from nsmor.model_nsmor_core import DirectionHead, LIFCell, NSMoRCore

# NOTE: every value asserted below is the *current, corrected* behavior; the
# comments name the historical defect the test guards against, not a value we
# tuned to pass.

try:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def _flatten_leaves(tree: Any, path: str = "") -> Dict[str, Any]:
    """Flatten a nested dict/list param tree to ``{dotted.path: leaf}``.

    Used to compare EVERY destination parameter leaf (and its aliases) after a
    roundtrip, not just a hand-picked subset.
    """
    out: Dict[str, Any] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            child = f"{path}.{key}" if path else str(key)
            out.update(_flatten_leaves(value, child))
    elif isinstance(tree, (list, tuple)):
        for idx, value in enumerate(tree):
            out.update(_flatten_leaves(value, f"{path}[{idx}]"))
    else:
        out[path] = tree
    return out


def _assert_all_leaves_equal(expected: Any, actual: Any) -> None:
    """Assert EVERY leaf of ``actual`` exactly equals the matching leaf of
    ``expected`` (same key set).  Corrupting any single exported leaf fails."""
    exp = _flatten_leaves(expected)
    act = _flatten_leaves(actual)
    assert set(exp) == set(act), (
        f"leaf key sets differ: only-expected={sorted(set(exp) - set(act))}, "
        f"only-actual={sorted(set(act) - set(exp))}"
    )
    for key in sorted(exp):
        assert np.array_equal(np.asarray(exp[key]), np.asarray(act[key])), (
            f"leaf {key!r} not preserved by the roundtrip"
        )


# ===============================================================
# ID 1 — SwiGLU dropout placement parity
# ===============================================================

class TestSwiGLUDropoutPlacement:
    """The normalized input is dropped BEFORE both gate/value projections."""

    def test_torch_drops_normalized_input_before_projections(self) -> None:
        torch.manual_seed(0)
        head = DirectionHead(hidden_dim=8, dropout=0.5, activation="swiglu").train()
        h = torch.randn(3, 5, 8)
        torch.manual_seed(11)
        y = head(h)
        torch.manual_seed(11)
        n = head.net(h)  # LayerNorm -> Dropout (normalized input)
        ref = head.out_proj(
            F.silu(head.gate_proj(n)) * head.value_proj(n)
        ).squeeze(-1)
        assert torch.equal(y, ref), "Torch dropped the wrong tensor"

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
    def test_flax_drops_normalized_input_before_projections(self) -> None:
        from nsmor.jax.model import DirectionHeadJAX

        h = np.random.RandomState(0).randn(2, 5, 8).astype(np.float32)
        dh = DirectionHeadJAX(hidden_dim=8, dropout_rate=0.5, activation="swiglu")
        key = jax.random.PRNGKey(3)
        params = dh.init(key, jnp.asarray(h), deterministic=True)
        y = dh.apply(
            params, jnp.asarray(h), deterministic=False, rngs={"dropout": key},
        )

        class _RefNorm(nn.Module):
            hidden_dim: int = 8

            @nn.compact
            def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
                n = nn.LayerNorm(
                    name="ln", epsilon=1e-5, use_fast_variance=False,
                )(x)
                if not deterministic:
                    n = nn.Dropout(0.5, deterministic=False)(n)
                g = nn.Dense(self.hidden_dim, name="gate")(n)
                v = nn.Dense(self.hidden_dim, name="value")(n)
                return jnp.squeeze(
                    nn.Dense(1, name="out")(jax.nn.silu(g) * v), -1,
                )

        class _RefProduct(nn.Module):
            hidden_dim: int = 8

            @nn.compact
            def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
                n = nn.LayerNorm(
                    name="ln", epsilon=1e-5, use_fast_variance=False,
                )(x)
                g = nn.Dense(self.hidden_dim, name="gate")(n)
                v = nn.Dense(self.hidden_dim, name="value")(n)
                a = jax.nn.silu(g) * v
                if not deterministic:
                    a = nn.Dropout(0.5, deterministic=False)(a)
                return jnp.squeeze(nn.Dense(1, name="out")(a), -1)

        # Reuse the exact DirectionHeadJAX parameter tree in both reference
        # modules so the only difference is dropout placement.
        y_norm = _RefNorm().apply(
            params, jnp.asarray(h), deterministic=False, rngs={"dropout": key},
        )
        y_prod = _RefProduct().apply(
            params, jnp.asarray(h), deterministic=False, rngs={"dropout": key},
        )
        assert bool(jnp.allclose(y, y_norm, atol=1e-5)), (
            "Flax must drop the normalized input, not the gated product"
        )
        assert not bool(jnp.allclose(y, y_prod, atol=1e-5)), (
            "dropout placement is not distinguishable -> test is inert"
        )


# ===============================================================
# ID 2 — Low-variance LayerNorm parity
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestLayerNormLowVarianceParity:
    """Flax LayerNorm must match Torch (eps=1e-5, centered variance)."""

    def test_low_variance_decoder_matches_torch(self) -> None:
        import jax.numpy as jnp
        from nsmor.jax.model import DirectionHeadJAX

        torch.manual_seed(3)
        dh = DirectionHead(hidden_dim=2, dropout=0.0, activation="swiglu").eval()
        flax_dh = DirectionHeadJAX(hidden_dim=2, dropout_rate=0.0, activation="swiglu")
        # Two regimes, each isolating one half of the claim (finding B17):
        #   row 0 [1, 1.0001]         -> near-zero variance, eps dominates.
        #   row 1 [1000, 1000.01]     -> large mean, small variance: this is
        #     where the naive "fast" variance (E[x^2]-E[x]^2) catastrophically
        #     cancels and diverges, so `use_fast_variance=True` is detected.
        h = np.array(
            [[1.0, 1.0001], [1000.0, 1000.01]], dtype=np.float32,
        )[None, :, :]  # (1,2,2)
        params = flax_dh.init(jax.random.PRNGKey(0), jnp.asarray(h), deterministic=True)
        params["params"]["ln"] = {
            "scale": jnp.asarray(dh.net[0].weight.detach().numpy()),
            "bias": jnp.asarray(dh.net[0].bias.detach().numpy()),
        }
        params["params"]["gate"] = {
            "kernel": jnp.asarray(dh.gate_proj.weight.detach().numpy().T),
            "bias": jnp.asarray(dh.gate_proj.bias.detach().numpy()),
        }
        params["params"]["value"] = {
            "kernel": jnp.asarray(dh.value_proj.weight.detach().numpy().T),
            "bias": jnp.asarray(dh.value_proj.bias.detach().numpy()),
        }
        params["params"]["out"] = {
            "kernel": jnp.asarray(dh.out_proj.weight.detach().numpy().T),
            "bias": jnp.asarray(dh.out_proj.bias.detach().numpy()),
        }
        y_flax = np.asarray(flax_dh.apply(params, jnp.asarray(h), deterministic=True))
        with torch.no_grad():
            y_torch = dh(torch.from_numpy(h.copy())).numpy()
        assert np.abs(y_torch - y_flax).max() < 1e-5, (
            f"low-variance LayerNorm parity broke: "
            f"torch={y_torch.ravel()} flax={y_flax.ravel()}"
        )


# ===============================================================
# ID 3 — Padding freezes every biological / cache state
# ===============================================================

class TestPaddingFreezesAllState:
    """A padded frame must not advance ANY per-sample recurrent state."""

    @staticmethod
    def _deterministic_model() -> NSMoRCore:
        torch.manual_seed(0)
        m = NSMoRCore(
            sensory_dim=4, mcmc_dim=4, hidden_dim=4, dt_ms=4.0,
            lif_alpha=0.9, lif_threshold=0.5, lif_beta=1.0,
            lif_rel_refract_ms=20.0, lif_abs_refract_ms=1.0,
            lif_lateral_inhibition=0.5, lif_dendritic_tau=5.0,
            dropout=0.0,
        ).eval()
        with torch.no_grad():
            m.lif_cell.W_in.weight.zero_()
            m.lif_cell.W_in.bias.copy_(torch.tensor([0.6, 0.1, 0.0, 0.0]))
        return m

    def test_all_exported_states_are_pad_invariant(self) -> None:
        m = self._deterministic_model()
        x1 = torch.zeros(1, 1, 8)
        x1[0, 0, 0] = 0.6
        x3 = torch.zeros(1, 3, 8)
        x3[0, 0, 0] = 0.6
        x3[0, 1:, :] = 10.0  # garbage in the padded suffix
        with torch.no_grad():
            _, _, s1 = m(x1, torch.tensor([1]), return_internals=True, states={})
            _, _, s3 = m(x3, torch.tensor([1]), return_internals=True, states={})
        assert set(s1.keys()) == set(s3.keys())
        for key in s1:
            assert torch.equal(s1[key], s3[key]), (
                f"padded frame advanced biological state {key!r}: "
                f"{s1[key]} vs {s3[key]}"
            )


# ===============================================================
# ID 4 — Raw all-layer GRU h_n carry (pre-gain, padded endpoint)
# ===============================================================

class TestRawGRUCarry:
    """The exported carry is the packed-GRU h_n, pre-gain, unpadded."""

    def test_carry_is_endpoint_not_zero_for_short_sample(self) -> None:
        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
        x = torch.randn(1, 5, 8)
        with torch.no_grad():
            _, _, st = m(x, torch.tensor([2]), return_internals=True, states={})
        assert st["gru_h"].shape == (1, 1, 4)
        assert st["gru_h"].abs().sum() > 0, "padded sample exported a zero carry"

    def test_carry_excludes_neuromod_gain(self) -> None:
        torch.manual_seed(0)
        m = NSMoRCore(
            hidden_dim=4, dt_ms=4.0, dropout=0.0, gru_neuromod_gain=0.5,
        ).eval()
        x = torch.randn(1, 3, 8)
        with torch.no_grad():
            out, _, st = m(x, torch.tensor([3]), return_internals=True, states={})
            # The raw carry must equal a direct GRU h_n, not the gated
            # ``gru_hidden`` trajectory (which is gain-scaled).  Reproduce the
            # exact encoder output the backend feeds the GRU.
            e = m.frontend(x[:, :, :4], torch.tensor([3]))
            _, h_n = m.gru_unit(e.float(), torch.tensor([3]), return_hidden=True)
        assert torch.allclose(st["gru_h"], h_n, atol=1e-6)

    def test_stacked_layers_export_all_layers_and_resume(self) -> None:
        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                      num_gru_layers=2).eval()
        x = torch.randn(1, 4, 8)
        with torch.no_grad():
            _, _, st = m(x, torch.tensor([4]), return_internals=True, states={})
        assert st["gru_h"].shape == (2, 1, 4), (
            f"stacked carry shape {tuple(st['gru_h'].shape)} != (2, 1, 4)"
        )
        # Resuming from the (2, B, H) carry must not raise on shape.
        with torch.no_grad():
            m(x[:, :2], torch.tensor([2]), return_internals=True, states=st)


# ===============================================================
# ID 5 — Fail-closed backend support envelope
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestBackendSupportEnvelope:
    """Unsupported settings are rejected BEFORE any silent mapping."""

    def test_unsupported_settings_rejected(self) -> None:
        from nsmor.jax.model import assert_flax_supported

        for kwargs in (
            {"num_gru_layers": 2},
            {"lif_dendritic_tau": 5.0},
            {"lif_tau_fac": 10.0, "lif_tau_rec": 50.0},
            {"lif_v_reset": 0.0},
        ):
            m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0, **kwargs)
            with pytest.raises(ValueError):
                assert_flax_supported(m, context="test")

    def test_supported_settings_pass(self) -> None:
        from nsmor.jax.model import assert_flax_supported

        m = NSMoRCore(
            hidden_dim=4, dt_ms=4.0, dropout=0.0,
            lif_tau_syn=5.0, lif_tau_w=100.0, lif_b_adapt=0.5,
            lif_lateral_inhibition=0.1, lif_rel_refract_ms=20.0,
        )
        assert_flax_supported(m, context="test")  # must not raise

    def test_eval_wrapper_and_raw_jax_fail_closed(self) -> None:
        from nsmor.analysis.jax_eval import JAXEvalWrapper
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        stacked = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                            num_gru_layers=2)
        with pytest.raises(ValueError):
            JAXEvalWrapper.from_torch(stacked)
        with pytest.raises(ValueError):
            NSMoRCoreJAX.from_torch(stacked)

        dendritic = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                              lif_dendritic_tau=5.0)
        with pytest.raises(ValueError):
            JAXEvalWrapper.from_torch(dendritic)


# ===============================================================
# ID 6 — Refractory physical-time discretization (ceil)
# ===============================================================

class TestRefractoryPhysicalTimeDiscretization:
    """Step counts round UP so a sub-frame duration still has effect."""

    @pytest.mark.parametrize(
        "dt_ms, rel_ms, abs_ms, exp_rel, exp_abs",
        [
            (4.0, 20.0, 1.0, 5, 1),    # ceil(0.25)=1, not floor 0
            (4.0, 10.0, 0.0, 3, 0),    # ceil(2.5)=3, not round 2
            (10.0, 25.0, 5.0, 3, 1),   # ceil(2.5)=3
            (4.0, 0.0, 0.0, 0, 0),     # disabled
        ],
    )
    def test_ceil_conversion(
        self, dt_ms: float, rel_ms: float, abs_ms: float,
        exp_rel: int, exp_abs: int,
    ) -> None:
        c = LIFCell(
            hidden_dim=4, alpha=0.9, v_threshold=1.0, beta=1.0,
            rel_refract_ms=rel_ms, abs_refract_ms=abs_ms, dt_ms=dt_ms,
        )
        assert c.rel_refract_steps == exp_rel
        assert c.abs_refract_steps == exp_abs
        if exp_rel > 0:
            assert math.isclose(c._k_rel, 1.0 / exp_rel)


# ===============================================================
# ID 7 — Dendritic filter touches the visual channel only
# ===============================================================

class TestDendriticModalityIsolation:
    """Only channel 0 traverses the slow dendritic low-pass filter."""

    def test_only_channel_zero_is_filtered(self) -> None:
        torch.manual_seed(0)
        dend = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                         lif_dendritic_tau=5.0).eval()
        plain = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
        plain.load_state_dict(
            {k: v for k, v in dend.state_dict().items()
             if not k.startswith("frontend.")},
            strict=False,
        )
        # Share the sensory-encoder weights so the comparison isolates the
        # filter (which lives on the frontend, not in the shared sub-module).
        plain.frontend.sensory_encoder.load_state_dict(
            dend.frontend.sensory_encoder.state_dict()
        )
        L = torch.tensor([6])

        # Channel 0 all-zero: filter(0)=0, so the encoding is identical.
        x0 = torch.randn(1, 6, 4)
        x0[:, :, 0] = 0.0
        with torch.no_grad():
            e_dend = dend.frontend(x0, L)
            e_plain = plain.frontend(x0, L)
        assert torch.allclose(e_dend, e_plain, atol=1e-6), (
            "channel-0=0 must leave the dendritic branch inert"
        )

        # Channel 0 driven: the ramp differs from the raw signal.
        x1 = x0.clone()
        x1[:, :, 0] = 1.0
        with torch.no_grad():
            e_dend1 = dend.frontend(x1, L)
            e_plain1 = plain.frontend(x1, L)
        assert not torch.allclose(e_dend1, e_plain1, atol=1e-4), (
            "channel 0 must be filtered (dendritic branch is inert)"
        )


# ===============================================================
# ID 8 — Raw-JAX spike-history + dendritic carry across calls
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestRawJAXStatefulCarry:
    """Whole call == chunked stateful calls within the raw-JAX kernel."""

    def test_whole_equals_chunk(self) -> None:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        torch.manual_seed(0)
        m = NSMoRCore(
            hidden_dim=4, dt_ms=4.0, dropout=0.0,
            lif_lateral_inhibition=0.5, lif_dendritic_tau=5.0,
            lif_rel_refract_ms=20.0, lif_abs_refract_ms=1.0,
        ).eval()
        # Constant drive so lateral inhibition / spike history actually fire.
        x = torch.zeros(1, 8, 8)
        x[:, :, 0] = 0.6
        x[:, :, 4] = 1.0
        j = NSMoRCoreJAX.from_torch(m)
        with torch.no_grad():
            y_whole, _, s_whole = j(x, torch.tensor([8]), return_internals=True,
                                    states={})
            y1, _, s1 = j(x[:, :4], torch.tensor([4]), return_internals=True,
                          states={})
            y2, _, s2 = j(x[:, 4:], torch.tensor([4]), return_internals=True,
                          states=s1)
        y_chunk = torch.cat([y1, y2], dim=1)
        assert torch.allclose(y_whole, y_chunk, atol=1e-5), (
            "raw-JAX chunked continuation diverged from the whole call"
        )
        for key in ("lif_spike_history", "frontend_dendritic_state"):
            assert key in s_whole and key in s2
            assert torch.allclose(s_whole[key], s2[key], atol=1e-5), (
                f"carry {key!r} not preserved across chunked calls"
            )


# ===============================================================
# ID 9 — Sensory noise vs MC-dropout epistemic attribution
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestUQEpistemicAttribution:
    """Default UQ reports dropout-only dispersion (sensory noise excluded)."""

    def test_default_excludes_sensory_noise(self) -> None:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.analysis.uq_jax import MCDropoutAnalyzerJAX
        from nsmor.jax.model import NSMoRModel

        model = NSMoRModel(
            hidden_dim=4, dt_ms=4.0, dropout_rate=0.0,
            sensory_noise_std=0.1, lif_lateral_inhibition=0.0,
        )
        x = jnp.ones((1, 3, 8), dtype=jnp.float32)
        L = jnp.array([3], dtype=jnp.int32)
        params = model.init(jax.random.PRNGKey(0), x, L)

        dropout_only = MCDropoutAnalyzerJAX(
            model, params, n_samples=8, seed=0,
        ).predict(x, L)["y_std"]
        mixed = MCDropoutAnalyzerJAX(
            model, params, n_samples=8, seed=0, include_sensory_noise=True,
        ).predict(x, L)["y_std"]

        assert float(np.abs(dropout_only).max()) < 1e-6, (
            "dropout disabled -> dropout-only std must be ~0, "
            f"got {np.abs(dropout_only).max()}"
        )
        assert float(mixed.max()) > 1e-4, (
            "mixed estimator must expose the injected sensory noise"
        )


# ===============================================================
# ID 10 — Legacy third-difference smoothness proxy (immutable numerics)
# ===============================================================

class TestLambdaJerkLegacyProxy:
    """``lambda_jerk`` is a frame-based 3rd-difference proxy, not jerk."""

    def test_constant_second_difference_is_not_penalized(self) -> None:
        loss = BioJointLoss()
        y = torch.tensor([[0.0, 1.0, 4.0, 9.0]])  # constant 2nd difference
        g = torch.zeros(1, 4, 1)
        value = loss(y, y.clone(), torch.tensor([4]), g,
                     lambda_reg=0.0, lambda_jerk=1.0)
        assert value.item() == 0.0, (
            "a quadratic velocity has zero 3rd difference -> zero penalty"
        )

    def test_cubic_third_difference_numerics_unchanged(self) -> None:
        loss = BioJointLoss()
        y = torch.tensor([[0.0, 1.0, 8.0, 27.0]])  # 3rd difference == 6
        g = torch.zeros(1, 4, 1)
        value = loss(y, y.clone(), torch.tensor([4]), g,
                     lambda_reg=0.0, lambda_jerk=1.0)
        assert math.isclose(value.item(), 36.0, rel_tol=1e-6), (
            f"historical lambda_jerk numerics changed: {value.item()}"
        )

    def test_documented_as_non_physical(self) -> None:
        from nsmor.loss import BioDecisionLoss

        doc = (BioDecisionLoss.__doc__ or "").lower()
        assert "not physical jerk" in doc, (
            "loss docstring must state this is NOT physical jerk"
        )
        # The prose must also state the correct definition (first derivative
        # of acceleration / second derivative of velocity), not the false
        # "second derivative of acceleration".
        assert "second derivative of acceleration" not in doc, (
            "loss docstring still defines jerk as the 2nd derivative of "
            "acceleration"
        )


# ===============================================================
# C1 — Zero valid frames preserve incoming carry (actual no-op)
# ===============================================================

class TestZeroLengthNoOp:
    """A ``lengths == 0`` sample must not advance any recurrent carry."""

    def test_gru_zero_length_preserves_incoming_carry(self) -> None:
        from nsmor.model_nsmor_core import GRUUnit

        torch.manual_seed(0)
        g = GRUUnit(hidden_dim=4, num_layers=1).eval()
        x = torch.randn(2, 3, 4)
        h0 = torch.randn(1, 2, 4)
        with torch.no_grad():
            out, h_n = g(x, torch.tensor([3, 0]), h0=h0, return_hidden=True)
        # Zero-length row is an exact no-op; the active row advances.
        assert torch.equal(h_n[:, 1], h0[:, 1])
        assert not torch.equal(h_n[:, 0], h0[:, 0])
        assert torch.equal(out[1], torch.zeros_like(out[1]))

    def test_frontend_zero_length_preserves_dendritic_carry(self) -> None:
        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                      lif_dendritic_tau=5.0).eval()
        x = torch.randn(2, 3, 4)
        incoming = torch.randn(2, 1)
        m.frontend._dendritic_state = incoming.clone()
        with torch.no_grad():
            m.frontend(x, torch.tensor([0, 2]))
        assert torch.equal(
            m.frontend._dendritic_state[0:1], incoming[0:1].detach()
        ), "zero-length row advanced the dendritic carry"

    def test_core_mixed_active_empty_both_activations(self) -> None:
        def _sample1(t: torch.Tensor) -> torch.Tensor:
            # Batch axis is 0 for (B, H) and 1 for (num_layers, B, H).
            return t[1] if t.dim() == 2 else t[:, 1]

        for activation in ("relu", "swiglu"):
            torch.manual_seed(0)
            m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                          activation=activation).eval()
            x = torch.randn(2, 4, 8)
            L = torch.tensor([4, 0])
            with torch.no_grad():
                _, _, s = m(x, L, return_internals=True, states={})
                # Resuming with the zero-length sample kept must leave its
                # carry byte-identical.
                _, _, s2 = m(x, L, return_internals=True, states=s)
            for key in s:
                assert torch.equal(_sample1(s[key]), _sample1(s2[key])), (
                    f"{activation}: zero-length carry {key!r} not preserved"
                )

    def test_zero_time_tensor_rejected_not_conflated(self) -> None:
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
        with pytest.raises(ValueError):
            m(torch.zeros(2, 0, 8), torch.zeros(2, dtype=torch.long))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
    def test_raw_jax_zero_length_dendritic_no_op(self) -> None:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                      lif_dendritic_tau=5.0).eval()
        j = NSMoRCoreJAX.from_torch(m)
        x = torch.zeros(2, 4, 8)
        x[:, :, 0] = 0.6
        din = torch.full((2, 1), 7.0)
        with torch.no_grad():
            _, _, s = j(x, torch.tensor([0, 4]), return_internals=True,
                       states={"frontend_dendritic_state": din})
        assert s["frontend_dendritic_state"][0, 0].item() == 7.0, (
            "raw-JAX zero-length dendritic carry advanced"
        )


# ===============================================================
# C2 — Direct Flax conversion fails closed and is complete
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestDirectConversionEnvelope:

    @staticmethod
    def _flax(activation: str = "relu", **kw):
        from nsmor.jax.model import NSMoRModel

        m = NSMoRModel(hidden_dim=8, dt_ms=4.0, activation=activation, **kw)
        p = m.init(jax.random.PRNGKey(0), jnp.zeros((2, 5, 8)), jnp.array([5, 3]))
        return m, p

    @pytest.mark.parametrize("activation", ["relu", "swiglu"])
    def test_supported_strict_roundtrip_all_tensors_exact(self, activation: str) -> None:
        from nsmor.jax.model import load_from_torch_state_dict, to_torch_state_dict

        m, p = self._flax(activation)
        sd = to_torch_state_dict(p)
        # Every exported key must appear under BOTH the hierarchical and the
        # legacy-flat alias with identical values, so a corruption of either
        # representation is visible (a corrupted single alias is also rejected
        # by the loader's duplicate-agreement check).
        for key in sd:
            if key.startswith("backend.") or key.startswith("frontend."):
                alias = key.replace("frontend.", "").replace("backend.", "")
                assert alias in sd, f"missing flat alias for {key!r}"
                assert torch.equal(sd[key], sd[alias]), (
                    f"hierarchical/flat aliases disagree for {key!r}"
                )
        r = load_from_torch_state_dict(m, sd)
        # EVERY destination leaf (17 relu / 23 swiglu), not a hand-picked
        # subset: corrupting any single exported leaf (e.g. the SwiGLU decoder
        # ``direction_head.out_proj.weight``) fails this comparison.
        _assert_all_leaves_equal(p["params"], r["params"])
        if activation == "swiglu":
            # Mutation control: a corrupted decoder export must be detected,
            # proving the all-leaf comparison above is not inert.
            import copy as _copy

            mutated = _copy.deepcopy(r["params"])
            mutated["direction_head"]["out"]["kernel"] = (
                np.asarray(mutated["direction_head"]["out"]["kernel"]) + 1.0
            )
            with pytest.raises(AssertionError):
                _assert_all_leaves_equal(p["params"], mutated)

    @staticmethod
    def _dest_from(src: NSMoRCore, *, activation: str = "relu", **over):
        """Build a Flax destination FROM *src*'s own settings.

        Every supported field is copied from the live source so the ONLY
        difference between source and destination is the field a negative test
        deliberately mutates.  Without this, ``_flax()``'s default
        ``dropout_rate=0.1`` already mismatched the ``dropout=0.0`` source and
        the negative test raised regardless of the guarded mechanism.
        """
        from nsmor.jax.model import NSMoRModel

        lif = src.lif_cell
        kw = dict(
            hidden_dim=8, dt_ms=4.0, lif_alpha=float(lif.alpha),
            lif_threshold=float(lif.v_threshold), lif_beta=float(lif.beta),
            lif_tau_syn=float(lif.tau_syn), lif_tau_w=float(lif.tau_w),
            lif_b_adapt=float(lif.b_adapt), lif_v_rest=float(lif.v_rest),
            lif_abs_refract_ms=float(lif.abs_refract_ms),
            lif_rel_refract_ms=float(lif.rel_refract_ms),
            lif_lateral_inhibition=float(lif.lateral_inhibition),
            lif_tbptt_steps=int(src.backend._tbptt_steps),
            dropout_rate=float(src.direction_head.dropout_rate),
            gru_neuromod_gain=float(src.backend.gru_neuromod_gain),
            activation=activation,
        )
        kw.update(over)
        return NSMoRModel(**kw)

    def test_unsupported_source_topology_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        # Positive control: a supported source maps into a destination built
        # from ITS OWN settings (same dropout, etc.) with no spurious raise.
        ok = NSMoRCore(hidden_dim=8, dt_ms=4.0, dropout=0.0)
        r = load_from_torch_state_dict(
            self._dest_from(ok), ok.state_dict(), source=ok,
        )
        assert "params" in r

        # Negative 1: stacked GRU.  The only difference is the guarded depth;
        # the layer-1 keys are stripped so the ONLY possible rejection is the
        # ``assert_flax_supported`` guard (remove it -> this load succeeds).
        stacked = NSMoRCore(hidden_dim=8, dt_ms=4.0, dropout=0.0,
                            num_gru_layers=2)
        sd = {k: v for k, v in stacked.state_dict().items()
              if "l1" not in k}
        with pytest.raises(ValueError, match="num_gru_layers"):
            load_from_torch_state_dict(
                self._dest_from(stacked), sd, source=stacked,
            )

        # Negative 2: short-term plasticity (STP).  The extra ``U_stp_raw``
        # leaf is stripped, so again the ONLY rejection source is the guard.
        stp = NSMoRCore(hidden_dim=8, dt_ms=4.0, dropout=0.0,
                        lif_tau_fac=10.0, lif_tau_rec=50.0)
        sd_stp = {k: v for k, v in stp.state_dict().items()
                  if "U_stp_raw" not in k}
        with pytest.raises(ValueError, match="lif_tau_fac"):
            load_from_torch_state_dict(
                self._dest_from(stp), sd_stp, source=stp,
            )

        # Negative 3: frontend dendritic filtering — a non-parameter setting
        # with NO extra state_dict keys, so the guard is the sole rejection.
        dend = NSMoRCore(hidden_dim=8, dt_ms=4.0, dropout=0.0,
                         lif_dendritic_tau=5.0)
        with pytest.raises(ValueError, match="lif_dendritic_tau"):
            load_from_torch_state_dict(
                self._dest_from(dend), dend.state_dict(), source=dend,
            )

    def test_gain_and_activation_mismatch_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        # Positive control: a source and a destination with the SAME nonzero
        # gain map cleanly.
        ok = NSMoRCore(hidden_dim=8, dt_ms=4.0, dropout=0.0,
                       gru_neuromod_gain=0.5)
        r = load_from_torch_state_dict(
            self._dest_from(ok), ok.state_dict(), source=ok,
        )
        assert "params" in r

        # Negative 1: gain VALUE differs (both enabled, so the state_dict keys
        # match and only ``assert_semantics_match`` can reject).  Disable that
        # matcher and the load would succeed -> the guard is load-bearing.
        gain = NSMoRCore(hidden_dim=8, dt_ms=4.0, dropout=0.0,
                         gru_neuromod_gain=1.0)
        with pytest.raises(ValueError, match="gru_neuromod_gain"):
            load_from_torch_state_dict(
                self._dest_from(gain, gru_neuromod_gain=0.5),
                gain.state_dict(), source=gain,
            )

        # Negative 2: the DECLARED activation string differs while the executed
        # module topology is identical (no gate/value keys), so only the
        # semantic matcher can reject; remove it -> this load succeeds.
        act = NSMoRCore(hidden_dim=8, dt_ms=4.0, dropout=0.0)
        act.activation = "swiglu"
        with pytest.raises(ValueError, match="activation"):
            load_from_torch_state_dict(
                self._dest_from(act), act.state_dict(), source=act,
            )

    def test_missing_and_extra_keys_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict, to_torch_state_dict

        m, p = self._flax()
        sd = to_torch_state_dict(p)
        extra = dict(sd)
        extra["bogus.key"] = torch.zeros(1)
        with pytest.raises(ValueError):
            load_from_torch_state_dict(m, extra)
        missing = {k: v for k, v in sd.items() if "W_in.weight" not in k}
        with pytest.raises(ValueError):
            load_from_torch_state_dict(m, missing)


# ===============================================================
# C3 — Raw-JAX rejects active stochastic training
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestRawJAXDeterministicContract:

    @staticmethod
    def _runner(**kw):
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, **kw)
        return m, NSMoRCoreJAX.from_torch(m)

    def test_supported_eval_stays_identical(self) -> None:
        _, j = self._runner(dropout=0.5, sensory_noise_std=0.2)
        j.eval()
        x = torch.randn(1, 3, 8)
        with torch.no_grad():
            y1 = j(x, torch.tensor([3]))
            y2 = j(x, torch.tensor([3]))
        assert torch.equal(y1, y2)

    @pytest.mark.parametrize(
        "kw", [{"dropout": 0.5}, {"sensory_noise_std": 0.2},
               {"dropout": 0.5, "sensory_noise_std": 0.2}],
    )
    def test_active_stochastic_training_rejected(self, kw) -> None:
        m, j = self._runner(**kw)
        m.train()
        x = torch.randn(1, 3, 8)
        with pytest.raises(ValueError):
            j(x, torch.tensor([3]))

    def test_zero_stochastic_training_allowed(self) -> None:
        m, j = self._runner(dropout=0.0, sensory_noise_std=0.0)
        m.train()
        x = torch.randn(1, 3, 8)
        with torch.no_grad():
            j(x, torch.tensor([3]))  # must not raise

    def test_mode_change_cannot_bypass(self) -> None:
        m, j = self._runner(dropout=0.5)
        m.eval()
        x = torch.randn(1, 3, 8)
        with torch.no_grad():
            j(x, torch.tensor([3]))
        m.train()  # underlying model mode change is honored
        with pytest.raises(ValueError):
            j(x, torch.tensor([3]))


# ===============================================================
# C4 — Invalid padding never computed into NaN / overflow
# ===============================================================

class TestInvalidPaddingSanitized:

    @staticmethod
    def _model(**kw) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0, **kw).eval()

    def test_finite_extreme_invalid_suffix_harmless(self) -> None:
        m = self._model(lif_dendritic_tau=5.0)
        x = torch.zeros(1, 3, 8)
        x[0, 0, 0] = 0.5
        x[0, 1:, :] = 1e30
        with torch.no_grad():
            _, _, s = m(x, torch.tensor([1]), return_internals=True, states={})
            _, _, s2 = m(x, torch.tensor([1]), return_internals=True, states=s)
        for key in s:
            assert torch.isfinite(s2[key]).all(), f"{key} nonfinite"
        # The unpadded reference must match exactly.
        with torch.no_grad():
            _, _, ref = m(x[:, :1], torch.tensor([1]), return_internals=True,
                          states={})
        assert torch.equal(s["lif_v"], ref["lif_v"])

    def test_nan_invalid_suffix_harmless(self) -> None:
        m = self._model()
        x = torch.zeros(1, 3, 8)
        x[0, 0, 0] = 0.5
        x[0, 1:, :] = float("nan")
        with torch.no_grad():
            _, _, s = m(x, torch.tensor([1]), return_internals=True, states={})
        assert all(torch.isfinite(v).all().item() for v in s.values())

    def test_nonfinite_valid_frame_refused(self) -> None:
        m = self._model()
        x = torch.randn(1, 3, 8)
        x[0, 0, 0] = float("nan")
        with pytest.raises(ValueError):
            m(x, torch.tensor([3]))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
    def test_raw_jax_nan_padded_harmless(self) -> None:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = self._model()
        j = NSMoRCoreJAX.from_torch(m)
        x = torch.zeros(1, 4, 8)
        x[0, 0, 0] = 0.5
        x[0, 2:, :] = float("nan")
        with torch.no_grad():
            _, _, s = j(x, torch.tensor([2]), return_internals=True, states={})
        assert all(torch.isfinite(v).all().item() for v in s.values())


# ===============================================================
# C5 — Selective per-sample TBPTT detachment
# ===============================================================

class TestSelectiveTBPTT:

    def _grad(self, lengths, T, tbptt=2):
        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                      lif_tbptt_steps=tbptt).eval()
        base = torch.randn(1, 5, 8)
        x = base[:, :T].clone().requires_grad_(True)
        _, _, s = m(x, torch.tensor(lengths), return_internals=True, states={})
        s["lif_v"].sum().backward()
        return x.grad

    def test_padding_does_not_zero_valid_gradient(self) -> None:
        g1 = self._grad([1], 1)
        g5 = self._grad([1], 5)
        # The valid-frame gradient must be preserved (up to fp reassociation
        # from the extra masked frames); the historical bug zeroed it.
        assert torch.allclose(g1[0, 0], g5[0, 0], rtol=1e-5, atol=1e-6), (
            "padding a 1-frame sample to 5 frames changed the valid gradient"
        )
        assert g5[0, 0].abs().max() > 0
        # Padded frames must receive no gradient.
        assert g5[0, 1:].abs().max().item() == 0.0

    def test_active_long_sequence_still_detaches(self) -> None:
        # A long active sequence must still be gradient-bounded by TBPTT
        # (the frame-0 gradient reaches only the most recent window).
        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                      lif_tbptt_steps=2).eval()
        x = torch.randn(1, 20, 8, requires_grad=True)
        _, _, s = m(x, torch.tensor([20]), return_internals=True, states={})
        s["lif_v"].sum().backward()
        # Early frames beyond the truncation window receive no gradient.
        assert x.grad[:, 0].abs().max().item() == 0.0
        assert x.grad[:, -1].abs().max().item() > 0.0


# ===============================================================
# C6 — Legacy dendritic carry migration + malformed rejection
# ===============================================================

class TestDendriticCarryMigration:

    def _model(self) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                         lif_dendritic_tau=5.0).eval()

    def test_legacy_b2_visual_component_migrated(self) -> None:
        m = self._model()
        x = torch.randn(2, 3, 4)
        legacy = torch.stack([torch.full((2,), 3.0), torch.zeros(2)], dim=1)
        m.frontend._dendritic_state = legacy.clone()
        with torch.no_grad():
            m.frontend(x, torch.tensor([3, 3]))
        migrated = m.frontend._dendritic_state.clone()
        m.frontend._dendritic_state = legacy[:, 0:1].clone()
        with torch.no_grad():
            m.frontend(x, torch.tensor([3, 3]))
        assert migrated.shape == (2, 1)
        assert torch.allclose(migrated, m.frontend._dendritic_state), (
            "legacy (B,2) did not migrate to the visual column"
        )

    def test_malformed_state_rejected(self) -> None:
        m = self._model()
        x = torch.randn(2, 3, 4)
        for bad in (torch.randn(2, 3), torch.randn(1, 1), torch.randn(2, 1, 1)):
            m.frontend._dendritic_state = bad
            with pytest.raises(ValueError):
                with torch.no_grad():
                    m.frontend(x, torch.tensor([3, 3]))


# ===============================================================
# C7 — Raw-JAX independent per-field carry restore
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestRawJAXIndependentCarry:

    def _pair(self):
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                      lif_dendritic_tau=5.0).eval()
        return m, NSMoRCoreJAX.from_torch(m)

    def test_frontend_only_carry_matches_torch(self) -> None:
        m, j = self._pair()
        x = torch.zeros(1, 2, 8)
        L = torch.tensor([2])
        din = torch.tensor([[3.0]])
        with torch.no_grad():
            _, _, st = m(x, L, return_internals=True,
                        states={"frontend_dendritic_state": din})
            _, _, sj = j(x, L, return_internals=True,
                        states={"frontend_dendritic_state": din})
        assert torch.allclose(
            st["frontend_dendritic_state"], sj["frontend_dendritic_state"],
            atol=1e-6,
        ), "raw-JAX ignored independently supplied frontend carry"

    def test_gru_only_carry_restored(self) -> None:
        # A partial state containing ONLY gru_h must be accepted and must
        # change the result vs the canonical zero default (proving the field
        # is not silently discarded).
        m, j = self._pair()
        x = torch.randn(1, 2, 8)
        L = torch.tensor([2])
        gru = torch.randn(1, 1, 4)
        with torch.no_grad():
            _, _, sj = j(x, L, return_internals=True, states={"gru_h": gru})
            _, _, s0 = j(x, L, return_internals=True, states={})
        assert sj["gru_h"].shape == (1, 1, 4)
        assert not torch.allclose(sj["gru_h"], s0["gru_h"]), (
            "independently supplied gru_h was silently ignored"
        )

    def test_malformed_dendritic_rejected(self) -> None:
        _, j = self._pair()
        x = torch.randn(1, 2, 8)
        with pytest.raises(ValueError):
            j(x, torch.tensor([2]),
              states={"frontend_dendritic_state": torch.randn(1, 3)})


# ===============================================================
# C8 — Mixed UQ attribution through all convenience APIs
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestUQAttribution:

    def _fixture(self):
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.jax.model import NSMoRModel

        model = NSMoRModel(hidden_dim=4, dt_ms=4.0, dropout_rate=0.0,
                           sensory_noise_std=0.1, lif_lateral_inhibition=0.0)
        x = jnp.ones((1, 3, 8), dtype=jnp.float32)
        L = jnp.array([3], dtype=jnp.int32)
        params = model.init(jax.random.PRNGKey(0), x, L)
        return model, params, x, L

    def test_default_is_dropout_only_epistemic(self) -> None:
        from nsmor.analysis.uq_jax import MCDropoutAnalyzerJAX

        model, params, x, L = self._fixture()
        r = MCDropoutAnalyzerJAX(model, params, n_samples=8, seed=0).predict(x, L)
        assert r["attribution"]["dispersion"] == "epistemic_dropout_only"
        assert r["attribution"]["epistemic"] is True
        assert float(np.abs(r["y_std"]).max()) < 1e-6

    def test_mixed_attribution_through_every_api(self) -> None:
        from nsmor.analysis.uq_jax import (
            MCDropoutAnalyzerJAX, mc_dropout_predict_jax,
            mc_dropout_uncertainty_jax,
        )

        model, params, x, L = self._fixture()
        a = MCDropoutAnalyzerJAX(model, params, n_samples=8, seed=0,
                                 include_sensory_noise=True)
        assert a.predict(x, L)["attribution"]["dispersion"] == (
            "mixed_dropout_and_sensory_noise"
        )
        assert a.uncertainty_per_trial(x, L)["attribution"][
            "includes_aleatoric"
        ] is True
        assert mc_dropout_predict_jax(
            model, params, np.asarray(x), np.asarray(L), n_samples=4,
            include_sensory_noise=True,
        )["attribution"]["dispersion"] == "mixed_dropout_and_sensory_noise"
        assert mc_dropout_uncertainty_jax(
            model, params, np.asarray(x), np.asarray(L), n_samples=4,
            include_sensory_noise=True,
        )["attribution"]["dispersion"] == "mixed_dropout_and_sensory_noise"

    def test_bool_trust_boundary(self) -> None:
        from nsmor.analysis.uq_jax import MCDropoutAnalyzerJAX

        model, params, _, _ = self._fixture()
        with pytest.raises(ValueError):
            MCDropoutAnalyzerJAX(model, params, n_samples=4,
                                 include_sensory_noise=1)


# ===============================================================
# R1 — fail-closed UQ finiteness (r3 root 1)
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestUQFinitenessFailClosed:
    """A nonfinite VALID observation must fail closed through every API."""

    def _fixture(self, dropout_rate: float = 0.35):
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.jax.model import NSMoRModel

        model = NSMoRModel(hidden_dim=4, dt_ms=4.0, dropout_rate=dropout_rate,
                           lif_lateral_inhibition=0.0)
        x = jnp.ones((1, 3, 8), dtype=jnp.float32)
        L = jnp.array([3], dtype=jnp.int32)
        params = model.init(jax.random.PRNGKey(0), x, L)
        return model, params

    def test_valid_nan_rejected_every_api(self) -> None:
        from nsmor.analysis.uq_jax import (
            MCDropoutAnalyzerJAX, mc_dropout_predict_jax,
            mc_dropout_uncertainty_jax,
        )

        model, params = self._fixture()
        x = jnp.ones((1, 3, 8), dtype=jnp.float32).at[0, 0, 0].set(jnp.nan)
        L = jnp.array([3], dtype=jnp.int32)
        analyzer = MCDropoutAnalyzerJAX(model, params, n_samples=4, seed=0)
        with pytest.raises(ValueError):
            analyzer.predict(x, L)
        with pytest.raises(ValueError):
            analyzer.uncertainty_per_trial(x, L)
        with pytest.raises(ValueError):
            mc_dropout_predict_jax(model, params, np.asarray(x), np.asarray(L),
                                   n_samples=4)
        with pytest.raises(ValueError):
            mc_dropout_uncertainty_jax(model, params, np.asarray(x),
                                       np.asarray(L), n_samples=4)

    def test_valid_inf_rejected(self) -> None:
        from nsmor.analysis.uq_jax import MCDropoutAnalyzerJAX

        model, params = self._fixture()
        x = jnp.ones((1, 3, 8), dtype=jnp.float32).at[0, 1, 2].set(jnp.inf)
        L = jnp.array([3], dtype=jnp.int32)
        with pytest.raises(ValueError):
            MCDropoutAnalyzerJAX(model, params, n_samples=4, seed=0).predict(x, L)

    def test_invalid_padding_suffix_harmless(self) -> None:
        from nsmor.analysis.uq_jax import MCDropoutAnalyzerJAX

        model, params = self._fixture(dropout_rate=0.1)
        # NaN only beyond the valid length (invalid suffix).
        x = jnp.ones((1, 3, 8), dtype=jnp.float32).at[0, 2, 0].set(jnp.nan)
        L = jnp.array([2], dtype=jnp.int32)
        r = MCDropoutAnalyzerJAX(model, params, n_samples=4, seed=0).predict(x, L)
        assert np.isfinite(r["y_mean"]).all()
        assert np.isfinite(r["y_std"]).all()
        assert np.isfinite(r["y_samples"]).all()

    def test_attribution_retained_after_guard(self) -> None:
        from nsmor.analysis.uq_jax import MCDropoutAnalyzerJAX

        model, params = self._fixture(dropout_rate=0.0)
        x = jnp.ones((1, 3, 8), dtype=jnp.float32)
        L = jnp.array([3], dtype=jnp.int32)
        r = MCDropoutAnalyzerJAX(model, params, n_samples=4, seed=0).predict(x, L)
        assert r["attribution"]["dispersion"] == "epistemic_dropout_only"
        assert r["attribution"]["epistemic"] is True

    def test_nonfinite_derived_output_rejected(self) -> None:
        # A finite input that produces a nonfinite MC output must fail closed
        # rather than be returned as a confidence estimate.  Uses a stub model
        # so the guard is exercised independently of any input preflight.
        from nsmor.analysis.uq_jax import MCDropoutAnalyzerJAX

        class _NonfiniteModel:
            hidden_dim = 4
            dropout_rate = 0.1
            sensory_noise_std = 0.0

            def apply(self, params, x, lengths, **kw):
                B, T, _ = x.shape
                y = jnp.full((B, T), jnp.inf)
                gates = jnp.zeros((B, T, 2))
                return y, {"routing_gates": gates}

        x = jnp.ones((1, 3, 8), dtype=jnp.float32)
        L = jnp.array([3], dtype=jnp.int32)
        analyzer = MCDropoutAnalyzerJAX(_NonfiniteModel(), {}, n_samples=4, seed=0)
        with pytest.raises(ValueError):
            analyzer.predict(x, L)


# ===============================================================
# R2 — unconditional shared Flax input/length preflight (r3 root 2)
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestFlaxInputPreflight:
    """Direct eager apply / eval / UQ reject malformed lengths at skip=0."""

    def _model_params(self, persistence_skip: float = 0.0):
        from nsmor.jax.model import NSMoRModel

        m = NSMoRModel(hidden_dim=4, dt_ms=4.0, dropout_rate=0.0,
                       persistence_skip=persistence_skip,
                       lif_lateral_inhibition=0.0)
        x = jnp.ones((1, 3, 8), dtype=jnp.float32)
        L = jnp.array([3], dtype=jnp.int32)
        p = m.init(jax.random.PRNGKey(0), x, L)
        return m, p, x

    def test_direct_apply_rejects_malformed_lengths_skip0(self) -> None:
        m, p, x = self._model_params(0.0)
        for bad in (
            jnp.array([2.5]),                       # fractional
            jnp.array([-1], dtype=jnp.int32),       # negative
            jnp.array([4], dtype=jnp.int32),        # overlong
        ):
            with pytest.raises((ValueError, AssertionError)):
                m.apply(p, x, bad)

    def test_direct_apply_rejects_nonfinite_valid(self) -> None:
        m, p, x = self._model_params(0.0)
        xn = x.at[0, 0, 0].set(jnp.nan)
        with pytest.raises(ValueError):
            m.apply(p, xn, jnp.array([3], dtype=jnp.int32))

    def test_zero_length_legal_and_t0_rejected(self) -> None:
        m, p, x = self._model_params(0.0)
        y = m.apply(p, x, jnp.array([0], dtype=jnp.int32))  # legal no-op
        assert np.isfinite(np.asarray(y)).all()
        with pytest.raises(ValueError):
            m.apply(p, jnp.zeros((1, 0, 8)), jnp.array([0], dtype=jnp.int32))

    def test_eval_wrapper_rejects_malformed_lengths(self) -> None:
        from nsmor.analysis.jax_eval import JAXEvalWrapper

        torch.manual_seed(0)
        w = JAXEvalWrapper.from_torch(
            NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
        )
        X = torch.ones(1, 3, 8)
        for bad in (torch.tensor([2.5]), torch.tensor([-1]),
                    torch.tensor([4]), torch.tensor([True])):
            with pytest.raises((ValueError, AssertionError)):
                w(X, bad)

    def test_checked_host_jit_contract(self) -> None:
        from nsmor.jax.model import validate_input_and_lengths

        m, p, x = self._model_params(0.0)
        L = jnp.array([3], dtype=jnp.int32)
        validate_input_and_lengths(np.asarray(x), np.asarray(L), context="t")
        y = jax.jit(lambda pp, xx, ll: m.apply(pp, xx, ll))(p, x, L)
        assert np.isfinite(np.asarray(y)).all()


# ===============================================================
# R3 — known source/destination semantic correspondence (r3 root 3)
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestSourceSemanticCorrespondence:

    def _torch(self, **kw) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=8, dropout=0.0,
                         lif_lateral_inhibition=0.0, **kw).eval()

    def _dest(self, m: NSMoRCore, dt_ms: float = 4.0,
              tau_syn: float = None, beta: float = None):
        from nsmor.jax.model import NSMoRModel

        lif = m.lif_cell
        return NSMoRModel(
            hidden_dim=8, dt_ms=dt_ms, lif_lateral_inhibition=0.0,
            lif_alpha=float(lif.alpha), lif_threshold=float(lif.v_threshold),
            lif_beta=float(lif.beta) if beta is None else beta,
            lif_tau_syn=float(lif.tau_syn) if tau_syn is None else tau_syn,
            lif_tau_w=float(lif.tau_w), lif_b_adapt=float(lif.b_adapt),
            lif_v_rest=float(lif.v_rest),
            lif_abs_refract_ms=float(lif.abs_refract_ms),
            lif_rel_refract_ms=float(lif.rel_refract_ms),
            lif_tbptt_steps=int(m.backend._tbptt_steps),
            dropout_rate=float(m.direction_head.dropout_rate),
            activation="relu",
        )

    def test_dt_mismatch_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._torch(dt_ms=4.0)
        dest = self._dest(m, dt_ms=10.0)
        with pytest.raises(ValueError):
            load_from_torch_state_dict(dest, m.state_dict(), source=m)

    def test_supported_lif_setting_mismatch_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._torch(dt_ms=4.0, lif_tau_syn=5.0)
        dest = self._dest(m, dt_ms=4.0, tau_syn=0.0)
        with pytest.raises(ValueError):
            load_from_torch_state_dict(dest, m.state_dict(), source=m)

    def test_matched_source_accepted(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._torch(dt_ms=4.0, lif_tau_syn=5.0)
        dest = self._dest(m, dt_ms=4.0)
        r = load_from_torch_state_dict(dest, m.state_dict(), source=m)
        assert "params" in r

    def test_bare_state_dict_remains_parameter_only(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._torch(dt_ms=4.0)
        dest = self._dest(m, dt_ms=10.0)
        # No source -> parameter-layout-only contract; dt is NOT compared.
        r = load_from_torch_state_dict(dest, m.state_dict())
        assert "params" in r


# ===============================================================
# R4 — validate every legacy/hierarchical alias (r3 root 4)
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestAliasShapeAndValueValidation:

    def _flax(self):
        from nsmor.jax.model import NSMoRModel

        m = NSMoRModel(hidden_dim=8, dt_ms=4.0, lif_lateral_inhibition=0.0)
        p = m.init(jax.random.PRNGKey(0), jnp.zeros((2, 5, 8)), jnp.array([5, 3]))
        return m, p

    def test_conflicting_duplicate_alias_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict, to_torch_state_dict

        m, p = self._flax()
        sd = to_torch_state_dict(p)
        assert "frontend.sensory_encoder.net.0.weight" in sd
        assert "sensory_encoder.net.0.weight" in sd
        # Corrupt ONLY the hierarchical alias; flat stays nonzero.
        sd["frontend.sensory_encoder.net.0.weight"] = torch.zeros_like(
            sd["frontend.sensory_encoder.net.0.weight"]
        )
        with pytest.raises(ValueError):
            load_from_torch_state_dict(m, sd)

    def test_wrong_shape_duplicate_alias_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict, to_torch_state_dict

        m, p = self._flax()
        sd = to_torch_state_dict(p)
        sd["sensory_encoder.net.0.weight"] = torch.zeros(1, 1)  # bad duplicate
        with pytest.raises(ValueError):
            load_from_torch_state_dict(m, sd)

    def test_complete_flat_and_hierarchical_layouts_accepted(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict, to_torch_state_dict

        m, p = self._flax()
        full = to_torch_state_dict(p)
        load_from_torch_state_dict(m, full)          # both representations
        flat = {k: v for k, v in full.items()
                if not k.startswith(("frontend.", "backend."))}
        load_from_torch_state_dict(m, flat)          # flat-only
        hier = {k: v for k, v in full.items()
                if k.startswith(("frontend.", "backend."))}
        load_from_torch_state_dict(m, hier)          # hierarchical-only


# ===============================================================
# R5 — inhibition buffer envelope (r3 root 5)
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestInhibitionBufferEnvelope:

    def _flax(self):
        from nsmor.jax.model import NSMoRModel

        m = NSMoRModel(hidden_dim=8, dt_ms=4.0, lif_lateral_inhibition=0.5)
        p = m.init(jax.random.PRNGKey(0), jnp.zeros((2, 5, 8)), jnp.array([5, 3]))
        return m, p

    def test_canonical_full_roundtrip_preserved(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict, to_torch_state_dict

        m, p = self._flax()
        r = load_from_torch_state_dict(m, to_torch_state_dict(p))
        assert "lif_w_inhib" in r["params"]
        # Compare EVERY destination leaf (incl. the inhibition matrix), not
        # just the presence of the key: corrupting any single exported leaf
        # fails here.
        _assert_all_leaves_equal(p["params"], r["params"])

    def test_both_mask_aliases_missing_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict, to_torch_state_dict

        m, p = self._flax()
        sd = to_torch_state_dict(p)
        for k in ("backend.lif_cell._inhib_diag_mask", "lif_cell._inhib_diag_mask"):
            del sd[k]
        with pytest.raises(ValueError):
            load_from_torch_state_dict(m, sd)

    def test_single_mask_alias_accepted(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict, to_torch_state_dict

        m, p = self._flax()
        sd = to_torch_state_dict(p)
        del sd["lif_cell._inhib_diag_mask"]          # hierarchical only
        load_from_torch_state_dict(m, sd)
        sd2 = to_torch_state_dict(p)
        del sd2["backend.lif_cell._inhib_diag_mask"]  # flat only
        load_from_torch_state_dict(m, sd2)

    def test_malformed_mask_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict, to_torch_state_dict

        m, p = self._flax()
        sd = to_torch_state_dict(p)
        sd["backend.lif_cell._inhib_diag_mask"] = torch.ones(1)
        sd["lif_cell._inhib_diag_mask"] = torch.ones(1)
        with pytest.raises(ValueError):
            load_from_torch_state_dict(m, sd)

    def test_noncanonical_mask_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict, to_torch_state_dict

        m, p = self._flax()
        sd = to_torch_state_dict(p)
        zeros = torch.zeros(8, 8)
        sd["backend.lif_cell._inhib_diag_mask"] = zeros
        sd["lif_cell._inhib_diag_mask"] = zeros
        with pytest.raises(ValueError):
            load_from_torch_state_dict(m, sd)


# ===============================================================
# R6 — Torch independent partial carry + cache isolation (r3 root 6)
# ===============================================================

class TestTorchPartialCarryIsolation:

    def _model(self, **kw) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0, **kw).eval()

    def test_single_field_partial_not_gated_on_lif_v(self) -> None:
        m = self._model()
        x = torch.randn(1, 3, 8)
        for key in ("lif_i_syn", "lif_w_adapt", "lif_rel_refract",
                    "lif_refract"):
            val = torch.full((1, 4), 2.0)
            with torch.no_grad():
                _, _, s = m(x, torch.tensor([0]), return_internals=True,
                            states={key: val})
            assert torch.equal(s[key], val), (
                f"partial carry {key!r} dropped when lif_v absent"
            )

    def test_stp_partial_fields_preserved(self) -> None:
        m = self._model(lif_tau_fac=10.0, lif_tau_rec=50.0)
        x = torch.randn(1, 3, 8)
        xr = torch.full((1, 4), 0.7)
        uf = torch.full((1, 4), 0.3)
        with torch.no_grad():
            _, _, s = m(x, torch.tensor([0]), return_internals=True,
                        states={"lif_x_resource": xr, "lif_u_facil": uf})
        assert torch.equal(s["lif_x_resource"], xr)
        assert torch.equal(s["lif_u_facil"], uf)

    def test_mixed_active_empty_rows_partial(self) -> None:
        m = self._model()
        x = torch.randn(2, 4, 8)
        val = torch.full((2, 4), 0.3)
        with torch.no_grad():
            _, _, s = m(x, torch.tensor([4, 0]), return_internals=True,
                        states={"lif_i_syn": val})
        assert torch.equal(s["lif_i_syn"][1], val[1])

    def test_omitted_spike_history_not_leaked(self) -> None:
        m = self._model(lif_lateral_inhibition=0.5)
        x = torch.zeros(1, 4, 8)
        x[:, :, 0] = 0.6
        with torch.no_grad():
            _, _, s1 = m(x, torch.tensor([4]), return_internals=True, states={})
        # Resume with ONLY lif_v: the omitted history must be the canonical
        # zero (init), not the cached s1 history.
        with torch.no_grad():
            _, _, s_v = m(x, torch.tensor([4]), return_internals=True,
                          states={"lif_v": s1["lif_v"]})
            _, _, s_vh = m(x, torch.tensor([4]), return_internals=True,
                           states={"lif_v": s1["lif_v"],
                                   "lif_spike_history": torch.zeros(1, 4)})
        assert torch.equal(s_v["lif_spike_history"],
                           s_vh["lif_spike_history"])
        assert torch.allclose(s_v["lif_v"], s_vh["lif_v"], atol=1e-6)


# ===============================================================
# R7 — component-boundary numerical padding safety (r3 root 7)
# ===============================================================

class TestDirectComponentPaddingSafety:

    def test_bio_decision_core_nan_suffix_gradients_finite(self) -> None:
        from nsmor.model_nsmor_core import BioDecisionCore

        torch.manual_seed(0)
        core = BioDecisionCore(hidden_dim=4, dropout=0.0, lif_tbptt_steps=2).eval()
        e = torch.randn(2, 3, 4)
        with torch.no_grad():
            e[1, 1:, :] = float("nan")          # NaN only in row 1's suffix
        e = e.requires_grad_(True)
        prior = torch.randn(2, 3, 4, requires_grad=True)
        y = core(e, prior, torch.tensor([3, 1]))
        valid = (torch.arange(3).unsqueeze(0)
                 < torch.tensor([3, 1]).unsqueeze(1))
        y[valid].sum().backward()
        assert torch.isfinite(e.grad[valid]).all()
        for name, prm in core.named_parameters():
            if prm.grad is not None:
                assert torch.isfinite(prm.grad).all(), name

    def test_bio_decision_core_nonfinite_valid_refused(self) -> None:
        from nsmor.model_nsmor_core import BioDecisionCore

        core = BioDecisionCore(hidden_dim=4, dropout=0.0)
        e = torch.randn(1, 3, 4)
        e[0, 0, 0] = float("nan")
        prior = torch.randn(1, 3, 4)
        with pytest.raises(ValueError):
            core(e, prior, torch.tensor([3]))

    def test_gru_unit_empty_row_gradients_finite(self) -> None:
        from nsmor.model_nsmor_core import GRUUnit

        torch.manual_seed(0)
        g = GRUUnit(hidden_dim=4, num_layers=1).eval()
        x = torch.randn(2, 3, 4)
        with torch.no_grad():
            x[1] = float("nan")                 # all-NaN empty row
        x = x.requires_grad_(True)
        h0 = torch.randn(1, 2, 4, requires_grad=True)
        out, hn = g(x, torch.tensor([3, 0]), h0=h0, return_hidden=True)
        assert torch.isfinite(out).all() and torch.isfinite(hn).all()
        hn.sum().backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(h0.grad).all()
        for name, prm in g.named_parameters():
            assert torch.isfinite(prm.grad).all(), name

    def test_gru_unit_invalid_suffix_zero_gradient(self) -> None:
        from nsmor.model_nsmor_core import GRUUnit

        torch.manual_seed(0)
        g = GRUUnit(hidden_dim=4, num_layers=1).eval()
        x = torch.randn(2, 3, 4, requires_grad=True)
        out = g(x, torch.tensor([2, 2]))
        out.sum().backward()
        assert x.grad[:, 2, :].abs().max().item() == 0.0


# ===============================================================
# R8 — original carry dtype validation before casts (r3 root 8)
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestRawJAXCarryDtypeValidation:

    def _pair(self):
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                      lif_dendritic_tau=5.0).eval()
        return m, NSMoRCoreJAX.from_torch(m)

    def test_integer_dendritic_rejected(self) -> None:
        _, j = self._pair()
        x = torch.zeros(1, 2, 8)
        with pytest.raises(ValueError):
            j(x, torch.tensor([2]),
              states={"frontend_dendritic_state": torch.tensor([[3]])})

    def test_complex_dendritic_rejected(self) -> None:
        _, j = self._pair()
        x = torch.zeros(1, 2, 8)
        with pytest.raises(ValueError):
            j(x, torch.tensor([2]),
              states={"frontend_dendritic_state": torch.tensor([[3 + 4j]])})

    def test_integer_gru_carry_rejected(self) -> None:
        _, j = self._pair()
        x = torch.zeros(1, 2, 8)
        with pytest.raises(ValueError):
            j(x, torch.tensor([2]),
              states={"gru_h": torch.zeros(1, 1, 4, dtype=torch.int64)})

    def test_integer_lif_field_rejected(self) -> None:
        _, j = self._pair()
        x = torch.zeros(1, 2, 8)
        with pytest.raises(ValueError):
            j(x, torch.tensor([2]),
              states={"lif_i_syn": torch.zeros(1, 4, dtype=torch.int64)})

    def test_torch_rejects_integer_dendritic_parity(self) -> None:
        m, _ = self._pair()
        x = torch.zeros(1, 2, 8)
        with pytest.raises(ValueError):
            m(x, torch.tensor([2]), return_internals=True,
              states={"frontend_dendritic_state": torch.tensor([[3]])})

    def test_legacy_migration_and_noop_preserved(self) -> None:
        _, j = self._pair()
        x = torch.zeros(1, 2, 8)
        x[:, :, 0] = 0.6
        legacy = torch.tensor([[3.0, 9.0]])  # unequal columns
        # Legacy (B, 2) width must migrate the VISUAL column 0, so its
        # continuation output and exported carry must EQUAL a raw-JAX call
        # initialized with the (B, 1) column-0 slice.
        with torch.no_grad():
            y_leg, _, s_leg = j(x, torch.tensor([2]), return_internals=True,
                                states={"frontend_dendritic_state": legacy})
            y_col0, _, s_col0 = j(
                x, torch.tensor([2]), return_internals=True,
                states={"frontend_dendritic_state": legacy[:, 0:1]},
            )
            y_col1, _, s_col1 = j(
                x, torch.tensor([2]), return_internals=True,
                states={"frontend_dendritic_state": legacy[:, 1:2]},
            )
        assert torch.equal(y_leg, y_col0), "legacy carry did not select col 0"
        assert torch.equal(
            s_leg["frontend_dendritic_state"], s_col0["frontend_dendritic_state"],
        ), "legacy carry exported a value other than the col-0 continuation"
        # Control: the WRONG column (col 1) must produce a DIFFERENT carry and
        # trajectory, proving the migration is not a trivial no-op.
        assert not torch.equal(
            s_leg["frontend_dendritic_state"], s_col1["frontend_dendritic_state"],
        ), "col 0 vs col 1 indistinguishable -> migration test is inert"
        assert not torch.equal(y_leg, y_col1)
        # zero-length dendritic carry is an exact no-op
        with torch.no_grad():
            _, _, s = j(x, torch.tensor([0]), return_internals=True,
                        states={"frontend_dendritic_state": torch.tensor([[3.0]])})
        assert float(s["frontend_dendritic_state"].item()) == 3.0


# ===============================================================
# r4 R1 — every original Torch carry field validated before cast
# (findings A1 / B3)
# ===============================================================

class TestR4TorchCarryValidation:

    def _model(self, **kw) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0, **kw).eval()

    def test_nonfinite_lif_v_rejected(self) -> None:
        m = self._model()
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError):
            m(x, torch.tensor([4]), return_internals=True,
              states={"lif_v": torch.full((1, 4), float("nan"))})

    def test_complex_gru_carry_rejected(self) -> None:
        m = self._model()
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError):
            m(x, torch.tensor([4]), return_internals=True,
              states={"gru_h": torch.full((1, 1, 4), 1 + 9j)})

    def test_integer_gru_carry_rejected(self) -> None:
        m = self._model()
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError):
            m(x, torch.tensor([4]), return_internals=True,
              states={"gru_h": torch.ones(1, 1, 4, dtype=torch.int64)})

    def test_bool_and_complex_lif_field_rejected(self) -> None:
        m = self._model()
        x = torch.randn(1, 4, 8)
        for bad in (torch.ones(1, 4, dtype=torch.bool),
                    torch.full((1, 4), 2 + 3j)):
            with pytest.raises(ValueError):
                m(x, torch.tensor([4]), return_internals=True,
                  states={"lif_i_syn": bad})

    def test_wrong_shape_field_rejected(self) -> None:
        m = self._model()
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError):
            m(x, torch.tensor([4]), return_internals=True,
              states={"lif_w_adapt": torch.zeros(1, 5)})

    def test_malformed_spike_history_rejected_not_reset(self) -> None:
        m = self._model(lif_lateral_inhibition=0.5)
        x = torch.randn(1, 4, 8)
        # (B, 1) history would previously be silently reset by LIFCell.
        with pytest.raises(ValueError):
            m(x, torch.tensor([4]), return_internals=True,
              states={"lif_spike_history": torch.zeros(1, 1)})

    def test_nonfinite_empty_row_carry_rejected(self) -> None:
        m = self._model()
        x = torch.randn(2, 4, 8)
        # NaN ONLY in the zero-length (inactive) row; the active row is finite.
        # If empty-row validation were removed the finite active row would let
        # the call through, so this test now discriminates that guard.
        bad = torch.zeros(2, 4)
        bad[1] = float("nan")
        with pytest.raises(ValueError, match="nonfinite"):
            m(x, torch.tensor([4, 0]), return_internals=True,
              states={"lif_i_syn": bad})

    def test_valid_partial_carry_still_accepted(self) -> None:
        m = self._model()
        x = torch.randn(1, 4, 8)
        val = torch.full((1, 4), 2.0)
        # Zero-length sample is a no-op, so the supplied partial field is
        # preserved byte-identically (validation must not reject it).
        with torch.no_grad():
            _, _, s = m(x, torch.tensor([0]), return_internals=True,
                        states={"lif_i_syn": val})
        assert torch.equal(s["lif_i_syn"], val)


# ===============================================================
# r4 R3 — effective inhibition-history decay semantics (findings A3 / B2)
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestR4InhibitionDecaySemantics:

    def _torch(self, inhib_tau_ms: float, tau_syn: float = 1.0,
               strength: float = 0.5) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=8, dt_ms=4.0, dropout=0.0, lif_tau_syn=tau_syn,
                         lif_lateral_inhibition=strength,
                         lif_inhib_tau_ms=inhib_tau_ms).eval()

    def _dest(self, m: NSMoRCore, inhib_tau_ms: float,
              strength: float = 0.5):
        from nsmor.jax.model import NSMoRModel

        lif = m.lif_cell
        return NSMoRModel(
            hidden_dim=8, dt_ms=4.0, lif_alpha=float(lif.alpha),
            lif_threshold=float(lif.v_threshold), lif_beta=float(lif.beta),
            lif_tau_syn=float(lif.tau_syn), lif_tau_w=float(lif.tau_w),
            lif_b_adapt=float(lif.b_adapt), lif_v_rest=float(lif.v_rest),
            lif_abs_refract_ms=float(lif.abs_refract_ms),
            lif_rel_refract_ms=float(lif.rel_refract_ms),
            lif_lateral_inhibition=strength, lif_inhib_tau_ms=inhib_tau_ms,
            lif_tbptt_steps=int(m.backend._tbptt_steps),
            dropout_rate=float(m.direction_head.dropout_rate),
        )

    def test_inhibition_tau_mismatch_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._torch(inhib_tau_ms=8.0, tau_syn=1.0)
        dest = self._dest(m, inhib_tau_ms=200.0)
        with pytest.raises(ValueError):
            load_from_torch_state_dict(dest, m.state_dict(), source=m)

    def test_matched_inhibition_tau_accepted(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._torch(inhib_tau_ms=8.0, tau_syn=1.0)
        dest = self._dest(m, inhib_tau_ms=8.0)
        r = load_from_torch_state_dict(dest, m.state_dict(), source=m)
        assert "params" in r

    def test_effective_max_normalization_matches(self) -> None:
        # Torch stores _inhib_tau_ms = max(tau_syn, inhib_tau_ms); a Flax
        # destination whose effective max matches must still transfer even
        # when the raw constructor values differ.
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._torch(inhib_tau_ms=8.0, tau_syn=20.0)  # effective max = 20
        dest = self._dest(m, inhib_tau_ms=20.0)          # effective max = 20
        r = load_from_torch_state_dict(dest, m.state_dict(), source=m)
        assert "params" in r

    def test_disabled_inhibition_not_compared(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._torch(inhib_tau_ms=8.0, tau_syn=1.0, strength=0.0)
        # Destination inhibition also disabled but with a different raw tau:
        # the effective value is absent on both, so no spurious mismatch.
        dest = self._dest(m, inhib_tau_ms=200.0, strength=0.0)
        r = load_from_torch_state_dict(dest, m.state_dict(), source=m)
        assert "params" in r


# ===============================================================
# r4 R4 — original public lengths validated before any cast
# (findings A4 / B5)
# ===============================================================

class TestR4OriginalLengthsAtComponents:

    def test_gru_unit_rejects_fractional_and_bool(self) -> None:
        from nsmor.model_nsmor_core import GRUUnit

        g = GRUUnit(hidden_dim=4, num_layers=1).eval()
        x = torch.randn(1, 4, 4)
        for bad in (torch.tensor([1.75]), torch.tensor([-0.5]),
                    torch.tensor([4.75]), torch.tensor([True])):
            with pytest.raises(ValueError):
                g(x, bad)

    def test_frontend_rejects_fractional_and_bool(self) -> None:
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
        x = torch.randn(1, 4, 4)
        for bad in (torch.tensor([1.75]), torch.tensor([4.75]),
                    torch.tensor([True])):
            with pytest.raises(ValueError):
                m.frontend(x, bad)

    def test_bio_decision_core_rejects_fractional_and_bool(self) -> None:
        from nsmor.model_nsmor_core import BioDecisionCore

        c = BioDecisionCore(hidden_dim=4, dropout=0.0).eval()
        e = torch.randn(1, 4, 4)
        p = torch.randn(1, 4, 4)
        for bad in (torch.tensor([1.75]), torch.tensor([-0.5]),
                    torch.tensor([True])):
            with pytest.raises(ValueError):
                c(e, p, bad)

    def test_legal_zero_length_still_noop(self) -> None:
        from nsmor.model_nsmor_core import GRUUnit

        g = GRUUnit(hidden_dim=4, num_layers=1).eval()
        x = torch.randn(2, 4, 4)
        h0 = torch.randn(1, 2, 4)
        with torch.no_grad():
            out, hn = g(x, torch.tensor([4, 0]), h0=h0, return_hidden=True)
        assert torch.equal(hn[:, 1], h0[:, 1])

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
    def test_raw_no_jax_fallback_rejects_fractional(self) -> None:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
        j = NSMoRCoreJAX.from_torch(m)
        j.use_jax = False                     # force the real no-JAX branch
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError):
            j(x, torch.tensor([1.75]))
        with pytest.raises(ValueError):
            j(x, torch.tensor([True]))
        # Legal input still runs through the fallback.
        with torch.no_grad():
            y = j(x, torch.tensor([4]))
        assert torch.isfinite(y).all()


# ===============================================================
# r4 R5 — UQ convenience boundaries validate before JAX narrowing
# (findings A5 / B4)
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestR4UQNarrowingBoundaries:

    def _fixture(self):
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.jax.model import NSMoRModel

        model = NSMoRModel(hidden_dim=4, dt_ms=4.0, dropout_rate=0.2,
                           lif_lateral_inhibition=0.0)
        x = jnp.ones((1, 3, 8), dtype=jnp.float32)
        L = jnp.array([3], dtype=jnp.int32)
        params = model.init(jax.random.PRNGKey(0), x, L)
        return model, params

    def test_int64_overlong_rejected_both_apis(self) -> None:
        from nsmor.analysis.uq_jax import (
            mc_dropout_predict_jax, mc_dropout_uncertainty_jax,
        )

        model, params = self._fixture()
        x = np.ones((1, 3, 8), dtype=np.float32)
        bad = np.array([4294967299], dtype=np.int64)   # wraps to 3 at int32
        with pytest.raises(ValueError):
            mc_dropout_predict_jax(model, params, x, bad, n_samples=4)
        with pytest.raises(ValueError):
            mc_dropout_uncertainty_jax(model, params, x, bad, n_samples=4)

    def test_uint64_overlong_rejected_both_apis(self) -> None:
        from nsmor.analysis.uq_jax import (
            mc_dropout_predict_jax, mc_dropout_uncertainty_jax,
        )

        model, params = self._fixture()
        x = np.ones((1, 3, 8), dtype=np.float32)
        bad = np.array([4294967297], dtype=np.uint64)  # wraps to 1 at int32
        with pytest.raises(ValueError):
            mc_dropout_predict_jax(model, params, x, bad, n_samples=4)
        with pytest.raises(ValueError):
            mc_dropout_uncertainty_jax(model, params, x, bad, n_samples=4)

    def test_int64_negative_rejected_both_apis(self) -> None:
        from nsmor.analysis.uq_jax import (
            mc_dropout_predict_jax, mc_dropout_uncertainty_jax,
        )

        model, params = self._fixture()
        x = np.ones((1, 3, 8), dtype=np.float32)
        bad = np.array([-4294967295], dtype=np.int64)  # wraps to 1 at int32
        with pytest.raises(ValueError):
            mc_dropout_predict_jax(model, params, x, bad, n_samples=4)
        with pytest.raises(ValueError):
            mc_dropout_uncertainty_jax(model, params, x, bad, n_samples=4)

    def test_legal_inputs_still_accepted(self) -> None:
        from nsmor.analysis.uq_jax import (
            mc_dropout_predict_jax, mc_dropout_uncertainty_jax,
        )

        model, params = self._fixture()
        x = np.ones((2, 3, 8), dtype=np.float32)
        L = np.array([3, 0], dtype=np.int64)           # mixed, includes zero
        r = mc_dropout_predict_jax(model, params, x, L, n_samples=4)
        assert np.isfinite(r["y_mean"]).all()
        r2 = mc_dropout_uncertainty_jax(model, params, x, L, n_samples=4)
        assert np.isfinite(r2["y_mean"]).all()


# ===============================================================
# r4 R6 — inactive frontend carry keeps identity gradient (finding A6)
# ===============================================================

class TestR4FrontendInactiveCarryGradient:

    def _model(self) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                         lif_dendritic_tau=5.0,
                         lif_tbptt_steps=2).eval()

    def test_empty_row_identity_gradient_one(self) -> None:
        m = self._model()
        x = torch.randn(2, 5, 4)
        incoming = torch.randn(2, 1, requires_grad=True)
        m.frontend._dendritic_state = incoming
        m.frontend(x, torch.tensor([5, 0]))
        exported = m.frontend._dendritic_state
        # Empty row value unchanged.
        assert torch.equal(exported[1:2], incoming[1:2])
        # Identity derivative for the empty row.
        exported[1, 0].backward()
        assert incoming.grad is not None
        assert float(incoming.grad[1, 0]) == 1.0

    def test_empty_row_value_unchanged_and_stateful(self) -> None:
        m = self._model()
        x = torch.randn(2, 5, 4)
        incoming = torch.full((2, 1), 3.0)
        with torch.no_grad():
            m.frontend._dendritic_state = incoming.clone()
            m.frontend(x, torch.tensor([5, 0]))
            empty_val = float(m.frontend._dendritic_state[1, 0])
            # Continue statefully: feed the exported carry back.
            m.frontend._dendritic_state = m.frontend._dendritic_state.clone()
            m.frontend(x, torch.tensor([5, 0]))
        assert empty_val == 3.0
        assert float(m.frontend._dendritic_state[1, 0]) == 3.0

    def test_active_row_still_truncated(self) -> None:
        # The active row's export must NOT carry gradient to the incoming
        # state (TBPTT truncation preserved).
        m = self._model()
        x = torch.randn(1, 5, 4)
        incoming = torch.randn(1, 1, requires_grad=True)
        m.frontend._dendritic_state = incoming
        m.frontend(x, torch.tensor([5]))
        exported = m.frontend._dendritic_state
        exported[0, 0].backward()
        grad = incoming.grad
        assert grad is None or float(grad[0, 0]) == 0.0


# ===============================================================
# r5 R1 — representability of a supplied carry AFTER conversion
# (findings A1 / B1)
# ===============================================================

class TestR5CarryRepresentability:

    def _model(self, **kw) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0, **kw).eval()

    def test_finite_float64_carry_overflowing_float32_rejected(self) -> None:
        # A finite float64 1e300 is representable in float64 but overflows to
        # +inf when narrowed to the float32 computation dtype.
        m = self._model()
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError, match="representable"):
            m(x, torch.tensor([4]), return_internals=True,
              states={"lif_v": torch.full((1, 4), 1e300, dtype=torch.float64)})

    def test_finite_representable_float64_accepted(self) -> None:
        m = self._model()
        x = torch.randn(1, 4, 8)
        val = torch.full((1, 4), 2.5, dtype=torch.float64)
        with torch.no_grad():
            _, _, s = m(x, torch.tensor([4]), return_internals=True,
                        states={"lif_v": val})
        assert torch.isfinite(s["lif_v"]).all()

    def test_frontend_dendritic_overflow_rejected(self) -> None:
        m = self._model(lif_dendritic_tau=5.0)
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError, match="representable"):
            m(x, torch.tensor([4]), return_internals=True,
              states={"frontend_dendritic_state":
                      torch.full((1, 1), 1e300, dtype=torch.float64)})


# ===============================================================
# r5 R2 — negative refractory domain + safe inactive LIF arithmetic
# (findings B2)
# ===============================================================

class TestR5RefractoryDomainAndInactiveArithmetic:

    def _model(self, **kw) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                         lif_rel_refract_ms=20.0, **kw).eval()

    def test_negative_rel_refract_rejected(self) -> None:
        m = self._model()
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError, match="non-negative"):
            m(x, torch.tensor([4]), return_internals=True,
              states={"lif_rel_refract": torch.full((1, 4), -1000.0)})

    def test_negative_abs_refract_rejected(self) -> None:
        m = self._model()
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError, match="non-negative"):
            m(x, torch.tensor([4]), return_internals=True,
              states={"lif_refract": torch.full((1, 4), -3.0)})

    def test_inactive_row_extreme_counter_value_preserved_and_finite(self) -> None:
        """Value-preservation smoke test (relabelled from a NaN claim, A9/B9).

        Row 1 is inactive (length 0) yet supplies a huge positive relative
        counter.  EMPIRICAL FINDING: ``exp(-k_rel * counter)`` underflows to
        exactly 0 with derivative 0 for any large non-negative counter, so the
        unguarded branch never actually produces ``0 * inf = NaN`` here -- no
        input could be found that poisons the unguarded backward (a 5-field x
        8-value x config sweep found none; the counters are also validated
        non-negative at the trust boundary, so no overflow sign is reachable).
        The test therefore pins the REAL, mutation-detectable contract: the
        inactive row's carry is preserved EXACTLY (identity, not advanced) and
        its backward stays finite.  Removing the inactive-row restoration
        (``lif_state = lif_state_new``) makes this fail.
        """
        m = self._model()
        x = torch.randn(2, 4, 8)
        huge = torch.full((2, 4), 1e30, requires_grad=True)
        with torch.enable_grad():
            _, _, s = m(x, torch.tensor([4, 0]), return_internals=True,
                        states={"lif_rel_refract": huge})
            # Inactive row's carried value is preserved exactly.
            assert torch.equal(s["lif_rel_refract"][1:2], huge.detach()[1:2])
            s["lif_rel_refract"][1, 0].backward()
        assert huge.grad is not None
        assert torch.isfinite(huge.grad).all()

    def test_active_row_uses_its_own_state(self) -> None:
        m = self._model()
        x = torch.randn(1, 4, 8)
        with torch.no_grad():
            _, _, s = m(x, torch.tensor([4]), return_internals=True,
                        states={"lif_rel_refract": torch.zeros(1, 4)})
        assert torch.isfinite(s["lif_rel_refract"]).all()
        assert (s["lif_rel_refract"] >= 0).all()


# ===============================================================
# r5 R3 — raw JAX guard inspects LIVE stochastic child behavior
# (findings A2 / B3)
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestR5LiveStochasticGuard:

    def _runner(self, **kw):
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, **kw)
        return m, NSMoRCoreJAX.from_torch(m)

    def test_child_train_mode_after_parent_eval_rejected(self) -> None:
        m, j = self._runner(dropout=0.5)
        m.eval()
        m.direction_head.train()  # child left in train mode
        x = torch.zeros(1, 2, 8)
        with pytest.raises(ValueError, match="stochastic"):
            j(x, torch.tensor([2]))

    def test_live_dropout_p_change_detected(self) -> None:
        m, j = self._runner(dropout=0.0)
        m.eval()
        for module in m.direction_head.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.75
        m.direction_head.train()
        x = torch.zeros(1, 2, 8)
        with pytest.raises(ValueError, match="stochastic"):
            j(x, torch.tensor([2]))

    def test_deterministic_eval_succeeds(self) -> None:
        m, j = self._runner(dropout=0.5, sensory_noise_std=0.2)
        m.eval()
        x = torch.zeros(1, 2, 8)
        y = j(x, torch.tensor([2]))
        assert y.shape == (1, 2)

    def test_sensory_child_train_mode_rejected(self) -> None:
        m, j = self._runner(sensory_noise_std=0.3)
        m.eval()
        m.sensory_encoder.train()
        x = torch.zeros(1, 2, 8)
        with pytest.raises(ValueError, match="stochastic"):
            j(x, torch.tensor([2]))


# ===============================================================
# r5 R4 — effective runtime recurrence coefficients (findings A3 / B4)
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestR5EffectiveRuntimeCoefficients:

    def _torch(self, tau_syn: float = 5.0) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=8, dt_ms=4.0, dropout=0.0,
                         lif_tau_syn=tau_syn).eval()

    def _dest(self, m: NSMoRCore, tau_syn: float = 5.0):
        from nsmor.jax.model import NSMoRModel

        lif = m.lif_cell
        return NSMoRModel(
            hidden_dim=8, dt_ms=4.0, lif_alpha=float(lif.alpha),
            lif_threshold=float(lif.v_threshold), lif_beta=float(lif.beta),
            lif_tau_syn=tau_syn, lif_tau_w=float(lif.tau_w),
            lif_b_adapt=float(lif.b_adapt), lif_v_rest=float(lif.v_rest),
            lif_abs_refract_ms=float(lif.abs_refract_ms),
            lif_rel_refract_ms=float(lif.rel_refract_ms),
            lif_lateral_inhibition=float(lif.lateral_inhibition),
            lif_tbptt_steps=int(m.backend._tbptt_steps),
            dropout_rate=float(m.direction_head.dropout_rate),
        )

    def test_mutated_alpha_syn_buffer_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._torch(tau_syn=20.0)
        # Canonical exp(-4/20) = 0.8187; mutate the registered nonpersistent
        # buffer away from it while leaving the declared tau_syn unchanged.
        with torch.no_grad():
            m.lif_cell._alpha_syn.fill_(0.0)
        dest = self._dest(m, tau_syn=20.0)
        with pytest.raises(ValueError, match="noncanonical"):
            load_from_torch_state_dict(dest, m.state_dict(), source=m)

    def test_canonical_buffer_accepted(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._torch(tau_syn=20.0)
        dest = self._dest(m, tau_syn=20.0)
        r = load_from_torch_state_dict(dest, m.state_dict(), source=m)
        assert "params" in r


# ===============================================================
# r5 R5 — raw recurrent trajectory coordinate for analysis
# (finding A4)
# ===============================================================

class TestR5RawTrajectoryCoordinate:

    def test_internals_expose_raw_key(self) -> None:
        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                      gru_neuromod_gain=1.0).eval()
        x = torch.randn(1, 4, 8)
        with torch.no_grad():
            _, internals = m(x, torch.tensor([4]), return_internals=True)
        assert "gru_hidden_raw" in internals
        assert internals["gru_hidden_raw"].shape == (1, 4, 4)

    def test_gain_scales_routed_but_not_raw(self) -> None:
        from torch.nn.utils.rnn import pad_packed_sequence

        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                      gru_neuromod_gain=1.0).eval()
        # Force a nonzero gain (sigmoid(bias)*2 with bias=1 -> ~1.46).
        # Padded batch: lengths [4, 2] so the raw trajectory must also be the
        # true padded GRU output.
        x = torch.randn(2, 4, 8)
        L = torch.tensor([4, 2])
        captured: dict = {}

        def _hook(module, _inp, out):
            padded, _ = pad_packed_sequence(
                out[0], batch_first=True, total_length=4,
            )
            captured["out"] = padded

        handle = m.backend.gru_unit.gru.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                _, internals = m(x, L, return_internals=True)
        finally:
            handle.remove()
        routed = internals["gru_hidden"]
        raw = internals["gru_hidden_raw"]
        # ``gru_hidden_raw`` must be EXACTLY the direct GRU trajectory captured
        # under the hook (not merely "different from routed"); a gain applied to
        # the raw coordinate, or a post-gain export, would fail this equality.
        assert torch.equal(raw, captured["out"]), (
            "gru_hidden_raw is not the direct (pre-gain) GRU trajectory"
        )
        # ...and the routed output carries the (non-unit) gain, so the two
        # coordinates genuinely differ.
        assert not torch.allclose(routed, raw)

    def test_zero_gain_keys_are_identical(self) -> None:
        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
        x = torch.randn(1, 4, 8)
        with torch.no_grad():
            _, internals = m(x, torch.tensor([4]), return_internals=True)
        assert torch.equal(internals["gru_hidden"], internals["gru_hidden_raw"])

    def test_raw_helper_prefers_raw_key(self) -> None:
        from nsmor.analysis.dynamics import raw_gru_trajectory

        raw = torch.zeros(1, 2, 3)
        routed = torch.ones(1, 2, 3)
        model = SimpleNamespace(gru_neuromod_gain=1.0)
        got = raw_gru_trajectory(
            model, {"gru_hidden": routed, "gru_hidden_raw": raw},
            context="test",
        )
        assert got is raw

    def test_raw_helper_rejects_gain_without_raw_key(self) -> None:
        from nsmor.analysis.dynamics import raw_gru_trajectory

        model = SimpleNamespace(gru_neuromod_gain=1.0)
        with pytest.raises(ValueError, match="gru_hidden_raw"):
            raw_gru_trajectory(
                model, {"gru_hidden": torch.ones(1, 2, 3)}, context="test",
            )

    def test_raw_helper_allows_explicit_unit_gain_probe(self) -> None:
        from nsmor.analysis.dynamics import raw_gru_trajectory

        # Explicit declaration: the routed output IS the raw state.
        model = SimpleNamespace(gru_hidden_is_raw=True)
        routed = torch.ones(1, 2, 3)
        got = raw_gru_trajectory(model, {"gru_hidden": routed}, context="test")
        assert got is routed

    def test_raw_helper_rejects_unknown_gain_metadata(self) -> None:
        from nsmor.analysis.dynamics import raw_gru_trajectory

        # r6 R8: missing gain metadata is UNKNOWN, not unit gain.
        with pytest.raises(ValueError, match="UNKNOWN|gru_neuromod_gain"):
            raw_gru_trajectory(
                SimpleNamespace(), {"gru_hidden": torch.ones(1, 2, 3)},
                context="test",
            )

    def test_extract_gru_states_uses_raw_coordinate(self) -> None:
        from nsmor.analysis.dynamics import FixedPointAdapter

        class _Probe(torch.nn.Module):
            hidden_dim = 3

            def __init__(self) -> None:
                super().__init__()
                self.gru_unit = SimpleNamespace(
                    gru=torch.nn.GRU(3, 3, batch_first=True),
                )
                self.gru_neuromod_gain = 1.0

            def eval(self):
                return self

            def parameters(self, recurse: bool = True):
                yield torch.zeros(1)

            def __call__(self, x, lengths, *, return_internals=False):
                B, T, _ = x.shape
                return torch.zeros(B, T), {
                    "gru_hidden": torch.ones(B, T, 3),
                    "gru_hidden_raw": torch.zeros(B, T, 3),
                }

        class _Loader:
            def __iter__(self):
                yield torch.zeros(1, 2, 5), torch.zeros(1, 2), torch.tensor([2])

        adapter = FixedPointAdapter(_Probe(), device=torch.device("cpu"))
        trajs = adapter.extract_gru_states(_Loader())
        assert torch.equal(trajs[0], torch.zeros(2, 3))


# ===============================================================
# r5 R6 — unsupported stacked-GRU analysis envelope (findings A5 / B5)
# ===============================================================

class TestR5StackedGRUAnalysisEnvelope:

    def test_torch_adapter_refuses_two_layer(self) -> None:
        from nsmor.analysis.dynamics import FixedPointAdapter

        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                      num_gru_layers=2).eval()
        with pytest.raises(ValueError, match="single-layer"):
            FixedPointAdapter(m, device=torch.device("cpu"))

    def test_factory_refuses_two_layer(self) -> None:
        from nsmor.analysis.analyze_jacobian_jax import create_jacobian_adapter

        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                      num_gru_layers=2).eval()
        with pytest.raises(ValueError, match="single-layer"):
            create_jacobian_adapter(m, device=torch.device("cpu"),
                                    backend="torch")

    def test_single_layer_adapter_still_constructs(self) -> None:
        from nsmor.analysis.dynamics import FixedPointAdapter

        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
        adapter = FixedPointAdapter(m, device=torch.device("cpu"))
        assert adapter.model is m

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
    def test_jax_adapter_refuses_two_layer(self) -> None:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.analysis.dynamics_jax import FixedPointAdapterJAX

        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                      num_gru_layers=2).eval()
        with pytest.raises(ValueError, match="single-layer"):
            FixedPointAdapterJAX(m, device=torch.device("cpu"), backend="jax")

    def test_depth_guard_allows_probe_without_gru(self) -> None:
        from nsmor.analysis.dynamics import assert_supported_gru_depth

        assert_supported_gru_depth(SimpleNamespace(), context="test")


# ===============================================================
# r6 R1 — actual computation-dtype representability
# ===============================================================

class TestR6ActualComputationDtype:

    def test_default_dtype_float64_still_rejects_unrepresentable(self) -> None:
        # Construct a float32 model, then set the global default dtype to
        # float64.  A finite float64 carry must still be validated against the
        # ACTUAL forced-FP32 computation representation.
        prev = torch.get_default_dtype()
        try:
            torch.manual_seed(0)
            m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
            torch.set_default_dtype(torch.float64)
            x = torch.zeros(1, 2, 8, dtype=torch.float32)
            with pytest.raises(ValueError, match="representable"):
                m(x, torch.tensor([0]), return_internals=True,
                  states={"lif_v": torch.full((1, 4), 1e300,
                                              dtype=torch.float64)})
        finally:
            torch.set_default_dtype(prev)

    def test_representable_carry_under_float64_default_accepted(self) -> None:
        prev = torch.get_default_dtype()
        try:
            torch.manual_seed(0)
            m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
            torch.set_default_dtype(torch.float64)
            x = torch.zeros(1, 2, 8, dtype=torch.float32)
            with torch.no_grad():
                _, _, s = m(x, torch.tensor([0]), return_internals=True,
                            states={"lif_v": torch.full((1, 4), 2.0,
                                                        dtype=torch.float64)})
            assert torch.isfinite(s["lif_v"]).all()
        finally:
            torch.set_default_dtype(prev)


# ===============================================================
# r6 R2 — inactive operands safe BEFORE nonlinear math
# ===============================================================

class TestR6InactiveOperandsSafeBeforeMath:

    def test_inactive_gru_identity_gradient_finite(self) -> None:
        # Direct packed GRU: inactive (length 0) row with finite huge h0 and
        # recurrent weights 10 must keep finite identity gradient.
        from nsmor.model_nsmor_core import GRUUnit

        torch.manual_seed(0)
        g = GRUUnit(hidden_dim=4, num_layers=1, dropout=0.0).eval()
        with torch.no_grad():
            g.gru.weight_hh_l0.fill_(10.0)
        x = torch.zeros(2, 2, 4)
        h0 = torch.zeros(1, 2, 4)
        h0[0, 1] = 3e38
        h0 = h0.clone().requires_grad_(True)
        out, hn = g(x, torch.tensor([2, 0]), h0=h0, return_hidden=True)
        # Inactive carry value preserved exactly.
        assert torch.equal(hn[0, 1], h0[0, 1])
        hn[0, 1].sum().backward()
        assert h0.grad is not None
        assert torch.isfinite(h0.grad).all()
        # Per-element identity derivative 1 on the inactive row.
        assert torch.allclose(
            h0.grad[0, 1], torch.ones(4), atol=1e-6,
        )

    def test_inactive_frontend_dendritic_gradient_finite(self) -> None:
        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                      lif_dendritic_tau=5.0).eval()
        with torch.no_grad():
            m.frontend.sensory_encoder.net[0].weight.fill_(2.0)
        x = torch.zeros(1, 2, 4)
        incoming = torch.full((1, 1), 3e38, requires_grad=True)
        m.frontend._dendritic_state = incoming
        y = m.frontend(x, torch.tensor([0]))
        y.sum().backward()
        assert incoming.grad is not None
        assert torch.isfinite(incoming.grad).all()
        for pr in m.frontend.sensory_encoder.parameters():
            if pr.grad is not None:
                assert torch.isfinite(pr.grad).all()

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
    def test_raw_jax_inactive_reset_candidate_finite(self) -> None:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
        gru = m.backend.gru_unit.gru
        with torch.no_grad():
            hh = gru.weight_hh_l0
            H = 4
            hh[:H].fill_(-1.0)   # reset block
            hh[H:2 * H].zero_()  # update block
            hh[2 * H:].fill_(1.0)  # candidate block
        j = NSMoRCoreJAX.from_torch(m)
        x = torch.zeros(2, 2, 8)
        # MIXED active/inactive batch with a HUGE finite inactive GRU carry:
        # feeding 3e38 into the reset/candidate nonlinearity overflows to NaN
        # unless the inactive row is kept out of the recurrence.  A zero carry
        # made the W_hh edits a bitwise no-op, so the old test could not fail.
        carry = torch.zeros(2, 4)
        carry[1] = 3e38
        with torch.no_grad():
            y, internals, s = j(
                x, torch.tensor([2, 0]), return_internals=True,
                states={"gru_h": carry},
            )
        assert torch.isfinite(y).all()
        assert torch.isfinite(internals["gru_hidden_raw"]).all()
        # The inactive row's carry is an EXACT no-op (preserved, not reset).
        assert torch.equal(
            s["gru_h"][0, 1], torch.full((4,), 3e38),
        ), "inactive GRU carry was not preserved exactly"


# ===============================================================
# r6 R3 — consistent physical carry domains
# ===============================================================

class TestR6PhysicalCarryDomains:

    def _model(self, **kw) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0, **kw).eval()

    def test_stp_fraction_out_of_range_rejected(self) -> None:
        m = self._model(lif_tau_fac=20.0, lif_tau_rec=100.0)
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError, match="0, 1|\\[0, 1\\]"):
            m(x, torch.tensor([4]), return_internals=True,
              states={"lif_x_resource": torch.full((1, 4), -1.0)})
        with pytest.raises(ValueError, match="0, 1|\\[0, 1\\]"):
            m(x, torch.tensor([4]), return_internals=True,
              states={"lif_u_facil": torch.full((1, 4), 2.0)})

    def test_negative_adaptation_rejected(self) -> None:
        m = self._model(lif_tau_w=50.0, lif_b_adapt=0.1)
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError, match="non-negative"):
            m(x, torch.tensor([4]), return_internals=True,
              states={"lif_w_adapt": torch.full((1, 4), -1.0)})

    def test_negative_history_rejected(self) -> None:
        m = self._model(lif_lateral_inhibition=0.5)
        x = torch.randn(1, 4, 8)
        with pytest.raises(ValueError, match="0, 1|\\[0, 1\\]"):
            m(x, torch.tensor([4]), return_internals=True,
              states={"lif_spike_history": torch.full((1, 4), -1.0)})

    def test_direct_lif_negative_counter_rejected(self) -> None:
        from nsmor.model_nsmor_core import LIFCell

        cell = LIFCell(hidden_dim=4, rel_refract_ms=20.0, dt_ms=4.0)
        inp = torch.zeros(1, 4)
        state = (
            torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, 4),
            torch.full((1, 4), 1.0), torch.zeros(1, 4),
            torch.full((1, 4), -1e5),
        )
        with pytest.raises(ValueError, match="non-negative"):
            cell(inp, state)

    def test_valid_domains_accepted(self) -> None:
        m = self._model(lif_tau_fac=20.0, lif_tau_rec=100.0,
                        lif_lateral_inhibition=0.5)
        x = torch.randn(1, 4, 8)
        with torch.no_grad():
            _, _, s = m(x, torch.tensor([4]), return_internals=True,
                        states={"lif_x_resource": torch.full((1, 4), 0.5),
                                "lif_u_facil": torch.full((1, 4), 0.5),
                                "lif_spike_history": torch.zeros(1, 4)})
        assert torch.isfinite(s["lif_x_resource"]).all()


# ===============================================================
# r6 R4 — actually executed stochastic children
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestR6ExecutedStochasticChildren:

    def _runner(self, **kw):
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX
        from nsmor.model_nsmor_core import SensoryEncoder

        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, **kw)
        return m, NSMoRCoreJAX.from_torch(m), SensoryEncoder

    def test_replaced_executed_encoder_rejected(self) -> None:
        m, j, SensoryEncoder = self._runner(sensory_noise_std=0.0)
        m.eval()
        # Replace the ACTUAL executed child with a same-weight train-mode
        # encoder; the stale top-level alias stays eval.
        old = m.frontend.sensory_encoder
        new = SensoryEncoder(sensory_dim=4, hidden_dim=4,
                             noise_std=0.5, activation=old.activation)
        new.load_state_dict(old.state_dict())
        new.train()
        m.frontend.sensory_encoder = new
        x = torch.zeros(1, 2, 8)
        with pytest.raises(ValueError, match="stochastic"):
            j(x, torch.tensor([2]))

    def test_replaced_executed_head_rejected(self) -> None:
        m, j, _ = self._runner(dropout=0.0)
        m.eval()
        old = m.backend.direction_head
        import copy

        new = copy.deepcopy(old)
        new.dropout_rate = 0.75
        for module in new.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.75
        new.train()
        m.backend.direction_head = new
        x = torch.zeros(1, 2, 8)
        with pytest.raises(ValueError, match="stochastic"):
            j(x, torch.tensor([2]))


# ===============================================================
# r6 R5 — live normalization / activation / dropout topology
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestR6LiveTopology:

    def _src(self, activation: str = "relu") -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=8, dt_ms=4.0, dropout=0.0,
                         activation=activation).eval()

    def _dest(self, m: NSMoRCore, activation: str = "relu"):
        from nsmor.jax.model import NSMoRModel

        lif = m.lif_cell
        return NSMoRModel(
            hidden_dim=8, dt_ms=4.0, lif_alpha=float(lif.alpha),
            lif_threshold=float(lif.v_threshold), lif_beta=float(lif.beta),
            lif_tau_syn=float(lif.tau_syn), lif_tau_w=float(lif.tau_w),
            lif_b_adapt=float(lif.b_adapt), lif_v_rest=float(lif.v_rest),
            lif_abs_refract_ms=float(lif.abs_refract_ms),
            lif_rel_refract_ms=float(lif.rel_refract_ms),
            lif_lateral_inhibition=float(lif.lateral_inhibition),
            lif_tbptt_steps=int(m.backend._tbptt_steps),
            dropout_rate=float(m.direction_head.dropout_rate),
            activation=activation,
        )

    def test_live_layernorm_eps_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        m.direction_head.net[0].eps = 0.1
        with pytest.raises(ValueError, match="eps|topology"):
            load_from_torch_state_dict(self._dest(m), m.state_dict(), source=m)

    def test_replaced_relu_with_sigmoid_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        # Replace the ReLU slot (index 1 in the relu Sequential) with Sigmoid.
        m.direction_head.net[1] = torch.nn.Sigmoid()
        with pytest.raises(ValueError, match="topology|Sigmoid"):
            load_from_torch_state_dict(self._dest(m), m.state_dict(), source=m)

    def test_live_dropout_p_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        for module in m.direction_head.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.75
        with pytest.raises(ValueError, match="dropout"):
            load_from_torch_state_dict(self._dest(m), m.state_dict(), source=m)

    def test_canonical_topology_accepted(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        for act in ("relu", "swiglu"):
            m = self._src(activation=act)
            r = load_from_torch_state_dict(
                self._dest(m, activation=act), m.state_dict(), source=m,
            )
            assert "params" in r

    def test_source_none_still_layout_only(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        m.direction_head.net[0].eps = 0.1
        # No source -> parameter-layout-only contract; topology not inspected.
        r = load_from_torch_state_dict(self._dest(m), m.state_dict())
        assert "params" in r


# ===============================================================
# r6 R6 — finite COMPLETE effective-runtime coefficient envelope
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestR6CompleteCoefficientEnvelope:

    def _src(self) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=8, dt_ms=4.0, dropout=0.0,
                         lif_tau_syn=5.0, lif_tau_w=50.0,
                         lif_b_adapt=0.1).eval()

    def _dest(self, m: NSMoRCore):
        from nsmor.jax.model import NSMoRModel

        lif = m.lif_cell
        return NSMoRModel(
            hidden_dim=8, dt_ms=4.0, lif_alpha=float(lif.alpha),
            lif_threshold=float(lif.v_threshold), lif_beta=float(lif.beta),
            lif_tau_syn=float(lif.tau_syn), lif_tau_w=float(lif.tau_w),
            lif_b_adapt=float(lif.b_adapt), lif_v_rest=float(lif.v_rest),
            lif_abs_refract_ms=float(lif.abs_refract_ms),
            lif_rel_refract_ms=float(lif.rel_refract_ms),
            lif_lateral_inhibition=float(lif.lateral_inhibition),
            lif_tbptt_steps=int(m.backend._tbptt_steps),
            dropout_rate=float(m.direction_head.dropout_rate),
        )

    def test_nan_alpha_syn_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        with torch.no_grad():
            m.lif_cell._alpha_syn.fill_(float("nan"))
        with pytest.raises(ValueError, match="nonfinite|noncanonical"):
            load_from_torch_state_dict(self._dest(m), m.state_dict(), source=m)

    def test_mutated_delta_theta_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        m.lif_cell._delta_theta = 10.0
        with pytest.raises(ValueError, match="delta_theta|noncanonical"):
            load_from_torch_state_dict(self._dest(m), m.state_dict(), source=m)

    def test_mutated_k_rel_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        m.lif_cell._k_rel = 0.0
        with pytest.raises(ValueError, match="k_rel|noncanonical"):
            load_from_torch_state_dict(self._dest(m), m.state_dict(), source=m)

    def test_canonical_accepted(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        r = load_from_torch_state_dict(self._dest(m), m.state_dict(), source=m)
        assert "params" in r

    def test_raw_nan_coefficient_rejected(self) -> None:
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = self._src()
        with torch.no_grad():
            m.lif_cell._alpha_syn.fill_(float("nan"))
        with pytest.raises(ValueError, match="nonfinite"):
            NSMoRCoreJAX.from_torch(m)


# ===============================================================
# r6 R7 — actual GRU depth at raw conversion boundary
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestR6RawActualGRUDepth:

    def test_stale_single_layer_wrapper_with_two_layer_child_rejected(self) -> None:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
        # Actual child is 2-layer while the wrapper num_layers stays 1.
        m.backend.gru_unit.gru = torch.nn.GRU(
            4, 4, num_layers=2, batch_first=True,
        )
        with pytest.raises(ValueError, match="inconsistent|unknown|stacked"):
            NSMoRCoreJAX.from_torch(m)

    def test_supported_single_layer_still_converts(self) -> None:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
        j = NSMoRCoreJAX.from_torch(m)
        assert j.use_jax


# ===============================================================
# r6 R8 — explicit legacy raw-coordinate contract
# ===============================================================

class TestR6LegacyRawCoordinateContract:

    def test_scaled_legacy_model_jacobian_refused(self) -> None:
        from nsmor.analysis.dynamics import FixedPointAdapter

        class _Legacy(torch.nn.Module):
            hidden_dim = 3

            def __init__(self) -> None:
                super().__init__()
                self.gru_unit = SimpleNamespace(
                    gru=torch.nn.GRU(3, 3, batch_first=True),
                )
                # No gru_neuromod_gain, no gru_hidden_is_raw: UNKNOWN gain.

            def eval(self):
                return self

            def parameters(self, recurse: bool = True):
                yield torch.zeros(1)

            def __call__(self, x, lengths, *, return_internals=False):
                B, T, _ = x.shape
                return torch.zeros(B, T), {
                    "gru_hidden": 2.0 * torch.ones(B, T, 3),
                }

        class _Loader:
            def __iter__(self):
                yield torch.zeros(1, 2, 5), torch.zeros(1, 2), torch.tensor([2])

        adapter = FixedPointAdapter(_Legacy(), device=torch.device("cpu"))
        with pytest.raises(ValueError, match="UNKNOWN|gru_neuromod_gain"):
            adapter.extract_gru_states(_Loader())


# ===============================================================
# r6 R9 — numerically safe derived UQ summaries
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestR6SafeUQSummaries:

    def _analyzer(self):
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax
        import jax.numpy as jnp
        from nsmor.jax.model import NSMoRModel
        from nsmor.analysis.uq_jax import MCDropoutAnalyzerJAX

        model = NSMoRModel(hidden_dim=4, dt_ms=4.0, dropout_rate=0.5,
                           persistence_skip=1.0, lif_lateral_inhibition=0.0)
        xx = jnp.ones((1, 9, 8), dtype=jnp.float32)
        LL = jnp.array([9], dtype=jnp.int32)
        params = model.init(jax.random.PRNGKey(0), xx, LL)
        return MCDropoutAnalyzerJAX(model, params, n_samples=8, seed=0), xx, LL

    def test_large_finite_frames_no_false_zero_cv(self) -> None:
        analyzer, xx, LL = self._analyzer()
        # A finite per-frame mean whose float32 temporal SUM overflows, with a
        # small finite std.  The reduction must not fabricate CV=0.
        B, T = 1, 9
        y_mean = np.full((B, T), 5e37, dtype=np.float32)
        y_std = np.full((B, T), 1e-1, dtype=np.float32)
        analyzer.predict = lambda x, lengths: {
            "y_mean": y_mean, "y_std": y_std, "y_samples": y_mean[None],
            "attribution": {"dispersion": "epistemic_dropout_only",
                            "epistemic": True},
        }
        res = analyzer.uncertainty_per_trial(np.zeros((B, T, 8), np.float32),
                                             np.array([T], np.int64))
        assert np.isfinite(res["trial_cv"]).all()
        assert res["trial_cv"][0] > 0.0


# ===============================================================
# r7 R1 — exact executed topology at both known-live converters
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestR7ExactExecutedTopology:

    def _src(self, activation: str = "relu") -> NSMoRCore:
        torch.manual_seed(17)
        return NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                         activation=activation).eval()

    def _dest(self, m: NSMoRCore, activation: str = "relu"):
        from nsmor.jax.model import NSMoRModel

        lif = m.lif_cell
        return NSMoRModel(
            hidden_dim=4, dt_ms=4.0, lif_alpha=float(lif.alpha),
            lif_threshold=float(lif.v_threshold), lif_beta=float(lif.beta),
            lif_tau_syn=float(lif.tau_syn), lif_tau_w=float(lif.tau_w),
            lif_b_adapt=float(lif.b_adapt), lif_v_rest=float(lif.v_rest),
            lif_abs_refract_ms=float(lif.abs_refract_ms),
            lif_rel_refract_ms=float(lif.rel_refract_ms),
            lif_lateral_inhibition=float(lif.lateral_inhibition),
            lif_tbptt_steps=int(m.backend._tbptt_steps),
            dropout_rate=float(m.direction_head.dropout_rate),
            activation=activation,
        )

    def _x(self):
        return torch.randn(1, 4, 8)

    def test_flax_identity_in_relu_slot_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        m.backend.direction_head.net[1] = torch.nn.Identity()
        with pytest.raises(ValueError, match="topology|net"):
            load_from_torch_state_dict(self._dest(m), m.state_dict(), source=m)

    def test_raw_identity_in_relu_slot_rejected(self) -> None:
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = self._src()
        m.backend.direction_head.net[1] = torch.nn.Identity()
        with pytest.raises(ValueError, match="topology|net"):
            NSMoRCoreJAX.from_torch(m)

    def test_flax_encoder_identity_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        m.frontend.sensory_encoder.net[2] = torch.nn.Identity()
        with pytest.raises(ValueError, match="topology|net"):
            load_from_torch_state_dict(self._dest(m), m.state_dict(), source=m)

    def test_flax_appended_operator_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        m.backend.direction_head.net.append(torch.nn.ReLU())
        with pytest.raises(ValueError, match="topology|net"):
            load_from_torch_state_dict(self._dest(m), m.state_dict(), source=m)

    def test_flax_swiglu_child_activation_mismatch_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src(activation="swiglu")
        # Stale child activation string while the executed net is gated.
        m.frontend.sensory_encoder.activation = "relu"
        with pytest.raises(ValueError, match="topology|net|activation"):
            load_from_torch_state_dict(
                self._dest(m, activation="swiglu"), m.state_dict(), source=m,
            )

    def test_flax_nan_epsilon_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        m.direction_head.net[0].eps = float("nan")
        with pytest.raises(ValueError, match="eps|topology"):
            load_from_torch_state_dict(self._dest(m), m.state_dict(), source=m)

    def test_raw_noncanonical_epsilon_rejected(self) -> None:
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = self._src()
        m.direction_head.net[0].eps = 0.1
        with pytest.raises(ValueError, match="eps|topology"):
            NSMoRCoreJAX.from_torch(m)

    def test_canonical_absolute_parity_and_accept(self) -> None:
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX
        from nsmor.jax.model import load_from_torch_state_dict
        from nsmor.analysis.jax_eval import JAXEvalWrapper

        for act in ("relu", "swiglu"):
            m = self._src(act)
            x = self._x()
            with torch.no_grad():
                yt = m(x, torch.tensor([4]))
                w = JAXEvalWrapper.from_torch(m, device=torch.device("cpu"))
                yj = w(x, torch.tensor([4]))
            assert torch.allclose(yt, yj, atol=1e-5), act
            r = load_from_torch_state_dict(
                self._dest(m, act), m.state_dict(), source=m,
            )
            assert "params" in r
            assert NSMoRCoreJAX.from_torch(m).use_jax

    def test_source_none_layout_only_preserved(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        m.direction_head.net[1] = torch.nn.Identity()
        # No source -> parameter-layout-only contract; topology not inspected.
        r = load_from_torch_state_dict(self._dest(m), m.state_dict())
        assert "params" in r

    def test_raw_noncanonical_layernorm_axes_rejected(self) -> None:
        # Architecture v1 root 1: a live LayerNorm((2, H)) normalizes different
        # axes than the destination's last-axis LayerNorm, so the raw converter
        # must reject it even though class + epsilon are canonical.
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = self._src()
        m.frontend.sensory_encoder.net[1] = torch.nn.LayerNorm((2, 4))
        with pytest.raises(ValueError, match="normalized_shape|topology|net"):
            NSMoRCoreJAX.from_torch(m)

    def test_flax_noncanonical_layernorm_axes_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        m.frontend.sensory_encoder.net[1] = torch.nn.LayerNorm((2, 4))
        with pytest.raises(ValueError, match="normalized_shape|topology|net"):
            load_from_torch_state_dict(self._dest(m), m.state_dict(), source=m)

    def test_raw_nonaffine_layernorm_rejected(self) -> None:
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = self._src()
        m.direction_head.net[0] = torch.nn.LayerNorm(4, elementwise_affine=False)
        with pytest.raises(ValueError, match="elementwise_affine|topology|net"):
            NSMoRCoreJAX.from_torch(m)


# ===============================================================
# r7 R2 — complete original direct-LIF input/state/cache boundary
# ===============================================================

class TestR7DirectLIFBoundary:

    def _cell(self, **kw) -> LIFCell:
        torch.manual_seed(1)
        return LIFCell(hidden_dim=4, dt_ms=4.0, **kw)

    def _state(self, cell: LIFCell, value: float = 0.0):
        st = list(cell.init_state(1, torch.device("cpu")))
        return st

    def test_negative_resource_rejected(self) -> None:
        cell = self._cell(tau_fac=20.0, tau_rec=100.0)
        st = self._state(cell)
        st[6] = torch.full((1, 4), -1.0)
        with pytest.raises(ValueError, match="0, 1|\\[0, 1\\]"):
            cell(torch.zeros(1, 4), tuple(st))

    def test_infinite_resource_rejected(self) -> None:
        cell = self._cell(tau_fac=20.0, tau_rec=100.0)
        st = self._state(cell)
        st[6] = torch.full((1, 4), float("inf"))
        with pytest.raises(ValueError, match="nonfinite|0, 1|\\[0, 1\\]"):
            cell(torch.zeros(1, 4), tuple(st))

    def test_nan_membrane_rejected(self) -> None:
        cell = self._cell()
        st = self._state(cell)
        st[0] = torch.full((1, 4), float("nan"))
        with pytest.raises(ValueError, match="nonfinite"):
            cell(torch.zeros(1, 4), tuple(st))

    def test_nan_input_rejected(self) -> None:
        cell = self._cell()
        st = self._state(cell)
        with pytest.raises(ValueError, match="input_t|finite"):
            cell(torch.full((1, 4), float("nan")), tuple(st))

    def test_integer_membrane_rejected(self) -> None:
        cell = self._cell()
        st = self._state(cell)
        st[0] = torch.zeros(1, 4, dtype=torch.int64)
        with pytest.raises(ValueError, match="floating"):
            cell(torch.zeros(1, 4), tuple(st))

    def test_nan_history_rejected(self) -> None:
        cell = self._cell(lateral_inhibition=1.0)
        st = self._state(cell)
        cell._spike_history = torch.full((1, 4), float("nan"))
        with pytest.raises(ValueError, match="nonfinite"):
            cell(torch.zeros(1, 4), tuple(st))

    def test_negative_history_rejected(self) -> None:
        cell = self._cell(lateral_inhibition=1.0)
        with torch.no_grad():
            cell.W_in.weight.zero_()
            cell.W_in.bias.zero_()
        st = self._state(cell)
        cell._spike_history = -torch.ones(1, 4)
        with pytest.raises(ValueError, match="0, 1|\\[0, 1\\]"):
            cell(torch.zeros(1, 4), tuple(st))

    def test_valid_state_and_gradient_accepted(self) -> None:
        cell = self._cell(tau_fac=20.0, tau_rec=100.0,
                          lateral_inhibition=0.5)
        st = self._state(cell)
        st[0] = torch.zeros(1, 4, requires_grad=True)
        st[6] = torch.full((1, 4), 0.5, requires_grad=True)
        cell._spike_history = torch.full((1, 4), 0.25)
        inp = torch.zeros(1, 4, requires_grad=True)
        spike, out = cell(inp, tuple(st))
        assert torch.isfinite(spike).all()
        assert torch.isfinite(out[0]).all()
        (spike.sum() + out[0].sum()).backward()
        assert inp.grad is not None and torch.isfinite(inp.grad).all()

    def test_legacy_defaults_still_work(self) -> None:
        cell = self._cell()
        # Single-tensor legacy form (membrane only) and None.
        s1, _ = cell(torch.zeros(1, 4), torch.zeros(1, 4))
        s2, _ = cell(torch.zeros(1, 4), None)
        assert torch.isfinite(s1).all() and torch.isfinite(s2).all()


# ===============================================================
# r7 R3 — original raw carry physical domain before narrowing
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestR7RawOriginalDomain:

    def _model(self) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0,
                         lif_tau_fac=20.0, lif_tau_rec=100.0,
                         lif_rel_refract_ms=20.0).eval()

    def test_fraction_just_above_one_rejected_on_original(self) -> None:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = self._model()
        x = torch.randn(1, 4, 8)
        val = float(np.nextafter(1.0, 2.0))
        with pytest.raises(ValueError, match="0, 1|\\[0, 1\\]"):
            m(x, torch.tensor([0]), return_internals=True,
              states={"lif_x_resource": torch.full((1, 4), val,
                                                   dtype=torch.float64)})
        runner = NSMoRCoreJAX.from_torch(m)
        with pytest.raises(ValueError, match="0, 1|\\[0, 1\\]"):
            runner(x, torch.tensor([0]), return_internals=True,
                   states={"lif_x_resource": torch.full((1, 4), val,
                                                        dtype=torch.float64)})

    def test_negative_underflow_counter_rejected_on_original(self) -> None:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = self._model()
        x = torch.randn(1, 4, 8)
        val = -1e-50
        with pytest.raises(ValueError, match="non-negative"):
            m(x, torch.tensor([0]), return_internals=True,
              states={"lif_rel_refract": torch.full((1, 4), val,
                                                    dtype=torch.float64)})
        runner = NSMoRCoreJAX.from_torch(m)
        with pytest.raises(ValueError, match="non-negative"):
            runner(x, torch.tensor([0]), return_internals=True,
                   states={"lif_rel_refract": torch.full((1, 4), val,
                                                         dtype=torch.float64)})

    def test_valid_near_boundary_accepted(self) -> None:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = self._model()
        x = torch.randn(1, 4, 8)
        runner = NSMoRCoreJAX.from_torch(m)
        with torch.no_grad():
            _, _, carry = runner(
                x, torch.tensor([4]), return_internals=True,
                states={"lif_x_resource": torch.full((1, 4), 1.0,
                                                     dtype=torch.float64),
                        "lif_rel_refract": torch.zeros(1, 4,
                                                       dtype=torch.float64)},
            )
        assert torch.isfinite(carry["lif_x_resource"]).all()


# ===============================================================
# r7 R4 — parameter original + destination-representation validity
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestR7ParameterRepresentability:

    def _src(self) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()

    def _dest(self, m: NSMoRCore):
        from nsmor.jax.model import NSMoRModel

        lif = m.lif_cell
        return NSMoRModel(
            hidden_dim=4, dt_ms=4.0, lif_alpha=float(lif.alpha),
            lif_threshold=float(lif.v_threshold), lif_beta=float(lif.beta),
            lif_tau_syn=float(lif.tau_syn), lif_tau_w=float(lif.tau_w),
            lif_b_adapt=float(lif.b_adapt), lif_v_rest=float(lif.v_rest),
            lif_abs_refract_ms=float(lif.abs_refract_ms),
            lif_rel_refract_ms=float(lif.rel_refract_ms),
            lif_lateral_inhibition=float(lif.lateral_inhibition),
            lif_tbptt_steps=int(m.backend._tbptt_steps),
            dropout_rate=float(m.direction_head.dropout_rate),
        )

    def test_nonfinite_alias_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        sd = m.state_dict()
        for key in ("backend.direction_head.net.3.weight",
                    "direction_head.net.3.weight"):
            sd[key] = torch.full_like(sd[key], float("nan"))
        with pytest.raises(ValueError, match="nonfinite"):
            load_from_torch_state_dict(self._dest(m), sd, source=m)

    def test_finite_float64_overflow_rejected(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        sd = m.state_dict()
        for key in ("backend.direction_head.net.3.weight",
                    "direction_head.net.3.weight"):
            sd[key] = torch.full_like(sd[key].double(), 1e300)
        with pytest.raises(ValueError, match="representable|nonfinite"):
            load_from_torch_state_dict(self._dest(m), sd, source=m)

    def test_finite_float64_overflow_rejected_source_none(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        sd = m.state_dict()
        for key in ("backend.direction_head.net.3.weight",
                    "direction_head.net.3.weight"):
            sd[key] = torch.full_like(sd[key].double(), 1e300)
        # source=None still validates original finiteness + destination
        # representability (only the topology semantics are skipped).
        with pytest.raises(ValueError, match="representable|nonfinite"):
            load_from_torch_state_dict(self._dest(m), sd)

    def test_canonical_conversion_accepted(self) -> None:
        from nsmor.jax.model import load_from_torch_state_dict

        m = self._src()
        r = load_from_torch_state_dict(self._dest(m), m.state_dict(), source=m)
        assert "params" in r


# ===============================================================
# r7 R5 — actual final UQ output representability
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
class TestR7UQFinalRepresentability:

    @pytest.fixture(autouse=True)
    def _restore_x64(self):
        """Save the incoming ``jax_enable_x64`` flag and restore it afterwards.

        The tests below flip the global x64 flag during setup AND body; the
        teardown here runs on both the pass and the failure/exception path, so
        the process-wide flag never leaks (and is never reset to a hard-coded
        value different from the incoming one).
        """
        import jax

        prev = jax.config.read("jax_enable_x64")
        try:
            yield
        finally:
            jax.config.update("jax_enable_x64", prev)

    def _real_mixed_precision(self):
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax
        import jax.numpy as jnp
        from nsmor.model_nsmor_core import NSMoRCore
        from nsmor.analysis.jax_eval import JAXEvalWrapper
        from nsmor.analysis.uq_jax import MCDropoutAnalyzerJAX

        jax.config.update("jax_enable_x64", False)
        torch.manual_seed(193)
        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.5).eval()
        wrapper = JAXEvalWrapper.from_torch(m)
        params = wrapper._params
        model = wrapper._jax_model
        jax.config.update("jax_enable_x64", True)
        dense = params["params"]["direction_head"]["dense"]
        dense["kernel"] = dense["kernel"].astype(jnp.float64) * (2.0 ** 150)
        dense["bias"] = dense["bias"].astype(jnp.float64)
        return MCDropoutAnalyzerJAX(model, params, n_samples=8, seed=2)

    def test_unrepresentable_uncertainty_fails_closed(self) -> None:
        import jax.numpy as jnp

        analyzer = self._real_mixed_precision()
        # Finite frame means/stds (float64) but a float32-unrepresentable
        # per-trial uncertainty must fail closed, not return Inf.
        with pytest.raises(ValueError, match="representable|nonfinite"):
            analyzer.uncertainty_per_trial(
                jnp.ones((1, 3, 8), jnp.float32), jnp.array([3], jnp.int32),
            )

    def test_representable_large_value_control(self) -> None:
        import jax
        import jax.numpy as jnp
        from nsmor.jax.model import NSMoRModel
        from nsmor.analysis.uq_jax import MCDropoutAnalyzerJAX

        jax.config.update("jax_enable_x64", False)
        model = NSMoRModel(hidden_dim=4, dt_ms=4.0, dropout_rate=0.5,
                           persistence_skip=1.0, lif_lateral_inhibition=0.0)
        xx = jnp.ones((1, 9, 8), dtype=jnp.float32)
        LL = jnp.array([9], dtype=jnp.int32)
        params = model.init(jax.random.PRNGKey(0), xx, LL)
        analyzer = MCDropoutAnalyzerJAX(model, params, n_samples=8, seed=0)
        B, T = 1, 9
        y_mean = np.full((B, T), 5e37, dtype=np.float32)
        y_std = np.full((B, T), 1e-1, dtype=np.float32)
        analyzer.predict = lambda x, lengths: {
            "y_mean": y_mean, "y_std": y_std, "y_samples": y_mean[None],
            "attribution": {"dispersion": "epistemic_dropout_only",
                            "epistemic": True},
        }
        res = analyzer.uncertainty_per_trial(np.zeros((B, T, 8), np.float32),
                                             np.array([T], np.int64))
        assert np.isfinite(res["trial_uncertainty"]).all()
        assert res["trial_uncertainty"][0] > 0.0
        assert np.isfinite(res["trial_cv"]).all()



