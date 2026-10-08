"""Focused regressions for architecture v1: adaptive latent refinement.

Synthetic, deterministic, in-memory.  No formal data, checkpoints or nested
artifacts.  Each test encodes a concrete specification from the whole-stage
scope so the mechanism cannot silently regress.

Covered:
* off / fixed / adaptive modes; strict internals keys (enabled modes only).
* ACT full-prefix output ``sum_{j<N} p_j u_j + R u_N`` (NOT two-state mix).
* max-K mass fully allocated; weights nonnegative and sum to one; padding 0.
* K=1 limit and immediate-halt documented zero-halt-gradient behavior.
* real active-row gather counters (``refinement_updates``).
* padding / empty identity (no recursion on padding; all-empty ponder zero).
* one external timestep = one LIF/GRU advance (refinement adds no temporal
  carry: state outputs are unchanged between off and enabled modes).
* task and cost gradients reach the shared halt head on a depth>1 case.
* ACT ponder value pinned to ``N + R`` and its gradient sign (negative wrt the
  halt bias) checked on a crossing row.
* default preservation pinned against a ``git show HEAD`` baseline: the default
  loss is bitwise-equal to the HEAD reference and off-mode construction consumes
  exactly the HEAD baseline RNG (no extra draws, no off-mode parameter keys).
* loss seam: configured cost reaches optimizer; missing/nonfinite ponder refused.
* config trust boundaries: bool/NaN/illegal mode, max steps, eps, cost.
* backend refusal (each guarded by a JAX-availability skip): raw JAX, Flax,
  JAXEvalWrapper, and full-system Jacobian, each keyed on the EXECUTED module.
"""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from nsmor.config_parser import ExperimentConfig
from nsmor.loss import BioJointLoss
from nsmor.model_nsmor_core import AdaptiveLatentRefinement, NSMoRCore

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    JAX_AVAILABLE = False

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_head_module(relpath: str, name: str):
    """Load the ``git show HEAD:<relpath>`` source as an importable module.

    Default-preservation guards must compare against a HISTORICAL reference,
    not the new code against itself (finding A8).  The reference is read from
    the actual git object so the pinned baseline cannot drift.
    """
    source = subprocess.check_output(
        ["git", "show", f"HEAD:{relpath}"], cwd=str(_REPO_ROOT), text=True,
    )
    tmp = Path(tempfile.mkdtemp()) / Path(relpath).name
    tmp.write_text(source)
    spec = importlib.util.spec_from_file_location(name, tmp)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _model(mode: str, **kw) -> NSMoRCore:
    return NSMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
        refinement_mode=mode, **kw,
    ).eval()


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    x = torch.randn(2, 5, 8)
    lengths = torch.tensor([5, 3], dtype=torch.long)
    return x, lengths


# ===============================================================
# Modes and internals keys
# ===============================================================

class TestModesAndInternals:
    def test_off_has_no_refinement_and_no_keys(self) -> None:
        m = _model("off")
        assert m.backend.refinement is None
        x, lengths = _batch()
        _, internals = m(x, lengths, return_internals=True)
        for key in (
            "refined_hidden", "refinement_depth", "refinement_updates",
            "refinement_ponder_cost", "refinement_weights",
        ):
            assert key not in internals, f"off must not expose {key}"

    def test_enabled_exposes_all_keys(self) -> None:
        for mode in ("fixed", "adaptive"):
            m = _model(mode, refinement_max_steps=3)
            x, lengths = _batch()
            y, internals = m(x, lengths, return_internals=True)
            assert torch.isfinite(y).all()
            assert internals["refined_hidden"].shape == (2, 5, 8)
            assert internals["refinement_depth"].shape == (2, 5)
            assert internals["refinement_weights"].shape == (2, 5, 3)
            assert internals["refinement_updates"].dtype == torch.long

    def test_off_vs_enabled_state_outputs_identical(self) -> None:
        """One external timestep advances LIF/GRU exactly once: the recurrent
        carry (states_out) is unchanged by enabling refinement."""
        x, lengths = _batch()
        m_off = _model("off")
        m_fix = _model("fixed", refinement_max_steps=3)
        # Copy all shared weights so the recurrence is identical.
        m_fix.load_state_dict(m_off.state_dict(), strict=False)
        _, _, s_off = m_off(x, lengths, return_internals=True, states={})
        _, _, s_fix = m_fix(x, lengths, return_internals=True, states={})
        for key in s_off:
            assert torch.allclose(s_off[key], s_fix[key]), key


# ===============================================================
# ACT equations
# ===============================================================

class TestACTEquations:
    def _forced(self, p: float, K: int, eps: float) -> AdaptiveLatentRefinement:
        ref = AdaptiveLatentRefinement(8, mode="adaptive", max_steps=K, eps=eps)
        ref.eval()
        with torch.no_grad():
            ref.halt.weight.zero_()
            ref.halt.bias.fill_(torch.logit(torch.tensor(p)).item())
        return ref

    def test_full_prefix_not_two_state(self) -> None:
        K, eps, p = 4, 0.05, 0.4  # cum 0.4,0.8,1.2 -> N=3
        ref = self._forced(p, K, eps)
        u0 = torch.randn(1, 1, 8)
        valid = torch.ones(1, 1, dtype=torch.bool)
        out, depth, _, _, weights = ref(u0, valid)
        us = [u0.reshape(1, 8)]
        cur = us[0]
        for _ in range(K):
            cur = ref._block(cur)
            us.append(cur)
        N = 3
        R = 1.0 - (N - 1) * p
        expected = sum(p * us[j] for j in range(1, N)) + R * us[N]
        assert int(depth[0, 0]) == N
        assert torch.allclose(out.reshape(1, 8), expected, atol=1e-5)
        wrong = p * us[N - 1] + R * us[N]
        assert not torch.allclose(out.reshape(1, 8), wrong, atol=1e-3)
        assert abs(float(weights.sum().detach()) - 1.0) < 1e-5
        assert (weights >= -1e-6).all()

    def test_max_depth_remainder_fully_allocated(self) -> None:
        # p small -> never crosses -> N=K, R = 1 - sum_{j<K} p_j.
        K, eps, p = 3, 0.01, 0.05
        ref = self._forced(p, K, eps)
        u0 = torch.randn(1, 1, 8)
        valid = torch.ones(1, 1, dtype=torch.bool)
        out, depth, _, _, weights = ref(u0, valid)
        us = [u0.reshape(1, 8)]
        cur = us[0]
        for _ in range(K):
            cur = ref._block(cur)
            us.append(cur)
        R = 1.0 - (K - 1) * p
        expected = sum(p * us[j] for j in range(1, K)) + R * us[K]
        assert int(depth[0, 0]) == K
        assert torch.allclose(out.reshape(1, 8), expected, atol=1e-5)
        assert abs(float(weights.sum().detach()) - 1.0) < 1e-5

    def test_k1_immediate_halt_is_identity(self) -> None:
        ref = self._forced(0.9, 1, 0.05)
        u0 = torch.randn(1, 1, 8)
        valid = torch.ones(1, 1, dtype=torch.bool)
        out, depth, _, ponder, weights = ref(u0, valid)
        # K=1: N=1 -> out = u_1 (one block application), ponder = 1 + R.
        us1 = ref._block(u0.reshape(1, 8))
        assert int(depth[0, 0]) == 1
        assert torch.allclose(out.reshape(1, 8), us1, atol=1e-6)
        assert abs(float(weights.sum().detach()) - 1.0) < 1e-6

    def test_weights_padding_zero(self) -> None:
        m = _model("adaptive", refinement_max_steps=3)
        x, lengths = _batch()
        _, internals = m(x, lengths, return_internals=True)
        valid = torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)
        w = internals["refinement_weights"]
        assert (w[~valid] == 0).all()
        assert (internals["refinement_depth"][~valid] == 0).all()
        wsum = w.sum(-1)
        assert torch.allclose(wsum[valid], torch.ones_like(wsum[valid]), atol=1e-5)


# ===============================================================
# Real gather / counters / gradients
# ===============================================================

class TestGatherCountersAndGradients:
    def test_fixed_updates_equal_n_valid_times_k(self) -> None:
        m = _model("fixed", refinement_max_steps=3)
        x, lengths = _batch()
        _, internals = m(x, lengths, return_internals=True)
        n_valid = int(lengths.sum())
        assert int(internals["refinement_updates"]) == n_valid * 3

    def test_adaptive_updates_leq_fixed(self) -> None:
        # Immediate halting should spend strictly fewer row-updates than all-K.
        ref = AdaptiveLatentRefinement(8, mode="adaptive", max_steps=5, eps=0.05)
        ref.eval()
        with torch.no_grad():
            ref.halt.weight.zero_()
            ref.halt.bias.fill_(torch.logit(torch.tensor(0.9)).item())
        u0 = torch.randn(4, 3, 8)
        valid = torch.ones(4, 3, dtype=torch.bool)
        _, depth, updates, _, _ = ref(u0, valid)
        assert int(updates) == int(depth.sum())
        assert int(updates) < 4 * 3 * 5

    def test_task_and_cost_gradients_reach_halt_head(self) -> None:
        ref = AdaptiveLatentRefinement(8, mode="adaptive", max_steps=5, eps=0.02)
        with torch.no_grad():
            ref.halt.weight.zero_()
            ref.halt.bias.fill_(torch.logit(torch.tensor(0.3)).item())
        u0 = torch.randn(2, 3, 8, requires_grad=True)
        valid = torch.ones(2, 3, dtype=torch.bool)
        out, depth, _, _, _ = ref(u0, valid)
        assert int(depth[0, 0]) > 1, "need a nondegenerate depth>1 case"
        target = torch.randn(2, 3, 8)
        ((out - target) ** 2).mean().backward()
        assert ref.halt.bias.grad is not None
        assert float(ref.halt.bias.grad.abs()) > 0.0
        ref.zero_grad()
        u0b = u0.detach().clone().requires_grad_(True)
        _, _, _, ponder, _ = ref(u0b, valid)
        ponder.backward()
        assert ref.halt.bias.grad is not None
        assert float(ref.halt.bias.grad.abs()) > 0.0


# ===============================================================
# Padding / empty identity
# ===============================================================

class TestPaddingAndEmpty:
    def test_all_empty_ponder_is_differentiable_zero(self) -> None:
        ref = AdaptiveLatentRefinement(8, mode="adaptive", max_steps=3)
        u0 = torch.randn(2, 4, 8, requires_grad=True)
        valid = torch.zeros(2, 4, dtype=torch.bool)
        out, depth, updates, ponder, weights = ref(u0, valid)
        assert (depth == 0).all()
        assert int(updates) == 0
        assert float(ponder.detach()) == 0.0
        # All-empty ponder stays connected to the graph (no leaf-less scalar).
        assert ponder.grad_fn is not None
        ponder.backward()
        assert out.shape == (2, 4, 8)

    def test_padding_does_not_affect_valid_rows(self) -> None:
        ref = AdaptiveLatentRefinement(8, mode="fixed", max_steps=3)
        u0 = torch.randn(2, 4, 8)
        v_all = torch.ones(2, 4, dtype=torch.bool)
        v_pad = torch.ones(2, 4, dtype=torch.bool)
        v_pad[0, 2:] = False
        out_all, _, _, _, _ = ref(u0, v_all)
        out_pad, _, _, _, _ = ref(u0, v_pad)
        assert torch.allclose(out_all[1], out_pad[1])
        assert torch.allclose(out_all[0, :2], out_pad[0, :2])


# ===============================================================
# Loss seam
# ===============================================================

class TestLossSeam:
    def test_default_matches_head_baseline(self) -> None:
        """Default loss must be bitwise identical to the HEAD reference.

        Finding A8: comparing the new loss with ITSELF is tautological.  The
        reference here is the historical ``nsmor/loss.py`` loaded from
        ``git show HEAD``; any silent change to the default path (e.g.
        ``x*1.001+1e-3``) makes this fail.
        """
        head_loss = _load_head_module("nsmor/loss.py", "head_loss_a8")
        crit = BioJointLoss()
        crit_head = head_loss.BioJointLoss()
        torch.manual_seed(3)
        yp = torch.randn(2, 5)
        yt = torch.randn(2, 5)
        g = torch.rand(2, 5, 1)
        lengths = torch.tensor([5, 3])
        a = crit(yp, yt, lengths, g, lambda_reg=0.2)
        ref = crit_head(yp, yt, lengths, g, lambda_reg=0.2)
        assert torch.equal(a, ref), (
            f"default loss drifted from HEAD: {float(a)} != {float(ref)}"
        )
        # The new opt-in kwargs, at their defaults, still equal the baseline.
        b = crit(yp, yt, lengths, g, lambda_reg=0.2,
                 lambda_compute=0.0, refinement_ponder_cost=None)
        assert torch.equal(a, b)

    def test_configured_cost_reaches_optimizer(self) -> None:
        crit = BioJointLoss()
        yp = torch.randn(2, 5)
        yt = torch.randn(2, 5)
        g = torch.rand(2, 5, 1)
        lengths = torch.tensor([5, 3])
        ponder = torch.tensor(2.5, requires_grad=True)
        loss = crit(yp, yt, lengths, g, lambda_reg=0.2,
                    lambda_compute=0.7, refinement_ponder_cost=ponder)
        loss.backward()
        assert abs(float(ponder.grad) - 0.7) < 1e-6

    def test_missing_or_nonfinite_ponder_refused(self) -> None:
        crit = BioJointLoss()
        yp = torch.randn(2, 5)
        yt = torch.randn(2, 5)
        g = torch.rand(2, 5, 1)
        lengths = torch.tensor([5, 3])
        with pytest.raises(ValueError):
            crit(yp, yt, lengths, g, lambda_compute=0.7,
                 refinement_ponder_cost=None)
        with pytest.raises(ValueError):
            crit(yp, yt, lengths, g, lambda_compute=0.7,
                 refinement_ponder_cost=torch.tensor(float("nan")))


# ===============================================================
# Default-preservation vs historical HEAD baseline (finding A8)
# ===============================================================

class TestDefaultPreservationVsHead:
    def test_off_construction_consumes_no_extra_rng(self) -> None:
        """Off-mode construction must consume exactly the HEAD baseline RNG.

        Finding A8: the G6 off-mode test never inspected RNG.  The historical
        model draws RNG for its own parameter initialization; the off-mode
        model must consume EXACTLY the same draws (bit-identical post-state).
        Any extra construction-time draw -- e.g. a silently-built refinement
        module -- shifts the RNG state and is caught.  The enabled path is the
        positive control that DOES consume extra RNG.
        """
        head_core = _load_head_module(
            "nsmor/model_nsmor_core.py", "head_core_a8_rng",
        )
        kwargs = dict(
            sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
        )
        torch.manual_seed(1234)
        m = NSMoRCore(refinement_mode="off", **kwargs)
        rng_new = torch.get_rng_state()
        torch.manual_seed(1234)
        ref = head_core.NSMoRCore(**kwargs)
        rng_head = torch.get_rng_state()
        assert torch.equal(rng_new, rng_head), (
            "off-mode construction consumed RNG beyond the HEAD baseline"
        )
        assert m.backend.refinement is None
        # Positive control: an enabled model draws extra RNG (the module init).
        torch.manual_seed(1234)
        NSMoRCore(refinement_mode="fixed", refinement_max_steps=3, **kwargs)
        rng_enabled = torch.get_rng_state()
        assert not torch.equal(rng_enabled, rng_head), (
            "enabled construction should consume extra RNG (control)"
        )

    def test_off_state_dict_matches_head_baseline(self) -> None:
        """Seeded off-mode parameters must equal the HEAD reference exactly.

        Finding A8: the reference is the historical ``nsmor/model_nsmor_core``
        from ``git show HEAD``, so an accidental default change in the shared
        weights is detected.  The refinement machinery must add no off-mode
        parameter keys.
        """
        head_core = _load_head_module(
            "nsmor/model_nsmor_core.py", "head_core_a8",
        )
        torch.manual_seed(1234)
        new = NSMoRCore(
            sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
            refinement_mode="off",
        )
        torch.manual_seed(1234)
        ref = head_core.NSMoRCore(
            sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
        )
        sd_new, sd_ref = new.state_dict(), ref.state_dict()
        assert set(sd_new) == set(sd_ref), (
            f"off-mode key drift: extra={sorted(set(sd_new) - set(sd_ref))}, "
            f"missing={sorted(set(sd_ref) - set(sd_new))}"
        )
        for key in sd_new:
            assert torch.equal(sd_new[key], sd_ref[key]), (
                f"off-mode parameter {key!r} drifted from HEAD"
            )
        assert not any("refinement" in n for n, _ in new.named_parameters())


# ===============================================================
# ACT ponder value and gradient sign (finding A13)
# ===============================================================

class TestACTPonderValueAndGradient:
    def _enabled(self, p: float, K: int, eps: float) -> AdaptiveLatentRefinement:
        ref = AdaptiveLatentRefinement(8, mode="adaptive", max_steps=K, eps=eps)
        ref.eval()
        with torch.no_grad():
            ref.halt.weight.zero_()
            ref.halt.bias.fill_(torch.logit(torch.tensor(p)).item())
        return ref

    def test_ponder_equals_n_plus_r(self) -> None:
        """``refinement_ponder_cost`` is the ACT surrogate ``N + R``.

        Finding A13: no test pinned the ponder VALUE.  For a single valid row
        the ponder must equal the recorded integer depth ``N`` plus the
        remainder mass ``R`` (the final-step weight).  A ponder returning
        ``-(N + R)`` (or ``R`` alone) fails.
        """
        ref = self._enabled(0.4, 5, 0.05)
        u0 = torch.randn(1, 1, 8)
        valid = torch.ones(1, 1, dtype=torch.bool)
        _out, depth, _updates, ponder, weights = ref(u0, valid)
        n = int(depth[0, 0])
        r = float(weights[0, 0, n - 1].detach())
        pv = float(ponder.detach())
        assert abs(pv - (n + r)) < 1e-5, f"ponder {pv} != N+R = {n + r}"
        # A batch mean must equal the mean of the per-row N+R surrogate.
        u1 = torch.randn(3, 4, 8)
        valid1 = torch.ones(3, 4, dtype=torch.bool)
        _o, d1, _u, ponder1, w1 = ref(u1, valid1)
        n1 = d1.float()
        r1 = torch.gather(
            w1, -1, (d1 - 1).clamp(min=0).unsqueeze(-1),
        ).squeeze(-1)
        assert abs(float(ponder1.detach()) - float((n1 + r1).mean())) < 1e-5

    def test_ponder_gradient_sign_negative(self) -> None:
        """For a crossing row the ACT surrogate is ``N + R`` with
        ``R = 1 - sum_{j<N} p_j`` and ``N`` detached.  Raising the halt bias
        raises ``p`` and lowers ``R``, so ``d(ponder)/d(halt.bias)`` is
        NEGATIVE: a positive compute penalty pushes the optimizer to halt
        earlier (more compute-saving), never to add depth.

        Finding A13: the SIGN was never pinned.  A ponder returning the wrong
        quantity (e.g. a negated or depth-inverted surrogate) flips this.
        """
        ref = self._enabled(0.3, 5, 0.02)
        valid = torch.ones(1, 1, dtype=torch.bool)
        u0 = torch.randn(1, 1, 8)
        _o, depth, _u, ponder, weights = ref(u0, valid)
        n = int(depth[0, 0])
        assert n > 1, "need a nondegenerate crossing depth for the sign check"
        assert n < 5, "need a CROSSING row (R differentiable); never-crossed R=1"
        ponder.backward()
        g = float(ref.halt.bias.grad)
        assert g < 0.0, f"d ponder/d halt.bias must be negative, got {g}"
        # K=1 / immediate halt is the documented ZERO-halt-gradient case: R=1
        # is constant, so the sign check above is meaningful only for N>1.
        ref1 = self._enabled(0.3, 1, 0.02)
        _o1, _d1, _u1, ponder1, _w1 = ref1(torch.randn(1, 1, 8), valid)
        ponder1.backward()
        assert abs(float(ref1.halt.bias.grad)) < 1e-9


# ===============================================================
# Config trust boundaries
# ===============================================================

class TestConfigTrustBoundaries:
    def test_illegal_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            ExperimentConfig(model=type(ExperimentConfig().model)(
                refinement_mode="bogus"))

    def test_bool_and_bad_max_steps_rejected(self) -> None:
        cfg = ExperimentConfig()
        cfg.model.refinement_max_steps = True
        with pytest.raises(ValueError):
            cfg.model.__post_init__()
        cfg = ExperimentConfig()
        cfg.model.refinement_max_steps = 0
        with pytest.raises(ValueError):
            cfg.model.__post_init__()

    def test_bad_eps_and_cost_rejected(self) -> None:
        cfg = ExperimentConfig()
        cfg.model.refinement_eps = float("nan")
        with pytest.raises(ValueError):
            cfg.model.__post_init__()
        cfg = ExperimentConfig()
        cfg.model.refinement_eps = 1.0
        with pytest.raises(ValueError):
            cfg.model.__post_init__()
        cfg = ExperimentConfig()
        cfg.model.refinement_update_scale = -1.0
        with pytest.raises(ValueError):
            cfg.model.__post_init__()

    def test_adaptive_requires_positive_compute_cost(self) -> None:
        cfg = ExperimentConfig()
        cfg.model.refinement_mode = "adaptive"
        cfg.loss.lambda_compute = 0.0
        with pytest.raises(ValueError):
            cfg.validate()
        cfg.loss.lambda_compute = 0.01
        cfg.validate()  # now accepted

    def test_compute_cost_without_module_rejected(self) -> None:
        cfg = ExperimentConfig()
        cfg.model.refinement_mode = "off"
        cfg.loss.lambda_compute = 0.01
        with pytest.raises(ValueError):
            cfg.validate()

    def test_bool_lambda_compute_rejected(self) -> None:
        cfg = ExperimentConfig()
        cfg.loss.lambda_compute = True
        with pytest.raises(ValueError):
            cfg.validate()


# ===============================================================
# Backend refusal
# ===============================================================

class TestBackendRefusal:
    """R1: refusals must key on the EXECUTED module, not the mode string.

    Each entry point is exercised with a model whose ``refinement_mode`` was
    mutated to ``"off"`` AFTER construction while the refinement module is
    still present -- the exact live state the reviewers constructed.  A guard
    that reads only the stale string would accept these models and let the
    fused kernel silently drop refinement.
    """

    @staticmethod
    def _mutated(mode: str = "fixed", **kw):
        m = _model(mode, **kw)
        # Post-construction mutation: the executed module stays present while
        # BOTH mode aliases read "off" (reviewers' repro).  Any string-keyed
        # guard accepts this model; only an executed-child guard refuses it.
        m.refinement_mode = "off"
        m.backend.refinement_mode = "off"
        return m

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_raw_jax_refuses_module_present_after_mode_mutation(self) -> None:
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = self._mutated("adaptive")
        assert m.backend.refinement is not None
        assert m.backend.refinement_mode == "off"
        with pytest.raises(ValueError):
            NSMoRCoreJAX.from_torch(m)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_flax_refuses_module_present_after_mode_mutation(self) -> None:
        from nsmor.jax.model import assert_flax_supported

        with pytest.raises(ValueError):
            assert_flax_supported(self._mutated("fixed"), context="test")

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_jax_eval_wrapper_refuses_after_mode_mutation(self) -> None:
        from nsmor.analysis.jax_eval import JAXEvalWrapper

        with pytest.raises(ValueError):
            JAXEvalWrapper.from_torch(self._mutated("adaptive"))

    def test_full_system_jacobian_refused_after_mode_mutation(self) -> None:
        from nsmor.analysis.dynamics import FixedPointAdapter

        adapter = FixedPointAdapter(self._mutated("fixed"))
        X = torch.randn(1, 1, 8)
        with pytest.raises(ValueError):
            adapter.compute_full_system_jacobian(X, torch.tensor([1]))

    def test_guard_helper_matches_executed_module(self) -> None:
        from nsmor.model_nsmor_core import refinement_module_present

        # Control: a genuinely off model carries no module.
        assert not refinement_module_present(_model("off"))
        assert refinement_module_present(_model("fixed"))
        assert refinement_module_present(self._mutated("adaptive"))


# ===============================================================
# Checkpoint rebuild (own checkpoint, synthetic)
# ===============================================================

class TestCheckpointRebuild:
    def test_state_dict_roundtrip_fixed_and_adaptive(self) -> None:
        for mode in ("fixed", "adaptive"):
            m = _model(mode, refinement_max_steps=3)
            x, lengths = _batch()
            with torch.no_grad():
                y0 = m(x, lengths)
            sd = copy.deepcopy(m.state_dict())
            m2 = _model(mode, refinement_max_steps=3)
            m2.load_state_dict(sd, strict=True)
            with torch.no_grad():
                y1 = m2(x, lengths)
            assert torch.allclose(y0, y1)
