"""Focused regressions for ADR 0009: time-consuming recursion (opt-in).

Synthetic, deterministic, in-memory.  No formal data, checkpoints or nested
artifacts.  Each test encodes a concrete specification from the phase-3
protocol (``docs/realdata-phase3-protocol-20261010.md``) and ADR 0009 so the
mechanism cannot silently regress.

Covered:
* off / depth_delay / accumulate modes; strict internals keys (enabled only).
* causality: perturbing the last frame never changes earlier outputs or the
  scored quantity; no future leak; ``out_valid`` frames receive a source at
  ``t <= tp``.
* emission collisions: mixed depth -> per-target arbitration (smallest delay
  wins, earliest source breaks ties); ``out``/``out_valid`` are consistent.
* the loss masks never-emitted frames via ``out_valid``; ``None`` is bitwise
  the historical length-only mask.
* depth -> delay mapping ``delay = delay_scale * (depth - 1)``; forced-halt
  depth and the ACT full-prefix value mixture.
* default preservation vs a genuine ``git show HEAD`` module: off-mode
  construction consumes exactly the HEAD baseline RNG, has the HEAD
  state_dict keys (both directions), and a HEAD state_dict loads ``strict=True``
  into the current off model and produces bitwise-identical outputs.
* one external timestep = one LIF/GRU advance (time-consuming adds no recurrent
  carry: ``states_out`` is unchanged between off and enabled).
* the integer depth is detached; task gradient reaches the shared halt head.
* padding / empty identity (depth 0, not emitted; no crash).
* config trust boundaries: bool/NaN/illegal mode, steps, eps, scales.
* train.py resume guard refuses a time_consuming_mode switch.
* backend refusal (each guarded by a JAX-availability skip): raw JAX, Flax and
  JAXEvalWrapper keyed on the EXECUTED module; full-system Jacobian refused.
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
from nsmor.model_nsmor_core import (
    NSMoRCore,
    TimeConsumingRecursion,
    _emit_delayed,
    time_consuming_module_present,
)

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    JAX_AVAILABLE = False

_REPO_ROOT = Path(__file__).resolve().parents[1]

_TC_KEYS = (
    "output_delay",
    "realized_delay",
    "time_consuming_depth",
    "time_consuming_updates",
    "time_consuming_max_delay",
    "out_valid",
    "time_consuming_hidden",
    "time_consuming_collisions",
    "time_consuming_truncated",
)


def _load_head_module(relpath: str, name: str):
    """Load the ``git show HEAD:<relpath>`` source as an importable module."""
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


def _model(mode: str = "off", **kw) -> NSMoRCore:
    return NSMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
        time_consuming_mode=mode, **kw,
    ).eval()


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    x = torch.randn(2, 6, 8)
    lengths = torch.tensor([6, 4], dtype=torch.long)
    return x, lengths


# ===============================================================
# Modes and internals keys
# ===============================================================

class TestModesAndInternals:
    def test_off_has_no_module_and_no_keys(self) -> None:
        m = _model("off")
        assert m.backend.time_consuming is None
        assert not time_consuming_module_present(m)
        x, lengths = _batch()
        _, internals = m(x, lengths, return_internals=True)
        for key in _TC_KEYS:
            assert key not in internals, f"off must not expose {key}"

    def test_enabled_exposes_all_keys(self) -> None:
        for mode in ("depth_delay", "accumulate"):
            m = _model(mode, time_consuming_max_steps=3)
            assert time_consuming_module_present(m)
            x, lengths = _batch()
            y, internals = m(x, lengths, return_internals=True)
            assert torch.isfinite(y).all()
            assert internals["time_consuming_depth"].shape == (2, 6)
            assert internals["output_delay"].shape == (2, 6)
            assert internals["out_valid"].shape == (2, 6)
            assert internals["realized_delay"].shape == (2, 6)
            assert internals["time_consuming_updates"].dtype == torch.long

    def test_deterministic_enabled(self) -> None:
        m = _model("depth_delay", time_consuming_max_steps=3)
        x, lengths = _batch()
        with torch.no_grad():
            a = m(x, lengths)
            b = m(x, lengths)
        assert torch.equal(a, b)

    def test_refined_hidden_not_aliased_by_time_consuming(self) -> None:
        """With both enabled, ``refined_hidden`` is the pure refinement readout
        and ``time_consuming_hidden`` the delayed emission (finding R2-B3)."""
        kwargs = dict(sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0)
        both = NSMoRCore(
            refinement_mode="fixed", refinement_max_steps=2,
            time_consuming_mode="depth_delay", time_consuming_max_steps=3,
            **kwargs,
        ).eval()
        only_ref = NSMoRCore(
            refinement_mode="fixed", refinement_max_steps=2, **kwargs,
        ).eval()
        both.load_state_dict(only_ref.state_dict(), strict=False)
        x, lengths = _batch()
        with torch.no_grad():
            _, ib = both(x, lengths, return_internals=True)
            _, ir = only_ref(x, lengths, return_internals=True)
        assert torch.allclose(ib["refined_hidden"], ir["refined_hidden"])
        assert "time_consuming_hidden" in ib
        assert ib["time_consuming_hidden"].shape == ib["refined_hidden"].shape
        assert int(ib["time_consuming_collisions"]) >= 0

    def test_threshold_bound_rejects_extrapolation(self) -> None:
        with pytest.raises(ValueError):
            TimeConsumingRecursion(8, mode="accumulate", threshold=1.5)
        # threshold == 1.0 is the boundary and is accepted.
        TimeConsumingRecursion(8, mode="accumulate", threshold=1.0)


# ===============================================================
# Causality
# ===============================================================

class TestCausality:
    def test_last_frame_perturbation_does_not_change_earlier_outputs(self) -> None:
        m = _model("accumulate", time_consuming_max_steps=3)
        x, lengths = _batch()
        with torch.no_grad():
            y0 = m(x, lengths)
        x2 = x.clone()
        x2[:, -1, :] = x2[:, -1, :] + 10.0
        with torch.no_grad():
            y1 = m(x2, lengths)
        # Frames strictly before the last are unaffected.
        assert torch.allclose(y0[:, :-1], y1[:, :-1], atol=1e-6)

    def test_out_valid_receives_source_at_or_after_itself(self) -> None:
        for mode in ("depth_delay", "accumulate"):
            m = _model(mode, time_consuming_max_steps=3)
            x, lengths = _batch()
            _, ints = m(x, lengths, return_internals=True)
            ov = ints["out_valid"]
            delay = ints["output_delay"]
            # ``output_delay`` is indexed by SOURCE frame t: source t emits to
            # ``t + delay[t]``.  Every valid source whose target is inside the
            # trial must have produced an emitted frame there.
            for b in range(ov.shape[0]):
                L = int(lengths[b])
                for t in range(L):
                    tp = t + int(round(float(delay[b, t])))
                    assert tp >= t  # causal: never earlier
                    if tp < L:
                        assert bool(ov[b, tp])

    def test_emission_collision_is_deterministic_and_injective(self) -> None:
        """Mixed depth -> two sources target the same frame; the smallest
        delay wins and out_valid counts exactly the surviving targets
        (finding R2-B1)."""
        B, T, H = 1, 8, 4
        refined = torch.randn(B, T, H)
        # delay 3 at t=0..3, delay 0 at t=4..7.  Targets:
        # 3,4,5,6,4,5,6,7 -> distinct {3,4,5,6,7}; collisions on 4,5,6.
        source_delay = torch.tensor(
            [[3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0]],
        )
        valid = torch.ones(B, T, dtype=torch.bool)
        lengths = torch.tensor([T])
        out, out_valid, max_delay, n_coll, n_trunc, realized = _emit_delayed(
            refined, source_delay, valid, lengths,
        )
        # 5 surviving targets (source 3's target 6 loses to source 6; the deep
        # sources 1,2 lose targets 4,5 to the shallow sources 4,5).
        assert int(out_valid.sum()) == 5
        assert max_delay == 3
        assert n_coll == 3  # sources 1, 2, 3 lose to the shallow sources
        assert n_trunc == 0  # every target is inside the 8-frame trial
        # Target 3 has a single source (0).
        assert torch.allclose(out[0, 3], refined[0, 0])
        # Targets 4,5,6 are won by the smallest delay (sources 4,5,6).
        for tp, s in ((4, 4), (5, 5), (6, 6)):
            assert torch.allclose(out[0, tp], refined[0, s])
        assert bool(out_valid[0, 7]) is True  # source 7 -> target 7
        # out / out_valid are consistent: unwritten frames are exact zeros.
        unwritten = ~out_valid[0]
        assert (out[0][unwritten] == 0).all()
        # realized_delay is indexed by TARGET and reports the SURVIVING
        # source's delay: target 3 -> 3.0, targets 4,5,6 -> 0.0 (the shallow
        # sources 4,5,6 win), target 7 -> 0.0; unwritten frames -> 0.
        assert torch.allclose(realized[0, 3], torch.tensor(3.0))
        assert torch.allclose(realized[0, 4:8], torch.zeros(4))
        assert (realized[0][unwritten] == 0).all()

    def test_emission_no_overwrite_last_processed(self) -> None:
        """A collision is arbitrated by smallest delay, not iteration order."""
        B, T, H = 1, 6, 3
        refined = torch.randn(B, T, H)
        # t=0 delay 2 -> target 2 ; t=2 delay 0 -> target 2 (collision).
        # Smallest delay (source 2) wins.
        source_delay = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        valid = torch.ones(B, T, dtype=torch.bool)
        lengths = torch.tensor([T])
        out, out_valid, _md, n_coll, _nt, _realized = _emit_delayed(
            refined, source_delay, valid, lengths,
        )
        assert bool(out_valid[0, 2])
        assert torch.allclose(out[0, 2], refined[0, 2])  # delay 0 wins
        assert n_coll == 1  # source 0 loses its target 2 to source 2

    def test_scored_quantity_causality_no_future_leak(self) -> None:
        """Perturbing x[t+k] leaves the loss contribution at frames <= t
        unchanged: causality holds at the SCORED quantity (finding R1-B1)."""
        from nsmor.loss import BioJointLoss

        m = _model("depth_delay", time_consuming_max_steps=3)
        crit = BioJointLoss()
        torch.manual_seed(5)
        x = torch.randn(2, 8, 8)
        lengths = torch.tensor([8, 6], dtype=torch.long)
        y = torch.randn(2, 8)
        g = torch.full((2, 8, 1), 0.5)
        with torch.no_grad():
            yp0, i0 = m(x, lengths, return_internals=True)
            l0 = crit(yp0, y, lengths, g, out_valid=i0["out_valid"])
        x2 = x.clone()
        x2[:, -1, :] = x2[:, -1, :] + 10.0
        with torch.no_grad():
            yp1, i1 = m(x2, lengths, return_internals=True)
            l1 = crit(yp1, y, lengths, g, out_valid=i1["out_valid"])
        # The scored quantity (delayed emission) at frames <= 6 of trial 0 is
        # unchanged by a last-frame perturbation.
        assert torch.allclose(
            yp0[:, :6], yp1[:, :6], atol=1e-6,
        ), "future frame leaked into the scored quantity"
        m0 = i0["out_valid"][0]
        m1 = i1["out_valid"][0]
        assert torch.equal(m0[:6], m1[:6]), "emission mask changed before t"
        assert torch.isfinite(l0) and torch.isfinite(l1)

    def test_loss_masks_unemitted_frames(self) -> None:
        """``out_valid`` removes zero-filled, never-emitted frames from the
        MSE (findings R1-B2 / R2-B2)."""
        from nsmor.loss import BioJointLoss

        crit = BioJointLoss()
        y_pred = torch.zeros(1, 4)
        y_true = torch.ones(1, 4) * 5.0
        lengths = torch.tensor([4])
        ov = torch.tensor([[True, False, False, False]])
        g = torch.zeros(1, 4, 1)
        # Length-only mask: all 4 frames scored -> MSE = 25.
        l_len = crit(y_pred, y_true, lengths, g)
        assert float(l_len) == 25.0
        # out_valid mask: only frame 0 scored.  With frame 0 correct the MSE
        # must be 0 — the never-emitted frames must not contribute.
        y_pred2 = y_pred.clone()
        y_pred2[0, 0] = 5.0  # frame 0 correct
        l_ov = crit(y_pred2, y_true, lengths, g, out_valid=ov)
        assert float(l_ov) == 0.0, "unemitted frames leaked into the MSE"

    def test_out_valid_none_is_bitwise_length_only(self) -> None:
        from nsmor.loss import BioJointLoss

        crit = BioJointLoss()
        y_pred = torch.randn(3, 5)
        y_true = torch.randn(3, 5)
        lengths = torch.tensor([5, 3, 1])
        g = torch.rand(3, 5, 1)
        l_default = crit(y_pred, y_true, lengths, g)
        l_none = crit(y_pred, y_true, lengths, g, out_valid=None)
        assert torch.equal(l_default, l_none)

    def test_jerk_term_ignores_never_emitted_frames(self) -> None:
        """A never-emitted (zero-filled) trailing frame cannot change the
        frame-based third-difference smoothness contribution (finding R1-B1).

        The term is scored with ``lambda_reg=0`` so only the jerk term moves;
        a ``y_pred`` frame masked out by ``out_valid`` must be invisible.
        """
        from nsmor.loss import BioJointLoss

        crit = BioJointLoss()
        g = torch.zeros(1, 6, 1)
        lengths = torch.tensor([6])
        # A never-emitted tail: frames 3..5 did not receive an emission.
        ov = torch.tensor([[True, True, True, False, False, False]])
        base = torch.tensor([[0.0, 1.0, 4.0, 9.0, 16.0, 25.0]])
        y_true = base.clone()
        l_base = crit(base, y_true, lengths, g, lambda_reg=0.0,
                      lambda_jerk=1.0, out_valid=ov)
        # Perturb ONLY the never-emitted frames; the jerk term must not move.
        perturbed = base.clone()
        perturbed[0, 3:] = torch.tensor([100.0, -50.0, 7.0])
        l_pert = crit(perturbed, y_true, lengths, g, lambda_reg=0.0,
                      lambda_jerk=1.0, out_valid=ov)
        assert torch.allclose(l_base, l_pert), (
            f"never-emitted frames leaked into the jerk term: "
            f"{float(l_base)} vs {float(l_pert)}"
        )
        # With no mask the perturbation DOES change the term (sanity: the
        # test is not vacuous).
        l_len = crit(perturbed, y_true, lengths, g, lambda_reg=0.0,
                     lambda_jerk=1.0, out_valid=None)
        assert not torch.allclose(l_base, l_len)

    def test_mixed_length_batch_does_not_crash(self) -> None:
        """A batch with a partial-validity column (n_active neither 0 nor B)
        must not raise: ordinary padded real-data batches are mixed-length
        (finding R2-B1; the focused tests previously used n_active in {0, B})."""
        B, T, H = 4, 6, 8
        h = torch.randn(B, T, H)
        valid = torch.zeros(B, T, dtype=torch.bool)
        valid[0, :4] = True
        valid[1, :4] = True
        valid[2, :2] = True
        valid[3, :2] = True
        for mode in ("depth_delay", "accumulate"):
            mod = TimeConsumingRecursion(H, mode=mode, max_steps=4)
            out, depth, _upd, _delay, _md, ov, _nc, _nt, _rd = mod(h, valid)
            assert out.shape == (B, T, H)
            # Padding never recurses and is never emitted.
            assert (depth[~valid] == 0).all()
            assert not bool(ov[~valid].any())

    def test_model_mixed_length_batch_end_to_end(self) -> None:
        """The FULL model (not just the module) survives a mixed-length batch
        in both time-consuming modes (finding R2-B1 end-to-end)."""
        for mode in ("depth_delay", "accumulate"):
            m = _model(mode, time_consuming_max_steps=3)
            x = torch.randn(4, 6, 8)
            lengths = torch.tensor([6, 6, 3, 3], dtype=torch.long)
            with torch.no_grad():
                y_pred, ints = m(x, lengths, return_internals=True)
            assert y_pred.shape == (4, 6)
            assert torch.isfinite(y_pred).all()
            assert ints["out_valid"].shape == (4, 6)



# ===============================================================
# Depth -> delay mapping and the ACT value mixture
# ===============================================================

class TestDepthDelayAndValue:
    def _forced(self, mode: str, p: float, K: int, eps: float, thr: float):
        mod = TimeConsumingRecursion(
            8, mode=mode, max_steps=K, eps=eps, threshold=thr,
        )
        mod.eval()
        with torch.no_grad():
            mod.halt.weight.zero_()
            mod.halt.bias.fill_(torch.logit(torch.tensor(p)).item())
        return mod

    def test_delay_equals_scale_times_depth_minus_one(self) -> None:
        m = _model("depth_delay", time_consuming_max_steps=3,
                   time_consuming_delay_scale=2.0)
        x, lengths = _batch()
        _, ints = m(x, lengths, return_internals=True)
        valid = torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)
        d = ints["time_consuming_depth"].float()
        expected = (d - 1.0).clamp(min=0.0) * 2.0
        assert torch.allclose(ints["output_delay"][valid], expected[valid])
        assert (ints["output_delay"][~valid] == 0).all()
        assert (ints["time_consuming_depth"][~valid] == 0).all()

    def test_forced_halt_depth_and_delay(self) -> None:
        # p=0.4, eps=0.05 -> cum 0.4, 0.8, 1.2 -> crosses at k=3.
        mod = self._forced("depth_delay", 0.4, 4, 0.05, 1.0)
        u0 = torch.randn(1, 1, 8)
        valid = torch.ones(1, 1, dtype=torch.bool)
        _out, depth, _upd, delay, _md, _ov, _nc, _nt, _rd = mod(u0, valid)
        assert int(depth[0, 0]) == 3
        assert float(delay[0, 0]) == 2.0

    def test_delay_scale_applied_to_forced_depth(self) -> None:
        mod = self._forced("depth_delay", 0.4, 4, 0.05, 1.0)
        mod.delay_scale = 2.0
        u0 = torch.randn(1, 1, 8)
        valid = torch.ones(1, 1, dtype=torch.bool)
        _out, depth, _upd, delay, _md, _ov, _nc, _nt, _rd = mod(u0, valid)
        assert int(depth[0, 0]) == 3
        assert float(delay[0, 0]) == 4.0

    def test_value_is_full_prefix_act_mixture(self) -> None:
        K, eps, p = 4, 0.05, 0.4  # cum 0.4, 0.8, 1.2 -> N=3
        mod = self._forced("depth_delay", p, K, eps, 1.0)
        # T long enough that source frame 0 (delay 2) is not truncated.
        u0 = torch.randn(1, 6, 8)
        valid = torch.ones(1, 6, dtype=torch.bool)
        out, depth, _upd, delay, _md, _ov, _nc, _nt, _rd = mod(u0, valid)
        u_src = u0[0, 0, :].reshape(1, 8)
        us = [u_src]
        cur = us[0]
        for _ in range(K):
            cur = mod._block(cur)
            us.append(cur)
        N = 3
        R = 1.0 - (N - 1) * p
        expected = sum(p * us[j] for j in range(1, N)) + R * us[N]
        assert int(depth[0, 0]) == N
        tp = int(round(float(delay[0, 0])))
        assert tp == N - 1  # delay_scale == 1
        assert torch.allclose(out[0, tp, :], expected.reshape(8), atol=1e-5)

    def test_updates_equal_n_valid_times_k(self) -> None:
        m = _model("accumulate", time_consuming_max_steps=3)
        x, lengths = _batch()
        _, ints = m(x, lengths, return_internals=True)
        assert int(ints["time_consuming_updates"]) == int(lengths.sum()) * 3

    def test_halt_gradient_reaches_head(self) -> None:
        mod = TimeConsumingRecursion(8, mode="depth_delay", max_steps=4, eps=0.05)
        with torch.no_grad():
            mod.halt.weight.zero_()
            mod.halt.bias.fill_(torch.logit(torch.tensor(0.4)).item())
        u0 = torch.randn(2, 3, 8)
        valid = torch.ones(2, 3, dtype=torch.bool)
        out, _depth, _upd, _delay, _md, _ov, _nc, _nt, _rd = mod(u0, valid)
        ((out - torch.randn_like(out)) ** 2).mean().backward()
        assert mod.halt.bias.grad is not None
        assert float(mod.halt.bias.grad.abs()) > 0.0


# ===============================================================
# Default preservation vs historical HEAD baseline
# ===============================================================

class TestDefaultPreservationVsHead:
    def test_off_construction_consumes_no_extra_rng(self) -> None:
        head_core = _load_head_module(
            "nsmor/model_nsmor_core.py", "head_core_tc_rng",
        )
        kwargs = dict(sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0)
        torch.manual_seed(1234)
        m = NSMoRCore(time_consuming_mode="off", **kwargs)
        rng_new = torch.get_rng_state()
        torch.manual_seed(1234)
        ref = head_core.NSMoRCore(**kwargs)
        rng_head = torch.get_rng_state()
        assert torch.equal(rng_new, rng_head), (
            "off-mode construction consumed RNG beyond the HEAD baseline"
        )
        assert m.backend.time_consuming is None
        # Positive control: an enabled model draws extra RNG.
        torch.manual_seed(1234)
        NSMoRCore(time_consuming_mode="depth_delay", **kwargs)
        assert not torch.equal(torch.get_rng_state(), rng_head)

    def test_off_state_dict_matches_head_and_loads(self) -> None:
        head_core = _load_head_module(
            "nsmor/model_nsmor_core.py", "head_core_tc_sd",
        )
        kwargs = dict(sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0)
        torch.manual_seed(7)
        new = NSMoRCore(time_consuming_mode="off", **kwargs).eval()
        # The HEAD reference is built from the ACTUAL ``git show HEAD`` module,
        # NOT the current repo module (finding R1-B2: building both sides from
        # the current module made the test tautological).
        torch.manual_seed(7)
        ref = head_core.NSMoRCore(**kwargs).eval()
        sd_new, sd_ref = new.state_dict(), ref.state_dict()
        # Key sets must be EQUAL in BOTH directions (extra AND missing/renamed).
        assert set(sd_new) == set(sd_ref), (
            f"off-mode key drift: extra={sorted(set(sd_new) - set(sd_ref))}, "
            f"missing={sorted(set(sd_ref) - set(sd_new))}"
        )
        assert not any("time_consuming" in n for n, _ in new.named_parameters())
        # Every shared parameter must be bitwise equal (same construction RNG).
        for name in sd_ref:
            assert torch.equal(sd_new[name], sd_ref[name]), name
        # A genuine HEAD state_dict loads strict=True into the current off
        # model and produces bitwise-identical outputs.
        new2 = NSMoRCore(time_consuming_mode="off", **kwargs).eval()
        new2.load_state_dict(sd_ref, strict=True)
        x, lengths = _batch()
        with torch.no_grad():
            y_head = ref(x, lengths)
            y_loaded = new2(x, lengths)
        assert torch.equal(y_head, y_loaded)

    def test_state_outputs_unchanged_by_enabling(self) -> None:
        """One external timestep advances LIF/GRU once: enabling the module
        does not change the recurrent carry (states_out)."""
        x, lengths = _batch()
        m_off = _model("off")
        m_tc = _model("depth_delay", time_consuming_max_steps=3)
        m_tc.load_state_dict(m_off.state_dict(), strict=False)
        _, _, s_off = m_off(x, lengths, return_internals=True, states={})
        _, _, s_tc = m_tc(x, lengths, return_internals=True, states={})
        for key in s_off:
            assert torch.allclose(s_off[key], s_tc[key]), key


# ===============================================================
# Padding / empty identity
# ===============================================================

class TestPaddingAndEmpty:
    def test_padding_depth_and_delay_zero(self) -> None:
        m = _model("depth_delay", time_consuming_max_steps=3)
        x, lengths = _batch()
        _, ints = m(x, lengths, return_internals=True)
        valid = torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)
        assert (ints["time_consuming_depth"][~valid] == 0).all()
        assert (ints["output_delay"][~valid] == 0).all()
        assert not bool(ints["out_valid"][~valid].any())

    def test_all_empty_is_finite_no_crash(self) -> None:
        m = _model("accumulate", time_consuming_max_steps=2)
        xe = torch.randn(2, 5, 8)
        le = torch.zeros(2, dtype=torch.long)
        y, ints = m(xe, le, return_internals=True)
        assert torch.isfinite(y).all()
        assert (ints["time_consuming_depth"] == 0).all()
        assert int(ints["time_consuming_updates"]) == 0


# ===============================================================
# Config trust boundaries and train.py resume guard
# ===============================================================

class TestConfigAndResume:
    def test_config_roundtrip_defaults_off(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.model.time_consuming_mode == "off"
        cfg.model.time_consuming_mode = "accumulate"
        cfg.validate()
        assert cfg.to_dict()["model"]["time_consuming_mode"] == "accumulate"

    def test_illegal_mode_and_bad_values_rejected(self) -> None:
        for kwargs in (
            dict(time_consuming_mode="bogus"),
            dict(time_consuming_max_steps=0),
            dict(time_consuming_max_steps=True),
            dict(time_consuming_eps=float("nan")),
            dict(time_consuming_eps=1.0),
            # eps below the declared lower bound: 1 - eps would round to 1.0,
            # forcing the never-crossed branch (finding R1-m2).
            dict(time_consuming_eps=1e-9),
            dict(time_consuming_update_scale=-1.0),
            dict(time_consuming_update_scale=float("inf")),
            dict(time_consuming_delay_scale=0.0),
            dict(time_consuming_threshold=float("inf")),
            dict(time_consuming_threshold=0.0),
            # threshold > 1 admits R < 0 (non-convex mixture): refused.
            dict(time_consuming_threshold=1.5),
        ):
            # Rejected even with mode="off" (module absent).
            with pytest.raises(ValueError):
                ExperimentConfig(model=type(ExperimentConfig().model)(**kwargs))

    def test_off_constructor_refuses_eps_and_update_scale(self) -> None:
        """The off-mode constructor validates eps/update_scale too (finding
        R1-M2 / R2-m1)."""
        for kwargs in (
            dict(time_consuming_eps=float("nan")),
            dict(time_consuming_update_scale=-1.0),
            dict(time_consuming_update_scale=float("inf")),
        ):
            with pytest.raises(ValueError):
                NSMoRCore(
                    sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
                    time_consuming_mode="off", **kwargs,
                )

    def test_direct_module_refuses_off(self) -> None:
        with pytest.raises(ValueError):
            TimeConsumingRecursion(8, mode="off")

    def test_train_resume_guard_refuses_mode_switch(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "nsmor_train_entry_tc", _REPO_ROOT / "scripts" / "train.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ckpt_cfg = ExperimentConfig()
        ckpt_cfg.model.time_consuming_mode = "off"
        active = ExperimentConfig()
        active.model.time_consuming_mode = "depth_delay"
        with pytest.raises(ValueError, match="time_consuming_mode"):
            mod._require_architecture_config_match(
                {"config": ckpt_cfg.to_dict()}, active, Path("ckpt.pth"),
            )

    def test_train_resume_guard_top_level_stamp(self) -> None:
        """A top-level ``time_consuming_mode`` stamp catches a switch even
        when the stored config lacks the option leaves (finding R1-B3)."""
        spec = importlib.util.spec_from_file_location(
            "nsmor_train_entry_tc2", _REPO_ROOT / "scripts" / "train.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        active = ExperimentConfig()
        active.model.time_consuming_mode = "depth_delay"
        # A legacy-looking checkpoint: a stored config WITHOUT the option
        # leaves, plus the additive top-level stamp recording "off".
        with pytest.raises(ValueError, match="time_consuming_mode"):
            mod._require_architecture_config_match(
                {"config": {"model": {}}, "time_consuming_mode": "off"},
                active, Path("ckpt.pth"),
            )
        # A checkpoint whose stamp matches the active mode passes the guard.
        mod._require_architecture_config_match(
            {"config": {"model": {}}, "time_consuming_mode": "depth_delay"},
            active, Path("ckpt.pth"),
        )

    def test_train_resume_guard_legacy_defaults_to_off(self) -> None:
        """A legacy checkpoint with NEITHER the top-level stamp NOR the stored
        config leaf defaults to 'off' and refuses an enabled active mode
        (finding R1-B3 legacy gap)."""
        spec = importlib.util.spec_from_file_location(
            "nsmor_train_entry_tc_legacy", _REPO_ROOT / "scripts" / "train.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        active = ExperimentConfig()
        active.model.time_consuming_mode = "depth_delay"
        # No stamp and no stored leaf -> effective "off" -> enabled active refused.
        with pytest.raises(ValueError, match="time_consuming_mode"):
            mod._require_architecture_config_match(
                {"config": {"model": {}}}, active, Path("legacy.pth"),
            )
        # The same legacy checkpoint is accepted under an 'off' active mode.
        off_active = ExperimentConfig()
        off_active.model.time_consuming_mode = "off"
        mod._require_architecture_config_match(
            {"config": {"model": {}}}, off_active, Path("legacy.pth"),
        )

    def test_provenance_key_registered(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "nsmor_train_entry_tc3", _REPO_ROOT / "scripts" / "train.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert "time_consuming_mode" in mod._PROVENANCE_KEYS


# ===============================================================
# Backend refusal
# ===============================================================

class TestBackendRefusal:
    @staticmethod
    def _mutated(mode: str = "depth_delay", **kw):
        m = _model(mode, **kw)
        # Post-construction mutation: the executed module stays present while
        # both mode aliases read "off".  Only an executed-child guard refuses.
        m.time_consuming_mode = "off"
        m.backend.time_consuming_mode = "off"
        return m

    def test_guard_helper_matches_executed_module(self) -> None:
        assert not time_consuming_module_present(_model("off"))
        assert time_consuming_module_present(_model("depth_delay"))
        assert time_consuming_module_present(self._mutated("accumulate"))

    def test_full_system_jacobian_refused(self) -> None:
        from nsmor.analysis.dynamics import FixedPointAdapter

        adapter = FixedPointAdapter(self._mutated("depth_delay"))
        X = torch.randn(1, 1, 8)
        with pytest.raises(ValueError):
            adapter.compute_full_system_jacobian(X, torch.tensor([1]))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_raw_jax_refuses(self) -> None:
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        with pytest.raises(ValueError):
            NSMoRCoreJAX.from_torch(self._mutated("depth_delay"))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_flax_refuses(self) -> None:
        from nsmor.jax.model import assert_flax_supported

        with pytest.raises(ValueError):
            assert_flax_supported(self._mutated("accumulate"), context="test")

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_jax_eval_wrapper_refuses(self) -> None:
        from nsmor.analysis.jax_eval import JAXEvalWrapper

        with pytest.raises(ValueError):
            JAXEvalWrapper.from_torch(self._mutated("depth_delay"))


# ===============================================================
# Own-checkpoint round trip
# ===============================================================

class TestCheckpointRebuild:
    def test_state_dict_roundtrip(self) -> None:
        for mode in ("depth_delay", "accumulate"):
            m = _model(mode, time_consuming_max_steps=3)
            x, lengths = _batch()
            with torch.no_grad():
                y0 = m(x, lengths)
            sd = copy.deepcopy(m.state_dict())
            m2 = _model(mode, time_consuming_max_steps=3)
            m2.load_state_dict(sd, strict=True)
            with torch.no_grad():
                y1 = m2(x, lengths)
            assert torch.allclose(y0, y1)


# ===============================================================
# Short synthetic CPU training smoke (each new mode trains, finite)
# ===============================================================

class TestSyntheticSmoke:
    @staticmethod
    def _loader(seed: int, n: int = 12, T: int = 8):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, T, 8, generator=g)
        w = torch.tensor(
            [[0.7, -0.4, 0.2, 0.5, -0.6, 0.1, 0.3, -0.2]], dtype=torch.float32,
        )
        y = (x @ w.t()).squeeze(-1) + 0.1
        lengths = torch.full((n,), T, dtype=torch.long)
        return x, y, lengths

    @pytest.mark.parametrize("mode", ["off", "depth_delay", "accumulate"])
    def test_mode_trains_and_stays_finite(self, mode: str) -> None:
        from nsmor.loss import BioJointLoss

        torch.manual_seed(0)
        model = NSMoRCore(
            sensory_dim=4, mcmc_dim=4, hidden_dim=16, dropout=0.0,
            time_consuming_mode=mode, time_consuming_max_steps=3,
        )
        crit = BioJointLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x, y, lengths = self._loader(0)
        losses = []
        for _ in range(3):
            model.train()
            opt.zero_grad()
            y_pred, internals = model(x, lengths, return_internals=True)
            g_gru = internals["routing_gates"][:, :, 1:2]
            loss = crit(
                y_pred, y, lengths, g_gru, lambda_reg=0.01,
                lambda_jerk=0.005, out_valid=internals.get("out_valid", None),
            )
            assert torch.isfinite(loss), f"{mode}: nonfinite loss"
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        assert all(losses[i] == losses[i] for i in range(len(losses)))  # not NaN
        assert torch.isfinite(torch.tensor(losses)).all()
