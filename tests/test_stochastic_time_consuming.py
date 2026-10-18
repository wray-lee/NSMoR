"""Focused regressions for CAMoR-L2 (ADR 0012): stochastic depth.

Synthetic, deterministic, in-memory.  No formal data, checkpoints or nested
artifacts.  The suite pins the implementation contract of spec-v2:

* opt-in ``time_consuming_mode="stochastic"`` (CAMoR-L-S) and its
  deterministic matched control ``"hard_depth"`` (CAMoR-L-H);
* float64 inverse-CDF sampling, per-frame ``P(N=k)``, differentiable
  ``E[delay]`` and the sampled depth;
* private training RNG (global RNG untouched), stateless ``eval_seed``;
* the STE non-finite guard;
* stimulus-anchored, prefix-level latency contrast loss (default off);
* the latency batch sampler / startup coverage contract;
* resume / backend refusals and byte-identical default preservation against
  the fixed pre-change ref ``5eadb22`` (never ``git show HEAD`` self-comparison).
"""

from __future__ import annotations

import copy
import importlib.util
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from nsmor.config_parser import ExperimentConfig, ModelConfig
from nsmor.loss import (
    BioDecisionLoss,
    BioJointLoss,
    stimulus_anchored_expected_delay,
)
from nsmor.model_nsmor_core import (
    NSMoRCore,
    TimeConsumingRecursion,
    _inverse_cdf_from_uniform,
    time_consuming_module_present,
)

try:  # pragma: no cover - environment dependent
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    JAX_AVAILABLE = False

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Fixed pre-change ref (spec §12.1).  NEVER replace this with ``HEAD``.
PRE_CHANGE_REF = "5eadb2268bcb6d67038dda3e981870895758ac46"

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
_NEW_KEYS = (
    "time_consuming_halt_probs",
    "time_consuming_expected_delay",
    "time_consuming_nonfinite_expected",
)


def _git_show(ref: str, relpath: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{ref}:{relpath}"], cwd=str(_REPO_ROOT), text=True,
    )


def _load_module_at_ref(ref: str, relpath: str, name: str):
    """Load ``git show <ref>:<relpath>`` as a module.

    Self-comparison guard: if the ref resolves to the current HEAD AND the
    working-tree file is byte-identical to the ref source, the comparison is
    vacuous and we fail closed.  During pre-commit development HEAD *is* the
    pre-change ref but the working tree differs, so the guard must key on the
    file content, not only the SHA (spec R9; the literal SHA check would make
    the required pre-commit run impossible).
    """
    source = _git_show(ref, relpath)
    current = (_REPO_ROOT / relpath).read_text()
    ref_sha = subprocess.check_output(
        ["git", "rev-parse", ref], cwd=str(_REPO_ROOT), text=True,
    ).strip()
    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True,
    ).strip()
    if ref_sha == head_sha and source == current:
        raise RuntimeError(
            f"self-comparison refused: {ref} == HEAD and {relpath} is "
            "byte-identical to the working tree"
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
    kw.setdefault("hidden_dim", 8)
    kw.setdefault("dropout", 0.0)
    return NSMoRCore(
        sensory_dim=4, mcmc_dim=4, time_consuming_mode=mode, **kw,
    ).eval()


def _batch(B: int = 2, T: int = 6, seed: int = 11):
    torch.manual_seed(seed)
    x = torch.randn(B, T, 8)
    lengths = torch.tensor(
        [T - (i % 3) for i in range(B)], dtype=torch.long,
    )
    return x, lengths


def _tc(mode: str, **kw) -> TimeConsumingRecursion:
    kw.setdefault("max_steps", 4)
    return TimeConsumingRecursion(8, mode=mode, **kw)


def _force_halt(mod: TimeConsumingRecursion, value: float) -> None:
    with torch.no_grad():
        mod.halt.weight.zero_()
        mod.halt.bias.fill_(value)


def _dist(mod: TimeConsumingRecursion, h: torch.Tensor, valid: torch.Tensor):
    out = mod(h, valid, return_distribution_internals=True)
    assert len(out) == 11
    return out


def _p_sum(p: torch.Tensor) -> torch.Tensor:
    return p.double().sum(dim=-1)


# ===============================================================
# 1. Options, config, constructors
# ===============================================================

class TestOptionsAndConfig:
    def test_new_modes_accepted_bogus_rejected(self) -> None:
        for mode in ("stochastic", "hard_depth"):
            cfg = ModelConfig(time_consuming_mode=mode)
            assert cfg.time_consuming_mode == mode
            _tc(mode)
        with pytest.raises(ValueError, match="stochastic"):
            ModelConfig(time_consuming_mode="bogus")
        with pytest.raises(ValueError, match="hard_depth"):
            _tc("bogus")

    @pytest.mark.parametrize(
        "bad", [True, False, 1.5, -1, 2**63, "3", None],
    )
    def test_seed_rejected(self, bad) -> None:
        with pytest.raises(ValueError):
            ModelConfig(time_consuming_seed=bad)
        with pytest.raises(ValueError):
            _tc("stochastic", seed=bad)
        with pytest.raises(ValueError):
            _model("off", time_consuming_seed=bad)

    def test_seed_accepted_and_validated_off(self) -> None:
        assert ModelConfig(time_consuming_seed=0).time_consuming_seed == 0
        big = 2**63 - 1
        assert ModelConfig(time_consuming_seed=big).time_consuming_seed == big
        # Off mode still validates the seed at the trust boundary.
        m = _model("off", time_consuming_seed=big)
        assert m.backend.time_consuming is None

    @pytest.mark.parametrize(
        "bad", [True, False, float("nan"), float("inf"), -0.1],
    )
    def test_lambda_latency_rejected(self, bad) -> None:
        with pytest.raises(ValueError):
            ExperimentConfig(loss=type(ExperimentConfig().loss)(
                lambda_latency=bad,
            ))

    def test_lambda_latency_accepts_zero_and_positive(self) -> None:
        assert ExperimentConfig().loss.lambda_latency == 0.0
        cfg = ExperimentConfig()
        cfg.model.time_consuming_mode = "stochastic"
        cfg.loss.lambda_latency = 0.5
        cfg.validate()
        assert cfg.to_dict()["loss"]["lambda_latency"] == 0.5

    def test_lambda_latency_requires_new_mode(self) -> None:
        cfg = ExperimentConfig()
        cfg.loss.lambda_latency = 0.1
        cfg.model.time_consuming_mode = "depth_delay"
        with pytest.raises(ValueError, match="lambda_latency"):
            cfg.validate()
        cfg.model.time_consuming_mode = "stochastic"
        cfg.validate()

    def test_config_roundtrip_new_leaves(self) -> None:
        cfg = ExperimentConfig.from_dict(
            {"model": {"time_consuming_seed": 7},
             "loss": {"lambda_latency": 0.0}},
        )
        d = cfg.to_dict()
        assert d["model"]["time_consuming_seed"] == 7
        assert d["loss"]["lambda_latency"] == 0.0

    def test_grad_checkpointing_refused_for_existing_modes(self) -> None:
        for mode in ("depth_delay", "accumulate"):
            with pytest.raises(ValueError, match="grad_checkpointing"):
                _tc(mode, grad_checkpointing=True)
        _tc("stochastic", grad_checkpointing=True)
        _tc("hard_depth", grad_checkpointing=True)

    def test_off_constructor_refuses_new_mode(self) -> None:
        with pytest.raises(ValueError):
            _tc("off")


# ===============================================================
# 2. Distribution and sampling
# ===============================================================

class TestDistributionAndSampling:
    def test_forced_halt_point_mass_at_one(self) -> None:
        mod = _tc("stochastic", max_steps=4, delay_scale=1.0)
        _force_halt(mod, 50.0)
        h = torch.randn(2, 5, 8)
        valid = torch.ones(2, 5, dtype=torch.bool)
        out = _dist(mod, h, valid)
        hp, ed = out[9], out[10]
        assert torch.allclose(hp[..., 0], torch.ones_like(hp[..., 0]))
        assert torch.allclose(hp[..., 1:], torch.zeros_like(hp[..., 1:]))
        assert torch.allclose(ed, torch.zeros_like(ed))
        assert (out[1][valid] == 1).all()
        assert (out[3][valid] == 0).all()

    def test_forced_halt_point_mass_at_k(self) -> None:
        K = 4
        mod = _tc("stochastic", max_steps=K, delay_scale=2.0)
        _force_halt(mod, -50.0)
        h = torch.randn(2, 5, 8)
        valid = torch.ones(2, 5, dtype=torch.bool)
        out = _dist(mod, h, valid)
        hp, ed = out[9], out[10]
        assert torch.allclose(hp[..., K - 1], torch.ones_like(hp[..., K - 1]))
        assert torch.allclose(hp[..., :K - 1], torch.zeros_like(hp[..., :K - 1]))
        assert torch.allclose(ed, torch.full_like(ed, 2.0 * (K - 1)))
        assert (out[1][valid] == K).all()
        assert (out[3][valid] == 2.0 * (K - 1)).all()

    def test_pmf_normalised_float64_and_float32(self) -> None:
        mod = _tc("stochastic", max_steps=8)
        torch.manual_seed(3)
        h = torch.randn(4, 7, 8)
        valid = torch.ones(4, 7, dtype=torch.bool)
        out = _dist(mod, h, valid)
        hp, ed = out[9], out[10]
        assert hp.dtype == torch.float32
        assert ed.dtype == torch.float32
        assert torch.allclose(
            _p_sum(hp), torch.ones(4, 7, dtype=torch.float64), atol=1e-5,
        )
        e_n = ed.double() + 1.0
        assert bool((e_n >= 1.0).all() and (e_n <= 8.0).all())

    def test_sampled_depth_in_range_and_delay_mapping(self) -> None:
        K, scale = 6, 3.0
        mod = _tc("stochastic", max_steps=K, delay_scale=scale)
        torch.manual_seed(5)
        h = torch.randn(3, 9, 8)
        valid = torch.ones(3, 9, dtype=torch.bool)
        out = _dist(mod, h, valid)
        depth, delay = out[1], out[3]
        assert bool(((depth[valid] >= 1) & (depth[valid] <= K)).all())
        assert torch.equal(
            delay[valid], (depth[valid].float() - 1.0) * scale,
        )
        assert int(out[4]) <= int(scale * (K - 1))

    def test_mc_mean_matches_expected_depth(self) -> None:
        K = 4
        mod = _tc("stochastic", max_steps=K)
        torch.manual_seed(7)
        B, T = 80, 250
        h = torch.randn(B, T, 8)
        valid = torch.ones(B, T, dtype=torch.bool)
        out = _dist(mod, h, valid)
        depth, ed = out[1], out[10]
        e_n = (ed.detach().double() + 1.0).reshape(-1)
        samples = depth.reshape(-1).double()
        n = samples.numel()
        mean = samples.mean()
        se = samples.std(unbiased=True) / math.sqrt(n)
        diff = (mean - e_n.mean()).abs()
        assert float(diff) < 4.0 * float(se)

    def test_inverse_cdf_tiny_atom_reachable_float64(self) -> None:
        # p_2 = (1 - 0.5) * 2e-9 = 1e-9; S_2 = 0.5 - 1e-9.
        g = torch.tensor([[0.5, 2e-9, 0.0]], dtype=torch.float64)
        u = torch.tensor([0.5 - 5e-10], dtype=torch.float64)
        n, p = _inverse_cdf_from_uniform(u, g)
        assert int(n[0]) == 2
        assert torch.allclose(p.sum(dim=1), torch.ones(1, dtype=torch.float64))
        assert abs(float(p[0, 1]) - 1e-9) < 1e-15

    def test_inverse_cdf_requires_float64(self) -> None:
        with pytest.raises(ValueError):
            _inverse_cdf_from_uniform(
                torch.zeros(1), torch.zeros(1, 3),
            )

    def test_padding_zero(self) -> None:
        mod = _tc("stochastic", max_steps=4)
        h = torch.randn(2, 6, 8)
        lengths = torch.tensor([6, 3])
        valid = torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)
        out = _dist(mod, h, valid)
        hp, ed, depth, delay = out[9], out[10], out[1], out[3]
        assert (hp[~valid] == 0).all()
        assert (ed[~valid] == 0).all()
        assert (depth[~valid] == 0).all()
        assert (delay[~valid] == 0).all()

    def test_all_empty_is_finite(self) -> None:
        mod = _tc("stochastic", max_steps=3)
        h = torch.randn(2, 4, 8)
        valid = torch.zeros(2, 4, dtype=torch.bool)
        out = _dist(mod, h, valid)
        assert torch.isfinite(out[0]).all()
        assert (out[1] == 0).all()
        assert int(out[2]) == 0
        assert (out[9] == 0).all() and (out[10] == 0).all()

    def test_hard_depth_matches_rounded_expected(self) -> None:
        K = 6
        mod = _tc("hard_depth", max_steps=K)
        torch.manual_seed(13)
        h = torch.randn(3, 8, 8)
        valid = torch.ones(3, 8, dtype=torch.bool)
        out = _dist(mod, h, valid)
        hp, depth = out[9], out[1]
        e_n = (hp.double() * torch.arange(
            1, K + 1, dtype=torch.float64,
        )).sum(dim=-1)
        expected = torch.floor(e_n + 0.5).clamp(1, K).long()
        assert torch.equal(depth[valid], expected[valid])
        assert bool(((depth[valid] >= 1) & (depth[valid] <= K)).all())

    def test_hard_depth_nonfinite_expected_fails_closed(self) -> None:
        mod = _tc("hard_depth", max_steps=3)
        mod._block = lambda u: torch.full_like(u, float("nan"))
        h = torch.randn(1, 2, 8)
        valid = torch.ones(1, 2, dtype=torch.bool)
        # NaN logits make the halt probabilities non-finite; the pmf validator
        # refuses them before the E[N] guard (both are fail-closed).
        with pytest.raises(ValueError, match="finite"):
            mod(h, valid, return_distribution_internals=True)


# ===============================================================
# 3. RNG / eval seed
# ===============================================================

class TestRNGAndEvalSeed:
    def test_training_stream_does_not_touch_global_rng(self) -> None:
        m = _model("stochastic", time_consuming_max_steps=4)
        m.train()
        x, lengths = _batch()
        before = torch.get_rng_state().clone()
        with torch.no_grad():
            m(x, lengths)
        assert torch.equal(before, torch.get_rng_state())

    def test_training_draws_advance_between_forwards(self) -> None:
        m = _model("stochastic", time_consuming_max_steps=4)
        m.train()
        x, lengths = _batch(B=8, T=12)
        with torch.no_grad():
            _, i1 = m(x, lengths, return_internals=True)
            _, i2 = m(x, lengths, return_internals=True)
        assert not torch.equal(
            i1["time_consuming_depth"], i2["time_consuming_depth"],
        )

    def test_eval_seed_is_reproducible_and_stateless(self) -> None:
        m = _model("stochastic", time_consuming_max_steps=4)
        x, lengths = _batch(B=4, T=10)
        with torch.no_grad():
            y1, i1 = m(x, lengths, return_internals=True, eval_seed=3)
            y2, i2 = m(x, lengths, return_internals=True, eval_seed=3)
            y3, i3 = m(x, lengths, return_internals=True, eval_seed=4)
        assert torch.equal(y1, y2)
        assert torch.equal(
            i1["time_consuming_depth"], i2["time_consuming_depth"],
        )
        assert not torch.equal(
            i1["time_consuming_depth"], i3["time_consuming_depth"],
        )
        assert m.backend.time_consuming.eval_seed == 0

    def test_eval_seed_fails_closed_in_wrong_context(self) -> None:
        x, lengths = _batch()
        m = _model("stochastic", time_consuming_max_steps=4)
        m.train()
        with pytest.raises(ValueError, match="eval_seed"):
            m(x, lengths, eval_seed=0)
        m.eval()
        for mode in ("off", "depth_delay", "accumulate", "hard_depth"):
            mm = _model(mode, time_consuming_max_steps=4)
            with pytest.raises(ValueError, match="eval_seed"):
                mm(x, lengths, eval_seed=0)

    def test_same_seed_models_match(self) -> None:
        kwargs = dict(
            sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
            time_consuming_mode="stochastic", time_consuming_max_steps=4,
            time_consuming_seed=99,
        )
        torch.manual_seed(21)
        a = NSMoRCore(**kwargs)
        torch.manual_seed(21)
        b = NSMoRCore(**kwargs)
        x, lengths = _batch(B=4, T=9)
        a.train(); b.train()
        with torch.no_grad():
            _, ia = a(x, lengths, return_internals=True)
            _, ib = b(x, lengths, return_internals=True)
        assert torch.equal(
            ia["time_consuming_depth"], ib["time_consuming_depth"],
        )
        a.eval(); b.eval()
        with torch.no_grad():
            ya = a(x, lengths, eval_seed=7)
            yb = b(x, lengths, eval_seed=7)
        assert torch.equal(ya, yb)

    def test_set_seed_methods(self) -> None:
        m = _model("stochastic", time_consuming_max_steps=4)
        tc = m.backend.time_consuming
        params = [p.detach().clone() for p in tc.parameters()]
        sd_keys = set(tc.state_dict())
        tc.set_train_seed(5)
        tc.set_eval_seed(6)
        assert tc.seed == 5 and tc.eval_seed == 6
        assert set(tc.state_dict()) == sd_keys
        for p, q in zip(tc.parameters(), params):
            assert torch.equal(p, q)
        for bad in (True, -1, 2**63, 1.5):
            with pytest.raises(ValueError):
                tc.set_train_seed(bad)
            with pytest.raises(ValueError):
                tc.set_eval_seed(bad)

    def test_eval_seed_base_set_is_deterministic(self) -> None:
        m = _model("stochastic", time_consuming_max_steps=4)
        x, lengths = _batch(B=3, T=8)
        draws = []
        with torch.no_grad():
            for s in range(5):
                _, internals = m(
                    x, lengths, return_internals=True, eval_seed=100 + s,
                )
                draws.append(internals["time_consuming_depth"].clone())
        with torch.no_grad():
            _, again_internals = m(
                x, lengths, return_internals=True, eval_seed=100,
            )
        assert torch.equal(draws[0], again_internals["time_consuming_depth"])


# ===============================================================
# 4. Gradients and numerical safety
# ===============================================================

class TestGradientsAndSafety:
    @staticmethod
    def _single_frame(mode: str = "stochastic", K: int = 4):
        mod = _tc(mode, max_steps=K, delay_scale=0.1)
        h = torch.randn(1, 1, 8)
        valid = torch.ones(1, 1, dtype=torch.bool)
        return mod, h, valid

    def test_hidden_equals_hard_value(self) -> None:
        mod, h, valid = self._single_frame()
        out = _dist(mod, h, valid)
        depth = int(out[1][0, 0])
        # Recompute the iterates with the same parameters.
        u = h.reshape(1, 8)
        iters = []
        for _ in range(mod.max_steps):
            u = mod._block(u)
            iters.append(u)
        # delay_scale=0.1 rounds every delay to 0 for K=4, so the emitted
        # value at t=0 is exactly the hard value.
        assert torch.equal(out[0][0, 0], iters[depth - 1][0])

    def test_halt_head_and_block_gradients(self) -> None:
        mod, h, valid = self._single_frame()
        out = mod(h, valid)
        loss = (out[0] ** 2).sum()
        loss.backward()
        assert mod.halt.weight.grad is not None
        assert mod.halt.weight.grad.abs().sum() > 0
        for name, p in mod.named_parameters():
            if name.startswith("halt"):
                continue
            assert p.grad is not None and torch.isfinite(p.grad).all(), name

    def test_ste_gradient_equals_expected_value_gradient(self) -> None:
        torch.manual_seed(31)
        mod, h, valid = self._single_frame()
        out = mod(h, valid)
        loss = (out[0] ** 2).sum()
        loss.backward()
        grads_hard = {
            n: p.grad.detach().clone()
            for n, p in mod.named_parameters() if p.grad is not None
        }
        mod.zero_grad(set_to_none=True)
        # Independent E[u] with the same parameters (p does not depend on U).
        u = h.reshape(1, 8)
        iters = []
        for _ in range(mod.max_steps):
            u = mod._block(u)
            iters.append(u)
        u_stack = torch.stack(iters, dim=1)              # (1, K, H)
        g = torch.sigmoid(mod.halt(u_stack)).squeeze(-1).double()
        _, p = _inverse_cdf_from_uniform(torch.zeros(1, dtype=torch.float64), g)
        e_u = (p.float().unsqueeze(-1) * u_stack).sum(dim=1)  # (1, H)
        # STE forward value is the HARD iterate; only the gradient comes from
        # E[u].  The reference loss is therefore <2 * v_hard, E[u]> with
        # v_hard held constant (not (E[u])^2, which uses the wrong outer
        # derivative).
        depth = int(out[1][0, 0])
        v_hard = iters[depth - 1]
        (2.0 * v_hard.detach() * e_u).sum().backward()
        for name, p_hard in grads_hard.items():
            p_e = dict(mod.named_parameters())[name].grad
            assert p_e is not None
            assert torch.allclose(p_hard, p_e, atol=1e-6, rtol=1e-5), name

    def test_k1_halt_head_has_no_gradient(self) -> None:
        mod, h, valid = self._single_frame(K=1)
        out = mod(h, valid)
        (out[0] ** 2).sum().backward()
        assert mod.halt.weight.grad is None or (
            mod.halt.weight.grad.abs().sum() == 0
        )

    def test_ste_nonfinite_guard(self) -> None:
        # K=2 so the injected non-finite iterate is the LAST one: the halt
        # head then still sees a finite input at k=1 and a +inf at k=2 (tiny
        # positive weight -> sigmoid(+inf)=1), keeping the pmf finite.  With
        # K>2 the next block step would turn inf into NaN and the pmf
        # validator would (correctly) refuse before the E[u] guard.
        mod = _tc("stochastic", max_steps=2, delay_scale=0.1)
        _force_halt(mod, 50.0)
        # A zeroed halt weight times ``inf`` is NaN (0 * inf), which would
        # abort in the pmf validator before the E[u] guard.  A tiny positive
        # weight makes halt(+inf) = +inf and sigmoid(+inf) = 1, i.e. a FINITE
        # point mass at N=1, so the guard itself is exercised: E[u] is NaN
        # (0 * inf atoms) while v_hard = u_1 is finite.
        with torch.no_grad():
            mod.halt.weight.fill_(1e-6)
        calls = {"n": 0}
        orig = mod._block

        def patched(u: torch.Tensor) -> torch.Tensor:
            calls["n"] += 1
            out = orig(u)
            if calls["n"] == 2:
                return torch.full_like(out, float("inf"))
            return out

        mod._block = patched
        h = torch.randn(1, 3, 8, requires_grad=True)
        valid = torch.ones(1, 3, dtype=torch.bool)
        out = mod(h, valid)
        assert torch.isfinite(out[0]).all()
        assert mod.last_nonfinite_expected == 3
        loss = out[0].sum()
        assert torch.isfinite(loss)
        loss.backward()
        assert h.grad is not None and torch.count_nonzero(h.grad) == 0
        for parameter in mod.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) == 0

    def test_grad_checkpointing_matches(self) -> None:
        torch.manual_seed(41)
        h = torch.randn(2, 4, 8)
        valid = torch.ones(2, 4, dtype=torch.bool)
        plain = _tc("stochastic", max_steps=4, delay_scale=0.1)
        ckpt = copy.deepcopy(plain)
        ckpt.grad_checkpointing = True
        o1 = plain(h, valid)[0]
        o2 = ckpt(h, valid)[0]
        assert torch.equal(o1, o2)
        (o1 ** 2).sum().backward()
        (o2 ** 2).sum().backward()
        for (n, p1), (_, p2) in zip(
            plain.named_parameters(), ckpt.named_parameters(),
        ):
            if p1.grad is None:
                continue
            assert torch.allclose(p1.grad, p2.grad, atol=1e-6, rtol=1e-5), n


# ===============================================================
# 5. Emission and causality
# ===============================================================

class TestEmissionAndCausality:
    def test_out_valid_has_a_source_at_or_before(self) -> None:
        m = _model(
            "stochastic", time_consuming_max_steps=4,
            time_consuming_delay_scale=2.0,
        )
        x, lengths = _batch(B=3, T=12)
        with torch.no_grad():
            _, ints = m(x, lengths, return_internals=True, eval_seed=2)
        out_valid = ints["out_valid"]
        delay = ints["output_delay"]
        valid = torch.arange(12).unsqueeze(0) < lengths.unsqueeze(1)
        B, T = out_valid.shape
        for b in range(B):
            for j in range(T):
                if not bool(out_valid[b, j]):
                    continue
                sources = [
                    t for t in range(T)
                    if bool(valid[b, t])
                    and t + int(round(float(delay[b, t]))) == j
                ]
                assert sources, (b, j)
                assert all(t <= j for t in sources)

    def test_future_perturbation_does_not_change_past_eval(self) -> None:
        m = _model("stochastic", time_consuming_max_steps=4)
        x, lengths = _batch(B=2, T=10)
        x2 = x.clone()
        x2[:, -1, :] += 25.0
        with torch.no_grad():
            _, i1 = m(x, lengths, return_internals=True, eval_seed=5)
            _, i2 = m(x2, lengths, return_internals=True, eval_seed=5)
        for key in (
            "output_delay", "realized_delay", "time_consuming_depth",
            "time_consuming_halt_probs", "time_consuming_expected_delay",
            "out_valid",
        ):
            assert torch.equal(i1[key][:, :-1], i2[key][:, :-1]), key

    def test_future_perturbation_does_not_change_past_train(self) -> None:
        kwargs = dict(
            sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
            time_consuming_mode="stochastic", time_consuming_max_steps=4,
            time_consuming_seed=17,
        )
        torch.manual_seed(51)
        a = NSMoRCore(**kwargs)
        torch.manual_seed(51)
        b = NSMoRCore(**kwargs)
        a.train(); b.train()
        x, lengths = _batch(B=2, T=10)
        x2 = x.clone()
        x2[:, -1, :] -= 9.0
        _, ia = a(x, lengths, return_internals=True)
        _, ib = b(x2, lengths, return_internals=True)
        for key in (
            "output_delay", "time_consuming_depth",
            "time_consuming_halt_probs", "time_consuming_expected_delay",
            "out_valid",
        ):
            assert torch.equal(ia[key][:, :-1], ib[key][:, :-1]), key

    def test_mixed_length_batch(self) -> None:
        m = _model("stochastic", time_consuming_max_steps=4)
        x = torch.randn(3, 8, 8)
        lengths = torch.tensor([8, 3, 0])
        with torch.no_grad():
            y, ints = m(x, lengths, return_internals=True, eval_seed=0)
        assert y.shape == (3, 8)
        assert torch.isfinite(y).all()
        assert (ints["time_consuming_depth"][2] == 0).all()


# ===============================================================
# 6. Internals
# ===============================================================

class TestInternals:
    def test_existing_modes_have_no_new_keys(self) -> None:
        x, lengths = _batch()
        for mode in ("off", "depth_delay", "accumulate"):
            m = _model(mode, time_consuming_max_steps=3)
            with torch.no_grad():
                _, ints = m(x, lengths, return_internals=True)
            for key in _NEW_KEYS:
                assert key not in ints, f"{mode} must not expose {key}"

    @pytest.mark.parametrize("mode", ["stochastic", "hard_depth"])
    def test_new_modes_expose_all_keys(self, mode: str) -> None:
        K = 5
        m = _model(mode, time_consuming_max_steps=K, time_consuming_delay_scale=2.0)
        x, lengths = _batch(B=3, T=9)
        # eval_seed is stochastic-only; hard_depth is deterministic.
        _seed_kw = {"eval_seed": 1} if mode == "stochastic" else {}
        with torch.no_grad():
            _, ints = m(x, lengths, return_internals=True, **_seed_kw)
        for key in _TC_KEYS:
            assert key in ints, key
        for key in _NEW_KEYS:
            assert key in ints, key
        hp = ints["time_consuming_halt_probs"]
        ed = ints["time_consuming_expected_delay"]
        nf = ints["time_consuming_nonfinite_expected"]
        assert hp.shape == (3, 9, K) and hp.dtype == torch.float32
        assert ed.shape == (3, 9) and ed.dtype == torch.float32
        assert nf.shape == () and nf.dtype == torch.long
        valid = torch.arange(9).unsqueeze(0) < lengths.unsqueeze(1)
        assert torch.allclose(
            _p_sum(hp)[valid],
            torch.ones(int(valid.sum()), dtype=torch.float64),
        )
        assert (hp[~valid] == 0).all() and (ed[~valid] == 0).all()
        assert int(ints["time_consuming_updates"]) == int(valid.sum()) * K
        assert int(ints["time_consuming_max_delay"]) <= int(2.0 * (K - 1))

    def test_return_distribution_internals_default_is_nine_tuple(self) -> None:
        mod = _tc("stochastic", max_steps=3)
        h = torch.randn(1, 4, 8)
        valid = torch.ones(1, 4, dtype=torch.bool)
        assert len(mod(h, valid)) == 9
        for mode in ("depth_delay", "accumulate"):
            mm = _tc(mode, max_steps=3)
            assert len(mm(h, valid)) == 9
            with pytest.raises(ValueError):
                mm(h, valid, return_distribution_internals=True)


# ===============================================================
# 7. Latency contrast loss
# ===============================================================

def _latency_inputs(
    B: int = 6,
    T: int = 8,
    *,
    groups=None,
    prefixes=None,
    anchors=None,
    targets=None,
    mask=None,
    pairs=None,
    d_values=None,
):
    lengths = torch.full((B,), T, dtype=torch.long)
    if groups is None:
        # Prefixes must be nested in groups (10, 30 -> group 0; 20 -> group 1).
        groups = [0, 0, 1, 1, 0, 0]
    if prefixes is None:
        prefixes = [10, 10, 20, 20, 30, 30]
    if anchors is None:
        anchors = [0, 1, 2, 0, 1, 2]
    if targets is None:
        targets = [2.0, 2.0, 0.0, 0.0, 2.0, 0.0]
    if mask is None:
        mask = [True] * B
    if d_values is None:
        d_values = [5.0, 5.0, 3.0, 3.0, 5.0, 3.0]
    expected = torch.zeros(B, T)
    for i, v in enumerate(d_values):
        expected[i, :] = v
    expected.requires_grad_(True)
    kwargs = dict(
        lambda_latency=1.0,
        # Isolate the latency term: the default router reg would add 0.005
        # for g_gru=0.5 and swamp the zero-delta assertion.
        lambda_reg=0.0,
        latency_expected_delay=expected,
        latency_anchor_frames=torch.tensor(anchors, dtype=torch.long),
        latency_target_frames=torch.tensor(targets, dtype=torch.float64),
        latency_target_mask=torch.tensor(mask, dtype=torch.bool),
        latency_group_id=torch.tensor(groups, dtype=torch.long),
        latency_prefix_id=torch.tensor(prefixes, dtype=torch.long),
        latency_target_group=0,
        latency_reference_group=1,
    )
    if pairs is not None:
        kwargs["latency_pair_id"] = torch.tensor(pairs, dtype=torch.long)
    return expected, kwargs


def _mutated_leaf(t: torch.Tensor, value: float) -> torch.Tensor:
    """Return a fresh differentiable leaf with ``t[0, 0]`` replaced."""
    out = t.detach().clone()
    out[0, 0] = value
    return out.requires_grad_(True)


def _base_loss_args(B: int = 6, T: int = 8):
    y_pred = torch.zeros(B, T)
    y_true = torch.zeros(B, T)
    lengths = torch.full((B,), T, dtype=torch.long)
    g_gru = torch.full((B, T, 1), 0.5)
    return y_pred, y_true, lengths, g_gru


class TestLatencyLoss:
    def test_helper_reads_anchor(self) -> None:
        ed = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        a = torch.tensor([0, 2, 3])
        out = stimulus_anchored_expected_delay(ed, a)
        assert out.shape == (3,)
        assert torch.equal(out, torch.tensor([0.0, 6.0, 11.0]))
        with pytest.raises(ValueError):
            stimulus_anchored_expected_delay(ed, torch.tensor([0, 2, 4]))

    def test_disabled_is_bitwise_unchanged(self) -> None:
        ref_loss = _load_module_at_ref(
            PRE_CHANGE_REF, "nsmor/loss.py", "ref_loss_latency",
        )
        torch.manual_seed(61)
        args = _base_loss_args()
        ref = ref_loss.BioJointLoss()(*args, lambda_reg=0.01)
        new = BioJointLoss()(*args, lambda_reg=0.01, lambda_latency=0.0)
        assert torch.equal(ref, new)
        assert new.item() == pytest.approx(float(ref))
        crit = BioJointLoss()
        new2 = crit(*args, lambda_reg=0.01, lambda_latency=0.0)
        assert torch.equal(ref, new2)
        assert crit.last_latency_stats["latency_skipped"] == 1
        assert crit.last_latency_stats["skip_reason"] == "disabled"

    @pytest.mark.parametrize(
        "bad", [True, False, float("nan"), float("inf"), -1e-6],
    )
    def test_invalid_lambda_rejected(self, bad) -> None:
        args = _base_loss_args()
        with pytest.raises(ValueError):
            BioJointLoss()(*args, lambda_latency=bad)

    def test_missing_metadata_rejected(self) -> None:
        args = _base_loss_args()
        crit = BioJointLoss()
        with pytest.raises(ValueError, match="latency_expected_delay"):
            crit(*args, lambda_latency=1.0)

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (lambda kw: kw.__setitem__(
                "latency_expected_delay", torch.zeros(6, 7),
            ), "shape"),
            (lambda kw: kw.__setitem__(
                "latency_expected_delay",
                _mutated_leaf(kw["latency_expected_delay"], float("nan")),
            ), "finite"),
            (lambda kw: kw.__setitem__(
                "latency_expected_delay",
                _mutated_leaf(kw["latency_expected_delay"], -1.0),
            ), "non-negative"),
            (lambda kw: kw.__setitem__(
                "latency_anchor_frames", torch.tensor([0, 1, 2, 0, 1, 8]),
            ), "anchor"),
            (lambda kw: kw.__setitem__(
                "latency_anchor_frames", torch.tensor([0, 1, 2, 0, 1, 2],
                                                      dtype=torch.int32),
            ), "anchor"),
            (lambda kw: kw.__setitem__(
                "latency_target_frames", torch.tensor(
                    [2.0, 2.0, 0.0, 0.0, 2.0, -1.0],
                ),
            ), "target"),
            (lambda kw: kw.__setitem__(
                "latency_target_mask", torch.ones(6, dtype=torch.long),
            ), "mask"),
            (lambda kw: kw.__setitem__(
                "latency_group_id", torch.tensor([0, 0, 1, 1, -1, 1]),
            ), "group"),
            (lambda kw: kw.__setitem__(
                "latency_prefix_id", torch.tensor([10, 10, 20, 20, 30, -1]),
            ), "prefix"),
            (lambda kw: kw.__setitem__("latency_target_group", 1),
             "distinct"),
            (lambda kw: kw.__setitem__("latency_target_group", True),
             "target_group"),
            # Prefix 20 (rows 2-3) must belong to ONE group; [0,0,0,1,1,1]
            # splits it across groups 0 and 1.
            (lambda kw: kw.__setitem__(
                "latency_group_id", torch.tensor([0, 0, 0, 1, 1, 1]),
            ), "prefix"),
        ],
    )
    def test_validation_failures(self, mutate, match) -> None:
        expected, kwargs = _latency_inputs()
        mutate(kwargs)
        args = _base_loss_args()
        with pytest.raises(ValueError, match=match):
            BioDecisionLoss()(*args, **kwargs)

    def test_zero_length_masked_trial_rejected(self) -> None:
        expected, kwargs = _latency_inputs()
        args = list(_base_loss_args())
        args[2] = torch.tensor([8, 8, 0, 8, 8, 8])
        kwargs["latency_target_mask"] = torch.ones(6, dtype=torch.bool)
        with pytest.raises(ValueError, match="length"):
            BioDecisionLoss()(*args, **kwargs)

    def test_anchor_beyond_length_for_masked_trial(self) -> None:
        expected, kwargs = _latency_inputs(anchors=[0, 1, 2, 0, 1, 7])
        args = list(_base_loss_args())
        args[2] = torch.tensor([8, 8, 8, 8, 8, 5])
        with pytest.raises(ValueError, match="anchor"):
            BioDecisionLoss()(*args, **kwargs)

    def test_equal_deltas_give_zero(self) -> None:
        expected, kwargs = _latency_inputs()
        args = _base_loss_args()
        crit = BioDecisionLoss()
        loss = crit(*args, **kwargs)
        assert loss.dim() == 0
        assert loss.item() == pytest.approx(0.0, abs=1e-12)
        assert crit.last_latency_stats["latency_skipped"] == 0
        assert crit.last_latency_stats["n_target_prefixes"] == 2
        assert crit.last_latency_stats["n_reference_prefixes"] == 1

    def test_perturbation_is_quadratic(self) -> None:
        expected, kwargs = _latency_inputs(B=2, groups=[0, 1],
                                           prefixes=[10, 20],
                                           anchors=[0, 0],
                                           targets=[2.0, 0.0],
                                           d_values=[5.0, 3.0])
        args = _base_loss_args(B=2)
        eps = 0.25
        with torch.no_grad():
            expected[0, :] += eps
        loss = BioDecisionLoss()(*args, **kwargs)
        assert loss.item() == pytest.approx(eps ** 2, rel=1e-6)

    def test_masked_trial_does_not_change_loss(self) -> None:
        expected, kwargs = _latency_inputs()
        args = _base_loss_args()
        base = BioDecisionLoss()(*args, **kwargs).item()
        kwargs["latency_target_mask"][0] = False
        with torch.no_grad():
            kwargs["latency_expected_delay"][0, :] = 1e6
        changed = BioDecisionLoss()(*args, **kwargs).item()
        # Prefix 10 still has trial 1 (d=5); the masked trial is ignored.
        assert changed == pytest.approx(base, rel=1e-6)

    def test_prefix_weighting_not_trial_weighting(self) -> None:
        # Target: prefix A (1 trial, d=10), prefix B (3 trials, d=2).
        # Unweighted prefix mean D_target = 6; trial mean would be 4.
        # Reference: 1 prefix, d=3.  Targets: target L=3, ref L=0 -> dT=3.
        B = 5
        groups = [0, 0, 0, 0, 1]
        prefixes = [10, 20, 20, 20, 30]
        d_values = [10.0, 2.0, 2.0, 2.0, 3.0]
        targets = [3.0, 3.0, 3.0, 3.0, 0.0]
        expected, kwargs = _latency_inputs(
            B=B, groups=groups, prefixes=prefixes, anchors=[0] * B,
            targets=targets, d_values=d_values,
        )
        args = _base_loss_args(B=B)
        loss = BioDecisionLoss()(*args, **kwargs)
        assert loss.item() == pytest.approx(0.0, abs=1e-12)

    def test_missing_group_returns_differentiable_zero(self) -> None:
        expected, kwargs = _latency_inputs(groups=[0, 0, 0, 0, 0, 0])
        args = _base_loss_args()
        crit = BioDecisionLoss()
        loss = crit(*args, **kwargs)
        assert loss.item() == 0.0
        assert loss.requires_grad
        assert crit.last_latency_stats["latency_skipped"] == 1
        assert crit.last_latency_stats["skip_reason"] == "no_reference_group"
        loss.backward()
        assert expected.grad is not None

    def test_no_masked_trials_returns_zero(self) -> None:
        expected, kwargs = _latency_inputs(mask=[False] * 6)
        args = _base_loss_args()
        crit = BioDecisionLoss()
        loss = crit(*args, **kwargs)
        assert loss.item() == 0.0
        assert crit.last_latency_stats["skip_reason"] == "no_masked_trials"

    def test_paired_variant(self) -> None:
        expected, kwargs = _latency_inputs(
            B=4, groups=[0, 1, 0, 1], prefixes=[10, 20, 30, 40],
            anchors=[0] * 4, targets=[2.0, 0.0, 2.0, 0.0],
            d_values=[5.0, 3.0, 6.0, 4.0], pairs=[0, 0, 1, 1],
        )
        args = _base_loss_args(B=4)
        crit = BioDecisionLoss()
        loss = crit(*args, **kwargs)
        assert loss.item() == pytest.approx(0.0, abs=1e-12)
        assert crit.last_latency_stats["n_complete_pairs"] == 2
        # Remove one reference member: that pair is skipped, one remains.
        kwargs["latency_target_mask"][3] = False
        crit2 = BioDecisionLoss()
        loss2 = crit2(*args, **kwargs)
        assert loss2.item() == pytest.approx(0.0, abs=1e-12)
        assert crit2.last_latency_stats["n_complete_pairs"] == 1
        # Remove both reference members: differentiable zero.
        kwargs["latency_target_mask"][1] = False
        crit3 = BioDecisionLoss()
        loss3 = crit3(*args, **kwargs)
        assert loss3.item() == 0.0
        assert loss3.requires_grad
        # Group presence is checked BEFORE pairing, so masking every
        # reference trial reports the (truthful) missing group.
        assert crit3.last_latency_stats["skip_reason"] == "no_reference_group"

    def test_prefix_cross_group_rejected(self) -> None:
        expected, kwargs = _latency_inputs(groups=[0, 1, 0, 1, 0, 1])
        args = _base_loss_args()
        with pytest.raises(ValueError, match="prefix"):
            BioDecisionLoss()(*args, **kwargs)

    def test_gradient_flows_to_expected_delay(self) -> None:
        # A NONZERO residual is required: at the exact minimum the squared
        # contrast has zero gradient by construction.
        expected, kwargs = _latency_inputs(
            targets=[3.0, 2.0, 0.0, 0.0, 2.0, 0.0],
        )
        args = _base_loss_args()
        loss = BioDecisionLoss()(*args, **kwargs)
        loss.backward()
        assert expected.grad is not None
        assert expected.grad.abs().sum() > 0

    def test_expected_delay_requires_grad_guard(self) -> None:
        expected, kwargs = _latency_inputs()
        kwargs["latency_expected_delay"] = expected.detach()
        args = _base_loss_args()
        with pytest.raises(ValueError, match="requires_grad"):
            BioDecisionLoss()(*args, **kwargs)


# ===============================================================
# 8. Latency sampler / startup coverage
# ===============================================================

class TestLatencySampler:
    @staticmethod
    def _ids():
        # target group 0: prefixes 10, 11, 12, 13 (2 trials each)
        # reference group 1: prefixes 20, 21, 22 (2 trials each)
        groups = [0] * 8 + [1] * 6
        prefixes = (
            [10, 10, 11, 11, 12, 12, 13, 13]
            + [20, 20, 21, 21, 22, 22]
        )
        return groups, prefixes

    def test_each_batch_has_both_groups_and_coverage(self) -> None:
        from nsmor.pipeline.latency_metadata import LatencyContrastBatchSampler

        groups, prefixes = self._ids()
        sampler = LatencyContrastBatchSampler(
            groups, prefixes, batch_size=16,
            target_group=0, reference_group=1,
            min_prefixes_per_group=2, min_trials_per_group=4,
            seed=3, epoch=0,
        )
        seen_t, seen_r = set(), set()
        for batch in sampler:
            batch = list(batch)
            assert len(batch) == 16
            g = np.asarray(groups)[batch]
            p = np.asarray(prefixes)[batch]
            t_pref = set(p[g == 0].tolist())
            r_pref = set(p[g == 1].tolist())
            assert len(t_pref) >= 2 and len(r_pref) >= 2
            assert int((g == 0).sum()) >= 4
            assert int((g == 1).sum()) >= 4
            seen_t |= t_pref
            seen_r |= r_pref
        assert seen_t == {10, 11, 12, 13}
        assert seen_r == {20, 21, 22}

    def test_deterministic_given_seed_and_epoch(self) -> None:
        from nsmor.pipeline.latency_metadata import LatencyContrastBatchSampler

        groups, prefixes = self._ids()
        kw = dict(
            batch_size=16, target_group=0, reference_group=1,
            min_prefixes_per_group=2, min_trials_per_group=4, seed=9,
        )
        a = LatencyContrastBatchSampler(groups, prefixes, epoch=0, **kw)
        b = LatencyContrastBatchSampler(groups, prefixes, epoch=0, **kw)
        assert [list(x) for x in a] == [list(x) for x in b]
        c = LatencyContrastBatchSampler(groups, prefixes, epoch=1, **kw)
        assert [list(x) for x in a] != [list(x) for x in c]

    def test_startup_coverage_fail_closed(self) -> None:
        from nsmor.pipeline.latency_metadata import (
            LatencyContrastBatchSampler,
            validate_latency_coverage,
        )

        groups, prefixes = self._ids()
        with pytest.raises(ValueError, match="prefix"):
            validate_latency_coverage(
                groups, prefixes, target_group=0, reference_group=2,
                min_prefixes_per_group=2, min_trials_per_group=4,
            )
        with pytest.raises(ValueError, match="prefix"):
            LatencyContrastBatchSampler(
                groups, prefixes, batch_size=16,
                target_group=0, reference_group=1,
                min_prefixes_per_group=99, min_trials_per_group=4,
            )
        with pytest.raises(ValueError, match="batch_size"):
            LatencyContrastBatchSampler(
                groups, prefixes, batch_size=8,
                target_group=0, reference_group=1,
                min_prefixes_per_group=2, min_trials_per_group=4,
            )


# ===============================================================
# 9. Resume / backend / default preservation
# ===============================================================

def _train_module():
    spec = importlib.util.spec_from_file_location(
        "nsmor_train_l2", _REPO_ROOT / "scripts" / "train.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestResumeAndBackend:
    def test_architecture_leaves_include_new_options(self) -> None:
        mod = _train_module()
        leaves = set(mod._ARCHITECTURE_CONFIG_LEAVES)
        assert ("model", "time_consuming_seed") in leaves
        assert ("loss", "lambda_latency") in leaves
        ckpt = ExperimentConfig()
        ckpt.model.time_consuming_seed = 1
        ckpt.loss.lambda_latency = 0.0
        active = ExperimentConfig()
        active.model.time_consuming_seed = 2
        with pytest.raises(ValueError, match="time_consuming_seed"):
            mod._require_architecture_config_match(
                {"config": ckpt.to_dict()}, active, Path("ckpt.pth"),
            )
        active = ExperimentConfig()
        active.loss.lambda_latency = 0.5
        with pytest.raises(ValueError, match="lambda_latency"):
            mod._require_architecture_config_match(
                {"config": ckpt.to_dict()}, active, Path("ckpt.pth"),
            )

    def test_lambda_latency_legacy_gap(self) -> None:
        mod = _train_module()
        active = ExperimentConfig()
        active.loss.lambda_latency = 0.5
        with pytest.raises(ValueError, match="lambda_latency"):
            mod._require_architecture_config_match(
                {"config": {"loss": {}}}, active, Path("legacy.pth"),
            )
        off = ExperimentConfig()
        mod._require_architecture_config_match(
            {"config": {"loss": {}}}, off, Path("legacy.pth"),
        )

    def test_latency_provenance_match(self) -> None:
        mod = _train_module()
        cfg = ExperimentConfig()
        cfg.loss.lambda_latency = 1.0
        cfg.model.time_consuming_mode = "stochastic"
        cfg._latency_settings = {
            "latency_target_table_sha256": "abc",
            "latency_target_group": 0, "latency_reference_group": 1,
            "latency_pairing": "none",
            "latency_min_prefixes_per_group": 2,
            "latency_min_trials_per_group": 4,
        }
        ckpt = {
            "eval_mc_samples": 5, "eval_seed_base": 0,
            **cfg._latency_settings,
        }
        mod._require_latency_provenance_match(ckpt, cfg, Path("c.pth"))
        bad = dict(ckpt, eval_mc_samples=3)
        with pytest.raises(ValueError, match="eval_mc_samples"):
            mod._require_latency_provenance_match(bad, cfg, Path("c.pth"))
        legacy = {"eval_seed_base": 0}
        with pytest.raises(ValueError, match="legacy"):
            mod._require_latency_provenance_match(
                legacy, cfg, Path("legacy.pth"),
            )

    def test_eval_mc_samples_resolution(self) -> None:
        mod = _train_module()
        assert mod._resolve_eval_mc_samples("stochastic", None) == 5
        assert mod._resolve_eval_mc_samples("stochastic", 3) == 3
        assert mod._resolve_eval_mc_samples("hard_depth", None) == 1
        assert mod._resolve_eval_mc_samples("hard_depth", 1) == 1
        with pytest.raises(ValueError, match="hard_depth"):
            mod._resolve_eval_mc_samples("hard_depth", 5)
        with pytest.raises(ValueError, match="stochastic"):
            mod._resolve_eval_mc_samples("off", 5)
        assert mod._resolve_eval_mc_samples("off", None) == 1
        assert mod._resolve_eval_mc_samples("depth_delay", 1) == 1
        with pytest.raises(ValueError):
            mod._resolve_eval_mc_samples("stochastic", 0)

    def test_resume_refused_for_new_modes(self) -> None:
        mod = _train_module()
        for mode in ("stochastic", "hard_depth"):
            with pytest.raises(ValueError, match="time_consuming"):
                mod._require_resume_equivalence_for_time_consuming(
                    _model(mode, time_consuming_max_steps=3), Path("c.pth"),
                )

    def test_provenance_keys_registered(self) -> None:
        mod = _train_module()
        for key in (
            "lambda_latency", "eval_mc_samples", "eval_seed_base",
            "latency_target_table_sha256", "latency_target_recipe",
            "latency_anchor_recipe", "latency_condition_mapping",
            "latency_target_group", "latency_reference_group",
            "latency_pairing", "latency_min_prefixes_per_group",
            "latency_min_trials_per_group", "latency_skipped_batches",
            "latency_coverage", "time_consuming_grad_checkpointing",
        ):
            assert key in mod._PROVENANCE_KEYS, key

    @pytest.mark.parametrize("mode", ["stochastic", "hard_depth"])
    def test_backends_refuse_new_modes(self, mode: str) -> None:
        m = _model(mode, time_consuming_max_steps=3)
        m.time_consuming_mode = "off"
        m.backend.time_consuming_mode = "off"
        assert time_consuming_module_present(m)
        from nsmor.analysis.dynamics import FixedPointAdapter

        adapter = FixedPointAdapter(m)
        with pytest.raises(ValueError):
            adapter.compute_full_system_jacobian(
                torch.randn(1, 1, 8), torch.tensor([1]),
            )
        if not JAX_AVAILABLE:
            return
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX
        from nsmor.jax.model import assert_flax_supported
        from nsmor.analysis.jax_eval import JAXEvalWrapper

        with pytest.raises(ValueError, match="time-consuming recursion"):
            NSMoRCoreJAX.from_torch(m)
        with pytest.raises(ValueError, match="time-consuming module present"):
            assert_flax_supported(m, context="test")
        with pytest.raises(ValueError, match="time-consuming"):
            JAXEvalWrapper.from_torch(m)


class TestDefaultPreservation:
    def test_off_matches_pre_change_ref(self) -> None:
        ref_core = _load_module_at_ref(
            PRE_CHANGE_REF, "nsmor/model_nsmor_core.py", "ref_core_l2",
        )
        kwargs = dict(sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0)
        torch.manual_seed(1234)
        new = NSMoRCore(time_consuming_mode="off", **kwargs).eval()
        rng_new = torch.get_rng_state()
        torch.manual_seed(1234)
        ref = ref_core.NSMoRCore(**kwargs).eval()
        rng_ref = torch.get_rng_state()
        assert torch.equal(rng_new, rng_ref)
        sd_new, sd_ref = new.state_dict(), ref.state_dict()
        assert set(sd_new) == set(sd_ref)
        for name in sd_ref:
            assert torch.equal(sd_new[name], sd_ref[name]), name
        new2 = NSMoRCore(time_consuming_mode="off", **kwargs).eval()
        new2.load_state_dict(sd_ref, strict=True)
        x, lengths = _batch()
        with torch.no_grad():
            y_ref = ref(x, lengths)
            y_new = new2(x, lengths)
            _, i_ref = ref(x, lengths, return_internals=True)
            _, i_new = new2(x, lengths, return_internals=True)
        assert torch.equal(y_ref, y_new)
        assert set(i_ref) == set(i_new)
        for key in i_ref:
            assert torch.equal(i_ref[key], i_new[key]), key

    @pytest.mark.parametrize("mode", ["depth_delay", "accumulate"])
    def test_existing_modes_bitwise_vs_ref(self, mode: str) -> None:
        ref_core = _load_module_at_ref(
            PRE_CHANGE_REF, "nsmor/model_nsmor_core.py", f"ref_core_{mode}",
        )
        kwargs = dict(
            sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
            time_consuming_mode=mode, time_consuming_max_steps=4,
            time_consuming_eps=0.05, time_consuming_delay_scale=2.0,
        )
        torch.manual_seed(77)
        new = NSMoRCore(**kwargs).eval()
        torch.manual_seed(77)
        ref = ref_core.NSMoRCore(**kwargs).eval()
        new.load_state_dict(ref.state_dict(), strict=True)
        x, lengths = _batch(B=3, T=10)
        with torch.no_grad():
            y_new, i_new = new(x, lengths, return_internals=True)
            y_ref, i_ref = ref(x, lengths, return_internals=True)
        assert torch.equal(y_new, y_ref)
        for key in i_ref:
            assert torch.equal(i_new[key], i_ref[key]), key

    def test_ref_is_ancestor_of_head(self) -> None:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", PRE_CHANGE_REF, "HEAD"],
            cwd=str(_REPO_ROOT),
        )


# ===============================================================
# 10. K=8/16 and short synthetic training smoke
# ===============================================================

class TestK8AndK16:
    @pytest.mark.parametrize("K", [8, 16])
    @pytest.mark.parametrize("mode", ["stochastic", "hard_depth"])
    def test_numeric_smoke(self, K: int, mode: str) -> None:
        m = _model(
            mode, time_consuming_max_steps=K,
            time_consuming_delay_scale=10.0,
        )
        x, lengths = _batch(B=2, T=12)
        _seed_kw = {"eval_seed": 0} if mode == "stochastic" else {}
        with torch.no_grad():
            y, ints = m(x, lengths, return_internals=True, **_seed_kw)
        assert torch.isfinite(y).all()
        hp = ints["time_consuming_halt_probs"]
        valid = torch.arange(12).unsqueeze(0) < lengths.unsqueeze(1)
        assert torch.allclose(
            _p_sum(hp)[valid],
            torch.ones(int(valid.sum()), dtype=torch.float64),
        )
        e_n = (
            hp.double() * torch.arange(1, K + 1, dtype=torch.float64)
        ).sum(dim=-1)
        assert bool((e_n[valid] >= 1.0).all() and (e_n[valid] <= K).all())
        assert int(ints["time_consuming_updates"]) == int(valid.sum()) * K

    @pytest.mark.parametrize("mode", ["stochastic", "hard_depth"])
    def test_short_training_is_finite(self, mode: str) -> None:
        torch.manual_seed(0)
        model = NSMoRCore(
            sensory_dim=4, mcmc_dim=4, hidden_dim=16, dropout=0.0,
            time_consuming_mode=mode, time_consuming_max_steps=8,
            time_consuming_delay_scale=10.0,
        )
        crit = BioJointLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(8, 10, 8, generator=g)
        y = torch.randn(8, 10, generator=g)
        lengths = torch.full((8,), 10, dtype=torch.long)
        for _ in range(3):
            model.train()
            opt.zero_grad()
            y_pred, ints = model(x, lengths, return_internals=True)
            loss = crit(
                y_pred, y, lengths,
                ints["routing_gates"][:, :, 1:2],
                lambda_reg=0.01,
                out_valid=ints.get("out_valid", None),
            )
            assert torch.isfinite(loss)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all()
            opt.step()

    def test_train_one_epoch_reports_adr0012_counters(self) -> None:
        mod = _train_module()
        torch.manual_seed(0)
        model = NSMoRCore(
            sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
            time_consuming_mode="stochastic", time_consuming_max_steps=3,
        )
        crit = BioJointLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(2)
        x = torch.randn(4, 6, 8, generator=g)
        y = torch.randn(4, 6, generator=g)
        lengths = torch.full((4,), 6, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y, lengths), batch_size=2,
        )
        loss, _health = mod.train_one_epoch(
            model=model, loader=loader, criterion=crit, optimizer=opt,
            device=torch.device("cpu"),
        )
        assert math.isfinite(loss)
        assert mod.train_one_epoch.last_nonfinite_expected == 0
        assert mod.train_one_epoch.last_latency_skipped_batches == 0

    def test_mc_validation_summary(self) -> None:
        mod = _train_module()
        torch.manual_seed(0)
        model = NSMoRCore(
            sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
            time_consuming_mode="stochastic", time_consuming_max_steps=3,
        )
        crit = BioJointLoss()
        g = torch.Generator().manual_seed(1)
        x = torch.randn(6, 7, 8, generator=g)
        y = torch.randn(6, 7, generator=g)
        lengths = torch.full((6,), 7, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y, lengths), batch_size=3,
        )
        val = mod.validate(
            model=model, loader=loader, criterion=crit, device=torch.device("cpu"),
            eval_mc_samples=3, eval_seed_base=50,
        )
        assert math.isfinite(val)
        assert mod.validate.last_eval_mc_samples == 3
        assert math.isfinite(mod.validate.last_val_loss_sd)
        assert mod.validate.last_val_mse is not None
        assert mod.validate.last_nonfinite_expected == 0


# ===============================================================
# 10. Analysis and benchmark scripts
# ===============================================================

class TestLatencyAnalysisScript:
    @staticmethod
    def _fixture(delta: float = 2.0):
        from scripts.analyze_latency import latency_contrast_report

        B, T = 6, 8
        expected = torch.zeros(B, T)
        realized = torch.zeros(B, T)
        # Three target and three reference prefixes (one trial each) so the
        # contrast and the bootstrap are defined.
        groups = torch.tensor([0, 0, 1, 1, 0, 1])
        prefixes = torch.tensor([10, 20, 30, 40, 50, 60])
        targets = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                               dtype=torch.float64)
        for i in range(B):
            # Target-group prefixes carry a +delta residual; reference = 0.
            # Small per-row jitter gives each group a non-zero variance so
            # Cohen's d is defined (the group means still differ by delta).
            base = 2.0 + (delta if int(groups[i]) == 0 else 0.0)
            jitter = 0.1 * i if int(groups[i]) == 0 else 0.05 * i
            expected[i, :] = base + jitter
        kwargs = dict(
            anchor_frames=torch.zeros(B, dtype=torch.long),
            target_frames=targets,
            target_mask=torch.ones(B, dtype=torch.bool),
            group_ids=groups,
            prefix_ids=prefixes,
            lengths=torch.full((B,), T, dtype=torch.long),
            target_group=0,
            reference_group=1,
            n_bootstrap=500,
            bootstrap_seed=3,
        )
        return latency_contrast_report, expected, realized, kwargs

    def test_holm_and_bh_known_values(self) -> None:
        from scripts.analyze_latency import bh_adjust, holm_adjust

        raw = [0.01, 0.04, 0.03]
        assert holm_adjust(raw) == pytest.approx([0.03, 0.06, 0.06])
        assert bh_adjust(raw) == pytest.approx([0.03, 0.04, 0.04])
        for bad in ([], [float("nan")], [-0.1], [1.1]):
            with pytest.raises(ValueError):
                holm_adjust(bad)
            with pytest.raises(ValueError):
                bh_adjust(bad)

    def test_report_recovers_known_contrast(self) -> None:
        fn, expected, realized, kwargs = self._fixture(delta=2.0)
        report = fn(expected, realized, **kwargs)
        assert report["available"] is True
        assert report["delta_d_frames"] == pytest.approx(2.0)
        assert report["delta_t_frames"] == pytest.approx(0.0)
        assert report["delta_residual_frames"] == pytest.approx(2.0)
        assert report["cohens_d"] is not None
        low, high = report["bootstrap_ci95"]
        assert low > 0.0 and high > 0.0
        assert report["adjusted_pvalues"]["holm"] is not None
        assert report["family_size"] == 2
        secondary = report["realized_delay_secondary"]
        assert secondary["delta_realized_delay_frames"] == pytest.approx(0.0)
        # Deterministic report for a fixed bootstrap seed.
        again = fn(expected, realized, **kwargs)
        assert again["bootstrap_ci95"] == report["bootstrap_ci95"]

    def test_prefix_spanning_groups_rejected(self) -> None:
        fn, expected, realized, kwargs = self._fixture()
        kwargs["group_ids"] = torch.tensor([0, 1, 0, 1, 0, 0])
        kwargs["prefix_ids"] = torch.tensor([10, 10, 20, 20, 30, 30])
        with pytest.raises(ValueError, match="spans multiple groups"):
            fn(expected, realized, **kwargs)

    def test_too_few_prefixes_is_unavailable(self) -> None:
        fn, expected, realized, kwargs = self._fixture()
        kwargs["group_ids"] = torch.tensor([0, 0, 1, 1, 1, 1])
        kwargs["prefix_ids"] = torch.tensor([10, 10, 20, 20, 20, 20])
        report = fn(expected, realized, **kwargs)
        assert report["available"] is False
        assert report["reason"] == "fewer_than_two_prefixes_in_a_group"

    def test_degenerate_variance_gives_none_effect_size(self) -> None:
        fn, expected, realized, kwargs = self._fixture(delta=0.0)
        # No jitter: every prefix residual is exactly 0, so the pooled SD is
        # 0 and the effect size is UNDEFINED (None), not infinite.
        expected.fill_(2.0)
        report = fn(expected, realized, **kwargs)
        assert report["available"] is True
        assert report["cohens_d"] is None
        assert report["delta_residual_frames"] == pytest.approx(0.0)


class TestBenchmarkScript:
    def test_synthetic_benchmark_smoke(self) -> None:
        from scripts.benchmark_time_consuming import benchmark

        report = benchmark(
            "hard_depth", K=2, hidden_dim=8, batch=2, timesteps=4,
            repeats=2,
        )
        assert report["synthetic_data_only"] is True
        assert math.isfinite(report["forward_only"]["mean_s"])
        assert math.isfinite(report["forward_backward"]["mean_s"])
        assert math.isfinite(report["loss"])
        with pytest.raises(ValueError, match="mode"):
            benchmark("off")
