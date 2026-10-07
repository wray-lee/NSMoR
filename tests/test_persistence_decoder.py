"""Bounded causal persistence-aware decoder: focused regression tests.

The decoder is a fixed, non-learned residual ``k * v_lag(t)`` added to the
open-loop velocity prediction on active (non-padded) frames only.  These
tests pin the contract that the residual is a *pure additive output* term:

1. Bit-exact legacy parity at ``persistence_skip=0.0`` against the immutable
   historical source object (git commit ``65b9d1a7...``), including strict
   state-dict loading and every internal/state tensor.
2. Fixed additive effect on active frames for finite ``k`` in (0, 1], with
   padding frames and all recurrent internals bit-identical.
3. Gradient parity between a deterministic k=0 model and a nonzero-k model
   (the residual is parameter-free), plus parameter/buffer isolation.
4. Fail-closed validation of ``k``, ``lengths``, feature width and restored
   target-transform metadata across torch, Flax and wrapper entry points.
5. Training and autoregressive entry points reject invalid configurations
   *before* any data/output seam is touched.
6. Real CPU JAX (no torch fallback) reproduces the additive differential.
"""

from __future__ import annotations

import copy
import math
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION
from nsmor.config_parser import ExperimentConfig, ModelConfig, TrainingConfig
from nsmor.model_nsmor_core import NSMoRCore

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HISTORICAL_COMMIT = "65b9d1a7c52e4b31b7225daa44c1011939c8e858"
_HISTORICAL_BLOB = f"{_HISTORICAL_COMMIT}:nsmor/model_nsmor_core.py"

# Reserved feature channel carrying the observed lag velocity (cm/s).
_LAG_CHANNEL = 2


# ── Helpers ──────────────────────────────────────────────────────


def _make_batch(
    b: int = 4,
    t: int = 20,
    sensory_dim: int = 4,
    mcmc_dim: int = 4,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic padded batch with a non-degenerate lag channel."""
    torch.manual_seed(seed)
    total_dim = sensory_dim + mcmc_dim
    x = torch.randn(b, t, total_dim, dtype=torch.float32)
    x[:, :, _LAG_CHANNEL] = torch.linspace(-15.0, 35.0, t).repeat(b, 1)
    lengths = torch.tensor([t - i * 2 for i in range(b)], dtype=torch.long)
    return x, lengths


def _load_historical_module() -> types.ModuleType:
    """Load the immutable pre-decoder source in memory via ``git show``.

    Uses the actual git object rather than a committed duplicate so the
    parity claim cannot drift from history.
    """
    result = subprocess.run(
        ["git", "show", _HISTORICAL_BLOB],
        cwd=str(_REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    source = result.stdout
    assert "persistence_skip" not in source, "historical blob is not pre-decoder"
    module = types.ModuleType("nsmor_historical_core_65b9d1a")
    module.__file__ = _HISTORICAL_BLOB
    exec(compile(source, _HISTORICAL_BLOB, "exec"), module.__dict__)
    return module


def _jax_modules() -> types.SimpleNamespace:
    """Import JAX/Flax backends lazily so a torch-only suite still collects."""
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    from nsmor.analysis.jax_eval import JAXEvalWrapper
    from nsmor.jax.model import NSMoRModel
    from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

    return types.SimpleNamespace(
        JAXEvalWrapper=JAXEvalWrapper,
        NSMoRModel=NSMoRModel,
        NSMoRCoreJAX=NSMoRCoreJAX,
    )


def _clone_with_skip(model: NSMoRCore, k: float) -> NSMoRCore:
    """Return a fresh model sharing ``model``'s exact weights at skip ``k``."""
    clone = NSMoRCore(
        sensory_dim=model.sensory_dim,
        mcmc_dim=model.mcmc_dim,
        hidden_dim=model.hidden_dim,
        dt_ms=model.dt_ms,
        persistence_skip=k,
    )
    clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    return clone


def _parameter_grads(
    model: NSMoRCore,
    x: torch.Tensor,
    lengths: torch.Tensor,
    states: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    """Backprop ``output.sum()`` and return finite-parameter gradient clones."""
    model.zero_grad(set_to_none=True)
    if states is not None:
        out = model(x, lengths, states=states)[0]
    else:
        out = model(x, lengths)
    out.sum().backward()
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


# ── 1. Historical parity ─────────────────────────────────────────


@pytest.mark.parametrize("sensory_dim", [4, 2])
def test_bit_exact_parity_with_historical_object(sensory_dim: int) -> None:
    """k=0 reproduces the immutable pre-decoder object exactly."""
    historical = _load_historical_module()
    x, lengths = _make_batch(sensory_dim=sensory_dim, seed=7)

    torch.manual_seed(1234)
    legacy = historical.NSMoRCore(sensory_dim=sensory_dim, dt_ms=4.0)
    state = copy.deepcopy(legacy.state_dict())

    torch.manual_seed(1234)
    current = NSMoRCore(sensory_dim=sensory_dim, dt_ms=4.0, persistence_skip=0.0)

    assert list(current.state_dict().keys()) == list(state.keys())
    assert sum(p.numel() for p in current.parameters()) == sum(
        p.numel() for p in legacy.parameters()
    )
    report = current.load_state_dict(state, strict=True)
    assert not report.missing_keys and not report.unexpected_keys

    legacy.eval()
    current.eval()
    with torch.no_grad():
        y_hist, int_hist = legacy(x, lengths, return_internals=True)
        y_k0, int_k0 = current(x, lengths, return_internals=True)
        y_hist_s, _, st_hist = legacy(x, lengths, return_internals=True, states={})
        y_k0_s, _, st_k0 = current(x, lengths, return_internals=True, states={})

    assert torch.equal(y_hist, y_k0)
    assert torch.equal(y_hist_s, y_k0_s)
    assert int_hist.keys() == int_k0.keys()
    for key in int_hist:
        assert torch.equal(int_hist[key], int_k0[key]), key
    assert st_hist.keys() == st_k0.keys()
    for key in st_hist:
        assert torch.equal(st_hist[key], st_k0[key]), key


@pytest.mark.parametrize("sensory_dim", [4, 2])
def test_default_and_explicit_zero_skip_are_bit_exact(sensory_dim: int) -> None:
    """Omitted and explicit ``persistence_skip=0.0`` build the same system."""
    x, lengths = _make_batch(sensory_dim=sensory_dim, seed=8)
    torch.manual_seed(55)
    default = NSMoRCore(sensory_dim=sensory_dim, dt_ms=4.0).eval()
    torch.manual_seed(55)
    explicit = NSMoRCore(
        sensory_dim=sensory_dim, dt_ms=4.0, persistence_skip=0.0
    ).eval()

    with torch.no_grad():
        y_d, i_d = default(x, lengths, return_internals=True)
        y_e, i_e = explicit(x, lengths, return_internals=True)

    assert torch.equal(y_d, y_e)
    for key in i_d:
        assert torch.equal(i_d[key], i_e[key]), key


@pytest.mark.parametrize("sensory_dim", [4, 2])
def test_historical_checkpoint_loads_strictly(
    tmp_path: Path, sensory_dim: int
) -> None:
    """A pre-decoder checkpoint payload restores strictly at k=0."""
    from nsmor.model_utils import load_model_from_checkpoint

    historical = _load_historical_module()
    torch.manual_seed(11)
    legacy = historical.NSMoRCore(
        sensory_dim=sensory_dim, hidden_dim=16, dt_ms=4.0
    )
    state = copy.deepcopy(legacy.state_dict())
    payload = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "config": {
            "model": {
                "sensory_dim": sensory_dim,
                "mcmc_dim": 4,
                "hidden_dim": 16,
                "dt_ms": 4.0,
            }
        },
        "model_state_dict": state,
    }
    path = tmp_path / "historical.pth"
    torch.save(payload, path)

    model = load_model_from_checkpoint(path, torch.device("cpu"))
    assert model.persistence_skip == 0.0
    restored = model.state_dict()
    assert list(restored.keys()) == list(state.keys())
    for key, tensor in state.items():
        assert torch.equal(restored[key], tensor), key


# ── 2. Additive effect and isolation ─────────────────────────────


@pytest.mark.parametrize("k", [0.5, 1.0])
def test_nonzero_k_adds_scaled_lag_on_active_frames(k: float) -> None:
    """Nonzero k strictly adds ``k * v_lag`` on active frames only."""
    x, lengths = _make_batch(b=3, t=15, seed=5)
    base = NSMoRCore(dt_ms=4.0, persistence_skip=0.0).eval()
    model = _clone_with_skip(base, k).eval()

    with torch.no_grad():
        y0, i0 = base(x, lengths, return_internals=True)
        yk, ik = model(x, lengths, return_internals=True)

    for key in i0:
        assert torch.equal(i0[key], ik[key]), key

    v_lag = x[:, :, _LAG_CHANNEL]
    for b in range(y0.shape[0]):
        n = int(lengths[b])
        assert torch.allclose(yk[b, :n] - y0[b, :n], k * v_lag[b, :n], atol=1e-6)
        if n < y0.shape[1]:
            assert torch.equal(yk[b, n:], y0[b, n:])


def test_supplied_states_and_internals_unchanged_by_skip() -> None:
    """Carried recurrent states are unaffected by the output residual."""
    x, lengths = _make_batch(b=2, t=6, seed=9)
    base = NSMoRCore(dt_ms=4.0, persistence_skip=0.0).eval()
    model = _clone_with_skip(base, 0.5).eval()

    with torch.no_grad():
        y0, i0, s0 = base(x, lengths, return_internals=True, states={})
        yk, ik, sk = model(x, lengths, return_internals=True, states={})

    assert s0.keys() == sk.keys()
    for key in s0:
        assert torch.equal(s0[key], sk[key]), key
    for key in i0:
        assert torch.equal(i0[key], ik[key]), key
    assert not torch.equal(y0, yk)


def test_nonempty_supplied_carry_isolated_from_residual() -> None:
    """A nonempty restored carry drives both models identically and is not
    mutated; the residual still appears only on active output frames."""
    x, lengths = _make_batch(b=2, t=6, seed=23)
    base = NSMoRCore(dt_ms=4.0, persistence_skip=0.0).eval()
    model = _clone_with_skip(base, 0.5).eval()

    with torch.no_grad():
        _, _, warm = base(x, lengths, return_internals=True, states={})
    carry = {key: value.clone() for key, value in warm.items()}
    assert carry, "carry must be nonempty"

    with torch.no_grad():
        y0, i0, s0 = base(x, lengths, return_internals=True, states=carry)
        yk, ik, sk = model(x, lengths, return_internals=True, states=carry)

    for key in s0:
        assert torch.equal(s0[key], sk[key]), key
    for key in i0:
        assert torch.equal(i0[key], ik[key]), key
    for key in carry:
        assert torch.equal(carry[key], warm[key]), key

    v_lag = x[:, :, _LAG_CHANNEL]
    for b in range(2):
        n = int(lengths[b])
        assert torch.allclose(yk[b, :n] - y0[b, :n], 0.5 * v_lag[b, :n], atol=1e-6)
        assert torch.equal(yk[b, n:], y0[b, n:])


def test_nan_padding_isolated_by_residual_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """NaN in padded lag frames must not leak through the residual mask.

    The encoder is blinded to the lag channel (input zeroed, not the weight,
    so ``0 * NaN`` cannot reappear) and the baseline is required to be
    finite.  A masked multiply instead of ``where`` would surface NaN.
    """
    x, lengths = _make_batch(b=3, t=12, seed=21)
    padded = torch.arange(x.shape[1]).unsqueeze(0) >= lengths.unsqueeze(1)
    x = x.clone()
    x[:, :, _LAG_CHANNEL] = torch.where(
        padded, torch.full_like(x[:, :, _LAG_CHANNEL], float("nan")),
        x[:, :, _LAG_CHANNEL],
    )

    def _blind_forward(self: Any, sensory: torch.Tensor) -> torch.Tensor:
        clean = sensory.clone()
        clean[..., _LAG_CHANNEL] = 0.0
        return self.net(clean)

    base = NSMoRCore(dt_ms=4.0, persistence_skip=0.0).eval()
    model = _clone_with_skip(base, 0.5).eval()
    for target in (base, model):
        monkeypatch.setattr(
            target.sensory_encoder,
            "forward",
            types.MethodType(_blind_forward, target.sensory_encoder),
        )

    with torch.no_grad():
        y0, i0 = base(x, lengths, return_internals=True)
        yk, ik = model(x, lengths, return_internals=True)

    assert torch.isfinite(y0).all()
    assert torch.isfinite(yk).all()
    for key in i0:
        assert torch.equal(i0[key], ik[key]), key

    v_lag = x[:, :, _LAG_CHANNEL]
    for b in range(y0.shape[0]):
        n = int(lengths[b])
        assert torch.allclose(yk[b, :n] - y0[b, :n], 0.5 * v_lag[b, :n], atol=1e-6)
        assert torch.equal(yk[b, n:], y0[b, n:])


# ── 3. Gradient parity and parameter isolation ───────────────────


def test_gradient_parity_and_parameter_isolation() -> None:
    """The residual is parameter-free; grads match a deterministic k=0 model.

    The current core does **not** detach the frontend: gradient isolation
    between stages is done by ``requires_grad`` toggling in the training
    script, so a single-phase backward reaches every parameter.
    """
    x, lengths = _make_batch(b=2, t=8, seed=3)
    base = NSMoRCore(dt_ms=4.0, persistence_skip=0.0).eval()
    model = _clone_with_skip(base, 0.75).eval()

    assert not any(n == "persistence_skip" for n, _ in model.named_parameters())
    assert not any(n == "persistence_skip" for n, _ in model.named_buffers())

    grads0 = _parameter_grads(base, x, lengths)
    grads_k = _parameter_grads(model, x, lengths)

    assert grads0.keys() == grads_k.keys()
    assert grads0, "no parameter received a gradient"
    for name in grads0:
        assert torch.isfinite(grads_k[name]).all(), name
        assert torch.equal(grads0[name], grads_k[name]), name

    assert (model.direction_head.net[3].weight.grad != 0).any()
    assert (model.backend.gru_unit.gru.weight_hh_l0.grad != 0).any()
    frontend_grads = [n for n in grads_k if n.startswith("frontend")]
    if frontend_grads:
        assert any((grads_k[n] != 0).any() for n in frontend_grads)

    # A nonempty restored carry must not change the gradient picture either.
    with torch.no_grad():
        _, _, warm = base(x, lengths, return_internals=True, states={})
    carry = {key: value.clone() for key, value in warm.items()}
    grads0_carry = _parameter_grads(base, x, lengths, states=carry)
    grads_k_carry = _parameter_grads(model, x, lengths, states=carry)
    assert grads0_carry.keys() == grads_k_carry.keys()
    for name in grads0_carry:
        assert torch.equal(grads0_carry[name], grads_k_carry[name]), name


# ── 4. Validation gates ──────────────────────────────────────────


@pytest.mark.parametrize(
    "bad", [-0.1, 1.5, float("nan"), float("inf"), True, "0.5"]
)
def test_model_and_config_reject_invalid_k(bad: Any) -> None:
    with pytest.raises(ValueError, match="persistence_skip must be"):
        NSMoRCore(dt_ms=4.0, persistence_skip=bad)
    with pytest.raises(ValueError, match="persistence_skip must be"):
        ModelConfig(persistence_skip=bad)


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([[5, 5], [5, 5]]),   # shape
        torch.tensor([True, False]),      # bool
        torch.tensor([5.0, 3.0]),         # floating
        torch.tensor([6, 5]),             # > T
        torch.tensor([-1, 5]),            # < 0
    ],
)
def test_nonzero_k_rejects_invalid_lengths(bad: torch.Tensor) -> None:
    model = NSMoRCore(dt_ms=4.0, persistence_skip=0.5)
    x = torch.randn(2, 5, 8)
    with pytest.raises((ValueError, AssertionError), match="lengths"):
        model(x, bad)


def test_nonzero_k_rejects_wrong_feature_width() -> None:
    model = NSMoRCore(dt_ms=4.0, persistence_skip=0.5)
    with pytest.raises(ValueError, match="Expected feature dim"):
        model(torch.randn(2, 5, 7), torch.tensor([5, 5]))


def test_nonzero_k_rejects_narrow_sensory_dim() -> None:
    """The lag channel is only defined for sensory_dim >= 3."""
    model = NSMoRCore(sensory_dim=2, mcmc_dim=4, dt_ms=4.0, persistence_skip=0.5)
    with pytest.raises(ValueError, match="sensory_dim >= 3"):
        model(torch.randn(2, 5, 6), torch.tensor([5, 5]))

    mods = _jax_modules()
    wrapper = mods.JAXEvalWrapper.from_torch(model)
    with pytest.raises(ValueError, match="sensory_dim >= 3"):
        wrapper(torch.randn(2, 5, 6), torch.tensor([5, 5]))


def test_jax_runner_rejects_invalid_lengths() -> None:
    """Converted JAX runner validates host lengths before the JIT boundary."""
    mods = _jax_modules()
    model = NSMoRCore(dt_ms=4.0, persistence_skip=0.5).eval()
    runner = mods.NSMoRCoreJAX.from_torch(model)
    x = torch.randn(2, 5, 8)

    with pytest.raises(ValueError, match="lengths must have an integer"):
        runner(x, torch.tensor([5.0, 3.0]))
    with pytest.raises(ValueError, match="lengths must satisfy"):
        runner(x, torch.tensor([6, 5]))


def _stub_core_io(model: NSMoRCore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace frontend/backend with shape-correct stubs for guard-only tests."""

    def _frontend(sensory: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return sensory[..., :1].expand(-1, -1, model.hidden_dim)

    def _backend(
        e: torch.Tensor,
        mcmc: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_internals: bool = False,
        override_gates: Any = None,
        states: Any = None,
    ) -> Any:
        y = torch.zeros(e.shape[0], e.shape[1], dtype=torch.float32)
        if not return_internals:
            return y
        internals = {
            "routing_gates": torch.zeros(e.shape[0], e.shape[1], 2),
        }
        return (y, internals, {}) if states is not None else (y, internals)

    monkeypatch.setattr(model.frontend, "forward", _frontend)
    monkeypatch.setattr(model.backend, "forward", _backend)


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int16])
@pytest.mark.parametrize("carry", [False, True])
def test_narrow_integer_lengths_do_not_overflow_range_check(
    dtype: torch.dtype, carry: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Narrow integer dtypes must be widened before the ``<= T`` comparison.

    ``torch.tensor([100], dtype=int8) <= 2400`` is False because int8 cannot
    hold 2400; an unwidened comparison would wrongly reject a valid length
    and wrongly accept an out-of-range one.  The frontend/backend are stubbed
    so the check is exercised without a 2400-step recurrent loop, with and
    without a nonempty supplied carry.
    """
    model = NSMoRCore(dt_ms=4.0, persistence_skip=0.5)
    x = torch.zeros(1, 2400, 8)
    _stub_core_io(model, monkeypatch)
    states = {} if carry else None

    model(x, torch.tensor([100], dtype=dtype), states=states)  # valid, in range

    with pytest.raises(ValueError, match="lengths must satisfy"):
        model(x, torch.tensor([2401], dtype=torch.int16), states=states)


@pytest.mark.parametrize(
    "dtype,timesteps", [(torch.int8, 140), (torch.uint8, 260)]
)
def test_narrow_lengths_preserve_real_recurrent_padding(
    dtype: torch.dtype, timesteps: int,
) -> None:
    """Validated lengths remain wide through the real recurrent mask."""
    torch.manual_seed(31)
    model = NSMoRCore(hidden_dim=8, dt_ms=4.0, persistence_skip=0.5).eval()
    x = torch.randn(1, timesteps, 8)
    x[..., _LAG_CHANNEL] = 2.0
    wide = torch.tensor([100], dtype=torch.int64)
    narrow = wide.to(dtype)
    with torch.no_grad():
        y_wide, i_wide, s_wide = model(x, wide, states={})
        y_narrow, i_narrow, s_narrow = model(x, narrow, states={})
    assert torch.equal(y_wide, y_narrow)
    for key in i_wide:
        assert torch.equal(i_wide[key], i_narrow[key]), key
    for key in s_wide:
        assert torch.equal(s_wide[key], s_narrow[key]), key
    for key in ("lif_potentials", "lif_spikes", "lif_thresholds"):
        assert torch.count_nonzero(i_narrow[key][:, 100:]) == 0, key


def test_forced_fused_fallback_rejects_invalid_lengths_before_cast() -> None:
    """With JAX disabled the fused runner still preflights lengths on the host
    before its integer cast to torch.long."""
    mods = _jax_modules()
    model = NSMoRCore(dt_ms=4.0, persistence_skip=0.5).eval()
    runner = mods.NSMoRCoreJAX.from_torch(model)
    runner.use_jax = False
    x = torch.randn(2, 5, 8)

    with pytest.raises(ValueError, match="lengths must have an integer"):
        runner(x, torch.tensor([5.0, 3.0]))
    with pytest.raises(ValueError, match="lengths must satisfy"):
        runner(x, torch.tensor([6, 5]))


def test_restored_target_transform_gate() -> None:
    """Restored normalization or any nonzero clip is rejected for k > 0."""
    model = NSMoRCore(dt_ms=4.0, persistence_skip=0.5)
    x, lengths = _make_batch(b=2, t=6, seed=17)

    model.target_clip_cm_s = 0.0
    model(x, lengths)  # exact-zero clip is the identity and stays valid

    for clip in (-1.0, float("nan"), 100.0):
        model.target_clip_cm_s = clip
        with pytest.raises(ValueError, match="unnormalized/unclipped"):
            model(x, lengths)
    model.target_clip_cm_s = 0.0

    model.target_mean = 1.5
    with pytest.raises(ValueError, match="unnormalized/unclipped"):
        model(x, lengths)
    model.target_mean = 0.0

    model.backend.target_std = 2.0
    with pytest.raises(ValueError, match="unnormalized/unclipped"):
        model(x, lengths)


def test_experiment_config_rejects_normalized_or_clipped_k() -> None:
    for training in (
        TrainingConfig(normalize_targets=True, target_clip_cm_s=0.0),
        TrainingConfig(normalize_targets=False, target_clip_cm_s=100.0),
        TrainingConfig(normalize_targets=False, target_clip_cm_s=-1.0),
        TrainingConfig(normalize_targets=False, target_clip_cm_s=float("nan")),
    ):
        cfg = ExperimentConfig(
            model=ModelConfig(persistence_skip=0.5), training=training
        )
        with pytest.raises(ValueError, match="unnormalized and unclipped"):
            cfg.validate()

    valid = ExperimentConfig(
        model=ModelConfig(persistence_skip=0.5),
        training=TrainingConfig(normalize_targets=False, target_clip_cm_s=0.0),
    )
    valid.validate()


# ── 5. Training and autoregressive seams ─────────────────────────


def test_train_rejects_invalid_config_before_data_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import train as train_module

    reached: list[int] = []

    def _boom(*args: Any, **kwargs: Any) -> Any:
        reached.append(1)
        raise AssertionError("data seam reached before config rejection")

    monkeypatch.setattr(train_module, "create_dataloaders_from_config", _boom)
    cfg = ExperimentConfig(
        model=ModelConfig(persistence_skip=0.5),
        training=TrainingConfig(normalize_targets=True, target_clip_cm_s=0.0),
    )
    with pytest.raises(ValueError, match="unnormalized and unclipped"):
        train_module.train(cfg)
    assert not reached


def test_train_jax_rejects_invalid_config_before_data_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jax_train = pytest.importorskip("nsmor.jax.train")

    reached: list[int] = []

    def _boom(*args: Any, **kwargs: Any) -> Any:
        reached.append(1)
        raise AssertionError("dataset seam reached before config rejection")

    monkeypatch.setattr(jax_train, "load_nsmor_dataset", _boom)
    cfg = ExperimentConfig(
        model=ModelConfig(persistence_skip=0.5),
        training=TrainingConfig(normalize_targets=True, target_clip_cm_s=0.0),
    )
    with pytest.raises(ValueError, match="unnormalized and unclipped"):
        jax_train.train_jax(cfg)
    assert not reached


def test_train_jax_accepts_legacy_duck_typed_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``JAXExperimentConfig`` has no ``validate``; a valid k=0 config still
    reaches the dataset seam, an invalid one is rejected before it."""
    jax_train = pytest.importorskip("nsmor.jax.train")
    from nsmor.jax.config import JAXExperimentConfig

    reached: list[int] = []

    def _seam(*args: Any, **kwargs: Any) -> Any:
        reached.append(1)
        raise RuntimeError("dataset seam")

    monkeypatch.setattr(jax_train, "load_nsmor_dataset", _seam)
    cfg = JAXExperimentConfig()
    cfg.checkpoint.output_dir = str(tmp_path / "out")
    with pytest.raises(RuntimeError, match="dataset seam"):
        jax_train.train_jax(cfg)
    assert reached

    reached.clear()
    bad = JAXExperimentConfig(
        model=ModelConfig(persistence_skip=0.5),
        training=TrainingConfig(normalize_targets=True, target_clip_cm_s=0.0),
    )
    with pytest.raises(ValueError, match="unnormalized and unclipped"):
        jax_train.train_jax(bad)
    assert not reached


def test_autoregressive_function_guard_fails_closed() -> None:
    from scripts.simulate_autoregressive import PARADIGMS, run_autoregressive_trial

    model = NSMoRCore(dt_ms=4.0, persistence_skip=0.5).eval()
    with pytest.raises(ValueError, match="does not support nonzero persistence_skip"):
        run_autoregressive_trial(
            model=model,
            paradigm=PARADIGMS["visual_only"],
            mcmc_prior=np.array([0.25, 0.25, 0.25, 0.25]),
            device=torch.device("cpu"),
        )


def test_autoregressive_cli_guard_fails_closed_before_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import simulate_autoregressive as ar

    model = NSMoRCore(dt_ms=4.0, persistence_skip=0.5).eval()
    monkeypatch.setattr(ar, "load_model_from_checkpoint", lambda *a, **k: model)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simulate_autoregressive.py",
            "--checkpoint",
            str(tmp_path / "unused.pth"),
            "--paradigms",
            "visual_only",
            "--output_dir",
            str(out_dir),
        ],
    )
    with pytest.raises(ValueError, match="does not support nonzero persistence_skip"):
        ar.main()
    assert not out_dir.exists()


# ── 6. JAX / Flax additive differential ──────────────────────────


def test_jax_wrapper_rejects_restored_normalization() -> None:
    mods = _jax_modules()
    model = NSMoRCore(dt_ms=4.0, persistence_skip=0.5)
    model.target_std = 2.0
    with pytest.raises(ValueError, match="requires physical unnormalized"):
        mods.JAXEvalWrapper.from_torch(model)


@pytest.mark.parametrize("consumer", ["self", "backend"])
def test_jax_converted_paths_reject_backend_metadata(consumer: str) -> None:
    """Restored metadata on *either* the model or its backend is rejected."""
    mods = _jax_modules()
    model = NSMoRCore(dt_ms=4.0, persistence_skip=0.5)
    target = model if consumer == "self" else model.backend
    target.target_mean = 1.0

    with pytest.raises(ValueError, match="requires physical unnormalized"):
        mods.JAXEvalWrapper.from_torch(model)

    runner = mods.NSMoRCoreJAX.from_torch(model)
    x, lengths = _make_batch(b=2, t=6, seed=13)
    with pytest.raises(ValueError, match="requires physical unnormalized"):
        runner(x, lengths)


def test_jax_fused_runner_rejects_lengths_batch_mismatch() -> None:
    """Fused runner keeps its lengths batch-shape assertion."""
    mods = _jax_modules()
    model = NSMoRCore(dt_ms=4.0, persistence_skip=0.5).eval()
    runner = mods.NSMoRCoreJAX.from_torch(model)
    x, _ = _make_batch(b=3, t=6, seed=15)
    with pytest.raises(AssertionError, match="Expected lengths"):
        runner(x, torch.tensor([6, 6]))


def test_jax_backends_additive_differential() -> None:
    """NSMoRCoreJAX and JAXEvalWrapper reproduce the residual on CPU JAX."""
    mods = _jax_modules()
    x, lengths = _make_batch(b=2, t=10, seed=31)
    base = NSMoRCore(dt_ms=4.0, persistence_skip=0.0).eval()
    model = _clone_with_skip(base, 0.5).eval()

    v_lag = x[:, :, _LAG_CHANNEL]
    for factory in (mods.NSMoRCoreJAX.from_torch, mods.JAXEvalWrapper.from_torch):
        runner0 = factory(base)
        runner_k = factory(model)
        assert getattr(runner_k, "persistence_skip") == 0.5
        if hasattr(runner_k, "use_jax"):
            assert runner_k.use_jax is True

        y0 = runner0(x, lengths)
        yk = runner_k(x, lengths)
        assert yk.shape == (2, 10)
        assert torch.isfinite(y0).all() and torch.isfinite(yk).all()

        for b in range(2):
            n = int(lengths[b])
            assert torch.allclose(
                yk[b, :n] - y0[b, :n], 0.5 * v_lag[b, :n], atol=1e-5
            )
            if n < yk.shape[1]:
                assert torch.allclose(yk[b, n:], y0[b, n:], atol=1e-6)

        for bad in (
            torch.tensor([10.0, 8.0]),  # non-integer
            torch.tensor([11, 8]),       # > T
        ):
            with pytest.raises(ValueError, match="lengths"):
                runner_k(x, bad)


def test_flax_model_direct_valid_and_invalid() -> None:
    """Direct Flax ``apply``: identical params differ only by the residual."""
    mods = _jax_modules()  # importorskip runs before any direct jax import
    import jax
    import jax.numpy as jnp
    from nsmor.jax.model import validate_lengths

    x, lengths = _make_batch(b=2, t=10, seed=41)
    x_j = jnp.asarray(x.numpy())
    l_j = jnp.asarray(lengths.numpy(), dtype=jnp.int32)

    model0 = mods.NSMoRModel(
        sensory_dim=4, mcmc_dim=4, hidden_dim=16, dt_ms=4.0, persistence_skip=0.0
    )
    model_k = mods.NSMoRModel(
        sensory_dim=4, mcmc_dim=4, hidden_dim=16, dt_ms=4.0, persistence_skip=0.5
    )
    params = model0.init(jax.random.PRNGKey(0), x_j, l_j)
    y0 = np.asarray(model0.apply(params, x_j, l_j))
    y = np.asarray(model_k.apply(params, x_j, l_j))
    assert y.shape == (2, 10)
    assert np.isfinite(y0).all() and np.isfinite(y).all()

    v_lag = np.asarray(x[:, :, _LAG_CHANNEL])
    for b in range(2):
        n = int(lengths[b])
        np.testing.assert_allclose(y[b, :n] - y0[b, :n], 0.5 * v_lag[b, :n],
                                   atol=1e-5)
        if n < y.shape[1]:
            np.testing.assert_allclose(y[b, n:], y0[b, n:], atol=1e-6)

    for bad_k in (1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite scalar in"):
            invalid = mods.NSMoRModel(
                sensory_dim=4, mcmc_dim=4, hidden_dim=16, persistence_skip=bad_k
            )
            invalid.init(jax.random.PRNGKey(0), x_j, l_j)

    for bad in (
        np.array([10, 11], dtype=np.int32),   # > T
        np.array([10.0, 8.0], dtype=np.float32),  # non-integer
        np.array([True, False], dtype=np.bool_),  # boolean
    ):
        with pytest.raises((ValueError, AssertionError), match="lengths"):
            model_k.apply(params, x_j, jnp.asarray(bad))

    # Host preflight is the supported contract for direct jitted callers.
    validate_lengths(np.asarray(lengths), 2, 10)
    jitted = jax.jit(lambda p, xx, ll: model_k.apply(p, xx, ll))
    y_jit = jitted(params, x_j, l_j)
    assert np.isfinite(np.asarray(y_jit)).all()
    assert math.isclose(
        float(np.abs(np.asarray(y_jit) - y).max()), 0.0, abs_tol=1e-5
    )
