"""Committed regression tests for the fail-closed cells in the real-data
preliminary protocol (§5 availability table).

These pin the runtime behaviour the protocol claims, so the availability
table cannot silently drift:

* A full-system Jacobian is refused whenever a refinement module is present.
* The Flax forward-eval wrapper refuses an enabled refinement mode.
* The §6 compute-accounting honours the core contract (``refinement_depth``
  ``(B,T)``, ``refinement_updates`` a 0-dim scalar) and fails closed on a
  validation condition outside the declared wind/visual classes.
* The arm-identity guard binds each checkpoint to its declared arm.

No real data, no training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nsmor.analysis.dynamics import (  # noqa: E402
    assert_no_adaptive_refinement_for_full_jacobian,
)
from nsmor.model_nsmor_core import NSMoRCore, refinement_module_present  # noqa: E402


def _model(mode: str) -> NSMoRCore:
    return NSMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
        refinement_mode=mode,
    ).eval()


# ── Full-system Jacobian refusal ────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["adaptive", "fixed"])
def test_full_system_jacobian_refuses_refinement(mode: str) -> None:
    model = _model(mode)
    assert refinement_module_present(model)
    with pytest.raises(ValueError):
        assert_no_adaptive_refinement_for_full_jacobian(model, context="test")


def test_full_system_jacobian_allows_off() -> None:
    model = _model("off")
    assert not refinement_module_present(model)
    # Must not raise.
    assert_no_adaptive_refinement_for_full_jacobian(model, context="test")


# ── Flax forward-eval refusal ───────────────────────────────────────────────
def test_jax_wrap_refuses_adaptive() -> None:
    jax_eval = pytest.importorskip("nsmor.analysis.jax_eval")
    if not jax_eval.JAX_AVAILABLE:
        pytest.skip("JAX not installed; wrap_eval_model falls back to PyTorch")
    model = _model("adaptive")
    with pytest.raises(ValueError):
        jax_eval.wrap_eval_model(model, backend="jax")


def test_jax_wrap_allows_off() -> None:
    jax_eval = pytest.importorskip("nsmor.analysis.jax_eval")
    if not jax_eval.JAX_AVAILABLE:
        pytest.skip("JAX not installed; wrap_eval_model falls back to PyTorch")
    model = _model("off")
    # Must not raise; returns a wrapper (or the model unchanged).
    assert jax_eval.wrap_eval_model(model, backend="jax") is not None


# ── Arm identity binding (protocol §2) ──────────────────────────────────────
def _cfg(activation: str, mode: str, lambda_compute: float):
    from nsmor.config_parser import ExperimentConfig

    return ExperimentConfig.from_dict({
        "model": {"activation": activation, "refinement_mode": mode},
        "loss": {"lambda_compute": lambda_compute},
    })


def test_arm_identity_accepts_declared_arms() -> None:
    from evaluate_realdata_preliminary import _validate_arm_config

    _validate_arm_config(_cfg("swiglu", "off", 0.0), "A1")       # must not raise
    _validate_arm_config(_cfg("swiglu", "adaptive", 0.01), "A2")  # must not raise


def test_arm_identity_rejects_swapped_or_wrong_arm() -> None:
    from evaluate_realdata_preliminary import _validate_arm_config

    # A2 config handed in as A1, and a relu/off config handed in as A1.
    with pytest.raises(ValueError):
        _validate_arm_config(_cfg("swiglu", "adaptive", 0.01), "A1")
    with pytest.raises(ValueError):
        _validate_arm_config(_cfg("relu", "off", 0.0), "A1")
    with pytest.raises(ValueError):
        _validate_arm_config(_cfg("swiglu", "off", 0.01), "A1")  # wrong lambda


# ── §6 compute accounting: condition-split math ─────────────────────────────
def _stamp(monkeypatch, E, conditions) -> None:
    """Pin the authoritative val condition stamp the accounting reads."""
    arr = np.asarray(conditions, dtype=object)
    monkeypatch.setattr(E, "_val_stimulus_conditions", lambda *a, **k: arr)


def test_condition_accounting_splits_by_wind_mask(monkeypatch) -> None:
    import evaluate_realdata_preliminary as E

    B, T = 2, 4
    # Row 0 wind_only (depth all 3), row 1 visual (depth all 1).
    depth = torch.tensor([[3.0, 3, 3, 3], [1.0, 1, 1, 1]])
    x = torch.zeros(B, T, 8)
    y = torch.zeros(B, T)
    lengths = torch.tensor([T, T])
    wind = torch.tensor([True, False])

    class _M(torch.nn.Module):
        def eval(self):  # noqa: D401
            return self

        def forward(self, xb, lb, return_internals=True):  # noqa: D401
            return torch.zeros(xb.size(0), xb.size(1)), {
                "refinement_depth": depth,
                # Core contract: updates is a 0-dim scalar (depth.sum()).
                "refinement_updates": depth.sum(),
                "refinement_ponder_cost": torch.tensor(2.0),
            }

    _stamp(monkeypatch, E, ["wind_only", "visual_only"])
    monkeypatch.setattr(
        E, "_build_model_and_loader",
        lambda *a, **k: (None, _M(), [(x, y, lengths, wind)]),
    )
    out = E.compute_condition_accounting(
        Path("ckpt"), Path("ds"), Path("np"), torch.device("cpu")
    )
    assert out["wind_only"]["n_valid_frames"] == T
    assert out["visual_present"]["n_valid_frames"] == T
    assert out["wind_only"]["mean_refinement_depth"] == 3.0
    assert out["visual_present"]["mean_refinement_depth"] == 1.0
    # Per-condition update totals are per-frame depth sums (updates==depth.sum).
    assert out["wind_only"]["total_refinement_updates"] == 12.0
    assert out["visual_present"]["total_refinement_updates"] == 4.0
    assert out["pooled"]["mean_refinement_depth"] == 2.0
    assert out["pooled"]["mean_refinement_ponder_cost"] == 2.0
    assert out["ponder_cost_scope"] == "pooled_only"


def test_condition_accounting_rejects_non_scalar_updates(monkeypatch) -> None:
    """A (B,T) 'updates' (the old mock contract) must be refused, not scored."""
    import evaluate_realdata_preliminary as E

    depth = torch.full((1, 2), 2.0)

    class _M(torch.nn.Module):
        def eval(self):  # noqa: D401
            return self

        def forward(self, xb, lb, return_internals=True):  # noqa: D401
            return torch.zeros(xb.size(0), xb.size(1)), {
                "refinement_depth": depth,
                "refinement_updates": depth,  # WRONG: not a scalar
                "refinement_ponder_cost": torch.tensor(1.0),
            }

    _stamp(monkeypatch, E, ["wind_only"])
    monkeypatch.setattr(
        E, "_build_model_and_loader",
        lambda *a, **k: (
            None, _M(),
            [(torch.zeros(1, 2, 8), torch.zeros(1, 2), torch.tensor([2]),
              torch.tensor([True]))],
        ),
    )
    with pytest.raises(AssertionError):
        E.compute_condition_accounting(
            Path("ckpt"), Path("ds"), Path("np"), torch.device("cpu")
        )


def test_condition_accounting_without_mask_degrades_to_pooled(monkeypatch) -> None:
    """A 3-tuple batch (no wind mask) must not abort the driver."""
    import evaluate_realdata_preliminary as E

    depth = torch.full((1, 2), 2.0)

    class _M(torch.nn.Module):
        def eval(self):  # noqa: D401
            return self

        def forward(self, xb, lb, return_internals=True):  # noqa: D401
            return torch.zeros(xb.size(0), xb.size(1)), {
                "refinement_depth": depth,
                "refinement_updates": depth.sum(),
                "refinement_ponder_cost": torch.tensor(1.5),
            }

    _stamp(monkeypatch, E, ["visual_only"])
    monkeypatch.setattr(
        E, "_build_model_and_loader",
        lambda *a, **k: (None, _M(), [(torch.zeros(1, 2, 8), torch.zeros(1, 2), torch.tensor([2]))]),
    )
    out = E.compute_condition_accounting(
        Path("ckpt"), Path("ds"), Path("np"), torch.device("cpu")
    )
    assert out["condition_split_available"] is False
    assert out["condition_split_unavailable_reason"] == "no_wind_only_mask"
    assert out["wind_only"] is None and out["visual_present"] is None
    assert out["pooled"]["mean_refinement_depth"] == 2.0


def test_condition_accounting_empty_group_updates_are_null(monkeypatch) -> None:
    """An empty condition group reports total_updates None, not 0.0."""
    import evaluate_realdata_preliminary as E

    depth = torch.full((2, 2), 3.0)

    class _M(torch.nn.Module):
        def eval(self):  # noqa: D401
            return self

        def forward(self, xb, lb, return_internals=True):  # noqa: D401
            return torch.zeros(xb.size(0), xb.size(1)), {
                "refinement_depth": depth,
                "refinement_updates": depth.sum(),
                "refinement_ponder_cost": torch.tensor(1.0),
            }

    _stamp(monkeypatch, E, ["wind_only", "wind_only"])
    monkeypatch.setattr(
        E, "_build_model_and_loader",
        lambda *a, **k: (
            None, _M(),
            [(torch.zeros(2, 2, 8), torch.zeros(2, 2), torch.tensor([2, 2]),
              torch.tensor([True, True]))],
        ),
    )
    out = E.compute_condition_accounting(
        Path("ckpt"), Path("ds"), Path("np"), torch.device("cpu")
    )
    # Both trials are wind_only and length 2 -> 4 valid wind frames, 0 visual.
    assert out["wind_only"]["n_valid_frames"] == 4
    assert out["visual_present"]["n_valid_frames"] == 0
    assert out["visual_present"]["mean_refinement_depth"] is None
    assert out["visual_present"]["total_refinement_updates"] is None
    assert out["pooled"]["n_valid_frames"] == 4


def test_val_stimulus_conditions_fails_closed_on_unknown(tmp_path, monkeypatch) -> None:
    """The authoritative-stamp reader refuses an out-of-class condition."""
    import evaluate_realdata_preliminary as E

    dp, pp = tmp_path / "ds.pt", tmp_path / "np.pt"
    dp.write_bytes(b"d")
    pp.write_bytes(b"p")
    artifacts = iter([
        {"stimulus_conditions": np.array(["wind_only", "no_stimulus"], dtype=object)},
        {"val_indices": np.array([0, 1])},
    ])
    monkeypatch.setattr(E, "load_artifact_bytes", lambda *a, **k: next(artifacts))
    with pytest.raises(ValueError, match="no_stimulus"):
        E._val_stimulus_conditions(dp, pp)


def test_val_stimulus_conditions_returns_allowed(tmp_path, monkeypatch) -> None:
    """Allowed classes pass through, selected by the nested val indices."""
    import evaluate_realdata_preliminary as E

    dp, pp = tmp_path / "ds.pt", tmp_path / "np.pt"
    dp.write_bytes(b"d")
    pp.write_bytes(b"p")
    artifacts = iter([
        {"stimulus_conditions": np.array(
            ["wind_only", "visual_only", "multisensory"], dtype=object)},
        {"val_indices": np.array([1, 2])},
    ])
    monkeypatch.setattr(E, "load_artifact_bytes", lambda *a, **k: next(artifacts))
    out = E._val_stimulus_conditions(dp, pp)
    assert out.tolist() == ["visual_only", "multisensory"]


# ── Frozen family parameterization: default must stay the k-family ──────────
def test_declared_family_default_preserves_k_family() -> None:
    from nsmor.analysis.model_comparison import (
        CANDIDATE_IDS,
        DECLARED_FAMILY_SIZE,
        declared_family_slots,
    )

    assert CANDIDATE_IDS == ("k_0.5", "k_1.0")
    slots = declared_family_slots()
    assert len(slots) == DECLARED_FAMILY_SIZE == 78
    assert slots[0].startswith("k_0.5|")


def test_declared_family_accepts_realdata_candidates() -> None:
    from nsmor.analysis.model_comparison import (
        DECLARED_FAMILY_SIZE,
        declared_family_slots,
    )

    slots = declared_family_slots(("A1", "A2"))
    assert len(slots) == DECLARED_FAMILY_SIZE == 78
    assert slots[0].startswith("A1|")
    # Wrong candidate cardinality is refused, not silently accepted.
    with pytest.raises(AssertionError):
        declared_family_slots(("A1", "A2", "A3"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
