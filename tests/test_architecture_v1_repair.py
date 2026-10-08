"""Focused regressions for the architecture-v1 REJECT-repair batch.

Covers the seven root groups confirmed by the merged A/B review disposition
(2026-10-08).  Synthetic, deterministic, in-memory; no formal data,
checkpoints, prior artifacts or protected paths.

Roots:
* G1  fixed refinement gathers only valid rows, skips all-empty block calls,
      scatters exact zero padding, counters equal real work, truthful weights.
* G2  versioned control harness has an independent validation split, per-epoch
      validation metric, computed convergence verdict (short smoke only).
* G3  original observation dtype validated (integer/complex/bool refused) at
      Torch core / backend / raw-JAX / fallback / Flax checked-host seams.
* G4  active direct-backend operands validated against the forced-FP32
      representation after safe padding selection.
* G5  enabled raw-JAX learned STP/gain scalar leaves validated.
* G6  refinement options validated even in off mode (no off params, no RNG).
* G7  gain mechanism-tree invariant fails closed before math/export.
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

from nsmor.config_parser import ExperimentConfig
from nsmor.model_nsmor_core import (
    AdaptiveLatentRefinement,
    BioDecisionCore,
    NSMoRCore,
)

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    JAX_AVAILABLE = False

# R4 / A7, B7: NO module-level ``os.environ`` writes.  A collection-time
# ``setdefault`` leaks CUDA-visibility into the WHOLE pytest session, hiding
# the GPU from every later test (e.g. test_epoch_recovery's CPU guard).  These
# focused tests assert refusals/validations, which are device-independent, so
# they no longer need any JAX platform pin.
_REPO_ROOT = Path(__file__).resolve().parents[1]


# ===============================================================
# G1 — fixed refinement valid-row work, zero padding, truthful weights
# ===============================================================

class TestG1FixedValidRowWork:

    def test_fixed_executes_only_valid_rows(self) -> None:
        """A mixed batch executes exactly n_valid rows per step, not B*T."""
        ref = AdaptiveLatentRefinement(4, mode="fixed", max_steps=3)
        ref.eval()
        calls = []
        h = ref.fc1.register_forward_hook(lambda m, a, o: calls.append(a[0].shape[0]))
        try:
            x = torch.zeros(2, 3, 4, requires_grad=True)
            valid = torch.tensor([[True, False, False], [False, False, False]])
            out, depth, updates, _ponder, weights = ref(x, valid)
        finally:
            h.remove()
        assert calls == [1, 1, 1], f"block must see only the 1 valid row: {calls}"
        assert sum(calls) == int(updates) == int(depth.sum()) == 3
        # Padding is exact zero and receives no gradient.
        assert float(out[~valid].abs().max().detach()) == 0.0
        grad = torch.autograd.grad(
            out[~valid].sum(), ref.fc2.bias, allow_unused=True,
        )[0]
        assert grad is None or float(grad.abs().sum()) == 0.0
        # Truthful final-state weight: one-hot at K for valid rows, 0 padding.
        assert float(weights[0, 0, -1]) == 1.0
        assert float(weights[~valid].abs().max()) == 0.0
        assert float(weights[0, 0].sum()) == 1.0

    def test_fixed_all_empty_skips_block_and_returns_zero(self) -> None:
        ref = AdaptiveLatentRefinement(4, mode="fixed", max_steps=3)
        ref.eval()
        calls = []
        h = ref.fc1.register_forward_hook(lambda m, a, o: calls.append(a[0].shape[0]))
        try:
            x = torch.zeros(2, 3, 4, requires_grad=True)
            valid = torch.zeros(2, 3, dtype=torch.bool)
            out, depth, updates, ponder, weights = ref(x, valid)
        finally:
            h.remove()
        assert calls == [], "all-empty must perform NO block call"
        assert int(updates) == 0 and int(depth.sum()) == 0
        assert float(out.abs().max()) == 0.0
        assert (out == 0).all()
        # Ponder stays a differentiable zero (connected to the graph).
        assert ponder.grad_fn is not None
        assert float(ponder.detach()) == 0.0
        ponder.backward()
        assert x.grad is None or float(x.grad.abs().sum()) == 0.0
        assert float(weights.abs().max()) == 0.0

    def test_fixed_padding_does_not_change_valid_rows(self) -> None:
        ref = AdaptiveLatentRefinement(4, mode="fixed", max_steps=3)
        ref.eval()
        u0 = torch.randn(2, 4, 4)
        v_all = torch.ones(2, 4, dtype=torch.bool)
        v_pad = torch.ones(2, 4, dtype=torch.bool)
        v_pad[0, 2:] = False
        out_all, _, _, _, _ = ref(u0, v_all)
        out_pad, _, _, _, _ = ref(u0, v_pad)
        assert torch.allclose(out_all[1], out_pad[1])
        assert torch.allclose(out_all[0, :2], out_pad[0, :2])
        assert float(out_pad[0, 2:].abs().max().detach()) == 0.0


# ===============================================================
# G3 — original observation dtype at every public seam
# ===============================================================

class TestG3ObservationDtype:

    def test_torch_core_refuses_non_real_floating(self) -> None:
        m = NSMoRCore(hidden_dim=4, dropout=0.0).eval()
        lengths = torch.tensor([2])
        for bad in (
            torch.zeros(1, 2, 8, dtype=torch.int64),
            torch.zeros(1, 2, 8, dtype=torch.bool),
            torch.zeros(1, 2, 8, dtype=torch.complex64),
        ):
            with pytest.raises(ValueError, match="real floating"):
                m(bad, lengths)
        # A valid real-floating control is accepted.
        y = m(torch.zeros(1, 2, 8), lengths)
        assert torch.isfinite(y).all()

    def test_backend_refuses_complex_and_integer_observations(self) -> None:
        b = BioDecisionCore(hidden_dim=4, dropout=0.0).eval()
        e = torch.zeros(1, 2, 4)
        for bad in (
            torch.zeros(1, 2, 4, dtype=torch.int64),
            torch.zeros(1, 2, 4, dtype=torch.bool),
            torch.full((1, 2, 4), 0.25, dtype=torch.complex64) + 3j,
        ):
            with pytest.raises(ValueError, match="real floating"):
                b(e, bad, torch.tensor([2]))
        # Integer/complex sensory encoding is refused too.
        for bad_e in (
            torch.zeros(1, 2, 4, dtype=torch.int64),
            torch.zeros(1, 2, 4, dtype=torch.complex64),
        ):
            with pytest.raises(ValueError, match="real floating"):
                b(bad_e, torch.full((1, 2, 4), 0.25), torch.tensor([2]))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_raw_jax_and_fallback_refuse_invalid_dtype(self) -> None:
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = NSMoRCore(hidden_dim=4, dt_ms=4.0, dropout=0.0).eval()
        r = NSMoRCoreJAX.from_torch(m)
        lengths = torch.tensor([2])
        for bad in (
            torch.zeros(1, 2, 8, dtype=torch.int64),
            torch.zeros(1, 2, 8, dtype=torch.complex64) + 3j,
            np.zeros((1, 2, 8), dtype=np.int64),
            np.zeros((1, 2, 8), dtype=np.complex64) + 3j,
        ):
            with pytest.raises(ValueError, match="real floating"):
                r(bad, lengths)
        # A valid real-floating control is accepted.
        y = r(torch.zeros(1, 2, 8), lengths)
        assert torch.isfinite(torch.as_tensor(y)).all()

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_flax_checked_host_refuses_invalid_dtype(self) -> None:
        from nsmor.jax.model import validate_input_and_lengths

        L = np.array([2], dtype=np.int32)
        for bad in (
            np.zeros((1, 2, 8), dtype=np.int64),
            np.zeros((1, 2, 8), dtype=bool),
            np.zeros((1, 2, 8), dtype=np.complex64),
        ):
            with pytest.raises(ValueError, match="real floating"):
                validate_input_and_lengths(bad, L, context="test")
        # Real-floating controls (float32/float64) are accepted.
        validate_input_and_lengths(np.zeros((1, 2, 8), dtype=np.float32), L,
                                   context="test")
        validate_input_and_lengths(np.zeros((1, 2, 8), dtype=np.float64), L,
                                   context="test")

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_flax_uq_convenience_refuses_integer(self) -> None:
        from nsmor.analysis.uq_jax import MCDropoutAnalyzerJAX
        from nsmor.jax.model import NSMoRModel

        model = NSMoRModel(hidden_dim=4, dt_ms=4.0)
        rng = jax.random.PRNGKey(0)
        dummy_x = jnp.zeros((1, 2, 8), dtype=jnp.float32)
        dummy_l = jnp.array([2], dtype=jnp.int32)
        params = model.init(rng, dummy_x, dummy_l)
        analyzer = MCDropoutAnalyzerJAX(model, params, n_samples=3, seed=0)
        with pytest.raises(ValueError, match="real floating"):
            analyzer.predict(
                jnp.zeros((1, 2, 8), dtype=jnp.int32),
                jnp.array([2], dtype=jnp.int32),
            )


# ===============================================================
# G4 — forced-FP32 representability after safe padding selection
# ===============================================================

class TestG4FP32Representability:

    def test_backend_rejects_active_overflow_preserves_controls(self) -> None:
        b = BioDecisionCore(hidden_dim=4, dropout=0.0).eval()
        e = torch.zeros(1, 2, 4)
        overflow = torch.full((1, 2, 4), 0.25, dtype=torch.float64)
        overflow[:, :, 0] = 1e300
        with pytest.raises(ValueError, match="representable|overflow"):
            b(e, overflow, torch.tensor([2]))
        # A representable float64 control remains finite.
        control = torch.full((1, 2, 4), 0.25, dtype=torch.float64)
        y, internals = b(e, control, torch.tensor([2]), return_internals=True)
        assert torch.isfinite(y).all()
        assert torch.isfinite(internals["routing_gates"]).all()

    def test_backend_harmless_overflow_padding_accepted(self) -> None:
        """An overflowing PADDED suffix (invalid frames) is a safe no-op."""
        b = BioDecisionCore(hidden_dim=4, dropout=0.0).eval()
        e = torch.zeros(1, 3, 4)
        prior = torch.full((1, 3, 4), 0.25, dtype=torch.float64)
        prior[:, 2, :] = 1e300  # frame index 2 is padding (length 2)
        y, internals = b(e, prior, torch.tensor([2]), return_internals=True)
        assert torch.isfinite(y).all()
        assert torch.isfinite(internals["routing_gates"]).all()


# ===============================================================
# G5 — enabled raw-JAX learned scalar leaves
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestG5RawScalarLeaves:

    def test_stp_utilization_leaf_validated(self) -> None:
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = NSMoRCore(
            hidden_dim=4, dropout=0.0, lif_tau_fac=20.0, lif_tau_rec=200.0,
        ).eval()
        # Zero-scalar control is accepted.
        NSMoRCoreJAX.from_torch(m)
        with torch.no_grad():
            m.lif_cell.U_stp_raw.fill_(float("nan"))
        with pytest.raises(ValueError, match="nonfinite|U_stp_raw"):
            NSMoRCoreJAX.from_torch(m)

    def test_gain_scalar_leaves_validated(self) -> None:
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        for leaf in ("_gain_scale", "_gain_bias"):
            m = NSMoRCore(hidden_dim=4, dropout=0.0, gru_neuromod_gain=1.0).eval()
            NSMoRCoreJAX.from_torch(m)  # control
            with torch.no_grad():
                getattr(m.backend, leaf).fill_(float("nan"))
            with pytest.raises(ValueError, match="nonfinite|gain"):
                NSMoRCoreJAX.from_torch(m)


# ===============================================================
# G6 — refinement options validated even in off mode
# ===============================================================

class TestG6OffModeRefinementOptions:

    @pytest.mark.parametrize("kwargs", [
        {"refinement_max_steps": True},
        {"refinement_max_steps": 0},
        {"refinement_eps": float("nan")},
        {"refinement_eps": 1.0},
        {"refinement_update_scale": -1.0},
        {"refinement_update_scale": float("inf")},
    ])
    def test_off_mode_rejects_malformed_options(self, kwargs) -> None:
        with pytest.raises(ValueError):
            NSMoRCore(hidden_dim=4, refinement_mode="off", **kwargs)

    def test_off_mode_default_adds_no_params_or_rng(self) -> None:
        torch.manual_seed(0)
        m = NSMoRCore(hidden_dim=4, dropout=0.0).eval()
        assert m.backend.refinement is None
        # No refinement parameter tree leaks into the off model.
        assert not any("refinement" in n for n, _ in m.named_parameters())

    def test_off_and_malformed_config_agree(self) -> None:
        """ModelConfig and the direct off-mode constructor share constraints."""
        cfg = ExperimentConfig()
        cfg.model.refinement_eps = float("nan")
        with pytest.raises(ValueError):
            cfg.model.__post_init__()
        with pytest.raises(ValueError):
            NSMoRCore(hidden_dim=4, refinement_mode="off",
                      refinement_eps=float("nan"))


# ===============================================================
# G7 — gain mechanism-tree invariant
# ===============================================================

class TestG7GainTreeInvariant:

    def test_mutated_gain_flag_fails_closed(self) -> None:
        m = NSMoRCore(hidden_dim=4, dropout=0.0, gru_neuromod_gain=0.0).eval()
        assert not hasattr(m.backend, "_gain_scale")
        # Inconsistent live flag: enable without creating the parameters.
        m.backend.gru_neuromod_gain = 1.0
        with pytest.raises(ValueError, match="gain"):
            m(torch.zeros(1, 2, 8), torch.tensor([2]))

    def test_constructor_configured_gain_still_works(self) -> None:
        m = NSMoRCore(hidden_dim=4, dropout=0.0, gru_neuromod_gain=1.0).eval()
        y, internals = m(torch.zeros(1, 2, 8), torch.tensor([2]),
                         return_internals=True)
        assert torch.isfinite(y).all()
        # The raw recurrent coordinate remains available for analysis.
        assert internals["gru_hidden_raw"].shape == (1, 2, 4)


# ===============================================================
# G2 / H8 — in-repo versioned control harness (no silent skip)
# ===============================================================

# H8 / findings A9, B7: the harness lives IN the repository (scripts/ is
# editable), so this regression never silently skips on a clean checkout.  If
# the file is missing the collection itself fails — a visible blocked gate, not
# a green skip.
_HARNESS = _REPO_ROOT / "scripts" / "final_control_harness.py"
assert _HARNESS.exists(), (
    f"required in-repo control harness missing: {_HARNESS} — this is a BLOCKED "
    "gate, not a skippable test."
)


class TestG2ControlHarnessSmoke:

    def test_harness_computes_convergence_verdict(self, tmp_path) -> None:
        env = os.environ.copy()
        env["NSMOR_CONTROL_JOB"] = str(tmp_path)
        env["NSMOR_CONTROL_OUT"] = str(tmp_path / "finalcontrol_v3")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run(
            [sys.executable, str(_HARNESS), "--epochs", "2", "--seeds", "2",
             "--attempt", "test01"],
            cwd=str(_REPO_ROOT), env=env, capture_output=True, text=True,
            timeout=900,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        result = json.loads(
            (tmp_path / "finalcontrol_v3"
             / "final_control_v3_results_test01.json").read_text()
        )
        # A real, COMPUTED convergence verdict — never an unconditional NOTRUN.
        gate = result["convergence_gate"]
        assert gate["status_per_seed"], "no convergence observations recorded"
        assert all(s in ("CONVERGED", "DIVERGED", "NOT_CONVERGED",
                         "INSUFFICIENT_EPOCHS", "NO_VALIDATION_OBSERVATIONS",
                         "FAILED_NONFINITE", "FAILED_SKIPPED_STEPS")
                   for s in gate["status_per_seed"])
        # R2 / A4, B6: a 2-epoch run is below window+burn-in, so it MUST be
        # INSUFFICIENT_EPOCHS rather than a spurious CONVERGED.
        assert all(s == "INSUFFICIENT_EPOCHS" for s in gate["status_per_seed"])
        # R2 / A3, B5: budget matching is on refinement work per valid token
        # with a half-step tolerance and a per-seed nearest-arm requirement.
        mb = result["matched_budget"]
        assert "tolerance_abs_refine_per_token" in mb
        assert mb["tolerance_fraction_of_one_step"] == 0.5
        assert len(mb["per_seed_nearest_fixed_arm"]) == 2
        # R2 / A2, B3: the paired comparison is on the held-out validation metric.
        assert "validation" in result["evidence_scope"].lower()
        for arm in result["per_arm"].values():
            assert len(arm["val_task_mse_per_seed"]) == 2
            assert arm["completed_epochs"] == 2
            # Per-epoch validation trajectory is recorded for every seed.
            assert all(len(v) == 2 for v in arm["val_losses_per_seed"])
            # R2 / B13: skipped nonfinite steps are surfaced per seed.
            assert arm["n_skipped_steps_per_seed"] == [0, 0]
            # H7: block AND halt rows are asserted per phase, not merely recorded.
            assert arm["row_count_assert_all_ok"]
            assert arm["accounting_ok_all"]
            assert arm["finite_phases_all"]

    @pytest.mark.parametrize("phase", ["train", "val", "eval"])
    def test_harness_aborts_on_accounting_mismatch(self, tmp_path, phase) -> None:
        """H7: a corrupted per-phase block-row count ABORTS (non-zero exit)."""
        env = os.environ.copy()
        env["NSMOR_CONTROL_JOB"] = str(tmp_path)
        env["NSMOR_CONTROL_OUT"] = str(tmp_path / "finalcontrol_v3")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run(
            [sys.executable, str(_HARNESS), "--epochs", "2", "--seeds", "1",
             "--attempt", f"inj_{phase}", "--inject-corrupt", phase],
            cwd=str(_REPO_ROOT), env=env, capture_output=True, text=True,
            timeout=900,
        )
        assert r.returncode != 0, (
            f"harness must abort on a {phase}-phase accounting mismatch"
        )

    @pytest.mark.parametrize("phase", ["train", "val", "eval"])
    def test_harness_aborts_on_halt_row_mismatch(self, tmp_path, phase) -> None:
        """R2 / B10: the halt-row half of the invariant is exercised too.

        The pre-R2 injection seam corrupted only block rows, so ``got_h !=
        exp_h`` was never reachable.  These cases corrupt HALT rows, which only
        adaptive arms execute.
        """
        env = os.environ.copy()
        env["NSMOR_CONTROL_JOB"] = str(tmp_path)
        env["NSMOR_CONTROL_OUT"] = str(tmp_path / "finalcontrol_v3")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run(
            [sys.executable, str(_HARNESS), "--epochs", "2", "--seeds", "1",
             "--attempt", f"inj_halt_{phase}", "--inject-corrupt",
             f"{phase}_halt"],
            cwd=str(_REPO_ROOT), env=env, capture_output=True, text=True,
            timeout=900,
        )
        assert r.returncode != 0, (
            f"harness must abort on a {phase}-phase HALT-row mismatch"
        )

    def test_measured_refine_uses_per_unit_costs(self, tmp_path) -> None:
        """Q1 / A1, B1: measured refine work is costed PER UNIT.

        measured == (block_rows*BLOCK + halt_rows*HALT) / valid_tokens, NOT
        (block_rows+halt_rows)*(BLOCK+HALT).  The old formula made adaptive
        exactly 2x and charged fixed arms a halt MAC they never run.  This test
        fails under the old formula for BOTH an adaptive and a fixed arm.
        """
        env = os.environ.copy()
        env["NSMOR_CONTROL_JOB"] = str(tmp_path)
        env["NSMOR_CONTROL_OUT"] = str(tmp_path / "finalcontrol_v3")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run(
            [sys.executable, str(_HARNESS), "--epochs", "2", "--seeds", "1",
             "--attempt", "q1unit"],
            cwd=str(_REPO_ROOT), env=env, capture_output=True, text=True,
            timeout=900,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        result = json.loads(
            (tmp_path / "finalcontrol_v3"
             / "final_control_v3_results_q1unit.json").read_text()
        )
        # H=16 -> BLOCK = 16*16+16*16 = 512, HALT = 16 (see budget_definition).
        bd = result["budget_definition"]
        BLOCK = bd["shared_block_mac_per_row"]
        HALT = bd["halt_mac_per_row"]
        assert (BLOCK, HALT) == (512, 16)
        # The metric string must describe the per-unit formula.
        assert "block_rows*BLOCK + halt_rows*HALT" in bd["matched_budget_metric"]

        def _check(arm_name, expect_halt_nonzero):
            arm = result["per_arm"][arm_name]
            per_seed = []
            for i, ntok in enumerate(arm["total_valid_tokens_per_seed"]):
                b = (arm["train_block_rows_per_seed"][i]
                     + arm["val_block_rows_per_seed"][i]
                     + arm["eval_block_rows_per_seed"][i])
                h = (arm["train_halt_rows_per_seed"][i]
                     + arm["val_halt_rows_per_seed"][i]
                     + arm["eval_halt_rows_per_seed"][i])
                per_unit = (b * BLOCK + h * HALT) / ntok
                old = (b + h) * (BLOCK + HALT) / ntok
                # The published measured value is the PER-UNIT quantity.
                assert abs(arm["measured_refine_per_token_per_seed"][i]
                           - per_unit) < 1e-9
                # It DIFFERS from the old (block+halt)*(BLOCK+HALT) formula, so
                # this assertion FAILS under the old implementation.
                assert abs(per_unit - old) > 1e-6
                assert (h > 0) == expect_halt_nonzero
                per_seed.append(per_unit)
            assert arm["measured_refine_per_token_per_seed"] == pytest.approx(
                per_seed, rel=0, abs=1e-9
            )

        # Adaptive executes halt rows; a fixed arm executes none.  Under the old
        # formula a fixed arm was charged HALT for rows it never ran.
        _check("adaptive", expect_halt_nonzero=True)
        _check("fixed_K3", expect_halt_nonzero=False)

    def test_default_cli_path_without_overrides_writes_results(
        self, tmp_path
    ) -> None:
        """Q2 / B2: the default CLI path (no NSMOR_CONTROL_JOB/OUT) must write.

        The child re-exec recomputes JOB/OUT under its own TMPDIR; without the
        parent passing them explicitly the final atomic write raised
        FileNotFoundError.  TMPDIR is pinned here so the test is deterministic.
        """
        env = os.environ.copy()
        env.pop("NSMOR_CONTROL_JOB", None)
        env.pop("NSMOR_CONTROL_OUT", None)
        env["TMPDIR"] = str(tmp_path)          # create nothing; let defaults derive
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run(
            [sys.executable, str(_HARNESS), "--epochs", "2", "--seeds", "1",
             "--attempt", "q2default"],
            cwd=str(_REPO_ROOT), env=env, capture_output=True, text=True,
            timeout=900,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        # Defaults: JOB = TMPDIR/nsmor_final_control, OUT = JOB/finalcontrol_v3.
        written = (tmp_path / "nsmor_final_control" / "finalcontrol_v3"
                   / "final_control_v3_results_q2default.json")
        assert written.exists(), (
            "default CLI path produced no results file (the child wrote to a "
            f"scratch TMPDIR); stdout={r.stdout[-500:]!r}"
        )
        result = json.loads(written.read_text())
        assert result["status"] == "FINAL_CONTROL_PREPARED_GATE_EXECUTABLE"


# ===============================================================
# H1 — refinement off-module contract
# ===============================================================

class TestH1RefinementOffContract:

    def test_public_off_constructor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="off"):
            AdaptiveLatentRefinement(8, mode="off")

    def test_enabled_modes_still_construct(self) -> None:
        for mode in ("fixed", "adaptive"):
            m = AdaptiveLatentRefinement(8, mode=mode)
            assert m.mode == mode
            assert sum(p.numel() for p in m.parameters()) > 0

    def test_off_option_is_still_validated_before_refusal(self) -> None:
        # The declared option contract is validated FIRST, so a malformed
        # off-mode request fails on the option, not merely on the mode.
        with pytest.raises(ValueError, match="max_steps"):
            AdaptiveLatentRefinement(8, mode="off", max_steps=True)

    def test_bio_decision_off_builds_no_module(self) -> None:
        m = NSMoRCore(hidden_dim=4, dropout=0.0).eval()
        assert m.backend.refinement is None
        assert not any("refinement" in n for n, _ in m.named_parameters())


# ===============================================================
# H2 — all-empty ponder safe zero (both modes)
# ===============================================================

class TestH2AllEmptyPonderSafeZero:

    @pytest.mark.parametrize("mode", ["fixed", "adaptive"])
    @pytest.mark.parametrize("padval", [float("nan"), 3e38])
    def test_all_empty_ponder_is_finite_zero(self, mode, padval) -> None:
        ref = AdaptiveLatentRefinement(8, mode=mode, max_steps=3)
        ref.eval()
        h = torch.full((1, 2, 8), padval, requires_grad=True)
        valid = torch.zeros(1, 2, dtype=torch.bool)
        out, depth, updates, ponder, _w = ref(h, valid)
        assert torch.isfinite(ponder)
        assert float(ponder.detach()) == 0.0
        assert int(updates) == 0
        assert float(out.abs().max().detach()) == 0.0

    @pytest.mark.parametrize("mode", ["fixed", "adaptive"])
    def test_all_empty_ponder_remains_differentiable(self, mode) -> None:
        ref = AdaptiveLatentRefinement(8, mode=mode, max_steps=3)
        ref.eval()
        h = torch.zeros(1, 2, 8, requires_grad=True)
        valid = torch.zeros(1, 2, dtype=torch.bool)
        _out, _d, _u, ponder, _w = ref(h, valid)
        assert ponder.grad_fn is not None
        ponder.backward()

    @pytest.mark.parametrize("mode", ["fixed", "adaptive"])
    def test_finite_control_still_zero(self, mode) -> None:
        ref = AdaptiveLatentRefinement(8, mode=mode, max_steps=3)
        ref.eval()
        h = torch.zeros(1, 2, 8)
        valid = torch.zeros(1, 2, dtype=torch.bool)
        _o, _d, _u, ponder, _w = ref(h, valid)
        assert float(ponder.detach()) == 0.0


# ===============================================================
# H3 — loss compute boundary
# ===============================================================

class TestH3LossComputeBoundary:

    def _args(self):
        torch.manual_seed(0)
        return (torch.randn(2, 5), torch.randn(2, 5), torch.tensor([5, 3]),
                torch.rand(2, 5, 1))

    def test_bool_lambda_compute_refused_before_fast_path(self) -> None:
        from nsmor.loss import BioJointLoss

        crit = BioJointLoss()
        yp, yt, L, g = self._args()
        with pytest.raises(ValueError, match="lambda_compute"):
            crit(yp, yt, L, g, lambda_reg=0.2, lambda_compute=False)
        with pytest.raises(ValueError, match="lambda_compute"):
            crit(yp, yt, L, g, lambda_reg=0.2, lambda_compute=True)

    def test_non_scalar_ponder_refused(self) -> None:
        from nsmor.loss import BioJointLoss

        crit = BioJointLoss()
        yp, yt, L, g = self._args()
        with pytest.raises(ValueError, match="scalar"):
            crit(yp, yt, L, g, lambda_reg=0.2, lambda_compute=0.2,
                 refinement_ponder_cost=torch.full((2,), 2.0))

    def test_one_element_1d_ponder_refused(self) -> None:
        # R6 / A11: numel()==1 is NOT sufficient; a (1,) ponder would make the
        # joint loss (1,)-shaped, breaking the scalar-loss contract.
        from nsmor.loss import BioJointLoss

        crit = BioJointLoss()
        yp, yt, L, g = self._args()
        with pytest.raises(ValueError, match="0-d scalar"):
            crit(yp, yt, L, g, lambda_reg=0.2, lambda_compute=0.2,
                 refinement_ponder_cost=torch.full((1,), 2.0))

    def test_scalar_ponder_control_keeps_scalar_loss(self) -> None:
        from nsmor.loss import BioJointLoss

        crit = BioJointLoss()
        yp, yt, L, g = self._args()
        loss = crit(yp, yt, L, g, lambda_reg=0.2, lambda_compute=0.2,
                    refinement_ponder_cost=torch.tensor(2.5, requires_grad=True))
        assert loss.shape == ()
        loss.backward()

    def test_default_zero_unchanged(self) -> None:
        from nsmor.loss import BioJointLoss

        crit = BioJointLoss()
        yp, yt, L, g = self._args()
        a = crit(yp, yt, L, g, lambda_reg=0.2)
        b = crit(yp, yt, L, g, lambda_reg=0.2, lambda_compute=0.0,
                 refinement_ponder_cost=None)
        assert torch.equal(a, b)


# ===============================================================
# H4 — shared mechanism-tree invariants (Torch forward + raw converter)
# ===============================================================

class TestH4MechanismTreeInvariants:

    def test_torch_forward_fails_closed_on_toggled_flags(self) -> None:
        for flag in ("stp", "latinhib", "gain"):
            m = NSMoRCore(hidden_dim=4, dropout=0.0).eval()
            if flag == "stp":
                m.backend.lif_cell.stp_enabled = True
            elif flag == "latinhib":
                m.backend.lif_cell.lateral_inhibition = 0.1
            else:
                m.backend.gru_neuromod_gain = 1.0
            with pytest.raises(ValueError, match="mechanism"):
                m(torch.zeros(1, 2, 8), torch.tensor([2]))

    def test_constructor_configured_mechanisms_run_finite(self) -> None:
        m = NSMoRCore(
            hidden_dim=4, dropout=0.0, gru_neuromod_gain=1.0,
            lif_tau_fac=20.0, lif_tau_rec=200.0, lif_lateral_inhibition=0.1,
        ).eval()
        y = m(torch.zeros(1, 2, 8), torch.tensor([2]))
        assert torch.isfinite(y).all()

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_raw_converter_fails_closed_before_dereference(self) -> None:
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = NSMoRCore(hidden_dim=4, dropout=0.0).eval()
        m.backend.gru_neuromod_gain = 1.0  # no _gain_scale/_gain_bias
        with pytest.raises(ValueError, match="mechanism"):
            NSMoRCoreJAX.from_torch(m)


# ===============================================================
# H5 — remaining observation seams (direct GRU + Flax representability)
# ===============================================================

class TestH5RemainingObservationSeams:

    def test_direct_gru_rejects_non_real_floating(self) -> None:
        from nsmor.model_nsmor_core import GRUUnit

        g = GRUUnit(4, dropout=0.0).to(torch.complex64)
        with pytest.raises(ValueError, match="real floating"):
            g(torch.ones(1, 2, 4, dtype=torch.complex64), torch.tensor([2]))
        g2 = GRUUnit(4, dropout=0.0)
        for bad in (torch.zeros(1, 2, 4, dtype=torch.int64),
                    torch.zeros(1, 2, 4, dtype=torch.bool)):
            with pytest.raises(ValueError, match="real floating"):
                g2(bad, torch.tensor([2]))
        out = g2(torch.ones(1, 2, 4), torch.tensor([2]))
        assert torch.isfinite(out).all()

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_flax_rejects_active_fp64_overflow_keeps_padding(self) -> None:
        from nsmor.jax.model import validate_input_and_lengths

        L = np.array([1], dtype=np.int32)  # only frame 0 active
        overflow_active = np.zeros((1, 2, 8), dtype=np.float64)
        overflow_active[:, 0, :] = 1e300
        with pytest.raises(ValueError, match="representable"):
            validate_input_and_lengths(overflow_active, L, context="test")
        overflow_padding = np.zeros((1, 2, 8), dtype=np.float64)
        overflow_padding[:, 1, :] = 1e300  # frame 1 is padding
        validate_input_and_lengths(overflow_padding, L, context="test")
        validate_input_and_lengths(
            np.zeros((1, 2, 8), dtype=np.float32), L, context="test",
        )


# ===============================================================
# H6 — raw-JAX exact required leaf shapes
# ===============================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestH6RawExactLeafShapes:

    def _model(self) -> NSMoRCore:
        torch.manual_seed(0)
        return NSMoRCore(hidden_dim=4, dropout=0.0).eval()

    def test_canonical_source_accepted(self) -> None:
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        NSMoRCoreJAX.from_torch(self._model())  # valid control

    @pytest.mark.parametrize("which", ["se_ln_w", "lif_b_in", "gru_b_ih",
                                       "router_b", "dh_ln_w"])
    def test_malformed_leaf_shape_rejected(self, which) -> None:
        from nsmor.model_nsmor_core_jax import NSMoRCoreJAX

        m = self._model()
        bad = torch.nn.Parameter(torch.ones(1))
        if which == "se_ln_w":
            m.frontend.sensory_encoder.net[1].weight = bad
        elif which == "lif_b_in":
            m.backend.lif_cell.W_in.bias = bad
        elif which == "gru_b_ih":
            m.backend.gru_unit.gru.bias_ih_l0 = bad
        elif which == "router_b":
            m.backend.router.gate.bias = bad
        elif which == "dh_ln_w":
            m.backend.direction_head.net[0].weight = bad
        with pytest.raises(ValueError, match="shape"):
            NSMoRCoreJAX.from_torch(m)


# ===============================================================
# R5 — LIF per-step validation is skipped in the internal T-loop only
# ===============================================================

class TestR5LifPerStepValidation:

    def test_internal_loop_skips_public_validation(self) -> None:
        """The T-loop must pass ``_checked=True`` (no per-step host syncs)."""
        import inspect

        from nsmor.model_nsmor_core import BioDecisionCore

        src = inspect.getsource(BioDecisionCore._run_lif_path)
        assert "_checked=True" in src, (
            "internal T-loop must call lif_cell(..., _checked=True)"
        )

    def test_direct_lif_cell_call_still_validated(self) -> None:
        """A PUBLIC direct LIFCell call keeps the full trust-boundary check."""
        from nsmor.model_nsmor_core import LIFCell

        cell = LIFCell(4, dt_ms=4.0)
        B, H = 2, 4
        state = cell.init_state(B, torch.device("cpu"))
        # A nonfinite supplied carry must be refused on the public path.
        bad = list(state)
        bad[0] = torch.full((B, H), float("nan"))
        with pytest.raises(ValueError):
            cell(torch.zeros(B, H), tuple(bad))
        # ``_checked=True`` is the documented internal-only escape hatch and
        # skips that validation (used only by the validated T-loop).
        cell(torch.zeros(B, H), tuple(bad), _checked=True)

    def test_checked_flag_does_not_change_numerics(self) -> None:
        """Skipping re-validation must not alter the arithmetic bitwise."""
        from nsmor.model_nsmor_core import LIFCell

        cell = LIFCell(4, dt_ms=4.0).eval()
        B, H = 2, 4
        state = cell.init_state(B, torch.device("cpu"))
        x = torch.randn(B, H)
        with torch.no_grad():
            s0, n0 = cell(x, state)
            s1, n1 = cell(x, state, _checked=True)
        assert torch.equal(s0, s1)
        for a, b in zip(n0, n1):
            assert torch.equal(a, b)

    def test_model_forward_numerics_unchanged_off(self) -> None:
        """A default-off model still runs finite and deterministic."""
        torch.manual_seed(3)
        m = NSMoRCore(hidden_dim=4, dropout=0.0).eval()
        x = torch.randn(2, 5, 8)
        L = torch.tensor([5, 3])
        with torch.no_grad():
            y0 = m(x, L)
            y1 = m(x, L)
        assert torch.equal(y0, y1)
        assert torch.isfinite(y0).all()


# ===============================================================
# R3 — resume fails closed on an architecture-config switch
# ===============================================================

class TestR3ResumeArchitecturePreflight:

    @staticmethod
    def _loader():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "nsmor_train_entry", _REPO_ROOT / "scripts" / "train.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_matching_config_is_accepted(self) -> None:
        train = self._loader()
        cfg = ExperimentConfig()
        cfg.model.refinement_mode = "fixed"
        cfg.model.refinement_max_steps = 3
        ckpt = {"config": cfg.to_dict()}
        # No raise: identical config.
        train._require_architecture_config_match(ckpt, cfg, Path("ckpt.pth"))

    def test_mode_switch_is_refused(self) -> None:
        train = self._loader()
        ckpt_cfg = ExperimentConfig()
        ckpt_cfg.model.refinement_mode = "fixed"
        ckpt_cfg.loss.lambda_compute = 0.0
        active = ExperimentConfig()
        active.model.refinement_mode = "adaptive"
        active.loss.lambda_compute = 0.05
        ckpt = {"config": ckpt_cfg.to_dict()}
        with pytest.raises(ValueError, match="refinement_mode"):
            train._require_architecture_config_match(
                ckpt, active, Path("ckpt.pth"))

    def test_activation_and_lambda_switch_refused(self) -> None:
        train = self._loader()
        ckpt_cfg = ExperimentConfig()
        ckpt_cfg.model.refinement_mode = "fixed"
        ckpt_cfg.model.activation = "relu"
        ckpt_cfg.loss.lambda_compute = 0.0
        active = ExperimentConfig()
        active.model.refinement_mode = "fixed"
        active.model.activation = "swiglu"
        active.loss.lambda_compute = 0.0
        with pytest.raises(ValueError, match="activation"):
            train._require_architecture_config_match(
                {"config": ckpt_cfg.to_dict()}, active, Path("ckpt.pth"))
        active.model.activation = "relu"
        active.loss.lambda_compute = 0.1
        active.model.refinement_mode = "fixed"
        with pytest.raises(ValueError, match="lambda_compute"):
            train._require_architecture_config_match(
                {"config": ckpt_cfg.to_dict()}, active, Path("ckpt.pth"))

    def test_legacy_checkpoint_without_config_is_skipped(self) -> None:
        train = self._loader()
        active = ExperimentConfig()
        active.model.refinement_mode = "adaptive"
        active.loss.lambda_compute = 0.05
        # No stored config -> limited fallback, no raise here.
        train._require_architecture_config_match(
            {"epoch": 3}, active, Path("legacy.pth"))


# ===============================================================
# R7 — module-level test-env leakage guard (this file)
# ===============================================================

class TestR7NoModuleEnvLeak:

    def test_this_file_has_no_module_level_environ_writes(self) -> None:
        """R4 / A7, B7: collection-time env writes leak session-wide."""
        import ast

        tree = ast.parse(Path(__file__).read_text())
        offenders = []
        for node in tree.body:  # MODULE-LEVEL statements only
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in ("setdefault", "setenv", "putenv")
                    and isinstance(sub.func.value, ast.Attribute)
                    and sub.func.value.attr == "environ"
                ):
                    offenders.append(sub.func.attr)
        assert not offenders, (
            f"module-level os.environ writes leak into the pytest session: "
            f"{offenders}"
        )

