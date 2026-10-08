"""Architecture-v1 FINAL matched-budget control harness (v3, executable gate).

Supersedes ``final_control_harness.v2.py`` (retained, immutable).  v3 repairs
the missing executable convergence mechanism reported by A2/B1:

  * an INDEPENDENT synthetic validation split (separate generator seed) that is
    never trained on;
  * a per-epoch COMMON task validation metric (masked MSE), recorded each epoch;
  * best epoch, last-five-epoch mean and an actually COMPUTED predeclared
    convergence verdict (no unconditional NOTRUN);
  * a fresh per-attempt output root validated before ANY write (exclusive
    ``x``/``xb`` writes; a used attempt is refused);
  * truthful, separated accounting: train/eval factual work via forward hooks
    asserted against ``depth.sum()``, total/requires-grad/executed capacity,
    base/refinement/halt forward-MAC estimates, separate wall clock, per-seed
    depth histograms/gather shapes, finite fail-closed output, and paired
    adjusted (Holm) statistics.

The final <=20-epoch / multiseed control gate is NOT executed here (it remains
gated on dual source ACCEPT).  A short ``--epochs 2`` smoke may run.

Synthetic in-memory data only.  No formal data/checkpoint/prior access.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
# Output roots are overridable (H8: the harness lives IN the repo, so it must
# not depend on a session-specific out-of-repo job path).  Defaults go to a
# system-temp scratch dir; a focused smoke/test overrides NSMOR_CONTROL_OUT.
import tempfile  # noqa: E402

JOB = Path(os.environ.get(
    "NSMOR_CONTROL_JOB", str(Path(tempfile.gettempdir()) / "nsmor_final_control"),
))
OUT = Path(os.environ.get("NSMOR_CONTROL_OUT", str(JOB / "finalcontrol_v3")))

# ── A-priori constants (declared BEFORE any run; never tuned to force a pass) ──
# R2 / A3, B5: budget matching is on REFINEMENT work PER VALID TOKEN with a
# tolerance SMALLER than half of one refinement step (each step costs
# BLOCK+HALT MAC per row).  Matching on TOTAL MAC let the shared base pathway
# dominate, so adjacent fixed arms were indistinguishable.
MATCH_TOL_FRACTION = 0.5   # tolerance = 0.5 x (one refinement step per row)
CONV_SLACK = 0.05          # last-5 validation mean may not exceed best by > 5%
CONV_WINDOW = 5            # last-N validation window
CONV_BURN_IN = 2           # epochs below window+burn-in => INSUFFICIENT_EPOCHS
# Refinement work per valid token is a model constant independent of the data
# (one row is processed per valid token), so the matched-budget verdict is
# computed on the DECLARED per-token constant (deterministic) and cross-checked
# against the ACTUAL measured block/halt rows per seed.
CONTRASTS = [("adaptive", "fixed_K4"), ("adaptive", "fixed_K3"),
             ("adaptive", "fixed_K2"), ("fixed_K4", "off")]


def _parse_attempt(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]{1,32}", value):
        raise argparse.ArgumentTypeError(
            "attempt must be 1-32 chars of [A-Za-z0-9_]"
        )
    return value


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"not an int: {value!r}")
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=_positive_int, required=True)
    p.add_argument("--seeds", type=_positive_int, required=True)
    p.add_argument("--attempt", type=_parse_attempt, required=True)
    # Bounded fault-injection seam for the accounting invariant (H7): a test
    # deliberately corrupts one phase's block-row count to prove the harness
    # ABORTS.  Default "none" leaves production accounting exact.
    p.add_argument("--inject-corrupt", default="none",
                   choices=("none", "train", "val", "eval",
                            "train_halt", "val_halt", "eval_halt"))
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Validate BEFORE any TMPDIR/output creation.
    assert isinstance(args.epochs, int) and not isinstance(args.epochs, bool)
    assert args.epochs >= 1 and args.seeds >= 1 and args.attempt
    name = f"final_control_v3_{args.attempt}"
    tmp = JOB / f"finalcontrol_v3_tmp_{args.attempt}"
    results_path = OUT / f"final_control_v3_results_{args.attempt}.json"
    # Fresh output root validated BEFORE any write: a used attempt is refused.
    if tmp.exists() or results_path.exists():
        raise SystemExit(f"attempt {args.attempt!r} already used; pick a new one")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(exist_ok=False)
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    # Q2 / B2: the child re-imports this module and recomputes JOB/OUT from its
    # OWN environment.  The child's TMPDIR is redirected to the scratch dir, so
    # on the default CLI path (no NSMOR_CONTROL_JOB/OUT) the child would derive
    # OUT under the scratch TMPDIR — where nothing was mkdir'd — and the final
    # atomic write would raise FileNotFoundError.  Pass the parent's resolved
    # JOB/OUT explicitly so the child writes to the directory the parent created.
    env["NSMOR_CONTROL_JOB"] = str(JOB)
    env["NSMOR_CONTROL_OUT"] = str(OUT)
    with (OUT / f"{name}.stdout.log").open("xb") as out, \
            (OUT / f"{name}.stderr.log").open("xb") as err:
        r = subprocess.run(
            [sys.executable, __file__, "--child", args.attempt,
             str(args.epochs), str(args.seeds), args.inject_corrupt],
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=out, stderr=err, env=env,
        )
    with (OUT / f"{name}.exit.json").open("x") as f:
        json.dump({"returncode": r.returncode, "epochs": args.epochs,
                   "seeds": args.seeds, "attempt": args.attempt}, f)
    print(json.dumps({"driver": name, "actual_child_returncode": r.returncode}))
    return r.returncode


def child(attempt: str, epochs: int, seeds: int, inject_corrupt: str = "none") -> int:
    # ``scripts`` is not an installed package, so the repo root is placed on
    # sys.path for the ``nsmor`` imports.  ``train.py`` is loaded by FILE PATH
    # (not ``from scripts.train import``) so this entry point stays runnable via
    # ``python scripts/final_control_harness.py`` (repo rule enforced by
    # tests/test_script_entry_points.py).
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import importlib.util
    import time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from scipy import stats as sstats
    from nsmor.config_parser import ExperimentConfig
    from nsmor.analysis.uq import holm_bonferroni

    _train_spec = importlib.util.spec_from_file_location(
        "nsmor_train_entry", Path(__file__).resolve().parent / "train.py",
    )
    _train = importlib.util.module_from_spec(_train_spec)
    _train_spec.loader.exec_module(_train)
    build_model = _train.build_model
    build_loss = _train.build_loss
    train_one_epoch = _train.train_one_epoch

    torch.set_num_threads(1)
    H, K_MAX, D, M = 16, 4, 4, 4

    ARMS = [
        {"name": "off", "mode": "off", "K": K_MAX, "lc": 0.0},
        {"name": "fixed_K4", "mode": "fixed", "K": 4, "lc": 0.0},
        {"name": "adaptive", "mode": "adaptive", "K": K_MAX, "lc": 0.05},
        {"name": "fixed_K2", "mode": "fixed", "K": 2, "lc": 0.0},
        {"name": "fixed_K3", "mode": "fixed", "K": 3, "lc": 0.0},
    ]

    def make_cfg(mode, K, lc):
        cfg = ExperimentConfig()
        cfg.model.hidden_dim = H
        cfg.model.dropout = 0.0
        cfg.model.refinement_mode = mode
        cfg.model.refinement_max_steps = K
        cfg.loss.lambda_compute = lc
        cfg.training.num_epochs = epochs
        cfg.validate()
        return cfg

    def make_loader(seed, n=12, T=8):
        # R2 / B2: a LEARNABLE, input-dependent synthetic task.  A fixed random
        # linear projection of the input (plus a fixed bias) is the target, so
        # a more expressive arm can actually reduce validation error and the
        # arms are distinguishable in principle.  Inputs are still iid across
        # samples; only the target is coupled to the input.
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, T, 8, generator=g)
        W = torch.tensor([[0.7, -0.4, 0.2, 0.5, -0.6, 0.1, 0.3, -0.2]],
                         dtype=torch.float32)
        bias = 0.1
        y = (x @ W.t()).squeeze(-1) + bias          # (n, T) input-dependent
        L = torch.full((n,), T, dtype=torch.long)
        return DataLoader(TensorDataset(x, y, L), batch_size=4)

    # ── Declared LINEAR-MAC estimate (per-token / per-row). ──
    def base_mac_per_token(H, D, M):
        enc = D * H
        gru = 3 * (H * H + H * H)   # 3 gates x (W_ih + W_hh), input_size=H
        lif = H * H
        router = (H + M) * 2
        head = H
        return {"encoder": enc, "gru": gru, "lif": lif, "router": router,
                "head": head, "total": enc + gru + lif + router + head}

    def block_mac_per_row(H):
        return H * H + H * H  # fc1 + fc2 only (LayerNorm is elementwise)

    def halt_mac_per_row(H):
        return H  # Linear(H,1)

    BASE = base_mac_per_token(H, D, M)
    BLOCK = block_mac_per_row(H)
    HALT = halt_mac_per_row(H)

    @torch.no_grad()
    def eval_task_mse(model, loader):
        """Masked task MSE over the VALID frames of *loader* (no grad)."""
        model.eval()
        sqerr, count = 0.0, 0
        for xb, yb, L in loader:
            yp = model(xb, L)
            mask = torch.arange(xb.shape[1]).unsqueeze(0) < L.unsqueeze(1)
            sqerr += float(((yp - yb) ** 2 * mask).sum())
            count += int(mask.sum())
        return sqerr / max(count, 1)

    def run_arm(arm, seed):
        torch.manual_seed(seed)
        cfg = make_cfg(arm["mode"], arm["K"], arm["lc"])
        model = build_model(cfg)
        crit = build_loss(cfg)
        train_loader = make_loader(seed)             # training split
        val_loader = make_loader(10_000 + seed)      # INDEPENDENT validation
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        total = sum(p.numel() for p in model.parameters())
        requires_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)

        ref = model.backend.refinement
        # Three separated phases: train (training loop), val (per-epoch
        # held-out validation), eval (final mechanism eval on the train loader).
        counters = {f"{p}_{k}": 0
                    for p in ("train", "val", "eval")
                    for k in ("block_calls", "block_rows", "halt_calls", "halt_rows")}
        gather_shapes = {"train": [], "val": [], "eval": []}
        mode_flag = {"phase": "train"}

        def fc1_hook(_mod, inp, _out):
            n = int(inp[0].shape[0])
            split = mode_flag["phase"]
            # Bounded fault-injection seam (H7): a per-phase ``--inject-corrupt
            # <phase>`` deliberately mis-counts block rows in that phase so the
            # accounting invariant can be proven to ABORT.  Default "none" keeps
            # production accounting exact.
            if inject_corrupt == split:
                n += 1
            counters[f"{split}_block_calls"] += 1
            counters[f"{split}_block_rows"] += n
            gather_shapes[split].append(n)

        def halt_hook(_mod, inp, _out):
            n = int(inp[0].shape[0])
            split = mode_flag["phase"]
            # R2 / B10: halt rows are also a fault-injection seam, so the
            # halt-row half of check_phase is genuinely exercised.
            if inject_corrupt == f"{split}_halt":
                n += 1
            counters[f"{split}_halt_calls"] += 1
            counters[f"{split}_halt_rows"] += n

        # H7 / findings A4, B4: the module's OWN detached accounting
        # (``refinement_updates`` == depth.sum()) is captured per forward and
        # compared against the ACTUAL fc1/halt hook rows in EVERY phase.  In
        # ``fixed`` mode the halt head is never executed (expected halt rows 0);
        # in ``adaptive`` mode block and halt each run once per active row, so
        # both equal the declared updates.  A mismatch ABORTS the arm before any
        # budget is derived or any result is written.
        expected = {p: {"block": 0, "halt": 0} for p in ("train", "val", "eval")}

        def ref_hook(_mod, _inp, out):
            u = int(out[2])  # (out, depth, updates, ponder, weights)
            phase = mode_flag["phase"]
            expected[phase]["block"] += u
            expected[phase]["halt"] += u if arm["mode"] == "adaptive" else 0

        def check_phase(phase: str) -> None:
            if ref is None:
                return
            got_b, exp_b = counters[f"{phase}_block_rows"], expected[phase]["block"]
            got_h, exp_h = counters[f"{phase}_halt_rows"], expected[phase]["halt"]
            if got_b != exp_b or got_h != exp_h:
                raise RuntimeError(
                    f"work-accounting mismatch in phase {phase!r}: block rows "
                    f"{got_b} != declared {exp_b} or halt rows {got_h} != "
                    f"declared {exp_h}; refusing to publish a control result "
                    f"with unverified work accounting."
                )

        handles = []
        if ref is not None:
            handles.append(ref.fc1.register_forward_hook(fc1_hook))
            handles.append(ref.halt.register_forward_hook(halt_hook))
            handles.append(ref.register_forward_hook(ref_hook))

        train_wall = 0.0
        train_losses = []
        val_losses = []
        completed_epochs = 0
        finite_ok = True
        # R2 / B13: a step that train_one_epoch SKIPPED as nonfinite never
        # updated the optimizer.  An arm that skipped work must never be
        # published as finite_stable / CONVERGED, so the skips are counted and
        # fail the arm closed.
        skip_counts = {"n_skipped_nonfinite_loss": 0,
                       "n_skipped_nonfinite_grad": 0}
        # H7 / B6: per-phase nonfinite tracking; a nonfinite value in ANY phase
        # marks the arm failed so it can never be labelled CONVERGED/stable.
        finite_phases = {"train": True, "val": True, "eval": True}
        try:
            mode_flag["phase"] = "train"
            for ep in range(epochs):
                t0 = time.perf_counter()
                tl, _health = train_one_epoch(
                    model=model, loader=train_loader, criterion=crit, optimizer=opt,
                    device=torch.device("cpu"), lambda_reg=cfg.loss.lambda_reg,
                    lambda_energy=0.0, lambda_sparse=0.0, lambda_jerk=0.0,
                    annealing_factor=1.0, grad_clip_norm=1.0, log_interval=999,
                    epoch=ep, phase=0, lambda_compute=arm["lc"],
                )
                train_wall += time.perf_counter() - t0
                completed_epochs += 1
                train_losses.append(float(tl))
                if not np.isfinite(tl):
                    finite_ok = False
                    finite_phases["train"] = False
                _sk = getattr(train_one_epoch, "last_skip_counts", None) or {}
                for _k in skip_counts:
                    skip_counts[_k] += int(_sk.get(_k, 0))
                # Per-epoch COMMON task validation metric on the held-out split.
                mode_flag["phase"] = "val"
                vl = eval_task_mse(model, val_loader)
                check_phase("val")
                mode_flag["phase"] = "train"
                val_losses.append(float(vl))
                if not np.isfinite(vl):
                    finite_ok = False
                    finite_phases["val"] = False
            check_phase("train")
        finally:
            for h in handles:
                h.remove()

        # Final no-grad pass on the INDEPENDENT VALIDATION split (R2 / A2, B3):
        # the paired arm comparison must be held-out, not in-sample training fit.
        # Reset the eval counters so this pass is accounted on its own, then
        # compare the executed block rows to the module's detached depth.sum().
        for _k in ("block_calls", "block_rows", "halt_calls", "halt_rows"):
            counters[f"eval_{_k}"] = 0
        gather_shapes["eval"] = []
        mode_flag["phase"] = "eval"
        handles = []
        if ref is not None:
            handles.append(ref.fc1.register_forward_hook(fc1_hook))
            handles.append(ref.halt.register_forward_hook(halt_hook))
            handles.append(ref.register_forward_hook(ref_hook))
        model.eval()
        task_sqerr, task_count = 0.0, 0
        eval_row_updates = 0
        depth_hist = {}
        eval_wall = 0.0
        try:
            t1 = time.perf_counter()
            with torch.no_grad():
                for xb, yb, L in val_loader:
                    yp, ints = model(xb, L, return_internals=True)
                    if not bool(torch.isfinite(yp).all()):
                        raise RuntimeError(
                            "nonfinite final-eval prediction; refusing to "
                            "publish a control result."
                        )
                    mask = (torch.arange(xb.shape[1]).unsqueeze(0) < L.unsqueeze(1))
                    task_sqerr += float(((yp - yb) ** 2 * mask).sum())
                    task_count += int(mask.sum())
                    if "refinement_updates" in ints:
                        d = ints["refinement_depth"]
                        eval_row_updates += int(ints["refinement_updates"])
                        for k in range(0, arm["K"] + 1):
                            depth_hist[k] = depth_hist.get(k, 0) + int((d == k).sum())
            eval_wall = time.perf_counter() - t1
        finally:
            for h in handles:
                h.remove()
        check_phase("eval")
        val_task_mse = task_sqerr / max(task_count, 1)
        if not np.isfinite(val_task_mse):
            finite_ok = False
            finite_phases["eval"] = False

        halt_grad = 0.0
        if ref is not None:
            for p in ref.halt.parameters():
                if p.grad is not None:
                    halt_grad += float(p.grad.abs().sum())

        # ── MAC estimate: base pathway (train + val + eval tokens) + shared
        # block (real block rows, all three phases) + halt (real halt rows).
        # Per-epoch validation is REAL forward work and is counted.  The
        # eval-phase token count is ``task_count`` (the final pass is now over
        # the validation split). ──
        def _valid_tokens(loader):
            n = 0
            for _xb, _yb, _L in loader:
                n += int(
                    (torch.arange(_xb.shape[1]).unsqueeze(0)
                     < _L.unsqueeze(1)).sum()
                )
            return n

        n_train_tokens = _valid_tokens(train_loader)
        n_val_tokens = _valid_tokens(val_loader)
        train_tokens = n_train_tokens * completed_epochs
        val_tokens = n_val_tokens * completed_epochs
        base_mac_train = train_tokens * BASE["total"]
        base_mac_val = val_tokens * BASE["total"]
        base_mac_eval = task_count * BASE["total"]
        block_mac_train = counters["train_block_rows"] * BLOCK
        block_mac_val = counters["val_block_rows"] * BLOCK
        block_mac_eval = counters["eval_block_rows"] * BLOCK
        halt_mac_train = counters["train_halt_rows"] * HALT
        halt_mac_val = counters["val_halt_rows"] * HALT
        halt_mac_eval = counters["eval_halt_rows"] * HALT
        total_mac = (base_mac_train + base_mac_val + base_mac_eval
                     + block_mac_train + block_mac_val + block_mac_eval
                     + halt_mac_train + halt_mac_val + halt_mac_eval)
        # Q1 / A1, B1: the budget that distinguishes the arms is the REFINEMENT
        # work per valid token, costed PER UNIT: block rows x BLOCK + halt rows
        # x HALT (NOT (block+halt) rows x (BLOCK+HALT)).  The old formula made
        # adaptive exactly 2x (it counts block AND halt rows) and charged fixed
        # arms a halt MAC they never execute (their halt rows are 0).  declared
        # uses the SAME per-unit costs on the expected rows (halt cost only for
        # the adaptive arm, whose expected halt rows are non-zero); measured uses
        # the real hook rows.  check_phase already forces the two row sets equal
        # per phase, so measured == declared identically — the equality is
        # asserted explicitly below and fails the arm closed.
        total_valid_tokens = (n_train_tokens * completed_epochs  # train
                              + n_val_tokens * completed_epochs   # per-epoch val
                              + task_count)                       # final eval
        _denom = max(total_valid_tokens, 1)
        declared_refine_mac = (
            (expected["train"]["block"] + expected["val"]["block"]
             + expected["eval"]["block"]) * BLOCK
            + (expected["train"]["halt"] + expected["val"]["halt"]
               + expected["eval"]["halt"]) * HALT
        )
        measured_refine_mac = (
            (counters["train_block_rows"] + counters["val_block_rows"]
             + counters["eval_block_rows"]) * BLOCK
            + (counters["train_halt_rows"] + counters["val_halt_rows"]
               + counters["eval_halt_rows"]) * HALT
        )
        declared_refine_per_token = float(declared_refine_mac) / _denom
        measured_refine_per_token = float(measured_refine_mac) / _denom
        # Fail closed if measured != declared (per arm, per phase) before publish.
        if ref is not None:
            for _ph in ("train", "val", "eval"):
                _exp = (expected[_ph]["block"] * BLOCK
                        + expected[_ph]["halt"] * HALT)
                _got = (counters[f"{_ph}_block_rows"] * BLOCK
                        + counters[f"{_ph}_halt_rows"] * HALT)
                if _got != _exp:
                    raise RuntimeError(
                        f"measured refine MAC != declared in phase {_ph!r} for "
                        f"arm {arm['name']!r}: measured {_got} != declared "
                        f"{_exp}; refusing to publish."
                    )
        if measured_refine_mac != declared_refine_mac:
            raise RuntimeError(
                f"measured refine MAC != declared for arm {arm['name']!r}: "
                f"measured {measured_refine_mac} != declared "
                f"{declared_refine_mac}; refusing to publish."
            )

        # ── Predeclared convergence criterion (COMPUTED from val trajectory) ──
        # H7 / B6: a nonfinite value in ANY phase forbids a CONVERGED verdict.
        # R2 / A4, B6, B12: a still-improving trajectory is NOT "DIVERGED"; a
        # run shorter than window+burn-in is INSUFFICIENT_EPOCHS; DIVERGED is
        # reserved for a worsening trajectory whose latest epoch is worse than
        # the best epoch AFTER the best (i.e. genuinely post-best divergence).
        n_skipped = sum(skip_counts.values())
        best_epoch = int(np.argmin(val_losses)) if val_losses else None
        best_val = float(min(val_losses)) if val_losses else None
        last5 = val_losses[-CONV_WINDOW:] if val_losses else []
        last5_mean = float(np.mean(last5)) if last5 else None
        if not finite_ok:
            conv_status = "FAILED_NONFINITE"
        elif n_skipped > 0:
            conv_status = "FAILED_SKIPPED_STEPS"
        elif best_val is None:
            conv_status = "NO_VALIDATION_OBSERVATIONS"
        elif not np.isfinite(last5_mean):
            conv_status = "DIVERGED"
        elif completed_epochs < CONV_WINDOW + CONV_BURN_IN:
            conv_status = "INSUFFICIENT_EPOCHS"
        elif last5_mean <= best_val * (1.0 + CONV_SLACK):
            conv_status = "CONVERGED"
        elif best_epoch is not None and last5[-1] > best_val * (1.0 + CONV_SLACK):
            conv_status = "DIVERGED"
        else:
            # Latest epoch is not worse than best, but the window mean is: a
            # still-improving (or mildly noisy) trajectory, not divergence.
            conv_status = "NOT_CONVERGED"

        # Declared-vs-actual equality (already enforced by check_phase); the
        # booleans are reported as verified facts, not as the enforcement.
        accounting_ok = {
            "train": (counters["train_block_rows"] == expected["train"]["block"]
                      and counters["train_halt_rows"] == expected["train"]["halt"]),
            "val": (counters["val_block_rows"] == expected["val"]["block"]
                    and counters["val_halt_rows"] == expected["val"]["halt"]),
            "eval": (counters["eval_block_rows"] == expected["eval"]["block"]
                     and counters["eval_halt_rows"] == expected["eval"]["halt"]),
        } if ref is not None else {"train": True, "val": True, "eval": True}

        return {
            "arm": arm["name"], "mode": arm["mode"], "K": arm["K"],
            "lambda_compute": arm["lc"], "seed": seed,
            "total_params": total, "requires_grad_count": requires_grad,
            "executed_block_rows": counters["train_block_rows"],
            "completed_epochs": completed_epochs,
            "finite_stable": finite_ok and n_skipped == 0,
            "finite_phases": finite_phases,
            "skip_counts": skip_counts,
            "n_skipped_steps": n_skipped,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch, "best_val": best_val,
            "last5_val_mean": last5_mean, "convergence_status": conv_status,
            "real_counters": counters,
            "declared_work": expected,
            "gather_shapes": gather_shapes,
            "eval_row_updates": eval_row_updates,
            "row_count_assert_ok": all(accounting_ok.values()),
            "accounting_ok": accounting_ok,
            "n_tokens": task_count,
            "base_mac_train": base_mac_train, "base_mac_val": base_mac_val,
            "base_mac_eval": base_mac_eval,
            "block_mac_train": block_mac_train, "block_mac_val": block_mac_val,
            "block_mac_eval": block_mac_eval,
            "halt_mac_train": halt_mac_train, "halt_mac_val": halt_mac_val,
            "halt_mac_eval": halt_mac_eval,
            "total_est_mac": total_mac,
            "total_valid_tokens": total_valid_tokens,
            "declared_refine_per_token": declared_refine_per_token,
            "measured_refine_per_token": measured_refine_per_token,
            "train_wall_s": train_wall, "eval_wall_s": eval_wall,
            "val_task_mse": val_task_mse, "halt_grad_sum": halt_grad,
            "depth_hist": depth_hist,
        }

    all_rows = []
    for arm in ARMS:
        for seed in range(seeds):
            all_rows.append(run_arm(arm, seed))

    def rows_of(name):
        return [r for r in all_rows if r["arm"] == name]

    def agg(name):
        rs = rows_of(name)
        return {
            "n": len(rs),
            "completed_epochs": rs[0]["completed_epochs"],
            "n_skipped_steps_per_seed": [r["n_skipped_steps"] for r in rs],
            "finite_stable_all": all(r["finite_stable"] for r in rs),
            "row_count_assert_all_ok": all(r["row_count_assert_ok"] for r in rs),
            "accounting_ok_all": all(
                all(r["accounting_ok"].values()) for r in rs
            ),
            "finite_phases_all": all(
                all(r["finite_phases"].values()) for r in rs
            ),
            "total_est_mac_per_seed": [r["total_est_mac"] for r in rs],
            "total_est_mac_mean": float(np.mean([r["total_est_mac"] for r in rs])),
            "eval_row_updates_per_seed": [r["eval_row_updates"] for r in rs],
            "train_block_rows_per_seed": [r["real_counters"]["train_block_rows"] for r in rs],
            "val_block_rows_per_seed": [r["real_counters"]["val_block_rows"] for r in rs],
            "eval_block_rows_per_seed": [r["real_counters"]["eval_block_rows"] for r in rs],
            "train_halt_rows_per_seed": [r["real_counters"]["train_halt_rows"] for r in rs],
            "val_halt_rows_per_seed": [r["real_counters"]["val_halt_rows"] for r in rs],
            "eval_halt_rows_per_seed": [r["real_counters"]["eval_halt_rows"] for r in rs],
            "val_task_mse_per_seed": [r["val_task_mse"] for r in rs],
            "val_task_mse_mean": float(np.mean([r["val_task_mse"] for r in rs])),
            "total_valid_tokens_per_seed": [r["total_valid_tokens"] for r in rs],
            "refine_per_token_per_seed": [r["declared_refine_per_token"] for r in rs],
            "measured_refine_per_token_per_seed": [
                r["measured_refine_per_token"] for r in rs],
            "val_losses_per_seed": [r["val_losses"] for r in rs],
            "best_epoch_per_seed": [r["best_epoch"] for r in rs],
            "last5_val_mean_per_seed": [r["last5_val_mean"] for r in rs],
            "convergence_status_per_seed": [r["convergence_status"] for r in rs],
            "halt_grad_sum_per_seed": [r["halt_grad_sum"] for r in rs],
            "train_wall_mean": float(np.mean([r["train_wall_s"] for r in rs])),
            "eval_wall_mean": float(np.mean([r["eval_wall_s"] for r in rs])),
            "total_params": rs[0]["total_params"],
            "requires_grad_count": rs[0]["requires_grad_count"],
        }

    per_arm = {a["name"]: agg(a["name"]) for a in ARMS}

    def matched_verdict():
        # R2 / A3, B5: match on REFINEMENT work per valid token, not total MAC.
        # One refinement step per row costs (BLOCK + HALT); the tolerance is
        # HALF a step, so two arms within the band differ by less than the
        # granularity of a single refinement update.
        one_step = (BLOCK + HALT)
        tol = MATCH_TOL_FRACTION * one_step
        a = per_arm["adaptive"]["refine_per_token_per_seed"]
        f = per_arm["fixed_K3"]["refine_per_token_per_seed"]
        if len(a) != len(f):
            return {"verdict": "NOTMATCHED", "reason": "seed count mismatch"}
        absdiff = [abs(x - y) for x, y in zip(a, f)]
        within = [d <= tol for d in absdiff]
        fixed_names = ("fixed_K4", "fixed_K3", "fixed_K2")
        per_seed_nearest = []
        for i in range(len(a)):
            best = min(fixed_names, key=lambda nm: abs(
                a[i] - per_arm[nm]["refine_per_token_per_seed"][i]))
            per_seed_nearest.append(best)
        # MATCHED requires BOTH: within the half-step band AND the predeclared
        # fixed_K3 is the per-seed nearest fixed arm.
        ok = all(within) and all(
            nm == "fixed_K3" for nm in per_seed_nearest)
        return {
            "predeclared_pair": ["adaptive", "fixed_K3"],
            "tolerance_abs_refine_per_token": tol,
            "tolerance_fraction_of_one_step": MATCH_TOL_FRACTION,
            "one_step_mac_per_row": one_step,
            "abs_diff_per_seed": absdiff,
            "within_band_per_seed": within,
            "verdict": "MATCHED" if ok else "NOTMATCHED",
            "per_seed_nearest_fixed_arm": per_seed_nearest,
            "policy": ("A single fixed arm must match adaptive PER-SEED within "
                       "tolerance. NOTMATCHED forbids any budget-advantage / "
                       "architecture-v1 condition claim; the band is fixed a "
                       "priori and is never tuned to force MATCHED."),
        }

    def paired_stats():
        out = {}
        pvals = {}
        for a_name, b_name in CONTRASTS:
            # R2 / A2, B3: held-out (independent validation) task metric, never
            # in-sample training fit.
            ga = np.array([r["val_task_mse"] for r in rows_of(a_name)])
            gb = np.array([r["val_task_mse"] for r in rows_of(b_name)])
            diff = ga - gb
            n = len(diff)
            entry = {"n_pairs": n, "mean_diff": float(diff.mean())}
            if n < 2 or np.allclose(diff, diff[0]):
                entry["method"] = "degenerate_zero_variance_undefined"
                entry["p_value"] = None
                entry["cohens_dz_paired"] = None
                entry["note"] = ("zero paired variance; paired t-statistic "
                                 "undefined. NOT fabricated as p=0 or p=1; "
                                 "the constant mean_diff is reported as-is.")
            else:
                t, p = sstats.ttest_rel(ga, gb)
                entry["method"] = "scipy.stats.ttest_rel"
                entry["t"] = float(t)
                entry["p_value"] = float(p)
                entry["cohens_dz_paired"] = float(diff.mean() / diff.std(ddof=1))
            out[f"{a_name}_vs_{b_name}"] = entry
            if entry["p_value"] is not None:
                pvals[f"{a_name}_vs_{b_name}"] = entry["p_value"]
        adj = holm_bonferroni(pvals) if pvals else {}
        return out, adj

    stats_out, adj = paired_stats()

    conv_statuses = sorted({
        r["convergence_status"] for r in all_rows
    })
    # H7 / B6: fail closed at the top level.  Any nonfinite arm or any
    # accounting mismatch forbids a successful prepared-gate status.
    any_nonfinite = any(not r["finite_stable"] for r in all_rows)
    any_skipped = any(r["n_skipped_steps"] > 0 for r in all_rows)
    any_accounting_bad = any(not r["row_count_assert_ok"] for r in all_rows)
    if any_nonfinite:
        run_status = "FINAL_CONTROL_ABORTED_NONFINITE"
    elif any_skipped:
        run_status = "FINAL_CONTROL_ABORTED_SKIPPED_STEPS"
    elif any_accounting_bad:
        run_status = "FINAL_CONTROL_ABORTED_ACCOUNTING"
    else:
        run_status = "FINAL_CONTROL_PREPARED_GATE_EXECUTABLE"
    result = {
        "status": run_status,
        "attempt": attempt, "epochs": epochs, "seeds": seeds, "H": H,
        "validation_split": ("independent synthetic split; generator seed "
                             "10_000 + training seed; never trained on"),
        "budget_definition": {
            "verified_topology": {
                "encoder": "Linear(4, H)", "gru": "GRU(input_size=H, hidden=H, layers=1)",
                "lif_W_in": "Linear(H, H)", "router": "Linear(H+M, 2)",
                "head": "Linear(H, 1)", "refinement_block": "fc1,fc2 = Linear(H,H)",
                "halt": "Linear(H, 1)",
            },
            "base_mac_per_token": BASE,
            "shared_block_mac_per_row": BLOCK,
            "halt_mac_per_row": HALT,
            "total_est_mac": ("(train_tokens + val_tokens + eval_tokens)*base_total "
                              "+ (train+val+eval block_rows)*block "
                              "+ (train+val+eval halt_rows)*halt"),
            "matched_budget_metric": ("refinement work per VALID TOKEN = "
                                      "(block_rows*BLOCK + halt_rows*HALT) / "
                                      "total_valid_tokens, costed per unit (a "
                                      "fixed arm executes no halt, so its halt "
                                      "rows are 0); one row per valid token, so "
                                      "a deterministic per-token constant. "
                                      "Tolerance = half a step."),
            "capacity": {"total_params": per_arm["off"]["total_params"],
                         "requires_grad": per_arm["off"]["requires_grad_count"]},
            "NOT_COUNTED": ["elementwise activations", "LayerNorm",
                            "softmax/nonlinearities", "backward pass",
                            "memory traffic"],
            "scope": ("LINEAR-MAC ESTIMATE, forward only, with declared per-unit "
                      "formulas. NOT measured hardware MAC, NOT ATP. Backward "
                      "cost is excluded; wall-clock is reported separately and "
                      "never merged."),
        },
        "matched_budget": matched_verdict(),
        "per_arm": per_arm,
        "comparison_paired_task_metric": stats_out,
        "holm_bonferroni_adjusted": adj,
        "convergence_gate": {
            "criterion": ("On the held-out validation loss: best epoch is "
                          "recorded; the run must span >= window+burn-in epochs "
                          "else INSUFFICIENT_EPOCHS; the mean of the last "
                          "CONV_WINDOW epochs must not exceed the best by more "
                          "than CONV_SLACK; a still-improving trajectory is "
                          "NOT_CONVERGED, and DIVERGED is reserved for a "
                          "worsening trajectory whose latest epoch exceeds the "
                          "best by more than CONV_SLACK."),
            "CONV_SLACK": CONV_SLACK,
            "CONV_WINDOW": CONV_WINDOW,
            "CONV_BURN_IN": CONV_BURN_IN,
            "status_per_seed": conv_statuses,
            "best_epoch_recorded": True,
            "last5_recorded": True,
            "gate_verdict": ("CONVERGED" if conv_statuses == ["CONVERGED"]
                             else "NOT_CONVERGED_OR_MIXED"),
            "note": ("The verdict is COMPUTED from the per-epoch validation "
                     "trajectory. The final <=20-epoch/multiseed control gate "
                     "remains gated on dual source ACCEPT; this run is a short "
                     "mechanism smoke."),
        },
        "evidence_scope": (
            "MECHANISM smoke: work counters/budget are exact (forward hooks "
            "asserted against depth.sum()); the PAIRED comparison and the "
            "convergence verdict both use the INDEPENDENT synthetic validation "
            "split (never trained on). Synthetic mechanism evidence ONLY; no "
            "extrapolation to animal or best-actual-data performance. "
            "Training-loss penalties are never used as the effect."
        ),
    }
    # R2 / B12: atomic, finite-only write.  Serialize FIRST (allow_nan=False
    # raises before any file is touched), then write to a temp file and rename.
    payload = json.dumps(result, indent=2, allow_nan=False)
    tmp_results = OUT / f".final_control_v3_{attempt}_results.tmp"
    final_results = OUT / f"final_control_v3_results_{attempt}.json"
    tmp_results.write_text(payload)
    os.replace(tmp_results, final_results)
    print(json.dumps({"attempt": attempt, "epochs": epochs, "seeds": seeds,
                      "matched": result["matched_budget"]["verdict"],
                      "convergence": result["convergence_gate"]["gate_verdict"],
                      "status": run_status}))
    # A nonfinite/skipped/accounting failure is a HARD failure (non-zero exit).
    return 1 if (any_nonfinite or any_skipped or any_accounting_bad) else 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        _inject = sys.argv[i + 4] if len(sys.argv) > i + 4 else "none"
        raise SystemExit(child(sys.argv[i + 1], int(sys.argv[i + 2]),
                               int(sys.argv[i + 3]), _inject))
    raise SystemExit(main())
