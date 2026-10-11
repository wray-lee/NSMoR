"""Tests for the resume-capable phase-2 launcher's decision helper (S6).

The launcher ``run_frozen_resume.sh`` refuses a non-empty output dir unless a
recoverable current-schema checkpoint exists AND its provenance matches (same
frozen commit, modules resolved inside the worktree, same config SHA-256).  The
decision and the ``resume_log`` writer live in a CPU-only helper,
``resume_decision.py``, which this module exercises directly so the logic has
regression coverage without running the launcher (which would train).

The helper path is outside the repo (the launcher lives in the job dir); the
tests skip cleanly if it is absent rather than failing a checkout that has no
job dir.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
import torch

_HELPER = Path(__file__).resolve().parents[1] / "scripts" / "resume_decision.py"
_WT = "/mnt/d/Projects/NSMoR-resume"
_FROZEN = "e475d663d85ae4285a223ec704c0fa3fe4322e40"

pytestmark = pytest.mark.skipif(
    not _HELPER.exists(), reason=f"launcher helper not present: {_HELPER}"
)


def _helper() -> Any:
    spec = importlib.util.spec_from_file_location("resume_decision", _HELPER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["resume_decision"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_ckpt(
    path: Path, epoch: int, version: int = 2, *, nested: bool = True,
    selection_metric: str = "total", zic: tuple = (),
) -> None:
    # ``nested=True`` models the phase-2 corpus (is_nested_cv=True).  A
    # non-nested historical payload additionally requires an explicit trusted
    # SHA pin the launcher does not supply, so it must be refused at decide()
    # time rather than half-starting at train() time.
    torch.save({
        "recovery_state_version": version,
        "epoch": epoch,
        "is_nested_cv": nested,
        "animal_identity_status": "verified" if nested else "historical_unknown",
        "selection_metric": selection_metric,
        "zero_input_channels": list(zic),
        "model_state_dict": {"w": torch.zeros(2)},
        "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 0.1}]},
        "scheduler_state_dict": {"last_epoch": epoch},
        "rng_state": torch.zeros(8, dtype=torch.uint8),
    }, path)


def _write_prov(out: Path, cfg: Path, head: str, modules: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps({
        "arm": out.name, "head": head, "resolved_modules": modules,
        "sha256": {str(cfg): hashlib.sha256(cfg.read_bytes()).hexdigest()},
        "resume_log": [],
    }, indent=2) + "\n")


def _decide(out: Path, cfg: Path, n: int = 10) -> tuple:
    mod = _helper()
    buf = StringIO()
    with redirect_stdout(buf):
        rc = mod.decide(str(out), _WT, _FROZEN, str(cfg), str(n))
    return rc, buf.getvalue().strip()


def test_resume_when_provenance_and_checkpoint_match(tmp_path: Path) -> None:
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py {_WT}/scripts/train.py")
    _write_ckpt(out / "epoch_4.pth", 4)
    _write_ckpt(out / "best_model.pth", 4)
    rc, msg = _decide(out, cfg)
    assert rc == 0, msg
    assert msg.endswith("epoch_4.pth"), msg


@pytest.mark.parametrize(
    "head,modules,version,epoch,n,needle",
    [
        ("deadbeef" * 5, f"{_WT}/nsmor/__init__.py", 2, 4, 10,
         "cross-code-version"),
        (_FROZEN, "/elsewhere/nsmor/__init__.py", 2, 4, 10, "outside the worktree"),
        (_FROZEN, f"{_WT}/nsmor/__init__.py", 1, 4, 10, "not a current-schema"),
        (_FROZEN, f"{_WT}/nsmor/__init__.py", 2, 9, 10, "no remaining epochs"),
    ],
)
def test_refusals(
    tmp_path: Path, head: str, modules: str, version: int, epoch: int, n: int,
    needle: str,
) -> None:
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, head, modules)
    _write_ckpt(out / "epoch_4.pth", epoch, version=version)
    rc, msg = _decide(out, cfg, n)
    assert rc == 1, msg
    assert needle in msg, msg


def test_refuses_config_sha_mismatch(tmp_path: Path) -> None:
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    out.mkdir()
    _write_ckpt(out / "epoch_4.pth", 4)
    (out / "provenance.json").write_text(json.dumps({
        "head": _FROZEN, "resolved_modules": f"{_WT}/nsmor/__init__.py",
        "sha256": {str(cfg): "0" * 64}, "resume_log": [],
    }))
    rc, msg = _decide(out, cfg)
    assert rc == 1 and "config sha mismatch" in msg, msg


def test_refuses_when_no_checkpoint(tmp_path: Path) -> None:
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    (out / "notes.txt").write_text("partial run, no checkpoint")
    rc, msg = _decide(out, cfg)
    assert rc == 1 and "no recoverable checkpoint" in msg, msg


def test_refuses_when_only_best_model_exists(tmp_path: Path) -> None:
    """F1: best_model.pth is the best epoch, not the last, so it is NOT a source.

    A dir holding ONLY ``best_model.pth`` has no last-complete-epoch checkpoint
    to resume from: the helper must REFUSE, never resume from the best (which
    would re-train every epoch after the best one).
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "best_model.pth", 3)
    rc, msg = _decide(out, cfg)
    assert rc == 1, msg
    assert "no recoverable checkpoint" in msg, msg


def test_best_model_is_not_used_even_when_newest(tmp_path: Path) -> None:
    """F1 repro: best_model.pth newest must NOT win; the last periodic does.

    ``best_model.pth`` stores the BEST selected epoch (here 8), which is newer
    than the newest periodic (``epoch_2.pth``, stored epoch 2).  A family-blind
    newest-file rule would resume from the best and silently re-train epochs
    3-8.  The candidate set must EXCLUDE ``best_model.pth`` and resume from the
    newest last-epoch family file (``epoch_2.pth``).
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_2.pth", 2)       # last-epoch family
    _write_ckpt(out / "best_model.pth", 8)    # best epoch, NEWER file
    rc, msg = _decide(out, cfg)
    assert rc == 0, msg
    assert msg.endswith("epoch_2.pth"), msg


@pytest.mark.parametrize(
    "modules",
    ["", "   ", "\t\n"],
    ids=["empty", "spaces", "whitespace"],
)
def test_refuses_empty_or_whitespace_resolved_modules(
    tmp_path: Path, modules: str,
) -> None:
    """F2: an empty / whitespace ``resolved_modules`` must fail closed, not open.

    The module list is what binds the run's code to this worktree.  An empty or
    whitespace-only field makes the containment loop a no-op, which previously
    let an unverifiable provenance pass as if verified (a FAIL-OPEN).  It must
    REFUSE.
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, modules)
    _write_ckpt(out / "epoch_4.pth", 4)
    rc, msg = _decide(out, cfg)
    assert rc == 1, msg
    assert "resolved_modules" in msg, msg


def test_refuses_absent_resolved_modules(tmp_path: Path) -> None:
    """F2: a provenance with NO ``resolved_modules`` key must also refuse."""
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    out.mkdir(parents=True)
    (out / "provenance.json").write_text(json.dumps({
        "head": _FROZEN,
        "sha256": {str(cfg): hashlib.sha256(cfg.read_bytes()).hexdigest()},
        "resume_log": [],
    }))
    _write_ckpt(out / "epoch_4.pth", 4)
    rc, msg = _decide(out, cfg)
    assert rc == 1, msg
    assert "resolved_modules" in msg, msg


def test_record_appends_resume_log(tmp_path: Path) -> None:
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    ckpt = out / "epoch_4.pth"
    _write_ckpt(ckpt, 4)

    mod = _helper()
    mod.record(str(out), str(ckpt), _FROZEN)
    prov = json.loads((out / "provenance.json").read_text())
    assert len(prov["resume_log"]) == 1
    entry = prov["resume_log"][0]
    assert entry["checkpoint_epoch"] == 4
    assert entry["source"] == "run_frozen_resume.sh"
    assert entry["head"] == _FROZEN
    assert entry["origin_commit"] is None
    assert "time" in entry
    # Atomic write leaves no temp file behind.
    assert not (out / "provenance.json.tmp").exists()


def test_record_writes_both_commits_on_successor_resume(tmp_path: Path) -> None:
    """A successor-commit resume records BOTH C (origin) and C' (head) (§11.1)."""
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    ckpt = out / "epoch_4.pth"
    _write_ckpt(ckpt, 4)

    mod = _helper()
    mod.record(str(out), str(ckpt), _FROZEN_C, _ORIGIN)
    entry = json.loads((out / "provenance.json").read_text())["resume_log"][0]
    assert entry["head"] == _FROZEN_C
    assert entry["origin_commit"] == _ORIGIN


def test_resume_from_final_model_when_no_periodic(tmp_path: Path) -> None:
    """checkpoint_interval=10 + a non-multiple-of-10 pause writes only final_model.pth.

    decide() must recover it (the segment's last complete epoch), not refuse.
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "best_model.pth", 3)
    _write_ckpt(out / "final_model.pth", 5)  # last complete epoch, no epoch_*.pth
    rc, msg = _decide(out, cfg)
    assert rc == 0, msg
    assert msg.endswith("final_model.pth"), msg


def test_resume_uses_newest_stored_epoch_across_file_families(tmp_path: Path) -> None:
    """BLOCKER repro: a newer final_model.pth must beat an older epoch_*.pth.

    With ``checkpoint_interval: 10`` a pause at epoch 12 writes only
    ``final_model.pth`` (epoch 12); the newest periodic is ``epoch_10.pth``
    (stored epoch 9).  A family-first ordering would resume from epoch 9 and
    silently re-train epochs 10-12, breaking §11.1's 'last complete-epoch
    checkpoint'.  The candidate order must be by stored epoch, not by family.
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 30\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_10.pth", 9)      # periodic, OLDER
    _write_ckpt(out / "final_model.pth", 12)  # newer than the periodic
    rc, msg = _decide(out, cfg, n=30)
    assert rc == 0, msg
    assert msg.endswith("final_model.pth"), msg


def test_resume_prefers_periodic_when_it_is_newer(tmp_path: Path) -> None:
    """The ordering is symmetric: a newer periodic beats an older final_model.pth.

    A SIGKILL mid-interval can leave an older ``final_model.pth`` from a prior
    segment and a newer periodic from the interrupted one; the newer stored
    epoch must win regardless of family.
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 30\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "final_model.pth", 7)   # stale prior segment
    _write_ckpt(out / "epoch_20.pth", 19)     # newer
    rc, msg = _decide(out, cfg, n=30)
    assert rc == 0, msg
    assert msg.endswith("epoch_20.pth"), msg


def test_refuses_control_switch_before_launch(tmp_path: Path) -> None:
    """A checkpoint whose phase-2 controls differ from the arm is refused early.

    ``train()`` would fail closed on the selection_metric/zero_input_channels
    mismatch, but the launcher must not spawn a doomed process.
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text(
        "training:\n  num_epochs: 10\n"
        "phase2:\n  selection_metric: total\n  zero_input_channels: []\n"
    )
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_4.pth", 4, selection_metric="mse", zic=(2, 3))
    rc, msg = _decide(out, cfg)
    assert rc == 1, msg
    assert "phase-2 controls" in msg, msg


def test_accepts_matching_mse_arm_controls(tmp_path: Path) -> None:
    """An R2-style arm (mse + zero_input_channels=[2,3]) resumes its own checkpoint."""
    cfg = tmp_path / "arm.yaml"
    cfg.write_text(
        "training:\n  num_epochs: 10\n"
        "phase2:\n  selection_metric: mse\n  zero_input_channels:\n  - 2\n  - 3\n"
    )
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_4.pth", 4, selection_metric="mse", zic=(2, 3))
    rc, msg = _decide(out, cfg)
    assert rc == 0, msg
    assert msg.endswith("epoch_4.pth"), msg


def test_refuses_when_newest_checkpoint_exhausts_budget(tmp_path: Path) -> None:
    """A completed run is refused, not resumed from an older periodic.

    If the NEWEST complete-epoch checkpoint (here ``final_model.pth`` at epoch
    29 of a 30-epoch run) has no remaining epochs, the run is done.  Falling
    through to an older ``epoch_20.pth`` would silently re-train completed
    epochs and exceed the budget.
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 30\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_20.pth", 19)     # older periodic
    _write_ckpt(out / "final_model.pth", 29)  # newest, run complete
    rc, msg = _decide(out, cfg, n=30)
    assert rc == 1, msg
    assert "no remaining epochs" in msg, msg


def test_falls_through_a_corrupt_newest_checkpoint(tmp_path: Path) -> None:
    """A truncated newest epoch_*.pth falls through to the next-oldest valid one."""
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_2.pth", 2)
    (out / "epoch_6.pth").write_bytes(b"not a torch checkpoint")
    rc, msg = _decide(out, cfg)
    assert rc == 0, msg
    assert msg.endswith("epoch_2.pth"), msg


def test_refuses_non_nested_historical_without_trusted_pin(tmp_path: Path) -> None:
    """A non-nested historical payload needs a trusted SHA the launcher omits.

    decide() must refuse it (fail closed early), not declare it resumable and
    then fail at train() time.
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_4.pth", 4, nested=False)
    rc, msg = _decide(out, cfg)
    assert rc == 1, msg
    assert "trusted_historical_checkpoint_sha256" in msg, msg


# ── Successor-commit resume (§11.1 amendment, 2026-10-11) ─────────────
# A run started on commit C may be resumed on a successor commit C' only with
# the ORIGIN_COMMIT=<C> opt-in AND git confirming C is an ancestor of C'.  These
# tests exercise the decision logic hermetically: ``_git_ancestor`` is
# monkeypatched so ancestry is controlled without a real repository (the WSL git
# cannot read the worktrees' Windows ``.git``).

_ORIGIN = "8ed5edb994ef105db0551aedc50f6003b97fe87f"
_FROZEN_C = "ef3c3aac51f1f60f926cc1b47ad27041f31464b0"


def _successor_fixture(tmp_path: Path) -> tuple:
    """A (cprime_worktree, origin_worktree, out, cfg) tuple for a successor run.

    The active worktree (C') holds the config; the run's ``code_worktree`` (C)
    is a DIFFERENT directory, so the config key and the recorded modules must be
    re-rooted onto the origin worktree to match.
    """
    cprime_wt = tmp_path / "wt_prime"
    origin_wt = tmp_path / "wt_origin"
    (cprime_wt / "config").mkdir(parents=True)
    origin_wt.mkdir()
    cfg = cprime_wt / "config" / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    out.mkdir(parents=True)
    (out / "provenance.json").write_text(json.dumps({
        "arm": out.name,
        "head": _ORIGIN,
        "code_worktree": str(origin_wt),
        "resolved_modules": f"{origin_wt}/nsmor/__init__.py",
        "sha256": {
            str(origin_wt / "config" / "arm.yaml"):
                hashlib.sha256(cfg.read_bytes()).hexdigest(),
        },
        "resume_log": [],
    }, indent=2) + "\n")
    _write_ckpt(out / "epoch_50.pth", 49)
    return cprime_wt, origin_wt, out, cfg


def _decide_successor(
    out: Path, cprime_wt: Path, cfg: Path, *, origin: str | None,
    ancestor: bool, n: int = 300,
) -> tuple:
    mod = _helper()
    mod._git_ancestor = lambda wt, a, d: ancestor  # type: ignore[assignment]
    buf = StringIO()
    with redirect_stdout(buf):
        rc = mod.decide(
            str(out), str(cprime_wt), _FROZEN_C, str(cfg), str(n), origin,
        )
    return rc, buf.getvalue().strip()


def test_successor_resume_accepted_when_origin_is_ancestor(tmp_path: Path) -> None:
    """head==C, C an ancestor of C', ORIGIN_COMMIT given → accepted, re-rooted."""
    cprime_wt, _origin_wt, out, cfg = _successor_fixture(tmp_path)
    rc, msg = _decide_successor(out, cprime_wt, cfg, origin=_ORIGIN, ancestor=True)
    assert rc == 0, msg
    assert msg.endswith("epoch_50.pth"), msg


def test_successor_resume_refused_when_origin_not_ancestor(tmp_path: Path) -> None:
    """ORIGIN_COMMIT given but C is NOT an ancestor of C' → refused."""
    cprime_wt, _origin_wt, out, cfg = _successor_fixture(tmp_path)
    rc, msg = _decide_successor(out, cprime_wt, cfg, origin=_ORIGIN, ancestor=False)
    assert rc == 1, msg
    assert "not an ancestor" in msg, msg


def test_successor_resume_refused_without_optin(tmp_path: Path) -> None:
    """head != frozen with NO ORIGIN_COMMIT → refused as cross-code-version."""
    cprime_wt, _origin_wt, out, cfg = _successor_fixture(tmp_path)
    rc, msg = _decide_successor(out, cprime_wt, cfg, origin=None, ancestor=True)
    assert rc == 1, msg
    assert "cross-code-version" in msg, msg


def test_successor_resume_refused_when_optin_names_other_head(
    tmp_path: Path,
) -> None:
    """ORIGIN_COMMIT must name the run's OWN recorded head, not any ancestor."""
    cprime_wt, _origin_wt, out, cfg = _successor_fixture(tmp_path)
    rc, msg = _decide_successor(
        out, cprime_wt, cfg, origin="deadbeef" * 5, ancestor=True,
    )
    assert rc == 1, msg
    assert "different origin" in msg, msg


def test_successor_resume_refused_when_git_check_fails(tmp_path: Path) -> None:
    """A git infrastructure failure (no git.exe / bad repo) fails closed."""
    cprime_wt, _origin_wt, out, cfg = _successor_fixture(tmp_path)
    mod = _helper()

    def boom(wt: str, a: str, d: str) -> bool:
        raise RuntimeError("git.exe not found")

    mod._git_ancestor = boom  # type: ignore[assignment]
    buf = StringIO()
    with redirect_stdout(buf):
        rc = mod.decide(str(out), str(cprime_wt), _FROZEN_C, str(cfg), "300",
                        _ORIGIN)
    assert rc == 1, buf.getvalue()
    assert "could not be performed" in buf.getvalue(), buf.getvalue()


def test_dry_run_returns_decision_without_writing(tmp_path: Path) -> None:
    """--dry-run reports the decision and writes NOTHING to the run dir."""
    cprime_wt, _origin_wt, out, cfg = _successor_fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    mod = _helper()
    mod._git_ancestor = lambda wt, a, d: True  # type: ignore[assignment]
    err = StringIO()
    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = mod.decide(
            str(out), str(cprime_wt), _FROZEN_C, str(cfg), "300", _ORIGIN,
            dry_run=True,
        )
    assert rc == 0, buf.getvalue()
    assert buf.getvalue().strip().endswith("epoch_50.pth")
    assert "next epoch 50" in err.getvalue(), err.getvalue()
    after = {p.name: p.read_bytes() for p in out.iterdir()}
    assert after == before, "dry run must not write to the run dir"


def test_dry_run_marks_same_commit_path_too(tmp_path: Path) -> None:
    """m2: --dry-run must emit the [dry-run] chosen-source marker on the
    SAME-commit path (recorded_head == frozen), not only the successor path.

    On the same-commit path the recorded module/config roots are the worktree
    itself, so the decider never enters the successor branch; the chosen-source
    ``_report`` still runs and must carry the ``[dry-run]`` marker, so a
    ``--dry-run`` never prints a bare path with no dry-run indication.
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 10\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_4.pth", 4)
    mod = _helper()
    err = StringIO()
    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = mod.decide(
            str(out), _WT, _FROZEN, str(cfg), "10", None, dry_run=True,
        )
    assert rc == 0, buf.getvalue()
    assert buf.getvalue().strip().endswith("epoch_4.pth")
    assert "[dry-run]" in err.getvalue(), err.getvalue()
    assert "chosen source" in err.getvalue(), err.getvalue()


def _write_log(out: Path, last_epoch_1based: int, budget: int = 300) -> None:
    """A minimal progress log whose last line is ``Epoch N/<budget>``."""
    (out / "train.stdout.log").write_text(
        f"[t] INFO - Epoch {last_epoch_1based}/{budget}  train_loss=1.0\n"
    )


def _write_segment(out: Path, starting_epoch: int, index: int = 1) -> None:
    """A minimal segment record: ``starting_epoch`` = the NEXT epoch to run."""
    (out / f"segment_{index:04d}.json").write_text(
        json.dumps({"starting_epoch": starting_epoch})
    )


def test_refuses_silent_rollback_when_log_reached_newer_epoch(
    tmp_path: Path,
) -> None:
    """F1: a source older than the run's LAST EXECUTED epoch is refused.

    Synthetic seed-43 shape: periodics only through ``epoch_50.pth`` (stored 49)
    but the log's last line is ``Epoch 58/300`` (0-based epoch 57 executed).  The
    newest-complete-epoch rule would pick ``epoch_50.pth``; F1 must refuse it and
    NAME both epochs, so the completed epochs 50..57 are never silently
    re-trained.
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 300\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_50.pth", 49)      # newest EXISTING checkpoint
    _write_ckpt(out / "best_model.pth", 57)    # best epoch, not a source
    _write_log(out, last_epoch_1based=58)      # 0-based epoch 57 was executed
    rc, msg = _decide(out, cfg, n=300)
    assert rc == 1, msg
    assert "no-silent-rollback" in msg, msg
    assert "49" in msg and "57" in msg, msg


def test_accepts_clean_boundary_when_log_matches_newest(
    tmp_path: Path,
) -> None:
    """F1: a source whose stored epoch equals the last executed epoch is OK.

    Synthetic seed-44 shape: the newest checkpoint is ``epoch_150.pth`` (stored
    149) and the log's last line is ``Epoch 150/300`` — a clean boundary with no
    rollback, so the resume is ACCEPTED and selects ``epoch_150.pth``.
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 300\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_150.pth", 149)
    _write_ckpt(out / "best_model.pth", 142)
    _write_log(out, last_epoch_1based=150)
    rc, msg = _decide(out, cfg, n=300)
    assert rc == 0, msg
    assert msg.endswith("epoch_150.pth"), msg


def test_dry_run_refusal_writes_nothing(tmp_path: Path) -> None:
    """F1: a dry-run refusal names both epochs and writes NOTHING to the dir."""
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 300\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_50.pth", 49)
    _write_log(out, last_epoch_1based=57)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    mod = _helper()
    err = StringIO()
    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = mod.decide(str(out), _WT, _FROZEN, str(cfg), "300", None, dry_run=True)
    assert rc == 1, buf.getvalue()
    assert "no-silent-rollback" in buf.getvalue(), buf.getvalue()
    after = {p.name: p.read_bytes() for p in out.iterdir()}
    assert after == before, "dry run must not write to the run dir"


def test_segment_starting_epoch_is_a_witness(tmp_path: Path) -> None:
    """F1: a segment record witnesses the epoch BEFORE its ``starting_epoch``.

    ``_write_segment_record`` stores ``starting_epoch = stored_epoch + 1`` (the
    NEXT epoch to execute, written before the loop), so a segment whose
    ``starting_epoch`` is S proves the run executed through S-1.  Here S=58, so
    the run executed through 57 while the newest checkpoint is ``epoch_50.pth``
    (stored 49): a rollback, refused with BOTH epochs (49 and 57) named.
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 300\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_50.pth", 49)
    _write_segment(out, 58)
    rc, msg = _decide(out, cfg, n=300)
    assert rc == 1, msg
    assert "no-silent-rollback" in msg and "49" in msg and "57" in msg, msg


def test_h1_clean_boundary_with_resumed_segment_is_accepted(tmp_path: Path) -> None:
    """H1 BLOCKER repro: a segment record must NOT over-claim by one.

    A run killed before completing its resumed segment's FIRST epoch leaves
    ``epoch_150.pth`` (stored 149), the log's last line ``Epoch 150/300``, and
    ``segment_0001.json``{``starting_epoch``: 150}.  The segment witnesses only
    epoch 149 (its start is the NEXT epoch to run), so the last executed epoch is
    149 — equal to the source's stored epoch: a CLEAN BOUNDARY that MUST be
    ACCEPTED.  The pre-fix code read ``starting_epoch`` RAW (150) and refused
    with no override, bricking the seed-44 resume (F1 round-4 BLOCKER).
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 300\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_150.pth", 149)
    _write_ckpt(out / "best_model.pth", 142)
    _write_log(out, last_epoch_1based=150)
    _write_segment(out, 150)
    rc, msg = _decide(out, cfg, n=300)
    assert rc == 0, msg
    assert msg.endswith("epoch_150.pth"), msg


def test_h1_segment_offbyone_still_refuses_real_rollback(tmp_path: Path) -> None:
    """H1: the start-1 correction must NOT turn a real rollback into an accept.

    A segment that started at 58 (so executed through 57) with the newest
    checkpoint ``epoch_50.pth`` (stored 49) is a rollback of epochs 50..57: the
    start-1 correction leaves 49 < 57, so it is still REFUSED (the fix narrows
    the false refusals, it does not weaken the true ones).
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 300\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_50.pth", 49)
    _write_segment(out, 58)
    rc, msg = _decide(out, cfg, n=300)
    assert rc == 1, msg
    assert "no-silent-rollback" in msg and "49" in msg and "57" in msg, msg


def test_final_model_is_a_witness_in_the_decider(tmp_path: Path) -> None:
    """H2: the decider's witness set includes ``final_model.pth`` (same as train.py).

    A completed/paused run writes ``final_model.pth`` with that run's last
    executed epoch, so it IS a valid executed-epoch witness.  Driven through the
    helper directly (the full ``decide`` path would select the newer
    ``final_model.pth`` itself as the source, masking the witness role).
    """
    mod = _helper()
    out = tmp_path / "arm"
    out.mkdir()
    _write_ckpt(out / "epoch_50.pth", 49)
    _write_ckpt(out / "final_model.pth", 57)  # last executed epoch of a segment
    assert mod._last_executed_epoch(out) == 57


def test_decider_and_train_share_the_checkpoint_witness_set() -> None:
    """H2: the two F1 enforcers must use the SAME checkpoint witness set.

    Round-4 MAJOR: the decider treated ``final_model.pth`` as a witness while
    train.py excluded it — the 'same rule' claim was false.  Pin the constants
    equal so they cannot silently drift apart again.
    """
    mod = _helper()
    from scripts import train as T

    assert tuple(mod._CHECKPOINT_WITNESS_FAMILIES) == tuple(
        T._CHECKPOINT_WITNESS_FAMILIES
    )
    assert set(mod._RESUME_EXCLUDED) == set(T._NON_RESUME_WITNESS_FAMILIES)


def test_decider_and_train_share_the_epoch_progress_regex() -> None:
    """R2 minor: the two F1 enforcers must share the SAME log-parse pattern.

    The decider previously compiled its epoch-progress pattern INLINE, so a
    future edit to one enforcer's literal would silently change the witness on
    that one only — the exact drift the H2 shared-witness-set fix exists to
    eliminate, but unpinned for the log regex.  Both now use a named
    ``_EPOCH_PROGRESS_RE``; pin the two literals identical.
    """
    mod = _helper()
    from scripts import train as T

    assert mod._EPOCH_PROGRESS_RE.pattern == T._EPOCH_PROGRESS_RE.pattern


def test_decider_accepts_post_resume_dir_with_segment_log(tmp_path: Path) -> None:
    """Round-5 BLOCKER: the decider must ACCEPT the dir after the first resume.

    The first resume writes the resumed segment's OWN ``train.segment1.log``
    (``Epoch 151/300`` once its first epoch completes) and ``epoch_160.pth`` (the
    next periodic after stored epoch 159).  The run has genuinely executed
    through 150, so the newer ``epoch_160.pth`` IS the newest complete-epoch
    checkpoint and the resume is a CLEAN BOUNDARY — accepted.  This is the
    decider counterpart of the state-independent real-dir test, on a synthetic
    dir (fast, and independent of the live seed-44 dir's mutable state).
    """
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  num_epochs: 300\n")
    out = tmp_path / "arm"
    _write_prov(out, cfg, _FROZEN, f"{_WT}/nsmor/__init__.py")
    _write_ckpt(out / "epoch_150.pth", 149)
    _write_ckpt(out / "epoch_160.pth", 159)
    _write_ckpt(out / "best_model.pth", 142)
    _write_log(out, last_epoch_1based=150)  # the first segment's log
    (out / "train.segment1.log").write_text(
        "[t] INFO - Epoch 151/300  train_loss=1.0\n"
    )
    _write_segment(out, 150)  # the first resume's segment record
    rc, msg = _decide(out, cfg, n=300)
    assert rc == 0, msg
    assert msg.endswith("epoch_160.pth"), msg


def test_git_ancestor_real_repository() -> None:
    """R1-minor: the ancestry gate is exercised against a REAL repository.

    The other ancestry cases monkeypatch ``_git_ancestor`` (they pin the
    DECISION logic).  This test drives the UNSTUBBED ``_git_ancestor`` against
    the real worktree repository: the ancestor direction (``HEAD~1`` is an
    ancestor of ``HEAD``) is true, the reverse is false, and an invalid SHA
    raises (fail closed) rather than returning a guess.  Skips cleanly where
    ``git.exe`` / ``wslpath`` are unavailable or the repo has no parent commit.
    """
    repo = str(Path(__file__).resolve().parents[1])
    mod = _helper()
    try:
        assert mod._git_ancestor(repo, "HEAD~1", "HEAD") is True
        assert mod._git_ancestor(repo, "HEAD", "HEAD~1") is False
    except RuntimeError as exc:  # pragma: no cover - env without git.exe/wslpath
        pytest.skip(f"git.exe/wslpath unavailable from this environment: {exc}")
    with pytest.raises(RuntimeError):
        mod._git_ancestor(repo, "deadbeef" * 5, "HEAD")


# ── Real seed-44 end-to-end decider regression (round-4 H4 minor) ──────────
# The accepted-path real evidence was previously the developer's manual dry run
# plus synthetic-dir tests; this drives the UNSTUBBED decider against the REAL
# seed-44 run dir, READ-ONLY (dry run writes nothing).  Skips when the
# read-only corpus / run dir is absent (a machine without the phase-2 artifacts).
_REAL_ROOT = Path(
    "/mnt/d/Projects/NSMoR-phase2-8ed5edb/.scratch/realdata-phase2-20261010"
)
_REAL_SEED44 = _REAL_ROOT / "R0-control-seed44"
_REAL_CORPUS = Path(
    "/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c"
)
_REAL_ORIGIN = "8ed5edb994ef105db0551aedc50f6003b97fe87f"


@pytest.mark.skipif(
    not (_REAL_SEED44 / "epoch_150.pth").exists()
    or not (_REAL_CORPUS / "etl-causal-final-zp12m0fa/nsmor_dataset.pt").exists(),
    reason="real seed-44 run dir / corpus absent (READ-ONLY inputs not on this machine)",
)
def test_real_seed44_dir_decider_accepts_clean_boundary_dry_run() -> None:
    """F3/round-4: the UNSTUBBED decider ACCEPTS the real seed-44 dir, writing nothing.

    Drives the REAL ``decide`` (real ``_git_ancestor`` via git.exe, real config
    sha, real checkpoint decode, real F1 witness) with the arm's frozen commit
    (this worktree HEAD) and ``ORIGIN_COMMIT=8ed5edb``; asserts it selects the
    newest complete-epoch checkpoint and, being a dry run, leaves the run dir
    byte-identical.  The run dir is READ-ONLY: nothing here writes to it.

    STATE-INDEPENDENT (round-5 BLOCKER): the previous version hard-pinned
    ``epoch_150.pth``, which the first resume this amendment authorizes
    invalidates — the resumed segment writes ``epoch_160.pth`` (the next periodic
    after stored epoch 159), which is then the newest complete-epoch checkpoint
    and the correct source.  The assertion accepts ``epoch_150.pth`` (pre-resume)
    or ``epoch_160.pth`` (post-resume), so it holds in either state; the
    byte-identical check is valid in both.
    """
    import subprocess

    mod = _helper()
    cfg = Path(__file__).resolve().parents[1] / (
        "config/realdata-phase2-20261010/R0-control-seed44.yaml"
    )
    wt = str(Path(__file__).resolve().parents[1])
    # The worktree's ``.git`` file points at a Windows gitdir the WSL git cannot
    # read; resolve HEAD via git.exe on the wslpath -w form, exactly as the
    # module's own ``_git_ancestor`` does.
    frozen = subprocess.run(
        ["git.exe", "-C", mod._to_windows_path(wt), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
              for p in _REAL_SEED44.iterdir()}
    err = StringIO()
    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = mod.decide(
            str(_REAL_SEED44), wt, frozen, str(cfg), "300", _REAL_ORIGIN,
            dry_run=True,
        )
    assert rc == 0, f"stdout={buf.getvalue()!r} stderr={err.getvalue()!r}"
    selected = buf.getvalue().strip()
    expected = "epoch_160.pth" if (_REAL_SEED44 / "epoch_160.pth").exists() else "epoch_150.pth"
    assert selected.endswith(expected), (
        f"decider selected {selected!r}, expected the newest complete-epoch "
        f"checkpoint {expected!r}"
    )
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
             for p in _REAL_SEED44.iterdir()}
    assert after == before, "the dry-run decider must not write to the run dir"

