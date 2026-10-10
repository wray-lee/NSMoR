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
from contextlib import redirect_stdout
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
    assert "time" in entry
    # Atomic write leaves no temp file behind.
    assert not (out / "provenance.json.tmp").exists()


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
