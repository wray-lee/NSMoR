"""Resume-decision helper for run_frozen_resume.sh.

Two subcommands, both CPU-only and side-effect-free except where noted:

  decide <out_dir> <worktree> <frozen_commit> <config_path> <num_epochs>
      Print the checkpoint path to resume from (exit 0), or an explanation to
      stdout with exit 1 when the run must be REFUSED.  A non-empty output dir
      resumes ONLY when a recoverable current-schema checkpoint exists AND its
      provenance matches: same frozen commit, a non-empty ``resolved_modules``
      recorded entirely inside the worktree, and the config's current SHA-256
      equal to the recorded one.  ``best_model.pth`` is NEVER a resume source
      (it is the best epoch, which may be far older than the last executed
      epoch); only the last-epoch families (``final_model.pth`` /
      ``epoch_*.pth``) qualify, and a dir holding only ``best_model.pth`` is
      refused.

  record <out_dir> <checkpoint> <head>
      Append {time, checkpoint, checkpoint_epoch, source, head} to the run's
      provenance.json ``resume_log`` (creating it if absent) and print one
      status.log line.  This is the only writing subcommand.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

RECOVERY_STATE_VERSION = 2


def _load_checkpoint(path: Path) -> dict:
    from nsmor.pipeline.nested_prior import load_artifact_bytes

    return load_artifact_bytes(path.read_bytes(), map_location="cpu")


def _tie_break(name: str) -> tuple:
    """Deterministic tie-break when two files store the SAME epoch.

    ``final_model.pth`` is the segment's last executed epoch and wins over the
    periodic copy of the same epoch; an ``epoch_<n>.pth`` is the periodic.
    ``best_model.pth`` never reaches here: it is NOT a resume source (it is the
    best epoch, which may be far OLDER than the last executed epoch, so resuming
    from it would re-train completed epochs).
    """
    if name == "final_model.pth":
        return (0, 0)
    m = re.match(r"epoch_(\d+)\.pth$", name)
    if m:
        return (1, int(m.group(1)))
    return (2, 0)


# ``best_model.pth`` is the BEST-selected epoch, which can be far OLDER than the
# last executed epoch.  Resuming from it would silently RE-TRAIN every completed
# epoch after the best one and break §11.1's 'last complete-epoch checkpoint'
# guarantee, so it is never a resume candidate — only the last-epoch families
# (``final_model.pth`` / ``epoch_*.pth``) are.
_RESUME_EXCLUDED = frozenset({"best_model.pth"})


def _resume_candidates(out: Path) -> list[tuple[Path, dict | None, str]]:
    """Recovery candidates ordered by the checkpoint's OWN stored epoch, newest first.

    Only the LAST-EPOCH families are candidates: ``final_model.pth`` and
    ``epoch_*.pth``.  ``best_model.pth`` is EXCLUDED — it stores the best
    SELECTED epoch, which can be far older than the last executed one, so
    resuming from it would silently re-train completed epochs (it is not the
    run's last complete-epoch checkpoint).  When only ``best_model.pth`` exists
    there is NO recoverable resume source and the caller REFUSES.

    Ordering must NOT depend on the file FAMILY.  With ``checkpoint_interval:
    10`` a safe pause or an OOM/WSL kill at a NON-multiple epoch (e.g. 12)
    writes only ``final_model.pth`` (epoch 12), while the newest periodic is
    ``epoch_10.pth`` (stored epoch 9).  A family-first ordering would resume
    from epoch 9 and silently RE-TRAIN epochs 10-12, breaking §11.1's 'last
    complete-epoch checkpoint' guarantee.  Ordering by the stored epoch across
    BOTH families makes ``final_model.pth`` (12) win over ``epoch_10.pth`` (9).

    A file whose epoch cannot be decoded (truncated / wrong schema) is kept
    LAST so the caller still attempts it and reports its decode error rather
    than dropping it silently.  Returns ``(path, state|None, error)`` triples.
    """
    triples: list[tuple[Path, dict | None, str]] = []
    for p in sorted(out.glob("*.pth")):
        if p.name in _RESUME_EXCLUDED:
            continue
        try:
            triples.append((p, _load_checkpoint(p), ""))
        except Exception as exc:  # noqa: BLE001 - fail closed on any decode error
            triples.append((p, None, repr(exc)))

    def _key(item: tuple[Path, dict | None, str]) -> tuple:
        path, state, _ = item
        raw = state.get("epoch") if state is not None else None
        readable = isinstance(raw, int) and not isinstance(raw, bool)
        return (
            0 if readable else 1,
            -(raw if readable else 0),
            _tie_break(path.name),
        )

    return sorted(triples, key=_key)


def _active_phase2_controls(config_path: Path) -> tuple:
    """The active arm's script-owned phase-2 controls, as (selection_metric, zic).

    Mirrors ``train.py``'s own resume guard so the launcher can refuse a
    control-switching resume BEFORE launching, instead of starting a process
    that fails closed only inside ``train()``.
    """
    import yaml

    data = yaml.safe_load(config_path.read_text()) or {}
    phase2 = data.get("phase2") or {}
    selection_metric = str(phase2.get("selection_metric", "total"))
    zic = tuple(sorted(int(c) for c in (phase2.get("zero_input_channels") or [])))
    return selection_metric, zic


def decide(out_dir: str, worktree: str, frozen: str, config_path: str,
           num_epochs: str) -> int:
    out = Path(out_dir)
    cfg = Path(config_path)
    n_epochs = int(num_epochs)

    prov_path = out / "provenance.json"
    if not prov_path.exists():
        print("no provenance.json in non-empty output dir")
        return 1
    prov = json.loads(prov_path.read_text())

    if str(prov.get("head")) != frozen:
        print(f"provenance head {prov.get('head')!r} != frozen {frozen!r} "
              "(cross-code-version resume refused)")
        return 1
    # The recorded module paths are what binds the run's code to THIS worktree.
    # An empty / absent / whitespace-only field would make the loop below a
    # no-op and let an untrusted or cross-worktree run resume as if verified —
    # a FAIL-OPEN.  Refuse unless the field resolves to at least one path, all
    # inside the worktree.
    modules = str(prov.get("resolved_modules", "")).split()
    if not modules:
        print("provenance resolved_modules is empty/absent (cannot verify the "
              "run's modules resolved inside the worktree; refusing fail-closed)")
        return 1
    for mod in modules:
        if not mod.startswith(worktree + "/"):
            print(f"recorded module {mod!r} is outside the worktree")
            return 1
    recorded = prov.get("sha256", {})
    cfg_sha = hashlib.sha256(cfg.read_bytes()).hexdigest()
    if recorded.get(str(cfg)) != cfg_sha:
        print(f"config sha mismatch: provenance {recorded.get(str(cfg))!r} "
              f"vs current {cfg_sha!r}")
        return 1

    active_sel, active_zic = _active_phase2_controls(cfg)

    cands = _resume_candidates(out)
    if not cands:
        print("no recoverable checkpoint (no epoch_*.pth / final_model.pth; "
              "best_model.pth is the best epoch, not the last, and is never a "
              "resume source)")
        return 1

    # Validate every candidate (newest stored epoch first) and resume from the
    # highest-epoch one that is a valid, non-exhausted current-schema payload.
    # Iterating means a truncated or wrong-schema newest checkpoint falls
    # through to the next-oldest complete-epoch checkpoint instead of refusing
    # a run that is in fact resumable.
    ckpt = None
    state = None
    reason = ""
    for cand, st, err in cands:
        if st is None:
            reason = f"checkpoint {cand.name} is not loadable: {err}"
            continue
        if st.get("recovery_state_version") != RECOVERY_STATE_VERSION:
            reason = (
                f"checkpoint {cand.name} is not a current-schema recovery "
                f"payload (recovery_state_version="
                f"{st.get('recovery_state_version')!r})"
            )
            continue
        epoch = st.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            reason = f"checkpoint {cand.name} has a non-integer epoch {epoch!r}"
            continue
        if epoch + 1 >= n_epochs:
            # The NEWEST complete-epoch checkpoint already exhausted the
            # budget, so the run is DONE.  This is TERMINAL, not a reason to
            # fall through to an older periodic: resuming an older one would
            # silently re-train already-completed epochs and exceed the
            # budget.  Refuse.
            print(
                f"newest complete-epoch checkpoint {cand.name} at epoch "
                f"{epoch!r} has no remaining epochs (num_epochs={n_epochs}); "
                "run already complete"
            )
            return 1
        # Cross-check the script-owned phase-2 controls against the active arm.
        # A mismatch would fail closed only inside train(); refusing here lets
        # the launcher decline before spawning a doomed process.
        rec_sel = st.get("selection_metric", "total")
        rec_zic = tuple(sorted(int(c) for c in st.get("zero_input_channels", [])))
        if rec_sel != active_sel or rec_zic != active_zic:
            reason = (
                f"checkpoint {cand.name} phase-2 controls "
                f"selection_metric={rec_sel!r}/zero_input_channels={list(rec_zic)} "
                f"!= active {active_sel!r}/{list(active_zic)}"
            )
            continue
        ckpt, state = cand, st
        break
    if ckpt is None:
        print(reason or "no recoverable checkpoint")
        return 1

    # Non-nested historical checkpoints additionally require an explicit
    # trusted SHA pin, exactly as ``train.py``'s resume preflight does.  The
    # phase-2 corpus is nested (``is_nested_cv=True``) and is unaffected, but a
    # generic non-nested arm would otherwise pass this preflight and then fail
    # closed only at train() time — after the checkpoint was declared
    # resumable.  Surface the requirement here so the launcher refuses early
    # rather than half-starting a run it cannot resume.
    if (state.get("animal_identity_status", "historical_unknown")
            == "historical_unknown"
            and not state.get("is_nested_cv", False)):
        print(
            "checkpoint is a non-nested historical payload requiring an "
            "explicit trusted_historical_checkpoint_sha256; the launcher does "
            "not pass one, so the resume would fail closed at train() time"
        )
        return 1

    print(str(ckpt))
    return 0


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write *payload* to *path* via temp file + fsync + atomic rename.

    Mirrors the checkpoint writer: a crash mid-write can never leave a
    truncated ``provenance.json`` (the file a resume reads to decide).
    """
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    try:
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:  # pragma: no cover - filesystems without fsync
        pass
    os.replace(tmp, path)


def record(out_dir: str, checkpoint: str, head: str) -> int:
    out = Path(out_dir)
    ckpt = Path(checkpoint)
    prov_path = out / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    epoch = _load_checkpoint(ckpt).get("epoch")
    entry = {
        "time": datetime.now(timezone.utc).astimezone().isoformat(),
        "checkpoint": str(ckpt),
        "checkpoint_epoch": int(epoch),
        "source": "run_frozen_resume.sh",
        "head": head,
    }
    prov.setdefault("resume_log", []).append(entry)
    _atomic_write_json(prov_path, prov)
    print(f"{entry['time']} {out.name} RESUME from {ckpt.name} "
          f"(epoch {epoch}) source={entry['source']}")
    return 0


def main(argv: list) -> int:
    if not argv:
        print("usage: resume_decision.py {decide|record} ...")
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd == "decide":
        return decide(*args)
    if cmd == "record":
        return record(*args)
    print(f"unknown subcommand {cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
