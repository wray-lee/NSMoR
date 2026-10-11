"""Resume-decision helper for run_frozen_resume.sh.

Two subcommands, both CPU-only and side-effect-free except where noted:

  decide <out_dir> <worktree> <frozen_commit> <config_path> <num_epochs> \\
         [origin_commit] [--dry-run]
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

      F1 no-silent-rollback (§11.1, 2026-10-11): the selected source must not
      be OLDER than the last epoch the run actually EXECUTED, as witnessed by
      the run's own records (the last ``Epoch N/…`` line in its progress logs,
      each segment record's ``starting_epoch`` read as ``start - 1``, its
      last-epoch checkpoints).  A run killed before writing a newer last-epoch
      file would otherwise silently re-train the completed epochs in between;
      the refusal names both epochs.  No override flag.  ``scripts/train.py``
      enforces the SAME rule — same witness set — in its own resume preflight.

      SUCCESSOR-COMMIT RESUME (protocol amendment §11.1, 2026-10-11): a run
      started on commit C may be resumed on a DIFFERENT commit C' only when the
      opt-in ``origin_commit`` (the launcher's ``ORIGIN_COMMIT=<C>``) is given
      AND git confirms C is an ancestor of the frozen commit C'.  The recorded
      modules and config are then bound to C's worktree (the run's
      ``code_worktree``) and re-rooted, so a head mismatch is accepted ONLY on
      that proven ancestry.  Without the opt-in a head != frozen is still
      refused as a cross-code-version resume (fail closed).

  record <out_dir> <checkpoint> <head> [origin_commit]
      Append {time, checkpoint, checkpoint_epoch, source, head, origin_commit}
      to the run's provenance.json ``resume_log`` (creating it if absent) and
      print one status.log line.  This is the only writing subcommand.  When a
      successor-commit resume is used BOTH commits (origin C and the frozen C')
      are recorded.

``--dry-run`` is accepted on ``decide`` and is a no-op on the filesystem: it
prints the chosen source and the checks passed to stderr and still prints only
the checkpoint path to stdout, so a dry run writes NOTHING to the run dir.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RECOVERY_STATE_VERSION = 2


def _to_windows_path(path: str) -> str:
    """Translate a WSL path to a Windows path for ``git.exe``.

    The worktrees' ``.git`` files point at Windows paths
    (``gitdir: C:/disk/...``), so the WSL ``git`` cannot resolve them; the
    launcher pattern is ``git.exe`` on the ``wslpath -w`` form (mirroring
    ``run_frozen2.sh``'s ``git.exe`` + ``tr -d '\\r'``).
    """
    out = subprocess.run(
        ["wslpath", "-w", path], capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _git_ancestor(worktree: str, ancestor: str, descendant: str) -> bool:
    """True iff *ancestor* is an ancestor of *descendant* in *worktree*'s repo.

    Uses ``git.exe -C <windows path> merge-base --is-ancestor`` (the WSL ``git``
    cannot read the worktree's Windows ``.git``).  Exit 0 = ancestor, exit 1 =
    not an ancestor; any other status — or a missing ``git.exe`` / ``wslpath`` —
    is an infrastructure failure and raises so the caller fails closed rather
    than guessing.
    """
    try:
        win = _to_windows_path(worktree)
        proc = subprocess.run(
            ["git.exe", "-C", win, "merge-base", "--is-ancestor",
             ancestor, descendant],
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"git ancestry check could not run: {exc}") from exc
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    raise RuntimeError(
        f"git ancestry check failed (rc={proc.returncode}): "
        f"{proc.stderr.strip() or proc.stdout.strip()}"
    )


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

# The LAST-EPOCH checkpoint families: the segment ``final_model.pth`` and the
# periodic ``epoch_<n>.pth``.  Used to witness the run's last executed epoch
# (F1).  (Resume CANDIDATES are ordered by stored epoch across every non-best
# ``*.pth``; this set only bounds the F1 witness.)
_EPOCH_PERIODIC_RE = re.compile(r"epoch_(\d+)\.pth$")

# The last "Epoch N/<budget>" progress line in a run's log, read only as a
# SECOND F1 witness to how far the run actually got.  IDENTICAL to
# ``scripts/train.py::_EPOCH_PROGRESS_RE`` — the two enforcers are the SAME rule,
# so this literal must not drift (the drift-pin test cross-checks the two).  See
# train.py's ``_EPOCH_PROGRESS_RE`` for the fail-closed pre-epoch-banner
# assumption (it also matches the ``lambda_routing_aux`` banner; MAX keeps the
# witness fail-CLOSED, never laxer).
_EPOCH_PROGRESS_RE = re.compile(r"\bEpoch\s+(\d+)\s*/")

# The checkpoint files that witness an EXECUTED epoch (F1).  IDENTICAL to
# ``scripts/train.py::_CHECKPOINT_WITNESS_FAMILIES`` — the two enforcers are the
# SAME rule, so their witness sets must not drift (a divergence is a latent hole
# even when a sibling check masks it today).  ``final_model.pth`` IS written at
# the end of a completed/paused run with that run's last executed epoch, so it IS
# a valid executed-epoch witness; ``best_model.pth`` stores the best SELECTED
# epoch (possibly far older) and is NEVER a witness.  Keep the two literals in
# lockstep; the F1 regression suite pins both enforcers.
_CHECKPOINT_WITNESS_FAMILIES = ("final_model.pth",)


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


def _last_executed_epoch(out: Path) -> int | None:
    """The highest 0-based epoch the run's OWN RECORDS witness as executed.

    F1 no-silent-rollback (§11.1, 2026-10-11): a resume whose selected source is
    OLDER than the last epoch the run actually executed would silently re-train
    the completed epochs in between.  The newest complete-epoch checkpoint does
    not always reveal that (a run killed before writing one leaves only an older
    periodic), so this reads the run's records directly:

    * the progress logs — every ``train.*.log`` in the run dir, taking the last
      ``Epoch N/<budget>`` line (N is 1-based);
    * every ``segment_XXXX.json``'s ``starting_epoch``, read as ``start - 1``:
      the writer stores ``stored_epoch + 1`` (the NEXT epoch to execute, before
      the loop), so a segment at ``starting_epoch`` S witnesses only epoch S-1 —
      the epoch before its first (F1 round-4 BLOCKER);
    * the last-epoch family checkpoints (``final_model.pth`` / ``epoch_*.pth``).

    ``best_model.pth`` is EXCLUDED (it is the best SELECTED epoch, which may be
    far older, so it is not a witness).  The MAX is taken so the rule can only
    be STRICTER.  ``None`` when no record exists — the caller then leaves the
    resume untouched.
    """
    witnesses: list[int] = []
    for name in sorted(p.name for p in out.glob("train.*.log")):
        try:
            text = (out / name).read_text(errors="ignore")
        except OSError:
            continue
        for m in _EPOCH_PROGRESS_RE.finditer(text):
            witnesses.append(int(m.group(1)) - 1)
    for rec in sorted(out.glob("segment_*.json")):
        try:
            data = json.loads(rec.read_text())
        except (OSError, ValueError):
            continue
        start = data.get("starting_epoch")
        if isinstance(start, int) and not isinstance(start, bool):
            # ``_write_segment_record`` stores ``starting_epoch = stored_epoch + 1``
            # — the NEXT epoch to execute, written BEFORE the loop — so it
            # witnesses execution of ``start - 1``, NOT ``start``.  Reading it
            # RAW over-claims by one and would refuse the clean-boundary recovery
            # of an epoch that was only STARTED (never completed), bricking the
            # seed-44 resume (F1 round-4 BLOCKER).
            witnesses.append(start - 1)
    for cand in sorted(out.glob("*.pth")):
        if cand.name in _RESUME_EXCLUDED:
            continue
        if (
            cand.name not in _CHECKPOINT_WITNESS_FAMILIES
            and not _EPOCH_PERIODIC_RE.match(cand.name)
        ):
            continue
        try:
            state = _load_checkpoint(cand)
        except Exception:  # noqa: BLE001 - an undecodable record is not evidence
            continue
        epoch = state.get("epoch")
        if isinstance(epoch, int) and not isinstance(epoch, bool):
            witnesses.append(epoch)
    return max(witnesses) if witnesses else None


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
           num_epochs: str, origin_commit: str | None = None,
           dry_run: bool = False) -> int:
    out = Path(out_dir)
    cfg = Path(config_path)
    n_epochs = int(num_epochs)

    def _report(line: str) -> None:
        if dry_run:
            print(f"[dry-run] {line}", file=sys.stderr)

    prov_path = out / "provenance.json"
    if not prov_path.exists():
        print("no provenance.json in non-empty output dir")
        return 1
    prov = json.loads(prov_path.read_text())

    recorded_head = str(prov.get("head"))
    # ── Successor-commit resume gate (protocol §11.1, 2026-10-11) ──
    # A head != frozen is a cross-code-version resume and is refused UNLESS the
    # caller opts in with the run's ORIGIN_COMMIT C and git confirms C is an
    # ancestor of the frozen commit C'.  The recorded modules/config are then
    # bound to C's worktree and re-rooted onto the C' worktree.
    if recorded_head != frozen:
        if not origin_commit:
            print(f"provenance head {recorded_head!r} != frozen {frozen!r} "
                  "(cross-code-version resume refused; pass ORIGIN_COMMIT=<C> "
                  "to opt in to a proven successor-commit resume)")
            return 1
        if recorded_head != origin_commit:
            print(f"provenance head {recorded_head!r} != ORIGIN_COMMIT "
                  f"{origin_commit!r} (the opt-in names a different origin)")
            return 1
        try:
            is_ancestor = _git_ancestor(worktree, origin_commit, frozen)
        except RuntimeError as exc:  # noqa: BLE001 - fail closed on git failure
            print(f"git ancestry check could not be performed: {exc}")
            return 1
        if not is_ancestor:
            print(f"ORIGIN_COMMIT {origin_commit!r} is not an ancestor of the "
                  f"frozen commit {frozen!r} (successor-commit resume refused)")
            return 1
        origin_wt = str(prov.get("code_worktree", "")).rstrip("/")
        if not origin_wt:
            print("provenance records no code_worktree to re-root the origin "
                  "config/modules onto (successor-commit resume refused)")
            return 1
        # Re-root the config lookup: the run's provenance keys the config by its
        # path inside C's worktree, but the active YAML lives in C''s worktree.
        try:
            rel_cfg = cfg.resolve().relative_to(Path(worktree).resolve())
        except ValueError:
            print(f"config {str(cfg)!r} is not inside the worktree "
                  f"{worktree!r} (cannot re-root the origin config)")
            return 1
        cfg_lookup = str(Path(origin_wt) / rel_cfg)
        module_root = origin_wt
        _report(
            f"successor-commit resume: origin {origin_commit} is an ancestor of "
            f"frozen {frozen}; modules re-rooted to {origin_wt}"
        )
    else:
        cfg_lookup = str(cfg)
        module_root = worktree

    # The recorded module paths are what binds the run's code to a worktree.
    # An empty / absent / whitespace-only field would make the loop below a
    # no-op and let an untrusted or cross-worktree run resume as if verified —
    # a FAIL-OPEN.  Refuse unless the field resolves to at least one path, all
    # inside the origin worktree.
    modules = str(prov.get("resolved_modules", "")).split()
    if not modules:
        print("provenance resolved_modules is empty/absent (cannot verify the "
              "run's modules resolved inside the worktree; refusing fail-closed)")
        return 1
    for mod in modules:
        if not mod.startswith(module_root + "/"):
            print(f"recorded module {mod!r} is outside the worktree "
                  f"{module_root!r}")
            return 1
    recorded = prov.get("sha256", {})
    cfg_sha = hashlib.sha256(cfg.read_bytes()).hexdigest()
    if recorded.get(cfg_lookup) != cfg_sha:
        print(f"config sha mismatch: provenance {recorded.get(cfg_lookup)!r} "
              f"vs current {cfg_sha!r} (key {cfg_lookup!r})")
        return 1
    _report("config sha256 matches the recorded origin config")

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

    # F1 no-silent-rollback (§11.1, 2026-10-11).  The newest-complete-epoch rule
    # above returns the newest checkpoint that EXISTS, but a run killed before
    # writing a newer last-epoch file (reached epoch 58 while the newest
    # periodic is epoch_50.pth) leaves the rule selecting an OLDER source — a
    # silent re-train of the completed epochs in between.  The run's own records
    # (progress logs / segment records / last-epoch checkpoints) witness how far
    # it actually got; refuse when the selected source is older.  No override.
    last_executed = _last_executed_epoch(out)
    if last_executed is not None and state.get("epoch") < last_executed:
        print(
            f"selected source {ckpt.name} (stored epoch {state.get('epoch')}) is "
            f"older than the last executed epoch {last_executed} evidenced by the "
            "run's records (train.*.log / segment_*.json / last-epoch "
            f"checkpoints): resuming would silently re-train epochs "
            f"{state.get('epoch') + 1}..{last_executed} (no-silent-rollback, "
            "§11.1; no override; refusing)"
        )
        return 1

    _report(
        f"chosen source {ckpt.name} (stored epoch {state.get('epoch')}, next "
        f"epoch {int(state.get('epoch')) + 1} of {n_epochs}); controls and "
        f"config sha verified; source is the newest complete-epoch checkpoint"
    )
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


def record(out_dir: str, checkpoint: str, head: str,
           origin_commit: str | None = None) -> int:
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
        # ``head`` is the frozen commit C' the resume RAN on; ``origin_commit``
        # is the run's own starting commit C (recorded only for a
        # successor-commit resume, else null).  Both are recorded so a run
        # resumed across a code version names both commits (§11.1).
        "head": head,
        "origin_commit": origin_commit,
    }
    prov.setdefault("resume_log", []).append(entry)
    _atomic_write_json(prov_path, prov)
    origin_note = f" origin={origin_commit}" if origin_commit else ""
    print(f"{entry['time']} {out.name} RESUME from {ckpt.name} "
          f"(epoch {epoch}) source={entry['source']} head={head}{origin_note}")
    return 0


def main(argv: list) -> int:
    if not argv:
        print("usage: resume_decision.py {decide|record} ...")
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd == "decide":
        dry_run = "--dry-run" in args
        args = [a for a in args if a != "--dry-run"]
        return decide(*args, dry_run=dry_run)
    if cmd == "record":
        return record(*args)
    print(f"unknown subcommand {cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
