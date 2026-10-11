"""Shared disclosed-rollback + abandoned-branch archiving for phase-2 resume.

Both resume enforcers — the launcher decider (``scripts/resume_decision.py``) and
``scripts/train.py``'s resume preflight — must agree on TWO things, so the logic
lives here ONCE and cannot drift.  This module lives under ``nsmor/pipeline/``
(not ``scripts/``) because ``scripts`` is not an installed package, so a
``scripts/*.py`` file may not import it.

* **How far a run actually EXECUTED** (``last_executed_epoch``).  A run's own
  records witness the last epoch it executed: the last ``Epoch N/<budget>``
  progress line in its logs (N is 1-based, so it witnesses ``N - 1``), every
  segment record's ``starting_epoch`` read as ``start - 1`` (the writer stores the
  NEXT epoch to execute), and every checkpoint's own stored epoch —
  ``best_model.pth`` (the best SELECTED epoch), ``final_model.pth`` (the segment's
  last epoch) and ``epoch_<n>.pth`` (a periodic) were each written only AFTER
  executing the epoch they store.  The MAX is taken so the rule can only be
  stricter, never laxer.

* **A disclosed rollback** (protocol §11.1, amendment 2026-10-11).  When the
  resume source's stored epoch is OLDER than ``last_executed_epoch``, the resume
  is ALLOWED — the user's rule is "resume from the LAST COMPLETE-EPOCH
  CHECKPOINT", and with ``checkpoint_interval: 10`` a crash between periodics
  always loses up to 9 completed epochs, so refusing the rollback would make
  resume unusable.  What must NEVER happen is a SILENT rollback or an OVERWRITE of
  artifacts the abandoned (un-checkpointed) branch produced.  So the re-trained
  range ``[source + 1, last_executed]`` is DISCLOSED (recorded + printed), and
  every abandoned-branch artifact that the resume would overwrite is MOVED aside
  (never deleted, never overwritten) into ``run_dir/abandoned_branch_<k>/`` with a
  manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The LAST-EPOCH checkpoint families: ``final_model.pth`` (the segment's last
# executed epoch) and ``epoch_<n>.pth`` (a periodic).
_EPOCH_PERIODIC_RE = re.compile(r"epoch_(\d+)\.pth$")

# The checkpoint files that witness an EXECUTED epoch.  ``best_model.pth`` IS a
# witness (it stores the best SELECTED epoch, which the run executed) AND
# ``final_model.pth`` IS a witness (the segment's last executed epoch); a
# periodic ``epoch_<n>.pth`` is matched by ``_EPOCH_PERIODIC_RE``.  A checkpoint's
# stored epoch means the run executed at least through that epoch.
_CHECKPOINT_WITNESS_FAMILIES = ("best_model.pth", "final_model.pth")

# ``best_model.pth`` is the best SELECTED epoch: it IS an executed-epoch witness
# but it is NEVER a RESUME SOURCE (resuming from a far-older best would re-train
# completed epochs).  No witness family is excluded any more.
_NON_RESUME_WITNESS_FAMILIES: frozenset = frozenset()

# The last "Epoch N/<budget>" progress line, read only as a SECOND witness to how
# far a run actually got.  ASSUMPTION (fail-closed): it matches ANY ``Epoch N/…``
# banner, including the pre-epoch ``lambda_routing_aux`` line (emitted BEFORE the
# epoch's updates; inert for the phase-2 arms, all of which set it to 0.0).  The
# MAX keeps the witness fail-CLOSED; the regex is deliberately NOT narrowed to the
# completion line's ``train_loss=`` suffix, which would fail OPEN if it changed.
_EPOCH_PROGRESS_RE = re.compile(r"\bEpoch\s+(\d+)\s*/")

# The abandoned-branch artifact FAMILIES that are moved aside on a disclosed
# rollback (§11.1 D2 names "at least ``best_model.pth`` and
# ``final_model.pth``").  A periodic ``epoch_<n>.pth`` is deliberately NOT
# archived: the resume source is always the run's NEWEST complete-epoch
# checkpoint (the preflight refuses a source older than any existing newer
# last-epoch file), so no periodic and no ``final_model.pth`` can carry a stored
# epoch greater than the source's, and any periodic the resumed run re-trains is
# regenerated at the SAME epoch — moving it aside would discard a valid
# post-rollback recovery point without preventing an overwrite the resumed run
# does not itself perform.  ``*.pth`` only: the run's logs and ``segment_*.json``
# records are NEVER truncated or moved (§11.1 condition 4).
ARCHIVE_DIR_PREFIX = "abandoned_branch_"
_MANIFEST_NAME = "manifest.json"
_ARCHIVE_FAMILIES = ("best_model.pth", "final_model.pth")


def _stored_epoch(path: Path) -> Optional[int]:
    """The checkpoint's own stored ``epoch`` (0-based), or ``None`` if absent.

    ``None`` covers a truncated / wrong-schema / epoch-less payload: an
    undecodable record carries no evidence rather than a guess.
    """
    from nsmor.pipeline.nested_prior import load_artifact_bytes

    try:
        state = load_artifact_bytes(path.read_bytes(), map_location="cpu")
    except Exception:  # noqa: BLE001 - an undecodable record is not evidence
        return None
    epoch = state.get("epoch")
    return epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else None


def last_executed_epoch(output_dir: Path) -> Optional[int]:
    """The highest 0-based epoch the run is evidenced to have EXECUTED.

    See the module docstring for the three record families and the fail-closed
    MAX.  ``None`` when NO record exists (a fresh dir, a test fixture, a renamed
    artifact) — the caller then leaves the resume untouched.
    """
    output_dir = Path(output_dir)
    witnesses: List[int] = []
    for name in sorted(p.name for p in output_dir.glob("train.*.log")):
        try:
            text = (output_dir / name).read_text(errors="ignore")
        except OSError:
            continue
        for m in _EPOCH_PROGRESS_RE.finditer(text):
            witnesses.append(int(m.group(1)) - 1)  # "Epoch N" is 1-based
    for rec in sorted(output_dir.glob("segment_*.json")):
        try:
            data = json.loads(rec.read_text())
        except (OSError, ValueError):
            continue
        start = data.get("starting_epoch")
        if isinstance(start, int) and not isinstance(start, bool):
            # ``_write_segment_record`` stores ``stored_epoch + 1`` — the NEXT
            # epoch to execute, written BEFORE the loop — so it witnesses epoch
            # ``start - 1``, not ``start``.
            witnesses.append(start - 1)
    for cand in sorted(output_dir.glob("*.pth")):
        if cand.name in _NON_RESUME_WITNESS_FAMILIES:
            continue
        if (
            cand.name not in _CHECKPOINT_WITNESS_FAMILIES
            and not _EPOCH_PERIODIC_RE.match(cand.name)
        ):
            continue
        epoch = _stored_epoch(cand)
        if epoch is not None:
            witnesses.append(epoch)
    return max(witnesses) if witnesses else None


def rollback_retrained_range(
    source_epoch: int, last_executed: Optional[int],
) -> Optional[Tuple[int, int]]:
    """The DISCLOSED re-trained epoch range ``[source + 1, last_executed]``.

    ``None`` when the source is the last executed epoch (a clean boundary, nothing
    to disclose) or when there is no record at all.  The range is INCLUSIVE and
    0-based, matching the checkpoints' own ``epoch`` field.
    """
    if last_executed is None:
        return None
    if source_epoch >= last_executed:
        return None
    return (source_epoch + 1, last_executed)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _next_archive_dir(output_dir: Path) -> Path:
    """The ``abandoned_branch_<k>`` dir to move into; refuse on a collision.

    *k* is the number of existing ``abandoned_branch_*`` directories, so the
    normal sequential case (0, 1, 2, …) advances cleanly.  If the computed path
    ALREADY exists (a non-contiguous or raced archive set) the move would overwrite
    an existing archive path, so this REFUSES (raises ``FileExistsError``) rather
    than clobbering it — the explicit "never overwrite an existing archive path"
    rule (§11.1 D2).
    """
    k = sum(1 for p in output_dir.glob(f"{ARCHIVE_DIR_PREFIX}*") if p.is_dir())
    candidate = output_dir / f"{ARCHIVE_DIR_PREFIX}{k}"
    if candidate.exists():
        raise FileExistsError(
            f"refusing to archive into existing path {candidate}; the "
            "abandoned-branch archive would overwrite an existing archive path "
            "(fail closed)"
        )
    return candidate


def list_abandoned_branch(
    output_dir: Path, source_epoch: int,
) -> List[Path]:
    """The run-dir ``*.pth`` files that belong to the ABANDONED branch.

    An abandoned-branch artifact is a run-dir ``*.pth`` in the ARCHIVED FAMILIES
    (``best_model.pth`` / ``final_model.pth`` — §11.1 D2) whose OWN stored epoch
    is GREATER than *source_epoch* — the run wrote it while continuing past the
    checkpoint we are rolling back to, so the resumed run would overwrite it.  A
    periodic ``epoch_<n>.pth`` is NOT an abandoned-branch artifact (it is
    regenerated at the same epoch by the resumed run; moving it aside would
    discard a valid recovery point without preventing an overwrite).  Pure: never
    moves or writes.  Logs and ``segment_*.json`` records are never included
    (§11.1 condition 4).
    """
    output_dir = Path(output_dir)
    out: List[Path] = []
    for cand in sorted(output_dir.glob("*.pth")):
        if cand.name not in _ARCHIVE_FAMILIES:
            continue
        epoch = _stored_epoch(cand)
        if epoch is not None and epoch > source_epoch:
            out.append(cand)
    return out


def has_best_model_archive(output_dir: Path) -> bool:
    """True iff some ``abandoned_branch_*/`` holds a ``best_model.pth``.

    The disclosed-rollback reconciliation tolerates the deliberately-absent
    companion best ONLY when the run's best was archived to
    ``abandoned_branch_*/`` on a prior invocation (this one OR an earlier one).
    The archive step moves the abandoned best OUT of the run dir, and the resumed
    segment can legitimately fail to produce a new best (its val never beats the
    seeded ``best_val_loss``), so keying the tolerance on THIS invocation's own
    rollback made a chained SECOND resume fail closed.  Pure: reads only.
    """
    output_dir = Path(output_dir)
    return any(
        (d / "best_model.pth").exists()
        for d in output_dir.glob(f"{ARCHIVE_DIR_PREFIX}*")
        if d.is_dir()
    )


def launcher_already_recorded(
    provenance: Dict[str, Any], checkpoint: str, rollback: Tuple[int, int],
) -> bool:
    """True iff ``resume_log`` already holds the LAUNCHER's entry for this resume.

    On the primary launcher path ``scripts/run_frozen_resume.sh`` calls
    ``resume_decision.py record`` (which appends a ``disclosed_rollback`` entry)
    and THEN launches ``train.py --resume`` (which would append a SECOND entry for
    the same resume — the round-7 double-record blocker).  train.py consults this
    to record ONLY when the launcher's entry for THIS checkpoint + range is
    absent, so a bare ``train.py --resume`` (which bypasses the launcher) still
    records while the launcher path records exactly once.  Pure: reads only.
    """
    lo, hi = int(rollback[0]), int(rollback[1])
    try:
        want = Path(checkpoint).resolve()
    except OSError:  # pragma: no cover - an unresolvable path compares by string
        want = None
    for entry in provenance.get("resume_log", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("source") != "run_frozen_resume.sh":
            continue
        rec = str(entry.get("checkpoint"))
        same_path = rec == str(checkpoint)
        if not same_path and want is not None:
            try:
                same_path = Path(rec).resolve() == want
            except OSError:  # pragma: no cover
                same_path = False
        if not same_path:
            continue
        if entry.get("disclosed_rollback") == [lo, hi]:
            return True
    return False


def archive_abandoned_branch(
    output_dir: Path,
    *,
    source_epoch: int,
    last_executed: int,
    source_name: str,
) -> List[Dict[str, Any]]:
    """MOVE every abandoned-branch artifact aside; return the manifest entries.

    Each abandoned-branch artifact (see :func:`list_abandoned_branch`) is RENAMED
    (never deleted, never overwritten) into ``output_dir/abandoned_branch_<k>/``
    for the smallest free *k*, and a ``manifest.json`` (file, sha256, stored
    epoch, reason) is written there.

    Refuses (raises ``FileExistsError``) if a move would overwrite an existing
    archive path — a partial archive from an earlier crash is never clobbered.
    Logs and ``segment_*.json`` records are never moved (§11.1 condition 4).
    Returns ``[]`` when there is nothing to archive.
    """
    output_dir = Path(output_dir)
    entries: List[Tuple[Path, Dict[str, Any]]] = []
    for cand in list_abandoned_branch(output_dir, source_epoch):
        entries.append((cand, {
            "file": cand.name,
            "sha256": _sha256_file(cand),
            "stored_epoch": int(_stored_epoch(cand)),
            "reason": (
                f"abandoned-branch artifact: stored epoch "
                f"{_stored_epoch(cand)} > resume source epoch {source_epoch}; "
                f"the disclosed rollback re-trains epochs "
                f"{source_epoch + 1}..{last_executed} and would overwrite it"
            ),
        }))
    if not entries:
        return []

    archive_dir = _next_archive_dir(output_dir)
    archive_dir.mkdir(parents=True, exist_ok=False)
    manifest: List[Dict[str, Any]] = []
    for cand, entry in entries:
        target = archive_dir / cand.name
        if target.exists():  # pragma: no cover - fresh dir; safety net
            raise FileExistsError(
                f"refusing to overwrite existing archive path {target}; the "
                "abandoned-branch archive would clobber a prior move (fail closed)"
            )
        os.rename(cand, target)
        manifest.append(entry)

    payload = {
        "archived_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": (
            f"disclosed rollback resume: source {source_name} stored epoch "
            f"{source_epoch}; re-training epochs {source_epoch + 1}.."
            f"{last_executed}"
        ),
        "source": source_name,
        "source_epoch": int(source_epoch),
        "last_executed_epoch": int(last_executed),
        "retrained_range": [int(source_epoch) + 1, int(last_executed)],
        "files": manifest,
    }
    (archive_dir / _MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    return manifest
