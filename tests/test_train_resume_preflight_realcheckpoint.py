"""A3: does ``train.py``'s OWN resume preflight accept a checkpoint written by
the pre-amendment commit ``8ed5edb``?

The successor-commit amendment (§11.1, 2026-10-11) asserts that ``train.py``'s
resume preflight would accept a checkpoint written by the earlier commit's code,
so the resume does not diverge from §11.1's fail-closed rules.  Prose is not
evidence: this test drives the REAL preflight helpers (never a
re-implementation) against the actual checkpoint.

The FIRST INTENDED USE is ``R0-control-seed44`` (started on ``8ed5edb``, died
right after writing ``epoch_150.pth`` — a clean boundary, so the disclosed-
rollback rule ACCEPTS it); its ``epoch_150.pth`` (stored 149) is the accepted
source exercised here.  ``R0-control-seed43`` reached epoch 57 with the newest
periodic at stored 49 — a DISCLOSED rollback (re-train 50..57), no longer a
refusal; that disclosure is covered by SYNTHETIC tests (no test depends on the
mutable live dirs).

The ``active_lineage`` it builds is NON-CIRCULAR: the nested split / fingerprint
/ provenance are loaded independently from the nested-prior artifact via
``load_nested_prior_split`` (the same loader ``train()`` uses), and the dataset
source SHA from the dataset file — NOT copied out of the checkpoint under test.
A checkpoint that disagreed with the artifact would therefore be refused by
``_check_checkpoint_prior_lineage`` rather than trivially passing.

The test SKIPS when the READ-ONLY corpus / run dirs are absent (a machine
without the phase-2 artifacts), so it never fails a checkout that lacks them.
It never writes to the run dir.

STATE-INDEPENDENCE (round-10 amendment): the tests here read ONLY the immutable
source checkpoint (``epoch_150.pth``) and the read-only corpus; NO assertion
depends on the mutable live seed-44 dir's evolving state (the resume this
amendment authorizes MUTATES it).  The previous live-dir state-pinning tests were
REMOVED — their synthetic replacements live in ``test_epoch_recovery.py`` — and
the live seed-44 dry-run evidence is recorded in the amendment log instead.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

# The seed-44 run dir (immutable source checkpoint) and the corpus are READ-ONLY
# inputs; these tests only read.
_ROOT = Path(
    "/mnt/d/Projects/NSMoR-phase2-8ed5edb/.scratch/realdata-phase2-20261010"
)
_RUN_DIR = _ROOT / "R0-control-seed44"
_CKPT = _RUN_DIR / "epoch_150.pth"
_CORPUS = Path(
    "/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c"
)
_DATASET = _CORPUS / "etl-causal-final-zp12m0fa/nsmor_dataset.pt"
_ARTIFACT = _CORPUS / "nested-causal-complete-dku91_m0/nested_split_seed42.pt"

_CFG = (
    Path(__file__).resolve().parents[1]
    / "config/realdata-phase2-20261010/R0-control-seed44.yaml"
)

pytestmark = pytest.mark.skipif(
    not (_CKPT.exists() and _DATASET.exists() and _ARTIFACT.exists()),
    reason="seed-44 run dir / corpus absent (READ-ONLY inputs not on this machine)",
)


def _load() -> Any:
    import yaml

    from nsmor.config_parser import ExperimentConfig
    from nsmor.pipeline.nested_prior import load_artifact_bytes

    return yaml, ExperimentConfig, load_artifact_bytes


def test_train_resume_preflight_accepts_pre_amendment_checkpoint() -> None:
    """Every preflight guard ACCEPTS the 8ed5edb-written seed-44 epoch_150.pth."""
    import hashlib as _hashlib

    yaml, ExperimentConfig, load_artifact_bytes = _load()

    from nsmor.pipeline.grouping import prior_identity_status
    from nsmor.pipeline.nested_prior import (
        compute_source_fingerprint,
        load_nested_prior_split,
    )
    from nsmor.pipeline.resampling import validate_checkpoint_clock
    from scripts import train as T

    config = ExperimentConfig.from_yaml(str(_CFG))
    p2 = (yaml.safe_load(_CFG.read_text()) or {}).get("phase2") or {}
    # Stamp the arm's script-owned controls exactly as build_config does, so
    # _require_architecture_config_match compares against the right arm.
    T._SELECTION_METRIC = str(p2.get("selection_metric", "total"))
    T._ZERO_INPUT_CHANNELS = tuple(
        sorted(int(c) for c in (p2.get("zero_input_channels") or []))
    )

    payload = _CKPT.read_bytes()
    peek = load_artifact_bytes(payload, map_location="cpu")

    ds = load_artifact_bytes(_DATASET.read_bytes(), map_location="cpu")
    n_total = len(ds["X_seqs"])
    session_ids = list(ds["session_ids"])

    # Non-circular: independently reconstruct the split + provenance from the
    # artifact, not from the checkpoint.
    _tr, _va, _pri, info = load_nested_prior_split(
        _ARTIFACT, _DATASET, n_total, session_ids,
    )
    active: dict[str, Any] = {
        "nested_prior_artifact": str(_ARTIFACT),
        "nested_prior_artifact_sha256": info["nested_prior_artifact_sha256"],
        "nested_prior_fingerprint": info["nested_prior_fingerprint"],
        "nested_split_seed": int(info["split_seed"]),
        "nested_val_split": float(info["val_split"]),
        "is_nested_cv": True,
        "validation_scope": "nested_outer_validation",
        "mcmc_prior_provenance": info["mcmc_prior_provenance"],
        "animal_identity_status": info["animal_identity_status"],
        "dataset_path": str(_DATASET),
        "dataset_source_sha256": compute_source_fingerprint(_DATASET),
    }

    # 1. Clock (dt_ms).
    validate_checkpoint_clock(peek, config.model.dt_ms, require_dt_ms=True)
    # 2. Newest complete-epoch source + disclosed-rollback resolution (the source
    #    is its own dir's newest, so no rollback is disclosed).
    T._resolve_resume_source_and_rollback(_CKPT, peek, _RUN_DIR)
    # 3. Complete current schema.
    T._require_complete_current_schema(peek, _CKPT)
    # 4. Architecture / script-owned controls match the active arm.
    T._require_architecture_config_match(peek, config, _CKPT)
    # 5. Prior lineage (nested branch + unverified identity branch).
    T._check_checkpoint_prior_lineage(
        peek, active, _CKPT,
        trusted_historical_checkpoint_sha256=None,
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_train_restore_recovery_state_accepts_pre_amendment_checkpoint() -> None:
    """R1-M3/R2-M2: the restoration seam accepts the 8ed5edb-written checkpoint.

    ``_restore_recovery_state`` is the recovery seam train.py runs on every
    resume (patience counter, training-history axis, required MT19937 state).
    Driving the REAL helper — not a re-implementation — demonstrates the guard
    that the earlier preflight-only artifact omitted.  ``resume_epoch`` is the
    checkpoint's own stored epoch + 1 (49 -> 50), matching train.py.
    """
    _yaml, _cfg, load_artifact_bytes = _load()

    from scripts import train as T

    peek = load_artifact_bytes(_CKPT.read_bytes(), map_location="cpu")
    resume_epoch = int(peek["epoch"]) + 1
    ewi, history, np_state, history_start = T._restore_recovery_state(
        peek, _CKPT, resume_epoch=resume_epoch,
    )
    # A current-schema (v2) payload restores positionally: the counter is a
    # non-negative int, the two loss series share one axis of ``resume_epoch``
    # entries (0..49), the axis starts at 0, and the MT19937 state is present.
    assert isinstance(ewi, int) and ewi >= 0
    assert set(history) == {"train_loss", "val_loss"}
    assert len(history["train_loss"]) == len(history["val_loss"]) == resume_epoch
    assert history_start == 0
    assert np_state is not None


def test_resume_loader_rng_guard_accepts_real_corpus_loaders() -> None:
    """R1-M3/R2-M2: ``_require_bitwise_resume_loader_rng`` accepts the arm loaders.

    The guard train.py runs on every resume (before the resume block) must PASS
    for the phase-2 corpus, whose ``NSMoRDataset.__getitem__`` and collate
    consume no worker-side RNG.  Built through the real ``build_dataloaders``
    with the arm's own config, so the assertion is on the path the resume takes.
    """
    yaml, ExperimentConfig, _load_bytes = _load()

    from scripts import train as T

    config = ExperimentConfig.from_yaml(str(_CFG))
    p2 = (yaml.safe_load(_CFG.read_text()) or {}).get("phase2") or {}
    T._SELECTION_METRIC = str(p2.get("selection_metric", "total"))
    T._ZERO_INPUT_CHANNELS = tuple(
        sorted(int(c) for c in (p2.get("zero_input_channels") or []))
    )
    train_loader, val_loader = T.build_dataloaders(
        config, dataset_path=str(_DATASET),
        nested_prior_artifact=str(_ARTIFACT),
    )
    # Must NOT raise: the phase-2 corpus satisfies the worker-RNG precondition.
    T._require_bitwise_resume_loader_rng(train_loader, val_loader)


def test_preflight_scope_is_disclosed_not_overstated() -> None:
    """The acceptance rests on the nested path; the strongest guard IS reached.

    The phase-2 checkpoint records ``animal_identity_status='unverified'`` and
    has NO ``dataset_source_binding`` key.  The strongest lineage guard
    (``_check_checkpoint_prior_lineage``'s ``unverified`` branch, which requires
    ``dataset_source_binding == 'sha256_bound'``) is therefore NOT skipped: the
    recorded value defaults to ``sha256_bound`` and is checked.  This test pins
    that scope so the acceptance cannot be presented as unqualified.
    """
    _yaml, _cfg, load_artifact_bytes = _load()

    from nsmor.pipeline.nested_prior import compute_source_fingerprint

    peek = load_artifact_bytes(_CKPT.read_bytes(), map_location="cpu")
    assert peek.get("animal_identity_status") == "unverified"
    assert "dataset_source_binding" not in peek  # defaults to sha256_bound
    # The dataset the checkpoint was trained on is the artifact's source.
    assert peek["dataset_source_sha256"] == compute_source_fingerprint(_DATASET)


def test_seed44_shape_clean_boundary_then_disclosed_rollback(tmp_path: Path) -> None:
    """The seed-44 SHAPE (synthetic): clean boundary, then a disclosed rollback.

    NO live dir is touched (round-10 amendment: the live-dir state-pinning tests
    were removed).  This drives the REAL helpers on a SYNTHETIC dir with the
    seed-44 shape (newest periodic ``epoch_150.pth`` stored 149, log
    ``Epoch 150/300``):
      * pre-resume — clean boundary (last executed 149 == source 149): NO rollback;
      * first resume IN PROGRESS — its segment record (start 150) still witnesses
        only 149, so it stays a clean boundary (the round-5 hazard: a RAW start
        read over-claimed to 150 and falsely disclosed a rollback);
      * first resume's FIRST epoch COMPLETE (log ``Epoch 151/300``) — the run has
        genuinely executed through 150, so the old ``epoch_150.pth`` is now a
        DISCLOSED rollback (re-train 150..150), not a refusal.
    """
    import json

    from scripts import train as T

    work = tmp_path / "seed44"
    work.mkdir()
    src = work / "epoch_150.pth"
    _save_ckpt(src, 149)
    peek = {"epoch": 149}

    # Pre-resume: last executed == newest checkpoint == 149 -> clean boundary.
    (work / "train.stdout.log").write_text("[t] INFO - Epoch 150/300  train_loss=1.0\n")
    assert T._last_executed_epoch(work) == 149
    assert T._resolve_resume_source_and_rollback(src, peek, work) == (None, [])

    # The first resume IN PROGRESS writes its segment record (start = 149 + 1);
    # its first epoch has not completed, so the boundary stays 149.
    (work / "segment_0001.json").write_text(json.dumps({"starting_epoch": 150}))
    assert T._last_executed_epoch(work) == 149, (
        "the resumed segment's start over-claimed the last executed epoch "
        "(round-5 hazard): the clean-boundary recovery would be falsely disclosed"
    )
    assert T._resolve_resume_source_and_rollback(src, peek, work) == (None, [])

    # The resumed segment's FIRST epoch completes: the run has now executed
    # through 150, so the old epoch_150.pth (149) is a DISCLOSED rollback.
    (work / "train.segment1.log").write_text(
        "[t] INFO - Epoch 151/300  train_loss=1.0\n"
    )
    assert T._last_executed_epoch(work) == 150
    assert T._resolve_resume_source_and_rollback(src, peek, work) == ((150, 150), [])


def _save_ckpt(path: Path, epoch: int) -> None:
    import torch

    torch.save({"recovery_state_version": 2, "epoch": epoch}, path)
