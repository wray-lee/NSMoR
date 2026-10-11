"""A3: does ``train.py``'s OWN resume preflight accept a checkpoint written by
the pre-amendment commit ``8ed5edb``?

The successor-commit amendment (§11.1, 2026-10-11) asserts that ``train.py``'s
resume preflight would accept a checkpoint written by the earlier commit's code,
so the resume does not diverge from §11.1's fail-closed rules.  Prose is not
evidence: this test drives the REAL preflight helpers (never a
re-implementation) against the actual checkpoint.

The FIRST INTENDED USE is ``R0-control-seed44`` (started on ``8ed5edb``, died
right after writing ``epoch_150.pth`` — a clean boundary, so F1 ACCEPTS it); its
``epoch_150.pth`` (stored 149) is the accepted source exercised here.
``R0-control-seed43`` is NOT resumed (F1 no-silent-rollback refuses its
``epoch_50.pth``: the run reached epoch 57 but the newest periodic is stored 49);
a test below pins that refusal against the real dir.

The ``active_lineage`` it builds is NON-CIRCULAR: the nested split / fingerprint
/ provenance are loaded independently from the nested-prior artifact via
``load_nested_prior_split`` (the same loader ``train()`` uses), and the dataset
source SHA from the dataset file — NOT copied out of the checkpoint under test.
A checkpoint that disagreed with the artifact would therefore be refused by
``_check_checkpoint_prior_lineage`` rather than trivially passing.

The test SKIPS when the READ-ONLY corpus / run dirs are absent (a machine
without the phase-2 artifacts), so it never fails a checkout that lacks them.
It never writes to the run dir.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

# The seed-44 (accepted) and seed-43 (F1-refused) runs and the corpus are
# READ-ONLY inputs; these tests only read.
_ROOT = Path(
    "/mnt/d/Projects/NSMoR-phase2-8ed5edb/.scratch/realdata-phase2-20261010"
)
_RUN_DIR = _ROOT / "R0-control-seed44"
_CKPT = _RUN_DIR / "epoch_150.pth"
_SEED43_DIR = _ROOT / "R0-control-seed43"
_SEED43_CKPT = _SEED43_DIR / "epoch_50.pth"
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
    # 2. Newest complete-epoch source (the source is its own dir's newest).
    T._require_resume_source_is_newest_complete_epoch(_CKPT, peek, _RUN_DIR)
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


def test_real_seed44_dir_is_a_clean_boundary() -> None:
    """The seed-44 dir's last executed epoch (149) equals its newest checkpoint.

    ``_last_executed_epoch`` reads the run's OWN records; for seed-44 the log's
    last line is ``Epoch 150/300`` (0-based 149) and the newest periodic is
    ``epoch_150.pth`` (stored 149), so the boundary is clean and the source is
    accepted (no rollback).

    STABLE ACROSS THE FIRST RESUME: seed-44's ``segment_0000.json`` records
    ``starting_epoch`` 0 (-> -1).  The first resume this commit authorizes writes
    ``segment_0001.json``{``starting_epoch``: 150}, which under the H1 start-1
    rule witnesses only epoch 149 — so this assertion (149) is NOT invalidated
    by the very action it gates (round-4 MAJOR: a raw start witness returned 150
    and broke the pin).
    """
    from scripts import train as T

    assert T._last_executed_epoch(_RUN_DIR) == 149


def test_real_seed43_dir_is_refused_by_f1_no_silent_rollback() -> None:
    """F1: the seed-43 dir is REFUSED — its reached epoch exceeds the source.

    The seed-43 run reached ``Epoch 58/300`` (0-based 57) but its newest
    last-epoch checkpoint is ``epoch_50.pth``, so resuming would silently
    re-train epochs 50..57.  Driving the REAL helper against the real dir
    (read-only) pins the refusal, and the message names both epochs.  The
    source's ``peek`` is the REAL ``epoch_50.pth``'s stored epoch (read from the
    file), not a hand-written value, so the test drives the real dir end-to-end
    (round-4 minor).
    """
    from nsmor.pipeline.nested_prior import load_artifact_bytes

    from scripts import train as T

    if not _SEED43_CKPT.exists():  # pragma: no cover - absent on some machines
        pytest.skip("seed-43 dir absent")
    peek = load_artifact_bytes(_SEED43_CKPT.read_bytes(), map_location="cpu")
    stored = peek.get("epoch")
    assert isinstance(stored, int) and not isinstance(stored, bool), peek.get("epoch")
    with pytest.raises(ValueError, match="no-silent-rollback") as exc:
        T._require_resume_source_is_newest_complete_epoch(
            _SEED43_CKPT, peek, _SEED43_DIR,
        )
    assert str(stored) in str(exc.value) and "57" in str(exc.value), str(exc.value)
