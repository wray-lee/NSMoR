"""A3: does ``train.py``'s OWN resume preflight accept a checkpoint written by
the pre-amendment commit ``8ed5edb``?

The successor-commit amendment (§11.1, 2026-10-11) asserts that ``train.py``'s
resume preflight would accept ``epoch_50.pth`` — a checkpoint written by the
earlier commit's code — so the resume does not diverge from §11.1's fail-closed
rules.  Prose is not evidence: this test drives the REAL preflight helpers
(never a re-implementation) against the actual checkpoint.

The ``active_lineage`` it builds is NON-CIRCULAR: the nested split / fingerprint
/ provenance are loaded independently from the nested-prior artifact via
``load_nested_prior_split`` (the same loader ``train()`` uses), and the dataset
source SHA from the dataset file — NOT copied out of the checkpoint under test.
A checkpoint that disagreed with the artifact would therefore be refused by
``_check_checkpoint_prior_lineage`` rather than trivially passing.

The test SKIPS when the READ-ONLY corpus / run dir are absent (a machine
without the phase-2 artifacts), so it never fails a checkout that lacks them.
It never writes to the run dir.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

# The seed-43 run and its corpus are READ-ONLY inputs; this test only reads.
_RUN_DIR = Path(
    "/mnt/d/Projects/NSMoR-phase2-8ed5edb/.scratch/"
    "realdata-phase2-20261010/R0-control-seed43"
)
_CKPT = _RUN_DIR / "epoch_50.pth"
_CORPUS = Path(
    "/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c"
)
_DATASET = _CORPUS / "etl-causal-final-zp12m0fa/nsmor_dataset.pt"
_ARTIFACT = _CORPUS / "nested-causal-complete-dku91_m0/nested_split_seed42.pt"

_CFG = (
    Path(__file__).resolve().parents[1]
    / "config/realdata-phase2-20261010/R0-control-seed43.yaml"
)

pytestmark = pytest.mark.skipif(
    not (_CKPT.exists() and _DATASET.exists() and _ARTIFACT.exists()),
    reason="seed-43 run dir / corpus absent (READ-ONLY inputs not on this machine)",
)


def _load() -> Any:
    import yaml

    from nsmor.config_parser import ExperimentConfig
    from nsmor.pipeline.nested_prior import load_artifact_bytes

    return yaml, ExperimentConfig, load_artifact_bytes


def test_train_resume_preflight_accepts_pre_amendment_checkpoint() -> None:
    """Every preflight guard ACCEPTS the 8ed5edb-written epoch_50.pth."""
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
