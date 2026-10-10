"""
NSMoR Main Training Engine.

Ties together the full training pipeline:

1. Load experiment configuration from YAML + CLI overrides.
2. Initialize model, optimizer, loss function, and dataloaders.
3. Run the training loop with validation and checkpointing.

Usage
-----
CLI::

    python scripts/train.py --config config/default.yaml --nested_prior_artifact nested.pt
    python scripts/train.py --config config/default.yaml --lr 5e-4 --epochs 200 --nested_prior_artifact nested.pt
    python scripts/train.py --config config/default.yaml --diagnostic_only

Programmatic::

    from scripts.train import train, build_config
    cfg = build_config(["--config", "config/default.yaml"])
    results = train(cfg)  # lambda_reg falls through to cfg.loss.lambda_reg
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Prefer the checkout's own ``nsmor`` package over any editable install that
# points at a different worktree/main checkout.  Running ``scripts/train.py``
# directly puts ``scripts/`` on ``sys.path[0]``, not the repo root, so without
# this the import would resolve to the installed ``nsmor`` and miss modules
# that exist only in this checkout (same pattern as the other scripts).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# ── Project imports ────────────────────────────────────────────
from nsmor.checkpoint import load_checkpoint as _canonical_load_checkpoint
from nsmor.checkpoint import save_checkpoint
from nsmor.config import DEFAULT_FEATURE
from nsmor.config_parser import ExperimentConfig
from nsmor.dataloader_factory import create_dataloaders_from_config
from nsmor.loss import BioJointLoss, BioDecisionLoss, FrontendLoss
from nsmor.model_utils import (
    require_trusted_historical_checkpoint_sha256, resolve_dataset_session_ids,
)
from nsmor.model_nsmor_core import NSMoRCore
from nsmor.pipeline.conditions import derive_stimulus_metadata
from nsmor.pipeline.grouping import grouped_train_val_split, prior_identity_status
from nsmor.pipeline.nested_prior import (
    compute_source_fingerprint,
    load_artifact_bytes,
    load_dataset_with_fingerprint,
    load_nested_prior_split,
)
from nsmor.pipeline.resampling import validate_checkpoint_clock, validate_dt_ms


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    *,
    map_location: str | torch.device | None = None,
    payload: bytes | None = None,
) -> Dict[str, Any]:
    """Guard resume and final-best clocks, then restore the same captured bytes."""
    if payload is None:
        payload = Path(path).read_bytes()
    checkpoint = load_artifact_bytes(payload, map_location="cpu")
    active_dt = (
        validate_dt_ms(model.dt_ms, name="active dt_ms")
        if hasattr(model, "dt_ms") else None
    )
    validate_checkpoint_clock(
        checkpoint, active_dt, require_dt_ms=hasattr(model, "dt_ms"),
    )
    return _canonical_load_checkpoint(
        path, model, optimizer, scheduler, map_location=map_location,
        payload=payload,
    )


# ── Deployment provenance keys ────────────────────────────────
# These keys are injected into every checkpoint by
# _atomic_save_checkpoint but are NOT part of the frozen
# save_checkpoint signature.  The wrapper pops them before
# forwarding kwargs, then patches them into the saved state
# dict on disk before the atomic rename.
#
# nested_prior_artifact / nested_prior_fingerprint / is_nested_cv are
# mandatory when the opt-in nested seam is used: a checkpoint that does
# not record which nested artifact produced its split/priors cannot be
# audited later, and a downstream consumer cannot refuse a mismatched
# resumption.  mcmc_prior_provenance is the human-readable lineage
# string (legacy OOF or nested).
_PROVENANCE_KEYS = frozenset({
    "target_mean",
    "target_std",
    "target_clip_cm_s",
    "training_phase",
    "dataset_path",
    "dataset_source_sha256",
    "dataset_source_binding",
    "mcmc_prior_train_serve_consistency",
    "nested_prior_artifact",
    "nested_prior_artifact_sha256",
    "nested_prior_fingerprint",
    "nested_split_seed",
    "nested_val_split",
    "is_nested_cv",
    "validation_scope",
    "mcmc_prior_provenance",
    "animal_identity_status",
    "best_val_loss",
    # Phase-2 script-owned controls (NOT part of the protected config
    # schema).  Recorded additively so a checkpoint names the exact
    # selection metric / history ablation it was trained under.
    "selection_metric",
    "zero_input_channels",
    # Epoch-boundary recovery state (see the recovery helpers below).  These
    # ride the same pop-then-patch seam so nsmor/checkpoint.py stays untouched.
    "recovery_state_version",
    "epochs_without_improvement",
    "training_history",
    "history_start_epoch",
    "numpy_rng_state",
})

# A current-schema checkpoint CLAIMS exact continuation, so these canonical
# state blocks must all be present (the recovery payload's counter / history /
# axis / NumPy MT19937 state are validated separately in
# ``_restore_recovery_state``).  The canonical loader tolerates an absent
# optimizer / scheduler; a current-schema payload that omits one would resume
# with a fresh optimizer or restarted schedule, so presence is enforced here.
_CURRENT_SCHEMA_REQUIRED_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "rng_state",
)

# A non-empty ``scheduler_state_dict`` is NOT enough.  ``LRScheduler.load_state_dict``
# SILENTLY MERGES whatever mapping it is given: it never resets to defaults and
# never validates the key set, so a truncated or arbitrary mapping loads without
# error while retaining the freshly-built scheduler's defaults.  The real
# producer is ``CosineAnnealingLR`` (built at train.py:~4019 for phase 1 and in
# ``_build_phase2_optimizer_scheduler`` for phase 2).  ``LRScheduler.state_dict``
# serializes every ``__dict__`` entry except the optimizer, so the EXACT key set
# is the production schema of the pinned torch: ``T_max`` / ``eta_min`` (the
# cosine endpoints), ``base_lrs`` / ``last_epoch`` / ``_last_lr`` (the per-group
# base rates, the step position, and the cached last LR), plus ``_step_count``,
# ``_get_lr_called_within_step`` and ``_is_initial`` (the torch-2.x step-state
# flags that decide which ``get_lr`` branch runs).  A resume that loads only a
# subset (e.g. ``{"last_epoch": 1}``) keeps the fresh ``base_lrs`` / ``T_max``
# and silently anneals the wrong schedule, so the complete key/value STRUCTURE
# is required for an exact continuation.  ``verbose`` is NOT part of this
# version's state (torch 2.14.0+cu132); an extra key is a foreign schema and
# fails closed too.
_SCHEDULER_STATE_REQUIRED_KEYS = (
    "T_max", "_get_lr_called_within_step", "_is_initial", "_last_lr",
    "_step_count", "base_lrs", "eta_min", "last_epoch",
)
_SCHEDULER_STATE_INT_KEYS = ("T_max", "last_epoch", "_step_count")
_SCHEDULER_STATE_LIST_KEYS = ("base_lrs", "_last_lr")
_SCHEDULER_STATE_BOOL_KEYS = ("_get_lr_called_within_step", "_is_initial")


def _is_real_number(value: Any) -> bool:
    """True for a FINITE numeric scalar (``bool`` excluded — not a scheduler rate).

    ``NaN`` / ``Inf`` are rejected here, not tolerated: the producer's
    ``CosineAnnealingLR`` is built with finite endpoints (``eta_min=1e-6``,
    finite group rates), so a non-finite rate is never a real continuation.
    ``load_state_dict`` would merge it verbatim onto the optimizer's
    ``group["lr"]``, where it poisons every subsequent update and the terminal
    checkpoint instead of failing loudly.
    """
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return False
    return math.isfinite(float(value))


def _require_complete_scheduler_state(
    sched_sd: Any, ckpt_path: Path, opt_group_count: int,
) -> None:
    """Fail closed unless *sched_sd* is the COMPLETE ``CosineAnnealingLR`` state.

    ``LRScheduler.load_state_dict`` silently merges a partial mapping, so mere
    non-emptiness cannot distinguish a full schedule state from a damaged one
    that would resume with the freshly-built scheduler's defaults.  This checks
    the producer's load-bearing key/value structure (see
    ``_SCHEDULER_STATE_REQUIRED_KEYS``); it is scoped to the current schema and
    to the ``CosineAnnealingLR`` the trainer actually builds, not a generic
    scheduler framework.

    The per-group rate sequences must also have exactly ``opt_group_count``
    entries — the number of param groups the SAME checkpoint's
    ``optimizer_state_dict`` records.  ``CosineAnnealingLR._update_lr`` zips
    ``values`` against ``optimizer.param_groups`` with ``strict=True`` but only
    after ``get_lr`` has already built ``values`` from ``base_lrs``, so a
    one-element ``base_lrs`` with a two-group optimizer would be truncated to
    the first group: the LIF group's LR would never be annealed while the run
    still claimed an exact continuation.
    """
    missing = [k for k in _SCHEDULER_STATE_REQUIRED_KEYS if k not in sched_sd]
    if missing:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} scheduler_state_dict is missing "
            f"required CosineAnnealingLR fields {missing}; a partial mapping "
            "would silently merge with fresh defaults (fail closed)."
        )
    extra = sorted(set(sched_sd) - set(_SCHEDULER_STATE_REQUIRED_KEYS))
    if extra:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} scheduler_state_dict carries "
            f"unexpected fields {extra} for the pinned torch "
            f"{torch.__version__} CosineAnnealingLR schema; fail closed."
        )
    for key in _SCHEDULER_STATE_BOOL_KEYS:
        value = sched_sd[key]
        if not isinstance(value, bool):
            raise ValueError(
                f"Resume checkpoint {ckpt_path} scheduler_state_dict[{key!r}] is "
                f"not a bool ({type(value).__name__}); fail closed."
            )
    for key in _SCHEDULER_STATE_INT_KEYS:
        value = sched_sd[key]
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(
                f"Resume checkpoint {ckpt_path} scheduler_state_dict[{key!r}] is "
                f"not an integer ({type(value).__name__}); fail closed."
            )
    if sched_sd["T_max"] < 1:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} scheduler_state_dict['T_max'] is "
            f"{sched_sd['T_max']}; the producer clamps the horizon to >= 1 "
            "(CosineAnnealingLR is built as max(1, num_epochs - ...)), so a "
            "non-positive T_max is not a real continuation (fail closed)."
        )
    if sched_sd["last_epoch"] < -1:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} scheduler_state_dict['last_epoch'] "
            f"is {sched_sd['last_epoch']}; the producer never steps before the "
            "initial -1 position (fail closed)."
        )
    if sched_sd["_step_count"] < 0:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} scheduler_state_dict['_step_count'] "
            f"is {sched_sd['_step_count']}; a negative step count is not a real "
            "continuation (fail closed)."
        )
    if sched_sd["_step_count"] != sched_sd["last_epoch"] + 1:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} scheduler_state_dict['_step_count'] "
            f"is {sched_sd['_step_count']} but last_epoch is "
            f"{sched_sd['last_epoch']}; the production scheduler increments both "
            "once per step (fail closed)."
        )
    if not _is_real_number(sched_sd["eta_min"]):
        raise ValueError(
            f"Resume checkpoint {ckpt_path} scheduler_state_dict['eta_min'] is "
            f"not a finite number ({type(sched_sd['eta_min']).__name__}); "
            "fail closed."
        )
    lengths = set()
    for key in _SCHEDULER_STATE_LIST_KEYS:
        value = sched_sd[key]
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(
                f"Resume checkpoint {ckpt_path} scheduler_state_dict[{key!r}] is "
                f"not a non-empty sequence ({type(value).__name__}); fail closed."
            )
        if not all(_is_real_number(v) for v in value):
            raise ValueError(
                f"Resume checkpoint {ckpt_path} scheduler_state_dict[{key!r}] "
                "contains a non-finite or non-numeric entry; fail closed."
            )
        lengths.add(len(value))
    if len(lengths) != 1:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} scheduler_state_dict base_lrs/_last_lr "
            "have mismatched group counts; fail closed."
        )
    if next(iter(lengths)) != opt_group_count:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} scheduler_state_dict base_lrs/_last_lr "
            f"have {next(iter(lengths))} entries but optimizer_state_dict records "
            f"{opt_group_count} param_groups; the extra group(s) would never be "
            "annealed (fail closed)."
        )


# ── Epoch-boundary recovery state ─────────────────────────────
# A resumed run must CONTINUE, not restart, the counters that shape its own
# trajectory.  The early-stopping counter and the accumulated loss history
# are read/advanced every epoch but were previously re-initialised after the
# resume block, so a SIGKILL/OOM restart silently reset patience and lost
# the plotted history.  They ride the provenance seam above (popped before
# the frozen save_checkpoint, patched into the on-disk state) so
# nsmor/checkpoint.py keeps its storage contract.
#
# Bump RECOVERY_STATE_VERSION when the schema or its continuation semantics
# change: a checkpoint stamped with a different version is not trusted for
# exact continuation, and the run says so loudly instead of silently
# restarting the counters.
#
# v2 semantics (reviewer repair): the diagnostic histories are persisted in
# FULL — one value per executed epoch, in order, positionally aligned across
# ``train_loss`` and ``val_loss``.  A non-finite scalar diagnostic (NaN/Inf
# from an unstable epoch) is written AS-IS, never dropped or coerced, so the
# epoch axis never collapses and a resumed run can keep appending.  A gap
# ``None`` marks an epoch at which a series was not recorded (e.g. no
# validation); it occupies its epoch slot so the axis stays aligned.
# ``history_start_epoch`` records the true 0-based epoch of entry 0, so a
# resume from a LEGACY checkpoint (unknown history) starts its axis at the
# resume epoch rather than falsely redrawing it from epoch 1.  The
# early-stopping counter is a property of the finite-validation series ONLY
# and is kept as an independent control value; it is never inferred from the
# diagnostic completeness.
RECOVERY_STATE_VERSION = 2


def _encode_numpy_rng_state(state: tuple) -> Dict[str, Any]:
    """Encode ``np.random.get_state()`` as JSON-safe scalars + one tensor.

    The legacy MT19937 key array (624 uint32) rides as a torch tensor, which
    the restricted decoder (``load_artifact_bytes``) already admits; the
    scalar fields are plain ints/floats.  ``np.random.set_state`` reproduces
    this stream exactly — the modern ``Generator`` has no equivalent
    serialisable form.  Storing NumPy state matters only for the deprecated
    legacy random crop; the anchor-aligned crop is deterministic, so the
    stream is frozen across epochs and this is a no-op there.  It is
    persisted anyway so recovery does not depend on which crop path runs.
    """
    name, keys, pos, has_gauss, cached = state
    # ``.clone()`` detaches from the live MT19937 buffer: ``ascontiguousarray``
    # may return the same array, and ``from_numpy`` would then alias state that
    # the next ``np.random`` call mutates before this is serialized.
    keys_tensor = torch.from_numpy(
        np.ascontiguousarray(keys, dtype=np.uint32)
    ).clone()
    # Shape/dtype contract of the persisted MT19937 key array.  A silent
    # change here (e.g. a different NumPy build handing back a shorter
    # state) would produce a checkpoint whose RNG stream cannot be
    # restored; assert loudly instead.
    assert keys_tensor.dtype == torch.uint32, keys_tensor.dtype
    assert keys_tensor.shape == (624,), keys_tensor.shape
    assert keys_tensor.device.type == "cpu", keys_tensor.device
    return {
        "bit_generator": str(name),
        "keys": keys_tensor,
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _decode_numpy_rng_state(record: Any) -> tuple:
    """Validate and rebuild an ``np.random.get_state()`` tuple, else raise.

    A current-schema ``numpy_rng_state`` is REQUIRED recovery state: if it is
    absent or malformed the resumed NumPy stream cannot be reproduced, so the
    run must fail closed BEFORE executing any epoch rather than silently
    reseeding a different stream (independent reviews R2 / B-1).  Every field
    is checked against the exact legacy MT19937 schema — a fractional or
    boolean ``pos``/``has_gauss``, a string/bool ``cached_gaussian``, a
    non-``MT19937`` bit generator, or a wrongly shaped/dtyped key array is
    rejected outright and NEVER coerced with ``int()``/``float()``.
    """
    if not isinstance(record, dict):
        raise ValueError(
            f"numpy_rng_state is not a mapping ({type(record).__name__})"
        )
    if record.get("bit_generator") != "MT19937":
        raise ValueError(
            f"numpy_rng_state bit_generator={record.get('bit_generator')!r} "
            "is not the supported 'MT19937'"
        )
    keys = record.get("keys")
    if not isinstance(keys, torch.Tensor):
        raise ValueError(
            f"numpy_rng_state keys is not a tensor ({type(keys).__name__})"
        )
    if keys.dtype != torch.uint32:
        raise ValueError(f"numpy_rng_state keys dtype={keys.dtype} is not uint32")
    if keys.shape != (624,):
        raise ValueError(
            f"numpy_rng_state keys shape={tuple(keys.shape)} is not (624,)"
        )
    if keys.device.type != "cpu":
        raise ValueError(
            f"numpy_rng_state keys device={keys.device} is not CPU"
        )
    pos = record.get("pos")
    if isinstance(pos, bool) or not isinstance(pos, (int, np.integer)):
        raise ValueError(f"numpy_rng_state pos={pos!r} is not an integer")
    if not (0 <= int(pos) <= 624):
        raise ValueError(f"numpy_rng_state pos={pos!r} is outside [0, 624]")
    has_gauss = record.get("has_gauss")
    if isinstance(has_gauss, bool) or not isinstance(has_gauss, (int, np.integer)):
        raise ValueError(
            f"numpy_rng_state has_gauss={has_gauss!r} is not an integer"
        )
    if int(has_gauss) not in (0, 1):
        raise ValueError(
            f"numpy_rng_state has_gauss={has_gauss!r} is not 0 or 1"
        )
    cached = record.get("cached_gaussian")
    if isinstance(cached, bool) or not isinstance(
        cached, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"numpy_rng_state cached_gaussian={cached!r} is not numeric"
        )
    if not math.isfinite(float(cached)):
        raise ValueError(
            f"numpy_rng_state cached_gaussian={cached!r} is not finite"
        )
    return (
        "MT19937", keys.numpy().astype(np.uint32),
        int(pos), int(has_gauss), float(cached),
    )


# R3 / A5, B14: mechanism settings that change the model's ARCHITECTURE or the
# loss, not merely its numeric coefficients.  Fixed and adaptive refinement
# share one parameter tree, so a strict ``load_state_dict`` accepts a resume
# under different flags and silently continues as a DIFFERENT control arm.  The
# preflight below compares these exact leaves against the active config.
_ARCHITECTURE_CONFIG_LEAVES: Tuple[Tuple[str, str], ...] = (
    ("model", "refinement_mode"),
    ("model", "refinement_max_steps"),
    ("model", "refinement_eps"),
    ("model", "refinement_update_scale"),
    ("model", "activation"),
    ("loss", "lambda_compute"),
)


def _require_architecture_config_match(
    ckpt: Dict[str, Any], config: ExperimentConfig, ckpt_path: Path,
) -> None:
    """Fail closed when a resume switches an architecture-defining setting.

    A checkpoint stores ``config.to_dict()``.  This compares the recorded
    refinement / activation / ``lambda_compute`` leaves against the ACTIVE
    config and refuses any mismatch, so a resume can never silently continue as
    a different architecture (finding R3 / A5, B14).  Legacy checkpoints
    without a stored config are left to the existing limited fallback.
    """
    ckpt_cfg = ckpt.get("config")
    if not isinstance(ckpt_cfg, dict):
        return
    active = config.to_dict()
    mismatches: List[str] = []
    for section, leaf in _ARCHITECTURE_CONFIG_LEAVES:
        ckpt_section = ckpt_cfg.get(section)
        if not isinstance(ckpt_section, dict) or leaf not in ckpt_section:
            continue
        recorded = ckpt_section[leaf]
        current = active[section][leaf]
        if recorded != current:
            mismatches.append(
                f"{section}.{leaf}: checkpoint={recorded!r} vs active={current!r}"
            )
    # ``selection_metric`` (a script-owned phase-2 control, NOT part of the
    # protected config schema) changes the meaning of the persisted
    # ``best_val_loss`` scalar (total objective vs masked MSE).  A resume that
    # switched it would compare a restored best under one metric against
    # candidates under the other, so refuse the mixed-metric continuation.
    # It is recorded as an additive top-level checkpoint key; legacy
    # checkpoints without it default to "total".
    recorded_sel = ckpt.get("selection_metric", "total")
    active_sel = _SELECTION_METRIC
    if recorded_sel != active_sel:
        mismatches.append(
            f"selection_metric: checkpoint={recorded_sel!r} vs "
            f"active={active_sel!r}"
        )
    # ``zero_input_channels`` (the history ablation) is likewise script-owned
    # and not part of the protected config schema, but it changes the ARM
    # identity (R2 zeroes channels 2-3).  A resume that switched it would
    # silently continue an arm under a different ablation, so refuse it
    # alongside the selection_metric guard.  Legacy checkpoints without the
    # key default to the unablated control ([]).
    recorded_zic = sorted(int(c) for c in ckpt.get("zero_input_channels", []))
    active_zic = sorted(int(c) for c in _ZERO_INPUT_CHANNELS)
    if recorded_zic != active_zic:
        mismatches.append(
            f"zero_input_channels: checkpoint={recorded_zic} vs "
            f"active={active_zic}"
        )
    if mismatches:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} was trained with a different "
            f"architecture/loss configuration: {'; '.join(mismatches)}. "
            "Fixed and adaptive refinement share one parameter tree, so a "
            "strict load would silently continue as another control arm. "
            "Refusing resume (fail closed)."
        )


def _require_complete_current_schema(
    ckpt: Dict[str, Any], ckpt_path: Path,
) -> None:
    """Fail closed on an INCOMPLETE current-schema continuation payload.

    A checkpoint stamped with the current ``recovery_state_version`` claims to
    carry the full epoch-boundary continuation state, so a resume treats it as
    exactly reproducible.  A current-schema payload MISSING any required state
    — model weights, optimizer moments, scheduler, Torch RNG — is damaged
    required state, not a legacy gap: the canonical loader tolerates a missing
    ``optimizer_state_dict`` / ``scheduler_state_dict`` (it restores only what
    is present), which would silently resume with a FRESH optimizer (zero
    moments) or a restarted LR schedule while still claiming an exact
    continuation.  Missing, explicitly ``None``, or otherwise malformed
    required state therefore fails closed here, BEFORE any update or terminal
    save.

    Scope is deliberately narrow: the presence and shape (not the numerical
    content) of the four canonical keys is checked, plus the recovery payload
    the current schema also promises (counter / history / axis / NumPy MT19937
    state, validated in ``_restore_recovery_state``).  LEGACY / version-
    mismatched checkpoints keep their limited fallback and are not examined
    here.
    """
    if ckpt.get("recovery_state_version") != RECOVERY_STATE_VERSION:
        return
    missing = [k for k in _CURRENT_SCHEMA_REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} is current-schema "
            f"(recovery_state_version={RECOVERY_STATE_VERSION}) but is "
            f"missing required continuation state {missing}; damaged state, "
            "not a legacy gap (fail closed)."
        )
    none_keys = [k for k in _CURRENT_SCHEMA_REQUIRED_KEYS if ckpt[k] is None]
    if none_keys:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} records an explicit None for "
            f"required continuation state {none_keys}; fail closed."
        )
    model_sd = ckpt["model_state_dict"]
    if not isinstance(model_sd, dict) or not model_sd:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} model_state_dict is not a "
            f"non-empty mapping ({type(model_sd).__name__}); fail closed."
        )
    opt_sd = ckpt["optimizer_state_dict"]
    if not isinstance(opt_sd, dict) or not opt_sd:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} optimizer_state_dict is not a "
            f"non-empty mapping ({type(opt_sd).__name__}); a fresh optimizer "
            "would silently discard the saved moments (fail closed)."
        )
    opt_groups = opt_sd.get("param_groups")
    if not isinstance(opt_groups, (list, tuple)) or not opt_groups:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} optimizer_state_dict has no "
            f"non-empty param_groups ({type(opt_groups).__name__}); the "
            "scheduler's per-group rates cannot be bound to the optimizer "
            "state (fail closed)."
        )
    sched_sd = ckpt["scheduler_state_dict"]
    if not isinstance(sched_sd, dict) or not sched_sd:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} scheduler_state_dict is not a "
            f"non-empty mapping ({type(sched_sd).__name__}); a restarted LR "
            "schedule is not an exact continuation (fail closed)."
        )
    # Non-emptiness is NOT sufficient: ``load_state_dict`` merges silently, so
    # a partial/arbitrary mapping would resume on the fresh scheduler's
    # defaults.  Require the complete CosineAnnealingLR structure, with the
    # per-group rate sequences matching the optimizer state's actual group
    # count.
    _require_complete_scheduler_state(sched_sd, ckpt_path, len(opt_groups))
    rng_state = ckpt["rng_state"]
    if not isinstance(rng_state, torch.Tensor) or rng_state.dtype != torch.uint8:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} rng_state is not a uint8 tensor "
            f"({type(rng_state).__name__}); fail closed."
        )


def _is_epoch_value(value: Any) -> bool:
    """True when *value* is a recorded scalar (finite OR non-finite NaN/Inf).

    Non-finite scalars ARE recorded epochs — an unstable epoch produced a
    real NaN/Inf diagnostic — so they are distinguished from a gap (``None``
    or a missing entry) and never treated as structural damage.
    """
    return isinstance(value, (int, float, np.integer, np.floating))


def _terminal_diagnostic_value(
    parent_value: Any,
    history_tail: Any,
    *,
    series: str,
    ckpt_path: Path,
) -> Optional[float]:
    """Resolve one terminal diagnostic (``train_loss`` / ``val_loss``).

    A zero-update finalization reports the LAST EXECUTED epoch's diagnostics.
    Two sources describe that epoch in the parent checkpoint:

    * ``parent_value`` — the scalar the parent recorded at the TOP level
      (``train_loss`` / ``val_loss``).  Absent (``None``) when the parent
      recorded no observation for that series.
    * ``history_tail`` — the final position of the restored history series:
      a recorded scalar, ``None`` (an executed-but-unobserved GAP), or absent.

    A recorded scalar WINS.  The parent's own observation must never be
    replaced by a ``None`` gap (that would discard a measured loss and make
    the finalized checkpoint claim a value the run did not measure).  A
    legitimate gap is therefore NOT a contradiction — it is filled from the
    other source.  When BOTH sources record a scalar they describe the same
    epoch and must agree; a disagreement is a structurally inconsistent
    parent and fails closed.  A malformed (non-scalar, non-gap) value fails
    closed.  ``NaN``/``Inf`` pass through verbatim — a non-finite epoch is a
    genuine recorded result, not a gap — and two ``NaN`` scalars count as
    equal.  When neither source has a value the result is ``None`` (an
    explicit unobserved value; the caller maps it to its own placeholder),
    never a fabricated ``0``.
    """
    if parent_value is None:
        parent_scalar: Optional[float] = None
    elif _is_epoch_value(parent_value) and not isinstance(parent_value, bool):
        parent_scalar = float(parent_value)
    else:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} records a non-scalar top-level "
            f"{series}={parent_value!r}; a terminal finalization cannot "
            "inherit a malformed diagnostic (fail closed)."
        )

    if history_tail is None:
        history_scalar: Optional[float] = None
    elif _is_epoch_value(history_tail) and not isinstance(history_tail, bool):
        history_scalar = float(history_tail)
    else:
        raise ValueError(
            f"Resume checkpoint {ckpt_path} history tail {series}="
            f"{history_tail!r} is neither a scalar nor a gap; fail closed."
        )

    if parent_scalar is not None and history_scalar is not None:
        _agree = (
            math.isnan(parent_scalar) and math.isnan(history_scalar)
        ) or parent_scalar == history_scalar
        if not _agree:
            raise ValueError(
                f"Resume checkpoint {ckpt_path} records {series}="
                f"{parent_scalar!r} but its history tail is "
                f"{history_scalar!r}; the parent's own diagnostics disagree "
                "(fail closed)."
            )
        return parent_scalar
    if parent_scalar is not None:
        return parent_scalar
    return history_scalar


def _terminal_train_diagnostic(
    ckpt: Dict[str, Any],
    history_tail: Any,
    *,
    ckpt_path: Path,
) -> Optional[float]:
    """Resolve the terminal TRAIN diagnostic, honouring the legacy ``loss`` field.

    The epoch train loss is recorded at the top level as ``train_loss``.  A
    LEGACY / older writer may instead carry only the independent ``loss``
    scalar (the step-level value ``save_checkpoint`` has always written).  A
    MISSING ``train_loss`` is therefore not the same as a ``None`` one:

    * ``train_loss`` ABSENT — no top-level train observation was recorded;
      the independent ``loss`` field, if present, IS the train diagnostic and
      must be kept (it is never discarded in favour of a gap).
    * ``train_loss`` PRESENT and ``None`` — the parent explicitly declared an
      unobserved train position; the legacy ``loss`` is NOT substituted for
      it (an absent field and an explicit gap are distinct).

    The chosen source then goes through ``_terminal_diagnostic_value`` so a
    recorded scalar still wins over a legitimate history gap, a malformed
    value fails closed, and ``NaN``/``Inf`` pass through verbatim.
    """
    if "train_loss" in ckpt:
        source = ckpt["train_loss"]
    else:
        source = ckpt.get("loss")
    return _terminal_diagnostic_value(
        source, history_tail, series="train_loss", ckpt_path=ckpt_path,
    )


def _recovery_state_kwargs(
    epochs_without_improvement: int,
    history: Dict[str, List[Any]],
    history_start_epoch: int,
) -> Dict[str, Any]:
    """Extra-metadata kwargs carrying the epoch-boundary recovery state.

    The diagnostic histories are persisted in FULL and IN ORDER, one entry
    per executed epoch, so a restart preserves the same epoch axis an
    uninterrupted run would have.  A non-finite scalar diagnostic (NaN/Inf
    from an unstable epoch) is written AS-IS — it is a genuine result and
    coercing it to zero or dropping it would fabricate a history the run
    never produced and collapse the epoch positions of every later value.

    Both series are written on the SAME epoch axis and stay the same length,
    but a position is not always an observation: an executed epoch with no
    validation loader records ``None`` at that validation-history position
    (the producer writes the gap explicitly rather than inventing a value).
    A ``None`` therefore means "this epoch executed, but no validation was
    observed here" — it is NOT the same as an unknown legacy position, which
    is simply absent because the prehistory was never recorded.  A reader
    keeps the epoch slot for either case rather than shortening the axis, and
    a no-validation position does not count toward the patience horizon.

    ``history_start_epoch`` is the true 0-based epoch index of
    ``training_history`` entry 0.  It is 0 for a fresh run, and equals the
    resume point for a LEGACY resume whose history was unknown: a legacy
    checkpoint completed at epoch 146 yields a first recorded entry for
    epoch 146 (not epoch 0), so the axis is never redrawn from epoch 1.

    ``epochs_without_improvement`` is persisted verbatim.  It counts only
    FINITE non-improving validation epochs (the training loop increments it
    behind ``math.isfinite(val_loss)``), so it is independent of how many
    diagnostic epochs were recorded and of how many were non-finite.
    """
    return {
        "recovery_state_version": RECOVERY_STATE_VERSION,
        "epochs_without_improvement": int(epochs_without_improvement),
        "training_history": {key: list(v) for key, v in history.items()},
        "history_start_epoch": int(history_start_epoch),
        "numpy_rng_state": _encode_numpy_rng_state(np.random.get_state()),
    }


def _restore_recovery_state(
    ckpt: Dict[str, Any], resume_path: Path, *, resume_epoch: int,
) -> Tuple[int, Dict[str, List[Any]], Optional[tuple], int]:
    """Recover epoch-boundary state from a resume checkpoint.

    Returns ``(epochs_without_improvement, history, numpy_rng_state,
    history_start_epoch)``, where ``history_start_epoch`` is the true 0-based
    epoch index of ``history`` entry 0.

    Semantics, in order:

    * **Legacy / missing** (no ``recovery_state_version``): the checkpoint
      predates recovery persistence.  The counter and history are unknown, so
      they are re-initialised fresh and a warning says exact continuation is
      not claimed.  The history is EMPTY and its start epoch is *resume_epoch*
      — the run resumes at epoch ``resume_epoch`` and its first recorded entry
      belongs to that epoch, never to epoch 0 (a legacy checkpoint completed at
      epoch 146 must not be drawn as if it began at epoch 1).
    * **Version mismatch**: the checkpoint was written by a different
      recovery schema; its counter/history semantics cannot be trusted, so the
      same fresh initialisation applies, loudly.
    * **Current version (v2)**: the FULL recorded histories are restored
      positionally, gaps (``None``) included.  A non-finite scalar is a real
      recorded epoch and is preserved AS-IS — it is never dropped or coerced
      to zero, and later epochs keep their positions.  ``history_start_epoch``
      (written alongside) plus the series length must match the checkpoint's
      completed-epoch count (``epoch + 1``); a disagreement is structural
      corruption and fails closed.
    * **Structurally malformed payload** (bad counter, wrong keys, non-scalar
      entries, offset mismatch): corruption, not a legacy gap — fail closed.
    """
    version = ckpt.get("recovery_state_version")
    if version is None:
        logger.warning(
            "Resume checkpoint %s has no recovery_state_version (legacy); "
            "epochs_without_improvement/history restart from scratch — "
            "patience and plotted history are NOT continued.  The history "
            "recorded from here starts at epoch %d, not epoch 0.",
            resume_path, resume_epoch,
        )
        return 0, {"train_loss": [], "val_loss": []}, None, int(resume_epoch)
    if isinstance(version, bool) or not isinstance(version, (int, np.integer)):
        raise ValueError(
            f"Resume checkpoint {resume_path} has invalid "
            f"recovery_state_version={version!r}; fail closed."
        )
    if int(version) != RECOVERY_STATE_VERSION:
        logger.warning(
            "Resume checkpoint %s carries recovery_state_version=%r, this "
            "code writes %d; counter/history restart from scratch.  The "
            "history recorded from here starts at epoch %d, not epoch 0.",
            resume_path, version, RECOVERY_STATE_VERSION, resume_epoch,
        )
        return 0, {"train_loss": [], "val_loss": []}, None, int(resume_epoch)

    raw_ewi = ckpt.get("epochs_without_improvement")
    if (
        isinstance(raw_ewi, bool)
        or not isinstance(raw_ewi, (int, np.integer))
        or raw_ewi < 0
    ):
        raise ValueError(
            f"Resume checkpoint {resume_path} has invalid "
            f"epochs_without_improvement={raw_ewi!r}; fail closed."
        )
    raw_history = ckpt.get("training_history")
    if (
        not isinstance(raw_history, dict)
        or set(raw_history) != {"train_loss", "val_loss"}
    ):
        bad_keys = (
            sorted(raw_history) if isinstance(raw_history, dict) else raw_history
        )
        raise ValueError(
            f"Resume checkpoint {resume_path} has invalid training_history "
            f"keys {bad_keys!r}; fail closed."
        )
    history: Dict[str, List[Any]] = {}
    for key, values in raw_history.items():
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                f"Resume checkpoint {resume_path} training_history[{key!r}] "
                f"is not a sequence ({type(values).__name__}); fail closed."
            )
        for pos, value in enumerate(values):
            if value is None or _is_epoch_value(value):
                continue
            raise ValueError(
                f"Resume checkpoint {resume_path} training_history[{key!r}]"
                f"[{pos}] is not a scalar epoch value ({value!r}); fail closed."
            )
        # NaN/Inf survive verbatim; only the container is copied.  No
        # finite-prefix truncation, so every later epoch keeps its position.
        history[key] = list(values)

    # ``train_loss`` and ``val_loss`` share one epoch axis (the loss plot
    # indexes both against ``range(1, len(train_loss)+1)``), so a mismatch
    # between their lengths would misalign the curves.  The writer keeps them
    # equal; a foreign payload that does not is structurally invalid.
    if len(history["train_loss"]) != len(history["val_loss"]):
        raise ValueError(
            f"Resume checkpoint {resume_path} history series have unequal "
            f"lengths (train_loss={len(history['train_loss'])}, "
            f"val_loss={len(history['val_loss'])}); the shared epoch axis is "
            "invalid, fail closed."
        )

    # ``history_start_epoch`` is the true 0-based epoch index of entry 0.
    # Together with the series length it must account for exactly the epochs
    # this checkpoint completed (``resume_epoch``): otherwise the recorded
    # axis is not the real one and a resume would mis-position every value
    # (a legacy checkpoint resumed at epoch 146 must record its first entry
    # as epoch 146, never epoch 0).
    raw_start = ckpt.get("history_start_epoch")
    if isinstance(raw_start, bool) or not isinstance(raw_start, (int, np.integer)):
        raise ValueError(
            f"Resume checkpoint {resume_path} has invalid history_start_epoch="
            f"{raw_start!r}; fail closed."
        )
    history_start_epoch = int(raw_start)
    if history_start_epoch < 0 or history_start_epoch > resume_epoch:
        raise ValueError(
            f"Resume checkpoint {resume_path} history_start_epoch="
            f"{history_start_epoch} is outside [0, {resume_epoch}]; fail closed."
        )
    if history_start_epoch + len(history["train_loss"]) != resume_epoch:
        raise ValueError(
            f"Resume checkpoint {resume_path} history covers epochs "
            f"[{history_start_epoch}, "
            f"{history_start_epoch + len(history['train_loss'])}) but the "
            f"checkpoint completed {resume_epoch} epoch(s); fail closed."
        )

    # The early-stopping counter counts FINITE non-improving validation
    # epochs (the loop's increment is guarded by ``math.isfinite``).  It is an
    # INDEPENDENT control value, so the upper bound must not be the number of
    # KNOWN-finite val entries alone: a ``None`` slot is a DECLARED GAP of
    # unknown finiteness — that epoch may have been finite and counted — so it
    # is a POSSIBLE patience opportunity.  Only an explicitly non-finite scalar
    # (NaN/Inf) is a KNOWN non-finite epoch that definitely did not count.
    # Bound = known-finite + unknown(None).  This keeps a true counter
    # restorable (repro: resume_epoch=3, val=[0.1, None, None], patience=2) while
    # still rejecting a counter that exceeds every epoch that could have counted.
    known_finite = 0
    unknown_slots = 0
    for v in history["val_loss"]:
        if v is None:
            unknown_slots += 1
        elif _is_epoch_value(v) and math.isfinite(float(v)):
            known_finite += 1
        # else: NaN/Inf -> known non-finite, cannot have incremented patience.
    max_possible_patience = known_finite + unknown_slots
    if int(raw_ewi) > max_possible_patience:
        raise ValueError(
            f"Resume checkpoint {resume_path} epochs_without_improvement="
            f"{int(raw_ewi)} exceeds the number of validation epochs that "
            f"could have counted ({max_possible_patience} = {known_finite} "
            f"finite + {unknown_slots} unknown); counter/history inconsistent, "
            "fail closed."
        )

    # Current-schema recovery REQUIRES a reproducible NumPy stream.  A
    # missing or malformed ``numpy_rng_state`` is damaged required state, not
    # a legacy gap, so it fails closed here — before the loop runs an epoch —
    # rather than silently reseeding a different stream (reviews R2 / B-1).
    # Legacy / version-mismatch checkpoints returned above keep their
    # warn-and-reset compatibility.
    np_state = _decode_numpy_rng_state(ckpt.get("numpy_rng_state"))
    return int(raw_ewi), history, np_state, history_start_epoch


def _resolved_loader_workers(
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
) -> Dict[str, Optional[int]]:
    """The ACTUAL worker count of each resolved loader.

    ``num_workers=-1`` auto-scales by dataset size, so the requested value in
    the config and the resolved value on the built loader can differ (and train
    and val can differ from each other).  The segment record stores the resolved
    value for BOTH loaders so an audit judges what actually ran, not what was
    asked for.  With no val loader the val count is ``None`` (explicitly
    absent), never a substituted 0.
    """
    return {
        "num_workers": int(getattr(train_loader, "num_workers", 0)),
        "val_num_workers": (
            int(getattr(val_loader, "num_workers", 0))
            if val_loader is not None else None
        ),
    }


def _require_reliable_recovery_loaders(
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    *,
    checkpoint_interval: int,
) -> None:
    """Fail closed when a RELIABLE-recovery cadence cannot load with zero workers.

    Reliable recovery is the cadence-one regime: ``checkpoint_interval == 1``
    is what makes a checkpoint an exact epoch boundary of a run whose
    trajectory is claimed reproducible.  That claim requires ZERO-WORKER
    loading.  With ``num_workers > 0`` (and especially ``persistent_workers``)
    the loader iterator owns multiprocessing worker state — the RNG stream and
    the live iterator's consumption position — that is neither captured in the
    checkpoint nor reconstructable from it.  An uninterrupted persistent
    iterator and a newly constructed resumed iterator therefore consume
    DIFFERENT global RNG histories, so a resume silently diverges even when no
    worker-side randomness is present (independent same-budget model/backprop
    probe: workers0 exact equality, persistent workers1 divergence).  Restoring
    only the parent's RNG state cannot make a new iterator equivalent to the
    existing persistent one.

    The requirement is tied to the CADENCE, not to ``--resume``:

    - ``checkpoint_interval == 1`` — reliable mode, whether fresh or resumed.
      The ACTUAL resolved train/val loaders must have zero workers, else the
      run is refused before any epoch.
    - ``checkpoint_interval != 1`` — the prior interval-10 behavior.  That
      cadence makes no exact-continuation claim (an interrupted interval-10
      run can only be continued as a fresh zero-worker segment), so no worker
      count is constrained and existing runs keep working unchanged.

    The guard never reconfigures the loader silently (that would change the
    user's scientific controls) and never claims an equivalence the
    configuration cannot provide.  ``val_loader`` is included for completeness
    of the refusal message; validation does not advance the training
    trajectory.
    """
    if int(checkpoint_interval) != 1:
        return
    resolved = {"train": int(getattr(train_loader, "num_workers", 0))}
    if val_loader is not None:
        resolved["val"] = int(getattr(val_loader, "num_workers", 0))
    nonzero = {name: n for name, n in resolved.items() if n > 0}
    if nonzero:
        persistent = getattr(train_loader, "persistent_workers", False)
        raise ValueError(
            "Reliable epoch-boundary recovery (checkpoint_interval=1) "
            f"requires zero-worker data loading, but the resolved loaders have "
            f"{nonzero} worker(s) (persistent_workers={persistent}).  With "
            "workers>0 the loader iterator's RNG/consumption state is not "
            "captured in the checkpoint, so a restart would silently diverge "
            "from an uninterrupted run.  Set training.num_workers=0 in the "
            "reliable run's config (persistent_workers is inactive at zero "
            "workers and need not be changed), or use a non-reliable cadence "
            "such as checkpoint_interval=10.  Fail closed rather than claim "
            "continuity this configuration cannot provide."
        )


def _checkpoint_lineage_identity(payload: bytes, path: Path) -> Dict[str, Any]:
    """Immutable identity of a resume checkpoint from its already-read bytes.

    The digest is taken over *payload* — the exact bytes the run resumed
    from — never a fresh read of the mutable *path*.  Re-reading would open
    a TOCTOU window in which the file is swapped between the lineage check
    and the segment record, binding the segment to bytes that were never
    loaded.  *path* is recorded for provenance only.
    """
    assert isinstance(payload, (bytes, bytearray)), type(payload)
    return {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _parent_training_controls(
    peek: Dict[str, Any], resume_path: Path,
) -> Optional[Dict[str, Any]]:
    """The parent checkpoint's ``config.training`` controls, or ``None`` if legacy.

    The parent's epoch budget, seed and checkpoint cadence are recorded under
    ``peek['config']['training']`` (the dict ``save_checkpoint`` wrote), NOT
    at the checkpoint top level.  Reading the active run's config for these
    would mislabel the parent's lineage whenever the two differ.

    Returns ``None`` only when the parent genuinely carries no ``config``
    (a legacy checkpoint): the caller then reports the controls as unknown.
    A ``config`` that is present but structurally malformed fails closed.
    """
    if "config" not in peek:
        logger.warning(
            "Resume checkpoint %s has no recorded config; parent training "
            "controls are unknown and recorded as null (not substituted "
            "from the active run).", resume_path,
        )
        return None
    parent_config = peek["config"]
    if not isinstance(parent_config, dict):
        raise ValueError(
            f"Resume checkpoint {resume_path} config is not a mapping "
            f"({type(parent_config).__name__}); fail closed."
        )
    if "training" not in parent_config:
        logger.warning(
            "Resume checkpoint %s config has no training section; parent "
            "training controls are unknown and recorded as null (not "
            "substituted from the active run).", resume_path,
        )
        return None
    training = parent_config["training"]
    if not isinstance(training, dict):
        raise ValueError(
            f"Resume checkpoint {resume_path} config.training is not a "
            f"mapping ({type(training).__name__}); fail closed."
        )
    return training


def _segment_loader_controls(
    config: ExperimentConfig,
    parent_training: Optional[Dict[str, Any]],
    resolved: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, Any]:
    """Loader controls a downstream audit needs to judge recovery fidelity.

    Records the active run's REQUESTED worker controls, the ACTUAL RESOLVED
    loader worker counts (``resolved`` — ``num_workers=-1`` auto-scales, so the
    request and the resolution can differ), and the parent's requested
    controls, and states truthfully whether the reliable zero-worker
    requirement applies to THIS segment.  Reliable mode is the cadence-one
    regime (``checkpoint_interval == 1``); a non-reliable cadence makes no
    exact-continuation claim, so the requirement is not asserted for it even
    when workers happen to be zero.  A legacy parent that predates these
    fields records ``None`` (unknown) rather than the active value.
    """
    def _parent_int(key: str) -> Optional[int]:
        if not parent_training or key not in parent_training:
            return None
        value = parent_training[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            return None
        return int(value)

    def _parent_bool(key: str) -> Optional[bool]:
        # ``persistent_workers`` is a bool in the parent's config; decode it
        # explicitly so a legitimate ``True`` is recorded as True, not dropped
        # to None by an int-only decoder.
        if not parent_training or key not in parent_training:
            return None
        value = parent_training[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        return None

    reliable = int(config.training.checkpoint_interval) == 1
    resolved = resolved or {}
    return {
        "requested_num_workers": int(config.training.num_workers),
        "requested_persistent_workers": bool(config.training.persistent_workers),
        "resolved_num_workers": (
            int(resolved["num_workers"]) if "num_workers" in resolved else None
        ),
        "resolved_val_num_workers": (
            int(resolved["val_num_workers"])
            if resolved.get("val_num_workers") is not None else None
        ),
        "parent_num_workers": _parent_int("num_workers"),
        "parent_persistent_workers": _parent_bool("persistent_workers"),
        # Reliable exact continuation is only supported with zero workers and
        # is only CLAIMED at cadence one; a nonzero parent->zero active change
        # is NOT established as trajectory-equivalent to the parent's original
        # execution.
        "reliable_mode": reliable,
        "reliable_zero_worker_required": reliable,
        "legacy_worker_equivalence_established": False,
    }


def _segment_provenance(
    peek: Optional[Dict[str, Any]],
    config: ExperimentConfig,
    *,
    resumed: bool,
    resume_path: Optional[Path] = None,
    resolved_workers: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, Any]:
    """Data/prior controls a downstream audit needs to compare segments.

    On a restart the parent controls (epoch budget, seed, checkpoint cadence)
    are read from the parent checkpoint's already-decoded *peek*, specifically
    ``peek['config']['training']`` — never from the active run config, which
    may differ and would silently mislabel the lineage.  The active run's own
    controls are recorded separately under ``current_segment_controls`` so the
    two are never confused.  Data/prior fields come from the parent's decoded
    top level.  Never re-reads the resume path and never reads formal
    DATA/NEST artifacts.

    Fail closed: a malformed parent control fails the segment; a genuinely
    legacy parent (no ``config``) records the controls as ``null``/unknown
    rather than substituting the active run's values.
    """
    provenance: Dict[str, Any] = {
        "original_num_epochs": None,
        "random_seed": None,
        "dataset_path": None,
        "dataset_source_sha256": None,
        "nested_prior_artifact": None,
        "nested_prior_artifact_sha256": None,
        "nested_prior_fingerprint": None,
        "nested_split_seed": None,
        "nested_val_split": None,
        "is_nested_cv": None,
        "validation_scope": None,
        "animal_identity_status": None,
        "mcmc_prior_provenance": None,
        "checkpoint_interval": None,
    }
    if not resumed:
        provenance["original_num_epochs"] = int(config.training.num_epochs)
        provenance["random_seed"] = int(config.training.random_seed)
        provenance["checkpoint_interval"] = int(config.training.checkpoint_interval)
        provenance["current_segment_controls"] = {
            "num_epochs": int(config.training.num_epochs),
            "random_seed": int(config.training.random_seed),
            "checkpoint_interval": int(config.training.checkpoint_interval),
            "loader_controls": _segment_loader_controls(
                config, None, resolved_workers,
            ),
        }
        return provenance

    if peek is None:
        raise ValueError(
            "Cannot record segment provenance: the resume checkpoint's "
            "decoded controls are unavailable. Refusing to substitute the "
            "active run's controls (which may differ from the parent's); "
            "fail closed."
        )
    if resume_path is None:
        raise ValueError(
            "Cannot record segment provenance for a resume without the "
            "resume checkpoint path; fail closed."
        )

    # Data/prior fields live at the parent checkpoint's top level.
    for key in provenance:
        if key in peek:
            provenance[key] = peek[key]

    training = _parent_training_controls(peek, resume_path)
    if training is not None:
        for record_key, config_key in (
            ("original_num_epochs", "num_epochs"),
            ("random_seed", "random_seed"),
            ("checkpoint_interval", "checkpoint_interval"),
        ):
            if config_key not in training:
                # Legacy parent config predating this control: report
                # unknown, never the active run's value.
                logger.warning(
                    "Resume checkpoint %s config.training lacks %r; parent "
                    "control recorded as null.", resume_path, config_key,
                )
                continue
            value = training[config_key]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(
                    f"Resume checkpoint {resume_path} config.training"
                    f"[{config_key!r}]={value!r} is not an integer; fail closed."
                )
            provenance[record_key] = int(value)

    provenance["current_segment_controls"] = {
        "num_epochs": int(config.training.num_epochs),
        "random_seed": int(config.training.random_seed),
        "checkpoint_interval": int(config.training.checkpoint_interval),
        "loader_controls": _segment_loader_controls(
            config, training, resolved_workers,
        ),
    }
    return provenance


def _write_segment_record(
    output_dir: Path,
    *,
    parent: Optional[Dict[str, Any]],
    config: ExperimentConfig,
    parent_peek: Optional[Dict[str, Any]],
    start_epoch: int,
    resumed: bool,
    resume_path: Optional[Path] = None,
    resolved_workers: Optional[Dict[str, Optional[int]]] = None,
) -> Path:
    """Write one immutable restart-segment record; never overwrite a prior one.

    Each invocation claims the next free ``segment_XXXX.json`` so prior
    segments — and their distinct parent identities — survive every restart,
    giving a downstream audit the full chain of who continued from which
    bytes, under which data/prior controls, at which epoch.

    Crash-atomicity is scoped to PROCESS death (SIGKILL / OOM / crash): the
    payload is written to a unique temp file in the SAME directory, flushed +
    fsynced, then ``os.link``ed into the final name.  The link fails with
    ``FileExistsError`` if the name is already taken (immutable,
    non-overwrite), so a SIGKILL mid-write leaves only an orphaned ``.tmp``
    that no later invocation treats as a completed segment.  This mirrors the
    temp+fsync discipline of :func:`_atomic_save_checkpoint`.

    POWER-LOSS durability is NOT claimed: the containing directory is not
    fsynced (``os.link`` only makes the name visible in the page cache), so a
    machine power-cut shortly after the link could lose the directory entry
    even though the file contents were synced.  That is acceptable here —
    the segment record is an audit trail, and a missing trailing segment
    under-claims rather than fabricates lineage.  The ``.tmp`` is unlinked
    after the link regardless, so no completed record is ever truncated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_record = {
        "segment_index": None,  # patched per attempt once a slot is claimed
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resumed": resumed,
        "starting_epoch": int(start_epoch),
        "parent_checkpoint": parent,
        "config_sha256": hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True, default=str).encode()
        ).hexdigest(),
        "provenance": _segment_provenance(
            parent_peek, config, resumed=resumed, resume_path=resume_path,
            resolved_workers=resolved_workers,
        ),
    }
    # Serialize once BEFORE claiming a slot so a serialization failure
    # cannot strand a claimed-but-empty final name.
    json.dumps(base_record, indent=2, allow_nan=False)

    index = 0
    while True:
        record = dict(base_record, segment_index=index)
        record_path = output_dir / f"segment_{index:04d}.json"
        payload = json.dumps(record, indent=2, allow_nan=False).encode("utf-8")
        # A FRESH temp inode per attempt (never reused across links: a hard
        # link shares the inode, so rewriting a linked temp would corrupt an
        # already-published segment).  Write+fsync the FULL payload, then
        # link it into the final name atomically.  A SIGKILL at any point
        # leaves only an orphaned ``.tmp`` — never a truncated
        # ``segment_XXXX.json``.
        tmp_path = output_dir / f".segment_{os.getpid()}_{time.time_ns()}.tmp"
        fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        try:
            os.link(tmp_path, record_path)
        except FileExistsError:
            index += 1
            continue
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        break
    return record_path


def _atomic_save_checkpoint(**kwargs) -> Path:
    """Atomic wrapper around the frozen :func:`save_checkpoint`.

    Writes to a temporary file in the same directory, fsyncs, then
    atomically replaces the target via :func:`os.replace`.  This
    prevents a truncated ``.pth`` (from a SIGKILL / OOM during
    ``torch.save``) from overwriting a previously valid checkpoint.

    The frozen ``nsmor/checkpoint.py`` calls ``torch.save(state, path)``
    directly — no atomic semantics.  Rather than editing the frozen
    module, we redirect the ``path`` keyword to a temp file and rename
    after the save completes.

    Deployment provenance fields (``target_mean``, ``target_std``,
    ``target_clip_cm_s``, ``training_phase``, ``dataset_path``) are
    popped from *kwargs* before forwarding to :func:`save_checkpoint`
    (which has a fixed signature), then patched into the on-disk state
    dict before the fsync + atomic rename.  This keeps the frozen
    ``nsmor/checkpoint.py`` untouched while guaranteeing every
    checkpoint carries the provenance metadata a downstream consumer
    needs to correctly rescale predictions and audit lineage.
    """
    target = Path(kwargs["path"])
    tmp = target.with_suffix(".pth.tmp")
    kwargs["path"] = tmp

    # Pop provenance fields before forwarding to frozen save_checkpoint.
    provenance: Dict[str, Any] = {}
    for key in _PROVENANCE_KEYS:
        if key in kwargs:
            provenance[key] = kwargs.pop(key)

    save_checkpoint(**kwargs)

    # Patch provenance fields into the saved state dict on disk.
    if provenance:
        state = load_artifact_bytes(tmp.read_bytes(), map_location="cpu")
        state.update(provenance)
        torch.save(state, tmp)

    # fsync the written bytes so the OS page-cache is flushed before the
    # atomic rename — protects against power-loss on non-journaled FS.
    # Must sync the directory entry after the file is closed.
    try:
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Windows may not support fsync on all filesystem types — skip it
        pass
    os.replace(tmp, target)
    return target


def _check_checkpoint_prior_lineage(ckpt: Dict[str, Any], expected: Dict[str, Any], path: Path, *,
                                    trusted_historical_checkpoint_sha256=None,
                                    checkpoint_sha256=None) -> None:
    """Refuse a changed prior source or animal-identity claim on resume/best selection."""
    status = expected["animal_identity_status"]
    recorded_status = ckpt.get("animal_identity_status", "historical_unknown")
    if recorded_status != status:
        raise ValueError(f"Checkpoint at {path} animal_identity_status mismatch; fail closed")
    if status == "historical_unknown" and not expected.get("is_nested_cv", False):
        require_trusted_historical_checkpoint_sha256(
            checkpoint_sha256, trusted_historical_checkpoint_sha256,
        )
    if status == "unverified":
        expected_digest = expected.get("dataset_source_sha256")
        if expected_digest is None or ckpt.get("dataset_source_sha256") != expected_digest:
            raise ValueError(f"modern checkpoint at {path} requires matching dataset_source_sha256; fail closed")
        if ckpt.get("dataset_source_binding", "sha256_bound") != "sha256_bound":
            raise ValueError(f"modern checkpoint at {path} requires sha256_bound dataset_source_binding; fail closed")
    expected_scope = expected.get("validation_scope")
    if expected_scope is not None and ckpt.get("validation_scope") != expected_scope:
        if not (status == "historical_unknown" and not expected.get("is_nested_cv", False)
                and ckpt.get("validation_scope") is None):
            raise ValueError(f"Checkpoint at {path} validation_scope mismatch; fail closed")
    recorded_tag = ckpt.get("mcmc_prior_provenance")
    if recorded_tag != expected["mcmc_prior_provenance"]:
        from nsmor.pipeline.grouping import prior_identity_status
        legacy_alias = (
            status == "historical_unknown"
            and ckpt.get("is_nested_cv", False) is False
            and expected.get("is_nested_cv", False) is False
            and recorded_tag in (None, "global_oof_animal_grouped_cv")
            and (expected["mcmc_prior_provenance"] == "global_oof_animal_grouped_cv"
                 or prior_identity_status(expected["mcmc_prior_provenance"]) == "historical_unknown")
        )
        if not legacy_alias:
            raise ValueError(f"Checkpoint at {path} mcmc_prior_provenance mismatch; fail closed")


def _validate_best_checkpoint(
    ckpt: Dict[str, Any],
    path: Path,
    expected_lineage: Dict[str, Any],
    *,
    trusted_historical_checkpoint_sha256=None,
    checkpoint_sha256=None,
    require_best: bool = True,
) -> Optional[float]:
    """Validate checkpoint lineage and, for best selection, its coupled score.

    Enforces:
    1. Lineage consistency (is_nested_cv, artifact/content SHA-256, source fingerprint, split_seed, val_split).
    2. Finite OWN validation metric (ckpt['val_loss']).
    3. Scalar-weight coupling: ckpt['best_val_loss'] must equal ckpt['val_loss'] (or be absent).
       A historical scalar cannot be attached to different candidate tensors.

    Returns:
        Certified validation loss float.

    Raises:
        ValueError: If checkpoint fails lineage, finite metric, or scalar-weight coupling checks.
    """
    ckpt_is_nested = ckpt.get("is_nested_cv", False)
    exp_is_nested = expected_lineage.get("is_nested_cv", False)
    if ckpt_is_nested != exp_is_nested:
        raise ValueError(
            f"Checkpoint at {path} has mismatched is_nested_cv "
            f"({ckpt_is_nested} vs {exp_is_nested}); fail closed."
        )
    _check_checkpoint_prior_lineage(
        ckpt, expected_lineage, path,
        trusted_historical_checkpoint_sha256=trusted_historical_checkpoint_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    if exp_is_nested:
        for mkey in (
            "nested_prior_fingerprint",
            "nested_prior_artifact_sha256",
            "nested_split_seed",
            "nested_val_split",
            "nested_prior_artifact",
        ):
            c_val = ckpt.get(mkey)
            e_val = expected_lineage.get(mkey)
            if c_val is None:
                raise ValueError(
                    f"Checkpoint at {path} is missing mandatory nested provenance key {mkey!r}; fail closed."
                )
            if mkey == "nested_prior_artifact":
                if Path(c_val).resolve() != Path(e_val).resolve():
                    raise ValueError(
                        f"Checkpoint at {path} has mismatched nested prior artifact "
                        f"({c_val!r} vs {e_val!r}); fail closed."
                    )
            elif mkey == "nested_val_split":
                try:
                    c_f = float(c_val)
                    e_f = float(e_val)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Checkpoint at {path} has non-numeric nested val_split ({c_val!r}); fail closed."
                    )
                if not math.isfinite(c_f) or not math.isfinite(e_f) or abs(c_f - e_f) > 1e-6:
                    raise ValueError(
                        f"Checkpoint at {path} has mismatched or non-finite nested val_split "
                        f"({c_val} vs {e_val}); fail closed."
                    )
            elif mkey == "nested_split_seed":
                if isinstance(c_val, float) and not c_val.is_integer():
                    raise ValueError(
                        f"Checkpoint at {path} has fractional nested split_seed ({c_val!r}); fail closed."
                    )
                try:
                    c_int = int(c_val)
                    e_int = int(e_val)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Checkpoint at {path} has non-integer nested split_seed ({c_val!r}); fail closed."
                    )
                if float(c_val) != float(c_int) or c_int != e_int:
                    raise ValueError(
                        f"Checkpoint at {path} has mismatched nested split_seed "
                        f"({c_val} vs {e_val}); fail closed."
                    )
            else:
                if str(c_val) != str(e_val):
                    raise ValueError(
                        f"Checkpoint at {path} has mismatched lineage key {mkey} "
                        f"({c_val!r} vs {e_val!r}); fail closed."
                    )
    else:
        expected_digest = expected_lineage.get("dataset_source_sha256")
        if "dataset_source_sha256" in ckpt:
            if ckpt["dataset_source_sha256"] != expected_digest:
                raise ValueError(f"Checkpoint at {path} dataset_source_sha256 mismatch; fail closed.")
        elif expected_lineage.get("dataset_source_binding") not in ("legacy_resume_unbound", "unbound_lazy"):
            raise ValueError(f"Checkpoint at {path} missing dataset_source_sha256; fail closed.")
        else:
            logger.warning("Best checkpoint %s has unbound dataset content", path)
        candidate_binding = ckpt.get(
            "dataset_source_binding",
            "sha256_bound" if "dataset_source_sha256" in ckpt else "legacy_resume_unbound",
        )
        if candidate_binding != expected_lineage.get("dataset_source_binding", "sha256_bound"):
            raise ValueError(f"Checkpoint at {path} dataset_source_binding mismatch; fail closed.")
        exp_ds = expected_lineage.get("dataset_path")
        c_ds = ckpt.get("dataset_path")
        if exp_ds is not None and c_ds is not None:
            if Path(c_ds).resolve() != Path(exp_ds).resolve():
                raise ValueError(
                    f"Checkpoint at {path} has mismatched dataset_path "
                    f"({c_ds!r} vs {exp_ds!r}); fail closed."
                )

    if not require_best:
        return None

    own_val = ckpt.get("val_loss")
    if own_val is None or not math.isfinite(float(own_val)):
        raise ValueError(
            f"Checkpoint at {path} lacks finite OWN validation loss metric (val_loss={own_val}); fail closed."
        )
    own_val_f = float(own_val)

    reported_best = ckpt.get("best_val_loss")
    if reported_best is not None:
        if not math.isfinite(float(reported_best)):
            raise ValueError(
                f"Checkpoint at {path} has non-finite best_val_loss={reported_best}; fail closed."
            )
        reported_best_f = float(reported_best)
        if abs(own_val_f - reported_best_f) > 1e-5:
            raise ValueError(
                f"Checkpoint at {path} is not a certified best model: OWN val_loss "
                f"({own_val_f:.6f}) differs from best_val_loss ({reported_best_f:.6f}); "
                "filename or scalar cannot certify non-best weights (fail closed)."
            )

    return own_val_f

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Opt-in escape-band sensitivity sweep list, set by build_config() from
# --sweep_escape_band and consumed by train().  ``None`` disables (default).
_SWEEP_BANDS: Optional[List[float]] = None

# ── Phase-2 controls, resolved by build_config() ──────────────
# Kept as script-owned module state (NOT added to the protected
# ``nsmor/config_parser.py`` schema, so ``ExperimentConfig.to_dict()`` and
# every existing config/checkpoint stay byte-unchanged).  Both default to the
# current behaviour: ``()`` = no column zeroed, ``"total"`` = full-objective
# selection.  build_config() sets them from the CLI/YAML; tests may set them
# directly or pass the explicit arguments to build_dataloaders()/validate().
_ZERO_INPUT_CHANNELS: Tuple[int, ...] = ()
_SELECTION_METRIC: str = "total"


def _parse_selection_metric(value: Optional[str]) -> str:
    """Validate a ``--selection_metric`` value; ``None`` keeps ``'total'``."""
    if value is None:
        return "total"
    if value not in ("total", "mse"):
        raise ValueError(
            f"--selection_metric must be 'total' or 'mse', got {value!r}"
        )
    return value


def _read_phase2_block(config_path: Optional[str]) -> Dict[str, Any]:
    """Read the script-owned ``phase2:`` block from a config YAML.

    The two phase-2 controls live OUTSIDE the protected
    ``nsmor/config_parser.py`` schema, so they cannot ride the dataclass YAML
    keys.  They are declared in a top-level ``phase2:`` mapping instead, which
    ``ExperimentConfig.from_yaml`` ignores (unknown top-level keys are dropped)
    and this reader consumes.  A missing file or block yields ``{}`` — the
    default-off behaviour.
    """
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = raw.get("phase2", {})
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(
            f"phase2: block in {config_path} must be a mapping, got "
            f"{type(block).__name__}"
        )
    return block


def _parse_zero_input_channels(
    value: Optional[Union[str, Sequence[int]]], config: ExperimentConfig,
) -> Tuple[int, ...]:
    """Parse and validate a ``zero_input_channels`` declaration.

    *value* is either a comma-separated CLI string (e.g. ``"2,3"``) or a
    sequence of ints from a YAML ``phase2:`` block.  Each entry must be an
    integer indexing a *physical sensory* column ``[0, sensory_dim)``.  The
    MCMC prior columns (``sensory_dim`` onward) are refused: they are
    validated to form a probability simplex and are never part of the history
    ablation.  ``None`` (nothing declared) yields ``()``.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        raw: List[Any] = [tok.strip() for tok in value.split(",")]
        raw = [tok for tok in raw if tok]
    elif isinstance(value, Sequence):
        raw = list(value)
    else:
        raise ValueError(
            "zero_input_channels must be a comma string or a list of ints, "
            f"got {type(value).__name__}"
        )
    channels: List[int] = []
    for tok in raw:
        if isinstance(value, str):
            # CLI comma string: coerce each token to an int (rejecting bools
            # and floats via the int() contract).
            if isinstance(tok, bool):
                raise ValueError(
                    f"zero_input_channels entry {tok!r} must be an integer"
                )
            try:
                ch = int(tok)
            except ValueError as exc:
                raise ValueError(
                    f"zero_input_channels token {tok!r} is not an integer"
                ) from exc
        else:
            if isinstance(tok, bool) or not isinstance(tok, int):
                raise ValueError(
                    f"zero_input_channels entry {tok!r} must be an integer"
                )
            ch = int(tok)
        if not 0 <= ch < config.model.sensory_dim:
            raise ValueError(
                f"zero_input_channels column {ch} must index a physical "
                f"sensory column in [0, {config.model.sensory_dim})"
            )
        if ch in channels:
            raise ValueError(
                f"zero_input_channels contains a duplicate column {ch}"
            )
        channels.append(ch)
    return tuple(channels)


# Default split for the loader and public standalone target-stat helper.
# train() fits eager statistics directly from its selected loader dataset.
_VAL_SPLIT = 0.2

# Default dataset path, overridden by ``--dataset`` for the current ETL output.
_DATASET_PATH = "data/processed/nsmor_dataset.pt"


# ═══════════════════════════════════════════════════════════════
# 1.  Argument Parsing
# ═══════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the training script.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="NSMoR Training Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config file ───────────────────────────────────────────
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Processed dataset to train on (default: "
             "data/processed/nsmor_dataset.pt).  The pipeline passes the "
             "dataset produced by the ETL stage of the same run.",
    )
    parser.add_argument(
        "--lazy_loading",
        action="store_true",
        help="Use ELT mode with lazy loading (requires metadata file from prepare_metadata.py). "
             "Dramatically reduces memory usage for large datasets.",
    )
    parser.add_argument(
        "--nested_prior_artifact",
        type=str,
        default=None,
        help="Optional nested-prior artifact (.pt) from evaluate_nested_prior.py. "
             "When set, train/val indices and MCMC priors come EXACTLY from the "
             "artifact (fail-closed on fingerprint/split mismatch) instead of "
             "recomputing a recording-prefix grouped split over global OOF priors.",
    )

    parser.add_argument(
        "--trusted_historical_checkpoint_sha256", action="append", default=None,
        help="Independent SHA-256 pin for each historical checkpoint used for resume/best selection; repeat for a separate best file.",
    )
    parser.add_argument(
        "--trusted_historical_artifact_sha256", type=str, default=None,
        help="Independently pinned SHA-256 of historical nested artifact bytes.",
    )
    parser.add_argument(
        "--diagnostic_only",
        action="store_true",
        help="Allow global OOF diagnostic validation without a nested artifact. "
             "Scores can contain outer-validation label leakage and are "
             "ineligible for strict QC/release gates.",
    )

    # ── Training overrides ────────────────────────────────────
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Crop sequences longer than this (cuDNN compatibility). 0 = disable.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Override hidden dimension.",
    )

    # ── Loss function ─────────────────────────────────────────
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=None,
        help="Router regularization weight for BioJointLoss. "
             "Overrides config.loss.lambda_reg when set.",
    )
    parser.add_argument(
        "--lambda_energy",
        type=float,
        default=0.0,
        help="ATP metabolic cost weight. Penalizes mean firing rate "
             "(Attwell & Laughlin 2001). 0 disables.",
    )
    parser.add_argument(
        "--lambda_sparse",
        type=float,
        default=0.0,
        help="Population sparsity L1 weight. Encourages target firing "
             "rate (Olshausen & Field 1996). 0 disables.",
    )
    parser.add_argument(
        "--lambda_routing_aux",
        type=float,
        default=None,
        help="Auxiliary routing differentiation weight. "
             "Overrides config.loss.lambda_routing_aux when set.",
    )
    parser.add_argument(
        "--target_rate",
        type=float,
        default=0.05,
        help="Target mean firing rate for sparsity L1 loss (default: 0.05).",
    )

    # ── Fine-tuning ───────────────────────────────────────────
    parser.add_argument(
        "--freeze",
        nargs="+",
        default=None,
        metavar="MODULE",
        help="Sub-modules to freeze (e.g. lif_cell router).",
    )

    # ── Two-phase training (Hybrid Funnel) ────────────────────
    parser.add_argument(
        "--phase1_epochs",
        type=int,
        default=None,
        help="Phase 1 epochs: train frontend only (MSE loss). "
             "Phase 2 runs for remaining epochs. 0 = skip phase 1. "
             "None = single-phase (backward compatible).",
    )

    # ── LR schedule (training-stability refactor) ──────────────
    # Linear warmup for the *main* MSE path.  Under a shared
    # AdamW across the coupled LIF + GRU parameter groups, taking
    # a full-LR step on the very first epochs (where recurrent
    # states and surrogate gradients are still settling) triggers
    # overshooting that the constant cosine schedule then cannot
    # recover within a short run.  A brief linear ramp keeps the
    # early updates small and clean.
    parser.add_argument(
        "--lr_warmup_epochs",
        type=int,
        default=None,
        help="Number of epochs over which the base LR is ramped "
             "linearly from 0 to its full value. 0 disables. "
             "Overrides config.training.lr_warmup_epochs.",
    )

    # ── Target normalization ───────────────────────────────────
    parser.add_argument(
        "--normalize_targets",
        default=None,
        action="store_true",
        help="Regress on mean-centered, std-scaled velocity instead of raw "
             "cm/s.  The raw target is heavy-tailed (a few frames reach "
             "~1e7 cm/s) which inflates and destabilises the masked MSE. "
             "Normalising reveals the resting-mode bulk where the escape "
             "response lives. Statistics are fit on training split only.",
    )
    parser.add_argument(
        "--no-normalize_targets",
        dest="normalize_targets",
        default=None,
        action="store_false",
        help="Disable target normalization (use raw velocity).",
    )
    parser.add_argument(
        "--target_clip_cm_s",
        type=float,
        default=None,
        help="Clip |velocity target| to this value (cm/s) before computing "
             "the loss. 0 disables.  Removes tracking-artifact frames whose "
             "huge |y| (>1e6 cm/s) otherwise dominate the masked MSE.",
    )
    parser.add_argument(
        "--persistence_skip",
        type=float,
        default=None,
        help="Fixed causal persistence skip scalar k in [0, 1]. 0 disables (default). "
             "Restricted to normalize_targets=False and target_clip_cm_s=0.0.",
    )
    parser.add_argument(
        "--zero_input_channels",
        type=str,
        default=None,
        help="Comma-separated per-frame sensory feature columns to force to 0.0 "
             "(history ablation), e.g. '2,3'. Default: none (no column zeroed). "
             "Only physical sensory columns [0, sensory_dim) are addressable.",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default=None,
        choices=["total", "mse"],
        help="Metric that selects best_model.pth and drives early stopping: "
             "'total' (default, full validation objective) or 'mse' (masked "
             "MSE term alone over the ALIGNED eligible frames, t>=1).",
    )

    # ── Adaptive latent refinement (architecture v1) ──────────
    parser.add_argument(
        "--refinement_mode",
        type=str,
        default=None,
        choices=["off", "fixed", "adaptive"],
        help="Adaptive latent refinement mode: off (default), fixed, or "
             "adaptive. Raw JAX/Flax backends reject an enabled mode.",
    )
    parser.add_argument(
        "--refinement_max_steps",
        type=int,
        default=None,
        help="Maximum internal refinement depth K (>= 1). Default 4.",
    )
    parser.add_argument(
        "--refinement_eps",
        type=float,
        default=None,
        help="ACT halting threshold in (0, 1). Default 0.01.",
    )
    parser.add_argument(
        "--refinement_update_scale",
        type=float,
        default=None,
        help="Bound on the per-step residual update (> 0). Default 1.0.",
    )
    parser.add_argument(
        "--lambda_compute",
        type=float,
        default=None,
        help="Adaptive-refinement compute-cost weight (>= 0). Default 0.0. "
             "Required > 0 for adaptive training.",
    )

    # ── Checkpointing ─────────────────────────────────────────
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=None,
        help="Override periodic checkpoint interval (epochs).",
    )
    parser.add_argument(
        "--sweep_escape_band",
        type=str,
        default=None,
        help="Comma-separated escape-band thresholds (cm/s) for the "
             "sensitivity sweep, e.g. '5,10,20,50'.  After training, "
             "rescores the best model's validation metrics at every band "
             "x min_run in {1,2,3} and writes "
             "escape_sensitivity.csv to the output dir.  Requires a "
             "completed training run with a val split.",
    )

    return parser


def build_config(argv: Optional[Sequence[str]] = None) -> Tuple[ExperimentConfig, float, Optional[int]]:
    """
    Parse CLI arguments and return a fully resolved config, lambda_reg,
    and phase1_epochs.

    If ``--config`` is given, YAML is loaded first, then CLI flags
    override individual values.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        ``(config, lambda_reg, phase1_epochs)`` tuple.
        ``phase1_epochs`` is ``None`` when two-phase training is disabled
        (backward-compatible single-phase mode).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # ── Load base config ──────────────────────────────────────
    if args.config is not None:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()

    # ── Apply CLI overrides ───────────────────────────────────
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if getattr(args, "max_seq_len", None) is not None:
        config.training.max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.hidden_dim is not None:
        config.model.hidden_dim = args.hidden_dim
    if getattr(args, "lr_warmup_epochs", None) is not None:
        config.training.lr_warmup_epochs = args.lr_warmup_epochs
    if getattr(args, "normalize_targets", None) is not None:
        config.training.normalize_targets = args.normalize_targets
    if getattr(args, "target_clip_cm_s", None) is not None:
        config.training.target_clip_cm_s = args.target_clip_cm_s
    if getattr(args, "persistence_skip", None) is not None:
        config.model.persistence_skip = args.persistence_skip
    # ── Phase-2 controls (kept OUT of the protected config schema) ──
    # ``data.zero_input_channels`` (history ablation) and
    # ``checkpoint.selection_metric`` (MSE-based checkpoint selection) are
    # resolved here into script-owned module state rather than added to the
    # protected ``nsmor/config_parser.py`` dataclasses.  Both default to the
    # current behaviour, so ``ExperimentConfig.to_dict()`` — and hence every
    # existing config and checkpoint — is byte-unchanged.  They are consumed
    # by ``build_dataloaders`` (ablation), ``train`` (selection) and recorded
    # as additive checkpoint provenance keys.
    global _ZERO_INPUT_CHANNELS, _SELECTION_METRIC
    phase2 = _read_phase2_block(args.config)
    _zero_decl = getattr(args, "zero_input_channels", None)
    if _zero_decl is None:
        _zero_decl = phase2.get("zero_input_channels")
    _ZERO_INPUT_CHANNELS = _parse_zero_input_channels(_zero_decl, config)
    _sel_decl = getattr(args, "selection_metric", None)
    if _sel_decl is None:
        _sel_decl = phase2.get("selection_metric")
    _SELECTION_METRIC = _parse_selection_metric(_sel_decl)
    if getattr(args, "refinement_mode", None) is not None:
        config.model.refinement_mode = args.refinement_mode
    if getattr(args, "refinement_max_steps", None) is not None:
        config.model.refinement_max_steps = args.refinement_max_steps
    if getattr(args, "refinement_eps", None) is not None:
        config.model.refinement_eps = args.refinement_eps
    if getattr(args, "refinement_update_scale", None) is not None:
        config.model.refinement_update_scale = args.refinement_update_scale
    if getattr(args, "lambda_compute", None) is not None:
        config.loss.lambda_compute = args.lambda_compute
    config.validate()
    if args.freeze is not None:
        config.finetune.freeze_modules = args.freeze
    if args.resume is not None:
        config.checkpoint.resume_from = args.resume
    if args.output_dir is not None:
        config.checkpoint.output_dir = args.output_dir
    if getattr(args, "checkpoint_interval", None) is not None:
        config.training.checkpoint_interval = args.checkpoint_interval
    if getattr(args, "lambda_routing_aux", None) is not None:
        config.loss.lambda_routing_aux = args.lambda_routing_aux

    # Sweep bands: parse the comma list into a module-level holder consumed
    # by train() (kept out of ExperimentConfig — it is a reporting option,
    # not a training hyperparameter).
    global _SWEEP_BANDS
    _SWEEP_BANDS = (
        [float(b) for b in args.sweep_escape_band.split(",") if b.strip()]
        if args.sweep_escape_band else None
    )

    # Resolve lambda_reg: CLI override > YAML config > dataclass default.
    # The CLI default is None (sentinel), so an unset flag falls through
    # to config.loss.lambda_reg (which the YAML or dataclass populates).
    lambda_reg = args.lambda_reg if args.lambda_reg is not None else config.loss.lambda_reg

    return config, lambda_reg, args.phase1_epochs


# ═══════════════════════════════════════════════════════════════
# 2.  Model / Optimizer / Loss Factory
# ═══════════════════════════════════════════════════════════════

def build_model(config: ExperimentConfig) -> NSMoRCore:
    """
    Construct a :class:`NSMoRCore` from the experiment config.

    Args:
        config: Parsed experiment configuration.

    Returns:
        Instantiated model (on CPU; move to device after).
    """
    validate_dt_ms(config.model.dt_ms, name="active config dt_ms")
    model = NSMoRCore(
        sensory_dim=config.model.sensory_dim,
        mcmc_dim=config.model.mcmc_dim,
        hidden_dim=config.model.hidden_dim,
        num_gru_layers=config.model.num_gru_layers,
        dropout=config.model.dropout,
        dt_ms=config.model.dt_ms,
        lif_alpha=config.model.lif_alpha,
        lif_threshold=config.model.lif_threshold,
        lif_beta=config.model.lif_beta,
        lif_abs_refract_ms=config.model.lif_abs_refract_ms,
        lif_rel_refract_ms=config.model.lif_rel_refract_ms,
        lif_tau_syn=config.model.lif_tau_syn,
        lif_v_rest=config.model.lif_v_rest,
        lif_v_reset=config.model.lif_v_reset,
        lif_tau_w=config.model.lif_tau_w,
        lif_b_adapt=config.model.lif_b_adapt,
        lif_tau_fac=config.model.lif_tau_fac,
        lif_tau_rec=config.model.lif_tau_rec,
        lif_U_stp_init=config.model.lif_U_stp_init,
        lif_lateral_inhibition=config.model.lif_lateral_inhibition,
        lif_inhib_tau_ms=config.model.lif_inhib_tau_ms,
        lif_dendritic_tau=config.model.lif_dendritic_tau,
        gru_neuromod_gain=config.model.gru_neuromod_gain,
        sensory_noise_std=config.model.sensory_noise_std,
        lif_tbptt_steps=config.model.lif_tbptt_steps,
        persistence_skip=config.model.persistence_skip,
        activation=config.model.activation,
        refinement_mode=config.model.refinement_mode,
        refinement_max_steps=config.model.refinement_max_steps,
        refinement_eps=config.model.refinement_eps,
        refinement_update_scale=config.model.refinement_update_scale,
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model initialized — %s parameters", f"{param_count:,}")
    return model


def build_optimizer(
    model: nn.Module,
    config: ExperimentConfig,
) -> torch.optim.AdamW:
    """
    Construct an ``AdamW`` optimizer with per-pathway learning rates.

    CF7 fix: The LIF pathway has discrete (spike) outputs, making its
    loss landscape highly sensitive to parameter perturbations.  A
    single LR for all parameters causes either LIF instability (LR too
    high) or GRU underfitting (LR too low).  Separate parameter groups
    with 0.3x LR for LIF parameters resolve this trade-off.

    Args:
        model: The model whose parameters to optimize.
        config: Parsed experiment configuration.

    Returns:
        Configured AdamW optimizer.
    """
    base_lr = config.training.learning_rate
    lif_lr = base_lr * 0.3  # Lower LR for spiking pathway

    lif_params = list(model.lif_cell.parameters())
    lif_param_ids = {id(p) for p in lif_params}
    other_params = [p for p in model.parameters() if id(p) not in lif_param_ids]

    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": base_lr, "base_lr": base_lr, "name": "non_lif"},
        {"params": lif_params, "lr": lif_lr, "base_lr": lif_lr, "name": "lif"},
    ], weight_decay=config.training.weight_decay)
    logger.info(
        "Optimizer: AdamW  base_lr=%.2e  lif_lr=%.2e  weight_decay=%.2e",
        base_lr, lif_lr, config.training.weight_decay,
    )
    return optimizer


def build_loss(config: ExperimentConfig) -> BioJointLoss:
    """
    Construct the bio-constrained joint loss function.

    Args:
        config: Parsed experiment configuration. Uses
            ``config.loss.reduction`` and ``config.loss.target_rate``.

    Returns:
        Configured :class:`BioJointLoss`.
    """
    return BioJointLoss(
        reduction=config.loss.reduction,
        target_rate=config.loss.target_rate,
    )


# ═══════════════════════════════════════════════════════════════
# 3.  DataLoader Factory
# ═══════════════════════════════════════════════════════════════

def resolve_condition_metadata(
    dataset: Dict[str, Any],
    x_seqs: Sequence[np.ndarray],
    lengths: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Return per-trial condition metadata, deriving it when absent.

    ``pipeline_semantics_version`` 2.2 does not imply the condition stamp:
    most processed corpora carry that version yet omit
    ``stimulus_conditions``/``is_pure_wind``.  Deriving from the physical
    channels keeps the routing auxiliary loss usable on those artifacts
    instead of silently disabling it.

    Derivation always runs, even when a stamp exists, so a stale stamp is
    caught rather than trusted blindly.

    Args:
        dataset: Loaded corpus mapping.
        x_seqs: Per-trial feature arrays (visual angle at index 0, wind
            state at index 1).
        lengths: Valid (unpadded) frame count per trial.

    Returns:
        ``(stimulus_conditions, is_pure_wind, derived)``; ``derived`` is
        True when either field came from the channels, not the stamp.
    """
    stored_cond = dataset.get("stimulus_conditions")
    stored_pw = dataset.get("is_pure_wind")
    derived_cond, derived_pw = derive_stimulus_metadata(x_seqs, lengths)

    if stored_cond is None and stored_pw is None:
        return derived_cond, derived_pw, True

    conditions = (
        np.asarray(stored_cond, dtype=object)
        if stored_cond is not None
        else derived_cond
    )
    is_pure_wind = (
        np.asarray(stored_pw, dtype=bool) if stored_pw is not None else derived_pw
    )
    if stored_pw is not None and not np.array_equal(is_pure_wind, derived_pw):
        logger.warning(
            "Stored is_pure_wind disagrees with the physical channels on "
            "%d trial(s); trusting the stored stamp.",
            int(np.sum(is_pure_wind != derived_pw)),
        )
    return conditions, is_pure_wind, stored_cond is None or stored_pw is None


def check_routing_aux_active(
    is_pure_wind: np.ndarray,
    stimulus_conditions: np.ndarray,
    lambda_routing_aux: float,
    split_name: str,
    corpus: str,
) -> bool:
    """Report whether the routing hinge can actually fire on this split.

    ``compute_routing_aux_loss`` returns exactly ``0.0`` when either
    condition group is empty, so a split without pure-wind trials makes
    the auxiliary term vanish -- and a vanished term is indistinguishable
    from a converged one in the training log.  That silence is the defect;
    the graceful non-crash is deliberate (User Story 12), so this reports
    rather than raises, at ERROR level because the configured term is
    contributing nothing to a run the operator believes is using it.

    Args:
        is_pure_wind: Per-trial pure-wind flags for this split.
        stimulus_conditions: Per-trial condition names, for the census.
        lambda_routing_aux: Configured auxiliary weight; 0 disables.
        split_name: ``"train"`` or ``"val"``, named in the message.
        corpus: Dataset path, named in the message.

    Returns:
        True when the term is active (both groups present, weight > 0);
        False when it is disabled or structurally inert.  Callers should
        surface a False alongside the run's metrics so an inert term is
        never mistaken for a trained one.
    """
    if lambda_routing_aux <= 0.0:
        return False

    flags = np.asarray(is_pure_wind, dtype=bool)
    n_wind = int(np.sum(flags))
    n_other = int(np.sum(~flags))
    if n_wind > 0 and n_other > 0:
        return True

    census = dict(collections.Counter(np.asarray(stimulus_conditions).tolist()))
    empty = "pure-wind" if n_wind == 0 else "non-pure-wind"
    logger.error(
        "ROUTING AUX INERT: lambda_routing_aux=%s is set but the %s split "
        "of %s contains no %s trials (condition census: %s). The hinge "
        "contrasts the two groups, so it returns exactly 0.0 for every "
        "batch and the term contributes NOTHING to this run -- the loss "
        "curve will look normal regardless. Use a corpus with both groups "
        "present, or set lambda_routing_aux=0 to disable it deliberately.",
        lambda_routing_aux,
        split_name,
        corpus,
        empty,
        census,
    )
    return False


def zero_input_channels_inplace(
    dataset: Any, channels: Sequence[int],
    sensory_dim: Optional[int] = None,
) -> None:
    """History-ablation control: zero declared feature columns in a dataset.

    Writes ``0.0`` into ``dataset.sequences[i][0][:, ch]`` for every declared
    physical sensory column ``ch``.  ``NSMoRDataset`` deep-copies each
    sequence at construction (``nsmor_dataloader.NSMoRDataset.__init__``), so
    this mutates only the loader's private copies — the stored dataset bytes
    and the caller's arrays are untouched.  No-op for an empty *channels*
    (the historical behaviour).  Only the physical sensory columns are
    addressable; the MCMC prior columns are validated upstream by the
    ``--zero_input_channels`` parser against ``model.sensory_dim``.

    *sensory_dim* is the SINGLE authoritative bound (the same
    ``config.model.sensory_dim`` the CLI parser validates against); when
    omitted it falls back to the dataset's ``per_frame_physical_dim``.  Both
    are 4 for the frozen corpus, but only ``sensory_dim`` is the quantity the
    parse path bounds, so callers that hold the config must pass it.
    """
    chans = tuple(int(c) for c in channels)
    if not chans:
        return
    sequences = getattr(dataset, "sequences", None)
    if sequences is None:
        raise ValueError(
            "zero_input_channels requires a dataset exposing .sequences"
        )
    if sensory_dim is None:
        feature_config = getattr(dataset, "feature_config", None)
        sensory_dim = getattr(feature_config, "per_frame_physical_dim", None)
    if sensory_dim is not None:
        for ch in chans:
            if not 0 <= ch < sensory_dim:
                raise ValueError(
                    f"zero_input_channels column {ch} must index a physical "
                    f"sensory column in [0, {sensory_dim})"
                )
    for i, (x_seq, y_seq, _label) in enumerate(sequences):
        assert x_seq.ndim == 2, (
            f"sequence {i} X must be (T, F); got shape {x_seq.shape}"
        )
        assert x_seq.shape[0] == y_seq.shape[0], (
            f"sequence {i} X/Y length mismatch: {x_seq.shape[0]} vs "
            f"{y_seq.shape[0]}"
        )
        for ch in chans:
            assert ch < x_seq.shape[1], (
                f"sequence {i} channel {ch} out of range for {x_seq.shape[1]} "
                f"feature columns"
            )
            x_seq[:, ch] = 0.0
    logger.info("History ablation applied: zeroed channels %s", list(chans))


def assert_finite_targets(Y_seqs: Sequence[Any]) -> None:
    """Fail closed on any non-finite target value.

    A single NaN/Inf in ``Y_seqs`` (train *or* val) poisons target
    statistics and then every standardized loss/metric for the whole run
    — silent, total corruption.  One guard, called from every entry point
    that touches ``Y_seqs`` (``build_dataloaders`` and
    ``compute_target_stats``).

    Args:
        Y_seqs: Per-trial 1-D continuous velocity targets.

    Raises:
        ValueError: On the first trial containing a NaN/Inf.
    """
    for i, y in enumerate(Y_seqs):
        y_arr = np.asarray(y)
        if not np.isfinite(y_arr).all():
            n_bad = int(y_arr.size - np.isfinite(y_arr).sum())
            raise ValueError(
                f"Y_seqs[{i}] contains {n_bad} non-finite value(s) "
                "(NaN/Inf). Target statistics would be non-finite and "
                "silently corrupt normalization/loss for the entire run; "
                "fail closed. Regenerate the dataset after sanitizing "
                "kinematics."
            )


def build_dataloaders(
    config: ExperimentConfig,
    dataset_path: str = "data/processed/nsmor_dataset.pt",
    val_split: float = 0.2,
    use_lazy_loading: bool = False,
    nested_prior_artifact: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
    zero_input_channels: Optional[Sequence[int]] = None,
) -> Tuple[Optional[torch.utils.data.DataLoader], Optional[torch.utils.data.DataLoader]]:
    """
    Build train and validation dataloaders from the prepared dataset.

    Supports two modes:
    - ETL mode (default): Load pre-processed dataset from ``nsmor_dataset.pt``
    - ELT mode (lazy): Load metadata and read trials on-demand from CSVs

    Args:
        config: Parsed experiment configuration.
        dataset_path: Path to dataset/metadata file.
        val_split: Fraction of data to use for validation (0-1).  Ignored
            when ``nested_prior_artifact`` is set, which prescribes the
            exact outer split.
        use_lazy_loading: If True, use ELT mode with lazy loading.
        nested_prior_artifact: Optional ``nested_split_seed*.pt`` produced
            by ``scripts/evaluate_nested_prior.py``.  When set, the exact
            persisted train/val indices and nested priors replace the
            recomputed split and global OOF priors.  Fail-closed on any
            fingerprint/split/recording-prefix disjointness mismatch.

    Returns:
        ``(train_loader, val_loader)`` — either may be ``None`` if
        the dataset file is not found or the split is empty.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the nested artifact is incompatible with the dataset
            or is combined with lazy loading.
    """
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        logger.warning(
            "Dataset file not found: %s.  "
            "Run 'python scripts/prepare_%s.py' first.",
            dataset_file,
            "metadata" if use_lazy_loading else "data",
        )
        return None, None

    if nested_prior_artifact is not None and use_lazy_loading:
        raise ValueError(
            "--nested_prior_artifact requires ETL mode: lazy loading rebuilds "
            "rows on demand and cannot guarantee the persisted outer split "
            "aligns with the artifact. Refusing to run (fail closed)."
        )

    # The history-ablation control zeroes columns on the eager NSMoRDataset.
    # Lazy rows are rebuilt on demand through a different path that does not
    # apply the ablation, so a lazy run with the option set would silently be
    # the UNABLATED control.  Refuse rather than mislabel the arm (fail closed).
    _zero_requested = (
        zero_input_channels if zero_input_channels is not None
        else _ZERO_INPUT_CHANNELS
    )
    if use_lazy_loading and _zero_requested:
        raise ValueError(
            "data.zero_input_channels (history ablation) is unsupported in "
            "lazy loading mode; use ETL mode or clear the option. Refusing to "
            "run a silently-unablated history control (fail closed)."
        )

    from nsmor.model_utils import validate_dataset_provenance

    # ── ELT Mode: Lazy Loading ────────────────────────────────
    if use_lazy_loading:
        from nsmor.pipeline.io import ClockAwareLazyDataset

        logger.info("Loading metadata from %s (lazy mode)", dataset_file)

        # Read one metadata snapshot for both lazy rows and provenance.
        raw_meta, loaded_source_fingerprint = load_dataset_with_fingerprint(
            dataset_file, expected_dt_ms=config.model.dt_ms, restore_provenance=False,
        )
        prior_status = validate_dataset_provenance(raw_meta, dataset_file)
        prior_tag = raw_meta["mcmc_prior_provenance"]
        full_dataset = ClockAwareLazyDataset(
            metadata_path=str(dataset_file),
            max_seq_len=config.training.max_seq_len,
            dt_ms=config.model.dt_ms,
            metadata=raw_meta,
        )

        # Recording-prefix grouped split, via the eager-path helper.
        # The previous implementation derived unique keys from
        # ``list(set(...))``, whose iteration order varies with
        # PYTHONHASHSEED — so the lazy split was not reproducible across
        # processes even at a fixed seed, and could not match the eager
        # path or compute_target_stats.
        session_ids = resolve_dataset_session_ids(raw_meta)
        train_indices, val_indices = grouped_train_val_split(
            session_ids,
            len(full_dataset),
            val_split=val_split,
            random_seed=config.training.random_seed,
        )
        train_indices = train_indices.tolist()
        val_indices = val_indices.tolist()

        # Create subset datasets
        from torch.utils.data import Subset

        # Extract stimulus condition metadata for lazy loading
        is_pure_wind = None
        if full_dataset.trial_specs and any(
            "is_pure_wind" in spec for spec in full_dataset.trial_specs
        ):
            is_pure_wind = np.array(
                [
                    bool(spec.get("is_pure_wind", False))
                    for spec in full_dataset.trial_specs
                ],
                dtype=bool,
            )
        else:
            if "is_pure_wind" in raw_meta:
                is_pure_wind = np.asarray(raw_meta["is_pure_wind"], dtype=bool)
            elif "stimulus_conditions" in raw_meta:
                is_pure_wind = np.asarray(
                    [c == "wind_only" for c in raw_meta["stimulus_conditions"]],
                    dtype=bool,
                )

        train_is_pure_wind = (
            is_pure_wind[train_indices] if is_pure_wind is not None else None
        )
        val_is_pure_wind = (
            is_pure_wind[val_indices] if is_pure_wind is not None else None
        )

        if train_is_pure_wind is not None:
            check_routing_aux_active(
                train_is_pure_wind,
                np.array(
                    ["wind_only" if pw else "other" for pw in train_is_pure_wind]
                ),
                config.loss.lambda_routing_aux,
                "train",
                str(dataset_file),
            )

        class LazySubset(Subset):
            """Subset that preserves condition metadata for routing aux loss."""

            def __init__(self, dataset, indices, is_pure_wind=None):
                super().__init__(dataset, indices)
                self.is_pure_wind = is_pure_wind

            def __getitem__(self, idx):
                item = self.dataset[self.indices[idx]]
                if self.is_pure_wind is not None:
                    return (
                        item[0],
                        item[1],
                        item[2],
                        bool(self.is_pure_wind[idx]),
                    )
                return item

            def __getitems__(self, indices):
                return [self.__getitem__(idx) for idx in indices]

        train_dataset = LazySubset(
            full_dataset, train_indices, train_is_pure_wind
        )
        train_dataset.prior_lineage = (prior_tag, prior_status)
        train_dataset.dataset_source_sha256 = loaded_source_fingerprint
        val_dataset = LazySubset(
            full_dataset, val_indices, val_is_pure_wind
        )

        # Create dataloaders with collate function
        def collate_fn(batch):
            if len(batch[0]) == 4:
                X_seqs, Y_seqs, lengths, pure_winds = zip(*batch)
                return (
                    torch.nn.utils.rnn.pad_sequence(X_seqs, batch_first=True),
                    torch.nn.utils.rnn.pad_sequence(Y_seqs, batch_first=True),
                    torch.tensor(lengths),
                    torch.tensor(pure_winds, dtype=torch.bool),
                )
            X_seqs, Y_seqs, lengths = zip(*batch)
            return (
                torch.nn.utils.rnn.pad_sequence(X_seqs, batch_first=True),
                torch.nn.utils.rnn.pad_sequence(Y_seqs, batch_first=True),
                torch.tensor(lengths),
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # CSV loading not thread-safe
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        logger.info(
            "DataLoaders created: train=%d batches, val=%d batches (batch_size=%d)",
            len(train_loader),
            len(val_loader),
            config.training.batch_size,
        )

        return train_loader, val_loader

    # ── ETL Mode: Pre-loaded Dataset ──────────────────────────
    from nsmor.nsmor_dataloader import NSMoRDataset

    logger.info("Loading dataset from %s", dataset_file)
    dataset, loaded_source_fingerprint = load_dataset_with_fingerprint(
        dataset_file, expected_dt_ms=config.model.dt_ms, restore_provenance=False,
    )

    # Round-2 CRITICAL-A: refuse pre-2.0 datasets (leaked priors,
    # np.max labels) — training on them is scientifically invalid.
    animal_identity_status = validate_dataset_provenance(dataset, dataset_file)
    session_ids = resolve_dataset_session_ids(dataset)

    X_seqs = dataset["X_seqs"]
    Y_seqs = dataset["Y_seqs"]
    mcmc_priors = dataset["mcmc_priors"]
    labels = dataset["labels"]
    lengths = dataset["lengths"]
    # Fail closed on NaN/Inf targets before any split/prior work: a
    # non-finite Y poisons target stats, loss and metrics silently.
    assert_finite_targets(Y_seqs)

    # Ticket #16: stimulus condition metadata for the routing aux loss.
    # Derived when the corpus lacks the stamp -- version 2.2 does not imply
    # it -- so the term is never silently disabled by a missing key.
    stimulus_conditions, is_pure_wind, derived = resolve_condition_metadata(
        dataset, X_seqs, lengths
    )
    logger.info(
        "Stimulus condition census (%s): %s",
        "derived from physical channels" if derived else "from stored stamp",
        dict(collections.Counter(np.asarray(stimulus_conditions).tolist())),
    )

    n_total = len(X_seqs)
    logger.info(
        "Loaded %d sequences, total_frames=%d",
        n_total, int(lengths.sum()),
    )

    prior_consistency = dataset.get("mcmc_prior_train_serve_consistency")
    if prior_consistency is not None:
        logger.info(
            "MCMC prior train-vs-serve consistency: %s",
            prior_consistency,
        )

    # Resolve anchor frames for anchor-aligned cropping
    anchor_frames = dataset.get("anchor_frames")
    if anchor_frames is None:
        from nsmor.pipeline.conditions import derive_anchor_frames
        anchor_frames = derive_anchor_frames(X_seqs, lengths)
        logger.info(
            "Derived %d anchor frames from physical channels.",
            len(anchor_frames),
        )

    # ── Deterministic train/val split ─────────────────────────
    # The split groups `_session_N` blocks by recording prefix to prevent
    # within-recording overlap. Distinct prefixes are not verified animals.
    # Full nested CV is available through `--nested_prior_artifact`.
    if nested_prior_artifact is not None:
        # Opt-in nested-prior seam: consume the EXACT persisted outer
        # split and the leak-free nested priors.  Recomputing a split
        # here would discard the outer hold-out the nested protocol
        # established; falling back to global OOF priors would let
        # outer-train fold models encode outer-val prefix group labels.
        feature_config = dataset.get("feature_config", DEFAULT_FEATURE)
        (
            train_indices,
            val_indices,
            nested_priors,
            nested_info,
        ) = load_nested_prior_split(
            Path(nested_prior_artifact),
            Path(dataset_file),
            n_total,
            session_ids,
            feature_config=feature_config,
            loaded_source_fingerprint=loaded_source_fingerprint,
            trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
        )
        mcmc_priors = nested_priors
        # Honest provenance — from the validated artifact, never a
        # hardcoded string.  mcmc_prior_provenance is the generator's
        # own lineage tag; nested_prior_fingerprint is the SHA-256 of
        # the source dataset the artifact was validated against.
        logger.info(
            "Nested-prior mode: replaced global OOF priors with nested "
            "priors from %s (provenance=%s, fingerprint=%s, split_seed=%s, "
            "val_split=%.3f requested / %.3f realized trials).",
            nested_info["nested_prior_artifact"],
            nested_info["mcmc_prior_provenance"],
            nested_info["nested_prior_fingerprint"][:12],
            nested_info["split_seed"],
            nested_info["val_split"],
            nested_info["val_trial_fraction"],
        )
    else:
        train_indices, val_indices = grouped_train_val_split(
            session_ids,
            n_total,
            val_split=val_split,
            random_seed=config.training.random_seed,
        )
    n_train = len(train_indices)
    n_val = len(val_indices)

    logger.info(
        "Split: %d train, %d val (%.0f%% val)",
        n_train, n_val, val_split * 100,
    )

    # ── Build sequence lists for each split ───────────────────
    def _build_split_sequences(
        split_indices: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """Build sequence list for a given split."""
        sequences = []
        for idx in split_indices:
            sequences.append((
                X_seqs[idx],
                Y_seqs[idx],
                int(labels[idx]),
            ))
        return sequences

    train_sequences = _build_split_sequences(train_indices)
    val_sequences = _build_split_sequences(val_indices)

    # ── Extract priors for each split ─────────────────────────
    train_priors = mcmc_priors[train_indices]
    val_priors = mcmc_priors[val_indices]

    # Ticket #16: Split stimulus condition metadata
    train_is_pure_wind = is_pure_wind[train_indices]
    val_is_pure_wind = is_pure_wind[val_indices]

    # The hinge is evaluated per split, so a split that strands every wind
    # trial on one side is as inert as a corpus with none at all.
    check_routing_aux_active(
        train_is_pure_wind,
        np.asarray(stimulus_conditions)[train_indices],
        config.loss.lambda_routing_aux,
        "train",
        str(dataset_file),
    )

    # ── Shape assertions ──
    assert len(train_sequences) == n_train, (
        f"Train sequences: {len(train_sequences)} != {n_train}"
    )
    assert len(val_sequences) == n_val, (
        f"Val sequences: {len(val_sequences)} != {n_val}"
    )
    assert train_priors.shape == (n_train, 4), (
        f"Train priors shape {train_priors.shape} != ({n_train}, 4)"
    )
    assert val_priors.shape == (n_val, 4), (
        f"Val priors shape {val_priors.shape} != ({n_val}, 4)"
    )

    # ── Extract anchor frames for each split ──────────────────
    train_anchor_frames = [anchor_frames[i] for i in train_indices]
    val_anchor_frames = [anchor_frames[i] for i in val_indices]

    # ── Create datasets ───────────────────────────────────────
    feature_config = dataset.get("feature_config", DEFAULT_FEATURE)

    max_seq_len = getattr(config.training, "max_seq_len", None)

    # History-ablation control: resolve the declared zeroed feature columns
    # once.  ``zero_input_channels`` (the explicit parameter) takes precedence
    # over the module-level ``_ZERO_INPUT_CHANNELS`` (set by build_config from
    # the CLI); both default to empty, so a caller that passes neither gets the
    # historical datasets bitwise unchanged.
    if zero_input_channels is None:
        zero_input_channels = tuple(_ZERO_INPUT_CHANNELS)
    else:
        zero_input_channels = tuple(zero_input_channels)
    if zero_input_channels:
        logger.info(
            "History ablation active: zeroing input channels %s",
            list(zero_input_channels),
        )

    train_dataset = NSMoRDataset(
        sequences=train_sequences,
        mcmc_priors=train_priors,
        feature_config=feature_config,
        max_seq_len=max_seq_len,
        anchor_frames=train_anchor_frames,
        source_indices=train_indices,
        is_pure_wind=train_is_pure_wind,
    )
    train_dataset.mcmc_prior_train_serve_consistency = prior_consistency
    train_dataset.dataset_source_sha256 = loaded_source_fingerprint
    train_dataset.prior_lineage = (dataset["mcmc_prior_provenance"], animal_identity_status)
    if nested_prior_artifact is not None:
        # Carried to train() so checkpoints/results record the exact
        # nested artifact + fingerprint the split/priors came from.
        train_dataset.nested_prior_info = nested_info
    val_dataset = NSMoRDataset(
        sequences=val_sequences,
        mcmc_priors=val_priors,
        feature_config=feature_config,
        max_seq_len=max_seq_len,
        anchor_frames=val_anchor_frames,
        source_indices=val_indices,
        is_pure_wind=val_is_pure_wind,
    )
    val_dataset.mcmc_prior_train_serve_consistency = prior_consistency
    if nested_prior_artifact is not None:
        val_dataset.nested_prior_info = nested_info

    # ── History-ablation control (no-op by default) ──────────
    # Zero the declared sensory columns in the datasets' deep-copied
    # sequences.  Applied AFTER construction and BEFORE the loaders, so both
    # splits see the identical ablation and the stored dataset bytes are never
    # modified.  ``zero_input_channels`` is empty unless explicitly set.
    _sensory_dim = getattr(getattr(config, "model", None), "sensory_dim", None)
    zero_input_channels_inplace(
        train_dataset, zero_input_channels, sensory_dim=_sensory_dim,
    )
    zero_input_channels_inplace(
        val_dataset, zero_input_channels, sensory_dim=_sensory_dim,
    )

    # ── Create dataloaders (via factory) ──────────────────────
    # Delegates to dataloader_factory for unified worker auto-scaling,
    # pin_memory, and persistent_workers policy.
    train_loader, val_loader, _ = create_dataloaders_from_config(
        config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    logger.info(
        "DataLoaders created: train=%d batches, val=%d batches (batch_size=%d)",
        len(train_loader), len(val_loader), config.training.batch_size,
    )

    return train_loader, val_loader


def compute_target_stats(
    dataset_path: str,
    config: ExperimentConfig,
    val_split: float = 0.2,
    nested_prior_artifact: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Compute training-split velocity mean and std for target normalization.

    Uses the *same* deterministic train/val split as
    :func:`build_dataloaders` (seeded by ``config.training.random_seed``,
    or the exact persisted split when ``nested_prior_artifact`` is set)
    and aggregates only the training sequences, so no validation signal
    leaks into the normalization statistics.

    The raw velocity target is heavy-tailed: 99.99% of frames satisfy
    ``|y| < 100 cm/s`` (resting cricket), but a handful reach ~1e7 cm/s.
    Using the std over the full distribution would still let those extreme
    frames dominate a standardized MSE.  We therefore compute statistics
    over the *robust* bulk (frames in the middle ``100% - 2*trim``
    percentile band) and, when :attr:`TrainingConfig.normalize_targets`
    is enabled, report them for downstream model fitting.

    Args:
        dataset_path: Path to the preprocessed ``nsmor_dataset.pt``.
        config: Parsed experiment configuration (for the split seed).
        val_split: Fraction held out for validation (must match
            :func:`build_dataloaders`).  Ignored when
            ``nested_prior_artifact`` is set.
        nested_prior_artifact: Optional nested-prior artifact whose
            persisted ``train_indices`` replace the recomputed split, so
            target statistics are fit on exactly the trials the dataloader
            trains on (no target-stat split mismatch).

    Returns:
        ``(train_mean, train_std, train_indices)`` where ``train_indices``
        is an ``int64`` array of dataset sequence indices included in the
        train split. When normalization is disabled or data is missing,
        returns ``(0.0, 1.0, np.empty(0, dtype=np.int64))``.
    """
    if not config.training.normalize_targets:
        return 0.0, 1.0, np.empty(0, dtype=np.int64)

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        logger.warning(
            "Dataset not found for target stats: %s — using identity.",
            dataset_file,
        )
        return 0.0, 1.0, np.empty(0, dtype=np.int64)

    dataset, loaded_source_fingerprint = load_dataset_with_fingerprint(
        dataset_file, expected_dt_ms=config.model.dt_ms, restore_provenance=False,
    )
    from nsmor.model_utils import validate_dataset_provenance
    validate_dataset_provenance(dataset, dataset_file)
    session_ids = resolve_dataset_session_ids(dataset)
    Y_seqs = dataset["Y_seqs"]
    n_total = len(Y_seqs)
    assert_finite_targets(Y_seqs)

    # ── Grouped split (must mirror build_dataloaders exactly) ─────────
    # Normalization statistics must be fit on the training split only, so
    # this calls the SAME helper build_dataloaders calls.  Two hand-copied
    # implementations previously drifted apart (sample-level here vs
    # session-level there), fitting the statistics on data that included
    # validation trials — a leakage channel that no test could catch while
    # both copies existed.  One shared function, one split.  With a nested
    # artifact, the same persisted indices are consumed here and in
    # ``build_dataloaders`` for standalone calls. train() instead fits from its
    # already-loaded train dataset so a same-path artifact swap cannot drift.
    if nested_prior_artifact is not None:
        feature_config = dataset.get("feature_config", DEFAULT_FEATURE)
        train_indices, _val_indices, _nested_priors, _info = load_nested_prior_split(
            Path(nested_prior_artifact),
            dataset_file,
            n_total,
            session_ids,
            feature_config=feature_config,
            loaded_source_fingerprint=loaded_source_fingerprint,
            trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
        )
    else:
        train_indices, _val_indices = grouped_train_val_split(
            session_ids,
            n_total,
            val_split=val_split,
            random_seed=config.training.random_seed,
        )

    train_y = np.concatenate([Y_seqs[i] for i in train_indices]).astype(np.float64)
    return _fit_target_stats(train_y, config, train_indices)


def _fit_target_stats(
    train_y: np.ndarray,
    config: ExperimentConfig,
    train_indices: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Fit the existing clip/trim transform on selected raw training targets."""
    # Fit the statistic in the SAME space the loss sees.  ``train_one_epoch``
    # clips the target to ``[-target_clip_cm_s, +target_clip_cm_s]`` before
    # standardising, so we clip here first; otherwise the ±(81..100] cm/s band
    # that survives clip would be divided by a *smaller* trimmed std, re-imposing
    # heavy-tail domination (the statistic and transform would disagree).
    clip = config.training.target_clip_cm_s
    if clip > 0.0:
        train_y = np.clip(train_y, -clip, clip)

    # Robust trim of the remaining bulk for a well-conditioned mean/std.
    lo, hi = np.percentile(train_y, [0.5, 99.5])
    bulk = train_y[(train_y >= lo) & (train_y <= hi)]
    if bulk.size == 0:
        bulk = train_y

    mean = float(bulk.mean())
    std = float(bulk.std())
    if std < 1e-3:
        # Degenerate constant target — fall back to identity.
        logger.warning("Target std near zero (%.6f) — using identity transform.", std)
        return 0.0, 1.0, train_indices

    logger.info(
        "Target normalization on train split: mean=%.4f cm/s  std=%.4f cm/s  "
        "(n=%d bulk frames, trimmed to [%s, %s])",
        mean, std, int(bulk.size),
        f"{lo:.4f}", f"{hi:.4f}",
    )
    return mean, std, train_indices


# ═══════════════════════════════════════════════════════════════
# 4.  Training Loop
# ═══════════════════════════════════════════════════════════════

def compute_lr_warmup_scale(epoch: int, lr_warmup_epochs: int) -> float:
    """
    Compute the linear LR warmup scale.

    During the first ``lr_warmup_epochs`` epochs, the base learning rate
    is ramped linearly (times each param group's configured base LR).
    After warmup the scale is ``1.0``.

    Optimization rationale: a full-LR first optimiser step on a cold
    recurrent state (LIF surrogate gradients and GRU hidden states not yet
    settled) overshoots the loss surface; the shared AdamW then accumulates
    a polluted second moment (``v``) that the cosine schedule cannot
    correct within a short run.  Ramping the LR keeps early updates small
    and clean so that ``v`` tracks the true landscape.

    The scale is applied multiplicatively to the param group's ``lr``
    before the scheduler-step of the same epoch (LR warmup precedence
    over cosine annealing, matching the effective phase of a
    ``LinearWarmupCosineAnnealingLR`` without a new optimiser group).

    Args:
        epoch: Current 0-indexed epoch.
        lr_warmup_epochs: Warmup epoch count.  ``0`` disables (scale = 1).

    Returns:
        ``float`` in ``[0.0, 1.0]`` — LR multiplier for this epoch.
    """
    if lr_warmup_epochs <= 0:
        return 1.0
    if epoch >= lr_warmup_epochs:
        return 1.0
    # Linear ramp over [0, lr_warmup_epochs): epoch e gets scale (e+1)/W.
    # The first step is 1/W (tiny but non-zero — avoids a dead start) and
    # the last warmup epoch reaches exactly 1.0, closing the window at the
    # boundary.  (The nominal half-open window is documented as "ramp over
    # W epochs"; reaching full LR on the final warmup epoch is the chosen
    # convention — the alternative e/W leaves the window one epoch short.)
    return float((epoch + 1) / lr_warmup_epochs)


def apply_lr_warmup(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    lr_warmup_epochs: int,
) -> None:
    """
    Ramp each param group's LR linearly during the warmup window.

    Each group must carry a ``base_lr`` key (set at optimiser
    construction) holding its FULL configured LR.  During warmup the
    group LR is set to ``base_lr * scale`` — overriding the cosine-
    scheduled value — so warmup is deterministic and does not compound
    with the cosine decay.  Setting ``base_lr`` separately from the
    cosine-updated ``lr`` preserves each group's relative rate, including
    the 0.3x LIF damping.

    The cosine ``scheduler.step()`` (called later in the same loop
    iteration) writes ``group["lr"]`` fresh each epoch, so the warmup
    override and the cosine anneal never accumulate.

    Args:
        optimizer: The AdamW optimiser to scale.
        epoch: Current 0-indexed epoch (local to the phase in two-phase
            training, so the warmup restarts cleanly at the transition).
        lr_warmup_epochs: Warmup epoch count.  ``0`` disables warmup.

    Raises:
        ValueError: If any param group is missing either ``"base_lr"``
            or ``"lr"``.
    """
    if lr_warmup_epochs <= 0:
        return
    scale = compute_lr_warmup_scale(epoch, lr_warmup_epochs)
    for group in optimizer.param_groups:
        if "base_lr" not in group or "lr" not in group:
            raise ValueError(
                "apply_lr_warmup: param group must carry both 'base_lr' and 'lr'.",
            )
        group["lr"] = group["base_lr"] * scale


def _maybe_step_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    lr_warmup_epochs: int,
    warmup_epoch: int,
) -> None:
    """
    Advance the LR scheduler once per epoch, holding it during the warmup
    window so the anneal budget isn't consumed by ``apply_lr_warmup``'s
    override.

    Backward-compat contract: when ``lr_warmup_epochs == 0`` (default), the
    step is unconditional — identical to the original single-schedule loop —
    so the default-config cosine trajectory is byte-for-byte the baseline's.
    """
    if lr_warmup_epochs == 0 or warmup_epoch >= lr_warmup_epochs:
        scheduler.step()


def compute_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    """
    Compute the warmup scaling factor for bio-loss regularization terms.

    During warmup (``epoch < warmup_epochs``), the factor ramps via a
    cosine curve from 0 to ``1.0``.  After warmup, the factor is
    exactly ``1.0``.

    CF7 fix: Cosine warmup replaces linear warmup to avoid the
    gradient discontinuity at the warmup boundary.  Linear warmup
    has a constant derivative (d/dt = 1/warmup_epochs), creating a
    sudden "step" in the effective loss gradient when warmup ends.
    Cosine warmup has zero derivative at both endpoints (smooth
    S-curve), preventing the gradient shock that can destabilize
    Adam's moment estimates.

    Note: ``lambda_reg`` is NOT scaled by this factor.  Only
    ``lambda_energy``, ``lambda_sparse``, and ``lambda_jerk`` are
    warmup-ramped via the ``annealing_factor`` parameter.  The router
    needs anti-collapse pressure from epoch 0 to prevent GRU
    monopolisation during the warmup window.

    Args:
        epoch: Current epoch number (0-indexed).
        warmup_epochs: Total warmup epoch count.  0 disables warmup
            (factor is always 1.0).

    Returns:
        Scaling factor in [0, 1] during warmup, 1.0 after.
    """
    if warmup_epochs > 0 and epoch < warmup_epochs:
        # Cosine ramp: 0.5 * (1 - cos(pi * progress))
        # At progress=0: factor=0.  At progress=1: factor=1.
        # Derivative at endpoints = 0 (smooth start and end).
        progress = float(epoch + 1) / float(warmup_epochs)
        return 0.5 * (1.0 - math.cos(math.pi * progress))
    return 1.0


def train_one_epoch(
    model: NSMoRCore,
    loader: torch.utils.data.DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_reg: float = 0.01,
    lambda_energy: float = 0.0,
    lambda_sparse: float = 0.0,
    lambda_jerk: float = 0.0,
    annealing_factor: float = 1.0,
    grad_clip_norm: float = 1.0,
    log_interval: int = 10,
    epoch: int = 0,
    lif_threshold: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_ctx = None,
    phase: int = 0,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    target_clip_cm_s: float = 0.0,
    lambda_routing_aux: float = 0.0,
    wind_only_mask_full: Optional[np.ndarray] = None,
    routing_aux_margin: float = 0.024,
    lambda_compute: float = 0.0,
) -> float:
    """
    Run one training epoch.

    Args:
        model: The NSMoR model.
        loader: Training DataLoader yielding ``(X_batch, Y_batch, lengths)``.
        criterion: Loss function — either :class:`FrontendLoss` (phase 1)
            or :class:`BioDecisionLoss` / :class:`BioJointLoss` (phase 2).
        optimizer: Optimizer.
        device: Device to train on.
        lambda_reg: Router regularization weight.
        lambda_energy: ATP metabolic cost weight.
        lambda_sparse: Population sparsity L1 weight.
        lambda_jerk: Frame-based third-velocity-difference smoothness weight
            (legacy name "jerk"; NOT physical jerk — no dt scaling).
        annealing_factor: Scaling factor for bio-loss lambdas.
        grad_clip_norm: Max gradient norm for clipping.
        log_interval: Log every N batches.
        epoch: Current epoch number (for logging).
        phase: Training phase — ``1`` = frontend-only (FrontendLoss),
            ``2`` = backend-only (BioDecisionLoss), ``0`` = single-phase
            (BioJointLoss, backward compatible).

    Returns:
        Average training loss for this epoch.  Skip counters
        (``n_skipped_nonfinite_loss``, ``n_skipped_nonfinite_grad``) are
        reported via :attr:`train_one_epoch.last_skip_counts` — a training
        stability audit must surface how many steps were silently dropped.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_skipped_nonfinite_loss = 0
    n_skipped_nonfinite_grad = 0

    # CF9: Per-epoch membrane health accumulators
    _epoch_v_max = 0.0
    _epoch_v_mean = 0.0
    _epoch_spike_rate = 0.0
    _epoch_w_adapt = 0.0
    _health_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False, dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        # Ticket #16: Unpack batch — (X, Y, lengths) legacy 3-tuple, or
        # (X, Y, lengths, wind_only_mask) 4-tuple when metadata present.
        if len(batch) == 4:
            x_batch, y_batch, lengths, wind_only_mask = batch
            wind_only_mask = wind_only_mask.to(device).contiguous()
        else:
            x_batch, y_batch, lengths = batch
            wind_only_mask = None

        x_batch = x_batch.to(device).contiguous()
        y_batch = y_batch.to(device).contiguous()
        lengths = lengths.to(device).contiguous()

        # ── Target normalization (if enabled) ──
        # Regress on (y - target_mean)/target_std so the loss is dominated
        # by the well-conditioned bulk of the velocity distribution rather
        # than the handful of extreme frames.  Applied on-device to y_true
        # only; the forward/backward treats it as the regression target.
        # Optional robust clip removes tracking-artifact frames (|y| > cap)
        # that would otherwise contribute (1e6+)² to the MSE.
        if target_clip_cm_s > 0.0:
            y_batch = torch.clamp(y_batch, -target_clip_cm_s, target_clip_cm_s)
        if target_std != 1.0 or target_mean != 0.0:
            y_batch = (y_batch - target_mean) / target_std

        # ── Forward pass (with internals for routing gates) ──
        _ctx = amp_ctx() if amp_ctx is not None else nullcontext()
        with _ctx:
            y_pred, internals = model(x_batch, lengths, return_internals=True)

            # ── Extract routing gates for bio-loss ──
            # routing_gates: (B, T, 2) — index 0 is g_lif, index 1 is g_gru
            g_gru = internals["routing_gates"][:, :, 1:2]           # (B, T, 1)
            g_lif = internals["routing_gates"][:, :, 0:1]           # (B, T, 1), Ticket #16
            lif_spikes = internals["lif_spikes"]                    # (B, T, H)

            # ── Compute loss ──
            if phase == 1:
                # Phase 1: FrontendLoss — MSE only, no bio penalties
                loss = criterion(
                    y_pred=y_pred,
                    y_true=y_batch,
                    lengths=lengths,
                )
            else:
                # Phase 2 / single-phase: full bio-constrained loss
                # Adaptive-refinement compute cost (architecture v1): the
                # ponder cost lives in the model internals; it is only added
                # at this loss seam.  A positive lambda_compute requires it.
                _ponder = internals.get("refinement_ponder_cost", None)
                loss = criterion(
                    y_pred=y_pred,
                    y_true=y_batch,
                    lengths=lengths,
                    g_gru=g_gru,
                    lambda_reg=lambda_reg,
                    lif_spikes=lif_spikes,
                    lambda_energy=lambda_energy,
                    lambda_sparse=lambda_sparse,
                    lambda_jerk=lambda_jerk,
                    annealing_factor=annealing_factor,
                    # Ticket #16: auxiliary routing loss is enabled only
                    # when collate_with_metadata supplies the condition mask.
                    lambda_routing_aux=lambda_routing_aux,
                    wind_only_mask=wind_only_mask,
                    routing_aux_margin=routing_aux_margin,
                    lambda_compute=lambda_compute,
                    refinement_ponder_cost=_ponder,
                )

        # ── Membrane health monitoring (CF9: per-epoch averages) ──
        # Tracks V_max, spike_rate, and adaptation current across all
        # batches for early detection of runaway dynamics or collapse.
        with torch.no_grad():
            lif_potentials = internals["lif_potentials"]
            _epoch_v_max = max(_epoch_v_max, lif_potentials.abs().max().item())
            _epoch_v_mean += lif_potentials.abs().mean().item()
            _epoch_spike_rate += lif_spikes.float().mean().item()
            # Track adaptation current if available in internals
            if "lif_w_adapt" in internals:
                _epoch_w_adapt += internals["lif_w_adapt"].abs().mean().item()
            _health_batches += 1

        # ── Backward pass (AMP-aware) ──
        # Use set_to_none=True to fully release old gradients from
        # GPU memory — avoids accumulating stale ghost gradients on
        # frozen parameters across phase transitions.
        model.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ── NaN/Inf guard (Issues 1 & 2) ──
        # Detect non-finite loss BEFORE clipping so we can skip the
        # optimizer step and avoid poisoning Adam's moment estimates.
        if not math.isfinite(loss.item()):
            logger.warning(
                "Epoch %d batch %d: non-finite loss=%s — skipping step",
                epoch, batch_idx, loss.item(),
            )
            # Plain continue: no unscale_() ran on this iteration and
            # step() was not called, so the GradScaler holds no per-iter
            # state that update() could reconcile.  (Calling update() here
            # would read an empty found_inf and GROW the scale, amplifying
            # exactly the overflow that produced the non-finite loss.)
            # AMP caveat: if the non-finite loss came from an FP16 forward
            # overflow, the scale is NOT reduced on this path — documented
            # limitation; the gradient check below is the backstop.
            n_skipped_nonfinite_loss += 1
            continue  # skip optimizer.step()

        # ── Gradient clipping (unscale first for AMP) ──
        # Only clip ACTIVE parameters — frozen parameters retain
        # stale gradients from Phase 1 whose norm may be Inf,
        # which would zero-out all trainable gradients.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)

        # ── Pre-clip gradient finiteness check ──
        # Non-finite gradients SKIP the whole step: zeroing them (the old
        # behavior) feeds artificial zeros into Adam's second moment,
        # systematically polluting v.  found_inf was set inside unscale_()
        # above, so scaler.step() would already have skipped — but we
        # continue explicitly so the skip is counted and logged.
        has_nonfinite_grad = False
        for p in trainable_params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                has_nonfinite_grad = True
                break
        if has_nonfinite_grad:
            n_skipped_nonfinite_grad += 1
            logger.warning(
                "Epoch %d batch %d: non-finite gradient before clipping — skipping step",
                epoch, batch_idx,
            )
            # GradScaler bookkeeping: found_inf is already set from
            # unscale_(); update() halves the scale and clears it.
            if scaler is not None and scaler.is_enabled():
                scaler.update()
            continue  # skip optimizer.step()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=grad_clip_norm,
            )

        # ── Post-clip gradient finiteness check (defense-in-depth) ──
        # CF10: With FP32 LIF loop, NaN gradients should be rare.
        # If they still occur, skip the step to preserve Adam's moment
        # estimates from corruption by artificial zeros.
        # Only check ACTIVE parameters — frozen params have no grad.
        has_nan_grad = False
        for p in trainable_params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                has_nan_grad = True
                break
        if has_nan_grad:
            logger.warning(
                "Epoch %d batch %d: non-finite gradient after clipping — skipping step",
                epoch, batch_idx,
            )
            # Must call scaler.update() to reset internal state even
            # when skipping, otherwise next unscale_() will raise.
            if scaler is not None and scaler.is_enabled():
                scaler.update()
            continue  # skip optimizer.step()

        # ── Per-pathway gradient norm logging (CF7 fix) ──
        # Monitors gradient balance between LIF and non-LIF pathways.
        # If LIF gradients are consistently 10x+ larger, it confirms
        # the LIF pathway as the instability source.
        if batch_idx == 0 and epoch % 10 == 0:
            lif_grad_norm = 0.0
            non_lif_grad_norm = 0.0
            for name, p in model.named_parameters():
                # Only read gradients from trainable, non-frozen
                # parameters — frozen params retain stale grads from
                # Phase 1 whose norms would corrupt the metric.
                if p.requires_grad and p.grad is not None:
                    gn = p.grad.data.norm(2).item()
                    if "lif_cell" in name:
                        lif_grad_norm += gn ** 2
                    else:
                        non_lif_grad_norm += gn ** 2
            lif_grad_norm = lif_grad_norm ** 0.5
            non_lif_grad_norm = non_lif_grad_norm ** 0.5
            logger.info(
                "Epoch %d grad norms: LIF=%.4f  non_LIF=%.4f  "
                "ratio=%.2f",
                epoch, lif_grad_norm, non_lif_grad_norm,
                lif_grad_norm / max(non_lif_grad_norm, 1e-8),
            )

        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # ── Logging ──
        # (No pbar.update here — iterating the tqdm wrapper already advances
        # it once per batch; an extra update double-counted progress.)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(n_batches, 1)

    # Surface skip counts for the stability audit (see docstring).
    train_one_epoch.last_skip_counts = {
        "n_skipped_nonfinite_loss": n_skipped_nonfinite_loss,
        "n_skipped_nonfinite_grad": n_skipped_nonfinite_grad,
    }
    if n_skipped_nonfinite_loss or n_skipped_nonfinite_grad:
        logger.warning(
            "Epoch %d skipped steps: %d non-finite loss, %d non-finite grad",
            epoch, n_skipped_nonfinite_loss, n_skipped_nonfinite_grad,
        )

    # CF9: Compute per-epoch membrane health summary
    health = {}
    if _health_batches > 0:
        health = {
            "v_max": _epoch_v_max,
            "v_mean": _epoch_v_mean / _health_batches,
            "spike_rate": _epoch_spike_rate / _health_batches,
            "w_adapt": _epoch_w_adapt / _health_batches,
        }

    return avg_loss, health


# ═══════════════════════════════════════════════════════════════
# 5.  Validation Loop
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model: NSMoRCore,
    loader: torch.utils.data.DataLoader,
    criterion,
    device: torch.device,
    lambda_reg: float = 0.01,
    lambda_energy: float = 0.0,
    lambda_sparse: float = 0.0,
    lambda_jerk: float = 0.0,
    phase: int = 0,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    target_clip_cm_s: float = 0.0,
    lambda_routing_aux: float = 0.0,
    wind_only_mask_full: Optional[np.ndarray] = None,
    routing_aux_margin: float = 0.024,
    lambda_compute: float = 0.0,
    selection_metric: str = "total",
) -> float:
    """
    Run validation (no gradient computation).

    Args:
        model: The NSMoR model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Device.
        lambda_reg: Router regularization weight.
        lambda_energy: ATP metabolic cost weight.
        lambda_sparse: Population sparsity L1 weight.
        lambda_jerk: Frame-based third-velocity-difference smoothness weight
            (legacy name "jerk"; NOT physical jerk — no dt scaling).
        phase: Training phase (1=frontend, 2=backend, 0=single-phase).
        target_mean: Target mean (cm/s) used when target normalization is
            enabled; paired with ``target_std`` to put ``y_true`` on the
            same scale the network predicts.  Default ``(0.0, 1.0)`` is
            the identity (no normalization).
        target_std: Target std (cm/s) used when target normalization is
            enabled.  Default ``(0.0, 1.0)`` is the identity.
        target_clip_cm_s: Robust clip magnitude (cm/s) applied to the
            validation target to mirror training.  ``0.0`` disables.
        lambda_routing_aux: Auxiliary routing loss weight for modality differentiation.
        wind_only_mask_full: Optional full-split boolean array for pure-wind trials.
        routing_aux_margin: Hinge margin for routing auxiliary loss.
        selection_metric: ``"total"`` (default) selects on the full validation
            objective; ``"mse"`` selects on the **frame-weighted pooled** masked
            MSE over the aligned eligible frames (t >= 1) alone (sum of aligned
            squared errors / total aligned frames, not a per-batch
            macro-average), so the selected checkpoint minimises the pooled
            quantity the scored primary metric divides by.  The returned float
            is the selected metric, so a caller's checkpoint-selection /
            patience comparison is unchanged in shape.

    Returns:
        Average validation loss under ``selection_metric`` (``"mse"`` returns
        the pooled masked MSE, or ``inf`` if no aligned frame was seen).  The
        pooled masked-MSE diagnostic (t >= 1) is additionally exposed as
        :attr:`validate.last_val_mse` (``None`` when no batch ran), mirroring
        :attr:`train_one_epoch.last_skip_counts`.
    """
    if selection_metric not in ("total", "mse"):
        raise ValueError(
            f"selection_metric must be 'total' or 'mse', got {selection_metric!r}"
        )
    validate.last_val_mse = None
    model.eval()
    total_loss = 0.0
    # Frame-weighted pooled masked MSE (sum of aligned squared errors / total
    # aligned frames).  Accumulate the raw sums here and divide once at the end
    # so the selected metric is the POOLED quantity the scored primary
    # (``1 - MSE_model/MSE_persist``) divides by, not a per-batch macro-average
    # (which would over-weight the small last batch).  Only the
    # ``selection_metric="mse"`` path consumes this; the ``"total"`` default
    # return is unchanged.
    total_se_sum = 0.0
    total_se_frames = 0
    n_batches = 0

    pbar = tqdm(loader, desc="Validation", leave=False, dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        # Ticket #16: Unpack batch — (X, Y, lengths) legacy 3-tuple, or
        # (X, Y, lengths, wind_only_mask) 4-tuple when metadata present.
        if len(batch) == 4:
            x_batch, y_batch, lengths, wind_only_mask = batch
            wind_only_mask = wind_only_mask.to(device).contiguous()
        else:
            x_batch, y_batch, lengths = batch
            # Extract wind_only_mask from full array by batch offset
            batch_size = x_batch.shape[0]
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            if wind_only_mask_full is not None and batch_end <= len(wind_only_mask_full):
                wind_only_mask = torch.from_numpy(wind_only_mask_full[batch_start:batch_end]).to(device)
            else:
                wind_only_mask = None

        x_batch = x_batch.to(device).contiguous()
        y_batch = y_batch.to(device).contiguous()
        lengths = lengths.to(device).contiguous()

        # Match the training-time target transformation (if enabled).
        if target_clip_cm_s > 0.0:
            y_batch = torch.clamp(y_batch, -target_clip_cm_s, target_clip_cm_s)
        if target_std != 1.0 or target_mean != 0.0:
            y_batch = (y_batch - target_mean) / target_std

        y_pred, internals = model(x_batch, lengths, return_internals=True)

        if phase == 1:
            # Phase 1: FrontendLoss — MSE only
            loss = criterion(
                y_pred=y_pred,
                y_true=y_batch,
                lengths=lengths,
            )
        else:
            # Phase 2 / single-phase: full bio loss
            g_gru = internals["routing_gates"][:, :, 1:2]
            g_lif = internals["routing_gates"][:, :, 0:1]  # Ticket #16
            lif_spikes = internals["lif_spikes"]
            _ponder = internals.get("refinement_ponder_cost", None)
            loss = criterion(
                y_pred=y_pred,
                y_true=y_batch,
                lengths=lengths,
                g_gru=g_gru,
                lambda_reg=lambda_reg,
                lif_spikes=lif_spikes,
                lambda_energy=lambda_energy,
                lambda_sparse=lambda_sparse,
                lambda_jerk=lambda_jerk,
                # Ticket #16: condition metadata is optional for legacy
                # datasets; no metadata means the routing auxiliary term is 0.
                lambda_routing_aux=lambda_routing_aux,
                wind_only_mask=wind_only_mask,
                routing_aux_margin=routing_aux_margin,
                lambda_compute=lambda_compute,
                refinement_ponder_cost=_ponder,
            )

        total_loss += loss.item()
        # Masked-MSE-only selection diagnostic on the ALIGNED eligible frames
        # (t >= 1, padding excluded) — the phase-2 protocol's checkpoint-
        # selection metric.  The loss' own MSE term is over t >= 0; the
        # single t=0 frame per trial has no persistence predecessor, so the
        # aligned partition is the honest one for selecting weights against the
        # skill-vs-persistence primary metric.  The t=0 exclusion is exact
        # (one frame per trial, ~0.04% of frames).  Used when
        # selection_metric == "mse".
        with torch.no_grad():
            B_t, T_t = y_pred.shape
            assert y_pred.shape == y_batch.shape == (B_t, T_t), (
                f"selection MSE shape mismatch: y_pred {tuple(y_pred.shape)} "
                f"vs y_true {tuple(y_batch.shape)}"
            )
            assert lengths.shape == (B_t,), (
                f"lengths shape {tuple(lengths.shape)} != ({B_t},)"
            )
            arange_t = torch.arange(T_t, device=y_pred.device).unsqueeze(0)
            _mask = (
                (arange_t >= 1) & (arange_t < lengths.unsqueeze(1))
            ).to(y_pred.dtype)
            assert _mask.shape == (B_t, T_t), (
                f"selection mask shape {tuple(_mask.shape)} != ({B_t}, {T_t})"
            )
            _se = (y_pred - y_batch) ** 2
            assert _se.shape == (B_t, T_t), (
                f"selection squared-error shape {tuple(_se.shape)} != "
                f"({B_t}, {T_t})"
            )
            total_se_sum += float((_se * _mask).sum())
            total_se_frames += int(_mask.sum().item())
        n_batches += 1
        pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(n_batches, 1)
    # Frame-weighted pooled masked MSE over the WHOLE split (one divide).
    pooled_mse = (
        total_se_sum / total_se_frames if total_se_frames > 0 else None
    )
    validate.last_val_mse = pooled_mse
    if selection_metric == "mse":
        return pooled_mse if pooled_mse is not None else float("inf")
    return avg_loss


# ═══════════════════════════════════════════════════════════════
# 6.  Evaluation Metrics & Loss Curve Plotting
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def _sustained_run(mask: np.ndarray, min_run: int = 2) -> np.ndarray:
    """Return a copy of ``mask`` keeping only elements that belong to a run of
    at least ``min_run`` consecutive ``True`` values.

    Used to exclude isolated single-frame out-of-band spikes (e.g. ~1e7 cm/s
    tracking artifacts) from the escape audit, keeping only temporally-extended
    events which we call escapes.  Runs shorter than ``min_run`` are zeroed.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"_sustained_run expects a 1-D mask, got {mask.ndim}-D")
    if min_run <= 1:
        return mask.copy()
    keep = np.zeros(mask.shape[0], dtype=bool)
    run = 0
    for i, on in enumerate(mask):
        if on:
            run += 1
        else:
            if run >= min_run:
                keep[i - run : i] = True
            run = 0
    if run >= min_run:  # trailing run reaches the array end
        keep[mask.shape[0] - run :] = True
    return keep


def _lag1_skill_ratio(mse_model: float, mse_base: float) -> Optional[float]:
    """Return ``1 - mse_model/mse_base`` or ``None`` when undefined.

    ``None`` (not a fabricated 0/NaN) is returned when the lag-one baseline
    MSE is zero, either input is non-finite, or the ratio itself is not
    representable (a finite huge model MSE over a finite tiny baseline
    overflows float64).  ``skill > 0`` means the model beats the lag-one
    target-history predictor; ``skill <= 0`` means it does not.  This is a
    conditional comparison against that one comparator, not a universal
    verdict: a model can score ``skill <= 0`` here yet still beat a trivial
    zero-predictor.
    """
    if not (math.isfinite(mse_model) and math.isfinite(mse_base)):
        return None
    if mse_base <= 0.0:
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        skill = 1.0 - mse_model / mse_base
    if not math.isfinite(skill):
        return None
    return float(skill)


def _paired_metrics(
    true_vals: np.ndarray,
    model_vals: np.ndarray,
    base_vals: np.ndarray,
    *,
    empty_reason: str,
) -> Dict[str, Any]:
    """Score aligned predictions, failing closed on unrepresentable errors.

    Subtract and square here, not at callers: signed residuals retain MAE and
    let us reject nonzero errors whose squares underflow to zero. Only empty
    blocks and undefined/unrepresentable ratios or target variance yield null
    with a keyed reason; residuals, squares and error reductions must be finite.
    """
    true_vals, model_vals, base_vals = (
        np.asarray(v, dtype=np.float64) for v in (true_vals, model_vals, base_vals)
    )
    assert true_vals.ndim == 1, f"Expected 1-D targets, got {true_vals.shape}"
    assert true_vals.shape == model_vals.shape == base_vals.shape, (
        f"Expected aligned {true_vals.shape}, got {model_vals.shape}/{base_vals.shape}"
    )
    n = int(true_vals.size)
    out: Dict[str, Any] = {"n_frames": n}
    if n == 0:
        for key in ("mse", "rmse", "mae", "r2", "baseline_mse",
                    "baseline_rmse", "baseline_mae", "baseline_r2",
                    "skill_vs_persistence"):
            out[key] = None
            out[f"{key}_unavailable_reason"] = empty_reason
        return out

    sums: List[float] = []
    for prefix, values in (("", model_vals), ("baseline_", base_vals)):
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                residual = true_vals - values
                squared = residual ** 2
                if not np.isfinite(residual).all() or not np.isfinite(squared).all():
                    raise ValueError("non-finite residual or squared error")
                if np.any((residual != 0.0) & (squared == 0.0)):
                    raise ValueError("nonzero residual square underflow to zero")
                sumsq = float(squared.sum())
                mse = sumsq / n
                mae = float(np.abs(residual).mean())
            if not all(math.isfinite(v) for v in (sumsq, mse, mae)):
                raise ValueError("non-finite error reduction")
            if (sumsq > 0.0 and mse == 0.0) or (np.any(residual) and mae == 0.0):
                raise ValueError("nonzero error reduction underflow to zero")
        except (FloatingPointError, ValueError) as exc:
            raise ValueError(
                f"{prefix or 'model_'}error arithmetic is not representable: {exc}"
            ) from exc
        sums.append(sumsq)
        out[f"{prefix}mse"] = mse
        out[f"{prefix}rmse"] = math.sqrt(mse)
        out[f"{prefix}mae"] = mae

    # Centered variance avoids raw-Gram cancellation. Distinguish a genuinely
    # constant target from a nonzero spread lost to float64 arithmetic.
    variance_reason = "zero_target_variance"
    sumsq_true = 0.0
    if not np.all(true_vals == true_vals[0]):
        variance_reason = "nonrepresentable_target_variance"
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                centered = true_vals - true_vals.mean()
                squared = centered ** 2
                sumsq_true = float(squared.sum())
            if (not np.isfinite(centered).all() or not math.isfinite(sumsq_true)
                    or np.any((centered != 0.0) & (squared == 0.0))):
                sumsq_true = 0.0
        except FloatingPointError:
            sumsq_true = 0.0
    for key, sumsq in zip(("r2", "baseline_r2"), sums):
        out[key] = _lag1_skill_ratio(sumsq, sumsq_true)
        if out[key] is None:
            out[f"{key}_unavailable_reason"] = (
                variance_reason if sumsq_true == 0.0 else "non_finite_r2"
            )

    out["skill_vs_persistence"] = _lag1_skill_ratio(out["mse"], out["baseline_mse"])
    if out["skill_vs_persistence"] is None:
        out["skill_vs_persistence_unavailable_reason"] = (
            "zero_baseline_mse" if out["baseline_mse"] == 0.0
            else "non_finite_skill_ratio"
        )
    return out


def _merge_block(
    metrics: Dict[str, Any],
    block: Dict[str, Any],
    *,
    prefix: str,
    count_key: str,
) -> None:
    """Merge one :func:`_paired_metrics` block into ``metrics`` in place.

    ``prefix`` namespaces the metric keys (``""`` for the pooled headline,
    ``"escape_"``/``"rest_"`` for the bands); the block's ``n_frames`` becomes
    ``metrics[count_key]``.
    """
    for key, value in block.items():
        if key == "n_frames":
            metrics[count_key] = value
        else:
            metrics[f"{prefix}{key}"] = value


def persistence_benchmark_metrics(
    y_true_seqs: Sequence[np.ndarray],
    y_pred_seqs: Sequence[np.ndarray],
    *,
    escape_band_cm_s: float = 10.0,
    target_clip_cm_s: float = 0.0,
) -> Dict[str, Any]:
    """Paired lag-one target-history benchmark over outer-validation sequences.

    Supplemental skill check for the existing ``compute_metrics`` seam: a
    trivial persistence predictor (``y_hat[t] = y_true[t-1]``) is a strong
    comparator for smooth, autocorrelated kinematics.  A model scoring
    ``skill_vs_persistence <= 0`` does not beat that one comparator on these
    frames — a conditional statement, not a claim that the model has no
    temporal skill at all (it may still beat a zero predictor).

    Both the model and the comparator are scored on **exactly the same
    frames**, per trial: the eligible set is ``t >= 1`` (the lag-one
    comparator needs a previous frame), so no frame is ever compared across a
    trial boundary and the comparator reads ``y_true[t-1]`` (the target),
    never a feature channel.  Every input sequence is reported as an explicit
    per-trial row — ordinal, length, eligible count, status — so no trial is
    dropped silently.

    Args:
        y_true_seqs: Per-sequence 1-D ground-truth targets, already cropped
            to the same window the model was scored on, in physical units
            (cm/s).  Unpadded true lengths only.
        y_pred_seqs: Per-sequence 1-D model predictions, aligned
            element-for-element with ``y_true_seqs`` (same crop, same
            length, same physical units).
        escape_band_cm_s: Absolute-velocity magnitude (cm/s) for the
            escape/rest split, matching ``compute_metrics``.
        target_clip_cm_s: Symmetric robust clip (cm/s) applied to BOTH the
            model prediction and the target before the headline errors are
            scored, mirroring ``compute_metrics``' headline convention
            (``0.0`` disables).  The escape/rest band membership and band
            errors are always measured on the RAW, unclipped values, exactly
            as the legacy band audit does.

    Returns:
        A supplemental metrics dict with headline MSE/RMSE/MAE/R² for both the
        model and the lag-one comparator, the escape/rest breakdown, the
        scalar ``skill_vs_persistence`` (``1 - MSE_model/MSE_base``, or
        ``None`` when undefined), the frame/trial denominators, and the
        ordered per-trial ``trials`` list.  Each ``trials`` row carries the
        trial ordinal, true length, eligible-frame count, an explicit status,
        and that trial's paired ``model_mse`` / ``baseline_mse`` /
        ``delta_model_minus_baseline_mse`` / ``skill_vs_persistence`` — so the
        pooled numbers are auditable per trial and no trial is dropped
        silently.  Degenerate bands and undefined statistics report ``None``
        plus a ``*_unavailable_reason`` string rather than a fabricated value.
        Every value is finite or ``None``, so ``json.dumps(..., allow_nan=False)``
        succeeds.

    Raises:
        ValueError: On mismatched sequence counts, per-sequence shape
            mismatch, a non-real/boolean/non-scalar band or clip parameter,
            or any non-finite value on an eligible true/model frame (or on
            the lag-one predecessor target), or unrepresentable residual,
            squared error or error reduction. The benchmark fails closed
            rather than dropping bad frames or emitting NaN/Inf.
    """
    for name, value in (("escape_band_cm_s", escape_band_cm_s),
                        ("target_clip_cm_s", target_clip_cm_s)):
        # Accept only a finite real scalar: reject bool (an int subclass) and
        # array-likes/sequences rather than silently casting them.
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"{name} must be a finite real scalar, got {value!r}"
            )
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")

    if len(y_true_seqs) != len(y_pred_seqs):
        raise ValueError(
            f"persistence benchmark requires aligned sequences, got "
            f"{len(y_true_seqs)} targets vs {len(y_pred_seqs)} predictions"
        )

    true_raw: List[np.ndarray] = []
    pred_raw: List[np.ndarray] = []
    prev_raw: List[np.ndarray] = []
    is_escape_seq: List[np.ndarray] = []
    trials: List[Dict[str, Any]] = []
    clip = float(target_clip_cm_s)
    clip_on = clip > 0.0

    for i, (true_i, pred_i) in enumerate(zip(y_true_seqs, y_pred_seqs)):
        true_i = np.asarray(true_i, dtype=np.float64)
        pred_i = np.asarray(pred_i, dtype=np.float64)
        # Require exactly 1-D BEFORE any reshape: a 0-D scalar (np.array(5.0))
        # or a >=2-D block is rejected, never silently flattened/reshaped into
        # a bogus "sequence" (a scalar would otherwise masquerade as a
        # length-one trial indistinguishable from np.array([5.0])).
        if true_i.ndim != 1 or pred_i.ndim != 1:
            raise ValueError(
                f"sequence {i}: expected 1-D per-sequence arrays, got shapes "
                f"{true_i.shape} / {pred_i.shape}"
            )
        if true_i.shape != pred_i.shape:
            raise ValueError(
                f"sequence {i}: y_true {true_i.shape} != y_pred {pred_i.shape}"
            )
        n = true_i.size
        if n < 2:
            # No t>=1 frame.  It contributes no frame, but is still reported
            # with an explicit status so no trial vanishes from the count.
            reason = "no_t_ge_1_frame"
            trials.append({"trial": i, "length": n, "eligible": 0,
                           "status": "empty_eligible_no_t_ge_1_frame",
                           "model_mse": None, "baseline_mse": None,
                           "delta_model_minus_baseline_mse": None,
                           "skill_vs_persistence": None,
                           "model_mse_unavailable_reason": reason,
                           "baseline_mse_unavailable_reason": reason,
                           "delta_model_minus_baseline_mse_unavailable_reason": reason,
                           "skill_vs_persistence_unavailable_reason": reason})
            continue

        eligible = np.arange(1, n)  # t >= 1, per sequence; never cross-sequence
        t_prev = eligible - 1
        # The predecessor target is consumed by every eligible frame, so it
        # must be finite (t=0 is a predecessor even though never scored).
        true_prev = true_i[t_prev]
        if not np.isfinite(true_prev).all():
            bad = int(t_prev[~np.isfinite(true_prev)][0])
            raise ValueError(
                f"sequence {i}: lag-one predecessor target at t={bad} is "
                f"non-finite; cannot score this trial"
            )
        # Scored true/model frames must be finite: fail closed instead of
        # emitting NaN/Inf or silently dropping bad frames.  A NaN model
        # output at the unscored t=0 is permitted (it is never used).
        for label, values in (("true target", true_i), ("model prediction", pred_i)):
            scored = values[eligible]
            if not np.isfinite(scored).all():
                bad = int(eligible[~np.isfinite(scored)][0])
                raise ValueError(f"sequence {i}: non-finite {label} at t={bad}")

        # Escape/rest membership is derived from the ORIGINAL target with the
        # existing _sustained_run guard, BEFORE the t>=1 slice, then sliced —
        # never recomputed on the sliced array (which would shift run
        # boundaries at the cut).
        escape_full = _sustained_run(np.abs(true_i) >= escape_band_cm_s, min_run=2)
        true_raw.append(true_i[eligible])
        pred_raw.append(pred_i[eligible])
        prev_raw.append(true_prev)
        is_escape_seq.append(escape_full[eligible])

        # Per-trial paired metrics in the same (clipped) headline space, so a
        # trial's contribution to the pooled numbers is auditable and no trial
        # is dropped silently.  Reuse the SAME paired helper (not a duplicated
        # formula) so a trial row carries identical guarded values and
        # null+reason semantics as the pooled block.
        t_true, t_pred, t_prev = true_i[eligible], pred_i[eligible], true_prev
        if clip_on:
            t_true = np.clip(t_true, -clip, clip)
            t_pred = np.clip(t_pred, -clip, clip)
            t_prev = np.clip(t_prev, -clip, clip)
        t = _paired_metrics(t_true, t_pred, t_prev, empty_reason="empty_trial")
        t_model_mse, t_base_mse = t["mse"], t["baseline_mse"]
        delta = t_model_mse - t_base_mse
        if not math.isfinite(delta):
            raise ValueError("per-trial MSE delta is not representable")
        row: Dict[str, Any] = {
            "trial": i, "length": n, "eligible": int(eligible.size),
            "status": "scored",
            "model_mse": t_model_mse,
            "baseline_mse": t_base_mse,
            "delta_model_minus_baseline_mse": delta,
            "skill_vs_persistence": t["skill_vs_persistence"],
        }
        if t["skill_vs_persistence"] is None:
            row["skill_vs_persistence_unavailable_reason"] = (
                t["skill_vs_persistence_unavailable_reason"]
            )
        trials.append(row)

    metrics: Dict[str, Any] = {
        "n_trials": len(y_true_seqs),
        "n_scored_trials": int(sum(e.size > 0 for e in true_raw)),
        "n_eligible_frames": int(sum(e.size for e in true_raw)),
        "trials": trials,
    }

    if not true_raw:
        # No eligible frames anywhere: every field is an explicit null with a
        # reason.  Never a fabricated 0/NaN/Inf.
        empty = _paired_metrics(np.zeros(0), np.zeros(0), np.zeros(0),
                                empty_reason="no_sequence_with_at_least_two_frames")
        _merge_block(metrics, empty, prefix="", count_key="n_pooled_frames")
        _merge_block(metrics, empty, prefix="escape_", count_key="n_escape_frames")
        _merge_block(metrics, empty, prefix="rest_", count_key="n_rest_frames")
        return metrics

    y_true = np.concatenate(true_raw)
    y_pred = np.concatenate(pred_raw)
    y_prev = np.concatenate(prev_raw)
    is_escape = np.concatenate(is_escape_seq)

    # Headline convention: same symmetric clip as ``compute_metrics`` applied
    # to the model prediction and to the target (and to the lag-one comparator
    # values, which are target frames), so model/true/lag are scored in the
    # same space.
    true_score, pred_score, prev_score = y_true, y_pred, y_prev
    if clip_on:
        true_score = np.clip(true_score, -clip, clip)
        pred_score = np.clip(pred_score, -clip, clip)
        prev_score = np.clip(prev_score, -clip, clip)

    _merge_block(
        metrics,
        _paired_metrics(true_score, pred_score, prev_score,
                        empty_reason="no_pooled_frames"),
        prefix="", count_key="n_pooled_frames",
    )

    # ── Escape / rest breakdown: membership AND error from RAW values ──
    for band, mask in (("escape", is_escape), ("rest", ~is_escape)):
        _merge_block(
            metrics,
            _paired_metrics(y_true[mask], y_pred[mask], y_prev[mask],
                            empty_reason=f"empty_{band}_band"),
            prefix=f"{band}_", count_key=f"n_{band}_frames",
        )

    return metrics


def _rescale_to_physical(
    values: np.ndarray, target_std: float, target_mean: float,
) -> np.ndarray:
    """Rescale standardized values to physical units in float64, fail closed.

    The multiply is done in float64 so a float32 input scaled by a tiny/large
    ``target_std`` cannot underflow to a spurious 0.0 (silently losing a
    representable error) or overflow to inf.  Genuinely unrepresentable
    float64 arithmetic raises ``ValueError`` rather than emitting NaN/Inf or a
    silently-lost nonzero value.
    """
    values = np.asarray(values, dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            scaled = values * target_std
            rescaled = scaled + target_mean
    except FloatingPointError as exc:
        raise ValueError(
            f"rescaled values are not representable in float64: {exc}"
        ) from exc
    if not np.isfinite(rescaled).all():
        raise ValueError(
            "rescaled values are not representable in float64: non-finite result"
        )
    if target_std != 0.0 and np.any((values != 0.0) & (scaled == 0.0)):
        raise ValueError(
            "rescaled values are not representable in float64: nonzero input "
            "underflowed to zero"
        )
    return rescaled


def _json_scalar(value: Any) -> Any:
    """Coerce a NumPy scalar to a native Python scalar for strict JSON.

    ``np.float32``/``np.int64`` (accepted band parameters) are not
    serialisable by the stdlib JSON encoder; ``np.generic.item()`` yields the
    native ``float``/``int`` that ``json.dumps(..., allow_nan=False)`` emits
    as a plain number.  ``np.longdouble`` is the exception: its ``.item()``
    stays a ``np.longdouble``, so it is promoted to ``float`` when that is
    lossless, else rejected rather than silently rounded.  Non-NumPy values
    pass through unchanged.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)):
        return value
    as_float = float(value)
    if as_float != value or not math.isfinite(as_float):
        raise ValueError(f"{value!r} is not a representable JSON scalar")
    return as_float


def _fmt_optional(value: Optional[float], spec: str = ".4f") -> str:
    """Format an optional metric for logging, spelling out unavailability.

    ``None`` (a guarded-but-undefined metric) is rendered as ``"unavailable"``
    rather than letting ``%.4f`` raise ``TypeError`` or substituting NaN.
    """
    return "unavailable" if value is None else format(value, spec)


def _log_best_metrics(metrics: Dict[str, Any]) -> None:
    """Log the best-model headline and escape audit, tolerating null metrics.

    The opted-in ``compute_metrics`` return can carry ``None`` for undefined
    statistics (e.g. constant-target R², empty-band RMSE); those are spelled
    out as ``unavailable`` instead of letting a ``%.4f`` directive raise
    ``TypeError`` on ``None`` (or substituting a fabricated NaN).
    """
    logger.info(
        "Best model metrics — MSE: %s  RMSE: %s  MAE: %s  R²: %s",
        _fmt_optional(metrics["mse"], ".6f"),
        _fmt_optional(metrics["rmse"], ".6f"),
        _fmt_optional(metrics["mae"], ".6f"),
        _fmt_optional(metrics["r2"]),
    )
    if "escape_rmse" in metrics:
        logger.info(
            "Escape-signal audit — escape_band=%.1f cm/s  n_escape=%d (%.3f%% of frames)  "
            "escape_rmse=%s  resting_rmse=%s",
            metrics["escape_band_cm_s"],
            metrics["n_escape_frames"],
            metrics["escape_ratio"] * 100.0,
            _fmt_optional(metrics["escape_rmse"]),
            _fmt_optional(metrics["resting_rmse"]),
        )


@torch.no_grad()
def compute_metrics(
    model: NSMoRCore,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    target_clip_cm_s: float = 0.0,
    escape_band_cm_s: float = 10.0,
    persistence_benchmark: bool = False,
) -> Dict[str, Any]:
    """
    Compute regression metrics on a dataset using the given model.

    Collects all predictions and ground-truth values (respecting
    variable sequence lengths via masking), then computes MSE, RMSE,
    MAE, and R² in a single pass.

    Args:
        model: Trained model (should be in eval mode).
        loader: DataLoader for the evaluation split.
        device: Device to run inference on.
        target_mean: Training-target mean (cm/s).  When target
            normalization was enabled during training, predictions are
            in standardised units and are rescaled back to cm/s via
            ``pred_cm_s = pred_norm * target_std + target_mean`` before
            computing metrics, so reported numbers are in physical units.
            Default ``(0.0, 1.0)`` is the identity.
        target_std: Training-target std (cm/s).
        target_clip_cm_s: Robust clip magnitude (cm/s) applied to both
            predictions and targets before computing metrics, mirroring
            the training-time target clip.  ``0.0`` disables.
        persistence_benchmark: When ``True``, additionally attach a
            supplemental ``"persistence_benchmark"`` sub-object scoring the
            model against a paired lag-one target-history predictor (see
            :func:`persistence_benchmark_metrics`). The full opted-in result
            uses guarded float64 scoring: undefined R²/empty bands become null
            with keyed reasons; unrepresentable errors raise ValueError.
            Legacy keys and behavior are unchanged when ``False`` (default).

    Returns:
        Dictionary with keys ``"mse"``, ``"rmse"``, ``"mae"``, ``"r2"``
        in physical (cm/s) units, plus a high-velocity-band escape-signal
        audit: ``"escape_band_cm_s"``, ``"n_escape_frames"``,
        ``"escape_rmse"`` (RMSE on ``|y_true| >= escape_band_cm_s`` frames),
        ``"resting_rmse"`` (RMSE on the remaining resting frames), and
        ``"escape_ratio"``.  The band split exposes whether the
        network learned the high-velocity escape transient or only the
        resting-mode bulk (a bulk-fitting model scores well on clipped
        ``rmse``/``r2`` yet shows ``escape_rmse >> resting_rmse``).

        The band audit is measured on the **raw (unclipped)** prediction and
        target, so frames whose true magnitude exceeds ``target_clip_cm_s`` are
        still measured at their true magnitude rather than flattened to the
        clip boundary.  The band is an absolute-velocity-magnitude heuristic
        (``escape_band_cm_s``), NOT a wind-stimulus-conditioned or
        per-trial-baseline-subtracted escape definition — see the
        ``escape_band_cm_s`` config docstring.  A single band value does not by
        itself prove escape learning; sweep the band for sensitivity.

    Raises:
        ValueError: When the physical-unit rescale is not representable in
            float64 (overflow to inf, or a nonzero standardized prediction
            underflowing to 0.0), rather than emitting NaN/Inf or silently
            losing a nonzero value.
    """
    model.eval()
    all_pred: List[np.ndarray] = []
    all_true: List[np.ndarray] = []

    for batch in loader:
        if len(batch) == 4:
            x_batch, y_batch, lengths, _ = batch
        else:
            x_batch, y_batch, lengths = batch
        x_batch = x_batch.to(device).contiguous()
        y_batch = y_batch.to(device).contiguous()
        lengths = lengths.to(device).contiguous()

        y_pred, _ = model(x_batch, lengths, return_internals=True)

        # Mask padded timesteps per sequence
        for i in range(x_batch.size(0)):
            n = int(lengths[i])
            all_pred.append(y_pred[i, :n].cpu().numpy())
            all_true.append(y_batch[i, :n].cpu().numpy())

    # Promote to float64 BEFORE any physical-unit rescaling so the rescale is
    # validated (and computed) at float64 precision — a float32 input scaled
    # by a tiny ``target_std`` must not underflow to a spurious 0.0.
    y_pred_all = np.concatenate(all_pred).astype(np.float64)
    y_true_all = np.concatenate(all_true).astype(np.float64)

    # Rescale predictions from standardized units back to cm/s (units
    # matching y_true) so metrics are reported in physical velocity units.
    if target_std != 1.0 or target_mean != 0.0:
        y_pred_all = _rescale_to_physical(y_pred_all, target_std, target_mean)

    # Symmetric robust clip before scoring (mirrors training-target clip).
    # Keep RAW copies of both prediction and target (post-rescale, pre-clip)
    # for the high-velocity-band audit below, so a real escape transient whose
    # true magnitude exceeds ±clip is still measured at its raw magnitude — not
    # flattened to the clip boundary (which would mask silent escape-signal loss).
    y_pred_all_raw = y_pred_all.copy()
    y_true_all_raw = y_true_all.copy()
    if target_clip_cm_s > 0.0:
        y_pred_all = np.clip(y_pred_all, -target_clip_cm_s, target_clip_cm_s)
        y_true_all = np.clip(y_true_all, -target_clip_cm_s, target_clip_cm_s)

    benchmark: Optional[Dict[str, Any]] = None
    headline: Dict[str, Any] = {}
    if persistence_benchmark:
        # Validate the paired benchmark BEFORE any legacy reduction. Use the
        # same guarded scoring for full-frame opt-in fields (including t=0).
        bench_pred = all_pred
        if target_std != 1.0 or target_mean != 0.0:
            bench_pred = [
                _rescale_to_physical(p, target_std, target_mean) for p in all_pred
            ]
        benchmark = persistence_benchmark_metrics(
            all_true, bench_pred, escape_band_cm_s=escape_band_cm_s,
            target_clip_cm_s=target_clip_cm_s,
        )
        headline = _paired_metrics(
            y_true_all, y_pred_all, y_true_all, empty_reason="no_frames",
        )
        mse, rmse, mae, r2 = (headline[k] for k in ("mse", "rmse", "mae", "r2"))
    else:
        mse = float(mean_squared_error(y_true_all, y_pred_all))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true_all, y_pred_all))
        r2 = float(r2_score(y_true_all, y_pred_all))

    # ── High-velocity-band escape-signal check ────────────────
    # Reviewer requirement: a bulk-fitting model can report an excellent
    # clipped RMSE/R² while failing to learn the biologically meaningful
    # escape transient (resting-dominant, zero-inflated target).  Break the
    # validation error into resting vs high-velocity (escape) frames so that
    # silent escape-signal loss is visible in the reported metrics rather
    # than masked by the resting-mode bulk.
    #   high-speed band: |y_true| >= escape_band_cm_s frames.
    # Critical: membership AND magnitude are both measured on the RAW
    # (unclipped) prediction and target.  Measuring in the raw space means an
    # escape transient whose true magnitude exceeds ±target_clip_cm_s is NOT
    # flattened to the clip boundary — a model predicting a flat ~clip for
    # every large escape scores a large raw escape_rmse, so the audit actually
    # exposes silent escape-signal loss rather than being blinded by the clip.
    # (The overall mse/rmse/mae/r2 above remain clipped-space, matching the
    # training-target clip; the band audit deliberately reports raw magnitude.)
    #
    # Sustained-membership guard: escape membership requires the frame to be
    # part of a *run* of at least ``min_run=2`` consecutive
    # |y_true|>=escape_band_cm_s frames.  min_run is a conservative artifact
    # filter, NOT a biophysical timescale claim: tracking artifacts are
    # single-frame sensor jumps, so min_run=2 already excludes them, while a
    # larger value would start trimming genuine short escapes.  (At 500 Hz,
    # 2 frames = 4 ms — well below the ~10-100 ms kick; the guard trades
    # sensitivity for artifact-robustness.  Sweep band x min_run for
    # sensitivity — see task/roadmap.)
    #
    # The guard is applied PER SEQUENCE, before concatenation: frame adjacency
    # in the concatenated array is meaningless across trial boundaries, so a
    # run computed post-concat could bridge two unrelated trials (false escape)
    # or split one truncated at a boundary.
    over_seq = [np.abs(t) >= escape_band_cm_s for t in all_true]
    keep_seq = [_sustained_run(o, min_run=2) for o in over_seq]
    is_escape = np.concatenate(keep_seq) if keep_seq else np.zeros(0, dtype=bool)
    n_escape = int(is_escape.sum())
    if persistence_benchmark:
        bands = {
            key: _paired_metrics(
                y_true_all_raw[mask], y_pred_all_raw[mask], y_true_all_raw[mask],
                empty_reason=reason,
            )
            for key, mask, reason in (
                ("escape_rmse", is_escape, "empty_escape_band"),
                ("resting_rmse", ~is_escape, "empty_rest_band"),
            )
        }
        escape_rmse = bands["escape_rmse"]["rmse"]
        resting_rmse = bands["resting_rmse"]["rmse"]
    else:
        escape_rmse = float(np.sqrt(mean_squared_error(
            y_true_all_raw[is_escape], y_pred_all_raw[is_escape]))) if n_escape else float("nan")
        resting_rmse = float(np.sqrt(mean_squared_error(
            y_true_all_raw[~is_escape], y_pred_all_raw[~is_escape]))) if (~is_escape).any() else float("nan")

    metrics: Dict[str, Any] = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "escape_band_cm_s": _json_scalar(escape_band_cm_s),
        "n_escape_frames": float(n_escape),
        "escape_rmse": escape_rmse,
        "resting_rmse": resting_rmse,
        "escape_ratio": n_escape / max(1, int(y_true_all.size)),
    }

    if persistence_benchmark:
        metrics["persistence_benchmark"] = benchmark
        for key in ("mse", "rmse", "mae", "r2"):
            if metrics[key] is None:
                metrics[f"{key}_unavailable_reason"] = (
                    headline[f"{key}_unavailable_reason"]
                )
        for key, block in bands.items():
            if metrics[key] is None:
                metrics[f"{key}_unavailable_reason"] = block["rmse_unavailable_reason"]

    return metrics


def sweep_escape_sensitivity(
    all_true: List[np.ndarray],
    all_pred: List[np.ndarray],
    bands_cm_s: List[float],
    min_runs: List[int] = (1, 2, 3),
) -> List[Dict[str, float]]:
    """
    Band x min_run sensitivity table for the escape audit.

    Rescores the SAME per-sequence raw predictions/targets (as collected by
    :func:`compute_metrics`) at every escape-band threshold and sustained-run
    length, so the reported headline number can be shown to be robust (or
    not) to its two free parameters rather than a single arbitrary-threshold
    point estimate.

    Returns one dict per (band, min_run) pair with keys ``band_cm_s``,
    ``min_run``, ``n_escape_frames``, ``n_escape_events`` (number of
    contiguous per-sequence runs — the event-level unit), ``escape_rmse``,
    ``resting_rmse``, and ``escape_ratio``.
    """
    rows: List[Dict[str, float]] = []
    y_pred_all_raw = np.concatenate(all_pred)
    y_true_all_raw = np.concatenate(all_true)
    err_sq_all = (y_pred_all_raw - y_true_all_raw) ** 2
    for band in bands_cm_s:
        for mr in min_runs:
            keep_seq = [
                _sustained_run(np.abs(t) >= band, min_run=mr) for t in all_true
            ]
            is_escape = np.concatenate(keep_seq)
            n_escape = int(is_escape.sum())
            # Event count: transitions into an over-band kept run.
            n_events = sum(
                int(np.count_nonzero(k[1:] & ~k[:-1]) + int(k[0]))
                for k in keep_seq
            )
            esc = float(np.sqrt(err_sq_all[is_escape].mean())) if n_escape else float("nan")
            rest = (
                float(np.sqrt(err_sq_all[~is_escape].mean()))
                if (~is_escape).any() else float("nan")
            )
            rows.append({
                "band_cm_s": band,
                "min_run": mr,
                "n_escape_frames": n_escape,
                "n_escape_events": n_events,
                "escape_rmse": esc,
                "resting_rmse": rest,
                "escape_ratio": n_escape / max(1, is_escape.size),
            })
    return rows


def plot_loss_curve(
    history: Dict[str, List[float]],
    output_dir: Path,
    history_start_epoch: int = 0,
) -> Path:
    """
    Plot train/val loss curves and save to disk.

    Args:
        history: Dictionary with ``"train_loss"`` and ``"val_loss"`` lists.
        output_dir: Directory to save the figure.
        history_start_epoch: True 0-based epoch of ``history`` entry 0.  A
            legacy resume starts its recorded history at its resume epoch, so
            the x-axis begins there rather than falsely at epoch 1.

    Returns:
        Path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)

    first = history_start_epoch + 1  # 1-based label of the first entry
    epochs = range(first, first + len(history["train_loss"]))
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.5)
    # A ``None`` slot is an epoch with no observed validation diagnostic (no
    # validation loader, or a declared gap).  It is plotted as ``nan`` so
    # matplotlib BREAKS the line at that position (a native masked gap): the
    # curve must neither be drawn through an unobserved epoch nor silently
    # CONNECT across it, and dropping the point would also collapse the
    # position of every later value.  Recorded non-finite diagnostics are
    # left as-is for matplotlib's own handling.
    val_values = [
        float("nan") if v is None else v for v in history["val_loss"]
    ]
    ax.plot(epochs, val_values, label="Val Loss", linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = output_dir / "loss_curve.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════
# 7.  Main Train Function
# ═══════════════════════════════════════════════════════════════

def _recovery_terminal(
    patience: int,
    epochs_without_improvement: int,
    *,
    two_phase: bool,
    current_phase: int,
    num_epochs: int,
    phase1_epochs: Optional[int],
) -> bool:
    """The early-stopping stop predicate, shared by the loop and a resume.

    ``train`` applies this at every completed-epoch boundary.  A resumed
    checkpoint was published at such a boundary but BEFORE the decision was
    evaluated, so the resume must evaluate the same predicate before it
    executes any epoch.  Both call sites pass the same values the loop sees at
    the boundary, so the phase-1 exemption and the phase-boundary counter
    reset (which happens before this is consulted) behave identically.
    """
    return (
        patience > 0
        and epochs_without_improvement >= patience
        and not (
            two_phase
            and current_phase == 1
            and num_epochs > (phase1_epochs or 0)
        )
    )


def train(
    config: ExperimentConfig,
    lambda_reg: Optional[float] = None,
    phase1_epochs: Optional[int] = None,
    dataset_path: Optional[str] = None,
    use_lazy_loading: bool = False,
    nested_prior_artifact: Optional[str] = None,
    *,
    require_nested_validation: bool = False,
    trusted_historical_checkpoint_sha256=None,
    trusted_historical_artifact_sha256=None,
) -> Dict[str, Any]:
    """
    Full training pipeline.

    Supports **two-phase training** (Hybrid Funnel) when
    ``phase1_epochs`` is set:

    - **Phase 1** (epochs 0 … phase1_epochs-1):
      Freeze :class:`BioDecisionCore`, train
      :class:`FrontendEncoder` with :class:`FrontendLoss`
      (simple MSE).  The ``.detach()`` boundary ensures
      gradients never reach the backend.

    - **Phase 2** (epochs phase1_epochs … num_epochs-1):
      Freeze ``FrontendEncoder``, train ``BioDecisionCore``
      with :class:`BioDecisionLoss` (MSE + router reg +
      ATP + sparsity + frame-based smoothness).

    When ``phase1_epochs`` is ``None`` (default), the pipeline
    runs in single-phase mode — fully backward compatible.

    Args:
        config: Parsed experiment configuration.
        lambda_reg: Router regularization weight.  ``None``
            (default) falls through to ``config.loss.lambda_reg``,
            matching the ``--lambda_reg`` CLI sentinel.  Pass
            ``0.0`` to disable router regularization outright.
        phase1_epochs: Number of Phase 1 epochs.  ``None`` =
            single-phase mode (backward compatible).  ``0`` =
            skip Phase 1 entirely (start with Phase 2).
        dataset_path: Processed dataset to train on.  ``None`` keeps the
            historical default (``data/processed/nsmor_dataset.pt``);
            the pipeline passes the dataset the current ETL produced so
            training cannot silently consume a leftover file.
        use_lazy_loading: Use ELT lazy-loading mode (incompatible with
            ``nested_prior_artifact``).
        nested_prior_artifact: Optional nested-prior artifact from
            ``scripts/evaluate_nested_prior.py``.  When set, the exact
            persisted outer train/val split and leak-free nested priors
            replace the recomputed split and global OOF priors.
            ``None`` produces diagnostic global OOF validation, which can
            encode eventual outer-validation labels and is ineligible for
            strict QC/release gates.
        require_nested_validation: Refuse diagnostic global OOF validation.
            The CLI enables this by default; programmatic diagnostic callers
            keep the compatibility default with explicit validation_scope.

    Returns:
        Dictionary with ``"best_val_loss`` and ``"final_train_loss"``.

    Raises:
        ValueError: If no training data is provided (loader is None).
        FileExistsError: If a fresh run would reuse an existing best checkpoint.
    """
    config.validate()
    if require_nested_validation and nested_prior_artifact is None:
        raise ValueError(
            "Scored validation requires a nested prior artifact; global OOF "
            "scores are diagnostic only. Supply --nested_prior_artifact or "
            "explicitly use --diagnostic_only for diagnostics."
        )

    # Reject stale best weights before building a model or writing any outputs.
    # Explicit resumes keep their lineage validation and best-model reconciliation.
    output_dir = Path(config.checkpoint.output_dir)
    best_path = output_dir / "best_model.pth"
    if config.checkpoint.resume_from is None and (
        best_path.exists() or best_path.is_symlink()
    ):
        raise FileExistsError(
            f"Pre-existing checkpoint at {best_path}; choose a fresh output_dir "
            "or set checkpoint.resume_from explicitly."
        )

    # ── Reproducibility ───────────────────────────────────────
    torch.manual_seed(config.training.random_seed)
    np.random.seed(config.training.random_seed)

    # Same sentinel contract as the CLI: only an explicit value overrides
    # the config, so `train(cfg)` cannot silently regress to a stale
    # constant while YAML says 0.2.  0.0 stays a valid opt-out.
    if lambda_reg is None:
        lambda_reg = config.loss.lambda_reg

    resolved_dataset_path = dataset_path or _DATASET_PATH
    logger.info("Dataset: %s", resolved_dataset_path)

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("lambda_reg: %.4f", lambda_reg)
    logger.info("lambda_routing_aux=%.4f", config.loss.lambda_routing_aux)
    if config.training.max_seq_len is not None:
        logger.info("max_seq_len: %d (sequences will be cropped)", config.training.max_seq_len)

    # ── Build components ──────────────────────────────────────
    model = build_model(config).to(device)

    for m in model.modules():
        if isinstance(m, nn.RNNBase):
            m.flatten_parameters()

    # ── Two-phase training setup (Hybrid Funnel) ──────────────
    # phase1_epochs=None → single-phase (backward compatible)
    # phase1_epochs=0    → skip Phase 1, start with Phase 2
    # phase1_epochs=N    → Phase 1 for N epochs, then Phase 2
    two_phase = phase1_epochs is not None
    if two_phase:
        phase2_epochs = config.training.num_epochs - phase1_epochs
        logger.info("=" * 60)
        logger.info("Two-phase training (Hybrid Funnel): Phase 1 = %d epochs, Phase 2 = %d epochs",
                     phase1_epochs, phase2_epochs)
        logger.info("=" * 60)

        # Phase 1: Freeze backend, train frontend with FrontendLoss
        # Phase 2: Freeze frontend, train backend with BioDecisionLoss
        frontend_criterion = FrontendLoss(reduction=config.loss.reduction).to(device)
        backend_criterion = BioDecisionLoss(
            reduction=config.loss.reduction,
            target_rate=config.loss.target_rate,
        ).to(device)

        if phase1_epochs > 0:
            # Start in Phase 1: freeze backend, train frontend
            for param in model.backend.parameters():
                param.requires_grad = False
            for param in model.frontend.parameters():
                param.requires_grad = True
            # Single group carrying ``base_lr`` (mirroring build_optimizer) so
            # apply_lr_warmup (used when lr_warmup_epochs>0) never hits its
            # "param group must carry both base_lr and lr" ValueError.
            optimizer = torch.optim.AdamW(
                [{
                    "params": list(model.frontend.parameters()),
                    "lr": config.training.learning_rate,
                    "base_lr": config.training.learning_rate,
                }],
                weight_decay=config.training.weight_decay,
            )
            criterion = frontend_criterion
            current_phase = 1
        else:
            # phase1_epochs == 0: skip Phase 1, start with Phase 2
            for param in model.frontend.parameters():
                param.requires_grad = False
            for param in model.backend.parameters():
                param.requires_grad = True
            base_lr = config.training.learning_rate
            lif_lr = base_lr * 0.3
            lif_params = list(model.backend.lif_cell.parameters())
            lif_param_ids = {id(p) for p in lif_params}
            other_backend = [p for p in model.backend.parameters() if id(p) not in lif_param_ids]
            optimizer = torch.optim.AdamW([
                {"params": other_backend, "lr": base_lr, "base_lr": base_lr, "name": "non_lif"},
                {"params": lif_params, "lr": lif_lr, "base_lr": lif_lr, "name": "lif"},
            ], weight_decay=config.training.weight_decay)
            criterion = backend_criterion
            current_phase = 2
    else:
        optimizer = build_optimizer(model, config)
        criterion = build_loss(config)
        current_phase = 0  # single-phase mode

    # ── Mixed Precision (AMP) ─────────────────────────────────
    use_amp = False  # Disabled: FP16 causes frequent NaN with new dataset
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_ctx = lambda: torch.amp.autocast(device_type="cuda", enabled=use_amp)
    if use_amp:
        logger.info("AMP enabled (FP16 forward/backward, FP32 master weights)")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.training.num_epochs - config.training.lr_warmup_epochs),
        eta_min=1e-6,
    )
    # LR budget accounting: `apply_lr_warmup` ramps the *base* LR during the
    # warmup window while the cosine is held at its start value (step() is a
    # no-op below while warming up).  The cosine therefore anneals from the
    # full base LR across the post-warmup epochs, matching the documented
    # semantics — the two schedules never race.  (T_max is reduced by the
    # warmup count so the cosine covers exactly the post-warmup horizon.)
    if not two_phase:
        pass  # criterion already set above
    train_loader, val_loader = build_dataloaders(
        config,
        dataset_path=resolved_dataset_path,
        val_split=_VAL_SPLIT,
        use_lazy_loading=use_lazy_loading,
        nested_prior_artifact=nested_prior_artifact,
        trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
    )

    if train_loader is None:
        raise ValueError(
            "Training DataLoader is None.  "
            "Wire build_dataloaders() to the real data pipeline."
        )

    # Reliable epoch-boundary recovery is the cadence-one regime and is scoped
    # to zero-worker loading.  The actual RESOLVED loaders are checked (auto
    # num_workers=-1 may resolve to >0 on a large dataset), and an unsupported
    # configuration fails closed BEFORE any epoch — the guard is inside
    # train() so no producer path can bypass it.  The check is tied to
    # checkpoint_interval==1, so it applies to a fresh reliable run AND to a
    # resume, while the prior interval-10 behavior is unchanged.
    _require_reliable_recovery_loaders(
        train_loader, val_loader,
        checkpoint_interval=config.training.checkpoint_interval,
    )
    _resolved_workers = _resolved_loader_workers(train_loader, val_loader)

    # Ticket #16: stimulus condition metadata for routing auxiliary loss
    train_is_pure_wind = getattr(getattr(train_loader, "dataset", None), "is_pure_wind", None)
    val_is_pure_wind = (
        getattr(getattr(val_loader, "dataset", None), "is_pure_wind", None)
        if val_loader is not None
        else None
    )
    mcmc_prior_consistency = getattr(
        getattr(train_loader, "dataset", None),
        "mcmc_prior_train_serve_consistency",
        None,
    )

    # ── Nested-prior provenance (opt-in seam) ──────────────────
    # Every checkpoint and the returned result dict must record which
    # nested artifact (path + content SHA-256 + source dataset fingerprint)
    # produced the split/priors when the seam is active — without it a
    # checkpoint cannot be audited for split-mismatch after the fact.
    # Legacy (no artifact) runs record is_nested_cv=False explicitly.
    nested_info = getattr(
        getattr(train_loader, "dataset", None), "nested_prior_info", None
    )
    if nested_prior_artifact is not None and nested_info is None:
        raise RuntimeError(
            "nested_prior_artifact was set but the train dataset carries "
            "no nested_prior_info; provenance would be lost. Fail closed."
        )
    lineage = getattr(train_loader.dataset, "prior_lineage", None)
    if lineage is None:
        raise ValueError("Train dataset lacks validated MCMC prior lineage; fail closed")
    prior_tag, prior_status = lineage
    prior_identity_status(prior_tag, prior_status)
    nested_provenance: Dict[str, Any] = {
        "nested_prior_artifact": (
            str(nested_info["nested_prior_artifact"]) if nested_info else ""
        ),
        "nested_prior_artifact_sha256": (
            str(nested_info["nested_prior_artifact_sha256"]) if nested_info else ""
        ),
        "nested_prior_fingerprint": (
            str(nested_info["nested_prior_fingerprint"]) if nested_info else ""
        ),
        "nested_split_seed": int(nested_info["split_seed"]) if nested_info else -1,
        "nested_val_split": float(nested_info["val_split"]) if nested_info else -1.0,
        "is_nested_cv": bool(nested_info is not None),
        "validation_scope": ("nested_outer_validation" if nested_info else "diagnostic_global_oof"),
        "mcmc_prior_provenance": (
            str(nested_info["mcmc_prior_provenance"])
            if nested_info
            else prior_tag
        ),
        "animal_identity_status": (
            prior_identity_status(
                nested_info["mcmc_prior_provenance"],
                nested_info.get("animal_identity_status"), nested=True,
            ) if nested_info else prior_status
        ),
    }
    if nested_info is None:
        logger.warning(
            "validation_scope=diagnostic_global_oof: global OOF prior fits can "
            "include eventual outer-validation labels. Scores are contaminated "
            "diagnostics, ineligible for strict QC/release gates; use a nested artifact."
        )
    # Phase-2 script-owned controls, stamped additively into every checkpoint
    # (they are NOT part of ``config.to_dict()``, which stays byte-unchanged).
    # A resume refuses a switch of ``selection_metric`` (see
    # ``_require_architecture_config_match``); ``zero_input_channels`` names
    # the history ablation a checkpoint was trained under.
    phase2_provenance: Dict[str, Any] = {
        "selection_metric": _SELECTION_METRIC,
        "zero_input_channels": list(_ZERO_INPUT_CHANNELS),
    }
    active_lineage: Dict[str, Any] = dict(nested_provenance)
    active_lineage["dataset_path"] = str(resolved_dataset_path)
    dataset_source_sha256 = getattr(train_loader.dataset, "dataset_source_sha256", None)
    if not use_lazy_loading and dataset_source_sha256 is None:
        raise ValueError("Loaded train dataset lacks its source SHA-256")
    active_lineage["dataset_source_sha256"] = dataset_source_sha256

    def validate_checkpoint_lineage(state, candidate_path, payload):
        validate_checkpoint_clock(state, model.dt_ms, require_dt_ms=True)
        _check_checkpoint_prior_lineage(
            state, active_lineage, candidate_path,
            trusted_historical_checkpoint_sha256=trusted_historical_checkpoint_sha256,
            checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        )

    generated_checkpoint_shas: Dict[Path, str] = {}

    def evaluation_checkpoint_pins(
        candidate_path: Path, payload: bytes,
    ) -> Any:
        """Trust own writes only while their captured bytes remain unchanged."""
        digest = hashlib.sha256(payload).hexdigest()
        if generated_checkpoint_shas.get(candidate_path) == digest:
            return digest
        return trusted_historical_checkpoint_sha256

    def certify_best_checkpoint(
        state: Dict[str, Any], candidate_path: Path, payload: bytes,
        *, evaluation: bool = False,
    ) -> float:
        validate_checkpoint_clock(state, model.dt_ms, require_dt_ms=True)
        return _validate_best_checkpoint(
            state, candidate_path, active_lineage,
            trusted_historical_checkpoint_sha256=(
                evaluation_checkpoint_pins(candidate_path, payload)
                if evaluation else trusted_historical_checkpoint_sha256
            ),
            checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def certify_final_checkpoint(
        state: Dict[str, Any], candidate_path: Path, payload: bytes,
    ) -> None:
        validate_checkpoint_clock(state, model.dt_ms, require_dt_ms=True)
        _validate_best_checkpoint(
            state, candidate_path, active_lineage,
            trusted_historical_checkpoint_sha256=evaluation_checkpoint_pins(
                candidate_path, payload,
            ),
            checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
            require_best=False,
        )
    dataset_source_binding = "sha256_bound" if dataset_source_sha256 is not None else "unbound_lazy"
    logger.info(
        "Checkpoint provenance: is_nested_cv=%s artifact=%s fingerprint=%s",
        nested_provenance["is_nested_cv"],
        nested_provenance["nested_prior_artifact"] or "<none>",
        (nested_provenance["nested_prior_fingerprint"][:12] or "<none>"),
    )

    # ── Target normalization statistics (train split only) ──
    # When config.training.normalize_targets is enabled, the velocity
    # target is mean-centered and std-scaled using training-split
    # statistics.  These same values are threaded through train_one_epoch,
    # validate, and compute_metrics so the loss, the selection-criteria,
    # and the reported metrics all live in one consistent space, and the
    # final metrics are rescaled back to cm/s.
    if config.training.normalize_targets and not use_lazy_loading:
        # The loaded train dataset owns the selected split
        # and raw target copies. Reopening the source path could select a
        # different valid object after the loader captured its source bytes.
        train_dataset = train_loader.dataset
        if not (
            hasattr(train_dataset, "sequences")
            and hasattr(train_dataset, "source_indices")
        ):
            raise ValueError(
                "Loaded train dataset lacks source targets/indices for normalization"
            )
        train_targets = [sequence[1] for sequence in train_dataset.sequences]
        train_indices = np.asarray(train_dataset.source_indices, dtype=np.int64)
        if not train_targets or train_indices.shape != (len(train_targets),):
            raise ValueError("Loaded train targets/indices are empty or misaligned")
        assert_finite_targets(train_targets)
        target_mean, target_std, _train_indices = _fit_target_stats(
            np.concatenate(train_targets).astype(np.float64), config, train_indices,
        )
    else:
        target_mean, target_std, _train_indices = compute_target_stats(
            resolved_dataset_path,
            config,
            val_split=_VAL_SPLIT,
            nested_prior_artifact=nested_prior_artifact,
            trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
        )

    # Statistical coherence guard: normalizing without a target clip amplifies
    # the very heavy-tail (tracking-artifact) frames it is meant to suppress —
    # the loss standardises every frame by the small bulk std, so a ~1e7 cm/s
    # outlier becomes O(1e5-1e6) sigma.  Recommend enabling clip together.
    if config.training.normalize_targets and config.training.target_clip_cm_s <= 0.0:
        logger.warning(
            "normalize_targets=True but target_clip_cm_s=%s (disabled).  "
            "Without a non-zero clip, the raw tracking-artifact outliers "
            "dominate the standardized MSE and can destabilize training.  "
            "Strongly recommend enabling both together for a coherent target.",
            config.training.target_clip_cm_s,
        )

    # ── Apply freezing strategy ───────────────────────────────
    if config.finetune.freeze_modules:
        logger.info(
            "Freezing modules: %s", config.finetune.freeze_modules,
        )
        model.freeze_modules(config.finetune.freeze_modules)

    # ── Phase-2 optimizer/scheduler builder (single source of truth) ──
    # Both the in-loop phase-1→2 transition and the mid-phase-2 resume path
    # construct the SAME 2-group backend optimizer.  Keeping this in one
    # helper guarantees a resumed phase-2 run rebuilds an optimizer that
    # exactly matches the shape the checkpoint's optimizer_state_dict was
    # saved from, so load_state_dict can restore the Adam moments.
    def _build_phase2_optimizer_scheduler(model, config, at_epoch):
        base_lr = config.training.learning_rate
        lif_lr = base_lr * 0.3
        lif_params = list(model.backend.lif_cell.parameters())
        lif_param_ids = {id(p) for p in lif_params}
        other_backend = [p for p in model.backend.parameters() if id(p) not in lif_param_ids]
        optimizer = torch.optim.AdamW([
            {"params": other_backend, "lr": base_lr, "base_lr": base_lr, "name": "non_lif"},
            {"params": lif_params, "lr": lif_lr, "base_lr": lif_lr, "name": "lif"},
        ], weight_decay=config.training.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                1,
                (config.training.num_epochs - at_epoch) - config.training.lr_warmup_epochs,
            ),
            eta_min=1e-6,
        )
        return optimizer, scheduler

    # ── Resume from checkpoint ────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    # Set when a validated current-schema parent lands EXACTLY on the original
    # total budget (``start_epoch == num_epochs``): the resumed run has zero
    # updates left, so it finalizes at the parent's saved executed epoch.
    # See the equality branch below.
    _terminal_at_total_budget = False

    if config.checkpoint.resume_from is not None:
        ckpt_path = Path(config.checkpoint.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file to resume from not found: {ckpt_path} "
                "(fail closed on missing resume checkpoint)"
            )
        logger.info("Resuming from checkpoint: %s", ckpt_path)
        # Peek BEFORE load_checkpoint: (a) detect legacy checkpoints
        # without scheduler state (loud warning instead of silent LR
        # restart), and (b) learn start_epoch so the restore can match
        # the phase the run will continue in.
        resume_payload = ckpt_path.read_bytes()
        ckpt_peek = load_artifact_bytes(resume_payload, map_location="cpu")
        validate_checkpoint_clock(ckpt_peek, model.dt_ms, require_dt_ms=True)
        # A current-schema parent claims an exact continuation, so its full
        # canonical state must be present before anything restores from it.
        # This runs FIRST in the resume preflight (before any epoch or any
        # terminal save) and only constrains the current schema; a legacy /
        # version-mismatched parent keeps its limited fallback.
        _require_complete_current_schema(ckpt_peek, ckpt_path)
        # R3: refuse a resume that switches an architecture-defining setting
        # (refinement / activation / lambda_compute) relative to the checkpoint.
        _require_architecture_config_match(ckpt_peek, config, ckpt_path)

        # ── Fail-closed nested resume validation ──
        # Checkpoint provenance vs active run configuration:
        # 1. is_nested_cv mode mismatch
        # 2. artifact identity / content SHA-256 / source fingerprint mismatch
        # 3. exact split configuration (split_seed, val_split) mismatch
        ckpt_is_nested = ckpt_peek.get("is_nested_cv", False)
        run_is_nested = bool(nested_prior_artifact is not None)

        if run_is_nested != ckpt_is_nested:
            raise ValueError(
                f"Resume nested mode mismatch: active run is_nested_cv={run_is_nested} "
                f"but checkpoint {ckpt_path} has is_nested_cv={ckpt_is_nested}. "
                "Refusing cross-mode resumption (fail closed)."
            )

        if run_is_nested:
            ckpt_art = ckpt_peek.get("nested_prior_artifact")
            run_art = str(nested_info["nested_prior_artifact"])
            # Resolve both paths for robust comparison
            if not ckpt_art or Path(ckpt_art).resolve() != Path(run_art).resolve():
                raise ValueError(
                    f"Resume nested prior artifact mismatch: checkpoint recorded artifact "
                    f"{ckpt_art!r} vs active run artifact {run_art!r}. "
                    "Refusing resume with mismatched nested prior artifact (fail closed)."
                )

            ckpt_fp = ckpt_peek.get("nested_prior_fingerprint")
            run_fp = str(nested_info["nested_prior_fingerprint"])
            if not ckpt_fp or ckpt_fp != run_fp:
                raise ValueError(
                    f"Resume nested prior source fingerprint mismatch: checkpoint recorded "
                    f"{ckpt_fp!r} vs active run {run_fp!r}. "
                    "Refusing resume on mismatched source dataset (fail closed)."
                )

            ckpt_artifact_sha256 = ckpt_peek.get("nested_prior_artifact_sha256")
            run_artifact_sha256 = nested_provenance["nested_prior_artifact_sha256"]
            if ckpt_artifact_sha256 != run_artifact_sha256:
                raise ValueError(
                    "Resume nested prior artifact SHA-256 mismatch: "
                    f"{ckpt_artifact_sha256!r} vs {run_artifact_sha256!r}; fail closed."
                )

            ckpt_seed = ckpt_peek.get("nested_split_seed")
            run_seed = int(nested_info["split_seed"])
            if ckpt_seed is None:
                raise ValueError("Resume checkpoint missing nested_split_seed (fail closed).")
            if isinstance(ckpt_seed, float) and not ckpt_seed.is_integer():
                raise ValueError(
                    f"Resume nested split_seed is fractional ({ckpt_seed}); refusing resume (fail closed)."
                )
            if int(ckpt_seed) != run_seed:
                raise ValueError(
                    f"Resume nested split_seed mismatch: checkpoint recorded seed={ckpt_seed} "
                    f"vs active run seed={run_seed}. Refusing resume on altered split (fail closed)."
                )

            ckpt_val_split = ckpt_peek.get("nested_val_split")
            run_val_split = float(nested_info["val_split"])
            if (
                ckpt_val_split is None
                or not math.isfinite(float(ckpt_val_split))
                or not math.isfinite(run_val_split)
                or abs(float(ckpt_val_split) - run_val_split) > 1e-6
            ):
                raise ValueError(
                    f"Resume nested val_split mismatch or non-finite: checkpoint recorded val_split={ckpt_val_split} "
                    f"vs active run val_split={run_val_split}. Refusing resume on altered split (fail closed)."
                )
        else:
            if prior_status == "unverified" and (
                ckpt_peek.get("dataset_source_sha256") != dataset_source_sha256
                or ckpt_peek.get("dataset_source_binding", "sha256_bound") != "sha256_bound"
            ):
                raise ValueError("modern checkpoint requires matching dataset_source_sha256 and sha256_bound binding; fail closed")
            if "dataset_source_sha256" in ckpt_peek:
                if ckpt_peek["dataset_source_sha256"] != dataset_source_sha256:
                    raise ValueError("Resume non-nested dataset_source_sha256 mismatch; fail closed.")
                dataset_source_binding = ckpt_peek.get("dataset_source_binding", "sha256_bound")
                if dataset_source_binding not in ("sha256_bound", "legacy_resume_unbound"):
                    raise ValueError("Resume non-nested invalid dataset_source_binding; fail closed.")
            else:
                dataset_source_binding = ckpt_peek.get("dataset_source_binding", "legacy_resume_unbound")
                if dataset_source_binding not in ("legacy_resume_unbound", "unbound_lazy"):
                    raise ValueError("Resume checkpoint missing dataset_source_sha256 with invalid binding status")
                logger.warning("Resume checkpoint %s has %s dataset content; historical weights cannot be content-bound",
                               ckpt_path, dataset_source_binding)
            active_lineage["dataset_source_binding"] = dataset_source_binding
            ckpt_ds = ckpt_peek.get("dataset_path")
            if ckpt_ds is not None and resolved_dataset_path is not None:
                if Path(ckpt_ds).resolve() != Path(resolved_dataset_path).resolve():
                    raise ValueError(
                        f"Resume non-nested dataset_path mismatch: checkpoint recorded dataset_path={ckpt_ds!r} "
                        f"vs active run dataset_path={str(resolved_dataset_path)!r}. "
                        "Refusing resume on mismatched source dataset (fail closed)."
                    )

        _check_checkpoint_prior_lineage(
            ckpt_peek, active_lineage, ckpt_path,
            trusted_historical_checkpoint_sha256=trusted_historical_checkpoint_sha256,
            checkpoint_sha256=hashlib.sha256(resume_payload).hexdigest(),
        )
        if "scheduler_state_dict" not in ckpt_peek:
            logger.warning(
                "Checkpoint %s has NO scheduler_state_dict (legacy "
                "format) — LR schedule restarts from scratch; resumed "
                "runs will NOT match an uninterrupted trajectory.",
                ckpt_path,
            )
        start_epoch = ckpt_peek.get("epoch", -1) + 1
        if start_epoch > config.training.num_epochs:
            # The parent overshot the run's own total budget.  This is NOT
            # budget exhaustion: the active config would shorten a longer
            # parent run, and its weights come from epochs this budget never
            # authorized, so it fails closed (unchanged).
            raise ValueError(
                f"Cannot resume checkpoint {ckpt_path} at epoch {start_epoch}: "
                f"start_epoch ({start_epoch}) exceeds target num_epochs "
                f"({config.training.num_epochs}); no remaining epochs to train "
                "(fail closed on validation provenance)."
            )
        if start_epoch == config.training.num_epochs:
            # ── Completed parent AT the original total budget ─────
            # ``start_epoch == num_epochs`` is a COMPLETED run of exactly this
            # budget, not an error: the loop has no remaining updates
            # (``range(num_epochs, num_epochs)`` is empty) and the parent
            # checkpoint is the run's own terminal epoch.  This is the
            # publication-crash window — the periodic checkpoint is written
            # BEFORE the stop check, so a process killed in that window leaves
            # a parent equal to the uninterrupted final state.  Refusing it
            # loses a completed run that needs no further work.  Equality with
            # the active budget is checked against the PARENT's own recorded
            # total budget below, because the budget is immutable across a
            # continuation and a longer parent run can land on the same index.
            #
            # Accepting it is deliberately NARROW.  Only the CURRENT recovery
            # schema is eligible: a legacy / version-mismatch parent has an
            # UNKNOWN counter and history (the reader resets them with a
            # warning), so it cannot be validated as a completed run and still
            # fails closed.  The counter/history/axis consistency, the
            # patience bound and the required MT19937 state are all validated
            # by ``_restore_recovery_state`` below, which is the FIRST thing
            # the terminal path runs — a corrupt current-schema or inconsistent
            # lineage parent raises there, before any finalization.
            if ckpt_peek.get("recovery_state_version") != RECOVERY_STATE_VERSION:
                raise ValueError(
                    f"Cannot resume checkpoint {ckpt_path} at epoch "
                    f"{start_epoch} == target num_epochs "
                    f"({config.training.num_epochs}) with recovery_state_version="
                    f"{ckpt_peek.get('recovery_state_version')!r}: a legacy or "
                    "version-mismatched parent has an UNKNOWN patience counter "
                    "and history, so a completed run cannot be validated; "
                    "resume it under its own original budget (fail closed)."
                )
            # Equality with the ACTIVE budget is not sufficient: the total
            # budget is IMMUTABLE across a continuation, so the parent must be
            # shown to have completed under THIS budget.  A parent run under a
            # longer budget can coincidentally land on the active budget's
            # final epoch (parent 6 epochs, active 3, parent checkpoint at
            # index 2); finalizing that here would stamp the segment with a
            # total budget the parent never ran under and truncate a longer
            # authorized run.  The parent's recorded
            # ``config.training.num_epochs`` is therefore required to be a
            # KNOWN strict positive integer equal to the active budget;
            # unknown, missing, malformed (bool / non-integer) or unequal
            # values fail closed BEFORE any finalization.
            _parent_training = _parent_training_controls(ckpt_peek, ckpt_path)
            _parent_num_epochs = (
                _parent_training.get("num_epochs")
                if _parent_training is not None else None
            )
            _parent_budget_known = (
                not isinstance(_parent_num_epochs, bool)
                and isinstance(_parent_num_epochs, (int, np.integer))
                and int(_parent_num_epochs) > 0
            )
            if not _parent_budget_known or int(_parent_num_epochs) != int(
                config.training.num_epochs
            ):
                raise ValueError(
                    f"Cannot resume checkpoint {ckpt_path} at epoch "
                    f"{start_epoch} == target num_epochs "
                    f"({config.training.num_epochs}): the parent's original "
                    f"total budget is {_parent_num_epochs!r}, which must be a "
                    "known strict positive integer equal to this run's "
                    "num_epochs. A completed run is finalizable only under its "
                    "OWN original total budget (fail closed)."
                )
            _terminal_at_total_budget = True
        if two_phase and ckpt_peek.get("training_phase") != (1 if start_epoch <= phase1_epochs else 2):
            raise ValueError(
                f"Resume checkpoint {ckpt_path} has missing or inconsistent training_phase "
                f"({ckpt_peek.get('training_phase')!r}) for start_epoch={start_epoch}; fail closed."
            )

        if ckpt_path.name == "best_model.pth":
            certify_best_checkpoint(ckpt_peek, ckpt_path, resume_payload)

        # Resume x two-phase: when the resume point is at/past the
        # phase-1→2 boundary, an uninterrupted run would have built a
        # FRESH phase-2 optimizer at the boundary (the phase-1 Adam
        # moments are deliberately discarded there).  Mirror that
        # exactly: restore ONLY model weights here and let the
        # in-loop transition construct the fresh phase-2 optimizer on
        # the first epoch.  Restoring into (or pre-building) the
        # phase-2 optimizer either crashes (1-group checkpoint vs
        # 2-group optimizer) or silently diverges from the canonical
        # uninterrupted trajectory.
        # Resume phase is decided by start_epoch, NOT the init-time
        # current_phase (which is 1 whenever phase1_epochs>0 and takes no
        # account of where the checkpoint actually is):
        #   * start_epoch <  phase1_epochs: within phase 1 → restore the
        #     full phase-1 optimizer/scheduler state (moment continuity).
        #   * start_epoch == phase1_epochs: landing exactly at the
        #     phase-1→2 boundary → restore ONLY model weights and let the
        #     in-loop transition build the fresh phase-2 optimizer
        #     (an uninterrupted run discards the phase-1 Adam moments
        #     there, so the resumed run must too).
        #   * start_epoch >  phase1_epochs: resuming inside an ESTABLISHED
        #     phase 2 → the checkpoint holds the 2-group phase-2 optimizer
        #     and scheduler.  Rebuild that exact optimizer (via the shared
        #     builder) BEFORE restoring so load_state_dict can rehydrate
        #     the saved Adam moments and scheduling instead of silently
        #     resetting them via a second in-loop transition.
        # A phase-1→2 boundary that IS the total budget (``phase1_epochs ==
        # num_epochs``) is not a crossing for a completed run: the phase-2
        # transition would have happened at the FIRST epoch of phase 2, which
        # this run never executes.  Treating it as a crossing would discard the
        # phase-1 best checkpoint and leave a budget-terminal resume with no
        # best model.  A ``phase1_epochs`` beyond the budget is a configuration
        # error and still fails closed at the two-phase validation below.
        _crosses_boundary = (
            two_phase
            and start_epoch == phase1_epochs
            and not _terminal_at_total_budget
        )
        _landing_phase2 = two_phase and start_epoch > phase1_epochs

        # Capture and validate every best candidate before any state restoration.
        # Reconciliation below uses these same bytes; never reread a mutable path.
        resume_phase = 2 if _landing_phase2 else current_phase
        best_target = output_dir / "best_model.pth"
        claimed_best: Optional[float] = None
        if "best_val_loss" in ckpt_peek and ckpt_peek["best_val_loss"] is not None and math.isfinite(float(ckpt_peek["best_val_loss"])):
            claimed_best = float(ckpt_peek["best_val_loss"])
        elif "val_loss" in ckpt_peek and ckpt_peek["val_loss"] is not None and math.isfinite(float(ckpt_peek["val_loss"])):
            claimed_best = float(ckpt_peek["val_loss"])

        src_cand_path: Optional[Path] = None
        src_cand_payload: Optional[bytes] = None
        src_cand_val: Optional[float] = None
        companion = ckpt_path.parent / "best_model.pth"
        companion_payload: Optional[bytes] = None
        comp_peek: Optional[Dict[str, Any]] = None
        if ckpt_path.name == "best_model.pth":
            src_cand_path = ckpt_path
            src_cand_payload = resume_payload
            if not _crosses_boundary:
                src_cand_val = certify_best_checkpoint(
                    ckpt_peek, ckpt_path, resume_payload,
                )
        elif companion.exists() and companion.resolve() != ckpt_path.resolve():
            companion_payload = companion.read_bytes()
            comp_peek = load_artifact_bytes(companion_payload, map_location="cpu")
            validate_checkpoint_clock(comp_peek, model.dt_ms, require_dt_ms=True)
            if two_phase and comp_peek.get("training_phase") not in (1, 2):
                raise ValueError(f"Best checkpoint {companion} lacks training_phase; fail closed.")
            if not _crosses_boundary and (
                not two_phase or comp_peek["training_phase"] == resume_phase
            ):
                src_cand_path = companion
                src_cand_payload = companion_payload
                src_cand_val = certify_best_checkpoint(
                    comp_peek, companion, companion_payload,
                )
            elif not _crosses_boundary and claimed_best is not None and ckpt_peek.get("val_loss") == claimed_best:
                src_cand_path = ckpt_path
                src_cand_payload = resume_payload
                src_cand_val = certify_best_checkpoint(
                    ckpt_peek, ckpt_path, resume_payload,
                )

        dest_val: Optional[float] = None
        dest_stale = False
        destination_payload: Optional[bytes] = None
        if best_target.exists():
            if best_target.resolve() == ckpt_path.resolve():
                destination_payload = resume_payload
                dest_peek = ckpt_peek
            elif best_target.resolve() == companion.resolve():
                destination_payload = companion_payload
                dest_peek = comp_peek
            else:
                destination_payload = best_target.read_bytes()
                dest_peek = load_artifact_bytes(destination_payload, map_location="cpu")
            validate_checkpoint_clock(dest_peek, model.dt_ms, require_dt_ms=True)
            if two_phase and dest_peek.get("training_phase") not in (1, 2):
                raise ValueError(f"Best checkpoint {best_target} lacks training_phase; fail closed.")
            if two_phase and dest_peek["training_phase"] != resume_phase:
                dest_stale = True
            elif not _crosses_boundary:
                dest_val = certify_best_checkpoint(
                    dest_peek, best_target, destination_payload,
                )

        if src_cand_val is not None:
            assert src_cand_payload is not None

        if _landing_phase2:
            # Restore the phase-2 freeze state (requires_grad is NOT carried
            # by the checkpoint): init leaves frontend unfrozen/backend frozen
            # (the phase-1 posture), but a mid-phase-2 checkpoint implies the
            # opposite.  Without this the rebuilt backend optimizer would hold
            # frozen params (dead groups) while the unfrozen frontend is not
            # covered by any optimizer — silently training nothing.
            for param in model.frontend.parameters():
                param.requires_grad = False
            for param in model.backend.parameters():
                param.requires_grad = True
            optimizer, scheduler = _build_phase2_optimizer_scheduler(
                model, config, start_epoch,
            )
            criterion = backend_criterion
            current_phase = 2
            load_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
                payload=resume_payload,
            )
        elif _crosses_boundary:
            logger.info(
                "Resume past phase boundary (start_epoch=%d >= "
                "phase1_epochs=%d) — restoring model weights only; "
                "fresh phase-2 optimizer built by the in-loop "
                "transition",
                start_epoch, phase1_epochs,
            )
            load_checkpoint(
                path=ckpt_path,
                model=model,
                map_location=device,
                payload=resume_payload,
            )
        else:
            load_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
                payload=resume_payload,
            )
        # Restore best_val_loss faithfully across resume:
        # Never trust filename or fall back to train_loss (ckpt['loss'])!
        if "best_val_loss" in ckpt_peek and ckpt_peek["best_val_loss"] is not None and math.isfinite(ckpt_peek["best_val_loss"]):
            best_val_loss = float(ckpt_peek["best_val_loss"])
        elif "val_loss" in ckpt_peek and ckpt_peek["val_loss"] is not None and math.isfinite(ckpt_peek["val_loss"]):
            best_val_loss = float(ckpt_peek["val_loss"])
        else:
            best_val_loss = float("inf")
            logger.warning(
                "Checkpoint %s lacks finite validation loss metadata (never falling back to train loss); "
                "best_val_loss initialized to inf.",
                ckpt_path,
            )
        logger.info("Resumed at epoch %d, best_val_loss=%.6f", start_epoch, best_val_loss)

    # ── Output directory ──────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase-1 best is discarded at the boundary; its loss cannot rank Phase-2 weights.
    if config.checkpoint.resume_from is not None and not _crosses_boundary:
        # Symmetric reconciliation uses only the preflighted snapshots.
        if dest_val is not None and src_cand_val is not None:
            if claimed_best is not None and min(dest_val, src_cand_val) > claimed_best + 1e-5:
                raise ValueError(
                    f"Claimed historical best_val_loss={claimed_best:.6f} has no matching "
                    "same-phase best weights; fail closed."
                )
            # Both source and destination exist: select the superior candidate
            if src_cand_val < dest_val:
                best_target.write_bytes(src_cand_payload)
                best_val_loss = src_cand_val
                logger.info(
                    "Replaced worse destination best model (val_loss=%.6f) with superior source "
                    "best model from %s (val_loss=%.6f)",
                    dest_val, src_cand_path, best_val_loss,
                )
            else:
                assert destination_payload is not None
                best_target.write_bytes(destination_payload)
                best_val_loss = dest_val
                logger.info(
                    "Retained existing destination best model at %s (val_loss=%.6f <= source %.6f)",
                    best_target, dest_val, src_cand_val,
                )
        elif dest_val is not None:
            # Only destination exists: refuse silent degradation if claimed historical best is better
            if claimed_best is not None and dest_val > claimed_best + 1e-5:
                raise ValueError(
                    f"Claimed historical best_val_loss={claimed_best:.6f} is unavailable: "
                    f"destination {best_target} has worse val_loss={dest_val:.6f} and companion "
                    f"source best model is absent; refusing to silently degrade best score (fail closed)."
                )
            assert destination_payload is not None
            best_target.write_bytes(destination_payload)
            best_val_loss = dest_val
            logger.info(
                "Validated existing destination best_model.pth at %s (val_loss=%.6f)",
                best_target, dest_val,
            )
        elif src_cand_val is not None:
            # Only source exists: copy certified candidate to destination
            if claimed_best is not None and src_cand_val > claimed_best + 1e-5:
                raise ValueError(
                    f"Source best model {src_cand_path} val_loss ({src_cand_val:.6f}) is worse than "
                    f"claimed best_val_loss ({claimed_best:.6f}); fail closed."
                )
            best_target.write_bytes(src_cand_payload)
            best_val_loss = src_cand_val
            logger.info(
                "Preserved certified resume best model from %s to %s (val_loss=%.6f)",
                src_cand_path, best_target, best_val_loss,
            )
        else:
            # Neither source candidate nor destination candidate exists
            if claimed_best is not None:
                raise ValueError(
                    f"Cannot resume checkpoint {ckpt_path} into fresh directory {output_dir}: "
                    f"absent historical best model {ckpt_path.parent / 'best_model.pth'} (fail closed)."
                )
            if dest_stale:
                best_target.unlink()
            best_val_loss = float("inf")
            logger.warning(
                "Checkpoint %s lacks historical best model; best_val_loss initialized to inf.",
                ckpt_path,
            )

    # ── Training loop ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting training for %d epochs", config.training.num_epochs)
    logger.info("=" * 60)

    # Diagnostic series are positionally aligned on the epoch axis.  Values
    # are finite OR non-finite scalars (a NaN/Inf epoch is a real result);
    # ``None`` marks an epoch at which the series recorded nothing (e.g. no
    # validation), so the axis stays intact.
    history: Dict[str, List[Any]] = {"train_loss": [], "val_loss": []}
    # Masked-MSE-only validation diagnostic on the ALIGNED eligible frames
    # (t >= 1), recorded alongside the selected ``val_loss`` but NOT persisted
    # through the recovery payload (whose ``training_history`` schema is frozen
    # to {train_loss, val_loss}).
    val_mse_history: List[Optional[float]] = []
    train_loss: float = float("nan")
    val_loss: float = float("nan")

    # ── Early stopping ───────────────────────────────────────
    early_stopping_patience = getattr(
        config.training, "early_stopping_patience", 0
    )
    epochs_without_improvement = 0

    # ── Epoch-boundary recovery (resume) ─────────────────────
    # The counter and history above must CONTINUE across a restart, not
    # restart.  Restore them from the resume checkpoint's recovery seam.
    # This runs AFTER the resume block so a corrupt payload fails closed
    # before any training, and BEFORE the loop so the first resumed epoch
    # already sees the accumulated patience and history.  The NumPy stream
    # is restored here too — it is read only inside the (anchor-aligned,
    # epoch-independent) data crop, so its exact restore point does not
    # perturb the model stream.
    #
    # A phase-1→2 boundary-crossing resume keeps HISTORY (the loss curve is
    # a property of the whole run, and the uninterrupted trajectory appends
    # across the boundary) but resets ONLY the early-stop counter — exactly
    # as the in-loop transition does.  See the boundary reset after this
    # block.
    # ``history_start_epoch`` is the true 0-based epoch of ``history`` entry
    # 0.  Fresh runs start at 0; a legacy resume starts at its resume epoch
    # because the parent's earlier history is unknown (never redrawn as 1).
    history_start_epoch = 0
    # Set when the restored state is ALREADY terminal (patience exhausted at
    # the resumed epoch).  Consumed after the loop to finalize without an
    # epoch.  See the terminal decision below.
    _terminal_on_resume = False
    if config.checkpoint.resume_from is not None:
        _ewi, _history, _np_state, history_start_epoch = _restore_recovery_state(
            ckpt_peek, ckpt_path, resume_epoch=start_epoch,
        )
        epochs_without_improvement = 0 if _crosses_boundary else _ewi
        history = _history
        if _np_state is not None:
            np.random.set_state(_np_state)
        # ── Terminal decision BEFORE any resumed epoch ────────────
        # The checkpoint is published at a completed-epoch boundary, BEFORE
        # the early-stop check for that epoch.  A process that dies in that
        # window leaves a checkpoint whose restored counter has already
        # reached the patience horizon: the uninterrupted run would have
        # stopped there and never executed another epoch.  The resumed run
        # must therefore apply the SAME stop predicate before training, or it
        # executes epochs the uninterrupted run never did (the resumed
        # trajectory diverges and the counter/history grow past the terminal
        # state).  A later improving validation cannot restart an exhausted
        # horizon — that would re-open a decision the original run had
        # already closed.
        #
        # The predicate is shared with the in-loop check (same exemption and
        # same phase reset), so a phase-1 exemption, a phase-boundary
        # counter reset and a phase-2 exhaustion all behave exactly as they
        # do at an epoch boundary inside the loop.  A boundary-crossing
        # resume resets the counter above, so it is never terminal here.
        _patience_terminal = _recovery_terminal(
            early_stopping_patience,
            epochs_without_improvement,
            two_phase=two_phase,
            current_phase=current_phase,
            num_epochs=config.training.num_epochs,
            phase1_epochs=phase1_epochs,
        )
        if _patience_terminal:
            logger.info(
                "Resumed checkpoint %s already exhausted early stopping "
                "(epochs_without_improvement=%d >= patience=%d at completed "
                "epoch %d); finalizing without executing a further epoch.",
                ckpt_path, epochs_without_improvement,
                early_stopping_patience, start_epoch,
            )
        if _terminal_at_total_budget:
            # The parent completed the run's own total budget: the loop below
            # would iterate over ``range(num_epochs, num_epochs)`` — an EMPTY
            # range — so there are no updates left to perform whatever the
            # patience counter says (it may be disabled, or simply never
            # exhausted).  Terminal on the BUDGET alone, not on patience.
            logger.info(
                "Resumed checkpoint %s completed the original total budget "
                "(epoch %d of %d); finalizing at the saved executed epoch "
                "without executing a further epoch "
                "(epochs_without_improvement=%d, patience=%d).",
                ckpt_path, start_epoch, config.training.num_epochs,
                epochs_without_improvement, early_stopping_patience,
            )
        if _patience_terminal or _terminal_at_total_budget:
            _terminal_on_resume = True
            # Finalization below reports the LAST EXECUTED epoch's diagnostics.
            # The loop never runs, so nothing would overwrite the ``nan``
            # placeholders (which would claim a loss the run never measured).
            # Take the parent's recorded value from EITHER source — its own
            # top-level scalar or the restored history tail — preferring a
            # recorded scalar and treating a ``None`` gap as fillable, not as
            # a contradiction.  A recorded parent scalar is NEVER discarded in
            # favour of a gap.  ``train_loss`` keeps its ``nan`` placeholder
            # only when NEITHER source has a value (unobserved); ``val_loss``
            # uses ``inf`` for that case so the final checkpoint serializes an
            # explicit ``None`` rather than a fabricated finite value.  The
            # train diagnostic also honours the independent legacy ``loss``
            # field when ``train_loss`` is ABSENT (see
            # ``_terminal_train_diagnostic``); an explicit ``train_loss=None``
            # is NOT substituted by it.
            _term_train = _terminal_train_diagnostic(
                ckpt_peek,
                history["train_loss"][-1] if history["train_loss"] else None,
                ckpt_path=ckpt_path,
            )
            train_loss = _term_train if _term_train is not None else None
            _term_val = _terminal_diagnostic_value(
                ckpt_peek.get("val_loss"),
                history["val_loss"][-1] if history["val_loss"] else None,
                series="val_loss", ckpt_path=ckpt_path,
            )
            val_loss = _term_val if _term_val is not None else float("inf")

    # ── Restart-segment lineage record ────────────────────────
    # One immutable record per invocation, written once training is
    # committed to (all preflight validation has passed) and before the
    # loop, so the segment survives a later SIGKILL/OOM.  A resume records
    # its parent checkpoint's EXACT resumed bytes (the same ``resume_payload``
    # the run loaded and lineage-checked — never a re-read of the mutable
    # path); a fresh run records parent=None.  Prior segments are never
    # touched.
    _segment_parent = (
        _checkpoint_lineage_identity(resume_payload, ckpt_path)
        if config.checkpoint.resume_from is not None else None
    )
    _write_segment_record(
        output_dir,
        parent=_segment_parent,
        config=config,
        parent_peek=ckpt_peek if config.checkpoint.resume_from is not None else None,
        start_epoch=start_epoch,
        resumed=config.checkpoint.resume_from is not None,
        resume_path=ckpt_path if config.checkpoint.resume_from is not None else None,
        resolved_workers=_resolved_workers,
    )

    # ── Bio-loss warmup schedule ─────────────────────────────
    warmup_epochs = config.loss.warmup_epochs

    epoch = start_epoch - 1
    for epoch in range(start_epoch, config.training.num_epochs):
        if _terminal_on_resume:
            # The restored counter was already terminal at the resumed epoch:
            # the original run stopped there, so no epoch may execute.  The
            # decision is re-checked against the phase state the transition
            # would have produced, so a phase-1 exemption / boundary reset
            # behaves as it does in the loop.
            #
            # ``for`` binds the loop variable BEFORE the body runs, so without
            # this restore the finalization below would report ``start_epoch``
            # — an epoch this process never executed.  The last EXECUTED epoch
            # is the parent's (``start_epoch - 1``), which is what the
            # uninterrupted run's terminal checkpoint records.
            epoch = start_epoch - 1
            break
        t0 = time.time()

        # ── Phase transition (Hybrid Funnel) ──────────────────
        if two_phase and current_phase == 1 and epoch >= phase1_epochs:
            logger.info("=" * 60)
            logger.info("Phase 1 → Phase 2 transition at epoch %d", epoch)
            logger.info("Freezing frontend, unfreezing backend")
            logger.info("=" * 60)

            # Purge ghost gradients before freezing frontend
            # Phase 1 MSE gradients are tiny → AMP accumulates a huge
            # Scale Factor.  Residual gradients on frozen frontend
            # parameters would cause clip_grad_norm_ to see Inf,
            # zeroing all trainable gradients to NaN.
            model.zero_grad(set_to_none=True)

            # Freeze frontend, unfreeze backend
            for param in model.frontend.parameters():
                param.requires_grad = False
            for param in model.backend.parameters():
                param.requires_grad = True

            # New optimizer for backend only (with per-pathway LRs)
            base_lr = config.training.learning_rate
            lif_lr = base_lr * 0.3
            lif_params = list(model.backend.lif_cell.parameters())
            lif_param_ids = {id(p) for p in lif_params}
            other_backend = [p for p in model.backend.parameters() if id(p) not in lif_param_ids]
            optimizer = torch.optim.AdamW([
                {"params": other_backend, "lr": base_lr, "base_lr": base_lr, "name": "non_lif"},
                {"params": lif_params, "lr": lif_lr, "base_lr": lif_lr, "name": "lif"},
            ], weight_decay=config.training.weight_decay)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(
                    1,
                    (config.training.num_epochs - epoch) - config.training.lr_warmup_epochs,
                ),
                eta_min=1e-6,
            )
            # LR budget accounting mirrors the default path: subtract the
            # warmup horizon from T_max and hold the cosine during warmup
            # (gated step below), so the phase-2 backend also anneals from
            # the full base LR instead of ramp-into-decayed-cosine.
            criterion = backend_criterion
            current_phase = 2
            best_val_loss = float("inf")
            epochs_without_improvement = 0
            best_path.unlink(missing_ok=True)  # Phase 1 loss cannot select Phase 2 weights.

        # ── Warmup factor for bio-loss terms (energy/sparse/jerk ONLY) ──
        # lambda_reg is NOT warmup-scaled: the router needs anti-collapse
        # pressure from epoch 0; ramping it with warmup_factor allows the
        # GRU to monopolise the hidden state before the gate can learn
        # (root cause of g_lif ~ 0.13 collapse).  Only lambda_energy,
        # lambda_sparse, and lambda_jerk are cosine-ramped.
        #
        # Two-phase fix: In Hybrid Funnel mode, Phase 2 starts at
        # global epoch = phase1_epochs.  We must use the *local*
        # Phase 2 epoch so the warmup restarts from 0 at the phase
        # transition; otherwise warmup_factor jumps straight to 1.0
        # and the untrained backend gets hit with full penalties.
        warmup_epoch = (epoch - phase1_epochs) if (two_phase and current_phase == 2) else epoch
        warmup_factor = compute_warmup_factor(warmup_epoch, warmup_epochs)
        if config.loss.lambda_routing_aux > 0.0:
            logger.info(
                "Epoch %d/%d  lambda_routing_aux=%.4f  effective=%.4f",
                epoch + 1,
                config.training.num_epochs,
                config.loss.lambda_routing_aux,
                config.loss.lambda_routing_aux * warmup_factor,
            )

        # ── LR warmup (linear ramp of the shared AdamW LR) ────
        # Applied *before* the epoch so every optimiser step this epoch
        # runs at the ramped LR.  Uses the same phase-local epoch so LR
        # warmup also restarts cleanly across the Phase 1→2 transition
        # (the backend's cold recurrent state gets a soft first step).
        # Dead-code guard: lr_warmup_epochs announced but previously
        # never consumed — this wires config.training.lr_warmup_epochs
        # into the loop, making the CLI flag meaningful.
        # Only override during the warmup window; once it ends the
        # override is skipped so the cosine scheduler's own `lr` value
        # (written by `scheduler.step()` at the prior epoch tail) is used
        # unchanged and the anneal proceeds from the full base LR.
        if (
            config.training.lr_warmup_epochs > 0
            and warmup_epoch < config.training.lr_warmup_epochs
        ):
            apply_lr_warmup(
                optimizer,
                epoch=warmup_epoch,
                lr_warmup_epochs=config.training.lr_warmup_epochs,
            )

        # ── Unfreeze if scheduled ─────────────────────────────
        if (
            config.finetune.unfreeze_after_epoch >= 0
            and epoch == config.finetune.unfreeze_after_epoch
        ):
            logger.info("Unfreezing all modules at epoch %d", epoch)
            for param in model.parameters():
                param.requires_grad = True

        # ── Train ─────────────────────────────────────────────
        train_loss, health = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            lambda_reg=lambda_reg,
            lambda_energy=config.loss.lambda_energy,
            lambda_sparse=config.loss.lambda_sparse,
            lambda_jerk=config.loss.lambda_jerk,
            annealing_factor=warmup_factor,
            grad_clip_norm=config.training.grad_clip_norm,
            log_interval=config.training.log_interval,
            epoch=epoch,
            lif_threshold=config.model.lif_threshold,
            scaler=scaler,
            amp_ctx=amp_ctx,
            phase=current_phase,
            target_mean=target_mean,
            target_std=target_std,
            target_clip_cm_s=config.training.target_clip_cm_s,
            lambda_routing_aux=config.loss.lambda_routing_aux,
            wind_only_mask_full=train_is_pure_wind,
            routing_aux_margin=config.loss.routing_aux_margin,
            lambda_compute=config.loss.lambda_compute,
        )
        # Advance the cosine.  When lr_warmup_epochs==0 (default), the step is
        # unconditional exactly as in the original pipeline, preserving the
        # byte-identical default LR trajectory (backward-compat invariant).
        # When warmup is active, the cosine is held while `apply_lr_warmup`
        # overrides the LR (so the anneal budget isn't consumed by the warmup
        # window), then released once the window has fully elapsed.
        _maybe_step_scheduler(
            scheduler, config.training.lr_warmup_epochs, warmup_epoch
        )
        history["train_loss"].append(train_loss)
        # First-class stability telemetry: how many optimizer steps were
        # silently dropped this epoch (non-finite loss / non-finite grad).
        skips = getattr(train_one_epoch, "last_skip_counts", None)
        if skips:
            logger.info(
                "Epoch %d skipped steps: %s",
                epoch,
                {k: v for k, v in skips.items() if v},
            )

        # ── Validate ──────────────────────────────────────────
        # CF1 fix: Validation uses FULL lambda values (no warmup scaling).
        val_loss = float("inf")
        if val_loader is not None:
            val_loss = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                lambda_reg=lambda_reg,
                lambda_energy=config.loss.lambda_energy,
                lambda_sparse=config.loss.lambda_sparse,
                lambda_jerk=config.loss.lambda_jerk,
                phase=current_phase,
                target_mean=target_mean,
                target_std=target_std,
                target_clip_cm_s=config.training.target_clip_cm_s,
                lambda_routing_aux=config.loss.lambda_routing_aux,
                wind_only_mask_full=val_is_pure_wind,
                routing_aux_margin=config.loss.routing_aux_margin,
                lambda_compute=config.loss.lambda_compute,
                selection_metric=_SELECTION_METRIC,
            )
            history["val_loss"].append(val_loss)
            val_mse_history.append(getattr(validate, "last_val_mse", None))
        else:
            # No validation loader: the epoch was still EXECUTED, so its
            # position on the shared epoch axis must exist.  ``None`` records
            # an unobserved diagnostic — not a finite value (which would
            # fabricate a patience opportunity) and not NaN/Inf (which would
            # claim a measured non-finite loss).  The counter increments only
            # behind ``math.isfinite(val_loss)`` and val_loss stays ``inf``
            # here, so patience does not advance on an unvalidated epoch.
            history["val_loss"].append(None)
            val_mse_history.append(None)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f  time=%.1fs",
            epoch + 1, config.training.num_epochs,
            train_loss, val_loss, elapsed,
        )

        # ── CF9: Per-epoch membrane health summary ─────────────
        if health:
            logger.info(
                "  Membrane: V_max=%.3f  V_mean=%.3f  spike_rate=%.4f  "
                "w_adapt=%.4f  (threshold=%.2f)",
                health["v_max"], health["v_mean"],
                health["spike_rate"], health["w_adapt"],
                config.model.lif_threshold,
            )
            if health["v_max"] > 10.0 * config.model.lif_threshold:
                logger.warning(
                    "  ⚠ V_max=%.2f >> threshold=%.2f — membrane runaway risk!",
                    health["v_max"], config.model.lif_threshold,
                )

        # ── Checkpointing ─────────────────────────────────────
        # Best-model checkpoint.
        # Round-4 fix: the previous gate ``warmup_factor >= 1.0`` made it
        # impossible to save *any* best checkpoint when the total number of
        # training epochs was <= warmup_epochs (e.g. short smoke tests,
        # or any --epochs <= config.loss.warmup_epochs).  ``best_val_loss``
        # stayed ``inf``, no ``best_model.pth`` was written, downstream
        # ``compute_metrics`` was skipped, and the pipeline gate failed.
        #
        # The warmup factor scales the bio-loss regularisation terms
        # (router reg, ATP cost, sparsity) but has NO effect on the
        # primary MSE reconstruction term that dominates val_loss during
        # warmup.  Comparing partially-regularised val_loss values across
        # warmup is valid: the MSE trend is monotonic and the bio-penalty
        # terms are only additive (removing the gate cannot select a
        # spuriously low val_loss caused by missing penalties — the
        # penalty is >= 0, so the warmed-up loss is always >=).
        #
        # Guard: val_loss must be finite (non-NaN, non-Inf) — heavy-tailed
        # targets with normalization disabled can produce non-finite loss,
        # which must not silently overwrite a good checkpoint.
        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_path = output_dir / "best_model.pth"
            _atomic_save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                config=config.to_dict(),
                path=best_path,
                train_loss=train_loss,
                val_loss=val_loss,
                target_mean=float(target_mean),
                target_std=float(target_std),
                target_clip_cm_s=float(config.training.target_clip_cm_s),
                training_phase=int(current_phase),
                dataset_path=str(resolved_dataset_path),
                **({"dataset_source_sha256": dataset_source_sha256} if dataset_source_sha256 is not None else {}),
                **({"dataset_source_binding": dataset_source_binding} if dataset_source_binding != "sha256_bound" else {}),
                mcmc_prior_train_serve_consistency=mcmc_prior_consistency,
                best_val_loss=float(best_val_loss),
                **_recovery_state_kwargs(
                    epochs_without_improvement, history, history_start_epoch,
                ),
                **nested_provenance,
                **phase2_provenance,
            )
            generated_checkpoint_shas[best_path] = hashlib.sha256(
                best_path.read_bytes(),
            ).hexdigest()
            logger.info("Saved best model (val_loss=%.6f): %s", val_loss, best_path)
        elif math.isfinite(val_loss):
            epochs_without_improvement += 1

        # Periodic checkpoint (saved after best_val_loss update so best_val_loss is consistent)
        if (epoch + 1) % config.training.checkpoint_interval == 0:
            epoch_path = output_dir / f"epoch_{epoch + 1}.pth"
            _atomic_save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=train_loss,
                config=config.to_dict(),
                path=epoch_path,
                train_loss=train_loss,
                val_loss=val_loss if val_loss != float("inf") else None,
                target_mean=float(target_mean),
                target_std=float(target_std),
                target_clip_cm_s=float(config.training.target_clip_cm_s),
                training_phase=int(current_phase),
                dataset_path=str(resolved_dataset_path),
                **({"dataset_source_sha256": dataset_source_sha256} if dataset_source_sha256 is not None else {}),
                **({"dataset_source_binding": dataset_source_binding} if dataset_source_binding != "sha256_bound" else {}),
                mcmc_prior_train_serve_consistency=mcmc_prior_consistency,
                best_val_loss=float(best_val_loss),
                **_recovery_state_kwargs(
                    epochs_without_improvement, history, history_start_epoch,
                ),
                **nested_provenance,
                **phase2_provenance,
            )
            logger.info("Saved periodic checkpoint: %s", epoch_path)

        # ── Early stopping check ─────────────────────────────
        # The SAME predicate a resume consults before its first epoch, so the
        # decision is identical whether it is reached inside the loop or
        # restored at a publication boundary.
        if _recovery_terminal(
            early_stopping_patience,
            epochs_without_improvement,
            two_phase=two_phase,
            current_phase=current_phase,
            num_epochs=config.training.num_epochs,
            phase1_epochs=phase1_epochs,
        ):
            logger.info(
                "Early stopping triggered: val_loss did not improve for %d epochs "
                "(best=%.6f at epoch %d)",
                early_stopping_patience,
                best_val_loss,
                epoch - epochs_without_improvement,
            )
            break

    # ── Final checkpoint ──────────────────────────────────────
    final_path = output_dir / "final_model.pth"
    _atomic_save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        loss=train_loss,
        config=config.to_dict(),
        path=final_path,
        train_loss=train_loss,
        val_loss=val_loss if val_loss != float("inf") else None,
        target_mean=float(target_mean),
        target_std=float(target_std),
        target_clip_cm_s=float(config.training.target_clip_cm_s),
        training_phase=int(current_phase),
        dataset_path=str(resolved_dataset_path),
        **({"dataset_source_sha256": dataset_source_sha256} if dataset_source_sha256 is not None else {}),
        **({"dataset_source_binding": dataset_source_binding} if dataset_source_binding != "sha256_bound" else {}),
        mcmc_prior_train_serve_consistency=mcmc_prior_consistency,
        best_val_loss=float(best_val_loss),
        **_recovery_state_kwargs(
            epochs_without_improvement, history, history_start_epoch,
        ),
        **nested_provenance,
        **phase2_provenance,
    )
    generated_checkpoint_shas[final_path] = hashlib.sha256(
        final_path.read_bytes(),
    ).hexdigest()
    logger.info("Saved final model: %s", final_path)

    logger.info("Final LR: %.2e", scheduler.get_last_lr()[0])
    logger.info("=" * 60)
    logger.info(
        "Training complete.  Best val %s: %.6f",
        _SELECTION_METRIC, best_val_loss,
    )
    logger.info("=" * 60)

    # ── Plot loss curve ──────────────────────────────────────────
    loss_curve_path = plot_loss_curve(history, output_dir, history_start_epoch)
    logger.info("Loss curve saved: %s", loss_curve_path)

    # ── Evaluate best model on validation set ────────────────────
    # If no best checkpoint was saved (e.g. val_loss was never finite),
    # fall back to the final checkpoint and flag the provenance so
    # downstream consumers know the model was NOT selected by val_loss.
    metrics: Dict[str, Any] = {}
    best_ckpt_path = output_dir / "best_model.pth"
    eval_ckpt_path: Optional[Path] = None
    eval_provenance = "best"
    if best_ckpt_path.exists():
        eval_ckpt_path = best_ckpt_path
    elif final_path.exists():
        eval_ckpt_path = final_path
        eval_provenance = "final_fallback"
        logger.warning(
            "No best_model.pth — falling back to final_model.pth for "
            "metric evaluation.  Val loss was never finite during "
            "training (best_val_loss=%.6f); metrics are computed on "
            "the LAST epoch's weights, NOT the best-validated.",
            best_val_loss,
        )

    if eval_ckpt_path is not None and val_loader is not None:
        eval_payload = eval_ckpt_path.read_bytes()
        eval_state = load_artifact_bytes(eval_payload, map_location="cpu")
        if eval_provenance == "best":
            certify_best_checkpoint(
                eval_state, eval_ckpt_path, eval_payload,
                evaluation=True,
            )
        else:
            certify_final_checkpoint(
                eval_state, eval_ckpt_path, eval_payload,
            )
        load_checkpoint(
            path=eval_ckpt_path,
            model=model,
            map_location=device,
            payload=eval_payload,
        )
        model.to(device)
        metrics = compute_metrics(
            model, val_loader, device,
            target_mean=target_mean, target_std=target_std,
            target_clip_cm_s=config.training.target_clip_cm_s,
            escape_band_cm_s=config.training.escape_band_cm_s,
            persistence_benchmark=True,
        )
        _log_best_metrics(metrics)
        metrics["eval_provenance"] = eval_provenance
        if mcmc_prior_consistency is not None:
            metrics["mcmc_prior_train_serve_consistency"] = mcmc_prior_consistency
        # Nested-prior provenance in metrics.json — a downstream reader
        # must be able to tie these numbers to a specific nested artifact
        # (or confirm the run was legacy) without loading the checkpoint.
        metrics.update(nested_provenance)
        if dataset_source_binding != "sha256_bound":
            metrics["dataset_source_binding"] = dataset_source_binding
        if dataset_source_sha256 is not None:
            metrics["dataset_source_sha256"] = dataset_source_sha256
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved: %s", metrics_path)

        # ── Band x min_run sensitivity sweep (opt-in) ──────────
        if _SWEEP_BANDS:
            all_true: List[np.ndarray] = []
            all_pred: List[np.ndarray] = []
            for batch in val_loader:
                assert len(batch) in (3, 4), f"Expected 3 or 4 batch fields, got {len(batch)}"
                x_batch, y_batch, lengths = batch[:3]
                x_batch = x_batch.to(device).contiguous()
                lengths = lengths.to(device).contiguous()
                with torch.no_grad():
                    y_pred, _ = model(x_batch, lengths, return_internals=True)
                    for i in range(x_batch.size(0)):
                        n = int(lengths[i])
                        p = y_pred[i, :n].cpu().numpy()
                        if target_std != 1.0 or target_mean != 0.0:
                            p = p * target_std + target_mean
                        all_pred.append(p)
                        all_true.append(y_batch[i, :n].cpu().numpy())
            rows = sweep_escape_sensitivity(
                all_true, all_pred, _SWEEP_BANDS,
            )
            sweep_path = output_dir / "escape_sensitivity.csv"
            with open(sweep_path, "w", encoding="utf-8") as f:
                keys = ["band_cm_s", "min_run", "n_escape_frames",
                        "n_escape_events", "escape_rmse", "resting_rmse",
                        "escape_ratio"]
                f.write(",".join(keys) + "\n")
                for r in rows:
                    f.write(",".join(str(r[k]) for k in keys) + "\n")
            logger.info("Escape sensitivity sweep (%d configs): %s",
                        len(rows), sweep_path)

    # ── Fail-loud guard: no checkpoint at all ────────────────────
    # If neither best nor final checkpoint was produced, the run is
    # broken.  Fail loudly rather than returning exit 0 with no model.
    if not best_ckpt_path.exists() and not final_path.exists():
        raise RuntimeError(
            "Training completed but NO checkpoint was written "
            "(neither best_model.pth nor final_model.pth).  "
            "This indicates a critical I/O or permission error."
        )

    return {
        "best_val_loss": best_val_loss,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float("inf"),
        "lambda_reg": lambda_reg,
        "metrics": metrics,
        "eval_provenance": eval_provenance,
        "history": history,
        # Masked-MSE-only validation diagnostic on the ALIGNED eligible frames
        # (t >= 1), positionally aligned with ``history["val_loss"]``.  NOT
        # persisted through resume.
        "val_mse_history": val_mse_history,
        # True 0-based epoch of history entry 0 (0 for a fresh run; the
        # resume epoch for a legacy resume whose earlier history is unknown).
        "history_start_epoch": history_start_epoch,
        # Nested-prior provenance (empty artifact / is_nested_cv False =
        # legacy run).  Mirrors what every checkpoint carries.
        "nested_prior_artifact": nested_provenance["nested_prior_artifact"],
        "nested_prior_artifact_sha256": nested_provenance["nested_prior_artifact_sha256"],
        "nested_prior_fingerprint": nested_provenance["nested_prior_fingerprint"],
        "nested_split_seed": nested_provenance["nested_split_seed"],
        "nested_val_split": nested_provenance["nested_val_split"],
        "is_nested_cv": nested_provenance["is_nested_cv"],
        "validation_scope": nested_provenance["validation_scope"],
        "mcmc_prior_provenance": nested_provenance["mcmc_prior_provenance"],
        "animal_identity_status": nested_provenance["animal_identity_status"],
        "dataset_source_sha256": dataset_source_sha256,
        "dataset_source_binding": dataset_source_binding,
    }


# ═══════════════════════════════════════════════════════════════
# 7.  CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    CLI entry point.

    Parses arguments, loads config, and runs :func:`train`.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    config, lambda_reg, phase1_epochs = build_config(argv)
    args = build_arg_parser().parse_args(argv)
    dataset_path = args.dataset
    lazy_loading = args.lazy_loading
    nested_prior_artifact = args.nested_prior_artifact
    logger.info("Config loaded: %s", config.checkpoint.output_dir)

    output_dir = Path(config.checkpoint.output_dir)
    results = train(
        config,
        lambda_reg=lambda_reg,
        phase1_epochs=phase1_epochs,
        dataset_path=dataset_path,
        use_lazy_loading=lazy_loading,
        nested_prior_artifact=nested_prior_artifact,
        require_nested_validation=not args.diagnostic_only,
        trusted_historical_checkpoint_sha256=args.trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=args.trusted_historical_artifact_sha256,
    )
    train_log_path = output_dir / "train.log"
    with open(train_log_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results: %s. Saved: %s", results, train_log_path)


if __name__ == "__main__":
    main()
