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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
})


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

    from nsmor.model_utils import validate_dataset_provenance

    # ── ELT Mode: Lazy Loading ────────────────────────────────
    if use_lazy_loading:
        from nsmor.lazy_dataloader import NSMoRLazyDataset

        logger.info("Loading metadata from %s (lazy mode)", dataset_file)

        # Read one metadata snapshot for both lazy rows and provenance.
        raw_meta, loaded_source_fingerprint = load_dataset_with_fingerprint(
            dataset_file, expected_dt_ms=config.model.dt_ms, restore_provenance=False,
        )
        prior_status = validate_dataset_provenance(raw_meta, dataset_file)
        prior_tag = raw_meta["mcmc_prior_provenance"]
        full_dataset = NSMoRLazyDataset(
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
        lambda_jerk: Temporal coherence weight.
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
        lambda_jerk: Temporal coherence weight.
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

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
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
            )

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(n_batches, 1)
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


@torch.no_grad()
def compute_metrics(
    model: NSMoRCore,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    target_clip_cm_s: float = 0.0,
    escape_band_cm_s: float = 10.0,
) -> Dict[str, float]:
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

    y_pred_all = np.concatenate(all_pred)
    y_true_all = np.concatenate(all_true)

    # Rescale predictions from standardized units back to cm/s (units
    # matching y_true) so metrics are reported in physical velocity units.
    if target_std != 1.0 or target_mean != 0.0:
        y_pred_all = y_pred_all * target_std + target_mean

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
    escape_rmse = float(np.sqrt(mean_squared_error(
        y_true_all_raw[is_escape], y_pred_all_raw[is_escape]))) if n_escape else float("nan")
    resting_rmse = float(np.sqrt(mean_squared_error(
        y_true_all_raw[~is_escape], y_pred_all_raw[~is_escape]))) if (~is_escape).any() else float("nan")

    metrics: Dict[str, float] = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "escape_band_cm_s": escape_band_cm_s,
        "n_escape_frames": float(n_escape),
        "escape_rmse": escape_rmse,
        "resting_rmse": resting_rmse,
        "escape_ratio": n_escape / max(1, int(y_true_all.size)),
    }
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
) -> Path:
    """
    Plot train/val loss curves and save to disk.

    Args:
        history: Dictionary with ``"train_loss"`` and ``"val_loss"`` lists.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.5)
    if history["val_loss"]:
        ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.5)

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
      ATP + sparsity + jerk).

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
        if start_epoch >= config.training.num_epochs:
            raise ValueError(
                f"Cannot resume checkpoint {ckpt_path} at epoch {start_epoch}: "
                f"target num_epochs ({config.training.num_epochs}) <= start_epoch ({start_epoch}); "
                "no remaining epochs to train (fail closed on validation provenance)."
            )
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
        _landing_phase2 = two_phase and start_epoch > phase1_epochs
        _crosses_boundary = two_phase and start_epoch == phase1_epochs

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

    history = {"train_loss": [], "val_loss": []}
    train_loss: float = float("nan")
    val_loss: float = float("nan")

    # ── Early stopping ───────────────────────────────────────
    early_stopping_patience = getattr(
        config.training, "early_stopping_patience", 0
    )
    epochs_without_improvement = 0

    # ── Bio-loss warmup schedule ─────────────────────────────
    warmup_epochs = config.loss.warmup_epochs

    epoch = start_epoch - 1
    for epoch in range(start_epoch, config.training.num_epochs):
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
            )
            history["val_loss"].append(val_loss)

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
                **nested_provenance,
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
                **nested_provenance,
            )
            logger.info("Saved periodic checkpoint: %s", epoch_path)

        # ── Early stopping check ─────────────────────────────
        if (
            early_stopping_patience > 0
            and not (two_phase and current_phase == 1 and config.training.num_epochs > phase1_epochs)
            and epochs_without_improvement >= early_stopping_patience
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
        **nested_provenance,
    )
    generated_checkpoint_shas[final_path] = hashlib.sha256(
        final_path.read_bytes(),
    ).hexdigest()
    logger.info("Saved final model: %s", final_path)

    logger.info("Final LR: %.2e", scheduler.get_last_lr()[0])
    logger.info("=" * 60)
    logger.info("Training complete.  Best val loss: %.6f", best_val_loss)
    logger.info("=" * 60)

    # ── Plot loss curve ──────────────────────────────────────────
    loss_curve_path = plot_loss_curve(history, output_dir)
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
        )
        logger.info(
            "Best model metrics — MSE: %.6f  RMSE: %.6f  MAE: %.6f  R²: %.4f",
            metrics["mse"], metrics["rmse"], metrics["mae"], metrics["r2"],
        )
        if "escape_rmse" in metrics:
            logger.info(
                "Escape-signal audit — escape_band=%.1f cm/s  n_escape=%d (%.3f%% of frames)  "
                "escape_rmse=%.4f  resting_rmse=%.4f",
                metrics["escape_band_cm_s"],
                metrics["n_escape_frames"],
                metrics["escape_ratio"] * 100.0,
                metrics["escape_rmse"],
                metrics["resting_rmse"],
            )
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
