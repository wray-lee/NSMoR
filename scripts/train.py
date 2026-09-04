"""
NSMoR Main Training Engine.

Ties together the full training pipeline:

1. Load experiment configuration from YAML + CLI overrides.
2. Initialize model, optimizer, loss function, and dataloaders.
3. Run the training loop with validation and checkpointing.

Usage
-----
CLI::

    python scripts/train.py --config config/default.yaml
    python scripts/train.py --config config/default.yaml --lr 5e-4 --epochs 200
    python scripts/train.py --config config/default.yaml --batch_size 64 --lambda_reg 0.05

Programmatic::

    from scripts.train import train, build_config
    cfg = build_config(["--config", "config/default.yaml"])
    results = train(cfg)  # lambda_reg falls through to cfg.loss.lambda_reg
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from nsmor.checkpoint import load_checkpoint, save_checkpoint
from nsmor.config import DEFAULT_FEATURE
from nsmor.config_parser import ExperimentConfig
from nsmor.dataloader_factory import create_dataloaders_from_config
from nsmor.loss import BioJointLoss, BioDecisionLoss, FrontendLoss
from nsmor.model_nsmor_core import NSMoRCore
from nsmor.pipeline.conditions import derive_stimulus_metadata


# ── Deployment provenance keys ────────────────────────────────
# These keys are injected into every checkpoint by
# _atomic_save_checkpoint but are NOT part of the frozen
# save_checkpoint signature.  The wrapper pops them before
# forwarding kwargs, then patches them into the saved state
# dict on disk before the atomic rename.
_PROVENANCE_KEYS = frozenset({
    "target_mean",
    "target_std",
    "target_clip_cm_s",
    "training_phase",
    "dataset_path",
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
        state = torch.load(tmp, map_location="cpu", weights_only=False)
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

# Single source of truth for the train/val split fraction, threaded to BOTH
# build_dataloaders and compute_target_stats — otherwise a non-default split
# silently desynchronises the normalization statistics from the training set
# (validation leakage into target mean/std).
_VAL_SPLIT = 0.2

# Default dataset for build_dataloaders and compute_target_stats.  Both
# call sites receive the same resolved path, which ``--dataset`` overrides
# so a run trains on the dataset its own ETL stage produced.
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


def build_dataloaders(
    config: ExperimentConfig,
    dataset_path: str = "data/processed/nsmor_dataset.pt",
    val_split: float = 0.2,
    use_lazy_loading: bool = False,
) -> Tuple[Optional[torch.utils.data.DataLoader], Optional[torch.utils.data.DataLoader]]:
    """
    Build train and validation dataloaders from the prepared dataset.

    Supports two modes:
    - ETL mode (default): Load pre-processed dataset from ``nsmor_dataset.pt``
    - ELT mode (lazy): Load metadata and read trials on-demand from CSVs

    Args:
        config: Parsed experiment configuration.
        dataset_path: Path to dataset/metadata file.
        val_split: Fraction of data to use for validation (0-1).
        use_lazy_loading: If True, use ELT mode with lazy loading.

    Returns:
        ``(train_loader, val_loader)`` — either may be ``None`` if
        the dataset file is not found or the split is empty.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
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

    # ── ELT Mode: Lazy Loading ────────────────────────────────
    if use_lazy_loading:
        from nsmor.lazy_dataloader import NSMoRLazyDataset

        logger.info("Loading metadata from %s (lazy mode)", dataset_file)

        # Create full dataset
        full_dataset = NSMoRLazyDataset(
            metadata_path=str(dataset_file),
            max_seq_len=config.training.max_seq_len,
            dt_ms=config.model.dt_ms,
        )

        # Session-grouped split
        session_ids = [full_dataset.get_session_id(i) for i in range(len(full_dataset))]
        unique_sessions = list(set(session_ids))

        rng = np.random.RandomState(config.training.random_seed)
        rng.shuffle(unique_sessions)

        n_val_sessions = max(1, int(len(unique_sessions) * val_split))
        val_sessions = set(unique_sessions[:n_val_sessions])

        train_indices = [i for i in range(len(full_dataset))
                        if session_ids[i] not in val_sessions]
        val_indices = [i for i in range(len(full_dataset))
                      if session_ids[i] in val_sessions]

        logger.info(
            "Session-grouped split: %d train (%d sessions), %d val (%d sessions)",
            len(train_indices),
            len(unique_sessions) - n_val_sessions,
            len(val_indices),
            n_val_sessions,
        )

        # Create subset datasets
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        # Create dataloaders with collate function
        def collate_fn(batch):
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
    dataset = torch.load(dataset_file, weights_only=False)

    # Round-2 CRITICAL-A: refuse pre-2.0 datasets (leaked priors,
    # np.max labels) — training on them is scientifically invalid.
    from nsmor.model_utils import validate_dataset_provenance
    validate_dataset_provenance(dataset, Path(dataset_file))

    X_seqs = dataset["X_seqs"]
    Y_seqs = dataset["Y_seqs"]
    mcmc_priors = dataset["mcmc_priors"]
    labels = dataset["labels"]
    lengths = dataset["lengths"]

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

    # ── Deterministic train/val split ─────────────────────────
    # Round-3 fix (Reviewer B CRITICAL-3b): the split is SESSION-GROUPED.
    # A sample-level shuffle lets one recording session straddle train
    # and validation — trials of the same session share the animal's
    # baseline locomotor statistics and gain state, so val metrics were
    # mildly inflated by session-level information.  Whole sessions are
    # now assigned to one side.  (Full nested CV — outer NSMoR split,
    # inner cross-fitting restricted to the outer-training sessions —
    # remains a documented limitation; grouped splitting removes the
    # dominant first-order channel at negligible cost.)
    rng = np.random.RandomState(config.training.random_seed)
    session_ids = dataset.get("session_ids")
    if session_ids is not None and len(session_ids) == n_total:
        session_arr = np.asarray(session_ids)
        unique_sessions = np.unique(session_arr)
        rng.shuffle(unique_sessions)
        n_val_sessions = max(1, int(len(unique_sessions) * val_split))
        val_sessions = set(unique_sessions[:n_val_sessions].tolist())
        is_val = np.array([s in val_sessions for s in session_arr])
        val_indices = np.nonzero(is_val)[0]
        train_indices = np.nonzero(~is_val)[0]
        logger.info(
            "Session-grouped split: %d train (%d sessions), "
            "%d val (%d sessions)",
            len(train_indices),
            len(set(session_arr[train_indices].tolist())),
            len(val_indices), n_val_sessions,
        )
    else:
        # Fallback: sample-level shuffle (datasets without session ids;
        # synthetic data only).  Warn loudly — this split is leak-prone.
        logger.warning(
            "Dataset lacks per-sequence 'session_ids'; falling back to "
            "sample-level train/val split.  Session-level information "
            "may inflate validation metrics.  Re-run prepare_data to "
            "regenerate the dataset with session ids."
        )
        indices = np.arange(n_total)
        rng.shuffle(indices)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
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

    # ── Create datasets ───────────────────────────────────────
    feature_config = dataset.get("feature_config", DEFAULT_FEATURE)

    max_seq_len = getattr(config.training, "max_seq_len", None)

    train_dataset = NSMoRDataset(
        sequences=train_sequences,
        mcmc_priors=train_priors,
        feature_config=feature_config,
        max_seq_len=max_seq_len,
        source_indices=train_indices,
        is_pure_wind=train_is_pure_wind,
    )
    val_dataset = NSMoRDataset(
        sequences=val_sequences,
        mcmc_priors=val_priors,
        feature_config=feature_config,
        max_seq_len=max_seq_len,
        source_indices=val_indices,
        is_pure_wind=val_is_pure_wind,
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
) -> Tuple[float, float, np.ndarray]:
    """
    Compute training-split velocity mean and std for target normalization.

    Uses the *same* deterministic train/val split as
    :func:`build_dataloaders` (seeded by ``config.training.random_seed``)
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
            :func:`build_dataloaders`).

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

    dataset = torch.load(dataset_file, weights_only=False)
    Y_seqs = dataset["Y_seqs"]
    n_total = len(Y_seqs)

    # ── Session-grouped split (must mirror build_dataloaders exactly) ──
    # Round-4 fix: the previous implementation used a SAMPLE-level
    # shuffle while build_dataloaders splits by SESSION.  That meant
    # normalization statistics were fit on data that included validation
    # sessions — a subtle data-leakage channel.  We now replicate the
    # exact same session-grouped logic so the two code paths produce
    # byte-identical train_indices from the same seed.
    rng = np.random.RandomState(config.training.random_seed)
    session_ids = dataset.get("session_ids")
    if session_ids is not None and len(session_ids) == n_total:
        session_arr = np.asarray(session_ids)
        unique_sessions = np.unique(session_arr)
        rng.shuffle(unique_sessions)
        n_val_sessions = max(1, int(len(unique_sessions) * val_split))
        val_sessions = set(unique_sessions[:n_val_sessions].tolist())
        is_val = np.array([s in val_sessions for s in session_arr])
        train_indices = np.nonzero(~is_val)[0].astype(np.int64)
    else:
        indices = np.arange(n_total, dtype=np.int64)
        rng.shuffle(indices)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        train_indices = indices[:n_train]

    train_y = np.concatenate([Y_seqs[i] for i in train_indices]).astype(np.float64)

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
) -> Dict[str, float]:
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

    Returns:
        Dictionary with ``"best_val_loss`` and ``"final_train_loss"``.

    Raises:
        ValueError: If no training data is provided (loader is None).
    """
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

    # ── Target normalization statistics (train split only) ──
    # When config.training.normalize_targets is enabled, the velocity
    # target is mean-centered and std-scaled using training-split
    # statistics.  These same values are threaded through train_one_epoch,
    # validate, and compute_metrics so the loss, the selection-criteria,
    # and the reported metrics all live in one consistent space, and the
    # final metrics are rescaled back to cm/s.
    target_mean, target_std, _train_indices = compute_target_stats(
        resolved_dataset_path,
        config,
        val_split=_VAL_SPLIT,
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
        if ckpt_path.exists():
            logger.info("Resuming from checkpoint: %s", ckpt_path)
            # Peek BEFORE load_checkpoint: (a) detect legacy checkpoints
            # without scheduler state (loud warning instead of silent LR
            # restart), and (b) learn start_epoch so the restore can match
            # the phase the run will continue in.
            ckpt_peek = torch.load(
                ckpt_path, map_location="cpu", weights_only=False,
            )
            if "scheduler_state_dict" not in ckpt_peek:
                logger.warning(
                    "Checkpoint %s has NO scheduler_state_dict (legacy "
                    "format) — LR schedule restarts from scratch; resumed "
                    "runs will NOT match an uninterrupted trajectory.",
                    ckpt_path,
                )
            start_epoch = ckpt_peek.get("epoch", -1) + 1

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
                )
            else:
                load_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    map_location=device,
                )
            best_val_loss = ckpt_peek.get("loss", float("inf"))
            logger.info("Resumed at epoch %d, loss=%.6f", start_epoch, best_val_loss)
        else:
            logger.warning("Checkpoint not found: %s — starting fresh", ckpt_path)

    # ── Output directory ──────────────────────────────────────
    output_dir = Path(config.checkpoint.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting training for %d epochs", config.training.num_epochs)
    logger.info("=" * 60)

    history = {"train_loss": [], "val_loss": []}

    # ── Early stopping ───────────────────────────────────────
    early_stopping_patience = getattr(
        config.training, "early_stopping_patience", 0
    )
    epochs_without_improvement = 0

    # ── Bio-loss warmup schedule ─────────────────────────────
    warmup_epochs = config.loss.warmup_epochs

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
        # Periodic checkpoint
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
            )
            logger.info("Saved periodic checkpoint: %s", epoch_path)

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
            )
            logger.info("Saved best model (val_loss=%.6f): %s", val_loss, best_path)
        elif math.isfinite(val_loss):
            epochs_without_improvement += 1

        # ── Early stopping check ─────────────────────────────
        if (
            early_stopping_patience > 0
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
        epoch=config.training.num_epochs - 1,
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
    )
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
    metrics: Dict[str, float] = {}
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
        load_checkpoint(path=eval_ckpt_path, model=model, map_location=device)
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
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved: %s", metrics_path)

        # ── Band x min_run sensitivity sweep (opt-in) ──────────
        if _SWEEP_BANDS:
            all_true: List[np.ndarray] = []
            all_pred: List[np.ndarray] = []
            for batch in val_loader:
                x_batch, y_batch, lengths = batch
                x_batch = x_batch.to(device).contiguous()
                lengths = lengths.to(device).contiguous()
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
    logger.info("Config loaded: %s", config.checkpoint.output_dir)

    output_dir = Path(config.checkpoint.output_dir)
    results = train(
        config,
        lambda_reg=lambda_reg,
        phase1_epochs=phase1_epochs,
        dataset_path=dataset_path,
        use_lazy_loading=lazy_loading,
    )
    train_log_path = output_dir / "train.log"
    with open(train_log_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results: %s. Saved: %s", results, train_log_path)


if __name__ == "__main__":
    main()
