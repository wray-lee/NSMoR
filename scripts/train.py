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
    results = train(cfg, lambda_reg=0.01)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import math

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
from nsmor.loss import BioJointLoss
from nsmor.model_nsmor_core import NSMoRCore

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
        default=0.01,
        help="Router regularization weight for BioJointLoss.",
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

    return parser


def build_config(argv: Optional[Sequence[str]] = None) -> Tuple[ExperimentConfig, float]:
    """
    Parse CLI arguments and return a fully resolved config plus lambda_reg.

    If ``--config`` is given, YAML is loaded first, then CLI flags
    override individual values.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        ``(config, lambda_reg)`` tuple.
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
    if args.freeze is not None:
        config.finetune.freeze_modules = args.freeze
    if args.resume is not None:
        config.checkpoint.resume_from = args.resume
    if args.output_dir is not None:
        config.checkpoint.output_dir = args.output_dir

    return config, args.lambda_reg


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
        lif_alpha=config.model.lif_alpha,
        lif_threshold=config.model.lif_threshold,
        lif_beta=config.model.lif_beta,
        lif_abs_refract_steps=config.model.lif_abs_refract_steps,
        lif_rel_refract_steps=config.model.lif_rel_refract_steps,
        lif_tau_syn=config.model.lif_tau_syn,
        lif_v_rest=config.model.lif_v_rest,
        lif_v_reset=config.model.lif_v_reset,
        lif_tau_w=config.model.lif_tau_w,
        lif_b_adapt=config.model.lif_b_adapt,
        lif_tau_fac=config.model.lif_tau_fac,
        lif_tau_rec=config.model.lif_tau_rec,
        lif_U_stp_init=config.model.lif_U_stp_init,
        lif_lateral_inhibition=config.model.lif_lateral_inhibition,
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
        {"params": other_params, "lr": base_lr, "name": "non_lif"},
        {"params": lif_params, "lr": lif_lr, "name": "lif"},
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

def build_dataloaders(
    config: ExperimentConfig,
    dataset_path: str = "data/processed/nsmor_dataset.pt",
    val_split: float = 0.2,
) -> Tuple[Optional[torch.utils.data.DataLoader], Optional[torch.utils.data.DataLoader]]:
    """
    Build train and validation dataloaders from the prepared dataset.

    Loads the preprocessed dataset from ``nsmor_dataset.pt`` (produced
    by ``scripts/prepare_data.py``), performs a deterministic train/val
    split, and returns two DataLoader instances.

    Args:
        config: Parsed experiment configuration.
        dataset_path: Path to the preprocessed dataset file.
        val_split: Fraction of data to use for validation (0-1).

    Returns:
        ``(train_loader, val_loader)`` — either may be ``None`` if
        the dataset file is not found or the split is empty.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    from nsmor.nsmor_dataloader import (
        NSMoRDataset,
        collate_variable_length,
    )

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        logger.warning(
            "Dataset file not found: %s.  "
            "Run 'python scripts/prepare_data.py' first.",
            dataset_file,
        )
        return None, None

    # ── Load preprocessed dataset ─────────────────────────────
    logger.info("Loading dataset from %s", dataset_file)
    dataset = torch.load(dataset_file, weights_only=False)

    X_seqs = dataset["X_seqs"]
    Y_seqs = dataset["Y_seqs"]
    mcmc_priors = dataset["mcmc_priors"]
    labels = dataset["labels"]
    lengths = dataset["lengths"]

    n_total = len(X_seqs)
    logger.info(
        "Loaded %d sequences, total_frames=%d",
        n_total, int(lengths.sum()),
    )

    # ── Deterministic train/val split ─────────────────────────
    rng = np.random.RandomState(config.training.random_seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

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
    )
    val_dataset = NSMoRDataset(
        sequences=val_sequences,
        mcmc_priors=val_priors,
        feature_config=feature_config,
        max_seq_len=max_seq_len,
    )

    # ── Create dataloaders ────────────────────────────────────
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_variable_length,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_length,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "DataLoaders created: train=%d batches, val=%d batches (batch_size=%d)",
        len(train_loader), len(val_loader), config.training.batch_size,
    )

    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════
# 4.  Training Loop
# ═══════════════════════════════════════════════════════════════

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

    Note: ``lambda_reg`` is NOT scaled by this factor -- only
    ``lambda_energy``, ``lambda_sparse``, and ``lambda_jerk`` are.

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
    criterion: BioJointLoss,
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
) -> float:
    """
    Run one training epoch.

    Args:
        model: The NSMoR model.
        loader: Training DataLoader yielding ``(X_batch, Y_batch, lengths)``.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        lambda_reg: Router regularization weight.  NOT scaled by
            annealing_factor (active from epoch 0).
        lambda_energy: ATP metabolic cost weight (base value before
            annealing).
        lambda_sparse: Population sparsity L1 weight (base value).
        lambda_jerk: Temporal coherence weight (base value).
        annealing_factor: Scaling factor for lambda_energy, lambda_sparse,
            and lambda_jerk.  Typically equals the warmup factor from
            ``compute_warmup_factor(epoch, warmup_epochs)``.  Default 1.0
            (no annealing).  lambda_reg is NOT affected.
        grad_clip_norm: Max gradient norm for clipping.
        log_interval: Log every N batches.
        epoch: Current epoch number (for logging).

    Returns:
        Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False, dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch — expect (X, Y, lengths) from collate_variable_length
        x_batch, y_batch, lengths = batch
        x_batch = x_batch.to(device).contiguous()
        y_batch = y_batch.to(device).contiguous()
        lengths = lengths.to(device).contiguous()

        # ── Forward pass (with internals for routing gates) ──
        y_pred, internals = model(x_batch, lengths, return_internals=True)

        # ── Extract g_gru from routing gates ──
        # routing_gates: (B, T, 2) — index 1 is g_gru
        g_gru = internals["routing_gates"][:, :, 1:2]           # (B, T, 1)
        lif_spikes = internals["lif_spikes"]                    # (B, T, H)

        # ── Compute loss ──
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
        )

        # ── Membrane health monitoring (CF7 fix) ──
        # Logs V_max and spike rate for early detection of runaway
        # membrane potential or LIF pathway collapse.
        if batch_idx == 0 and epoch % 10 == 0:
            with torch.no_grad():
                lif_potentials = internals["lif_potentials"]
                v_max = lif_potentials.abs().max().item()
                v_mean = lif_potentials.abs().mean().item()
                spike_rate = lif_spikes.float().mean().item()
                logger.info(
                    "Epoch %d LIF stats: V_max=%.3f V_mean=%.3f "
                    "spike_rate=%.4f (threshold=%.2f)",
                    epoch, v_max, v_mean, spike_rate, lif_threshold,
                )
                if v_max > 10.0 * lif_threshold:
                    logger.warning(
                        "Epoch %d: V_max=%.2f >> threshold=%.2f — "
                        "potential membrane runaway detected!",
                        epoch, v_max, lif_threshold,
                    )

        # ── Backward pass ──
        optimizer.zero_grad()
        loss.backward()

        # ── NaN/Inf guard (Issues 1 & 2) ──
        # Detect non-finite loss BEFORE clipping so we can skip the
        # optimizer step and avoid poisoning Adam's moment estimates.
        if not math.isfinite(loss.item()):
            logger.warning(
                "Epoch %d batch %d: non-finite loss=%s — skipping step",
                epoch, batch_idx, loss.item(),
            )
            continue  # skip optimizer.step()

        # ── Gradient clipping ──
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm,
            )

        # ── Post-clip gradient finiteness check (defense-in-depth) ──
        # Even after clipping, NaN gradients can survive if the grad
        # was already NaN (clip_grad_norm_ divides by a norm that may
        # be NaN).  Detect and skip to avoid corrupting parameters.
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                has_nan_grad = True
                break
        if has_nan_grad:
            logger.warning(
                "Epoch %d batch %d: non-finite gradient after clipping — skipping step",
                epoch, batch_idx,
            )
            continue  # skip optimizer.step()

        # ── Per-pathway gradient norm logging (CF7 fix) ──
        # Monitors gradient balance between LIF and non-LIF pathways.
        # If LIF gradients are consistently 10x+ larger, it confirms
        # the LIF pathway as the instability source.
        if batch_idx == 0 and epoch % 10 == 0:
            lif_grad_norm = 0.0
            non_lif_grad_norm = 0.0
            for name, p in model.named_parameters():
                if p.grad is not None:
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

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # ── Logging ──
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


# ═══════════════════════════════════════════════════════════════
# 5.  Validation Loop
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model: NSMoRCore,
    loader: torch.utils.data.DataLoader,
    criterion: BioJointLoss,
    device: torch.device,
    lambda_reg: float = 0.01,
    lambda_energy: float = 0.0,
    lambda_sparse: float = 0.0,
    lambda_jerk: float = 0.0,
) -> float:
    """
    Run validation (no gradient computation).

    Args:
        model: The NSMoR model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Device.
        lambda_reg: Router regularization weight.
        lambda_energy: ATP metabolic cost weight (full value, NOT annealed).
        lambda_sparse: Population sparsity L1 weight (full value).
        lambda_jerk: Temporal coherence weight (full value).

    Note:
        Validation uses FULL lambda values (no annealing_factor).
        This is intentional: annealing would bias best_model selection
        toward early epochs where bio-loss is artificially suppressed,
        producing a lower val_loss that doesn't reflect true performance.
        See CF1 fix in train() for details.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Validation", leave=False, dynamic_ncols=True)
    for batch in pbar:
        x_batch, y_batch, lengths = batch
        x_batch = x_batch.to(device).contiguous()
        y_batch = y_batch.to(device).contiguous()
        lengths = lengths.to(device).contiguous()

        y_pred, internals = model(x_batch, lengths, return_internals=True)
        g_gru = internals["routing_gates"][:, :, 1:2]
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
def compute_metrics(
    model: NSMoRCore,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
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

    Returns:
        Dictionary with keys ``"mse"``, ``"rmse"``, ``"mae"``, ``"r2"``.
    """
    model.eval()
    all_pred: List[np.ndarray] = []
    all_true: List[np.ndarray] = []

    for batch in loader:
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

    mse = float(mean_squared_error(y_true_all, y_pred_all))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true_all, y_pred_all))
    r2 = float(r2_score(y_true_all, y_pred_all))

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


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
    lambda_reg: float = 0.01,
) -> Dict[str, float]:
    """
    Full training pipeline.

    Args:
        config: Parsed experiment configuration.
        lambda_reg: Router regularization weight for BioJointLoss.

    Returns:
        Dictionary with ``"best_val_loss"`` and ``"final_train_loss"``.

    Raises:
        ValueError: If no training data is provided (loader is None).
    """
    # ── Reproducibility ───────────────────────────────────────
    torch.manual_seed(config.training.random_seed)
    np.random.seed(config.training.random_seed)

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("lambda_reg: %.4f", lambda_reg)
    if config.training.max_seq_len is not None:
        logger.info("max_seq_len: %d (sequences will be cropped)", config.training.max_seq_len)

    # ── Build components ──────────────────────────────────────
    model = build_model(config).to(device)

    for m in model.modules():
        if isinstance(m, nn.RNNBase):
            m.flatten_parameters()
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.num_epochs, eta_min=1e-6,
    )
    criterion = build_loss(config)
    train_loader, val_loader = build_dataloaders(config)

    if train_loader is None:
        raise ValueError(
            "Training DataLoader is None.  "
            "Wire build_dataloaders() to the real data pipeline."
        )

    # ── Apply freezing strategy ───────────────────────────────
    if config.finetune.freeze_modules:
        logger.info(
            "Freezing modules: %s", config.finetune.freeze_modules,
        )
        model.freeze_modules(config.finetune.freeze_modules)

    # ── Resume from checkpoint ────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")

    if config.checkpoint.resume_from is not None:
        ckpt_path = Path(config.checkpoint.resume_from)
        if ckpt_path.exists():
            logger.info("Resuming from checkpoint: %s", ckpt_path)
            checkpoint = load_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                map_location=device,
            )
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("loss", float("inf"))
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

    # ── Bio-loss warmup schedule ─────────────────────────────
    warmup_epochs = config.loss.warmup_epochs

    for epoch in range(start_epoch, config.training.num_epochs):
        t0 = time.time()

        # ── Warmup factor for bio-loss terms (lambda_reg NOT scaled) ──
        warmup_factor = compute_warmup_factor(epoch, warmup_epochs)

        # ── Unfreeze if scheduled ─────────────────────────────
        if (
            config.finetune.unfreeze_after_epoch >= 0
            and epoch == config.finetune.unfreeze_after_epoch
        ):
            logger.info("Unfreezing all modules at epoch %d", epoch)
            for param in model.parameters():
                param.requires_grad = True

        # ── Train ─────────────────────────────────────────────
        train_loss = train_one_epoch(
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
        )
        scheduler.step()
        history["train_loss"].append(train_loss)

        # ── Validate ──────────────────────────────────────────
        # CF1 fix: Validation uses FULL lambda values (no warmup scaling).
        # Warmup only applies to training gradients.  If validation also
        # scaled by warmup_factor, best_model selection would be biased
        # toward early epochs where bio-loss is artificially suppressed,
        # producing a lower val_loss that doesn't reflect true performance.
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
            )
            history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f  time=%.1fs",
            epoch + 1, config.training.num_epochs,
            train_loss, val_loss, elapsed,
        )

        # ── Checkpointing ─────────────────────────────────────
        # Periodic checkpoint
        if (epoch + 1) % config.training.checkpoint_interval == 0:
            epoch_path = output_dir / f"epoch_{epoch + 1}.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                config=config.to_dict(),
                path=epoch_path,
                train_loss=train_loss,
                val_loss=val_loss if val_loss != float("inf") else None,
            )
            logger.info("Saved periodic checkpoint: %s", epoch_path)

        # Best-model checkpoint (skip during warmup to avoid scale mismatch)
        if warmup_factor >= 1.0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best_model.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                config=config.to_dict(),
                path=best_path,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            logger.info("Saved best model (val_loss=%.6f): %s", val_loss, best_path)

    # ── Final checkpoint ──────────────────────────────────────
    final_path = output_dir / "final_model.pth"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config.training.num_epochs - 1,
        loss=train_loss,
        config=config.to_dict(),
        path=final_path,
        train_loss=train_loss,
        val_loss=val_loss if val_loss != float("inf") else None,
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
    metrics: Dict[str, float] = {}
    best_ckpt_path = output_dir / "best_model.pth"
    if best_ckpt_path.exists() and val_loader is not None:
        load_checkpoint(path=best_ckpt_path, model=model, map_location=device)
        model.to(device)
        metrics = compute_metrics(model, val_loader, device)
        logger.info(
            "Best model metrics — MSE: %.6f  RMSE: %.6f  MAE: %.6f  R²: %.4f",
            metrics["mse"], metrics["rmse"], metrics["mae"], metrics["r2"],
        )
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved: %s", metrics_path)

    return {
        "best_val_loss": best_val_loss,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float("inf"),
        "metrics": metrics,
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
    config, lambda_reg = build_config(argv)
    logger.info("Config loaded: %s", config.checkpoint.output_dir)

    output_dir = Path(config.checkpoint.output_dir)
    results = train(config, lambda_reg=lambda_reg)
    train_log_path = output_dir / "train.log"
    with open(train_log_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results: %s. Saved: %s", results, train_log_path)


if __name__ == "__main__":
    main()
