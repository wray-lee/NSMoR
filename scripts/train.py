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
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

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
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model initialized — %s parameters", f"{param_count:,}")
    return model


def build_optimizer(
    model: nn.Module,
    config: ExperimentConfig,
) -> torch.optim.AdamW:
    """
    Construct an ``AdamW`` optimizer from the experiment config.

    Args:
        model: The model whose parameters to optimize.
        config: Parsed experiment configuration.

    Returns:
        Configured AdamW optimizer.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    logger.info(
        "Optimizer: AdamW  lr=%.2e  weight_decay=%.2e",
        config.training.learning_rate,
        config.training.weight_decay,
    )
    return optimizer


def build_loss() -> BioJointLoss:
    """
    Construct the bio-constrained joint loss function.

    Returns:
        Configured :class:`BioJointLoss` with mean reduction.
    """
    return BioJointLoss(reduction="mean")


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

    train_dataset = NSMoRDataset(
        sequences=train_sequences,
        mcmc_priors=train_priors,
        feature_config=feature_config,
    )
    val_dataset = NSMoRDataset(
        sequences=val_sequences,
        mcmc_priors=val_priors,
        feature_config=feature_config,
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

def train_one_epoch(
    model: NSMoRCore,
    loader: torch.utils.data.DataLoader,
    criterion: BioJointLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_reg: float = 0.01,
    grad_clip_norm: float = 1.0,
    log_interval: int = 10,
    epoch: int = 0,
) -> float:
    """
    Run one training epoch.

    Args:
        model: The NSMoR model.
        loader: Training DataLoader yielding ``(X_batch, Y_batch, lengths)``.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        lambda_reg: Router regularization weight.
        grad_clip_norm: Max gradient norm for clipping.
        log_interval: Log every N batches.
        epoch: Current epoch number (for logging).

    Returns:
        Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        # Unpack batch — expect (X, Y, lengths) from collate_variable_length
        x_batch, y_batch, lengths = batch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        lengths = lengths.to(device)

        # ── Forward pass (with internals for routing gates) ──
        y_pred, internals = model(x_batch, lengths, return_internals=True)

        # ── Extract g_gru from routing gates ──
        # routing_gates: (B, T, 2) — index 1 is g_gru
        g_gru = internals["routing_gates"][:, :, 1:2]           # (B, T, 1)

        # ── Compute loss ──
        loss = criterion(
            y_pred=y_pred,
            y_true=y_batch,
            lengths=lengths,
            g_gru=g_gru,
            lambda_reg=lambda_reg,
        )

        # ── Backward pass ──
        optimizer.zero_grad()
        loss.backward()

        # ── Gradient clipping ──
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm,
            )

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # ── Logging ──
        if (batch_idx + 1) % log_interval == 0:
            avg_so_far = total_loss / n_batches
            logger.info(
                "  [Epoch %d] batch %d/%d  loss=%.6f",
                epoch + 1, batch_idx + 1, len(loader), avg_so_far,
            )

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
) -> float:
    """
    Run validation (no gradient computation).

    Args:
        model: The NSMoR model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Device.
        lambda_reg: Router regularization weight.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x_batch, y_batch, lengths = batch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        lengths = lengths.to(device)

        y_pred, internals = model(x_batch, lengths, return_internals=True)
        g_gru = internals["routing_gates"][:, :, 1:2]

        loss = criterion(
            y_pred=y_pred,
            y_true=y_batch,
            lengths=lengths,
            g_gru=g_gru,
            lambda_reg=lambda_reg,
        )

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


# ═══════════════════════════════════════════════════════════════
# 6.  Main Train Function
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

    # ── Build components ──────────────────────────────────────
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    criterion = build_loss()
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

    for epoch in range(start_epoch, config.training.num_epochs):
        t0 = time.time()

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
            grad_clip_norm=config.training.grad_clip_norm,
            log_interval=config.training.log_interval,
            epoch=epoch,
        )
        history["train_loss"].append(train_loss)

        # ── Validate ──────────────────────────────────────────
        val_loss = float("inf")
        if val_loader is not None:
            val_loss = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                lambda_reg=lambda_reg,
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
            )
            logger.info("Saved periodic checkpoint: %s", epoch_path)

        # Best-model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best_model.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                config=config.to_dict(),
                path=best_path,
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
    )
    logger.info("Saved final model: %s", final_path)

    logger.info("=" * 60)
    logger.info("Training complete.  Best val loss: %.6f", best_val_loss)
    logger.info("=" * 60)

    return {
        "best_val_loss": best_val_loss,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float("inf"),
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

    results = train(config, lambda_reg=lambda_reg)
    logger.info("Results: %s", results)


if __name__ == "__main__":
    main()
