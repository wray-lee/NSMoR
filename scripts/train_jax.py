#!/usr/bin/env python3
"""
NSMoR JAX-Accelerated Training Entrypoint.

Usage:
    python scripts/train_jax.py --config config/default.yaml
    python scripts/train_jax.py --config config/default.yaml --dataset data/processed/nsmor_dataset_3cond_v2.pt --epochs 10
    python scripts/train_jax.py --batch_size 128 --grad_accum_steps 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

# Set XLA memory allocation before importing JAX
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from nsmor.config_parser import ExperimentConfig
from nsmor.jax.train import train_jax


def setup_logging(output_dir: Optional[Path] = None) -> logging.Logger:
    """Configure console and file logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "train_jax.log", mode="w")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logging.getLogger("nsmor.jax")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Train NSMoR with JAX/XLA acceleration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to experiment configuration YAML.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/nsmor_dataset_3cond_v2.pt",
        help="Path to preprocessed PyTorch dataset .pt artifact.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (microbatches per update).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pth or JAX state) to resume from.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI main routine."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config and Path(args.config).exists():
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()

    out_dir = Path(args.output_dir or config.checkpoint.output_dir)
    logger = setup_logging(out_dir)

    logger.info("Starting NSMoR JAX Accelerated Training")
    logger.info("Config file: %s", args.config)
    logger.info("Dataset:     %s", args.dataset)

    results = train_jax(
        config=config,
        dataset_path=args.dataset,
        output_dir=str(out_dir),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        grad_accum_steps=args.grad_accum_steps,
        resume_from=args.resume,
    )

    logger.info("Training successfully finished.")
    logger.info("Best validation loss: %.4f", results["best_val_loss"])
    logger.info("Average epoch duration: %.2fs", results["avg_epoch_time_s"])


if __name__ == "__main__":
    main()
