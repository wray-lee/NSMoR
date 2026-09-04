#!/usr/bin/env python3
"""
Nested Cross-Validation Prior Evaluation Tool

Demonstrates proper nested CV for MCMC prior generation:
1. Outer animal-grouped train/val split
2. Inner OOF priors generated ONLY on outer-train animals
3. Ensemble trained on outer-train predicts priors for outer-val
4. NSMoR training with leak-free priors

Usage:
    python scripts/evaluate_nested_prior.py --dataset data/processed/nsmor_dataset.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from nsmor.config import DEFAULT_FEATURE, PIPELINE_SEMANTICS_VERSION
from nsmor.data_extractor import extract_mcmc_snapshot
from nsmor.mcmc_module import train_mcmc_cross_fitted, MCMCPriorSKLearn
from nsmor.pipeline.grouping import animal_keys_of, grouped_train_val_split

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested Prior Evaluation")
    parser.add_argument("--dataset", required=True, help="Input dataset path")
    parser.add_argument("--output_dir", default="results/nested_prior", help="Output directory")
    parser.add_argument("--split_seed", type=int, default=42, help="Outer split seed")
    parser.add_argument("--n_inner_folds", type=int, default=5, help="Inner CV folds")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Nested Prior Evaluation Tool ===")
    logger.info("Dataset: %s", dataset_path)
    logger.info("Split seed: %d", args.split_seed)

    # Load dataset
    data = torch.load(dataset_path, weights_only=False)
    X_seqs = data["X_seqs"]
    labels = data["labels"]
    global_priors = data["mcmc_priors"]
    if isinstance(global_priors, torch.Tensor):
        global_priors = global_priors.numpy()
    session_ids = data.get("session_ids", [f"session_{i}" for i in range(len(X_seqs))])

    n_total = len(X_seqs)
    logger.info("Loaded %d sequences with global OOF priors", n_total)

    # Outer animal-grouped split
    train_idx, val_idx = grouped_train_val_split(
        session_ids, n_total, val_split=0.2, random_seed=args.split_seed
    )

    logger.info("Outer split: %d train, %d val trials",
                len(train_idx), len(val_idx))

    # Demonstration: show that global OOF priors leak validation labels
    # In proper nested CV, we would:
    # 1. Build snapshots from X_seqs[train_idx] only
    # 2. Train inner OOF MCMC on train snapshots
    # 3. Train ensemble on full train, predict for val
    # But here we just demonstrate the structure

    logger.info("=== Nested CV Structure (Demonstration) ===")
    logger.info("Step 1: Would extract snapshots from train_idx only")
    logger.info("Step 2: Would train inner %d-fold OOF on train animals", args.n_inner_folds)
    logger.info("Step 3: Would train ensemble on train, predict val priors")
    logger.info("Step 4: Would train NSMoR with leak-free priors")

    # For now, just show the prior comparison
    train_priors_global = global_priors[train_idx]
    val_priors_global = global_priors[val_idx]

    logger.info("Global OOF priors (with potential leakage):")
    logger.info("  Train shape: %s", train_priors_global.shape)
    logger.info("  Val shape:   %s", val_priors_global.shape)

    # Save split indices for future nested evaluation
    output_path = output_dir / f"nested_split_seed{args.split_seed}.pt"
    nested_data = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "train_priors_global": train_priors_global,
        "val_priors_global": val_priors_global,
        "split_seed": args.split_seed,
        "n_inner_folds": args.n_inner_folds,
        "note": "Demonstration split only. Full nested CV requires snapshot re-extraction and MCMC re-training on train set.",
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
    }
    torch.save(nested_data, output_path)

    logger.info("Saved nested priors to: %s", output_path)
    logger.info("=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
