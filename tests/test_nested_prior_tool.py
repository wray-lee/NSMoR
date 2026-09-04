"""Test nested prior evaluation tool."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION, FeatureConfig


def test_evaluate_nested_prior_cli_help():
    """Tool must have CLI with --dataset."""
    res = subprocess.run(
        [sys.executable, "scripts/evaluate_nested_prior.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "--dataset" in res.stdout


def test_nested_prior_tool_runs_on_synthetic(tmp_path: Path):
    """Tool must run and produce nested priors without val-label leakage."""
    # Create synthetic dataset with enough animals per class for 5-fold CV
    n_trials = 60  # 60 trials across 12 animals
    n_frames = 100
    feature_config = FeatureConfig()

    X_seqs = []
    for i in range(n_trials):
        X = np.random.randn(n_frames, 8).astype(np.float32)
        X_seqs.append(X)

    # Ensure each class has at least 6 animals (for 5-fold CV)
    labels = np.array([i % 4 for i in range(n_trials)], dtype=np.int64)
    mcmc_priors = np.random.dirichlet([1, 1, 1, 1], n_trials).astype(np.float32)
    # 12 animals, each with 5 trials
    session_ids = [f"0.{500 + (i // 5)}cricket_{i // 5:03d}" for i in range(n_trials)]

    ds_path = tmp_path / "synthetic.pt"
    torch.save({
        "X_seqs": X_seqs,
        "labels": labels,
        "mcmc_priors": mcmc_priors,
        "session_ids": session_ids,
        "feature_config": feature_config,
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
    }, ds_path)

    # Run tool
    output_dir = tmp_path / "nested_output"
    res = subprocess.run(
        [
            sys.executable, "scripts/evaluate_nested_prior.py",
            "--dataset", str(ds_path),
            "--output_dir", str(output_dir),
            "--split_seed", "123",
        ],
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, f"Tool failed: {res.stderr}"
    assert "Nested CV Structure" in res.stderr  # Logging goes to stderr

    # Verify output
    output_file = output_dir / "nested_split_seed123.pt"
    assert output_file.exists()

    nested = torch.load(output_file, weights_only=False)
    assert "train_indices" in nested
    assert "val_indices" in nested
    assert "train_priors_global" in nested
    assert "val_priors_global" in nested

    # Verify no overlap
    train_set = set(nested["train_indices"])
    val_set = set(nested["val_indices"])
    assert len(train_set & val_set) == 0, "Train/val overlap!"
