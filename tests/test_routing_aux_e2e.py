"""End-to-end smoke test for routing auxiliary loss (Ticket #18, Spec #12).

Verifies:
1. 2-epoch training run completes without NaN/Inf.
2. When lambda_routing_aux > 0, loss decreases (sanity check: epoch 2 < epoch 1).
3. Training log records the lambda_routing_aux value (lambda_routing_aux=0.XXX).
4. Graceful handling when lambda_routing_aux is disabled (0.0) or when dataset
   lacks pure-wind trials / condition metadata.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION, FeatureConfig
from nsmor.config_parser import ExperimentConfig
from scripts.train import train, logger as train_logger


@pytest.fixture
def synthetic_routing_aux_data(tmp_path: Path) -> Tuple[ExperimentConfig, str]:
    """Create minimal synthetic dataset with 10 trials, 50 frames each,
    half pure-wind and half visual, and a matching ExperimentConfig.
    """
    rng = np.random.RandomState(42)
    n_total = 10
    n_frames = 50
    _X_DIM = 8  # 4 physical + 4 MCMC prior slots

    X_seqs = [rng.randn(n_frames, _X_DIM).astype(np.float32) for _ in range(n_total)]
    Y_seqs = [rng.randn(n_frames).astype(np.float32) for _ in range(n_total)]
    labels = np.zeros(n_total, dtype=np.int64)
    lengths = np.full(n_total, n_frames, dtype=np.int64)

    # Normalized MCMC priors
    _raw = rng.randn(n_total, 4).astype(np.float32)
    _exp = np.exp(_raw - _raw.max(axis=1, keepdims=True))
    mcmc_priors = (_exp / _exp.sum(axis=1, keepdims=True)).astype(np.float32)

    # 5 sessions: 2 trials per session ensures both train & val get wind and visual
    session_ids = [f"sess_{i % 5}" for i in range(n_total)]

    # Half pure-wind (True) / half visual (False)
    is_pure_wind = np.array([True, False] * 5, dtype=bool)

    ds_path = tmp_path / "synthetic_routing_aux_ds.pt"
    torch.save(
        {
            "X_seqs": X_seqs,
            "Y_seqs": Y_seqs,
            "labels": labels,
            "lengths": lengths,
            "mcmc_priors": mcmc_priors,
            "session_ids": session_ids,
            "feature_config": FeatureConfig(),
            "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
            "is_pure_wind": is_pure_wind,
        },
        ds_path,
    )

    config = ExperimentConfig()
    config.model.hidden_dim = 16
    config.model.dropout = 0.0
    config.training.num_epochs = 2
    config.training.batch_size = 10
    config.training.max_seq_len = n_frames
    config.training.random_seed = 42
    config.training.checkpoint_interval = 999
    config.loss.warmup_epochs = 0
    config.loss.lambda_reg = 0.2
    config.loss.lambda_routing_aux = 0.05
    config.checkpoint.output_dir = str(tmp_path / "run")

    return config, str(ds_path)


def test_routing_aux_e2e_smoke(
    synthetic_routing_aux_data: Tuple[ExperimentConfig, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End-to-end smoke test for 2-epoch training with auxiliary routing loss."""
    caplog.set_level(logging.INFO)

    # Also attach custom handler to train_logger to guarantee record capture
    captured_messages: list[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_messages.append(record.getMessage())

    handler = _CaptureHandler()
    train_logger.addHandler(handler)
    try:
        config, ds_path = synthetic_routing_aux_data
        results = train(config, dataset_path=ds_path)
    finally:
        train_logger.removeHandler(handler)

    # 1. 2-epoch training completed without NaN/Inf
    best_val_loss = results["best_val_loss"]
    assert math.isfinite(best_val_loss), f"best_val_loss is not finite: {best_val_loss}"
    assert not math.isnan(best_val_loss), "best_val_loss is NaN"
    assert not math.isinf(best_val_loss), "best_val_loss is Inf"

    final_train_loss = results["final_train_loss"]
    assert math.isfinite(final_train_loss), f"final_train_loss is not finite: {final_train_loss}"
    assert not math.isnan(final_train_loss), "final_train_loss is NaN"

    # 2. lambda_routing_aux > 0 loss decreases (sanity check)
    history = results.get("history")
    assert history is not None, "train() results missing history"
    train_losses = history["train_loss"]
    assert len(train_losses) == 2, f"Expected 2 epoch losses, got {len(train_losses)}"
    assert train_losses[1] < train_losses[0], (
        f"Expected Epoch 2 train_loss ({train_losses[1]:.6f}) < "
        f"Epoch 1 train_loss ({train_losses[0]:.6f})"
    )

    # 3. Training log records lambda_routing_aux value (lambda_routing_aux=0.XXX)
    all_logs = " ".join([caplog.text] + captured_messages)
    pattern = r"lambda_routing_aux=0\.\d+"
    match = re.search(pattern, all_logs)
    assert match is not None, (
        f"Expected log to contain 'lambda_routing_aux=0.XXX', but found:\n{all_logs}"
    )


def test_routing_aux_zero_disables_gracefully(
    synthetic_routing_aux_data: Tuple[ExperimentConfig, str],
) -> None:
    """When lambda_routing_aux=0.0, training succeeds and produces finite loss."""
    config, ds_path = synthetic_routing_aux_data
    config.loss.lambda_routing_aux = 0.0
    config.checkpoint.output_dir = str(Path(config.checkpoint.output_dir) / "zero_aux")

    results = train(config, dataset_path=ds_path)
    assert math.isfinite(results["best_val_loss"])
    assert math.isfinite(results["final_train_loss"])


def test_routing_aux_all_visual_trials_graceful(
    tmp_path: Path,
) -> None:
    """When dataset has no pure-wind trials (visual-only corpus), auxiliary loss
    degrades gracefully to zero and does not NaN or crash (User Story 12).
    """
    rng = np.random.RandomState(42)
    n_total = 10
    n_frames = 50
    _X_DIM = 8

    X_seqs = [rng.randn(n_frames, _X_DIM).astype(np.float32) for _ in range(n_total)]
    Y_seqs = [rng.randn(n_frames).astype(np.float32) for _ in range(n_total)]
    labels = np.zeros(n_total, dtype=np.int64)
    lengths = np.full(n_total, n_frames, dtype=np.int64)
    _raw = rng.randn(n_total, 4).astype(np.float32)
    _exp = np.exp(_raw - _raw.max(axis=1, keepdims=True))
    mcmc_priors = (_exp / _exp.sum(axis=1, keepdims=True)).astype(np.float32)
    session_ids = [f"sess_{i % 5}" for i in range(n_total)]
    # All visual (no wind trials)
    is_pure_wind = np.zeros(n_total, dtype=bool)

    ds_path = tmp_path / "all_visual_ds.pt"
    torch.save(
        {
            "X_seqs": X_seqs,
            "Y_seqs": Y_seqs,
            "labels": labels,
            "lengths": lengths,
            "mcmc_priors": mcmc_priors,
            "session_ids": session_ids,
            "feature_config": FeatureConfig(),
            "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
            "is_pure_wind": is_pure_wind,
        },
        ds_path,
    )

    config = ExperimentConfig()
    config.model.hidden_dim = 16
    config.model.dropout = 0.0
    config.training.num_epochs = 2
    config.training.batch_size = 10
    config.training.max_seq_len = n_frames
    config.training.random_seed = 42
    config.training.checkpoint_interval = 999
    config.loss.warmup_epochs = 0
    config.loss.lambda_reg = 0.2
    config.loss.lambda_routing_aux = 0.05
    config.checkpoint.output_dir = str(tmp_path / "run_all_visual")

    results = train(config, dataset_path=str(ds_path))
    assert math.isfinite(results["best_val_loss"])
    assert math.isfinite(results["final_train_loss"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
