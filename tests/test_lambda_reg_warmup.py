"""Tests for lambda_reg warmup exemption and CLI/YAML resolution.

Verifies:
1. build_config reads lambda_reg from YAML when CLI does not override.
2. build_config uses CLI --lambda_reg when explicitly set.
3. The training loop passes unscaled lambda_reg (warmup does NOT scale it).
4. compute_warmup_factor still ramps energy-like terms.
5. train() without lambda_reg falls through to config.loss.lambda_reg,
   while an explicit 0.0 still disables router regularization.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# ═════════════════════════════════════════════════════════════
# Test 1: build_config reads lambda_reg from YAML default (0.2)
# ═════════════════════════════════════════════════════════════

def test_build_config_reads_lambda_reg_from_yaml() -> None:
    """When --lambda_reg is NOT passed on the CLI, build_config must
    return config.loss.lambda_reg (0.2 from YAML), NOT the old
    argparse default of 0.01."""
    from scripts.train import build_config

    config, lambda_reg, _ = build_config(
        ["--config", str(REPO_ROOT / "config" / "default.yaml")]
    )
    assert lambda_reg == pytest.approx(0.2), (
        f"Expected lambda_reg=0.2 from YAML, got {lambda_reg}"
    )
    assert config.loss.lambda_reg == pytest.approx(0.2)


# ═════════════════════════════════════════════════════════════
# Test 2: CLI --lambda_reg overrides YAML
# ═════════════════════════════════════════════════════════════

def test_build_config_cli_overrides_yaml() -> None:
    """Explicit --lambda_reg 0.05 on the CLI must override YAML."""
    from scripts.train import build_config

    config, lambda_reg, _ = build_config(
        ["--config", str(REPO_ROOT / "config" / "default.yaml"),
         "--lambda_reg", "0.05"]
    )
    assert lambda_reg == pytest.approx(0.05), (
        f"Expected lambda_reg=0.05 from CLI, got {lambda_reg}"
    )


# ═════════════════════════════════════════════════════════════
# Test 3: dataclass default is 0.2 when no YAML is loaded
# ═════════════════════════════════════════════════════════════

def test_build_config_dataclass_default() -> None:
    """When no --config is given and no --lambda_reg, the dataclass
    default (0.2) is returned."""
    from scripts.train import build_config

    _, lambda_reg, _ = build_config([])
    assert lambda_reg == pytest.approx(0.2), (
        f"Expected lambda_reg=0.2 from dataclass default, got {lambda_reg}"
    )


# ═════════════════════════════════════════════════════════════
# Test 4: compute_warmup_factor ramps energy-like terms
# ═════════════════════════════════════════════════════════════

def test_warmup_factor_ramps_during_warmup() -> None:
    """compute_warmup_factor must produce values < 1 during warmup
    and == 1.0 at/after the boundary."""
    from scripts.train import compute_warmup_factor

    W = 20
    # During warmup: factor < 1.0
    for epoch in range(W):
        f = compute_warmup_factor(epoch, W)
        assert 0.0 <= f <= 1.0, f"epoch {epoch}: factor={f} out of [0,1]"
        if epoch < W - 1:
            assert f < 1.0, f"epoch {epoch}: factor should be < 1.0 during warmup"

    # Post-warmup: factor == 1.0
    assert compute_warmup_factor(W, W) == 1.0
    assert compute_warmup_factor(W + 10, W) == 1.0

    # Disabled: always 1.0
    assert compute_warmup_factor(0, 0) == 1.0


# ═════════════════════════════════════════════════════════════
# Test 5: lambda_reg is NOT multiplied by warmup_factor in the
#          training loop wiring
# ═════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_run(tmp_path: Path):
    """Minimal 2-epoch training setup: (config, dataset_path).

    ``warmup_epochs=20`` with ``num_epochs=2`` keeps every epoch inside
    warmup, so any warmup scaling of ``lambda_reg`` is observable.
    """
    import numpy as np
    import torch
    from nsmor.config import PIPELINE_SEMANTICS_VERSION, FeatureConfig

    # Build a minimal synthetic dataset
    rng = np.random.RandomState(0)
    n_total = 10
    _X_DIM = 8
    X_seqs = [rng.randn(50, _X_DIM).astype(np.float32) for _ in range(n_total)]
    Y_seqs = [rng.randn(50).astype(np.float32) for _ in range(n_total)]
    labels = np.zeros(n_total, dtype=np.int64)
    lengths = np.full(n_total, 50, dtype=np.int64)
    _raw = rng.randn(n_total, 4).astype(np.float32)
    _exp = np.exp(_raw - _raw.max(axis=1, keepdims=True))
    mcmc_priors = (_exp / _exp.sum(axis=1, keepdims=True)).astype(np.float32)
    session_ids = [f"sess_{i % 5}" for i in range(n_total)]

    ds_path = tmp_path / "test_ds.pt"
    torch.save({
        "X_seqs": X_seqs, "Y_seqs": Y_seqs, "labels": labels,
        "lengths": lengths, "mcmc_priors": mcmc_priors,
        "session_ids": session_ids, "feature_config": FeatureConfig(),
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
    }, ds_path)

    from nsmor.config_parser import ExperimentConfig
    config = ExperimentConfig()
    config.model.hidden_dim = 16
    config.model.dropout = 0.0
    config.training.num_epochs = 2
    config.training.batch_size = 10
    config.training.max_seq_len = 50
    config.training.random_seed = 42
    config.training.checkpoint_interval = 999
    config.loss.warmup_epochs = 20  # warmup active for all 2 epochs
    config.loss.lambda_reg = 0.2
    config.checkpoint.output_dir = str(tmp_path / "run")
    return config, str(ds_path)


def _observed_lambda_regs(config, dataset_path: str, **train_kwargs) -> list:
    """Run ``train`` and return the lambda_reg seen by each epoch."""
    from scripts.train import train, train_one_epoch

    observed = []
    orig = train_one_epoch

    def _spy(*args, **kwargs):
        observed.append(
            kwargs.get("lambda_reg", args[5] if len(args) > 5 else None)
        )
        return orig(*args, **kwargs)

    with mock.patch("scripts.train.train_one_epoch", side_effect=_spy):
        train(config, dataset_path=dataset_path, **train_kwargs)
    return observed


def test_lambda_reg_not_warmup_scaled_in_train_loop(synthetic_run) -> None:
    """Observe the lambda_reg value passed to train_one_epoch during
    warmup: it must be the FULL configured value (0.2), not scaled
    by warmup_factor."""
    config, ds_path = synthetic_run
    observed = _observed_lambda_regs(config, ds_path, lambda_reg=0.2)

    # Both epochs are within warmup (warmup_epochs=20, total=2).
    # lambda_reg must be the full 0.2 for BOTH epochs.
    assert len(observed) == 2
    for i, lr in enumerate(observed):
        assert lr == pytest.approx(0.2), (
            f"Epoch {i}: lambda_reg={lr} was warmup-scaled; expected full 0.2"
        )


# ═════════════════════════════════════════════════════════════
# Test 6: train() without lambda_reg falls through to the config
# ═════════════════════════════════════════════════════════════

def test_train_lambda_reg_defaults_to_config(synthetic_run) -> None:
    """``train(config)`` with no lambda_reg must use
    ``config.loss.lambda_reg``, not a hard-coded signature default.

    Regression guard: the old signature default (0.01) silently
    overrode a YAML/dataclass 0.2 for every programmatic caller.
    """
    config, ds_path = synthetic_run
    config.loss.lambda_reg = 0.2

    observed = _observed_lambda_regs(config, ds_path)

    assert observed, "train_one_epoch was never called"
    for i, lr in enumerate(observed):
        assert lr == pytest.approx(0.2), (
            f"Epoch {i}: lambda_reg={lr}; expected config value 0.2"
        )


def test_train_lambda_reg_zero_is_honoured(synthetic_run) -> None:
    """An explicit 0.0 must disable router regularization rather than
    being treated as 'unset' and replaced by the config value."""
    config, ds_path = synthetic_run
    config.loss.lambda_reg = 0.2

    observed = _observed_lambda_regs(config, ds_path, lambda_reg=0.0)

    assert observed, "train_one_epoch was never called"
    for i, lr in enumerate(observed):
        assert lr == pytest.approx(0.0), (
            f"Epoch {i}: lambda_reg={lr}; explicit 0.0 was overridden"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
