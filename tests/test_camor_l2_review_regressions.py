"""Synthetic regressions for the CAMoR-L2 round-two review findings."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nsmor.loss import BioJointLoss
from nsmor.model_nsmor_core import TimeConsumingRecursion
from nsmor.pipeline.latency_metadata import (
    LatencyContrastBatchSampler,
    LatencyTargetTable,
)
from scripts import train as training


def _table(n: int = 16) -> LatencyTargetTable:
    """Return a target table with two prefixes in each contrast group."""
    return LatencyTargetTable(
        target_frames=torch.arange(n).float() % 4,
        target_mask=torch.ones(n, dtype=torch.bool),
        group_ids=torch.arange(n) % 2,
        prefix_ids=torch.arange(n) % 4,
        anchor_frames=torch.zeros(n, dtype=torch.long),
        pair_ids=(torch.arange(n) % 4) // 2,
        condition_mapping={"multisensory": 0, "wind_only": 1},
    )


@pytest.mark.parametrize("mode", ["stochastic", "hard_depth"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_nonfinite_proxy_backward_is_zero(mode: str, checkpointing: bool) -> None:
    """An unused infinite iterate must not poison any backward tensor."""
    module = TimeConsumingRecursion(
        8, mode=mode, max_steps=2, delay_scale=0.1,
        grad_checkpointing=checkpointing,
    )
    with torch.no_grad():
        module.halt.bias.fill_(50.0)
        module.halt.weight.fill_(1e-6)
    original = module._step_block
    calls = 0

    def injected(u: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        out = original(u)
        return torch.full_like(out, float("inf")) if calls == 2 else out

    module._step_block = injected
    h = torch.randn(1, 3, 8, requires_grad=True)
    outputs = module(h, torch.ones(1, 3, dtype=torch.bool))
    assert torch.isfinite(outputs[0]).all()
    assert module.last_nonfinite_expected == 3
    outputs[0].sum().backward()
    assert h.grad is not None and torch.count_nonzero(h.grad) == 0
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("mode", ["stochastic", "hard_depth"])
def test_nonfinite_rows_are_isolated_from_healthy_rows(mode: str) -> None:
    """Bad rows have zero gradient; healthy rows match a clean-only run."""
    torch.manual_seed(5)
    module = TimeConsumingRecursion(
        8, mode=mode, max_steps=2, delay_scale=0.1,
    )
    with torch.no_grad():
        module.halt.bias.fill_(2.0)
        module.halt.weight.fill_(1e-6)
    clean = copy.deepcopy(module)
    h = torch.randn(1, 2, 8, requires_grad=True)
    clean_h = h[:, 1:].detach().clone().requires_grad_()
    original = module._step_block
    calls = 0

    def injected(u: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        out = original(u)
        if calls == 2:
            out = out.clone()
            out[0] = float("inf")
        return out

    def high_uniform(
        b: int, t: int, device: torch.device, seed: int | None,
    ) -> torch.Tensor:
        return torch.full((b, t), 0.99, dtype=torch.float64, device=device)

    module._step_block = injected
    module._draw_uniform = clean._draw_uniform = high_uniform
    valid = torch.ones(1, 2, dtype=torch.bool)
    outputs = module(h, valid, return_distribution_internals=True)
    reference = clean(
        clean_h, valid[:, 1:], return_distribution_internals=True,
    )
    # Include BOTH value loss and latency supervision in the backward probe.
    (outputs[0].sum() + outputs[-1].sum()).backward()
    (reference[0].sum() + reference[-1].sum()).backward()
    assert torch.isfinite(outputs[0]).all()
    assert module.last_nonfinite_expected == 1
    assert h.grad is not None and clean_h.grad is not None
    assert torch.count_nonzero(h.grad[:, :1]) == 0
    assert torch.allclose(h.grad[:, 1:], clean_h.grad, atol=1e-6, rtol=1e-5)
    for (_, actual), (_, expected) in zip(
        module.named_parameters(), clean.named_parameters(),
    ):
        assert actual.grad is not None and expected.grad is not None
        assert torch.isfinite(actual.grad).all()
        assert torch.allclose(actual.grad, expected.grad, atol=1e-6, rtol=1e-5)


def test_hard_depth_halt_gradient_matches_stochastic() -> None:
    """Matched A/B arms both learn halt probabilities from the task loss."""
    torch.manual_seed(2)
    hard = TimeConsumingRecursion(8, mode="hard_depth", max_steps=3)
    stochastic = copy.deepcopy(hard)
    stochastic.mode = "stochastic"
    h = torch.randn(2, 5, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)
    for module in (hard, stochastic):
        module(h, valid)[0].sum().backward()
        assert module.halt.weight.grad is not None
        assert torch.isfinite(module.halt.weight.grad).all()
        assert module.halt.weight.grad.norm() > 0


@pytest.mark.parametrize("size", [12, 16])
def test_imbalanced_sampler_covers_every_trial_and_prefix(size: int) -> None:
    """Rare prefixes and non-contrast trials must survive every epoch."""
    groups = [0] * 101 + [1] * 101 + [2] * 17
    prefixes = [10] * 100 + [11] + [20] * 100 + [21] + [30] * 17
    sampler = LatencyContrastBatchSampler(
        groups, prefixes, size, target_group=0, reference_group=1, seed=0,
    )
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert set(i for batch in batches for i in batch) == set(range(len(groups)))
    for batch in batches:
        assert len(batch) == size
        assert len(batch) == len(set(batch))
        for group in (0, 1):
            indices = [i for i in batch if groups[i] == group]
            assert len(indices) >= 4
            assert len({prefixes[i] for i in indices}) >= 2
    assert batches == list(sampler)
    sampler.set_epoch(1)
    assert batches != list(sampler)


def test_loader_rebuild_preserves_options_and_validation_sidecar() -> None:
    """Exercise the same reconstruction seam used by train() startup."""
    dataset = TensorDataset(
        torch.randn(16, 5, 8), torch.zeros(16, 5), torch.full((16,), 5),
    )
    dataset.source_indices = list(range(16))
    dataset.tensors[0][:, :, :2] = 0.0
    dataset.tensors[0][:, :, 1] = 1.0
    dataset.tensors[0][::2, :, 0] = 2.0
    # Explicit deterministic crop metadata is required for latency alignment.
    dataset.sequences = [
        (dataset.tensors[0][i], dataset.tensors[1][i], 0) for i in range(16)
    ]
    dataset.anchor_frames = [0] * 16
    dataset.max_seq_len = None
    dataset.pre_anchor_frames = 0
    generator = torch.Generator().manual_seed(4)
    loader = DataLoader(
        dataset, batch_size=12, shuffle=True, generator=generator,
        pin_memory=False, num_workers=0,
    )
    table = _table()
    train, val, coverage = training._prepare_latency_loaders(
        loader, loader, table, target_group=0, reference_group=1,
        min_prefixes=2, min_trials=4, seed=0, pairing="none",
    )
    assert loader.dataset is dataset
    assert train.dataset.dataset is dataset
    assert train.generator is val.generator is generator
    assert train.num_workers == loader.num_workers
    assert val.batch_sampler is loader.batch_sampler
    assert coverage["target_prefixes"] == 2
    assert len(next(iter(train))) == len(next(iter(val))) == 5
    assert "latency_pair_id" not in next(iter(train))[-1]


def test_pairing_declaration_controls_estimator() -> None:
    """Pair ids in a table do not opt an unpaired arm into paired loss."""
    meta = {
        "latency_pair_id": torch.tensor([0, 0, 1, 1]),
        "latency_anchor_frames": torch.zeros(4, dtype=torch.long),
        "latency_target_frames": torch.zeros(4),
        "latency_target_mask": torch.ones(4, dtype=torch.bool),
        "latency_group_id": torch.tensor([0, 1, 0, 1]),
        "latency_prefix_id": torch.arange(4),
    }
    expected = torch.tensor([[6.0] * 3, [0.0] * 3, [0.0] * 3, [6.0] * 3])
    expected.requires_grad_()
    args = (torch.zeros(4, 3), torch.zeros(4, 3), torch.full((4,), 3),
            torch.zeros(4, 3, 1))
    values = []
    for pairing in ("none", "pair_id"):
        kwargs = training._latency_loss_kwargs(
            {"time_consuming_expected_delay": expected}, meta, 1.0, 0, 1,
            pairing=pairing,
        )
        values.append(float(BioJointLoss()(
            *args, lambda_reg=0.0, lambda_latency=1.0, **kwargs,
        ).detach()))
    assert values == [0.0, 36.0]
    with pytest.raises(ValueError, match="pair_id"):
        training._latency_loss_kwargs(
            {"time_consuming_expected_delay": expected}, {}, 1.0, 0, 1,
            pairing="pair_id",
        )


def test_mc_validation_includes_latency_term(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five-tuple metadata reaches every MC scoring call and total selection."""
    from nsmor.model_nsmor_core import NSMoRCore
    from nsmor.pipeline.latency_metadata import (
        IndexedDataset, build_latency_sidecar, collate_with_latency_metadata,
    )

    table = _table(4)
    dataset = TensorDataset(
        torch.randn(4, 6, 8), torch.zeros(4, 6), torch.full((4,), 6),
    )
    loader = DataLoader(
        IndexedDataset(dataset, build_latency_sidecar(table, range(4))),
        batch_size=4, collate_fn=collate_with_latency_metadata,
    )
    model = NSMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
        time_consuming_mode="stochastic", time_consuming_max_steps=3,
    )
    criterion = BioJointLoss()
    seen: list[dict[str, Any]] = []
    original = criterion.forward

    def capture(*args: Any, **kwargs: Any) -> torch.Tensor:
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(criterion, "forward", capture)
    common = dict(
        model=model, loader=loader, criterion=criterion,
        device=torch.device("cpu"), lambda_reg=0.0, eval_mc_samples=3,
        eval_seed_base=10, latency_target_group=0, latency_reference_group=1,
    )
    plain = training.validate(**common, lambda_latency=0.0)
    active = training.validate(**common, lambda_latency=0.5)
    assert len(seen) == 6
    assert all(k["lambda_latency"] == 0.5 for k in seen[3:])
    assert all("latency_target_frames" in k for k in seen[3:])
    assert active > plain
    assert training.validate.last_latency_skipped_batches == 0


def test_latency_train_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete train(), validation, checkpoint reload and metrics on fixtures."""
    from nsmor.config import PIPELINE_SEMANTICS_VERSION
    from nsmor.pipeline.grouping import grouped_train_val_split

    n, timesteps = 32, 12
    generator = torch.Generator().manual_seed(8)
    sessions = [f"fixture{i}_session_1" for i in range(n)]
    train_indices, val_indices = grouped_train_val_split(
        sessions, n, val_split=0.25, random_seed=42,
    )
    groups = torch.zeros(n, dtype=torch.long)
    for indices in (train_indices, val_indices):
        groups[torch.as_tensor(indices)] = torch.arange(len(indices)) % 2
    dataset_path = tmp_path / "synthetic.pt"
    physical = torch.randn(n, timesteps, 8, generator=generator)
    physical[:, :, :2] = 0.0
    physical[:, :, 1] = 1.0
    physical[groups == 0, :, 0] = 2.0
    torch.save({
        "X_seqs": [values.numpy() for values in physical],
        "Y_seqs": [torch.randn(timesteps, generator=generator).numpy()
                   for _ in range(n)],
        "mcmc_priors": np.full((n, 4), 0.25, dtype=np.float32),
        "labels": np.zeros(n, dtype=np.int64),
        "lengths": np.full(n, timesteps, dtype=np.int64),
        "anchor_frames": np.zeros(n, dtype=np.int64),
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_5fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified", "session_ids": sessions,
    }, dataset_path)
    table_path = tmp_path / "latency.pt"
    torch.save({
        "target_frames": groups.float() + 1.0,
        "mask": torch.ones(n, dtype=torch.bool),
        "group_ids": groups, "prefix_ids": torch.arange(n),
        "anchor_frames": torch.zeros(n, dtype=torch.long),
        "target_ms": (groups.float() + 1.0) * 4.0,
        "dt_ms": 4.0,
        "condition_mapping": {"multisensory": 0, "wind_only": 1},
        "recipe": "synthetic", "anchor_recipe": "fixture frame zero",
    }, table_path)
    for name, value in {
        "_VAL_SPLIT": 0.25, "_SWEEP_BANDS": None,
    }.items():
        monkeypatch.setattr(training, name, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Plot rendering is unrelated to this data/loss integration regression.
    monkeypatch.setattr(training, "plot_loss_curve", lambda *a, **k: None)
    cfg, _, _ = training.build_config([
        "--time_consuming_mode", "stochastic", "--lambda_latency", "0.5",
        "--latency_target_table", str(table_path),
        "--latency_target_group", "0", "--latency_reference_group", "1",
    ])
    cfg.model.hidden_dim = 8
    cfg.model.dropout = 0.0
    cfg.model.time_consuming_mode = "stochastic"
    cfg.model.time_consuming_max_steps = 3
    cfg.loss.lambda_latency = 0.5
    cfg.training.batch_size = 12
    cfg.training.num_workers = 0
    cfg.training.num_epochs = 1
    cfg.training.checkpoint_interval = 1
    cfg.training.random_seed = 42
    cfg.training.lr_warmup_epochs = 0
    cfg.checkpoint.output_dir = str(tmp_path / "run")
    result = training.train(
        cfg, phase1_epochs=0, dataset_path=str(dataset_path), eval_mc_samples=3,
    )
    assert len(result["history"]["train_loss"]) == 1
    assert np.isfinite(result["history"]["val_loss"]).all()
    assert training.validate.last_latency_skipped_batches == 0
    assert (tmp_path / "run" / "best_model.pth").exists()
    # Reuse the real train() startup path: an undersized budget must refuse
    # before train_one_epoch is entered, without touching any real artifact.
    cfg.checkpoint.output_dir = str(tmp_path / "denied")
    cfg._time_consuming_memory_budget_mb = 0.001

    def forbidden_epoch(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Memory refusal must precede the first training batch")

    monkeypatch.setattr(training, "train_one_epoch", forbidden_epoch)
    with pytest.raises(ValueError, match="budget"):
        training.train(cfg, phase1_epochs=0, dataset_path=str(dataset_path))


def test_memory_budget_guard_fails_before_iteration() -> None:
    """Representative K=16 activation estimates must respect the budget."""
    from scripts.benchmark_time_consuming import require_memory_budget

    with pytest.raises(ValueError, match="budget"):
        require_memory_budget(
            batch=64, timesteps=1500, hidden_dim=64, K=16,
            grad_checkpointing=False, budget_mb=1.0,
        )
    plain = require_memory_budget(
        batch=64, timesteps=1500, hidden_dim=64, K=16,
        grad_checkpointing=False, budget_mb=8192.0,
    )
    checkpointed = require_memory_budget(
        batch=64, timesteps=1500, hidden_dim=64, K=16,
        grad_checkpointing=True, budget_mb=8192.0,
    )
    assert plain["estimated_bytes"] > checkpointed["estimated_bytes"] > 0
