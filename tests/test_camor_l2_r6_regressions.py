"""Synthetic regressions at CAMoR-L2 startup and evaluation boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from nsmor.nsmor_dataloader import NSMoRDataset
from nsmor.pipeline.latency_metadata import load_latency_target_table
from nsmor.pipeline.nested_prior import load_artifact_bytes
from tests.test_camor_l2_r4_regressions import _payload
from tests.test_camor_l2_r5_regressions import script_settings  # noqa: F401
from tests.test_train_checkpoint import _make_config, _make_synthetic_dataset


@pytest.mark.parametrize(
    "previous,current,declared,expected",
    [("stochastic", "off", None, 1), ("off", "stochastic", None, 5),
     ("off", "stochastic", 3, 3)],
)
def test_mc_settings_belong_to_current_run(
    tmp_path: Path, script_settings: dict[str, Any],
    previous: str, current: str, declared: int | None, expected: int,
) -> None:
    """Programmatic runs resolve their own defaults and preserve CLI overrides."""
    import scripts.train as training

    training.build_config(["--time_consuming_mode", previous])
    config = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    config.model.time_consuming_mode = current
    config.model.time_consuming_max_steps = 3
    if declared is not None:
        parsed, _, _ = training.build_config([
            "--time_consuming_mode", current, "--eval_mc_samples", str(declared),
        ])
        # Use the CLI-created config, retaining its invocation-specific settings.
        parsed.training = config.training
        parsed.model.hidden_dim = config.model.hidden_dim
        parsed.model.dropout = 0.0
        parsed.model.time_consuming_max_steps = 3
        parsed.checkpoint = config.checkpoint
        config = parsed
        training.build_config(["--time_consuming_mode", previous])
    dataset = _make_synthetic_dataset(tmp_path)
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        result = training.train(config, dataset_path=str(dataset))
    checkpoint = load_artifact_bytes(
        (Path(config.checkpoint.output_dir) / "final_model.pth").read_bytes()
    )
    assert checkpoint["eval_mc_samples"] == expected
    assert training.validate.last_eval_mc_samples == expected
    assert len(result["history"]["val_loss"]) == 1


def test_single_draw_checkpoint_and_metrics_provenance(
    tmp_path: Path, script_settings: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A nonzero module seed cannot override the run's declared eval seed."""
    import logging
    import scripts.train as training

    training.build_config([])
    config = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    config.model.time_consuming_mode = "stochastic"
    config.model.time_consuming_max_steps = 3
    config.model.time_consuming_seed = 17
    dataset = _make_synthetic_dataset(tmp_path)
    with caplog.at_level(logging.INFO, logger="scripts.train"):
        with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
            result = training.train(
                config, dataset_path=str(dataset), eval_mc_samples=1,
                eval_seed_base=123,
            )
    checkpoint = load_artifact_bytes(
        (Path(config.checkpoint.output_dir) / "final_model.pth").read_bytes()
    )
    assert checkpoint["eval_seed_base"] == 123
    assert checkpoint["eval_seeds"] == [123]
    assert checkpoint["eval_single_draw"] is True
    assert result["metrics"]["stochastic_evaluation"]["eval_seeds"] == [123]
    assert "single-draw" in caplog.text and "eval_seed=123" in caplog.text


def test_final_metrics_use_declared_single_draw_seed() -> None:
    """The metric report names the actual hard sample rather than implying MC."""
    from nsmor.model_nsmor_core import NSMoRCore
    from scripts.train import compute_metrics

    torch.manual_seed(6)
    model = NSMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
        time_consuming_mode="stochastic", time_consuming_max_steps=3,
        time_consuming_seed=17,
    ).eval()
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(4, 20, 8, generator=generator)
    y = torch.randn(4, 20, generator=generator)
    lengths = torch.full((4,), 20, dtype=torch.long)
    loader = DataLoader(torch.utils.data.TensorDataset(x, y, lengths), batch_size=4)
    with torch.no_grad():
        prediction, internals = model(x, lengths, return_internals=True, eval_seed=0)
        expected_mse = float((prediction - y).double().square()[
            internals["out_valid"]
        ].mean())
    report = compute_metrics(model, loader, torch.device("cpu"))
    assert report["mse"] == pytest.approx(expected_mse, rel=1e-6)
    assert report["stochastic_evaluation"] == {
        "eval_mc_samples": 1, "eval_seed_base": 0, "eval_seeds": [0],
        "single_draw": True,
        "note": "single hard draw; sampling noise remains; not the MC selection mean",
    }


def test_single_draw_validation_honors_explicit_seed() -> None:
    """S=1 must score the requested draw, independent of the module default."""
    from nsmor.loss import BioJointLoss
    from nsmor.model_nsmor_core import NSMoRCore
    from scripts.train import validate

    torch.manual_seed(6)
    model = NSMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=8, dropout=0.0,
        time_consuming_mode="stochastic", time_consuming_max_steps=3,
        time_consuming_seed=17,
    ).eval()
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(4, 20, 8, generator=generator)
    y = torch.randn(4, 20, generator=generator)
    lengths = torch.full((4,), 20, dtype=torch.long)
    loader = DataLoader(torch.utils.data.TensorDataset(x, y, lengths), batch_size=4)
    with torch.no_grad():
        predicted, internals = model(x, lengths, return_internals=True, eval_seed=123)
        valid = internals["out_valid"].clone()
        valid[:, 0] = False  # Validation's aligned t>=1 MSE.
        expected = float(((predicted - y).square())[valid].mean())
    actual = validate(
        model, loader, BioJointLoss(), torch.device("cpu"),
        selection_metric="mse", eval_mc_samples=1, eval_seed_base=123,
    )
    assert actual == pytest.approx(expected, rel=1e-6)
    assert validate.last_eval_seeds == [123]
    assert validate.last_eval_single_draw is True


@pytest.mark.parametrize("case", ["units", "alignment"])
def test_analysis_refuses_unverifiable_target_coordinates(
    tmp_path: Path, case: str,
) -> None:
    """Analysis cannot infer a physical clock or saved anchor from the table."""
    from scripts.analyze_latency import main

    payload = _payload()
    if case == "units":
        payload.pop("target_ms")
        payload.pop("dt_ms")
    table_path = tmp_path / "table.pt"
    torch.save(payload, table_path)
    internals = tmp_path / "internals.pt"
    torch.save({
        "expected_delay": torch.ones(4, 8), "realized_delay": torch.ones(4, 8),
        "lengths": torch.full((4,), 8), "source_indices": torch.arange(4),
        "dt_ms": 4.0, "max_seq_len": 8, "pre_anchor_frames": 0,
    }, internals)
    with pytest.raises(ValueError, match="target_ms|dataset"):
        main([
            "--internals", str(internals), "--latency-target-table", str(table_path),
            "--target-group", "0", "--reference-group", "1", "--bootstrap", "10",
        ])


@pytest.mark.parametrize("anchor", [1200, 1800])
def test_analysis_checks_saved_anchor_against_dataset(
    tmp_path: Path, anchor: int,
) -> None:
    """A crop starting at 600 must report the contrast at saved frame 1200."""
    import json
    from scripts.analyze_latency import main

    payload = _payload()
    payload["anchor_frames"].fill_(anchor)
    table = tmp_path / "table.pt"
    torch.save(payload, table)
    x = torch.zeros(4, 3000, 8)
    x[:, :, 1] = 1.0
    x[:2, :, 0] = 2.0
    dataset = tmp_path / "dataset.pt"
    torch.save({"X_seqs": list(x), "Y_seqs": list(torch.zeros(4, 3000)),
                "anchor_frames": [1800] * 4}, dataset)
    expected = torch.zeros(4, 2400)
    expected[:, 1200] = torch.tensor([9.0, 11.0, 1.0, 3.0])
    expected[:, 1800] = 99.0
    internals = tmp_path / "internals.pt"
    torch.save({
        "expected_delay": expected, "realized_delay": expected,
        "lengths": torch.full((4,), 2400), "source_indices": torch.arange(4),
        "dt_ms": 4.0, "max_seq_len": 2400, "pre_anchor_frames": 1200,
    }, internals)
    output = tmp_path / "report.json"
    args = ["--internals", str(internals), "--latency-target-table", str(table),
            "--dataset", str(dataset), "--target-group", "0",
            "--reference-group", "1", "--bootstrap", "10", "--output", str(output)]
    if anchor != 1200:
        with pytest.raises(ValueError, match="anchor mismatch"):
            main(args)
        assert not output.exists()
    else:
        assert main(args) == 0
        report = json.loads(output.read_text())
        assert report["available"] is True
        assert report["delta_d_frames"] == 8.0


def test_canonical_startup_rejects_condition_mismatch_before_epoch(
    tmp_path: Path, script_settings: dict[str, Any],
) -> None:
    """Canonical loading must reject false groups before any epoch execution."""
    import scripts.train as training

    dataset_path = _make_synthetic_dataset(tmp_path)
    raw = load_artifact_bytes(dataset_path.read_bytes())
    for values in raw["X_seqs"]:
        values[:, :2] = 0.0
        values[:, 1] = 1.0
    raw["anchor_frames"] = np.zeros(len(raw["X_seqs"]), dtype=np.int64)
    torch.save(raw, dataset_path)
    n = len(raw["X_seqs"])
    payload = _payload()
    payload.update({
        "target_frames": torch.ones(n), "target_ms": torch.full((n,), 4.0),
        "mask": torch.ones(n, dtype=torch.bool),
        "group_ids": torch.arange(n) % 2, "prefix_ids": torch.arange(n),
        "anchor_frames": torch.zeros(n, dtype=torch.long),
    })
    payload.pop("pair_ids")
    table = tmp_path / "table.pt"
    torch.save(payload, table)
    config, _, _ = training.build_config([
        "--time_consuming_mode", "stochastic", "--lambda_latency", "1",
        "--latency_target_table", str(table), "--latency_target_group", "0",
        "--latency_reference_group", "1",
    ])
    small = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    config.training, config.checkpoint = small.training, small.checkpoint
    config.model.hidden_dim = 8
    with mock.patch(
        "scripts.train.train_one_epoch", side_effect=AssertionError("epoch entered"),
    ):
        with pytest.raises(ValueError, match="condition mismatch.*source"):
            training.train(config, dataset_path=str(dataset_path))
    assert not (Path(config.checkpoint.output_dir) / "final_model.pth").exists()


def test_startup_refuses_false_condition_coverage(tmp_path: Path) -> None:
    """Table group labels cannot manufacture a contrast from all-wind trials."""
    from scripts.train import _prepare_latency_loaders

    x = np.zeros((8, 8), dtype=np.float32)
    x[:, 1] = 1.0
    # Split order deliberately differs from the frozen table order.
    dataset = NSMoRDataset(
        [(x.copy(), np.zeros(8, dtype=np.float32), 0) for _ in range(4)],
        mcmc_priors=np.full((4, 4), 0.25, dtype=np.float32),
        source_indices=[3, 0, 2, 1], anchor_frames=[0] * 4,
    )
    path = tmp_path / "targets.pt"
    torch.save(_payload(), path)
    table = load_latency_target_table(path)
    loader = DataLoader(dataset, batch_size=4)
    with pytest.raises(ValueError, match="condition.*source.*0"):
        _prepare_latency_loaders(
            loader, None, table, target_group=0, reference_group=1,
            min_prefixes=1, min_trials=1, seed=0, pairing="none",
        )
