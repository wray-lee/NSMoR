"""Synthetic regressions for failed CLI setup leaking into default recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch

from nsmor.pipeline.nested_prior import load_artifact_bytes
from tests.test_camor_l2_r4_regressions import _payload
from tests.test_train_checkpoint import _make_config, _make_synthetic_dataset


@pytest.fixture
def script_settings(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Restore script settings after observing the public CLI/run boundary."""
    import scripts.train as training

    names = (
        "_ZERO_INPUT_CHANNELS", "_SELECTION_METRIC", "_RESUME_ID", "_SWEEP_BANDS",
    )
    settings = {name: getattr(training, name) for name in names}
    for name, value in settings.items():
        monkeypatch.setattr(training, name, value)
    return settings


def test_failed_cli_keeps_default_run_settings(
    tmp_path: Path, script_settings: dict[str, Any],
) -> None:
    """A rejected stochastic clock must not change a later off checkpoint."""
    import scripts.train as training

    training.build_config([])
    payload = _payload()
    payload["dt_ms"] = 5.0
    payload["target_ms"] = payload["target_frames"] * 5.0
    table_path = tmp_path / "wrong_clock.pt"
    torch.save(payload, table_path)
    with pytest.raises(ValueError, match="dt_ms"):
        training.build_config([
            "--time_consuming_mode", "stochastic", "--lambda_latency", "1",
            "--latency_target_table", str(table_path),
            "--latency_target_group", "0", "--latency_reference_group", "1",
        ])

    config = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    dataset = _make_synthetic_dataset(tmp_path)
    # The task forbids parameter updates; exercise actual validation and writes
    # while replacing the epoch executor, as in the recovery regression suite.
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        result = training.train(config, dataset_path=str(dataset))
    assert len(result["history"]["val_loss"]) == 1
    checkpoint = load_artifact_bytes(
        (Path(config.checkpoint.output_dir) / "final_model.pth").read_bytes()
    )
    assert checkpoint["time_consuming_mode"] == "off"
    assert checkpoint["lambda_latency"] == 0.0
    assert checkpoint["eval_mc_samples"] == 1
    assert checkpoint["latency_target_table_sha256"] is None
