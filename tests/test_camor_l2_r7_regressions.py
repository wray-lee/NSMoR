"""Synthetic invocation-isolation and analysis regressions for CAMoR-L2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION
from nsmor.pipeline.grouping import grouped_train_val_split
from nsmor.pipeline.nested_prior import load_artifact_bytes
from tests.test_camor_l2_r4_regressions import _payload
from tests.test_camor_l2_r5_regressions import script_settings  # noqa: F401


def _corpus(tmp_path: Path) -> tuple[Path, torch.Tensor]:
    """Make a corpus with both physical conditions in each grouped split."""
    n, timesteps = 32, 12
    sessions = [f"fixture{i}_session_1" for i in range(n)]
    groups = torch.zeros(n, dtype=torch.long)
    for indices in grouped_train_val_split(
        sessions, n, val_split=0.2, random_seed=42,
    ):
        groups[torch.as_tensor(indices)] = torch.arange(len(indices)) % 2
    x = torch.zeros(n, timesteps, 8)
    x[:, :, 1] = 1.0
    x[groups == 0, :, 0] = 2.0
    path = tmp_path / "synthetic.pt"
    torch.save({
        "X_seqs": [values.numpy() for values in x],
        "Y_seqs": [np.zeros(timesteps, dtype=np.float32) for _ in range(n)],
        "mcmc_priors": np.full((n, 4), 0.25, dtype=np.float32),
        "labels": np.zeros(n, dtype=np.int64),
        "lengths": np.full(n, timesteps, dtype=np.int64),
        "anchor_frames": np.zeros(n, dtype=np.int64),
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_5fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified", "session_ids": sessions,
    }, path)
    return path, groups


def _config(
    tmp_path: Path,
    groups: torch.Tensor,
    name: str,
    zero_input_channels: int | None = None,
) -> Any:
    """Resolve independently declared supervision through the public CLI.

    ``zero_input_channels`` is declared in the same invocation that creates the
    returned config, so the ablation is bound to the config itself rather than
    to mutable module state that a later ``build_config`` call would replace.
    """
    from scripts.train import build_config

    payload = _payload()
    frames = torch.where(groups == 0, 6.0 if name == "a" else 21.0, 1.0)
    payload.update({
        "target_frames": frames, "target_ms": frames * 4.0,
        "mask": torch.ones(len(groups), dtype=torch.bool),
        "group_ids": groups if name == "a" else torch.where(groups == 0, 7, 9),
        "prefix_ids": torch.arange(len(groups)),
        "anchor_frames": torch.zeros(len(groups), dtype=torch.long),
        "pair_ids": torch.arange(len(groups)) // 2,
        "recipe": f"synthetic target {name}", "anchor_recipe": f"anchor {name}",
        "condition_mapping": (
            {"multisensory": 0, "wind_only": 1} if name == "a"
            else {"multisensory": 7, "wind_only": 9}
        ),
    })
    path = tmp_path / f"{name}.pt"
    torch.save(payload, path)
    argv = [
        "--time_consuming_mode", "stochastic", "--lambda_latency", "1",
        "--latency_target_table", str(path), "--latency_target_group",
        "0" if name == "a" else "9", "--latency_reference_group",
        "1" if name == "a" else "7", "--latency_pairing",
        "none" if name == "a" else "pair_id",
        "--latency_min_prefixes_per_group", "1" if name == "a" else "2",
        "--latency_min_trials_per_group", "2" if name == "a" else "4",
    ]
    if zero_input_channels is not None:
        argv += ["--zero_input_channels", str(zero_input_channels)]
    config, _, _ = build_config(argv)
    config.model.hidden_dim = 8
    config.model.dropout = 0.0
    config.model.time_consuming_max_steps = 3
    config.training.num_workers = 0
    config.training.num_epochs = 1
    config.training.batch_size = 12
    config.training.max_seq_len = 12
    config.training.lr_warmup_epochs = 0
    config.checkpoint.output_dir = str(tmp_path / f"run-{name}")
    return config


def test_latency_settings_follow_config_through_checkpoint(
    tmp_path: Path, script_settings: dict[str, Any],
) -> None:
    """Parsing B cannot replace A's targets, contrast, pairing or sampler."""
    import scripts.train as training

    corpus, groups = _corpus(tmp_path)
    config = _config(tmp_path, groups, "a")
    schema = config.to_dict()
    # No parameter updates: exercise real startup, validation and checkpoint I/O.
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        control = training.train(config, dataset_path=str(corpus))
    _config(tmp_path, groups, "b")
    config.checkpoint.output_dir = str(tmp_path / "run-a-after-b")
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})) as epoch:
        actual = training.train(config, dataset_path=str(corpus))
    checkpoint = load_artifact_bytes(
        (Path(config.checkpoint.output_dir) / "final_model.pth").read_bytes()
    )
    assert checkpoint["latency_target_table_sha256"] == hashlib.sha256(
        (tmp_path / "a.pt").read_bytes()
    ).hexdigest()
    assert checkpoint["latency_target_recipe"] == "synthetic target a"
    assert checkpoint["latency_anchor_recipe"] == "anchor a"
    assert checkpoint["latency_condition_mapping"] == {
        "multisensory": 0, "wind_only": 1,
    }
    assert checkpoint["latency_target_group"] == 0
    assert checkpoint["latency_reference_group"] == 1
    assert checkpoint["latency_pairing"] == "none"
    assert checkpoint["latency_min_prefixes_per_group"] == 1
    assert checkpoint["latency_min_trials_per_group"] == 2
    assert actual["history"]["val_loss"] == control["history"]["val_loss"]
    loader = epoch.call_args.kwargs["loader"]
    assert set(loader.dataset.sidecar["latency_target_frames"].tolist()) == {1, 6}
    assert "latency_pair_id" not in loader.dataset.sidecar
    # Invocation metadata must not become config/hash leaves.
    config.checkpoint.output_dir = schema["checkpoint"]["output_dir"]
    assert config.to_dict() == schema


@pytest.mark.parametrize("mode", ["off", "depth_delay", "accumulate"])
def test_checkpointing_follows_its_own_config(
    mode: str, script_settings: dict[str, Any],
) -> None:
    """A checkpointed stochastic parse must not affect an existing model."""
    from scripts.train import build_config, build_model

    config, _, _ = build_config(["--time_consuming_mode", mode])
    config.model.hidden_dim = 8
    torch.manual_seed(41)
    control = build_model(config)
    control_rng = torch.get_rng_state()
    other, _, _ = build_config([
        "--time_consuming_mode", "stochastic",
        "--time_consuming_grad_checkpointing",
    ])
    torch.manual_seed(41)
    actual = build_model(config)
    assert torch.equal(torch.get_rng_state(), control_rng)
    assert control.state_dict().keys() == actual.state_dict().keys()
    assert all(torch.equal(value, actual.state_dict()[key])
               for key, value in control.state_dict().items())
    build_config([])
    assert build_model(other).backend.time_consuming.grad_checkpointing is True


def test_memory_budget_follows_config_at_startup(
    tmp_path: Path, script_settings: dict[str, Any],
) -> None:
    """A later permissive budget cannot override A's startup refusal."""
    import scripts.train as training

    corpus, _ = _corpus(tmp_path)
    config, _, _ = training.build_config([
        "--time_consuming_mode", "stochastic",
        "--time_consuming_memory_budget_mb", "0.001",
    ])
    config.model.hidden_dim = 8
    config.training.num_workers = 0
    config.checkpoint.output_dir = str(tmp_path / "denied")
    training.build_config([
        "--time_consuming_mode", "stochastic",
        "--time_consuming_memory_budget_mb", "16384",
    ])
    with mock.patch("scripts.train.train_one_epoch", side_effect=AssertionError(
        "Memory refusal must precede the epoch executor"
    )):
        with pytest.raises(ValueError, match="budget"):
            training.train(config, dataset_path=str(corpus))
    assert not (tmp_path / "denied" / "final_model.pth").exists()


def _analysis_inputs(tmp_path: Path) -> list[str]:
    """Make valid analysis inputs with a known eight-frame contrast."""
    x = torch.zeros(4, 8, 8)
    x[:, 2:, 1] = 1.0
    x[:2, 2:, 0] = 2.0
    torch.save({
        "X_seqs": list(x), "Y_seqs": list(torch.zeros(4, 8)),
        "anchor_frames": [2] * 4,
    }, tmp_path / "dataset.pt")
    table = _payload()
    table["anchor_frames"].fill_(2)
    torch.save(table, tmp_path / "targets.pt")
    expected = torch.tensor([9.0, 11.0, 1.0, 3.0])[:, None].expand(4, 8)
    torch.save({
        "expected_delay": expected, "realized_delay": expected,
        "lengths": torch.full((4,), 8), "source_indices": torch.arange(4),
        "dt_ms": 4.0, "max_seq_len": 8, "pre_anchor_frames": 0,
    }, tmp_path / "internals.pt")
    return [
        "--internals", str(tmp_path / "internals.pt"),
        "--latency-target-table", str(tmp_path / "targets.pt"),
        "--dataset", str(tmp_path / "dataset.pt"), "--target-group", "0",
        "--reference-group", "1", "--bootstrap", "10",
        "--output", str(tmp_path / "report.json"),
    ]


def test_analysis_refuses_duplicate_trial_rows(tmp_path: Path) -> None:
    """One source trial cannot be reweighted by duplicated internals rows."""
    from scripts.analyze_latency import main

    args = _analysis_inputs(tmp_path)
    path = tmp_path / "internals.pt"
    internals = load_artifact_bytes(path.read_bytes())
    internals["source_indices"] = torch.tensor([0, 0, 2, 3])
    torch.save(internals, path)
    with pytest.raises(ValueError, match="source_indices.*unique"):
        main(args)
    assert not (tmp_path / "report.json").exists()


def test_analysis_derives_missing_legacy_anchors(tmp_path: Path) -> None:
    """Legacy physical channels give the same anchor as training startup."""
    import json
    from scripts.analyze_latency import main

    args = _analysis_inputs(tmp_path)
    dataset = tmp_path / "dataset.pt"
    raw = load_artifact_bytes(dataset.read_bytes())
    raw.pop("anchor_frames")
    torch.save(raw, dataset)
    assert main(args) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["available"] is True
    assert report["delta_d_frames"] == 8.0


@pytest.mark.parametrize("channel", [0, 1])
def test_latency_conditions_survive_input_ablation(
    tmp_path: Path, script_settings: dict[str, Any], channel: int,
) -> None:
    """Ablated network input cannot redefine the physical trial condition."""
    import scripts.train as training
    from nsmor.pipeline.latency_metadata import build_latency_sidecar

    corpus, groups = _corpus(tmp_path)
    # The ablation must travel with the config under test: a second
    # build_config call would only mutate module globals and the run would
    # silently train unablated (train.py prefers config-bound controls).
    config = _config(tmp_path, groups, "a", zero_input_channels=channel)
    assert getattr(config, "_zero_input_channels", None) == (channel,)
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})) as epoch:
        training.train(config, dataset_path=str(corpus))
    loader = epoch.call_args.kwargs["loader"]
    assert all(np.count_nonzero(sequence[0][:, channel]) == 0
               for sequence in loader.dataset.sequences)
    # The preserved physical metadata must still reject a relabelled target.
    table = load_artifact_bytes((tmp_path / "a.pt").read_bytes())
    table["group_ids"] = 1 - table["group_ids"]
    wrong = tmp_path / "wrong.pt"
    torch.save(table, wrong)
    from nsmor.pipeline.latency_metadata import load_latency_target_table

    with pytest.raises(ValueError, match="condition mismatch"):
        build_latency_sidecar(
            load_latency_target_table(wrong), loader.dataset.source_indices,
            dataset=loader.dataset,
        )
