"""Focused tests for train.py script-owned controls isolation.

Verifies that module-level phase-2 and script-owned controls
(_ZERO_INPUT_CHANNELS, _SELECTION_METRIC, _RESUME_ID, _SWEEP_BANDS) are bound
to the parsed ExperimentConfig / train call, so consecutive build_config and
train invocations cannot leak into one another.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch

from nsmor.config_parser import ExperimentConfig
from nsmor.pipeline.latency_metadata import build_latency_sidecar
from nsmor.pipeline.nested_prior import load_artifact_bytes
import scripts.train as T
from tests.test_train_checkpoint import _make_config, _make_synthetic_dataset


@pytest.fixture(autouse=True)
def _restore_trainer_globals() -> Iterator[None]:
    """Restore trainer module globals around each test."""
    saved = (
        T._ZERO_INPUT_CHANNELS,
        T._SELECTION_METRIC,
        T._RESUME_ID,
        T._SWEEP_BANDS,
    )
    yield
    (
        T._ZERO_INPUT_CHANNELS,
        T._SELECTION_METRIC,
        T._RESUME_ID,
        T._SWEEP_BANDS,
    ) = saved


def _cli_config(
    tmp_path: Path,
    name: str,
    *,
    epochs: int = 1,
    extra: tuple[str, ...] = (),
) -> ExperimentConfig:
    """Build a config through the public CLI with test-sized settings."""
    cfg, _, _ = T.build_config([
        "--epochs", str(epochs),
        "--hidden_dim", "8",
        "--batch_size", "8",
        "--checkpoint_interval", "1",
        "--output_dir", str(tmp_path / name),
        *extra,
    ])
    cfg.training.num_workers = 0
    cfg.training.persistent_workers = False
    cfg.training.lr_warmup_epochs = 0
    return cfg


def test_build_config_multi_call_selection_metric_isolation(
    tmp_path: Path,
) -> None:
    """A=mse followed by B=default must not make train(A) select on total."""
    ds = _make_synthetic_dataset(tmp_path)
    cfg_a = _cli_config(
        tmp_path, "run_a", extra=("--selection_metric", "mse"),
    )
    cfg_b = _cli_config(tmp_path, "run_b")

    assert getattr(cfg_a, "_selection_metric", None) == "mse"
    assert getattr(cfg_b, "_selection_metric", None) == "total"

    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", wraps=T.validate) as validate_mock,
    ):
        T.train(cfg_a, lambda_reg=0.01, dataset_path=str(ds))

    assert validate_mock.call_args_list
    assert all(
        call.kwargs["selection_metric"] == "mse"
        for call in validate_mock.call_args_list
    )
    best_a = Path(cfg_a.checkpoint.output_dir) / "best_model.pth"
    ckpt_a = load_artifact_bytes(best_a.read_bytes(), map_location="cpu")
    assert ckpt_a["selection_metric"] == "mse"


def test_build_config_multi_call_zero_input_channels_isolation(
    tmp_path: Path,
) -> None:
    """A=ablate 2,3 followed by B=default must keep train(A) ablated."""
    ds = _make_synthetic_dataset(tmp_path)
    cfg_a = _cli_config(
        tmp_path, "run_a", extra=("--zero_input_channels", "2,3"),
    )
    cfg_b = _cli_config(tmp_path, "run_b")

    assert getattr(cfg_a, "_zero_input_channels", None) == (2, 3)
    assert getattr(cfg_b, "_zero_input_channels", None) == ()

    captured: dict[str, Any] = {}
    real_build = T.build_dataloaders

    def _spy_build(*args: Any, **kwargs: Any) -> Any:
        loaders = real_build(*args, **kwargs)
        captured["loaders"] = loaders
        return loaders

    with (
        mock.patch("scripts.train.build_dataloaders", side_effect=_spy_build),
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
    ):
        T.train(cfg_a, lambda_reg=0.01, dataset_path=str(ds))

    train_loader, _ = captured["loaders"]
    batch = next(iter(train_loader))
    x = batch[0]
    assert torch.equal(x[:, :, 2], torch.zeros_like(x[:, :, 2]))
    assert torch.equal(x[:, :, 3], torch.zeros_like(x[:, :, 3]))
    ckpt_a = load_artifact_bytes(
        (Path(cfg_a.checkpoint.output_dir) / "best_model.pth").read_bytes(),
        map_location="cpu",
    )
    assert ckpt_a["zero_input_channels"] == [2, 3]


def test_default_config_does_not_inherit_later_nondefault_invocation(
    tmp_path: Path,
) -> None:
    """Dangerous order: default cfg first, non-default cfg second, train first.

    The later ``build_config`` must not add a sweep or a resume identity to
    the already-bound default config: no ``escape_sensitivity.csv`` and a
    bare-resume provenance entry (``resume_id is None``).
    """
    ds = _make_synthetic_dataset(tmp_path)
    cfg_a = _cli_config(tmp_path, "run_a", epochs=2)
    out = Path(cfg_a.checkpoint.output_dir)

    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        T.train(cfg_a, lambda_reg=0.01, dataset_path=str(ds))

    src = out / "epoch_1.pth"
    assert src.exists()
    (out / "epoch_2.pth").unlink()
    final = out / "final_model.pth"
    if final.exists():
        final.unlink()
    (out / "train.stdout.log").write_text(
        "[t] INFO - Epoch 1/2  train_loss=1.0\n"
        "[t] INFO - Epoch 2/2  train_loss=1.0\n",
        encoding="utf-8",
    )

    # A LATER invocation declares a sweep and a launcher resume id.  cfg_a is
    # already bound to None/None and must keep those values.
    _cli_config(
        tmp_path, "run_b",
        extra=(
            "--sweep_escape_band", "5,10,20",
            "--resume-id", "uuid-B",
        ),
    )
    assert T._SWEEP_BANDS == [5.0, 10.0, 20.0]
    assert T._RESUME_ID == "uuid-B"

    cfg_a.checkpoint.resume_from = str(src)
    pin = tuple(
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (out / "best_model.pth", src)
    )
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        T.train(
            cfg_a, lambda_reg=0.01, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=pin,
        )

    assert not (out / "escape_sensitivity.csv").exists()
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    entries = [
        entry for entry in prov.get("resume_log", [])
        if entry.get("disclosed_rollback")
    ]
    assert entries
    assert all(entry.get("resume_id") is None for entry in entries)


def test_programmatic_config_refuses_stale_module_globals(
    tmp_path: Path,
) -> None:
    """An unbound config must not silently inherit a prior invocation's arm."""
    ds = _make_synthetic_dataset(tmp_path)
    cfg = _make_config(tmp_path, epochs=1)
    T._SELECTION_METRIC = "mse"
    T._ZERO_INPUT_CHANNELS = (1, 2)

    with pytest.raises(ValueError, match="not bound to this config"):
        T.train(cfg, lambda_reg=0.01, dataset_path=str(ds))


def test_programmatic_config_uses_defaults_when_globals_default(
    tmp_path: Path,
) -> None:
    """A directly constructed config still works while module state is default."""
    ds = _make_synthetic_dataset(tmp_path)
    cfg = _make_config(tmp_path, epochs=1)
    T._SELECTION_METRIC = T._DEFAULT_SELECTION_METRIC
    T._ZERO_INPUT_CHANNELS = T._DEFAULT_ZERO_INPUT_CHANNELS

    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        T.train(cfg, lambda_reg=0.01, dataset_path=str(ds))

    ckpt = load_artifact_bytes(
        (Path(cfg.checkpoint.output_dir) / "best_model.pth").read_bytes(),
        map_location="cpu",
    )
    assert ckpt["selection_metric"] == "total"
    assert ckpt["zero_input_channels"] == []


def test_resume_guard_uses_call_resolved_controls(tmp_path: Path) -> None:
    """The guard must compare the EXECUTED overrides, not config defaults."""
    cfg = ExperimentConfig()
    cfg._selection_metric = "total"
    cfg._zero_input_channels = ()
    ckpt = {
        "config": cfg.to_dict(),
        "selection_metric": "mse",
        "zero_input_channels": [2, 3],
    }
    path = tmp_path / "ckpt.pth"

    with pytest.raises(ValueError, match="selection_metric"):
        T._require_architecture_config_match(ckpt, cfg, path)

    T._require_architecture_config_match(
        ckpt, cfg, path,
        selection_metric="mse", zero_input_channels=(2, 3),
    )
    with pytest.raises(ValueError, match="zero_input_channels"):
        T._require_architecture_config_match(
            ckpt, cfg, path,
            selection_metric="mse", zero_input_channels=(),
        )


def test_train_override_resume_round_trip(tmp_path: Path) -> None:
    """An override arm is written to provenance and must be repeated on resume."""
    ds = _make_synthetic_dataset(tmp_path)
    cfg = _make_config(tmp_path, epochs=1)
    cfg.training.checkpoint_interval = 1

    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        T.train(
            cfg, lambda_reg=0.01, dataset_path=str(ds),
            selection_metric="mse", zero_input_channels=(2, 3),
        )

    out = Path(cfg.checkpoint.output_dir)
    ckpt_path = out / "epoch_1.pth"
    assert ckpt_path.exists()
    ckpt = load_artifact_bytes(ckpt_path.read_bytes(), map_location="cpu")
    assert ckpt["selection_metric"] == "mse"
    assert ckpt["zero_input_channels"] == [2, 3]

    def _pins() -> tuple[str, ...]:
        return tuple(
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (out / "best_model.pth", ckpt_path)
            if path.exists()
        )

    resume = _make_config(tmp_path, epochs=2)
    resume.checkpoint.resume_from = str(ckpt_path)
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        T.train(
            resume, lambda_reg=0.01, dataset_path=str(ds),
            selection_metric="mse", zero_input_channels=(2, 3),
            trusted_historical_checkpoint_sha256=_pins(),
        )

    # The accepted resume left final_model.pth as the newest complete-epoch
    # checkpoint, so resume from it (not the older periodic) for the
    # fail-closed arm check.
    latest = out / "final_model.pth"
    assert latest.exists()
    bare = _make_config(tmp_path, epochs=2)
    bare.checkpoint.resume_from = str(latest)
    with pytest.raises(ValueError, match="selection_metric"):
        T.train(
            bare, lambda_reg=0.01, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_pins() + (
                hashlib.sha256(latest.read_bytes()).hexdigest(),
            ),
        )


def _assert_max_line_length(path: Path) -> None:
    for idx, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1,
    ):
        assert len(line) <= 88, (
            f"{path}:{idx} exceeds 88 cols ({len(line)}): {line}"
        )


def test_latency_metadata_style_and_docstring() -> None:
    """The changed latency metadata and this test file stay within 88 cols."""
    root = Path(__file__).resolve().parents[1]
    _assert_max_line_length(root / "nsmor" / "pipeline" / "latency_metadata.py")
    _assert_max_line_length(Path(__file__))

    doc = inspect.getdoc(build_latency_sidecar) or ""
    assert "Raises:" in doc
    assert "ValueError:" in doc
    assert "missing" in doc.lower() or "disagree" in doc.lower()


def test_train_public_api_style_and_docstring() -> None:
    """The new train() API lines and Args/Raises entries meet the style gate."""
    src = inspect.getsource(T.train)
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith((
            "selection_metric:", "zero_input_channels:",
            "resume_id:", "sweep_bands:",
        )):
            assert len(line) <= 88, line

    doc = inspect.getdoc(T.train) or ""
    for name in (
        "selection_metric:", "zero_input_channels:",
        "resume_id:", "sweep_bands:", "Raises:",
    ):
        assert name in doc
