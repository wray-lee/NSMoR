"""Synthetic public-boundary regressions for the CAMoR-L2 r4 corrections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from nsmor.pipeline.latency_metadata import load_latency_target_table


def _payload() -> dict[str, Any]:
    """Return a valid four-row target table in saved-frame coordinates."""
    return {
        "target_frames": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "target_ms": torch.tensor([4.0, 8.0, 12.0, 16.0]),
        "dt_ms": 4.0,
        "mask": torch.ones(4, dtype=torch.bool),
        "group_ids": torch.tensor([0, 0, 1, 1]),
        "prefix_ids": torch.arange(4),
        "anchor_frames": torch.zeros(4, dtype=torch.long),
        "pair_ids": torch.tensor([0, 1, 0, 1]),
        "recipe": "synthetic stimulus-to-response latency",
        "anchor_recipe": "synthetic frame zero after deterministic crop",
        "condition_mapping": {"multisensory": 0, "wind_only": 1},
    }


@pytest.mark.parametrize(
    "field,bad",
    [
        ("mask", torch.tensor([0.0, 0.1, 1.0, -1.0])),
        ("group_ids", torch.tensor([0.0, 0.1, 1.0, 1.9])),
        ("prefix_ids", torch.tensor([0.0, 1.1, 2.0, 3.0])),
        ("anchor_frames", torch.tensor([0.0, 0.5, 1.0, 2.0])),
        ("pair_ids", torch.tensor([0.0, 1.1, 0.0, 1.0])),
        ("target_frames", torch.tensor([True, False, True, False])),
    ],
)
def test_table_loader_rejects_lossy_coercions(
    tmp_path: Path, field: str, bad: torch.Tensor,
) -> None:
    """Malformed raw tensor types must fail before any coercion changes rows."""
    payload = _payload()
    payload[field] = bad
    path = tmp_path / "targets.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_latency_target_table(path)


def _build_config(path: Path, *extra: str) -> Any:
    """Exercise the enabled public CLI configuration boundary."""
    from scripts.train import build_config

    return build_config([
        "--time_consuming_mode", "stochastic", "--lambda_latency", "1",
        "--latency_target_table", str(path), "--latency_target_group", "0",
        "--latency_reference_group", "1", *extra,
    ])


@pytest.mark.parametrize(
    "field,value",
    [
        ("recipe", None), ("recipe", "  "), ("anchor_recipe", None),
        ("anchor_recipe", ""), ("condition_mapping", None),
        ("condition_mapping", {"multisensory": 0}),
        ("condition_mapping", {"multisensory": False, "wind_only": 1}),
        ("condition_mapping", {"multisensory": 0, "wind_only": 0}),
    ],
)
def test_enabled_cli_requires_supervision_definition(
    tmp_path: Path, field: str, value: Any,
) -> None:
    """No enabled run may proceed without declared target/anchor semantics."""
    payload = _payload()
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    path = tmp_path / "targets.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match=field):
        _build_config(path)


@pytest.mark.parametrize("mode", ["stochastic", "hard_depth"])
def test_positive_latency_refuses_untrainable_single_depth(mode: str) -> None:
    """K=1 has constant zero delay and must be refused before a batch."""
    from nsmor.config_parser import ExperimentConfig

    config = ExperimentConfig()
    config.model.time_consuming_mode = mode
    config.model.time_consuming_max_steps = 1
    config.loss.lambda_latency = 0.5
    with pytest.raises(ValueError, match="max_steps.*2"):
        config.validate()
    config.loss.lambda_latency = 0.0
    config.validate()  # The unsupervised K=1 degeneration stays valid.


@pytest.mark.parametrize("weight", [False, True, -1.0, float("nan"), float("inf")])
def test_mutated_config_refuses_invalid_latency_weight(weight: Any) -> None:
    """Validation must enforce the loss contract after direct assignment."""
    from nsmor.config_parser import ExperimentConfig

    config = ExperimentConfig()
    config.model.time_consuming_mode = "stochastic"
    config.loss.lambda_latency = weight
    with pytest.raises(ValueError, match="lambda_latency"):
        config.validate()


@pytest.mark.parametrize("weight", ["-1", "nan", "inf"])
def test_cli_refuses_invalid_latency_weight(weight: str) -> None:
    """CLI overrides must fail before any target table is opened."""
    from scripts.train import build_config

    with pytest.raises(ValueError, match="lambda_latency.*finite"):
        build_config(["--lambda_latency", weight])


@pytest.mark.parametrize("field", ["expected_delay", "realized_delay", "target_frames"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_report_refuses_nonfinite_contrast_observations(
    field: str, bad: float,
) -> None:
    """Invalid included observations cannot become an available NaN report."""
    from scripts.analyze_latency import latency_contrast_report

    data = {
        "expected_delay": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        "realized_delay": torch.zeros(4, 1),
        "anchor_frames": torch.zeros(4, dtype=torch.long),
        "target_frames": torch.ones(4),
        "target_mask": torch.ones(4, dtype=torch.bool),
        "group_ids": torch.tensor([0, 0, 1, 1]),
        "prefix_ids": torch.arange(4),
        "lengths": torch.ones(4, dtype=torch.long),
    }
    data[field][0] = bad
    with pytest.raises(ValueError, match="finite"):
        latency_contrast_report(
            **data, target_group=0, reference_group=1, n_bootstrap=10,
        )


@pytest.mark.parametrize("saved_anchor", [1200, 1800, 2400])
def test_latency_loader_checks_actual_saved_anchor(saved_anchor: int) -> None:
    """A 600-frame crop must read anchor 1200, never raw anchor 1800."""
    import numpy as np
    from torch.utils.data import DataLoader

    from nsmor.nsmor_dataloader import NSMoRDataset
    from nsmor.pipeline.latency_metadata import LatencyTargetTable
    from scripts.train import _prepare_latency_loaders

    physical = np.zeros((3000, 8), dtype=np.float32)
    physical[:, 1] = 1.0
    sequences = []
    for row in range(4):
        values = physical.copy()
        if row < 2:
            values[:, 0] = 2.0
        sequences.append((values, np.zeros(3000, dtype=np.float32), 0))
    dataset = NSMoRDataset(
        sequences,
        mcmc_priors=np.full((4, 4), 0.25, dtype=np.float32),
        anchor_frames=[1800] * 4, max_seq_len=2400, pre_anchor_frames=1200,
    )
    payload = _payload()
    table = LatencyTargetTable(
        target_frames=payload["target_frames"], target_mask=payload["mask"],
        group_ids=payload["group_ids"], prefix_ids=payload["prefix_ids"],
        anchor_frames=torch.full((4,), saved_anchor, dtype=torch.long),
        condition_mapping=payload["condition_mapping"],
    )
    loader = DataLoader(dataset, batch_size=4)
    kwargs = dict(
        target_group=0, reference_group=1, min_prefixes=1, min_trials=1,
        seed=0, pairing="none",
    )
    if saved_anchor != 1200:
        with pytest.raises(ValueError, match="anchor"):
            _prepare_latency_loaders(loader, None, table, **kwargs)
    else:
        rebuilt, _, _ = _prepare_latency_loaders(loader, None, table, **kwargs)
        anchors = rebuilt.dataset.sidecar["latency_anchor_frames"]
        assert anchors.tolist() == [1200] * 4


@pytest.mark.parametrize("dt_ms", [2.0, 4.0, 5.0])
def test_table_converts_ms_to_frames_and_retains_units(
    tmp_path: Path, dt_ms: float,
) -> None:
    """A known 20 ms observation converts independently of the anchor clock."""
    payload = _payload()
    payload.pop("target_frames")
    payload["target_ms"] = torch.full((4,), 20.0)
    payload["dt_ms"] = dt_ms
    path = tmp_path / "targets.pt"
    torch.save(payload, path)
    table = load_latency_target_table(path)
    assert table.target_frames.tolist() == [20.0 / dt_ms] * 4
    assert table.target_ms.tolist() == [20.0] * 4
    assert table.dt_ms == dt_ms


@pytest.mark.parametrize("case", ["mismatch", "bad_dt", "missing_dt", "bad_ms"])
def test_table_refuses_invalid_latency_timebase(tmp_path: Path, case: str) -> None:
    """Ambiguous units and contradictory converted targets fail closed."""
    payload = _payload()
    if case == "mismatch":
        payload["target_ms"] += 4
    elif case == "bad_dt":
        payload["dt_ms"] = float("nan")
    elif case == "missing_dt":
        payload.pop("dt_ms")
    else:
        payload["target_ms"][0] = float("inf")
    path = tmp_path / "targets.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="target_ms|dt_ms|timebase"):
        load_latency_target_table(path)


@pytest.mark.parametrize("case", ["missing_units", "wrong_clock"])
def test_enabled_cli_requires_target_clock(tmp_path: Path, case: str) -> None:
    """An enabled objective must carry ms evidence on the active model grid."""
    payload = _payload()
    if case == "missing_units":
        payload.pop("target_ms")
        payload.pop("dt_ms")
    else:
        payload["dt_ms"] = 5.0
        payload["target_ms"] = payload["target_frames"] * 5.0
    path = tmp_path / "targets.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="target_ms|dt_ms"):
        _build_config(path)


def test_loss_typing_names_resolve() -> None:
    """Every typing annotation in the loss module must resolve (F821 guard)."""
    import ast

    import nsmor.loss as loss_module

    tree = ast.parse(Path(loss_module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            annotation = ast.Expression(node.annotation)
            eval(compile(annotation, "loss annotation", "eval"), vars(loss_module))


@pytest.mark.parametrize("mode", ["stochastic", "hard_depth"])
def test_nonfinite_halt_policy_is_documented_and_fail_closed(mode: str) -> None:
    """Reject an undefined pmf before the finite-pmf STE fallback can apply."""
    from nsmor.model_nsmor_core import TimeConsumingRecursion

    module = TimeConsumingRecursion(8, mode=mode, max_steps=3).eval()
    inputs = torch.zeros(2, 8, 8)
    inputs[0, 7] = float("nan")
    with pytest.raises(ValueError, match="halt probabilities must be finite"):
        module(inputs, torch.ones(2, 8, dtype=torch.bool))
    adr = Path(__file__).resolve().parents[1] / (
        "docs/adr/0012-stochastic-time-consuming-depth.md"
    )
    assert "a non-finite halt probability rejects the whole batch" in adr.read_text()


def test_eval_common_random_numbers_are_documented() -> None:
    """Reused position-wise draws must not be reported as independent trials."""
    from nsmor.model_nsmor_core import TimeConsumingRecursion

    module = TimeConsumingRecursion(8, mode="stochastic", max_steps=4).eval()
    with torch.no_grad():
        module.halt.weight.zero_()
    valid = torch.ones(4, 30, dtype=torch.bool)
    generator = torch.Generator().manual_seed(1)
    first = torch.randn(4, 30, 8, generator=generator)
    second = torch.randn(4, 30, 8, generator=generator)
    assert torch.equal(module(first, valid)[1], module(second, valid)[1])
    reordered = module(first.flip(0), valid)[1].flip(0)
    assert not torch.equal(module(first, valid)[1], reordered)
    adr = Path(__file__).resolve().parents[1] / (
        "docs/adr/0012-stochastic-time-consuming-depth.md"
    )
    text = adr.read_text()
    assert "common random numbers" in text
    assert "SD across seeds is not a trial-level standard error" in text


def test_secondary_delay_description_names_target_index() -> None:
    """The secondary anchor readout must explicitly name its target clock."""
    from scripts.analyze_latency import latency_contrast_report

    report = latency_contrast_report(
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]), torch.zeros(4, 1),
        torch.zeros(4, dtype=torch.long), torch.ones(4),
        torch.ones(4, dtype=torch.bool), torch.tensor([0, 0, 1, 1]),
        torch.arange(4), torch.ones(4, dtype=torch.long),
        target_group=0, reference_group=1, n_bootstrap=10,
    )
    assert "target-indexed" in report["realized_delay_secondary"]["note"]


def test_memory_guard_documents_excluded_full_model_activations() -> None:
    """The startup estimate cannot be presented as a full-model fit check."""
    from scripts.train import _check_time_consuming_memory_budget

    assert "excludes LIF/GRU/frontend activations" in (
        _check_time_consuming_memory_budget.__doc__ or ""
    )


def test_stochastic_regression_file_respects_line_limit() -> None:
    """Keep the reviewed test file within the mandatory 88-column limit."""
    path = Path(__file__).with_name("test_stochastic_time_consuming.py")
    too_long = [i for i, line in enumerate(path.read_text().splitlines(), 1)
                if len(line) > 88]
    assert too_long == []
