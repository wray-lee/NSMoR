"""Synthetic CLI clock and source-anchor regressions for CAMoR-L2 r8."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


def _analysis_inputs(
    tmp_path: Path, *, dataset_dt_ms: float = 4.0,
    stored_anchor: int = 16, legacy: bool = False,
) -> list[str]:
    """Write eight trials with a source stimulus at 64 ms and known delays."""
    n, timesteps = 8, 28
    source_anchor = int(64.0 / dataset_dt_ms)
    x = torch.zeros(n, timesteps, 8)
    x[:, source_anchor:, 1] = 1.0
    x[:4, source_anchor:, 0] = 2.0
    assert x.shape == (n, timesteps, 8)
    raw = {
        "X_seqs": list(x), "Y_seqs": list(torch.zeros(n, timesteps)),
        "lengths": torch.full((n,), timesteps),
    }
    if not legacy:
        raw.update({
            "anchor_frames": [stored_anchor] * n,
            "model_dt_ms": dataset_dt_ms,
            "model_grid_provenance": [{
                "origin_ms": 0.0, "dt_ms": dataset_dt_ms,
                "source_n": timesteps, "model_n": timesteps,
                "source_end_ms": (timesteps - 1) * dataset_dt_ms,
                "model_end_ms": (timesteps - 1) * dataset_dt_ms,
                "synthetic_prepend_frames": 0, "source_anchor_ms": 64.0,
                "source_anchor_rule": "stimulus_onset",
            } for _ in range(n)],
        })
    torch.save(raw, tmp_path / "dataset.pt")
    frames = torch.tensor([6.0] * 4 + [1.0] * 4)
    torch.save({
        "target_frames": frames, "target_ms": frames * 4.0, "dt_ms": 4.0,
        "mask": torch.ones(n, dtype=torch.bool),
        "group_ids": torch.tensor([0] * 4 + [1] * 4),
        "prefix_ids": torch.arange(n),
        "anchor_frames": torch.full((n,), stored_anchor),
        "recipe": "synthetic response-onset latency",
        "anchor_recipe": "synthetic source event in saved coordinates",
        "condition_mapping": {"multisensory": 0, "wind_only": 1},
    }, tmp_path / "targets.pt")
    expected = torch.arange(1.0, n + 1)[:, None].expand(n, timesteps).clone()
    expected[:4, source_anchor] += 10.0
    assert expected.shape == (n, timesteps)
    torch.save({
        "expected_delay": expected, "realized_delay": expected,
        "lengths": torch.full((n,), timesteps), "source_indices": torch.arange(n),
        "dt_ms": 4.0, "max_seq_len": timesteps, "pre_anchor_frames": 0,
    }, tmp_path / "internals.pt")
    return [
        "--internals", str(tmp_path / "internals.pt"),
        "--latency-target-table", str(tmp_path / "targets.pt"),
        "--dataset", str(tmp_path / "dataset.pt"),
        "--target-group", "0", "--reference-group", "1", "--bootstrap", "10",
        "--output", str(tmp_path / "report.json"),
    ]


@pytest.mark.parametrize(
    "dataset_dt_ms,stored_anchor,reason",
    [
        (8.0, 8, r"model_dt_ms=8\.0 conflicts with expected_dt_ms=4\.0"),
        (4.0, 12, "source anchor disagrees with stored anchor_frames"),
    ],
    ids=["eight-ms-dataset-four-ms-internals", "frame16-source-frame12-claims"],
)
def test_analysis_refuses_contradictory_source_clock(
    tmp_path: Path, dataset_dt_ms: float, stored_anchor: int, reason: str,
) -> None:
    """Consistent internals/table claims cannot override the dataset clock."""
    from scripts.analyze_latency import main

    args = _analysis_inputs(
        tmp_path, dataset_dt_ms=dataset_dt_ms, stored_anchor=stored_anchor,
    )
    with pytest.raises(ValueError, match=reason):
        main(args)
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("legacy", [False, True], ids=["modern", "legacy"])
def test_analysis_uses_verified_or_derived_source_anchor(
    tmp_path: Path, legacy: bool,
) -> None:
    """Frame16 yields +6 frames; genuine legacy can omit stored anchors."""
    from scripts.analyze_latency import main

    args = _analysis_inputs(tmp_path, legacy=legacy)
    assert main(args) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["available"] is True
    # At frame16, target delays [11, 12, 13, 14] minus [5, 6, 7, 8] give +6.
    # Accepting the contradictory frame12 claims would instead report -4.
    assert report["delta_d_frames"] == 6.0


@pytest.mark.parametrize(
    "missing,expected_message",
    [
        (
            "physical_units",
            "latency target definition requires target_ms and dt_ms evidence",
        ),
        ("recipe", "latency target definition requires a nonempty recipe"),
        (
            "anchor_recipe",
            "latency target definition requires a nonempty anchor_recipe",
        ),
    ],
)
def test_analysis_definition_error_is_independent_of_loss_weight(
    tmp_path: Path, missing: str, expected_message: str,
) -> None:
    """Analysis requires evidence even for controls with no latency loss."""
    from scripts.analyze_latency import main

    args = _analysis_inputs(tmp_path)
    path = tmp_path / "targets.pt"
    table = torch.load(path, map_location="cpu", weights_only=True)
    if missing == "physical_units":
        table.pop("target_ms")
        table.pop("dt_ms")
    else:
        table.pop(missing)
    torch.save(table, path)
    with pytest.raises(ValueError) as exc_info:
        main(args)
    assert str(exc_info.value) == expected_message
    assert not (tmp_path / "report.json").exists()
