"""n==1 must not be reported as SEM=0 (T5).

A single observation has no estimable dispersion; reporting ``sem=0.0``
fabricates perfect precision.  The statistics must return ``None`` (an
unavailable estimate) for n==1, and every plotting/serialization
consumer must tolerate it.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.analyze_integration import (
    compute_condition_statistics,
    create_integration_figure,
    export_integration_summary,
)


def test_singleton_sem_is_none_not_zero() -> None:
    stats = compute_condition_statistics({"latencies": [10.0], "peak_velocities": [5.0]})
    assert stats["latency"] == {"mean": 10.0, "sem": None, "n": 1}
    assert stats["peak_velocity"] == {"mean": 5.0, "sem": None, "n": 1}


def test_two_observations_keep_real_sem() -> None:
    stats = compute_condition_statistics({"latencies": [10.0, 20.0]})
    assert stats["latency"]["n"] == 2
    assert stats["latency"]["sem"] == pytest.approx(np.std([10.0, 20.0], ddof=1) / np.sqrt(2))


def test_singleton_summary_serializes_null_sem(tmp_path) -> None:
    stats = compute_condition_statistics({"latencies": [10.0], "peak_velocities": [5.0]})
    path = tmp_path / "integration.json"
    export_integration_summary({"multisensory_ttc_-225ms": stats}, path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    cell = summary["conditions"]["multisensory_ttc_-225ms"]["latency_to_peak_ms"]
    assert cell == {"mean": 10.0, "sem": None, "n": 1}


def test_singleton_wind_point_plots_without_errorbar(tmp_path) -> None:
    """A singleton wind group must plot (no crash) with an absent error bar."""
    singleton = compute_condition_statistics({"latencies": [10.0], "peak_velocities": [5.0]})
    multi = compute_condition_statistics({
        "latencies": [100.0, 200.0, 300.0], "peak_velocities": [1.0, 2.0, 3.0],
    })
    stats = {
        "multisensory_ttc_-225ms": singleton,
        "multisensory_ttc_-261ms": multi,
    }
    figure = tmp_path / "integration.png"
    create_integration_figure(stats, figure)
    assert figure.stat().st_size > 0
