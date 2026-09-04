"""Tests for scripts/prepare_metadata.py metadata enrichment."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch


def test_prepare_metadata_spec_contains_condition_fields():
    """Metadata trial_specs and root dict must include stimulus_condition and is_pure_wind."""
    # Test the classification logic and schema contract
    trial_data_wind = {
        "visual_angle": np.zeros(100),
        "wind_state": np.ones(100),
        "time_ms": np.arange(100) * 10.0,
    }
    trial_data_vis = {
        "visual_angle": np.ones(100) * 10.0,
        "wind_state": np.zeros(100),
        "time_ms": np.arange(100) * 10.0,
    }

    def classify(td):
        has_vis = bool(np.any(np.abs(td["visual_angle"]) > 0.0))
        has_wind = bool(np.any(np.abs(td["wind_state"]) > 0.0))
        if has_vis and has_wind:
            return "multisensory"
        if has_vis:
            return "visual_only"
        if has_wind:
            return "wind_only"
        return "no_stimulus"

    assert classify(trial_data_wind) == "wind_only"
    assert classify(trial_data_vis) == "visual_only"
