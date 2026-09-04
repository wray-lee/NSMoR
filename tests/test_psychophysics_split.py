"""Test that simulate_psychophysics uses animal-grouped validation split."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION, FeatureConfig
from nsmor.pipeline.grouping import animal_of


def test_simulate_psychophysics_grouped_split(tmp_path: Path):
    """load_validation_data must use animal-grouped split, not trailing 20%."""
    from scripts.simulate_psychophysics import load_validation_data

    n_animals = 5
    blocks = 2
    n_total = n_animals * blocks  # 10 trials
    n_frames = 50
    _X_DIM = 8

    # Create session IDs where animals are interleaved or ordered
    # e.g. animal 0..4 session 1, then animal 0..4 session 2
    # In trailing 20% (indices 8, 9), it would select session 2 of animal 3 and 4,
    # leaking animals 3 and 4 if train were indices 0..7.
    session_ids = [
        f"0.{500 + a}cricket_001_20260101_00000{a}_session_1"
        for a in range(n_animals)
    ] + [
        f"0.{500 + a}cricket_001_20260101_00000{a}_session_2"
        for a in range(n_animals)
    ]

    rng = np.random.RandomState(42)
    X_seqs = [rng.randn(n_frames, _X_DIM).astype(np.float32) for _ in range(n_total)]
    Y_seqs = [rng.randn(n_frames).astype(np.float32) for _ in range(n_total)]
    labels = np.zeros(n_total, dtype=np.int64)
    lengths = np.full(n_total, n_frames, dtype=np.int64)
    mcmc_priors = np.full((n_total, 4), 0.25, dtype=np.float32)

    ds_path = tmp_path / "nsmor_dataset.pt"
    torch.save(
        {
            "X_seqs": X_seqs,
            "Y_seqs": Y_seqs,
            "labels": labels,
            "lengths": lengths,
            "mcmc_priors": mcmc_priors,
            "mcmc_prior_provenance": "oof_5fold_animal_grouped_cv",
            "session_ids": session_ids,
            "feature_config": FeatureConfig(),
            "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        },
        ds_path,
    )

    device = torch.device("cpu")
    # In trailing 20% (split=8), 2 trials would be loaded: indices 8 and 9 (session 2 of animals 3 and 4).
    # In animal-grouped split at val_split=0.2 on 5 animals, 1 animal (both sessions = 2 trials)
    # is selected as val.
    # If animal 4 is selected as val, its session 1 (idx 4) and session 2 (idx 9) are in val.
    # Trailing 20% has idx 8 (animal 3) and idx 9 (animal 4), which is 2 different animals.
    # An animal-grouped split will have all trials in val belonging to the SAME animal (or set of animals)!
    X_val, Y_val, lengths_val = load_validation_data(
        device, max_seq_len=100, dataset_path=str(ds_path)
    )

    # Let's verify that trailing 20% would give a different set of trials than animal-grouped
    # More directly: mock grouped_train_val_split or inspect the loaded data
    assert X_val.shape[0] == 2
    # To check that grouped_train_val_split was called:
    # Under grouped split (seed=42), let's see which animal is selected:
    from nsmor.pipeline.grouping import grouped_train_val_split
    _, expected_val_indices = grouped_train_val_split(
        session_ids, n_total, val_split=0.2, random_seed=42
    )
    # Expected trials should match expected_val_indices
    # Trailing 20% indices are [8, 9].
    # Let's verify expected_val_indices != [8, 9] to make sure the test fails before the fix.
    assert list(expected_val_indices) != [8, 9], "Fixture must distinguish grouped from trailing"
    # Check that the loaded X_val matches expected_val_indices
    for i, orig_idx in enumerate(expected_val_indices):
        np.testing.assert_allclose(
            X_val[i, :n_frames, :4].numpy(), X_seqs[orig_idx][:, :4], atol=1e-5
        )
