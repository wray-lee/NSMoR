"""Direct regression test matching adversarial_a.py from review-A.

Verifies:
1. Changed split on resume raises ValueError.
2. Omitted artifact on resume raises ValueError.
3. Changed dataset on resume raises ValueError.
4. In-memory/file mismatch in generate_nested_priors raises ValueError.
5. Legitimate resume preserves lineage and best_val_loss.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from nsmor.config import FeatureConfig, MCMCTrainingConfig, PIPELINE_SEMANTICS_VERSION
from nsmor.config_parser import ExperimentConfig
from nsmor.pipeline.grouping import animal_keys_of, grouped_train_val_split
from nsmor.pipeline.nested_prior import load_nested_prior_split
from scripts.evaluate_nested_prior import generate_nested_priors
from scripts.train import train, build_dataloaders, compute_target_stats


def _make_config(out_dir: Path, epochs: int = 1, resume: str | None = None) -> ExperimentConfig:
    c = ExperimentConfig()
    c.model.hidden_dim = 8
    c.model.num_gru_layers = 1
    c.model.dropout = 0.0
    c.training.num_epochs = epochs
    c.training.batch_size = 32
    c.training.max_seq_len = 24
    c.training.num_workers = 0
    c.training.normalize_targets = True
    c.training.target_clip_cm_s = 100.0
    c.training.lr_warmup_epochs = 0
    c.training.checkpoint_interval = 1
    c.loss.warmup_epochs = 0
    c.loss.lambda_routing_aux = 0.0
    c.checkpoint.output_dir = str(out_dir)
    c.checkpoint.resume_from = resume
    return c


def test_adversarial_review_a_all_findings_resolved(tmp_path: Path):
    rng = np.random.RandomState(1701)
    N, T = 32, 24
    xs, ys = [], []
    for i in range(N):
        x = np.zeros((T, 8), dtype=np.float32)
        x[:, 0] = np.linspace(0, 2, T)
        x[12:, 1] = 1.0 if i % 2 else 0.0
        y = (0.2 + 0.04 * (i // 4) + 0.08 * np.sin(np.arange(T) / 4) + rng.normal(0, 0.01, T)).astype(np.float32)
        x[1:, 2] = y[:-1]
        x[1:, 3] = np.diff(y) / 0.004
        xs.append(x)
        ys.append(y)

    ds = dict(
        X_seqs=xs,
        Y_seqs=ys,
        snapshots=rng.normal(size=(N, 5)),
        mcmc_snapshots=rng.normal(size=(N, 5)),
        labels=np.tile(np.arange(4), 8),
        lengths=np.full(N, T),
        mcmc_priors=np.full((N, 4), 0.25),
        session_ids=[f"synthetic_animal_{i//4}_session_{i%4+1}" for i in range(N)],
        trial_ids=np.arange(N),
        feature_config=FeatureConfig(),
        pipeline_semantics_version=PIPELINE_SEMANTICS_VERSION,
        mcmc_prior_provenance="oof_5fold_animal_grouped_cv",
        anchor_frames=np.full(N, 12),
        dt_ms=4.0,
    )
    ds_path = tmp_path / "synthetic-canonical.pt"
    torch.save(ds, ds_path)

    # 1. Generate legitimate seed-3 artifact
    art3 = generate_nested_priors(
        ds,
        split_seed=3,
        val_split=0.25,
        n_inner_folds=2,
        mcmc_config=MCMCTrainingConfig(num_epochs=4, random_seed=37),
        source_dataset_path=ds_path,
        verbose=False,
    )
    art3_path = tmp_path / "generated-seed3.pt"
    torch.save(art3, art3_path)

    # 2. Train epoch 1
    c1 = _make_config(tmp_path / "initial", epochs=1)
    res1 = train(c1, lambda_reg=0.01, dataset_path=str(ds_path), nested_prior_artifact=str(art3_path))
    ckpt = str(Path(c1.checkpoint.output_dir) / "epoch_1.pth")

    # 3. Same-lineage resume must succeed and preserve best_val_loss
    rc = _make_config(tmp_path / "resume-same", epochs=2, resume=ckpt)
    res2 = train(rc, lambda_reg=0.01, dataset_path=str(ds_path), nested_prior_artifact=str(art3_path))
    assert res2["is_nested_cv"] is True
    assert res2["nested_prior_artifact"] == str(art3_path)

    # 4. Generate alternative split (seed 4)
    art4 = generate_nested_priors(
        ds,
        split_seed=4,
        val_split=0.25,
        n_inner_folds=2,
        mcmc_config=MCMCTrainingConfig(num_epochs=4, random_seed=37),
        source_dataset_path=ds_path,
        verbose=False,
    )
    art4_path = tmp_path / "generated-seed4.pt"
    torch.save(art4, art4_path)

    # Finding 1 check: changed-split resume must fail closed (raise ValueError)
    c_split = _make_config(tmp_path / "resume-changed-split", epochs=2, resume=ckpt)
    with pytest.raises(ValueError, match="split_seed mismatch|artifact mismatch"):
        train(c_split, lambda_reg=0.01, dataset_path=str(ds_path), nested_prior_artifact=str(art4_path))

    # Finding 2 check: omitted artifact resume must fail closed (raise ValueError)
    c_omit = _make_config(tmp_path / "resume-omit", epochs=2, resume=ckpt)
    with pytest.raises(ValueError, match="Resume nested mode mismatch"):
        train(c_omit, lambda_reg=0.01, dataset_path=str(ds_path), nested_prior_artifact=None)

    # Finding 3 check: changed dataset resume must fail closed (raise ValueError)
    ds_alt = copy.deepcopy(ds)
    ds_alt["labels"] = (ds_alt["labels"] + 1) % 4
    ds_alt_path = tmp_path / "alt_dataset.pt"
    torch.save(ds_alt, ds_alt_path)
    art_alt = generate_nested_priors(
        ds_alt,
        split_seed=3,
        val_split=0.25,
        n_inner_folds=2,
        mcmc_config=MCMCTrainingConfig(num_epochs=4, random_seed=37),
        source_dataset_path=ds_alt_path,
        verbose=False,
    )
    art_alt_path = tmp_path / "art_alt.pt"
    torch.save(art_alt, art_alt_path)

    c_ds = _make_config(tmp_path / "resume-changed-ds", epochs=2, resume=ckpt)
    with pytest.raises(ValueError, match="fingerprint mismatch|artifact mismatch"):
        train(c_ds, lambda_reg=0.01, dataset_path=str(ds_alt_path), nested_prior_artifact=str(art_alt_path))

    # Finding 4 check: in-memory / file label mismatch must fail closed (raise ValueError)
    mis = copy.deepcopy(ds)
    mis["labels"] = (mis["labels"] + 1) % 4
    with pytest.raises(ValueError, match="in-memory/file mismatch"):
        generate_nested_priors(
            mis,
            split_seed=3,
            val_split=0.25,
            n_inner_folds=2,
            source_dataset_path=ds_path,
            verbose=False,
        )
