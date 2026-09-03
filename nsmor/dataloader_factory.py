"""
NSMoR DataLoader Factory — optimized DataLoader construction.

Single source of truth for the DataLoader parallelism policy across the
training engine and every analysis script.  Centralizing the worker /
pin-memory / prefetch decisions guarantees that all entry points use
the same, spawn-safe loading strategy instead of hand-rolled
``torch.utils.data.DataLoader(...)`` calls whose keyword choices drift
between scripts.

Policy summary
--------------
* ``num_workers == -1`` (config default): **auto-scale** — ``0`` workers
  for small datasets (below :data:`SMALL_DATASET_THRESHOLD` sequences,
  where worker start-up overhead exceeds the loading cost) and
  ``min(MAX_AUTO_WORKERS, os.cpu_count() - 1)`` otherwise.  Dataset
  size is measured by ``len(dataset)`` (sequence count), preserving
  the PyTorch Dataset abstraction.
* ``num_workers >= 0``: honored verbatim (explicit user override;
  ``0`` = single-process, deterministic, debuggable).
* ``pin_memory`` is silently disabled on CPU-only builds — page-locked
  allocation only pays off with CUDA host-to-device transfer.
* ``persistent_workers`` / ``prefetch_factor`` require
  ``num_workers > 0``; PyTorch raises ``ValueError`` for the invalid
  combinations, so the factory coerces them instead of forwarding.

Worker-process safety
---------------------
Windows (and any ``spawn`` start method) re-imports the constructing
module in each worker.  Every object a worker must see — the dataset
class and the collate function — therefore lives at module top level
of :mod:`nsmor.nsmor_dataloader` and is passed **by reference**.  This
module never wraps them in closures, lambdas, or locally defined
classes, so pickling the loader configuration is always safe.

Example
-------
Python::

    from nsmor.dataloader_factory import (
        create_optimized_dataloader, create_dataloaders_from_config,
    )

    loader = create_optimized_dataloader(dataset, batch_size=64)

    # Or straight from an ExperimentConfig (train / val / test):
    train_loader, val_loader, _ = create_dataloaders_from_config(
        cfg, train_ds, val_ds,
    )
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nsmor.config_parser import ExperimentConfig
from nsmor.nsmor_dataloader import collate_variable_length

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Policy constants
# ──────────────────────────────────────────────────────────────

SMALL_DATASET_THRESHOLD: int = 200
"""Sequence count below which multi-process loading never pays off.

Worker process start-up (~seconds under ``spawn``, plus per-batch IPC)
dominates the total loading time of small datasets.  Measured on NSMoR
trial data: 200 sequences is the empirical breakeven where worker overhead
equals the loading benefit.  Uses sequence count (``len(dataset)``) instead
of total frames to preserve PyTorch Dataset abstraction — subclasses need
not expose internal structure."""

MAX_AUTO_WORKERS: int = 4
"""Upper bound for auto-scaled worker count.  Beyond ~4 workers the
collate + IPC overhead saturates the consumer process for this workload;
more workers only multiply memory footprint."""


# ──────────────────────────────────────────────────────────────
# Worker-count resolution
# ──────────────────────────────────────────────────────────────

def compute_num_workers(
    dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]],
    num_workers: int = -1,
) -> int:
    """
    Resolve the effective worker count for a dataset.

    Auto-scale rule (``num_workers == -1``): small datasets run
    single-process because worker start-up overhead dominates; larger
    datasets scale to ``min(MAX_AUTO_WORKERS, cpu_count - 1)`` so the
    consumer process always keeps one core.  Dataset size is measured
    by ``len(dataset)`` (sequence count), preserving the PyTorch
    Dataset abstraction without requiring subclasses to expose internal
    structure like ``.sequences``.

    Args:
        dataset: A sized dataset yielding ``(X_seq, Y_seq)`` tuples.
            Only ``len()`` is called; no internal attributes accessed.
        num_workers: Requested worker count.  ``-1`` selects
            auto-scaling; ``>= 0`` is honored verbatim; anything below
            ``-1`` is rejected.

    Returns:
        Effective ``num_workers`` for ``torch.utils.data.DataLoader``
        (``>= 0``).

    Raises:
        ValueError: If *num_workers* is below ``-1``.
    """
    if num_workers < -1:
        raise ValueError(
            f"num_workers must be -1 (auto) or >= 0, got {num_workers}"
        )

    # Explicit override wins — no silent second-guessing of the caller.
    if num_workers >= 0:
        return num_workers

    # Sequence count (len) instead of total frames to preserve abstraction
    n_sequences: int = len(dataset)
    if n_sequences <= SMALL_DATASET_THRESHOLD:
        return 0
    # Reserve one core for the consumer; guard the degenerate cpu_count=1 box.
    return min(MAX_AUTO_WORKERS, max(1, (os.cpu_count() or 2) - 1))


# ──────────────────────────────────────────────────────────────
# Unified factory
# ──────────────────────────────────────────────────────────────

def create_optimized_dataloader(
    dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = -1,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Build a DataLoader with the project-standard parallelism policy.

    Wraps :func:`compute_num_workers` for worker resolution and coerces
    the worker-dependent keywords so every combination PyTorch would
    reject is normalized away before construction.  Keyword-only
    arguments prevent positional-argument drift at call sites.

    Ticket #16: Automatically selects ``collate_with_metadata`` when
    the dataset has ``is_pure_wind`` metadata, enabling the auxiliary
    routing loss.

    Args:
        dataset: A sized dataset yielding ``(X_seq, Y_seq)`` tuples —
            typically an
            :class:`~nsmor.nsmor_dataloader.NSMoRDataset`.  Must be
            constructed at module scope of the caller (spawn safety).
        batch_size: Batch size (``>= 1``).
        shuffle: Shuffle trials each epoch.  Use ``False`` for
            evaluation splits where ordering carries label identity.
        num_workers: Worker count request; ``-1`` auto-scales by
            sequence count (see :func:`compute_num_workers`).
        pin_memory: Page-lock host memory before H2D transfer.
            Automatically disabled on CPU-only builds.
        persistent_workers: Keep worker processes alive across epochs.
            Coerced to ``False`` when the resolved worker count is 0.
        prefetch_factor: Batches pre-loaded per worker ahead of the
            consumer.  Not forwarded when the resolved worker count
            is 0 (PyTorch rejects the combination).

    Returns:
        A ``DataLoader`` yielding ``(X_batch, Y_batch, lengths)``
        tuples via :func:`~nsmor.nsmor_dataloader.collate_variable_length`,
        or ``(X_batch, Y_batch, lengths, wind_only_mask)`` when metadata
        is present.

    Raises:
        ValueError: If *batch_size* is below 1.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    n_sequences: int = len(dataset)
    assert isinstance(n_sequences, int) and n_sequences >= 0, (
        f"dataset must be sized, got len={n_sequences!r}"
    )

    nw: int = compute_num_workers(dataset, num_workers)

    # Ticket #16: Select collate function based on metadata availability.
    # If dataset.is_pure_wind exists, use collate_with_metadata; otherwise
    # fall back to legacy collate_variable_length.
    from functools import partial
    from nsmor.nsmor_dataloader import collate_with_metadata

    if hasattr(dataset, "is_pure_wind") and dataset.is_pure_wind is not None:
        collate_fn = partial(collate_with_metadata, is_pure_wind=dataset.is_pure_wind)
    else:
        collate_fn = collate_variable_length

    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        # Module-level collate — spawn-safe (see module docstring).
        "collate_fn": collate_fn,
        # Page-locked memory is meaningless without CUDA H2D transfer.
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "num_workers": nw,
    }
    if nw > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    logger.debug(
        "DataLoader built: n_seqs=%d batch_size=%d "
        "shuffle=%s num_workers=%d pin_memory=%s",
        n_sequences,
        batch_size, shuffle, nw, loader_kwargs["pin_memory"],
    )
    return DataLoader(**loader_kwargs)


# ──────────────────────────────────────────────────────────────
# Config-driven convenience (train / val / test)
# ──────────────────────────────────────────────────────────────

def create_dataloaders_from_config(
    config: ExperimentConfig,
    train_dataset: Optional[Dataset[Tuple[torch.Tensor, torch.Tensor]]],
    val_dataset: Optional[Dataset[Tuple[torch.Tensor, torch.Tensor]]] = None,
    test_dataset: Optional[Dataset[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[
    Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]
]:
    """
    Build train/val/test DataLoaders from an :class:`ExperimentConfig`.

    Reads ``batch_size``, ``num_workers``, ``pin_memory``,
    ``persistent_workers``, and ``prefetch_factor`` from
    ``config.training`` and delegates each split to
    :func:`create_optimized_dataloader`.  Train is shuffled; val and
    test preserve ordering (evaluation metrics and label matching rely
    on trial order).

    Args:
        config: Parsed experiment configuration.
        train_dataset: Training split (``None`` skips the split).
        val_dataset: Validation split (optional).
        test_dataset: Test split (optional).

    Returns:
        ``(train_loader, val_loader, test_loader)`` — entries are
        ``None`` where the corresponding input was ``None``.
    """
    t: Any = config.training  # TrainingConfig
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None

    if train_dataset is not None:
        train_loader = create_optimized_dataloader(
            train_dataset,
            batch_size=t.batch_size,
            shuffle=True,
            num_workers=t.num_workers,
            pin_memory=t.pin_memory,
            persistent_workers=t.persistent_workers,
            prefetch_factor=t.prefetch_factor,
        )
    if val_dataset is not None:
        val_loader = create_optimized_dataloader(
            val_dataset,
            batch_size=t.batch_size,
            shuffle=False,
            num_workers=t.num_workers,
            pin_memory=t.pin_memory,
            persistent_workers=t.persistent_workers,
            prefetch_factor=t.prefetch_factor,
        )
    if test_dataset is not None:
        test_loader = create_optimized_dataloader(
            test_dataset,
            batch_size=t.batch_size,
            shuffle=False,
            num_workers=t.num_workers,
            pin_memory=t.pin_memory,
            persistent_workers=t.persistent_workers,
            prefetch_factor=t.prefetch_factor,
        )
    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    from nsmor.config import DEFAULT_FEATURE
    from nsmor.nsmor_dataloader import NSMoRDataset

    rng = np.random.RandomState(42)
    n_seqs, feat_dim = 8, DEFAULT_FEATURE.per_frame_total_dim
    seqs, priors = [], []
    for i in range(n_seqs):
        sl = int(rng.randint(20, 60))
        X = rng.rand(sl, feat_dim).astype(np.float32)
        X[:, 4:8] = [0.25, 0.25, 0.25, 0.25]
        p = rng.rand(DEFAULT_FEATURE.mcmc_dim)
        p /= p.sum()
        seqs.append((X, rng.rand(sl).astype(np.float32), 0))
        priors.append(p)

    ds = NSMoRDataset(sequences=seqs, mcmc_priors=np.asarray(priors))
    assert compute_num_workers(ds, -1) == 0, f"small dataset ({len(ds)} seqs) must resolve to 0"

    # Build large synthetic dataset (>200 seqs) for auto-scale test
    seqs_large, priors_large = [], []
    for i in range(250):
        sl = int(rng.randint(40, 80))
        X = rng.rand(sl, feat_dim).astype(np.float32)
        X[:, 4:8] = [0.25, 0.25, 0.25, 0.25]
        p = rng.rand(DEFAULT_FEATURE.mcmc_dim)
        p /= p.sum()
        seqs_large.append((X, rng.rand(sl).astype(np.float32), 0))
        priors_large.append(p)
    ds_large = NSMoRDataset(sequences=seqs_large, mcmc_priors=np.asarray(priors_large))
    nw_large = compute_num_workers(ds_large, -1)
    assert nw_large >= 1, f"large dataset ({len(ds_large)} seqs) must enable workers"

    # Override test
    assert compute_num_workers(ds, num_workers=3) == 3, "override wins"

    loader = create_optimized_dataloader(ds, batch_size=4, shuffle=False)
    X_b, Y_b, L_b = next(iter(loader))
    assert X_b.shape == (4, L_b.max().item(), feat_dim), f"{X_b.shape}"
    assert Y_b.shape == (4, L_b.max().item()), f"{Y_b.shape}"
    assert L_b.shape == (4,) and L_b.dtype == torch.int64
    print(f"[OK] dataloader_factory: X={tuple(X_b.shape)} "
          f"L={L_b.tolist()} n_seqs={len(ds)}/{len(ds_large)} "
          f"nw=0/{nw_large} cuda={torch.cuda.is_available()}")
