"""Canonical animal-level grouping for leakage-free dataset splits.

``session_ids`` look like ``0.513cricket_001_20260707_193143_session_2``.
The ``_session_N`` suffix splits **one recording of one animal** into
blocks, so two ids sharing the prefix are the *same animal*.  Grouping a
train/val split by session therefore does **not** prevent leakage: an
animal's ``_session_1`` can land in train while its ``_session_2`` lands
in validation, and trials of one animal share that animal's baseline
locomotor statistics, gain state, and body mass.

Measured on the corpora as of this module's introduction, a
session-grouped split at ``val_split=0.2``, ``random_seed=42`` put this
fraction of validation trials in the same animal as a training trial:

- ``nsmor_dataset.pt``          33.3% (1 of 2 val animals)
- ``nsmor_dataset_full_backup`` 50.0% (2 of 3 val animals)
- ``nsmor_dataset_3cond_v2.pt`` 87.5% (14 of 15 val animals)

Animal grouping is strictly coarser than session grouping, so an
animal-disjoint split is automatically session-disjoint.

The trade-off is split granularity: with few animals, whole-animal
assignment cannot hit the requested fraction exactly (8 animals at
``val_split=0.2`` yields 1 val animal, ~12.5% of trials).  The achieved
fraction is logged so the coarsening is visible rather than silent.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "animal_of",
    "animal_keys_of",
    "check_group_disjoint",
    "grouped_train_val_split",
]

# ``..._session_1`` / ``..._session_12`` -> animal-recording prefix.
_SESSION_SUFFIX = re.compile(r"_session_\d+$")


def animal_of(session_id: Any) -> str:
    """Strip the ``_session_N`` block suffix to get the animal key."""
    return _SESSION_SUFFIX.sub("", str(session_id))


def animal_keys_of(session_ids: Sequence[Any]) -> np.ndarray:
    """Map per-trial session ids to per-trial animal keys.

    Args:
        session_ids: Per-trial session identifiers.

    Returns:
        ``(n_trials,)`` object array of animal keys.
    """
    keys = np.array([animal_of(s) for s in session_ids], dtype=object)
    assert keys.shape == (len(session_ids),), (
        f"Expected ({len(session_ids)},), got {keys.shape}"
    )
    return keys


def check_group_disjoint(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    group_keys: np.ndarray,
) -> None:
    """Raise if any group key appears on both sides of the split.

    Called inside :func:`grouped_train_val_split` so the invariant cannot
    be bypassed by using the function.  Leakage of this kind is silent by
    nature -- it inflates validation metrics rather than failing -- so it
    is asserted at construction time instead of being left to review.

    Args:
        train_indices: Trial indices assigned to train.
        val_indices: Trial indices assigned to validation.
        group_keys: ``(n_trials,)`` per-trial group keys.

    Raises:
        ValueError: If a group key spans both splits.
    """
    both = set(group_keys[train_indices].tolist()) & set(
        group_keys[val_indices].tolist()
    )
    if both:
        raise ValueError(
            f"Split leaks {len(both)} group(s) across train and val: "
            f"{sorted(both)[:5]}"
        )


def grouped_train_val_split(
    session_ids: Optional[Sequence[Any]],
    n_total: int,
    val_split: float = 0.2,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split trial indices so that no *animal* spans train and validation.

    Deterministic across processes: unique keys come from ``np.unique``
    (sorted) rather than ``set`` iteration order, which varies with
    ``PYTHONHASHSEED``.

    Args:
        session_ids: Per-trial session identifiers, or None/wrong-length
            to force the ungrouped fallback.
        n_total: Number of trials.
        val_split: Target fraction of *animals* held out (0-1).
        random_seed: Seed for the animal shuffle.

    Returns:
        ``(train_indices, val_indices)`` as int64 arrays.

    Raises:
        ValueError: If the resulting split leaks an animal across sides.
    """
    rng = np.random.RandomState(random_seed)

    if session_ids is None or len(session_ids) != n_total:
        # Synthetic / single-session data only.  Warn loudly: without
        # group ids there is no way to keep an animal on one side.
        logger.warning(
            "Dataset lacks per-sequence 'session_ids' (got %s for %d "
            "trials); falling back to sample-level train/val split. "
            "Animal-level information may inflate validation metrics. "
            "Re-run prepare_data to regenerate the dataset with them.",
            None if session_ids is None else len(session_ids),
            n_total,
        )
        indices = np.arange(n_total, dtype=np.int64)
        rng.shuffle(indices)
        n_val = max(1, int(n_total * val_split))
        return indices[n_val:], indices[:n_val]

    group_keys = animal_keys_of(session_ids)
    unique_animals = np.unique(group_keys)
    rng.shuffle(unique_animals)

    n_val_animals = max(1, int(len(unique_animals) * val_split))
    val_animals = set(unique_animals[:n_val_animals].tolist())
    is_val = np.array([k in val_animals for k in group_keys])

    val_indices = np.nonzero(is_val)[0].astype(np.int64)
    train_indices = np.nonzero(~is_val)[0].astype(np.int64)

    check_group_disjoint(train_indices, val_indices, group_keys)

    logger.info(
        "Animal-grouped split: %d train (%d animals) / %d val (%d "
        "animals) -- %.1f%% of trials held out (target %.0f%%)",
        len(train_indices),
        len(unique_animals) - n_val_animals,
        len(val_indices),
        n_val_animals,
        100.0 * len(val_indices) / max(1, n_total),
        val_split * 100.0,
    )
    return train_indices, val_indices
