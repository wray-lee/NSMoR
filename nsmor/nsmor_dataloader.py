"""
NSMoR DataLoader — PyTorch Dataset and DataLoader for the continuous model.

Combines Trial-Start anchored sequences with **pre-computed** static MCMC
priors into a unified DataLoader for downstream recurrent training.

Supports YAML-driven dataset switching via :func:`create_dataloader_from_config`,
which reads dataset paths from an :class:`~nsmor.config_parser.ExperimentConfig`
and dynamically assembles train / val / test splits.

Per-frame feature layout (``feature_dim = 8``)
----------------------------------------------
    [0] v_vis(t)        — real-time visual angle (deg)
    [1] wind(t)         — real-time wind state (0 / 1)
    [2] v_kine(t-1)     — previous-frame velocity (cm / s)
    [3] a_kine(t-1)     — previous-frame acceleration (cm / s²)
    [4] P_startle       ┐
    [5] P_walk          │ static MCMC prior, identical at every
    [6] P_pre_active    │ frame within a trial
    [7] P_no_response   ┘

Collate return signature
------------------------
``collate_variable_length`` returns a 3-tuple:

    ``(X_batch, Y_batch, lengths)``

where *lengths* is a 1-D ``LongTensor`` of true (unpadded) sequence
lengths for each sample in the batch.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nsmor.config import DEFAULT_FEATURE, FeatureConfig


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class NSMoRDataset(Dataset):
    """
    PyTorch Dataset for NSMoR continuous modelling.

    Each item is a ``(X_seq, Y_seq)`` pair where

    * ``X_seq`` has shape ``(seq_len, 8)``
    * ``Y_seq`` has shape ``(seq_len,)``

    The four MCMC columns (indices 4-7) **must** be pre-filled from
    a pre-computed ``mcmc_priors`` array.  Dynamic inference is not
    supported — callers must run the MCMC module upstream and pass
    the resulting probability matrix.
    """

    def __init__(
        self,
        sequences: List[Tuple[np.ndarray, np.ndarray, int]],
        mcmc_priors: np.ndarray,
        feature_config: FeatureConfig = DEFAULT_FEATURE,
    ) -> None:
        """
        Args:
            sequences: List of ``(X_seq, Y_seq, label)`` tuples
                from :func:`data_extractor.build_sequence_dataset`.
            mcmc_priors: **Required** pre-computed probability vectors,
                shape ``(n_trials, 4)``.  Each row must sum to 1.
            feature_config: Feature dimension constants.

        Raises:
            ValueError: If *mcmc_priors* is ``None`` or its shape does
                not match the number of sequences.
        """
        if mcmc_priors is None:
            raise ValueError(
                "mcmc_priors is required.  Run the MCMC module upstream "
                "and pass the resulting (n, 4) probability matrix.  "
                "Dynamic inference inside the DataLoader is not supported."
            )

        self.feature_config = feature_config
        self.sequences = list(sequences)  # defensive copy

        n = len(sequences)
        expected_shape = (n, feature_config.mcmc_dim)
        if mcmc_priors.shape != expected_shape:
            raise ValueError(
                f"mcmc_priors shape {mcmc_priors.shape} does not match "
                f"expected {expected_shape} ({n} sequences, "
                f"{feature_config.mcmc_dim} classes)."
            )
        self._fill_priors(mcmc_priors)

    # ── Internal helpers ─────────────────────────────────────

    def _fill_priors(self, priors: np.ndarray) -> None:
        """Write the static prior vector into every frame of each sequence."""
        for i, (X_seq, Y_seq, label) in enumerate(self.sequences):
            X_seq[:, 4:8] = priors[i]
            self.sequences[i] = (X_seq, Y_seq, label)

    # ── Dataset interface ────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return ``(X_seq, Y_seq)`` for trial *idx*.

        Shape assertions are enforced on every access.

        Returns:
            X_seq: ``(seq_len, 8)``
            Y_seq: ``(seq_len,)``
        """
        X_seq, Y_seq, _label = self.sequences[idx]

        X_tensor = torch.as_tensor(X_seq, dtype=torch.float32)
        Y_tensor = torch.as_tensor(Y_seq, dtype=torch.float32)

        seq_len = X_seq.shape[0]
        feat = self.feature_config.per_frame_total_dim  # 8

        # ── Shape assertions ──
        assert X_tensor.shape == (seq_len, feat), (
            f"[getitem idx={idx}] X_seq shape {X_tensor.shape} "
            f"!= expected ({seq_len}, {feat})"
        )
        assert Y_tensor.shape == (seq_len,), (
            f"[getitem idx={idx}] Y_seq shape {Y_tensor.shape} "
            f"!= expected ({seq_len},)"
        )

        # ── MCMC probability sanity ──
        mcmc_probs = X_tensor[:, 4:8]
        prob_sums = mcmc_probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(seq_len), atol=1e-5), (
            f"[getitem idx={idx}] MCMC probabilities do not sum to 1: "
            f"min={prob_sums.min():.6f}  max={prob_sums.max():.6f}"
        )

        return X_tensor, Y_tensor


# ──────────────────────────────────────────────────────────────
# Variable-length collate (returns lengths)
# ──────────────────────────────────────────────────────────────

def collate_variable_length(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad variable-length sequences to the max length in the batch.

    Returns:
        ``(X_batch, Y_batch, lengths)`` where

        * ``X_batch``: ``(batch_size, max_seq_len, 8)``
        * ``Y_batch``: ``(batch_size, max_seq_len)``
        * ``lengths``: ``(batch_size,)`` — true (unpadded) sequence
          lengths as ``int64``, suitable for
          ``torch.nn.utils.rnn.pack_padded_sequence``.
    """
    max_len = max(x.shape[0] for x, _y in batch)
    feat_dim = batch[0][0].shape[1]
    bs = len(batch)

    X_batch = torch.zeros(bs, max_len, feat_dim)
    Y_batch = torch.zeros(bs, max_len)
    lengths = torch.empty(bs, dtype=torch.int64)

    for i, (X_seq, Y_seq) in enumerate(batch):
        sl = X_seq.shape[0]
        X_batch[i, :sl, :] = X_seq
        Y_batch[i, :sl] = Y_seq
        lengths[i] = sl

    return X_batch, Y_batch, lengths


# ──────────────────────────────────────────────────────────────
# Factory (programmatic)
# ──────────────────────────────────────────────────────────────

def create_dataloader(
    sequences: List[Tuple[np.ndarray, np.ndarray, int]],
    mcmc_priors: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    feature_config: FeatureConfig = DEFAULT_FEATURE,
) -> DataLoader:
    """
    Create a :class:`~torch.utils.data.DataLoader` for NSMoR.

    Args:
        sequences: From :func:`data_extractor.build_sequence_dataset`.
        mcmc_priors: **Required** pre-computed ``(n, 4)`` prior matrix.
        batch_size: Batch size.
        shuffle: Shuffle trials each epoch.
        num_workers: Parallel data-loading workers.
        feature_config: Feature dimension constants.

    Returns:
        A ``DataLoader`` yielding ``(X_batch, Y_batch, lengths)`` tuples.
    """
    dataset = NSMoRDataset(
        sequences=sequences,
        mcmc_priors=mcmc_priors,
        feature_config=feature_config,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_variable_length,
    )


# ──────────────────────────────────────────────────────────────
# Dynamic dataset combination
# ──────────────────────────────────────────────────────────────

def combine_datasets(
    *dataset_parts: List[Tuple[np.ndarray, np.ndarray, int]],
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Concatenate multiple sequence lists into one.

    Useful for mixing pure-wind baseline datasets with looming datasets,
    or combining data from different experimental sessions.

    Args:
        *dataset_parts: Variable number of sequence lists, each as
            returned by :func:`data_extractor.build_sequence_dataset`.

    Returns:
        A single merged list of ``(X_seq, Y_seq, label)`` tuples.

    Example::

        looming_seqs = build_sequence_dataset(looming_trials)
        wind_seqs = build_sequence_dataset(wind_trials)
        combined = combine_datasets(looming_seqs, wind_seqs)
    """
    merged: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for part in dataset_parts:
        merged.extend(part)
    return merged


# ──────────────────────────────────────────────────────────────
# YAML-driven factory
# ──────────────────────────────────────────────────────────────

def create_dataloader_from_config(
    config: Any,  # ExperimentConfig — avoid circular import
    sequences: List[Tuple[np.ndarray, np.ndarray, int]],
    mcmc_priors: np.ndarray,
    split: str = "train",
    feature_config: FeatureConfig = DEFAULT_FEATURE,
) -> DataLoader:
    """
    Create a DataLoader from an :class:`~nsmor.config_parser.ExperimentConfig`.

    Reads ``batch_size``, ``shuffle``, and ``num_workers`` from the
    config's ``training`` section.  The *split* parameter selects
    which dataset section to use (for documentation / logging only —
    the caller passes the actual sequences).

    Args:
        config: An :class:`~nsmor.config_parser.ExperimentConfig` instance.
        sequences: Pre-assembled sequence list for this split.
        mcmc_priors: Pre-computed prior matrix.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        feature_config: Feature dimension constants.

    Returns:
        A ``DataLoader`` configured according to *config*.
    """
    shuffle = split == "train"
    return create_dataloader(
        sequences=sequences,
        mcmc_priors=mcmc_priors,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=0,
        feature_config=feature_config,
    )
