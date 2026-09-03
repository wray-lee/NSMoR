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

from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

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
        max_seq_len: Optional[int] = 2400,
        pre_anchor_frames: int = 1200,
        anchor_frames: Optional[Sequence[int]] = None,
        source_indices: Optional[Sequence[int]] = None,
        is_pure_wind: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            sequences: List of ``(X_seq, Y_seq, label)`` tuples
                from :func:`data_extractor.build_sequence_dataset`.
            mcmc_priors: **Required** pre-computed probability vectors,
                shape ``(n_trials, 4)``.  Each row must sum to 1.
            feature_config: Feature dimension constants.
            max_seq_len: Maximum sequence length for cropping. If ``None``,
                no cropping is applied (full sequences). Default 2400 covers
                anchor + 2s response at 250 Hz.
            pre_anchor_frames: Number of frames before anchor to include
                in anchor-aligned crop (baseline window). Default 1200
                provides ~5s baseline at 250 Hz.
            anchor_frames: Optional frame index of stimulus anchor per trial.
                If provided, enables anchor-aligned cropping. If ``None``,
                falls back to legacy random crop (deprecated).
            source_indices: Optional provenance — the row index each
                sequence occupied in the *unsplit* dataset artifact.
                Callers that hand this dataset a train/val subset should
                pass the subset's original indices so the split is
                auditable without reverse-engineering it from tensor
                contents.  Defaults to ``range(len(sequences))``, i.e.
                identity, which is correct when no subsetting occurred.
                Exposed as :attr:`source_indices`; it does not affect
                ``__getitem__`` or any training behaviour.
            is_pure_wind: Optional per-trial boolean array indicating
                pure-wind trials (``True``) vs visual-present trials
                (``False``). Shape ``(n_trials,)``. If provided, enables
                the auxiliary routing loss (Ticket #16). Defaults to
                ``None`` (no condition metadata).

        Raises:
            ValueError: If *mcmc_priors* is ``None`` or its shape does
                not match the number of sequences, or if
                *source_indices* is given with a mismatched length.
        """
        if mcmc_priors is None:
            raise ValueError(
                "mcmc_priors is required.  Run the MCMC module upstream "
                "and pass the resulting (n, 4) probability matrix.  "
                "Dynamic inference inside the DataLoader is not supported."
            )

        self.feature_config = feature_config
        self.max_seq_len = max_seq_len
        self.pre_anchor_frames = pre_anchor_frames
        self.anchor_frames = anchor_frames
        # Deep-copy each (X_seq, Y_seq, label) tuple so that _fill_priors
        # writes into private arrays and never corrupts the caller's data.
        self.sequences = [
            (X.copy(), Y.copy(), lbl) for X, Y, lbl in sequences
        ]

        # Split provenance.  Read-only sidecar: the tuple layout of
        # ``self.sequences`` is deliberately unchanged so ``__getitem__``
        # and ``_fill_priors`` keep their exact contracts.
        if source_indices is None:
            self.source_indices: List[int] = list(range(len(self.sequences)))
        else:
            self.source_indices = [int(i) for i in source_indices]
            if len(self.source_indices) != len(self.sequences):
                raise ValueError(
                    f"source_indices length {len(self.source_indices)} does "
                    f"not match sequence count {len(self.sequences)}."
                )

        # Ticket #16: Stimulus condition metadata for routing aux loss.
        # Store as instance variable; accessed by custom collate function.
        self.is_pure_wind: Optional[np.ndarray] = None
        if is_pure_wind is not None:
            if len(is_pure_wind) != len(self.sequences):
                raise ValueError(
                    f"is_pure_wind length {len(is_pure_wind)} does not match "
                    f"sequence count {len(self.sequences)}."
                )
            self.is_pure_wind = np.asarray(is_pure_wind, dtype=bool)

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
        """Write the static prior vector into every frame of each sequence.

        Each row of *priors* must sum to 1 (valid probability simplex).
        The method writes into the deep-copied ``self.sequences`` arrays,
        so the caller's original data is never mutated.
        """
        for i, (X_seq, Y_seq, label) in enumerate(self.sequences):
            # Guard: detect if MCMC columns already hold a valid probability
            # simplex (every row sums to ~1), which indicates a double-fill
            # bug or caller passed pre-filled sequences.
            existing_row0 = X_seq[0, 4:8]
            existing_sum = float(existing_row0.sum())
            if (
                np.all(existing_row0 >= 0.0)
                and abs(existing_sum - 1.0) < 1e-4
            ):
                raise ValueError(
                    f"[_fill_priors] trial {i}: MCMC columns (4:8) already "
                    f"contain a valid probability simplex "
                    f"(row-0 sum={existing_sum:.6f}).  Source sequences may "
                    f"have been mutated by a previous DataLoader — ensure a "
                    f"fresh deep copy."
                )
            prior_vec = priors[i]
            prior_sum = float(prior_vec.sum())
            if abs(prior_sum - 1.0) > 1e-4:
                raise ValueError(
                    f"[_fill_priors] trial {i}: prior sum={prior_sum:.6f} "
                    f"deviates from 1.0 by {abs(prior_sum - 1.0):.6f} "
                    f"(tolerance=1e-4).  Upstream MCMC priors may be corrupt."
                )
            X_seq[:, 4:8] = prior_vec
            self.sequences[i] = (X_seq, Y_seq, label)

    # ── Dataset interface ────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, torch.Tensor, torch.Tensor]]:
        """
        Return ``(X_seq, Y_seq)`` or ``(idx, X_seq, Y_seq)`` for trial *idx*.

        Applies anchor-aligned cropping if ``anchor_frames`` was provided;
        otherwise falls back to legacy random crop (deprecated).

        Shape assertions are enforced on every access.

        Returns:
            If ``is_pure_wind`` is ``None`` (legacy mode):
                ``(X_seq, Y_seq)`` — 2-tuple.

            If ``is_pure_wind`` is present (Ticket #16):
                ``(idx, X_seq, Y_seq)`` — 3-tuple, where ``idx`` is used
                by ``collate_with_metadata`` to attach condition metadata.

            X_seq: ``(seq_len, 8)``
            Y_seq: ``(seq_len,)``
        """
        X_seq, Y_seq, _label = self.sequences[idx]

        # ── Crop long sequences ──
        if self.max_seq_len is not None and X_seq.shape[0] > self.max_seq_len:
            if self.anchor_frames is not None:
                # Anchor-aligned crop (preserves stimulus + response)
                anchor_frame = self.anchor_frames[idx]
                start = max(0, anchor_frame - self.pre_anchor_frames)
                end = min(X_seq.shape[0], start + self.max_seq_len)

                # Adjust start if end clamped (keeps window size consistent)
                if end - start < self.max_seq_len:
                    start = max(0, end - self.max_seq_len)

                X_seq = X_seq[start:end]
                Y_seq = Y_seq[start:end]
            else:
                # Legacy random crop (deprecated — low stimulus capture rate)
                start = np.random.randint(0, X_seq.shape[0] - self.max_seq_len + 1)
                X_seq = X_seq[start : start + self.max_seq_len]
                Y_seq = Y_seq[start : start + self.max_seq_len]

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

        # ── Return format based on metadata availability ──
        # Ticket #16: If is_pure_wind is present, return (idx, X, Y) for
        # collate_with_metadata to attach condition mask. Otherwise return
        # (X, Y) for backward compatibility.
        if self.is_pure_wind is not None:
            return idx, X_tensor, Y_tensor
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


def collate_with_metadata(
    batch: List[Tuple[int, torch.Tensor, torch.Tensor]],
    is_pure_wind: Optional[np.ndarray] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Pad variable-length sequences and attach stimulus condition metadata.

    Ticket #16: Extended collate function that returns ``wind_only_mask``
    when ``is_pure_wind`` metadata is available.

    Args:
        batch: List of ``(idx, X_seq, Y_seq)`` tuples from dataset.
        is_pure_wind: Optional boolean array indicating pure-wind trials.
            If ``None``, falls back to legacy 3-tuple return.

    Returns:
        If ``is_pure_wind`` is ``None``:
            ``(X_batch, Y_batch, lengths)`` — legacy 3-tuple.

        Otherwise:
            ``(X_batch, Y_batch, lengths, wind_only_mask)`` where
            ``wind_only_mask`` is a boolean tensor of shape ``(batch_size,)``.
    """
    # Unpack (idx, X_seq, Y_seq) and build index list
    indices = [item[0] for item in batch]
    X_Y_batch = [(item[1], item[2]) for item in batch]

    # Standard padding
    max_len = max(x.shape[0] for x, _y in X_Y_batch)
    feat_dim = X_Y_batch[0][0].shape[1]
    bs = len(X_Y_batch)

    X_batch = torch.zeros(bs, max_len, feat_dim)
    Y_batch = torch.zeros(bs, max_len)
    lengths = torch.empty(bs, dtype=torch.int64)

    for i, (X_seq, Y_seq) in enumerate(X_Y_batch):
        sl = X_seq.shape[0]
        X_batch[i, :sl, :] = X_seq
        Y_batch[i, :sl] = Y_seq
        lengths[i] = sl

    # Attach metadata if available
    if is_pure_wind is not None:
        wind_only_mask = torch.tensor(
            [is_pure_wind[idx] for idx in indices],
            dtype=torch.bool,
        )
        return X_batch, Y_batch, lengths, wind_only_mask

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
    max_seq_len: Optional[int] = None,
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
        max_seq_len: If set, crop sequences longer than this to a
            random window of this length (data augmentation).
            Recommended for cuDNN compatibility with very long sequences.

    Returns:
        A ``DataLoader`` yielding ``(X_batch, Y_batch, lengths)`` tuples.
    """
    dataset = NSMoRDataset(
        sequences=sequences,
        mcmc_priors=mcmc_priors,
        feature_config=feature_config,
        max_seq_len=max_seq_len,
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
