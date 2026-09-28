"""
JAX DataLoader for NSMoR — High-Throughput Sequence Ingestion.

Reads provenance-checked PyTorch ETL datasets. The shared splitter groups
recording prefixes; distinct prefixes do not verify distinct animals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from nsmor.model_utils import validate_dataset_provenance
from nsmor.pipeline.grouping import grouped_train_val_split
from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

logger = logging.getLogger("nsmor.jax.dataloader")


def load_nsmor_dataset(dataset_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load preprocessed NSMoR dataset from a PyTorch .pt artifact.

    Args:
        dataset_path: Path to nsmor_dataset_*.pt.

    Returns:
        Dataset dictionary with keys:
        'X_seqs', 'Y_seqs', 'mcmc_priors', 'lengths', 'session_ids', 'labels'.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    data, fingerprint = load_dataset_with_fingerprint(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Expected dataset dict, got {type(data)}")
    required = ("X_seqs", "Y_seqs", "mcmc_priors", "lengths", "labels")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Dataset missing required keys: {missing}")
    identity_status = validate_dataset_provenance(data, path)
    n_total = len(data["X_seqs"])
    sessions = data.get("session_ids")
    if sessions is None and isinstance(data.get("trial_specs"), (list, tuple)):
        sessions = [spec.get("session_id") if isinstance(spec, dict) else None
                    for spec in data["trial_specs"]]
    if (not isinstance(sessions, (list, tuple, np.ndarray))
            or len(sessions) != n_total
            or any(not isinstance(s, str) or not s.strip()
                   or s.strip().lower() in ("nan", "none") for s in sessions)):
        raise ValueError("session_ids must contain one usable recording id per trial")
    priors = np.asarray(data["mcmc_priors"])
    if (priors.shape != (n_total, 4) or not np.isfinite(priors).all()
            or np.any(priors < 0) or np.any(priors > 1)
            or not np.allclose(priors.sum(axis=1), 1.0, atol=1e-4)):
        raise ValueError(f"mcmc_priors must be finite ({n_total}, 4) probability rows")

    # Convert torch tensors to numpy arrays if necessary
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    converted: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, list):
            converted[k] = [_to_np(item) for item in v]
        else:
            converted[k] = _to_np(v)

    converted["session_ids"] = sessions
    converted["animal_identity_status"] = identity_status
    converted["dataset_sha256"] = fingerprint
    return converted


def recording_prefix_train_val_split(
    session_ids: Sequence[Any],
    n_total: int,
    val_split: float = 0.2,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by recording prefix; animal identity remains unverified."""
    return grouped_train_val_split(
        session_ids,
        n_total,
        val_split=val_split,
        random_seed=random_seed,
    )


# Historical public name is an interface alias, not an animal-level claim.
session_grouped_train_val_split = recording_prefix_train_val_split


def compute_target_stats(
    Y_seqs: Sequence[np.ndarray],
    train_indices: np.ndarray,
    lengths: np.ndarray,
    target_clip_cm_s: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute velocity mean and std over training sequences only.

    Args:
        Y_seqs: List of target velocity 1-D arrays.
        train_indices: Indices of training trials.
        lengths: Array of valid lengths for each trial.
        target_clip_cm_s: If > 0, clip outlier frames before computing stats.

    Returns:
        (target_mean, target_std)
    """
    train_frames = []
    for idx in train_indices:
        seq = Y_seqs[idx]
        valid_len = int(lengths[idx])
        y = seq[:valid_len]
        if target_clip_cm_s > 0.0:
            y = np.clip(y, -target_clip_cm_s, target_clip_cm_s)
        train_frames.append(y)

    all_y = np.concatenate(train_frames, axis=0)
    target_mean = float(np.mean(all_y))
    target_std = float(np.std(all_y))
    if target_std < 1e-6:
        target_std = 1.0
    return target_mean, target_std


class JAXDataset:
    """
    Padded in-memory array container for NSMoR training sequences.

    Pre-fills static MCMC priors into columns 4..8 and pads all sequences
    to a fixed `max_seq_len` (e.g. 2400 frames). This gives static tensor
    shapes across all batches, allowing XLA to compile the recurrent loop
    exactly once without dynamic shape recompilations.
    """

    def __init__(
        self,
        X_seqs: Sequence[np.ndarray],
        Y_seqs: Sequence[np.ndarray],
        mcmc_priors: np.ndarray,
        indices: Optional[Sequence[int]] = None,
        max_seq_len: int = 2400,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        normalize_targets: bool = False,
        target_clip_cm_s: float = 0.0,
        anchor_frames: Optional[Sequence[int]] = None,
        pre_anchor_frames: int = 1200,
    ) -> None:
        if indices is None:
            indices = list(range(len(X_seqs)))

        N = len(indices)
        self.N = N
        self.max_seq_len = max_seq_len

        # Preallocate contiguous numpy arrays
        self.X = np.zeros((N, max_seq_len, 8), dtype=np.float32)
        self.Y = np.zeros((N, max_seq_len), dtype=np.float32)
        self.lengths = np.zeros((N,), dtype=np.int32)

        for i, idx in enumerate(indices):
            x_seq = np.asarray(X_seqs[idx], dtype=np.float32)
            y_seq = np.asarray(Y_seqs[idx], dtype=np.float32).ravel()
            prior = np.asarray(mcmc_priors[idx], dtype=np.float32)

            orig_len = len(y_seq)

            # MAJOR-1 fix: Anchor-aligned cropping matching PyTorch
            # nsmor_dataloader.py:202-220.  When anchor_frames is provided
            # and the sequence exceeds max_seq_len, crop around the anchor
            # to guarantee stimulus capture.
            if orig_len > max_seq_len and anchor_frames is not None:
                anchor_frame = int(anchor_frames[idx])
                start = max(0, anchor_frame - pre_anchor_frames)
                end = min(orig_len, start + max_seq_len)
                # Adjust start if end clamped (keeps window size consistent)
                if end - start < max_seq_len:
                    start = max(0, end - max_seq_len)
                x_seq = x_seq[start:end]
                y_seq = y_seq[start:end]
                orig_len = len(y_seq)

            actual_len = min(orig_len, max_seq_len)
            self.lengths[i] = actual_len

            # Fill sensory (cols 0..3) and prior (cols 4..7)
            if x_seq.shape[1] >= 8:
                self.X[i, :actual_len, :4] = x_seq[:actual_len, :4]
            else:
                self.X[i, :actual_len, :x_seq.shape[1]] = x_seq[:actual_len, :]

            self.X[i, :actual_len, 4:8] = prior[None, :]

            # Process targets
            y_proc = y_seq[:actual_len]
            if target_clip_cm_s > 0.0:
                y_proc = np.clip(y_proc, -target_clip_cm_s, target_clip_cm_s)
            if normalize_targets:
                y_proc = (y_proc - target_mean) / target_std

            self.Y[i, :actual_len] = y_proc

    def __len__(self) -> int:
        return self.N


class JAXDataLoader:
    """
    High-throughput zero-copy batch iterator for JAX.

    Yields batches of `(X_batch, Y_batch, lengths_batch)` as `jax.Array`
    with uniform shape `(B, max_seq_len, 8)`.
    """

    def __init__(
        self,
        dataset: JAXDataset,
        batch_size: int = 128,
        shuffle: bool = True,
        seed: int = 42,
        pad_last_batch: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pad_last_batch = pad_last_batch
        self.rng = np.random.RandomState(seed)

        N = len(dataset)
        if pad_last_batch and N % batch_size != 0:
            self.n_batches = (N + batch_size - 1) // batch_size
        else:
            self.n_batches = N // batch_size if not pad_last_batch else (N + batch_size - 1) // batch_size

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        N = len(self.dataset)
        indices = np.arange(N)
        if self.shuffle:
            self.rng.shuffle(indices)

        for b in range(self.n_batches):
            start = b * self.batch_size
            end = min(start + self.batch_size, N)
            batch_idx = indices[start:end]

            curr_bs = len(batch_idx)
            if curr_bs == self.batch_size:
                x_b = self.dataset.X[batch_idx]
                y_b = self.dataset.Y[batch_idx]
                l_b = self.dataset.lengths[batch_idx]
            elif self.pad_last_batch:
                # Pad to uniform batch_size with dummy items (length=0 so loss ignores them)
                x_b = np.zeros((self.batch_size, self.dataset.max_seq_len, 8), dtype=np.float32)
                y_b = np.zeros((self.batch_size, self.dataset.max_seq_len), dtype=np.float32)
                l_b = np.zeros((self.batch_size,), dtype=np.int32)

                x_b[:curr_bs] = self.dataset.X[batch_idx]
                y_b[:curr_bs] = self.dataset.Y[batch_idx]
                l_b[:curr_bs] = self.dataset.lengths[batch_idx]
            else:
                x_b = self.dataset.X[batch_idx]
                y_b = self.dataset.Y[batch_idx]
                l_b = self.dataset.lengths[batch_idx]

            if JAX_AVAILABLE:
                yield jnp.asarray(x_b), jnp.asarray(y_b), jnp.asarray(l_b)
            else:
                yield x_b, y_b, l_b
