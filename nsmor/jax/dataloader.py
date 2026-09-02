"""
JAX DataLoader for NSMoR — High-Throughput Sequence Ingestion.

Compatible with PyTorch ETL preprocessed datasets (e.g. ``nsmor_dataset_3cond_v2.pt``).
Provides session-grouped train/val splitting identical to PyTorch pipelines,
automatic MCMC prior broadcasting into feature columns 4..8, and zero-overhead
batched slice generation into contiguous JAX arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

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

    data = torch.load(path, weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dataset dict, got {type(data)}")

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

    return converted


def session_grouped_train_val_split(
    session_ids: Sequence[Any],
    n_total: int,
    val_split: float = 0.2,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform deterministic session-grouped train/val split.

    Matches the exact logic in scripts/train.py:
    Trials belonging to the same recording session are kept entirely within
    either the training set or the validation set to prevent baseline leakage.
    """
    rng = np.random.RandomState(random_seed)
    if session_ids is not None and len(session_ids) == n_total:
        session_arr = np.asarray(session_ids)
        unique_sessions = np.unique(session_arr)
        rng.shuffle(unique_sessions)
        n_val_sessions = max(1, int(len(unique_sessions) * val_split))
        val_sessions = set(unique_sessions[:n_val_sessions].tolist())
        is_val = np.array([s in val_sessions for s in session_arr])
        val_indices = np.nonzero(is_val)[0]
        train_indices = np.nonzero(~is_val)[0]
    else:
        # Fallback to sample-level split
        logger.warning("Dataset lacks valid session_ids; falling back to sample shuffle")
        indices = np.arange(n_total)
        rng.shuffle(indices)
        n_val = max(1, int(n_total * val_split))
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

    return train_indices, val_indices


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
