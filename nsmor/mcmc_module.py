"""
MCMC Prior Generator — snapshot-to-probability mapping.

Provides three complementary implementations:

1. :class:`MCMCPriorGenerator` — PyTorch ``nn.Module``
   (softmax regression, trainable end-to-end).
2. :class:`MCMCPriorSKLearn` — scikit-learn
   ``LogisticRegression`` wrapper for quick prototyping.
3. :class:`MarkovTransitionEstimator` — discrete first-order
   Markov transition matrix from label sequences.

The primary training entry-point is :func:`train_mcmc`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nsmor.config import (
    DEFAULT_FEATURE,
    DEFAULT_MCMC_TRAINING,
    FeatureConfig,
    MCMCTrainingConfig,
)


# ═══════════════════════════════════════════════════════════════
# 1.  PyTorch MCMC Prior Generator
# ═══════════════════════════════════════════════════════════════

class MCMCPriorGenerator(nn.Module):
    """
    Lightweight MCMC prior generator (softmax regression).

    Architecture
    ------------
    ``Linear(snapshot_dim, num_classes)`` → softmax

    Input:  ``(batch, 5)`` snapshot feature vector
    Output: ``(batch, 4)`` probability vector that sums to 1

        ``P = [P_startle, P_walk, P_pre_active, P_no_response]``
    """

    def __init__(
        self,
        snapshot_dim: int = DEFAULT_FEATURE.snapshot_dim,
        num_classes: int = DEFAULT_FEATURE.num_classes,
    ) -> None:
        super().__init__()
        self.snapshot_dim = snapshot_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(snapshot_dim, num_classes)
        # Xavier init for stable initial softmax outputs
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # ── Forward ──────────────────────────────────────────────

    def forward(self, snapshot: torch.Tensor) -> torch.Tensor:
        """
        Snapshot → probability vector.

        Args:
            snapshot: ``(batch, 5)`` or ``(5,)``

        Returns:
            ``(batch, 4)`` or ``(4,)`` — probabilities, sum = 1
        """
        logits = self.classifier(snapshot)
        return F.softmax(logits, dim=-1)

    def get_logits(self, snapshot: torch.Tensor) -> torch.Tensor:
        """Raw logits (pre-softmax).  Use with ``CrossEntropyLoss``."""
        return self.classifier(snapshot)

    # ── Numpy convenience ────────────────────────────────────

    def predict_proba(self, snapshot: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities from a NumPy array.

        Args:
            snapshot: ``(5,)`` or ``(n, 5)``

        Returns:
            ``(4,)`` or ``(n, 4)`` — probabilities, each row sums to 1.
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(snapshot, dtype=torch.float32)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            probs = self.forward(tensor)
            return probs.squeeze(0).cpu().numpy()


# ═══════════════════════════════════════════════════════════════
# 2.  Scikit-learn wrapper
# ═══════════════════════════════════════════════════════════════

class MCMCPriorSKLearn:
    """
    Scikit-learn multinomial logistic-regression wrapper.

    Provides the same ``predict_proba`` interface as
    :class:`MCMCPriorGenerator` for seamless interchange.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_FEATURE.num_classes,
        random_state: int = DEFAULT_MCMC_TRAINING.random_seed,
    ) -> None:
        from sklearn.linear_model import LogisticRegression

        self.num_classes = num_classes
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
        self._is_fitted: bool = False

    def fit(
        self, snapshots: np.ndarray, labels: np.ndarray,
    ) -> "MCMCPriorSKLearn":
        """
        Train on snapshot features + discrete labels.

        Args:
            snapshots: ``(n, 5)``
            labels: ``(n,)`` integer class labels

        Returns:
            *self* (for method chaining).
        """
        self.model.fit(snapshots, labels)
        self._is_fitted = True
        return self

    def predict_proba(self, snapshot: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            snapshot: ``(5,)`` or ``(n, 5)``

        Returns:
            ``(4,)`` or ``(n, 4)`` — probabilities, each row sums to 1.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        single = snapshot.ndim == 1
        if single:
            snapshot = snapshot.reshape(1, -1)
        probs = self.model.predict_proba(snapshot)
        return probs.squeeze(0) if single else probs


# ═══════════════════════════════════════════════════════════════
# 3.  Markov Transition Estimator
# ═══════════════════════════════════════════════════════════════

class MarkovTransitionEstimator:
    """
    First-order discrete Markov transition matrix estimator.

    Estimates ``P(next_state | current_state)`` from observed label
    sequences.  Useful for modelling temporal dependencies in
    behavioural state transitions.
    """

    def __init__(self, num_states: int = DEFAULT_FEATURE.num_classes) -> None:
        self.num_states = num_states
        self.transition_matrix: Optional[np.ndarray] = None

    def fit(self, label_sequences: List[np.ndarray]) -> "MarkovTransitionEstimator":
        """
        Estimate the transition matrix from one or more label sequences.

        Args:
            label_sequences: List of 1-D integer label arrays.

        Returns:
            *self*.
        """
        counts = np.zeros(
            (self.num_states, self.num_states), dtype=np.float64,
        )
        for seq in label_sequences:
            for i in range(len(seq) - 1):
                s_from = int(seq[i])
                s_to = int(seq[i + 1])
                if 0 <= s_from < self.num_states and 0 <= s_to < self.num_states:
                    counts[s_from, s_to] += 1

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        self.transition_matrix = counts / row_sums
        return self

    def predict_proba(self, current_state: int) -> np.ndarray:
        """
        Next-state distribution given *current_state*.

        Args:
            current_state: Integer label.

        Returns:
            ``(num_states,)`` probability vector.
        """
        if self.transition_matrix is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self.transition_matrix[current_state]


# ═══════════════════════════════════════════════════════════════
# 4.  Training entry-point
# ═══════════════════════════════════════════════════════════════

def train_mcmc(
    snapshots: np.ndarray,
    labels: np.ndarray,
    config: MCMCTrainingConfig = DEFAULT_MCMC_TRAINING,
    feature_config: FeatureConfig = DEFAULT_FEATURE,
    verbose: bool = True,
) -> MCMCPriorGenerator:
    """
    Train the PyTorch MCMC prior generator via cross-entropy.

    Args:
        snapshots: ``(n_trials, 5)`` snapshot feature matrix.
        labels: ``(n_trials,)`` integer ground truth labels.
        config: Training hyperparameters.
        feature_config: Feature dimension constants.
        verbose: Print loss every 50 epochs.

    Returns:
        Trained :class:`MCMCPriorGenerator` in eval mode.

    Raises:
        ValueError: If snapshot / label counts differ.
    """
    if snapshots.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Count mismatch: {snapshots.shape[0]} snapshots vs "
            f"{labels.shape[0]} labels."
        )

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    model = MCMCPriorGenerator(
        snapshot_dim=feature_config.snapshot_dim,
        num_classes=feature_config.num_classes,
    )

    X = torch.as_tensor(snapshots, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(X, y), batch_size=config.batch_size, shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    prev_loss = float("inf")

    for epoch in range(1, config.num_epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model.get_logits(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if verbose and epoch % 50 == 0:
            print(f"[MCMC] epoch {epoch:>4d}/{config.num_epochs}  "
                  f"loss={avg_loss:.6f}")

        if abs(prev_loss - avg_loss) < config.convergence_tol:
            if verbose:
                print(f"[MCMC] converged at epoch {epoch}")
            break
        prev_loss = avg_loss

    model.eval()
    return model
