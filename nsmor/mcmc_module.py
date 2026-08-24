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

from typing import Dict, List, Optional, Tuple, Union

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


# ═══════════════════════════════════════════════════════════════
# 2b.  Cross-fitted (out-of-fold) prior generation
# ═══════════════════════════════════════════════════════════════

def train_mcmc_cross_fitted(
    snapshots: np.ndarray,
    labels: np.ndarray,
    config: MCMCTrainingConfig = DEFAULT_MCMC_TRAINING,
    feature_config: FeatureConfig = DEFAULT_FEATURE,
    n_folds: int = 5,
    groups: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[MCMCPriorGenerator], List[Dict]]:
    """
    Generate **out-of-fold (OOF)** MCMC priors via K-fold cross-fitting.

    Reviewer Round-1 BLOCKER-2 fix: the previous pipeline trained the
    prior generator on ALL snapshots and then predicted on the SAME
    snapshots, leaking ground-truth labels into the NSMoR input
    features.  Cross-fitting guarantees that each snapshot's prior is
    produced by a model that NEVER saw that snapshot's label:

      1. Partition trials into ``n_folds`` folds.
      2. For each fold, train on the remaining folds.
      3. Predict priors for the held-out fold only.

    Round-2 fix (Reviewer B B-2): when ``groups`` is provided (e.g.
    ``session_id`` per trial), the split uses ``StratifiedGroupKFold``
    so that all trials from one recording session stay on the SAME
    side of the split.  Trials within a session share the animal's
    baseline locomotor statistics and gain state; a sample-level
    shuffle lets the same session straddle train/test folds and leaks
    session-level information into the "held-out" priors.  Without
    ``groups``, stratified sample-level splits are used (only valid
    for synthetic / single-session data).

    The returned OOF matrix can be safely used as downstream input
    features (no same-sample label leakage), while the fold models are
    also returned so held-out data at inference time can use an
    ensemble of them.

    Inference-time protocol (Reviewer B M-3): for genuinely new data,
    predict with every fold model and average the probability rows,
    then renormalise to sum to 1::

        probs = np.mean([m.predict_proba(x) for m in fold_models], axis=0)
        probs = probs / probs.sum(axis=1, keepdims=True)

    Args:
        snapshots: ``(n_trials, 5)`` snapshot feature matrix.
        labels: ``(n_trials,)`` integer ground truth labels.
        config: Training hyperparameters.
        feature_config: Feature dimension constants.
        n_folds: Number of cross-fitting folds (>= 2).
        groups: Optional ``(n_trials,)`` session/group ids; enables
            session-level group splitting (recommended).
        verbose: Print per-fold progress.

    Returns:
        ``(oof_priors, fold_models, fold_diagnostics)`` where
        - ``oof_priors``: ``(n_trials, 4)`` out-of-fold probabilities;
          every row was produced by a model trained WITHOUT its label
          or any other trial from its session (when grouped).
        - ``fold_models``: list of the ``n_folds`` trained generators
          (for ensembling on genuinely new data).
        - ``fold_diagnostics``: per-fold composition records (fold id,
          session counts on both sides, class histograms) — Round-3
          MAJ-3C requires these to be persisted so fold imbalance is
          auditable in saved artefacts.

    Raises:
        ValueError: If counts mismatch or ``n_folds < 2`` or the
            stratification constraint cannot be satisfied.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if snapshots.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Count mismatch: {snapshots.shape[0]} snapshots vs "
            f"{labels.shape[0]} labels."
        )
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if groups is not None and len(groups) != snapshots.shape[0]:
        raise ValueError(
            f"groups length {len(groups)} != "
            f"snapshots count {snapshots.shape[0]}."
        )

    # Stratification requires enough members per class.  With groups,
    # StratifiedGroupKFold additionally needs each class to appear in
    # enough distinct sessions to populate every fold's training side.
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    if groups is None:
        for cls, cnt in zip(unique_classes, class_counts):
            if int(cnt) < n_folds:
                raise ValueError(
                    f"Class {cls} has only {cnt} samples; "
                    f"stratified {n_folds}-fold split is impossible."
                )

    oof_priors = np.zeros(
        (snapshots.shape[0], feature_config.num_classes), dtype=np.float64,
    )
    fold_models: List[MCMCPriorGenerator] = []
    fold_diagnostics: List[Dict] = []

    if groups is not None:
        # Round-3 fix (Reviewer B CRITICAL-3a): StratifiedGroupKFold does
        # NOT guarantee that every class appears in every training side.
        # When a behaviour class lives in only a handful of sessions, an
        # n_folds=5 split can leave some folds' training side without
        # that class entirely — sklearn only warns, the fold model then
        # never predicts it, and the corresponding OOF prior column
        # degenerates toward zero for held-out trials.  The pipeline's
        # own provenance philosophy ("reject rather than degrade
        # silently") demands a hard check, so we pre-compute the group-
        # level composition and verify EVERY class × EVERY fold's
        # training-side coverage BEFORE any model is trained.
        group_arr = np.asarray(groups)
        session_classes: Dict = {}
        for g in np.unique(group_arr):
            session_classes[g] = set(
                np.unique(labels[group_arr == g]).tolist()
            )
        for cls in unique_classes:
            n_sessions_with_cls = sum(
                1 for cl in session_classes.values() if cls in cl
            )
            if n_sessions_with_cls < n_folds:
                raise ValueError(
                    f"Class {cls} appears in only {n_sessions_with_cls} "
                    f"session(s); grouped {n_folds}-fold cross-fitting "
                    f"requires at least {n_folds} sessions containing "
                    f"each class so every fold's training side covers "
                    f"it.  Merge sessions or reduce n_folds."
                )

        splitter = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=config.random_seed,
        )
        split_iter = splitter.split(snapshots, labels, groups=groups)
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=config.random_seed)
        split_iter = skf.split(snapshots, labels)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        if verbose:
            print(f"[MCMC-CV] fold {fold_idx + 1}/{n_folds}  "
                  f"(train={len(train_idx)}, oof={len(test_idx)})")

        # Round-3 (CRITICAL-3a): post-split hard assertion — every class
        # must be present on the TRAINING side of every fold.  The
        # pre-split session-count guard above makes this near-certain
        # but not provable (StratifiedGroupKFold assignment heuristics);
        # this closes the gap.
        train_classes_fold = np.unique(labels[train_idx])
        missing = set(unique_classes.tolist()) - set(train_classes_fold.tolist())
        if groups is not None and missing:
            raise ValueError(
                f"Fold {fold_idx}: classes {sorted(missing)} absent from "
                f"the training side ({len(train_idx)} trials from "
                f"{len(np.unique(np.asarray(groups)[train_idx]))} "
                f"sessions).  OOF priors would be degenerate; refusing "
                f"to continue."
            )

        fold_model = train_mcmc(
            snapshots[train_idx], labels[train_idx],
            config=config, feature_config=feature_config, verbose=False,
        )
        oof_priors[test_idx] = fold_model.predict_proba(snapshots[test_idx])
        fold_models.append(fold_model)

        # Round-3 (Reviewer A MAJ-3C): persist per-fold composition so
        # imbalance is auditable in saved artefacts.
        if groups is not None:
            g_arr = np.asarray(groups)
            fold_diagnostics.append({
                "fold": int(fold_idx),
                "n_train_sessions": int(np.unique(g_arr[train_idx]).size),
                "n_oof_sessions": int(np.unique(g_arr[test_idx]).size),
                "train_classes": labels[train_idx].astype(int).tolist(),
                "oof_classes": labels[test_idx].astype(int).tolist(),
            })
        else:
            fold_diagnostics.append({
                "fold": int(fold_idx),
                "n_train_sessions": -1,
                "n_oof_sessions": -1,
                "train_classes": labels[train_idx].astype(int).tolist(),
                "oof_classes": labels[test_idx].astype(int).tolist(),
            })

    assert np.isfinite(oof_priors).all(), "OOF priors contain non-finite values"
    assert abs(oof_priors.sum(axis=1) - 1.0).max() < 1e-4, (
        "OOF prior rows must sum to 1"
    )

    return oof_priors, fold_models, fold_diagnostics

