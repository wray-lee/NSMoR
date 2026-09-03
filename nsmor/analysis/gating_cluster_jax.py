"""JAX-Accelerated Gating Cluster Analysis.

Corresponds to :mod:`nsmor.analysis.gating_cluster` (PyTorch version).

Acceleration strategy:
  - Model inference via :class:`nsmor.jax.model.NSMoRModel` under ``jax.jit``.
  - Trial-level fingerprint extraction vectorized with ``jax.vmap``.
  - KMeans / UMAP remain sklearn (not bottleneck; no native JAX impl).

All public functions match the PyTorch API; outputs are numerically
compatible (same 16-dim fingerprint, same clustering contract).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    JAX_AVAILABLE = False

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from nsmor.analysis.gating_cluster import (
    ClusterGatingConfig,
    GatingClusterAdapter as _PyTorchGatingClusterAdapter,
    interpolate_for_viz,
)

if JAX_AVAILABLE:
    from nsmor.jax.model import NSMoRModel

logger = logging.getLogger(__name__)


# ===============================================================
# Pure-JAX Fingerprint Kernels (vmap-friendly)
# ===============================================================

if JAX_AVAILABLE:

    @jax.jit
    def _gate_stats_single(gate: jnp.ndarray) -> jnp.ndarray:
        """Compute 6-dim statistics for a single gate channel.

        Features: [mean, std, max, min, dominant_fraction, 0.0]
        Entropy (idx 5) is computed in NumPy post-hoc because the
        binned histogram estimator is not XLA-friendly.

        Args:
            gate: 1-D array of shape (T,).

        Returns:
            jnp.ndarray of shape (6,).
        """
        mean_val = jnp.mean(gate)
        std_val = jnp.std(gate, ddof=1)
        max_val = jnp.max(gate)
        min_val = jnp.min(gate)
        dominant_frac = jnp.mean((gate > 0.5).astype(jnp.float32))
        # Placeholder for entropy; filled in post-hoc
        return jnp.array([mean_val, std_val, max_val, min_val, dominant_frac, 0.0])

    def _fingerprint_single_jax(
        gates: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute 16-dim fingerprint from a (T, 2) gate array.

        The entropy slot (indices 5 and 11) is left as 0.0;
        the caller patches them in NumPy.

        Args:
            gates: (T, 2) gate sequence.

        Returns:
            (16,) fingerprint vector.
        """
        T = gates.shape[0]
        g_lif = gates[:, 0]
        g_gru = gates[:, 1]

        feat_lif = _gate_stats_single(g_lif)  # (6,)
        feat_gru = _gate_stats_single(g_gru)  # (6,)

        # Pearson correlation (NaN-safe)
        std_l = jnp.std(g_lif, ddof=1)
        std_g = jnp.std(g_gru, ddof=1)
        safe = (std_l > 1e-12) & (std_g > 1e-12)
        corr_raw = jnp.corrcoef(g_lif, g_gru)[0, 1]
        corr = jnp.where(safe & jnp.isfinite(corr_raw), corr_raw, 0.0)

        T_f = T.astype(jnp.float32) if hasattr(T, 'astype') else jnp.float32(T)
        t_max = jnp.argmax(g_lif).astype(jnp.float32) / jnp.maximum(T_f, 1.0)
        t_min = jnp.argmin(g_lif).astype(jnp.float32) / jnp.maximum(T_f, 1.0)

        max_grad = jnp.where(
            T_f > 1,
            jnp.max(jnp.abs(jnp.diff(g_lif))),
            0.0,
        )

        return jnp.concatenate([
            feat_lif,
            feat_gru,
            jnp.array([corr]),
            jnp.array([t_max, t_min]),
            jnp.array([max_grad]),
        ])  # (16,)


# ===============================================================
# GatingClusterAdapterJAX
# ===============================================================

class GatingClusterAdapterJAX:
    """JAX-accelerated adapter for gating cluster analysis.

    Uses :class:`NSMoRModel` for inference, ``jax.vmap`` / ``jax.jit``
    for trial-level fingerprint extraction, and delegates clustering
    and evaluation to the PyTorch adapter's sklearn-based routines
    (they operate on NumPy arrays only).

    Attributes:
        model: NSMoRModel Flax instance.
        params: Frozen Flax parameter PyTree.
        config: ClusterGatingConfig.
    """

    def __init__(
        self,
        model: "NSMoRModel",
        params: Dict[str, Any],
        config: Optional[ClusterGatingConfig] = None,
    ) -> None:
        """Initialize the JAX gating cluster adapter.

        Args:
            model: Flax NSMoRModel instance.
            params: Flax parameter PyTree (e.g. from ``model.init`` or
                ``load_from_torch_state_dict``).
            config: ClusterGatingConfig (uses defaults if None).
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is required for GatingClusterAdapterJAX.")

        self.model = model
        self.params = params
        self.config = config if config is not None else ClusterGatingConfig()

        # Pre-compile the forward pass
        self._apply_jit = jax.jit(
            lambda p, x, lengths: model.apply(
                p, x, lengths, deterministic=True, return_internals=True,
            )
        )

    def extract_gating_sequences(
        self,
        X_batches: List[jnp.ndarray],
        lengths_batches: List[jnp.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Extract per-trial gating sequences via JIT-compiled forward pass.

        Args:
            X_batches: List of (B_i, T, 8) input arrays.
            lengths_batches: Corresponding (B_i,) length arrays.
            labels: Optional ground truth labels (n_total_trials,).

        Returns:
            List of dicts identical to the PyTorch version:
            ``{trial_id, gates, length, true_4way, true_3way_merged}``.
        """
        sequences: List[Dict[str, Any]] = []
        trial_idx = 0

        for X_batch, lengths in zip(X_batches, lengths_batches):
            B = X_batch.shape[0]
            T = X_batch.shape[1]

            _y_pred, internals = self._apply_jit(self.params, X_batch, lengths)
            routing_gates = internals["routing_gates"]  # (B, T, 2)

            assert routing_gates.shape == (B, T, 2), (
                f"routing_gates shape {routing_gates.shape} != (B={B}, T={T}, 2)"
            )

            # Convert to NumPy once per batch
            gates_np = np.asarray(routing_gates)
            lengths_np = np.asarray(lengths)

            for i in range(B):
                length_i = int(lengths_np[i])
                gates_i = gates_np[i, :length_i, :]  # (T_valid, 2)

                if labels is not None and trial_idx < len(labels):
                    true_4way = int(labels[trial_idx])
                else:
                    true_4way = -1

                true_3way = _PyTorchGatingClusterAdapter._map_4way_to_3way(true_4way)

                sequences.append({
                    "trial_id": trial_idx,
                    "gates": gates_i,
                    "length": length_i,
                    "true_4way": true_4way,
                    "true_3way_merged": true_3way,
                })
                trial_idx += 1

        return sequences

    def compute_fingerprints(
        self,
        sequences: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Compute window-free 16-dim fingerprints using JAX vectorization.

        For sequences long enough to vectorize (>= 2 timesteps), the core
        statistics are computed in a batch via ``jax.vmap``.  Entropy is
        patched in post-hoc with NumPy (histogram-based, not XLA-friendly).

        Args:
            sequences: List of dicts from :meth:`extract_gating_sequences`.

        Returns:
            np.ndarray of shape (N, 16).
        """
        N = len(sequences)
        if N == 0:
            return np.zeros((0, self.config.fingerprint_dim), dtype=np.float32)

        # Separate empty vs non-empty trials
        valid_mask = []
        valid_gates: List[np.ndarray] = []
        for seq in sequences:
            T = seq["gates"].shape[0]
            valid_mask.append(T >= 2)
            if T >= 2:
                valid_gates.append(seq["gates"])

        fingerprints = np.zeros((N, self.config.fingerprint_dim), dtype=np.float32)

        if not valid_gates:
            return fingerprints

        # Pad to uniform length for vmap
        max_T = max(g.shape[0] for g in valid_gates)
        padded = np.zeros((len(valid_gates), max_T, 2), dtype=np.float32)
        lengths = np.zeros(len(valid_gates), dtype=np.int32)
        for i, g in enumerate(valid_gates):
            T_i = g.shape[0]
            padded[i, :T_i, :] = g
            lengths[i] = T_i

        padded_jax = jnp.array(padded)

        # vmap the core fingerprint over trials
        batch_fp = jax.vmap(_fingerprint_single_jax)(padded_jax)  # (N_valid, 16)
        batch_fp_np = np.asarray(batch_fp)

        # Patch entropy (indices 5 and 11) from NumPy
        import torch
        pt_adapter = _PyTorchGatingClusterAdapter(
            model=torch.nn.Module(),
            device=torch.device("cpu"),
            config=self.config,
        )
        idx_valid = 0
        for seq_idx, seq in enumerate(sequences):
            if not valid_mask[seq_idx]:
                continue

            gates = seq["gates"]
            T = gates.shape[0]
            g_lif = gates[:, 0]
            g_gru = gates[:, 1]

            entropy_lif = pt_adapter._compute_entropy(g_lif, T)
            entropy_gru = pt_adapter._compute_entropy(g_gru, T)

            fp = batch_fp_np[idx_valid].copy()
            fp[5] = entropy_lif
            fp[11] = entropy_gru

            assert fp.shape == (self.config.fingerprint_dim,), (
                f"Fingerprint shape {fp.shape} != ({self.config.fingerprint_dim},)"
            )
            fingerprints[seq_idx] = fp
            idx_valid += 1

        return fingerprints

    def cluster(
        self,
        fingerprints: np.ndarray,
    ) -> Dict[str, Any]:
        """Perform unsupervised clustering (delegates to PyTorch adapter).

        The clustering logic is pure sklearn/NumPy; no JAX acceleration
        needed.

        Args:
            fingerprints: (N, 16) array.

        Returns:
            Dict identical to PyTorch GatingClusterAdapter.cluster.
        """
        import torch
        pt_adapter = _PyTorchGatingClusterAdapter(
            model=torch.nn.Module(),
            device=torch.device("cpu"),
            config=self.config,
        )
        return pt_adapter.cluster(fingerprints)

    def compute_umap_embedding(
        self,
        fingerprints_scaled: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute UMAP embedding (delegates to PyTorch adapter).

        Args:
            fingerprints_scaled: (N, 16) scaled fingerprints.

        Returns:
            (N, 2) UMAP embedding or None.
        """
        import torch
        pt_adapter = _PyTorchGatingClusterAdapter(
            model=torch.nn.Module(),
            device=torch.device("cpu"),
            config=self.config,
        )
        return pt_adapter.compute_umap_embedding(fingerprints_scaled)

    def evaluate_clustering(
        self,
        labels_pred: np.ndarray,
        labels_true: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate clustering quality (delegates to PyTorch adapter).

        Args:
            labels_pred: Predicted cluster labels.
            labels_true: True class labels.

        Returns:
            Dict with "ari" and "nmi" scores.
        """
        # evaluate_clustering is a pure sklearn call; build a minimal adapter
        import torch
        pt_adapter = _PyTorchGatingClusterAdapter(
            model=torch.nn.Module(),
            device=torch.device("cpu"),
            config=self.config,
        )
        return pt_adapter.evaluate_clustering(labels_pred, labels_true)


# ===============================================================
# Module-level convenience functions
# ===============================================================

def fingerprint_jax(gates: np.ndarray, config: ClusterGatingConfig) -> np.ndarray:
    """Compute 16-dim fingerprint from a (T, 2) gate array.

    Thin wrapper around :meth:`GatingClusterAdapterJAX.compute_fingerprints`
    for a single trial.

    Args:
        gates: (T, 2) gate sequence.
        config: ClusterGatingConfig.

    Returns:
        (16,) fingerprint vector.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for fingerprint_jax.")

    T = gates.shape[0]
    if T < 2:
        return np.zeros(config.fingerprint_dim, dtype=np.float32)

    # JAX core stats
    gates_jax = jnp.array(gates)
    fp_jax = _fingerprint_single_jax(gates_jax)
    fp = np.asarray(fp_jax).copy()

    # Patch entropy from NumPy
    import torch
    pt_adapter = _PyTorchGatingClusterAdapter(
        model=torch.nn.Module(),
        device=torch.device("cpu"),
        config=config,
    )
    fp[5] = pt_adapter._compute_entropy(gates[:, 0], T)
    fp[11] = pt_adapter._compute_entropy(gates[:, 1], T)

    assert fp.shape == (config.fingerprint_dim,), (
        f"Fingerprint shape {fp.shape} != ({config.fingerprint_dim},)"
    )
    return fp


def extract_and_cluster_gates_jax(
    model: "NSMoRModel",
    params: Dict[str, Any],
    X_batches: List[Any],
    lengths_batches: List[Any],
    labels: Optional[np.ndarray] = None,
    config: Optional[ClusterGatingConfig] = None,
) -> Dict[str, Any]:
    """End-to-end pipeline: extract gates, compute fingerprints, cluster.

    JAX equivalent of :func:`nsmor.analysis.gating_cluster.extract_and_cluster_gates`.

    Args:
        model: Flax NSMoRModel.
        params: Flax parameter PyTree.
        X_batches: List of (B, T, 8) input arrays.
        lengths_batches: Corresponding (B,) length arrays.
        labels: Optional ground truth labels.
        config: Optional ClusterGatingConfig.

    Returns:
        Dict with sequences, fingerprints, clustering results, etc.
    """
    adapter = GatingClusterAdapterJAX(model, params, config=config)
    cfg = adapter.config

    sequences = adapter.extract_gating_sequences(X_batches, lengths_batches, labels)
    fingerprints = adapter.compute_fingerprints(sequences)
    cluster_result = adapter.cluster(fingerprints)
    umap_embedding = adapter.compute_umap_embedding(cluster_result["fingerprints_scaled"])

    evaluation: Dict[str, Dict[str, float]] = {}
    if labels is not None:
        labels_4way = np.array([s["true_4way"] for s in sequences])
        labels_3way = np.array([s["true_3way_merged"] for s in sequences])
        evaluation["k4_vs_4way"] = adapter.evaluate_clustering(
            cluster_result["labels_k4"], labels_4way,
        )
        evaluation["k3_vs_3way"] = adapter.evaluate_clustering(
            cluster_result["labels_k3"], labels_3way,
        )

    trajectories_k4 = {}
    for cid in np.unique(cluster_result["labels_k4"]):
        mask = cluster_result["labels_k4"] == cid
        trajs = []
        for i, seq in enumerate(sequences):
            if mask[i] and seq["gates"].shape[0] > 0:
                trajs.append(interpolate_for_viz(seq["gates"], cfg.interp_length))
        if trajs:
            trajectories_k4[int(cid)] = np.mean(trajs, axis=0)

    return {
        "sequences": sequences,
        "fingerprints": fingerprints,
        "fingerprints_scaled": cluster_result["fingerprints_scaled"],
        "k_opt": cluster_result["k_opt"],
        "silhouette_scores": cluster_result["silhouette_scores"],
        "labels_k4": cluster_result["labels_k4"],
        "labels_k3": cluster_result["labels_k3"],
        "labels_kopt": cluster_result["labels_kopt"],
        "umap_embedding": umap_embedding,
        "evaluation": evaluation,
        "trajectories_k4": trajectories_k4,
        "config": cfg,
    }
