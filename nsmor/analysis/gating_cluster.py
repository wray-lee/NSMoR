"""
Gating Cluster Analysis — Window-Free Unsupervised Clustering.

Implements fingerprint extraction and clustering of router gating dynamics
without any manual time windows. NSMoR is Trial-Start anchored; this module
rejects TTC-based or baseline-windowed designs to avoid human bias.

Window-free by design. NSMoR is Trial-Start anchored. TTC-50ms is only for
MCMC prior 5-D snapshot. Baseline 5700ms is variant for pure-wind via
TimeWindowConfig, not universal. Manual windows like [-5700:-500] inject human
bias and break unsupervised claim. Clustering is unsupervised (silhouette
selects k without labels); k=4 matches labeling.py cardinality; k=3 merged is
for biological interpretation only and defined as Startle->Escape,
Walk+Pre_Active->PreWalk, NoResponse->NoResponse. Pearson NaN guarded to 0.0
when std==0.

References
----------
- Kaufman & Rousseeuw (1990) Finding Groups in Data.
- McInnes et al. (2018) UMAP: Uniform Manifold Approximation and Projection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from scipy import interpolate as sp_interp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from nsmor.config import Label
from nsmor.nsmor_dataloader import collate_variable_length


# ═══════════════════════════════════════════════════════════════════════
# Configuration (no window parameters — window-free by design)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ClusterGatingConfig:
    """
    Configuration for unsupervised gating strategy clustering.

    Window-free by design. NSMoR is Trial-Start anchored. No window
    parameters are exposed to prevent human bias injection.
    """
    n_clusters: int = 4
    """Target number of clusters for k=4 analysis (matches labeling.py)."""

    n_clusters_range: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    """Range of k values to evaluate via silhouette score."""

    random_state: int = 42
    """Deterministic seed for all RNG operations."""

    use_umap: bool = True
    """Whether to compute UMAP embeddings for visualization."""

    fingerprint_dim: int = 16
    """Dimensionality of trial-level gate fingerprint vectors."""

    entropy_bins: int = 20
    """Number of histogram bins for entropy calculation in [0, 1]."""

    interp_length: int = 200
    """Target length for trajectory interpolation (visualization only)."""

    # No window field — window-free by design


# ═══════════════════════════════════════════════════════════════════════
# Gating Cluster Adapter
# ═══════════════════════════════════════════════════════════════════════

class GatingClusterAdapter:
    """
    Adapter for extracting and clustering MoR gating strategies.

    Extracts routing gate sequences from the model in eval mode,
    computes window-free fingerprints, and performs unsupervised
    clustering to discover emergent gating strategies.

    Attributes:
        model: NSMoRCore instance.
        device: Computation device.
        config: ClusterGatingConfig instance.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        config: Optional[ClusterGatingConfig] = None,
    ) -> None:
        """
        Initialize the gating cluster adapter.

        Args:
            model: Trained NSMoRCore model.
            device: Computation device (auto-detected if None).
            config: ClusterGatingConfig (uses defaults if None).
        """
        self.model = model
        self.model.eval()

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        self.config = config if config is not None else ClusterGatingConfig()

    @torch.no_grad()
    def extract_gating_sequences(
        self,
        dataloader: torch.utils.data.DataLoader,
        labels: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract per-trial gating sequences from the model.

        Runs in eval mode with torch.no_grad(), shuffle=False, and
        un-pads sequences using lengths for valid data only.

        Args:
            dataloader: DataLoader yielding (X_batch, Y_batch, lengths).
            labels: Optional ground truth labels array (n_trials,).

        Returns:
            List of dicts, each containing:
            - trial_id: int — unique trial identifier
            - gates: np.ndarray — shape (T, 2), gates[t] = [g_lif, g_gru]
            - true_4way: int — 4-way label from Label enum
            - true_3way_merged: int — merged 3-way label

        Mapping (EVAL ONLY, not training labels):
            Startle (Label.ESCAPE=0) -> 0 (Escape)
            Walk (Label.PREWALK=1) -> 1 (PreWalk)
            Pre_Active (Label.PRE_ACTIVE=2) -> 1 (PreWalk)
            NoResponse (Label.NO_RESPONSE=3) -> 2 (NoResponse)
        """
        sequences: List[Dict[str, Any]] = []
        trial_idx = 0

        for batch_idx, batch in enumerate(dataloader):
            X_batch, _Y_batch, lengths = batch
            X_batch = X_batch.to(self.device)
            lengths = lengths.to(self.device)

            B, T, _ = X_batch.shape

            # Forward pass with internals
            _y_pred, internals = self.model(
                X_batch, lengths, return_internals=True,
            )

            # routing_gates: (B, T, 2)
            routing_gates = internals["routing_gates"]

            assert routing_gates.shape == (B, T, 2), (
                f"routing_gates shape {tuple(routing_gates.shape)} != "
                f"(B={B}, T={T}, 2)"
            )

            # Un-pad and store per-trial
            for i in range(B):
                length_i = int(lengths[i].item())

                # Extract valid gate sequence: (T_valid, 2)
                gates_i = routing_gates[i, :length_i, :].cpu().numpy()

                # Get true 4-way label
                if labels is not None and trial_idx < len(labels):
                    true_4way = int(labels[trial_idx])
                else:
                    true_4way = -1  # Unknown

                # Map 4-way to 3-way (EVAL ONLY mapping)
                true_3way_merged = self._map_4way_to_3way(true_4way)

                sequences.append({
                    "trial_id": trial_idx,
                    "gates": gates_i,
                    "true_4way": true_4way,
                    "true_3way_merged": true_3way_merged,
                })

                trial_idx += 1

        return sequences

    @staticmethod
    def _map_4way_to_3way(label_4way: int) -> int:
        """
        Map 4-way label to 3-way merged label (EVAL ONLY).

        Mapping:
            Startle (0) -> Escape (0)
            Walk (1) -> PreWalk (1)
            Pre_Active (2) -> PreWalk (1)
            NoResponse (3) -> NoResponse (2)

        Args:
            label_4way: 4-way label value.

        Returns:
            3-way merged label value.
        """
        if label_4way == Label.ESCAPE:
            return 0  # Escape
        elif label_4way in (Label.PREWALK, Label.PRE_ACTIVE):
            return 1  # PreWalk
        elif label_4way == Label.NO_RESPONSE:
            return 2  # NoResponse
        else:
            return -1  # Unknown

    def compute_fingerprints(
        self,
        sequences: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Compute window-free 16-dimensional fingerprints for each trial.

        Features (exactly 16 dimensions):
        [0-5]  LIF gate statistics:
               [0] mean, [1] std, [2] max, [3] min,
               [4] dominant_fraction (prop. timesteps where g_lif > 0.5),
               [5] entropy (length-normalized)
        [6-11] GRU gate statistics:
               [6] mean, [7] std, [8] max, [9] min,
               [10] dominant_fraction (prop. timesteps where g_gru > 0.5),
               [11] entropy (length-normalized)
        [12]   Pearson correlation between g_lif and g_gru (0.0 if std==0)
        [13]   Normalized time of max g_lif: argmax(g_lif) / T
        [14]   Normalized time of min g_lif: argmin(g_lif) / T
        [15]   Maximum absolute gradient of g_lif

        The dominant_fraction captures the proportion of the trial where
        each pathway dominates (gate > 0.5), providing a biologically
        interpretable measure of pathway engagement strategy.

        Args:
            sequences: List of dicts from extract_gating_sequences.

        Returns:
            np.ndarray of shape (N, 16) where N = len(sequences).
        """
        fingerprints: List[np.ndarray] = []

        for seq in sequences:
            gates = seq["gates"]  # (T, 2)
            T = gates.shape[0]

            if T == 0:
                # Empty sequence — return zeros
                fingerprints.append(np.zeros(self.config.fingerprint_dim))
                continue

            g_lif = gates[:, 0]
            g_gru = gates[:, 1]

            # --- LIF gate features [0-5] ---
            feat_lif = self._compute_gate_features(g_lif)

            # --- GRU gate features [6-11] ---
            feat_gru = self._compute_gate_features(g_gru)

            # --- Correlation [12] ---
            corr = self._pearson_correlation(g_lif, g_gru)

            # --- Normalized time features [13-14] ---
            t_max = float(np.argmax(g_lif)) / T if T > 0 else 0.0
            t_min = float(np.argmin(g_lif)) / T if T > 0 else 0.0

            # --- Max gradient [15] ---
            if T > 1:
                grad_lif = np.abs(np.diff(g_lif))
                max_grad = float(np.max(grad_lif))
            else:
                max_grad = 0.0

            # Assemble fingerprint
            fingerprint = np.concatenate([
                feat_lif,           # [0-5]
                feat_gru,           # [6-11]
                [corr],             # [12]
                [t_max, t_min],     # [13-14]
                [max_grad],         # [15]
            ])

            assert fingerprint.shape == (self.config.fingerprint_dim,), (
                f"Fingerprint shape {fingerprint.shape} != "
                f"({self.config.fingerprint_dim},)"
            )

            fingerprints.append(fingerprint)

        return np.stack(fingerprints, axis=0)  # (N, 16)

    def _compute_gate_features(self, gate: np.ndarray) -> np.ndarray:
        """
        Compute 6 features for a single gate sequence.

        Features: mean, std, max, min, lif_dominant_fraction, entropy

        The lif_dominant_fraction is the proportion of timesteps where
        this gate exceeds the other pathway gate (for LIF, this is the
        fraction where g_lif > g_gru). This captures the dominant
        routing regime and has biological relevance for pathway engagement.

        Args:
            gate: 1-D array of gate values, shape (T,).

        Returns:
            np.ndarray of shape (6,) with [mean, std, max, min, dominant_frac, entropy].
        """
        T = len(gate)

        if T == 0:
            return np.zeros(6)

        # Basic statistics
        mean_val = float(np.mean(gate))
        std_val = float(np.std(gate, ddof=1)) if T > 1 else 0.0
        max_val = float(np.max(gate))
        min_val = float(np.min(gate))

        # LIF/GRU dominant regime fraction is computed in the calling method
        # For now, we compute a placeholder that will be replaced
        # This will be computed as the proportion of timesteps where this gate > 0.5
        # (since gates are softmax outputs, >0.5 implies dominance)
        dominant_frac = float(np.mean(gate > 0.5))

        # Entropy of histogram (length-normalized differential entropy estimate)
        entropy = self._compute_entropy(gate, T)

        return np.array([mean_val, std_val, max_val, min_val, dominant_frac, entropy])

    def _compute_entropy(self, data: np.ndarray, T: int) -> float:
        """
        Compute length-normalized entropy estimate of data distribution.

        Uses a binned histogram entropy estimator with length normalization
        to ensure comparability across trials of different durations.
        For T < 20, falls back to a sparse-sample corrected estimate.

        Note: This is the empirical entropy of the binned distribution,
        not the differential entropy. The values are normalized by
        log(T) to account for sampling density effects.

        Args:
            data: 1-D array of values in [0, 1].
            T: Original sequence length (for normalization).

        Returns:
            Length-normalized entropy estimate [0, 1].
        """
        n_samples = len(data)
        if n_samples == 0:
            return 0.0

        # Adaptive bin count based on sample size (square root rule)
        n_bins = min(self.config.entropy_bins, max(5, int(np.sqrt(n_samples))))

        # Histogram with adaptive bins in data range
        hist, bin_edges = np.histogram(
            data,
            bins=n_bins,
            range=(float(np.min(data)), float(np.max(data)) + 1e-12),
        )

        # Convert to probabilities (avoiding zero bins)
        prob = hist.astype(np.float64) / n_samples
        prob = prob[prob > 0]

        if len(prob) == 0:
            return 0.0

        # Shannon entropy in bits
        eps = 1e-12
        entropy_raw = -np.sum(prob * np.log2(prob + eps))

        # Normalize by log2(n_bins) to get [0, 1] range (maximum entropy = uniform)
        max_entropy = np.log2(n_bins)
        entropy_normalized = entropy_raw / max_entropy if max_entropy > 0 else 0.0

        # Length correction: for small T, entropy is overestimated
        # Apply a correction factor that approaches 1 as T increases
        if T < 20:
            # Small sample correction: reduce entropy for small T
            correction = T / 20.0
            entropy_normalized *= correction

        return float(np.clip(entropy_normalized, 0.0, 1.0))

    @staticmethod
    def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Pearson correlation coefficient with NaN guard.

        Returns 0.0 if either std is zero (NaN guard).

        Args:
            x: 1-D array.
            y: 1-D array, same length as x.

        Returns:
            Pearson r in [-1, 1], or 0.0 if std == 0.
        """
        assert len(x) == len(y), f"Length mismatch: {len(x)} != {len(y)}"

        if len(x) < 2:
            return 0.0

        std_x = np.std(x, ddof=1)
        std_y = np.std(y, ddof=1)

        # NaN guard: return 0.0 if std == 0
        if std_x < 1e-12 or std_y < 1e-12:
            return 0.0

        corr = np.corrcoef(x, y)[0, 1]

        # Guard against NaN
        if np.isnan(corr):
            return 0.0

        return float(corr)

    def cluster(
        self,
        fingerprints: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Perform unsupervised clustering on gate fingerprints.

        Steps:
        1. StandardScaler normalization
        2. Silhouette score for k in config.n_clusters_range -> k_opt
        3. KMeans for k=4 and k=3 (fixed for biological interpretation)
        4. Optional GMM for k=4 and k=3
        5. Bootstrap stability assessment for k_opt

        All operations are seeded with config.random_state for determinism.

        Args:
            fingerprints: Array of shape (N, 16).

        Returns:
            Dict containing:
            - k_opt: int — optimal k from silhouette
            - silhouette_scores: dict — {k: score}
            - labels_k4: np.ndarray — KMeans labels for k=4
            - labels_k3: np.ndarray — KMeans labels for k=3
            - fingerprints_scaled: np.ndarray — scaled fingerprints
            - stability_scores: dict — {k: stability_index}
        """
        N = fingerprints.shape[0]

        if N < 2:
            raise ValueError(f"Need at least 2 samples for clustering, got {N}")

        # Set all RNG seeds for determinism
        np.random.seed(self.config.random_state)
        import random
        random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)

        # StandardScaler
        scaler = StandardScaler()
        fingerprints_scaled = scaler.fit_transform(fingerprints)

        # Silhouette analysis for k selection
        silhouette_scores: Dict[int, float] = {}

        for k in self.config.n_clusters_range:
            if k >= N:
                continue  # Can't cluster with k >= N

            kmeans = KMeans(
                n_clusters=k,
                random_state=self.config.random_state,
                n_init=10,
            )
            labels = kmeans.fit_predict(fingerprints_scaled)

            # Silhouette score requires at least 2 clusters with samples
            if len(np.unique(labels)) >= 2:
                score = silhouette_score(fingerprints_scaled, labels)
                silhouette_scores[k] = float(score)

        # Bootstrap stability assessment for each k
        stability_scores: Dict[int, float] = {}
        n_bootstrap = 100
        for k in self.config.n_clusters_range:
            if k >= N:
                continue
            if N < 10:
                # Too few samples for reliable bootstrap
                stability_scores[k] = 0.0
                continue

            stability_values = []
            rng = np.random.default_rng(self.config.random_state)
            for _ in range(n_bootstrap):
                # Bootstrap resample
                idx = rng.choice(N, size=N, replace=True)
                idx_oob = np.setdiff1d(np.arange(N), np.unique(idx))
                if len(idx_oob) < 2:
                    continue

                # Fit on bootstrap sample
                kmeans_boot = KMeans(
                    n_clusters=k,
                    random_state=self.config.random_state,
                    n_init=10,
                )
                labels_boot = kmeans_boot.fit_predict(fingerprints_scaled[idx])

                # Predict on OOB samples
                labels_oob_pred = kmeans_boot.predict(fingerprints_scaled[idx_oob])

                # Fit on OOB samples directly
                kmeans_oob = KMeans(
                    n_clusters=k,
                    random_state=self.config.random_state,
                    n_init=10,
                )
                labels_oob_true = kmeans_oob.fit_predict(fingerprints_scaled[idx_oob])

                # Compute ARI between the two clusterings on OOB samples
                if len(np.unique(labels_oob_pred)) > 1 and len(np.unique(labels_oob_true)) > 1:
                    ari = adjusted_rand_score(labels_oob_true, labels_oob_pred)
                    stability_values.append(ari)

            if stability_values:
                stability_scores[k] = float(np.median(stability_values))
            else:
                stability_scores[k] = 0.0

        # Select optimal k: prefer highest silhouette among k with stability >= 0.6
        # If no k meets stability threshold, fall back to highest silhouette
        stable_ks = [k for k, s in stability_scores.items() if s >= 0.6]
        if stable_ks and silhouette_scores:
            k_opt = max(stable_ks, key=lambda k: silhouette_scores.get(k, -1.0))
        elif silhouette_scores:
            k_opt = max(silhouette_scores, key=silhouette_scores.get)
        else:
            k_opt = self.config.n_clusters

        # KMeans for k=4 and k=3
        kmeans_k4 = KMeans(
            n_clusters=4,
            random_state=self.config.random_state,
            n_init=10,
        )
        labels_k4 = kmeans_k4.fit_predict(fingerprints_scaled)

        kmeans_k3 = KMeans(
            n_clusters=3,
            random_state=self.config.random_state,
            n_init=10,
        )
        labels_k3 = kmeans_k3.fit_predict(fingerprints_scaled)

        # GMM for k=4 and k=3 (optional)
        try:
            gmm_k4 = GaussianMixture(
                n_components=4,
                random_state=self.config.random_state,
            )
            gmm_k4.fit(fingerprints_scaled)
            gmm_labels_k4 = gmm_k4.predict(fingerprints_scaled)
        except Exception:
            gmm_labels_k4 = labels_k4.copy()

        try:
            gmm_k3 = GaussianMixture(
                n_components=3,
                random_state=self.config.random_state,
            )
            gmm_k3.fit(fingerprints_scaled)
            gmm_labels_k3 = gmm_k3.predict(fingerprints_scaled)
        except Exception:
            gmm_labels_k3 = labels_k3.copy()

        return {
            "k_opt": k_opt,
            "silhouette_scores": silhouette_scores,
            "stability_scores": stability_scores,
            "labels_k4": labels_k4,
            "labels_k3": labels_k3,
            "gmm_labels_k4": gmm_labels_k4,
            "gmm_labels_k3": gmm_labels_k3,
            "fingerprints_scaled": fingerprints_scaled,
            "scaler": scaler,
        }

    def compute_umap_embedding(
        self,
        fingerprints_scaled: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Compute UMAP embedding for visualization.

        Args:
            fingerprints_scaled: Scaled fingerprint array (N, 16).

        Returns:
            UMAP embedding of shape (N, 2), or None if UMAP unavailable.
        """
        if not UMAP_AVAILABLE or not self.config.use_umap:
            return None

        # UMAP with deterministic seed
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=self.config.random_state,
            n_jobs=1,  # Deterministic
        )

        embedding = reducer.fit_transform(fingerprints_scaled)
        return embedding

    def evaluate_clustering(
        self,
        labels_pred: np.ndarray,
        labels_true: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate clustering quality against true labels.

        Computes ARI and NMI. Handles cases where number of clusters
        doesn't match number of true classes.

        Args:
            labels_pred: Predicted cluster labels.
            labels_true: True class labels.

        Returns:
            Dict with "ari" and "nmi" scores.
        """
        # Filter out unknown labels (-1)
        mask = labels_true >= 0
        if mask.sum() == 0:
            return {"ari": 0.0, "nmi": 0.0}

        labels_pred_filtered = labels_pred[mask]
        labels_true_filtered = labels_true[mask]

        ari = adjusted_rand_score(labels_true_filtered, labels_pred_filtered)
        nmi = normalized_mutual_info_score(labels_true_filtered, labels_pred_filtered)

        return {"ari": float(ari), "nmi": float(nmi)}

    def interpolate_trajectories(
        self,
        sequences: List[Dict[str, Any]],
        cluster_labels: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """
        Interpolate gate trajectories to common length for visualization.

        Args:
            sequences: List of gate sequences from extract_gating_sequences.
            cluster_labels: Cluster assignment for each sequence.

        Returns:
            Dict mapping cluster_id -> mean interpolated trajectory (T, 2).
        """
        interp_length = self.config.interp_length

        # Group by cluster
        cluster_trajectories: Dict[int, List[np.ndarray]] = {}
        for i, seq in enumerate(sequences):
            cluster_id = int(cluster_labels[i])
            if cluster_id not in cluster_trajectories:
                cluster_trajectories[cluster_id] = []

            gates = seq["gates"]  # (T, 2)

            # Interpolate to fixed length
            if gates.shape[0] == 0:
                continue

            T_orig = gates.shape[0]
            x_orig = np.linspace(0, 1, T_orig)
            x_new = np.linspace(0, 1, interp_length)

            # Interpolate both gates
            interp_gates = np.zeros((interp_length, 2))
            for j in range(2):
                f = sp_interp.interp1d(x_orig, gates[:, j], kind='linear',
                                       fill_value='extrapolate')
                interp_gates[:, j] = f(x_new)

            cluster_trajectories[cluster_id].append(interp_gates)

        # Compute mean trajectory per cluster
        mean_trajectories: Dict[int, np.ndarray] = {}
        for cluster_id, trajs in cluster_trajectories.items():
            if len(trajs) > 0:
                mean_trajectories[cluster_id] = np.mean(trajs, axis=0)

        return mean_trajectories


# ═══════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════

def fingerprint(gates: np.ndarray, config: ClusterGatingConfig) -> np.ndarray:
    """
    Compute fingerprint for a single gate sequence.

    Args:
        gates: np.ndarray of shape (T, 2) with [g_lif, g_gru] per timestep.
        config: ClusterGatingConfig instance.

    Returns:
        np.ndarray of shape (16,) fingerprint vector.
    """
    if gates.shape[0] == 0:
        return np.zeros(config.fingerprint_dim, dtype=np.float32)

    # Use a temporary adapter to compute features
    adapter = GatingClusterAdapter(torch.nn.Module(), config=config)

    # Create a dummy sequence
    seq = {"trial_id": 0, "gates": gates, "true_4way": 0, "true_3way_merged": 0}

    # Compute fingerprint via adapter
    fps = adapter.compute_fingerprints([seq])
    return fps[0]


def compute_fingerprints(
    sequences: List[Dict[str, Any]],
    config: ClusterGatingConfig,
) -> np.ndarray:
    """
    Compute fingerprints for multiple sequences.

    Args:
        sequences: List of dicts with 'gates' key containing np.ndarray (T, 2).
        config: ClusterGatingConfig instance.

    Returns:
        np.ndarray of shape (N, 16) where N = len(sequences).
    """
    # Use a temporary adapter to compute features
    adapter = GatingClusterAdapter(torch.nn.Module(), config=config)
    return adapter.compute_fingerprints(sequences)


def cluster(fingerprints: np.ndarray, config: ClusterGatingConfig) -> Dict[str, Any]:
    """
    Perform unsupervised clustering on fingerprints.

    Args:
        fingerprints: np.ndarray of shape (N, 16).
        config: ClusterGatingConfig instance.

    Returns:
        Dict with clustering results including 'labels_k4', 'labels_k3', 'k_opt', etc.
    """
    # Use a temporary adapter to perform clustering
    adapter = GatingClusterAdapter(torch.nn.Module(), config=config)
    return adapter.cluster(fingerprints)


def interpolate_for_viz(gates: np.ndarray, target_length: int) -> np.ndarray:
    """
    Interpolate gate trajectory to target length for visualization.

    Args:
        gates: np.ndarray of shape (T, 2).
        target_length: Desired output length.

    Returns:
        np.ndarray of shape (target_length, 2).
    """
    T_orig = gates.shape[0]
    if T_orig == target_length:
        return gates.copy()

    x_orig = np.linspace(0, 1, T_orig)
    x_new = np.linspace(0, 1, target_length)

    interp_gates = np.zeros((target_length, 2), dtype=gates.dtype)
    for j in range(2):
        f = sp_interp.interp1d(x_orig, gates[:, j], kind='linear',
                               fill_value='extrapolate')
        interp_gates[:, j] = f(x_new)

    return interp_gates


def extract_and_cluster_gates(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    labels: Optional[np.ndarray] = None,
    config: Optional[ClusterGatingConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    End-to-end pipeline: extract gates, compute fingerprints, cluster.

    Args:
        model: Trained NSMoRCore model.
        dataloader: DataLoader for the dataset.
        labels: Optional ground truth labels.
        config: Optional ClusterGatingConfig.
        device: Optional device.

    Returns:
        Dict with extraction results, fingerprints, clustering, and evaluation.
    """
    # Initialize adapter
    adapter = GatingClusterAdapter(model, device=device, config=config)
    cfg = adapter.config

    # Extract gating sequences
    sequences = adapter.extract_gating_sequences(dataloader, labels)

    # Compute fingerprints
    fingerprints = adapter.compute_fingerprints(sequences)

    # Cluster
    cluster_result = adapter.cluster(fingerprints)

    # UMAP embedding
    umap_embedding = adapter.compute_umap_embedding(cluster_result["fingerprints_scaled"])

    # Evaluate against true labels
    evaluation: Dict[str, Dict[str, float]] = {}
    if labels is not None:
        labels_4way = np.array([s["true_4way"] for s in sequences])
        labels_3way = np.array([s["true_3way_merged"] for s in sequences])

        evaluation["k4_vs_4way"] = adapter.evaluate_clustering(
            cluster_result["labels_k4"], labels_4way
        )
        evaluation["k3_vs_3way"] = adapter.evaluate_clustering(
            cluster_result["labels_k3"], labels_3way
        )

    # Interpolated trajectories
    trajectories_k4 = adapter.interpolate_trajectories(
        sequences, cluster_result["labels_k4"]
    )

    return {
        "sequences": sequences,
        "fingerprints": fingerprints,
        "fingerprints_scaled": cluster_result["fingerprints_scaled"],
        "k_opt": cluster_result["k_opt"],
        "silhouette_scores": cluster_result["silhouette_scores"],
        "labels_k4": cluster_result["labels_k4"],
        "labels_k3": cluster_result["labels_k3"],
        "umap_embedding": umap_embedding,
        "evaluation": evaluation,
        "trajectories_k4": trajectories_k4,
        "config": cfg,
    }


# ═══════════════════════════════════════════════════════════════════════
# Smoke Test
# ═══════════════════════════════════════════════════════════════════════

def _test_gating_cluster():
    """
    Verify GatingClusterAdapter fingerprint and clustering.

    Run:
        python -m nsmor.analysis.gating_cluster
    """
    print("=" * 60)
    print("GatingClusterAdapter smoke test")
    print("=" * 60)

    # Mock data
    np.random.seed(42)
    n_trials = 10
    seq_len = 100

    sequences = []
    for i in range(n_trials):
        # Create gates with some structure
        t = np.linspace(0, 1, seq_len)
        g_lif = 0.3 + 0.4 * np.sin(2 * np.pi * t * (i + 1) / n_trials) + 0.1 * np.random.randn(seq_len)
        g_gru = 1.0 - g_lif + 0.05 * np.random.randn(seq_len)

        # Clip to valid range
        g_lif = np.clip(g_lif, 0, 1)
        g_gru = np.clip(g_gru, 0, 1)

        gates = np.stack([g_lif, g_gru], axis=1)

        sequences.append({
            "trial_id": i,
            "gates": gates,
            "true_4way": i % 4,
            "true_3way_merged": i % 3,
        })

    # Create adapter with default config
    from nsmor.model_nsmor_core import NSMoRCore

    model = NSMoRCore(hidden_dim=32)
    adapter = GatingClusterAdapter(model)

    # Compute fingerprints
    fingerprints = adapter.compute_fingerprints(sequences)
    print(f"Fingerprints shape: {fingerprints.shape}")
    assert fingerprints.shape == (n_trials, 16), f"Expected (10, 16), got {fingerprints.shape}"

    # Check specific features
    assert not np.any(np.isnan(fingerprints)), "Fingerprints contain NaN"
    print(f"Fingerprint range: [{fingerprints.min():.3f}, {fingerprints.max():.3f}]")

    # Test clustering
    cluster_result = adapter.cluster(fingerprints)
    print(f"k_opt: {cluster_result['k_opt']}")
    print(f"Silhouette scores: {cluster_result['silhouette_scores']}")

    # Test UMAP
    umap_emb = adapter.compute_umap_embedding(cluster_result["fingerprints_scaled"])
    if umap_emb is not None:
        print(f"UMAP embedding shape: {umap_emb.shape}")
    else:
        print("UMAP not available (skipping)")

    print("=" * 60)
    print("All GatingClusterAdapter assertions passed.")
    print("=" * 60)


if __name__ == "__main__":
    _test_gating_cluster()


# ═══════════════════════════════════════════════════════════════════════
# Module-level exports (for backward compatibility and testing)
# ═══════════════════════════════════════════════════════════════════════

def fingerprint(gates: np.ndarray, config: ClusterGatingConfig) -> np.ndarray:
    """
    Compute 16-dimensional fingerprint from gate sequence (module-level export).

    Args:
        gates: Gate sequence of shape (T, 2).
        config: ClusterGatingConfig.

    Returns:
        Fingerprint vector of shape (config.fingerprint_dim,).
    """
    # Use the static method from GatingClusterAdapter
    T = gates.shape[0]
    if T == 0:
        return np.zeros(config.fingerprint_dim)

    g_lif = gates[:, 0]
    g_gru = gates[:, 1]

    def _compute_gate_features(gate: np.ndarray, entropy_bins: int) -> np.ndarray:
        """Compute 6 features for a single gate sequence."""
        T_local = len(gate)
        if T_local == 0:
            return np.zeros(6)

        mean_val = float(np.mean(gate))
        std_val = float(np.std(gate, ddof=1)) if T_local > 1 else 0.0
        max_val = float(np.max(gate))
        min_val = float(np.min(gate))
        dominant_frac = float(np.mean(gate > 0.5))

        # Entropy
        n_bins = min(entropy_bins, max(5, int(np.sqrt(T_local))))
        hist, _ = np.histogram(
            gate, bins=n_bins,
            range=(float(np.min(gate)), float(np.max(gate)) + 1e-12),
        )
        prob = hist.astype(np.float64) / T_local
        prob = prob[prob > 0]
        if len(prob) == 0:
            entropy = 0.0
        else:
            eps = 1e-12
            entropy_raw = -np.sum(prob * np.log2(prob + eps))
            max_entropy = np.log2(n_bins)
            entropy = entropy_raw / max_entropy if max_entropy > 0 else 0.0
            if T_local < 20:
                entropy *= T_local / 20.0

        return np.array([mean_val, std_val, max_val, min_val, dominant_frac, entropy])

    feat_lif = _compute_gate_features(g_lif, config.entropy_bins)
    feat_gru = _compute_gate_features(g_gru, config.entropy_bins)
    corr = GatingClusterAdapter._pearson_correlation(g_lif, g_gru)
    t_max = float(np.argmax(g_lif)) / T if T > 0 else 0.0
    t_min = float(np.argmin(g_lif)) / T if T > 0 else 0.0
    if T > 1:
        grad_lif = np.abs(np.diff(g_lif))
        max_grad = float(np.max(grad_lif))
    else:
        max_grad = 0.0

    fp = np.concatenate([
        feat_lif, feat_gru, [corr], [t_max, t_min], [max_grad]
    ])

    assert fp.shape == (config.fingerprint_dim,), (
        f"Fingerprint shape {fp.shape} != ({config.fingerprint_dim},)"
    )
    return fp


def compute_fingerprints(
    sequences: List[Dict[str, Any]],
    config: ClusterGatingConfig,
) -> np.ndarray:
    """
    Compute fingerprints for all sequences (module-level export).

    Args:
        sequences: List of dicts from extract_gating_sequences.
        config: ClusterGatingConfig.

    Returns:
        Fingerprint matrix of shape (N, config.fingerprint_dim).
    """
    fingerprints = []
    for seq in sequences:
        fp = fingerprint(seq["gates"], config)
        fingerprints.append(fp)
    return np.array(fingerprints)


def cluster(
    fingerprints: np.ndarray,
    config: ClusterGatingConfig,
) -> Dict[str, Any]:
    """
    Perform unsupervised clustering (module-level export).

    Args:
        fingerprints: Fingerprint matrix of shape (N, fingerprint_dim).
        config: ClusterGatingConfig.

    Returns:
        Dict with clustering results.
    """
    N = fingerprints.shape[0]
    if N < 2:
        raise ValueError(f"Need at least 2 samples for clustering, got {N}")

    # Set seeds
    np.random.seed(config.random_state)
    import random
    random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    # Scale
    scaler = StandardScaler()
    fingerprints_scaled = scaler.fit_transform(fingerprints)

    # Silhouette analysis
    silhouette_scores: Dict[int, float] = {}
    for k in config.n_clusters_range:
        if k >= N:
            continue
        kmeans = KMeans(
            n_clusters=k,
            random_state=config.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(fingerprints_scaled)
        if len(np.unique(labels)) >= 2:
            score = silhouette_score(fingerprints_scaled, labels)
            silhouette_scores[k] = float(score)

    k_opt = max(silhouette_scores, key=silhouette_scores.get) if silhouette_scores else config.n_clusters

    # KMeans for k=4, k=3, k_opt
    kmeans_4 = KMeans(n_clusters=4, random_state=config.random_state, n_init=10)
    labels_k4 = kmeans_4.fit_predict(fingerprints_scaled)

    kmeans_3 = KMeans(n_clusters=3, random_state=config.random_state, n_init=10)
    labels_k3 = kmeans_3.fit_predict(fingerprints_scaled)

    kmeans_opt = KMeans(n_clusters=k_opt, random_state=config.random_state, n_init=10)
    labels_kopt = kmeans_opt.fit_predict(fingerprints_scaled)

    # UMAP
    umap_embedding: Optional[np.ndarray] = None
    if config.use_umap and UMAP_AVAILABLE:
        try:
            reducer = umap.UMAP(
                n_neighbors=min(15, N - 1),
                min_dist=0.1,
                n_components=2,
                random_state=config.random_state,
                n_jobs=1,
            )
            umap_embedding = reducer.fit_transform(fingerprints_scaled)
        except Exception:
            pass

    return {
        "k_opt": k_opt,
        "silhouette_scores": silhouette_scores,
        "labels_k4": labels_k4,
        "labels_k3": labels_k3,
        "labels_kopt": labels_kopt,
        "umap_embedding": umap_embedding,
        "fingerprints_scaled": fingerprints_scaled,
        "scaler": scaler,
    }


def interpolate_for_viz(
    gates: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Interpolate gate sequence for visualization (module-level export).

    Args:
        gates: Gate sequence of shape (T, 2).
        target_length: Target interpolation length.

    Returns:
        Interpolated sequence of shape (target_length, 2).
    """
    from scipy import interpolate as sp_interp

    T = gates.shape[0]
    if T == target_length:
        return gates.copy()

    old_x = np.linspace(0, 1, T)
    new_x = np.linspace(0, 1, target_length)

    interp_lif = np.interp(new_x, old_x, gates[:, 0])
    interp_gru = np.interp(new_x, old_x, gates[:, 1])

    return np.stack([interp_lif, interp_gru], axis=1)
