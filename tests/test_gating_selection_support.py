"""Synthetic producer checks for H selection support across extraction and JSON export."""
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from nsmor.analysis import gating_cluster
from nsmor.analysis.gating_cluster import ClusterGatingConfig
from scripts import analyze_gating as gating


@pytest.mark.parametrize("stability,basis,k_opt", [
    ({2: 0.72, 3: None, 4: 0.3}, "silhouette among bootstrap-stable k", 2),
    ({2: None, 3: 0.42, 4: 0.59}, "silhouette fallback (no k passed the stability gate)", 3),
])
def test_selection_support_reaches_summary(monkeypatch, stability, basis, k_opt):
    sequences = [
        {"true_4way": i, "true_3way_merged": (0 if i == 0 else 2 if i == 3 else 1),
         "trial_id": i, "length": 8, "gate_seq": np.full((8, 2), 0.5)}
        for i in range(4)
    ]
    labels = np.arange(4)

    class Adapter:
        def __init__(self, model, device=None, config=None):
            self.config = config

        def extract_gating_sequences(self, dataloader, labels):
            return sequences

        def compute_fingerprints(self, sequences):
            return np.zeros((4, 16))

        def cluster(self, fingerprints):
            return {"k_opt": k_opt, "k_selection_basis": basis,
                    "silhouette_scores": {2: 0.3, 3: 0.6, 4: 0.2},
                    "stability_scores": stability,
                    "fingerprints_scaled": fingerprints,
                    "labels_k4": labels, "labels_k3": np.array([0, 1, 1, 2]),
                    "labels_kopt": labels % k_opt}

        def compute_umap_embedding(self, fingerprints_scaled):
            return None

        def evaluate_clustering(self, predicted, true):
            return {"ari": adjusted_rand_score(predicted, true),
                    "nmi": normalized_mutual_info_score(predicted, true)}

        def interpolate_trajectories(self, sequences, labels):
            return {}

    monkeypatch.setattr(gating_cluster, "GatingClusterAdapter", Adapter)
    config = ClusterGatingConfig(n_clusters_range=[2, 3, 4], use_umap=False)
    result = gating_cluster.extract_and_cluster_gates(
        None, None, labels, config, is_pure_wind=np.array([False] * 4),
    )
    summary = gating.build_summary_json(result, config)
    assert summary["k_opt"] == k_opt
    assert summary["k_selection_basis"] == basis
    assert summary["stability_scores"] == {str(k): v for k, v in stability.items()}
    assert summary["silhouette_scores"] == {"2": 0.3, "3": 0.6, "4": 0.2}
    assert summary["ARI_4way_k4"] == 1.0
    assert summary["ARI_3way_k3"] == 1.0
    assert [seq["is_pure_wind"] for seq in result["sequences"]] == [False] * 4
