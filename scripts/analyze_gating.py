"""
NSMoR Gating Strategy Clustering Analysis.

Unsupervised clustering of MoR routing gate strategies across trials.
Window-free by design. NSMoR is Trial-Start anchored. TTC-50ms is only
for MCMC prior 5-D snapshot. Baseline 5700ms is variant for pure-wind
via TimeWindowConfig, not universal. Manual windows like [-5700:-500]
inject human bias and break unsupervised claim. Clustering is unsupervised
(silhouette selects k without labels); k=4 matches labeling.py cardinality;
k=3 merged is for biological interpretation only and defined as:
    Startle->Escape, Walk+Pre_Active->PreWalk, NoResponse->NoResponse.
Pearson NaN guarded to 0.0 when std==0.

Outputs (all saved to results/ at 300 DPI):
    - gating_umap_true_4way.png
    - gating_umap_true_3way.png
    - gating_umap_pred_k4.png
    - gating_umap_pred_kopt.png
    - gating_trajectories_by_cluster.png (x normalized 0-1)
    - gating_cluster_summary.json
    - gating_cluster_statistics.csv

Usage
-----
CLI::

    python scripts/analyze_gating.py --checkpoint runs/default/best_model.pth
    python scripts/analyze_gating.py --checkpoint runs/default/best_model.pth --dataset data/processed/nsmor_dataset.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from nsmor.analysis.gating_cluster import (
    ClusterGatingConfig,
    GatingClusterAdapter,
    extract_and_cluster_gates,
)
from nsmor.config import Label
from nsmor.config_parser import ExperimentConfig
from nsmor.model_utils import load_model_from_checkpoint as _shared_load_model
from nsmor.nsmor_dataloader import NSMoRDataset, collate_variable_length

# Use non-interactive backend for headless environments
matplotlib.use("Agg")

# -- Logging ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -- Constants --------------------------------------------------------------
DPI: int = 300
OUTPUT_FILES = {
    "umap_true_4way": "gating_umap_true_4way.png",
    "umap_true_3way": "gating_umap_true_3way.png",
    "umap_pred_k4": "gating_umap_pred_k4.png",
    "umap_pred_kopt": "gating_umap_pred_kopt.png",
    "trajectories": "gating_trajectories_by_cluster.png",
    "summary": "gating_cluster_summary.json",
    "statistics": "gating_cluster_statistics.csv",
}

LABEL_NAMES_4WAY: Dict[int, str] = {
    Label.ESCAPE.value: "Escape",
    Label.PREWALK.value: "Prewalk",
    Label.PRE_ACTIVE.value: "Pre_Active",
    Label.NO_RESPONSE.value: "NoResponse",
}

LABEL_NAMES_3WAY: Dict[int, str] = {
    0: "Escape",
    1: "PreWalk",
    2: "NoResponse",
}

# -- Color palettes ---------------------------------------------------------
CMAP_4WAY = {0: "#E64B35", 1: "#4DBBD5", 2: "#00A087", 3: "#3C5488"}
CMAP_3WAY = {0: "#E64B35", 1: "#4DBBD5", 2: "#3C5488"}
CMAP_PRED = plt.cm.tab10


# =========================================================================
# 1. Model & Dataset Loading
# =========================================================================

def load_model_and_dataset(
    checkpoint_path: Path,
    dataset_path: Path,
    batch_size: int = 32,
    max_seq_len: Optional[int] = 1000,
) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader, np.ndarray]:
    """Load model and dataset for gating extraction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model
    model = _shared_load_model(checkpoint_path, device)
    logger.info("Loaded model from %s", checkpoint_path)

    # Load dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info("Loading dataset from %s", dataset_path)
    dataset = torch.load(dataset_path, weights_only=False)

    X_seqs = dataset["X_seqs"]
    Y_seqs = dataset["Y_seqs"]
    mcmc_priors = dataset["mcmc_priors"]
    labels = dataset["labels"]

    n_total = len(X_seqs)
    logger.info("Loaded %d sequences.", n_total)

    sequences = [
        (X_seqs[i], Y_seqs[i], int(labels[i]))
        for i in range(n_total)
    ]

    from nsmor.config import DEFAULT_FEATURE
    feature_config = dataset.get("feature_config", DEFAULT_FEATURE)
    bio_dataset = NSMoRDataset(
        sequences=sequences,
        mcmc_priors=mcmc_priors,
        feature_config=feature_config,
        max_seq_len=max_seq_len,
    )

    dataloader = torch.utils.data.DataLoader(
        bio_dataset,
        batch_size=batch_size,
        shuffle=False,  # Required for deterministic ordering
        num_workers=0,
        collate_fn=collate_variable_length,
    )

    return model, dataloader, labels


# =========================================================================
# 2. Visualization Functions
# =========================================================================

def plot_umap(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    label_names: Dict[int, str],
    color_map: Dict[int, str],
    output_path: Path,
) -> None:
    """Plot UMAP embedding colored by labels."""
    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = sorted(np.unique(labels))
    for lbl in unique_labels:
        mask = labels == lbl
        color = color_map.get(int(lbl), "#888888")
        name = label_names.get(int(lbl), f"Cluster {lbl}")
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            label=f"{name} (n={mask.sum()})",
            alpha=0.7,
            s=30,
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.legend(fontsize=9, frameon=True, loc="best")
    ax.grid(True, alpha=0.15, linestyle="--")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    logger.info("Saved %s", output_path)
    plt.close(fig)


def plot_trajectories_by_cluster(
    sequences: List[Dict[str, Any]],
    cluster_labels: np.ndarray,
    output_path: Path,
    interp_length: int = 200,
) -> None:
    """Plot mean gate trajectories per cluster (x normalized 0-1)."""
    from scipy import interpolate as sp_interp

    # Group by cluster
    cluster_data: Dict[int, List[np.ndarray]] = {}
    for i, seq in enumerate(sequences):
        cid = int(cluster_labels[i])
        gates = seq["gates"]  # (T, 2)
        T = gates.shape[0]
        if T == 0:
            continue

        # Normalize x to 0-1
        x_orig = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, interp_length)

        interp_gates = np.zeros((interp_length, 2))
        for j in range(2):
            f = sp_interp.interp1d(
                x_orig, gates[:, j], kind="linear",
                fill_value="extrapolate",
            )
            interp_gates[:, j] = f(x_new)

        if cid not in cluster_data:
            cluster_data[cid] = []
        cluster_data[cid].append(interp_gates)

    n_clusters = len(cluster_data)
    if n_clusters == 0:
        logger.warning("No clusters to plot trajectories for.")
        return

    fig, axes = plt.subplots(
        n_clusters, 1,
        figsize=(10, 3 * n_clusters),
        sharex=True,
    )
    if n_clusters == 1:
        axes = [axes]

    x_norm = np.linspace(0, 1, interp_length)

    for idx, (cid, trajs) in enumerate(sorted(cluster_data.items())):
        ax = axes[idx]
        trajs_arr = np.array(trajs)  # (n_trials, interp_length, 2)

        mean_lif = trajs_arr[:, :, 0].mean(axis=0)
        std_lif = trajs_arr[:, :, 0].std(axis=0)
        mean_gru = trajs_arr[:, :, 1].mean(axis=0)
        std_gru = trajs_arr[:, :, 1].std(axis=0)

        ax.plot(x_norm, mean_lif, color="#E64B35", linewidth=2.0, label="g_lif (mean)")
        ax.fill_between(
            x_norm, mean_lif - std_lif, mean_lif + std_lif,
            color="#E64B35", alpha=0.15,
        )
        ax.plot(x_norm, mean_gru, color="#4DBBD5", linewidth=2.0, label="g_gru (mean)")
        ax.fill_between(
            x_norm, mean_gru - std_gru, mean_gru + std_gru,
            color="#4DBBD5", alpha=0.15,
        )

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.set_title(f"Cluster {cid} (n={len(trajs)})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Gate Value", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.15, linestyle="--")

    axes[-1].set_xlabel("Normalized Time (0-1)", fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    logger.info("Saved %s", output_path)
    plt.close(fig)


# =========================================================================
# 3. Summary & Statistics
# =========================================================================

def build_summary_json(
    result: Dict[str, Any],
    config: ClusterGatingConfig,
) -> Dict[str, Any]:
    """Build the summary JSON dict."""
    sequences = result["sequences"]
    evaluation = result.get("evaluation", {})

    # True labels arrays
    labels_4way = np.array([s["true_4way"] for s in sequences])
    labels_3way = np.array([s["true_3way_merged"] for s in sequences])

    # ARI / NMI
    ari_4way_k4 = evaluation.get("k4_vs_4way", {}).get("ari", np.nan)
    nmi_4way_k4 = evaluation.get("k4_vs_4way", {}).get("nmi", np.nan)
    ari_3way_k3 = evaluation.get("k3_vs_3way", {}).get("ari", np.nan)
    nmi_3way_k3 = evaluation.get("k3_vs_3way", {}).get("nmi", np.nan)

    # Config hash
    config_dict = {
        "n_clusters": config.n_clusters,
        "n_clusters_range": config.n_clusters_range,
        "random_state": config.random_state,
        "use_umap": config.use_umap,
        "fingerprint_dim": config.fingerprint_dim,
        "entropy_bins": config.entropy_bins,
        "interp_length": config.interp_length,
    }
    config_hash = json.dumps(config_dict, sort_keys=True)

    summary = {
        "k_opt": int(result["k_opt"]),
        "silhouette_scores": {
            str(k): float(v)
            for k, v in result["silhouette_scores"].items()
        },
        "ARI_4way_k4": float(ari_4way_k4) if not np.isnan(ari_4way_k4) else None,
        "NMI_4way_k4": float(nmi_4way_k4) if not np.isnan(nmi_4way_k4) else None,
        "ARI_3way_k3": float(ari_3way_k3) if not np.isnan(ari_3way_k3) else None,
        "NMI_3way_k3": float(nmi_3way_k3) if not np.isnan(nmi_3way_k3) else None,
        "window_used": False,
        "config_hash": config_hash,
        "mapping_definition": {
            "Startle (0)": "Escape (0)",
            "Walk (1)": "PreWalk (1)",
            "Pre_Active (2)": "PreWalk (1)",
            "NoResponse (3)": "NoResponse (2)",
        },
        "n_trials": len(sequences),
        "fingerprint_dim": config.fingerprint_dim,
    }

    return summary


def build_statistics_csv(
    result: Dict[str, Any],
    config: ClusterGatingConfig,
) -> pd.DataFrame:
    """Build the cluster statistics DataFrame."""
    sequences = result["sequences"]
    labels_k4 = result["labels_k4"]
    labels_k3 = result["labels_k3"]
    labels_kopt = result["labels_kopt"]

    rows = []
    for i, seq in enumerate(sequences):
        rows.append({
            "trial_id": seq["trial_id"],
            "length": seq["length"],
            "true_4way": seq["true_4way"],
            "true_3way_merged": seq["true_3way_merged"],
            "cluster_k4": int(labels_k4[i]),
            "cluster_k3": int(labels_k3[i]),
            "cluster_kopt": int(labels_kopt[i]),
        })

    df = pd.DataFrame(rows)

    # Add per-cluster counts
    k4_counts = df["cluster_k4"].value_counts().to_dict()
    k3_counts = df["cluster_k3"].value_counts().to_dict()
    kopt_counts = df["cluster_kopt"].value_counts().to_dict()

    logger.info("Cluster k=4 counts: %s", k4_counts)
    logger.info("Cluster k=3 counts: %s", k3_counts)
    logger.info("Cluster k_opt=%d counts: %s", result["k_opt"], kopt_counts)

    return df


# =========================================================================
# 4. Main Pipeline
# =========================================================================

def run_analysis(
    checkpoint_path: Path,
    dataset_path: Path,
    output_dir: Path,
    batch_size: int = 32,
    max_seq_len: Optional[int] = 1000,
) -> None:
    """Run the full gating cluster analysis pipeline."""
    logger.info("=" * 60)
    logger.info("NSMoR Gating Strategy Clustering Analysis")
    logger.info("=" * 60)

    # Load experiment config
    exp_config = ExperimentConfig.from_yaml("config/default.yaml")
    cluster_config = exp_config.cluster_gating

    # Load model and dataset
    model, dataloader, labels = load_model_and_dataset(
        checkpoint_path, dataset_path, batch_size=batch_size, max_seq_len=max_seq_len,
    )

    # Extract and cluster
    result = extract_and_cluster_gates(
        model=model,
        dataloader=dataloader,
        labels=labels,
        config=cluster_config,
    )

    umap_emb = result.get("umap_embedding")

    # -- Plot UMAP embeddings --
    if umap_emb is not None:
        # True 4-way labels
        labels_4way = np.array([s["true_4way"] for s in result["sequences"]])
        plot_umap(
            umap_emb, labels_4way,
            "UMAP: True 4-Way Labels",
            LABEL_NAMES_4WAY, CMAP_4WAY,
            output_dir / OUTPUT_FILES["umap_true_4way"],
        )

        # True 3-way merged labels
        labels_3way = np.array([s["true_3way_merged"] for s in result["sequences"]])
        plot_umap(
            umap_emb, labels_3way,
            "UMAP: True 3-Way Merged Labels",
            LABEL_NAMES_3WAY, CMAP_3WAY,
            output_dir / OUTPUT_FILES["umap_true_3way"],
        )

        # Predicted k=4 clusters
        plot_umap(
            umap_emb, result["labels_k4"],
            f"UMAP: Predicted Clusters (k=4)",
            {i: f"C{i}" for i in range(4)},
            {i: CMAP_PRED(i) for i in range(4)},
            output_dir / OUTPUT_FILES["umap_pred_k4"],
        )

        # Predicted k_opt clusters
        k_opt = result["k_opt"]
        pred_names = {i: f"C{i}" for i in range(k_opt)}
        pred_colors = {i: CMAP_PRED(i) for i in range(k_opt)}
        plot_umap(
            umap_emb, result["labels_kopt"],
            f"UMAP: Predicted Clusters (k_opt={k_opt})",
            pred_names, pred_colors,
            output_dir / OUTPUT_FILES["umap_pred_kopt"],
        )
    else:
        logger.warning("UMAP embedding not available; skipping UMAP plots.")

    # -- Plot trajectories by cluster --
    plot_trajectories_by_cluster(
        result["sequences"],
        result["labels_k4"],
        output_dir / OUTPUT_FILES["trajectories"],
        interp_length=cluster_config.interp_length,
    )

    # -- Save summary JSON --
    summary = build_summary_json(result, cluster_config)
    summary_path = output_dir / OUTPUT_FILES["summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    logger.info("Saved summary JSON to %s", summary_path)

    # -- Save statistics CSV --
    df = build_statistics_csv(result, cluster_config)
    csv_path = output_dir / OUTPUT_FILES["statistics"]
    df.to_csv(csv_path, index=False)
    logger.info("Saved statistics CSV to %s", csv_path)

    logger.info("=" * 60)
    logger.info("Gating cluster analysis complete!")
    logger.info("=" * 60)


# =========================================================================
# 5. CLI Entry Point
# =========================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="NSMoR Gating Strategy Clustering Analysis (Window-Free)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/nsmor_dataset.pt",
        help="Path to preprocessed dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for all analysis files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for data loading.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1000,
        help="Crop sequences longer than this (cuDNN compatibility). 0 = disable.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    run_analysis(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
    )


if __name__ == "__main__":
    main()
