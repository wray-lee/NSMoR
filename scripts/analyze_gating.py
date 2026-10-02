"""
NSMoR Gating Strategy Clustering Analysis.

Unsupervised clustering of MoR routing gate strategies across trials.
Window-free by design. NSMoR is Trial-Start anchored. TTC-50ms is only
for MCMC prior 5-D snapshot. Baseline 5700ms is variant for pure-wind
via TimeWindowConfig, not universal. Manual windows like [-5700:-500]
inject human bias and break unsupervised claim. Clustering is unsupervised
(silhouette selects among bootstrap-stable k, with a documented fallback);
k=4 matches labeling.py cardinality;
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
import sys
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Direct CLI execution must use this checkout, including its prior/lineage helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Prevent thread contention in scikit-learn / OpenMP / MKL on high-core CPUs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

# Use non-interactive backend for headless environments — set BEFORE pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from nsmor.analysis.gating_cluster import (
    ClusterGatingConfig,
    extract_and_cluster_gates,
)
from nsmor.analysis.uq import cohens_d
from nsmor.config import Label
from nsmor.config_parser import ExperimentConfig
from nsmor.dataloader_factory import create_optimized_dataloader
from nsmor.analysis.analysis_priors import (
    describe_analysis_population, load_analysis_priors, population_for_output,
)
from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint
from nsmor.analysis.prediction_units import (
    load_model_from_checkpoint as _shared_load_model, resolve_dt_ms,
)
from nsmor.model_utils import validate_dataset_provenance
from nsmor.nsmor_dataloader import NSMoRDataset
from nsmor.pipeline.conditions import derive_stimulus_metadata

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
    max_seq_len: Optional[int] = 2400,
    pre_anchor_frames: int = 1200,
    nested_prior_artifact: Optional[Path] = None,
    qc_sealed_nested_prior_sha256: Optional[str] = None,
    trusted_historical_checkpoint_sha256: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load model and dataset for gating extraction.

    Returns:
        (model, dataloader, labels, is_pure_wind, stimulus_conditions)
        where is_pure_wind and stimulus_conditions are None if not in dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model
    model = _shared_load_model(checkpoint_path, device)
    logger.info("Loaded model from %s", checkpoint_path)

    # Load dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info("Loading dataset from %s", dataset_path)
    dataset, loaded_source_fingerprint = load_dataset_with_fingerprint(
        dataset_path, expected_dt_ms=resolve_dt_ms(model), restore_provenance=False,
    )

    # Round-2 CRITICAL-A: refuse pre-2.0 datasets (leaked priors, np.max labels)
    validate_dataset_provenance(dataset, Path(dataset_path))

    X_seqs = dataset["X_seqs"]
    Y_seqs = dataset["Y_seqs"]
    mcmc_priors, val_indices = load_analysis_priors(
        dataset, dataset_path, nested_prior_artifact, model, qc_sealed_nested_prior_sha256,
        loaded_source_fingerprint=loaded_source_fingerprint,
        trusted_historical_checkpoint_sha256=trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
    )
    labels = dataset["labels"]

    # Ticket #17: Load stimulus condition metadata if available
    is_pure_wind = dataset.get("is_pure_wind")
    stimulus_conditions = dataset.get("stimulus_conditions")
    if is_pure_wind is None:
        # Legacy artifacts predate the condition stamp. Derive it with the
        # canonical classifier rather than a local approximation: pure-wind
        # is "wind present AND visual absent", so a no_stimulus trial (both
        # channels silent) must NOT land in the wind group.
        lengths = dataset.get("lengths")
        if lengths is None:
            lengths = [len(x) for x in X_seqs]
        stimulus_conditions, is_pure_wind = derive_stimulus_metadata(
            X_seqs, lengths
        )
        logger.info(
            "Dataset lacks is_pure_wind; derived from physical channels "
            "(visual_angle/wind_state)."
        )

    n_total = len(X_seqs)
    logger.info("Loaded %d sequences.", n_total)
    if is_pure_wind is not None:
        logger.info(
            "Stimulus condition metadata: %d wind_only, %d other",
            int(np.sum(is_pure_wind)), int(np.sum(~is_pure_wind)),
        )
        if not np.any(is_pure_wind):
            logger.warning(
                "No wind_only trials in this corpus -- per-condition gate "
                "statistics will be reported as unavailable rather than "
                "computed against an empty group."
            )

    anchor_frames = dataset.get("anchor_frames")
    if anchor_frames is None:
        from nsmor.pipeline.conditions import derive_anchor_frames
        lengths = dataset.get("lengths")
        anchor_frames = derive_anchor_frames(X_seqs, lengths)
        logger.info(
            "Derived %d anchor frames from physical channels.",
            len(anchor_frames),
        )

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
        pre_anchor_frames=pre_anchor_frames,
        anchor_frames=anchor_frames,
        is_pure_wind=is_pure_wind,  # Ticket #17
    )

    bio_dataset.analysis_population = describe_analysis_population(
        n_total, val_indices, bio_dataset.source_indices,
        getattr(model, "analysis_validation_scope", None),
    )
    dataloader = create_optimized_dataloader(
        bio_dataset,
        batch_size=batch_size,
        shuffle=False,  # Required for deterministic ordering
        num_workers=0,  # Whole corpus is already resident; workers only replicate it
    )

    return model, dataloader, labels, is_pure_wind, stimulus_conditions


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
            color=color,
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
    fig.suptitle("Whole corpus — in-sample descriptive gate trajectories", fontsize=12)
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

    # Ticket #17: Per-condition gate statistics (if metadata available)
    condition_stats = _compute_condition_gate_stats(sequences)

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
        "stability_scores": {
            str(k): None if v is None else float(v)
            for k, v in result["stability_scores"].items()
        },
        "k_selection_basis": result["k_selection_basis"],
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

    # Ticket #17: Add condition-specific routing stats if available
    if condition_stats is not None:
        summary["condition_specific_routing"] = condition_stats

    return summary


def _compute_condition_gate_stats(
    sequences: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Compute per-condition gate statistics (Ticket #17).

    Args:
        sequences: List of sequence dicts from extract_and_cluster_gates.
            Each dict must have 'gate_seq' (T, 2) with [:, 0]=g_lif, [:, 1]=g_gru.
            Optional: 'is_pure_wind' (bool) or 'stimulus_condition' (str).

    Returns:
        Descriptive condition counts and contrast statistics, or None when
        all metadata is absent. Unknown/conflicting conditions are excluded
        explicitly; a false pure-wind flag alone never establishes vision.
        Multisensory and visual-only counts remain separate within the
        visual-present contrast. Undefined effect sizes are None with a reason.
    """
    # Check if any sequence has condition metadata
    has_wind_flag = any("is_pure_wind" in s for s in sequences)
    has_condition = any("stimulus_condition" in s for s in sequences)

    if not (has_wind_flag or has_condition):
        return None  # No metadata, skip

    # A false pure-wind flag does not establish visual presence: it also
    # describes no-stimulus trials. Keep the four physical conditions distinct.
    groups: Dict[str, List[float]] = {
        key: [] for key in (
            "wind_only", "visual_only", "multisensory", "no_stimulus"
        )
    }
    excluded: Dict[str, int] = {}
    for seq in sequences:
        condition = seq.get("stimulus_condition")
        wind_flag = seq.get("is_pure_wind")
        reason = None
        if condition is None:
            if isinstance(wind_flag, (bool, np.bool_)) and wind_flag:
                condition = "wind_only"
            else:
                reason = "missing_condition"
        elif condition not in groups:
            reason = "unknown_condition"
        if reason is None and wind_flag is not None:
            if not isinstance(wind_flag, (bool, np.bool_)) or bool(wind_flag) != (
                condition == "wind_only"
            ):
                reason = "conflicting_metadata"
        gate_seq = seq.get("gates")
        if gate_seq is None:
            gate_seq = seq.get("gate_seq")
        if reason is None:
            if gate_seq is None:
                reason = "missing_gates"
            else:
                gates = np.asarray(gate_seq)
                assert gates.ndim == 2 and gates.shape[1] == 2, (
                    f"Expected gates (T, 2), got {gates.shape}"
                )
                if len(gates) == 0 or not np.isfinite(gates).all():
                    reason = "empty_or_nonfinite_gates"
        if reason is not None:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        groups[condition].append(float(np.mean(gates[:, 0])))

    wind_g_lif_trials = groups["wind_only"]
    visual_g_lif_trials = groups["visual_only"] + groups["multisensory"]
    audit = {
        "condition_counts": {key: len(values) for key, values in groups.items()},
        "per_condition": {
            key: {
                "n_trials": len(values),
                "mean_g_lif": float(np.mean(values)) if values else None,
                "std_g_lif": float(np.std(values)) if values else None,
            }
            for key, values in groups.items()
        },
        "excluded_trial_counts": excluded,
        "visual_group_conditions": ["visual_only", "multisensory"],
        "evidence_scope": "in_sample_descriptive_only",
    }

    # Keep condition/exclusion counts even when a contrast cannot be computed.
    if not wind_g_lif_trials or not visual_g_lif_trials:
        return {
            **audit,
            "comparison_available": False,
            "unavailable_reason": "missing_wind_or_visual_present_group",
            "mean_g_lif_wind": None,
            "mean_g_lif_visual": None,
            "std_g_lif_wind": None,
            "std_g_lif_visual": None,
            "separation": None,
            "cohens_d": None,
            "cohens_d_unavailable_reason": "missing_wind_or_visual_present_group",
            "n_wind_trials": len(wind_g_lif_trials),
            "n_visual_trials": len(visual_g_lif_trials),
        }

    mean_g_lif_wind = float(np.mean(wind_g_lif_trials))
    mean_g_lif_visual = float(np.mean(visual_g_lif_trials))
    std_g_lif_wind = float(np.std(wind_g_lif_trials))
    std_g_lif_visual = float(np.std(visual_g_lif_trials))
    separation = abs(mean_g_lif_wind - mean_g_lif_visual)

    # Trial-level descriptive effect, not animal-independent inference.
    effect = cohens_d(
        np.asarray(wind_g_lif_trials), np.asarray(visual_g_lif_trials)
    )
    effect_reason = None
    if len(wind_g_lif_trials) < 2 or len(visual_g_lif_trials) < 2:
        effect_reason = "fewer_than_two_trials_per_group"
    elif not np.isfinite(effect):
        effect_reason = "zero_pooled_variance"

    return {
        **audit,
        "comparison_available": True,
        "mean_g_lif_wind": mean_g_lif_wind,
        "mean_g_lif_visual": mean_g_lif_visual,
        "std_g_lif_wind": std_g_lif_wind,
        "std_g_lif_visual": std_g_lif_visual,
        "separation": float(separation),
        "cohens_d": float(effect) if np.isfinite(effect) else None,
        "cohens_d_unavailable_reason": effect_reason,
        "n_wind_trials": len(wind_g_lif_trials),
        "n_visual_trials": len(visual_g_lif_trials),
    }


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
    max_seq_len: Optional[int] = 2400,
    pre_anchor_frames: int = 1200,
    config_path: Optional[Path] = None,
    nested_prior_artifact: Optional[Path] = None,
    qc_sealed_nested_prior_sha256: Optional[str] = None,
    trusted_historical_checkpoint_sha256: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
) -> None:
    """Run the full gating cluster analysis pipeline."""
    logger.info("=" * 60)
    logger.info("NSMoR Gating Strategy Clustering Analysis")
    logger.info("=" * 60)

    # Load experiment config.  The path used to be hardcoded and
    # cwd-relative, so the clustering silently used the repo default
    # config no matter which config trained the checkpoint — and the
    # script could only run from the repository root.
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    logger.info("Cluster config: %s", config_path)
    exp_config = ExperimentConfig.from_yaml(config_path)
    cluster_config = exp_config.cluster_gating

    # Load model and dataset
    model, dataloader, labels, is_pure_wind, stimulus_conditions = load_model_and_dataset(
        checkpoint_path,
        dataset_path,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        nested_prior_artifact=nested_prior_artifact,
        qc_sealed_nested_prior_sha256=qc_sealed_nested_prior_sha256,
        trusted_historical_checkpoint_sha256=trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
    )

    # Extract and cluster
    result = extract_and_cluster_gates(
        model=model,
        dataloader=dataloader,
        labels=labels,
        config=cluster_config,
        is_pure_wind=is_pure_wind,  # Ticket #17
        stimulus_conditions=stimulus_conditions,  # Ticket #17
    )

    umap_emb = result.get("umap_embedding")

    # -- Plot UMAP embeddings --
    if umap_emb is not None:
        # True 4-way labels
        labels_4way = np.array([s["true_4way"] for s in result["sequences"]])
        plot_umap(
            umap_emb, labels_4way,
            "UMAP: True 4-Way Labels (whole corpus; in-sample descriptive)",
            LABEL_NAMES_4WAY, CMAP_4WAY,
            output_dir / OUTPUT_FILES["umap_true_4way"],
        )

        # True 3-way merged labels
        labels_3way = np.array([s["true_3way_merged"] for s in result["sequences"]])
        plot_umap(
            umap_emb, labels_3way,
            "UMAP: True 3-Way Merged Labels (whole corpus; in-sample descriptive)",
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
            f"UMAP: Predicted Clusters (k_opt={k_opt}; whole corpus, in-sample descriptive)",
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
    summary["analysis_population"] = population_for_output(dataloader, len(labels))
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

    # -- Per-Condition Gate Statistics (Ticket #17) --
    cond_stats = _compute_condition_gate_stats(result["sequences"])
    if cond_stats is not None and cond_stats["comparison_available"]:
        logger.info("=" * 60)
        logger.info("Per-Condition Gate Statistics (whole corpus; in-sample descriptive):")
        logger.info(
            f"  Pure-wind trials (N={cond_stats['n_wind_trials']}):  "
            f"mean g_lif = {cond_stats['mean_g_lif_wind']:.3f} ± {cond_stats['std_g_lif_wind']:.3f}"
        )
        logger.info(
            f"  Visual-present (N={cond_stats['n_visual_trials']}):    "
            f"mean g_lif = {cond_stats['mean_g_lif_visual']:.3f} ± {cond_stats['std_g_lif_visual']:.3f}"
        )
        logger.info(f"  Separation: |Δ| = {cond_stats['separation']:.3f}")
        logger.info("=" * 60)
    else:
        logger.info("Stimulus condition metadata (is_pure_wind) not available or insufficient groups; per-condition gate stats skipped.")


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
        "--nested_prior_artifact",
        type=str,
        default=None,
        help="Optional validated nested prior artifact used by the checkpoint.",
    )
    parser.add_argument(
        "--qc_sealed_nested_prior_sha256", type=str, default=None,
        help="Optional external QC-sealed SHA-256 to check the nested checkpoint's embedded artifact digest; cannot authenticate a checkpoint missing that digest.",
    )
    parser.add_argument(
        "--trusted_historical_checkpoint_sha256", type=str, default=None,
        help="Independently pinned SHA-256 of historical checkpoint bytes (required for historical analysis).",
    )
    parser.add_argument(
        "--trusted_historical_artifact_sha256", type=str, default=None,
        help="Independently pinned SHA-256 of historical nested artifact bytes.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config supplying cluster_gating settings "
             "(default: <repo>/config/default.yaml).",
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
        default=2400,
        help="Crop sequences longer than this (cuDNN compatibility). 0 = disable.",
    )
    parser.add_argument(
        "--pre_anchor_frames",
        type=int,
        default=1200,
        help="Number of frames before anchor to include in anchor-aligned crop.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    pre_anchor_frames = getattr(args, "pre_anchor_frames", 1200)
    run_analysis(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        nested_prior_artifact=Path(args.nested_prior_artifact) if args.nested_prior_artifact else None,
        qc_sealed_nested_prior_sha256=args.qc_sealed_nested_prior_sha256,
        trusted_historical_checkpoint_sha256=args.trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=args.trusted_historical_artifact_sha256,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        config_path=Path(args.config) if args.config else None,
    )


if __name__ == "__main__":
    main()
