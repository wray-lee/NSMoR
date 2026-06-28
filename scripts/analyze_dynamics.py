"""
NSMoR Mechanism Analysis — Dual-Panel Publication Figure.

Generates a Lancet/Cell-quality dual-panel figure for publication:
  - Panel A: 3D Phase-Space Manifold (PCA of GRU hidden states)
  - Panel B: Temporal Gating Dynamics (MoR routing probabilities)

Output: ``results/mechanism_analysis.png`` at 300 DPI.

Usage
-----
CLI::

    python scripts/analyze_dynamics.py --checkpoint runs/default/best_model.pth
    python scripts/analyze_dynamics.py --checkpoint runs/default/best_model.pth --dataset data/processed/nsmor_dataset.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D projection)
from sklearn.decomposition import PCA

from nsmor.analysis.dynamics import FixedPointAdapter
from nsmor.nsmor_dataloader import (
    NSMoRDataset,
    collate_variable_length,
)
from nsmor.checkpoint import load_checkpoint
from nsmor.config import DEFAULT_FEATURE, Label
from nsmor.model_nsmor_core import NSMoRCore

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Lancet / Cell Publication Style Constants
# ═══════════════════════════════════════════════════════════════

# ── High-contrast categorical color mapping ──
# STRICTLY REJECT pale, desaturated, or pastel palettes
LANCET_COLORS: Dict[int, str] = {
    Label.ESCAPE.value: "#C92A2A",      # Lancet Crimson Red — urgent reflex
    Label.PREWALK.value: "#1C7ED6",         # Cell Cobalt Blue — continuous locomotion
    Label.PRE_ACTIVE.value: "#495057",   # Strong Slate Gray — baseline activity
    Label.NO_RESPONSE.value: "#000000",  # Absolute Jet Black — non-responsive control
}

# Routing gate curve colors
GATE_GRU_COLOR: str = "#1C7ED6"   # Cell Cobalt Blue for g_gru(t)
GATE_LIF_COLOR: str = "#C92A2A"   # Lancet Crimson Red for g_lif(t)

# Label display names
LABEL_NAMES: Dict[int, str] = {
    Label.ESCAPE.value: "Startle",
    Label.PREWALK.value: "Prewalk",
    Label.PRE_ACTIVE.value: "Pre-Active",
    Label.NO_RESPONSE.value: "No Response",
}

# Marker styles per label
LABEL_MARKERS: Dict[int, str] = {
    Label.ESCAPE.value: "^",    # Triangle up (urgency)
    Label.PREWALK.value: "o",       # Circle (locomotion)
    Label.PRE_ACTIVE.value: "s", # Square (baseline)
    Label.NO_RESPONSE.value: "D", # Diamond (control)
}

# ── Typography ─────────────────────────────────────────────────
FONT_FAMILY: str = "Arial"
FONT_SIZE_AXIS_TITLE: int = 12
FONT_SIZE_TICK: int = 10
FONT_SIZE_LEGEND: int = 9
FONT_SIZE_PANEL_LABEL: int = 14

# ── Figure properties ─────────────────────────────────────────
DPI: int = 300
FIG_WIDTH_INCHES: float = 14.0
FIG_HEIGHT_INCHES: float = 6.0
BACKGROUND_COLOR: str = "#FFFFFF"
AXIS_COLOR: str = "#212529"  # Solid dark charcoal

# ── Plot properties ───────────────────────────────────────────
LINE_WIDTH: float = 2.5
SCATTER_ALPHA: float = 0.75
SCATTER_SIZE: float = 25.0
TRAJECTORY_ALPHA: float = 0.4
TRAJECTORY_LINEWIDTH: float = 0.8


# ═══════════════════════════════════════════════════════════════
# 1.  Model Loading
# ═══════════════════════════════════════════════════════════════

def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> NSMoRCore:
    """
    Load a trained NSMoRCore model from a checkpoint.

    Args:
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        device: Device to load the model onto.

    Returns:
        Loaded model in eval mode.

    Raises:
        FileNotFoundError: If checkpoint does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    config_dict = checkpoint.get("config", {})
    model_config = config_dict.get("model", {})

    # Build model with saved config
    model = NSMoRCore(
        sensory_dim=model_config.get("sensory_dim", 4),
        mcmc_dim=model_config.get("mcmc_dim", 4),
        hidden_dim=model_config.get("hidden_dim", 64),
        num_gru_layers=model_config.get("num_gru_layers", 1),
        dropout=model_config.get("dropout", 0.1),
        lif_alpha=model_config.get("lif_alpha", 0.9),
        lif_threshold=model_config.get("lif_threshold", 1.0),
        lif_beta=model_config.get("lif_beta", 0.5),
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded: %s parameters, hidden_dim=%d",
        f"{param_count:,}", model.hidden_dim,
    )

    return model


# ═══════════════════════════════════════════════════════════════
# 2.  Dataset Loading
# ═══════════════════════════════════════════════════════════════

def load_dataset(
    dataset_path: Path,
    batch_size: int = 32,
    max_seq_len: Optional[int] = 1000,
) -> Tuple[torch.utils.data.DataLoader, np.ndarray]:
    """
    Load the preprocessed dataset and create a DataLoader.

    Args:
        dataset_path: Path to ``nsmor_dataset.pt``.
        batch_size: Batch size for the DataLoader.

    Returns:
        ``(dataloader, labels)`` tuple.

    Raises:
        FileNotFoundError: If dataset file does not exist.
    """
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

    # Build sequence list
    sequences = [
        (X_seqs[i], Y_seqs[i], int(labels[i]))
        for i in range(n_total)
    ]

    # Create dataset and dataloader
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
        shuffle=False,  # Preserve ordering for label matching
        num_workers=0,
        collate_fn=collate_variable_length,
    )

    return dataloader, labels


# ═══════════════════════════════════════════════════════════════
# 3.  Trajectory Extraction and PCA
# ═══════════════════════════════════════════════════════════════

def extract_and_reduce(
    model: NSMoRCore,
    dataloader: torch.utils.data.DataLoader,
    labels: np.ndarray,
    device: torch.device,
    n_components: int = 3,
) -> Tuple[List[np.ndarray], np.ndarray, PCA]:
    """
    Extract GRU trajectories and reduce to 3D via PCA.

    Args:
        model: Trained NSMoRCore model.
        dataloader: DataLoader yielding (X, Y, lengths) tuples.
        labels: Ground truth labels for each sequence.
        device: Computation device.
        n_components: Number of PCA components (default 3).

    Returns:
        ``(trajectories_3d, all_labels, pca)`` where:
        - ``trajectories_3d``: List of arrays, each (T_i, 3)
        - ``all_labels``: Concatenated labels for each state
        - ``pca``: Fitted PCA object
    """
    # ── Extract GRU states ────────────────────────────────────
    logger.info("Extracting GRU hidden states...")
    adapter = FixedPointAdapter(model, device=device)
    trajectories = adapter.extract_gru_states(dataloader)

    logger.info(
        "Extracted %d trajectories, total_states=%d",
        len(trajectories), sum(t.shape[0] for t in trajectories),
    )

    # ── Concatenate all valid states for PCA fitting ──────────
    all_states = torch.cat(trajectories, dim=0).numpy()
    logger.info("Concatenated states shape: %s", all_states.shape)

    # ── Build per-state labels ────────────────────────────────
    all_labels_list = []
    for i, traj in enumerate(trajectories):
        T_i = traj.shape[0]
        all_labels_list.extend([labels[i]] * T_i)
    all_labels = np.array(all_labels_list, dtype=np.int64)

    assert all_labels.shape[0] == all_states.shape[0], (
        f"Label/state count mismatch: {all_labels.shape[0]} vs {all_states.shape[0]}"
    )

    # ── Fit PCA ───────────────────────────────────────────────
    logger.info("Fitting PCA with %d components...", n_components)
    pca = PCA(n_components=n_components)
    pca.fit(all_states)

    explained_var = pca.explained_variance_ratio_
    logger.info(
        "PCA explained variance: %.2f%%, %.2f%%, %.2f%% (total=%.2f%%)",
        explained_var[0] * 100, explained_var[1] * 100, explained_var[2] * 100,
        sum(explained_var) * 100,
    )

    # ── Transform trajectories ────────────────────────────────
    trajectories_3d = []
    for traj in trajectories:
        traj_np = traj.numpy()
        traj_3d = pca.transform(traj_np)
        assert traj_3d.shape == (traj_np.shape[0], n_components), (
            f"PCA transform shape mismatch: {traj_3d.shape} != "
            f"({traj_np.shape[0]}, {n_components})"
        )
        trajectories_3d.append(traj_3d)

    logger.info("Transformed %d trajectories to 3D.", len(trajectories_3d))

    return trajectories_3d, all_labels, pca


# ═══════════════════════════════════════════════════════════════
# 4.  Routing Gate Extraction (for Panel B)
# ═══════════════════════════════════════════════════════════════

def extract_routing_gates(
    model: NSMoRCore,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_representative: int = 5,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Extract MoR routing gate trajectories for representative trials.

    Args:
        model: Trained NSMoRCore model.
        dataloader: DataLoader yielding (X, Y, lengths) tuples.
        device: Computation device.
        n_representative: Number of representative trials to extract.

    Returns:
        ``(g_gru_trajs, g_lif_trajs, traj_labels)`` where:
        - ``g_gru_trajs``: List of arrays, each (T_i,) for g_gru(t)
        - ``g_lif_trajs``: List of arrays, each (T_i,) for g_lif(t)
        - ``traj_labels``: Array of labels for each trajectory
    """
    logger.info("Extracting routing gates for %d representative trials...", n_representative)

    model.eval()
    g_gru_trajs: List[np.ndarray] = []
    g_lif_trajs: List[np.ndarray] = []
    traj_labels_list: List[int] = []
    extracted_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if extracted_count >= n_representative:
                break

            X_batch, Y_batch, lengths = batch
            X_batch = X_batch.to(device).contiguous()
            lengths = lengths.to(device).contiguous()

            # Forward pass with internals
            _, internals = model(X_batch, lengths, return_internals=True)

            # routing_gates shape: [B, T, 2] where [0]=g_lif, [1]=g_gru
            routing_gates = internals["routing_gates"].cpu().numpy()  # (B, T, 2)

            B, T, _ = routing_gates.shape

            for i in range(B):
                if extracted_count >= n_representative:
                    break

                length_i = int(lengths[i].item())
                g_gru_i = routing_gates[i, :length_i, 1]  # g_gru column
                g_lif_i = routing_gates[i, :length_i, 0]  # g_lif column

                # Shape assertions
                assert g_gru_i.shape == (length_i,), (
                    f"g_gru shape {g_gru_i.shape} != ({length_i},)"
                )
                assert g_lif_i.shape == (length_i,), (
                    f"g_lif shape {g_lif_i.shape} != ({length_i},)"
                )

                # Compute time relative to TTC (approximate: midpoint = TTC)
                # For visualization, we center at the middle of the sequence
                g_gru_trajs.append(g_gru_i)
                g_lif_trajs.append(g_lif_i)

                # Get label from dataset ordering
                global_idx = batch_idx * B + i
                if global_idx < len(dataloader.dataset):
                    # Access label from the dataset
                    _, _, label_val = dataloader.dataset.sequences[global_idx]
                    traj_labels_list.append(int(label_val))
                else:
                    traj_labels_list.append(0)  # Default

                extracted_count += 1

    traj_labels = np.array(traj_labels_list, dtype=np.int64)
    logger.info("Extracted %d routing gate trajectories.", len(g_gru_trajs))

    return g_gru_trajs, g_lif_trajs, traj_labels


# ═══════════════════════════════════════════════════════════════
# 5.  Lancet/Cell Dual-Panel Figure
# ═══════════════════════════════════════════════════════════════

def setup_lancet_style() -> None:
    """Configure matplotlib for Lancet/Cell publication aesthetics."""
    plt.rcParams.update({
        # ── Font ──
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE_TICK,
        "axes.titlesize": FONT_SIZE_AXIS_TITLE,
        "axes.labelsize": FONT_SIZE_AXIS_TITLE,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,

        # ── Axes ──
        "axes.linewidth": 1.5,
        "axes.edgecolor": AXIS_COLOR,
        "axes.labelcolor": AXIS_COLOR,
        "xtick.color": AXIS_COLOR,
        "ytick.color": AXIS_COLOR,

        # ── Grid ──
        "axes.grid": False,  # Disabled by default
        "grid.alpha": 0.15,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,

        # ── Figure ──
        "figure.facecolor": BACKGROUND_COLOR,
        "savefig.facecolor": BACKGROUND_COLOR,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,

        # ── Legend ──
        "legend.frameon": True,
        "legend.facecolor": BACKGROUND_COLOR,
        "legend.edgecolor": AXIS_COLOR,
        "legend.framealpha": 1.0,
    })


def plot_panel_a_3d_manifold(
    ax: Axes3D,
    trajectories_3d: List[np.ndarray],
    labels: np.ndarray,
    pca_explained_var: np.ndarray,
) -> None:
    """
    Plot Panel A: 3D Phase-Space Manifold.

    Args:
        ax: Matplotlib 3D axes.
        trajectories_3d: List of arrays, each (T_i, 3).
        labels: Per-sequence labels.
        pca_explained_var: PCA explained variance ratios.
    """
    # ── Plot each trajectory ──────────────────────────────────
    for i, traj in enumerate(trajectories_3d):
        label_val = int(labels[i])
        color = LANCET_COLORS.get(label_val, "#000000")

        # Plot trajectory line with subtle alpha
        ax.plot(
            traj[:, 0], traj[:, 1], traj[:, 2],
            color=color,
            alpha=TRAJECTORY_ALPHA,
            linewidth=TRAJECTORY_LINEWIDTH,
            solid_capstyle="round",
        )

    # ── Plot scatter points (one per label type for legend) ──
    legend_handles = []
    seen_labels = set()
    for i, traj in enumerate(trajectories_3d):
        label_val = int(labels[i])
        if label_val in seen_labels:
            continue
        seen_labels.add(label_val)

        color = LANCET_COLORS.get(label_val, "#000000")
        marker = LABEL_MARKERS.get(label_val, "o")
        label_name = LABEL_NAMES.get(label_val, f"Unknown({label_val})")
        count = int(np.sum(labels == label_val))

        # Subsample for scatter (max 500 points per label)
        n_points = traj.shape[0]
        if n_points > 500:
            indices = np.linspace(0, n_points - 1, 500, dtype=int)
            traj_sub = traj[indices]
        else:
            traj_sub = traj

        scatter = ax.scatter(
            traj_sub[:, 0], traj_sub[:, 1], traj_sub[:, 2],
            color=color,
            marker=marker,
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            edgecolors="white",
            linewidths=0.3,
            label=f"{label_name} (n={count})",
            depthshade=True,
            zorder=3,
        )
        legend_handles.append(scatter)

    # ── Axes styling ──────────────────────────────────────────
    ax.set_xlabel(
        f"PC1 ({pca_explained_var[0]*100:.1f}%)",
        fontsize=FONT_SIZE_AXIS_TITLE,
        color=AXIS_COLOR,
        labelpad=10,
    )
    ax.set_ylabel(
        f"PC2 ({pca_explained_var[1]*100:.1f}%)",
        fontsize=FONT_SIZE_AXIS_TITLE,
        color=AXIS_COLOR,
        labelpad=10,
    )
    ax.set_zlabel(
        f"PC3 ({pca_explained_var[2]*100:.1f}%)",
        fontsize=FONT_SIZE_AXIS_TITLE,
        color=AXIS_COLOR,
        labelpad=10,
    )

    # Set pane colors to white
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(AXIS_COLOR)
    ax.yaxis.pane.set_edgecolor(AXIS_COLOR)
    ax.zaxis.pane.set_edgecolor(AXIS_COLOR)

    # Grid: ultra-faint major grid lines
    ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)

    # Legend
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        facecolor=BACKGROUND_COLOR,
        edgecolor=AXIS_COLOR,
        framealpha=1.0,
    )

    # Panel label
    ax.set_title(
        "A",
        fontsize=FONT_SIZE_PANEL_LABEL,
        fontweight="bold",
        color=AXIS_COLOR,
        loc="left",
        pad=20,
    )


def plot_panel_b_temporal_gating(
    ax: plt.Axes,
    g_gru_trajs: List[np.ndarray],
    g_lif_trajs: List[np.ndarray],
    traj_labels: np.ndarray,
    dt_ms: float = 10.0,
) -> None:
    """
    Plot Panel B: Temporal Gating Dynamics.

    Shows g_gru(t) and g_lif(t) over time relative to TTC for
    representative multi-sensory trials.

    Args:
        ax: Matplotlib axes.
        g_gru_trajs: List of arrays, each (T_i,) for g_gru(t).
        g_lif_trajs: List of arrays, each (T_i,) for g_lif(t).
        traj_labels: Array of labels for each trajectory.
        dt_ms: Frame interval in milliseconds.
    """
    # ── Plot each trajectory ──────────────────────────────────
    for i, (g_gru, g_lif, label_val) in enumerate(
        zip(g_gru_trajs, g_lif_trajs, traj_labels)
    ):
        T_i = len(g_gru)

        # Time relative to TTC (approximate: center = TTC)
        # We normalize so that midpoint = 0 (TTC reference)
        time_rel_ttc = (np.arange(T_i) - T_i // 2) * dt_ms / 1000.0  # seconds

        # Only plot multi-sensory trials (not pure wind)
        if np.all(g_gru < 0.01):  # Skip if no gating activity
            continue

        # Plot g_gru(t) in Cell Cobalt Blue
        ax.plot(
            time_rel_ttc, g_gru,
            color=GATE_GRU_COLOR,
            linewidth=LINE_WIDTH,
            alpha=0.8,
            solid_capstyle="round",
        )

        # Plot g_lif(t) in Lancet Crimson Red
        ax.plot(
            time_rel_ttc, g_lif,
            color=GATE_LIF_COLOR,
            linewidth=LINE_WIDTH,
            alpha=0.8,
            linestyle="--",
            solid_capstyle="round",
        )

    # ── Reference lines ───────────────────────────────────────
    # TTC vertical line
    ax.axvline(
        x=0.0,
        color=AXIS_COLOR,
        linewidth=1.0,
        linestyle=":",
        alpha=0.5,
        label="TTC",
    )

    # 50% probability horizontal line
    ax.axhline(
        y=0.5,
        color=AXIS_COLOR,
        linewidth=0.8,
        linestyle="--",
        alpha=0.3,
    )

    # ── Axes styling ──────────────────────────────────────────
    ax.set_xlabel(
        "Time relative to TTC (s)",
        fontsize=FONT_SIZE_AXIS_TITLE,
        color=AXIS_COLOR,
    )
    ax.set_ylabel(
        "Routing Probability",
        fontsize=FONT_SIZE_AXIS_TITLE,
        color=AXIS_COLOR,
    )

    # Set axis limits
    ax.set_ylim(-0.05, 1.05)

    # Tick formatting
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.tick_params(axis="both", colors=AXIS_COLOR, width=1.5)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(1.5)

    # Grid: ultra-faint major grid lines
    ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)

    # Legend with custom handles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=GATE_GRU_COLOR, linewidth=LINE_WIDTH,
               label=r"$g_{gru}(t)$ — GRU pathway"),
        Line2D([0], [0], color=GATE_LIF_COLOR, linewidth=LINE_WIDTH,
               linestyle="--", label=r"$g_{lif}(t)$ — LIF pathway"),
        Line2D([0], [0], color=AXIS_COLOR, linewidth=1.0,
               linestyle=":", label="TTC"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        facecolor=BACKGROUND_COLOR,
        edgecolor=AXIS_COLOR,
        framealpha=1.0,
    )

    # Panel label
    ax.set_title(
        "B",
        fontsize=FONT_SIZE_PANEL_LABEL,
        fontweight="bold",
        color=AXIS_COLOR,
        loc="left",
    )


def create_dual_panel_figure(
    trajectories_3d: List[np.ndarray],
    all_labels: np.ndarray,
    pca_explained_var: np.ndarray,
    g_gru_trajs: List[np.ndarray],
    g_lif_trajs: List[np.ndarray],
    traj_labels: np.ndarray,
    output_path: Path,
    dt_ms: float = 10.0,
) -> None:
    """
    Create the Lancet/Cell dual-panel publication figure.

    Layout: [1, 2] — Panel A (left), Panel B (right)

    Args:
        trajectories_3d: List of 3D trajectory arrays.
        all_labels: Per-state labels for Panel A.
        pca_explained_var: PCA explained variance ratios.
        g_gru_trajs: Routing gate trajectories for Panel B.
        g_lif_trajs: LIF routing gate trajectories for Panel B.
        traj_labels: Labels for Panel B trajectories.
        output_path: Path to save the figure.
        dt_ms: Frame interval in milliseconds.
    """
    # ── Apply Lancet/Cell style ───────────────────────────────
    setup_lancet_style()

    # ── Create figure with [1, 2] layout ─────────────────────
    fig = plt.figure(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))

    # Panel A: 3D Manifold (left)
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    plot_panel_a_3d_manifold(ax_3d, trajectories_3d, all_labels, pca_explained_var)

    # Panel B: Temporal Gating (right)
    ax_gate = fig.add_subplot(1, 2, 2)
    plot_panel_b_temporal_gating(
        ax_gate, g_gru_trajs, g_lif_trajs, traj_labels, dt_ms=dt_ms
    )

    # ── Save ──────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    logger.info("Saved dual-panel figure to %s (%d DPI)", output_path, DPI)

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 6.  Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════

def run_analysis(
    checkpoint_path: Path,
    dataset_path: Path,
    output_path: Path,
    batch_size: int = 32,
    max_seq_len: Optional[int] = 1000,
    n_representative: int = 5,
) -> None:
    """
    Run the full dual-panel mechanism analysis pipeline.

    Args:
        checkpoint_path: Path to the trained model checkpoint.
        dataset_path: Path to the preprocessed dataset.
        output_path: Path to save the dual-panel figure.
        batch_size: Batch size for data loading.
        n_representative: Number of representative trials for Panel B.
    """
    logger.info("=" * 60)
    logger.info("NSMoR Mechanism Analysis — Dual-Panel Figure")
    logger.info("=" * 60)

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────
    model = load_model_from_checkpoint(checkpoint_path, device)

    # ── Load dataset ──────────────────────────────────────────
    dataloader, labels = load_dataset(dataset_path, batch_size=batch_size, max_seq_len=max_seq_len)

    # ── Panel A: Extract and reduce ───────────────────────────
    trajectories_3d, all_labels, pca = extract_and_reduce(
        model, dataloader, labels, device, n_components=3,
    )

    # ── Panel B: Extract routing gates ────────────────────────
    g_gru_trajs, g_lif_trajs, traj_labels = extract_routing_gates(
        model, dataloader, device, n_representative=n_representative,
    )

    # ── Create dual-panel figure ──────────────────────────────
    create_dual_panel_figure(
        trajectories_3d=trajectories_3d,
        all_labels=all_labels,
        pca_explained_var=pca.explained_variance_ratio_,
        g_gru_trajs=g_gru_trajs,
        g_lif_trajs=g_lif_trajs,
        traj_labels=traj_labels,
        output_path=output_path,
    )

    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
# 7.  CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="NSMoR Mechanism Analysis — Dual-Panel Publication Figure",
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
        "--output",
        type=str,
        default="results/mechanism_analysis.png",
        help="Output path for dual-panel figure.",
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
    parser.add_argument(
        "--n_representative",
        type=int,
        default=5,
        help="Number of representative trials for Panel B.",
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
        output_path=Path(args.output),
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
        n_representative=args.n_representative,
    )


if __name__ == "__main__":
    main()
