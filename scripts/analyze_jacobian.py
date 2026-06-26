"""
BioMoR Jacobian Eigenvalue Spectrum Analysis — Phase 8.

Computes the Jacobian of the GRU at different trial phases (epochs)
and plots the eigenvalues on the complex plane (unit circle) to prove
the continuous integration mechanics of the GRU pathway.

Target: Label.WALK trials (sustained locomotion response).

Epochs relative to TTC:
  1. Early (Baseline):     TTC - 1000ms
  2. Transient (Burst):    TTC
  3. Sustained (Late Walk): TTC + 1000ms

Hypothesis: During the "Sustained" epoch, eigenvalues should cluster
near the boundary of the unit circle (Real part ≈ 1), proving the GRU
operates as a continuous integrator / line attractor.

Output: ``results/jacobian_spectrum.png`` at 300 DPI.

Usage
-----
CLI::

    python scripts/analyze_jacobian.py --checkpoint runs/default/best_model.pth
    python scripts/analyze_jacobian.py --checkpoint runs/default/best_model.pth --dataset data/processed/biomor_dataset.pt
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

from biomor.analysis.dynamics import FixedPointAdapter
from biomor.biomor_dataloader import (
    BioMoRDataset,
    collate_variable_length,
)
from biomor.checkpoint import load_checkpoint
from biomor.config import DEFAULT_FEATURE, Label
from biomor.model_biomor_core import BioMoRCore

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

# ── High-contrast color mapping ──
EIGENVALUE_COLOR: str = "#1C7ED6"       # Cell Cobalt Blue
UNIT_CIRCLE_COLOR: str = "#495057"      # Strong Slate Gray
AXIS_COLOR: str = "#212529"             # Solid dark charcoal
BACKGROUND_COLOR: str = "#FFFFFF"       # Clean white

# ── Typography ─────────────────────────────────────────────────
FONT_FAMILY: str = "Arial"
FONT_SIZE_AXIS_TITLE: int = 12
FONT_SIZE_TICK: int = 10
FONT_SIZE_LEGEND: int = 9
FONT_SIZE_PANEL_LABEL: int = 14

# ── Figure properties ─────────────────────────────────────────
DPI: int = 300
FIG_WIDTH_INCHES: float = 15.0
FIG_HEIGHT_INCHES: float = 5.0

# ── Plot properties ───────────────────────────────────────────
EIGENVALUE_ALPHA: float = 0.6
EIGENVALUE_SIZE: float = 15.0
UNIT_CIRCLE_LINEWIDTH: float = 1.5
UNIT_CIRCLE_LINESTYLE: str = "--"

# ── Epoch definitions ─────────────────────────────────────────
# Time offsets relative to TTC (in ms)
EPOCH_DEFINITIONS: Dict[str, Dict[str, float]] = {
    "early": {
        "offset_ms": -1000.0,
        "label": "Early (Baseline)\nTTC − 1000 ms",
    },
    "transient": {
        "offset_ms": 0.0,
        "label": "Transient (Stimulus Burst)\nTTC",
    },
    "sustained": {
        "offset_ms": 1000.0,
        "label": "Sustained (Late Walk)\nTTC + 1000 ms",
    },
}


# ═══════════════════════════════════════════════════════════════
# 1.  Model Loading
# ═══════════════════════════════════════════════════════════════

def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> BioMoRCore:
    """
    Load a trained BioMoRCore model from a checkpoint.

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
    model = BioMoRCore(
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
) -> Tuple[torch.utils.data.DataLoader, np.ndarray, List[int]]:
    """
    Load the preprocessed dataset and create a DataLoader.

    Args:
        dataset_path: Path to ``biomor_dataset.pt``.
        batch_size: Batch size for the DataLoader.

    Returns:
        ``(dataloader, labels, lengths_list)`` tuple.

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
    lengths = dataset["lengths"]

    n_total = len(X_seqs)
    logger.info("Loaded %d sequences.", n_total)

    # Build sequence list
    sequences = [
        (X_seqs[i], Y_seqs[i], int(labels[i]))
        for i in range(n_total)
    ]

    # Create dataset and dataloader
    feature_config = dataset.get("feature_config", DEFAULT_FEATURE)
    bio_dataset = BioMoRDataset(
        sequences=sequences,
        mcmc_priors=mcmc_priors,
        feature_config=feature_config,
    )

    dataloader = torch.utils.data.DataLoader(
        bio_dataset,
        batch_size=batch_size,
        shuffle=False,  # Preserve ordering for label matching
        num_workers=0,
        collate_fn=collate_variable_length,
    )

    lengths_list = [int(l) for l in lengths]
    return dataloader, labels, lengths_list


# ═══════════════════════════════════════════════════════════════
# 3.  GRU State Extraction at Specific Epochs
# ═══════════════════════════════════════════════════════════════

def extract_gru_states_at_epochs(
    model: BioMoRCore,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    target_class: int = Label.WALK.value,
    dt_ms: float = 10.0,
    stim_onset_frame: int = 200,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract GRU hidden states at specific trial epochs for a target class.

    For each trial, extracts the hidden state at three time points
    relative to TTC (Time-To-Collision / stimulus onset).

    Args:
        model: Trained BioMoRCore model.
        dataloader: DataLoader yielding (X, Y, lengths) tuples.
        device: Computation device.
        target_class: Label value to filter by (default: WALK).
        dt_ms: Frame interval in milliseconds.
        stim_onset_frame: Frame index of stimulus onset.

    Returns:
        Dictionary mapping epoch name to ``(h_states, x_inputs)`` tuples:
        - ``h_states``: ``(N, H)`` tensor of hidden states.
        - ``x_inputs``: ``(N, H)`` tensor of corresponding inputs
          (sensory encoding output).

    Raises:
        ValueError: If no valid states found for any epoch.
    """
    logger.info("Extracting GRU states for class %d (%s) at target epochs...",
                target_class, Label(target_class).name)

    # Initialize storage for each epoch
    epoch_states: Dict[str, List[torch.Tensor]] = {
        name: [] for name in EPOCH_DEFINITIONS
    }
    epoch_inputs: Dict[str, List[torch.Tensor]] = {
        name: [] for name in EPOCH_DEFINITIONS
    }

    # ── Compute target frame indices for each epoch ───────────
    epoch_frames: Dict[str, int] = {}
    for epoch_name, epoch_def in EPOCH_DEFINITIONS.items():
        offset_ms = epoch_def["offset_ms"]
        # Convert offset to frame index relative to stim_onset
        frame_offset = int(offset_ms / dt_ms)
        target_frame = stim_onset_frame + frame_offset
        epoch_frames[epoch_name] = target_frame
        logger.info(
            "  Epoch '%s': offset=%.0fms -> frame %d",
            epoch_name, offset_ms, target_frame,
        )

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            X_batch, _Y_batch, lengths = batch
            X_batch = X_batch.to(device)
            lengths = lengths.to(device)

            B, T, _ = X_batch.shape

            # Forward pass with internals to get GRU hidden states
            _y_pred, internals = model(X_batch, lengths, return_internals=True)

            # gru_hidden: (B, T, H)
            gru_hidden = internals["gru_hidden"]
            H = gru_hidden.shape[2]

            # Also get sensory encoding for Jacobian inputs
            # We need the encoder output, not the raw input
            sensory_x = X_batch[:, :, :model.sensory_dim]  # (B, T, D)
            e_sensory = model.sensory_encoder(sensory_x)    # (B, T, H)

            for i in range(B):
                global_idx = batch_idx * B + i

                # Check if this trial is of the target class
                if global_idx < len(dataloader.dataset):
                    _, _, label_val = dataloader.dataset.sequences[global_idx]
                    if int(label_val) != target_class:
                        continue
                else:
                    continue

                length_i = int(lengths[i].item())

                # Extract states at each epoch
                for epoch_name, target_frame in epoch_frames.items():
                    # Check if the target frame is valid for this trial
                    if target_frame < 0 or target_frame >= length_i:
                        logger.debug(
                            "  Trial %d: epoch '%s' frame %d out of bounds (length=%d), skipping.",
                            global_idx, epoch_name, target_frame, length_i,
                        )
                        continue

                    # Extract hidden state and input at this frame
                    h_t = gru_hidden[i, target_frame, :]   # (H,)
                    x_t = e_sensory[i, target_frame, :]    # (H,)

                    # Shape assertions
                    assert h_t.shape == (H,), (
                        f"h_t shape {tuple(h_t.shape)} != (H={H},)"
                    )
                    assert x_t.shape == (H,), (
                        f"x_t shape {tuple(x_t.shape)} != (H={H},)"
                    )

                    epoch_states[epoch_name].append(h_t.cpu())
                    epoch_inputs[epoch_name].append(x_t.cpu())

    # ── Stack into tensors ────────────────────────────────────
    result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for epoch_name in EPOCH_DEFINITIONS:
        if not epoch_states[epoch_name]:
            logger.warning("No valid states found for epoch '%s'.", epoch_name)
            continue

        h_stack = torch.stack(epoch_states[epoch_name], dim=0)  # (N, H)
        x_stack = torch.stack(epoch_inputs[epoch_name], dim=0)  # (N, H)

        assert h_stack.shape == x_stack.shape, (
            f"Shape mismatch: h_stack {tuple(h_stack.shape)} != "
            f"x_stack {tuple(x_stack.shape)}"
        )

        result[epoch_name] = (h_stack, x_stack)
        logger.info(
            "  Epoch '%s': extracted %d states (shape=%s)",
            epoch_name, h_stack.shape[0], tuple(h_stack.shape),
        )

    if not result:
        raise ValueError("No valid states found for any epoch.")

    return result


# ═══════════════════════════════════════════════════════════════
# 4.  Jacobian Eigenvalue Computation
# ═══════════════════════════════════════════════════════════════

def compute_eigenvalues_at_epochs(
    adapter: FixedPointAdapter,
    epoch_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_states_per_epoch: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute Jacobian eigenvalues at each epoch.

    Args:
        adapter: FixedPointAdapter instance.
        epoch_data: Dictionary from :func:`extract_gru_states_at_epochs`.
        device: Computation device.
        max_states_per_epoch: Maximum number of states to process per
            epoch (for computational efficiency).

    Returns:
        Dictionary mapping epoch name to complex eigenvalue array (N, H).
    """
    logger.info("Computing Jacobian eigenvalues at each epoch...")

    eigenvalue_results: Dict[str, np.ndarray] = {}

    for epoch_name, (h_states, x_inputs) in epoch_data.items():
        N = h_states.shape[0]
        H = h_states.shape[1]

        logger.info(
            "  Epoch '%s': %d states (H=%d), processing up to %d...",
            epoch_name, N, H, max_states_per_epoch,
        )

        # Subsample if too many states
        if N > max_states_per_epoch:
            indices = torch.randperm(N)[:max_states_per_epoch]
            h_sub = h_states[indices]
            x_sub = x_inputs[indices]
            logger.info("    Subsampled to %d states.", max_states_per_epoch)
        else:
            h_sub = h_states
            x_sub = x_inputs

        N_sub = h_sub.shape[0]

        # ── Prepare for Jacobian computation ──────────────────
        # Move to device and enable gradients
        h_sub = h_sub.to(device).requires_grad_(True)
        x_sub = x_sub.to(device)

        # ── Compute Jacobians in batches ──────────────────────
        batch_size = 32  # Process in smaller batches for memory
        all_eigenvalues: List[np.ndarray] = []

        for start_idx in range(0, N_sub, batch_size):
            end_idx = min(start_idx + batch_size, N_sub)
            h_batch = h_sub[start_idx:end_idx]
            x_batch = x_sub[start_idx:end_idx]

            # Compute Jacobians for this batch
            J_batch = adapter.compute_jacobian_batch(h_batch, x_batch)  # (B, H, H)

            # ── Shape assertion ───────────────────────────────
            assert J_batch.shape == (h_batch.shape[0], H, H), (
                f"J_batch shape {tuple(J_batch.shape)} != "
                f"({h_batch.shape[0]}, {H}, {H})"
            )

            # ── Extract eigenvalues ───────────────────────────
            # torch.linalg.eigvals returns complex eigenvalues
            eigvals = torch.linalg.eigvals(J_batch)  # (B, H) complex

            # Shape assertion
            assert eigvals.shape == (h_batch.shape[0], H), (
                f"eigvals shape {tuple(eigvals.shape)} != ({h_batch.shape[0]}, {H})"
            )

            # Move to CPU and convert to numpy
            all_eigenvalues.append(eigvals.cpu().numpy())

        # Concatenate all eigenvalues for this epoch
        epoch_eigvals = np.concatenate(all_eigenvalues, axis=0)  # (N_total, H)

        # ── Shape and type assertions ─────────────────────────
        assert epoch_eigvals.shape[1] == H, (
            f"Eigenvalue dim {epoch_eigvals.shape[1]} != H={H}"
        )
        assert np.iscomplexobj(epoch_eigvals), (
            f"Eigenvalues should be complex, got dtype={epoch_eigvals.dtype}"
        )

        eigenvalue_results[epoch_name] = epoch_eigvals

        # ── Log summary statistics ────────────────────────────
        real_parts = np.real(epoch_eigvals.flatten())
        imag_parts = np.imag(epoch_eigvals.flatten())
        magnitudes = np.abs(epoch_eigvals.flatten())

        logger.info(
            "    Eigenvalue stats: |λ|_mean=%.4f, |λ|_max=%.4f, "
            "Re(λ)_mean=%.4f, Im(λ)_mean=%.4f",
            np.mean(magnitudes), np.max(magnitudes),
            np.mean(real_parts), np.mean(imag_parts),
        )

    return eigenvalue_results


# ═══════════════════════════════════════════════════════════════
# 5.  Lancet/Cell Publication Figure
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
        "axes.grid": False,
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


def plot_eigenvalue_spectrum(
    eigenvalue_results: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """
    Plot the Jacobian eigenvalue spectrum on the complex plane.

    Creates a subplots(1, 3) figure with one panel per epoch.
    Each panel shows eigenvalues scattered on the complex plane
    with a unit circle reference.

    Args:
        eigenvalue_results: Dictionary from :func:`compute_eigenvalues_at_epochs`.
        output_path: Path to save the figure.
    """
    setup_lancet_style()

    # ── Create figure with [1, 3] layout ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))

    # Panel labels
    panel_labels = ["A", "B", "C"]

    for idx, (epoch_name, epoch_def) in enumerate(EPOCH_DEFINITIONS.items()):
        ax = axes[idx]

        if epoch_name not in eigenvalue_results:
            logger.warning("No eigenvalues for epoch '%s', skipping.", epoch_name)
            ax.text(
                0.5, 0.5, "No data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_SIZE_AXIS_TITLE, color=AXIS_COLOR,
            )
            continue

        eigvals = eigenvalue_results[epoch_name]  # (N, H) complex

        # Flatten to 1D for plotting
        eigvals_flat = eigvals.flatten()
        real_parts = np.real(eigvals_flat)
        imag_parts = np.imag(eigvals_flat)

        # ── Draw unit circle ──────────────────────────────────
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(
            np.cos(theta), np.sin(theta),
            color=UNIT_CIRCLE_COLOR,
            linewidth=UNIT_CIRCLE_LINEWIDTH,
            linestyle=UNIT_CIRCLE_LINESTYLE,
            label="Unit circle",
        )

        # ── Scatter eigenvalues ───────────────────────────────
        ax.scatter(
            real_parts, imag_parts,
            color=EIGENVALUE_COLOR,
            s=EIGENVALUE_SIZE,
            alpha=EIGENVALUE_ALPHA,
            edgecolors="white",
            linewidths=0.3,
            label=r"Eigenvalues ($\lambda$)",
            zorder=3,
        )

        # ── Reference lines (real=1, imaginary=0) ─────────────
        ax.axvline(
            x=1.0,
            color=UNIT_CIRCLE_COLOR,
            linewidth=0.8,
            linestyle=":",
            alpha=0.5,
        )
        ax.axhline(
            y=0.0,
            color=UNIT_CIRCLE_COLOR,
            linewidth=0.8,
            linestyle=":",
            alpha=0.5,
        )

        # ── Axes styling ──────────────────────────────────────
        ax.set_xlabel(
            r"Re($\lambda$)",
            fontsize=FONT_SIZE_AXIS_TITLE,
            color=AXIS_COLOR,
        )
        ax.set_ylabel(
            r"Im($\lambda$)",
            fontsize=FONT_SIZE_AXIS_TITLE,
            color=AXIS_COLOR,
        )

        # Set equal aspect ratio for proper circle visualization
        ax.set_aspect("equal", adjustable="datalim")

        # Set axis limits (with some padding)
        max_mag = max(np.max(np.abs(real_parts)), np.max(np.abs(imag_parts)))
        limit = min(max_mag * 1.2, 2.0)  # Cap at 2.0 for readability
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

        # Tick formatting
        ax.tick_params(axis="both", colors=AXIS_COLOR, width=1.5)

        # Spine styling
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOR)
            spine.set_linewidth(1.5)

        # Grid: ultra-faint major grid lines
        ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)

        # Legend
        ax.legend(
            loc="upper left",
            fontsize=FONT_SIZE_LEGEND,
            frameon=True,
            facecolor=BACKGROUND_COLOR,
            edgecolor=AXIS_COLOR,
            framealpha=1.0,
        )

        # Panel label and epoch title
        ax.set_title(
            f"{panel_labels[idx]}  {epoch_def['label']}",
            fontsize=FONT_SIZE_PANEL_LABEL,
            fontweight="bold",
            color=AXIS_COLOR,
            loc="left",
        )

        # ── Add statistics annotation ─────────────────────────
        n_eigenvalues = len(eigvals_flat)
        mean_magnitude = np.mean(np.abs(eigvals_flat))
        pct_near_unity = np.mean(np.abs(np.abs(eigvals_flat) - 1.0) < 0.1) * 100

        stats_text = (
            f"n = {n_eigenvalues}\n"
            f"|λ|$_{{mean}}$ = {mean_magnitude:.3f}\n"
            f"% near |λ|=1: {pct_near_unity:.1f}%"
        )
        ax.text(
            0.97, 0.03, stats_text,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=FONT_SIZE_LEGEND,
            color=AXIS_COLOR,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=BACKGROUND_COLOR,
                edgecolor=UNIT_CIRCLE_COLOR,
                alpha=0.9,
            ),
        )

    # ── Suptitle ──────────────────────────────────────────────
    fig.suptitle(
        "Jacobian Eigenvalue Spectrum — GRU Pathway Dynamics",
        fontsize=14,
        fontweight="bold",
        color=AXIS_COLOR,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # ── Save ──────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    logger.info("Saved Jacobian spectrum figure to %s (%d DPI)", output_path, DPI)

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 6.  Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════

def run_jacobian_analysis(
    checkpoint_path: Path,
    dataset_path: Path,
    output_path: Path,
    target_class: int = Label.WALK.value,
    batch_size: int = 32,
    dt_ms: float = 10.0,
    stim_onset_frame: int = 200,
    max_states_per_epoch: int = 100,
) -> None:
    """
    Run the full Jacobian eigenvalue spectrum analysis.

    Args:
        checkpoint_path: Path to the trained model checkpoint.
        dataset_path: Path to the preprocessed dataset.
        output_path: Path to save the spectrum figure.
        target_class: Label value to analyze (default: WALK).
        batch_size: Batch size for data loading.
        dt_ms: Frame interval in milliseconds.
        stim_onset_frame: Frame index of stimulus onset.
        max_states_per_epoch: Maximum states to process per epoch.
    """
    logger.info("=" * 60)
    logger.info("BioMoR Jacobian Eigenvalue Spectrum Analysis (Phase 8)")
    logger.info("=" * 60)

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────
    model = load_model_from_checkpoint(checkpoint_path, device)

    # ── Load dataset ──────────────────────────────────────────
    dataloader, labels, lengths_list = load_dataset(dataset_path, batch_size=batch_size)

    # ── Initialize FixedPointAdapter ──────────────────────────
    adapter = FixedPointAdapter(model, device=device)

    # ── Extract GRU states at epochs ──────────────────────────
    epoch_data = extract_gru_states_at_epochs(
        model=model,
        dataloader=dataloader,
        device=device,
        target_class=target_class,
        dt_ms=dt_ms,
        stim_onset_frame=stim_onset_frame,
    )

    # ── Compute eigenvalues ───────────────────────────────────
    eigenvalue_results = compute_eigenvalues_at_epochs(
        adapter=adapter,
        epoch_data=epoch_data,
        device=device,
        max_states_per_epoch=max_states_per_epoch,
    )

    # ── Log hypothesis verification ───────────────────────────
    logger.info("-" * 60)
    logger.info("Hypothesis Verification:")
    for epoch_name, eigvals in eigenvalue_results.items():
        magnitudes = np.abs(eigvals.flatten())
        real_parts = np.real(eigvals.flatten())
        near_unity = np.mean(np.abs(magnitudes - 1.0) < 0.1) * 100
        near_real_one = np.mean(np.abs(real_parts - 1.0) < 0.1) * 100

        logger.info(
            "  %s: |λ|_mean=%.4f, %%near|λ|=1: %.1f%%, %%nearRe(λ)=1: %.1f%%",
            epoch_name, np.mean(magnitudes), near_unity, near_real_one,
        )

    # ── Create figure ─────────────────────────────────────────
    plot_eigenvalue_spectrum(
        eigenvalue_results=eigenvalue_results,
        output_path=output_path,
    )

    logger.info("=" * 60)
    logger.info("Jacobian analysis complete!")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
# 7.  CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="BioMoR Jacobian Eigenvalue Spectrum Analysis",
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
        default="data/processed/biomor_dataset.pt",
        help="Path to preprocessed dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/jacobian_spectrum.png",
        help="Output path for Jacobian spectrum figure.",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=1,
        help="Label value to analyze (0=STARTLE, 1=WALK, 2=PRE_ACTIVE, 3=NO_RESPONSE).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for data loading.",
    )
    parser.add_argument(
        "--dt_ms",
        type=float,
        default=10.0,
        help="Frame interval in milliseconds.",
    )
    parser.add_argument(
        "--stim_onset_frame",
        type=int,
        default=200,
        help="Frame index of stimulus onset (default 200 for 2s baseline at 10ms).",
    )
    parser.add_argument(
        "--max_states",
        type=int,
        default=100,
        help="Maximum states to process per epoch.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    run_jacobian_analysis(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        target_class=args.target_class,
        batch_size=args.batch_size,
        dt_ms=args.dt_ms,
        stim_onset_frame=args.stim_onset_frame,
        max_states_per_epoch=args.max_states,
    )


if __name__ == "__main__":
    main()
