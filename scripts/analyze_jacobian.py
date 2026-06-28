"""
NSMoR Jacobian Eigenvalue Spectrum Analysis — Phase 8.

Computes the Jacobian of the GRU at different trial phases (epochs)
and plots the eigenvalues on the complex plane (unit circle) to prove
the continuous integration mechanics of the GRU pathway.

Target: Label.PREWALK trials (sustained locomotion response).

Epochs relative to detected stimulus onset:
  1. Early (Baseline):     onset - 1000ms
  2. Transient (Burst):    onset
  3. Sustained (Late Walk): onset + 1000ms

Slow-point search: For each epoch, a ±5-frame window is searched to
find the frame that minimises kinetic energy ||h_{t+1} - h_t||₂.

Hypothesis: During the "Sustained" epoch, eigenvalues should cluster
near the boundary of the unit circle (Real part ≈ 1), proving the GRU
operates as a continuous integrator / line attractor.

Output: ``results/jacobian_spectrum.png`` at 300 DPI.

Usage
-----
CLI::

    python scripts/analyze_jacobian.py --checkpoint runs/default/best_model.pth
    python scripts/analyze_jacobian.py --checkpoint runs/default/best_model.pth --dataset data/processed/nsmor_dataset.pt
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

# ── High-contrast color mapping ──
EIGENVALUE_COLOR: str = "#1C7ED6"       # Cell Cobalt Blue
UNIT_CIRCLE_COLOR: str = "#495057"      # Strong Slate Gray
AXIS_COLOR: str = "#212529"             # Solid dark charcoal
BACKGROUND_COLOR: str = "#FFFFFF"       # Clean white
CMAP_HEATMAP: str = "inferno"           # Perceptually uniform heatmap

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
HEXBIN_GRIDSIZE: int = 40               # Resolution for hexbin density plot

# ── Epoch definitions ─────────────────────────────────────────
# Time offsets relative to stimulus onset (in ms)
EPOCH_DEFINITIONS: Dict[str, Dict[str, float]] = {
    "early": {
        "offset_ms": -1000.0,
        "label": "Early (Baseline)\nonset − 1000 ms",
    },
    "transient": {
        "offset_ms": 0.0,
        "label": "Transient (Stimulus Burst)\nonset",
    },
    "sustained": {
        "offset_ms": 1000.0,
        "label": "Sustained (Late Walk)\nonset + 1000 ms",
    },
}

# ── Slow-point search ────────────────────────────────────────
SLOW_POINT_RADIUS: int = 5   # ±5 frames around each epoch centre


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
) -> Tuple[torch.utils.data.DataLoader, np.ndarray, List[int], List[np.ndarray]]:
    """
    Load the preprocessed dataset and create a DataLoader.

    Also returns the raw X_seqs so that stimulus onset can be
    detected dynamically from the sensory channels.

    Args:
        dataset_path: Path to ``nsmor_dataset.pt``.
        batch_size: Batch size for the DataLoader.

    Returns:
        ``(dataloader, labels, lengths_list, X_seqs)`` tuple.

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

    lengths_list = [int(l) for l in lengths]
    return dataloader, labels, lengths_list, X_seqs


# ═══════════════════════════════════════════════════════════════
# 3.  Dynamic Stimulus Onset Detection
# ═══════════════════════════════════════════════════════════════

def detect_stimulus_onset_frames(
    X_seqs: List[np.ndarray],
    dt_ms: float = 10.0,
    threshold: float = 1e-6,
) -> List[int]:
    """
    Detect the stimulus onset frame for each trial from the data.

    Scans the sensory channels (visual angle feature 0, wind state
    feature 1) to find the first frame where *either* channel
    becomes non-zero.  This replaces the hardcoded
    ``stim_onset_frame = 200`` with a data-driven anchor.

    For looming trials the visual angle transitions from 0 to a
    positive value at stimulus onset.  For pure-wind trials the
    wind channel transitions from 0 to 1.  Both are captured.

    Args:
        X_seqs: List of arrays, each ``(T_i, 8)``.
        dt_ms: Frame interval in ms (for logging only).
        threshold: Absolute value below which a channel is
            considered zero.

    Returns:
        List of frame indices (one per sequence).  Falls back to
        200 when no onset is detected (e.g. fully padded sequences).
    """
    default_onset = 200  # Fallback for degenerate sequences
    onset_frames: List[int] = []

    for i, X in enumerate(X_seqs):
        T_i = X.shape[0]
        visual = np.abs(X[:, 0])   # v_vis(t)
        wind = np.abs(X[:, 1])     # wind(t)

        # First frame where either sensory channel is non-zero
        nonzero_mask = (visual > threshold) | (wind > threshold)
        nonzero_indices = np.where(nonzero_mask)[0]

        if len(nonzero_indices) > 0:
            onset_frames.append(int(nonzero_indices[0]))
        else:
            logger.warning(
                "Trial %d: no non-zero sensory channel detected "
                "(length=%d). Using default onset frame %d.",
                i, T_i, default_onset,
            )
            onset_frames.append(default_onset)

    # Log statistics
    onset_arr = np.array(onset_frames)
    logger.info(
        "Detected stimulus onset frames: mean=%.1f, std=%.1f, "
        "min=%d, max=%d (N=%d)",
        onset_arr.mean(), onset_arr.std(),
        int(onset_arr.min()), int(onset_arr.max()),
        len(onset_frames),
    )

    return onset_frames


# ═══════════════════════════════════════════════════════════════
# 4.  GRU State Extraction at Specific Epochs
# ═══════════════════════════════════════════════════════════════

def _find_slow_point(
    gru_hidden: torch.Tensor,
    window_centre: int,
    length_i: int,
    radius: int = SLOW_POINT_RADIUS,
) -> Tuple[int, torch.Tensor]:
    """
    Find the slow-point frame within a ±radius window.

    The slow point is the frame *t* that minimises the kinetic
    energy  ‖h_{t+1} − h_t‖₂  within the search window.

    Args:
        gru_hidden: ``(T, H)`` GRU hidden-state trajectory for one trial.
        window_centre: Centre frame index of the search window.
        length_i: True (unpadded) sequence length.
        radius: Search radius in frames.

    Returns:
        ``(slow_frame, h_slow)`` where *slow_frame* is the index
        and *h_slow* is the ``(H,)`` hidden state at that frame.
    """
    lo = max(0, window_centre - radius)
    hi = min(length_i - 2, window_centre + radius)  # need t+1 to exist

    if lo > hi:
        # Degenerate window — fall back to centre clamped to valid range
        frame = max(0, min(window_centre, length_i - 2))
        return frame, gru_hidden[frame]

    # Kinetic energy: ||h_{t+1} - h_t||_2 for t in [lo, hi]
    h_window = gru_hidden[lo:hi + 2]         # (window+1, H)
    diffs = h_window[1:] - h_window[:-1]     # (window, H)
    ke = diffs.norm(dim=1)                   # (window,)

    best_local = int(ke.argmin().item())
    slow_frame = lo + best_local
    return slow_frame, gru_hidden[slow_frame]


def extract_gru_states_at_epochs(
    model: NSMoRCore,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    onset_frames: List[int],
    target_class: int = Label.ESCAPE.value,
    dt_ms: float = 10.0,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract GRU hidden states at specific trial epochs for a target class.

    For each trial, computes epoch frames relative to the
    **dynamically detected** stimulus onset, then searches a
    ±``SLOW_POINT_RADIUS`` window for the slow point (minimum
    kinetic energy).

    The input passed to the Jacobian adapter is the **full sensory
    encoding** ``e_sensory_t`` (dim H), i.e. the exact vector the
    GRU cell receives at time *t*.  This captures the partial
    derivative ∂h_{t+1}/∂h_t holding the GRU input fixed.

    Args:
        model: Trained NSMoRCore model.
        dataloader: DataLoader yielding (X, Y, lengths) tuples.
        device: Computation device.
        onset_frames: Per-trial stimulus onset frame indices
            (from :func:`detect_stimulus_onset_frames`).
        target_class: Label value to filter by (default: WALK).
        dt_ms: Frame interval in milliseconds.

    Returns:
        Dictionary mapping epoch name to ``(h_states, x_inputs)``
        tuples:
        - ``h_states``: ``(N, H)`` tensor of hidden states at slow points.
        - ``x_inputs``: ``(N, H)`` tensor of sensory encoding inputs
          at those same slow points.

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

    model.eval()

    with torch.no_grad():
        global_idx = 0  # Tracks position across batches

        for batch_idx, batch in enumerate(dataloader):
            X_batch, _Y_batch, lengths = batch
            X_batch = X_batch.to(device).contiguous()
            lengths = lengths.to(device).contiguous()

            B, T, _ = X_batch.shape

            # Forward pass with internals to get GRU hidden states
            _y_pred, internals = model(X_batch, lengths, return_internals=True)

            # gru_hidden: (B, T, H)
            gru_hidden = internals["gru_hidden"]
            H = gru_hidden.shape[2]

            # ── Task 2: Exact input reconstruction ─────────────
            # The GRU cell receives e_sensory = sensory_encoder(X[:, :, :4]).
            # We encode the FULL sensory slice so that x_t reflects
            # the exact input the GRU saw at each frame.
            sensory_x = X_batch[:, :, :model.sensory_dim]  # (B, T, D_sensory)
            e_sensory = model.sensory_encoder(sensory_x)    # (B, T, H)

            for i in range(B):
                if global_idx >= len(dataloader.dataset):
                    break

                # Check if this trial is of the target class
                _, _, label_val = dataloader.dataset.sequences[global_idx]
                if int(label_val) != target_class:
                    global_idx += 1
                    continue

                length_i = int(lengths[i].item())

                # Detect onset from the ACTUAL batch data (after cropping)
                wind_channel = X_batch[i, :length_i, 1].cpu().numpy()
                onset_frame = int(np.argmax(wind_channel > 0.5)) if np.any(wind_channel > 0.5) else 0

                # ── Compute epoch centre frames ────────────────
                for epoch_name, epoch_def in EPOCH_DEFINITIONS.items():
                    offset_ms = epoch_def["offset_ms"]
                    frame_offset = int(offset_ms / dt_ms)
                    centre_frame = onset_frame + frame_offset

                    # Bounds check (need at least frame+1 for KE)
                    if centre_frame < 0 or centre_frame >= length_i - 1:
                        logger.debug(
                            "  Trial %d: epoch '%s' centre frame %d "
                            "out of bounds (length=%d), skipping.",
                            global_idx, epoch_name, centre_frame, length_i,
                        )
                        continue

                    # ── Task 3: Slow-point search ──────────────
                    slow_frame, h_slow = _find_slow_point(
                        gru_hidden[i], centre_frame, length_i,
                    )
                    x_slow = e_sensory[i, slow_frame, :]    # (H,)

                    # Shape assertions
                    assert h_slow.shape == (H,), (
                        f"h_slow shape {tuple(h_slow.shape)} != (H={H},)"
                    )
                    assert x_slow.shape == (H,), (
                        f"x_slow shape {tuple(x_slow.shape)} != (H={H},)"
                    )

                    epoch_states[epoch_name].append(h_slow.cpu())
                    epoch_inputs[epoch_name].append(x_slow.cpu())

                    logger.debug(
                        "  Trial %d epoch '%s': centre=%d, slow=%d, "
                        "Δframe=%d",
                        global_idx, epoch_name, centre_frame,
                        slow_frame, slow_frame - centre_frame,
                    )

                global_idx += 1

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
            "  Epoch '%s': extracted %d slow-point states (shape=%s)",
            epoch_name, h_stack.shape[0], tuple(h_stack.shape),
        )

    if not result:
        raise ValueError("No valid states found for any epoch.")

    return result


# ═══════════════════════════════════════════════════════════════
# 5.  Jacobian Eigenvalue Computation
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
# 6.  Lancet/Cell Publication Figure
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

    Creates a ``subplots(1, 3)`` figure with one panel per epoch.
    Each panel uses a **hexbin density heatmap** to handle massive
    point overlap, with the unit circle reference overlaid.

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
            zorder=4,
        )

        # ── Task 4: Density-based spectrum visualization ──────
        # Use hexbin for density heatmap when many points,
        # fall back to scatter for small datasets.
        n_points = len(eigvals_flat)

        if n_points > 200:
            # Hexbin density heatmap
            hb = ax.hexbin(
                real_parts, imag_parts,
                gridsize=HEXBIN_GRIDSIZE,
                cmap=CMAP_HEATMAP,
                mincnt=1,
                linewidths=0.2,
                edgecolors="face",
                alpha=0.85,
                zorder=3,
            )
            # Colour bar
            cb = fig.colorbar(hb, ax=ax, shrink=0.78, pad=0.02)
            cb.set_label("Count", fontsize=FONT_SIZE_LEGEND, color=AXIS_COLOR)
            cb.ax.tick_params(labelsize=FONT_SIZE_LEGEND - 1, colors=AXIS_COLOR)
            for spine in cb.ax.spines.values():
                spine.set_color(AXIS_COLOR)

            # Legend entry for hexbin (manual proxy)
            from matplotlib.patches import Patch
            hex_proxy = Patch(facecolor=plt.get_cmap(CMAP_HEATMAP)(0.6), label="Density")
            unit_proxy = plt.Line2D([0], [0], color=UNIT_CIRCLE_COLOR,
                                    linewidth=UNIT_CIRCLE_LINEWIDTH,
                                    linestyle=UNIT_CIRCLE_LINESTYLE,
                                    label="Unit circle")
            ax.legend(
                handles=[hex_proxy, unit_proxy],
                loc="upper left",
                fontsize=FONT_SIZE_LEGEND,
                frameon=True,
                facecolor=BACKGROUND_COLOR,
                edgecolor=AXIS_COLOR,
                framealpha=1.0,
            )
        else:
            # Scatter for small datasets
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
            ax.legend(
                loc="upper left",
                fontsize=FONT_SIZE_LEGEND,
                frameon=True,
                facecolor=BACKGROUND_COLOR,
                edgecolor=AXIS_COLOR,
                framealpha=1.0,
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
# 7.  Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════

def run_jacobian_analysis(
    checkpoint_path: Path,
    dataset_path: Path,
    output_path: Path,
    target_class: int = Label.ESCAPE.value,
    batch_size: int = 32,
    max_seq_len: Optional[int] = 1000,
    dt_ms: float = 10.0,
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
        max_states_per_epoch: Maximum states to process per epoch.
    """
    logger.info("=" * 60)
    logger.info("NSMoR Jacobian Eigenvalue Spectrum Analysis (Phase 8)")
    logger.info("=" * 60)

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────
    model = load_model_from_checkpoint(checkpoint_path, device)

    # ── Load dataset (returns raw X_seqs for onset detection) ─
    dataloader, labels, lengths_list, X_seqs = load_dataset(
        dataset_path, batch_size=batch_size, max_seq_len=max_seq_len,
    )

    # ── Task 1: Dynamic stimulus onset detection ──────────────
    onset_frames = detect_stimulus_onset_frames(X_seqs, dt_ms=dt_ms)

    # ── Initialize FixedPointAdapter ──────────────────────────
    adapter = FixedPointAdapter(model, device=device)

    # ── Extract GRU states at epochs (slow-point search) ──────
    epoch_data = extract_gru_states_at_epochs(
        model=model,
        dataloader=dataloader,
        device=device,
        onset_frames=onset_frames,
        target_class=target_class,
        dt_ms=dt_ms,
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
# 8.  CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="NSMoR Jacobian Eigenvalue Spectrum Analysis",
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
        default="results/jacobian_spectrum.png",
        help="Output path for Jacobian spectrum figure.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (alternative to --output; "
             "saves as <output_dir>/jacobian_spectrum.png).",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=0,
        help="Label value to analyze (0=ESCAPE, 1=PREWALK, 2=PRE_ACTIVE, 3=NO_RESPONSE).",
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
        "--dt_ms",
        type=float,
        default=10.0,
        help="Frame interval in milliseconds.",
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

    # Resolve output path: --output_dir takes precedence when provided
    if args.output_dir is not None:
        output_path = Path(args.output_dir) / "jacobian_spectrum.png"
    else:
        output_path = Path(args.output)

    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    run_jacobian_analysis(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        output_path=output_path,
        target_class=args.target_class,
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
        dt_ms=args.dt_ms,
        max_states_per_epoch=args.max_states,
    )


if __name__ == "__main__":
    main()
