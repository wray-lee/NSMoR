"""
NSMoR In-Silico Lesion (Virtual Ablation) Experiment — Phase 7.

Runs a deterministic ablation experiment comparing the behavioral
(kinematic) output of the Intact model versus Lesioned models:
  - Condition 1 (Intact): Natural routing
  - Condition 2 (LIF-Lesioned): Forces all routing through GRU pathway
  - Condition 3 (GRU-Lesioned): Forces all routing through LIF pathway

Generates a Lancet/Cell-quality publication figure demonstrating
behavioral collapse when specific pathways are lesioned.

Output: ``results/ablation_kinematics.png`` at 300 DPI.

Usage
-----
CLI::

    python scripts/simulate_lesion.py --checkpoint runs/default/best_model.pth
    python scripts/simulate_lesion.py --checkpoint runs/default/best_model.pth --dataset data/processed/nsmor_dataset.pt --target_class 0
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

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
GROUND_TRUTH_COLOR: str = "#C92A2A"   # Lancet Crimson Red
PREDICTED_COLOR: str = "#495057"       # Strong Slate Gray

# ── Lesion condition names ──
CONDITION_NAMES: Dict[str, str] = {
    "intact": "Intact Model",
    "lif_lesioned": "LIF-Lesioned (g_lif=0)",
    "gru_lesioned": "GRU-Lesioned (g_gru=0)",
}

# ── Lesion gate overrides ──
LESION_OVERRIDES: Dict[str, Optional[Dict[str, float]]] = {
    "intact": None,
    "lif_lesioned": {"g_lif": 0.0, "g_gru": 1.0},
    "gru_lesioned": {"g_lif": 1.0, "g_gru": 0.0},
}

# ── Typography ─────────────────────────────────────────────────
FONT_FAMILY: str = "Arial"
FONT_SIZE_AXIS_TITLE: int = 12
FONT_SIZE_TICK: int = 10
FONT_SIZE_LEGEND: int = 9
FONT_SIZE_PANEL_LABEL: int = 14

# ── Figure properties ─────────────────────────────────────────
DPI: int = 300
FIG_WIDTH_INCHES: float = 10.0
FIG_HEIGHT_INCHES: float = 12.0
BACKGROUND_COLOR: str = "#FFFFFF"
AXIS_COLOR: str = "#212529"  # Solid dark charcoal

# ── Plot properties ───────────────────────────────────────────
LINE_WIDTH_GT: float = 2.5
LINE_WIDTH_PRED: float = 2.0
DASHED_LINESTYLE: str = "--"


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
) -> Tuple[torch.utils.data.DataLoader, np.ndarray, List[int]]:
    """
    Load the preprocessed dataset and create a DataLoader.

    Args:
        dataset_path: Path to ``nsmor_dataset.pt``.
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
    bio_dataset = NSMoRDataset(
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
# 3.  Ablation Experiment Runner
# ═══════════════════════════════════════════════════════════════

def run_ablation_condition(
    model: NSMoRCore,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    override_gates: Optional[Dict[str, float]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Run the model under a single lesion condition.

    Args:
        model: Trained NSMoRCore model.
        dataloader: DataLoader yielding (X, Y, lengths) tuples.
        device: Computation device.
        override_gates: Gate override dict (None for intact).

    Returns:
        ``(y_preds, y_trues, trial_labels)`` where:
        - ``y_preds``: List of arrays, each (T_i,)
        - ``y_trues``: List of arrays, each (T_i,)
        - ``trial_labels``: List of label values
    """
    y_preds: List[np.ndarray] = []
    y_trues: List[np.ndarray] = []
    trial_labels: List[int] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            X_batch, Y_batch, lengths = batch
            X_batch = X_batch.to(device)
            lengths = lengths.to(device)

            # Forward pass with override
            Y_pred = model(X_batch, lengths, override_gates=override_gates)

            B, T = Y_pred.shape

            for i in range(B):
                length_i = int(lengths[i].item())

                # Extract valid (unpadded) predictions and targets
                y_pred_i = Y_pred[i, :length_i].cpu().numpy()  # (T_i,)
                y_true_i = Y_batch[i, :length_i].cpu().numpy()  # (T_i,)

                # Shape assertions
                assert y_pred_i.shape == (length_i,), (
                    f"y_pred shape {y_pred_i.shape} != ({length_i},)"
                )
                assert y_true_i.shape == (length_i,), (
                    f"y_true shape {y_true_i.shape} != ({length_i},)"
                )

                y_preds.append(y_pred_i)
                y_trues.append(y_true_i)

                # Get label
                global_idx = batch_idx * B + i
                if global_idx < len(dataloader.dataset):
                    _, _, label_val = dataloader.dataset.sequences[global_idx]
                    trial_labels.append(int(label_val))
                else:
                    trial_labels.append(-1)  # Unknown

    return y_preds, y_trues, trial_labels


def run_full_ablation(
    model: NSMoRCore,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[int]]]:
    """
    Run the full ablation experiment (all three conditions).

    Args:
        model: Trained NSMoRCore model.
        dataloader: DataLoader yielding (X, Y, lengths) tuples.
        device: Computation device.

    Returns:
        Dictionary mapping condition name to
        ``(y_preds, y_trues, trial_labels)`` tuples.
    """
    results: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[int]]] = {}

    for condition_name, override in LESION_OVERRIDES.items():
        logger.info("Running condition: %s", CONDITION_NAMES[condition_name])

        if override is not None:
            logger.info("  Override gates: %s", override)
        else:
            logger.info("  Using natural routing (no override)")

        y_preds, y_trues, trial_labels = run_ablation_condition(
            model=model,
            dataloader=dataloader,
            device=device,
            override_gates=override,
        )

        results[condition_name] = (y_preds, y_trues, trial_labels)

        # Log summary statistics
        n_trials = len(y_preds)
        mean_pred_velocity = np.mean([np.mean(np.abs(yp)) for yp in y_preds])
        mean_true_velocity = np.mean([np.mean(np.abs(yt)) for yt in y_trues])
        logger.info(
            "  Trials: %d, Mean |predicted| velocity: %.3f cm/s, "
            "Mean |true| velocity: %.3f cm/s",
            n_trials, mean_pred_velocity, mean_true_velocity,
        )

    return results


# ═══════════════════════════════════════════════════════════════
# 4.  Class-Specific Trajectory Averaging
# ═══════════════════════════════════════════════════════════════

def average_trajectories_by_class(
    y_preds: List[np.ndarray],
    y_trues: List[np.ndarray],
    trial_labels: List[int],
    target_class: int,
    dt_ms: float = 10.0,
    max_time_ms: float = 5000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average velocity trajectories across trials of a specific class.

    Aligns trajectories relative to stimulus onset (t=0) and averages
    them.  Trajectories shorter than the analysis window are zero-padded.

    Args:
        y_preds: List of predicted velocity arrays.
        y_trues: List of ground truth velocity arrays.
        trial_labels: List of label values.
        target_class: Label value to filter by.
        dt_ms: Frame interval in milliseconds.
        max_time_ms: Maximum analysis window in ms.

    Returns:
        ``(time_ms, mean_pred, mean_true)`` arrays, all (n_frames,).
    """
    # Filter by target class
    class_indices = [i for i, l in enumerate(trial_labels) if l == target_class]

    if not class_indices:
        raise ValueError(f"No trials found for class {target_class}")

    logger.info(
        "Found %d trials for class %d (%s).",
        len(class_indices), target_class,
        Label(target_class).name if target_class in [e.value for e in Label] else "Unknown",
    )

    # Determine analysis window
    n_frames = int(max_time_ms / dt_ms)
    time_ms = np.arange(n_frames) * dt_ms - 2000.0  # Relative to stimulus onset (2s baseline)

    # Collect and align trajectories
    pred_matrix = np.zeros((len(class_indices), n_frames))
    true_matrix = np.zeros((len(class_indices), n_frames))

    for idx, trial_idx in enumerate(class_indices):
        y_pred = y_preds[trial_idx]
        y_true = y_trues[trial_idx]

        # Find stimulus onset frame (approximate: frame 200 for 2s baseline at 10ms)
        stim_frame = min(200, len(y_pred) // 4)  # Heuristic

        # Extract post-stimulus portion
        post_stim_pred = y_pred[stim_frame:]
        post_stim_true = y_true[stim_frame:]

        # Copy into matrix (zero-pad if shorter)
        n_copy = min(len(post_stim_pred), n_frames)
        pred_matrix[idx, :n_copy] = post_stim_pred[:n_copy]
        true_matrix[idx, :n_copy] = post_stim_true[:n_copy]

    # Average across trials
    mean_pred = np.mean(pred_matrix, axis=0)
    mean_true = np.mean(true_matrix, axis=0)
    sem_pred = np.std(pred_matrix, axis=0) / np.sqrt(len(class_indices))
    sem_true = np.std(true_matrix, axis=0) / np.sqrt(len(class_indices))

    return time_ms, mean_pred, mean_true


# ═══════════════════════════════════════════════════════════════
# 4b.  Scalar Metrics Extraction for Statistical Analysis
# ═══════════════════════════════════════════════════════════════

def extract_scalar_metrics(
    y_preds: List[np.ndarray],
    y_trues: List[np.ndarray],
    trial_labels: List[int],
    target_class: int,
    dt_ms: float = 10.0,
    stim_onset_frame: int = 200,
) -> Dict[str, float]:
    """
    Extract scalar metrics from velocity trajectories for a given class.

    Computes:
        - **Peak Velocity (V_max):** Maximum absolute velocity in the
          post-stimulus window.
        - **Latency to Peak (T_max):** Time (ms) relative to stimulus
          onset when V_max is reached.
        - **Mean MSE:** Mean squared error between predicted and true
          velocity across all frames.

    Args:
        y_preds: List of predicted velocity arrays, each (T_i,).
        y_trues: List of ground truth velocity arrays, each (T_i,).
        trial_labels: List of label values for each trial.
        target_class: Label value to filter by.
        dt_ms: Frame interval in milliseconds.
        stim_onset_frame: Frame index of stimulus onset (default 200
            for 2s baseline at 10ms/frame).

    Returns:
        Dictionary with keys:
        - ``"Peak_Velocity_cms"``: Max absolute velocity (cm/s).
        - ``"Latency_to_Peak_ms"``: Time of peak relative to stimulus (ms).
        - ``"Mean_MSE"``: Mean squared error across all frames.

    Raises:
        ValueError: If no trials found for target_class.
    """
    # ── Filter by target class ────────────────────────────────
    class_indices = [i for i, l in enumerate(trial_labels) if l == target_class]

    if not class_indices:
        raise ValueError(
            f"No trials found for class {target_class} "
            f"({Label(target_class).name if target_class in [e.value for e in Label] else 'Unknown'})"
        )

    logger.info(
        "Extracting metrics for %d trials of class %d (%s).",
        len(class_indices), target_class,
        Label(target_class).name if target_class in [e.value for e in Label] else "Unknown",
    )

    # ── Collect post-stimulus velocity arrays ─────────────────
    peak_velocities: List[float] = []
    latencies: List[float] = []
    mse_values: List[float] = []

    for trial_idx in class_indices:
        y_pred = y_preds[trial_idx]
        y_true = y_trues[trial_idx]

        # Ensure we have valid post-stimulus data
        if len(y_pred) <= stim_onset_frame or len(y_true) <= stim_onset_frame:
            logger.warning(
                "Trial %d too short (len=%d) for stim_onset_frame=%d, skipping.",
                trial_idx, min(len(y_pred), len(y_true)), stim_onset_frame,
            )
            continue

        # Extract post-stimulus portions
        post_pred = y_pred[stim_onset_frame:]
        post_true = y_true[stim_onset_frame:]

        n_post = min(len(post_pred), len(post_true))
        post_pred = post_pred[:n_post]
        post_true = post_true[:n_post]

        # ── Peak Velocity (V_max): maximum absolute velocity ──
        abs_velocity = np.abs(post_true)
        v_max = float(np.max(abs_velocity))

        # ── Latency to Peak (T_max): time of V_max ───────────
        peak_frame = int(np.argmax(abs_velocity))
        t_max = float(peak_frame * dt_ms)  # Convert frames to ms

        # ── Mean MSE ──────────────────────────────────────────
        mse = float(np.mean((post_pred - post_true) ** 2))

        peak_velocities.append(v_max)
        latencies.append(t_max)
        mse_values.append(mse)

    if not peak_velocities:
        raise ValueError(f"No valid post-stimulus data for class {target_class}")

    # ── Aggregate across trials ───────────────────────────────
    metrics = {
        "Peak_Velocity_cms": float(np.mean(peak_velocities)),
        "Latency_to_Peak_ms": float(np.mean(latencies)),
        "Mean_MSE": float(np.mean(mse_values)),
    }

    logger.info(
        "  Metrics: V_max=%.3f cm/s, T_max=%.1f ms, MSE=%.4f",
        metrics["Peak_Velocity_cms"],
        metrics["Latency_to_Peak_ms"],
        metrics["Mean_MSE"],
    )

    return metrics


def export_lesion_statistics_csv(
    results: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[int]]],
    output_path: Path,
    target_classes: List[int],
    dt_ms: float = 10.0,
    stim_onset_frame: int = 200,
) -> None:
    """
    Export lesion statistics to a CSV file for ANOVA testing.

    Generates a CSV with columns:
    ``Class, Condition, Peak_Velocity_cms, Latency_to_Peak_ms, Mean_MSE``

    Args:
        results: Dictionary from :func:`run_full_ablation` mapping
            condition names to ``(y_preds, y_trues, trial_labels)``.
        output_path: Path to save the CSV file.
        target_classes: List of label values to analyze.
        dt_ms: Frame interval in milliseconds.
        stim_onset_frame: Frame index of stimulus onset.

    Raises:
        ValueError: If no valid statistics could be computed.
    """
    logger.info("=" * 60)
    logger.info("Exporting lesion statistics to CSV...")
    logger.info("=" * 60)

    # ── Collect all rows ──────────────────────────────────────
    csv_rows: List[Dict[str, str]] = []

    for target_class in target_classes:
        # Get class name
        try:
            class_name = Label(target_class).name
        except ValueError:
            class_name = f"Class_{target_class}"

        for condition_name, (y_preds, y_trues, trial_labels) in results.items():
            try:
                metrics = extract_scalar_metrics(
                    y_preds=y_preds,
                    y_trues=y_trues,
                    trial_labels=trial_labels,
                    target_class=target_class,
                    dt_ms=dt_ms,
                    stim_onset_frame=stim_onset_frame,
                )

                csv_rows.append({
                    "Class": class_name,
                    "Condition": CONDITION_NAMES[condition_name],
                    "Peak_Velocity_cms": f"{metrics['Peak_Velocity_cms']:.4f}",
                    "Latency_to_Peak_ms": f"{metrics['Latency_to_Peak_ms']:.2f}",
                    "Mean_MSE": f"{metrics['Mean_MSE']:.6f}",
                })

            except ValueError as e:
                logger.warning(
                    "Skipping class %s, condition %s: %s",
                    class_name, condition_name, e,
                )
                continue

    if not csv_rows:
        raise ValueError("No valid statistics could be computed for any class/condition.")

    # ── Write CSV ─────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Class", "Condition", "Peak_Velocity_cms", "Latency_to_Peak_ms", "Mean_MSE"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info("Saved %d rows to %s", len(csv_rows), output_path)

    # ── Log summary table ─────────────────────────────────────
    logger.info("-" * 80)
    logger.info("%-15s %-25s %-15s %-15s %-12s",
                "Class", "Condition", "V_max (cm/s)", "T_max (ms)", "MSE")
    logger.info("-" * 80)
    for row in csv_rows:
        logger.info(
            "%-15s %-25s %-15s %-15s %-12s",
            row["Class"], row["Condition"],
            row["Peak_Velocity_cms"], row["Latency_to_Peak_ms"], row["Mean_MSE"],
        )
    logger.info("-" * 80)


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


def create_ablation_figure(
    results: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[int]]],
    target_class: int,
    output_path: Path,
    dt_ms: float = 10.0,
) -> None:
    """
    Create the Lancet/Cell ablation comparison figure.

    Layout: subplots(3, 1) — Intact, LIF-Lesioned, GRU-Lesioned

    Args:
        results: Dictionary from :func:`run_full_ablation`.
        target_class: Label value to plot.
        output_path: Path to save the figure.
        dt_ms: Frame interval in milliseconds.
    """
    setup_lancet_style()

    # Get class name
    try:
        class_name = Label(target_class).name.replace("_", " ").title()
    except ValueError:
        class_name = f"Class {target_class}"

    # ── Create figure with [3, 1] layout ──
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))

    # Panel labels
    panel_labels = ["A", "B", "C"]

    for idx, (condition_name, (y_preds, y_trues, trial_labels)) in enumerate(results.items()):
        ax = axes[idx]

        logger.info("Plotting condition: %s", CONDITION_NAMES[condition_name])

        # Average trajectories by class
        try:
            time_ms, mean_pred, mean_true = average_trajectories_by_class(
                y_preds=y_preds,
                y_trues=y_trues,
                trial_labels=trial_labels,
                target_class=target_class,
                dt_ms=dt_ms,
            )
        except ValueError as e:
            logger.warning("Skipping condition %s: %s", condition_name, e)
            ax.text(
                0.5, 0.5, f"No data for {class_name}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_SIZE_AXIS_TITLE, color=AXIS_COLOR,
            )
            continue

        # Plot Ground Truth (Lancet Crimson Red, solid)
        ax.plot(
            time_ms, mean_true,
            color=GROUND_TRUTH_COLOR,
            linewidth=LINE_WIDTH_GT,
            solid_capstyle="round",
            label="Ground Truth",
        )

        # Plot Predicted (Strong Slate Gray, dashed)
        ax.plot(
            time_ms, mean_pred,
            color=PREDICTED_COLOR,
            linewidth=LINE_WIDTH_PRED,
            linestyle=DASHED_LINESTYLE,
            solid_capstyle="round",
            label="Predicted",
        )

        # ── Stimulus onset vertical line ──
        ax.axvline(
            x=0.0,
            color=AXIS_COLOR,
            linewidth=1.0,
            linestyle=":",
            alpha=0.5,
        )

        # ── Axes styling ──
        ax.set_xlabel(
            "Time relative to stimulus onset (ms)",
            fontsize=FONT_SIZE_AXIS_TITLE,
            color=AXIS_COLOR,
        )
        ax.set_ylabel(
            "Velocity (cm/s)",
            fontsize=FONT_SIZE_AXIS_TITLE,
            color=AXIS_COLOR,
        )

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

        # Panel label and condition title
        ax.set_title(
            f"{panel_labels[idx]}  {CONDITION_NAMES[condition_name]}",
            fontsize=FONT_SIZE_PANEL_LABEL,
            fontweight="bold",
            color=AXIS_COLOR,
            loc="left",
        )

    # ── Suptitle ──
    fig.suptitle(
        f"In-Silico Lesion Analysis — {class_name} Trials",
        fontsize=14,
        fontweight="bold",
        color=AXIS_COLOR,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Save ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    logger.info("Saved ablation figure to %s (%d DPI)", output_path, DPI)

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 6.  Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════

def run_lesion_experiment(
    checkpoint_path: Path,
    dataset_path: Path,
    output_path: Path,
    stats_output_path: Optional[Path] = None,
    target_class: int = 0,
    target_classes: Optional[List[int]] = None,
    batch_size: int = 32,
    dt_ms: float = 10.0,
    stim_onset_frame: int = 200,
) -> None:
    """
    Run the full in-silico lesion experiment.

    Args:
        checkpoint_path: Path to the trained model checkpoint.
        dataset_path: Path to the preprocessed dataset.
        output_path: Path to save the ablation figure.
        stats_output_path: Path to save the statistics CSV. If None,
            defaults to ``results/lesion_statistics.csv``.
        target_class: Label value to plot in the figure (default 0 = STARTLE).
        target_classes: List of label values to include in statistics CSV.
            If None, defaults to all classes [0, 1, 2, 3].
        batch_size: Batch size for data loading.
        dt_ms: Frame interval in milliseconds.
        stim_onset_frame: Frame index of stimulus onset (default 200
            for 2s baseline at 10ms/frame).
    """
    logger.info("=" * 60)
    logger.info("NSMoR In-Silico Lesion Experiment (Phase 7)")
    logger.info("=" * 60)

    # ── Default paths ─────────────────────────────────────────
    if stats_output_path is None:
        stats_output_path = Path("results/lesion_statistics.csv")
    if target_classes is None:
        target_classes = [e.value for e in Label]  # All 4 classes

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────
    model = load_model_from_checkpoint(checkpoint_path, device)

    # ── Load dataset ──────────────────────────────────────────
    dataloader, labels, lengths_list = load_dataset(dataset_path, batch_size=batch_size)

    # ── Run ablation ──────────────────────────────────────────
    results = run_full_ablation(model, dataloader, device)

    # ── Log comparative statistics ────────────────────────────
    logger.info("-" * 60)
    logger.info("Comparative Statistics (for plotting class):")
    for condition_name, (y_preds, y_trues, trial_labels) in results.items():
        class_indices = [i for i, l in enumerate(trial_labels) if l == target_class]
        if not class_indices:
            continue

        # Compute MSE for this class
        mse_sum = 0.0
        n_total = 0
        for idx in class_indices:
            pred = y_preds[idx]
            true = y_trues[idx]
            n = min(len(pred), len(true))
            mse_sum += np.sum((pred[:n] - true[:n]) ** 2)
            n_total += n

        mse = mse_sum / max(n_total, 1)
        logger.info("  %s: MSE = %.4f", CONDITION_NAMES[condition_name], mse)

    # ── Export statistics CSV ──────────────────────────────────
    try:
        export_lesion_statistics_csv(
            results=results,
            output_path=stats_output_path,
            target_classes=target_classes,
            dt_ms=dt_ms,
            stim_onset_frame=stim_onset_frame,
        )
    except ValueError as e:
        logger.warning("Could not export statistics CSV: %s", e)

    # ── Create figure ─────────────────────────────────────────
    create_ablation_figure(
        results=results,
        target_class=target_class,
        output_path=output_path,
        dt_ms=dt_ms,
    )

    logger.info("=" * 60)
    logger.info("Lesion experiment complete!")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
# 7.  CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="NSMoR In-Silico Lesion (Virtual Ablation) Experiment",
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
        default="results/ablation_kinematics.png",
        help="Output path for ablation figure.",
    )
    parser.add_argument(
        "--stats_output",
        type=str,
        default="results/lesion_statistics.csv",
        help="Output path for lesion statistics CSV.",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=0,
        help="Label value to plot in figure (0=STARTLE, 1=WALK, 2=PRE_ACTIVE, 3=NO_RESPONSE).",
    )
    parser.add_argument(
        "--target_classes",
        type=int,
        nargs="+",
        default=None,
        help="Label values to include in statistics CSV (default: all classes).",
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    run_lesion_experiment(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        stats_output_path=Path(args.stats_output),
        target_class=args.target_class,
        target_classes=args.target_classes,
        batch_size=args.batch_size,
        dt_ms=args.dt_ms,
        stim_onset_frame=args.stim_onset_frame,
    )


if __name__ == "__main__":
    main()
