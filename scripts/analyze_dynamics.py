"""
NSMoR Mechanism Analysis -- Multi-Panel Publication Figure.

Generates a Cell/Science-quality multi-panel figure for publication:
  Vivid, high-contrast colors — rejects pale/desaturated palettes.
  - Panel A: 3D Phase-Space Manifold (PCA of combined LIF+GRU states)
  - Panel B: Per-Class Routing Gate Dynamics (mean +/- SEM by label)
  - Panel C: LIF Spike Rate Dynamics (mean +/- SEM by label)
  - Panel D: Pathway Dominance (mean g_gru per behavioral class)

Output: ``results/mechanism_analysis.png`` at 300 DPI.

Usage
-----
CLI::

    python scripts/analyze_dynamics.py --checkpoint runs/default/best_model.pth
    python scripts/analyze_dynamics.py --checkpoint runs/default/best_model.pth --dataset data/processed/nsmor_dataset.pt
    python scripts/analyze_dynamics.py --checkpoint runs/default/best_model.pth --layout 1x2
"""

from __future__ import annotations

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

# Direct CLI execution must use this checkout, including its prior/lineage helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D projection)
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip

from nsmor.analysis.uq import cohens_d, log_pca_variance
from nsmor.dataloader_factory import create_optimized_dataloader
from nsmor.nsmor_dataloader import NSMoRDataset
from nsmor.config import DEFAULT_FEATURE, Label
from nsmor.model_nsmor_core import NSMoRCore
from nsmor.analysis.analysis_priors import (
    describe_analysis_population, load_analysis_priors, population_for_output,
)
from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint
from nsmor.analysis.prediction_units import (
    load_model_from_checkpoint as _shared_load_model, resolve_dt_ms,
)
from nsmor.model_utils import validate_dataset_provenance

# -- Logging ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
# Lancet / Cell Publication Style Constants
# =========================================================================

# -- Cell-style vivid high-contrast categorical color mapping --
# Reject pale/desaturated palettes. All colors are bright, saturated,
# and clearly distinguishable — suitable for Cell/Science publication.
LANCET_COLORS: Dict[int, str] = {
    Label.ESCAPE.value: "#E64B35",      # Vivid Red — urgent escape response
    Label.PREWALK.value: "#4DBBD5",     # Bright Blue — locomotion
    Label.PRE_ACTIVE.value: "#00A087",  # Vivid Green — baseline activity
    Label.NO_RESPONSE.value: "#3C5488", # Bright Purple — non-responsive
}

# Routing gate curve colors
GATE_GRU_COLOR: str = "#4DBBD5"   # Bright Blue for g_gru(t)
GATE_LIF_COLOR: str = "#E64B35"   # Vivid Red for g_lif(t)

# Label display names
LABEL_NAMES: Dict[int, str] = {
    Label.ESCAPE.value: "Escape",
    Label.PREWALK.value: "Prewalk",
    Label.PRE_ACTIVE.value: "Pre-Active",
    Label.NO_RESPONSE.value: "No Response",
}

# Marker styles per label
LABEL_MARKERS: Dict[int, str] = {
    Label.ESCAPE.value: "^",    # Triangle up
    Label.PREWALK.value: "o",   # Circle
    Label.PRE_ACTIVE.value: "s", # Square
    Label.NO_RESPONSE.value: "D", # Diamond
}

# -- Typography ------------------------------------------------------------
FONT_FAMILY: str = "Arial"
FONT_SIZE_AXIS_TITLE: int = 12
FONT_SIZE_TICK: int = 10
FONT_SIZE_LEGEND: int = 9
FONT_SIZE_PANEL_LABEL: int = 14

# -- Figure properties -----------------------------------------------------
DPI: int = 300
BACKGROUND_COLOR: str = "#FFFFFF"
AXIS_COLOR: str = "#343A40"  # Softer dark gray (Cell-style, not harsh black)

# -- Plot properties -------------------------------------------------------
LINE_WIDTH: float = 2.5
SCATTER_ALPHA: float = 0.80
SCATTER_SIZE: float = 25.0
TRAJECTORY_ALPHA: float = 0.45
TRAJECTORY_LINEWIDTH: float = 0.8
SEM_ALPHA: float = 0.20  # Alpha for SEM fill region


# =========================================================================
# 1.  Model Loading
# =========================================================================

def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> NSMoRCore:
    """Load trained NSMoRCore from checkpoint."""
    return _shared_load_model(checkpoint_path, device)


# =========================================================================
# 2.  Dataset Loading
# =========================================================================

def load_dataset(
    dataset_path: Path,
    batch_size: int = 32,
    max_seq_len: Optional[int] = 2400,
    pre_anchor_frames: int = 1200,
    nested_prior_artifact: Optional[Path] = None,
    qc_sealed_nested_prior_sha256: Optional[str] = None,
    checkpoint_model: Optional[NSMoRCore] = None,
    trusted_historical_checkpoint_sha256: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
) -> Tuple[torch.utils.data.DataLoader, np.ndarray]:
    """
    Load the preprocessed dataset and create a DataLoader.

    Returns:
        ``(dataloader, labels)`` tuple.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info("Loading dataset from %s", dataset_path)
    dataset, loaded_source_fingerprint = load_dataset_with_fingerprint(
        dataset_path, restore_provenance=False,
        expected_dt_ms=(
            resolve_dt_ms(checkpoint_model) if checkpoint_model is not None else None
        ),
    )

    # Round-2 CRITICAL-A: refuse pre-2.0 datasets (leaked priors, np.max labels)
    validate_dataset_provenance(dataset, Path(dataset_path))

    X_seqs = dataset["X_seqs"]
    Y_seqs = dataset["Y_seqs"]
    mcmc_priors, val_indices = load_analysis_priors(
        dataset, dataset_path, nested_prior_artifact, checkpoint_model, qc_sealed_nested_prior_sha256,
        loaded_source_fingerprint=loaded_source_fingerprint,
        trusted_historical_checkpoint_sha256=trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
    )
    labels = dataset["labels"]

    n_total = len(X_seqs)
    logger.info("Loaded %d sequences.", n_total)

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

    feature_config = dataset.get("feature_config", DEFAULT_FEATURE)
    bio_dataset = NSMoRDataset(
        sequences=sequences,
        mcmc_priors=mcmc_priors,
        feature_config=feature_config,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        anchor_frames=anchor_frames,
    )

    bio_dataset.analysis_population = describe_analysis_population(
        n_total, val_indices, bio_dataset.source_indices,
        getattr(checkpoint_model, "analysis_validation_scope", None),
    )
    dataloader = create_optimized_dataloader(
        bio_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Whole corpus is already resident; workers only replicate it
    )

    return dataloader, labels


# =========================================================================
# 3.  Full Dynamics Extraction (Single Model Pass)
# =========================================================================

class DynamicsBundle:
    """
    Container for all extracted dynamics data from a single model pass.

    Attributes:
        gru_trajectories: List of GRU hidden state arrays, each (T_i, H).
        lif_spike_trajs: List of LIF spike arrays, each (T_i, H).
        lif_potential_trajs: List of LIF membrane potential arrays, each (T_i, H).
        lif_rate_trajs: List of mean spike rate arrays, each (T_i,).
        g_gru_trajs: List of g_gru gate arrays, each (T_i,).
        g_lif_trajs: List of g_lif gate arrays, each (T_i,).
        labels: Per-trajectory integer labels.
    """

    def __init__(self) -> None:
        self.gru_trajectories: List[torch.Tensor] = []
        self.lif_spike_trajs: List[torch.Tensor] = []
        self.lif_potential_trajs: List[torch.Tensor] = []
        self.lif_rate_trajs: List[np.ndarray] = []
        self.g_gru_trajs: List[np.ndarray] = []
        self.g_lif_trajs: List[np.ndarray] = []
        self.labels: List[int] = []

    @property
    def n_trajectories(self) -> int:
        return len(self.gru_trajectories)

    def log_summary(self) -> None:
        """Log extraction summary with per-label counts and LIF potential status."""
        label_counts: Dict[int, int] = {}
        for lbl in self.labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        total_states = sum(t.shape[0] for t in self.gru_trajectories)
        has_lif_potentials = len(self.lif_potential_trajs) > 0
        logger.info(
            "Extracted %d trajectories, total_states=%d, lif_potentials=%s",
            self.n_trajectories, total_states, has_lif_potentials,
        )
        for lbl_val in sorted(label_counts.keys()):
            name = LABEL_NAMES.get(lbl_val, f"Unknown({lbl_val})")
            logger.info("  %s (label=%d): n=%d", name, lbl_val, label_counts[lbl_val])


def extract_full_dynamics(
    model: NSMoRCore,
    dataloader: torch.utils.data.DataLoader,
    labels: np.ndarray,
    device: torch.device,
) -> DynamicsBundle:
    """
    Extract GRU states, LIF spikes, and routing gates in a single model pass.

    This replaces the two separate extraction functions (extract_and_reduce
    and extract_routing_gates) with a unified pass that collects all
    dynamics data needed for the multi-panel figure.

    Args:
        model: Trained NSMoRCore model.
        dataloader: DataLoader yielding (X, Y, lengths) tuples.
        labels: Ground truth labels for each sequence.
        device: Computation device.

    Returns:
        DynamicsBundle with all extracted trajectories.
    """
    logger.info("Extracting full dynamics (single model pass)...")
    model.eval()

    bundle = DynamicsBundle()
    global_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            X_batch, Y_batch, lengths = batch
            X_batch = X_batch.to(device).contiguous()
            lengths = lengths.to(device).contiguous()

            B, T, _ = X_batch.shape

            # Forward pass with all internals
            _, internals = model(X_batch, lengths, return_internals=True)

            # Extract tensors
            gru_hidden = internals["gru_hidden"]       # (B, T, H)
            lif_spikes = internals["lif_spikes"]       # (B, T, H)
            routing_gates = internals["routing_gates"] # (B, T, 2)

            # Extract LIF membrane potentials
            lif_potentials = internals["lif_potentials"]  # (B, T, H)

            for i in range(B):
                length_i = int(lengths[i].item())

                # GRU hidden states
                traj_gru = gru_hidden[i, :length_i, :].cpu()
                bundle.gru_trajectories.append(traj_gru)

                # LIF spike tensor
                traj_lif = lif_spikes[i, :length_i, :].cpu()
                bundle.lif_spike_trajs.append(traj_lif)

                # LIF membrane potentials (full H-dimensional state)
                traj_lif_v = lif_potentials[i, :length_i, :].cpu()
                bundle.lif_potential_trajs.append(traj_lif_v)

                # LIF mean spike rate per timestep (average across hidden dim)
                lif_rate = lif_spikes[i, :length_i, :].mean(dim=-1).cpu().numpy()
                bundle.lif_rate_trajs.append(lif_rate)

                # Routing gates
                g_gru_i = routing_gates[i, :length_i, 1].cpu().numpy()
                g_lif_i = routing_gates[i, :length_i, 0].cpu().numpy()
                bundle.g_gru_trajs.append(g_gru_i)
                bundle.g_lif_trajs.append(g_lif_i)

                # Label
                if global_idx < len(labels):
                    bundle.labels.append(int(labels[global_idx]))
                else:
                    logger.warning(
                        "global_idx=%d >= len(labels)=%d, defaulting to label 0",
                        global_idx, len(labels),
                    )
                    bundle.labels.append(0)

                global_idx += 1

    # Consistency check: all trajectory lists must have the same length
    n_gru = len(bundle.gru_trajectories)
    n_lif_v = len(bundle.lif_potential_trajs)
    if n_lif_v > 0 and n_lif_v != n_gru:
        logger.warning(
            "Trajectory count mismatch: gru=%d, lif_potentials=%d. "
            "PCA combined mode may fail.",
            n_gru, n_lif_v,
        )

    bundle.log_summary()
    return bundle


# =========================================================================
# 4.  PCA Dimensionality Reduction
# =========================================================================

def _iter_feature_blocks(
    bundle: DynamicsBundle,
    use_combined: bool,
) -> Iterator[np.ndarray]:
    """Yield per-trajectory feature matrices ``(T_i, D)``.

    Feature order is GRU hidden states first, then full LIF membrane potentials
    (matching the historical ``np.concatenate([gru, lif], axis=1)``), with the
    LIF-rate fallback and GRU-only variants preserved.  Trials are yielded one
    at a time so the full corpus is never materialised.

    Args:
        bundle: DynamicsBundle from extract_full_dynamics.
        use_combined: If True, use LIF potentials (or rate fallback) + GRU.

    Yields:
        ``(T_i, D)`` arrays; ``D`` is 2H, H+1, or H depending on mode.

    Raises:
        ValueError: If the LIF potential/rate count does not match the number
            of GRU trajectories.
    """
    n_traj = len(bundle.gru_trajectories)
    lif_pot = bundle.lif_potential_trajs if use_combined else []
    lif_rate = bundle.lif_rate_trajs if use_combined else []
    if lif_pot and len(lif_pot) != n_traj:
        raise ValueError(
            f"LIF potential count {len(lif_pot)} != GRU trajectory count {n_traj}"
        )
    if lif_rate and len(lif_rate) != n_traj:
        raise ValueError(
            f"LIF rate count {len(lif_rate)} != GRU trajectory count {n_traj}"
        )
    for i, traj_gru in enumerate(bundle.gru_trajectories):
        gru_np = traj_gru.numpy()  # (T_i, H)
        if lif_pot:
            yield np.concatenate([gru_np, lif_pot[i].numpy()], axis=1)  # (T_i, 2H)
        elif lif_rate:
            lif_col = np.asarray(lif_rate[i]).reshape(-1, 1)  # (T_i, 1)
            yield np.concatenate([gru_np, lif_col], axis=1)  # (T_i, H+1)
        else:
            yield gru_np


def _fit_centered_pca(
    blocks: Iterator[np.ndarray],
    n_components: int,
) -> PCA:
    """Exact whole-corpus centered PCA fit with O(D^2) memory.

    Streams per-trajectory ``(T_i, D)`` blocks and merges their centered
    moments (Chan-style pairwise update), so the ``(n_states, D)`` stacked
    corpus is never built.  Every frame keeps equal weight and the mean is the
    exact population mean.  Merging *centered* scatter (not a raw ``X^T X``
    Gram) avoids catastrophic cancellation when features have a large offset
    and small variance.

    Args:
        blocks: Iterator of per-trajectory ``(T_i, D)`` arrays.
        n_components: Number of principal components to retain.

    Returns:
        Fitted ``PCA`` (``svd_solver="covariance_eigh"``) exposing the standard
        ``mean_``/``components_``/``explained_variance[_ratio]_``/
        ``singular_values_``/``noise_variance_``/``n_samples_`` attributes so
        ``pca.transform`` behaves exactly as before.

    Raises:
        ValueError: On empty, singleton, non-finite, or inconsistent-dimension
            input, or ``n_components`` outside ``[1, min(n_samples, D)]``.
    """
    n_samples = 0
    mean: Optional[np.ndarray] = None
    scatter: Optional[np.ndarray] = None
    n_features: Optional[int] = None

    for block in blocks:
        if block.ndim != 2:
            raise ValueError(f"Feature block must be 2D; got shape {block.shape}")
        if block.shape[0] == 0:
            continue
        if n_features is None:
            n_features = block.shape[1]
        elif block.shape[1] != n_features:
            raise ValueError(
                "Inconsistent feature dimension across trajectories: "
                f"expected {n_features}, got {block.shape[1]}"
            )
        if not np.isfinite(block).all():
            raise ValueError(
                "Non-finite value in PCA feature block; refusing to fit a "
                "centered covariance on non-finite states."
            )
        block = block.astype(np.float64, copy=False)
        block_n = block.shape[0]
        block_mean = block.mean(axis=0)
        centered = block - block_mean
        block_scatter = centered.T @ centered
        if n_samples == 0:
            mean, scatter = block_mean, block_scatter
        else:
            total = n_samples + block_n
            delta = block_mean - mean
            scatter = scatter + block_scatter + np.outer(delta, delta) * (
                n_samples * block_n / total
            )
            mean = mean + delta * (block_n / total)
        n_samples += block_n

    if n_samples < 2 or n_features is None or mean is None or scatter is None:
        raise ValueError(
            "No states available for PCA fit." if n_samples == 0
            else f"PCA needs at least 2 states; got {n_samples}."
        )
    assert mean.shape == (n_features,)
    assert scatter.shape == (n_features, n_features)

    if isinstance(n_components, bool) or not float(n_components).is_integer():
        raise ValueError(f"n_components must be an integer; got {n_components!r}")
    n_components = int(n_components)
    if n_components < 1 or n_components > min(n_samples, n_features):
        raise ValueError(
            f"n_components={n_components} must be in [1, "
            f"{min(n_samples, n_features)}] with svd_solver='covariance_eigh'"
        )

    covariance = scatter / (n_samples - 1)
    eigenvals, eigenvecs = np.linalg.eigh(covariance)
    # eigh returns ascending eigenvalues: reorder to descending (PC1 first) and
    # clip fp-rounding of tiny negatives (covariance is PSD by construction).
    eigenvals = np.clip(eigenvals[::-1], 0.0, None)
    components = eigenvecs[:, ::-1].T
    # Deterministic sign convention identical to sklearn (svd_flip on Vt rows).
    _, components = svd_flip(None, components, u_based_decision=False)

    singular_values = np.sqrt(eigenvals * (n_samples - 1))
    total_var = float(eigenvals.sum())
    explained_variance_ratio = (
        eigenvals / total_var if total_var > 0.0 else np.zeros_like(eigenvals)
    )
    noise_variance = (
        float(eigenvals[n_components:].mean())
        if n_components < min(n_features, n_samples)
        else 0.0
    )

    pca = PCA(n_components=n_components, svd_solver="covariance_eigh")
    pca.mean_ = mean
    pca.components_ = np.ascontiguousarray(components[:n_components, :])
    pca.explained_variance_ = np.ascontiguousarray(eigenvals[:n_components])
    pca.explained_variance_ratio_ = np.ascontiguousarray(
        explained_variance_ratio[:n_components]
    )
    pca.singular_values_ = np.ascontiguousarray(singular_values[:n_components])
    pca.noise_variance_ = noise_variance
    pca.n_samples_ = n_samples
    pca.n_components_ = n_components
    pca.n_features_in_ = n_features
    return pca


def compute_pca_manifold(
    bundle: DynamicsBundle,
    n_components: int = 3,
    use_combined: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray, PCA]:
    """
    Fit PCA on extracted trajectories and transform to 3D.

    When ``use_combined=True``, concatenates LIF membrane potentials
    (H dims) with GRU hidden states (H dims) before PCA, giving the
    LIF pathway equal representation (2H total features).  This is a
    critical fix: the previous version appended only a 1-D mean spike
    rate scalar to the 64-D GRU state, effectively drowning the LIF
    signal in PCA.

    The fit streams the corpus in per-trajectory blocks and accumulates only
    the ``(D, D)`` centered scatter, so the historical ``(n_states, D)``
    stacked array (which OOM'd a 8 GiB cgroup at n_states≈5.5M) is never
    materialised.  All frames are retained; the centered covariance spectrum
    is exact.

    Args:
        bundle: DynamicsBundle from extract_full_dynamics.
        n_components: Number of PCA components (default 3).
        use_combined: If True, use LIF potentials + GRU states.

    Returns:
        ``(trajectories_3d, all_labels, pca)`` tuple, with ``all_labels`` a
        per-state ``int64`` array in trial order (one entry per state).
    """
    logger.info("Fitting PCA with %d components (combined=%s)...", n_components, use_combined)

    lengths = [int(traj.shape[0]) for traj in bundle.gru_trajectories]

    # Fit the exact whole-corpus centered PCA without materialising the corpus.
    pca = _fit_centered_pca(
        _iter_feature_blocks(bundle, use_combined), n_components,
    )
    logger.info("PCA input shape: (%d, %d)", pca.n_samples_, pca.n_features_in_)

    explained_var = pca.explained_variance_ratio_
    logger.info(
        "PCA explained variance: %s (total=%.2f%%)",
        ", ".join(f"{v * 100:.2f}%" for v in explained_var),
        float(explained_var.sum()) * 100,
    )
    log_pca_variance(explained_var, logger.info)

    # Transform trajectories (second streaming pass; the fitted mean_ and
    # components_ give the same per-trajectory projection as pca.transform).
    trajectories_3d = []
    for feat in _iter_feature_blocks(bundle, use_combined):
        traj_3d = pca.transform(feat)
        trajectories_3d.append(traj_3d)

    logger.info("Transformed %d trajectories to 3D.", len(trajectories_3d))
    if len(bundle.labels) != len(lengths):
        raise ValueError(
            f"Label count {len(bundle.labels)} != trajectory count {len(lengths)}"
        )
    all_labels = np.repeat(
        np.asarray(bundle.labels, dtype=np.int64), np.asarray(lengths, dtype=np.int64)
    )
    assert all_labels.shape[0] == sum(lengths), (
        f"Expected {sum(lengths)} per-state labels, got {all_labels.shape[0]}"
    )
    return trajectories_3d, all_labels, pca


# =========================================================================
# 5.  Per-Label Statistics Helpers
# =========================================================================

def group_trajectories_by_label(
    trajectories: List[np.ndarray],
    labels: List[int],
) -> Dict[int, List[np.ndarray]]:
    """Group trajectories by their integer label."""
    groups: Dict[int, List[np.ndarray]] = {}
    for traj, lbl in zip(trajectories, labels):
        groups.setdefault(lbl, []).append(traj)
    return groups


def compute_mean_sem_over_time(
    trajectories: List[np.ndarray],
    max_len: Optional[int] = None,
    dt_ms: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean and SEM of trajectories aligned to sequence start.

    Trajectories of different lengths are aligned at index 0 and
    truncated to the shortest length for valid averaging.

    Args:
        trajectories: List of 1D arrays, each (T_i,).
        max_len: Maximum time length to consider. None = use shortest.
        dt_ms: Frame interval in ms.

    Returns:
        ``(time_axis, mean, sem)`` arrays.
    """
    if not trajectories:
        return np.array([]), np.array([]), np.array([])

    min_len = min(len(t) for t in trajectories)
    if max_len is not None:
        min_len = min(min_len, max_len)

    # Stack into (n_trials, T) matrix
    stacked = np.zeros((len(trajectories), min_len))
    for i, traj in enumerate(trajectories):
        stacked[i, :min_len] = traj[:min_len]

    mean = stacked.mean(axis=0)
    sem = stacked.std(axis=0, ddof=1) / np.sqrt(len(trajectories))
    time_axis = np.arange(min_len) * dt_ms / 1000.0  # seconds

    return time_axis, mean, sem


# =========================================================================
# 6.  Lancet/Cell Style Setup
# =========================================================================

def setup_lancet_style() -> None:
    """Configure matplotlib for Cell/Science publication aesthetics.

    Uses vivid, high-contrast colors — rejects pale/desaturated palettes.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE_TICK,
        "axes.titlesize": FONT_SIZE_AXIS_TITLE,
        "axes.labelsize": FONT_SIZE_AXIS_TITLE,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "axes.linewidth": 1.5,
        "axes.edgecolor": AXIS_COLOR,
        "axes.labelcolor": AXIS_COLOR,
        "xtick.color": AXIS_COLOR,
        "ytick.color": AXIS_COLOR,
        "axes.grid": False,
        "grid.alpha": 0.15,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "figure.facecolor": BACKGROUND_COLOR,
        "savefig.facecolor": BACKGROUND_COLOR,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "legend.frameon": True,
        "legend.facecolor": BACKGROUND_COLOR,
        "legend.edgecolor": AXIS_COLOR,
        "legend.framealpha": 1.0,
    })


def _style_axes(ax: plt.Axes) -> None:
    """Apply consistent spine and tick styling to a 2D axes."""
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(1.5)
    ax.tick_params(axis="both", colors=AXIS_COLOR, width=1.5)


# =========================================================================
# 7.  Panel A: 3D Phase-Space Manifold
# =========================================================================

def plot_panel_a_3d_manifold(
    ax: Axes3D,
    trajectories_3d: List[np.ndarray],
    labels: np.ndarray,
    pca_explained_var: np.ndarray,
    max_points_per_class: int = 400,
) -> None:
    """
    Plot Panel A: 3D Phase-Space Manifold (combined LIF+GRU PCA).

    Uses balanced per-class subsampling to prevent the majority class
    from visually overwhelming minority classes (Escape, No Response).

    Args:
        ax: Matplotlib 3D axes.
        trajectories_3d: List of arrays, each (T_i, 3).
        labels: Per-state labels (one per state point).
        pca_explained_var: PCA explained variance ratios.
        max_points_per_class: Maximum scatter points per label class
            for balanced visualization.
    """
    # Reconstruct per-trajectory labels from per-state labels
    traj_labels: List[int] = []
    offset = 0
    for traj in trajectories_3d:
        T_i = traj.shape[0]
        traj_labels.append(int(labels[offset]))
        offset += T_i

    # Plot trajectory lines (all classes)
    for i, traj in enumerate(trajectories_3d):
        label_val = traj_labels[i]
        color = LANCET_COLORS.get(label_val, "#000000")
        ax.plot(
            traj[:, 0], traj[:, 1], traj[:, 2],
            color=color,
            alpha=TRAJECTORY_ALPHA,
            linewidth=TRAJECTORY_LINEWIDTH,
            solid_capstyle="round",
        )

    # Balanced per-class scatter for legend and visual clarity
    legend_handles = []
    seen_labels: set = set()

    for i, traj in enumerate(trajectories_3d):
        label_val = traj_labels[i]
        if label_val in seen_labels:
            continue
        seen_labels.add(label_val)

        color = LANCET_COLORS.get(label_val, "#000000")
        marker = LABEL_MARKERS.get(label_val, "o")
        label_name = LABEL_NAMES.get(label_val, f"Unknown({label_val})")

        # Count trajectories for this label
        count = sum(1 for lbl in traj_labels if lbl == label_val)

        # Collect all states for this label
        all_label_states = []
        for j, tj in enumerate(trajectories_3d):
            if traj_labels[j] == label_val:
                all_label_states.append(tj)
        all_label_states_cat = np.concatenate(all_label_states, axis=0)
        n_points = all_label_states_cat.shape[0]

        # Balanced subsampling: cap each class at max_points_per_class
        # This prevents the majority class from visually dominating
        if n_points > max_points_per_class:
            indices = np.linspace(0, n_points - 1, max_points_per_class, dtype=int)
            states_sub = all_label_states_cat[indices]
        else:
            states_sub = all_label_states_cat

        # Plot class centroid as a prominent marker
        centroid = all_label_states_cat.mean(axis=0)
        ax.scatter(
            [centroid[0]], [centroid[1]], [centroid[2]],
            color=color,
            marker="*",
            s=SCATTER_SIZE * 4,
            alpha=1.0,
            edgecolors="black",
            linewidths=0.8,
            depthshade=False,
            zorder=5,
        )

        scatter = ax.scatter(
            states_sub[:, 0], states_sub[:, 1], states_sub[:, 2],
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

    # Axes styling
    n_comp = len(pca_explained_var)
    ax.set_xlabel(
        f"PC1 ({pca_explained_var[0]*100:.1f}%)",
        fontsize=FONT_SIZE_AXIS_TITLE, color=AXIS_COLOR, labelpad=10,
    )
    ax.set_ylabel(
        f"PC2 ({pca_explained_var[1]*100:.1f}%)",
        fontsize=FONT_SIZE_AXIS_TITLE, color=AXIS_COLOR, labelpad=10,
    )
    if n_comp >= 3:
        ax.set_zlabel(
            f"PC3 ({pca_explained_var[2]*100:.1f}%)",
            fontsize=FONT_SIZE_AXIS_TITLE, color=AXIS_COLOR, labelpad=10,
        )

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(AXIS_COLOR)
    ax.yaxis.pane.set_edgecolor(AXIS_COLOR)
    ax.zaxis.pane.set_edgecolor(AXIS_COLOR)
    ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        facecolor=BACKGROUND_COLOR,
        edgecolor=AXIS_COLOR,
        framealpha=1.0,
    )

    ax.set_title(
        "A",
        fontsize=FONT_SIZE_PANEL_LABEL,
        fontweight="bold",
        color=AXIS_COLOR,
        loc="left",
        pad=20,
    )


# =========================================================================
# 8.  Panel B: Per-Class Routing Gate Dynamics
# =========================================================================

def plot_panel_b_routing_gates(
    ax: plt.Axes,
    bundle: DynamicsBundle,
    dt_ms: float = 10.0,
) -> None:
    """
    Plot Panel B: Per-label-class routing gate dynamics (mean +/- SEM).

    Shows both g_gru(t) (solid) and g_lif(t) (dashed) for each
    behavioral class with shaded SEM regions.  This reveals the full
    routing decision structure: which pathway dominates for which class.

    Args:
        ax: Matplotlib axes.
        bundle: DynamicsBundle with extracted data.
        dt_ms: Frame interval in milliseconds.
    """
    # Group gate trajectories by label
    g_gru_groups = group_trajectories_by_label(
        bundle.g_gru_trajs, bundle.labels,
    )
    g_lif_groups = group_trajectories_by_label(
        bundle.g_lif_trajs, bundle.labels,
    )

    from matplotlib.lines import Line2D
    legend_elements: List = []

    for lbl_val in sorted(g_gru_groups.keys()):
        gru_trajs = g_gru_groups[lbl_val]
        lif_trajs = g_lif_groups.get(lbl_val, [])
        if len(gru_trajs) < 2:
            continue

        color = LANCET_COLORS.get(lbl_val, "#000000")
        label_name = LABEL_NAMES.get(lbl_val, f"Label {lbl_val}")
        n = len(gru_trajs)

        # -- g_gru: solid line with SEM fill --
        time_gru, mean_gru, sem_gru = compute_mean_sem_over_time(gru_trajs, dt_ms=dt_ms)
        if len(time_gru) > 0:
            ax.plot(
                time_gru, mean_gru,
                color=color,
                linewidth=LINE_WIDTH,
                alpha=0.85,
                solid_capstyle="round",
                linestyle="-",
            )
            ax.fill_between(
                time_gru, mean_gru - sem_gru, mean_gru + sem_gru,
                color=color,
                alpha=SEM_ALPHA,
            )

        # -- g_lif: dashed line (no fill for clarity) --
        if len(lif_trajs) >= 2:
            time_lif, mean_lif, _sem_lif = compute_mean_sem_over_time(lif_trajs, dt_ms=dt_ms)
            if len(time_lif) > 0:
                ax.plot(
                    time_lif, mean_lif,
                    color=color,
                    linewidth=LINE_WIDTH * 0.7,
                    alpha=0.65,
                    linestyle="--",
                )

        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=LINE_WIDTH,
                   linestyle="-", label=f"{label_name} $g_{{gru}}$ (n={n})")
        )
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=LINE_WIDTH * 0.7,
                   linestyle="--", label=f"{label_name} $g_{{lif}}$")
        )

    # Reference line at 0.5 (equal routing)
    ax.axhline(
        y=0.5, color=AXIS_COLOR, linewidth=0.8,
        linestyle="--", alpha=0.3,
    )

    # Axes styling
    ax.set_xlabel("Time (s)", fontsize=FONT_SIZE_AXIS_TITLE, color=AXIS_COLOR)
    ax.set_ylabel(
        "Routing Probability $g(t)$ (trial SEM; descriptive)",
        fontsize=FONT_SIZE_AXIS_TITLE, color=AXIS_COLOR,
    )
    ax.set_ylim(-0.05, 1.05)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    _style_axes(ax)
    ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=FONT_SIZE_LEGEND - 1,
        frameon=True,
        facecolor=BACKGROUND_COLOR,
        edgecolor=AXIS_COLOR,
        framealpha=1.0,
        ncol=2,
    )

    ax.set_title(
        "B",
        fontsize=FONT_SIZE_PANEL_LABEL,
        fontweight="bold",
        color=AXIS_COLOR,
        loc="left",
    )


# =========================================================================
# 9.  Panel C: LIF Spike Rate Dynamics
# =========================================================================

def plot_panel_c_lif_spike_rates(
    ax: plt.Axes,
    bundle: DynamicsBundle,
    dt_ms: float = 10.0,
) -> None:
    """
    Plot Panel C: Per-label-class LIF spike rate dynamics (mean +/- SEM).

    Shows mean population spike rate over time for each behavioral class.

    Args:
        ax: Matplotlib axes.
        bundle: DynamicsBundle with extracted data.
        dt_ms: Frame interval in milliseconds.
    """
    # Group LIF rate trajectories by label
    rate_groups = group_trajectories_by_label(
        bundle.lif_rate_trajs, bundle.labels,
    )

    from matplotlib.lines import Line2D
    legend_elements = []

    for lbl_val in sorted(rate_groups.keys()):
        trajs = rate_groups[lbl_val]
        if len(trajs) < 2:
            continue

        time_axis, mean, sem = compute_mean_sem_over_time(trajs, dt_ms=dt_ms)
        if len(time_axis) == 0:
            continue

        color = LANCET_COLORS.get(lbl_val, "#000000")
        label_name = LABEL_NAMES.get(lbl_val, f"Label {lbl_val}")

        ax.plot(
            time_axis, mean,
            color=color,
            linewidth=LINE_WIDTH,
            alpha=0.85,
            solid_capstyle="round",
        )

        ax.fill_between(
            time_axis, mean - sem, mean + sem,
            color=color,
            alpha=SEM_ALPHA,
        )

        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=LINE_WIDTH,
                   label=f"{label_name} (n={len(trajs)})")
        )

    # Axes styling
    ax.set_xlabel("Time (s)", fontsize=FONT_SIZE_AXIS_TITLE, color=AXIS_COLOR)
    ax.set_ylabel(
        "LIF Spike Rate (mean ± trial SEM; descriptive)",
        fontsize=FONT_SIZE_AXIS_TITLE, color=AXIS_COLOR,
    )
    ax.set_ylim(bottom=-0.005)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    _style_axes(ax)
    ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        facecolor=BACKGROUND_COLOR,
        edgecolor=AXIS_COLOR,
        framealpha=1.0,
    )

    ax.set_title(
        "C",
        fontsize=FONT_SIZE_PANEL_LABEL,
        fontweight="bold",
        color=AXIS_COLOR,
        loc="left",
    )


# =========================================================================
# 10.  Panel D: Pathway Dominance
# =========================================================================

def plot_panel_d_pathway_dominance(
    ax: plt.Axes,
    bundle: DynamicsBundle,
) -> None:
    """
    Plot Panel D: Pathway dominance per behavioral class.

    Bar chart showing mean g_gru per label class with SD across the
    recorded trial means. Values > 0.5 indicate GRU dominance; < 0.5
    indicate LIF dominance. No animal-level CI or population inference
    is available without verified independent animal identities.

    Annotates a descriptive, trial-level Escape vs No-Response Cohen's d
    when the pooled trial spread is defined.

    Args:
        ax: Matplotlib axes.
        bundle: DynamicsBundle with extracted data.
    """
    g_gru_groups = group_trajectories_by_label(
        bundle.g_gru_trajs, bundle.labels,
    )

    label_vals = sorted(g_gru_groups.keys())
    bar_positions = []
    bar_heights = []
    bar_spread = []
    bar_colors = []
    bar_labels_list = []
    bar_counts = []
    trial_means_dict: Dict[int, np.ndarray] = {}

    for lbl_val in label_vals:
        trajs = g_gru_groups[lbl_val]
        if len(trajs) < 2:
            continue

        # Describe observed model routing on these trials, with no
        # resampling claim about an unidentified animal population.
        trial_means = np.array([np.mean(t) for t in trajs])
        trial_means_dict[lbl_val] = trial_means
        point_est = float(np.mean(trial_means))
        trial_sd = float(np.std(trial_means, ddof=1))

        bar_positions.append(lbl_val)
        bar_heights.append(point_est)
        bar_spread.append(trial_sd)
        bar_colors.append(LANCET_COLORS.get(lbl_val, "#888888"))
        bar_labels_list.append(LABEL_NAMES.get(lbl_val, f"Label {lbl_val}"))
        bar_counts.append(len(trajs))

    if not bar_positions:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_SIZE_AXIS_TITLE)
        return

    x = np.arange(len(bar_positions))

    bars = ax.bar(
        x, bar_heights,
        color=bar_colors,
        edgecolor=AXIS_COLOR,
        linewidth=1.2,
        width=0.6,
        yerr=bar_spread,
        capsize=4,
        error_kw={"linewidth": 1.2, "color": AXIS_COLOR},
        alpha=0.9,
    )

    # 50% reference line (equal routing)
    ax.axhline(
        y=0.5, color=AXIS_COLOR, linewidth=1.0,
        linestyle="--", alpha=0.5,
    )

    # Annotate bars with counts and mean values
    for i, (pos, height, count) in enumerate(zip(bar_positions, bar_heights, bar_counts)):
        ax.text(
            i, height + 0.02,
            f"n={count}\n{height:.2f}",
            ha="center", va="bottom",
            fontsize=FONT_SIZE_LEGEND - 1,
            color=AXIS_COLOR,
        )

    # Axes styling
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels_list, rotation=30, ha="right")
    ax.set_xlabel("Animal identity unverified (descriptive_only)")
    ax.set_ylabel(
        r"Mean $g_{gru}$ ± trial SD (descriptive)",
        fontsize=FONT_SIZE_AXIS_TITLE, color=AXIS_COLOR,
    )
    ax.set_ylim(0.0, 1.15)

    _style_axes(ax)
    ax.grid(True, axis="y", alpha=0.15, linestyle="--", linewidth=0.5)

    # Dominance annotation
    ax.text(
        0.98, 0.95, "GRU dominant",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=FONT_SIZE_LEGEND - 1, color=GATE_GRU_COLOR, fontstyle="italic",
    )
    ax.text(
        0.98, 0.05, "LIF dominant",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=FONT_SIZE_LEGEND - 1, color=GATE_LIF_COLOR, fontstyle="italic",
    )

    # Compute and display Escape vs No-Response effect size if both exist
    esc_val = Label.ESCAPE.value
    nr_val = Label.NO_RESPONSE.value
    if esc_val in trial_means_dict and nr_val in trial_means_dict:
        esc = trial_means_dict[esc_val]
        nr = trial_means_dict[nr_val]
        # Rounded gate trajectories can leave tiny floating-point pooled
        # spread even for constant classes; standardized d is undefined.
        pooled_variance = (
            (len(esc) - 1) * np.var(esc, ddof=1)
            + (len(nr) - 1) * np.var(nr, ddof=1)
        ) / (len(esc) + len(nr) - 2)
        pooled_sd = float(np.sqrt(pooled_variance))
        d = (cohens_d(esc, nr) if np.isfinite(pooled_sd) and pooled_sd > 1e-12
             else float("nan"))
        d_text = (f"d={d:.2f} (trial-level descriptive)" if np.isfinite(d)
                  else "d undefined (zero or nonfinite trial spread)")
        ax.text(
            0.5, 0.97,
            f"Escape vs NoResp: {d_text}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_SIZE_LEGEND - 1,
            color=AXIS_COLOR,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BACKGROUND_COLOR,
                      edgecolor=AXIS_COLOR, alpha=0.8),
        )

    ax.set_title(
        "D",
        fontsize=FONT_SIZE_PANEL_LABEL,
        fontweight="bold",
        color=AXIS_COLOR,
        loc="left",
    )


# =========================================================================
# 11.  Figure Assembly
# =========================================================================

def create_panel_figure(
    trajectories_3d: List[np.ndarray],
    all_labels: np.ndarray,
    pca_explained_var: np.ndarray,
    bundle: DynamicsBundle,
    output_path: Path,
    layout: str = "2x2",
    dt_ms: float = 10.0,
    analysis_population: Optional[Dict[str, object]] = None,
) -> None:
    """
    Create the Lancet/Cell multi-panel publication figure.

    Args:
        trajectories_3d: List of 3D trajectory arrays.
        all_labels: Per-state labels for Panel A.
        pca_explained_var: PCA explained variance ratios.
        bundle: DynamicsBundle with all extracted data.
        output_path: Path to save the figure.
        layout: Panel layout, "2x2" (4 panels) or "1x2" (2 panels, backward compat).
        dt_ms: Frame interval in milliseconds.
    """
    setup_lancet_style()

    if layout == "2x2":
        fig = plt.figure(figsize=(14.0, 12.0))

        # Panel A: 3D Manifold (top-left)
        ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
        plot_panel_a_3d_manifold(ax_3d, trajectories_3d, all_labels, pca_explained_var)

        # Panel B: Routing gates (top-right)
        ax_gate = fig.add_subplot(2, 2, 2)
        plot_panel_b_routing_gates(ax_gate, bundle, dt_ms=dt_ms)

        # Panel C: LIF spike rates (bottom-left)
        ax_lif = fig.add_subplot(2, 2, 3)
        plot_panel_c_lif_spike_rates(ax_lif, bundle, dt_ms=dt_ms)

        # Panel D: Pathway dominance (bottom-right)
        ax_dom = fig.add_subplot(2, 2, 4)
        plot_panel_d_pathway_dominance(ax_dom, bundle)

        plt.tight_layout(pad=2.0)

    elif layout == "1x2":
        # Backward-compatible 2-panel layout
        fig = plt.figure(figsize=(14.0, 6.0))

        ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
        plot_panel_a_3d_manifold(ax_3d, trajectories_3d, all_labels, pca_explained_var)

        ax_gate = fig.add_subplot(1, 2, 2)
        plot_panel_b_routing_gates(ax_gate, bundle, dt_ms=dt_ms)

    else:
        raise ValueError(f"Unknown layout '{layout}'. Use '2x2' or '1x2'.")

    if analysis_population is not None:
        fig.suptitle(
            "%s — %s (train=%s, outer val=%s)"
            % (analysis_population["selection"].replace("_", " "),
               analysis_population["evidence_scope"].replace("_", " "),
               analysis_population["n_analyzed_train"], analysis_population["n_analyzed_val"]),
            fontsize=12, y=1.02,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.1,
        metadata={"Description": json.dumps({"analysis_population": analysis_population})}
        if analysis_population is not None else None,
    )
    logger.info("Saved %s figure to %s (%d DPI)", layout, output_path, DPI)
    plt.close(fig)


# =========================================================================
# 12.  Main Analysis Pipeline
# =========================================================================

def run_analysis(
    checkpoint_path: Path,
    dataset_path: Path,
    output_path: Path,
    batch_size: int = 32,
    max_seq_len: Optional[int] = 2400,
    pre_anchor_frames: int = 1200,
    layout: str = "2x2",
    nested_prior_artifact: Optional[Path] = None,
    qc_sealed_nested_prior_sha256: Optional[str] = None,
    trusted_historical_checkpoint_sha256: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
) -> None:
    """
    Run the full multi-panel mechanism analysis pipeline.

    Args:
        checkpoint_path: Path to the trained model checkpoint.
        dataset_path: Path to the preprocessed dataset.
        output_path: Path to save the figure.
        batch_size: Batch size for data loading.
        max_seq_len: Maximum sequence length (cuDNN compat). None = no limit.
        pre_anchor_frames: Baseline frames before stimulus anchor.
        layout: Panel layout ("2x2" or "1x2").
    """
    logger.info("=" * 60)
    logger.info("NSMoR Mechanism Analysis -- Multi-Panel Figure")
    logger.info("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model
    model = load_model_from_checkpoint(checkpoint_path, device)

    # Load dataset
    dataloader, labels = load_dataset(
        dataset_path,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        nested_prior_artifact=nested_prior_artifact,
        qc_sealed_nested_prior_sha256=qc_sealed_nested_prior_sha256,
        checkpoint_model=model,
        trusted_historical_checkpoint_sha256=trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
    )

    # Single-pass extraction of all dynamics
    bundle = extract_full_dynamics(model, dataloader, labels, device)

    # PCA manifold (combined LIF potentials + GRU hidden states)
    # This uses the full H-dimensional LIF membrane potential state
    # (not just the 1-D mean spike rate) to give the LIF pathway
    # equal representation in the manifold visualization.
    trajectories_3d, all_labels, pca = compute_pca_manifold(
        bundle, n_components=3, use_combined=True,
    )

    # Create figure
    create_panel_figure(
        trajectories_3d=trajectories_3d,
        all_labels=all_labels,
        pca_explained_var=pca.explained_variance_ratio_,
        bundle=bundle,
        output_path=output_path,
        layout=layout,
        dt_ms=float(model.dt_ms),
        analysis_population=population_for_output(dataloader, len(labels)),
    )

    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)


# =========================================================================
# 13.  CLI Entry Point
# =========================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="NSMoR Mechanism Analysis -- Multi-Panel Publication Figure",
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
        "--output",
        type=str,
        default="results/mechanism_analysis.png",
        help="Output path for the figure.",
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
    parser.add_argument(
        "--layout",
        type=str,
        default="2x2",
        choices=["2x2", "1x2"],
        help="Panel layout: '2x2' (4 panels) or '1x2' (backward-compatible 2 panels).",
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
        output_path=Path(args.output),
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        layout=args.layout,
    )


if __name__ == "__main__":
    main()
