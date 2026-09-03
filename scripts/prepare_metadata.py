#!/usr/bin/env python3
"""
NSMoR Metadata-Only Preparation Pipeline (ELT Mode)

Extracts trial boundaries, labels, and MCMC priors WITHOUT loading full sequences.
Produces a lightweight metadata file (~MB) instead of full dataset (~GB).

Usage:
    python scripts/prepare_metadata.py --raw_dir data/staging_3cond_1440
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nsmor.config import (
    DEFAULT_FEATURE,
    DEFAULT_THRESHOLD,
    DEFAULT_TIME_WINDOW,
    FeatureConfig,
    PIPELINE_SEMANTICS_VERSION,
    TimeWindowConfig,
)
from nsmor.data_extractor import (
    build_snapshot_dataset,
    resolve_snapshot_anchor,
)
from nsmor.mcmc_module import train_mcmc_cross_fitted
from nsmor.pipeline.io import (
    extract_trial_data,
    load_and_concat_sessions,
)
from nsmor.pipeline.labeling import (
    assign_ground_truth_labels,
    labeling_funnel_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. Data Pairing (reuse prepare_data.py logic)
# ═══════════════════════════════════════════════════════════════


def pair_csv_files(raw_dir: Path) -> List[Tuple[Path, Path]]:
    """Scan raw_dir for session subdirectories, pair kinematics/events CSVs."""
    pairs: List[Tuple[Path, Path]] = []
    for session_dir in sorted(raw_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        kin_candidates = list(session_dir.glob("*kinematics*.csv"))
        evt_candidates = list(session_dir.glob("*events*.csv"))
        if kin_candidates and evt_candidates:
            pairs.append((kin_candidates[0], evt_candidates[0]))
            logger.info("Paired: %s <-> %s", kin_candidates[0].name, evt_candidates[0].name)
    if not pairs:
        raise FileNotFoundError(f"No kinematics/events CSV pairs in {raw_dir}")
    return pairs


# ═══════════════════════════════════════════════════════════════
# 2. Main Pipeline
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    """Metadata extraction pipeline (ELT mode).

    Reuses the SAME labeling, snapshot extraction, and cross-fitted MCMC
    prior generation as :mod:`scripts.prepare_data`, but produces only
    lightweight trial metadata (~MB) instead of full sequences (~GB).
    """
    parser = argparse.ArgumentParser(
        description="NSMoR Metadata-Only Preparation (ELT Mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Root directory containing raw session data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/nsmor_metadata.pt",
        help="Output path for metadata file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for MCMC training.",
    )
    parser.add_argument(
        "--dt_ms",
        type=float,
        default=10.0,
        help="Frame interval in milliseconds (for reference only in ELT mode).",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    random_seed = args.seed

    feature_config = DEFAULT_FEATURE
    time_config = DEFAULT_TIME_WINDOW

    logger.info("=" * 60)
    logger.info("NSMoR Metadata Extraction Pipeline (ELT Mode)")
    logger.info("Pipeline semantics version: %s", PIPELINE_SEMANTICS_VERSION)
    logger.info("=" * 60)

    # ── Step 1: Data Pairing ───────────────────────────────────────
    logger.info("[Step 1] Scanning for data pairs in %s", raw_dir)
    csv_pairs = pair_csv_files(raw_dir)
    logger.info("Found %d session pairs.", len(csv_pairs))

    # ── Step 2: Load and concatenate sessions ─────────────────────
    logger.info("[Step 2] Loading and concatenating sessions...")
    kin_paths = [p[0] for p in csv_pairs]
    evt_paths = [p[1] for p in csv_pairs]

    session_data = load_and_concat_sessions(kin_paths, evt_paths)
    logger.info(
        "Loaded %d kinematics rows, %d events rows.",
        len(session_data["kinematics"]),
        len(session_data["events"]),
    )

    # ── Step 3: Per-trial extraction and labeling ─────────────────
    logger.info("[Step 3] Extracting trials and assigning labels...")
    trial_groups = session_data["kinematics"].groupby(["session_id", "trial_id"])
    trials: List[Dict[str, Any]] = []
    for (session_id, trial_id), _ in trial_groups:
        try:
            trial = extract_trial_data(session_data, session_id, trial_id)
            trials.append(trial)
        except ValueError as e:
            logger.warning("Skipping trial: %s", e)
            continue

    logger.info("Extracted %d valid trials.", len(trials))

    # Assign ground truth labels (same logic as prepare_data.py)
    labeled_trials = assign_ground_truth_labels(trials, return_funnel=True)
    logger.info("Labeled %d trials.", len(labeled_trials))

    funnel = labeling_funnel_summary(labeled_trials)
    logger.info("Labeling elimination funnel: %s", funnel)

    # ── Step 4: Build snapshot dataset + MCMC priors ─────────────
    logger.info("[Step 4] Building snapshot dataset and training MCMC priors...")

    snapshots, snapshot_labels, kept_indices, snapshot_anchor_rules = (
        build_snapshot_dataset(
            labeled_trials,
            time_config=time_config,
            feature_config=feature_config,
            return_kept_indices=True,
            return_anchor_rules=True,
            on_unanchorable="skip",
        )
    )
    logger.info(
        "Snapshot dataset: %s snapshots, %s labels. Anchor rules: %s",
        snapshots.shape,
        snapshot_labels.shape,
        {
            rule: snapshot_anchor_rules.count(rule)
            for rule in sorted(set(snapshot_anchor_rules))
        },
    )

    # Session-grouped cross-fitted MCMC priors (same as prepare_data.py)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    labeled_kept = [labeled_trials[i] for i in kept_indices]
    snapshot_groups = np.array(
        [str(info["session_id"]) for info in labeled_kept], dtype=object,
    )
    assert len(snapshot_groups) == len(snapshots), (
        f"Session-group count {len(snapshot_groups)} != "
        f"snapshot count {len(snapshots)}"
    )

    mcmc_priors, fold_models, fold_diagnostics = train_mcmc_cross_fitted(
        snapshots,
        snapshot_labels,
        n_folds=5,
        groups=snapshot_groups,
        verbose=True,
    )
    n_sessions = len(set(snapshot_groups))
    logger.info(
        "Generated out-of-fold MCMC priors (5-fold session-grouped "
        "cross-fitting over %d sessions): %s",
        n_sessions,
        mcmc_priors.shape,
    )
    assert mcmc_priors.shape == (len(snapshots), feature_config.mcmc_dim), (
        f"MCMC priors shape {mcmc_priors.shape} != "
        f"({len(snapshots)}, {feature_config.mcmc_dim})"
    )

    # ── Step 5: Build trial specs for lazy loading ─────────────────
    logger.info("[Step 5] Building trial specs for lazy loader...")
    trial_specs: List[Dict[str, Any]] = []

    for meta_idx, kept_idx in enumerate(kept_indices):
        info = labeled_trials[kept_idx]
        trial_data = info["trial_data"]

        session_id = str(info["session_id"])
        trial_id = int(info["trial_id"])
        stimulus_onset_ms = float(info["stimulus_onset_ms"])

        anchor_ms, anchor_rule = resolve_snapshot_anchor(
            trial_data, stimulus_onset_ms,
        )

        # Find source CSV paths: match session_id back to csv_pairs
        # Session data is concat'd, so we find the session dir from the
        # original csv_pairs list.
        session_dir: Optional[str] = None
        kin_file: Optional[str] = None
        evt_file: Optional[str] = None
        for kin_path, evt_path in csv_pairs:
            if session_id in kin_path.stem or session_id in kin_path.parent.name:
                session_dir = str(kin_path.parent)
                kin_file = kin_path.name
                evt_file = evt_path.name
                break

        if session_dir is None:
            # Fallback: derive from session_id pattern
            # Session dirs are named by session_id in the staging layout
            candidate = raw_dir / session_id
            if candidate.is_dir():
                session_dir = str(candidate)
                kin_candidates = list(candidate.glob("*kinematics*.csv"))
                evt_candidates = list(candidate.glob("*events*.csv"))
                if kin_candidates and evt_candidates:
                    kin_file = kin_candidates[0].name
                    evt_file = evt_candidates[0].name

        time_ms = trial_data["time_ms"]
        n_frames = len(time_ms)

        # Compute anchor frame index for lazy loader crop alignment
        anchor_frame = int(np.argmin(np.abs(time_ms - anchor_ms)))

        trial_specs.append({
            "session_id": session_id,
            "session_dir": session_dir or "",
            "kinematics_file": kin_file or "",
            "events_file": evt_file or "",
            "trial_id": trial_id,
            "n_frames": n_frames,
            "trial_start_ms": float(time_ms[0]),
            "stimulus_onset_ms": stimulus_onset_ms,
            "anchor_ms": float(anchor_ms),
            "anchor_frame": anchor_frame,
            "anchor_rule": anchor_rule,
            "label": info["label"].name,  # Label enum -> string
        })

    logger.info("Built %d trial specs.", len(trial_specs))

    # ── Step 6: Save lightweight metadata ─────────────────────────
    logger.info("[Step 6] Saving metadata to %s", output_path)

    metadata = {
        "trial_specs": trial_specs,
        "mcmc_priors": torch.from_numpy(mcmc_priors).float(),
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "n_trials": len(trial_specs),
        "label_encoder": {label.name: label.value for label in __import__("nsmor.config", fromlist=["Label"]).Label},
        "feature_config": feature_config,
        "snapshot_anchor_rules": snapshot_anchor_rules,
        "n_sessions": n_sessions,
        "session_ids": [spec["session_id"] for spec in trial_specs],
    }

    torch.save(metadata, output_path)
    logger.info("Saved metadata to %s (%.2f MB)", output_path, output_path.stat().st_size / 1e6)

    # Print statistics
    from collections import Counter

    label_counts = Counter(spec["label"] for spec in trial_specs)
    logger.info("Label distribution: %s", dict(label_counts))
    logger.info("Total frames: %d", sum(spec["n_frames"] for spec in trial_specs))
    logger.info(
        "Anchor rule distribution: %s",
        {
            rule: snapshot_anchor_rules.count(rule)
            for rule in sorted(set(snapshot_anchor_rules))
        },
    )

    logger.info("=" * 60)
    logger.info("Metadata extraction complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
