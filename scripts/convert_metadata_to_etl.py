"""Convert metadata format to ETL format for pre-loading.

Loads trial specs from metadata and converts to pre-loaded X_seqs/Y_seqs format.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nsmor.lazy_dataloader import NSMoRLazyDataset


def populate_etl_provenance_and_conditions(
    output: Dict[str, Any],
    metadata: Dict[str, Any],
    X_seqs: Optional[List[np.ndarray]] = None,
    lengths: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Populate provenance and condition metadata into output ETL dictionary.

    Adds "mcmc_prior_provenance", "anchor_frames", "stimulus_conditions",
    and "is_pure_wind", copying from input metadata where available.

    Args:
        output: ETL dataset dictionary to enrich in-place.
        metadata: Source metadata dictionary.
        X_seqs: Optional extracted physical sequences for fallback derivation.
        lengths: Optional sequence lengths.

    Returns:
        The enriched output dictionary.
    """
    # ── MCMC prior provenance ─────────────────────────────────────
    if "mcmc_prior_provenance" in metadata:
        output["mcmc_prior_provenance"] = metadata["mcmc_prior_provenance"]

    # ── Anchor frames ─────────────────────────────────────────────
    if "anchor_frames" in metadata:
        output["anchor_frames"] = metadata["anchor_frames"]
    elif "trial_specs" in metadata and all("anchor_frame" in s for s in metadata["trial_specs"]):
        output["anchor_frames"] = [int(s["anchor_frame"]) for s in metadata["trial_specs"]]
    elif X_seqs is not None and lengths is not None:
        try:
            from nsmor.pipeline.conditions import derive_anchor_frames
            output["anchor_frames"] = derive_anchor_frames(X_seqs, lengths)
        except Exception:
            pass

    # ── Stimulus conditions ───────────────────────────────────────
    if "stimulus_conditions" in metadata:
        output["stimulus_conditions"] = metadata["stimulus_conditions"]
    elif "trial_specs" in metadata and all("stimulus_condition" in s for s in metadata["trial_specs"]):
        output["stimulus_conditions"] = [s["stimulus_condition"] for s in metadata["trial_specs"]]
    elif X_seqs is not None and lengths is not None:
        try:
            from nsmor.pipeline.conditions import derive_stimulus_metadata
            conditions_derived, pure_wind_derived = derive_stimulus_metadata(X_seqs, lengths)
            output["stimulus_conditions"] = conditions_derived
            if "is_pure_wind" not in output:
                output["is_pure_wind"] = pure_wind_derived
        except Exception:
            pass

    # ── is_pure_wind ──────────────────────────────────────────────
    if "is_pure_wind" in metadata:
        is_pw = metadata["is_pure_wind"]
        output["is_pure_wind"] = is_pw if isinstance(is_pw, np.ndarray) else np.array(is_pw, dtype=bool)
    elif "trial_specs" in metadata and all("is_pure_wind" in s for s in metadata["trial_specs"]):
        output["is_pure_wind"] = np.array([s["is_pure_wind"] for s in metadata["trial_specs"]], dtype=bool)
    elif "stimulus_conditions" in output and "is_pure_wind" not in output:
        output["is_pure_wind"] = np.array([c == "wind_only" for c in output["stimulus_conditions"]], dtype=bool)

    return output


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert metadata format to ETL pre-loaded dataset.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/nsmor_metadata_3cond_v2.pt",
        help="Path to input metadata file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/nsmor_dataset_3cond_v2.pt",
        help="Path to output pre-loaded dataset file.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2400,
        help="Maximum sequence length for cropping.",
    )
    parser.add_argument(
        "--pre_anchor_frames",
        type=int,
        default=1200,
        help="Frames before anchor to include.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    metadata_path = args.input
    output_path = args.output

    print(f"Loading metadata from {metadata_path}...")
    metadata = torch.load(metadata_path, weights_only=False)

    trial_specs = metadata["trial_specs"]
    mcmc_priors = metadata["mcmc_priors"]
    feature_config = metadata["feature_config"]

    print(f"Converting {len(trial_specs)} trials to ETL format...")

    # Get label encoder
    label_encoder = metadata["label_encoder"]

    # Create lazy dataset to leverage existing loading logic
    lazy_ds = NSMoRLazyDataset(
        metadata_path=metadata_path,
        max_seq_len=args.max_seq_len,
        pre_anchor_frames=args.pre_anchor_frames,
        feature_config=feature_config,
    )

    X_seqs = []
    Y_seqs = []
    labels = []
    lengths = []

    for i in tqdm(range(len(lazy_ds)), desc="Loading sequences"):
        X, Y, length = lazy_ds[i]
        # Create 8-D array: physical (4) + MCMC placeholder (4)
        # NSMoRDataset will fill columns 4:8
        X_8d = np.zeros((X.shape[0], 8), dtype=np.float32)
        X_8d[:, :4] = X[:, :4].numpy()  # Copy physical features
        X_seqs.append(X_8d)
        Y_seqs.append(Y.numpy())
        label_str = lazy_ds.get_label(i)
        labels.append(label_encoder[label_str])  # Convert to int
        lengths.append(length)

    # Build output dict matching old format
    priors_np = (
        mcmc_priors.numpy()
        if isinstance(mcmc_priors, torch.Tensor)
        else np.asarray(mcmc_priors)
    )
    output = {
        "X_seqs": X_seqs,
        "Y_seqs": Y_seqs,
        "mcmc_priors": priors_np,
        "labels": labels,
        "lengths": np.array(lengths, dtype=np.int64),  # Convert to numpy array
        "session_ids": metadata.get("session_ids", []),
        "feature_config": feature_config,
        "pipeline_semantics_version": metadata.get(
            "pipeline_semantics_version", "unknown"
        ),
    }

    populate_etl_provenance_and_conditions(
        output, metadata, X_seqs=X_seqs, lengths=lengths
    )

    print(f"Saving to {output_path}...")
    torch.save(output, output_path)
    print(f"[OK] Saved {len(X_seqs)} sequences")
    print(f"  Total frames: {sum(X.shape[0] for X in X_seqs)}")


if __name__ == "__main__":
    main()
