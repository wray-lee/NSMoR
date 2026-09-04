"""Convert metadata format to ETL format for pre-loading.

Loads trial specs from metadata and converts to pre-loaded X_seqs/Y_seqs format.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nsmor.lazy_dataloader import NSMoRLazyDataset


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
    output = {
        "X_seqs": X_seqs,
        "Y_seqs": Y_seqs,
        "mcmc_priors": mcmc_priors.numpy(),
        "labels": labels,
        "lengths": np.array(lengths, dtype=np.int64),  # Convert to numpy array
        "session_ids": metadata.get("session_ids", []),
        "feature_config": feature_config,
        "pipeline_semantics_version": metadata.get(
            "pipeline_semantics_version", "unknown"
        ),
    }

    print(f"Saving to {output_path}...")
    torch.save(output, output_path)
    print(f"[OK] Saved {len(X_seqs)} sequences")
    print(f"  Total frames: {sum(X.shape[0] for X in X_seqs)}")


if __name__ == "__main__":
    main()
