"""Convert metadata format to ETL format for pre-loading.

Loads trial specs from metadata and converts to pre-loaded X_seqs/Y_seqs format.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nsmor.pipeline.io import ClockAwareLazyDataset
from nsmor.pipeline.nested_prior import load_artifact_bytes
from nsmor.pipeline.grouping import prior_identity_status


def populate_etl_provenance_and_conditions(
    output: Dict[str, Any],
    metadata: Dict[str, Any],
    X_seqs: Optional[List[np.ndarray]] = None,
    lengths: Optional[Sequence[int]] = None,
    max_seq_len: Optional[int] = 2400,
) -> Dict[str, Any]:
    """Populate provenance and condition metadata into output ETL dictionary.

    Adds "mcmc_prior_provenance", "anchor_frames", "stimulus_conditions",
    and "is_pure_wind", copying from input metadata where available.
    When X_seqs and lengths are provided, derives anchor_frames strictly for the
    SAVED arrays and asserts that required stimulus channels were not lost during cropping.

    Args:
        output: ETL dataset dictionary to enrich in-place.
        metadata: Source metadata dictionary.
        X_seqs: Optional extracted physical sequences for derivation and crop verification.
        lengths: Optional sequence lengths.
        max_seq_len: Maximum sequence length used for cropping (to translate
            raw anchors into saved coordinates for uncropped sequences).
        metadata: Source metadata dictionary.
        X_seqs: Optional extracted physical sequences for derivation and crop verification.
        lengths: Optional sequence lengths.

    Returns:
        The enriched output dictionary.
    """
    # ── MCMC prior provenance ─────────────────────────────────────
    if "mcmc_prior_provenance" in metadata:
        output["mcmc_prior_provenance"] = metadata["mcmc_prior_provenance"]
        output["animal_identity_status"] = prior_identity_status(
            metadata["mcmc_prior_provenance"], metadata.get("animal_identity_status")
        )

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

    # ── Anchor frames ─────────────────────────────────────────────
    # Anchor_frames stored in ETL MUST describe the SAVED arrays.
    if X_seqs is not None and lengths is not None:
        from nsmor.pipeline.conditions import derive_anchor_frames

        saved_anchors: List[int] = []
        derived_all = derive_anchor_frames(X_seqs, lengths)
        meta_anchors = metadata.get("anchor_frames")
        if meta_anchors is None and "trial_specs" in metadata:
            meta_anchors = [
                int(s["anchor_frame"]) for s in metadata["trial_specs"] if "anchor_frame" in s
            ]
        max_seq = max_seq_len if max_seq_len is not None else 2400

        for i, (x_seq, valid_len, der_anchor) in enumerate(zip(X_seqs, lengths, derived_all)):
            v_len = int(valid_len)
            x_arr = np.asarray(x_seq)[:v_len]
            has_vis = bool(np.any(np.abs(x_arr[:, 0]) > 1e-4))
            has_wind = bool(np.any(x_arr[:, 1] > 0.5))

            if has_vis or has_wind:
                saved_anchors.append(int(der_anchor))
            else:
                # Channels are silent (synthetic/mock or no-stimulus).
                # If the sequence is strictly shorter than max_seq_len it was
                # certainly not cropped, so the metadata anchor is already in
                # saved coordinates and is trustworthy.  A saved length of
                # exactly max_seq_len is ambiguous (may have been cropped from
                # a longer raw sequence); without the pre-crop length we cannot
                # compute the crop start offset, so fall back to der_anchor (0)
                # as an honest unanchored marker rather than leaking a raw
                # coordinate that may exceed the saved array length.
                meta_a = int(meta_anchors[i]) if meta_anchors is not None and i < len(meta_anchors) else None
                if meta_a is not None and v_len < max_seq and 0 <= meta_a < v_len:
                    saved_anchors.append(meta_a)
                else:
                    saved_anchors.append(int(der_anchor))

        output["anchor_frames"] = saved_anchors
    elif "anchor_frames" in metadata:
        output["anchor_frames"] = metadata["anchor_frames"]
    elif "trial_specs" in metadata and all("anchor_frame" in s for s in metadata["trial_specs"]):
        output["anchor_frames"] = [int(s["anchor_frame"]) for s in metadata["trial_specs"]]

    return output


def resolve_legacy_pure_wind_prepend_frames(
    spec: Dict[str, Any],
    dt_ms: Optional[float],
) -> int:
    """Return prepend frames for a legacy pure-wind spec lacking recorded prepend.

    Uses the explicit effective ``dt_ms`` when available; fails closed when
    the frame interval is unknown rather than hardcoding 4.0 ms.
    """
    from nsmor.data_extractor import _compute_pure_wind_prepend_frames

    recorded = int(spec.get("pure_wind_prepended_frames", 0) or 0)
    if recorded > 0:
        return recorded
    if dt_ms is None:
        raise ValueError(
            "Legacy pure-wind metadata lacks pure_wind_prepended_frames and "
            "dt_ms is unknown (CLI --dt_ms / metadata dt_ms). "
            "Failing closed rather than hardcoding a frame interval."
        )
    return _compute_pure_wind_prepend_frames(float(dt_ms))


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
    parser.add_argument(
        "--dt_ms",
        type=float,
        default=None,
        help="Frame interval in ms. Used for legacy pure-wind prepend "
             "migration; falls back to metadata dt_ms when omitted.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    metadata_path = args.input
    output_path = Path(args.output)

    print(f"Loading metadata from {metadata_path}...")
    metadata = load_artifact_bytes(Path(metadata_path).read_bytes(), map_location="cpu")

    trial_specs = metadata["trial_specs"]
    mcmc_priors = metadata["mcmc_priors"]
    feature_config = metadata["feature_config"]

    print(f"Converting {len(trial_specs)} trials to ETL format...")

    # Get label encoder
    label_encoder = metadata["label_encoder"]

    # Resolve effective dt_ms explicitly (CLI > metadata > feature_config).
    # The same value is used for NSMoRLazyDataset and legacy pure-wind
    # prepend migration so the two cannot disagree.
    dt_ms: Optional[float] = args.dt_ms
    if dt_ms is None:
        dt_ms = metadata.get("dt_ms")
    if dt_ms is None and isinstance(feature_config, dict):
        dt_ms = feature_config.get("dt_ms")
    if dt_ms is not None:
        dt_ms = float(dt_ms)

    # Create lazy dataset to leverage existing loading logic
    lazy_ds = ClockAwareLazyDataset(
        metadata_path=metadata_path,
        metadata=metadata,
        max_seq_len=args.max_seq_len,
        pre_anchor_frames=args.pre_anchor_frames,
        feature_config=feature_config,
        dt_ms=dt_ms,
    )

    # Validate / migrate pure-wind anchor_frames for lazy cropping
    # NSMoRLazyDataset._build_sequence prepends baseline frames to pure-wind trials.
    # If a legacy metadata spec does not have pure_wind_prepended_frames recorded,
    # adjust the in-memory spec anchor_frame so the lazy crop window captures stimulus.

    for spec in lazy_ds.trial_specs:
        if spec.get("is_pure_wind") and spec.get("pure_wind_prepended_frames", 0) == 0:
            prepend = resolve_legacy_pure_wind_prepend_frames(spec, dt_ms)
            spec["anchor_frame"] = int(spec["anchor_frame"]) + prepend
            spec["pure_wind_prepended_frames"] = prepend

    X_seqs = []
    Y_seqs = []
    labels = []
    lengths = []

    with lazy_ds.captured_sources() as revision:
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

    # Propagate trial identity and declared experimental parameters
    trial_ids = metadata.get("trial_ids")
    if trial_ids is None and "trial_specs" in metadata:
        trial_ids = [s["trial_id"] for s in metadata["trial_specs"] if "trial_id" in s]
    if trial_ids is not None:
        if len(trial_ids) != len(X_seqs):
            raise ValueError(f"trial_ids count {len(trial_ids)} != X_seqs count {len(X_seqs)}")
        output["trial_ids"] = trial_ids

    target_ttc_ms = metadata.get("target_ttc_ms")
    if target_ttc_ms is None and "trial_specs" in metadata:
        target_ttc_ms = [s.get("target_ttc_ms") for s in metadata["trial_specs"]]
    if target_ttc_ms is not None:
        if len(target_ttc_ms) != len(X_seqs):
            raise ValueError(f"target_ttc_ms count {len(target_ttc_ms)} != X_seqs count {len(X_seqs)}")
        output["target_ttc_ms"] = target_ttc_ms

    populate_etl_provenance_and_conditions(
        output, metadata, X_seqs=X_seqs, lengths=lengths,
        max_seq_len=args.max_seq_len,
    )

    output["raw_input_revision"] = {
        "claim": "SHA-256 of captured CSV bytes; paths are labels, not a live-path guarantee",
        "source_pairs": revision,
    }

    print(f"Saving to {output_path}...")
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(output, temporary_path)
        lazy_ds.verify_sources()  # Catch changes after earlier trials and during serialization.
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"[OK] Saved {len(X_seqs)} sequences")
    print(f"  Total frames: {sum(X.shape[0] for X in X_seqs)}")


if __name__ == "__main__":
    main()
