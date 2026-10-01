"""Carve a recording-prefix grouped subset from a processed NSMoR dataset.

A recording prefix is a session id without its final _session_N block suffix.
Distinct prefixes may be recordings of the same animal; animal independence is
unverified. Prior provenance and identity status follow the source dataset.

Usage: python scripts/make_subset_dataset.py --input source.pt --output subset.pt
       --n_recording_prefixes 8
"""

from __future__ import annotations

import argparse
import collections
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nsmor.pipeline.conditions import derive_stimulus_metadata
from nsmor.pipeline.grouping import animal_of, prior_identity_status
from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint
from nsmor.model_utils import validate_dataset_provenance

logger = logging.getLogger(__name__)

# Re-exported: this module was the original home of both names, and tests
# plus other scripts import them from here.  ``animal_of`` now lives in
# the installed package so scripts/train.py can share the one definition
# — the split that consumes it could not import it from here.
__all__ = [
    "animal_of",
    "derive_stimulus_metadata",
    "subset_dataset",
    "main",
]

# Keys carrying one entry per trial; sliced by the sampled index set.
_PER_TRIAL_KEYS: Tuple[str, ...] = (
    "X_seqs",
    "Y_seqs",
    "labels",
    "lengths",
    "mcmc_priors",
    "snapshots",
    "mcmc_snapshots",
    "session_ids",
    "trial_ids",
    "trial_specs",
    "target_ttc_ms",
    "stimulus_conditions",
    "is_pure_wind",
    "anchor_frames",
    "anchor_rules",
    "anchor_rule",
    "model_grid_provenance",
    "source_clock_provenance",
)


def group_by_recording_prefix(session_ids: Sequence[str]) -> Dict[str, List[int]]:
    """Map recording-prefix key to its trial indices."""
    groups: Dict[str, List[int]] = collections.defaultdict(list)
    for idx, sid in enumerate(session_ids):
        groups[animal_of(sid)].append(idx)
    return dict(groups)


def select_recording_prefixes(
    groups: Dict[str, List[int]],
    labels: np.ndarray,
    n_recording_prefixes: int,
    seed: int,
) -> List[str]:
    """Choose prefixes covering rare labels, then fill by trial count."""
    if n_recording_prefixes < 1:
        raise ValueError(f"n_recording_prefixes must be >= 1, got {n_recording_prefixes}")
    if n_recording_prefixes > len(groups):
        raise ValueError(
            f"requested {n_recording_prefixes} recording prefixes but only {len(groups)} exist"
        )

    rng = np.random.RandomState(seed)
    # Rarest label first: it constrains the cover the most.
    label_counts = collections.Counter(labels.tolist())
    labels_by_rarity = [lbl for lbl, _ in reversed(label_counts.most_common())]

    prefix_labels = {
        key: set(labels[idxs].tolist()) for key, idxs in groups.items()
    }
    chosen: List[str] = []

    for label in labels_by_rarity:
        if any(label in prefix_labels[key] for key in chosen):
            continue
        holders = sorted(k for k in groups if label in prefix_labels[k])
        if not holders:
            continue
        if len(chosen) >= n_recording_prefixes:
            logger.warning(
                "n_recording_prefixes=%d too small to cover label %s; increase it "
                "to keep that class in the subset",
                n_recording_prefixes,
                label,
            )
            break
        chosen.append(str(rng.choice(holders)))

    # Fill remaining slots: most trials first, name as deterministic tiebreak.
    remaining = sorted(
        (k for k in groups if k not in chosen),
        key=lambda k: (-len(groups[k]), k),
    )
    chosen.extend(remaining[: max(0, n_recording_prefixes - len(chosen))])
    return sorted(chosen)


def subset_dataset(
    data: Dict[str, object],
    n_animals: int,
    seed: int,
) -> Tuple[Dict[str, object], List[int], List[str]]:
    """Return ``(subset, kept_indices, kept_recording_prefixes)``.

    The historical ``n_animals`` parameter counts recording prefixes, not
    verified animals. Source priors and their provenance are retained: OOF
    models fitted on the full parent can use labels outside this subset.
    """
    missing = [k for k in ("labels", "session_ids") if k not in data]
    if missing:
        raise KeyError(f"dataset missing required keys: {missing}")

    session_ids = list(data["session_ids"])  # type: ignore[arg-type]
    labels = np.asarray(data["labels"])
    n_total = len(session_ids)
    assert labels.shape[0] == n_total, (
        f"labels/session_ids length mismatch: {labels.shape[0]} vs {n_total}"
    )

    if "trial_specs" in data:
        specs = data["trial_specs"]
        if len(specs) != n_total:
            raise ValueError(f"trial_specs: expected {n_total} source trials, got {len(specs)}")
        trial_ids = data.get("trial_ids")
        if trial_ids is not None and len(trial_ids) != n_total:
            raise ValueError(f"trial_ids: expected {n_total} source trials, got {len(trial_ids)}")
        for i, spec in enumerate(specs):
            if not isinstance(spec, dict) or spec.get("session_id") != session_ids[i]:
                raise ValueError(f"trial_specs[{i}].session_id disagrees with session_ids")
            if trial_ids is not None and "trial_id" in spec and spec["trial_id"] != trial_ids[i]:
                raise ValueError(f"trial_specs[{i}].trial_id disagrees with trial_ids")

    groups = group_by_recording_prefix(session_ids)
    kept_prefixes = select_recording_prefixes(groups, labels, n_animals, seed)
    kept = sorted(idx for key in kept_prefixes for idx in groups[key])

    subset: Dict[str, object] = {}
    for key, value in data.items():
        if key in _PER_TRIAL_KEYS and len(value) != n_total:
            raise ValueError(f"{key}: expected {n_total} source trials, got {len(value)}")
        if key not in _PER_TRIAL_KEYS:
            subset[key] = value
            continue
        if isinstance(value, torch.Tensor):
            subset[key] = value[kept]
        elif isinstance(value, np.ndarray):
            subset[key] = value[kept]
        else:  # list of ragged per-trial arrays
            subset[key] = [value[i] for i in kept]

    if "labeling_eligibility" in data:
        # This ledger is not per sequence: keep every parent unavailable trial,
        # but only labels whose identity belongs to a selected sequence.
        selected = {
            (str(session_ids[i]), int(data["trial_ids"][i])) for i in kept
        }
        ledger = [
            row for row in data["labeling_eligibility"]
            if row["status"] != "labeled"
            or (str(row["session_id"]), int(row["trial_id"])) in selected
        ]
        labeled = [row for row in ledger if row["status"] == "labeled"]
        labeled_keys = {
            (str(row["session_id"]), int(row["trial_id"])) for row in labeled
        }
        if len(labeled) != len(kept) or labeled_keys != selected:
            raise ValueError("labeling_eligibility must identify every selected sequence")
        subset["labeling_eligibility"] = ledger
        subset["subset_labeling_accounting"] = {
            "scope": "selected_sequences_and_all_parent_unavailable",
            "n_sequences": len(kept),
            "n_parent_entries": len(data["labeling_eligibility"]),
            "n_excluded_labeled": len(data["labeling_eligibility"]) - len(ledger),
            "n_labeled": len(labeled),
            "n_unavailable": len(ledger) - len(labeled),
            "n_entries": len(ledger),
            # These aggregate audits remain parent-scoped, not subset counts.
            "parent_summary_keys": [
                key for key in ("labeling_funnel", "labeling_funnel_retention",
                                "labeling_threshold_sensitivity") if key in data
            ],
        }

    subset["animal_identity_status"] = prior_identity_status(
        data.get("mcmc_prior_provenance"), data.get("animal_identity_status")
    )

    # Legacy datasets predate the explicit condition schema. Derive it from
    # the physical channels rather than silently producing a subset that
    # cannot activate the routing-aux loss.
    if "stimulus_conditions" not in subset or "is_pure_wind" not in subset:
        conditions, pure_wind = derive_stimulus_metadata(
            subset["X_seqs"],  # type: ignore[arg-type]
            subset["lengths"],  # type: ignore[arg-type]
        )
        subset["stimulus_conditions"] = conditions
        subset["is_pure_wind"] = pure_wind

    for key in _PER_TRIAL_KEYS:
        if key in subset:
            got = len(subset[key])  # type: ignore[arg-type]
            assert got == len(kept), (
                f"{key}: expected {len(kept)} trials after subsetting, got {got}"
            )
    return subset, kept, kept_prefixes


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Carve a recording-prefix grouped subset; animal identities are unverified."
    )
    parser.add_argument("--input", required=True, help="Source .pt dataset.")
    parser.add_argument("--output", required=True, help="Destination .pt.")
    parser.add_argument(
        "--n_recording_prefixes", "--n_animals",
        dest="n_recording_prefixes",
        type=int,
        default=8,
        help="Whole recording prefixes (default: 8); --n_animals is a legacy alias, not an animal count.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Sampling seed (default: 42)."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(f"input dataset not found: {src}")

    data, source_sha256 = load_dataset_with_fingerprint(src, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("source dataset must be a dictionary")
    validate_dataset_provenance(data, src)
    subset, kept, kept_prefixes = subset_dataset(data, args.n_recording_prefixes, args.seed)
    subset["subset_source_sha256"] = source_sha256

    dst = Path(args.output)
    validate_dataset_provenance(subset, dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(subset, dst)

    labels_before = collections.Counter(np.asarray(data["labels"]).tolist())
    labels_after = collections.Counter(np.asarray(subset["labels"]).tolist())
    n_sessions = len({str(s) for s in subset["session_ids"]})  # type: ignore[arg-type]

    logger.info("input:  %s (%d trials)", src, len(data["session_ids"]))
    logger.info(
        "output: %s (%d trials, %d recording prefixes, %d sessions, %.1f MB)",
        dst,
        len(kept),
        len(kept_prefixes),
        n_sessions,
        dst.stat().st_size / 1e6,
    )
    logger.info("labels before: %s", dict(sorted(labels_before.items())))
    logger.info("labels after:  %s", dict(sorted(labels_after.items())))
    dropped = set(labels_before) - set(labels_after)
    if dropped:
        logger.warning(
            "labels %s absent from subset; raise --n_recording_prefixes", sorted(dropped)
        )
    logger.info("version: %s", subset.get("pipeline_semantics_version"))
    logger.info("prior provenance: %s; animal identity: %s; source SHA-256: %s",
                subset["mcmc_prior_provenance"], subset["animal_identity_status"], source_sha256)
    logger.info("Distinct recording prefixes do not verify independent animal identities")
    logger.warning("Retained parent-corpus OOF priors; nonnested validation remains a development diagnostic")
    for key in kept_prefixes:
        logger.info("  recording prefix %s", key)


if __name__ == "__main__":
    main()
