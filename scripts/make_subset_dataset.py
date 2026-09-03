"""Carve a small, grouping-safe subset out of a processed NSMoR dataset.

Full corpora (1440 trials x 2400 frames) are too slow for iteration on a
single consumer GPU.  This tool produces a drop-in ``.pt`` with the same
schema so ``train.py --dataset`` / ``pytest`` can use it unchanged.

Two invariants the subset must not break:

1. **Group integrity.**  Sampling is done over *animals*, never over
   trials.  ``session_ids`` look like
   ``0.513cricket_001_20260707_193143_session_2`` — the ``_session_N``
   suffix splits one recording of one animal into blocks, so two
   sessions sharing the prefix are the *same* animal.  Sampling whole
   animals keeps a session-grouped split from leaking that animal
   across train/val.
2. **Label coverage.**  Escape/no-response classes are heavily
   imbalanced (label 1 is ~3% of the corpus).  Animals are picked by a
   greedy coverage pass so every label present upstream survives, then
   padded up to ``--n_animals`` by descending trial count.

Usage::

    python scripts/make_subset_dataset.py \\
        --input data/processed/nsmor_dataset_3cond_v2.pt \\
        --output data/processed/nsmor_subset_small.pt \\
        --n_animals 8
"""

from __future__ import annotations

import argparse
import collections
import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ``..._session_1`` / ``..._session_12`` -> animal-recording prefix.
_SESSION_SUFFIX = re.compile(r"_session_\d+$")

# Keys carrying one entry per trial; sliced by the sampled index set.
_PER_TRIAL_KEYS: Tuple[str, ...] = (
    "X_seqs",
    "Y_seqs",
    "labels",
    "lengths",
    "mcmc_priors",
    "session_ids",
    "stimulus_conditions",
    "is_pure_wind",
)


def derive_stimulus_metadata(
    x_seqs: Sequence[np.ndarray],
    lengths: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive auditable condition metadata from physical input channels.

    This mirrors ``scripts.prepare_data.classify_stimulus_condition`` and
    makes legacy processed artifacts usable by the routing-aux path without
    guessing from behavioural labels or session names.

    Args:
        x_seqs: Per-trial feature arrays with visual angle at index 0 and
            wind state at index 1.
        lengths: Valid (unpadded) frame count for each trial.

    Returns:
        ``(stimulus_conditions, is_pure_wind)`` arrays aligned 1:1 with
        ``x_seqs``.
    """
    if len(x_seqs) != len(lengths):
        raise ValueError(
            f"x_seqs/lengths mismatch: {len(x_seqs)} != {len(lengths)}"
        )

    conditions: List[str] = []
    for index, (x_seq, length) in enumerate(zip(x_seqs, lengths)):
        valid_length = int(length)
        if valid_length < 1 or valid_length > len(x_seq):
            raise ValueError(
                f"trial {index}: invalid length {valid_length} for "
                f"sequence length {len(x_seq)}"
            )
        physical = np.asarray(x_seq)[:valid_length, :2]
        has_visual = bool(np.any(np.abs(physical[:, 0]) > 0.0))
        has_wind = bool(np.any(np.abs(physical[:, 1]) > 0.0))
        if has_visual and has_wind:
            conditions.append("multisensory")
        elif has_visual:
            conditions.append("visual_only")
        elif has_wind:
            conditions.append("wind_only")
        else:
            conditions.append("no_stimulus")

    stimulus_conditions = np.asarray(conditions, dtype=object)
    is_pure_wind = stimulus_conditions == "wind_only"
    return stimulus_conditions, is_pure_wind


def animal_of(session_id: str) -> str:
    """Strip the ``_session_N`` block suffix to get the animal key."""
    return _SESSION_SUFFIX.sub("", str(session_id))


def group_by_animal(session_ids: Sequence[str]) -> Dict[str, List[int]]:
    """Map animal key -> trial indices belonging to that animal."""
    groups: Dict[str, List[int]] = collections.defaultdict(list)
    for idx, sid in enumerate(session_ids):
        groups[animal_of(sid)].append(idx)
    return dict(groups)


def select_animals(
    groups: Dict[str, List[int]],
    labels: np.ndarray,
    n_animals: int,
    seed: int,
) -> List[str]:
    """Pick ``n_animals`` keys that cover every label in ``labels``.

    Greedy set-cover on the rarest label first, so minority classes are
    not sampled away; remaining slots go to the animals with the most
    trials.  Deterministic given ``seed``.
    """
    if n_animals < 1:
        raise ValueError(f"n_animals must be >= 1, got {n_animals}")
    if n_animals > len(groups):
        raise ValueError(
            f"requested {n_animals} animals but only {len(groups)} exist"
        )

    rng = np.random.RandomState(seed)
    # Rarest label first: it constrains the cover the most.
    label_counts = collections.Counter(labels.tolist())
    labels_by_rarity = [lbl for lbl, _ in reversed(label_counts.most_common())]

    animal_labels = {
        key: set(labels[idxs].tolist()) for key, idxs in groups.items()
    }
    chosen: List[str] = []

    for label in labels_by_rarity:
        if any(label in animal_labels[key] for key in chosen):
            continue
        holders = sorted(k for k in groups if label in animal_labels[k])
        if not holders:
            continue
        if len(chosen) >= n_animals:
            logger.warning(
                "n_animals=%d too small to cover label %s; increase it "
                "to keep that class in the subset",
                n_animals,
                label,
            )
            break
        chosen.append(str(rng.choice(holders)))

    # Fill remaining slots: most trials first, name as deterministic tiebreak.
    remaining = sorted(
        (k for k in groups if k not in chosen),
        key=lambda k: (-len(groups[k]), k),
    )
    chosen.extend(remaining[: max(0, n_animals - len(chosen))])
    return sorted(chosen)


def subset_dataset(
    data: Dict[str, object],
    n_animals: int,
    seed: int,
) -> Tuple[Dict[str, object], List[int], List[str]]:
    """Return ``(subset, kept_indices, kept_animals)``.

    Non per-trial keys (``feature_config``,
    ``pipeline_semantics_version``, ...) are copied verbatim so the
    version guard in ``nsmor/config.py`` still matches.
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

    groups = group_by_animal(session_ids)
    kept_animals = select_animals(groups, labels, n_animals, seed)
    kept = sorted(idx for key in kept_animals for idx in groups[key])

    subset: Dict[str, object] = {}
    for key, value in data.items():
        if key not in _PER_TRIAL_KEYS:
            subset[key] = value
            continue
        if isinstance(value, torch.Tensor):
            subset[key] = value[kept]
        elif isinstance(value, np.ndarray):
            subset[key] = value[kept]
        else:  # list of ragged per-trial arrays
            subset[key] = [value[i] for i in kept]

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
    return subset, kept, kept_animals


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Carve a grouping-safe subset out of a processed dataset."
    )
    parser.add_argument("--input", required=True, help="Source .pt dataset.")
    parser.add_argument("--output", required=True, help="Destination .pt.")
    parser.add_argument(
        "--n_animals",
        type=int,
        default=8,
        help="Number of whole animals to keep (default: 8).",
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

    data = torch.load(src, map_location="cpu", weights_only=False)
    subset, kept, kept_animals = subset_dataset(data, args.n_animals, args.seed)

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(subset, dst)

    labels_before = collections.Counter(np.asarray(data["labels"]).tolist())
    labels_after = collections.Counter(np.asarray(subset["labels"]).tolist())
    n_sessions = len({str(s) for s in subset["session_ids"]})  # type: ignore[arg-type]

    logger.info("input:  %s (%d trials)", src, len(data["session_ids"]))
    logger.info(
        "output: %s (%d trials, %d animals, %d sessions, %.1f MB)",
        dst,
        len(kept),
        len(kept_animals),
        n_sessions,
        dst.stat().st_size / 1e6,
    )
    logger.info("labels before: %s", dict(sorted(labels_before.items())))
    logger.info("labels after:  %s", dict(sorted(labels_after.items())))
    dropped = set(labels_before) - set(labels_after)
    if dropped:
        logger.warning(
            "labels %s absent from subset; raise --n_animals", sorted(dropped)
        )
    logger.info("version: %s", subset.get("pipeline_semantics_version"))
    for key in kept_animals:
        logger.info("  animal %s", key)


if __name__ == "__main__":
    main()
