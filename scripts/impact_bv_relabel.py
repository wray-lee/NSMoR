"""READ-ONLY impact of the visual-only label re-clock (ADR 0011, F4).

Recomputes the ``visual_only`` ground-truth labels from the STAGED session
CSVs under two clocks and counts the transitions:

* the current clock — the injected ``stimulus_onset`` at the looming
  onset (pipeline semantics <= 2.2), and
* the collision clock — ``stimulus_onset = looming_onset +
  lv_ratio_ms / tan(radians(init_deg/2))`` (pipeline semantics 2.3,
  ADR 0011).

The stored per-trial labels of the processed corpus are read only to
(i) restrict the count to the corpus's retained trials and (ii) verify
that the staged replay reproduces the stored labels under the OLD clock
(a self-check that the re-labelled deltas are attributable to the clock
change, not to a replay drift).

READ-ONLY GUARANTEE: every input (the staged CSVs and the processed
dataset) is opened read-only; the script writes exactly one file — the
``--output`` JSON — and verifies after the run that every input's size
and mtime are unchanged, recording that verdict in the output.

Usage::

    python scripts/impact_bv_relabel.py \\
        --staged_dir <dir with per-session subdirs> \\
        --dataset <processed nsmor_dataset.pt> \\
        --output <json path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsmor.pipeline.conditions import looming_ttc_ms  # noqa: E402
from nsmor.pipeline.events import parse_event_details  # noqa: E402
from nsmor.pipeline.io import (  # noqa: E402
    extract_trial_data,
    load_and_concat_sessions,
)
from nsmor.pipeline.labeling import assign_ground_truth_labels  # noqa: E402
from nsmor.pipeline.nested_prior import load_artifact_bytes  # noqa: E402

logger = logging.getLogger("impact_bv_relabel")


def _staged_session_pairs(
    staged_dir: Path,
) -> List[Tuple[str, Path, Path]]:
    """Deterministically pair staged kinematics/events CSVs (read-only).

    The staged layout is one directory per session holding
    ``<name>_kinematics.csv`` + ``<name>_events.csv``.  Missing or
    duplicated siblings fail closed rather than guessing a pairing.
    """
    kin_map: Dict[str, Path] = {}
    evt_map: Dict[str, Path] = {}
    for path in sorted(Path(staged_dir).rglob("*.csv")):
        name = path.name
        if name.endswith("_kinematics.csv"):
            key = name.removesuffix("_kinematics.csv")
            kin_map.setdefault(key, path)
        elif name.endswith("_events.csv"):
            key = name.removesuffix("_events.csv")
            evt_map.setdefault(key, path)
    missing = sorted(set(kin_map) - set(evt_map))
    orphans = sorted(set(evt_map) - set(kin_map))
    if missing or orphans:
        raise ValueError(
            f"staged pairs incomplete: missing events for {missing}, "
            f"orphan events for {orphans} (fail closed)"
        )
    return [(key, kin_map[key], evt_map[key]) for key in sorted(kin_map)]


def _condition_from_channels(trial: Dict[str, Any]) -> str:
    """Condition name from the trial's physical channels (read-only)."""
    has_visual = bool(np.any(np.abs(trial["visual_angle"]) > 0.0))
    has_wind = bool(np.any(np.abs(trial["wind_state"]) > 0.0))
    if has_visual and has_wind:
        return "multisensory"
    if has_visual:
        return "visual_only"
    if has_wind:
        return "wind_only"
    return "no_stimulus"


def _lv_ratio_ms(trial: Dict[str, Any]) -> Optional[float]:
    """Declared ``lv_ratio_ms`` from the trial_start details, if any."""
    for etype, evalue in zip(trial["event_types"], trial["event_values"]):
        if str(etype) == "trial_start":
            det = parse_event_details(evalue)
            raw = det.get("lv_ratio_ms")
            if raw is None:
                return None
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) and value > 0 else None
    return None


def _stimulus_onset_time(trial: Dict[str, Any]) -> Optional[float]:
    """First finite ``stimulus_onset`` event time, else None."""
    times = np.asarray(trial["event_times"], dtype=float)
    mask = np.asarray(trial["event_types"], dtype=object) == "stimulus_onset"
    if not mask.any():
        return None
    values = times[mask]
    values = values[np.isfinite(values)]
    return float(values.min()) if values.size else None


def _reclocked_trial(
    trial: Dict[str, Any], collision_ms: float
) -> Dict[str, Any]:
    """Copy of *trial* with every stimulus_onset time set to the collision."""
    new_trial = dict(trial)
    types = np.asarray(trial["event_types"], dtype=object).copy()
    times = np.asarray(trial["event_times"], dtype=float).copy()
    mask = types == "stimulus_onset"
    times[mask] = collision_ms
    new_trial["event_types"] = types
    new_trial["event_times"] = times
    return new_trial


def _label_of(trial: Dict[str, Any]) -> Optional[str]:
    """Label name under the trial's own stimulus_onset clock (None = none)."""
    labeled = assign_ground_truth_labels([trial])
    if not labeled:
        return None
    return labeled[0]["label"].name


def _input_fingerprint(paths: List[Path]) -> Dict[str, List[int]]:
    """Size + mtime_ns per input path (read-only change detector)."""
    out: Dict[str, List[int]] = {}
    for path in paths:
        stat = path.stat()
        out[str(path)] = [int(stat.st_size), int(stat.st_mtime_ns)]
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged_dir", type=Path, required=True,
                        help="staged session directory (per-session subdirs)")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="processed nsmor_dataset.pt (read-only)")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON output path (the ONLY file written)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the read-only impact count and write the JSON."""
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)

    pairs = _staged_session_pairs(args.staged_dir)
    logger.info("staged pairs: %d", len(pairs))

    input_paths: List[Path] = [args.dataset]
    for _, kin_path, evt_path in pairs:
        input_paths.extend([kin_path, evt_path])
    before = _input_fingerprint(input_paths)

    # Stored labels, restricted to the corpus's retained trials.
    dataset = load_artifact_bytes(args.dataset.read_bytes(), map_location="cpu")
    dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    stored: Dict[Tuple[str, int], int] = {}
    for sid, tid, label in zip(
        dataset["session_ids"], dataset["trial_ids"], dataset["labels"],
    ):
        stored[(str(sid), int(tid))] = int(label)
    logger.info("stored corpus trials: %d", len(stored))

    label_names = {0: "ESCAPE", 1: "PREWALK", 2: "PRE_ACTIVE", 3: "NO_RESPONSE"}
    transitions: Dict[str, int] = {}
    old_counts: Dict[str, int] = {}
    new_counts: Dict[str, int] = {}
    n_visual_trials = 0
    n_not_in_corpus = 0
    n_not_reclockable = 0
    n_unlabeled_old_clock = 0
    n_old_clock_matches_stored = 0
    n_old_clock_mismatches = 0
    mismatch_examples: List[Dict[str, Any]] = []

    for session_id, kin_path, evt_path in pairs:
        data = load_and_concat_sessions([kin_path], [evt_path])
        kin = data["kinematics"]
        evt = data["events"]
        for key in kin.groupby(["session_id", "trial_id"], sort=False).groups:
            sid, tid = str(key[0]), int(key[1])
            if (sid, tid) not in stored:
                n_not_in_corpus += 1
                continue
            evt_trial = evt.loc[
                (evt["session_id"] == sid) & (evt["trial_id"] == tid)
            ]
            trial = extract_trial_data(
                {"kinematics": kin, "events": evt_trial}, sid, tid,
            )
            if _condition_from_channels(trial) != "visual_only":
                continue
            n_visual_trials += 1
            old_label = _label_of(trial)
            stored_label = stored[(sid, tid)]
            if old_label is None:
                n_unlabeled_old_clock += 1
                continue
            old_counts[old_label] = old_counts.get(old_label, 0) + 1
            if label_names[stored_label] == old_label:
                n_old_clock_matches_stored += 1
            else:
                n_old_clock_mismatches += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append({
                        "session_id": sid, "trial_id": tid,
                        "stored": label_names[stored_label],
                        "replayed_old_clock": old_label,
                    })
            lv_ratio_ms = _lv_ratio_ms(trial)
            onset_ms = _stimulus_onset_time(trial)
            if lv_ratio_ms is None or onset_ms is None:
                n_not_reclockable += 1
                continue
            collision_ms = onset_ms + looming_ttc_ms(lv_ratio_ms)
            new_label = _label_of(_reclocked_trial(trial, collision_ms))
            if new_label is None:
                n_not_reclockable += 1
                continue
            new_counts[new_label] = new_counts.get(new_label, 0) + 1
            transition = f"{old_label}->{new_label}"
            transitions[transition] = transitions.get(transition, 0) + 1
        del data, kin, evt

    after = _input_fingerprint(input_paths)
    unchanged = before == after
    if not unchanged:
        raise RuntimeError(
            "READ-ONLY VIOLATION: an input's size/mtime changed during the "
            "run (fail closed)"
        )

    n_changes = sum(
        count for transition, count in transitions.items()
        if transition.split("->")[0] != transition.split("->")[1]
    )
    report: Dict[str, Any] = {
        "schema": "bv_relabel_impact_v1",
        "read_only": True,
        "inputs_unchanged": True,
        "dataset_sha256": dataset_sha256,
        "staged_dir": str(args.staged_dir),
        "dataset": str(args.dataset),
        "n_staged_session_pairs": len(pairs),
        "n_visual_only_trials_in_corpus": n_visual_trials,
        "n_visual_only_trials_kept_here": sum(old_counts.values()),
        "n_skipped_unlabeled_old_clock": n_unlabeled_old_clock,
        "n_skipped_not_reclockable": n_not_reclockable,
        "n_staged_trials_not_in_corpus": n_not_in_corpus,
        "old_clock_label_counts": dict(sorted(old_counts.items())),
        "new_clock_label_counts": dict(sorted(new_counts.items())),
        "transition_matrix": dict(sorted(transitions.items())),
        "n_label_changes": int(n_changes),
        "n_old_clock_matches_stored": n_old_clock_matches_stored,
        "n_old_clock_mismatches_stored": n_old_clock_mismatches,
        "old_clock_mismatch_examples": mismatch_examples,
        "collision_clock": (
            "stimulus_onset = looming_onset + lv_ratio_ms / "
            "tan(radians(init_deg/2)) with init_deg = 2.0 (ADR 0011); the "
            "same geometry reconstruct_visual_angle and the visual-only "
            "snapshot anchor use"
        ),
        "notes": [
            "Labels are recomputed on the SOURCE staged cadence with the "
            "real labeling pipeline (assign_ground_truth_labels), not on "
            "the resampled model grid.",
            "The old-clock replay must reproduce the stored corpus labels; "
            "any mismatch is reported in n_old_clock_mismatches_stored and "
            "limits the attribution of the deltas to the clock change.",
            "No corpus artifact is written; the only output is this JSON.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    logger.info("wrote %s", args.output)
    print(json.dumps({
        "n_visual_only_trials_in_corpus": n_visual_trials,
        "n_label_changes": int(n_changes),
        "old_clock_label_counts": report["old_clock_label_counts"],
        "new_clock_label_counts": report["new_clock_label_counts"],
        "n_old_clock_mismatches_stored": n_old_clock_mismatches,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
