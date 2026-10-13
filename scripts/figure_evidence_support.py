#!/usr/bin/env python3
"""Export descriptive evidence support from two small, read-only artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nsmor.pipeline.nested_prior import load_artifact_bytes  # noqa: E402

CONDITION_NAMES = {
    "visual_only": "Visual only",
    "wind_only": "Wind only",
    "multisensory": "Multisensory",
}


def _fields(data, keys, name):
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a mapping")
    missing = set(keys) - data.keys()
    if missing:
        raise ValueError(f"{name} missing required keys: {sorted(missing)}")


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer count")
    return int(value)


def build_support(summary, prior):
    """Validate aggregate alignment and derive support from the exact split."""
    _fields(summary, ("meta", "prefix_condition", "eligibility"), "summary")
    _fields(prior, (
        "group_keys", "recording_prefix_keys", "train_indices", "val_indices",
        "train_recording_prefixes", "val_recording_prefixes", "source_fingerprint",
        "split_seed", "is_nested_cv", "animal_identity_status",
    ), "prior")
    meta, eligibility, condition_map = (
        summary["meta"], summary["eligibility"], summary["prefix_condition"]
    )
    _fields(meta, (
        "n_trials", "n_recording_prefixes", "n_sessions", "counts_by_condition",
        "prefix_condition_pure", "animal_identity_status",
    ), "meta")
    _fields(eligibility, (
        "by_condition", "window_ms", "threshold_cms", "sensitivity_threshold_cms",
    ), "eligibility")
    n = _count(meta["n_trials"], "n_trials")
    if n == 0:
        raise ValueError("n_trials must be positive")
    if meta["prefix_condition_pure"] is not True:
        raise ValueError("prefix_condition_pure must be true")
    identity = meta["animal_identity_status"]
    if (not isinstance(identity, str) or identity.split(" -> ", 1)[0] != "unverified"
            or prior["animal_identity_status"] != "unverified"):
        raise ValueError("animal_identity_status must be unverified in both sources")
    for flag in ("exploratory", "in_sample"):
        if flag in meta and meta[flag] is not True:
            raise ValueError(f"behavior {flag} conflicts with EXPLORATORY / IN-SAMPLE scope")
    fingerprint = prior["source_fingerprint"]
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint) is None:
        raise ValueError("source_fingerprint must be a 64-hex dataset SHA256")
    if prior["is_nested_cv"] is not True or _count(prior["split_seed"], "split_seed") != 42:
        raise ValueError("expected a nested split with split_seed 42")

    prefixes = np.asarray(prior["recording_prefix_keys"])
    groups = np.asarray(prior["group_keys"])
    if prefixes.shape != (n,) or groups.shape != (n,):
        raise ValueError("summary n_trials must match both prior key arrays")
    if not all(isinstance(p, str) and p.strip() for p in prefixes):
        raise ValueError("recording_prefix_keys must contain nonempty strings")
    if not np.array_equal(groups, prefixes):
        raise ValueError("group_keys conflict with recording_prefix_keys")
    prefix_counts = Counter(prefixes.tolist())
    if _count(meta["n_recording_prefixes"], "n_recording_prefixes") != len(prefix_counts):
        raise ValueError("summary n_recording_prefixes conflicts with prior keys")
    sessions = _count(meta["n_sessions"], "n_sessions")
    if not len(prefix_counts) <= sessions <= n:
        raise ValueError("reported n_sessions must lie between prefix and trial counts")
    _fields(condition_map, prefix_counts, "prefix_condition")
    if set(condition_map) != set(prefix_counts):
        raise ValueError("prefix_condition contains prefixes absent from prior")
    counts = meta["counts_by_condition"]
    _fields(counts, (), "counts_by_condition")
    if not counts or set(counts) - CONDITION_NAMES.keys():
        raise ValueError("counts_by_condition contains missing or unsupported conditions")
    if any(c not in counts for c in condition_map.values()):
        raise ValueError("prefix_condition contains an unsupported condition")
    conditions = [c for c in CONDITION_NAMES if c in counts]
    by_condition = eligibility["by_condition"]
    _fields(by_condition, conditions, "eligibility.by_condition")
    if set(by_condition) != set(counts) or set(condition_map.values()) != set(counts):
        raise ValueError("condition keys conflict between summary counts, map and eligibility")

    split_indices, split_prefixes = {}, {}
    for split, key in (("train", "train_indices"), ("validation", "val_indices")):
        indices = np.asarray(prior[key])
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or indices.size == 0:
            raise ValueError(f"{key} must be a nonempty one-dimensional integer array")
        if np.any(indices < 0) or np.any(indices >= n):
            raise ValueError(f"{key} has out-of-bounds indices")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{key} has duplicate indices")
        selected = prefixes[indices]
        assert selected.shape == indices.shape
        split_indices[split] = indices
        split_prefixes[split] = set(selected.tolist())
    train, val = split_indices["train"], split_indices["validation"]
    if np.intersect1d(train, val).size:
        raise ValueError("train/validation indices overlap")
    if train.size + val.size != n:
        raise ValueError("train/validation indices are not a complete partition")
    if split_prefixes["train"] & split_prefixes["validation"]:
        raise ValueError("a recording prefix straddles train/validation splits")
    for split, stem in (("train", "train"), ("validation", "val")):
        declared = np.asarray(prior[f"{stem}_recording_prefixes"])
        if (declared.ndim != 1 or any(not isinstance(p, str) for p in declared)
                or len(set(declared.tolist())) != declared.size
                or set(declared.tolist()) != split_prefixes[split]):
            raise ValueError(f"{stem}_recording_prefixes conflict with exact split indices")
        for key, expected in (
            (f"n_{stem}", split_indices[split].size),
            (f"n_{stem}_recording_prefixes", len(split_prefixes[split])),
        ):
            if key in prior and _count(prior[key], key) != expected:
                raise ValueError(f"prior {key} conflicts with exact split indices")

    for key, expected, meta_key in (
        ("window_ms", 250, "stationary_window_ms"),
        ("threshold_cms", 10, "stationary_threshold_cms"),
        ("sensitivity_threshold_cms", 5, "stationary_threshold_sensitivity_cms"),
    ):
        value = eligibility[key]
        if isinstance(value, bool) or value != expected:
            raise ValueError(f"eligibility {key} conflicts with existing D1/5 cm/s gates")
        if meta_key in meta and meta[meta_key] != expected:
            raise ValueError(f"meta {meta_key} conflicts with eligibility thresholds")

    prefix_rows = [
        {"prefix": p, "condition": condition_map[p],
         "split": "train" if p in split_prefixes["train"] else "validation",
         "n_trials": prefix_counts[p]}
        for p in sorted(prefix_counts)
    ]
    condition_rows = []
    required_counts = (
        "n", "n_eligible", "n_excluded", "primary_responders", "n_eligible_5cms",
        "n_excluded_5cms", "elig5_responders",
    )
    for condition in conditions:
        e = by_condition[condition]
        _fields(e, required_counts, f"eligibility.{condition}")
        c = {key: _count(e[key], f"{condition}.{key}") for key in required_counts}
        rows = [row for row in prefix_rows if row["condition"] == condition]
        observed = sum(row["n_trials"] for row in rows)
        if _count(counts[condition], f"counts_by_condition.{condition}") != observed or c["n"] != observed:
            raise ValueError(f"{condition} trial counts conflict with prior prefix mapping")
        if c["n_eligible"] + c["n_excluded"] != observed or c["n_eligible_5cms"] + c["n_excluded_5cms"] != observed:
            raise ValueError(f"{condition} eligibility denominators do not sum to n")
        # shortcut: summaries expose subset counts only; check membership if trial masks are added.
        if (c["primary_responders"] > c["n_eligible"]
                or c["n_eligible_5cms"] > c["n_eligible"]
                or c["elig5_responders"] > min(c["n_eligible_5cms"], c["primary_responders"])):
            raise ValueError(f"{condition} eligibility/responder subset counts are inconsistent")
        if "excluded_fraction" in e:
            fraction = e["excluded_fraction"]
            if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                    or not math.isclose(fraction, c["n_excluded"] / observed, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f"{condition} excluded_fraction conflicts with eligibility counts")
        for key, upper in (
            ("labeling_responders", c["n_eligible"]),
            ("labeling_responders_ungated", observed),
            ("all_trials_responders_s2", observed),
            ("legacy_window_responders_s3", observed),
        ):
            if key in e and _count(e[key], f"{condition}.{key}") > upper:
                raise ValueError(f"{condition}.{key} exceeds its denominator")
        if ("labeling_responders" in e and "labeling_responders_ungated" in e
                and e["labeling_responders"] > e["labeling_responders_ungated"]):
            raise ValueError(f"{condition} gated labeling responders exceed ungated responders")
        if "all_trials_responders_s2" in e and c["primary_responders"] > e["all_trials_responders_s2"]:
            raise ValueError(f"{condition} primary responders exceed all-trials responders")
        row = {"condition": condition, "n_trials": observed, "n_recording_prefixes": len(rows)}
        for split in ("train", "validation"):
            selected = [r for r in rows if r["split"] == split]
            row[f"{split}_trials"] = sum(r["n_trials"] for r in selected)
            row[f"{split}_recording_prefixes"] = len(selected)
        row.update({key: c[key] for key in required_counts if key != "n"})
        row["primary_nonresponders"] = c["n_eligible"] - c["primary_responders"]
        row["elig5_nonresponders"] = c["n_eligible_5cms"] - c["elig5_responders"]
        condition_rows.append(row)

    return {
        "scope": {
            "behavior": "EXPLORATORY / IN-SAMPLE",
            "model_split": "nested_outer_validation",
            "model_selection": "validation-guided model selection; no untouched test",
            "animal_identity_status": "unverified; recording prefix is not animal identity",
            "condition_interpretation": "condition confounded with recording prefix; no condition causality",
            "stored_labels": "post-stimulus categories, NOT prestimulus states",
            "statistics": "descriptive counts; no trial-independent binomial CI or inferential tests",
            "eligibility_validation": "aggregate denominators and subset inequalities; no per-trial masks",
        },
        "totals": {
            "n_trials": n, "n_recording_prefixes": len(prefix_counts),
            "n_sessions_reported_by_summary": sessions,
            "train_trials": int(train.size), "validation_trials": int(val.size),
            "train_recording_prefixes": len(split_prefixes["train"]),
            "validation_recording_prefixes": len(split_prefixes["validation"]),
        },
        "eligibility_gates": {
            "window_ms": eligibility["window_ms"],
            "d1_threshold_cms": eligibility["threshold_cms"],
            "existing_sensitivity_threshold_cms": eligibility["sensitivity_threshold_cms"],
            "status": "existing gates; thresholds not optimized",
        },
        "missing_validation_conditions": [r["condition"] for r in condition_rows if r["validation_trials"] == 0],
        "recording_prefix_support": prefix_rows,
        "condition_summary": condition_rows,
    }


def _grouped(ax, rows, fields, legend, denominators=None):
    x = np.arange(len(rows))
    assert x.shape == (len(rows),)
    maximum = 1
    for i, (field, name, color) in enumerate(zip(fields, legend, ("#2878B5", "#D97828"))):
        values = [r[field] for r in rows]
        maximum = max(maximum, *values)
        bars = ax.bar(x + (i - 0.5) * 0.35, values, width=0.35, label=name, color=color)
        labels = [f"{v}\nN={r[denominators[i]]}" if denominators else str(v) for v, r in zip(values, rows)]
        if field in ("validation_trials", "validation_recording_prefixes"):
            labels = [label if value else "0\nMISSING" for label, value in zip(labels, values)]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    ax.set_ylim(0, maximum * 1.42)
    ax.legend(loc="upper left", fontsize=9)


def _stacked(ax, rows, fields, legend):
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows), dtype=int)
    assert x.shape == bottom.shape == (len(rows),)
    for field, name, color in zip(fields, legend, ("#2878B5", "#CCCCCC")):
        values = np.asarray([r[field] for r in rows])
        assert values.shape == bottom.shape
        ax.bar(x, values, bottom=bottom, width=0.65, label=name, color=color)
        bottom = bottom + values
        assert bottom.shape == (len(rows),)
    for i, row in enumerate(rows):
        ax.annotate(f"{row[fields[0]]} / {row[fields[1]]}\nN={bottom[i]}",
                    (i, bottom[i]), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_ylim(0, max(1, int(bottom.max())) * 1.42)
    ax.legend(loc="upper left", fontsize=9)


def export_figure(report, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["condition_summary"]
    with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 10}):
        fig, axes = plt.subplots(3, 2, figsize=(14, 11))
        assert axes.shape == (3, 2)
        fig.subplots_adjust(top=0.87, bottom=0.14, hspace=0.65, wspace=0.30)
        fig.suptitle(
            "Evidence support from behavior summary and nested split artifacts\n"
            "Behavior: EXPLORATORY / IN-SAMPLE | Model split: nested_outer_validation\n"
            "Validation-guided model selection; no untouched test",
            fontsize=13, y=0.98,
        )
        split_legend = ("Train", "Validation")
        gate_legend = ("D1: |Y| < 10 cm/s", "Existing gate: |Y| < 5 cm/s")
        _grouped(axes[0, 0], rows, ("train_trials", "validation_trials"), split_legend)
        _grouped(axes[0, 1], rows, ("train_recording_prefixes", "validation_recording_prefixes"), split_legend)
        _stacked(axes[1, 0], rows, ("n_eligible", "n_excluded"), ("Eligible", "Excluded"))
        _stacked(axes[1, 1], rows, ("primary_responders", "primary_nonresponders"), ("Responders", "Nonresponders"))
        _grouped(axes[2, 0], rows, ("n_eligible", "n_eligible_5cms"), gate_legend, ("n_trials", "n_trials"))
        _grouped(axes[2, 1], rows, ("primary_responders", "elig5_responders"), gate_legend, ("n_eligible", "n_eligible_5cms"))
        titles = (
            "(A) Exact outer split: trials", "Exact outer split: recording prefixes",
            "(B) Observed D1 eligibility funnel", "Responders within D1-eligible trials",
            "(C) Existing eligibility gates: retained / observed", "Same response rule: responders / eligible",
        )
        ylabels = ("Trials", "Recording prefixes", "Observed trials", "D1-eligible trials", "Eligible trial counts", "Responder trial counts")
        for ax, title, ylabel in zip(axes.flat, titles, ylabels):
            ax.set_title(title, fontsize=11, pad=10)
            ax.set_ylabel(ylabel)
            ax.set_xticks(np.arange(len(rows)), [CONDITION_NAMES[r["condition"]] for r in rows])
            ax.spines[["top", "right"]].set_visible(False)
        fig.text(
            0.5, 0.025,
            "Counts only; stacked annotations follow legend order and N is the bar denominator.\n"
            "Existing gates use the same 250 ms pre-trigger window; no threshold optimization.\n"
            "Recording prefix is not animal identity; condition is confounded with prefix (no causal condition claim).\n"
            "Stored labels are post-stimulus categories, NOT prestimulus states; no trial-independent binomial CI.\n"
            "Behavior dataset binding is absent from the source summary; prior corpus fingerprint was not rechecked.",
            ha="center", fontsize=10,
        )
        try:
            for extension in ("pdf", "svg", "png"):
                fig.savefig(output_dir / f"figure_evidence_support.{extension}", dpi=180)
        finally:
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-summary", type=Path,
                        default=Path("results/behavior-20261010/behavior/behavior_summary.json"))
    parser.add_argument("--nested-prior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are refused")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(f"output directory already exists: {args.output_dir}")
        summary_bytes = args.behavior_summary.read_bytes()
        prior_bytes = args.nested_prior.read_bytes()
        summary = json.loads(summary_bytes)
        prior = load_artifact_bytes(prior_bytes, map_location="cpu")
        report = build_support(summary, prior)
        script_path = Path(__file__).resolve()
        report["provenance"] = {
            "source_artifacts": {
                name: {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}
                for name, path, payload in (
                    ("behavior_summary", args.behavior_summary, summary_bytes),
                    ("nested_prior", args.nested_prior, prior_bytes),
                    ("script", script_path, script_path.read_bytes()),
                )
            },
            "behavior_dataset_binding": "not_recorded_in_source_summary",
            "prior_source_fingerprint": prior["source_fingerprint"],
            "prior_source_fingerprint_status": "recorded in prior; corpus not rehashed or loaded",
            "split_seed": prior["split_seed"],
            "alignment": "summary counts and prefix mapping checked against exact prior keys/indices; corpus not loaded",
            "source_summary_animal_identity_status": summary["meta"]["animal_identity_status"],
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for filename, rows in (
            ("recording_prefix_support.csv", report["recording_prefix_support"]),
            ("condition_summary.csv", report["condition_summary"]),
        ):
            with (args.output_dir / filename).open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        with (args.output_dir / "evidence_support.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        export_figure(report, args.output_dir)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Exported evidence support to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
