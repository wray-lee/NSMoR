"""Exploratory, read-only supplement to the frozen PHASE-2 JSON scorer."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import posixpath
import shlex
from pathlib import Path
from statistics import mean

ARMS = ("R0", "R1", "R2", "R3")
COMPARATORS = ("persistence", "zero", "R0")
SCOPES = ("aligned_pooled", "escape", "h5", "onset")
SEEDS = (42, 43, 44)
HORIZONS = ("h5", "h25", "h125")
ALL_SCOPES = SCOPES + ("h25", "h125", "h5_escape", "h25_escape", "h125_escape")
SLOTS = tuple(f"{a}|{c}|{s}" for a in ARMS for c in COMPARATORS for s in SCOPES)
DELTA = "candidate_minus_comparator_mse"
MSE_UNIT = "(cm/s)^2"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _path_key(value):
    return posixpath.normpath(value.replace("\\", "/")).casefold()


def prefix_summary(prefix_means):
    """Equal-prefix estimand; deletions hold the fitted model fixed."""
    if prefix_means is None or prefix_means == {}:
        return {"delta_units": MSE_UNIT, "available": False, "reason": "prefix_means_missing" if prefix_means is None
                else "no_prefix_support", "n": None if prefix_means is None else 0,
                "equal_prefix_mean_delta": None, "deletion": {"available": False,
                "reason": "fewer_than_two_prefixes", "estimates": {}, "range": None,
                "sign_stable": None}}
    _require(isinstance(prefix_means, dict), "prefix_means must be a mapping")
    _require(all(isinstance(k, str) and k.strip() and _finite(v)
                 for k, v in prefix_means.items()), "prefix_means must contain finite numeric deltas")
    values = list(prefix_means.values())
    n, point = len(values), float(mean(values))
    estimates = {k: float(mean(v for p, v in prefix_means.items() if p != k))
                 for k in prefix_means} if n >= 2 else {}
    sign = lambda v: (v > 0) - (v < 0)
    negative, zero, positive = (sum(v < 0 for v in values), sum(v == 0 for v in values),
                                sum(v > 0 for v in values))
    return {"delta_units": MSE_UNIT, "available": True, "reason": None, "n": n, "equal_prefix_mean_delta": point,
            "negative": negative, "zero": zero, "positive": positive,
            "prefix_wins": negative, "prefix_win_fraction": negative / n,
            "variance_status": "undefined_fewer_than_two_prefixes" if n < 2 else
            "zero_delta_variance" if len(set(values)) == 1 else "nonzero",
            "deletion": {"available": n >= 2, "reason": None if n >= 2 else "fewer_than_two_prefixes",
                         "estimates": estimates,
                         "range": [min(estimates.values()), max(estimates.values())] if estimates else None,
                         "sign_stable": all(sign(v) == sign(point) for v in estimates.values())
                         if estimates else None}}


def validate_family(data):
    """Validate recorded structure/lineage without loading checkpoints or data."""
    for key, expected in (("candidates", ARMS), ("comparators", COMPARATORS), ("scopes", SCOPES)):
        _require(data.get(key) == list(expected), f"incorrect frozen {key}")
    family = data["declared_family"]
    holm = family["holm_bonferroni"]
    _require(all(x == 48 for x in (data.get("declared_family_size"), family.get("family_size"),
                                    holm.get("family_size"))), "family must have exactly 48 slots")
    _require(set(family["slots"]) == set(SLOTS) == set(holm["slots"]), "incorrect frozen 48-slot keys")
    _require(data.get("protocol") == "docs/realdata-phase2-protocol-20261010.md", "incorrect frozen protocol")
    exchangeability = data.get("exchangeability_asserted")
    _require(type(exchangeability) is bool and family.get("exchangeability_asserted") is exchangeability,
             "inconsistent frozen exchangeability availability")
    valid = 0
    for slot in SLOTS:
        record, adjusted = family["slots"][slot], holm["slots"][slot]
        available = record.get("available")
        _require(type(available) is bool and adjusted.get("available") is available,
                 f"{slot}: inconsistent frozen availability")
        for key in ("raw_p", "adjusted_p"):
            p = adjusted.get(key)
            _require((_finite(p) and 0 <= p <= 1) if available else p is None,
                     f"{slot}: invalid frozen {key}")
        _require(type(adjusted.get("significant")) is bool if available else
                 adjusted.get("significant") is None, f"{slot}: invalid significance availability")
        if available:
            _require(exchangeability, f"{slot}: available test requires both exchangeability flags true")
            _require(adjusted["adjusted_p"] >= adjusted["raw_p"], f"{slot}: adjusted p below raw p")
            _require(adjusted["significant"] == (adjusted["adjusted_p"] <= 0.05),
                     f"{slot}: significant must agree with adjusted p <= 0.05")
        if "sign_flip" in record:
            _require(record["sign_flip"].get("p_value") == adjusted["raw_p"], f"{slot}: raw p mismatch")
        if slot.startswith("R0|R0|"):
            _require(not available and record.get("reason") == "self_comparison", f"{slot}: invalid self slot")
        if "prefix_means" in record:
            summary = prefix_summary(record["prefix_means"])
            _require(record.get("delta_convention") == DELTA, f"{slot}: incorrect delta convention")
            if summary["available"]:
                _require(record["counts"]["prefixes"] == summary["n"], f"{slot}: prefix count mismatch")
        valid += available
    _require(holm.get("valid_test_count") == valid and holm.get("unavailable_slot_count") == 48 - valid,
             "incorrect frozen availability counts")
    tokens = shlex.split(data.get("scorer_command", ""))
    _require(any(Path(t).name == "evaluate_realdata_phase2.py" for t in tokens), "missing scorer command")
    _require(("--exchangeability_asserted" in tokens) is exchangeability, "scorer command exchangeability mismatch")
    lineage = {}
    paths, hashes, vector_lengths = [], [], {}
    for name in ("dataset", "nested_prior") + tuple(a.lower() for a in ARMS):
        flag = f"--{name}"
        _require(tokens.count(flag) == 1, f"missing/ambiguous input lineage {flag}")
        start = tokens.index(flag) + 1
        count = 1 if name in ("dataset", "nested_prior") else 3
        values = tokens[start:start + count]
        _require(len(values) == count and all(v and not v.startswith("--") for v in values),
                 f"invalid input lineage {flag}")
        lineage[name] = values[0] if count == 1 else values
    _require(set(data["arms"]) == set(ARMS), "incorrect checkpoint arms")
    for arm in ARMS:
        entry = data["arms"][arm]
        checkpoints, digests = entry["checkpoints"], entry["sha256"]
        _require(len(checkpoints) == len(digests) == 3 and
                 all(isinstance(p, str) and p.strip() for p in checkpoints), f"{arm}: three checkpoint paths required")
        _require(all(isinstance(h, str) and len(h) == 64 and
                     all(c in "0123456789abcdefABCDEF" for c in h) for h in digests), f"{arm}: invalid SHA256")
        _require(list(map(_path_key, checkpoints)) == list(map(_path_key, lineage[arm.lower()])),
                 f"{arm}: checkpoint command lineage mismatch")
        paths.extend(map(_path_key, checkpoints))
        hashes.extend(h.lower() for h in digests)
        meta = data["h5_input_protocol"][arm]
        _require(meta.get("zero_input_channels") == ([2, 3] if arm == "R2" else []) and
                 meta.get("rollout_regime") == "feedback_then_arm_transform", f"{arm}: incorrect input protocol")
        for scope in ALL_SCOPES:
            seed_record = data["seed_level_summary"][arm].get(scope)
            vectors = data["seed_trial_mse_vectors"][arm].get(scope)
            _require(scope not in SCOPES or (seed_record is not None and vectors is not None),
                     f"{arm}/{scope}: missing primary seed means/vectors")
            if vectors is None:
                continue
            rows = vectors["per_seed_per_trial_mse"]
            _require(vectors.get("seeds") == list(SEEDS) and
                     all(type(seed) is int for seed in vectors["seeds"]) and
                     isinstance(rows, list) and len(rows) == 3 and all(isinstance(row, list) for row in rows),
                     f"{arm}/{scope}: requires seeds 42/43/44 and three vector rows")
            length = vector_lengths.setdefault(scope, len(rows[0]))
            _require(all(len(row) == length for row in rows), f"{arm}/{scope}: seed-vector row shape mismatch")
            _require(all(v is None or (_finite(v) and v >= 0) for row in rows for v in row),
                     f"{arm}/{scope}: vectors require nonnegative finite MSE or null")
            if seed_record is None:
                continue
            stored = seed_record["per_seed_trial_mean"]
            _require(type(seed_record.get("n_seeds")) is int and seed_record["n_seeds"] == 3 and
                     isinstance(stored, list) and len(stored) == 3, f"{arm}/{scope}: incorrect seed count")
            for value, row in zip(stored, rows):
                retained = [v for v in row if v is not None]
                _require((_finite(value) and math.isclose(value, mean(retained), rel_tol=1e-12, abs_tol=1e-12))
                         if retained else value is None or (type(value) is float and math.isnan(value)),
                         f"{arm}/{scope}: stored per_seed_trial_mean does not match retained-vector nanmean")
    _require(len(set(paths)) == len(set(hashes)) == 12, "duplicate checkpoint path or SHA256")
    return lineage


def build_report(data):
    lineage = validate_family(data)
    family, skills, seeds = {}, {}, {}
    for slot in SLOTS:
        frozen = data["declared_family"]["slots"][slot]
        family[slot] = {"frozen": frozen, "adjustment": data["declared_family"]["holm_bonferroni"]["slots"][slot],
                        **prefix_summary(frozen.get("prefix_means"))}
    for arm in ARMS:
        skills[arm], seeds[arm] = {}, {}
        for scope in ALL_SCOPES:
            entry = data["primary_skill_vs_persistence"][arm].get(scope)
            _require(entry is not None or scope not in SCOPES, f"{arm}/{scope}: missing primary skill")
            entry = entry or {}
            available = all(_finite(entry.get(k)) for k in ("skill_vs_persistence", "mse_model", "mse_persistence"))
            if available:
                _require(entry["mse_model"] >= 0 and entry["mse_persistence"] > 0 and
                         math.isclose(entry["skill_vs_persistence"], 1 - entry["mse_model"] / entry["mse_persistence"],
                                      rel_tol=1e-12, abs_tol=1e-12), f"{arm}/{scope}: inconsistent frozen skill")
            skills[arm][scope] = {"skill_units": "dimensionless", "mse_units": MSE_UNIT, "available": available,
                                 "reason": entry.get("reason") or (None if available else "missing_or_nonfinite_frozen_skill"),
                                 "frozen": entry,
                                 "comparator": "persistence_held_at_rollout_start" if scope.startswith("h") else
                                 "persistence_previous_frame"}
            seed_entry = data["seed_level_summary"][arm].get(scope, {})
            means = seed_entry.get("per_seed_trial_mean") if data["seed_trial_mse_vectors"][arm].get(scope) is not None else None
            if means is not None:
                _require(seed_entry.get("n_seeds") == 3 and len(means) == 3, f"{arm}/{scope}: incorrect seed count")
            seeds[arm][scope] = {"mse_units": MSE_UNIT, "seeds": list(SEEDS), "n_seeds": 3,
                                 "per_seed_trial_mean": [v if _finite(v) else None for v in means] if means else None,
                                 "reason": None if means and all(map(_finite, means)) else "missing_or_nonfinite_seed_means"}
    for scope in ALL_SCOPES:
        comparators = [skills[a][scope]["frozen"]["mse_persistence"] for a in ARMS if skills[a][scope]["available"]]
        _require(not comparators or all(math.isclose(p, comparators[0], rel_tol=1e-12, abs_tol=1e-12)
                                       for p in comparators), f"{scope}: inconsistent persistence comparators across arms")
    return {"prefix_robustness": family, "frozen_skill_vs_persistence": skills,
            "seed_optimisation_variability": seeds,
            "provenance": {"exploratory": True, "animal_identity": "unverified",
                           "animal_independence_from_prefixes": False, "validation_guided_selection": True,
                           "no_new_testing": True, "new_p_values": False, "new_family_correction": False,
                           "delta_convention": DELTA, "prefix_wins": "negative_delta",
                           "units": {"prefix_delta": MSE_UNIT, "per_seed_trial_mean": MSE_UNIT, "skill": "dimensionless"},
                           "prefix_estimand": "equal_weight_mean_of_supplied_recording_prefix_means",
                           "deletion_interpretation": "point-estimate sensitivity of a fixed fitted model; no cross validation or retraining",
                           "bootstrap_ci": "not_computed", "standardized_prefix_effect": "not_computed; undefined for zero delta variance",
                           "seed_interpretation": "optimisation variability: stored unweighted per-trial MSE means; not pooled MSE or biological n or 12 replicates",
                           "recorded_input_lineage": lineage, "input_lineage_verification": "recorded paths/checkpoint hashes only; no checkpoint rehash or corpus identity assertion; dataset/nested-prior content SHA unavailable in frozen JSON",
                           "frozen_protocol": data["protocol"], "frozen_scorer_command": data["scorer_command"],
                           "frozen_exchangeability_asserted": data["exchangeability_asserted"],
                           "frozen_checkpoints": data["arms"], "h5_input_protocol": data["h5_input_protocol"]}}


def _csv(path, fields, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_csvs(report, output):
    rows, details = [], []
    for slot, record in report["prefix_robustness"].items():
        frozen, adjustment, deletion = record["frozen"], record["adjustment"], record["deletion"]
        row = {"slot": slot, "delta_units": MSE_UNIT, "cohens_dz_units": "dimensionless", "frozen_available": frozen["available"], "frozen_reason": frozen.get("reason"),
               "adjusted_available": adjustment["available"], "raw_p": adjustment["raw_p"],
               "adjusted_p": adjustment["adjusted_p"], "significant": adjustment["significant"],
               "frozen_trial_mean_delta": frozen.get("mean_delta"), "frozen_cohens_dz": frozen.get("cohens_dz", {}).get("value"),
               "frozen_dz_reason": frozen.get("cohens_dz", {}).get("reason"), "frozen_counts": json.dumps(frozen.get("counts")),
               **{k: record.get(k) for k in ("available", "reason", "equal_prefix_mean_delta", "n", "negative", "zero", "positive",
                                             "prefix_wins", "prefix_win_fraction", "variance_status")},
               "deletion_available": deletion["available"], "deletion_reason": deletion["reason"],
               "deletion_min": deletion["range"][0] if deletion["range"] else None,
               "deletion_max": deletion["range"][1] if deletion["range"] else None, "deletion_sign_stable": deletion["sign_stable"]}
        rows.append(row)
        for prefix, delta in (frozen.get("prefix_means") or {}).items():
            details.append({"slot": slot, "delta_units": MSE_UNIT, "recording_prefix": prefix, "delta": delta, "prefix_win": delta < 0,
                            "deletion_estimate": deletion["estimates"].get(prefix)})
    _csv(output / "family_summary.csv", list(rows[0]), rows)
    _csv(output / "prefix_detail.csv", ["slot", "delta_units", "recording_prefix", "delta", "prefix_win", "deletion_estimate"], details)
    skill_rows, seed_rows = [], []
    for arm in ARMS:
        for scope in ALL_SCOPES:
            skill = report["frozen_skill_vs_persistence"][arm][scope]
            skill_rows.append({"arm": arm, "scope": scope, "skill_units": "dimensionless", "mse_units": MSE_UNIT, "available": skill["available"], "reason": skill["reason"],
                               "comparator": skill["comparator"], **{k: skill["frozen"].get(k) if skill["available"] else None
                               for k in ("skill_vs_persistence", "mse_model", "mse_persistence")}})
            seed = report["seed_optimisation_variability"][arm][scope]
            for i, number in enumerate(SEEDS):
                value = seed["per_seed_trial_mean"][i] if seed["per_seed_trial_mean"] else None
                seed_rows.append({"arm": arm, "scope": scope, "mse_units": MSE_UNIT, "seed": number, "per_seed_trial_mean": value,
                                  "available": value is not None, "reason": None if value is not None else seed["reason"]})
    _csv(output / "skill_summary.csv", list(skill_rows[0]), skill_rows)
    _csv(output / "seed_optimisation_variability.csv", list(seed_rows[0]), seed_rows)


def _write_figures(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name, caption):
        fig.text(0.5, 0.01, caption, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        try:
            for extension in ("pdf", "svg", "png"):
                with (output / f"{name}.{extension}").open("xb") as handle:
                    fig.savefig(handle, format=extension, dpi=180)
        finally:
            plt.close(fig)

    def na(ax, reason, y=0.02):
        ax.text(0.02, y, f"NA: {reason}", transform=ax.transAxes, fontsize=6, va="bottom", wrap=True)

    colors = ("#225ea8", "#238b45", "#d95f0e", "#88419d")
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), squeeze=False)
    for col, scope in enumerate(SCOPES):
        top, bottom = axes[:, col]
        top.set_title(scope)
        for i, arm in enumerate(ARMS):
            skill = report["frozen_skill_vs_persistence"][arm][scope]
            if skill["available"]:
                top.scatter(i, skill["frozen"]["skill_vs_persistence"], color=colors[i])
            else:
                na(top, f"{arm} {skill['reason']}", 0.02 + i * 0.08)
            family = report["prefix_robustness"][f"{arm}|persistence|{scope}"]["frozen"]
            if not family["available"]:
                na(top, f"frozen test {arm}: {family.get('reason')}", 0.02 + i * 0.08)
            seed = report["seed_optimisation_variability"][arm][scope]
            for j, number in enumerate(SEEDS):
                value = seed["per_seed_trial_mean"][j]
                if value is not None:
                    bottom.scatter(i + (j - 1) * 0.16, value, color=colors[i], marker=("o", "s", "^")[j],
                                   label=f"seed {number}" if col == 0 and i == 0 else None)
                else:
                    na(bottom, f"{arm}/seed {number} {seed['reason']}", 0.02 + i * 0.08)
        for ax in (top, bottom):
            ax.set_xticks(range(4), ARMS)
        top.axhline(0, color="gray", linewidth=0.7)
        top.set_ylabel("Frozen pooled skill (dimensionless)")
        bottom.set_ylabel("Stored per_seed_trial_mean ((cm/s)^2)")
    axes[1, 0].legend(fontsize=7)
    save(fig, "primary_skill", "Exploratory frozen pooled skill (top); three seeds show optimisation variability (bottom), not biological n or pooled MSE.")

    fig, axes = plt.subplots(4, 4, figsize=(15, 12), squeeze=False)
    for row, scope in enumerate(SCOPES):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            record = report["prefix_robustness"][f"{arm}|persistence|{scope}"]
            ax.set_title(f"{arm} / {scope}", fontsize=9)
            ax.axhline(0, color="gray", linewidth=0.7)
            if record["available"]:
                values = record["frozen"]["prefix_means"]
                ax.scatter(range(len(values)), list(values.values()), color=colors[col], s=20, label="prefix delta")
                ax.axhline(record["equal_prefix_mean_delta"], color="black", linestyle="--", label="equal-prefix mean")
                deletion = record["deletion"]
                if deletion["available"]:
                    ax.axhspan(*deletion["range"], color=colors[col], alpha=0.18, label="deletion range (not CI)")
                else:
                    na(ax, f"deletion {deletion['reason']}", 0.12)
                ax.set_xticks(range(len(values)), list(values), rotation=70, fontsize=6)
            else:
                na(ax, record["reason"], 0.12)
            if not record["frozen"]["available"]:
                na(ax, f"frozen test: {record['frozen'].get('reason')}")
            if col == 0:
                ax.set_ylabel("Candidate minus persistence MSE ((cm/s)^2)")
    axes[0, 0].legend(fontsize=6)
    save(fig, "prefix_robustness", "Exploratory recording-prefix description; negative = win. Fixed-model deletion sensitivity, not cross validation/retraining. Animal identity unverified.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    for col, suffix in enumerate(("", "_escape")):
        ax = axes[0, col]
        ax.set_title("Full rollout" if not suffix else "Escape frames")
        for i, arm in enumerate(ARMS):
            points = []
            for horizon in HORIZONS:
                entry = report["frozen_skill_vs_persistence"][arm][horizon + suffix]
                points.append(entry["frozen"]["skill_vs_persistence"] if entry["available"] else float("nan"))
                if not entry["available"]:
                    na(ax, f"{arm}/{horizon}: {entry['reason']}", 0.02 + i * 0.06)
            ax.plot(range(3), points, marker="o", color=colors[i], label=arm)
        ax.set_xticks(range(3), (5, 25, 125))
        ax.set_xlabel("Rollout horizon (frames)")
        ax.set_ylabel("Frozen pooled skill (dimensionless)")
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.legend()
    save(fig, "horizon_skill", "Exploratory frozen skill; persistence held at rollout start in both panels. Missing secondary metrics remain NA. No new statistical testing.")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return None if type(value) is float and not math.isfinite(value) else value


def write_supplement(input_path, output_dir, command):
    input_path, output = Path(input_path).resolve(), Path(output_dir)
    raw = input_path.read_bytes()
    report = build_report(json.loads(raw, object_pairs_hook=_unique_keys))
    repo = Path(__file__).resolve().parents[2]
    report["provenance"].update({"input_path": str(input_path), "input_sha256": hashlib.sha256(raw).hexdigest(),
                                 "code_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__).resolve(), repo / "scripts/analyze_publication_robustness.py")},
                                 "local_frozen_scorer_sha256": hashlib.sha256((repo / "scripts/evaluate_realdata_phase2.py").read_bytes()).hexdigest(),
                                 "command": list(command), "compute": "CPU1; no model/data loading or training"})
    output.mkdir(parents=True, exist_ok=False)
    report = _json_safe(report)
    for filename, content in (("publication_robustness.json", report), ("provenance.json", report["provenance"])):
        with (output / filename).open("x", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2, allow_nan=False)
            handle.write("\n")
    _write_csvs(report, output)
    _write_figures(report, output)
    return output
