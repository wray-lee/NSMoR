"""Small CPU fixtures for the frozen PHASE-2 publication postprocessor."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nsmor.analysis.publication_robustness import (
    ALL_SCOPES, ARMS, COMPARATORS, DELTA, SCOPES, SEEDS, SLOTS,
    build_report, prefix_summary, validate_family, write_supplement,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen():
    slots, adjusted, arms = {}, {}, {}
    for slot in SLOTS:
        self_comparison = slot.startswith("R0|R0|")
        slots[slot] = {"available": False, "reason": "self_comparison" if self_comparison else "fewer_than_six_prefixes"}
        if not self_comparison:
            slots[slot].update(delta_convention=DELTA, counts={"prefixes": 2, "trials": 9, "frames": 27},
                               mean_delta=4 / 9, cohens_dz={"value": 0.25, "reason": None},
                               prefix_means={"recordingA": -4.0, "recordingB": 1.0}, cluster_statistic=-1.5,
                               sign_flip={"p_value": None, "reason": "fewer_than_six_prefixes"})
        adjusted[slot] = {"available": False, "raw_p": None, "adjusted_p": None, "significant": None}
    command = ["python", "scripts/evaluate_realdata_phase2.py", "--dataset", "/recorded/data.pt",
               "--nested_prior", "/recorded/prior.pt"]
    for arm in ARMS:
        paths = [f"/recorded/{arm}/seed{s}/best.pth" for s in SEEDS]
        arms[arm] = {"checkpoints": paths, "sha256": [hashlib.sha256(p.encode()).hexdigest() for p in paths]}
        command.extend([f"--{arm.lower()}", *paths])
    command.extend(["--output_dir", "/recorded/scorer-output"])
    skill = {a: {s: {"mse_model": 2 + i / 10, "mse_persistence": 4 + j,
                     "skill_vs_persistence": 1 - (2 + i / 10) / (4 + j), "reason": None}
                 for j, s in enumerate(ALL_SCOPES)} for i, a in enumerate(ARMS)}
    return {"protocol": "docs/realdata-phase2-protocol-20261010.md", "declared_family_size": 48,
            "candidates": list(ARMS), "comparators": list(COMPARATORS), "scopes": list(SCOPES),
            "exchangeability_asserted": False, "scorer_command": " ".join(command), "arms": arms,
            "h5_input_protocol": {a: {"zero_input_channels": [2, 3] if a == "R2" else [],
                                      "rollout_regime": "feedback_then_arm_transform"} for a in ARMS},
            "primary_skill_vs_persistence": skill,
            "seed_level_summary": {a: {s: {"n_seeds": 3, "per_seed_trial_mean": [3.0, 7.0, 11.0]}
                                       for s in ALL_SCOPES} for a in ARMS},
            "seed_trial_mse_vectors": {a: {s: {"seeds": list(SEEDS), "per_seed_per_trial_mse":
                                               [[v] * 9 for v in (3.0, 7.0, 11.0)]}
                                          for s in ALL_SCOPES} for a in ARMS},
            "declared_family": {"family_size": 48, "exchangeability_asserted": False, "slots": slots,
                                "holm_bonferroni": {"family_size": 48, "valid_test_count": 0,
                                                    "unavailable_slot_count": 48, "slots": adjusted}}}


def test_unequal_trial_sizes_keep_distinct_estimands_and_unavailable_tests(frozen):
    # One A trial (-4), eight B trials (+1): trial mean is positive, prefix mean negative.
    report = build_report(frozen)
    result = report["prefix_robustness"]["R1|persistence|aligned_pooled"]
    assert result["frozen"]["mean_delta"] == pytest.approx(sum([-4] + [1] * 8) / 9)
    assert result["equal_prefix_mean_delta"] == -1.5
    assert result["n"] == 2 and result["negative"] == result["prefix_wins"] == 1
    assert result["prefix_win_fraction"] == 0.5
    assert result["available"] and not result["frozen"]["available"]
    assert result["frozen"]["cohens_dz"] == frozen["declared_family"]["slots"]["R1|persistence|aligned_pooled"]["cohens_dz"]
    assert report["seed_optimisation_variability"]["R1"]["aligned_pooled"]["per_seed_trial_mean"] == [3.0, 7.0, 11.0]
    assert report["frozen_skill_vs_persistence"]["R1"]["aligned_pooled"]["frozen"]["mse_model"] == 2.1
    assert not report["provenance"]["animal_independence_from_prefixes"]


def test_influential_prefix_reverses_deletion_sign():
    result = prefix_summary({"influential": -10.0, "b": 1.0, "c": 1.0, "d": 1.0})
    assert result["equal_prefix_mean_delta"] == -1.75
    assert result["deletion"]["estimates"] == {"influential": 1.0, "b": -8 / 3, "c": -8 / 3, "d": -8 / 3}
    assert result["deletion"]["range"] == [-8 / 3, 1.0]
    assert result["deletion"]["sign_stable"] is False


def test_ties_one_prefix_and_missing_support():
    tie = prefix_summary({"a": 0.0, "b": 0.0})
    assert (tie["negative"], tie["zero"], tie["positive"]) == (0, 2, 0)
    assert tie["variance_status"] == "zero_delta_variance"
    assert tie["deletion"]["range"] == [0.0, 0.0] and tie["deletion"]["sign_stable"]
    one = prefix_summary({"a": -2.0})
    assert one["equal_prefix_mean_delta"] == -2 and one["variance_status"].startswith("undefined")
    assert one["deletion"]["estimates"] == {} and one["deletion"]["range"] is None
    assert one["deletion"]["sign_stable"] is None
    for missing in (None, {}):
        result = prefix_summary(missing)
        assert not result["available"] and result["equal_prefix_mean_delta"] is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), None, "1", True])
def test_nonfinite_or_nonnumeric_prefix_fails_closed(bad):
    with pytest.raises(ValueError, match="finite"):
        prefix_summary({"a": -1.0, "b": bad})


def test_frozen_adjusted_p_copied_without_new_testing(frozen):
    slot = "R1|persistence|aligned_pooled"
    record = frozen["declared_family"]["slots"][slot]
    frozen["exchangeability_asserted"] = frozen["declared_family"]["exchangeability_asserted"] = True
    frozen["scorer_command"] += " --exchangeability_asserted"
    record.update(available=True, reason=None, prefix_means={str(i): float(i) for i in range(6)})
    record["counts"]["prefixes"] = 6
    record["sign_flip"]["p_value"] = 0.03125
    holm = frozen["declared_family"]["holm_bonferroni"]
    holm["slots"][slot] = {"available": True, "raw_p": 0.03125, "adjusted_p": 1.0, "significant": False}
    holm.update(valid_test_count=1, unavailable_slot_count=47)
    result = build_report(frozen)["prefix_robustness"][slot]
    assert result["adjustment"] == holm["slots"][slot]


@pytest.mark.parametrize("problem", ["missing_slot", "extra_slot", "adjusted_slot", "axis", "seed_count", "seed_ids",
                                     "duplicate_hash", "duplicate_path", "invalid_hash", "lineage", "arm_protocol", "availability"])
def test_frozen_family_integrity_rejects(problem, frozen):
    slot = SLOTS[0]
    if problem == "missing_slot":
        del frozen["declared_family"]["slots"][slot]
    elif problem == "extra_slot":
        frozen["declared_family"]["slots"]["R4|zero|escape"] = {}
    elif problem == "adjusted_slot":
        del frozen["declared_family"]["holm_bonferroni"]["slots"][slot]
    elif problem == "axis":
        frozen["scopes"][-1] = "h25"
    elif problem == "seed_count":
        frozen["seed_level_summary"]["R0"]["escape"]["n_seeds"] = 4
    elif problem == "seed_ids":
        frozen["seed_trial_mse_vectors"]["R0"]["h5"]["seeds"] = [42, 43, 43]
    elif problem == "duplicate_hash":
        frozen["arms"]["R1"]["sha256"][0] = frozen["arms"]["R0"]["sha256"][0]
    elif problem == "duplicate_path":
        previous = frozen["arms"]["R1"]["checkpoints"][0]
        duplicate = frozen["arms"]["R0"]["checkpoints"][0]
        frozen["arms"]["R1"]["checkpoints"][0] = duplicate
        frozen["scorer_command"] = frozen["scorer_command"].replace(previous, duplicate)
    elif problem == "invalid_hash":
        frozen["arms"]["R0"]["sha256"][0] = "z" * 64
    elif problem == "lineage":
        frozen["scorer_command"] = frozen["scorer_command"].replace("--nested_prior", "--unknown")
    elif problem == "arm_protocol":
        frozen["h5_input_protocol"]["R2"]["zero_input_channels"] = []
    else:
        frozen["declared_family"]["holm_bonferroni"]["slots"][slot]["available"] = True
    with pytest.raises(ValueError):
        validate_family(frozen)


@pytest.mark.parametrize("problem", ["exchangeability", "family_exchangeability", "false_significance", "missed_significance",
                                     "adjusted_below_raw", "unavailable_raw", "unavailable_adjusted", "unavailable_significance"])
def test_frozen_p_relationships_reject_without_recomputing_holm(problem, frozen):
    slot = SLOTS[0]
    record = frozen["declared_family"]["slots"][slot]
    adjusted = frozen["declared_family"]["holm_bonferroni"]["slots"][slot]
    if problem.startswith("unavailable_"):
        field = {"unavailable_raw": "raw_p", "unavailable_adjusted": "adjusted_p", "unavailable_significance": "significant"}[problem]
        adjusted[field] = False if field == "significant" else 0.9
    else:
        frozen["exchangeability_asserted"] = frozen["declared_family"]["exchangeability_asserted"] = True
        frozen["scorer_command"] += " --exchangeability_asserted"
        record.update(available=True, reason=None)
        record["sign_flip"]["p_value"] = 0.9
        adjusted.update(available=True, raw_p=0.9, adjusted_p=0.9, significant=False)
        holm = frozen["declared_family"]["holm_bonferroni"]
        holm.update(valid_test_count=1, unavailable_slot_count=47)
        if problem == "exchangeability":
            frozen["exchangeability_asserted"] = frozen["declared_family"]["exchangeability_asserted"] = False
        elif problem == "family_exchangeability":
            frozen["declared_family"]["exchangeability_asserted"] = False
        elif problem == "false_significance":
            adjusted["significant"] = True
        elif problem == "missed_significance":
            record["sign_flip"]["p_value"] = 0.01
            adjusted.update(raw_p=0.01, adjusted_p=0.01)
        else:
            adjusted["adjusted_p"] = 0.5
    with pytest.raises(ValueError):
        validate_family(frozen)


def test_negative_stored_mean_rejected_for_positive_observations(frozen):
    for arm in ARMS:
        frozen["seed_trial_mse_vectors"][arm]["escape"]["per_seed_per_trial_mse"] = [[1, 1]] * 3
        frozen["seed_level_summary"][arm]["escape"]["per_seed_trial_mean"] = [1, 1, 1]
    validate_family(frozen)  # Family eligibility counts do not define retained-vector length.
    frozen["seed_level_summary"]["R1"]["escape"]["per_seed_trial_mean"] = [-99, 1, 1]
    with pytest.raises(ValueError, match="nanmean"):
        validate_family(frozen)


@pytest.mark.parametrize("problem", ["short_seed_row", "other_arm_shape", "negative", "infinite", "nan", "boolean", "corrupt_mean"])
def test_seed_vectors_and_stored_means_fail_closed(problem, frozen):
    rows = frozen["seed_trial_mse_vectors"]["R1"]["escape"]["per_seed_per_trial_mse"]
    if problem == "short_seed_row":
        rows[1].pop()
    elif problem == "other_arm_shape":
        for row in rows:
            row.pop()
    elif problem == "corrupt_mean":
        frozen["seed_level_summary"]["R1"]["escape"]["per_seed_trial_mean"][0] = 100.0
    else:
        rows[0][0] = {"negative": -1.0, "infinite": float("inf"), "nan": float("nan"), "boolean": True}[problem]
    with pytest.raises(ValueError, match="shape|MSE|nanmean"):
        validate_family(frozen)


def test_null_seed_entries_follow_nanmean_and_missing_support(frozen):
    rows = frozen["seed_trial_mse_vectors"]["R1"]["escape"]["per_seed_per_trial_mse"]
    rows[0][0] = None
    rows[1] = [None] * 9
    frozen["seed_level_summary"]["R1"]["escape"]["per_seed_trial_mean"][1] = None
    result = build_report(frozen)["seed_optimisation_variability"]["R1"]["escape"]
    assert result["per_seed_trial_mean"] == [3.0, None, 11.0]
    assert result["reason"] == "missing_or_nonfinite_seed_means"


def test_missing_secondary_unavailable_without_fabricating(frozen):
    del frozen["primary_skill_vs_persistence"]["R1"]["h25"]
    del frozen["seed_level_summary"]["R1"]["h25"]
    result = build_report(frozen)
    assert not result["frozen_skill_vs_persistence"]["R1"]["h25"]["available"]
    assert result["seed_optimisation_variability"]["R1"]["h25"]["per_seed_trial_mean"] is None
    bad = copy.deepcopy(frozen)
    bad["primary_skill_vs_persistence"]["R0"]["h5"]["mse_persistence"] += 1
    with pytest.raises(ValueError, match="skill"):
        build_report(bad)


def test_cli_json_csv_and_nine_figures_smoke(tmp_path, frozen):
    del frozen["primary_skill_vs_persistence"]["R1"]["h25"]
    input_path, output = tmp_path / "realdata_phase2_family.json", tmp_path / "supplement"
    original = json.dumps(frozen).encode()
    input_path.write_bytes(original)
    command = [sys.executable, str(REPO / "scripts/analyze_publication_robustness.py"),
               "--input", str(input_path), "--output-dir", str(output)]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=60,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1",
                                 "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLCONFIGDIR": str(tmp_path / "mpl")})
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "publication_robustness.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance == report["provenance"]
    assert provenance["input_sha256"] == hashlib.sha256(original).hexdigest()
    assert provenance["command"][1:] == command[2:]
    for relative, digest in provenance["code_sha256"].items():
        assert digest == hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
    assert provenance["no_new_testing"] and provenance["validation_guided_selection"]
    with (output / "family_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 48
    selected = next(r for r in rows if r["slot"] == "R1|persistence|aligned_pooled")
    assert float(selected["equal_prefix_mean_delta"]) == -1.5 and float(selected["frozen_trial_mean_delta"]) == 4 / 9
    assert selected["delta_units"] == "(cm/s)^2" and selected["cohens_dz_units"] == "dimensionless"
    assert provenance["units"] == {"prefix_delta": "(cm/s)^2", "per_seed_trial_mean": "(cm/s)^2", "skill": "dimensionless"}
    assert report["seed_optimisation_variability"]["R1"]["escape"]["mse_units"] == "(cm/s)^2"
    assert selected["adjusted_p"] == "" and selected["frozen_available"] == "False"
    with (output / "prefix_detail.csv").open(newline="") as handle:
        detail = list(csv.DictReader(handle))
    assert len(detail) == 44 * 2 and {float(d["deletion_estimate"]) for d in detail} == {-4, 1}
    for name in ("primary_skill", "prefix_robustness", "horizon_skill"):
        for extension, signature in (("pdf", b"%PDF"), ("svg", b"<?xml"), ("png", b"\x89PNG")):
            artifact = output / f"{name}.{extension}"
            assert artifact.stat().st_size > 1000 and artifact.read_bytes().startswith(signature)
    assert "per_seed_trial_mean" in (output / "primary_skill.svg").read_text()
    assert "(cm/s)^2" in (output / "primary_skill.svg").read_text()
    assert "(cm/s)^2" in (output / "prefix_robustness.svg").read_text()
    assert "dimensionless" in (output / "horizon_skill.svg").read_text()
    assert "fewer_than_six_prefixes" in (output / "prefix_robustness.svg").read_text()
    assert "recordingA" in (output / "prefix_robustness.svg").read_text()
    assert input_path.read_bytes() == original
    with pytest.raises(FileExistsError):
        write_supplement(input_path, output, command)
    assert input_path.read_bytes() == original


def test_duplicate_json_keys_and_figure_failure_are_loud(tmp_path, frozen, monkeypatch):
    input_path = tmp_path / "input.json"
    input_path.write_text('{"candidates": [], "candidates": []}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        write_supplement(input_path, tmp_path / "duplicate", [])
    assert not (tmp_path / "duplicate").exists()
    input_path.write_text(json.dumps(frozen))
    import matplotlib.figure
    def fail(*args, **kwargs):
        raise OSError("figure export failed")
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fail)
    with pytest.raises(OSError, match="figure export failed"):
        write_supplement(input_path, tmp_path / "failure", [])
