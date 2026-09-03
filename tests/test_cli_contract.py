"""
CLI contract tests — the runner must call the scripts it actually has.

``run_pipeline.sh`` and the ``Makefile`` are the only end-to-end entry
points, yet nothing verified that the flags they pass exist.  The runner
had drifted far enough that Phase A died on ``argparse`` before a single
byte of data was read (``prepare_data.py`` has ``--output``, never
``--output_dir``), so the "end-to-end reproducible" claim could not have
been true for any of the phases behind it.

These tests parse the *declared* options straight out of each script's
AST (no imports, no torch) and compare them with the argv the runner and
Makefile really emit, so the next drift fails in CI instead of in a
publication run.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "run_pipeline.sh"
MAKEFILE = REPO_ROOT / "Makefile"

# The six analyses the README/PRD count as the publication figure set,
# plus the two stages that produce their inputs.
REQUIRED_PIPELINE_SCRIPTS = {
    "scripts/prepare_data.py",
    "scripts/train.py",
    "scripts/analyze_dynamics.py",
    "scripts/simulate_lesion.py",
    "scripts/analyze_jacobian.py",
    "scripts/analyze_integration.py",
    "scripts/simulate_psychophysics.py",
    "scripts/analyze_gating.py",
    "scripts/simulate_autoregressive.py",
}

def _declared_options(script: Path) -> Tuple[Set[str], Set[str]]:
    """
    Extract ``(all_options, required_options)`` from a script's AST.

    Static parsing keeps the test cheap and import-free: the analysis
    scripts pull in torch, matplotlib, sklearn and umap at module level.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    declared: Set[str] = set()
    required: Set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        options = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("-")
        ]
        if not options:
            continue
        declared.update(options)
        is_required = any(
            kw.arg == "required"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in node.keywords
        )
        if is_required:
            required.add(options[0])
    return declared, required


def _flags(tokens: List[str]) -> List[str]:
    """Long options in an argv token list."""
    return [t for t in tokens if t.startswith("--")]


def _run_runner(**env_overrides: str) -> subprocess.CompletedProcess:
    """Execute ``run_pipeline.sh`` in DRY_RUN mode (nothing is launched)."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")

    env = dict(os.environ)
    env["DRY_RUN"] = "1"
    # POSIX interpreter path only: a Windows-style sys.executable is not
    # resolvable by the runner's `command -v` preflight.
    if sys.executable.startswith("/"):
        env["PYTHON"] = sys.executable
    env.update(env_overrides)

    return subprocess.run(
        [bash, str(RUNNER)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _planned_stages() -> List[List[str]]:
    """``[[script, *args], ...]`` as the runner would really invoke them."""
    result = _run_runner()
    assert result.returncode == 0, (
        f"DRY_RUN=1 run_pipeline.sh failed:\n{result.stdout}\n{result.stderr}"
    )
    stages: List[List[str]] = []
    for line in result.stdout.splitlines():
        if line.startswith("PLAN "):
            tokens = line.split()[1:]  # drop "PLAN"
            assert len(tokens) >= 2, f"malformed plan line: {line}"
            stages.append(tokens[1:])  # drop the interpreter
    assert stages, "runner emitted no stages"
    return stages


@pytest.fixture(scope="module")
def planned_stages() -> List[List[str]]:
    return _planned_stages()


def test_runner_only_passes_declared_flags(planned_stages) -> None:
    """Every flag the runner passes must exist in that script's parser."""
    errors: List[str] = []
    for tokens in planned_stages:
        script = REPO_ROOT / tokens[0]
        assert script.is_file(), f"runner invokes a missing script: {tokens[0]}"
        declared, _ = _declared_options(script)
        for flag in _flags(tokens[1:]):
            if flag not in declared:
                errors.append(f"{tokens[0]}: unknown flag {flag}")
    assert not errors, "run_pipeline.sh passes undeclared flags:\n" + "\n".join(errors)


def test_runner_supplies_every_required_flag(planned_stages) -> None:
    """A stage must not die on a missing ``required=True`` argument."""
    errors: List[str] = []
    for tokens in planned_stages:
        _, required = _declared_options(REPO_ROOT / tokens[0])
        passed = set(_flags(tokens[1:]))
        for flag in sorted(required - passed):
            errors.append(f"{tokens[0]}: required flag {flag} not passed")
    assert not errors, "\n".join(errors)


def test_runner_never_repeats_a_flag(planned_stages) -> None:
    for tokens in planned_stages:
        flags = _flags(tokens[1:])
        assert len(flags) == len(set(flags)), f"{tokens[0]}: duplicated flags {flags}"


def test_runner_covers_every_pipeline_stage(planned_stages) -> None:
    """
    The gating-cluster analysis used to be absent from the runner while
    ``make analyze`` and the README both counted six analyses.
    """
    planned = {tokens[0].replace("\\", "/") for tokens in planned_stages}
    missing = REQUIRED_PIPELINE_SCRIPTS - planned
    assert not missing, f"run_pipeline.sh never runs: {sorted(missing)}"


PINNED_FLAGS = ("--dataset", "--dt_ms", "--config", "--raw_dir")


def test_runner_pins_dataset_and_dt_ms_where_supported(planned_stages) -> None:
    """
    Provenance: a stage that can read a dataset, a raw directory or a
    config must be told which one this run uses, and a stage that
    converts frames to physical time must be given the same dt_ms.
    Relying on the argparse defaults lets an analysis silently score a
    stale dataset, a foreign config, or a wrong sampling interval.
    """
    errors: List[str] = []
    for tokens in planned_stages:
        script = tokens[0]
        declared, _ = _declared_options(REPO_ROOT / script)
        passed = set(_flags(tokens[1:]))
        for flag in PINNED_FLAGS:
            if flag in declared and flag not in passed:
                errors.append(f"{script}: declares {flag} but the runner omits it")
    assert not errors, "\n".join(errors)


def _makefile_invocations() -> List[Tuple[int, str, List[str]]]:
    """``[(lineno, script, tokens), ...]`` for every recipe line running a script."""
    pattern = re.compile(r"(scripts/[A-Za-z0-9_]+\.py)")
    found: List[Tuple[int, str, List[str]]] = []
    for lineno, raw in enumerate(
        MAKEFILE.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.startswith("\t"):
            continue
        match = pattern.search(raw)
        if match is None:
            continue
        found.append((lineno, match.group(1), raw.split()))
    return found


def test_makefile_only_passes_declared_flags() -> None:
    invocations = _makefile_invocations()
    assert invocations, "no script invocations found in the Makefile"
    errors: List[str] = []
    for lineno, script, tokens in invocations:
        path = REPO_ROOT / script
        if not path.is_file():
            errors.append(f"Makefile:{lineno}: missing script {script}")
            continue
        declared, required = _declared_options(path)
        passed = set(_flags(tokens))
        for flag in _flags(tokens):
            if flag not in declared:
                errors.append(f"Makefile:{lineno}: {script} has no flag {flag}")
        for flag in sorted(required - passed):
            errors.append(f"Makefile:{lineno}: {script} needs {flag}")
    assert not errors, "\n".join(errors)


def test_runner_rejects_dt_ms_that_contradicts_the_config() -> None:
    """
    A frame interval that disagrees with ``model.dt_ms`` silently rescales
    every latency and time constant reported by the analyses, so the
    runner must refuse it instead of producing plausible-looking figures.
    """
    result = _run_runner(DT_MS="25")
    assert result.returncode != 0
    assert "contradicts" in result.stderr
    assert "PLAN " not in result.stdout


def test_makefile_pins_dataset_and_dt_ms_where_supported() -> None:
    """Same provenance rule as the runner: no reliance on stale defaults."""
    errors: List[str] = []
    for lineno, script, tokens in _makefile_invocations():
        path = REPO_ROOT / script
        if not path.is_file():
            continue
        declared, _ = _declared_options(path)
        passed = set(_flags(tokens))
        for flag in ("--dataset", "--dt_ms"):
            if flag in declared and flag not in passed:
                errors.append(
                    f"Makefile:{lineno}: {script} declares {flag} but the recipe omits it"
                )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("script", sorted(REQUIRED_PIPELINE_SCRIPTS))
def test_entry_script_is_executable(script: str) -> None:
    """
    ``--help`` exercises the whole import chain.  Nine of the entry
    scripts import ``nsmor`` without bootstrapping ``sys.path``, so a
    clone without ``pip install -e .`` used to die on
    ModuleNotFoundError before argparse ever ran — exactly the
    environment the runner now provides via PYTHONPATH.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{script} --help failed (rc={result.returncode}):\n{result.stderr[-2000:]}"
    )


def test_runner_rejects_phase1_epochs_without_a_total_budget() -> None:
    """``--epochs`` is the TOTAL budget: phase 2 = epochs - phase1_epochs."""
    result = _run_runner(PHASE1_EPOCHS="150")
    assert result.returncode != 0
    assert "EPOCHS" in result.stderr


def test_runner_rejects_phase1_epochs_that_starve_phase2() -> None:
    result = _run_runner(PHASE1_EPOCHS="150", EPOCHS="150")
    assert result.returncode != 0
    assert "0 epochs" in result.stderr


def test_runner_forwards_two_phase_training_when_requested() -> None:
    stages = {
        tokens[0]: tokens
        for tokens in [
            line.split()[2:]
            for line in _run_runner(
                PHASE1_EPOCHS="150", EPOCHS="300"
            ).stdout.splitlines()
            if line.startswith("PLAN ")
        ]
    }
    train = stages.get("scripts/train.py")
    assert train is not None, "training stage missing from the plan"
    assert "--phase1_epochs" in train and "150" in train
    assert "--epochs" in train and "300" in train


def test_runner_forwards_batch_size_lr_seed() -> None:
    """The runner forwards training, ETL, and psychophysics overrides."""
    result = _run_runner(
        EPOCHS="300",
        PHASE1_EPOCHS="150",
        BATCH_SIZE="32",
        LR="0.001",
        SEED="99",
    )
    assert result.returncode == 0, (
        f"DRY_RUN=1 failed:\n{result.stdout}\n{result.stderr}"
    )
    stages = _planned_stage_map(result.stdout)
    train = stages["scripts/train.py"]
    assert _arg_value(train, "--batch_size") == "32"
    assert _arg_value(train, "--lr") == "0.001"
    etl = stages["scripts/prepare_data.py"]
    assert _arg_value(etl, "--seed") == "99"
    psych = stages["scripts/simulate_psychophysics.py"]
    assert _arg_value(psych, "--seed") == "99"


def _run_make_pipeline(**make_overrides: str) -> subprocess.CompletedProcess:
    """Execute the public Makefile pipeline seam in dry-run mode."""
    make = shutil.which("make")
    if make is None:
        pytest.skip("make unavailable")

    env = dict(os.environ)
    env["DRY_RUN"] = "1"
    if sys.executable.startswith("/"):
        env["PYTHON"] = sys.executable
    arguments = [
        make,
        "--no-print-directory",
        "pipeline",
        *(f"{key}={value}" for key, value in make_overrides.items()),
    ]
    return subprocess.run(
        arguments,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _planned_stage_map(output: str) -> dict[str, List[str]]:
    """Parse runner ``PLAN`` lines into ``script -> argv`` entries."""
    stages: dict[str, List[str]] = {}
    for line in output.splitlines():
        if not line.startswith("PLAN "):
            continue
        tokens = line.split()
        assert len(tokens) >= 3, f"malformed plan line: {line}"
        stages[tokens[2]] = tokens[3:]
    assert stages, "pipeline emitted no planned stages"
    return stages


def _arg_value(tokens: List[str], flag: str) -> str:
    """Return the value following a unique CLI flag."""
    index = tokens.index(flag)
    assert index + 1 < len(tokens), f"{flag} has no value in {tokens}"
    return tokens[index + 1]


def test_makefile_pipeline_dry_run_forwards_all_required_variables() -> None:
    """Distinct Make overrides must reach the runner's planned script argv."""
    result = _run_make_pipeline(
        EPOCHS="301",
        PHASE1_EPOCHS="17",
        PRE_EPOCHS="23",
        BATCH_SIZE="37",
        LR="0.0123",
        SEED="99",
        DT_MS="4.0",
    )
    assert result.returncode == 0, (
        f"DRY_RUN=1 make pipeline failed:\n{result.stdout}\n{result.stderr}"
    )
    stages = _planned_stage_map(result.stdout)

    train = stages["scripts/train.py"]
    assert _arg_value(train, "--epochs") == "301"
    assert _arg_value(train, "--phase1_epochs") == "17"
    assert _arg_value(train, "--batch_size") == "37"
    assert _arg_value(train, "--lr") == "0.0123"

    etl = stages["scripts/prepare_data.py"]
    assert _arg_value(etl, "--seed") == "99"
    assert _arg_value(etl, "--dt_ms") == "4.0"

    psych = stages["scripts/simulate_psychophysics.py"]
    assert _arg_value(psych, "--seed") == "99"

    for script in (
        "scripts/simulate_lesion.py",
        "scripts/analyze_jacobian.py",
        "scripts/analyze_integration.py",
        "scripts/simulate_autoregressive.py",
    ):
        assert _arg_value(stages[script], "--dt_ms") == "4.0"


# ── Loader entry points must honour the raw_dir they are given ───────
#
# ``Makefile`` passes ``$(RAW)`` to both pre-load scripts, but their
# ``__main__`` blocks used to call the worker with no arguments and never
# read ``sys.argv`` -- so ``make load RAW=/somewhere/else`` archived and
# converted ``data/raw`` instead, printed a success line, and exited 0.
# A silent no-op on the directory the operator named is the worst
# possible failure mode for an ingestion step, and nothing asserted that
# the argument was used at all.
#
# Both tests run with ``cwd`` set to a scratch directory so the buggy
# behaviour cannot reach the repository's real ``data/raw``.

_LEGACY_KIN_CSV = "sys_time,ard_time,dx,dy,dz,stim_state,global_trial_id\n" + "".join(
    # Two trials so the converter's per-trial groupby sees more than one
    # group; a single-group frame is not what the real corpus looks like
    # (18 trials per session block).
    f"{t * 0.004:.3f},{t * 0.004:.3f},{t % 3},{(t + 1) % 3},0,"
    f"{1 if t % 5 == 0 else 0},{trial}\n"
    for trial in (1, 2)
    for t in range(12)
)
_LEGACY_EVT_CSV = (
    "event_name,timestamp,session_num,trial_in_session,global_trial_id,details\n"
    'trial_start,0.0,1,1,1,"{""type"": ""baseline_wind"", ""wind_dir"": ""left""}"\n'
    'trial_start,0.048,1,2,2,"{""type"": ""baseline_wind"", ""wind_dir"": ""right""}"\n'
)

_SESSION_STEM = "0.500cricket_001_20260101_000000_session_1"


def _seed_legacy_session(root: Path) -> Tuple[Path, Path]:
    """Drop an unarchived legacy kinematics/events pair into *root*."""
    root.mkdir(parents=True, exist_ok=True)
    kin = root / f"{_SESSION_STEM}_kinematics.csv"
    evt = root / f"{_SESSION_STEM}_events.csv"
    kin.write_text(_LEGACY_KIN_CSV, encoding="utf-8")
    evt.write_text(_LEGACY_EVT_CSV, encoding="utf-8")
    return kin, evt


def _run_entry_script(script: str, cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_pre_load_data_archives_into_the_raw_dir_it_is_given(
    tmp_path: Path,
) -> None:
    """``pre_load_data.py <raw_dir>`` must archive inside *that* directory."""
    staging = tmp_path / "staging"
    _seed_legacy_session(staging)

    result = _run_entry_script("scripts/pre_load_data.py", tmp_path, str(staging))
    assert result.returncode == 0, (
        f"pre_load_data.py failed:\n{result.stdout}\n{result.stderr}"
    )

    archived_dir = staging / _SESSION_STEM
    assert archived_dir.is_dir(), (
        "pre_load_data.py ignored the raw_dir argument: no session folder was "
        f"created under {staging}. Directory now holds: "
        f"{sorted(p.name for p in staging.iterdir())}"
    )
    assert (archived_dir / f"{_SESSION_STEM}_kinematics.csv").is_file()
    assert (archived_dir / f"{_SESSION_STEM}_events.csv").is_file()
    # The default tree must not have been touched or invented.
    assert not (tmp_path / "data").exists()


def test_pre_load_adapt_converts_the_raw_dir_it_is_given(
    tmp_path: Path,
) -> None:
    """``pre_load_adapt.py <raw_dir>`` must convert inside *that* directory."""
    staging = tmp_path / "staging"
    session_dir = staging / _SESSION_STEM
    _seed_legacy_session(session_dir)

    result = _run_entry_script("scripts/pre_load_adapt.py", tmp_path, str(staging))
    assert result.returncode == 0, (
        f"pre_load_adapt.py failed:\n{result.stdout}\n{result.stderr}"
    )

    kin_text = (session_dir / f"{_SESSION_STEM}_kinematics.csv").read_text(
        encoding="utf-8"
    )
    header = kin_text.splitlines()[0]
    assert "session_id" in header and "time_ms" in header, (
        "pre_load_adapt.py ignored the raw_dir argument: the legacy schema "
        f"was never rewritten. Header is still: {header!r}"
    )
    assert not (tmp_path / "data").exists()
