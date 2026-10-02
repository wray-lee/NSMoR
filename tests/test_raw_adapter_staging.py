"""Tests for safe raw data adaptation and staging.

Validates non-destructive adaptation from flat raw session files to
canonical nested session directories consumable by prepare_data.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from nsmor.pipeline.grouping import animal_of
from nsmor.pipeline.io import (
    EVENT_COLUMNS,
    KINEMATICS_COLUMNS,
    load_and_concat_sessions,
    load_events_csv,
    load_kinematics_csv,
)
from scripts.prepare_data import pair_csv_files
from scripts.pre_load_adapt import (
    adapt_cercus_to_nsmor,
    adapt_session_pair,
    derive_session_id,
    extract_file_stem,
    find_session_pairs,
    is_protected_raw_path,
    validate_output_paths,
)


def _file_sha256(path: Path) -> str:
    """Compute SHA256 checksum of a file by streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _create_synthetic_raw_pair(
    directory: Path,
    session_id: str,
    n_samples: int = 20,
    trial_id: int = 1,
    trial_type: str = "baseline_visual",
    lv_ratio_ms: float = 120.0,
    wind_state: int = 0,
) -> Tuple[Path, Path]:
    """Create a pair of raw kinematics and events CSV files."""
    kin_path = directory / f"{session_id}_kinematics.csv"
    evt_path = directory / f"{session_id}_events.csv"

    sys_time = np.arange(n_samples, dtype=np.float64) * 0.004
    ard_time = (sys_time * 1000.0).astype(np.int64)
    dx = np.ones(n_samples, dtype=np.float64) * 0.1
    dy = np.ones(n_samples, dtype=np.float64) * 0.05
    dz = np.zeros(n_samples, dtype=np.float64)
    stim = np.full(n_samples, wind_state, dtype=np.int64)
    trials = np.full(n_samples, trial_id, dtype=np.int64)

    df_k = pd.DataFrame({
        "sys_time": sys_time,
        "ard_time": ard_time,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "stim_state": stim,
        "global_trial_id": trials,
    })
    df_k.to_csv(kin_path, index=False)

    details = json.dumps({
        "type": trial_type,
        "target_ttc_ms": None,
        "lv_ratio_ms": lv_ratio_ms,
        "wind_dir": "none",
        "screen_side": "right",
    })
    df_e = pd.DataFrame([
        {
            "event_name": "trial_start",
            "timestamp": 0.0,
            "session_num": 1,
            "trial_in_session": 1,
            "global_trial_id": trial_id,
            "details": details,
        },
        {
            "event_name": "phase_transition",
            "timestamp": 0.0,
            "session_num": 1,
            "trial_in_session": 1,
            "global_trial_id": trial_id,
            "details": json.dumps({"from_phase": "TrialStart", "to_phase": "Looming"}),
        },
    ])
    df_e.to_csv(evt_path, index=False)
    return kin_path, evt_path


class TestFlatSessionsPairingAndImmutability:
    """1) 2 flat sessions from different animals correctly pair and produce

    distinct nested IDs/outputs; input bytes unchanged; stable trial IDs preserved.
    """

    def test_two_flat_sessions_paired_and_immutable(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_flat"
        raw_dir.mkdir()
        out_dir = tmp_path / "staging_adapted"

        s1 = "0.513cricket_001_20260707_193143_session_1"
        s2 = "0.585cricket_001_20260824_104031_session_1"
        assert animal_of(s1) != animal_of(s2)

        k1, e1 = _create_synthetic_raw_pair(raw_dir, s1, trial_id=37)
        k2, e2 = _create_synthetic_raw_pair(raw_dir, s2, trial_id=42)

        hashes_before = {
            k1: _file_sha256(k1),
            e1: _file_sha256(e1),
            k2: _file_sha256(k2),
            e2: _file_sha256(e2),
        }

        report = adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        # Source CSV bytes strictly unchanged
        for path, expected_hash in hashes_before.items():
            assert _file_sha256(path) == expected_hash, (
                f"Source file mutated: {path}"
            )

        # Distinct nested outputs
        s1_out = out_dir / s1
        s2_out = out_dir / s2
        assert s1_out.is_dir()
        assert s2_out.is_dir()

        k1_out = s1_out / f"{s1}_kinematics.csv"
        e1_out = s1_out / f"{s1}_events.csv"
        k2_out = s2_out / f"{s2}_kinematics.csv"
        e2_out = s2_out / f"{s2}_events.csv"
        assert k1_out.is_file() and e1_out.is_file()
        assert k2_out.is_file() and e2_out.is_file()

        df_k1 = pd.read_csv(k1_out)
        df_e1 = pd.read_csv(e1_out)
        df_k2 = pd.read_csv(k2_out)
        df_e2 = pd.read_csv(e2_out)

        assert (df_k1["session_id"] == s1).all()
        assert (df_e1["session_id"] == s1).all()
        assert (df_k2["session_id"] == s2).all()
        assert (df_e2["session_id"] == s2).all()

        # Stable trial IDs preserved
        assert (df_k1["trial_id"] == 37).all()
        assert (df_e1["trial_id"] == 37).all()
        assert (df_k2["trial_id"] == 42).all()
        assert (df_e2["trial_id"] == 42).all()


class TestLegacyNestedAndValidationRejections:
    """2) Nested legacy layout still works; orphan/mismatched pair and

    duplicate resolved session ID fail clearly instead of arbitrary first match.
    """

    def test_nested_legacy_layout_adapted(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_nested"
        raw_dir.mkdir()
        out_dir = tmp_path / "staging_adapted"

        s1 = "session_001"
        s1_dir = raw_dir / s1
        s1_dir.mkdir()
        _create_synthetic_raw_pair(s1_dir, s1, trial_id=10)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        out_s1_dir = out_dir / s1
        assert out_s1_dir.is_dir()
        df_k = pd.read_csv(out_s1_dir / f"{s1}_kinematics.csv")
        assert (df_k["session_id"] == s1).all()
        assert (df_k["trial_id"] == 10).all()

    def test_orphan_kinematics_rejected(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_orphan_k"
        raw_dir.mkdir()
        out_dir = tmp_path / "staging"
        k, _ = _create_synthetic_raw_pair(raw_dir, "session_001")
        # Remove event file
        (raw_dir / "session_001_events.csv").unlink()

        with pytest.raises(ValueError, match="[Uu]npaired"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

    def test_orphan_events_rejected(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_orphan_e"
        raw_dir.mkdir()
        out_dir = tmp_path / "staging"
        _create_synthetic_raw_pair(raw_dir, "session_001")
        # Remove kinematics file
        (raw_dir / "session_001_kinematics.csv").unlink()

        with pytest.raises(ValueError, match="[Uu]npaired"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

    def test_duplicate_session_id_rejected(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_dups"
        raw_dir.mkdir()
        out_dir = tmp_path / "staging"

        sub1 = raw_dir / "sub1"
        sub2 = raw_dir / "sub2"
        sub1.mkdir()
        sub2.mkdir()

        # Both subdirectories contain files resolving to session_001
        _create_synthetic_raw_pair(sub1, "session_001")
        _create_synthetic_raw_pair(sub2, "session_001")

        with pytest.raises(ValueError, match="[Dd]uplicate"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)


class TestSafetyGuardsAndCollisions:
    """3) Source=output, path alias if safely testable, and existing target

    collisions reject BEFORE writing. If testing in-place legacy behavior, use
    ONLY temp synthetic fixtures; NEVER real data root.
    """

    def test_source_equals_output_rejected(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        with pytest.raises(ValueError, match="identical"):
            validate_output_paths(raw_dir, raw_dir)

    def test_adapt_cercus_to_nsmor_entrypoint_rejects_source_equals_output(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        _create_synthetic_raw_pair(raw_dir, "session_001")
        with pytest.raises(ValueError, match="identical"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=raw_dir)

    def test_output_inside_raw_rejected(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = raw_dir / "nested_output"
        with pytest.raises(ValueError, match="inside"):
            validate_output_paths(raw_dir, out_dir)

    def test_raw_inside_output_rejected(self, tmp_path: Path):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        raw_dir = out_dir / "nested_raw"
        raw_dir.mkdir()
        with pytest.raises(ValueError, match="inside"):
            validate_output_paths(raw_dir, out_dir)

    def test_path_alias_samefile_rejected(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        alias_dir = tmp_path / "foo" / ".." / "raw"
        with pytest.raises(ValueError, match="identical|resolves to the same"):
            validate_output_paths(raw_dir, alias_dir)

    def test_case_insensitive_alias_rejected_on_drvfs(self):
        drvfs_test = Path("/mnt/c/Users/Wray/AppData/Local/Temp")
        if not drvfs_test.is_dir():
            pytest.skip("DrvFs path not available")
        alias_path = Path("/mnt/c/users/wray/appdata/local/temp")
        with pytest.raises(ValueError, match="resolves to the same"):
            validate_output_paths(drvfs_test, alias_path)

    def test_existing_target_collision_rejects_before_write(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        s1 = "session_001"
        _create_synthetic_raw_pair(raw_dir, s1)

        # Pre-seed conflicting target file
        target_dir = out_dir / s1
        target_dir.mkdir(parents=True)
        conflicting_file = target_dir / f"{s1}_kinematics.csv"
        conflicting_file.write_text("pre-existing content", encoding="utf-8")

        with pytest.raises(FileExistsError, match="already exists"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        # Ensure pre-existing file was NOT overwritten
        assert conflicting_file.read_text(encoding="utf-8") == "pre-existing content"

    @pytest.mark.parametrize("single_pair", [False, True], ids=["batch", "single-pair"])
    @pytest.mark.parametrize("existing", ["old-session", "empty-subdirectory", "hidden-file"])
    def test_nonempty_output_rejected_without_source_or_target_writes(
        self, tmp_path: Path, single_pair: bool, existing: str
    ):
        """A stale session with a different ID must not join this run's ETL input."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        s1, s2 = "session_001", "session_002"
        current_pair = _create_synthetic_raw_pair(raw_dir, s1)
        _create_synthetic_raw_pair(raw_dir, s2, trial_id=2)
        out_dir = tmp_path / "staging"
        out_dir.mkdir()

        if existing == "old-session":
            old_raw = tmp_path / "old_raw"
            old_raw.mkdir()
            old_sid = "unrelated_old_session"
            old_pair = _create_synthetic_raw_pair(old_raw, old_sid, trial_id=99)
            adapt_session_pair(*old_pair, output_session_dir=out_dir / old_sid)
            # Verify this fixture is a real, consumable stale session, not just
            # a same-name target collision covered by the earlier regression.
            assert len(pair_csv_files(out_dir)) == 1
            assert not (out_dir / s1).exists()
            assert not (out_dir / s2).exists()
        elif existing == "empty-subdirectory":
            (out_dir / "unrelated_empty_session").mkdir()
        else:
            (out_dir / ".old-run-marker").write_bytes(b"preserve old staging marker")

        def snapshot(root: Path):
            result = {}
            for path in [root, *root.rglob("*")]:
                st = path.lstat()
                result[str(path.relative_to(root))] = (
                    st.st_mode, st.st_ino, st.st_size, st.st_mtime_ns,
                    path.read_bytes() if path.is_file() else None,
                )
            return result

        source_before = snapshot(raw_dir)
        target_before = snapshot(out_dir)
        kwargs = {"single_pair": current_pair} if single_pair else {}
        with pytest.raises(FileExistsError, match="output_dir.*not empty"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir, **kwargs)

        assert snapshot(raw_dir) == source_before, "Source was modified during preflight"
        assert snapshot(out_dir) == target_before, "Target was modified during preflight"
        assert not (out_dir / s1).exists()
        assert not (out_dir / s2).exists()

    def test_output_file_rejected_without_source_or_target_writes(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        sources = _create_synthetic_raw_pair(raw_dir, "session_001")
        source_before = {path: path.read_bytes() for path in sources}
        out_dir = tmp_path / "staging"
        out_dir.write_bytes(b"existing file must survive")
        target_before = out_dir.read_bytes()
        with pytest.raises(FileExistsError, match="output_dir.*not a directory"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        assert {path: path.read_bytes() for path in sources} == source_before
        assert out_dir.read_bytes() == target_before

    def test_broken_symlink_collision_rejected_in_adapt_cercus_to_nsmor(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_sym"
        raw_dir.mkdir()
        out_dir = tmp_path / "out_sym"
        out_dir.mkdir()

        s1 = "session_001"
        _create_synthetic_raw_pair(raw_dir, s1)

        s_dir = out_dir / s1
        s_dir.mkdir(parents=True)
        broken_link = s_dir / f"{s1}_kinematics.csv"
        try:
            os.symlink(tmp_path / "nonexistent.csv", broken_link)
        except OSError:
            pytest.skip("Symlink creation not supported in this filesystem")

        with pytest.raises(FileExistsError, match="already exists"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

    def test_case_insensitive_parent_alias_rejected_when_output_dir_does_not_exist_on_drvfs(self):
        drvfs_test = Path("/mnt/c/Users/Wray/AppData/Local/Temp")
        if not drvfs_test.is_dir():
            pytest.skip("DrvFs path not available")
        raw_dir = drvfs_test / "TEST_DRVFS_RAW_UNCREATED_STAGING"
        raw_dir.mkdir(exist_ok=True)
        try:
            out_dir = drvfs_test / "test_drvfs_raw_uncreated_staging" / "nonexistent_staging"
            with pytest.raises(ValueError, match="cannot be inside"):
                validate_output_paths(raw_dir, out_dir)
        finally:
            if raw_dir.exists():
                raw_dir.rmdir()

    def test_symlink_session_dir_rejected_before_writes_in_adapt_cercus_to_nsmor(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        _create_synthetic_raw_pair(raw_dir, "session_001")

        staging = tmp_path / "staging"
        staging.mkdir()
        victim = tmp_path / "victim"
        victim.mkdir()

        symlink_session = staging / "session_001"
        try:
            os.symlink(victim, symlink_session)
        except OSError:
            pytest.skip("Symlink creation not supported on this filesystem")

        with pytest.raises(FileExistsError, match="cannot be a symlink"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=staging)

        assert len(list(victim.iterdir())) == 0, "Victim directory was contaminated"

    def test_is_protected_raw_path_matches_known_aliases(self):
        """Path-independent coverage: no mount or real data required."""
        assert is_protected_raw_path("/mnt/d/Data/train/all")
        assert is_protected_raw_path("/mnt/d/data/train/all")
        assert is_protected_raw_path("/mnt/d/Data/train/all/")
        assert is_protected_raw_path(r"D:\Data\train\all")
        assert is_protected_raw_path("D:/Data/train/all")
        assert is_protected_raw_path("/mnt/d/Data/train/all/some_session_kinematics.csv")
        assert is_protected_raw_path(r"D:\Data\train\all\sub\file.csv")
        assert not is_protected_raw_path("/mnt/c/Users/Wray/AppData/Local/Temp/scratch")
        assert not is_protected_raw_path("/mnt/d/Data/train/all_other")
        assert not is_protected_raw_path(None)
        assert not is_protected_raw_path("/tmp/synthetic_raw")

    def test_authoritative_raw_dir_in_place_prohibited(self):
        """Guard fires before any scan; PermissionError even if path is unmounted."""
        with pytest.raises(PermissionError, match="strictly prohibited"):
            adapt_cercus_to_nsmor(raw_dir="/mnt/d/Data/train/all")

        with pytest.raises(PermissionError, match="strictly prohibited"):
            adapt_cercus_to_nsmor(raw_dir="/mnt/d/data/train/all")

        k = "/mnt/d/Data/train/all/0.513cricket_001_20260707_193143_session_1_kinematics.csv"
        e = "/mnt/d/Data/train/all/0.513cricket_001_20260707_193143_session_1_events.csv"
        with pytest.raises(PermissionError, match="strictly prohibited"):
            adapt_session_pair(k, e, in_place=True)

    def test_authoritative_single_pair_in_place_prohibited(self, tmp_path: Path):
        """single_pair mode must not reach find_session_pairs / source reads."""
        k = "/mnt/d/Data/train/all/0.513cricket_001_20260707_193143_session_1_kinematics.csv"
        e = "/mnt/d/Data/train/all/0.513cricket_001_20260707_193143_session_1_events.csv"
        with pytest.raises(PermissionError, match="strictly prohibited"):
            adapt_cercus_to_nsmor(
                raw_dir=str(tmp_path),
                single_pair=(k, e),
            )

    def test_protected_guard_precedes_scan_wrong_exception_not_raised(self):
        """Regression: unmounted protected path must not yield FileNotFoundError.

        Pre-repair ordering called find_session_pairs first, which raised
        FileNotFoundError('Raw directory does not exist') when D: was absent.
        """
        with pytest.raises(PermissionError):
            adapt_cercus_to_nsmor(raw_dir="/mnt/d/Data/train/all")


class TestCanonicalConsumptionAndCli:
    """4) Produced schema is accepted by existing canonical loader or prepare_data

    pairing on temp staging; calibration/time units unchanged vs original transform
    for same valid fixture. Test actual command CLI --output_dir end-to-end.
    """

    def test_staging_consumed_by_pair_csv_and_loaders(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "staging"

        s1 = "0.513cricket_001_20260707_193143_session_1"
        _create_synthetic_raw_pair(raw_dir, s1, trial_id=1, lv_ratio_ms=100.0)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        pairs = pair_csv_files(out_dir)
        assert len(pairs) == 1
        kin_p, evt_p = pairs[0]

        df_k = load_kinematics_csv(kin_p)
        df_e = load_events_csv(evt_p)
        assert list(df_k.columns) == KINEMATICS_COLUMNS
        assert list(df_e.columns) == EVENT_COLUMNS

        concat = load_and_concat_sessions([kin_p], [evt_p])
        assert len(concat["kinematics"]) == len(df_k)
        assert len(concat["events"]) == len(df_e)

    def test_cli_output_dir_end_to_end(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_cli"
        raw_dir.mkdir()
        out_dir = tmp_path / "staging_cli"

        s1 = "0.513cricket_001_20260707_193143_session_1"
        _create_synthetic_raw_pair(raw_dir, s1, trial_id=5)

        cmd = [
            sys.executable,
            "scripts/pre_load_adapt.py",
            str(raw_dir),
            "--output_dir",
            str(out_dir),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, f"CLI failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

        target_kin = out_dir / s1 / f"{s1}_kinematics.csv"
        target_evt = out_dir / s1 / f"{s1}_events.csv"
        assert target_kin.is_file()
        assert target_evt.is_file()

    def test_cli_rejects_conflicting_output_dir_and_in_place(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "staging"
        _create_synthetic_raw_pair(raw_dir, "session_001")

        cmd = [
            sys.executable,
            "scripts/pre_load_adapt.py",
            str(raw_dir),
            "--output_dir",
            str(out_dir),
            "--in_place",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode != 0
        assert "Cannot specify both" in result.stderr

    def test_cli_prohibits_in_place_on_authoritative_raw_dir(self):
        cmd = [
            sys.executable,
            "scripts/pre_load_adapt.py",
            "/mnt/d/Data/train/all",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode != 0
        assert "strictly prohibited" in result.stderr

    def test_cli_legacy_in_place_default_is_documented_not_silent(self, tmp_path: Path):
        """Bare invocation keeps legacy in-place (CLI contract) but warns on stderr."""
        raw_dir = tmp_path / "raw_warn"
        raw_dir.mkdir()
        s1 = "session_001"
        k_path, _ = _create_synthetic_raw_pair(raw_dir, s1)

        cmd = [sys.executable, "scripts/pre_load_adapt.py", str(raw_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, f"CLI failed:\n{result.stdout}\n{result.stderr}"
        assert "WARNING" in result.stderr and "IN-PLACE" in result.stderr
        kin_header = k_path.read_text(encoding="utf-8").splitlines()[0]
        assert "session_id" in kin_header

    def test_cli_explicit_in_place_suppresses_warning(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_quiet"
        raw_dir.mkdir()
        _create_synthetic_raw_pair(raw_dir, "session_001")

        cmd = [sys.executable, "scripts/pre_load_adapt.py", str(raw_dir), "--in_place"]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, f"CLI failed:\n{result.stdout}\n{result.stderr}"
        assert "IN-PLACE" not in result.stderr

    def test_cli_prohibits_authoritative_even_without_in_place_flag(self):
        """Protected path must fail with PermissionError, not FileNotFoundError."""
        cmd = [sys.executable, "scripts/pre_load_adapt.py", "/mnt/d/Data/train/all"]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode != 0
        assert "strictly prohibited" in result.stderr
        assert "does not exist" not in result.stderr

    def test_cli_supports_explicit_in_place_flag_on_synthetic_raw(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_in_place"
        raw_dir.mkdir()
        s1 = "session_001"
        k_path, e_path = _create_synthetic_raw_pair(raw_dir, s1)

        cmd = [
            sys.executable,
            "scripts/pre_load_adapt.py",
            str(raw_dir),
            "--in_place",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, f"CLI failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

        # Verify converted header in-place
        kin_header = k_path.read_text(encoding="utf-8").splitlines()[0]
        assert "session_id" in kin_header and "time_ms" in kin_header

    def test_transform_staged_vs_inplace_self_consistency(self, tmp_path: Path):
        """Staged and legacy in-place modes of THIS codebase agree numerically.

        NOTE: this is a self-consistency check (same transform on both sides),
        not a check against a pre-existing base. Independent golden values are
        asserted in test_transform_known_calibration_golden_values.
        """
        raw_dir_legacy = tmp_path / "raw_legacy"
        raw_dir_legacy.mkdir()
        raw_dir_staged = tmp_path / "raw_staged"
        raw_dir_staged.mkdir()
        out_dir = tmp_path / "staging"

        s1 = "session_001"
        _create_synthetic_raw_pair(raw_dir_legacy, s1, trial_id=1, lv_ratio_ms=80.0, wind_state=1)
        _create_synthetic_raw_pair(raw_dir_staged, s1, trial_id=1, lv_ratio_ms=80.0, wind_state=1)

        adapt_cercus_to_nsmor(raw_dir=raw_dir_legacy)
        legacy_k = pd.read_csv(raw_dir_legacy / f"{s1}_kinematics.csv")
        legacy_e = pd.read_csv(raw_dir_legacy / f"{s1}_events.csv")

        adapt_cercus_to_nsmor(raw_dir=raw_dir_staged, output_dir=out_dir)
        staged_k = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")
        staged_e = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")

        num_cols = ["time_ms", "x_pos", "y_pos", "heading", "velocity", "acceleration", "visual_angle", "wind_state", "l_v_ratio"]
        for col in num_cols:
            np.testing.assert_allclose(
                legacy_k[col].to_numpy(),
                staged_k[col].to_numpy(),
                err_msg=f"Discrepancy in numeric column {col}",
            )
        assert (legacy_k["trial_id"].to_numpy() == staged_k["trial_id"].to_numpy()).all()

    def test_transform_known_calibration_golden_values(self, tmp_path: Path):
        """Independently expected calibration/time values from the raw fixture.

        Raw fixture (_create_synthetic_raw_pair defaults):
          n_samples=20, sys_time = arange(20)*0.004 s, dx=0.1, dy=0.05, dz=0.0,
          single trial, trial_type='baseline_visual', lv_ratio_ms=120.0.

        Expected (from the documented base physics, not from a previous run):
          abs_time_ms = sys_time*1000 -> step 4.0 ms; time_ms[0] = 0.
          heading = 0 (dz=0); x_pos = cumsum(0.1)/10 -> 0.01..0.20 cm-steps;
          y_pos = cumsum(0.05)/10.
          step_dist_mm(i>0) = sqrt(0.01^2 + 0.005^2)*10;
          velocity(i>0) = step_dist_mm / 0.004 / 10.
          visual_angle[0] = init_deg = 2.0 by construction of the looming formula.
        """
        raw_dir = tmp_path / "raw_gold"
        raw_dir.mkdir()
        out_dir = tmp_path / "out_gold"
        s1 = "session_001"
        _create_synthetic_raw_pair(raw_dir, s1, trial_id=1, lv_ratio_ms=120.0)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        k = pd.read_csv(out_dir / s1 / f"{s1}_kinematics.csv")

        # Time base: 4 ms sample period, t0 = 0
        assert k["time_ms"].iloc[0] == 0.0
        np.testing.assert_allclose(k["time_ms"].iloc[1], 4.0, atol=1e-9)

        # Position integration (dx=0.1, dy=0.05 per sample, /10 cm scale)
        np.testing.assert_allclose(k["x_pos"].iloc[0], 0.01, atol=1e-12)
        np.testing.assert_allclose(k["x_pos"].iloc[-1], 0.20, atol=1e-12)
        np.testing.assert_allclose(k["y_pos"].iloc[0], 0.005, atol=1e-12)

        # Velocity from documented step-distance / dt formula
        step_dist_mm = np.sqrt(0.01**2 + 0.005**2) * 10.0
        expected_v = step_dist_mm / 0.004 / 10.0
        np.testing.assert_allclose(k["velocity"].iloc[0], 0.0, atol=1e-9)
        np.testing.assert_allclose(k["velocity"].iloc[1], expected_v, atol=1e-9)

        # Acceleration: only the first velocity step is meaningfully non-zero
        np.testing.assert_allclose(k["acceleration"].iloc[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(k["acceleration"].iloc[2], 0.0, atol=1e-6)
        assert abs(k["acceleration"].iloc[1]) > 1.0

        # Visual angle starts at the documented init_deg=2.0
        np.testing.assert_allclose(k["visual_angle"].iloc[0], 2.0, atol=1e-9)
        np.testing.assert_allclose(k["l_v_ratio"].to_numpy(), 120.0, atol=1e-12)

        # Global trial identity preserved
        assert (k["trial_id"].to_numpy() == 1).all()
        assert (k["session_id"].astype(str).to_numpy() == s1).all()

    def test_null_ttc_preserved_in_event_value(self, tmp_path: Path):
        """Unimodal null target_ttc_ms stays null in canonical event_value JSON."""
        raw_dir = tmp_path / "raw_null_ttc"
        raw_dir.mkdir()
        out_dir = tmp_path / "staging_null_ttc"
        s1 = "session_001"
        _create_synthetic_raw_pair(
            raw_dir, s1, trial_type="baseline_wind", lv_ratio_ms=0.0, wind_state=1
        )

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        evt = pd.read_csv(out_dir / s1 / f"{s1}_events.csv")
        starts = evt[evt["event_type"] == "trial_start"]
        assert len(starts) >= 1
        payload = json.loads(starts.iloc[0]["event_value"])
        assert payload.get("target_ttc_ms") is None

    def test_retry_after_collision_is_deterministic(self, tmp_path: Path):
        """Second write without clearing prior output refuses; after explicit
        removal of OUR staged files and session directory, retry succeeds."""
        raw_dir = tmp_path / "raw_retry"
        raw_dir.mkdir()
        out_dir = tmp_path / "out_retry"
        out_dir.mkdir()
        s1 = "session_001"
        _create_synthetic_raw_pair(raw_dir, s1, trial_id=9)

        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        first_k = out_dir / s1 / f"{s1}_kinematics.csv"
        first_bytes = first_k.read_bytes()

        with pytest.raises(FileExistsError, match="already exists"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        assert first_k.read_bytes() == first_bytes

        first_k.unlink()
        (out_dir / s1 / f"{s1}_events.csv").unlink()
        # Even an empty old session directory keeps the staging root nonempty.
        with pytest.raises(FileExistsError, match="not empty"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        (out_dir / s1).rmdir()
        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        assert first_k.is_file()

    def test_interrupted_publish_cleans_tmp_and_leaves_no_partial_pair(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """If second-file publish fails, rollback removes the first published
        file and all tmp files created in this call."""
        raw_dir = tmp_path / "raw_int"
        raw_dir.mkdir()
        out_dir = tmp_path / "out_int"
        out_dir.mkdir()
        s1 = "session_001"
        _create_synthetic_raw_pair(raw_dir, s1)

        real_to_csv = pd.DataFrame.to_csv
        calls = {"n": 0}

        def flaky_to_csv(self, path, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:  # events tmp write fails
                raise RuntimeError("simulated interruption")
            return real_to_csv(self, path, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_csv", flaky_to_csv)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        s_dir = out_dir / s1
        leftovers = sorted(p.name for p in s_dir.iterdir()) if s_dir.is_dir() else []
        assert leftovers == [], f"partial/tmp leftovers: {leftovers}"


class TestDirectSeamGuardsAndNamingContract:
    """Direct seam (adapt_session_pair) path guards and strict naming contract."""

    def test_direct_seam_rejects_source_as_output(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        k, e = _create_synthetic_raw_pair(raw_dir, "session_001")

        with pytest.raises(ValueError, match="cannot be identical"):
            adapt_session_pair(k, e, output_session_dir=raw_dir)

    def test_direct_seam_rejects_nested_output_inside_source(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_sub = raw_dir / "staging_sub"
        k, e = _create_synthetic_raw_pair(raw_dir, "session_001")

        with pytest.raises(ValueError, match="cannot be nested inside"):
            adapt_session_pair(k, e, output_session_dir=out_sub)

    def test_direct_seam_rejects_both_output_dir_and_in_place(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        k, e = _create_synthetic_raw_pair(raw_dir, "session_001")

        with pytest.raises(ValueError, match="Cannot specify both"):
            adapt_session_pair(k, e, output_session_dir=out_dir, in_place=True)

    def test_direct_seam_collision_rejection(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        s1 = "session_001"
        k, e = _create_synthetic_raw_pair(raw_dir, s1)

        conflict = out_dir / f"{s1}_kinematics.csv"
        conflict.write_text("existing", encoding="utf-8")

        with pytest.raises(FileExistsError, match="already exists"):
            adapt_session_pair(k, e, output_session_dir=out_dir)

    def test_direct_seam_broken_symlink_collision_rejected(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_sym"
        raw_dir.mkdir()
        out_dir = tmp_path / "out_sym"
        out_dir.mkdir()
        s1 = "session_001"
        k, e = _create_synthetic_raw_pair(raw_dir, s1)

        broken_link = out_dir / f"{s1}_kinematics.csv"
        try:
            os.symlink(tmp_path / "nonexistent.csv", broken_link)
        except OSError:
            pytest.skip("Symlink creation not supported in this filesystem")

        with pytest.raises(FileExistsError, match="already exists"):
            adapt_session_pair(k, e, output_session_dir=out_dir)

    def test_direct_seam_rejects_symlink_session_dir(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        k, e = _create_synthetic_raw_pair(raw_dir, "session_001")

        staging = tmp_path / "staging"
        staging.mkdir()
        victim = tmp_path / "victim"
        victim.mkdir()

        symlink_dir = staging / "session_001"
        try:
            os.symlink(victim, symlink_dir)
        except OSError:
            pytest.skip("Symlink creation not supported on this filesystem")

        with pytest.raises(FileExistsError, match="cannot be a symlink"):
            adapt_session_pair(k, e, output_session_dir=symlink_dir)

        assert len(list(victim.iterdir())) == 0, "Victim directory was contaminated"

    def test_direct_seam_rejects_nested_case_alias_when_output_dir_does_not_exist_on_drvfs(self):
        drvfs_test = Path("/mnt/c/Users/Wray/AppData/Local/Temp")
        if not drvfs_test.is_dir():
            pytest.skip("DrvFs path not available")
        raw_dir = drvfs_test / "TEST_DRVFS_RAW_SEAM_CASE"
        raw_dir.mkdir(exist_ok=True)
        try:
            k, e = _create_synthetic_raw_pair(raw_dir, "session_001")
            out_session = drvfs_test / "test_drvfs_raw_seam_case" / "session_001"
            with pytest.raises(ValueError, match="cannot be nested inside"):
                adapt_session_pair(k, e, output_session_dir=out_session)
        finally:
            for f in raw_dir.iterdir():
                f.unlink()
            if raw_dir.exists():
                raw_dir.rmdir()

    def test_derive_session_id_and_extract_stem_accept_str(self):
        sid = derive_session_id(
            "/path/to/0.513cricket_001_20260707_193143_session_1_kinematics.csv",
            "/path/to/0.513cricket_001_20260707_193143_session_1_events.csv",
        )
        assert sid == "0.513cricket_001_20260707_193143_session_1"

        stem, role = extract_file_stem("dummy_kinematics.csv")
        assert stem == "dummy"
        assert role == "kinematics"

    def test_naming_contract_rejects_malformed_filenames(self, tmp_path: Path):
        # Non-conforming suffix
        bad_file = tmp_path / "session_001_kinematics_corrupted.csv"
        bad_file.write_text("dummy", encoding="utf-8")
        with pytest.raises(ValueError, match="naming contract"):
            extract_file_stem(bad_file)

        # Bare filename in root flat directory
        flat_bare = tmp_path / "kinematics.csv"
        flat_bare.write_text("dummy", encoding="utf-8")
        with pytest.raises(ValueError, match="flat directory"):
            derive_session_id(flat_bare, root_dir=tmp_path)


class TestSessionIdContainment:
    """P1: session_id must be a single safe path component and match the
    source stem contract; traversal must never escape the output staging dir."""

    def _pair(self, raw_dir: Path) -> Tuple[Path, Path]:
        raw_dir.mkdir(parents=True, exist_ok=True)
        return _create_synthetic_raw_pair(raw_dir, "session_001")

    @pytest.mark.parametrize(
        "evil_sid",
        [
            "../raw/evil",
            "./raw/evil",
            "/abs/victim/evil2",
            "..",
            ".",
            "a/b",
            "a\\b",
            "C:evil",
            "",
        ],
    )
    def test_rejects_unsafe_session_id_on_direct_pair(self, tmp_path: Path, evil_sid: str):
        raw_dir = tmp_path / "raw"
        k, e = self._pair(raw_dir)
        out_dir = tmp_path / "staging"
        out_dir.mkdir()
        with pytest.raises(ValueError):
            adapt_session_pair(k, e, session_id=evil_sid, output_session_dir=out_dir)
        assert sorted(x.name for x in raw_dir.iterdir()) == sorted(
            ["session_001_kinematics.csv", "session_001_events.csv"]
        )
        assert list(out_dir.iterdir()) == []

    def test_rejects_session_id_that_does_not_match_source_stem(self, tmp_path: Path):
        raw_dir = tmp_path / "raw"
        k, e = self._pair(raw_dir)
        out_dir = tmp_path / "staging"
        out_dir.mkdir()
        with pytest.raises(ValueError, match="stem"):
            adapt_session_pair(
                k, e, session_id="other_name", output_session_dir=out_dir
            )
        assert list(out_dir.iterdir()) == []

    def test_legitimate_decimal_dotted_session_id_accepted(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_dotted"
        raw_dir.mkdir()
        s1 = "0.513cricket_001_20260707_193143_session_1"
        _create_synthetic_raw_pair(raw_dir, s1)
        out_dir = tmp_path / "out_dotted"
        report = adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
        assert report["adapted_sessions"] == [s1]
        assert (out_dir / s1 / f"{s1}_kinematics.csv").is_file()

    def test_batch_single_pair_rejects_traversal_sid(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_b"
        k, e = self._pair(raw_dir)
        out_dir = tmp_path / "out_b"
        with pytest.raises(ValueError):
            adapt_cercus_to_nsmor(
                raw_dir=raw_dir,
                output_dir=out_dir,
                single_pair=(k, e),
                session_id="../p1/evil3",
            )
        assert not (tmp_path / "p1").exists()
        assert sorted(x.name for x in raw_dir.iterdir()) == sorted(
            ["session_001_kinematics.csv", "session_001_events.csv"]
        )

    def test_batch_single_pair_rejects_mismatched_sid(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_b2"
        k, e = self._pair(raw_dir)
        out_dir = tmp_path / "out_b2"
        with pytest.raises(ValueError, match="stem"):
            adapt_cercus_to_nsmor(
                raw_dir=raw_dir,
                output_dir=out_dir,
                single_pair=(k, e),
                session_id="custom_sid",
            )


class TestExclusiveTempAndNoReplacePublish:
    """P2/P3/P5: predictable tmp names must not be followed; publish must be
    no-replace; intermediate symlink components must be rejected."""

    def test_predicable_tmp_symlink_is_not_followed(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_tmp"
        raw_dir.mkdir()
        out_dir = tmp_path / "out_tmp"
        out_dir.mkdir()
        s1 = "session_001"
        _create_synthetic_raw_pair(raw_dir, s1)

        victim = tmp_path / "victim.txt"
        victim.write_text("precious-data", encoding="utf-8")
        victim_hash = _file_sha256(victim)

        s_dir = out_dir / s1
        s_dir.mkdir()
        planted = s_dir / f"{s1}_kinematics.csv.tmp"
        try:
            os.symlink(victim, planted)
        except OSError:
            pytest.skip("Symlink creation not supported on this filesystem")

        # This direct pair API still supports safe publication into a session
        # directory containing unrelated files; batch roots must be empty.
        adapt_session_pair(
            raw_dir / f"{s1}_kinematics.csv",
            raw_dir / f"{s1}_events.csv",
            output_session_dir=s_dir,
        )
        assert _file_sha256(victim) == victim_hash, "victim destroyed via tmp symlink"
        assert victim.read_text(encoding="utf-8") == "precious-data"
        assert (s_dir / f"{s1}_kinematics.csv").is_file()
        assert (s_dir / f"{s1}_events.csv").is_file()

    def test_publish_no_replace_refuses_racer_and_preserves_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        raw_dir = tmp_path / "raw_race"
        raw_dir.mkdir()
        out_dir = tmp_path / "out_race"
        out_dir.mkdir()
        s1 = "session_001"
        _create_synthetic_raw_pair(raw_dir, s1)

        s_dir = out_dir / s1
        target = s_dir / f"{s1}_kinematics.csv"
        racer_content = "RACER-PREEXISTING"

        real_link = os.link

        def racing_link(src, dst, *args, **kwargs):
            if Path(dst) == target and not Path(dst).exists():
                Path(dst).write_text(racer_content, encoding="utf-8")
            return real_link(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "link", racing_link)

        with pytest.raises(FileExistsError):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        assert target.read_text(encoding="utf-8") == racer_content, (
            "racer file was clobbered"
        )
        assert not (s_dir / f"{s1}_events.csv").exists()

    def test_intermediate_symlink_component_rejected(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_sym_mid"
        raw_dir.mkdir()
        k, e = _create_synthetic_raw_pair(raw_dir, "session_001")

        staging = tmp_path / "staging"
        staging.mkdir()
        victim = tmp_path / "victim_mid"
        victim.mkdir()
        link = staging / "link"
        try:
            os.symlink(victim, link)
        except OSError:
            pytest.skip("Symlink creation not supported on this filesystem")

        out_session = link / "sub"
        with pytest.raises(FileExistsError, match="symlink"):
            adapt_session_pair(k, e, output_session_dir=out_session)
        assert list(victim.iterdir()) == [], "victim contaminated via intermediate symlink"

    def test_batch_intermediate_symlink_rejected(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_sym_b"
        raw_dir.mkdir()
        _create_synthetic_raw_pair(raw_dir, "session_001")
        staging = tmp_path / "staging_b"
        staging.mkdir()
        victim = tmp_path / "victim_b"
        victim.mkdir()
        link = staging / "link"
        try:
            os.symlink(victim, link)
        except OSError:
            pytest.skip("Symlink creation not supported on this filesystem")

        with pytest.raises(FileExistsError, match="symlink"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=link)
        assert list(victim.iterdir()) == []


class TestSamefileFailClosed:
    """P4: samefile PermissionError/OSError must fail CLOSED, not disable
    DrvFs case-alias protection."""

    def test_validate_output_paths_fails_closed_on_samefile_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        def broken_samefile(a, b):
            raise PermissionError("simulated ACL denial")

        monkeypatch.setattr(os.path, "samefile", broken_samefile)
        with pytest.raises(ValueError, match="failing closed"):
            validate_output_paths(raw_dir, out_dir)

    def test_adapt_session_pair_fails_closed_on_samefile_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        k, e = _create_synthetic_raw_pair(raw_dir, "session_001")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        def broken_samefile(a, b):
            raise PermissionError("simulated ACL denial")

        monkeypatch.setattr(os.path, "samefile", broken_samefile)
        with pytest.raises(ValueError, match="failing closed"):
            adapt_session_pair(k, e, output_session_dir=out_dir)

    def test_samefile_permissionerror_does_not_enable_case_alias_nesting(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """With samefile broken, nesting rejection must still fail closed."""
        drvfs_test = Path("/mnt/c/Users/Wray/AppData/Local/Temp")
        if not drvfs_test.is_dir():
            pytest.skip("DrvFs path not available")
        raw_dir = drvfs_test / "TEST_SAMEFILE_FAILCLOSED_RAW"
        raw_dir.mkdir(exist_ok=True)
        try:
            k, e = _create_synthetic_raw_pair(raw_dir, "session_001")

            def broken_samefile(a, b):
                raise PermissionError("simulated ACL denial")

            monkeypatch.setattr(os.path, "samefile", broken_samefile)
            out_session = drvfs_test / "test_samefile_failclosed_raw" / "session_001"
            with pytest.raises(ValueError, match="failing closed"):
                adapt_session_pair(k, e, output_session_dir=out_session)
        finally:
            for f in raw_dir.iterdir():
                f.unlink()
            if raw_dir.exists():
                raw_dir.rmdir()


class TestRollbackOwnership:
    """P6: rollback must never delete a file another writer replaced."""

    def test_rollback_preserves_competing_replacement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        raw_dir = tmp_path / "raw_rb"
        raw_dir.mkdir()
        out_dir = tmp_path / "out_rb"
        out_dir.mkdir()
        s1 = "session_001"
        _create_synthetic_raw_pair(raw_dir, s1)

        s_dir = out_dir / s1
        target_k = s_dir / f"{s1}_kinematics.csv"
        target_e = s_dir / f"{s1}_events.csv"
        rival_content = "RIVAL-REPLACEMENT-CONTENT"

        real_link = os.link
        state = {"racer_installed": False}

        def failing_link(src, dst, *args, **kwargs):
            if Path(dst) == target_e:
                if target_k.is_file():
                    cur = target_k.read_text(encoding="utf-8")
                    if cur != rival_content:
                        os.unlink(target_k)
                        target_k.write_text(rival_content, encoding="utf-8")
                        state["racer_installed"] = True
                raise OSError(28, "No space left on device")
            return real_link(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "link", failing_link)

        with pytest.raises(OSError):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        assert target_k.is_file(), "rollback deleted the competing replacement"
        assert target_k.read_text(encoding="utf-8") == rival_content
        assert not target_e.exists()
        leftovers = [x for x in s_dir.iterdir() if x.name.endswith(".tmp")]
        assert leftovers == []

    def test_rollback_requires_ctime_identity(self, tmp_path: Path) -> None:
        """Ownership proof must include st_ctime_ns, not just (dev, ino).

        On Linux, unlinking and recreating a path reuses the inode number, so
        a competing replacement can share ``(st_dev, st_ino)`` with the file
        this process created.  A stale ``st_ctime_ns`` marks that file as
        foreign and it must be preserved; only an exact dev+ino+ctime match
        authorizes unlinking.
        """
        import scripts.pre_load_adapt as adapter

        foreign = tmp_path / "foreign.tmp"
        foreign.write_text("foreign", encoding="utf-8")
        st = os.stat(foreign)
        adapter._rollback_owned([(foreign, st.st_dev, st.st_ino, st.st_ctime_ns - 1)], [])
        assert foreign.is_file(), "rollback deleted a file with a stale ctime"
        adapter._rollback_owned([(foreign, st.st_dev, st.st_ino, st.st_ctime_ns)], [])
        assert not foreign.exists(), "rollback failed to remove an owned file"

        published = tmp_path / "published.csv"
        published.write_text("mine", encoding="utf-8")
        pst = os.stat(published)
        adapter._rollback_owned([], [(published, pst.st_dev, pst.st_ino, pst.st_ctime_ns - 1)])
        assert published.is_file(), "rollback deleted a published file with a stale ctime"
        adapter._rollback_owned([], [(published, pst.st_dev, pst.st_ino, pst.st_ctime_ns)])
        assert not published.exists(), "rollback failed to remove an owned published file"


class TestMultiSessionPartialFailure:
    """P7: multi-session partial batch failure reporting regression."""

    def test_second_session_failure_keeps_first_and_reports_truthfully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ):
        raw_dir = tmp_path / "raw_multi"
        raw_dir.mkdir()
        out_dir = tmp_path / "out_multi"
        s1 = "0.100cricket_a_session_1"
        s2 = "0.200cricket_b_session_1"
        _create_synthetic_raw_pair(raw_dir, s1, trial_id=1)
        _create_synthetic_raw_pair(raw_dir, s2, trial_id=2)

        real_to_csv = pd.DataFrame.to_csv
        calls = {"n": 0}

        def flaky_to_csv(self, path, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("simulated second-session failure")
            return real_to_csv(self, path, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_csv", flaky_to_csv)

        with pytest.raises(RuntimeError, match="simulated second-session failure"):
            adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)

        assert (out_dir / s1 / f"{s1}_kinematics.csv").is_file()
        assert (out_dir / s1 / f"{s1}_events.csv").is_file()

        s2_dir = out_dir / s2
        if s2_dir.is_dir():
            leftovers = list(s2_dir.iterdir())
            assert leftovers == [], f"partial leftovers: {leftovers}"

        captured = capsys.readouterr()
        assert "Absolute alignment complete" not in captured.out


class TestStagingInPlaceAliasSafety:
    """Safe atomic replacement under in-place mode protects hard link aliases."""

    def test_inplace_hardlink_alias_preserved(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_alias"
        raw_dir.mkdir()
        s1 = "0.100cricket_session_1"
        k_path, e_path = _create_synthetic_raw_pair(raw_dir, s1, trial_id=1)

        victim = tmp_path / "independent_hardlink.csv"
        os.link(k_path, victim)
        original_bytes = victim.read_bytes()

        adapt_session_pair(k_path, e_path, in_place=True)

        # Victim must not have been mutated in-place
        assert victim.read_bytes() == original_bytes, (
            "In-place adaptation mutated underlying inode of hardlink alias"
        )
        # But k_path itself was adapted to canonical schema
        df_adapted = pd.read_csv(k_path)
        assert "time_ms" in df_adapted.columns
        assert "x_pos" in df_adapted.columns


class TestRollbackTemporaryOwnership:
    """Rollback only unlinks temporaries with matching inode ownership."""

    def test_rollback_preserves_reused_temporary_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        raw_dir = tmp_path / "raw_temp"
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_dir = tmp_path / "out_temp"
        s1 = "0.100cricket_session_1"
        k_path, e_path = _create_synthetic_raw_pair(raw_dir, s1, trial_id=1)

        import scripts.pre_load_adapt as adapter
        real_link = os.link
        released: List[Path] = []
        marker = "independent-other-writer-content"

        def racing_link(src, dst, *args, **kwargs):
            if str(dst).endswith("_kinematics.csv"):
                released.append(Path(src))
                return real_link(src, dst, *args, **kwargs)
            # Recreate a new file at the released temporary path
            assert released and not released[0].exists()
            with released[0].open("x") as fh:
                fh.write(marker)
            raise OSError(28, "injected second-publish failure")

        monkeypatch.setattr(adapter.os, "link", racing_link)
        with pytest.raises(OSError, match="second-publish"):
            adapter.adapt_session_pair(k_path, e_path, output_session_dir=out_dir)

        assert released[0].is_file(), "Rival file was improperly deleted by rollback"
        assert released[0].read_text() == marker


class TestDirectorySwapSafety:
    """Directory swap to symlink before publication is intercepted."""

    def test_directory_swap_caught_before_publish(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        raw_dir = tmp_path / "raw_swap"
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_dir = tmp_path / "out_swap"
        s1 = "0.100cricket_session_1"
        k_path, e_path = _create_synthetic_raw_pair(raw_dir, s1, trial_id=1)
        out_dir.mkdir()
        victim = raw_dir / "victim_dir"
        victim.mkdir()

        before_raw = sorted(p.name for p in raw_dir.iterdir())
        import scripts.pre_load_adapt as adapter
        real_mkstemp = adapter.tempfile.mkstemp
        swapped = False

        def racing_mkstemp(*args, **kwargs):
            nonlocal swapped
            if not swapped:
                out_dir.rename(tmp_path / "parked_output")
                out_dir.symlink_to(victim, target_is_directory=True)
                swapped = True
            return real_mkstemp(*args, **kwargs)

        monkeypatch.setattr(adapter.tempfile, "mkstemp", racing_mkstemp)
        with pytest.raises((OSError, ValueError, FileExistsError)):
            adapter.adapt_session_pair(k_path, e_path, output_session_dir=out_dir)

        after_raw = sorted(p.name for p in raw_dir.iterdir())
        assert after_raw == before_raw, "Files were leaked into source directory via symlink swap"

