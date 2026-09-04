"""Test CLI entrypoints and bug fixes."""
import subprocess
import sys


def test_simulate_psychophysics_help_no_unicode_error():
    """simulate_psychophysics.py --help must not crash on cp1252 with Greek sigma."""
    res = subprocess.run(
        [sys.executable, "scripts/simulate_psychophysics.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Failed with: {res.stderr}"
    assert "--noise_levels" in res.stdout


def test_convert_metadata_to_etl_cli():
    """convert_metadata_to_etl.py must have an argparse CLI with --input and --output."""
    res = subprocess.run(
        [sys.executable, "scripts/convert_metadata_to_etl.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Failed with: {res.stderr}"
    assert "--input" in res.stdout
    assert "--output" in res.stdout
