"""CLI entry points must import when run directly, not just under pytest.

``pyproject.toml`` installs ``nsmor*`` and NOT ``scripts``, so ``scripts``
is importable only when the repo root happens to be on ``sys.path``.
Pytest puts it there; ``python scripts/train.py`` does not -- running a
file directly puts *its own directory* on ``sys.path[0]``.

A cross-script import such as ``from scripts.make_subset_dataset import x``
therefore passes the whole test suite while breaking every documented
invocation, including the Makefile's ``$(PYTHON) scripts/train.py``.  That
happened, and no test caught it because CI runs ``make test`` and never
executes a CLI.

``--help`` is deliberate: argparse exits 0 before any real work, so this
exercises module-level imports without touching a corpus, a GPU, or the
filesystem.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Scripts with an argparse CLI that must be runnable directly.
_ENTRY_POINTS = [
    "analyze_gating.py",
    "make_subset_dataset.py",
    "train.py",
]


@pytest.mark.parametrize("script", _ENTRY_POINTS)
def test_entry_point_imports_when_run_directly(script: str) -> None:
    """``python scripts/<name>.py --help`` must exit 0."""
    path = _REPO_ROOT / "scripts" / script
    assert path.exists(), f"missing entry point: {path}"

    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"`python scripts/{script} --help` exited {result.returncode}.\n"
        f"This usually means a module-level import is unreachable during "
        f"direct execution -- most often `from scripts.X import Y`, which "
        f"only resolves when the repo root is on sys.path. Move the shared "
        f"symbol into the installed nsmor package instead.\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )


def test_no_script_imports_the_scripts_package() -> None:
    """Guard the bug class, not just the three known entry points.

    ``scripts`` is not installed, so importing it from inside another
    script is unreachable during direct execution.  Shared symbols belong
    in ``nsmor.*``.
    """
    offenders = []
    for path in sorted((_REPO_ROOT / "scripts").glob("*.py")):
        # Parse rather than grep: docstrings legitimately show programmatic
        # usage such as ``from scripts.train import train``, and a textual
        # scan cannot tell that apart from a real import.
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "scripts" or module.startswith("scripts."):
                    offenders.append(
                        f"{path.name}:{node.lineno}: from {module} import ..."
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "scripts" or alias.name.startswith(
                        "scripts."
                    ):
                        offenders.append(
                            f"{path.name}:{node.lineno}: import {alias.name}"
                        )

    assert not offenders, (
        "scripts/ must not import the `scripts` package -- it is not "
        "installed (pyproject includes only `nsmor*`), so these imports "
        "break `python scripts/<name>.py` while still passing pytest.\n"
        + "\n".join(offenders)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
