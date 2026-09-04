"""Tests for default dt_ms consistency across production code paths.

Validates that residual 10.0 ms defaults from pre-v2.1 / 100 Hz legacy
code paths have been completely migrated to hardware-nominal 4.0 ms (250 Hz):
1. Function signatures in scripts/prepare_data.py have default dt_ms=4.0
2. CLI parsers in scripts/prepare_data.py and scripts/prepare_metadata.py default to 4.0
3. Configuration defaults in config/default.yaml and ModelConfig default to 4.0
4. AST / text scanning over production data preparation code paths confirms no 10.0 ms defaults
5. Pure-wind padding calculation under 4.0 ms gives 1425 frames (5.7 s @ 250 Hz)
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import List, Tuple

import pytest
import yaml

from nsmor.config_parser import ModelConfig
from nsmor.data_extractor import _compute_pure_wind_prepend_frames
import scripts.prepare_data as prep_data

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SCRIPTS = [
    REPO_ROOT / "scripts" / "prepare_data.py",
    REPO_ROOT / "scripts" / "prepare_metadata.py",
]


def test_prepare_data_function_signatures():
    """All function signatures in prepare_data.py must default dt_ms to 4.0, never 10.0."""
    target_functions = [
        prep_data.prepare_dataset,
        prep_data.reconstruct_visual_looming,
        prep_data.reconstruct_trial_visual_features,
        prep_data.apply_hardware_time_correction,
    ]

    for fn in target_functions:
        sig = inspect.signature(fn)
        assert "dt_ms" in sig.parameters, f"{fn.__name__} missing dt_ms parameter"
        param = sig.parameters["dt_ms"]
        assert param.default != 10.0, (
            f"{fn.__name__} has residual 10.0 ms default for dt_ms"
        )
        assert param.default == 4.0, (
            f"{fn.__name__} expected dt_ms=4.0, got {param.default}"
        )


def test_prepare_data_cli_defaults():
    """prepare_data.py CLI argument --dt_ms must default to 4.0, never 10.0."""
    parser = prep_data.build_parser()
    action = next((a for a in parser._actions if "--dt_ms" in a.option_strings), None)
    assert action is not None, "--dt_ms action not found in prepare_data CLI parser"
    assert action.default != 10.0, "--dt_ms action has residual default 10.0 in prepare_data"
    assert action.default == 4.0, f"--dt_ms expected default 4.0, got {action.default}"


def test_prepare_metadata_cli_defaults():
    """prepare_metadata.py CLI argument --dt_ms must default to 4.0, never 10.0."""
    metadata_script = REPO_ROOT / "scripts" / "prepare_metadata.py"
    source = metadata_script.read_text(encoding="utf-8")
    tree = ast.parse(source)

    dt_ms_defaults = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            arg_names = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
            if "--dt_ms" in arg_names:
                for kw in node.keywords:
                    if kw.arg == "default":
                        if isinstance(kw.value, ast.Constant):
                            dt_ms_defaults.append(kw.value.value)

    assert dt_ms_defaults, "No --dt_ms default found in prepare_metadata.py"
    for d in dt_ms_defaults:
        assert d != 10.0, "prepare_metadata.py CLI has residual default 10.0 for --dt_ms"
        assert d == 4.0, f"prepare_metadata.py expected default 4.0 for --dt_ms, got {d}"


def test_config_yaml_dt_ms():
    """config/default.yaml must configure model.dt_ms as 4.0, never 10.0."""
    yaml_path = REPO_ROOT / "config" / "default.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "model" in cfg, "model section missing in config/default.yaml"
    assert cfg["model"].get("dt_ms") != 10.0, "config/default.yaml has model.dt_ms=10.0"
    assert cfg["model"].get("dt_ms") == 4.0, (
        f"config/default.yaml model.dt_ms expected 4.0, got {cfg['model'].get('dt_ms')}"
    )


def test_model_config_dataclass_default():
    """nsmor.config_parser.ModelConfig dataclass must default dt_ms to 4.0, never 10.0."""
    cfg = ModelConfig()
    assert cfg.dt_ms != 10.0, "ModelConfig has residual 10.0 ms default"
    assert cfg.dt_ms == 4.0, f"ModelConfig expected dt_ms=4.0, got {cfg.dt_ms}"


def test_ast_scan_production_scripts_no_10ms_defaults():
    """AST scanner: inspect all function definitions and add_argument calls in production scripts."""
    for script_path in PRODUCTION_SCRIPTS:
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script_path))

        # Check function parameter defaults
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Match args with defaults
                args = node.args.args
                defaults = node.args.defaults
                # defaults align with the end of args
                offset = len(args) - len(defaults)
                for i, default_node in enumerate(defaults):
                    arg = args[offset + i]
                    if arg.arg == "dt_ms" and isinstance(default_node, ast.Constant):
                        assert default_node.value != 10.0, (
                            f"{script_path.name}:{node.name} has residual dt_ms=10.0 default"
                        )
                        assert default_node.value == 4.0, (
                            f"{script_path.name}:{node.name} expected dt_ms=4.0, got {default_node.value}"
                        )

            # Check parser.add_argument("--dt_ms", ..., default=...)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
                str_args = [
                    a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
                ]
                if "--dt_ms" in str_args:
                    for kw in node.keywords:
                        if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                            assert kw.value.value != 10.0, (
                                f"{script_path.name} add_argument('--dt_ms') has residual default 10.0"
                            )
                            assert kw.value.value == 4.0, (
                                f"{script_path.name} add_argument('--dt_ms') expected default 4.0, got {kw.value.value}"
                            )


def test_regex_grep_production_scripts_no_10ms_literals():
    """Regex search: ensure no dt_ms: float = 10.0 or --dt_ms 10.0 in prepare_data.py."""
    prep_data_path = REPO_ROOT / "scripts" / "prepare_data.py"
    content = prep_data_path.read_text(encoding="utf-8")

    # Check for parameter default assignments
    param_matches = re.findall(r"dt_ms\s*:\s*float\s*=\s*10(?:\.0)?", content)
    assert not param_matches, (
        f"Found residual dt_ms=10 parameter defaults in prepare_data.py: {param_matches}"
    )

    # Check for CLI flag examples
    cli_matches = re.findall(r"--dt_ms\s+10(?:\.0)?", content)
    assert not cli_matches, (
        f"Found residual --dt_ms 10 CLI references in prepare_data.py: {cli_matches}"
    )

    # Check for docstring defaults like 'default 10ms'
    doc_matches = re.findall(r"default\s+10\s*ms", content, re.IGNORECASE)
    assert not doc_matches, (
        f"Found residual 'default 10ms' in prepare_data.py docstrings: {doc_matches}"
    )


def test_pure_wind_prepend_frames_under_4ms():
    """Pure-wind padding calculation under 4.0 ms must equal 1425 frames, not 570."""
    frames_4ms = _compute_pure_wind_prepend_frames(4.0)
    assert frames_4ms == 1425, (
        f"Expected 1425 frames (5.7s / 4.0ms), got {frames_4ms}"
    )

    frames_10ms_legacy = _compute_pure_wind_prepend_frames(10.0)
    assert frames_10ms_legacy == 570, (
        f"Legacy 10.0ms should produce 570 frames, got {frames_10ms_legacy}"
    )
