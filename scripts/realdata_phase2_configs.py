"""Generate the real-data PHASE-2 arm configs and their resolution diff.

PROSPECTIVE: this script only WRITES the frozen per-arm YAMLs and the
config-resolution evidence for `docs/realdata-phase2-protocol-20261010.md`.
It trains nothing and modifies no protected path.

Arms (all: A1's resolved config, swiglu/off, persistence_skip 0 unless stated,
same dataset/nested prior/split, 300 epochs, patience 20, plus
`checkpoint.selection_metric: mse`):

    R0  control              (A1 config + MSE selection)
    R1  residual-over-persistence  model.persistence_skip = 1.0
    R2  history-ablated      data.zero_input_channels = [2, 3]
    R3  jerk ablation        loss.lambda_jerk = 0.0

Seeds 42, 43, 44 -> 12 configs under `config/realdata-phase2-20261010/`.

The resolution diff resolves every arm through `scripts.train.build_config`
and compares it leaf-by-leaf against A1's resolved config
(`config/realdata-preliminary-20261009/A1-swiglu-off.yaml`), proving that ONLY
the declared keys differ.  Usage::

    python scripts/realdata_phase2_configs.py            # write everything
    python scripts/realdata_phase2_configs.py --check    # verify, no writes
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "config" / "realdata-phase2-20261010"
A1_YAML = REPO / "config" / "realdata-preliminary-20261009" / "A1-swiglu-off.yaml"
DIFF_PATH = OUT_DIR / "arm-resolution-diff.json"
SEEDS = (42, 43, 44)

# Arm id -> nested override dict applied on top of the A1 config, plus the
# human-readable arm keys bound by the scorer / protocol.
#
# The two phase-2 controls (``selection_metric``, ``zero_input_channels``)
# are NOT part of the protected ``nsmor/config_parser.py`` schema.  They are
# declared in a script-owned top-level ``phase2:`` YAML block consumed by
# ``scripts/train.py::build_config``, so ``ExperimentConfig.to_dict()`` (and
# hence every existing config / checkpoint) stays byte-unchanged.
ARMS: Dict[str, Dict[str, Any]] = {
    "R0-control": {},
    "R1-residual-persistence": {"model": {"persistence_skip": 1.0}},
    "R2-history-ablated": {},
    "R3-jerk-ablated": {"loss": {"lambda_jerk": 0.0}},
}

# Per-arm script-owned phase-2 block.  Every arm selects on masked MSE; R2
# additionally zeroes the lagged velocity/acceleration sensory channels.
PHASE2: Dict[str, Dict[str, Any]] = {
    "R0-control": {"selection_metric": "mse", "zero_input_channels": []},
    "R1-residual-persistence": {"selection_metric": "mse",
                                "zero_input_channels": []},
    "R2-history-ablated": {"selection_metric": "mse",
                           "zero_input_channels": [2, 3]},
    "R3-jerk-ablated": {"selection_metric": "mse", "zero_input_channels": []},
}

# The declared arm-identity keys the scorer must bind (never the byte-identical
# leaves).  Values are the arm's intended resolved values.  These MUST equal
# `scripts.evaluate_realdata_phase2.EXPECTED_ARM_KEYS`; the selection_metric
# leaf (mse for every arm) is bound separately by the scorer.  The non-arm
# leaf ``lambda_jerk`` is derived from the A1 REFERENCE config (single source),
# so it cannot drift silently if A1's loss changes; only the arm-specific
# ``persistence_skip`` / ``zero_input_channels`` are literals here.
def _a1_lambda_jerk() -> float:
    raw = yaml.safe_load(A1_YAML.read_text(encoding="utf-8"))
    return float(raw["loss"]["lambda_jerk"])


_A1_LAMBDA_JERK = _a1_lambda_jerk()

ARM_KEYS: Dict[str, Dict[str, Any]] = {
    "R0-control": {"persistence_skip": 0.0,
                   "zero_input_channels": [], "lambda_jerk": _A1_LAMBDA_JERK},
    "R1-residual-persistence": {"persistence_skip": 1.0,
                                "zero_input_channels": [], "lambda_jerk": _A1_LAMBDA_JERK},
    "R2-history-ablated": {"persistence_skip": 0.0,
                           "zero_input_channels": [2, 3], "lambda_jerk": _A1_LAMBDA_JERK},
    "R3-jerk-ablated": {"persistence_skip": 0.0,
                        "zero_input_channels": [], "lambda_jerk": 0.0},
}


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _arm_yaml_dict(arm: str, seed: int) -> Dict[str, Any]:
    base = yaml.safe_load(A1_YAML.read_text(encoding="utf-8"))
    cfg = _deep_merge(base, ARMS[arm])
    cfg["training"]["random_seed"] = int(seed)
    cfg["checkpoint"]["output_dir"] = (
        f".scratch/realdata-phase2-20261010/{arm}-seed{seed}"
    )
    # Script-owned phase-2 block (outside the protected config schema).
    cfg["phase2"] = dict(PHASE2[arm])
    return cfg


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _write_configs() -> List[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for arm in ARMS:
        for seed in SEEDS:
            path = OUT_DIR / f"{arm}-seed{seed}.yaml"
            path.write_text(
                yaml.dump(_arm_yaml_dict(arm, seed),
                          default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
            written.append(path)
    return written


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_resolution_diff() -> Dict[str, Any]:
    # Import here so the module is import-free (torch) until actually run.
    # ``scripts/`` must NOT import the ``scripts`` package (it is not
    # installed); a sibling import works because ``scripts/`` is on
    # ``sys.path`` when the file is run as ``python scripts/<name>.py``.
    import sys
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    from train import build_config  # type: ignore

    import train  # type: ignore

    a1_cfg, _, _ = build_config(["--config", str(A1_YAML)])
    a1_flat = _flatten(a1_cfg.to_dict())
    arms_out: Dict[str, Any] = {}
    resolution: Dict[str, Any] = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = OUT_DIR / f"{arm}-seed{seed}.yaml"
            cfg, _, _ = build_config(["--config", str(path)])
            flat = _flatten(cfg.to_dict())
            changed = sorted(
                [
                    [k, a1_flat.get(k, "<absent>"), flat[k]]
                    for k in flat
                    if k in a1_flat and flat[k] != a1_flat[k]
                ],
                key=lambda row: row[0],
            )
            added = sorted(
                [[k, flat[k]] for k in flat if k not in a1_flat],
                key=lambda row: row[0],
            )
            missing = sorted([k for k in a1_flat if k not in flat])
            # The script-owned phase-2 controls resolve to module state in
            # ``scripts.train`` (never into ``to_dict()``); capture and
            # cross-check them here so the diff proves the CLI/YAML path
            # actually produced the declared arm.
            resolved = {
                "selection_metric": train._SELECTION_METRIC,
                "zero_input_channels": list(train._ZERO_INPUT_CHANNELS),
            }
            declared = {
                "selection_metric": PHASE2[arm]["selection_metric"],
                "zero_input_channels": list(PHASE2[arm]["zero_input_channels"]),
            }
            assert resolved == declared, (
                f"{arm}-seed{seed} resolved {resolved} != declared {declared}"
            )
            key = f"{arm}-seed{seed}"
            resolution[key] = resolved
            arms_out[key] = {
                "arm": arm,
                "seed": int(seed),
                "arm_keys": ARM_KEYS[arm],
                "phase2_resolved": resolved,
                "changed_leaves": changed,
                "added_leaves": added,
                "missing_leaves": missing,
                "yaml_sha256": _sha256(path),
                "yaml_path": str(path.relative_to(REPO)),
            }
    return {
        "reference": str(A1_YAML.relative_to(REPO)),
        "reference_sha256": _sha256(A1_YAML),
        "seeds": list(SEEDS),
        "arms": arms_out,
    }


def main(argv: Optional[Tuple[str, ...]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Verify configs/diff are current; write nothing.")
    args = parser.parse_args(argv)

    if args.check:
        for arm in ARMS:
            for seed in SEEDS:
                path = OUT_DIR / f"{arm}-seed{seed}.yaml"
                assert path.exists(), f"missing {path}"
                expected = yaml.dump(
                    _arm_yaml_dict(arm, seed),
                    default_flow_style=False, sort_keys=False)
                assert path.read_text(encoding="utf-8") == expected, (
                    f"{path} is stale; re-run without --check"
                )
        diff = _build_resolution_diff()
        assert DIFF_PATH.exists(), f"missing {DIFF_PATH}"
        on_disk = json.loads(DIFF_PATH.read_text(encoding="utf-8"))
        assert on_disk == diff, f"{DIFF_PATH} is stale; re-run without --check"
        print("phase2 configs + resolution diff are current")
        return 0

    written = _write_configs()
    diff = _build_resolution_diff()
    DIFF_PATH.write_text(json.dumps(diff, indent=2) + "\n", encoding="utf-8")
    written.append(DIFF_PATH)
    for path in written:
        print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
