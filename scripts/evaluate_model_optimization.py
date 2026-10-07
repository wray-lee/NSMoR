"""Canonical evaluation runner for NSMoR model optimization protocol.

Validates immutable dataset, split, checkpoint, and target hashes BEFORE loader
or model execution. Enforces protocol constraints from
``docs/model-optimization-protocol-20261004.md``:
  * Strict immutable pin validation.
  * Rigorous matched-experiment checkpoint configuration matching.
  * Complete fail-closed candidate run/assessor receipt evidence validation.
  * Shared cache binding contract (role, checkpoint, dataset, prior, targets,
    ordered composite IDs, prefix IDs, trial lengths, lineage).
  * Isolated output paths with early fail-closed overwrite refusal.
  * Canonical restricted loaders and model loading with flexible 3/4 batch items.
  * Flat NPZ prediction caching with allow_pickle=False and bound metadata.
  * Individual candidate evaluation and complete 78-slot family assembly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from numbers import Real
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION

from nsmor.analysis.model_comparison import (
    CANDIDATE_IDS,
    COMPARATOR_IDS,
    DECLARED_FAMILY_SIZE,
    declared_family_slots,
)
from nsmor.analysis.optimization_evaluation import (
    assemble_declared_family,
    compute_aligned_grid_metrics,
    hash_target_sequences,
    load_prediction_cache,
    save_prediction_cache,
    validate_target_binding,
)
from nsmor.analysis.prediction_units import load_model_from_checkpoint
from nsmor.config_parser import ExperimentConfig
from nsmor.pipeline.conditions import resolve_anchor_crop
from nsmor.pipeline.nested_prior import load_artifact_bytes
if "scripts.train" in sys.modules:
    build_dataloaders = sys.modules["scripts.train"].build_dataloaders
else:
    from train import build_dataloaders

# Immutable protocol pins (docs/model-optimization-protocol-20261004.md)
EXPECTED_DATASET_SHA256 = (
    "b1bd5578025fb5eaa6f4eb3e355c097b6576d44dbedaa001027f1de5447e06b2"
)
EXPECTED_NESTED_PRIOR_SHA256 = (
    "f25856304a5857695945bf5083ca54745f47fe5dffe915111f6ba28d524ddf94"
)
EXPECTED_FULL_TARGET_SHA256 = (
    "038200fa85055deaab17c63295d0d838bd64d713461620e0a1728dcc6d8ad391"
)
EXPECTED_ELIG_TARGET_SHA256 = (
    "d6bb1cb1b91089da9808ac018e9f0ecbe67d8996bb20d5f02c3cdea8ffe45ab4"
)
EXPECTED_VAL_TRIALS = 432
EXPECTED_FULL_FRAMES = 1036748
EXPECTED_ELIG_FRAMES = 1036316

# Verified baseline checkpoint hash and output dir
BASELINE_CHECKPOINT_SHA256 = (
    "709ca1f1a121c2b0904d14ca056ad335dac14f5e178e9edb3f22b541ba18ff1b"
)
BASELINE_OUTPUT_DIR = ".scratch/model-opt-baseline-300-20261003-rerun"

# Reference baseline controls matching full_config_dump.json saved_config_full
BASELINE_SAVED_CONFIG: Dict[str, Dict[str, Any]] = {
    "model": {
        "sensory_dim": 4,
        "mcmc_dim": 4,
        "hidden_dim": 64,
        "num_gru_layers": 1,
        "dropout": 0.1,
        "dt_ms": 4.0,
        "lif_alpha": 0.9587,
        "lif_threshold": 0.5,
        "lif_beta": 2.0,
        "lif_abs_refract_ms": 0.0,
        "lif_rel_refract_ms": 20.0,
        "lif_tau_syn": 5.0,
        "lif_v_rest": 0.0,
        "lif_v_reset": None,
        "lif_tau_w": 100.0,
        "lif_b_adapt": 0.5,
        "lif_tau_fac": 0.0,
        "lif_tau_rec": 0.0,
        "lif_U_stp_init": 0.5,
        "lif_lateral_inhibition": 0.1,
        "lif_inhib_tau_ms": 50.0,
        "lif_dendritic_tau": 0.0,
        "gru_neuromod_gain": 0.0,
        "sensory_noise_std": 0.01,
        "lif_tbptt_steps": 32,
    },
    "training": {
        "learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "num_epochs": 300,
        "batch_size": 128,
        "grad_clip_norm": 1.0,
        "log_interval": 10,
        "checkpoint_interval": 10,
        "early_stopping_patience": 20,
        "random_seed": 42,
        "max_seq_len": 2400,
        "lr_warmup_epochs": 0,
        "normalize_targets": False,
        "target_clip_cm_s": 0,
        "num_workers": -1,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "escape_band_cm_s": 10.0,
    },
    "loss": {
        "reduction": "mean",
        "target_rate": 0.05,
        "lambda_reg": 0.2,
        "lambda_energy": 0.001,
        "lambda_sparse": 0.005,
        "lambda_jerk": 0.005,
        "lambda_routing_aux": 0.0,
        "routing_aux_margin": 0.024,
        "jerk_threshold": 0.1,
        "warmup_epochs": 20,
    },
    "data": {
        "train_kinematics": [
            "data/session_0/kinematics.csv",
            "data/session_1/kinematics.csv",
        ],
        "train_events": [
            "data/session_0/events.csv",
            "data/session_1/events.csv",
        ],
        "val_kinematics": ["data/val/kinematics.csv"],
        "val_events": ["data/val/events.csv"],
        "test_kinematics": [],
        "test_events": [],
    },
    "finetune": {
        "freeze_modules": [],
        "unfreeze_after_epoch": 10,
    },
    "checkpoint": {
        "output_dir": ".scratch/model-opt-baseline-300-20261003-rerun",
        "resume_from": None,
    },
    "cluster_gating": {
        "n_clusters": 4,
        "n_clusters_range": [2, 3, 4, 5],
        "random_state": 42,
        "use_umap": True,
        "fingerprint_dim": 16,
        "entropy_bins": 20,
        "interp_length": 200,
    },
}

BASELINE_MODEL_CONTROLS = BASELINE_SAVED_CONFIG["model"]
BASELINE_TRAINING_CONTROLS = BASELINE_SAVED_CONFIG["training"]
BASELINE_LOSS_CONTROLS = BASELINE_SAVED_CONFIG["loss"]


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of file bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _match_control_leaf(sec: str, key: str, actual_v: Any, expected_v: Any) -> None:
    """Validate a single configuration leaf value against reference control."""
    if expected_v is None:
        if actual_v is not None:
            raise ValueError(f"{sec}.{key} mismatch: expected None, got {actual_v}")
    elif isinstance(expected_v, bool):
        if type(actual_v) is not bool or actual_v != expected_v:
            raise ValueError(
                f"{sec}.{key} mismatch: expected bool {expected_v}, got {actual_v}"
            )
    elif isinstance(expected_v, float):
        if type(actual_v) is bool or not isinstance(actual_v, Real):
            raise ValueError(
                f"{sec}.{key} must be float, got {type(actual_v).__name__}"
            )
        if not math.isclose(float(actual_v), expected_v, rel_tol=1e-7, abs_tol=0.0):
            raise ValueError(
                f"{sec}.{key} mismatch: expected {expected_v}, got {actual_v}"
            )
    elif isinstance(expected_v, int):
        if type(actual_v) is bool or not isinstance(actual_v, (int, np.integer)):
            raise ValueError(
                f"{sec}.{key} must be int, got {type(actual_v).__name__}"
            )
        if int(actual_v) != expected_v:
            raise ValueError(
                f"{sec}.{key} mismatch: expected {expected_v}, got {actual_v}"
            )
    elif isinstance(expected_v, list):
        if actual_v != expected_v:
            raise ValueError(
                f"{sec}.{key} mismatch: expected {expected_v}, got {actual_v}"
            )
    else:
        if actual_v != expected_v:
            raise ValueError(
                f"{sec}.{key} mismatch: expected {expected_v}, got {actual_v}"
            )


def validate_checkpoint_dict_config(
    ckpt: Dict[str, Any],
    expected_candidate_id: str,
    label: str = "Checkpoint",
) -> None:
    """Validate a loaded checkpoint dict against the authorized matched experiment contract."""
    if not isinstance(ckpt, dict) or not ckpt:
        raise ValueError(f"{label} failed to deserialize to dict")

    cfg_dict = ckpt.get("config")
    if not isinstance(cfg_dict, dict) or not cfg_dict:
        raise ValueError(f"{label} config must be a non-empty dictionary")
    model_cfg = cfg_dict.get("model")
    train_cfg = cfg_dict.get("training")
    if not isinstance(model_cfg, dict) or not isinstance(train_cfg, dict):
        raise ValueError(f"{label} model and training configs must be dicts")

    # 1. Match all configuration sections against saved baseline reference
    for sec in cfg_dict:
        if sec not in BASELINE_SAVED_CONFIG:
            raise ValueError(f"{label} config contains unexpected section: '{sec}'")

    for sec in (
        "model",
        "training",
        "loss",
        "data",
        "finetune",
        "checkpoint",
        "cluster_gating",
    ):
        if sec not in cfg_dict:
            raise ValueError(f"{label} config missing required section: '{sec}'")
        actual_sec = cfg_dict[sec]
        if not isinstance(actual_sec, dict):
            raise ValueError(f"{label} config section '{sec}' must be a dict")
        expected_sec = BASELINE_SAVED_CONFIG[sec]
        for k, expected_v in expected_sec.items():
            if k not in actual_sec:
                raise ValueError(
                    f"{label} {sec} config missing required key: {k}"
                )
            actual_v = actual_sec[k]
            if sec == "checkpoint":
                if k == "resume_from":
                    if actual_v is not None:
                        raise ValueError(
                            f"{label} config resume_from must be null "
                            f"(fresh training required), got {actual_v}"
                        )
                elif k == "output_dir":
                    if not isinstance(actual_v, str) or not actual_v.strip():
                        raise ValueError(
                            f"{label} config output_dir must be non-empty string, got {actual_v!r}"
                        )
                    if (
                        expected_candidate_id in ("k_0.5", "k_1.0")
                        and actual_v == BASELINE_OUTPUT_DIR
                    ):
                        raise ValueError(
                            f"Candidate {expected_candidate_id} must use an isolated output_dir, "
                            f"cannot reuse baseline output_dir '{BASELINE_OUTPUT_DIR}'"
                        )
                else:
                    _match_control_leaf(sec, k, actual_v, expected_v)
            else:
                _match_control_leaf(sec, k, actual_v, expected_v)
        for k in actual_sec:
            if k not in expected_sec:
                if sec == "model" and k == "persistence_skip":
                    continue
                raise ValueError(
                    f"{label} {sec} config contains unexpected key: {k}"
                )

    # 2. Check persistence_skip field (the only model difference permitted)
    raw_k = model_cfg.get("persistence_skip")
    if expected_candidate_id == "head_only_baseline":
        if raw_k is None:
            k_val = 0.0
        else:
            if type(raw_k) is bool or not isinstance(raw_k, Real):
                raise ValueError(
                    "persistence_skip must be a genuine numeric scalar"
                )
            k_val = float(raw_k)
            if not math.isfinite(k_val) or k_val != 0.0:
                raise ValueError(f"Baseline must have k=0.0, got {k_val}")
    elif expected_candidate_id == "k_0.5":
        if raw_k is None:
            raise ValueError("Candidate k_0.5 checkpoint missing persistence_skip")
        if type(raw_k) is bool or not isinstance(raw_k, Real):
            raise ValueError("persistence_skip must be a genuine numeric scalar")
        k_val = float(raw_k)
        if not math.isfinite(k_val) or k_val != 0.5:
            raise ValueError(f"Candidate k_0.5 must have k=0.5, got {k_val}")
    elif expected_candidate_id == "k_1.0":
        if raw_k is None:
            raise ValueError("Candidate k_1.0 checkpoint missing persistence_skip")
        if type(raw_k) is bool or not isinstance(raw_k, Real):
            raise ValueError("persistence_skip must be a genuine numeric scalar")
        k_val = float(raw_k)
        if not math.isfinite(k_val) or k_val != 1.0:
            raise ValueError(f"Candidate k_1.0 must have k=1.0, got {k_val}")
    else:
        raise ValueError(f"Unhandled candidate ID: {expected_candidate_id}")

    # 3. Checkpoint lineage checks & immutable pins
    psv = ckpt.get("pipeline_semantics_version")
    if psv is None:
        raise ValueError(f"{label} missing required 'pipeline_semantics_version'")
    if str(psv) != PIPELINE_SEMANTICS_VERSION:
        raise ValueError(
            f"{label} pipeline_semantics_version mismatch: "
            f"expected '{PIPELINE_SEMANTICS_VERSION}', got '{psv}'"
        )

    np_fp = ckpt.get("nested_prior_fingerprint")
    if np_fp is None:
        raise ValueError(f"{label} missing required 'nested_prior_fingerprint'")
    if str(np_fp) != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"{label} nested_prior_fingerprint mismatch: "
            f"expected {EXPECTED_DATASET_SHA256}, got '{np_fp}'"
        )

    ds_src_sha = ckpt.get("dataset_source_sha256")
    if ds_src_sha is None:
        raise ValueError(f"{label} missing required 'dataset_source_sha256'")
    if str(ds_src_sha) != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"{label} dataset_source_sha256 mismatch: "
            f"expected {EXPECTED_DATASET_SHA256}, got {ds_src_sha}"
        )

    nest_src_sha = ckpt.get("nested_prior_artifact_sha256")
    if nest_src_sha is None:
        raise ValueError(f"{label} missing required 'nested_prior_artifact_sha256'")
    if str(nest_src_sha) != EXPECTED_NESTED_PRIOR_SHA256:
        raise ValueError(
            f"{label} nested_prior_artifact_sha256 mismatch: "
            f"expected {EXPECTED_NESTED_PRIOR_SHA256}, got {nest_src_sha}"
        )

    if ckpt.get("is_nested_cv") is not True:
        raise ValueError(
            f"{label} is_nested_cv must be True, got {ckpt.get('is_nested_cv')}"
        )

    val_scope = ckpt.get("validation_scope")
    if val_scope != "nested_outer_validation":
        raise ValueError(
            f"{label} validation_scope must be 'nested_outer_validation', "
            f"got '{val_scope}'"
        )

    split_seed = ckpt.get("nested_split_seed")
    if (
        type(split_seed) is bool
        or not isinstance(split_seed, (int, np.integer))
        or int(split_seed) != 42
    ):
        raise ValueError(
            f"{label} nested_split_seed must be integer 42, "
            f"got {split_seed}"
        )

    val_split = ckpt.get("nested_val_split")
    if (
        type(val_split) is bool
        or not isinstance(val_split, Real)
        or not math.isclose(float(val_split), 0.2, rel_tol=1e-5, abs_tol=1e-5)
    ):
        raise ValueError(
            f"{label} nested_val_split must be 0.2, "
            f"got {val_split}"
        )

    for unit_key, expected_unit in (
        ("target_mean", 0.0),
        ("target_std", 1.0),
        ("target_clip_cm_s", 0.0),
    ):
        if unit_key in ckpt:
            u_val = ckpt[unit_key]
            if type(u_val) is bool or not isinstance(u_val, Real):
                raise ValueError(
                    f"{label} {unit_key} must be numeric scalar"
                )
            if float(u_val) != expected_unit:
                raise ValueError(
                    f"{label} {unit_key} mismatch: expected {expected_unit}, "
                    f"got {u_val}"
                )

    # 4. State dict finiteness and substance
    sd = ckpt.get("model_state_dict")
    if not isinstance(sd, dict) or len(sd) == 0:
        raise ValueError(f"{label} missing mandatory non-empty 'model_state_dict'")
    for t_name, t_val in sd.items():
        if not isinstance(t_val, torch.Tensor):
            raise ValueError(
                f"{label} model_state_dict['{t_name}'] must be a torch.Tensor, got {type(t_val).__name__}"
            )
        if not torch.isfinite(t_val).all():
            raise ValueError(
                f"{label} model tensor '{t_name}' contains non-finite values"
            )


def validate_immutable_pins(
    dataset_path: Path,
    nested_prior_path: Path,
) -> None:
    """Verify that dataset and split match protocol hashes before execution."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not nested_prior_path.exists():
        raise FileNotFoundError(f"Nested prior not found: {nested_prior_path}")

    ds_sha = _compute_sha256(dataset_path)
    if ds_sha != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"Dataset SHA-256 mismatch: expected {EXPECTED_DATASET_SHA256}, "
            f"got {ds_sha}"
        )

    nest_sha = _compute_sha256(nested_prior_path)
    if nest_sha != EXPECTED_NESTED_PRIOR_SHA256:
        raise ValueError(
            f"Nested prior SHA-256 mismatch: expected {EXPECTED_NESTED_PRIOR_SHA256}, "
            f"got {nest_sha}"
        )


def validate_run_receipt(
    receipt_path: Path,
    expected_candidate_id: str,
    expected_checkpoint_sha: str,
    ckpt: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate sealed run receipt or assessor report for candidate completion.

    Fails closed if any required field is missing, null, wrong type, or mismatched.
    """
    if not receipt_path.exists():
        raise FileNotFoundError(f"Run receipt not found: {receipt_path}")
    raw_data = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict) or not raw_data:
        raise ValueError("Run receipt must be a non-empty JSON object")

    if "status" in raw_data and raw_data["status"] != "PASS":
        raise ValueError(
            f"Run receipt top-level status is '{raw_data['status']}', not 'PASS'"
        )
    if "exit_code" in raw_data:
        exit_code = raw_data["exit_code"]
        if (
            type(exit_code) is bool
            or not isinstance(exit_code, (int, np.integer))
            or int(exit_code) != 0
        ):
            raise ValueError(f"Run receipt top-level exit_code is {exit_code}, not 0")
    if raw_data.get("failures"):
        raise ValueError(f"Run receipt contains failures: {raw_data['failures']}")
    if raw_data.get("evaluator_binding_problems"):
        raise ValueError(
            f"Run receipt contains evaluator_binding_problems: {raw_data['evaluator_binding_problems']}"
        )

    eb = raw_data.get("evaluator_binding")
    if eb is None:
        eb = raw_data
    if not isinstance(eb, dict) or not eb:
        raise ValueError("Run receipt missing non-empty evaluator_binding block")

    # 1. Role matching (strict, fail closed on missing/unknown role)
    actual_role = eb.get("role")
    if actual_role is None:
        actual_role = eb.get("candidate_id")
    if actual_role is None:
        raise ValueError("Run receipt missing required 'role' field")
    expected_roles = {
        "k_0.5": ("candidate_k0.5", "candidate_k_0.5", "k_0.5"),
        "k_1.0": ("candidate_k1.0", "candidate_k_1.0", "k_1.0"),
        "head_only_baseline": ("head_only_baseline", "baseline"),
    }
    valid_roles = expected_roles.get(
        expected_candidate_id, (expected_candidate_id,)
    )
    if actual_role not in valid_roles:
        raise ValueError(
            f"Run receipt role mismatch: expected one of {valid_roles}, "
            f"got '{actual_role}'"
        )

    # 2. Budget completion and natural exit 0 (must be exact integer 300, not bool)
    declared = eb.get("declared_epochs")
    if (
        type(declared) is bool
        or not isinstance(declared, (int, np.integer))
        or int(declared) != 300
    ):
        raise ValueError(
            f"Run receipt declared_epochs must be integer 300, got {declared}"
        )

    actual_epochs = eb.get("actual_completed_epochs")
    if actual_epochs is None:
        actual_epochs = eb.get("completed_epochs")
    if (
        type(actual_epochs) is bool
        or not isinstance(actual_epochs, (int, np.integer))
        or int(actual_epochs) != 300
    ):
        raise ValueError(
            f"Run receipt actual_completed_epochs must be integer 300, got {actual_epochs}"
        )

    completion = eb.get("completion")
    if completion != "natural_completion":
        raise ValueError(
            f"Run receipt completion must be 'natural_completion', got '{completion}'"
        )

    natural_exit_zero = eb.get("natural_exit_zero")
    if type(natural_exit_zero) is not bool or natural_exit_zero is not True:
        raise ValueError(
            f"Run receipt natural_exit_zero must be True, got {natural_exit_zero}"
        )

    sealed_status = eb.get("sealed_status")
    if sealed_status is None:
        sealed_status = raw_data.get("status")
    if sealed_status != "PASS":
        raise ValueError(
            f"Run receipt sealed_status must be 'PASS', got '{sealed_status}'"
        )

    # 3. Fresh unresumed training (fail closed on missing/resumed/contradictory)
    if raw_data.get("resumed") is True or eb.get("resumed") is True:
        raise ValueError("Run receipt records resumed=True; fresh training required")
    if raw_data.get("fresh_training") is False or eb.get("fresh_training") is False:
        raise ValueError("Run receipt records fresh_training=False; fresh training required")

    not_resumed = eb.get("not_resumed")
    if not_resumed is None:
        if eb.get("fresh_training") is True and eb.get("resumed") is False:
            not_resumed = True
    if type(not_resumed) is not bool or not_resumed is not True:
        raise ValueError(
            f"Run receipt not_resumed must be True, got {not_resumed}"
        )

    resumed_from = eb.get("resumed_from")
    if resumed_from is not None:
        raise ValueError(
            f"Run receipt resumed_from must be null, got '{resumed_from}'"
        )

    early_stopped = eb.get("early_stopped")
    if type(early_stopped) is not bool or early_stopped is not False:
        raise ValueError(
            f"Run receipt early_stopped must be False, got {early_stopped}"
        )

    # 4. Protocol contract keys (fail closed on missing protocol block)
    proto = eb.get("protocol")
    if not isinstance(proto, dict) or not proto:
        raise ValueError("Run receipt missing mandatory 'protocol' dictionary")

    expected_k = (
        0.5
        if expected_candidate_id == "k_0.5"
        else (1.0 if expected_candidate_id == "k_1.0" else 0.0)
    )
    k_val = proto.get("persistence_skip_k")
    if type(k_val) is bool or not isinstance(k_val, Real):
        raise ValueError("protocol.persistence_skip_k must be numeric scalar")
    if not math.isclose(float(k_val), expected_k, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(
            f"protocol.persistence_skip_k mismatch: expected {expected_k}, got {k_val}"
        )

    seed_val = proto.get("random_seed")
    if (
        type(seed_val) is bool
        or not isinstance(seed_val, (int, np.integer))
        or int(seed_val) != 42
    ):
        raise ValueError(
            f"protocol.random_seed mismatch: expected integer 42, got {seed_val}"
        )

    proto_epochs = proto.get("num_epochs")
    if (
        type(proto_epochs) is bool
        or not isinstance(proto_epochs, (int, np.integer))
        or int(proto_epochs) != 300
    ):
        raise ValueError(
            f"protocol.num_epochs mismatch: expected integer 300, got {proto_epochs}"
        )

    proto_norm = proto.get("normalize_targets")
    if type(proto_norm) is not bool or proto_norm is not False:
        raise ValueError(
            f"protocol.normalize_targets mismatch: expected False, got {proto_norm}"
        )

    proto_clip = proto.get("target_clip_cm_s")
    if (
        type(proto_clip) is bool
        or not isinstance(proto_clip, Real)
        or float(proto_clip) != 0.0
    ):
        raise ValueError(
            f"protocol.target_clip_cm_s mismatch: expected 0.0, got {proto_clip}"
        )

    proto_mean = proto.get("target_mean")
    if (
        type(proto_mean) is bool
        or not isinstance(proto_mean, Real)
        or float(proto_mean) != 0.0
    ):
        raise ValueError(
            f"protocol.target_mean mismatch: expected 0.0, got {proto_mean}"
        )

    proto_std = proto.get("target_std")
    if (
        type(proto_std) is bool
        or not isinstance(proto_std, Real)
        or float(proto_std) != 1.0
    ):
        raise ValueError(
            f"protocol.target_std mismatch: expected 1.0, got {proto_std}"
        )

    # 5. Pins verification (fail closed on missing pins block)
    pins = eb.get("pins")
    if not isinstance(pins, dict) or not pins:
        raise ValueError("Run receipt missing mandatory 'pins' dictionary")

    ds_sha = pins.get("dataset_source_sha256")
    if ds_sha != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"pins.dataset_source_sha256 mismatch: expected {EXPECTED_DATASET_SHA256}, "
            f"got {ds_sha}"
        )

    nest_sha = pins.get("nested_prior_artifact_sha256")
    if nest_sha != EXPECTED_NESTED_PRIOR_SHA256:
        raise ValueError(
            f"pins.nested_prior_artifact_sha256 mismatch: expected {EXPECTED_NESTED_PRIOR_SHA256}, "
            f"got {nest_sha}"
        )

    val_scope = pins.get("validation_scope")
    if val_scope != "nested_outer_validation":
        raise ValueError(
            f"pins.validation_scope mismatch: expected 'nested_outer_validation', "
            f"got '{val_scope}'"
        )

    split_seed = pins.get("nested_split_seed")
    if (
        type(split_seed) is bool
        or not isinstance(split_seed, (int, np.integer))
        or int(split_seed) != 42
    ):
        raise ValueError(
            f"pins.nested_split_seed mismatch: expected integer 42, got {split_seed}"
        )

    # 6. Best checkpoint selection (fail closed on missing best_checkpoint block)
    best_info = eb.get("best_checkpoint")
    if not isinstance(best_info, dict) or not best_info:
        raise ValueError("Run receipt missing mandatory 'best_checkpoint' dictionary")

    b_sha = best_info.get("sha256")
    if b_sha != expected_checkpoint_sha:
        raise ValueError(
            f"best_checkpoint.sha256 mismatch: expected {expected_checkpoint_sha}, "
            f"got {b_sha}"
        )

    b_epoch0 = best_info.get("epoch_zero_based")
    if (
        type(b_epoch0) is bool
        or not isinstance(b_epoch0, (int, np.integer))
        or int(b_epoch0) != int(ckpt["epoch"])
    ):
        raise ValueError(
            f"best_checkpoint.epoch_zero_based ({b_epoch0}) does not match "
            f"checkpoint epoch ({ckpt['epoch']})"
        )

    b_epoch1 = best_info.get("epoch_one_based")
    if (
        type(b_epoch1) is bool
        or not isinstance(b_epoch1, (int, np.integer))
        or int(b_epoch1) != int(b_epoch0) + 1
    ):
        raise ValueError(
            f"best_checkpoint.epoch_one_based ({b_epoch1}) does not match "
            f"epoch_zero_based + 1 ({int(b_epoch0) + 1})"
        )

    b_loss = best_info.get("val_loss")
    if (
        type(b_loss) is bool
        or not isinstance(b_loss, Real)
        or not math.isfinite(float(b_loss))
        or float(b_loss) < 0.0
    ):
        raise ValueError(
            f"best_checkpoint.val_loss must be finite non-negative float, got {b_loss}"
        )
    if not math.isclose(
        float(b_loss), float(ckpt["val_loss"]), rel_tol=1e-5, abs_tol=1e-5
    ):
        raise ValueError(
            f"best_checkpoint.val_loss ({b_loss}) does not match "
            f"checkpoint val_loss ({ckpt['val_loss']})"
        )

    b_best_loss = best_info.get("best_val_loss")
    if b_best_loss is not None:
        if (
            type(b_best_loss) is bool
            or not isinstance(b_best_loss, Real)
            or not math.isclose(
                float(b_best_loss), float(b_loss), rel_tol=1e-5, abs_tol=1e-5
            )
        ):
            raise ValueError(
                f"best_checkpoint.best_val_loss ({b_best_loss}) does not match "
                f"val_loss ({b_loss})"
            )

    # 7. Final checkpoint verification (fail closed on missing final_checkpoint block)
    final_info = eb.get("final_checkpoint")
    if not isinstance(final_info, dict) or not final_info:
        raise ValueError("Run receipt missing mandatory 'final_checkpoint' dictionary")

    f_sha = final_info.get("sha256")
    if (
        not isinstance(f_sha, str)
        or len(f_sha) != 64
        or not all(c in "0123456789abcdefABCDEF" for c in f_sha)
    ):
        raise ValueError(
            f"final_checkpoint.sha256 must be a 64-char hexadecimal digest, got '{f_sha}'"
        )

    f_epoch0 = final_info.get("epoch_zero_based")
    if (
        type(f_epoch0) is bool
        or not isinstance(f_epoch0, (int, np.integer))
        or int(f_epoch0) != 299
    ):
        raise ValueError(
            f"final_checkpoint.epoch_zero_based must be 299 (last epoch of 300), "
            f"got {f_epoch0}"
        )

    # Selection consistency: best epoch cannot exceed final epoch
    if int(b_epoch0) > int(f_epoch0):
        raise ValueError(
            f"best_checkpoint epoch_zero_based ({b_epoch0}) cannot be greater than "
            f"final_checkpoint epoch_zero_based ({f_epoch0})"
        )

    f_loss = final_info.get("val_loss")
    if (
        type(f_loss) is bool
        or not isinstance(f_loss, Real)
        or not math.isfinite(float(f_loss))
        or float(f_loss) < 0.0
    ):
        raise ValueError(
            f"final_checkpoint.val_loss must be finite non-negative float, got {f_loss}"
        )

    # Locate and verify actual backing final checkpoint file
    f_path_str = final_info.get("path")
    if f_path_str:
        p_final = Path(f_path_str)
        if not p_final.is_absolute():
            p_final = (receipt_path.parent / p_final).resolve()
    else:
        cands = [
            receipt_path.parent / "final_model.pth",
            receipt_path.parent / "checkpoint_last.pth",
        ]
        found_cands = [c for c in cands if c.exists()]
        if found_cands:
            p_final = found_cands[0]
        else:
            raise ValueError(
                "Run receipt final_checkpoint missing 'path' and no final "
                f"checkpoint file found in {receipt_path.parent}"
            )

    if not p_final.exists():
        raise FileNotFoundError(f"final_checkpoint file not found: {p_final}")

    actual_f_sha = _compute_sha256(p_final)
    if actual_f_sha != f_sha:
        raise ValueError(
            f"final_checkpoint SHA-256 mismatch: expected {f_sha}, got {actual_f_sha}"
        )

    # Restricted-decode final checkpoint bytes and verify facts
    f_ckpt_bytes = p_final.read_bytes()
    f_ckpt = load_artifact_bytes(f_ckpt_bytes, map_location="cpu")
    validate_checkpoint_dict_config(
        f_ckpt, expected_candidate_id, label="final_checkpoint"
    )

    # Embedded epoch must be 299
    f_ep = f_ckpt.get("epoch")
    if (
        type(f_ep) is bool
        or not isinstance(f_ep, (int, np.integer))
        or int(f_ep) != 299
    ):
        raise ValueError(
            f"final_checkpoint embedded epoch must be 299, got {f_ep}"
        )

    # Embedded val_loss must be finite non-negative and match receipt
    f_ckpt_loss = f_ckpt.get("val_loss")
    if (
        type(f_ckpt_loss) is bool
        or not isinstance(f_ckpt_loss, Real)
        or not math.isfinite(float(f_ckpt_loss))
        or float(f_ckpt_loss) < 0.0
    ):
        raise ValueError(
            f"final_checkpoint embedded val_loss must be finite non-negative float, got {f_ckpt_loss}"
        )
    if not math.isclose(
        float(f_ckpt_loss), float(f_loss), rel_tol=1e-5, abs_tol=1e-5
    ):
        raise ValueError(
            f"final_checkpoint embedded val_loss ({f_ckpt_loss}) does not match "
            f"receipt val_loss ({f_loss})"
        )

    # Contradictory best/final loss ordering: best val loss cannot be greater than final val loss
    if float(b_loss) > float(f_loss) + 1e-5:
        raise ValueError(
            f"Contradictory loss ordering: best_checkpoint val_loss ({b_loss}) "
            f"is greater than final_checkpoint val_loss ({f_loss})"
        )

    # Contradictory checkpoint hashes: best and final cannot have identical SHA if epochs differ
    if int(b_epoch0) != int(f_epoch0) and b_sha == f_sha:
        raise ValueError(
            f"Contradictory checkpoint hashes: best_checkpoint epoch {b_epoch0} and "
            f"final_checkpoint epoch {f_epoch0} have identical SHA-256 {b_sha}"
        )

    # 8. Source seal verification (fail closed on missing source_seal block, drift, or contradictory facts)
    source_seal = eb.get("source_seal")
    if not isinstance(source_seal, dict) or not source_seal:
        raise ValueError("Run receipt missing mandatory 'source_seal' dictionary")

    s_rec = source_seal.get("sealed_receipt")
    s_rec_sha = source_seal.get("sealed_receipt_sha256")
    if not s_rec or not s_rec_sha:
        raise ValueError("source_seal missing sealed_receipt path or sha256")
    p_rec = Path(s_rec)
    if not p_rec.is_absolute():
        p_rec = (receipt_path.parent / p_rec).resolve()
    if not p_rec.exists():
        raise FileNotFoundError(f"source_seal sealed_receipt path not found: {p_rec}")
    actual_rec_sha = _compute_sha256(p_rec)
    if actual_rec_sha != s_rec_sha:
        raise ValueError(
            f"source_seal sealed_receipt SHA mismatch: expected {s_rec_sha}, "
            f"got {actual_rec_sha}"
        )
    # Parse sealed receipt and reconcile factual completion
    rec_obj = json.loads(p_rec.read_text(encoding="utf-8"))
    if not isinstance(rec_obj, dict):
        raise ValueError("source_seal sealed_receipt must be a JSON object")
    if rec_obj.get("status") != "PASS":
        raise ValueError(
            f"source_seal sealed_receipt records status '{rec_obj.get('status')}', not 'PASS'"
        )
    if "exit_code" in rec_obj:
        rec_exit = rec_obj["exit_code"]
        if (
            type(rec_exit) is bool
            or not isinstance(rec_exit, (int, np.integer))
            or int(rec_exit) != 0
        ):
            raise ValueError(
                f"source_seal sealed_receipt records exit_code {rec_exit}, not 0"
            )
    if rec_obj.get("resumed") is True:
        raise ValueError("source_seal sealed_receipt records resumed=True")
    if "actual_completed_epochs" in rec_obj:
        rec_ace = rec_obj["actual_completed_epochs"]
        if (
            type(rec_ace) is bool
            or not isinstance(rec_ace, (int, np.integer))
            or int(rec_ace) != 300
        ):
            raise ValueError(
                f"source_seal sealed_receipt actual_completed_epochs must be integer 300, got {rec_ace!r}"
            )
    if "completed_epochs" in rec_obj:
        rec_ce = rec_obj["completed_epochs"]
        if (
            type(rec_ce) is bool
            or not isinstance(rec_ce, (int, np.integer))
            or int(rec_ce) != 300
        ):
            raise ValueError(
                f"source_seal sealed_receipt completed_epochs must be integer 300, got {rec_ce!r}"
            )

    s_log = source_seal.get("log")
    s_log_sha = source_seal.get("log_sha256")
    if not s_log or not s_log_sha:
        raise ValueError("source_seal missing log path or log_sha256")
    p_log = Path(s_log)
    if not p_log.is_absolute():
        p_log = (receipt_path.parent / p_log).resolve()
    if not p_log.exists():
        raise FileNotFoundError(f"source_seal log path not found: {p_log}")
    actual_log_sha = _compute_sha256(p_log)
    if actual_log_sha != s_log_sha:
        raise ValueError(
            f"source_seal log SHA mismatch: expected {s_log_sha}, got {actual_log_sha}"
        )

    # Cross-check sealed_receipt embedded log vs source_seal log
    if "log" in rec_obj:
        if Path(rec_obj["log"]).resolve() != p_log.resolve():
            raise ValueError(
                f"source_seal log path ({p_log}) contradicts sealed_receipt embedded log path ({rec_obj['log']})"
            )
    if "log_sha256" in rec_obj:
        if rec_obj["log_sha256"] != s_log_sha:
            raise ValueError(
                f"source_seal log SHA ({s_log_sha}) contradicts sealed_receipt embedded log_sha256 ({rec_obj['log_sha256']})"
            )

    # Parse log content and reconcile facts
    log_content = p_log.read_text(encoding="utf-8", errors="replace")
    if not log_content.strip():
        raise ValueError("source_seal log is empty")
    for bad_token in (
        "Early stopping",
        "Early stopped",
        "early stopping",
        "Process exit code 1",
        "Process exit code 2",
        "Process exit code 17",
        "RuntimeError",
        "Traceback (most recent call last)",
    ):
        if bad_token in log_content:
            raise ValueError(
                f"source_seal log contains failure or early termination indicator: '{bad_token}'"
            )

    # Positively verify contiguous stdout telemetry and natural completion
    # 1. Strictly refuse zero-based epoch progression
    if "Epoch 0/300" in log_content or re.search(r"\bEpoch\s+0/300\b", log_content):
        raise ValueError(
            "source_seal log contains zero-based epoch telemetry ('Epoch 0/300'); expected 1-based progression Epoch 1/300..300/300"
        )

    # 2. Positively verify contiguous epoch progression tokens: require every epoch 1..300
    for ep_idx in range(1, 301):
        tok = f"Epoch {ep_idx}/300"
        if tok not in log_content:
            raise ValueError(
                f"source_seal log does not contain required contiguous epoch progression telemetry (missing epoch {ep_idx}: expected '{tok}')"
            )

    # 3. Parse per-epoch telemetry and strictly validate ordering, uniqueness, and finiteness
    # Pattern matches standard trainer output and test log variants:
    # 'Epoch 1/300  train_loss=8.000000  val_loss=7.500000  time=0.1s'
    # 'Epoch 1/300: train_loss 8.000, val_loss 7.500'
    epoch_telemetry_re = re.compile(
        r"Epoch\s+(\d+)/(\d+)(?::)?\s+train_loss[=:\s]+([^\s,]+)[,\s]+val_loss[=:\s]+([^\s,]+)"
    )
    parsed_stdout_telemetry: Dict[int, Dict[str, float]] = {}
    last_seen_epoch = 0
    telemetry_matches = list(epoch_telemetry_re.finditer(log_content))
    if len(telemetry_matches) != 300:
        raise ValueError(
            f"source_seal log must contain exactly 300 valid per-epoch telemetry entries ('Epoch i/300 train_loss=... val_loss=...'), found {len(telemetry_matches)}"
        )

    for match in telemetry_matches:
        ep_num = int(match.group(1))
        tot_num = int(match.group(2))
        if tot_num != 300:
            raise ValueError(
                f"source_seal log epoch telemetry total epochs must be 300, got {tot_num} at epoch {ep_num}"
            )
        if ep_num == 0:
            raise ValueError(
                "source_seal log contains zero-based epoch telemetry ('Epoch 0/300'); expected 1-based progression Epoch 1/300..300/300"
            )
        if ep_num in parsed_stdout_telemetry:
            raise ValueError(
                f"source_seal log contains duplicate epoch telemetry for epoch {ep_num}"
            )
        if ep_num != last_seen_epoch + 1:
            raise ValueError(
                f"source_seal log epoch telemetry out of order or discontiguous: epoch {ep_num} followed epoch {last_seen_epoch}"
            )
        tl_str = match.group(3)
        vl_str = match.group(4)
        try:
            tl = float(tl_str)
            vl = float(vl_str)
        except ValueError as exc:
            raise ValueError(
                f"source_seal log epoch {ep_num} contains unparseable loss value: train_loss={tl_str!r}, val_loss={vl_str!r}"
            ) from exc

        if not math.isfinite(tl) or not math.isfinite(vl):
            raise ValueError(
                f"source_seal log epoch {ep_num} contains non-finite loss: train_loss={tl_str}, val_loss={vl_str}"
            )
        parsed_stdout_telemetry[ep_num] = {
            "train_loss": tl,
            "val_loss": vl,
            "train_loss_str": tl_str.strip(),
            "val_loss_str": vl_str.strip(),
        }
        last_seen_epoch = ep_num

    lower_log = log_content.lower()
    if not any(
        tok in lower_log
        for tok in (
            "training complete",
            "training completed",
            "completed naturally",
            "natural_completion",
            "exit 0",
        )
    ):
        raise ValueError(
            "source_seal log does not positively verify natural training completion"
        )

    # Mandatory sealer manifest for candidates
    s_man = source_seal.get("manifest")
    s_man_sha = source_seal.get("manifest_sha256")
    if s_man is None or s_man_sha is None:
        raise ValueError(
            "source_seal missing mandatory manifest path or manifest_sha256"
        )
    p_man = Path(s_man)
    if not p_man.is_absolute():
        p_man = (receipt_path.parent / p_man).resolve()
    if not p_man.exists():
        raise FileNotFoundError(f"source_seal manifest path not found: {p_man}")
    actual_man_sha = _compute_sha256(p_man)
    if actual_man_sha != s_man_sha:
        raise ValueError(
            f"source_seal manifest SHA mismatch: expected {s_man_sha}, "
            f"got {actual_man_sha}"
        )

    # Cross-check sealed_receipt embedded manifest vs source_seal manifest
    if "manifest" in rec_obj:
        if Path(rec_obj["manifest"]).resolve() != p_man.resolve():
            raise ValueError(
                f"source_seal manifest path ({p_man}) contradicts sealed_receipt embedded manifest ({rec_obj['manifest']})"
            )
    if "manifest_sha256" in rec_obj:
        if rec_obj["manifest_sha256"] != s_man_sha:
            raise ValueError(
                f"source_seal manifest SHA ({s_man_sha}) contradicts sealed_receipt embedded manifest_sha256 ({rec_obj['manifest_sha256']})"
            )

    man_obj = json.loads(p_man.read_text(encoding="utf-8"))
    if not isinstance(man_obj, dict):
        raise ValueError("source_seal manifest must be a JSON object")

    # Enforce standard manifest metadata keys: label, kind, created_utc, cwd
    for m_field in ("label", "kind", "created_utc", "cwd"):
        if m_field not in man_obj or not man_obj[m_field]:
            raise ValueError(
                f"source_seal manifest missing mandatory metadata field: '{m_field}'"
            )
    if man_obj["kind"] not in ("source", "merged", "manifest"):
        raise ValueError(
            f"source_seal manifest 'kind' must be 'source', 'merged', or 'manifest', got '{man_obj['kind']}'"
        )

    # Positive schema verification: require non-empty paths mapping
    path_dict = None
    if "paths" in man_obj and isinstance(man_obj["paths"], dict):
        path_dict = man_obj["paths"]
    else:
        merged: Dict[str, Any] = {}
        for sec in ("files", "dependencies"):
            if sec in man_obj and isinstance(man_obj[sec], dict):
                merged.update(man_obj[sec])
        if merged:
            path_dict = merged

    if path_dict is None or len(path_dict) == 0:
        raise ValueError(
            "source_seal manifest missing mandatory non-empty 'paths' (or 'files') mapping"
        )

    for p_k, p_digest in path_dict.items():
        if not isinstance(p_k, str) or not p_k.strip():
            raise ValueError(f"source_seal manifest has invalid path key: {p_k!r}")
        if (
            not isinstance(p_digest, str)
            or len(p_digest) != 64
            or not all(c in "0123456789abcdef" for c in p_digest)
        ):
            raise ValueError(
                f"source_seal manifest path {p_k!r} has invalid sha256 (must be 64-char lowercase hex): {p_digest!r}"
            )

    if man_obj.get("status") not in (None, "PASS"):
        raise ValueError(
            f"source_seal manifest records status '{man_obj.get('status')}', not 'PASS'"
        )

    # 9. Source provenance block verification (mandatory for candidates)
    prov = eb.get("source_provenance")
    if not isinstance(prov, dict) or not prov:
        raise ValueError(
            "Run receipt missing mandatory non-empty 'source_provenance' block"
        )

    cmd = prov.get("command")
    if not cmd or (not isinstance(cmd, (list, str))):
        raise ValueError(
            "source_provenance missing mandatory non-empty 'command'"
        )

    prov_exit = prov.get("sealed_exit_code")
    if (
        type(prov_exit) is bool
        or not isinstance(prov_exit, (int, np.integer))
        or int(prov_exit) != 0
    ):
        raise ValueError(
            f"source_provenance sealed_exit_code is {prov_exit}, not 0"
        )

    if prov.get("sealed_status") != "PASS":
        raise ValueError(
            f"source_provenance sealed_status is '{prov.get('sealed_status')}', not 'PASS'"
        )

    probe = prov.get("runtime_probe")
    if not isinstance(probe, dict) or not probe:
        raise ValueError(
            "source_provenance missing mandatory non-empty 'runtime_probe'"
        )
    if probe.get("status") != "PASS":
        raise ValueError(
            f"source_provenance runtime_probe status is '{probe.get('status')}', not 'PASS'"
        )

    binding_block = probe.get("output_dir_binding")
    if not isinstance(binding_block, dict) or not binding_block:
        raise ValueError(
            "source_provenance runtime_probe missing mandatory non-empty 'output_dir_binding'"
        )
    if binding_block.get("checked") is not True:
        raise ValueError(
            "source_provenance runtime_probe output_dir_binding 'checked' must be True"
        )
    if binding_block.get("satisfied") is not True:
        raise ValueError(
            "source_provenance runtime_probe output_dir_binding 'satisfied' must be True"
        )

    calls = binding_block.get("train_call_resolved")
    if not isinstance(calls, list) or not calls:
        raise ValueError(
            "source_provenance runtime_probe missing mandatory non-empty 'train_call_resolved'"
        )
    first = calls[0]
    if not isinstance(first, dict):
        raise ValueError("train_call_resolved[0] must be a dict")
    if first.get("resume_from") is not None:
        raise ValueError(
            f"source_provenance runtime_probe train call was resumed: {first.get('resume_from')}"
        )
    call_epochs = first.get("num_epochs")
    if (
        type(call_epochs) is bool
        or not isinstance(call_epochs, (int, np.integer))
        or int(call_epochs) != 300
    ):
        raise ValueError(
            f"source_provenance runtime_probe train call num_epochs mismatch: expected integer 300, got {call_epochs}"
        )
    call_seed = first.get("random_seed")
    if (
        type(call_seed) is bool
        or not isinstance(call_seed, (int, np.integer))
        or int(call_seed) != 42
    ):
        raise ValueError(
            f"source_provenance runtime_probe train call random_seed mismatch: expected integer 42, got {call_seed}"
        )

    # Output dir binding reconciliation
    call_out_dir = first.get("output_dir")
    if not call_out_dir or not isinstance(call_out_dir, str):
        raise ValueError("train_call_resolved[0] missing valid 'output_dir'")
    if "output_dir" in probe:
        if Path(probe["output_dir"]).resolve() != Path(call_out_dir).resolve():
            raise ValueError(
                f"runtime_probe output_dir ({probe['output_dir']}) does not match "
                f"train_call_resolved output_dir ({call_out_dir})"
            )
    if "resolved_config" in prov and isinstance(prov["resolved_config"], dict):
        rc = prov["resolved_config"]
        if "output_dir" in rc and Path(rc["output_dir"]).resolve() != Path(call_out_dir).resolve():
            raise ValueError(
                f"resolved_config output_dir ({rc['output_dir']}) does not match "
                f"train_call_resolved output_dir ({call_out_dir})"
            )

    # Reconcile checkpoint embedded output_dir against runtime output_dir
    ckpt_out_dir = ckpt.get("config", {}).get("checkpoint", {}).get("output_dir")
    if ckpt_out_dir and isinstance(ckpt_out_dir, str):
        if Path(ckpt_out_dir).resolve() != Path(call_out_dir).resolve():
            raise ValueError(
                f"checkpoint embedded output_dir ({ckpt_out_dir}) does not match "
                f"runtime train_call_resolved output_dir ({call_out_dir})"
            )
    if f_ckpt is not None:
        f_ckpt_out_dir = f_ckpt.get("config", {}).get("checkpoint", {}).get("output_dir")
        if f_ckpt_out_dir and isinstance(f_ckpt_out_dir, str):
            if Path(f_ckpt_out_dir).resolve() != Path(call_out_dir).resolve():
                raise ValueError(
                    f"final_checkpoint embedded output_dir ({f_ckpt_out_dir}) does not match "
                    f"runtime train_call_resolved output_dir ({call_out_dir})"
                )

    # History reconciliation
    hist = prov.get("history")
    if not isinstance(hist, dict) or not hist:
        raise ValueError("source_provenance missing mandatory non-empty 'history'")
    n_ep = hist.get("n_epochs")
    if (
        type(n_ep) is bool
        or not isinstance(n_ep, (int, np.integer))
        or int(n_ep) != 300
    ):
        raise ValueError(
            f"source_provenance history n_epochs mismatch: expected integer 300, got {n_ep}"
        )
    if hist.get("best_val_loss") is None or not math.isclose(
        float(hist["best_val_loss"]), float(b_loss), rel_tol=1e-5, abs_tol=1e-5
    ):
        raise ValueError(
            f"source_provenance history best_val_loss ({hist.get('best_val_loss')}) does "
            f"not match best_checkpoint val_loss ({b_loss})"
        )
    b_ep0_hist = hist.get("best_epoch_zero_based")
    if (
        type(b_ep0_hist) is bool
        or not isinstance(b_ep0_hist, (int, np.integer))
        or int(b_ep0_hist) != int(b_epoch0)
    ):
        raise ValueError(
            f"source_provenance history best_epoch_zero_based ({b_ep0_hist}) does "
            f"not match best_checkpoint epoch_zero_based ({b_epoch0})"
        )
    if "best_epoch_one_based" in hist:
        b_ep1_hist = hist["best_epoch_one_based"]
        if (
            type(b_ep1_hist) is bool
            or not isinstance(b_ep1_hist, (int, np.integer))
            or int(b_ep1_hist) != int(b_epoch0) + 1
        ):
            raise ValueError(
                f"source_provenance history best_epoch_one_based mismatch: expected integer {int(b_epoch0) + 1}, got {b_ep1_hist}"
            )

    # Authoritative on-disk train.log resolution & hash verification (mandatory for candidates)
    p_tlog: Optional[Path] = None
    if source_seal.get("train_log") is not None:
        p_tlog = Path(source_seal["train_log"])
        if not p_tlog.is_absolute():
            p_tlog = (receipt_path.parent / p_tlog).resolve()
    elif call_out_dir:
        p_tlog = Path(call_out_dir) / "train.log"
        if not p_tlog.is_absolute():
            p_tlog = (receipt_path.parent / p_tlog).resolve()

    if p_tlog is None or not p_tlog.exists():
        raise FileNotFoundError(
            f"source_seal train_log path not found: {p_tlog}"
        )

    tlog_sha = source_seal.get("train_log_sha256")
    if (
        not tlog_sha
        or not isinstance(tlog_sha, str)
        or len(tlog_sha.strip()) != 64
        or not all(c in "0123456789abcdefABCDEF" for c in tlog_sha.strip())
    ):
        raise ValueError(
            f"source_seal missing mandatory 64-character hex 'train_log_sha256', got {tlog_sha!r}"
        )
    expected_tsha = tlog_sha.strip().lower()
    actual_tlog_sha = _compute_sha256(p_tlog)
    if actual_tlog_sha != expected_tsha:
        raise ValueError(
            f"source_seal train_log SHA mismatch: expected {expected_tsha}, "
            f"got {actual_tlog_sha}"
        )

    try:
        tlog_obj = json.loads(p_tlog.read_text(encoding="utf-8"))
    except Exception as err:
        raise ValueError(f"failed to parse train.log as JSON: {err}") from err
    if not isinstance(tlog_obj, dict):
        raise ValueError("train.log must be a JSON object")

    tlog_hist = (
        tlog_obj["history"]
        if "history" in tlog_obj and isinstance(tlog_obj["history"], dict)
        else tlog_obj
    )
    tlog_val_loss = tlog_hist.get("val_loss")
    if not isinstance(tlog_val_loss, list) or len(tlog_val_loss) != 300:
        raise ValueError(
            f"train.log history missing mandatory 300-length 'val_loss' array, "
            f"got {len(tlog_val_loss) if isinstance(tlog_val_loss, list) else type(tlog_val_loss).__name__}"
        )
    for idx, v in enumerate(tlog_val_loss):
        if (
            type(v) is bool
            or not isinstance(v, Real)
            or not math.isfinite(float(v))
        ):
            raise ValueError(
                f"train.log history val_loss[{idx}] must be finite numeric float, got {v!r}"
            )

    tlog_train_loss: Optional[List[Any]] = None
    if "train_loss" in tlog_hist and isinstance(tlog_hist["train_loss"], list):
        if len(tlog_hist["train_loss"]) != 300:
            raise ValueError(
                f"train.log history 'train_loss' must have 300 entries, got {len(tlog_hist['train_loss'])}"
            )
        for idx, v in enumerate(tlog_hist["train_loss"]):
            if (
                type(v) is bool
                or not isinstance(v, Real)
                or not math.isfinite(float(v))
            ):
                raise ValueError(
                    f"train.log history train_loss[{idx}] must be finite numeric float, got {v!r}"
                )
        tlog_train_loss = tlog_hist["train_loss"]

    val_loss_arr = tlog_val_loss

    # Reconcile inline history val_loss if present in receipt (no bypass permitted)
    if "val_loss" in hist:
        receipt_vl = hist["val_loss"]
        if not isinstance(receipt_vl, list) or len(receipt_vl) != 300:
            raise ValueError(
                f"source_provenance history inline 'val_loss' must have 300 entries, "
                f"got {len(receipt_vl) if isinstance(receipt_vl, list) else type(receipt_vl).__name__}"
            )
        for idx, (rv, tv) in enumerate(zip(receipt_vl, val_loss_arr)):
            if (
                type(rv) is bool
                or not isinstance(rv, Real)
                or not math.isfinite(float(rv))
            ):
                raise ValueError(
                    f"source_provenance history val_loss[{idx}] must be finite numeric float, got {rv!r}"
                )
            if not math.isclose(float(rv), float(tv), rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(
                    f"source_provenance history inline val_loss[{idx}] ({rv}) contradicts "
                    f"bound train.log val_loss ({tv})"
                )

    if "train_loss" in hist and isinstance(hist["train_loss"], list):
        if len(hist["train_loss"]) != 300:
            raise ValueError(
                f"source_provenance history train_loss must have 300 entries, got {len(hist['train_loss'])}"
            )
        for idx, v in enumerate(hist["train_loss"]):
            if (
                type(v) is bool
                or not isinstance(v, Real)
                or not math.isfinite(float(v))
            ):
                raise ValueError(
                    f"source_provenance history train_loss[{idx}] must be finite numeric float, got {v!r}"
                )
        if tlog_train_loss is not None:
            for idx, (rh_tl, tl_tl) in enumerate(zip(hist["train_loss"], tlog_train_loss)):
                if not math.isclose(float(rh_tl), float(tl_tl), rel_tol=1e-5, abs_tol=1e-5):
                    raise ValueError(
                        f"source_provenance history inline train_loss[{idx}] ({rh_tl}) contradicts "
                        f"bound train.log train_loss ({tl_tl})"
                    )

    # Reconcile strict '<' running-minimum best/first-tie selection (scripts/train.py:3878)
    calc_best_loss = float("inf")
    calc_best_epoch = -1
    for idx, v in enumerate(val_loss_arr):
        fv = float(v)
        if fv < calc_best_loss:
            calc_best_loss = fv
            calc_best_epoch = idx

    if not math.isclose(
        calc_best_loss, float(b_loss), rel_tol=1e-5, abs_tol=1e-5
    ):
        raise ValueError(
            f"bound train.log val_loss running minimum ({calc_best_loss:.6f}) does not match best_checkpoint val_loss ({float(b_loss):.6f})"
        )
    if calc_best_epoch != int(b_epoch0):
        raise ValueError(
            f"bound train.log val_loss running minimum epoch ({calc_best_epoch}) does not match best_checkpoint epoch_zero_based ({b_epoch0})"
        )
    if not math.isclose(
        float(val_loss_arr[-1]), float(f_loss), rel_tol=1e-5, abs_tol=1e-5
    ):
        raise ValueError(
            f"bound train.log val_loss final epoch ({float(val_loss_arr[-1]):.6f}) does not match final_checkpoint val_loss ({float(f_loss):.6f})"
        )

    # Reconcile parsed stdout telemetry against bound train.log arrays (strict producer %.6f token precision)
    for ep_num in range(1, 301):
        idx0 = ep_num - 1
        expected_vl = float(val_loss_arr[idx0])
        stdout_ep = parsed_stdout_telemetry[ep_num]
        stdout_vl = stdout_ep["val_loss"]
        stdout_vl_str = stdout_ep.get("val_loss_str", f"{stdout_vl:.6f}")
        expected_vl_fmt = format(expected_vl, ".6f")
        if stdout_vl_str != expected_vl_fmt and stdout_vl != float(expected_vl_fmt):
            raise ValueError(
                f"source_seal log Epoch {ep_num} val_loss ({stdout_vl_str}) contradicts "
                f"bound train.log val_loss ({expected_vl_fmt})"
            )
        if tlog_train_loss is not None:
            expected_tl = float(tlog_train_loss[idx0])
            stdout_tl = stdout_ep["train_loss"]
            stdout_tl_str = stdout_ep.get("train_loss_str", f"{stdout_tl:.6f}")
            expected_tl_fmt = format(expected_tl, ".6f")
            if stdout_tl_str != expected_tl_fmt and stdout_tl != float(expected_tl_fmt):
                raise ValueError(
                    f"source_seal log Epoch {ep_num} train_loss ({stdout_tl_str}) contradicts "
                    f"bound train.log train_loss ({expected_tl_fmt})"
                )

    # Checkpoints block in provenance
    ckpts_block = prov.get("checkpoints")
    if not isinstance(ckpts_block, dict) or not ckpts_block:
        raise ValueError("source_provenance missing mandatory non-empty 'checkpoints' block")
    if ckpts_block.get("best_sha256") != b_sha:
        raise ValueError(
            f"source_provenance checkpoints best_sha256 mismatch: expected {b_sha}, got {ckpts_block.get('best_sha256')}"
        )
    if ckpts_block.get("final_sha256") != f_sha:
        raise ValueError(
            f"source_provenance checkpoints final_sha256 mismatch: expected {f_sha}, got {ckpts_block.get('final_sha256')}"
        )
    if "final_epoch_zero_based" in ckpts_block:
        f_ep_hist = ckpts_block["final_epoch_zero_based"]
        if (
            type(f_ep_hist) is bool
            or not isinstance(f_ep_hist, (int, np.integer))
            or int(f_ep_hist) != 299
        ):
            raise ValueError(
                f"source_provenance checkpoints final_epoch_zero_based must be genuine integer 299, got {f_ep_hist!r}"
            )

    return raw_data


def validate_checkpoint_config(
    checkpoint_path: Path,
    expected_candidate_id: str,
    *,
    run_receipt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate model checkpoint against the frozen matched experiment contract."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    valid_ids = ("head_only_baseline", "k_0.5", "k_1.0")
    if expected_candidate_id not in valid_ids:
        raise ValueError(
            f"Unknown candidate_id '{expected_candidate_id}'; "
            f"must be one of {valid_ids}"
        )

    # 1. Strict baseline SHA enforcement
    actual_sha = _compute_sha256(checkpoint_path)
    if expected_candidate_id == "head_only_baseline":
        if actual_sha != BASELINE_CHECKPOINT_SHA256:
            raise ValueError(
                f"Baseline checkpoint SHA-256 mismatch: "
                f"expected {BASELINE_CHECKPOINT_SHA256}, got {actual_sha}"
            )

    ckpt_bytes = checkpoint_path.read_bytes()
    ckpt = load_artifact_bytes(ckpt_bytes, map_location="cpu")
    validate_checkpoint_dict_config(
        ckpt, expected_candidate_id, label="Checkpoint"
    )

    # 8. Selection epoch and finite non-negative loss metadata (allows 0.0)
    if "epoch" not in ckpt or "val_loss" not in ckpt or "best_val_loss" not in ckpt:
        raise ValueError(
            "Checkpoint missing required training loss / epoch metadata"
        )

    epoch_val = ckpt["epoch"]
    if type(epoch_val) is bool or not isinstance(epoch_val, (int, np.integer)):
        raise ValueError(
            f"Checkpoint epoch must be genuine int, got {type(epoch_val).__name__}"
        )
    if int(epoch_val) < 0:
        raise ValueError(
            f"Checkpoint epoch must be non-negative, got {epoch_val}"
        )

    for loss_key in ("val_loss", "best_val_loss"):
        l_val = ckpt[loss_key]
        if type(l_val) is bool or not isinstance(l_val, Real):
            raise ValueError(f"Checkpoint {loss_key} must be numeric scalar")
        f_l_val = float(l_val)
        if not math.isfinite(f_l_val) or f_l_val < 0.0:
            raise ValueError(
                f"Checkpoint {loss_key} must be finite non-negative float, got {l_val}"
            )

    # Selected checkpoint val_loss must be consistent with best_val_loss
    if not math.isclose(
        float(ckpt["val_loss"]),
        float(ckpt["best_val_loss"]),
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError(
            f"Selected checkpoint val_loss ({ckpt['val_loss']}) does not match "
            f"best_val_loss ({ckpt['best_val_loss']})"
        )

    # 9. Verify sealed run receipt (mandatory for candidates k_0.5 and k_1.0)
    if expected_candidate_id in ("k_0.5", "k_1.0"):
        if run_receipt_path is not None:
            validate_run_receipt(
                run_receipt_path, expected_candidate_id, actual_sha, ckpt
            )
        else:
            candidates_to_try = [
                checkpoint_path.parent / "assessor_report.json",
                checkpoint_path.parent / "run_receipt.json",
            ]
            found = False
            for cand_p in candidates_to_try:
                if cand_p.exists():
                    validate_run_receipt(
                        cand_p, expected_candidate_id, actual_sha, ckpt
                    )
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Candidate {expected_candidate_id} requires sealed run/assessor "
                    f"receipt evidence; none provided or found at {checkpoint_path.parent}"
                )
    elif run_receipt_path is not None:
        validate_run_receipt(
            run_receipt_path, expected_candidate_id, actual_sha, ckpt
        )

    return ckpt


def validate_cache_binding(
    cache_meta: Mapping[str, Any],
    y_true_seqs: Sequence[np.ndarray],
    composite_trial_ids: Sequence[str],
    prefix_ids: Sequence[str],
    expected_candidate_id: str,
    *,
    expected_checkpoint_sha: Optional[str] = None,
) -> None:
    """Strictly validate cache metadata binding across arm and family routes.

    Enforces matching candidate_id role, protocol dataset/prior/target hashes,
    exact ordered composite_trial_ids, exact ordered prefix_ids, exact
    trial_lengths, lineage, and checkpoint SHA.
    """
    if not cache_meta or not isinstance(cache_meta, Mapping):
        raise ValueError("Cache metadata must be a non-empty mapping")

    cand_id = cache_meta.get("candidate_id")
    if cand_id != expected_candidate_id:
        raise ValueError(
            f"Cache candidate_id mismatch: expected '{expected_candidate_id}', "
            f"got '{cand_id}'"
        )

    for hash_key, expected_hash in (
        ("dataset_sha256", EXPECTED_DATASET_SHA256),
        ("nested_prior_sha256", EXPECTED_NESTED_PRIOR_SHA256),
        ("full_target_sha256", EXPECTED_FULL_TARGET_SHA256),
        ("eligible_target_sha256", EXPECTED_ELIG_TARGET_SHA256),
    ):
        actual_hash = cache_meta.get(hash_key)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Cache {hash_key} mismatch: expected {expected_hash}, "
                f"got {actual_hash}"
            )

    actual_ckpt_sha = cache_meta.get("checkpoint_sha256")
    if expected_candidate_id in ("k_0.5", "k_1.0", "head_only_baseline"):
        if (
            not isinstance(actual_ckpt_sha, str)
            or len(actual_ckpt_sha) != 64
            or not all(c in "0123456789abcdefABCDEF" for c in actual_ckpt_sha)
        ):
            raise ValueError(
                f"Cache {expected_candidate_id} must have a valid 64-character hexadecimal checkpoint_sha256, got {actual_ckpt_sha!r}"
            )
    if expected_candidate_id == "head_only_baseline":
        if actual_ckpt_sha != BASELINE_CHECKPOINT_SHA256:
            raise ValueError(
                f"Cache baseline checkpoint_sha256 mismatch: expected {BASELINE_CHECKPOINT_SHA256}, got {actual_ckpt_sha}"
            )
    if expected_checkpoint_sha is not None:
        if actual_ckpt_sha != expected_checkpoint_sha:
            raise ValueError(
                f"Cache checkpoint_sha256 mismatch: "
                f"expected {expected_checkpoint_sha}, got {actual_ckpt_sha}"
            )

    # Validate exact ordered composite trial IDs
    saved_trial_ids = cache_meta.get("composite_trial_ids")
    if not isinstance(saved_trial_ids, list):
        raise ValueError("Cache metadata missing 'composite_trial_ids' list")
    if saved_trial_ids != list(composite_trial_ids):
        raise ValueError(
            "Cache composite_trial_ids do not match canonical trial IDs or ordering"
        )

    # Validate exact ordered trial lengths (reject floats, strings, bools)
    saved_lengths = cache_meta.get("trial_lengths")
    if not isinstance(saved_lengths, list):
        raise ValueError("Cache metadata missing 'trial_lengths' list")
    for idx, l in enumerate(saved_lengths):
        if type(l) is bool or not isinstance(l, (int, np.integer)):
            raise ValueError(
                f"Cache trial_lengths[{idx}] must be genuine integer, got {type(l).__name__}: {l}"
            )
    expected_lengths = [int(len(y)) for y in y_true_seqs]
    if [int(l) for l in saved_lengths] != expected_lengths:
        raise ValueError(
            "Cache trial_lengths do not match canonical target lengths"
        )

    # Validate mandatory ordered prefix IDs
    if "prefix_ids" not in cache_meta:
        raise ValueError("Cache metadata missing mandatory 'prefix_ids' list")
    saved_prefixes = cache_meta["prefix_ids"]
    if not isinstance(saved_prefixes, list):
        raise ValueError("Cache metadata 'prefix_ids' must be a list")
    if saved_prefixes != list(prefix_ids):
        raise ValueError("Cache prefix_ids do not match canonical prefix IDs")

    # Validate mandatory structured lineage
    if "lineage" not in cache_meta:
        raise ValueError("Cache metadata missing mandatory 'lineage' dictionary")
    lin = cache_meta["lineage"]
    if not isinstance(lin, dict):
        raise ValueError(f"Cache lineage must be a dictionary, got {type(lin).__name__}")
    if lin.get("validation_scope") != "nested_outer_validation":
        raise ValueError("Cache lineage validation_scope mismatch")
    if lin.get("nested_split_seed") != 42:
        raise ValueError("Cache lineage nested_split_seed mismatch")
    if lin.get("nested_val_split") != 0.2:
        raise ValueError("Cache lineage nested_val_split mismatch")
    crop = lin.get("crop")
    if not isinstance(crop, dict):
        raise ValueError("Cache lineage missing mandatory 'crop' dictionary")
    if crop.get("max_seq_len") != 2400 or crop.get("pre_anchor_frames") != 1200:
        raise ValueError("Cache lineage crop configuration mismatch")

    # If config block is present in cache, ensure it matches matched experiment controls
    if "config" in cache_meta:
        cfg = cache_meta["config"]
        if not isinstance(cfg, dict):
            raise ValueError("Cache config must be a dictionary")
        tr = cfg.get("training", {})
        if isinstance(tr, dict):
            if tr.get("num_epochs") not in (None, 300):
                raise ValueError(
                    f"Cache config training num_epochs must be 300, got {tr.get('num_epochs')}"
                )
            if tr.get("random_seed") not in (None, 42):
                raise ValueError(
                    f"Cache config training random_seed must be 42, got {tr.get('random_seed')}"
                )


def extract_canonical_validation_targets(
    dataset_path: Path,
    nested_prior_path: Path,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Extract canonical validation target sequences, composite trial IDs, and prefixes.

    Validates target binding hashes and trial counts before model execution.
    """
    validate_immutable_pins(dataset_path, nested_prior_path)
    nest_data = load_artifact_bytes(
        nested_prior_path.read_bytes(), map_location="cpu"
    )
    ds_data = load_artifact_bytes(
        dataset_path.read_bytes(), map_location="cpu"
    )

    val_indices = [int(i) for i in nest_data["val_indices"]]
    group_keys = nest_data["group_keys"]
    session_ids = ds_data.get("session_ids")
    trial_ids_raw = ds_data["trial_ids"]
    anchor_frames = ds_data.get("anchor_frames")
    y_seqs_raw = ds_data["Y_seqs"]

    if session_ids is None:
        raise ValueError("Dataset missing required 'session_ids' array")

    y_true_seqs: List[np.ndarray] = []
    composite_trial_ids: List[str] = []
    prefix_ids: List[str] = []

    for idx in val_indices:
        y_raw = y_seqs_raw[idx]
        anc = anchor_frames[idx] if anchor_frames is not None else None
        # Canonical anchor-aligned crop if sequence exceeds max_seq_len (2400)
        if len(y_raw) > 2400 and anc is not None:
            start, end = resolve_anchor_crop(
                n_frames=len(y_raw),
                anchor_frame=anc,
                max_seq_len=2400,
                pre_anchor_frames=1200,
            )
            y_cropped = y_raw[start:end]
        else:
            y_cropped = y_raw

        y_arr = np.asarray(y_cropped, dtype=np.float64)
        y_true_seqs.append(y_arr)
        # Composite trial ID: session_id + raw trial_id
        composite_id = f"{session_ids[idx]}::{trial_ids_raw[idx]}"
        composite_trial_ids.append(composite_id)
        prefix_ids.append(str(group_keys[idx]))

    # Target binding check (fail closed)
    expected_binding = {
        "n_trials": EXPECTED_VAL_TRIALS,
        "n_full_frames": EXPECTED_FULL_FRAMES,
        "n_eligible_frames": EXPECTED_ELIG_FRAMES,
        "full_target_sha256": EXPECTED_FULL_TARGET_SHA256,
        "eligible_target_sha256": EXPECTED_ELIG_TARGET_SHA256,
    }
    validate_target_binding(y_true_seqs, expected_binding)

    return y_true_seqs, composite_trial_ids, prefix_ids


def evaluate_arm(
    candidate_id: str,
    checkpoint_path: Path,
    dataset_path: Path,
    nested_prior_path: Path,
    output_dir: Path,
    *,
    load_cache_path: Optional[Path] = None,
    save_cache: bool = False,
    run_receipt_path: Optional[Path] = None,
    exchangeability_asserted: bool = False,
    allow_overwrite: bool = False,
) -> Dict[str, Any]:
    """Execute complete evaluation of a single model arm (inference or cache)."""
    if type(exchangeability_asserted) is not bool:
        raise ValueError(
            "exchangeability_asserted must be a genuine bool, got "
            f"{type(exchangeability_asserted).__name__}"
        )

    # Preflight all requested output paths BEFORE reading checkpoint, data, or computing
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / f"{candidate_id}_metrics.json"
    cache_base = output_dir / f"{candidate_id}_predictions"
    cache_npz = output_dir / f"{candidate_id}_predictions.npz"
    cache_json = output_dir / f"{candidate_id}_predictions.json"

    if not allow_overwrite:
        if metrics_file.exists():
            raise FileExistsError(f"Metrics file already exists: {metrics_file}")
        if save_cache:
            if cache_npz.exists():
                raise FileExistsError(f"Prediction cache NPZ exists: {cache_npz}")
            if cache_json.exists():
                raise FileExistsError(
                    f"Prediction cache JSON exists: {cache_json}"
                )

    ckpt_dict = validate_checkpoint_config(
        checkpoint_path, candidate_id, run_receipt_path=run_receipt_path
    )

    y_true_seqs, composite_trial_ids, prefix_ids = (
        extract_canonical_validation_targets(dataset_path, nested_prior_path)
    )

    y_pred_seqs: List[np.ndarray] = []

    if load_cache_path is not None:
        logging.info("Loading predictions from cache: %s", load_cache_path)
        loaded_preds, cache_meta = load_prediction_cache(load_cache_path)

        ckpt_sha = _compute_sha256(checkpoint_path)
        validate_cache_binding(
            cache_meta=cache_meta,
            y_true_seqs=y_true_seqs,
            composite_trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
            expected_candidate_id=candidate_id,
            expected_checkpoint_sha=ckpt_sha,
        )

        if len(loaded_preds) != len(y_true_seqs):
            raise ValueError(
                f"Cache trial count mismatch: {len(loaded_preds)} != {len(y_true_seqs)}"
            )
        for i, (p, t) in enumerate(zip(loaded_preds, y_true_seqs)):
            if p.shape != t.shape:
                raise ValueError(
                    f"Cache trial {i} shape mismatch: {p.shape} != {t.shape}"
                )
        y_pred_seqs = loaded_preds
    else:
        cfg = ExperimentConfig.from_dict(ckpt_dict["config"])
        cfg.training.num_workers = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_from_checkpoint(checkpoint_path, device)
        _, val_loader = build_dataloaders(
            config=cfg,
            dataset_path=str(dataset_path),
            nested_prior_artifact=str(nested_prior_path),
        )
        if val_loader is None:
            raise RuntimeError("Failed to build validation dataloader")

        model.eval()
        loader_true_seqs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x_b, y_b, len_b = batch
                elif len(batch) == 4:
                    x_b, y_b, len_b, _wind = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")

                x_b = x_b.to(device).contiguous()
                len_b = len_b.to(device).contiguous()
                y_p, _ = model(x_b, len_b, return_internals=True)
                for i in range(x_b.size(0)):
                    n = int(len_b[i])
                    yt_i = y_b[i, :n].cpu().numpy().astype(np.float64)
                    yp_i = y_p[i, :n].cpu().numpy().astype(np.float64)
                    loader_true_seqs.append(yt_i)
                    y_pred_seqs.append(yp_i)

        # Validate loader-produced targets match canonical targets
        if len(loader_true_seqs) != len(y_true_seqs):
            raise ValueError(
                f"Dataloader trial count {len(loader_true_seqs)} != "
                f"canonical {len(y_true_seqs)}"
            )
        for i, (lt, ct) in enumerate(zip(loader_true_seqs, y_true_seqs)):
            if not np.array_equal(lt, ct):
                raise ValueError(
                    f"Dataloader trial {i} targets do not match canonical"
                )

    # Compute grid metrics
    eval_results = compute_aligned_grid_metrics(
        y_true_seqs=y_true_seqs,
        y_pred_seqs=y_pred_seqs,
        trial_ids=composite_trial_ids,
        prefix_ids=prefix_ids,
    )

    # Save cache if requested
    if save_cache:
        recorded_receipt = run_receipt_path
        if recorded_receipt is None and candidate_id in ("k_0.5", "k_1.0"):
            for cand_p in [
                checkpoint_path.parent / "assessor_report.json",
                checkpoint_path.parent / "run_receipt.json",
            ]:
                if cand_p.exists():
                    recorded_receipt = cand_p
                    break

        recorded_receipt_sha = None
        if recorded_receipt is not None:
            p_rec = Path(recorded_receipt)
            if not p_rec.is_absolute():
                p_rec = (checkpoint_path.parent / p_rec).resolve()
            if p_rec.exists():
                recorded_receipt_sha = _compute_sha256(p_rec)

        cache_meta = {
            "candidate_id": candidate_id,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _compute_sha256(checkpoint_path),
            "run_receipt": str(recorded_receipt) if recorded_receipt else None,
            "run_receipt_sha256": recorded_receipt_sha,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "nested_prior_sha256": EXPECTED_NESTED_PRIOR_SHA256,
            "full_target_sha256": EXPECTED_FULL_TARGET_SHA256,
            "eligible_target_sha256": EXPECTED_ELIG_TARGET_SHA256,
            "trial_lengths": [len(y) for y in y_true_seqs],
            "composite_trial_ids": composite_trial_ids,
            "prefix_ids": prefix_ids,
            "lineage": {
                "validation_scope": "nested_outer_validation",
                "nested_split_seed": 42,
                "nested_val_split": 0.2,
                "crop": {
                    "max_seq_len": 2400,
                    "pre_anchor_frames": 1200,
                },
            },
        }
        save_prediction_cache(
            cache_base,
            y_pred_seqs,
            metadata=cache_meta,
            allow_overwrite=allow_overwrite,
        )

    # Write metrics JSON with complete scope data
    metrics_payload = {
        "candidate_id": candidate_id,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _compute_sha256(checkpoint_path),
        "exchangeability_asserted": exchangeability_asserted,
        "results": {
            "raw_full_frame": eval_results["raw_full_frame"],
            "aligned_overall": eval_results["aligned_overall"],
            "sustained_cells": eval_results["sustained_cells"],
            "scope_data": eval_results["scope_data"],
        },
    }
    metrics_file.write_text(
        json.dumps(metrics_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    return {
        "candidate_id": candidate_id,
        "eval_results": eval_results,
        "composite_trial_ids": composite_trial_ids,
        "prefix_ids": prefix_ids,
    }


def assemble_and_save_family(
    baseline_cache_path: Path,
    cand_05_cache_path: Optional[Path],
    cand_10_cache_path: Optional[Path],
    dataset_path: Path,
    nested_prior_path: Path,
    output_path: Path,
    *,
    exchangeability_asserted: bool = False,
    allow_overwrite: bool = False,
) -> Dict[str, Any]:
    """Assemble the frozen 78-slot inferential family from cache files."""
    if type(exchangeability_asserted) is not bool:
        raise ValueError(
            "exchangeability_asserted must be a genuine bool, got "
            f"{type(exchangeability_asserted).__name__}"
        )

    if output_path.exists() and not allow_overwrite:
        raise FileExistsError(f"Family output file already exists: {output_path}")

    # Refuse identical cache files across baseline and candidate
    if cand_05_cache_path is not None:
        if cand_05_cache_path.resolve() == baseline_cache_path.resolve():
            raise ValueError(
                "Candidate k_0.5 cache cannot be identical to baseline cache"
            )
    if cand_10_cache_path is not None:
        if cand_10_cache_path.resolve() == baseline_cache_path.resolve():
            raise ValueError(
                "Candidate k_1.0 cache cannot be identical to baseline cache"
            )

    y_true_seqs, composite_trial_ids, prefix_ids = (
        extract_canonical_validation_targets(dataset_path, nested_prior_path)
    )

    # 1. Baseline predictions & scope data
    base_preds, base_meta = load_prediction_cache(baseline_cache_path)
    validate_cache_binding(
        cache_meta=base_meta,
        y_true_seqs=y_true_seqs,
        composite_trial_ids=composite_trial_ids,
        prefix_ids=prefix_ids,
        expected_candidate_id="head_only_baseline",
        expected_checkpoint_sha=BASELINE_CHECKPOINT_SHA256,
    )
    if "checkpoint" in base_meta and base_meta["checkpoint"]:
        base_ckpt_p = Path(base_meta["checkpoint"])
        if not base_ckpt_p.is_absolute():
            base_ckpt_p = (baseline_cache_path.parent / base_ckpt_p).resolve()
        if base_ckpt_p.exists():
            base_sha = _compute_sha256(base_ckpt_p)
            if base_sha != BASELINE_CHECKPOINT_SHA256:
                raise ValueError(
                    f"Baseline checkpoint SHA mismatch: expected {BASELINE_CHECKPOINT_SHA256}, got {base_sha}"
                )
    base_grid = compute_aligned_grid_metrics(
        y_true_seqs=y_true_seqs,
        y_pred_seqs=base_preds,
        trial_ids=composite_trial_ids,
        prefix_ids=prefix_ids,
    )

    # Comparators scope data: baseline, persistence, zero
    comparator_scopes: Dict[str, Dict[str, Any]] = {
        "head_only_baseline": base_grid["scope_data"],
        "persistence": base_grid["scope_data"],
        "zero": base_grid["scope_data"],
    }

    # 2. Candidate scopes
    candidate_scopes: Dict[str, Dict[str, Any]] = {}
    cache_provenance: Dict[str, Any] = {
        "head_only_baseline": {
            "checkpoint_sha256": base_meta.get("checkpoint_sha256"),
            "dataset_sha256": base_meta.get("dataset_sha256"),
            "nested_prior_sha256": base_meta.get("nested_prior_sha256"),
            "full_target_sha256": base_meta.get("full_target_sha256"),
            "eligible_target_sha256": base_meta.get("eligible_target_sha256"),
            "n_trials": len(base_preds),
        }
    }

    if cand_05_cache_path is not None and cand_05_cache_path.exists():
        preds_05, meta_05 = load_prediction_cache(cand_05_cache_path)
        validate_cache_binding(
            cache_meta=meta_05,
            y_true_seqs=y_true_seqs,
            composite_trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
            expected_candidate_id="k_0.5",
        )
        # Substantive verification: candidate checkpoint and run receipt chain
        ckpt_str_05 = meta_05.get("checkpoint")
        if not ckpt_str_05:
            raise ValueError(
                "Candidate k_0.5 cache metadata missing mandatory 'checkpoint' path"
            )
        ckpt_p_05 = Path(ckpt_str_05)
        if not ckpt_p_05.is_absolute():
            ckpt_p_05 = (cand_05_cache_path.parent / ckpt_p_05).resolve()
        if not ckpt_p_05.exists():
            raise FileNotFoundError(
                f"Candidate k_0.5 checkpoint file not found: {ckpt_p_05}"
            )
        actual_sha_05 = _compute_sha256(ckpt_p_05)
        if actual_sha_05 != meta_05.get("checkpoint_sha256"):
            raise ValueError(
                f"Candidate k_0.5 checkpoint SHA mismatch: expected {meta_05.get('checkpoint_sha256')}, "
                f"got {actual_sha_05}"
            )
        rec_str_05 = meta_05.get("run_receipt")
        if not rec_str_05:
            raise ValueError(
                "Candidate k_0.5 cache metadata missing mandatory 'run_receipt' path"
            )
        rec_p_05 = Path(rec_str_05)
        if not rec_p_05.is_absolute():
            rec_p_05 = (cand_05_cache_path.parent / rec_p_05).resolve()
        if not rec_p_05.exists():
            raise FileNotFoundError(
                f"Candidate k_0.5 run receipt file not found: {rec_p_05}"
            )
        rec_sha_05 = _compute_sha256(rec_p_05)
        expected_rec_sha_05 = meta_05.get("run_receipt_sha256")
        if not expected_rec_sha_05:
            raise ValueError(
                "Candidate k_0.5 cache metadata missing mandatory 'run_receipt_sha256'"
            )
        if rec_sha_05 != expected_rec_sha_05:
            raise ValueError(
                f"Candidate k_0.5 run receipt SHA mismatch: expected {expected_rec_sha_05}, "
                f"got {rec_sha_05}"
            )
        validate_checkpoint_config(
            ckpt_p_05,
            "k_0.5",
            run_receipt_path=rec_p_05,
        )
        grid_05 = compute_aligned_grid_metrics(
            y_true_seqs=y_true_seqs,
            y_pred_seqs=preds_05,
            trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
        )
        candidate_scopes["k_0.5"] = grid_05["scope_data"]
        cache_provenance["k_0.5"] = {
            "checkpoint_sha256": meta_05.get("checkpoint_sha256"),
            "run_receipt_sha256": meta_05.get("run_receipt_sha256"),
            "dataset_sha256": meta_05.get("dataset_sha256"),
            "nested_prior_sha256": meta_05.get("nested_prior_sha256"),
            "full_target_sha256": meta_05.get("full_target_sha256"),
            "eligible_target_sha256": meta_05.get("eligible_target_sha256"),
            "n_trials": len(preds_05),
        }

    if cand_10_cache_path is not None and cand_10_cache_path.exists():
        preds_10, meta_10 = load_prediction_cache(cand_10_cache_path)
        validate_cache_binding(
            cache_meta=meta_10,
            y_true_seqs=y_true_seqs,
            composite_trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
            expected_candidate_id="k_1.0",
        )
        # Substantive verification: candidate checkpoint and run receipt chain
        ckpt_str_10 = meta_10.get("checkpoint")
        if not ckpt_str_10:
            raise ValueError(
                "Candidate k_1.0 cache metadata missing mandatory 'checkpoint' path"
            )
        ckpt_p_10 = Path(ckpt_str_10)
        if not ckpt_p_10.is_absolute():
            ckpt_p_10 = (cand_10_cache_path.parent / ckpt_p_10).resolve()
        if not ckpt_p_10.exists():
            raise FileNotFoundError(
                f"Candidate k_1.0 checkpoint file not found: {ckpt_p_10}"
            )
        actual_sha_10 = _compute_sha256(ckpt_p_10)
        if actual_sha_10 != meta_10.get("checkpoint_sha256"):
            raise ValueError(
                f"Candidate k_1.0 checkpoint SHA mismatch: expected {meta_10.get('checkpoint_sha256')}, "
                f"got {actual_sha_10}"
            )
        rec_str_10 = meta_10.get("run_receipt")
        if not rec_str_10:
            raise ValueError(
                "Candidate k_1.0 cache metadata missing mandatory 'run_receipt' path"
            )
        rec_p_10 = Path(rec_str_10)
        if not rec_p_10.is_absolute():
            rec_p_10 = (cand_10_cache_path.parent / rec_p_10).resolve()
        if not rec_p_10.exists():
            raise FileNotFoundError(
                f"Candidate k_1.0 run receipt file not found: {rec_p_10}"
            )
        rec_sha_10 = _compute_sha256(rec_p_10)
        expected_rec_sha_10 = meta_10.get("run_receipt_sha256")
        if not expected_rec_sha_10:
            raise ValueError(
                "Candidate k_1.0 cache metadata missing mandatory 'run_receipt_sha256'"
            )
        if rec_sha_10 != expected_rec_sha_10:
            raise ValueError(
                f"Candidate k_1.0 run receipt SHA mismatch: expected {expected_rec_sha_10}, "
                f"got {rec_sha_10}"
            )
        validate_checkpoint_config(
            ckpt_p_10,
            "k_1.0",
            run_receipt_path=rec_p_10,
        )
        grid_10 = compute_aligned_grid_metrics(
            y_true_seqs=y_true_seqs,
            y_pred_seqs=preds_10,
            trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
        )
        candidate_scopes["k_1.0"] = grid_10["scope_data"]
        cache_provenance["k_1.0"] = {
            "checkpoint_sha256": meta_10.get("checkpoint_sha256"),
            "run_receipt_sha256": meta_10.get("run_receipt_sha256"),
            "dataset_sha256": meta_10.get("dataset_sha256"),
            "nested_prior_sha256": meta_10.get("nested_prior_sha256"),
            "full_target_sha256": meta_10.get("full_target_sha256"),
            "eligible_target_sha256": meta_10.get("eligible_target_sha256"),
            "n_trials": len(preds_10),
        }

    family_results = assemble_declared_family(
        candidate_scopes=candidate_scopes,
        comparator_scopes=comparator_scopes,
        exchangeability_asserted=exchangeability_asserted,
    )
    family_results["cache_provenance"] = cache_provenance

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(family_results, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return family_results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Evaluate NSMoR model optimization protocol candidate."
    )
    parser.add_argument(
        "--candidate_id",
        type=str,
        default=None,
        choices=["head_only_baseline", "k_0.5", "k_1.0"],
        help="Model identifier (head_only_baseline, k_0.5, or k_1.0).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint .pth file.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            ".scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/"
            "etl-causal-final-zp12m0fa/nsmor_dataset.pt"
        ),
        help="Path to canonical dataset .pt file.",
    )
    parser.add_argument(
        "--nested_prior",
        type=Path,
        default=Path(
            ".scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/"
            "nested-causal-complete-dku91_m0/nested_split_seed42.pt"
        ),
        help="Path to canonical nested split .pt file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/candidate_eval"),
        help="Isolated directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--save_cache",
        action="store_true",
        help="Save predictions to flat NPZ cache.",
    )
    parser.add_argument(
        "--load_cache",
        type=Path,
        default=None,
        help="Load predictions from existing cache instead of forward pass.",
    )
    parser.add_argument(
        "--run_receipt",
        type=Path,
        default=None,
        help="Path to sealed run receipt JSON for candidate verification.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate pins and configuration without running forward pass.",
    )
    parser.add_argument(
        "--exchangeability_asserted",
        action="store_true",
        help="Assert joint sign-exchangeability null for permutation test.",
    )
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow overwriting existing outputs in output directory.",
    )
    # Family assembly arguments
    parser.add_argument(
        "--assemble_family",
        action="store_true",
        help="Assemble frozen 78-slot inferential family from caches.",
    )
    parser.add_argument(
        "--baseline_cache",
        type=Path,
        default=None,
        help="Path to baseline prediction cache for family assembly.",
    )
    parser.add_argument(
        "--cand_05_cache",
        type=Path,
        default=None,
        help="Path to k_0.5 prediction cache for family assembly.",
    )
    parser.add_argument(
        "--cand_10_cache",
        type=Path,
        default=None,
        help="Path to k_1.0 prediction cache for family assembly.",
    )
    parser.add_argument(
        "--family_output",
        type=Path,
        default=None,
        help="Output path for family JSON results.",
    )
    args = parser.parse_args()

    if args.assemble_family:
        if args.baseline_cache is None:
            parser.error("--assemble_family requires --baseline_cache")
        out_family = (
            args.family_output
            if args.family_output is not None
            else args.output_dir / "declared_family_78_slots.json"
        )
        # Preflight family output before loading
        if out_family.exists() and not args.allow_overwrite:
            raise FileExistsError(
                f"Family output file already exists: {out_family}"
            )

        logging.info("Assembling declared 78-slot family...")
        assemble_and_save_family(
            baseline_cache_path=args.baseline_cache,
            cand_05_cache_path=args.cand_05_cache,
            cand_10_cache_path=args.cand_10_cache,
            dataset_path=args.dataset,
            nested_prior_path=args.nested_prior,
            output_path=out_family,
            exchangeability_asserted=args.exchangeability_asserted,
            allow_overwrite=args.allow_overwrite,
        )
        logging.info("Family assembly complete. Output written to %s", out_family)
        return 0

    if args.candidate_id is None or args.checkpoint is None:
        parser.error("Scoring an arm requires --candidate_id and --checkpoint")

    # Preflight arm outputs before input loading or model execution
    metrics_file = args.output_dir / f"{args.candidate_id}_metrics.json"
    if metrics_file.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Metrics file already exists: {metrics_file}")
    if args.save_cache:
        cache_npz = args.output_dir / f"{args.candidate_id}_predictions.npz"
        cache_json = args.output_dir / f"{args.candidate_id}_predictions.json"
        if (cache_npz.exists() or cache_json.exists()) and not args.allow_overwrite:
            raise FileExistsError(
                "Prediction cache files already exist in output directory"
            )

    # Step 1: Pre-flight immutable pin validation
    logging.info("Validating immutable dataset and split pins...")
    validate_immutable_pins(args.dataset, args.nested_prior)

    logging.info("Validating checkpoint configuration...")
    validate_checkpoint_config(
        args.checkpoint, args.candidate_id, run_receipt_path=args.run_receipt
    )

    if args.dry_run:
        logging.info("Dry run complete: pins and configuration verified.")
        return 0

    # Step 2: Evaluate candidate arm
    evaluate_arm(
        candidate_id=args.candidate_id,
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        nested_prior_path=args.nested_prior,
        output_dir=args.output_dir,
        load_cache_path=args.load_cache,
        save_cache=args.save_cache,
        run_receipt_path=args.run_receipt,
        exchangeability_asserted=args.exchangeability_asserted,
        allow_overwrite=args.allow_overwrite,
    )
    logging.info("Evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
