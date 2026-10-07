"""Synthetic contract tests for NSMoR model optimization evaluation.

Verifies the mathematical and structural invariants declared in
``docs/model-optimization-protocol-20261004.md`` and reviews:
  1. Target/frame identity and hash validation fail-closed on mismatch.
  2. Trial boundaries: persistence and sustained runs never cross trials.
  3. Empty masks/unavailable scopes emit null metrics with explicit reasons.
  4. Complete 78-slot inferential family assembly and Holm-Bonferroni reduction.
  5. Contiguous flat NPZ cache round-trip with allow_pickle=False and safe paths.
  6. Non-overflowing Python integer cache length validation (no uint64 wrap).
  7. Matched experiment checkpoint configuration matching and pin enforcement.
  8. Fail-closed candidate sealed run/assessor receipt evidence validation.
  9. Shared cache binding contract (role, checkpoint, dataset, prior, targets,
     ordered composite IDs, trial lengths).
  10. Early preflight refusal on existing outputs before computation or side effects.
  11. Synthetic CLI integration: mocked loaders, batch formats, and refusal gates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import torch

from nsmor.analysis.model_comparison import (
    CANDIDATE_IDS,
    COMPARATOR_IDS,
    DECLARED_FAMILY_SIZE,
    declared_family_slots,
)
from nsmor.analysis.optimization_evaluation import (
    ALIGNED_OVERALL_SCOPE,
    assemble_declared_family,
    compute_aligned_grid_metrics,
    compute_regression_metrics,
    compute_skill_ratio,
    hash_target_sequences,
    load_prediction_cache,
    save_prediction_cache,
    validate_target_binding,
)
from scripts.evaluate_model_optimization import (
    BASELINE_CHECKPOINT_SHA256,
    BASELINE_MODEL_CONTROLS,
    BASELINE_SAVED_CONFIG,
    BASELINE_TRAINING_CONTROLS,
    EXPECTED_DATASET_SHA256,
    EXPECTED_ELIG_FRAMES,
    EXPECTED_ELIG_TARGET_SHA256,
    EXPECTED_FULL_FRAMES,
    EXPECTED_FULL_TARGET_SHA256,
    EXPECTED_NESTED_PRIOR_SHA256,
    EXPECTED_VAL_TRIALS,
    assemble_and_save_family,
    evaluate_arm,
    validate_cache_binding,
    validate_checkpoint_config,
    validate_immutable_pins,
    validate_run_receipt,
)


def _compute_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _make_dummy_file_with_sha(path: Path, content: bytes = b"dummy content") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _make_valid_checkpoint_dict(
    candidate_id: str = "k_0.5",
    persistence_skip: float = 0.5,
) -> dict:
    """Build a synthetic checkpoint dict satisfying matched experiment controls."""
    cfg = copy.deepcopy(BASELINE_SAVED_CONFIG)
    cfg["model"]["persistence_skip"] = persistence_skip
    if candidate_id == "head_only_baseline":
        cfg["checkpoint"]["output_dir"] = ".scratch/model-opt-baseline-300-20261003-rerun"
    else:
        cfg["checkpoint"]["output_dir"] = f".scratch/cand_{candidate_id}"
    return {
        "config": cfg,
        "epoch": 295,
        "val_loss": 7.196,
        "best_val_loss": 7.196,
        "pipeline_semantics_version": "2.2",
        "nested_prior_fingerprint": EXPECTED_DATASET_SHA256,
        "is_nested_cv": True,
        "validation_scope": "nested_outer_validation",
        "nested_split_seed": 42,
        "nested_val_split": 0.2,
        "target_mean": 0.0,
        "target_std": 1.0,
        "target_clip_cm_s": 0.0,
        "dataset_source_sha256": EXPECTED_DATASET_SHA256,
        "nested_prior_artifact_sha256": EXPECTED_NESTED_PRIOR_SHA256,
        "model_state_dict": {
            "weight": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "bias": torch.tensor([0.1], dtype=torch.float32),
        },
    }


def _make_valid_assessor_report_dict(
    ckpt_sha: str,
    ckpt_epoch: int = 295,
    ckpt_val_loss: float = 7.196,
    seal_dir: Optional[Path] = None,
) -> dict:
    """Build a synthetic sealed assessor report satisfying fail-closed receipt schema."""
    out_d_str = ".scratch/cand_k_0.5"
    val_loss_arr = [7.500] * 300
    val_loss_arr[ckpt_epoch] = ckpt_val_loss
    val_loss_arr[299] = 7.200
    train_loss_arr = [8.000] * 300

    if seal_dir is None:
        p_rec = Path("/tmp/dummy_sealed_receipt.json")
        p_log = Path("/tmp/dummy_stdout.log")
        p_final = Path("/tmp/dummy_final.pth")
        p_man = Path("/tmp/dummy_manifest.json")
        p_tlog = Path("/tmp/dummy_train.log")
        rec_sha = "a" * 64
        log_sha = "b" * 64
        final_sha = "f" * 64
        man_sha = "m" * 64
        tlog_sha = "t" * 64
    else:
        seal_dir.mkdir(parents=True, exist_ok=True)
        p_rec = seal_dir / "sealed_receipt.json"
        p_log = seal_dir / "stdout.log"
        p_final = seal_dir / "final_model.pth"
        p_man = seal_dir / "manifest.json"
        p_tlog = seal_dir / "train.log"

        # Valid final checkpoint dict
        final_dict = _make_valid_checkpoint_dict("k_0.5", 0.5)
        final_dict["epoch"] = 299
        final_dict["val_loss"] = 7.200
        final_dict["best_val_loss"] = 7.196
        torch.save(final_dict, p_final)
        final_sha = _compute_sha(p_final)

        # Valid sealer manifest (kind="source" from tester_seal)
        man_data = {
            "label": "candidate_seal",
            "kind": "source",
            "created_utc": "2026-10-05T00:00:00Z",
            "cwd": "/mnt/d/Projects/NSMoR",
            "paths": {
                "scripts/train.py": "a" * 64,
                "nsmor/model_nsmor_core.py": "b" * 64,
            },
        }
        man_bytes = json.dumps(man_data).encode("utf-8")
        man_sha = _make_dummy_file_with_sha(p_man, man_bytes)

        # Authoritative on-disk train.log with 300 epochs of losses
        tlog_data = {
            "best_val_loss": ckpt_val_loss,
            "final_train_loss": 8.000,
            "history": {
                "train_loss": train_loss_arr,
                "val_loss": val_loss_arr,
            },
        }
        tlog_bytes = json.dumps(tlog_data).encode("utf-8")
        tlog_sha = _make_dummy_file_with_sha(p_tlog, tlog_bytes)

        # Contiguous 300-epoch progress telemetry plus canonical training completion
        log_lines = []
        for i in range(1, 301):
            tl_val = train_loss_arr[i - 1]
            vl_val = val_loss_arr[i - 1]
            log_lines.append(
                f"Epoch {i}/300  train_loss={tl_val:.6f}  val_loss={vl_val:.6f}  time=0.1s"
            )
        log_lines.append(f"Training complete.  Best val loss: {ckpt_val_loss:.6f}")
        log_bytes = "\n".join(log_lines).encode("utf-8")
        log_sha = _make_dummy_file_with_sha(p_log, log_bytes)

        rec_bytes = json.dumps(
            {
                "status": "PASS",
                "exit_code": 0,
                "actual_completed_epochs": 300,
                "resumed": False,
                "log": str(p_log),
                "log_sha256": log_sha,
                "manifest": str(p_man),
                "manifest_sha256": man_sha,
            }
        ).encode("utf-8")
        rec_sha = _make_dummy_file_with_sha(p_rec, rec_bytes)

    return {
        "status": "PASS",
        "exit_code": 0,
        "evaluator_binding": {
            "role": "candidate_k0.5",
            "declared_epochs": 300,
            "actual_completed_epochs": 300,
            "completion": "natural_completion",
            "natural_exit_zero": True,
            "sealed_status": "PASS",
            "not_resumed": True,
            "resumed_from": None,
            "early_stopped": False,
            "protocol": {
                "persistence_skip_k": 0.5,
                "random_seed": 42,
                "num_epochs": 300,
                "normalize_targets": False,
                "target_clip_cm_s": 0.0,
                "target_mean": 0.0,
                "target_std": 1.0,
            },
            "pins": {
                "pipeline_semantics_version": "2.2",
                "nested_prior_fingerprint": EXPECTED_DATASET_SHA256,
                "dataset_source_sha256": EXPECTED_DATASET_SHA256,
                "nested_prior_artifact_sha256": EXPECTED_NESTED_PRIOR_SHA256,
                "validation_scope": "nested_outer_validation",
                "nested_split_seed": 42,
            },
            "best_checkpoint": {
                "sha256": ckpt_sha,
                "val_loss": ckpt_val_loss,
                "best_val_loss": ckpt_val_loss,
                "epoch_zero_based": ckpt_epoch,
                "epoch_one_based": ckpt_epoch + 1,
            },
            "final_checkpoint": {
                "path": str(p_final),
                "sha256": final_sha,
                "epoch_zero_based": 299,
                "val_loss": 7.200,
            },
            "source_seal": {
                "sealed_receipt": str(p_rec),
                "sealed_receipt_sha256": rec_sha,
                "log": str(p_log),
                "log_sha256": log_sha,
                "manifest": str(p_man),
                "manifest_sha256": man_sha,
                "train_log": str(p_tlog),
                "train_log_sha256": tlog_sha,
            },
            "source_provenance": {
                "command": ["python", "scripts/train.py"],
                "cwd": "/mnt/d/Projects/NSMoR",
                "sealed_exit_code": 0,
                "sealed_status": "PASS",
                "runtime_probe": {
                    "status": "PASS",
                    "output_dir": out_d_str,
                    "output_dir_binding": {
                        "checked": True,
                        "satisfied": True,
                        "train_call_resolved": [
                            {
                                "output_dir": out_d_str,
                                "resume_from": None,
                                "num_epochs": 300,
                                "random_seed": 42,
                            }
                        ],
                    },
                },
                "history": {
                    "n_epochs": 300,
                    "best_val_loss": ckpt_val_loss,
                    "best_epoch_one_based": ckpt_epoch + 1,
                    "best_epoch_zero_based": ckpt_epoch,
                    "val_loss": val_loss_arr,
                    "train_loss": train_loss_arr,
                },
                "checkpoints": {
                    "best_sha256": ckpt_sha,
                    "final_sha256": final_sha,
                    "final_epoch_zero_based": 299,
                },
            },
        },
    }


def test_target_binding_validation() -> None:
    """Target hashing and binding must fail closed on any discrepancy."""
    seqs = [
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([4.0, 5.0], dtype=np.float64),
        np.array([6.0, 7.0, 8.0, 9.0], dtype=np.float64),
    ]
    meta = hash_target_sequences(seqs)
    assert meta["n_trials"] == 3
    assert meta["n_full_frames"] == 9
    assert meta["n_eligible_frames"] == 6

    # Exact binding succeeds
    validate_target_binding(seqs, meta)

    # Empty expected dict fails closed
    with pytest.raises(ValueError, match="must not be empty"):
        validate_target_binding(seqs, {})

    # Incomplete expected dict fails closed
    with pytest.raises(ValueError, match="Missing mandatory binding key"):
        validate_target_binding(seqs, {"n_trials": 3})

    # Fail closed on modified sequence
    bad_seqs = [
        np.array([1.0, 2.0, 3.01], dtype=np.float64),
        seqs[1],
        seqs[2],
    ]
    with pytest.raises(ValueError, match="Target binding mismatch"):
        validate_target_binding(bad_seqs, meta)

    # Fail closed on missing trial
    with pytest.raises(ValueError, match="Target binding mismatch"):
        validate_target_binding(seqs[:2], meta)

    # Fail closed on non-finite entries
    nan_seqs = [np.array([1.0, np.nan, 3.0]), seqs[1], seqs[2]]
    with pytest.raises(ValueError, match="contains non-finite"):
        hash_target_sequences(nan_seqs)


def test_trial_boundary_and_persistence_isolation() -> None:
    """Persistence and sustained run masks must be strictly trial-local."""
    t0 = np.array([0.0, 0.0, 20.0], dtype=np.float64)
    t1 = np.array([20.0, 0.0, 0.0], dtype=np.float64)
    p0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    p1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    res = compute_aligned_grid_metrics(
        y_true_seqs=[t0, t1],
        y_pred_seqs=[p0, p1],
        trial_ids=["trial_0", "trial_1"],
        prefix_ids=["pref_A", "pref_B"],
        thresholds=[10.0],
        min_runs=[2],
    )

    cell = res["sustained_cells"][0]
    assert cell["scope_id"] == "band10_run2"
    assert cell["counts"]["aligned"]["n_trials_with_escape"] == 0
    assert cell["aligned"]["escape"]["model"]["n_frames"] == 0
    assert cell["aligned"]["escape"]["model"]["mse"] is None

    ov_pers = res["aligned_overall"]["persistence"]
    assert ov_pers["n_frames"] == 4
    assert ov_pers["sse"] == pytest.approx(800.0)
    assert ov_pers["mse"] == pytest.approx(200.0)


def test_empty_mask_and_unavailable_scopes() -> None:
    """Empty masks must report null metrics with reasons, never zero error."""
    t0 = np.array([1.0, 2.0, 1.5, 1.0], dtype=np.float64)
    p0 = np.array([1.1, 1.9, 1.4, 1.0], dtype=np.float64)

    res = compute_aligned_grid_metrics(
        y_true_seqs=[t0],
        y_pred_seqs=[p0],
        trial_ids=["trial_0"],
        prefix_ids=["pref_A"],
        thresholds=[50.0],
        min_runs=[1],
    )
    cell = res["sustained_cells"][0]
    esc_model = cell["aligned"]["escape"]["model"]
    assert esc_model["n_frames"] == 0
    assert esc_model["sse"] is None
    assert esc_model["mse"] is None
    assert esc_model["rmse"] is None
    assert esc_model["mae"] is None
    assert esc_model["r2"] is None
    assert esc_model["reason"] is not None

    skill_missing = compute_skill_ratio(esc_model["mse"], 1.0)
    assert skill_missing["value"] is None
    assert skill_missing["reason"] == "missing_model_mse"

    skill_zero_base = compute_skill_ratio(1.0, 0.0)
    assert skill_zero_base["value"] is None
    assert skill_zero_base["reason"] == "zero_comparator_mse"


def test_per_trial_scope_partition_persistence() -> None:
    """M5: All 12 cells and aligned overall must persist and reconcile ordered per-trial frame counts."""
    t0 = np.array([0.0, 10.0, 25.0, 25.0, 0.0], dtype=np.float64)
    t1 = np.array([0.0, 60.0, 60.0, 60.0, 0.0], dtype=np.float64)
    p0 = np.array([0.1, 9.9, 24.8, 25.1, 0.2], dtype=np.float64)
    p1 = np.array([0.0, 59.0, 60.1, 59.8, 0.1], dtype=np.float64)

    res = compute_aligned_grid_metrics(
        y_true_seqs=[t0, t1],
        y_pred_seqs=[p0, p1],
        trial_ids=["t0", "t1"],
        prefix_ids=["pA", "pB"],
    )
    assert len(res["sustained_cells"]) == 12
    for cell in res["sustained_cells"]:
        part = cell["per_trial_partitions"]
        assert part["trial_ids"] == ["t0", "t1"]
        assert part["prefix_ids"] == ["pA", "pB"]
        assert len(part["full_frame"]["escape_frames"]) == 2
        assert len(part["full_frame"]["rest_frames"]) == 2
        assert len(part["aligned"]["escape_frames"]) == 2
        assert len(part["aligned"]["rest_frames"]) == 2

        # Frame count conservation
        assert sum(part["full_frame"]["escape_frames"]) == cell["counts"]["full_frame"]["n_escape_frames"]
        assert sum(part["full_frame"]["rest_frames"]) == cell["counts"]["full_frame"]["n_rest_frames"]
        assert sum(part["aligned"]["escape_frames"]) == cell["counts"]["aligned"]["n_escape_frames"]
        assert sum(part["aligned"]["rest_frames"]) == cell["counts"]["aligned"]["n_rest_frames"]


def test_regression_metrics_overflow_and_tiny_variance() -> None:
    """compute_regression_metrics handles overflow, zero and tiny variance."""
    # Constant target gives r2=None with zero_target_variance
    y_const = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
    y_pred = np.array([5.1, 4.9, 5.0, 5.2], dtype=np.float64)
    res = compute_regression_metrics(y_const, y_pred)
    assert res["r2"] is None
    assert res["r2_reason"] == "zero_target_variance"
    assert res["mse"] is not None

    # Tiny target variance handles non-finite r2 safely without -inf
    res_tiny = compute_regression_metrics(
        np.array([0.0, 1e-160], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
    )
    assert res_tiny["r2"] is None
    assert res_tiny["r2_reason"] == "numerical_instability"

    # Overflow raises FloatingPointError
    y_huge = np.array([1e200, 1e200], dtype=np.float64)
    y_neg = np.array([-1e200, -1e200], dtype=np.float64)
    with pytest.raises(FloatingPointError):
        compute_regression_metrics(y_huge, y_neg)


def test_declared_family_complete_78_slots() -> None:
    """Evaluator must assemble all 78 frozen slots and run Holm-Bonferroni."""
    rng = np.random.default_rng(42)
    n_trials = 8
    trial_ids = [f"trial_{i}" for i in range(n_trials)]
    prefix_ids = [f"prefix_{i % 6}" for i in range(n_trials)]

    cand_scopes: dict[str, dict[str, Any]] = {"k_0.5": {}, "k_1.0": {}}
    comp_scopes: dict[str, dict[str, Any]] = {
        "head_only_baseline": {},
        "persistence": {},
        "zero": {},
    }

    slots = declared_family_slots()
    assert len(slots) == DECLARED_FAMILY_SIZE == 78

    from nsmor.analysis.model_comparison import SCOPE_IDS

    for sc in SCOPE_IDS:
        for cid in CANDIDATE_IDS:
            cand_scopes[cid][sc] = {
                "trial_ids": trial_ids,
                "prefix_ids": prefix_ids,
                "frame_counts": [100] * n_trials,
                "model_mse": list(rng.uniform(1.0, 2.0, n_trials)),
            }
        comp_scopes["head_only_baseline"][sc] = {
            "trial_ids": trial_ids,
            "prefix_ids": prefix_ids,
            "frame_counts": [100] * n_trials,
            "model_mse": list(rng.uniform(2.0, 3.0, n_trials)),
        }
        comp_scopes["persistence"][sc] = {
            "trial_ids": trial_ids,
            "prefix_ids": prefix_ids,
            "frame_counts": [100] * n_trials,
            "persistence_mse": list(rng.uniform(1.5, 2.5, n_trials)),
        }
        comp_scopes["zero"][sc] = {
            "trial_ids": trial_ids,
            "prefix_ids": prefix_ids,
            "frame_counts": [100] * n_trials,
            "zero_mse": list(rng.uniform(5.0, 6.0, n_trials)),
        }

    # Non-bool exchangeability_asserted fails closed
    with pytest.raises(ValueError, match="genuine bool"):
        assemble_declared_family(
            cand_scopes, comp_scopes, exchangeability_asserted="true"  # type: ignore
        )

    # Run family assembly with exchangeability asserted
    family = assemble_declared_family(
        candidate_scopes=cand_scopes,
        comparator_scopes=comp_scopes,
        exchangeability_asserted=True,
    )

    assert family["family_size"] == 78
    assert len(family["slots"]) == 78
    assert set(family["slots"].keys()) == set(slots)
    assert family["both_arms_present"] is True

    # Missing candidate arm preserves declared slots with unavailable reason
    partial_cand = {"k_0.5": cand_scopes["k_0.5"]}
    partial_family = assemble_declared_family(
        candidate_scopes=partial_cand,
        comparator_scopes=comp_scopes,
        exchangeability_asserted=True,
    )
    assert partial_family["family_size"] == 78
    assert partial_family["both_arms_present"] is False
    assert "k_1.0" not in partial_family["present_candidates"]
    k10_slot = f"k_1.0|head_only_baseline|{SCOPE_IDS[0]}"
    assert not partial_family["slots"][k10_slot]["available"]
    assert partial_family["slots"][k10_slot]["reason"] == "missing_candidate_arm"


def test_prediction_cache_npz_roundtrip_and_overflow_protection(
    tmp_path: Path,
) -> None:
    """Prediction cache round-trip and non-overflowing length checks."""
    cache_base = tmp_path / "test_k_0.5"
    orig_seqs = [
        np.array([1.23, 4.56, 7.89], dtype=np.float64),
        np.array([10.11, 12.13], dtype=np.float64),
        np.array([14.15, 16.17, 18.19, 20.21], dtype=np.float64),
    ]
    meta = {
        "candidate_id": "k_0.5",
        "checkpoint": "best_model.pth",
        "epochs": 300,
    }

    save_prediction_cache(cache_base, orig_seqs, meta)
    loaded_seqs, loaded_meta = load_prediction_cache(cache_base)

    assert len(loaded_seqs) == len(orig_seqs)
    for orig, loaded in zip(orig_seqs, loaded_seqs):
        np.testing.assert_array_equal(orig, loaded)
    assert loaded_meta == meta

    # Refuse overwrite unless explicitly enabled
    with pytest.raises(FileExistsError):
        save_prediction_cache(cache_base, orig_seqs, meta, allow_overwrite=False)

    # Overwrite works when enabled
    save_prediction_cache(cache_base, orig_seqs, meta, allow_overwrite=True)

    # uint64 arithmetic overflow attack must fail closed
    overflow_cache = tmp_path / "overflow_cache"
    np.savez_compressed(
        f"{overflow_cache}.npz",
        flat_preds=np.array([], dtype=np.float64),
        lengths=np.array([2**63, 2**63], dtype=np.uint64),
    )
    Path(f"{overflow_cache}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        ValueError, match="exceeds flat_preds size|Sum of lengths"
    ):
        load_prediction_cache(overflow_cache)

    # M4/m1: float, bool, or non-integer lengths in npz must fail closed before int coercion
    float_len_cache = tmp_path / "float_len_cache"
    np.savez_compressed(
        f"{float_len_cache}.npz",
        flat_preds=np.array([1.0, 2.0], dtype=np.float64),
        lengths=np.array([1.5], dtype=np.float64),
    )
    Path(f"{float_len_cache}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="non-boolean integer dtype"):
        load_prediction_cache(float_len_cache)

    bool_len_cache = tmp_path / "bool_len_cache"
    np.savez_compressed(
        f"{bool_len_cache}.npz",
        flat_preds=np.array([1.0], dtype=np.float64),
        lengths=np.array([True], dtype=bool),
    )
    Path(f"{bool_len_cache}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="non-boolean integer dtype"):
        load_prediction_cache(bool_len_cache)


def test_synthetic_checkpoint_config_validation(tmp_path: Path) -> None:
    """Validate checkpoint controls, baseline pins, and fail-closed gates."""
    ckpt_path = tmp_path / "synthetic_ckpt.pth"

    # 1. Invalid candidate ID
    torch.save({}, ckpt_path)
    with pytest.raises(ValueError, match="Unknown candidate_id"):
        validate_checkpoint_config(ckpt_path, "k_invalid")

    # 2. String persistence_skip rejected
    bad_cfg = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_cfg["config"]["model"]["persistence_skip"] = "0.5"
    torch.save(bad_cfg, ckpt_path)
    with pytest.raises(
        ValueError, match="persistence_skip must be a genuine numeric scalar"
    ):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # 3. Mismatched training control (learning rate / seed) rejected
    bad_train = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_train["config"]["training"]["random_seed"] = 999
    torch.save(bad_train, ckpt_path)
    with pytest.raises(ValueError, match="training.random_seed mismatch"):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # 3b. Mismatched training workers rejected
    bad_workers = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_workers["config"]["training"]["num_workers"] = 4
    torch.save(bad_workers, ckpt_path)
    with pytest.raises(ValueError, match="training.num_workers mismatch"):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # 3c. Mismatched loss reduction rejected
    bad_loss_red = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_loss_red["config"]["loss"]["reduction"] = "sum"
    torch.save(bad_loss_red, ckpt_path)
    with pytest.raises(ValueError, match="loss.reduction mismatch"):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # 3d. Mismatched finetune unfreeze_after_epoch rejected
    bad_ft = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_ft["config"]["finetune"]["unfreeze_after_epoch"] = 99
    torch.save(bad_ft, ckpt_path)
    with pytest.raises(ValueError, match="finetune.unfreeze_after_epoch mismatch"):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # 3e. Mismatched cluster gating n_clusters rejected
    bad_cg = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_cg["config"]["cluster_gating"]["n_clusters"] = 8
    torch.save(bad_cg, ckpt_path)
    with pytest.raises(ValueError, match="cluster_gating.n_clusters mismatch"):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # 3f. Unexpected config key or section rejected
    bad_extra = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_extra["config"]["unexpected_section"] = {}
    torch.save(bad_extra, ckpt_path)
    with pytest.raises(ValueError, match="unexpected section"):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # 3g. M2: Strict float control drift (rel_tol=1e-7, abs_tol=0.0) rejected
    bad_lr = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_lr["config"]["training"]["learning_rate"] = 0.00055  # drifted from 0.0005
    torch.save(bad_lr, ckpt_path)
    with pytest.raises(ValueError, match="training.learning_rate mismatch"):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    bad_aux = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_aux["config"]["loss"]["lambda_routing_aux"] = 0.00009  # drifted from 0.0
    torch.save(bad_aux, ckpt_path)
    with pytest.raises(ValueError, match="loss.lambda_routing_aux mismatch"):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    bad_lr_bool = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_lr_bool["config"]["training"]["learning_rate"] = True
    torch.save(bad_lr_bool, ckpt_path)
    with pytest.raises(ValueError, match="training.learning_rate must be float"):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # 3h. M2: Missing mandatory checkpoint section rejected
    bad_no_ckpt_sec = _make_valid_checkpoint_dict("k_0.5", 0.5)
    del bad_no_ckpt_sec["config"]["checkpoint"]
    torch.save(bad_no_ckpt_sec, ckpt_path)
    with pytest.raises(ValueError, match="missing required section: 'checkpoint'"):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # 4. Negative epoch rejected
    bad_epoch = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_epoch["epoch"] = -5
    torch.save(bad_epoch, ckpt_path)
    with pytest.raises(ValueError, match="epoch must be non-negative"):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # 5. Non-finite or negative val_loss rejected, finite 0.0 accepted
    bad_loss = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_loss["val_loss"] = float("nan")
    torch.save(bad_loss, ckpt_path)
    with pytest.raises(
        ValueError, match="val_loss must be finite non-negative float"
    ):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    neg_loss = _make_valid_checkpoint_dict("k_0.5", 0.5)
    neg_loss["val_loss"] = -0.01
    neg_loss["best_val_loss"] = -0.01
    torch.save(neg_loss, ckpt_path)
    with pytest.raises(
        ValueError, match="val_loss must be finite non-negative float"
    ):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # Zero loss checkpoint for baseline (no receipt required for baseline)
    zero_loss = _make_valid_checkpoint_dict("head_only_baseline", 0.0)
    zero_loss["val_loss"] = 0.0
    zero_loss["best_val_loss"] = 0.0
    torch.save(zero_loss, ckpt_path)
    with patch(
        "scripts.evaluate_model_optimization._compute_sha256",
        return_value=BASELINE_CHECKPOINT_SHA256,
    ):
        loaded_zero = validate_checkpoint_config(ckpt_path, "head_only_baseline")
        assert loaded_zero["val_loss"] == 0.0

    # 6. Unpinned baseline SHA rejected
    base_cfg = _make_valid_checkpoint_dict("head_only_baseline", 0.0)
    torch.save(base_cfg, ckpt_path)
    with pytest.raises(ValueError, match="Baseline checkpoint SHA-256 mismatch"):
        validate_checkpoint_config(ckpt_path, "head_only_baseline")

    # 7. Candidate reusing baseline output_dir rejected
    bad_out = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_out["config"]["checkpoint"] = {
        "output_dir": ".scratch/model-opt-baseline-300-20261003-rerun"
    }
    torch.save(bad_out, ckpt_path)
    with pytest.raises(ValueError, match="isolated output_dir"):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # 8. Candidate with resume_from rejected (fresh training required)
    bad_resume = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_resume["config"]["checkpoint"] = {
        "output_dir": ".scratch/cand_05",
        "resume_from": "some_checkpoint.pth",
    }
    torch.save(bad_resume, ckpt_path)
    with pytest.raises(ValueError, match="resume_from must be null"):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # 9. Candidate k_0.5 requires sealed run/assessor receipt evidence
    good_cfg = _make_valid_checkpoint_dict("k_0.5", 0.5)
    torch.save(good_cfg, ckpt_path)
    with pytest.raises(
        ValueError, match="requires sealed run/assessor receipt evidence"
    ):
        validate_checkpoint_config(ckpt_path, "k_0.5")

    # 10. Fail-closed receipt validation regressions
    actual_ckpt_sha = _compute_sha(ckpt_path)
    seal_dir = tmp_path / "seals"
    ckpt_val_loss = 7.196
    valid_report = _make_valid_assessor_report_dict(
        ckpt_sha=actual_ckpt_sha,
        ckpt_epoch=295,
        ckpt_val_loss=ckpt_val_loss,
        seal_dir=seal_dir,
    )

    # 10a. Empty receipt {} fails closed
    empty_receipt = tmp_path / "empty_receipt.json"
    empty_receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Run receipt must be a non-empty JSON object"):
        validate_checkpoint_config(
            ckpt_path, "k_0.5", run_receipt_path=empty_receipt
        )

    # 10b. Empty evaluator_binding {} fails closed
    empty_eb = tmp_path / "empty_eb.json"
    empty_eb.write_text('{"evaluator_binding": {}}', encoding="utf-8")
    with pytest.raises(
        ValueError, match="missing non-empty evaluator_binding block"
    ):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=empty_eb)

    # 10c. Missing or wrong role fails closed
    r_bad_role = copy.deepcopy(valid_report)
    del r_bad_role["evaluator_binding"]["role"]
    p_bad_role = tmp_path / "r_bad_role.json"
    p_bad_role.write_text(json.dumps(r_bad_role), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required 'role' field"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_bad_role)

    # 10d. String or boolean budget fails closed
    r_str_budget = copy.deepcopy(valid_report)
    r_str_budget["evaluator_binding"]["declared_epochs"] = "300"
    p_str_budget = tmp_path / "r_str_budget.json"
    p_str_budget.write_text(json.dumps(r_str_budget), encoding="utf-8")
    with pytest.raises(ValueError, match="declared_epochs must be integer 300"):
        validate_checkpoint_config(
            ckpt_path, "k_0.5", run_receipt_path=p_str_budget
        )

    r_bool_budget = copy.deepcopy(valid_report)
    r_bool_budget["evaluator_binding"]["actual_completed_epochs"] = True
    p_bool_budget = tmp_path / "r_bool_budget.json"
    p_bool_budget.write_text(json.dumps(r_bool_budget), encoding="utf-8")
    with pytest.raises(
        ValueError, match="actual_completed_epochs must be integer 300"
    ):
        validate_checkpoint_config(
            ckpt_path, "k_0.5", run_receipt_path=p_bool_budget
        )

    # 10e. Incomplete budget (< 300) fails closed
    r_short = copy.deepcopy(valid_report)
    r_short["evaluator_binding"]["actual_completed_epochs"] = 299
    p_short = tmp_path / "r_short.json"
    p_short.write_text(json.dumps(r_short), encoding="utf-8")
    with pytest.raises(
        ValueError, match="actual_completed_epochs must be integer 300"
    ):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_short)

    # 10f. natural_exit_zero not True fails closed
    r_exit_bad = copy.deepcopy(valid_report)
    r_exit_bad["evaluator_binding"]["natural_exit_zero"] = False
    p_exit_bad = tmp_path / "r_exit_bad.json"
    p_exit_bad.write_text(json.dumps(r_exit_bad), encoding="utf-8")
    with pytest.raises(ValueError, match="natural_exit_zero must be True"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_exit_bad)

    # 10g. Resumed or early stopped fails closed
    r_resumed = copy.deepcopy(valid_report)
    r_resumed["evaluator_binding"]["not_resumed"] = False
    p_resumed = tmp_path / "r_resumed.json"
    p_resumed.write_text(json.dumps(r_resumed), encoding="utf-8")
    with pytest.raises(ValueError, match="not_resumed must be True"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_resumed)

    r_early = copy.deepcopy(valid_report)
    r_early["evaluator_binding"]["early_stopped"] = True
    p_early = tmp_path / "r_early.json"
    p_early.write_text(json.dumps(r_early), encoding="utf-8")
    with pytest.raises(ValueError, match="early_stopped must be False"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_early)

    # 10h. Best checkpoint sha mismatch fails closed
    r_best_sha = copy.deepcopy(valid_report)
    r_best_sha["evaluator_binding"]["best_checkpoint"]["sha256"] = "0" * 64
    p_best_sha = tmp_path / "r_best_sha.json"
    p_best_sha.write_text(json.dumps(r_best_sha), encoding="utf-8")
    with pytest.raises(ValueError, match="best_checkpoint.sha256 mismatch"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_best_sha)

    # 10i. Final checkpoint epoch != 299 fails closed
    r_fin_ep = copy.deepcopy(valid_report)
    r_fin_ep["evaluator_binding"]["final_checkpoint"]["epoch_zero_based"] = 298
    p_fin_ep = tmp_path / "r_fin_ep.json"
    p_fin_ep.write_text(json.dumps(r_fin_ep), encoding="utf-8")
    with pytest.raises(
        ValueError, match="final_checkpoint.epoch_zero_based must be 299"
    ):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_fin_ep)

    # 10j. Missing source seal path or hash mismatch fails closed
    r_bad_seal = copy.deepcopy(valid_report)
    r_bad_seal["evaluator_binding"]["source_seal"]["sealed_receipt_sha256"] = (
        "0" * 64
    )
    p_bad_seal = tmp_path / "r_bad_seal.json"
    p_bad_seal.write_text(json.dumps(r_bad_seal), encoding="utf-8")
    with pytest.raises(
        ValueError, match="source_seal sealed_receipt SHA mismatch"
    ):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_bad_seal)

    # 10k. M1: Sealed receipt contradictory facts fail closed
    p_rec_fail = seal_dir / "sealed_receipt_fail.json"
    rec_fail_bytes = json.dumps(
        {
            "status": "FAIL",
            "exit_code": 17,
            "actual_completed_epochs": 1,
            "resumed": False,
        }
    ).encode("utf-8")
    rec_fail_sha = _make_dummy_file_with_sha(p_rec_fail, rec_fail_bytes)
    r_fact_fail = copy.deepcopy(valid_report)
    r_fact_fail["evaluator_binding"]["source_seal"]["sealed_receipt"] = str(p_rec_fail)
    r_fact_fail["evaluator_binding"]["source_seal"]["sealed_receipt_sha256"] = rec_fail_sha
    p_fact_fail = tmp_path / "r_fact_fail.json"
    p_fact_fail.write_text(json.dumps(r_fact_fail), encoding="utf-8")
    with pytest.raises(ValueError, match="records status 'FAIL', not 'PASS'"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_fact_fail)

    # 10l. M1: Training log failure indicators fail closed
    p_log_bad = seal_dir / "stdout_bad.log"
    log_bad_bytes = b"Epoch 1/300\nRuntimeError: CUDA out of memory\nEpoch 300/300\nProcess exit code 1"
    log_bad_sha = _make_dummy_file_with_sha(p_log_bad, log_bad_bytes)
    p_rec_bad_log = seal_dir / "rec_bad_log.json"
    rec_bad_log_bytes = json.dumps(
        {
            "status": "PASS",
            "exit_code": 0,
            "actual_completed_epochs": 300,
            "resumed": False,
            "log": str(p_log_bad),
            "log_sha256": log_bad_sha,
            "manifest": str(seal_dir / "manifest.json"),
            "manifest_sha256": _compute_sha(seal_dir / "manifest.json"),
        }
    ).encode("utf-8")
    rec_bad_log_sha = _make_dummy_file_with_sha(p_rec_bad_log, rec_bad_log_bytes)
    r_log_bad = copy.deepcopy(valid_report)
    r_log_bad["evaluator_binding"]["source_seal"]["sealed_receipt"] = str(p_rec_bad_log)
    r_log_bad["evaluator_binding"]["source_seal"]["sealed_receipt_sha256"] = rec_bad_log_sha
    r_log_bad["evaluator_binding"]["source_seal"]["log"] = str(p_log_bad)
    r_log_bad["evaluator_binding"]["source_seal"]["log_sha256"] = log_bad_sha
    p_log_bad_json = tmp_path / "r_log_bad.json"
    p_log_bad_json.write_text(json.dumps(r_log_bad), encoding="utf-8")
    with pytest.raises(ValueError, match="failure or early termination indicator"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_log_bad_json)

    # 10m. M1: Non-hex final checkpoint SHA fails closed
    r_non_hex = copy.deepcopy(valid_report)
    r_non_hex["evaluator_binding"]["final_checkpoint"]["sha256"] = "z" * 64
    p_non_hex = tmp_path / "r_non_hex.json"
    p_non_hex.write_text(json.dumps(r_non_hex), encoding="utf-8")
    with pytest.raises(ValueError, match="64-char hexadecimal digest"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_non_hex)

    # 10n. M1: Selection bound violation (best epoch > final epoch) fails closed
    r_epoch_viol = copy.deepcopy(valid_report)
    r_epoch_viol["evaluator_binding"]["best_checkpoint"]["epoch_zero_based"] = 400
    r_epoch_viol["evaluator_binding"]["best_checkpoint"]["epoch_one_based"] = 401
    bad_ckpt = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_ckpt["epoch"] = 400
    p_bad_ckpt = tmp_path / "bad_ckpt_400.pth"
    torch.save(bad_ckpt, p_bad_ckpt)
    r_epoch_viol["evaluator_binding"]["best_checkpoint"]["sha256"] = _compute_sha(p_bad_ckpt)
    p_epoch_viol = tmp_path / "r_epoch_viol.json"
    p_epoch_viol.write_text(json.dumps(r_epoch_viol), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be greater than final_checkpoint"):
        validate_checkpoint_config(p_bad_ckpt, "k_0.5", run_receipt_path=p_epoch_viol)

    # 10o. Missing final checkpoint file fails closed
    r_no_final = copy.deepcopy(valid_report)
    r_no_final["evaluator_binding"]["final_checkpoint"]["path"] = str(seal_dir / "nonexistent_final.pth")
    p_no_final = tmp_path / "r_no_final.json"
    p_no_final.write_text(json.dumps(r_no_final), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="final_checkpoint file not found"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_no_final)

    # 10p. Corrupted final checkpoint sha mismatch fails closed
    r_bad_fsha = copy.deepcopy(valid_report)
    r_bad_fsha["evaluator_binding"]["final_checkpoint"]["sha256"] = "0" * 64
    p_bad_fsha = tmp_path / "r_bad_fsha.json"
    p_bad_fsha.write_text(json.dumps(r_bad_fsha), encoding="utf-8")
    with pytest.raises(ValueError, match="final_checkpoint SHA-256 mismatch"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_bad_fsha)

    # 10q. Vacuous manifest fails closed
    p_vacuous_man = seal_dir / "vacuous_manifest.json"
    vac_bytes = json.dumps({"foo": "bar"}).encode("utf-8")
    vac_sha = _make_dummy_file_with_sha(p_vacuous_man, vac_bytes)
    # update rec.json to match p_vacuous_man so it gets past rec_obj cross check
    rec_vac_bytes = json.dumps(
        {
            "status": "PASS",
            "exit_code": 0,
            "actual_completed_epochs": 300,
            "resumed": False,
            "log": str(seal_dir / "stdout.log"),
            "log_sha256": _compute_sha(seal_dir / "stdout.log"),
            "manifest": str(p_vacuous_man),
            "manifest_sha256": vac_sha,
        }
    ).encode("utf-8")
    p_rec_vac = seal_dir / "rec_vac.json"
    rec_vac_sha = _make_dummy_file_with_sha(p_rec_vac, rec_vac_bytes)
    r_vac_man = copy.deepcopy(valid_report)
    r_vac_man["evaluator_binding"]["source_seal"]["sealed_receipt"] = str(p_rec_vac)
    r_vac_man["evaluator_binding"]["source_seal"]["sealed_receipt_sha256"] = rec_vac_sha
    r_vac_man["evaluator_binding"]["source_seal"]["manifest"] = str(p_vacuous_man)
    r_vac_man["evaluator_binding"]["source_seal"]["manifest_sha256"] = vac_sha
    p_vac_json = tmp_path / "r_vac_man.json"
    p_vac_json.write_text(json.dumps(r_vac_man), encoding="utf-8")
    with pytest.raises(
        ValueError, match="source_seal manifest missing mandatory metadata field: 'label'"
    ):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_vac_json)

    # 10r. Manifest with empty paths dictionary fails closed
    p_empty_paths_man = seal_dir / "empty_paths_manifest.json"
    empty_p_bytes = json.dumps(
        {
            "label": "candidate_seal",
            "kind": "manifest",
            "created_utc": "2026-10-05T00:00:00Z",
            "cwd": "/mnt/d/Projects/NSMoR",
            "paths": {},
        }
    ).encode("utf-8")
    empty_p_sha = _make_dummy_file_with_sha(p_empty_paths_man, empty_p_bytes)
    rec_empty_bytes = json.dumps(
        {
            "status": "PASS",
            "exit_code": 0,
            "actual_completed_epochs": 300,
            "resumed": False,
            "log": str(seal_dir / "stdout.log"),
            "log_sha256": _compute_sha(seal_dir / "stdout.log"),
            "manifest": str(p_empty_paths_man),
            "manifest_sha256": empty_p_sha,
        }
    ).encode("utf-8")
    p_rec_empty = seal_dir / "rec_empty.json"
    rec_empty_sha = _make_dummy_file_with_sha(p_rec_empty, rec_empty_bytes)
    r_empty_p = copy.deepcopy(valid_report)
    r_empty_p["evaluator_binding"]["source_seal"]["sealed_receipt"] = str(p_rec_empty)
    r_empty_p["evaluator_binding"]["source_seal"]["sealed_receipt_sha256"] = rec_empty_sha
    r_empty_p["evaluator_binding"]["source_seal"]["manifest"] = str(p_empty_paths_man)
    r_empty_p["evaluator_binding"]["source_seal"]["manifest_sha256"] = empty_p_sha
    p_empty_p_json = tmp_path / "r_empty_p.json"
    p_empty_p_json.write_text(json.dumps(r_empty_p), encoding="utf-8")
    with pytest.raises(
        ValueError, match="source_seal manifest missing mandatory non-empty 'paths'"
    ):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_empty_p_json)

    # 10s. Final checkpoint file with epoch != 299 fails closed
    p_bad_ep_final = seal_dir / "bad_ep_final.pth"
    bad_ep_final_dict = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_ep_final_dict["epoch"] = 280
    torch.save(bad_ep_final_dict, p_bad_ep_final)
    bad_ep_final_sha = _compute_sha(p_bad_ep_final)
    r_bad_fep = copy.deepcopy(valid_report)
    r_bad_fep["evaluator_binding"]["final_checkpoint"]["path"] = str(p_bad_ep_final)
    r_bad_fep["evaluator_binding"]["final_checkpoint"]["sha256"] = bad_ep_final_sha
    p_bad_fep_json = tmp_path / "r_bad_fep.json"
    p_bad_fep_json.write_text(json.dumps(r_bad_fep), encoding="utf-8")
    with pytest.raises(ValueError, match="final_checkpoint embedded epoch must be 299"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_bad_fep_json)

    # 10t. Final checkpoint file with non-finite tensor fails closed
    p_nan_final = seal_dir / "nan_final.pth"
    nan_final_dict = _make_valid_checkpoint_dict("k_0.5", 0.5)
    nan_final_dict["epoch"] = 299
    nan_final_dict["val_loss"] = 7.200
    nan_final_dict["model_state_dict"] = {
        "weight": torch.tensor([float("nan"), 1.0])
    }
    torch.save(nan_final_dict, p_nan_final)
    nan_final_sha = _compute_sha(p_nan_final)
    r_nan_f = copy.deepcopy(valid_report)
    r_nan_f["evaluator_binding"]["final_checkpoint"]["path"] = str(p_nan_final)
    r_nan_f["evaluator_binding"]["final_checkpoint"]["sha256"] = nan_final_sha
    p_nan_f_json = tmp_path / "r_nan_f.json"
    p_nan_f_json.write_text(json.dumps(r_nan_f), encoding="utf-8")
    with pytest.raises(ValueError, match="contains non-finite values"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_nan_f_json)

    # 10u. Contradictory nested probe train call fails closed
    r_bad_probe = copy.deepcopy(valid_report)
    r_bad_probe["evaluator_binding"]["source_provenance"]["runtime_probe"]["output_dir_binding"]["train_call_resolved"] = [{
        "output_dir": str(seal_dir),
        "resume_from": "/path/to/resumed.pth",
        "num_epochs": 300,
        "random_seed": 42,
    }]
    p_bad_probe_json = tmp_path / "r_bad_probe.json"
    p_bad_probe_json.write_text(json.dumps(r_bad_probe), encoding="utf-8")
    with pytest.raises(ValueError, match="train call was resumed"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_bad_probe_json)

    # 10v. R9 Root 1: Manifest kind validation (rejects invalid kinds)
    p_bad_kind_man = seal_dir / "bad_kind_manifest.json"
    bad_kind_bytes = json.dumps(
        {
            "label": "candidate_seal",
            "kind": "custom_kind",
            "created_utc": "2026-10-05T00:00:00Z",
            "cwd": "/mnt/d/Projects/NSMoR",
            "paths": {"scripts/train.py": "a" * 64},
        }
    ).encode("utf-8")
    bad_kind_sha = _make_dummy_file_with_sha(p_bad_kind_man, bad_kind_bytes)
    rec_bad_kind_bytes = json.dumps(
        {
            "status": "PASS",
            "exit_code": 0,
            "actual_completed_epochs": 300,
            "resumed": False,
            "log": str(seal_dir / "stdout.log"),
            "log_sha256": _compute_sha(seal_dir / "stdout.log"),
            "manifest": str(p_bad_kind_man),
            "manifest_sha256": bad_kind_sha,
        }
    ).encode("utf-8")
    p_rec_bad_kind = seal_dir / "rec_bad_kind.json"
    rec_bad_kind_sha = _make_dummy_file_with_sha(p_rec_bad_kind, rec_bad_kind_bytes)
    r_bad_kind = copy.deepcopy(valid_report)
    r_bad_kind["evaluator_binding"]["source_seal"]["sealed_receipt"] = str(p_rec_bad_kind)
    r_bad_kind["evaluator_binding"]["source_seal"]["sealed_receipt_sha256"] = rec_bad_kind_sha
    r_bad_kind["evaluator_binding"]["source_seal"]["manifest"] = str(p_bad_kind_man)
    r_bad_kind["evaluator_binding"]["source_seal"]["manifest_sha256"] = bad_kind_sha
    p_bad_kind_json = tmp_path / "r_bad_kind.json"
    p_bad_kind_json.write_text(json.dumps(r_bad_kind), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest 'kind' must be 'source', 'merged', or 'manifest'"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_bad_kind_json)

    # 10w. R9 Root 2: Discontiguous telemetry or missing completion in stdout log
    p_log_discontig = seal_dir / "stdout_discontig.log"
    # Skip epoch 150
    discontig_lines = [
        f"Epoch {i}/300  train_loss=8.000000  val_loss=7.500000  time=0.1s"
        for i in range(1, 301)
        if i != 150
    ]
    discontig_lines.append(f"Training complete.  Best val loss: {ckpt_val_loss:.6f}")
    log_discontig_bytes = "\n".join(discontig_lines).encode("utf-8")
    log_discontig_sha = _make_dummy_file_with_sha(p_log_discontig, log_discontig_bytes)
    p_rec_discontig = seal_dir / "rec_discontig.json"
    rec_discontig_bytes = json.dumps(
        {
            "status": "PASS",
            "exit_code": 0,
            "actual_completed_epochs": 300,
            "resumed": False,
            "log": str(p_log_discontig),
            "log_sha256": log_discontig_sha,
            "manifest": str(seal_dir / "manifest.json"),
            "manifest_sha256": _compute_sha(seal_dir / "manifest.json"),
        }
    ).encode("utf-8")
    rec_discontig_sha = _make_dummy_file_with_sha(p_rec_discontig, rec_discontig_bytes)
    r_discontig = copy.deepcopy(valid_report)
    r_discontig["evaluator_binding"]["source_seal"]["sealed_receipt"] = str(p_rec_discontig)
    r_discontig["evaluator_binding"]["source_seal"]["sealed_receipt_sha256"] = rec_discontig_sha
    r_discontig["evaluator_binding"]["source_seal"]["log"] = str(p_log_discontig)
    r_discontig["evaluator_binding"]["source_seal"]["log_sha256"] = log_discontig_sha
    p_discontig_json = tmp_path / "r_discontig.json"
    p_discontig_json.write_text(json.dumps(r_discontig), encoding="utf-8")
    with pytest.raises(ValueError, match="missing epoch 150"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_discontig_json)

    # 10x. R9 Root 3: Missing or empty model_state_dict fails closed
    empty_sd_ckpt = _make_valid_checkpoint_dict("k_0.5", 0.5)
    empty_sd_ckpt["model_state_dict"] = {}
    p_empty_sd = tmp_path / "empty_sd.pth"
    torch.save(empty_sd_ckpt, p_empty_sd)
    r_empty_sd = copy.deepcopy(valid_report)
    r_empty_sd["evaluator_binding"]["best_checkpoint"]["sha256"] = _compute_sha(p_empty_sd)
    p_empty_sd_json = tmp_path / "r_empty_sd.json"
    p_empty_sd_json.write_text(json.dumps(r_empty_sd), encoding="utf-8")
    with pytest.raises(ValueError, match="missing mandatory non-empty 'model_state_dict'"):
        validate_checkpoint_config(p_empty_sd, "k_0.5", run_receipt_path=p_empty_sd_json)

    # 10y. R9 Root 4: Missing or invalid val_loss array in history fails closed
    r_bad_hist = copy.deepcopy(valid_report)
    r_bad_hist["evaluator_binding"]["source_provenance"]["history"]["val_loss"] = [7.5] * 299  # incomplete
    p_bad_hist_json = tmp_path / "r_bad_hist.json"
    p_bad_hist_json.write_text(json.dumps(r_bad_hist), encoding="utf-8")
    with pytest.raises(ValueError, match=r"val_loss.*must have 300 entries"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_bad_hist_json)

    # 10z. R9 Root 5: Missing or mismatched pipeline_semantics_version fails closed
    bad_psv_ckpt = _make_valid_checkpoint_dict("k_0.5", 0.5)
    bad_psv_ckpt["pipeline_semantics_version"] = "2.1"
    p_bad_psv = tmp_path / "bad_psv.pth"
    torch.save(bad_psv_ckpt, p_bad_psv)
    r_bad_psv = copy.deepcopy(valid_report)
    r_bad_psv["evaluator_binding"]["best_checkpoint"]["sha256"] = _compute_sha(p_bad_psv)
    p_bad_psv_json = tmp_path / "r_bad_psv.json"
    p_bad_psv_json.write_text(json.dumps(r_bad_psv), encoding="utf-8")
    with pytest.raises(ValueError, match="pipeline_semantics_version mismatch"):
        validate_checkpoint_config(p_bad_psv, "k_0.5", run_receipt_path=p_bad_psv_json)

    # 10aa. R9 Root 6: Floating-point seed/epoch rejected
    r_float_seed = copy.deepcopy(valid_report)
    r_float_seed["evaluator_binding"]["protocol"]["random_seed"] = 42.0
    p_float_seed_json = tmp_path / "r_float_seed.json"
    p_float_seed_json.write_text(json.dumps(r_float_seed), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol.random_seed mismatch: expected integer 42"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_float_seed_json)

    # 10ab. R10 Root 2: Contradictory stdout losses fail closed (Reviewer A blocker C2)
    p_log_contra = seal_dir / "stdout_contra.log"
    contra_lines = [
        f"Epoch {i}/300  train_loss=8.000000  val_loss=9.999000  time=0.1s"
        for i in range(1, 301)
    ]
    contra_lines.append(f"Training complete.  Best val loss: {ckpt_val_loss:.6f}")
    contra_bytes = "\n".join(contra_lines).encode("utf-8")
    contra_sha = _make_dummy_file_with_sha(p_log_contra, contra_bytes)
    p_rec_contra = seal_dir / "rec_contra.json"
    rec_contra_bytes = json.dumps(
        {
            "status": "PASS",
            "exit_code": 0,
            "actual_completed_epochs": 300,
            "resumed": False,
            "log": str(p_log_contra),
            "log_sha256": contra_sha,
            "manifest": str(seal_dir / "manifest.json"),
            "manifest_sha256": _compute_sha(seal_dir / "manifest.json"),
            "train_log": str(seal_dir / "train.log"),
            "train_log_sha256": _compute_sha(seal_dir / "train.log"),
        }
    ).encode("utf-8")
    rec_contra_sha = _make_dummy_file_with_sha(p_rec_contra, rec_contra_bytes)
    r_contra = copy.deepcopy(valid_report)
    r_contra["evaluator_binding"]["source_seal"]["sealed_receipt"] = str(p_rec_contra)
    r_contra["evaluator_binding"]["source_seal"]["sealed_receipt_sha256"] = rec_contra_sha
    r_contra["evaluator_binding"]["source_seal"]["log"] = str(p_log_contra)
    r_contra["evaluator_binding"]["source_seal"]["log_sha256"] = contra_sha
    p_contra_json = tmp_path / "r_contra.json"
    p_contra_json.write_text(json.dumps(r_contra), encoding="utf-8")
    with pytest.raises(
        ValueError, match=r"contradicts bound train\.log val_loss"
    ):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_contra_json)

    # 10ac. R10 Root 1: Missing on-disk train.log fails closed (Reviewer B blocker C1/C2)
    p_rec_notlog = seal_dir / "rec_notlog.json"
    rec_notlog_bytes = json.dumps(
        {
            "status": "PASS",
            "exit_code": 0,
            "actual_completed_epochs": 300,
            "resumed": False,
            "log": str(seal_dir / "stdout.log"),
            "log_sha256": _compute_sha(seal_dir / "stdout.log"),
            "manifest": str(seal_dir / "manifest.json"),
            "manifest_sha256": _compute_sha(seal_dir / "manifest.json"),
            "train_log": str(seal_dir / "nonexistent_train.log"),
            "train_log_sha256": "0" * 64,
        }
    ).encode("utf-8")
    rec_notlog_sha = _make_dummy_file_with_sha(p_rec_notlog, rec_notlog_bytes)
    r_notlog = copy.deepcopy(valid_report)
    r_notlog["evaluator_binding"]["source_seal"]["sealed_receipt"] = str(p_rec_notlog)
    r_notlog["evaluator_binding"]["source_seal"]["sealed_receipt_sha256"] = rec_notlog_sha
    r_notlog["evaluator_binding"]["source_seal"]["train_log"] = str(seal_dir / "nonexistent_train.log")
    r_notlog["evaluator_binding"]["source_seal"]["train_log_sha256"] = "0" * 64
    p_notlog_json = tmp_path / "r_notlog.json"
    p_notlog_json.write_text(json.dumps(r_notlog), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError), match="train_log.*not found"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_notlog_json)

    # 10ad. R10 Root 1: train.log hash mismatch fails closed
    p_rec_badtlog_sha = seal_dir / "rec_badtlog_sha.json"
    rec_badtlog_sha_bytes = json.dumps(
        {
            "status": "PASS",
            "exit_code": 0,
            "actual_completed_epochs": 300,
            "resumed": False,
            "log": str(seal_dir / "stdout.log"),
            "log_sha256": _compute_sha(seal_dir / "stdout.log"),
            "manifest": str(seal_dir / "manifest.json"),
            "manifest_sha256": _compute_sha(seal_dir / "manifest.json"),
            "train_log": str(seal_dir / "train.log"),
            "train_log_sha256": "e" * 64,
        }
    ).encode("utf-8")
    rec_badtlog_sha_sha = _make_dummy_file_with_sha(p_rec_badtlog_sha, rec_badtlog_sha_bytes)
    r_badtlog_sha = copy.deepcopy(valid_report)
    r_badtlog_sha["evaluator_binding"]["source_seal"]["sealed_receipt"] = str(p_rec_badtlog_sha)
    r_badtlog_sha["evaluator_binding"]["source_seal"]["sealed_receipt_sha256"] = rec_badtlog_sha_sha
    r_badtlog_sha["evaluator_binding"]["source_seal"]["train_log_sha256"] = "e" * 64
    p_badtlog_sha_json = tmp_path / "r_badtlog_sha.json"
    p_badtlog_sha_json.write_text(json.dumps(r_badtlog_sha), encoding="utf-8")
    with pytest.raises(ValueError, match="train.log SHA mismatch"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_badtlog_sha_json)

    # 10ae. R10 Root 3: Non-integer actual_completed_epochs in sealed_receipt fails closed (Reviewer A major)
    p_rec_float_ace = seal_dir / "rec_float_ace.json"
    rec_float_ace_bytes = json.dumps(
        {
            "status": "PASS",
            "exit_code": 0,
            "actual_completed_epochs": 300.0,
            "resumed": False,
            "log": str(seal_dir / "stdout.log"),
            "log_sha256": _compute_sha(seal_dir / "stdout.log"),
            "manifest": str(seal_dir / "manifest.json"),
            "manifest_sha256": _compute_sha(seal_dir / "manifest.json"),
            "train_log": str(seal_dir / "train.log"),
            "train_log_sha256": _compute_sha(seal_dir / "train.log"),
        }
    ).encode("utf-8")
    rec_float_ace_sha = _make_dummy_file_with_sha(p_rec_float_ace, rec_float_ace_bytes)
    r_float_ace = copy.deepcopy(valid_report)
    r_float_ace["evaluator_binding"]["source_seal"]["sealed_receipt"] = str(p_rec_float_ace)
    r_float_ace["evaluator_binding"]["source_seal"]["sealed_receipt_sha256"] = rec_float_ace_sha
    p_float_ace_json = tmp_path / "r_float_ace.json"
    p_float_ace_json.write_text(json.dumps(r_float_ace), encoding="utf-8")
    with pytest.raises(
        ValueError, match="actual_completed_epochs must be integer 300, got 300.0"
    ):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_float_ace_json)

    # 10af. R10 output isolation: Checkpoint embedded output_dir mismatch fails closed
    r_bad_od = copy.deepcopy(valid_report)
    r_bad_od["evaluator_binding"]["source_provenance"]["runtime_probe"]["output_dir"] = "/different/path"
    r_bad_od["evaluator_binding"]["source_provenance"]["runtime_probe"]["output_dir_binding"]["train_call_resolved"] = [{
        "output_dir": "/different/path",
        "resume_from": None,
        "num_epochs": 300,
        "random_seed": 42,
    }]
    p_bad_od_json = tmp_path / "r_bad_od.json"
    p_bad_od_json.write_text(json.dumps(r_bad_od), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint embedded output_dir .* does not match"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_bad_od_json)

    # 10ag. R11 Root 1: Missing train_log_sha256 fails closed
    r_missing_tsha = copy.deepcopy(valid_report)
    r_missing_tsha["evaluator_binding"]["source_seal"].pop("train_log_sha256", None)
    p_missing_tsha_json = tmp_path / "r_missing_tsha.json"
    p_missing_tsha_json.write_text(json.dumps(r_missing_tsha), encoding="utf-8")
    with pytest.raises(ValueError, match="missing mandatory.*train_log_sha256"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_missing_tsha_json)

    # 10ah. R11 Root 2: Six-decimal loss precision contradiction fails closed
    # (e.g. interior val_loss drifted by 0.000001 in train.log [7.500001] while stdout kept genuine token [7.500000])
    drift_tlog_p = seal_dir / "drift_train.log"
    drift_vla = [7.500000] * 300
    drift_vla[295] = 7.196000
    drift_vla[299] = 7.200000
    drift_vla[150] = 7.500001  # 151st epoch drifted by 0.000001
    drift_tlog_dict = {
        "best_val_loss": 7.196000,
        "history": {
            "train_loss": [8.000000] * 300,
            "val_loss": drift_vla,
        },
    }
    drift_tlog_bytes = json.dumps(drift_tlog_dict).encode("utf-8")
    drift_tsha = _make_dummy_file_with_sha(drift_tlog_p, drift_tlog_bytes)
    r_drift = copy.deepcopy(valid_report)
    r_drift["evaluator_binding"]["source_seal"]["train_log"] = str(drift_tlog_p)
    r_drift["evaluator_binding"]["source_seal"]["train_log_sha256"] = drift_tsha
    r_drift["evaluator_binding"]["source_provenance"]["history"]["val_loss"] = drift_vla
    p_drift_json = tmp_path / "r_drift.json"
    p_drift_json.write_text(json.dumps(r_drift), encoding="utf-8")
    with pytest.raises(ValueError, match=r"contradicts bound train\.log val_loss"):
        validate_checkpoint_config(ckpt_path, "k_0.5", run_receipt_path=p_drift_json)

    # 11. Fully valid assessor report succeeds
    good_receipt = tmp_path / "good_assessor_report.json"
    good_receipt.write_text(json.dumps(valid_report), encoding="utf-8")
    loaded_with_assessor = validate_checkpoint_config(
        ckpt_path, "k_0.5", run_receipt_path=good_receipt
    )
    assert loaded_with_assessor is not None
    assert loaded_with_assessor["config"]["model"]["persistence_skip"] == 0.5


def test_cache_binding_validation_strictly_ordered() -> None:
    """validate_cache_binding strictly enforces role, hashes, ordered IDs, and lengths."""
    y_true_seqs = [
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([4.0, 5.0], dtype=np.float64),
    ]
    composite_trial_ids = ["sess1::0", "sess2::1"]
    prefix_ids = ["pref_A", "pref_B"]

    valid_meta = {
        "candidate_id": "k_0.5",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "nested_prior_sha256": EXPECTED_NESTED_PRIOR_SHA256,
        "full_target_sha256": EXPECTED_FULL_TARGET_SHA256,
        "eligible_target_sha256": EXPECTED_ELIG_TARGET_SHA256,
        "checkpoint_sha256": "c" * 64,
        "composite_trial_ids": ["sess1::0", "sess2::1"],
        "trial_lengths": [3, 2],
        "prefix_ids": ["pref_A", "pref_B"],
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

    # 1. Exact valid binding passes
    validate_cache_binding(
        cache_meta=valid_meta,
        y_true_seqs=y_true_seqs,
        composite_trial_ids=composite_trial_ids,
        prefix_ids=prefix_ids,
        expected_candidate_id="k_0.5",
        expected_checkpoint_sha="c" * 64,
    )

    # 2. Candidate role mismatch fails closed
    with pytest.raises(ValueError, match="Cache candidate_id mismatch"):
        validate_cache_binding(
            cache_meta=valid_meta,
            y_true_seqs=y_true_seqs,
            composite_trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
            expected_candidate_id="k_1.0",
        )

    # 3. Checkpoint SHA mismatch fails closed
    with pytest.raises(ValueError, match="Cache checkpoint_sha256 mismatch"):
        validate_cache_binding(
            cache_meta=valid_meta,
            y_true_seqs=y_true_seqs,
            composite_trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
            expected_candidate_id="k_0.5",
            expected_checkpoint_sha="d" * 64,
        )

    # 4. Target hashes match, but composite_trial_ids reversed -> fails on composite IDs specifically
    reversed_id_meta = copy.deepcopy(valid_meta)
    reversed_id_meta["composite_trial_ids"] = ["sess2::1", "sess1::0"]
    with pytest.raises(
        ValueError,
        match="Cache composite_trial_ids do not match canonical trial IDs or ordering",
    ):
        validate_cache_binding(
            cache_meta=reversed_id_meta,
            y_true_seqs=y_true_seqs,
            composite_trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
            expected_candidate_id="k_0.5",
        )

    # 5. Target hashes match, but trial_lengths mismatched -> fails on trial_lengths specifically
    bad_len_meta = copy.deepcopy(valid_meta)
    bad_len_meta["trial_lengths"] = [2, 3]  # swap lengths
    with pytest.raises(
        ValueError, match="Cache trial_lengths do not match canonical target lengths"
    ):
        validate_cache_binding(
            cache_meta=bad_len_meta,
            y_true_seqs=y_true_seqs,
            composite_trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
            expected_candidate_id="k_0.5",
        )

    # 6. M3: Non-integer or boolean trial lengths in cache fail closed
    bad_float_len = copy.deepcopy(valid_meta)
    bad_float_len["trial_lengths"] = [3.0, 2.0]
    with pytest.raises(ValueError, match="must be genuine integer"):
        validate_cache_binding(
            cache_meta=bad_float_len,
            y_true_seqs=y_true_seqs,
            composite_trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
            expected_candidate_id="k_0.5",
        )

    bad_bool_len = copy.deepcopy(valid_meta)
    bad_bool_len["trial_lengths"] = [True, False]
    with pytest.raises(ValueError, match="must be genuine integer"):
        validate_cache_binding(
            cache_meta=bad_bool_len,
            y_true_seqs=y_true_seqs,
            composite_trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
            expected_candidate_id="k_0.5",
        )

    # 7. M3: Missing prefix_ids fails closed
    no_pref_meta = copy.deepcopy(valid_meta)
    del no_pref_meta["prefix_ids"]
    with pytest.raises(ValueError, match="missing mandatory 'prefix_ids' list"):
        validate_cache_binding(
            cache_meta=no_pref_meta,
            y_true_seqs=y_true_seqs,
            composite_trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
            expected_candidate_id="k_0.5",
        )

    # 8. M3: Missing or invalid lineage fails closed
    no_lin_meta = copy.deepcopy(valid_meta)
    del no_lin_meta["lineage"]
    with pytest.raises(ValueError, match="missing mandatory 'lineage' dictionary"):
        validate_cache_binding(
            cache_meta=no_lin_meta,
            y_true_seqs=y_true_seqs,
            composite_trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
            expected_candidate_id="k_0.5",
        )

    # 9. M3: Candidate missing or non-hex checkpoint_sha256 fails closed
    no_sha_meta = copy.deepcopy(valid_meta)
    no_sha_meta["checkpoint_sha256"] = None
    with pytest.raises(ValueError, match="valid 64-character hexadecimal checkpoint_sha256"):
        validate_cache_binding(
            cache_meta=no_sha_meta,
            y_true_seqs=y_true_seqs,
            composite_trial_ids=composite_trial_ids,
            prefix_ids=prefix_ids,
            expected_candidate_id="k_0.5",
        )


def test_synthetic_cli_evaluate_and_family_flow(tmp_path: Path) -> None:
    """Synthetic integration test of evaluate_arm and family assembly."""
    ds_path = tmp_path / "ds.pt"
    nest_path = tmp_path / "nest.pt"
    ckpt_path = tmp_path / "ckpt.pth"
    out_dir = tmp_path / "eval_out"

    # Make synthetic sequences
    y_seqs = [np.array([1.0, 2.0, 3.0], dtype=np.float32) for _ in range(3)]
    ds_payload = {
        "Y_seqs": y_seqs,
        "session_ids": ["sess1", "sess1", "sess2"],
        "trial_ids": [10, 11, 10],  # trial 10 repeated across sessions
        "anchor_frames": [None, None, None],
    }
    nest_payload = {
        "val_indices": [0, 1, 2],
        "group_keys": ["group_A", "group_A", "group_B"],
    }
    torch.save(ds_payload, ds_path)
    torch.save(nest_payload, nest_path)

    binding = hash_target_sequences([y.astype(np.float64) for y in y_seqs])
    ckpt_payload = _make_valid_checkpoint_dict("head_only_baseline", 0.0)
    ckpt_payload["dataset_source_sha256"] = "mock_ds_sha"
    ckpt_payload["nested_prior_fingerprint"] = "mock_ds_sha"
    ckpt_payload["nested_prior_artifact_sha256"] = "mock_nest_sha"
    torch.save(ckpt_payload, ckpt_path)

    mock_ckpt_hex = "c" * 64
    with patch(
        "scripts.evaluate_model_optimization.EXPECTED_DATASET_SHA256",
        "mock_ds_sha",
    ), patch(
        "scripts.evaluate_model_optimization.EXPECTED_NESTED_PRIOR_SHA256",
        "mock_nest_sha",
    ), patch(
        "scripts.evaluate_model_optimization.BASELINE_CHECKPOINT_SHA256",
        mock_ckpt_hex,
    ), patch(
        "scripts.evaluate_model_optimization.EXPECTED_VAL_TRIALS",
        3,
    ), patch(
        "scripts.evaluate_model_optimization.EXPECTED_FULL_FRAMES",
        binding["n_full_frames"],
    ), patch(
        "scripts.evaluate_model_optimization.EXPECTED_ELIG_FRAMES",
        binding["n_eligible_frames"],
    ), patch(
        "scripts.evaluate_model_optimization.EXPECTED_FULL_TARGET_SHA256",
        binding["full_target_sha256"],
    ), patch(
        "scripts.evaluate_model_optimization.EXPECTED_ELIG_TARGET_SHA256",
        binding["eligible_target_sha256"],
    ), patch(
        "scripts.evaluate_model_optimization._compute_sha256",
        side_effect=lambda p: (
            "mock_ds_sha"
            if p == ds_path
            else (
                "mock_nest_sha"
                if p == nest_path
                else mock_ckpt_hex
            )
        ),
    ):
        mock_batch = (
            torch.zeros(3, 3, 4),
            torch.tensor(
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                dtype=torch.float32,
            ),
            torch.tensor([3, 3, 3], dtype=torch.int64),
        )

        mock_loader = [mock_batch]
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = (
            torch.tensor(
                [[1.1, 1.9, 3.1], [1.0, 2.1, 2.9], [1.1, 2.0, 3.0]],
                dtype=torch.float32,
            ),
            None,
        )

        with patch(
            "scripts.evaluate_model_optimization.build_dataloaders",
            return_value=(None, mock_loader),
        ), patch(
            "scripts.evaluate_model_optimization.load_model_from_checkpoint",
            return_value=mock_model,
        ):
            # 1. Forward pass evaluation with save_cache
            arm_res = evaluate_arm(
                candidate_id="head_only_baseline",
                checkpoint_path=ckpt_path,
                dataset_path=ds_path,
                nested_prior_path=nest_path,
                output_dir=out_dir,
                save_cache=True,
                exchangeability_asserted=True,
            )
            assert arm_res["candidate_id"] == "head_only_baseline"
            assert arm_res["composite_trial_ids"] == [
                "sess1::10",
                "sess1::11",
                "sess2::10",
            ]

            # 2. String bool exchangeability_asserted rejected at API boundary
            with pytest.raises(ValueError, match="genuine bool"):
                evaluate_arm(
                    candidate_id="head_only_baseline",
                    checkpoint_path=ckpt_path,
                    dataset_path=ds_path,
                    nested_prior_path=nest_path,
                    output_dir=out_dir,
                    exchangeability_asserted="false",  # type: ignore
                )

            # 3. Early preflight refusal before any cache overwrite
            with pytest.raises(FileExistsError, match="Metrics file already exists"):
                evaluate_arm(
                    candidate_id="head_only_baseline",
                    checkpoint_path=ckpt_path,
                    dataset_path=ds_path,
                    nested_prior_path=nest_path,
                    output_dir=out_dir,
                    save_cache=True,
                    allow_overwrite=False,
                )

            # 4. Cache loading branch with strict metadata validation
            cache_npz = out_dir / "head_only_baseline_predictions.npz"
            assert cache_npz.exists()

            cache_arm_res = evaluate_arm(
                candidate_id="head_only_baseline",
                checkpoint_path=ckpt_path,
                dataset_path=ds_path,
                nested_prior_path=nest_path,
                output_dir=out_dir,
                load_cache_path=cache_npz,
                allow_overwrite=True,
                exchangeability_asserted=True,
            )
            assert (
                cache_arm_res["composite_trial_ids"]
                == arm_res["composite_trial_ids"]
            )

            # 5. Assemble family
            family_out = out_dir / "family.json"
            family = assemble_and_save_family(
                baseline_cache_path=cache_npz,
                cand_05_cache_path=None,
                cand_10_cache_path=None,
                dataset_path=ds_path,
                nested_prior_path=nest_path,
                output_path=family_out,
                exchangeability_asserted=True,
                allow_overwrite=True,
            )
            assert family["family_size"] == 78
            assert family["both_arms_present"] is False
            assert family_out.exists()

            # 6. Unbacked candidate cache (e.g. "deadbeef"*8 with no checkpoint file) fails closed
            unbacked_cand_cache = out_dir / "unbacked_cand_predictions.npz"
            unbacked_cand_json = out_dir / "unbacked_cand_predictions.json"
            np.savez_compressed(
                unbacked_cand_cache,
                flat_preds=np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float64),
                lengths=np.array([3, 3, 3], dtype=np.int64),
            )
            unbacked_cache_meta = {
                "candidate_id": "k_0.5",
                "checkpoint": str(out_dir / "nonexistent_ckpt.pth"),
                "checkpoint_sha256": "deadbeef" * 8,
                "dataset_sha256": "mock_ds_sha",
                "nested_prior_sha256": "mock_nest_sha",
                "full_target_sha256": binding["full_target_sha256"],
                "eligible_target_sha256": binding["eligible_target_sha256"],
                "trial_lengths": [3, 3, 3],
                "composite_trial_ids": ["sess1::10", "sess1::11", "sess2::10"],
                "prefix_ids": arm_res["prefix_ids"],
                "lineage": {
                    "validation_scope": "nested_outer_validation",
                    "nested_split_seed": 42,
                    "nested_val_split": 0.2,
                    "crop": {"max_seq_len": 2400, "pre_anchor_frames": 1200},
                },
            }
            unbacked_cand_json.write_text(json.dumps(unbacked_cache_meta), encoding="utf-8")

            with pytest.raises(FileNotFoundError, match="Candidate k_0.5 checkpoint file not found"):
                assemble_and_save_family(
                    baseline_cache_path=cache_npz,
                    cand_05_cache_path=unbacked_cand_cache,
                    cand_10_cache_path=None,
                    dataset_path=ds_path,
                    nested_prior_path=nest_path,
                    output_path=out_dir / "family_unbacked.json",
                    exchangeability_asserted=True,
                    allow_overwrite=True,
                )


def test_synthetic_cli_refusal_gates(tmp_path: Path) -> None:
    """Verify fail-closed refusals on corrupted inputs, pins, and overwrite."""
    missing_ds = tmp_path / "missing_ds.pt"
    missing_nest = tmp_path / "missing_nest.pt"

    # 1. Missing pin files
    with pytest.raises(FileNotFoundError, match="Dataset path not found"):
        validate_immutable_pins(missing_ds, missing_nest)

    fake_ds = tmp_path / "fake_ds.pt"
    fake_ds.write_bytes(b"bad_bytes")
    with pytest.raises(FileNotFoundError, match="Nested prior not found"):
        validate_immutable_pins(fake_ds, missing_nest)

    fake_nest = tmp_path / "fake_nest.pt"
    fake_nest.write_bytes(b"bad_bytes")

    # 2. SHA mismatch on pins
    with pytest.raises(ValueError, match="Dataset SHA-256 mismatch"):
        validate_immutable_pins(fake_ds, fake_nest)

    # 3. Cache corruption refusal: corrupt lengths sum
    corrupt_npz = tmp_path / "corrupt_cache.npz"
    corrupt_json = tmp_path / "corrupt_cache.json"
    np.savez_compressed(
        corrupt_npz,
        flat_preds=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        lengths=np.array([2], dtype=np.int64),  # sum=2 != size=3
    )
    corrupt_json.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Sum of lengths"):
        load_prediction_cache(tmp_path / "corrupt_cache")

    # 4. Overwrite refusal on family output
    fam_out = tmp_path / "fam_exist.json"
    fam_out.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        assemble_and_save_family(
            baseline_cache_path=tmp_path / "nonexistent.npz",
            cand_05_cache_path=None,
            cand_10_cache_path=None,
            dataset_path=fake_ds,
            nested_prior_path=fake_nest,
            output_path=fam_out,
            allow_overwrite=False,
        )
