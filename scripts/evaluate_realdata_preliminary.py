"""Score the real-data PRELIMINARY-conclusions arms into one declared family.

Reuses the frozen, already-reviewed statistical primitives instead of
re-implementing them:

* :func:`scripts.evaluate_model_optimization.extract_canonical_validation_targets`
  for the canonical ``nested_outer_validation`` targets, composite trial ids
  and recording-prefix ids (with its target-binding hash gate).
* :func:`nsmor.analysis.optimization_evaluation.compute_aligned_grid_metrics`
  for the pooled metrics and the 13 aligned scopes.
* :func:`nsmor.analysis.optimization_evaluation.assemble_declared_family` for
  the whole 78-slot family (delegated to, not re-implemented); it in turn uses
  :func:`nsmor.analysis.model_comparison.paired_mse_comparison` (trial-level
  paired ``d_z`` + prefix-cluster sign-flip) and
  :func:`nsmor.analysis.model_comparison.holm_correct_declared_family`
  (family-wise adjustment).

The declared inferential family is FROZEN by
``docs/realdata-preliminary-protocol-20261009.md``:

    candidates   A1, A2                          (2)
    comparators  head_only_baseline, persistence, zero   (3)
    scopes       aligned_overall + 12 sustained cells    (13)
    family size  2 * 3 * 13 = 78

This script is a THIN DRIVER. It does not train, does not touch the frozen
k-family protocol, and does not modify any core module. It also emits the §6
A2 compute-accounting report (descriptive halting statistics).

Usage::

    python scripts/evaluate_realdata_preliminary.py \\
        --baseline .scratch/model-opt-baseline-300-20261003-rerun/best_model.pth \\
        --a1 .scratch/realdata-preliminary-20261009/A1-swiglu-off/best_model.pth \\
        --a2 .scratch/realdata-preliminary-20261009/A2-swiglu-adaptive/best_model.pth \\
        --output_dir results/realdata-preliminary-20261009
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

# Scripts are run as ``python scripts/<name>.py``; make sibling imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_model_optimization import (  # noqa: E402
    BASELINE_CHECKPOINT_SHA256,
    extract_canonical_validation_targets,
    validate_checkpoint_config,
)
from nsmor.analysis.model_comparison import (  # noqa: E402
    COMPARATOR_IDS,
    DECLARED_FAMILY_SIZE,
    SCOPE_IDS,
    declared_family_slots as _frozen_declared_family_slots,
)
from nsmor.analysis.optimization_evaluation import (  # noqa: E402
    assemble_declared_family,
    compute_aligned_grid_metrics,
)
from nsmor.analysis.prediction_units import load_model_from_checkpoint  # noqa: E402
from nsmor.config_parser import ExperimentConfig  # noqa: E402
from nsmor.pipeline.nested_prior import load_artifact_bytes  # noqa: E402
from train import build_dataloaders  # noqa: E402

logger = logging.getLogger(__name__)

# ── Frozen declared family (docs/realdata-preliminary-protocol-20261009.md) ──
# Only the candidate IDs differ from the k-family; the comparator grid, the 13
# scopes and the family size are imported verbatim from the frozen module so
# there is exactly one definition of the family algebra.
CANDIDATE_IDS: Tuple[str, ...] = ("A1", "A2")
assert DECLARED_FAMILY_SIZE == 78, "Declared real-data family must be 78"
assert len(CANDIDATE_IDS) == 2, "Real-data family declares exactly two candidates"

# ── Arm identity contract (protocol §2) ─────────────────────────────────────
# Each arm's declared config delta. A checkpoint handed in under the wrong arm
# label (or a swapped path) is refused rather than scored with inverted
# semantics.
EXPECTED_ARM_KEYS: Dict[str, Dict[str, Any]] = {
    "A1": {"activation": "swiglu", "refinement_mode": "off", "lambda_compute": 0.0},
    "A2": {"activation": "swiglu", "refinement_mode": "adaptive", "lambda_compute": 0.01},
}


def declared_family_slots() -> Tuple[str, ...]:
    """Return the frozen 78 slots for the A1/A2 real-data family."""
    slots = _frozen_declared_family_slots(CANDIDATE_IDS)
    assert len(slots) == DECLARED_FAMILY_SIZE == 78
    return slots


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_arm_config(cfg: ExperimentConfig, arm_id: str) -> None:
    """Fail closed unless a checkpoint's config matches its declared arm.

    Binds the runtime config to the protocol's frozen arm deltas so a swapped
    or wrong checkpoint cannot be scored under an arm label it does not have.
    """
    exp = EXPECTED_ARM_KEYS[arm_id]
    got = (str(cfg.model.activation), str(cfg.model.refinement_mode))
    want = (exp["activation"], exp["refinement_mode"])
    if got != want:
        raise ValueError(
            f"{arm_id}: checkpoint activation/refinement_mode {got} != "
            f"declared {want}"
        )
    if not math.isclose(
        float(cfg.loss.lambda_compute), float(exp["lambda_compute"]),
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError(
            f"{arm_id}: checkpoint lambda_compute {cfg.loss.lambda_compute} != "
            f"declared {exp['lambda_compute']}"
        )
    if not math.isclose(
        float(getattr(cfg.model, "persistence_skip", 0.0)), 0.0,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError(f"{arm_id}: persistence_skip must be 0.0")


def _build_model_and_loader(
    checkpoint_path: Path,
    dataset_path: Path,
    nested_prior_path: Path,
    device: torch.device,
) -> Tuple[ExperimentConfig, torch.nn.Module, Any]:
    """Rebuild a checkpoint's model (from its own config) and val loader."""
    if nested_prior_path is None:
        # Fail closed: without the artifact ``build_dataloaders`` would
        # silently recompute a 0.2 split, which is NOT the canonical split.
        raise ValueError("nested_prior_path is required for the canonical split")

    ckpt = load_artifact_bytes(checkpoint_path.read_bytes(), map_location="cpu")
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    cfg.training.num_workers = 0
    model = load_model_from_checkpoint(checkpoint_path, device)
    _, val_loader = build_dataloaders(
        config=cfg,
        dataset_path=str(dataset_path),
        nested_prior_artifact=str(nested_prior_path),
    )
    if val_loader is None:
        raise RuntimeError(f"No validation loader for {checkpoint_path}")
    return cfg, model, val_loader


def predict_arm(
    checkpoint_path: Path,
    dataset_path: Path,
    nested_prior_path: Path,
    y_true_seqs: Sequence[np.ndarray],
    device: torch.device,
    *,
    expected_arm: Optional[str] = None,
) -> List[np.ndarray]:
    """Run one checkpoint over the canonical validation loader.

    Rebuilds the model from the checkpoint's own saved config (so the arm's
    ``activation`` / ``refinement_mode`` are honored), refuses a config that
    does not match ``expected_arm``, then verifies the loader-produced targets
    match the canonical targets element-for-element before returning.
    """
    cfg, model, val_loader = _build_model_and_loader(
        checkpoint_path, dataset_path, nested_prior_path, device
    )
    if expected_arm is not None:
        _validate_arm_config(cfg, expected_arm)

    model.eval()
    preds: List[np.ndarray] = []
    loader_true: List[np.ndarray] = []
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
            assert y_p.shape == (x_b.size(0), x_b.size(1)), (
                f"Prediction shape {tuple(y_p.shape)} != "
                f"{(x_b.size(0), x_b.size(1))}"
            )
            for i in range(x_b.size(0)):
                n = int(len_b[i])
                loader_true.append(y_b[i, :n].cpu().numpy().astype(np.float64))
                preds.append(y_p[i, :n].cpu().numpy().astype(np.float64))

    if len(preds) != len(y_true_seqs):
        raise ValueError(
            f"{checkpoint_path}: loader trial count {len(preds)} != "
            f"canonical {len(y_true_seqs)}"
        )
    for i, (lt, ct) in enumerate(zip(loader_true, y_true_seqs)):
        if not np.array_equal(lt, ct):
            raise ValueError(
                f"{checkpoint_path}: loader trial {i} targets do not match canonical"
            )
    return preds


def _val_stimulus_conditions(
    dataset_path: Path, nested_prior_path: Path
) -> np.ndarray:
    """Return the authoritative stimulus condition per validation trial.

    Reads the dataset's own ``stimulus_conditions`` stamp and the nested
    prior's ``val_indices`` (the canonical split), so the condition label is
    NOT inferred from cropped loader tensors. Fails closed on any val trial
    whose condition is outside the three physical classes used by the routing
    contrast, so a future ``no_stimulus`` batch cannot be silently folded into
    the ``visual_present`` complement.
    """
    data = load_artifact_bytes(dataset_path.read_bytes(), map_location="cpu")
    prior = load_artifact_bytes(nested_prior_path.read_bytes(), map_location="cpu")
    if "stimulus_conditions" not in data:
        raise ValueError("dataset has no stimulus_conditions stamp")
    conditions = np.asarray(data["stimulus_conditions"])
    val_idx = np.asarray(prior["val_indices"])
    val_conditions = conditions[val_idx]
    allowed = {"wind_only", "visual_only", "multisensory"}
    unexpected = sorted(set(val_conditions.tolist()) - allowed)
    if unexpected:
        raise ValueError(
            "validation split contains condition(s) outside the declared "
            f"wind/visual classes: {unexpected}; refusing to fold them into "
            "the visual_present complement"
        )
    return val_conditions


def compute_condition_accounting(
    checkpoint_path: Path,
    dataset_path: Path,
    nested_prior_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """§6 A2 compute-accounting: descriptive halting statistics over val.

    Reports the ACTUAL refinement depth / update counts per valid frame, split
    by condition (wind_only vs visual_present) and pooled, from the model's own
    internals. Per the core contract ``refinement_depth`` is ``(B, T)`` and
    ``refinement_updates`` is a **0-dim scalar** (``depth.sum()``), so the
    per-condition update total is the per-frame depth sum over that condition's
    valid frames; ``refinement_ponder_cost`` is likewise a batch scalar and is
    reported pooled only (labelled as such).

    The condition label comes from the dataset's authoritative
    ``stimulus_conditions`` (via the nested split), and the split is applied
    with the loader's ``wind_only_mask``; the two are cross-checked. Any val
    trial outside {wind_only, visual_only, multisensory} raises (fail closed).
    When the loader carries no ``wind_only_mask`` the split is reported
    unavailable and pooled statistics are still emitted.
    """
    val_conditions = _val_stimulus_conditions(dataset_path, nested_prior_path)
    n_wind_expected = int(np.sum(val_conditions == "wind_only"))

    _cfg, model, val_loader = _build_model_and_loader(
        checkpoint_path, dataset_path, nested_prior_path, device
    )
    model.eval()

    dep_all = 0.0
    nf_all = 0
    dep_w = dep_v = 0.0
    nf_w = nf_v = 0
    n_wind_seen = 0
    n_trials_seen = 0
    ponders: List[float] = []
    has_mask = True
    with torch.no_grad():
        for batch in val_loader:
            x_b, _y_b, len_b = batch[0], batch[1], batch[2]
            wind = batch[3] if len(batch) >= 4 else None
            if wind is None:
                has_mask = False
            x_b = x_b.to(device).contiguous()
            len_b = len_b.to(device).contiguous()
            _y, internals = model(x_b, len_b, return_internals=True)
            depth = internals["refinement_depth"]
            upd = internals["refinement_updates"]
            B, T = x_b.size(0), x_b.size(1)
            # Core contract: depth is (B, T); updates is a 0-dim scalar total.
            assert depth.shape == (B, T), (
                f"refinement_depth shape {tuple(depth.shape)} != {(B, T)}"
            )
            assert upd.dim() == 0, (
                f"refinement_updates must be a 0-dim scalar, got "
                f"{tuple(upd.shape)}"
            )
            valid = torch.arange(T, device=device)[None, :] < len_b[:, None]
            dep_all += float(depth[valid].sum())
            nf_all += int(valid.sum())
            n_trials_seen += B
            if wind is not None:
                wind_b = wind.to(device).bool()[:, None]
                n_wind_seen += int(wind_b.sum())
                m_w = valid & wind_b
                m_v = valid & (~wind_b)
                dep_w += float(depth[m_w].sum())
                nf_w += int(m_w.sum())
                dep_v += float(depth[m_v].sum())
                nf_v += int(m_v.sum())
            ponders.append(float(internals["refinement_ponder_cost"]))

    # Cross-check the loader's wind mask against the authoritative condition
    # stamp so a loader/label mismatch fails closed rather than mis-splitting.
    if has_mask and (n_wind_seen != n_wind_expected
                     or n_trials_seen != int(val_conditions.size)):
        raise ValueError(
            "val loader wind mask disagrees with the dataset condition stamp: "
            f"wind {n_wind_seen} != {n_wind_expected}, "
            f"trials {n_trials_seen} != {int(val_conditions.size)}"
        )

    def _mean(total: float, n: int) -> Optional[float]:
        return (total / n) if n else None

    def _total(total: float, n: int) -> Optional[float]:
        return total if n else None

    pooled = {
        "n_valid_frames": nf_all,
        "mean_refinement_depth": _mean(dep_all, nf_all),
        # total updates == sum of per-frame depth (core: updates = depth.sum()).
        "total_refinement_updates": _total(dep_all, nf_all),
        "mean_refinement_ponder_cost": (
            float(np.mean(ponders)) if ponders else None
        ),
    }
    if not has_mask:
        return {
            "pooled": pooled,
            "wind_only": None,
            "visual_present": None,
            "condition_split_available": False,
            "condition_split_unavailable_reason": "no_wind_only_mask",
            "ponder_cost_scope": "pooled_only",
            "ponder_cost_note": (
                "refinement_ponder_cost is emitted as a batch scalar; it is "
                "not split per condition and is not a biological/ATP measure."
            ),
        }
    return {
        "pooled": pooled,
        "wind_only": {
            "n_valid_frames": nf_w,
            "mean_refinement_depth": _mean(dep_w, nf_w),
            "total_refinement_updates": _total(dep_w, nf_w),
        },
        "visual_present": {
            "n_valid_frames": nf_v,
            "mean_refinement_depth": _mean(dep_v, nf_v),
            "total_refinement_updates": _total(dep_v, nf_v),
        },
        "condition_split_available": True,
        "condition_split_unavailable_reason": None,
        "ponder_cost_scope": "pooled_only",
        "ponder_cost_note": (
            "refinement_ponder_cost is emitted as a batch scalar; it is not "
            "split per condition and is not a biological/ATP measure."
        ),
    }


def assemble_family(
    candidate_grids: Mapping[str, Mapping[str, Any]],
    baseline_grid: Mapping[str, Any],
    *,
    exchangeability_asserted: bool = False,
) -> Dict[str, Any]:
    """Assemble the frozen 78-slot family by DELEGATING to the frozen assembler.

    The comparator scope data all come from the baseline grid: ``persistence``
    and ``zero`` vectors are computed inside
    :func:`compute_aligned_grid_metrics` from the canonical targets, so the
    baseline grid already carries them. No family algebra is re-implemented
    here — the 78 slots, the ``paired_mse_comparison`` statistics and the Holm
    bookkeeping are produced by the frozen primitives.
    """
    comparator_scopes: Dict[str, Mapping[str, Any]] = {
        "head_only_baseline": baseline_grid["scope_data"],
        "persistence": baseline_grid["scope_data"],
        "zero": baseline_grid["scope_data"],
    }
    candidate_scopes: Dict[str, Mapping[str, Any]] = {
        cand_id: grid["scope_data"] for cand_id, grid in candidate_grids.items()
    }
    return assemble_declared_family(
        candidate_scopes,
        comparator_scopes,
        exchangeability_asserted=exchangeability_asserted,
        candidate_ids=CANDIDATE_IDS,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, required=True,
        help="Verified baseline (relu/off) best_model.pth (read-only).",
    )
    parser.add_argument("--a1", type=Path, default=None, help="A1 swiglu/off checkpoint.")
    parser.add_argument("--a2", type=Path, default=None, help="A2 swiglu/adaptive checkpoint.")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path(
            ".scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/"
            "etl-causal-final-zp12m0fa/nsmor_dataset.pt"
        ),
    )
    parser.add_argument(
        "--nested_prior", type=Path,
        default=Path(
            ".scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/"
            "nested-causal-complete-dku91_m0/nested_split_seed42.pt"
        ),
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("results/realdata-preliminary-20261009"),
    )
    parser.add_argument(
        "--exchangeability_asserted", action="store_true",
        help="Assert the joint sign-exchangeability null (descriptive only).",
    )
    args = parser.parse_args(argv)

    # The head-only comparator MUST be the frozen baseline; a wrong checkpoint
    # silently used as the reference would produce a fully "valid" family
    # against the wrong reference. Pin the SHA and role-validate.
    baseline_sha = _sha256(args.baseline)
    if baseline_sha != BASELINE_CHECKPOINT_SHA256:
        raise ValueError(
            f"Baseline checkpoint SHA-256 mismatch: expected "
            f"{BASELINE_CHECKPOINT_SHA256}, got {baseline_sha}"
        )
    validate_checkpoint_config(args.baseline, "head_only_baseline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true_seqs, trial_ids, prefix_ids = extract_canonical_validation_targets(
        args.dataset, args.nested_prior
    )
    assert len(y_true_seqs) == len(trial_ids) == len(prefix_ids)

    def grid_for(path: Path, arm_id: Optional[str] = None) -> Dict[str, Any]:
        preds = predict_arm(
            path, args.dataset, args.nested_prior, y_true_seqs, device,
            expected_arm=arm_id,
        )
        return compute_aligned_grid_metrics(
            y_true_seqs=y_true_seqs,
            y_pred_seqs=preds,
            trial_ids=trial_ids,
            prefix_ids=prefix_ids,
        )

    baseline_grid = grid_for(args.baseline)
    candidate_grids: Dict[str, Dict[str, Any]] = {}
    arm_provenance: Dict[str, Any] = {}
    for arm_id, path in (("A1", args.a1), ("A2", args.a2)):
        if path is None:
            continue
        candidate_grids[arm_id] = grid_for(path, arm_id)
        arm_provenance[arm_id] = {
            "checkpoint": str(path),
            "sha256": _sha256(path),
            "expected_keys": EXPECTED_ARM_KEYS[arm_id],
        }

    family = assemble_family(
        candidate_grids, baseline_grid,
        exchangeability_asserted=args.exchangeability_asserted,
    )

    # §6 compute accounting (A2 only; descriptive, not a biological claim).
    accounting: Optional[Dict[str, Any]] = None
    if args.a2 is not None:
        accounting = compute_condition_accounting(
            args.a2, args.dataset, args.nested_prior, device
        )

    payload = {
        "protocol": "docs/realdata-preliminary-protocol-20261009.md",
        "declared_family_size": DECLARED_FAMILY_SIZE,
        "candidate_ids": list(CANDIDATE_IDS),
        "comparator_ids": list(COMPARATOR_IDS),
        "scope_ids": list(SCOPE_IDS),
        "baseline": {
            "checkpoint": str(args.baseline),
            "sha256": baseline_sha,
        },
        "arms": arm_provenance,
        "pooled_metrics": {
            "baseline": baseline_grid["raw_full_frame"],
            **{k: v["raw_full_frame"] for k, v in candidate_grids.items()},
        },
        "aligned_overall": {
            "baseline": baseline_grid["aligned_overall"],
            **{k: v["aligned_overall"] for k, v in candidate_grids.items()},
        },
        "declared_family": family,
        "a2_compute_accounting": accounting,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "realdata_preliminary_family.json"
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    logger.info(
        "Wrote %s (family=%d, valid_tests=%d, unavailable=%d)",
        out, family["family_size"],
        family["holm_bonferroni"]["valid_test_count"],
        family["holm_bonferroni"]["unavailable_slot_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
