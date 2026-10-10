"""Score the real-data PHASE-2 arms (R0-R3, 3 seeds) into one declared family.

Prospective driver for `docs/realdata-phase2-protocol-20261010.md`.  It trains
nothing and modifies no core module; it REUSES the frozen, already-reviewed
statistical primitives :func:`nsmor.analysis.model_comparison.paired_mse_comparison`
(trial-level paired ``d_z`` + prefix-cluster sign-flip) and
:func:`nsmor.analysis.uq.holm_bonferroni` (Holm step-down) for every slot.

Declared family (protocol §4):

    candidates   R0, R1, R2, R3                          (4)
    comparators  persistence, zero, R0                   (3)
    scopes       aligned_pooled, escape, h5, onset       (4)
    family size  4 * 3 * 4 = 48

``R0|<...>`` self-comparison slots are reported unavailable (reason
``self_comparison``); they are never silently dropped.  Holm-Bonferroni is
applied over the FIXED 48-slot family with every unavailable slot carried as
``p = 1`` (protocol §6), so the first threshold is ``0.05 / 48`` and the
resolution-limit arithmetic in the protocol holds.

The two script-owned phase-2 controls (``selection_metric``,
``zero_input_channels``) are recorded by ``scripts/train.py`` as additive
top-level checkpoint keys (NOT in ``config.to_dict()``); this scorer reads them
from there to bind each checkpoint to its declared arm and to reproduce the
history ablation at inference (train/serve consistency).

Exchangeability is NOT asserted by default (the preliminary phase's mishap):
p-values are null with an explicit reason unless ``--exchangeability_asserted``
is passed, and the exact flag/command is recorded in the output provenance.

Usage::

    python scripts/evaluate_realdata_phase2.py \\
        --r0 <ckpt> <ckpt> <ckpt> --r1 ... --r2 ... --r3 ... \\
        --dataset <nsmor_dataset.pt> --nested_prior <nested_split_seed42.pt> \\
        --output_dir results/realdata-phase2-20261010 \\
        --exchangeability_asserted
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_model_optimization import (  # noqa: E402
    extract_canonical_validation_targets,
)
from nsmor.analysis.model_comparison import paired_mse_comparison  # noqa: E402
from nsmor.analysis.prediction_units import load_model_from_checkpoint  # noqa: E402
from nsmor.analysis.uq import holm_bonferroni  # noqa: E402
from nsmor.config_parser import ExperimentConfig  # noqa: E402
from nsmor.pipeline.conditions import resolve_anchor_crop  # noqa: E402
from nsmor.pipeline.nested_prior import load_artifact_bytes  # noqa: E402
from train import build_dataloaders, _sustained_run  # noqa: E402

logger = logging.getLogger(__name__)

ARMS: Tuple[str, ...] = ("R0", "R1", "R2", "R3")
COMPARATORS: Tuple[str, ...] = ("persistence", "zero", "R0")
SCOPES: Tuple[str, ...] = ("aligned_pooled", "escape", "h5", "onset")
# Secondary scopes are reported, never substituted for the primary, and never
# enter the declared family (protocol §5).
SECONDARY_HORIZONS: Tuple[int, ...] = (25, 125)
SECONDARY_SCOPES: Tuple[str, ...] = tuple(f"h{h}" for h in SECONDARY_HORIZONS)
# Every horizon is reported BOTH overall and in escape frames (protocol §5,
# M1): the escape ceiling is only ever attached to an escape-frame scalar.
HORIZON_SCOPES: Tuple[str, ...] = ("h5",) + SECONDARY_SCOPES
HORIZON_ESCAPE_SCOPES: Tuple[str, ...] = tuple(
    f"{s}_escape" for s in HORIZON_SCOPES
)
# Every per-trial series the scorer computes (primary + secondary + horizon
# escape variants).  Only ``SCOPES`` enter the declared family.
ALL_SCOPES: Tuple[str, ...] = SCOPES + SECONDARY_SCOPES + HORIZON_ESCAPE_SCOPES
FAMILY_SIZE = len(ARMS) * len(COMPARATORS) * len(SCOPES)
assert FAMILY_SIZE == 48, "phase-2 declared family must be 48"

DT_MS = 4.0
ESCAPE_THRESHOLD_CM_S = 10.0
PRIMARY_HORIZON = 5
MULTISTEP_STARTS = 8
ONSET_WIN_MS = 200.0
BATCH = 16
# Anchor-aligned crop constants.  ``max_seq_len`` is read from the checkpoint's
# resolved config (the same crop base the arm's val dataloader used); the
# pre-anchor baseline is the ``NSMoRDataset`` default used by build_dataloaders.
PRE_ANCHOR_FRAMES = 1200

# Measured per-scope persistence ceilings (protocol §5).  The horizon scopes
# use the scorer's OWN comparator — persistence HELD AT y_true[start-1] over
# the whole rollout (``_hN_trial``), which is NOT the pure lag-N frame the
# step-1 setup audit reported.  The two diverged sharply at h=5 (held-at-start
# MSE 0.925 vs pure lag-5 10.07), so the table must carry the comparator the
# scorer actually uses.  Values measured on the canonical nested validation
# split, frozen in ``docs/realdata-phase2-ceilings-20261010.json``.
PERSISTENCE_CEILING: Dict[str, Dict[str, float]] = {
    "aligned_pooled": {"r2": 0.927, "mse": 1.910},
    "escape": {"r2": 0.857, "mse": 53.05},
    "h5": {"r2": 0.741, "mse": 0.925},           # held-at-start (scorer comparator)
    "h5_escape": {"r2": -0.205, "mse": 41.44},   # held-at-start, escape frames
    "h25": {"r2": 0.424, "mse": 2.289},          # held-at-start
    "h25_escape": {"r2": -1.277, "mse": 132.13},  # held-at-start, escape frames
    "h125": {"r2": -0.305, "mse": 8.125},        # held-at-start
    "h125_escape": {"r2": -2.001, "mse": 366.95},  # held-at-start, escape frames
    "onset": {"r2": 0.880, "mse": 26.65},
}

# Pure lag-N audit values (step-1 setup audit).  These describe a DIFFERENT
# comparator (persistence of the frame exactly N steps back, not held at the
# rollout start) and are reference-only: they never score a phase-2 slot.
PURE_LAG_CEILING: Dict[str, Dict[str, float]] = {
    "h5": {"r2": 0.615, "mse": 10.07},
    "h5_escape": {"r2": 0.179, "mse": 303.65},
    "h25": {"r2": -0.246, "mse": 32.85},
    "h25_escape": {"r2": -1.157, "mse": 798.34},
    "h125": {"r2": -0.770, "mse": 48.50},
    "h125_escape": {"r2": -1.401, "mse": 890.26},
}

# Peak-speed reference.  The scorer measures the true per-trial peak on the
# SCORED split (all 432 val trials, aligned frames t>=1) and reports its own
# mean; this constant is the step-1 EVENT-level value (362 paired trials) and
# is retained only as a provenance annotation, never as the scored reference.
PEAK_SPEED_OBSERVED_CM_S = 65.8
PEAK_SPEED_OBSERVED_PROVENANCE = "step1_event_level_362_paired_trials"

# Declared arm-identity keys (protocol §2): the scorer binds exactly these,
# never the byte-identical leaves.  ``selection_metric`` and
# ``zero_input_channels`` are read from the checkpoint's top-level additive
# provenance (they are not part of the protected config schema).  The
# non-arm leaves (``lambda_jerk``) are derived from the A1 REFERENCE config so
# they cannot drift silently if A1 changes — only ``persistence_skip`` and
# ``zero_input_channels`` are arm-specific literals.
_A1_REFERENCE_YAML = (
    Path(__file__).resolve().parents[1]
    / "config" / "realdata-preliminary-20261009" / "A1-swiglu-off.yaml"
)


def _a1_reference_lambda_jerk() -> float:
    """Read the A1 reference config's ``loss.lambda_jerk`` (single source)."""
    import yaml

    raw = yaml.safe_load(_A1_REFERENCE_YAML.read_text(encoding="utf-8"))
    return float(raw["loss"]["lambda_jerk"])


def _a1_reference_shared() -> Dict[str, Any]:
    """Read A1's DECLARED SHARED base leaves (single source, cannot drift)."""
    import yaml

    raw = yaml.safe_load(_A1_REFERENCE_YAML.read_text(encoding="utf-8"))
    return {
        "activation": raw["model"]["activation"],
        "refinement_mode": raw["model"]["refinement_mode"],
        "lambda_compute": float(raw["loss"]["lambda_compute"]),
        "num_epochs": int(raw["training"]["num_epochs"]),
        "early_stopping_patience": int(raw["training"]["early_stopping_patience"]),
        "max_seq_len": int(raw["training"]["max_seq_len"]),
    }


_A1_LAMBDA_JERK = _a1_reference_lambda_jerk()
_A1_SHARED = _a1_reference_shared()

EXPECTED_ARM_KEYS: Dict[str, Dict[str, Any]] = {
    "R0": {"persistence_skip": 0.0, "zero_input_channels": [],
           "lambda_jerk": _A1_LAMBDA_JERK},
    "R1": {"persistence_skip": 1.0, "zero_input_channels": [],
           "lambda_jerk": _A1_LAMBDA_JERK},
    "R2": {"persistence_skip": 0.0, "zero_input_channels": [2, 3],
           "lambda_jerk": _A1_LAMBDA_JERK},
    "R3": {"persistence_skip": 0.0, "zero_input_channels": [],
           "lambda_jerk": 0.0},
}


def declared_family_slots() -> Tuple[str, ...]:
    """Return the frozen 48 phase-2 slots in deterministic order."""
    return tuple(
        f"{cand}|{comp}|{scope}"
        for cand in ARMS
        for comp in COMPARATORS
        for scope in SCOPES
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ckpt_phase2_keys(ckpt: Mapping[str, Any]) -> Dict[str, Any]:
    """Read the additive script-owned phase-2 keys from a checkpoint."""
    zic = ckpt.get("zero_input_channels", [])
    return {
        "zero_input_channels": sorted(int(c) for c in zic),
        "selection_metric": str(ckpt.get("selection_metric", "total")),
    }


def _validate_arm_config(ckpt: Mapping[str, Any], arm: str) -> None:
    """Fail closed unless a checkpoint's config matches its declared arm.

    Beyond the arm-specific leaves (``persistence_skip`` / ``lambda_jerk`` /
    ``zero_input_channels`` / ``selection_metric``) the DECLARED SHARED base
    (``activation`` / ``refinement_mode`` / ``lambda_compute`` /
    ``num_epochs`` / ``early_stopping_patience`` / ``max_seq_len``) is
    re-verified for defense in depth, so a checkpoint trained under a different
    common config cannot slip through the arm gate.
    """
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    exp = EXPECTED_ARM_KEYS[arm]
    if not math.isclose(float(cfg.model.persistence_skip), exp["persistence_skip"],
                        rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{arm}: persistence_skip {cfg.model.persistence_skip} != "
            f"declared {exp['persistence_skip']}"
        )
    if not math.isclose(float(cfg.loss.lambda_jerk), exp["lambda_jerk"],
                        rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{arm}: lambda_jerk {cfg.loss.lambda_jerk} != "
            f"declared {exp['lambda_jerk']}"
        )
    # Shared base leaves (single source: A1's resolved reference config).
    shared = {
        "model.activation": (str(cfg.model.activation), str(_A1_SHARED["activation"])),
        "model.refinement_mode": (
            str(cfg.model.refinement_mode), str(_A1_SHARED["refinement_mode"])),
        "loss.lambda_compute": (
            float(cfg.loss.lambda_compute), float(_A1_SHARED["lambda_compute"])),
        "training.num_epochs": (
            int(cfg.training.num_epochs), int(_A1_SHARED["num_epochs"])),
        "training.early_stopping_patience": (
            int(cfg.training.early_stopping_patience),
            int(_A1_SHARED["early_stopping_patience"])),
        "training.max_seq_len": (
            int(cfg.training.max_seq_len), int(_A1_SHARED["max_seq_len"])),
    }
    for leaf, (got, want) in shared.items():
        if got != want:
            raise ValueError(
                f"{arm}: shared base leaf {leaf} is {got!r}, declared {want!r}"
            )
    keys = _ckpt_phase2_keys(ckpt)
    if keys["zero_input_channels"] != sorted(exp["zero_input_channels"]):
        raise ValueError(
            f"{arm}: zero_input_channels {keys['zero_input_channels']} != "
            f"declared {sorted(exp['zero_input_channels'])}"
        )
    if keys["selection_metric"] != "mse":
        raise ValueError(f"{arm}: selection_metric must be 'mse'")


def _build_model_and_loader(
    checkpoint_path: Path, dataset_path: Path, nested_prior_path: Path,
    device: torch.device,
) -> Tuple[ExperimentConfig, Dict[str, Any], torch.nn.Module, Any]:
    """Rebuild a checkpoint's model (from its own config) and val loader.

    The history ablation is re-applied to the val loader through
    ``build_dataloaders`` so an ablated arm (R2) is scored on the SAME ablated
    inputs it was trained on (train/serve consistency).
    """
    ckpt = load_artifact_bytes(checkpoint_path.read_bytes(), map_location="cpu")
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    cfg.training.num_workers = 0
    model = load_model_from_checkpoint(checkpoint_path, device)
    zic = [int(c) for c in ckpt.get("zero_input_channels", [])]
    _, val_loader = build_dataloaders(
        config=cfg, dataset_path=str(dataset_path),
        nested_prior_artifact=str(nested_prior_path),
        zero_input_channels=zic,
    )
    if val_loader is None:
        raise RuntimeError(f"No validation loader for {checkpoint_path}")
    return cfg, ckpt, model, val_loader


def _require_loader_targets_match(
    loader_y: np.ndarray, canonical_y: np.ndarray, gi: int,
) -> None:
    """Fail closed unless a loader trial's targets equal the canonical series.

    B3 (protocol §5): the aligned scopes score the val LOADER's own ``y``, the
    rollout/onset scopes score ``extract_canonical_validation_targets``'
    ``y_true_seqs``.  This asserts the two are ONE series (per trial, in order),
    exactly as ``scripts/evaluate_realdata_preliminary.py:206-212`` does.
    """
    if not np.array_equal(
        np.asarray(loader_y, dtype=np.float64),
        np.asarray(canonical_y, dtype=np.float64),
    ):
        raise ValueError(
            f"val loader trial {gi} targets do not match the canonical "
            "extract_canonical_validation_targets series (fail closed)"
        )


def _prestim_false_alarm(
    yp: np.ndarray, yt: np.ndarray, onset: int,
) -> Tuple[int, int, int, int, int, int, int, int]:
    """Pre-stimulus spontaneous-escape false-alarm counts (protocol §8.1).

    Returns ``(pred_frame_num, pred_frame_den, pred_trial_num, pred_trial_den,
    animal_frame_num, animal_frame_den, animal_trial_num, animal_trial_den)``
    over the pre-stimulus segment ``[0, onset)`` inside the crop.  ``*_num`` /
    ``*_den`` are frame/trial numerators and denominators; the frame numerator
    counts model frames ``>= 10 cm/s`` while the animal is below threshold
    (``< 10 cm/s``), and the trial numerator counts windows with a predicted
    sustained (>= 2-frame) escape bout.
    """
    if onset <= 0:
        return (0, 0, 0, 0, 0, 0, 0, 0)
    pre_p = np.asarray(yp, dtype=np.float64)[:onset]
    pre_t = np.asarray(yt, dtype=np.float64)[:onset]
    below = pre_t < ESCAPE_THRESHOLD_CM_S
    pred_frame_num = int(((pre_p >= ESCAPE_THRESHOLD_CM_S) & below).sum())
    pred_bout = _sustained_run(pre_p >= ESCAPE_THRESHOLD_CM_S, min_run=2)
    animal_bout = _sustained_run(pre_t >= ESCAPE_THRESHOLD_CM_S, min_run=2)
    return (
        pred_frame_num, int(below.sum()),
        int(pred_bout.any()), 1,
        int((pre_t >= ESCAPE_THRESHOLD_CM_S).sum()), int(pre_t.size),
        int(animal_bout.any()), 1,
    )


def _response_onset(v_true: np.ndarray, onset: int,
                    thr: float = ESCAPE_THRESHOLD_CM_S) -> Optional[int]:
    a = np.abs(np.asarray(v_true, dtype=np.float64))
    idx = np.where(a[onset:] >= thr)[0]
    return None if idx.size == 0 else int(onset + idx[0])


def _compute_onsets(
    dataset_path: Path, nested_prior_path: Path, max_seq_len: Optional[int],
) -> np.ndarray:
    """Per-validation-trial response-onset frame in the CROPPED time base.

    ``max_seq_len`` must be the checkpoint's resolved crop length (2400 for
    every phase-2 arm), so the onset frame shares the exact crop base as the
    arm's val dataloader and ``extract_canonical_validation_targets``.  The
    bare ``ExperimentConfig()`` default (1000) would read a different crop.
    """
    if max_seq_len is None:
        raise ValueError(
            "checkpoint config has max_seq_len=None; cannot derive the "
            "anchor-aligned onset base"
        )
    if int(max_seq_len) != 2400:
        raise ValueError(
            f"checkpoint max_seq_len={max_seq_len} != the canonical 2400 crop "
            "pinned by extract_canonical_validation_targets; the onset base "
            "would not match the scored targets (fail closed)"
        )
    data = load_artifact_bytes(dataset_path.read_bytes(), map_location="cpu")
    prior = load_artifact_bytes(nested_prior_path.read_bytes(), map_location="cpu")
    val_idx = np.asarray(prior["val_indices"])
    anchors_all = np.asarray(data["anchor_frames"])
    lengths_all = np.asarray(data["lengths"])
    onsets = np.zeros(len(val_idx), dtype=int)
    for i in range(len(val_idx)):
        j = int(val_idx[i])
        start, _end = resolve_anchor_crop(
            n_frames=int(lengths_all[j]), anchor_frame=int(anchors_all[j]),
            max_seq_len=int(max_seq_len),
            pre_anchor_frames=PRE_ANCHOR_FRAMES,
        )
        onsets[i] = int(anchors_all[j]) - start
    return onsets


def _per_trial_scope_mse(
    checkpoint_path: Path, dataset_path: Path, nested_prior_path: Path,
    device: torch.device, y_true_seqs: Sequence[np.ndarray],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray],
           Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray],
           np.ndarray, Dict[str, Any], Dict[str, np.ndarray]]:
    """Per-trial MSE for the model and the two deterministic comparators.

    Scopes (identical eligible frames for model / persistence / zero):
      aligned_pooled  t >= 1, all frames
      escape          t >= 1 and |y_true[t]| >= 10 cm/s
      h5              PRIMARY 5-step own-prediction-feedback rollout
      h5_escape       the same rollout frames restricted to |y_true| >= 10
      onset           [onset, response_onset + 200 ms) aligned frames
      h25, h125       SECONDARY 25/125-step rollouts (reported, not in family)
      h25_escape, h125_escape  the same rollout frames restricted to escape

    The rollout uses ONE regime for every arm (protocol §3.2): feed the model's
    own predicted velocity/acceleration back into channels 2-3, THEN apply the
    arm's declared input transform (for R2, which declares channels 2-3
    ablated, the feedback write is followed by zeroing those channels).  This
    mirrors training, where the ablation is applied once to the dataset before
    the loaders, and keeps R2's h-slots available and comparable to R0/R1/R3.
    """
    cfg, ckpt, model, val_loader = _build_model_and_loader(
        checkpoint_path, dataset_path, nested_prior_path, device
    )
    zic = sorted(int(c) for c in ckpt.get("zero_input_channels", []))
    # The arm's declared input transform, applied AFTER the feedback write.
    ablate_channels = tuple(zic)
    onsets = _compute_onsets(dataset_path, nested_prior_path,
                             cfg.training.max_seq_len)
    model.eval()
    n = len(val_loader.dataset)
    if onsets.size != n:
        raise ValueError(
            f"onset count {onsets.size} != validation trials {n}"
        )
    m_mse = {s: np.full(n, np.nan) for s in ALL_SCOPES}
    p_mse = {s: np.full(n, np.nan) for s in ALL_SCOPES}
    z_mse = {s: np.full(n, np.nan) for s in ALL_SCOPES}
    counts = {s: np.zeros(n, dtype=int) for s in ALL_SCOPES}
    peak_err = np.full(n, np.nan)  # pred peak - true peak (cm/s), aligned frames
    true_peak = np.full(n, np.nan)  # observed peak (cm/s), aligned frames
    # Declared descriptive secondary: pooled model MSE over ALL valid frames
    # (t >= 0, padding excluded).  The aligned scopes are t >= 1 only.
    t0_mse = np.full(n, np.nan)
    t0_counts = np.zeros(n, dtype=int)
    # Pre-stimulus spontaneous-escape false-alarm accumulators (protocol §8.1,
    # F2).  Per trial: model-predicted escape frames/bouts and the animals' own
    # escape frames/bouts in the pre-stimulus segment [0, onset) inside the crop.
    pa = {
        "pred_frame_num": np.zeros(n, dtype=int), "pred_frame_den": np.zeros(n, dtype=int),
        "pred_trial_num": np.zeros(n, dtype=int), "pred_trial_den": np.zeros(n, dtype=int),
        "animal_frame_num": np.zeros(n, dtype=int), "animal_frame_den": np.zeros(n, dtype=int),
        "animal_trial_num": np.zeros(n, dtype=int), "animal_trial_den": np.zeros(n, dtype=int),
    }

    def _acc(scope: str, gi: int, m: float, p: float, z: float, c: int) -> None:
        if c <= 0:
            return
        m_mse[scope][gi] = m
        p_mse[scope][gi] = p
        z_mse[scope][gi] = z
        counts[scope][gi] = c

    row = 0
    with torch.no_grad():
        for batch in val_loader:
            x_b = batch[0].to(device).contiguous()
            y_b = batch[1].to(device).contiguous()
            len_b = batch[2].to(device).contiguous()
            B, T, F = x_b.shape
            assert x_b.shape == (B, T, 8), f"x shape {tuple(x_b.shape)}"
            assert y_b.shape == (B, T), f"y shape {tuple(y_b.shape)}"
            assert len_b.shape == (B,), f"lengths shape {tuple(len_b.shape)}"
            y_p, _ = model(x_b, len_b, return_internals=True)
            assert y_p.shape == (B, T), f"y_p {tuple(y_p.shape)}"
            for i in range(B):
                gi = row + i
                L = int(len_b[i])
                yt = y_b[i, :L].cpu().numpy().astype(np.float64)
                yp = y_p[i, :L].cpu().numpy().astype(np.float64)
                assert yt.size == L
                # B3 (fail closed): the loader's own targets and the canonical
                # extract_canonical_validation_targets series must be ONE series
                # (the aligned scopes score the loader's y, the rollout/onset
                # scopes score the canonical y_true_seqs).
                _require_loader_targets_match(yt, y_true_seqs[gi], gi)

                # aligned_pooled (t >= 1)
                if L >= 2:
                    se_m = (yp[1:] - yt[1:]) ** 2
                    se_p = (yt[:-1] - yt[1:]) ** 2
                    se_z = yt[1:] ** 2
                    _acc("aligned_pooled", gi,
                         float(se_m.mean()), float(se_p.mean()),
                         float(se_z.mean()), se_m.size)
                    # peak-speed error on aligned frames (descriptive)
                    peak_err[gi] = float(yp[1:].max() - yt[1:].max())
                    true_peak[gi] = float(yt[1:].max())

                # pooled t >= 0 (ALL valid frames, descriptive secondary)
                if L >= 1:
                    t0_mse[gi] = float(((yp - yt) ** 2).mean())
                    t0_counts[gi] = L

                # escape (t >= 1, |y_true| >= 10)
                if L >= 2:
                    mask = np.abs(yt[1:]) >= ESCAPE_THRESHOLD_CM_S
                    if mask.any():
                        _acc("escape", gi,
                             float(((yp[1:] - yt[1:]) ** 2)[mask].mean()),
                             float(((yt[:-1] - yt[1:]) ** 2)[mask].mean()),
                             float((yt[1:] ** 2)[mask].mean()),
                             int(mask.sum()))

                # onset window (aligned frames t in [onset, r+200ms), t >= 1)
                onset = int(onsets[gi])
                r = _response_onset(yt, onset)
                # Pre-stimulus spontaneous-escape false-alarm (protocol §8.1).
                if onset > 0:
                    (
                        pa["pred_frame_num"][gi], pa["pred_frame_den"][gi],
                        pa["pred_trial_num"][gi], pa["pred_trial_den"][gi],
                        pa["animal_frame_num"][gi], pa["animal_frame_den"][gi],
                        pa["animal_trial_num"][gi], pa["animal_trial_den"][gi],
                    ) = _prestim_false_alarm(yp, yt, onset)
                if r is not None:
                    k = int(round(ONSET_WIN_MS / DT_MS))
                    t0 = max(onset, 1)
                    hi = min(r + k, L)
                    if hi > t0:
                        idx = np.arange(t0, hi)
                        _acc("onset", gi,
                             float(((yp[idx] - yt[idx]) ** 2).mean()),
                             float(((yt[idx - 1] - yt[idx]) ** 2).mean()),
                             float((yt[idx] ** 2).mean()),
                             int(idx.size))
            row += B

            # ── multi-step own-prediction-feedback rollouts ──
            for horizon, scope in [(PRIMARY_HORIZON, "h5")] + [
                (h, f"h{h}") for h in SECONDARY_HORIZONS
            ]:
                _rollout_hN(model, x_b, y_true_seqs, device, row - B,
                            m_mse, p_mse, z_mse, counts, horizon, scope,
                            ablate_channels)
    if row != n:
        raise ValueError(
            f"loader yielded {row} trials but the canonical split has {n}"
        )
    return ({"model": m_mse, "persistence": p_mse, "zero": z_mse},
            counts, peak_err, true_peak, t0_mse, t0_counts,
            {"zero_input_channels": zic,
             "rollout_regime": "feedback_then_arm_transform"}, pa)


def _hN_trial(
    model: torch.nn.Module, x_i: torch.Tensor, yt: np.ndarray,
    device: torch.device, horizon: int,
    ablate_channels: Sequence[int],
) -> Tuple[float, float, float, int, float, float, float, int]:
    """N-step own-prediction-feedback rollout for one trial.

    Returns per-trial ``(mse_model, mse_persistence, mse_zero, count)`` for the
    **overall** rollout frames followed by the same triple/count for the
    **escape** subset (frames with ``|y_true| >= 10 cm/s``), each averaged over
    up to ``MULTISTEP_STARTS`` evenly-spaced start frames.

    ONE rollout regime for every arm (protocol §3.2): each step feeds the
    previous prediction back into channel 2 (velocity) and 3 (acceleration =
    ``(v_new - v_prev)/dt``), **then** applies the arm's declared input
    transform (``ablate_channels`` forced to 0.0) — exactly as in training.
    Stimulus channels 0,1,4:8 are copied from the source trial's frame
    ``start+k``.  Persistence is held at ``y_true[start-1]`` over the whole
    horizon (the step-1 definition).
    """
    L = int(yt.size)
    dt_s = DT_MS / 1000.0
    starts = np.unique(
        np.linspace(1, max(1, L - horizon), MULTISTEP_STARTS).astype(int)
    )
    starts = starts[(starts >= 1) & (starts + horizon - 1 < L)]
    if starts.size == 0:
        return (np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, 0)
    se_m = se_p = se_z = 0.0
    cnt = 0
    se_m_e = se_p_e = se_z_e = 0.0
    cnt_e = 0
    x_i = x_i.to(device)
    for s in starts:
        with torch.no_grad():
            _y0, _in0, states = model(
                x_i[:s].unsqueeze(0).contiguous(),
                torch.tensor([s], dtype=torch.int64, device=device),
                return_internals=True, states={},
            )
        v_fb = float(x_i[s, 2].item())
        a_fb = float(x_i[s, 3].item())
        p_hold = float(yt[s - 1])
        for k in range(horizon):
            frame = min(s + k, L - 1)
            xs = x_i[frame].clone()
            # 1) feed the model's own prediction back into channels 2-3 ...
            xs[2] = v_fb
            xs[3] = a_fb
            # 2) ... then apply the arm's declared input transform.
            for ch in ablate_channels:
                xs[ch] = 0.0
            with torch.no_grad():
                y_step, _in, states = model(
                    xs.unsqueeze(0).unsqueeze(0).contiguous(),
                    torch.tensor([1], dtype=torch.int64, device=device),
                    return_internals=True, states=states,
                )
            v_new = float(y_step[0, 0].item())
            tgt = float(yt[frame])
            se_m += (v_new - tgt) ** 2
            se_p += (p_hold - tgt) ** 2
            se_z += tgt ** 2
            cnt += 1
            if abs(tgt) >= ESCAPE_THRESHOLD_CM_S:
                se_m_e += (v_new - tgt) ** 2
                se_p_e += (p_hold - tgt) ** 2
                se_z_e += tgt ** 2
                cnt_e += 1
            a_fb = (v_new - v_fb) / dt_s
            v_fb = v_new
    if cnt == 0:
        return (np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, 0)
    m_e, p_e, z_e = (
        (se_m_e / cnt_e, se_p_e / cnt_e, se_z_e / cnt_e)
        if cnt_e > 0 else (np.nan, np.nan, np.nan)
    )
    return (se_m / cnt, se_p / cnt, se_z / cnt, cnt, m_e, p_e, z_e, cnt_e)


def _rollout_hN(
    model: torch.nn.Module, x_b: torch.Tensor,
    y_true_seqs: Sequence[np.ndarray],
    device: torch.device, base: int,
    m_mse: Dict[str, np.ndarray], p_mse: Dict[str, np.ndarray],
    z_mse: Dict[str, np.ndarray], counts: Dict[str, np.ndarray],
    horizon: int, scope: str, ablate_channels: Sequence[int],
) -> None:
    """Fill the per-trial N-step rollout MSE for one batch into the scope arrays.

    Writes both the overall ``scope`` series and its ``scope_escape`` companion
    (the same (start, k) frames restricted to ``|y_true| >= 10 cm/s``), so an
    escape ceiling is only ever read against an escape-frame scalar (M1).
    """
    B = x_b.shape[0]
    for i in range(B):
        gi = base + i
        yt = np.asarray(y_true_seqs[gi], dtype=np.float64)
        m, p, z, c, m_e, p_e, z_e, c_e = _hN_trial(
            model, x_b[i], yt, device, horizon, ablate_channels
        )
        if c > 0:
            m_mse[scope][gi] = m
            p_mse[scope][gi] = p
            z_mse[scope][gi] = z
            counts[scope][gi] = c
        if c_e > 0:
            m_mse[f"{scope}_escape"][gi] = m_e
            p_mse[f"{scope}_escape"][gi] = p_e
            z_mse[f"{scope}_escape"][gi] = z_e
            counts[f"{scope}_escape"][gi] = c_e


def _holm(pvals: Mapping[str, Optional[float]]) -> Dict[str, Any]:
    """Holm-Bonferroni over the FIXED declared family.

    Reuses the frozen :func:`nsmor.analysis.uq.holm_bonferroni` primitive over
    the full declared slot set (m = 48): every unavailable slot is carried as
    ``p = 1`` for bookkeeping (protocol §6), so the first threshold is
    ``0.05 / 48``.  Unavailable slots are reported with null ``adjusted_p`` /
    ``significant`` (never a fabricated ``False``).
    """
    slots = declared_family_slots()
    if set(pvals) != set(slots):
        raise ValueError("p-value keys must equal the declared slots")
    internal: Dict[str, float] = {
        s: (1.0 if pvals[s] is None else float(pvals[s])) for s in slots
    }
    adjusted = holm_bonferroni(internal)
    out: Dict[str, Any] = {}
    unavailable = 0
    for slot in slots:
        if pvals[slot] is None:
            unavailable += 1
            out[slot] = {"available": False, "raw_p": None,
                         "adjusted_p": None, "significant": None}
        else:
            adj, sig = adjusted[slot]
            out[slot] = {"available": True, "raw_p": float(pvals[slot]),
                         "adjusted_p": float(adj), "significant": bool(sig)}
    return {
        "family_size": len(slots),
        "valid_test_count": len(slots) - unavailable,
        "unavailable_slot_count": unavailable,
        "slots": out,
    }


def _pooled_mse(mse_vec: np.ndarray, frame_counts: np.ndarray) -> float:
    """Frame-weighted pooled MSE over trials with finite values and >0 frames."""
    valid = (
        np.isfinite(mse_vec) & np.isfinite(frame_counts) & (frame_counts > 0)
    )
    den = float(frame_counts[valid].sum())
    if den <= 0.0:
        return float("nan")
    return float((mse_vec[valid] * frame_counts[valid]).sum() / den)


def _reconcile_ceiling(
    measured_mse: float, frozen_mse: float, rel_tol: float = 1e-3,
) -> Dict[str, Any]:
    """M3: compare a freshly measured persistence MSE to the frozen ceiling.

    Returns a JSON-safe record with the frozen/measured values, the signed
    ``ceiling_vs_measured_delta`` and a ``matches`` flag.  The caller fails
    closed on ``matches is False`` — a stale ceiling would make the reported
    skill read against a comparator the scored data no longer matches.
    """
    delta = (
        float(measured_mse) - float(frozen_mse)
        if math.isfinite(measured_mse) else float("nan")
    )
    ok = bool(
        math.isfinite(measured_mse)
        and math.isclose(measured_mse, frozen_mse, rel_tol=rel_tol, abs_tol=0.0)
    )
    return {
        "frozen_mse": float(frozen_mse),
        "measured_mse": float(measured_mse),
        "ceiling_vs_measured_delta": delta,
        "matches": ok,
    }


def _skill_vs_persistence(
    model_mse: np.ndarray, persist_mse: np.ndarray, frame_counts: np.ndarray,
) -> Dict[str, Any]:
    """Primary scalar: ``1 - MSE_model / MSE_persist`` (pooled over frames)."""
    m = _pooled_mse(model_mse, frame_counts)
    p = _pooled_mse(persist_mse, frame_counts)
    if not (math.isfinite(m) and math.isfinite(p)) or p <= 0.0:
        return {"skill_vs_persistence": None, "mse_model": m,
                "mse_persistence": p, "reason": "non_finite_or_zero_persistence"}
    return {"skill_vs_persistence": float(1.0 - m / p),
            "mse_model": m, "mse_persistence": p, "reason": None}


def _json_array(values: np.ndarray) -> List[Optional[float]]:
    """JSON-safe list: non-finite entries become ``None`` (never ``NaN``)."""
    return [None if not math.isfinite(float(v)) else float(v) for v in values]


def _seed_spread(seed_arrays: List[np.ndarray]) -> Dict[str, Any]:
    """Per-trial seed spread of a per-seed per-trial MSE series.

    ``per_seed_trial_mean`` is the **unweighted** mean over the per-trial MSE
    vector (one trial = one observation), named to match what it is — not the
    frame-weighted pooled mean (which ``_pooled_mse`` reports separately).
    """
    stack = np.vstack(seed_arrays)
    assert stack.ndim == 2
    finite = np.isfinite(stack)
    per_trial_std = np.where(finite.all(axis=0), stack.std(axis=0, ddof=1), np.nan)
    return {
        "n_seeds": int(stack.shape[0]),
        "per_seed_trial_mean": [
            float(np.nanmean(stack[i])) for i in range(stack.shape[0])
        ],
        "per_trial_std_mean": float(np.nanmean(per_trial_std)),
        "per_trial_std_max": float(np.nanmax(per_trial_std)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm.lower()}", type=Path, nargs=3, required=True,
                            metavar=("S42", "S43", "S44"),
                            help=f"{arm} best_model.pth for seeds 42/43/44.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--nested_prior", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--exchangeability_asserted", action="store_true",
                        help="Assert the joint sign-exchangeability null.")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true_seqs, trial_ids, prefix_ids = extract_canonical_validation_targets(
        args.dataset, args.nested_prior
    )
    n = len(y_true_seqs)
    assert len(trial_ids) == len(prefix_ids) == n

    # Per-arm seed-mean per-trial MSE (model) and comparators.
    arm_model: Dict[str, Dict[str, np.ndarray]] = {}
    arm_seed_mse: Dict[str, Dict[str, List[np.ndarray]]] = {}
    arm_prov: Dict[str, Any] = {}
    comps: Dict[str, Dict[str, np.ndarray]] = {}
    peak: Dict[str, Dict[str, Any]] = {}
    frame_counts: Dict[str, np.ndarray] = {}
    h5_protocol: Dict[str, Any] = {}
    t0_mse: Dict[str, np.ndarray] = {}
    t0_counts: Dict[str, np.ndarray] = {}
    prestim: Dict[str, Dict[str, float]] = {}
    true_peak_ref: Optional[np.ndarray] = None
    for arm in ARMS:
        paths = getattr(args, arm.lower())
        acc = {s: [] for s in ALL_SCOPES}
        peak_seed = []
        t0_seed = []
        t0_cnt_seed = []
        pa_seed: List[Dict[str, np.ndarray]] = []
        counts_ref: Optional[Dict[str, np.ndarray]] = None
        for p in paths:
            ckpt = load_artifact_bytes(p.read_bytes(), map_location="cpu")
            _validate_arm_config(ckpt, arm)
            m, counts, peak_err, true_peak, t0m, t0c, meta, pa = (
                _per_trial_scope_mse(
                    p, args.dataset, args.nested_prior, device, y_true_seqs
                )
            )
            for s in ALL_SCOPES:
                acc[s].append(m["model"][s])
            peak_seed.append(peak_err)
            t0_seed.append(t0m)
            t0_cnt_seed.append(t0c)
            pa_seed.append(pa)
            # The true peak is model-independent; capture it once (identical
            # across arms/seeds) so ``observed_peak_cm_s`` is the SCORED
            # split's own value, not the step-1 event-level constant.
            if true_peak_ref is None:
                true_peak_ref = true_peak
            comps.setdefault("persistence",
                             {s: m["persistence"][s] for s in ALL_SCOPES})
            comps.setdefault("zero", {s: m["zero"][s] for s in ALL_SCOPES})
            counts_ref = counts
            h5_protocol[arm] = {
                "zero_input_channels": meta["zero_input_channels"],
                "rollout_regime": meta["rollout_regime"],
            }
        arm_model[arm] = {
            s: np.nanmean(np.vstack(acc[s]), axis=0) for s in ALL_SCOPES
        }
        arm_seed_mse[arm] = acc
        # Pre-stimulus false-alarm rates (protocol §8.1): frame rate = pooled
        # predicted-escape frames / pooled below-threshold frames; trial rate =
        # pooled predicted-bout trials / pre-stimulus trials.  Both for the
        # model and for the animals' own pre-stimulus escapes.
        def _rate(num: str, den: str) -> Optional[float]:
            nsum = sum(int(p_[num].sum()) for p_ in pa_seed)
            dsum = sum(int(p_[den].sum()) for p_ in pa_seed)
            return float(nsum / dsum) if dsum > 0 else None

        prestim[arm] = {
            "model_prestim_false_alarm_frame_rate": _rate(
                "pred_frame_num", "pred_frame_den"),
            "model_prestim_false_alarm_trial_rate": _rate(
                "pred_trial_num", "pred_trial_den"),
            "animal_prestim_escape_frame_rate": _rate(
                "animal_frame_num", "animal_frame_den"),
            "animal_prestim_escape_trial_rate": _rate(
                "animal_trial_num", "animal_trial_den"),
            "n_prestim_frames": sum(
                int(p_["pred_frame_den"].sum()) for p_ in pa_seed),
            "n_prestim_trials": sum(
                int(p_["pred_trial_den"].sum()) for p_ in pa_seed),
        }
        frame_counts = counts_ref if counts_ref is not None else frame_counts
        t0_mse[arm] = np.nanmean(np.vstack(t0_seed), axis=0)
        t0_counts[arm] = t0_cnt_seed[0]  # frame counts are model-independent
        peak_stack = np.vstack(peak_seed)
        peak[arm] = {
            "mean_pred_minus_true_cm_s": float(np.nanmean(peak_stack)),
            "observed_peak_cm_s": (
                float(np.nanmean(true_peak_ref))
                if true_peak_ref is not None else None
            ),
            "observed_peak_provenance": "scored_split_aligned_frames",
            "step1_event_level_observed_peak_cm_s": PEAK_SPEED_OBSERVED_CM_S,
            "step1_event_level_provenance": PEAK_SPEED_OBSERVED_PROVENANCE,
            "per_seed_mean_pred_minus_true_cm_s": [
                float(np.nanmean(peak_stack[i])) for i in range(peak_stack.shape[0])
            ],
        }
        arm_prov[arm] = {"checkpoints": [str(p) for p in paths],
                         "sha256": [_sha256(p) for p in paths]}

    # Primary skill-vs-persistence per arm per scope, against the ceiling table.
    # The horizon escape variants are emitted too (M1): an escape ceiling is
    # only ever attached to an escape-frame scalar, never to an overall one.
    skill: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        skill[arm] = {}
        for scope in HORIZON_SCOPES:
            entry = _skill_vs_persistence(
                arm_model[arm][scope], comps["persistence"][scope],
                frame_counts.get(scope, np.zeros(n, dtype=int)),
            )
            entry["persistence_ceiling"] = PERSISTENCE_CEILING.get(scope)
            skill[arm][scope] = entry
            esc = f"{scope}_escape"
            entry_e = _skill_vs_persistence(
                arm_model[arm][esc], comps["persistence"][esc],
                frame_counts.get(esc, np.zeros(n, dtype=int)),
            )
            entry_e["persistence_ceiling"] = PERSISTENCE_CEILING.get(esc)
            skill[arm][esc] = entry_e
        for scope in ("aligned_pooled", "escape", "onset"):
            entry = _skill_vs_persistence(
                arm_model[arm][scope], comps["persistence"][scope],
                frame_counts.get(scope, np.zeros(n, dtype=int)),
            )
            entry["persistence_ceiling"] = PERSISTENCE_CEILING.get(scope)
            skill[arm][scope] = entry

    # Secondary metrics: h25 / h125 skill (overall AND escape), peak-speed
    # error, and the declared descriptive pooled t>=0 MSE.
    secondary: Dict[str, Any] = {}
    for arm in ARMS:
        horizons: Dict[str, Any] = {}
        for scope in HORIZON_SCOPES:
            horizons[scope] = {
                **skill[arm][scope],
                "persistence_ceiling_escape": PERSISTENCE_CEILING.get(
                    f"{scope}_escape"),
            }
            horizons[f"{scope}_escape"] = dict(skill[arm][f"{scope}_escape"])
        secondary[arm] = {
            "peak_speed_error": peak[arm],
            "horizons": horizons,
            "pooled_mse_t_ge_0": {
                "mse_model": _pooled_mse(t0_mse[arm], t0_counts[arm]),
                "n_frames": int(t0_counts[arm].sum()),
            },
            "prestim_spontaneous_escape": prestim[arm],
        }

    # Seed-level summary (the seed mean is the inferential unit) plus the raw
    # per-seed per-trial MSE vectors the protocol declares are retained.
    seed_level: Dict[str, Any] = {
        arm: {s: _seed_spread(arm_seed_mse[arm][s]) for s in ALL_SCOPES}
        for arm in ARMS
    }
    seed_trial_vectors: Dict[str, Any] = {
        arm: {
            s: {"seeds": [42, 43, 44],
                "per_seed_per_trial_mse": [
                    _json_array(arr) for arr in arm_seed_mse[arm][s]
                ]}
            for s in ALL_SCOPES
        }
        for arm in ARMS
    }

    # Assemble the declared family.
    p_values: Dict[str, Optional[float]] = {}
    slots_out: Dict[str, Any] = {}
    for slot in declared_family_slots():
        cand, comp, scope = slot.split("|")
        cand_vec = arm_model[cand][scope]
        comp_vec = arm_model[comp][scope] if comp == "R0" else comps[comp][scope]
        eligible = np.isfinite(cand_vec) & np.isfinite(comp_vec)
        fc = frame_counts.get(scope, np.zeros(n, dtype=int))
        if comp == cand:
            p_values[slot] = None
            slots_out[slot] = {"available": False, "reason": "self_comparison"}
            continue
        if eligible.sum() < 2 or int(fc[eligible].sum()) == 0:
            p_values[slot] = None
            slots_out[slot] = {
                "available": False,
                "reason": "fewer_than_two_trials" if eligible.sum() < 2
                else "zero_eligible_frames",
            }
            continue
        res = paired_mse_comparison(
            candidate_mse=cand_vec[eligible],
            comparator_mse=comp_vec[eligible],
            trial_ids=[trial_ids[i] for i in np.where(eligible)[0]],
            prefix_ids=[prefix_ids[i] for i in np.where(eligible)[0]],
            frame_counts=fc[eligible].tolist(),
            exchangeability_asserted=bool(args.exchangeability_asserted),
        )
        sign_flip = res["sign_flip"]
        p_val = sign_flip["p_value"]
        p_values[slot] = p_val
        # Derive availability from the PRIMITIVE, not from the fact that we
        # called it: ``paired_mse_comparison`` returns a null p with a reason
        # (e.g. ``fewer_than_six_prefixes`` / ``exchangeability_not_asserted``)
        # even on a successful call, and ``_holm`` marks such slots
        # ``available=False``.  Reporting ``available=True`` here would make
        # the SAME payload contradict itself (R2 finding).
        slots_out[slot] = {
            "available": p_val is not None,
            "reason": None if p_val is not None else sign_flip.get("reason"),
            **res,
        }

    holm = _holm(p_values)
    # M3: reconcile the freshly measured persistence MSE against the frozen
    # ceiling per scope (rel 1e-3).  A stale ceiling means the reported skill
    # is read against a comparator the scored data no longer matches, so fail
    # closed; the per-scope deltas are emitted for audit either way.
    ceiling_reconciliation: Dict[str, Any] = {}
    ceiling_mismatches: List[str] = []
    for scope, frozen in PERSISTENCE_CEILING.items():
        measured = _pooled_mse(
            comps["persistence"][scope],
            frame_counts.get(scope, np.zeros(n, dtype=int)),
        )
        rec = _reconcile_ceiling(measured, float(frozen["mse"]))
        ceiling_reconciliation[scope] = rec
        if not rec["matches"]:
            ceiling_mismatches.append(scope)
    if ceiling_mismatches:
        raise ValueError(
            "frozen persistence ceiling does not match the freshly measured "
            f"persistence MSE (rel 1e-3) for scope(s) {ceiling_mismatches}; "
            "the reported skill would be read against a stale comparator "
            "(fail closed)"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "docs/realdata-phase2-protocol-20261010.md",
        "declared_family_size": FAMILY_SIZE,
        "candidates": list(ARMS),
        "comparators": list(COMPARATORS),
        "scopes": list(SCOPES),
        "secondary_scopes": list(SECONDARY_SCOPES),
        "inferential_unit": "seed_mean_over_3_seeds",
        "exchangeability_asserted": bool(args.exchangeability_asserted),
        "scorer_command": " ".join(sys.argv),
        "h5_input_protocol": h5_protocol,
        "persistence_ceiling": PERSISTENCE_CEILING,
        "ceiling_reconciliation": ceiling_reconciliation,
        "pure_lag_ceiling_reference_only": PURE_LAG_CEILING,
        "arms": arm_prov,
        "primary_skill_vs_persistence": skill,
        "secondary_metrics": secondary,
        "seed_level_summary": seed_level,
        "seed_trial_mse_vectors": seed_trial_vectors,
        "declared_family": {
            "family_size": FAMILY_SIZE,
            "exchangeability_asserted": bool(args.exchangeability_asserted),
            "holm_bonferroni": holm,
            "slots": slots_out,
        },
    }
    out = args.output_dir / "realdata_phase2_family.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %s (family=%d, valid=%d, unavailable=%d)",
        out, FAMILY_SIZE, holm["valid_test_count"],
        holm["unavailable_slot_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
