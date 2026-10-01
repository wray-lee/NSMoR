#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 10 — In-Silico Psychophysics: Routing-Gate Noise Sensitivity.

Injects graded Gaussian noise into the visual input channel and quantifies
the resulting shifts in MoR routing gates and kinematic latencies.

Key outputs:
    results/bayesian_reliability.png  — Dual-panel Lancet/Cell figure
    results/psychophysics_summary.json — Aggregated statistics

Round-1 fix (Reviewer A MAJOR-2): the MCMC prior columns
(X[:, :, 4:7]) are held FIXED across noise levels — only the visual
channel degrades.  The router's response therefore demonstrates
*input-noise sensitivity of the routing gate*, NOT Bayesian cue
re-weighting (which would require the prior to degrade with evidence
reliability).  Claims are scoped accordingly.

Noise definition (SNR): sigma is in DEGREES of visual angle and is
injected ADDITIVELY onto the raw visual-angle channel before sensory
encoding.  The per-trial signal scale is the peak |visual_angle|
excursion; SNR_dB = 20*log10(peak|angle| / sigma) is reported per
condition so the physical meaning of each sigma is explicit.

Hypothesis (scoped):
    Higher visual noise → delayed/suppressed GRU gating and systematic
    latency increase, consistent with reliability-dependent routing.
    This is NECESSARY but not SUFFICIENT evidence for optimal cue
    combination; a causal test would additionally require perturbing
    prior reliability.

Respects all BOUNDARY.md constraints — never modifies frozen core.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Bootstrap: resolve paths, import project modules
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from nsmor.analysis.analysis_priors import load_analysis_priors
from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint
from nsmor.analysis.prediction_units import resolve_dt_ms
from nsmor.analysis.prediction_units import load_model_from_checkpoint  # noqa: E402
from nsmor.pipeline.events import load_declared_events_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lancet/Cell colour palette (strict Phase 9 aesthetic)
# ---------------------------------------------------------------------------
# Panel A: monochromatic gradient — dark (σ=0) → light (high σ)
GATE_COLOURS = [
    "#1C7ED6",  # σ=0.0   Cell Cobalt Blue (saturated)
    "#4DABF7",  # σ=5.0   lighter
    "#A5D8FF",  # σ=15.0  pastel
    "#D0EBFF",  # σ=30.0  very faded
]
LATENCY_COLOUR = "#C92A2A"  # Lancet Crimson Red
BASELINE_COLOUR = "#495057"  # Strong Slate Gray
AXIS_COLOUR = "#212529"  # Dark charcoal
BG_COLOUR = "#FFFFFF"
LINEWIDTH = 1.5
DPI = 300


# ===================================================================
# Data Loading (reuse logic from analyze_integration.py)
# ===================================================================

def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load model checkpoint with ALL biophysical parameters.

    Delegates to the shared :func:
smor.analysis.prediction_units.load_model_from_checkpoint`
    which guarantees every biophysical parameter is reconstructed from
    the saved config (refractory periods, synaptic delay, STP, lateral
    inhibition, dendritic compartmentalization, neuromodulatory gain,
    sensory noise).  The original local implementation only forwarded
    8 of 21 parameters, silently using defaults for the rest.
    """
    return load_model_from_checkpoint(Path(ckpt_path), device)


STIM_ONSET_FRAME = 1200
NOISE_LEVELS = [0.0, 5.0, 15.0, 30.0]  # σ in degrees
LATENCY_SCOPE = (
    "Signed latency to the largest absolute peak in [-2s, +4s) relative to onset; "
    "negative latencies include pre-collision escapes. Flat, nonfinite, and "
    "empty windows are excluded as NaN. Peak velocity uses the same window."
)


@dataclass
class TrialMeta:
    """Explicit per-trial metadata that travels beside tensors.

    Survives slicing/filtering via :meth:`subset`, unlike ad-hoc tensor
    attributes which PyTorch strips on ``X[mask]``.
    """

    val_indices: Optional[List[int]] = None
    session_ids: Optional[List[str]] = None
    trial_ids: Optional[List[int]] = None
    target_ttc_ms: Optional[List[Optional[float]]] = None
    stimulus_conditions: Optional[List[str]] = None
    anchor_frames: Optional[List[int]] = None
    stim_onset_frame: int = STIM_ONSET_FRAME
    pre_anchor_frames: int = 1200

    def subset(self, indices: Sequence[int]) -> "TrialMeta":
        """Return a new TrialMeta containing only the given row indices."""
        def _pick(seq):
            return [seq[i] for i in indices] if seq is not None else None
        return TrialMeta(
            val_indices=_pick(self.val_indices),
            session_ids=_pick(self.session_ids),
            trial_ids=_pick(self.trial_ids),
            target_ttc_ms=_pick(self.target_ttc_ms),
            stimulus_conditions=_pick(self.stimulus_conditions),
            anchor_frames=_pick(self.anchor_frames),
            stim_onset_frame=self.stim_onset_frame,
            pre_anchor_frames=self.pre_anchor_frames,
        )


class ValidationData:
    """Tensor bundle with explicit side-by-side metadata.

    Unpacks as ``(X, Y, lengths)`` for backward compatibility with
    callers that expect a 3-tuple; access ``.meta`` for TrialMeta.
    """

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        lengths: torch.Tensor,
        meta: TrialMeta,
    ):
        self.X = X
        self.Y = Y
        self.lengths = lengths
        self.meta = meta

    def __iter__(self):
        return iter((self.X, self.Y, self.lengths))

    def __len__(self):
        return 3


def load_validation_data(
    device: torch.device,
    max_seq_len: Optional[int] = 2400,
    pre_anchor_frames: int = 1200,
    dataset_path: Optional[str] = None,
    val_split: float = 0.2,
    random_seed: int = 42,
    nested_prior_artifact: Optional[Path] = None,
    qc_sealed_nested_prior_sha256: Optional[str] = None,
    checkpoint_model=None,
    trusted_historical_checkpoint_sha256: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
):
    """
    Load `
smor_dataset.pt`` and return the validation split.

    ``dataset_path`` defaults to the in-repo processed dataset so existing
    callers keep working, but the pipeline passes the dataset the current
    run produced — otherwise this stage silently scores whatever file is
    left over in ``data/processed`` from an earlier run.

    Uses the recording-prefix grouped train/val split shared with train.py.
    Distinct prefixes do not establish independent animal identities.

    Returns a :class:`ValidationData` that unpacks as ``(X, Y, lengths)``
    and carries explicit :class:`TrialMeta` via ``.meta``.
    """
    if dataset_path is None:
        dataset_path = os.path.join(
            _PROJECT_ROOT, "data", "processed", "nsmor_dataset.pt"
        )
    if not os.path.exists(dataset_path):
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    data, loaded_source_fingerprint = load_dataset_with_fingerprint(
        dataset_path, map_location="cpu", restore_provenance=False,
        expected_dt_ms=(
            resolve_dt_ms(checkpoint_model) if checkpoint_model is not None else None
        ),
    )
    # Round-2 CRITICAL-A: refuse pre-2.0 datasets (leaked priors)
    from nsmor.model_utils import resolve_dataset_session_ids, validate_dataset_provenance
    validate_dataset_provenance(data, Path(dataset_path))
    session_ids = resolve_dataset_session_ids(data)
    X_seqs = data["X_seqs"]
    Y_seqs = data["Y_seqs"]
    lengths = data["lengths"]
    mcmc_priors, persisted_val_indices = load_analysis_priors(
        data, Path(dataset_path), nested_prior_artifact, checkpoint_model, qc_sealed_nested_prior_sha256,
        loaded_source_fingerprint=loaded_source_fingerprint,
        trusted_historical_checkpoint_sha256=trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=trusted_historical_artifact_sha256,
    )

    n_total = len(X_seqs)
    from nsmor.pipeline.grouping import grouped_train_val_split

    if persisted_val_indices is None:
        _, val_indices = grouped_train_val_split(
            session_ids,
            n_total,
            val_split=val_split,
            random_seed=random_seed,
        )
    else:
        val_indices = persisted_val_indices  # Exact checkpoint outer hold-out.

    anchor_frames = data.get("anchor_frames")
    if anchor_frames is None:
        from nsmor.pipeline.conditions import derive_anchor_frames
        anchor_frames = derive_anchor_frames(X_seqs, lengths)
        logger.info(
            "Derived %d anchor frames from physical channels.",
            len(anchor_frames),
        )

    val_anchor_frames_pre_crop = [int(anchor_frames[i]) for i in val_indices]
    val_raw_lengths = [
        int(lengths[i].item()) if torch.is_tensor(lengths) else int(lengths[i])
        for i in val_indices
    ]

    # Use DataLoader with collate_variable_length for proper padding
    from nsmor.nsmor_dataloader import NSMoRDataset
    from nsmor.dataloader_factory import create_optimized_dataloader
    from nsmor.config import DEFAULT_FEATURE

    sequences = [(X_seqs[i], Y_seqs[i], 0) for i in val_indices]
    feature_config = data.get("feature_config", DEFAULT_FEATURE)
    val_priors = mcmc_priors[val_indices] if mcmc_priors is not None else None

    val_dataset = NSMoRDataset(
        sequences=sequences,
        mcmc_priors=(
            val_priors
            if val_priors is not None
            else np.ones((len(sequences), 4)) * 0.25
        ),
        feature_config=feature_config,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        anchor_frames=val_anchor_frames_pre_crop,
    )

    # Create a single batch with all validation data
    val_loader = create_optimized_dataloader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=0,  # Whole corpus is already resident; workers only replicate it
    )

    X_val, Y_val, lengths_val = next(iter(val_loader))

    X_val = X_val.to(device).contiguous()
    Y_val = Y_val.to(device).contiguous()
    lengths_val = lengths_val.to(device).contiguous()

    # Map pre-crop metadata anchors into the coordinate frame of the
    # SAVED (post-crop) tensors using the exact same crop window that
    # NSMoRDataset.__getitem__ applied.  Without this, extract_gate_trajectory
    # shifts by a pre-crop anchor against a post-crop sequence and the
    # aligned peak lands at the wrong index (or is dropped entirely).
    from nsmor.pipeline.conditions import resolve_anchor_crop

    val_anchor_frames: List[int] = []
    for i, (raw_len, anchor_pre) in enumerate(
        zip(val_raw_lengths, val_anchor_frames_pre_crop)
    ):
        start, _end = resolve_anchor_crop(
            n_frames=raw_len,
            anchor_frame=anchor_pre,
            max_seq_len=max_seq_len,
            pre_anchor_frames=pre_anchor_frames,
        )
        cropped_anchor = anchor_pre - start
        # Keep anchors inside the saved sequence; fall back to the
        # in-window stimulus position rather than leaking an OOB index.
        saved_len = int(
            lengths_val[i].item() if torch.is_tensor(lengths_val) else lengths_val[i]
        )
        if not (0 <= cropped_anchor < max(saved_len, 1)):
            cropped_anchor = min(max(0, cropped_anchor), max(saved_len - 1, 0))
        val_anchor_frames.append(int(cropped_anchor))

    # Explicit metadata travels beside tensors (survives X_val[mask] slicing,
    # unlike ad-hoc tensor attributes which PyTorch strips).
    trial_ids = data.get("trial_ids")
    target_ttc_ms = data.get("target_ttc_ms")
    stimulus_conditions = data.get("stimulus_conditions")
    meta = TrialMeta(
        val_indices=list(val_indices),
        session_ids=[session_ids[i] for i in val_indices] if session_ids is not None else None,
        trial_ids=[trial_ids[i] for i in val_indices] if trial_ids is not None else None,
        target_ttc_ms=[target_ttc_ms[i] for i in val_indices] if target_ttc_ms is not None else None,
        stimulus_conditions=(
            [stimulus_conditions[i] for i in val_indices]
            if stimulus_conditions is not None else None
        ),
        anchor_frames=list(val_anchor_frames),
        stim_onset_frame=pre_anchor_frames,
        pre_anchor_frames=pre_anchor_frames,
    )

    logger.info("Validation data loaded: %d trials", X_val.shape[0])
    return ValidationData(X_val, Y_val, lengths_val, meta)


# ===================================================================
# Condition Filtering & Noise Injection
# ===================================================================


def detect_wind_onset_frame(x_seq: torch.Tensor) -> int | None:
    """Return first frame index where wind(t) > 0.5, or None."""
    wind_channel = x_seq[:, 1]
    indices = (wind_channel > 0.5).nonzero(as_tuple=False)
    if indices.numel() == 0:
        return None
    return int(indices[0].item())


class TTC0Selection(NamedTuple):
    """Result of declared multisensory TTC=0 selection.

    Attributes:
        mask: Boolean mask over trials (empty when status is not_applicable).
        n_candidates: Multisensory trials with resolvable declared TTC.
        n_ttc0: Count of declared TTC=0 trials.
        status: ``"ok"`` when n_ttc0 > 0, else ``"not_applicable"``.
    """

    mask: torch.Tensor
    n_candidates: int
    n_ttc0: int
    status: str


def psychophysics_gate_verdict(output_dir: str | Path) -> str:
    """Return the Phase-G artefact gate verdict for ``output_dir``.

    Reads the FRESH ``bayesian_reliability.json`` status written by the
    current run. Stale PNG/summary files alone can never satisfy the gate
    because the status artefact is required first.

    Returns:
        ``"not_applicable"`` — legitimate empty TTC=0 subset, no figure required.
        ``"unavailable_latency"`` — TTC=0 present, no measurable latency; PNG + summary required.
        ``"ok"``            — measured path, PNG + summary required.

    Raises:
        FileNotFoundError / ValueError: missing or unparsable status artefact.
    """
    status_json = Path(output_dir) / "bayesian_reliability.json"
    if not status_json.is_file() or status_json.stat().st_size == 0:
        raise FileNotFoundError(
            f"missing fresh status artefact: {status_json}"
        )
    with open(status_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    status = data.get("status", "unknown")
    if status not in ("ok", "not_applicable", "unavailable_latency"):
        raise ValueError(f"Unknown psychophysics status: {status!r}")
    if status == "unavailable_latency":
        # ponytail: preserve the measured-absence result, not an estimated latency.
        if (type(data.get("n_ttc0")) is not int or data["n_ttc0"] <= 0 or
                not isinstance(data.get("reason"), str) or not data["reason"].strip()):
            raise ValueError("unavailable_latency requires TTC=0 trials and a reason")
    return status


def find_multisensory_ttc0(
    X_seqs: torch.Tensor,
    lengths: torch.Tensor,
    raw_dir: Optional[str] = "data/raw",
    val_indices: Optional[Sequence[int]] = None,
    stim_onset_frame: Optional[int] = None,
    dt_ms: float = 4.0,
    session_ids: Optional[Sequence[str]] = None,
    trial_ids: Optional[Sequence[int]] = None,
    target_ttc_ms: Optional[Sequence[Optional[float]]] = None,
    stimulus_conditions: Optional[Sequence[str]] = None,
) -> TTC0Selection:
    """Select trials matching declared multisensory_ttc_0ms.

    Uses declared target_ttc_ms and declared multisensory condition joined by
    stable (session_id, trial_id) or persisted dataset metadata.

    A complete valid declared subset with ZERO TTC=0 trials returns an empty
    mask with ``status="not_applicable"`` (legitimate absence, not an error).
    Raises ValueError on missing/unresolved identity, missing raw_dir for
    required keyed lookup, unresolved required multisensory TTC, or corrupt
    numeric TTC strings. Unimodal null TTC stays null.
    """
    B, T, _ = X_seqs.shape
    assert X_seqs.dim() == 3, f"Expected X_seqs (B, T, F), got {X_seqs.shape}"
    mask = torch.zeros(B, dtype=torch.bool, device=X_seqs.device)
    n_candidates = 0

    if val_indices is None:
        val_indices = getattr(X_seqs, "val_indices", None)
    if stim_onset_frame is None:
        stim_onset_frame = STIM_ONSET_FRAME
    if session_ids is None:
        session_ids = getattr(X_seqs, "session_ids", None)
    if trial_ids is None:
        trial_ids = getattr(X_seqs, "trial_ids", None)
    if target_ttc_ms is None:
        target_ttc_ms = getattr(X_seqs, "target_ttc_ms", None)
    if stimulus_conditions is None:
        stimulus_conditions = getattr(X_seqs, "stimulus_conditions", None)

    declared_index: Optional[dict] = None

    def _load_raw_index() -> dict:
        nonlocal declared_index
        if declared_index is None:
            if raw_dir is None or not os.path.exists(raw_dir):
                raise ValueError(
                    f"raw_dir {raw_dir!r} does not exist for declared events lookup. "
                    "Failing closed without inferred TTC0."
                )
            declared_index = load_declared_events_index(raw_dir)
        return declared_index

    def _raw_lookup(i: int) -> dict:
        if session_ids is None or trial_ids is None or i >= len(session_ids) or i >= len(trial_ids):
            raise ValueError(
                f"missing session_ids/trial_ids for keyed raw lookup at index {i}; "
                "failing closed without inferred TTC0."
            )
        key = (str(session_ids[i]), int(trial_ids[i]))
        index = _load_raw_index()
        if key not in index:
            raise ValueError(
                f"Unresolved declared events key {key}; failing closed without inferred TTC0."
            )
        return index[key]

    def _is_multisensory(i: int, cond: Optional[str]) -> bool:
        if cond is not None:
            return cond in ("multisensory", "looming_wind")
        L = int(lengths[i].item())
        vis = X_seqs[i, :L, VISUAL_ANGLE_IDX].abs()
        wind = X_seqs[i, :L, 1]
        return bool((vis > 1e-4).any().item()) and bool((wind > 0.5).any().item())

    def _parse_ttc(raw_ttc: object, context: str) -> Optional[float]:
        if raw_ttc is None or (isinstance(raw_ttc, float) and math.isnan(raw_ttc)):
            return None
        try:
            val = float(raw_ttc)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Corrupt target_ttc_ms {raw_ttc!r} in {context}: {e}"
            ) from e
        if not math.isfinite(val):
            raise ValueError(
                f"Non-finite target_ttc_ms {raw_ttc!r} in {context}"
            )
        return val

    if target_ttc_ms is not None and len(target_ttc_ms) == B:
        # Path 1: Persisted declared metadata in dataset
        for i in range(B):
            cond = (
                stimulus_conditions[i]
                if stimulus_conditions is not None and i < len(stimulus_conditions)
                else None
            )
            is_multi = _is_multisensory(i, cond)
            raw_ttc = target_ttc_ms[i]
            ttc_val = _parse_ttc(raw_ttc, context=f"dataset target_ttc_ms[{i}]")
            if is_multi:
                if ttc_val is None:
                    # Present-but-null/partial: keyed raw lookup when raw_dir provided
                    info = _raw_lookup(i)
                    ttc_val = _parse_ttc(
                        info.get("target_ttc_ms"),
                        context=f"raw events for index {i}",
                    )
                    if ttc_val is None:
                        raise ValueError(
                            f"Unresolved required target_ttc_ms for multisensory "
                            f"trial index {i} after raw lookup; failing closed."
                        )
                n_candidates += 1
                if abs(ttc_val - 0.0) < 1e-3:
                    mask[i] = True
            # Unimodal: legitimate null stays null — not a candidate.

    elif session_ids is not None and trial_ids is not None and len(session_ids) == B and len(trial_ids) == B:
        # Path 2: Exact keyed lookup by stable (session_id, trial_id) in declared events
        index = _load_raw_index()
        for i in range(B):
            cond = (
                stimulus_conditions[i]
                if stimulus_conditions is not None and i < len(stimulus_conditions)
                else None
            )
            if not _is_multisensory(i, cond):
                continue
            key = (str(session_ids[i]), int(trial_ids[i]))
            if key not in index:
                raise ValueError(
                    f"Unresolved declared events key {key}; failing closed without inferred TTC0."
                )
            info = index[key]
            is_multi = info["type"] in ("looming_wind", "multisensory")
            if not is_multi:
                continue
            ttc_val = _parse_ttc(info.get("target_ttc_ms"), context=f"raw events {key}")
            if ttc_val is None:
                raise ValueError(
                    f"Unresolved required target_ttc_ms for multisensory trial {key}; "
                    "failing closed."
                )
            n_candidates += 1
            if abs(ttc_val - 0.0) < 1e-3:
                mask[i] = True
    else:
        # Legacy dataset lacking trial_ids and target_ttc_ms: fail closed
        raise ValueError(
            "No multisensory TTC=0ms trials found: dataset lacks trial_ids and target_ttc_ms "
            "metadata required for exact TTC=0 selection. Cannot safely infer condition from kinematics. "
            "Regenerate dataset with trial_ids or provide raw_dir."
        )

    n_ttc0 = int(mask.sum().item())
    status = "ok" if n_ttc0 > 0 else "not_applicable"
    return TTC0Selection(
        mask=mask, n_candidates=n_candidates, n_ttc0=n_ttc0, status=status
    )


# Visual-angle channel index in the feature dimension (X[:, :, 0]).
VISUAL_ANGLE_IDX = 0


def inject_visual_noise(
    X_batch: torch.Tensor,
    lengths: torch.Tensor,
    sigma: float,
    seed: int | None = None,
    trial_seed_offset: int = 0,
) -> torch.Tensor:
    """
    Add N(0, σ²) noise to visual channel (X[:,:,0]).

    Noise is applied only to non-padded frames (respects sequence masks).
    Returns a new tensor (does not mutate the original).

    Round-1 note (Reviewer A MAJOR-2): only this channel is degraded;
    the MCMC prior columns are held fixed, so the experiment measures
    gate sensitivity to evidence noise, not cue re-weighting.

    Round-2 fix (Reviewer A MAJOR-D-3 / Reviewer B M-1a): noise is drawn
    from an explicit ``torch.Generator`` — bare ``torch.randn`` left
    results irreproducible.

    Round-3 fix (Reviewer B MAJOR-2): *trial_seed_offset* gives each
    σ level an independent, reproducible noise realisation.  The
    previous design reused ONE seed across all σ levels, so the SAME
    standard-normal draw was scaled by different σ — i.e. every
    condition saw the identical noise realisation, not independent
    ones.  That is not a paired design over independent noisy
    presentations; it is one noise field repeatedly rescaled, and the
    paired tests' error terms did not contain the between-condition
    noise variance they claim to test.  With per-sigma offsets, each
    trial keeps a fixed index across conditions (the pairing unit is
    still the stimulus trial) while the noise realisations differ.
    """
    if sigma == 0.0:
        return X_batch.clone()

    gen = torch.Generator(device=X_batch.device)
    if seed is None:
        gen.seed()
    else:
        # Large coprime stride decorrelates the per-sigma streams.
        gen.manual_seed(int(seed) + 1_000_003 * int(trial_seed_offset))

    X_noisy = X_batch.clone()
    for i in range(X_noisy.shape[0]):
        L = int(lengths[i].item())
        noise = torch.randn(L, device=X_noisy.device, generator=gen) * sigma
        X_noisy[i, :L, VISUAL_ANGLE_IDX] += noise

    return X_noisy


# ===================================================================
# Metric Extraction
# ===================================================================

def extract_gate_trajectory(
    internals: dict,
    lengths: torch.Tensor,
    anchor_frames: Optional[Sequence[int]] = None,
    target_anchor: int = STIM_ONSET_FRAME,
) -> np.ndarray:
    """
    Extract mean g_gru(t) across trials at each time-step, aligned by anchor_frames.

    Returns: (T,) numpy array of mean gate probabilities.
    """
    g_gru = internals["routing_gates"][:, :, 1]  # (B, T)
    B, T = g_gru.shape
    if anchor_frames is not None and len(anchor_frames) == B:
        aligned = torch.zeros((B, T), device=g_gru.device, dtype=g_gru.dtype)
        mask = torch.zeros((B, T), device=g_gru.device, dtype=torch.bool)
        for i in range(B):
            L = int(lengths[i].item())
            anchor_i = int(anchor_frames[i])
            shift = target_anchor - anchor_i
            src_start = max(0, -shift)
            src_end = min(L, T - shift)
            dst_start = max(0, shift)
            dst_end = min(T, shift + L)
            if dst_end > dst_start and src_end > src_start:
                copy_len = min(dst_end - dst_start, src_end - src_start)
                aligned[i, dst_start : dst_start + copy_len] = g_gru[
                    i, src_start : src_start + copy_len
                ]
                mask[i, dst_start : dst_start + copy_len] = True
        # Unobserved positions (count == 0) must be NaN, not 0 — short
        # T < target_anchor sequences leave shifted windows empty.
        count = mask.float().sum(dim=0)
        safe_count = count.clamp(min=1)
        result = aligned.sum(dim=0) / safe_count
        result[count == 0] = float("nan")
        return result.cpu().numpy()
    else:
        mask = torch.arange(T, device=g_gru.device).unsqueeze(0) < lengths.unsqueeze(1)
        g_gru_masked = g_gru * mask.float()
        count = mask.float().sum(dim=0)
        safe_count = count.clamp(min=1)
        result = g_gru_masked.sum(dim=0) / safe_count
        result[count == 0] = float("nan")
        return result.cpu().numpy()


def extract_latency_to_peak(
    Y_pred: torch.Tensor,
    lengths: torch.Tensor,
    dt_ms: float = 4.0,
    stim_onset_frame: Optional[int] = None,
    search_window_frames: Optional[int] = None,
    pre_stim_window_frames: Optional[int] = None,
    anchor_frames: Optional[Sequence[int]] = None,
) -> list[float]:
    """
    Per-trial latency to peak velocity (ms) relative to stimulus onset / anchor.

    Searches in physiologically relevant window around stimulus onset:
    [anchor - pre_stim_window : anchor + search_window] to capture both
    pre-collision escapes (biologically valid) and post-stimulus responses,
    while avoiding spurious peaks from distant baseline drift.

    Returns NaN for zero/flat/nonfinite windows or zero-length trials.

    Args:
        search_window_frames: Post-stimulus frames (default +4s from dt_ms).
        pre_stim_window_frames: Pre-stimulus frames (default -2s from dt_ms).
        anchor_frames: Optional per-trial anchor frame indices.
    """
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be finite and positive")
    if search_window_frames is None:
        search_window_frames = math.ceil(4000.0 / dt_ms)
    if pre_stim_window_frames is None:
        pre_stim_window_frames = math.floor(2000.0 / dt_ms)
    if stim_onset_frame is None:
        stim_onset_frame = STIM_ONSET_FRAME
    latencies = []
    B, T = Y_pred.shape
    for i in range(B):
        L = int(lengths[i].item())
        if L == 0:
            latencies.append(float("nan"))
            continue

        anchor = (
            int(anchor_frames[i])
            if anchor_frames is not None and i < len(anchor_frames)
            else stim_onset_frame
        )

        # Search in window around stimulus: [anchor - pre_window : anchor + post_window]
        search_start = max(0, anchor - pre_stim_window_frames)
        search_end = min(L, anchor + search_window_frames)

        if search_start >= L or search_end <= search_start:
            latencies.append(float("nan"))
            continue

        vel_window = Y_pred[i, search_start:search_end]
        if not torch.isfinite(vel_window).all():
            latencies.append(float("nan"))
            continue

        abs_vel = vel_window.abs()
        assert abs_vel.shape == vel_window.shape
        if abs_vel.max().item() - abs_vel.min().item() < 1e-6:
            # ponytail: range rejects tonic output; calibrate prominence if drift yields false peaks.
            latencies.append(float("nan"))
            continue

        peak_in_window = torch.argmax(abs_vel).item()
        peak_frame = search_start + peak_in_window

        latency_ms = float((peak_frame - anchor) * dt_ms)
        latencies.append(latency_ms)
    return latencies


def extract_peak_velocity(
    Y_pred: torch.Tensor,
    lengths: torch.Tensor,
    dt_ms: Optional[float] = None,
    stim_onset_frame: Optional[int] = None,
    anchor_frames: Optional[Sequence[int]] = None,
) -> list[float]:
    """Peak magnitude in the latency window when dt_ms is supplied; legacy full-trial otherwise."""
    if dt_ms is not None and (not math.isfinite(dt_ms) or dt_ms <= 0):
        raise ValueError("dt_ms must be finite and positive")
    if stim_onset_frame is None:
        stim_onset_frame = STIM_ONSET_FRAME
    peaks = []
    B, T = Y_pred.shape
    for i in range(B):
        L = int(lengths[i].item())
        if dt_ms is None:
            start, end = 0, L
        else:
            anchor = int(anchor_frames[i]) if anchor_frames is not None and i < len(anchor_frames) else stim_onset_frame
            start = max(0, anchor - math.floor(2000.0 / dt_ms))
            end = min(L, anchor + math.ceil(4000.0 / dt_ms))
        if start >= end:
            peaks.append(float("nan") if dt_ms is not None else 0.0)
            continue
        vel = Y_pred[i, start:end]
        if not torch.isfinite(vel).all():
            peaks.append(float("nan"))
            continue
        peak = vel.abs().max().item()
        peaks.append(peak if dt_ms is None or peak >= 1e-6 else float("nan"))
    return peaks

# ===================================================================
# Figure Creation
# ===================================================================

def create_figure(
    gate_trajectories: dict[float, np.ndarray],
    latency_stats: dict,
    T: int,
    dt_ms: float,
    output_path: str,
    stim_onset_frame: int = STIM_ONSET_FRAME,
) -> None:
    """
    Dual-panel Lancet/Cell figure.

    Panel A: Gate modulation by noise level (g_gru vs time).
    Panel B: Psychometric curve (latency vs noise level).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG_COLOUR)

    for ax in axes:
        ax.set_facecolor(BG_COLOUR)
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOUR)
        ax.tick_params(colors=AXIS_COLOUR, labelsize=10)
        ax.xaxis.label.set_color(AXIS_COLOUR)
        ax.yaxis.label.set_color(AXIS_COLOUR)
        ax.title.set_color(AXIS_COLOUR)

    # ---- Panel A: Gate trajectories ----
    ax_a = axes[0]
    time_ms = (np.arange(T) - stim_onset_frame) * dt_ms

    for idx, sigma in enumerate(gate_trajectories):
        colour = GATE_COLOURS[idx % len(GATE_COLOURS)]
        style = "-" if sigma == 0.0 else "--"
        lw = LINEWIDTH + 0.3 if sigma == 0.0 else LINEWIDTH
        ax_a.plot(
            time_ms,
            gate_trajectories[sigma],
            color=colour,
            linewidth=lw,
            linestyle=style,
            label=f"σ = {sigma:.0f}°",
            alpha=0.95 if sigma == 0.0 else 0.85,
        )

    ax_a.axvline(0, color=BASELINE_COLOUR, linewidth=0.8, linestyle=":", alpha=0.6)
    ax_a.set_xlabel("Time relative to stimulus onset (ms)", fontsize=11)
    ax_a.set_ylabel("MoR Gate Probability  g_gru(t)", fontsize=11)
    ax_a.set_title("A. Gate Modulation by Visual Noise", fontsize=12, fontweight="bold")
    ax_a.legend(fontsize=9, loc="upper left", framealpha=0.85)
    ax_a.set_ylim(-0.05, 1.05)

    # ---- Panel B: Psychometric curve ----
    ax_b = axes[1]
    sigmas = sorted(latency_stats.keys())
    measured = [s for s in sigmas if latency_stats[s]["n"] > 0]
    means = [latency_stats[s]["mean"] for s in measured]
    sems = [latency_stats[s]["sem"] for s in measured]

    ax_b.errorbar(
        measured,
        means,
        yerr=sems,
        color=LATENCY_COLOUR,
        marker="o",
        markersize=7,
        markeredgecolor=LATENCY_COLOUR,
        markerfacecolor="white",
        linewidth=LINEWIDTH,
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
    )

    ax_b.set_xlabel("Visual Noise Level σ (degrees)", fontsize=11)
    ax_b.set_ylabel("Mean signed latency ± trial SEM (ms; descriptive)", fontsize=11)
    ax_b.set_title("B. Fixed recorded trials (descriptive)", fontsize=12, fontweight="bold")
    ax_b.set_xlim(-2, max(sigmas) + 5)

    # Annotate N per point
    for s in measured:
        n = latency_stats[s]["n"]
        ax_b.annotate(
            f"n={n}",
            xy=(s, latency_stats[s]["mean"]),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=BASELINE_COLOUR,
        )

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", output_path)


# ===================================================================
# Main Pipeline
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 10 — In-Silico Psychophysics: "
                    "Routing-Gate Noise Sensitivity "
                    "(priors held fixed; NOT a cue-combination test)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "runs", "default", "best_model.pth"),
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--noise_levels",
        type=float,
        nargs="+",
        default=NOISE_LEVELS,
        help="Visual noise sigma values in degrees.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the processed dataset (default: "
             "<repo>/data/processed/nsmor_dataset.pt).",
    )
    parser.add_argument(
        "--nested_prior_artifact",
        type=str,
        default=None,
        help="Optional validated nested artifact with the saved outer-validation split.",
    )
    parser.add_argument(
        "--qc_sealed_nested_prior_sha256", type=str, default=None,
        help="Optional external QC-sealed SHA-256 to check the nested checkpoint's embedded artifact digest; cannot authenticate a checkpoint missing that digest.",
    )
    parser.add_argument(
        "--trusted_historical_checkpoint_sha256", type=str, default=None,
        help="Independently pinned SHA-256 of historical checkpoint bytes (required for historical analysis).",
    )
    parser.add_argument(
        "--trusted_historical_artifact_sha256", type=str, default=None,
        help="Independently pinned SHA-256 of historical nested artifact bytes.",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "data", "raw"),
        help="Raw session directory whose events CSVs identify the "
             "multisensory_ttc_0ms condition.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "results"),
        help="Directory for output figures and JSON.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2400,
        help="Crop sequences longer than this (cuDNN compatibility). 0 = disable.",
    )
    parser.add_argument(
        "--pre_anchor_frames",
        type=int,
        default=1200,
        help="Number of frames before anchor to include in anchor-aligned crop.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the visual-noise generator (paired design must be "
             "reproducible; recorded in the JSON summary).",
    )
    parser.add_argument(
        "--dt_ms",
        type=float,
        default=None,
        help="Frame interval in ms (default: saved model.dt_ms; explicit value must match).",
    )
    args = parser.parse_args()
    if any(not math.isfinite(sigma) or sigma < 0 for sigma in args.noise_levels):
        parser.error("--noise_levels requires finite nonnegative sigma values")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Load model & data ---
    model = load_checkpoint(args.checkpoint, device)
    args.dt_ms = resolve_dt_ms(model, args.dt_ms)
    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    pre_anchor_frames = getattr(args, "pre_anchor_frames", 1200)
    val_data = load_validation_data(
        device,
        max_seq_len=max_seq_len,
        pre_anchor_frames=pre_anchor_frames,
        dataset_path=args.dataset,
        nested_prior_artifact=Path(args.nested_prior_artifact) if args.nested_prior_artifact else None,
        qc_sealed_nested_prior_sha256=args.qc_sealed_nested_prior_sha256,
        checkpoint_model=model,
        trusted_historical_checkpoint_sha256=args.trusted_historical_checkpoint_sha256,
        trusted_historical_artifact_sha256=args.trusted_historical_artifact_sha256,
    )
    X_val, Y_val, lengths_val = val_data
    meta: TrialMeta = val_data.meta
    validation_scope = getattr(model, "analysis_validation_scope", "historical_unscoped")
    analysis_evidence = {
        "validation_scope": validation_scope,
        "sample_scope": ("outer_validation" if validation_scope == "nested_outer_validation"
                         else "diagnostic_validation"),
        "n_validation_rows": len(meta.val_indices),
        "dataset_binding": getattr(model, "analysis_dataset_binding", "unverified"),
        "holdout_eligible": False,
        "validation_note": ("Checkpoint-selected outer validation; descriptive only, "
                            "ineligible as independent holdout evidence"
                            if validation_scope == "nested_outer_validation"
                            else "Contaminated/unscoped diagnostic; ineligible for holdout or QC claims"),
    }

    # --- Filter to multisensory_ttc_0ms ---
    selection = find_multisensory_ttc0(
        X_val,
        lengths_val,
        raw_dir=args.raw_dir,
        val_indices=meta.val_indices,
        stim_onset_frame=meta.stim_onset_frame,
        dt_ms=args.dt_ms,
        session_ids=meta.session_ids,
        trial_ids=meta.trial_ids,
        target_ttc_ms=meta.target_ttc_ms,
        stimulus_conditions=meta.stimulus_conditions,
    )
    ttc0_mask = selection.mask
    n_ttc0 = selection.n_ttc0
    logger.info(
        "multisensory_ttc_0ms trials found: %d / %d (candidates=%d, status=%s)",
        n_ttc0, X_val.shape[0], selection.n_candidates, selection.status,
    )
    if selection.status == "not_applicable":
        # Structured not_applicable: missing experimental condition is
        # legitimate (real corpus has 0 declared TTC=0). No dummy figure.
        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, "bayesian_reliability.json")
        not_applicable = {
            **analysis_evidence,
            "status": "not_applicable",
            "reason": (
                "No multisensory_ttc_0ms trials found. "
                "Declared TTC=0 subset is empty; this is a legitimate "
                "missing experimental condition, not a data error."
            ),
            "requested_condition": "multisensory_ttc_0ms",
            "n_trials_total": int(X_val.shape[0]),
            "n_candidates": selection.n_candidates,
            "n_ttc0": selection.n_ttc0,
            "n_trials_matched": 0,
            "dt_ms": args.dt_ms,
            "raw_dir": args.raw_dir,
        }
        with open(summary_path, "w") as f:
            json.dump(not_applicable, f, indent=2, allow_nan=False)
        logger.warning(
            "not_applicable: no multisensory_ttc_0ms trials. Summary: %s",
            summary_path,
        )
        return

    X_ttc0 = X_val[ttc0_mask]
    Y_ttc0 = Y_val[ttc0_mask]
    L_ttc0 = lengths_val[ttc0_mask]
    ttc0_indices = ttc0_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    meta_ttc0 = meta.subset(ttc0_indices)
    ttc0_anchor_frames = meta_ttc0.anchor_frames

    # --- Run noise sweep ---
    gate_trajectories: dict[float, np.ndarray] = {}
    latency_stats: dict = {}
    latency_arrays: dict[float, np.ndarray] = {}  # per-trial latencies (NaN=excluded)
    snr_by_sigma: dict[float, Optional[float]] = {}
    derived_seeds: dict[float, Optional[int]] = {}  # Round-3 m-3a
    T = X_ttc0.shape[1]

    for sigma_idx, sigma in enumerate(args.noise_levels):
        # the peak |visual_angle| excursion within the valid length;
        # report the median SNR across trials so each sigma has an
        # explicit physical meaning.  Round-2 fix (Reviewer A MAJOR-D-4):
        # the per-sigma SNR value is persisted in the JSON summary, not
        # only logged.
        peak_excursions = []
        for i in range(X_ttc0.shape[0]):
            L = int(L_ttc0[i].item())
            angle = X_ttc0[i, :L, VISUAL_ANGLE_IDX].cpu().numpy()
            if angle.size:
                peak_excursions.append(float(np.abs(angle).max()))
        sig_scale = float(np.median(peak_excursions)) if peak_excursions else 0.0
        snr_db = (
            float(20.0 * np.log10(sig_scale / sigma))
            if sigma > 0.0 and np.isfinite(sig_scale) and sig_scale > 0.0
            else None  # Clean σ=0 or undefined signal scale has no finite dB SNR.
        )
        snr_by_sigma[sigma] = snr_db if snr_db is None or math.isfinite(snr_db) else None
        logger.info("--- Noise level σ = %.1f° (median SNR %s dB) ---",
                    sigma, f"{snr_db:.1f}" if snr_db is not None else "undefined")

        X_noisy = inject_visual_noise(
            X_ttc0, L_ttc0, sigma, seed=args.seed,
            trial_seed_offset=sigma_idx,
        )
        # Round-3 (Reviewer A m-3a): record the ACTUAL derived seed per
        # σ so each condition is independently reproducible.
        derived_seeds[sigma] = (
            int(args.seed) + 1_000_003 * int(sigma_idx)
            if sigma > 0.0 else None
        )

        with torch.no_grad():
            Y_pred, internals = model(
                X_noisy, L_ttc0, return_internals=True
            )

        # Gate trajectory — align to the per-run stim onset so the
        # trajectory origin, figure time axis, and post-stim split all
        # share one coordinate frame (F1: nondefault --pre_anchor_frames).
        gate_traj = extract_gate_trajectory(
            internals, L_ttc0, anchor_frames=ttc0_anchor_frames,
            target_anchor=meta.stim_onset_frame,
        )
        gate_trajectories[sigma] = gate_traj
        post_stim_gate = gate_traj[meta.stim_onset_frame:]
        valid_gate = post_stim_gate[np.isfinite(post_stim_gate)]
        mean_post_gate = float(np.mean(valid_gate)) if valid_gate.size > 0 else float("nan")
        logger.info(
            "  g_gru mean (post-stim): %.4f",
            mean_post_gate,
        )

        # Latency
        latencies = extract_latency_to_peak(
            Y_pred, L_ttc0, dt_ms=args.dt_ms,
            stim_onset_frame=meta.stim_onset_frame,
            anchor_frames=ttc0_anchor_frames,
        )
        latency_arrays[sigma] = np.asarray(latencies, dtype=np.float64)
        valid = latency_arrays[sigma][np.isfinite(latency_arrays[sigma])]
        mean_lat = float(np.mean(valid)) if valid.size else None
        sem_lat = (
            float(np.std(valid, ddof=1) / np.sqrt(valid.size))
            if valid.size > 1 else 0.0 if valid.size else None
        )
        latency_stats[sigma] = {
            "mean": mean_lat,
            "sem": sem_lat,
            "n": int(valid.size),
            # Honest label: NaN exclusions cover flat/zero/nonfinite windows
            # and zero-length trials, not only prestim peaks.
            "n_excluded": int((~np.isfinite(latency_arrays[sigma])).sum()),
            "std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0 if valid.size else None,
        }
        # Round-2 fix (Reviewer B M-1c): report the VALID count, not the
        # raw array length (which includes NaN-excluded trials).
        logger.info("  Latency: %s ± %s ms (n=%d)",
                    f"{mean_lat:.1f}" if mean_lat is not None else "unavailable",
                    f"{sem_lat:.1f}" if sem_lat is not None else "unavailable", int(valid.size))

        # Peak velocity
        peaks = extract_peak_velocity(
            Y_pred, L_ttc0, dt_ms=args.dt_ms,
            stim_onset_frame=meta.stim_onset_frame,
            anchor_frames=ttc0_anchor_frames,
        )
        finite_peaks = np.asarray(peaks)[np.isfinite(peaks)]
        logger.info("  Peak Vel (signed-latency window -2/+4s): %.2f cm/s (n=%d)",
                    float(np.mean(finite_peaks)) if finite_peaks.size else float("nan"),
                    int(finite_peaks.size))

    # Paired descriptive contrasts on the fixed recorded TTC=0 trial set.
    # A session/recording prefix is not a verified animal identity. Trials
    # can share an animal, so their contrasts cannot supply animal-level p
    # values, multiplicity corrections, or confidence intervals. Keep the
    # observed model shift, including constant positive shifts for which a
    # trial-level significance test would be degenerate.
    effect_sizes: dict[float, float] = {}
    if 0.0 in latency_arrays:
        baseline = latency_arrays[0.0]
        clean_valid = np.isfinite(baseline)
        for sigma in args.noise_levels:
            if sigma == 0.0:
                continue
            noisy = latency_arrays[sigma]
            # Each contrast uses trials measured in both this condition
            # and the clean condition; missing responses stay excluded.
            valid = clean_valid & np.isfinite(noisy)
            n_pairs = int(valid.sum())
            latency_stats[sigma]["n_pairs_vs_clean"] = n_pairs
            if not n_pairs:
                continue
            diffs = noisy[valid] - baseline[valid]
            if n_pairs > 1 and float(np.std(diffs, ddof=1)) <= 1e-12:
                latency_stats[sigma]["n_degenerate_zero_variance"] = n_pairs

            # Descriptive Hodges-Lehmann location of paired trial shifts;
            # no population test is attached to this fixed-set estimate.
            pairwise_avg = (
                diffs[:, None] + diffs[None, :]
            )[np.triu_indices(n_pairs, k=1)]
            hl = float(np.median(np.concatenate([pairwise_avg, diffs])))
            effect_sizes[sigma] = hl
            latency_stats[sigma]["hodges_lehmann_ms"] = hl
            logger.info(
                "  σ=%.1f° vs σ=0°: descriptive paired HL shift=%+.3f ms (n=%d)",
                sigma, hl, n_pairs,
            )
    else:
        logger.warning("σ=0 absent; no clean-condition paired contrasts available.")

    # --- Create figure ---
    fig_path = os.path.join(args.output_dir, "bayesian_reliability.png")
    create_figure(
        gate_trajectories, latency_stats, T, args.dt_ms, fig_path,
        stim_onset_frame=meta.stim_onset_frame,
    )

    # --- Export JSON summary ---
    summary = {
        **analysis_evidence,
        "noise_levels": args.noise_levels,
        "n_ttc0_trials": n_ttc0,
        "n_candidates": selection.n_candidates,
        "stim_onset_frame": meta.stim_onset_frame,
        "dt_ms": args.dt_ms,
        # Round-1 (Reviewer A MAJOR-2): explicit scope + SNR semantics
        "scope": (
            "Routing-gate sensitivity to VISUAL-channel noise. "
            "MCMC prior columns held fixed across conditions; this is "
            "NOT a Bayesian cue-combination test. Descriptive model perturbations "
            "on fixed recorded trials; animal population inference unavailable. " + LATENCY_SCOPE
        ),
        "snr_definition": (
            "SNR_dB = 20*log10(median peak |visual_angle| / sigma); "
            "sigma is additive noise in degrees on the raw visual-angle "
            "channel before sensory encoding."
        ),
        # Round-2 fix (Reviewer A MAJOR-D-4): per-sigma SNR values are
        # persisted so the summary is statistically self-contained.
        "snr_db_by_sigma": {str(k): v for k, v in snr_by_sigma.items()},
        "snr_clean_condition": "null: σ=0 has no finite dB SNR; null for undefined signal scale",
        "noise_seed": args.seed,
        # Round-3 (Reviewer A m-3a): per-σ derived seeds for exact
        # per-condition reproduction.
        "derived_seeds_by_sigma": {
            str(k): v for k, v in derived_seeds.items()
        },
        "noise_realisations": (
            "independent per sigma level (seed + 1000003*sigma_index); "
            "paired contrasts use the recorded stimulus trial; "
            "independent animal identities are unverified"
        ),
        "latency_dispersion": "SD and SEM describe variation across recorded trials; SEM is not an animal-level confidence interval",
        "latency_stats": {
            str(k): v for k, v in latency_stats.items()
        },
        "gate_post_stim_mean": {
            str(sigma): (
                float(np.mean(gate_trajectories[sigma][meta.stim_onset_frame:][np.isfinite(gate_trajectories[sigma][meta.stim_onset_frame:])]))
                if np.any(np.isfinite(gate_trajectories[sigma][meta.stim_onset_frame:]))
                else None
            )
            for sigma in args.noise_levels
        },
        "inference": {
            "status": "descriptive_only",
            "design": (
                "paired contrasts on the fixed recorded trial set; clean-condition "
                "exclusions plus complete pairs per noise level; noise realisations "
                "separate across σ levels" if 0.0 in latency_arrays
                else "σ=0 absent; no paired contrasts on fixed recorded trial set"
            ),
            "independent_animal_count": None,
            "population_inference_status": "unavailable_unverified_animal_identity_and_independence",
            "correction": None,
            "test": None,
            "effect_size": "descriptive paired-trial Hodges-Lehmann shift (ms); fixed recorded trials only",
            "p_values_uncorrected": None,
            "p_values_holm_corrected": None,
            "effect_sizes_hodges_lehmann_ms": {
                str(k): v for k, v in effect_sizes.items()
            },
        },
    }
    json_path = os.path.join(args.output_dir, "psychophysics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    logger.info("JSON summary saved → %s", json_path)

    # Status artefact for Phase G runner gate
    status_path = os.path.join(args.output_dir, "bayesian_reliability.json")
    latency_available = any(stats["n"] > 0 for stats in latency_stats.values())
    status_data = {
        **analysis_evidence,
        "status": "ok" if latency_available else "unavailable_latency",
        "n_ttc0": n_ttc0,
        "n_candidates": selection.n_candidates,
        "n_trials_total": int(X_val.shape[0]),
        "dt_ms": args.dt_ms,
    }
    if not latency_available:
        status_data["reason"] = "No finite latency at any measured visual-noise level."
    with open(status_path, "w") as f:
        json.dump(status_data, f, indent=2, allow_nan=False)
    logger.info("Status artefact saved → %s", status_path)

    logger.info("Done. All outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
