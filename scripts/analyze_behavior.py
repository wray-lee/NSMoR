"""Model-free exploratory behavior analyses of the NSMoR cricket corpus.

EXPLORATORY / in-sample / model-free.  No checkpoint, no nested prior, no
neural network is used: only the physical input channels (visual angle,
wind state) and the recorded body velocity.  This is a clean port of the
throwaway prototype chain (``analyse.py`` / ``robust.py`` / ``final.py`` /
``facil.py``) into one reproducible script with type hints, shape
assertions, Google-style docstrings and PEP 8 layout.

"Model-free" is qualified: the PRIMARY race / facilitation / accumulation
statistics use only the 10 cm/s crossing and the physical channels, but the
**s1 labeling-responder quantities** (``resp_label``, the s1 race V term and
the s1 rate contrast) reuse the labeling pipeline's own response criterion
(``nsmor.pipeline.labeling._check_sustained_speed``), so those quantities
depend on the labeling pipeline, not on a learned model but not on the raw
crossing either.

Design flags (also stamped into ``behavior_summary.json``):

* ``exploratory=true``        -- hypotheses were generated on this corpus.
* ``in_sample=true``          -- every trial is used; no held-out split.
* ``between_group_design=true`` -- condition is CONFOUNDED with recording
  prefix (all 64 prefixes are condition-pure), so no condition contrast can
  be separated from a session / animal effect.

Unit of resampling (B1)
-----------------------
The prototype bootstrapped over 128 ``session_ids`` as if they were
independent.  They are not: sessions ending in ``_session_N`` share a
recording prefix, so the 128 sessions collapse to 64 prefixes (verified in
the corpus).  The cluster unit used here is the *recording prefix* taken
from the nested-prior artifact's ``recording_prefix_keys`` (64 prefixes)
when supplied, otherwise derived with
:func:`nsmor.pipeline.grouping.animal_keys_of`.  Animal identity is
UNVERIFIED -- a recording prefix is treated as one animal only as a working
assumption, stated wherever it matters.

The bootstrap is a **true with-replacement cluster bootstrap**: within each
group (stratum) exactly ``len(pool)`` prefixes are drawn WITH replacement,
and a prefix drawn ``m`` times contributes multiplicity weight ``m`` to
every one of its trials.  The earlier implementation unioned the drawn
prefixes into a boolean mask, so a prefix selected twice was silently
counted once (under-dispersed / wrong).  Every prefix-cluster CI below uses
the multiplicity weights; ``tests/test_analyze_behavior.py`` compares them
against a reference cluster bootstrap.

Primary trigger rule (D1, declared before looking at any outcome)
------------------------------------------------------------------
Reference (trigger) event per condition: wind onset for wind-containing
trials, visual collision for visual-only trials -- exactly
:func:`nsmor.pipeline.conditions.derive_anchor_frames`, the same onset
reference the labeling pipeline uses.

D1 fixes a round-2 blocker: the visual response window
``[collision - 2000 ms, collision + 1000 ms]`` reached ~1750 ms BEFORE the
250 ms stationarity gate, so a visual responder could be certified from a
pre-gate crossing and its latency could be negative.  D1 applies ONE trigger
definition per modality, identically to eligibility and response search.

PRIMARY INCLUSION RULE (fixed before any latency / CDF / race / facilitation
statistic is computed; NOT chosen by outcome).  A trial is ELIGIBLE only if
the animal is STATIONARY at the trigger: ``|Y| < 10 cm/s`` at every frame of
the pre-trigger window ``[trigger - 250 ms, trigger]``.  Latency is then the
first frame STRICTLY after the trigger (``> trigger``) with ``|Y| >= 10
cm/s`` inside ``[trigger, trigger + W]`` with ``W = 2000 ms`` for EVERY
condition, so latency 0 is impossible and the visual window covers
``collision + 1000 ms``.  Non-eligible trials are excluded and their
per-condition counts are reported.  The labeling-consistent variant
(``|Y| < 5 cm/s`` in the same window) is reported alongside.  SENSITIVITY
analyses NEST inside / alongside the primary rule and are always labeled:

* (s1) labeling-pipeline responder definition -- sustained ``> 5 cm/s`` for
  250 ms with a 200 ms initiation latency bound
  (:func:`nsmor.pipeline.labeling._check_sustained_speed`) -- used for the
  response criterion in place of the 10 cm/s crossing, under the SAME D1
  eligibility gate as the primary response.  The ungated count is kept
  separately as ``resp_label_ungated`` (diagnostic only).
* (s2) the old all-trials rule (no stationarity gate), reported only as a
  contamination illustration.
* (s3) the round-1/2 collision-anchored visual window
  ``[collision - 2000 ms, collision + 1000 ms]``, kept only to show what D1
  changed.

Reference events, definitions, the rule and both handlings are written to
``behavior_summary.json``.

Usage::

    python scripts/analyze_behavior.py \\
        --dataset data/processed/nsmor_dataset.pt \\
        --output_dir results/behavior-20261010
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsmor.pipeline.conditions import (  # noqa: E402
    derive_anchor_frames,
    derive_stimulus_metadata,
)
from nsmor.pipeline.grouping import animal_keys_of  # noqa: E402
from nsmor.pipeline.labeling import _check_sustained_speed  # noqa: E402
from nsmor.pipeline.nested_prior import load_artifact_bytes  # noqa: E402

logger = logging.getLogger("analyze_behavior")

# ── Physical / analysis constants ────────────────────────────────────
DT_MS_DEFAULT = 4.0
ESCAPE_THRESHOLD_CMS = 10.0
# D1: ONE response window, W = 2000 ms, applied identically to every
# condition, measured STRICTLY AFTER the per-condition trigger.  For
# visual_only the trigger is the looming collision, so [collision,
# collision + 2000 ms] covers collision + 1000 ms by construction.
RESPONSE_WINDOW_MS = 2000.0
# Legacy collision-anchored window (round 1/2): [collision - 2000 ms,
# collision + 1000 ms].  Demoted to the labeled SENSITIVITY s3 only.
PRE_TRIGGER_WINDOW_MS = 1000.0
KINEMATIC_WINDOW_MS = 1000.0
SPONTANEOUS_THRESHOLD_CMS = 1.0  # labeling PREWALK threshold
CONDITIONS = ("visual_only", "wind_only", "multisensory")
LABEL_NAMES = {0: "ESCAPE", 1: "PREWALK", 2: "PRE_ACTIVE", 3: "NO_RESPONSE"}
RESPONDER_LABELS = frozenset({0, 1})  # ESCAPE, PREWALK (labeling pipeline)

# D1 primary inclusion rule: the animal must be stationary at the trigger.
STATIONARY_WINDOW_MS = 250.0  # pre-trigger window [trigger - 250 ms, trigger]
STATIONARY_THRESHOLD_CMS = 10.0  # primary stationarity threshold (|Y| < 10)
STATIONARY_THRESHOLD_SENS_CMS = 5.0  # labeling-consistent variant (|Y| < 5)
# s1: labeling-pipeline responder criterion (sustained > 5 cm/s for 250 ms).
LABELING_RESPONDER_THRESHOLD_CMS = 5.0
LABELING_RESPONDER_SUSTAINED_MS = 250.0
LABELING_RESPONDER_MAX_LATENCY_MS = 200.0
LABELING_RESPONDER_MIN_FRACTION = 0.5
LABELING_RESPONDER_ANCHOR_FRAMES = 2
RACE_CDF_CLIP = 1.0  # M1: the Miller bound is a probability, clipped at 1
GMM_N_BOOT = 20  # bootstrap replicates for the descriptive GMM ARI


# ══════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════
def load_corpus(dataset_path: Path) -> Dict[str, Any]:
    """Load the processed corpus through the restricted artifact decoder.

    Args:
        dataset_path: Path to ``nsmor_dataset.pt``.

    Returns:
        Mapping with the physical channels and metadata needed downstream.

    Raises:
        FileNotFoundError: ``dataset_path`` does not exist.
        ValueError: A required key is missing or metadata is inconsistent.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    data = load_artifact_bytes(dataset_path.read_bytes(), map_location="cpu")
    required = (
        "X_seqs", "Y_seqs", "labels", "session_ids", "stimulus_conditions",
        "target_ttc_ms", "lengths", "anchor_frames",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"dataset missing required keys: {missing}")

    n = len(data["X_seqs"])
    x = np.empty(n, dtype=object)
    y = np.empty(n, dtype=object)
    for i in range(n):
        xi = data["X_seqs"][i]
        xi = xi.detach().cpu().numpy() if hasattr(xi, "detach") else np.asarray(xi)
        yi = data["Y_seqs"][i]
        yi = yi.detach().cpu().numpy() if hasattr(yi, "detach") else np.asarray(yi)
        assert xi.ndim == 2 and xi.shape[1] >= 2, f"trial {i}: X shape {xi.shape}"
        assert yi.ndim == 1 and yi.shape[0] == xi.shape[0], (
            f"trial {i}: Y shape {yi.shape} vs X {xi.shape}"
        )
        x[i] = xi[:, :2].astype(np.float32)
        y[i] = yi.astype(np.float32)

    lengths = np.asarray(data["lengths"]).astype(int)
    conditions = np.asarray(data["stimulus_conditions"], dtype=object)
    ttc = np.asarray(data["target_ttc_ms"], dtype=float)
    labels = np.asarray(data["labels"]).astype(int)
    session_ids = np.asarray(data["session_ids"], dtype=object)
    anchors = np.asarray(data["anchor_frames"]).astype(int)
    assert conditions.shape == (n,), f"conditions shape {conditions.shape}"
    assert ttc.shape == (n,), f"ttc shape {ttc.shape}"
    assert labels.shape == (n,), f"labels shape {labels.shape}"

    # Reuse the pipeline's own condition derivation to validate the stamp.
    derived_cond, _ = derive_stimulus_metadata(list(x), lengths.tolist())
    if not np.array_equal(np.asarray(derived_cond, dtype=object), conditions):
        raise ValueError(
            "stamped stimulus_conditions disagree with channels; fail closed"
        )

    dt_ms = float(data.get("model_dt_ms", DT_MS_DEFAULT))
    return {
        "x": x,
        "y": y,
        "lengths": lengths,
        "conditions": conditions,
        "ttc": ttc,
        "labels": labels,
        "session_ids": session_ids,
        "anchors": anchors,
        "dt_ms": dt_ms,
        "pipeline_semantics_version": str(
            data.get("pipeline_semantics_version", "unknown")
        ),
        "feature_config": str(data.get("feature_config", "unknown")),
        "labeling_funnel": dict(data.get("labeling_funnel", {})),
        "n_trials": n,
    }


def resolve_prefix_keys(
    session_ids: np.ndarray, nested_prior_artifact: Optional[Path]
) -> Tuple[np.ndarray, str]:
    """Return the recording-prefix cluster key per trial.

    Args:
        session_ids: Per-trial session identifiers.
        nested_prior_artifact: Optional nested-prior artifact carrying
            ``recording_prefix_keys``.

    Returns:
        ``(prefix_keys, source)`` where ``source`` records provenance.

    Raises:
        ValueError: The artifact's keys disagree with the session-derived
            prefixes (fail closed on misaligned evidence).
    """
    derived = animal_keys_of(list(session_ids))
    assert derived.shape == (session_ids.shape[0],), (
        f"derived prefix shape {derived.shape}"
    )
    if nested_prior_artifact is None:
        return derived, "derived:animal_keys_of"
    artifact = load_artifact_bytes(
        nested_prior_artifact.read_bytes(), map_location="cpu"
    )
    keys = artifact.get("recording_prefix_keys")
    if keys is None:
        logger.warning(
            "artifact %s has no recording_prefix_keys; using derived prefixes",
            nested_prior_artifact,
        )
        return derived, "derived:animal_keys_of"
    keys = np.asarray(keys, dtype=object)
    if keys.shape != derived.shape or not np.array_equal(
        keys.astype(str), derived.astype(str)
    ):
        raise ValueError(
            "artifact recording_prefix_keys disagree with dataset session_ids; "
            "fail closed on misaligned prefix evidence"
        )
    return keys, f"artifact:{nested_prior_artifact.name}"


# ══════════════════════════════════════════════════════════════════════
# Event detection
# ══════════════════════════════════════════════════════════════════════
def first_above(series: np.ndarray, threshold: float) -> int:
    """Return the first index whose value exceeds ``threshold`` else ``-1``."""
    idx = np.where(series > threshold)[0]
    return int(idx[0]) if idx.size else -1


def visual_onset(x: np.ndarray) -> int:
    """First frame the visual angle departs its early floor (ill-defined)."""
    floor = float(np.min(x[:20, 0])) if x.shape[0] >= 20 else float(x[:, 0].min())
    return first_above(x[:, 0], floor + 0.02)


def wind_onset(x: np.ndarray) -> int:
    """First frame the wind channel is active (state > 0.5)."""
    return first_above(x[:, 1], 0.5)


def escape_frame(
    y_abs: np.ndarray, threshold: float, lo: int, hi: int, strict: bool = False
) -> int:
    """First frame in ``[lo, hi)`` reaching ``threshold`` else ``-1``.

    Args:
        y_abs: ``(T,)`` absolute velocity.
        threshold: Velocity threshold (cm/s).
        lo: Inclusive lower frame bound.
        hi: Exclusive upper frame bound.
        strict: When ``True`` the search starts at ``lo + 1`` (strictly
            after the trigger), so a response AT the trigger cannot be
            counted and latency 0 is impossible (D1).

    Returns:
        The frame index, or ``-1`` when no frame qualifies.
    """
    lo = max(0, lo) + (1 if strict else 0)
    hi = min(y_abs.shape[0], hi)
    if hi <= lo:
        return -1
    idx = np.where(y_abs[lo:hi] >= threshold)[0]
    return int(idx[0]) + lo if idx.size else -1


def detect_events(
    corpus: Dict[str, Any], dt_ms: float
) -> Dict[str, np.ndarray]:
    """Detect per-trial visual onset, wind onset and visual collision.

    The trigger / reference event reuses
    :func:`nsmor.pipeline.conditions.derive_anchor_frames`: first wind frame
    for wind-containing trials, peak visual angle (collision) for
    visual-only trials.
    """
    n = corpus["n_trials"]
    x, conditions = corpus["x"], corpus["conditions"]
    v_on = np.full(n, -1, dtype=int)
    w_on = np.full(n, -1, dtype=int)
    v_col = np.full(n, -1, dtype=int)
    for i in range(n):
        if conditions[i] != "wind_only":
            v_on[i] = visual_onset(x[i])
            v_col[i] = int(np.argmax(x[i][:, 0]))
        if conditions[i] != "visual_only":
            w_on[i] = wind_onset(x[i])
    anchor_list = derive_anchor_frames(list(x), corpus["lengths"].tolist())
    ref = np.asarray(anchor_list, dtype=int)
    assert ref.shape == (n,), f"reference shape {ref.shape}"
    # Cross-check the derived trigger against the physical events.
    mismatch = 0
    for i in range(n):
        expected = w_on[i] if conditions[i] != "visual_only" else v_col[i]
        if expected >= 0 and ref[i] != expected:
            mismatch += 1
    if mismatch:
        logger.warning(
            "%d/%d trials: derived anchor != physical trigger", mismatch, n
        )
    return {"v_on": v_on, "w_on": w_on, "v_col": v_col, "ref": ref}


# ══════════════════════════════════════════════════════════════════════
# Response detection (primary + sensitivity)
# ══════════════════════════════════════════════════════════════════════
def detect_responses(
    corpus: Dict[str, Any], events: Dict[str, np.ndarray], dt_ms: float
) -> Dict[str, np.ndarray]:
    """Detect escape responses under the D1 trigger rule and sensitivities.

    D1 TRIGGER RULE (one definition per modality, applied identically to the
    eligibility gate and the response search).  The trigger is the SAME
    per-condition reference the labeling pipeline uses
    (:func:`nsmor.pipeline.conditions.derive_anchor_frames`): wind onset for
    wind-containing trials, looming collision for visual_only.

    * ELIGIBILITY -- ``|Y| < 10 cm/s`` at every frame of the pre-trigger
      window ``[trigger - 250 ms, trigger]`` (the stationarity gate).
    * RESPONSE -- the first frame STRICTLY after the trigger (``> trigger``)
      with ``|Y| >= 10 cm/s`` inside ``[trigger, trigger + W]`` with
      ``W = 2000 ms`` for EVERY condition (``latency`` is thus always
      ``> 0`` and measured from the same clock as the gate).  For
      visual_only this window covers ``collision + 1000 ms``.

    Round 2 searched the visual response over ``[collision - 2000 ms,
    collision + 1000 ms]`` while gating eligibility over
    ``[collision - 250 ms, collision]``: the window reached ~1750 ms BEFORE
    the gate, so a responder could be certified from a pre-gate (pre-collision)
    crossing and its latency could be negative.  D1 removes that
    inconsistency by construction.

    SENSITIVITY (reported, never chosen by outcome):

    * ``resp_label`` / ``latency_label_ms`` -- s1: the SAME D1 eligibility
      gate as the primary ``resp`` (stationary at the trigger), with the
      labeling pipeline's sustained rule (``> 5 cm/s`` for 250 ms, 200 ms
      initiation bound) as the response criterion from the trigger.
      ``resp_label_ungated`` / ``latency_label_ungated_ms`` keep the same
      sustained criterion WITHOUT the gate, a labeled diagnostic only -- it
      is never the D1 labeling responder.
    * ``resp_all`` -- s2: the same D1 window with no stationarity gate, a
      contamination illustration only.
    * ``resp_elig5`` -- the labeling-consistent eligibility variant
      (``|Y| < 5 cm/s``) with the primary 10 cm/s crossing.
    * ``resp_legacy`` / ``latency_legacy_ms`` -- s3: the round-2
      collision-anchored window ``[collision - 2000 ms, collision + 1000 ms]``
      for visual_only (identical to primary elsewhere), a LABELED sensitivity
      that shows what the D1 fix changed.

    Args:
        corpus: Loaded corpus.
        events: Event frames from :func:`detect_events`.
        dt_ms: Frame interval (ms).

    Returns:
        Dict with the primary ``resp`` / ``latency_ms`` / ``eligible`` plus
        the sensitivity arrays and their eligibility masks.
    """
    n = corpus["n_trials"]
    y, ref = corpus["y"], events["ref"]
    v_col = events["v_col"]
    conditions = corpus["conditions"]
    win = int(round(RESPONSE_WINDOW_MS / dt_ms))
    pre = int(round(STATIONARY_WINDOW_MS / dt_ms))
    legacy_hi = int(round(PRE_TRIGGER_WINDOW_MS / dt_ms))

    resp = np.full(n, -1, dtype=int)          # primary (eligible only)
    resp_label = np.full(n, -1, dtype=int)    # s1 labeling responder (GATED)
    resp_label_ungated = np.full(n, -1, dtype=int)  # s1 UNGATED (diagnostic)
    resp_all = np.full(n, -1, dtype=int)      # s2 all-trials
    resp_elig5 = np.full(n, -1, dtype=int)    # labeling-consistent gate
    resp_legacy = np.full(n, -1, dtype=int)   # s3 collision-anchored window
    eligible = np.zeros(n, dtype=bool)        # |Y| < 10 throughout window
    eligible5 = np.zeros(n, dtype=bool)       # |Y| < 5 throughout window

    for i in range(n):
        if ref[i] < 0:
            continue
        y_abs = np.abs(y[i])
        t_ms = np.arange(y_abs.shape[0], dtype=float) * dt_ms
        lo_pre = max(0, ref[i] - pre)
        # Stationarity at the trigger over [trigger - 250 ms, trigger].
        eligible[i] = bool(np.all(y_abs[lo_pre : ref[i] + 1] < STATIONARY_THRESHOLD_CMS))
        eligible5[i] = bool(
            np.all(y_abs[lo_pre : ref[i] + 1] < STATIONARY_THRESHOLD_SENS_CMS)
        )
        # D1: ONE post-trigger window for every condition.
        lo = ref[i]
        hi = ref[i] + win
        if eligible[i]:
            resp[i] = escape_frame(
                y_abs, ESCAPE_THRESHOLD_CMS, lo, hi, strict=True
            )
        if eligible5[i]:
            resp_elig5[i] = escape_frame(
                y_abs, ESCAPE_THRESHOLD_CMS, lo, hi, strict=True
            )
        resp_all[i] = escape_frame(y_abs, ESCAPE_THRESHOLD_CMS, lo, hi, strict=True)
        # s1: labeling sustained criterion (from the trigger).  The ELIGIBILITY
        # gate applies here exactly as it does to the primary `resp`: a
        # responder may not be certified from a trial that was moving at the
        # trigger.  The sustained predicate is evaluated once; it is stored
        # into the gated array only when eligible and into the separately
        # named UNGATED array always (diagnostic, never the D1 labeling
        # responder).
        sustained = _check_sustained_speed(
            y[i], t_ms,
            start_ms=float(ref[i]) * dt_ms,
            duration_ms=LABELING_RESPONDER_SUSTAINED_MS,
            threshold=LABELING_RESPONDER_THRESHOLD_CMS,
            min_fraction=LABELING_RESPONDER_MIN_FRACTION,
            max_latency_ms=LABELING_RESPONDER_MAX_LATENCY_MS,
            anchor_min_frames=LABELING_RESPONDER_ANCHOR_FRAMES,
        )
        if sustained:
            # First post-trigger frame above 5 cm/s = the sustained response.
            frame = escape_frame(
                y_abs, LABELING_RESPONDER_THRESHOLD_CMS, ref[i], ref[i] + win,
                strict=True,
            )
            resp_label_ungated[i] = frame
            if eligible[i]:
                resp_label[i] = frame
        # s3: the legacy collision-anchored window (visual_only only).
        if conditions[i] == "visual_only" and v_col[i] >= 0:
            resp_legacy[i] = escape_frame(
                y_abs, ESCAPE_THRESHOLD_CMS, v_col[i] - win,
                v_col[i] + legacy_hi, strict=True,
            )
        else:
            resp_legacy[i] = escape_frame(
                y_abs, ESCAPE_THRESHOLD_CMS, lo, hi, strict=True
            )

    assert resp.shape == (n,), f"response shape {resp.shape}"
    assert eligible.shape == (n,), f"eligible shape {eligible.shape}"
    latency = np.where((resp >= 0) & eligible, (resp - ref) * dt_ms, np.nan)
    latency_label = np.where(
        (resp_label >= 0) & eligible, (resp_label - ref) * dt_ms, np.nan
    )
    # Diagnostic only: the same sustained criterion WITHOUT the D1 gate.
    latency_label_ungated = np.where(
        resp_label_ungated >= 0, (resp_label_ungated - ref) * dt_ms, np.nan
    )
    latency_elig5 = np.where(
        (resp_elig5 >= 0) & eligible5, (resp_elig5 - ref) * dt_ms, np.nan
    )
    latency_legacy = np.where(
        (resp_legacy >= 0) & eligible, (resp_legacy - ref) * dt_ms, np.nan
    )
    return {
        "resp": resp,
        "latency_ms": latency,
        "eligible": eligible,
        "eligible5": eligible5,
        "resp_label": resp_label,
        "latency_label_ms": latency_label,
        "resp_label_ungated": resp_label_ungated,
        "latency_label_ungated_ms": latency_label_ungated,
        "resp_all": resp_all,
        "resp_elig5": resp_elig5,
        "latency_elig5_ms": latency_elig5,
        "resp_legacy": resp_legacy,
        "latency_legacy_ms": latency_legacy,
    }


def kinematics(
    y: np.ndarray, resp: int, dt_ms: float, threshold: float
) -> Tuple[float, float, float, float]:
    """Return (peak_speed, peak_accel, time_to_peak_ms, bout_dur_ms)."""
    if resp < 0:
        return (np.nan,) * 4
    lo = resp
    hi = min(y.shape[0], resp + int(round(KINEMATIC_WINDOW_MS / dt_ms)))
    seg = np.abs(y[lo:hi])
    if seg.size < 3:
        return (np.nan,) * 4
    peak = float(seg.max())
    ttp = float(np.argmax(seg)) * dt_ms
    acc = np.abs(np.diff(seg)) / dt_ms * 1000.0
    pk_acc = float(acc.max()) if acc.size else np.nan
    above = np.where(seg >= threshold)[0]
    dur = float((above[-1] - above[0]) * dt_ms) if above.size else 0.0
    return peak, pk_acc, ttp, dur


# ══════════════════════════════════════════════════════════════════════
# Statistics helpers
# ══════════════════════════════════════════════════════════════════════
def finite(a: np.ndarray) -> np.ndarray:
    """Return the finite entries of ``a`` as a float array."""
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def summary_stats(a: np.ndarray) -> Optional[Dict[str, float]]:
    """Median / IQR / count over finite entries (None when empty)."""
    a = finite(a)
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "iqr": [float(np.percentile(a, 25)), float(np.percentile(a, 75))],
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
    }


def percentiles(a: np.ndarray, ps: Sequence[float]) -> Optional[Dict[str, float]]:
    """Percentile dictionary over finite entries (None when empty)."""
    a = finite(a)
    if a.size == 0:
        return None
    return {f"p{p:g}": float(np.percentile(a, p)) for p in ps}


def ecdf(sorted_vals: np.ndarray, t: np.ndarray, n_total: int) -> np.ndarray:
    """Right-continuous empirical CDF (defective when n_total > responders)."""
    if n_total <= 0:
        return np.zeros_like(t, dtype=float)
    if sorted_vals.size == 0:
        return np.zeros_like(t, dtype=float)
    return np.searchsorted(sorted_vals, t, side="right") / float(n_total)


def miller_bound(
    tgrid: np.ndarray,
    lat_a: np.ndarray,
    n_a: int,
    lat_v: np.ndarray,
    n_v: int,
    soa_vals: Sequence[float],
    soa_weights: Sequence[float],
) -> np.ndarray:
    """Miller (1982) race bound on the wind clock, SOA-corrected.

    ``bound(t) = F_A(t) + sum_k w_k F_V(t - soa_k)`` with ``F_A`` the
    wind-only defective CDF (from wind onset), ``F_V`` the visual-only
    defective CDF (from collision) and ``soa_k`` the wind->collision SOA
    of variant ``k``.  A race model implies ``F_AV <= bound`` everywhere.

    Args:
        tgrid: ``(T,)`` wind-clock time grid in ms.
        lat_a: Sorted finite wind-only latencies (ms).
        n_a: Wind-only population size (denominator of ``F_A``).
        lat_v: Sorted finite visual-only latencies (ms, from collision).
        n_v: Visual-only population size (denominator of ``F_V``).
        soa_vals: Per-variant wind->collision SOAs (ms).
        soa_weights: Per-variant mixture weights (sum to 1).

    Returns:
        ``(T,)`` Miller bound on ``tgrid``.
    """
    assert tgrid.ndim == 1, f"tgrid shape {tgrid.shape}"
    fa = ecdf(lat_a, tgrid, n_a)
    bound = fa.copy()
    weights = np.asarray(soa_weights, dtype=float)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    for soa, w in zip(soa_vals, weights):
        bound = bound + w * ecdf(lat_v, tgrid - soa, n_v)
    # M1: a Miller bound is a probability -- clip at 1 (F_A + F_V can exceed
    # 1 for late t, which is meaningless as a CDF upper envelope).
    bound = np.clip(bound, 0.0, RACE_CDF_CLIP)
    assert bound.shape == tgrid.shape, f"bound shape {bound.shape}"
    return bound


def violation_profile(
    fav: np.ndarray, bound: np.ndarray, tgrid: np.ndarray
) -> Dict[str, Any]:
    """Return Miller violation profile, area and time range.

    Args:
        fav: ``(T,)`` multisensory defective CDF.
        bound: ``(T,)`` Miller bound.
        tgrid: ``(T,)`` time grid.

    Returns:
        Dict with ``area`` (probability*ms), ``max`` (probability),
        ``peak_time_ms`` (time of the maximum violation) and ``time_range``
        (ms or None).
    """
    assert fav.shape == bound.shape == tgrid.shape, (
        f"shape mismatch {fav.shape} {bound.shape} {tgrid.shape}"
    )
    viol = np.maximum(0.0, fav - bound)
    area = float(np.trapezoid(viol, tgrid))
    mask = viol > 0
    peak_idx = int(np.argmax(viol))
    time_range = (
        [float(tgrid[mask].min()), float(tgrid[mask].max())] if mask.any() else None
    )
    return {
        "area": area,
        "max": float(viol.max()),
        "peak_time_ms": float(tgrid[peak_idx]),
        "time_range": time_range,
    }


def permutation_p_value(
    observed: float, null_dist: np.ndarray
) -> Optional[float]:
    """One-sided permutation p-value with the +1 correction (never 0)."""
    if not np.isfinite(observed) or null_dist.size == 0:
        return None
    b = int(null_dist.size)
    ge = int(np.sum(null_dist >= observed))
    return float((ge + 1.0) / (b + 1.0))


def holm_adjust(pvals: Sequence[float]) -> List[Optional[float]]:
    """Holm (1979) step-down family-wise-error-adjusted p-values.

    ``m`` one-sided p-values are ordered ascending; the ``k``-th (1-based)
    is multiplied by ``m - k + 1`` and forced monotone non-decreasing.  NaNs
    (an uninformative test with no permutation null) are excluded from ``m``
    and returned as NaN.  This is the family-wise correction the race family
    declares; Benjamini-Hochberg is deliberately NOT used.

    Args:
        pvals: Raw p-values (a NaN marks an unavailable test).

    Returns:
        Holm-adjusted p-values aligned with ``pvals`` (NaN in, NaN out).
    """
    out: List[Optional[float]] = [None] * len(pvals)
    idx = [i for i, p in enumerate(pvals) if p is not None and np.isfinite(p)]
    m = len(idx)
    if m == 0:
        return [float("nan") if p is None else p for p in pvals]
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for k, i in enumerate(order, start=1):
        adj = min(1.0, (m - k + 1) * float(pvals[i]))
        running = max(running, adj)
        out[i] = running
    for i, p in enumerate(pvals):
        if out[i] is None:
            out[i] = float("nan")
    return out


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size between two 1-D samples.

    ``delta = P(x > y) - P(x < y)`` computed rank-wise in ``O(n log n)``.
    """
    x = np.sort(finite(x))
    y = np.sort(finite(y))
    if x.size == 0 or y.size == 0:
        return float("nan")
    greater = np.sum(np.searchsorted(y, x, side="left"))
    less = np.sum(y.size - np.searchsorted(y, x, side="right"))
    return float((greater - less) / (x.size * y.size))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Pooled-SD Cohen's d between two 1-D samples."""
    x = finite(x)
    y = finite(y)
    if x.size < 2 or y.size < 2:
        return float("nan")
    nx, ny = x.size, y.size
    sp2 = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    if sp2 <= 0:
        return float("nan")
    return float((x.mean() - y.mean()) / np.sqrt(sp2))


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Optional[List[float]]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


# ══════════════════════════════════════════════════════════════════════
# Stratified prefix-cluster bootstrap (B1, true with-replacement)
# ══════════════════════════════════════════════════════════════════════
def prefix_indices(prefix_keys: np.ndarray) -> Dict[Any, np.ndarray]:
    """Map each recording prefix to its trial indices."""
    assert prefix_keys.ndim == 1, f"prefix_keys shape {prefix_keys.shape}"
    out: Dict[Any, np.ndarray] = {}
    for p in np.unique(prefix_keys):
        out[p] = np.where(prefix_keys == p)[0]
    return out


def bootstrap_weights(
    prefix_keys: np.ndarray,
    strata: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Iterator[np.ndarray]:
    """Yield ``n_boot`` ``(N,)`` prefix-multiplicity weight vectors.

    This is a **true with-replacement cluster bootstrap**: within each
    stratum exactly ``len(pool)`` prefixes are drawn WITH replacement, and
    a prefix drawn ``m`` times contributes weight ``m`` to every one of its
    trials.  The multiplicity matters -- collapsing draws to a boolean mask
    (the earlier implementation) counts a twice-drawn prefix once, which
    under-disperses the resample and shrinks every cluster CI.

    Args:
        prefix_keys: ``(N,)`` recording-prefix key per trial.
        strata: ``(N,)`` group label per trial (constant within a prefix).
        n_boot: Number of replicates.
        rng: Seeded generator.

    Yields:
        ``(N,)`` float weight vectors (integer multiplicities per trial).

    Raises:
        ValueError: A prefix spans more than one stratum (fail closed).
    """
    n = prefix_keys.shape[0]
    assert strata.shape == (n,), f"strata shape {strata.shape}"
    by_prefix = prefix_indices(prefix_keys)
    prefix_stratum: Dict[Any, Any] = {}
    for p, idx in by_prefix.items():
        vals = set(np.asarray(strata[idx]).tolist())
        if len(vals) != 1:
            raise ValueError(f"prefix {p!r} spans strata {sorted(vals)}")
        prefix_stratum[p] = vals.pop()
    pools: Dict[Any, List[Any]] = {}
    for p, s in prefix_stratum.items():
        pools.setdefault(s, []).append(p)
    for _ in range(n_boot):
        weights = np.zeros(n, dtype=float)
        for pool in pools.values():
            picks = rng.choice(np.asarray(pool, dtype=object), size=len(pool))
            for p in picks:
                weights[by_prefix[p]] += 1.0
        assert weights.shape == (n,), f"weights shape {weights.shape}"
        yield weights


def expand_weighted(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Repeat ``values`` by integer ``weights`` (weighted-sample expansion)."""
    assert values.shape == weights.shape, (
        f"values/weights shape {values.shape} vs {weights.shape}"
    )
    reps = np.rint(weights).astype(int)
    reps = np.clip(reps, 0, None)
    return np.repeat(values, reps)


def weighted_count(sel: np.ndarray, weights: np.ndarray) -> float:
    """Total resample weight over a boolean selection."""
    assert sel.shape == weights.shape, (
        f"sel/weights shape {sel.shape} vs {weights.shape}"
    )
    return float(np.sum(weights[sel]))


# ══════════════════════════════════════════════════════════════════════
# Race model per TTC group (C2, B2/B3/B4/B5/M1)
# ══════════════════════════════════════════════════════════════════════
def race_strata(cond: np.ndarray, ttc: np.ndarray) -> np.ndarray:
    """Per-trial bootstrap stratum: condition, split by TTC for multisensory."""
    strata = np.empty(cond.shape[0], dtype=object)
    for i in range(cond.shape[0]):
        if cond[i] == "multisensory":
            strata[i] = f"multi_{int(ttc[i])}"
        else:
            strata[i] = str(cond[i])
    return strata


def _variant_soa(
    m_multi: np.ndarray, ttc: np.ndarray, w_on: np.ndarray, v_col: np.ndarray,
    dt_ms: float, v: float,
) -> float:
    """Median wind->collision SOA (ms) for one multisensory TTC variant."""
    sel = m_multi & (ttc == v)
    d = finite((v_col[sel] - w_on[sel]) * dt_ms)
    return float(np.median(d)) if d.size else 0.0


def run_race_model(
    cond: np.ndarray,
    ttc: np.ndarray,
    latency: np.ndarray,
    w_on: np.ndarray,
    v_col: np.ndarray,
    prefix_keys: np.ndarray,
    eligible: np.ndarray,
    dt_ms: float,
    n_boot: int,
    rng: np.random.Generator,
    latency_label: Optional[np.ndarray] = None,
    n_perm: int = 2000,
    perm_rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """SOA-corrected Miller test per TTC variant and pooled (D1/D2/D5).

    Each multisensory TTC variant (each prefix has exactly one TTC) is a
    separate between-group comparison against the SAME eligible wind_only
    and visual_only populations.  Only ELIGIBLE trials (stationary at the
    D1 trigger) enter any CDF; ``n_eligible`` records the per-condition
    denominators and the excluded counts are reported.

    Violation areas carry **true with-replacement** prefix-cluster bootstrap
    CIs (B1).  PRIMARY race inference (D2) is, per TTC variant and pooled:

    * a prefix-cluster **bootstrap CI of the violation area** -- the same
      with-replacement cluster bootstrap that produced the point estimate;
    * a **prefix-level permutation test** of the AV vs A group labels.  The
      single-modality permutation of wind vs visual labels used in round 2 is
      NOT a race-model null (a race model puts no restriction on which
      modality wins a given trial), so it is removed as primary.  The
      exchangeable unit is the whole prefix; the multisensory variant
      prefixes are the AV group and the wind_only prefixes the A group, and
      the null statistic is ``area(max(0, F_AV - bound))``.  Because under
      D1 the visual-only term is small, ``bound`` is close to ``F_A``, so
      this permutation tests AV facilitation beyond the wind-alone CDF (the
      boundary case of the race null), which is stated explicitly.

    No pseudo-p (``1 - frac > 0``) and no BH-FDR on such a pseudo-p are
    reported.  The one-sided prefix-permutation p-values are corrected for
    multiplicity with **Holm** (X2) over the declared race family (pooled +
    the 4 TTC variants); raw and Holm-adjusted p are both reported.  Power
    (D5) is reported for the FACILITATION contrast only;
    no signed one-sample race power is emitted (a violation area is a
    non-negative quantity whose sign is uninformative).  A simulation-based
    power for the permutation test is estimated from the cluster bootstrap
    when ``n_boot`` permits.

    Args:
        cond: ``(N,)`` condition per trial.
        ttc: ``(N,)`` target TTC per trial (ms).
        latency: ``(N,)`` PRIMARY eligible latency, ms.
        w_on: ``(N,)`` wind onset frame.
        v_col: ``(N,)`` visual collision frame.
        prefix_keys: ``(N,)`` recording-prefix key.
        eligible: ``(N,)`` D1 eligibility mask.
        dt_ms: Frame interval (ms).
        n_boot: Prefix-cluster bootstrap replicates.
        rng: Seeded generator (bootstrap).
        latency_label: Optional s1 labeling-responder latency for the
            sensitivity race (B3).
        n_perm: Prefix-label permutation replicates.
        perm_rng: Seeded generator for the permutation test.

    Returns:
        Race-model summary dict.
    """
    m_wind = cond == "wind_only"
    m_vis = cond == "visual_only"
    m_multi = cond == "multisensory"
    el = eligible
    variants = np.unique(ttc[m_multi])
    tgrid = np.arange(0.0, RESPONSE_WINDOW_MS + dt_ms, dt_ms)
    if perm_rng is None:
        perm_rng = np.random.default_rng(1234)

    soa_vals = [
        _variant_soa(m_multi, ttc, w_on, v_col, dt_ms, float(v)) for v in variants
    ]
    counts = np.array([int((m_multi & (ttc == v) & el).sum()) for v in variants], float)
    weights = counts / counts.sum() if counts.sum() else counts
    strata = race_strata(cond, ttc)

    def cdf_terms(
        w: np.ndarray, v: Optional[float]
    ) -> Tuple[np.ndarray, int, np.ndarray, int, np.ndarray, int, np.ndarray]:
        """Return (lat_av, n_av, lat_a, n_a, lat_v, n_v, variant_weights)."""
        if v is None:
            av_sel = m_multi & el
            wv = np.array(
                [weighted_count(m_multi & (ttc == vv) & el, w) for vv in variants]
            )
            wv = wv / wv.sum() if wv.sum() else weights
            sv = soa_vals
        else:
            av_sel = m_multi & (ttc == v) & el
            wv = np.array([1.0])
            sv = [_variant_soa(m_multi, ttc, w_on, v_col, dt_ms, float(v))]
        lat_av = expand_weighted(latency[av_sel], w[av_sel])
        n_av = int(round(weighted_count(av_sel, w)))
        lat_a = expand_weighted(latency[m_wind & el], w[m_wind & el])
        n_a = int(round(weighted_count(m_wind & el, w)))
        lat_v = expand_weighted(latency[m_vis & el], w[m_vis & el])
        n_v = int(round(weighted_count(m_vis & el, w)))
        return (
            np.sort(finite(lat_av)), n_av,
            np.sort(finite(lat_a)), n_a,
            np.sort(finite(lat_v)), n_v,
            sv, wv,
        )

    def area_for(w: np.ndarray, v: Optional[float]) -> Dict[str, Any]:
        lat_av, n_av, lat_a, n_a, lat_v, n_v, sv, wv = cdf_terms(w, v)
        fav = ecdf(lat_av, tgrid, n_av)
        bound = miller_bound(tgrid, lat_a, n_a, lat_v, n_v, sv, wv)
        prof = violation_profile(fav, bound, tgrid)
        return {
            "n_av": n_av,
            "cdf_av_at_2s": float(fav[-1]),
            "bound_at_2s": float(bound[-1]),
            "violation": prof,
        }

    unit = np.ones(cond.shape[0], dtype=float)  # observed weights (1 per trial)
    out: Dict[str, Any] = {
        "design": (
            "A=wind_only (from wind onset), V=visual_only (from collision), "
            "AV=multisensory (from wind onset); V trigger shifted by the "
            "per-variant wind->collision SOA."
        ),
        "reference_per_condition": {
            "wind_only": "wind onset",
            "multisensory": "wind onset",
            "visual_only": "visual collision",
        },
        "inclusion_rule": (
            "D1 primary: eligible only if |Y| < 10 cm/s throughout "
            "[trigger-250 ms, trigger]; latency = first frame STRICTLY after "
            "the trigger with |Y| >= 10 cm/s in [trigger, trigger + 2000 ms], "
            "trigger = wind onset (wind-containing) / collision (visual_only)"
        ),
        "nonresponder_handling": (
            "defective CDF: no |Y|>=10 cm/s in the post-trigger window "
            "counts as a non-responder in the denominator (eligible only)"
        ),
        "n_eligible_by_condition": {
            c: {
                "n": int((cond == c).sum()),
                "n_eligible": int(((cond == c) & el).sum()),
                "n_excluded": int(((cond == c) & ~el).sum()),
            }
            for c in ("visual_only", "wind_only", "multisensory")
        },
        "soa_wind_to_collision_ms": [float(x) for x in soa_vals],
        "variant_weights": [float(x) for x in weights],
        "cdf_at_2s": {
            "F_wind_only": float(
                ecdf(np.sort(finite(latency[m_wind & el])), tgrid,
                     int((m_wind & el).sum()))[-1]
            ),
            "F_visual_only": float(
                ecdf(np.sort(finite(latency[m_vis & el])), tgrid,
                     int((m_vis & el).sum()))[-1]
            ),
            "F_multisensory": float(
                ecdf(np.sort(finite(latency[m_multi & el])), tgrid,
                     int((m_multi & el).sum()))[-1]
            ),
        },
        "visual_only_response": {
            "n_eligible_visual_only": int((m_vis & el).sum()),
            "n_visual_responders": int(
                np.isfinite(latency[m_vis & el]).sum()
            ),
            "note": (
                "D1: the visual-only CDF reflects post-collision responses of "
                "stationary animals (first |Y|>=10 cm/s strictly after "
                "collision). If F_V is ~0 the Miller bound reduces to F_A and "
                "the test becomes AV vs A facilitation beyond the wind-alone "
                "CDF (D2 boundary case)"
            ),
        },
        "pooled": {"point": area_for(unit, None)["violation"]},
        "per_variant": {},
        "assumptions": (
            "context invariance (single-modality CDFs transferable to the AV "
            "context) and the same-population assumption (the SAME wind_only "
            "and visual_only populations are the comparators for every "
            "variant). Condition is confounded with recording prefix, so a "
            "violation cannot separate coactivation from session effects; a "
            "null result is NOT evidence against integration."
        ),
        "bootstrap_unit": "recording_prefix (stratified by group, with replacement)",
    }
    boot_pooled: List[float] = []
    for w in bootstrap_weights(prefix_keys, strata, n_boot, rng):
        boot_pooled.append(area_for(w, None)["violation"]["area"])
    boot_pooled_arr = np.asarray(boot_pooled)
    out["pooled"]["bootstrap_mean"] = float(boot_pooled_arr.mean())
    out["pooled"]["ci95"] = [
        float(np.percentile(boot_pooled_arr, 2.5)),
        float(np.percentile(boot_pooled_arr, 97.5)),
    ]
    out["pooled"]["n_prefixes"] = int(len({str(p) for p in prefix_keys[m_multi]}))

    for v in variants:
        point = area_for(unit, float(v))
        boots = [
            area_for(w, float(v))["violation"]["area"]
            for w in bootstrap_weights(prefix_keys, strata, n_boot, rng)
        ]
        arr = np.asarray(boots)
        n_pref_v = int(len({str(p) for p in prefix_keys[m_multi & (ttc == v)]}))
        out["per_variant"][f"ttc_{int(v)}"] = {
            "n_av": point["n_av"],
            "n_prefixes": n_pref_v,
            # D4: a variant with < 5 prefixes is flagged uninformative.
            "informative": bool(n_pref_v >= 5),
            "soa_wind_to_collision_ms": _variant_soa(
                m_multi, ttc, w_on, v_col, dt_ms, float(v)
            ),
            "violation_area": point["violation"]["area"],
            "violation_max": point["violation"]["max"],
            "violation_peak_time_ms": point["violation"]["peak_time_ms"],
            "violation_time_range_ms": point["violation"]["time_range"],
            "cdf_av_at_2s": point["cdf_av_at_2s"],
            "bound_at_2s": point["bound_at_2s"],
            "bootstrap_mean": float(arr.mean()),
            "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        }
    out["pooled"]["violation_peak_time_ms"] = out["pooled"]["point"]["peak_time_ms"]

    # ── D2: prefix-level permutation test of the violation area ─────────
    out["permutation_test"] = prefix_permutation_test(
        cond, ttc, latency, w_on, v_col, prefix_keys, el, dt_ms,
        n_perm, perm_rng,
    )

    # ── Sensitivity: SOA = 0 (same-clock) pooled area ──────────────────
    lat_av = np.sort(finite(latency[m_multi & el]))
    fav0 = ecdf(lat_av, tgrid, int((m_multi & el).sum()))
    bound0 = miller_bound(
        tgrid,
        np.sort(finite(latency[m_wind & el])), int((m_wind & el).sum()),
        np.sort(finite(latency[m_vis & el])), int((m_vis & el).sum()),
        [0.0], [1.0],
    )
    out["sensitivity_soa0_pooled_area"] = violation_profile(fav0, bound0, tgrid)["area"]
    out["n_multisensory_prefixes"] = int(
        len({str(p) for p in prefix_keys[m_multi]})
    )
    out["n_prefixes_per_variant"] = {
        f"ttc_{int(v)}": int(len({str(p) for p in prefix_keys[m_multi & (ttc == v)]}))
        for v in variants
    }

    # ── D5: race power.  The signed one-sample power is DROPPED (a
    # violation area is non-negative; a signed mean power is uninformative).
    # Only a simulation-based power for the D2 permutation test is reported,
    # estimated from the cluster-bootstrap null draws when n_boot permits.
    out["power_analysis"] = _race_power(
        boot_pooled_arr, out["permutation_test"]["pooled"], n_perm
    )

    # ── B3 s1: labeling-responder V term ───────────────────────────────
    if latency_label is not None:
        lat2 = np.asarray(latency_label, dtype=float)
        lat_a2 = np.sort(finite(lat2[m_wind & el]))
        na2 = int((m_wind & el).sum())
        lat_v2 = np.sort(finite(lat2[m_vis & el]))
        nv2 = int((m_vis & el).sum())
        fav2 = ecdf(np.sort(finite(lat2[m_multi & el])), tgrid,
                    int((m_multi & el).sum()))
        bound2 = miller_bound(tgrid, lat_a2, na2, lat_v2, nv2, soa_vals, weights)
        out["sensitivity_labeling_responder"] = {
            "note": (
                "s1: response criterion = labeling sustained > 5 cm/s for "
                "250 ms, under the SAME D1 eligibility gate as the primary "
                "resp; V term uses the same criterion, so F_V is nonzero"
            ),
            "F_visual_only_at_2s": float(ecdf(lat_v2, tgrid, nv2)[-1]),
            "n_visual_responders": int(np.isfinite(lat2[m_vis & el]).sum()),
            "violation_area": violation_profile(fav2, bound2, tgrid)["area"],
        }
    return out


def _race_power(
    boot_areas: np.ndarray, perm_test: Dict[str, Any], n_perm: int
) -> Dict[str, Any]:
    """D5: simulation-based power for the D2 permutation test.

    No signed one-sample race power is reported.  The power for a one-sided
    prefix-permutation test at the observed alternative is approximated by
    the fraction of cluster-bootstrap violation areas that exceed the
    permutation null's 95th percentile -- i.e. how often a resampled corpus
    of the same size would reject.  This is an order-of-magnitude planning
    figure; the bootstrap draws the observed clusters, so it does not model
    a true alternative and cannot be read as a confirmatory power.

    Args:
        boot_areas: ``(B,)`` pooled violation areas from the cluster
            bootstrap.
        perm_test: The D2 ``permutation_test`` dict.
        n_perm: Permutation replicates (for the caveat text).

    Returns:
        Dict describing the (non-)availability of a race power estimate.
    """
    p95 = perm_test.get("null_p95")
    if p95 is None or boot_areas.size == 0:
        return {
            "race_power_estimated": False,
            "note": (
                "race power not estimated: no permutation null (all prefixes "
                "in one group) or no bootstrap draws"
            ),
        }
    reject_rate = float(np.mean(boot_areas > float(p95)))
    return {
        "race_power_estimated": True,
        "method": (
            "simulation from the cluster-bootstrap violation-area distribution: "
            "fraction of bootstrap draws above the permutation null's 95th "
            "percentile"
        ),
        "n_boot": int(boot_areas.size),
        "n_perm": int(n_perm),
        "null_p95_area": float(p95),
        "power_approx": reject_rate,
        "caveat": (
            "the bootstrap resamples the OBSERVED clusters, so this is a "
            "self-referential planning figure, not a power against a "
            "specified alternative; the signed one-sample race power is "
            "deliberately NOT reported (a violation area is non-negative)"
        ),
    }


def prefix_permutation_test(
    cond: np.ndarray,
    ttc: np.ndarray,
    latency: np.ndarray,
    w_on: np.ndarray,
    v_col: np.ndarray,
    prefix_keys: np.ndarray,
    eligible: np.ndarray,
    dt_ms: float,
    n_perm: int,
    perm_rng: np.random.Generator,
) -> Dict[str, Any]:
    """Prefix-level permutation test of the race violation (D2).

    PRIMARY race inference.  The exchangeable unit is the RECORDING PREFIX.
    Under the null the group label is exchangeable across prefixes, so the
    test permutes the AV vs A group labels across prefixes (multisensory
    variant prefixes vs wind_only prefixes) and recomputes the statistic
    ``area(max(0, F_AV - bound))`` with the relabeled groups.

    The round-2 test permuted the SINGLE-MODALITY label (wind vs visual)
    across single-modality prefixes, holding the multisensory CDF fixed.
    That is NOT a race-model null: a race model places no restriction on
    which modality wins a trial, so relabeling wind vs visual prefixes does
    not sample from the race hypothesis.  It is removed as primary.

    Because under D1 the visual-only term is small, the Miller bound is
    close to ``F_A`` (the wind-alone CDF), so ``area(max(0, F_AV - bound))``
    is essentially the AV-vs-A facilitation area -- this is the boundary
    case of the race null, stated explicitly in ``note``.

    Reported per TTC variant AND pooled.  For a variant the AV group is that
    variant's prefixes and the A group is all wind_only prefixes; the visual
    term (if any) uses that variant's SOA.

    Args:
        cond, ttc, latency, w_on, v_col, prefix_keys, eligible, dt_ms:
            As in :func:`run_race_model`.
        n_perm: Permutation replicates.
        perm_rng: Seeded generator.

    Returns:
        Dict keyed by ``pooled`` and ``ttc_<v>``; each entry carries
        ``observed_area_ms``, ``null_mean``, ``null_p95``, ``p_one_sided``,
        ``n_perm``, ``n_prefixes`` and a shared ``note``.
    """
    m_multi = cond == "multisensory"
    m_wind = cond == "wind_only"
    m_vis = cond == "visual_only"
    el = eligible
    tgrid = np.arange(0.0, RESPONSE_WINDOW_MS + dt_ms, dt_ms)
    variants = np.unique(ttc[m_multi])

    # Fixed visual term (D1: post-collision responders).  It is small and is
    # held constant across the AV-vs-A relabeling.
    lat_v = np.sort(finite(latency[m_vis & el]))
    n_v = int((m_vis & el).sum())

    prefixes = np.array(sorted(set(prefix_keys.tolist())), dtype=object)
    wind_prefixes = np.array(
        [p for p in prefixes if np.any(m_wind & (prefix_keys == p))], dtype=object
    )
    in_wind = np.isin(prefix_keys, wind_prefixes)
    lat_a_obs = np.sort(finite(latency[m_wind & el]))
    n_a_obs = int((m_wind & el).sum())

    note = (
        "D2 PRIMARY: permutes the AV vs A group label across whole prefixes "
        "(multisensory variant prefixes vs wind_only prefixes), recomputing "
        "BOTH CDFs -- F_AV from the relabeled AV group and F_A from the "
        "relabeled A group -- and the statistic area(max(0, F_AV - bound)). "
        "With the D1 visual term small the bound is close to F_A, so this "
        "tests AV facilitation beyond the wind-alone CDF (the boundary case "
        "of the race null). Valid only under prefix exchangeability, which "
        "the condition-prefix confound does not guarantee."
    )

    def permutation_for(
        av_prefixes: np.ndarray, soa_vals: np.ndarray, wv: np.ndarray
    ) -> Dict[str, Any]:
        """One AV-vs-A prefix permutation for a given AV prefix set.

        The pool is the AV prefixes plus the wind_only prefixes; a
        permutation splits it into a relabeled AV group and a relabeled A
        group, and BOTH CDFs are recomputed -- ``F_AV`` from the relabeled AV
        group and the bound's ``F_A`` from the relabeled A group (the visual
        term is small and held fixed).  Holding ``F_AV`` fixed (the round-2
        style) would not be a label permutation of the AV-vs-A contrast.
        """
        n_av_pref = int(av_prefixes.size)
        if n_av_pref == 0 or wind_prefixes.size == 0:
            return {"observed_area_ms": None, "p_one_sided": None,
                    "n_perm": int(n_perm), "n_prefixes": n_av_pref}
        in_av = np.isin(prefix_keys, av_prefixes)
        fav_obs = ecdf(np.sort(finite(latency[el & in_av])), tgrid,
                       int((el & in_av).sum()))
        bound_obs = miller_bound(tgrid, lat_a_obs, n_a_obs, lat_v, n_v,
                                 soa_vals, wv)
        observed = float(np.trapezoid(np.maximum(0.0, fav_obs - bound_obs), tgrid))
        pool = np.concatenate([av_prefixes, wind_prefixes])
        null = np.empty(n_perm, dtype=float)
        for b in range(n_perm):
            perm = perm_rng.permutation(pool)
            av_set = set(perm[: n_av_pref].tolist())
            a_set = set(perm[n_av_pref:].tolist())
            in_av_p = np.array([p in av_set for p in prefix_keys], dtype=bool)
            in_a_p = np.array([p in a_set for p in prefix_keys], dtype=bool)
            fav_p = ecdf(np.sort(finite(latency[el & in_av_p])), tgrid,
                         int((el & in_av_p).sum()))
            lat_a_p = np.sort(finite(latency[el & in_a_p]))
            n_a_p = int((el & in_a_p).sum())
            bound_p = miller_bound(tgrid, lat_a_p, n_a_p, lat_v, n_v,
                                   soa_vals, wv)
            null[b] = float(np.trapezoid(np.maximum(0.0, fav_p - bound_p), tgrid))
        return {
            "observed_area_ms": observed,
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p_one_sided": permutation_p_value(observed, null),
            "n_perm": int(n_perm),
            "n_prefixes": n_av_pref,
        }

    out: Dict[str, Any] = {"note": note}

    # Pooled AV: all multisensory prefixes; visual term is the SOA mixture.
    pooled_soa = np.array(
        [_variant_soa(m_multi, ttc, w_on, v_col, dt_ms, float(v)) for v in variants]
    )
    pooled_w = np.array(
        [int((m_multi & (ttc == v) & el).sum()) for v in variants], dtype=float
    )
    pooled_w = pooled_w / pooled_w.sum() if pooled_w.sum() else pooled_w
    multi_prefixes = np.array(
        sorted({str(p) for p in prefix_keys[m_multi]}), dtype=object
    )
    out["pooled"] = permutation_for(multi_prefixes, pooled_soa, pooled_w)

    # Per-variant AV: that variant's prefixes; visual term = variant SOA.
    for v in variants:
        av_pref = np.array(
            sorted({str(p) for p in prefix_keys[m_multi & (ttc == v)]}), dtype=object
        )
        sv = np.array([_variant_soa(m_multi, ttc, w_on, v_col, dt_ms, float(v))])
        wv1 = np.array([1.0])
        out[f"ttc_{int(v)}"] = permutation_for(av_pref, sv, wv1)

    # ── X2: Holm correction over the DECLARED race family (pooled + the 4
    # TTC variants).  The family is declared before the p-values are read.
    family_keys = ["pooled"] + [f"ttc_{int(v)}" for v in variants]
    raw = [out[k].get("p_one_sided") for k in family_keys]
    adj = holm_adjust(raw)
    for k, adj_p in zip(family_keys, adj):
        out[k]["p_one_sided_holm"] = float(adj_p)
    out["holm_family"] = {
        "correction": "Holm step-down (family-wise error rate)",
        "family": "the race violation family: pooled + the 4 TTC variants",
        "m": int(sum(1 for p in raw if p is not None)),
        "note": (
            "Holm over the declared 5-test race family (pooled + 4 TTC "
            "variants) on the prefix-permutation one-sided p-values; raw "
            "`p_one_sided` and adjusted `p_one_sided_holm` are both reported. "
            "A variant with no permutation null is excluded from m and its "
            "Holm p is NaN."
        ),
    }
    return out


# ══════════════════════════════════════════════════════════════════════
# Discrete escape modes (descriptive; negative item)
# ══════════════════════════════════════════════════════════════════════
def run_discrete_modes(
    cond: np.ndarray, latency: np.ndarray, kin: np.ndarray, rng: np.random.Generator
) -> Dict[str, Any]:
    """Gaussian-mixture sweep over responder kinematics (descriptive)."""
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    rs = np.where(np.isfinite(latency) & np.all(np.isfinite(kin), axis=1))[0]
    feats = np.column_stack([latency[rs], kin[rs]])
    mu = feats.mean(0)
    sd = feats.std(0)
    sd[sd == 0] = 1.0
    z = (feats - mu) / sd
    assert z.ndim == 2, f"feature matrix shape {z.shape}"

    bic: Dict[int, Dict[str, float]] = {}
    for k in range(1, 5):
        gm = GaussianMixture(
            k, covariance_type="full", n_init=4, random_state=0, max_iter=400
        ).fit(z)
        bic[k] = {
            "bic": float(gm.bic(z)),
            "aic": float(gm.aic(z)),
            "mean_max_posterior": float(gm.predict_proba(z).max(1).mean()),
        }
    best_k = min(bic, key=lambda k: bic[k]["bic"])
    gm = GaussianMixture(
        best_k, covariance_type="full", n_init=8, random_state=0, max_iter=500
    ).fit(z)
    clus = gm.predict(z)
    ari: List[float] = []
    for b in range(GMM_N_BOOT):
        idx = rng.choice(z.shape[0], size=z.shape[0], replace=True)
        g2 = GaussianMixture(
            best_k, covariance_type="full", n_init=4, random_state=b, max_iter=300
        ).fit(z[idx])
        ari.append(adjusted_rand_score(clus[idx], g2.predict(z[idx])))
    props: Dict[str, Any] = {}
    for c in range(best_k):
        s = clus == c
        props[f"component_{c}"] = {
            "n": int(s.sum()),
            "latency_ms": summary_stats(feats[s, 0]),
            "peak_speed": summary_stats(feats[s, 1]),
            "peak_accel": summary_stats(feats[s, 2]),
            "time_to_peak_ms": summary_stats(feats[s, 3]),
            "bout_dur_ms": summary_stats(feats[s, 4]),
            "n_by_condition": {
                cname: int(np.sum(cond[rs][s] == cname)) for cname in CONDITIONS
            },
        }
    return {
        "features": [
            "latency_ms", "peak_speed", "peak_accel", "time_to_peak_ms",
            "bout_dur_ms",
        ],
        "n_responders": int(z.shape[0]),
        "bic_sweep": {str(k): v for k, v in bic.items()},
        "best_k_bic": int(best_k),
        "delta_bic_1_minus_best": float(bic[1]["bic"] - bic[best_k]["bic"]),
        "ari_bootstrap_mean": float(np.mean(ari)),
        "ari_bootstrap_min": float(np.min(ari)),
        "component_properties": props,
        "caveats": (
            "in-sample fit (no held-out split); no prefix grouping, so the "
            "ARI bootstrap resamples trials and cannot see the recording-"
            "prefix clustering; the feature matrix MIXES all three conditions"
        ),
        "verdict": (
            "descriptive only: components align with condition (see "
            "n_by_condition) and ARI is moderate, so this is NOT evidence "
            "for a condition-independent discrete escape-mode taxonomy"
        ),
    }


# ══════════════════════════════════════════════════════════════════════
# Evidence accumulation (C6c)
# ══════════════════════════════════════════════════════════════════════
def fit_latency_distribution(x: np.ndarray) -> Dict[str, Any]:
    """Fit normal / log-normal / exponential / gamma / ex-Gaussian by MLE.

    Every candidate is fitted by maximum likelihood on the SAME sample and
    scored with AIC/BIC from its own log-likelihood and its TRUE number of
    free parameters (M4: like-for-like comparison).  The log-normal fixes
    ``loc = 0``, so it carries 2 free parameters, not 3.
    """
    from scipy import stats as sps

    x = finite(x)
    if x.size < 20:
        return {"n": int(x.size)}
    n = int(x.size)
    out: Dict[str, Any] = {
        "n": n,
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "skew": float(sps.skew(x)),
        "excess_kurtosis": float(sps.kurtosis(x)),
    }

    def scored(ll: float, k: int, **extra: Any) -> Dict[str, Any]:
        return {
            "loglik": float(ll),
            "n_params": int(k),
            "aic": float(2 * k - 2 * ll),
            "bic": float(k * np.log(n) - 2 * ll),
            **extra,
        }

    m, s = x.mean(), x.std(ddof=1)
    out["normal"] = scored(float(np.sum(sps.norm.logpdf(x, m, s))), 2)
    try:
        sh, lo, sc = sps.lognorm.fit(x, floc=0)
        out["lognormal"] = scored(
            float(np.sum(sps.lognorm.logpdf(x, sh, lo, sc))), 2,
            params=[float(sh), float(sc)],
        )
    except Exception:  # pragma: no cover - numerical failure on odd samples
        out["lognormal"] = None
    try:
        out["exponential"] = scored(
            float(np.sum(sps.expon.logpdf(x, 0.0, x.mean()))), 1,
            params=[float(x.mean())],
        )
    except Exception:  # pragma: no cover
        out["exponential"] = None
    try:
        a, lo3, sc3 = sps.gamma.fit(x, floc=0)
        out["gamma"] = scored(
            float(np.sum(sps.gamma.logpdf(x, a, lo3, sc3))), 2,
            params=[float(a), float(sc3)],
        )
    except Exception:  # pragma: no cover
        out["gamma"] = None
    try:
        k, lo2, sc2 = sps.exponnorm.fit(x)
        out["exgaussian"] = scored(
            float(np.sum(sps.exponnorm.logpdf(x, k, lo2, sc2))), 3,
            tau_ms=float(k * sc2), params=[float(k), float(lo2), float(sc2)],
        )
    except Exception:  # pragma: no cover
        out["exgaussian"] = None
    names = ("normal", "lognormal", "exponential", "gamma", "exgaussian")
    cands = {k: out[k] for k in names if out.get(k)}
    out["best_by_bic"] = min(cands, key=lambda k: cands[k]["bic"]) if cands else None
    out["best_by_aic"] = min(cands, key=lambda k: cands[k]["aic"]) if cands else None
    return out


def run_accumulation(
    cond: np.ndarray, ttc: np.ndarray, latency: np.ndarray
) -> Dict[str, Any]:
    """Ex-Gaussian fits per group and the mean-SD trend across TTC variants."""
    from scipy import stats as sps

    m_multi = cond == "multisensory"
    acc: Dict[str, Any] = {
        "wind_only": fit_latency_distribution(latency[cond == "wind_only"]),
        "multisensory": fit_latency_distribution(latency[m_multi]),
        "visual_only_from_collision": fit_latency_distribution(
            latency[cond == "visual_only"]
        ),
    }
    groups: List[Dict[str, Any]] = []
    for v in np.unique(ttc[m_multi]):
        sel = m_multi & (ttc == v)
        x = finite(latency[sel])
        acc[f"multisensory_ttc_{int(v)}"] = fit_latency_distribution(x)
        if x.size >= 20:
            groups.append(
                {
                    "ttc_ms": int(v),
                    "n": int(x.size),
                    "mean": float(x.mean()),
                    "sd": float(x.std(ddof=1)),
                }
            )
    rho = pval = float("nan")
    if len(groups) >= 3:
        rho, pval = sps.spearmanr(
            [g["mean"] for g in groups], [g["sd"] for g in groups]
        )
    return {
        "per_condition": acc,
        "strength_groups": groups,
        "mean_sd_spearman": (
            {"rho": float(rho), "p": float(pval), "n_groups": len(groups)}
            if len(groups) >= 3
            else None
        ),
        "interpretation": (
            "right skew + ex-Gaussian best-fit + mean-SD co-increase is the "
            "accumulation-to-threshold signature; 'suggestive' at 4 points"
        ),
    }


# ══════════════════════════════════════════════════════════════════════
# Looming trigger rule (descriptive; negative item)
# ══════════════════════════════════════════════════════════════════════
def run_looming_trigger(
    cond: np.ndarray, ttc: np.ndarray, latency: np.ndarray, w_on: np.ndarray,
    v_col: np.ndarray, dt_ms: float,
) -> Dict[str, Any]:
    """Escape time relative to collision across the four TTC variants."""
    from scipy import stats as sps

    m_multi = cond == "multisensory"
    per_variant: List[Dict[str, Any]] = []
    for v in np.unique(ttc[m_multi]):
        sel = m_multi & (ttc == v)
        lat_from_col = np.array(
            [
                (np.nan if not np.isfinite(latency[i]) else latency[i] - (v_col[i] - w_on[i]) * dt_ms)
                for i in np.where(sel)[0]
            ]
        )
        per_variant.append(
            {
                "ttc_ms": int(v),
                "l_over_v_proxy_ms": float(-v - 50.0),
                "n_responders": int(np.sum(np.isfinite(lat_from_col))),
                "escape_minus_collision_ms": summary_stats(lat_from_col),
            }
        )
    lv = np.array([r["l_over_v_proxy_ms"] for r in per_variant])
    med = np.array(
        [
            r["escape_minus_collision_ms"]["median"]
            if r["escape_minus_collision_ms"]
            else np.nan
            for r in per_variant
        ]
    )
    ok = np.isfinite(med)
    fit = None
    if ok.sum() >= 2:
        sl, ic, r, pval, se = sps.linregress(lv[ok], med[ok])
        fit = {
            "slope": float(sl),
            "intercept": float(ic),
            "r": float(r),
            "p": float(pval),
            "se": float(se),
            "n_points": int(ok.sum()),
        }
    return {
        "available_variants": [int(v) for v in np.unique(ttc[m_multi])],
        "per_variant": per_variant,
        "linear_fit_median_escape_vs_lv": fit,
        "note": (
            "escape is wind-locked in multisensory, so escape time relative to "
            "collision is dominated by the wind, not the looming angle; a "
            "fixed-angular-size threshold (LGMD/DCMD-like) is a kinematic "
            "identity here and is NOT identifiable. Only ONE collision "
            "geometry exists. The wind-locked timing is read from "
            "per_variant[*].escape_minus_collision_ms (median), not hardcoded."
        ),
    }


def run_reliability_status() -> Dict[str, str]:
    """Report why cue-reliability weighting is untestable in this corpus."""
    return {
        "needed": (
            "TTC=0 trials or deliberate cue-conflict conditions to vary "
            "single-cue reliability and measure weighting"
        ),
        "present": (
            "4 TTC variants (-373/-308/-261/-225 ms); no TTC=0; no "
            "cue-conflict or reliability manipulation"
        ),
        "approximation": (
            "the variants change the wind-visual SOA but keep wind the sole "
            "trigger"
        ),
        "verdict": "NOT TESTABLE in this corpus",
    }


# ══════════════════════════════════════════════════════════════════════
# Covariates + mixed model + power (C5)
# ══════════════════════════════════════════════════════════════════════
def prefix_covariates(
    corpus: Dict[str, Any], events: Dict[str, np.ndarray], prefix_keys: np.ndarray
) -> Dict[str, Any]:
    """Per-prefix pre-stimulus baseline covariates."""
    y, ref = corpus["y"], events["ref"]
    n = corpus["n_trials"]
    rest = np.full(n, np.nan)
    spont = np.full(n, np.nan)
    for i in range(n):
        if ref[i] <= 1:
            continue
        base = np.abs(y[i][: ref[i]])
        rest[i] = float(np.median(base))
        spont[i] = float(np.mean(base > SPONTANEOUS_THRESHOLD_CMS))
    out: Dict[str, Any] = {}
    for p in np.unique(prefix_keys):
        idx = prefix_keys == p
        out[str(p)] = {
            "n_trials": int(idx.sum()),
            "resting_speed_cms": float(np.nanmedian(rest[idx])),
            "spontaneous_move_rate": float(np.nanmean(spont[idx])),
        }
    return out


def random_intercept_lmm(
    y: np.ndarray, x_fixed: np.ndarray, groups: np.ndarray
) -> Dict[str, Any]:
    """Fit ``y = X beta + b_group + eps`` (random intercept) by REML.

    Args:
        y: ``(n,)`` response.
        x_fixed: ``(n, p)`` fixed-effect design (intercept in column 0).
        groups: ``(n,)`` integer group codes.

    Returns:
        Dict with ``beta``, ``se``, ``z``, ``p``, ``sigma_b``, ``sigma_e``,
        ``n_groups``, ``n_obs``.
    """
    from scipy import stats as sps
    from scipy.optimize import minimize

    y = np.asarray(y, dtype=float)
    x_fixed = np.asarray(x_fixed, dtype=float)
    groups = np.asarray(groups)
    n, p = x_fixed.shape
    assert y.shape == (n,), f"y shape {y.shape}"
    codes, inverse = np.unique(groups, return_inverse=True)
    blocks: List[Tuple[np.ndarray, np.ndarray]] = []
    for g in range(codes.size):
        idx = np.where(inverse == g)[0]
        blocks.append((x_fixed[idx], y[idx]))

    def neg_reml(theta: np.ndarray) -> float:
        sb2 = np.exp(2.0 * theta[0])
        se2 = np.exp(2.0 * theta[1])
        a = np.zeros((p, p))
        b = np.zeros(p)
        ytvinvy = 0.0
        logdet_v = 0.0
        for xb, yb in blocks:
            nj = xb.shape[0]
            cj = sb2 / (se2 + nj * sb2)
            s = xb.sum(axis=0)
            t = yb.sum()
            a += (xb.T @ xb - cj * np.outer(s, s)) / se2
            b += (xb.T @ yb - cj * s * t) / se2
            ytvinvy += (yb @ yb - cj * t * t) / se2
            logdet_v += (nj - 1) * np.log(se2) + np.log(se2 + nj * sb2)
        sign, logdet_a = np.linalg.slogdet(a)
        if sign <= 0:
            return 1e12
        beta = np.linalg.solve(a, b)
        resid = ytvinvy - beta @ b
        return 0.5 * (logdet_v + logdet_a + max(resid, 0.0))

    init = np.array([np.log(max(y.std(ddof=1) * 0.5, 1e-3)),
                     np.log(max(y.std(ddof=1) * 0.5, 1e-3))])
    res = minimize(neg_reml, init, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
    theta = res.x
    sb2 = float(np.exp(2.0 * theta[0]))
    se2 = float(np.exp(2.0 * theta[1]))
    a = np.zeros((p, p))
    b = np.zeros(p)
    for xb, yb in blocks:
        nj = xb.shape[0]
        cj = sb2 / (se2 + nj * sb2)
        s = xb.sum(axis=0)
        t = yb.sum()
        a += (xb.T @ xb - cj * np.outer(s, s)) / se2
        b += (xb.T @ yb - cj * s * t) / se2
    beta = np.linalg.solve(a, b)
    cov = np.linalg.inv(a)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    z = beta / se
    pval = 2.0 * sps.norm.sf(np.abs(z))
    return {
        "beta": [float(v) for v in beta],
        "se": [float(v) for v in se],
        "z": [float(v) for v in z],
        "p": [float(v) for v in pval],
        "sigma_b": float(np.sqrt(sb2)),
        "sigma_e": float(np.sqrt(se2)),
        "n_groups": int(codes.size),
        "n_obs": int(n),
        "converged": bool(res.success),
    }


def power_two_sample(
    delta: float, sd: float, alpha: float = 0.05, power: float = 0.8,
    max_n: int = 100000,
) -> Optional[int]:
    """Prefixes per group for a two-sample t-test at target power."""
    from scipy import stats as sps

    if not np.isfinite(delta) or not np.isfinite(sd) or sd <= 0 or delta == 0:
        return None
    for n in range(2, max_n):
        df = 2 * n - 2
        crit = sps.t.ppf(1 - alpha / 2, df)
        nc = abs(delta) / (sd * np.sqrt(2.0 / n))
        achieved = 1 - sps.nct.cdf(crit, df, nc) + sps.nct.cdf(-crit, df, nc)
        if achieved >= power:
            return n
    return None


def power_one_sample(
    mean: float, sd: float, alpha: float = 0.05, power: float = 0.8,
    max_n: int = 100000,
) -> Optional[int]:
    """Prefixes for a one-sample t-test at target power."""
    from scipy import stats as sps

    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0 or mean == 0:
        return None
    for n in range(2, max_n):
        df = n - 1
        crit = sps.t.ppf(1 - alpha / 2, df)
        nc = abs(mean) / (sd / np.sqrt(n))
        achieved = 1 - sps.nct.cdf(crit, df, nc) + sps.nct.cdf(-crit, df, nc)
        if achieved >= power:
            return n
    return None


def prefix_latency_summaries(
    latency: np.ndarray,
    prefix_keys: np.ndarray,
    sel: np.ndarray,
    stat: str = "median",
) -> Dict[str, float]:
    """One latency summary per prefix over the selected eligible trials."""
    out: Dict[str, float] = {}
    for p in np.unique(prefix_keys[sel]):
        vals = finite(latency[sel & (prefix_keys == p)])
        if vals.size == 0:
            continue
        out[str(p)] = float(np.median(vals) if stat == "median" else vals.mean())
    return out


def small_sample_two_sample(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_perm: int = 5000
) -> Dict[str, Any]:
    """Prefix-level two-sample test for small n (M3/D4).

    A Welch t-test on the per-prefix summaries plus a label-permutation test
    of the mean difference.  The normal (z) approximation is not used: with
    ~11 prefixes in one arm the t / permutation reference is required.
    ``n_perm`` controls the permutation replicates.
    """
    from scipy import stats as sps

    x = finite(x)
    y = finite(y)
    if x.size < 2 or y.size < 2:
        return {"n_x": int(x.size), "n_y": int(y.size)}
    t_stat, p_t = sps.ttest_ind(x, y, equal_var=False)
    obs = float(x.mean() - y.mean())
    pool = np.concatenate([x, y])
    nx = x.size
    null = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        perm = rng.permutation(pool)
        null[b] = perm[:nx].mean() - perm[nx:].mean()
    p_perm = permutation_p_value(abs(obs), np.abs(null))
    return {
        "n_x": int(nx),
        "n_y": int(y.size),
        "mean_x": float(x.mean()),
        "mean_y": float(y.mean()),
        "mean_diff": obs,
        "welch_t": float(t_stat),
        "welch_p": float(p_t),
        "permutation_p_two_sided": p_perm,
        "cliffs_delta": cliffs_delta(x, y),
        "cohens_d": cohens_d(x, y),
        "note": (
            "prefix-level summaries; small n -> permutation p is the primary "
            "reference, the Welch t is secondary"
        ),
    }


# ══════════════════════════════════════════════════════════════════════
# Facilitation contrast (C6b) + covariates (C5)
# ══════════════════════════════════════════════════════════════════════
def _facilitation_contrast(
    latency: np.ndarray,
    resp: np.ndarray,
    prefix_keys: np.ndarray,
    m_wind: np.ndarray,
    m_multi: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = 5000,
) -> Dict[str, Any]:
    """Prefix-level facilitation contrast (D4) for one multisensory subset.

    HEADLINE inference is prefix-level: per-prefix median latencies, a Welch
    t and a label-permutation test on those summaries.  The trial-level
    Mann-Whitney U and the response-rate Fisher test are DEMOTED to a
    descriptive block and explicitly flagged as NOT valid under nesting
    (condition is perfectly nested in prefix, so trial-level p-values
    pseudoreplicate).

    Args:
        latency: ``(N,)`` eligible latency (ms, NaN for non-responders).
        resp: ``(N,)`` primary response frame (``-1`` = non-responder).
        prefix_keys: ``(N,)`` recording-prefix key.
        m_wind: ``(N,)`` boolean wind_only mask (eligible).
        m_multi: ``(N,)`` boolean multisensory-subset mask (eligible).
        rng: Seeded generator for the permutation test.
        n_perm: Permutation replicates for the prefix-level test.

    Returns:
        Dict with ``n_trials``, ``n_prefixes``, the headline
        ``prefix_level_test`` and the descriptive ``trial_level`` block.
    """
    from scipy import stats as sps

    a = finite(latency[m_wind])
    av = finite(latency[m_multi])
    pre_wind = prefix_latency_summaries(latency, prefix_keys, m_wind)
    pre_multi = prefix_latency_summaries(latency, prefix_keys, m_multi)
    small = small_sample_two_sample(
        np.array(list(pre_wind.values())), np.array(list(pre_multi.values())),
        rng, n_perm=n_perm,
    )

    trial: Dict[str, Any] = {
        "note": (
            "DESCRIPTIVE ONLY: trial-level tests treat trials as independent, "
            "which is NOT valid under nesting (condition is perfectly nested "
            "in recording prefix); the prefix_level_test is the headline"
        ),
    }
    if a.size >= 1 and av.size >= 1:
        u, p_u = sps.mannwhitneyu(a, av, alternative="two-sided")
        trial["mannwhitney_U"] = float(u)
        trial["mannwhitney_p"] = float(p_u)
        trial["cohens_d_multi_minus_wind"] = cohens_d(av, a)
        trial["cliffs_delta_multi_minus_wind"] = cliffs_delta(av, a)
    k_w = int(np.sum(resp[m_wind] >= 0)) if m_wind.any() else 0
    k_m = int(np.sum(resp[m_multi] >= 0)) if m_multi.any() else 0
    nw = int(m_wind.sum())
    nm = int(m_multi.sum())
    if nw > 0 and nm > 0:
        tab = np.array([[k_w, nw - k_w], [k_m, nm - k_m]])
        _, p_rate = sps.fisher_exact(tab)
        trial["response_rate_fisher_p"] = float(p_rate)
    return {
        "n_trials": {"wind_only": nw, "multisensory": nm},
        "n_prefixes": {"wind_only": len(pre_wind), "multisensory": len(pre_multi)},
        "latency_ms": {"wind_only": summary_stats(a), "multisensory": summary_stats(av)},
        "prefix_level_test": small,
        "trial_level": trial,
    }


def run_facilitation(
    cond: np.ndarray,
    ttc: np.ndarray,
    latency: np.ndarray,
    resp: np.ndarray,
    prefix_keys: np.ndarray,
    eligible: np.ndarray,
    covariates: Dict[str, Any],
    rng: np.random.Generator,
    n_boot: int,
    latency_label: Optional[np.ndarray] = None,
    resp_label: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Wind-only vs multisensory latency + response-rate facilitation (D1/D4/D5).

    Only ELIGIBLE trials (stationary at the D1 trigger) enter any latency or
    rate statistic.  The latency contrast is a between-prefix comparison
    (D4): prefixes are condition-pure, so the fixed effect is estimated from
    prefix summaries and the random intercept cannot adjust the session
    confound.  The headline inference is prefix-level (Welch on per-prefix
    medians + permutation p); the trial-level Mann-Whitney / Fisher tests are
    descriptive only and flagged as not valid under nesting.

    D4: the contrast is reported PER TTC VARIANT and POOLED, each with
    ``n_trials`` and ``n_prefixes``.  The variant with fewer than 5 prefixes
    (TTC -225, 3 prefixes) is flagged ``informative=False``.

    D5: power uses the SD of the per-prefix summary (not the LMM
    ``sigma_b`` alone) and is reported for the FACILITATION contrast only.

    The ``resp_label`` array passed in is the ELIGIBILITY-GATED s1 responder
    (see :func:`detect_responses`), so the s1 rate below is a gated rate.
    """
    from scipy import stats as sps

    el = eligible
    m_wind = (cond == "wind_only") & el
    m_multi = (cond == "multisensory") & el
    a = finite(latency[m_wind])
    av = finite(latency[m_multi])

    norm_a = sps.shapiro(a) if a.size >= 3 else None
    norm_av = sps.shapiro(av) if av.size >= 3 else None
    levene = sps.levene(a, av) if a.size >= 2 and av.size >= 2 else None
    if a.size and av.size:
        u, p_u = sps.mannwhitneyu(a, av, alternative="two-sided")
        d = cohens_d(av, a)
        cd = cliffs_delta(av, a)
    else:
        u = p_u = d = cd = float("nan")

    # Stratified prefix-cluster bootstrap (true with-replacement, B1) of the
    # median latency difference using prefix multiplicity weights.
    strata = np.where(cond == "wind_only", "wind_only",
                      np.where(cond == "multisensory", "multisensory", "other"))
    diffs: List[float] = []
    for w in bootstrap_weights(prefix_keys, strata, n_boot, rng):
        aa = finite(expand_weighted(latency[m_wind], w[m_wind]))
        bb = finite(expand_weighted(latency[m_multi], w[m_multi]))
        if aa.size and bb.size:
            diffs.append(float(np.median(bb) - np.median(aa)))
    diffs_arr = np.asarray(diffs)
    obs = float(np.median(av) - np.median(a))

    # Response rates (primary 10 cm/s crossing, eligible trials only).
    rate_wind = float(np.mean(resp[m_wind] >= 0)) if m_wind.any() else float("nan")
    rate_multi = float(np.mean(resp[m_multi] >= 0)) if m_multi.any() else float("nan")
    k_w = int(np.sum(resp[m_wind] >= 0))
    k_m = int(np.sum(resp[m_multi] >= 0))
    n_w = int(m_wind.sum())
    n_m = int(m_multi.sum())
    tab = np.array([[k_w, n_w - k_w], [k_m, n_m - k_m]])
    _, p_rate = sps.fisher_exact(tab)
    expected = tab.sum(axis=1, keepdims=True) @ tab.sum(axis=0, keepdims=True)
    expected = expected / tab.sum()
    if np.all(expected > 0):
        chi2 = float(sps.chi2_contingency(tab, correction=False)[0])
        phi = float(np.sqrt(chi2 / tab.sum()))
    else:
        phi = float("nan")

    # D4: prefix-level summaries and the small-sample test (headline).
    pre_wind = prefix_latency_summaries(latency, prefix_keys, m_wind)
    pre_multi = prefix_latency_summaries(latency, prefix_keys, m_multi)
    small = small_sample_two_sample(
        np.array(list(pre_wind.values())), np.array(list(pre_multi.values())), rng
    )

    # Mixed-effects model: prefix random intercept, condition fixed.
    sel = m_wind | m_multi
    y = finite_subset(latency, sel)
    if y["n"] >= 6:
        x_fixed = np.column_stack(
            [np.ones(y["n"]), (cond[sel][y["mask"]] == "multisensory").astype(float)]
        )
        groups = prefix_keys[sel][y["mask"]]
        lmm = random_intercept_lmm(y["y"], x_fixed, groups)
    else:
        lmm = None

    # D5: power from the per-prefix summary SD (not sigma_b alone).
    sd_wind = float(np.std(list(pre_wind.values()), ddof=1)) if len(pre_wind) > 1 else np.nan
    sd_multi = (
        float(np.std(list(pre_multi.values()), ddof=1)) if len(pre_multi) > 1 else np.nan
    )
    nw, nm = len(pre_wind), len(pre_multi)
    if np.isfinite(sd_wind) and np.isfinite(sd_multi) and nw > 1 and nm > 1:
        pooled_sd = float(np.sqrt(
            ((nw - 1) * sd_wind ** 2 + (nm - 1) * sd_multi ** 2) / (nw + nm - 2)
        ))
    else:
        pooled_sd = float("nan")
    sigma_b = lmm["sigma_b"] if lmm else float("nan")
    power = {
        "facilitation_two_sample": {
            "observed_delta_ms": obs,
            "per_prefix_summary": "median latency per prefix",
            "sd_per_prefix_median_ms": {
                "wind_only": sd_wind, "multisensory": sd_multi,
                "pooled": pooled_sd,
            },
            "n_prefixes": {"wind_only": nw, "multisensory": nm},
            "between_prefix_sd_ms_lmm_sigma_b": sigma_b,
            "prefixes_per_group_for_80pct": power_two_sample(obs, pooled_sd),
            "caveat": (
                "noncentral-t small-sample approximation; treat the returned n "
                "as an order-of-magnitude planning figure, not an exact target"
            ),
        },
        "race_power": (
            "not estimated here; see race_model.power_analysis (the signed "
            "one-sample race power is deliberately dropped, D5)"
        ),
    }
    # s1: labeling-responder rate contrast (B3), eligible trials only.  The
    # s1 responder array is already ELIGIBILITY-GATED in detect_responses, so
    # this rate is gated twice (once in the array, once by the mask).
    if resp_label is not None:
        rl = np.asarray(resp_label)
        k_wl = int(np.sum(rl[m_wind] >= 0))
        k_ml = int(np.sum(rl[m_multi] >= 0))
        tabl = np.array([[k_wl, n_w - k_wl], [k_ml, n_m - k_ml]])
        _, p_rate_l = sps.fisher_exact(tabl)
        power["labeling_responder_rate"] = {
            "note": (
                "s1 responders are eligibility-gated (D1); the rate is over "
                "eligible trials only"
            ),
            "wind_only": {"responders": k_wl, "n": n_w},
            "multisensory": {"responders": k_ml, "n": n_m},
            "fisher_p": float(p_rate_l),
        }

    # D4: per-variant contrasts (each multisensory variant vs wind_only).
    per_variant: Dict[str, Any] = {}
    for v in np.unique(ttc[cond == "multisensory"]):
        m_v = m_multi & (ttc == v)
        row = _facilitation_contrast(latency, resp, prefix_keys, m_wind, m_v, rng)
        row["ttc_ms"] = int(v)
        row["informative"] = bool(row["n_prefixes"]["multisensory"] >= 5)
        per_variant[f"ttc_{int(v)}"] = row

    # X2: Holm over the DECLARED facilitation family (pooled + the 4 TTC
    # variants) on the prefix-permutation two-sided p-values (the primary
    # reference; the Welch t is secondary and is left uncorrected).
    fam = ["pooled"] + list(per_variant.keys())
    raw_fam = [small.get("permutation_p_two_sided")] + [
        per_variant[k]["prefix_level_test"].get("permutation_p_two_sided")
        for k in per_variant
    ]
    for k, adj_p in zip(fam, holm_adjust(raw_fam)):
        if k == "pooled":
            small["permutation_p_two_sided_holm"] = float(adj_p)
        else:
            per_variant[k]["prefix_level_test"]["permutation_p_two_sided_holm"] = float(adj_p)
    holm_family = {
        "correction": "Holm step-down (family-wise error rate)",
        "family": "the facilitation family: pooled + the 4 TTC variants",
        "m": int(sum(1 for p in raw_fam if p is not None and np.isfinite(p))),
        "note": (
            "Holm over the declared 5-test facilitation family (pooled + 4 TTC "
            "variants) on the prefix-permutation two-sided p-values; raw "
            "`permutation_p_two_sided` and adjusted "
            "`permutation_p_two_sided_holm` are both reported."
        ),
    }

    return {
        "inclusion_rule": (
            "D1 primary: stationary at trigger (|Y| < 10 cm/s in [t-250, t]); "
            "response = first |Y| >= 10 cm/s strictly after the trigger"
        ),
        "wind_only": {
            "n": n_w,
            "responders": k_w,
            "response_rate": rate_wind,
            "response_rate_ci95": wilson_ci(k_w, n_w),
            "latency_ms": summary_stats(a),
            "latency_percentiles_ms": percentiles(a, (5, 25, 50, 75, 95)),
            "n_prefixes": nw,
        },
        "multisensory": {
            "n": n_m,
            "responders": k_m,
            "response_rate": rate_multi,
            "response_rate_ci95": wilson_ci(k_m, n_m),
            "latency_ms": summary_stats(av),
            "latency_percentiles_ms": percentiles(av, (5, 25, 50, 75, 95)),
            "n_prefixes": nm,
        },
        "contrast": {
            "test": "Mann-Whitney U (two-sided, trial-level; DESCRIPTIVE ONLY)",
            "validity_note": (
                "trial-level; NOT valid under nesting (condition is perfectly "
                "nested in recording prefix) -- the prefix_level_test is the "
                "headline inference"
            ),
            "U": float(u) if a.size and av.size else None,
            "p": float(p_u) if a.size and av.size else None,
            "cohens_d_multi_minus_wind": d,
            "cliffs_delta_multi_minus_wind": cd,
            "normality_shapiro_wind": (
                {"W": float(norm_a.statistic), "p": float(norm_a.pvalue)}
                if norm_a
                else None
            ),
            "normality_shapiro_multisensory": (
                {"W": float(norm_av.statistic), "p": float(norm_av.pvalue)}
                if norm_av
                else None
            ),
            "variance_levene": (
                {"W": float(levene.statistic), "p": float(levene.pvalue)}
                if levene
                else None
            ),
            "median_diff_ms": obs,
            "bootstrap_median_diff_mean": float(diffs_arr.mean())
            if diffs_arr.size
            else None,
            "bootstrap_median_diff_ci95": [
                float(np.percentile(diffs_arr, 2.5)),
                float(np.percentile(diffs_arr, 97.5)),
            ]
            if diffs_arr.size
            else None,
            "response_rate_fisher_p": float(p_rate),
            "response_rate_phi": phi,
        },
        "prefix_level_test": {
            "model": "between-prefix comparison (prefixes are condition-pure)",
            **small,
        },
        "per_variant": per_variant,
        "holm_family": holm_family,
        "mixed_effects": {
            "model": "latency ~ condition + (1|recording_prefix), Gaussian REML",
            "condition_levels": ["wind_only", "multisensory"],
            "fixed_effects": lmm,
            "caveat": (
                "condition is PERFECTLY NESTED in prefix (prefixes are "
                "condition-pure), so the fixed effect is a between-prefix "
                "comparison and the random intercept CANNOT adjust the "
                "session confound; the z-approximation is not small-sample "
                "appropriate -- see prefix_level_test"
            ),
        },
        "power_analysis": power,
        "covariates": covariates,
        "bootstrap_unit": "recording_prefix (stratified by group, with replacement)",
    }


def finite_subset(latency: np.ndarray, sel: np.ndarray) -> Dict[str, Any]:
    """Return finite latencies and their original mask within ``sel``."""
    sub = latency[sel]
    ok = np.isfinite(sub)
    return {"y": sub[ok], "mask": ok, "n": int(ok.sum())}


# ══════════════════════════════════════════════════════════════════════
# Figures (C6)
# ══════════════════════════════════════════════════════════════════════
def _style(ax: Any, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.99, 0.01, "EXPLORATORY", transform=ax.transAxes, ha="right",
        va="bottom", fontsize=7, color="crimson", alpha=0.8,
    )


def figure_race(
    out_dir: Path,
    cond: np.ndarray,
    ttc: np.ndarray,
    latency: np.ndarray,
    v_col: np.ndarray,
    w_on: np.ndarray,
    dt_ms: float,
    n_boot: int,
    prefix_keys: np.ndarray,
    eligible: np.ndarray,
    rng: np.random.Generator,
) -> Optional[str]:
    """Panel (a): race CDFs per TTC group with violation + bootstrap band.

    The Miller bound is clipped at 1 (M1).  The band is the prefix-cluster
    bootstrap envelope of the bound (10% of the requested replicates, for
    render cost); it is BOUND-ONLY uncertainty -- it does not include the
    F_AV sampling variability -- and its replicate count is stated in the
    caption.  The numeric CI of the violation area lives in the JSON.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m_wind = (cond == "wind_only") & eligible
    m_vis = (cond == "visual_only") & eligible
    m_multi = (cond == "multisensory") & eligible
    variants = np.unique(ttc[cond == "multisensory"])
    if variants.size == 0:
        return None
    tgrid = np.arange(0.0, RESPONSE_WINDOW_MS + dt_ms, dt_ms)
    lat_a = np.sort(finite(latency[m_wind]))
    n_a = int(m_wind.sum())
    lat_v = np.sort(finite(latency[m_vis]))
    n_v = int(m_vis.sum())

    strata = race_strata(cond, ttc)
    band_weights = list(bootstrap_weights(prefix_keys, strata, n_boot, rng))

    fig, axes = plt.subplots(
        1, variants.size, figsize=(4 * variants.size, 3.4), sharey=True,
        squeeze=False,
    )
    for ax, v in zip(axes[0], variants):
        sel = m_multi & (ttc == v)
        lat_av = np.sort(finite(latency[sel]))
        n_av = int(sel.sum())
        soa = _variant_soa(cond == "multisensory", ttc, w_on, v_col, dt_ms, float(v))
        fav = ecdf(lat_av, tgrid, n_av)
        bound = miller_bound(tgrid, lat_a, n_a, lat_v, n_v, [soa], [1.0])
        band = np.vstack(
            [
                miller_bound(
                    tgrid,
                    np.sort(finite(expand_weighted(latency[m_wind], w[m_wind]))),
                    int(round(weighted_count(m_wind, w))),
                    np.sort(finite(expand_weighted(latency[m_vis], w[m_vis]))),
                    int(round(weighted_count(m_vis, w))),
                    [soa],
                    [1.0],
                )
                for w in band_weights
            ]
        )
        ax.fill_between(
            tgrid, np.percentile(band, 2.5, axis=0),
            np.percentile(band, 97.5, axis=0), color="C0", alpha=0.15,
            label=f"bound 95% band (n={len(band_weights)}, bound-only)",
        )
        ax.plot(tgrid, fav, color="C3", label="F_AV (multisensory)")
        ax.plot(tgrid, bound, color="C0", ls="--", label="Miller bound")
        ax.fill_between(
            tgrid, bound, np.maximum(fav, bound), color="C3", alpha=0.25,
            label="violation",
        )
        _style(ax, f"TTC = {int(v)} ms (SOA {soa:.0f} ms)")
    axes[0][0].set_ylabel("defective CDF")
    for ax in axes[0]:
        ax.set_xlabel("time from wind onset (ms)")
        ax.set_xlim(0, RESPONSE_WINDOW_MS)
    axes[0][0].legend(fontsize=6, loc="lower right")
    fig.suptitle(
        "Race model: F_AV vs Miller bound (between-group; EXPLORATORY)",
        fontsize=10,
    )
    fig.tight_layout()
    path = out_dir / "race_model_cdfs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def figure_facilitation(
    out_dir: Path,
    cond: np.ndarray,
    latency: np.ndarray,
    resp: np.ndarray,
    eligible: np.ndarray,
) -> Optional[str]:
    """Panel (b): wind_only vs multisensory latency + response rate (D1)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m_wind = (cond == "wind_only") & eligible
    m_multi = (cond == "multisensory") & eligible
    a = finite(latency[m_wind])
    av = finite(latency[m_multi])
    if a.size == 0 or av.size == 0:
        return None
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3.4))
    bins = np.arange(0, RESPONSE_WINDOW_MS + 20, 20)
    ax0.hist(a, bins=bins, density=True, alpha=0.5, color="C0", label="wind_only")
    ax0.hist(av, bins=bins, density=True, alpha=0.5, color="C3",
             label="multisensory")
    ax0.set_xlim(0, 400)
    ax0.set_xlabel("latency from wind onset (ms)")
    ax0.set_ylabel("density")
    ax0.legend(fontsize=7)
    _style(ax0, "Escape latency (eligible)")
    rates = [
        float(np.mean(resp[m_wind] >= 0)) if m_wind.any() else float("nan"),
        float(np.mean(resp[m_multi] >= 0)) if m_multi.any() else float("nan"),
    ]
    ax1.bar(["wind_only", "multisensory"], rates, color=["C0", "C3"], alpha=0.7)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("response rate")
    for x, r in enumerate(rates):
        ax1.text(x, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
    _style(ax1, "Response rate (eligible)")
    fig.tight_layout()
    path = out_dir / "facilitation_latency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def figure_accumulation(
    out_dir: Path,
    cond: np.ndarray,
    ttc: np.ndarray,
    latency: np.ndarray,
    eligible: np.ndarray,
) -> Optional[str]:
    """Panel (c): latency distributions + ex-Gaussian fits, mean-SD by TTC."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    m_multi = (cond == "multisensory") & eligible
    variants = np.unique(ttc[cond == "multisensory"])
    if variants.size < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    xs = np.linspace(0, 400, 200)
    for v in variants:
        x = finite(latency[m_multi & (ttc == v)])
        if x.size < 20:
            continue
        axes[0].hist(x, bins=np.arange(0, 400, 20), density=True, alpha=0.25)
        try:
            k, lo, sc = sps.exponnorm.fit(x)
            axes[0].plot(xs, sps.exponnorm.pdf(xs, k, lo, sc), lw=1.2)
        except Exception:  # pragma: no cover
            pass
    axes[0].set_xlabel("latency from wind onset (ms)")
    axes[0].set_ylabel("density")
    _style(axes[0], "Latency + ex-Gaussian fits (eligible)")
    means, sds, xs_v = [], [], []
    for v in variants:
        x = finite(latency[m_multi & (ttc == v)])
        if x.size >= 20:
            xs_v.append(int(v))
            means.append(float(x.mean()))
            sds.append(float(x.std(ddof=1)))
    if means:
        axes[1].errorbar(
            xs_v, means, yerr=sds, fmt="o-",
            capsize=3, color="C3",
        )
        axes[1].set_xlabel("TTC (ms)")
        axes[1].set_ylabel("mean +/- SD latency (ms)")
        _style(axes[1], "Mean-SD across TTC (suggestive)")
    fig.tight_layout()
    path = out_dir / "accumulation_latency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def figure_trigger(
    out_dir: Path,
    cond: np.ndarray,
    latency: np.ndarray,
    v_col: np.ndarray,
    w_on: np.ndarray,
    dt_ms: float,
    eligible: np.ndarray,
) -> Optional[str]:
    """Optional panel: escape locked to wind onset rather than collision."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m_multi = (cond == "multisensory") & eligible
    lat_wind = finite(latency[m_multi])
    lat_col = finite(
        np.array(
            [
                latency[i] - (v_col[i] - w_on[i]) * dt_ms
                for i in np.where(m_multi)[0]
                if np.isfinite(latency[i])
            ]
        )
    )
    if lat_wind.size == 0 or lat_col.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    bins = np.arange(-1000, 400, 25)
    ax.hist(lat_wind, bins=bins, alpha=0.6, color="C3",
            label="from wind onset")
    ax.hist(lat_col, bins=bins, alpha=0.6, color="C0", label="from collision")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("escape time (ms)")
    ax.set_ylabel("count")
    ax.legend(fontsize=7)
    _style(ax, "Multisensory escape is wind-locked")
    fig.tight_layout()
    path = out_dir / "trigger_locking.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════
def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="processed nsmor_dataset.pt")
    parser.add_argument("--nested_prior_artifact", type=Path, default=None,
                        help="optional artifact carrying recording_prefix_keys")
    parser.add_argument("--dt_ms", type=float, default=DT_MS_DEFAULT,
                        help="frame interval in ms (validated vs dataset)")
    parser.add_argument("--output_dir", type=Path, default=Path("results/behavior-20261010"),
                        help="output directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--n_boot", type=int, default=2000,
                        help="prefix-cluster bootstrap replicates")
    parser.add_argument("--n_perm", type=int, default=2000,
                        help="prefix-level permutation replicates for the race test")
    return parser


def run(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute the full analysis.

    Returns:
        ``(summary, context)`` where ``context`` carries the loaded corpus,
        events, responses and prefix keys so figures are rendered without
        re-reading the multi-GB dataset.
    """
    corpus = load_corpus(args.dataset)
    dt_ms = float(args.dt_ms)
    if not np.isclose(dt_ms, corpus["dt_ms"], rtol=1e-9, atol=0.0):
        logger.warning(
            "--dt_ms=%s differs from dataset model_dt_ms=%s", dt_ms, corpus["dt_ms"]
        )
    cond = corpus["conditions"]
    ttc = corpus["ttc"]
    prefix_keys, prefix_source = resolve_prefix_keys(
        corpus["session_ids"], args.nested_prior_artifact
    )
    n_prefix = len(set(prefix_keys.tolist()))
    events = detect_events(corpus, dt_ms)
    responses = detect_responses(corpus, events, dt_ms)
    latency = responses["latency_ms"]
    resp = responses["resp"]
    eligible = responses["eligible"]
    rng = np.random.default_rng(args.seed)

    kin = np.array(
        [kinematics(corpus["y"][i], int(resp[i]), dt_ms, ESCAPE_THRESHOLD_CMS)
         for i in range(corpus["n_trials"])]
    )
    assert kin.shape == (corpus["n_trials"], 4), f"kinematics shape {kin.shape}"

    # Condition purity of prefixes (honesty gate for the between-group design).
    prefix_pure = True
    for p in np.unique(prefix_keys):
        idx = prefix_keys == p
        if len(set(cond[idx].tolist())) != 1:
            prefix_pure = False
            break
    prefix_condition = {}
    for p in np.unique(prefix_keys):
        prefix_condition[str(p)] = str(cond[prefix_keys == p][0])

    covariates = prefix_covariates(corpus, events, prefix_keys)

    summary: Dict[str, Any] = {
        "meta": {
            "status": "EXPLORATORY_IN_SAMPLE_MODEL_FREE",
            "exploratory": True,
            "in_sample": True,
            "between_group_design": True,
            "model_free_qualifier": (
                "model-free refers to the PRIMARY statistics (10 cm/s crossing "
                "on the physical channels); the s1 labeling-responder "
                "quantities reuse the labeling pipeline's sustained criterion"
            ),
            "n_trials": corpus["n_trials"],
            "dt_ms": dt_ms,
            "escape_threshold_cms": ESCAPE_THRESHOLD_CMS,
            "response_window_ms": RESPONSE_WINDOW_MS,
            "pre_trigger_window_ms": PRE_TRIGGER_WINDOW_MS,
            "stationary_window_ms": STATIONARY_WINDOW_MS,
            "stationary_threshold_cms": STATIONARY_THRESHOLD_CMS,
            "stationary_threshold_sensitivity_cms": STATIONARY_THRESHOLD_SENS_CMS,
            "seed": int(args.seed),
            "n_boot": int(args.n_boot),
            "n_perm": int(args.n_perm),
            "counts_by_condition": {
                c: int((cond == c).sum()) for c in CONDITIONS
            },
            "counts_by_label": {
                LABEL_NAMES[k]: int((corpus["labels"] == k).sum()) for k in range(4)
            },
            "label_x_condition": {
                c: {
                    LABEL_NAMES[k]: int(
                        ((corpus["labels"] == k) & (cond == c)).sum()
                    )
                    for k in range(4)
                }
                for c in CONDITIONS
            },
            "n_sessions": int(len(set(corpus["session_ids"].tolist()))),
            "n_recording_prefixes": int(n_prefix),
            "prefix_condition_pure": bool(prefix_pure),
            "prefix_source": prefix_source,
            "animal_identity_status": (
                "unverified -> recording prefix is the working cluster unit"
            ),
            "pipeline_semantics_version": corpus["pipeline_semantics_version"],
            "feature_config": corpus["feature_config"],
            "labeling_funnel": corpus["labeling_funnel"],
            "resampling_unit": (
                "recording_prefix (stratified by group, with replacement; "
                "prefix multiplicity weights)"
            ),
            "definitions": {
                "reference_event": {
                    "wind_only": "wind onset (first wind channel > 0.5)",
                    "multisensory": "wind onset",
                    "visual_only": "visual collision (peak visual angle)",
                },
                "primary_inclusion_rule": (
                    "D1: a trial is ELIGIBLE only if |Y| < 10 cm/s at every "
                    "frame of [trigger-250 ms, trigger]; ineligible trials are "
                    "excluded from every latency/CDF/race/facilitation statistic"
                ),
                "primary_response": (
                    "D1: first frame STRICTLY after the trigger (latency 0 "
                    "impossible) with |Y| >= 10 cm/s inside [trigger, "
                    "trigger+2000 ms] for EVERY condition; trigger = wind onset "
                    "(wind-containing) / collision (visual_only), so the visual "
                    "window covers collision+1000 ms"
                ),
                "eligibility_sensitivity_5cms": (
                    "labeling-consistent variant: |Y| < 5 cm/s throughout "
                    "[trigger-250 ms, trigger]"
                ),
                "sensitivity_labeling_responder": (
                    "s1: response = sustained > 5 cm/s for 250 ms with a 200 ms "
                    "initiation bound (nsmor/pipeline/labeling._check_sustained_speed), "
                    "under the SAME D1 eligibility gate as the primary response; "
                    "resp_label_ungated is the ungated diagnostic"
                ),
                "sensitivity_all_trials": (
                    "s2: no stationarity gate (contamination illustration only)"
                ),
                "sensitivity_legacy_visual_window": (
                    "s3: the round-1/2 collision-anchored window "
                    "[collision-2000 ms, collision+1000 ms] for visual_only "
                    "(identical to primary elsewhere); shows what D1 changed"
                ),
                "labeling_responder": (
                    "label in {ESCAPE, PREWALK}: post-stimulus sustained "
                    "> 5 cm/s for 250 ms (nsmor/pipeline/labeling.py)"
                ),
            },
        },
        "anchors": {},
        "multisensory_soa_by_variant": {},
        "prefix_condition": prefix_condition,
    }

    # Anchors / SOA.
    for c in CONDITIONS:
        m = cond == c
        summary["anchors"][c] = {
            "visual_onset_ms": summary_stats(events["v_on"][m][events["v_on"][m] >= 0] * dt_ms),
            "visual_collision_ms": summary_stats(
                events["v_col"][m][events["v_col"][m] >= 0] * dt_ms
            ),
            "wind_onset_ms": summary_stats(
                events["w_on"][m][events["w_on"][m] >= 0] * dt_ms
            ),
            "escape_abs_frame_ms": summary_stats(
                resp[m][resp[m] >= 0] * dt_ms
            ),
            "latency_from_trigger_ms": summary_stats(latency[m]),
        }
    m_multi = cond == "multisensory"
    for v in np.unique(ttc[m_multi]):
        sel = m_multi & (ttc == v)
        summary["multisensory_soa_by_variant"][f"ttc_{int(v)}"] = {
            "n": int(sel.sum()),
            "soa_wind_to_collision_ms": summary_stats(
                (events["v_col"][sel] - events["w_on"][sel]) * dt_ms
            ),
            "latency_from_wind_ms": summary_stats(latency[sel]),
        }

    # D1/D3: eligibility + per-condition excluded-count report.
    summary["eligibility"] = {
        "rule": (
            "D1 primary: eligible only if |Y| < 10 cm/s throughout "
            "[trigger-250 ms, trigger]"
        ),
        "window_ms": STATIONARY_WINDOW_MS,
        "threshold_cms": STATIONARY_THRESHOLD_CMS,
        "sensitivity_threshold_cms": STATIONARY_THRESHOLD_SENS_CMS,
        "by_condition": {
            c: {
                "n": int((cond == c).sum()),
                "n_eligible": int(np.sum(eligible & (cond == c))),
                "n_excluded": int(np.sum((~eligible) & (cond == c))),
                "excluded_fraction": float(np.mean(~eligible[cond == c])),
                "n_eligible_5cms": int(
                    np.sum(responses["eligible5"] & (cond == c))
                ),
                "n_excluded_5cms": int(
                    np.sum((~responses["eligible5"]) & (cond == c))
                ),
                "primary_responders": int(np.sum((resp >= 0) & (cond == c))),
                # D3: the D1-trigger sustained criterion, ELIGIBILITY-GATED
                # ("labeling responder").
                "labeling_responders": int(
                    np.sum((responses["resp_label"] >= 0) & (cond == c))
                ),
                # Diagnostic: the same sustained criterion WITHOUT the D1
                # gate; never the D1 labeling responder.
                "labeling_responders_ungated": int(
                    np.sum((responses["resp_label_ungated"] >= 0) & (cond == c))
                ),
                "all_trials_responders_s2": int(
                    np.sum((responses["resp_all"] >= 0) & (cond == c))
                ),
                "elig5_responders": int(
                    np.sum((responses["resp_elig5"] >= 0) & (cond == c))
                ),
                "legacy_window_responders_s3": int(
                    np.sum((responses["resp_legacy"] >= 0) & (cond == c))
                ),
            }
            for c in CONDITIONS
        },
        "note": (
            "excluded trials are pre-trigger movers under the D1 rule; s2 "
            "(all_trials_responders_s2) is the old all-trials rule shown only "
            "as a contamination illustration; s3 (legacy_window_responders_s3) "
            "uses the round-1/2 collision-anchored visual window"
        ),
    }

    # D3: reconcile the recomputed D1-trigger sustained criterion with the
    # STORED dataset labels.  Both are called "labeling responder" nowhere:
    # the recomputed one is `labeling_responders` in the eligibility ledger;
    # the stored one lives ONLY here as `stored_label_agreement`.
    stored_label_responder = np.isin(corpus["labels"], list(RESPONDER_LABELS))
    recomputed_responder = responses["resp_label"] >= 0  # ELIGIBILITY-GATED
    recomputed_ungated = responses["resp_label_ungated"] >= 0
    summary["stored_label_agreement"] = {
        "recomputed_definition": (
            "D1 trigger AND D1 eligibility gate: sustained > 5 cm/s for 250 ms "
            "(200 ms initiation bound) measured from the per-condition trigger "
            "with the D1 window, on trials STATIONARY at the trigger "
            "(|Y| < 10 cm/s in [trigger-250 ms, trigger])"
        ),
        "stored_definition": (
            "stored dataset label in {ESCAPE, PREWALK}; produced by the "
            "labeling pipeline (nsmor/pipeline/labeling.py) on the raw "
            "stimulus_onset clock, NOT re-derived here"
        ),
        "note": (
            "the recomputed criterion is ELIGIBILITY-GATED and is the same "
            "object as `labeling_responders` in the eligibility ledger; the "
            "stored label is a different object. `recomputed_ungated` keeps "
            "the sustained criterion WITHOUT the D1 gate as a diagnostic only "
            "(never the D1 labeling responder)"
        ),
        "by_condition": {
            c: {
                "n": int((cond == c).sum()),
                "recomputed_responders": int(
                    np.sum(recomputed_responder & (cond == c))
                ),
                "recomputed_ungated": int(
                    np.sum(recomputed_ungated & (cond == c))
                ),
                "stored_responders": int(
                    np.sum(stored_label_responder & (cond == c))
                ),
                "agreement": int(
                    np.sum((recomputed_responder & stored_label_responder)
                           & (cond == c))
                ),
                "recomputed_only": int(
                    np.sum((recomputed_responder & ~stored_label_responder)
                           & (cond == c))
                ),
                "stored_only": int(
                    np.sum((~recomputed_responder & stored_label_responder)
                           & (cond == c))
                ),
            }
            for c in CONDITIONS
        },
    }

    # C2: race model (D1/D2/D5).  Its power_analysis is produced inside.
    summary["race_model"] = run_race_model(
        cond, ttc, latency, events["w_on"], events["v_col"], prefix_keys,
        eligible, dt_ms, args.n_boot, rng,
        latency_label=responses["latency_label_ms"],
        n_perm=args.n_perm,
    )

    # Descriptive / negative items.
    summary["discrete_modes"] = run_discrete_modes(cond, latency, kin, rng)
    summary["accumulation"] = run_accumulation(cond, ttc, latency)
    summary["looming_trigger"] = run_looming_trigger(
        cond, ttc, latency, events["w_on"], events["v_col"], dt_ms
    )
    summary["reliability_weighting_status"] = run_reliability_status()
    summary["facilitation"] = run_facilitation(
        cond, ttc, latency, resp, prefix_keys, eligible, covariates, rng, args.n_boot,
        latency_label=responses["latency_label_ms"],
        resp_label=responses["resp_label"],
    )

    context = {
        "corpus": corpus,
        "events": events,
        "responses": responses,
        "prefix_keys": prefix_keys,
        "eligible": eligible,
        "dt_ms": dt_ms,
    }
    return summary, context


def write_figures(
    out_dir: Path,
    corpus: Dict[str, Any],
    events: Dict[str, np.ndarray],
    responses: Dict[str, np.ndarray],
    prefix_keys: np.ndarray,
    eligible: np.ndarray,
    dt_ms: float,
    n_boot: int,
    seed: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Render the figures for positive results.

    Returns:
        ``(paths, figure_notes)`` -- the figure paths and a per-figure list
        of uncertainty notes (M1) so the JSON carries each figure's band /
        replicate provenance next to the file it describes.
    """
    cond = corpus["conditions"]
    ttc = corpus["ttc"]
    latency = responses["latency_ms"]
    resp = responses["resp"]
    band_n = max(50, n_boot // 10)
    rng = np.random.default_rng(seed + 1)
    jobs = [
        (
            figure_race,
            (out_dir, cond, ttc, latency, events["v_col"], events["w_on"],
             dt_ms, band_n, prefix_keys, eligible, rng),
            {
                "panel": "(a) race_model_cdfs.png",
                "uncertainty": (
                    f"bound-only prefix-cluster bootstrap band, "
                    f"{band_n} replicates; F_AV sampling variability is NOT in "
                    "the band (its CI is in race_model[*].ci95)"
                ),
            },
        ),
        (
            figure_facilitation,
            (out_dir, cond, latency, resp, eligible),
            {
                "panel": "(b) facilitation_latency.png",
                "uncertainty": (
                    "eligible trials only; interval estimates are in "
                    "facilitation (Wilson CI, prefix-cluster bootstrap CI)"
                ),
            },
        ),
        (
            figure_accumulation,
            (out_dir, cond, ttc, latency, eligible),
            {
                "panel": "(c) accumulation_latency.png",
                "uncertainty": (
                    "eligible trials only; ex-Gaussian fits are point MLEs; "
                    "the 4-point mean-SD trend is suggestive, no band"
                ),
            },
        ),
        (
            figure_trigger,
            (out_dir, cond, latency, events["v_col"], events["w_on"], dt_ms,
             eligible),
            {
                "panel": "(d) trigger_locking.png",
                "uncertainty": "eligible trials only; descriptive histogram",
            },
        ),
    ]
    paths: List[str] = []
    notes: List[Dict[str, Any]] = []
    for fn, fargs, note in jobs:
        try:
            path = fn(*fargs)
            if path:
                paths.append(path)
                notes.append({**note, "path": path})
        except Exception as exc:  # pragma: no cover - figure best-effort
            logger.warning("figure %s failed: %s", fn.__name__, exc)
    return paths, notes


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = build_arg_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, context = run(args)
    figures, figure_notes = write_figures(
        args.output_dir,
        context["corpus"],
        context["events"],
        context["responses"],
        context["prefix_keys"],
        context["eligible"],
        context["dt_ms"],
        args.n_boot,
        args.seed,
    )
    summary["figures"] = figures
    summary["figure_notes"] = figure_notes
    out = args.output_dir / "behavior_summary.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, default=str)
    logger.info("wrote %s", out)
    logger.info("figures: %s", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
