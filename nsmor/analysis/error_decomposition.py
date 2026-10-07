"""Stage-0 error decomposition and missingness diagnostics for NSMoR.

Read-only instrumentation used to decide whether a model/loss optimization is
justified *before* any frozen-core edit is licensed. It answers these
auditable questions on the accepted checkpoint and its pinned nested
outer-validation split:

1. **High-magnitude contribution** — how much of the pooled (raw cm/s) masked
   squared error is contributed by high-magnitude frames (``|y_true| >= band``
   cm/s) versus the resting bulk. The bands are *cumulative and nested*
   (``|y_true| >= b``), so their shares overlap and must not be summed; only
   ``rest`` (``|y_true| < bands[0]``) partitions disjointly against
   ``bands[0]``. This reports a descriptive share so a target-scaling
   hypothesis can be tested on evidence; it establishes no biological,
   causal, or stability conclusion and vetoes nothing by itself. Every
   configured band must admit at least one eligible frame; an empty band
   fails closed.
2. **Lag-one identity** — whether input channel ``[2]`` (``v_kine(t-1)``) is
   numerically the target's own previous frame, which tests the hypothesis that
   the persistence comparator is already present in the decoder input.
3. **Sustained escape-band sensitivity** — the escape metric is *not* the bare
   ``|y_true| >= band`` table: it reuses ``scripts.train._sustained_run`` so a
   frame must belong to a run of at least ``min_run`` consecutive over-band
   frames, applied per trial. The sweep rescores the same per-trial raw
   predictions/targets over ``band x min_run``. The grid is reported as two
   explicitly-named denominators (``full`` over every trial vs ``aligned`` over
   the trials that admit a scored frame) that must not be conflated.
4. **Padding/masking vs removal** — batch padding is masked (retained,
   unscored); the ``unavailable_no_stimulus_anchor`` trials are removed
   upstream. The two are reported as distinct integers, never conflated.
   The padding count is only truthful when measured from the tensors the
   evaluated loader actually yielded; when it cannot be, the audit reports an
   explicit ``status="unavailable"`` rather than inventing a corpus-wide count
   from the raw dataset lengths.
5. **Missingness ledger** — an explicit, per-condition *description* of the
   dropped trials. Observed condition separation (every drop's condition
   absent from the retained set) is reported as an observation only; it is not
   a mechanism. Without an independent drop rule it cannot be labelled
   structural-not-MCAR, and no biological completeness (e.g. BV censoring) is
   inferred. The with-vs-without-unavailable comparison fails closed when the
   dropped rows cannot be replayed from the accepted artifacts.

Everything here is pure and evaluation-only: no model parameter, loss term, or
training tensor is touched. The magnitude-band decomposition, the zero-predictor
skill and the lag-one persistence comparator are computed on the lag-one
eligible scope (``t >= 1`` within each trial) so the model, the zero predictor
and the persistence comparator are scored on exactly the same frames. The
sustained escape sweep is the exception: it reports BOTH the full valid-trial
scope (train.py semantics) and the ``t >= 1`` aligned scope as explicitly named,
separately-available denominators.
Non-finite, malformed, or empty-band inputs fail closed rather than being
silently treated as observed.

Shape legend
------------
    T  = per-trial sequence length
    F  = feature dimension (8 for the frozen NSMoR contract)
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


__all__ = [
    "decompose_squared_error",
    "escape_sensitivity_sweep",
    "lag_one_identity_check",
    "missingness_ledger",
    "padding_masking_audit",
    "unavailable_trial_comparison",
    "main",
]


#: High-magnitude band thresholds (cm/s) used by the accepted Stage-0
#: decomposition. The CLI default must reproduce these, not the rejected
#: ``10,100`` narrow set.
_ARTIFACT_BANDS_CM_S: Tuple[float, ...] = (1e3, 1e4)

#: Scope label for the ``unavailable_trials_removed`` count. It is a
#: full-corpus ``labeling_eligibility`` fact, never a validation-population
#: denominator; the explicit label keeps the two from being conflated.
_REMOVED_SCOPE = "full_corpus_labeling_eligibility"

#: Explicit reason attached to a partition that admits zero eligible frames
#: (only surfaced for a configured band under the ``allow_unavailable`` opt-in).
_EMPTY_BLOCK_REASON = (
    "no eligible t>=1 frame in this partition on the scored scope; "
    "statistics are unavailable (not zero)"
)

#: Explicit reason attached to a sweep cell that admits no sustained escape
#: frame. The cell is still emitted (with null statistics) so the Cartesian
#: grid stays complete instead of aborting at the first empty cell.
_SWEEP_EMPTY_REASON = (
    "no sustained escape frame at this (band, min_run) cell on the scored "
    "scope; statistics are unavailable (not zero)"
)

#: Reason for a sweep partition whose escape set is empty while the *other*
#: scope (or a wider cell) does admit frames -- e.g. a singleton over-band trial
#: is escape on the full scope but its only t>=1 frame is not, so the aligned
#: partition has zero escape frames though the aligned denominator is non-zero.
_SWEEP_NO_ESCAPE_REASON = (
    "no sustained escape frame in this partition on the scored scope; the "
    "escape statistic is unavailable (not zero)"
)

#: Reason for a sweep partition whose resting set is empty (every scored frame
#: is an escape frame); the resting statistic is unavailable, not zero.
_SWEEP_NO_REST_REASON = (
    "no resting frame in this partition on the scored scope; the resting "
    "statistic is unavailable (not zero)"
)

#: Label stating that band ``sse_share`` / ``frame_share`` are cumulative
#: nested shares (``|y_true| >= b``), NOT disjoint per-band contributions and
#: therefore NOT summable across bands.
_CUMULATIVE_SHARE_SCOPE = (
    "cumulative_nested_share_of_pooled_scope_not_summable_across_bands"
)


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════

def _as_finite_1d(values: Any, name: str) -> np.ndarray:
    """Return a 1-D float64 array, rejecting non-1-D or non-finite input.

    A 0-D scalar or >=2-D block is rejected rather than silently reshaped so a
    scalar cannot masquerade as a length-one trial. NaN/Inf are rejected so a
    missing or unmeasured frame is never silently treated as observed.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_length(value: Any, name: str) -> int:
    """Coerce a length-like value to ``int``, rejecting non-integral input.

    A float such as ``2.5`` is rejected rather than silently truncated to 2,
    and ``bool`` is rejected rather than coerced to 0/1, so a caller can never
    silently score a different crop than the one requested.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(
            f"{name} must be an integral length, got {value!r} (non-integral "
            f"lengths are rejected, never truncated)"
        )
    return int(value)


def _validate_lengths(
    lengths: Optional[Sequence[int]], n_trials: int, seq_lengths: Sequence[int],
) -> List[int]:
    """Resolve per-trial crop lengths, defaulting to full length; fail closed."""
    if lengths is None:
        return [int(n) for n in seq_lengths]
    if len(lengths) != n_trials:
        raise ValueError(
            f"lengths has {len(lengths)} entries, expected {n_trials}"
        )
    resolved: List[int] = []
    for i, (length, n) in enumerate(zip(lengths, seq_lengths)):
        length = _as_length(length, f"lengths[{i}]")
        if length < 1 or length > n:
            raise ValueError(
                f"lengths[{i}]={length} is outside [1, {n}]"
            )
        resolved.append(length)
    return resolved


def _skill(mse_model: float, mse_base: float) -> Optional[float]:
    """``1 - mse_model/mse_base`` or ``None`` when undefined (never fabricated).

    The ratio is evaluated under overflow/invalid trapping: a finite tiny
    comparator (e.g. ``mse_base=1e-160``) divided into a large finite model MSE
    can overflow to ``inf``, which would make the skill ``-inf``. Such an
    unrepresentable quotient yields ``None`` (unavailable), never a fabricated
    ``-inf``/``NaN``. ``mse_base <= 0`` is also undefined (``None``).
    """
    if not (np.isfinite(mse_model) and np.isfinite(mse_base)) or mse_base <= 0.0:
        return None
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            ratio = mse_model / mse_base
            skill = 1.0 - ratio
    except FloatingPointError:
        return None
    if not (np.isfinite(ratio) and np.isfinite(skill)):
        return None
    return float(skill)


def _as_finite_scalar(value: Any, name: str) -> float:
    """Return ``float(value)``, rejecting bool and non-finite/non-scalar input."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real scalar, got {value!r}")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _as_band(value: Any, name: str) -> float:
    """Validate one strictly-positive finite escape band threshold (cm/s)."""
    out = _as_finite_scalar(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive, got {out}")
    return out


def _as_min_run(value: Any, name: str) -> int:
    """Validate one positive integral sustained-run length (frames)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(
            f"{name} must be an integral min_run, got {value!r} (a non-integral "
            f"min_run is rejected, never truncated)"
        )
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def _parse_float_list(raw: str, name: str) -> List[float]:
    """Parse a comma-separated float list, failing closed when it is empty.

    An empty or all-blank ``raw`` (e.g. the CLI value ``""``) raises
    ``ValueError`` rather than returning ``[]``; downstream code indexes
    ``bands[0]`` for the ``rest`` block and would otherwise raise an opaque
    ``IndexError`` (or silently emit a zero-band run).
    """
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a comma-separated string, got {raw!r}")
    parts = [p.strip() for p in raw.split(",")]
    values = [float(p) for p in parts if p]
    if not values:
        raise ValueError(
            f"{name} must contain at least one comma-separated value, got {raw!r}"
        )
    return values


def _file_sha256(path: Path) -> str:
    """Stream a file's SHA-256 so the run can bind the exact input bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output_dir(out_dir: Path) -> Path:
    """Create ``out_dir`` fresh and refuse to overwrite an existing artifact.

    The output artifact is content-addressed, so re-running into a populated
    directory would silently replace a previous receipt. A non-empty directory
    is refused; the artifact path is opened exclusively (``"x"``) so even a
    race cannot clobber an existing file.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"output_dir is not fresh (already contains files): {out_dir}; "
            f"refusing to overwrite a prior artifact"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "error_decomposition.json"


# ═══════════════════════════════════════════════════════════════
# 1.  Magnitude-band decomposition
# ═══════════════════════════════════════════════════════════════

def decompose_squared_error(
    y_true_seqs: Sequence[np.ndarray],
    y_pred_seqs: Sequence[np.ndarray],
    lengths: Optional[Sequence[int]] = None,
    bands_cm_s: Sequence[float] = _ARTIFACT_BANDS_CM_S,
    *,
    allow_unavailable: bool = False,
) -> Dict[str, Any]:
    """Decompose the pooled masked squared error by target-magnitude band.

    Scores the model against a zero predictor and a lag-one persistence
    predictor on the **same** eligible frames: within each trial only ``t >= 1``
    frames are scored (the persistence comparator needs a predecessor), so no
    frame is ever compared across a trial boundary and the comparator reads
    ``y_true[t-1]``, never a feature channel.

    The band rows are **cumulative and nested** (``|y_true| >= b``): the ``1e4``
    row is a subset of the ``1e3`` row, so their ``sse_share`` / ``frame_share``
    overlap and MUST NOT be summed to a total. Only ``rest``
    (``|y_true| < bands[0]``) is disjoint from the ``bands[0]`` row.

    Args:
        y_true_seqs: Per-trial 1-D ground-truth targets (cm/s), unpadded.
        y_pred_seqs: Per-trial 1-D model predictions (cm/s), aligned
            element-for-element with ``y_true_seqs``.
        lengths: Optional per-trial valid length (crop). Defaults to each
            sequence's full length. Any length outside ``[1, T]`` raises, and a
            non-integral length (e.g. ``2.5`` or a bool) is rejected rather
            than silently truncated.
        bands_cm_s: Strictly increasing absolute-magnitude thresholds (cm/s).
            Each band ``b`` groups eligible frames with ``|y_true| >= b``; the
            ``rest`` block groups ``|y_true| < min(bands_cm_s)``. An empty
            *specification* always raises (scope error). A configured band that
            admits no frame raises by default; see ``allow_unavailable``.
        allow_unavailable: Keyword-only opt-in. When ``True`` a configured band
            that admits zero eligible frames is emitted as an explicit
            ``status="unavailable"`` row with null statistics and a ``reason``,
            instead of raising. The threshold is preserved verbatim (never
            substituted for a narrower one) and no zero is fabricated. When
            ``False`` (the default) such a band still fails closed.

    Returns:
        A JSON-safe dict with pooled ``mse``/``rmse``/``mae``, the eligible
        ``n_trials`` (all trials considered) / ``n_scored_trials`` (those with
        at least one ``t >= 1`` frame) / ``n_frames``, the zero-predictor
        ``zero_mse`` and ``skill_vs_zero``, the lag-one ``baseline_mse`` and
        ``skill_vs_persistence``, a ``bands`` list (one row per threshold with
        ``status``, ``reason``, ``n_frames``, ``frame_share``, ``sse``,
        ``sse_share``, ``mse``, ``rmse``, ``baseline_mse``,
        ``skill_vs_persistence``, ``zero_mse`` and ``skill_vs_zero``) and a
        ``rest`` block. The ``sse_share`` / ``frame_share`` fields are
        explicitly *cumulative nested* shares (see ``share_scope``), not
        disjoint contributions. Undefined statistics are ``None`` (with a
        ``status`` / ``reason`` on unavailable rows) rather than a fabricated
        value.

    Raises:
        ValueError: On misaligned sequence counts, non-1-D or non-finite
            sequences, a length outside the sequence, non-finite or
            non-increasing band thresholds, an empty band list, an empty band
            without ``allow_unavailable``, or unrepresentable (overflowing)
            difference/squared/cumulative sums.
    """
    if len(y_true_seqs) != len(y_pred_seqs):
        raise ValueError(
            f"decompose requires aligned sequences, got {len(y_true_seqs)} "
            f"targets vs {len(y_pred_seqs)} predictions"
        )
    bands = [float(b) for b in bands_cm_s]
    if not bands:
        # Fail closed BEFORE any indexing: ``rest`` is defined relative to
        # ``bands[0]``, so an empty band list would otherwise raise IndexError.
        raise ValueError(
            "decompose_squared_error: bands_cm_s must contain at least one "
            "band threshold (an empty band list is a scope error)"
        )
    for b in bands:
        if not np.isfinite(b) or b <= 0.0:
            raise ValueError(f"band threshold must be finite and positive, got {b}")
    if any(b2 <= b1 for b1, b2 in zip(bands, bands[1:])):
        raise ValueError(f"band thresholds must be strictly increasing, got {bands}")

    true_arrays = [
        _as_finite_1d(t, f"y_true[{i}]") for i, t in enumerate(y_true_seqs)
    ]
    pred_arrays = [
        _as_finite_1d(p, f"y_pred[{i}]") for i, p in enumerate(y_pred_seqs)
    ]
    for i, (t, p) in enumerate(zip(true_arrays, pred_arrays)):
        if t.shape != p.shape:
            raise ValueError(
                f"sequence {i}: y_true {t.shape} != y_pred {p.shape}"
            )
    seq_lengths = [t.shape[0] for t in true_arrays]
    crop = _validate_lengths(lengths, len(true_arrays), seq_lengths)

    true_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []
    prev_all: List[np.ndarray] = []
    for t, p, n in zip(true_arrays, pred_arrays, crop):
        if n < 2:
            continue  # no t>=1 frame; contributes nothing (never cross-trial)
        eligible = np.arange(1, n)
        true_all.append(t[eligible])
        pred_all.append(p[eligible])
        prev_all.append(t[eligible - 1])

    n_trials = len(y_true_seqs)
    if not true_all:
        # No trial contributes an eligible (t >= 1) frame anywhere. Every
        # configured band is therefore empty; the rejection review requires
        # this to FAIL CLOSED rather than emit a null table that could be
        # mistaken for a scored (all-zero) result.
        raise ValueError(
            "decompose_squared_error: empty band scope -- no trial contributes "
            "an eligible t>=1 frame; refusing to emit a null decomposition"
        )

    y_true = np.concatenate(true_all)
    y_pred = np.concatenate(pred_all)
    y_prev = np.concatenate(prev_all)
    assert y_true.shape == y_pred.shape == y_prev.shape, (
        f"aligned eligible arrays, got {y_true.shape}/{y_pred.shape}/{y_prev.shape}"
    )

    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            residual = y_true - y_pred
            squared = residual ** 2
            base_squared = (y_true - y_prev) ** 2
            zero_squared = y_true ** 2
            if not (
                np.isfinite(squared).all()
                and np.isfinite(base_squared).all()
                and np.isfinite(zero_squared).all()
            ):
                raise ValueError("squared error is not representable in float64")
            # The per-frame squares are finite, but their *cumulative* sums can
            # still overflow (e.g. many frames near 1e308); guard the sums too.
            sse = float(squared.sum())
            base_sse = float(base_squared.sum())
            zero_sse = float(zero_squared.sum())
            if not (
                np.isfinite(sse) and np.isfinite(base_sse) and np.isfinite(zero_sse)
            ):
                raise ValueError(
                    "cumulative squared error overflows float64; statistics are "
                    "unavailable (not fabricated)"
                )
    except FloatingPointError as exc:
        raise ValueError(
            "squared error is not representable in float64 (overflow); "
            "statistics are unavailable (not fabricated)"
        ) from exc
    n_frames = int(y_true.size)
    mse = sse / n_frames

    def _block(mask: np.ndarray) -> Dict[str, Any]:
        n = int(mask.sum())
        if n == 0:
            # An empty partition has no finite statistic. Statistics stay
            # ``None`` (never a fabricated zero) with an explicit status/reason
            # so an unavailable partition cannot be read as an all-zero score.
            return {"status": "unavailable", "reason": _EMPTY_BLOCK_REASON,
                    "n_frames": 0, "frame_share": 0.0,
                    "share_scope": _CUMULATIVE_SHARE_SCOPE, "sse": 0.0,
                    "sse_share": None, "mse": None, "rmse": None,
                    "baseline_mse": None, "skill_vs_persistence": None,
                    "zero_mse": None, "skill_vs_zero": None}
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            block_sse = float(squared[mask].sum())
            block_base = float(base_squared[mask].sum()) / n
            block_zero = float(zero_squared[mask].sum()) / n
        if not (
            np.isfinite(block_sse)
            and np.isfinite(block_base)
            and np.isfinite(block_zero)
        ):
            raise ValueError(
                "band squared error overflows float64; band statistics are "
                "unavailable (not fabricated)"
            )
        block_mse = block_sse / n
        return {
            "status": "observed",
            "reason": None,
            "n_frames": n,
            "frame_share": n / n_frames,
            "share_scope": _CUMULATIVE_SHARE_SCOPE,
            "sse": block_sse,
            "sse_share": block_sse / sse if sse > 0.0 else None,
            "mse": block_mse,
            "rmse": float(np.sqrt(block_mse)),
            "baseline_mse": block_base,
            "skill_vs_persistence": _skill(block_mse, block_base),
            "zero_mse": block_zero,
            "skill_vs_zero": _skill(block_mse, block_zero),
        }

    abs_true = np.abs(y_true)
    band_rows: List[Dict[str, Any]] = []
    for b in bands:
        mask = abs_true >= b
        if not mask.any():
            if not allow_unavailable:
                # Fail closed: an empty configured band is a scope error, not a
                # zero-scored band. Emitting an all-null row here would silently
                # hide that the threshold admitted no frame. The opt-in
                # ``allow_unavailable`` surfaces the same fact explicitly.
                raise ValueError(
                    f"empty band: |y_true| >= {b} admits no eligible frame "
                    f"(n_frames={n_frames}); refusing to emit a null band row "
                    f"(pass allow_unavailable=True to report it explicitly)"
                )
        row = {"band_cm_s": b}
        row.update(_block(mask))
        band_rows.append(row)
    rest = _block(abs_true < bands[0])

    return {
        "n_trials": n_trials,
        "n_scored_trials": len(true_all),
        "n_frames": n_frames,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.abs(residual).mean()),
        "zero_mse": zero_sse / n_frames,
        "skill_vs_zero": _skill(mse, zero_sse / n_frames),
        "baseline_mse": base_sse / n_frames,
        "skill_vs_persistence": _skill(mse, base_sse / n_frames),
        "band_share_scope": _CUMULATIVE_SHARE_SCOPE,
        "bands": band_rows,
        "rest": rest,
    }


# ═══════════════════════════════════════════════════════════════
# 2.  Input-channel lag-one identity check
# ═══════════════════════════════════════════════════════════════

def lag_one_identity_check(
    X_seqs: Sequence[np.ndarray],
    y_true_seqs: Sequence[np.ndarray],
    lengths: Optional[Sequence[int]] = None,
    channel: int = 2,
) -> Dict[str, Any]:
    """Check whether feature ``channel`` equals the target's lag-one frame.

    For the frozen NSMoR contract channel ``2`` is ``v_kine(t-1)``, the
    previous-frame velocity. If it is numerically identical to ``y_true(t-1)``
    then the persistence comparator is already an input to the decoder; this
    tests that hypothesis. No over-smoothing or stability claim is inferred
    from the result.

    Only ``t >= 1`` frames are compared (the lag-one slot is undefined at
    ``t = 0`` and is deliberately zero-filled upstream).

    Args:
        X_seqs: Per-trial ``(T, F)`` feature tensors (unpadded).
        y_true_seqs: Per-trial 1-D targets (cm/s), aligned with ``X_seqs``.
        lengths: Optional per-trial valid length (crop); defaults to full.
        channel: Feature column index to test (default ``2``).

    Returns:
        Dict with ``channel``, ``n_frames_compared``, ``n_exact``,
        ``exact_fraction``, ``max_abs_residual``, ``mean_abs_residual`` and
        ``correlation`` (``None`` when a group has zero variance or fewer than
        two frames). A residual that overflows float64 (e.g. ``+1e308`` vs
        ``-1e308``) fails closed rather than emitting ``inf``.

    Raises:
        ValueError: On misaligned counts, a non-2-D ``X``, too few feature
            channels, a length outside ``[1, T]``, non-finite inputs, or an
            unrepresentable residual.
    """
    if len(X_seqs) != len(y_true_seqs):
        raise ValueError(
            f"identity check requires aligned inputs, got {len(X_seqs)} "
            f"features vs {len(y_true_seqs)} targets"
        )
    if channel < 0:
        raise ValueError(f"channel must be non-negative, got {channel}")

    chan_all: List[np.ndarray] = []
    true_prev_all: List[np.ndarray] = []
    seq_lengths: List[int] = []
    for i, (x, y) in enumerate(zip(X_seqs, y_true_seqs)):
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 2:
            raise ValueError(f"X_seqs[{i}] must be 2-D (T, F), got {x_arr.shape}")
        if x_arr.shape[1] <= channel:
            raise ValueError(
                f"X_seqs[{i}] has {x_arr.shape[1]} channels; channel {channel} "
                f"is out of range"
            )
        y_arr = _as_finite_1d(y, f"y_true[{i}]")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"sequence {i}: X has {x_arr.shape[0]} frames, y has {y_arr.shape[0]}"
            )
        seq_lengths.append(int(x_arr.shape[0]))

    crop = _validate_lengths(lengths, len(X_seqs), seq_lengths)
    for i, (x, y, n) in enumerate(zip(X_seqs, y_true_seqs, crop)):
        x_arr = np.asarray(x, dtype=np.float64)[:n, channel]
        if not np.isfinite(x_arr).all():
            raise ValueError(f"X_seqs[{i}] channel {channel} contains non-finite values")
        y_arr = np.asarray(y, dtype=np.float64)[:n]
        if n < 2:
            continue
        eligible = np.arange(1, n)
        chan_all.append(x_arr[eligible])
        true_prev_all.append(y_arr[eligible - 1])

    if not chan_all:
        return {
            "channel": channel, "n_frames_compared": 0, "n_exact": 0,
            "exact_fraction": None, "max_abs_residual": None,
            "mean_abs_residual": None, "correlation": None,
        }

    chan = np.concatenate(chan_all)
    prev = np.concatenate(true_prev_all)
    assert chan.shape == prev.shape, f"aligned channel/prev, got {chan.shape}/{prev.shape}"
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            residual = chan - prev
            if not np.isfinite(residual).all():
                raise ValueError(
                    "lag-one residual overflows float64 (opposite-sign extreme "
                    "values); statistics are unavailable (not fabricated)"
                )
            n = int(chan.size)
            corr: Optional[float] = None
            if n >= 2:
                std_chan = float(np.std(chan))
                std_prev = float(np.std(prev))
                if std_chan > 0.0 and std_prev > 0.0:
                    corr = float(np.corrcoef(chan, prev)[0, 1])
    except FloatingPointError as exc:
        raise ValueError(
            "lag-one residual is not representable in float64 (overflow); "
            "statistics are unavailable (not fabricated)"
        ) from exc
    if corr is not None and not np.isfinite(corr):
        # A near-constant channel/prev pair can make the correlation overflow
        # or be undefined; report None rather than an inf/NaN that would break
        # strict JSON downstream.
        corr = None
    return {
        "channel": channel,
        "n_frames_compared": n,
        "n_exact": int(np.count_nonzero(residual == 0.0)),
        "exact_fraction": float(np.count_nonzero(residual == 0.0) / n),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "mean_abs_residual": float(np.mean(np.abs(residual))),
        "correlation": corr,
    }


# ═══════════════════════════════════════════════════════════════
# 3.  Missingness ledger
# ═══════════════════════════════════════════════════════════════

def missingness_ledger(eligibility_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Describe the retention funnel over ``labeling_eligibility`` rows.

    Each row is one candidate trial with a ``status`` (``"labeled"`` or an
    ``unavailable_*`` drop reason) and optional ``source_channel_condition`` /
    ``label`` fields. The ledger reports the drop reasons by condition and
    label, and it *never* deletes a row to force agreement.

    It reports the *observation* that every dropped row's condition is absent
    from the retained set (``observed_condition_separation``). That is only an
    observed separation: without an independent drop rule it is NOT evidence
    that the missingness is structural / missing-by-design, so
    ``missingness_assumption`` stays ``unresolved_requires_mechanism_analysis``
    whenever any row was dropped. ``biological_completeness`` is likewise
    ``not_established``; no BV-censoring or MCAR/MAR/MNAR mechanism is
    inferred from condition separation alone.

    Args:
        eligibility_rows: Sequence of per-trial mappings. Each must be a
            mapping carrying a non-empty ``status``.

    Returns:
        Dict with ``n_eligibility_rows``, ``n_labeled``, ``n_unavailable``,
        ``by_status``, ``labeled_conditions``, ``unavailable_by_condition``,
        ``unavailable_by_label``, ``observed_condition_separation``,
        ``missingness_assumption``, ``biological_completeness`` and a
        ``mechanism`` label (the single unavailable status, ``"none"``, or
        ``"mixed_statuses"``).

    Raises:
        ValueError: When a row is not a mapping or lacks a non-empty ``status``.
    """
    by_status: Dict[str, int] = {}
    unavail_condition: Dict[str, int] = {}
    unavail_label: Dict[str, int] = {}
    labeled_conditions: set[str] = set()
    n_unavailable = 0
    for i, row in enumerate(eligibility_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"eligibility row {i} must be a mapping, got {type(row).__name__}")
        status = row.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"eligibility row {i} is missing a non-empty 'status'")
        by_status[status] = by_status.get(status, 0) + 1
        condition = str(row.get("source_channel_condition"))
        if status != "labeled":
            n_unavailable += 1
            unavail_condition[condition] = unavail_condition.get(condition, 0) + 1
            label = str(row.get("label"))
            unavail_label[label] = unavail_label.get(label, 0) + 1
        else:
            labeled_conditions.add(condition)

    n_rows = len(eligibility_rows)
    n_labeled = by_status.get("labeled", 0)
    unavailable_statuses = [s for s in by_status if s != "labeled"]
    mechanism = (
        unavailable_statuses[0] if len(unavailable_statuses) == 1
        else ("none" if not unavailable_statuses else "mixed_statuses")
    )
    # Condition separation is an OBSERVATION, not a mechanism: it only says
    # every dropped row's condition is absent from the retained set. Inferring
    # structural / missing-by-design (or BV censoring) from that alone is
    # unjustified without an independent drop rule, so the assumption stays
    # unresolved whenever any row was dropped and biological completeness is
    # never asserted. ``observed_condition_separation`` records the raw fact.
    observed_condition_separation = bool(
        n_unavailable > 0 and not (set(unavail_condition) & labeled_conditions)
    )
    if n_unavailable == 0:
        missingness_assumption = "no_missingness"
    else:
        missingness_assumption = "unresolved_requires_mechanism_analysis"
    return {
        "n_eligibility_rows": n_rows,
        "n_labeled": n_labeled,
        "n_unavailable": n_unavailable,
        "by_status": by_status,
        "labeled_conditions": sorted(labeled_conditions),
        "unavailable_by_condition": unavail_condition,
        "unavailable_by_label": unavail_label,
        "mechanism": mechanism,
        "observed_condition_separation": observed_condition_separation,
        "missingness_assumption": missingness_assumption,
        "biological_completeness": "not_established",
    }


# ═══════════════════════════════════════════════════════════════
# 4.  Formal sustained escape-band sensitivity sweep
# ═══════════════════════════════════════════════════════════════

def _escape_run_masks(
    seqs: Sequence[np.ndarray], band_cm_s: float, min_run: int,
) -> List[np.ndarray]:
    """Per-sequence sustained-run escape membership (train.py semantics).

    Delegates to :func:`scripts.train._sustained_run` so the escape metric is
    byte-identical to the training-time audit; a lone single-frame spike is
    excluded unless ``min_run == 1``. Imported lazily to avoid an import cycle
    and to keep the pure statistics importable without torch.
    """
    from scripts.train import _sustained_run

    masks: List[np.ndarray] = []
    for i, seq in enumerate(seqs):
        arr = _as_finite_1d(seq, f"y_true[{i}]")
        mask = np.asarray(
            _sustained_run(np.abs(arr) >= band_cm_s, min_run=min_run),
            dtype=bool,
        )
        # The sustained-run mask is per-frame and must stay 1-D and aligned
        # with the trial it came from (never a cross-trial shape).
        assert mask.shape == arr.shape, (
            f"escape mask {i} shape {mask.shape} != sequence shape {arr.shape}"
        )
        assert mask.ndim == 1, f"escape mask {i} must be 1-D, got {mask.ndim}-D"
        masks.append(mask)
    return masks


def _count_runs(masks: Sequence[np.ndarray]) -> int:
    """Count contiguous ``True`` runs across per-sequence masks (train.py rule).

    A run that starts at index 0 counts as one event; each ``False -> True``
    transition counts as one. Runs never bridge a trial boundary.
    """
    n_events = 0
    for m in masks:
        if m.size == 0:
            continue
        n_events += int(np.count_nonzero(m[1:] & ~m[:-1]) + int(m[0]))
    return n_events


def escape_sensitivity_sweep(
    y_true_seqs: Sequence[np.ndarray],
    y_pred_seqs: Sequence[np.ndarray],
    bands_cm_s: Sequence[float] = (5.0, 10.0, 20.0, 50.0),
    min_runs: Sequence[int] = (1, 2, 3),
) -> Dict[str, Any]:
    """Sweep the escape band x sustained-run grid with the training metric.

    The ``|y_true| >= band`` table alone is *not* the escape metric: the
    training audit additionally requires a frame to belong to a run of at
    least ``min_run`` consecutive over-band frames, applied per sequence so a
    run never bridges a trial boundary. This function reuses
    ``scripts.train._sustained_run`` so the two definitions cannot drift, and
    rescores the SAME per-trial raw predictions/targets at every
    ``(band, min_run)`` cell — never a cross-trial concatenation.

    Two scopes are reported explicitly and MUST NOT be conflated:

    * ``full`` — every valid frame of every trial, exactly the training-time
      scope (``scripts.train.sweep_escape_sensitivity``);
    * ``aligned`` — the ``t >= 1`` frames only, matching the
      persistence-eligible scope :func:`decompose_squared_error` scores.

    The sustained-run mask is computed ONCE per trial on the full valid
    sequence and then sliced to ``t >= 1`` for the aligned scope; it is never
    recomputed on the sliced array (which would split a run that began at
    ``t = 0``). Each scope's escape and resting partitions carry their OWN
    availability: a partition that admits no frame is explicitly
    ``unavailable`` with a reason (never a fabricated zero, never ``inf``),
    even when the other partition or the other scope does admit frames — e.g. a
    singleton over-band trial is an escape on ``full`` but its single ``t >= 1``
    frame is not, so the aligned escape partition is unavailable though the
    aligned denominator is non-zero. A cell with no sustained escape frame on
    either scope is still emitted (``status="unavailable"`` + reason + valid
    zero counts), so the full Cartesian grid is always retained.

    Args:
        y_true_seqs: Per-trial 1-D raw targets (cm/s), unpadded.
        y_pred_seqs: Per-trial 1-D raw predictions (cm/s), aligned.
        bands_cm_s: Positive finite escape-band thresholds (cm/s).
        min_runs: Positive integral sustained-run lengths (frames).

    Returns:
        Dict with the Cartesian ``rows`` list (``band_cm_s``, ``min_run``,
        ``status``, ``reason``, ``n_escape_frames_full``/``_aligned``,
        ``n_escape_events_full``/``_aligned``, ``escape_rmse_full``/``_aligned``,
        ``resting_rmse_full``/``_aligned``, ``escape_ratio_full``/``_aligned``,
        the full/aligned frame + trial denominators, and per-scope ``full`` /
        ``aligned`` blocks each carrying ``escape_status``/``escape_reason``,
        ``rest_status``/``rest_reason`` and an overall ``status``), the
        top-level ``full`` / ``aligned`` denominator blocks, and the raw
        ``max_abs_y_true`` observed on the scored scope (a single
        explicitly-scoped value, never a corpus-wide claim).

    Raises:
        ValueError: On misaligned/non-1-D/non-finite sequences, a non-integral
            ``min_run``, an empty band/min_run *specification*, or an
            unrepresentable (overflowing) squared/cumulative error.
    """
    if len(y_true_seqs) != len(y_pred_seqs):
        raise ValueError(
            f"escape sweep requires aligned sequences, got {len(y_true_seqs)} "
            f"targets vs {len(y_pred_seqs)} predictions"
        )
    bands = [_as_band(b, f"bands_cm_s[{i}]") for i, b in enumerate(bands_cm_s)]
    runs = [_as_min_run(m, f"min_runs[{i}]") for i, m in enumerate(min_runs)]
    if not bands or not runs:
        raise ValueError("escape sweep requires at least one band and min_run")

    true_arrays = [
        _as_finite_1d(t, f"y_true[{i}]") for i, t in enumerate(y_true_seqs)
    ]
    pred_arrays = [
        _as_finite_1d(p, f"y_pred[{i}]") for i, p in enumerate(y_pred_seqs)
    ]
    for i, (t, p) in enumerate(zip(true_arrays, pred_arrays)):
        if t.shape != p.shape:
            raise ValueError(f"sequence {i}: y_true {t.shape} != y_pred {p.shape}")
    n_trials = len(true_arrays)
    n_frames_full = int(sum(t.size for t in true_arrays))
    # Aligned scope drops t=0 of every trial that has >=1 frame (the lag-one
    # slot is undefined there), matching decompose_squared_error's t>=1 scope.
    n_frames_aligned = int(sum(max(t.size - 1, 0) for t in true_arrays))

    rows: List[Dict[str, Any]] = []
    for band in bands:
        for mr in runs:
            # Mask computed on the FULL valid trial, then sliced to t>=1; never
            # recomputed on the sliced array.
            masks_full = _escape_run_masks(true_arrays, band, mr)
            # Per-trial squared error, guarded against overflow before any
            # accumulation.
            sq_full: List[np.ndarray] = []
            try:
                with np.errstate(over="raise", invalid="raise", under="ignore"):
                    for t, p in zip(true_arrays, pred_arrays):
                        sq = (p - t) ** 2
                        if not np.isfinite(sq).all():
                            raise ValueError(
                                f"escape sweep squared error overflows float64 "
                                f"at band={band}, min_run={mr}; statistics are "
                                f"unavailable (not fabricated)"
                            )
                        sq_full.append(sq)
            except FloatingPointError as exc:
                raise ValueError(
                    f"escape sweep squared error is not representable in "
                    f"float64 at band={band}, min_run={mr}; statistics are "
                    f"unavailable (not fabricated)"
                ) from exc
            masks_aligned = [m[1:] for m in masks_full]
            sq_aligned = [s[1:] for s in sq_full]
            n_esc_full = int(sum(int(m.sum()) for m in masks_full))
            n_esc_aligned = int(sum(int(m.sum()) for m in masks_aligned))
            n_ev_full = _count_runs(masks_full)
            n_ev_aligned = _count_runs(masks_aligned)

            def _partition(
                sq_list: List[np.ndarray], mask_list: List[np.ndarray],
                select_escape: bool, kind: str,
            ) -> Dict[str, Any]:
                """Escape or resting RMSE with its OWN status/reason.

                Availability is per partition, not per cell: an empty escape
                set on a non-zero denominator (e.g. singleton over-band trial
                whose aligned slice is not an escape) is explicitly unavailable
                with a reason, never a fabricated value or ``inf``.
                """
                vals = [
                    s[m] if select_escape else s[~m]
                    for s, m in zip(sq_list, mask_list)
                    if (m.any() if select_escape else (~m).any())
                ]
                if not vals:
                    return {
                        "rmse": None,
                        "status": "unavailable",
                        "reason": (
                            _SWEEP_NO_ESCAPE_REASON if select_escape
                            else _SWEEP_NO_REST_REASON
                        ),
                    }
                try:
                    with np.errstate(over="raise", invalid="raise", under="ignore"):
                        mean = float(np.concatenate(vals).mean())
                except FloatingPointError as exc:
                    raise ValueError(
                        f"escape sweep {kind} mean squared error is not "
                        f"representable in float64 at band={band}, min_run={mr}; "
                        f"statistics are unavailable (not fabricated)"
                    ) from exc
                if not np.isfinite(mean):
                    raise ValueError(
                        f"escape sweep {kind} mean squared error overflows "
                        f"float64 at band={band}, min_run={mr}; statistics are "
                        f"unavailable (not fabricated)"
                    )
                return {"rmse": float(np.sqrt(mean)), "status": "observed",
                        "reason": None}

            esc_full = _partition(sq_full, masks_full, True, "escape")
            esc_aligned = _partition(sq_aligned, masks_aligned, True, "escape")
            rest_full = _partition(sq_full, masks_full, False, "resting")
            rest_aligned = _partition(
                sq_aligned, masks_aligned, False, "resting",
            )

            def _scope_status(
                esc: Dict[str, Any], rest: Dict[str, Any],
            ) -> str:
                if esc["status"] == "observed":
                    return "observed"
                if rest["status"] == "observed":
                    return "partial"
                return "unavailable"

            # Cell-level status/reason kept for backward compatibility: an
            # observed escape on either scope makes the cell "observed"; a cell
            # with no escape on both scopes keeps the explicit empty reason.
            empty = n_esc_full == 0 and n_esc_aligned == 0
            status = (
                "unavailable" if empty
                else ("observed" if esc_full["status"] == "observed"
                      else "partial")
            )
            rows.append({
                "band_cm_s": band,
                "min_run": mr,
                "status": status,
                "reason": _SWEEP_EMPTY_REASON if empty else None,
                # Valid zero counts even on an empty cell; statistics are null.
                "n_escape_frames_full": n_esc_full,
                "n_escape_frames_aligned": n_esc_aligned,
                "n_escape_events_full": n_ev_full,
                "n_escape_events_aligned": n_ev_aligned,
                # Per-scope availability: escape and resting each carry their
                # own status/reason so an empty partition on a non-zero
                # denominator cannot be read as a zero statistic.
                "full": {
                    "scope": "all valid frames of all trials (train.py scope)",
                    "n_frames": n_frames_full,
                    "escape_status": esc_full["status"],
                    "escape_reason": esc_full["reason"],
                    "rest_status": rest_full["status"],
                    "rest_reason": rest_full["reason"],
                    "status": _scope_status(esc_full, rest_full),
                },
                "aligned": {
                    "scope": "t>=1 frames (persistence-eligible, decompose scope)",
                    "n_frames": n_frames_aligned,
                    "escape_status": esc_aligned["status"],
                    "escape_reason": esc_aligned["reason"],
                    "rest_status": rest_aligned["status"],
                    "rest_reason": rest_aligned["reason"],
                    "status": _scope_status(esc_aligned, rest_aligned),
                },
                "escape_rmse_full": esc_full["rmse"],
                "escape_rmse_aligned": esc_aligned["rmse"],
                "resting_rmse_full": rest_full["rmse"],
                "resting_rmse_aligned": rest_aligned["rmse"],
                "escape_ratio_full": (
                    n_esc_full / n_frames_full if n_frames_full else None
                ),
                "escape_ratio_aligned": (
                    n_esc_aligned / n_frames_aligned if n_frames_aligned else None
                ),
                "n_trials_full": n_trials,
                "n_frames_full": n_frames_full,
                "n_trials_aligned": sum(
                    1 for t in true_arrays if t.size >= 2
                ),
                "n_frames_aligned": n_frames_aligned,
            })

    max_abs = float(max(
        (np.abs(t).max() for t in true_arrays if t.size), default=0.0,
    ))
    return {
        "n_trials": n_trials,
        "n_frames": n_frames_full,
        "max_abs_y_true": max_abs,
        # Two explicitly-named denominators; never conflated, never summed.
        "denominators": {
            "full": {
                "scope": "all valid frames of all trials (train.py scope)",
                "n_trials": n_trials,
                "n_frames": n_frames_full,
            },
            "aligned": {
                "scope": "t>=1 frames (persistence-eligible, decompose scope)",
                "n_frames": n_frames_aligned,
            },
            "note": (
                "full and aligned are different denominators; do not sum or "
                "conflate; no run ever bridges a trial boundary"
            ),
        },
        "rows": rows,
    }


# ═══════════════════════════════════════════════════════════════
# 5.  Padding/masking vs removal audit
# ═══════════════════════════════════════════════════════════════

def padding_masking_audit(
    batch_true_lengths: Optional[Sequence[Sequence[int]]] = None,
    batch_padded_lengths: Optional[Sequence[int]] = None,
    unavailable_trials: Optional[int] = 0,
    *,
    scope: str = "unavailable",
) -> Dict[str, Any]:
    """Separate batch padding (masked) from unavailable trials (removed).

    Two different mechanisms are conflated by a bare frame count. Batch
    padding inserts zero-filled frames up to the batch's max length; those
    frames are *masked* (retained in the tensor, never scored) — not removed.
    Unavailable trials (e.g. ``unavailable_no_stimulus_anchor``) are *removed*
    upstream, before any tensor exists.

    The padding count is only truthful when it is measured from the tensors the
    evaluated loader actually yielded (``collate_variable_length`` /
    ``collate_with_metadata`` pad each batch to its own ``max`` row length).
    Passing raw dataset lengths here is a *different* scope and would
    manufacture a corpus-wide padding count that the loader never produced.
    The default ``scope="unavailable"`` therefore FAILS CLOSED: it reports the
    padding count as unavailable rather than inventing it. Callers that did
    observe real batches must pass ``scope="observed_batches"``.

    Args:
        batch_true_lengths: Per-batch list of true (unpadded) per-trial
            lengths, measured from observed batches.
        batch_padded_lengths: Per-batch padded tensor length (``x_batch.size(1)``).
        unavailable_trials: Count of upstream-removed unavailable trials, or
            ``None`` when the ``labeling_eligibility`` ledger is absent
            (UNKNOWN, not zero). It is a dataset-artifact fact, so it is
            reported even when the padding scope is unavailable.
        scope: ``"observed_batches"`` to compute the padding count from the
            supplied lengths; ``"unavailable"`` (default) to fail closed.

    Returns:
        Dict with ``status`` (``"observed"`` or ``"unavailable"``), a
        ``reason``, ``scope``, ``n_batches``, ``scored_frames`` (sum of true
        lengths), ``padding_frames_masked`` (padded minus true, summed),
        ``unavailable_trials_removed`` and the explicit
        ``unavailable_trials_removed_scope`` label
        (``"full_corpus_labeling_eligibility"``). The removal count is a
        full-corpus ledger fact and must never be read as a validation-population
        denominator; under ``status="unavailable"`` the two frame counts are
        ``None``.

    Raises:
        ValueError: On an unknown ``scope``, a length-list count mismatch, a
            true length exceeding its padded length, or a non-integral/
            non-positive length.
    """
    if unavailable_trials is None:
        # Absent ledger: the removed count is UNKNOWN. Report null, never a
        # fabricated zero that would read as "nothing was removed".
        removed: Optional[int] = None
    else:
        removed = _as_length(unavailable_trials, "unavailable_trials")
        if removed < 0:
            raise ValueError(f"unavailable_trials must be >= 0, got {removed}")
    if scope == "unavailable":
        return {
            "status": "unavailable",
            "reason": (
                "batch padding cannot be counted from the evaluated scope: "
                "raw dataset lengths are not the loader's per-batch collate "
                "padding; pass scope='observed_batches' with the true/padded "
                "lengths actually observed in the evaluated loader"
            ),
            "scope": scope,
            "n_batches": None,
            "scored_frames": None,
            "padding_frames_masked": None,
            "unavailable_trials_removed": removed,
            "unavailable_trials_removed_scope": _REMOVED_SCOPE,
        }
    if scope != "observed_batches":
        raise ValueError(
            f"scope must be 'observed_batches' or 'unavailable', got {scope!r}"
        )
    if batch_true_lengths is None or batch_padded_lengths is None:
        raise ValueError(
            "scope='observed_batches' requires both batch_true_lengths and "
            "batch_padded_lengths"
        )
    if len(batch_true_lengths) != len(batch_padded_lengths):
        raise ValueError(
            f"got {len(batch_true_lengths)} true-length groups but "
            f"{len(batch_padded_lengths)} padded lengths"
        )
    scored = 0
    masked = 0
    for b, (true_group, padded) in enumerate(
        zip(batch_true_lengths, batch_padded_lengths)
    ):
        padded_len = _as_length(padded, f"batch_padded_lengths[{b}]")
        if padded_len < 1:
            raise ValueError(f"batch_padded_lengths[{b}] must be >= 1")
        for i, true_len in enumerate(true_group):
            n = _as_length(true_len, f"batch_true_lengths[{b}][{i}]")
            if n < 1 or n > padded_len:
                raise ValueError(
                    f"batch {b} trial {i}: true length {n} is outside "
                    f"[1, {padded_len}] (padding cannot be shorter than a "
                    f"retained trial)"
                )
            scored += n
            masked += padded_len - n
    return {
        "status": "observed",
        "reason": None,
        "scope": scope,
        "n_batches": len(batch_true_lengths),
        "scored_frames": scored,
        "padding_frames_masked": masked,
        "unavailable_trials_removed": removed,
        "unavailable_trials_removed_scope": _REMOVED_SCOPE,
    }


# ═══════════════════════════════════════════════════════════════
# 6.  With-vs-without unavailable-trial comparison
# ═══════════════════════════════════════════════════════════════

def unavailable_trial_comparison(
    with_unavailable: Optional[Mapping[str, Any]] = None,
    without_unavailable: Optional[Mapping[str, Any]] = None,
    n_unavailable_trials: Optional[int] = 0,
    unavailable_trials_replayable: bool = False,
) -> Dict[str, Any]:
    """Compare metrics with vs without the unavailable anchor trials.

    The ``unavailable_no_stimulus_anchor`` trials were dropped upstream; a
    like-for-like comparison requires replaying them, which is only possible
    if their raw per-frame predictions/targets are available. When they are
    not (the default here), this FAILS CLOSED with ``status="unavailable"``
    and a reason rather than fabricating a comparison from mismatched scopes.

    Args:
        with_unavailable: Optional ``{"mse": ..., "n_frames": ...}`` block for
            the scope that includes the unavailable trials.
        without_unavailable: Optional block for the retained-only scope.
        n_unavailable_trials: Count of dropped unavailable trials.
        unavailable_trials_replayable: Whether both blocks were actually
            computed from replayed data.

    Returns:
        Dict with ``status`` (``"not_applicable"`` when nothing was dropped,
        ``"unavailable"`` when the dropped count is unknown OR not replayable,
        else ``"computed"``), the two blocks (or ``None``), and ``delta_mse``
        when computed.

    Raises:
        ValueError: When ``unavailable_trials_replayable`` is set but either
            block is missing, or a block's ``mse`` is not finite.
    """
    if n_unavailable_trials is None:
        # Unknown dropped count (absent ledger): cannot assert there was
        # nothing to compare, so fail closed rather than report
        # not_applicable (which would imply a verified zero).
        return {
            "status": "unavailable",
            "n_unavailable_trials": None,
            "reason": (
                "unavailable-trial count is unknown (no labeling_eligibility "
                "ledger); refusing to report not_applicable as if verified zero"
            ),
            "with_unavailable": None,
            "without_unavailable": None,
            "delta_mse": None,
        }
    n_unavailable = _as_length(n_unavailable_trials, "n_unavailable_trials")
    if n_unavailable < 0:
        raise ValueError(f"n_unavailable_trials must be >= 0, got {n_unavailable}")
    if n_unavailable == 0:
        return {
            "status": "not_applicable",
            "n_unavailable_trials": 0,
            "reason": "no_unavailable_trials_to_compare",
            "with_unavailable": None,
            "without_unavailable": None,
            "delta_mse": None,
        }
    if not unavailable_trials_replayable:
        return {
            "status": "unavailable",
            "n_unavailable_trials": n_unavailable,
            "reason": (
                "unavailable trials cannot be replayed from the accepted "
                "artifacts; refusing to fabricate a with/without comparison"
            ),
            "with_unavailable": None,
            "without_unavailable": None,
            "delta_mse": None,
        }
    if with_unavailable is None or without_unavailable is None:
        raise ValueError(
            "replayable comparison requires both with_unavailable and "
            "without_unavailable metric blocks"
        )
    with_mse = _as_finite_scalar(with_unavailable["mse"], "with_unavailable.mse")
    without_mse = _as_finite_scalar(
        without_unavailable["mse"], "without_unavailable.mse"
    )
    # The subtraction is guarded: two finite MSEs of opposite extreme sign can
    # overflow float64. An unrepresentable delta is reported as None (with a
    # reason) rather than an inf that would break strict JSON.
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            delta_mse: Optional[float] = float(with_mse - without_mse)
    except FloatingPointError:
        delta_mse = None
    if delta_mse is not None and not np.isfinite(delta_mse):
        delta_mse = None
    return {
        "status": "computed",
        "n_unavailable_trials": n_unavailable,
        "reason": (
            None if delta_mse is not None
            else "delta_mse is not representable in float64 (overflow); "
                 "the two blocks are reported but the difference is unavailable"
        ),
        "with_unavailable": dict(with_unavailable),
        "without_unavailable": dict(without_unavailable),
        "delta_mse": delta_mse,
    }


# ═══════════════════════════════════════════════════════════════
# 7.  Runnable Stage-0 diagnostic (evaluation only)
# ═══════════════════════════════════════════════════════════════

#: Default accepted artifacts (pinned). These are the ONLY genuine
#: long-run model + corpus on disk; the config-only defaults do not exist.
_DEFAULT_CKPT = (
    ".scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/"
    "train-causal-formal-compact-qfadtol8/best_model.pth"
)
_DEFAULT_DATA = (
    ".scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/"
    "etl-causal-final-zp12m0fa/nsmor_dataset.pt"
)
_DEFAULT_NEST = (
    ".scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/"
    "nested-causal-complete-dku91_m0/nested_split_seed42.pt"
)


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage-0 NSMoR error decomposition and missingness ledger "
                    "(evaluation only; no training, no model/loss change)."
    )
    parser.add_argument("--checkpoint", default=_DEFAULT_CKPT,
                        help="Accepted NSMoRCore checkpoint (best_model.pth).")
    parser.add_argument("--dataset", default=_DEFAULT_DATA,
                        help="Pinned ETL dataset artifact.")
    parser.add_argument("--nested_prior_artifact", default=_DEFAULT_NEST,
                        help="Pinned nested outer-validation split artifact.")
    parser.add_argument("--output_dir", required=True,
                        help="Fresh directory for error_decomposition.json.")
    parser.add_argument(
        "--bands",
        default=",".join(f"{b:g}" for b in _ARTIFACT_BANDS_CM_S),
        help="Comma-separated |y_true| magnitude bands (cm/s). Every band must "
             "admit at least one frame; an empty band raises. Defaults to the "
             "artifact bands 1000,10000 (the decomposition's original scope); "
             "pass --bands to override explicitly.",
    )
    parser.add_argument("--sweep_bands", default="5,10,20,50",
                        help="Comma-separated escape-band thresholds (cm/s) for "
                             "the sustained-run sensitivity sweep.")
    parser.add_argument("--min_runs", default="1,2,3",
                        help="Comma-separated sustained-run lengths (frames).")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=2400)
    parser.add_argument("--pre_anchor_frames", type=int, default=1200)
    parser.add_argument("--channel", type=int, default=2,
                        help="Feature channel to test for lag-one identity.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the Stage-0 diagnostic and write ``error_decomposition.json``.

    Reconstructs the accepted checkpoint's pinned nested outer-validation set
    through the same loader chain the persistence comparator used, then reports
    the high-magnitude band MSE share, the zero/lag-one skill table, the
    channel-2 lag-one identity, the formal sustained escape-band sweep, the
    padding-vs-removal audit, the with/without-unavailable comparison (which
    fails closed when the dropped rows cannot be replayed), and the
    missingness ledger. Evaluation only.

    ``--output_dir`` must be fresh: a populated directory is refused and the
    artifact is written exclusively (never overwritten). The exact SHA-256 of
    the checkpoint/dataset/nested-split inputs is hashed before the run,
    recorded in the receipt, and re-checked afterwards so input drift fails
    closed instead of producing a mixed receipt.
    """
    import json
    import logging

    import torch

    from nsmor.analysis.analysis_priors import load_analysis_priors
    from nsmor.analysis.prediction_units import (
        load_model_from_checkpoint,
        resolve_dt_ms,
    )
    from nsmor.config import DEFAULT_FEATURE
    from nsmor.dataloader_factory import create_optimized_dataloader
    from nsmor.model_utils import validate_dataset_provenance
    from nsmor.nsmor_dataloader import NSMoRDataset
    from nsmor.pipeline.conditions import derive_anchor_frames
    from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s -- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("error_decomposition")

    args = _build_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Empty-band CLI values (e.g. --bands "") fail closed here, before any
    # indexing, rather than surfacing an opaque IndexError deep in the stats.
    bands = _parse_float_list(args.bands, "bands")
    sweep_bands = _parse_float_list(args.sweep_bands, "sweep_bands")
    min_runs: List[int] = []
    for raw_run in _parse_float_list(args.min_runs, "min_runs"):
        if not raw_run.is_integer():
            # Match _as_min_run: a non-integral min_run is rejected, never
            # silently truncated to a different sweep grid.
            raise ValueError(f"min_runs must be integral, got {raw_run!r}")
        min_runs.append(int(raw_run))
    out_path = _prepare_output_dir(Path(args.output_dir))

    # Bind the exact input bytes: hashed BEFORE and re-checked AFTER the run so
    # a concurrent mutation of a checkpoint/dataset/split cannot be silently
    # scored (drift fails closed).
    input_paths = [
        Path(args.checkpoint), Path(args.dataset), Path(args.nested_prior_artifact),
    ]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"input artifact does not exist: {p}")
    inputs_before = {str(p): _file_sha256(p) for p in input_paths}

    # ── 1. Model (restores physical cm/s units) ────────────────────
    model = load_model_from_checkpoint(Path(args.checkpoint), device)
    dt_ms = resolve_dt_ms(model)

    # ── 2. Dataset via the exact fingerprint loader chain ──────────
    dataset, loaded_fp = load_dataset_with_fingerprint(
        Path(args.dataset), map_location="cpu", expected_dt_ms=dt_ms,
        restore_provenance=False,
    )
    validate_dataset_provenance(dataset, Path(args.dataset))
    mcmc_priors, val_indices = load_analysis_priors(
        dataset, Path(args.dataset), Path(args.nested_prior_artifact), model,
        loaded_source_fingerprint=loaded_fp,
    )
    val_indices = [int(i) for i in val_indices]
    logger.info("Outer-validation trials: %d", len(val_indices))

    X_seqs = dataset["X_seqs"]
    Y_seqs = dataset["Y_seqs"]
    lengths = dataset["lengths"]
    anchor_frames = dataset.get("anchor_frames")
    if anchor_frames is None:
        anchor_frames = derive_anchor_frames(X_seqs, lengths)

    # ── 3. Missingness ledger (all candidate trials, not just val) ──
    # An absent ``labeling_eligibility`` ledger is UNKNOWN, never "removed=0":
    # reporting 0 would masquerade as a verified no-missingness fact and could
    # silently drop the removed-trial scope. Fail closed with an explicit
    # null/unavailable ledger + reason instead.
    eligibility = dataset.get("labeling_eligibility")
    if eligibility is None:
        ledger: Optional[Dict[str, Any]] = {
            "status": "unavailable",
            "reason": (
                "dataset carries no labeling_eligibility ledger; the removed-"
                "trial count and missingness description are UNKNOWN (not "
                "zero) and must not be reported as no_missingness"
            ),
            "n_eligibility_rows": None,
            "n_labeled": None,
            "n_unavailable": None,
            "by_status": None,
            "labeled_conditions": None,
            "unavailable_by_condition": None,
            "unavailable_by_label": None,
            "observed_condition_separation": None,
            "mechanism": None,
            "missingness_assumption": "unavailable",
            "biological_completeness": "not_established",
        }
    else:
        ledger = missingness_ledger(eligibility)
    n_unavailable_ledger: Optional[int] = (
        int(ledger["n_unavailable"]) if ledger["n_unavailable"] is not None else None
    )

    # ── 4. Val-only loader, matching the accepted comparator scope ──
    sequences = [(X_seqs[i], Y_seqs[i], 0) for i in val_indices]
    val_priors = mcmc_priors[val_indices]
    val_anchor_frames = [int(anchor_frames[i]) for i in val_indices]
    val_dataset = NSMoRDataset(
        sequences=sequences,
        mcmc_priors=val_priors,
        feature_config=dataset.get("feature_config", DEFAULT_FEATURE),
        max_seq_len=args.max_seq_len,
        pre_anchor_frames=args.pre_anchor_frames,
        anchor_frames=val_anchor_frames,
        source_indices=val_indices,
        is_pure_wind=None,
    )
    loader = create_optimized_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    # ── 5. Single model pass → per-trial raw cm/s preds/targets ────
    model.eval()
    y_true_seqs: List[np.ndarray] = []
    y_pred_seqs: List[np.ndarray] = []
    X_crop_seqs: List[np.ndarray] = []
    crop_lengths: List[int] = []
    # Per-batch observed lengths, captured from the tensors the loader actually
    # yielded, so the padding audit is bound to real collate semantics.
    observed_true_lengths: List[List[int]] = []
    observed_padded_lengths: List[int] = []
    with torch.no_grad():
        for batch in loader:
            assert len(batch) == 3, f"Expected 3-tuple val batch, got {len(batch)}"
            x_batch, y_batch, batch_lengths = batch
            x_batch = x_batch.to(device).contiguous()
            batch_lengths = batch_lengths.to(device).contiguous()
            y_pred, _ = model(x_batch, batch_lengths, return_internals=True)
            observed_padded_lengths.append(int(x_batch.size(1)))
            observed_true_lengths.append(
                [int(batch_lengths[i]) for i in range(x_batch.size(0))]
            )
            for i in range(x_batch.size(0)):
                n = int(batch_lengths[i])
                assert y_pred[i, :n].shape == (n,), "prediction crop shape"
                y_true_seqs.append(y_batch[i, :n].cpu().numpy().astype(np.float64))
                y_pred_seqs.append(y_pred[i, :n].cpu().numpy().astype(np.float64))
                X_crop_seqs.append(x_batch[i, :n].cpu().numpy().astype(np.float64))
                crop_lengths.append(n)

    # ── 6. Diagnostics ─────────────────────────────────────────────
    # The band decomposition is scoped to the t>=1 (persistence-eligible)
    # frames, so it is scored on the SAME frames as the persistence
    # comparator below. ``allow_unavailable=True`` (keyword-only, opt-in) lets a
    # *configured* band that admits no frame on this scope (e.g. the artifact
    # bands 1000/10000 cm/s against max|y_true|~139.78 cm/s) be reported as an
    # explicit unavailable row with null statistics rather than aborting the
    # run. The thresholds are preserved verbatim and no zero is fabricated.
    decomposition = decompose_squared_error(
        y_true_seqs, y_pred_seqs, bands_cm_s=bands,
        allow_unavailable=True,
    )
    identity = lag_one_identity_check(
        X_crop_seqs, y_true_seqs, channel=args.channel,
    )
    # Formal sustained escape-band sweep, using train.py's _sustained_run
    # semantics (not a bare |y_true|>=band table).
    sweep = escape_sensitivity_sweep(
        y_true_seqs, y_pred_seqs, bands_cm_s=sweep_bands, min_runs=min_runs,
    )
    # Padding/masking vs removal: both are stated explicitly rather than
    # conflated. The unavailable anchor trials are removed upstream (they never
    # reach the loader); batch padding is masked, not removed. The padding count
    # is measured from the OBSERVED evaluated batches (real collate semantics),
    # never from the raw dataset lengths.
    padding_audit = padding_masking_audit(
        batch_true_lengths=observed_true_lengths,
        batch_padded_lengths=observed_padded_lengths,
        unavailable_trials=n_unavailable_ledger,
        scope="observed_batches",
    )
    # With-vs-without the unavailable trials. Their raw per-frame
    # predictions/targets are not retained in the accepted artifacts, so this
    # FAILS CLOSED (status="unavailable") instead of fabricating a comparison.
    unavailable_cmp = unavailable_trial_comparison(
        n_unavailable_trials=n_unavailable_ledger,
        unavailable_trials_replayable=False,
    )

    # Verify the input bytes are byte-identical to those hashed before the run.
    # Any drift means the scored artifact is not the one the receipt names, so
    # the run fails closed rather than emitting a stale/mixed receipt.
    inputs_after = {str(p): _file_sha256(p) for p in input_paths}
    if inputs_after != inputs_before:
        changed = [k for k in inputs_before if inputs_before[k] != inputs_after.get(k)]
        raise RuntimeError(
            f"input artifact changed during the run (refusing a mixed receipt): "
            f"{changed}"
        )

    result: Dict[str, Any] = {
        "scope": {
            "evaluation": "checkpoint_selected_nested_outer_validation",
            "descriptive_only": True,
            "untouched_holdout": False,
            "independent_animal_inference": False,
            "evaluated_checkpoint": str(args.checkpoint),
            "evaluated_dataset": str(args.dataset),
            "nested_prior_artifact": str(args.nested_prior_artifact),
            # Content binding: the exact SHA-256 of every input artifact, so the
            # receipt names bytes, not merely a path that could later change.
            "input_sha256": dict(inputs_before),
            "bands_cm_s": bands,
            "channel": args.channel,
            "escape_sweep_bands_cm_s": sweep_bands,
            "escape_sweep_min_runs": min_runs,
            "max_abs_y_true_scope": "checkpoint_selected_nested_outer_validation",
            # Denominator scopes are stated explicitly so the full-corpus
            # missingness ledger (2432 rows / 128 unavailable) is never read as
            # the validation population, and the observed-batch padding audit is
            # never read as corpus-wide.
            "n_val_trials_scope": "checkpoint_selected_nested_outer_validation",
            # ``n_frames_scored`` is the FULL valid per-trial length (the loader
            # crop, before the t>=1 lag-one drop), summed over the nested
            # outer-validation trials. It is NOT the t>=1 aligned scope the
            # decomposition/identity comparators score; that is
            # ``artifact_decomposition.n_frames``. The label states the scope so
            # the two denominators are never conflated.
            "n_frames_scored_scope": (
                "checkpoint_selected_nested_outer_validation_full_valid_length_"
                "pre_t_ge_1"
            ),
            "padding_masking_audit_scope": (
                "validation_observed_batches"
                if padding_audit["status"] == "observed" else "unavailable"
            ),
            "missingness_ledger_scope": (
                "full_corpus_labeling_eligibility"
                if ledger.get("status") != "unavailable"
                else "unavailable_no_ledger"
            ),
            "unavailable_trials_removed_scope": _REMOVED_SCOPE,
        },
        "n_val_trials": len(val_indices),
        "n_frames_scored": int(sum(crop_lengths)),
        "artifact_decomposition": decomposition,
        "lag_one_identity": identity,
        "escape_sensitivity": sweep,
        "padding_masking_audit": padding_audit,
        "unavailable_trial_comparison": unavailable_cmp,
        "missingness_ledger": ledger,
    }
    # Exclusive create ("x"): even a race cannot clobber a prior artifact.
    with open(out_path, "x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, allow_nan=False)
    logger.info("Wrote %s", out_path)
    print(json.dumps({
        "artifact_decomposition": {
            k: decomposition[k]
            for k in ("n_frames", "mse", "baseline_mse", "skill_vs_persistence",
                      "skill_vs_zero", "bands", "rest")
        },
        "lag_one_identity": identity,
        "escape_sensitivity": sweep,
        "padding_masking_audit": padding_audit,
        "unavailable_trial_comparison": unavailable_cmp,
        "missingness_ledger": ledger,
    }, indent=2))
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
