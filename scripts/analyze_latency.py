"""Stimulus-anchored latency contrast analysis (ADR 0012).

Reads a model-internals artifact (``expected_delay``, ``realized_delay``,
``lengths``, ``source_indices``, ``dt_ms``, ``max_seq_len``,
``pre_anchor_frames``), its original processed dataset and the frozen latency
target table. Verifies physical conditions, saved anchors and units, then
reports the pre-registered target-vs-reference contrast with the statistical
machinery the project requires:

* per-prefix (unweighted) means, so a prefix with many trials gets ONE weight;
* Cohen's d effect size on the prefix-level residual ``d - L_obs``;
* Shapiro-Wilk normality and Levene variance-homogeneity diagnostics;
* a Welch t-test as the primary test and Mann-Whitney U as a rank sensitivity;
* Holm and Benjamini-Hochberg adjustment over the declared test family;
* a deterministic prefix bootstrap CI for the contrast.

The primary estimand is ``Delta L_obs = Delta L_fixed + Delta L_proc``. Only
the condition-relative difference is identified; ``L_fixed`` being
condition-independent is a declared assumption, not a result. Processing time
is supervised, not emergent. ``realized_delay`` is a secondary DESCRIPTION and
is excluded from the primary test family.

The CLI consumes synthetic or real internals artifacts; it never trains and
never reads the protected raw-data directories by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import stats as sps

# Direct CLI execution must resolve this checkout, including in a worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsmor.loss import stimulus_anchored_expected_delay
from nsmor.pipeline.latency_metadata import (
    build_latency_sidecar,
    load_latency_target_table,
    validate_latency_target_definition,
)


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (family-wise error control).

    Args:
        pvalues: Raw p-values, any finite non-negative floats.

    Returns:
        Adjusted p-values in the input order, each in ``[0, 1]``.

    Raises:
        ValueError: On an empty/non-finite/out-of-range input.
    """
    values = np.asarray(list(pvalues), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pvalues must be a non-empty 1-D sequence")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("pvalues must be finite and lie in [0, 1]")
    m = values.size
    order = np.argsort(values)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def bh_adjust(pvalues: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg FDR-adjusted p-values.

    Args:
        pvalues: Raw p-values, any finite non-negative floats.

    Returns:
        Adjusted p-values in the input order, each in ``[0, 1]``.

    Raises:
        ValueError: On an empty/non-finite/out-of-range input.
    """
    values = np.asarray(list(pvalues), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pvalues must be a non-empty 1-D sequence")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("pvalues must be finite and lie in [0, 1]")
    m = values.size
    order = np.argsort(values)[::-1]
    adjusted = np.empty(m, dtype=np.float64)
    running = 1.0
    for rank_from_top, idx in enumerate(order):
        rank = m - rank_from_top
        running = min(running, values[idx] * m / rank)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def _prefix_means(
    d_values: np.ndarray,
    observed: np.ndarray,
    group_ids: np.ndarray,
    prefix_ids: np.ndarray,
    *,
    target_group: int,
    reference_group: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-prefix means of ``d`` and ``observed``, split by group.

    Args:
        d_values: ``(N,)`` anchor-frame expected delay.
        observed: ``(N,)`` observed latency in frames.
        group_ids: ``(N,)`` condition ids.
        prefix_ids: ``(N,)`` recording-prefix ids.
        target_group: Pre-registered target condition id.
        reference_group: Pre-registered reference condition id.

    Returns:
        ``(target_d, target_obs, reference_d, reference_obs)``, each a
        per-prefix mean vector.

    Raises:
        ValueError: If a prefix spans both groups.
    """
    target_d: List[float] = []
    target_obs: List[float] = []
    reference_d: List[float] = []
    reference_obs: List[float] = []
    selected = np.isin(group_ids, (target_group, reference_group))
    for prefix in np.unique(prefix_ids[selected]).tolist():
        rows = selected & (prefix_ids == prefix)
        groups = np.unique(group_ids[rows])
        if groups.size != 1:
            raise ValueError(
                f"prefix {prefix} spans multiple groups {groups.tolist()}; "
                "the contrast is not identifiable"
            )
        if int(groups[0]) == target_group:
            target_d.append(float(np.mean(d_values[rows])))
            target_obs.append(float(np.mean(observed[rows])))
        else:
            reference_d.append(float(np.mean(d_values[rows])))
            reference_obs.append(float(np.mean(observed[rows])))
    return (
        np.asarray(target_d, dtype=np.float64),
        np.asarray(target_obs, dtype=np.float64),
        np.asarray(reference_d, dtype=np.float64),
        np.asarray(reference_obs, dtype=np.float64),
    )


def _cohens_d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Pooled-SD Cohen's d for ``mean(a) - mean(b)`` (``None`` if undefined)."""
    if a.size < 2 or b.size < 2:
        return None
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = math.sqrt(
        ((a.size - 1) * var_a + (b.size - 1) * var_b)
        / (a.size + b.size - 2)
    )
    if not math.isfinite(pooled) or pooled <= 0.0:
        return None
    return float((float(np.mean(a)) - float(np.mean(b))) / pooled)


def _bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> Optional[Tuple[float, float]]:
    """Percentile bootstrap CI for ``mean(a) - mean(b)`` (prefix resampling)."""
    if a.size < 1 or b.size < 1:
        return None
    generator = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        diffs[i] = (
            float(np.mean(generator.choice(a, size=a.size, replace=True)))
            - float(np.mean(generator.choice(b, size=b.size, replace=True)))
        )
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def _safe_test(name: str, fn, *args, **kwargs) -> Dict[str, Any]:
    """Run a scipy test, returning an explicit unavailable reason on failure."""
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:  # scipy raises on degenerate inputs
        return {"test": name, "available": False, "reason": str(exc)}
    statistic = float(result.statistic)
    pvalue = float(result.pvalue)
    if not (math.isfinite(statistic) and math.isfinite(pvalue)):
        return {
            "test": name,
            "available": False,
            "reason": "non-finite test statistic (degenerate input)",
        }
    return {
        "test": name,
        "available": True,
        "statistic": statistic,
        "pvalue": pvalue,
    }


def latency_contrast_report(
    expected_delay: torch.Tensor,
    realized_delay: torch.Tensor,
    anchor_frames: torch.Tensor,
    target_frames: torch.Tensor,
    target_mask: torch.Tensor,
    group_ids: torch.Tensor,
    prefix_ids: torch.Tensor,
    lengths: torch.Tensor,
    *,
    target_group: int,
    reference_group: int,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    """Compute the pre-registered latency contrast and its statistics.

    Args:
        expected_delay: ``(B, T)`` differentiable per-source expected delay.
        realized_delay: ``(B, T)`` post-arbitration emitted delay (secondary).
        anchor_frames: ``(B,)`` int64 stimulus anchors in saved coordinates.
        target_frames: ``(B,)`` observed latency in frames.
        target_mask: ``(B,)`` bool observed-latency mask.
        group_ids: ``(B,)`` int64 condition ids.
        prefix_ids: ``(B,)`` int64 recording-prefix ids.
        lengths: ``(B,)`` true sequence lengths.
        target_group: Pre-registered target condition id.
        reference_group: Pre-registered reference condition id.
        n_bootstrap: Prefix bootstrap resamples.
        bootstrap_seed: Private bootstrap seed (deterministic report).

    Returns:
        A JSON-serialisable report.  ``available=False`` with a ``reason``
        when a group has too few prefixes; the contrast is never silently
        changed.

    Raises:
        ValueError: On malformed tensors, a non-distinct group pair, or a
            prefix that spans both groups.
    """
    B, T = expected_delay.shape
    assert expected_delay.shape == (B, T)
    if realized_delay.shape != (B, T):
        raise ValueError(
            f"realized_delay shape {tuple(realized_delay.shape)} != ({B}, {T})"
        )
    for name, value, shape in (
        ("anchor_frames", anchor_frames, (B,)),
        ("target_frames", target_frames, (B,)),
        ("target_mask", target_mask, (B,)),
        ("group_ids", group_ids, (B,)),
        ("prefix_ids", prefix_ids, (B,)),
        ("lengths", lengths, (B,)),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")
    if target_group == reference_group:
        raise ValueError("target_group and reference_group must be distinct")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int):
        raise ValueError("n_bootstrap must be an int")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")

    d_all = stimulus_anchored_expected_delay(
        expected_delay.detach().double(), anchor_frames,
    ).cpu().numpy()
    realized_all = realized_delay.detach().double().gather(
        1, anchor_frames.unsqueeze(1),
    ).squeeze(1).cpu().numpy()
    observed_all = target_frames.detach().double().cpu().numpy()
    mask = target_mask.cpu().numpy().astype(bool)
    groups = group_ids.cpu().numpy().astype(np.int64)
    prefixes = prefix_ids.cpu().numpy().astype(np.int64)
    lengths_np = lengths.cpu().numpy().astype(np.int64)

    in_contrast = mask & np.isin(groups, (target_group, reference_group))
    if not in_contrast.any():
        return {
            "available": False,
            "reason": "no_masked_trials_in_contrast_groups",
            "target_group": int(target_group),
            "reference_group": int(reference_group),
        }
    if bool((anchor_frames.cpu().numpy() >= lengths_np).any()):
        raise ValueError("an anchor frame lies outside its trial length")

    for name, values in (
        ("expected_delay", d_all), ("realized_delay", realized_all),
        ("target_frames", observed_all),
    ):
        if not np.isfinite(values[in_contrast]).all():
            raise ValueError(f"{name} must be finite on included contrast trials")

    t_d, t_obs, r_d, r_obs = _prefix_means(
        d_all[in_contrast], observed_all[in_contrast],
        groups[in_contrast], prefixes[in_contrast],
        target_group=target_group, reference_group=reference_group,
    )
    target_res = t_d - t_obs
    reference_res = r_d - r_obs
    if target_res.size < 2 or reference_res.size < 2:
        return {
            "available": False,
            "reason": "fewer_than_two_prefixes_in_a_group",
            "n_target_prefixes": int(target_res.size),
            "n_reference_prefixes": int(reference_res.size),
            "target_group": int(target_group),
            "reference_group": int(reference_group),
        }

    # Prefix-level (unweighted) group means: a prefix with many trials gets
    # ONE weight, so delta_d - delta_t equals the prefix residual contrast.
    delta_d = float(np.mean(t_d) - np.mean(r_d))
    delta_t = float(np.mean(t_obs) - np.mean(r_obs))
    delta = float(np.mean(target_res) - np.mean(reference_res))

    shapiro_t = (
        _safe_test("shapiro_target", sps.shapiro, target_res)
        if target_res.size >= 3 else {
            "test": "shapiro_target", "available": False,
            "reason": "n < 3",
        }
    )
    shapiro_r = (
        _safe_test("shapiro_reference", sps.shapiro, reference_res)
        if reference_res.size >= 3 else {
            "test": "shapiro_reference", "available": False,
            "reason": "n < 3",
        }
    )
    levene = _safe_test("levene", sps.levene, target_res, reference_res)
    welch = _safe_test("welch_t", sps.ttest_ind, target_res, reference_res,
                       equal_var=False)
    mann = _safe_test("mann_whitney", sps.mannwhitneyu, target_res,
                      reference_res, alternative="two-sided")

    family = [welch, mann]
    raw_p = [t["pvalue"] for t in family if t.get("available")]
    adjusted = {"holm": None, "bh": None}
    if raw_p:
        adjusted["holm"] = holm_adjust(raw_p)
        adjusted["bh"] = bh_adjust(raw_p)

    rt_d, rt_obs, rr_d, rr_obs = _prefix_means(
        realized_all[in_contrast], observed_all[in_contrast],
        groups[in_contrast], prefixes[in_contrast],
        target_group=target_group, reference_group=reference_group,
    )
    realized_secondary = {
        "description": "secondary, excluded from the primary test family",
        "delta_realized_delay_frames": float(np.mean(rt_d) - np.mean(rr_d)),
        "note": (
            "target-indexed realized_delay evaluated at the anchor target frame; "
            "post-arbitration emitted delay, not the anchor source expected delay"
        ),
    }
    if rt_d.size >= 2 and rr_d.size >= 2:
        realized_secondary["cohens_d"] = _cohens_d(
            rt_d - rt_obs, rr_d - rr_obs,
        )

    report = {
        "available": True,
        "target_group": int(target_group),
        "reference_group": int(reference_group),
        "n_target_prefixes": int(target_res.size),
        "n_reference_prefixes": int(reference_res.size),
        "n_target_trials": int(
            (in_contrast & (groups == target_group)).sum()
        ),
        "n_reference_trials": int(
            (in_contrast & (groups == reference_group)).sum()
        ),
        "delta_d_frames": delta_d,
        "delta_t_frames": delta_t,
        "delta_residual_frames": delta,
        "cohens_d": _cohens_d(target_res, reference_res),
        "bootstrap_ci95": _bootstrap_ci(
            target_res, reference_res,
            n_bootstrap=n_bootstrap, seed=bootstrap_seed,
        ),
        "normality": [shapiro_t, shapiro_r],
        "variance_homogeneity": levene,
        "primary_test": welch,
        "sensitivity_test": mann,
        "family_size": len(raw_p),
        "adjusted_pvalues": adjusted,
        "realized_delay_secondary": realized_secondary,
        "assumptions": [
            "L_fixed is condition-independent; only Delta L_obs is identified",
            "processing time is supervised, not emergent",
            "prefix-level unweighted aggregation (conditions nested in prefixes)",
        ],
    }
    # Also reject arithmetic overflow in reductions/bootstrap/effect sizes.
    # Never emit nonstandard JSON NaN/Infinity, even from finite inputs.
    try:
        json.dumps(report, allow_nan=False)
    except ValueError as exc:
        raise ValueError("latency report reductions must be finite") from exc
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: analyse a model-internals artifact against the target table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internals", required=True,
        help=".pt dict with expected_delay, realized_delay, lengths, "
             "source_indices",
    )
    parser.add_argument("--latency-target-table", required=True)
    parser.add_argument(
        "--dataset", default=None,
        help="Original processed dataset, for source-row/condition/crop checks",
    )
    parser.add_argument("--target-group", type=int, required=True)
    parser.add_argument("--reference-group", type=int, required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    payload = torch.load(args.internals, map_location="cpu", weights_only=True)
    required = ("expected_delay", "realized_delay", "lengths", "source_indices")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"internals artifact is missing {missing}")
    table = load_latency_target_table(args.latency_target_table)
    dt_ms = payload.get("dt_ms")
    if (
        isinstance(dt_ms, bool) or not isinstance(dt_ms, (int, float))
        or not math.isfinite(dt_ms) or dt_ms <= 0
    ):
        raise ValueError("internals artifact requires a finite positive dt_ms")
    validate_latency_target_definition(
        table, target_group=args.target_group, reference_group=args.reference_group,
        dt_ms=dt_ms,
    )
    if args.dataset is None:
        raise ValueError("--dataset is required to verify latency anchor alignment")
    from nsmor.nsmor_dataloader import NSMoRDataset
    from nsmor.pipeline.conditions import derive_anchor_frames, resolve_anchor_crop
    from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint

    raw, _ = load_dataset_with_fingerprint(
        args.dataset, map_location="cpu", expected_dt_ms=dt_ms,
        restore_provenance=False,
    )
    for key in ("X_seqs", "Y_seqs"):
        if key not in raw:
            raise ValueError(f"dataset is missing {key}")
    anchors = raw.get("anchor_frames")
    if anchors is None:
        anchors = derive_anchor_frames(raw["X_seqs"], raw.get("lengths"))
    if "max_seq_len" not in payload or "pre_anchor_frames" not in payload:
        raise ValueError("internals artifact requires dataset crop settings")
    max_len, pre = payload["max_seq_len"], payload["pre_anchor_frames"]
    if (max_len is not None and (
        isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 1
    )) or isinstance(pre, bool) or not isinstance(pre, int) or pre < 0:
        raise ValueError("invalid internals dataset crop settings")
    indices = payload["source_indices"]
    if (not isinstance(indices, torch.Tensor) or indices.ndim != 1
            or indices.dtype != torch.long):
        raise ValueError("source_indices must be a 1-D long tensor")
    rows = indices.tolist()
    if len(set(rows)) != len(rows):
        raise ValueError("source_indices must be unique")
    n = len(rows)
    if any(not 0 <= row < len(raw["X_seqs"]) for row in rows):
        raise ValueError("source_indices outside dataset row range")
    dataset = NSMoRDataset(
        [(np.asarray(raw["X_seqs"][row]), np.asarray(raw["Y_seqs"][row]), 0)
         for row in rows],
        mcmc_priors=np.full((n, 4), 0.25, dtype=np.float32),
        source_indices=rows, anchor_frames=[anchors[row] for row in rows],
        max_seq_len=max_len, pre_anchor_frames=pre,
    )
    saved_lengths = []
    for sequence, anchor in zip(dataset.sequences, dataset.anchor_frames):
        start, end = resolve_anchor_crop(
            n_frames=len(sequence[0]), anchor_frame=int(anchor),
            max_seq_len=max_len, pre_anchor_frames=pre,
        )
        saved_lengths.append(end - start)
    if payload["lengths"].tolist() != saved_lengths:
        raise ValueError("internals lengths disagree with dataset crop")
    sidecar = build_latency_sidecar(table, rows, dataset=dataset)
    report = latency_contrast_report(
        payload["expected_delay"],
        payload["realized_delay"],
        sidecar["latency_anchor_frames"],
        sidecar["latency_target_frames"],
        sidecar["latency_target_mask"],
        sidecar["latency_group_id"],
        sidecar["latency_prefix_id"],
        payload["lengths"],
        target_group=args.target_group,
        reference_group=args.reference_group,
        n_bootstrap=args.bootstrap,
        bootstrap_seed=args.seed,
    )
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
