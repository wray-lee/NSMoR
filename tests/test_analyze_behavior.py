"""Focused tests for ``scripts/analyze_behavior.py``.

Covers the claims a reviewer can falsify cheaply:

1. The Miller bound / violation-area primitives are correct on constructed
   race (independence) and coactivation distributions, the bound is clipped
   at 1, and a wind->collision SOA shift moves it right.
2. The prefix-cluster bootstrap is a TRUE with-replacement cluster
   bootstrap (prefix multiplicity weights), verified against an independent
   reference implementation on synthetic clusters.
3. The B2 primary inclusion rule (stationary at the trigger) excludes
   pre-trigger movers and makes latency 0 impossible.
4. The small-sample / power routines behave (monotone, exact on known cases).
5. ``main()`` runs end to end on a tiny synthetic corpus.

The synthetic corpus is written with ``torch.save`` and read back through
the same restricted decoder the script uses, so the test exercises the real
load path rather than a mock.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import analyze_behavior as ab  # noqa: E402


# ══════════════════════════════════════════════════════════════════════
# 1. Miller bound / violation area
# ══════════════════════════════════════════════════════════════════════
def test_miller_bound_matches_independence_race() -> None:
    """A race whose AV CDF equals the bound yields zero violation area."""
    dt = 10.0
    tgrid = np.arange(0.0, 1000.0 + dt, dt)
    rng = np.random.default_rng(0)
    rate = 1.0 / 200.0
    n = 2000
    lat_a = np.sort(-np.log(1.0 - rng.uniform(0, 0.999, n)) / rate)
    lat_v = np.sort(-np.log(1.0 - rng.uniform(0, 0.999, n)) / rate)
    bound = ab.miller_bound(tgrid, lat_a, n, lat_v, n, [0.0], [1.0])
    fav = 1.0 - (1.0 - ab.ecdf(lat_a, tgrid, n)) * (1.0 - ab.ecdf(lat_v, tgrid, n))
    prof = ab.violation_profile(fav, bound, tgrid)
    assert prof["area"] < 1e-9, prof
    assert prof["time_range"] is None
    assert 0.0 <= prof["peak_time_ms"] <= 1000.0


def test_miller_bound_flags_coactivation() -> None:
    """A coactivation AV CDF strictly above the bound has positive area."""
    dt = 10.0
    tgrid = np.arange(0.0, 1000.0 + dt, dt)
    n = 2000
    rng = np.random.default_rng(1)
    rate = 1.0 / 200.0
    lat_a = np.sort(-np.log(1.0 - rng.uniform(0, 0.999, n)) / rate)
    lat_v = np.sort(-np.log(1.0 - rng.uniform(0, 0.999, n)) / rate)
    bound = ab.miller_bound(tgrid, lat_a, n, lat_v, n, [0.0], [1.0])
    fav = np.clip(1.0 - np.exp(-tgrid / 60.0), 0.0, 1.0)
    prof = ab.violation_profile(fav, bound, tgrid)
    assert prof["area"] > 0.0
    assert prof["time_range"] is not None
    assert prof["time_range"][1] >= prof["time_range"][0]


def test_miller_bound_clipped_at_one() -> None:
    """The bound never exceeds 1 even when F_A + F_V would (M1)."""
    dt = 10.0
    tgrid = np.arange(0.0, 1000.0 + dt, dt)
    n = 200
    lat_a = np.linspace(50.0, 300.0, n)
    lat_v = np.linspace(50.0, 300.0, n)
    bound = ab.miller_bound(tgrid, lat_a, n, lat_v, n, [0.0], [1.0])
    assert np.all(bound <= 1.0 + 1e-12), bound.max()
    # Late in the grid F_A ~ 1 and F_V ~ 1, so an unclipped bound would be 2.
    assert bound[-1] == pytest.approx(1.0)


def test_soa_shift_moves_bound() -> None:
    """A larger wind->collision SOA shifts the visual term right (lower bound early)."""
    dt = 10.0
    tgrid = np.arange(0.0, 1000.0 + dt, dt)
    n = 500
    lat_a = np.linspace(50.0, 300.0, n)
    lat_v = np.linspace(50.0, 300.0, n)
    b0 = ab.miller_bound(tgrid, lat_a, n, lat_v, n, [0.0], [1.0])
    b1 = ab.miller_bound(tgrid, lat_a, n, lat_v, n, [200.0], [1.0])
    # The visual term is shifted later, so the bound is no larger anywhere.
    assert np.all(b1 <= b0 + 1e-12), (b1 - b0).max()
    assert b1[-1] == pytest.approx(b0[-1])


def test_permutation_p_value_never_zero() -> None:
    """The +1 correction keeps the one-sided permutation p strictly positive."""
    null = np.linspace(0.0, 10.0, 100)
    p = ab.permutation_p_value(100.0, null)
    assert p is not None and p == pytest.approx(1.0 / 101.0)
    assert ab.permutation_p_value(5.0, null) > 0.5


# ══════════════════════════════════════════════════════════════════════
# 2. True with-replacement prefix-cluster bootstrap (B1)
# ══════════════════════════════════════════════════════════════════════
def test_bootstrap_weights_are_with_replacement_multiplicities() -> None:
    """Each replicate draws exactly len(pool) prefixes per stratum, with repeats."""
    prefix_keys = np.array(["p1", "p1", "p2", "p2", "p3", "p3"], dtype=object)
    strata = np.array(["g1", "g1", "g1", "g1", "g2", "g2"], dtype=object)
    rng = np.random.default_rng(0)
    saw_multiplicity_gt1 = False
    for w in ab.bootstrap_weights(prefix_keys, strata, 200, rng):
        assert w.shape == prefix_keys.shape
        # A prefix is taken whole: its trials share one multiplicity.
        for p in np.unique(prefix_keys):
            assert len(set(w[prefix_keys == p].tolist())) == 1
        # Stratum g1 has 2 prefixes -> exactly 2 draws (multiplicity sum 2).
        mult_g1 = [w[prefix_keys == p][0] for p in ("p1", "p2")]
        mult_g2 = [w[prefix_keys == p][0] for p in ("p3",)]
        assert sum(mult_g1) == pytest.approx(2.0)
        assert sum(mult_g2) == pytest.approx(1.0)
        if w.max() > 1.0:
            saw_multiplicity_gt1 = True
    assert saw_multiplicity_gt1, "no replicate ever drew a prefix twice"


def test_bootstrap_matches_reference_cluster_bootstrap() -> None:
    """The weighted bootstrap matches an independent explicit cluster bootstrap.

    The reference implementation is fully self-contained (it does NOT reuse
    ``ab.prefix_indices``): it draws whole prefixes with replacement by hand,
    CONCATENATES the drawn clusters' trial values explicitly (a drawn-twice
    prefix contributes its trials twice), and averages.  The module's
    multiplicity-weighted mean must agree distributionally (mean and
    2.5/50/97.5 percentiles within the Monte-Carlo tolerance sd/sqrt(B)).
    """
    rng = np.random.default_rng(7)
    n_pref, per = 6, 5
    prefix_keys = np.array([f"p{i}" for i in range(n_pref) for _ in range(per)],
                           dtype=object)
    strata = np.array(
        ["g0" if i < n_pref // 2 else "g1" for i in range(n_pref) for _ in range(per)],
        dtype=object,
    )
    values = rng.normal(10.0, 3.0, prefix_keys.shape[0])
    b = 3000

    def weighted_mean(w: np.ndarray) -> float:
        return float(np.sum(w * values) / np.sum(w))

    got = np.array([weighted_mean(w)
                    for w in ab.bootstrap_weights(prefix_keys, strata, b, rng)])

    # Independent reference: build the per-prefix value blocks by hand, draw
    # whole clusters with replacement per stratum, CONCATENATE them, average.
    ref_rng = np.random.default_rng(99)
    blocks = {f"p{i}": values[i * per:(i + 1) * per] for i in range(n_pref)}
    pools = {"g0": [f"p{i}" for i in range(n_pref // 2)],
             "g1": [f"p{i}" for i in range(n_pref // 2, n_pref)]}
    ref = []
    for _ in range(b):
        acc = []
        for pool in pools.values():
            picks = ref_rng.choice(np.asarray(pool, dtype=object), size=len(pool))
            for p in picks:
                acc.append(blocks[p])
        ref.append(float(np.concatenate(acc).mean()))
    ref = np.asarray(ref)
    # Monte-Carlo tolerance: sd of the bootstrap mean / sqrt(B), inflated a bit.
    tol = 4.0 * ref.std(ddof=1) / np.sqrt(b)
    assert abs(got.mean() - ref.mean()) < tol, (got.mean(), ref.mean(), tol)
    for q in (2.5, 50.0, 97.5):
        assert abs(np.percentile(got, q) - np.percentile(ref, q)) < 3.0 * tol


def test_bootstrap_rejects_mixed_stratum_prefix() -> None:
    """A prefix spanning two groups is refused (fail closed)."""
    prefix_keys = np.array(["p1", "p1", "p2"], dtype=object)
    strata = np.array(["g1", "g2", "g1"], dtype=object)
    with pytest.raises(ValueError):
        list(ab.bootstrap_weights(prefix_keys, strata, 1, np.random.default_rng(0)))


def test_bootstrap_prefix_draws_removed() -> None:
    """D7: the unused bootstrap_prefix_draws helper was deleted."""
    assert not hasattr(ab, "bootstrap_prefix_draws")
    assert not hasattr(ab, "weighted_quantile")
    assert not hasattr(ab, "benjamini_hochberg")


# ══════════════════════════════════════════════════════════════════════
# 2b. X2 Holm family-wise correction
# ══════════════════════════════════════════════════════════════════════
def test_holm_adjust_known_values() -> None:
    """Holm on a known family: monotone, capped at 1, NaN preserved."""
    # m = 5.  Sorted raw: 0.0025, 0.005, 0.030, 0.115, 0.115.
    raw = [0.0015, 0.030, 0.005, 0.0025, 0.115]
    adj = ab.holm_adjust(raw)
    # 0.0015 * 5 = 0.0075 (smallest); 0.0025 * 4 = 0.01; 0.005 * 3 = 0.015;
    # 0.030 * 2 = 0.06; 0.115 * 1 = 0.115 -> monotone already.
    assert adj[0] == pytest.approx(0.0075)
    assert adj[3] == pytest.approx(0.010)
    assert adj[2] == pytest.approx(0.015)
    assert adj[1] == pytest.approx(0.060)
    assert adj[4] == pytest.approx(0.115)
    # Adjusted p is aligned by index; it is not itself monotone in the input.


def test_holm_adjust_enforces_monotonicity_and_cap() -> None:
    """Enforced monotonicity lifts a later step above its raw product."""
    # m = 4; sorted raw 0.01, 0.01, 0.01, 0.04 -> products 0.04, 0.03, 0.02,
    # 0.04; the 2nd/3rd products fall below the running max, so all are 0.04.
    adj = ab.holm_adjust([0.04, 0.01, 0.01, 0.01])
    assert adj == [pytest.approx(0.04)] * 4
    # A family where m*p exceeds 1 is capped at 1.
    adj2 = ab.holm_adjust([0.9, 0.8])
    assert adj2[0] == pytest.approx(1.0)
    assert adj2[1] == pytest.approx(1.0)


def test_holm_adjust_excludes_nan_from_family_size() -> None:
    """An unavailable (NaN) test is excluded from m and stays NaN."""
    adj = ab.holm_adjust([0.01, float("nan"), 0.02])
    assert adj[1] != adj[1]  # NaN
    assert adj[0] == pytest.approx(0.02)  # m = 2
    assert adj[2] == pytest.approx(0.02)


# ══════════════════════════════════════════════════════════════════════
# 3. B2 primary inclusion rule
# ══════════════════════════════════════════════════════════════════════
def _corpus_from_trials(
    specs: list[tuple[str, np.ndarray, np.ndarray]], session: str = "a_s_1",
) -> dict:
    """Assemble a minimal corpus dict for detect_responses()."""
    x = [np.zeros((y.shape[0], 2), dtype=np.float32) for _, y, _ in specs]
    cond = np.array([c for c, _, _ in specs], dtype=object)
    ttc = np.array([np.nan for _ in specs], dtype=float)
    return {
        "x": x, "y": [y for _, y, _ in specs], "conditions": cond,
        "ttc": ttc, "n_trials": len(specs),
        "lengths": np.array([y.shape[0] for _, y, _ in specs], dtype=int),
    }


def test_stationary_rule_excludes_pretrigger_mover() -> None:
    """A trial moving at the trigger is ineligible and gets no latency (B2)."""
    n = 200
    ref = 100
    y_still = np.zeros(n, dtype=np.float32)
    y_still[ref + 5 : ref + 20] = 30.0  # escape strictly after the trigger
    y_mover = y_still.copy()
    y_mover[ref - 10] = 25.0  # already moving before the trigger
    corpus = _corpus_from_trials(
        [("multisensory", y_still, None), ("multisensory", y_mover, None)]
    )
    events = {"ref": np.array([ref, ref]), "v_col": np.array([-1, -1])}
    out = ab.detect_responses(corpus, events, dt_ms=4.0)
    assert out["eligible"].tolist() == [True, False]
    assert np.isfinite(out["latency_ms"][0])
    assert np.isnan(out["latency_ms"][1])


def test_latency_is_strictly_positive() -> None:
    """A response AT the trigger frame does not count (latency 0 impossible)."""
    # escape_frame with strict=True must skip the lower-bound frame itself.
    y = np.zeros(20, dtype=np.float32)
    y[5] = 40.0
    assert ab.escape_frame(y, 10.0, 5, 20, strict=False) == 5
    assert ab.escape_frame(y, 10.0, 5, 20, strict=True) == -1

    # Integration: stationary at the trigger, first crossing at ref+1 -> 4 ms.
    n, ref = 60, 30
    y2 = np.zeros(n, dtype=np.float32)
    y2[ref + 1] = 40.0
    corpus = _corpus_from_trials([("multisensory", y2, None)])
    events = {"ref": np.array([ref]), "v_col": np.array([-1])}
    out = ab.detect_responses(corpus, events, dt_ms=4.0)
    assert out["eligible"][0]
    assert out["latency_ms"][0] == pytest.approx(4.0)
    assert out["latency_ms"][0] > 0.0


def test_s1_labeling_responder_is_eligibility_gated() -> None:
    """X1: the s1 sustained responder obeys the D1 gate; ungated is separate.

    A trial that is MOVING at the trigger is ineligible, so it cannot be a
    D1 labeling responder even when a sustained > 5 cm/s run follows the
    trigger.  The same sustained run is still recorded in the separately
    named ``resp_label_ungated`` diagnostic.
    """
    n, ref = 400, 100
    y = np.zeros(n, dtype=np.float32)
    y[ref - 10] = 25.0  # moving at the trigger -> ineligible
    y[ref + 2 : ref + 70] = 30.0  # sustained > 5 cm/s run after the trigger
    corpus = _corpus_from_trials([("multisensory", y, None)])
    events = {"ref": np.array([ref]), "v_col": np.array([-1])}
    out = ab.detect_responses(corpus, events, dt_ms=4.0)
    assert out["eligible"].tolist() == [False]
    assert out["resp_label"][0] == -1  # gated: no labeling responder
    assert np.isnan(out["latency_label_ms"][0])
    assert out["resp_label_ungated"][0] == ref + 2  # diagnostic keeps it
    assert out["latency_label_ungated_ms"][0] == pytest.approx(8.0)


def test_visual_response_cannot_precede_eligibility_window() -> None:
    """D1: a visual_only responder must cross STRICTLY AFTER collision.

    Round 2 searched the visual response over [collision-2000, collision+1000]
    while gating eligibility over [collision-250, collision], so a trial that
    was stationary at collision but moved long before it was certified as a
    responder with NEGATIVE latency.  Under D1 the only eligible crossing is
    the one strictly after the collision trigger.
    """
    n, ref = 600, 500  # collision at frame 500
    y = np.zeros(n, dtype=np.float32)
    # A pre-collision burst well inside the round-2 window but before the gate.
    y[ref - 400 : ref - 380] = 40.0
    corpus = _corpus_from_trials([("visual_only", y, None)])
    events = {"ref": np.array([ref]), "v_col": np.array([ref])}
    out = ab.detect_responses(corpus, events, dt_ms=4.0)
    # Stationary at the trigger -> eligible; but the pre-collision burst is
    # NOT a valid D1 response.
    assert out["eligible"][0]
    assert out["resp"][0] == -1
    assert np.isnan(out["latency_ms"][0])
    # The legacy s3 window still sees the pre-collision burst (negative latency).
    assert out["resp_legacy"][0] >= 0
    assert out["latency_legacy_ms"][0] < 0.0


def test_d1_window_covers_collision_plus_1000ms() -> None:
    """D1: a visual crossing at collision + 1000 ms is inside the window."""
    n, ref = 1200, 400
    y = np.zeros(n, dtype=np.float32)
    y[ref + int(round(1000.0 / 4.0))] = 40.0  # collision + 1000 ms
    corpus = _corpus_from_trials([("visual_only", y, None)])
    events = {"ref": np.array([ref]), "v_col": np.array([ref])}
    out = ab.detect_responses(corpus, events, dt_ms=4.0)
    assert out["eligible"][0]
    assert out["resp"][0] == ref + 250
    assert out["latency_ms"][0] == pytest.approx(1000.0)


def test_d1_pre_trigger_burst_is_not_a_primary_response() -> None:
    """D1: a burst before the trigger is not a primary response.

    The trial stays ELIGIBLE (the burst is 280 ms before the trigger, outside
    the 250 ms stationarity gate), but the primary window starts AT the
    trigger, so the pre-trigger burst is invisible to it; the legacy s3
    collision-anchored window still sees it with a negative latency.
    """
    n, ref = 200, 120
    y = np.zeros(n, dtype=np.float32)
    y[ref - 70] = 40.0  # 280 ms before the trigger, outside the gate
    corpus = _corpus_from_trials([("visual_only", y, None)])
    events = {"ref": np.array([ref]), "v_col": np.array([ref])}
    out = ab.detect_responses(corpus, events, dt_ms=4.0)
    assert out["eligible"][0]
    assert out["resp"][0] == -1
    assert np.isnan(out["latency_ms"][0])
    assert out["resp_legacy"][0] == ref - 70
    assert out["latency_legacy_ms"][0] < 0.0


# ══════════════════════════════════════════════════════════════════════
# 4. Small-sample / power routines (M3, B5)
# ══════════════════════════════════════════════════════════════════════
def test_small_sample_test_detects_shift() -> None:
    """A clear between-prefix shift is significant; the permutation p is finite."""
    rng = np.random.default_rng(0)
    x = rng.normal(120.0, 10.0, 12)
    y = rng.normal(80.0, 10.0, 14)
    out = ab.small_sample_two_sample(x, y, rng)
    assert out["mean_diff"] > 0
    assert out["permutation_p_two_sided"] is not None
    assert out["permutation_p_two_sided"] < 0.01
    assert out["welch_p"] < 0.01


def test_power_two_sample_decreases_with_effect() -> None:
    """A larger effect size needs no more prefixes."""
    n_small = ab.power_two_sample(delta=20.0, sd=30.0)
    n_big = ab.power_two_sample(delta=40.0, sd=30.0)
    assert n_small is not None and n_big is not None
    assert n_big <= n_small


def test_power_one_sample_handles_degenerate() -> None:
    """Zero effect / zero SD returns None instead of looping forever."""
    assert ab.power_one_sample(mean=0.0, sd=10.0) is None
    assert ab.power_one_sample(mean=5.0, sd=0.0) is None


def test_random_intercept_lmm_recovers_known_effects() -> None:
    """D7: the REML random-intercept fit recovers a known fixed effect.

    Data are generated as ``y = 100 + 40*x + b_group + eps`` with a known
    between-group SD; the recovered intercept and slope must be close and
    the variance components positive.
    """
    rng = np.random.default_rng(3)
    n_groups, per = 12, 8
    groups = np.repeat(np.arange(n_groups), per)
    b_group = rng.normal(0.0, 15.0, n_groups)[groups]
    x = rng.integers(0, 2, groups.size).astype(float)
    y = 100.0 + 40.0 * x + b_group + rng.normal(0.0, 8.0, groups.size)
    x_fixed = np.column_stack([np.ones(groups.size), x])
    out = ab.random_intercept_lmm(y, x_fixed, groups)
    assert out["n_groups"] == n_groups
    assert out["n_obs"] == groups.size
    assert out["beta"][0] == pytest.approx(100.0, abs=8.0)
    assert out["beta"][1] == pytest.approx(40.0, abs=10.0)
    assert out["sigma_b"] > 0.0 and out["sigma_e"] > 0.0


def test_facilitation_contrast_prefix_level_is_headline() -> None:
    """D4: the prefix-level test is the headline; trial-level is flagged."""
    rng = np.random.default_rng(5)
    # 8 wind prefixes (median ~150 ms) vs 10 multisensory prefixes (~100 ms).
    n_w, n_m = 8, 10
    prefix = np.array([f"w{i}" for i in range(n_w)] + [f"m{i}" for i in range(n_m)],
                      dtype=object)
    cond = np.array(["wind_only"] * n_w + ["multisensory"] * n_m, dtype=object)
    lat = np.concatenate([rng.normal(150.0, 8.0, n_w),
                          rng.normal(100.0, 8.0, n_m)])
    resp = np.where(np.isfinite(lat), 5, -1)
    m_wind = cond == "wind_only"
    m_multi = cond == "multisensory"
    out = ab._facilitation_contrast(lat, resp, prefix, m_wind, m_multi, rng)
    assert out["n_prefixes"] == {"wind_only": n_w, "multisensory": n_m}
    assert out["prefix_level_test"]["welch_p"] < 0.01
    assert out["prefix_level_test"]["permutation_p_two_sided"] < 0.05
    assert "NOT valid under nesting" in out["trial_level"]["note"]





# ══════════════════════════════════════════════════════════════════════
# 5. Synthetic corpus + main() smoke run
# ══════════════════════════════════════════════════════════════════════
def _synth_trial(
    condition: str, n_frames: int = 400, wind_on: int = 200,
    coll: int = 340,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one (X[T,8], Y[T]) synthetic trial (stationary at the trigger)."""
    x = np.zeros((n_frames, 8), dtype=np.float32)
    y = np.zeros(n_frames, dtype=np.float32)
    if condition != "wind_only":
        x[:coll, 0] = np.linspace(0.0, 170.0, coll)
        x[coll:, 0] = 170.0
    if condition != "visual_only":
        x[wind_on:, 1] = 1.0
    trig = wind_on if condition != "visual_only" else coll
    start = trig + 20
    y[start:start + 40] = 30.0
    return x, y


def _write_synth_dataset(path: Path) -> None:
    """Write a tiny 3-condition, multi-prefix corpus decodable by the loader."""
    specs = [
        ("visual_only", "a_20260101_000000_session_1", 2),
        ("visual_only", "b_20260101_000000_session_1", 2),
        ("wind_only", "c_20260101_000000_session_1", 3),
        ("wind_only", "d_20260101_000000_session_1", 3),
        ("multisensory", "e_20260101_000000_session_1", 4),
        ("multisensory", "f_20260101_000000_session_1", 4),
        ("multisensory", "g_20260101_000000_session_1", 5),
    ]
    x_seqs, y_seqs, lengths, conds, ttcs, labels, sess, anchors = (
        [], [], [], [], [], [], [], [],
    )
    ttc_by_cond = {
        "visual_only": float("nan"),
        "wind_only": float("nan"),
        "multisensory": -308.0,
    }
    for condition, session, n in specs:
        for _ in range(n):
            x, y = _synth_trial(condition)
            x_seqs.append(torch.from_numpy(x))
            y_seqs.append(torch.from_numpy(y))
            lengths.append(x.shape[0])
            conds.append(condition)
            ttcs.append(ttc_by_cond[condition])
            labels.append(0)
            sess.append(session)
            anchors.append(200 if condition != "visual_only" else 340)
    payload = {
        "X_seqs": x_seqs,
        "Y_seqs": y_seqs,
        "lengths": np.asarray(lengths, dtype=int),
        "stimulus_conditions": np.asarray(conds, dtype=object),
        "target_ttc_ms": np.asarray(ttcs, dtype=float),
        "labels": np.asarray(labels, dtype=int),
        "session_ids": np.asarray(sess, dtype=object),
        "anchor_frames": np.asarray(anchors, dtype=int),
        "model_dt_ms": 4.0,
        "pipeline_semantics_version": "test",
        "feature_config": "test",
        "labeling_funnel": {"n_trials": len(specs)},
    }
    torch.save(payload, path)


def test_main_smoke(tmp_path: Path) -> None:
    """``main()`` runs end to end and writes a summary + figures."""
    ds = tmp_path / "synth.pt"
    _write_synth_dataset(ds)
    out = tmp_path / "out"
    rc = ab.main(
        [
            "--dataset", str(ds),
            "--dt_ms", "4.0",
            "--output_dir", str(out),
            "--seed", "42",
            "--n_boot", "20",
            "--n_perm", "50",
        ]
    )
    assert rc == 0
    summary = json.loads((out / "behavior_summary.json").read_text())
    assert summary["meta"]["exploratory"] is True
    assert summary["meta"]["in_sample"] is True
    assert summary["meta"]["between_group_design"] is True
    assert summary["meta"]["n_trials"] == 23
    assert summary["meta"]["prefix_condition_pure"] is True
    assert "race_model" in summary
    assert "facilitation" in summary
    # D1 eligibility ledger and the D2 permutation test are present.
    assert "eligibility" in summary
    assert "permutation_test" in summary["race_model"]
    assert "pooled" in summary["race_model"]["permutation_test"]
    assert "prefix_level_test" in summary["facilitation"]
    # X1: the s1 labeling responder is eligibility-gated; the ungated count is
    # a separate, clearly named diagnostic.
    by_cond = summary["eligibility"]["by_condition"]
    for c in ("visual_only", "wind_only", "multisensory"):
        assert "labeling_responders_ungated" in by_cond[c]
        assert by_cond[c]["labeling_responders_ungated"] >= by_cond[c]["labeling_responders"]
    # X2: the declared race + facilitation families carry Holm-adjusted p.
    assert summary["race_model"]["permutation_test"]["holm_family"]["m"] >= 1
    assert "p_one_sided_holm" in summary["race_model"]["permutation_test"]["pooled"]
    assert summary["facilitation"]["holm_family"]["m"] >= 1
    assert "permutation_p_two_sided_holm" in summary["facilitation"]["prefix_level_test"]
    # D3: the stored-label reconciliation table is separately named.
    assert "stored_label_agreement" in summary
    assert "responder_reconciliation" not in summary
    assert "eligibility" in summary["stored_label_agreement"]["note"].lower()
    # D4: facilitation reports per-variant + pooled with n_prefixes.
    assert "per_variant" in summary["facilitation"]
    assert "n_x" in summary["facilitation"]["prefix_level_test"]
    assert (out / "behavior_summary.json").exists()


def test_load_corpus_rejects_bad_condition_stamp(tmp_path: Path) -> None:
    """A condition stamp contradicting the channels fails closed."""
    ds = tmp_path / "bad.pt"
    _write_synth_dataset(ds)
    payload = torch.load(ds, weights_only=False)
    payload["stimulus_conditions"] = np.asarray(
        ["multisensory"] * len(payload["stimulus_conditions"]), dtype=object
    )
    torch.save(payload, ds)
    with pytest.raises(ValueError):
        ab.load_corpus(ds)
