"""Focused tests for the phase-3 prerequisite baseline fixes (F1-F5, ADR 0011).

Covers, with synthetic fixtures:

* F1 — signed SOA derivation (positive / zero / negative), variant keying on
  ``(target_ttc_ms, SOA bucket)`` for paradigms sharing a TTC, the
  SOA-homogeneity audit (pass within tolerance / fail closed beyond it /
  declared tolerance), and prose generated from the variants actually used.
* F2 — the first-stimulus reference clock ``min(wind onset, collision)`` for
  the D1 trigger/gate/search, the Miller bound's A-term shift for
  wind-lagging variants (bit-identical to the legacy formula for wind-first),
  the pre-reference responder named class, and byte-identity of the
  wind-first path.
* F3 — fail-fast on an undeclared multisensory ``target_ttc_ms`` listing the
  affected sessions.
* F4 — the visual-only label re-clock at the collision in the staging
  adapter, the retained (legacy-readable) version machinery, and the
  read-only impact script.
* F5 — unknown trial_start types and non-binary wind states fail closed; the
  unified wind-presence threshold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nsmor.pipeline import conditions as C  # noqa: E402
from scripts import analyze_behavior as ab  # noqa: E402
from scripts import impact_bv_relabel as impact  # noqa: E402
from scripts.pre_load_adapt import (  # noqa: E402
    _binary_wind_state,
    adapt_cercus_to_nsmor,
    looming_ttc_ms,
)

DT = 4.0


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════
def _trial_x(
    n_frames: int,
    wind_on: Optional[int],
    coll: Optional[int],
    init_deg: float = 2.0,
) -> np.ndarray:
    """(T, 8) feature array: visual ramp to 179 deg at *coll*, wind step."""
    x = np.zeros((n_frames, 8), dtype=np.float32)
    if coll is not None:
        x[:coll, 0] = np.linspace(init_deg, 178.0, coll)
        x[coll:, 0] = 179.0
    if wind_on is not None:
        x[wind_on:, 1] = 1.0
    return x


def _write_dataset(
    path: Path,
    specs: List[Dict[str, Any]],
    dt_ms: float = DT,
    additive_keys: bool = True,
) -> None:
    """Write a tiny synthetic corpus decodable by ``load_corpus``.

    Each spec: condition, session, ttc, n_frames, wind_on, coll, resp
    (start frame or None), resp_len, label.
    """
    x_seqs, y_seqs, lengths, conds, ttcs, labels, sess, anchors = (
        [], [], [], [], [], [], [], [],
    )
    soas, variants, refs = [], [], []
    for spec in specs:
        x = _trial_x(spec["n_frames"], spec.get("wind_on"), spec.get("coll"))
        y = np.zeros(spec["n_frames"], dtype=np.float32)
        if spec.get("resp") is not None:
            y[spec["resp"]:spec["resp"] + spec.get("resp_len", 40)] = 30.0
        x_seqs.append(torch.from_numpy(x))
        y_seqs.append(torch.from_numpy(y))
        lengths.append(spec["n_frames"])
        conds.append(spec["condition"])
        ttcs.append(spec["ttc"])
        labels.append(spec["label"])
        sess.append(spec["session"])
        anchors.append(C.first_stimulus_frame(x, spec["n_frames"]))
        soa = C.stimulus_soa_ms(x, spec["n_frames"], dt_ms)
        soas.append(soa)
        variants.append(
            C.stimulus_variant_key(spec["condition"], spec["ttc"], soa)
        )
        refs.append(C.stimulus_reference_event(spec["condition"], soa))
    payload: Dict[str, Any] = {
        "X_seqs": x_seqs,
        "Y_seqs": y_seqs,
        "lengths": np.asarray(lengths, dtype=int),
        "stimulus_conditions": np.asarray(conds, dtype=object),
        "target_ttc_ms": np.asarray(ttcs, dtype=float),
        "labels": np.asarray(labels, dtype=int),
        "session_ids": np.asarray(sess, dtype=object),
        "anchor_frames": np.asarray(anchors, dtype=int),
        "model_dt_ms": dt_ms,
        "pipeline_semantics_version": "test",
        "feature_config": "test",
        "labeling_funnel": {"n_trials": len(specs)},
    }
    if additive_keys:
        payload["stimulus_soa_ms"] = np.asarray(soas, dtype=float)
        payload["stimulus_variant"] = np.asarray(variants, dtype=object)
        payload["stimulus_reference_event"] = np.asarray(refs, dtype=object)
    torch.save(payload, path)


def _corpus(specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble a corpus dict for detect_events/detect_responses."""
    x = [_trial_x(s["n_frames"], s.get("wind_on"), s.get("coll")) for s in specs]
    y = []
    for s in specs:
        yy = np.zeros(s["n_frames"], dtype=np.float32)
        if s.get("resp") is not None:
            yy[s["resp"]:s["resp"] + s.get("resp_len", 40)] = 30.0
        y.append(yy)
    return {
        "x": x,
        "y": y,
        "conditions": np.asarray([s["condition"] for s in specs], dtype=object),
        "ttc": np.asarray([s["ttc"] for s in specs], dtype=float),
        "labels": np.asarray([s["label"] for s in specs], dtype=int),
        "session_ids": np.asarray([s["session"] for s in specs], dtype=object),
        "lengths": np.asarray([s["n_frames"] for s in specs], dtype=int),
        "n_trials": len(specs),
        "stimulus_soa_ms": None,
    }


def _raw_pair(
    directory: Path,
    session_id: str,
    trial_type: str = "looming_wind",
    lv_ratio_ms: float = 20.0,
    n_samples: int = 400,
    wind_on_sample: Optional[int] = 100,
    stim_state: float = 0,
) -> Tuple[Path, Path]:
    """Raw-schema fixture pair (sys_time / ard_time / stim_state)."""
    kin_path = directory / f"{session_id}_kinematics.csv"
    evt_path = directory / f"{session_id}_events.csv"
    sys_time = np.arange(n_samples, dtype=np.float64) * 0.004
    df_k = pd.DataFrame({
        "sys_time": sys_time,
        "ard_time": (sys_time * 1000.0).astype(np.int64),
        "dx": np.full(n_samples, 0.1),
        "dy": np.full(n_samples, 0.05),
        "dz": np.zeros(n_samples),
        "stim_state": (
            np.where(np.arange(n_samples) >= (wind_on_sample or 0),
                     stim_state, 0.0)
            if wind_on_sample is not None
            else np.zeros(n_samples)
        ),
        "global_trial_id": np.full(n_samples, 1, dtype=np.int64),
    })
    df_k.to_csv(kin_path, index=False)
    details = json.dumps({
        "type": trial_type,
        "target_ttc_ms": -225 if trial_type == "looming_wind" else None,
        "lv_ratio_ms": lv_ratio_ms,
        "wind_dir": "none",
        "screen_side": "left",
    })
    df_e = pd.DataFrame([
        {"event_name": "trial_start", "timestamp": 0.0, "session_num": 1,
         "trial_in_session": 1, "global_trial_id": 1, "details": details},
        {"event_name": "phase_transition", "timestamp": 0.0, "session_num": 1,
         "trial_in_session": 1, "global_trial_id": 1,
         "details": json.dumps({"from_phase": "TrialStart", "to_phase": "Looming"})},
    ])
    df_e.to_csv(evt_path, index=False)
    return kin_path, evt_path


# ══════════════════════════════════════════════════════════════════════
# F1: SOA sign, variant identity, homogeneity audit, generated prose
# ══════════════════════════════════════════════════════════════════════
def test_stimulus_soa_sign_convention() -> None:
    """SOA = collision - wind onset (ms): positive = wind first."""
    wind_first = _trial_x(100, wind_on=10, coll=60)
    assert C.stimulus_soa_ms(wind_first, 100, DT) == pytest.approx((60 - 10) * DT)
    lagging = _trial_x(100, wind_on=60, coll=10)
    assert C.stimulus_soa_ms(lagging, 100, DT) == pytest.approx((10 - 60) * DT)
    simultaneous = _trial_x(100, wind_on=40, coll=40)
    assert C.stimulus_soa_ms(simultaneous, 100, DT) == pytest.approx(0.0)
    wind_only = _trial_x(100, wind_on=10, coll=None)
    assert np.isnan(C.stimulus_soa_ms(wind_only, 100, DT))


def test_first_stimulus_frame_is_min() -> None:
    """The reference event is min(wind onset, collision) for multisensory."""
    wind_first = _trial_x(100, wind_on=10, coll=60)
    assert C.first_stimulus_frame(wind_first, 100) == 10
    lagging = _trial_x(100, wind_on=60, coll=10)
    assert C.first_stimulus_frame(lagging, 100) == 10
    assert C.first_stimulus_frame(_trial_x(100, None, 60), 100) == 60
    assert C.first_stimulus_frame(_trial_x(100, 30, None), 100) == 30
    assert C.first_stimulus_frame(np.zeros((100, 8), np.float32), 100) == -1


def test_derive_anchor_wind_first_matches_legacy_rule() -> None:
    """Wind-first corpora keep the historical first-wind-frame anchor."""
    x = _trial_x(100, wind_on=10, coll=60)
    assert C.derive_anchor_frames([x]) == [10]
    # Wind-lagging: the anchor moves to the collision (intended change).
    xl = _trial_x(100, wind_on=60, coll=10)
    assert C.derive_anchor_frames([xl]) == [10]


def test_variant_ids_split_same_ttc_paradigms() -> None:
    """Two paradigms sharing a TTC but not the SOA are distinct variants."""
    cond = np.asarray(["multisensory", "multisensory", "wind_only"], dtype=object)
    ttc = np.asarray([-225.0, -225.0, np.nan])
    soa = np.asarray([208.0, -200.0, np.nan])
    ids = ab.variant_ids(cond, ttc, soa)
    assert ids[0] != ids[1]
    # The key carries the (ttc, SOA-bucket) identity: 208 -> bucket 200.
    assert str(ids[0]) == "multisensory_ttc-225_soa200"
    assert str(ids[1]) == "multisensory_ttc-225_soa-200"
    assert str(ids[2]) == "wind_only"
    variants = ab.multisensory_variants(cond, ttc, soa, DT)
    assert {v["key"] for v in variants} == {str(ids[0]), str(ids[1])}
    assert all(v["mask"].sum() == 1 for v in variants)


def test_race_model_keys_variants_on_ttc_and_soa() -> None:
    """run_race_model emits one entry per (ttc, SOA bucket) cell."""
    n = 40
    cond = np.asarray(
        ["multisensory"] * 16 + ["wind_only"] * 12 + ["visual_only"] * 12,
        dtype=object,
    )
    ttc = np.concatenate([
        np.full(8, -225.0), np.full(8, -225.0),
        np.full(12, np.nan), np.full(12, np.nan),
    ])
    soa = np.concatenate([
        np.full(8, 208.0), np.full(8, -200.0),
        np.full(12, np.nan), np.full(12, np.nan),
    ])
    latency = np.full(n, np.nan)
    latency[:24] = 120.0  # eligible responders
    resp_frame = np.where(np.isfinite(latency), 10, -1)
    eligible = np.ones(n, dtype=bool)
    prefix_keys = np.asarray(
        [f"m{i}" for i in range(16)] + [f"w{i}" for i in range(12)]
        + [f"v{i}" for i in range(12)],
        dtype=object,
    )
    out = ab.run_race_model(
        cond, ttc, soa, latency, prefix_keys, eligible, DT, n_boot=4,
        rng=np.random.default_rng(0), n_perm=5,
    )
    assert len(out["per_variant"]) == 2
    keys = set(out["per_variant"])
    # Keys carry the SOA BUCKET (208 -> 200), not the raw median.
    assert keys == {"multisensory_ttc-225_soa200",
                    "multisensory_ttc-225_soa-200"}
    assert all(v["n_av"] == 8 for v in out["per_variant"].values())
    # The declared Holm family names exactly the variants used.
    assert out["permutation_test"]["holm_family"]["m"] == 3
    assert all(k in out["permutation_test"]["holm_family"]["family"]
               for k in keys)
    assert "4 TTC" not in json.dumps(out)
    _ = resp_frame


def test_soa_homogeneity_audit_fails_closed() -> None:
    """Spread > declared tolerance raises; a declared tolerance passes."""
    cond = np.asarray(["multisensory"] * 3, dtype=object)
    ttc = np.asarray([-225.0] * 3)
    soa = np.asarray([200.0, 204.0, 216.0])  # spread 16 ms = 4 frames
    variants = ab.multisensory_variants(cond, ttc, soa, DT)
    with pytest.raises(ab.SOAHomogeneityError):
        ab.audit_soa_homogeneity(variants, DT, 1.0)
    report = ab.audit_soa_homogeneity(variants, DT, 4.0)
    assert report["passed"] is True
    assert report["declared_tolerance_frames"] == 4.0
    # Two paradigms whose SOAs fall in the SAME 50 ms bucket (200 and 220)
    # are exactly the silent merge the audit catches.
    soa_merged = np.asarray([200.0, 204.0, 220.0])
    with pytest.raises(ab.SOAHomogeneityError):
        ab.audit_soa_homogeneity(
            ab.multisensory_variants(cond, ttc, soa_merged, DT), DT, 1.0,
        )
    # A paradigm 200 ms away lands in its own bucket: no violation.
    soa_split = np.asarray([200.0, 204.0, 0.0])
    assert ab.audit_soa_homogeneity(
        ab.multisensory_variants(cond, ttc, soa_split, DT), DT, 1.0,
    )["passed"] is True


def test_miller_bound_negative_soa_shifts_a_term() -> None:
    """Wind-lagging: bound(t) = F_A(t - |soa|) + F_V(t) on the ref clock."""
    tgrid = np.arange(0.0, 2000.0 + DT, DT)
    lat_a = np.asarray([100.0, 200.0, 300.0])
    lat_v = np.asarray([150.0, 250.0, 350.0])
    n_a = n_v = 4
    soa = -200.0
    bound = ab.miller_bound(tgrid, lat_a, n_a, lat_v, n_v, [soa], [1.0])
    fa = ab.ecdf(lat_a, tgrid - 200.0, n_a)
    fv = ab.ecdf(lat_v, tgrid, n_v)
    assert np.array_equal(bound, np.clip(fa + fv, 0.0, 1.0))
    # For a wind-first variant the V term keeps the historical shift.
    bound_pos = ab.miller_bound(tgrid, lat_a, n_a, lat_v, n_v, [200.0], [1.0])
    assert np.array_equal(
        bound_pos, np.clip(ab.ecdf(lat_a, tgrid, n_a)
                           + ab.ecdf(lat_v, tgrid - 200.0, n_v), 0.0, 1.0),
    )
    # A zero SOA reduces to the same-clock sum.
    bound_zero = ab.miller_bound(tgrid, lat_a, n_a, lat_v, n_v, [0.0], [1.0])
    assert np.array_equal(
        bound_zero, np.clip(ab.ecdf(lat_a, tgrid, n_a)
                            + ab.ecdf(lat_v, tgrid, n_v), 0.0, 1.0),
    )


def test_miller_bound_wind_first_bitwise_legacy_mixture() -> None:
    """Wind-first mixtures match the historical expression bit for bit."""
    tgrid = np.arange(0.0, 1000.0 + DT, DT)
    rng = np.random.default_rng(3)
    lat_a = np.sort(rng.uniform(50, 400, 200))
    lat_v = np.sort(rng.uniform(50, 400, 150))
    soas = [356.0, 288.0]
    weights = [0.6, 0.4]
    new = ab.miller_bound(tgrid, lat_a, 220, lat_v, 170, soas, weights)
    # Reference: the historical implementation, re-expressed literally.
    bound = ab.ecdf(lat_a, tgrid, 220).copy()
    w = np.asarray(weights) / np.sum(weights)
    for s, wk in zip(soas, w):
        bound = bound + wk * ab.ecdf(lat_v, tgrid - s, 170)
    assert np.array_equal(new, np.clip(bound, 0.0, 1.0))


def test_reliability_prose_generated_from_variants() -> None:
    """The reliability status text names the variants actually used."""
    variants = ab.multisensory_variants(
        np.asarray(["multisensory"] * 2, dtype=object),
        np.asarray([-225.0, 0.0]),
        np.asarray([208.0, 0.0]),
        DT,
    )
    status = ab.run_reliability_status(variants)
    assert "2 multisensory variant(s)" in status["present"]
    assert "208" in status["present"] and "0" in status["present"]
    assert "TTC=0" not in status["present"]  # a 0-TTC variant exists -> no claim
    assert "4 TTC" not in json.dumps(status)


# ══════════════════════════════════════════════════════════════════════
# F2: first-stimulus reference clock, pre-reference responders
# ══════════════════════════════════════════════════════════════════════
def test_wind_lagging_d1_uses_collision_clock() -> None:
    """A collision-triggered escape before the wind is a valid D1 response."""
    spec = {"condition": "multisensory", "session": "a_s_1", "ttc": -225.0,
            "n_frames": 200, "wind_on": 150, "coll": 100, "resp": 110,
            "resp_len": 40, "label": 0}
    corpus = _corpus([spec])
    events = ab.detect_events(corpus, DT)
    assert events["ref"][0] == 100  # collision, not the wind onset (150)
    responses = ab.detect_responses(corpus, events, DT)
    assert responses["eligible"][0]
    assert responses["resp"][0] == 110
    assert responses["latency_ms"][0] == pytest.approx((110 - 100) * DT)
    # A mover between the collision and the wind does not gate the trial
    # (the gate covers [collision - 250 ms, collision] only).
    corpus2 = _corpus([{**spec, "resp": None}])
    corpus2["y"][0][120] = 25.0  # moving after the collision
    responses2 = ab.detect_responses(corpus2, events, DT)
    assert responses2["eligible"][0]
    assert responses2["resp"][0] == 120


def test_pre_reference_responders_named_class() -> None:
    """Responses before the wind onset are reported as a named class."""
    lagging = {"condition": "multisensory", "session": "a_s_1", "ttc": -225.0,
               "n_frames": 200, "wind_on": 150, "coll": 100, "resp": 110,
               "resp_len": 40, "label": 0}
    wind_first = {"condition": "multisensory", "session": "b_s_1", "ttc": -225.0,
                  "n_frames": 300, "wind_on": 100, "coll": 240, "resp": 120,
                  "resp_len": 40, "label": 0}
    corpus = _corpus([lagging, wind_first])
    events = ab.detect_events(corpus, DT)
    responses = ab.detect_responses(corpus, events, DT)
    ids = ab.variant_ids(corpus["conditions"], corpus["ttc"],
                         np.asarray([(100 - 150) * DT, (240 - 100) * DT]))
    out = ab.pre_reference_responders(
        corpus["conditions"], events["w_on"], responses, corpus["labels"], ids,
    )
    assert out["counts_by_condition"]["primary"]["multisensory"] == 1
    assert out["counts_by_condition"]["primary"]["wind_only"] == 0
    assert "STRICTLY BEFORE" in out["definition"]
    assert out["counts_by_variant_primary"][str(ids[0])] == 1


def test_wind_first_fixture_stays_byte_identical() -> None:
    """Wind-first trials keep the legacy ref, latency and stored keys."""
    spec = {"condition": "multisensory", "session": "a_s_1", "ttc": -308.0,
            "n_frames": 300, "wind_on": 100, "coll": 240, "resp": 120,
            "resp_len": 40, "label": 0}
    corpus = _corpus([spec])
    events = ab.detect_events(corpus, DT)
    # Legacy rule: the trigger is the first wind frame.
    assert events["ref"][0] == 100
    assert events["ref"][0] == C.derive_anchor_frames(
        [corpus["x"][0]], [spec["n_frames"]]
    )[0]
    responses = ab.detect_responses(corpus, events, DT)
    assert responses["latency_ms"][0] == pytest.approx((120 - 100) * DT)
    soa = ab.per_trial_soa_ms(corpus, events, DT)
    assert soa[0] == pytest.approx((240 - 100) * DT)


def test_looming_trigger_wind_lagging_uses_collision_clock() -> None:
    """Wind-lagging: ``escape_minus_collision_ms`` is the collision clock.

    Regression for the round-1 blocker: for SOA <= 0 the reference clock
    IS the collision, so ``latency`` must pass through UNshifted.  The old
    ``latency - (v_col - w_on) * dt`` subtracted a negative offset and
    inflated the reading by exactly |SOA| (40 ms reported as 240 ms here).
    """
    spec = {"condition": "multisensory", "session": "a_s_1", "ttc": -225.0,
            "n_frames": 200, "wind_on": 150, "coll": 100, "resp": 110,
            "resp_len": 40, "label": 0}
    corpus = _corpus([spec])
    events = ab.detect_events(corpus, DT)
    responses = ab.detect_responses(corpus, events, DT)
    soa = ab.per_trial_soa_ms(corpus, events, DT)
    assert soa[0] == pytest.approx(-200.0)
    out = ab.run_looming_trigger(
        corpus["conditions"], corpus["ttc"], soa, responses["latency_ms"],
        events["w_on"], events["v_col"], DT,
    )
    assert len(out["per_variant"]) == 1
    pv = out["per_variant"][0]
    assert pv["variant"] == "multisensory_ttc-225_soa-200"
    assert pv["n_responders"] == 1
    stats = pv["escape_minus_collision_ms"]
    true_ms = (110 - 100) * DT  # 40 ms, measured from the collision
    assert stats["median"] == pytest.approx(true_ms)
    assert stats["mean"] == pytest.approx(true_ms)
    # The blocker's signature: the reading inflated by |SOA| = 200 ms.
    assert stats["median"] != pytest.approx(true_ms + 200.0)


def test_looming_trigger_simultaneous_soa_zero_no_shift() -> None:
    """SOA = 0: the collision is the reference; no shift is applied."""
    spec = {"condition": "multisensory", "session": "a_s_1", "ttc": -225.0,
            "n_frames": 200, "wind_on": 100, "coll": 100, "resp": 110,
            "resp_len": 40, "label": 0}
    corpus = _corpus([spec])
    events = ab.detect_events(corpus, DT)
    responses = ab.detect_responses(corpus, events, DT)
    soa = ab.per_trial_soa_ms(corpus, events, DT)
    assert soa[0] == pytest.approx(0.0)
    assert events["ref"][0] == 100
    out = ab.run_looming_trigger(
        corpus["conditions"], corpus["ttc"], soa, responses["latency_ms"],
        events["w_on"], events["v_col"], DT,
    )
    stats = out["per_variant"][0]["escape_minus_collision_ms"]
    assert stats["median"] == pytest.approx((110 - 100) * DT)


def test_looming_trigger_mixed_paradigms_same_ttc() -> None:
    """Same-TTC paradigms convert per (ttc, SOA) cell, not per ttc."""
    specs = [
        # Wind-first (SOA +200 ms): reference = wind onset (100).
        {"condition": "multisensory", "session": "a_s_1", "ttc": -225.0,
         "n_frames": 300, "wind_on": 100, "coll": 150, "resp": 110,
         "resp_len": 40, "label": 0},
        # Wind-lagging (SOA -200 ms): reference = collision (100).
        {"condition": "multisensory", "session": "b_s_1", "ttc": -225.0,
         "n_frames": 300, "wind_on": 150, "coll": 100, "resp": 110,
         "resp_len": 40, "label": 0},
    ]
    corpus = _corpus(specs)
    events = ab.detect_events(corpus, DT)
    responses = ab.detect_responses(corpus, events, DT)
    soa = ab.per_trial_soa_ms(corpus, events, DT)
    assert soa[0] == pytest.approx(200.0) and soa[1] == pytest.approx(-200.0)
    out = ab.run_looming_trigger(
        corpus["conditions"], corpus["ttc"], soa, responses["latency_ms"],
        events["w_on"], events["v_col"], DT,
    )
    by_key = {pv["variant"]: pv for pv in out["per_variant"]}
    assert set(by_key) == {"multisensory_ttc-225_soa200",
                           "multisensory_ttc-225_soa-200"}
    first = by_key["multisensory_ttc-225_soa200"]["escape_minus_collision_ms"]
    lagging = by_key["multisensory_ttc-225_soa-200"][
        "escape_minus_collision_ms"]
    # Wind-first: latency (from wind onset 100) minus the 200 ms shift.
    assert first["median"] == pytest.approx((110 - 150) * DT)
    # Wind-lagging: already on the collision clock, no shift.
    assert lagging["median"] == pytest.approx((110 - 100) * DT)


def test_looming_trigger_wind_first_bitwise_legacy_shift() -> None:
    """Wind-first: the collision conversion is the historical shift, bitwise."""
    specs = [
        {"condition": "multisensory", "session": "a_s_1", "ttc": -308.0,
         "n_frames": 300, "wind_on": 100, "coll": 240, "resp": 120,
         "resp_len": 40, "label": 0},
        {"condition": "multisensory", "session": "a_s_1", "ttc": -308.0,
         "n_frames": 300, "wind_on": 100, "coll": 240, "resp": 140,
         "resp_len": 40, "label": 0},
    ]
    corpus = _corpus(specs)
    events = ab.detect_events(corpus, DT)
    responses = ab.detect_responses(corpus, events, DT)
    soa = ab.per_trial_soa_ms(corpus, events, DT)
    out = ab.run_looming_trigger(
        corpus["conditions"], corpus["ttc"], soa, responses["latency_ms"],
        events["w_on"], events["v_col"], DT,
    )
    # The historical expression, re-expressed literally on the same inputs.
    expected = ab.summary_stats(np.asarray([
        responses["latency_ms"][i]
        - (events["v_col"][i] - events["w_on"][i]) * DT
        for i in range(len(specs))
    ]))
    got = out["per_variant"][0]["escape_minus_collision_ms"]
    assert got == expected
    assert got["median"] == pytest.approx(((120 + 140) / 2 - 240) * DT)


def test_variant_soa_frozen_accessor_restored() -> None:
    """``_variant_soa`` keeps its historical signature and values (spec §8)."""
    m_multi = np.asarray([True, True, False])
    ttc = np.asarray([-308.0, -225.0, np.nan])
    w_on = np.asarray([100, 150, 10])
    v_col = np.asarray([240, 100, -1])
    # Wind-first group: (240 - 100) * 4 = 560 ms, the frozen value.
    assert ab._variant_soa(m_multi, ttc, w_on, v_col, DT, -308.0) == 560.0
    # Wind-lagging group: signed negative median, (100 - 150) * 4 = -200 ms.
    assert ab._variant_soa(m_multi, ttc, w_on, v_col, DT, -225.0) == -200.0
    # Empty group keeps the historical 0.0 fallback.
    assert ab._variant_soa(m_multi, ttc, w_on, v_col, DT, -999.0) == 0.0
    # Bitwise against the literal historical expression on a two-trial group.
    m2 = np.asarray([True, True])
    ttc2 = np.asarray([-225.0, -225.0])
    w2 = np.asarray([100, 100])
    c2 = np.asarray([240, 200])
    assert ab._variant_soa(m2, ttc2, w2, c2, DT, -225.0) == float(
        np.median(np.asarray([(240 - 100) * DT, (200 - 100) * DT]))
    )


def test_etl_persists_additive_paradigm_keys(tmp_path: Path) -> None:
    """prepare_dataset persists the SOA / variant / reference-event keys."""
    from nsmor.pipeline.nested_prior import load_artifact_bytes
    from scripts.prepare_data import prepare_dataset
    from tests.test_pipeline import _make_label_audit_csvs

    raw_dir = tmp_path / "raw"
    _make_label_audit_csvs(raw_dir)
    out = tmp_path / "ds.pt"
    prepare_dataset(raw_dir=raw_dir, output_path=out, random_seed=42)
    ds = load_artifact_bytes(out.read_bytes())
    n = len(ds["X_seqs"])
    assert np.asarray(ds["stimulus_soa_ms"]).shape == (n,)
    assert np.asarray(ds["stimulus_variant"]).shape == (n,)
    assert np.asarray(ds["stimulus_reference_event"]).shape == (n,)
    cond = np.asarray(ds["stimulus_conditions"], dtype=object)
    soa = np.asarray(ds["stimulus_soa_ms"], dtype=float)
    ttc = np.asarray(ds["target_ttc_ms"], dtype=float)
    variants = np.asarray(ds["stimulus_variant"], dtype=object)
    assert np.all(np.isfinite(soa[cond == "multisensory"]))
    for i in range(n):
        assert variants[i] == C.stimulus_variant_key(cond[i], ttc[i], soa[i])
    # The analysis keying agrees with the persisted key on multisensory rows.
    ids = ab.variant_ids(cond, ttc, soa)
    assert set(ids[cond == "multisensory"].tolist()) == set(
        variants[cond == "multisensory"].tolist()
    )


def test_per_trial_soa_uses_stored_key_when_present(tmp_path: Path) -> None:
    """The stored additive key is read and cross-checked, not trusted blindly."""
    ds = tmp_path / "ds.pt"
    _write_dataset(ds, [{
        "condition": "multisensory", "session": "a_s_1", "ttc": -308.0,
        "n_frames": 300, "wind_on": 100, "coll": 240, "resp": 120,
        "resp_len": 40, "label": 0,
    }])
    loaded = ab.load_corpus(ds)
    assert loaded["stimulus_soa_ms"] is not None
    assert loaded["stimulus_soa_ms"][0] == pytest.approx((240 - 100) * DT)
    assert loaded["stimulus_variant"][0] == "multisensory_ttc-308_soa550"
    assert loaded["stimulus_reference_event"][0] == "wind_onset"


# ══════════════════════════════════════════════════════════════════════
# F3: fail-fast on undeclared multisensory TTC
# ══════════════════════════════════════════════════════════════════════
def test_undeclared_multisensory_ttc_fails_fast(tmp_path: Path) -> None:
    """NaN TTC on a multisensory trial raises at entry, naming sessions."""
    ds = tmp_path / "nan_ttc.pt"
    _write_dataset(ds, [
        {"condition": "multisensory", "session": "bad_session_1",
         "ttc": float("nan"), "n_frames": 120, "wind_on": 20, "coll": 60,
         "resp": 70, "resp_len": 40, "label": 0},
        {"condition": "wind_only", "session": "w_s_1", "ttc": float("nan"),
         "n_frames": 120, "wind_on": 20, "coll": None, "resp": 30,
         "resp_len": 40, "label": 0},
    ])
    args = ab.build_arg_parser().parse_args([
        "--dataset", str(ds), "--output_dir", str(tmp_path / "out"),
        "--n_boot", "5", "--n_perm", "5",
    ])
    with pytest.raises(ab.UndeclaredTargetTTCError) as excinfo:
        ab.run(args)
    message = str(excinfo.value)
    assert "bad_session_1" in message
    assert "target_ttc_ms" in message
    # wind_only NaN TTC is legitimate and must not trip the guard.
    ab.fail_fast_on_undeclared_ttc(
        np.asarray(["wind_only"], dtype=object), np.asarray([np.nan]),
        np.asarray(["w_s_1"], dtype=object),
    )


# ══════════════════════════════════════════════════════════════════════
# F4: visual-only re-clock, version retention, impact script
# ══════════════════════════════════════════════════════════════════════
def test_adapter_reclocks_visual_only_at_collision(tmp_path: Path) -> None:
    """A visual-only trial's injected stimulus_onset is the collision."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "staged"
    sid = "0.700cricket_001_20260101_000000_session_1"
    _raw_pair(raw_dir, sid, trial_type="baseline_visual", lv_ratio_ms=20.0,
              n_samples=400, wind_on_sample=None)
    adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
    evt = pd.read_csv(out_dir / sid / f"{sid}_events.csv")
    injected = evt.loc[evt["event_type"] == "stimulus_onset"]
    assert len(injected) == 1
    # Looming onset is the trial start (t=0); the collision is ttc ms later.
    assert injected["time_ms"].iloc[0] == pytest.approx(
        looming_ttc_ms(20.0), rel=1e-9
    )
    assert "visual_collision" in str(injected["event_value"].iloc[0])


def test_adapter_wind_first_injection_is_byte_identical(tmp_path: Path) -> None:
    """Wind-first multisensory trials keep the legacy onset and event value."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "staged"
    sid = "0.700cricket_001_20260101_000000_session_1"
    kin_path, _ = _raw_pair(raw_dir, sid, trial_type="looming_wind",
                            lv_ratio_ms=20.0, n_samples=400,
                            wind_on_sample=100, stim_state=1)
    adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=out_dir)
    kin = pd.read_csv(kin_path)  # raw source unchanged (read-only proof)
    kin_staged = pd.read_csv(out_dir / sid / f"{sid}_kinematics.csv")
    first_wind_ms = float(kin_staged.loc[kin_staged["wind_state"] == 1,
                                         "time_ms"].min())
    evt = pd.read_csv(out_dir / sid / f"{sid}_events.csv")
    injected = evt.loc[evt["event_type"] == "stimulus_onset"]
    assert len(injected) == 1
    assert injected["time_ms"].iloc[0] == pytest.approx(first_wind_ms)
    assert injected["event_value"].iloc[0] == '{"source": "kinematics_injected"}'
    assert kin["stim_state"].max() == 1  # source untouched


def test_pipeline_version_retention_machinery() -> None:
    """2.2 is retained (readable, marked); anything else is refused."""
    from nsmor.config import (
        PIPELINE_SEMANTICS_VERSION,
        pipeline_version_status,
    )
    from nsmor.checkpoint import _require_pipeline_version

    assert PIPELINE_SEMANTICS_VERSION == "2.3"
    assert pipeline_version_status("2.3") == "current"
    assert pipeline_version_status("2.2") == "retained"
    assert pipeline_version_status("2.1") == "rejected"
    assert pipeline_version_status(None) == "missing"
    assert _require_pipeline_version("2.3", "artifact") == "current"
    assert _require_pipeline_version("2.2", "artifact") == "retained"
    with pytest.raises(RuntimeError):
        _require_pipeline_version("2.1", "artifact")
    with pytest.raises(RuntimeError):
        _require_pipeline_version(None, "artifact")


def test_checkpoint_with_retained_version_loads(tmp_path: Path) -> None:
    """A v2.2 checkpoint still loads (marked legacy) under v2.3 code."""
    from nsmor.checkpoint import load_checkpoint, save_checkpoint
    from nsmor.model_nsmor_core import NSMoRCore

    model = NSMoRCore()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, epoch=0, loss=1.0, config={}, path=path)
    payload = torch.load(path, weights_only=False)
    payload["pipeline_semantics_version"] = "2.2"
    torch.save(payload, path)
    loaded = load_checkpoint(path, model, optimizer)
    assert loaded["pipeline_semantics_version"] == "2.2"


def _staged_session(
    root: Path, session_id: str, trial_id: int, coll_frame: int,
    resp_start: int, n_frames: int, lv_ratio_ms: float = 20.0,
) -> None:
    """Write one canonical staged session with a visual-only trial."""
    out = root / session_id
    out.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_frames, dtype=np.float64) * DT
    vis = np.where(
        np.arange(n_frames) < coll_frame,
        np.linspace(2.0, 178.0, coll_frame)[np.clip(np.arange(n_frames), 0, coll_frame - 1)],
        179.0,
    )
    vel = np.zeros(n_frames)
    vel[resp_start:resp_start + 60] = 30.0
    pd.DataFrame({
        "session_id": session_id, "trial_id": trial_id, "time_ms": t,
        "x_pos": np.zeros(n_frames), "y_pos": np.zeros(n_frames),
        "heading": np.zeros(n_frames), "velocity": vel,
        "acceleration": np.zeros(n_frames), "visual_angle": vis,
        "wind_state": np.zeros(n_frames), "l_v_ratio": np.full(n_frames, lv_ratio_ms),
    }).to_csv(out / f"{session_id}_kinematics.csv", index=False)
    # The OLD clock: stimulus_onset at the looming onset (t = 0).
    pd.DataFrame([
        {"session_id": session_id, "trial_id": trial_id, "time_ms": 0.0,
         "event_type": "trial_start",
         "event_value": json.dumps({
             "type": "baseline_visual", "target_ttc_ms": None,
             "lv_ratio_ms": lv_ratio_ms, "wind_dir": "none",
             "screen_side": "left",
         })},
        {"session_id": session_id, "trial_id": trial_id, "time_ms": 0.0,
         "event_type": "stimulus_onset",
         "event_value": '{"source": "kinematics_injected"}'},
    ]).to_csv(out / f"{session_id}_events.csv", index=False)


def test_impact_script_counts_collision_locked_escape(tmp_path: Path) -> None:
    """The read-only impact script detects a NO_RESPONSE -> ESCAPE change."""
    staged = tmp_path / "staged"
    sid = "0.700cricket_001_20260101_000000_session_1"
    n_frames = 400
    coll_frame = int(round(looming_ttc_ms(20.0) / DT))  # ~287
    _staged_session(staged, sid, 1, coll_frame, resp_start=coll_frame + 5,
                    n_frames=n_frames)
    ds = tmp_path / "corpus.pt"
    # Stored corpus: the trial is a NO_RESPONSE under the old (looming) clock.
    x = np.zeros((n_frames, 8), dtype=np.float32)
    x[:coll_frame, 0] = np.linspace(2.0, 178.0, coll_frame)
    x[coll_frame:, 0] = 179.0
    y = np.zeros(n_frames, dtype=np.float32)
    y[coll_frame + 5:coll_frame + 65] = 30.0
    torch.save({
        "X_seqs": [torch.from_numpy(x)],
        "Y_seqs": [torch.from_numpy(y)],
        "lengths": np.asarray([n_frames]),
        "stimulus_conditions": np.asarray(["visual_only"], dtype=object),
        "target_ttc_ms": np.asarray([np.nan]),
        "labels": np.asarray([3]),  # NO_RESPONSE
        "session_ids": np.asarray([sid], dtype=object),
        "trial_ids": np.asarray([1], dtype=np.int64),
        "anchor_frames": np.asarray([coll_frame]),
        "model_dt_ms": DT,
        "pipeline_semantics_version": "2.2",
        "feature_config": "test",
        "labeling_funnel": {},
    }, ds)
    out = tmp_path / "impact.json"
    rc = impact.main([
        "--staged_dir", str(staged), "--dataset", str(ds),
        "--output", str(out),
    ])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["read_only"] is True
    assert report["inputs_unchanged"] is True
    assert report["n_visual_only_trials_in_corpus"] == 1
    assert report["n_old_clock_matches_stored"] == 1
    assert report["transition_matrix"].get("NO_RESPONSE->ESCAPE") == 1
    assert report["n_label_changes"] == 1


# ══════════════════════════════════════════════════════════════════════
# F5: data-contract guards
# ══════════════════════════════════════════════════════════════════════
def test_adapter_unknown_trial_type_fails_closed(tmp_path: Path) -> None:
    """A declared but unrecognized trial type aborts the staging."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sid = "0.700cricket_001_20260101_000000_session_1"
    _raw_pair(raw_dir, sid, trial_type="looming_wind_80", lv_ratio_ms=20.0)
    with pytest.raises(ValueError) as excinfo:
        adapt_cercus_to_nsmor(raw_dir=raw_dir, output_dir=tmp_path / "staged")
    assert "unrecognized trial_start type" in str(excinfo.value)
    assert "looming_wind_80" in str(excinfo.value)


def test_adapter_nonbinary_wind_state_fails_closed(tmp_path: Path) -> None:
    """A coherent non-binary wind coding aborts instead of zeroing."""
    for bad_value in (2, 0.5, float("nan")):
        raw_dir = tmp_path / f"raw_{bad_value}"
        raw_dir.mkdir()
        sid = "0.700cricket_001_20260101_000000_session_1"
        _raw_pair(raw_dir, sid, trial_type="baseline_wind", lv_ratio_ms=None,
                  n_samples=60, wind_on_sample=10, stim_state=bad_value)
        with pytest.raises(ValueError) as excinfo:
            adapt_cercus_to_nsmor(raw_dir=raw_dir,
                                  output_dir=tmp_path / f"staged_{bad_value}")
        assert "wind-state contract" in str(excinfo.value)
    # A REPEATED sentinel (an intensity coding) fails even when isolated.
    with pytest.raises(ValueError) as excinfo:
        _binary_wind_state(
            pd.Series([0.0, 0.8, 0.0, 0.0, 0.8, 0.0]), "s", "stim_state",
        )
    assert "repeated non-binary value" in str(excinfo.value)
    # The helper itself: valid 0/1 passes; the shared threshold decides.
    ok = _binary_wind_state(pd.Series([0, 1, 1, 0]), "s", "stim_state")
    assert ok.tolist() == [0, 1, 1, 0]


def test_adapter_tolerates_isolated_wind_artifact() -> None:
    """A single isolated telemetry glitch is warned about and read as 0."""
    with pytest.warns(UserWarning, match="isolated non-binary"):
        out = _binary_wind_state(
            pd.Series([0.0, 1.0, 1.0, 587.0, 0.0, 0.0]), "s", "stim_state",
        )
    assert out.tolist() == [0, 1, 1, 0, 0, 0]
    # Two isolated glitches with DISTINCT values (the phase-2 pattern) pass;
    # two with the same value do not (caught above).
    with pytest.warns(UserWarning):
        out2 = _binary_wind_state(
            pd.Series([0.0, 407.0, 1.0, 1.0, 0.0, 548.0]), "s", "stim_state",
        )
    assert out2.tolist() == [0, 0, 1, 1, 0, 0]


def test_wind_presence_threshold_unified() -> None:
    """One threshold: a sub-threshold wind is absent for stamp AND anchor."""
    x = _trial_x(100, wind_on=None, coll=60)
    x[10:, 1] = 0.4  # below WIND_PRESENCE_THRESHOLD (0.5)
    conds, _ = C.derive_stimulus_metadata([x], [100])
    assert conds[0] == "visual_only"
    assert C.wind_onset_frame(x, 100) == -1
    assert C.derive_anchor_frames([x]) == [60]  # collision, no wind anchor
    x2 = x.copy()
    x2[10:, 1] = 0.6
    conds2, _ = C.derive_stimulus_metadata([x2], [100])
    assert conds2[0] == "multisensory"
    assert C.wind_onset_frame(x2, 100) == 10
    assert C.derive_anchor_frames([x2]) == [10]
