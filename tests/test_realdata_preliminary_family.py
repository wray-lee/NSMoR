"""Focused tests for the real-data preliminary family scorer.

Exercises the frozen 78-slot declared family and the comparator mapping of
:mod:`scripts.evaluate_realdata_preliminary`, which now DELEGATES the family
algebra to the frozen ``assemble_declared_family`` (with ``candidate_ids``).
No real data, no training, no core modules touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_realdata_preliminary import (  # noqa: E402
    CANDIDATE_IDS,
    DECLARED_FAMILY_SIZE,
    assemble_family,
    declared_family_slots,
)
from nsmor.analysis.model_comparison import (  # noqa: E402
    COMPARATOR_IDS,
    SCOPE_IDS,
)
from nsmor.analysis.optimization_evaluation import (  # noqa: E402
    compute_aligned_grid_metrics,
)


def _targets(seed: int, n_trials: int = 8, length: int = 40):
    """Shared canonical targets: identical across arms, as on real data."""
    rng = np.random.default_rng(seed)
    return [
        rng.normal(0.0, 3.0, size=length).astype(np.float64) for _ in range(n_trials)
    ]


def _grid(y_true, pred_seed: int, prefix_seed: int = 0):
    """Build a grid-metrics result: shared targets, arm-specific predictions.

    Band membership depends only on ``y_true``, so sharing targets guarantees
    the trial/prefix/frame alignment the family assembly requires. Eight
    unique prefixes clears the frozen ``_MIN_PREFIXES`` (=6) so the
    unasserted-exchangeability reason is the one actually exercised.
    """
    rng = np.random.default_rng(pred_seed)
    y_pred = [t + rng.normal(0.0, 0.5, size=t.size) for t in y_true]
    trial_ids = [f"trial{i}" for i in range(len(y_true))]
    prefix_ids = [f"prefix{i + prefix_seed}" for i in range(len(y_true))]
    return compute_aligned_grid_metrics(
        y_true_seqs=y_true, y_pred_seqs=y_pred,
        trial_ids=trial_ids, prefix_ids=prefix_ids,
    )


def test_declared_family_is_frozen_78() -> None:
    slots = declared_family_slots()
    assert len(slots) == DECLARED_FAMILY_SIZE == 78
    assert len(set(slots)) == 78
    assert len(SCOPE_IDS) == 13
    assert CANDIDATE_IDS == ("A1", "A2")
    assert COMPARATOR_IDS == ("head_only_baseline", "persistence", "zero")
    assert "A1|head_only_baseline|aligned_overall" in slots
    assert "A2|zero|band50_run3" in slots


def test_missing_candidate_yields_null_unavailable_slots() -> None:
    baseline = _grid(_targets(0), pred_seed=0)
    family = assemble_family({}, baseline, exchangeability_asserted=False)
    assert family["family_size"] == 78
    assert family["both_arms_present"] is False
    assert family["present_candidates"] == []
    holm = family["holm_bonferroni"]
    assert holm["unavailable_slot_count"] == 78
    assert holm["valid_test_count"] == 0
    for slot in family["slots"].values():
        assert slot["available"] is False
        assert slot["reason"] == "missing_candidate_arm"
        assert slot["sign_flip"]["p_value"] is None


def test_present_candidate_comparator_mapping() -> None:
    targets = _targets(0)
    baseline = _grid(targets, pred_seed=0)
    a1 = _grid(targets, pred_seed=1)
    family = assemble_family({"A1": a1}, baseline, exchangeability_asserted=False)
    assert family["family_size"] == 78
    assert family["present_candidates"] == ["A1"]

    # A1's aligned_overall slot is always available (8 trials, t>=1 frames).
    slot = family["slots"]["A1|persistence|aligned_overall"]
    assert slot["available"] is True
    assert slot["counts"]["trials"] == 8
    assert "cohens_dz" in slot
    assert slot["delta_convention"] == "candidate_minus_comparator_mse"
    # Exchangeability not asserted -> no permutation p-value, with a reason.
    assert slot["sign_flip"]["p_value"] is None
    assert slot["sign_flip"]["reason"] == "exchangeability_not_asserted"

    # A2 was not supplied: every A2 slot is an unavailable null.
    assert all(
        not v["available"]
        for k, v in family["slots"].items()
        if k.startswith("A2|")
    )
    # The aligned-overall A1 scopes are all available (3 comparators).
    assert all(
        family["slots"][f"A1|{c}|aligned_overall"]["available"]
        for c in COMPARATOR_IDS
    )


def test_exchangeability_asserted_requires_bool() -> None:
    targets = _targets(0)
    baseline = _grid(targets, pred_seed=0)
    with pytest.raises(ValueError):
        assemble_family(
            {"A1": _grid(targets, pred_seed=1)}, baseline,
            exchangeability_asserted="false",
        )


def test_misaligned_bands_fail_closed() -> None:
    """A candidate scored on different targets than the comparator must refuse.

    Band membership is a function of the targets, so scoring the two arms on
    different trials changes the per-band trial sets. The assembly must raise
    rather than compare mismatched frames.
    """
    baseline = _grid(_targets(0), pred_seed=0)
    other = _grid(_targets(1), pred_seed=1)  # different targets -> different bands
    with pytest.raises(ValueError):
        assemble_family({"A1": other}, baseline, exchangeability_asserted=False)


def test_delegation_uses_frozen_assembler_slot_shape() -> None:
    """The family must carry the frozen assembler's own key shape (not a copy)."""
    targets = _targets(0)
    baseline = _grid(targets, pred_seed=0)
    family = assemble_family(
        {"A1": _grid(targets, pred_seed=1)}, baseline, exchangeability_asserted=False
    )
    # Keys produced only by the frozen assemble_declared_family.
    assert "holm_bonferroni" in family
    assert "present_candidates" in family
    assert "both_arms_present" in family
    assert set(family["slots"]) == set(declared_family_slots())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
