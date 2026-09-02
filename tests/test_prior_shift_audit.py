"""
Train-serve MCMC prior audit.

``prepare_data.py`` used to abort the whole ETL when a two-sample KS test
found the out-of-fold priors and the fold-ensemble priors to be
differently distributed.  They always are: training reads one fold
model, serving averages K of them, and averaging shrinks variance by
construction.  On the real 360-trial dataset every class rejected
(p = 9e-26 … 1e-31), so no dataset could be built at all.

These tests pin the replacement: expected shrinkage is recorded, and
only genuinely invalid priors or a collapse in decision agreement raise.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.prepare_data import (
    PRIOR_MIN_ARGMAX_AGREEMENT,
    audit_prior_train_serve_shift,
)

CLASS_NAMES = ["ESCAPE", "PREWALK", "PRE_ACTIVE", "NO_RESPONSE"]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _oof_and_ensemble(
    n: int = 400, shrink: float = 0.55, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """OOF priors plus the variance-compressed ensemble they'd be served as."""
    rng = np.random.default_rng(seed)
    oof = _softmax(rng.normal(0.0, 2.0, size=(n, len(CLASS_NAMES))))
    ensemble = shrink * oof + (1.0 - shrink) * oof.mean(axis=0, keepdims=True)
    ensemble /= ensemble.sum(axis=1, keepdims=True)
    return oof, ensemble


def test_expected_ensemble_shrinkage_is_recorded_not_fatal() -> None:
    oof, ensemble = _oof_and_ensemble()
    record = audit_prior_train_serve_shift(oof, ensemble, CLASS_NAMES)

    assert record["n_trials"] == oof.shape[0]
    assert record["n_classes"] == oof.shape[1]
    assert record["argmax_agreement"] > 0.8

    # Every column must show the shrinkage the averaging causes, and the
    # KS test must be marked descriptive rather than used as a gate.
    for col in record["columns"].values():
        assert col["var_ratio_serve_over_oof"] < 1.0
        assert "ks_pvalue_descriptive_only" in col
    assert record["ks_is_descriptive_only"] is True

    # Proof the retired gate would have killed this healthy input: at
    # least one column rejects at the old Bonferroni threshold 0.01/4.
    assert any(
        col["ks_pvalue_descriptive_only"] < 0.0025
        for col in record["columns"].values()
    )


def test_record_is_json_serialisable() -> None:
    """It is persisted into nsmor_dataset.pt and read by reviewers."""
    oof, ensemble = _oof_and_ensemble(n=64)
    record = audit_prior_train_serve_shift(oof, ensemble, CLASS_NAMES)
    round_tripped = json.loads(json.dumps(record))
    assert round_tripped["columns"]["0"]["class"] == "ESCAPE"


def test_identical_priors_show_no_shift() -> None:
    oof, _ = _oof_and_ensemble(n=128)
    record = audit_prior_train_serve_shift(oof, oof.copy(), CLASS_NAMES)
    assert record["argmax_agreement"] == 1.0
    assert record["mean_total_variation_distance"] == pytest.approx(0.0)
    for col in record["columns"].values():
        assert col["mean_signed_diff"] == pytest.approx(0.0)
        assert col["max_abs_diff"] == pytest.approx(0.0)


def test_signed_bias_is_reported_with_a_paired_interval() -> None:
    """A systematic shift must surface as signed bias, not just |Δ|."""
    oof, _ = _oof_and_ensemble(n=256)
    biased = oof + np.array([0.05, -0.05, 0.0, 0.0])
    biased = np.clip(biased, 1e-12, None)
    biased /= biased.sum(axis=1, keepdims=True)

    record = audit_prior_train_serve_shift(oof, biased, CLASS_NAMES)
    escape = record["columns"]["0"]
    assert escape["mean_signed_diff"] > 0.0
    low, high = escape["mean_signed_diff_ci95"]
    assert low <= escape["mean_signed_diff"] <= high


def test_non_finite_priors_are_rejected() -> None:
    oof, ensemble = _oof_and_ensemble(n=32)
    ensemble[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        audit_prior_train_serve_shift(oof, ensemble, CLASS_NAMES)


def test_negative_priors_are_rejected() -> None:
    oof, ensemble = _oof_and_ensemble(n=32)
    oof[0] = np.array([1.2, -0.2, 0.0, 0.0])
    with pytest.raises(ValueError, match="negative"):
        audit_prior_train_serve_shift(oof, ensemble, CLASS_NAMES)


def test_unnormalised_rows_are_rejected() -> None:
    oof, ensemble = _oof_and_ensemble(n=32)
    ensemble[5] *= 0.5
    with pytest.raises(ValueError, match="deviate from 1"):
        audit_prior_train_serve_shift(oof, ensemble, CLASS_NAMES)


def test_shape_mismatch_is_rejected() -> None:
    oof, ensemble = _oof_and_ensemble(n=32)
    with pytest.raises(ValueError, match="shape mismatch"):
        audit_prior_train_serve_shift(oof, ensemble[:16], CLASS_NAMES)


def test_decision_collapse_still_aborts_the_etl() -> None:
    """
    The gate that survives: if the served prior nominates a different
    class than the trained-on prior for most trials, the deployment
    protocol really is broken.
    """
    oof, _ = _oof_and_ensemble(n=200)
    reversed_priors = oof[:, ::-1].copy()
    agreement = float(
        np.mean(oof.argmax(axis=1) == reversed_priors.argmax(axis=1))
    )
    assert agreement < PRIOR_MIN_ARGMAX_AGREEMENT, "fixture must disagree"

    with pytest.raises(ValueError, match="most likely class"):
        audit_prior_train_serve_shift(oof, reversed_priors, CLASS_NAMES)


def test_empty_input_is_rejected() -> None:
    empty = np.zeros((0, 4))
    with pytest.raises(ValueError, match="at least one trial"):
        audit_prior_train_serve_shift(empty, empty, CLASS_NAMES)


# ── Boundary-condition tests for PRIOR_MIN_ARGMAX_AGREEMENT ──────────
#
# CONTEXT.md defines Argmax Agreement as "Fraction of trials where OOF
# and ensemble priors agree on the most-likely behavioral class. Hard
# floor at 0.65; below this the deployment protocol is invalidated."
# The constant PRIOR_MIN_ARGMAX_AGREEMENT = 0.65 implements that floor.
#
# These tests exercise the exact boundary (0.65 passes) and the first
# value below it (0.64 raises) using synthetic priors with surgically
# controlled argmax agreement.  100 trials give agreement = k/100 so
# the boundary can be expressed as an exact IEEE-754 float.


def _make_controlled_agreement_priors(
    n: int,
    n_agree: int,
    *,
    seed: int = 99,
) -> tuple[np.ndarray, np.ndarray]:
    """Build valid (n, 4) probability pairs with exact argmax agreement.

    For the first ``n_agree`` trials, both arrays share the same argmax
    (class 0 dominant).  For the remaining ``n - n_agree`` trials, the
    ensemble argmax is rotated to class 1, guaranteeing disagreement.

    Both arrays are strictly positive, finite, and row-normalised.

    Args:
        n: Total number of synthetic trials.
        n_agree: Number of trials whose argmax agrees.
        seed: RNG seed for reproducibility.

    Returns:
        ``(oof, ensemble)`` each of shape ``(n, 4)``.
    """
    assert 0 <= n_agree <= n, f"n_agree={n_agree} out of [0, {n}]"
    rng = np.random.default_rng(seed)

    # -- OOF priors: class-0 dominant for every trial --
    # Start with a small uniform floor, then boost class 0.
    oof = rng.uniform(0.01, 0.05, size=(n, 4))
    oof[:, 0] += 0.80  # class 0 is always argmax for OOF
    oof /= oof.sum(axis=1, keepdims=True)
    assert np.all(oof.argmax(axis=1) == 0), "OOF fixture: class 0 must dominate"

    # -- Ensemble priors: agree on first n_agree, disagree on the rest --
    ensemble = oof.copy()
    n_disagree = n - n_agree
    if n_disagree > 0:
        # For disagreeing trials, swap mass from class 0 to class 1.
        disagree_block = ensemble[n_agree:].copy()
        disagree_block[:, 0] = 0.02
        disagree_block[:, 1] = 0.85
        disagree_block[:, 2] = 0.08
        disagree_block[:, 3] = 0.05
        # Re-normalise to guarantee row sums == 1.
        disagree_block /= disagree_block.sum(axis=1, keepdims=True)
        assert np.all(
            disagree_block.argmax(axis=1) == 1
        ), "Ensemble disagree fixture: class 1 must dominate"
        ensemble[n_agree:] = disagree_block

    # Sanity: exact agreement fraction
    actual = float(np.mean(oof.argmax(axis=1) == ensemble.argmax(axis=1)))
    expected = n_agree / n
    assert actual == expected, (
        f"Fixture agreement {actual} != expected {expected}"
    )
    return oof, ensemble


def test_exact_floor_agreement_passes() -> None:
    """agreement = 0.65 (the hard floor) must NOT raise.

    CONTEXT.md: 'Hard floor at 0.65' means >= 0.65 is valid.
    100 trials, 65 agreeing => agreement = 65/100 = 0.65 exactly.
    """
    n, n_agree = 100, 65
    oof, ensemble = _make_controlled_agreement_priors(n, n_agree)

    # Pre-condition: agreement is exactly at the floor
    assert float(
        np.mean(oof.argmax(axis=1) == ensemble.argmax(axis=1))
    ) == PRIOR_MIN_ARGMAX_AGREEMENT

    # Must return a record without raising
    record = audit_prior_train_serve_shift(oof, ensemble, CLASS_NAMES)
    assert record["argmax_agreement"] == PRIOR_MIN_ARGMAX_AGREEMENT
    assert record["n_trials"] == n
    assert record["min_argmax_agreement_gate"] == PRIOR_MIN_ARGMAX_AGREEMENT


def test_below_floor_agreement_raises_with_most_likely_class() -> None:
    """agreement = 0.64 (one trial below the floor) must raise ValueError.

    CONTEXT.md: 'below this the deployment protocol is invalidated.'
    The error message must mention 'most likely class' per the existing
    gate in audit_prior_train_serve_shift.
    """
    n, n_agree = 100, 64
    oof, ensemble = _make_controlled_agreement_priors(n, n_agree)

    # Pre-condition: agreement is strictly below the floor
    actual_agreement = float(
        np.mean(oof.argmax(axis=1) == ensemble.argmax(axis=1))
    )
    assert actual_agreement == 0.64
    assert actual_agreement < PRIOR_MIN_ARGMAX_AGREEMENT

    with pytest.raises(ValueError, match="most likely class"):
        audit_prior_train_serve_shift(oof, ensemble, CLASS_NAMES)
