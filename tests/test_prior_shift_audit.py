"""
Train-serve MCMC prior audit.

``prepare_data.py`` used to abort the whole ETL when a two-sample KS test
found the out-of-fold priors and the fold-ensemble priors to be
differently distributed.  They always are: training reads one fold
model, serving averages K of them, and averaging shrinks variance by
construction.  On the real 360-trial dataset every class rejected
(p = 9e-26 … 1e-31), so no dataset could be built at all.

The KS gate was replaced by an argmax-agreement floor, and that floor has
now been removed too, for a reason the tests below pin down.  The MCMC
Prior is a 4-D continuous *input feature*: its only consumer is the MoR
Router, a ``Linear(hidden + mcmc_dim, 2)`` reading the concatenated
vector.  The router never sees an argmax.  So

* argmax *collapse* is not a failure mode for this consumer — and it
  trivially satisfies an agreement floor, which made the floor
  satisfiable by degrading the generator;
* the failure that does matter, the prior vector collapsing to a
  constant, is already covered by the bootstrap per-column variance
  floor in ``prepare_dataset`` — left untouched;
* the honest train-serve quantity is a distance on the *vector*, which
  the record already carries as ``mean_total_variation_distance``.

These tests pin: invalid priors still raise, everything else is recorded,
and total-variation distance sees changes that argmax agreement is blind
to.  No new threshold is introduced — deliberately.  The 0.65 floor was
itself an unvalidated constant that aborted the real-data ETL at 0.611,
and replacing it with another guessed number would repeat that.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.prepare_data import audit_prior_train_serve_shift

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


def test_maximal_decision_disagreement_is_recorded_not_fatal() -> None:
    """
    Reversed priors — near-total argmax disagreement — are recorded.

    This replaces a test formerly named for "decision collapse".  It never
    built a collapsed generator: it reverses the columns, which is maximal
    *disagreement*, the opposite of collapse.  Neither condition is a gate
    any more, so what it pins now is that even the most extreme argmax
    divergence returns a record, and that the vector distance registers it.
    """
    oof, _ = _oof_and_ensemble(n=200)
    reversed_priors = oof[:, ::-1].copy()
    agreement = float(
        np.mean(oof.argmax(axis=1) == reversed_priors.argmax(axis=1))
    )
    assert agreement < 0.65, "fixture must disagree"

    record = audit_prior_train_serve_shift(oof, reversed_priors, CLASS_NAMES)
    assert record["argmax_agreement"] == agreement
    assert record["mean_total_variation_distance"] > 0.0


def test_empty_input_is_rejected() -> None:
    empty = np.zeros((0, 4))
    with pytest.raises(ValueError, match="at least one trial"):
        audit_prior_train_serve_shift(empty, empty, CLASS_NAMES)


# ── Why the argmax floor was removed ─────────────────────────────────
#
# Three properties, each a separate test below:
#
#   1. The real-data value that used to abort the ETL (0.611) must now
#      produce a record.
#   2. A mode-collapsed generator — argmax constant across every trial,
#      predicting zero instances of the minority classes — scores PERFECT
#      agreement.  Measured on real data: forcing 3 folds collapsed the
#      generator and drove agreement to 1.000 while Prewalk and
#      Pre_Active each received zero predictions.  A floor satisfiable by
#      degrading the model is worse than no floor.
#   3. Total-variation distance registers vector changes that leave every
#      argmax untouched — i.e. it sees what the MoR Router reads and
#      argmax agreement does not.
#
# 100 trials give agreement = k/100, so a target is an exact IEEE-754
# float.


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


def test_real_data_agreement_is_recorded_not_fatal() -> None:
    """
    0.611 — the value measured on the real corpus — must not abort.

    This is the exact number that made ``scripts/prepare_data.py`` raise on
    ``data/raw``, blocking the only end-to-end path through the pipeline.
    Everything downstream of the ETL ran fine once the abort was bypassed,
    so this single comparison was the whole blockage.
    """
    n, n_agree = 1000, 611
    oof, ensemble = _make_controlled_agreement_priors(n, n_agree)
    assert float(
        np.mean(oof.argmax(axis=1) == ensemble.argmax(axis=1))
    ) == 0.611

    record = audit_prior_train_serve_shift(oof, ensemble, CLASS_NAMES)
    assert record["argmax_agreement"] == 0.611
    assert record["n_trials"] == n
    assert record["argmax_agreement_is_descriptive_only"] is True


def test_mode_collapse_scores_perfect_agreement() -> None:
    """
    A collapsed generator satisfies an agreement floor trivially.

    Both sides nominate class 0 for every trial, exactly as a generator
    that has collapsed onto the majority class would.  Agreement is 1.0 —
    the best possible score — while the generator is at its least
    informative.  Pinning this keeps the removed floor from being
    reintroduced as an apparent safety improvement.
    """
    n = 200
    oof, ensemble = _make_controlled_agreement_priors(n, n)
    assert np.all(oof.argmax(axis=1) == 0)
    assert np.all(ensemble.argmax(axis=1) == 0)

    record = audit_prior_train_serve_shift(oof, ensemble, CLASS_NAMES)
    assert record["argmax_agreement"] == 1.0


def test_total_variation_sees_what_argmax_agreement_cannot() -> None:
    """
    Perfect argmax agreement, materially different vectors.

    The MoR Router consumes the prior vector, so ``[0.97, 0.01, 0.01, 0.01]``
    and ``[0.40, 0.20, 0.20, 0.20]`` are different inputs even though both
    nominate class 0.  Argmax agreement scores this pair 1.0 and is blind
    to the shift; total-variation distance is not.
    """
    n = 128
    confident = np.tile([0.97, 0.01, 0.01, 0.01], (n, 1))
    diffuse = np.tile([0.40, 0.20, 0.20, 0.20], (n, 1))

    record = audit_prior_train_serve_shift(confident, diffuse, CLASS_NAMES)

    assert record["argmax_agreement"] == 1.0, "argmax is blind here"
    assert record["mean_total_variation_distance"] == pytest.approx(0.57)
