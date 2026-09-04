"""The routing auxiliary loss must never be silently absent.

``compute_routing_aux_loss`` returns exactly ``0.0`` when either condition
group is empty (``nsmor/loss_ext.py``).  Combined with a corpus that omits
the condition stamp, that gave two independent ways for the whole auxiliary
term to vanish from training while the log looked normal:

1. ``scripts/train.py`` read ``dataset.get("is_pure_wind")`` and, when the
   key was absent, neither derived it nor warned.  ``is_pure_wind=None``
   then made ``NSMoRDataset`` yield 2-tuples, so ``collate_with_metadata``
   never attached a mask and the frozen loss skipped the term outright.
2. Even with metadata present, a corpus containing no ``wind_only`` trials
   leaves the hinge with an empty group, so it returns 0 for every batch.

Four of the five processed corpora carry ``pipeline_semantics_version``
2.2 yet no stamp, and the Makefile default corpus has zero ``wind_only``
trials -- so both paths were live, not hypothetical.

These tests use synthetic arrays only: CI has no corpora, so a test that
needs a ``.pt`` file is a hard CI failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.make_subset_dataset import derive_stimulus_metadata
from scripts.train import (
    check_routing_aux_active,
    resolve_condition_metadata,
)


def _trial(has_visual: bool, has_wind: bool, n_frames: int = 6) -> np.ndarray:
    """Build an ``(n_frames, 8)`` trial with the requested channels."""
    x = np.zeros((n_frames, 8), dtype=np.float32)
    if has_visual:
        x[2:, 0] = 12.5  # visual_angle
    if has_wind:
        x[3:, 1] = 1.0  # wind_state
    return x


def _corpus(specs: list[tuple[bool, bool]]) -> dict[str, object]:
    """An unstamped corpus dict: X_seqs + lengths only."""
    x_seqs = [_trial(has_visual=v, has_wind=w) for v, w in specs]
    return {"X_seqs": x_seqs, "lengths": [6] * len(x_seqs)}


# ── resolve_condition_metadata ────────────────────────────────────────


class TestResolveConditionMetadata:
    """Metadata must be produced whether or not the corpus stamps it."""

    def test_derives_when_stamp_absent(self):
        """An unstamped corpus yields derived metadata, flagged as derived."""
        corpus = _corpus([(False, True), (True, False)])

        conditions, is_pure_wind, derived = resolve_condition_metadata(
            corpus, corpus["X_seqs"], corpus["lengths"]
        )

        assert derived is True
        assert list(conditions) == ["wind_only", "visual_only"]
        assert is_pure_wind.tolist() == [True, False]

    def test_derived_matches_canonical(self):
        """Derivation defers to the canonical function, not a local copy."""
        corpus = _corpus(
            [(False, False), (True, False), (False, True), (True, True)]
        )

        conditions, is_pure_wind, _ = resolve_condition_metadata(
            corpus, corpus["X_seqs"], corpus["lengths"]
        )
        want_conditions, want_pw = derive_stimulus_metadata(
            corpus["X_seqs"], corpus["lengths"]
        )

        assert list(conditions) == list(want_conditions)
        assert is_pure_wind.tolist() == want_pw.tolist()

    def test_prefers_stored_stamp(self):
        """A stamped corpus is trusted and not reported as derived."""
        corpus = _corpus([(False, True), (True, False)])
        corpus["stimulus_conditions"] = np.asarray(
            ["wind_only", "visual_only"], dtype=object
        )
        corpus["is_pure_wind"] = np.asarray([True, False], dtype=bool)

        conditions, is_pure_wind, derived = resolve_condition_metadata(
            corpus, corpus["X_seqs"], corpus["lengths"]
        )

        assert derived is False
        assert list(conditions) == ["wind_only", "visual_only"]
        assert is_pure_wind.tolist() == [True, False]

    def test_warns_when_stamp_contradicts_channels(self, caplog):
        """A stale stamp is surfaced rather than trusted in silence."""
        corpus = _corpus([(False, True), (True, False)])
        # Deliberately wrong: channels say [wind_only, visual_only].
        corpus["is_pure_wind"] = np.asarray([False, True], dtype=bool)

        with caplog.at_level("WARNING"):
            _, is_pure_wind, _ = resolve_condition_metadata(
                corpus, corpus["X_seqs"], corpus["lengths"]
            )

        assert is_pure_wind.tolist() == [False, True], "stamp must still win"
        assert "disagrees with the physical channels" in caplog.text

    def test_no_stimulus_is_not_folded_into_wind(self):
        """The regression that motivated all of this.

        A no_stimulus trial has both channels silent.  Any derivation that
        tests only "visual is silent" labels it pure-wind and contaminates
        the wind group.
        """
        corpus = _corpus([(False, False), (True, False)])

        conditions, is_pure_wind, _ = resolve_condition_metadata(
            corpus, corpus["X_seqs"], corpus["lengths"]
        )

        assert list(conditions) == ["no_stimulus", "visual_only"]
        assert not is_pure_wind.any(), (
            "no_stimulus was labeled pure-wind; the wind group of every "
            "per-condition statistic is contaminated"
        )


# ── assert_routing_aux_trainable ──────────────────────────────────────


class TestCheckRoutingAuxActive:
    """An inert term must be reported, but must not crash the run.

    Not crashing is deliberate (User Story 12, covered by
    ``tests/test_routing_aux_e2e.py``): a sweep over the weight should not
    die on a corpus that happens to lack wind trials.  The defect being
    fixed is the *silence*, so the contract is "return False and log at
    ERROR", not "raise".
    """

    @staticmethod
    def _call(specs, weight, split="train"):
        corpus = _corpus(specs)
        conditions, is_pure_wind = derive_stimulus_metadata(
            corpus["X_seqs"], corpus["lengths"]
        )
        return check_routing_aux_active(
            is_pure_wind, conditions, weight, split, "synthetic.pt"
        )

    def test_inert_when_no_wind_trials(self, caplog):
        """The Makefile default corpus shape: visual and no_stimulus only."""
        with caplog.at_level("ERROR"):
            active = self._call(
                [(True, False), (False, False), (True, True)], 0.5
            )

        assert active is False
        assert "ROUTING AUX INERT" in caplog.text
        assert "no pure-wind trials" in caplog.text
        assert "visual_only" in caplog.text, "census must be logged"
        assert "lambda_routing_aux=0" in caplog.text, "name the opt-out"

    def test_inert_when_no_visual_trials(self, caplog):
        """The mirror case: every trial is pure wind."""
        with caplog.at_level("ERROR"):
            active = self._call([(False, True), (False, True)], 0.5)

        assert active is False
        assert "no non-pure-wind trials" in caplog.text

    def test_names_the_split(self, caplog):
        """The message must say which split is inert."""
        with caplog.at_level("ERROR"):
            self._call([(True, False)], 0.5, split="val")

        assert "val split" in caplog.text

    def test_active_when_both_groups_present(self, caplog):
        """A corpus with both groups reports active and logs nothing."""
        with caplog.at_level("ERROR"):
            active = self._call([(False, True), (True, False)], 0.5)

        assert active is True
        assert caplog.text == "", "an active term must not warn"

    def test_inactive_but_quiet_when_weight_disabled(self, caplog):
        """Weight 0 is not active, and is not worth an ERROR."""
        with caplog.at_level("ERROR"):
            active = self._call([(True, False), (False, False)], 0.0)

        assert active is False
        assert "ROUTING AUX INERT" not in caplog.text

    def test_boundary_single_trial_each_side(self):
        """One trial per group is the minimum the hinge can contrast."""
        assert self._call([(False, True), (True, False)], 1e-9) is True

    def test_tiny_positive_weight_still_reported(self, caplog):
        """Inertness is reported for any positive weight, however small."""
        with caplog.at_level("ERROR"):
            active = self._call([(True, False)], 1e-12)

        assert active is False
        assert "ROUTING AUX INERT" in caplog.text


# ── the frozen hinge's own degenerate-group behaviour ──────────────────


class TestHingeGoesSilentOnEmptyGroup:
    """Pin the behaviour that makes the guard necessary.

    If this ever starts raising instead of returning 0, the guard above is
    redundant and should be reconsidered rather than left in place.
    """

    def test_returns_exactly_zero_when_no_wind_trials(self):
        import torch

        from nsmor.loss_ext import compute_routing_aux_loss

        g_lif = torch.rand(4, 6)
        lengths = torch.full((4,), 6, dtype=torch.long)
        no_wind = torch.zeros(4, dtype=torch.bool)

        loss = compute_routing_aux_loss(g_lif, lengths, no_wind, margin=0.024)

        assert loss.item() == 0.0, (
            "an empty wind group must yield exactly 0.0 -- this is the "
            "silent-inactivity mode the train.py guard exists to catch"
        )

    def test_fires_when_both_groups_present_and_separation_is_short(self):
        """Prove the term is genuinely active, not merely wired up."""
        import torch

        from nsmor.loss_ext import compute_routing_aux_loss

        # Wind gate equals visual gate -> separation 0, well under margin.
        g_lif = torch.full((4, 6), 0.5)
        lengths = torch.full((4,), 6, dtype=torch.long)
        mask = torch.tensor([True, True, False, False])

        loss = compute_routing_aux_loss(g_lif, lengths, mask, margin=0.024)

        assert loss.item() == pytest.approx(0.024, abs=1e-6), (
            "zero separation must incur the full margin as penalty"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
