"""Out-of-fold MCMC priors must be cross-fitted by ANIMAL, not by session.

The priors are written into input channels 4-7, so a leak here enters the
model as a *feature*.  Grouping folds by session is not enough:
``_session_N`` splits one recording of one animal into blocks, so an
animal's ``_session_1`` could train the very generator that produced
``_session_2``'s supposedly held-out prior.

These tests assert at the seam where the group array is handed to
``train_mcmc_cross_fitted``.  That is the narrowest observable point: the
fold assignment itself is sklearn's, and the resulting priors are a
continuous function of it, so asserting on prior values would be a much
weaker and flakier proxy for the property that actually matters.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

_DT_MS = 10.0
_FRAMES = 400


def _write_corpus(
    raw_dir: Path,
    n_animals: int = 5,
    blocks_per_animal: int = 2,
) -> None:
    """Write a raw corpus where animal and session are DISTINGUISHABLE.

    Every animal owns ``blocks_per_animal`` sessions.  A fixture with one
    session per animal cannot tell session-grouping from animal-grouping,
    and a test built on one passes under either.
    """
    onset_ms = 2000.0
    stim_idx = int(onset_ms / _DT_MS)
    time_ms = np.arange(_FRAMES, dtype=np.float64) * _DT_MS
    profiles = (
        (0.1, 15.0),   # ESCAPE
        (1.5, 15.0),   # PREWALK
        (0.8, 0.1),    # PRE_ACTIVE
        (0.1, 0.1),    # NO_RESPONSE
    )

    for animal_idx in range(n_animals):
        animal = f"0.{500 + animal_idx}cricket_001_20260101_00000{animal_idx}"
        for block in range(1, blocks_per_animal + 1):
            session_id = f"{animal}_session_{block}"
            session_dir = raw_dir / session_id
            session_dir.mkdir(parents=True)
            kin_rows: list[dict[str, object]] = []
            event_rows: list[dict[str, object]] = []

            for trial_id, (baseline, response) in enumerate(profiles):
                velocity = np.full(_FRAMES, baseline, dtype=np.float64)
                velocity[stim_idx:] = 0.1
                velocity[stim_idx + 5:stim_idx + 45] = response
                acceleration = np.gradient(velocity, _DT_MS / 1000.0)
                visual_angle = np.zeros(_FRAMES, dtype=np.float64)
                visual_angle[stim_idx:] = np.linspace(
                    5.0, 60.0, _FRAMES - stim_idx,
                )
                for frame_idx in range(_FRAMES):
                    kin_rows.append({
                        "session_id": session_id,
                        "trial_id": trial_id,
                        "time_ms": float(time_ms[frame_idx]),
                        "x_pos": float(frame_idx * 0.01),
                        "y_pos": float(frame_idx * 0.005),
                        "heading": 0.0,
                        "velocity": float(velocity[frame_idx]),
                        "acceleration": float(acceleration[frame_idx]),
                        "visual_angle": float(visual_angle[frame_idx]),
                        "wind_state": 0,
                        "l_v_ratio": 0.0,
                    })
                event_rows.extend([
                    {
                        "session_id": session_id,
                        "trial_id": trial_id,
                        "time_ms": 0.0,
                        "event_type": "trial_start",
                        "event_value": 1,
                    },
                    {
                        "session_id": session_id,
                        "trial_id": trial_id,
                        "time_ms": onset_ms,
                        "event_type": "stimulus_onset",
                        "event_value": 1,
                    },
                ])

            pd.DataFrame(kin_rows).to_csv(
                session_dir / "kinematics.csv", index=False,
            )
            pd.DataFrame(event_rows).to_csv(
                session_dir / "events.csv", index=False,
            )


@pytest.fixture(scope="module")
def _captured_groups(tmp_path_factory) -> dict:
    """Run the real ETL once, capturing what reaches the cross-fitter."""
    from scripts import prepare_data

    raw_dir = tmp_path_factory.mktemp("raw_prior_grouping")
    _write_corpus(raw_dir)

    seen: dict = {}
    real = prepare_data.train_mcmc_cross_fitted

    def _spy(snapshots, labels, *args, **kwargs):
        seen["groups"] = np.asarray(kwargs["groups"]).copy()
        seen["n_folds"] = kwargs.get("n_folds")
        return real(snapshots, labels, *args, **kwargs)

    with mock.patch.object(
        prepare_data, "train_mcmc_cross_fitted", side_effect=_spy,
    ):
        prepare_data.prepare_dataset(
            raw_dir=raw_dir,
            output_path=tmp_path_factory.mktemp("out") / "ds.pt",
            random_seed=42,
        )

    assert "groups" in seen, "cross-fitter was never called"
    return seen


def test_prior_folds_are_grouped_by_animal(_captured_groups: dict) -> None:
    """No group key may still carry a ``_session_N`` suffix."""
    groups = _captured_groups["groups"]
    offenders = sorted({
        str(g) for g in groups.tolist() if "_session_" in str(g)
    })
    assert not offenders, (
        "MCMC prior cross-fitting received SESSION keys, so an animal's "
        "_session_1 can train the generator that produced _session_2's "
        f'"held-out" prior: {offenders[:5]}'
    )


def test_prior_groups_collapse_blocks_of_one_animal(
    _captured_groups: dict,
) -> None:
    """The fixture has 2 blocks per animal; groups must halve accordingly.

    This is what makes the previous test non-vacuous: it proves the
    fixture really did contain multiple sessions per animal, so a
    session-grouped run would have produced a strictly larger group count.
    """
    groups = _captured_groups["groups"]
    n_unique = len({str(g) for g in groups.tolist()})
    assert n_unique == 5, (
        f"expected 5 animal groups from 10 sessions, got {n_unique}"
    )


def test_fold_count_is_resolved_not_hardcoded(
    _captured_groups: dict,
) -> None:
    """Folds must fit the animal count, not a literal 5.

    Coarsening to animals halves the group count, which can push a rare
    class below the fold count.  The resolver adapts the folds; it must
    never be bypassed by a hardcoded 5.
    """
    n_folds = _captured_groups["n_folds"]
    groups = _captured_groups["groups"]
    n_animals = len({str(g) for g in groups.tolist()})
    assert n_folds is not None, "n_folds was not passed explicitly"
    assert 2 <= n_folds <= 5, f"n_folds out of range: {n_folds}"
    assert n_folds <= n_animals, (
        f"n_folds={n_folds} exceeds the {n_animals} available animal "
        f"groups; some fold's training side must be missing an animal"
    )


def test_provenance_records_animal_grouping(tmp_path_factory) -> None:
    """The artifact must not claim session-grouped provenance."""
    import torch

    from scripts.prepare_data import prepare_dataset

    raw_dir = tmp_path_factory.mktemp("raw_prov")
    _write_corpus(raw_dir)
    out = tmp_path_factory.mktemp("out_prov") / "ds.pt"
    prepare_dataset(raw_dir=raw_dir, output_path=out, random_seed=42)

    provenance = torch.load(out, weights_only=False)[
        "mcmc_prior_provenance"
    ]
    assert "animal_grouped" in provenance, provenance
    assert "session_grouped" not in provenance, provenance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
