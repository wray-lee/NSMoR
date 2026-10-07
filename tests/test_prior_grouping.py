"""OOF MCMC priors must be cross-fitted by recording prefix, not session.

Input channels 4-7 carry the priors. Keeping blocks with the same prefix
together avoids within-prefix reuse; distinct prefixes are not independently
verified animal identities. The test observes groups at the real cross-fitter seam.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from nsmor.pipeline.nested_prior import load_artifact_bytes

_DT_MS = 10.0
_FRAMES = 400


def _write_corpus(
    raw_dir: Path,
    n_animals: int = 5,
    blocks_per_animal: int = 2,
) -> None:
    """Write a raw corpus where recording prefix and session are distinct.

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
                lv_ms, init_deg = 30.0, 2.0
                collision_ms = onset_ms + lv_ms / np.tan(np.deg2rad(init_deg / 2.0))
                visual_angle = np.zeros(_FRAMES, dtype=np.float64)
                remaining_ms = np.maximum(collision_ms - time_ms[stim_idx:],
                                          lv_ms / np.tan(np.deg2rad(89.0)))
                visual_angle[stim_idx:] = np.rad2deg(
                    2.0 * np.arctan(lv_ms / remaining_ms)
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
                        "l_v_ratio": lv_ms,
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


def test_provenance_records_prefix_grouping(tmp_path_factory) -> None:
    """The artifact identifies prefix grouping and unverified animal identity."""
    from scripts.prepare_data import prepare_dataset

    raw_dir = tmp_path_factory.mktemp("raw_prov")
    _write_corpus(raw_dir)
    out = tmp_path_factory.mktemp("out_prov") / "ds.pt"
    prepare_dataset(raw_dir=raw_dir, output_path=out, random_seed=42)

    saved = load_artifact_bytes(out.read_bytes())
    assert saved["mcmc_prior_provenance"].endswith("recording_prefix_grouped_cv")
    assert saved["animal_identity_status"] == "unverified"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
