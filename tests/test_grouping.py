"""The train/val split must be ANIMAL-disjoint, not merely session-disjoint.

``session_ids`` look like ``0.513cricket_001_20260707_193143_session_2``.
The ``_session_N`` suffix splits one recording of one animal into blocks,
so grouping the split by session leaves the animal free to straddle both
sides.  Measured on ``nsmor_dataset_3cond_v2.pt`` at the project's default
seed, 87.5% of validation trials shared an animal with a training trial.

That defect is invisible to any session-granularity assertion, which is
why these tests assert at animal granularity and use realistic ids.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from nsmor.pipeline.grouping import (
    animal_of,
    animal_keys_of,
    check_group_disjoint,
    grouped_train_val_split,
    resolve_group_folds,
)


def _session_ids(n_animals: int, blocks: int = 2, trials: int = 3):
    """Realistic ids: ``n_animals`` animals x ``blocks`` sessions each."""
    return [
        f"0.{500 + a}cricket_001_20260101_00000{a}_session_{b + 1}"
        for a in range(n_animals)
        for b in range(blocks)
        for _ in range(trials)
    ]


class TestAnimalOf:
    """The suffix strip is the whole definition of 'same animal'."""

    def test_strips_single_digit_suffix(self) -> None:
        assert (
            animal_of("0.513cricket_001_20260707_193143_session_2")
            == "0.513cricket_001_20260707_193143"
        )

    def test_strips_multi_digit_suffix(self) -> None:
        assert animal_of("animal_x_session_12") == "animal_x"

    def test_leaves_unsuffixed_id_untouched(self) -> None:
        assert animal_of("sess_0") == "sess_0"

    def test_only_strips_a_trailing_suffix(self) -> None:
        # An interior "_session_1" is part of the animal key, not a block.
        assert animal_of("a_session_1_b") == "a_session_1_b"

    def test_accepts_non_str(self) -> None:
        assert animal_of(np.str_("x_session_3")) == "x"

    def test_two_blocks_of_one_animal_share_a_key(self) -> None:
        base = "0.513cricket_001_20260707_193143"
        assert animal_of(f"{base}_session_1") == animal_of(f"{base}_session_2")


class TestAnimalKeysOf:
    def test_shape_and_grouping(self) -> None:
        sids = _session_ids(n_animals=3, blocks=2, trials=1)
        keys = animal_keys_of(sids)
        assert keys.shape == (6,)
        assert len(set(keys.tolist())) == 3


class TestGroupedTrainValSplit:
    """The core invariant, and the fallback's honesty about lacking it."""

    def test_no_animal_spans_both_splits(self) -> None:
        sids = _session_ids(n_animals=10)
        tr, va = grouped_train_val_split(sids, len(sids), 0.2, 42)
        keys = animal_keys_of(sids)
        assert not (
            set(keys[tr].tolist()) & set(keys[va].tolist())
        ), "an animal appeared in both train and val"

    def test_session_disjoint_is_not_sufficient(self) -> None:
        """Pin the distinction the old implementation got wrong.

        A session-grouped split on this fixture IS session-disjoint yet
        NOT animal-disjoint, so the weaker assertion cannot detect it.
        """
        sids = _session_ids(n_animals=10)
        session_arr = np.asarray(sids)
        rng = np.random.RandomState(42)
        unique_sessions = np.unique(session_arr)
        rng.shuffle(unique_sessions)
        n_val = max(1, int(len(unique_sessions) * 0.2))
        val_sessions = set(unique_sessions[:n_val].tolist())
        is_val = np.array([s in val_sessions for s in session_arr])

        old_tr = np.nonzero(~is_val)[0]
        old_va = np.nonzero(is_val)[0]
        keys = animal_keys_of(sids)
        leaked = set(keys[old_tr].tolist()) & set(keys[old_va].tolist())
        assert leaked, (
            "fixture no longer reproduces the session-grouping defect, so "
            "this test would pass vacuously"
        )

        # The real function must not leak on the same input.
        new_tr, new_va = grouped_train_val_split(sids, len(sids), 0.2, 42)
        assert not (
            set(keys[new_tr].tolist()) & set(keys[new_va].tolist())
        )

    def test_partition_is_total_and_disjoint(self) -> None:
        sids = _session_ids(n_animals=7)
        tr, va = grouped_train_val_split(sids, len(sids), 0.2, 42)
        assert sorted(tr.tolist() + va.tolist()) == list(range(len(sids)))
        assert not set(tr.tolist()) & set(va.tolist())

    def test_val_is_never_empty(self) -> None:
        # Two animals at val_split=0.2 floors to 0; max(1, ...) rescues it.
        sids = _session_ids(n_animals=2)
        tr, va = grouped_train_val_split(sids, len(sids), 0.2, 42)
        assert va.size > 0 and tr.size > 0

    def test_same_seed_same_split(self) -> None:
        sids = _session_ids(n_animals=9)
        a = grouped_train_val_split(sids, len(sids), 0.2, 42)
        b = grouped_train_val_split(sids, len(sids), 0.2, 42)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_different_seed_different_split(self) -> None:
        sids = _session_ids(n_animals=9)
        a = grouped_train_val_split(sids, len(sids), 0.2, 42)
        b = grouped_train_val_split(sids, len(sids), 0.2, 7)
        assert not np.array_equal(a[1], b[1])

    def test_returns_int64(self) -> None:
        sids = _session_ids(n_animals=5)
        tr, va = grouped_train_val_split(sids, len(sids), 0.2, 42)
        assert tr.dtype == np.int64 and va.dtype == np.int64

    def test_fallback_on_missing_ids_warns(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            tr, va = grouped_train_val_split(None, 10, 0.2, 42)
        assert tr.size + va.size == 10
        assert "sample-level" in caplog.text

    def test_fallback_on_length_mismatch_warns(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            tr, va = grouped_train_val_split(["a", "b"], 10, 0.2, 42)
        assert tr.size + va.size == 10
        assert "sample-level" in caplog.text

    def test_logs_achieved_fraction(self, caplog) -> None:
        """Whole-animal assignment cannot hit the target exactly; say so."""
        sids = _session_ids(n_animals=8)
        with caplog.at_level("INFO"):
            grouped_train_val_split(sids, len(sids), 0.2, 42)
        assert "Animal-grouped split" in caplog.text
        assert "target" in caplog.text


class TestResolveGroupFolds:
    """Coarsening to animals can starve a rare class of groups.

    The escape class is ~3% of the corpus and concentrated in few animals,
    so the fold count must adapt.  What must NOT happen is the grouping
    weakening back to sessions to keep a round fold number.
    """

    def test_returns_max_when_coverage_is_ample(self) -> None:
        labels = np.array([0, 1] * 10)
        keys = np.array([f"a{i}" for i in range(20)], dtype=object)
        assert resolve_group_folds(labels, keys, max_folds=5) == 5

    def test_caps_at_rarest_class_coverage(self) -> None:
        # class 1 lives in exactly 3 groups -> at most 3 folds.
        labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
        keys = np.array(
            ["a", "b", "c", "d", "e", "f", "g", "h", "i"], dtype=object
        )
        keys[6:] = ["g", "h", "i"]
        assert resolve_group_folds(labels, keys, max_folds=5) == 3

    def test_never_exceeds_max_folds(self) -> None:
        labels = np.array([0, 1] * 25)
        keys = np.array([f"a{i}" for i in range(50)], dtype=object)
        assert resolve_group_folds(labels, keys, max_folds=3) == 3

    def test_warns_when_reducing(self, caplog) -> None:
        labels = np.array([0, 0, 0, 0, 1, 1])
        keys = np.array(["a", "b", "c", "d", "e", "f"], dtype=object)
        with caplog.at_level("WARNING"):
            n = resolve_group_folds(labels, keys, max_folds=5)
        assert n == 2
        assert "Reducing cross-fitting folds" in caplog.text
        assert "NOT weakened" in caplog.text

    def test_raises_when_a_class_owns_one_group(self) -> None:
        # class 1 lives in a single animal: no fold can hold it on both
        # sides, so this must fail loudly rather than degrade.
        labels = np.array([0, 0, 0, 0, 1, 1])
        keys = np.array(["a", "b", "c", "d", "e", "e"], dtype=object)
        with pytest.raises(ValueError, match="fewer than 2 groups"):
            resolve_group_folds(labels, keys, max_folds=5)

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length"):
            resolve_group_folds(
                np.array([0, 1]), np.array(["a"], dtype=object), 5
            )

    def test_animal_coarsening_can_cost_folds(self) -> None:
        """The whole reason this function exists, stated as a test.

        One animal, two sessions, carrying the only instances of a rare
        class: session-grouping sees 2 groups, animal-grouping sees 1.
        """
        sids = np.array(
            [
                "x_session_1", "x_session_2",  # rare class, one animal
                "y_session_1", "z_session_1", "w_session_1",
            ],
            dtype=object,
        )
        labels = np.array([1, 1, 0, 0, 0])
        # By session the rare class spans 2 groups -> a split is possible.
        assert resolve_group_folds(labels, sids, max_folds=5) == 2
        # By animal it spans 1 -> impossible, and it says so.
        with pytest.raises(ValueError, match="fewer than 2 groups"):
            resolve_group_folds(labels, animal_keys_of(sids), max_folds=5)


class TestCheckGroupDisjoint:
    def test_raises_on_leak(self) -> None:
        keys = np.array(["a", "a", "b"], dtype=object)
        with pytest.raises(ValueError, match="leaks 1 group"):
            check_group_disjoint(np.array([0]), np.array([1, 2]), keys)

    def test_silent_when_clean(self) -> None:
        keys = np.array(["a", "a", "b"], dtype=object)
        check_group_disjoint(np.array([0, 1]), np.array([2]), keys)


class TestDeterminismAcrossProcesses:
    """``set`` iteration order varies with PYTHONHASHSEED; ``np.unique``
    does not.  The lazy path used ``list(set(...))``, so its split was not
    reproducible across processes even at a fixed seed -- and therefore
    could not match the eager path or ``compute_target_stats``.
    """

    def test_split_is_stable_under_hash_randomization(self) -> None:
        prog = (
            "import numpy as np;"
            "from nsmor.pipeline.grouping import grouped_train_val_split;"
            "sids=[f'0.{500+a}cricket_001_20260101_00000{a}_session_{b+1}'"
            " for a in range(9) for b in range(2) for _ in range(3)];"
            "tr,va=grouped_train_val_split(sids,len(sids),0.2,42);"
            "print(','.join(map(str,va.tolist())))"
        )
        outs = set()
        for seed in ("0", "1", "12345"):
            res = subprocess.run(
                [sys.executable, "-c", prog],
                capture_output=True,
                text=True,
                timeout=300,
                env={"PYTHONHASHSEED": seed, "PATH": ""},
            )
            assert res.returncode == 0, res.stderr[-1500:]
            outs.add(res.stdout.strip())
        assert len(outs) == 1, (
            f"split varied with PYTHONHASHSEED: {outs}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
