"""Focused tests for epoch-boundary recovery (Task #25).

Covers the recovery seam added to scripts/train.py:

- A resumed run CONTINUES ``epochs_without_improvement`` and the accumulated
  loss history instead of silently restarting them (interrupted vs
  uninterrupted synthetic CPU trajectory).
- The recovery state persists in every checkpoint type.
- Legacy checkpoints (no recovery stamp) restart counters with explicit
  compatibility semantics; a corrupted current-version payload fails closed.
- The NumPy RNG stream round-trips through the restricted decoder.
- Each restart writes an immutable segment record bound to its parent
  checkpoint's exact bytes; prior segments are never overwritten.

All tests are synthetic and CPU-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from unittest import mock

import numpy as np
import pytest
import torch

from nsmor.pipeline.nested_prior import load_artifact_bytes
from tests.test_train_checkpoint import (
    _checkpoint_shas,
    _make_config,
    _make_synthetic_dataset,
)

if TYPE_CHECKING:  # annotation-only import; no runtime dependency added
    from nsmor.config_parser import ExperimentConfig


def _valid_encoded_rng() -> dict:
    """A valid current-schema ``numpy_rng_state`` for reader unit tests."""
    from scripts.train import _encode_numpy_rng_state

    np.random.seed(7)
    np.random.rand(5)
    return _encode_numpy_rng_state(np.random.get_state())


@pytest.fixture(scope="module", autouse=True)
def _cpu_single_thread() -> Any:
    """Pin this module to single-thread CPU and refuse CUDA.

    The trajectory-equality claim of the real-chain tests is CPU-only (see
    ``docs/epoch-recovery.md``, "Exactness evidence").  A visible CUDA device
    is a hard failure here rather than a silent measurement on a different
    device, and the intra/inter-op thread counts are pinned so the measured
    configuration is explicit rather than inherited from the host.

    Module-scoped because ``torch.set_num_interop_threads`` may be called only
    once per process; if the inter-op pool has already started at a different
    width the pin raises loudly rather than silently measuring a wider
    configuration.  The intra-op width is restored afterwards (the inter-op
    width cannot be un-set, so the module pins it rather than restoring it).
    """
    assert not torch.cuda.is_available(), (
        "epoch-recovery equality tests must run on CPU; CUDA is visible, so "
        "the exact-continuation claim would be measured on the wrong device. "
        "Run with CUDA_VISIBLE_DEVICES='' (see patches/evidence-r5/capture.sh)."
    )
    prev_intra = torch.get_num_threads()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert torch.get_num_threads() == 1, "intra-op threads not pinned to 1"
    assert torch.get_num_interop_threads() == 1, "inter-op threads not pinned"
    yield
    torch.set_num_threads(prev_intra)


# ═════════════════════════════════════════════════════════════
# Interrupted vs uninterrupted continuation
# ═════════════════════════════════════════════════════════════

def test_resume_continues_patience_and_history(tmp_path: Path) -> None:
    """A restart must continue the early-stop counter and history.

    Source run (patience=2) sees val_loss [0.1, 0.2, 0.3]: epoch 0 is the
    best, then two non-improving epochs exhaust the horizon, leaving
    ``epochs_without_improvement == 2`` and history [0.1, 0.2, 0.3] in the
    periodic checkpoint.  The periodic checkpoint is published BEFORE the
    stop check, so the saved counter is already terminal.

    The restart must therefore execute NO epoch and reproduce that terminal
    state.  Executing anything here is the evidence that the counter was NOT
    continued: a run that silently reset it to 0 would sail past the horizon
    and run epochs 3, 4, … .  History and counter are preserved verbatim.
    """
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=10, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    source.training.early_stopping_patience = 2
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.1, 0.2, 0.3]),
    ):
        train(source, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_3.pth"
    saved = load_artifact_bytes(periodic.read_bytes())
    assert saved["recovery_state_version"] == 2
    assert saved["epochs_without_improvement"] == 2
    assert saved["training_history"]["val_loss"] == [0.1, 0.2, 0.3]

    resumed = _make_config(tmp_path, epochs=6, warmup_epochs=0)
    resumed.training.early_stopping_patience = 2
    resumed.checkpoint.output_dir = str(tmp_path / "resumed")
    resumed.checkpoint.resume_from = str(periodic)
    executed: list[int] = []

    def run_epoch(**kwargs: Any) -> Any:
        executed.append(kwargs["epoch"])
        return 1.0, {}

    with (
        mock.patch("scripts.train.train_one_epoch", side_effect=run_epoch),
        mock.patch("scripts.train.validate", side_effect=[0.4, 0.5, 0.6]),
    ):
        result = train(
            resumed, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(
                periodic, source_dir / "best_model.pth"),
        )

    assert executed == [], (
        f"recovery did not continue the patience counter; executed {executed}"
    )
    assert result["history"]["val_loss"] == [0.1, 0.2, 0.3], (
        f"history was not continued across resume: {result['history']['val_loss']}"
    )
    final = load_artifact_bytes(
        (Path(resumed.checkpoint.output_dir) / "final_model.pth").read_bytes())
    assert final["epochs_without_improvement"] == 2
    assert final["epoch"] == 2


def test_recovery_state_present_in_every_checkpoint_type(tmp_path: Path) -> None:
    """best/periodic/final checkpoints all carry the recovery seam."""
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    config.training.checkpoint_interval = 1
    train(config, dataset_path=str(ds))

    output_dir = Path(config.checkpoint.output_dir)
    for name in ("best_model.pth", "epoch_1.pth", "final_model.pth"):
        ckpt = load_artifact_bytes((output_dir / name).read_bytes())
        assert ckpt["recovery_state_version"] == 2, name
        assert isinstance(ckpt["epochs_without_improvement"], int), name
        assert set(ckpt["training_history"]) == {"train_loss", "val_loss"}, name
        assert ckpt["history_start_epoch"] == 0, name
        assert ckpt["numpy_rng_state"]["bit_generator"] == "MT19937", name
        assert ckpt["numpy_rng_state"]["keys"].numel() == 624, name


# ═════════════════════════════════════════════════════════════
# Legacy compatibility and fail-closed validation
# ═════════════════════════════════════════════════════════════

def test_legacy_checkpoint_restarts_counters(tmp_path: Path) -> None:
    """A checkpoint without the recovery stamp restarts counters, loudly."""
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        train(source, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_1.pth"
    legacy = load_artifact_bytes(periodic.read_bytes())
    for key in (
        "recovery_state_version", "epochs_without_improvement",
        "training_history", "history_start_epoch", "numpy_rng_state",
    ):
        legacy.pop(key, None)
    torch.save(legacy, periodic)

    resumed = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "legacy_resume")
    resumed.checkpoint.resume_from = str(periodic)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", return_value=0.5),
    ):
        result = train(
            resumed, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(
                periodic, source_dir / "best_model.pth"),
        )

    # Only the resumed epoch's value is present — the legacy run's history
    # is unknown and must not be fabricated.  Its first entry belongs to the
    # RESUME epoch (1), not epoch 0, so the axis is not redrawn from 1.
    assert result["history"]["val_loss"] == [0.5]
    assert result["history_start_epoch"] == 1


def test_legacy_resume_offsets_history_to_true_epoch(tmp_path: Path) -> None:
    """A legacy resume must record its true history start epoch, not 0.

    Reviewer requirement: resuming a legacy checkpoint completed at epoch 2
    gives an unknown earlier history; the first recorded entry belongs to
    epoch 2, so ``history_start_epoch`` must be 2 — the curve must never be
    redrawn as if it began at epoch 1, and a second restart must not mistake
    the unknown prefix for a complete history.
    """
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", return_value=0.5),
    ):
        train(source, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_2.pth"
    legacy = load_artifact_bytes(periodic.read_bytes())
    for key in (
        "recovery_state_version", "epochs_without_improvement",
        "training_history", "history_start_epoch", "numpy_rng_state",
    ):
        legacy.pop(key, None)
    torch.save(legacy, periodic)

    resumed = _make_config(tmp_path, epochs=4, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "legacy_offset")
    resumed.checkpoint.resume_from = str(periodic)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", return_value=0.5),
    ):
        result = train(
            resumed, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(
                periodic, source_dir / "best_model.pth"),
        )
    # Recorded epochs are 2 and 3; entry 0 is epoch 2, not epoch 0.
    assert result["history"]["val_loss"] == [0.5, 0.5]
    assert result["history_start_epoch"] == 2
    # The saved checkpoint carries the same offset so a second restart is safe.
    final = load_artifact_bytes(
        (Path(resumed.checkpoint.output_dir) / "final_model.pth").read_bytes())
    assert final["history_start_epoch"] == 2


def test_double_resume_does_not_self_reject(tmp_path: Path) -> None:
    """Two consecutive resumes must not brick on the second restart.

    Reviewer requirement: a resumed checkpoint is itself resumed; its
    recorded offset+length must keep satisfying the reader, so recovery is
    repeatable rather than single-use.
    """
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", return_value=0.5),
    ):
        train(source, dataset_path=str(ds))
    source_dir = Path(source.checkpoint.output_dir)
    first_parent = source_dir / "epoch_1.pth"

    second = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    second.training.checkpoint_interval = 1
    second.checkpoint.output_dir = str(tmp_path / "second")
    second.checkpoint.resume_from = str(first_parent)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", return_value=0.5),
    ):
        train(
            second, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(
                first_parent, source_dir / "best_model.pth"),
        )
    second_dir = Path(second.checkpoint.output_dir)
    second_parent = second_dir / "epoch_2.pth"

    third = _make_config(tmp_path, epochs=4, warmup_epochs=0)
    third.checkpoint.output_dir = str(tmp_path / "third")
    third.checkpoint.resume_from = str(second_parent)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", return_value=0.5),
    ):
        result = train(
            third, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(
                second_parent, second_dir / "best_model.pth"),
        )
    # History accumulated across both resumes and stays positionally aligned.
    assert result["history"]["val_loss"] == [0.5, 0.5, 0.5, 0.5]
    assert result["history_start_epoch"] == 0


@pytest.mark.parametrize(
    "corrupt",
    [
        "epochs_without_improvement_negative",
        "training_history_keys",
        "history_non_scalar_entry",
        "recovery_state_version_bad",
        "history_start_epoch_missing",
        "history_start_epoch_mismatch",
        "history_start_epoch_out_of_range",
    ],
)
def test_corrupt_recovery_state_fails_closed(tmp_path: Path, corrupt: str) -> None:
    """A current-version payload with STRUCTURALLY malformed fields must not train.

    Non-finite history is deliberately NOT part of this set: the writer
    persists non-finite scalars AS-IS (a real recorded epoch), so a foreign
    non-finite value is preserved, never rejected — see
    ``test_foreign_nan_tail_is_preserved_verbatim``.  Only structural breakage
    (impossible counter, wrong keys, length mismatch, non-scalar entry) fails
    closed.
    """
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        train(source, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_1.pth"
    state = load_artifact_bytes(periodic.read_bytes())
    if corrupt == "epochs_without_improvement_negative":
        state["epochs_without_improvement"] = -1
    elif corrupt == "training_history_keys":
        state["training_history"] = {"val_loss": [], "bogus": []}
    elif corrupt == "recovery_state_version_bad":
        state["recovery_state_version"] = "two"
    elif corrupt == "history_start_epoch_missing":
        state.pop("history_start_epoch", None)
    elif corrupt == "history_start_epoch_mismatch":
        # 0-based start 1 + 1 recorded epoch != 1 completed epoch.
        state["history_start_epoch"] = 1
    elif corrupt == "history_start_epoch_out_of_range":
        state["history_start_epoch"] = -1
    else:
        state["training_history"] = {
            "val_loss": [{"not": "a scalar"}], "train_loss": [1.0],
        }
    torch.save(state, periodic)

    resumed = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "corrupt_resume")
    resumed.checkpoint.resume_from = str(periodic)
    with (
        mock.patch("scripts.train.train_one_epoch") as epoch_runner,
        mock.patch("scripts.train.validate") as validator,
    ):
        with pytest.raises(ValueError, match="fail closed"):
            train(
                resumed, dataset_path=str(ds),
                trusted_historical_checkpoint_sha256=_checkpoint_shas(
                    periodic, source_dir / "best_model.pth"),
        )
    epoch_runner.assert_not_called()
    validator.assert_not_called()


# ═════════════════════════════════════════════════════════════
# (A) Non-finite history: full-history semantics
# ═════════════════════════════════════════════════════════════

def test_nonfinite_train_with_finite_nonimproving_val_roundtrips(
    tmp_path: Path,
) -> None:
    """Non-finite train + finite non-improving val must be resumable (v2).

    Reviewer-confirmed defect: the v1 writer truncated BOTH series to a
    common finite prefix while persisting the unchanged counter.  A producer
    whose ``train_loss`` went non-finite at epoch 1 but whose ``val_loss``
    stayed finite and non-improving left a checkpoint with
    ``epochs_without_improvement > len(persisted val history)`` — the reader
    then rejected the checkpoint the pipeline itself had written.

    v2 fix: the FULL histories are persisted positionally (the non-finite
    train scalar AS-IS) and the counter is validated against the number of
    FINITE val epochs only.  This drives the producer path end to end:
    ``train_one_epoch`` returns NaN at epoch 1, ``validate`` returns
    [0.5, 0.6, 0.6] (epoch 1+ non-improving → counter 2).  The checkpoint
    must keep all three epochs of both series and the resume must succeed.
    """
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=3, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    source.training.early_stopping_patience = 0  # let all 3 epochs run
    with (
        mock.patch(
            "scripts.train.train_one_epoch",
            side_effect=[(1.0, {}), (float("nan"), {}), (1.2, {})],
        ),
        mock.patch("scripts.train.validate", side_effect=[0.5, 0.6, 0.6]),
    ):
        train(source, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_3.pth"
    saved = load_artifact_bytes(periodic.read_bytes())
    # The non-finite train scalar is a real recorded epoch, kept AS-IS, and
    # every later epoch keeps its position (the axis is NOT collapsed).
    train_hist = saved["training_history"]["train_loss"]
    assert train_hist[0] == 1.0 and math.isnan(train_hist[1])
    assert train_hist[2] == 1.2
    assert saved["training_history"]["val_loss"] == [0.5, 0.6, 0.6]
    assert saved["history_start_epoch"] == 0
    # The counter counts finite non-improving val epochs (0.6, 0.6) only.
    assert saved["epochs_without_improvement"] == 2

    # Resuming must SUCCEED and continue all three epochs.
    resumed = _make_config(tmp_path, epochs=5, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "resumed_nonfinite_train")
    resumed.checkpoint.resume_from = str(periodic)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.3, {})),
        mock.patch("scripts.train.validate", return_value=0.7),
    ):
        result = train(
            resumed, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(
                periodic, source_dir / "best_model.pth"),
        )
    assert result["history"]["val_loss"] == [0.5, 0.6, 0.6, 0.7, 0.7]
    assert math.isnan(result["history"]["train_loss"][1])
    assert result["history"]["train_loss"][-1] == 1.3


def test_nonfinite_val_tail_then_valid_metrics_is_preserved(tmp_path: Path) -> None:
    """A non-finite val epoch followed by valid metrics keeps the axis.

    Reviewer-confirmed defect: truncating at the first non-finite value
    dropped the SUBSEQUENT valid epochs and collapsed the epoch axis.  v2
    keeps every epoch positionally, so a NaN at epoch 1 does not erase the
    valid epoch 2.
    """
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=3, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.1, float("nan"), 0.2]),
    ):
        train(source, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_3.pth"
    saved = load_artifact_bytes(periodic.read_bytes())
    # The NaN keeps its slot; the valid epoch after it is NOT dropped.
    assert saved["training_history"]["val_loss"][0] == 0.1
    assert math.isnan(saved["training_history"]["val_loss"][1])
    assert saved["training_history"]["val_loss"][2] == 0.2
    assert saved["history_start_epoch"] == 0
    # A non-finite val epoch neither improves nor counts; only the finite
    # non-improving epoch 2 increments the counter.
    assert saved["epochs_without_improvement"] == 1

    resumed = _make_config(tmp_path, epochs=5, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "resumed_nan_val")
    resumed.checkpoint.resume_from = str(periodic)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", return_value=0.3),
    ):
        result = train(
            resumed, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(
                periodic, source_dir / "best_model.pth"),
        )
    got = result["history"]["val_loss"]
    assert got[0] == 0.1 and math.isnan(got[1]) and got[2:] == [0.2, 0.3, 0.3]


def test_interrupted_vs_uninterrupted_history_matches(tmp_path: Path) -> None:
    """Interrupted-and-resumed history equals the uninterrupted trajectory.

    The strongest correctness statement for recovery: run A continuously for
    3 epochs with one non-finite val epoch; run B does the same but is
    interrupted after epoch 1 and resumed.  Both must yield the SAME history
    (NaN included) and the same counter.
    """
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    vals = [0.5, float("nan"), 0.4, 0.4]

    uninterrupted = _make_config(tmp_path, epochs=4, warmup_epochs=0)
    uninterrupted.checkpoint.output_dir = str(tmp_path / "uninterrupted")
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=vals),
    ):
        full = train(uninterrupted, dataset_path=str(ds))

    source = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.5, float("nan")]),
    ):
        train(source, dataset_path=str(ds))
    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_2.pth"

    resumed = _make_config(tmp_path, epochs=4, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "resumed")
    resumed.checkpoint.resume_from = str(periodic)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.4, 0.4]),
    ):
        restart = train(
            resumed, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(
                periodic, source_dir / "best_model.pth"),
        )

    assert restart["history"]["train_loss"] == full["history"]["train_loss"]
    got = restart["history"]["val_loss"]
    want = full["history"]["val_loss"]
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert (math.isnan(g) and math.isnan(w)) or g == w


def test_foreign_nan_tail_is_preserved_verbatim(tmp_path: Path) -> None:
    """A foreign checkpoint's non-finite tail is preserved, not dropped."""
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        train(source, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_1.pth"
    state = load_artifact_bytes(periodic.read_bytes())
    # A foreign writer that persisted a non-finite value at epoch 1, having
    # completed two epochs.
    state["epoch"] = 1
    state["training_history"] = {
        "val_loss": [0.5, float("inf")], "train_loss": [1.0, 1.1],
    }
    state["history_start_epoch"] = 0
    state["epochs_without_improvement"] = 0
    torch.save(state, periodic)

    resumed = _make_config(tmp_path, epochs=4, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "foreign_nan")
    resumed.checkpoint.resume_from = str(periodic)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", return_value=0.7),
    ):
        result = train(
            resumed, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(
                periodic, source_dir / "best_model.pth"),
        )
    # The inf epoch keeps its slot (never coerced to 0, never dropped), and
    # the resumed epochs continue after it.
    assert result["history"]["val_loss"][0] == 0.5
    assert result["history"]["val_loss"][1] == float("inf")
    assert result["history"]["val_loss"][2:] == [0.7, 0.7]
    assert result["history"]["train_loss"][0] == 1.0


def test_stale_counter_exceeding_possible_val_epochs_fails_closed(
    tmp_path: Path,
) -> None:
    """A counter claiming more non-improving epochs than could have counted fails.

    The bound counts KNOWN-finite val epochs plus DECLARED GAPS (``None``, whose
    finiteness is unknown), because a gap epoch may have been finite and
    counted.  Only an explicit NaN/Inf is a known non-counting epoch.  Here all
    three val slots are explicitly non-finite (NaN/Inf/-Inf), so no epoch could
    have counted and a counter of 3 is impossible.
    """
    from scripts.train import _restore_recovery_state

    nan = float("nan")
    inf = float("inf")
    ckpt = {
        "recovery_state_version": 2,
        "epochs_without_improvement": 3,
        "training_history": {
            "val_loss": [nan, inf, -inf], "train_loss": [1.0, 1.1, 1.2],
        },
        "history_start_epoch": 0,
        "numpy_rng_state": _valid_encoded_rng(),
    }
    with pytest.raises(ValueError, match="could have counted"):
        _restore_recovery_state(ckpt, Path("x.pth"), resume_epoch=3)


def test_true_patience_across_unknown_gaps_is_restored(tmp_path: Path) -> None:
    """A REAL independent patience counter is NOT rejected by unknown gaps.

    Regression for the confirmed r5 blocker: ``resume_epoch=3`` with
    ``train=[1.0, 1.1, 1.2]`` and ``val=[0.1, None, None]`` and patience 2 is a
    legitimate current-schema checkpoint.  The two ``None`` slots are declared
    gaps whose finiteness is unknown, so the true counter (2) must be restored —
    a bound of "known-finite val epochs" alone (1) would wrongly reject it.
    """
    from scripts.train import _restore_recovery_state

    ckpt = {
        "recovery_state_version": 2,
        "epochs_without_improvement": 2,
        "training_history": {
            "val_loss": [0.1, None, None], "train_loss": [1.0, 1.1, 1.2],
        },
        "history_start_epoch": 0,
        "numpy_rng_state": _valid_encoded_rng(),
    }
    ewi, history, _, start = _restore_recovery_state(
        ckpt, Path("x.pth"), resume_epoch=3,
    )
    assert ewi == 2, f"true patience counter was not restored: {ewi}"
    assert history["val_loss"] == [0.1, None, None]
    assert start == 0


def test_history_start_epoch_mismatch_fails_closed(tmp_path: Path) -> None:
    """An offset + length that does not cover the completed epochs is damage."""
    from scripts.train import _restore_recovery_state

    ckpt = {
        "recovery_state_version": 2,
        "epochs_without_improvement": 0,
        "training_history": {"val_loss": [0.1, 0.2], "train_loss": [1.0, 1.1]},
        "history_start_epoch": 0,
        "numpy_rng_state": _valid_encoded_rng(),
    }
    with pytest.raises(ValueError, match="history covers epochs"):
        _restore_recovery_state(ckpt, Path("x.pth"), resume_epoch=3)


def test_unequal_series_lengths_fail_closed(tmp_path: Path) -> None:
    """A payload whose two series have different lengths breaks the epoch axis."""
    from scripts.train import _restore_recovery_state

    ckpt = {
        "recovery_state_version": 2,
        "epochs_without_improvement": 0,
        "training_history": {"val_loss": [0.1], "train_loss": [1.0, 1.1]},
        "history_start_epoch": 0,
        "numpy_rng_state": _valid_encoded_rng(),
    }
    with pytest.raises(ValueError, match="unequal"):
        _restore_recovery_state(ckpt, Path("x.pth"), resume_epoch=1)


def test_none_gap_keeps_its_epoch_slot(tmp_path: Path) -> None:
    """A foreign payload's ``None`` gap is preserved at its position.

    The trainer records both series every committed epoch, so this is the
    reader's tolerance for a foreign gap representation — the gap must
    occupy its epoch slot (not shorten the axis) and must not be coerced.
    """
    from scripts.train import _restore_recovery_state

    ckpt = {
        "recovery_state_version": 2,
        "epochs_without_improvement": 1,
        "training_history": {
            "val_loss": [0.1, None, 0.2], "train_loss": [1.0, 1.1, 1.2],
        },
        "history_start_epoch": 0,
        "numpy_rng_state": _valid_encoded_rng(),
    }
    ewi, history, _, start = _restore_recovery_state(
        ckpt, Path("x.pth"), resume_epoch=3,
    )
    assert ewi == 1
    assert start == 0
    assert history["val_loss"] == [0.1, None, 0.2]
    assert len(history["val_loss"]) == 3


# ═════════════════════════════════════════════════════════════
# NumPy RNG round-trip
# ═════════════════════════════════════════════════════════════

def test_numpy_rng_state_round_trips_through_restricted_decoder() -> None:
    from scripts.train import _decode_numpy_rng_state, _encode_numpy_rng_state

    np.random.seed(1234)
    np.random.rand(17)
    original = np.random.get_state()
    expected_next = np.random.rand(4)

    encoded = _encode_numpy_rng_state(original)
    # The encoded dict must survive the restricted artifact decoder intact.
    import io
    buffer = io.BytesIO()
    torch.save({"numpy_rng_state": encoded}, buffer)
    decoded = load_artifact_bytes(buffer.getvalue(), map_location="cpu")
    restored = _decode_numpy_rng_state(decoded["numpy_rng_state"])

    np.random.set_state(restored)
    np.testing.assert_array_equal(np.random.rand(4), expected_next)


def test_numpy_rng_state_round_trips_gaussian_cache_branch() -> None:
    """The cached-gaussian branch survives the restricted decoder.

    A Gaussian draw leaves ``has_gauss=1`` and a non-zero ``cached_gaussian``.
    If the encoder/decoder dropped or zeroed either field, the restored stream
    would silently produce DIFFERENT Gaussian draws even though the uniform
    stream (and the raw keys/pos) matched.  This is the branch the full-state
    comparison in the real chains must be able to see.
    """
    from scripts.train import _decode_numpy_rng_state, _encode_numpy_rng_state

    np.random.seed(2026)
    np.random.standard_normal(7)  # leaves the cache primed in general
    # Force a primed cache deterministically: an ODD number of Gaussian draws
    # leaves one value cached, so has_gauss == 1.
    while np.random.get_state()[3] != 1:
        np.random.standard_normal()
    original = np.random.get_state()
    assert original[3] == 1, "expected a primed Gaussian cache"
    expected_next = np.random.standard_normal(4)

    encoded = _encode_numpy_rng_state(original)
    decoded = _decode_numpy_rng_state(encoded)
    kind, keys, pos, has_gauss, cached = decoded
    assert has_gauss == 1
    assert cached == float(original[4])
    np.testing.assert_array_equal(keys, original[1])

    np.random.set_state(decoded)
    np.testing.assert_array_equal(np.random.standard_normal(4), expected_next)


@pytest.mark.parametrize("bad", [
    None, {}, {"bit_generator": "PCG64"},
    # Fractional pos must NOT be int()-coerced to 17 (review B-1).
    {"bit_generator": "MT19937", "keys": torch.zeros(624, dtype=torch.uint32),
     "pos": 17.5, "has_gauss": 0, "cached_gaussian": 0.0},
    # Boolean / string flags must NOT be coerced.
    {"bit_generator": "MT19937", "keys": torch.zeros(624, dtype=torch.uint32),
     "pos": True, "has_gauss": 0, "cached_gaussian": 0.0},
    {"bit_generator": "MT19937", "keys": torch.zeros(624, dtype=torch.uint32),
     "pos": 0, "has_gauss": "0", "cached_gaussian": 0.0},
    {"bit_generator": "MT19937", "keys": torch.zeros(624, dtype=torch.uint32),
     "pos": 0, "has_gauss": 0, "cached_gaussian": "0.0"},
    {"bit_generator": "MT19937", "keys": torch.zeros(624, dtype=torch.uint32),
     "pos": 0, "has_gauss": 0, "cached_gaussian": float("nan")},
    # Wrong key shape / dtype / generator.
    {"bit_generator": "MT19937", "keys": torch.zeros(3, dtype=torch.uint32),
     "pos": 0, "has_gauss": 0, "cached_gaussian": 0.0},
    {"bit_generator": "MT19937", "keys": torch.zeros(1, 624, dtype=torch.uint32),
     "pos": 0, "has_gauss": 0, "cached_gaussian": 0.0},
    {"bit_generator": "MT19937", "keys": torch.zeros(624, dtype=torch.int32),
     "pos": 0, "has_gauss": 0, "cached_gaussian": 0.0},
    {"bit_generator": "MT19937", "keys": torch.zeros(624, dtype=torch.uint32),
     "pos": -1, "has_gauss": 0, "cached_gaussian": 0.0},
    {"bit_generator": "MT19937", "keys": torch.zeros(624, dtype=torch.uint32),
     "pos": 625, "has_gauss": 0, "cached_gaussian": 0.0},
])
def test_decode_numpy_rng_state_rejects_malformed(bad: object) -> None:
    """A current-schema malformed RNG payload raises, never coerces."""
    from scripts.train import _decode_numpy_rng_state

    with pytest.raises(ValueError):
        _decode_numpy_rng_state(bad)


@pytest.mark.parametrize(
    "mutate",
    ["missing", "fractional_pos", "bool_flag", "string_flag",
     "wrong_generator", "wrong_shape"],
)
def test_current_schema_malformed_rng_fails_before_epoch(
    tmp_path: Path, mutate: str,
) -> None:
    """A current-version checkpoint with malformed RNG state must not train.

    Reviewer R2 / B-1: the previous behavior warned and resumed with a
    reseeded NumPy stream (fractional ``pos`` silently coerced to 17).
    The run must instead FAIL CLOSED before executing any epoch.
    """
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        train(source, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_1.pth"
    state = load_artifact_bytes(periodic.read_bytes())
    assert state["recovery_state_version"] == 2
    if mutate == "missing":
        state.pop("numpy_rng_state", None)
    elif mutate == "fractional_pos":
        state["numpy_rng_state"]["pos"] = 17.5
    elif mutate == "bool_flag":
        state["numpy_rng_state"]["has_gauss"] = True
    elif mutate == "string_flag":
        state["numpy_rng_state"]["cached_gaussian"] = "0.0"
    elif mutate == "wrong_generator":
        state["numpy_rng_state"] = {"bit_generator": "PCG64"}
    else:
        state["numpy_rng_state"]["keys"] = torch.zeros(1, 624, dtype=torch.uint32)
    torch.save(state, periodic)

    resumed = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / f"rng_{mutate}")
    resumed.checkpoint.resume_from = str(periodic)
    executed: list[int] = []

    def run_epoch(**kwargs: Any) -> Any:
        executed.append(kwargs["epoch"])
        return 1.0, {}

    with (
        mock.patch("scripts.train.train_one_epoch", side_effect=run_epoch),
        mock.patch("scripts.train.validate", return_value=0.5),
    ):
        with pytest.raises(ValueError, match="numpy_rng_state"):
            train(
                resumed, dataset_path=str(ds),
                trusted_historical_checkpoint_sha256=_checkpoint_shas(
                    periodic, source_dir / "best_model.pth"),
        )
    assert executed == [], (
        f"malformed current-schema RNG ({mutate}) executed epochs {executed}"
    )


# ═════════════════════════════════════════════════════════════
# Segment lineage records
# ═════════════════════════════════════════════════════════════

def test_segment_records_are_immutable_and_parent_bound(tmp_path: Path) -> None:
    from scripts.train import _write_segment_record

    checkpoint = tmp_path / "parent.pth"
    checkpoint.write_bytes(b"parent checkpoint bytes")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    parent = {"path": str(checkpoint.resolve()), "sha256": digest, "size_bytes": 23}

    config = _make_config(tmp_path, epochs=5, warmup_epochs=0)
    # The parent's controls live under peek["config"]["training"] (what
    # save_checkpoint wrote), NOT at the checkpoint top level.
    peek = {
        "dataset_path": "/data/x.pt",
        "config": {"training": {
            "num_epochs": 5, "random_seed": 42, "checkpoint_interval": 1,
        }},
    }
    out = tmp_path / "seg"
    first = _write_segment_record(
        out, parent=parent, config=config, parent_peek=peek,
        start_epoch=3, resumed=True, resume_path=checkpoint,
    )
    second = _write_segment_record(
        out, parent=None, config=config, parent_peek=None,
        start_epoch=0, resumed=False,
    )
    assert first.name == "segment_0000.json"
    assert second.name == "segment_0001.json"

    record0 = json.loads(first.read_text())
    record1 = json.loads(second.read_text())
    assert record0["segment_index"] == 0 and record1["segment_index"] == 1
    assert record0["parent_checkpoint"]["sha256"] == digest
    assert record0["starting_epoch"] == 3 and record0["resumed"] is True
    assert record1["parent_checkpoint"] is None and record1["resumed"] is False
    # Writing segment 1 must not have touched segment 0.
    assert json.loads(first.read_text()) == record0
    assert record0["provenance"]["original_num_epochs"] == 5
    assert record0["provenance"]["checkpoint_interval"] == 1
    # No orphaned temp files remain.
    assert not list(out.glob("*.tmp"))


def test_segment_records_parent_controls_not_active_config(tmp_path: Path) -> None:
    """Parent controls come from the parent's config, never the active run's.

    Reviewer-confirmed defect: the segment was seeded from the active config,
    so a resume that changed ``num_epochs``/``random_seed``/``checkpoint_interval``
    mislabelled the parent's lineage.  The parent's values must be recorded
    (and the active run's kept separately as ``current_segment_controls``).
    """
    from scripts.train import _write_segment_record

    checkpoint = tmp_path / "parent.pth"
    checkpoint.write_bytes(b"parent bytes")
    config = _make_config(tmp_path, epochs=99, warmup_epochs=0)
    config.training.random_seed = 7
    config.training.checkpoint_interval = 50
    config.training.checkpoint_interval = 1  # active reliable cadence
    config.training.num_workers = 0
    config.training.persistent_workers = False
    peek = {
        "config": {"training": {
            "num_epochs": 5, "random_seed": 42, "checkpoint_interval": 10,
            "num_workers": 4, "persistent_workers": True,
        }},
    }
    record_path = _write_segment_record(
        tmp_path / "seg", parent={"path": "p", "sha256": "0" * 64},
        config=config, parent_peek=peek, start_epoch=3, resumed=True,
        resume_path=checkpoint,
    )
    prov = json.loads(record_path.read_text())["provenance"]
    assert prov["original_num_epochs"] == 5
    assert prov["random_seed"] == 42
    assert prov["checkpoint_interval"] == 10
    # The active run's own controls are recorded separately, never confused
    # with the parent's.
    ctrl = prov["current_segment_controls"]
    assert ctrl["num_epochs"] == 99
    assert ctrl["random_seed"] == 7
    assert ctrl["checkpoint_interval"] == 1
    # Loader controls: the parent's REQUESTED values are recorded verbatim —
    # including a legitimate bool persistent_workers=True, which must not be
    # dropped to null by an int-only decoder.
    lc = ctrl["loader_controls"]
    assert lc["parent_num_workers"] == 4
    assert lc["parent_persistent_workers"] is True
    assert lc["requested_num_workers"] == 0
    assert lc["requested_persistent_workers"] is False
    # Reliable requirement is tied to the ACTIVE cadence (1), and is truthful.
    assert lc["reliable_mode"] is True
    assert lc["reliable_zero_worker_required"] is True
    # A persistent multi-worker resume IS a bitwise continuation (the
    # pre-warm fix), so that equivalence is ESTABLISHED; a resume that CHANGES
    # the worker count is NOT (the two consume the global stream differently).
    assert lc["worker_resume_equivalence_established"] is True
    assert lc["worker_count_change_equivalence_established"] is False


def test_segment_records_resolved_loader_workers(tmp_path: Path) -> None:
    """The segment records the ACTUAL resolved worker counts, not just the request.

    ``num_workers=-1`` auto-scales, so the request and the resolution differ;
    an audit must see what actually ran.
    """
    from scripts.train import _write_segment_record

    checkpoint = tmp_path / "parent.pth"
    checkpoint.write_bytes(b"parent bytes")
    config = _make_config(tmp_path, epochs=3, warmup_epochs=0)
    config.training.checkpoint_interval = 1
    config.training.num_workers = -1  # requested auto
    record_path = _write_segment_record(
        tmp_path / "seg", parent=None,
        config=config, parent_peek=None, start_epoch=0, resumed=False,
        resolved_workers={"num_workers": 4, "val_num_workers": 4},
    )
    lc = json.loads(record_path.read_text())["provenance"][
        "current_segment_controls"
    ]["loader_controls"]
    assert lc["requested_num_workers"] == -1
    assert lc["resolved_num_workers"] == 4
    assert lc["reliable_mode"] is True
    # A non-cadence-one segment makes no reliable claim even at zero workers.
    config.training.checkpoint_interval = 10
    config.training.num_workers = 0
    rec2 = _write_segment_record(
        tmp_path / "seg2", parent=None,
        config=config, parent_peek=None, start_epoch=0, resumed=False,
        resolved_workers={"num_workers": 0},
    )
    lc2 = json.loads(rec2.read_text())["provenance"][
        "current_segment_controls"
    ]["loader_controls"]
    assert lc2["reliable_mode"] is False
    assert lc2["reliable_zero_worker_required"] is False
    assert lc2["resolved_num_workers"] == 0


def test_segment_records_train_and_val_resolved_workers(tmp_path: Path) -> None:
    """The segment records the RESOLVED worker count for BOTH train and val.

    Train and val loaders can resolve differently, and a val loader can be
    absent.  The record must carry both actual counts (with an explicit ``None``
    when there is no val loader), never a collapsed single value.
    """
    from scripts.train import _write_segment_record

    config = _make_config(tmp_path, epochs=3, warmup_epochs=0)
    config.training.checkpoint_interval = 1
    config.training.num_workers = 0

    # Different resolved counts for train vs val.
    rec = _write_segment_record(
        tmp_path / "seg", parent=None,
        config=config, parent_peek=None, start_epoch=0, resumed=False,
        resolved_workers={"num_workers": 0, "val_num_workers": 7},
    )
    lc = json.loads(rec.read_text())["provenance"]["current_segment_controls"][
        "loader_controls"
    ]
    assert lc["resolved_num_workers"] == 0
    assert lc["resolved_val_num_workers"] == 7

    # No val loader -> explicit None, not a substituted 0.
    rec2 = _write_segment_record(
        tmp_path / "seg2", parent=None,
        config=config, parent_peek=None, start_epoch=0, resumed=False,
        resolved_workers={"num_workers": 3, "val_num_workers": None},
    )
    lc2 = json.loads(rec2.read_text())["provenance"]["current_segment_controls"][
        "loader_controls"
    ]
    assert lc2["resolved_num_workers"] == 3
    assert lc2["resolved_val_num_workers"] is None


def test_resolved_loader_workers_reads_actual_loaders(tmp_path: Path) -> None:
    """``_resolved_loader_workers`` reads the ACTUAL built loaders (auto-scaling)."""
    from scripts.train import _resolved_loader_workers

    class _Loader:
        def __init__(self, nw: int) -> None:
            self.num_workers = nw

    assert _resolved_loader_workers(_Loader(0), _Loader(7)) == {
        "num_workers": 0, "val_num_workers": 7,
    }
    assert _resolved_loader_workers(_Loader(2), None) == {
        "num_workers": 2, "val_num_workers": None,
    }


def test_segment_legacy_parent_controls_recorded_as_null(tmp_path: Path) -> None:
    """A parent without a recorded config reports unknown, not the active run's."""
    from scripts.train import _write_segment_record

    checkpoint = tmp_path / "legacy.pth"
    checkpoint.write_bytes(b"legacy bytes")
    config = _make_config(tmp_path, epochs=99, warmup_epochs=0)
    config.training.random_seed = 7
    record_path = _write_segment_record(
        tmp_path / "seg", parent={"path": "p", "sha256": "0" * 64},
        config=config, parent_peek={"dataset_path": "/d.pt"},
        start_epoch=1, resumed=True, resume_path=checkpoint,
    )
    prov = json.loads(record_path.read_text())["provenance"]
    assert prov["original_num_epochs"] is None
    assert prov["random_seed"] is None
    assert prov["checkpoint_interval"] is None
    # Not silently substituted from the active run.
    assert prov["current_segment_controls"]["num_epochs"] == 99


def test_segment_malformed_parent_controls_fail_closed(tmp_path: Path) -> None:
    """A present-but-malformed parent config.training fails closed."""
    from scripts.train import _write_segment_record

    checkpoint = tmp_path / "parent.pth"
    checkpoint.write_bytes(b"parent bytes")
    config = _make_config(tmp_path, epochs=5, warmup_epochs=0)
    with pytest.raises(ValueError, match="fail closed"):
        _write_segment_record(
            tmp_path / "seg", parent={"path": "p", "sha256": "0" * 64},
            config=config,
            parent_peek={"config": {"training": {"num_epochs": "many"}}},
            start_epoch=1, resumed=True, resume_path=checkpoint,
        )


def test_segment_provenance_fails_closed_without_parent_peek(tmp_path: Path) -> None:
    """A resume with no decoded parent controls must not be stamped with the
    active run's controls (which may differ from the parent's)."""
    from scripts.train import _write_segment_record

    config = _make_config(tmp_path, epochs=5, warmup_epochs=0)
    with pytest.raises(ValueError, match="fail closed"):
        _write_segment_record(
            tmp_path / "seg", parent={"path": "p", "sha256": "0" * 64},
            config=config, parent_peek=None, start_epoch=3, resumed=True,
        )


def test_segment_write_crash_leaves_no_complete_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash between temp-write and link must leave NO segment_XXXX.json.

    Reviewer B defect: the O_EXCL claim + plain write left a truncated final
    file that later runs treated as a completed segment.  With temp+link, a
    SIGKILL before the link leaves only an orphaned temp file; the final name
    is never created, so the next invocation claims index 0 cleanly.
    """
    from scripts.train import _write_segment_record

    config = _make_config(tmp_path, epochs=5, warmup_epochs=0)
    out = tmp_path / "seg"
    real_link = os.link

    def _crash_link(src: Any, dst: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated SIGKILL before atomic link")

    monkeypatch.setattr(os, "link", _crash_link)
    with pytest.raises(RuntimeError, match="simulated SIGKILL"):
        _write_segment_record(
            out, parent=None, config=config, parent_peek=None,
            start_epoch=0, resumed=False,
        )
    # No completed record masquerading as the segment.
    assert not list(out.glob("segment_*.json"))
    # The temp file is cleaned up (no orphaned slot either).
    assert not list(out.glob("*.tmp"))

    # A subsequent, healthy invocation claims index 0.
    monkeypatch.setattr(os, "link", real_link)
    path = _write_segment_record(
        out, parent=None, config=config, parent_peek=None,
        start_epoch=0, resumed=False,
    )
    assert path.name == "segment_0000.json"
    assert json.loads(path.read_text())["segment_index"] == 0


def test_segment_partial_temp_not_treated_as_complete(tmp_path: Path) -> None:
    """A leftover truncated temp file is not a segment and does not block index 0."""
    from scripts.train import _write_segment_record

    out = tmp_path / "seg"
    out.mkdir(parents=True)
    # Simulate an orphaned partial temp write from a killed process.
    (out / ".segment_999_123.tmp").write_bytes(b'{"segment_index": 0, "trunc')

    config = _make_config(tmp_path, epochs=5, warmup_epochs=0)
    path = _write_segment_record(
        out, parent=None, config=config, parent_peek=None,
        start_epoch=0, resumed=False,
    )
    assert path.name == "segment_0000.json"
    # The new record is complete and parseable despite the orphan.
    assert json.loads(path.read_text())["segment_index"] == 0


def test_train_writes_segment_bound_to_parent(tmp_path: Path) -> None:
    """A resume run records its segment bound to the parent's exact bytes."""
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        trainer.train(source, dataset_path=str(ds))
    source_dir = Path(source.checkpoint.output_dir)
    parent = source_dir / "epoch_1.pth"
    parent_bytes = parent.read_bytes()

    out_dir = tmp_path / "resumed_segment"
    resumed = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(out_dir)
    resumed.checkpoint.resume_from = str(parent)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", return_value=0.5),
    ):
        trainer.train(
            resumed, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(
                parent, source_dir / "best_model.pth"),
        )

    record = json.loads((out_dir / "segment_0000.json").read_text())
    assert record["resumed"] is True
    assert record["starting_epoch"] == 1
    assert record["parent_checkpoint"]["path"] == str(parent.resolve())
    assert record["parent_checkpoint"]["sha256"] == hashlib.sha256(
        parent_bytes).hexdigest()
    assert record["parent_checkpoint"]["size_bytes"] == len(parent_bytes)
    assert record["provenance"]["dataset_path"] == str(ds)


def test_segment_parent_bound_to_resumed_bytes_not_reread(tmp_path: Path) -> None:
    """Mutation/re-read guard: the segment binds to the LOADED bytes.

    Reviewer A/B defect (TOCTOU): the parent hash was computed from a second
    read of the mutable path, so swapping the file between load and segment
    write bound the segment to bytes that were never resumed from.  The fix
    passes the exact ``resume_payload`` into the identity helper.

    This test drives the PRODUCER: after the resume load, we overwrite the
    on-disk parent with DIFFERENT bytes.  The recorded hash must equal the
    bytes actually loaded (captured pre-mutation), not the new on-disk bytes.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        trainer.train(source, dataset_path=str(ds))
    source_dir = Path(source.checkpoint.output_dir)
    parent = source_dir / "epoch_1.pth"
    loaded_digest = hashlib.sha256(parent.read_bytes()).hexdigest()
    trusted = _checkpoint_shas(parent, source_dir / "best_model.pth")

    out_dir = tmp_path / "mutated_segment"
    resumed = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(out_dir)
    resumed.checkpoint.resume_from = str(parent)

    # After the run's resume load (which reads the parent), mutate the path.
    # The mutation is observable only if identity re-reads the path.
    real_read = Path.read_bytes
    real_write = Path.write_bytes
    mutated = {"done": False}

    def _mutating_read(self: Path) -> bytes:
        data = real_read(self)
        if self.resolve() == parent.resolve() and not mutated["done"]:
            # Rewrite the on-disk file to different bytes after it is read,
            # simulating a concurrent substitution between reads.
            real_write(self, data + b"# mutated")
            mutated["done"] = True
        return data

    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", return_value=0.5),
        mock.patch.object(Path, "read_bytes", _mutating_read),
    ):
        trainer.train(
            resumed, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=trusted,
        )

    record = json.loads((out_dir / "segment_0000.json").read_text())
    on_disk_digest = hashlib.sha256(real_read(parent)).hexdigest()
    assert record["parent_checkpoint"]["sha256"] == loaded_digest, (
        "segment parent hash was bound to a re-read, not the loaded bytes "
        f"(recorded {record['parent_checkpoint']['sha256']}, loaded "
        f"{loaded_digest}, on-disk {on_disk_digest})"
    )
    # Sanity: the mutation really did change the on-disk bytes.
    assert on_disk_digest != loaded_digest


def test_fresh_run_writes_parentless_segment(tmp_path: Path) -> None:
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    config = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        trainer.train(config, dataset_path=str(ds))

    record = json.loads(
        (Path(config.checkpoint.output_dir) / "segment_0000.json").read_text())
    assert record["resumed"] is False
    assert record["starting_epoch"] == 0
    assert record["parent_checkpoint"] is None


# ═════════════════════════════════════════════════════════════
# (E) Opt-in checkpoint_interval=1 profile
# ═════════════════════════════════════════════════════════════

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_epoch_recovery_profile_is_opt_in_cadence_and_zero_workers() -> None:
    """The opt-in profile differs from default in cadence + zero workers.

    Reliable recovery is the cadence-one regime scoped to zero-worker loading,
    so the profile must set cadence 1 AND ``num_workers=0``.  Its
    ``persistent_workers: false`` is INACTIVE at zero workers (the loader
    factory only forwards it when workers>0), i.e. documentation only, not a
    scientific control.  The default/formal cadence and worker policy are
    untouched.
    """
    from nsmor.config_parser import ExperimentConfig

    default_path = _REPO_ROOT / "config" / "default.yaml"
    profile_path = _REPO_ROOT / "config" / "epoch_recovery.yaml"
    assert default_path.exists(), default_path
    assert profile_path.exists(), (
        "the opt-in epoch-recovery profile was not delivered"
    )

    default_cfg = ExperimentConfig.from_yaml(default_path).to_dict()
    profile_cfg = ExperimentConfig.from_yaml(profile_path).to_dict()

    diffs = sorted(
        (section, key, default_cfg[section][key], profile_cfg[section][key])
        for section in default_cfg
        if isinstance(default_cfg[section], dict)
        for key in default_cfg[section]
        if default_cfg[section][key] != profile_cfg[section][key]
    )
    assert diffs == [
        ("training", "checkpoint_interval", 10, 1),
        ("training", "num_workers", -1, 0),
        ("training", "persistent_workers", True, False),
    ], f"unexpected profile diff vs default: {diffs}"

    profile = ExperimentConfig.from_yaml(profile_path)
    assert profile.training.checkpoint_interval == 1
    assert profile.training.num_workers == 0
    assert profile.training.persistent_workers is False
    # The default/formal cadence and worker policy are untouched.
    default = ExperimentConfig.from_yaml(default_path)
    assert default.training.checkpoint_interval == 10
    assert default.training.num_workers == -1
    assert default.training.persistent_workers is True


# ═════════════════════════════════════════════════════════════
# (B) Two-phase boundary resume keeps history, resets counter
# ═════════════════════════════════════════════════════════════

def test_two_phase_boundary_resume_keeps_history_resets_counter(tmp_path: Path) -> None:
    """A boundary-crossing resume must CONTINUE history but RESET patience.

    Reviewer A defect: history was discarded at the boundary, diverging from
    an uninterrupted trajectory (which appends across the boundary).  The
    in-loop transition resets only ``epochs_without_improvement``; history is
    never reset.  The resume path must mirror exactly that.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    # Source: phase1_epochs=3, epochs=3 → the source STOPS exactly at the
    # boundary and saves epoch_3.pth (start_epoch=3 == phase1_epochs).
    source = _make_config(tmp_path, epochs=3, warmup_epochs=0)
    source.training.checkpoint_interval = 3
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.5, 0.4, 0.3]),
    ):
        trainer.train(source, phase1_epochs=3, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    boundary_ckpt = source_dir / "epoch_3.pth"
    saved = load_artifact_bytes(boundary_ckpt.read_bytes())
    assert saved["training_phase"] == 1
    assert saved["training_history"]["val_loss"] == [0.5, 0.4, 0.3], saved["training_history"]

    # Resume landing exactly at the boundary (start_epoch == phase1_epochs).
    resumed = _make_config(tmp_path, epochs=5, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "boundary_resume")
    resumed.checkpoint.resume_from = str(boundary_ckpt)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.2, 0.2]),
    ):
        result = trainer.train(
            resumed, phase1_epochs=3, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(
                boundary_ckpt, source_dir / "best_model.pth"),
        )

    # History continues across the boundary (phase-1 values preserved).
    assert result["history"]["val_loss"] == [0.5, 0.4, 0.3, 0.2, 0.2], (
        f"phase-boundary resume dropped phase-1 history: "
        f"{result['history']['val_loss']}"
    )
    # The counter was reset at the boundary (phase-local patience).
    final = load_artifact_bytes(
        (Path(resumed.checkpoint.output_dir) / "final_model.pth").read_bytes())
    assert final["training_history"]["val_loss"] == [0.5, 0.4, 0.3, 0.2, 0.2]


def test_two_phase_uninterrupted_history_matches_resumed(tmp_path: Path) -> None:
    """The resumed boundary history must equal the uninterrupted trajectory's."""
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    vals = [0.5, 0.4, 0.3, 0.2, 0.2]

    # Uninterrupted run.
    uninterrupted = _make_config(tmp_path, epochs=5, warmup_epochs=0)
    uninterrupted.checkpoint.output_dir = str(tmp_path / "uninterrupted")
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=vals),
    ):
        full = trainer.train(uninterrupted, phase1_epochs=3, dataset_path=str(ds))

    # Interrupted run: stop at the boundary, resume.
    source = _make_config(tmp_path, epochs=3, warmup_epochs=0)
    source.training.checkpoint_interval = 3
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.5, 0.4, 0.3]),
    ):
        trainer.train(source, phase1_epochs=3, dataset_path=str(ds))
    source_dir = Path(source.checkpoint.output_dir)
    boundary_ckpt = source_dir / "epoch_3.pth"

    resumed = _make_config(tmp_path, epochs=5, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "resumed2")
    resumed.checkpoint.resume_from = str(boundary_ckpt)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", side_effect=[0.2, 0.2]),
    ):
        restart = trainer.train(
            resumed, phase1_epochs=3, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(
                boundary_ckpt, source_dir / "best_model.pth"),
        )
    assert restart["history"]["val_loss"] == full["history"]["val_loss"] == vals


# ═════════════════════════════════════════════════════════════
# Real (non-mocked) same-budget multi-resume equivalence
# ═════════════════════════════════════════════════════════════
#
# Reviewer R1 / criterion 3: the mocked-history tests cannot detect a missing
# model/optimizer/scheduler/RNG restoration because no model update or
# stochastic training occurs.  These tests run the REAL model, loss, backprop,
# optimizer, scheduler and loaders (dropout=0.1, sensory_noise_std>0 so the
# Torch RNG is consumed) on the small CPU fixture, with the SAME total epoch
# budget in every arm.  Interruption is a thrown exception after a saved
# completed epoch, so the scheduler horizon is not reduced.  Only validation
# is pinned to a finite constant (documented) to isolate the training
# trajectory from best-checkpoint selection.

class _TestInterruption(Exception):
    """Raised to simulate a SIGKILL/OOM at an epoch boundary."""


def _real_config(
    tmp_path: Path, *, epochs: int, name: str, patience: int = 0,
) -> Any:
    """A small CPU config that actually consumes the Torch RNG.

    Dropout and sensory noise are ON, so the model stream advances every
    epoch; ``num_workers=0`` keeps the loader in the supported recovery scope.
    ``patience`` defaults to 0 (no early stop) so a chain can run its full
    budget; tests that must observe the patience counter set it explicitly.
    """
    config = _make_config(tmp_path, epochs=epochs, warmup_epochs=0)
    config.model.dropout = 0.1
    config.model.sensory_noise_std = 0.01
    config.training.num_workers = 0
    config.training.persistent_workers = False
    config.training.checkpoint_interval = 1
    config.training.early_stopping_patience = patience
    config.checkpoint.output_dir = str(tmp_path / name)
    return config


def _load_state(path: Path) -> dict:
    return load_artifact_bytes(path.read_bytes(), map_location="cpu")


def _existing_shas(*paths: Path) -> tuple:
    """Trust shas of the paths that actually exist.

    A phase-1->2 boundary transition unlinks phase-1 ``best_model.pth``
    before the first phase-2 epoch, so a boundary interrupt may legitimately
    have no best checkpoint to trust.
    """
    return _checkpoint_shas(*[p for p in paths if p.exists()])


def _rewrite_parent_training_control(
    parent: Path, key: str, value: Any,
) -> None:
    """Rewrite ONE ``config.training`` control in an already-saved parent.

    Used only to synthesise malformed / unknown-budget parents: the rest of the
    checkpoint is untouched, so the budget guard is the only variable.  The
    decoded state dict is written back whole (the same shape the legacy-parent
    test already writes), so the restricted reader still accepts it.
    """
    state = _load_state(parent)
    state["config"]["training"][key] = value
    torch.save(state, parent)


def _rewrite_parent_diagnostics(
    parent: Path,
    *,
    top: Optional[Dict[str, Any]] = None,
    history_tail: Optional[Dict[str, Any]] = None,
) -> None:
    """Rewrite the parent's terminal diagnostics for the two sources.

    ``top`` sets (or, with value ``None``, removes) the top-level
    ``train_loss`` / ``val_loss`` scalars ``save_checkpoint`` wrote;
    ``history_tail`` sets the LAST position of ``training_history``.  The rest
    of the checkpoint is untouched, so the terminal-diagnostic selection is the
    only variable under test.
    """
    state = _load_state(parent)
    for key, value in (top or {}).items():
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
    for key, value in (history_tail or {}).items():
        series = state["training_history"][key]
        assert series, f"history {key} is empty; fixture invalid"
        series[-1] = value
    torch.save(state, parent)


def _assert_model_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys(), "model state-dict key sets differ"
    for key in a:
        assert torch.equal(a[key], b[key]), f"model parameter {key} differs"


def _assert_optimizer_equal(a: dict, b: dict) -> None:
    assert a["state"].keys() == b["state"].keys(), "optimizer param ids differ"
    for pid in a["state"]:
        sa, sb = a["state"][pid], b["state"][pid]
        assert sa.keys() == sb.keys(), f"optimizer state keys differ for {pid}"
        for key in sa:
            va, vb = sa[key], sb[key]
            if isinstance(va, torch.Tensor):
                assert torch.equal(va, vb), f"optimizer {pid}.{key} differs"
            else:
                assert va == vb, f"optimizer {pid}.{key} differs"
    assert len(a["param_groups"]) == len(b["param_groups"])
    for ga, gb in zip(a["param_groups"], b["param_groups"]):
        assert ga.keys() == gb.keys()
        for key in ga:
            if key == "params":
                assert ga[key] == gb[key], "optimizer group membership differs"
            else:
                assert ga[key] == gb[key], f"optimizer group {key} differs"


def _sched_state(ckpt: dict) -> dict:
    return {k: v for k, v in ckpt["scheduler_state_dict"].items()}


def _final_ckpt(config: Any) -> dict:
    return _load_state(Path(config.checkpoint.output_dir) / "final_model.pth")


def _numpy_state_and_streams(ckpt: dict, n: int = 8) -> dict:
    """Decode the checkpoint's NumPy stream and draw its NEXT outputs.

    Proves the stream is not merely present but continues to the same values a
    restart would produce.  Uses the production decoder (never a test-local
    re-implementation) so the schema under test is the one train() restores.

    Compares the FULL decoded MT19937 state — the 624-word ``keys``, ``pos``,
    ``has_gauss`` AND ``cached_gaussian`` — not just the uniform stream.  A
    state with identical keys/pos but a different ``has_gauss``/``cached``
    (which a uniform-only comparison cannot see) would pass while producing
    different Gaussian draws, so both continuations are compared: the uniform
    stream AND the Gaussian stream (the latter forces the cached-gaussian
    branch).  ``keys`` is returned as bytes for a stable equality comparison.
    """
    from scripts.train import _decode_numpy_rng_state

    kind, keys, pos, has_gauss, cached = _decode_numpy_rng_state(
        ckpt["numpy_rng_state"],
    )
    saved = np.random.get_state()
    try:
        np.random.set_state((kind, keys, pos, has_gauss, cached))
        uniform = np.random.random_sample(n).tolist()
        np.random.set_state((kind, keys, pos, has_gauss, cached))
        gaussian = np.random.standard_normal(n).tolist()
    finally:
        np.random.set_state(saved)
    return {
        "kind": kind,
        "keys": keys.tobytes(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
        "uniform": uniform,
        "gaussian": gaussian,
    }


def _epoch_with_numpy_draw(
    real_epoch: Any, cut: Optional[int] = None,
) -> Any:
    """Wrap the REAL ``train_one_epoch`` and exercise the global NumPy stream.

    The fixture's anchor-aligned crop does not itself draw from the global
    NumPy stream (the legacy random-crop path is what did), so the recovery
    seam's NumPy restore would otherwise be untested by a real trajectory.
    This wrapper calls the REAL ``train_one_epoch`` — no training is mocked —
    and then performs one supported per-epoch stochastic draw, exactly the
    consumption the seam exists to preserve.  Drawing here (before the loop
    writes the checkpoint) means every saved checkpoint carries the
    post-draw stream, so an unrestored resume produces different subsequent
    draws and the equality assertions fail.

    Both a uniform draw AND a Gaussian draw are taken so the cached-gaussian
    branch (``has_gauss``/``cached_gaussian``) is actually EXERCISED by the
    real chain — a uniform-only consumption leaves ``has_gauss`` at 0 and the
    cached branch untested.  The Gaussian count per epoch is ODD, so the
    legacy Box-Muller cache TOGGLES: it holds one value, an odd draw leaves
    ``has_gauss == 1`` and an even draw clears it.  Successive epochs therefore
    alternate the flag rather than pinning it, and an ODD total epoch count
    leaves the final checkpoint's cache primed.
    """
    def hook(**kwargs: Any) -> Any:
        if cut is not None and kwargs["epoch"] >= cut:
            raise _TestInterruption()
        out = real_epoch(**kwargs)
        np.random.random_sample(8)
        np.random.standard_normal(7)
        return out
    return hook


def _run_full_real(
    tmp_path: Path, ds: Path, *, epochs: int, name: str,
) -> Any:
    """Run ``epochs`` real epochs uninterrupted; return (config, result)."""
    from scripts import train as trainer

    config = _real_config(tmp_path, epochs=epochs, name=name)
    real_epoch = trainer.train_one_epoch
    with (
        mock.patch.object(
            trainer, "train_one_epoch",
            side_effect=_epoch_with_numpy_draw(real_epoch),
        ),
        mock.patch.object(trainer, "validate", return_value=0.5)

    ):
        result = trainer.train(config, dataset_path=str(ds))
    return config, result


def _run_chain_real(
    tmp_path: Path, ds: Path, *, total_epochs: int, cuts: Any,
    name_prefix: str, phase1_epochs: Optional[int] = None, patience: int = 0,
) -> Any:
    """Run ``total_epochs`` real epochs across ``cuts`` interruptions.

    Every segment keeps ``num_epochs=total_epochs`` so the scheduler horizon
    (T_max) is IDENTICAL in all arms — a shorter interrupted budget would
    change the cosine and confound the comparison.  ``cuts`` holds the 1-based
    epoch counts at which to interrupt: epoch ``cut`` completes and is saved
    as ``epoch_{cut}.pth``, then the run raises at the start of epoch ``cut+1``
    (a completed-epoch boundary, as SIGKILL/OOM would leave it).  Each
    interrupted run resumes from that saved checkpoint into a fresh segment.

    ``phase1_epochs`` is threaded to every segment (two-phase Hybrid Funnel).
    ``patience`` is the early-stop horizon for every segment, so a chain that
    must NOT stop early on validation must set the same value the uninterrupted
    arm uses.

    Returns ``(final_config, result, segment_configs)``.
    """
    from scripts import train as trainer

    segments = []
    resume_from = None
    resume_shas = None
    for i, cut in enumerate(cuts):
        config = _real_config(
            tmp_path, epochs=total_epochs, name=f"{name_prefix}_seg{i}",
            patience=patience,
        )
        if resume_from is not None:
            config.checkpoint.resume_from = str(resume_from)
        real_epoch = trainer.train_one_epoch

        with (
            mock.patch.object(
                trainer, "train_one_epoch",
                side_effect=_epoch_with_numpy_draw(real_epoch, cut=cut),
            ),
            mock.patch.object(trainer, "validate", return_value=0.5)

        ):
            try:
                trainer.train(
                    config, phase1_epochs=phase1_epochs, dataset_path=str(ds),
                    trusted_historical_checkpoint_sha256=resume_shas,
                )
            except _TestInterruption:
                pass
        seg_dir = Path(config.checkpoint.output_dir)
        resume_from = seg_dir / f"epoch_{cut}.pth"
        resume_shas = _existing_shas(resume_from, seg_dir / "best_model.pth")
        segments.append(config)

    final = _real_config(
        tmp_path, epochs=total_epochs, name=f"{name_prefix}_final", patience=patience,
    )
    final.checkpoint.resume_from = str(resume_from)
    real_epoch = trainer.train_one_epoch
    with (
        mock.patch.object(
            trainer, "train_one_epoch",
            side_effect=_epoch_with_numpy_draw(real_epoch),
        ),
        mock.patch.object(trainer, "validate", return_value=0.5)

    ):
        result = trainer.train(
            final, phase1_epochs=phase1_epochs, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=resume_shas,
        )
    return final, result, segments


def _assert_full_equivalence(
    full: dict, chain: dict, *, expect_start_epoch: int = 0,
) -> None:
    """Compare the complete continuation state of two checkpoints.

    Model, optimizer (state + groups), scheduler, Torch RNG, patience,
    epoch-positioned history and history origin.  This is the state a resume
    must restore for the trajectory to be identical.
    """
    _assert_model_equal(full["model_state_dict"], chain["model_state_dict"])
    _assert_optimizer_equal(
        full["optimizer_state_dict"], chain["optimizer_state_dict"],
    )
    assert _sched_state(full) == _sched_state(chain), "scheduler state differs"
    assert torch.equal(full["rng_state"], chain["rng_state"]), "Torch RNG differs"
    # NumPy stream: the FULL decoded MT19937 state must match (keys/pos/
    # has_gauss/cached_gaussian), and BOTH the uniform and Gaussian
    # continuations must agree.  Comparing only the uniform stream would let a
    # different cached_gaussian (same keys/pos, different has_gauss/cached)
    # pass while the Gaussian draws diverge.
    full_np = _numpy_state_and_streams(full)
    chain_np = _numpy_state_and_streams(chain)
    for field in ("kind", "keys", "pos", "has_gauss", "cached_gaussian"):
        assert full_np[field] == chain_np[field], (
            f"NumPy decoded state field {field!r} differs "
            f"({full_np[field]!r} != {chain_np[field]!r})"
        )
    assert full_np["uniform"] == chain_np["uniform"], (
        "NumPy uniform continuation differs"
    )
    assert full_np["gaussian"] == chain_np["gaussian"], (
        "NumPy Gaussian continuation differs"
    )
    assert (
        full["epochs_without_improvement"] == chain["epochs_without_improvement"]
    ), "patience counter differs"
    assert full["training_history"] == chain["training_history"], "history differs"
    assert (
        full["history_start_epoch"] == chain["history_start_epoch"] == expect_start_epoch
    ), "history origin differs"


def test_real_multiresume_single_phase_full_state_matches(tmp_path: Path) -> None:
    """Same-budget real chain with TWO resumes == uninterrupted, full state.

    Compares complete model state, optimizer state+groups, scheduler state,
    Torch RNG bytes, patience and epoch-positioned history after the same
    TOTAL epoch budget.  This is the sensitivity the mocked tests lack.  The
    total is ODD so the per-epoch odd Gaussian draw leaves the Box-Muller cache
    primed in the final checkpoint, making the cached-gaussian branch live.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    full_cfg, _ = _run_full_real(tmp_path, ds, epochs=5, name="full")
    chain_cfg, _, _ = _run_chain_real(
        tmp_path, ds, total_epochs=5, cuts=[1, 3], name_prefix="chain",
    )

    full = _final_ckpt(full_cfg)
    # The real chain's per-epoch Gaussian draw must leave a PRIMED cache, so the
    # cached-gaussian branch is actually covered (not merely declared).
    assert _numpy_state_and_streams(full)["has_gauss"] == 1, (
        "the real chain did not leave a primed Gaussian cache; the cached "
        "branch would be untested"
    )
    _assert_full_equivalence(full, _final_ckpt(chain_cfg))


@pytest.mark.parametrize(
    "phase1_epochs,label,cuts",
    [
        (2, "boundary", [2, 4]),      # both resumes land on / cross the boundary
        (1, "established", [2, 4]),   # both resumes inside established phase 2
    ],
)
def test_real_multiresume_two_phase_full_state_matches(
    tmp_path: Path, phase1_epochs: int, label: str, cuts: list[int],
) -> None:
    """Two-phase real continuation equality across TWO resumes.

    ``label=boundary`` interrupts so the first resume lands exactly at the
    phase boundary (start_epoch == phase1_epochs); ``label=established``
    interrupts inside phase 2.  Both arms share the same total budget,
    ``phase1_epochs`` and patience, and both resumes are compared on the full
    state (model/optimizer/scheduler/Torch RNG/patience/history/origin).
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    total = 5
    torch.manual_seed(0)

    full_cfg = _real_config(tmp_path, epochs=total, name=f"tp_full_{label}")
    real_epoch = trainer.train_one_epoch
    with (
        mock.patch.object(
            trainer, "train_one_epoch",
            side_effect=_epoch_with_numpy_draw(real_epoch),
        ),
        mock.patch.object(trainer, "validate", return_value=0.5)

    ):
        trainer.train(full_cfg, phase1_epochs=phase1_epochs, dataset_path=str(ds))

    chain_cfg, _, _ = _run_chain_real(
        tmp_path, ds, total_epochs=total, cuts=cuts,
        name_prefix=f"tp_chain_{label}", phase1_epochs=phase1_epochs,
    )
    _assert_full_equivalence(_final_ckpt(full_cfg), _final_ckpt(chain_cfg))


def test_numpy_state_comparison_detects_cached_gaussian_only_difference() -> None:
    """Negative control: a cached-gaussian-only difference is CAUGHT.

    Two states with the SAME keys and pos but different ``has_gauss`` /
    ``cached_gaussian`` produce identical uniform draws yet different Gaussian
    draws.  A uniform-only comparison would pass; the full-state comparison
    (and the Gaussian continuation) must fail.  This proves the fix-2
    comparison is sensitive to the branch it was added to cover.
    """
    from scripts.train import _decode_numpy_rng_state, _encode_numpy_rng_state

    np.random.seed(99)
    np.random.random_sample(5)
    while np.random.get_state()[3] != 1:
        np.random.standard_normal()
    primed = np.random.get_state()

    # Same keys/pos, but the cache cleared and a different cached value.
    cleared = (primed[0], primed[1], primed[2], 0, 0.0)
    other_cached = (primed[0], primed[1], primed[2], 1, primed[4] + 1.0)

    a = {"numpy_rng_state": _encode_numpy_rng_state(primed)}
    b = {"numpy_rng_state": _encode_numpy_rng_state(cleared)}
    c = {"numpy_rng_state": _encode_numpy_rng_state(other_cached)}

    sa = _numpy_state_and_streams(a)
    sb = _numpy_state_and_streams(b)
    sc = _numpy_state_and_streams(c)

    # Uniform continuation is identical across all three (keys/pos match) ...
    assert sa["keys"] == sb["keys"] == sc["keys"]
    assert sa["pos"] == sb["pos"] == sc["pos"]
    assert sa["uniform"] == sb["uniform"] == sc["uniform"]
    # ... so a uniform-only check would pass, but the full-state check must not.
    assert sa["has_gauss"] != sb["has_gauss"]
    assert sa["cached_gaussian"] != sc["cached_gaussian"]
    assert sa["gaussian"] != sb["gaussian"], (
        "cleared cache must change the Gaussian continuation"
    )
    assert sa["gaussian"] != sc["gaussian"], (
        "different cached value must change the Gaussian continuation"
    )


def test_real_legacy_origin_second_restart_preserves_origin(tmp_path: Path) -> None:
    """A legacy nonzero-epoch origin survives a SECOND restart (no invention).

    Legacy parent (no recovery stamp) completes epoch 2 -> resume records
    history starting at epoch 2 -> that new checkpoint is resumed AGAIN and
    must still report history_start_epoch == 2, with the legacy prehistory
    absent (never fabricated).
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    source = _real_config(tmp_path, epochs=2, name="legacy_src")
    with (
        mock.patch.object(trainer, "validate", return_value=0.5)

    ):
        trainer.train(source, dataset_path=str(ds))
    src_dir = Path(source.checkpoint.output_dir)
    periodic = src_dir / "epoch_2.pth"
    legacy = _load_state(periodic)
    for key in (
        "recovery_state_version", "epochs_without_improvement",
        "training_history", "history_start_epoch", "numpy_rng_state",
    ):
        legacy.pop(key, None)
    torch.save(legacy, periodic)

    first = _real_config(tmp_path, epochs=3, name="legacy_r1")
    first.checkpoint.resume_from = str(periodic)
    with (
        mock.patch.object(trainer, "validate", return_value=0.5)

    ):
        r1 = trainer.train(
            first, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                periodic, src_dir / "best_model.pth"),
        )
    assert r1["history_start_epoch"] == 2, "legacy origin not at resume epoch"
    first_dir = Path(first.checkpoint.output_dir)
    r1_ckpt = first_dir / "epoch_3.pth"
    r1_state = _load_state(r1_ckpt)
    assert r1_state["history_start_epoch"] == 2
    assert len(r1_state["training_history"]["val_loss"]) == 1  # epoch 2 only

    second = _real_config(tmp_path, epochs=4, name="legacy_r2")
    second.checkpoint.resume_from = str(r1_ckpt)
    with (
        mock.patch.object(trainer, "validate", return_value=0.5)

    ):
        r2 = trainer.train(
            second, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                r1_ckpt, first_dir / "best_model.pth"),
        )
    assert r2["history_start_epoch"] == 2, (
        f"second restart lost the legacy origin: {r2['history_start_epoch']}"
    )
    # Two recorded epochs (2 and 3); the unknown prehistory stays absent.
    assert len(r2["history"]["val_loss"]) == 2


def test_real_cadence1_workers_rejected_before_epoch(tmp_path: Path) -> None:
    """A cadence-one run with workers>0 is refused before any epoch executes.

    Covers BOTH the fresh reliable run and the reliable resume: the reliable
    requirement is tied to ``checkpoint_interval==1``, not to ``--resume``.
    The guard must fire before ``train_one_epoch`` is ever called.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    source = _real_config(tmp_path, epochs=1, name="wk_src")
    with (
        mock.patch.object(trainer, "validate", return_value=0.5)

    ):
        trainer.train(source, dataset_path=str(ds))
    src_dir = Path(source.checkpoint.output_dir)
    periodic = src_dir / "epoch_1.pth"

    def _run(config: Any, trusted: Any = None) -> Any:
        executed: list[int] = []

        def hook(**kwargs: Any) -> Any:
            executed.append(kwargs["epoch"])
            return 1.0, {}

        with (
            mock.patch.object(trainer, "train_one_epoch", side_effect=hook),
            mock.patch.object(trainer, "validate", return_value=0.5),
        ):
            with pytest.raises(ValueError, match="zero-worker"):
                trainer.train(
                    config, dataset_path=str(ds),
                    trusted_historical_checkpoint_sha256=trusted,
                )
        assert executed == [], f"refused cadence-one run executed {executed}"

    # (a) FRESH cadence-one run with workers>0 must be refused.
    fresh = _real_config(tmp_path, epochs=1, name="wk_fresh")
    fresh.training.num_workers = 1
    fresh.training.persistent_workers = True
    _run(fresh)

    # (b) RESUMED cadence-one run with workers>0 must be refused.
    resumed = _real_config(tmp_path, epochs=2, name="wk_resume")
    resumed.training.num_workers = 1
    resumed.training.persistent_workers = True
    resumed.checkpoint.resume_from = str(periodic)
    _run(resumed, _existing_shas(periodic, src_dir / "best_model.pth"))


def test_real_cadence10_workers_allowed_and_compatible(tmp_path: Path) -> None:
    """The prior interval-10 behavior is preserved: workers>0 still runs.

    A non-reliable cadence makes no exact-continuation claim, so the guard must
    NOT fire and the run must complete normally (fresh and resumed).
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    source = _real_config(tmp_path, epochs=1, name="c10_src")
    source.training.checkpoint_interval = 10
    source.training.num_workers = 1
    source.training.persistent_workers = True
    with (
        mock.patch.object(trainer, "validate", return_value=0.5)

    ):
        result = trainer.train(source, dataset_path=str(ds))
    assert result is not None
    src_dir = Path(source.checkpoint.output_dir)

    # Resume a cadence-10 parent at cadence 10 with workers>0: still allowed.
    # (cadence 10 writes no epoch_1.pth, so resume from the segment's last
    # complete-epoch checkpoint, final_model.pth — best_model.pth is NOT a
    # resume source and is refused by _require_resume_source_is_newest_complete_epoch.)
    periodic = src_dir / "final_model.pth"
    assert periodic.exists(), "the source segment must write final_model.pth"
    resumed = _real_config(tmp_path, epochs=2, name="c10_resume")
    resumed.training.checkpoint_interval = 10
    resumed.training.num_workers = 1
    resumed.training.persistent_workers = True
    resumed.checkpoint.resume_from = str(periodic)
    with (
        mock.patch.object(trainer, "validate", return_value=0.5)

    ):
        result = trainer.train(
            resumed, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                periodic, src_dir / "best_model.pth"),
        )
    assert result is not None


# ═════════════════════════════════════════════════════════════
# No-validation execution: the epoch axis must stay complete
# ═════════════════════════════════════════════════════════════
#
# ``train``/``build_dataloaders`` accept an optional validation split, so a
# run with no validation loader is a supported invocation.  Such a run still
# EXECUTES epochs, so each executed epoch must occupy one position on the
# shared train/val epoch axis.  Recording nothing for the unobserved
# validation diagnostic makes the two series unequal, which the current-schema
# reader correctly rejects as a broken axis — so the run cannot resume from its
# own checkpoint.  These tests run the REAL model/forward/backward/optimizer/
# scheduler with only the loader omission and the plot controlled.

def _no_val_epoch(
    real_epoch: Any, cut: Optional[int] = None,
) -> Any:
    """Real ``train_one_epoch`` plus the same supported NumPy consumption."""
    def hook(**kwargs: Any) -> Any:
        if cut is not None and kwargs["epoch"] >= cut:
            raise _TestInterruption()
        out = real_epoch(**kwargs)
        np.random.random_sample(8)
        np.random.standard_normal(7)
        return out
    return hook


def _run_no_val_real(
    tmp_path: Path, ds: Path, *, epochs: int, name: str,
    cut: Optional[int] = None,
) -> Any:
    """Run real epochs with NO validation loader; return (config, result)."""
    from scripts import train as trainer

    config = _real_config(tmp_path, epochs=epochs, name=name)
    train_loader, _val = trainer.build_dataloaders(config, dataset_path=str(ds))
    assert _val is not None, "fixture expects a real val loader to drop"
    real_epoch = trainer.train_one_epoch
    with (
        mock.patch.object(trainer, "build_dataloaders",
                          return_value=(train_loader, None)),
        mock.patch.object(trainer, "plot_loss_curve",
                          return_value=tmp_path / "unused.png"),
        mock.patch.object(
            trainer, "train_one_epoch",
            side_effect=_no_val_epoch(real_epoch, cut=cut),
        ),
    ):
        try:
            result = trainer.train(config, dataset_path=str(ds))
        except _TestInterruption:
            result = None
    return config, result


def test_no_validation_run_history_axis_is_complete(tmp_path: Path) -> None:
    """Every executed epoch occupies a val position (``None``) with no loader.

    The unobserved diagnostic is recorded as ``None`` — not a fabricated finite
    value and not a claimed NaN/Inf — so the shared epoch axis stays complete
    and patience does not advance (no best checkpoint is selected).
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)
    config, result = _run_no_val_real(tmp_path, ds, epochs=2, name="noval_axis")

    ckpt = _load_state(Path(config.checkpoint.output_dir) / "epoch_2.pth")
    hist = ckpt["training_history"]
    assert len(hist["train_loss"]) == 2, hist
    assert hist["val_loss"] == [None, None], hist
    assert ckpt["epochs_without_improvement"] == 0, "unvalidated epochs counted"
    assert ckpt["history_start_epoch"] == 0
    assert not (Path(config.checkpoint.output_dir) / "best_model.pth").exists(), (
        "no epoch had a finite validation loss, so no best model may be chosen"
    )
    assert result is not None and result["history"]["val_loss"] == [None, None]


def test_no_validation_checkpoint_is_accepted_by_its_own_reader(
    tmp_path: Path,
) -> None:
    """A no-validation checkpoint must not be rejected by the reader."""
    from scripts.train import _restore_recovery_state

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)
    config, _ = _run_no_val_real(tmp_path, ds, epochs=2, name="noval_reader")
    ckpt_path = Path(config.checkpoint.output_dir) / "epoch_2.pth"
    ckpt = _load_state(ckpt_path)
    ewi, history, np_state, start = _restore_recovery_state(
        ckpt, ckpt_path, resume_epoch=2,
    )
    assert ewi == 0
    assert history["val_loss"] == [None, None]
    assert history["train_loss"] == ckpt["training_history"]["train_loss"]
    assert start == 0
    assert np_state is not None, "no-validation run still records its RNG stream"


def test_no_validation_same_budget_chain_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    """No-validation: two fresh-output resumes == uninterrupted, full state.

    Same total epoch budget in every arm.  The first arm runs straight
    through; the second is interrupted at completed epoch 1 and at completed
    epoch 3, each time resuming from the saved boundary checkpoint into a fresh
    segment.  The final states must be identical (model, optimizer state and
    groups, scheduler, Torch RNG, full NumPy state and both continuations,
    patience, history, origin).
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    total = 5
    torch.manual_seed(0)

    full_cfg, _ = _run_no_val_real(tmp_path, ds, epochs=total, name="noval_full")
    assert _final_ckpt(full_cfg)["history_start_epoch"] == 0

    # Interrupted chain: cut at epoch 1, resume; cut at epoch 3, resume.
    seg0_cfg, _ = _run_no_val_real(
        tmp_path, ds, epochs=total, name="noval_seg0", cut=1,
    )
    seg0_dir = Path(seg0_cfg.checkpoint.output_dir)
    parent = seg0_dir / "epoch_1.pth"
    assert parent.exists(), "boundary checkpoint was not saved"

    real_epoch = trainer.train_one_epoch
    seg1_cfg = _real_config(tmp_path, epochs=total, name="noval_seg1")
    seg1_cfg.checkpoint.resume_from = str(parent)
    train_loader, _val = trainer.build_dataloaders(seg1_cfg, dataset_path=str(ds))
    with (
        mock.patch.object(trainer, "build_dataloaders",
                          return_value=(train_loader, None)),
        mock.patch.object(trainer, "plot_loss_curve",
                          return_value=tmp_path / "unused.png"),
        mock.patch.object(
            trainer, "train_one_epoch",
            side_effect=_no_val_epoch(real_epoch, cut=3),
        ),
    ):
        try:
            trainer.train(
                seg1_cfg, dataset_path=str(ds),
                trusted_historical_checkpoint_sha256=_existing_shas(parent),
            )
        except _TestInterruption:
            pass
    seg1_dir = Path(seg1_cfg.checkpoint.output_dir)
    parent2 = seg1_dir / "epoch_3.pth"
    assert parent2.exists(), "second boundary checkpoint was not saved"

    seg2_cfg = _real_config(tmp_path, epochs=total, name="noval_seg2")
    seg2_cfg.checkpoint.resume_from = str(parent2)
    train_loader, _val = trainer.build_dataloaders(seg2_cfg, dataset_path=str(ds))
    with (
        mock.patch.object(trainer, "build_dataloaders",
                          return_value=(train_loader, None)),
        mock.patch.object(trainer, "plot_loss_curve",
                          return_value=tmp_path / "unused.png"),
        mock.patch.object(
            trainer, "train_one_epoch",
            side_effect=_no_val_epoch(real_epoch),
        ),
    ):
        trainer.train(
            seg2_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(parent2),
        )

    _assert_full_equivalence(_final_ckpt(full_cfg), _final_ckpt(seg2_cfg))
    chain = _final_ckpt(seg2_cfg)
    assert chain["training_history"]["val_loss"] == [None] * total, chain[
        "training_history"
    ]


def test_no_validation_segment_records_null_val_workers(tmp_path: Path) -> None:
    """A no-validation segment declares the absent val resolution as null."""
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)
    config, _ = _run_no_val_real(tmp_path, ds, epochs=1, name="noval_segment")
    record = json.loads(
        (Path(config.checkpoint.output_dir) / "segment_0000.json").read_text()
    )
    controls = record["provenance"]["current_segment_controls"]["loader_controls"]
    assert controls["resolved_num_workers"] == 0
    assert controls["resolved_val_num_workers"] is None
    assert controls["requested_num_workers"] == 0


def test_loss_curve_gaps_at_unobserved_positions(tmp_path: Path) -> None:
    """The loss curve breaks at a ``None`` slot instead of connecting across it.

    A ``None`` is an unobserved epoch.  Dropping the point would collapse the
    x-position of every later value; a plain plot would draw a segment straight
    through the missing epoch.  The curve must carry the full epoch axis and
    leave the unobserved position as a break (``nan``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.train import plot_loss_curve

    history = {"train_loss": [1.0, 0.9, 0.8, 0.7], "val_loss": [0.5, None, 0.4, None]}
    # plot_loss_curve closes its figure; keep it alive to inspect the artists.
    with mock.patch.object(plt, "close"):
        out = plot_loss_curve(history, tmp_path, history_start_epoch=0)
    assert out.exists()

    fig = plt.gcf()
    val_lines = [ln for ln in fig.axes[0].get_lines() if ln.get_label() == "Val Loss"]
    assert len(val_lines) == 1
    xdata, ydata = val_lines[0].get_xdata(), val_lines[0].get_ydata()
    assert list(xdata) == [1, 2, 3, 4], "the epoch axis was collapsed"
    assert math.isnan(ydata[1]) and math.isnan(ydata[3]), (
        f"unobserved positions were not left as gaps: {ydata}"
    )
    assert ydata[0] == 0.5 and ydata[2] == 0.4
    plt.close(fig)


# ═════════════════════════════════════════════════════════════
# Terminal publication crash (death between save and stop check)
# ═════════════════════════════════════════════════════════════
#
# The periodic checkpoint is published at a completed-epoch boundary BEFORE
# the early-stopping decision for that epoch is evaluated.  A process that
# dies in that window leaves a checkpoint whose restored counter has ALREADY
# reached the patience horizon: the uninterrupted run would have stopped
# there and never executed another epoch.  These tests run the REAL model,
# forward/backward, optimizer and scheduler (only validation is pinned to a
# controlled constant, disclosed below) with the SAME total epoch budget in
# every arm, and prove the restored terminal decision takes effect before any
# resumed epoch.  The constant validation is a controlled fixture: a constant
# 0.5 makes epoch 0 the sole improvement and every later finite epoch a
# non-improving count, so patience exhaustion is deterministic.

class _PublicationCrash(Exception):
    """Raised AFTER a completed checkpoint is published, BEFORE the stop check."""


def _run_publication_crash(
    tmp_path: Path, ds: Path, *, epochs: int, name: str, crash_after: str,
    patience: int, phase1_epochs: Optional[int] = None,
) -> Any:
    """Run real epochs; die immediately after ``crash_after`` is published."""
    from scripts import train as trainer

    config = _real_config(tmp_path, epochs=epochs, name=name, patience=patience)
    real_epoch = trainer.train_one_epoch
    real_save = trainer._atomic_save_checkpoint

    def save_then_die(**kwargs: Any) -> Any:
        path = real_save(**kwargs)
        if Path(path).name == crash_after:
            raise _PublicationCrash(
                f"died after publishing {crash_after}, before the stop check"
            )
        return path

    with (
        mock.patch.object(
            trainer, "train_one_epoch",
            side_effect=_epoch_with_numpy_draw(real_epoch),
        ),
        mock.patch.object(trainer, "validate", return_value=0.5),
        mock.patch.object(
            trainer, "_atomic_save_checkpoint", side_effect=save_then_die,
        ),
    ):
        try:
            trainer.train(
                config, phase1_epochs=phase1_epochs, dataset_path=str(ds),
            )
        except _PublicationCrash:
            pass
        else:
            raise AssertionError(f"run did not die after publishing {crash_after}")
    return config


def _resume_counting(
    tmp_path: Path,
    ds: Path,
    *,
    parent: Path,
    name: str,
    epochs: int,
    patience: int,
    phase1_epochs: Optional[int] = None,
    validate_return: float = 0.5,
    record_validate: Optional[List[Any]] = None,
) -> Tuple["ExperimentConfig", List[int], Dict[str, Any]]:
    """Resume from ``parent`` and record every epoch index that EXECUTED.

    ``record_validate`` is an optional list that also collects one entry per
    ``validate`` call, so a zero-update resume can be shown to execute no
    TRAIN *and* no VALIDATION call (default ``None`` leaves existing callers
    unchanged).
    """
    from scripts import train as trainer

    config = _real_config(tmp_path, epochs=epochs, name=name, patience=patience)
    config.checkpoint.resume_from = str(parent)
    executed: list[int] = []
    draw = _epoch_with_numpy_draw(trainer.train_one_epoch)

    def counted(**kwargs: Any) -> Any:
        executed.append(kwargs["epoch"])
        return draw(**kwargs)

    def counted_validate(**kwargs: Any) -> float:
        if record_validate is not None:
            record_validate.append(kwargs.get("loader"))
        return validate_return

    with (
        mock.patch.object(trainer, "train_one_epoch", side_effect=counted),
        mock.patch.object(trainer, "validate", side_effect=counted_validate),
    ):
        result = trainer.train(
            config, phase1_epochs=phase1_epochs, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                parent, Path(parent).parent / "best_model.pth",
            ),
        )
    return config, executed, result


def test_recovery_terminal_predicate_phase_semantics() -> None:
    """Narrow mutation check of the shared stop predicate's phase semantics.

    The resume and the in-loop stop check both consult this one predicate, so
    its boundary behaviour is the contract: phase-1 exemption only while a
    later phase 2 remains, phase-2 exhaustion terminal, a disabled patience
    never terminal.
    """
    from scripts.train import _recovery_terminal

    single = dict(two_phase=False, current_phase=0, num_epochs=6, phase1_epochs=None)
    assert _recovery_terminal(2, 2, **single), "exhausted single-phase must stop"
    assert not _recovery_terminal(2, 1, **single), "below horizon must not stop"
    assert not _recovery_terminal(0, 5, **single), "patience 0 disables stopping"

    # Phase 1 with a later phase 2 is exempt (a boundary reset is still ahead).
    assert not _recovery_terminal(
        2, 2, two_phase=True, current_phase=1, num_epochs=6, phase1_epochs=5,
    ), "phase-1 exemption lost"
    # ... but a phase 1 that IS the whole budget has no later reset: terminal.
    assert _recovery_terminal(
        2, 2, two_phase=True, current_phase=1, num_epochs=5, phase1_epochs=5,
    ), "phase 1 with no phase 2 must still stop"
    # Established phase 2 is terminal.
    assert _recovery_terminal(
        2, 2, two_phase=True, current_phase=2, num_epochs=6, phase1_epochs=1,
    ), "phase-2 exhaustion must stop"


def test_real_terminal_publication_crash_resume_executes_no_epoch(
    tmp_path: Path,
) -> None:
    """A published-but-terminal checkpoint must not execute another epoch.

    Uninterrupted: constant validation 0.5 with patience 2 exhausts at
    completed epoch index 2 (``epoch_3.pth``), and the run stops there.  A
    death immediately AFTER ``epoch_3.pth`` is published leaves a checkpoint
    that is byte-for-byte the uninterrupted terminal state.  Resuming it must
    reproduce that terminal state — no further epoch, no history growth, no
    patience growth — rather than run an epoch the uninterrupted run never did.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    full_cfg = _real_config(tmp_path, epochs=6, name="terminal_full", patience=2)
    real_epoch = trainer.train_one_epoch
    with (
        mock.patch.object(
            trainer, "train_one_epoch",
            side_effect=_epoch_with_numpy_draw(real_epoch),
        ),
        mock.patch.object(trainer, "validate", return_value=0.5),
    ):
        trainer.train(full_cfg, dataset_path=str(ds))
    full = _final_ckpt(full_cfg)
    assert full["epoch"] == 2, f"uninterrupted terminal epoch {full['epoch']}"
    assert full["epochs_without_improvement"] == 2
    assert len(full["training_history"]["train_loss"]) == 3

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=6, name="terminal_crash",
        crash_after="epoch_3.pth", patience=2,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    parent_state = _load_state(parent)
    assert parent_state["epoch"] == 2
    assert parent_state["epochs_without_improvement"] == 2, (
        "the published checkpoint is not already terminal; fixture invalid"
    )
    # The parent at publication equals the uninterrupted terminal state, so the
    # resume is the only thing that can introduce a divergence.
    _assert_model_equal(full["model_state_dict"], parent_state["model_state_dict"])
    assert torch.equal(full["rng_state"], parent_state["rng_state"])

    res_cfg, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="terminal_resume", epochs=6, patience=2,
    )
    assert executed == [], (
        f"terminal resume executed epochs {executed}; the exhausted horizon was "
        "reopened"
    )
    resumed = _final_ckpt(res_cfg)
    assert resumed["epoch"] == 2, "terminal finalization invented an epoch"
    assert len(resumed["training_history"]["train_loss"]) == 3
    assert resumed["epochs_without_improvement"] == 2
    _assert_full_equivalence(full, resumed)


def test_real_terminal_resume_ignores_would_be_improvement(tmp_path: Path) -> None:
    """A later improving validation cannot restart an exhausted horizon.

    Same publication crash as above, but the resumed validation returns 0.1 —
    strictly better than the terminal best of 0.5.  If the terminal decision
    were deferred to the in-loop check, the resumed run would execute epoch 3,
    reset patience to 0 and rewrite the best checkpoint, re-opening a decision
    the original run had already closed.  No epoch may execute, and the final
    checkpoint must report the parent's recorded 0.5, not the unobserved 0.1.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=6, name="improve_crash",
        crash_after="epoch_3.pth", patience=2,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"

    res_cfg, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="improve_resume", epochs=6, patience=2,
        validate_return=0.1,
    )
    assert executed == [], f"improving resume executed epochs {executed}"
    resumed = _final_ckpt(res_cfg)
    assert resumed["epoch"] == 2
    assert resumed["epochs_without_improvement"] == 2, "patience was reset"
    assert len(resumed["training_history"]["val_loss"]) == 3
    assert resumed["val_loss"] == 0.5, (
        f"finalization reported an unexecuted validation {resumed['val_loss']!r}"
    )


def test_real_terminal_phase1_exemption_preserved_on_resume(
    tmp_path: Path,
) -> None:
    """Phase 1 is exempt from early stopping, on resume exactly as in-loop.

    ``num_epochs=6 > phase1_epochs=5`` leaves a later phase 2, so an exhausted
    counter inside phase 1 must NOT stop the run.  The publication crash is the
    same shape as the terminal case, but the resumed epoch index 3 is still
    phase 1 and must EXECUTE — the guard may not fire where the in-loop check
    would not.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=6, name="phase1_crash",
        crash_after="epoch_3.pth", patience=2, phase1_epochs=5,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    assert _load_state(parent)["epochs_without_improvement"] == 2

    _, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="phase1_resume", epochs=6, patience=2,
        phase1_epochs=5,
    )
    assert executed and executed[0] == 3, (
        f"phase-1 exemption lost on resume; executed {executed}"
    )


def test_real_terminal_phase_boundary_reset_preserved_on_resume(
    tmp_path: Path,
) -> None:
    """A boundary-crossing resume resets patience, so it is never terminal.

    ``phase1_epochs=3`` and a crash after ``epoch_3.pth`` means the resume
    lands exactly at the phase-1→2 boundary: the counter resets, exactly as the
    in-loop transition resets it, so the guard must NOT stop the resumed run.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=6, name="boundary_crash",
        crash_after="epoch_3.pth", patience=2, phase1_epochs=3,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"

    _, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="boundary_resume", epochs=6, patience=2,
        phase1_epochs=3,
    )
    assert executed and executed[0] == 3, (
        f"phase-boundary reset lost on resume; executed {executed}"
    )


def test_real_terminal_phase2_exhaustion_stops_on_resume(tmp_path: Path) -> None:
    """Inside established phase 2, an exhausted counter is terminal on resume.

    ``phase1_epochs=1``: epoch 0 is phase 1, epoch 1 crosses the boundary and
    resets the counter, and constant 0.5 then exhausts patience 2 at completed
    epoch index 3 (``epoch_4.pth``).  A crash after that publication must not
    let the resume execute epoch index 4 — the phase-2 terminal decision
    applies before any resumed epoch.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    full_cfg = _real_config(tmp_path, epochs=6, name="phase2_full", patience=2)
    real_epoch = trainer.train_one_epoch
    with (
        mock.patch.object(
            trainer, "train_one_epoch",
            side_effect=_epoch_with_numpy_draw(real_epoch),
        ),
        mock.patch.object(trainer, "validate", return_value=0.5),
    ):
        trainer.train(full_cfg, phase1_epochs=1, dataset_path=str(ds))
    full = _final_ckpt(full_cfg)
    assert full["epoch"] == 3, f"uninterrupted terminal epoch {full['epoch']}"
    assert full["epochs_without_improvement"] == 2
    assert full["training_phase"] == 2

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=6, name="phase2_crash",
        crash_after="epoch_4.pth", patience=2, phase1_epochs=1,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_4.pth"
    parent_state = _load_state(parent)
    assert parent_state["epoch"] == 3
    assert parent_state["epochs_without_improvement"] == 2
    _assert_model_equal(full["model_state_dict"], parent_state["model_state_dict"])
    assert torch.equal(full["rng_state"], parent_state["rng_state"])

    res_cfg, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="phase2_resume", epochs=6, patience=2,
        phase1_epochs=1,
    )
    assert executed == [], f"phase-2 terminal resume executed epochs {executed}"
    resumed = _final_ckpt(res_cfg)
    assert resumed["epoch"] == 3
    assert resumed["epochs_without_improvement"] == 2
    assert len(resumed["training_history"]["train_loss"]) == 4
    _assert_full_equivalence(full, resumed)


def test_real_publication_crash_at_total_budget_finalizes_saved_state(
    tmp_path: Path,
) -> None:
    """A completed parent AT the original total budget must finalize, not fail.

    Real publication crash with the run's OWN budget: total 3, patience 2,
    cadence 1, zero workers, controlled constant validation 0.5.  The run
    completes epoch index 2 (``epoch_3.pth``) — which both reaches the total
    budget AND exhausts patience — and dies immediately after that checkpoint
    is published, before the stop check.  The parent is therefore the
    uninterrupted run's terminal state.

    Resuming it with the SAME budget has zero remaining updates
    (``range(3, 3)`` is empty), so it must finalize at the SAVED executed
    epoch (index 2) with the parent's complete continuation state — model,
    optimizer state and groups, scheduler, Torch RNG, the full decoded
    MT19937 state with both continuations, history, origin and patience —
    executing NO train or val call.  Before the fix this raised
    ``ValueError`` (``target num_epochs (3) <= start_epoch (3)``) and wrote
    no ``final_model.pth``.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    full_cfg = _real_config(tmp_path, epochs=3, name="budget_full", patience=2)
    real_epoch = trainer.train_one_epoch
    with (
        mock.patch.object(
            trainer, "train_one_epoch",
            side_effect=_epoch_with_numpy_draw(real_epoch),
        ),
        mock.patch.object(trainer, "validate", return_value=0.5),
    ):
        trainer.train(full_cfg, dataset_path=str(ds))
    full = _final_ckpt(full_cfg)
    assert full["epoch"] == 2, f"uninterrupted terminal epoch {full['epoch']}"
    assert full["epochs_without_improvement"] == 2
    assert len(full["training_history"]["train_loss"]) == 3

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name="budget_crash",
        crash_after="epoch_3.pth", patience=2,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    parent_state = _load_state(parent)
    assert parent_state["epoch"] == 2
    assert parent_state["epochs_without_improvement"] == 2, (
        "the published checkpoint is not already terminal; fixture invalid"
    )
    # The parent equals the uninterrupted terminal state, so only the resume
    # can introduce a divergence.
    _assert_model_equal(full["model_state_dict"], parent_state["model_state_dict"])
    assert torch.equal(full["rng_state"], parent_state["rng_state"])

    validated: list[object] = []
    res_cfg, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="budget_resume", epochs=3, patience=2,
        record_validate=validated,
    )
    assert executed == [], (
        f"budget-terminal resume executed epochs {executed}; a completed run "
        "has no remaining updates"
    )
    assert validated == [], f"budget-terminal resume called validate {len(validated)}x"
    resumed = _final_ckpt(res_cfg)
    assert resumed["epoch"] == 2, "budget finalization invented an epoch"
    assert resumed["epochs_without_improvement"] == 2
    assert len(resumed["training_history"]["train_loss"]) == 3
    assert resumed["training_history"] == full["training_history"]
    _assert_full_equivalence(full, resumed)


def test_real_total_budget_exhaustion_with_patience_disabled_finalizes(
    tmp_path: Path,
) -> None:
    """Budget exhaustion alone is terminal — patience is NOT required.

    Same real crash shape but ``patience=0`` (early stopping disabled).  The
    counter still reaches 2 (the loop counts finite non-improving epochs
    whatever the horizon), but the shared stop predicate is False for
    ``patience=0``, so ONLY the total budget makes this parent terminal.  A
    fix that keyed off ``patience`` would raise here; the resume must still
    finalize at the saved epoch with no executed epoch.
    """
    from scripts.train import _recovery_terminal

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name="budgetonly_crash",
        crash_after="epoch_3.pth", patience=0,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    parent_state = _load_state(parent)
    assert parent_state["epoch"] == 2
    assert parent_state["epochs_without_improvement"] == 2
    assert not _recovery_terminal(
        0, parent_state["epochs_without_improvement"],
        two_phase=False, current_phase=0, num_epochs=3, phase1_epochs=None,
    ), "fixture invalid: patience=0 must make the stop predicate False"

    res_cfg, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="budgetonly_resume", epochs=3, patience=0,
    )
    assert executed == [], f"budget-only resume executed epochs {executed}"
    resumed = _final_ckpt(res_cfg)
    assert resumed["epoch"] == 2
    assert resumed["epochs_without_improvement"] == 2
    assert resumed["training_history"] == parent_state["training_history"]


def test_real_resume_beyond_total_budget_still_fails_closed(
    tmp_path: Path,
) -> None:
    """An OVERSHOT parent is still refused — only equality is accepted.

    A parent completed at epoch index 3 (``epoch_4.pth``) resumed under a
    SHORTER total budget of 3 has ``start_epoch=4 > num_epochs=3``: the
    active budget would truncate a longer parent run, so it must fail closed
    exactly as before, with no checkpoint written.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=4, name="overshoot_crash",
        crash_after="epoch_4.pth", patience=0,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_4.pth"
    assert _load_state(parent)["epoch"] == 3

    res_cfg = _real_config(tmp_path, epochs=3, name="overshoot_resume", patience=0)
    res_cfg.checkpoint.resume_from = str(parent)
    with pytest.raises(ValueError, match="exceeds target num_epochs|fail closed"):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(parent),
        )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


def test_real_legacy_parent_at_total_budget_still_fails_closed(
    tmp_path: Path,
) -> None:
    """A legacy parent at the total budget cannot be validated — fail closed.

    Without ``recovery_state_version`` the patience counter and history are
    UNKNOWN (the reader would reset them with a warning), so a completed run
    cannot be validated.  The budget-equality acceptance is current-schema
    only and must not swallow this case.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name="legacybudget_crash",
        crash_after="epoch_3.pth", patience=0,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    legacy = _load_state(parent)
    for key in (
        "recovery_state_version",
        "epochs_without_improvement",
        "training_history",
        "history_start_epoch",
        "numpy_rng_state",
    ):
        legacy.pop(key, None)
    torch.save(legacy, parent)

    res_cfg = _real_config(tmp_path, epochs=3, name="legacybudget_resume", patience=0)
    res_cfg.checkpoint.resume_from = str(parent)
    with pytest.raises(ValueError, match="legacy or version-mismatched|fail closed"):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(parent),
        )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


def test_real_current_schema_parent_at_total_budget_with_bad_counter_fails(
    tmp_path: Path,
) -> None:
    """Current-schema corruption at the total budget fails BEFORE finalization.

    The budget-equality path must still validate the recovery state first: a
    counter that exceeds every validation epoch that could have counted is
    structural corruption and must raise before any ``final_model.pth``.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name="badcounter_crash",
        crash_after="epoch_3.pth", patience=0,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    corrupt = _load_state(parent)
    assert corrupt["recovery_state_version"] == 2
    corrupt["epochs_without_improvement"] = 99  # exceeds 3 possible epochs
    torch.save(corrupt, parent)

    res_cfg = _real_config(tmp_path, epochs=3, name="badcounter_resume", patience=0)
    res_cfg.checkpoint.resume_from = str(parent)
    with pytest.raises(ValueError, match="exceeds the number of validation epochs|fail closed"):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            # The crash run's companion best model must also be trusted, so the
            # companion reconciliation does not fail first and mask the counter
            # check under test.
            trusted_historical_checkpoint_sha256=_existing_shas(
                parent, parent.parent / "best_model.pth",
            ),
        )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


def test_real_budget_terminal_two_phase_phase1_equals_budget_finalizes(
    tmp_path: Path,
) -> None:
    """``phase1_epochs == num_epochs``: no phase-2 epoch ever runs.

    The phase-1→2 transition fires at the FIRST phase-2 epoch, which this
    completed run never executes, so the boundary must NOT be treated as
    crossed on a budget-terminal resume.  Treating it as a crossing would
    discard the phase-1 best checkpoint and leave the finalized run with no
    best model.  The resume executes nothing, stays in phase 1, and the
    phase-1 best survives.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name="phase1budget_crash",
        crash_after="epoch_3.pth", patience=0, phase1_epochs=3,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    parent_state = _load_state(parent)
    assert parent_state["epoch"] == 2
    assert parent_state["training_phase"] == 1

    res_cfg, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="phase1budget_resume", epochs=3,
        patience=0, phase1_epochs=3,
    )
    assert executed == [], f"phase-1 budget resume executed epochs {executed}"
    resumed = _final_ckpt(res_cfg)
    assert resumed["epoch"] == 2
    assert resumed["training_phase"] == 1
    assert resumed["training_history"] == parent_state["training_history"]
    assert (Path(res_cfg.checkpoint.output_dir) / "best_model.pth").exists(), (
        "phase-1 best was discarded by a boundary crossing that never happened"
    )


def test_real_budget_terminal_two_phase_phase2_state_preserved(
    tmp_path: Path,
) -> None:
    """A budget-terminal resume INSIDE phase 2 keeps the phase-2 posture.

    ``phase1_epochs=1`` with total budget 3: the boundary is crossed at epoch 1
    (inside the run), so the completed parent is phase 2 with the 2-group
    backend optimizer.  The budget-terminal resume must restore that same
    posture — phase 2, a 2-group optimizer, the same scheduler state — not a
    phase-1 or single-group shape.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name="phase2budget_crash",
        crash_after="epoch_3.pth", patience=0, phase1_epochs=1,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    parent_state = _load_state(parent)
    assert parent_state["epoch"] == 2
    assert parent_state["training_phase"] == 2
    assert len(parent_state["optimizer_state_dict"]["param_groups"]) == 2

    res_cfg, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="phase2budget_resume", epochs=3,
        patience=0, phase1_epochs=1,
    )
    assert executed == [], f"phase-2 budget resume executed epochs {executed}"
    resumed = _final_ckpt(res_cfg)
    assert resumed["epoch"] == 2
    assert resumed["training_phase"] == 2
    assert len(resumed["optimizer_state_dict"]["param_groups"]) == 2
    _assert_optimizer_equal(
        parent_state["optimizer_state_dict"], resumed["optimizer_state_dict"],
    )
    assert _sched_state(parent_state) == _sched_state(resumed)
    assert resumed["training_history"] == parent_state["training_history"]


def test_real_longer_parent_at_active_budget_epoch_still_fails_closed(
    tmp_path: Path,
) -> None:
    """A parent run under a LONGER total budget is not this run's completion.

    Real publication crash at total budget 6 with patience 0: the run completes
    epoch index 2 and dies after publishing ``epoch_3.pth``.  Resuming it under
    a SHORTER active budget of 3 gives ``start_epoch == 3 == num_epochs``, so
    the equality branch fires — but the parent's own recorded total budget is
    6, not 3.  Finalizing would stamp the segment with a budget the parent never
    ran under and truncate a longer authorized run, so it must fail closed with
    no ``final_model.pth``.  The total budget is immutable across a
    continuation; equality with the ACTIVE budget alone is not completion.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=6, name="longparent_crash",
        crash_after="epoch_3.pth", patience=0,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    parent_state = _load_state(parent)
    assert parent_state["epoch"] == 2
    assert parent_state["config"]["training"]["num_epochs"] == 6

    res_cfg = _real_config(tmp_path, epochs=3, name="longparent_resume", patience=0)
    res_cfg.checkpoint.resume_from = str(parent)
    with pytest.raises(ValueError, match="original total budget|fail closed"):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                parent, parent.parent / "best_model.pth",
            ),
        )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


def test_real_parent_budget_equal_but_malformed_still_fails_closed(
    tmp_path: Path,
) -> None:
    """A known parent budget of the WRONG TYPE is not a valid completion.

    Same real shape with the parent's recorded total budget rewritten to the
    string ``"3"``: it is neither an integer nor equal to the active budget by
    type-safe comparison, so the equality branch must fail closed rather than
    coerce it.  (``True`` is likewise rejected as a bool, never coerced to 1.)
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name="malformedbudget_crash",
        crash_after="epoch_3.pth", patience=0,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    _rewrite_parent_training_control(parent, "num_epochs", "3")

    res_cfg = _real_config(tmp_path, epochs=3, name="malformedbudget_resume", patience=0)
    res_cfg.checkpoint.resume_from = str(parent)
    with pytest.raises(ValueError, match="original total budget|fail closed"):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                parent, parent.parent / "best_model.pth",
            ),
        )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


def test_real_parent_budget_unknown_still_fails_closed(
    tmp_path: Path,
) -> None:
    """A parent with NO recorded total budget cannot prove completion.

    The parent's ``config.training`` is stripped of ``num_epochs`` while the
    rest of the current-schema recovery state stays intact.  The budget is
    unknown, so the equality branch must fail closed rather than substitute the
    active run's budget.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name="unknownbudget_crash",
        crash_after="epoch_3.pth", patience=0,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    state = _load_state(parent)
    assert state["recovery_state_version"] == 2
    state["config"]["training"].pop("num_epochs")
    torch.save(state, parent)

    res_cfg = _real_config(tmp_path, epochs=3, name="unknownbudget_resume", patience=0)
    res_cfg.checkpoint.resume_from = str(parent)
    with pytest.raises(ValueError, match="original total budget|fail closed"):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                parent, parent.parent / "best_model.pth",
            ),
        )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


def _terminal_parent_with_diagnostics(
    tmp_path: Path,
    ds: Path,
    *,
    name: str,
    top: Optional[Dict[str, Any]] = None,
    history_tail: Optional[Dict[str, Any]] = None,
) -> Path:
    """A real publication-crash parent at budget 3, patience 2, diagnostics rewritten.

    The parent is the uninterrupted run's terminal state (epoch index 2,
    counter 2).  ``top`` / ``history_tail`` override the recorded diagnostics
    so a resume's terminal-diagnostic selection is the only variable.
    """
    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name=name, crash_after="epoch_3.pth", patience=2,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    if top is not None or history_tail is not None:
        _rewrite_parent_diagnostics(parent, top=top, history_tail=history_tail)
    return parent


def test_real_terminal_keeps_parent_scalar_when_history_tail_is_gap(
    tmp_path: Path,
) -> None:
    """A parent scalar MUST survive a legitimate history GAP.

    The parent recorded ``train_loss`` 1.3165016174316406 and ``val_loss`` 0.5
    at its last executed epoch, but the history tail for BOTH series is the
    explicit ``None`` GAP a no-validation / unrecorded epoch writes.  The
    finalized checkpoint must keep the parent's measured scalars, never
    overwrite them with the gap.  Before the fix the final ``loss``/
    ``train_loss`` became ``None`` and ``val_loss`` was dropped.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    parent = _terminal_parent_with_diagnostics(
        tmp_path, ds, name="diag_gap_crash",
        top={"train_loss": 1.3165016174316406, "val_loss": 0.5},
        history_tail={"train_loss": None, "val_loss": None},
    )
    parent_state = _load_state(parent)
    assert parent_state["train_loss"] == 1.3165016174316406
    assert parent_state["val_loss"] == 0.5
    assert parent_state["training_history"]["train_loss"][-1] is None

    res_cfg, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="diag_gap_resume", epochs=3, patience=2,
    )
    assert executed == [], f"budget-terminal resume executed epochs {executed}"
    resumed = _final_ckpt(res_cfg)
    assert resumed["loss"] == 1.3165016174316406, "parent train scalar was discarded"
    assert resumed["train_loss"] == 1.3165016174316406
    assert resumed["val_loss"] == 0.5, "parent val scalar was discarded"


def test_real_terminal_uses_history_tail_when_top_scalar_absent(
    tmp_path: Path,
) -> None:
    """A recorded history scalar fills in when the top-level scalar is ABSENT.

    The parent carries no top-level ``train_loss``/``val_loss`` (a shape an
    older writer could produce) but its history tail holds the recorded
    scalars.  The terminal finalization must use them, not fall back to the
    ``nan`` placeholder / drop the value.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    parent = _terminal_parent_with_diagnostics(
        tmp_path, ds, name="diag_tail_crash",
        top={"train_loss": None, "val_loss": None},
    )
    parent_state = _load_state(parent)
    assert "train_loss" not in parent_state
    assert "val_loss" not in parent_state
    assert isinstance(parent_state["training_history"]["train_loss"][-1], float)
    assert parent_state["training_history"]["val_loss"][-1] == 0.5

    res_cfg, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="diag_tail_resume", epochs=3, patience=2,
    )
    assert executed == []
    resumed = _final_ckpt(res_cfg)
    tail_train = parent_state["training_history"]["train_loss"][-1]
    assert resumed["train_loss"] == tail_train, "history scalar was not used"
    assert resumed["val_loss"] == 0.5


def test_real_terminal_truly_unobserved_writes_no_fabricated_loss(
    tmp_path: Path,
) -> None:
    """Neither source has a value: the finalization must NOT fabricate one.

    With the top-level scalars removed (including the independent legacy
    ``loss``, which is otherwise a valid train diagnostic) AND the history tail
    set to the ``None`` gap, the parent genuinely recorded no observation for
    the last epoch.  The final checkpoint must serialize the explicit absence
    (no ``train_loss``/``val_loss``), never a ``0``/finite placeholder.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    parent = _terminal_parent_with_diagnostics(
        tmp_path, ds, name="diag_unobserved_crash",
        top={"train_loss": None, "val_loss": None, "loss": None},
        history_tail={"train_loss": None, "val_loss": None},
    )
    res_cfg, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="diag_unobserved_resume",
        epochs=3, patience=2,
    )
    assert executed == []
    resumed = _final_ckpt(res_cfg)
    assert "train_loss" not in resumed, "fabricated a train loss the run never measured"
    assert "val_loss" not in resumed, "fabricated a val loss the run never measured"


def test_real_terminal_nonfinite_parent_scalar_preserved_verbatim(
    tmp_path: Path,
) -> None:
    """A non-finite parent diagnostic is a RECORDED result, kept verbatim.

    Both sources record the same ``NaN`` train loss (a genuinely unstable
    epoch).  ``NaN`` must pass through as a recorded value — not be treated as
    a gap, not be replaced by the other source, and not be coerced.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    parent = _terminal_parent_with_diagnostics(
        tmp_path, ds, name="diag_nan_crash",
        top={"train_loss": float("nan")},
        history_tail={"train_loss": float("nan")},
    )
    res_cfg, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="diag_nan_resume", epochs=3, patience=2,
    )
    assert executed == []
    resumed = _final_ckpt(res_cfg)
    assert "train_loss" in resumed
    assert math.isnan(resumed["train_loss"]), "recorded NaN was not preserved"


def _incomplete_schema_parent(
    tmp_path: Path, ds: Path, *, name: str, drop: str,
) -> Path:
    """A current-schema publication-crash parent with one required block removed.

    The parent is a real completed run at its own total budget (epoch index 2,
    counter 2).  ``drop`` names the canonical state block deleted from the saved
    dict (``model_state_dict`` / ``optimizer_state_dict`` /
    ``scheduler_state_dict`` / ``rng_state``), leaving every recovery field
    intact, so the completeness preflight is the only variable under test.
    """
    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name=name, crash_after="epoch_3.pth", patience=2,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    state = _load_state(parent)
    assert state["recovery_state_version"] == 2
    assert drop in state, f"fixture invalid: {drop} absent"
    del state[drop]
    torch.save(state, parent)
    return parent


def test_real_current_schema_missing_optimizer_state_fails_closed(
    tmp_path: Path,
) -> None:
    """A current-schema parent missing optimizer moments must NOT resume.

    The canonical loader tolerates an absent ``optimizer_state_dict`` and
    restores only what is present, so resuming would silently continue with a
    FRESH optimizer (zero Adam moments) while still claiming an exact
    continuation.  A current-schema payload that omits it is damaged required
    state and must fail closed before any update or terminal save.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    parent = _incomplete_schema_parent(
        tmp_path, ds, name="missingopt_crash", drop="optimizer_state_dict",
    )
    res_cfg = _real_config(tmp_path, epochs=3, name="missingopt_resume", patience=2)
    res_cfg.checkpoint.resume_from = str(parent)
    with pytest.raises(ValueError, match="optimizer_state_dict|fail closed"):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                parent, parent.parent / "best_model.pth",
            ),
        )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


def test_real_current_schema_missing_scheduler_state_fails_closed(
    tmp_path: Path,
) -> None:
    """A current-schema parent missing the scheduler must NOT resume.

    Same shape with ``scheduler_state_dict`` removed: the loader would restart
    the LR schedule from scratch, which is not an exact continuation, so the
    completeness preflight fails closed with no ``final_model.pth``.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    parent = _incomplete_schema_parent(
        tmp_path, ds, name="missingsched_crash", drop="scheduler_state_dict",
    )
    res_cfg = _real_config(tmp_path, epochs=3, name="missingsched_resume", patience=2)
    res_cfg.checkpoint.resume_from = str(parent)
    with pytest.raises(ValueError, match="scheduler_state_dict|fail closed"):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                parent, parent.parent / "best_model.pth",
            ),
        )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


def test_real_current_schema_missing_rng_state_fails_closed(
    tmp_path: Path,
) -> None:
    """A current-schema parent missing the Torch RNG state must NOT resume."""
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    parent = _incomplete_schema_parent(
        tmp_path, ds, name="missingrng_crash", drop="rng_state",
    )
    res_cfg = _real_config(tmp_path, epochs=3, name="missingrng_resume", patience=2)
    res_cfg.checkpoint.resume_from = str(parent)
    with pytest.raises(ValueError, match="rng_state|fail closed"):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                parent, parent.parent / "best_model.pth",
            ),
        )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


def test_real_current_schema_explicit_none_state_fails_closed(
    tmp_path: Path,
) -> None:
    """An explicit ``None`` for required state is damaged, not a legacy gap."""
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name="nonestate_crash",
        crash_after="epoch_3.pth", patience=2,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    state = _load_state(parent)
    state["optimizer_state_dict"] = None
    torch.save(state, parent)

    res_cfg = _real_config(tmp_path, epochs=3, name="nonestate_resume", patience=2)
    res_cfg.checkpoint.resume_from = str(parent)
    with pytest.raises(ValueError, match="None|fail closed"):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                parent, parent.parent / "best_model.pth",
            ),
        )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


def _terminal_parent_loss_alias(
    tmp_path: Path, ds: Path, *, name: str,
    train_loss: Any = "KEEP", loss: Any = "KEEP",
    history_train_tail: Any = "KEEP",
) -> Path:
    """A terminal parent with the train diagnostic fields set explicitly.

    ``"KEEP"`` leaves a field as the real run wrote it; any other value sets it
    (``None`` sets an explicit ``None``; the string ``"DROP"`` deletes the key).
    """
    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name=name, crash_after="epoch_3.pth", patience=2,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    state = _load_state(parent)
    for key, val in (("train_loss", train_loss), ("loss", loss)):
        if val == "KEEP":
            continue
        if val == "DROP":
            state.pop(key, None)
        else:
            state[key] = val
    if history_train_tail != "KEEP":
        state["training_history"]["train_loss"][-1] = history_train_tail
    torch.save(state, parent)
    return parent


def test_real_terminal_loss_alias_used_when_train_loss_absent(
    tmp_path: Path,
) -> None:
    """A missing ``train_loss`` with an independent ``loss`` present keeps it.

    The parent recorded the step-level ``loss`` (1.2345) but no top-level
    ``train_loss``; its history train tail is the ``None`` gap.  The finalized
    checkpoint must preserve the independent ``loss`` value as the train
    diagnostic — a MISSING field is not the same as a ``None`` gap, and the
    present legacy scalar must not be discarded.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    parent = _terminal_parent_loss_alias(
        tmp_path, ds, name="alias_crash",
        train_loss="DROP", loss=1.2345, history_train_tail=None,
    )
    state = _load_state(parent)
    assert "train_loss" not in state
    assert state["loss"] == 1.2345

    res_cfg, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="alias_resume", epochs=3, patience=2,
    )
    assert executed == []
    resumed = _final_ckpt(res_cfg)
    assert resumed["train_loss"] == 1.2345, "independent loss field was discarded"


def test_real_terminal_explicit_none_train_loss_not_substituted_by_loss(
    tmp_path: Path,
) -> None:
    """An EXPLICIT ``train_loss=None`` is NOT substituted by the legacy ``loss``.

    Here ``train_loss`` is present and explicitly ``None`` (an unobserved train
    position) while ``loss`` carries a value.  An explicit gap and a missing
    field are distinct: the terminal diagnostic must stay unobserved (no
    ``train_loss`` written), never adopt the legacy scalar.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    parent = _terminal_parent_loss_alias(
        tmp_path, ds, name="explicitnone_crash",
        train_loss=None, loss=1.2345, history_train_tail=None,
    )
    state = _load_state(parent)
    assert "train_loss" in state and state["train_loss"] is None

    res_cfg, executed, _ = _resume_counting(
        tmp_path, ds, parent=parent, name="explicitnone_resume",
        epochs=3, patience=2,
    )
    assert executed == []
    resumed = _final_ckpt(res_cfg)
    assert "train_loss" not in resumed, (
        "an explicit None train position was wrongly filled from the loss field"
    )


def test_real_terminal_loss_alias_conflict_with_history_fails_closed(
    tmp_path: Path,
) -> None:
    """A legacy ``loss`` that DISAGREES with the history tail fails closed.

    With ``train_loss`` absent, the legacy ``loss`` (1.2345) is the train
    source; the history train tail records a DIFFERENT scalar (9.8765).  Both
    describe the same epoch and must agree, so the inconsistency fails closed.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    parent = _terminal_parent_loss_alias(
        tmp_path, ds, name="aliasconflict_crash",
        train_loss="DROP", loss=1.2345, history_train_tail=9.8765,
    )
    res_cfg = _real_config(tmp_path, epochs=3, name="aliasconflict_resume", patience=2)
    res_cfg.checkpoint.resume_from = str(parent)
    with pytest.raises(ValueError, match="disagree|fail closed"):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                parent, parent.parent / "best_model.pth",
            ),
        )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


# The EXACT ``CosineAnnealingLR.state_dict()`` key set produced by the pinned
# torch (2.14.0+cu132) used by this project.  ``LRScheduler.state_dict``
# serializes every ``__dict__`` entry except the optimizer, so this is the real
# production schema: the cosine endpoints/rates plus the torch-2.x step-state
# flags (``_step_count`` / ``_get_lr_called_within_step`` / ``_is_initial``).
# ``verbose`` is NOT part of this version's state.
_PRODUCTION_SCHEDULER_KEYS = {
    "T_max", "_get_lr_called_within_step", "_is_initial", "_last_lr",
    "_step_count", "base_lrs", "eta_min", "last_epoch",
}


def _scheduler_partial_parent(
    tmp_path: Path, ds: Path, *, name: str, mutate: str,
) -> Path:
    """A current-schema parent whose ``scheduler_state_dict`` is damaged.

    The parent is a real completed run at its own total budget (epoch index 2,
    counter 2) carrying the FULL ``CosineAnnealingLR`` state.  ``mutate``
    rewrites only that block, leaving every other recovery field intact, so the
    scheduler-schema preflight is the only variable under test:

    * ``"truncate"`` — keep only ``{"last_epoch": 1}`` (the reviewer probe);
      the real ``load_state_dict`` would merge it onto the fresh scheduler and
      silently resume on the freshly-built ``base_lrs`` / ``T_max``.
    * ``"arbitrary"`` — a non-empty mapping with no real scheduler field at all.
    * ``"group_mismatch"`` — a complete, finite state whose ``base_lrs`` /
      ``_last_lr`` carry ONE entry while the parent's optimizer_state_dict
      records TWO param groups (the reviewer's phase-2 probe).  The real
      ``get_lr`` would build a single value from the one-element ``base_lrs``
      and the zip against two groups would truncate: group 2's LR is never
      annealed.
    * ``"nan_eta_min"`` — a finite-structure state whose ``eta_min`` is ``NaN``;
      ``load_state_dict`` would merge it and poison the cosine denominator.
    * ``"inf_base_lrs"`` — ``base_lrs`` carries ``inf`` for the first group.
    * ``"nan_last_lr"`` — the cached ``_last_lr`` carries ``NaN``.
    * ``"nonpositive_tmax"`` — ``T_max`` = 0 (the producer always clamps >= 1).
    * ``"extra_key"`` — a complete state plus a foreign ``"verbose"`` key (not
      part of the pinned torch 2.14.0+cu132 ``CosineAnnealingLR`` schema).
    * ``"missing_key"`` — a state with a real production field (``_is_initial``)
      removed, so the key set is incomplete.
    """
    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name=name, crash_after="epoch_3.pth", patience=2,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    state = _load_state(parent)
    assert state["recovery_state_version"] == 2
    sched_sd = state["scheduler_state_dict"]
    # The real producer's EXACT schema for the pinned torch.
    assert set(sched_sd) == _PRODUCTION_SCHEDULER_KEYS
    if mutate == "truncate":
        state["scheduler_state_dict"] = {"last_epoch": 1}
    elif mutate == "arbitrary":
        state["scheduler_state_dict"] = {"malformed-but-nonempty": 1}
    elif mutate == "group_mismatch":
        # A COMPLETE, finite structure that is nonetheless inconsistent with
        # the optimizer state: one rate entry, two recorded param groups.
        assert len(state["optimizer_state_dict"]["param_groups"]) == 2
        sched_sd["base_lrs"] = list(sched_sd["base_lrs"])[:1]
        sched_sd["_last_lr"] = list(sched_sd["_last_lr"])[:1]
    elif mutate == "nan_eta_min":
        sched_sd["eta_min"] = float("nan")
    elif mutate == "inf_base_lrs":
        sched_sd["base_lrs"] = [float("inf")] + list(sched_sd["base_lrs"])[1:]
    elif mutate == "nan_last_lr":
        sched_sd["_last_lr"] = [float("nan")] + list(sched_sd["_last_lr"])[1:]
    elif mutate == "nonpositive_tmax":
        sched_sd["T_max"] = 0
    elif mutate == "extra_key":
        sched_sd["verbose"] = False
    elif mutate == "missing_key":
        del sched_sd["_is_initial"]
    else:
        raise AssertionError(f"unknown mutate={mutate!r}")
    torch.save(state, parent)
    return parent


def test_real_current_schema_partial_scheduler_state_fails_closed(
    tmp_path: Path,
) -> None:
    """A TRUNCATED scheduler state must NOT resume on fresh defaults.

    Reviewer probe: ``{"last_epoch": 1}`` passes the non-empty check and
    ``CosineAnnealingLR.load_state_dict`` silently merges it, keeping the
    freshly-built ``base_lrs`` / ``T_max`` — a wrong LR trajectory claimed as an
    exact continuation.  The completeness preflight must fail closed before any
    update or terminal save.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    parent = _scheduler_partial_parent(
        tmp_path, ds, name="partialsched_crash", mutate="truncate",
    )
    res_cfg = _real_config(tmp_path, epochs=3, name="partialsched_resume", patience=2)
    res_cfg.checkpoint.resume_from = str(parent)
    with pytest.raises(ValueError, match="scheduler_state_dict|fail closed"):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                parent, parent.parent / "best_model.pth",
            ),
        )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


def test_real_current_schema_arbitrary_scheduler_state_fails_closed(
    tmp_path: Path,
) -> None:
    """A NON-EMPTY but arbitrary scheduler mapping must NOT resume.

    Reviewer probe: ``{"malformed-but-nonempty": 1}`` is accepted by the
    non-empty check and ignored by ``load_state_dict``, so the run would resume
    on the fresh scheduler with no error.  The completeness preflight must fail
    closed.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    parent = _scheduler_partial_parent(
        tmp_path, ds, name="arbitrarysched_crash", mutate="arbitrary",
    )
    res_cfg = _real_config(tmp_path, epochs=3, name="arbitrarysched_resume", patience=2)
    res_cfg.checkpoint.resume_from = str(parent)
    with pytest.raises(ValueError, match="scheduler_state_dict|fail closed"):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                parent, parent.parent / "best_model.pth",
            ),
        )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


def _assert_scheduler_resume_fails_closed(
    tmp_path: Path, ds: Path, parent: Path, *, name: str,
) -> None:
    """Resume ``parent`` and prove the preflight rejected it BEFORE any update.

    The mapping must fail in the resume preflight: ``train_one_epoch`` is never
    called (no optimizer update) and no ``final_model.pth`` is written (no
    terminal save).  A rejection that only surfaced later — or after an epoch —
    would already have poisoned the run.
    """
    from scripts import train as trainer

    res_cfg = _real_config(tmp_path, epochs=3, name=name, patience=2)
    res_cfg.checkpoint.resume_from = str(parent)
    real_epoch = trainer.train_one_epoch
    executed: list[int] = []

    def counted(**kwargs: Any) -> Any:
        executed.append(kwargs["epoch"])
        return real_epoch(**kwargs)

    with (
        mock.patch.object(trainer, "train_one_epoch", side_effect=counted),
        mock.patch.object(trainer, "validate", return_value=0.5),
        pytest.raises(ValueError, match="scheduler_state_dict|fail closed"),
    ):
        trainer.train(
            res_cfg, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                parent, parent.parent / "best_model.pth",
            ),
        )
    assert executed == [], (
        f"scheduler preflight let epoch(s) {executed} execute before failing"
    )
    assert not (Path(res_cfg.checkpoint.output_dir) / "final_model.pth").exists()


def test_real_scheduler_group_count_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    """A one-entry rate list with a TWO-group optimizer must NOT resume.

    Reviewer A blocker (phase-2 probe): ``base_lrs`` / ``_last_lr`` of length 1
    are individually complete and finite, so the r11 structure check passed
    them.  But the parent's ``optimizer_state_dict`` records two param groups,
    and ``CosineAnnealingLR.get_lr`` builds its value list from ``base_lrs``
    BEFORE the ``zip(..., strict=True)`` — so a one-element list is truncated to
    group 1 and the LIF group's LR is never annealed while the run still claims
    an exact continuation.  The per-group count must match the optimizer state.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)
    parent = _scheduler_partial_parent(
        tmp_path, ds, name="groupmismatch_crash", mutate="group_mismatch",
    )
    _assert_scheduler_resume_fails_closed(
        tmp_path, ds, parent, name="groupmismatch_resume",
    )


def test_real_scheduler_nan_eta_min_fails_closed(tmp_path: Path) -> None:
    """``eta_min = NaN`` must NOT resume (finite-number check).

    Reviewer A blocker: ``_is_real_number`` accepted ``NaN``/``Inf``, so a
    ``NaN`` ``eta_min`` passed preflight and ``load_state_dict`` merged it onto
    the production cosine scheduler, poisoning every subsequent LR and the
    terminal checkpoint.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)
    parent = _scheduler_partial_parent(
        tmp_path, ds, name="nanetamin_crash", mutate="nan_eta_min",
    )
    _assert_scheduler_resume_fails_closed(
        tmp_path, ds, parent, name="nanetamin_resume",
    )


def test_real_scheduler_inf_base_lr_fails_closed(tmp_path: Path) -> None:
    """An ``inf`` entry in ``base_lrs`` must NOT resume."""
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)
    parent = _scheduler_partial_parent(
        tmp_path, ds, name="infbaselr_crash", mutate="inf_base_lrs",
    )
    _assert_scheduler_resume_fails_closed(
        tmp_path, ds, parent, name="infbaselr_resume",
    )


def test_real_scheduler_nan_last_lr_fails_closed(tmp_path: Path) -> None:
    """A ``NaN`` entry in the cached ``_last_lr`` must NOT resume."""
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)
    parent = _scheduler_partial_parent(
        tmp_path, ds, name="nanlastlr_crash", mutate="nan_last_lr",
    )
    _assert_scheduler_resume_fails_closed(
        tmp_path, ds, parent, name="nanlastlr_resume",
    )


def test_real_scheduler_nonpositive_tmax_fails_closed(tmp_path: Path) -> None:
    """``T_max <= 0`` must NOT resume (the producer always clamps to >= 1).

    Reviewer A blocker: a non-positive horizon was accepted, but
    ``CosineAnnealingLR`` is built as ``max(1, ...)`` at both phase-1 and
    phase-2 construction sites, so ``T_max`` = 0 is not a real continuation.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)
    parent = _scheduler_partial_parent(
        tmp_path, ds, name="zerotmax_crash", mutate="nonpositive_tmax",
    )
    _assert_scheduler_resume_fails_closed(
        tmp_path, ds, parent, name="zerotmax_resume",
    )


def test_real_scheduler_extra_key_fails_closed(tmp_path: Path) -> None:
    """A foreign scheduler key (``verbose``) must NOT resume.

    Reviewer B: the pinned torch (2.14.0+cu132) ``CosineAnnealingLR`` state has
    an EXACT key set; ``verbose`` is not in it.  A mapping carrying an extra key
    is a foreign/other-version schema and must fail closed rather than be
    silently accepted as an exact continuation.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)
    parent = _scheduler_partial_parent(
        tmp_path, ds, name="extraschedkey_crash", mutate="extra_key",
    )
    _assert_scheduler_resume_fails_closed(
        tmp_path, ds, parent, name="extraschedkey_resume",
    )


def test_real_scheduler_missing_production_key_fails_closed(
    tmp_path: Path,
) -> None:
    """A missing real production field (``_is_initial``) must NOT resume.

    Reviewer B: the complete production schema includes the step-state flags
    (``_step_count`` / ``_get_lr_called_within_step`` / ``_is_initial``).  A
    state missing one of them is incomplete and must fail closed.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)
    parent = _scheduler_partial_parent(
        tmp_path, ds, name="missingschedkey_crash", mutate="missing_key",
    )
    _assert_scheduler_resume_fails_closed(
        tmp_path, ds, parent, name="missingschedkey_resume",
    )


def test_real_current_schema_complete_scheduler_state_still_resumes(
    tmp_path: Path,
) -> None:
    """A COMPLETE scheduler state must still resume (no false positive).

    The full ``CosineAnnealingLR`` state from the real producer must pass the
    new structure check and the run must finalize normally — the guard rejects
    damaged state, not valid continuations.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    src_cfg = _run_publication_crash(
        tmp_path, ds, epochs=3, name="completesched_crash",
        crash_after="epoch_3.pth", patience=2,
    )
    parent = Path(src_cfg.checkpoint.output_dir) / "epoch_3.pth"
    res_cfg = _real_config(tmp_path, epochs=3, name="completesched_resume", patience=2)
    res_cfg.checkpoint.resume_from = str(parent)
    result = trainer.train(
        res_cfg, dataset_path=str(ds),
        trusted_historical_checkpoint_sha256=_existing_shas(
            parent, parent.parent / "best_model.pth",
        ),
    )
    final = Path(res_cfg.checkpoint.output_dir) / "final_model.pth"
    assert final.exists()
    # The budget-terminal resume finalizes at the parent's executed epoch with
    # its continuation state intact — no fresh-schedule restart.
    assert result["history_start_epoch"] == 0


def test_legacy_parent_without_scheduler_state_still_resumes(
    tmp_path: Path,
) -> None:
    """A LEGACY parent (no scheduler block) keeps its limited fallback.

    The structure check is scoped to the CURRENT schema: a legacy checkpoint
    without ``recovery_state_version`` is not examined here and must still
    resume with the loud restart warning, exactly as before this repair.
    """
    from scripts.train import train

    ds = _make_synthetic_dataset(tmp_path)
    source = _make_config(tmp_path, epochs=1, warmup_epochs=0)
    source.training.checkpoint_interval = 1
    with mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})):
        train(source, dataset_path=str(ds))

    source_dir = Path(source.checkpoint.output_dir)
    periodic = source_dir / "epoch_1.pth"
    legacy = load_artifact_bytes(periodic.read_bytes())
    for key in (
        "recovery_state_version", "epochs_without_improvement",
        "training_history", "history_start_epoch", "numpy_rng_state",
        "scheduler_state_dict",
    ):
        legacy.pop(key, None)
    torch.save(legacy, periodic)

    resumed = _make_config(tmp_path, epochs=2, warmup_epochs=0)
    resumed.checkpoint.output_dir = str(tmp_path / "legacy_sched_resume")
    resumed.checkpoint.resume_from = str(periodic)
    with (
        mock.patch("scripts.train.train_one_epoch", return_value=(1.0, {})),
        mock.patch("scripts.train.validate", return_value=0.5),
    ):
        result = train(
            resumed, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_checkpoint_shas(
                periodic, source_dir / "best_model.pth"),
        )
    assert result["history_start_epoch"] == 1


# ═════════════════════════════════════════════════════════════
# G2: train.py itself refuses a non-newest-complete-epoch resume source
# ═════════════════════════════════════════════════════════════
#
# §11.1 authorizes resuming only from a run's LAST complete-epoch checkpoint.
# The engine layer (not only the launcher helper) must enforce that: a bare
# ``train.py --resume`` cannot pick ``best_model.pth`` (the best SELECTED epoch,
# possibly far older) or a stale periodic when a newer last-epoch file exists.


def test_resume_from_best_model_is_refused_by_engine(tmp_path: Path) -> None:
    """A bare train.py --resume from best_model.pth fails closed before an epoch.

    ``best_model.pth`` is the best SELECTED epoch, which can be far older than
    the last executed epoch; resuming from it would silently re-train completed
    epochs.  The engine refuses it regardless of directory contents.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    source = _real_config(tmp_path, epochs=2, name="g2_src")
    with mock.patch.object(trainer, "validate", return_value=0.5):
        trainer.train(source, dataset_path=str(ds))
    src_dir = Path(source.checkpoint.output_dir)
    best = src_dir / "best_model.pth"
    assert best.exists()

    resumed = _real_config(tmp_path, epochs=3, name="g2_resume")
    resumed.checkpoint.resume_from = str(best)
    executed: list[int] = []

    def hook(**kwargs: Any) -> Any:
        executed.append(kwargs["epoch"])
        return 1.0, {}

    with mock.patch.object(trainer, "train_one_epoch", side_effect=hook):
        with pytest.raises(ValueError, match="best_model.pth is the best SELECTED epoch"):
            trainer.train(
                resumed, dataset_path=str(ds),
                trusted_historical_checkpoint_sha256=_existing_shas(
                    best, src_dir / "final_model.pth"),
            )
    assert executed == [], f"a refused resume executed epochs: {executed}"


def test_resume_from_stale_periodic_is_refused_by_engine(tmp_path: Path) -> None:
    """An in-place resume from an older periodic is refused when a newer exists.

    A SIGKILL at a non-multiple of ``checkpoint_interval`` can leave a newer
    ``final_model.pth`` and an older ``epoch_*.pth`` in the same run dir.
    Resuming IN PLACE (output_dir == the run's own dir) from the older periodic
    would re-train completed epochs, so it is refused.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    source = _real_config(tmp_path, epochs=3, name="g2_stale_src")
    source.training.checkpoint_interval = 1
    with mock.patch.object(trainer, "validate", return_value=0.5):
        trainer.train(source, dataset_path=str(ds))
    src_dir = Path(source.checkpoint.output_dir)
    older = src_dir / "epoch_1.pth"          # stored epoch 0
    newer = src_dir / "final_model.pth"      # stored epoch 2
    assert older.exists() and newer.exists()

    # IN-PLACE resume: same output dir as the run, older source.
    resumed = _real_config(tmp_path, epochs=5, name="g2_stale_src")
    resumed.checkpoint.resume_from = str(older)
    with pytest.raises(ValueError, match="newer complete-epoch checkpoint"):
        trainer.train(
            resumed, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                older, src_dir / "best_model.pth"),
        )

    # The NEWEST source (final_model.pth) in the same dir IS accepted.
    ok = _real_config(tmp_path, epochs=5, name="g2_stale_src")
    ok.checkpoint.resume_from = str(newer)
    with mock.patch.object(trainer, "validate", return_value=0.5):
        result = trainer.train(
            ok, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=_existing_shas(
                newer, src_dir / "best_model.pth"),
        )
    assert result is not None
