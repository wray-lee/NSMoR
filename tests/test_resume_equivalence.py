"""Deterministic resume-equivalence tests (formal epoch-recovery, S1).

Proves, on CPU with a fixed seed and tiny synthetic data, that resuming from the
last COMPLETE-epoch checkpoint reproduces uninterrupted training **bitwise**
across every continuation-bearing quantity the task enumerates:

* model weights; optimizer state; LR scheduler state;
* NumPy and Torch RNG states;
* the per-epoch train-shuffle index sequences (the DataLoader sampler order);
* the early-stopping counter, the best metric and the best-checkpoint record
  (best_model.pth compared byte-for-byte);
* the phase-2 ``selection_metric`` and ``zero_input_channels`` provenance;
* the per-epoch logged train/val losses.

The main harness runs the REAL ``train()`` (real model, forward, backward,
optimizer, scheduler, validation) with ``num_workers=0`` and
``checkpoint_interval=1`` — the reliable cadence.  The anchor-aligned crop this
fixture takes does NOT consume the global NumPy stream, so each epoch wrapper
performs one supported stochastic draw (a uniform draw and a Gaussian draw) —
exactly the per-epoch consumption the recovery seam exists to preserve.  A
resume that fails to restore ``np.random`` state therefore produces a different
subsequent stream and the NumPy equality assertion FAILS (verified by mutation:
returning ``np_state=None`` from the restore seam makes this module red).

The Python **stdlib** RNG is deliberately NOT part of this claim: training never
consumes ``random.*`` and no checkpoint stores its state, so it is not a
continuation-bearing quantity and is not asserted here.

GPU is deliberately not exercised here: resume on GPU is only statistically
equivalent, never bitwise (cuDNN/cuBLAS atomics, non-deterministic kernels).
The tolerance statement lives in the phase-2 protocol, section 11.

The SAME bitwise contract holds in the REAL phase-2 arm regime —
``num_workers>0`` (auto), ``persistent_workers=True``, ``checkpoint_interval=10``
— because a resumed run re-establishes the persistent loader iterator around a
save/restore of the global RNG (``_prewarm_persistent_loaders``), so it
consumes the global stream exactly as the uninterrupted run did at that epoch
boundary.  Without that fix the first-construction path draws an extra base
seed and re-seeds the sampler, shifting the whole resumed trajectory.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest
import torch
from unittest import mock

from nsmor.pipeline.nested_prior import load_artifact_bytes
from tests.test_train_checkpoint import _make_synthetic_dataset
from tests.test_epoch_recovery import (
    _TestInterruption,
    _assert_model_equal,
    _assert_optimizer_equal,
    _final_ckpt,
    _load_state,
    _numpy_state_and_streams,
    _real_config,
    _sched_state,
)

_NP_DRAW = 8


@pytest.fixture(scope="module", autouse=True)
def _cpu_single_thread() -> Any:
    """Pin to single-thread CPU and refuse CUDA (the claim is CPU-only)."""
    assert not torch.cuda.is_available(), (
        "resume-equivalence tests must run on CPU; CUDA is visible, so the "
        "bitwise claim would be measured on the wrong device. "
        "Run with CUDA_VISIBLE_DEVICES=''."
    )
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    yield
    torch.set_num_threads(prev)


@pytest.fixture(autouse=True)
def _restore_trainer_globals() -> Any:
    """Save/restore the script-owned phase-2 globals around each test."""
    from scripts import train as trainer

    saved = (trainer._SELECTION_METRIC, trainer._ZERO_INPUT_CHANNELS)
    yield
    trainer._SELECTION_METRIC, trainer._ZERO_INPUT_CHANNELS = saved


@contextmanager
def _record_train_indices(sink: List[int]) -> Any:
    """Record the dataset indices the loader fetches, in fetch order.

    With ``num_workers=0`` and no prefetch reordering, the fetch order IS the
    sampler's shuffle order for the epoch.  Patches the class method so the
    order is captured without touching the RNG stream.
    """
    from nsmor.nsmor_dataloader import NSMoRDataset

    orig = NSMoRDataset.__getitem__

    def rec(self: Any, idx: int) -> Any:
        sink.append(int(idx))
        return orig(self, idx)

    NSMoRDataset.__getitem__ = rec  # type: ignore[assignment]
    try:
        yield
    finally:
        NSMoRDataset.__getitem__ = orig  # type: ignore[assignment]


def _index_recording_epoch(
    real: Any, sink: List[int], epochs_idx: List[List[int]],
    cut: Optional[int] = None,
) -> Any:
    """Wrap the real ``train_one_epoch`` to snapshot its index sequence.

    Clears the sink, runs the real epoch, then records the fetched indices
    BEFORE validation runs (validation uses a different, unshuffled loader).
    A ``cut`` simulates a SIGKILL/OOM at the START of epoch ``cut`` — the
    prior epoch completed and its checkpoint is on disk.

    After the real epoch, one supported per-epoch stochastic draw consumes the
    global NumPy stream.  The anchor-aligned crop this fixture takes never
    touches that stream, so without this draw the NumPy-restore assertion would
    compare two never-advanced (trivially equal) states and would NOT fail on a
    real gap.  Drawing here — before the loop writes the checkpoint — makes
    every saved checkpoint carry the post-draw stream, so an unrestored resume
    diverges and the assertion catches it.
    """
    def hook(**kwargs: Any) -> Any:
        if cut is not None and kwargs["epoch"] >= cut:
            raise _TestInterruption()
        sink.clear()
        out = real(**kwargs)
        epochs_idx.append(list(sink))
        np.random.random_sample(_NP_DRAW)
        return out

    return hook


@contextmanager
def _record_sampler_order(sink: List[List[int]]) -> Any:
    """Record the PARENT-side ``RandomSampler`` order for each ``iter(loader)``.

    With ``num_workers>0`` the dataset ``__getitem__`` runs in worker
    processes, so ``_record_train_indices`` (which patches ``__getitem__``)
    sees nothing.  The sampler, however, is built and iterated in the PARENT
    process: ``iter(loader)`` calls ``RandomSampler.__iter__``, which draws the
    per-epoch seed from the global torch RNG and yields the shuffle.  This
    materializes that order and appends it to *sink* (returning an equivalent
    iterator so the loader is unaffected), capturing the per-epoch shuffle
    parent-side for ANY worker count.  Materializing does not advance the
    global RNG beyond the single seed the sampler draws anyway.
    """
    from torch.utils.data import RandomSampler

    orig = RandomSampler.__iter__

    def rec(self: Any) -> Any:
        order = list(orig(self))
        sink.append(order)
        return iter(order)

    RandomSampler.__iter__ = rec  # type: ignore[assignment]
    try:
        yield
    finally:
        RandomSampler.__iter__ = orig  # type: ignore[assignment]


def _sampler_order_epoch(
    real: Any, sink: List[List[int]], epochs_order: List[List[int]],
) -> Any:
    """Wrap ``train_one_epoch`` to snapshot the parent-side sampler order.

    The sink is cleared at entry (discarding any pre-warm ``iter(loader)`` the
    resume path performed before the loop), then the wrapped epoch runs and its
    single per-epoch sampler order is recorded.  Works at any worker count
    because it observes the parent-process ``RandomSampler`` (not the worker
    dataset fetch order).  ``real`` may itself be a pause/cut wrapper.
    """
    def hook(**kwargs: Any) -> Any:
        sink.clear()
        out = real(**kwargs)
        assert len(sink) == 1, (
            f"expected exactly one parent-side sampler order per epoch, got {len(sink)}"
        )
        epochs_order.append(list(sink[0]))
        return out

    return hook


def _run_segment(
    config: Any, ds: Path, *, cut: Optional[int] = None,
    selection_metric: str = "total", zic: tuple = (),
    trusted: Any = None,
) -> Tuple[Optional[Any], List[List[int]], Path]:
    """Run one real segment; return (result|None, index seqs, outdir).

    ``result`` is ``None`` when the segment was interrupted by ``cut``.
    """
    from scripts import train as trainer

    trainer._SELECTION_METRIC = selection_metric
    trainer._ZERO_INPUT_CHANNELS = tuple(zic)

    sink: List[int] = []
    epochs_idx: List[List[int]] = []
    real = trainer.train_one_epoch
    with (
        _record_train_indices(sink),
        mock.patch.object(
            trainer, "train_one_epoch",
            side_effect=_index_recording_epoch(real, sink, epochs_idx, cut=cut),
        ),
    ):
        try:
            result = trainer.train(
                config, dataset_path=str(ds),
                trusted_historical_checkpoint_sha256=trusted,
            )
        except _TestInterruption:
            result = None
    return result, epochs_idx, Path(config.checkpoint.output_dir)


def _run_full(
    tmp_path: Path, ds: Path, *, epochs: int, name: str,
    selection_metric: str = "total", zic: tuple = (),
) -> Any:
    """Uninterrupted run; return (config, result, index seqs, outdir)."""
    config = _real_config(tmp_path, epochs=epochs, name=name)
    result, idx, outdir = _run_segment(
        config, ds, selection_metric=selection_metric, zic=zic,
    )
    assert result is not None, "uninterrupted run must complete"
    return config, result, idx, outdir


def _run_chain(
    tmp_path: Path, ds: Path, *, total_epochs: int, cut: int, name_prefix: str,
    selection_metric: str = "total", zic: tuple = (),
) -> Any:
    """Interrupt once at ``cut`` and resume to ``total_epochs`` (same budget)."""
    seg0 = _real_config(tmp_path, epochs=total_epochs, name=f"{name_prefix}_seg0")
    _, idx0, outdir0 = _run_segment(
        seg0, ds, cut=cut, selection_metric=selection_metric, zic=zic,
    )
    resume_from = outdir0 / f"epoch_{cut}.pth"
    best0 = outdir0 / "best_model.pth"
    assert resume_from.exists(), f"cut epoch {cut} wrote no periodic checkpoint"
    trusted = tuple(
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (resume_from, best0) if p.exists()
    )

    final = _real_config(tmp_path, epochs=total_epochs, name=f"{name_prefix}_final")
    final.checkpoint.resume_from = str(resume_from)
    result, idx1, outdir1 = _run_segment(
        final, ds, selection_metric=selection_metric, zic=zic, trusted=trusted,
    )
    assert result is not None, "resumed segment must complete"
    return final, result, idx0 + idx1, outdir1


def _best_record(outdir: Path) -> Optional[dict]:
    """The best checkpoint's semantic record (run-identity paths excluded).

    ``best_model.pth`` embeds the saved ``config`` whose ``checkpoint.
    output_dir`` is the RUN's own output path and whose ``checkpoint.
    resume_from`` is the segment's own resume source — both run-identity
    values, not training state — so the raw file bytes legitimately differ
    between the uninterrupted and resumed arms (they write to different dirs
    and only the resumed one has a ``resume_from``).  The best-SELECTION record
    that must match is the weights, the selected epoch and loss, and the
    continuation state; that is what this returns.
    """
    best = outdir / "best_model.pth"
    if not best.exists():
        return None
    ckpt = load_artifact_bytes(best.read_bytes(), map_location="cpu")
    cfg = ckpt["config"]
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    cfg.get("checkpoint", {}).pop("output_dir", None)
    cfg.get("checkpoint", {}).pop("resume_from", None)
    return {
        "model_state_dict": ckpt["model_state_dict"],
        "optimizer_state_dict": ckpt["optimizer_state_dict"],
        "scheduler_state_dict": ckpt["scheduler_state_dict"],
        "rng_state": ckpt["rng_state"],
        "numpy_rng_state": ckpt["numpy_rng_state"],
        "epoch": ckpt["epoch"],
        "val_loss": ckpt["val_loss"],
        "best_val_loss": ckpt["best_val_loss"],
        "train_loss": ckpt["train_loss"],
        "training_history": ckpt["training_history"],
        "epochs_without_improvement": ckpt["epochs_without_improvement"],
        "selection_metric": ckpt["selection_metric"],
        "zero_input_channels": ckpt["zero_input_channels"],
        "config": cfg,
    }


def _assert_full_equivalence(
    full: Any, chain: Any, *, selection_metric: str, zic: tuple,
    check_indices: bool = True,
) -> None:
    """Compare every continuation-bearing quantity the task enumerates.

    ``check_indices`` is False in the arm regime (``num_workers>0``): the fetch
    order is produced in WORKER processes, so the parent-side index recorder
    sees nothing.  The state equality (weights, optimizer, scheduler, both RNG
    streams, patience, best record, logged losses) still fully characterizes the
    trajectory, so the bitwise claim holds without the index sequence.
    """
    full_cfg, full_res, full_idx, full_dir = full
    chain_cfg, chain_res, chain_idx, chain_dir = chain

    full_final = _final_ckpt(full_cfg)
    chain_final = _final_ckpt(chain_cfg)

    # 1. Model weights (bitwise).
    _assert_model_equal(full_final["model_state_dict"], chain_final["model_state_dict"])
    # 2. Optimizer state + groups (bitwise).
    _assert_optimizer_equal(
        full_final["optimizer_state_dict"], chain_final["optimizer_state_dict"],
    )
    # 3. LR scheduler state (bitwise).
    assert _sched_state(full_final) == _sched_state(chain_final), "scheduler differs"
    # 4. Torch RNG (bitwise bytes).
    assert torch.equal(full_final["rng_state"], chain_final["rng_state"]), "torch RNG differs"
    # 5. NumPy RNG (full decoded MT19937 state + uniform/gaussian continuations).
    fnp, cnp = _numpy_state_and_streams(full_final), _numpy_state_and_streams(chain_final)
    for field in ("kind", "keys", "pos", "has_gauss", "cached_gaussian",
                  "uniform", "gaussian"):
        assert fnp[field] == cnp[field], f"numpy {field!r} differs"
    # 6. Per-epoch train-shuffle index sequences (parent-observable at
    #    num_workers=0; produced in worker processes otherwise, so skipped).
    if check_indices:
        assert full_idx == chain_idx, (
            f"per-epoch shuffle index sequences differ: {full_idx} != {chain_idx}"
        )
        assert len(full_idx) == full_cfg.training.num_epochs, (
            "did not record one index sequence per executed epoch"
        )
    # 7. Early-stopping counter, best metric, best-checkpoint record.
    assert (
        full_final["epochs_without_improvement"]
        == chain_final["epochs_without_improvement"]
    ), "patience counter differs"
    assert full_res["best_val_loss"] == chain_res["best_val_loss"], "best metric differs"
    full_best, chain_best = _best_record(full_dir), _best_record(chain_dir)
    assert full_best is not None, "no best checkpoint was written"
    assert chain_best is not None, "no best checkpoint was written"
    assert full_best.keys() == chain_best.keys()
    for key in full_best:
        if key == "model_state_dict":
            _assert_model_equal(full_best[key], chain_best[key])
        elif key == "optimizer_state_dict":
            _assert_optimizer_equal(full_best[key], chain_best[key])
        elif key == "numpy_rng_state":
            fb_rng, cb_rng = full_best[key], chain_best[key]
            assert fb_rng["bit_generator"] == cb_rng["bit_generator"]
            assert fb_rng["pos"] == cb_rng["pos"]
            assert fb_rng["has_gauss"] == cb_rng["has_gauss"]
            assert fb_rng["cached_gaussian"] == cb_rng["cached_gaussian"]
            assert torch.equal(fb_rng["keys"], cb_rng["keys"]), "best numpy keys differ"
        elif isinstance(full_best[key], torch.Tensor):
            assert torch.equal(full_best[key], chain_best[key]), f"best {key} differs"
        else:
            assert full_best[key] == chain_best[key], f"best record {key} differs"
    # 9. Phase-2 provenance recorded in the checkpoint.
    assert full_final["selection_metric"] == chain_final["selection_metric"] == selection_metric
    assert sorted(full_final["zero_input_channels"]) == sorted(
        chain_final["zero_input_channels"]
    ) == sorted(zic)
    # 10. Per-epoch logged losses.
    assert full_res["history"]["train_loss"] == chain_res["history"]["train_loss"]
    assert full_res["history"]["val_loss"] == chain_res["history"]["val_loss"]
    assert full_final["training_history"] == chain_final["training_history"]


@pytest.mark.parametrize(
    "selection_metric,zic,label",
    [
        ("total", (), "total-unablated"),
        ("mse", (2, 3), "mse-r2-history-ablated"),
    ],
)
def test_resume_matches_uninterrupted_full_state(
    tmp_path: Path, selection_metric: str, zic: tuple, label: str,
) -> None:
    """Interrupt once at N/2 and resume == uninterrupted, across every quantity."""
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    full = _run_full(
        tmp_path, ds, epochs=4, name=f"full_{label}",
        selection_metric=selection_metric, zic=zic,
    )
    chain = _run_chain(
        tmp_path, ds, total_epochs=4, cut=2, name_prefix=f"chain_{label}",
        selection_metric=selection_metric, zic=zic,
    )
    _assert_full_equivalence(
        full, chain, selection_metric=selection_metric, zic=zic,
    )


def test_resume_refuses_switched_selection_metric(tmp_path: Path) -> None:
    """A resume that switches ``selection_metric`` fails closed before an epoch."""
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)
    config = _real_config(tmp_path, epochs=2, name="sel_parent")
    _run_segment(config, ds, selection_metric="total", zic=())
    periodic = Path(config.checkpoint.output_dir) / "epoch_1.pth"

    from scripts import train as trainer

    resume = _real_config(tmp_path, epochs=2, name="sel_resume")
    resume.checkpoint.resume_from = str(periodic)
    trainer._SELECTION_METRIC = "mse"
    trainer._ZERO_INPUT_CHANNELS = ()
    with mock.patch.object(trainer, "train_one_epoch", side_effect=AssertionError("epoch ran")):
        with pytest.raises(ValueError, match="selection_metric"):
            trainer.train(resume, dataset_path=str(ds))


def test_safe_pause_stops_at_epoch_boundary_and_resumes(tmp_path: Path) -> None:
    """A STOP sentinel pauses at a checkpointed epoch boundary; resume continues.

    The pause must stop AFTER the periodic checkpoint of the paused epoch is
    fully written (so ``epoch_N.pth`` exists), and the resumed run must produce
    the same final state as an uninterrupted run — a pause is a pause, not a
    divergence.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    # Uninterrupted 4-epoch reference.
    full = _run_full(tmp_path, ds, epochs=4, name="pause_full")
    assert full[1]["paused"] is False, "an uninterrupted run must not report paused"

    # Paused run: the sentinel appears after epoch 0 completes, so the loop
    # must stop at the END of epoch 0 — after epoch_1.pth is written.
    paused_dir = tmp_path / "pause_run"
    config = _real_config(tmp_path, epochs=4, name="pause_run")
    real = trainer.train_one_epoch

    def pause_after_first(**kwargs: Any) -> Any:
        out = real(**kwargs)
        # Match the uninterrupted arm's per-epoch NumPy draw so the paused
        # segment's RNG stream is comparable; then request the pause.
        np.random.random_sample(_NP_DRAW)
        (paused_dir / trainer._PAUSE_SENTINEL_NAME).write_text("")
        return out

    with mock.patch.object(trainer, "train_one_epoch", side_effect=pause_after_first):
        result = trainer.train(config, dataset_path=str(ds))
    assert result["paused"] is True, "the run must report it paused on the sentinel"
    assert (paused_dir / "epoch_1.pth").exists(), (
        "the pause must stop AFTER the epoch-1 checkpoint is written"
    )
    assert not (paused_dir / "epoch_2.pth").exists(), "the pause did not stop at the boundary"
    # The sentinel is left in place for the operator/launcher, not consumed by
    # the paused segment itself.
    assert (paused_dir / trainer._PAUSE_SENTINEL_NAME).exists()

    # Resume from the paused segment's last complete-epoch checkpoint, writing
    # back to the SAME output dir (as the real launcher does), so the STOP
    # sentinel is visible to the resumed run.
    periodic = paused_dir / "epoch_1.pth"
    trusted = tuple(
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (periodic, paused_dir / "best_model.pth") if p.exists()
    )
    resume = _real_config(tmp_path, epochs=4, name="pause_run")
    resume.checkpoint.resume_from = str(periodic)
    _, _, resumed_dir = _run_segment(resume, ds, trusted=trusted)

    # A direct resume (no launcher) must CONSUME the stale sentinel, or the
    # resumed run would re-pause after one epoch.
    assert not (resumed_dir / trainer._PAUSE_SENTINEL_NAME).exists(), (
        "a resume must consume the STOP sentinel so it does not re-pause"
    )

    full_final = _final_ckpt(full[0])
    resumed_final = _load_state(resumed_dir / "final_model.pth")
    # A pause must be a pause, not a divergence: assert the FULL continuation
    # state, not just weights.  The pause path shares the resume seam, so the
    # same bitwise contract holds across a pause as across an interruption.
    _assert_model_equal(
        full_final["model_state_dict"], resumed_final["model_state_dict"],
    )
    _assert_optimizer_equal(
        full_final["optimizer_state_dict"], resumed_final["optimizer_state_dict"],
    )
    assert _sched_state(full_final) == _sched_state(resumed_final), "scheduler differs"
    assert torch.equal(
        full_final["rng_state"], resumed_final["rng_state"],
    ), "torch RNG differs across a pause"
    fnp, rnp = _numpy_state_and_streams(full_final), _numpy_state_and_streams(resumed_final)
    for field in ("kind", "keys", "pos", "has_gauss", "cached_gaussian",
                  "uniform", "gaussian"):
        assert fnp[field] == rnp[field], f"numpy {field!r} differs across a pause"
    assert (
        full_final["epochs_without_improvement"]
        == resumed_final["epochs_without_improvement"]
    ), "patience counter differs across a pause"
    assert full_final["training_history"] == resumed_final["training_history"], (
        "training history differs across a pause"
    )
    # The best-checkpoint RECORD must match across a pause too, not just be
    # implied: a pause is a pause, so the selected best weights/epoch/loss and
    # the restored continuation state in best_model.pth must be bitwise equal.
    full_best = _best_record(full[3])
    resumed_best = _best_record(resumed_dir)
    assert full_best is not None and resumed_best is not None, "no best checkpoint"
    for key in full_best:
        if key == "model_state_dict":
            _assert_model_equal(full_best[key], resumed_best[key])
        elif key == "optimizer_state_dict":
            _assert_optimizer_equal(full_best[key], resumed_best[key])
        elif key == "numpy_rng_state":
            fb_rng, rb_rng = full_best[key], resumed_best[key]
            assert fb_rng["pos"] == rb_rng["pos"]
            assert fb_rng["has_gauss"] == rb_rng["has_gauss"]
            assert fb_rng["cached_gaussian"] == rb_rng["cached_gaussian"]
            assert torch.equal(fb_rng["keys"], rb_rng["keys"]), "best numpy keys differ"
        elif isinstance(full_best[key], torch.Tensor):
            assert torch.equal(full_best[key], resumed_best[key]), f"best {key} differs"
        else:
            assert full_best[key] == resumed_best[key], f"best record {key} differs"


def test_paused_run_exits_with_distinct_code(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """A paused run exits with the distinct PAUSE_EXIT_CODE, a normal one does not.

    The launcher must be able to tell a safe pause from a completion (0) or a
    failure (nonzero) from the exit status alone (protocol §11.1 / S4).
    """
    from scripts import train as trainer

    config = _real_config(tmp_path, epochs=2, name="main_pause")
    Path(config.checkpoint.output_dir).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        trainer, "build_config", lambda argv=None: (config, 0.0, None),
    )
    monkeypatch.setattr(trainer, "train", lambda *a, **k: {"paused": True})
    with pytest.raises(SystemExit) as excinfo:
        trainer.main([])
    assert excinfo.value.code == trainer.PAUSE_EXIT_CODE

    # A non-paused run returns normally (no SystemExit) — unchanged behavior.
    monkeypatch.setattr(trainer, "train", lambda *a, **k: {"paused": False})
    assert trainer.main([]) is None


def _weights_sha(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state["model_state_dict"]):
        h.update(key.encode())
        h.update(state["model_state_dict"][key].cpu().numpy().tobytes())
    return h.hexdigest()


# Fresh-training fingerprint (S2): the exact final-weights SHA-256 a fresh
# 6-epoch run on the config in ``test_fresh_training_weights_are_unchanged``
# produces.  It is identical on the pre-amendment base (8ed5edb) and on this
# code (verified with an out-of-tree probe), so a change that perturbs fresh
# (non-resumed) training — breaking comparability with the phase-2 runs already
# started on the base commit — turns this test red.
_FRESH_WEIGHTS_SHA = (
    "6a8a43517d1985a5c3732df026cdb39c0d097c3389fab8d797e08bd2a72052a0"
)


def test_fresh_training_weights_are_unchanged(tmp_path: Path) -> None:
    """Fresh (non-resumed) training is bitwise pinned (S2 regression guard).

    The amendment claims 'no default behaviour changed'.  This asserts the
    exact final-weights SHA-256 of a fresh run, which was measured equal on the
    pre-amendment base commit (8ed5edb) and on this code, so a future change
    cannot silently break the fresh-training comparability the amendment rests
    on.  The config is representative (dropout + sensory noise so the RNG
    stream advances; workers=1, interval=2) and deterministic on CPU.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)
    config = _real_config(tmp_path, epochs=6, name="fresh_pin")
    config.training.num_workers = 1
    config.training.persistent_workers = True
    config.training.checkpoint_interval = 2

    from scripts import train as trainer

    real = trainer.train_one_epoch

    def hook(**kwargs: Any) -> Any:
        out = real(**kwargs)
        np.random.random_sample(_NP_DRAW)
        return out

    with mock.patch.object(trainer, "train_one_epoch", side_effect=hook):
        trainer.train(config, dataset_path=str(ds))

    ckpt = _load_state(Path(config.checkpoint.output_dir) / "final_model.pth")
    assert _weights_sha(ckpt) == _FRESH_WEIGHTS_SHA, (
        "fresh training weights changed; the amendment's fresh-training "
        "equivalence claim no longer holds for the base commit 8ed5edb."
    )


def _arm_config(
    tmp_path: Path, *, epochs: int, name: str, interval: int,
    workers: int, persistent: bool, batch_size: int = 4,
) -> Any:
    """A config in the PHASE-2 ARM regime (workers>0, checkpoint_interval=10).

    ``batch_size`` defaults to 4 (< the 8-trial train split), so each epoch
    yields TWO train batches.  A single-batch epoch would let a wrong
    per-epoch shuffle order go unobserved (one batch consumes the whole
    sampler), so the shuffle-order assertion needs >= 2 batches to be
    non-vacuous.
    """
    config = _real_config(tmp_path, epochs=epochs, name=name)
    config.training.num_workers = workers
    config.training.persistent_workers = persistent
    config.training.checkpoint_interval = interval
    config.training.batch_size = batch_size
    return config


def _run_arm_segment(
    config: Any, ds: Path, *, pause_after: Optional[int] = None, trusted: Any = None,
    selection_metric: str = "total", zic: tuple = (),
) -> Any:
    """Run a segment in the arm regime, bypassing the interval==1 worker guard.

    The guard refuses workers>0 at ``checkpoint_interval==1`` because the loader
    iterator state was not captured; the fix now re-establishes the persistent
    iterator around an RNG save/restore, so the regime IS reliable.  This helper
    measures that, so it reaches the regime.  All other preflight (lineage,
    clock, schema) is unchanged.

    ``pause_after`` drops the STOP sentinel once that epoch completes, so the
    segment pauses CLEANLY at the epoch boundary and writes ``final_model.pth``
    — the arm regime's last complete-epoch checkpoint at ``checkpoint_interval
    = 10`` (an exception-based kill writes no ``final_model.pth``, and interval
    10 writes no periodic within a short budget).  A pause is a pause: the
    segment's continuation state is the boundary state, so resuming it must be
    bitwise.

    Returns ``(result, outdir, epochs_order)`` where ``epochs_order`` is the
    PARENT-side ``RandomSampler`` shuffle for each executed epoch (recorded via
    :func:`_record_sampler_order`), so the per-epoch shuffle order can be
    compared even at ``num_workers>0`` (where the dataset fetch order lives in
    worker processes).
    """
    from scripts import train as trainer

    trainer._SELECTION_METRIC = selection_metric
    trainer._ZERO_INPUT_CHANNELS = tuple(zic)
    real = trainer.train_one_epoch
    outdir = Path(config.checkpoint.output_dir)
    sink: List[List[int]] = []
    epochs_order: List[List[int]] = []

    def pause_hook(**kwargs: Any) -> Any:
        out = real(**kwargs)
        np.random.random_sample(_NP_DRAW)
        if pause_after is not None and kwargs["epoch"] == pause_after:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / trainer._PAUSE_SENTINEL_NAME).write_text("")
        return out

    with (
        _record_sampler_order(sink),
        mock.patch.object(
            trainer, "train_one_epoch",
            side_effect=_sampler_order_epoch(pause_hook, sink, epochs_order),
        ),
        mock.patch.object(
            trainer, "_require_reliable_recovery_loaders", lambda *a, **k: None,
        ),
    ):
        result = trainer.train(
            config, dataset_path=str(ds),
            trusted_historical_checkpoint_sha256=trusted,
        )
    return result, outdir, epochs_order


@pytest.mark.parametrize(
    "workers,selection_metric,zic,label",
    [
        (1, "total", (), "workers1-total"),
        (1, "mse", (2, 3), "workers1-mse-r2"),
        (0, "total", (), "workers0-total"),
        (0, "mse", (2, 3), "workers0-mse-r2"),
    ],
)
def test_arm_regime_resume_is_bitwise(
    tmp_path: Path, workers: int, selection_metric: str, zic: tuple, label: str,
) -> None:
    """F3: resume == uninterrupted BITWISE in the real arm regime.

    The twelve phase-2 arms run ``num_workers: -1`` (auto >0 on the full corpus;
    an explicit ``1`` here reaches the same multi-worker path the tiny fixture
    would otherwise auto-scale below), ``persistent_workers: true`` and
    ``checkpoint_interval: 10``, on CPU.  A resumed segment must reproduce the
    uninterrupted run's every continuation-bearing quantity — weights,
    optimizer, scheduler, both RNG streams, patience counter, best record and
    logged losses — because the resumed process re-establishes the persistent
    loader iterator around an RNG save/restore, consuming the global stream
    exactly as the uninterrupted run did at that epoch boundary.  Exercised for
    ``selection_metric=total`` and an R2-style ``mse`` + ``zero_input_channels``
    config, at ``num_workers=1`` and ``num_workers=0`` (both at interval 10).

    The per-epoch shuffle order IS checked, parent-side, at every worker count
    (``_record_sampler_order`` observes the parent-process ``RandomSampler``),
    and the fixture uses ``batch_size=4`` on an 8-trial split so each epoch has
    TWO batches — a single-batch epoch would let a wrong shuffle order go
    unobserved.
    """
    ds = _make_synthetic_dataset(tmp_path)
    torch.manual_seed(0)

    full_cfg = _arm_config(
        tmp_path, epochs=4, name=f"arm_full_{label}", interval=10, workers=workers,
        persistent=workers > 0,
    )
    full_res, full_dir, full_order = _run_arm_segment(
        full_cfg, ds, selection_metric=selection_metric, zic=zic,
    )
    assert full_res is not None and full_res["paused"] is False

    # Interrupt by SAFE PAUSE at the epoch-1 boundary (writes final_model.pth).
    seg0 = _arm_config(
        tmp_path, epochs=4, name=f"arm_seg0_{label}", interval=10, workers=workers,
        persistent=workers > 0,
    )
    seg0_res, out0, seg0_order = _run_arm_segment(
        seg0, ds, pause_after=1, selection_metric=selection_metric, zic=zic,
    )
    assert seg0_res is not None and seg0_res["paused"] is True
    resume_from = out0 / "final_model.pth"
    assert resume_from.exists(), "the paused segment must write final_model.pth"
    trusted = tuple(
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (resume_from, out0 / "best_model.pth") if p.exists()
    )
    final_cfg = _arm_config(
        tmp_path, epochs=4, name=f"arm_final_{label}", interval=10, workers=workers,
        persistent=workers > 0,
    )
    final_cfg.checkpoint.resume_from = str(resume_from)
    res, final_dir, final_order = _run_arm_segment(
        final_cfg, ds, trusted=trusted, selection_metric=selection_metric, zic=zic,
    )
    assert res is not None

    # The arm regime must reproduce the uninterrupted trajectory BITWISE,
    # INCLUDING the per-epoch shuffle order (captured parent-side, so it is
    # observable at workers>0 too).  Two batches per epoch make a wrong order
    # observable rather than swallowed by a single batch.
    _assert_full_equivalence(
        (full_cfg, full_res, full_order, full_dir),
        (final_cfg, res, seg0_order + final_order, final_dir),
        selection_metric=selection_metric, zic=zic, check_indices=True,
    )


def _pad_above_auto_scale(ds: Path, tmp_path: Path) -> Path:
    """Copy a fixture dataset, padded above the auto-scale worker threshold.

    ``num_workers=-1`` resolves to 0 for a small dataset and to >0 above
    ``SMALL_DATASET_THRESHOLD``; padding makes the real auto-scaling path (what
    the phase-2 arms use) reachable on a tiny synthetic corpus.
    """
    from nsmor.dataloader_factory import SMALL_DATASET_THRESHOLD

    loaded = torch.load(ds, weights_only=False)
    # The TRAIN split (~80%) is what the factory sizes, so pad the TOTAL well
    # above the threshold (x2.5) to leave the train split above it too.
    target_total = int(SMALL_DATASET_THRESHOLD * 2.5)
    pad = target_total - len(loaded["X_seqs"])
    assert pad > 0, "fixture already above the threshold"
    rng = np.random.RandomState(7)
    for i in range(pad):
        loaded["X_seqs"].append(
            rng.randn(loaded["X_seqs"][0].shape[0], 8).astype(np.float32),
        )
        loaded["Y_seqs"].append(rng.randn(loaded["Y_seqs"][0].shape[0]).astype(np.float32))
        loaded["labels"] = np.append(loaded["labels"], np.int64(0))
        loaded["lengths"] = np.append(loaded["lengths"], loaded["lengths"][0])
        loaded["mcmc_priors"] = np.vstack([
            loaded["mcmc_priors"], loaded["mcmc_priors"][0:1],
        ])
        loaded["session_ids"].append(
            f"0.{600 + (i % 5)}cricket_001_20260202_00000{i % 5}"
            f"_session_{1 + (i // 5)}",
        )
    big = tmp_path / "big_dataset.pt"
    torch.save(loaded, big)
    return big


def test_auto_scaled_workers_resume_is_bitwise(tmp_path: Path) -> None:
    """F3: the real ``num_workers: -1`` auto-scaling path is bitwise too.

    The arms do not request a worker count; they set ``num_workers: -1`` and the
    factory auto-scales to >0 on a large corpus.  This exercises that EXACT path
    (a dataset above the auto-scale threshold, so ``-1`` resolves to >0), with
    ``persistent_workers=True`` and ``checkpoint_interval=10``, and asserts the
    resumed segment reproduces the uninterrupted run bitwise.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    big = _pad_above_auto_scale(ds, tmp_path)

    def _cfg(name: str) -> Any:
        config = _real_config(tmp_path, epochs=4, name=name)
        config.training.num_workers = -1
        config.training.persistent_workers = True
        config.training.checkpoint_interval = 10
        return config

    # Confirm the auto-scale actually resolved >0 (else the test is vacuous).
    torch.manual_seed(0)
    tl, _vl = trainer.build_dataloaders(_cfg("arm_auto_probe"), dataset_path=str(big))
    assert int(tl.num_workers) > 0, (
        f"auto-scale resolved {tl.num_workers} workers; the fixture is too small "
        "to exercise the multi-worker path"
    )
    del tl

    full_cfg = _cfg("arm_auto_full")
    full_res, full_dir, full_order = _run_arm_segment(full_cfg, big)
    assert full_res is not None and full_res["paused"] is False

    seg0 = _cfg("arm_auto_seg0")
    seg0_res, out0, seg0_order = _run_arm_segment(seg0, big, pause_after=1)
    assert seg0_res is not None and seg0_res["paused"] is True
    resume_from = out0 / "final_model.pth"
    assert resume_from.exists()
    trusted = tuple(
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (resume_from, out0 / "best_model.pth") if p.exists()
    )
    final_cfg = _cfg("arm_auto_final")
    final_cfg.checkpoint.resume_from = str(resume_from)
    res, final_dir, final_order = _run_arm_segment(final_cfg, big, trusted=trusted)
    assert res is not None

    # Per-epoch shuffle order is checked parent-side (observable at any worker
    # count), and the padded corpus gives many batches per epoch.
    _assert_full_equivalence(
        (full_cfg, full_res, full_order, full_dir),
        (final_cfg, res, seg0_order + final_order, final_dir),
        selection_metric="total", zic=(), check_indices=True,
    )


# ═════════════════════════════════════════════════════════════
# G1: the worker-RNG precondition is explicit and ENFORCED
# ═════════════════════════════════════════════════════════════
#
# Bitwise resume equivalence holds only when the dataset __getitem__ and
# collate consume no worker-side RNG.  (a) a dataset whose __getitem__ DOES
# consume RNG must be detected (resume refused), (b) the real-corpus dataset
# class used by the phase-2 arms must consume none (proven by running
# __getitem__ under a seeded RNG and asserting both global streams unchanged,
# on the anchor-crop branch it names), and (c) a collate_fn that draws global
# RNG must be detected even when the dataset is clean.


class _RngConsumingDataset(torch.utils.data.Dataset):
    """A dataset whose ``__getitem__`` advances the global RNG (the hazard)."""

    def __init__(self, n: int = 16) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Any:
        # A draw the checkpoint does not carry: in a worker this advances the
        # WORKER's stream, so a resumed persistent worker cannot reproduce it.
        jitter = np.random.random()
        x = torch.full((4, 8), float(idx) + jitter)
        y = torch.full((4,), jitter)
        return x, y


class _GaussianCacheConsumingDataset(torch.utils.data.Dataset):
    """A dataset whose ``__getitem__`` draws ONLY from the NumPy gaussian cache.

    ``np.random.standard_normal()`` served from the cached Gaussian leaves
    ``pos`` and ``keys`` UNCHANGED and only flips ``has_gauss`` /
    ``cached_gaussian``.  A probe that compared only ``pos``/``keys`` would miss
    it (false negative); the full-state comparison must catch it.
    """

    def __init__(self, n: int = 16) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Any:
        jitter = float(np.random.standard_normal())
        return torch.full((4, 8), float(idx) + jitter), torch.full((4,), jitter)


def test_rng_probe_detects_gaussian_cache_draw(tmp_path: Path) -> None:
    """A gaussian-cache-only draw must be detected (not a pos/keys false negative).

    Prime the global NumPy stream so its cached Gaussian is populated
    (``has_gauss == 1``); a one-``standard_normal`` ``__getitem__`` then consumes
    ONLY the cache and leaves ``pos`` and ``keys`` untouched.  The probe must
    still report a hazard, and a bare ``pos``/``keys`` comparison must NOT —
    proving the full-state comparison is load-bearing.
    """
    from scripts import train as trainer

    np.random.seed(1234)
    np.random.standard_normal()  # prime the cached Gaussian -> has_gauss == 1
    primed = np.random.get_state()
    assert primed[3] == 1, "the gaussian cache must be primed for this test"

    ds = _GaussianCacheConsumingDataset()
    assert trainer._dataset_getitem_consumes_rng(ds), (
        "a standard_normal draw served from the cache was not detected"
    )

    # Confirm the false negative the full-state comparison closes: a pos/keys
    # comparison alone would miss the draw.
    np.random.set_state(primed)
    ds[0]
    after = np.random.get_state()
    assert after[2] == primed[2] and np.array_equal(after[1], primed[1]), (
        "this draw was expected to leave pos/keys unchanged"
    )
    assert after[3] != primed[3], "the draw should have consumed the cache"
    np.random.set_state(primed)  # leave the global stream as found


def test_resume_refused_when_dataset_consumes_worker_rng(tmp_path: Path) -> None:
    """(a) A dataset whose __getitem__ consumes RNG must be detected on resume.

    The loader has workers>0 (the arm regime), so a draw inside __getitem__ runs
    in a worker process and is not captured by the checkpoint.  The resume
    preflight must REFUSE it (fail closed), before any epoch executes.
    """
    from scripts import train as trainer

    ds = _make_synthetic_dataset(tmp_path)
    config = _arm_config(
        tmp_path, epochs=2, name="rng_hazard", interval=10, workers=1,
        persistent=True,
    )

    loader = torch.utils.data.DataLoader(
        _RngConsumingDataset(), batch_size=4, num_workers=1,
        persistent_workers=True,
    )
    hazard = trainer._resume_worker_rng_hazard(loader, None)
    assert hazard is not None and "__getitem__ consumes global RNG" in hazard, hazard
    with pytest.raises(ValueError, match="consume no worker-side RNG"):
        trainer._require_bitwise_resume_loader_rng(loader, None)

    # End-to-end: a resume whose loaders would use the hazard is refused.
    executed: list[int] = []

    def forbidden_epoch(**kwargs: Any) -> Any:
        executed.append(kwargs["epoch"])
        return 1.0, {}

    with (
        mock.patch.object(
            trainer, "build_dataloaders",
            return_value=(loader, loader),
        ),
        mock.patch.object(trainer, "train_one_epoch", side_effect=forbidden_epoch),
        mock.patch.object(
            trainer, "_require_reliable_recovery_loaders", lambda *a, **k: None,
        ),
        mock.patch.object(
            trainer, "_require_resume_source_is_newest_complete_epoch",
            lambda *a, **k: None,
        ),
    ):
        # A resume source is needed only to enter the resume preflight; the
        # loader-RNG guard must fire before any epoch regardless of the source.
        resume = tmp_path / "source.pth"
        torch.save({"epoch": 0, "recovery_state_version": 2}, resume)
        config.checkpoint.resume_from = str(resume)
        with pytest.raises(ValueError, match="consume no worker-side RNG"):
            trainer.train(config, dataset_path=str(ds))
    assert executed == [], f"a refused resume executed epochs: {executed}"


def _assert_no_rng_draw(fn: Any, *args: Any, **kwargs: Any) -> None:
    """Assert *fn* consumes neither the torch nor the NumPy global RNG."""
    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    fn(*args, **kwargs)
    assert torch.equal(torch.get_rng_state(), torch_state), (
        "the dataset __getitem__ advanced the global torch RNG"
    )
    after = np.random.get_state()
    assert after[2] == numpy_state[2] and np.array_equal(after[1], numpy_state[1]), (
        "the dataset __getitem__ advanced the global NumPy RNG"
    )


def test_real_corpus_dataset_consumes_no_worker_rng(tmp_path: Path) -> None:
    """(b) The phase-2 arm dataset class consumes no worker-side RNG.

    The eager ``NSMoRDataset`` (what the twelve arms build from the ETL corpus)
    uses anchor-aligned cropping when ``anchor_frames`` is provided — no RNG.
    ``max_seq_len`` is set STRICTLY BELOW the fixture sequence length
    (``_SEQ_LEN``) so the ``X_seq.shape[0] > max_seq_len`` guard is TRUE and the
    ``resolve_anchor_crop`` branch actually runs: the crop property must be
    asserted on the code path it names, not on a pass-through where the guard is
    never taken.  With anchor frames the ``__getitem__`` must consume neither
    the torch nor the NumPy global RNG (the precondition the resume preflight
    enforces).  ``num_workers=0`` here because the parent-side probe is the
    point; the guard's worker check is covered separately.
    """
    from scripts import train as trainer
    from tests.test_train_checkpoint import _SEQ_LEN

    ds = _make_synthetic_dataset(tmp_path)
    config = _arm_config(
        tmp_path, epochs=1, name="rng_clean", interval=10, workers=0,
        persistent=False,
    )
    crop_len = _SEQ_LEN - 10
    config.training.max_seq_len = crop_len
    train_loader, _val = trainer.build_dataloaders(config, dataset_path=str(ds))
    assert train_loader is not None
    leaf = trainer._unwrap_dataset(train_loader.dataset)

    # The anchor-crop branch is REACHABLE: the fixture trials are longer than
    # max_seq_len and anchor_frames are present, so __getitem__ crops.
    assert leaf.max_seq_len == crop_len
    assert leaf.anchor_frames is not None
    assert leaf.sequences[0][0].shape[0] > crop_len, (
        "fixture trials must be longer than max_seq_len so the crop branch runs"
    )
    item = leaf[0]
    # With is_pure_wind present the item is (idx, X, Y); else (X, Y).
    x_cropped = item[1] if len(item) == 3 else item[0]
    assert x_cropped.shape[0] == crop_len, (
        "the anchor-aligned crop did not run; the RNG-free property would be "
        "asserted on a code path that never executes"
    )

    torch.manual_seed(1234)
    np.random.seed(1234)
    for idx in range(len(leaf)):
        _assert_no_rng_draw(leaf.__getitem__, idx)

    # And the guard agrees: no hazard for this dataset even at workers>0.
    probe = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=4, num_workers=1, persistent_workers=True,
    )
    assert trainer._resume_worker_rng_hazard(probe, None) is None


def test_resume_refused_when_collate_consumes_worker_rng(tmp_path: Path) -> None:
    """A collate_fn that draws global RNG must be detected on resume.

    The collate runs in the WORKER process, so a draw inside it advances a
    stream the checkpoint does not carry — the same hazard as a draw inside
    ``__getitem__``, and part of §11.1's enforced precondition.  A clean
    dataset paired with an RNG-consuming collate must still be refused.
    """
    from scripts import train as trainer

    def rng_collate(batch: Any) -> Any:
        jitter = float(np.random.random())  # a draw the checkpoint does not carry
        # The probe runs this on the loader's real items, which are (idx, X, Y).
        pairs = [(b[-2], b[-1]) for b in batch]
        X = torch.stack([x for x, _ in pairs]) + jitter
        Y = torch.stack([y for _, y in pairs])
        return X, Y

    ds = _make_synthetic_dataset(tmp_path)
    config = _arm_config(
        tmp_path, epochs=2, name="collate_hazard", interval=10, workers=1,
        persistent=True,
    )
    train_loader, _val = trainer.build_dataloaders(config, dataset_path=str(ds))
    leaf = trainer._unwrap_dataset(train_loader.dataset)
    # The dataset itself is clean; only the collate draws.
    assert not trainer._dataset_getitem_consumes_rng(leaf)

    loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=4, num_workers=1,
        persistent_workers=True, collate_fn=rng_collate,
    )
    hazard = trainer._resume_worker_rng_hazard(loader, None)
    assert hazard is not None and "collate_fn consumes global RNG" in hazard, hazard
    with pytest.raises(ValueError, match="consume no worker-side RNG"):
        trainer._require_bitwise_resume_loader_rng(loader, None)

