"""Focused tests for the real-data PHASE-2 controls (2026-10-10).

Covers the two new, default-off controls added for
`docs/realdata-phase2-protocol-20261010.md`.  They are SCRIPT-OWNED (resolved
by ``scripts/train.py::build_config`` into module state and an additive
``phase2:`` YAML block) — deliberately NOT added to the protected
``nsmor/config_parser.py`` schema, so ``ExperimentConfig.to_dict()`` stays
byte-unchanged:

* ``zero_input_channels``  — history ablation (R2)
* ``selection_metric``     — MSE-based checkpoint selection (all arms)

Plus the arm-config resolution diff (only declared keys differ from A1).

Every control must default to the CURRENT behaviour; these tests pin that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from nsmor.config import DEFAULT_FEATURE
from nsmor.config_parser import ExperimentConfig
from nsmor.nsmor_dataloader import NSMoRDataset

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_phase2_globals():
    """``build_config`` mutates script-owned module state; restore it."""
    import scripts.train as T

    saved = (T._ZERO_INPUT_CHANNELS, T._SELECTION_METRIC)
    yield
    T._ZERO_INPUT_CHANNELS, T._SELECTION_METRIC = saved


# ──────────────────────────────────────────────────────────────
# zero_input_channels — script-owned parse + defaults
# ──────────────────────────────────────────────────────────────

def test_zero_input_channels_default_is_empty() -> None:
    from scripts.train import _ZERO_INPUT_CHANNELS

    assert _ZERO_INPUT_CHANNELS == ()


def test_zero_input_channels_accepts_sensory_columns() -> None:
    from scripts.train import _parse_zero_input_channels

    cfg = ExperimentConfig()
    assert _parse_zero_input_channels([2, 3], cfg) == (2, 3)
    assert _parse_zero_input_channels("2,3", cfg) == (2, 3)
    assert _parse_zero_input_channels(None, cfg) == ()


@pytest.mark.parametrize("bad", [[4], [7], [-1], [1.0], [True], ["2"]])
def test_zero_input_channels_rejects_non_sensory_or_non_int(bad) -> None:
    from scripts.train import _parse_zero_input_channels

    with pytest.raises(ValueError):
        _parse_zero_input_channels(bad, ExperimentConfig())


def test_zero_input_channels_rejects_duplicates() -> None:
    from scripts.train import _parse_zero_input_channels

    with pytest.raises(ValueError):
        _parse_zero_input_channels([2, 2], ExperimentConfig())


# ──────────────────────────────────────────────────────────────
# History ablation — scripts-level in-place zeroing (no pipeline change)
# ──────────────────────────────────────────────────────────────

def _dataset() -> NSMoRDataset:
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(3):
        x = rng.standard_normal((5, 8)).astype(np.float64)
        x[:, 4:8] = 0.0  # MCMC columns are filled by _fill_priors
        y = rng.standard_normal(5).astype(np.float64)
        seqs.append((x, y, 0))
    priors = np.full((3, 4), 0.25, dtype=np.float64)
    return NSMoRDataset(
        sequences=seqs, mcmc_priors=priors, feature_config=DEFAULT_FEATURE,
        max_seq_len=None,
    )


def test_zero_input_channels_inplace_default_is_noop() -> None:
    from scripts.train import zero_input_channels_inplace

    ds = _dataset()
    before = ds.sequences[0][0][:, 2].copy()
    zero_input_channels_inplace(ds, [])
    assert np.array_equal(ds.sequences[0][0][:, 2], before)


def test_zero_input_channels_inplace_zeroes_declared_channels() -> None:
    from scripts.train import zero_input_channels_inplace

    ds = _dataset()
    zero_input_channels_inplace(ds, [2, 3])
    x, _y = ds[0]
    assert torch.equal(x[:, 2], torch.zeros(5))
    assert torch.equal(x[:, 3], torch.zeros(5))
    # Channels 0,1 untouched; MCMC columns still sum to 1.
    assert np.allclose(x[:, 4:8].sum(dim=1).numpy(), 1.0, atol=1e-5)


def test_zero_input_channels_inplace_requires_sequences() -> None:
    from scripts.train import zero_input_channels_inplace

    class _NoSequences:
        pass

    with pytest.raises(ValueError):
        zero_input_channels_inplace(_NoSequences(), [2])


def test_train_mse_selection_records_phase2_provenance(
    tmp_path: Path, monkeypatch,
) -> None:
    """A 1-epoch MSE-selected run writes the additive phase-2 provenance and a
    finite selected metric into best_model.pth (config dict untouched)."""
    import scripts.train as T
    from nsmor.pipeline.nested_prior import load_artifact_bytes
    from tests.test_train_checkpoint import _make_config, _make_synthetic_dataset

    ds = _make_synthetic_dataset(tmp_path)
    cfg = _make_config(tmp_path, epochs=1)
    monkeypatch.setattr(T, "_SELECTION_METRIC", "mse")
    monkeypatch.setattr(T, "_ZERO_INPUT_CHANNELS", ())
    T.train(cfg, lambda_reg=0.01, dataset_path=str(ds))

    best = Path(cfg.checkpoint.output_dir) / "best_model.pth"
    ckpt = load_artifact_bytes(best.read_bytes())
    assert ckpt["selection_metric"] == "mse"
    assert ckpt["zero_input_channels"] == []
    assert np.isfinite(ckpt["val_loss"])
    # The protected config dict is byte-unchanged: the controls never enter it.
    assert "selection_metric" not in ckpt["config"]["checkpoint"]
    assert "zero_input_channels" not in ckpt["config"]["data"]


def test_build_dataloaders_applies_zero_channels() -> None:
    """End-to-end: build_dataloaders(..., zero_input_channels=[2,3]) zeroes X."""
    from scripts.train import build_dataloaders
    from tests.test_train_checkpoint import _make_config, _make_synthetic_dataset
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ds = _make_synthetic_dataset(tmp)
        cfg = _make_config(tmp, epochs=1)
        cfg.training.num_workers = 0
        cfg.training.persistent_workers = False
        train_loader, _val = build_dataloaders(
            cfg, dataset_path=str(ds), zero_input_channels=[2, 3],
        )
        batch = next(iter(train_loader))
        x = batch[0]
        assert torch.equal(x[:, :, 2], torch.zeros_like(x[:, :, 2]))
        assert torch.equal(x[:, :, 3], torch.zeros_like(x[:, :, 3]))



# ──────────────────────────────────────────────────────────────
# selection_metric — script-owned parse + defaults
# ──────────────────────────────────────────────────────────────

def test_selection_metric_default_is_total() -> None:
    from scripts.train import _SELECTION_METRIC

    assert _SELECTION_METRIC == "total"


def test_selection_metric_accepts_mse_and_total() -> None:
    from scripts.train import _parse_selection_metric

    assert _parse_selection_metric("mse") == "mse"
    assert _parse_selection_metric("total") == "total"
    assert _parse_selection_metric(None) == "total"


def test_selection_metric_rejects_unknown() -> None:
    from scripts.train import _parse_selection_metric

    with pytest.raises(ValueError):
        _parse_selection_metric("rmse")


def test_config_schema_is_unaffected() -> None:
    """The controls must NOT enter the protected ExperimentConfig schema."""
    cfg = ExperimentConfig()
    assert "selection_metric" not in cfg.checkpoint.__dataclass_fields__
    assert "zero_input_channels" not in cfg.data.__dataclass_fields__
    assert "selection_metric" not in cfg.to_dict()["checkpoint"]
    assert "zero_input_channels" not in cfg.to_dict()["data"]


# ──────────────────────────────────────────────────────────────
# build_config — CLI + phase2 YAML plumbing, defaults preserved
# ──────────────────────────────────────────────────────────────

def test_build_config_zero_input_channels_cli() -> None:
    import scripts.train as T

    T.build_config(["--zero_input_channels", "2,3"])
    assert T._ZERO_INPUT_CHANNELS == (2, 3)


def test_build_config_selection_metric_cli() -> None:
    import scripts.train as T

    T.build_config(["--selection_metric", "mse"])
    assert T._SELECTION_METRIC == "mse"


def test_build_config_defaults_unchanged() -> None:
    import scripts.train as T

    T.build_config([])
    assert T._ZERO_INPUT_CHANNELS == ()
    assert T._SELECTION_METRIC == "total"


def test_build_config_reads_phase2_yaml_block(tmp_path: Path) -> None:
    """A script-owned ``phase2:`` YAML block is honoured; CLI wins over YAML."""
    import scripts.train as T

    yml = tmp_path / "arm.yaml"
    yml.write_text(
        "phase2:\n  selection_metric: mse\n  zero_input_channels: [2, 3]\n",
        encoding="utf-8",
    )
    T.build_config(["--config", str(yml)])
    assert T._SELECTION_METRIC == "mse"
    assert T._ZERO_INPUT_CHANNELS == (2, 3)
    # CLI overrides the YAML block.
    T.build_config(["--config", str(yml), "--zero_input_channels", "0"])
    assert T._ZERO_INPUT_CHANNELS == (0,)


# ──────────────────────────────────────────────────────────────
# validate() — MSE selection returns the masked-MSE term
# ──────────────────────────────────────────────────────────────

class _TinyModel(torch.nn.Module):
    """Emits a constant zero prediction; internals satisfy the loss contract."""

    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.hidden = hidden
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, lengths, return_internals=False):  # noqa: D401
        B, T, _ = x.shape
        y = torch.zeros(B, T)
        internals = {
            "routing_gates": torch.full((B, T, 2), 0.5),
            "lif_spikes": torch.zeros(B, T, self.hidden),
        }
        return (y, internals) if return_internals else y


def _val_loader():
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.zeros(2, 6, 8)
    y = torch.ones(2, 6)  # constant error 1.0 -> MSE == 1.0
    lengths = torch.tensor([6, 6])
    ds = TensorDataset(x, y, lengths)
    return DataLoader(ds, batch_size=2)


def test_validate_mse_selection_equals_masked_mse() -> None:
    from scripts.train import build_loss, validate

    model = _TinyModel()
    loader = _val_loader()
    criterion = build_loss(ExperimentConfig())
    selected = validate(
        model=model, loader=loader, criterion=criterion,
        device=torch.device("cpu"), selection_metric="mse",
    )
    # Constant error 1.0 over all valid frames -> masked MSE == 1.0.
    assert selected == pytest.approx(1.0, abs=1e-6)
    assert validate.last_val_mse == pytest.approx(1.0, abs=1e-6)


class _ScaleModel(torch.nn.Module):
    """Emits ``-a`` (so error vs y=+a is 2a) keyed by a per-trial scale tag."""

    def __init__(self, scales) -> None:
        super().__init__()
        self.scales = list(scales)

    def forward(self, x, lengths, return_internals=False):
        B, T, _ = x.shape
        # x[:, :, 0] carries the per-trial scale tag; predict -scale.
        y = -x[:, :, 0]
        internals = {
            "routing_gates": torch.full((B, T, 2), 0.5),
            "lif_spikes": torch.zeros(B, T, self.hidden),
        }
        return (y, internals) if return_internals else y


def test_validate_mse_selection_is_frame_weighted_pooled() -> None:
    """M4: the selection MSE is the frame-weighted pooled masked MSE, NOT a
    per-batch macro-average (which over-weights a small trailing batch)."""
    from torch.utils.data import DataLoader, TensorDataset

    from scripts.train import build_loss, validate

    # Batch A: 3 trials of length 6, constant error 2.0 -> MSE 4.0.
    # Batch B (size 1): 1 trial of length 6, constant error 4.0 -> MSE 16.0.
    # Pooled (correct) = (3*6*4 + 1*6*16) / 24 = (72+96)/24 = 7.0.
    # Macro-average (wrong) = (4.0 + 16.0) / 2 = 10.0.
    x = torch.zeros(4, 6, 8)
    x[:3, :, 0] = 1.0   # predict -1 -> error 2 (vs y=1)
    x[3, :, 0] = 2.0    # predict -2 -> error 4 (vs y=2)
    y = torch.zeros(4, 6)
    y[:3] = 1.0
    y[3] = 2.0
    lengths = torch.full((4,), 6, dtype=torch.long)
    ds = TensorDataset(x, y, lengths)
    loader = DataLoader(ds, batch_size=3)  # ragged: [3, 1]

    class _M(_ScaleModel):
        hidden = 4

    selected = validate(
        model=_M([1.0, 2.0]), loader=loader,
        criterion=build_loss(ExperimentConfig()),
        device=torch.device("cpu"), selection_metric="mse",
    )
    assert selected == pytest.approx(7.0, abs=1e-6)
    assert validate.last_val_mse == pytest.approx(7.0, abs=1e-6)
    # And it is NOT the per-batch macro-average.
    assert abs(selected - 10.0) > 1.0


def test_validate_total_selection_differs_from_mse_when_regularised() -> None:
    from scripts.train import build_loss, validate

    model = _TinyModel()
    loader = _val_loader()
    criterion = build_loss(ExperimentConfig())
    total = validate(
        model=model, loader=loader, criterion=criterion,
        device=torch.device("cpu"), selection_metric="total",
        lambda_reg=0.5,
    )
    mse = validate.last_val_mse
    # The router term (g_gru=0.5 -> 0.25 * 0.5) is strictly positive here.
    assert total > mse + 1e-6


def test_validate_rejects_unknown_selection_metric() -> None:
    from scripts.train import build_loss, validate

    with pytest.raises(ValueError):
        validate(
            model=_TinyModel(), loader=_val_loader(),
            criterion=build_loss(ExperimentConfig()),
            device=torch.device("cpu"), selection_metric="nope",
        )


# ──────────────────────────────────────────────────────────────
# Arm-config resolution diff — only declared keys differ from A1
# ──────────────────────────────────────────────────────────────

_DECLARED = {
    "checkpoint.output_dir",
    "training.random_seed",
    "model.persistence_skip",
    "loss.lambda_jerk",
}

# Per-arm exact changed-leaf set.  ``selection_metric`` / ``zero_input_channels``
# are script-owned (they never appear in ``to_dict()``); each arm must change
# ONLY its own declared leaf (plus the per-seed output dir / seed).
_ARM_LEAF = {
    "R0-control": None,
    "R1-residual-persistence": "model.persistence_skip",
    "R2-history-ablated": None,
    "R3-jerk-ablated": "loss.lambda_jerk",
}
_PHASE2_EXPECTED = {
    "R0-control": {"selection_metric": "mse", "zero_input_channels": []},
    "R1-residual-persistence": {"selection_metric": "mse", "zero_input_channels": []},
    "R2-history-ablated": {"selection_metric": "mse", "zero_input_channels": [2, 3]},
    "R3-jerk-ablated": {"selection_metric": "mse", "zero_input_channels": []},
}


def test_resolution_diff_only_declared_keys_change() -> None:
    import json

    diff = json.loads(
        (REPO / "config/realdata-phase2-20261010/arm-resolution-diff.json")
        .read_text(encoding="utf-8")
    )
    assert len(diff["arms"]) == 12
    for key, entry in diff["arms"].items():
        assert entry["missing_leaves"] == []
        assert entry["added_leaves"] == []
        changed = {row[0] for row in entry["changed_leaves"]}
        assert changed <= _DECLARED, f"{key} changed undeclared leaves: {changed}"
        # Per-arm: ONLY this arm's own declared leaf changed (catches a
        # cross-wired arm that flipped an extra leaf, e.g. R1 also zeroing).
        expected = {"checkpoint.output_dir"}
        if entry["seed"] != 42:
            expected.add("training.random_seed")
        leaf = _ARM_LEAF[entry["arm"]]
        if leaf is not None:
            expected.add(leaf)
        assert changed == expected, f"{key} changed {changed} != {expected}"
        # The script-owned phase-2 block resolved to the declared values.
        assert entry["phase2_resolved"] == _PHASE2_EXPECTED[entry["arm"]]


def test_phase2_configs_current() -> None:
    """The generator must be able to reproduce the on-disk configs."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "phase2_cfg", REPO / "scripts" / "realdata_phase2_configs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--check"]) == 0


# ──────────────────────────────────────────────────────────────
# Phase-2 scorer — declared family + Holm + arm binding
# ──────────────────────────────────────────────────────────────

def _scorer():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "phase2_scorer", REPO / "scripts" / "evaluate_realdata_phase2.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_phase2_family_slots_shape() -> None:
    sc = _scorer()
    slots = sc.declared_family_slots()
    assert len(slots) == 48
    assert len(set(slots)) == 48
    # 4 self-comparison slots (candidate == comparator R0).
    selfs = [s for s in slots if s.split("|")[0] == s.split("|")[1]]
    assert len(selfs) == 4


def test_phase2_onset_uses_checkpoint_crop_not_dataclass_default(
    monkeypatch, tmp_path: Path,
) -> None:
    """The onset base must use the checkpoint's crop (2400), never the bare
    ``ExperimentConfig()`` default (1000)."""
    import numpy as np

    sc = _scorer()
    ds = tmp_path / "dataset.pt"
    pr = tmp_path / "prior.pt"
    ds.write_bytes(b"x")
    pr.write_bytes(b"y")
    # ``_compute_onsets`` loads the dataset first, then the prior; side-effect
    # the two calls in order.
    seqs = [
        {"anchor_frames": np.array([3000]), "lengths": np.array([4200])},
        {"val_indices": np.array([0])},
    ]
    calls = {"n": 0}

    def _fake(raw, map_location=None):
        out = seqs[calls["n"]]
        calls["n"] += 1
        return out

    monkeypatch.setattr(sc, "load_artifact_bytes", _fake)
    # n=4200, anchor=3000 -> start = 3000-1200 = 1800, onset = 3000-1800 = 1200.
    onsets = sc._compute_onsets(ds, pr, 2400)
    assert onsets.tolist() == [1200]
    # A non-canonical crop is refused rather than silently misaligned.
    with pytest.raises(ValueError):
        sc._compute_onsets(ds, pr, 1000)
    with pytest.raises(ValueError):
        sc._compute_onsets(ds, pr, None)


def test_phase2_holm_orders_and_marks_unavailable() -> None:
    sc = _scorer()
    slots = sc.declared_family_slots()
    pvals = {s: 0.5 for s in slots}
    # Make one slot strongly significant and one unavailable.
    pvals[slots[0]] = 0.0001
    pvals[slots[1]] = None
    res = sc._holm(pvals)
    assert res["family_size"] == 48
    assert res["unavailable_slot_count"] == 1
    assert res["valid_test_count"] == 47
    # Holm over the FIXED 48-slot family (unavailable carried as p=1):
    # the rank-1 adjusted p is 0.0001 * 48, NOT 0.0001 * 47.
    assert res["slots"][slots[0]]["adjusted_p"] <= 0.0001 * 48 + 1e-12
    assert res["slots"][slots[0]]["adjusted_p"] > 0.0001 * 47 + 1e-12
    assert res["slots"][slots[1]]["available"] is False
    assert res["slots"][slots[1]]["adjusted_p"] is None
    # Unavailable slots report null significance (never a fabricated False).
    assert res["slots"][slots[1]]["significant"] is None


def _a1_shaped_config() -> ExperimentConfig:
    """A config carrying A1's declared SHARED base leaves (so the arm gate's
    shared-base re-verification passes); the bare ``ExperimentConfig()``
    default is relu/off and would be refused."""
    sc = _scorer()
    cfg = ExperimentConfig()
    cfg.model.activation = str(sc._A1_SHARED["activation"])
    cfg.model.refinement_mode = str(sc._A1_SHARED["refinement_mode"])
    cfg.loss.lambda_compute = float(sc._A1_SHARED["lambda_compute"])
    cfg.training.num_epochs = int(sc._A1_SHARED["num_epochs"])
    cfg.training.early_stopping_patience = int(
        sc._A1_SHARED["early_stopping_patience"])
    cfg.training.max_seq_len = int(sc._A1_SHARED["max_seq_len"])
    return cfg


def test_phase2_arm_binding_accepts_and_rejects() -> None:
    sc = _scorer()
    cfg = _a1_shaped_config()
    cfg.model.persistence_skip = 0.0
    cfg.loss.lambda_jerk = 0.005
    ckpt = {
        "config": cfg.to_dict(),
        "zero_input_channels": [2, 3],
        "selection_metric": "mse",
    }
    sc._validate_arm_config(ckpt, "R2")  # no raise
    with pytest.raises(ValueError):
        sc._validate_arm_config(ckpt, "R3")  # lambda_jerk mismatch
    with pytest.raises(ValueError):
        sc._validate_arm_config(ckpt, "R0")  # zero_input_channels mismatch
    # A checkpoint whose selection_metric was not 'mse' is refused.
    bad = dict(ckpt, selection_metric="total")
    with pytest.raises(ValueError):
        sc._validate_arm_config(bad, "R2")


def test_phase2_arm_binding_rejects_wrong_shared_base() -> None:
    """Defense in depth: a checkpoint trained under a different shared base
    (here activation=relu) is refused even with matching arm leaves."""
    sc = _scorer()
    cfg = _a1_shaped_config()
    cfg.model.activation = "relu"  # not the declared swiglu base
    cfg.model.persistence_skip = 0.0
    cfg.loss.lambda_jerk = sc._A1_LAMBDA_JERK
    ckpt = {"config": cfg.to_dict(), "zero_input_channels": [],
            "selection_metric": "mse"}
    with pytest.raises(ValueError):
        sc._validate_arm_config(ckpt, "R0")


def test_phase2_skill_vs_persistence_is_emitted() -> None:
    import numpy as np

    sc = _scorer()
    model = np.array([2.0, 4.0, 1.0])
    persist = np.array([1.0, 2.0, 1.0])
    fc = np.array([10, 10, 10])
    res = sc._skill_vs_persistence(model, persist, fc)
    # Frame-weighted pooled: model 7/3, persistence 4/3 -> 1 - 7/4 = -0.75.
    assert res["skill_vs_persistence"] == pytest.approx(-0.75)
    assert res["reason"] is None


def test_phase2_seed_spread_and_json_vectors() -> None:
    import numpy as np

    sc = _scorer()
    spread = sc._seed_spread([
        np.array([1.0, 2.0, 3.0]),
        np.array([1.5, 2.5, 3.5]),
        np.array([2.0, 3.0, 4.0]),
    ])
    assert spread["n_seeds"] == 3
    assert spread["per_trial_std_mean"] == pytest.approx(0.5)
    # Non-finite entries are serialized as null, never NaN.
    assert sc._json_array(np.array([1.0, np.nan, np.inf])) == [1.0, None, None]


def test_phase2_scorer_keys_match_generated_configs() -> None:
    import json

    sc = _scorer()
    diff = json.loads(
        (REPO / "config/realdata-phase2-20261010/arm-resolution-diff.json")
        .read_text(encoding="utf-8")
    )
    prefix = {
        "R0": "R0-control", "R1": "R1-residual-persistence",
        "R2": "R2-history-ablated", "R3": "R3-jerk-ablated",
    }
    # For every arm, the scorer's declared keys equal the config's arm keys.
    for arm, arm_name in prefix.items():
        entry = diff["arms"][f"{arm_name}-seed42"]
        assert entry["arm_keys"] == sc.EXPECTED_ARM_KEYS[arm]


# ──────────────────────────────────────────────────────────────
# R1/R2 review fixes — comparator, availability, ceilings, bounds
# ──────────────────────────────────────────────────────────────

def test_phase2_lambda_jerk_derived_from_a1_reference() -> None:
    """The non-arm ``lambda_jerk`` leaf is read from A1, not hardcoded."""
    import yaml

    sc = _scorer()
    a1 = yaml.safe_load(
        (REPO / "config/realdata-preliminary-20261009/A1-swiglu-off.yaml")
        .read_text(encoding="utf-8")
    )
    assert sc._A1_LAMBDA_JERK == float(a1["loss"]["lambda_jerk"])
    # R0/R1/R2 carry the A1 value; R3 is the jerk-ablation arm (0.0).
    assert sc.EXPECTED_ARM_KEYS["R0"]["lambda_jerk"] == sc._A1_LAMBDA_JERK
    assert sc.EXPECTED_ARM_KEYS["R3"]["lambda_jerk"] == 0.0


def test_phase2_h5_ceiling_is_held_at_start_not_pure_lag() -> None:
    """The h=5 comparator ceiling must be the held-at-start value, not the
    step-1 pure lag-5 audit value (10.07)."""
    sc = _scorer()
    h5 = sc.PERSISTENCE_CEILING["h5"]
    assert h5["mse"] == pytest.approx(0.925, abs=1e-3)
    # The pure lag-5 reference is retained separately and never scores a slot.
    assert sc.PURE_LAG_CEILING["h5"]["mse"] == pytest.approx(10.07, abs=1e-2)
    # Onset is one of the four primary scopes and MUST carry a ceiling.
    assert sc.PERSISTENCE_CEILING["onset"]["mse"] == pytest.approx(26.65, abs=1e-2)
    # The rollout escape ceilings the protocol quotes are present.
    for scope in ("h25_escape", "h125_escape"):
        assert scope in sc.PERSISTENCE_CEILING


def test_phase2_ceilings_artifact_matches_scorer_constants() -> None:
    """The durable ceilings artifact and the scorer constants cannot drift."""
    import json

    sc = _scorer()
    art = json.loads(
        (REPO / "docs/realdata-phase2-ceilings-20261010.json")
        .read_text(encoding="utf-8")
    )
    held = art["held_at_start"]
    for scope, entry in sc.PERSISTENCE_CEILING.items():
        assert scope in held, scope
        assert held[scope]["mse"] == pytest.approx(entry["mse"], rel=1e-3)
        assert held[scope]["r2"] == pytest.approx(entry["r2"], rel=1e-3)
    pure = art["pure_lag_reference_only"]
    for scope, entry in sc.PURE_LAG_CEILING.items():
        assert pure[scope]["mse"] == pytest.approx(entry["mse"], rel=1e-3)
        assert pure[scope]["r2"] == pytest.approx(entry["r2"], rel=1e-3)


def test_phase2_slot_availability_follows_primitive_pvalue() -> None:
    """A slot whose sign-flip primitive returns a null p must be emitted
    available=False with the primitive's reason (never available=True)."""
    sc = _scorer()
    # Simulate the primitive output for an unasserted-exchangeability slot.
    res = {
        "sign_flip": {"p_value": None, "reason": "exchangeability_not_asserted"},
        "delta_convention": "candidate_minus_comparator_mse",
    }
    p_val = res["sign_flip"]["p_value"]
    slot = {
        "available": p_val is not None,
        "reason": None if p_val is not None else res["sign_flip"].get("reason"),
        **res,
    }
    assert slot["available"] is False
    assert slot["reason"] == "exchangeability_not_asserted"
    # And a real p yields available=True with no reason.
    res2 = {"sign_flip": {"p_value": 0.01, "reason": None}}
    p2 = res2["sign_flip"]["p_value"]
    slot2 = {"available": p2 is not None,
             "reason": None if p2 is not None else res2["sign_flip"].get("reason"),
             **res2}
    assert slot2["available"] is True and slot2["reason"] is None


def test_phase2_resume_guard_refuses_zero_input_channels_switch(
    monkeypatch, tmp_path: Path,
) -> None:
    """A resume that switches the history ablation must be refused."""
    import scripts.train as T

    cfg = ExperimentConfig()
    ckpt = {
        "config": cfg.to_dict(),
        "selection_metric": "mse",
        "zero_input_channels": [2, 3],
    }
    monkeypatch.setattr(T, "_SELECTION_METRIC", "mse")
    monkeypatch.setattr(T, "_ZERO_INPUT_CHANNELS", ())
    with pytest.raises(ValueError):
        T._require_architecture_config_match(ckpt, cfg, tmp_path / "x.pth")
    # Matching ablation is accepted.
    monkeypatch.setattr(T, "_ZERO_INPUT_CHANNELS", (2, 3))
    T._require_architecture_config_match(ckpt, cfg, tmp_path / "x.pth")


def test_phase2_zero_input_channels_inplace_honours_sensory_dim() -> None:
    """The ablation bound is the passed sensory_dim, the same quantity the
    CLI parser validates against (not the dataset's physical dim)."""
    from scripts.train import zero_input_channels_inplace
    from tests.test_realdata_phase2_options import _dataset

    ds = _dataset()
    # A channel inside the dataset's physical dim but outside a narrower
    # sensory_dim must be refused (the two bounds are now one).
    with pytest.raises(ValueError):
        zero_input_channels_inplace(ds, [3], sensory_dim=2)
    # The default (no sensory_dim) still uses the dataset's physical dim.
    zero_input_channels_inplace(ds, [3])


def test_phase2_main_payload_availability_is_consistent(
    monkeypatch, tmp_path: Path,
) -> None:
    """End-to-end: slots_out.available must equal the Holm slot's availability,
    and the declared secondary (t>=0 pooled, peak, ceilings) is emitted."""
    import json

    import numpy as np

    sc = _scorer()
    n = 12
    # 12 trials, 12 distinct prefixes -> exact sign-flip branch.
    y_seqs = [np.abs(np.random.default_rng(i).standard_normal(30)) for i in range(n)]
    trial_ids = [f"t{i}" for i in range(n)]
    prefix_ids = [f"p{i}" for i in range(n)]

    monkeypatch.setattr(
        sc, "extract_canonical_validation_targets",
        lambda d, p: (y_seqs, trial_ids, prefix_ids),
    )
    monkeypatch.setattr(sc, "_validate_arm_config", lambda ckpt, arm: None)
    monkeypatch.setattr(sc, "load_artifact_bytes",
                        lambda raw, map_location=None: {})
    monkeypatch.setattr(sc, "_sha256", lambda p: "0" * 64)

    counts = {s: np.full(n, 10, dtype=int) for s in sc.ALL_SCOPES}
    true_peak = np.linspace(1.0, 5.0, n)

    def _fake_scope(cp, dp, npp, dev, yts):
        rng = np.random.default_rng(0)
        model = {s: np.full(n, 1.0) + rng.random(n) for s in sc.ALL_SCOPES}
        # Persistence per scope equals the frozen ceiling so the M3
        # reconciliation passes; the model is slightly better on every scope.
        persist = {
            s: np.full(n, float(sc.PERSISTENCE_CEILING[s]["mse"]))
            for s in sc.ALL_SCOPES
        }
        zero = {s: np.full(n, 5.0) for s in sc.ALL_SCOPES}
        t0 = np.full(n, 1.5)
        pa = {
            "pred_frame_num": np.zeros(n, dtype=int),
            "pred_frame_den": np.full(n, 5, dtype=int),
            "pred_trial_num": np.zeros(n, dtype=int),
            "pred_trial_den": np.ones(n, dtype=int),
            "animal_frame_num": np.zeros(n, dtype=int),
            "animal_frame_den": np.full(n, 5, dtype=int),
            "animal_trial_num": np.zeros(n, dtype=int),
            "animal_trial_den": np.ones(n, dtype=int),
        }
        return ({"model": model, "persistence": persist, "zero": zero},
                counts, np.zeros(n), true_peak, t0, np.full(n, 10, dtype=int),
                {"zero_input_channels": [],
                 "rollout_regime": "feedback_then_arm_transform"}, pa)

    monkeypatch.setattr(sc, "_per_trial_scope_mse", _fake_scope)

    # Build 12 dummy checkpoint paths (content is mocked away).
    arm_paths = []
    for arm in sc.ARMS:
        ps = []
        for seed in (42, 43, 44):
            f = tmp_path / f"{arm}-{seed}.pth"
            f.write_bytes(b"x")
            ps.append(f)
        arm_paths.append((arm, ps))
    argv = []
    for arm, ps in arm_paths:
        argv += [f"--{arm.lower()}", *[str(p) for p in ps]]
    argv += ["--dataset", str(tmp_path / "d.pt"),
             "--nested_prior", str(tmp_path / "p.pt"),
             "--output_dir", str(tmp_path / "out"),
             "--exchangeability_asserted"]
    assert sc.main(argv) == 0
    payload = json.loads(
        (tmp_path / "out" / "realdata_phase2_family.json").read_text("utf-8")
    )
    slots = payload["declared_family"]["slots"]
    holm = payload["declared_family"]["holm_bonferroni"]["slots"]
    for slot, entry in slots.items():
        assert entry["available"] == holm[slot]["available"], slot
    # Declared descriptive secondaries are emitted.
    sec = payload["secondary_metrics"]["R0"]
    assert "pooled_mse_t_ge_0" in sec
    assert sec["peak_speed_error"]["observed_peak_cm_s"] == pytest.approx(
        float(true_peak.mean())
    )
    assert sec["peak_speed_error"]["observed_peak_provenance"] == (
        "scored_split_aligned_frames"
    )
    for scope in ("h5", "h25", "h125"):
        h = sec["horizons"][scope]
        assert "persistence_ceiling" in h
        assert "persistence_ceiling_escape" in h
        # The escape-frame scalar is emitted too (M1).
        he = sec["horizons"][f"{scope}_escape"]
        assert he["persistence_ceiling"] == sc.PERSISTENCE_CEILING[
            f"{scope}_escape"]
    # Pre-stimulus false-alarm secondary is emitted (F2).
    pa = sec["prestim_spontaneous_escape"]
    for key in ("model_prestim_false_alarm_frame_rate",
                "model_prestim_false_alarm_trial_rate",
                "animal_prestim_escape_frame_rate",
                "animal_prestim_escape_trial_rate"):
        assert key in pa
    # M3 reconciliation is emitted and matched.
    rec = payload["ceiling_reconciliation"]
    for scope in sc.PERSISTENCE_CEILING:
        assert rec[scope]["matches"] is True
    assert "persistence_ceiling" in payload
    assert "pure_lag_ceiling_reference_only" in payload


# ──────────────────────────────────────────────────────────────
# Round-3 fixes — B2 single rollout regime, B3 target equality, M3 ceiling
# reconciliation, M4 pooled selection, F2 false-alarm
# ──────────────────────────────────────────────────────────────

def test_phase2_rollout_regime_feedback_then_ablation() -> None:
    """B2: the ONE rollout regime feeds the prediction back into channels 2-3,
    THEN zeroes the arm's declared channels — for an ablated arm (R2) the
    feedback is written but then zeroed, so channels 2-3 stay 0."""
    sc = _scorer()

    captured = []

    class _Recorder(torch.nn.Module):
        # ``priming`` calls receive ``states={}``; rollout calls receive the
        # model's own returned (non-empty) state, so we can tell them apart.
        def forward(self, x, lengths, return_internals=False, states=None):
            is_rollout = states is not None and len(states) > 0
            captured.append((x.detach().clone(), is_rollout))
            B, T, _ = x.shape
            y = torch.zeros(B, T)
            internals = {
                "routing_gates": torch.full((B, T, 2), 0.5),
                "lif_spikes": torch.zeros(B, T, 4),
            }
            if states is not None:
                return y, internals, {"h": torch.zeros(1)}
            return (y, internals) if return_internals else y

    model = _Recorder()
    x_i = torch.zeros(6, 8)
    x_i[:, 2] = 5.0   # lagged speed channel (would be fed back)
    x_i[:, 3] = 1.0
    yt = np.arange(6, dtype=np.float64)

    # Ablated arm: channels 2-3 declared zeroed -> the feedback write is
    # followed by zeroing, so every rollout step sees channels 2-3 == 0.
    sc._hN_trial(model, x_i, yt, torch.device("cpu"), horizon=2,
                 ablate_channels=(2, 3))
    steps = [c for c, is_rollout in captured if is_rollout]
    assert steps, "rollout produced no steps"
    for call in steps:
        assert torch.equal(call[0, 0, 2], torch.tensor(0.0))
        assert torch.equal(call[0, 0, 3], torch.tensor(0.0))

    # Unablated arm: the seed feedback (x_i[s, 2] = 5.0) is written into the
    # first rollout step; with a zero prediction it then decays to 0.0.
    captured.clear()
    sc._hN_trial(model, x_i, yt, torch.device("cpu"), horizon=2,
                 ablate_channels=())
    steps = [c for c, is_rollout in captured if is_rollout]
    assert steps, "rollout produced no steps"
    assert torch.equal(steps[0][0, 0, 2], torch.tensor(5.0))


def test_phase2_hN_trial_returns_overall_and_escape() -> None:
    """The rollout returns overall AND escape-frame triples/counts."""
    sc = _scorer()

    class _Zero(torch.nn.Module):
        def forward(self, x, lengths, return_internals=False, states=None):
            B, T, _ = x.shape
            internals = {"routing_gates": torch.full((B, T, 2), 0.5),
                         "lif_spikes": torch.zeros(B, T, 4)}
            y = torch.zeros(B, T)
            if states is not None:
                return y, internals, states
            return (y, internals) if return_internals else y

    x_i = torch.zeros(8, 8)
    yt = np.zeros(8, dtype=np.float64)
    yt[3] = 50.0  # one escape frame
    out = sc._hN_trial(_Zero(), x_i, yt, torch.device("cpu"), horizon=2,
                       ablate_channels=())
    assert len(out) == 8
    m, p, z, c, m_e, p_e, z_e, c_e = out
    assert c > 0
    # The escape subset is a subset of the overall frames.
    assert 0 < c_e <= c


def test_phase2_loader_target_mismatch_fails_closed() -> None:
    """B3: a loader target series that differs from the canonical series is
    refused (per trial, in order)."""
    sc = _scorer()
    good = np.array([1.0, 2.0, 3.0])
    sc._require_loader_targets_match(good, good.copy(), 0)  # no raise
    with pytest.raises(ValueError):
        sc._require_loader_targets_match(np.array([1.0, 2.0, 3.0]),
                                         np.array([1.0, 2.0, 3.5]), 4)


def test_phase2_ceiling_reconciliation_matches_and_delta() -> None:
    """M3: a measured MSE within rel 1e-3 matches; a drifted one does not and
    carries a signed ceiling_vs_measured_delta."""
    sc = _scorer()
    ok = sc._reconcile_ceiling(1.9100, 1.910)
    assert ok["matches"] is True
    assert ok["ceiling_vs_measured_delta"] == pytest.approx(0.0, abs=1e-6)
    bad = sc._reconcile_ceiling(1.970, 1.910)
    assert bad["matches"] is False
    assert bad["ceiling_vs_measured_delta"] == pytest.approx(0.060, abs=1e-6)


def test_phase2_main_fails_closed_on_ceiling_mismatch(
    monkeypatch, tmp_path: Path,
) -> None:
    """M3: a scored persistence MSE that no longer matches the frozen ceiling
    aborts the run (fail closed) instead of reading the skill against it."""
    import numpy as np

    sc = _scorer()
    n = 12
    y_seqs = [np.abs(np.random.default_rng(i).standard_normal(30)) for i in range(n)]
    monkeypatch.setattr(
        sc, "extract_canonical_validation_targets",
        lambda d, p: (y_seqs, [f"t{i}" for i in range(n)], [f"p{i}" for i in range(n)]),
    )
    monkeypatch.setattr(sc, "_validate_arm_config", lambda ckpt, arm: None)
    monkeypatch.setattr(sc, "load_artifact_bytes",
                        lambda raw, map_location=None: {})
    monkeypatch.setattr(sc, "_sha256", lambda p: "0" * 64)

    counts = {s: np.full(n, 10, dtype=int) for s in sc.ALL_SCOPES}
    pa = {k: np.zeros(n, dtype=int) for k in
          ("pred_frame_num", "pred_frame_den", "pred_trial_num",
           "pred_trial_den", "animal_frame_num", "animal_frame_den",
           "animal_trial_num", "animal_trial_den")}

    def _fake_scope(cp, dp, npp, dev, yts):
        model = {s: np.full(n, 1.0) for s in sc.ALL_SCOPES}
        # Persistence is 2x the frozen ceiling -> reconciliation must fail.
        persist = {s: np.full(n, 2.0 * float(sc.PERSISTENCE_CEILING[s]["mse"]))
                   for s in sc.ALL_SCOPES}
        zero = {s: np.full(n, 5.0) for s in sc.ALL_SCOPES}
        return ({"model": model, "persistence": persist, "zero": zero},
                counts, np.zeros(n), np.linspace(1.0, 5.0, n),
                np.full(n, 1.5), np.full(n, 10, dtype=int),
                {"zero_input_channels": [],
                 "rollout_regime": "feedback_then_arm_transform"}, pa)

    monkeypatch.setattr(sc, "_per_trial_scope_mse", _fake_scope)
    argv = []
    for arm in sc.ARMS:
        ps = []
        for seed in (42, 43, 44):
            f = tmp_path / f"{arm}-{seed}.pth"
            f.write_bytes(b"x")
            ps.append(f)
        argv += [f"--{arm.lower()}", *[str(p) for p in ps]]
    argv += ["--dataset", str(tmp_path / "d.pt"),
             "--nested_prior", str(tmp_path / "p.pt"),
             "--output_dir", str(tmp_path / "out"),
             "--exchangeability_asserted"]
    with pytest.raises(ValueError):
        sc.main(argv)


def test_phase2_prestim_false_alarm_metric() -> None:
    """F2: the pre-stimulus false-alarm counts separate model vs animal."""
    sc = _scorer()
    # 5 pre-stimulus frames; animal stays below threshold; the model predicts
    # a 2-frame sustained escape bout in the middle.
    yt = np.zeros(10)
    yp = np.zeros(10)
    yp[1] = 20.0
    yp[2] = 30.0
    onset = 5
    (pf_num, pf_den, pt_num, pt_den,
     af_num, af_den, at_num, at_den) = sc._prestim_false_alarm(yp, yt, onset)
    assert pf_den == 5 and af_den == 5          # 5 pre-stimulus frames
    assert pf_num == 2                          # two predicted >= 10 while below
    assert af_num == 0                          # animal never escaped
    assert pt_num == 1 and pt_den == 1          # model predicted a bout
    assert at_num == 0 and at_den == 1          # animal had no bout
    # An isolated single-frame spike is NOT a bout (>= 2 frames required).
    yp2 = np.zeros(10)
    yp2[3] = 20.0
    assert sc._prestim_false_alarm(yp2, yt, onset)[2] == 0
    # No pre-stimulus segment -> all zero (no crash, no fabricated rate).
    assert sc._prestim_false_alarm(yp, yt, 0) == (0, 0, 0, 0, 0, 0, 0, 0)

