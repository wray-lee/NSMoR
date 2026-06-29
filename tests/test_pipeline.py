"""
End-to-end pipeline validation with synthetic data.

Generates fake kinematics / events CSVs, runs the full pipeline
(load → label → extract → train MCMC → DataLoader), and asserts
all tensor shapes and invariants.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nsmor.config import (
    DEFAULT_FEATURE,
    DEFAULT_MCMC_TRAINING,
    DEFAULT_THRESHOLD,
    DEFAULT_TIME_WINDOW,
    FeatureConfig,
    Label,
    MCMCTrainingConfig,
    ThresholdConfig,
    TimeWindowConfig,
)
from nsmor.pipeline.io import (
    extract_trial_data,
    load_and_concat_sessions,
)
from nsmor.pipeline.labeling import assign_ground_truth_labels
from nsmor.data_extractor import (
    PURE_WIND_PREPEND_FRAMES,
    build_sequence_dataset,
    build_snapshot_dataset,
    extract_mcmc_snapshot,
    extract_trial_sequence,
)
from nsmor.mcmc_module import (
    MCMCPriorGenerator,
    MCMCPriorSKLearn,
    MarkovTransitionEstimator,
    train_mcmc,
)
from nsmor.nsmor_dataloader import create_dataloader
from nsmor.model_nsmor_core import NSMoR


# ═══════════════════════════════════════════════════════════════
# Synthetic data generators
# ═══════════════════════════════════════════════════════════════

def _make_synthetic_csvs(
    tmp_dir: Path,
    n_sessions: int = 2,
    trials_per_session: int = 10,
    frames_per_trial: int = 400,
    dt_ms: float = 10.0,
) -> tuple[Path, Path]:
    """
    Write synthetic kinematics + events CSVs to *tmp_dir*.

    Trials are labelled deterministically by *trial_id*:
      0-2 → Startle (high velocity spike after stimulus)
      3-5 → Walk (sustained moderate velocity)
      6-7 → Pre_Active (high baseline velocity)
      8-9 → NoResponse (no movement)
    """
    kin_rows = []
    evt_rows = []

    for s in range(n_sessions):
        sid = f"session_{s}"
        for t in range(trials_per_session):
            time_ms = np.arange(frames_per_trial) * dt_ms
            stimulus_onset = 2000.0  # 2 s baseline

            # Baseline velocity: low or high (pre-active)
            if 6 <= t <= 7:
                base_vel = np.random.uniform(0.8, 1.5, size=frames_per_trial)
            else:
                base_vel = np.random.uniform(0.0, 0.2, size=frames_per_trial)

            velocity = base_vel.copy()

            # Post-stimulus response
            stim_idx = int(stimulus_onset / dt_ms)
            if t <= 2:
                # Startle: sharp velocity spike
                spike_start = stim_idx + np.random.randint(5, 20)
                spike_end = min(spike_start + 5, frames_per_trial)
                velocity[spike_start:spike_end] = np.random.uniform(8.0, 15.0)
            elif 3 <= t <= 5:
                # Walk: sustained moderate velocity
                walk_start = stim_idx + np.random.randint(10, 50)
                walk_end = min(walk_start + 80, frames_per_trial)
                velocity[walk_start:walk_end] = np.random.uniform(1.5, 4.0)

            acceleration = np.gradient(velocity, dt_ms / 1000.0)

            # Visual angle: ramp from 0 after stimulus
            visual_angle = np.zeros(frames_per_trial)
            visual_angle[stim_idx:] = np.linspace(
                5.0, 60.0, frames_per_trial - stim_idx,
            )

            wind_state = np.zeros(frames_per_trial)
            if t % 2 == 0:
                wind_state[stim_idx:] = 1.0

            l_v_ratio = visual_angle * 0.1

            for f in range(frames_per_trial):
                kin_rows.append({
                    "session_id": sid,
                    "trial_id": t,
                    "time_ms": float(time_ms[f]),
                    "x_pos": float(f * 0.01),
                    "y_pos": float(f * 0.005),
                    "heading": 0.0,
                    "velocity": float(velocity[f]),
                    "acceleration": float(acceleration[f]),
                    "visual_angle": float(visual_angle[f]),
                    "wind_state": float(wind_state[f]),
                    "l_v_ratio": float(l_v_ratio[f]),
                })

            # Events
            evt_rows.append({
                "session_id": sid,
                "trial_id": t,
                "time_ms": 0.0,
                "event_type": "trial_start",
                "event_value": 1,
            })
            evt_rows.append({
                "session_id": sid,
                "trial_id": t,
                "time_ms": stimulus_onset,
                "event_type": "stimulus_onset",
                "event_value": 1,
            })

    kin_df = pd.DataFrame(kin_rows)
    evt_df = pd.DataFrame(evt_rows)

    kin_path = tmp_dir / "kinematics.csv"
    evt_path = tmp_dir / "events.csv"
    kin_df.to_csv(kin_path, index=False)
    evt_df.to_csv(evt_path, index=False)

    return kin_path, evt_path


def _make_pure_wind_csvs(
    tmp_dir: Path,
    frames_per_trial: int = 400,
    dt_ms: float = 10.0,
) -> tuple[Path, Path]:
    """
    Write synthetic CSVs for a Pure Wind trial (visual_angle ≡ 0).
    """
    kin_rows = []
    evt_rows = []

    time_ms = np.arange(frames_per_trial) * dt_ms
    stimulus_onset = 2000.0
    stim_idx = int(stimulus_onset / dt_ms)

    velocity = np.random.uniform(0.0, 0.1, size=frames_per_trial)
    acceleration = np.gradient(velocity, dt_ms / 1000.0)
    visual_angle = np.zeros(frames_per_trial)       # ← pure wind: no looming
    wind_state = np.zeros(frames_per_trial)
    wind_state[stim_idx:] = 1.0
    l_v_ratio = np.zeros(frames_per_trial)

    for f in range(frames_per_trial):
        kin_rows.append({
            "session_id": "wind_session",
            "trial_id": 0,
            "time_ms": float(time_ms[f]),
            "x_pos": float(f * 0.01),
            "y_pos": float(f * 0.005),
            "heading": 0.0,
            "velocity": float(velocity[f]),
            "acceleration": float(acceleration[f]),
            "visual_angle": float(visual_angle[f]),
            "wind_state": float(wind_state[f]),
            "l_v_ratio": float(l_v_ratio[f]),
        })

    evt_rows.append({
        "session_id": "wind_session",
        "trial_id": 0,
        "time_ms": 0.0,
        "event_type": "trial_start",
        "event_value": 1,
    })
    evt_rows.append({
        "session_id": "wind_session",
        "trial_id": 0,
        "time_ms": stimulus_onset,
        "event_type": "stimulus_onset",
        "event_value": 1,
    })

    kin_df = pd.DataFrame(kin_rows)
    evt_df = pd.DataFrame(evt_rows)

    kin_path = tmp_dir / "kinematics_wind.csv"
    evt_path = tmp_dir / "events_wind.csv"
    kin_df.to_csv(kin_path, index=False)
    evt_df.to_csv(evt_path, index=False)

    return kin_path, evt_path


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

class TestPipelineIO:
    """Tests for pipeline.io module."""

    def test_load_and_concat_sessions(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])
        assert "kinematics" in data
        assert "events" in data
        assert len(data["kinematics"]) > 0
        assert len(data["events"]) > 0

    def test_extract_trial_data(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])
        trial = extract_trial_data(data, "session_0", 0)
        assert trial["session_id"] == "session_0"
        assert trial["trial_id"] == 0
        assert len(trial["time_ms"]) == 400
        assert trial["velocity"].dtype == np.float64


class TestLabeling:
    """Tests for pipeline.labeling module."""

    def test_assign_labels(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = []
        for t in range(10):
            trials.append(extract_trial_data(data, "session_0", t))

        labeled = assign_ground_truth_labels(trials)
        assert len(labeled) == 10

        labels = [info["label"] for info in labeled]
        # Trials 0-2 should be Escape (startle-like response)
        assert all(l == Label.ESCAPE for l in labels[:3])
        # Trials 6-7 should be Pre_Active
        assert all(l == Label.PRE_ACTIVE for l in labels[6:8])


class TestSnapshotExtraction:
    """Tests for data_extractor snapshot functions."""

    def test_extract_snapshot_shape(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])
        trial = extract_trial_data(data, "session_0", 0)

        snapshot = extract_mcmc_snapshot(trial, stimulus_onset_ms=2000.0)
        assert snapshot.shape == (5,)
        assert snapshot.dtype == np.float64

    def test_build_snapshot_dataset(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(10)]
        labeled = assign_ground_truth_labels(trials)

        snapshots, labels = build_snapshot_dataset(labeled)
        assert snapshots.shape == (10, 5)
        assert labels.shape == (10,)


class TestSequenceExtraction:
    """Tests for data_extractor sequence functions."""

    def test_extract_sequence_shape(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])
        trial = extract_trial_data(data, "session_0", 0)

        X_seq, Y_seq = extract_trial_sequence(trial)
        assert X_seq.shape == (400, 8)
        assert Y_seq.shape == (400,)

        # First frame: t-1 features should be zero
        assert X_seq[0, 2] == 0.0  # v_kine(-1)
        assert X_seq[0, 3] == 0.0  # a_kine(-1)

    def test_build_sequence_dataset(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(10)]
        labeled = assign_ground_truth_labels(trials)

        sequences = build_sequence_dataset(labeled)
        assert len(sequences) == 10
        X_seq, Y_seq, label = sequences[0]
        assert X_seq.shape[1] == 8

    def test_pure_wind_baseline_prepend(self, tmp_path: Path) -> None:
        """Pure Wind trials get 570 zero-frames prepended."""
        kin_path, evt_path = _make_pure_wind_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])
        trial = extract_trial_data(data, "wind_session", 0)

        X_seq, Y_seq = extract_trial_sequence(trial)
        expected_len = PURE_WIND_PREPEND_FRAMES + 400  # 570 + 400 = 970
        assert X_seq.shape == (expected_len, 8), (
            f"Expected ({expected_len}, 8), got {X_seq.shape}"
        )
        assert Y_seq.shape == (expected_len,)

        # Prepended region should be all zeros
        assert np.all(X_seq[:PURE_WIND_PREPEND_FRAMES, :] == 0.0)
        assert np.all(Y_seq[:PURE_WIND_PREPEND_FRAMES] == 0.0)

        # Original region should have non-zero wind after stimulus
        assert np.any(X_seq[PURE_WIND_PREPEND_FRAMES:, 1] != 0.0)


class TestMCMCModule:
    """Tests for mcmc_module."""

    def test_pytorch_forward_shape(self) -> None:
        model = MCMCPriorGenerator()
        x = torch.randn(3, 5)
        probs = model(x)
        assert probs.shape == (3, 4)
        assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)

    def test_pytorch_single_sample(self) -> None:
        model = MCMCPriorGenerator()
        x = torch.randn(5)
        probs = model(x)
        assert probs.shape == (4,)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_predict_proba_numpy(self) -> None:
        model = MCMCPriorGenerator()
        x = np.random.randn(5).astype(np.float64)
        probs = model.predict_proba(x)
        assert probs.shape == (4,)
        assert np.allclose(probs.sum(), 1.0, atol=1e-5)

    def test_train_mcmc(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(10)]
        labeled = assign_ground_truth_labels(trials)
        snapshots, labels = build_snapshot_dataset(labeled)

        model = train_mcmc(
            snapshots, labels,
            config=MCMCTrainingConfig(num_epochs=50),
            verbose=False,
        )
        probs = model.predict_proba(snapshots[0])
        assert probs.shape == (4,)
        assert np.allclose(probs.sum(), 1.0, atol=1e-5)

    def test_sklearn_wrapper(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(10)]
        labeled = assign_ground_truth_labels(trials)
        snapshots, labels = build_snapshot_dataset(labeled)

        model = MCMCPriorSKLearn()
        model.fit(snapshots, labels)
        probs = model.predict_proba(snapshots[0])
        n_classes = len(np.unique(labels))
        assert probs.shape[0] == n_classes, (
            f"probs.shape[0]={probs.shape[0]} != n_classes={n_classes}"
        )
        assert np.allclose(probs.sum(), 1.0, atol=1e-5)

    def test_markov_transition(self) -> None:
        estimator = MarkovTransitionEstimator(num_states=4)
        seq = np.array([0, 0, 1, 1, 2, 3, 3, 3])
        estimator.fit([seq])
        assert estimator.transition_matrix is not None
        assert estimator.transition_matrix.shape == (4, 4)
        # Rows should sum to 1
        assert np.allclose(
            estimator.transition_matrix.sum(axis=1), 1.0, atol=1e-10,
        )


class TestNSMoRDataLoader:
    """Tests for nsmor_dataloader."""

    def test_dataloader_with_priors(self, tmp_path: Path) -> None:
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(10)]
        labeled = assign_ground_truth_labels(trials)
        snapshots, labels = build_snapshot_dataset(labeled)
        sequences = build_sequence_dataset(labeled)

        # Train MCMC and get priors
        model = train_mcmc(
            snapshots, labels,
            config=MCMCTrainingConfig(num_epochs=50),
            verbose=False,
        )
        priors = model.predict_proba(snapshots)

        loader = create_dataloader(
            sequences, mcmc_priors=priors, batch_size=4,
        )

        for X_batch, Y_batch, lengths in loader:
            bs = X_batch.shape[0]
            seq_len = X_batch.shape[1]
            assert X_batch.shape == (bs, seq_len, 8), (
                f"X_batch shape {X_batch.shape} != (bs, seq_len, 8)"
            )
            assert Y_batch.shape == (bs, seq_len), (
                f"Y_batch shape {Y_batch.shape} != (bs, seq_len)"
            )
            assert lengths.shape == (bs,), (
                f"lengths shape {lengths.shape} != (bs,)"
            )
            assert lengths.dtype == torch.int64
            assert (lengths <= seq_len).all(), (
                f"Some lengths exceed seq_len: {lengths}"
            )

            # MCMC probabilities should sum to 1
            mcmc = X_batch[:, :, 4:8]
            sums = mcmc.sum(dim=2)
            assert torch.allclose(sums, torch.ones(bs, seq_len), atol=1e-4), (
                f"MCMC prob sums: min={sums.min():.6f} max={sums.max():.6f}"
            )
            break  # one batch is enough

    def test_dataloader_requires_priors(self, tmp_path: Path) -> None:
        """NSMoRDataset raises ValueError when mcmc_priors is None."""
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(3)]
        labeled = assign_ground_truth_labels(trials)
        sequences = build_sequence_dataset(labeled)

        with pytest.raises(ValueError, match="mcmc_priors is required"):
            create_dataloader(sequences, mcmc_priors=None, batch_size=2)


class TestNSMoRModel:
    """Tests for the NSMoR Mixture-of-Recursions network."""

    def test_forward_shape(self) -> None:
        B, T, H = 4, 100, 32
        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=H)
        X_batch = torch.randn(B, T, 8)
        lengths = torch.tensor([100, 80, 50, 20], dtype=torch.int64)

        model.eval()
        with torch.no_grad():
            Y_pred = model(X_batch, lengths)

        assert Y_pred.shape == (B, T), (
            f"Y_pred shape {Y_pred.shape} != ({B}, {T})"
        )

    def test_forward_variable_lengths(self) -> None:
        """Different length sequences produce correct shapes."""
        B, T, H = 2, 60, 16
        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=H)
        X_batch = torch.randn(B, T, 8)
        lengths = torch.tensor([60, 30], dtype=torch.int64)

        model.eval()
        with torch.no_grad():
            Y_pred = model(X_batch, lengths)

        assert Y_pred.shape == (B, T)

    def test_forward_single_sample(self) -> None:
        """Batch size 1 works correctly."""
        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=16)
        X = torch.randn(1, 50, 8)
        lengths = torch.tensor([50], dtype=torch.int64)

        model.eval()
        with torch.no_grad():
            Y = model(X, lengths)

        assert Y.shape == (1, 50)

    def test_gradient_flow(self) -> None:
        """Gradients flow through both LIF and GRU paths."""
        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=16)
        X = torch.randn(2, 40, 8, requires_grad=True)
        lengths = torch.tensor([40, 20], dtype=torch.int64)

        Y = model(X, lengths)
        loss = Y.sum()
        loss.backward()

        assert X.grad is not None
        assert X.grad.shape == (2, 40, 8)
        # Valid frames should have non-zero gradient
        assert X.grad[:, :20, :].abs().sum() > 0

    def test_invalid_feature_dim_raises(self) -> None:
        """ValueError when feature dim != 8."""
        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=16)
        X_bad = torch.randn(2, 40, 6)  # wrong dim
        lengths = torch.tensor([40, 20], dtype=torch.int64)

        with pytest.raises(ValueError, match="Expected feature dim 8"):
            model(X_bad, lengths)

    def test_end_to_end_pipeline(self, tmp_path: Path) -> None:
        """Full pipeline: CSV → DataLoader → NSMoR forward."""
        kin_path, evt_path = _make_synthetic_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trials = [extract_trial_data(data, "session_0", t) for t in range(10)]
        labeled = assign_ground_truth_labels(trials)
        snapshots, labels = build_snapshot_dataset(labeled)
        sequences = build_sequence_dataset(labeled)

        mcmc_model = train_mcmc(
            snapshots, labels,
            config=MCMCTrainingConfig(num_epochs=50),
            verbose=False,
        )
        priors = mcmc_model.predict_proba(snapshots)

        loader = create_dataloader(
            sequences, mcmc_priors=priors, batch_size=4,
        )

        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for X_batch, Y_batch, lengths in loader:
            optimizer.zero_grad()
            Y_pred = model(X_batch, lengths)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            assert loss.isfinite(), f"Loss is not finite: {loss}"
            break  # one step is enough

    def test_end_to_end_with_pure_wind(self, tmp_path: Path) -> None:
        """Pure Wind trials (with 570-frame prepend) flow through model."""
        kin_path, evt_path = _make_pure_wind_csvs(tmp_path)
        data = load_and_concat_sessions([kin_path], [evt_path])

        trial = extract_trial_data(data, "wind_session", 0)
        X_seq, Y_seq = extract_trial_sequence(trial)

        # Should be 570 + 400 = 970 frames
        assert X_seq.shape[0] == 970

        # Wrap in a minimal dataset with dummy priors
        priors = np.array([[0.1, 0.1, 0.7, 0.1]], dtype=np.float64)
        sequences = [(X_seq, Y_seq, Label.NO_RESPONSE)]

        loader = create_dataloader(sequences, mcmc_priors=priors, batch_size=1)

        model = NSMoR(sensory_dim=4, mcmc_dim=4, hidden_dim=16)
        model.eval()
        for X_batch, Y_batch, lengths in loader:
            with torch.no_grad():
                Y_pred = model(X_batch, lengths)
            assert Y_pred.shape == (1, 970)
            break


# Need nn import for the end-to-end test
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════
# CLI Override Tests (Flaw #2: regression guard for build_config)
# ═══════════════════════════════════════════════════════════════

class TestCLIOverrides:
    """Verify that build_config correctly applies CLI overrides."""

    def test_freeze_modules_override(self):
        """--freeze lif_cell router sets config.finetune.freeze_modules."""
        from scripts.train import build_config
        config, _ = build_config(["--freeze", "lif_cell", "router"])
        assert config.finetune.freeze_modules == ["lif_cell", "router"]

    def test_lr_override(self):
        """--lr 5e-4 sets training.learning_rate."""
        from scripts.train import build_config
        config, _ = build_config(["--lr", "5e-4"])
        assert config.training.learning_rate == 5e-4

    def test_epochs_override(self):
        """--epochs 200 sets training.num_epochs."""
        from scripts.train import build_config
        config, _ = build_config(["--epochs", "200"])
        assert config.training.num_epochs == 200

    def test_hidden_dim_override(self):
        """--hidden_dim 128 sets model.hidden_dim."""
        from scripts.train import build_config
        config, _ = build_config(["--hidden_dim", "128"])
        assert config.model.hidden_dim == 128

    def test_batch_size_override(self):
        """--batch_size 64 sets training.batch_size."""
        from scripts.train import build_config
        config, _ = build_config(["--batch_size", "64"])
        assert config.training.batch_size == 64

    def test_output_dir_override(self):
        """--output_dir runs/test sets checkpoint.output_dir."""
        from scripts.train import build_config
        config, _ = build_config(["--output_dir", "runs/test"])
        assert config.checkpoint.output_dir == "runs/test"

    def test_lambda_reg(self):
        """--lambda_reg 0.05 returns lambda_reg=0.05."""
        from scripts.train import build_config
        _, lambda_reg = build_config(["--lambda_reg", "0.05"])
        assert lambda_reg == 0.05

    def test_unfreeze_after_epoch_in_config(self):
        """unfreeze_after_epoch field exists in FineTuneConfig."""
        from nsmor.config_parser import FineTuneConfig
        cfg = FineTuneConfig()
        assert hasattr(cfg, "unfreeze_after_epoch")
        assert cfg.unfreeze_after_epoch == -1


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
