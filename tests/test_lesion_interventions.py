"""BIO-LESION-002: readout-intervention regressions for scripts/simulate_lesion.py.

Covers the distinction between pathway *substitutions* (zero one gate, force the
complementary gate to 1.0) and unrenormalized single-key *readout knockouts*
(zero one gate, retain the other's natural time-varying value). Exercises the
real tiny-CPU NSMoRCore lesion hook, the condition registry, result pairing and
prediction-to-intact divergence.
"""
from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION
from nsmor.model_nsmor_core import NSMoRCore


def _diagnostic_lineage(dataset_path):
    from nsmor.pipeline.nested_prior import compute_source_fingerprint

    return {
        'is_nested_cv': False, 'validation_scope': 'diagnostic_global_oof',
        'mcmc_prior_provenance': 'oof_2fold_recording_prefix_grouped_cv',
        'animal_identity_status': 'unverified',
        'dataset_source_sha256': compute_source_fingerprint(dataset_path),
        'dataset_source_binding': 'sha256_bound', 'dataset_path': str(dataset_path),
    }


# =========================================================================
# 1. Real tiny-CPU model hook: single-key knockouts retain natural complement
# =========================================================================

def test_single_key_knockout_retains_natural_complement_and_branch_states():
    """The model's single-key override must not force the complementary gate."""
    from scripts import simulate_lesion as lesion

    model = NSMoRCore(hidden_dim=4, dt_ms=4.0, sensory_noise_std=0.0).eval()
    x = torch.randn(2, 6, 8)
    x[..., 4:] = 0.25
    lengths = torch.tensor([6, 5])

    def run(override):
        with torch.no_grad():
            return model(x, lengths, return_internals=True, override_gates=override)

    _, intact = run(None)
    _, lif_knockout = run(lesion.condition_override('lif_knockout'))
    _, gru_knockout = run(lesion.condition_override('gru_knockout'))
    _, lif_sub = run(lesion.condition_override('lif_lesioned'))

    natural = intact['natural_gates']
    assert natural.shape == (2, 6, 2)

    # LIF knockout zeroes g_lif and keeps the natural g_gru trajectory.
    assert torch.all(lif_knockout['routing_gates'][:, :, 0] == 0)
    torch.testing.assert_close(lif_knockout['routing_gates'][:, :, 1], natural[:, :, 1])
    # GRU knockout zeroes g_gru and keeps the natural g_lif trajectory.
    assert torch.all(gru_knockout['routing_gates'][:, :, 1] == 0)
    torch.testing.assert_close(gru_knockout['routing_gates'][:, :, 0], natural[:, :, 0])
    # The substitution forces the complementary gate to 1.0 instead.
    assert torch.all(lif_sub['routing_gates'][:, :, 1] == 1.0)
    assert not torch.allclose(lif_knockout['routing_gates'][:, :, 1],
                              lif_sub['routing_gates'][:, :, 1])

    # Both recurrent branches keep computing; their states are readout-invariant.
    torch.testing.assert_close(lif_knockout['gru_hidden'], intact['gru_hidden'])
    torch.testing.assert_close(gru_knockout['lif_potentials'], intact['lif_potentials'])
    torch.testing.assert_close(lif_knockout['lif_spikes'], intact['lif_spikes'])


# =========================================================================
# 2. Condition registry: knockouts are single-key, substitutions force 1.0
# =========================================================================

def test_condition_registry_knockouts_are_single_key():
    from scripts import simulate_lesion as lesion

    assert lesion.LESION_OVERRIDES['intact'] is None
    assert lesion.LESION_OVERRIDES['lif_lesioned'] == {'g_lif': 0.0, 'g_gru': 1.0}
    assert lesion.LESION_OVERRIDES['gru_lesioned'] == {'g_lif': 1.0, 'g_gru': 0.0}
    assert lesion.READOUT_KNOCKOUT_OVERRIDES == {
        'lif_knockout': {'g_lif': 0.0}, 'gru_knockout': {'g_gru': 0.0},
    }
    # Swapped-gate guard: each knockout keys only its own gate.
    assert set(lesion.condition_override('lif_knockout')) == {'g_lif'}
    assert set(lesion.condition_override('gru_knockout')) == {'g_gru'}
    # Default condition set stays at the three substitutions.
    assert set(lesion.CONDITION_NAMES) == set(lesion.DEFAULT_CONDITIONS) == {
        'intact', 'lif_lesioned', 'gru_lesioned',
    }
    assert set(lesion.READOUT_KNOCKOUT_CONDITIONS) == {'lif_knockout', 'gru_knockout'}
    assert 'lif_knockout' not in lesion.CONDITION_NAMES


# =========================================================================
# 3. run_full_ablation dispatch order and per-condition overrides
# =========================================================================

class _Sequences:
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)


class _Loader(list):
    def __init__(self, batches, sequences):
        super().__init__(batches)
        self.dataset = _Sequences(sequences)


class _GateRecorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, lengths, override_gates=None):
        self.calls.append(override_gates)
        return x[:, :, 0]


def test_run_full_ablation_dispatches_single_key_overrides_for_knockouts():
    from scripts import simulate_lesion as lesion

    x = torch.zeros(1, 4, 8)
    y = torch.zeros(1, 4)
    lengths = torch.tensor([4])
    sequences = [(x[0], y[0], 0)]
    loader = _Loader([(x, y, lengths)], sequences)
    model = _GateRecorder()

    conditions = lesion.DEFAULT_CONDITIONS + lesion.READOUT_KNOCKOUT_CONDITIONS
    results = lesion.run_full_ablation(model, loader, torch.device('cpu'), conditions)

    assert list(results) == list(conditions)
    assert model.calls == [None, {'g_lif': 0.0, 'g_gru': 1.0},
                           {'g_lif': 1.0, 'g_gru': 0.0},
                           {'g_lif': 0.0}, {'g_gru': 0.0}]


# =========================================================================
# 4. End-to-end CLI: five conditions, pairing, prediction divergence
# =========================================================================

def _write_dataset(path, lengths, anchors, bases, targets):
    features = []
    for length, base in zip(lengths, bases):
        x = np.zeros((length, 8), dtype=np.float32)
        x[:, 0] = base
        features.append(x)
    torch.save({
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'mcmc_prior_provenance': 'oof_2fold_recording_prefix_grouped_cv',
        'animal_identity_status': 'unverified',
        'X_seqs': features, 'Y_seqs': targets, 'labels': np.zeros(len(lengths), dtype=int),
        'lengths': lengths, 'anchor_frames': anchors,
        'anchor_rules': ['recorded_snapshot_ttc_minus_50ms'] * len(lengths),
        'mcmc_priors': np.full((len(lengths), 4), 0.25, dtype=np.float32),
        'session_ids': [f'recording_{i}_session_1' for i in range(len(lengths))],
        'trial_ids': [100 + i for i in range(len(lengths))],
    }, path)


def test_five_condition_cli_pairs_trials_and_reports_prediction_divergence(
        monkeypatch, tmp_path):
    from scripts import simulate_lesion as lesion

    lengths = [6, 4]
    anchors = [1, 1]
    bases = [5.0, 7.0]
    targets = [np.full(6, 10.0, dtype=np.float32), np.full(4, 20.0, dtype=np.float32)]
    dataset_path = tmp_path / 'interventions.pt'
    _write_dataset(dataset_path, lengths, anchors, bases, targets)

    # Gate-dependent constant readout shift; the recorded target is unchanged.
    shifts = {None: 0.0, 'lif_lesioned': 1.0, 'gru_lesioned': 2.0,
              'lif_knockout': 3.0, 'gru_knockout': 4.0}

    class ShiftModel(torch.nn.Module):
        dt_ms = 4.0
        analysis_checkpoint_lineage = _diagnostic_lineage(dataset_path)

        def forward(self, x, lengths, override_gates=None):
            name = 'intact'
            for candidate in ('lif_lesioned', 'gru_lesioned', 'lif_knockout', 'gru_knockout'):
                if override_gates == lesion.condition_override(candidate):
                    name = candidate
            return x[:, :, 0] + shifts[None if override_gates is None else name]

    monkeypatch.setattr(lesion, 'load_model_from_checkpoint', lambda *_a: ShiftModel())
    panels = []
    monkeypatch.setattr(plt, 'savefig',
                        lambda *_a, **_k: panels.extend(plt.gcf().axes))
    csv_path = tmp_path / 'lesion.csv'
    lesion.main(['--checkpoint', str(tmp_path / 'unused.pth'), '--dataset', str(dataset_path),
                 '--output', str(tmp_path / 'lesion.png'), '--stats_output', str(csv_path),
                 '--batch_size', '2', '--include_readout_knockouts'])

    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5 * len(lengths)  # Five conditions, two eligible trials.
    assert {row['Condition'] for row in rows} == {
        lesion.condition_display(name) for name in
        lesion.DEFAULT_CONDITIONS + lesion.READOUT_KNOCKOUT_CONDITIONS}
    # Recorded target peaks are identical across conditions, so target-peak
    # metrics cannot distinguish them; prediction divergence can.
    assert {row['Peak_Velocity_cms'] for row in rows} == {'10.0000', '20.0000'}
    assert len(panels) == 5

    summary = json.loads(csv_path.with_suffix('.block_sensitivity.json').read_text())
    assert set(summary['conditions']) == {
        'intact', 'lif_lesioned', 'gru_lesioned', 'lif_knockout', 'gru_knockout'}
    assert set(summary['comparisons']) == {
        'lif_lesioned', 'gru_lesioned', 'lif_knockout', 'gru_knockout'}
    assert summary['intervention_semantics'] == (
        'readout_gate_intervention_not_biological_lesion')
    # Peak/latency columns describe the recorded target, not model predictions,
    # relative to a trial reference that is not a verified stimulus onset.
    assert summary['peak_latency_metrics_source'] == (
        'recorded_target_not_model_prediction')
    assert summary['peak_latency_reference'] == (
        'trial_reference_not_verified_stimulus_onset')

    divergence = summary['prediction_divergence']
    assert divergence['lif_knockout']['mean_trial_rmse'] == pytest.approx(3.0)
    assert divergence['gru_knockout']['mean_trial_rmse'] == pytest.approx(4.0)
    assert divergence['lif_lesioned']['mean_trial_rmse'] == pytest.approx(1.0)
    assert divergence['gru_lesioned']['mean_trial_rmse'] == pytest.approx(2.0)
    assert all(entry['unit'] == 'cm/s' for entry in divergence.values())
    assert all(entry['status'] == 'descriptive_only' for entry in divergence.values())

    # Per-trial pairing: row index, observed window and lengths align.
    for index, length in enumerate(lengths):
        trial = summary['trials'][index]
        expected_frames = length - anchors[index]
        for name in lesion.READOUT_KNOCKOUT_CONDITIONS:
            entry = trial['prediction_divergence_by_condition'][name]
            assert entry['n_frames'] == expected_frames
            assert entry['mae'] == pytest.approx(entry['rmse'])
        assert trial['post_reference_frames']['lif_knockout'] == expected_frames


def test_default_cli_still_emits_three_conditions(monkeypatch, tmp_path):
    from scripts import simulate_lesion as lesion

    lengths, anchors, bases = [4], [1], [5.0]
    targets = [np.full(4, 10.0, dtype=np.float32)]
    dataset_path = tmp_path / 'default.pt'
    _write_dataset(dataset_path, lengths, anchors, bases, targets)

    class IdentityModel(torch.nn.Module):
        dt_ms = 4.0
        analysis_checkpoint_lineage = _diagnostic_lineage(dataset_path)

        def forward(self, x, lengths, override_gates=None):
            return x[:, :, 0]

    monkeypatch.setattr(lesion, 'load_model_from_checkpoint', lambda *_a: IdentityModel())
    monkeypatch.setattr(plt, 'savefig', lambda *_a, **_k: None)
    csv_path = tmp_path / 'lesion.csv'
    lesion.main(['--checkpoint', str(tmp_path / 'unused.pth'), '--dataset', str(dataset_path),
                 '--output', str(tmp_path / 'lesion.png'), '--stats_output', str(csv_path)])
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(csv_path.with_suffix('.block_sensitivity.json').read_text())
    assert len(rows) == 3 * len(lengths)
    assert set(summary['conditions']) == set(lesion.DEFAULT_CONDITIONS)
    # Divergence includes the intact self-comparison (a zero-divergence control).
    assert set(summary['prediction_divergence']) == set(lesion.DEFAULT_CONDITIONS)
    assert summary['prediction_divergence']['intact']['mean_trial_rmse'] == 0.0
