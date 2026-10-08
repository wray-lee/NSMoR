"""Analysis unit/cadence regressions.

Round 2 DEVELOP: D/E/F/G/I inherit saved cadence and reject explicit mismatch;
actual C Panel B/C, D CSV/figure and E epoch-selection regressions added.
Phase D round 2 adds observed per-trial reference and descriptive group regressions.
Disjoint from tests/test_precollection_mutation_contracts.py.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nsmor.analysis import prediction_units
from nsmor.config import PIPELINE_SEMANTICS_VERSION
from nsmor.model_nsmor_core import NSMoRCore
from nsmor.model_utils import load_model_from_checkpoint as canonical_load
from nsmor.pipeline.nested_prior import load_artifact_bytes


def _diagnostic_lineage(dataset_path):
    from nsmor.pipeline.nested_prior import compute_source_fingerprint

    return {
        'is_nested_cv': False, 'validation_scope': 'diagnostic_global_oof',
        'mcmc_prior_provenance': 'oof_2fold_recording_prefix_grouped_cv',
        'animal_identity_status': 'unverified',
        'dataset_source_sha256': compute_source_fingerprint(dataset_path),
        'dataset_source_binding': 'sha256_bound', 'dataset_path': str(dataset_path),
    }


@pytest.fixture
def normalized_checkpoint(tmp_path):
    model = NSMoRCore(hidden_dim=4, dt_ms=4.0, sensory_noise_std=0.0)
    payload = {
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'config': {'model': {'hidden_dim': 4, 'dt_ms': 4.0, 'sensory_noise_std': 0.0},
                   'training': {'normalize_targets': True}},
        'model_state_dict': model.state_dict(),
        'target_mean': 5.0, 'target_std': 2.0, 'target_clip_cm_s': 100.0,
    }
    path = tmp_path / 'normalized.pth'
    torch.save(payload, path)
    return path, payload


@pytest.mark.parametrize('kwargs', [{}, {'return_internals': True}, {'return_internals': True, 'states': {}}])
def test_loaded_outputs_restore_physical_units_once_and_preserve_api(normalized_checkpoint, kwargs):
    path, _ = normalized_checkpoint
    device = torch.device('cpu')
    raw = canonical_load(path, device)
    physical = prediction_units.load_model_from_checkpoint(path, device)
    x = torch.zeros(1, 3, 8)
    x[..., 4:] = 0.25
    lengths = torch.tensor([3])
    raw_out, out = raw(x, lengths, **kwargs), physical(x, lengths, **kwargs)
    raw_y = raw_out[0] if isinstance(raw_out, tuple) else raw_out
    y = out[0] if isinstance(out, tuple) else out
    assert y.shape == raw_y.shape == (1, 3)  # Shape alone cannot catch wrong units.
    torch.testing.assert_close(y, raw_y * 2.0 + 5.0)
    assert not torch.allclose(y, raw_y)
    if isinstance(raw_out, tuple):
        assert len(out) == len(raw_out)
        for raw_extra, extra in zip(raw_out[1:], out[1:]):
            assert raw_extra.keys() == extra.keys()
            for key in extra:
                torch.testing.assert_close(extra[key], raw_extra[key])
    assert raw.state_dict().keys() == physical.state_dict().keys()
    for key, value in raw.state_dict().items():
        torch.testing.assert_close(physical.state_dict()[key], value)


def test_direct_backend_prediction_has_same_physical_units(normalized_checkpoint):
    path, _ = normalized_checkpoint
    raw = canonical_load(path, torch.device('cpu'))
    physical = prediction_units.load_model_from_checkpoint(path, torch.device('cpu'))
    encoded = torch.zeros(1, 3, 4)
    prior = torch.full((1, 3, 4), 0.25)
    lengths = torch.tensor([3])
    raw_y, raw_internals = raw.backend(encoded, prior, lengths, return_internals=True)
    y, internals = physical.backend(encoded, prior, lengths, return_internals=True)
    assert y.shape == raw_y.shape == (1, 3)
    torch.testing.assert_close(y, raw_y * 2.0 + 5.0)
    for key in internals:
        torch.testing.assert_close(internals[key], raw_internals[key])


def test_unit_conversion_is_unclipped_shape_safe_and_differentiable():
    model = SimpleNamespace(target_mean=5.0, target_std=2.0, target_clip_cm_s=100.0)
    raw = torch.tensor([[-1.0, 0.0, 2.0, 60.0]], dtype=torch.float64, requires_grad=True)
    actual = prediction_units.prediction_to_physical(raw, model)
    assert actual.shape == raw.shape and actual.dtype == raw.dtype
    torch.testing.assert_close(actual, torch.tensor([[3.0, 5.0, 9.0, 125.0]], dtype=torch.float64))
    assert actual[0, -1] > model.target_clip_cm_s  # Target clipping is not an inference bound.
    actual.sum().backward()
    torch.testing.assert_close(raw.grad, torch.full_like(raw, 2.0))
    with pytest.raises(ValueError, match='Nonfinite'):
        prediction_units.prediction_to_physical(torch.tensor([[float('nan')]]), model)


@pytest.mark.parametrize('explicit_identity', [False, True])
def test_legacy_unnormalized_checkpoint_preserves_predictions(normalized_checkpoint, explicit_identity):
    path, payload = normalized_checkpoint
    if explicit_identity:
        payload['config']['training']['normalize_targets'] = False
        payload.update(target_mean=0.0, target_std=1.0)
    else:
        payload['config'].pop('training')
        for key in ('target_mean', 'target_std', 'target_clip_cm_s'):
            payload.pop(key)
    torch.save(payload, path)
    raw = canonical_load(path, torch.device('cpu'))
    physical = prediction_units.load_model_from_checkpoint(path, torch.device('cpu'))
    x, lengths = torch.zeros(1, 3, 8), torch.tensor([3])
    torch.testing.assert_close(physical(x, lengths), raw(x, lengths), rtol=0, atol=0)
    assert physical.target_mean == 0.0 and physical.target_std == 1.0


@pytest.mark.parametrize('change', [
    {'target_mean': None}, {'target_std': None}, {'target_clip_cm_s': None},
    {'target_mean': float('nan')}, {'target_std': float('inf')},
    {'target_std': 0.0}, {'target_std': -1.0}, {'target_std': True},
    {'target_mean': '5.0'}, {'target_clip_cm_s': -1.0}, {'target_clip_cm_s': float('inf')},
    {'config': {'training': {'normalize_targets': False}}},
    {'config': {'training': {'normalize_targets': 'yes'}}},
])
def test_bad_or_incomplete_normalization_fails_before_model_build(tmp_path, monkeypatch, change):
    payload = {'config': {'training': {'normalize_targets': True}},
               'target_mean': 5.0, 'target_std': 2.0, 'target_clip_cm_s': 100.0}
    for key, value in change.items():
        if value is None:
            payload.pop(key)
        else:
            payload[key] = value
    path = tmp_path / 'bad.pth'
    torch.save(payload, path)
    monkeypatch.setattr(prediction_units, '_canonical_load_model',
                        lambda *_args: pytest.fail('Invalid metadata reached canonical model build'))
    with pytest.raises(ValueError):
        prediction_units.load_model_from_checkpoint(path, torch.device('cpu'))


def test_autoregressive_feedback_uses_physical_velocity_acceleration_and_gain(normalized_checkpoint, monkeypatch):
    from scripts import simulate_autoregressive as ar

    class ProbeBackend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []
            self.states_seen = []

        def forward(self, x, lengths, *, return_internals, states):
            assert x.shape == (1, 1, 8) and lengths.shape == (1,)
            self.inputs.append(x.detach().clone())
            self.states_seen.append(states)
            step = len(self.inputs)
            y = torch.full((1, 1), float(step), dtype=x.dtype)
            internals = {'routing_gates': torch.full((1, 1, 2), 0.5)}
            return y, internals, {'step': step}

    class ProbeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backend = ProbeBackend()
            self.dt_ms = 4.0

        def forward(self, x, lengths, **kwargs):
            return self.backend(x, lengths, **kwargs)

    path, _ = normalized_checkpoint
    probe = ProbeModel()
    monkeypatch.setattr(prediction_units, '_canonical_load_model', lambda *_args: probe)
    model = prediction_units.load_model_from_checkpoint(path, torch.device('cpu'))
    monkeypatch.setattr(ar, 'generate_stimulus_paradigm',
                        lambda *_args: (np.array([0.0, 4.0, 8.0]), np.zeros(3), np.zeros(3)))
    result = ar.run_autoregressive_trial(model, ar.PARADIGMS['visual_only'],
                                        np.full(4, 0.25), torch.device('cpu'), dt_ms=4.0,
                                        current_fatigue=0.5, max_fatigue_penalty=0.2)
    expected_velocity = np.array([7.0, 9.0, 11.0]) * 0.9
    expected_acceleration = np.diff(np.r_[0.0, expected_velocity]) / 0.004
    np.testing.assert_allclose(result.velocity, expected_velocity, rtol=1e-6)
    np.testing.assert_allclose(result.acceleration, expected_acceleration, rtol=1e-6)
    np.testing.assert_allclose(result.position, np.cumsum(expected_velocity) * 0.004, rtol=1e-6)
    for index, inputs in enumerate(probe.backend.inputs):
        previous_v = 0.0 if index == 0 else expected_velocity[index - 1]
        previous_a = 0.0 if index == 0 else expected_acceleration[index - 1]
        torch.testing.assert_close(inputs[0, 0, 2:4], torch.tensor([previous_v, previous_a], dtype=inputs.dtype))
    assert probe.backend.states_seen == [{}, {'step': 1}, {'step': 2}]


@pytest.mark.parametrize('dt_ms', [4.0, 10.0])
def test_phase_c_actual_panel_b_and_c_axes_use_checkpoint_clock(monkeypatch, tmp_path, dt_ms):
    import matplotlib.pyplot as plt
    from scripts import analyze_dynamics as dynamics

    bundle = dynamics.DynamicsBundle()
    bundle.labels = [0, 0]
    bundle.g_gru_trajs = [np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6])]
    bundle.g_lif_trajs = [1.0 - values for values in bundle.g_gru_trajs]
    bundle.lif_rate_trajs = [np.array([0.0, 0.2, 0.4]), np.array([0.1, 0.3, 0.5])]
    monkeypatch.setattr(dynamics, 'load_model_from_checkpoint', lambda *_args: SimpleNamespace(dt_ms=dt_ms))
    monkeypatch.setattr(dynamics, 'load_dataset', lambda *_args, **_kwargs: (None, np.array([0, 0])))
    monkeypatch.setattr(dynamics, 'extract_full_dynamics', lambda *_args: bundle)
    monkeypatch.setattr(dynamics, 'compute_pca_manifold',
                        lambda *_args, **_kwargs: ([], [], SimpleNamespace(explained_variance_ratio_=np.ones(3) / 3)))
    # Only unrelated A/D panels are stubbed; run_analysis and the real 2x2
    # figure render both cadence-sensitive B/C panels before savefig.
    monkeypatch.setattr(dynamics, 'plot_panel_a_3d_manifold', lambda *_args: None)
    monkeypatch.setattr(dynamics, 'plot_panel_d_pathway_dominance', lambda *_args: None)
    recorded = {}

    def capture_axes(*_args, **_kwargs):
        axes = plt.gcf().axes
        assert len(axes) == 4
        for panel, index in (('B', 1), ('C', 2)):
            assert axes[index].lines, f'Actual Panel {panel} did not plot a trajectory'
            recorded[panel] = axes[index].lines[0].get_xdata().copy()

    monkeypatch.setattr(plt, 'savefig', capture_axes)
    dynamics.run_analysis(tmp_path / 'checkpoint.pth', tmp_path / 'unused.pt', tmp_path / 'unused.png')
    for panel in ('B', 'C'):
        np.testing.assert_allclose(recorded[panel], np.arange(3) * dt_ms / 1000.0)


@pytest.mark.parametrize('return_internals', [False, True])
def test_jax_weight_copy_restores_units_without_torch_backend_hook(normalized_checkpoint, monkeypatch, return_internals):
    # Stub JAX's apply result to check wrapper semantics without requiring a JAX device.
    from nsmor.analysis import jax_eval

    path, _ = normalized_checkpoint
    model = prediction_units.load_model_from_checkpoint(path, torch.device('cpu'))
    monkeypatch.setattr(jax_eval, 'JAX_AVAILABLE', True)
    monkeypatch.setattr(jax_eval, 'NSMoRModel', lambda **_kwargs: object())
    monkeypatch.setattr(jax_eval, 'load_from_torch_state_dict', lambda *_args, **_kwargs: {})
    monkeypatch.setattr(jax_eval, 'jnp', SimpleNamespace(asarray=np.asarray))
    wrapper = jax_eval.JAXEvalWrapper.from_torch(model, torch.device('cpu'))
    assert (wrapper.target_mean, wrapper.target_std, wrapper.target_clip_cm_s) == (5.0, 2.0, 100.0)

    class ReadyArray:
        def __array__(self, dtype=None, copy=None):
            return np.full((1, 3), 2.0, dtype=dtype or np.float32)

        def block_until_ready(self):
            return self

    gates = np.full((1, 3, 2), 0.5, dtype=np.float32)
    monkeypatch.setattr(wrapper, '_apply_fn',
                        lambda internals, _overrides: lambda *_args: (ReadyArray(), {'routing_gates': gates}) if internals else ReadyArray())
    output = wrapper(torch.zeros(1, 3, 8), torch.tensor([3]), return_internals=return_internals)
    y = output[0] if return_internals else output
    torch.testing.assert_close(y, torch.full((1, 3), 9.0))
    if return_internals:
        torch.testing.assert_close(output[1]['routing_gates'], torch.from_numpy(gates))


@pytest.mark.parametrize('dt_ms', [4.0, 10.0])
@pytest.mark.parametrize('explicit', [False, True])
def test_phase_d_cli_csv_latency_figure_axis_and_raw_mse(monkeypatch, tmp_path, dt_ms, explicit):
    import matplotlib.pyplot as plt
    from nsmor.analysis import uq
    from scripts import simulate_lesion as lesion

    pred = np.array([0.0, 25.0, 0.0, 125.0, 0.0])
    true = np.array([0.0, 30.0, 0.0, 200.0, 0.0])
    results = {'intact': ([pred], [true], [0])}
    monkeypatch.setattr(lesion, 'load_model_from_checkpoint', lambda *_args: SimpleNamespace(dt_ms=dt_ms))
    monkeypatch.setattr(lesion, 'load_dataset', lambda *_args, **_kwargs: (None, np.array([0]), [5]))
    monkeypatch.setattr(lesion, 'run_full_ablation', lambda *_args: results)
    monkeypatch.setattr(uq, 'bootstrap_ci', lambda values, *_args, **_kwargs: (float(np.mean(values)),) * 3)
    axes_data = {}

    def capture_figure(*_args, **_kwargs):
        axis = plt.gcf().axes[0]
        assert len(axis.lines) >= 2
        axes_data['time_ms'] = axis.lines[0].get_xdata().copy()
        axes_data['prediction'] = axis.lines[1].get_ydata().copy()

    monkeypatch.setattr(plt, 'savefig', capture_figure)
    stats_path = tmp_path / 'lesion.csv'
    argv = ['--checkpoint', str(tmp_path / 'unused.pth'), '--dataset', str(tmp_path / 'unused.pt'),
            '--output', str(tmp_path / 'unused.png'), '--stats_output', str(stats_path), '--stim_onset_frame', '1']
    if explicit:
        argv += ['--dt_ms', str(dt_ms)]
    lesion.main(argv)
    with stats_path.open(newline='') as stream:
        row, = csv.DictReader(stream)
    assert float(row['Latency_to_Peak_ms']) == 2 * dt_ms
    assert float(row['Peak_Velocity_cms']) == 200.0
    expected_mse = float(np.mean((pred[1:] - true[1:]) ** 2))
    clipped_mse = float(np.mean((np.clip(pred[1:], -100, 100) - np.clip(true[1:], -100, 100)) ** 2))
    assert float(row['MSE']) == pytest.approx(expected_mse) and expected_mse != clipped_mse
    n_frames = len(pred) - 1  # Only observed post-onset samples.
    np.testing.assert_allclose(axes_data['time_ms'], np.arange(n_frames) * dt_ms)
    assert axes_data['time_ms'][0] == 0.0
    assert axes_data['prediction'][0] == pred[1] == 25.0  # Onset sample is t=0.
    np.testing.assert_allclose(axes_data['prediction'][:4], pred[1:])
    assert np.max(axes_data['prediction']) == 125.0  # No inference clipping to 100.
    with pytest.raises(ValueError, match='No observed post-reference'):
        lesion.average_trajectories_by_class(
            [np.array([99.0])], [np.array([99.0])], [0], 0,
            dt_ms=dt_ms, max_time_ms=2 * dt_ms, stim_onset_frame=1,
        )


@pytest.mark.parametrize('dt_ms', [4.0, 10.0])
@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('wind_present', [False, True])
def test_phase_e_cli_selects_actual_epoch_frames_from_saved_clock(monkeypatch, tmp_path, dt_ms, explicit, wind_present):
    from scripts import analyze_jacobian as jacobian

    class ProbeModel(torch.nn.Module):
        # r6 R8: legacy probe exports only gru_hidden; declare that the routed
        # output IS the raw recurrent coordinate (no gain).
        gru_hidden_is_raw = True

        def __init__(self):
            super().__init__()
            self.dt_ms, self.sensory_dim = dt_ms, 4
            self.sensory_encoder = lambda x: x[:, :, 2:3]

        def forward(self, x, lengths, *, return_internals):
            return torch.zeros(x.shape[:2]), {'gru_hidden': x[:, :, 2:3]}

    class Dataset(list):
        @property
        def sequences(self):
            return self

    class Loader(list):
        dataset = Dataset([(None, None, 1)])

    x = torch.zeros(1, 601, 8)
    x[0, 300:, 1 if wind_present else 0] = 1.0
    x[0, :, 2] = torch.arange(601)
    loader = Loader([(x, torch.zeros(1, 601), torch.tensor([601]))])
    model, adapter = ProbeModel(), SimpleNamespace()
    monkeypatch.setattr(jacobian, 'load_model_from_checkpoint', lambda *_args: model)
    monkeypatch.setattr(jacobian, 'load_dataset', lambda *_args, **_kwargs: (loader, np.array([1]), [601], [x[0].numpy()]))
    monkeypatch.setattr(jacobian, 'create_jacobian_adapter', lambda *_args, **_kwargs: adapter)
    # Isolate cadence from slow-point optimization and GMM calibration while
    # exercising the real GRU extraction loop that selects each epoch frame.
    monkeypatch.setattr(jacobian, '_find_slow_point', lambda h, centre, _length: (centre, h[centre]))
    monkeypatch.setattr(jacobian, '_fixed_point_residual', lambda *_args: 0.1)
    monkeypatch.setattr(jacobian, '_calibrate_fp_threshold', lambda *_args, **_kwargs: (1.0, {}))
    actual_extract = jacobian.extract_gru_states_at_epochs
    observed = {}

    class EpochsCollected(Exception):
        pass

    def capture_epochs(**kwargs):
        states = actual_extract(**kwargs)
        for name in ('early', 'transient', 'sustained'):
            hidden, encoded = states[name]
            assert hidden.shape == encoded.shape == (1, 1)
            observed[name] = int(hidden.item())
            torch.testing.assert_close(hidden, encoded)
        raise EpochsCollected  # No eigenvalue solver/plot is needed for cadence.

    monkeypatch.setattr(jacobian, 'extract_gru_states_at_epochs', capture_epochs)
    argv = ['--checkpoint', str(tmp_path / 'unused.pth'), '--dataset', str(tmp_path / 'unused.pt')]
    if explicit:
        argv += ['--dt_ms', str(dt_ms)]
    with pytest.raises(EpochsCollected):
        jacobian.main(argv)
    offset = int(1000.0 / dt_ms)
    assert observed == {'early': 300 - offset, 'transient': 300, 'sustained': 300 + offset}


@pytest.mark.parametrize('module_name', ['simulate_lesion', 'analyze_jacobian', 'analyze_integration',
                                        'simulate_psychophysics', 'simulate_autoregressive'])
@pytest.mark.parametrize('saved_dt, requested_dt', [(4.0, 10.0), (10.0, 4.0)])
def test_model_driven_cli_rejects_explicit_mismatch_before_data_or_generation(monkeypatch, tmp_path, module_name, saved_dt, requested_dt):
    import importlib
    import sys
    module = importlib.import_module('scripts.' + module_name)
    loader = 'load_checkpoint' if module_name == 'simulate_psychophysics' else 'load_model_from_checkpoint'
    monkeypatch.setattr(module, loader, lambda *_args: SimpleNamespace(dt_ms=saved_dt))
    data_loader = 'load_validation_data' if module_name == 'simulate_psychophysics' else 'load_dataset'
    if hasattr(module, data_loader):
        monkeypatch.setattr(module, data_loader, lambda *_args, **_kwargs: pytest.fail('Mismatched clock reached data load'))
    if module_name == 'simulate_autoregressive':
        monkeypatch.setattr(module, 'generate_stimulus_paradigm', lambda *_args: pytest.fail('Mismatched clock reached generation'))
    argv = ['--checkpoint', str(tmp_path / 'unused.pth'), '--dt_ms', str(requested_dt)]
    with pytest.raises(ValueError, match='conflicts with saved'):
        if module_name in ('simulate_psychophysics', 'simulate_autoregressive'):
            monkeypatch.setattr(sys, 'argv', [module_name, *argv, '--output_dir', str(tmp_path)])
            module.main()
        else:
            module.main(argv)


@pytest.mark.parametrize('bad_dt', [0.0, -4.0, float('nan'), float('inf'), True])
def test_bad_cadence_fails_loudly(bad_dt):
    with pytest.raises(ValueError, match='finite positive'):
        prediction_units.resolve_dt_ms(SimpleNamespace(dt_ms=bad_dt))
    with pytest.raises(ValueError, match='finite positive'):
        prediction_units.resolve_dt_ms(SimpleNamespace(dt_ms=4.0), bad_dt)


@pytest.mark.parametrize('dt_ms, onset', [(4.0, 1200), (10.0, 1)])
def test_phase_d_figure_and_csv_use_only_observed_post_onset_samples(monkeypatch, tmp_path, dt_ms, onset):
    import matplotlib.pyplot as plt
    from nsmor.analysis import uq
    from scripts import simulate_lesion as lesion

    if dt_ms == 4.0:
        predictions = [np.r_[np.zeros(onset), np.full(1200, 5.0)]]
        expected = np.full(1200, 5.0)
    else:
        predictions = [np.array([0.0, 5.0, 6.0]),
                       np.array([0.0, 10.0]), np.array([99.0])]
        expected = np.array([7.5, 6.0])
    results = {'intact': (predictions, predictions, [0] * len(predictions))}
    monkeypatch.setattr(lesion, 'load_model_from_checkpoint',
                        lambda *_args: SimpleNamespace(dt_ms=dt_ms))
    monkeypatch.setattr(lesion, 'load_dataset',
                        lambda *_args, **_kwargs: (None, np.zeros(len(predictions)),
                                                  [len(x) for x in predictions]))
    monkeypatch.setattr(lesion, 'run_full_ablation', lambda *_args: results)
    monkeypatch.setattr(uq, 'bootstrap_ci',
                        lambda values, *_args, **_kwargs: (float(np.mean(values)),) * 3)
    captured = {}

    def capture_figure(*_args, **_kwargs):
        line = plt.gcf().axes[0].lines[1]
        captured['x'] = line.get_xdata().copy()
        captured['y'] = line.get_ydata().copy()

    monkeypatch.setattr(plt, 'savefig', capture_figure)
    csv_path = tmp_path / 'lesion.csv'
    lesion.main(['--checkpoint', str(tmp_path / 'unused.pth'), '--dataset',
                 str(tmp_path / 'unused.pt'), '--output', str(tmp_path / 'unused.png'),
                 '--stats_output', str(csv_path), '--stim_onset_frame', str(onset)])
    np.testing.assert_array_equal(captured['x'], np.arange(len(expected)) * dt_ms)
    np.testing.assert_allclose(captured['y'], expected)
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row['Trial_ID']) for row in rows} == set(range(1 if dt_ms == 4 else 2))
    assert all(float(row['MSE']) == 0.0 for row in rows)

@pytest.mark.parametrize('dt_ms', [4.0, 10.0])
def test_phase_d_default_cli_uses_each_observed_cropped_reference(monkeypatch, tmp_path, dt_ms):
    """A 700-frame trial at anchor 200 has 500 real post-reference samples."""
    import json
    import matplotlib.pyplot as plt
    from scripts import simulate_lesion as lesion

    anchors = [200, 3000, 10]
    lengths = [700, 3200, 70]
    targets = []
    features = []
    for index, (length, anchor) in enumerate(zip(lengths, anchors)):
        y = np.full(length, 500.0, dtype=np.float32)  # Baseline must not enter D metrics.
        y[anchor:] = [50.0, 20.0, 900.0][index]
        y[anchor + [3, 7, 1][index]] = [80.0, 100.0, 999.0][index]
        x = np.zeros((length, 8), dtype=np.float32)
        x[:, 0] = y + 2.0
        targets.append(y)
        features.append(x)
    dataset_path = tmp_path / 'observed.pt'
    torch.save({
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'mcmc_prior_provenance': 'oof_2fold_recording_prefix_grouped_cv',
        'animal_identity_status': 'unverified',
        'X_seqs': features, 'Y_seqs': targets, 'labels': np.array([0, 0, 1]),
        'lengths': lengths, 'anchor_frames': anchors,
        'anchor_rules': ['recorded_snapshot_ttc_minus_50ms_visual',
                         'recorded_snapshot_wind_fallback', 'recorded_other_reference'],
        'mcmc_priors': np.full((3, 4), 0.25, dtype=np.float32),
        'session_ids': ['recording_A_session_1', 'recording_A_session_2',
                        'recording_B_session_1'],
    }, dataset_path)

    class PhysicalModel(torch.nn.Module):
        analysis_checkpoint_lineage = _diagnostic_lineage(dataset_path)

        def __init__(self):
            super().__init__()
            self.dt_ms = dt_ms

        def forward(self, x, lengths, override_gates=None):
            return x[:, :, 0]

    monkeypatch.setattr(lesion, 'load_model_from_checkpoint', lambda *_args: PhysicalModel())
    captured = {}

    def capture_figure(*_args, **_kwargs):
        captured['x'] = plt.gcf().axes[0].lines[0].get_xdata().copy()
        captured['prediction'] = plt.gcf().axes[0].lines[1].get_ydata().copy()
        captured['xlabel'] = plt.gcf().axes[0].get_xlabel()

    monkeypatch.setattr(plt, 'savefig', capture_figure)
    stats_path = tmp_path / 'lesion.csv'
    lesion.main(['--checkpoint', str(tmp_path / 'unused.pth'), '--dataset', str(dataset_path),
                 '--output', str(tmp_path / 'unused.png'), '--stats_output', str(stats_path),
                 '--batch_size', '2'])
    with stats_path.open(newline='') as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [row for row in all_rows if row['Class'] == 'ESCAPE']
    assert len(all_rows) == 9 and len(rows) == 6  # Includes all three non-ESCAPE rows.
    assert {int(row['Trial_ID']) for row in rows} == {0, 1}
    for row in rows:
        index = int(row['Trial_ID'])
        assert float(row['Peak_Velocity_cms']) == [80.0, 100.0][index]
        assert float(row['Latency_to_Peak_ms']) == [3, 7][index] * dt_ms
        assert float(row['MSE']) == 4.0
    np.testing.assert_array_equal(captured['x'], np.arange(500) * dt_ms)
    expected = np.full(500, 52.0)
    expected[:200] = 37.0
    expected[3] = 52.0
    expected[7] = 77.0
    np.testing.assert_allclose(captured['prediction'], expected)
    assert captured['xlabel'] == 'Time relative to trial reference (ms)'
    summary = json.loads(stats_path.with_suffix('.block_sensitivity.json').read_text())
    assert summary['dt_ms'] == dt_ms
    assert summary['mse_units'] == '(cm/s)^2' and summary['mse_window'] == 'observed_post_reference'
    for row in all_rows:
        name = next(name for name, display in lesion.CONDITION_NAMES.items()
                    if display == row['Condition'])
        trial = summary['trials'][int(row['Trial_ID'])]
        assert abs(float(row['MSE']) - trial['mse_by_condition'][name]) <= 5.01e-7
    assert sum(row['Class'] != 'ESCAPE' for row in all_rows) == 3
    assert all(row['Trial_ID'] == '2' for row in all_rows if row['Class'] != 'ESCAPE')
    for index, (start, end, frame, n_post) in enumerate([(0, 700, 200, 500),
                                                       (800, 3200, 2200, 200), (0, 70, 10, 60)]):
        trial = summary['trials'][index]
        assert trial['original_reference_frame'] == anchors[index]
        assert trial['trial_id'] == trial['row_index'] == index
        assert trial['source_trial_id'] is None  # Session alone does not establish a source trial ID.
        assert (trial['crop_start_frame'], trial['crop_end_frame']) == (start, end)
        assert trial['cropped_length'] == end - start
        assert trial['cropped_reference_frame'] == trial['analysis_reference_frame'] == frame
        assert trial['anchor_rule'] == ['recorded_snapshot_ttc_minus_50ms_visual',
                                        'recorded_snapshot_wind_fallback', 'recorded_other_reference'][index]
        assert trial['reference_source'] == trial['analysis_reference_source'] == 'recorded_dataset_anchor'
        assert trial['reference_status'] == 'recorded_reference_rule_available'
        assert trial['post_reference_frames'] == {name: n_post for name in lesion.CONDITION_NAMES}
    for comparison in summary['comparisons'].values():
        assert comparison['n_paired_recording_prefixes'] == 1
        assert comparison['effect_size'] is None
        assert comparison['effect_size_status'] == 'unavailable_fewer_than_two_recording_prefixes'

@pytest.mark.parametrize('paired', [False, True])
@pytest.mark.parametrize('left,right', [([2.0] * 4, [1.0] * 4),
                                        ([1.0] * 4, [1.0] * 4), ([2.0], [1.0])])
def test_cohens_d_undefined_denominator_is_not_zero_effect(paired, left, right):
    from nsmor.analysis.uq import cohens_d

    assert np.isnan(cohens_d(np.array(left), np.array(right), paired=paired))


@pytest.mark.parametrize('with_groups', [True, False])
def test_phase_d_describes_recording_prefixes_without_independent_animal_claims(
        monkeypatch, tmp_path, with_groups):
    import hashlib
    import json
    import matplotlib.pyplot as plt
    from scripts import simulate_lesion as lesion

    x_seqs, y_seqs = [], []
    for offset in [1.0, 1.0, 1.0, 3.0, 0.0]:
        # The fifth trial has no observed sample at/after its reference.
        length = 6 if offset else 1
        x = np.zeros((length, 8), dtype=np.float32)
        y = np.full(length, 10.0, dtype=np.float32)
        x[:, 0], x[:, 2] = 11.0, offset
        x[0, 0], y[0] = 1000.0, 100.0  # A large baseline error must be excluded.
        x_seqs.append(x)
        y_seqs.append(y)
    payload = {
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'mcmc_prior_provenance': ('oof_2fold_recording_prefix_grouped_cv'
                                  if with_groups else 'oof_2fold_animal_grouped_cv'),
        'animal_identity_status': 'unverified' if with_groups else 'historical_unknown',
        'X_seqs': x_seqs, 'Y_seqs': y_seqs, 'labels': np.zeros(5, dtype=int),
        'lengths': [6, 6, 6, 6, 1], 'anchor_frames': [1] * 5,
        'anchor_rules': ['recorded_snapshot_ttc_minus_50ms'] * 5,
        'mcmc_priors': np.full((5, 4), 0.25, dtype=np.float32),
    }
    if with_groups:
        payload['session_ids'] = ['recording_A_session_1', 'recording_A_session_2',
                                  'recording_A_session_3', 'recording_B_session_1',
                                  'missing_session_1']
    dataset_path = tmp_path / 'grouped.pt'
    torch.save(payload, dataset_path)
    checkpoint_path = tmp_path / 'unused.pth'
    if not with_groups:
        checkpoint_path.write_bytes(b'phase D historical checkpoint')
        historical_pin = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    class PhysicalModel(torch.nn.Module):
        dt_ms = 4.0
        analysis_checkpoint_lineage = (_diagnostic_lineage(dataset_path) if with_groups
                                       else {'is_nested_cv': False})
        analysis_checkpoint_sha256 = (hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
                                      if not with_groups else None)

        def forward(self, x, lengths, override_gates=None):
            if override_gates is None:
                return x[:, :, 0]
            return x[:, :, 0] + (1.0 if override_gates['g_lif'] == 0 else x[:, :, 2])

    monkeypatch.setattr(lesion, 'load_model_from_checkpoint', lambda *_args: PhysicalModel())
    monkeypatch.setattr(plt, 'savefig', lambda *_args, **_kwargs: None)
    csv_path = tmp_path / 'lesion.csv'
    argv = ['--checkpoint', str(checkpoint_path), '--dataset', str(dataset_path),
            '--output', str(tmp_path / 'unused.png'), '--stats_output', str(csv_path),
            '--batch_size', '2']
    if not with_groups:
        argv.extend(['--trusted_historical_checkpoint_sha256', historical_pin])
    lesion.main(argv)
    with csv_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['Class', 'Condition', 'Trial_ID', 'Peak_Velocity_cms',
                                     'Latency_to_Peak_ms', 'MSE']
        assert {int(row['Trial_ID']) for row in reader} == {0, 1, 2, 3}
    summary = json.loads(csv_path.with_suffix('.block_sensitivity.json').read_text())
    assert summary['analysis_status'] == 'descriptive_only'
    assert summary['mse_units'] == '(cm/s)^2'
    assert summary['trials'][4]['mse_by_condition'] == dict.fromkeys(lesion.CONDITION_NAMES)
    assert summary['conditions']['intact']['mean_trial_mse'] == 1.0
    assert summary['conditions']['gru_lesioned']['mean_trial_mse'] == 7.0
    assert summary['trials'][4]['post_reference_frames'] == {name: 0 for name in lesion.CONDITION_NAMES}
    assert summary['trials'][0]['anchor_rule'] == 'recorded_snapshot_ttc_minus_50ms'
    for comparison in summary['comparisons'].values():
        assert comparison['p_value'] is comparison['p_adjusted'] is comparison['significant'] is None
        assert comparison['p_status'] == 'unavailable_unverified_animal_identity_and_independence'
        assert comparison['n_paired_trials'] == 4
    lif = summary['comparisons']['lif_lesioned']
    gru = summary['comparisons']['gru_lesioned']
    if with_groups:
        assert summary['conditions']['gru_lesioned']['mean_recording_prefix_mse'] == 10.0
        assert lif['n_paired_recording_prefixes'] == gru['n_paired_recording_prefixes'] == 2
        assert lif['effect_size'] is None
        assert lif['effect_size_status'] == 'undefined_zero_variance_paired_differences'
        assert lif['delta_mse'] == 3.0
        assert gru['effect_size'] == pytest.approx(1.0606601717798212)
        assert gru['delta_mse'] == 9.0
        assert gru['effect_size_unit'] == 'paired_recording_prefix_means'
    else:
        assert lif['effect_size'] is gru['effect_size'] is None
        assert lif['effect_size_status'] == gru['effect_size_status'] == 'unavailable_recording_group_metadata'
        assert lif['n_paired_recording_prefixes'] == gru['n_paired_recording_prefixes'] == 0


def test_phase_d_strict_unpadded_loader_refuses_padding_then_exports_valid_control(
        monkeypatch, tmp_path):
    import hashlib
    import json
    import matplotlib.pyplot as plt
    from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint
    from scripts import simulate_lesion as lesion

    features = [np.zeros((4, 8), dtype=np.float32) for _ in range(2)]
    features[0][:, 0] = [10, 10, 0, 0]
    features[1][:, 0] = [5, 0, 0, 0]
    path = tmp_path / 'padded.pt'
    payload = {
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'mcmc_prior_provenance': 'oof_2fold_recording_prefix_grouped_cv',
        'animal_identity_status': 'unverified',
        'X_seqs': features,
        'Y_seqs': [np.array([5, 5, 0, 0], dtype=np.float32), np.zeros(4, dtype=np.float32)],
        'labels': np.array([0, 3]), 'lengths': [2, 1], 'anchor_frames': [1, 1],
        'anchor_rules': ['recorded_snapshot_ttc_minus_50ms'] * 2,
        'mcmc_priors': np.full((2, 4), 0.25, dtype=np.float32),
        'session_ids': ['animalA_session_1', 'animalB_session_1'], 'trial_ids': [11, 12],
        'stimulus_conditions': ['visual_only'] * 2,
    }
    torch.save(payload, path)
    for loader in (load_dataset_with_fingerprint, lesion.load_dataset):
        with pytest.raises(ValueError, match='unpadded'):
            loader(path)

    class PhysicalModel(torch.nn.Module):
        dt_ms = 4.0

        def __init__(self):
            super().__init__()
            self.analysis_checkpoint_lineage = _diagnostic_lineage(path)

        def forward(self, x, lengths, override_gates=None):
            assert x.shape == (2, 2, 8) and lengths.tolist() == [2, 1]
            return x[:, :, 0]

    monkeypatch.setattr(lesion, 'load_model_from_checkpoint', lambda *_args: PhysicalModel())
    monkeypatch.setattr(plt, 'savefig', lambda *_args, **_kwargs: None)
    csv_path = tmp_path / 'lesion.csv'
    argv = ['--checkpoint', str(tmp_path / 'unused.pth'), '--dataset', str(path),
            '--output', str(tmp_path / 'lesion.png'), '--stats_output', str(csv_path),
            '--batch_size', '2']
    with pytest.raises(ValueError, match='unpadded'):
        lesion.main(argv)
    assert not csv_path.exists() and not csv_path.with_suffix('.block_sensitivity.json').exists()

    payload['X_seqs'] = [x[:n] for x, n in zip(payload['X_seqs'], payload['lengths'])]
    payload['Y_seqs'] = [y[:n] for y, n in zip(payload['Y_seqs'], payload['lengths'])]
    torch.save(payload, path)
    lesion.main(argv)
    with csv_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['Class', 'Condition', 'Trial_ID', 'Peak_Velocity_cms',
                                     'Latency_to_Peak_ms', 'MSE']
        rows = list(reader)
    sidecar = json.loads(csv_path.with_suffix('.block_sensitivity.json').read_text())
    assert sidecar['csv_trial_id_semantics'] == 'zero_based_dataset_row_index'
    assert len(rows) == len(lesion.CONDITION_NAMES) == 3
    assert {row['Condition'] for row in rows} == set(lesion.CONDITION_NAMES.values())
    assert all(row['Class'] == 'ESCAPE' and row['Trial_ID'] == '0'
               and float(row['MSE']) == 25.0 for row in rows)
    for index, (source_id, expected_length, support) in enumerate([(11, 2, 1), (12, 1, 0)]):
        trial = sidecar['trials'][index]
        assert (trial['row_index'], trial['source_trial_id'], trial['session_id']) == (
            index, source_id, f'animal{chr(65 + index)}_session_1')
        assert (trial['crop_start_frame'], trial['crop_end_frame'], trial['cropped_length']) == (
            0, expected_length, expected_length)
        assert trial['post_reference_frames'] == dict.fromkeys(lesion.CONDITION_NAMES, support)
        assert trial['mse_by_condition'] == dict.fromkeys(lesion.CONDITION_NAMES,
                                                          25.0 if index == 0 else None)
        assert trial['post_reference_status'] == dict.fromkeys(
            lesion.CONDITION_NAMES, 'observed' if index == 0 else 'missing_reference')
    assert sidecar['trials'][0]['analysis_reference_frame'] == 1
    assert sidecar['trials'][1]['analysis_reference_frame'] is None
    assert sidecar['trials'][1]['reference_status'] == 'unavailable_reference_outside_crop'
    for row in rows:
        name = next(name for name, display in lesion.CONDITION_NAMES.items()
                    if display == row['Condition'])
        assert float(row['MSE']) == sidecar['trials'][int(row['Trial_ID'])]['mse_by_condition'][name]
    assert sidecar['analysis_status'] == 'descriptive_only'
    assert all(sidecar['conditions'][name]['n_finite_trials'] == 1
               for name in lesion.CONDITION_NAMES)


@pytest.mark.parametrize('field,value', [
    ('lengths', [True, 4]), ('lengths', [4, True]), ('lengths', [4.0, 4]),
    ('lengths', ['4', 4]), ('lengths', [0, 4]), ('lengths', [-1, 4]),
    ('lengths', [5, 4]), ('lengths', [4]), ('lengths', [4, 4, 4]),
    ('lengths', [[4], [4]]), ('lengths', 4), ('lengths', None),
    ('lengths', torch.tensor([True, True])), ('lengths', torch.tensor([[4], [4]])),
    ('X_seqs', [np.zeros((4, 8))]), ('Y_seqs', [np.zeros(4)]),
    ('labels', [0]), ('labels', 0), ('labels', np.zeros((2, 1))),
    ('X_seqs', [np.zeros((3, 8)), np.zeros((4, 8))]),
    ('Y_seqs', [np.zeros(4), np.zeros(3)]),
    ('X_seqs', [np.zeros(4), np.zeros((4, 8))]),
    ('Y_seqs', [np.zeros((4, 1)), np.zeros(4)]),
])
def test_shared_loader_rejects_invalid_declared_length_contract(tmp_path, field, value):
    from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint

    path = tmp_path / 'invalid_lengths.pt'
    payload = {
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'X_seqs': [np.zeros((4, 8), dtype=np.float32) for _ in range(2)],
        'Y_seqs': [np.zeros(4, dtype=np.float32) for _ in range(2)],
        'labels': [0, 3], 'lengths': [4, 4],
    }
    torch.save(dict(payload, **{field: value}), path)
    with pytest.raises(ValueError, match='length|unpadded|aligned'):
        load_dataset_with_fingerprint(path)


@pytest.mark.parametrize('lengths', [[2, 1], (2, 1), np.array([2, 1], dtype=np.int64),
                                     torch.tensor([2, 1])])
def test_shared_loader_accepts_exact_unpadded_lengths_and_keeps_byte_binding(tmp_path, lengths):
    import hashlib
    from nsmor.pipeline.nested_prior import load_dataset_with_fingerprint

    path = tmp_path / 'unpadded.pt'
    payload = {
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'X_seqs': [np.zeros((n, 8), dtype=np.float32) for n in (2, 1)],
        'Y_seqs': [np.zeros(n, dtype=np.float32) for n in (2, 1)],
        'labels': [0, 3], 'lengths': lengths,
    }
    torch.save(payload, path)
    loaded, digest = load_dataset_with_fingerprint(path, map_location='cpu')
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    np.testing.assert_array_equal(loaded['lengths'], [2, 1])
    for index, length in enumerate((2, 1)):
        assert loaded['X_seqs'][index].shape == (length, 8)
        assert loaded['Y_seqs'][index].shape == (length,)
    payload.pop('lengths')
    torch.save(payload, path)
    with pytest.raises(ValueError, match='lengths'):
        load_dataset_with_fingerprint(path)
    # Legacy artifacts without a declared modern length contract stay readable.
    payload.pop('pipeline_semantics_version')
    torch.save(payload, path)
    assert len(load_dataset_with_fingerprint(path)[0]['X_seqs']) == 2


@pytest.mark.parametrize('max_seq_len', [None, 2400])
def test_phase_d_loader_marks_derived_reference_rules_and_exact_crop(tmp_path, max_seq_len):
    from scripts import simulate_lesion as lesion

    x_seqs = [np.zeros((length, 8), dtype=np.float32) for length in [700, 3200, 10]]
    x_seqs[0][200:, 1] = 1.0  # First active wind frame.
    x_seqs[1][3000, 0] = 90.0  # Peak visual angle proxy, not an event-verified collision.
    y_seqs = [np.full(len(x), index + 1.0, dtype=np.float32) for index, x in enumerate(x_seqs)]
    dataset_path = tmp_path / 'derived.pt'
    torch.save({
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'mcmc_prior_provenance': 'oof_2fold_animal_grouped_cv',
        'X_seqs': x_seqs, 'Y_seqs': y_seqs, 'labels': np.zeros(3, dtype=int),
        'lengths': [700, 3200, 10], 'mcmc_priors': np.full((3, 4), 0.25, dtype=np.float32),
    }, dataset_path)
    loader, labels, lengths = lesion.load_dataset(dataset_path, batch_size=2, max_seq_len=max_seq_len)
    assert len(labels) == 3
    assert lengths == [700, 3200 if max_seq_len is None else 2400, 10]
    assert loader.dataset.cropped_reference_frames == [200, 3000 if max_seq_len is None else 2200, None]
    for index, rule in enumerate(['first_wind_channel_gt_0.5', 'peak_visual_angle_proxy',
                                   'no_stimulus_frame_zero_fallback']):
        reference = loader.dataset.analysis_reference_trials[index]
        assert reference['reference_source'] == 'derived_physical_channel_anchor'
        assert reference['anchor_rule'] == rule
        assert reference['recording_prefix'] is None
        assert reference['trial_id'] == reference['row_index'] == index
        assert reference['source_trial_id'] is reference['session_id'] is None
        assert reference['reference_status'] == [
            'derived_wind_threshold_reference', 'derived_peak_visual_proxy_reference',
            'unavailable_no_stimulus_reference'][index]
        x, y = loader.dataset[index]
        assert x.shape == (lengths[index], 8) and y.shape == (lengths[index],)
        if reference['cropped_reference_frame'] is not None:
            assert y[reference['cropped_reference_frame']] == index + 1.0
        else:
            assert reference['original_reference_frame'] == 0
    # The processed dataset stores derived anchors. A recorded frame-zero
    # fallback with no physical event still cannot support a time origin.
    saved = load_artifact_bytes(dataset_path.read_bytes())
    saved['anchor_frames'] = [200, 3000, 0]
    torch.save(saved, dataset_path)
    recorded, _, _ = lesion.load_dataset(dataset_path, batch_size=2, max_seq_len=max_seq_len)
    assert recorded.dataset.cropped_reference_frames[2] is None
    assert recorded.dataset.analysis_reference_trials[2]['reference_status'] == 'unavailable_no_stimulus_reference'


@pytest.mark.parametrize('dt_ms', [4.0, 10.0])
def test_phase_d_scalar_default_and_per_trial_metric_references_remain_supported(dt_ms):
    from scripts import simulate_lesion as lesion

    legacy = np.r_[np.full(1200, 999.0), [5.0, 6.0, 20.0]]
    short = np.array([999.0, 999.0, 10.0, 30.0])
    unavailable = np.array([999.0])
    default_metrics = lesion.extract_scalar_metrics([legacy], [legacy], [0], 0, dt_ms=dt_ms)
    assert default_metrics == {'Peak_Velocity_cms': 20.0, 'Latency_to_Peak_ms': 2 * dt_ms, 'Mean_MSE': 0.0}
    t, pred, true = lesion.average_trajectories_by_class([legacy], [legacy], [0], 0, dt_ms=dt_ms)
    np.testing.assert_array_equal(t, [0.0, dt_ms, 2 * dt_ms])
    np.testing.assert_array_equal(pred, [5.0, 6.0, 20.0])
    np.testing.assert_array_equal(pred, true)
    metrics = lesion.extract_scalar_metrics([legacy, short, unavailable], [legacy, short, unavailable],
                                            [0, 0, 0], 0, dt_ms=dt_ms, reference_frames=[1200, 2, None])
    assert metrics == {'Peak_Velocity_cms': 25.0, 'Latency_to_Peak_ms': 1.5 * dt_ms, 'Mean_MSE': 0.0}
    with pytest.raises(ValueError, match='No observed post-reference'):
        lesion.average_trajectories_by_class([unavailable], [unavailable], [0], 0,
                                             reference_frames=[None], dt_ms=dt_ms)


@pytest.mark.parametrize('references', [[1, 2], [-1], [1.5]])
def test_phase_d_invalid_reference_vectors_do_not_silently_realign_trials(references):
    from scripts import simulate_lesion as lesion

    with pytest.raises(ValueError, match='reference'):
        lesion.average_trajectories_by_class([np.array([5.0, 10.0])], [np.array([5.0, 10.0])],
                                             [0], 0, reference_frames=references)

@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_phase_d_nonfinite_observations_are_not_exported_as_measurements(tmp_path, bad):
    from scripts import simulate_lesion as lesion

    good, invalid = np.array([5.0, 20.0]), np.array([5.0, bad])
    metrics = lesion.extract_scalar_metrics([good, invalid], [good, invalid], [0, 0], 0,
                                            dt_ms=4.0, reference_frames=[0, 0])
    assert metrics == {'Peak_Velocity_cms': 20.0, 'Latency_to_Peak_ms': 4.0, 'Mean_MSE': 0.0}
    path = tmp_path / 'finite.csv'
    lesion.export_lesion_statistics_csv({'intact': ([good, invalid], [good, invalid], [0, 0])},
                                        path, [0], dt_ms=4.0, reference_frames=[0, 0])
    with path.open(newline='') as stream:
        row, = csv.DictReader(stream)
    assert row['Trial_ID'] == '0' and float(row['MSE']) == 0.0

@pytest.mark.parametrize('dt_ms', [4.0, 10.0])
def test_phase_d_cli_finite_trial_mix_keeps_figure_csv_and_bin_support(monkeypatch, tmp_path, dt_ms):
    import json
    import matplotlib.pyplot as plt
    from scripts import simulate_lesion as lesion

    targets = [np.array([100.0, 5.0, 20.0, 30.0], dtype=np.float32),
               np.array([100.0, 5.0, np.nan], dtype=np.float32),
               np.array([100.0, 10.0, 40.0], dtype=np.float32)]
    features = []
    for index, target in enumerate(targets):
        x = np.zeros((len(target), 8), dtype=np.float32)
        x[:, 0] = np.nan_to_num(target, nan=5.0) + 2.0
        features.append(x)
    dataset_path = tmp_path / 'mixed.pt'
    torch.save({
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'mcmc_prior_provenance': 'oof_2fold_recording_prefix_grouped_cv',
        'animal_identity_status': 'unverified',
        'X_seqs': features, 'Y_seqs': targets, 'labels': np.zeros(3, dtype=int),
        'lengths': [len(target) for target in targets], 'anchor_frames': [1, 1, 1],
        'anchor_rules': ['recorded_snapshot_ttc_minus_50ms'] * 3,
        'mcmc_priors': np.full((3, 4), 0.25, dtype=np.float32),
        'session_ids': ['recording_A_session_1'] * 3,
    }, dataset_path)

    class PhysicalModel(torch.nn.Module):
        analysis_checkpoint_lineage = _diagnostic_lineage(dataset_path)

        def __init__(self):
            super().__init__()
            self.dt_ms = dt_ms

        def forward(self, x, lengths, override_gates=None):
            return x[:, :, 0]

    monkeypatch.setattr(lesion, 'load_model_from_checkpoint', lambda *_args: PhysicalModel())
    panels = []

    def capture_figure(*_args, **_kwargs):
        for ax in plt.gcf().axes:
            panels.append(([(line.get_xdata().copy(), line.get_ydata().copy()) for line in ax.lines],
                           [text.get_text() for text in ax.texts]))

    monkeypatch.setattr(plt, 'savefig', capture_figure)
    stats_path = tmp_path / 'mixed.csv'
    lesion.main(['--checkpoint', str(tmp_path / 'unused.pth'), '--dataset', str(dataset_path),
                 '--output', str(tmp_path / 'mixed.png'), '--stats_output', str(stats_path),
                 '--batch_size', '2'])
    with stats_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['Class', 'Condition', 'Trial_ID', 'Peak_Velocity_cms',
                                     'Latency_to_Peak_ms', 'MSE']
        rows = list(reader)
    assert len(rows) == 6
    assert {row['Trial_ID'] for row in rows} == {'0', '2'}
    assert all(float(row['MSE']) == 4.0 for row in rows)
    assert len(panels) == 3
    for lines, texts in panels:
        assert not texts and len(lines) == 3  # target, prediction, reference at zero
        np.testing.assert_array_equal(lines[0][0], np.arange(3) * dt_ms)
        np.testing.assert_allclose(lines[0][1], [7.5, 30.0, 30.0])
        np.testing.assert_allclose(lines[1][1], [9.5, 32.0, 32.0])
    sidecar = json.loads(stats_path.with_suffix('.block_sensitivity.json').read_text())
    assert sidecar['schema_version'] == 2
    assert sidecar['analysis_status'] == 'descriptive_only'
    for name in lesion.CONDITION_NAMES:
        assert sidecar['conditions'][name]['n_finite_trials'] == 2
        assert sidecar['trials'][1]['post_reference_frames'][name] == 2
        assert sidecar['trials'][1]['post_reference_status'][name] == 'nonfinite_observations'
        assert sidecar['trials'][1]['mse_by_condition'][name] is None
        assert sidecar['trials'][2]['post_reference_status'][name] == 'observed'
        assert sidecar['trials'][2]['mse_by_condition'][name] == 4.0
    assert all(comp['p_adjusted'] is None for comp in sidecar['comparisons'].values())


def test_phase_d_cli_all_invalid_fails_and_invalidates_stale_csv(monkeypatch, tmp_path):
    import json
    import matplotlib.pyplot as plt
    from scripts import simulate_lesion as lesion

    dataset_path = tmp_path / 'no_reference.pt'
    torch.save({
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'mcmc_prior_provenance': 'oof_2fold_recording_prefix_grouped_cv',
        'animal_identity_status': 'unverified',
        'X_seqs': [np.zeros((2, 8), dtype=np.float32)],
        'Y_seqs': [np.array([10.0, 10.0], dtype=np.float32)],
        'labels': np.array([0]), 'lengths': [2], 'anchor_frames': [0],
        'mcmc_priors': np.full((1, 4), 0.25, dtype=np.float32),
        'session_ids': ['recording_A_session_1'],
    }, dataset_path)

    class PhysicalModel(torch.nn.Module):
        dt_ms = 4.0
        analysis_checkpoint_lineage = _diagnostic_lineage(dataset_path)

        def forward(self, x, lengths, override_gates=None):
            return x[:, :, 0]

    monkeypatch.setattr(lesion, 'load_model_from_checkpoint', lambda *_args: PhysicalModel())
    monkeypatch.setattr(plt, 'savefig', lambda *_args, **_kwargs: None)
    stats_path = tmp_path / 'stale.csv'
    stats_path.write_text('Class,Condition,Trial_ID,Peak_Velocity_cms,Latency_to_Peak_ms,MSE\n'
                          'ESCAPE,Intact Model,999,1,0,0\n')
    with pytest.raises(ValueError, match='No valid statistics'):
        lesion.main(['--checkpoint', str(tmp_path / 'unused.pth'), '--dataset', str(dataset_path),
                     '--output', str(tmp_path / 'no_reference.png'), '--stats_output', str(stats_path)])
    assert not stats_path.exists()
    sidecar = json.loads(stats_path.with_suffix('.block_sensitivity.json').read_text())
    assert sidecar['schema_version'] == 2
    assert sidecar['analysis_status'] == 'unavailable_no_valid_measurements'
    assert sidecar['trials'][0]['reference_status'] == 'unavailable_no_stimulus_reference'
    for name in lesion.CONDITION_NAMES:
        assert sidecar['conditions'][name]['n_finite_trials'] == 0
        assert sidecar['trials'][0]['post_reference_status'][name] == 'missing_reference'
        assert sidecar['trials'][0]['mse_by_condition'][name] is None
    assert all(comp['p_value'] is comp['p_adjusted'] is None
               for comp in sidecar['comparisons'].values())

@pytest.mark.parametrize('loader', [canonical_load, prediction_units.load_model_from_checkpoint],
                         ids=['canonical', 'analysis'])
def test_checkpoint_loaders_reject_reducer_before_side_effect(normalized_checkpoint, tmp_path, loader):
    path, payload = normalized_checkpoint
    marker = tmp_path / 'checkpoint_pickle_executed.txt'

    class Reducer:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",)

    torch.save(dict(payload, malicious_metadata=Reducer()), path)
    try:
        with pytest.raises(ValueError, match='Unexpected serialized global'):
            loader(path, torch.device('cpu'))
    finally:
        assert not marker.exists(), 'Checkpoint reducer ran before rejection'

def test_phase_d_csv_sidecar_joins_original_trials_across_sessions(monkeypatch, tmp_path):
    import json
    import matplotlib.pyplot as plt
    from scripts import simulate_lesion as lesion

    sessions = ['recording_A_session_1', 'recording_A_session_1', 'recording_B_session_1']
    source_ids = [101, 103, 101]  # Reused across sessions, nonconsecutive within A.
    peaks = [5.0, 9.0, 13.0]
    targets = [np.array([0.0, 0.0, peak, 0.0], dtype=np.float32) for peak in peaks]
    features = []
    for target in targets:
        x = np.zeros((4, 8), dtype=np.float32)
        x[:, 0] = target
        features.append(x)
    path = tmp_path / 'original_trials.pt'
    torch.save({
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'mcmc_prior_provenance': 'oof_2fold_recording_prefix_grouped_cv',
        'animal_identity_status': 'unverified',
        'X_seqs': features, 'Y_seqs': targets, 'labels': np.zeros(3, dtype=int),
        'lengths': [4] * 3, 'anchor_frames': [1] * 3,
        'mcmc_priors': np.full((3, 4), 0.25, dtype=np.float32),
        'session_ids': sessions, 'trial_ids': np.array(source_ids, dtype=np.int64),
        'stimulus_conditions': ['visual_only'] * 3,
    }, path)

    class PhysicalModel(torch.nn.Module):
        dt_ms = 4.0
        analysis_checkpoint_lineage = _diagnostic_lineage(path)

        def forward(self, x, lengths, override_gates=None):
            return x[:, :, 0]

    monkeypatch.setattr(lesion, 'load_model_from_checkpoint', lambda *_args: PhysicalModel())
    monkeypatch.setattr(plt, 'savefig', lambda *_args, **_kwargs: None)
    csv_path = tmp_path / 'lesion.csv'
    lesion.main(['--checkpoint', str(tmp_path / 'unused.pth'), '--dataset', str(path),
                 '--output', str(tmp_path / 'lesion.png'), '--stats_output', str(csv_path),
                 '--batch_size', '2'])
    with csv_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['Class', 'Condition', 'Trial_ID', 'Peak_Velocity_cms',
                                     'Latency_to_Peak_ms', 'MSE']
        rows = list(reader)
    sidecar = json.loads(csv_path.with_suffix('.block_sensitivity.json').read_text())
    assert sidecar['csv_trial_id_semantics'] == 'zero_based_dataset_row_index'
    assert len(rows) == 3 * len(lesion.CONDITION_NAMES)
    raw_records = dict(zip(zip(sessions, source_ids), peaks))
    for row in rows:
        index = int(row['Trial_ID'])
        trial = sidecar['trials'][index]
        assert trial['trial_id'] == trial['row_index'] == index
        assert trial['source_trial_id'] == source_ids[index]
        assert trial['session_id'] == sessions[index]
        assert float(row['Peak_Velocity_cms']) == raw_records[(trial['session_id'], trial['source_trial_id'])]
        assert float(row['MSE']) == 0.0
    assert len({(trial['session_id'], trial['source_trial_id']) for trial in sidecar['trials']}) == 3


@pytest.mark.parametrize('field,value', [
    ('trial_ids', [101]), ('trial_ids', [101, 103.5]),
    ('trial_ids', [101, True, 103]), ('trial_ids', np.array([[101], [103], [105]])),
    ('trial_ids', None), ('session_ids', None),
    ('session_ids', ['recording_A_session_1', 42, 'recording_B_session_1']),
    ('trial_ids', [101, 101, 103]),
])
def test_phase_d_rejects_malformed_modern_trial_lineage(tmp_path, field, value):
    from scripts import simulate_lesion as lesion

    path = tmp_path / 'invalid_lineage.pt'
    payload = {
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'mcmc_prior_provenance': 'oof_2fold_animal_grouped_cv',
        'X_seqs': [np.zeros((4, 8), dtype=np.float32) for _ in range(3)],
        'Y_seqs': [np.zeros(4, dtype=np.float32) for _ in range(3)],
        'labels': [0] * 3, 'lengths': [4] * 3, 'anchor_frames': [1] * 3,
        'mcmc_priors': np.full((3, 4), 0.25, dtype=np.float32),
        'stimulus_conditions': ['visual_only'] * 3,
        'session_ids': ['recording_A_session_1', 'recording_A_session_1', 'recording_B_session_1'],
        'trial_ids': [101, 103, 101],
    }
    if value is None:
        del payload[field]
    else:
        payload[field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match='trial_ids|session_ids|source trial'):
        lesion.load_dataset(path)


@pytest.mark.parametrize('dt_ms', [4.0, 10.0])
def test_lesion_flat_target_has_unavailable_timing_but_retains_vigor_and_mse(tmp_path, dt_ms):
    from scripts import simulate_lesion as lesion

    zero = np.zeros(20, dtype=np.float32)
    tonic = np.full(20, 5.0, dtype=np.float32)
    peaked = np.zeros(20, dtype=np.float32)
    peaked[4 + 3] = -20.0
    arrays = [zero, tonic, peaked]
    for target, velocity in [(zero, 0.0), (tonic, 5.0)]:
        metrics = lesion.extract_scalar_metrics([target], [target], [0], 0,
                                                dt_ms=dt_ms, reference_frames=[4])
        assert metrics['Peak_Velocity_cms'] == velocity
        assert metrics['Mean_MSE'] == 0.0
        assert metrics['Latency_to_Peak_ms'] is None
    mixed = lesion.extract_scalar_metrics(arrays, arrays, [0] * 3, 0,
                                         dt_ms=dt_ms, reference_frames=[4] * 3)
    assert mixed['Peak_Velocity_cms'] == pytest.approx(25.0 / 3)
    assert mixed['Mean_MSE'] == 0.0
    assert mixed['Latency_to_Peak_ms'] == 3 * dt_ms
    path = tmp_path / 'lesion.csv'
    lesion.export_lesion_statistics_csv({'intact': (arrays, arrays, [0] * 3)}, path,
                                        [0], dt_ms=dt_ms, reference_frames=[4] * 3)
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert [row['Latency_to_Peak_ms'] for row in rows] == ['', '', f'{3 * dt_ms:.2f}']
    assert [float(row['Peak_Velocity_cms']) for row in rows] == [0.0, 5.0, 20.0]
    assert [float(row['MSE']) for row in rows] == [0.0] * 3


def test_phase_d_sidecar_retains_row_specific_mse_for_same_prefix(monkeypatch, tmp_path):
    """Distinct source pairs in one recording must keep distinct per-condition scores."""
    import json
    import matplotlib.pyplot as plt
    from scripts import simulate_lesion as lesion

    dataset_path = tmp_path / 'same_prefix.pt'
    torch.save({
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'mcmc_prior_provenance': 'oof_2fold_recording_prefix_grouped_cv',
        'animal_identity_status': 'unverified',
        'X_seqs': [np.full((4, 8), [value, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
                   for value in (6.0, 8.0, 10.0)],
        'Y_seqs': [np.full(4, 5.0, dtype=np.float32) for _ in range(3)],
        'labels': np.array([0, 0, 1]), 'lengths': [4] * 3, 'anchor_frames': [1] * 3,
        'mcmc_priors': np.full((3, 4), 0.25, dtype=np.float32),
        'session_ids': ['recording_A_session_1'] * 3, 'trial_ids': [101, 102, 103],
    }, dataset_path)

    class PhysicalModel(torch.nn.Module):
        dt_ms = 4.0
        analysis_checkpoint_lineage = _diagnostic_lineage(dataset_path)

        def forward(self, x, lengths, override_gates=None):
            shift = 0 if override_gates is None else (1 if override_gates['g_lif'] == 0 else 2)
            return x[:, :, 0] + shift

    monkeypatch.setattr(lesion, 'load_model_from_checkpoint', lambda *_args: PhysicalModel())
    monkeypatch.setattr(plt, 'savefig', lambda *_args, **_kwargs: None)
    csv_path = tmp_path / 'lesion.csv'
    lesion.main(['--checkpoint', str(tmp_path / 'unused.pth'), '--dataset', str(dataset_path),
                 '--output', str(tmp_path / 'unused.png'), '--stats_output', str(csv_path)])
    with csv_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['Class', 'Condition', 'Trial_ID', 'Peak_Velocity_cms',
                                     'Latency_to_Peak_ms', 'MSE']
        rows = list(reader)
    summary = json.loads(csv_path.with_suffix('.block_sensitivity.json').read_text())
    assert summary['mse_units'] == '(cm/s)^2'
    assert summary['mse_window'] == 'observed_post_reference'
    assert len(rows) == 9  # Two ESCAPE pairs plus an eligible non-ESCAPE source pair.
    expected = {'intact': (1.0, 9.0, 25.0), 'lif_lesioned': (4.0, 16.0, 36.0),
                'gru_lesioned': (9.0, 25.0, 49.0)}
    assert summary['conditions']['intact']['mean_trial_mse'] == 5.0  # ESCAPE only.
    assert {trial['recording_prefix'] for trial in summary['trials']} == {'recording_A'}
    for index, trial in enumerate(summary['trials']):
        assert (trial['row_index'], trial['session_id'], trial['source_trial_id']) == (
            index, 'recording_A_session_1', 101 + index)
        assert trial['mse_by_condition'] == {name: values[index] for name, values in expected.items()}
    for row in rows:
        index = int(row['Trial_ID'])
        name = next(name for name, display in lesion.CONDITION_NAMES.items()
                    if display == row['Condition'])
        assert (row['Class'] == 'ESCAPE') == (index < 2)
        assert row['Peak_Velocity_cms'] == '5.0000' and row['Latency_to_Peak_ms'] == ''
        assert abs(float(row['MSE']) - summary['trials'][index]['mse_by_condition'][name]) <= 5.01e-7
    swapped = [dict(row) for row in rows]
    first = next(row for row in swapped if row['Trial_ID'] == '0' and row['Condition'] == 'Intact Model')
    second = next(row for row in swapped if row['Trial_ID'] == '1' and row['Condition'] == 'Intact Model')
    first['MSE'], second['MSE'] = second['MSE'], first['MSE']
    assert any(abs(float(row['MSE']) - summary['trials'][int(row['Trial_ID'])]['mse_by_condition'][
        next(name for name, display in lesion.CONDITION_NAMES.items() if display == row['Condition'])
    ]) > 5.01e-7 for row in swapped)


@pytest.mark.parametrize('sigma_hi,weight_lo', [(0.4, 0.7), (0.15, 0.6)])
def test_jacobian_gmm_boundary_matches_weighted_posterior(
    sigma_hi: float, weight_lo: float,
) -> None:
    from sklearn.mixture import GaussianMixture
    from scripts import analyze_jacobian as jacobian

    gm = GaussianMixture(2, covariance_type='full')
    gm.weights_ = np.array([weight_lo, 1.0 - weight_lo])
    gm.means_ = np.array([[-5.0], [-4.0]])
    gm.covariances_ = np.array([[[0.15**2]], [[sigma_hi**2]]])
    gm.precisions_cholesky_ = 1.0 / np.sqrt(gm.covariances_)
    boundary = jacobian._gmm_posterior_half_boundary_log(
        -5.0, 0.15, weight_lo, -4.0, sigma_hi, 1.0 - weight_lo,
    )
    assert -5.0 < boundary < -4.0
    np.testing.assert_allclose(gm.predict_proba([[boundary]]), [[0.5, 0.5]],
                               atol=1e-12)
    assert gm.predict_proba([[boundary - 1e-4]])[0, 0] > 0.5
    assert gm.predict_proba([[boundary + 1e-4]])[0, 0] < 0.5


def test_jacobian_calibration_caller_matches_fitted_gmm() -> None:
    from sklearn.mixture import GaussianMixture
    from scripts import analyze_jacobian as jacobian

    rng = np.random.default_rng(42)
    residuals = np.exp(np.r_[rng.normal(-5.3, 0.2, 150),
                              rng.normal(-3.9, 0.4, 50)])
    threshold, diag = jacobian._calibrate_fp_threshold(residuals)
    x_log = np.log(residuals)[:, None]
    gm = GaussianMixture(2, random_state=42).fit(x_log)
    lo = int(np.argmin(gm.means_.ravel()))
    assert diag['bic_1comp'] - diag['bic_2comp'] > 200.0
    assert gm.weights_[lo] != pytest.approx(0.5)
    assert gm.covariances_[0, 0, 0] != pytest.approx(gm.covariances_[1, 0, 0])
    assert gm.predict_proba([[np.log(threshold)]])[0, lo] == pytest.approx(0.5)
    # Across the component means, the caller's threshold classifies exactly
    # like the fitted posterior (not the reversed prior/variance polynomial).
    grid = np.linspace(*sorted(gm.means_.ravel()), 2000)
    np.testing.assert_array_equal(grid < np.log(threshold),
                                   gm.predict_proba(grid[:, None])[:, lo] > 0.5)
    # Unequal variance can put high-component mass below the between-means
    # threshold too. The accepted set must use the posterior, not just x < gate.
    probes = np.r_[residuals, np.exp(grid), np.exp(-8.0)]
    np.testing.assert_array_equal(
        jacobian._posterior_keep(probes, threshold, diag),
        gm.predict_proba(np.log(probes)[:, None])[:, lo] > 0.5,
    )


@pytest.mark.parametrize(
    'seed,scale,n',
    [(42, 0.4, 128), (42, 0.5, 200)],
)
def test_jacobian_unimodal_tail_stretch_is_withheld(
    seed: int, scale: float, n: int,
) -> None:
    """Round-4 #1: BIC mixture preference alone is not evidence of two modes.

    ``exp(-6 + Exp(scale))`` is a single log-density mode, yet a second
    component stretches one tail and raises ΔBIC above 10.  The gate must
    not declare two quasi-fixed-point subpopulations here.
    """
    from sklearn.mixture import GaussianMixture
    from scripts import analyze_jacobian as jacobian

    rng = np.random.default_rng(seed)
    residuals = np.exp(-6.0 + rng.exponential(scale, n))
    # Confirm the supplied data really does pass the ΔBIC criterion, so
    # this test fails on the old behaviour and proves the fix.
    gm = GaussianMixture(2, random_state=42).fit(np.log(residuals)[:, None])
    gm1 = GaussianMixture(1, random_state=42).fit(np.log(residuals)[:, None])
    assert gm1.bic(np.log(residuals)[:, None]) - gm.bic(
        np.log(residuals)[:, None]
    ) > 10.0
    with pytest.raises(ValueError, match='FITTED DENSITY'):
        jacobian._calibrate_fp_threshold(residuals)


def test_jacobian_degenerate_component_is_withheld() -> None:
    """Round-4 #2: an unsupported point-mass component is not a subpopulation."""
    from scripts import analyze_jacobian as jacobian

    rng = np.random.default_rng(42)
    unsupported = np.r_[np.exp(rng.normal(-5.0, 0.2, 128)), np.exp(-12.0)]
    point_mass = np.exp(np.r_[[-10.0], np.linspace(-3.01, -2.99, 19)])
    for candidate in (unsupported, point_mass):
        with pytest.raises(ValueError, match='DEGENERATE'):
            jacobian._calibrate_fp_threshold(candidate)


def test_jacobian_density_mode_count_matches_structure() -> None:
    """Round-4: the density-mode counter is a real structure test."""
    from scripts import analyze_jacobian as jacobian

    # Two well-separated, equally weighted components -> two interior modes.
    assert jacobian._fitted_density_mode_count(-5.0, 0.2, 0.5, -3.0, 0.2, 0.5) == 2
    # Narrow, widely separated components are resolved by scale-aware grid.
    assert jacobian._fitted_density_mode_count(-100.0, 0.01, 0.5, 0.0, 0.01, 0.5) == 2
    # Separation 765 counterexample cases
    assert jacobian._fitted_density_mode_count(0.0, 0.1, 0.5, 765.0, 0.1, 0.5) == 2
    assert jacobian._fitted_density_mode_count(0.0, 0.01, 0.5, 765.0, 0.01, 0.5) == 2
    # Concrete narrow separated mixture parameters:
    assert (
        jacobian._fitted_density_mode_count(
            0.6745945468, 0.1302761059, 0.5, 20.3873394026, 0.1693216743, 0.5
        )
        == 2
    )
    assert (
        jacobian._fitted_density_mode_count(
            np.float32(0.6745945468),
            np.float32(0.1302761059),
            np.float32(0.5),
            np.float32(20.3873394026),
            np.float32(0.1693216743),
            np.float32(0.5),
        )
        == 2
    )
    # Coincident components fuse into a single mode regardless of weights.
    assert jacobian._fitted_density_mode_count(-5.0, 0.3, 0.999, -5.0, 0.3, 0.001) == 1
    assert jacobian._fitted_density_mode_count(0.0, 0.1, 0.5, 0.0, 1.0, 0.5) == 1
    with pytest.raises(ValueError):
        jacobian._fitted_density_mode_count(np.nan, 0.2, 0.5, -3.0, 0.2, 0.5)


@pytest.mark.parametrize(
    'loc_hi,n_hi',
    [(-1.0, 50), (0.0, 100)],
)
def test_jacobian_widely_separated_supported_mixture_is_accepted(
    loc_hi: float, n_hi: int,
) -> None:
    """Supported widely separated synthetic residuals pass GMM mode detection."""
    from scripts import analyze_jacobian as jacobian

    rng = np.random.default_rng(42)
    n_lo = 150 if n_hi == 50 else 100
    residuals = np.exp(
        np.r_[rng.normal(-5.0, 0.2, n_lo), rng.normal(loc_hi, 0.2, n_hi)]
    )
    threshold, diag = jacobian._calibrate_fp_threshold(residuals)
    assert diag['fitted_density_modes'] == 2
    assert diag['low_component_support'] >= jacobian.FP_MIN_COMPONENT_SUPPORT
    assert diag['high_component_support'] >= jacobian.FP_MIN_COMPONENT_SUPPORT
    assert np.isfinite(threshold) and threshold > 0.0


def test_jacobian_supported_bimodal_mixture_is_accepted() -> None:
    """Round-4 positive control: a genuinely supported bimodal mixture passes."""
    from scripts import analyze_jacobian as jacobian

    rng = np.random.default_rng(42)
    residuals = np.exp(np.r_[rng.normal(-5.3, 0.2, 150), rng.normal(-3.9, 0.4, 50)])
    threshold, diag = jacobian._calibrate_fp_threshold(residuals)
    assert diag['fitted_density_modes'] >= 2
    assert diag['low_component_support'] >= jacobian.FP_MIN_COMPONENT_SUPPORT
    assert diag['high_component_support'] >= jacobian.FP_MIN_COMPONENT_SUPPORT
    assert np.isfinite(threshold) and threshold > 0.0


class _JacobianProbeModel(torch.nn.Module):
    # r6 R8: declare the routed gru_hidden IS the raw recurrent coordinate.
    gru_hidden_is_raw = True

    def __init__(self) -> None:
        super().__init__()
        self.dt_ms, self.sensory_dim, self.hidden_dim = 4.0, 4, 1
        self.sensory_encoder = lambda x: x[:, :, 2:3]

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, *, return_internals: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert x.shape == (len(lengths), x.shape[1], 8)
        assert lengths.shape == (x.shape[0],)
        return torch.zeros(x.shape[:2]), {'gru_hidden': x[:, :, 2:3]}


def _jacobian_probe_loader(
    *, wind: bool = False, n_frames: int = 601,
    anchor: int = 300, max_seq_len: int | None = None,
) -> torch.utils.data.DataLoader:
    from nsmor.nsmor_dataloader import NSMoRDataset, collate_variable_length

    x = np.zeros((n_frames, 8), dtype=np.float32)
    x[:, 0] = 2.0  # Static visual baseline does not establish visual onset.
    x[:, 2] = np.arange(n_frames)
    if wind:
        x[anchor:, 1] = 1.0
    dataset = NSMoRDataset(
        [(x, np.zeros(n_frames, dtype=np.float32), 1)],
        np.full((1, 4), 0.25, dtype=np.float32), anchor_frames=[anchor],
        max_seq_len=max_seq_len, pre_anchor_frames=300,
        source_indices=[17],
    )
    dataset.trial_ids = [101]
    dataset.session_ids = ['recording_session_1']
    return torch.utils.data.DataLoader(dataset, collate_fn=collate_variable_length)


@pytest.mark.parametrize('wind', [False, True])
@pytest.mark.parametrize('cropped', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_jacobian_both_callers_preserve_reference_and_crop(
    monkeypatch: pytest.MonkeyPatch, wind: bool, cropped: bool, legacy: bool,
) -> None:
    from scripts import analyze_jacobian as jacobian

    anchor = 600 if cropped else 300
    loader = _jacobian_probe_loader(wind=wind, n_frames=1201 if cropped else 601,
                                    anchor=anchor, max_seq_len=601 if cropped else None)
    monkeypatch.setattr(jacobian, '_find_slow_point',
                        lambda h, centre, _length: (centre, h[centre]))
    monkeypatch.setattr(jacobian, '_fixed_point_residual', lambda *_args: 0.01)
    monkeypatch.setattr(jacobian, '_calibrate_fp_threshold',
                        lambda *_args, **_kwargs: (0.02, {}))
    model = _JacobianProbeModel()
    references = [] if legacy and wind else [anchor]
    states = jacobian.extract_gru_states_at_epochs(
        model, loader, torch.device('cpu'), references,
        adapter=SimpleNamespace(),
    )
    offset = anchor - 300
    for name, centre in zip(jacobian.EPOCH_DEFINITIONS, (50, 300, 550)):
        assert states[name][0].item() == centre + offset
    seen = []

    def full_jacobian(x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, dict]:
        assert torch.is_grad_enabled()
        assert x.shape == (1, 1, 8) and lengths.shape == (1,)
        seen.append(int(x[0, 0, 2]))
        return torch.tensor([[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), {}

    result = jacobian.compute_full_system_eigenvalues(
        model, SimpleNamespace(compute_full_system_jacobian=full_jacobian),
        loader, torch.device('cpu'), references,
    )
    assert seen == [50 + offset, 300 + offset, 550 + offset]
    for values in result.values():
        np.testing.assert_allclose(values, [[5.0]])
        assert not np.iscomplexobj(values)


@pytest.mark.parametrize('reference', [None, -1, 601, float('nan'), True])
@pytest.mark.parametrize('wind', [False, True])
def test_jacobian_unavailable_reference_fails_closed(
    monkeypatch: pytest.MonkeyPatch, reference: object, wind: bool,
) -> None:
    from scripts import analyze_jacobian as jacobian

    monkeypatch.setattr(jacobian, '_fixed_point_residual', lambda *_args: 0.01)
    monkeypatch.setattr(jacobian, '_calibrate_fp_threshold',
                        lambda *_args, **_kwargs: (0.02, {}))
    loader = _jacobian_probe_loader(wind=wind)
    with pytest.raises(ValueError, match='reference.*unavailable|unavailable.*reference'):
        jacobian.extract_gru_states_at_epochs(
            _JacobianProbeModel(), loader, torch.device('cpu'), [reference],
            adapter=SimpleNamespace(),
        )
    with pytest.raises(ValueError, match='reference.*unavailable|unavailable.*reference'):
        jacobian.compute_full_system_eigenvalues(
            _JacobianProbeModel(), SimpleNamespace(), loader, torch.device('cpu'),
            [reference],
        )


def test_jacobian_onset_detection_does_not_invent_references() -> None:
    from scripts import analyze_jacobian as jacobian

    silent = np.zeros((601, 8))
    static = silent.copy()
    static[:, 0] = 2.0
    visual = silent.copy()
    visual[300:, 0] = 2.0
    wind = static.copy()
    wind[300:, 1] = 1.0
    assert jacobian.detect_stimulus_onset_frames([silent, static, visual, wind]) == [
        None, None, 300, 300,
    ]


def test_jacobian_full_system_all_fail_is_unavailable() -> None:
    from scripts import analyze_jacobian as jacobian

    def fail(*_args: object) -> None:
        raise RuntimeError('synthetic SVD failure')

    with pytest.raises(ValueError, match='unavailable'):
        jacobian.compute_full_system_eigenvalues(
            _JacobianProbeModel(), SimpleNamespace(compute_full_system_jacobian=fail),
            _jacobian_probe_loader(), torch.device('cpu'), [300],
        )


def test_jacobian_capped_sampling_is_seeded_and_auditable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import analyze_jacobian as jacobian

    class Adapter:
        def test_attractor_convergence(self, *_args: object) -> tuple[bool, float, bool]:
            return False, 0.1, False

        def compute_jacobian_batch(
            self, h: torch.Tensor, x: torch.Tensor,
        ) -> torch.Tensor:
            assert h.shape == x.shape == (len(h), 1)
            return h.detach().unsqueeze(-1)

    monkeypatch.setattr(jacobian, '_fixed_point_residual', lambda *_args: 0.01)
    monkeypatch.setattr(jacobian, '_calibrate_fp_threshold',
                        lambda *_args, **_kwargs: (0.02, {}))
    h = torch.arange(30, dtype=torch.float32).reshape(-1, 1)
    identities = [{'candidate_id': f'early:{i}:300', 'source_row_index': 10 + i,
                   'source_trial_id': 101 + i, 'session_id': 'recording_session_1'}
                  for i in range(30)]
    data = {'early': (h, h), 'early__gate_diag': {
        'n_candidates': 40, 'n_accepted': 30, 'n_rejected': 10,
        'accepted_candidates': identities,
    }}
    first = jacobian.compute_eigenvalues_at_epochs(
        Adapter(), data, torch.device('cpu'), max_states_per_epoch=5,
    )
    torch.rand(100)  # Unrelated callers must not affect the selection.
    second = jacobian.compute_eigenvalues_at_epochs(
        Adapter(), data, torch.device('cpu'), max_states_per_epoch=5,
    )
    np.testing.assert_array_equal(first[0]['early'], second[0]['early'])
    assert first[1] == second[1]
    stats = first[1]['early']
    assert stats['sampling_seed'] == 42
    assert (stats['n_candidates'], stats['n_accepted'], stats['n_rejected'],
            stats['n_sampled']) == (40, 30, 10, 5)
    for index, identity, value in zip(stats['selected_state_indices'],
                                     stats['selected_candidates'], first[0]['early']):
        assert identity == identities[index]
        assert value[0].real == index


def test_jacobian_full_system_runner_uses_singular_plot_and_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from scripts import analyze_jacobian as jacobian

    loader = _jacobian_probe_loader()
    loader.dataset.analysis_reference_frames = [300]
    loader.dataset.analysis_reference_rules = ['looming_collision']
    monkeypatch.setattr(jacobian, 'load_model_from_checkpoint',
                        lambda *_args: _JacobianProbeModel())
    monkeypatch.setattr(jacobian, 'load_dataset', lambda *_args, **_kwargs:
                        (loader, np.array([1]), [601], [loader.dataset.sequences[0][0]]))
    monkeypatch.setattr(jacobian, 'extract_gru_states_at_epochs',
                        lambda **_kwargs: pytest.fail('Input sensitivity uses no GRU FP gate'))
    monkeypatch.setattr(jacobian, 'create_jacobian_adapter', lambda *_args, **_kwargs:
                        SimpleNamespace(compute_full_system_jacobian=
                                        lambda *_args: (torch.ones(1, 8), {})))
    monkeypatch.setattr(jacobian, 'plot_eigenvalue_spectrum',
                        lambda **_kwargs: pytest.fail('Singular values are not eigenvalues'))
    figures = []
    monkeypatch.setattr(plt, 'savefig', lambda *_args, **_kwargs: figures.append(plt.gcf()))
    output = tmp_path / 'sensitivity.png'
    jacobian.run_jacobian_analysis(tmp_path / 'unused.pth', tmp_path / 'unused.pt',
                                   output, full_system=True)
    assert figures
    figure = figures[0]
    assert 'singular' in figure._suptitle.get_text().lower()
    assert 'surrogate' in figure._suptitle.get_text().lower()
    for axis in figure.axes:
        assert 'singular' in axis.get_xlabel().lower()
        assert 'Re(' not in axis.get_xlabel()
    summary = json.loads(output.with_suffix('.json').read_text())
    assert summary['status'] == 'ok'
    assert summary['spectral_quantity'] == 'surrogate_input_singular_values'
    assert summary['stability_interpretation'] is False
    assert 'pct_near_unit_circle' not in summary['spectral_statistics']['early']

    monkeypatch.setattr(jacobian, 'compute_full_system_eigenvalues',
                        lambda **_kwargs: {})
    with pytest.raises(ValueError, match='unavailable'):
        jacobian.run_jacobian_analysis(tmp_path / 'unused.pth', tmp_path / 'unused.pt',
                                       output, full_system=True)
    summary = json.loads(output.with_suffix('.json').read_text())
    assert summary['status'] == 'unavailable'
    assert summary['spectral_statistics'] == {}


def test_jacobian_actual_adapter_sampling_and_full_system_svd() -> None:
    from nsmor.analysis.dynamics import FixedPointAdapter
    from scripts import analyze_jacobian as jacobian

    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = NSMoRCore(hidden_dim=4, dt_ms=4.0, sensory_noise_std=0.0,
                          dropout=0.0).cpu().eval()
        h = torch.randn(24, 4)
        x = torch.randn(24, 4)
    adapter = FixedPointAdapter(model, device=torch.device('cpu'))
    data = {'early': (h, x)}
    first = jacobian.compute_eigenvalues_at_epochs(
        adapter, data, torch.device('cpu'), max_states_per_epoch=5,
        sampling_seed=19,
    )
    second = jacobian.compute_eigenvalues_at_epochs(
        adapter, data, torch.device('cpu'), max_states_per_epoch=5,
        sampling_seed=19,
    )
    np.testing.assert_array_equal(first[0]['early'], second[0]['early'])
    indices = first[1]['early']['selected_state_indices']
    expected = torch.linalg.eigvals(adapter.compute_jacobian_batch(h[indices], x[indices]))
    np.testing.assert_allclose(first[0]['early'], expected.numpy())
    assert 'n_candidates' not in first[1]['early']  # No imaginary upstream gate.
    assert 'selected_candidates' not in first[1]['early']
    other = jacobian.compute_eigenvalues_at_epochs(
        adapter, data, torch.device('cpu'), max_states_per_epoch=5,
        sampling_seed=20,
    )
    assert indices != other[1]['early']['selected_state_indices']

    loader = _jacobian_probe_loader()
    selected = {}
    with torch.no_grad():
        result = jacobian.compute_full_system_eigenvalues(
            model, adapter, loader, torch.device('cpu'), [300],
            max_states_per_epoch=1, selection_metadata=selected,
        )
    for name, frame in zip(jacobian.EPOCH_DEFINITIONS, (50, 300, 550)):
        batch = loader.dataset[0][0][frame:frame + 1].unsqueeze(0)
        with torch.enable_grad():
            matrix, _ = adapter.compute_full_system_jacobian(batch, torch.tensor([1]))
        assert matrix.shape == (4, 8)
        np.testing.assert_allclose(
            result[name][0], np.linalg.svd(matrix.detach().numpy(), compute_uv=False),
            rtol=1e-5, atol=1e-7,
        )
        assert selected[name]['n_candidates'] == selected[name]['n_sampled'] == 1
        assert selected[name]['n_accepted'] == 1 and selected[name]['n_rejected'] == 0


def test_jacobian_full_system_cap_records_exact_sampled_candidates() -> None:
    from scripts import analyze_jacobian as jacobian
    from nsmor.nsmor_dataloader import NSMoRDataset, collate_variable_length

    source = _jacobian_probe_loader().dataset.sequences[0][0]
    sequences = []
    for i in range(12):
        x = source.copy()
        x[:, 2] = i + 1
        x[:, 4:8] = 0.0
        sequences.append((x, np.zeros(601), 1))
    dataset = NSMoRDataset(
        sequences, np.full((12, 4), 0.25), max_seq_len=None,
        anchor_frames=[300] * 12, source_indices=list(range(100, 112)),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=collate_variable_length,
    )
    seen = []

    def jacobian_matrix(x: torch.Tensor, _lengths: torch.Tensor) -> tuple[torch.Tensor, dict]:
        seen.append(int(x[0, 0, 2]))
        return x[0, :, :], {}

    metadata = {}
    result = jacobian.compute_full_system_eigenvalues(
        _JacobianProbeModel(), SimpleNamespace(compute_full_system_jacobian=jacobian_matrix),
        loader, torch.device('cpu'), [300] * 12, max_states_per_epoch=3,
        sampling_seed=17, selection_metadata=metadata,
    )
    assert len(seen) == 9
    expected = torch.randperm(12, generator=torch.Generator().manual_seed(17))[:3].tolist()
    for name in jacobian.EPOCH_DEFINITIONS:
        stats = metadata[name]
        assert stats['selected_state_indices'] == expected
        assert (stats['n_candidates'], stats['n_sampled'], stats['n_accepted'],
                stats['n_rejected'], stats['n_unsampled']) == (12, 3, 3, 0, 9)
        assert [i['source_row_index'] for i in stats['selected_candidates']] == [100 + i for i in expected]
        assert result[name].shape == (3, 1)
        assert all('source_trial_id' not in item and 'session_id' not in item
                   for item in stats['selected_candidates'])


def test_jacobian_loader_maps_wind_onset_through_producer_crop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A crop anchor is not necessarily the observed wind onset."""
    from scripts import analyze_jacobian as jacobian

    features = [np.zeros((1201, 8), dtype=np.float32) for _ in range(3)]
    features[0][:, 0] = 2.0
    features[0][600:, 1] = 1.0
    features[0][:, 2] = np.arange(1201)
    features[1][:, 0] = 2.0  # Visual collision supplied; onset is not observed.
    features[1][:, 2] = np.arange(1201)
    path = tmp_path / 'jacobian_crop.pt'
    torch.save({
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'mcmc_prior_provenance': 'oof_2fold_recording_prefix_grouped_cv',
        'animal_identity_status': 'unverified',
        'X_seqs': features, 'Y_seqs': [np.zeros(1201) for _ in features],
        'labels': np.ones(3, dtype=np.int64), 'lengths': [1201] * 3,
        'anchor_frames': [800, 600, 600],
        'session_ids': ['recording_session_1'] * 3,
        'mcmc_priors': np.full((3, 4), 0.25),
    }, path)
    model = SimpleNamespace(dt_ms=4., analysis_checkpoint_lineage=_diagnostic_lineage(path))
    loader, *_ = jacobian.load_dataset(
        path, max_seq_len=601, pre_anchor_frames=300, checkpoint_model=model,
    )
    dataset = loader.dataset
    assert dataset.analysis_reference_frames == [100, 300, None]
    assert dataset.analysis_reference_rules == [
        'wind_onset', 'looming_collision', 'unavailable',
    ]
    assert torch.nonzero(dataset[0][0][:, 1] > 0.5)[0].item() == 100

    # Only wind and visual trials enter either caller; the silent row is not PREWALK.
    dataset.sequences[2] = (*dataset.sequences[2][:2], 3)
    monkeypatch.setattr(jacobian, '_find_slow_point',
                        lambda h, centre, _length: (centre, h[centre]))
    monkeypatch.setattr(jacobian, '_fixed_point_residual', lambda *_args: 0.01)
    monkeypatch.setattr(jacobian, '_calibrate_fp_threshold',
                        lambda *_args, **_kwargs: (0.02, {}))
    probe = _JacobianProbeModel()
    states = jacobian.extract_gru_states_at_epochs(
        probe, loader, torch.device('cpu'), [800, 600, None],
        adapter=SimpleNamespace(),
    )
    expected = {'early': [350], 'transient': [600, 600], 'sustained': [850, 850]}
    for name, frames in expected.items():
        assert states[name][0][:, 0].tolist() == frames
    selected = {}
    jacobian.compute_full_system_eigenvalues(
        probe, SimpleNamespace(compute_full_system_jacobian=lambda *_args:
                               (torch.ones(1, 8), {})),
        loader, torch.device('cpu'), [800, 600, None], selection_metadata=selected,
    )
    for name, frames in {'early': [50], 'transient': [100, 300],
                         'sustained': [350, 550]}.items():
        assert [item['frame'] for item in
                selected[name]['selected_candidates']] == frames


@pytest.mark.parametrize('caller', ['gru', 'full_system'])
@pytest.mark.parametrize('reference_source', ['mapped', 'unavailable', 'supplied',
                                            'legacy'])
@pytest.mark.parametrize('n_frames,first,second', [(1601, 400, 1000),
                                                (1201, 100, 600)])
def test_jacobian_original_wind_onset_outside_crop_fails_closed(
    tmp_path: Path, caller: str, reference_source: str,
    n_frames: int, first: int, second: int,
) -> None:
    """A later cropped pulse cannot replace the original analysis time origin."""
    from scripts import analyze_jacobian as jacobian

    x = np.zeros((n_frames, 8), dtype=np.float32)
    x[first:first + 100, 1] = 1.0
    x[second:second + 100, 1] = 1.0
    path = tmp_path / 'two_wind_pulses.pt'
    torch.save({
        'pipeline_semantics_version': PIPELINE_SEMANTICS_VERSION,
        'mcmc_prior_provenance': 'oof_2fold_recording_prefix_grouped_cv',
        'animal_identity_status': 'unverified',
        'X_seqs': [x], 'Y_seqs': [np.zeros(n_frames, dtype=np.float32)],
        'labels': np.ones(1, dtype=np.int64), 'lengths': [n_frames],
        'anchor_frames': [second], 'session_ids': ['recording_session_1'],
        'mcmc_priors': np.full((1, 4), 0.25),
    }, path)
    loader, *_ = jacobian.load_dataset(
        path, max_seq_len=601, pre_anchor_frames=300,
        checkpoint_model=SimpleNamespace(
            dt_ms=4., analysis_checkpoint_lineage=_diagnostic_lineage(path),
        ),
    )
    assert loader.dataset.anchor_frames == [second]
    assert loader.dataset.analysis_reference_frames == [first - (second - 300)]
    assert loader.dataset.analysis_reference_rules == ['wind_onset']
    assert torch.nonzero(loader.dataset[0][0][:, 1] > 0.5)[0].item() == 300
    references = [second]  # The mapped sidecar takes precedence over crop anchor.
    if reference_source == 'unavailable':
        loader.dataset.analysis_reference_frames = [None]
    elif reference_source in ('supplied', 'legacy'):
        del loader.dataset.analysis_reference_frames
        references = [first] if reference_source == 'supplied' else []
    extract = (jacobian.extract_gru_states_at_epochs if caller == 'gru'
               else jacobian.compute_full_system_eigenvalues)
    with pytest.raises(ValueError, match='analysis reference unavailable'):
        extract(
            model=_JacobianProbeModel(), adapter=SimpleNamespace(),
            dataloader=loader, device=torch.device('cpu'), onset_frames=references,
        )


def test_jacobian_actual_residual_gate_accepts_posterior_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Exercise extraction + actual GRU residuals + fitted posterior end to end."""
    from sklearn.mixture import GaussianMixture
    from nsmor.nsmor_dataloader import NSMoRDataset, collate_variable_length
    from scripts import analyze_jacobian as jacobian

    class ResidualModel(_JacobianProbeModel):
        def __init__(self) -> None:
            super().__init__()
            self.sensory_encoder = lambda x: torch.zeros_like(x[:, :, 2:3])

    rng = np.random.default_rng(42)
    residuals = np.exp(np.r_[rng.normal(-5.3, 0.2, 150),
                              rng.normal(-3.9, 0.4, 50), -8.0])
    features = []
    for residual in residuals:
        x = np.zeros((601, 8), dtype=np.float32)
        x[:, 0] = 2.0
        x[300:, 1] = 1.0
        x[:, 2] = 2 * residual
        features.append(x)
    dataset = NSMoRDataset(
        [(x, np.zeros(601), 1) for x in features],
        np.full((len(features), 4), 0.25), max_seq_len=None,
        anchor_frames=[300] * len(features),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, collate_fn=collate_variable_length,
    )
    gru = torch.nn.GRU(1, 1, batch_first=True)
    with torch.no_grad():
        for parameter in gru.parameters():
            parameter.zero_()  # Exact one-step residual = 0.5 * |h|.
    result = jacobian.extract_gru_states_at_epochs(
        ResidualModel(), loader, torch.device('cpu'), [300] * len(features),
        adapter=SimpleNamespace(_gru_cell=gru),
    )
    gm = GaussianMixture(2, random_state=42).fit(
        np.log(np.asarray(residuals, dtype=np.float32).astype(float))[:, None],
    )
    low = int(np.argmin(gm.means_.ravel()))
    expected = np.flatnonzero(gm.predict_proba(np.log(residuals)[:, None])[:, low] > 0.5)
    assert len(expected) < len(residuals)
    for name in jacobian.EPOCH_DEFINITIONS:
        diag = result[name + '__gate_diag']
        accepted = [item['source_row_index'] for item in diag['accepted_candidates']]
        np.testing.assert_array_equal(accepted, expected)
        assert diag['n_candidates'] == len(features)
        assert diag['n_accepted'] + diag['n_rejected'] == len(features)
        np.testing.assert_allclose(result[name][0][:, 0], residuals[expected] * 2)

    adapter = SimpleNamespace(
        _gru_cell=gru,
        test_attractor_convergence=lambda *_args: (False, 0.1, False),
        compute_jacobian_batch=lambda h, _x: torch.full((len(h), 1, 1), 0.5),
    )
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(jacobian, 'load_model_from_checkpoint',
                        lambda *_args: ResidualModel())
    monkeypatch.setattr(jacobian, 'load_dataset', lambda *_args, **_kwargs: (
        loader, np.ones(len(features), dtype=int), [601] * len(features), features,
    ))
    monkeypatch.setattr(jacobian, 'create_jacobian_adapter',
                        lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(jacobian, 'plot_eigenvalue_spectrum',
                        lambda **_kwargs: None)
    output = tmp_path / 'posterior_gate.png'
    jacobian.run_jacobian_analysis(tmp_path / 'unused.pth', tmp_path / 'unused.pt',
                                   output, max_states_per_epoch=1)
    summary = json.loads(output.with_suffix('.json').read_text())
    for name in jacobian.EPOCH_DEFINITIONS:
        diag = summary['fp_gate_calibration'][name + '__gate_diag']
        assert diag['n_accepted'] == 151
        assert (diag['n_accept_at_75pct'], diag['n_accept_at_125pct']) == (148, 154)
        assert diag['acceptance_rule'] == (
            'low_component_posterior > 0.5 and residual < residual_cap'
        )
        assert diag['residual_cap'] == 0.3
        assert diag['threshold_sensitivity_status'] == (
            'scalar_cutoff_diagnostic_not_posterior_gate_robustness'
        )
        assert diag['threshold_sensitivity_interpretation'] == (
            'n_accept_at_75pct/n_accept_at_125pct count residual < scaled '
            'fp_threshold only; they do not perturb the fitted posterior gate.'
        )
        assert [jacobian._posterior_keep(residuals, diag['fp_threshold'] * scale,
                                         diag).sum()
                for scale in (0.75, 1.0, 1.25)] == [151, 151, 151]


@pytest.mark.parametrize("module_name,entry", [
    ("analyze_dynamics", "load_model_from_checkpoint"),
    ("analyze_gating", "load_model_and_dataset"),
    ("analyze_integration", "load_model_from_checkpoint"),
    ("analyze_jacobian", "load_model_from_checkpoint"),
    ("simulate_autoregressive", "load_model_from_checkpoint"),
    ("simulate_lesion", "load_model_from_checkpoint"),
    ("simulate_psychophysics", "load_checkpoint"),
])
def test_all_analysis_loaders_reject_serialized_boolean_clock(
    normalized_checkpoint: tuple, tmp_path: Path,
    module_name: str, entry: str,
) -> None:
    """Every production wrapper must reject before canonical reconstruction."""
    import importlib
    from unittest import mock

    path, payload = normalized_checkpoint
    payload["config"]["model"]["dt_ms"] = True
    torch.save(payload, path)
    with mock.patch.object(prediction_units, "_canonical_load_model") as construct:
        module = importlib.import_module("scripts." + module_name)
        argument = (
            tmp_path / "absent-dataset.pt" if entry == "load_model_and_dataset"
            else torch.device("cpu")
        )
        with pytest.raises(ValueError, match="dt_ms.*finite positive"):
            getattr(module, entry)(path, argument)
        construct.assert_not_called()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("change,accepted", [

    ({}, True),
    ({"dataset_source_sha256": None}, False),
    ({"dataset_source_sha256": "a" * 63}, False),
    ({"dataset_source_sha256": "G" * 64}, False),
    ({"dataset_source_sha256": 7}, False),
    ({"dataset_source_binding": "legacy_resume_unbound"}, False),
])
def test_analysis_model_requires_modern_dataset_binding(normalized_checkpoint, nested, change, accepted):
    path, payload = normalized_checkpoint
    payload.update(is_nested_cv=nested, animal_identity_status="unverified",
                   mcmc_prior_provenance=("nested_outer_seed9_inner_2fold_recording_prefix_grouped"
                                          if nested else "oof_2fold_recording_prefix_grouped_cv"),
                   dataset_source_sha256="a" * 64)
    if nested:
        payload["nested_prior_fingerprint"] = "a" * 64
    for key, value in change.items():
        if value is None:
            payload.pop(key)
        else:
            payload[key] = value
    torch.save(payload, path)
    if not accepted:
        with pytest.raises(ValueError, match="dataset_source_sha256|dataset_source_binding"):
            prediction_units.load_model_from_checkpoint(path, torch.device("cpu"))
        return
    model = prediction_units.load_model_from_checkpoint(path, torch.device("cpu"))
    assert model.analysis_checkpoint_lineage["dataset_source_sha256"] == "a" * 64
    assert model.analysis_animal_identity_status == "unverified"


def test_analysis_model_rejects_conflicting_nested_source_digest(normalized_checkpoint):
    path, payload = normalized_checkpoint
    payload.update(is_nested_cv=True, animal_identity_status="unverified",
                   mcmc_prior_provenance="nested_outer_seed9_inner_2fold_recording_prefix_grouped",
                   dataset_source_sha256="a" * 64, nested_prior_fingerprint="b" * 64)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="dataset_source_sha256.*nested_prior_fingerprint"):
        prediction_units.load_model_from_checkpoint(path, torch.device("cpu"))


# =========================================================================
# T2: Exact GRU-input reconstruction must follow the FrontendEncoder path
# =========================================================================

def _capture_gru_input(monkeypatch, model):
    """Record the exact (B, T, H) tensor the GRU cell receives in forward()."""
    captured = {}
    original = model.gru_unit.forward

    def spy(x, lengths, h0=None, return_hidden=False):
        captured["input"] = x.detach().clone()
        return original(x, lengths, h0=h0, return_hidden=return_hidden)

    monkeypatch.setattr(model.gru_unit, "forward", spy)
    return captured


def test_jacobian_reconstruction_matches_dendritic_frontend(monkeypatch) -> None:
    """Real NSMoRCore: reconstruction must route through the dendritic frontend.

    With dendritic filtering enabled the GRU input is the *filtered*
    frontend output, so reconstructing via ``sensory_encoder`` alone
    yields a different vector than the GRU actually saw.
    """
    from scripts import analyze_jacobian as jacobian

    torch.manual_seed(0)
    model = NSMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=8, num_gru_layers=1,
        dropout=0.0, lif_dendritic_tau=20.0, dt_ms=4.0,
    )
    model.eval()
    assert model.frontend._dendritic_enabled

    X = torch.randn(2, 12, 8)
    lengths = torch.tensor([12, 9], dtype=torch.int64)

    captured = _capture_gru_input(monkeypatch, model)
    with torch.no_grad():
        model(X, lengths, return_internals=True)
    assert "input" in captured, "GRU spy never fired"

    reconstructed = jacobian._reconstruct_gru_input(model, X, lengths)
    assert reconstructed.shape == (2, 12, 8)
    torch.testing.assert_close(reconstructed, captured["input"])

    # The raw encoder path (no dendritic filtering) must NOT reproduce the
    # GRU input — otherwise this regression would pass on the broken code.
    raw = model.sensory_encoder(X[:, :, :4])
    assert not torch.allclose(raw, captured["input"])


def test_jacobian_reconstruction_falls_back_for_frontendless_probe() -> None:
    """Legacy probe models without .frontend still reconstruct via encoder."""
    from scripts import analyze_jacobian as jacobian

    class ProbeModel:
        def __init__(self) -> None:
            self.sensory_dim = 4
            self.sensory_encoder = lambda x: x[:, :, 2:4]

    model = ProbeModel()
    X = torch.randn(2, 6, 8)
    lengths = torch.tensor([6, 4], dtype=torch.int64)
    out = jacobian._reconstruct_gru_input(model, X, lengths)
    assert out.shape == (2, 6, 2)
    torch.testing.assert_close(out, X[:, :, 2:4])
