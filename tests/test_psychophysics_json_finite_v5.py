"""Synthetic TTC=0 positive and absent Phase G output, with default clean level.

These are deferred runtime regressions while formal QC training holds memory.
They do not assert that the observed corpus contains a TTC=0 condition.
"""

import json
import math
import sys

import pytest
import torch


def assert_finite_numbers(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            assert_finite_numbers(item)
    else:
        raise AssertionError(f"Unsupported JSON value {type(value).__name__}")


@pytest.mark.parametrize("dt_ms", [4.0, 10.0])
@pytest.mark.parametrize("noise_levels", [None, [0.0], [0.0, 15.0], [5.0, 30.0]])
def test_synthetic_declared_ttc0_noise_sweep_is_finite_json(
        monkeypatch, tmp_path, dt_ms, noise_levels):
    from scripts import simulate_psychophysics as phase_g

    class PredictableModel:
        def __init__(self):
            self.dt_ms = dt_ms

        def __call__(self, X, lengths, return_internals=False):
            B, T, F = X.shape
            assert F == 8 and lengths.shape == (B,)
            y = torch.zeros((B, T), dtype=X.dtype, device=X.device)
            y[:, 4] = 25.0  # A stable, finite +2-frame peak for all noise levels.
            gates = torch.full((B, T, 2), 0.5, dtype=X.dtype, device=X.device)
            return y, {"routing_gates": gates}

    X = torch.zeros(2, 12, 8)
    X[:, :, 0] = 20.0
    X[:, :, 1] = 1.0
    X[:, :, 4:] = 0.25
    lengths = torch.full((2,), 12, dtype=torch.int64)
    meta = phase_g.TrialMeta(
        val_indices=[0, 1], target_ttc_ms=[0.0, 0.0],
        stimulus_conditions=["multisensory", "multisensory"],
        anchor_frames=[2, 2], stim_onset_frame=2, pre_anchor_frames=2,
    )
    data = phase_g.ValidationData(X, torch.zeros(2, 12), lengths, meta)
    monkeypatch.setattr(phase_g, "load_checkpoint", lambda *_args: PredictableModel())
    monkeypatch.setattr(phase_g, "load_validation_data", lambda *_args, **_kw: data)

    out = tmp_path / "present"
    argv = ["simulate_psychophysics.py", "--dataset", "synthetic.pt",
            "--output_dir", str(out)]
    if noise_levels is not None:
        argv.extend(["--noise_levels", *map(str, noise_levels)])
    monkeypatch.setattr(sys, "argv", argv)
    phase_g.main()  # Exercise the production parser, sweep, figure and both JSON outputs.
    with (out / "psychophysics_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream, parse_constant=lambda val: (_ for _ in ()).throw(ValueError(val)))
    with (out / "bayesian_reliability.json").open(encoding="utf-8") as stream:
        status = json.load(stream, parse_constant=lambda val: (_ for _ in ()).throw(ValueError(val)))
    assert out.joinpath("bayesian_reliability.png").is_file()
    assert status["status"] == "ok" and status["n_ttc0"] == 2
    expected_levels = phase_g.NOISE_LEVELS if noise_levels is None else noise_levels
    assert summary["noise_levels"] == expected_levels
    assert set(summary["latency_stats"]) == {str(level) for level in expected_levels}
    assert set(summary["gate_post_stim_mean"]) == {str(level) for level in expected_levels}
    assert all(stats["mean"] == 2 * dt_ms for stats in summary["latency_stats"].values())
    assert out.joinpath("bayesian_reliability.png").stat().st_size > 0
    assert "σ=0" in summary["snr_clean_condition"]
    if 0.0 in expected_levels:
        assert summary["snr_db_by_sigma"]["0.0"] is None
    else:
        assert "σ=0 absent" in summary["inference"]["design"]
        assert summary["inference"]["effect_sizes_hodges_lehmann_ms"] == {}
    assert_finite_numbers(summary)
    assert_finite_numbers(status)

    # A separate synthetic sample without declared TTC0 must stay unsupported.
    data.meta.target_ttc_ms = [-225.0, -261.0]
    absent = tmp_path / "absent"
    monkeypatch.setattr(sys, "argv", ["simulate_psychophysics.py", "--dataset", "synthetic.pt",
                                           "--output_dir", str(absent)])
    phase_g.main()
    with (absent / "bayesian_reliability.json").open(encoding="utf-8") as stream:
        status = json.load(stream, parse_constant=lambda val: (_ for _ in ()).throw(ValueError(val)))
    assert status["status"] == "not_applicable" and status["n_ttc0"] == 0
    assert not (absent / "psychophysics_summary.json").exists()
    assert not (absent / "bayesian_reliability.png").exists()
    assert_finite_numbers(status)

@pytest.mark.parametrize("degenerate", [False, True])
def test_single_recording_prefix_twenty_trials_stays_descriptive(monkeypatch, tmp_path, degenerate):
    """Twenty paired trials under one prefix cannot identify independent animals."""
    from scripts import simulate_psychophysics as phase_g

    class ShiftedModel:
        dt_ms = 4.0

        def __call__(self, X, lengths, return_internals=False):
            count, frames, features = X.shape
            assert (count, frames, features) == (20, 40, 8)
            assert lengths.shape == (count,)
            noisy = bool(X[0, 0, 0].item())
            output = torch.zeros((count, frames), dtype=X.dtype, device=X.device)
            for index in range(count):
                offset = (1 if degenerate else index % 3 + 1) if noisy else 0
                output[index, 4 + offset] = 25.0
            gates = torch.full((count, frames, 2), 0.5, dtype=X.dtype, device=X.device)
            return output, {"routing_gates": gates}

    X = torch.zeros(20, 40, 8)
    X[:, :, 0] = 20.0
    X[:, :, 1] = 1.0
    X[:, :, 4:] = 0.25
    lengths = torch.full((20,), 40, dtype=torch.int64)
    meta = phase_g.TrialMeta(
        val_indices=list(range(20)), session_ids=["one_recording_prefix"] * 20,
        trial_ids=list(range(20)), target_ttc_ms=[0.0] * 20,
        stimulus_conditions=["multisensory"] * 20,
        anchor_frames=[2] * 20, stim_onset_frame=2, pre_anchor_frames=2,
    )
    data = phase_g.ValidationData(X, torch.zeros(20, 40), lengths, meta)
    monkeypatch.setattr(phase_g, "load_checkpoint", lambda *_: ShiftedModel())
    monkeypatch.setattr(phase_g, "load_validation_data", lambda *a, **k: data)

    def controlled_noise(values, lengths, sigma, seed, trial_seed_offset):
        altered = values.clone()
        altered[:, 0, 0] = sigma
        return altered

    monkeypatch.setattr(phase_g, "inject_visual_noise", controlled_noise)
    out = tmp_path / ("degenerate" if degenerate else "variable")
    monkeypatch.setattr(sys, "argv", ["simulate_psychophysics.py", "--dataset", "synthetic.pt",
                                         "--output_dir", str(out)])
    import matplotlib.pyplot as plt
    close_figure = plt.close
    figure_labels = {}

    def capture_labels(figure):
        figure_labels["latency_axis"] = figure.axes[1].get_ylabel()
        figure_labels["title"] = figure.axes[1].get_title()
        close_figure(figure)

    monkeypatch.setattr(plt, "close", capture_labels)
    phase_g.main()
    assert "descriptive" in figure_labels["latency_axis"]
    assert "fixed recorded trials" in figure_labels["title"].lower()
    with (out / "psychophysics_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream, parse_constant=lambda val: (_ for _ in ()).throw(ValueError(val)))
    assert out.joinpath("bayesian_reliability.png").is_file()
    assert summary["latency_stats"]["5.0"]["n_pairs_vs_clean"] == 20
    assert summary["latency_stats"]["5.0"]["mean"] > summary["latency_stats"]["0.0"]["mean"]
    assert summary["latency_stats"]["5.0"]["hodges_lehmann_ms"] > 0
    if degenerate:
        assert summary["latency_stats"]["5.0"]["n_degenerate_zero_variance"] == 20
    assert "test" not in summary["latency_stats"]["5.0"]
    assert "shapiro_p" not in summary["latency_stats"]["5.0"]
    inference = summary["inference"]
    assert inference["status"] == "descriptive_only"
    assert inference["independent_animal_count"] is None
    assert "unverified" in inference["population_inference_status"]
    assert "fixed recorded trial" in inference["design"]
    assert inference["p_values_uncorrected"] is None
    assert inference["p_values_holm_corrected"] is None
    assert inference["effect_sizes_hodges_lehmann_ms"]["5.0"] > 0
    assert_finite_numbers(summary)


@pytest.mark.parametrize("degenerate", [False, True])
def test_panel_d_describes_trial_spread_without_animal_ci(degenerate):
    """Panel D must show recorded responses without a trial bootstrap CI."""
    import matplotlib.pyplot as plt
    import numpy as np
    from nsmor.config import Label
    from scripts.analyze_dynamics import DynamicsBundle, plot_panel_d_pathway_dominance

    bundle = DynamicsBundle()
    escape = [0.70 if degenerate else 0.70 + 0.01 * i for i in range(10)]
    no_response = [0.30 if degenerate else 0.30 + 0.01 * i for i in range(10)]
    bundle.g_gru_trajs = [np.full(4, v) for v in escape + no_response]
    bundle.labels = [Label.ESCAPE.value] * 10 + [Label.NO_RESPONSE.value] * 10
    fig, ax = plt.subplots()
    try:
        plot_panel_d_pathway_dominance(ax, bundle)
        assert len(ax.patches) == 2
        assert [patch.get_height() for patch in ax.patches] == pytest.approx(
            [np.mean(escape), np.mean(no_response)]
        )
        bars = next(container for container in ax.containers if hasattr(container, "patches"))
        segments = bars.errorbar.lines[2][0].get_segments()
        for segment, values in zip(segments, (escape, no_response)):
            mean, sd = np.mean(values), np.std(values, ddof=1)
            assert segment[:, 1] == pytest.approx([mean - sd, mean + sd])
        labels = " ".join([ax.get_xlabel(), ax.get_ylabel()] + [text.get_text() for text in ax.texts])
        assert "descriptive_only" in labels
        assert "animal" in labels.lower()
        assert "SD" in labels
        assert "CI" not in labels
        assert "d=nan" not in labels.lower()
        if degenerate:
            assert "d undefined" in labels
        else:
            assert "d=" in labels
    finally:
        plt.close(fig)

def test_tonic_baseline_has_no_latency_but_real_peaks_keep_signed_time():
    from scripts.simulate_psychophysics import extract_latency_to_peak

    values = torch.full((3, 1000), 5.0)
    values[1, 400] = 25.0
    values[2, 600] = -25.0
    assert values.shape == (3, 1000)
    latencies = extract_latency_to_peak(
        values, torch.tensor([1000] * 3), dt_ms=4.0, anchor_frames=[500] * 3,
    )
    assert math.isnan(latencies[0])
    assert latencies[1:] == [-400.0, 400.0]


@pytest.mark.parametrize("tonic_velocity", [0.0, 5.0])
def test_ttc0_with_no_measurable_latency_is_explicitly_unavailable(monkeypatch, tmp_path, tonic_velocity):
    """A present TTC=0 group with flat model output is not an estimated latency curve."""
    from scripts import simulate_psychophysics as phase_g

    class FlatModel:
        dt_ms = 4.0

        def __call__(self, X, lengths, return_internals=False):
            batch, frames, channels = X.shape
            assert channels == 8 and lengths.shape == (batch,)
            values = torch.full((batch, frames), tonic_velocity, dtype=X.dtype, device=X.device)
            gates = torch.full((batch, frames, 2), 0.5, dtype=X.dtype, device=X.device)
            return values, {"routing_gates": gates}

    X = torch.zeros(2, 12, 8)
    X[:, :, 0] = 20.0
    X[:, :, 1] = 1.0
    X[:, :, 4:] = 0.25
    lengths = torch.full((2,), 12, dtype=torch.int64)
    meta = phase_g.TrialMeta(
        val_indices=[0, 1], target_ttc_ms=[0.0, 0.0],
        stimulus_conditions=["multisensory", "multisensory"],
        anchor_frames=[2, 2], stim_onset_frame=2, pre_anchor_frames=2,
    )
    data = phase_g.ValidationData(X, torch.zeros(2, 12), lengths, meta)
    monkeypatch.setattr(phase_g, "load_checkpoint", lambda *_: FlatModel())
    monkeypatch.setattr(phase_g, "load_validation_data", lambda *a, **k: data)
    out = tmp_path / "flat"
    monkeypatch.setattr(sys, "argv", ["simulate_psychophysics.py", "--dataset", "synthetic.pt",
                                           "--output_dir", str(out)])
    phase_g.main()
    status = json.loads((out / "bayesian_reliability.json").read_text())
    summary = json.loads((out / "psychophysics_summary.json").read_text())
    assert status["status"] == "unavailable_latency"
    assert status["n_ttc0"] == 2 and "No finite latency" in status["reason"]
    assert out.joinpath("bayesian_reliability.png").is_file()
    assert all(cell["n"] == 0 and cell["mean"] is None for cell in summary["latency_stats"].values())
    assert summary["inference"]["effect_sizes_hodges_lehmann_ms"] == {}
    assert_finite_numbers(summary)

    # Exercise the actual runner gate against this freshly written producer output.
    assert phase_g.psychophysics_gate_verdict(out) == "unavailable_latency"
    import os
    import shlex
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    runner = (root / "run_pipeline.sh").read_text(encoding="utf-8")
    def definition(name):
        start = runner.index(f"{name}() {{")
        return runner[start:runner.index("\n}\n", start) + 3]
    gate = "\n".join(definition(name) for name in
                     ("abort", "require_outputs", "require_psychophysics_outputs"))
    command = ("set -euo pipefail\nRED=''; RESET=''; YELLOW=''\n"
               f"DRY_RUN=0; OUTPUT_DIR={shlex.quote(str(out))}; "
               f"PYTHON={shlex.quote(sys.executable)}\n"
               + gate + "\nrequire_psychophysics_outputs 'Phase G'\n")
    env = os.environ | {"PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    passed = subprocess.run(["bash", "-c", command], cwd=root, env=env,
                            text=True, capture_output=True, check=False)
    assert passed.returncode == 0, passed.stderr
    (out / "bayesian_reliability.png").unlink()
    missing = subprocess.run(["bash", "-c", command], cwd=root, env=env,
                             text=True, capture_output=True, check=False)
    assert missing.returncode != 0 and "missing or empty artefact" in missing.stderr


def test_not_applicable_removes_stale_figure_and_summary(monkeypatch, tmp_path):
    """T3: a not_applicable run must not leave a stale figure/summary behind."""
    from scripts import simulate_psychophysics as phase_g

    class FlatModel:
        dt_ms = 4.0

        def __call__(self, X, lengths, return_internals=False):
            batch, frames, channels = X.shape
            assert channels == 8 and lengths.shape == (batch,)
            values = torch.zeros((batch, frames), dtype=X.dtype, device=X.device)
            gates = torch.full((batch, frames, 2), 0.5, dtype=X.dtype, device=X.device)
            return values, {"routing_gates": gates}

    X = torch.zeros(2, 12, 8)
    X[:, :, 0] = 20.0
    X[:, :, 1] = 1.0
    X[:, :, 4:] = 0.25
    lengths = torch.full((2,), 12, dtype=torch.int64)
    meta = phase_g.TrialMeta(
        val_indices=[0, 1], target_ttc_ms=[-225.0, -261.0],
        stimulus_conditions=["multisensory", "multisensory"],
        anchor_frames=[2, 2], stim_onset_frame=2, pre_anchor_frames=2,
    )
    data = phase_g.ValidationData(X, torch.zeros(2, 12), lengths, meta)
    monkeypatch.setattr(phase_g, "load_checkpoint", lambda *_: FlatModel())
    monkeypatch.setattr(phase_g, "load_validation_data", lambda *a, **k: data)

    out = tmp_path / "stale"
    out.mkdir()
    (out / "bayesian_reliability.png").write_bytes(b"\x89PNG stale")
    (out / "psychophysics_summary.json").write_text(
        '{"stale": true}', encoding="utf-8"
    )

    monkeypatch.setattr(sys, "argv", ["simulate_psychophysics.py",
                                      "--dataset", "synthetic.pt",
                                      "--output_dir", str(out)])
    phase_g.main()
    with (out / "bayesian_reliability.json").open(encoding="utf-8") as stream:
        status = json.load(
            stream,
            parse_constant=lambda val: (_ for _ in ()).throw(ValueError(val)),
        )
    assert status["status"] == "not_applicable"
    assert not (out / "bayesian_reliability.png").exists()
    assert not (out / "psychophysics_summary.json").exists()