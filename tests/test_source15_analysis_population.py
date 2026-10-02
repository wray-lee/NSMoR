"""Synthetic split-population controls for C/D/E/F/H descriptive analyses."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.analyze_dynamics import describe_analysis_population


@pytest.mark.parametrize("selected, selection, train, val", [
    ([0, 2, 4], "outer_train_only", 3, 0),
    ([1, 3], "outer_validation_only", 0, 2),
    (list(range(5)), "whole_corpus", 3, 2),
])
def test_synthetic_train_val_and_whole_corpus_controls(selected, selection, train, val):
    population = describe_analysis_population(5, [1, 3], selected)
    assert population["selection"] == selection
    assert population["evidence_scope"] == "in_sample_descriptive_only"
    assert population["holdout_evidence"] is False
    assert population["n_corpus"] == 5
    assert population["n_train"] == 3 and population["n_val"] == 2
    assert population["n_analyzed"] == len(selected)
    assert population["n_analyzed_train"] == train
    assert population["n_analyzed_val"] == val
    assert population["train_coverage"] == train / 3
    assert population["val_coverage"] == val / 2
    assert population["outer_val_indices"] == [1, 3]


@pytest.mark.parametrize("scope", ["nested_outer_validation", "diagnostic_global_oof"])
def test_selected_checkpoint_outer_val_is_descriptive_not_holdout(scope):
    population = describe_analysis_population(5, [1, 3], [1, 3], validation_scope=scope)
    assert population["checkpoint_validation_scope"] == scope
    assert population["selection"] == "outer_validation_only"
    assert (population["n_train"], population["n_val"]) == (3, 2)
    assert (population["n_analyzed_train"], population["n_analyzed_val"]) == (0, 2)
    assert population["outer_val_indices"] == [1, 3]
    assert population["evidence_scope"] == "in_sample_descriptive_only"
    assert population["holdout_evidence"] is False


def test_no_persisted_split_cannot_claim_holdout():
    population = describe_analysis_population(5, None, list(range(5)))
    assert population["selection"] == "whole_corpus"
    assert population["evidence_scope"] == "in_sample_descriptive_only"
    assert population["holdout_evidence"] is False
    assert population["n_train"] is population["n_val"] is None
    assert population["n_analyzed_train"] is population["n_analyzed_val"] is None
    assert population["outer_val_indices"] is None


@pytest.mark.parametrize("phase", ["analyze_dynamics", "simulate_lesion", "analyze_jacobian",
                                   "analyze_integration", "analyze_gating"])
def test_producer_loader_keeps_persisted_split_and_full_corpus_denominator(monkeypatch, phase):
    module = importlib.import_module("scripts." + phase)
    n = 5
    source = {
        "X_seqs": [np.zeros((6, 8), dtype=np.float32) for _ in range(n)],
        "Y_seqs": [np.zeros(6, dtype=np.float32) for _ in range(n)],
        "labels": np.array([0, 1, 2, 3, 0]),
        "lengths": np.full(n, 6),
        "anchor_frames": [2] * n,
        "mcmc_priors": np.full((n, 4), 0.25, dtype=np.float32),
        "is_pure_wind": np.zeros(n, dtype=bool),
        "stimulus_conditions": ["visual_only"] * n,
        "session_ids": [f"animal_{i}_session_1" for i in range(n)],
        "trial_ids": list(range(n)),
    }
    monkeypatch.setattr(Path, "exists", lambda self: True)
    def load_compact(
        path: Path, *, restore_provenance: bool = True, **kwargs: object,
    ) -> tuple[dict, str]:
        assert restore_provenance is False
        return source, "f" * 64

    monkeypatch.setattr(module, "load_dataset_with_fingerprint", load_compact)
    monkeypatch.setattr(module, "validate_dataset_provenance", lambda *args: None)
    monkeypatch.setattr(module, "load_analysis_priors", lambda *args, **kwargs:
                        (source["mcmc_priors"], np.array([1, 3])))
    monkeypatch.setattr(module, "create_optimized_dataloader", lambda dataset, **kwargs:
                        SimpleNamespace(dataset=dataset))
    if phase == "analyze_gating":
        monkeypatch.setattr(module, "_shared_load_model", lambda *args: SimpleNamespace(dt_ms=4.))
        loader = module.load_model_and_dataset(Path("model.pth"), Path("dataset.pt"))[1]
    else:
        loader = module.load_dataset(Path("dataset.pt"))[0]
    assert loader.dataset.source_indices == list(range(n))
    assert len(loader.dataset) == n
    population = loader.dataset.analysis_population
    assert population["selection"] == "whole_corpus"
    assert population["evidence_scope"] == "in_sample_descriptive_only"
    assert population["holdout_evidence"] is False
    assert (population["n_train"], population["n_val"]) == (3, 2)
    assert (population["n_analyzed_train"], population["n_analyzed_val"]) == (3, 2)
    assert population["outer_val_indices"] == [1, 3]


@pytest.mark.parametrize("selected", [[0, 2, 4], [1, 3], list(range(5))])
def test_png_and_integration_json_report_selected_population(tmp_path, monkeypatch, selected):
    """Persist train/val/whole controls in the existing C and F artifact paths."""
    from PIL import Image
    from scripts import analyze_dynamics as dynamics, analyze_integration as integration

    population = describe_analysis_population(5, [1, 3], selected,
                                             validation_scope="nested_outer_validation")
    assert population["checkpoint_validation_scope"] == "nested_outer_validation"
    for panel in ("plot_panel_a_3d_manifold", "plot_panel_b_routing_gates",
                  "plot_panel_c_lif_spike_rates", "plot_panel_d_pathway_dominance"):
        monkeypatch.setattr(dynamics, panel, lambda *args, **kwargs: None)
    png = tmp_path / "mechanism_analysis.png"
    dynamics.create_panel_figure([], np.array([]), np.array([0.0, 0.0, 0.0]),
                                 SimpleNamespace(), png, layout="1x2",
                                 analysis_population=population)
    with Image.open(png) as image:
        embedded = json.loads(image.info["Description"])["analysis_population"]
    assert embedded == population
    assert len(list(tmp_path.iterdir())) == 1  # Phase C cannot add a sidecar.

    stats = {"visual_only": {
        "latency": {"mean": 8.0, "sem": 0.0, "n": len(selected)},
        "peak_velocity": {"mean": 2.0, "sem": 0.0, "n": len(selected)},
    }}
    summary_path = tmp_path / "integration_summary.json"
    integration.export_integration_summary(stats, summary_path, analysis_population=population)
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "ok"
    assert summary["conditions"]["visual_only"]["latency_to_peak_ms"]["n"] == len(selected)
    assert summary["analysis_population"] == population
    assert summary["analysis_population"]["holdout_evidence"] is False


def test_gating_runner_persists_loader_population(tmp_path, monkeypatch):
    from scripts import analyze_gating as gating
    import pandas as pd

    population = describe_analysis_population(5, [1, 3], range(5),
                                             validation_scope="nested_outer_validation")
    monkeypatch.setattr(gating.ExperimentConfig, "from_yaml", lambda *args: SimpleNamespace(cluster_gating=SimpleNamespace(interp_length=20)))
    monkeypatch.setattr(gating, "load_model_and_dataset", lambda *args, **kwargs:
                        (object(), SimpleNamespace(dataset=SimpleNamespace(analysis_population=population)),
                         np.arange(5), None, None))
    monkeypatch.setattr(gating, "extract_and_cluster_gates", lambda **kwargs:
                        {"umap_embedding": None, "sequences": [], "labels_k4": np.array([])})
    monkeypatch.setattr(gating, "plot_trajectories_by_cluster", lambda *args, **kwargs: None)
    monkeypatch.setattr(gating, "build_summary_json", lambda *args, **kwargs: {"n_trials": 5})
    monkeypatch.setattr(gating, "build_statistics_csv", lambda *args, **kwargs: pd.DataFrame({"n": [5]}))
    gating.run_analysis(Path("checkpoint.pth"), Path("dataset.pt"), tmp_path)
    summary = json.loads((tmp_path / "gating_cluster_summary.json").read_text())
    assert summary["analysis_population"] == population
    assert summary["n_trials"] == 5


def test_lesion_sidecar_declares_full_corpus_split(tmp_path, monkeypatch):
    from scripts import simulate_lesion as lesion

    population = describe_analysis_population(2, [1], range(2),
                                             validation_scope="nested_outer_validation")
    data = SimpleNamespace(analysis_population=population,
                           cropped_reference_frames=[1, 1],
                           analysis_reference_trials=[
                               {"trial_id": i, "row_index": i, "label": 0,
                                "reference_source": "recorded_dataset_anchor",
                                "recording_prefix": f"recording_{i}"}
                               for i in range(2)])
    monkeypatch.setattr(lesion, "load_model_from_checkpoint", lambda *args: SimpleNamespace(dt_ms=4.0))
    monkeypatch.setattr(lesion, "load_dataset", lambda *args, **kwargs:
                        (SimpleNamespace(dataset=data), np.array([0, 0]), [3, 3]))
    trial = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 1.0])]
    monkeypatch.setattr(lesion, "run_full_ablation", lambda *args:
                        {name: (trial, trial, [0, 0]) for name in lesion.CONDITION_NAMES})
    monkeypatch.setattr(lesion, "export_lesion_statistics_csv", lambda **kwargs: None)
    monkeypatch.setattr(lesion, "create_ablation_figure", lambda **kwargs: None)
    stats = tmp_path / "lesion_statistics.csv"
    lesion.run_lesion_experiment(Path("checkpoint.pth"), Path("dataset.pt"),
                                 tmp_path / "ablation.png", stats_output_path=stats)
    sidecar = json.loads(stats.with_suffix(".block_sensitivity.json").read_text())
    assert sidecar["analysis_population"] == population
    assert len(sidecar["trials"]) == population["n_analyzed"] == 2
    assert sidecar["analysis_population"]["n_analyzed_train"] == 1
    assert sidecar["analysis_population"]["n_analyzed_val"] == 1


def test_jacobian_summary_declares_full_corpus_split(tmp_path, monkeypatch):
    from scripts import analyze_jacobian as jacobian

    population = describe_analysis_population(2, [1], range(2),
                                             validation_scope="nested_outer_validation")
    data = SimpleNamespace(analysis_population=population)
    monkeypatch.setattr(jacobian, "load_model_from_checkpoint", lambda *args: SimpleNamespace(dt_ms=4.0))
    monkeypatch.setattr(jacobian, "load_dataset", lambda *args, **kwargs:
                        (SimpleNamespace(dataset=data), np.array([1, 1]), [3, 3],
                         [np.zeros((3, 8)) for _ in range(2)]))
    monkeypatch.setattr(jacobian, "detect_stimulus_onset_frames", lambda *args, **kwargs: [1, 1])
    monkeypatch.setattr(jacobian, "create_jacobian_adapter", lambda *args, **kwargs: object())
    monkeypatch.setattr(jacobian, "extract_gru_states_at_epochs", lambda **kwargs: {})
    spectra = {name: np.array([[0.9 + 0j]]) for name in jacobian.EPOCH_DEFINITIONS}
    controlled = {name: {"status": "ok", "n_pass": 1, "n_total": 1}
                  for name in jacobian.EPOCH_DEFINITIONS}
    monkeypatch.setattr(jacobian, "compute_eigenvalues_at_epochs", lambda **kwargs:
                        (spectra, {}, controlled))
    monkeypatch.setattr(jacobian, "plot_eigenvalue_spectrum", lambda **kwargs: None)
    output = tmp_path / "jacobian_spectrum.png"
    jacobian.run_jacobian_analysis(Path("checkpoint.pth"), Path("dataset.pt"), output)
    summary = json.loads(output.with_suffix(".json").read_text())
    assert summary["analysis_population"] == population
    assert set(summary["spectral_statistics"]) == set(jacobian.EPOCH_DEFINITIONS)


def test_jacobian_spectra_withheld_when_frozen_control_fails(tmp_path, monkeypatch):
    """A failed frozen-input control must not publish own-input spectra."""
    from scripts import analyze_jacobian as jacobian

    population = describe_analysis_population(2, [1], range(2),
                                             validation_scope="nested_outer_validation")
    data = SimpleNamespace(analysis_population=population)
    monkeypatch.setattr(jacobian, "load_model_from_checkpoint", lambda *args: SimpleNamespace(dt_ms=4.0))
    monkeypatch.setattr(jacobian, "load_dataset", lambda *args, **kwargs:
                        (SimpleNamespace(dataset=data), np.array([1, 1]), [3, 3],
                         [np.zeros((3, 8)) for _ in range(2)]))
    monkeypatch.setattr(jacobian, "detect_stimulus_onset_frames", lambda *args, **kwargs: [1, 1])
    monkeypatch.setattr(jacobian, "create_jacobian_adapter", lambda *args, **kwargs: object())
    monkeypatch.setattr(jacobian, "extract_gru_states_at_epochs", lambda **kwargs: {})
    spectra = {name: np.array([[0.9 + 0j]]) for name in jacobian.EPOCH_DEFINITIONS}
    withheld = {
        name: {"status": "withheld", "withheld_reason": "frozen_map_gate_unavailable",
               "n_pass": 0, "n_total": 5, "residual_median": 0.4}
        for name in jacobian.EPOCH_DEFINITIONS
    }
    monkeypatch.setattr(jacobian, "compute_eigenvalues_at_epochs", lambda **kwargs:
                        (spectra, {}, withheld))
    rendered: list[str] = []
    monkeypatch.setattr(jacobian, "plot_eigenvalue_spectrum",
                        lambda **kwargs: rendered.append("spectrum"))
    monkeypatch.setattr(jacobian, "plot_withheld_spectrum_notice",
                        lambda *args, **kwargs: rendered.append("withheld"))
    output = tmp_path / "jacobian_spectrum.png"
    jacobian.run_jacobian_analysis(Path("checkpoint.pth"), Path("dataset.pt"), output)
    summary = json.loads(output.with_suffix(".json").read_text())
    assert summary["status"] == "withheld"
    assert summary["spectral_statistics"] == {}
    # No own-input spectral magnitude leaked into the artifact.
    assert "mag_mean" not in json.dumps(summary)
    # Finite diagnostics survive the withholding.
    assert summary["frozen_input_control"] == withheld
    assert rendered == ["withheld"]
    # Every epoch is named as withheld, and none carries a spectrum.
    assert set(summary["withheld_epochs"]) == set(jacobian.EPOCH_DEFINITIONS)
    assert summary["controlled_epochs"] == []
    assert set(summary["withheld_spectra"]) == set(jacobian.EPOCH_DEFINITIONS)
    assert all("mag_mean" not in entry for entry in summary["withheld_spectra"].values())


def _run_jacobian_with_controls(tmp_path, monkeypatch, spectra, frozen_control):
    """Drive ``run_jacobian_analysis`` with synthetic spectra/control states."""
    from scripts import analyze_jacobian as jacobian

    population = describe_analysis_population(2, [1], range(2),
                                             validation_scope="nested_outer_validation")
    data = SimpleNamespace(analysis_population=population)
    monkeypatch.setattr(jacobian, "load_model_from_checkpoint", lambda *args: SimpleNamespace(dt_ms=4.0))
    monkeypatch.setattr(jacobian, "load_dataset", lambda *args, **kwargs:
                        (SimpleNamespace(dataset=data), np.array([1, 1]), [3, 3],
                         [np.zeros((3, 8)) for _ in range(2)]))
    monkeypatch.setattr(jacobian, "detect_stimulus_onset_frames", lambda *args, **kwargs: [1, 1])
    monkeypatch.setattr(jacobian, "create_jacobian_adapter", lambda *args, **kwargs: object())
    monkeypatch.setattr(jacobian, "extract_gru_states_at_epochs", lambda **kwargs: {})
    monkeypatch.setattr(jacobian, "compute_eigenvalues_at_epochs", lambda **kwargs:
                        (spectra, {}, frozen_control))
    monkeypatch.setattr(jacobian, "plot_eigenvalue_spectrum", lambda **kwargs: None)
    monkeypatch.setattr(jacobian, "plot_withheld_spectrum_notice",
                        lambda *args, **kwargs: None)
    output = tmp_path / "jacobian_spectrum.png"
    jacobian.run_jacobian_analysis(Path("checkpoint.pth"), Path("dataset.pt"), output)
    return json.loads(output.with_suffix(".json").read_text())


def test_compute_eigenvalues_does_not_log_magnitudes_before_frozen_gate(
    monkeypatch, caplog,
):
    """The eigenvalue computer must not log magnitudes before the gate.

    ``compute_eigenvalues_at_epochs`` runs BEFORE the frozen-input control
    decides which epochs are publishable.  Logging own-input magnitudes
    there leaks unverified spectra into run.log for epochs the control
    later withholds.
    """
    import logging

    import torch

    from scripts import analyze_jacobian as jacobian

    leak_mag = 0.7654321

    class _MockAdapter:
        def __init__(self) -> None:
            self._gru_cell = SimpleNamespace(training=True)

        def compute_jacobian_batch(self, h_batch, x_batch):
            b, h = h_batch.shape
            return torch.diag(torch.full((h,), leak_mag, dtype=torch.float32)).repeat(b, 1, 1)

        def test_attractor_convergence(self, h, x):
            return False, 0.0, None

    h_states = torch.zeros(6, 4)
    x_inputs = torch.zeros(6, 4)
    epoch_data = {"early": (h_states, x_inputs)}

    monkeypatch.setattr(jacobian, "_fixed_point_residual",
                        lambda adapter, h, x: 0.2)
    # Force the frozen-map gate to be unavailable, so the epoch is withheld.
    monkeypatch.setattr(jacobian, "_calibrate_fp_threshold",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            ValueError("no bimodal residual distribution")))

    with caplog.at_level(logging.INFO, logger="scripts.analyze_jacobian"):
        _eigs, _attr, frozen = jacobian.compute_eigenvalues_at_epochs(
            adapter=_MockAdapter(), epoch_data=epoch_data,
            device=torch.device("cpu"),
        )

    assert frozen["early"]["status"] == "withheld"
    # No own-input magnitude for the withheld epoch reached the log.
    assert "0.7654" not in caplog.text
    assert "Eigenvalue stats" not in caplog.text


def test_plot_eigenvalue_spectrum_renders_withheld_panel(tmp_path):
    """The withheld epoch renders a labeled panel, not a spectrum."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts import analyze_jacobian as jacobian

    controlled = {"early": np.array([[0.9 + 0j]]), "sustained": np.array([[0.8 + 0j]])}
    out = tmp_path / "jacobian_spectrum.png"
    try:
        jacobian.plot_eigenvalue_spectrum(
            eigenvalue_results=controlled, output_path=out,
            withheld_epochs=["transient"],
        )
    finally:
        plt.close("all")
    assert out.exists()


# Distinct magnitudes make any leak of a withheld epoch's spectrum detectable.
_MIXED_SPECTRA = {
    "early": np.array([[0.1234567 + 0j]]),
    "transient": np.array([[0.7654321 + 0j]]),   # withheld
    "sustained": np.array([[0.2222222 + 0j]]),
}
_MIXED_CONTROL = {
    "early": {"status": "ok", "n_pass": 3, "n_total": 5},
    "transient": {"status": "withheld",
                  "withheld_reason": "insufficient_stationary_states",
                  "n_pass": 1, "n_total": 5, "residual_median": 0.31},
    "sustained": {"status": "ok", "n_pass": 4, "n_total": 5},
}


def test_jacobian_mixed_control_publishes_only_controlled_epochs(tmp_path, monkeypatch):
    """A partially failed control must publish only the controlled subset."""
    summary = _run_jacobian_with_controls(
        tmp_path, monkeypatch, _MIXED_SPECTRA, _MIXED_CONTROL,
    )
    assert summary["status"] == "partial"
    assert set(summary["spectral_statistics"]) == {"early", "sustained"}
    assert summary["controlled_epochs"] == ["early", "sustained"]
    assert summary["withheld_epochs"] == ["transient"]
    # The withheld epoch is named with a truthful reason and finite
    # diagnostics, but carries NO spectrum of its own.
    assert set(summary["withheld_spectra"]) == {"transient"}
    withheld_entry = summary["withheld_spectra"]["transient"]
    assert withheld_entry["withheld_reason"] == "insufficient_stationary_states"
    assert withheld_entry["residual_median"] == 0.31
    assert "mag_mean" not in withheld_entry
    # The withheld epoch's own-input magnitude never reaches the artifact.
    assert "0.7654" not in json.dumps(summary)


def test_jacobian_mixed_control_does_not_log_withheld_magnitudes(
    tmp_path, monkeypatch, caplog,
):
    """run.log must consume the same controlled subset — no pre-gate leak."""
    import logging

    with caplog.at_level(logging.INFO, logger="scripts.analyze_jacobian"):
        _run_jacobian_with_controls(
            tmp_path, monkeypatch, _MIXED_SPECTRA, _MIXED_CONTROL,
        )
    logged = caplog.text
    # Controlled epochs are logged ...
    assert "early: |λ|_mean=0.1235" in logged
    assert "sustained: |λ|_mean=0.2222" in logged
    # ... the withheld epoch's magnitude is NOT.
    assert "0.7654" not in logged
    # The withholding itself is reported.
    assert "withheld" in logged.lower()


def test_jacobian_all_withheld_does_not_log_own_input_magnitudes(
    tmp_path, monkeypatch, caplog,
):
    """All-withheld must stay an explicit withholding with no logged spectra."""
    import logging

    withheld = {
        name: {"status": "withheld", "withheld_reason": "frozen_map_gate_unavailable",
               "n_pass": 0, "n_total": 5, "residual_median": 0.4}
        for name in _MIXED_SPECTRA
    }
    with caplog.at_level(logging.INFO, logger="scripts.analyze_jacobian"):
        summary = _run_jacobian_with_controls(
            tmp_path, monkeypatch, _MIXED_SPECTRA, withheld,
        )
    assert summary["status"] == "withheld"
    assert summary["spectral_statistics"] == {}
    assert "0.7654" not in caplog.text
    assert "0.1235" not in caplog.text
    assert "withheld" in caplog.text.lower()


def _strict_json_load(text: str):
    """Parse JSON with RFC-8259 strictness (no NaN/Infinity constants)."""
    def _reject(token: str):
        raise AssertionError(f"non-strict JSON token {token!r}")

    return json.loads(text, parse_constant=_reject)


def test_jacobian_nonfinite_residuals_publish_strict_json_without_nan(
    tmp_path, monkeypatch,
):
    """Nonfinite frozen-map residuals must never be summarized into the JSON.

    ``_calibrate_fp_threshold`` rejects a residual array containing
    NaN/Inf, so the epoch is withheld with reason
    ``frozen_map_gate_unavailable``.  Summarizing that same nonfinite
    array would emit bare ``NaN``/``Infinity`` tokens into the artifact
    (rejected by strict RFC-8259 readers).  Drive a real nonfinite
    residual through ``run_jacobian_analysis`` and require the published
    JSON to be strict, spectrum-free, and diagnostic-null.
    """
    import torch

    from scripts import analyze_jacobian as jacobian

    class _MockAdapter:
        def __init__(self) -> None:
            self._gru_cell = SimpleNamespace(training=True)

        def compute_jacobian_batch(self, h_batch, x_batch):
            b, h = h_batch.shape
            return torch.diag(
                torch.full((h,), 0.5, dtype=torch.float32)
            ).repeat(b, 1, 1)

        def test_attractor_convergence(self, h, x):
            return False, 0.0, None

    population = describe_analysis_population(
        2, [1], range(2), validation_scope="nested_outer_validation",
    )
    data = SimpleNamespace(analysis_population=population)
    monkeypatch.setattr(jacobian, "load_model_from_checkpoint",
                        lambda *args: SimpleNamespace(dt_ms=4.0))
    monkeypatch.setattr(jacobian, "load_dataset", lambda *args, **kwargs:
                        (SimpleNamespace(dataset=data), np.array([1, 1]),
                         [3, 3], [np.zeros((3, 8)) for _ in range(2)]))
    monkeypatch.setattr(jacobian, "detect_stimulus_onset_frames",
                        lambda *args, **kwargs: [1, 1])
    monkeypatch.setattr(jacobian, "create_jacobian_adapter",
                        lambda *args, **kwargs: _MockAdapter())
    monkeypatch.setattr(jacobian, "extract_gru_states_at_epochs",
                        lambda **kwargs: {
                            "early": (torch.zeros(6, 4), torch.zeros(6, 4)),
                        })
    # Every frozen-map residual is nonfinite, so the real
    # `_calibrate_fp_threshold` rejects the array and withholds the epoch.
    monkeypatch.setattr(jacobian, "_fixed_point_residual",
                        lambda adapter, h, x: float("nan"))
    monkeypatch.setattr(jacobian, "plot_eigenvalue_spectrum",
                        lambda **kwargs: None)
    monkeypatch.setattr(jacobian, "plot_withheld_spectrum_notice",
                        lambda *args, **kwargs: None)

    output = tmp_path / "jacobian_spectrum.png"
    jacobian.run_jacobian_analysis(
        Path("checkpoint.pth"), Path("dataset.pt"), output,
    )

    raw = output.with_suffix(".json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    summary = _strict_json_load(raw)
    assert summary["status"] == "withheld"
    assert summary["spectral_statistics"] == {}
    assert summary["controlled_epochs"] == []

    entry = summary["withheld_spectra"]["early"]
    assert entry["withheld_reason"] == "frozen_map_gate_unavailable"
    # Unavailable residual summaries are explicit null, never a fabricated
    # finite sentinel and never a nonfinite number.
    assert entry["residual_median"] is None
    # Finite counts survive the withholding.
    assert entry["n_pass"] == 0
    assert entry["n_total"] == 6

    frozen_entry = summary["frozen_input_control"]["early"]
    assert frozen_entry["status"] == "withheld"
    assert frozen_entry["withheld_reason"] == "frozen_map_gate_unavailable"
    assert "residual_median" not in frozen_entry
    assert "residual_min" not in frozen_entry
    assert "residual_max" not in frozen_entry


@pytest.mark.parametrize("indices", [[1.5, 3], [True, 3], [1, 1], [-1, 3]])
def test_invalid_outer_validation_indices_cannot_misstate_coverage(indices):
    with pytest.raises(ValueError, match="Outer-validation indices"):
        describe_analysis_population(5, indices, range(5))
