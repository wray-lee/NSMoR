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
    monkeypatch.setattr(module, "load_dataset_with_fingerprint", lambda path: (source, "f" * 64))
    monkeypatch.setattr(module, "validate_dataset_provenance", lambda *args: None)
    monkeypatch.setattr(module, "load_analysis_priors", lambda *args, **kwargs:
                        (source["mcmc_priors"], np.array([1, 3])))
    monkeypatch.setattr(module, "create_optimized_dataloader", lambda dataset, **kwargs:
                        SimpleNamespace(dataset=dataset))
    if phase == "analyze_gating":
        monkeypatch.setattr(module, "_shared_load_model", lambda *args: SimpleNamespace())
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
    monkeypatch.setattr(jacobian, "compute_eigenvalues_at_epochs", lambda **kwargs:
                        (spectra, {}, {}))
    monkeypatch.setattr(jacobian, "plot_eigenvalue_spectrum", lambda **kwargs: None)
    output = tmp_path / "jacobian_spectrum.png"
    jacobian.run_jacobian_analysis(Path("checkpoint.pth"), Path("dataset.pt"), output)
    summary = json.loads(output.with_suffix(".json").read_text())
    assert summary["analysis_population"] == population
    assert set(summary["spectral_statistics"]) == set(jacobian.EPOCH_DEFINITIONS)


@pytest.mark.parametrize("indices", [[1.5, 3], [True, 3], [1, 1], [-1, 3]])
def test_invalid_outer_validation_indices_cannot_misstate_coverage(indices):
    with pytest.raises(ValueError, match="Outer-validation indices"):
        describe_analysis_population(5, indices, range(5))
