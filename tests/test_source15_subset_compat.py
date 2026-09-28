"""SOURCE15 subset and restricted loader regressions on synthetic artifacts."""

import io
from pathlib import Path
import types
import zipfile

import numpy as np
import pytest
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION
from nsmor.model_utils import validate_dataset_provenance
from nsmor.pipeline import nested_prior
from scripts.make_subset_dataset import main


def _dataset():
    sessions = [f"recording_{name}_session_{block}" for name in "ABCD" for block in (1, 2)]
    trial_ids = [f"trial_{i}" for i in range(8)]
    return {
        "X_seqs": [np.full((3, 8), i, dtype=np.float32) for i in range(8)],
        "Y_seqs": [np.zeros(3, dtype=np.float32) for _ in range(8)],
        "labels": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        "lengths": np.full(8, 3),
        "mcmc_priors": np.full((8, 4), 0.25, dtype=np.float32),
        "session_ids": sessions,
        "trial_ids": trial_ids,
        "trial_specs": [
            {"session_id": sid, "trial_id": tid, "source_pairs": [f"raw_{i}"]}
            for i, (sid, tid) in enumerate(zip(sessions, trial_ids))
        ],
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "mcmc_prior_provenance": "oof_2fold_recording_prefix_grouped_cv",
        "animal_identity_status": "unverified",
    }


def test_subset_slices_specs_and_reloads(tmp_path):
    data = _dataset()
    source, output = tmp_path / "source.pt", tmp_path / "subset.pt"
    torch.save(data, source)
    main(["--input", str(source), "--output", str(output), "--n_recording_prefixes", "2"])
    result, _ = nested_prior.load_dataset_with_fingerprint(output)
    assert len(result["X_seqs"]) == len(result["trial_specs"]) == 4
    selected = [data["trial_ids"].index(tid) for tid in result["trial_ids"]]
    assert result["trial_specs"] == [data["trial_specs"][i] for i in selected]
    assert [int(x[0, 0]) for x in result["X_seqs"]] == selected
    assert validate_dataset_provenance(result, output) == "unverified"


def test_subset_rejects_mismatched_spec_identity_before_write(tmp_path):
    data = _dataset()
    data["trial_specs"][0]["trial_id"] = "another_trial"
    source, output = tmp_path / "source.pt", tmp_path / "subset.pt"
    torch.save(data, source)
    with pytest.raises(ValueError, match="trial_specs.*trial_id"):
        main(["--input", str(source), "--output", str(output), "--n_recording_prefixes", "4"])
    assert not output.exists()


def test_subset_validates_generated_rows_before_write(tmp_path, monkeypatch):
    from scripts import make_subset_dataset as subset_module

    data = _dataset()
    source, output = tmp_path / "source.pt", tmp_path / "subset.pt"
    torch.save(data, source)
    original = subset_module.subset_dataset

    def unaligned(*args):
        result, kept, prefixes = original(*args)
        result["trial_specs"] = data["trial_specs"]
        return result, kept, prefixes

    monkeypatch.setattr(subset_module, "subset_dataset", unaligned)
    with pytest.raises(RuntimeError, match="aligned usable session_ids"):
        main(["--input", str(source), "--output", str(output), "--n_recording_prefixes", "2"])
    assert not output.exists()


def _spelling(payload, old, new):
    src, dst = io.BytesIO(payload), io.BytesIO()
    with zipfile.ZipFile(src) as reader, zipfile.ZipFile(dst, "w") as writer:
        for info in reader.infolist():
            value = reader.read(info.filename)
            if info.filename.endswith("/data.pkl"):
                assert old in value
                value = value.replace(old, new)
            writer.writestr(info, value)
    return dst.getvalue()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("numpy1_namespace", [False, True])
def test_numpy_array_both_pickle_names_and_numpy1_namespace(legacy, numpy1_namespace, monkeypatch):
    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    buffer = io.BytesIO()
    torch.save({"values": values}, buffer)
    payload = buffer.getvalue()
    modern = b"cnumpy._core.multiarray\n_reconstruct\n"
    old = b"cnumpy.core.multiarray\n_reconstruct\n"
    if legacy:
        payload = _spelling(payload, modern, old)
    if numpy1_namespace:
        # Simulate the NumPy 1 public module surface while NumPy itself remains loaded.
        monkeypatch.setattr(nested_prior, "np", types.SimpleNamespace(
            dtype=np.dtype, ndarray=np.ndarray,
            core=types.SimpleNamespace(multiarray=types.SimpleNamespace(
                _reconstruct=np._core.multiarray._reconstruct)),
        ))
    np.testing.assert_array_equal(nested_prior.load_artifact_bytes(payload)["values"], values)


def _write_marker(path):
    Path(path).write_text("executed")


class _Malicious:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return _write_marker, (str(self.path),)


def test_restricted_loader_rejects_unknown_reducer(tmp_path):
    marker = tmp_path / "marker"
    buffer = io.BytesIO()
    torch.save({"unsafe": _Malicious(marker)}, buffer)
    with pytest.raises(ValueError, match="Unexpected serialized global"):
        nested_prior.load_artifact_bytes(buffer.getvalue())
    assert not marker.exists()


def test_declared_lower_bound_and_container_supply_restricted_api():
    root = Path(__file__).resolve().parents[1]
    assert "numpy>=1.26.0" in (root / "pyproject.toml").read_text()
    assert "torch>=2.6.0" in (root / "pyproject.toml").read_text()
    assert "numpy>=1.26.0" in (root / "requirements.txt").read_text()
    assert "torch>=2.6.0" in (root / "requirements.txt").read_text()
    assert "https://download.pytorch.org/whl/cu124" in (root / "requirements.txt").read_text()
    dockerfile = (root / "Dockerfile").read_text()
    assert "FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime" in dockerfile
    assert "umap-learn>=0.5.0" in (root / "requirements.txt").read_text()
    assert "umap-learn>=0.5.0" in (root / "pyproject.toml").read_text()
    assert "|| true" not in dockerfile
    assert dockerfile.index("RUN python -c \"import torch, numpy as np; from torch.serialization import") > dockerfile.index("RUN pip install --no-cache-dir -e")
    assert "torch.__version__.split('+')[0] == '2.6.0'" in dockerfile
    assert "np.__version__.split('.')[:2]" in dockerfile
    import ast
    ast.parse(dockerfile.split('RUN python -c "', 1)[1].split('"', 1)[0])
    for api in ("get_safe_globals", "get_unsafe_globals_in_checkpoint", "safe_globals",
                "clear_safe_globals", "add_safe_globals"):
        assert hasattr(torch.serialization, api)
