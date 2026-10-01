"""Nested-prior artifact loading with fail-closed split/provenance validation.

The nested artifact produced by ``scripts/evaluate_nested_prior.py`` pins an
exact recording-prefix-disjoint outer train/val split and an (N, 4) prior matrix
(inner OOF on outer-train, fold-ensemble on outer-val).  Downstream training
must consume those indices and priors *verbatim*: recomputing a split or
falling back to global OOF priors exposes outer-validation labels to the
prior fit. Prefixes derived from session names do not verify distinct animals.

Every check here is fail-closed.  A missing key, a fingerprint mismatch, a
non-partition index pair, or a recording prefix that crosses the outer boundary raises
rather than silently degrading to the legacy path.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from nsmor.config import PIPELINE_SEMANTICS_VERSION, FeatureConfig, TimeWindowConfig
from nsmor.mcmc_module import MCMCPriorGenerator
from nsmor.pipeline.clock_storage import restore_clock_provenance
from nsmor.pipeline.grouping import (
    animal_keys_of,
    check_group_disjoint,
    grouped_train_val_split,
    prior_identity_status,
)

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = (
    "nested_priors",
    "train_indices",
    "val_indices",
    "is_nested_cv",
    "source_fingerprint",
    "pipeline_semantics_version",
    "split_seed",
    "val_split",
)


_ALLOWED_PICKLE_GLOBALS = {
    "nsmor.config.FeatureConfig", "nsmor.config.TimeWindowConfig",
    "nsmor.mcmc_module.MCMCPriorGenerator", "torch.nn.modules.linear.Linear",
    "numpy._core.multiarray._reconstruct", "numpy.core.multiarray._reconstruct",
    "numpy.dtype", "numpy.ndarray",
}


def load_artifact_bytes(payload: bytes, map_location: Any = None) -> Any:
    """Restricted-decode captured artifact bytes using the fixed QC allowlist.

    Unknown globals fail before deserialization; no unrestricted fallback exists.
    """
    dtypes = [np.dtype(name).__class__ for name in (
        "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
        "uint64", "float16", "float32", "float64", "complex64", "complex128",
        "str", "bytes", "object",
    )]
    reconstruct = (np._core if hasattr(np, "_core") else np.core).multiarray._reconstruct
    safe = [FeatureConfig, TimeWindowConfig, MCMCPriorGenerator, torch.nn.Linear,
            (reconstruct, "numpy._core.multiarray._reconstruct"),
            (reconstruct, "numpy.core.multiarray._reconstruct"),
            np.dtype, np.ndarray, *dtypes]
    # ponytail: PyTorch's registry is process-wide; serialize loads if threads are added.
    previous = torch.serialization.get_safe_globals()
    try:
        torch.serialization.clear_safe_globals()
        found = set(torch.serialization.get_unsafe_globals_in_checkpoint(io.BytesIO(payload)))
        unexpected = found - _ALLOWED_PICKLE_GLOBALS
        if unexpected:
            raise ValueError(f"Unexpected serialized global(s): {sorted(unexpected)}")
        with torch.serialization.safe_globals(safe):
            return torch.load(io.BytesIO(payload), weights_only=True, map_location=map_location)
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(previous)


def compute_source_fingerprint(dataset_path: Union[str, Path]) -> str:
    """Compute SHA-256 hex digest of the source dataset file.

    Single canonical implementation (``str`` and ``Path`` both accepted).
    Fails closed on an unusable path: a provenance fingerprint that is
    silently "" would let a generated artifact claim an unverifiable
    source and push the rejection to some later consumer.

    Args:
        dataset_path: Path to dataset file (``str`` or ``Path``).

    Returns:
        Hex-encoded SHA-256 string.

    Raises:
        TypeError: ``dataset_path`` is ``None`` or not path-like.
        FileNotFoundError: ``dataset_path`` does not exist.
    """
    if dataset_path is None:
        raise TypeError(
            "compute_source_fingerprint requires a dataset path; "
            "None is not a fingerprintable source (fail closed)."
        )
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot fingerprint missing dataset file: {path}"
        )
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_dataset_with_fingerprint(
    dataset_path: Union[str, Path], *, map_location: Any = None,
    expected_dt_ms: Optional[float] = None,
    restore_provenance: bool = True,
) -> Tuple[Any, str]:
    """Deserialize and fingerprint one immutable source-dataset byte snapshot.

    The digest describes precisely the bytes passed to torch.load. A second
    pathname read after deserialization could otherwise identify another valid
    dataset substituted at the same path. ``expected_dt_ms`` binds the loaded
    record clock to the consuming model/config; it never rewrites the artifact.
    ``restore_provenance=False`` validates every packed clock record sequentially
    but retains the original compressed objects and their shared aliases.
    """
    if dataset_path is None:
        raise TypeError("load_dataset_with_fingerprint requires a dataset path")
    if expected_dt_ms is not None:
        if (isinstance(expected_dt_ms, (bool, np.bool_))
                or not isinstance(expected_dt_ms, (int, float, np.integer, np.floating))
                or not np.isfinite(expected_dt_ms) or expected_dt_ms <= 0):
            raise ValueError("expected_dt_ms must be a finite positive interval")
    source_bytes = Path(dataset_path).read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    dataset = load_artifact_bytes(source_bytes, map_location=map_location)
    del source_bytes
    if isinstance(dataset, dict):
        restore_clock_provenance(dataset, materialize=restore_provenance)
    # Version 2.2 and held-channel anchors also predate the model-grid contract.
    # Any grid-era marker requires sidecars; losing them is not genuine legacy.
    modern_clock = isinstance(dataset, dict) and any(
        key in dataset for key in (
            "model_dt_ms", "model_grid_provenance", "labeling_eligibility",
            "source_clock_provenance",
        )
    )
    if modern_clock and any(
        key not in dataset for key in (
            "model_dt_ms", "model_grid_provenance", "anchor_frames",
            "X_seqs", "Y_seqs", "lengths",
        )
    ):
        raise ValueError("Incomplete modern model grid clock/anchor contract")
    if not modern_clock:
        logger.warning(
            "Dataset has no complete modern grid clock contract; cadence remains "
            "unverified for expected_dt_ms=%s",
            expected_dt_ms,
        )
    # Declared sequences must be stored unpadded; every consumer sees
    # the same frame support before deriving anchors, constructing datasets or scoring.
    if isinstance(dataset, dict) and ("X_seqs" in dataset or "Y_seqs" in dataset) and (
        "lengths" in dataset or "pipeline_semantics_version" in dataset
        or "model_grid_provenance" in dataset
    ):
        xs, ys, lengths = (dataset.get(key) for key in ("X_seqs", "Y_seqs", "lengths"))
        if not all(isinstance(items, (list, tuple, np.ndarray, torch.Tensor))
                   for items in (xs, ys, lengths)) or getattr(lengths, "ndim", 1) != 1:
            raise ValueError("modern dataset requires 1-D integer lengths and aligned X/Y sequences")
        n = len(xs)
        labels = dataset.get("labels")
        if len(ys) != n or len(lengths) != n or (
            "labels" in dataset and (not isinstance(labels, (list, tuple, np.ndarray, torch.Tensor))
                                     or getattr(labels, "ndim", 1) != 1 or len(labels) != n)
        ):
            raise ValueError("X/Y/labels/lengths must be aligned by trial")
        for i, length in enumerate(lengths.tolist() if isinstance(lengths, torch.Tensor) else lengths):
            if (isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer))
                    or length < 1):
                raise ValueError(f"trial {i}: valid length must be a positive nonboolean integer")
            if getattr(xs[i], "shape", None) != (int(length), 8) or getattr(ys[i], "shape", None) != (int(length),):
                raise ValueError(f"trial {i}: stored X/Y must be unpadded and match declared valid length {length}")
        # Convert CPU torch.Tensor to numpy for downstream consumers that expect ndarray
        for key in ("X_seqs", "Y_seqs"):
            if key in dataset and isinstance(dataset[key], (list, tuple)):
                dataset[key] = [seq.detach().cpu().numpy() if isinstance(seq, torch.Tensor) else seq
                                for seq in dataset[key]]
    if isinstance(dataset, dict) and "model_grid_provenance" in dataset:
        from nsmor.pipeline.conditions import derive_anchor_frames
        from nsmor.pipeline.resampling import resolve_model_anchor_frame

        model_dt_ms = dataset["model_dt_ms"]
        if (isinstance(model_dt_ms, (bool, np.bool_))
                or not isinstance(model_dt_ms, (int, float, np.integer, np.floating))
                or not np.isfinite(model_dt_ms) or model_dt_ms <= 0):
            raise ValueError("model_dt_ms must be a finite positive interval")
        if expected_dt_ms is not None and not np.isclose(
                float(model_dt_ms), float(expected_dt_ms), rtol=1e-9, atol=0.):
            raise ValueError(
                f"Dataset model_dt_ms={model_dt_ms} conflicts with "
                f"expected_dt_ms={expected_dt_ms}"
            )
        records = dataset["model_grid_provenance"]
        xs, anchors = dataset.get("X_seqs", []), dataset.get("anchor_frames")
        if not isinstance(records, (list, tuple)) or len(records) != len(xs):
            raise ValueError("model_grid_provenance must be aligned with X_seqs")
        if (not isinstance(anchors, (list, tuple, np.ndarray, torch.Tensor))
                or getattr(anchors, "ndim", 1) != 1 or len(anchors) != len(xs)):
            raise ValueError("source anchor_frames must be aligned with X_seqs")
        if isinstance(anchors, torch.Tensor):
            anchors = anchors.tolist()
        legacy = derive_anchor_frames(xs)
        for i, record in enumerate(records):
            expected = resolve_model_anchor_frame(record)
            if not np.isclose(record["dt_ms"], model_dt_ms, rtol=1e-9, atol=0.):
                raise ValueError(f"trial {i}: grid dt_ms disagrees with model_dt_ms")
            if (len(xs[i]) != record["model_n"] + record["synthetic_prepend_frames"]
                    or expected >= len(xs[i])):
                raise ValueError(f"trial {i}: source anchor/grid disagrees with sequence")
            if (isinstance(anchors[i], (bool, np.bool_))
                    or not isinstance(anchors[i], (int, np.integer))
                    or anchors[i] != expected):
                raise ValueError(f"trial {i}: source anchor disagrees with stored anchor_frames")
            if legacy[i] != expected:
                logger.warning(
                    "trial %d: source anchor frame %d (%s at %.17g ms) supersedes "
                    "legacy held-channel anchor %d; physical values unchanged",
                    i, expected, record["source_anchor_rule"],
                    record["source_anchor_ms"], legacy[i],
                )
    return dataset, digest


def _as_int64_index(name: str, raw: Any, n_total: int) -> np.ndarray:
    """Coerce *raw* to a validated int64 index array."""
    if isinstance(raw, torch.Tensor):
        raw = raw.cpu().numpy()
    arr = np.asarray(raw)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.issubdtype(arr.dtype, np.integer):
        if not (np.issubdtype(arr.dtype, np.floating) and np.all(arr == np.floor(arr))):
            raise ValueError(f"{name} must be integer-valued, got dtype {arr.dtype}")
    idx = arr.astype(np.int64)
    if (idx < 0).any() or (idx >= n_total).any():
        raise ValueError(
            f"{name} contains out-of-range values for n_total={n_total} "
            f"(min={int(idx.min())}, max={int(idx.max())})"
        )
    return idx


def _check_probability_matrix(name: str, raw: Any, n_rows: int, n_cols: int) -> np.ndarray:
    """Validate a probability matrix: shape, finiteness, simplex rows."""
    if isinstance(raw, torch.Tensor):
        raw = raw.cpu().numpy()
    arr = np.asarray(raw, dtype=np.float64)
    expected = (n_rows, n_cols)
    if arr.shape != expected:
        raise ValueError(f"{name} shape {arr.shape} != expected {expected}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    if (arr < 0.0).any():
        raise ValueError(f"{name} contains negative values")
    row_sums = arr.sum(axis=1)
    if np.abs(row_sums - 1.0).max() > 1e-4:
        raise ValueError(
            f"{name} rows must sum to 1 (max deviation "
            f"{float(np.abs(row_sums - 1.0).max()):.6f})"
        )
    return arr


def load_nested_prior_split(
    artifact_path: Path,
    dataset_path: Path,
    n_total: int,
    session_ids: Optional[Sequence[Any]],
    feature_config: Optional[FeatureConfig] = None,
    *,
    loaded_source_fingerprint: Optional[str] = None,
    trusted_historical_artifact_sha256: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a nested-prior artifact and fail closed on any contract violation.

    Validates dataset identity (SHA-256 fingerprint), pipeline semantics,
    exact train/val partition, agreement of the tag with structured seed/fold
    metadata, prior alignment/dim/probability simplex,
    recording-prefix disjointness of the persisted outer split, and *split metadata
    honesty*: the declared ``split_seed`` / ``val_split`` must reproduce
    the persisted partition when fed to
    :func:`~nsmor.pipeline.grouping.grouped_train_val_split` (the same
    function the generator uses).  A sidecar claiming ``val_split=0.99``
    for a 16/8 split is thereby refused.  Realized fractions are always
    derived from the persisted indices (``val_trial_fraction``) — grouped
    split granularity means the requested prefix-level ``val_split`` is
    NOT an exact trial fraction and is never reported as one.

    Args:
        artifact_path: Path to ``nested_split_seed*.pt`` written by
            ``scripts/evaluate_nested_prior.py``.
        dataset_path: Path to the source ``nsmor_dataset.pt`` the artifact
            was generated from (fingerprinted).
        n_total: Trial count of the loaded dataset.
        session_ids: Per-trial session identifiers of the loaded dataset.
            Required — sample-split fallback is forbidden here because the
            outer split is recording-prefix-grouped by construction.
        feature_config: Feature dimension constants (for ``mcmc_dim``).
        trusted_historical_artifact_sha256: Explicit compatibility authorization
            from an independently trusted historical reference, matched to the
            captured sidecar bytes. Required for every animal-named historical
            sidecar; permits missing canonical recording-prefix fields and
            historical fold counts recorded only in the validated tag.
            Computing a digest from an untrusted input does not authenticate its
            historical origin. No artifact tag or field authorizes this opt-in.

    Returns:
        ``(train_indices, val_indices, nested_priors, info)`` where the
        indices are int64 arrays forming an exact partition of
        ``range(n_total)``, ``nested_priors`` is a float64
        ``(n_total, mcmc_dim)`` matrix, and ``info`` is validated
        provenance/split metadata:
        ``nested_prior_artifact``, ``nested_prior_artifact_sha256`` (SHA-256
        of the bytes deserialized), ``nested_prior_fingerprint`` (SHA-256
        of the source dataset), ``mcmc_prior_provenance``,
        ``is_nested_cv``, ``split_seed``, ``n_inner_folds``, ``val_split`` (requested
        recording-prefix target fraction), ``val_trial_fraction`` (realized),
        ``n_train``, ``n_val``, ``n_train_animals``, ``n_val_animals``.
        The legacy animal-named counts are recording-prefix counts, not
        verified independent animals.

    Raises:
        FileNotFoundError: Artifact or dataset path missing.
        ValueError: Any provenance, alignment, disjointness, or split
            metadata violation.
    """
    fc = feature_config if feature_config is not None else FeatureConfig()
    artifact_path = Path(artifact_path)
    dataset_path = Path(dataset_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Nested prior artifact not found: {artifact_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {dataset_path}")
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")

    artifact_bytes = artifact_path.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    artifact = load_artifact_bytes(artifact_bytes)
    if not isinstance(artifact, dict):
        raise ValueError(
            f"Nested prior artifact must be a dict, got {type(artifact).__name__}"
        )

    missing = [k for k in _REQUIRED_KEYS if k not in artifact]
    if missing:
        raise ValueError(f"Nested prior artifact missing required key(s): {missing}")

    if artifact["is_nested_cv"] is not True:
        raise ValueError(
            f"Artifact is_nested_cv={artifact['is_nested_cv']!r} is not True; "
            "only genuine nested-CV artifacts are accepted."
        )

    psv = artifact["pipeline_semantics_version"]
    if str(psv) != str(PIPELINE_SEMANTICS_VERSION):
        raise ValueError(
            f"Artifact pipeline_semantics_version {psv!r} != expected "
            f"{PIPELINE_SEMANTICS_VERSION!r}"
        )

    identity_status = prior_identity_status(
        artifact.get("mcmc_prior_provenance"), artifact.get("animal_identity_status"), nested=True
    )
    if identity_status == "historical_unknown":
        if (not isinstance(trusted_historical_artifact_sha256, str)
                or trusted_historical_artifact_sha256 != artifact_sha256):
            raise ValueError("historical nested artifact requires explicit trusted_historical_artifact_sha256 matching its captured SHA-256")
    elif trusted_historical_artifact_sha256 is not None:
        raise ValueError("trusted_historical_artifact_sha256 is only valid for historical nested artifacts")

    # ── Dataset identity: fingerprint must bind artifact to THIS file ──
    recorded = artifact["source_fingerprint"]
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError(
            f"Artifact source_fingerprint must be a 64-char SHA-256 hex digest, "
            f"got {recorded!r}"
        )
    # Production callers pass the digest of the bytes they deserialized.
    # The path fallback preserves the standalone sidecar-validation API only;
    # it cannot authorize objects already loaded by a caller.
    actual = (loaded_source_fingerprint if loaded_source_fingerprint is not None
              else compute_source_fingerprint(dataset_path))
    if not isinstance(actual, str) or len(actual) != 64 or any(
        c not in "0123456789abcdef" for c in actual
    ):
        raise ValueError("Loaded source fingerprint must be a SHA-256 hex digest")
    if recorded != actual:
        raise ValueError(
            f"Nested prior artifact fingerprint mismatch: artifact was "
            f"generated from a different dataset file.\n"
            f"  artifact: {recorded}\n  dataset:  {actual}\n"
            f"  dataset_path={dataset_path}"
        )

    # ── Exact train/val partition ───────────────────────────────────
    train_idx = _as_int64_index("train_indices", artifact["train_indices"], n_total)
    val_idx = _as_int64_index("val_indices", artifact["val_indices"], n_total)
    if np.intersect1d(train_idx, val_idx).size > 0:
        raise ValueError("train_indices and val_indices overlap")
    if np.unique(train_idx).size != train_idx.size:
        raise ValueError("train_indices contains duplicates")
    if np.unique(val_idx).size != val_idx.size:
        raise ValueError("val_indices contains duplicates")
    if train_idx.size + val_idx.size != n_total:
        raise ValueError(
            f"train ({train_idx.size}) + val ({val_idx.size}) != n_total ({n_total}); "
            "the persisted split must be an exact partition of the dataset"
        )

    # ── Nested prior matrix ─────────────────────────────────────────
    nested_priors = _check_probability_matrix(
        "nested_priors", artifact["nested_priors"], n_total, fc.mcmc_dim
    )

    # Redundant sidecar matrices, when present, must agree with the
    # canonical matrix sliced at the persisted indices (alignment).
    if "train_priors" in artifact:
        tp = _check_probability_matrix(
            "train_priors", artifact["train_priors"], train_idx.size, fc.mcmc_dim
        )
        if not np.allclose(tp, nested_priors[train_idx], atol=1e-8):
            raise ValueError("train_priors is not aligned with nested_priors[train_indices]")
    if "val_priors" in artifact:
        vp = _check_probability_matrix(
            "val_priors", artifact["val_priors"], val_idx.size, fc.mcmc_dim
        )
        if not np.allclose(vp, nested_priors[val_idx], atol=1e-8):
            raise ValueError("val_priors is not aligned with nested_priors[val_indices]")

    # ── Recording-prefix disjointness ────────────────────────────────
    if session_ids is None or len(session_ids) != n_total:
        raise ValueError(
            "session_ids are required to verify the recording-prefix outer "
            f"split (got {None if session_ids is None else len(session_ids)} "
            f"for n_total={n_total}). Sample-split fallback is forbidden."
        )
    group_keys = animal_keys_of(session_ids)
    assert group_keys.shape == (n_total,), (
        f"group_keys shape {group_keys.shape} != ({n_total},)"
    )
    if "group_keys" in artifact:
        recorded_groups = artifact["group_keys"]
        if isinstance(recorded_groups, torch.Tensor):
            recorded_groups = recorded_groups.cpu().numpy()
        recorded_groups = np.asarray(recorded_groups, dtype=object)
        if recorded_groups.shape != (n_total,):
            raise ValueError(
                f"Artifact group_keys shape {recorded_groups.shape} != ({n_total},)"
            )
        if not np.array_equal(recorded_groups.astype(str), group_keys.astype(str)):
            raise ValueError(
                "Artifact group_keys disagree with recording prefixes derived from "
                "the dataset session_ids; artifact and dataset are misaligned."
            )
    check_group_disjoint(train_idx, val_idx, group_keys)
    train_prefixes = sorted(set(group_keys[train_idx].tolist()))
    val_prefixes = sorted(set(group_keys[val_idx].tolist()))
    for field, expected in (
        ("recording_prefix_keys", group_keys.tolist()),
        ("train_recording_prefixes", train_prefixes),
        ("val_recording_prefixes", val_prefixes),
        ("n_train_recording_prefixes", len(train_prefixes)),
        ("n_val_recording_prefixes", len(val_prefixes)),
    ):
        if identity_status == "unverified" and field not in artifact:
            raise ValueError(f"Modern nested artifact missing {field}; fail closed")
        if field in artifact and not np.array_equal(np.asarray(artifact[field]), np.asarray(expected)):
            raise ValueError(f"Artifact {field} disagrees with dataset recording prefixes")

    # ── Split metadata honesty (split_seed / val_split) ─────────
    # The sidecar's declared seed and target fraction must *reproduce*
    # the persisted partition via the generator's own splitter.  Anything
    # else means the metadata misrepresents the actual split and every
    # downstream "seed=…" claim would be a lie.  Realized fractions are
    # recomputed from the persisted indices — a grouped split cannot
    # honour an exact trial fraction, so the requested ``val_split``
    # (recording-prefix target) is never echoed back as if it were the
    # realized fraction.
    raw_seed = artifact["split_seed"]
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, (int, np.integer)):
        raise ValueError(
            f"Artifact split_seed={raw_seed!r} must be an integer, got "
            f"{type(raw_seed).__name__}; fail closed on unverifiable split metadata."
        )
    split_seed = int(raw_seed)
    if "n_inner_folds" not in artifact and identity_status == "unverified":
        raise ValueError("Modern nested artifact missing n_inner_folds")
    # Old sidecars stored only the tag's fold count; external byte trust is required above.
    raw_folds = (artifact["n_inner_folds"] if "n_inner_folds" in artifact else
                 int(artifact["mcmc_prior_provenance"].split("_inner_", 1)[1].split("fold_", 1)[0]))
    if isinstance(raw_folds, bool) or not isinstance(raw_folds, (int, np.integer)) or raw_folds < 2:
        raise ValueError("Artifact n_inner_folds must be an integer with at least 2 folds")
    n_inner_folds = int(raw_folds)
    grouping = "recording_prefix" if identity_status == "unverified" else "animal"
    expected_provenance = f"nested_outer_seed{split_seed}_inner_{n_inner_folds}fold_{grouping}_grouped"
    if artifact["mcmc_prior_provenance"] != expected_provenance:
        raise ValueError("Artifact mcmc_prior_provenance disagrees with split_seed/n_inner_folds")
    raw_val_split = artifact["val_split"]
    if isinstance(raw_val_split, bool) or not isinstance(
        raw_val_split, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Artifact val_split={raw_val_split!r} must be numeric, got "
            f"{type(raw_val_split).__name__}; fail closed on unverifiable split metadata."
        )
    val_split = float(raw_val_split)
    if not (0.0 < val_split < 1.0):
        raise ValueError(
            f"Artifact val_split={val_split} must lie in (0, 1); fail closed "
            "on unverifiable split metadata."
        )
    exp_train, exp_val = grouped_train_val_split(
        session_ids, n_total, val_split=val_split, random_seed=split_seed
    )
    if not (
        np.array_equal(np.sort(exp_train), np.sort(train_idx))
        and np.array_equal(np.sort(exp_val), np.sort(val_idx))
    ):
        raise ValueError(
            "Artifact split_seed/val_split metadata does not reproduce the "
            f"persisted split (claimed split_seed={split_seed}, "
            f"val_split={val_split}; recomputed {exp_train.size}/{exp_val.size} "
            f"vs persisted {train_idx.size}/{val_idx.size}). Sidecar metadata "
            "misrepresents the actual split; fail closed."
        )
    # Realized (trial-level) fractions — computed from the persisted
    # indices, then cross-checked against the sidecar when the sidecar
    # claims them.
    n_train = int(train_idx.size)
    n_val = int(val_idx.size)
    realized_val_fraction = float(n_val) / float(n_total)
    if "val_trial_fraction" in artifact:
        claimed_frac = float(artifact["val_trial_fraction"])
        if abs(claimed_frac - realized_val_fraction) > 1e-9:
            raise ValueError(
                f"Artifact val_trial_fraction={claimed_frac} disagrees with "
                f"the persisted split's realized fraction {realized_val_fraction}; "
                "fail closed on unverifiable split metadata."
            )
    if "n_train" in artifact and int(artifact["n_train"]) != n_train:
        raise ValueError(
            f"Artifact n_train={artifact['n_train']!r} disagrees with the "
            f"persisted train split size {n_train}; fail closed."
        )
    if "n_val" in artifact and int(artifact["n_val"]) != n_val:
        raise ValueError(
            f"Artifact n_val={artifact['n_val']!r} disagrees with the "
            f"persisted val split size {n_val}; fail closed."
        )
    # Legacy animal-named counts are aliases for recording-prefix counts.
    n_train_animals = len(train_prefixes)
    n_val_animals = len(val_prefixes)

    info: Dict[str, Any] = {
        "nested_prior_artifact": str(artifact_path),
        "nested_prior_artifact_sha256": artifact_sha256,
        "nested_prior_fingerprint": str(recorded),
        "mcmc_prior_provenance": artifact["mcmc_prior_provenance"],
        "animal_identity_status": identity_status,
        "train_recording_prefixes": train_prefixes,
        "val_recording_prefixes": val_prefixes,
        "n_train_recording_prefixes": n_train_animals,
        "n_val_recording_prefixes": n_val_animals,
        "is_nested_cv": True,
        "split_seed": split_seed,
        "n_inner_folds": n_inner_folds,
        "val_split": val_split,
        "val_trial_fraction": realized_val_fraction,
        "n_train": n_train,
        "n_val": n_val,
        "n_train_animals": n_train_animals,
        "n_val_animals": n_val_animals,
    }
    logger.info(
        "Nested-prior artifact accepted: %d train / %d val trials "
        "(realized %.1f%% trials; requested recording-prefix val_split=%.2f; "
        "split_seed=%d), fingerprint=%.12s…, provenance=%s",
        n_train,
        n_val,
        100.0 * realized_val_fraction,
        val_split,
        split_seed,
        recorded,
        info["mcmc_prior_provenance"],
    )
    logger.info(
        "Recording prefixes: %d train, %d val (legacy n_train_animals/n_val_animals; "
        "animal identities across prefixes unverified).",
        n_train_animals, n_val_animals,
    )
    return train_idx, val_idx, nested_priors, info
