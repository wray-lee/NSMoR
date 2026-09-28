"""NSMoR Lazy-Loading Dataset (ELT Mode)

Loads trial sequences on-demand from raw CSV files instead of
pre-loading everything.  Dramatically reduces memory footprint
for large datasets.

The feature layout EXACTLY matches the ETL-mode
:class:`nsmor.nsmor_dataloader.NSMoRDataset`:

    [0] v_vis(t)        — visual angle
    [1] wind(t)         — wind state (0/1)
    [2] v_kine(t-1)     — previous-frame velocity
    [3] a_kine(t-1)     — previous-frame acceleration
    [4] P_startle       ┐
    [5] P_walk          │ MCMC prior (static per trial,
    [6] P_pre_active    │ tiled across all frames)
    [7] P_no_response   ┘

    Y_t = continuous velocity at time t.
"""
from __future__ import annotations

import hashlib
import re
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from nsmor.config import DEFAULT_FEATURE, FeatureConfig
from nsmor.data_extractor import (
    PURE_WIND_PREPEND_FRAMES,
    _compute_pure_wind_prepend_frames,
    _is_pure_wind,
)
from nsmor.pipeline.io import load_kinematics_csv, load_events_csv, extract_trial_data
from nsmor.pipeline.nested_prior import load_artifact_bytes


def csv_sha256(path: Path, snapshot=None) -> str:
    """Stream the raw bytes, optionally copying exactly those bytes for parsing."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            if snapshot is not None:
                snapshot.write(block)
    return digest.hexdigest()


def source_paths(source: Dict) -> Tuple[Path, Path]:
    session_dir = Path(source["session_dir"].replace("\\", "/"))
    return (
        (session_dir / source["kinematics_file"]).resolve(),
        (session_dir / source["events_file"]).resolve(),
    )


def source_digests(source: Dict) -> Tuple[str, str]:
    """Legacy metadata can be inspected, but cannot safely replay unbound CSVs."""
    digests = (source.get("kinematics_sha256"), source.get("events_sha256"))
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in digests):
        raise ValueError("missing or invalid SHA-256 for source CSVs; regenerate metadata from raw CSVs")
    return digests


def verify_source_pair(source: Dict) -> None:
    for path, expected in zip(source_paths(source), source_digests(source)):
        if csv_sha256(path) != expected:
            raise ValueError(f"Source CSV changed: {path}; regenerate metadata from raw CSVs")


def load_csv_snapshot(path: Path, loader, expected: Optional[str] = None):
    """Parse the same bounded-memory byte snapshot whose SHA-256 is recorded."""
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b") as snapshot:
        digest = csv_sha256(path, snapshot)
        if expected is not None and digest != expected:
            raise ValueError(f"Source CSV changed: {path}; regenerate metadata from raw CSVs")
        snapshot.seek(0)
        table = loader(snapshot)
    if csv_sha256(path) != digest:
        raise ValueError(f"Source CSV changed: {path}; regenerate metadata from raw CSVs")
    return table, digest

class NSMoRLazyDataset(Dataset):
    """Lazy-loading dataset for NSMoR training (ELT mode).

    Loads trial sequences on-demand from raw CSV files and assembles
    the 8-D input tensor using the SAME feature engineering as
    :func:`nsmor.data_extractor.extract_trial_sequence`.

    Memory footprint is ~constant regardless of dataset size.

    Args:
        metadata_path: Path to metadata file (from prepare_metadata.py)
        max_seq_len: Maximum sequence length (longer sequences are cropped)
        feature_config: Feature dimension config
        dt_ms: Frame interval in milliseconds (if None, inferred from timestamps)
        metadata: Already restricted-decoded snapshot, when supplied by the converter
    """

    def __init__(
        self,
        metadata_path: str,
        max_seq_len: int = 2400,
        pre_anchor_frames: int = 1200,
        feature_config: FeatureConfig = DEFAULT_FEATURE,
        dt_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if metadata is None:
            metadata = load_artifact_bytes(Path(metadata_path).read_bytes())

        from nsmor.model_utils import validate_dataset_provenance
        validate_dataset_provenance(metadata, Path(metadata_path))
        self.trial_specs: List[Dict] = [dict(spec) for spec in metadata["trial_specs"]]
        self.mcmc_priors: torch.Tensor = metadata["mcmc_priors"]  # (N, 4)
        self.max_seq_len = max_seq_len
        self.pre_anchor_frames = pre_anchor_frames
        self.feature_config = feature_config
        self.dt_ms = dt_ms

        # Validate shapes
        assert self.mcmc_priors.shape[0] == len(self.trial_specs), (
            f"MCMC priors count {self.mcmc_priors.shape[0]} != "
            f"trial_specs count {len(self.trial_specs)}"
        )
        assert self.mcmc_priors.shape[1] == feature_config.mcmc_dim, (
            f"MCMC dim {self.mcmc_priors.shape[1]} != {feature_config.mcmc_dim}"
        )

        # LRU-like cache for session-level data (avoid re-reading CSVs)
        self._session_cache: Dict[Tuple[Path, Path, str, str], Dict[str, pd.DataFrame]] = {}
        self._cache_keys: List[Tuple[Path, Path, str, str]] = []  # insertion order for LRU eviction
        self._cache_size = 10
        self._source_snapshots = None

    @contextmanager
    def captured_sources(self):
        """Bind a conversion to private, pathless copies of all raw CSV bytes."""
        with ExitStack() as stack:
            snapshots = {}
            revision = []
            seen = set()
            for spec in self.trial_specs:
                for source in spec.get("source_pairs", [spec]):
                    paths = source_paths(source)
                    digests = source_digests(source)
                    if (*paths, *digests) in seen:
                        continue
                    for path, expected in zip(paths, digests):
                        if path in snapshots:
                            if snapshots[path][1] != expected:
                                raise ValueError(f"Conflicting SHA-256 for source CSV: {path}")
                            continue
                        snapshot = stack.enter_context(
                            tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")
                        )
                        if csv_sha256(path, snapshot) != expected or csv_sha256(path) != expected:
                            raise ValueError(f"Source CSV changed: {path}; regenerate metadata from raw CSVs")
                        snapshots[path] = (snapshot, expected)
                    seen.add((*paths, *digests))
                    revision.append({key: source[key] for key in (
                        "session_dir", "kinematics_file", "events_file",
                        "kinematics_sha256", "events_sha256",
                    )})
            self._session_cache.clear()
            self._cache_keys.clear()
            self._source_snapshots = snapshots
            try:
                yield revision
            finally:
                self._source_snapshots = None
                self._session_cache.clear()
                self._cache_keys.clear()

    def __len__(self) -> int:
        return len(self.trial_specs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Load and transform a single trial sequence.

        Applies anchor-aligned cropping to guarantee stimulus + response
        are within the returned window.

        Returns:
            X_seq: (T, 8) input features — same layout as ETL mode
            Y_seq: (T,) target velocities
            length: actual sequence length before any padding
        """
        spec = self.trial_specs[idx]

        # 1. Load trial data through the canonical pipeline.io path
        session_data = self._load_session(spec)
        trial_data = extract_trial_data(
            session_data,
            session_id=spec["session_id"],
            trial_id=spec["trial_id"],
        )

        # 2. Build 8-D feature tensor (same logic as extract_trial_sequence)
        X_seq, Y_seq = self._build_sequence(trial_data, idx)
        if "source_pairs" in spec and X_seq.shape[0] != spec["n_frames"]:
            raise ValueError(
                f"Trial {spec['session_id']!r}/{spec['trial_id']!r} has "
                f"{X_seq.shape[0]} reconstructed frames, expected {spec['n_frames']}"
            )

        # 3. Anchor-aligned crop (preserves stimulus + response).
        # Single source of truth: nsmor.pipeline.conditions.resolve_anchor_crop
        anchor_frame = spec["anchor_frame"]
        actual_length = X_seq.shape[0]

        if actual_length > self.max_seq_len:
            from nsmor.pipeline.conditions import resolve_anchor_crop

            start, end = resolve_anchor_crop(
                n_frames=actual_length,
                anchor_frame=anchor_frame,
                max_seq_len=self.max_seq_len,
                pre_anchor_frames=self.pre_anchor_frames,
            )
            X_seq = X_seq[start:end]
            Y_seq = Y_seq[start:end]
            actual_length = X_seq.shape[0]

        return (
            torch.from_numpy(X_seq).float(),
            torch.from_numpy(Y_seq).float(),
            actual_length,
        )

    def _build_sequence(
        self,
        trial_data: Dict[str, np.ndarray],
        idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Replicate extract_trial_sequence + MCMC prior injection.

        Matches :func:`nsmor.data_extractor.extract_trial_sequence` exactly,
        then fills the MCMC prior columns from the pre-computed OOF priors.
        """
        visual_angle = trial_data["visual_angle"]
        wind_state = trial_data["wind_state"]
        velocity = trial_data["velocity"]
        acceleration = trial_data["acceleration"]
        n_frames = len(trial_data["time_ms"])
        fc = self.feature_config

        # ── Physical features (n_frames, 4) ──
        physical = np.zeros(
            (n_frames, fc.per_frame_physical_dim), dtype=np.float64,
        )
        physical[:, 0] = visual_angle      # v_vis(t)
        physical[:, 1] = wind_state        # wind(t)
        # v_kine(t-1) and a_kine(t-1): shift by one frame
        physical[1:, 2] = velocity[:-1]
        physical[1:, 3] = acceleration[:-1]
        # Frame 0 has no predecessor → already zero

        # ── Pure-wind baseline alignment ──
        spec = self.trial_specs[idx]
        is_pw = spec.get("is_pure_wind")
        if is_pw is None:
            wind_state = np.asarray(trial_data["wind_state"])
            has_wind = bool(np.any(wind_state > 0.5))
            is_pw = _is_pure_wind(visual_angle) and has_wind
        else:
            is_pw = bool(is_pw)

        if is_pw:
            # Prefer an explicitly recorded prepend (set by prepare_metadata
            # or the convert_metadata_to_etl legacy-migration pass) so the
            # crop window and the prepend always agree on the same count.
            spec = self.trial_specs[idx]
            recorded_prepend = int(spec.get("pure_wind_prepended_frames", 0) or 0)
            if recorded_prepend > 0:
                prepend_frames = recorded_prepend
            else:
                if self.dt_ms is not None:
                    dt_ms_eff = self.dt_ms
                elif n_frames > 1:
                    dt_ms_eff = float(np.median(np.diff(trial_data["time_ms"])))
                else:
                    raise ValueError(
                        "Pure-wind trial with a single frame and unknown dt_ms: "
                        "cannot compute 5.7 s baseline prepend frames. Provide an "
                        "explicit dt_ms (CLI / metadata / constructor) rather than "
                        "fabricating a frame interval."
                    )
                prepend_frames = _compute_pure_wind_prepend_frames(dt_ms_eff)
            prepend_zeros = np.zeros(
                (prepend_frames, fc.per_frame_physical_dim),
                dtype=np.float64,
            )
            physical = np.concatenate([prepend_zeros, physical], axis=0)
            target_zeros = np.zeros(prepend_frames, dtype=np.float64)
            Y_seq = np.concatenate([target_zeros, velocity.copy()], axis=0)
        else:
            Y_seq = velocity.copy()

        # ── MCMC prior (tiled across all frames) ──
        total_frames = physical.shape[0]
        mcmc_prior = self.mcmc_priors[idx].numpy()  # (4,)
        mcmc_tiled = np.tile(mcmc_prior, (total_frames, 1))  # (T, 4)

        # ── Concatenate → (T, 8) ──
        X_seq = np.concatenate([physical, mcmc_tiled], axis=1)

        # ── Shape assertions ──
        assert X_seq.shape == (total_frames, fc.per_frame_total_dim), (
            f"X_seq shape: expected ({total_frames}, "
            f"{fc.per_frame_total_dim}), got {X_seq.shape}"
        )
        assert Y_seq.shape == (total_frames,), (
            f"Y_seq shape: expected ({total_frames},), got {Y_seq.shape}"
        )
        return X_seq, Y_seq

    def _load_session(
        self,
        spec: Dict,
    ) -> Dict[str, pd.DataFrame]:
        """Replay the producer's contributing CSV pairs in their original order."""
        sources = spec["source_pairs"] if "source_pairs" in spec else [spec]
        if not sources:
            raise ValueError("Trial metadata has no source pairs")

        kin_parts = []
        evt_parts = []
        for source in sources:
            kin_path, evt_path = source_paths(source)
            kin_digest, evt_digest = source_digests(source)
            cache_key = (kin_path, evt_path, kin_digest, evt_digest)
            if cache_key not in self._session_cache:
                if self._source_snapshots is None:
                    pair = {
                        "kinematics": load_csv_snapshot(kin_path, load_kinematics_csv, kin_digest)[0],
                        "events": load_csv_snapshot(evt_path, load_events_csv, evt_digest)[0],
                    }
                    verify_source_pair(source)
                else:
                    kin_snapshot = self._source_snapshots[kin_path][0]
                    evt_snapshot = self._source_snapshots[evt_path][0]
                    kin_snapshot.seek(0)
                    evt_snapshot.seek(0)
                    pair = {
                        "kinematics": load_kinematics_csv(kin_snapshot),
                        "events": load_events_csv(evt_snapshot),
                    }
                if len(self._session_cache) >= self._cache_size:
                    evict_key = self._cache_keys.pop(0)
                    self._session_cache.pop(evict_key)
                self._session_cache[cache_key] = pair
                self._cache_keys.append(cache_key)
            else:
                # The converter reuses its private captured revision; training checks live paths.
                if self._source_snapshots is None:
                    verify_source_pair(source)
                self._cache_keys.remove(cache_key)
                self._cache_keys.append(cache_key)
                pair = self._session_cache[cache_key]
            kin_parts.append(pair["kinematics"])
            evt_parts.append(pair["events"])

        if len(sources) == 1:
            return pair
        return {
            "kinematics": pd.concat(kin_parts, ignore_index=True),
            "events": pd.concat(evt_parts, ignore_index=True),
        }

    def verify_sources(self) -> None:
        """Recheck every contributing pair before an ETL artifact is published."""
        seen = set()
        for spec in self.trial_specs:
            for source in spec.get("source_pairs", [spec]):
                key = (*source_paths(source), *source_digests(source))
                if key not in seen:
                    verify_source_pair(source)
                    seen.add(key)

    def get_label(self, idx: int) -> str:
        """Get behavioral label for a trial."""
        return self.trial_specs[idx]["label"]

    def get_session_id(self, idx: int) -> str:
        """Get session ID for a trial."""
        return self.trial_specs[idx]["session_id"]
