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

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """

    def __init__(
        self,
        metadata_path: str,
        max_seq_len: int = 2400,
        pre_anchor_frames: int = 1200,
        feature_config: FeatureConfig = DEFAULT_FEATURE,
        dt_ms: Optional[float] = None,
    ) -> None:
        metadata = torch.load(metadata_path, weights_only=False)

        self.trial_specs: List[Dict] = metadata["trial_specs"]
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
        self._session_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._cache_keys: List[str] = []  # insertion order for LRU eviction
        self._cache_size = 10

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

        # 3. Anchor-aligned crop (preserves stimulus + response)
        anchor_frame = spec["anchor_frame"]
        actual_length = X_seq.shape[0]

        if actual_length > self.max_seq_len:
            # Crop centered on anchor: [anchor - pre, anchor + post)
            start = max(0, anchor_frame - self.pre_anchor_frames)
            end = min(actual_length, start + self.max_seq_len)

            # Adjust start if end clamped (keeps window size consistent)
            if end - start < self.max_seq_len:
                start = max(0, end - self.max_seq_len)

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
        if _is_pure_wind(visual_angle):
            if self.dt_ms is not None:
                dt_ms_eff = self.dt_ms
            elif n_frames > 1:
                dt_ms_eff = float(np.median(np.diff(trial_data["time_ms"])))
            else:
                dt_ms_eff = 10.0
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
        """Load session data with LRU caching.

        Uses :func:`pipeline.io.load_kinematics_csv` and
        :func:`pipeline.io.load_events_csv` for consistent column
        validation and artifact sanitization.
        """
        cache_key = spec["kinematics_file"]

        if cache_key not in self._session_cache:
            # Convert Windows paths to Unix (for WSL compatibility)
            session_dir_str = spec["session_dir"].replace("\\", "/")
            session_dir = Path(session_dir_str)
            kin_path = session_dir / spec["kinematics_file"]
            evt_path = session_dir / spec["events_file"]

            kin_df = load_kinematics_csv(kin_path)
            evt_df = load_events_csv(evt_path)

            # LRU eviction
            if len(self._session_cache) >= self._cache_size:
                evict_key = self._cache_keys.pop(0)
                self._session_cache.pop(evict_key, None)

            self._session_cache[cache_key] = {
                "kinematics": kin_df,
                "events": evt_df,
            }
            self._cache_keys.append(cache_key)
        else:
            # Move to end (most recently used)
            if cache_key in self._cache_keys:
                self._cache_keys.remove(cache_key)
                self._cache_keys.append(cache_key)

        return self._session_cache[cache_key]

    def get_label(self, idx: int) -> str:
        """Get behavioral label for a trial."""
        return self.trial_specs[idx]["label"]

    def get_session_id(self, idx: int) -> str:
        """Get session ID for a trial."""
        return self.trial_specs[idx]["session_id"]
