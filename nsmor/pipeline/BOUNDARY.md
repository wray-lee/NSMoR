# BOUNDARY — `nsmor/pipeline/` (Data Ingestion & Feature Engineering)

## Status: 🔓 SAFE TO EXTEND & MAINTAIN

This directory handles **data ingestion, feature extraction, labeling, and collation**. Modules here convert raw sensory logs into standardized PyTorch tensors for the frozen core model.

---

## Key Modules

- `io.py`: Raw dataset loading, file I/O, and raw tensor parsing.
- `kinematics.py`: Kinematic calculations (velocities, accelerations, visual angles, turn rates).
- `labeling.py`: MCMC prior estimation and categorical behavioral state annotations.
- `dataloader_factory.py` (and `nsmor_dataloader.py`): Parallel preprocessing, batch collation, variable-length padding, and dataset fixtures.

---

## Input/Output Contract

All feature extractors and dataloaders must produce feature tensors strictly conforming to the 8-channel input format expected by `NSMoRCore`:

```
Input Feature Layout [B, T, 8]:
  [0] v_vis(t)        — visual angle (degrees)
  [1] wind(t)         — wind stimulus state (0/1)
  [2] v_kine(t-1)     — previous velocity (cm/s)
  [3] a_kine(t-1)     — previous acceleration (cm/s²)
  [4] P_startle       — MCMC startle prior
  [5] P_walk          — MCMC walk prior
  [6] P_pre_active    — MCMC pre-active prior
  [7] P_no_response   — MCMC no-response prior
```

---

## Engineering Requirements & Modification Rules

1. **Shape Assertions**: Output batches must pass `assert X_batch.shape == (B, T, 8)` and `assert Y_batch.shape == (B, T)`.
2. **Deterministic Preprocessing**: Ensure random seeds (`rng_state`) are respected across parallel dataloader workers.
3. **Missing Value Protection**: Replace any missing data or zero-division outputs with clean defaults using `np.nan_to_num` / `torch.nan_to_num`.
4. **Backward Compatibility**: Do not modify public function signatures used by `scripts/train.py` or `nsmor/dataloader_factory.py`.
5. **Statistical Integrity**: Preserve temporal sequence order per trial during variable-length batch packing.
