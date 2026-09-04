# MCMC Prior Cross-Fitting with Animal-Grouped Folds

MCMC priors are generated via K-fold cross-fitting where folds are partitioned
by animal identity (stripping the `_session_N` suffix from session IDs) rather
than by individual recording session. This prevents an animal's `_session_1`
from training the prior generator evaluated on `_session_2`, eliminating cross-session
information leakage.

## Status of Corpora

| Corpus | Status | Provenance | Trials / Animals | Notes |
|---|---|---|---|---|
| `nsmor_dataset_full_backup.pt` | Regenerated | `oof_4fold_animal_grouped_cv` | 396 trials / 11 animals | Baseline 1-condition corpus |
| `nsmor_subset_routing_calibration.pt` | Regenerated | `oof_4fold_animal_grouped_cv` | Derived from full_backup | Routing calibration subset |
| `nsmor_dataset_3cond_v2.pt` | Regenerated | `oof_5fold_animal_grouped_cv` | 1332 trials / 37 animals | 3-condition corpus (`data/raw_3cond_adapted/`) |
| `nsmor_subset_small.pt` | Pending | `MISSING` (None) | 288 trials | Derived from pre-fix session-grouped corpus; symlinked by `nsmor_dataset.pt` |
| `backup_before_animal_grouped/nsmor_dataset_3cond_v2.pt` | Preserved | `MISSING` (session-grouped) | 1440 trials / 40 animals | Safety net backup from `staging_3cond_1440` |

Overall progress: 3 of 4 primary corpora are regenerated with animal-grouped priors.
Only `nsmor_subset_small.pt` remains pending regeneration.

## Critical Distinction: 3cond_v2 vs Backup

The regenerated `nsmor_dataset_3cond_v2.pt` and the preserved backup in
`backup_before_animal_grouped/` represent **different data** and must never be
directly compared as identical datasets:

1. **Source cohort delta (Audit 2):**
   - The backup (1440 trials / 40 animals) was built from `data/staging_3cond_1440`.
   - The new `3cond_v2` (1332 trials / 37 animals) was built from `data/raw_3cond_adapted/`,
     which staged only 37 numeric-prefixed animals.
   - The 3 excluded animals exhibited zero escapes (58 No-Response, 48 Pre-Active,
     2 Prewalk). Escape counts (label 0) are identical at 412 across both datasets;
     per-trial labels across all 74 shared sessions match 100%.
2. **Representation & format delta (Audit 3):**
   - Backup: 2400-frame anchor-aligned sequences in `float32` (128.2 MB).
   - New `3cond_v2`: Uncapped raw continuous recordings up to 121,918 frames
     (median 18,002 frames) in `float64` (1900.8 MB).

## Pipeline & Scientific Caveats

Findings from recent audits that downstream analysis and training must account for:

- **Uncapped sequence length & random crop hazard (Audit 3 & 5):**
  `3cond_v2.pt` stores full continuous recordings without anchor-aligned cropping.
  Passing `max_seq_len=2400` to `NSMoRDataset` without `anchor_frames` causes random
  cropping that misses stimulus onset in 88%-95% of crops. Uncropped batch training
  risks CUDA OOM (>40 GB for batch size 128). `float64` storage is redundant
  because dataloaders immediately cast to `float32`.
- **Sampling interval mismatch (Audit 1):**
  Hardware acquisition median is 4.01 ms (nominal 250 Hz), but ETL ran with
  `--dt_ms 10.0`. In pure-wind trials, `prepend_frames` was set to 570 frames
  (2.28 s) instead of ~1425 frames (5.7 s), creating an 851-frame (3.41 s) onset
  misalignment relative to multisensory trials. Discrete dynamical parameters
  configured for 10 ms scale by 2.49x in real time.
- **MCMC prior train-serve shift (Audit 5):**
  The cross-fitted MCMC prior exhibits a distribution shift between out-of-fold
  training and full-fit serving (mean total variation distance 0.2739, argmax
  agreement 0.676). This shift is stored in `mcmc_prior_train_serve_consistency`
  and must be reported alongside any evaluation utilizing the MCMC prior channel.
- **Provenance validation (Audit 5):**
  `validate_dataset_provenance` checks `pipeline_semantics_version` (2.2) but not
  `mcmc_prior_provenance`. `nsmor_subset_small.pt` currently has `mcmc_prior_provenance: None`
  and must be regenerated before use in leak-free evaluation.

## Verification

Check corpus provenance and group fold resolution:

```python
import torch

data = torch.load("data/processed/nsmor_dataset_3cond_v2.pt", weights_only=False)
print("Provenance:", data.get("mcmc_prior_provenance"))
# Output: "oof_5fold_animal_grouped_cv"

print("Train/serve shift:", data.get("mcmc_prior_train_serve_consistency"))
# Output: {'argmax_agreement': 0.676, 'mean_tv_distance': 0.2739, ...}
```

Fold count `N` adapts dynamically: when a rare class occupies fewer than 5 animals,
`resolve_group_folds` caps `N` to the minimum animal coverage for that class
(for `3cond_v2`, label 1 covers 17 distinct animals, safely resolving to 5 folds).

## Regeneration Commands

To regenerate `nsmor_subset_small.pt` from the animal-grouped `3cond_v2` corpus:

```bash
python scripts/make_subset_dataset.py \
    --input data/processed/nsmor_dataset_3cond_v2.pt \
    --output data/processed/nsmor_subset_small.pt \
    --n_animals 8 \
    --seed 42
```

To re-run the full 3cond_v2 ETL:

```bash
python scripts/prepare_data.py \
    --raw_dir data/raw_3cond_adapted \
    --output data/processed/nsmor_dataset_3cond_v2.pt \
    --dt_ms 4.0 \
    --seed 42
```
