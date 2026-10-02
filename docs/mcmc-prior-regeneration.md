# MCMC Prior Cross-Fitting with Recording-Prefix-Grouped Folds

MCMC priors use K-fold cross-fitting grouped by the recording prefix obtained by
stripping `_session_N` from a session ID. A held-out trial and its prefix stay
out of its prior fit. Different prefixes may belong to the same animal; animal
identity, independence, cross-recording leakage control, and animal-level
generalization remain unverified without an auditable identity mapping and
identity-based split.

## Historical corpus report (not SOURCE13 revalidation)

The counts below used "animals" for recording prefixes. The legacy provenance
strings record prefix grouping only; they do not establish animal identity.

| Corpus | Historical status | Legacy provenance | Trials / recording prefixes | Notes |
|---|---|---|---|---|
| `nsmor_dataset_full_backup.pt` | ✅ Regenerated | `oof_4fold_animal_grouped_cv` | 396 trials / 11 recording prefixes | Baseline 1-condition corpus |
| `nsmor_subset_routing_calibration.pt` | ✅ Regenerated | `oof_4fold_animal_grouped_cv` | Derived from full_backup | Routing calibration subset |
| `nsmor_dataset_3cond_v2.pt` | ✅ Regenerated | `oof_5fold_animal_grouped_cv` | 1332 trials / 37 recording prefixes | 3-condition corpus with dt_ms=4.0 + anchor_frames |
| `nsmor_subset_small.pt` | ✅ Regenerated | `oof_5fold_animal_grouped_cv` | 288 trials / 8 recording prefixes | Derived from new 3cond_v2; symlinked by `nsmor_dataset.pt` |

Historical progress report: **4/4 primary corpora regenerated with recording-prefix-grouped priors**; no SOURCE13 corpus or animal-identity verification is implied.

## Critical Distinction: 3cond_v2 vs Backup

The regenerated `nsmor_dataset_3cond_v2.pt` and the preserved backup in
`backup_before_animal_grouped/` represent **different data** and must never be
directly compared as identical datasets:

1. **Source cohort delta (Audit 2):**
   - The backup (1440 trials / 40 recording prefixes) was built from `data/staging_3cond_1440`.
   - The new `3cond_v2` (1332 trials / 37 recording prefixes) was built from `data/raw_3cond_adapted/`,
     which staged only 37 numeric recording prefixes.
   - The 3 excluded recording prefixes exhibited zero escapes (58 No-Response, 48 Pre-Active,
     2 Prewalk). Escape counts (label 0) are identical at 412 across both datasets;
     per-trial labels across all 74 shared sessions match 100%.
2. **Representation & format delta (Audit 3):**
   - Backup: 2400-frame anchor-aligned sequences in `float32` (128.2 MB).
   - New `3cond_v2`: Uncapped raw continuous recordings up to 121,918 frames
     (median 18,002 frames) in `float64` (1900.8 MB).

## Pipeline & Scientific Caveats (Historical Audit & Current Resolutions)

Findings from recent audits and their current resolution status:

- **[RESOLVED] Sampling interval mismatch (Audit 1):**
  *Historical issue*: The source cadence is approximately 5 ms (inspected firmware nominal, about 200 Hz; the flashed firmware and any emission calibration remain unverified), and the ~4.01 ms legacy figure was a host-arrival batching diagnostic rather than a measured acquisition rate. Legacy ETL previously ran with `--dt_ms 10.0`. In pure-wind trials, `prepend_frames` was set to 570 frames (2.28 s at 4 ms) instead of 1425 frames (5.7 s at 4 ms), creating an 851-frame onset misalignment relative to multisensory trials.
  *Resolution*: Fixed across CLI defaults and ETL functions (`dt_ms=4.0`). All corpora regenerated with 1425 prepended frames on the aligned 4.0 ms model grid. Switching the model grid from 10.0 ms to 4.0 ms preserves trial duration and does not inflate it by 25%.

- **[RESOLVED] Uncapped sequence length & random crop hazard (Audit 3 & 5):**
  *Historical issue*: Passing `max_seq_len=2400` to `NSMoRDataset` without anchor indices caused random cropping that missed stimulus onset in 88%-95% of crops.
  *Resolution*: Per-trial `anchor_frames` are now derived and persisted in the dataset artifacts; all training and downstream analysis scripts use anchor-aligned cropping (`max_seq_len=2400`, `pre_anchor_frames=1200`), guaranteeing 100% stimulus capture.

- **[HISTORICAL; ANIMAL IDENTITY UNVERIFIED] Provenance validation (Audit 5):**
  *Historical issue*: `validate_dataset_provenance` checked only `pipeline_semantics_version` (2.2), allowing un-grouped or missing prior provenance to pass silently.
  *Historical resolution*: the guard checked the legacy `oof_5fold_animal_grouped_cv` string. That string cannot verify animal identity or independence. Historical corpus regeneration is reported above; SOURCE13 runtime provenance is owned separately.

- **[ACTIVE TELEMETRY] MCMC prior train-serve shift (Audit 5):**
  The cross-fitted MCMC prior exhibits a distribution shift between out-of-fold training and full-fit serving (mean total variation distance 0.2739, argmax agreement 0.676). This shift is stored in `mcmc_prior_train_serve_consistency` and logged to checkpoints and metrics to ensure complete scientific auditability.

## Historical artifact inspection

For historical artifacts, inspect the recorded string without interpreting it as animal proof:

```python
import torch

data = torch.load("data/processed/nsmor_dataset_3cond_v2.pt", weights_only=False)
print("Provenance:", data.get("mcmc_prior_provenance"))
# Output: "oof_5fold_animal_grouped_cv"

print("Train/serve shift:", data.get("mcmc_prior_train_serve_consistency"))
# Output: {'argmax_agreement': 0.676, 'mean_tv_distance': 0.2739, ...}
```

Fold count `N` adapts to recording-prefix coverage by class; historically,
label 1 occupied 17 distinct prefixes in `3cond_v2`, allowing 5 folds.

## Regeneration Commands

Historical command to derive `nsmor_subset_small.pt` from the prefix-grouped `3cond_v2` corpus (`--n_animals` is a legacy CLI name for prefix count):

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
