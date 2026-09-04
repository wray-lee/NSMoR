# MCMC prior cross-fitting now uses animal-grouped folds

Commit `6b88e46` fixed the MCMC prior leakage: out-of-fold priors are now
cross-fitted with groups derived by **animal** (stripping `_session_N`),
not by session. This prevents an animal's `_session_1` from training the
generator that produces `_session_2`'s supposedly held-out prior.

## Status of existing corpora

| Corpus | Status | Provenance | Notes |
|--------|--------|------------|-------|
| `nsmor_dataset_full_backup.pt` | ✅ **Regenerated** | `oof_4fold_animal_grouped_cv` | 396 trials, 11 animals |
| `nsmor_subset_routing_calibration.pt` | ✅ **Regenerated** | `oof_4fold_animal_grouped_cv` | Derived from full_backup |
| `nsmor_dataset_3cond_v2.pt` | ⏸ **Pending** | `MISSING` (session-grouped) | ETL ready; interrupted by 10min timeout |
| `nsmor_subset_small.pt` | ⏸ **Pending** | `MISSING` (session-grouped) | Derives from 3cond_v2 |

**Progress: 50% complete (2/4 corpora regenerated with animal-grouped priors)**

### About the pending corpora

`3cond_v2` ETL (1440 trials, ~15 minutes) was interrupted 5 times during
the final file write by system timeout/termination mechanisms. The MCMC
cross-fitting itself completes successfully (5-fold animal-grouped, 37
animals, 1332 trials), but writing the 123MB output file is always
terminated. The raw data has been successfully adapted via
`pre_load_adapt.py` and is staged in `data/raw_3cond_adapted/`.

**All code-level fixes are complete** — any new ETL run will correctly use
animal grouping. These two corpora can be regenerated when a suitable
environment is available (longer timeout, manual execution, or scheduled
batch job).

## Regeneration procedure

For corpora still carrying session-grouped priors:

```bash
# For 3cond_v2 (raw data already adapted and staged):
cd /path/to/NSMoR
conda activate torch
python scripts/prepare_data.py \
    --raw_dir data/raw_3cond_adapted \
    --output data/processed/nsmor_dataset_3cond_v2.pt \
    --seed 42

# Then regenerate subset_small from the new 3cond_v2:
python scripts/make_subset_dataset.py \
    --input data/processed/nsmor_dataset_3cond_v2.pt \
    --output data/processed/nsmor_subset_small.pt \
    --n_animals 8 \
    --seed 42
```

For other corpora, follow the same pattern:

Post-regeneration, check the provenance field:

```python
import torch
d = torch.load("data/processed/nsmor_dataset.pt", weights_only=False)
print(d["mcmc_prior_provenance"])
# Should print: "oof_Nfold_animal_grouped_cv" where N is the resolved fold count
```

The fold count N adapts to the corpus: when a rare class occupies fewer
than 5 animals, `resolve_group_folds` caps N at the minimum per-class
animal coverage so cross-fitting remains statistically sound.
