# MCMC prior cross-fitting now uses animal-grouped folds

Commit `6b88e46` fixed the MCMC prior leakage: out-of-fold priors are now
cross-fitted with groups derived by **animal** (stripping `_session_N`),
not by session. This prevents an animal's `_session_1` from training the
generator that produces `_session_2`'s supposedly held-out prior.

## Status of existing corpora

| Corpus | Status | Provenance |
|--------|--------|------------|
| `nsmor_dataset_full_backup.pt` | ✅ **Regenerated** | `oof_4fold_animal_grouped_cv` |
| `nsmor_subset_routing_calibration.pt` | ✅ **Regenerated** | `oof_4fold_animal_grouped_cv` |
| `nsmor_dataset_3cond_v2.pt` | ⏳ **Pending** | Still `MISSING` (old version) |
| `nsmor_subset_small.pt` | ⏳ **Pending** | Still `MISSING` (derived from 3cond_v2) |

**Note:** `3cond_v2` regeneration attempted but ETL was interrupted (exit
code 15) during the final write after MCMC priors were successfully
generated (5-fold animal-grouped, 37 animals, 1332 trials). The raw data
was successfully adapted via `pre_load_adapt.py` and is staged in
`data/raw_3cond_adapted/`. Re-running the full ETL on this 1440-trial
corpus takes approximately 15 minutes on this machine.

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
