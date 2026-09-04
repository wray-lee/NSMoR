# MCMC prior cross-fitting now uses animal-grouped folds

Commit `6b88e46` fixed the MCMC prior leakage: out-of-fold priors are now
cross-fitted with groups derived by **animal** (stripping `_session_N`),
not by session. This prevents an animal's `_session_1` from training the
generator that produces `_session_2`'s supposedly held-out prior.

## Impact on existing corpora

**All `.pt` files generated before this fix still carry session-grouped
priors in input channels 4-7.** The fix is ETL-time only — those priors
are baked into the tensors during `scripts/prepare_data.py`, so no
deployment-time patch can repair them retroactively.

## Regeneration requirement

The following corpora must be regenerated via `prepare_data.py` to receive
animal-grouped priors:

- `data/processed/nsmor_dataset.pt`
- `data/processed/nsmor_dataset_3cond_v2.pt`
- `data/processed/nsmor_dataset_full_backup.pt`
- `data/processed/nsmor_subset_routing_calibration.pt`
- Any other `.pt` files whose `mcmc_prior_provenance` field reads
  `"oof_5fold_session_grouped_cv"` instead of `"oof_Nfold_animal_grouped_cv"`.

The full dataset regeneration is scheduled as part of the next data
collection window (roughly 1 month). Until then, validation metrics on the
existing corpora carry this leakage channel.

## Verification

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
