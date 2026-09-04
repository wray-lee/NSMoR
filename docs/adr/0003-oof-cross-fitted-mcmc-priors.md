# Out-of-Fold Cross-Fitted MCMC Priors with Animal-Grouped Stratification

MCMC priors are generated via K-fold cross-fitting with StratifiedGroupKFold using animal-level groups (stripping `_session_N` suffixes) instead of individual recording sessions or unpartitioned data. Each trial's prior probability vector is produced by a fold model that never saw that trial's label or any trial from the same animal.

Training the prior generator on all data and then using its predictions as features for the same data creates a direct label leakage path: the MCMC prior encodes the ground-truth label, and the downstream model learns to read it rather than the sensory features. While session-level grouping was initially adopted to prevent within-session leakage, animals recorded across multiple sessions still leaked individual baseline and gain state across folds. Animal-level grouping strictly eliminates this cross-session animal leakage.

## Considered Options

- **Full-data training**: simpler, but creates same-sample label leakage. The downstream model learns to decode the prior rather than the sensory input.
- **Sample-level cross-fitting**: prevents same-sample leakage but allows session-level and animal-level information sharing.
- **Session-grouped cross-fitting** (historical): prevents same-session leakage but permits multi-session animal identity leakage.
- **Animal-grouped cross-fitting** (chosen / current): prevents sample, session, and animal-level leakage (`oof_5fold_animal_grouped_cv`). Requires enough distinct animals per behavioral class to populate every fold.

## Consequences

Requires a minimum number of distinct animals per behavioral class. Datasets with very few animals may fail to populate all folds, in which case `resolve_group_folds()` dynamically adapts fold count to animal coverage. Provenance validation strictly enforces `mcmc_prior_provenance="oof_5fold_animal_grouped_cv"`.
