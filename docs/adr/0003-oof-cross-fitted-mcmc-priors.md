# Out-of-Fold Cross-Fitted MCMC Priors with Session-Grouped Stratification

MCMC priors are generated via K-fold cross-fitting with StratifiedGroupKFold (session-level groups) instead of training on all data. Each trial's prior probability vector is produced by a fold model that never saw that trial's label or any trial from its recording session.

Training the prior generator on all data and then using its predictions as features for the same data creates a direct label leakage path: the MCMC prior encodes the ground-truth label, and the downstream model learns to read it rather than the sensory features. Session-level grouping prevents a subtler leak where trials from the same session share baseline locomotor state, gain calibration, and animal identity.

## Considered Options

- **Full-data training**: simpler, but creates same-sample label leakage. The downstream model learns to decode the prior rather than the sensory input.
- **Sample-level cross-fitting**: prevents same-sample leakage but allows session-level information sharing.
- **Session-grouped cross-fitting** (chosen): prevents both sample-level and session-level leakage. Requires enough sessions per behavioral class to populate every fold.

## Consequences

Requires a minimum number of recording sessions per behavioral class. Datasets with very few sessions may fail to populate all folds, causing the cross-fitting to degrade or fail.
