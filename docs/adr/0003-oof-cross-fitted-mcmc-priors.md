# Out-of-Fold MCMC Priors with Recording-Prefix Grouping

Priors use `StratifiedGroupKFold` on session IDs after stripping `_session_N`.
A held-out trial and its recording prefix are excluded from its prior fit.
Different prefixes may still identify the same animal. Animal identity and
independence, absence of cross-recording leakage, and animal-level
generalization are unverified.

## Considered Options

- Full-data fitting uses the held-out trial's label when predicting its prior.
- Trial-only folds omit the held-out trial but may share its recording prefix.
- Recording-prefix folds (used here) separate prefixes, with unknown animal overlap.
- Verified-animal folds would require an auditable identity per trial and
  identity-based splitting before any animal-independent claim.

Historical artifacts labeled `oof_5fold_animal_grouped_cv` used prefix groups;
the legacy string does not certify animal identity. `resolve_group_folds()`
adapts to distinct prefixes per class, not distinct animals. Runtime
provenance labels and validation are handled separately from this correction.
