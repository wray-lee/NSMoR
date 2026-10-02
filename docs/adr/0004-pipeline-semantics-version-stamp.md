# Pipeline Semantics Version Provenance Stamp

Every checkpoint and dataset carries a version string. The current value is defined by ``PIPELINE_SEMANTICS_VERSION`` in ``nsmor/config.py`` (currently ``'2.2'``); that constant is the single source of truth, so this ADR deliberately does not pin a literal. Loaders hard-reject artifacts with a missing or mismatched version, preventing silent scientific invalidation from mixing incompatible artifacts.

The pipeline has undergone breaking semantic changes: v2.0 switched LIF time constants from frame units to physical milliseconds and introduced session-grouped MCMC cross-fitting; v2.1 changed the behavioral labeling branch order. Loading a pre-2.0 checkpoint under current code would silently run a completely different biophysical system. Loading a pre-2.1 dataset would use scientifically incorrect labels. The version stamp is the sole programmatic barrier against both failure modes.

## Considered Options

- **No version guard**: simpler, but any pre-existing artifact silently produces wrong results with no error or warning.
- **Warning-only guard**: alerts the user but allows the run to proceed, risking unnoticed scientific invalidation.
- **Hard-reject guard** (chosen): refuses to load mismatched artifacts. Forces full regeneration when semantics change.

## Consequences

Any semantic change to the pipeline requires bumping the version and regenerating all artifacts. There is no migration path -- artifacts must be reproduced from raw data under the current code.
