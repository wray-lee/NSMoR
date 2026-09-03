# Responder-First Label Branch Order (v2.1)

The behavioral labeling classifier tests the post-stimulus escape response FIRST, then splits responders by pre-stimulus locomotion (PREWALK vs ESCAPE), reserving PRE_ACTIVE for non-responders. This replaced the pre-active-first order from v2.0, which structurally absorbed every walking animal into PRE_ACTIVE and collapsed the PREWALK class to zero trials.

The v2.0 order checked pre-stimulus activity first: any animal walking before stimulus onset was classified PRE_ACTIVE regardless of its post-stimulus escape. This eliminated the PREWALK class entirely (PREWALK=0 across all datasets), which is biologically incorrect -- cricket GI-mediated escape is stimulus-locked and occurs whether or not the animal was already walking. The reordering produces the expected four-class distribution (ESCAPE=204, NO_RESPONSE=129, PRE_ACTIVE=52, PREWALK=11) but invalidates all v2.0 datasets and checkpoints, since every trial's label can change.

## Considered Options

- **Pre-active-first** (v2.0): check baseline locomotion first. Simpler logic but biologically wrong -- PREWALK class collapses to zero.
- **Responder-first** (v2.1, chosen): check post-stimulus escape first. Correct behavioral semantics at the cost of full pipeline re-run.

## Consequences

All v2.0 datasets and checkpoints are scientifically invalid for v2.1 code. The Pipeline Semantics Version stamp (bumped to 2.1) causes loaders to reject pre-2.1 artifacts, preventing silent misinterpretation.
