# Clock Recovery R4: Method and Validation Status

**Status date:** 2026-09-29

## Method

The experimental path is opt-in through `--experimental-clock-residual-ms`; a suspect-prefix manifest may be supplied with `--experimental-clock-prefix-manifest`. Experimental mapping requires staged `--output_dir` output and rejects in-place execution. No operational tolerance has been selected for the corpus.

Each trial fits centered OLS between hardware tick tokens and host arrival time. The leave-one-out residual is an operational rejection criterion, not a bound on acquisition synchronization or stimulus latency. Event timestamps remain host timestamps; subtracting a common trial origin does not independently prove that the event and frame streams share a physical clock.

The external manifest binds complete kinematics and event file hashes, source row indices, and raw time tokens. Declared leading prefixes use median mapped suffix cadence for backward extrapolation. Those prefix values are model estimates, never observed measurements. There is no automatic prefix expansion, reset stitching, epsilon repair, or trial deletion.

Provenance is retained at source-CSV-row scope; it does not assert one-to-one correspondence with resampled tensor frames. The strict canonical duplicate/nonfinite-time guard remains. Firmware's inspected nominal cadence (about 5 ms) is distinct from the model grid (4 ms), and the historically flashed firmware is not verified.

## Gating refinement

Condition handling distinguishes `wind_only`, `visual_only`, `multisensory`, and `no_stimulus`. The descriptive contrast is wind-only versus visual-present (`visual_only` plus `multisensory`); condition counts remain separate and `no_stimulus` is not included in that contrast. Unknown or conflicting metadata is excluded with a reason. Cohen's *d* uses pooled sample variance; insufficient sample size or zero variance is unavailable, not a false zero. Recording prefixes are not verified independent animals because animal identity is unavailable.

## Verified evidence

- Full candidate regression: 1,248 passed, 1 skipped, 11 warnings, exit 0; candidate and frozen-core hashes matched before and after.
- The skipped real-data alignment test requires unavailable `nsmor_subset_small.pt`.
- Stage-1 descriptive corpus check: 256 files (128 kinematics and 128 events), 2,432 trial pairs, 166 declared prefix trials / 167 rows, 2,431 fitted trials plus one singleton, and no structural failures before residual rejection. The zero-tolerance diagnostic exits before the production helper's final mapped-axis check for positive-residual fits; this count does not certify every reconstructed axis. Maximum leave-one-out residual summary: median 24.283855 ms, P95 35.396806 ms, maximum 108.340565 ms.

These residuals describe host-arrival association only. They do not establish scientific timing acceptance, biological response-window robustness, model convergence, independent-animal inference, or an untouched holdout.

## Pending release evidence

Prefix sensitivity, an evidence-based operational tolerance, staged ETL, nested-prior evaluation, training, downstream analyses, and scientific acceptance remain pending. The historical failed Stage-02 QC evidence is preserved unchanged. Missing experimental conditions and verified animal identity remain explicit limitations. No claim is made that this revision is already delivered to `main`.

Evidence paths (external scratch, not release artifacts):

- `.scratch/combined-validation-r4/verified-wsl-An7UkhYc/`
- `.scratch/clock-corpus-r4/verified-run-H0kiCAI6/`
