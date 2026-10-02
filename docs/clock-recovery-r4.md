# Clock Recovery R4: Method and Validation Status

**Status date:** 2026-09-29

## Method

The experimental path is opt-in through `--experimental-clock-residual-ms`; a suspect-prefix manifest may be supplied with `--experimental-clock-prefix-manifest`. Experimental mapping requires staged `--output_dir` output and rejects in-place execution. No operational tolerance has been selected for the corpus.

Each trial fits centered OLS between hardware tick tokens and host arrival time. The leave-one-out residual is an operational rejection criterion, not a bound on acquisition synchronization or stimulus latency. Event timestamps remain host timestamps; subtracting a common trial origin does not independently prove that the event and frame streams share a physical clock.

The external manifest binds complete kinematics and event file hashes, source row indices, and raw time tokens. Declared leading prefixes use median mapped suffix cadence for backward extrapolation. Those prefix values are model estimates, never observed measurements. There is no automatic prefix expansion, reset stitching, epsilon repair, or trial deletion.

Provenance is retained at source-CSV-row scope; it does not assert one-to-one correspondence with resampled tensor frames. The strict canonical duplicate/nonfinite-time guard remains. Firmware's inspected nominal cadence (about 5 ms) is distinct from the model grid (4 ms), and the historically flashed firmware is not verified.

## Lazy (ELT) model-clock contract

The enhanced lazy entry point `nsmor.pipeline.io.ClockAwareLazyDataset` resamples each trial on demand from the source CSV onto a declared model grid, reusing the shared `resample_trial_for_model`. `scripts/prepare_metadata.py` emits `lazy_model_clock_contract` (schema `lazy-model-clock-v1`) in a namespace distinct from the eager grid keys, so tensor-free metadata does not trip the eager completeness gate. Model-grid samples are causal previous-source-sample-hold estimates, not observations; a pure-wind trial's leading zeros are synthetic alignment padding computed from the model dt (1425 frames at 4 ms), never the source cadence (1141 at 4.997 ms). `scripts/convert_metadata_to_etl.py` emits the full eager `model_dt_ms`/`model_grid_provenance`/`anchor_frames` contract for resampled arrays, validated by `load_dataset_with_fingerprint(..., expected_dt_ms=...)`. Legacy markerless metadata loads on the source cadence with an explicit unverified-clock warning and is never stamped with a false `model_grid_provenance`; an invalid contract or a contradictory consumer clock fails closed. This is engineering plumbing only and makes no biological or timing-accuracy claim.

## Gating refinement

Condition handling distinguishes `wind_only`, `visual_only`, `multisensory`, and `no_stimulus`. The descriptive contrast is wind-only versus visual-present (`visual_only` plus `multisensory`); condition counts remain separate and `no_stimulus` is not included in that contrast. Unknown or conflicting metadata is excluded with a reason. Cohen's *d* uses pooled sample variance; insufficient sample size or zero variance is unavailable, not a false zero. Recording prefixes are not verified independent animals because animal identity is unavailable.

## Verified evidence

- Full candidate regression: 1,248 passed, 1 skipped, 11 warnings, exit 0; candidate and frozen-core hashes matched before and after.
- The skipped real-data alignment test requires unavailable `nsmor_subset_small.pt`.
- Stage-1 descriptive corpus check: 256 files (128 kinematics and 128 events), 2,432 trial pairs, 166 declared prefix trials / 167 rows, 2,431 fitted trials plus one singleton, and no structural failures before residual rejection. The zero-tolerance diagnostic exits before the production helper's final mapped-axis check for positive-residual fits; this count does not certify every reconstructed axis. Maximum leave-one-out residual summary: median 24.283855 ms, P95 35.396806 ms, maximum 108.340565 ms.

- Stage-2 descriptive axis and prefix sensitivity: 22 deterministic diagnostic tests passed. The pinned corpus run rechecked 256 input files and artifact hashes before/after, reconstructed 2,431 strictly increasing fitted suffix axes plus one host-only singleton, and covered all 166 declared prefixes / 167 rows. Both prefix alternatives were strictly increasing in these cases; this does not select either as physical truth.
- Prefix origin difference (cadence estimate minus original-host prefix): minimum -4.912455 ms, median -1.432334 ms, maximum 0.932573 ms. Across the reported boundary rows, the maximum absolute differences were 2.528680 cm/s in velocity and 1,538.502596 cm/s² in acceleration. Of 1,254 trial-matched events, 31 changed recorded-window membership across 18 trials (13 `trial_start`, 18 `phase_transition`). These nonzero sensitivities must not be reported as evidence that prefix reconstruction is harmless.

These diagnostics describe host-arrival association and model-dependent prefix sensitivity only. They do not establish scientific timing acceptance, biological response-window robustness, model convergence, independent-animal inference, or an untouched holdout.

## Pending release evidence

Biological response-window sensitivity and an evidence-based operational tolerance remain pending. Staged ETL, nested-prior evaluation, training, and the downstream descriptive analyses have since executed (current receipt `/mnt/d/Projects/NSMoR/.scratch/seven-analysis-sorted-20261001-r6/status.json`; `scientific_acceptance=pending`). The earlier `.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/seven-final-analyses-single-process-20261001/status.json` receipt is superseded history only and is not current evidence. The Jacobian frozen-input control failed, so `jacobian_spectrum.json` is published `status=withheld` with empty spectral statistics for every epoch and carries no stability interpretation. The historical failed Stage-02 QC evidence is preserved unchanged. Missing experimental conditions and verified animal identity remain explicit limitations. The engineering safeguards and descriptive diagnostics are not a completed scientific release.

Evidence paths (external scratch, not release artifacts):

- `.scratch/combined-validation-r4/verified-wsl-An7UkhYc/`
- `.scratch/clock-corpus-r4/verified-run-H0kiCAI6/`
- `.scratch/clock-corpus-r4/r3-checks-qZiIekrh/` — 22 deterministic tests; source SHA bindings.
- `.scratch/clock-corpus-r4/r3-corpus-szalVZ5Y/` — Stage-2 corpus run; report SHA256 `c909b40458e3aeb3745e700d1fa70d8f8ed1583e9359c58dd871d7579ddb709e`.

Stage-2 runner: `.scratch/clock-corpus-r4/stage2_reviewed_r3.py`; tests: `.scratch/clock-corpus-r4/test_stage2_reviewed_r3.py`. Stage-1 result SHA256 is `6187c988a2eef6202e752dcd83939a6e60b2785bc6bccf8cee3a7e34d932e51b`, distinct from the historical failed-QC report. The rejected R1 execution and incomplete R2 draft are preserved separately and are not validation evidence. The first R3 test run had one error-message-regex mismatch (21 passed, 1 failed); the source already rejected the input, and the corrected assertion passed in the cited 22-test run.
