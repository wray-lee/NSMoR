# Phase-2 secondary battery proposal — October 13, 2026

## Motivation

Implement the already declared secondary descriptive analyses in phase-2
protocol §8/8.1 as one reusable CPU checkpoint CLI. Existing squared-error
artifacts cannot recover signed underprediction or natural pathway timing.
The original lesion loader constructs the whole corpus even when its population
sidecar records validation membership, so inference must physically select the
persisted outer-validation rows.

## Implementation

Only these four paths belong to this task:

- `nsmor/analysis/secondary_battery.py`
- `scripts/analyze_phase2_secondary.py`
- `tests/test_secondary_battery.py`
- `docs/proposals/secondary-battery-20261013.md`

The CLI requires `--checkpoint --arm {R0,R1,R2,R3} --dataset --nested_prior
--output_dir`. It imports the frozen scorer arm validator read-only and verifies
its SHA-256 `786f0bc80bed8e45e686e61018f5d43825971663e09ff9796c9ed3e8f1a21a9d`.
Exact seeds 42/43/44, arm zero channels, unverified recording-prefix identity,
nested validation lineage, physical scalar predictions, and the frozen 4 ms
clock are enforced. Adaptive/refinement/time-delay modes are refused. Shared
checkpoint and prior helpers verify dataset/path/byte/split bindings.

Reuse `simulate_lesion.load_dataset(checkpoint_model=..., nested_prior_artifact=...,
max_seq_len=<checkpoint>, pre_anchor_frames=1200)`, then an actual `Subset`
with the exact validated `outer_val_indices`, batch size 1, zero workers. Join
references by source row. R2 explicitly zeroes channels 2/3 on every actual
validated `xb`; targets and prior columns are preserved. No real dataset is
loaded during development, and CUDA is hidden for the CLI and tests.

Each trial receives one natural full-sequence prediction/internals pass and
both existing gate readout replacements: LIF replaced `(g_lif=0,g_gru=1)` and
GRU replaced `(g_lif=1,g_gru=0)`. These are readout replacements, with windowed
scoring of full-sequence outputs; no state lesion or timed intervention claim.
Reference is the helper's cropped anchor. With `r` the first observed speed
`>=10` at/after reference, onset is `[ref,r+200ms)`, sustained is frames at/after
`r+200ms` with observed speed `>=10` and at least five eligible frames, and
overall is `[ref,L)`. Nonresponders have unavailable onset/sustained scores.
Natural residual moments per window are signed `predicted-observed` bias in
cm/s and population variance/MSE in `(cm/s)^2`, with the identity
`MSE=bias^2+variance` checked. This is not an across-seed variance or calibration
analysis. Shared paired comparisons retain trial d_z and prefix trial-mean
deltas with exchangeability false and no new p-value.

Natural `lif_spikes`/`gru_hidden` RMS first departure uses a requested 250 ms
baseline, round(250/4)=62 frames, effective 248 ms, and mean plus three times
population SD (SD floor 1e-6). Too-short/missing baselines are unavailable;
complete baselines with no departure are valid right-censoring. Save exclusive
opportunity end per trial. Observed median/IQR and common-uncensored paired
LIF-minus-GRU deltas are descriptive; exact zero is a tie. No population ordering
is inferred from marginal medians. The response curve uses all eligible timing
trials as its denominator and is a defective CDF with varying available followup,
not Kaplan–Meier or common-followup survival; timing is RMS-scale dependent.

Prestimulus uses signed observed/predicted `>=10` and two adjacent frames for
bouts; uncompressed masks retain gaps. Save observed and predicted frame/bout
rates, frame false alarms conditional on observed `<10`, all-prestimulus model
bout rate, and strict trial false alarms conditional on no observed bout.
Report frame distributions for exactly zero, `(0,10)`, and `>=10` as speed
proxies only. Independently annotated states are unavailable. Per-condition
summaries are explicitly unavailable because validated reference metadata does
not establish trustworthy conditions.

Export `trial_level.csv`, `summary.json`, and PDF/SVG/PNG publication figures.
Record source trial identity, prefix, crop/reference, timing status/opportunity,
window MSE/delta/count/moments, prestimulus counts, config and code/artifact SHAs.
Nonempty output directories are rejected and export errors propagate. Nonfinite
inputs, predictions, internals and moments fail; errors are never replaced.

## Rationale

Reuse the accepted loaders, crop/reference helper, gate overrides and comparison
primitive; retain only scalar trial rows while streaming inference. No core,
frozen scorer, support/robustness code or parent evidence assessment is edited.
The analysis claim is `model-internal exploratory`, with no animal/circuit claim
or extension of the frozen statistical family. No commits or pushes are made.

## Validation

All commands use atomic interactive WSL Zsh with the torch environment and
CPU/BLAS one-thread limits. Focused synthetic test command:

```powershell
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' && cd /mnt/d/Projects/NSMoR-publication-analysis && pytest tests/test_secondary_battery.py -q --disable-warnings --tb=short"
```

The runner fixture builds a tiny model and synthetic corpus, proves only nested
validation rows reach all three forwards, verifies actual R2 zeros and nested
priors, and exercises all exports. Other tests cover window boundaries, short
sustained scoring, invalid/censored/event timing and exact ties, bout gaps,
false-alarm denominators, signed residual moments and provenance/finite guards.
Final focused result: **28 passed in 4.95 s**. All four arms, accepted seeds,
validation-only streaming, R2 zeroing, invalid/censored timing denominators,
common-uncensored pairs, residual moments, unavailable windows, finite guards,
provenance rejection and loud export failures are covered with synthetic inputs.

Adjacent synthetic regression result: **458 passed in 28.86 s** before the final
removal of the extra prefix d_z calculation and the additional focused cases.
Exact adjacent regression command:

```powershell
wsl -e zsh -i -c "source ~/.zshrc && openconda && conda activate torch && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' && cd /mnt/d/Projects/NSMoR-publication-analysis && pytest tests/test_secondary_battery.py tests/test_analysis_nested_prior_inputs.py tests/test_lesion_interventions.py tests/test_model_comparison.py tests/test_realdata_phase2_options.py tests/test_analysis_prediction_units.py -q --disable-warnings --tb=short"
```

Two independent read-only snapshot reviews returned **ACCEPT** before that final
simplification. The parent owns the final sealed-snapshot review. The final
contract is **trial d_z and prefix trial-mean deltas**, with no extra prefix d_z.
The prior SHA check reports a mismatch to checkpoint provenance; the shared
helper authenticates the actual loaded prior bytes.

No real dataset, production checkpoint, CUDA run, full-suite regression or
training/data-pipeline smoke run was used for this development task. Existing
shared loading remains eager in corpus RAM; only inference is batch-one and
physically restricted to saved validation rows. No commit or push was made.
