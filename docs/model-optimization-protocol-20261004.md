# Model optimization protocol — 2026-10-04

Status: design accepted independently; implementation, testing, and candidate
training remain gated. This protocol is frozen before any candidate training.

## Scope and immutable evidence

- Engineering observed-history prediction only. The lagged physical velocity
  `X[..., 2] = v(t-1)` is an existing causal input, not new target information.
  Evaluation persistence is calculated independently from each trial's targets,
  excludes frame zero, and never crosses a trial boundary.
- Existing fresh baseline completed naturally at epoch 300, exit 0. Its best
  validation checkpoint was selected at epoch 296. Do not resume, restart,
  overwrite, or duplicate that training run or its finalized outputs.
- Dataset SHA-256:
  `b1bd5578025fb5eaa6f4eb3e355c097b6576d44dbedaa001027f1de5447e06b2`.
- Nested-prior SHA-256:
  `f25856304a5857695945bf5083ca54745f47fe5dffe915111f6ba28d524ddf94`.
- Nested seed 42; validation fraction 0.2; exact saved split indices and prior
  mapping must match. Evaluation scope is `nested_outer_validation`, not an
  untouched holdout. Validation-guided model selection is exploratory.
- `animal_identity_status=unverified` and
  `biological_completeness=not_established`. Recording prefixes are not animals.
  Missing stimulus anchors do not establish BV-induced afterstimulus censoring.

## Intervention and compatibility

`prediction = existing_head + k * raw_lag_velocity`.

- `k` is a fixed configuration scalar, finite and in `[0, 1]`, not a learned
  parameter. It is persisted in checkpoint configuration, not `state_dict`.
  Only the existing head/backbone are optimized; there is no coefficient
  identifiability or biological-pathway claim.
- Default `k=0` must retain bit-exact historical predictions, parameter count,
  state dictionary, and strict historical checkpoint loading.
- Nonzero `k` is supported only with `normalize_targets=False` and
  `target_clip_cm_s=0`. Unsupported combinations fail closed in both supported
  training entries. Analysis models with nonidentity restored target mean/std
  or nonzero clip also fail closed, including the JAX analysis wrapper.
- Canonical loaders, data loaders, checkpoint storage, and `loss.py` remain
  unchanged. The canonical loader does not certify normalization metadata;
  manual low-level normalized use is unsupported. Runtime guards are defense
  in depth, not proof that manually fabricated checkpoint metadata is valid.
- Validate nonzero-intervention feature schema, lengths before any JAX cast,
  and lag/mask shapes. Padding must not contribute a lag residual. Recurrent
  states/internals and head-gradient semantics must remain unchanged.
- PyTorch, fused JAX, Flax training, and JAX analysis must propagate `k`.
  Real-JAX checks may not pass through a PyTorch fallback. Deterministic
  additive-effect tests are distinct from documented fp32 spike-timing drift.
- Nonzero `k` is not supported for autoregressive rollout. Both rollout API
  and CLI reject it. Conditional fixed-input hidden-state Jacobians do not
  certify closed-loop stability; no such claim or AR result will be reported.

## Fixed experiment set and gates

Initial bounded set: two fresh candidates, `k=0.5` and `k=1.0`.

1. Finish separate read-only fresh-baseline zero/persistence/sensitivity and
   loss-contribution diagnostics. Do not substitute historical diagnostics.
2. Implement compatibility gates and focused regressions; obtain two
   independent source-bound code ACCEPTs.
3. Tester verifies smoke, strict checkpoint/default compatibility, numerical
   finiteness, head gradients, state/padding invariants, real-JAX behavior,
   and regression suite. Failed or unavailable gates are reported explicitly.
4. Run a separately labeled stability/convergence smoke of at most 20 epochs;
   it is not a matched 300-epoch candidate or optimization result.
5. Only then train the two fresh 300-epoch candidates with the baseline's
   exact saved configuration, seed, dataset/prior/split, budget, optimizer,
   loss coefficients, cadence, device/precision policy, and checkpoint
   selection rule. Only `k` and isolated output locations change. No baseline
   resume and no candidate initialization from its best checkpoint.
6. Score best checkpoints with the definitions below. Report both candidates,
   including failed or unfavorable results. Any further experiment requires
   a new prospective protocol; never enlarge this family retrospectively.

The baseline head-only model, head-plus-skip candidates, and independent
persistence form a matched ablation. Incremental value is measured against
head-only, not merely against persistence. This bounded set can establish
its best observed result, never a global optimum or the data's absolute ceiling.

## Metrics fixed before candidates

Primary: raw pooled MSE on all valid frames, with RMSE, MAE, and R-squared.
Do not replace this primary metric after observing unfavorable results.

Co-report:

- Trial-local `t>=1` aligned model, independent persistence, and zero metrics,
  including MSE-based skill and explicit undefined skill denominators.
- Sustained high-velocity band `abs(y)>=10 cm/s`, minimum run length 2, as an
  engineering guardrail, not a stimulus-conditioned escape label.
- Exact sensitivity grid: thresholds `[5, 10, 20, 50] cm/s` crossed with minimum
  run lengths `[1, 2, 3]`. Report all 12 cells; no best-cell selection.
- Sustained membership is calculated within complete valid trials before
  excluding frame zero for aligned scoring. No cross-trial runs. Full-frame
  and aligned partitions have separate counts and must not be conflated.
- Rest complements and per-trial/frame/prefix counts in each cell. Empty
  bands are unavailable, not zero error.

Retrospective high-velocity masks never enter model features. Raw errors
remain visible even if a later separately authorized robust objective is used.

## Statistical family

The inferential/descriptive comparison grid uses `t>=1` throughout so the
persistence denominator is independently defined on identical frames.

- Scopes: aligned overall plus the 12 sustained high-velocity cells = 13.
- Comparators per candidate: verified head-only baseline, independent
  persistence, and zero = 3.
- Candidates: 2. Declared family size: `2 * 3 * 13 = 78`.
- Report every declared comparison, including unavailable cells. No extra
  full-frame or resting p-tests are silently added to this family; their
  numerical metrics remain co-reported.
- Per-trial delta is candidate MSE minus comparator MSE on the same eligible
  frames. Negative is better. Trial-level paired Cohen's `d_z` is the mean
  delta divided by its sample standard deviation; report trial/frame/prefix
  counts. Fewer than two trials or zero delta variance gives null `d_z` with
  an explicit reason, not an invented finite effect size.
- Prefix-cluster statistic: first average eligible trial deltas within each
  recording prefix, then take the unweighted mean of those prefix means.
  Sign-flip entire prefix means, not individual frames or trials. This p-value
  has a different unit from trial-level descriptive `d_z`.
- Two-sided sign-flip p-values are conditional descriptive diagnostics under
  a stated joint sign-exchangeability null, not proof of animal independence.
  Minimum 6 eligible prefixes; exact enumeration for at most 16 prefixes;
  otherwise 100,000 Monte Carlo sign patterns, fixed seed 42 and plus-one
  correction. Unsupported exchangeability or too few prefixes yields null
  p-value with reason. No unadjusted significance or biological claims.
- Holm-Bonferroni uses the fixed 78-slot family, with unavailable slots
  conservatively treated as p=1 for adjustment bookkeeping but still reported
  as null. Report family size and the valid-test count explicitly.
- Validation tuning and unverified identities preclude untouched-holdout or
  animal-generalization claims regardless of these numerical diagnostics.

## Loss-dilution diagnosis

Separate valid-frame prediction SSE/MSE, regularizer contributions, and
measured gradient contributions. The fresh aligned baseline's 2.42% escape
frames account for 88.58% of prediction SSE: this is not proof of loss dilution
or a mandate for reweighting. An end-checkpoint sampled gradient probe is
local descriptive evidence, not a reconstruction of training dynamics.
Unmeasured quantities remain explicitly unmeasured.

## Release boundary

Conditional user authorization permits a PR/merge only after actual work,
independent reviews, testing, and experiment reporting pass. Stage explicit
reviewed paths only; never `git add -A`. Preserve protected untracked paths
and all formal DATA/NEST/checkpoints/receipts. No release is authorized by
this design document alone.
