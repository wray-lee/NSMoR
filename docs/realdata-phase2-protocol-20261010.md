# Real-data phase-2 prospective protocol — 2026-10-10

Status: **PROSPECTIVE**. Frozen before any arm training. Every arm, config
delta, metric, comparator, analysis and claim boundary below is fixed in
advance. No retrospective enlargement is permitted; any further experiment
requires a new prospective protocol.

Companion documents: `docs/realdata-preliminary-protocol-20261009.md` (format,
rigor), `docs/realdata-preliminary-conclusions-20261009.md` and
`docs/realdata-exploratory-step1-20261010.md` (including its read-only setup
audit section). The preliminary phase established that **neither A1 nor A2
beats persistence**; phase 2 tests *why*, with the controls the preliminary
protocol did not have.

## 1. Scope and immutable evidence

- Engineering observed-history prediction on the existing real corpus only.
  No new data collection, no synthetic data, no animal-level inference.
- Dataset SHA-256:
  `b1bd5578025fb5eaa6f4eb3e355c097b6576d44dbedaa001027f1de5447e06b2`.
- Nested-prior SHA-256:
  `f25856304a5857695945bf5083ca54745f47fe5dffe915111f6ba28d524ddf94`.
- Nested seed 42; validation fraction 0.2; the exact saved split indices and
  prior mapping are read from the artifact. Evaluation scope is
  `nested_outer_validation`, **not** an untouched holdout. Validation-guided
  model selection is exploratory.
- Corpus: 2304 labelled trials (multisensory 1188, visual_only 720, wind_only
  396; **no** no_stimulus condition) + 128 unlabelled no_stimulus anchors;
  64 recording prefixes, 52 train / 12 val; **432 validation trials, 12
  recording prefixes, 1,036,748 frames**. `animal_identity_status=unverified`;
  `biological_completeness=not_established`. Recording prefixes are not
  animals.
- **Target is scalar speed** (non-negative, `sqrt(dx²+dy²)/dt`, 4.006 ms grid),
  unchanged by user decision. 80.4% of validation frames are exactly 0 and
  2.53% are ≥ 10 cm/s. No signed-velocity/heading target is introduced; the
  data contract and the frozen target binding are untouched.
- Frozen baseline `.scratch/model-opt-baseline-300-20261003-rerun/`
  (relu/off, k=0, seed 42) is **read-only** (best_model.pth sha
  `709ca1f1a121c2b0904d14ca056ad335dac14f5e178e9edb3f22b541ba18ff1b`). No
  baseline rerun, resume, overwrite, or candidate initialization from it.
- The A1 arm `.scratch/realdata-preliminary-20261009/A1-swiglu-off/`
  (best_model.pth sha
  `4a43fb0afb3f67e0a86ab3565d0d6fcf7f0f3c7aa55d8e5745e125922c3ed579`) is
  **read-only**: it supplies the shared base config and the §6 pre-training
  diagnostic, and is never retrained, resumed or overwritten.

## 2. User decisions (frozen)

- **KEEP the scalar speed target.** No pipeline change. The signed-velocity /
  heading target flagged in the step-1 audit is explicitly **out of scope**;
  it would change the data contract and the frozen target binding.
- **Budget = 4 key configs × 3 seeds = 12 serial runs** (~10 h each, ~120 h
  serial). No extra GPUs.
- No new data collection. **No `no_stimulus` trials are required** (§8.1): the
  spontaneous baseline is the pre-stimulus segment inside the existing crop.
- **Phase 2 is the FOUNDATION layer of a three-layer plan** (foundation →
  biological description → mechanism tests); the mechanism line is a separate
  phase-3 pre-registration (§9, F1).

### 2.1 Data needs (F3)

Still needed, and **not** supplied this phase:

- `wind_only` trials **in the validation split** (for the modality contrast);
- a **TTC = 0** condition;
- **verified animal identity** (currently `unverified`);
- an **animal-split test set**.

With 12 validation prefixes, the prefix-cluster sign-flip requires **12/12
agreement** to reach Holm significance (§6 resolution limit), so **more
prefixes / animals is the single most effective improvement** — it raises the
smallest attainable sign-flip p and relaxes the all-same-sign requirement. None
of these additions is performed here.

## 3. Common arm configuration

All four arms share **A1's resolved config**: `activation=swiglu`,
`refinement_mode='off'`, `lambda_compute=0.0`, `persistence_skip=0.0` unless
the arm says otherwise, the same dataset / nested prior / split, 300 epochs,
`early_stopping_patience=20`, `batch_size=128`, `max_seq_len=2400`,
`normalize_targets=false`, `target_clip_cm_s=0`, and every other baseline leaf.

Two new, default-off controls implement the phase-2 design. Both default to
the current behaviour, so a config that omits them is bitwise the preliminary
phase's config. They are **script-owned**: declared in a top-level `phase2:`
YAML block (plus the `--selection_metric` / `--zero_input_channels` CLI flags),
resolved by `scripts/train.py::build_config` into module state
(`_SELECTION_METRIC`, `_ZERO_INPUT_CHANNELS`) and consumed by
`build_dataloaders` / `train`. They are **not** added to the protected
`nsmor/config_parser.py` schema, so `ExperimentConfig.to_dict()` — and hence
every existing config and checkpoint dictionary — is byte-unchanged.

**Vocabulary (unambiguous).** There is **no** protected-schema leaf named
`checkpoint.selection_metric` or `data.zero_input_channels`, and no such leaf
exists in `ExperimentConfig`. The only spellings are the script-owned
`phase2.selection_metric` / `phase2.zero_input_channels` keys of the top-level
`phase2:` YAML block and the equivalent `--selection_metric` /
`--zero_input_channels` CLI flags. Section headings below use those script-owned
names deliberately; they are **not** config-schema leaves.

### 3.1 `phase2.selection_metric: total | mse` (default `total`)

Step 1 found the trainer selects `best_model.pth` and drives early stopping on
validation **total** loss (`scripts/train.py::train`, the checkpointing block),
of which MSE is 96.5% but
the `g_gru²` router term (2.1%) and jerk (1.3%) pull the selected weights off
the MSE optimum. `phase2.selection_metric: mse` selects on the **frame-weighted
pooled masked MSE over aligned eligible frames (t ≥ 1, padding excluded)**
alone: the sum of all aligned squared errors divided by the total aligned frame
count over the whole validation split (not a per-batch macro-average), so the
selected checkpoint minimises exactly the pooled quantity the scored primary
metric `1 − MSE_model/MSE_persist` divides by. The t=0
exclusion is exact (one frame per trial, ~0.04% of frames) and matches the
skill-vs-persistence primary metric's `t ≥ 1` convention. Only `best_model.pth`
selection and the patience counter change; the checkpoint dictionary layout,
the stored `val_loss`/`best_val_loss` scalars (both remain the selected
metric) and every other leaf are unchanged. A resume that switches this control
is refused (fail closed), because the persisted `best_val_loss` would change
meaning. The selected metric is recorded as an additive top-level checkpoint
key `selection_metric` (alongside the existing additive provenance keys); the
frozen checkpoint dictionary structure and the config dict are unchanged.

### 3.2 `phase2.zero_input_channels: [int, ...]` (default `[]`)

Per-frame feature columns of `X` (dim 8) forced to `0.0`. Only the physical
sensory columns `[0, sensory_dim)` may be listed; the MCMC prior columns (4:8)
are validated to sum to 1 and are refused at the parse boundary. The ablation
is applied **in `scripts/train.py`** (`zero_input_channels_inplace`) to the
`NSMoRDataset`'s **deep-copied** `sequences`, after dataset construction and
before the loaders, so **the stored dataset bytes and the caller's arrays are
never modified** and the data pipeline (`nsmor/nsmor_dataloader.py`) is
untouched. Lazy loading refuses a non-empty list (fail closed), because the
lazy path would silently be the unablated control. The declared channels are
recorded as an additive top-level checkpoint key `zero_input_channels`; the
scorer re-applies the identical ablation to the val loader (train/serve
consistency). A resume that switches the ablation is refused (fail closed,
alongside the `selection_metric` guard), because it would silently continue an
arm under a different input transform.

**Single rollout regime (frozen).** Every arm is scored under **one** rollout
regime, the one used in training: at each rollout step the model's own
predicted speed/acceleration are fed back into channels 2–3, **and then the
arm's declared input transform is applied** (for R2, which declares
`phase2.zero_input_channels: [2, 3]`, those channels are zeroed after the
feedback write). The order — feedback, then the arm's declared transform —
mirrors the training pipeline exactly (`scripts/train.py` zeroes the deep-copied
dataset channels once, before the loaders; the model never sees a non-zero
channel 2–3 under R2). R2's h-slots are therefore **available** and `R2|R0|h5`
contrasts the ablation alone, not two different experiments. This rule is frozen
here and recorded per arm in the scorer output (`h5_input_protocol`).

Both controls are exposed on the CLI (`--selection_metric`,
`--zero_input_channels`) and in the script-owned `phase2:` YAML block; both
default off. `scripts/train.py::build_config` resolves them into module state
(never into `ExperimentConfig`), so the protected config schema
(`nsmor/config_parser.py`) and every `to_dict()` / checkpoint dictionary are
unchanged. The history ablation itself is applied in `scripts/train.py`; no
core file is touched.

## 4. Arms and exact config deltas

| Arm | delta on the common A1 config | role |
| :-- | :-- | :-- |
| **R0** control | `phase2.selection_metric: mse` only | MSE-selected baseline |
| **R1** residual-over-persistence | `model.persistence_skip: 1.0` | residual-skill arm |
| **R2** history-ablated | `phase2.zero_input_channels: [2, 3]` | remove lagged speed/accel inputs |
| **R3** jerk ablation | `loss.lambda_jerk: 0.0` | remove the smoothness term |

Every arm also carries `phase2.selection_metric: mse`. Seeds **42, 43, 44** for
every arm.

Frozen arm configs and the resolution-diff evidence (all under
`config/realdata-phase2-20261010/`):

- `R0-control-seed{42,43,44}.yaml`
- `R1-residual-persistence-seed{42,43,44}.yaml`
- `R2-history-ablated-seed{42,43,44}.yaml`
- `R3-jerk-ablated-seed{42,43,44}.yaml`
- `arm-resolution-diff.json`: each arm resolved through
  `scripts/train.py::build_config` differs from **A1's resolved config**
  (`config/realdata-preliminary-20261009/A1-swiglu-off.yaml`) **only** in
  `checkpoint.output_dir`, `training.random_seed` (43/44 for the non-42 seeds)
  and the single declared arm leaf (`model.persistence_skip` /
  `loss.lambda_jerk`). The two script-owned controls never enter `to_dict()`;
  the diff records their resolved values under `phase2_resolved`
  (`selection_metric: mse` for every arm; `zero_input_channels: [2, 3]` for R2
  only) and asserts each equals the declared `phase2:` block. No baseline leaf
  changes value; `added_leaves` and `missing_leaves` are empty for every arm.
  The generator
  `scripts/realdata_phase2_configs.py --check` reproduces the configs and the
  diff exactly.

### 4.1 Why R1 is reported as skill vs persistence, not as a win

With `persistence_skip=1.0` the model output is `y = f(recurrent state) +
v_lag`, where `v_lag = X[:, :, 2]` is the observed lagged speed and `f` is the
learned residual head. Let `e = Y − v_lag` be the residual target. The scorer's
reported scalar is the **full-output skill**

```
S_full = 1 − MSE(f + v_lag, Y) / MSE(v_lag, Y)
       = 1 − E[(e − f)²] / E[e²]          (since MSE(v_lag, Y) = E[e²]).
```

This is **exactly** the residual skill of the head `f` predicting the residual
`e` against the zero-residual predictor: `f = 0` gives `S_full = 0`, `f = e`
gives `S_full = 1`. The additive lag is present identically in the numerator's
target (`Y`) and in the comparator (`v_lag`), so it cancels; there is no
"contamination" and no separate `MSE_f` quantity to reconcile.

The single honest caveat is definitional, not arithmetic: R1's output is
**persistence plus a learned residual**, not a from-scratch forecast. A positive
`S_full` therefore means the residual head adds information **beyond
persistence**; it does **not** mean the model out-forecasts the trivial copy
from scratch. The k=1.0 "improvement by construction" cannot masquerade as a
win: R1 is reported alongside R0 and the persistence comparator, and any R1
advantage is read as the residual skill of a model whose output already
contains `v_lag`, never as beating the trivial copy.

### 4.2 Why R2 / R3 are the right ablations

R2 removes the lagged observed velocity/acceleration channels (2, 3) the step-1
audit identified as the trivial-copy source: the model's inputs currently
contain, verbatim, the value persistence predicts. If the model's skill
survives zeroing them, the skill is not merely a copy of the observed history;
if it collapses to the zero comparator, the preliminary phase's "lower error
than baseline" was an input-copy artifact. R3 removes the jerk smoothness term,
which the audit flagged as one of only two terms acting on the prediction and
the likely cause of the observed peak-speed under-prediction; it is checked
against peak-speed error, not only MSE.

## 5. Metrics fixed before training

**Primary — skill vs persistence = `1 − MSE_model / MSE_persist`** (negative =
worse than persistence). **Persistence is the must-beat baseline.** Reported
for each arm and each scope:

1. **(a) pooled one-step** on aligned frames (t ≥ 1), all validation frames.
2. **(b) escape frames** (`|y_true| ≥ 10 cm/s`, aligned).
3. **(c) multi-step h = 5** with own-prediction feedback (predicted velocity and
   the derived acceleration fed back into channels 2, 3, **then the arm's
   declared input transform applied**; persistence held at `y[start−1]`; 8
   evenly-spaced start frames per trial), as in step 1. Every arm uses this
   **single rollout regime** (§3.2), so R2's h=5 slots are available and
   `R2|R0|h5` contrasts the ablation alone; the rule is recorded per arm in the
   scorer output (`h5_input_protocol`).
4. **(d) response-onset window** `[onset, response_onset + 200 ms)`, aligned.
   The onset frame is derived from the checkpoint's **resolved** `max_seq_len`
   (2400), so it shares the exact anchor-aligned crop base as the arm's val
   dataloader and `extract_canonical_validation_targets`. **Two target
   sources, asserted equal (fail closed).** As each arm's val loader is scored,
   the scorer asserts elementwise (per trial, in order) that the loader-produced
   targets equal `extract_canonical_validation_targets`' `y_true_seqs` and
   refuses the run otherwise — exactly as
   `scripts/evaluate_realdata_preliminary.py:206-212` does — so the aligned
   scopes (loader `y`) and the rollout/onset scopes (canonical `y_true_seqs`)
   provably share one target series.

Each scope is reported **against the measured persistence ceiling of its own
comparator**, computed on the canonical nested validation split with the exact
arithmetic the scorer uses. The horizon (c) scopes use persistence **held at
`y_true[start−1]` over the whole rollout** (`_hN_trial`), which is the
comparator `skill_vs_persistence` divides by; it is **not** the pure lag-N
frame the step-1 setup audit reported, and the two diverge sharply at h=5
(held-at-start MSE 0.925 vs pure lag-5 10.07). Both are recorded, but only the
held-at-start row is the comparator. All values are frozen in
`docs/realdata-phase2-ceilings-20261010.json` and the scorer's
`PERSISTENCE_CEILING` / `PURE_LAG_CEILING` tables.

**Ceiling frame partition (identical to the scored skill).** Every ceiling is
computed with the **same per-trial frame partition and frame-weighting the
scorer's skill uses** (`_pooled_mse` over the per-trial eligible frames), not a
single global pool:

- aligned / onset: per-trial eligible aligned frames (t ≥ 1); onset restricts
  to `[onset, response_onset + 200 ms) ∩ [1, L)`.
- escape (b): per trial, the `t ≥ 1` frames with `|y_true[t]| ≥ 10 cm/s`; the
  ceiling pools the per-trial escape-frame MSEs weighted by their own escape
  frame counts.
- h = 5 / 25 / 125: per trial, the (start, k) rollout frames over the 8
  evenly-spaced starts; the **escape** variant restricts the *same* (start, k)
  pairs to frames with `|y_true| ≥ 10 cm/s` and pools them weighted by those
  counts — the identical partition the scorer's `h{h}_escape` skill uses.

**Ceiling reconciliation (fail closed, M3).** The scorer recomputes the
persistence MSE on the scored data for every scope and asserts it matches the
frozen `PERSISTENCE_CEILING` at `rel 1e-3`; on mismatch it emits a
`ceiling_vs_measured_delta` record and **fails closed** rather than reading the
skill against a stale ceiling. A ceiling that no longer matches its comparator
aborts the run.

| Scope | comparator | persistence R² | persistence MSE (cm/s²) |
| :-- | :-- | :-- | :-- |
| one-step (a), aligned | lag-1 (`y_true[t−1]`) | 0.927 | 1.910 |
| escape (b) | lag-1, escape frames | 0.857 | 53.05 |
| h = 5 (c), overall | **held-at-start** | 0.741 | 0.925 |
| h = 5 (c), escape | **held-at-start** | −0.205 | 41.44 |
| onset (d) | lag-1 in window | 0.880 | 26.65 |
| h = 25, overall | held-at-start | 0.424 | 2.289 |
| h = 25, escape | held-at-start | −1.277 | 132.13 |
| h = 125, overall | held-at-start | −0.305 | 8.125 |
| h = 125, escape | held-at-start | −2.001 | 366.95 |

Pure lag-N audit values (step-1 setup audit, reference only — a *different*
comparator that scores no phase-2 slot): h=5 0.615 / 10.07 (escape 0.179 /
303.65); h=25 −0.246 / 32.85 (escape −1.157 / 798.34); h=125 −0.770 / 48.50
(escape −1.401 / 890.26).

At h = 1 a model can add at most ≈ 7% of variance overall and ≈ 14% in escape
frames; the reported skill must be read against these ceilings.

**Secondary (reported, never substituted for the primary):** h = 25 (persistence
R² 0.424 overall / −1.277 escape, held-at-start) and h = 125 (−0.305 / −2.001);
peak-speed error; the zero comparator; and pooled MSE over **all valid frames
(t ≥ 0**, padding excluded, descriptive). For each horizon the scorer emits the
skill-vs-persistence scalar **both overall and in escape frames**
(`secondary_metrics[arm].horizons[scope]` / `.horizons[scope]_escape`), each
beside the matching held-at-start ceiling (`persistence_ceiling` /
`persistence_ceiling_escape`), so no escape ceiling is ever attached to an
overall scalar. It also emits the per-arm peak-speed error and the declared
t ≥ 0 pooled MSE (`secondary_metrics[arm].pooled_mse_t_ge_0`); these never
enter the declared family. The peak-speed reference is the **scored split's own**
observed peak (`observed_peak_cm_s`, the mean over the 432 val trials' aligned
per-trial peaks); the step-1 event-level value 65.8 cm/s (362 paired trials) is
retained only as a provenance annotation (`step1_event_level_observed_peak_cm_s`).

## 6. Statistical family (frozen, declared up front)

The phase-1 exchangeability mishap is not repeated: this protocol **asserts the
joint sign-exchangeability null up front** and records the exact scorer flag.

- **Candidates:** `{R0, R1, R2, R3}` (4).
- **Comparators:** `{persistence, zero, R0}` (3). The `R0` comparator is the
  arm-vs-R0 contrast; `persistence` is the must-beat baseline; `zero` is the
  trivial floor.
- **Scopes:** `{aligned_pooled, escape, h5, onset}` (4).
- **Declared family size: 4 × 3 × 4 = 48.** The four self-comparison slots
  (`R0|R0|<scope>`) are reported **unavailable** with reason `self_comparison`,
  never silently dropped; 44 slots are potentially available.
- Per-trial delta = candidate MSE − comparator MSE on identical eligible frames
  (negative favours the candidate). Trial-level paired Cohen's `d_z` =
  mean delta / sample sd(delta); trial / frame / prefix counts reported. Fewer
  than two trials or zero delta variance gives null `d_z` with a reason.
- **Prefix-cluster sign-flip:** average eligible trial deltas within each
  recording prefix, then the unweighted mean of prefix means; two-sided
  sign-flip p over whole prefix means, minimum 6 eligible prefixes. **Branch-
  specific correction:** with 12 prefixes the frozen primitive takes the
  **exact** branch (enumeration of all `2^12 = 4096` sign patterns, no plus-one
  correction; `nsmor/analysis/model_comparison.py:506-513`), so the smallest
  attainable two-sided p is `2/4096 ≈ 0.00049`. The plus-one correction applies
  only to the **Monte-Carlo** branch, taken at more than 16 prefixes (100,000
  seed-42 sign patterns). It is a **conditional descriptive diagnostic under a
  stated joint sign-exchangeability null**, not proof of animal independence.
- **Exchangeability is asserted** via the scorer flag `--exchangeability_asserted`
  (recorded in the output JSON's `exchangeability_asserted` and
  `scorer_command`). The exact scoring command is:

  ```
  python scripts/evaluate_realdata_phase2.py \
    --r0 <R0-s42> <R0-s43> <R0-s44> \
    --r1 <R1-s42> <R1-s43> <R1-s44> \
    --r2 <R2-s42> <R2-s43> <R2-s44> \
    --r3 <R3-s42> <R3-s43> <R3-s44> \
    --dataset <corpus>/nsmor_dataset.pt \
    --nested_prior <corpus>/nested_split_seed42.pt \
    --output_dir results/realdata-phase2-20261010 \
    --exchangeability_asserted
  ```

  Omitting the flag yields null p-values with reason
  `exchangeability_not_asserted`; the primary reported run **includes** it.
- **Holm-Bonferroni** over the fixed 48 slots. The family is the fixed
  48-shape declared above; the frozen `holm_correct_declared_family` is bound
  to the 78-slot k-family shape and cannot be called at 48, so the adjustment
  reuses the frozen **step-down primitive**
  `nsmor.analysis.uq.holm_bonferroni` over all 48 slots with every unavailable
  slot carried as **p = 1** for bookkeeping. The first threshold is therefore
  `0.05 / 48`, and the unavailable slots are reported null (never a fabricated
  significance). Family size, valid-test count and unavailable-slot count are
  reported explicitly. The per-slot statistic is the frozen
  `nsmor.analysis.model_comparison.paired_mse_comparison` primitive — no
  paired-delta or sign-flip algebra is re-implemented.
- **Seed-level and prefix-level effects.** The **inferential unit is the
  per-trial mean over the 3 seeds** (42, 43, 44), which enters the family as a
  single per-trial observation; a seed-mean is treated as one observation and
  cannot by itself propagate seed-to-seed disagreement. To keep that visible,
  the scorer writes every **per-seed per-trial MSE vector** and a **seed-level
  spread** summary (per-seed trial mean — the unweighted mean over the trial
  MSEs, named `per_seed_trial_mean` — and the mean/max per-trial seed sd) into
  the output JSON's `seed_level_summary`, for **every scope it reports**
  (primary and secondary), and records
  `inferential_unit: seed_mean_over_3_seeds`. Seed spread is descriptive;
  every preliminary claim is stated at the seed-mean level. Prefix-cluster
  effects are the sign-flip above; trial-level effects are `d_z`.
- **Resolution limit (stated honestly).** With **12 prefixes** the smallest
  attainable sign-flip p is `2/4096 ≈ 0.00049` and Holm's first threshold is
  `0.05/48 ≈ 0.00104` (m = 48, unavailable slots at p = 1), so a slot can reach
  Holm significance only if **all 12 prefix means share one sign**. Scopes with
  fewer eligible prefixes cannot reach it at all. This is a hard floor on this
  corpus, not a defect of the analysis.
- **Contrasts:** arm-vs-R0 (does the change help relative to the MSE-selected
  control?) and arm-vs-persistence (does the arm beat the trivial copy?).

## 7. Pre-training diagnostic (EXPLORATORY, run before freezing)

Run before any training, on the existing A1 checkpoint (read-only), CPU,
validation split. Rationale for the R1/R2/R3 design. The result is preserved
durably in the repo at `docs/realdata-phase2-prediag-20261010.json` (copied
from the ephemeral job scratch path
`C:/Users/Wray/.claude/jobs/a4ff95ef/tmp/realdata-phase2-prediag-20261010/prediag.json`,
which is not a durable location).

The question: why does the trained model fail to reach the trivial copy of its
own lagged-speed input (`X[:, :, 2]`)? Candidate causes were eval-mode sensory
noise, dropout, or the readout path.

Recorded result (A1 checkpoint sha `4a43fb0a…`):

- **Sensory noise and dropout are both OFF in eval** (`sensory_noise_std=0.01`
  and `dropout=0.1` are applied only under `model.training`); neither explains
  the shortfall at inference.
- **The readout does not copy the lagged input.** The output gain on channel 2
  (finite difference of `y_pred` w.r.t. a uniform +1e-3 on channel 2, aligned
  frames) is **mean 0.074, median 0.235** — far from the ~1.0 a copy would
  give. This local input→output sensitivity does **not** separate the readout
  from the encoder; the result is therefore **consistent with a readout-side
  shortfall**, not a causal attribution, and the shortfall is not eval-mode
  sensory noise or dropout.
- A1 aligned (t ≥ 1) MSE **2.366** vs persistence **1.910** (skill −0.239),
  consistent with the preliminary conclusions.

Implication for the design: the model does **not** trivially copy its lagged
input, so the trivial-copy failure is a readout/optimization property, and the
R1 (residual-over-persistence) and R2 (history-ablated) controls are the right
way to separate "learned forecasting" from "input copy". This diagnostic is
exploratory and cannot enter the declared family.

## 8. Biological analysis battery (secondary, descriptive)

These are descriptive and per seed; they cannot enter the declared family.

- **Windowed lesion** (`scripts/simulate_lesion.py`), per seed: replacing the
  GRU vs LIF readout in the onset and sustained windows (step 1's H2 gap). Step
  1 found the GRU readout dominates in both windows; the phase-2 runs test
  whether this replicates across seeds.
- **Latency-based pathway timing**, per seed: first departure > 3 SD from a
  250 ms pre-reference baseline, LIF vs GRU, on the validation split. Step 1
  found the ordering arm-dependent (baseline GRU-first, A1 tie, A2 LIF-first)
  and censoring 29–81%; the phase-2 arms (which change one factor each) test
  whether the flip is attributable to the activation or to a single ablation.
  Censoring is reported per arm.

**Jacobian / fixed-point / dynamics methods are listed for a later phase only**
(§5.2 of the preliminary conclusions: per-condition frozen inputs, slow-point
analysis, local linearisation / Lyapunov exponents, a state space including LIF
state). They are **not** run in phase 2 and are not part of any phase-2 claim.

### 8.1 Pre-stimulus / spontaneous-escape baseline (secondary, descriptive)

**No `no_stimulus` trials are required.** The spontaneous baseline is the
**pre-stimulus segment already inside the ±4.8 s crop** (the `pre_anchor_frames`
= 1200 baseline before the anchor), which is usable now, plus the ISI/ITI
periods. The ISI/ITI periods are **NOT in the processed dataset** and would
require extraction from the raw CSVs / pipeline — a **protected** path that
needs separate user authorization and is **not done in this phase**.

**Declared secondary descriptive metric — spontaneous-escape false-alarm rate.**
For each arm, on the pre-stimulus window of the canonical validation split:

- `model_prestim_false_alarm_frame_rate` — fraction of pre-stimulus frames
  (animal below threshold, `|y_true| < 10 cm/s`) the model predicts `≥ 10 cm/s`;
- `model_prestim_false_alarm_trial_rate` — fraction of pre-stimulus windows in
  which the model predicts an escape bout (a sustained run of `≥ 2` frames at
  `≥ 10 cm/s`, the same `_sustained_run` convention the escape audit uses)
  while the animal is below threshold;
- the animals' **own** pre-stimulus escape rate over the same frames/trials
  (`animal_prestim_escape_frame_rate` / `animal_prestim_escape_trial_rate`), as
  the reference the model is compared against.

This is reported per arm and per seed and **never enters the declared family**.

**Limitations (stated up front).**

- **Post-stimulus after-effects early in the ITI.** Immediately after a
  stimulus the animal's response bleeds into the ITI; either drop the ITI
  head or stratify by time since the last stimulus. Not computed this phase
  (ITI is not in the processed dataset).
- **Anticipation if the ITI is fixed.** A fixed inter-trial interval lets an
  animal anticipate; the fixed-vs-random ITI structure must be checked before
  any anticipatory claim.
- **Behavioural state (resting / grooming / walking).** The pre-stimulus
  segment mixes behavioural states; the false-alarm rate must either control
  for the state distribution or report that distribution alongside it.

## 9. Claim boundaries

Regardless of the numerical outcomes, this phase **cannot** license:

- Animal-level generalization or circuit-level claims (animal identity
  unverified; recording prefixes are not animals).
- An untouched-holdout or test-set claim (validation-guided selection;
  `nested_outer_validation` only).
- A global optimum or the data's absolute ceiling (a bounded four-arm set).
- A causal-inference claim about the MoR router (it is a representational
  routing gate, not a causal estimator).
- Any claim requiring the `no_stimulus` condition (it is no longer a required
  data item, §8.1; the spontaneous baseline is the pre-stimulus segment inside
  the existing crop) or about clock tolerances (unverified).
- Any claim that the model beats persistence **by construction** (R1): an R1
  advantage is the residual skill of a head whose output already contains
  `v_lag` (§4.1) — a positive `S_full` means the residual head adds information
  beyond persistence, never that the model out-forecasts the trivial copy from
  scratch.
- Any adaptive-recursion or signed-target claim (out of scope by user
  decision).
- **Any mechanism interpretation of the router, recursion depth, or
  pathway division** (F1). Phase 2 is the **FOUNDATION layer of a three-layer
  plan** — foundation → biological description → mechanism tests. Phase 2
  establishes only whether a *foundation* (an observed-history model that beats
  persistence in the onset window and at h ≥ 5) exists on this corpus. If no
  arm beats persistence in the onset window and at h ≥ 5, the mechanism
  interpretation is **downgraded**: the router/recursion/pathway story is
  untestable on this corpus and phase 2 licenses no mechanism claim at all. The
  mechanism line — an A2-style adaptive arm, budget-matched K2–K4, a
  symmetric-pathway control, and a "recursion consumes time" variant with a new
  ADR — is a **separate phase-3 pre-registration, frozen before any phase-2
  result is seen**, and is not pre-registered here.

What the outcomes *would* license (preliminary, conditional):

- If an arm beats persistence on a primary scope with a consistent paired
  effect size and a non-null, Holm-adjusted sign-flip p-value: a **preliminary**
  statement that the change is associated with lower observed-history error on
  this corpus's nested outer validation — still incomplete-data, non-animal.
- If R2 collapses to the zero comparator: evidence that the preliminary phase's
  "lower error than baseline" was an input-copy artifact.
- If R3 lowers peak-speed error without hurting MSE: evidence that the jerk term
  was causing the under-prediction.
- Unavailable analyses are reported as unavailable with reasons — never as
  negative results.

## 10. OOM policy (decided now)

Measured on the RTX 5060 Ti (8.52 GB) at the real batch/sequence size (B=128,
T=2400) for A1 (swiglu/off): peak allocated 2.74 GB, peak reserved 3.06 GB.
None of the phase-2 deltas adds memory: R1's skip is a fixed elementwise
residual over existing tensors; R2 zeroes two input channels in place; R3
removes a loss term. OOM is therefore not expected.

Pre-declared policy if a run nevertheless OOMs at batch 128:

1. `scripts/train.py` exposes **no** gradient-accumulation option; changing the
   batch size would change the comparison, which is **not permitted**.
2. The only permitted mitigation is a non-semantic memory setting that keeps the
   effective batch identical (e.g. `PYTORCH_CUDA_ALLOC_CONF` /
   `expandable_segments`) plus a reduced `num_workers`. These change no
   optimizer step and no batch composition.
3. If a run still cannot complete at the identical effective batch, **that run
   is declared unavailable** in every analysis and its family contribution is
   reported null. The comparison is never silently weakened.

## 11. Run order, exact commands, outputs, provenance

**Run order: serial; R0 first**, then R1, R2, R3, each at seeds 42, 43, 44
(12 runs). R0 first fixes the MSE-selected control before any ablation is read.

Training command shape (each of the 12 runs; `--epochs` not overridden — the
config's 300 is authoritative). A run starts fresh, or — only under the
interruption-recovery rule in §11.1 — resumes from its last complete-epoch
checkpoint:

```
python scripts/train.py \
  --config config/realdata-phase2-20261010/R0-control-seed42.yaml \
  --dataset <corpus>/nsmor_dataset.pt \
  --nested_prior_artifact <corpus>/nested_split_seed42.pt
```

Output dirs (one per run, isolated): `.scratch/realdata-phase2-20261010/`
`{R0-control,R1-residual-persistence,R2-history-ablated,R3-jerk-ablated}-seed{42,43,44}/`.
The dataset/nested artifacts are the same paths used by the preliminary phase.

Provenance: each run's checkpoint carries `config.to_dict()`, the additive
top-level keys `selection_metric` and `zero_input_channels` (the script-owned
phase-2 controls, recorded outside the config dict so the schema is
byte-unchanged), the nested-prior artifact path/SHA/fingerprint, split seed,
`dataset_source_sha256` and `animal_identity_status`; the arm's YAML SHA is in
`arm-resolution-diff.json`. Scoring output
`results/realdata-phase2-20261010/realdata_phase2_family.json` records the
protocol path, family size, comparator/scope ids, the `scorer_command`, the
`exchangeability_asserted` flag, the per-arm `h5_input_protocol`, the primary
`skill_vs_persistence` per arm per scope (against the ceiling table, each with
`persistence_ceiling`; the rollout scopes additionally carry the escape-frame
scalars `h5_escape` / `h25_escape` / `h125_escape`, each with its own
`persistence_ceiling`), the `persistence_ceiling` /
`pure_lag_ceiling_reference_only` tables, the
`ceiling_reconciliation` record (per-scope measured-vs-frozen persistence MSE,
`rel 1e-3`, with `ceiling_vs_measured_delta`; the run **fails closed** on a
mismatch, M3), the `secondary_metrics` (h5/h25/h125 skill **overall and escape**
against their matching ceilings, the per-arm peak-speed error with its
`observed_peak_cm_s` scored-split value and provenance, the descriptive
`pooled_mse_t_ge_0`, and the `prestim_spontaneous_escape` false-alarm rates),
the
`seed_level_summary` (per-seed trial mean and per-trial seed sd, for **every**
reported scope), the retained per-seed per-trial MSE vectors
(`seed_trial_mse_vectors`) and every per-slot statistic. Every slot's
`available` flag is derived from the sign-flip primitive's own `p_value`, so
the declared-family slot map and the Holm map agree by construction. A run
whose checkpoint config does not match its declared arm keys — including a
`selection_metric` other than `mse` or an unexpected `zero_input_channels` — is
refused by the scorer (fail closed), and each arm's val loader targets are
asserted elementwise equal to `extract_canonical_validation_targets`' series
before any scoring (fail closed, §5 item 4).

### 11.1 Interruption recovery (amendment 2026-10-10)

The original rule "no `--resume`" is replaced by an interruption-recovery rule.
A run may be interrupted — to free the GPU for same-machine PsychoPy stimulus
timing, or to survive a WSL failure / memory reclaim — and resumed from its
**last complete-epoch checkpoint** only when ALL of the following hold:

1. **Same frozen commit and worktree.** The resuming process runs inside the
   same frozen worktree, at the same frozen commit, with a clean tracked tree;
   `nsmor` and `scripts.train` resolve inside that worktree.
2. **Same config SHA-256.** The run's `provenance.json` records the arm YAML's
   SHA-256; it must equal the current YAML's SHA-256.
3. **Resume preflight passes.** The checkpoint is a current-schema
   (`recovery_state_version=2`) recoverable payload whose lineage, clock,
   optimizer/scheduler/RNG state and phase match the active run; a preflight
   failure refuses the resume (fail closed). The source is the run's LAST
   complete-epoch checkpoint: `scripts/train.py` itself refuses `best_model.pth`
   (the best SELECTED epoch, which may be far older) and, for an in-place
   recovery, any source older than a newer last-epoch family file in the same
   run dir (`_require_resume_source_is_newest_complete_epoch`) — so a bare
   `train.py --resume` enforces this, not only the launcher.
4. **All original logs and recovery records retained.** The interrupted
   segment's `train.stdout.log`, checkpoints and `segment_*.json` records are
   never overwritten or deleted: a fresh segment owns `train.stdout.log`, and
   every resume writes a NEW segment-indexed log (`train.segment<N>.log`)
   rather than truncating the prior segment's. Each resume appends an entry to
   `status.log` and to a `resume_log` in `provenance.json` recording the
   resume time, the checkpoint's epoch and the resume source; `provenance.json`
   is rewritten in place (atomically, temp + fsync + rename) with its
   `resume_log` extended, so its prior fields and prior entries are preserved —
   it is never truncated to a partial or empty file. The `status.log` /
   `resume_log` records are written by the resume LAUNCHER
   (`run_frozen_resume.sh` + `resume_decision.py`); a bare `train.py --resume`
   bypasses the launcher, satisfies the equivalence guarantee, but writes
   neither, so condition 5's per-run interruption reporting is guaranteed only
   on the launcher path.
5. **Results report interruptions.** For every run, the results report the
   number of interruptions, the interruption epochs and the resume sources.

Resume equivalence to uninterrupted training is established at code level
(CPU, fixed seed): model weights, optimizer state, LR scheduler state, NumPy
and Torch RNG states, the per-epoch shuffle order, the early-stopping counter,
the best metric and best-checkpoint record, the `selection_metric` /
`zero_input_channels` provenance and the per-epoch logged losses are all
bitwise equal between an uninterrupted run and an interrupted-then-resumed run
(`tests/test_resume_equivalence.py`). The NumPy stream is EXERCISED, not
vacuous: the anchor-aligned crop the fixture takes does not itself draw from
the global NumPy stream, so each epoch performs one supported stochastic draw;
dropping the NumPy restore then makes the assertion fail (mutation-checked).
The stdlib `random` RNG is deliberately NOT claimed: training never consumes
`random.*` and no checkpoint stores its state, so it is not a
continuation-bearing quantity.

This bitwise guarantee holds in BOTH the reliable-recovery cadence
(`num_workers=0`, `checkpoint_interval=1`) AND the REAL phase-2 arm regime
(`num_workers: -1` auto-scaling to >0 on the full corpus, `persistent_workers:
true`, `checkpoint_interval: 10`), on CPU with a fixed seed. The arm-regime
equality is asserted for `selection_metric=total` and an R2-style
`selection_metric=mse` + `zero_input_channels=[2,3]` config, at both
`num_workers=1` and `num_workers=0`
(`tests/test_resume_equivalence.py::test_arm_regime_resume_is_bitwise`).

The enabling fix targets the exact root cause. A persistent multi-worker
DataLoader builds its iterator ONCE and, at each later epoch, reuses it via
`_reset`, so `iter(loader)` at an epoch boundary draws only the per-epoch
`RandomSampler` seed from the global torch RNG. The FIRST construction instead
draws that seed AND the parent-process base seed, and re-seeds the sampler from
the shifted stream. A resumed process that simply built a fresh iterator
therefore drew one extra global-RNG word (and a different sampler seed), which
shifted the entire resumed trajectory — a genuinely different optimization
path, not a benign re-shuffle. The fix
(`scripts/train.py::_prewarm_persistent_loaders`) re-establishes the persistent
iterator(s) on resume around a save/restore of the global RNG, so the resumed
process consumes the global stream exactly as the uninterrupted run did at that
epoch boundary. It is a pure no-op on the RNG stream (the warm-up draws are
undone) and runs ONLY on a resume, so fresh (non-resumed) training is bitwise
unchanged. Removing the pre-warm reproduces the divergence (mutation-checked:
the `num_workers=1` cases fail, the `num_workers=0` cases stay bitwise).

**Precondition (explicit and ENFORCED): worker-side RNG.** The fix above
restores only the loader's OWN seeds; it cannot recover a draw taken INSIDE a
dataset `__getitem__` OR a `collate_fn` running in a worker process, whose
stream state after k epochs is not in the checkpoint. Bitwise resume is
therefore guaranteed ONLY when the dataset `__getitem__` and collate consume no
worker-side RNG. This is now an enforced precondition, not a caveat: on a
resume `scripts/train.py::_require_bitwise_resume_loader_rng` audits the
actual loaders and REFUSES (fail closed) any run whose dataset would use the
legacy random-crop path (`nsmor/nsmor_dataloader.py` ~L246, `anchor_frames is
None`), any other dataset whose `__getitem__` advances the global torch or
NumPy stream, OR a loader `collate_fn` that advances either stream (all three
via side-effect-free runtime probes — the dataset probe over several indices,
the collate probe on a real clean item). The probes compare the FULL NumPy
state, `has_gauss`/`cached_gaussian` included, so a `standard_normal` draw
served from the Gaussian cache (which leaves `pos`/`keys` unchanged) is
detected rather than missed. The phase-2 corpus provides `anchor_frames` and
its `NSMoRDataset.__getitem__` draws nothing, so the twelve arms satisfy the
precondition; the eager `NSMoRDataset` legacy random crop, a synthetic
RNG-consuming dataset, and an RNG-consuming collate are all covered by tests
(`test_real_corpus_dataset_consumes_no_worker_rng`,
`test_resume_refused_when_dataset_consumes_worker_rng`,
`test_resume_refused_when_collate_consumes_worker_rng`,
`test_rng_probe_detects_gaussian_cache_draw`). A resume whose loaders
fail this precondition is refused before any epoch rather than silently
diverging.

On GPU the guarantee is **not** bitwise: the training code sets NO determinism
flags (`torch.use_deterministic_algorithms`, `cudnn.deterministic` and
`CUBLAS_WORKSPACE_CONFIG` are all unset), so cuDNN/cuBLAS reductions and
atomics (and non-deterministic kernel selection) make a resumed and an
uninterrupted run differ by GPU-reduction rounding — exactly as two
uninterrupted GPU runs differ. No tolerance is measured or claimed for this
code, so a GPU resume is **explicitly unsupported** for any phase-2 arm: it is
reported as state-exact and trajectory-unverified, and must NOT be used to
produce a phase-2 result. What IS exact on GPU is the restored continuation
STATE (model/optimizer/scheduler/RNG bytes); everything downstream of it is
unverified. For the twelve GPU arms the bitwise claim is therefore carried by
the LOGICAL guard, not by a measured GPU trajectory: the pre-warm runs only
when `checkpoint.resume_from` is set, so a fresh GPU run (no `--resume`) never
touches the resume seam and is unchanged; the GPU arm's resume is state-exact
only.

The bitwise measurements above are on CPU pinned to a SINGLE thread
(`torch.set_num_threads(1)`, interop threads 1, CUDA hidden —
`tests/test_resume_equivalence.py::_cpu_single_thread`). Thread count is part
of the reproducibility contract: multi-threaded CPU reductions (and any GPU
kernel) are not bitwise, so the claim is scoped to the thread-pinned CPU the
tests pin. The equality is measured on a tiny SYNTHETIC corpus and a SHORT run
(4-6 epochs) — an emulation of the arm regime's loader configuration
(`num_workers>0`, `persistent_workers=true`, `checkpoint_interval=10`), not the
full corpus at 300 epochs; the arm-regime equality rests on the loader
configuration being exercised, which the emulation reproduces exactly. A
SIGKILL at a non-multiple epoch (which writes no `final_model.pth`, only an
older periodic) is covered by the `final_model.pth`/newest-complete-epoch UNIT
tests and the launcher `decide` tests, NOT end-to-end: the equivalence tests
interrupt via a safe pause (which writes `final_model.pth`) or a
checkpointed-epoch exception, and the newest-source rule itself is unit-tested.

`seed 42` (`R0-control-seed42`) was started from the LIVE tree
(`D:/Projects/NSMoR`) by `run_arms.sh`, NOT the frozen worktree, so its
provenance head differs from the frozen commit. It must NOT be resumed across
code versions: the resume launcher's provenance check refuses it. If it is
interrupted it is a deviation — rerun it from the frozen worktree, or disclose
it as interrupted — and it is not silently continued on the new code.

The safe-pause mechanism (a `STOP` sentinel file in the run's output dir,
checked only after the epoch's checkpoint is fully written via temp + fsync +
atomic rename) never interrupts a save, and a paused run exits with the
distinct status `17` (`train.py`'s `PAUSE_EXIT_CODE`) so a launcher can tell a
pause from a completion (`0`) or a failure (nonzero) from the exit code alone.
The preliminary-phase conclusions and
the k1.0 recovery segment/gaps remain exactly as recorded; this amendment
changes only the run-order/resume rule.

**Amendment log — 2026-10-10.** Reason: make the already-implemented epoch
recovery (commit `56253c0`; 2026-10-06 authorization) a formal mechanism so
training can be paused for same-machine PsychoPy experiments and survive WSL
failures, instead of the blanket "no `--resume`". Equivalence evidence:
`tests/test_resume_equivalence.py` (CPU, seed 42, tiny synthetic data) proves
bitwise equality of every continuation-bearing quantity between an
uninterrupted run and an interrupted-then-resumed run, under
`selection_metric=total` and an R2-style `selection_metric=mse` +
`zero_input_channels=[2,3]` config, in the reliable cadence
(`num_workers=0`, `checkpoint_interval=1`) AND the real arm regime
(`num_workers=1`, `persistent_workers=True`, `checkpoint_interval=10`,
`test_arm_regime_resume_is_bitwise`); a resume that switches `selection_metric`
is refused (fail closed). The arm-regime equality rests on
`_prewarm_persistent_loaders` (re-establish the persistent iterator around an
RNG save/restore on resume), mutation-checked to be load-bearing. The
stdlib-Python RNG dimension the task enumerated is NOT asserted because
training never consumes `random.*` (no such call in `scripts/train.py` or
`nsmor/*.py`) and no checkpoint stores its state, so it is not a
continuation-bearing quantity; CUDA RNG state (`cuda_rng_state`) is likewise
not asserted, since the claim is CPU-only and no GPU is used. Fresh-training
equivalence is preserved AND regression-pinned:
`test_fresh_training_weights_are_unchanged` asserts the exact final-weights
SHA-256 of a fresh run, measured identical on the pre-amendment base commit
(8ed5edb) and on the amended code in the arm regime (workers=1, persistent,
interval=10) as well (no default behaviour changed).

**Amendment log — 2026-10-11.** Reason: the 2026-10-10 text scoped the bitwise
guarantee to `num_workers=0` and disclosed that the arms resume "approximately".
That disclosure was a bug report, not a scientific limit: the divergence had a
single, fixable root cause (a fresh loader iterator re-drawing the base seed and
re-seeding the sampler on resume). `_prewarm_persistent_loaders` fixes it, so
the arm regime is now a bitwise continuation and the "`resume == uninterrupted`
FALSE" headline is RETRACTED — `resume == uninterrupted` is TRUE (bitwise, CPU,
fixed seed) for the twelve phase-2 arms' loader configuration. The GPU statement
is unchanged: state-exact, trajectory not bitwise, same as two uninterrupted GPU
runs. The earlier characterization test that asserted the arms diverge is
replaced by `test_arm_regime_resume_is_bitwise`.

**Limitations of this amendment (first-class, not buried).** The bitwise result
is CPU-only and requires a fixed seed; on GPU it is state-exact but not bitwise
(no determinism flags), exactly as for two uninterrupted GPU runs — a GPU
resume is reported as state-exact and trajectory-unverified and must NOT be used
to produce a phase-2 result. The `seed 42` arm is a cross-code-version deviation
(started from the live tree) and is never silently resumed. `final_model.pth`
is an accepted resume source: the safe-pause and the recovery helper both treat
it as the segment's last complete-epoch checkpoint, and the helper orders resume
candidates by the checkpoint's OWN stored epoch across the last-epoch families
(never by family, and NEVER from `best_model.pth` — the best epoch can be far
older than the last executed epoch, so resuming from it would silently re-train
completed epochs; a dir holding only `best_model.pth` is refused), so an
interruption at a non-multiple of `checkpoint_interval` recovers the NEWEST
complete epoch rather than an older periodic — for `checkpoint_interval: 10`, a
pause at epoch 12 resumes from `final_model.pth` (epoch 12), not `epoch_10.pth`
(epoch 9).

**Amendment log — 2026-10-11 (round-4).** Reason: close five review findings.
(1) G1: the worker-RNG precondition above is now explicit and enforced
(`_require_bitwise_resume_loader_rng`), with tests for both a detected
RNG-consuming dataset and the real-corpus dataset's no-RNG property. (2) G2:
`train.py` itself now refuses `best_model.pth` as a resume source and refuses a
source older than a newer last-epoch family file in its own run dir
(`_require_resume_source_is_newest_complete_epoch`), so a bare `--resume`
enforces §11.1's "last complete-epoch checkpoint" rule, not only the launcher.
(3) G3: the stale loader-control claims are corrected — the segment record now
carries `worker_resume_equivalence_established: true` (the persistent
multi-worker resume IS a bitwise continuation) and
`worker_count_change_equivalence_established: false` (a resume that CHANGES the
worker count is not), replacing the conflated
`legacy_worker_equivalence_established: false`; the
`_require_reliable_recovery_loaders` docstring no longer claims workers>0 cannot
be resumed. (4) G4: the arm-regime equivalence test now uses TWO train batches
per epoch (`batch_size=4` on the 8-trial split) and asserts the per-epoch
`RandomSampler` index order parent-side (observable at any worker count) in
addition to weights. (5) G5: this section now states the CPU-single-thread
scope, the GPU logical-guard basis, the synthetic-corpus/short-run emulation,
and that a SIGKILL at a non-multiple epoch is covered by unit tests, not
end-to-end.

**Amendment log — 2026-10-11 (round-5).** Reason: close the round-5 review
findings on the round-4 fixes. (1) G1 (blocker): the enforced worker-RNG
precondition now actually covers the loader's `collate_fn` — a collate that
draws global torch/NumPy RNG in a worker is detected and refused
(`_collate_fn_consumes_rng`), so the code matches the "dataset `__getitem__`
AND collate" guarantee the text already stated. (2) Major: the dataset probe's
NumPy comparison now checks the FULL state (`has_gauss`/`cached_gaussian`
included), closing a false negative where a `standard_normal` draw served from
the Gaussian cache left `pos`/`keys` unchanged; the probe also runs over more
than one index, so an index-dependent draw (e.g. a crop that fires only on long
trials) is not missed by an index-0-only probe. (3) Major: the real-corpus
no-RNG test now sets `max_seq_len` strictly below the fixture sequence length
and asserts the anchor-crop branch actually executes, so the property is
asserted on the code path it names. (4) Major: the truncated duplicate G2
comment banner at EOF of `tests/test_epoch_recovery.py` is removed, and the
dead `_LAST_EPOCH_FAMILIES` constant is now the single source of truth used by
`_require_resume_source_is_newest_complete_epoch` (code and constant agree).
The launcher (`run_frozen_resume.sh`) precondition comment states the same
scope. A new test covers the collate hazard, the Gaussian-cache draw, and the
executed crop branch (`tests/test_resume_equivalence.py`).

*This amendment's text is bound to the reviewed worktree commit by the
`sha256` of `docs/realdata-phase2-protocol-20261010.md` recorded in each run's
`provenance.json`; that hash is left to the review commit and is not
hand-written here.*

## 12. Release boundary

No release is authorized by this document alone. Any PR stages explicit
reviewed paths only (never `git add -A`), preserves all formal
DATA/NEST/checkpoints/receipts and protected untracked paths, and requires
independent review plus the standard test gates. This protocol changes only
`scripts/`, `config/`, `tests/` and `docs/`; `nsmor/model_nsmor_core.py`,
`nsmor/loss.py`, the protected `nsmor/config_parser.py` config schema, the data
pipeline and the data/checkpoint contracts are untouched.
