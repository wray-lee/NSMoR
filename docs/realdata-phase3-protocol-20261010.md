# Real-data phase-3 prospective protocol (mechanism line) — 2026-10-10

Status: **PROSPECTIVE**. Frozen before any phase-2 result is seen. Every arm,
mode, metric, comparator, analysis, decision rule and claim boundary below is
fixed in advance. No retrospective enlargement is permitted; any further
experiment requires a new prospective protocol.

Companion documents: `docs/realdata-phase2-protocol-20261010.md` (the
**foundation** this phase depends on), `docs/realdata-preliminary-conclusions-
20261009.md`, `docs/realdata-exploratory-step1-20261010.md` (including its
read-only setup audit), `docs/behavior-exploratory-20261010.md` (model-free
behavior), and `docs/adr/0008-adaptive-latent-recursion.md` /
`docs/adr/0009-time-consuming-recursion.md`.

Phase 3 is the **mechanism layer** of the three-layer plan (foundation →
biological description → mechanism tests). It asks whether the model's
*processing depth / division of labor* can be compared with the animal's
**response latency** and its variability, and whether a candidate architecture
reproduces the observed behavioral signatures — not whether it lowers MSE.

## 1. Scope and immutable evidence

- Engineering mechanism modeling on the existing real corpus only. No new data
  collection, no synthetic data, no animal-level inference.
- Dataset SHA-256:
  `b1bd5578025fb5eaa6f4eb3e355c097b6576d44dbedaa001027f1de5447e06b2`.
- Nested-prior SHA-256:
  `f25856304a5857695945bf5083ca54745f47fe5dffe915111f6ba28d524ddf94`.
- Nested seed 42; validation fraction 0.2; evaluation scope is
  `nested_outer_validation`, **not** an untouched holdout. Corpus: 2304
  labelled trials (multisensory 1188, visual_only 720, wind_only 396; **no**
  no_stimulus) + 128 unlabelled no_stimulus anchors; 64 recording prefixes
  (52 train / 12 val); **432 validation trials, 12 recording prefixes,
  1,036,748 frames**. `animal_identity_status=unverified`. Recording prefixes
  are not animals.
- **Target is scalar speed**, unchanged by user decision. 80.4% of validation
  frames are exactly 0; 2.53% are ≥ 10 cm/s. No signed-velocity / heading
  target is introduced.
- The frozen baseline, the A1 arm and the A2 arm are **read-only** (SHAs as in
  the preliminary conclusions); they are reused for diagnostics, never
  retrained, resumed or overwritten.
- Phase-2 outputs (`.scratch/realdata-phase2-20261010/`, `config/realdata-
  phase2-20261010/`, `docs/realdata-phase2-*`,
  `scripts/evaluate_realdata_phase2.py`, `scripts/realdata_phase2_configs.py`)
  are **read-only** to phase 3.

### 1.1 Model-free behavioral signatures (frozen targets for R6)

From `docs/behavior-exploratory-20261010.md` (in-sample, between-group,
model-free; condition == recording prefix):

- **Race violation / facilitation.** Pooled violation area 389.4 prob·ms above
  the prefix-relabeling null mean 51.6; raw one-sided p = 0.0015, **Holm
  p = 0.0075** over the declared 5-test family (pooled + 4 TTC variants); three
  of five survive at 0.05 (pooled 0.0075, TTC −308 0.015, TTC −261 0.010).
  Under D1 the visual term is small (**F_V 0.016**, 8 visual responders), so
  the violation is **AV-vs-A facilitation**, not a clean Miller coactivation
  excess.
- **Latency facilitation.** Multisensory median 84 ms vs wind_only 120 ms;
  per-prefix mean difference **+38.4 ms** (Welch t = 8.53, permutation
  p = 0.0002, Holm 0.0010, Cliff's delta 0.95, Cohen's d 2.53).
- **Latency shape.** wind_only ex-Gaussian best (mean 126.1, SD 33.2, skew
  1.76, tau 28.7 ms); multisensory ex-Gaussian best (mean 93.2, SD 69.2, skew
  13.2, tau 30.2 ms). Mean-SD trend across 4 TTC variants rho 0.80
  (**suggestive**, 4 points).
- **Wind-locked trigger.** Escape in multisensory is wind-locked (median
  escape − collision −264 to −120 ms across variants); the l/|v| proxy is a
  kinematic identity here, so an LGMD/DCMD-like fixed-angular-size threshold
  is **not identifiable**.
- **Confounds (stated up front).** Condition is perfectly nested in prefix (all
  64 prefixes condition-pure); animal identity is unverified; there is no
  within-animal TTC / looming variation; visual alone rarely triggers escape.

## 2. User decisions (frozen)

- **The mechanism line is gated on the phase-2 foundation (R7, §3).**
- **Budget must be approved before training** (§10). No run starts until the
  user approves the arm × seed matrix.
- The scalar speed target, the data contract, the data pipeline and the
  data/checkpoint contracts are untouched. ISI/ITI extraction is **not**
  authorized.
- All core changes follow `nsmor/BOUNDARY.md`'s controlled-change protocol:
  every new behavior is an **opt-in mode**, defaults and existing
  checkpoints/historical experiment semantics byte-unchanged, shape assertions
  kept, focused + complete regression tests, numerical safety checks.

## 3. R7 — Foundation dependency and the downgrade decision rule

Mechanism interpretation is **conditional on the phase-2 foundation**. The
phase-2 primary metric is skill vs persistence = `1 − MSE_model / MSE_persist`
(negative = worse than persistence), reported per arm per scope (aligned,
escape, h = 5, onset) against the measured persistence ceiling.

**Frozen decision rule (evaluated once, before any phase-3 training):**

**Sign convention (stated once, used everywhere below).** Skill vs persistence
is `skill = 1 − MSE_model / MSE_persist` (positive = better than persistence).
The paired trial-level effect size is `d_z = mean(candidate − comparator) / sd`
with the arm as candidate and the persistence prediction as comparator. Because
a LOWER MSE is better, a skill advantage corresponds to a NEGATIVE mean paired
difference (the arm's per-trial error is smaller), so `d_z < 0`. Beating
persistence therefore requires **both** `skill > 0` **and** `d_z < 0`.

Let *onset-beat* = some phase-2 arm has a positive skill vs persistence in the
onset window `[onset, response_onset + 200 ms)`; and *h5-beat* = some phase-2
arm has a positive skill vs persistence at h = 5 (held-at-start ceiling),
**with a non-null, Holm-adjusted prefix sign-flip p-value** on the same arm and
scope.

- **Scope set (frozen).** The gate is evaluated over exactly the two phase-2
  scopes `{onset, h5}` of the phase-2 family (`aligned_pooled` and `escape`
  are **not** gate scopes). The arm set is the **four trained phase-2 arms**
  `{R0, R1, R2, R3}`; `persistence` and `zero` are comparators, never
  candidates.
- **Numeric thresholds (frozen).** *onset-beat* requires, for the same arm and
  the `onset` scope, all of: `skill_vs_persistence > 0` **and** the paired
  trial-level Cohen's `d_z < 0` (the sign convention above) **and** the
  Holm-adjusted prefix sign-flip `p <= 0.05`. *h5-beat* requires the same three
  conditions at the `h5` scope. The `skill > 0`, the `d_z < 0` and the
  Holm-adjusted p must be on the **same arm and the same scope** (a positive
  `skill` at `onset` with a null `p` at `h5`, or a positive `skill` with a
  non-negative `d_z`, does **not** satisfy the rule).
- **Holm family for the gate (frozen).** The gate reads the **already-frozen
  phase-2 family** (`m = 48`, §6 of the phase-2 protocol), not a new family:
  the per-scope Holm-adjusted p is taken verbatim from the frozen phase-2
  output, so no new multiplicity correction is introduced here.

- If **onset-beat AND h5-beat**: the foundation exists; phase-3 mechanism
  conclusions may be stated as **preliminary, conditional** mechanism claims
  within §12.
- If **NOT (onset-beat AND h5-beat)**: the foundation is absent; every phase-3
  mechanism conclusion is **downgraded** to a purely descriptive,
  exploratory statement. The router / recursion / pathway story is then
  **untestable on this corpus**, and phase 3 licenses **no** mechanism claim.
  The phase-3 runs may still be reported, but only as descriptive
  computational-sufficiency evidence.

The rule is evaluated from the frozen phase-2 scoring output
(`results/realdata-phase2-20261010/realdata_phase2_family.json`); it is never
re-evaluated after seeing a phase-3 outcome.

## 4. R1 — Recursion that consumes time (ADR 0009)

An opt-in mode in which processing depth is tied to output timing: each extra
recursion step **delays the emitted frame** by one model-grid step, so model
depth becomes directly comparable with the animal's latency and its
variability. Semantics, causality, alignment/scoring and identifiability are
normative in `docs/adr/0009-time-consuming-recursion.md`. Summary:

- `time_consuming_mode: "off" | "depth_delay" | "accumulate"`, default `"off"`
  (module absent; historical numerics / `state_dict` / RNG byte-unchanged).
- **`depth_delay`** — adaptive halt depth `d = min{k : cumsum_{j<=k}
  sigmoid(halt(u_j)) >= 1 - eps}` (else `K`) sets the delay `d - 1`.
- **`accumulate`** — a shared scalar accumulator integrates a learned gate
  `sigmoid(halt(u_k))`; the frame is emitted at the first step `d` at which the
  accumulator reaches `threshold` (else `K`).
- **Causality.** Source frame `t` is written only to `t + delay`
  (`delay = delay_scale * (depth - 1) >= 0`); no future information leaks
  backward. The map is **not injective** (a deep source can overtake a shallow
  one), so two sources can target the same frame; per-target emissions are
  arbitrated deterministically (smallest delay wins, earliest source breaks
  ties). The two ways a source can be dropped are **distinct and named**
  (matching the ADR and the code):
  - a source whose **target falls outside the trial** (past the trial tail) is
    **truncated** — `internals["time_consuming_truncated"]`; the effective
    output length shrinks by up to `K - 1` frames;
  - a source whose target is **in-trial but loses per-target arbitration** is a
    **collision** — `internals["time_consuming_collisions"]` (it is NOT a
    truncation).
  Padding never recurses (never emitted). Both counts are reported per arm.
  Causality holds at the **scored** quantity: the delayed emission
  (`internals["time_consuming_hidden"]`) is what the direction head decodes and
  the loss scores, so perturbing `x[t+k]` cannot change any emitted frame at or
  before `t`.
- **Alignment / scoring.** Scored on `internals["out_valid"]` frames only
  (`BioJointLoss(out_valid=...)`), so the zero-filled, never-emitted trailing
  frames cannot bias the MSE; latency uses `internals["realized_delay"]`
  (post-arbitration, indexed by **target** frame; §4.1). A metric never compares
  a truncated and a full-length window (the protocol fixes the aligned window
  per arm), and the truncation and collision counts are reported per arm.
- **Dropped head and tail (frozen, restated precisely).** The strictly causal
  shift `t → t + delay(t)` maps each source onto a target; the emitted set is
  exactly `{j : some source t = j − delay(t) ≥ 0 wins the per-target
  arbitration for j}`. It is **not** true that the leading frames can never
  receive a source: a frame `j < max_delay` is emitted **iff** some source
  `t = j − delay(t) ≥ 0` targets it and wins arbitration, which happens
  whenever a shallow (small-delay) source reaches that frame. The only frames
  the code **proves** unemitted are (a) the **trailing** frames whose every
  targeting source falls past the trial tail (truncated, §4), and (b) interior
  frames that are the target of no valid source. No fixed leading-frame count
  is asserted unemitted; the emitted set is read directly from `out_valid`.
  - **Worst-case bound vs observed max (stated separately).** The worst-case
    delay an arm can produce is `delay_scale · (K − 1) = 30` frames at `K = 4`,
    `delay_scale = 10.0` (§4.1); this bounds the **delay**, not the dropped
    head. The per-batch `internals["time_consuming_max_delay"]` is the observed
    maximum over the emitted sources of that batch and may be smaller.
  - **Stimulus-anchor guarantee (frozen).** The stimulus anchor is the crop's
    **index 1200** (`pre_anchor_frames = 1200`;
    `nsmor/pipeline/conditions.resolve_anchor_crop`); the anchor's **saved**
    index is `1200` when the window is not clamped, else the trial's own anchor
    index (when `anchor_frame < 1200` the window starts at 0). The scorer
    **asserts** that the stimulus anchor frame is **emitted** (`out_valid` true
    at the anchor's saved index) and, if it is not emitted for any trial,
    reports that trial **unavailable** rather than silently scoring an
    unemitted frame. This is an emitted-frame assertion, **not** a head-drop
    claim: it does **not** assert that every frame before the anchor is
    unemitted. The anchor is emitted **iff** some in-trial source targets it:
    the earliest possible targeting source is `1200 − 30 = 1170` (the maximum
    delay `delay_scale·(K−1) = 30`, §4.1), which lies at index `1170 ≫ 0` and is
    therefore always in-trial, so the anchor is **never excluded by
    truncation**. Whether a *particular* trial's delay profile actually targets
    it is read from `out_valid` (a varying profile can leave an interior frame
    untargeted, above); the assertion fails closed on any trial whose anchor is
    not emitted.
  - **R6 latency conditioning (stated).** The §4.1 per-trial mean
    `realized_delay` is conditioned on `out_valid` frames only, so the dropped
    head/tail never enter the latency statistic; it **excludes truncated /
    unemitted targets** (a target with no surviving source carries
    `realized_delay = 0` and is not in `out_valid`, so it is never averaged);
    the `out_valid` frame count per trial is reported beside it.
- **Scored-target convention (frozen).** The delayed emission at frame
  `t + delay` (computed from sources up to `t`) is scored against the target
  `y[t + delay]` at that **emission** frame, not against `y[t]`. The model is
  therefore a **delay-ahead forecaster**: it predicts the target `delay` frames
  beyond its newest input, which is exactly what "the response arrives `delay`
  frames after the source" means on a per-frame regression target. This
  convention is the same one the phase-2 aligned scope uses (score the frame's
  own target), so the `out_valid`-masked MSE is directly comparable to the
  phase-2 skill; the `delay`-frame forecast handicap is stated, not hidden.
  The R1 depth→latency statistic (§4.1) is unaffected by this convention
  because it reads `realized_delay`, not the target.
- **Identifiability.** "Stop after `k` recursions" and "continuous accumulation
  to a threshold" are nearly indistinguishable; the strongest licensable claim
  is **"an adaptive processing duration exists; MoR is one discrete
  approximation"** — never "the cricket has MoR" (§12).
- The module is a **model-grid timing device**, NOT measured neural latency,
  NOT work/MAC, NOT ATP.

### 4.1 Pre-registered delay scale and step → ms conversion (frozen)

The default `time_consuming_delay_scale = 1.0` yields a `K = 4` maximum delay
of `3` grid steps = **12 ms** at `dt_ms = 4.0` (the real-data configs use
`dt_ms = 4.0`), an order of magnitude below the observed `wind_only` median
latency (120 ms) and the multisensory median (84 ms). The primary "compare
model depth with animal latency" statistic would therefore be **vacuous** at
the default scale. To make the delay distribution comparable to the observed
latencies we **pre-register** `time_consuming_delay_scale = 10.0` for every
time-consuming phase-3 arm, giving a `K = 4` maximum delay of `30` steps =
**120 ms**, spanning the observed `[84, 120]` ms latency range at the top of
the range. Because the delayed emission is the scored quantity, the scale
changes the training loss, so **train and score must use the same scale**: the
phase-3 arm configs for `C1`/`C2` (§13 prerequisite, emitted by
`scripts/train_phase3_arms.py`) **carry `time_consuming_delay_scale = 10.0`**,
and `scripts/evaluate_realdata_phase3.py` **asserts** the loaded checkpoint's
recorded `time_consuming_delay_scale` is `10.0` and **fails closed otherwise**
(never silently scoring a checkpoint trained at a different scale). The
`config/default.yaml` default stays `1.0` so the `off` default is
byte-unchanged. If the arm configs cannot be emitted with `10.0`, the affected
`C1`/`C2` runs are **not** scored (the value is never applied by the scorer
alone to a checkpoint trained at another scale).
**Where `10.0` lives (frozen, stated explicitly).** This phase-3 document
ships **no** arm config file carrying `10.0`: the value is applied **only** by
the deferred arm-config generator (`scripts/train_phase3_arms.py`, §13
prerequisite, not present in the tree) at training time and **asserted** by the
deferred scorer (`scripts/evaluate_realdata_phase3.py`) at scoring time. Until
those scripts exist, the live `config/default.yaml` default (`1.0`) is the only
scale present, and no `C1`/`C2` run exists to score.

- **Step → ms mapping (frozen):** `latency_ms = delay_frames * dt_ms` with
  `dt_ms = config.model.dt_ms` (4.0 ms for this corpus). No other conversion is
  licensed; a delay in frames is never reported as a latency without this
  conversion.
- **Modeled latency (frozen):** `latency_ms = (delay_scale * (depth - 1)) *
  dt_ms`, reported per arm per seed. The delay is measured **relative to the
  source frame** whose value the emission carries (the causal anchor: source
  `t` emits at `t + delay`), so **no additive frame-offset term is
  pre-registered**; the frozen formula is the offset-free one above, and the
  ADR uses the identical expression. A source→scored-target frame offset is
  **not** part of the latency statistic.
- **Pre-registered R6 delay statistic (frozen, pinned here).** The headline
  "model depth vs animal latency" statistic uses the **post-arbitration
  realized delay** `internals["realized_delay"]` — the delay of the *surviving*
  source, indexed by TARGET frame — **not** the raw per-source
  `internals["output_delay"]`. `realized_delay` is exposed for exactly this
  reason: the depth→delay map is **non-injective** (a deep source can overtake a
  shallow one), so a raw per-source delay would double-count collided targets
  and report a delay the model never emitted. The statistic is the per-trial
  mean of `realized_delay` over `out_valid` frames (each surviving target
  weighted equally), reported per arm per seed; the collision and truncation
  counts (§4) are reported beside it. The raw per-source `output_delay` may be
  reported **descriptively only** and never enters the family (§11).
- **Support limit (frozen, stated honestly).** With `K = 4` the integer depth
  is in `[1, 4]`, so the modeled delay support is **at most four atoms**
  `{0, 10, 20, 30}` frames = `{0, 40, 80, 120}` ms at `delay_scale = 10.0`. A
  ≤ 4-atom distribution **cannot** reproduce a continuous ex-Gaussian shape.
  The continuous "match the observed mean/SD/tau" comparison (§1.1) is
  therefore **declared untestable on this corpus and is descriptive only**; it
  is **not** a scored signature. The **scored** latency statistic is a
  **quantile/support** comparison: per condition, the modeled latency median and
  inter-quartile interval (from the ≤ 4 atoms) are compared against the observed
  median and IQR with the same per-prefix cluster resampling as §11, and the
  modeled support is reported. A ≤ 4-atom model can falsify this statistic
  (its quantiles are pinned to the atoms); it cannot falsify a continuous shape.
- **Determinism limit (frozen, stated honestly).** Depth is a deterministic
  function of the fused latent. All 64 prefixes are condition-pure (§1.1), so
  within a condition the fused latent varies only through the per-trial input
  and the modeled **per-trial latency SD is ≈ 0** for a single fixed-depth
  mode. The pre-registered latency statistic is therefore compared as a
  **central tendency / support** statistic, never as a within-condition spread
  match; the observed within-condition SDs (33.2 / 69.2 ms) are reported beside
  it as a **stated non-match**, not a failure of the model. No per-trial
  stochastic driver is introduced (that would be a separate mechanism under a
  fresh protocol); if a later protocol adds one, its tolerance must be declared
  before any run.
- **Caveat (stated).** The delay scale is a device-to-physical-time calibration
  choice; it fixes comparability of the units, it does not evidence that the
  model's depth mechanism is the animal's. Model depth is **rank-comparable**
  with the animal latency distribution under this mapping, not identical to it.

## 5. R2 — Budget-matched comparison (frozen rule)

Fixed depth `K2`/`K3`/`K4` vs adaptive vs the time-consuming variant are matched
on **per-token refinement work**, extending the rule of
`scripts/final_control_harness.py` to the module that actually runs:

**One per-step cost model (frozen, used by every arm).** The costed quantity is
the per-valid-token refinement work, exactly the forward-hook identity of
`scripts/final_control_harness.py`:

    per_token_work = (block_rows * BLOCK + halt_rows * HALT) / total_valid_tokens

with `BLOCK = 2·H²` (fc1 + fc2, `H = 64` → 8192) and `HALT = H` (→ 64). A step
costs `BLOCK` (the shared block always runs) **plus** `HALT` **only when the
halt head executes on that step**. `block_rows` / `halt_rows` are the
**ACTUAL executed rows of the executed module**, asserted against forward hooks
on that module's own `fc1`/`fc2` (block) and `halt` (halt) — NOT on
`backend.refinement`, which is `None` for a time-consuming arm. The three arm
classes instantiate the one model as:

- **fixed refinement arms** (`C3`/`C4`/`C5`, `K` steps, no early exit, halt head
  **not executed**): `block_rows = valid·K`, `halt_rows = 0` →
  `per_token = K·BLOCK`.
- **adaptive arm** (`C6`, halt head executed once per active row): `block_rows =
  halt_rows = sum(N) = N̄·valid` → `per_token = N̄·(BLOCK + HALT)`.
- **time-consuming arms** (`C1`/`C2`, the module runs the block **and** the halt
  head on every one of its `K` steps for every valid row, no early exit):
  `block_rows = halt_rows = time_consuming_updates = valid·K` →
  `per_token = K·(BLOCK + HALT)`.

- **Implemented-module requirement (frozen).** `scripts/final_control_harness.py`
  costs `model.backend.refinement`, which is `None` for a time-consuming arm, so
  the budget-match hooks must be registered on the **executed** module's own
  `fc1`/`fc2`/`halt`. The delivered `TimeConsumingRecursion` has those leaves, so
  the R2 verdict is implementable; the phase-3 scorer
  `scripts/evaluate_realdata_phase3.py` (§13) registers them. Until that scorer
  exists, **no phase-3 budget verdict is claimed** — the R2 contrast stays
  `unavailable` (never a fabricated MATCHED).
- **Expected per-token cost (declared before any run, from the one model
  above).** With `H = 64`, `BLOCK = 8192`, `HALT = 64`, `BLOCK + HALT = 8256`:
  fixed at `K` → `K·8192` (`C3` 16384, `C4` 24576, `C5` 32768); time-consuming
  at `K` → `K·8256` (`C1`/`C2` 33024); adaptive → `N̄·8256` with `N̄` the measured
  mean per-row depth in `[1, K]`.
- **Tolerance (frozen, imported from the harness constant).** `tol =
  MATCH_TOL_FRACTION · (BLOCK + HALT)` with `MATCH_TOL_FRACTION = 0.5`
  (`scripts/final_control_harness.py:50`), i.e. `0.5·8256 = 4128` MAC/token =
  **±0.5 of one block+halt step**. The constant and the `(BLOCK + HALT)`
  reference step are the harness's, not redefined here. A pair must be within
  the band **per seed**.
- **Adaptive swing, stated.** Because `N̄ ∈ [1, K]`, the adaptive arm's
  per-token cost swings over the full `[1, K]·(BLOCK + HALT)` range, while the
  band is ±0.5 step. An adaptive arm is therefore `MATCHED` to a fixed arm at
  `K_f` **only if the measured `N̄` lands within ±0.5 of `K_f`**. This is a
  **measured outcome, not an assumption**; `N̄` is reported per seed and the
  `MATCHED`/`NOTMATCHED` verdict follows. A `N̄` landing between two fixed arms
  is `NOTMATCHED` **by construction** and the affected contrast is descriptive
  only (this is exactly the prior architecture-v1 outcome). The protocol does
  **not** tune the band or the arms to force `MATCHED`.
- **Predeclared nearest-arm rule (pinned).** The predeclared fixed arm is the
  one with the per-token work nearest to the arm under test, decided by
  `K_f = round(N̄)`; **when two fixed arms are equidistant the lower-`K` arm is
  predeclared**, fixed here before any run. The rule is stated in per-token
  work, so it is well-defined even when the fixed arms are not ordered by work.
- **Unmatched policy (frozen).** A `NOTMATCHED` verdict **forbids any
  budget-advantage or mechanism-condition claim** for that pair; the band is
  fixed a priori and is **never** tuned to force `MATCHED`. The prior
  architecture-v1 control was `NOTMATCHED`; phase 3 reports the verdict and, if
  unmatched, downgrades the affected contrast to descriptive.
- **Plumbing note.** `scripts/final_control_harness.py` costs
  `model.backend.refinement` and has no time-consuming variant; the phase-3
  scorer `scripts/evaluate_realdata_phase3.py` (§13) implements this extended
  rule against the executed module's hooks. No phase-3 budget verdict is
  issued by the phase-2 harness.

## 6. R3 — Symmetric-pathway control (fast/slow by response latency)

To test whether a fast/slow division emerges **spontaneously** (rather than
being imposed by the LIF+GRU architecture), the control is a **symmetric**
model:

- **`homologous`**: two pathways of the SAME type with **learnable time
  constants** and independent weights — either GRU+GRU or LIF+LIF — so no
  built-in type asymmetry exists.
- **Fast/slow is defined by RESPONSE LATENCY**, not leak constants. The
  latency measure is the first departure > 3 SD from a 250 ms pre-reference
  baseline (the step-1 latency method), reported per pathway with its
  censoring fraction.
- **Cross-seed stability is required**: the fast/slow ordering must hold in
  every seed and the prefix-cluster sign-flip must be non-null; a single-seed
  ordering is descriptive only.
- Rationale from the preliminary findings: by leak time constant the GRU
  (≈ 3–5 ms) is *faster* than the LIF (≈ 95 ms), and the latency ordering is
  arm-dependent (baseline GRU-first, A1 tie, A2 LIF-first). "LIF is the fast
  path" is therefore **not** assumed and must be re-derived from latency.

**Implementation status (frozen).** The minimal opt-in implementation delivered
by this protocol is the R1 time-consuming mode (§4). The symmetric-pathway mode
is **explicitly scoped out of this frozen protocol** rather than left as a
half-declared prerequisite: the delivered code contains **no** mechanism for
making LIF/GRU time constants learnable (`LIFCell`'s `alpha`/`tau_*` and
`GRUUnit`'s gates are constructor constants), and adding a learnable-time-constant
`pathway_config` opt-in is a separate controlled change that must be
implemented and tested under `nsmor/BOUNDARY.md` before it can be scored.

Consequently the symmetric-pathway candidates (`C7` GRU+GRU, `C8` LIF+LIF) and
the single-pathway candidate (`C9`) are **not** scored in this protocol: their
family slots are reported **`unavailable` with reason
`prerequisite_mode_absent`** (§11.1) and carried as `p = 1` for Holm
bookkeeping. This is the honest, pre-registered state — the R3 question is
declared and unanswered, never silently dropped and never fabricated. A
symmetric-pathway study requires a fresh prospective protocol once the mode
exists; the R3 hypothesis (a fast/slow division emerging spontaneously from
homologous pathways with learnable time constants) is **not** tested here.

## 7. R4 — Secondary pre-registered analysis: depth vs condition / difficulty

Reported per arm and per seed, **never** in the declared family (§11):

- Per-frame depth (`time_consuming_depth` / `refinement_depth`) aggregated per
  trial; compared across stimulus condition and, where defined, difficulty
  (TTC variant).
- **Explicitly tested against input magnitude.** Because A2's depth was an
  input-magnitude readout (depth ≈ input magnitude, r = 0.93 with lagged speed;
  cap 4 never reached), the depth statistic is **residualized** on input
  statistics: per-trial depth is regressed on the four input statistics
  (mean visual angle magnitude, mean wind, mean lagged speed, mean lagged
  acceleration) plus the MCMC prior, and the **residual** depth is compared
  across condition. The condition difference is reported both raw and
  residualized.
- **Confound stated.** Condition == recording prefix (all 64 prefixes
  condition-pure), so the residual test **cannot** separate a condition-specific
  processing policy from a session/prefix effect. The result is descriptive.
- A null (depth independent of condition after residualization) is a result,
  not a failure.

## 8. R5 — Model-as-virtual-within-subject and cross-condition generalization

Two pre-registered analyses, population level:

1. **Virtual within-subject (counterfactual).** Apply **all three conditions**
   (wind_only, visual_only, multisensory) to the **same** trained model by
   swapping only the physical sensory channels of the input (the counterfactual
   within-subject design the between-group animal data cannot provide), and
   compare the model's per-condition latency/depth distributions. The swap
   changes only channels the model reads; the target and the alignment are
   held fixed. This is a counterfactual **prediction**, not animal data.
   - **Named channels (frozen, exact indices).** The input feature layout is
     fixed by `nsmor/nsmor_dataloader.py:11-20` (and `nsmor/config.py`
     `DEFAULT_FEATURE`): `[0] = visual_angle` (degrees, the looming/visual
     channel), `[1] = wind_state` (0/1, the wind channel),
     `[2] = v_kine(t−1)` (lagged velocity), `[3] = a_kine(t−1)` (lagged
     acceleration), `[4:8] = the static MCMC prior columns`
     (`P_startle, P_walk, P_pre_active, P_no_response`). `sensory_dim = 4`.
     Only columns `0` and `1` are swapped; columns `2..` (lagged history and
     the MCMC prior) are **held fixed** for every counterfactual, so the prior
     cannot leak condition identity and the comparison isolates the two
     stimulus channels.
   - **Reference event (frozen, one named event).** The counterfactual is
     defined on the single named physical reference **wind onset** — the first
     frame with `wind_state > 0.5`, exactly `derive_anchor_frames`'s wind rule
     (`nsmor/pipeline/conditions.py`), which is also the crop anchor for every
     wind-containing trial. The overlaid visual stream is re-anchored to this
     same timeline so that its **visual collision** (peak angle) lands at
     `wind_onset + SOA`, where `SOA` is the **measured** wind→collision SOA of
     the multisensory corpus (per TTC variant, `analyze_behavior._variant_soa`:
     `356 / 288 / 244 / 208` ms for TTC −373 / −308 / −261 / −225; the pooled
     statistic uses the trial-count-weighted mixture). This reproduces a REAL
     multisensory SOA rather than an arbitrary within-frame-index overlay,
     because those SOAs are exactly the ones the recorded multisensory trials
     carry.
   - **Re-anchoring both directions (frozen).** A recorded `visual_only`
     trial's visual channel is anchored to its own collision (peak angle) at
     its own crop anchor. To overlay it on a wind-containing timeline (the
     **forward** direction) the visual stream is shifted by an integer number
     of frames so its collision lands at `wind_onset + round(SOA_ms / dt_ms)`
     (`dt_ms = 4.0`). To overlay a `wind_only` partner's wind channel on a
     `visual_only` timeline (the **reverse** direction) the wind stream is
     shifted by an integer number of frames so its **wind onset** lands at
     `collision − round(SOA_ms / dt_ms)` — the same SOA, solved for the wind
     onset instead of the collision (the exact mirror of the forward
     direction). **Tolerance (both directions):** the shift is integer-frame,
     so the realized SOA differs from the target SOA by at most ±0.5 frame =
     ±2 ms; the realized SOA is recorded per trial and asserted within this
     tolerance, else the trial is reported **unavailable** (never silently
     mis-anchored). **Frames outside `[0, L)` (both directions).** Frames of
     the shifted overlaid stream that fall outside the target trial's `[0, L)`
     are dropped — leading frames before frame 0, trailing frames past the
     target trial end — mirroring the visual stream's explicit `[0, L)` rule
     for the wind stream. A shift that would leave **no** overlaid frame inside
     the trial (in particular a reverse shift whose wind onset lands before
     frame 0 and whose entire wind stream then falls before frame 0) makes that
     counterfactual **unavailable** for the trial, never fabricated.
     **Short-partner rule (reverse direction).** The wind stream of the
     `wind_only` partner is defined only over the partner's own valid extent
     `[0, L_partner)`; after the integer shift a target frame that no partner
     frame covers receives wind `0` (no wind). When the partner is **shorter
     than the target timeline** the overlaid wind covers only the partner's
     shifted extent and the remaining target frames are wind `0`; this is
     recorded per trial (partner length vs target length) and is **not** an
     error, but a target frame whose only coverage would have come from
     trailing partner frames past the target end is dropped by the `[0, L)`
     rule like any other trailing frame.
   - **Counterfactual construction (frozen).** For a trial recorded as
     `wind_only`, the `multisensory` counterfactual sets column `0` to the
     re-anchored visual stream of the matched `visual_only` trial (shifted so
     its collision lands at `wind_onset + SOA` as above), leaving column `1`
     (wind) at its recorded value; the `visual_only` counterfactual zeroes
     column `1` (wind removed) and keeps the re-anchored visual stream. For a
     `visual_only` trial the `multisensory` counterfactual sets column `1` to
     the re-anchored wind channel of the matched `wind_only` trial (shifted so
     its wind onset lands at `collision − SOA` as above; the short-partner and
     `[0, L)` rules apply) and leaves column `0` unchanged; the `wind_only`
     counterfactual zeroes column `0` (visual removed) and keeps the re-anchored
     wind channel. The MCMC prior columns and the target are never modified.
   - **Cross-condition pairing key (frozen).** A counterfactual pairs a
     `wind_only` trial with a `visual_only` trial. Because **TTC variants exist
     only for multisensory trials** (§1.1 — the corpus has no within-animal TTC
     / looming variation) and a `visual_only` trial carries no TTC variant, the
     pairing key must be a quantity that exists for **both** single-modality
     conditions. The frozen key is the **ordinal within the condition**: the
     `wind_only` trial at ordinal `o` is matched to the `visual_only` trial at
     ordinal `o` (the ordinal is the trial's position, in a frozen deterministic
     order — ascending prefix index then ascending within-prefix index — among
     the trials of its own condition). The **TTC variant is not a pairing key**;
     it enters only through the **SOA** assigned to the counterfactual (below).
     When the two conditions have unequal sizes the shorter condition is
     **cycled in order** (`o mod n_shorter`); the deterministic tie rule is
     **lower prefix index wins, then lower ordinal**. All 64 prefixes are
     **condition-pure**, so a matched partner is always from a different
     (single-modality) prefix and no condition identity can leak through the
     pairing. Because the corpus has **no within-animal** trials, "matched" is a
     **declared pairing rule**, stated as an assumption, not an animal fact. A
     trial with no partner in the target condition (or an unavailable shift,
     above) is reported **unavailable** for that counterfactual, never
     fabricated.
   - **SOA variant assignment (frozen).** Each paired counterfactual is
     instantiated **once per TTC variant** — four copies, TTC −373 / −308 /
     −261 / −225 — each copy carrying that variant's measured wind→collision
     SOA (`356 / 288 / 244 / 208` ms). The assignment is deterministic (no
     random draw), so it is reproducible; the four SOAs are distinct, so no tie
     can arise. The four copies feed the four per-variant behavioral-signature
     slots (§1.1, §9), matching the model-free per-variant analysis; the pooled
     counterfactual statistic uses the trial-count-weighted mixture as in §8.1.
2. **Cross-condition generalization (pinned split and leakage control).**
   Train on **wind_only + visual_only only** (single-modality trials), predict
   **multisensory**. The split is fixed here:
   - **Train:** every `wind_only` and `visual_only` trial. **Test:** every
     `multisensory` trial, never seen in training.
   - **Leakage control (exact).** All 64 recording prefixes are
     **condition-pure**, so the multisensory prefixes are **disjoint** from the
     single-modality prefixes by construction: the train/test split is
     prefix-disjoint and no multisensory trial, prefix or frame is seen in
     training. Because the split is disjoint **by prefix**, an additional
     "held out by prefix within each training condition" restriction is
     redundant and is **not** claimed as an extra safeguard; the leakage
     control is exactly prefix-disjointness plus prior isolation.
   - **Prior isolation (frozen; the nested artifact prior is FORBIDDEN).** The
     nested artifact prior (`docs/realdata-phase3-protocol` §1 nested-prior
     SHA) was fit on the full corpus **including multisensory trials**, so it
     carries multisensory label information and would leak into the held-out
     multisensory prediction; it is **not** usable here. The MCMC behavioral
     state prior feeding the held-out multisensory prefixes **must be refit on
     the `wind_only` + `visual_only` trials only** by a scripted, fail-closed
     step (a declared `scripts/train_phase3_arms.py` prerequisite). The scorer
     asserts the refit prior's fit set is exactly the single-modality trials
     (no multisensory trial, prefix or frame in the fit) and **refuses
     otherwise**; a prior that saw multisensory labels is rejected, never
     silently substituted. **If this single-modality-only refit is infeasible
     (e.g. the fit does not converge under the frozen pipeline), R5.2 is
     declared `unavailable`** with reason `prior_isolation_infeasible` — it is
     never scored against the nested-artifact prior. The same
     single-modality-only prior is used for the R5.1 counterfactual's fixed
     prior columns; the nested-artifact prior is likewise forbidden there.
   - **Comparison (falsifiable).** Two prediction rules are compared against
     the observed multisensory latency distribution, per prefix:
     - **race** — the Miller bound from the model's own single-modality CDFs;
     - **integration** — the model's direct multisensory prediction.
     The **frozen decision rule**: a rule "reproduces the facilitation
     signature" iff its per-prefix facilitation statistic lies inside the
     observed prefix-cluster 95% interval and shares the sign, in **all three
     seeds**; if both rules reproduce it, they are indistinguishable on this
     corpus and **neither is preferred**. Because the corpus has no
     within-animal TTC variation and the visual arm rarely responds
     (F_V 0.016), the test is stated as **facilitation** (AV vs A), matching
     the model-free finding, and the visual term is reported with its small
     support.
   The observed target is the model-free behavior metrics from
   `scripts/analyze_behavior.py` (`run_race_model`, `run_facilitation`,
   `run_accumulation`), **reused, not copied**: phase 3 imports the frozen
   functions and the frozen D1 trigger / eligibility / Miller-bound definitions
   rather than re-implementing them.

## 9. R6 — Model-comparison criterion (behavioral signatures, not MSE)

The comparison criterion is **which architecture reproduces the behavioral
signatures**, not lowest MSE. The declared signatures (from §1.1) and their
matching statistics:

| signature | statistic | frozen reference |
| :-- | :-- | :-- |
| race violation / facilitation | pooled violation area, prefix permutation, Holm | 389.4 prob·ms, Holm p 0.0075 |
| latency facilitation | multisensory − wind_only per-prefix mean | +38.4 ms |
| latency quantiles / support | modeled median & IQR vs observed (≤ 4 atoms; §4.1) | wind 126.1 ms (SD 33.2); multi 93.2 ms (SD 69.2) |
| wind-locked trigger | median escape − collision | −264 to −120 ms across variants |

**Declared candidate set:**

- **C0** dual pathway + gating (baseline / A1 config; `off`) — the reference;
- **C1** time-consuming `depth_delay` (K4);
- **C2** time-consuming `accumulate` (K4);
- **C3** fixed-depth `K2` / **C4** `K3` / **C5** `K4` (budget-matched controls);
- **C6** adaptive (ACT; `lambda_compute > 0`);
- **C7** symmetric GRU+GRU / **C8** symmetric LIF+LIF — **not scored** here
  (`prerequisite_mode_absent`, §6);
- **C9** single pathway + accumulate-to-threshold — **not scored** here
  (`prerequisite_mode_absent`; a single recurrent pathway with the
  `accumulate` timing mode);
- **C10** single-modality-trained cross-condition model (§8.2) — a **scored**
  candidate with its **own declared model family** (family `X`, §11.1): it is
  the dual-pathway architecture trained on `wind_only` + `visual_only` only and
  evaluated on the held-out `multisensory` prefixes. It is **not** folded into
  the `C0`–`C6` family (its train set, prior and evaluation scope differ), so it
  carries its own multiplicity correction and its own slots (below).

A candidate "reproduces a signature" only if its matched statistic lies inside
the observed prefix-cluster 95% interval and shares the sign, per seed; a
candidate that lowers MSE while failing a signature is **not** preferred on
this criterion. MSE is reported descriptively only.

### 9.1 Race-vs-integration scope (frozen, R5/§9)

The time-consuming mode is a **post-fusion emission-timing device**: it delays
the frame the model emits but does not add a pathway that could shape a
single-modality → multisensory latency transfer. Its signature mapping is
therefore identified for the **latency** signatures (facilitation magnitude,
latency shape, wind-locked trigger) — the delay directly sets the modeled
latency — but **not** for the race-vs-integration contrast: a timing device
cannot by itself produce Miller-bound violation from single-modality CDFs.

The pre-registered race-vs-integration test (§8.2, the `run_race_model`
primitive) is applied to the **dual-pathway candidates** (`C0` and the
single-modality-trained cross-condition model), where a fast/slow division of
labor can in principle produce sub-Miller facilitation. For the time-consuming
arms `C1`/`C2` the race contrast is **descriptive only**, reported with the
stated (weak) assumption under which a post-fusion timing device could affect
it — namely that the device's per-condition delay shifts the *trial-level
latency CDF* the race test consumes, without any pathway-level integration. No
mechanism claim for `C1`/`C2` may rest on the race signature.

## 10. R8 — Budget (requires user approval before training)

One RTX 5060 Ti (8.52 GB), serial; ~10 h per run at the real batch/sequence
size (B = 128, T = 2400), as measured for the phase-2 arms (all `off`). The
time-consuming and adaptive arms add a per-frame recursion loop whose cost is
`O(T · K · (BLOCK + HALT))` on top of the base forward; the emission shift
itself is **vectorized** (a `scatter_reduce` arbitration, no `O(B·T)` Python
loop), so the added cost is the recursion compute, not the alignment. **Every
enabled recursion mode** — fixed (`C3`/`C4`/`C5`), adaptive (`C6`) and
time-consuming (`C1`/`C2`) — runs that per-frame loop, so all six are budgeted
at the **same** `1.5×` the `off` per-run estimate (~15 h). Only `C0` (`off`,
no module) is at the base ~10 h. Until measured, the 1.5× figure is a planning
estimate; it is **re-derived after the first completed enabled run** and
recorded before the remaining runs (never used to select an outcome).
Estimated hours are `runs × per-run h` plus scoring (CPU, hours).

| block | arms | seeds | runs | per-run h | GPU h (est.) |
| :-- | :-- | :-- | --: | --: | --: |
| Foundation gate + primary mechanism (`off`) | C0 | 42, 43, 44 | 3 | 10 | 30 |
| Foundation gate + primary mechanism (recursion enabled) | C1–C6 | 42, 43, 44 | 18 | 15 | 270 |
| **Total** | | | **21** | | **~300** |

The declared per-run figures give `3 × 10 h + 18 × 15 h = 300 h`; the earlier
draft's 240 h was arithmetically inconsistent with its own per-run estimates
and is superseded by the table above (finding R1-B2).

- The symmetric-pathway (`C7`/`C8`) and single-pathway (`C9`) arms are **not**
  budgeted: their prerequisite mode is absent (§6), and no run is scheduled.
- Serial wall time ≈ 300 h ≈ 12.5 days on one device. No extra GPU is assumed.
- **No run starts until the user approves this matrix.** A reduction (e.g.
  fewer seeds) is a **fresh prospective decision** recorded before any run; it
  is never made after seeing an outcome.
- OOM policy: identical to phase-2 §10 (no batch-size change; only non-semantic
  memory settings; a run that cannot complete at the identical effective batch
  is declared **unavailable** and its family contribution is null).

## 11. Statistical family (frozen, declared up front)

- **Inferential unit:** the per-trial mean over the 3 seeds (a seed-mean is one
  observation; seed-to-seed spread is reported descriptively via the per-seed
  per-trial vectors).
- **Normality / variance:** before any paired parametric test, Shapiro-Wilk
  (per group) and Levene (equal variance); a non-normal sample uses the
  rank-based / permutation path and reports the reason. Effect sizes are always
  reported: paired Cohen's `d_z` for trial-level deltas, Cliff's delta and
  Cohen's d for between-group latency.
- **Prefix-cluster resampling:** average eligible trial statistics within each
  recording prefix, then the unweighted mean of prefix means. **Two distinct
  frozen primitives** are used, one per contrast class:
  - **Budget/metric contrasts** use
    `nsmor.analysis.model_comparison._sign_flip_cluster_p` — a two-sided
    whole-prefix **sign-flip** over the prefix means. With 12 prefixes it takes
    the **exact** branch (`2^12 = 4096` sign patterns, decided in integer
    arithmetic, **no plus-one correction**), so the smallest attainable
    two-sided p is `2/4096 ≈ 0.00049`.
  - **Behavioral-signature contrasts** use the frozen prefix-label
    **permutation** primitive `scripts/analyze_behavior.py::permutation_p_value`
    (Monte-Carlo relabeling with the `+1` correction, never 0), whose floor is
    `1/(n_perm + 1)` — `≈ 0.0005` at the default `n_perm = 2000` and `≈ 0.0002`
    at `n_perm = 5000` (the facilitation test's setting, which produced the
    reported `p = 0.0002` in `docs/behavior-exploratory-20261010.md`). This is
    a **different null** (label permutation, not sign flip) and is **not** the
    exact-enumeration branch; the two floors are stated separately and neither
    is claimed for the other's contrast class.
  Exchangeability is a **stated joint null**, asserted via
  `--exchangeability_asserted`; it is a conditional descriptive diagnostic, not
  proof of animal independence.
- **Multiplicity:** Holm-Bonferroni over the **fixed** declared family
  (§11.1); unavailable slots are carried as p = 1 (reported null, never
  fabricated). The per-slot primitive is
  `nsmor.analysis.model_comparison.paired_mse_comparison` (budget/metric
  contrasts) and the frozen prefix-permutation primitives in
  `scripts/analyze_behavior.py` (behavioral-signature contrasts) — no
  re-implemented algebra.
- **Resolution limit (stated honestly).** With 12 prefixes and the sign-flip
  branch, a slot can reach Holm significance only if **all 12 prefix means share
  one sign**; the first threshold is `0.05 / m`. This is a hard floor on this
  corpus. For the permutation branch the floor is `1/(n_perm + 1)` as above.

### 11.1 Declared family (exact count, frozen now)

Collapsed to the seed mean (per-trial mean over seeds 42, 43, 44). Every slot
is named below; the count is **pinned**, not a range.

**Scored behavioral-signature slots (26).** The four signatures in §9 (race
area, latency facilitation, latency-shape quantiles, wind-locked trigger
offset) are scored for the candidates whose mechanism can produce them:

- `{C0, C3, C4, C5, C6} × {race, facilitation, latency-quantiles, trigger}` =
  **20** slots.
- `{C1, C2} × {facilitation, latency-quantiles, trigger}` = **6** slots.
  (The `C1`/`C2` race slot is **not** scored: §9.1 declares a post-fusion
  timing device cannot produce Miller-bound violation, so the race contrast for
  the time-consuming arms is **descriptive only**.)

**Scored budget/mode slots (11).** `{C1, C2, C6} × {C3, C4, C5}` (9) plus
`{C1, C2} × {C0}` (2) = **11** slots.

**Declared scored family size = 26 + 11 = 37.** Holm-Bonferroni is applied
over these 37 slots; the first threshold is `0.05 / 37`.

**Separate cross-condition family `X` (declared now).** The single-modality-
trained cross-condition model `C10` (§8.2) is scored on the same four
signatures (`race`, `facilitation`, `latency-quantiles`, `trigger`) but on the
held-out **multisensory** prefixes, under its **own** prior and train set. It
is therefore **not** in the `C0`–`C6` family and is corrected **separately**:
`C10 × {race, facilitation, latency-quantiles, trigger}` = **4** slots, Holm
over `m_X = 4`, first threshold `0.05 / 4`. Folding `C10` into the main family
would mix two different nulls (two train sets, two priors); it is declared as
family `X` instead, and the scorer reports `m_main = 37`, `m_X = 4` and their
adjusted p-values side by side. If the §8.2 single-modality-only prior refit is
infeasible (A4), the whole `X` family is reported `unavailable` with reason
`prior_isolation_infeasible`.

**Unavailable / descriptive slots (14, reported but OUTSIDE the family).**
These are reported with an explicit reason and a `p = 1` marker, but are
**excluded from `m`** so they cannot inflate the Holm bar for the scored slots
(finding R1-M5: counting non-existent candidate slots as carried p=1 slots is a
discretionary inflation of the family):

- `{C7, C8, C9} × {race, facilitation, latency-quantiles, trigger}` = **12**
  slots, reason `prerequisite_mode_absent` (§6/§9: the symmetric-pathway and
  single-pathway modes are not implemented).
- `{C1, C2} × {race}` = **2** slots, reason `timing_device_race_descriptive`
  (§9.1).

The scorer reports main family size (`37`), cross-condition family size (`4`),
valid-test count, unavailable-slot count (`12`), and descriptive-slot count
(`2`) explicitly, and additionally reports a **sensitivity** Holm adjustment
under the inclusive `m = 51` so the effect of the exclusion is visible. The
families are fixed here, before any phase-3 training, and are never enlarged
after seeing an outcome. Every slot reports the effect size and the
Holm-adjusted p beside the raw p.

## 12. Claim boundaries

Regardless of numerical outcomes, this phase **cannot** license:

- Any animal-level, circuit-level or causal claim (animal identity unverified;
  prefixes are not animals; the MoR router is a representational routing gate,
  not a causal estimator).
- An untouched-holdout or test-set claim (validation-guided selection;
  `nested_outer_validation` only).
- **"The cricket has MoR."** The strongest licensable mechanism claim is that
  **an adaptive processing duration exists and MoR is one discrete
  approximation of it**; `depth_delay` and `accumulate` are behaviorally nearly
  indistinguishable, so no claim may depend on which is chosen.
- A claim that model-grid delay is measured neural latency, work, MAC or ATP.
- A mechanism claim when the §3 foundation gate fails (downgraded to
  descriptive) or when the §5 budget match is `NOTMATCHED` (the affected
  contrast is descriptive only).
- A condition-specific processing claim separated from the session/prefix
  confound (condition == prefix; §7).
- Any claim requiring data not in the corpus: within-animal TTC / looming
  variation, TTC = 0, cue-conflict, verified animal identity, an animal-split
  test set, or ISI/ITI extraction (not authorized).
- Any signed-velocity / heading claim (the target is non-negative speed).

What the outcomes *would* license (preliminary, conditional on §3):

- If a time-consuming arm's delay distribution matches the observed latency
  distribution on the **pre-registered quantile/support statistic** (§4.1 —
  never the untestable continuous-shape match) and the budget match holds: a
  **preliminary** statement that the model's processing duration is consistent
  with the animal's response latency on this corpus.
- If cross-condition generalization reproduces the facilitation signature: a
  **preliminary** statement that single-modality training transfers to the
  multisensory latency signature.
- Unavailable analyses are reported as unavailable with reasons, never as
  negative results. The symmetric-pathway fast/slow claim (§6) is **not**
  licensed by this protocol (its mode is absent).

## 13. Implementation status (frozen before phase-2 results)

Delivered and frozen now, with tests (causality at the scored quantity,
depth→delay mapping, an injective per-target emission with collision
accounting, defaults byte-identical on fixed seeds, old checkpoints load and
produce identical outputs, a short CPU smoke):

- `nsmor/model_nsmor_core.py`: opt-in `TimeConsumingRecursion` (vectorized,
  injective emission with deterministic per-target arbitration) and the
  `time_consuming_*` options on `BioDecisionCore` / `NSMoRCore`, default `off`;
  the delayed emission is exposed as `internals["time_consuming_hidden"]`
  (distinct from `refined_hidden`), the dropped-source counts as
  `internals["time_consuming_collisions"]` /
  `internals["time_consuming_truncated"]`, and the post-arbitration
  `internals["realized_delay"]` (the R6 latency statistic's quantity, §4.1).
- `nsmor/loss.py`: additive `out_valid` mask on `FrontendLoss` / `BioDecisionLoss`
  / `BioJointLoss` (default `None` keeps the length-only mask bitwise
  unchanged); the training/validation losses pass `internals["out_valid"]` so
  never-emitted frames are not scored, including the frame-based third-difference
  (jerk) term, which masks every four-frame window whose contributing frames are
  not all emitted.
- `scripts/train.py`: additive CLI flags, `build_model` plumbing, resume guard
  (leaf comparison plus the top-level `time_consuming_mode` stamp; a legacy
  checkpoint with **neither** the stamp nor the stored config leaf defaults to
  `off` and refuses an enabled active mode — finding R1-B3) and additive
  checkpoint provenance; the reported `compute_metrics` headline / escape audit /
  persistence benchmark and the checkpoint-selection MSE apply
  `internals["out_valid"]` so the gate reads the same masked quantity the loss
  optimizes (finding R2-B2). **Never compress (finding R4-B2):** `out_valid` is
  applied as a **per-frame mask on the uncompressed arrays** — never by
  dropping the masked frames. A frame `t` is scored for the model AND the lag-1
  persistence comparator only if `out_valid[t]` **and** `out_valid[t-1]` (the
  comparator reads the true, uncompressed `y_true[t-1]`); escape-band
  membership and `_sustained_run` are computed on the uncompressed target with
  `out_valid` folded into the per-frame mask (a hole breaks a run instead of
  bridging it); the same rule applies to the checkpoint-selection MSE and the
  escape sweep. Off-mode (no `out_valid`) is bitwise identical.
- `nsmor/config_parser.py`: additive `time_consuming_*` `ModelConfig` leaves;
  `eps`/`update_scale`/`threshold` validated under `"off"` too, `eps ∈ [1e-6,
  1)` (finding R1-m2), `threshold ∈ (0, 1]`. **`config_sha256` drifts for
  EVERY config, including `off`** (finding R2-M2, restated): `to_dict()` is
  `asdict(self)`, so the six new leaves are serialized even when the module is
  absent and the segment hash differs from a pre-0009 run. The guarantee is
  **behavior / `state_dict` byte-equality** on fixed seeds — the numerical
  outputs, the `state_dict` keys and the construction RNG are unaffected; the
  config-hash lineage string is **not** the guarantee and is expected to move.
- `config/default.yaml`: documented, default-off.
- Raw-JAX / Flax / `JAXEvalWrapper` and the full-system Jacobian refuse an
  enabled mode (keyed on the EXECUTED module).

Declared prerequisites (implemented only after this protocol is frozen, under
the controlled-change protocol, before their runs are scored):

- the phase-3 model-comparison framework `nsmor/analysis/model_comparison_phase3.py`
  (signature statistics, counterfactual within-subject, cross-condition
  generalization), which **imports** `scripts/analyze_behavior.py`'s
  `run_race_model` / `run_facilitation` / `run_accumulation` and the D1
  trigger / eligibility / Miller-bound definitions — it does **not**
  re-implement them;
- `scripts/train_phase3_arms.py` (train on wind_only + visual_only only for the
  cross-condition arms; refit the behavioral-state prior on single-modality
  trials only for §8.2/§8.1, fail-closed; emit the `C1`/`C2` arm configs with
  `time_consuming_delay_scale = 10.0`) and `scripts/evaluate_realdata_phase3.py`
  (the scorer that emits the frozen 37-slot family plus the 14
  reported-but-excluded slots of §11.1 and the separate 4-slot cross-condition
  family `X`, asserts the loaded checkpoint's `time_consuming_delay_scale` is
  `10.0`, and implements the §5 extended budget-matching rule against the
  executed module);
- a symmetric-pathway mode (§6) under a **fresh** prospective protocol (R3 is
  not tested by this document).

No phase-3 training or scoring begins until the §3 foundation gate has been
evaluated on the frozen phase-2 output and the §10 budget has been approved.

## 14. Release boundary

No release is authorized by this document alone. Any PR stages explicit
reviewed paths only (never `git add -A`), preserves all formal
DATA/NEST/checkpoints/receipts and protected untracked paths, and requires
independent review plus the standard test gates.
