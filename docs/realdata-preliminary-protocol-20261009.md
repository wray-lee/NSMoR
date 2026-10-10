# Real-data preliminary-conclusions protocol — 2026-10-09

Status: **PROSPECTIVE**. Frozen before any arm training. Every arm, config
delta, metric, comparator, analysis and claim boundary below is fixed in
advance. No retrospective enlargement is permitted; any further experiment
requires a new prospective protocol.

User goal (verbatim): "真实数据上做，目标是得出科学严谨生物完备的前置结论（因为
目前数据还没齐，所以暂且是前置不完整结论）" — i.e. preliminary, deliberately
*incomplete* conclusions on the real corpus, not final or generalizable claims.

## 1. Scope and immutable evidence

- Engineering observed-history prediction on the existing real corpus only.
  No new data collection, no synthetic data, no animal-level inference.
- Dataset SHA-256:
  `b1bd5578025fb5eaa6f4eb3e355c097b6576d44dbedaa001027f1de5447e06b2`.
- Nested-prior SHA-256:
  `f25856304a5857695945bf5083ca54745f47fe5dffe915111f6ba28d524ddf94`.
- Nested seed 42; validation fraction 0.2; exact saved split indices and prior
  mapping are read from the artifact. Evaluation scope is
  `nested_outer_validation`, **not** an untouched holdout. Validation-guided
  model selection is exploratory.
- Corpus: 2304 labelled trials (multisensory 1188, visual_only 720, wind_only
  396; **no** no_stimulus condition) + 128 unlabelled no_stimulus anchors;
  64 recording prefixes, 52 train / 12 val; 432 validation trials.
  `animal_identity_status=unverified`; `biological_completeness=not_established`.
  Recording prefixes are not animals.
- Baseline `.scratch/model-opt-baseline-300-20261003-rerun/` (relu/off, k=0,
  seed 42) is **read-only**: best_model.pth sha
  `709ca1f1a121c2b0904d14ca056ad335dac14f5e178e9edb3f22b541ba18ff1b`,
  stored `epoch` field 295 (0-indexed; the 296th completed epoch), val loss
  7.196, R² 0.747. No baseline rerun, resume, overwrite, or candidate
  initialization from it.

## 2. Arms and exact config deltas

Two fresh arms, ONE seed (42) each, k(persistence_skip)=0 for both:

| Arm | activation | refinement_mode | lambda_compute | output_dir |
| :-- | :-- | :-- | :-- | :-- |
| A1  | `swiglu` | `off`      | 0.0  | `.scratch/realdata-preliminary-20261009/A1-swiglu-off` |
| A2  | `swiglu` | `adaptive` | 0.01 | `.scratch/realdata-preliminary-20261009/A2-swiglu-adaptive` |

`lambda_compute=0.01` is a **pre-declared small compute-cost weight**, chosen
before any training and not tuned on any result. Adaptive training is refused at
the config boundary unless `lambda_compute > 0` (ADR 0008), so a positive value
is mandatory; `0.01` is a deliberately weak compute penalty. For provenance: the
architecture-v1 control harness (`scripts/final_control_harness.py`) used
`lambda_compute=0.05` for its adaptive arm; this protocol uses `0.01`. Both are
arbitrary positive constants; neither is a biological parameter and neither was
selected by observing outcomes. The value is reported explicitly so A2 is not a
bare-`swiglu` duplicate of A1, and the choice is flagged as an open question in
the proposal.

Every other leaf is the baseline's exact saved config, seed, dataset, nested
prior, split, budget (300 epochs), optimizer, loss coefficients, early-stopping
(patience 20) and checkpoint-selection rule. Only the arm keys above and the
isolated `output_dir` change.

The arm-identity guard in the scorer binds `activation`, `refinement_mode`,
`lambda_compute` and `persistence_skip`. It does **not** bind
`refinement_max_steps` / `refinement_eps` / `refinement_update_scale`: those
three are byte-identical across A1 and A2 (`4` / `0.01` / `1.0`) and so carry
no arm-identity information — they appear only as additive leaves in
`arm-resolution-diff.json`, never as changed leaves. They are still the frozen
values above; the omission is deliberate and does not weaken the A1/A2
distinction.

Frozen arm configs and the resolution-diff evidence:

- `config/realdata-preliminary-20261009/A1-swiglu-off.yaml`
- `config/realdata-preliminary-20261009/A2-swiglu-adaptive.yaml`
- `config/realdata-preliminary-20261009/arm-resolution-diff.json` (sha
  `3f81f095dba8c4ee79d9cc7912f137e5b0d8295b4d1aeeea5f165ef7155cffba`):
  each arm resolved through `scripts/train.py::build_config` differs from the
  baseline's saved config **only** in `checkpoint.output_dir`, plus the seven
  additive architecture keys (`model.activation`, `model.refinement_mode`,
  `model.refinement_max_steps`, `model.refinement_eps`,
  `model.refinement_update_scale`, `loss.lambda_compute`, and the already-
  default `model.persistence_skip=0.0`). No baseline leaf changes value;
  `lambda_reg` resolves to 0.2, `phase1_epochs=None`.

Training command shape (each arm; no `--resume`):

```
python scripts/train.py \
  --config config/realdata-preliminary-20261009/A1-swiglu-off.yaml \
  --dataset <corpus>/nsmor_dataset.pt \
  --nested_prior_artifact <corpus>/nested_split_seed42.pt
```

## 3. Metrics fixed before training

**Primary:** raw pooled MSE on all valid frames (t >= 0), with RMSE, MAE and
R-squared. It is **not** replaced after observing unfavorable results.

**Co-reports** (never substituted for the primary):

- Trial-local `t >= 1` aligned model, independent persistence (`y_true[t-1]`)
  and zero comparators, with MSE-based skill and explicit undefined-skill
  denominators. Persistence never crosses a trial boundary.
- Sustained high-velocity band `abs(y) >= 10 cm/s`, minimum run length 2, as an
  engineering guardrail — **not** a stimulus-conditioned escape label.
- Exact sensitivity grid: thresholds `[5, 10, 20, 50] cm/s` x min run lengths
  `[1, 2, 3]`; all 12 cells reported, no best-cell selection.
- Sustained membership is computed on complete valid trials before slicing
  `t >= 1`. Full-frame and aligned partitions have separate counts.
- Retrospective high-velocity masks never enter model features.

## 4. Statistical family (frozen)

The inferential/descriptive grid uses `t >= 1` throughout.

- Candidates: `{A1, A2}` (2).
- Comparators per candidate: `{head_only_baseline, persistence, zero}` (3).
- Scopes: `aligned_overall` + the 12 sustained cells = 13.
- **Declared family size: 2 * 3 * 13 = 78.** Reported in full, including
  unavailable cells.
- Per-trial delta = candidate MSE minus comparator MSE on identical eligible
  frames (negative is better). Trial-level paired Cohen's `d_z` = mean delta /
  sample sd(delta); report trial/frame/prefix counts. Fewer than two trials or
  zero delta variance gives null `d_z` with an explicit reason.
- Prefix-cluster statistic: average eligible trial deltas within each recording
  prefix, then the unweighted mean of prefix means. Two-sided sign-flip
  p-value over **whole prefix means**, minimum 6 eligible prefixes, exact
  enumeration at <= 16 prefixes else 100,000 Monte-Carlo sign patterns, fixed
  seed 42, plus-one correction. It is a conditional descriptive diagnostic
  under a stated joint sign-exchangeability null, not proof of animal
  independence. Too few prefixes or unasserted exchangeability yields null
  p with reason.
- Holm-Bonferroni over the fixed 78 slots; unavailable slots are conservatively
  p=1 for adjustment bookkeeping but still reported null. Report family size,
  available-slot count and non-null-p count explicitly.
- Unavailable slots (e.g. empty sustained bands, A2 absent) are reported null,
  never zero error.

Scoring is done by the thin driver `scripts/evaluate_realdata_preliminary.py`,
which **delegates** the whole family to the frozen
`nsmor.analysis.optimization_evaluation.assemble_declared_family` (with
`candidate_ids=("A1","A2")`), which in turn uses the frozen
`compute_aligned_grid_metrics`, `paired_mse_comparison` and
`holm_correct_declared_family` — no family algebra is re-implemented. The
head-only comparator is pinned to the frozen baseline SHA before scoring, and
each candidate arm is **bound to its declared config delta** before scoring:
`--a1` must carry `activation=swiglu, refinement_mode=off, lambda_compute=0.0`
and `--a2` must carry `activation=swiglu, refinement_mode=adaptive,
lambda_compute=0.01` (both `persistence_skip=0.0`), else the driver refuses. A
swapped or wrong arm path therefore cannot be scored under an arm label it does
not have.

## 5. Biological analysis battery and acceptance controls

Every analysis below has a runnable path for the listed checkpoints, verified
by loading each arm through the canonical loader
(`nsmor/analysis/prediction_units.load_model_from_checkpoint`) and by the
runtime probes in this work. Availability is **per checkpoint**:

| Analysis (script) | baseline | A1 | A2 | Acceptance control |
| :-- | :-: | :-: | :-: | :-- |
| Dynamics, raw-GRU coordinate (`analyze_dynamics.py`) | ✅ | ✅ | ✅ | raw `gru_hidden_raw` present, shape `(B,T,H)` |
| LIF spike/membrane statistics (same pass) | ✅ | ✅ | ✅ | `lif_spikes`/`lif_potentials` shape `(B,T,H)`, finite |
| Routing gates by condition (`analyze_gating.py`) | ✅ | ✅ | ✅ | **whole-corpus in-sample descriptive** (`evidence_scope=in_sample_descriptive_only`); contrast needs both groups non-empty, effect size withheld below 2 trials/group; the val split has **0** wind_only trials, so the contrast is **UNAVAILABLE on the declared evaluation scope** |
| Lesion / readout ablation (`simulate_lesion.py`) | ✅ | ✅ | ✅ | gate override observed in `routing_gates` |
| Psychophysics (`simulate_psychophysics.py`) | ✅ | ✅ | ✅ | noise sweep finite; JSON finite |
| Integration window (`analyze_integration.py`) | ✅ | ✅ | ✅ | in-sample only; labelled as such |
| Gating clustering (`analyze_gating.py`) | ✅ | ✅ | ✅ | unsupervised silhouette; no label leakage |
| Jacobian / fixed points (`analyze_jacobian.py`) | ⚠️ | ⚠️ | ⚠️ | **WITHHELD unless** the frozen-input control passes (residual <= 0.3); prior chain failed (1.32–2.62) |
| Full-system Jacobian (`--full_system`) | ✅ | ✅ | ❌ | A2 **UNAVAILABLE**: refused (pre-refinement coordinate) |
| JAX GRU-Jacobian (`dynamics_jax.FixedPointAdapterJAX`, `analyze_jacobian.py --backend jax`) | ✅ | ✅ | ✅ | differentiates the raw GRU operator, which refinement does not alter; **conditional on a working JAX install** (else silently falls back to PyTorch); refuses stacked GRU only |
| Flax forward-eval (`nsmor/analysis/jax_eval.wrap_eval_model`) | ✅ | ✅ | ❌ | A2 **UNAVAILABLE**: refuses enabled refinement (fail closed); also silently falls back to PyTorch if JAX is missing; **no `scripts/` caller** — not the Jacobian path |
| Autoregressive (`simulate_autoregressive.py`) | ✅ | ✅ | ✅ | k=0 required (both arms qualify); k!=0 refused |

Legend: ✅ available, ⚠️ conditional on the acceptance control, ❌ unavailable
with reason.

Rationale for the unavailable cells (fail-closed, not silent): architecture v1
refuses a **full-system** Jacobian whenever a refinement module is present
because the differentiated coordinate is the *pre-refinement* blend, not the
model's refined readout (`nsmor/analysis/dynamics.assert_no_adaptive_refinement_for_full_jacobian`),
and the Flax forward-eval wrapper (`nsmor/analysis/jax_eval.wrap_eval_model`)
refuses an enabled refinement mode rather than silently dropping the module.
A2's **raw-GRU** fixed-point/Jacobian coordinate is unchanged by refinement
(the refined latent is a readout, not a new autonomous recurrent state), so
both the non-full-system Jacobian and the JAX GRU-Jacobian adapter
(`dynamics_jax.FixedPointAdapterJAX`, which differentiates the raw GRU
operator) remain available for A2 subject to the same frozen-input control.
The only A2 cells that are unavailable are the **full-system** Jacobian and
the Flax **forward-eval** wrapper; the latter has no `scripts/` caller.

Per-analysis acceptance for the routing-gate comparison: `analyze_gating.py`
runs over the **whole corpus** (2304 trials), not the validation split, and
stamps `evidence_scope=in_sample_descriptive_only`. It reports a routing
contrast only when **both** groups are non-empty; an **empty** group yields
`comparison_available=False` with reason `missing_wind_or_visual_present_group`.
For a present-but-singleton group the code still emits the group means but
withholds the effect size (`cohens_d=None`,
`cohens_d_unavailable_reason=fewer_than_two_trials_per_group`) — the module's
**existing** minimum-of-two convention, unchanged. Because the nested
**validation split contains zero wind_only trials** (all 396 pure-wind trials
are in train), the wind vs visual-present routing contrast is **UNAVAILABLE on
the declared `nested_outer_validation` scope** and is reported as such; only
the whole-corpus in-sample descriptive counts are emitted. No routing
difference is claimed when either group is empty or a group has fewer than two
trials (the effect size is withheld in the latter case).

## 6. A2 compute-accounting report (descriptive)

For A2, record the **actual** halting behaviour over the validation set from the
model internals: mean `refinement_depth` per valid frame and total
`refinement_updates`, **per condition** (wind_only vs visual-present) and
pooled; and mean `refinement_ponder_cost` **pooled only**. Per the core
contract `refinement_depth` is `(B, T)` while `refinement_updates` and
`refinement_ponder_cost` are **0-dim batch scalars** (`updates == depth.sum()`),
so the per-condition update total is the per-frame depth sum over that
condition's valid frames and the ponder is reported pooled only. This is an
honest accounting of the internal recursion the model actually used. It is
emitted by
`scripts/evaluate_realdata_preliminary.py::compute_condition_accounting` as the
`a2_compute_accounting` field of the family JSON (a runnable path, not an
aspiration).

The condition split is taken from the loader's `wind_only_mask` and
cross-checked against the dataset's authoritative `stimulus_conditions` (via
the nested val indices); the two must agree. Because the corpus is incomplete
and a future batch may add `no_stimulus` trials, the split **fails closed**:
if any validation trial is outside `{wind_only, visual_only, multisensory}`,
`compute_condition_accounting` raises rather than folding it into the
`visual_present` complement. If the loader carries no wind mask, the split is
reported unavailable (`condition_split_available=false`) and pooled statistics
are still emitted.

It is **not** a claim of biological depth, ATP cost, or adaptive advantage.
Per ADR 0008 the discrete depth is detached and the ponder is a differentiable
compute *proxy*, not measured work/MAC/ATP. A2 is not asserted to outperform
A1 or the baseline; all outcomes are reported.

## 7. OOM policy (decided now)

Measured on the RTX 5060 Ti (8.52 GB) at the real batch/sequence size
(B=128, T=2400), synthetic shapes, forward+backward:

- A1 (swiglu/off): peak allocated 2.74 GB, peak reserved 3.06 GB.
- A2 (swiglu/adaptive): peak allocated 4.29 GB, peak reserved 4.38 GB.

A2 uses ~4.4 GB of 8.52 GB — comfortable headroom, so OOM is not expected.

Pre-declared policy if A2 nevertheless OOMs at batch 128:

1. `scripts/train.py` exposes **no** gradient-accumulation option; changing the
   batch size would change the comparison, which is **not permitted**.
2. The only permitted mitigation is a non-semantic memory setting that keeps the
   effective batch identical (e.g. `PYTORCH_CUDA_ALLOC_CONF` /
   `expandable_segments`) plus a reduced `num_workers`. These change no
   optimizer step and no batch composition.
3. If A2 still cannot complete at the identical effective batch, **A2 is
   declared unavailable** in every analysis and its 39 family slots are reported
   null. The comparison is never silently weakened.

## 8. Claim boundaries

Regardless of the numerical outcomes, this phase **cannot** license:

- Animal-level generalization or circuit-level claims (animal identity
  unverified; recording prefixes are not animals).
- An untouched-holdout or test-set claim (validation-guided selection;
  `nested_outer_validation` only).
- A global optimum or the data's absolute ceiling (bounded two-arm set).
- A causal-inference claim about the MoR router (it is a representational
  routing gate, not a causal estimator).
- Any claim about the missing `no_stimulus` condition, or about clock
  tolerances (unverified).
- Any claim that A2 (adaptive) is biologically deeper or superior; adaptive
  need not beat fixed, and no superiority is asserted before measurement.

What the outcomes *would* license (preliminary, conditional):

- If A1/A2 beat the baseline and persistence on the primary metric with a
  consistent paired effect size and non-null, Holm-adjusted sign-flip p-values:
  a **preliminary** statement that the gated activation (A1) and/or adaptive
  refinement (A2) are associated with lower observed-history error on this
  corpus's nested outer validation — still incomplete-data, non-animal.
- If A2 does not beat A1: report honestly that adaptive recursion shows no
  measured advantage here, consistent with ADR 0008's "adaptive need not
  outperform fixed".
- Unavailable analyses (A2 full-system Jacobian, JAX paths, any withheld
  Jacobian) are reported as unavailable with reasons — never as negative
  results.

## 9. Release boundary

No release is authorized by this document alone. Any PR stages explicit
reviewed paths only (never `git add -A`), preserves all formal
DATA/NEST/checkpoints/receipts and protected untracked paths, and requires
independent review plus the standard test gates.
