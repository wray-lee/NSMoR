# Real-data preliminary conclusions (2026-10-09) — PRELIMINARY, INCOMPLETE DATA

Status: **DRAFT — submitted for dual independent review.** Governing protocol:
`docs/realdata-preliminary-protocol-20261009.md` (sha `80fc88fe…`, commit
`000c232`, frozen before training). Every claim below is bounded by protocol §8.

## 1. What was run

| Arm | activation | refinement | lambda_compute | output |
| :-- | :-- | :-- | :-- | :-- |
| baseline (frozen, sha `709ca1f1…`) | relu | off | 0.0 | `.scratch/model-opt-baseline-300-20261003-rerun/` |
| A1 | swiglu | off | 0.0 | `.scratch/realdata-preliminary-20261009/A1-swiglu-off/` |
| A2 | swiglu | adaptive | 0.01 | `.scratch/realdata-preliminary-20261009/A2-swiglu-adaptive/` |

One seed (42) per arm; persistence_skip k=0; baseline config otherwise. Early
stopping is armed (patience 20); runs are not "exact 300".

| Arm | epochs run | best epoch (0-idx) | best val loss |
| :-- | :-- | :-- | :-- |
| baseline | 300 | 295 | 7.196 |
| A1 | 292 (early stop) | 271 | 2.695 |
| A2 | 296 (early stop) | 275 | 2.364 |

Training-time val loss is the training objective, not an evaluation metric;
evaluation is §2.

## 2. Primary metric and the declared 78-slot family

Protocol §3 fixes **raw pooled MSE** as the primary metric; the 78-slot grid
(protocol §4) adds paired effect sizes and prefix-cluster sign-flip
diagnostics. Nothing in this phase is a confirmatory hypothesis test: protocol
§4 defines the sign-flip p as "a conditional descriptive diagnostic under a
stated joint sign-exchangeability null".

Validation set: 432 trials, 12 recording prefixes, 1,036,748 frames.
Delta = candidate MSE − comparator MSE on identical eligible frames (negative
favours the candidate); `d_z` is trial-level (mean delta / sd of trial deltas).

**Primary (protocol-literal) scoring run** — `scoring/`: exchangeability not
asserted, so all 78 p-values are null (`valid_test_count=0`); effect sizes,
deltas and prefix counts are as tabulated below.

**Sensitivity run, chosen after seeing the primary output** —
`scoring-exchangeability-asserted/`: the same family with the joint
sign-exchangeability null asserted (`valid_test_count=78`); effect sizes are
byte-identical. Protocol §4 did not say which setting applies, and the flag was
set only after the primary run returned no p-values, so these p-values are a
disclosed post-hoc sensitivity analysis, conditional on that null, and carry no
confirmatory weight. Recording prefixes are not verified animals, so prefix
means may not be independent.

**Resolution limit.** With 12 prefixes the smallest attainable p is
2/4096 ≈ 0.00049 and Holm's first threshold is 0.05/78 ≈ 0.00064, so a slot
can reach Holm significance only if all 12 prefix means share one sign. Bands
with 10–11 eligible prefixes cannot reach it at all.

**Primary metric — pooled validation MSE** (frames as above): zero 27.34,
baseline 6.60, A1 2.37, A2 2.07 (descriptive; no test is declared on it).

Family grid ("Holm p" = sensitivity run only):

| Comparison | aligned_overall: mean Δ, d_z, prefixes favouring candidate, Holm p | 12 sustained cells |
| :-- | :-- | :-- |
| A1 vs baseline | −4.22, −0.63, 12/12, **0.038** | Δ −115 to −1518, d_z −0.65 to −2.01, 10–12 of 10–12 prefixes; 2/12 below the Holm threshold |
| A2 vs baseline | −4.52, −0.65, 12/12, **0.038** | Δ −123 to −1570, d_z −0.68 to −2.10; 0/12 below the Holm threshold |
| A1 vs persistence | **+0.46**, +0.31, 2/12, 0.13 | Δ +11 to +98 (worse), d_z +0.10 to +0.25, 0–2 prefixes favour A1; 0/12 below the Holm threshold |
| A2 vs persistence | **+0.16**, +0.24, 3/12, 0.49 | Δ +3 to +50 (worse), d_z +0.09 to +0.29, 1–4 prefixes favour A2; 0/12 below the Holm threshold |
| A1 vs zero | −24.98, −0.91, 12/12, **0.038** | 8/12 below the Holm threshold; the other 4 have only 10–11 eligible prefixes, all favouring A1 |
| A2 vs zero | −25.28, −0.91, 12/12, **0.038** | 8/12 below the Holm threshold; same 4 resolution-limited cells |

In the sensitivity run, 22 of 78 slots fall below the Holm threshold, all
favouring the candidate (2 vs baseline overall, 2 A1 sustained cells vs
baseline, 18 vs zero). Full tables:
`.scratch/realdata-preliminary-20261009/scoring/realdata_preliminary_family.json`
(primary) and
`.scratch/realdata-preliminary-20261009/scoring-exchangeability-asserted/realdata_preliminary_family.json`
(sensitivity).

Reading, bounded by §6:

- Both arms have lower observed-history error than the frozen baseline on this
  split (pooled MSE 2.37/2.07 vs 6.60; d_z ≈ −0.63/−0.65 overall, every prefix
  agreeing). One seed each against one baseline seed.
- **Neither arm beats persistence.** Both have *higher* error than simply
  carrying the last observed velocity forward, overall and in every sustained
  band, with most prefixes agreeing on the direction; none of these
  differences falls below the Holm threshold even in the sensitivity run. The
  sustained escape bands do not reverse this.
- A1 vs A2 is not in the declared family; there is no declared comparison
  between them. Descriptively A2's pooled MSE is lower (2.07 vs 2.37) and its
  gap to persistence smaller, from one seed each — not evidence of an
  adaptive-recursion advantage.

## 3. A2 compute accounting (descriptive, §6)

Over 1,036,748 valid validation frames A2 used a mean refinement depth of 2.03
steps (configured cap `refinement_max_steps=4`; no observed maximum is
recorded), 2,101,858 refinement updates in total, and a pooled mean ponder
cost of 2.44. The condition split is available but degenerate: the validation
split has 0 wind_only frames, so all frames fall in the visual-present group.
The ponder cost is a compute proxy, not measured work or ATP.

## 4. Biological battery (§5)

Outputs: `.scratch/realdata-preliminary-20261009/battery/<arm>/`. All
descriptive; gating and integration use the whole corpus (2304 trials,
in-sample).

| Analysis | baseline | A1 | A2 |
| :-- | :-- | :-- | :-- |
| Dynamics (raw GRU + LIF) | ok; states finite; PCA top-3 71% | ok; top-3 82% | ok; top-3 86% |
| Routing gates (whole-corpus) | ok; g_lif wind 0.805 > visual 0.677; d=1.05 | ok; 0.738 > 0.619; d=1.01 | ok; 0.799 > 0.650; d=1.11 |
| Gating clustering (unsupervised) | ok; ARI 0.12 (3-way, k=3) / 0.20 (4-way, k=4); NMI 0.16 / 0.22 | ok; ARI 0.11 / 0.19; NMI 0.16 / 0.22 | ok; ARI 0.11 / 0.11; NMI 0.16 / 0.18 |
| Lesion / readout ablation² | ok; GRU substitution ΔMSE +28.97 (d_z 1.96) vs LIF +0.02 (0.52) | ok; GRU +37.18 (1.54) vs LIF −0.03 (−0.96) | ok; GRU +48.44 (2.10) vs LIF +0.27 (1.79) |
| Psychophysics | **unavailable** (0/432 val multisensory_ttc_0ms)¹ | **unavailable** (same) | **unavailable** (same) |
| Integration window (in-sample) | ok; wind_only latency +248 ms, multisensory −54 to −144 ms | ok; +246 ms, −51 to −139 ms | ok; +255 ms, −57 to −141 ms |
| Jacobian / fixed points (JAX GRU-Jacobian backend, `--backend jax`) | **withheld**: frozen-map gate unavailable — two-component fit degenerate in every epoch³ | **withheld**: same | **withheld**: same |
| Full-system Jacobian | ok; surrogate input singular values, no stability claim | ok; same | **unavailable by design**: 300 sampled candidates (100/epoch × 3 epochs) all refused because a refinement module is present |
| Flax forward-eval wrapper | not run (no `scripts/` caller) | not run | not run; refuses adaptive refinement by design |
| Autoregressive (k=0; synthetic paradigms, not the real corpus) | ok; 9 synthetic paradigms finite; V_peak 0.9–29.3 cm/s | ok; V_peak 75.8–91.3 cm/s | ok; V_peak 4.7–88.6 cm/s |

¹ The baseline battery ran before the runner read the psychophysics JSON, so its
`battery_status.tsv` says `ok`; its `bayesian_reliability.json` says
`not_applicable`. The JSON is authoritative.

² Lesion ΔMSE and d_z here are computed on paired **recording-prefix means**
(`effect_size_unit=paired_recording_prefix_means`), a different unit from the
trial-level d_z in §2; ΔMSE is the mean of per-prefix (lesioned − intact)
means. It runs over the whole corpus and scores one post-reference window, with
no onset vs sustained split; p-values are withheld
(`unavailable_unverified_animal_identity_and_independence`).

³ `withheld_reason=frozen_map_gate_unavailable`: the two-component fit that
identifies a quasi-fixed-point population lacks support in one component.
Frozen-input residual medians per epoch (early/transient/sustained) are
0.60/0.08/0.11 (baseline), 0.94/0.14/0.23 (A1), 0.98/0.27/0.02 (A2), with
maxima 3.06/3.22/3.32; the residual is secondary context, not the operative
gate.

### 4.1 Biological reading by hypothesis

Each row states what this phase can and cannot license. Unavailable or withheld
items are reported with their reason, never as negative results.

| Hypothesis | Evidence | Licensable now |
| :-- | :-- | :-- |
| H1 fast/slow division with LIF→GRU hand-off; wind relies more on LIF | gating by condition; §4.2 checks | Higher g_lif for wind than visual in all three arms (in-sample). **But** the measured time constants invert the premise (LIF ≈ 95 ms, GRU leak ≈ 3–5 ms, §4.2.4), and the condition difference in gating cannot be separated from input statistics (§4.2.3), so it is not evidence of condition-specific routing. "LIF is the fast path" is not supported. |
| H2 double dissociation: LIF lesion hits onset, GRU lesion hits the sustained phase | lesion | **Untestable** with this battery (single window). Descriptively, replacing the GRU output costs far more than replacing the LIF output in every arm. |
| H3 integration window; reliability-weighted / Bayes-optimal integration | integration; psychophysics | Window: in-sample description only. Per `scripts/analyze_integration.py:1207-1243` (no reference label is written to the integration outputs), visual-containing conditions are referenced to a computed visual-collision frame (argmax of the visual channel) and wind_only to the crop anchor (wind onset); since the references differ, latencies are not compared across conditions here. Optimal integration: **untestable** (no TTC=0 trials). |
| H4 dynamics / attractors | Jacobian | **No conclusion** — withheld in all arms (§5.2). |
| H5 sparse spike coding | LIF statistics | Plausibility check only; states finite. |
| H6 closed-loop behaviour generation | autoregressive | Model sufficiency only: all arms generate finite trajectories for 9 synthetic paradigms (`scope=synthetic_autoregressive`, not bound to the real corpus); peak speeds differ widely between arms. |

### 4.2 Exploratory checks (NOT pre-registered; labelled exploratory)

These sit outside protocol §5 and cannot enter the declared-family results; they
qualify the reading above and feed the next pre-registration. Single seed,
CPU, nested validation split (432 trials, 1,036,748 frames) unless stated.
Outputs: `C:/Users/Wray/.claude/jobs/a4ff95ef/tmp/realdata-exploratory-20261010/`
(`exploratory_summary.json`, `timecourse.csv`, `trial_level.csv`).

1. **Cross-architecture consistency** (window aligned to the dataset crop anchor, not verified stimulus onset: −50 to +250 frames,
   i.e. −200 to +1000 ms at 4 ms/frame, all 2304 trials; `timecourse.csv`).
   The three arms agree on gating direction per condition: mean g_lif >
   g_gru for wind_only (g_lif baseline 0.694, A1 0.601, A2 0.635) and
   visual_only (baseline 0.578, A1 0.511, A2 0.510); g_gru > g_lif for
   multisensory (g_lif baseline 0.466, A1 0.401, A2 0.398). These window means
   differ from the whole-trial gating means in §4. The first frame at which
   g_gru overtakes g_lif after g_lif ≥ g_gru:
   - wind_only: baseline +29, A1 +21, A2 0;
   - multisensory: baseline +12, A1 −11, A2 +1;
   - visual_only: baseline no crossing in the window, A1 +49, A2 +57.

   Direction replicates across architectures; hand-off timing does not.
2. **Gate collapse.** None. Mean g_gru 0.34/0.40/0.37 (sd 0.27/0.27/0.31);
   frames with g_gru > 0.95: 0.8%/6.9%/14.4%; < 0.05: 0.01%/0%/10.6%. A2's
   routing is the most bimodal.
3. **Input-statistics confound.** The multisensory − visual_only g_lif
   difference (+0.093/+0.078/+0.098) vanishes (residual ≈ −2e-5) after
   regressing per-trial g_lif on four input statistics; mean wind input
   dominates. Because wind input is nearly determined by condition, this test
   cannot separate the two — but it gives no evidence of a condition-specific
   routing policy beyond the inputs.
4. **Effective time constants.** LIF: τ = −dt/ln(0.9587) = 94.8 ms. GRU: in
   this code `h_t = (1−z)·n + z·h_{t−1}`, so z is a per-step retention and
   τ = −dt/ln(z): median 3.1/4.7/4.4 ms (IQR 1.7–6.1/2.9–7.6/3.0–7.0 ms). By
   leak time constant the GRU is the fast pathway and the LIF the slow
   integrator — the reverse of the architecture's premise. Caveat: the
   z-derived τ is a leak timescale; recurrent dynamics through the candidate
   state can still create slower network modes, which this check does not
   measure. Likewise the LIF has a 5 ms synaptic filter (`lif_tau_syn`) and
   emits discrete spikes, so its event output can react faster than its
   95 ms membrane leak suggests. "Fast" and "slow" therefore need a
   response-latency measure, not leak constants alone, before either pathway
   is labelled.

## 5. Deviations and withheld analyses

### 5.1 Deviations from the protocol

1. **Psychophysics not applicable on the declared scope.** Protocol §5 marked it
   available for all arms, but the nested validation split has no
   `multisensory_ttc_0ms` trials, so the noise sweep cannot run. Reported as
   unavailable, not as a negative result.
2. **Exchangeability asserted only after the primary scoring run returned no
   p-values** (§2). The protocol-literal run (`scoring/`) is primary; the
   asserted run is a disclosed post-hoc sensitivity analysis with no
   confirmatory weight. Both outputs are retained.
3. **Jacobian acceptance control.** Protocol §5 states the control as
   "residual <= 0.3"; the code's operative gate is the two-component-fit
   check (`frozen_map_gate_unavailable`), which failed first (§5.2). The
   outcome is the same (withheld), but the recorded gate differs from the
   protocol's stated control. The run-level `battery_status.tsv` note
   (`frozen_input_control_failed: no epoch produced a quasi-fixed-point
   population`) summarises the same per-epoch failure.

### 5.2 Why the Jacobian is withheld

The recorded reason, in all three arms and all three epochs, is
`frozen_map_gate_unavailable`: the two-component fit used to identify a
quasi-fixed-point population is degenerate (one component lacks support), so no
own-input spectrum is publishable. The frozen-input residual (medians 0.02–0.98,
maxima 3.06–3.32 across arms and epochs) is secondary context, not the operative
gate. No gate or threshold is relaxed after the fact. Candidate reasons, none
established here:

- the pooled-median frozen input may correspond to no real state, leaving too
  few candidates near a fixed point for the fit;
- the analysis covers GRU state only and omits LIF state;
- the strict fixed-point requirement may be too strict;
- the system may genuinely have no fixed points (transients, slow drift, limit
  cycles) — which would itself be a result.

Next-phase methods (to pre-register, not to apply now): per-condition frozen
inputs, slow-point (q-function) analysis, local linearisation along
trajectories or Lyapunov exponents, and a state space that includes LIF state.

## 6. Claim boundaries (inherited from protocol §8)

- Preliminary and incomplete: data collection is unfinished; no `no_stimulus`
  condition exists yet.
- Recording prefixes are not animals; no animal-level or circuit-level claim.
- Validation-guided selection on `nested_outer_validation` only; no held-out
  test claim.
- Single seed per arm: seed variance is unmeasured.
- The MoR router is a representational routing gate, not a causal estimator.
- The wind-vs-visual routing contrast is unavailable on the validation scope
  (0 wind_only val trials); only whole-corpus in-sample description is given.
- A2's ponder cost is a compute proxy, not measured biological or ATP cost;
  no adaptive superiority is asserted.
- **Circularity:** the architecture builds in a LIF path and a GRU path, so a
  fast/slow division may be imposed rather than discovered. It counts as a
  finding only after measured time constants, a symmetric (same-type pathway)
  control, cross-seed and cross-architecture consistency, and behaviourally
  testable predictions (§8).
- Positioning: computational sufficiency plus consistency with known
  mechanisms — not proof of a cricket neural mechanism. Related systems
  (Drosophila giant-fibre vs non-giant-fibre escapes, crayfish giant vs
  non-giant escapes, the cricket cercal–giant-interneuron system, locust
  LGMD/DCMD) are analogies; novelty, if any, lies in multisensory integration
  and hand-off timing and needs a systematic literature search.

## 7. Preliminary conclusions

Preliminary, on incomplete data, one seed per arm, nested validation split
only. Conclusions 1–3 come from the pre-registered primary metric and declared
family; per protocol §4 they are descriptive, and the Holm p-values cited exist
only in the post-hoc sensitivity run (§2). Conclusions 4–7 are descriptive or
exploratory.

1. **Both SwiGLU arms are associated with lower observed-history error than
   the frozen relu baseline on this corpus's nested outer validation**
   (pooled MSE 2.37 for A1 and 2.07 for A2 vs 6.60; overall d_z ≈ −0.63 and
   −0.65, all 12 recording prefixes agreeing; sensitivity-run Holm p = 0.038).
2. **Neither arm beats persistence.** Carrying the last observed velocity
   forward has lower error than A1 and A2 overall and in every sustained
   escape band (none below the Holm threshold even in the sensitivity run). On this corpus the models do not yet
   add predictive value beyond the observed history, so improvements over
   the baseline must not be read as better-than-trivial forecasting.
3. **Both arms have lower error than the zero predictor** overall and in 8 of 12 sustained
   cells below the sensitivity-run Holm threshold (the other 4 are
   resolution-limited).
4. **No adaptive-recursion advantage is shown.** A2 vs A1 was not a declared
   comparison; A2's lower pooled error (2.07 vs 2.37) is one seed each. A2
   used a mean of 2.03 refinement steps of a possible 4.
5. **Routing direction replicates across three architectures** — more LIF
   weight for wind and visual-only, more GRU weight for multisensory — but
   the condition difference cannot be separated from input statistics (wind
   input is nearly determined by condition), and the hand-off timing is not
   consistent. This is description, not evidence of a
   condition-specific routing policy.
6. **The "fast LIF / slow GRU" premise is not supported.** By leak time
   constant the GRU is faster (≈ 3–5 ms) than the LIF (≈ 95 ms); a proper
   response-latency measure is still needed. The lesion analysis cannot test
   the onset vs sustained dissociation (H2).
7. **Dynamics (H4) and optimal integration (H3) remain open**: the Jacobian
   is withheld in all arms and the TTC=0 condition is absent.

What would change these conclusions: multiple seeds, wind_only and
`no_stimulus` trials in held-out data, verified animal identity, a windowed
lesion analysis, latency-based pathway timing, and the §5.2 dynamics methods
(§8).

## 8. Next phase (to pre-register)

- Scale for publication: 6–10 model configurations × 5 seeds (about 30–50
  training runs; zero and persistence need no training or seeds, so the count
  sits toward the lower end). Seed variance is the main gap of this phase.
- Data: `no_stimulus` trials, wind_only trials in validation, TTC=0 trials,
  verified animal identity; an independent test set, ideally split by animal.
- Controls: zero, persistence, LIF-only, GRU-only, fixed 50/50 routing,
  parameter-matched RNN/LSTM, relu and SwiGLU variants; for any adaptive-
  recursion claim, fixed-depth K2–K4 and adaptive with matched compute budget.
- Biology: measured effective time constants, symmetric-pathway control,
  behavioural predictions (latency vs TTC, effects of disrupting cercal input),
  and the §5.2 dynamics methods.
- Throughput: a faster GPU speeds a single run only modestly (estimate, not
  profiled; the T=2400 step loop is launch-bound); extra GPUs pay off by running
  seeds in parallel. Code-level speedups go through the controlled-change
  protocol and must preserve A2's "actually executes fewer steps" semantics.
