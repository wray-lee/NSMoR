# Model-free exploratory behavior analyses (2026-10-10)

**Status: EXPLORATORY / IN-SAMPLE / BETWEEN-GROUP / MODEL-FREE (qualified).**
Every hypothesis below was generated on this corpus, no data were held out,
and condition is confounded with recording prefix (all 64 prefixes are
condition-pure). These numbers motivate a confirmatory design; they do not
establish integration.

**"Model-free" is qualified.** The PRIMARY race / facilitation / accumulation
statistics use only the 10 cm/s velocity crossing on the physical channels
(visual angle, wind state) — no checkpoint, no model. The **s1
labeling-responder quantities** (the `labeling_responders` ledger column, the
s1 race V term and the s1 rate contrast) instead reuse the labeling pipeline's
own response criterion (`nsmor.pipeline.labeling._check_sustained_speed`), so
they depend on the labeling pipeline's rule; they are not learned-model
outputs, but they are not the raw crossing either. The D1 revision documented
here is **round 3** (round 2 doc sha `3d9cc2fe…` is superseded); this revision
additionally applies the round-3 review fixes (X1–X4): the s1 responder is now
**eligibility-gated** exactly like the primary response, and the race /
facilitation families carry a **Holm** family-wise correction. Those deltas are
recorded under "Changed vs round 3".

Round 2's central blocker was a **trigger inconsistency**: the visual response
was searched over `[collision − 2000 ms, collision + 1000 ms]` while
eligibility was gated over `[collision − 250 ms, collision]`, so the response
window reached ~1750 ms BEFORE the gate. A "visual responder" could therefore
be a pre-gate (pre-collision) mover with a **negative latency** — a reviewer
reproduced only 2/53 such "responders" with latency > 0. Round 3 (D1) uses ONE
trigger definition per modality for BOTH the eligibility gate and the response
search. A "Changed vs round 2" section states exactly which conclusions moved;
a "Changed vs round 3" section records the eligibility-gating and Holm
corrections applied in this revision.

Script: `scripts/analyze_behavior.py` (no checkpoint, no model, no training).
Run: `make behavior DATA=<dataset> OUTPUT=results/behavior-20261010`.
Outputs: `results/behavior-20261010/behavior/behavior_summary.json` plus four
figures. All results below are from that re-run. The round-1 files at the TOP
LEVEL of `results/behavior-20261010/` are superseded (see the README there);
only the `behavior/` path is cited.

## Corpus

- 2304 trials: visual_only 720, wind_only 396, multisensory 1188.
- Labels: ESCAPE 1067, PREWALK 126, PRE_ACTIVE 181, NO_RESPONSE 930.
- 128 `session_ids` collapse to **64 recording prefixes** (each prefix = 2
  sessions, 36 trials); prefixes are condition-pure (visual_only 20,
  wind_only 11, multisensory 33). Animal identity is UNVERIFIED, so a prefix
  is the working cluster unit and not a proven animal.
- dt = 4 ms; escape threshold 10 cm/s; response window W = 2000 ms.

Reference (trigger) events, reused from `nsmor.pipeline.conditions`
(`derive_anchor_frames`, the same onset reference the labeling pipeline uses):
wind onset for wind-containing trials, visual collision (peak angle) for
visual_only. Reference events, definitions and flags are stamped in the JSON
(`meta.definitions`, `meta.exploratory`, `meta.in_sample`,
`meta.between_group_design`).

## Resampling unit (B1, unchanged)

The cluster unit is the **64 recording prefixes**. The bootstrap is a **true
with-replacement cluster bootstrap**: within each group exactly `len(pool)`
prefixes are drawn with replacement, and a prefix drawn `m` times carries
multiplicity weight `m` on every one of its trials. All prefix-cluster CIs
below use the multiplicity weights; `tests/test_analyze_behavior.py` checks
them against an independent reference cluster bootstrap that concatenates
resampled clusters explicitly.

## D1 trigger rule (one definition per modality)

The trigger is the SAME per-condition reference for the eligibility gate and
the response search:

- **Eligibility** — `|Y| < 10 cm/s` at every frame of `[trigger − 250 ms,
  trigger]` (the stationarity gate).
- **Response** — the first frame **strictly after** the trigger
  (`> trigger`) with `|Y| >= 10 cm/s` inside `[trigger, trigger + 2000 ms]`
  for **every** condition. For visual_only the trigger is the collision, so
  this window covers `collision + 1000 ms`; latency is always `> 0`.

The round-2 collision-anchored window is retained only as the labeled
**sensitivity s3** (`resp_legacy` / `latency_legacy_ms`).

Eligibility and responders by condition (`eligibility` in the JSON):

| condition | n | eligible | excluded | excl % | elig (5 cm/s) | responders (D1) | labeling responders (s1, gated) | ungated (s1, diagnostic) | s3 legacy window | s2 all-trials |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visual_only | 720 | 497 | 223 | 31.0 | 471 | **8** | **3** | 80 | 276 | 144 |
| wind_only | 396 | 385 | 11 | 2.8 | 365 | 298 | **123** | 132 | 309 | 309 |
| multisensory | 1188 | 852 | 336 | 28.3 | 816 | 830 | **731** | 1065 | 1166 | 1166 |

The **D1 primary visual responder count collapses from 53 (round 2) to 8**: the
other 45 round-2 "responders" crossed before the collision trigger. The
labeling-consistent gate (`|Y| < 5 cm/s`) excludes a further **5.2% of the
eligible visual_only trials (497→471), 5.2% of wind_only (385→365) and 4.2% of
multisensory (852→816)** — reported per condition, not as a single figure.
**Sensitivity labels** (nested inside / alongside the primary, never chosen by
outcome): **(s1)** response = labeling sustained `> 5 cm/s` for 250 ms with a
200 ms initiation bound, **under the SAME D1 eligibility gate as the primary
response**; **(s2)** the old all-trials rule (no stationarity gate), a
contamination illustration only; **(s3)** the round-2 collision-anchored
visual window.

**The s1 labeling responder is eligibility-gated (round-3 fix).** It is the
same D1 gate as `resp`, so it counts **3 / 123 / 731** for
visual_only / wind_only / multisensory — not the ungated 80 / 132 / 1065. The
ungated count is kept only as the clearly named `labeling_responders_ungated`
diagnostic and is **never** the D1 labeling responder. The gate removes 77
visual_only, 9 wind_only and 334 multisensory ungated responders (pre-trigger
movers).

## D3: labeling responders vs the recomputed criterion

Two different objects are now kept apart. The **recomputed** criterion applies
the D1 trigger/window **and the D1 eligibility gate** to the sustained
`> 5 cm/s` rule (`labeling_responders` in the eligibility ledger: **3 / 123 /
731**). The **stored** dataset label (`ESCAPE`/`PREWALK`, produced by the
labeling pipeline on the raw `stimulus_onset` clock) appears ONLY in the
separately named `stored_label_agreement` table:

| condition | recomputed (gated) | stored | agreement | recomputed-only | stored-only |
| :--- | ---: | ---: | ---: | ---: | ---: |
| visual_only | 3 | 1 | 0 | 3 | 1 |
| wind_only | 123 | 128 | 118 | 5 | 10 |
| multisensory | 731 | 1064 | 727 | 4 | 337 |

The stored labels are near-deterministic in `condition`: visual_only carries a
single stored ESCAPE, and the stored count is dominated by wind-containing
trials. The **single stored visual_only ESCAPE is a visual-sustained crossing,
not a wind-triggered one**: it is the trial with 0 wind-active frames and a
sustained `> 5 cm/s` run beginning ~52 ms into the trial (verified on the
corpus), so it is *not* explained by wind leaking into a "visual_only"
recording. The gated recomputed criterion applied to visual_only finds 3
sustained crossings the stored labels do not record (the ungated criterion
finds 80). Neither object is privileged as ground truth.

## Race model (D1/D2/D5, M1)

A = wind_only (clock at wind onset), V = visual_only (clock at collision),
AV = multisensory (clock at wind onset). Each multisensory TTC variant is a
separate between-group comparison against the SAME eligible wind_only and
visual_only populations. The visual term is shifted by the variant's
wind→collision SOA: 356 / 288 / 244 / 208 ms. Non-responders stay in the
denominator (defective CDFs). The Miller bound is clipped at 1 (M1).
**Eligible-trial CDFs at 2 s: F_A 0.774, F_V 0.016, F_AV 0.974.**

| group | n_AV | n_pref | violation area (prob·ms) | 95% prefix CI | peak (ms) |
| :--- | ---: | ---: | ---: | :--- | ---: |
| pooled | 852 | 33 | 389.4 | [110.4, 723.1] | 96 |
| TTC −373 | 289 | 10 | 331.7 | [37.7, 664.6] | 96 |
| TTC −308 | 237 | 10 | 430.7 | [158.3, 785.0] | 96 |
| TTC −261 | 257 | 10 | 424.8 | [152.9, 782.3] | 100 |
| TTC −225 | 69 | **3** | 357.5 | [54.0, 714.5] | 100 |

Area units are probability·ms. **TTC −225 has only 3 prefixes and is flagged
`informative=False`.** The peak-violation time is stored as
`violation_peak_time_ms` (96–100 ms), not hardcoded.

**B3 visual term.** Under D1 the visual-only arm keeps 497 eligible trials but
only **8 respond** (F_V = 0.016; the gated labeling criterion leaves just 3).
F_V is therefore small; the bound is dominated by F_A, so the pooled
"violation" is largely **AV vs A facilitation beyond the wind-alone CDF**, not
a clean coactivation excess. The s1 labeling-responder V term makes this
explicit: F_V collapses to 0.006 (3 responders) and the area rises to 1034 —
under the sustained criterion the race test is a facilitation test.

**D2 primary inference.** The round-2 **single-modality permutation** (wind vs
visual labels across single-modality prefixes, multisensory CDF held fixed) is
**removed as primary**: a race model places no restriction on which modality
wins a trial, so relabeling wind vs visual does not sample from the race
hypothesis. The primary test is a **prefix-level permutation of the AV vs A
group label**: whole multisensory variant prefixes vs whole wind_only prefixes,
statistic `area(max(0, F_AV − bound))`. Because under D1 the visual term is
small, the bound is close to F_A, so this tests AV facilitation beyond the
wind-alone CDF (the boundary case of the race null). Per variant and pooled,
with a **Holm** family-wise correction over the declared 5-test family (pooled
+ the 4 TTC variants; the family is declared before the p-values are read):

| group | observed | null mean | null 95th | raw one-sided p | Holm p | n_pref |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| pooled | 389.4 | 51.6 | 234.5 | **0.0015** | **0.0075** | 33 |
| TTC −373 | 331.7 | 75.8 | 311.4 | 0.030 | 0.060 | 10 |
| TTC −308 | 430.7 | 91.7 | 338.1 | 0.0050 | **0.015** | 10 |
| TTC −261 | 424.8 | 83.5 | 340.2 | 0.0025 | **0.010** | 10 |
| TTC −225 | 357.5 | 151.3 | 396.4 | 0.115 | 0.115 | 3 |

The permutation recomputes **both** CDFs (the relabeled AV group's `F_AV` and
the relabeled A group's `F_A`), so the null is a true label permutation of the
AV-vs-A contrast, not a fixed-`F_AV` recompute. **The race violation DOES
survive D1, as an AV-vs-A facilitation excess**: the pooled observed area
(389.4) sits above the prefix-relabeling null mean (51.6) with raw one-sided
p = 0.0015 (Holm p = 0.0075). Under the Holm correction **three of the five
family tests survive at 0.05 — pooled (0.0075), TTC −308 (0.015) and TTC −261
(0.010) — while TTC −373 does not (raw 0.030 → Holm 0.060)**; TTC −225 is
uninformative (3 prefixes, p = 0.115). The round-2 conclusion — that the
violation was below chance (p = 0.99) — was an artifact of the wrong null and
the pre-collision "responders"; both are corrected. The claim remains
**AV-vs-A facilitation**, not a clean Miller-race coactivation excess, because
F_V is small.

**D5 race power.** The signed one-sample race power is **dropped** (a violation
area is non-negative). A simulation-based power for the D2 permutation test is
reported instead: the fraction of cluster-bootstrap violation areas above the
permutation null's 95th percentile (234.5) is **0.826** at n_boot = 2000
(`race_model.power_analysis`). This is a self-referential planning figure (the
bootstrap resamples the observed clusters), not power against a specified
alternative.

**Interpretation (conservative, between-group).** A violation supports
coactivation only if it exceeds the bound; a null would NOT be evidence
against integration. Context invariance and the same-population assumption
are required and unverified. Because condition is confounded with prefix, a
violation cannot be separated from a session/animal effect, and the
permutation test does not clear that confound.

## Facilitation: wind_only vs multisensory (D1/D4/D5)

Eligible-only latency from the trigger: wind_only median 120 ms [104, 136]
(n 298, rate 0.774 [0.729, 0.813], 11 prefixes); multisensory median 84 ms
[72, 104] (n 830, rate 0.974 [0.961, 0.983], 33 prefixes). Both latency samples
are non-normal (Shapiro p < 1e-15); Levene p = 0.48 (equal variance here).

**D4 headline inference is prefix-level.** Condition is **perfectly nested** in
prefix, so the fixed effect is a **between-prefix** comparison and the random
intercept **cannot** adjust the session confound. Primary reference:
per-prefix median latencies with a Welch t and a label-permutation test — 11
wind vs 33 multisensory prefixes, mean difference **+38.4 ms**, Welch
t = 8.53 (p = 1.1e-8), permutation p = 0.0002 (Holm 0.0010), Cliff's
delta = 0.95, Cohen's d = 2.53. Per TTC variant (each vs the same wind_only),
with the same **Holm** correction over the declared 5-test facilitation family
(pooled + the 4 variants) applied to the permutation p:

| TTC variant | n_trials (wind / multi) | n_prefixes (wind / multi) | mean diff (ms) | Welch p | perm p | Holm p | informative |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| pooled | 385 / 852 | 11 / 33 | +38.4 | 1.1e-8 | 0.0002 | 0.0010 | yes |
| TTC −373 | 385 / 289 | 11 / 10 | +37.1 | 2.0e-4 | 0.0002 | 0.0010 | yes |
| TTC −308 | 385 / 237 | 11 / 10 | +40.9 | 4.6e-7 | 0.0002 | 0.0010 | yes |
| TTC −261 | 385 / 257 | 11 / 10 | +37.9 | 3.7e-5 | 0.0004 | 0.0010 | yes |
| TTC −225 | 385 / 69 | 11 / **3** | +36.1 | 6.1e-7 | 0.0034 | 0.0034 | **no** (3 prefixes) |

**Trial-level tests are DEMOTED to descriptive** with an explicit
`validity_note` ("NOT valid under nesting"): the trial-level Mann-Whitney
(U = 206245.5, p = 8.2e-66, Cohen's d = −0.53, Cliff's delta = −0.67) and the
response-rate Fisher test (0.774 vs 0.974, p = 3.9e-28, phi = 0.327) treat
trials as independent, which they are not. The median difference is −36 ms
(prefix-cluster bootstrap CI [−40, −24]). The LMM is reported for continuity:
intercept 127.0 (se 5.8), condition(multisensory) −36.3 ms (se 6.7,
z = −5.39, p = 7.2e-8), between-prefix SD 14.8, residual SD 60.2, 44 groups,
1128 obs.

**D5 power.** Using the SD of the per-prefix median (wind 11.7, multisensory
16.1, pooled 15.1 ms), the observed −36 ms needs **5 prefixes per group** for
80% (not `sigma_b` alone); the corpus has 33 multisensory and 11 wind_only
prefixes, so the facilitation contrast is well-powered. The `nct` small-sample
caveat applies (planning figure, not an exact target). Race power is **not**
estimated here; it is in `race_model.power_analysis` (above). The s1
labeling-responder rate contrast (eligibility-gated, over eligible trials) is
also large: **123/385 vs 731/852** (Fisher p = 4.1e-78); the ungated diagnostic
would read 132/396 vs 1065/1188.

Per-prefix pre-stimulus covariates (resting speed, spontaneous movement rate)
are in `facilitation.covariates`.

## Evidence accumulation (C6c) -- "suggestive"

Per-condition MLE fits (M4: normal / log-normal / exponential / gamma /
ex-Gaussian, all fitted on the same sample, AIC/BIC from each model's own
log-likelihood and true parameter count). wind_only (n 298) mean 126.1, SD
33.2, skew 1.76, ex-Gaussian best (tau 28.7 ms); multisensory (n 830) mean
93.2, SD 69.2, skew 13.2, ex-Gaussian best (tau 30.2 ms); visual_only (from
collision, n 8) too few to fit. Per-variant means/SDs: −373 (274) 102.6 / 111.5;
−308 (235) 84.9 / 30.8; −261 (255) 91.5 / 33.8; −225 (66) 89.9 / 21.1. Mean-SD
Spearman rho 0.80, p 0.20 over only 4 points — **suggestive**, not established.
Figure (c) labels it so.

## Negative / non-identifiable items

Reported in the JSON and here, without figures:

- **Discrete escape modes.** GMM on responder kinematics (n 1136) prefers
  k = 4 (bootstrap ARI mean 0.364). Caveats (JSON `caveats`): the fit is
  in-sample, has **no prefix grouping** (the ARI bootstrap resamples trials
  and cannot see the prefix clustering), and the feature matrix **mixes all
  three conditions**. Components track condition, so this is NOT a
  condition-independent escape-mode taxonomy.
- **Looming l/|v| rule.** Escape in multisensory is wind-locked, so escape
  time relative to collision is dominated by the wind, not the looming angle
  (median escape − collision: −264 / −208 / −156 / −120 ms for −373 / −308 /
  −261 / −225; read from `looming_trigger.per_variant`, not hardcoded). The
  l/|v| proxy is a kinematic identity here (4 points, one collision geometry),
  so an LGMD/DCMD-like fixed-angular-size threshold is NOT identifiable.
- **Reliability weighting.** Needs TTC = 0 or deliberate cue-conflict
  conditions; the corpus has four TTC variants but wind is always the sole
  trigger. Verdict: **NOT TESTABLE in this corpus**.

## Changed vs round 2 (which conclusions moved)

- **The visual trigger blocker is fixed.** Round 2 searched the visual response
  over `[collision − 2000 ms, collision + 1000 ms]` while gating eligibility
  over `[collision − 250 ms, collision]`, so responders could have negative
  latency. D1 uses `[trigger, trigger + 2000 ms]` for every condition. The D1
  primary visual responder count is **8** (was 53); the 45 dropped responders
  crossed before the collision.
- **The race inference is replaced.** The round-2 single-modality (wind vs
  visual) permutation is not a race-model null and is removed as primary. The
  D2 AV-vs-A prefix permutation gives **raw p = 0.0015** pooled (Holm 0.0075;
  three of five family tests survive correction, both CDFs recomputed): the
  violation **now survives** as AV-vs-A facilitation, reversing round 2's
  p = 0.99 "below chance" reading.
- **F_V is now 0.016** (was 0.107), so the bound is dominated by F_A and the
  remaining claim is facilitation, not coactivation.
- **Responder counts changed** because the D1 trigger/window and the
  stationarity gate are now consistent: multisensory primary responders
  830 (unchanged, wind-anchored), visual_only 53 → 8.
- **The signed one-sample race power is withdrawn** (D5); a simulation-based
  permutation power (**0.826** pooled, the single reported value, with its
  self-referential caveat) replaces it.
- **Facilitation is now reported per TTC variant + pooled** with explicit
  `n_trials` and `n_prefixes`; TTC −225 (3 prefixes) is flagged uninformative.
  The direction is unchanged (multisensory faster and responds more); the
  trial-level tests are demoted to descriptive.
- **D3 naming.** The recomputed sustained criterion and the stored dataset
  label are no longer both called "labeling responders": the stored one lives
  only in `stored_label_agreement`.
- **Accumulation and negative items are qualitatively unchanged**; the
  ex-Gaussian MLE refit keeps ex-Gaussian best for wind/multisensory, and the
  discrete-modes ARI is reported with its caveats.

## Changed vs round 3 (which conclusions moved)

- **The s1 labeling responder is now eligibility-gated (X1).** Round 3's doc
  claimed the s1 responder was gated but the code computed it ungated, so the
  ledger read 80 / 132 / 1065. It now applies the same D1 gate as `resp` and
  reads **3 / 123 / 731**; the ungated count survives only as the clearly named
  `labeling_responders_ungated` diagnostic and is never the D1 labeling
  responder. The `stored_label_agreement` recomputed column, the s1 race V term
  and the s1 rate contrast all follow the gated count.
- **The race and facilitation families are Holm-corrected (X2).** Raw and
  Holm-adjusted prefix-permutation p are now both reported over the declared
  5-test families. The race variant sentence changed from "all four variants
  are individually significant" to: **pooled, −308 and −261 survive; −373 does
  not (0.030 → 0.060); −225 is uninformative (3 prefixes)**.
- **One power value (X3).** The stale "0.663 pooled" power is removed; the
  single reported value is **0.826** with its self-referential caveat.
- **Minors (X4).** The single stored visual_only ESCAPE is attributed to a
  **visual-sustained crossing** (0 wind-active frames), not wind; "model-free"
  is qualified for the s1 labeling-responder quantities; the
  `stored_label_agreement` note states the gate.

## Figures

All figures carry an EXPLORATORY label and use eligible trials only. Each
figure's uncertainty note is in the JSON `figure_notes` list.

- (a) `race_model_cdfs.png` -- per-TTC F_AV vs the Miller bound (clipped at 1),
  violation shaded, bound 95% prefix-bootstrap band. The band is **bound-only**
  uncertainty (200 replicates) and does not include F_AV sampling variability.
- (b) `facilitation_latency.png` -- wind_only vs multisensory latency
  distributions and response rate (eligible).
- (c) `accumulation_latency.png` -- latency distributions with ex-Gaussian
  fits and mean-SD across TTC variants (labeled suggestive, 4 points).
- (d) `trigger_locking.png` -- descriptive panel (escape locked to wind onset).

## Limitations

- Condition == recording prefix; between-group only, animal identity
  unverified; the random intercept cannot adjust the session confound.
- Ill-defined visual onset / SOA anchor; a single looming geometry.
- No TTC = 0 and no cue-conflict condition; TTC −225 has only 3 prefixes.
- Under D1 the visual_only arm keeps only 8 responders, so F_V is small and
  the race test is dominated by AV-vs-A facilitation.
- In-sample, exploratory: the race family uses a prefix permutation test with
  a **Holm** family-wise correction (pooled + 4 variants); no BH-FDR on a
  pseudo-p.

## Data-collection recommendations

1. Randomize conditions within day and device so condition is not confounded
   with session.
2. Verify animal identity per recording; record it as a first-class key.
3. Vary TTC / looming speed within an animal to break the condition-prefix
   confound.
4. Include wind_only trials in the validation set for the race comparators.
5. Add TTC = 0 and cue-conflict conditions to make reliability weighting
   testable.

## Reproducibility

`scripts/analyze_behavior.py --dataset <ds> --dt_ms 4.0 --seed 42
--n_boot 2000 --n_perm 2000 --output_dir results/behavior-20261010/behavior`,
or `make behavior DATA=<ds> OUTPUT=results/behavior-20261010`. Tests:
`tests/test_analyze_behavior.py`.
