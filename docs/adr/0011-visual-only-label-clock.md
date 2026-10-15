# Visual-only label clock at the collision; first-stimulus reference for multisensory

**Status:** accepted (phase-3 prerequisites, 2026-10-15). Changes
`PIPELINE_SEMANTICS_VERSION` from `2.2` to `2.3`; `2.2` is retained
(readable, marked legacy).

## Context

The phase-3 protocol introduces multisensory paradigms defined by wind
TIMING relative to the looming sweep: wind at the 80-degree looming
angle (wind first), wind at the collision (SOA 0), and wind 200 ms AFTER
the collision (SOA -200 ms). The existing corpus carries only wind-first
variants (wind->collision SOA 356/288/244/208 ms). Two audited defects
(`.scratch/phase3-prereq-20261015/baseline-audit-v1.json`,
`bv-label-clock-looming-onset` and `pre-anchor-response-labels`) make the
current label clocks unusable for the new paradigms:

1. **Visual-only labels are clocked at the looming onset.** The staging
   adapter injects `stimulus_onset = looming onset` for visual-only
   trials, so the labeler's response window is
   `[looming_onset, looming_onset + 200 ms)` — seconds before the
   collision. A collision-locked visual escape can therefore never be
   labelled; on the phase-2 corpus 1 visual-only trial is stored
   ESCAPE while 8 are collision-anchored D1 responders.
2. **Wind-bearing labels are clocked at the wind onset.** For a
   wind-lagging trial (collision first) a collision-triggered escape is
   either excluded by the stationarity gate (it is still running at wind
   onset), misdated to ~0 ms after the wind, or — once ended — absorbed
   into PRE_ACTIVE. The Miller-bound shift `t - soa` is defined on the
   wind clock and cannot represent the collision-first case without
   moving the A term too.

## Decision

* **Visual-only (`bv`) trials**: the injected `stimulus_onset` — the
  label response clock, the snapshot anchor and the analysis D1 trigger
  — is the looming COLLISION,
  `looming_onset + lv_ratio_ms / tan(radians(init_deg/2))` with
  `init_deg = 2.0`, the same geometry `reconstruct_visual_angle` and the
  visual-only snapshot anchor (`looming_collision` rule) already use.
* **Multisensory trials**: the reference event is the FIRST stimulus
  event of the trial, `min(wind onset, collision)`. For every wind-first
  corpus this is the wind onset, bit-for-bit the historical behaviour;
  for wind-lagging variants it is the collision. Wind-only trials are
  unchanged (wind onset).
* The Miller bound is evaluated on that same reference clock: the
  visual term shifts by `max(0, soa)` and the wind-only term by
  `max(0, -soa)` per variant (`soa = collision - wind onset`, positive =
  wind first). For wind-first corpora this reduces bit-for-bit to the
  historical `F_A(t) + sum_k w_k F_V(t - soa_k)`.
* The descriptive trigger reading (`run_looming_trigger`, the §9
  "wind-locked trigger" signature) converts the reference-clock latency
  to the collision clock with the same `max(0, v_col - w_on)` shift:
  exactly zero for simultaneous and wind-lagging variants (whose
  reference IS the collision), the historical `v_col - w_on` shift for
  wind-first variants. `figure_trigger` already used this expression; the
  two now agree.
* New additive per-trial dataset keys record the paradigm identity:
  `stimulus_soa_ms` (signed channel-derived SOA), `stimulus_variant`
  (keyed on `(target_ttc_ms, SOA bucket)`, bucket width 50 ms) and
  `stimulus_reference_event`. Old loaders ignore them; a missing key
  means "unknown".
* The analysis keys every multisensory variant on
  `(target_ttc_ms, SOA bucket)` and fails closed when a variant's
  within-cell SOA spread exceeds the declared tolerance
  (`--soa_tolerance_frames`, default 1 frame). The phase-2 corpus's
  measured within-variant spread is 24-30 ms (6-7 frames, wind-delivery
  jitter), so phase-2 reruns must declare e.g. 8 frames explicitly
  instead of silently pooling.
* An undeclared multisensory `target_ttc_ms` (NaN) fails fast at the
  analysis entry with a named error listing the affected sessions.

## Retention policy (old artifacts stay readable)

`PIPELINE_SEMANTICS_VERSION` is `2.3`; `2.2` is listed in
`RETAINED_PIPELINE_SEMANTICS_VERSIONS` and every provenance gate
(checkpoints, datasets, nested-prior artifacts, evaluation scripts)
accepts it with an explicit warning that stamps the artifact as legacy.
Anything else (pre-2.0, unknown, absent) is still refused: those
artifacts carry leaked priors / frame-unit time constants that load
silently wrong. The frozen phase-2 corpus, its checkpoints and its
analyses therefore keep loading, clearly marked, without any re-ETL.

## Impact (read-only count, no corpus artifact written)

`scripts/impact_bv_relabel.py` replays the real staged phase-2 sessions
through the real labeling pipeline under both clocks (read-only; inputs
verified unchanged). Results in
`.scratch/baseline-fixes-20261015/bv-relabel-impact.json`:

- 720 visual-only trials; the old-clock replay reproduces every stored
  label (720/720, 0 mismatches), validating the replay;
- the collision clock yields 80 collision-locked responders where the
  stored v2.2 labels carry none: NO_RESPONSE→ESCAPE 27,
  NO_RESPONSE→PREWALK 53; the remaining transitions are
  NO_RESPONSE→PRE_ACTIVE 358, NO_RESPONSE→NO_RESPONSE 281 and
  ESCAPE→PRE_ACTIVE 1 (the single stored ESCAPE is not a sustained
  post-collision response). 439 of 720 labels change. The stored v2.2
  label arrays are NOT interchangeable with v2.3 for visual-only
  trials.

## Wind-first byte-identity (evidence)

`scripts/analyze_behavior.py` run at HEAD (pristine `git archive` tree)
and at the changed tree on the phase-2 corpus with identical arguments
(seed 42, n_boot 2000, n_perm 2000; the new run declares
`--soa_tolerance_frames 8`) produces **zero numeric differences**: every
D1 latency, eligibility count, CDF, violation area, bootstrap CI,
permutation p, Holm adjustment, facilitation test, accumulation fit and
GMM result is bit-identical. The only diffs are the mandated variant-key
renames (`ttc_<v>` -> `multisensory_ttc<v>_soa<b>`), added fields, and
prose. Evidence:
`.scratch/baseline-fixes-20261015/byte-identity-report.txt`
(`0 numeric/mismatch diffs`).

Staging a real multisensory session pair with the old and new adapter
produces byte-identical kinematics AND events CSVs; a real visual-only
session pair differs ONLY in its 18 `stimulus_onset` rows (re-clocked
from the looming onset to the collision), with byte-identical
kinematics. Evidence: `.scratch/baseline-fixes-20261015/adapter-proof/`.

The wind-presence guard is calibrated on the raw corpus: 9 non-{0,1}
`stim_state` samples across 8 of 128 files, all isolated single-sample
telemetry corruption with distinct values (e.g. -1, 88, 587, 9702); they
are reported and read as 0 (the historical reading), while any repeated,
adjacent or numerous non-binary value fails the staging closed.

## Consequences

* The real dataset is NOT rebuilt here. The re-ETL that produces v2.3
  labels for the real corpus is a separate, user-authorized step.
* Phase-2 artifacts (corpus, checkpoints, analyses) remain valid under
  their own 2.2 semantics and load with a legacy warning.
* `tests/test_adapter_event_time.py` assertions that pinned the old
  visual-only clock (the injected `stimulus_onset` = looming onset) are
  repaired to the collision clock via a shared `_collision_ms` helper;
  every other claim of those tests (per-trial relative ms, no double
  scaling, offset invariance, chronological-first Looming, source
  immutability) is unchanged.
* `tests/test_phase3_prereq_baseline_fixes.py` covers the sign of the
  SOA, the variant keying, the homogeneity audit, the first-stimulus
  trigger, the pre-reference responder class, the NaN-TTC fail-fast, the
  adapter guards and the version retention.
* `scripts/analyze_behavior.py` keeps `_variant_soa(m_multi, ttc, w_on,
  v_col, dt_ms, v)` at its historical signature and semantics — the
  frozen SOA accessor the phase-3 model-comparison spec imports and pins
  to 356/288/244/208 ms on the existing corpus. On a corpus where one TTC
  carries several paradigms the `(ttc, SOA bucket)`-aware
  `multisensory_variants` / `per_trial_soa_ms` pair is the API.
