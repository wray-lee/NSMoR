# Real-data phase 2: frozen 48-slot family results and secondary battery (2026-10-15)

**Status: PRELIMINARY.** The data are incomplete: 12 validation recording prefixes,
and animal identity is unverified. The user's 2026-10-13 plan also adds 22 more
crickets that are not yet in this corpus. Confirmatory claims are
limited to the pre-registered family in `docs/realdata-phase2-protocol-20261010.md`
(frozen 8ed5edb; §11.1 amendment be4a5cf, protocol sha 6009c531). Everything in
§5–§6 below is descriptive or exploratory.

## 1. Runs and provenance

- **Arms × seeds:** 4 arms × 3 seeds (42/43/44), with one fixed nested split
  (seed 42).
  - R0: control.
  - R1: residual over persistence, `persistence_skip` 1.0.
  - R2: lagged speed/acceleration input channels 2–3 zeroed.
  - R3: `lambda_jerk` 0.
- **Devices:** all 12 scored runs trained on CUDA. Fresh GPU runs never touch the
  resume seam.
- **Training deviations.** Two of these depart from the amended §11.1 text (amendment
  be4a5cf, as run from the a79898a worktree) and affect only the **R0 arm**.
  - **R0-s43:**
    - The original GPU run was interrupted at epoch 58; its dir is kept as
      evidence.
    - §11.1 states the remedy as "re-run from scratch CPU in a new dir
      (`R0-control-seed43-rerun`)". The executed rerun was fresh in that new dir
      (single segment, epochs 0–300) but ran on **CUDA** (`Device: cuda` in its
      `train.stdout.log`).
    - Its status therefore equals the other ten fresh GPU runs, not the
      CPU-bitwise status §11.1 named. **Deviation from the stated remedy.**
  - **R0-s44:**
    - Resumed with a successor commit (8ed5edb → a79898a) from the clean
      epoch-150 boundary (stored epoch 149; `disclosed_rollback` null). Start and
      resume both ran on **CUDA**.
    - Per §11.1 condition (v) and its GPU clause, this continuation is
      **state-exact, NOT bitwise, and must not be misrepresented as a
      bitwise-continuous run**.
    - §11.1 also states that a GPU resume "must NOT be used to produce a phase-2
      result". Scoring this checkpoint in the frozen family is therefore a
      **protocol deviation**. Every family slot that uses the R0 arm carries a
      **GPU / trajectory-unverified** caveat:
      - R0's own 12 slots;
      - the R1|R0, R2|R0 and R3|R0 contrasts.
  - **R0-s42:** started from the live tree (`D:/Projects/NSMoR`, via `run_arms.sh`),
    not from the frozen worktree. It was not interrupted. Its checkpoint was
    re-evaluated with the frozen code (relative difference about 5e-5), and it
    records repo-relative prior/dataset paths (see §5).
  - **R3-s44:** the first start was killed by a Windows reboot during dataset load
    (0 epochs). That dir was archived, and the run restarted fresh. This is not a
    resume.
  - **Impact on the gate (§3).** The gate verdict depends on h5. No arm reaches h5
    significance vs persistence, and R0 is negative there, so the deviations
    cannot change the NOT-passed verdict.
  - **Impact on other contrasts.** The Holm-adjusted p of every slot depends on
    the whole 48-slot ordering, including R0 slots. No re-scored family without
    R0-s44 exists, and building one would alter the frozen family, so none was
    built.
- **Checkpoint preflight:** an exact-12 audit passed twice, independently
  (`exact12-audit-20261014-v2/`; Codex `exact12-preflight-v1.json`).
  - An earlier attempt (`exact12-audit-20261014/`, kept) reported FAIL only
    because of an audit-script bug: it read the seed from the wrong config key.
  - v2 fixed that key and added nested-split-seed and prior-in-checkpoint checks.
  - The audit checked:
  - positional arm/seed;
  - unique paths and hashes;
  - finite tensors;
  - common dataset `b1bd5578…` and prior `f2585630…`.
- **Scoring:** the unmodified frozen scorer, 8ed5edb
  `scripts/evaluate_realdata_phase2.py`, sha `786f0bc8…a21a9d`, with
  `--exchangeability_asserted` as pre-registered.
  - Ran on CPU: the launcher `run_family48.sh` sets `CUDA_VISIBLE_DEVICES=""`.
    `run.log` itself records no device line.
  - Output: `.scratch/realdata-phase2-20261010/family48-20261014/realdata_phase2_family.json`,
    sha `111e3ae4…`, `MANIFEST.sha256`.
- **Scoring deviations:**
  1. The launch preflight compared an 8-character prefix of the scorer SHA. The
     full SHA recorded at launch (`inputs.sha256`) matches.
  2. `rc.txt` = 127 is **not** the scorer's exit code. The wrapper script was
     edited while bash was still executing it, and bash then ran a fragment after
     python returned. The scorer's final log line ("Wrote … family=48, valid=44,
     unavailable=4") is immediately followed by `return 0` in the frozen source,
     and there is no traceback. The exit is therefore inferred to be 0 but was not
     recorded (`rc-addendum.json`).

## 2. Primary result: skill vs persistence (seed mean)

`skill = 1 − MSE_model / MSE_persistence`; positive means better than
persistence. Holm p is taken from the fixed 48-slot family: 44 valid slots plus 4
unavailable self-comparison slots carried as p = 1.

| Arm | Aligned | Escape | Onset | h5 |
| :-- | :-- | :-- | :-- | :-- |
| R0 | −0.209 (d_z +0.33, Holm .199) | −0.256 (+0.13, .097) | −0.148 (+0.40, .176) | −1.888 (+0.26, **.023**) |
| R1 | **+0.159** (d_z −0.84, Holm **.023**) | +0.163 (−0.67, .227) | **+0.175** (−0.67, **.030**) | +0.027 (−0.03, 1.0) |
| R2 | −8.52 (+0.83, **.023**) | −9.76 (+0.95, **.023**) | −7.36 (+1.09, **.030**) | −4.22 (+0.31, **.023**) |
| R3 | −0.218 (+0.34, .199) | −0.269 (+0.14, .113) | −0.153 (+0.41, .167) | −1.684 (+0.28, **.023**) |

d_z is the paired candidate-minus-persistence MSE effect over trials, so d_z < 0
favors the candidate. Bold marks Holm p ≤ 0.05. A bold entry with positive d_z
means the arm is significantly **worse** than persistence.

Other comparators in the family:
- **R1 vs R0:** significantly better in all four scopes (Holm .023–.035).
- **R2 vs R0:** significantly worse in aligned, escape and onset; h5 is not
  significant.
- **R3 vs R0:** no scope reaches significance (Holm .43–.90). This is not
  evidence of equivalence.
- **vs zero predictor:** R0, R1 and R3 are significantly better in aligned, escape
  and onset. R2 is significantly better in escape and onset but not in aligned
  (Holm .146). No arm is significant at h5 vs zero.

## 3. Phase-3 foundation gate (frozen rule)

Phase-3 protocol §3 requires both of the following. Each may come from any arm,
but within a scope the arm and slot must be the same.
- *onset-beat:* skill > 0, d_z < 0 and Holm p ≤ 0.05 on onset.
- *h5-beat:* the same three conditions on h5.

Result:
- onset-beat is met by R1 (+0.175, d_z −0.67, Holm .030).
- h5-beat is met by no arm. R1 has +0.027 but d_z −0.03 and Holm 1.0; every other
  arm is negative.
- **The gate is NOT passed.**

Per the phase-3 protocol and the 2026-10-14 interpretation boundaries, a failed
gate downgrades mechanism interpretation; it does not ban descriptive runs. The
next round of training still waits for the user's single overall start
authorization.

This gate read-out was computed from the sealed family JSON by a read-only
script, `C:/Users/Wray/.claude/jobs/a4ff95ef/tmp/realdata-phase2-train-20261010/slots_family48.py`.
Two reviewers re-derived it independently from the JSON with their own scripts.
Both got NOT passed, and one recomputed all 44 Holm-adjusted p exactly.

## 4. Interpretation boundaries

- **R1:** it is a residual model over persistence, so part of its advantage over
  persistence comes from its structure. Its improvement is significant in the
  aligned and onset scopes only; escape is positive but not significant, and h5
  is indistinguishable. Selection MSE is not interchangeable with these frozen
  scores.
- **R2:** removing lagged speed and acceleration makes prediction much worse
  *in this setup*. This does not quantify how much "history explains", and it
  does not identify a biological mechanism.
- **R3:** no detected difference from R0. That is not evidence that jerk is
  irrelevant.
- **Inference unit:** the 12 recording prefixes are the clustering unit and are
  assumed exchangeable (`--exchangeability_asserted`). They are not verified
  animals.
- **Recursion:** none of R0–R3 enables a recursion mode (`refinement_mode` and
  `time_consuming_mode` are both off), so phase 2 says nothing about NS-CAMoR or
  adaptive recursion (ADR 0010).
- **h25/h125:** secondary and descriptive only. Seed-mean skill is large-negative
  for R0 and R3 at h25/h125, −0.79/−0.53 for R1, and −1.25/+0.17 for R2. The
  escape-restricted variants differ, e.g. R1 is +0.16/+0.17. None of this enters
  the family.

## 5. Secondary battery (protocol §8/§8.1; descriptive, model-internal)

- **Run:** 12/12 completed with rc 0, using the sealed publication-analysis code
  02b29e6 at default batch 1, on CPU with one thread.
  - Output: `.scratch/realdata-phase2-20261010/secondary-20261015-v2/`,
    MANIFEST sha `3ff52cdd…`.
  - Each summary.json was checked against the exact-12 checkpoint SHAs, arm/seed,
    R2 zero channels and the scorer SHA.
- **Deviation:** the first attempt (`secondary-20261015/`, kept) failed on R0-s42.
  That checkpoint was trained from the live tree and records repo-relative prior
  and dataset paths. Rerunning with the original training cwd fixed it; the code
  still came from the 02b29e6 tree (`module-path-note.txt`).

| Arm | Onset ΔMSE, LIF readout replaced | Onset ΔMSE, GRU readout replaced | Paired LIF−GRU RMS timing median (ms; seeds 42/43/44) |
| :-- | :-- | :-- | :-- |
| R0 | −0.53 / −0.58 / −0.66 | +490 / +457 / +458 | −24 / −278 / −56 |
| R1 | +0.12 / +0.20 / +0.31 | +6.0 / +5.9 / +6.3 | +132 / −628 / −320 |
| R2 | −1.08 / −0.46 / −0.15 | +275 / +273 / +283 | +220 / −128 / NA |
| R3 | −0.59 / −0.65 / −0.65 | +491 / +459 / +464 | −100 / −348 / −152 |

ΔMSE is the prefix-mean cluster statistic, replacement minus natural.

- **Readout:** replacing the GRU readout raises error sharply, while replacing the
  LIF readout leaves it nearly unchanged or slightly lower. The models' output
  depends on the GRU readout. This is model-internal readout dependence only: it
  is not pathway shutdown, neural causality or biological isomorphism.
- **Timing order:** the paired LIF−GRU RMS departure median is LIF-first in all
  three seeds for R0 and R3. It changes sign across seeds for R1 (+132/−628/−320)
  and R2 (+220/−128/NA). There is therefore no ordering that holds across arms.
  - The comparison is RMS-scale dependent and uses common-uncensored pairs only.
  - LIF censoring rates are high: about 0.28–0.64 outside R2.
  - R2's LIF timing is degenerate for seed 44 (no LIF event; censor 1.0) and
    near-degenerate for seed 43 (median 4 ms), but not for seed 42.
- **Prestimulus:** the predicted ≥10 cm/s frame rate (about 0.020) matches the
  observed rate (0.020) for R0, R1 and R3; R2 predicts 0. The strict false-alarm
  trial rate is ≤ 0.0042. Speed bins are proxies only; annotated behavioral states
  are unavailable, so state control is **unavailable**.

## 6. Figure review (the 12 `secondary_battery.png/pdf/svg`)

Reviewed R1-s42 and R2-s44 as a typical case and the most degenerate case. The
titles carry the required caveats: "model-internal exploratory", "recording
prefixes are not animals", "not Kaplan–Meier", "annotated states unavailable".

Fix-forward presentation issues; none affects validity:
1. In the prestimulus panel, a true zero rate drew no bar and looked like a
   missing bar.
2. In the timing panel, the legend could overlap a curve that saturates at 1.0
   (R2-s44).

**Fixed (2026-10-15):** commit 8026bf4 on `feat/secondary-figure-fix-20261015`
(dual ACCEPT, focused non-training tests 51/51) labels every rate bar and moves
the legend below the axes.
- **Re-render:** all 12 runs were re-rendered with that code (batch 1, CPU, one
  thread) into `.scratch/realdata-phase2-20261010/secondary-20261015-v3-figfix/`
  (MANIFEST sha `bcfcbf70…`).
- **Equivalence with v2:** `compare-vs-v2.json` shows every `trial_level.csv` is
  byte-identical to v2 and every summary.json matches v2 apart from provenance.
  The scientific outputs are unchanged.
- **Real-data check of the batching code:** 8026bf4 builds on the batching
  refactor 84baab7, so this also shows the refactored default (batch 1) path
  reproduces the 02b29e6 outputs exactly on real data. Batch > 1 remains
  unchecked on real data.
- **Visual check:** the R2-s44 figure was re-checked by eye. Zero rates now read
  "0", and the legend sits below the timing axes.

## 7. Limitations and next steps

- Preliminary data: 12 prefixes, unverified identity. The 22 additional crickets
  in the user's 2026-10-13 plan go into a new versioned dataset; their timing is
  undecided.
- R0 arm conformance: R0-s44 is a GPU successor-commit resume and R0-s43 is a GPU
  rather than CPU rerun (§1).
  - **User decision (2026-10-15):** keep both checkpoints in the reported family,
    with the §1 caveats; no R0 retraining.
  - Every R0-involving slot and contrast is reported as GPU /
    trajectory-unverified. The R0 contribution is a disclosed deviation from
    §11.1, not a §11.1-conformant phase-2 result.
  - The gate verdict is unaffected (§3).
- Pending:
  - a CPU equivalence check before any production use of `batch_size > 1`
    (`feat/secondary-batching-20261014`, 84baab7).
- No training was run for this report. The next training round, including the
  phase-3 plan (C0–C6, about 300 GPU-hours; C1–C6 alone 270), starts only on the user's single
  overall "start".
