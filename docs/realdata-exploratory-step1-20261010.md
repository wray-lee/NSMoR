# Real-data exploratory step 1 (2026-10-10) — EXPLORATORY, NOT PRE-REGISTERED

Companion to `docs/realdata-preliminary-conclusions-20261009.md`. Nothing here
enters that document's declared-family results; it informs the next phase's
pre-registration. Existing checkpoints only (baseline, A1, A2; one seed each),
no retraining; nested validation split (432 trials, 12 recording prefixes; 0
wind_only). Effect sizes are trial-level d_z; prefix counts are descriptive;
no p-values. Window references are the dataset crop anchor, not a verified
stimulus onset.

Outputs: `C:/Users/Wray/.claude/jobs/a4ff95ef/tmp/realdata-exploratory-step1-20261010/`
(`step1_summary.json`, `step1_provenance.json`, `task4_windowed_lesion.json`,
`task5_latency.json`, `parity.json`).

## Decision gate

**No arm beats persistence in any onset window or at any rollout horizon
h ≥ 5.** The only non-negative h ≥ 5 cell is the baseline's stimulus-onset
rollout at h = 25 (skill +0.16, d_z −0.07, 6/12 prefixes), which A1/A2 do not
share and which is gone by h = 125. Per the agreed step-3 gate, the shortfall
is not only an evaluation artefact: inputs, loss weights and data must be
audited before the next training phase.

## Prediction

Skill = 1 − MSE_model / MSE_persistence (negative = worse than persistence).

| Measure | baseline | A1 | A2 |
| :-- | :-- | :-- | :-- |
| Anchor windows pre / onset / early / sustained¹ | −2.52 / −1.17 / −3.35 / −0.42 | −0.35 / −0.45 / −0.12 / −0.29 | −0.15 / −0.21 / −0.005 / −0.26 |
| Response-onset windows² | −2.71 / −3.34 / −2.89 / −0.41 | −0.29 / −0.18 / −0.15 / −0.35 | −0.18 / −0.04 / −0.02 / −0.32 |
| Multi-step h = 1 / 5 / 25 / 125³ | −3.91 / −0.81 / −6.89 / −9.84 | −0.43 / −2.07 / −6.07 / −103.2 | −0.18 / −0.52 / −5.55 / −21.4 |
| Stimulus-onset rollout h = 1 / 5 / 25 / 125 | −0.87 / −0.57 / +0.16 / −0.46 | −0.23 / −3.16 / −7.75 / −34.9 | +0.23 / −1.87 / −9.11 / −39.5 |
| Residual R² vs persistence (overall) | −2.44 | −0.24 | −0.08 |

¹ Frames from the anchor at 4 ms/frame: pre [−500, 0), onset [0, 25),
early [25, 100), sustained [100, 1000). Prefixes favouring the model ≤ 4/12
in every cell.
² Windows from the first frame with observed |v| ≥ 10 cm/s (317/432 trials).
³ Own predictions fed back into the lagged-velocity/acceleration inputs; 8
start frames per trial; persistence held at y[start−1].

Event level (362 paired trials): observed peak speed 65.8 cm/s; models
under-predict it (25.3 / 51.7 / 52.0 cm/s; d_z −1.15 / −0.37 / −0.37). Peak
timing has small mean bias and large scatter. **Heading is not identifiable:
the target is a non-negative speed**, so a direction metric is degenerate.

## Mechanism

**Windowed lesion (fills the H2 gap).** Replacing the GRU readout (g_lif = 1)
is the dominant harm in both windows: onset ΔMSE +384 / +596 / +579
(d_z 1.16 / 1.11 / 1.23, 11/11 prefixes) and sustained +422 / +650 / +667
(d_z 0.76 / 0.72 / 0.82, 10/10). Replacing the LIF readout is neutral to
slightly helpful in both windows. **H2 (LIF loss hits onset, GRU loss hits the
sustained phase) is not supported.** The sustained window scores only 175/432
trials (responders, mostly multisensory).

**Pathway response latency** (first departure > 3 SD from a 250 ms
pre-reference baseline; median ms, LIF / GRU): baseline 330 / 68, A1 292 / 310,
A2 272 / 368. Paired LIF − GRU: baseline +136 ms (GRU first, 9/10 prefixes),
A1 −12 ms (tie), A2 −128 ms (LIF first, 0/12). Censoring is 29–81% and differs
by arm; the cross-correlation lag sits at the search-window edge and is not
used. **Latency does not give a stable fast/slow ordering**: it is
arm-dependent, so neither the leak-constant ordering nor the architectural
premise is confirmed. A1/A2 change activation and refinement together, so the
flip cannot be attributed to either.

## Implications for the next phase

- The target is non-negative **speed** at 4 ms resolution, with observed lagged
  velocity and acceleration among the inputs. Persistence is close to optimal
  for one-step speed; an audit of the target definition (speed vs signed
  velocity / heading), the lagged inputs, and the loss terms that pull outputs
  off the MSE optimum (regularisation, energy, sparsity, jerk) comes first.
- The GRU readout carries nearly all predictive load in every window; the LIF
  readout is close to dispensable. A symmetric-pathway control is needed before
  any division-of-labour claim.
- Peak speed is under-predicted by every arm, consistent with smoothing loss
  terms.
