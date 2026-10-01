# Delivery work log — 2026-09-30

## Status

Training is complete and operationally accepted; the seven analyses have run.
The formal run was segmented across three processes: human epochs 1–66
(`train-causal-formal-persistent-50r7by4t`, interrupted), 67–98
(`train-causal-formal-resumed-eqnr7_4c`, low-memory terminated), and 99–150
(`train-causal-formal-compact-qfadtol8`, exit 0). Partial epochs 67 and 99 are
excluded and no deterministic uninterrupted-equivalence claim is made; epochs
1–98 preserve six-decimal log precision and epoch 99 onward uses serialized full
precision. The accepted best/final checkpoint is
`4fc2b90663124869049acf1c1a8b9a50250d449e9b2d053472ab01d760ae6f02`, stored
epoch 149 = completed epoch 150. All seven analyses ran as seven sequential
single-process stages with exit 0 and a stable checkpoint hash; the current
receipt is
`/mnt/d/Projects/NSMoR/.scratch/seven-analysis-sorted-20261001-r6/status.json`.
Scientific acceptance remains `pending`. Operational acceptance does not
establish biological validation, physical synchronization, or animal
independence. Historical sections below are preserved; the 2026-10-01 terminal
update supersedes stale running/pending claims. The earlier
`.../seven-final-analyses-single-process-20261001/status.json` receipt is
retained as superseded history only.

## Compact acceptance and planned recovery — 2026-10-01

- Both reviewers accepted compact retention across all 11 production consumers
  and restoration of the default (`wqdbkllp4`). This changes in-memory source
  provenance representation only; full strict record validation is unchanged.
  No weakening, relabeling, model/configuration or inference-input change, or
  artifact-byte change was accepted.
- Actual single-load full-corpus validation (`wp22aklaz`) accepted exact dataset,
  nested-prior, and resume pins, source rows, aliases, and raw timing tokens.
  Evidence:
  `/mnt/d/Projects/NSMoR/.scratch/compact-corpus-acceptance-20261001-5e9c43/acceptance.json`.
  Peak RSS 6,672,196 KiB; steady RSS 3,458,960 KiB; loader 28.20 s; full validation
  52.96 s. Peak reduction versus restored-list 14,523,752 KiB is 54.06%.
  The actual full regression in that directory's `pytest.log` reports
  **1607 passed, 1 skipped, 11 warnings in 165.60s, exit 0**.
- Prepared, not executed, standalone supervisor:
  `/mnt/d/Projects/NSMoR/.scratch/formal_compact_supervisor_20261001_43f72c9d.py`.
  Parent task will launch only after its checks. The supervisor checks Linux/
  torch Python, the exact epoch-98 resume SHA, and actual `/proc` trainer
  arguments before creating a unique `train-causal-formal-compact-*` directory.
  It uses only config/dataset/nested/resume/output CLI arguments, inherits the
  formal GPU environment unchanged, and records logs, lineage, status, elapsed
  time, and child peak RSS there. No historical artifacts/statuses are overwritten.
- Lineage remains original epochs 1–66, resumed epochs 67–98, partial epoch 99
  excluded; stored checkpoint epoch 97 means 98 completed human epochs. Budget
  remains 150 and deterministic uninterrupted equivalence is **not** claimed.
  No new training or GPU job was launched during preparation. (Superseded by the
  terminal update below: the supervisor ran, formal recovery and checkpoint
  acceptance completed, and all seven analyses ran; scientific/release approval
  alone remains pending.)

## Compact recovery completed — 2026-10-01

- The compact supervisor completed. Unique parent task `bvwpwv7g1` (PID 68339)
  wrote to
  `/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/train-causal-formal-compact-qfadtol8`
  and exited 0 (`state=trainer_exited`, `returncode=0`).
- It resumed from the exact epoch-98 checkpoint
  (`8d937ae2f81530d58d0dee5a459b6d96e9071221f4805fe1257ef7e1fb0bd22f`) bound to the
  accepted dataset pin `b1bd5578...` and nested-prior pin `f2585630...` over the
  staged raw source. Science configuration was unchanged: budget 150, early-stop
  patience 20, dt 4 ms, hidden dimension 64, batch size 128, max sequence length
  2,400, pre-anchor 1,200, learning rate 0.0005, seed 42, no target norm clipping,
  clip 0.
- The run completed epochs 99–150 to the full 150-epoch budget with stopping reason
  `budget` and zero non-improving epochs. `best_model.pth` is
  `4fc2b90663124869049acf1c1a8b9a50250d449e9b2d053472ab01d760ae6f02` and
  `final_model.pth` is `2679223a7281c4717ce9f2e8cb1ab1ea9bbb2a5131d2eb3b54004da6f59e2183`.
  Acceptance record: `.../formal-terminal-acceptance-46yjj_ne/acceptance.json`
  (`status=operational_formal_accepted`).
- Final metrics (best weights): train loss 7.325285180409749, val loss =
  best_val_loss 11.127355694770813, RMSE 3.2286346653606017 cm/s, MAE
  0.5784122347831726, R² 0.600713312625885, escape-band (10 cm/s) RMSE
  19.824984206281147 cm/s over 25,075 escape frames, resting RMSE 0.9700182642151225
  cm/s. The escape-band RMSE is the declared limitation. Scientific acceptance
  remains `pending`; no clock-scientific-resolved or biological-validity claim is
  made.

## Verified work

- Diagnosed workflow `wf_47746847-108`: 240 tool calls, 183 failed edits,
  including 176 identical failed replacements. Read-output separator tabs
  were incorrectly included in source-match strings. The worker executed no
  pytest command and its claimed test additions were absent.
- Inspected the actual producer: tensor-backed X/Y serialization is inside
  `prepare_dataset`, using a shallow dictionary copy and default ZIP format.
  The shared restricted loader restores NumPy sequences. Its immutable-byte
  fingerprint binding and restricted deserialization remain intact.
- Coordinator ran:

  ```sh
  python -m pytest tests/test_pipeline.py tests/test_pipeline_resampling.py tests/test_prepare_data_producer_identity.py tests/test_nested_prior_tool.py -q --tb=short
  ```

  Result: **172 passed, 2 warnings in 66.32s**. The warnings originate from
  deliberate float32-overflow rejection cases. This predates the additional
  storage-regression work below and must not be reused as its acceptance.
- Located the existing staged input at
  `/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/adapter/staged`.
  Availability was checked, not scientific acceptance. Historical partial
  dataset files must not be loaded or reused.

## Active bounded workflow

`wf_e9e10cc0-4e7`: finish tensor-save assertions and NumPy/tensor restricted-loader
round-trip tests; run the real focused regression; measure save/load memory
in separate bounded synthetic processes if feasible. No raw-data modification,
model/loss modification, runtime configuration changes, commit, or push.
Coordinator verification is required after return; workflow completion alone
is not acceptance. Repeated identical tool failures must stop the attempt.

## Execution update — fresh ETL started

- Stopped `wf_e9e10cc0-4e7` after inspecting 324 calls: 209 errors,
  including 202 repeated invalid `ListAgents` calls. No storage-regression
  or memory-measurement result was accepted.
- `wf_a88becc6-ca0` did not run ETL. Its four commands used Linux mount
  paths directly in Git Bash, rather than entering WSL. Two `ls` calls
  returned exit 2. Its reported exit 137 was unsupported and is rejected;
  it is not evidence of an ETL OOM.
- Coordinator verified inside WSL that the producer exists and staging
  contains 128 session directories. To avoid another identical workflow
  retry, directly launched the already specified ETL command as a tracked
  background process. No source changes or runtime reconfiguration.
- First launcher failed before starting ETL because `/usr/bin/time` is
  absent. Retained its empty `run.log`; switched measurement to Python
  stdlib `time.monotonic` and `resource.getrusage`, with exclusive
  `run-2.log` creation and no artifact overwrite.
- Actual ETL started at 2026-09-30 11:50:50 (log clock) and is pairing the
  staged kinematics/events files. Task: `bk6e62jq1`. Output directory:
  `/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/etl-final`.
  This is a running process, not an accepted dataset. Final return code,
  elapsed time and child peak RSS will be recorded in `status.json`.

## Full ETL failure — kernel OOM confirmed

- Background task `bk6e62jq1` ended with wrapper exit **143**, not 137.
  No `status.json` was written and no matching ETL process remained in `/proc`
  at inspection. The produced `nsmor_dataset.pt` is **zero bytes**, invalid,
  and must not be loaded or used for training.
- Actual ETL log reached serialization at 11:58:32 after extracting 2,304
  sequences and 58,156,483 model frames. Conditions: 1,188 multisensory,
  720 visual-only, 396 wind-only. OOF/ensemble mean total-variation distance
  was 0.1370, recorded as descriptive prior shift, not an acceptance guarantee.
- Kernel log at 11:59:24 reports a global OOM and killed Python PID 2160:
  `anon-rss:15120732kB`, `total-vm:71413564kB`. This establishes an OOM during
  the run; the wrapper's SIGTERM status is a separate observation.
- Tensorizing X/Y alone did not make this full run complete. Source inspection
  identifies a remaining large serialization input: `io.py:421-424` creates
  four per-source-row Python lists (indices, raw host tokens, raw hardware
  tokens, time-source tags), retained in clock provenance. Their contribution
  is a hypothesis until separately measured; do not discard provenance or
  convert raw timing tokens to lossy numeric values to save memory.
- A bounded fresh-process measurement was attempted but WSL failed to start
  with `Wsl/Service/E_UNEXPECTED` (tool exit 127). No benchmark result exists.
  No WSL services, resource limits, runtime settings, or raw files were changed.

## Measured recovery and next storage fix

- User restarted WSL and authorized routine shutdown/start recovery for this
  kind of failure without repeated confirmation. WSL Python execution now works.
- Fresh-process synthetic measurements (500,000 source rows, float32 tensor
  X/Y, KiB peak RSS): numeric-only save 557,460 → 557,840; adding Python-list
  provenance 647,388 → 732,524. These are bounded measurements, not a proven
  full-corpus peak attribution.
- A proposed Unicode-array optimization was rejected experimentally:
  list version 647,172 → 732,364 versus compact Unicode arrays
  615,036 → 826,628. Lower resident representation did not mean lower
  serialization peak. Stopped `wf_05045f6e-d87` before any source edits.
- Per-trial lossless UTF-8 JSON + zlib measurement, 20 chunks of 25,000 rows:
  lists 646,660 → 731,872; compressed 573,364 → 591,504; compressed payload
  3,429,573 bytes. Exact JSON round trip asserted for every chunk. These
  synthetic figures do not establish full-corpus acceptance.
- Active implementation workflow `wf_76426a30-cdb`: compact provenance as
  each trial is loaded; preserve the public extraction contract; restore exact
  lists and shared aliases in the restricted dataset loader; add real tests.
  Existing raw observations, metadata and all trial identities must survive.
  No full ETL rerun until coordinator reviews the actual change and tests.

## Coordinator gate and fresh full run

- `wf_76426a30-cdb` made no successful implementation edits; its repeated
  file-list result and failed tab-prefixed edits were rejected. Coordinator
  produced only the stuck storage candidate as a fallback. User then clarified
  that implementation belongs in Workflow workers, with coordinator fallback
  limited to genuinely stuck local changes. Subsequent code work was delegated.
- Earlier review/finalization reports were not supported by executed tests or
  successful edits and were not accepted as independent scientific review.
- The final EOF worker in `wf_fa34c0bd-a28` actually performed the exact Edit
  successfully. Coordinator then verified `git diff --check` and ran the six
  focused files including compact storage and clock mapping:
  **214 passed, 2 deliberate overflow warnings in 160.33s; exit 0**.
  Frozen model and loss are absent from the tracked source diff.
- Active `wf_07173b8c-c67` delegates two independent activities: fresh full ETL
  to `etl-compact-v1` with real restricted-loader acceptance, and read-only
  nested-prior/training/seven-analysis scientific-contract review. No source
  changes, commit, push, or historical hash audit are authorized in this run.
  Full-corpus memory suitability and dataset acceptance remain unverified.

## Current evidence update — corrected dataset and training

- Corrected dataset acceptance passed at
  `/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/etl-causal-final-zp12m0fa/nsmor_dataset.pt`:
  2,716,755,245 bytes, SHA-256
  `b1bd5578025fb5eaa6f4eb3e355c097b6576d44dbedaa001027f1de5447e06b2`, ETL
  exit 0 in 226.025 s with peak RSS 8,068,764 KiB. Restricted verification
  accepted finite float32
  `X(T,8)`/`Y(T)`, 2,432 eligible identities, 2,304 labeled sequences and 128
  unavailable anchors, 46,873,366 source rows, 58,156,483 model frames, and
  physical conditions 1,188 multisensory / 720 visual-only / 396 wind-only.
  `pipeline_semantics_version` is 2.2; animal identity remains unverified.
  The acceptance record is
  `dataset-acceptance-8d5ac304df8f4f7998f939e7f45f5199.json` in that directory.
- Nested artifact accepted at
  `/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/nested-causal-complete-dku91_m0/nested_split_seed42.pt`:
  323,253 bytes, SHA-256
  `f25856304a5857695945bf5083ca54745f47fe5dffe915111f6ba28d524ddf94`, exit 0
  in 59.796 s; source-bound to the exact dataset, with 1,872 train / 432
  validation trials, 52 / 12 disjoint train/validation recording prefixes,
  five inner folds, and seed 42. Requested validation fraction 0.2 realized
  as 0.1875 of trials.
- Timing fixes passed 302 focused tests with two deliberate overflow warnings;
  coordinator resampling verification passed 66 tests. Independent double-blind
  timing reviewers accepted. Jacobian/attractor/autoregressive fixes have reported
  real results of 170 prediction-unit/attractor tests and 114 nested tests;
  independent scientific reviews of those fixes remain pending.
- One-epoch smoke training in
  `/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/train-causal-smoke-t_8k9raw`
  passed with status 0 in 325.514 s: train loss
  18.07954692840576 and validation loss 25.608824729919434. Checkpoint lineage,
  finiteness, and final/best checkpoint checks passed. Batch 128, dt 4 ms,
  hidden dimension 64, sequence length 2,400, learning rate 0.0005, and seed 42
  were unchanged; only smoke epochs were explicitly set to 1. This is not
  formal training.
- The earlier `train-causal-formal-lacwbu9l` launch died when its worker finished;
  coordinator confirmed its PIDs absent. Its status is stale and no checkpoint
  from it was accepted.
- Formal replacement training was previously live in
  `/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/train-causal-formal-persistent-50r7by4t`
  (PID 63063 confirmed live at 22:27:38 +09:00). As documented below, this job
  subsequently terminated abruptly without final history or completed status.

The causal 4 ms grid contains previous-hold estimates, not new source observations.
Source-authoritative anchors select the first supported frame at/after the event;
unsupported or merged pulses fail closed without trial deletion. Raw canonical host
clocks are not corrected again; the exploratory 200 ms grid is not a physical
bound, and recording prefixes do not verify animal identity. Synthetic
autoregression uses a declared uniform or supplied prior rather than an empirical
nested prior, and fatigue is uncalibrated. Whole-corpus analysis is descriptive,
not independent holdout evaluation.

## Recovery update and current evidence — 2026-10-01

- Old formal training job (PID 63063) is absent; its `running` status was stale.
  The process was interrupted without writing a final checkpoint or training
  history, but completed through human epoch 66 with an immutable checkpoint:
  SHA-256 `c505741af29a3586732dc35e5530fddbf511b2bcd4fe2e4c594c45f4166d6c48`.
- Resumed formal training was launched under unique parent Bash task `bxp4p27ns`
  (PID 17145) writing to
  `/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/train-causal-formal-resumed-eqnr7_4c`.
  The run resumed from the epoch 66 checkpoint under the identical configuration.
  It was later terminated by the harness under a low-memory condition; see the
  recovery record below. The run is **not completed or accepted** and carries no
  claim of exact deterministic uninterrupted equivalence.
- Science configuration is strictly preserved: dt 4 ms, hidden dimension 64,
  batch size 128, max sequence length 2,400, learning rate 0.0005, epoch budget
  150, early-stop patience 20, seed 42, single-phase mode, no target norm clipping.
- Root core architecture and loss modules (`checkpoint.py`, `model_utils.py`,
  `model_nsmor_core.py`, `loss.py`) are protected and HEAD-identical.
- Accepted clock repairs and preflight review (`wf9ni5nw6`) both evaluated to
  ACCEPT. Accepted Jacobian known reference and sensitivity fixes, modern clock
  completeness, cadence binding, and preflight guards are verified on exact
  shared production seams with root protection restored.
- Fresh artifact validation (`acceptance-kla41ty5.json`, exit 0 in 48.496 s,
  peak RSS 14,523,752 KiB, no regeneration) confirmed:
  - Exact DATA: `/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/etl-causal-final-zp12m0fa/nsmor_dataset.pt`
    (SHA-256 `b1bd5578025fb5eaa6f4eb3e355c097b6576d44dbedaa001027f1de5447e06b2`)
  - Exact NEST: `/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/nested-causal-complete-dku91_m0/nested_split_seed42.pt`
    (SHA-256 `f25856304a5857695945bf5083ca54745f47fe5dffe915111f6ba28d524ddf94`)
  - RAW source: `/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/adapter/staged`
  - Validated metrics: strict dt 4 ms, finite 2,304 sequences, 58,156,483 frames,
    2,432 eligible trials, 128 unlabeled anchors, 46,873,366 source rows, 2,304×4
    priors, 1,872/432 train/validation split, 52/12 recording-prefix groups,
    `animal_identity_status="unverified"`.
- Full WSL regression (`bzol4d105`) executed cleanly:
  **1547 passed, 1 skipped, 11 warnings in 184.71s (exit 0)**.
  The single skip is `tests/test_downstream_anchor_alignment.py:281` because small
  real data is absent in test environment; regression count is 1 skipped, not
  all passed.
- All seven downstream analyses (dynamics, lesion, Jacobian, integration,
  psychophysics, gating, autoregressive simulation) subsequently ran against the
  accepted final checkpoint; see the terminal update below.
- Hardware clock certainty and animal holdout are not exaggerated: median 4.997 ms
  is an observed canonical interval diagnostic, not proof of raw unbatched
  timestamps; the 4 ms grid represents causal model estimates.

## Recovery record — 2026-10-01

- The resumed formal run under parent task `bxp4p27ns` (PID 17145) ended when the
  harness stopped it under a low-memory condition. Coordinator verified
  `/proc/17145` is absent and WSL `MemAvailable` was ~15 GiB after termination.
  No exit code 137, kernel OOM log, or specific killed PID is claimed; the
  observed event is a harness low-memory stop.
- The run's `status.json` still reports `state: "running"`; this is stale. No
  `final_model.pth`, `train.log`, or trainer success marker exists. Trainer
  absence and the absence of an error log are operational observations, not
  scientific completion.
- Checkpoint verified by read-only tooling at
  `/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/train-causal-formal-resumed-eqnr7_4c/best_model.pth`:
  459,549 bytes, SHA-256
  `8d937ae2f81530d58d0dee5a459b6d96e9071221f4805fe1257ef7e1fb0bd22f`.
  Stored `epoch` field is 0-based 97 == human epoch 98; `train_loss`
  7.748280874888102, `val_loss` = `best_val_loss` 11.777471899986267, optimizer
  step 1470, scheduler last epoch 98. Tensor finiteness sweep: 36
  `model_state_dict` + 51 `optimizer_state_dict` tensors, 0 non-finite; strict
  `load_state_dict` shows no missing or unexpected keys. Dataset pin
  `b1bd5578...`, nested-prior pin `f2585630...`, and config lineage all match;
  provenance is `validation_scope="nested_outer_validation"`,
  `animal_identity_status="unverified"`.
- Joined epoch log: original run completed epochs 1–66, resumed run completed
  epochs 67–98, with no gaps or duplicates across the two logs; the partial
  epoch 99 is excluded. This checkpoint is valid for recovery, not formal
  completion, and does not establish uninterrupted equivalence with a
  hypothetical single continuous run.
- Memory context: the default loader path carries full provenance and has
  measured ~14.5 GiB RSS on the full corpus. A separate workflow
  (`w62rt95mm`) is implementing an optional compact validated consumer route
  with the default restored `True`. No memory fix has been accepted and no
  runtime global memory setting has been changed.
- All seven downstream analyses subsequently ran against the accepted final
  checkpoint; see the terminal update below.

## Terminal update — 2026-10-01 (current)

Formal training completed to the 150-epoch budget and was operationally accepted
at
`/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c/formal-terminal-acceptance-46yjj_ne/acceptance.json`
(`status=operational_formal_accepted`, `scope=operational_formal_acceptance_only`,
exit 0, validator 11.61 s). Joined history
(`formal-terminal-acceptance-46yjj_ne/joined_history.json`) is `complete` with 150
epochs and `partial_epochs_excluded=[67, 99]`; its precision note states epochs
1–98 preserve six-decimal log precision while epoch 99 onward uses `train.log`
serialized full precision. Stopping reason is `budget` with zero non-improving
epochs. No deterministic uninterrupted-equivalence claim is made.

- Accepted best/final checkpoint
  `.../train-causal-formal-compact-qfadtol8/best_model.pth`
  SHA-256 `4fc2b90663124869049acf1c1a8b9a50250d449e9b2d053472ab01d760ae6f02`
  (stored epoch 149 = completed epoch 150; 33,995 parameters; strict-loadable;
  model/optimizer/scheduler tensors all finite). `final_model.pth` is
  `2679223a7281c4717ce9f2e8cb1ab1ea9bbb2a5131d2eb3b54004da6f59e2183`.
- Best/final train loss 7.325285180409749; val loss = best_val_loss
  11.127355694770813.
- Formal metrics (best weights): MSE 10.424081802368164, RMSE 3.2286346653606017
  cm/s, MAE 0.5784122347831726, R² 0.600713312625885, escape-band (10 cm/s) RMSE
  19.824984206281147 cm/s over 25,075 escape frames (2.42% of frames), resting RMSE
  0.9700182642151225 cm/s. The escape-band RMSE is the declared limitation:
  escape frames are rare and their error is an order of magnitude above resting
  error. Scope `nested_outer_validation`, `animal_identity_status=unverified`,
  `clock_scientific_acceptance=experimental_unresolved`, `uninterrupted_determinism=false`.
- Compact-provenance memory evidence
  (`/mnt/d/Projects/NSMoR/.scratch/compact-corpus-acceptance-20261001-5e9c43/acceptance.json`,
  exit 0, 52.96 s): single dataset load and single nested load, peak `ru_maxrss`
  **6,672,196 KiB**, steady `VmRSS` **3,458,960 KiB**, loader 28.20 s; **54.06%**
  lower peak than the restored-list 14,523,752 KiB; max simplex error 1.51e-07;
  all 2,432 records strict-validated once. Dataset/nested/resume byte pins
  unchanged.
- Two accepted loader/PCA repairs: the dynamics PCA is now an exact whole-corpus
  mean-centered fit that streams per-trajectory blocks and accumulates only the
  centered scatter, so all 5,529,448 frames keep equal weight rather than a raw
  uncentered `XᵀX`. The bounded *fit workspace* is O(block_rows·D + D²) — one
  streamed block plus the `(D, D)` scatter; the retained 3-D scores, the per-state
  int64 labels, and the resident corpus bundle are outside that bound, so total
  analysis memory is not O(D²). The analysis loader is single-process
  (`num_workers=0`) because the whole corpus is already resident and workers only
  replicated it.
- Seven analyses ran as seven sequential single-process stages; current receipt
  `/mnt/d/Projects/NSMoR/.scratch/seven-analysis-sorted-20261001-r6/status.json`
  (`state=complete, all_seven_executed=true, all_seven_successful=true`,
  `scientific_acceptance=pending`, `ended_unix_seconds=1790846716.01`). Per-stage
  exit/runtime/peak-child-RSS (KiB): dynamics 0 / 208.04 s / 8,550,968; lesion 0 /
  484.34 s / 6,916,040; jacobian 0 / 177.72 s / 7,058,808; integration 0 /
  179.30 s / 6,949,272; psychophysics 0 / 26.71 s / 6,919,880; gating 0 /
  198.48 s / 7,232,776; autoregressive 0 / 36.44 s / 1,544,500. The checkpoint
  SHA-256 was identical before and after every stage. All per-stage required
  outputs are present and non-zero. The integration stage source SHA-256 is
  `fed23f12…` and the Jacobian stage source SHA-256 is `8a79475b…`, both matching
  this checkout. Stage findings: dynamics centered-PCA 45.02/19.18/5.69% over
  2,304 trajectories / 5,529,448 states; lesion `descriptive_only` with
  `p_status=unavailable_unverified_animal_identity` (52 train / 12 val prefixes;
  6,912 rows); jacobian frozen-input control failed honestly so
  `jacobian_spectrum.json` is published `status=withheld` with empty
  `spectral_statistics` for all epochs and no stability interpretation;
  integration `in_sample_descriptive_only` with sorted wind axis `[-373, -308,
  -261, -225]` and visual-only as a disconnected labelled baseline; psychophysics
  `not_applicable` (`n_ttc0=0`, `n_trials_matched=0`, `n_candidates=252`,
  `holdout_eligible=false`); gating ARI 4-way 0.198 / NMI 0.227 / Cohen's d 1.19;
  autoregressive `scope=synthetic`, not DATA/NEST-bound, replay not guaranteed,
  fatigue uncalibrated.
- **Superseded (history, not current evidence).** The earlier
  `.../seven-final-analyses-single-process-20261001/status.json` run is retained
  as history only and is explicitly superseded by the r6 receipt above: its
  integration predates the visual-only exclusion and sorted wind-axis fixes, and
  its Jacobian log is inconsistent with the withheld-spectrum contract. Its
  per-stage numbers (dynamics 206.54 s, lesion 485.25 s, jacobian 181.46 s,
  integration 145.00 s, psychophysics 26.41 s, gating 198.32 s, autoregressive
  35.71 s) must not be cited as current evidence.
- Full regression on this checkout (HEAD `f8b6acf7...`, branch
  `delivery/source17-v6`), rerun after the sorting-regression strengthening with
  `python -m pytest tests/ -q -ra -p no:cacheprovider`: **1656 passed, 1
  skipped, 11 warnings in 169.20s, exit 0** (raw log
  `/mnt/d/Projects/NSMoR/.scratch/final-delivery-regression-20261001.log`). The
  single skip is the legitimate missing real-subset fixture
  (`tests/test_downstream_anchor_alignment.py::test_all_six_loaders_on_real_small_dataset`,
  `nsmor_subset_small.pt` absent), reported as skipped, not passed. The earlier
  1643- and 1607-passed figures are superseded.

## Remaining delivery gates

1. Coordinator release review and approval (all seven analyses and final
   checkpoint acceptance are complete).
2. Only after approval may an authorized release commit and push occur. No
   commit or push has occurred in this ticket.

Final source/artifact review of this checkout (branch `delivery/source17-v6`,
HEAD `f8b6acf7...`) returned ACCEPT with zero blockers, majors, or minors. The
ACCEPT is scoped to source and artifact verification (r6 citations, seven stage
hashes, withheld Jacobian contract, honest psychophysics/autoregressive
outcomes, real regression); it does not resolve the clock estimates, verify
animal identity, or establish an untouched holdout. The runner receipt's
`scientific_acceptance=pending` is historical execution state and is not
rewritten to invent acceptance.

## Historical remaining delivery gates

1. Accept the actual storage diff and new regression output; report bounded
   memory measurements without extrapolating them to a full-corpus guarantee.
2. Run fresh full ETL, restricted loading, finite-value and identity accounting
   checks. Historical accounting reference: 2,432 extracted trials, 2,304 labeled
   sequences, 128 unavailable stimulus anchors. Explain any discrepancy; do not
   delete trials to force agreement.
3. Preserve source-timed observations and provenance, causal 4 ms model-grid
   estimates, active-pulse fail-closed checks, and unavailable-anchor accounting.
   Do not claim model-grid held values are new independent observations.
4. Generate nested priors for the exact accepted dataset, retaining grouped
   leakage controls; recording-prefix groups do not prove animal independence.
5. Train using the agreed configuration; verify checkpoint and history.
   A short smoke run is not full training, and early stopping is not proof of
   convergence.
6. Run dynamics, lesion, Jacobian, integration, psychophysics, gating, and
   autoregressive analyses against the same dataset/checkpoint/configuration.
   Preserve valid unavailable/not-applicable outcomes without inventing effects.
7. Finish full regression and documentation, then coordinator release review.
   Only accepted source changes belong in the commit; exclude scratch data and
   unrelated root-checkout changes. No release has occurred yet.
