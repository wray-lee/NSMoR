# Epoch-boundary recovery (opt-in)

This document is the canonical description of the epoch-boundary recovery
seam added to `scripts/train.py`. It describes the capability for **new and
future** runs and the authorized **fresh-segment** continuation of an
interrupted run. It does not alter, re-cadence, or reinterpret any existing
formal run's evidence: the original interrupted runs keep their sealed
interval-10 cadence and their incomplete outcomes, and are never relabeled as
complete.

## What it does

A SIGKILL/OOM restart of a training run must CONTINUE the state that shapes
its own trajectory, not silently reset it. The resume seam persists and
restores, in every checkpoint type (`best_model.pth`, `epoch_N.pth`,
`final_model.pth`):

- the early-stopping counter `epochs_without_improvement`;
- the full diagnostic loss histories (`train_loss`, `val_loss`);
- the NumPy legacy-MT19937 RNG state.

These ride the existing provenance pop-then-patch seam in
`_atomic_save_checkpoint`, so the frozen `nsmor/checkpoint.py` storage
contract is untouched.

## Supported scope: cadence one with zero-worker loading

Reliable continuation is the **cadence-one regime** (`checkpoint_interval: 1`)
and is scoped to **`num_workers=0`**. Both conditions are needed: cadence one
makes every checkpoint an exact epoch boundary of a run whose trajectory is
claimed reproducible, and zero workers is what makes that trajectory
reconstructable after a restart. With `num_workers>0` (and especially
`persistent_workers=true`) the loader iterator owns multiprocessing worker
state — the RNG stream and the live iterator's consumption position — that is
neither captured in the checkpoint nor reconstructable from it. An
uninterrupted persistent iterator and a newly constructed resumed iterator
therefore consume *different* global RNG histories, so a resume silently
diverges even when no worker-side randomness is present (independent
same-budget model/backprop probe: workers0 exact equality, persistent workers1
divergence).

Because silently reconfiguring the loader would change the user's scientific
controls and claiming equivalence would be false, a run at
`checkpoint_interval: 1` — **fresh or resumed** — whose ACTUAL resolved
train/val loaders have `num_workers>0` is **refused before any epoch** with a
clear error. A run at a non-reliable cadence (e.g. the prior
`checkpoint_interval: 10`) is unchanged: it makes no exact-continuation claim,
so no worker count is constrained. `config/epoch_recovery.yaml` therefore sets
`checkpoint_interval: 1` and `num_workers: 0`; its `persistent_workers: false`
is inactive at zero workers and is documentation only.

## State semantics (recovery_state_version = 2)

- The diagnostic histories are persisted **in full and in order**, one entry
  per executed epoch. A resumed run appends to the same epoch axis an
  uninterrupted run would have produced.
- A **non-finite scalar** (NaN/Inf from an unstable epoch) is a genuine
  recorded result and is written **as-is**. It is never dropped or coerced to
  zero, so it never collapses the positions of later epochs. This is a
  *diagnostic* value; the model/gradient numerical guards are unchanged.
- `history_start_epoch` records the true 0-based epoch of history entry 0, so
  a resumed run re-derives the loss-curve x-axis at the correct offset instead
  of redrawing legacy history from epoch 1. The reader requires
  `history_start_epoch + len(train_loss) == resume_epoch` and requires the two
  series to have equal length (they share one epoch axis). A legacy checkpoint
  has no recorded start epoch; the reader sets it to the resume epoch so
  prehistory is explicitly treated as unknown rather than fabricated.
- `epochs_without_improvement` is an **independent control value**: it counts
  only FINITE non-improving validation epochs (the loop increments it behind
  `math.isfinite(val_loss)`). Its consistency bound is the number of epochs
  that *could* have counted: recorded finite epochs **plus declared-unknown
  (`None`) slots**, whose finiteness is not observable from the history. A
  counter above that bound is inconsistent and fails closed. Two mistakes are
  avoided: an observed-finite upper bound would wrongly reject a legitimate
  counter that spans declared gaps (the gaps may each have been a finite
  non-improving epoch), and a raw-completeness bound would accept a counter
  that exceeds the epochs actually executed. Recorded NaN/Inf epochs are known
  non-counting and are excluded from the bound.
- **A run without a validation loader still occupies the epoch axis.** Every
  EXECUTED epoch appends one entry to each series. With no validation loader
  the validation entry is `None` — an unobserved diagnostic, never a
  fabricated finite value and never a claimed NaN/Inf. Patience does not
  advance on such an epoch (`val_loss` stays `inf`, so the `math.isfinite`
  gate is not met), and no best checkpoint is selected. This keeps
  `len(train_loss) == len(val_loss)` on the shared epoch axis, so a
  current-schema checkpoint written by a no-validation run is accepted by its
  own reader; the loss curve plots only the observed validation points.
- The **current schema requires** a valid `numpy_rng_state`. A missing or
  malformed current-version state (non-`MT19937` generator, fractional or
  boolean `pos`/`has_gauss`, string/bool cached value, wrong shape/dtype,
  non-CPU device) **fails closed before any epoch** — it is damaged required
  state, not a legacy gap, and is never coerced with `int()`/`float()`.
- A current-schema checkpoint also requires its **complete canonical state**:
  `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict` and
  `rng_state` must all be present, non-`None` and non-empty. The canonical
  loader tolerates an absent optimizer or scheduler (it restores only what is
  present), so a current-schema payload that omitted one would resume with a
  FRESH optimizer (zero Adam moments) or a restarted LR schedule while still
  claiming an exact continuation. Missing, explicitly `None`, or malformed
  required state therefore **fails closed in the resume preflight**, before any
  update or terminal save. This constrains the CURRENT schema only: a legacy /
  version-mismatched parent keeps its limited warn-and-reset fallback.
- Non-emptiness is **not** sufficient for `scheduler_state_dict`.
  `LRScheduler.load_state_dict` silently MERGES the mapping it is given — it
  never resets to defaults and never validates the key set — so a truncated or
  arbitrary mapping (e.g. `{"last_epoch": 1}`, or `{"malformed": 1}`) loads
  without error while retaining the freshly-built scheduler's `base_lrs` /
  `T_max`, i.e. it resumes on the WRONG LR trajectory while claiming exact
  continuation. The preflight therefore requires the complete
  `CosineAnnealingLR` structure the trainer actually builds: the integer
  `T_max` / `last_epoch`, the numeric `eta_min`, and the non-empty numeric
  `base_lrs` / `_last_lr` lists of equal length. This is scoped to the current
  schema and to `CosineAnnealingLR` — not a generic scheduler framework — and a
  legacy parent without the block keeps its warn-and-restart fallback.
- The legacy MT19937 Gaussian cache is a **one-slot** cache: `has_gauss` is 1
  exactly when an ODD number of `standard_normal` draws have been taken since
  the last reset, and 0 after an even number. A restart that restores the
  stream therefore alternates the flag across epochs rather than pinning it,
  and the restored `cached_gaussian` value is what determines the next Gaussian
  draw — which is why the recovery comparison checks the full decoded state
  (`keys`/`pos`/`has_gauss`/`cached_gaussian`) and both the uniform and the
  Gaussian continuation, not just the uniform stream.
- **An already-terminal restored checkpoint executes no further epoch.** A
  periodic checkpoint is published at a completed-epoch boundary *before* the
  early-stopping decision for that epoch is evaluated, so a process that dies
  in that window leaves a checkpoint whose restored counter has already reached
  the patience horizon. The uninterrupted run would have stopped there, so the
  resume applies the **same stop predicate** (one shared function, used by both
  the in-loop check and the resume) *before* executing any epoch. It finalizes
  with the parent's recorded last-executed epoch, counter, history and RNG
  state — never an invented epoch, and never a diagnostic value the run did not
  measure. A later improving validation cannot re-open an exhausted horizon.
  The predicate preserves the two phase rules exactly as the loop does: a
  phase-1 epoch is exempt while a later phase 2 remains (`num_epochs >
  phase1_epochs`), and a phase-boundary-crossing resume resets the counter, so
  neither case is terminal on resume.
- **The terminal diagnostic is taken from the parent's own record, preferring a
  scalar over a gap.** A zero-update finalization reports the last executed
  epoch's `train_loss`/`val_loss` from EITHER source the parent wrote: its
  top-level scalar (`save_checkpoint`'s `train_loss`/`val_loss`) or the tail of
  the restored history series. A **recorded scalar always wins** — a legitimate
  `None` history GAP (an executed-but-unobserved epoch) must never overwrite a
  scalar the parent measured, and the reverse fills a scalar into a gap. When
  BOTH sources hold a scalar they describe the same epoch and must agree; a
  disagreement is a structurally inconsistent parent and fails closed, as does
  a non-scalar value. `NaN`/`Inf` pass through verbatim (a non-finite epoch is a
  genuine result, not a gap) and two `NaN`s count as equal. Only when NEITHER
  source has a value is the diagnostic truly unobserved, and the final
  checkpoint then carries the explicit absence (no `train_loss`/`val_loss`) —
  never a fabricated `0` or finite placeholder.
- **The terminal TRAIN diagnostic honours the legacy `loss` alias.** A parent
  written by an older/legacy writer may carry only the step-level `loss` scalar
  and no `train_loss`. A **MISSING** `train_loss` key is therefore distinct from
  an explicit `train_loss=None`: the former falls back to the parent's `loss`
  field (when present), the latter is a recorded unknown and is never
  substituted. The resolved value is fed through the same selection rule above,
  so a `loss` scalar that disagrees with a scalar history tail still fails
  closed and a `loss`-backed scalar still beats a `None` history gap. This is a
  field-preservation rule only: it never invents a diagnostic the parent did
  not record, and the `val_loss` series has no alias.
- **A completed parent at the ORIGINAL total budget finalizes without a
  further epoch.** `start_epoch == num_epochs` means the run already completed
  its own budget — `range(num_epochs, num_epochs)` is empty — so the resume has
  no updates left and finalizes at the parent's saved executed epoch with its
  complete continuation state: model, optimizer state and groups, scheduler,
  Torch RNG, the full decoded MT19937 state, history, origin and patience. No
  epoch is invented and none executes. This is terminal on the **budget
  alone**: it holds whether early stopping is enabled, disabled, exhausted or
  not. It is deliberately narrow. Only a **current-schema** parent is eligible
  — a legacy or version-mismatched parent has an unknown counter and history
  and still fails closed. The recovery state is validated *before*
  finalization, so a corrupt counter/history/axis or a damaged MT19937 payload
  still fails closed without writing a checkpoint. A parent that **overshot**
  the budget (`start_epoch > num_epochs`) is still refused, because the active
  budget would truncate a longer parent run. In two-phase mode the
  phase-1→2 boundary fires at the FIRST phase-2 epoch, which a completed run
  never executes, so `phase1_epochs == num_epochs` is NOT treated as a
  crossing: the run stays in phase 1 and its phase-1 best checkpoint survives.
  A budget-terminal resume inside an established phase 2 keeps the phase-2
  posture (2-group optimizer, phase-2 scheduler state).
- A **legacy** checkpoint (no `recovery_state_version`) or a checkpoint from a
  different schema version loads with an explicit unknown-continuity warning
  and restarts counters. A structurally malformed current-version payload
  (bad counter, wrong keys, length mismatch, non-scalar entry, unequal series)
  fails closed.

## Invocation

Fresh run into a new output directory (this profile enables cadence 1 and
zero workers):

```
python scripts/train.py --config config/epoch_recovery.yaml \
    --output_dir runs/<new_run_name> \
    --nested_prior_artifact <nested.pt>
```

Resume from an exact complete checkpoint (`epoch_N.pth`, not a `*.tmp` or a
partially-written file). Each resume writes to a **fresh, independent output
directory** so the parent segment's `best_model.pth`, `final_model.pth` and
receipts are never overwritten. `--resume` points at the parent's exact old
epoch checkpoint:

```
python scripts/train.py --config <parent_resolved_config.yaml> \
    --output_dir runs/<parent>_resume_<n> \
    --resume runs/<parent>/epoch_N.pth \
    --checkpoint_interval 1 \
    --nested_prior_artifact <nested.pt>
```

`--config` must be the **parent's resolved run config** (the same YAML the
parent used), so the total epoch budget, model/loss/data/prior/split/seed and
all scientific controls are preserved. Do not swap in
`config/epoch_recovery.yaml` to resume an existing run: that template's
`num_epochs` (150) would silently replace a longer parent budget (e.g. the
k1.0 arm's 300). Instead, from the copied parent config make exactly the two
reliable-mode edits — set `checkpoint_interval=1` (or pass
`--checkpoint_interval 1`) and `num_workers=0` — and leave
`persistent_workers`/`prefetch_factor` as the parent had them, since they are
inactive at zero workers. `config/epoch_recovery.yaml` is for a NEW run.

## Epoch-budget semantics

`num_epochs` is the **total** budget for the run, not the number of ADDITIONAL
epochs: the loop runs `range(start_epoch, num_epochs)`. A run interrupted at
epoch 80 of 300 and resumed with the parent's config trains toward the
original total budget — it neither doubles nor reduces it. Early stopping uses
the restored `epochs_without_improvement` against the parent's
`early_stopping_patience`, so the patience horizon is not restarted; because
early stopping is unchanged, a resumed run that has already exhausted patience
can stop **before** reaching the total budget — and if the parent's published
checkpoint was already terminal, it stops **without executing any epoch at
all**. Reaching the total budget is the other terminal case: a parent completed
at exactly `num_epochs` has no remaining updates and finalizes at its saved
executed epoch (see "An already-terminal restored checkpoint" above). The total
budget is the cap, not a guaranteed endpoint — and reaching it is completion,
not an error.

That completion is only recognized under the parent's **own** original total
budget. The budget is immutable across a continuation, so a parent whose
checkpoint happens to sit at the active budget's final epoch is finalizable
only if the parent's recorded `config.training.num_epochs` is a known strict
positive integer **equal** to the active `num_epochs`. A parent run under a
LONGER budget (e.g. 6) resumed under a shorter active budget (3) can land on
epoch index 2 and satisfy `start_epoch == num_epochs`; finalizing there would
stamp the segment with a budget the parent never ran under and truncate a
longer authorized run, so it fails closed with no `final_model.pth`. Unknown,
missing, malformed (bool or non-integer) or unequal parent budgets likewise
fail closed before any finalization.

## Scope and limits

- **Epoch-boundary only.** There is no mid-epoch / batch-level resume; the
  optimizer and scheduler are restored at the epoch boundary.
- **Cadence one with zero workers only.** See "Supported scope" above; a
  cadence-one run (fresh or resumed) with other worker configurations is
  refused before any epoch. A non-reliable cadence such as the prior
  interval 10 is unaffected.
- **Legacy continuity is partial.** A checkpoint written before this schema
  (`recovery_state_version` absent) has no recorded patience counter or
  epoch-positioned history. On resume the counter restarts from zero and the
  prehistory is marked unknown via `history_start_epoch = resume_epoch`; the
  loss curve shows only the resumed segment, never fabricated old points. A
  selected best checkpoint may imply a zero counter only if selection and
  checkpoint identity are verified, not assumed. A legacy resume's
  worker-count change (e.g. original workers>0 → new workers0) is **not**
  established as trajectory-equivalent to the parent's original execution;
  only the new zero-worker segment's own epochs are exactly reproducible.
- **Exactness evidence is CPU, single-thread, zero-worker.** The
  trajectory-equality evidence (model, optimizer, scheduler, Torch RNG, NumPy
  RNG — full MT19937 state plus both the uniform and the Gaussian
  continuation — patience, history) was established on a deterministic
  single-thread CPU zero-worker fixture. The evidence run pins the measured
  configuration explicitly: `CUDA_VISIBLE_DEVICES=''` (a visible CUDA device
  makes the fixture fail loudly rather than silently measuring a different
  claim), `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)`.
  The capture records what was ACTUALLY resolved, not what was requested:
  `patches/evidence-r9/env.txt` (torch/NumPy versions, the device the
  interpreter resolves, pinned thread counts),
  `patches/evidence-r9/imports.txt` (each imported module's resolved file
  path — the candidate source, not another checkout),
  `patches/evidence-r9/dependencies.txt` (installed distributions), and the
  measured source hashes in `patches/evidence-r9/{pre,post}.sha256`.
  The trajectory-equality tests themselves are re-run in
  `patches/evidence-r9/focused-run.log` against THIS root's bytes, so the
  exactness claim is bound to the evidence files named here and to no other
  snapshot: a result captured against an earlier root's bytes is **historical**
  and must not be cited as verification of these bytes. (An r6-era capture of
  this evidence ran with a CUDA device visible; that older result is not the
  CPU claim either.) Equality on CUDA or multi-thread execution is **not**
  established — kernel nondeterminism and atomics can differ across a restart
  — so no exactness claim is made there.
- **Opt-in.** `config/epoch_recovery.yaml` differs from `config/default.yaml`
  in the two ACTIVE controls `training.checkpoint_interval: 1` and
  `training.num_workers: 0`; its `training.persistent_workers: false` is
  inactive at zero workers (documentation only).
- **Does not recadence or reinterpret** the current fixed formal runs (their
  evidence is sealed at interval 10 and must not be mutated). A continuation
  of an interrupted run is a NEW segment that records its own provenance; it
  does not turn the original incomplete execution into a completed one.
