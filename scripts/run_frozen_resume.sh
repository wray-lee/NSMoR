#!/usr/bin/env bash
# Resume-capable phase-2 launcher (protocol amendment S6).
#
# Same preflight as run_frozen.sh (cd into the frozen worktree; HEAD == frozen
# commit and the tracked tree is clean; python resolves nsmor / scripts.train
# from the worktree), but a NON-EMPTY output dir is no longer refused outright:
# it RESUMES only when a recoverable checkpoint exists AND its provenance
# matches (same frozen commit, same config sha, a NON-EMPTY resolved_modules
# list entirely inside the worktree); otherwise it still REFUSES.  The resume
# source is the newest LAST-epoch checkpoint (final_model.pth / epoch_*.pth);
# best_model.pth is NEVER used (the best epoch can be far older than the last
# executed epoch) and a dir holding only best_model.pth is refused.
#
# On CPU with a fixed seed the resumed arm is BITWISE-equivalent to the
# uninterrupted run: train.py re-establishes the persistent loader iterator
# around a save/restore of the global RNG on resume, so the resumed process
# consumes the global stream exactly as the uninterrupted run did at that epoch
# boundary (protocol §11.1).  This holds ONLY under §11.1's enforced
# precondition: the dataset __getitem__ and collate consume no worker-side RNG
# (train.py refuses a resume whose dataset would use the legacy random-crop
# path, any __getitem__ that advances the global torch/NumPy stream, or any
# collate_fn that advances either stream).  On GPU
# the resume is state-exact but not bitwise (no determinism flags), same as two
# uninterrupted GPU runs.
#
# Usage:
#   run_frozen_resume.sh <frozen_commit> <worktree> <arm>...
#   run_frozen_resume.sh pause <worktree> <arm>...     # safe-pause helper
#
# `pause` drops an empty STOP sentinel in each arm's output dir.  The training
# loop checks for it ONLY after that epoch's best/periodic checkpoint is fully
# written (temp + fsync + atomic rename) and then exits cleanly, so a pause
# never interrupts a save.  Resume the arm with the normal (non-pause) form;
# the training layer itself consumes the sentinel on any resume (this launcher
# also removes it defensively), so a direct `train.py --resume` cannot re-pause.
#
# Logs: a fresh segment writes $OUT/train.stdout.log; every resume writes a NEW
# $OUT/train.segment<N>.log.  A prior segment's train.stdout.log is never
# truncated (protocol §11.1 condition 4).
#
# seed 42 (R0-control-seed42) was started from the LIVE tree (D:/Projects/NSMoR)
# by run_arms.sh, NOT the frozen worktree, so its provenance head differs from
# the frozen commit.  The provenance check below therefore REFUSES to resume it
# across code versions.  If seed 42 is interrupted it is a DEVIATION: rerun it
# from the frozen worktree, or disclose it as interrupted in the results.
set -u

JOBDIR=$(dirname "$(readlink -f "$0")")
# Status log defaults next to this script; set STATUS_LOG to keep it in a job dir.
LOG=${STATUS_LOG:-$JOBDIR/status.log}
SENTINEL=STOP

if [ "${1:-}" = "pause" ]; then
  shift
  WT=$1; shift
  for ARM in "$@"; do
    OUT=$WT/.scratch/realdata-phase2-20261010/$ARM
    mkdir -p "$OUT"
    : > "$OUT/$SENTINEL"
    echo "$(date -Is) PAUSE requested for $ARM (STOP sentinel at $OUT/$SENTINEL)" >> "$LOG"
  done
  exit 0
fi

FROZEN=${1:?usage: run_frozen_resume.sh <frozen_commit> <worktree> <arm>...}
WT=${2:?usage: run_frozen_resume.sh <frozen_commit> <worktree> <arm>...}
shift 2
cd "$WT"   # run INSIDE the frozen worktree so cwd/sys.path cannot reach the live repo
export PYTHONPATH=$WT

# Inputs default to the frozen phase-2 corpus (identical to run_frozen.sh).
# They are overridable ONLY for CPU smoke-testing the resume decision on a
# throwaway worktree; the preflight (HEAD/clean/modules) is never overridden.
CORPUS=${CORPUS:-/mnt/d/Projects/NSMoR/.scratch/clock-corpus-r4/exploratory-chain-tiZddw2c}
DATASET=${DATASET:-$CORPUS/etl-causal-final-zp12m0fa/nsmor_dataset.pt}
PRIOR=${PRIOR:-$CORPUS/nested-causal-complete-dku91_m0/nested_split_seed42.pt}
ARMS_DIR=${ARMS_DIR:-config/realdata-phase2-20261010}

# Resume decision + resume-log writer live in one CPU-only helper so the
# decision logic is unit-testable without running the launcher.
HELPER=$JOBDIR/resume_decision.py

for ARM in "$@"; do
  OUT=.scratch/realdata-phase2-20261010/$ARM
  CFG=$WT/$ARMS_DIR/$ARM.yaml
  HEAD=$(git -C "$WT" rev-parse HEAD)
  DIRTY=$(git -C "$WT" status --porcelain --untracked-files=no)
  if [ "$HEAD" != "$FROZEN" ] || [ -n "$DIRTY" ]; then
    echo "$(date -Is) $ARM REFUSED: worktree head=$HEAD dirty=[$DIRTY]" >> "$LOG"; exit 1
  fi
  RESOLVED=$(python -c "import nsmor, nsmor.model_nsmor_core as m, nsmor.loss as l, scripts.train as t; print(m.__file__, l.__file__, t.__file__)")
  for P in $RESOLVED; do
    case "$P" in "$WT"/*) ;; *) echo "$(date -Is) $ARM REFUSED: module resolved outside worktree: $P" >> "$LOG"; exit 1;; esac
  done

  NUM_EPOCHS=$(python -c "import yaml,sys; print(int(yaml.safe_load(open('$CFG'))['training']['num_epochs']))")

  RESUME_CKPT=""
  if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    DECISION=$(python "$HELPER" decide "$OUT" "$WT" "$FROZEN" "$CFG" "$NUM_EPOCHS")
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "$(date -Is) $ARM REFUSED: $DECISION" >> "$LOG"; continue
    fi
    RESUME_CKPT=$DECISION
    python "$HELPER" record "$OUT" "$RESUME_CKPT" "$FROZEN" >> "$LOG"
    # Consume the pause sentinel now that we are resuming from its checkpoint,
    # so the resumed run does not immediately re-pause.
    rm -f "$OUT/$SENTINEL"
  else
    mkdir -p "$OUT"
    {
      echo "{"
      echo "  \"arm\": \"$ARM\","
      echo "  \"started\": \"$(date -Is)\","
      echo "  \"code_worktree\": \"$WT\","
      echo "  \"head\": \"$HEAD\","
      echo "  \"worktree_tracked_dirty\": \"$DIRTY\","
      echo "  \"resolved_modules\": \"$RESOLVED\","
      echo "  \"resume_log\": [],"
      echo "  \"sha256\": {"
      for F in "$CFG" "$DATASET" "$PRIOR" "$WT/scripts/train.py" "$WT/nsmor/model_nsmor_core.py" "$WT/nsmor/loss.py" "$WT/docs/realdata-phase2-protocol-20261010.md"; do
        echo "    \"$F\": \"$(sha256sum "$F" | cut -d' ' -f1)\","
      done
      echo "    \"_\": null"
      echo "  }"
      echo "}"
    } > "$OUT/provenance.json"
  fi

  if [ -n "$RESUME_CKPT" ]; then
    # NEVER truncate a prior segment's log.  The fresh segment owns
    # train.stdout.log; every resume writes a NEW indexed segment log, so the
    # interrupted segment's train.stdout.log is preserved verbatim (protocol
    # §11.1 condition 4).  A truncating `>` to the same path would erase the
    # evidence this amendment promises to retain.
    SEG=$(( $(ls -1 "$OUT"/train.segment*.log 2>/dev/null | wc -l) + 1 ))
    RUNLOG=$OUT/train.segment${SEG}.log
    echo "$(date -Is) $ARM START (resume from $RESUME_CKPT, frozen $FROZEN, log $RUNLOG)" >> "$LOG"
    python "$WT/scripts/train.py" --config "$CFG" --dataset "$DATASET" \
      --nested_prior_artifact "$PRIOR" --resume "$RESUME_CKPT" > "$RUNLOG" 2>&1
  else
    RUNLOG=$OUT/train.stdout.log
    echo "$(date -Is) $ARM START (fresh, frozen $FROZEN)" >> "$LOG"
    python "$WT/scripts/train.py" --config "$CFG" --dataset "$DATASET" \
      --nested_prior_artifact "$PRIOR" > "$RUNLOG" 2>&1
  fi
  rc=$?
  if [ $rc -eq 17 ]; then
    # train.py's distinct safe-pause exit code (PAUSE_EXIT_CODE): the run
    # stopped at an epoch boundary AFTER its checkpoint was written, so it is
    # neither a completion nor a failure.  Resume with the same command.
    echo "$(date -Is) $ARM PAUSED rc=17 (safe pause at epoch boundary; resume to continue) (log $RUNLOG)" >> "$LOG"
  else
    echo "$(date -Is) $ARM EXIT rc=$rc (log $RUNLOG)" >> "$LOG"
  fi
done
echo "$(date -Is) ALL DONE" >> "$LOG"
