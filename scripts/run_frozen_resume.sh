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
# executed epoch) and a dir holding only best_model.pth is refused.  DISCLOSED
# ROLLBACK (§11.1, amendment 2026-10-11): when the source's stored epoch is
# OLDER than the last epoch the run actually executed (read from the run's own
# progress logs / segment records / checkpoints), the resume is ALLOWED — with
# checkpoint_interval 10 a crash between periodics always loses up to 9 epochs,
# so refusing would make resume unusable — but the re-trained range
# [source+1, last_executed] is recorded (resume_log + status.log) and printed,
# and train.py MOVES the abandoned-branch artifacts to abandoned_branch_<k>/
# (never overwrites).  The MOVE lives in train.py's resume path ONLY (chosen over
# a launcher-side move): train.py is the single writer of the run dir, so putting
# the move there covers a bare `train.py --resume` too, and the launcher's
# `record` subcommand merely RECORDS the disclosed range in the resume_log.  No
# silent rollback; no override flag.
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
# Environment:
#   ORIGIN_COMMIT=<C>  successor-commit resume opt-in (§11.1 amendment
#                      2026-10-11): accept a provenance head == C only when git
#                      confirms C is an ancestor of the frozen commit C' (via
#                      git.exe on the Windows path form).  Without it a head
#                      != C' is refused as a cross-code-version resume.
#   OUT_ROOT=<abs dir> absolute output root (default: this worktree's
#                      .scratch/realdata-phase2-20261010).  Set it to the run's
#                      own dir root so the launcher, running inside C', appends
#                      the new segment to the run started under C.
#   DRY_RUN=1          run the decider with --dry-run and report the decision;
#                      writes NOTHING to the run dir and launches no training.
#                      The guarantee is precisely "the DECIDER writes nothing"
#                      (proven by the same-commit and successor dry-run tests);
#                      the launcher's OUT_ROOT/--output_dir and sentinel-consume
#                      logic are not exercised end-to-end under DRY_RUN=1, so it
#                      is NOT claimed that the whole launcher path writes
#                      nothing.  For the successor case OUT_ROOT points at the
#                      origin worktree's existing run dir, so this is limited.
#
# The arm's RECORDED start device (from train.stdout.log "Device: ...") is
# passed to train.py as --expect-device on a resume (§11.1 device clause), so a
# CPU arm cannot silently resume onto a CUDA host and a CUDA arm is forced to
# run as CUDA; the device is named in the status.log START line.
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

# The worktrees' .git files point at Windows paths (gitdir: C:/disk/...), so the
# WSL git cannot resolve them; run git.exe on the wslpath -w form and strip CR
# (mirrors run_frozen2.sh's git.exe + tr -d '\r').  The pipeline's exit status is
# tr's (0), so callers that need git's own status run git.exe directly.
gitw() { local wt="$1"; shift; git.exe -C "$(wslpath -w "$wt")" "$@" 2>/dev/null | tr -d '\r'; }

# Output root: ABSOLUTE so the launcher, running inside the C' frozen worktree,
# can resume a run dir that lives under a DIFFERENT (origin) worktree's
# .scratch.  When NOT overridden it defaults to this worktree's own scratch and
# train.py keeps its own (relative) output_dir, so the default path stays
# byte-identical.
OUT_ROOT_SET=${OUT_ROOT:+1}
OUT_ROOT=${OUT_ROOT:-$WT/.scratch/realdata-phase2-20261010}
# Successor-commit resume opt-in (§11.1): the run's own starting commit C.
# Empty (default) keeps the strict same-commit rule (a head != frozen is
# refused).  When set, the decider accepts a provenance head == C only if git
# confirms C is an ancestor of the frozen commit C'.
ORIGIN_COMMIT=${ORIGIN_COMMIT:-}
# DRY_RUN=1 runs the decider (--dry-run) and reports the decision without
# recording or launching training; it writes NOTHING to the run dir.
DRY_RUN=${DRY_RUN:-0}

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
  OUT=$OUT_ROOT/$ARM
  CFG=$WT/$ARMS_DIR/$ARM.yaml
  HEAD=$(gitw "$WT" rev-parse HEAD)
  DIRTY=$(gitw "$WT" status --porcelain --untracked-files=no)
  if [ "$HEAD" != "$FROZEN" ] || [ -n "$DIRTY" ]; then
    echo "$(date -Is) $ARM REFUSED: worktree head=$HEAD dirty=[$DIRTY]" >> "$LOG"; exit 1
  fi
  RESOLVED=$(python -c "import nsmor, nsmor.model_nsmor_core as m, nsmor.loss as l, scripts.train as t; print(m.__file__, l.__file__, t.__file__)")
  for P in $RESOLVED; do
    case "$P" in "$WT"/*) ;; *) echo "$(date -Is) $ARM REFUSED: module resolved outside worktree: $P" >> "$LOG"; exit 1;; esac
  done

  NUM_EPOCHS=$(python -c "import yaml,sys; print(int(yaml.safe_load(open('$CFG'))['training']['num_epochs']))")

  # §11.1 device clause: establish the arm's RECORDED start device from the
  # first segment's stdout log ("Device: cuda" / "Device: cpu") and pass it to
  # train.py as a resume gate, so a CPU arm cannot silently resume onto a CUDA
  # host (state-exact-only, must not produce a phase-2 result) and a CUDA arm
  # is forced to run as CUDA.  Empty (undeterminable) leaves a bare resume.
  EXPECT_DEVICE=$(python - "$OUT" <<'PY'
import pathlib, sys
out = pathlib.Path(sys.argv[1])
dev = ""
p = out / "train.stdout.log"
if p.exists():
    for line in p.read_text(errors="ignore").splitlines():
        if "Device:" in line:
            dev = "cuda" if "cuda" in line else "cpu"
            break
print(dev)
PY
)

  RESUME_CKPT=""
  if [ "$DRY_RUN" = "1" ] && { [ ! -d "$OUT" ] || [ -z "$(ls -A "$OUT" 2>/dev/null)" ]; }; then
    # A dry run must write NOTHING, so never create an empty run dir for it.
    echo "$(date -Is) $ARM DRY-RUN: no non-empty run dir at $OUT (nothing to resume; nothing written)" >> "$LOG"
    continue
  fi
  if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    DECIDE_ARGS=("$OUT" "$WT" "$FROZEN" "$CFG" "$NUM_EPOCHS")
    if [ -n "$ORIGIN_COMMIT" ]; then DECIDE_ARGS+=("$ORIGIN_COMMIT"); fi
    if [ "$DRY_RUN" = "1" ]; then DECIDE_ARGS+=(--dry-run); fi
    # A plain assignment from a command substitution takes the SUBSTITUTION's
    # exit status (POSIX; verified on bash — a decider crash yields rc != 0 and
    # is refused below), so a decider crash is fail-closed, not read as rc=0.
    DECISION=$(python "$HELPER" decide "${DECIDE_ARGS[@]}")
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "$(date -Is) $ARM REFUSED: $DECISION" >> "$LOG"; continue
    fi
    if [ "$DRY_RUN" = "1" ]; then
      echo "$(date -Is) $ARM DRY-RUN OK: would resume from $DECISION (origin=${ORIGIN_COMMIT:-none}, frozen=$FROZEN)" >> "$LOG"
      continue
    fi
    RESUME_CKPT=$DECISION
    if [ -n "$ORIGIN_COMMIT" ]; then
      python "$HELPER" record "$OUT" "$RESUME_CKPT" "$FROZEN" "$ORIGIN_COMMIT" >> "$LOG"
    else
      python "$HELPER" record "$OUT" "$RESUME_CKPT" "$FROZEN" >> "$LOG"
    fi
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

  # --output_dir pins writes to the run's OWN dir.  When OUT_ROOT points at the
  # origin worktree's .scratch (a successor-commit resume) this appends the new
  # segment/checkpoints/logs to the existing run instead of creating a fresh dir
  # under the C' worktree; when OUT_ROOT is unset the config's own output_dir is
  # used unchanged, so the default path is byte-identical.
  OUTDIR_ARGS=()
  if [ -n "$OUT_ROOT_SET" ]; then OUTDIR_ARGS=(--output_dir "$OUT"); fi

  if [ -n "$RESUME_CKPT" ]; then
    # NEVER truncate a prior segment's log.  The fresh segment owns
    # train.stdout.log; every resume writes a NEW indexed segment log, so the
    # interrupted segment's train.stdout.log is preserved verbatim (protocol
    # §11.1 condition 4).  A truncating `>` to the same path would erase the
    # evidence this amendment promises to retain.
    SEG=$(( $(ls -1 "$OUT"/train.segment*.log 2>/dev/null | wc -l) + 1 ))
    RUNLOG=$OUT/train.segment${SEG}.log
    # §11.1 device clause: a recorded start device pins the resume's device.
    DEVICE_ARGS=()
    if [ -n "$EXPECT_DEVICE" ]; then DEVICE_ARGS=(--expect-device "$EXPECT_DEVICE"); fi
    echo "$(date -Is) $ARM START (resume from $RESUME_CKPT, frozen $FROZEN, origin ${ORIGIN_COMMIT:-none}, device ${EXPECT_DEVICE:-unpinned}, log $RUNLOG)" >> "$LOG"
    python "$WT/scripts/train.py" --config "$CFG" --dataset "$DATASET" \
      --nested_prior_artifact "$PRIOR" "${OUTDIR_ARGS[@]}" \
      "${DEVICE_ARGS[@]}" \
      --resume "$RESUME_CKPT" > "$RUNLOG" 2>&1
  else
    RUNLOG=$OUT/train.stdout.log
    echo "$(date -Is) $ARM START (fresh, frozen $FROZEN)" >> "$LOG"
    python "$WT/scripts/train.py" --config "$CFG" --dataset "$DATASET" \
      --nested_prior_artifact "$PRIOR" "${OUTDIR_ARGS[@]}" > "$RUNLOG" 2>&1
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
