#!/bin/bash
# Process-level resilient training wrapper.
#
# If the training process dies for any reason (pod reclaim, OOM kill, host
# reboot), relaunch it from the latest checkpoint. The training script
# refuses to resume when the experiment identity changed (see
# --resume-from / manifest.json), so a blind relaunch can never graft one
# experiment's optimizer state onto another.
#
# Environment:
#   ROOT         project directory on the training host (required)
#   OUT          training output directory, holds checkpoints/ (required)
#   TRAIN_CMD    full training command; the literal token {RESUME} is
#                replaced with "--resume-from <latest checkpoint>" or ""
#   MAX_ATTEMPTS relaunch budget (default 40)
#   SLEEP_BETWEEN seconds between attempts (default 15)
#   LOG          wrapper log path (default $ROOT/train_resilient.log)
set -u

: "${ROOT:?set ROOT to the project directory}"
: "${OUT:?set OUT to the training output directory}"
: "${TRAIN_CMD:?set TRAIN_CMD (use {RESUME} where --resume-from belongs)}"
LOG="${LOG:-$ROOT/train_resilient.log}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-15}"

cd "$ROOT" || exit 1
attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    latest=$(ls -d "$OUT"/checkpoints/step_* 2>/dev/null | sort | tail -1)
    if [ -n "$latest" ]; then
        resume_args="--resume-from $latest"
    else
        resume_args=""
    fi
    echo "=== $(date) attempt $attempt resume='${resume_args:-none}' ===" >> "$LOG"
    bash -c "${TRAIN_CMD//\{RESUME\}/$resume_args}" >> "$ROOT/train_full.log" 2>&1
    code=$?
    echo "=== $(date) attempt $attempt exited code=$code ===" >> "$LOG"
    if [ "$code" -eq 0 ]; then
        echo "TRAINING_COMPLETE" >> "$LOG"
        break
    fi
    if tail -20 "$ROOT/train_full.log" | grep -q "nothing to resume"; then
        echo "TRAINING_ALREADY_COMPLETE" >> "$LOG"
        break
    fi
    if [ ! -d "$OUT/checkpoints" ]; then
        echo "NO_CHECKPOINT_YET_FAIL" >> "$LOG"
        break
    fi
    sleep "$SLEEP_BETWEEN"
done
echo "RESILIENT_WRAPPER_DONE attempts=$attempt" >> "$LOG"
