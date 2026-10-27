#!/bin/bash
# Wait for the resilient training wrapper to finish, then run the
# post-training generations sequentially on one GPU.
#
# Same-device rule: every arm is generated on the same card, one after
# another, so an A/B comparison stays attributable.
#
# Environment:
#   ROOT      project directory on the training host (required)
#   GEN_CMDS  commands to run after training, one per line (required)
#   GPU       CUDA_VISIBLE_DEVICES value for the generations (default 0)
#   LOG       log path (default $ROOT/pipeline.log)
set -e

: "${ROOT:?set ROOT to the project directory}"
: "${GEN_CMDS:?set GEN_CMDS (one command per line)}"
LOG="${LOG:-$ROOT/pipeline.log}"
GPU="${GPU:-0}"

cd "$ROOT" || exit 1
echo "=== $(date) pipeline start, waiting for the training wrapper ===" >> "$LOG"
while pgrep -f "train_resilient.sh" > /dev/null; do sleep 30; done
echo "=== $(date) training wrapper finished ===" >> "$LOG"
tail -5 "$ROOT/train_resilient.log" >> "$LOG" 2>/dev/null || true
tail -3 "$ROOT/train_full.log" >> "$LOG" 2>/dev/null || true

if ! grep -q "TRAINING_COMPLETE\|TRAINING_ALREADY_COMPLETE" "$ROOT/train_resilient.log"; then
    echo "ERROR: training did not complete successfully, refusing to generate" >> "$LOG"
    exit 1
fi
echo "=== $(date) training verified successful, proceeding to generation ===" >> "$LOG"

while IFS= read -r command; do
    [ -n "$command" ] || continue
    echo "=== $(date) running: $command ===" >> "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" bash -c "$command" >> "$LOG" 2>&1
done <<< "$GEN_CMDS"

echo "PIPELINE_DONE" >> "$LOG"
