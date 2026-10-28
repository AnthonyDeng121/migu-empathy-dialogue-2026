#!/bin/bash
# Cron self-healing supervisor for unattended training runs.
#
# Install with: crontab -e
#   */10 * * * * /path/to/supervisor.sh >> /path/to/supervisor.log 2>&1
#
# Every 10 minutes it makes sure the training wrapper and the generation
# pipeline are alive (relaunching them when they are not), and stops once
# the expected arm files exist and parse as complete JSONL. A host reboot
# is covered too: the cron daemon restarts this script on its own.
#
# Environment:
#   ROOT       project directory on the training host (required)
#   ARMS       space-separated arm output files, relative to ROOT (required)
#   ARM_LINES  expected line count per arm file (required)
#   DONE_FILES optional space-separated post-processing outputs (metric
#              reports, ...) that must exist before the run counts as
#              finished; without it a pipeline that dies between the last
#              generation and its metrics is declared done and never
#              relaunched
set -u

: "${ROOT:?set ROOT to the project directory}"
: "${ARMS:?set ARMS to the expected arm files}"
: "${ARM_LINES:?set ARM_LINES to the expected per-arm line count}"
cd "$ROOT" || exit 1
ts() { date '+%F %T'; }

wrapper_alive() { pgrep -f "train_resilient.sh" > /dev/null; }
pipeline_alive() { pgrep -f "watch_and_generate.sh" > /dev/null; }
train_complete() {
    grep -q "TRAINING_COMPLETE\|TRAINING_ALREADY_COMPLETE" "$ROOT/train_resilient.log" 2>/dev/null
}
arms_done() {
    for f in $ARMS; do
        [ -s "$f" ] || return 1
        [ "$(wc -l < "$f")" -ge "$ARM_LINES" ] || return 1
        tail -1 "$f" | python3 -c "import json,sys; json.loads(sys.stdin.read())" 2>/dev/null || return 1
    done
    for f in ${DONE_FILES:-}; do
        [ -s "$f" ] || return 1
    done
}

if arms_done; then
    echo "$(ts) ALL_DONE"
    exit 0
fi

if train_complete; then
    if pipeline_alive; then
        echo "$(ts) training done, pipeline running"
    else
        echo "$(ts) training done, pipeline dead -> relaunch"
        setsid nohup bash "$(dirname "$0")/watch_and_generate.sh" \
            >> "$ROOT/pipeline_console.log" 2>&1 < /dev/null &
    fi
    exit 0
fi

if wrapper_alive; then
    echo "$(ts) training wrapper alive: $(tail -1 train_full.log 2>/dev/null | head -c 60)"
else
    echo "$(ts) wrapper dead -> relaunch (auto-resume from latest checkpoint)"
    setsid nohup bash "$(dirname "$0")/train_resilient.sh" \
        >> "$ROOT/train_resilient.log" 2>&1 < /dev/null &
    if ! pipeline_alive; then
        setsid nohup bash "$(dirname "$0")/watch_and_generate.sh" \
            >> "$ROOT/pipeline_console.log" 2>&1 < /dev/null &
    fi
fi
