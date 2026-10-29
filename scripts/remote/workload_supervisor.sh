#!/bin/bash
# Cron self-healing supervisor for the unattended generation workload.
#
# Install (launch_workload.sh does this for you):
#   */10 * * * * . $ROOT/workload.env && bash $ROOT/scripts/remote/workload_supervisor.sh >> $ROOT/supervisor.log 2>&1
#
# Every 10 minutes it checks the artifacts and either declares ALL_DONE,
# reports that run_workload.sh is still working, or relaunches it. A host
# reboot is covered too: cron starts this script again on its own, and
# run_workload.sh resumes from its stage markers.
#
# Environment (from workload.env):
#   ROOT         project directory on the GPU host (required)
#   ARMS_SPEC    "relpath:expected_lines" pairs, space separated (required)
#   DONE_FILES   post-processing outputs that must exist before ALL_DONE
#   OUT_DIR      where the decider writes its verdict (default
#                results/response_gen); needed for the conditional-arm gate
#   PY           python for the last-line JSON check (default python3)
#   MAX_ATTEMPTS relaunch budget before WORKLOAD_GAVE_UP (default 6)
#
# Retry budget: $ROOT/supervisor_attempts counts consecutive relaunches that
# made no progress; the counter resets as soon as a stage marker appears
# (progress) or the workload is done. A workload that keeps dying at the same
# stage therefore stops after MAX_ATTEMPTS tries and writes WORKLOAD_GAVE_UP
# instead of looping forever.
set -u

# Same contract as run_workload.sh: default to this script's repo location,
# then let workload.env (written by launch_workload.sh) fill in ROOT and the
# audit lists, so the user-space loop works even without cron sourcing it.
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-$(cd "$SELF_DIR/../.." && pwd)}"
if [ -f "$ROOT/workload.env" ]; then
    # shellcheck source=/dev/null
    . "$ROOT/workload.env"
fi

: "${ROOT:?set ROOT to the project directory}"
: "${ARMS_SPEC:?set ARMS_SPEC (\"relpath:expected_lines\" pairs, space separated)}"
# Resolve the sibling before cd: $0 may be relative, and the relaunch must
# work when the supervisor was started by hand from inside $ROOT.
SELF="$SELF_DIR/run_workload.sh"
cd "$ROOT" || exit 1
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"
ATTEMPTS_FILE="$ROOT/supervisor_attempts"
MARKERS_FILE="$ROOT/supervisor_markers"
GAVE_UP_FILE="$ROOT/WORKLOAD_GAVE_UP"
DONE_MARKER="$ROOT/WORKLOAD_DONE"
# Character class, not the literal name: pgrep -f matches full command lines,
# and a supervisor whose own cmdline contained "run_workload.sh" would find
# itself and never relaunch anything.
SELF_PATTERN="run_workloa[d].sh"

ts() { date '+%F %T'; }

case "$MAX_ATTEMPTS" in
    '' | *[!0-9]*) echo "$(ts) ERROR: MAX_ATTEMPTS must be a non-negative integer, got '$MAX_ATTEMPTS'" ; exit 1 ;;
esac

JSON_PY="${PY:-python3}"
command -v "$JSON_PY" > /dev/null 2>&1 || JSON_PY=python3
if ! command -v "$JSON_PY" > /dev/null 2>&1; then
    JSON_PY=""
fi

# The decider may obligate a ninth (conditional) arm whose paths are only
# known at runtime. run_workload.sh freezes them into the stage marker when
# the arm completes; until then static completion must not pass for done:
# the runner can die between the triggered decision and that arm, and the
# effective lists are only written at the very end of a successful pass.
# Keep these two paths in sync with run_workload.sh (asserted by tests).
COND_STAGE_NAME="16_t5_conditional_val"
COND_ANALYSIS_REL="reports/t5_conditional_val_error_analysis_newcard.json"
OUT_DIR_REL="${OUT_DIR:-results/response_gen}"
COND_DECISION_REL="$OUT_DIR_REL/t5_conditional_decision.json"
conditional_output() {
    # Prints the chosen output path when the frozen decision obligates the
    # conditional arm and nothing when it does not; returns non-zero when
    # the decision exists but cannot be read (never "nothing owed").
    local decision="$ROOT/$COND_DECISION_REL"
    [ -s "$decision" ] || return 1
    if [ -n "$JSON_PY" ]; then
        "$JSON_PY" - "$decision" 2>/dev/null <<'COND_PY'
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(3)
chosen = data.get("chosen") or {}
sys.stdout.write(str(chosen.get("output") or ""))
COND_PY
        return $?
    fi
    # No interpreter at all: the repo's own writer puts "output": "..." on a
    # single line, so a conservative grep keeps the gate closed, never open.
    grep -o '"output"[[:space:]]*:[[:space:]]*"[^"]\+"' "$decision" 2>/dev/null \
        | head -1 | sed -e 's/^.*:[[:space:]]*"//' -e 's/"$//'
    return 0
}
conditional_arm_met() {
    local output marker spec_line
    if ! output="$(conditional_output)"; then
        # No decision file yet: the runner has not reached stage 15, so the
        # static arms are all there is. An unreadable decision must not pass.
        [ -s "$ROOT/$COND_DECISION_REL" ] && return 1
        return 0
    fi
    [ -n "$output" ] || return 0
    marker="$ROOT/stages/$COND_STAGE_NAME.done"
    [ -f "$marker" ] || return 1
    spec_line="$(sed -n '1p' "$marker" 2>/dev/null)"
    if [ -n "$spec_line" ]; then
        [ "${spec_line%%:*}" = "$output" ] || return 1
    fi
    [ -s "$ROOT/$output" ] || return 1
    last_line_ok "$ROOT/$output" || return 1
    [ -s "$ROOT/$COND_ANALYSIS_REL" ] || return 1
    return 0
}

# run_workload.sh writes the effective lists when it reaches the audit: they
# are ARMS_SPEC/DONE_FILES plus the conditional arm, whose path is only known
# at runtime. Prefer them so a runtime-only arm is audited too.
effective_list() {
    local file="$1" fallback="$2"
    if [ -s "$file" ]; then
        tr '\n' ' ' < "$file"
    else
        printf '%s ' "$fallback"
    fi
}
ARMS_SPEC_EFFECTIVE="$(effective_list "$ROOT/workload_arms_effective.txt" "$ARMS_SPEC")"
DONE_FILES_EFFECTIVE="$(effective_list "$ROOT/workload_done_effective.txt" "${DONE_FILES:-}")"

last_line_ok() {
    if [ -n "$JSON_PY" ]; then
        tail -1 "$1" | "$JSON_PY" -c 'import json,sys; json.loads(sys.stdin.read())' > /dev/null 2>&1
    else
        # No interpreter at all: fall back to "ends like a JSON object" so a
        # missing python cannot make the supervisor loop forever.
        tail -1 "$1" | grep -q '}$'
    fi
}

arms_done() {
    local spec path want lines
    for spec in $ARMS_SPEC_EFFECTIVE; do
        path="${spec%%:*}"
        want="${spec##*:}"
        [ -n "$path" ] || return 1
        [ -s "$ROOT/$path" ] || return 1
        lines="$(wc -l < "$ROOT/$path" | tr -d ' ')"
        [ "$lines" -ge "$want" ] || return 1
        last_line_ok "$ROOT/$path" || return 1
    done
    for path in $DONE_FILES_EFFECTIVE; do
        [ -s "$ROOT/$path" ] || return 1
    done
}

read_counter() {
    local file="$1" value
    value="$(cat "$file" 2>/dev/null | tr -dc '0-9')"
    printf '%s' "${value:-0}"
}

marker_count() {
    ls -1 "$ROOT"/stages/*.done 2>/dev/null | grep -c . || true
}

if [ -f "$DONE_MARKER" ]; then
    rm -f "$ATTEMPTS_FILE" "$MARKERS_FILE"
    echo "$(ts) ALL_DONE (WORKLOAD_DONE present)"
    exit 0
fi

if arms_done && conditional_arm_met; then
    rm -f "$ATTEMPTS_FILE" "$MARKERS_FILE"
    echo "$(ts) ALL_DONE"
    exit 0
fi

if [ -f "$GAVE_UP_FILE" ]; then
    echo "$(ts) GAVE_UP recorded earlier, not relaunching (rm $GAVE_UP_FILE to re-arm)"
    exit 0
fi

if pgrep -f "$SELF_PATTERN" > /dev/null 2>&1; then
    echo "$(ts) workload running: $(tail -1 "$ROOT/workload.log" 2>/dev/null | head -c 80)"
    exit 0
fi

# run_workload.sh is dead and the artifacts are incomplete: relaunch, unless
# the budget is spent or the last relaunch made no progress at all.
attempts="$(read_counter "$ATTEMPTS_FILE")"
last_markers="$(read_counter "$MARKERS_FILE")"
now_markers="$(marker_count)"
if [ "$now_markers" -gt "$last_markers" ]; then
    attempts=0
fi
if [ "$attempts" -ge "$MAX_ATTEMPTS" ]; then
    : > "$GAVE_UP_FILE"
    echo "$(ts) GAVE_UP after $attempts relaunches without progress (markers=$now_markers)"
    exit 0
fi
attempts=$((attempts + 1))
printf '%s\n' "$attempts" > "$ATTEMPTS_FILE"
printf '%s\n' "$now_markers" > "$MARKERS_FILE"
setsid nohup bash "$SELF" >> "$ROOT/workload_console.log" 2>&1 < /dev/null &
echo "$(ts) workload dead -> relaunch attempt $attempts/$MAX_ATTEMPTS (markers=$now_markers)"
