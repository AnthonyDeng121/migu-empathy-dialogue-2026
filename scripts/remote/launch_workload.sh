#!/bin/bash
# One-shot launcher for the unattended generation workload on a fresh GPU
# instance. Safe to re-run: it never regenerates an existing workload.env,
# never starts a second run_workload.sh, and picks exactly one supervisor
# transport (cron when installable, a user-space loop otherwise).
#
# What it does, in order:
#   1. probe for a python that imports torch/transformers/peft with CUDA
#      (never silently falls back to CPU);
#   2. pick the GPU with the most free memory;
#   3. write $ROOT/workload.env when it does not exist yet;
#   4. install the 10-minute supervisor (cron, else a detached loop);
#   5. detach run_workload.sh so an SSH drop cannot kill it;
#   6. print where to look.
#
# Environment:
#   ROOT   project directory on the GPU host (default: this script's repo root)
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SUPERVISOR="$SCRIPT_DIR/workload_supervisor.sh"
RUNNER="$SCRIPT_DIR/run_workload.sh"
ENV_FILE="$ROOT/workload.env"
CONSOLE_LOG="$ROOT/workload_console.log"
LAUNCH_LOG="$ROOT/launch_workload.log"
SUPERVISOR_LOG="$ROOT/supervisor.log"
CRON_LINE="*/10 * * * * . $ENV_FILE && bash $SUPERVISOR >> $SUPERVISOR_LOG 2>&1"
PY_PROBE='import torch, transformers, peft; assert torch.cuda.is_available()'

mkdir -p "$ROOT" || exit 1
: > "$LAUNCH_LOG"
say() { echo "$*"; echo "=== $(date '+%F %T') $* ===" >> "$LAUNCH_LOG"; }

# 1. python: an explicit $PY wins if it works, then the usual suspects.
# AutoDL keeps conda in /root/miniconda3 and a non-interactive ssh session has
# no conda on PATH, so the absolute paths are probed explicitly.
PY_CANDIDATES=()
[ -n "${PY:-}" ] && PY_CANDIDATES+=("$PY")
PY_CANDIDATES+=(
    "/opt/conda/bin/python"
    "/root/miniconda3/bin/python"
    "python3"
    "python"
)
PY=""
for candidate in "${PY_CANDIDATES[@]}"; do
    command -v "$candidate" > /dev/null 2>&1 || continue
    if "$candidate" -c "$PY_PROBE" > /dev/null 2>&1; then
        PY="$candidate"
        break
    fi
    say "python candidate rejected (no CUDA torch stack): $candidate"
done
if [ -z "$PY" ]; then
    say "ERROR: no python with torch/transformers/peft and torch.cuda.is_available() found"
    say "ERROR: tried: ${PY_CANDIDATES[*]}"
    exit 1
fi
# cron runs with a minimal PATH, so record an absolute interpreter path.
PY_ABS="$("$PY" -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
[ -n "$PY_ABS" ] || PY_ABS="$PY"
say "python: $PY_ABS (from $PY)"

# 2. gpu: the card with the most free memory; a single-card instance gives 0.
# nvidia-smi prints CSV ("0, 24081"), so normalise the separator before the
# numeric parse or every row is rejected and the picker silently defaults.
GPU=""
if command -v nvidia-smi > /dev/null 2>&1; then
    best_free=-1
    while read -r idx free; do
        case "$idx" in '' | *[!0-9]*) continue ;; esac
        case "$free" in '' | *[!0-9]*) continue ;; esac
        if [ "$free" -gt "$best_free" ]; then
            best_free="$free"
            GPU="$idx"
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' ' ' || true)
fi
if [ -z "$GPU" ]; then
    GPU="0"
    say "nvidia-smi unavailable or unparsable: defaulting to GPU 0"
else
    say "gpu: $GPU (most free memory)"
fi

# 3. env file: only written once so a hand-tuned file is never clobbered.
if [ -f "$ENV_FILE" ]; then
    say "keeping existing $ENV_FILE (delete it to regenerate with python=$PY_ABS gpu=$GPU)"
else
    OUT_DIR="results/response_gen"
    ARMS_SPEC="$OUT_DIR/t4_longweight_selection_newcard.jsonl:160 \
$OUT_DIR/d4v2_selection_newcard.jsonl:160 \
$OUT_DIR/t4_emotion_only_selection_newcard.jsonl:160 \
$OUT_DIR/t4_emotion_profile_selection_newcard.jsonl:160 \
$OUT_DIR/d4v2_emotion_only_selection_newcard.jsonl:160 \
$OUT_DIR/d4v2_emotion_profile_selection_newcard.jsonl:160 \
$OUT_DIR/t4_longweight_val_newcard.jsonl:2354 \
$OUT_DIR/d4v2_val_newcard.jsonl:2354"
    DONE_FILES="\
$OUT_DIR/t4_arms_comparison_newcard.json \
reports/t4_val_error_analysis_newcard.json \
$OUT_DIR/t5_d4v2_emotion_vs_off_newcard.json \
$OUT_DIR/t5_d4v2_eprofile_vs_off_newcard.json \
$OUT_DIR/t5_t4_emotion_vs_off_newcard.json \
$OUT_DIR/t5_t4_eprofile_vs_off_newcard.json \
$OUT_DIR/t5_conditional_decision.json"
    # Quoting is by hand: these values are repo-relative paths, and a value
    # with a quote/backslash/dollar in it would silently change meaning when
    # cron sources the file, so refuse instead of writing it.
    # ADAPTER_SHA256S pins the two adapters by full digest: the upload
    # manifest proves the bytes survived the transfer, these prove the bytes
    # are the weights this cycle actually produced (T4 bc6ca023..., D4v2
    # 6b75245a...).
    ADAPTER_SHA256S="weights/t4_longweight_2ep:bc6ca0234eb7222e526e651364326131c7b98f5d56c18a6301ca8c4b193d2f4e weights/response_sft_d4_2ep:6b75245a4eddff63da77d260a509028ce85a89c73bef58bda4e33934ca7b767e"
    for value in "$ROOT" "$PY_ABS" "$GPU" "weights/hf/Qwen2.5-1.5B-Instruct" \
        "weights/t4_longweight_2ep" "weights/response_sft_d4_2ep" \
        "results/response_gen/t2_selection_input.jsonl" \
        "data/processed/b1_full_v1/response_val.jsonl" \
        "results/response_gen/c5_conditions_val_full.jsonl" \
        "results/response_gen/t5_conditions_selection_both.jsonl" \
        "results/response_gen/t5_conditions_val_both.jsonl" "$OUT_DIR" \
        "$ARMS_SPEC" "$DONE_FILES" "$ADAPTER_SHA256S"; do
        case "$value" in
            *'"'* | *'$'* | *'`'* | *'\'*)
                say "ERROR: refusing to write $ENV_FILE: unsafe value '$value'"
                exit 1
                ;;
        esac
    done
    {
        printf '# Generated by launch_workload.sh on %s\n' "$(date '+%F %T')"
        printf '# Sourced by cron, the supervisor and run_workload.sh; edit freely.\n'
        printf 'export ROOT="%s"\n' "$ROOT"
        printf 'export PY="%s"\n' "$PY_ABS"
        printf 'export GPU="%s"\n' "$GPU"
        printf 'export MODEL_DIR="%s"\n' "weights/hf/Qwen2.5-1.5B-Instruct"
        printf 'export T4_ADAPTER="%s"\n' "weights/t4_longweight_2ep"
        printf 'export D4V2_ADAPTER="%s"\n' "weights/response_sft_d4_2ep"
        printf 'export SEL_INPUT="%s"\n' "results/response_gen/t2_selection_input.jsonl"
        printf 'export VAL_INPUT="%s"\n' "data/processed/b1_full_v1/response_val.jsonl"
        printf 'export C5_CONDITIONS="%s"\n' "results/response_gen/c5_conditions_val_full.jsonl"
        printf 'export BOTH_COND_SEL="%s"\n' "results/response_gen/t5_conditions_selection_both.jsonl"
        printf 'export BOTH_COND_VAL="%s"\n' "results/response_gen/t5_conditions_val_both.jsonl"
        printf 'export OUT_DIR="%s"\n' "$OUT_DIR"
        printf 'export ARMS_SPEC="%s"\n' "$ARMS_SPEC"
        printf 'export DONE_FILES="%s"\n' "$DONE_FILES"
        printf 'export MAX_ATTEMPTS="%s"\n' "6"
        printf 'export BASE_WAIT_MINUTES="%s"\n' "120"
        printf 'export ADAPTER_SHA256S="%s"\n' "$ADAPTER_SHA256S"
    } > "$ENV_FILE"
    say "wrote $ENV_FILE"
fi

# 4. supervisor: cron when it can be installed, otherwise a detached loop.
# Exactly one of the two may run, or both would relaunch the workload.
kill_user_loop() {
    local pid cmdline
    for pid in $(pgrep -f 'while true; do bash' 2>/dev/null || true); do
        [ "$pid" = "$$" ] && continue
        cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
        case "$cmdline" in
            *workload_supervisor.sh*)
                kill "$pid" 2>/dev/null || true
                say "stopped a user-space supervisor loop pid=$pid"
                ;;
        esac
    done
}

install_cron() {
    service cron start > /dev/null 2>&1 || cron > /dev/null 2>&1 || true
    if ! command -v crontab > /dev/null 2>&1; then
        return 1
    fi
    ( crontab -l 2>/dev/null | grep -v 'workload_supervisor.sh'
      echo "$CRON_LINE" ) | crontab - > /dev/null 2>&1
}

cron_mode="loop"
if command -v cron > /dev/null 2>&1; then
    cron_mode="cron"
else
    say "cron not installed, trying apt-get install -y cron"
    if DEBIAN_FRONTEND=noninteractive apt-get install -y cron >> "$LAUNCH_LOG" 2>&1; then
        hash -r
        if command -v cron > /dev/null 2>&1; then
            cron_mode="cron"
        fi
    else
        say "apt-get install cron failed (not root? no network?): falling back to a user-space loop"
    fi
fi

if [ "$cron_mode" = "cron" ]; then
    kill_user_loop
    if install_cron; then
        say "supervisor installed in crontab: $CRON_LINE"
    else
        cron_mode="loop"
        say "crontab install failed: falling back to a user-space loop"
    fi
fi

if [ "$cron_mode" = "loop" ]; then
    # Source the env file first, exactly like the crontab line does: the
    # supervisor self-defaults ROOT too, but this keeps both transports
    # identical (and skips the loop entirely if the env file is missing).
    setsid nohup bash -c 'while true; do . "$1" && bash "$2" >> "$3" 2>&1; sleep 600; done' \
        _ "$ENV_FILE" "$SUPERVISOR" "$SUPERVISOR_LOG" >> "$LAUNCH_LOG" 2>&1 < /dev/null &
    say "supervisor started as a detached 10-minute loop (no cron on this image)"
fi

# 5. the workload itself, detached from this SSH session.
if pgrep -f 'run_workloa[d].sh' > /dev/null 2>&1; then
    say "run_workload.sh is already running, not starting a second copy"
else
    setsid nohup bash "$RUNNER" >> "$CONSOLE_LOG" 2>&1 < /dev/null &
    say "run_workload.sh launched, console log: $CONSOLE_LOG"
fi

# 6. where to look.
say "----- $ENV_FILE -----"
cat "$ENV_FILE"
say "---------------------"
say "logs: $ROOT/workload.log (stages), $ROOT/preflight.json, $ROOT/workload_status.json, $SUPERVISOR_LOG"
say "next: tail -f $ROOT/workload.log"
say "done check: grep -c WORKLOAD_DONE $ROOT/workload.log"
