#!/bin/bash
# Unattended generation workload for the remote 4090 instance (no training).
#
# Order of business: preflight, then the stages listed by $STAGES_FILE, then
# the conditional full-val arm the decider picked, then the artifact audit.
# Recovery is idempotent on two levels:
#   * stage markers   $ROOT/stages/<NN>_<name>.done  -> a relaunch skips
#     finished stages instead of regenerating them;
#   * resume journals $OUT_DIR/<arm>_progress        -> a half-written arm
#     continues from the verified records (src/common/inference_resume.py).
#
# Failure policy: any preflight problem, any failed stage and any arm that
# misses its expected line count writes $ROOT/workload_status.json and exits
# 1 *without touching the next stage*. workload_supervisor.sh owns retries;
# a relaunch continues from the markers, so retrying is cheap.
#
# Environment (injected through $ROOT/workload.env, written by
# launch_workload.sh; cron and the supervisor source the same file):
#   ROOT                project directory on the GPU host (required)
#   PY                  python with torch/transformers/peft + CUDA (required)
#   GPU                 CUDA_VISIBLE_DEVICES value for this run (required)
#   MODEL_DIR           base snapshot, relative to ROOT (required)
#   T4_ADAPTER          T4 LoRA adapter, relative to ROOT (required)
#   D4V2_ADAPTER        D4v2 LoRA adapter, relative to ROOT (required)
#   SEL_INPUT           160-row selection input jsonl (required)
#   VAL_INPUT           2354-row val input jsonl (required)
#   C5_CONDITIONS       emotion-only condition jsonl (required)
#   BOTH_COND_SEL       emotion+profile conditions, selection set (required)
#   BOTH_COND_VAL       emotion+profile conditions, val set (required)
#   OUT_DIR             predictions and reports (required)
#   ARMS_SPEC           "relpath:expected_lines" pairs, space separated (required)
#   DONE_FILES          extra outputs that must exist before ALL_DONE (optional)
#   STAGES_FILE         stage list (default scripts/remote/workload_stages.sh)
#   BASE_READY_FILE     base readiness marker (default $ROOT/base_ready)
#   BASE_WAIT_MINUTES   base wait budget (default 120; 0 = fail immediately)
#   BASE_SHA256_EXPECTED  known-good base sha256 (default dd924a11...)
#   BASE_MODEL_SIZE     expected base size in bytes (default 3087467144)
#   BASE_SHA256_STRICT  1 = wrong base sha256 aborts (default), 0 = warn only
#   ADAPTER_SHA256S     "relpath:sha256_prefix" pairs; an empty prefix means
#                       presence+size check only (default: the two adapters)
#   MAX_ATTEMPTS        supervisor relaunch budget (default 6; recorded only)
#   PIP_PACKAGES        what a missing transformers/peft/accelerate is
#                       repaired with (default: the requirements-response.txt
#                       pins; torch is never installed here)
set -u

: "${ROOT:?set ROOT to the project directory}"
cd "$ROOT" || exit 1

# workload.env is the single source of truth for this run: launch_workload.sh
# writes it, cron and the supervisor source it, and re-sourcing it here makes
# a manual `bash run_workload.sh` behave exactly like a supervisor relaunch.
if [ -f "$ROOT/workload.env" ]; then
    # shellcheck source=/dev/null
    . "$ROOT/workload.env"
fi

: "${PY:?set PY to a python that imports torch/transformers/peft with CUDA}"
: "${GPU:?set GPU to the CUDA_VISIBLE_DEVICES value for this run}"
: "${MODEL_DIR:?set MODEL_DIR (base snapshot, relative to ROOT)}"
: "${T4_ADAPTER:?set T4_ADAPTER (relative to ROOT)}"
: "${D4V2_ADAPTER:?set D4V2_ADAPTER (relative to ROOT)}"
: "${SEL_INPUT:?set SEL_INPUT (selection input jsonl)}"
: "${VAL_INPUT:?set VAL_INPUT (val input jsonl)}"
: "${C5_CONDITIONS:?set C5_CONDITIONS (emotion-only conditions)}"
: "${BOTH_COND_SEL:?set BOTH_COND_SEL (emotion+profile conditions, selection set)}"
: "${BOTH_COND_VAL:?set BOTH_COND_VAL (emotion+profile conditions, val set)}"
: "${OUT_DIR:?set OUT_DIR (predictions and reports)}"
: "${ARMS_SPEC:?set ARMS_SPEC (\"relpath:expected_lines\" pairs, space separated)}"

DONE_FILES="${DONE_FILES:-}"
STAGES_FILE="${STAGES_FILE:-scripts/remote/workload_stages.sh}"
BASE_READY_FILE="${BASE_READY_FILE:-$ROOT/base_ready}"
BASE_WAIT_MINUTES="${BASE_WAIT_MINUTES:-120}"
BASE_SHA256_EXPECTED="${BASE_SHA256_EXPECTED:-dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee}"
BASE_MODEL_SIZE="${BASE_MODEL_SIZE:-3087467144}"
BASE_SHA256_STRICT="${BASE_SHA256_STRICT:-1}"
ADAPTER_SHA256S="${ADAPTER_SHA256S:-}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"
# requirements-response.txt pins. A fresh image ships torch but may lack
# transformers/peft/accelerate; those three are the only ones ever installed
# here (torch matches the pin already and is never touched).
PIP_PACKAGES="${PIP_PACKAGES:-transformers==5.17.0 peft==0.21.2 accelerate==1.15.0}"

# Numeric env values reach this script from a hand-editable file, so refuse to
# trust a non-numeric one (an arithmetic error here would abort mid-preflight).
for numeric in BASE_WAIT_MINUTES BASE_MODEL_SIZE MAX_ATTEMPTS; do
    case "${!numeric}" in
        '' | *[!0-9]*) echo "ERROR: $numeric must be a non-negative integer, got '${!numeric}'" >&2; exit 1 ;;
    esac
done

LOG="$ROOT/workload.log"
PREFLIGHT_JSON="$ROOT/preflight.json"
STATUS_JSON="$ROOT/workload_status.json"
STAGES_DIR="$ROOT/stages"
STAGE_STATUS_TSV="$ROOT/workload_stage_status.tsv"
ARMS_EFFECTIVE="$ROOT/workload_arms_effective.txt"
DONE_EFFECTIVE="$ROOT/workload_done_effective.txt"
GPU_SAMPLER_PID_FILE="$ROOT/gpu_sampler.pid"
GPU_USAGE_LOG="$ROOT/gpu_usage.log"
DEPS_LOG="$ROOT/workload_deps_install.log"
START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# These two paths must stay in sync with workload_stages.sh; the consistency
# is asserted by tests/test_workload_scripts.py.
T4_VAL_OUT="$OUT_DIR/t4_longweight_val_newcard.jsonl"
D4V2_VAL_OUT="$OUT_DIR/d4v2_val_newcard.jsonl"
COND_STAGE_NAME="16_t5_conditional_val"
COND_ANALYSIS="reports/t5_conditional_val_error_analysis_newcard.json"

mkdir -p "$STAGES_DIR" "$OUT_DIR" "$ROOT/reports" || exit 1
: > "$STAGE_STATUS_TSV"

log() { echo "=== $(date '+%F %T') $* ===" >> "$LOG"; }
say() { echo "$*"; log "$*"; }
json_escape() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | tr -d '\r\n'; }
sha_of() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
recorded_base_sha() {
    [ -s "$BASE_READY_FILE" ] || return 1
    grep -o -m1 -E '[0-9a-fA-F]{64}' "$BASE_READY_FILE" 2>/dev/null | head -1 | tr 'A-F' 'a-f'
}
last_line_ok() {
    tail -1 "$1" | "$PY" -c 'import json,sys; json.loads(sys.stdin.read())' > /dev/null 2>&1
}

WARNINGS=()
warn() {
    log "WARNING: $*"
    WARNINGS+=("$*")
}

record_stage() { printf '%s\t%s\n' "$1" "$2" >> "$STAGE_STATUS_TSV"; }

# ---------------------------------------------------------------- preflight --
PF_PYTHON="$PY"
PF_TORCH="unknown"
PF_TRANSFORMERS="unknown"
PF_PEFT="unknown"
PF_ACCELERATE="unknown"
PF_DEPS_REPAIRED=""
PF_GPU_NAME="unavailable"
PF_CUDA="false"
PF_MODEL_SHA_RECORDED=""
PF_MODEL_SHA_ACTUAL=""
PF_MODEL_SIZE="0"
PF_BASE_WAIT_SECONDS="0"
PF_MANIFEST="MANIFEST.sha256"
PF_MANIFEST_OK="false"
PF_ADAPTERS=""

write_preflight_json() {
    # $1 = outcome ("pass"/"fail"), $2 = reason
    local outcome="$1" reason="$2" warnings_json="" deps_json="" entry w module
    case "$outcome" in
        pass) : ;;
        *) outcome="fail" ;;
    esac
    for w in ${WARNINGS[@]+"${WARNINGS[@]}"}; do
        [ -n "$warnings_json" ] && warnings_json="$warnings_json, "
        warnings_json="$warnings_json\"$(json_escape "$w")\""
    done
    for module in $PF_DEPS_REPAIRED; do
        [ -n "$deps_json" ] && deps_json="$deps_json, "
        deps_json="$deps_json\"$(json_escape "$module")\""
    done
    {
        printf '{\n'
        printf '  "outcome": "%s",\n' "$(json_escape "$outcome")"
        printf '  "reason": "%s",\n' "$(json_escape "$reason")"
        printf '  "checked_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '  "root": "%s",\n' "$(json_escape "$ROOT")"
        printf '  "python": "%s",\n' "$(json_escape "$PF_PYTHON")"
        printf '  "torch": "%s",\n' "$(json_escape "$PF_TORCH")"
        printf '  "transformers": "%s",\n' "$(json_escape "$PF_TRANSFORMERS")"
        printf '  "peft": "%s",\n' "$(json_escape "$PF_PEFT")"
        printf '  "accelerate": "%s",\n' "$(json_escape "$PF_ACCELERATE")"
        printf '  "deps_repaired": [%s],\n' "$deps_json"
        printf '  "deps_install_log": "%s",\n' "$(json_escape "$DEPS_LOG")"
        printf '  "pip_packages": "%s",\n' "$(json_escape "$PIP_PACKAGES")"
        printf '  "cuda_available": %s,\n' "$PF_CUDA"
        printf '  "gpu_index": "%s",\n' "$(json_escape "$GPU")"
        printf '  "gpu_name": "%s",\n' "$(json_escape "$PF_GPU_NAME")"
        printf '  "model_dir": "%s",\n' "$(json_escape "$MODEL_DIR")"
        printf '  "model_file": "%s",\n' "$(json_escape "$ROOT/$MODEL_DIR/model.safetensors")"
        printf '  "model_sha256_recorded": "%s",\n' "$(json_escape "$PF_MODEL_SHA_RECORDED")"
        printf '  "model_sha256_actual": "%s",\n' "$(json_escape "$PF_MODEL_SHA_ACTUAL")"
        printf '  "model_sha256_expected": "%s",\n' "$(json_escape "$BASE_SHA256_EXPECTED")"
        printf '  "model_size_bytes": %s,\n' "${PF_MODEL_SIZE:-0}"
        printf '  "model_size_expected": %s,\n' "$BASE_MODEL_SIZE"
        printf '  "base_ready_file": "%s",\n' "$(json_escape "$BASE_READY_FILE")"
        printf '  "base_wait_seconds": %s,\n' "${PF_BASE_WAIT_SECONDS:-0}"
        printf '  "manifest": "%s",\n' "$(json_escape "$PF_MANIFEST")"
        printf '  "manifest_ok": %s,\n' "$PF_MANIFEST_OK"
        printf '  "adapters": [%s],\n' "$PF_ADAPTERS"
        printf '  "warnings": [%s]\n' "$warnings_json"
        printf '}\n'
    } > "$PREFLIGHT_JSON"
}

write_status_json() {
    # $1 = outcome, $2 = reason, $3 = failed stage ("" when none)
    local outcome="$1" reason="$2" failed_stage="${3:-}"
    local stages_json="" sname sstatus w
    if [ -f "$STAGE_STATUS_TSV" ]; then
        while IFS=$'\t' read -r sname sstatus; do
            [ -n "${sname:-}" ] || continue
            [ -n "$stages_json" ] && stages_json="$stages_json, "
            stages_json="$stages_json\"$(json_escape "$sname")\": \"$(json_escape "${sstatus:-pending}")\""
        done < "$STAGE_STATUS_TSV"
    fi
    {
        printf '{\n'
        printf '  "workload": "response_generation_unattended_v1",\n'
        printf '  "outcome": "%s",\n' "$(json_escape "$outcome")"
        printf '  "reason": "%s",\n' "$(json_escape "$reason")"
        printf '  "failed_stage": "%s",\n' "$(json_escape "$failed_stage")"
        printf '  "started_at": "%s",\n' "$START_TS"
        printf '  "finished_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '  "root": "%s",\n' "$(json_escape "$ROOT")"
        printf '  "python": "%s",\n' "$(json_escape "$PF_PYTHON")"
        printf '  "torch": "%s",\n' "$(json_escape "$PF_TORCH")"
        printf '  "transformers": "%s",\n' "$(json_escape "$PF_TRANSFORMERS")"
        printf '  "peft": "%s",\n' "$(json_escape "$PF_PEFT")"
        printf '  "gpu_index": "%s",\n' "$(json_escape "$GPU")"
        printf '  "gpu_name": "%s",\n' "$(json_escape "$PF_GPU_NAME")"
        printf '  "stages": {%s},\n' "$stages_json"
        printf '  "arms": {%s},\n' "$ARMS_JSON"
        printf '  "done_files": {%s},\n' "$DONE_JSON"
        printf '  "preflight_file": "%s",\n' "$(json_escape "$PREFLIGHT_JSON")"
        printf '  "preflight_ok": %s,\n' "$PF_MANIFEST_OK"
        printf '  "max_attempts": %s\n' "$MAX_ATTEMPTS"
        printf '}\n'
    } > "$STATUS_JSON"
}

ARMS_JSON=""
DONE_JSON=""
abort_workload() {
    # $1 outcome, $2 reason, $3 failed stage, $4 "yes" to also write preflight.json
    local outcome="$1" reason="$2" failed_stage="${3:-}" also_preflight="${4:-no}"
    [ "$also_preflight" = "yes" ] && write_preflight_json "$outcome" "$reason"
    write_status_json "$outcome" "$reason" "$failed_stage"
    say "WORKLOAD ABORTED ($outcome): $reason"
    exit 1
}
preflight_fail() { abort_workload "preflight_failed" "$1" "" "yes"; }

say "workload start root=$ROOT gpu=$GPU stages=$STAGES_FILE"

# (a) python stack: torch must already be importable (the image ships torch
# 2.5.1, matching requirements-response.txt; the code runs on transformers
# 4.57.6 and 5.17.0 alike, and torch is never installed here).
# transformers/peft/accelerate are what a fresh image may lack, so those are
# the three this workload repairs from the same pins.
if ! "$PY" -c 'import torch' >> "$LOG" 2>&1; then
    preflight_fail "python stack probe failed: $PY cannot import torch (the image ships torch 2.5.1; this workload never installs torch)"
fi
say "preflight: torch importable"
PF_TORCH="$("$PY" -c 'import torch; print(torch.__version__)' 2>> "$LOG" || echo unknown)"

module_imports() { "$PY" -c "import $1" >> "$LOG" 2>&1; }
DEPS_MISSING=""
for module in transformers peft accelerate; do
    module_imports "$module" || DEPS_MISSING="$DEPS_MISSING $module"
done
if [ -n "$DEPS_MISSING" ]; then
    say "preflight: repairing missing modules:$DEPS_MISSING"
    log "dependency repair: $PY -m pip install --no-input $PIP_PACKAGES"
    : > "$DEPS_LOG"
    if ! "$PY" -m pip install --no-input $PIP_PACKAGES >> "$DEPS_LOG" 2>&1; then
        preflight_fail "pip install $PIP_PACKAGES failed (see $DEPS_LOG)"
    fi
    for module in $DEPS_MISSING; do
        module_imports "$module" || preflight_fail "module $module is still not importable after the repair (see $DEPS_LOG)"
    done
    PF_DEPS_REPAIRED="$DEPS_MISSING"
    say "preflight: dependency repair ok (installed: $DEPS_MISSING, see $DEPS_LOG)"
fi
PF_TRANSFORMERS="$("$PY" -c 'import transformers; print(transformers.__version__)' 2>> "$LOG" || echo unknown)"
PF_PEFT="$("$PY" -c 'import peft; print(peft.__version__)' 2>> "$LOG" || echo unknown)"
PF_ACCELERATE="$("$PY" -c 'import accelerate; print(accelerate.__version__)' 2>> "$LOG" || echo unknown)"
if ! "$PY" -c 'import torch, transformers, peft; assert torch.cuda.is_available()' >> "$LOG" 2>&1; then
    preflight_fail "python stack probe failed: $PY must import torch/transformers/peft with torch.cuda.is_available()"
fi
say "preflight: python stack ok (torch=$PF_TORCH transformers=$PF_TRANSFORMERS peft=$PF_PEFT accelerate=$PF_ACCELERATE)"
PF_CUDA="true"
if command -v nvidia-smi > /dev/null 2>&1; then
    gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    case "$GPU" in
        '' | *[!0-9]*) PF_GPU_NAME="$(printf '%s\n' "$gpu_names" | head -1)" ;;
        *) PF_GPU_NAME="$(printf '%s\n' "$gpu_names" | sed -n "$((GPU + 1))p")" ;;
    esac
    [ -n "$PF_GPU_NAME" ] || PF_GPU_NAME="unavailable"
fi

# (b) base model: the upload script drops $ROOT/base_ready with the sha256 it
# uploaded; wait for it to describe the file that is actually on disk.
base_model_file="$ROOT/$MODEL_DIR/model.safetensors"
base_deadline=$(( $(date +%s) + BASE_WAIT_MINUTES * 60 ))
base_wait_started="$(date +%s)"
base_ok=0
while :; do
    if [ -s "$base_model_file" ]; then
        PF_MODEL_SIZE="$(wc -c < "$base_model_file" | tr -d ' ')"
        PF_MODEL_SHA_RECORDED="$(recorded_base_sha || true)"
        if [ -n "$PF_MODEL_SHA_RECORDED" ]; then
            PF_MODEL_SHA_ACTUAL="$(sha_of "$base_model_file")"
            if [ -n "$PF_MODEL_SHA_ACTUAL" ] && [ "$PF_MODEL_SHA_RECORDED" = "$PF_MODEL_SHA_ACTUAL" ]; then
                base_ok=1
                break
            fi
        fi
    fi
    [ "$(date +%s)" -ge "$base_deadline" ] && break
    log "base model not ready ($base_model_file): waiting for $BASE_READY_FILE to record a matching sha256"
    sleep 30
done
PF_BASE_WAIT_SECONDS=$(( $(date +%s) - base_wait_started ))
if [ "$base_ok" -ne 1 ]; then
    preflight_fail "base model not ready after ${PF_BASE_WAIT_SECONDS}s (marker=$BASE_READY_FILE recorded=${PF_MODEL_SHA_RECORDED:-none} file=$base_model_file)"
fi
if [ "$BASE_SHA256_STRICT" = "1" ] && [ "$PF_MODEL_SHA_ACTUAL" != "$BASE_SHA256_EXPECTED" ]; then
    preflight_fail "base sha256 $PF_MODEL_SHA_ACTUAL is not the expected $BASE_SHA256_EXPECTED (set BASE_SHA256_STRICT=0 to allow a different base)"
fi
if [ "$PF_MODEL_SHA_ACTUAL" != "$BASE_SHA256_EXPECTED" ]; then
    warn "base sha256 differs from the known-good $BASE_SHA256_EXPECTED (allowed by BASE_SHA256_STRICT=0)"
fi
if [ "$PF_MODEL_SIZE" != "$BASE_MODEL_SIZE" ]; then
    warn "base size ${PF_MODEL_SIZE}B differs from the expected ${BASE_MODEL_SIZE}B"
fi
say "preflight: base model ok (sha256=$PF_MODEL_SHA_ACTUAL size=${PF_MODEL_SIZE}B wait=${PF_BASE_WAIT_SECONDS}s)"

# (c) upload integrity: the manifest the upload script wrote covers every
# uploaded file, so a truncated rsync can never reach the stages.
if [ ! -f "$ROOT/$PF_MANIFEST" ]; then
    preflight_fail "upload manifest missing: $ROOT/$PF_MANIFEST"
fi
manifest_out="$(cd "$ROOT" && sha256sum -c "$PF_MANIFEST" 2>&1)"
manifest_rc=$?
printf '%s\n' "$manifest_out" >> "$LOG"
if [ "$manifest_rc" -ne 0 ]; then
    manifest_bad="$(printf '%s\n' "$manifest_out" | grep -v ': OK$' | head -3 | tr '\n' ';')"
    preflight_fail "upload manifest check failed: ${manifest_bad:-sha256sum -c exited $manifest_rc}"
fi
PF_MANIFEST_OK="true"
say "preflight: upload manifest ok"

# (d) adapters: expected hashes are optional (a prefix is enough); without
# them we only prove the weights are present and non-empty.
if [ -z "$ADAPTER_SHA256S" ]; then
    ADAPTER_SHA256S="$T4_ADAPTER: $D4V2_ADAPTER:"
    warn "ADAPTER_SHA256S unset: $T4_ADAPTER and $D4V2_ADAPTER checked for presence and size only"
fi
PF_ADAPTERS=""
for entry in $ADAPTER_SHA256S; do
    apath="${entry%%:*}"
    expect="${entry#*:}"
    if [ "$apath" = "$entry" ] || [ -z "$apath" ]; then
        warn "ADAPTER_SHA256S entry '$entry' is not 'relpath:sha256' - skipped"
        continue
    fi
    afile="$ROOT/$apath/adapter_model.safetensors"
    if [ ! -s "$afile" ]; then
        preflight_fail "adapter weights missing or empty: $afile"
    fi
    asize="$(wc -c < "$afile" | tr -d ' ')"
    aok="true"
    if [ -z "$expect" ]; then
        warn "no expected sha256 for $apath (presence+size check only, ${asize}B)"
    else
        if [ "${#expect}" -lt 16 ]; then
            warn "expected sha256 for $apath is shorter than 16 hex chars - weak check"
        fi
        aactual="$(sha_of "$afile")"
        case "$aactual" in
            "$expect"*) : ;;
            *) preflight_fail "adapter sha256 mismatch for $apath: expected prefix '$expect', got '${aactual:-<unreadable>}'" ;;
        esac
    fi
    [ -n "$PF_ADAPTERS" ] && PF_ADAPTERS="$PF_ADAPTERS, "
    PF_ADAPTERS="$PF_ADAPTERS{\"path\": \"$(json_escape "$apath")\", \"size_bytes\": $asize, \"sha256_ok\": $aok}"
done
say "preflight: adapters ok"

write_preflight_json "pass" "all preflight checks passed"
say "PREFLIGHT_OK (details in $PREFLIGHT_JSON)"

# ------------------------------------------------------------ gpu sampling --
start_gpu_sampler() {
    if ! command -v nvidia-smi > /dev/null 2>&1; then
        warn "nvidia-smi unavailable: no GPU sampling for this run"
        return 0
    fi
    if [ -f "$GPU_SAMPLER_PID_FILE" ]; then
        stale_pid="$(cat "$GPU_SAMPLER_PID_FILE" 2>/dev/null || true)"
        if [ -n "$stale_pid" ] && kill -0 "$stale_pid" 2>/dev/null; then
            kill "$stale_pid" 2>/dev/null || true
            log "stopped stale gpu sampler pid=$stale_pid"
        fi
        rm -f "$GPU_SAMPLER_PID_FILE"
    fi
    nohup nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 5 \
        >> "$GPU_USAGE_LOG" 2>&1 < /dev/null &
    echo $! > "$GPU_SAMPLER_PID_FILE"
    log "gpu sampler started pid=$(cat "$GPU_SAMPLER_PID_FILE") every 5s -> $GPU_USAGE_LOG"
}
stop_gpu_sampler() {
    [ -f "$GPU_SAMPLER_PID_FILE" ] || return 0
    local pid
    pid="$(cat "$GPU_SAMPLER_PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        log "gpu sampler stopped pid=$pid"
    fi
    rm -f "$GPU_SAMPLER_PID_FILE"
}
start_gpu_sampler
trap 'stop_gpu_sampler' EXIT

# ------------------------------------------------------------------- stages --
if [ ! -f "$STAGES_FILE" ]; then
    abort_workload "stage_failed" "stages file missing: $STAGES_FILE" ""
fi
# shellcheck source=/dev/null
. "$STAGES_FILE"
if ! command -v workload_stages > /dev/null 2>&1; then
    abort_workload "stage_failed" "$STAGES_FILE does not define workload_stages()" ""
fi
stage_lines="$(workload_stages 2>> "$LOG")"
if [ -z "$stage_lines" ]; then
    abort_workload "stage_failed" "workload_stages() printed no stages" ""
fi
say "stages loaded from $STAGES_FILE ($(printf '%s\n' "$stage_lines" | grep -c .) entries)"

while IFS= read -r line; do
    [ -n "$line" ] || continue
    case "$line" in
        *'|'*) : ;;
        *) abort_workload "stage_failed" "stage line missing '|' separator: $line" "" ;;
    esac
    name="${line%%|*}"
    command="${line#*|}"
    marker="$STAGES_DIR/$name.done"
    if [ -f "$marker" ]; then
        log "stage $name skipped (marker present)"
        record_stage "$name" "skipped"
        continue
    fi
    log "stage $name started: $command"
    # stdin comes from /dev/null so a stage that reads stdin cannot eat the
    # remaining stage lines out of this loop.
    CUDA_VISIBLE_DEVICES="$GPU" bash -c "$command" >> "$LOG" 2>&1 < /dev/null
    rc=$?
    if [ "$rc" -ne 0 ]; then
        record_stage "$name" "failed"
        log "stage $name FAILED rc=$rc; log tail: $(tail -3 "$LOG" | tr '\n' ' ')"
        abort_workload "stage_failed" "stage $name exited with code $rc" "$name"
    fi
    touch "$marker"
    record_stage "$name" "done"
    log "stage $name finished"
done <<< "$stage_lines"

# ---------------------------------------------- conditional full-val (T5) ----
# The decider's verdict is the only thing that may add a ninth arm, and its
# output path is only known at runtime, so this stage lives outside
# workload_stages.sh and still gets a marker of its own.
COND_ARM_SPEC=""
COND_ANALYSIS_DONE=""
if [ -f "$STAGES_DIR/$COND_STAGE_NAME.done" ]; then
    log "conditional full-val arm skipped (marker present)"
    record_stage "$COND_STAGE_NAME" "skipped"
else
    decision="$OUT_DIR/t5_conditional_decision.json"
    if [ ! -s "$decision" ]; then
        abort_workload "stage_failed" "conditional decision file missing or empty: $decision" "15_t5_decide"
    fi
    if ! chosen_fields="$("$PY" -c '
import json, shlex, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
chosen = data.get("chosen") or {}
for key in ("arm", "adapter", "condition_jsonl", "condition_fields", "output", "resume_dir"):
    value = chosen.get(key)
    print("CHOSEN_%s=%s" % (key.upper(), shlex.quote("" if value is None else str(value))))
' "$decision" 2>> "$LOG")"; then
        abort_workload "stage_failed" "could not parse $decision (see workload.log)" "$COND_STAGE_NAME"
    fi
    eval "$chosen_fields"
    if [ -z "$CHOSEN_OUTPUT" ]; then
        log "conditional full-val arm skipped: decider chose no arm (trigger=false)"
        record_stage "$COND_STAGE_NAME" "skipped"
        touch "$STAGES_DIR/$COND_STAGE_NAME.done"
    else
        for value in "$CHOSEN_ADAPTER" "$CHOSEN_CONDITION_JSONL" "$CHOSEN_OUTPUT" "$CHOSEN_RESUME_DIR" "$CHOSEN_CONDITION_FIELDS"; do
            case "$value" in
                *[[:space:]]* | *\'* | *\"*)
                    abort_workload "stage_failed" "decider field with unsafe characters: '$value'" "$COND_STAGE_NAME" ;;
            esac
        done
        cond_report="${CHOSEN_OUTPUT%.jsonl}_report.json"
        cond_cmd="$PY scripts/run_response_inference.py --input $VAL_INPUT --model $MODEL_DIR --adapter $CHOSEN_ADAPTER --condition-jsonl $CHOSEN_CONDITION_JSONL --condition-fields $CHOSEN_CONDITION_FIELDS --output $CHOSEN_OUTPUT --performance-report $cond_report --resume-dir $CHOSEN_RESUME_DIR --max-new-tokens 256 --device cuda"
        log "stage $COND_STAGE_NAME started ($CHOSEN_ARM): $cond_cmd"
        CUDA_VISIBLE_DEVICES="$GPU" bash -c "$cond_cmd" >> "$LOG" 2>&1 < /dev/null
        rc=$?
        if [ "$rc" -ne 0 ]; then
            record_stage "$COND_STAGE_NAME" "failed"
            abort_workload "stage_failed" "conditional full-val generation ($CHOSEN_ARM) exited with code $rc" "$COND_STAGE_NAME"
        fi
        cond_analysis_cmd="$PY scripts/analyze_response_errors.py --gold $VAL_INPUT --arm conditional=$CHOSEN_OUTPUT --arm t4=$T4_VAL_OUT --arm d4v2=$D4V2_VAL_OUT --main conditional --base d4v2 --output $COND_ANALYSIS"
        log "stage $COND_STAGE_NAME analysis: $cond_analysis_cmd"
        CUDA_VISIBLE_DEVICES="$GPU" bash -c "$cond_analysis_cmd" >> "$LOG" 2>&1 < /dev/null
        rc=$?
        if [ "$rc" -ne 0 ]; then
            record_stage "$COND_STAGE_NAME" "failed"
            abort_workload "stage_failed" "conditional error analysis exited with code $rc" "$COND_STAGE_NAME"
        fi
        val_lines="$(wc -l < "$ROOT/$VAL_INPUT" | tr -d ' ')"
        COND_ARM_SPEC="$CHOSEN_OUTPUT:$val_lines"
        COND_ANALYSIS_DONE="$COND_ANALYSIS"
        touch "$STAGES_DIR/$COND_STAGE_NAME.done"
        record_stage "$COND_STAGE_NAME" "done"
        log "stage $COND_STAGE_NAME finished ($CHOSEN_ARM -> $CHOSEN_OUTPUT)"
    fi
fi

# ----------------------------------------------------------------- finalize --
# The supervisor reads these two files when they exist, so a runtime-only arm
# (the conditional one) is audited exactly like the static ones.
: > "$ARMS_EFFECTIVE"
for entry in $ARMS_SPEC; do
    printf '%s\n' "$entry" >> "$ARMS_EFFECTIVE"
done
[ -n "$COND_ARM_SPEC" ] && printf '%s\n' "$COND_ARM_SPEC" >> "$ARMS_EFFECTIVE"
: > "$DONE_EFFECTIVE"
for entry in $DONE_FILES; do
    printf '%s\n' "$entry" >> "$DONE_EFFECTIVE"
done
[ -n "$COND_ANALYSIS_DONE" ] && printf '%s\n' "$COND_ANALYSIS_DONE" >> "$DONE_EFFECTIVE"

arms_failed=0
while IFS= read -r entry; do
    [ -n "$entry" ] || continue
    path="${entry%%:*}"
    want="${entry##*:}"
    [ -n "$path" ] || continue
    case "$want" in
        '' | *[!0-9]*) want=0 ;;
    esac
    lines=0
    ok="false"
    note=""
    if [ ! -s "$ROOT/$path" ]; then
        note="missing or empty"
    else
        lines="$(wc -l < "$ROOT/$path" | tr -d ' ')"
        if [ "$lines" -lt "$want" ]; then
            note="only $lines of $want lines"
        elif ! last_line_ok "$ROOT/$path"; then
            note="last line is not valid JSON"
        else
            ok="true"
        fi
    fi
    [ -n "$ARMS_JSON" ] && ARMS_JSON="$ARMS_JSON, "
    ARMS_JSON="$ARMS_JSON\"$(json_escape "$path")\": {\"lines\": $lines, \"expected\": $want, \"ok\": $ok, \"note\": \"$(json_escape "$note")\"}"
    if [ "$ok" != "true" ]; then
        arms_failed=$((arms_failed + 1))
        log "arm incomplete: $path ($note)"
    fi
done < "$ARMS_EFFECTIVE"

while IFS= read -r entry; do
    [ -n "$entry" ] || continue
    if [ -s "$ROOT/$entry" ]; then
        ok="true"
    else
        ok="false"
    fi
    [ -n "$DONE_JSON" ] && DONE_JSON="$DONE_JSON, "
    DONE_JSON="$DONE_JSON\"$(json_escape "$entry")\": $ok"
    if [ "$ok" != "true" ]; then
        arms_failed=$((arms_failed + 1))
        log "done file missing: $entry"
    fi
done < "$DONE_EFFECTIVE"

stop_gpu_sampler

if [ "$arms_failed" -gt 0 ]; then
    abort_workload "arms_incomplete" "$arms_failed artifact(s) failed the line-count/JSON audit (see $STATUS_JSON)" ""
fi

write_status_json "ok" "all stages finished and every audited artifact is complete"
touch "$ROOT/WORKLOAD_DONE"
say "WORKLOAD_DONE"
exit 0
