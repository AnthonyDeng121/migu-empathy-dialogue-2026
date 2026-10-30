#!/bin/bash
# Push the local workload to the remote 4090 instance and verify both sides.
#
# Runs on the local/WSL side. It builds the upload manifest, pushes four
# parallel rsync streams (the link only reaches its ~700 KB/s aggregate when
# parallelised), re-checks every SHA-256 on the remote host, and then kicks
# off a detached base-model pull there. Nothing here needs a GPU.
#
# Environment:
#   UPLOAD_HOST  remote host, domain or IP (required)
#   UPLOAD_PORT  ssh port (default 22)
#   UPLOAD_USER  ssh user (default root)
#   REMOTE_ROOT  project directory on the remote host (default /root/migu/repo)
#   PASSFILE     executable ssh askpass script (either this or UPLOAD_KEY)
#   UPLOAD_KEY   private key path (either this or PASSFILE)
#   UPLOAD_BASE  0 (default): pull the base model on the remote via hf-mirror
#                1: rsync the local weights/hf/Qwen2.5-1.5B-Instruct instead
#   LOCAL_ROOT   local project directory (default: two levels above this script)
#   SSH_BIN / RSYNC_BIN / PYTHON_BIN  binary overrides (dry-run harness only)
#
# Target host (AutoDL, single RTX 4090 24G, PyTorch 2.5.1 image, Python 3.12,
# Ubuntu 22.04, 30G system disk): its conda lives in /root/miniconda3 and a
# non-interactive ssh session does not source conda, so every remote python
# call goes through the explicit PY_CANDIDATES list below instead of relying
# on PATH.
set -u

die() { echo "ERROR: $*" >&2; exit 1; }
ts() { date '+%F %T'; }
log() { echo "[upload $(ts)] $*"; }

: "${UPLOAD_HOST:?set UPLOAD_HOST to the remote host (domain or IP)}"
UPLOAD_PORT="${UPLOAD_PORT:-22}"
UPLOAD_USER="${UPLOAD_USER:-root}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/migu/repo}"
UPLOAD_BASE="${UPLOAD_BASE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"
SSH_BIN="${SSH_BIN:-ssh}"
MANIFEST_NAME="MANIFEST.sha256"
MANIFEST="$LOCAL_ROOT/$MANIFEST_NAME"
LOG_DIR="$LOCAL_ROOT/logs"
START_TS=$(date +%s)

[ -n "$PYTHON_BIN" ] || die "no python3/python found to build the manifest"
[ -d "$LOCAL_ROOT" ] || die "local root not found: $LOCAL_ROOT"
command -v "$RSYNC_BIN" > /dev/null 2>&1 || die "rsync not found: $RSYNC_BIN"
command -v "$SSH_BIN" > /dev/null 2>&1 || die "ssh not found: $SSH_BIN"
case "$UPLOAD_BASE" in
    0 | 1) ;;
    *) die "UPLOAD_BASE must be 0 or 1, got: $UPLOAD_BASE" ;;
esac

# --- ssh plumbing -----------------------------------------------------------
# The gateway drops idle connections every couple of minutes, so every remote
# call is a short one and everything long-lived is started with setsid nohup.
if [ -n "${PASSFILE:-}" ] && [ -n "${UPLOAD_KEY:-}" ]; then
    die "set either PASSFILE or UPLOAD_KEY, not both"
fi
SSH_OPTS=(-p "$UPLOAD_PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=15)
if [ -n "${UPLOAD_KEY:-}" ]; then
    [ -f "$UPLOAD_KEY" ] || die "UPLOAD_KEY not found: $UPLOAD_KEY"
    SSH_OPTS+=(-i "$UPLOAD_KEY" -o IdentitiesOnly=yes)
fi
if [ -n "${PASSFILE:-}" ]; then
    [ -f "$PASSFILE" ] || die "PASSFILE not found: $PASSFILE"
    [ -x "$PASSFILE" ] || die "PASSFILE must be executable (chmod +x): $PASSFILE"
    export SSH_ASKPASS="$PASSFILE"
    export SSH_ASKPASS_REQUIRE=force
fi
SSH_LAUNCHER=()
command -v setsid > /dev/null 2>&1 && SSH_LAUNCHER=(setsid)
TARGET="$UPLOAD_USER@$UPLOAD_HOST"

# rsync needs the same options as a single -e string; quote every token so a
# path with spaces survives the trip through /bin/sh.
RSH_CMD="$SSH_BIN"
for opt in "${SSH_OPTS[@]}"; do
    RSH_CMD+=" $(printf '%q' "$opt")"
done
command -v setsid > /dev/null 2>&1 && RSH_CMD="setsid $RSH_CMD"

ssh_run() { "${SSH_LAUNCHER[@]}" "$SSH_BIN" "${SSH_OPTS[@]}" "$TARGET" "$@"; }

human() {
    awk -v b="$1" 'BEGIN {
        if (b >= 1073741824) printf "%.2fG", b / 1073741824;
        else if (b >= 1048576) printf "%.1fM", b / 1048576;
        else if (b >= 1024) printf "%.0fK", b / 1024;
        else printf "%d", b;
    }'
}

stream_bytes() { du -cb "$@" 2> /dev/null | tail -1 | cut -f1; }

# --- step 1: manifest -------------------------------------------------------
cd "$LOCAL_ROOT" || die "cannot cd to $LOCAL_ROOT"
mkdir -p "$LOG_DIR" || die "cannot create $LOG_DIR"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

log "step 1/6: building $MANIFEST_NAME"
manifest_summary="$("$PYTHON_BIN" "$SCRIPT_DIR/build_upload_manifest.py" --root "$LOCAL_ROOT" --output "$MANIFEST")" ||
    die "manifest build failed"
echo "$manifest_summary" | sed 's/^/    /'
file_count="$(sed -n 's/^PASS files=\([0-9]*\) .*/\1/p' <<< "$manifest_summary")"
total_bytes="$(sed -n 's/^PASS .* total_bytes=\([0-9]*\).*/\1/p' <<< "$manifest_summary")"
[ -n "$file_count" ] && [ -n "$total_bytes" ] || die "could not parse the manifest summary"
[ -s "$MANIFEST" ] || die "manifest was not written: $MANIFEST"

# --- step 2: remote preparation --------------------------------------------
log "step 2/6: preparing $TARGET:$REMOTE_ROOT"
ssh_run "mkdir -p '$REMOTE_ROOT' '$REMOTE_ROOT/data/processed/b1_full_v1' '$REMOTE_ROOT/results/response_gen' '$REMOTE_ROOT/scripts/remote' '$REMOTE_ROOT/logs'" ||
    die "ssh mkdir failed (host reachable? credentials right?)"
if ! ssh_run "command -v rsync > /dev/null 2>&1"; then
    log "rsync is missing on the remote host, installing it"
    ssh_run "apt-get update -qq && apt-get install -y -qq rsync" ||
        die "could not install rsync on the remote host"
fi
remote_rsync_path="$(ssh_run "command -v rsync")" || die "cannot run commands on the remote host"
[ -n "$remote_rsync_path" ] || die "rsync is not available on the remote host"
log "remote rsync: $remote_rsync_path"

# --- step 3: four parallel rsync streams -----------------------------------
REPO_SOURCES=(scripts src configs tests docs reports README.md .github)
REPO_SOURCES+=(requirements*.txt)
ADAPTER_SOURCES=(weights/t4_longweight_2ep weights/response_sft_d4_2ep weights/README.md)
VAL_SOURCES=(data/processed/b1_full_v1/response_val.jsonl data/processed/b1_full_v1/SHA256SUMS data/processed/b1_full_v1/manifest.json)
PRED_SOURCES=(results/response_gen/*.jsonl results/response_gen/*.json)

validate_sources() {
    local label="$1"
    shift
    local src
    for src in "$@"; do
        [ -e "$src" ] || die "stream $label source missing: $src"
    done
}
validate_sources a_repo_tree "${REPO_SOURCES[@]}"
validate_sources b_adapters "${ADAPTER_SOURCES[@]}"
validate_sources c_val_data "${VAL_SOURCES[@]}"
validate_sources d_predictions "${PRED_SOURCES[@]}"

# Each stream logs to logs/<name>.log and records "<rc> <seconds>" on exit, so
# a failure is visible after the parallel wait instead of vanishing with the
# background subshell. --relative is what keeps multi-component sources
# (weights/t4_longweight_2ep, data/processed/b1_full_v1/response_val.jsonl)
# under their own directories on the remote side; without it rsync would drop
# the parent path and land the adapters directly in $REMOTE_ROOT.
run_stream() {
    local name="$1" dest="$2"
    shift 2
    local start rc
    start=$(date +%s)
    {
        echo "=== $(ts) stream $name ==="
        echo "dest: $dest"
        echo "sources: $*"
    } > "$LOG_DIR/${name}.log"
    "$RSYNC_BIN" --archive --relative --compress --progress \
        --exclude '__pycache__' --exclude '*.pyc' \
        -e "$RSH_CMD" "$@" "$dest" >> "$LOG_DIR/${name}.log" 2>&1
    rc=$?
    printf '%s %s\n' "$rc" "$(( $(date +%s) - start ))" > "$TMP_DIR/$name.status"
    exit "$rc"
}

STREAM_NAMES=()
STREAM_PIDS=()
launch_stream() {
    local name="$1" dest="$2"
    shift 2
    STREAM_NAMES+=("$name")
    stream_bytes "$@" > "$TMP_DIR/$name.bytes"
    run_stream "$name" "$dest" "$@" &
    STREAM_PIDS+=($!)
}

log "step 3/6: pushing four parallel rsync streams (~$(human "$total_bytes") total)"
launch_stream a_repo_tree "$TARGET:$REMOTE_ROOT/" "${REPO_SOURCES[@]}"
launch_stream b_adapters "$TARGET:$REMOTE_ROOT/" "${ADAPTER_SOURCES[@]}"
launch_stream c_val_data "$TARGET:$REMOTE_ROOT/" "${VAL_SOURCES[@]}"
launch_stream d_predictions "$TARGET:$REMOTE_ROOT/" "${PRED_SOURCES[@]}"

stream_failed=0
for index in "${!STREAM_PIDS[@]}"; do
    if ! wait "${STREAM_PIDS[$index]}"; then
        stream_failed=1
    fi
done
if [ "$stream_failed" -ne 0 ]; then
    for name in "${STREAM_NAMES[@]}"; do
        tail -5 "$LOG_DIR/${name}.log" | sed "s/^/    [$name] /"
    done
    die "one or more rsync streams failed; see $LOG_DIR/<stream>.log"
fi
log "all four streams finished"

# --- step 4: manifest upload + double-sided sha256 check --------------------
log "step 4/6: uploading $MANIFEST_NAME and verifying both sides"
"$RSYNC_BIN" --archive -e "$RSH_CMD" "$MANIFEST_NAME" "$TARGET:$REMOTE_ROOT/" ||
    die "could not upload $MANIFEST_NAME"

write_remote_helper() {
    local name="$1"
    ssh_run "cat > '$REMOTE_ROOT/scripts/remote/$name'" < "$TMP_DIR/$name" ||
        die "could not upload the $name helper to the remote host"
}

cat > "$TMP_DIR/upload_verify_remote.sh" <<'EOF'
#!/bin/bash
# Remote half of the double-sided SHA-256 check.
#
# usage: upload_verify_remote.sh REMOTE_ROOT
# Exits non-zero when any manifest file is missing or corrupt, so the caller
# fails the upload instead of starting a run on damaged inputs.
set -u

ROOT="${1:?usage: upload_verify_remote.sh REMOTE_ROOT}"
cd "$ROOT" || exit 1
sha256sum -c MANIFEST.sha256 > upload_verify.log 2>&1
rc=$?
cat upload_verify.log
if [ "$rc" -ne 0 ]; then
    echo "UPLOAD_VERIFY_FAILED rc=$rc (full log: $ROOT/upload_verify.log)" >&2
fi
exit "$rc"
EOF

write_remote_helper upload_verify_remote.sh
verify_output="$(ssh_run "bash '$REMOTE_ROOT/scripts/remote/upload_verify_remote.sh' '$REMOTE_ROOT'")"
verify_rc=$?
if [ "$verify_rc" -eq 0 ]; then
    verify_state="UPLOAD_VERIFIED ($file_count files, sha256sum -c on the remote host)"
    log "$verify_state"
else
    verify_state="UPLOAD_VERIFY_FAILED"
    echo "$verify_output" | grep -E "FAILED|No such file|improperly formatted|Is a directory" |
        sed 's/^/    /' || true
    die "remote sha256 verification failed (log: $REMOTE_ROOT/upload_verify.log)"
fi

# --- step 5: base model -----------------------------------------------------
log "step 5/6: base model (UPLOAD_BASE=$UPLOAD_BASE)"
base_state="unknown"
base_ready_output="$(ssh_run "cat '$REMOTE_ROOT/base_ready' 2>/dev/null" || true)"

cat > "$TMP_DIR/base_ready_remote.sh" <<'EOF'
#!/bin/bash
# Verify the base model on this host and write the base_ready marker.
#
# usage: base_ready_remote.sh REMOTE_ROOT [MODEL_DIR]
# The expected SHA-256 is the known-good Qwen2.5-1.5B-Instruct
# model.safetensors; anything else means the snapshot is unusable and must
# not be declared ready.
set -u

ROOT="${1:?usage: base_ready_remote.sh REMOTE_ROOT [MODEL_DIR]}"
MODEL_DIR="${2:-$ROOT/weights/hf/Qwen2.5-1.5B-Instruct}"
EXPECTED="${EXPECTED_SHA256:-dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee}"
READY="${READY_MARKER:-$ROOT/base_ready}"
model="$MODEL_DIR/model.safetensors"

if [ ! -f "$model" ]; then
    echo "BASE_MODEL_MISSING $model" >&2
    exit 1
fi
actual="$(sha256sum "$model" | awk '{print $1}')"
if [ "$actual" != "$EXPECTED" ]; then
    echo "BASE_HASH_MISMATCH model=$model" >&2
    echo "  actual=$actual" >&2
    echo "  expected=$EXPECTED" >&2
    exit 2
fi
bytes="$(stat -c %s "$model" 2>/dev/null || wc -c < "$model")"
{
    echo "base_sha256=$actual"
    echo "bytes=$bytes"
    echo "model=$model"
    echo "verified_at=$(date -u '+%F %T %Z')"
} > "$READY"
echo "BASE_READY base_sha256=$actual bytes=$bytes model=$model"
EOF

if [ "$UPLOAD_BASE" = "1" ]; then
    LOCAL_BASE="$LOCAL_ROOT/weights/hf/Qwen2.5-1.5B-Instruct"
    [ -f "$LOCAL_BASE/model.safetensors" ] ||
        die "UPLOAD_BASE=1 but $LOCAL_BASE/model.safetensors is missing"
    write_remote_helper base_ready_remote.sh
    if [ -n "$base_ready_output" ]; then
        base_state="already ready on the remote host"
    else
        log "uploading the local base model (~3.09 GB; re-run to resume if the gateway drops)"
        "$RSYNC_BIN" --archive --compress --progress --partial \
            --exclude '.cache/' --exclude '__pycache__' \
            -e "$RSH_CMD" "$LOCAL_BASE" "$TARGET:$REMOTE_ROOT/weights/hf/" ||
            die "base model upload failed (re-run to resume)"
        if ssh_run "bash '$REMOTE_ROOT/scripts/remote/base_ready_remote.sh' '$REMOTE_ROOT'"; then
            base_state="uploaded and hash-verified"
        else
            die "base model hash mismatch after upload"
        fi
    fi
else
    cat > "$TMP_DIR/base_pull_remote.sh" <<'EOF'
#!/bin/bash
# Detached base-model pull for the remote workload host.
#
# hf-mirror is reachable (if slow) from the instance, so the 3.09 GB base is
# pulled here instead of uploaded. Every attempt is hash-verified before the
# run may start: an incomplete or silently corrupt snapshot must never look
# like a ready base. A failed attempt is deleted and retried.
#
# Environment (defaults match the unattended run):
#   REMOTE_ROOT   project directory on this host (default /root/migu/repo)
#   PY_CANDIDATES whitespace-separated python candidates, tried in order
#                 (default "/opt/conda/bin/python /root/miniconda3/bin/python python3 python")
#   MODEL_ID      huggingface repo (default Qwen/Qwen2.5-1.5B-Instruct)
#   HF_ENDPOINT   mirror (default https://hf-mirror.com)
#   MAX_ATTEMPTS  pull attempts (default 5)
#   RETRY_SLEEP   seconds between attempts (default 60)
#
# The candidate order matters on AutoDL: conda sits in /root/miniconda3 and a
# non-interactive ssh session never sources conda, so PATH alone would only
# ever find python3/python.
#
# HF_HUB_DISABLE_XET=1: hf-mirror does not serve the xet CAS protocol (the
# reconstruction endpoint answers 401), so force the plain HTTP download path.
set -u

REMOTE_ROOT="${REMOTE_ROOT:-/root/migu/repo}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
RETRY_SLEEP="${RETRY_SLEEP:-60}"
PY_CANDIDATES="${PY_CANDIDATES:-/opt/conda/bin/python /root/miniconda3/bin/python python3 python}"
MODEL_DIR="${MODEL_DIR:-$REMOTE_ROOT/weights/hf/Qwen2.5-1.5B-Instruct}"
READY_MARKER="${READY_MARKER:-$REMOTE_ROOT/base_ready}"
CHECKER="$REMOTE_ROOT/scripts/remote/base_ready_remote.sh"
export HF_ENDPOINT
export HF_HUB_DISABLE_XET=1

ts() { date '+%F %T'; }

case "$MODEL_DIR" in
    "" | "/" | "$REMOTE_ROOT")
        echo "$(ts) ERROR: refusing to run with unsafe MODEL_DIR: $MODEL_DIR" >&2
        exit 1
        ;;
esac

if [ -f "$READY_MARKER" ]; then
    echo "$(ts) base already ready: $(tr '\n' ' ' < "$READY_MARKER")"
    exit 0
fi

PY=""
for candidate in $PY_CANDIDATES; do
    if command -v "$candidate" > /dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "$(ts) ERROR: no usable python among: $PY_CANDIDATES" >&2
    exit 1
fi
echo "$(ts) using python $PY ($("$PY" -V 2>&1)) endpoint $HF_ENDPOINT"

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo "$(ts) pull attempt $attempt/$MAX_ATTEMPTS model=$MODEL_ID"
    if "$PY" -c 'import huggingface_hub' 2> /dev/null; then
        :
    else
        echo "$(ts) huggingface_hub missing, installing"
        "$PY" -m pip install --no-input huggingface_hub 2>&1 ||
            "$PY" -m pip install --no-input --break-system-packages huggingface_hub 2>&1 ||
            "$PY" -m pip install --no-input --user huggingface_hub 2>&1 ||
            true
    fi
    if "$PY" - "$MODEL_ID" "$MODEL_DIR" <<'PYEOF'
import sys

from huggingface_hub import snapshot_download

repo, local_dir = sys.argv[1], sys.argv[2]
path = snapshot_download(repo, local_dir=local_dir)
print("snapshot at", path)
PYEOF
    then
        if bash "$CHECKER" "$REMOTE_ROOT" "$MODEL_DIR"; then
            echo "$(ts) BASE_READY"
            exit 0
        fi
        echo "$(ts) snapshot failed verification, deleting $MODEL_DIR"
        rm -rf "$MODEL_DIR"
    else
        echo "$(ts) snapshot_download failed"
        rm -rf "$MODEL_DIR"
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -le "$MAX_ATTEMPTS" ]; then
        echo "$(ts) sleeping $RETRY_SLEEP s before the next attempt"
        sleep "$RETRY_SLEEP"
    fi
done
echo "$(ts) BASE_PULL_FAILED after $MAX_ATTEMPTS attempts"
exit 1
EOF

    if [ -n "$base_ready_output" ]; then
        base_state="already ready on the remote host ($(echo "$base_ready_output" | tr '\n' ' '))"
    else
        write_remote_helper base_ready_remote.sh
        write_remote_helper base_pull_remote.sh
        ssh_run "setsid nohup bash '$REMOTE_ROOT/scripts/remote/base_pull_remote.sh' > '$REMOTE_ROOT/base_pull.log' 2>&1 < /dev/null & echo BASE_PULL_STARTED" ||
            die "could not start the detached base pull"
        base_state="pull started in the background (log: $REMOTE_ROOT/base_pull.log)"
    fi
fi

# --- step 6: summary --------------------------------------------------------
elapsed=$(( $(date +%s) - START_TS ))
echo
echo "============================== upload summary =============================="
printf '%-18s %-8s %-9s %12s\n' "stream" "status" "seconds" "bytes"
stream_bytes_total=0
stream_seconds_max=0
for name in "${STREAM_NAMES[@]}"; do
    read -r rc secs < "$TMP_DIR/$name.status"
    bytes="$(cat "$TMP_DIR/$name.bytes")"
    stream_bytes_total=$((stream_bytes_total + ${bytes:-0}))
    [ "${secs:-0}" -gt "$stream_seconds_max" ] && stream_seconds_max="$secs"
    if [ "$rc" = "0" ]; then status="ok"; else status="FAIL"; fi
    printf '%-18s %-8s %-9s %12s\n' "$name" "$status" "$secs" "$(human "$bytes")"
done
printf '%-18s %-8s %-9s %12s\n' "4 streams" "" "$stream_seconds_max" "$(human "$stream_bytes_total")"
echo "manifest:  $file_count files, $(human "$total_bytes") bytes -> $REMOTE_ROOT/$MANIFEST_NAME"
echo "verify:    $verify_state"
echo "base:      $base_state"
echo "elapsed:   ${elapsed}s wall clock (the four streams ran in parallel)"
echo "local logs: $LOG_DIR/<stream>.log"
if [ "$UPLOAD_BASE" != "1" ] && [ -z "$base_ready_output" ]; then
    echo "base check: ssh -p $UPLOAD_PORT $TARGET 'cat $REMOTE_ROOT/base_ready' (or tail $REMOTE_ROOT/base_pull.log)"
fi
echo "next:      bash $REMOTE_ROOT/scripts/remote/launch_workload.sh"
echo "==========================================================================="
