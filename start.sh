#!/usr/bin/env sh
set -eu

if [ "${1:-}" = "--help" ] || [ "$#" -ne 2 ]; then
  echo "Usage: $0 TEST_FILE RESULT_DIR"
  [ "${1:-}" = "--help" ] && exit 0
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TRAIN_FILE=${BASELINE_TRAIN_FILE:-/opt/data/train.jsonl}
exec python3 "$SCRIPT_DIR/scripts/run_baseline.py" "$1" "$2" --train "$TRAIN_FILE"

