#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
PYTHON_BIN=${PYTHON_BIN:-python3}
export PYTHON_BIN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export INFERENCE_BACKEND=integrated
export INTEGRATED_CONFIG=${INTEGRATED_CONFIG:-$PROJECT_DIR/configs/integrated.json}
"$PYTHON_BIN" scripts/cloud_preflight.py --config "$INTEGRATED_CONFIG" --verify-hashes
bash start.sh "${1:-data/raw/test_inference_data.jsonl}" "${2:-results/cloud_smoke_$(date +%Y%m%d_%H%M%S)}"
