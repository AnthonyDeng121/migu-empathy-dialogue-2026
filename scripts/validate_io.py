#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl
from src.data.schema import validate_prediction, validate_test_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate official test input and/or submission output")
    parser.add_argument("--input", type=Path, help="Official test JSONL")
    parser.add_argument("--submission", type=Path, help="Prediction JSONL")
    args = parser.parse_args()
    if not args.input and not args.submission:
        parser.error("provide --input or --submission")
    inputs = read_jsonl(args.input) if args.input else []
    for row in inputs:
        validate_test_row(row)
    outputs = read_jsonl(args.submission) if args.submission else []
    if inputs and outputs and len(inputs) != len(outputs):
        raise SystemExit(f"row count mismatch: input={len(inputs)}, submission={len(outputs)}")
    for index, row in enumerate(outputs):
        validate_prediction(row, inputs[index]["id"] if inputs else None)
    print(f"PASS input={len(inputs)} submission={len(outputs)}")


if __name__ == "__main__":
    main()

