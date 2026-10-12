#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl, write_jsonl
from src.data.schema import validate_prediction, validate_test_row
from src.response.retrieval import RetrievalBaseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic character n-gram retrieval baseline")
    parser.add_argument("test_file", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "baseline.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train = read_jsonl(args.train)
    test = read_jsonl(args.test_file)
    for row in test:
        validate_test_row(row)
    started = time.perf_counter()
    model = RetrievalBaseline(train, config["retrieval_ngram"])
    built = time.perf_counter()
    predictions = [model.predict(row, config["candidate_limit"]) for row in test]
    for sample, prediction in zip(test, predictions):
        validate_prediction(prediction, sample["id"])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.result_dir / "submission.jsonl", predictions)
    report = {
        "version": "retrieval_char_ngram_v1",
        "samples": len(predictions),
        "index_build_seconds": round(built - started, 6),
        "inference_seconds": round(time.perf_counter() - built, 6),
        "seed": config["seed"],
    }
    (args.result_dir / "performance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {args.result_dir / 'submission.jsonl'}")


if __name__ == "__main__":
    main()

