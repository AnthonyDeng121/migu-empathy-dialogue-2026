#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl
from src.evaluation.metrics import accuracy, bleu4, profile_dimension_accuracy, rouge_l


def targets(path: Path) -> dict[tuple[str, int], dict]:
    result = {}
    for conversation in read_jsonl(path):
        for point in conversation["prediction_points"]:
            result[(conversation["conversation_id"], point["target_user_turn_id"])] = point["target"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions aligned to an input JSONL with gold labels")
    parser.add_argument("--gold-dataset", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    gold_map = targets(args.gold_dataset)
    inputs, predictions = read_jsonl(args.input), read_jsonl(args.submission)
    if len(inputs) != len(predictions):
        raise SystemExit("input/submission row count mismatch")
    gold = [gold_map[(row["conversation_id"], row["target_user_turn_id"])] for row in inputs]
    result = {
        "samples": len(gold),
        "emotion_accuracy": accuracy((x["emotion_label"] for x in gold), (x["emotion_label"] for x in predictions)),
        "profile_dimension_accuracy": statistics.fmean(profile_dimension_accuracy(g["user_profile"], p["user_profile"]) for g, p in zip(gold, predictions)),
        "bleu4_char": statistics.fmean(bleu4(g["response_text"], p["response_text"]) for g, p in zip(gold, predictions)),
        "rouge_l_char_f1": statistics.fmean(rouge_l(g["response_text"], p["response_text"]) for g, p in zip(gold, predictions)),
        "warning": "Official tokenization, smoothing, BERTScore model and metric weights are not published in supplied files; these scores are diagnostic only."
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

