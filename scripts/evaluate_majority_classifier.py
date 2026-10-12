#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.classification.majority import MajorityClassifier
from src.common.io import read_jsonl
from src.data.schema import PROFILE_LABELS, validate_classification_labels


def targets(rows: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for conversation in rows:
        points = conversation.get("prediction_points")
        if not isinstance(points, list):
            raise ValueError("conversation must contain prediction_points")
        for point in points:
            target = point.get("target") if isinstance(point, dict) else None
            if not isinstance(target, dict):
                raise ValueError("prediction point must contain a target object")
            if not {"emotion_label", "user_profile"} <= target.keys():
                raise ValueError("validation target must contain emotion_label and user_profile")
            validate_classification_labels(target["emotion_label"], target["user_profile"])
            yield target


def evaluate(model: MajorityClassifier, validation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    profile_correct: Counter[str] = Counter()
    emotion_correct = 0
    samples = 0
    prediction = model.predict()

    for target in targets(validation_rows):
        samples += 1
        emotion_correct += prediction["emotion_label"] == target["emotion_label"]
        for dimension in PROFILE_LABELS:
            profile_correct[dimension] += (
                set(prediction["user_profile"][dimension])
                == set(target["user_profile"][dimension])
            )

    if samples == 0:
        raise ValueError("validation data contains no labeled prediction points")
    profile_accuracy = {
        dimension: round(profile_correct[dimension] / samples, 6)
        for dimension in PROFILE_LABELS
    }
    return {
        "version": "majority_classifier_v1",
        "train_samples": model.training_samples,
        "validation_samples": samples,
        "majority": model.predict(),
        "majority_support": model.majority_support,
        "metrics": {
            "emotion_accuracy": round(emotion_correct / samples, 6),
            "profile_exact_match_accuracy": profile_accuracy,
            "profile_macro_accuracy": round(
                sum(profile_accuracy.values()) / len(profile_accuracy), 6
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# 多数类分类 baseline",
        "",
        f"训练样本：{report['train_samples']}；验证样本：{report['validation_samples']}。",
        "",
        (
            f"情感多数类：`{report['majority']['emotion_label']}`，"
            f"支持数 {report['majority_support']['emotion_label']}；"
            f"验证集 Accuracy {metrics['emotion_accuracy']:.2%}。"
        ),
        "",
        "画像按完整标签集合计算 exact-match，空数组按合法类别处理：",
        "",
        "| 维度 | 多数类预测 | 训练支持数 | 验证 exact-match accuracy |",
        "|---|---|---:|---:|",
    ]
    for dimension in PROFILE_LABELS:
        labels = report["majority"]["user_profile"][dimension]
        rendered = ", ".join(f"`{label}`" for label in labels) if labels else "`[]`"
        support = report["majority_support"]["user_profile"][dimension]
        accuracy = metrics["profile_exact_match_accuracy"][dimension]
        lines.append(f"| {dimension} | {rendered} | {support} | {accuracy:.2%} |")
    lines.extend(
        [
            "",
            f"画像三维平均 exact-match accuracy：{metrics['profile_macro_accuracy']:.2%}。",
            "",
            "使用方法：",
            "",
            "```bash",
            "python scripts/evaluate_majority_classifier.py \\",
            "  --train data/raw/训练-验证-数据集/train/train_public.jsonl \\",
            "  --val data/raw/训练-验证-数据集/val/val_public.jsonl \\",
            "  --output-json reports/classification_baseline.json \\",
            "  --output-md reports/CLASSIFICATION_BASELINE.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate majority-class emotion/profile baseline"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    model = MajorityClassifier(read_jsonl(args.train))
    report = evaluate(model, read_jsonl(args.val))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"PASS: {args.output_json}")


if __name__ == "__main__":
    main()
