#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summary(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values) if values else 0,
        "mean": round(statistics.fmean(values), 3) if values else 0.0,
        "p50": round(percentile(values, 0.5), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": max(values) if values else 0,
    }


def normalized(text: str) -> str:
    return "".join(ch.lower() for ch in unicodedata.normalize("NFKC", text) if not ch.isspace())


def context_text(turns: list[dict[str, Any]], target_id: int) -> str:
    return "\n".join(f"{turn['role']}:{turn['content']}" for turn in turns if turn["turn_id"] <= target_id)


def digest(text: str) -> str:
    return hashlib.sha256(normalized(text).encode("utf-8")).hexdigest()


def validate_conversation(row: dict[str, Any], split: str) -> list[str]:
    errors: list[str] = []
    if set(row) != {"conversation_id", "turns", "prediction_points"}:
        errors.append("unexpected conversation fields")
    turns = row.get("turns", [])
    ids = [turn.get("turn_id") for turn in turns if isinstance(turn, dict)]
    if ids != list(range(1, len(ids) + 1)):
        errors.append("turn_id is not contiguous from 1")
    for index, turn in enumerate(turns):
        expected = "user" if index % 2 == 0 else "assistant"
        if turn.get("role") != expected:
            errors.append("turn roles do not alternate starting with user")
            break
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append("empty/non-string turn content")
    id_set = set(ids)
    for point in row.get("prediction_points", []):
        if point.get("target_user_turn_id") not in id_set or point.get("assistant_response_turn_id") not in id_set:
            errors.append("prediction point references a missing turn")
            continue
        user_turn = turns[point["target_user_turn_id"] - 1]
        assistant_turn = turns[point["assistant_response_turn_id"] - 1]
        if user_turn["role"] != "user" or assistant_turn["role"] != "assistant" or assistant_turn["turn_id"] != user_turn["turn_id"] + 1:
            errors.append("prediction point does not reference adjacent user/assistant turns")
        target = point.get("target", {})
        if target.get("response_text") != assistant_turn.get("content"):
            errors.append("target response differs from referenced assistant turn")
    return errors


def analyze(path: Path, split: str) -> tuple[dict[str, Any], dict[str, set[str]]]:
    rows = read_jsonl(path)
    conversation_ids: list[str] = []
    turn_counts: list[int] = []
    context_lengths: list[int] = []
    response_lengths: list[int] = []
    emotion = Counter()
    profiles = {name: Counter() for name in ("personality_traits", "interests", "style")}
    missing = Counter()
    contexts = Counter()
    responses = Counter()
    last_user_texts = Counter()
    errors = Counter()
    hashes = {"conversation_ids": set(), "contexts": set(), "responses": set(), "last_user_texts": set()}
    prediction_count = 0
    control_chars = 0
    replacement_chars = 0
    for row in rows:
        conversation_id = row.get("conversation_id", "")
        conversation_ids.append(conversation_id)
        hashes["conversation_ids"].add(conversation_id)
        turns = row.get("turns", [])
        turn_counts.append(len(turns))
        for turn in turns:
            content = turn.get("content", "")
            replacement_chars += content.count("\ufffd")
            control_chars += sum(unicodedata.category(ch) == "Cc" and ch not in "\n\r\t" for ch in content)
        for message in validate_conversation(row, split):
            errors[message] += 1
        by_id = {turn["turn_id"]: turn for turn in turns}
        for point in row.get("prediction_points", []):
            prediction_count += 1
            target_id = point["target_user_turn_id"]
            context = context_text(turns, target_id)
            target = point["target"]
            response = target.get("response_text", "")
            last_user = by_id[target_id]["content"]
            context_lengths.append(len(context))
            response_lengths.append(len(response))
            context_hash, response_hash, user_hash = digest(context), digest(response), digest(last_user)
            contexts[context_hash] += 1
            responses[response_hash] += 1
            last_user_texts[user_hash] += 1
            hashes["contexts"].add(context_hash)
            hashes["responses"].add(response_hash)
            hashes["last_user_texts"].add(user_hash)
            label = target.get("emotion_label")
            emotion[label if label is not None else "<missing>"] += 1
            profile = target.get("user_profile")
            if not isinstance(profile, dict):
                missing["user_profile"] += 1
                profile = {}
            for dimension, counts in profiles.items():
                values = profile.get(dimension)
                if values is None:
                    missing[f"user_profile.{dimension}"] += 1
                elif not values:
                    counts["<empty>"] += 1
                else:
                    counts.update(values)
            if not target.get("memory_refs"):
                missing["memory_refs_empty"] += 1
    report = {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "conversations": len(rows),
        "unique_conversation_ids": len(set(conversation_ids)),
        "users": None,
        "users_note": "No user_id field exists in the supplied schema; user count cannot be measured.",
        "prediction_samples": prediction_count,
        "turns_per_conversation": summary(turn_counts),
        "context_length_chars": summary(context_lengths),
        "response_length_chars": summary(response_lengths),
        "emotion_distribution": dict(emotion.most_common()),
        "profile_distribution": {name: dict(values.most_common()) for name, values in profiles.items()},
        "missing_or_empty": dict(missing),
        "duplicate_context_groups": sum(count > 1 for count in contexts.values()),
        "duplicate_context_samples": sum(count for count in contexts.values() if count > 1),
        "duplicate_response_groups": sum(count > 1 for count in responses.values()),
        "duplicate_last_user_groups": sum(count > 1 for count in last_user_texts.values()),
        "replacement_characters": replacement_chars,
        "unexpected_control_characters": control_chars,
        "validation_errors": dict(errors),
    }
    return report, hashes


def render_markdown(report: dict[str, Any]) -> str:
    train, val, leak = report["splits"]["train"], report["splits"]["val"], report["cross_split_overlap"]
    lines = [
        "# 数据审计报告",
        "",
        "审计脚本按原始 JSONL 只读扫描，固定随机种子为 20261012；本次审计不抽样，也不生成或覆盖原始数据。",
        "",
        "## 核心规模",
        "",
        "| 指标 | 训练集 | 验证集 |",
        "|---|---:|---:|",
        f"| 会话数 | {train['conversations']} | {val['conversations']} |",
        f"| 预测样本数 | {train['prediction_samples']} | {val['prediction_samples']} |",
        f"| 平均轮数（消息数） | {train['turns_per_conversation']['mean']} | {val['turns_per_conversation']['mean']} |",
        f"| 最大轮数（消息数） | {train['turns_per_conversation']['max']} | {val['turns_per_conversation']['max']} |",
        f"| 平均上下文字符数 | {train['context_length_chars']['mean']} | {val['context_length_chars']['mean']} |",
        f"| 平均回复字符数 | {train['response_length_chars']['mean']} | {val['response_length_chars']['mean']} |",
        "",
        "官方数据没有 `user_id` 字段，因此无法可靠统计用户数或检测同一用户跨集合泄漏；`conversation_id` 可用于会话级隔离。",
        "",
        "## 交叉集合泄漏检查",
        "",
        f"- 相同 conversation_id：{leak['conversation_ids']}。",
        f"- 完全相同规范化上下文：{leak['contexts']}。",
        f"- 完全相同最后一条用户消息：{leak['last_user_texts']}。",
        f"- 完全相同标准回复：{leak['responses']}（回复复用不等同于上下文泄漏，需单独解释）。",
        "",
        "## 标签与质量",
        "",
        f"训练集情感分布：`{json.dumps(train['emotion_distribution'], ensure_ascii=False)}`。",
        "",
        f"验证集情感分布：`{json.dumps(val['emotion_distribution'], ensure_ascii=False)}`。",
        "",
        f"训练集画像分布：`{json.dumps(train['profile_distribution'], ensure_ascii=False)}`。",
        "",
        f"验证集画像分布：`{json.dumps(val['profile_distribution'], ensure_ascii=False)}`。",
        "",
        f"训练集结构错误：`{json.dumps(train['validation_errors'], ensure_ascii=False)}`；验证集结构错误：`{json.dumps(val['validation_errors'], ensure_ascii=False)}`。",
        f"Unicode 替换字符：训练 {train['replacement_characters']}，验证 {val['replacement_characters']}；异常控制字符：训练 {train['unexpected_control_characters']}，验证 {val['unexpected_control_characters']}。",
        "",
        "## 结论与处理约束",
        "",
        "- 训练单位应展开为 prediction point，但划分必须以 conversation_id 为组，禁止把同一会话的不同预测点拆到训练和验证两侧。",
        "- 空画像数组是合法标签，不应当作缺失值；画像指标应按三个维度分别比较后取平均。",
        "- memory_refs 依赖未提供的 memory bank。测试输入也没有 memory bank 字段，当前 baseline 输出空数组；正式方案必须向官方确认隐藏评测是否另行提供记忆库。",
        "- 高相似上下文检测当前给出完全一致与最后用户句完全一致两档。语义近重复需要在建模阶段使用向量或 MinHash 分块检查，且不能用于修改官方原始数据。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official train/validation JSONL without modifying source files")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    train, train_hashes = analyze(args.train, "train")
    val, val_hashes = analyze(args.val, "val")
    overlap = {name: len(train_hashes[name] & val_hashes[name]) for name in train_hashes}
    report = {"audit_version": 1, "seed": 20261012, "splits": {"train": train, "val": val}, "cross_split_overlap": overlap}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"train_samples": train["prediction_samples"], "val_samples": val["prediction_samples"], "overlap": overlap}, ensure_ascii=False))


if __name__ == "__main__":
    main()

