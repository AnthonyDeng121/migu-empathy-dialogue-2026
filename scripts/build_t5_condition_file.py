#!/usr/bin/env python3
"""按 id 合并 C 的情感预测与画像预测，产出 T5 的「情感+画像」双条件文件。

T5 的 emotion 臂只消费情感标签，eprofile 臂同时消费情感标签与画像；
``run_response_inference.py --condition-fields both`` 要求每行同时具备
str 型 emotion_label 与 dict 型 user_profile，故按 input 的 id 从两个来源
各取一列逐行合并，缺任一来源即整体失败，不写半成品文件。

用法：
    python scripts/build_t5_condition_file.py \
        --emotions results/response_gen/c5_conditions_val_full.jsonl \
        --profiles reports/c_cycle2_profile_v1_predictions.jsonl \
        --input results/response_gen/t2_selection_input.jsonl \
        --output results/response_gen/t5_conditions_selection_both.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

PROFILE_KEYS = ("personality_traits", "interests", "style")


def fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def require_file(path: Path) -> None:
    if not path.is_file():
        fail(f"文件不存在: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each row must be an object")
            rows.append(row)
    return rows


def load_emotions(path: Path) -> dict[str, str]:
    emotions: dict[str, str] = {}
    for row in read_jsonl(path):
        row_id = row.get("id")
        label = row.get("emotion_label")
        if not isinstance(row_id, str) or not row_id:
            fail(f"{path}: 存在缺 id 的情绪行")
        if not isinstance(label, str) or not label:
            fail(f"{path}: id {row_id!r} 的 emotion_label 不是非空 str")
        if row_id in emotions:
            fail(f"{path}: id 重复 {row_id!r}")
        emotions[row_id] = label
    if not emotions:
        fail(f"{path}: 情绪文件为空")
    return emotions


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        row_id = row.get("id")
        profile = row.get("user_profile")
        if not isinstance(row_id, str) or not row_id:
            fail(f"{path}: 存在缺 id 的画像行")
        if not isinstance(profile, dict) or not all(
            isinstance(profile.get(key), list) for key in PROFILE_KEYS
        ):
            fail(
                f"{path}: id {row_id!r} 的 user_profile 不是含 "
                f"{list(PROFILE_KEYS)} 三个 list 的 dict"
            )
        if row_id in profiles:
            fail(f"{path}: id 重复 {row_id!r}")
        profiles[row_id] = profile
    if not profiles:
        fail(f"{path}: 画像文件为空")
    return profiles


def load_input_ids(path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        row_id = row.get("sample_id")
        if not isinstance(row_id, str) or not row_id:
            fail(f"{path}: 存在缺 sample_id 的输入行")
        if row_id in seen:
            fail(f"{path}: sample_id 重复 {row_id!r}")
        seen.add(row_id)
        ids.append(row_id)
    if not ids:
        fail(f"{path}: 输入文件为空")
    return ids


def describe_ids(ids: list[str]) -> str:
    shown = ", ".join(ids[:10])
    return shown if len(ids) <= 10 else f"{shown}, ...（共 {len(ids)} 个）"


def build_conditions(
    input_ids: list[str],
    emotions: dict[str, str],
    profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    missing_emotion = [row_id for row_id in input_ids if row_id not in emotions]
    missing_profile = [row_id for row_id in input_ids if row_id not in profiles]
    if missing_emotion:
        fail(f"情绪来源缺少 {len(missing_emotion)} 个 id: {describe_ids(missing_emotion)}")
    if missing_profile:
        fail(f"画像来源缺少 {len(missing_profile)} 个 id: {describe_ids(missing_profile)}")
    return [
        {"id": row_id, "emotion_label": emotions[row_id], "user_profile": profiles[row_id]}
        for row_id in input_ids
    ]


def write_conditions(path: Path, conditions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in conditions:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.emotions, args.profiles, args.input):
        require_file(path)
    emotions = load_emotions(args.emotions)
    profiles = load_profiles(args.profiles)
    input_ids = load_input_ids(args.input)
    conditions = build_conditions(input_ids, emotions, profiles)
    write_conditions(args.output, conditions)
    digest = sha256_file(args.output)
    print(f"PASS: {len(conditions)} conditions -> {args.output} (sha256 {digest})", flush=True)
    return {"rows": len(conditions), "output": str(args.output), "sha256": digest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emotions", type=Path, required=True,
        help="C 情感预测 jsonl（每行 id + emotion_label）",
    )
    parser.add_argument(
        "--profiles", type=Path, required=True,
        help="C 画像预测 jsonl（每行 id + user_profile）",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="筛选集或 val 的推理输入 jsonl（id 字段为 sample_id）",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
