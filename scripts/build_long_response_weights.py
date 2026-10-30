#!/usr/bin/env python3
"""给 B1 response_train.jsonl 的每条样本计算 long-sample 权重（T4 单变量）。

规则（先定后跑，reports/D_CYCLE2_T4_LONG_RESPONSE.md 第二节）：
  gold 回复字符数 > 200 -> 2.0，其余 -> 1.0
输出 JSONL，每行在原始样本基础上加 "sample_weight" 字段，行序与原文件一致。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl, write_jsonl  # noqa: E402

LONG_THRESHOLD_CHARS = 200
LONG_WEIGHT = 2.0
BASE_WEIGHT = 1.0


def sample_weight(response_text: str) -> float:
    return LONG_WEIGHT if len(response_text.strip()) > LONG_THRESHOLD_CHARS else BASE_WEIGHT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="B1 response_train.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSONL（含 sample_weight）")
    parser.add_argument("--long-threshold", type=int, default=LONG_THRESHOLD_CHARS)
    parser.add_argument("--long-weight", type=float, default=LONG_WEIGHT)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    if not rows:
        raise SystemExit(f"no rows in {args.input}")

    weighted: list[dict] = []
    long_count = 0
    for row in rows:
        weight = args.long_weight if len(row["response_text"].strip()) > args.long_threshold else BASE_WEIGHT
        if weight != BASE_WEIGHT:
            long_count += 1
        new_row = dict(row)
        new_row["sample_weight"] = weight
        weighted.append(new_row)

    write_jsonl(args.output, weighted)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(weighted),
        "long_threshold_chars": args.long_threshold,
        "long_weight": args.long_weight,
        "long_samples": long_count,
        "long_share": round(long_count / len(weighted), 6),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
