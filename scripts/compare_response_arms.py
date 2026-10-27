#!/usr/bin/env python3
"""两个回复臂在同一筛选集上的同设备比较（字符级诊断指标 + 逐条配对）。

用法：
    python scripts/compare_response_arms.py \\
        --input results/response_gen/t2_selection_input.jsonl \\
        --arm t3=results/response_gen/t3_unconditional.jsonl \\
        --arm d4v2=results/response_gen/t3_d4v2_off_samedevice.jsonl \\
        --output results/response_gen/t3_arms_comparison_same_device.json

假设：两个臂均在同一张 GPU 上以相同权重族、相同解码参数生成（同设备纪律）；
本脚本只做本地字符级诊断，不冒充官方指标。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl
from src.evaluation.metrics import bleu4, rouge_l
from src.response.error_analysis import length_bucket


def _nonws_len(text: str) -> int:
    return len("".join(text.split()))


def load_arm(path: Path, name: str) -> dict[str, str]:
    rows = read_jsonl(path)
    seen: dict[str, str] = {}
    for row in rows:
        row_id = row.get("id")
        text = row.get("response_text", "")
        if not row_id:
            raise SystemExit(f"{name}: 存在缺 id 的行（{path}）")
        if row_id in seen:
            raise SystemExit(f"{name}: id 重复 {row_id!r}（{path}）")
        seen[row_id] = text
    return seen


def arm_metrics(gold: dict[str, str], arm: dict[str, str]) -> dict:
    bleus, rouges, buckets = [], [], {}
    for row_id, ref in gold.items():
        hyp = arm[row_id]
        bleus.append(bleu4(ref, hyp))
        rouges.append(rouge_l(ref, hyp))
        bucket = length_bucket(_nonws_len(hyp))
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return {
        "bleu4_mean": statistics.fmean(bleus),
        "rouge_l_mean": statistics.fmean(rouges),
        "mean_nonws_len": statistics.fmean(_nonws_len(arm[r]) for r in gold),
        "length_buckets": dict(sorted(buckets.items())),
        "empty_responses": sum(1 for r in gold if not arm[r].strip()),
    }


def paired(gold: dict[str, str], main: dict[str, str], base: dict[str, str]) -> dict:
    rouge_deltas, bleu_deltas = [], []
    wins = losses = ties = identical = 0
    longer = shorter = same_len = 0
    for row_id in gold:
        m, b = main[row_id], base[row_id]
        rd = rouge_l(gold[row_id], m) - rouge_l(gold[row_id], b)
        bd = bleu4(gold[row_id], m) - bleu4(gold[row_id], b)
        rouge_deltas.append(rd)
        bleu_deltas.append(bd)
        if rd > 0:
            wins += 1
        elif rd < 0:
            losses += 1
        else:
            ties += 1
        if m == b:
            identical += 1
        lm, lb = _nonws_len(m), _nonws_len(b)
        if lm > lb:
            longer += 1
        elif lm < lb:
            shorter += 1
        else:
            same_len += 1
    return {
        "rouge_l_delta_median": statistics.median(rouge_deltas),
        "rouge_l_delta_p25": statistics.quantiles(rouge_deltas, n=4)[0],
        "rouge_l_delta_p75": statistics.quantiles(rouge_deltas, n=4)[2],
        "bleu4_delta_mean": statistics.fmean(bleu_deltas),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "identical_text": identical,
        "length_longer": longer,
        "length_shorter": shorter,
        "length_same": same_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="筛选集（含 gold response_text）")
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=PATH", help="预测臂，可多次")
    parser.add_argument("--main", help="head-to-head 主臂名（默认第一个 --arm）")
    parser.add_argument("--base", help="head-to-head 对照臂名（默认第二个 --arm）")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    gold_rows = read_jsonl(args.input)
    gold = {row["sample_id"]: row["response_text"] for row in gold_rows}
    gold_ids = set(gold)

    arms: dict[str, dict[str, str]] = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"--arm 须为 NAME=PATH 形式: {spec!r}")
        arm = load_arm(Path(path), name)
        if set(arm) != gold_ids:
            raise SystemExit(f"{name}: id 集合与筛选集不一致（缺 {len(gold_ids - set(arm))}，多 {len(set(arm) - gold_ids)}）")
        arms[name] = arm

    main_name = args.main or next(iter(arms))
    base_name = args.base or (list(arms)[1] if len(arms) > 1 else None)
    if main_name not in arms:
        raise SystemExit(f"主臂 {main_name!r} 不在 --arm 列表")

    report = {
        "input": str(args.input),
        "input_count": len(gold_rows),
        "arms": {name: arm_metrics(gold, arm) for name, arm in arms.items()},
        "head_to_head": None,
    }
    if base_name is not None:
        if base_name not in arms:
            raise SystemExit(f"对照臂 {base_name!r} 不在 --arm 列表")
        report["head_to_head"] = {"main": main_name, "base": base_name, **paired(gold, arms[main_name], arms[base_name])}

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"written: {args.output}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
