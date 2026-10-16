#!/usr/bin/env python3
"""对回复候选的全量 val 预测做格式与内容错误分析。

用法：
    python scripts/analyze_response_errors.py \
        --gold data/processed/b1_full_v1/response_val.jsonl \
        --arm d2=results/response_gen/d2_1p5b_val_full.jsonl \
        --arm original=results/response_gen/original_1p5b_val_full.jsonl \
        --arm retrieval=results/baseline/submission.jsonl \
        --bridge data/processed/val_eval_input.jsonl \
        --output reports/d_day4_response_error_analysis.json
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
from src.response.error_analysis import format_issues, length_bucket


def pp_key(row_id: str) -> tuple[str, int] | None:
    """把 pp:["conv",target_turn,...] 形式的 id 解析成 (conversation_id, target_user_turn_id)。"""
    if not row_id.startswith("pp:"):
        return None
    parsed = json.loads(row_id[3:])
    return (parsed[0], parsed[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="B1 response_val.jsonl（含 gold response_text）")
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=PATH", help="预测臂，可多次")
    parser.add_argument("--bridge", type=Path, help="val_eval_input.jsonl（eval 系 id 与 B1 sample_id 的桥接）")
    parser.add_argument("--main", help="head-to-head 主臂名（默认第一个 --arm）")
    parser.add_argument("--base", help="head-to-head 对照臂名（默认第二个 --arm）")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # eval 系 id -> (conversation_id, target_user_turn_id)
    eval_to_key = {}
    if args.bridge:
        for row in read_jsonl(args.bridge):
            eval_to_key[row["id"]] = (row["conversation_id"], row["target_user_turn_id"])

    def normalize(row_id: str) -> tuple[str, int]:
        key = pp_key(row_id)
        if key is not None:
            return key
        if row_id in eval_to_key:
            return eval_to_key[row_id]
        raise SystemExit(f"无法解析预测 id: {row_id!r}（缺 --bridge？）")

    gold = {(row["conversation_id"], row["target_user_turn_id"]): row["response_text"] for row in read_jsonl(args.gold)}
    gold_ids = {(row["conversation_id"], row["target_user_turn_id"]): row["sample_id"] for row in read_jsonl(args.gold)}
    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = {normalize(row["id"]): row["response_text"] for row in read_jsonl(Path(path))}

    report: dict = {"samples": len(gold), "gold_length_buckets": {}, "arms": {}}
    bucket_counts: dict[str, int] = {}
    for text in gold.values():
        bucket = length_bucket(len(text.strip()))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    report["gold_length_buckets"] = dict(sorted(bucket_counts.items()))

    per_row: dict[tuple[str, int], dict[str, dict]] = {sid: {} for sid in gold}
    for name, preds in arms.items():
        missing = [sid for sid in gold if sid not in preds]
        if missing:
            raise SystemExit(f"arm {name}: {len(missing)} predictions missing (e.g. {missing[:3]})")
        issue_counts: dict[str, int] = {}
        bucket_metrics: dict[str, list[float]] = {}
        rows = []
        for sid, gtext in gold.items():
            ptext = preds[sid]
            issues = format_issues(ptext)
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            bucket = length_bucket(len(gtext.strip()))
            b4, rl = bleu4(gtext, ptext), rouge_l(gtext, ptext)
            bucket_metrics.setdefault(bucket, []).append(rl)
            rows.append({
                "sample_id": gold_ids[sid],
                "gold_len": len(gtext.strip()),
                "pred_len": len(ptext.strip()),
                "gold_bucket": bucket,
                "issues": issues,
                "bleu4": b4,
                "rouge_l": rl,
            })
            per_row[sid][name] = {"rouge_l": rl, "bleu4": b4, "issues": issues, "pred_len": len(ptext.strip())}
        report["arms"][name] = {
            "rows": rows,
            "issue_counts": dict(sorted(issue_counts.items(), key=lambda kv: -kv[1])),
            "rouge_l_by_gold_bucket": {b: round(statistics.fmean(v), 4) for b, v in sorted(bucket_metrics.items())},
            "mean_rouge_l": round(statistics.fmean(r["rouge_l"] for r in rows), 4),
            "mean_bleu4": round(statistics.fmean(r["bleu4"] for r in rows), 4),
        }

    names = list(arms)
    if len(names) >= 2:
        main_name = args.main or names[0]
        base_name = args.base or names[1]
        if main_name not in arms or base_name not in arms:
            raise SystemExit(f"--main/--base must be arm names among {names}")
        if main_name == base_name:
            raise SystemExit("--main and --base must differ")
        wins = losses = ties = 0
        regressions = []
        for sid in gold:
            m = per_row[sid][main_name]["rouge_l"]
            b = per_row[sid][base_name]["rouge_l"]
            if m > b + 1e-9:
                wins += 1
            elif m < b - 1e-9:
                losses += 1
                if b - m > 0.15:
                    regressions.append({"sample_id": gold_ids[sid], f"{main_name}_rouge_l": round(m, 4), f"{base_name}_rouge_l": round(b, 4), "gold_len": len(gold[sid].strip())})
            else:
                ties += 1
        report["head_to_head"] = {
            "main": main_name, "base": base_name,
            "wins": wins, "losses": losses, "ties": ties,
            "regressions_gt_0p15": sorted(regressions, key=lambda r: r[f"{base_name}_rouge_l"] - r[f"{main_name}_rouge_l"], reverse=True)[:20],
        }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "samples": report["samples"],
        "gold_length_buckets": report["gold_length_buckets"],
        "arms": {name: {"issues": arm["issue_counts"], "mean_rouge_l": arm["mean_rouge_l"], "mean_bleu4": arm["mean_bleu4"], "rouge_l_by_gold_bucket": arm["rouge_l_by_gold_bucket"]} for name, arm in report["arms"].items()},
        "head_to_head": {k: v for k, v in report.get("head_to_head", {}).items() if k != "regressions_gt_0p15"} or None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
