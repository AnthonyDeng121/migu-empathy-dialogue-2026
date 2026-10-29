#!/usr/bin/env python3
"""T5 条件臂全量生成的预注册判定器（判定线先定后跑）。

六个筛选集臂与两个全量 val 臂跑完后，由本脚本决定是否触发第九个臂
（条件触发的全量 val 生成）。判定线写死为 CLI 默认值：条件臂相对同
adapter 关闭臂的 ROUGE-L 均值提升 >= --delta-rouge，且逐条配对
wins-losses >= --min-win-margin；eligible 臂中按提升降序、margin
降序、固定臂序（t4_emotion < t4_eprofile < d4v2_emotion <
d4v2_eprofile）取第一个。缺比较文件或结构不符即以退出码 2 失败并
指明是哪个文件。

用法：
    python scripts/remote/decide_conditional_fullval.py \
        --comparison results/response_gen/t5_t4_emotion_vs_off.json \
        --comparison results/response_gen/t5_t4_eprofile_vs_off.json \
        --comparison results/response_gen/t5_d4v2_emotion_vs_off.json \
        --comparison results/response_gen/t5_d4v2_eprofile_vs_off.json \
        --output results/response_gen/t5_conditional_fullval_decision.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

DECISION_VERSION = "t5_conditional_fullval_v1"
DEFAULT_DELTA_ROUGE = 0.005
DEFAULT_MIN_WIN_MARGIN = 20
DEFAULT_T4_ADAPTER = "weights/t4_longweight_2ep"
DEFAULT_D4V2_ADAPTER = "weights/response_sft_d4_2ep"
DEFAULT_C5_CONDITIONS = "results/response_gen/c5_conditions_val_full.jsonl"
DEFAULT_BOTH_CONDITIONS = "results/response_gen/t5_conditions_val_both.jsonl"
DEFAULT_OUT_DIR = "results/response_gen"
ARM_ORDER = ("t4_emotion", "t4_eprofile", "d4v2_emotion", "d4v2_eprofile")


def fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def arm_from_filename(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("t5_"):
        fail(f"{path}: 比较文件名须以 t5_ 开头")
    body = stem[len("t5_"):]
    head, marker, tail = body.partition("_vs_off")
    if not marker or tail not in ("", "_newcard"):
        fail(f"{path}: 比较文件名须为 t5_<arm>_vs_off[_newcard].json")
    if head not in ARM_ORDER:
        fail(f"{path}: 无法识别的条件臂 {head!r}（可选 {list(ARM_ORDER)}）")
    return head


def load_comparison(path: Path, arm: str) -> dict[str, float]:
    if not path.is_file():
        fail(f"比较文件缺失: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"{path}: 不是合法 JSON（{exc.msg}）")
    if not isinstance(data, dict):
        fail(f"{path}: 顶层必须是对象")
    key = arm.split("_", 1)[1]
    arms = data.get("arms")
    if not isinstance(arms, dict):
        fail(f"{path}: 缺少 arms 对象")
    for name in (key, "off"):
        metrics = arms.get(name)
        if not isinstance(metrics, dict) or not isinstance(
            metrics.get("rouge_l_mean"), (int, float)
        ):
            fail(f"{path}: arms 缺少 {name!r} 的数值 rouge_l_mean")
    head_to_head = data.get("head_to_head")
    if not isinstance(head_to_head, dict):
        fail(f"{path}: 缺少 head_to_head 对象")
    for field in ("wins", "losses"):
        value = head_to_head.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            fail(f"{path}: head_to_head 缺少整数 {field}")
    if head_to_head.get("main", key) != key or head_to_head.get("base", "off") != "off":
        fail(
            f"{path}: head_to_head 的 main/base 应为 {key!r}/'off'，"
            f"实为 {head_to_head.get('main')!r}/{head_to_head.get('base')!r}"
        )
    return {
        "delta_rouge_l": float(arms[key]["rouge_l_mean"]) - float(arms["off"]["rouge_l_mean"]),
        "win_margin": head_to_head["wins"] - head_to_head["losses"],
    }


def candidate(arm: str, args: argparse.Namespace, measured: dict[str, float]) -> dict[str, Any]:
    adapter = args.t4_adapter if arm.startswith("t4_") else args.d4v2_adapter
    if arm.endswith("_emotion"):
        condition_jsonl, condition_fields = args.c5_conditions, "emotion"
    else:
        condition_jsonl, condition_fields = args.both_conditions, "both"
    delta = measured["delta_rouge_l"]
    margin = measured["win_margin"]
    out_dir = Path(args.out_dir)
    return {
        "arm": arm,
        "adapter": str(adapter),
        "condition_jsonl": str(condition_jsonl),
        "condition_fields": condition_fields,
        "output": (out_dir / f"t5_conditional_{arm}_val_newcard.jsonl").as_posix(),
        "resume_dir": (out_dir / f"t5_conditional_{arm}_val_newcard_progress").as_posix(),
        "delta_rouge_l": delta,
        "win_margin": margin,
        "eligible": delta >= args.delta_rouge and margin >= args.min_win_margin,
    }


def decide(args: argparse.Namespace) -> dict[str, Any]:
    measured: dict[str, dict[str, float]] = {}
    for raw in args.comparison:
        path = Path(raw)
        arm = arm_from_filename(path)
        if arm in measured:
            fail(f"条件臂 {arm!r} 出现多次比较文件（{path}）")
        measured[arm] = load_comparison(path, arm)
    candidates = [
        candidate(arm, args, measured[arm]) for arm in ARM_ORDER if arm in measured
    ]
    eligible = sorted(
        (item for item in candidates if item["eligible"]),
        key=lambda item: (
            -item["delta_rouge_l"],
            -item["win_margin"],
            ARM_ORDER.index(item["arm"]),
        ),
    )
    chosen = eligible[0] if eligible else None
    if chosen is None:
        reason = "no comparison files provided" if not candidates else "no arm cleared the bar"
    else:
        reason = (
            f"{chosen['arm']} cleared the bar "
            f"(delta_rouge_l={chosen['delta_rouge_l']:.6f} >= {args.delta_rouge}, "
            f"win_margin={chosen['win_margin']} >= {args.min_win_margin})"
        )
    return {
        "decision_version": DECISION_VERSION,
        "bar": {"delta_rouge_l": args.delta_rouge, "min_win_margin": args.min_win_margin},
        "candidates": candidates,
        "chosen": chosen,
        "trigger": chosen is not None,
        "reason": reason,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison", action="append", required=True, type=Path,
        help="compare_response_arms.py 产出的比较 JSON，可多次",
    )
    parser.add_argument(
        "--delta-rouge", type=float, default=DEFAULT_DELTA_ROUGE,
        help="条件臂 ROUGE-L 均值提升下限（预注册默认 0.005）",
    )
    parser.add_argument(
        "--min-win-margin", type=int, default=DEFAULT_MIN_WIN_MARGIN,
        help="逐条配对 wins-losses 下限（预注册默认 20）",
    )
    parser.add_argument("--t4-adapter", default=DEFAULT_T4_ADAPTER)
    parser.add_argument("--d4v2-adapter", default=DEFAULT_D4V2_ADAPTER)
    parser.add_argument("--c5-conditions", default=DEFAULT_C5_CONDITIONS)
    parser.add_argument("--both-conditions", default=DEFAULT_BOTH_CONDITIONS)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--output", type=Path, required=True, help="决策 JSON 输出路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision = decide(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    chosen = decision["chosen"]
    if chosen is None:
        print(f"NO_TRIGGER: {decision['reason']}")
    else:
        print(f"TRIGGER: {chosen['arm']} -> {chosen['output']}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
