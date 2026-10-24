#!/usr/bin/env python3
"""T1 B: audit existing files and freeze manifests; CPU/standard library only."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.audit_dataset import summary, validate_conversation
from scripts.b_validate_submission import sha256_file
from scripts.prepare_training_data import sample_id
from src.classification.emotion_encoder import classification_metrics
from src.classification.emotion_profile import (
    EMOTION_LABELS, PROFILE_LABELS_BY_DIMENSION, build_b1_classification_examples,
    profile_metrics, validate_disjoint_splits,
)
from src.common.io import read_jsonl
from src.evaluation.metrics import profile_dimension_accuracy

SEED = 20261015
B1 = Path("data/processed/b1_full_v1")
RAW = Path("data/raw/训练-验证-数据集")
ARCHIVE = Path(".codex_tmp/acceptance/a_day6_final/unpacked")
COMMAND = "python -B scripts/b_t1_profile_diagnostics.py --seed 20261015 --screening-size 160"
RULES = [
    "冻结日期 2026-10-22（Asia/Shanghai），只针对第二周期 B T1；尚无本周期新候选结果。",
    "主指标固定为三个画像维度的集合 exact match 的等权宏平均；同一 B1 验证 ID、完整 2354 条、相同金标、预处理和评测协议下比较空画像、历史头、新候选。数组顺序不影响得分。",
    "本次重算核实空画像基线 0.628292（未舍入值见 JSON）。只在上述同口径下用作门槛；换数据或协议须重算空画像，不搬用此数值。历史报告与新候选若缺少输入/预处理证据，隔离标记，不直接排名。",
    "候选 macro exact match 必须严格超过同集空画像；相等或更低原则上 no-go，保留空画像。比较未舍入值，避免六位小数掩盖持平；非空预测率提高不构成提升。超过空画像只是必要条件，逐维/关键分层退步和稳定性仍需独立复核。",
    "同时报告每维 exact、micro precision/recall/F1、固定标签 macro precision/recall/F1、全局 micro 及跨维宏平均、预测非空率、金标空/非空、长度/情感切片和非空精度。无预测时相关指标为缺失，不能以预测标签个数冒充非空样本率。零分母按 0 计，未支持标签也计入固定标签宏平均。",
    "情感固定报告 Accuracy、16 类 Macro-F1、逐类 precision/recall/F1/support 和混淆矩阵（行金标、列预测），单列 mixed/care；不能仅凭 Accuracy 选型。T1 不另设情感数值硬门槛。",
    "阈值、融合、checkpoint 及早停只能用训练内 conversation-disjoint 校准；验证 2354 条及筛选 160 条均禁止阈值拟合。筛选集只用于固定候选淘汰，不能作为最终提分证明或新盲测。不得事后换主指标、挑样本或按新结果改清单。",
    "建议（B 本次设计值，非既有团队决策）：沿用旧画像 seed=20261015/15% 会话留出；筛选 160 条、每情感约 10 条，按历史长度×三维空非空细分均衡，稀有类不足时按规则补足。依据为复用旧划分与计划规定的 128–200 条预算；不增加其他无依据数值采用阈值。",
    "旧情感编码器已在完整 B1 训练集拟合，旧画像头的校准会话可能被其见过；不能把旧冻结头校准称为全模型未见。新候选应先划分，再只在 fit 学习全部可训练参数；若沿用见过校准的起点，显式披露，公平重训由 C 后续负责。",
    "Day4 历史 Micro-F1>0 阶段交接门槛保留为历史事实，不代替本周期 exact 采用规则；官方评分定义未知，本地结果不等于官方分数或隐藏集提升。T2 会话级配对比较留待下一任务。",
]


def ratio(a, b):
    return a / b if b else 0.0


def length_bucket(turns):
    # Design bins in messages, including target user; no tokenizer dependence.
    return "short_1_5" if turns <= 5 else "medium_6_15" if turns <= 15 else "long_16_plus"


def strata(row):
    return (row["emotion_label"], length_bucket(len(row["context_turns"])),
            "".join("1" if row["user_profile"][d] else "0" for d in PROFILE_LABELS_BY_DIMENSION))


def validate_rows(train, val):
    examples = [build_b1_classification_examples(rows, expected_split=split)
                for rows, split in ((train, "train"), (val, "val"))]
    validate_disjoint_splits(*examples)


def internal_split(train, seed=SEED):
    # Exactly the algorithm in train_profile_heads.py / exact calibration CLI.
    if not train or any(r["source_split"] != "train" for r in train):
        raise ValueError("internal split must contain training rows only")
    if len({r["sample_id"] for r in train}) != len(train):
        raise ValueError("duplicate training sample ID")
    conversations = sorted({r["conversation_id"] for r in train})
    if len(conversations) < 2:
        raise ValueError("at least two training conversations required")
    random.Random(seed).shuffle(conversations)
    calibration = set(conversations[:max(1, round(len(conversations) * 0.15))])
    return [{"sample_id": r["sample_id"], "conversation_id": r["conversation_id"],
             "source_split": "train", "source_file": (B1 / "classification_train.jsonl").as_posix(),
             "subset": "calibration" if r["conversation_id"] in calibration else "fit"}
            for r in sorted(train, key=lambda r: r["sample_id"])]


def screening_manifest(val, size=160, seed=SEED):
    if not 128 <= size <= 200 or len(val) < size:
        raise ValueError("screening size must be 128..200 and not exceed validation")
    if any(r["source_split"] != "val" for r in val):
        raise ValueError("screening uses validation only")
    if len({r["sample_id"] for r in val}) != len(val):
        raise ValueError("duplicate validation sample ID")
    groups = defaultdict(list)
    for row in val:
        groups[strata(row)].append(row)
    rng = random.Random(seed)
    for key in sorted(groups):
        groups[key].sort(key=lambda r: r["sample_id"])
        rng.shuffle(groups[key])
    support = Counter(r["emotion_label"] for r in val)
    # Near-equal class quotas; unavailable quota flows to largest support first.
    labels = sorted(support)
    quota = {label: min(support[label], size // len(labels)) for label in labels}
    while sum(quota.values()) < size:
        for label in sorted(labels, key=lambda x: (-support[x], x)):
            if quota[label] < support[label]:
                quota[label] += 1
                if sum(quota.values()) == size:
                    break
    selected = []
    for label in labels:
        keys = sorted((k for k in groups if k[0] == label), key=lambda k: (len(groups[k]), k))
        positions = Counter()
        count = 0
        while count < quota[label]:
            for key in keys:
                if positions[key] < len(groups[key]):
                    row = groups[key][positions[key]]
                    positions[key] += 1
                    selected.append(row)
                    count += 1
                    if count == quota[label]:
                        break
    return [{"sample_id": r["sample_id"], "conversation_id": r["conversation_id"],
             "source_split": "val", "source_file": (B1 / "classification_val.jsonl").as_posix(),
             "emotion_label": strata(r)[0], "history_bucket": strata(r)[1],
             "profile_nonempty_mask": strata(r)[2]}
            for r in sorted(selected, key=lambda r: r["sample_id"])]


def conversation_coverage(raw_rows, manifest, usable_rows, split):
    """Reconcile raw points, usable manifest entries and actual B1 sample IDs."""
    raw_ids = {r["conversation_id"] for r in raw_rows}
    point_pairs = [(sample_id(r["conversation_id"], p, i), r["conversation_id"])
                   for r in raw_rows for i, p in enumerate(r["prediction_points"], 1)]
    entries = [r for r in manifest if r["source_split"] == split]
    manifest_pairs = [(r["sample_id"], r["conversation_id"]) for r in entries]
    if Counter(point_pairs) != Counter(manifest_pairs):
        raise ValueError(f"{split}: raw prediction points / manifest mismatch")
    usable_pairs = [(r["sample_id"], r["conversation_id"]) for r in entries if r["usable"] is True]
    actual_pairs = [(r["sample_id"], r["conversation_id"]) for r in usable_rows]
    if Counter(usable_pairs) != Counter(actual_pairs):
        raise ValueError(f"{split}: usable=true / actual B1 samples mismatch")
    point_ids = {cid for _, cid in point_pairs}
    usable_ids = {cid for _, cid in usable_pairs}
    no_points = raw_ids - point_ids
    all_unusable = point_ids - usable_ids
    other_missing = (raw_ids - usable_ids) - no_points - all_unusable
    unexpected = usable_ids - raw_ids
    return {
        "raw_records": len(raw_rows), "raw_conversations": len(raw_ids),
        "prediction_point_conversations": len(point_ids),
        "usable_sample_conversations": len(usable_ids),
        "no_prediction_point_conversations": len(no_points),
        "no_prediction_point_conversation_ids": sorted(no_points),
        "all_prediction_points_unusable_conversations": len(all_unusable),
        "all_prediction_points_unusable_conversation_ids": sorted(all_unusable),
        "all_prediction_points_unusable_evidence": [
            {"conversation_id": cid,
             "prediction_points": sum(r["conversation_id"] == cid for r in entries),
             "usable_samples": 0,
             "excluded_points": [{"sample_id": r["sample_id"], "reasons": r["reasons"]}
                                 for r in entries if r["conversation_id"] == cid]}
            for cid in sorted(all_unusable)],
        "other_missing_conversations": len(other_missing),
        "other_missing_conversation_ids": sorted(other_missing),
        "usable_conversations_not_in_raw": len(unexpected),
        "usable_conversation_ids_not_in_raw": sorted(unexpected),
        "raw_manifest_prediction_point_pairs_match": True,
        "manifest_usable_actual_sample_pairs_match": True,
        "difference_reconciles": len(raw_ids) - len(usable_ids) == len(no_points) + len(all_unusable) + len(other_missing) - len(unexpected),
    }


def dataset_stats(rows):
    n = len(rows)
    dimensions = {}
    for dim, labels in PROFILE_LABELS_BY_DIMENSION.items():
        counts = Counter(label for r in rows for label in r["user_profile"][dim])
        empty = sum(not r["user_profile"][dim] for r in rows)
        dimensions[dim] = {"empty_count": empty, "empty_rate": ratio(empty, n),
                           "nonempty_count": n-empty, "nonempty_rate": ratio(n-empty, n),
                           "missing_or_null_count": 0,
                           "label_support": {label: counts[label] for label in labels}}
    lengths = {}
    for bucket in ("short_1_5", "medium_6_15", "long_16_plus"):
        subset = [r for r in rows if length_bucket(len(r["context_turns"])) == bucket]
        lengths[bucket] = {"samples": len(subset),
            "all_empty_count": sum(not any(r["user_profile"].values()) for r in subset),
            "profile_nonempty_rate": {d: ratio(sum(bool(r["user_profile"][d]) for r in subset), len(subset))
                                     for d in PROFILE_LABELS_BY_DIMENSION}}
    return {"samples": n, "unique_sample_ids": len({r["sample_id"] for r in rows}),
            "usable_sample_conversations": len({r["conversation_id"] for r in rows}),
            "duplicate_sample_ids": n-len({r["sample_id"] for r in rows}),
            "missing_or_null_required_fields": {}, "b1_adapter_validation": "passed all rows",
            "emotion_support": {k: sum(r["emotion_label"] == k for r in rows) for k in EMOTION_LABELS},
            "profile": dimensions, "profile_mask_support": dict(sorted(Counter(strata(r)[2] for r in rows).items())),
            "history_messages": summary([len(r["context_turns"]) for r in rows]),
            "history_content_chars": summary([sum(len(t["content"]) for t in r["context_turns"]) for r in rows]),
            "history_profile": lengths}


def prf(tp, fp, fn):
    return {"precision": ratio(tp, tp+fp), "recall": ratio(tp, tp+fn), "f1": ratio(2*tp, 2*tp+fp+fn)}


def extended_profile_metrics(gold, predicted):
    result = profile_metrics(gold, predicted)  # reuse established exact/micro F1
    totals = Counter()
    for dim, labels in PROFILE_LABELS_BY_DIMENSION.items():
        a, b = [set(p[dim]) for p in gold], [set(p[dim]) for p in predicted]
        per_label = {}
        for label in labels:
            tp = sum(label in x and label in y for x, y in zip(a, b))
            fp = sum(label not in x and label in y for x, y in zip(a, b))
            fn = sum(label in x and label not in y for x, y in zip(a, b))
            per_label[label] = dict(prf(tp, fp, fn), support=tp+fn)
            totals.update(tp=tp, fp=fp, fn=fn)
        tp = sum(len(x & y) for x, y in zip(a, b))
        fp = sum(len(y-x) for x, y in zip(a, b))
        fn = sum(len(x-y) for x, y in zip(a, b))
        result["per_dimension"][dim].update({
            "micro": prf(tp, fp, fn),
            "macro": {m: sum(v[m] for v in per_label.values())/len(labels) for m in ("precision", "recall", "f1")},
            "per_label": per_label,
            "predicted_nonempty_rate": ratio(sum(bool(x) for x in b), len(b)),
            "gold_empty_exact": ratio(sum(not x and x == y for x, y in zip(a, b)), sum(not x for x in a)),
            "gold_nonempty_exact": ratio(sum(bool(x) and x == y for x, y in zip(a, b)), sum(bool(x) for x in a)),
            "nonempty_prediction_exact_precision": ratio(sum(bool(y) and x == y for x, y in zip(a, b)), sum(bool(y) for y in b)),
        })
    result["global_micro"] = prf(totals["tp"], totals["fp"], totals["fn"])
    result["dimension_macro"] = {kind: {m: sum(v[kind][m] for v in result["per_dimension"].values())/3
                                        for m in ("precision", "recall", "f1")} for kind in ("micro", "macro")}
    result["macro_exact_unrounded"] = sum(
        set(a[d]) == set(b[d]) for a, b in zip(gold, predicted)
        for d in PROFILE_LABELS_BY_DIMENSION) / (3 * len(gold))
    # Cross-check against the other established evaluator without float summation drift.
    if abs(result["macro_exact_unrounded"] - sum(profile_dimension_accuracy(a, b) for a, b in zip(gold, predicted))/len(gold)) > 1e-10:
        raise ValueError("profile metric definitions disagree")
    return result


def replay(rows, predictions):
    ids = [r.get("id") for r in predictions]
    expected = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("prediction IDs missing, extra or duplicated")
    index = {r["id"]: r for r in predictions}
    aligned = [index[k] for k in expected]
    result = {"samples": len(rows), "input_order_identical": ids == expected,
              "emotion": classification_metrics([r["emotion_label"] for r in rows],
                   [r["emotion_label"] for r in aligned], EMOTION_LABELS),
              "profile": extended_profile_metrics([r["user_profile"] for r in rows], [r["user_profile"] for r in aligned])}
    slices = defaultdict(list)
    for i, r in enumerate(rows):
        emotion, length, mask = strata(r)
        for key in ("history:"+length, "profile_mask:"+mask, "emotion:"+emotion):
            slices[key].append(i)
    result["slices"] = {}
    for key, indices in sorted(slices.items()):
        gold = [rows[i]["user_profile"] for i in indices]
        pred = [aligned[i]["user_profile"] for i in indices]
        emotion = classification_metrics([rows[i]["emotion_label"] for i in indices],
                                        [aligned[i]["emotion_label"] for i in indices], EMOTION_LABELS)
        result["slices"][key] = {"samples": len(indices), "profile": profile_metrics(gold, pred),
                                "emotion": {k: emotion[k] for k in ("accuracy", "macro_f1", "per_class")}}
    return result


def verify_b1_hashes(root=PROJECT):
    manifest = json.loads((root / B1 / "manifest.json").read_text(encoding="utf-8"))
    checks = {}
    for name, info in manifest["outputs"].items():
        p = root / B1 / name
        actual = sha256_file(p)
        count = len(read_jsonl(p)) if p.suffix == ".jsonl" else 1
        checks[name] = {"sha256": actual, "records": count, "matches_manifest": actual == info["sha256"] and count == info["record_count"]}
        if not checks[name]["matches_manifest"]:
            raise ValueError(f"B1 manifest mismatch: {name}")
    for split in ("train", "val"):
        p = root / RAW / split / f"{split}_public.jsonl"
        if sha256_file(p) != manifest["inputs"][split]["sha256"]:
            raise ValueError(f"raw {split} manifest hash mismatch")
    return checks


def historical_reports(train_hash, val_hash):
    profiles, emotions, inventory = [], [], []
    def add_profile(name, path, node, data, metrics, preprocessing):
        compatible = data.get("train_sha256") == train_hash and data.get("validation_sha256") == val_hash and data.get("validation_samples") == 2354
        profiles.append({"name": name, "source": path, "node": node, "data_hashes_match": compatible,
                         "preprocessing": preprocessing, "metrics": metrics,
                         "comparability": "reported same B1 + exact-set definition; report-level only; input length/objective varies, not a controlled single-variable ranking" if compatible else "unconfirmed or partial; do not rank",
                         "prediction_file": None, "missing": ["per-row profile predictions", "precision/recall", "label macro F1", "nonempty sample rate", "history/empty slices"]})
    for filename in ("c_day3_emotion_profile.json", "c_day3_retrained_metrics.json", "c_day4_profile_unweighted_metrics.json", "c_day4_profile_long_context_metrics.json", "c_day5_emotion_longtail075_metrics.json", "emotion_encoder_baseline.json", "b_day3_b2_eval.json"):
        path = Path("reports") / filename
        if not path.exists():
            inventory.append({"path": path.as_posix(), "status": "missing"})
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        inventory.append({"path": path.as_posix(), "status": "available", "sha256": sha256_file(path)})
        if filename == "c_day3_emotion_profile.json":
            profiles.append({"name": "Day3 dry-run", "source": path.as_posix(), "metrics": d["metrics"], "samples": 32,
                             "comparability": "32-row exploratory subset; cannot rank against full 2354", "prediction_file": None})
        elif filename in ("c_day3_retrained_metrics.json", "c_day4_profile_unweighted_metrics.json"):
            add_profile(filename, path.as_posix(), "profile.validation", d["data"], d["profile"]["validation"], "speaker lines, left truncation, max_length=256; training-only micro-F1 calibration")
            if filename == "c_day3_retrained_metrics.json":
                profiles.append({"name": "Day3 exact check", "source": path.as_posix(), "metrics": d["profile"]["training_only_threshold_calibration_exact_match_check"], "comparability": "report summary only; no per-dimension predictions; not independently replayable"})
        elif filename == "c_day4_profile_long_context_metrics.json":
            for key in ("weighted_512_training", "weighted_512_exact_calibration", "unweighted_512_training", "unweighted_512_exact_calibration"):
                v = d[key]
                add_profile(key, path.as_posix(), key, v["data"], v.get("profile", {}).get("validation", v.get("validation")),
                            "speaker lines, left truncation, max_length=512; " + ("exact calibration" if "exact_calibration" in key else "micro-F1 calibration"))
        elif filename == "c_day5_emotion_longtail075_metrics.json":
            emotions.append({"source": path.as_posix(), "data_hashes_match": d["data"]["train_sha256"] == train_hash and d["data"]["validation_sha256"] == val_hash,
                             "Day3": d["comparison"]["previous_day3_candidate"], "Day5": d["metrics"]["after"],
                             "preprocessing": "speaker lines; left truncation; max_length=256; seed=20261015; weight exponent .5 vs .75"})
    return {"inventory": inventory, "profile": profiles, "emotion": emotions}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def markdown_report(r):
    lines = ["# B T1：历史画像诊断与固定实验清单", "",
        "日期：2026-10-22（Asia/Shanghai）。范围：第二周期 T1 / B 数据评测。状态：本地交付待 C 审查，非团队 Done。",
        "工作区开工干净，分支 feature/b-eval-tooling，HEAD de8f0da；未找到根目录、父目录或工作区内 AGENTS.md。未训练、下载、使用 GPU 或新推理；未修改 A/C/D 实现、官方数据、历史报告与权重，未提交、推送。", "",
        "## 结论与证据边界", "",
        "空画像 macro exact match 重算为 **0.628292**，可在本次同集同协议下作为数值门槛。历史画像头的全量报告均未证明超过它；Day4 不加权 512 exact 校准与它持平，第二周期采用结论仍为 no-go / 保留空画像。旧报告的 Micro-F1 阶段交接通过与采用是不同结论。",
        "2354 条固定验证集曾被历史实验反复使用，不能称为新盲测。Day3/Day5 情感可在本机归档逐条回放，旧画像头只有聚合报告，未补造逐条预测。历史训练的冻结编码器见过训练内校准样本，新的会话划分不能消除既有表示学习暴露。", "",
        "## 数据与字段", "",
        "B1 实际路径为 data/processed/b1_full_v1/，原始数据为 data/raw/训练-验证-数据集/{train,val}/{train,val}_public.jsonl。B1 分类字段为 sample_id、conversation_id、target_user_turn_id、source_split、context_turns、context_text、emotion_label、user_profile；history 截止目标 user。所有行通过仓库 B1 adapter 和标签校验；缺字段/NULL 与合法空数组分别处理。无 user_id，不能证明用户级隔离。", "",
        "| 数据 | raw unique conversation_id | 所有预测点覆盖的 unique conversation_id | usable=true / B1 样本覆盖的 unique conversation_id | 原始预测点 | B1 样本 | ID 重复 | 缺失/NULL 必需字段 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s, key in (("train", "train"), ("val", "validation")):
        d = r[key]
        raw = r['raw'][s]
        lines.append(f"| {s} | {raw['raw_conversations']} | {raw['prediction_point_conversations']} | {d['usable_sample_conversations']} | {raw['prediction_points']} | {d['samples']} | {d['duplicate_sample_ids']} | 0 |")
    lines += ["", "8329 的定义是 classification_train.jsonl 中 18653 条 usable 训练样本实际覆盖的 unique conversation_id 数，与 split_manifest.jsonl 中 train 且 usable=true 的会话集合完全一致；它不是 raw train 的会话总数。raw、所有预测点、usable=true 和实际 B1 样本分别从真实文件核算，并逐个核对 (sample_id, conversation_id) 及出现次数。", ""]
    for s in ("train", "val"):
        raw = r['raw'][s]
        lines += [f"{s} 会话差异：{raw['raw_conversations']} → {raw['prediction_point_conversations']} → {raw['usable_sample_conversations']}。无 prediction point 的会话 {raw['no_prediction_point_conversations']} 个：{', '.join(raw['no_prediction_point_conversation_ids']) or '无'}；有预测点但全部 unusable 的会话 {raw['all_prediction_points_unusable_conversations']} 个：{', '.join(raw['all_prediction_points_unusable_conversation_ids']) or '无'}；其他缺失原因 {raw['other_missing_conversations']} 个，usable 会话不在 raw 中 {raw['usable_conversations_not_in_raw']} 个。", ""]
        for evidence in raw['all_prediction_points_unusable_evidence']:
            points = "; ".join(f"{p['sample_id']}: {', '.join(p['reasons'])}" for p in evidence['excluded_points'])
            lines += [f"- {evidence['conversation_id']}：{evidence['prediction_points']} 个预测点，usable 样本 0；{points}。"]
        lines.append("")
    lines += ["训练的 9 个会话差额由 5 个零预测点会话和 4 个全部预测点不可用会话组成：8338 − 5 − 4 = 8329；没有其他差额。4 个不可用会话各自只有 1 个预测点，均因 assistant_target_not_adjacent_future_turn、target_assistant_in_context 被排除，不能将它们概括为没有 prediction point。会话 ID 与逐点原因见 JSON raw.train。", "",
        "训练/验证 sample_id 交集 0，会话交集 0；fit/calibration 会话交集 0。原始训练的 4 个非相邻目标预测点已被既有 B1 转换排除（ID 见 JSON raw.train.b1_excluded_ids），不得再把原始 18657 点当作 B1 训练数。原始结构检查细节、每个 B1 文件条数与哈希匹配结果见 JSON。", "",
        "| 集合 / 画像维度 | 空数 | 空率 | 非空数 | 非空率 | 缺失/NULL |", "|---|---:|---:|---:|---:|---:|"]
    for key in ("train", "validation"):
        for dim, d in r[key]["profile"].items():
            lines.append(f"| {key}/{dim} | {d['empty_count']} | {d['empty_rate']:.6f} | {d['nonempty_count']} | {d['nonempty_rate']:.6f} | {d['missing_or_null_count']} |")
    lines += ["", "各维度标签支持数（多标签支持数之和不等于非空样本数；包含零支持标签）：", "",
              "| 维度 | 标签 | Train | Val |", "|---|---|---:|---:|"]
    for dim in PROFILE_LABELS_BY_DIMENSION:
        for label in PROFILE_LABELS_BY_DIMENSION[dim]:
            lines.append(f"| {dim} | {label} | {r['train']['profile'][dim]['label_support'][label]} | {r['validation']['profile'][dim]['label_support'][label]} |")
    lines += ["", "## 历史长度与画像", "",
        "历史以包含目标 user 的消息条数分桶：短 1–5、中 6–15、长 ≥16；字符数为 content 长度总和，不含格式符号。这里不以字符数猜 tokenizer 截断长度。分桶为 B 本次预设设计，非历史团队阈值。", "",
        "| 集合/历史 | 样本 | 全空画像数 | personality 非空率 | interests 非空率 | style 非空率 |", "|---|---:|---:|---:|---:|---:|"]
    for key in ("train", "validation"):
        for bucket, d in r[key]["history_profile"].items():
            values = " | ".join(f"{d['profile_nonempty_rate'][dim]:.6f}" for dim in PROFILE_LABELS_BY_DIMENSION)
            lines.append(f"| {key}/{bucket} | {d['samples']} | {d['all_empty_count']} | {values} |")
    lines += ["", "长短历史的画像非空率存在差异，不能把所有空数组视为漏标；上述为关联统计，不能推出因果。训练与验证的历史消息/字符 min、mean、p50、p95、max 见 JSON。", "",
        "## 指标定义与历史可比性", "",
        "复用 src.classification.emotion_profile.profile_metrics、emotion_encoder.classification_metrics 和 evaluation.metrics.profile_dimension_accuracy。exact 是每个维度整组标签集合完全相等，macro exact 对三维等权平均。micro 按标签 TP/FP/FN 池化；标签 macro 对该维所有固定枚举（含无支持标签）取均值，零分母为 0。global_micro 合并三维计数，dimension_macro 分别平均三维的 micro 与标签 macro。情感固定 16 类，混淆矩阵行是真值、列是预测。JSON 保留未舍入 macro exact，主要历史展示六位小数。",
        "历史 full-val 画像的 B1 train/val 哈希与本机完全一致；旧 head tensor exact 的 bool 行全等与当前集合 exact 定义等价。256/512 使用相同 speaker 行序列、左截断，但长度和校准目标不同，只能作历史端到端结果描述，不能当作统一预处理的单变量排名。空画像不依赖输入截断，因而可在同一金标上核对。缺预测则不能独立复算画像误例、非空率或长度切片。", "",
        "| 历史候选（按时间/配置列示，非排名） | macro exact | 各维 exact（personality/interests/style） | 各维 Micro-F1 | 证据 |", "|---|---:|---|---|---|"]
    p = r["empty_baseline"]["profile"]
    def dims(metrics, field):
        return " / ".join(f"{metrics['per_dimension'][d][field]:.6f}" for d in PROFILE_LABELS_BY_DIMENSION)
    lines.append(f"| 空画像 | {p['profile_macro_exact_match_accuracy']:.6f} | {dims(p, 'exact_match_accuracy')} | {dims(p, 'micro_f1')} | 本次完整重算 |")
    for h in r["history"]["profile"]:
        m = h["metrics"]
        if "per_dimension" in m:
            lines.append(f"| {h['name']} | {m['profile_macro_exact_match_accuracy']:.6f} | {dims(m, 'exact_match_accuracy')} | {dims(m, 'micro_f1')} | 聚合报告；{h['preprocessing']} |")
    lines += ["", "Day3 dry-run 仅 32 条：macro exact 0、空画像 0.666667、情感 Accuracy 0.3125/Macro-F1 0.029762，不与全量排名。Day3 另有训练内 exact 校准摘要 0.6223，缺分维/逐条结果，单独列在 JSON，不扩大结论。",
        "历史正类预测数显示过度输出：Day3 三维 1780/829/2256，对应金标正类数 1143/643/1333；Day4 256 不加权为 2445/1148/1870。它们是标签个数，不是非空样本数。Day4 512 不加权 exact 校准后仅预测 25/24/14 个标签，macro Micro-F1 0.019807，不能以非空输出说明超过空画像。",
        "空画像每维及全局 precision/recall/micro/macro F1、预测非空率均为 0；金标为空时 exact=1、非空时 exact=0。完整值及历史聚合原字段见 JSON。旧画像头的 precision/recall、标签 macro F1、非空样本率和关键切片缺失，明确不可回放；本任务不启动模型补推理。", "",
        "## 情感诊断与混淆", "",
        "| 来源 | Accuracy | Macro-F1 | 范围/可比性 |", "|---|---:|---:|---|"]
    e = r["empty_baseline"]["emotion"]
    lines.append(f"| 多数类 anxiety | {e['accuracy']:.6f} | {e['macro_f1']:.6f} | 同一全量金标；本次重算 |")
    for name, d in r["predictions"].items():
        if "emotion" in d:
            lines.append(f"| {name} | {d['emotion']['accuracy']:.6f} | {d['emotion']['macro_f1']:.6f} | {d['status']}; matches_report={d.get('matches_report', 'unknown')} |")
    lines += ["", "本地归档 Day3/Day5 完整 ID 对齐，情感 Accuracy/Macro-F1/逐类指标与 C 报告核对；归档输出的画像为空，不能误称旧画像头预测。旧 conditions 文件虽然可按 ID 重算，但权重/预处理来源不足，单独标为不可排名。Day3/Day5 的训练配置与输入预处理一致性依据 C 报告及脚本；此次未复载权重，不能据回放声称当前代码推理已复验。", "",
        "| 类别 | Train支持 | Val支持 | Day3 P/R/F1 | Day5 P/R/F1 |", "|---|---:|---:|---|---|"]
    emotions = r["history"]["emotion"][0]
    for label in EMOTION_LABELS:
        values = []
        for key in ("Day3", "Day5"):
            pc = emotions[key]["per_class"][label]
            values.append(" / ".join(f"{pc[k]:.6f}" for k in ("precision", "recall", "f1")))
        lines.append(f"| {label} | {r['train']['emotion_support'][label]} | {r['validation']['emotion_support'][label]} | {values[0]} | {values[1]} |")
    lines += ["", "重点混淆（各类支持数作分母，箭头为实际预测）：", ""]
    for name in ("Day3_archive", "Day5_archive"):
        if "emotion" not in r["predictions"][name]:
            lines.append(f"- {name} 缺失，无法逐条检查混淆。")
            continue
        m = r["predictions"][name]["emotion"]
        for label in ("mixed", "care"):
            counts = m["confusion_matrix"][m["labels"].index(label)]
            text = ", ".join(f"{other}={count}" for other, count in sorted(zip(m["labels"], counts), key=lambda t: (-t[1], t[0])) if count)
            lines.append(f"- {name} / {label} → {text}。")
    lines += ["", "Day5 相比 Day3 Accuracy +0.007222、Macro-F1 +0.011091，但 mixed F1 从 0.271318 降为 0.166667、recall 从 0.197740 降为 0.101695；care 两者 F1=0。不能只以总体准确率宣称全面改善。完整 16×16 混淆及历史/画像空非空/情感切片见 JSON predictions；这些历史成绩不是新盲测。", "",
        "## 固定训练内校准与筛选清单", "",
        f"种子 {r['seed']}；旧画像既有算法按 sorted(unique conversation_id) 后 Random(seed).shuffle，首 round(n×0.15) 个会话校准，剩余 fit。实际 fit 15944 条/7080 会话，calibration 2709 条/1249 会话，交集 0；验证条目进入训练内清单为 0。清单覆盖每条 B1 train，保留原 sample_id、会话、来源和 subset，不复制正文。后续 C 必须实际消费该清单或验证自身重建结果相同，B 未修改 C 训练程序。", "",
        f"固定筛选 {r['screening']['samples']} 条/{r['screening']['selected_sample_conversations']} 会话。每类目标 floor(160/16)=10；surprise 仅 9 条，缺额按类别支持数降序补给 anxiety（11 条）；care=10、mixed=10，其余每类 10。每类按 history×三维非空 mask 联合分层从稀有到常见轮转选一条，组内先按 ID 排序再用同一 Random(seed) 流依联合分层排序打乱，结果按 ID 排序。选择只读取金标分层、不读取候选成绩；完整规则与种子在 JSON，测试验证输入反序不影响输出。", "",
        f"历史覆盖 {r['screening']['history_support']}；三维 mask（personality/interests/style，1=非空）覆盖 {r['screening']['profile_mask_support']}。269 个可用联合分层覆盖 147 个，未覆盖 122 个，逐项列于 JSON screening.uncovered_joint_strata。160 条预算无法覆盖所有联合层，稀有优先导致样本不代表总体分布；不得把小集分数当作全量提升证明。该筛选集与剩余验证集可能共享会话，不用于阈值拟合；训练 fit/calibration 始终会话隔离。", "",
        "## 冻结的本地采用规则", ""]
    lines += [f"{i}. {rule}" for i, rule in enumerate(r["rules"], 1)]
    lines += ["", "## 溯源、复现与交接", "",
        "本报告和 JSON 是 B T1 实验诊断交付，不是独立验收记录；本地测试、重跑哈希证据和差异审查记录放 .codex_tmp/acceptance/cycle2/，不进普通 Git。输入文件逐个真实路径、字节数及 SHA-256 在 JSON sources；B1 原始/处理文件全部与 manifest 核对。源脚本和指标实现也列哈希，两个清单列输出哈希。", "",
        "```powershell", r["generation_command"],
        "python -B -m unittest discover -s tests -p test_b_t1_profile_diagnostics.py -v",
        "python -B -m unittest discover -s tests -v", "python -B scripts/ci_smoke.py", "git diff --check", "```", "",
        "生成文件：reports/B_T1_PROFILE_DIAGNOSTICS.md、reports/b_t1_profile_diagnostics.json、reports/b_t1_train_internal_split.jsonl、reports/b_t1_screening_manifest.jsonl。生成前严格核对输入；重跑会更新这四个 B T1 产物，不写原始数据与历史产物。跨机器缺本地归档时明确 missing，不能得到同一历史回放部分；复现需先核对 sources 哈希。",
        "已完成 T1 B 的历史诊断、清单冻结、采用规则。未完成项限于缺失证据：旧画像头逐条预测及其切片；模型推理时真实输入 token/截断比例；官方评分实现/权重；全模型未见校准的旧对照。这些未伪造、未用训练或大推理补齐。下一步由 C 在既定划分上训练并交付概率、阈值和完整 ID 预测；B 按 T2 做配对工具、T3 独立复核；D 使用固定筛选清单生成输入时剔除分层金标，仅取当轮及此前历史。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(SEED,), default=SEED)
    parser.add_argument("--screening-size", type=int, choices=(160,), default=160)
    args = parser.parse_args()
    # Work with repository-relative paths so evidence is portable.
    import os
    os.chdir(PROJECT)
    checks = verify_b1_hashes()
    train, val = (read_jsonl(B1 / f"classification_{s}.jsonl") for s in ("train", "val"))
    validate_rows(train, val)
    alignment = {}
    split_manifest = read_jsonl(B1 / "split_manifest.jsonl")
    for s, classification in (("train", train), ("val", val)):
        responses = read_jsonl(B1 / f"response_{s}.jsonl")
        expected = [r["sample_id"] for r in classification]
        included = [r["sample_id"] for r in split_manifest if r["source_split"] == s and r["classification_included"]]
        alignment[s] = {"classification_response_order_identical": expected == [r["sample_id"] for r in responses],
                        "classification_manifest_order_identical": expected == included,
                        "response_fields": sorted(set().union(*(set(r) for r in responses))),
                        "split_manifest_fields": sorted(set(split_manifest[0]))}
        if not all(alignment[s][k] for k in ("classification_response_order_identical", "classification_manifest_order_identical")):
            raise ValueError(f"B1 {s} response or split manifest ID alignment mismatch")
        del responses
    raw = {}
    for split in ("train", "val"):
        rows = read_jsonl(RAW / split / f"{split}_public.jsonl")
        points = [sample_id(r["conversation_id"], p, i) for r in rows for i, p in enumerate(r["prediction_points"], 1)]
        coverage = conversation_coverage(rows, split_manifest, train if split == "train" else val, split)
        raw[split] = {**coverage,
            "prediction_points": len(points), "unique_sample_ids": len(set(points)),
            "missing_fields": dict(Counter(k for r in rows for p in r["prediction_points"] for k in ("emotion_label", "user_profile", "response_text") if p.get("target", {}).get(k) is None)),
            "structure_errors": dict(Counter(e for r in rows for e in validate_conversation(r, split))),
            "b1_excluded_ids": sorted(set(points)-{r["sample_id"] for r in (train if split == "train" else val)})}
    split = internal_split(train, args.seed)
    screen = screening_manifest(val, args.screening_size, args.seed)
    out = Path("reports")
    split_path, screen_path = out / "b_t1_train_internal_split.jsonl", out / "b_t1_screening_manifest.jsonl"
    write_jsonl(split_path, split)
    write_jsonl(screen_path, screen)
    empty = [{"id": r["sample_id"], "emotion_label": "anxiety", "user_profile": {d: [] for d in PROFILE_LABELS_BY_DIMENSION}} for r in val]
    result = {"version": "b_t1_v2", "frozen_date": "2026-10-22", "timezone": "Asia/Shanghai", "seed": args.seed,
        "generation_command": COMMAND if args.seed == SEED and args.screening_size == 160 else f"python -B scripts/b_t1_profile_diagnostics.py --seed {args.seed} --screening-size {args.screening_size}",
        "rules": RULES, "profile_mask_dimension_order": list(PROFILE_LABELS_BY_DIMENSION),
        "history_definition": "message count includes target user; short <=5, medium 6..15, long >=16; chars=sum(len(content)), no separators; no tokenizer lengths inferred",
        "metric_definitions": "repository profile_metrics + classification_metrics; fixed labels; zero denominator=0; full ID set joined by ID; macro exact unrounded is mean sample dimension accuracy; global micro pools all dimensions; dimension_macro.micro averages 3 micro F1; dimension_macro.macro averages each dimension fixed-label macro F1",
        "python": sys.version, "raw": raw, "b1_file_checks": checks, "b1_alignment": alignment,
        "train": dataset_stats(train), "validation": dataset_stats(val),
        "cross_split_sample_overlap": len({r["sample_id"] for r in train} & {r["sample_id"] for r in val}),
        "cross_split_conversation_overlap": len({r["conversation_id"] for r in train} & {r["conversation_id"] for r in val}),
        "internal_split": {"algorithm": "sorted unique conversation IDs; Random(seed).shuffle; first max(1,round(n*.15)) calibration; output sample_id sorted", "fraction": .15,
            "subsets": {s: {"samples": sum(r["subset"] == s for r in split), "usable_sample_conversations": len({r["conversation_id"] for r in split if r["subset"] == s})} for s in ("fit", "calibration")},
            "fit_calibration_conversation_overlap": len({r["conversation_id"] for r in split if r["subset"] == "fit"} & {r["conversation_id"] for r in split if r["subset"] == "calibration"}),
            "validation_rows": 0, "historical_algorithm_reused": "scripts/train_profile_heads.py and scripts/calibrate_profile_exact_thresholds.py; does not remove prior encoder exposure"},
        "screening": {"samples": len(screen), "selected_sample_conversations": len({r["conversation_id"] for r in screen}),
            "selection_rule": "near-equal emotion quota floor(size/classes), capped at support; deficit round robin descending class support; within class visit joint length/mask strata ascending support then tuple, one row per stratum per round; sample_id sorted then seeded shuffle per stratum; no predictions read",
            "emotion_support": dict(sorted(Counter(r["emotion_label"] for r in screen).items())),
            "history_support": dict(sorted(Counter(r["history_bucket"] for r in screen).items())),
            "profile_mask_support": dict(sorted(Counter(r["profile_nonempty_mask"] for r in screen).items())),
            "available_joint_strata": len({strata(r) for r in val}), "covered_joint_strata": len({(r["emotion_label"],r["history_bucket"],r["profile_nonempty_mask"]) for r in screen}),
            "uncovered_joint_strata": [list(k) for k in sorted({strata(r) for r in val}-{(r["emotion_label"],r["history_bucket"],r["profile_nonempty_mask"]) for r in screen})],
            "use": "fixed candidate screening only; never threshold fitting; not blind"},
        "empty_baseline": replay(val, empty), "predictions": {},
        "history": historical_reports(checks["classification_train.jsonl"]["sha256"], checks["classification_val.jsonl"]["sha256"])}
    sources = [B1 / "manifest.json", *[B1 / name for name in checks], *[RAW / s / f"{s}_public.jsonl" for s in ("train", "val")]]
    for name, path in (("old_conditions_unknown_provenance", Path("results/response_gen/c2_conditions_val_full.jsonl")),
                       ("Day3_archive", ARCHIVE / "baseline_submission.jsonl"), ("Day5_archive", ARCHIVE / "submission.jsonl")):
        if path.exists():
            sources.append(path)
            try:
                metrics = replay(val, read_jsonl(path))
                metrics["status"] = "full ID-aligned replay; archived emotion metrics must match C report; profile is deployed empty baseline, not old profile heads" if name != "old_conditions_unknown_provenance" else "full ID-aligned replay, model/weight/preprocessing provenance unconfirmed; do not rank"
                metrics["source"] = path.as_posix()
                if name != "old_conditions_unknown_provenance":
                    ref = result["history"]["emotion"][0]["Day3" if name == "Day3_archive" else "Day5"]
                    metrics["matches_report"] = all(metrics["emotion"][k] == ref[k] for k in ("accuracy", "macro_f1", "per_class"))
                    metrics["profile_is_all_empty"] = all(not any(r["user_profile"].values()) for r in read_jsonl(path))
                    archive_input = ARCHIVE / "val2354.jsonl"
                    if archive_input.exists():
                        inputs = read_jsonl(archive_input)
                        metrics["archived_context_matches_b1"] = len(inputs) == len(val) and all(
                            a["id"] == b["sample_id"] and a["conversation_id"] == b["conversation_id"]
                            and a["target_user_turn_id"] == b["target_user_turn_id"] and a["history"] == b["context_turns"]
                            for a, b in zip(inputs, val))
                    if name == "Day5_archive" and (ARCHIVE / "performance_report.json").exists():
                        performance = json.loads((ARCHIVE / "performance_report.json").read_text(encoding="utf-8"))
                        metrics["archive_submission_hash_matches_performance"] = performance.get("submission_sha256") == sha256_file(path)
                        metrics["archive_input_hash_matches_performance"] = archive_input.exists() and performance.get("test_file_sha256") == sha256_file(archive_input)
                    if not metrics["matches_report"] or metrics.get("archived_context_matches_b1") is False:
                        metrics["status"] = "historical report or input mismatch; do not rank"
                result["predictions"][name] = metrics
            except (ValueError, KeyError, TypeError) as exc:
                result["predictions"][name] = {"source": path.as_posix(), "status": "unusable", "error": str(exc)}
        else:
            result["predictions"][name] = {"source": path.as_posix(), "status": "missing; no inference performed"}
    sources += [Path(x["path"]) for x in result["history"]["inventory"] if x["status"] == "available"]
    sources += [ARCHIVE / n for n in ("provenance.json", "metrics_comparison.json", "val2354.jsonl", "performance_report.json", "integrated.json", "integrated_artifacts.json") if (ARCHIVE / n).exists()]
    sources += [Path(p) for p in ("scripts/b_t1_profile_diagnostics.py", "scripts/train_profile_heads.py", "scripts/calibrate_profile_exact_thresholds.py", "scripts/finetune_emotion_balanced.py", "scripts/prepare_training_data.py", "scripts/audit_dataset.py", "scripts/b_validate_submission.py", "src/common/io.py", "src/classification/emotion_profile.py", "src/classification/emotion_encoder.py", "src/data/schema.py", "src/evaluation/metrics.py", "docs/EXECUTION_PLAN_CYCLE2.md", "docs/OFFICIAL_REQUIREMENTS.md", "docs/COLLABORATION.md", "tests/test_b_t1_profile_diagnostics.py")]
    result["sources"] = [{"path": p.as_posix(), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(set(sources))]
    result["outputs"] = [{"path": p.as_posix(), "sha256": sha256_file(p), "records": len(rows)} for p, rows in ((split_path, split), (screen_path, screen))]
    write_json(out / "b_t1_profile_diagnostics.json", result)
    (out / "B_T1_PROFILE_DIAGNOSTICS.md").write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps({"train": len(train), "val": len(val), "split": result["internal_split"]["subsets"], "screening_samples": len(screen), "empty_exact": result["empty_baseline"]["profile"]["profile_macro_exact_match_accuracy"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
