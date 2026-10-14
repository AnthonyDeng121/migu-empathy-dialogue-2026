#!/usr/bin/env python3
"""Build B1 smoke or full samples; preserve official split membership.

context_text is one compact JSON object per line, in ascending turn_id order,
with keys turn_id, role, content in that order (UTF-8, no trailing newline).
Content is preserved verbatim; JSON escaping makes embedded newlines unambiguous.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl
from src.data.schema import (
    validate_classification_labels,
    validate_prediction,
    validate_test_row,
)

RAW = PROJECT / "data/raw/训练-验证-数据集"
MEMORY_STATUS = "无法进一步验证引用：未提供外部 memory bank"
CONTEXT_TEXT_SPEC = (
    "Ascending turn_id <= target_user_turn_id, including the target user turn; "
    "one compact JSON object per line, keys in order: turn_id, role, content; "
    "ensure_ascii=False, separators=(',', ':'), joined with LF, no trailing LF; "
    "content preserved verbatim with JSON escaping."
)
FULL_JSONL = {
    "classification_train": "classification_train.jsonl",
    "classification_val": "classification_val.jsonl",
    "response_train": "response_train.jsonl",
    "response_val": "response_val.jsonl",
    "manifest": "split_manifest.jsonl",
    "anomalies": "anomalies.jsonl",
}


def positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def sample_id(conversation_id: str, point: Any, point_index: int) -> str:
    if isinstance(point, dict):
        for field in ("sample_id", "id"):
            value = point.get(field)
            if isinstance(value, str) and value.strip():
                return value
        target_id = point.get("target_user_turn_id")
    else:
        target_id = None
    # JSON tuple encoding avoids delimiter collisions in conversation IDs.
    return "pp:" + json.dumps(
        [conversation_id, target_id, point_index], ensure_ascii=False, separators=(",", ":")
    )


def index_conversations(rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, dict]]]:
    groups: dict[str, list[tuple[int, dict]]] = {}
    for row_index, row in enumerate(rows, 1):
        cid = row.get("conversation_id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"source record {row_index}: invalid conversation_id")
        if not isinstance(row.get("prediction_points"), list):
            raise ValueError(f"{cid}: prediction_points must be an array")
        groups.setdefault(cid, []).append((row_index, row))
    return groups


def expand_point(row: dict, point: Any, point_index: int, row_index: int,
                 source_split: str) -> tuple[dict, dict | None, dict | None]:
    if source_split not in {"train", "val"}:
        raise ValueError("source_split must be train or val")
    cid = row["conversation_id"]
    point_data = point if isinstance(point, dict) else {}
    target_id = point_data.get("target_user_turn_id")
    assistant_id = point_data.get("assistant_response_turn_id")
    sid = sample_id(cid, point, point_index)
    reasons: list[str] = []
    manifest = {
        "sample_id": sid,
        "conversation_id": cid,
        "target_user_turn_id": target_id,
        "assistant_response_turn_id": assistant_id,
        "source_split": source_split,
        "source_record_index": row_index,
        "prediction_point_index": point_index,
        "usable": False,
        "reasons": reasons,
        "classification_included": False,
        "response_included": False,
        "memory_refs_verification": MEMORY_STATUS,
    }
    if not isinstance(point, dict):
        reasons.append("prediction_point_not_object")

    turns = row.get("turns")
    ids = []
    safe = isinstance(turns, list) and bool(turns)
    if safe:
        for turn in turns:
            if not isinstance(turn, dict) or not positive_int(turn.get("turn_id")):
                safe = False
                break
            ids.append(turn["turn_id"])
        safe = safe and ids == sorted(ids) and len(ids) == len(set(ids))
    if not safe:
        reasons.append("unsafe_context_turn_ids")
    by_id = {
        turn["turn_id"]: turn
        for turn in (turns if isinstance(turns, list) else [])
        if isinstance(turn, dict) and positive_int(turn.get("turn_id"))
    }
    user = by_id.get(target_id) if positive_int(target_id) else None
    assistant = by_id.get(assistant_id) if positive_int(assistant_id) else None
    if user is None:
        reasons.append("target_user_turn_missing")
    elif user.get("role") != "user":
        reasons.append("target_user_turn_not_user")
    if assistant is None:
        reasons.append("assistant_target_turn_missing")
    else:
        if assistant.get("role") != "assistant":
            reasons.append("assistant_target_turn_not_assistant")
        if positive_int(target_id) and assistant_id != target_id + 1:
            reasons.append("assistant_target_not_adjacent_future_turn")

    gold = point_data.get("target")
    if not isinstance(gold, dict):
        reasons.append("target_labels_not_object")
        gold = {}
    if assistant is not None and gold.get("response_text") != assistant.get("content"):
        reasons.append("assistant_target_gold_response_mismatch")
    try:
        validate_classification_labels(gold.get("emotion_label"), gold.get("user_profile"))
    except (ValueError, TypeError) as exc:
        reasons.append(f"classification_labels_invalid: {exc}")
    try:
        validate_prediction({
            "id": sid,
            "response_text": gold.get("response_text"),
            "emotion_label": gold.get("emotion_label"),
            "user_profile": gold.get("user_profile"),
            "memory_refs": gold.get("memory_refs"),
        })
    except (ValueError, TypeError) as exc:
        message = f"target_labels_invalid: {exc}"
        reasons.append(message)

    context = []
    if safe and user is not None:
        context = [
            {"turn_id": turn["turn_id"], "role": turn.get("role"), "content": turn.get("content")}
            for turn in turns if turn["turn_id"] <= target_id
        ]
        try:
            validate_test_row({
                "id": sid, "conversation_id": cid,
                "target_user_turn_id": target_id, "history": context,
            })
        except (ValueError, TypeError) as exc:
            reasons.append(f"unsafe_context: {exc}")
        if any(turn["turn_id"] == assistant_id for turn in context):
            reasons.append("target_assistant_in_context")
    if reasons:
        return manifest, None, None

    base = {
        "sample_id": sid, "conversation_id": cid,
        "target_user_turn_id": target_id, "source_split": source_split,
        "context_turns": context,
        "context_text": "\n".join(
            json.dumps(turn, ensure_ascii=False, separators=(",", ":")) for turn in context
        ),
    }
    classification = {**base, "emotion_label": gold["emotion_label"], "user_profile": gold["user_profile"]}
    response = {
        **base, "response_text": gold["response_text"], "memory_refs": gold["memory_refs"],
        "memory_refs_verification": MEMORY_STATUS,
    }
    manifest.update(usable=True, classification_included=True, response_included=True)
    return manifest, classification, response


def source_checks(train_groups: dict, val_groups: dict) -> dict:
    overlap = sorted(train_groups.keys() & val_groups.keys())
    # Check IDs across both full source splits, without expanding full training data.
    id_counts = Counter(
        sample_id(cid, point, point_index)
        for groups in (train_groups, val_groups)
        for cid, records in groups.items()
        for _, row in records
        for point_index, point in enumerate(row["prediction_points"], 1)
    )
    duplicate_ids = sorted(sid for sid, frequency in id_counts.items() if frequency > 1)
    return {
        "train_val_conversation_overlap": len(overlap),
        "overlap_conversation_ids": overlap,
        "duplicate_sample_ids": len(duplicate_ids),
        "duplicate_sample_id_scope": "all train/val prediction points",
        "duplicate_sample_id_values": duplicate_ids,
    }


def build_smoke(train: list[dict], val: list[dict], count: int = 10, seed: int = 20261013) -> dict:
    if not positive_int(count):
        raise ValueError("smoke_conversations must be a positive integer")
    train_groups, val_groups = index_conversations(train), index_conversations(val)
    checks = source_checks(train_groups, val_groups)
    summary = {
        **checks,
        "seed": seed,
        "smoke_conversations": 0,
        "selected_conversation_ids": [],
        "prediction_points": 0, "usable": 0, "excluded": 0,
        "classification_samples": 0, "response_samples": 0,
        "context_after_target": False,
        "target_assistant_in_context": False,
        "reason_counts": {},
        "checks_passed": False,
    }
    result = {"classification": [], "response": [], "manifest": [], "summary": summary}
    if checks["train_val_conversation_overlap"] or checks["duplicate_sample_ids"]:
        return result
    selected = sorted(random.Random(seed).sample(sorted(train_groups), min(count, len(train_groups))))
    summary["smoke_conversations"] = len(selected)
    summary["selected_conversation_ids"] = selected
    for cid in selected:
        for row_index, row in train_groups[cid]:
            for point_index, point in enumerate(row["prediction_points"], 1):
                manifest, classification, response = expand_point(row, point, point_index, row_index, "train")
                result["manifest"].append(manifest)
                if classification is not None:
                    result["classification"].append(classification)
                    result["response"].append(response)
    assistant_ids = {m["sample_id"]: m["assistant_response_turn_id"] for m in result["manifest"]}
    samples = result["classification"] + result["response"]
    summary.update(
        prediction_points=len(result["manifest"]),
        usable=sum(m["usable"] for m in result["manifest"]),
        excluded=sum(not m["usable"] for m in result["manifest"]),
        classification_samples=len(result["classification"]),
        response_samples=len(result["response"]),
        context_after_target=any(
            turn["turn_id"] > sample["target_user_turn_id"]
            for sample in samples for turn in sample["context_turns"]
        ),
        target_assistant_in_context=any(
            turn["turn_id"] == assistant_ids[sample["sample_id"]]
            for sample in samples for turn in sample["context_turns"]
        ),
        reason_counts=dict(sorted(Counter(
            reason for m in result["manifest"] for reason in m["reasons"]
        ).items())),
    )
    summary["checks_passed"] = not (
        summary["context_after_target"] or summary["target_assistant_in_context"]
    )
    return result


def check_full_result(result: dict) -> dict:
    """Validate both output splits against the complete prediction point ledger.

    Leak flags describe emitted C/D contexts. A rejected source reference pointing
    into the past is an anomaly, not a leak in the emitted training data.
    """
    report = result["summary"]
    ledger = result["manifest"]
    by_id = {m["sample_id"]: m for m in ledger}
    report["duplicate_sample_ids"] = max(
        report["duplicate_sample_ids"],
        sum(n > 1 for n in Counter(m["sample_id"] for m in ledger).values()),
    )
    samples = [sample for split in ("train", "val") for task in ("classification", "response")
               for sample in result[f"{task}_{split}"]]
    report["context_after_target"] = any(
        turn["turn_id"] > sample["target_user_turn_id"]
        for sample in samples for turn in sample["context_turns"]
    )
    report["target_assistant_in_context"] = any(
        sample["sample_id"] in by_id
        and turn["turn_id"] == by_id[sample["sample_id"]]["assistant_response_turn_id"]
        for sample in samples for turn in sample["context_turns"]
    )
    consistent = len(ledger) == report["train_prediction_points"] + report["val_prediction_points"]
    consistent = consistent and len(by_id) == len(ledger)
    consistent = consistent and result["anomalies"] == [m for m in ledger if not m["usable"]]
    for split in ("train", "val"):
        records = [m for m in ledger if m["source_split"] == split]
        usable = [m for m in records if m["usable"]]
        consistent = consistent and len(records) == report[f"{split}_prediction_points"]
        consistent = consistent and all(
            m["classification_included"] == m["response_included"] == m["usable"]
            and bool(m["reasons"]) == (not m["usable"]) for m in records
        )
        report[f"excluded_{split}"] = len(records) - len(usable)
        expected_ids = [m["sample_id"] for m in usable]
        for task in ("classification", "response"):
            output_samples = result[f"{task}_{split}"]
            report[f"{task}_{split}_samples"] = len(output_samples)
            consistent = consistent and [s["sample_id"] for s in output_samples] == expected_ids
            for sample in output_samples:
                record = by_id.get(sample["sample_id"])
                if record is None:
                    consistent = False
                    continue
                consistent = consistent and all(
                    sample[field] == record[field]
                    for field in ("sample_id", "conversation_id", "target_user_turn_id", "source_split")
                ) and sample["source_split"] == split
        consistent = consistent and all(
            c["context_turns"] == d["context_turns"] and c["context_text"] == d["context_text"]
            for c, d in zip(result[f"classification_{split}"], result[f"response_{split}"])
        )
    consistent = consistent and all(m["source_split"] in {"train", "val"} for m in ledger)
    report["manifest_samples_consistent"] = consistent
    report["reason_counts"] = dict(sorted(Counter(
        reason for m in ledger for reason in m["reasons"]
    ).items()))
    report["checks_passed"] = not (
        report["train_val_conversation_overlap"] or report["duplicate_sample_ids"]
        or report["context_after_target"] or report["target_assistant_in_context"]
        or not consistent
    )
    return report


def build_full(train: list[dict], val: list[dict]) -> dict:
    groups = {"train": index_conversations(train), "val": index_conversations(val)}
    checks = source_checks(groups["train"], groups["val"])
    summary = {
        **checks,
        "train_conversations": len(groups["train"]),
        "val_conversations": len(groups["val"]),
        "train_prediction_points": sum(len(row["prediction_points"]) for row in train),
        "val_prediction_points": sum(len(row["prediction_points"]) for row in val),
        "classification_train_samples": 0, "classification_val_samples": 0,
        "response_train_samples": 0, "response_val_samples": 0,
        "excluded_train": 0, "excluded_val": 0,
        "context_after_target": False, "target_assistant_in_context": False,
        "manifest_samples_consistent": False,
        "reason_counts": {}, "checks_passed": False,
    }
    result = {key: [] for key in FULL_JSONL}
    result["summary"] = summary
    if checks["train_val_conversation_overlap"] or checks["duplicate_sample_ids"]:
        return result
    # Stable order: train then val, sorted conversation ID, source record order,
    # then the original 1-based prediction point order. No random split changes.
    for split in ("train", "val"):
        for cid in sorted(groups[split]):
            for row_index, row in groups[split][cid]:
                for point_index, point in enumerate(row["prediction_points"], 1):
                    record, classification, response = expand_point(row, point, point_index, row_index, split)
                    result["manifest"].append(record)
                    if record["usable"]:
                        result[f"classification_{split}"].append(classification)
                        result[f"response_{split}"].append(response)
                    else:
                        result["anomalies"].append(record)
    check_full_result(result)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, write_content: Callable[[TextIO], None]) -> None:
    """Atomically replace one output file without following a leaf symlink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            write_content(handle)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json(path: Path, value: dict) -> None:
    atomic_write(
        path,
        lambda handle: handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n"),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    def write_rows(handle: TextIO) -> None:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    atomic_write(path, write_rows)


def validate_output_paths(output: Path, filenames: list[str], train: Path, val: Path) -> None:
    output = output.resolve()
    raw = (PROJECT / "data/raw").resolve()
    sources = {train.resolve(), val.resolve()}
    output_files = [output / name for name in filenames]
    for path in output_files:
        if path.is_symlink():
            raise ValueError(f"output file must not be a symlink: {path.name}")
    destinations = [output] + [path.resolve() for path in output_files]
    for path in destinations:
        if path == raw or raw in path.parents:
            raise ValueError("output-dir and output files must not be inside data/raw")
        if path in sources:
            raise ValueError("output must not overwrite a source dataset")
        if path.is_file() and any(path.samefile(source) for source in sources):
            raise ValueError("output must not overwrite a source dataset through a hard link")
    if len(set(destinations)) != len(destinations):
        raise ValueError("output paths must be distinct")


def write_full(output: Path, result: dict, train: Path, val: Path, input_hashes: dict) -> None:
    if not check_full_result(result)["checks_passed"]:
        raise ValueError("full output failed leakage or manifest/sample consistency checks")
    filenames = list(FULL_JSONL.values()) + ["leakage.json", "manifest.json", "SHA256SUMS"]
    validate_output_paths(output, filenames, train, val)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for key, filename in FULL_JSONL.items():
        path = output / filename
        write_jsonl(path, result[key])
        # Verify actual persisted content, including counts, before certifying it.
        if read_jsonl(path) != result[key]:
            raise ValueError(f"persisted records differ from checked records: {filename}")
        metadata[filename] = {"record_count": len(result[key]), "sha256": sha256(path)}
    write_json(output / "leakage.json", result["summary"])
    metadata["leakage.json"] = {"record_count": 1, "sha256": sha256(output / "leakage.json")}
    manifest = {
        "conversion_version": "b1_v1",
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "inputs": {
            split: {"path": str(path.resolve()), "sha256": input_hashes[split]}
            for split, path in (("train", train), ("val", val))
        },
        "outputs": metadata,
        "anomaly_count": len(result["anomalies"]),
        "key_checks": result["summary"],
        "context_text_spec": CONTEXT_TEXT_SPEC,
        "ordering": "train then val; sorted conversation_id; source record order; prediction point order",
        "hash_scope": "outputs lists data and leakage hashes; SHA256SUMS additionally hashes manifest.json; SHA256SUMS excludes itself",
    }
    write_json(output / "manifest.json", manifest)
    def write_hashes(handle: TextIO) -> None:
        for filename in sorted(list(metadata) + ["manifest.json"]):
            handle.write(f"{sha256(output / filename)}  {filename}\n")

    atomic_write(output / "SHA256SUMS", write_hashes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare B1 smoke v0 or full v1 with official splits")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--train", type=Path, default=RAW / "train/train_public.jsonl")
    parser.add_argument("--val", type=Path, default=RAW / "val/val_public.jsonl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-conversations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args(argv)
    try:
        output = (args.output_dir or PROJECT / "data/processed" / (
            "b1_full_v1" if args.mode == "full" else "b1_smoke_v0"
        )).resolve()
        input_hashes = {"train": sha256(args.train), "val": sha256(args.val)} if args.mode == "full" else {}
        train, val = read_jsonl(args.train), read_jsonl(args.val)
        result = (build_full(train, val) if args.mode == "full" else
                  build_smoke(train, val, args.smoke_conversations, args.seed))
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        if not result["summary"]["checks_passed"]:
            print("FAIL: split overlap, duplicate sample IDs, context leakage, or inconsistent counts; no outputs written", file=sys.stderr)
            return 1
        if args.mode == "full":
            if input_hashes != {"train": sha256(args.train), "val": sha256(args.val)}:
                raise ValueError("source datasets changed during conversion")
            write_full(output, result, args.train, args.val, input_hashes)
        else:
            smoke_files = {
                "classification": "classification_smoke.jsonl",
                "response": "response_smoke.jsonl",
                "manifest": "split_manifest_smoke.jsonl",
            }
            validate_output_paths(output, list(smoke_files.values()), args.train, args.val)
            for key, filename in smoke_files.items():
                write_jsonl(output / filename, result[key])
        print(f"PASS output_dir={output}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
