#!/usr/bin/env python3
"""Turn C's standard emotion classifier into a D condition file.

Cycle-2 T1 compares the closed-condition arm against an emotion-only arm, so
the conditions must be rebuilt from the latest C Day5 checkpoint. That
checkpoint is a standard HuggingFace sequence-classification model with no
profile heads, which the packaged joint format loaded by
``scripts/build_condition_file.py`` (EmotionProfileCheckpoint) cannot read.

Inference here copies ``scripts/finetune_emotion_balanced.py`` verbatim — same
speaker formatting, truncation side, max length, batching and argmax-to-label
mapping — so the accuracy and macro-F1 reported alongside the condition file
reconcile with C's training report instead of forming a second, slightly
different inference path. Row validation follows ``build_condition_file.py``:
contexts must end at the target user turn (no future-turn leakage), malformed
rows are refused, and a short prediction list is never written as a partial
file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.classification.emotion_encoder import classification_metrics  # noqa: E402
from src.common.io import write_jsonl  # noqa: E402
from src.data.schema import EMOTIONS  # noqa: E402

LABELS = tuple(sorted(EMOTIONS))
SPEAKERS = {"user": "用户", "assistant": "助手"}
MAX_LENGTH = 256
BATCH_SIZE = 32

REQUIRED_FIELDS = (
    "sample_id",
    "conversation_id",
    "target_user_turn_id",
    "source_split",
    "context_turns",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_context_text(turns: Sequence[dict[str, Any]]) -> str:
    """Render context turns exactly the way C's training pipeline does."""
    for turn in turns:
        if turn.get("role") not in SPEAKERS:
            raise ValueError(f"unknown speaker role: {turn.get('role')!r}")
        if not isinstance(turn.get("content"), str):
            raise ValueError(f"turn {turn.get('turn_id')!r} has no string content")
    return "\n".join(
        f"{SPEAKERS[turn['role']]}：{turn['content']}" for turn in turns
    )


def labels_from_indices(indices: Sequence[int]) -> list[str]:
    """Map argmax positions to labels, refusing anything outside the schema."""
    labels: list[str] = []
    for index in indices:
        if not 0 <= index < len(LABELS):
            raise ValueError(f"prediction index {index} outside the label schema")
        labels.append(LABELS[index])
    return labels


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read B1 classification rows, refusing anything not cut at the target turn.

    The gold emotion_label is optional: it only feeds the accuracy/macro-F1
    reconciliation against C's report, while the condition file itself needs
    just the sample id and the predicted label.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {missing}")
            sample_id = row["sample_id"]
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{path}:{line_number}: invalid sample_id")
            if sample_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id")
            seen.add(sample_id)
            turns = row["context_turns"]
            if not isinstance(turns, list) or not turns:
                raise ValueError(f"{path}:{line_number}: empty context_turns")
            for turn in turns:
                if (
                    not isinstance(turn, dict)
                    or "turn_id" not in turn
                    or "role" not in turn
                    or "content" not in turn
                ):
                    raise ValueError(f"{path}:{line_number}: malformed context turn")
            if turns[-1]["role"] != "user":
                raise ValueError(
                    f"{path}:{line_number}: context does not end with a user turn"
                )
            if turns[-1]["turn_id"] != row["target_user_turn_id"]:
                raise ValueError(
                    f"{path}:{line_number}: context is not cut at target_user_turn_id"
                )
            gold = row.get("emotion_label")
            if gold is not None and gold not in LABELS:
                raise ValueError(f"{path}:{line_number}: invalid emotion label {gold!r}")
            rows.append({
                "sample_id": sample_id,
                "conversation_id": row["conversation_id"],
                "target_user_turn_id": row["target_user_turn_id"],
                "source_split": row["source_split"],
                "text": format_context_text(turns),
                "gold": gold,
            })
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def condition_rows(
    rows: list[dict[str, Any]], predicted: Sequence[str]
) -> list[dict[str, Any]]:
    """Pair each prediction with its sample id, refusing to drop or reorder any."""
    if len(rows) != len(predicted):
        raise ValueError(
            f"{len(predicted)} predictions for {len(rows)} rows; refusing to "
            "write a partial condition file"
        )
    conditions: list[dict[str, Any]] = []
    for row, label in zip(rows, predicted):
        if label not in LABELS:
            raise ValueError(
                f"prediction {label!r} for {row['sample_id']!r} is not a schema label"
            )
        conditions.append({"id": row["sample_id"], "emotion_label": label})
    return conditions


def _select_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def predict_labels(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    device: Any,
    batch_size: int,
) -> list[str]:
    """Batch argmax inference with C's exact tokenizer and decoding settings."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    predicted: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            tokens = tokenizer(
                batch, truncation=True, max_length=MAX_LENGTH,
                padding=True, return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            guesses = model(**tokens).logits.argmax(dim=-1).cpu().tolist()
            predicted.extend(labels_from_indices(guesses))
    return predicted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="standard sequence-classification checkpoint directory",
    )
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT / "data/processed/b1_full_v1/classification_val.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = read_rows(args.input)
    device = _select_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.checkpoint), local_files_only=True
    )
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.checkpoint), local_files_only=True
    ).to(device).eval()
    if int(model.config.num_labels) != len(LABELS):
        raise ValueError(
            f"checkpoint exposes {model.config.num_labels} labels; "
            f"the schema needs {len(LABELS)}"
        )
    predicted = predict_labels(
        model, tokenizer, [row["text"] for row in rows], device, args.batch_size
    )
    conditions = condition_rows(rows, predicted)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, conditions)

    gold = [row["gold"] for row in rows]
    metrics = (
        classification_metrics(gold, predicted, LABELS)
        if all(label is not None for label in gold)
        else None
    )
    predicted_counts = {label: predicted.count(label) for label in LABELS}
    gold_counts = {label: gold.count(label) for label in LABELS}
    report = {
        "version": "emotion_condition_file_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_model_sha256": _sha256(args.checkpoint / "model.safetensors"),
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "output": str(args.output),
        "rows": len(rows),
        "device": str(device),
        "batch_size": args.batch_size,
        "max_length": MAX_LENGTH,
        "gold_labels_present": metrics is not None,
        "metrics": metrics,
        "predicted_label_counts": dict(
            sorted(predicted_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "gold_label_counts": dict(
            sorted(gold_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if metrics:
        print(
            f"PASS: {len(conditions)} conditions -> {args.output} "
            f"(accuracy {metrics['accuracy']}, macro-F1 {metrics['macro_f1']})",
            flush=True,
        )
    else:
        print(f"PASS: {len(conditions)} conditions -> {args.output}", flush=True)
    return report


if __name__ == "__main__":
    run(parse_args())
