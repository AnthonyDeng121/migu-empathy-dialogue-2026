#!/usr/bin/env python3
"""Export ID-aligned emotion or profile probabilities without dialogue text."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.build_emotion_condition_file import (  # noqa: E402
    LABELS as EMOTION_LABELS,
    MAX_LENGTH as EMOTION_MAX_LENGTH,
    read_rows,
)
from src.classification.emotion_profile_checkpoint import (  # noqa: E402
    EmotionProfileCheckpoint,
)
from src.data.schema import EMOTIONS, PROFILE_LABELS  # noqa: E402


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    sample_id = row.get("sample_id", row.get("id"))
    conversation_id = row.get("conversation_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("prediction row has an invalid sample ID")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError(f"prediction row {sample_id!r} has an invalid conversation ID")
    return sample_id, conversation_id


def _validate_probability(value: Any) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probabilities must be finite numbers in [0, 1]")
    return float(value)


def emotion_prediction_records(
    rows: Sequence[dict[str, Any]],
    probability_rows: Sequence[Sequence[float]],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Build privacy-minimal records from multiclass probabilities."""
    if len(rows) != len(probability_rows):
        raise ValueError("row and probability counts differ")
    if not labels or len(labels) != len(set(labels)) or set(labels) != EMOTIONS:
        raise ValueError("emotion labels do not match the official schema")

    records = []
    seen: set[str] = set()
    for row, values in zip(rows, probability_rows):
        sample_id, conversation_id = _identity(row)
        if sample_id in seen:
            raise ValueError(f"duplicate sample ID: {sample_id}")
        seen.add(sample_id)
        if len(values) != len(labels):
            raise ValueError("emotion probability label count does not match labels")
        probabilities = {
            label: _validate_probability(value) for label, value in zip(labels, values)
        }
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
            raise ValueError(f"emotion probabilities do not sum to one for {sample_id!r}")
        predicted = max(labels, key=probabilities.__getitem__)
        records.append({
            "id": sample_id,
            "conversation_id": conversation_id,
            "emotion_label": predicted,
            "emotion_probabilities": probabilities,
        })
    return records


def profile_prediction_records(
    rows: Sequence[dict[str, Any]],
    probabilities_by_dimension: dict[str, Sequence[Sequence[float]]],
    labels_by_dimension: dict[str, Sequence[str]],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Decode per-label profile probabilities using packaged frozen thresholds."""
    dimensions = set(PROFILE_LABELS)
    if set(probabilities_by_dimension) != dimensions:
        raise ValueError("probability dimensions do not match the official schema")
    if set(labels_by_dimension) != dimensions:
        raise ValueError("label dimensions do not match the official schema")
    if set(thresholds) != dimensions:
        raise ValueError("threshold dimensions do not match the official schema")
    for dimension, allowed in PROFILE_LABELS.items():
        labels = list(labels_by_dimension[dimension])
        if not labels or len(labels) != len(set(labels)) or set(labels) != allowed:
            raise ValueError(f"profile labels do not match the official {dimension} schema")
        threshold = thresholds[dimension]
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"invalid threshold for {dimension}")
        if len(probabilities_by_dimension[dimension]) != len(rows):
            raise ValueError(f"probability count differs for {dimension}")

    records = []
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        sample_id, conversation_id = _identity(row)
        if sample_id in seen:
            raise ValueError(f"duplicate sample ID: {sample_id}")
        seen.add(sample_id)
        probability_maps: dict[str, dict[str, float]] = {}
        profile: dict[str, list[str]] = {}
        for dimension in PROFILE_LABELS:
            labels = labels_by_dimension[dimension]
            values = probabilities_by_dimension[dimension][row_index]
            if len(values) != len(labels):
                raise ValueError(f"profile probability label count differs for {dimension}")
            probability_maps[dimension] = {
                label: _validate_probability(value)
                for label, value in zip(labels, values)
            }
            profile[dimension] = [
                label for label, value in probability_maps[dimension].items()
                if value >= thresholds[dimension]
            ]
        records.append({
            "id": sample_id,
            "conversation_id": conversation_id,
            "user_profile": profile,
            "profile_probabilities": probability_maps,
        })
    return records


def _select_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _emotion_probabilities(
    rows: Sequence[dict[str, Any]], checkpoint_dir: Path, device: Any, batch_size: int,
) -> list[list[float]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), local_files_only=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir), local_files_only=True
    ).to(device).eval()
    id2label = model.config.id2label
    labels = [id2label.get(index, id2label.get(str(index))) for index in range(len(EMOTION_LABELS))]
    if labels != list(EMOTION_LABELS):
        raise ValueError("emotion checkpoint label order does not match the official schema")
    texts = [row["text"] for row in rows]
    probabilities: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size], truncation=True,
                max_length=EMOTION_MAX_LENGTH, padding=True, return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probabilities.extend(torch.softmax(logits, dim=-1).cpu().tolist())
    return probabilities


def _profile_probabilities(
    rows: Sequence[dict[str, Any]], checkpoint: EmotionProfileCheckpoint, batch_size: int,
) -> dict[str, list[list[float]]]:
    torch = checkpoint._torch
    texts = [row["text"] for row in rows]
    probabilities = {dimension: [] for dimension in checkpoint.profile_labels}
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = checkpoint.tokenizer(
                texts[start : start + batch_size], truncation=True,
                max_length=int(checkpoint.metadata["max_length"]), padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(checkpoint.device) for key, value in encoded.items()}
            pooled = checkpoint.encoder(**encoded, return_dict=True).pooler_output
            for dimension, head in checkpoint.profile_heads.items():
                probabilities[dimension].extend(torch.sigmoid(head(pooled)).cpu().tolist())
    return probabilities


def _write_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("emotion", "profile"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    rows = read_rows(args.input)
    import torch

    device = _select_device(torch, args.device)
    checkpoint_dir = args.checkpoint_dir.resolve()
    if args.kind == "emotion":
        probabilities = _emotion_probabilities(rows, checkpoint_dir, device, args.batch_size)
        records = emotion_prediction_records(rows, probabilities, EMOTION_LABELS)
    else:
        checkpoint = EmotionProfileCheckpoint(checkpoint_dir, device=str(device))
        probabilities = _profile_probabilities(rows, checkpoint, args.batch_size)
        records = profile_prediction_records(
            rows, probabilities, checkpoint.profile_labels, checkpoint.thresholds
        )
    _write_records(args.output.resolve(), records)
    print(json.dumps({
        "kind": args.kind,
        "rows": len(records),
        "device": str(device),
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
