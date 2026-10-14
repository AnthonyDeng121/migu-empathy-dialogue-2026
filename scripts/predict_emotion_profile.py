#!/usr/bin/env python3
"""Run the local C classifier on official eval-input JSONL rows."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.train_emotion_profile import (  # noqa: E402
    _load_dependencies,
    _make_model,
    _select_device,
)
from src.classification.emotion_profile import (  # noqa: E402
    EMOTION_LABELS,
    PROFILE_LABELS_BY_DIMENSION,
    build_inference_context,
    decode_emotion_profile_scores,
)


def _metadata(checkpoint_dir: Path) -> dict[str, Any]:
    path = checkpoint_dir / "checkpoint_metadata.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("emotion_labels") != list(EMOTION_LABELS):
        raise ValueError("checkpoint emotion label order is incompatible")
    expected_profile_labels = {
        dimension: list(labels)
        for dimension, labels in PROFILE_LABELS_BY_DIMENSION.items()
    }
    if value.get("profile_labels") != expected_profile_labels:
        raise ValueError("checkpoint profile label order is incompatible")
    return value


def _predict_group(
    rows: list[dict[str, Any]], model: Any, tokenizer: Any, torch: Any,
    device: Any, max_length: int, threshold: float,
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        [build_inference_context(row) for row in rows],
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    with torch.inference_mode():
        output = model(**inputs)
        emotion_scores = output["emotion_logits"].detach().cpu().tolist()
        profile_scores = {
            dimension: torch.sigmoid(logits).detach().cpu().tolist()
            for dimension, logits in output["profile_logits"].items()
        }
    predictions = []
    for index, row in enumerate(rows):
        decoded = decode_emotion_profile_scores(
            emotion_scores[index],
            {dimension: scores[index] for dimension, scores in profile_scores.items()},
            threshold=threshold,
        )
        predictions.append({"id": row["id"], **decoded})
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    checkpoint = args.checkpoint_dir.resolve()
    metadata = _metadata(checkpoint)
    torch, (AutoModel, AutoTokenizer), (load_file, _) = _load_dependencies()
    device = _select_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    encoder = AutoModel.from_pretrained(checkpoint / "encoder", local_files_only=True)
    model = _make_model(torch, encoder)
    incompatibility = model.load_state_dict(
        load_file(str(checkpoint / "heads.safetensors")), strict=False
    )
    if incompatibility.unexpected_keys or any(
        not name.startswith("encoder.") for name in incompatibility.missing_keys
    ):
        raise RuntimeError("checkpoint head state is incompatible with the classifier")
    model.to(device).eval()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    count = 0
    try:
        with args.input.open("r", encoding="utf-8") as source, os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as destination:
            pending: list[dict[str, Any]] = []
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    build_inference_context(row)
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(f"{args.input}:{line_number}: {error}") from error
                pending.append(row)
                if len(pending) == args.batch_size:
                    for prediction in _predict_group(
                        pending, model, tokenizer, torch, device,
                        int(metadata["max_length"]), float(metadata["profile_threshold"]),
                    ):
                        destination.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += len(pending)
                    pending.clear()
            if pending:
                for prediction in _predict_group(
                    pending, model, tokenizer, torch, device,
                    int(metadata["max_length"]), float(metadata["profile_threshold"]),
                ):
                    destination.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += len(pending)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    if count == 0:
        raise ValueError("inference input contains no rows")
    print(json.dumps({"predictions": count, "output": str(output_path), "device": str(device)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
