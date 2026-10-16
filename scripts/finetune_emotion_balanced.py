#!/usr/bin/env python3
"""Fine-tune the emotion baseline with moderate class-balanced loss on B1 v1."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.classification.emotion_encoder import classification_metrics  # noqa: E402
from src.data.schema import EMOTIONS  # noqa: E402

LABELS = tuple(sorted(EMOTIONS))
SPEAKERS = {"user": "用户", "assistant": "助手"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, expected_split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = {
        "sample_id", "conversation_id", "target_user_turn_id", "source_split",
        "context_turns", "context_text", "emotion_label", "user_profile",
    }
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != required or row["source_split"] != expected_split:
                raise ValueError(f"{path}:{line_number}: invalid B1 fields or split")
            if row["sample_id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id")
            seen.add(row["sample_id"])
            turns = row["context_turns"]
            if not turns or turns[-1]["role"] != "user" or turns[-1]["turn_id"] != row["target_user_turn_id"]:
                raise ValueError(f"{path}:{line_number}: context is not cut off at target user turn")
            emotion = row["emotion_label"]
            if emotion not in LABELS:
                raise ValueError(f"{path}:{line_number}: invalid emotion label")
            rows.append({
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
                "text": "\n".join(
                    f"{SPEAKERS[turn['role']]}：{turn['content']}" for turn in turns
                ),
                "label": emotion,
            })
    return rows


def _metrics(model: Any, rows: list[dict[str, Any]], tokenizer: Any, torch: Any, device: Any) -> dict[str, Any]:
    model.eval()
    expected: list[str] = []
    predicted: list[str] = []
    batch_size = 32
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            tokens = tokenizer(
                [row["text"] for row in batch], truncation=True, max_length=256,
                padding=True, return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            guesses = model(**tokens).logits.argmax(dim=-1).cpu().tolist()
            predicted.extend(LABELS[index] for index in guesses)
            expected.extend(row["label"] for row in batch)
    return classification_metrics(expected, predicted, LABELS)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=PROJECT / "data/processed/b1_full_v1/classification_train.jsonl")
    parser.add_argument("--val", type=Path, default=PROJECT / "data/processed/b1_full_v1/classification_val.jsonl")
    parser.add_argument("--base-checkpoint", type=Path, default=PROJECT / "weights/emotion_baseline")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "weights/emotion_balanced_b1_20261015")
    parser.add_argument("--report", type=Path, default=PROJECT / "results/emotion_balanced_b1_20261015.json")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--class-weight-exponent", type=float, default=0.5,
        help="Power applied to inverse-frequency class weights (0=unweighted, 0.5=square-root).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if not 0.0 <= args.class_weight_exponent <= 1.0:
        parser.error("--class-weight-exponent must be between 0 and 1")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install requirements-emotion.txt before training") from error

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
    else:
        device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    train = _read(args.train, "train")
    val = _read(args.val, "val")
    overlap = {row["conversation_id"] for row in train} & {row["conversation_id"] for row in val}
    if overlap:
        raise ValueError(f"conversation split leakage: {len(overlap)}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "mps":
        torch.mps.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_checkpoint, local_files_only=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_checkpoint, local_files_only=True
    ).to(device)
    initial_metrics = _metrics(model, val, tokenizer, torch, device)

    label_to_id = {label: index for index, label in enumerate(LABELS)}
    train_counts = torch.zeros(len(LABELS), dtype=torch.float32)
    for row in train:
        train_counts[label_to_id[row["label"]]] += 1
    # A configurable power allows one-variable comparison of long-tail emphasis.
    class_weights = (
        train_counts.sum() / (len(LABELS) * train_counts.clamp(min=1))
    ).pow(args.class_weight_exponent)
    class_weights = (class_weights / class_weights.mean()).clamp(min=0.35, max=4.0).to(device)

    encoded = tokenizer(
        [row["text"] for row in train], truncation=True, max_length=256,
        padding="max_length",
    )
    features = [
        {key: encoded[key][index] for key in encoded}
        for index in range(len(train))
    ]
    batch_size = args.batch_size
    generator = torch.Generator().manual_seed(args.seed)
    labels_by_sample = torch.tensor(
        [label_to_id[row["label"]] for row in train], dtype=torch.long
    )
    # Keep labels attached to the shuffled examples during collation.
    features = [
        {**feature, "labels": int(labels_by_sample[index])}
        for index, feature in enumerate(features)
    ]
    loader = torch.utils.data.DataLoader(
        features, batch_size=batch_size, shuffle=True, generator=generator,
        collate_fn=lambda batch: {
            **tokenizer.pad(
                [{key: value for key, value in row.items() if key != "labels"} for row in batch],
                padding=True, return_tensors="pt",
            ),
            "labels": torch.tensor([row["labels"] for row in batch], dtype=torch.long),
        },
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(loader)
    warmup_steps = max(1, round(total_steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: step / warmup_steps if step < warmup_steps else max(
            0.0, (total_steps - step) / max(total_steps - warmup_steps, 1)
        ),
    )
    started = time.perf_counter()
    losses: list[float] = []
    model.train()
    for step, batch in enumerate(loader, 1):
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**{key: value for key, value in batch.items() if key != "labels"})
        loss = torch.nn.functional.cross_entropy(
            output.logits, batch["labels"], weight=class_weights
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite balanced emotion loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if step % 100 == 0 or step == total_steps:
            print(f"step={step}/{total_steps} loss={sum(losses[-100:]) / len(losses[-100:]):.4f}", flush=True)

    final_metrics = _metrics(model, val, tokenizer, torch, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    report = {
        "status": "full B1 v1 one-epoch class-balanced fine-tune",
        "device": str(device),
        "runtime": {
            "python": __import__("platform").python_version(),
            "torch": str(torch.__version__),
            "transformers": __import__("transformers").__version__,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "model": {
            "initial_checkpoint": str(args.base_checkpoint),
            "base_checkpoint_sha256": _sha256(args.base_checkpoint / "model.safetensors"),
            "output_checkpoint": str(args.output_dir),
        },
        "config": {
            "seed": args.seed, "epochs": 1, "batch_size": batch_size,
            "max_length": 256, "learning_rate": args.learning_rate,
            "loss": (
                "cross entropy with normalized inverse-frequency weights raised to "
                f"exponent {args.class_weight_exponent}, clipped to [0.35, 4.0]"
            ),
            "class_weight_exponent": args.class_weight_exponent,
        },
        "data": {
            "train_path": str(args.train), "train_sha256": _sha256(args.train),
            "train_samples": len(train), "train_conversations": len({r["conversation_id"] for r in train}),
            "validation_path": str(args.val), "validation_sha256": _sha256(args.val),
            "validation_samples": len(val), "validation_conversations": len({r["conversation_id"] for r in val}),
            "conversation_overlap": 0,
            "class_counts": {label: int(train_counts[i].item()) for i, label in enumerate(LABELS)},
            "class_weights": {label: round(float(class_weights[i].item()), 6) for i, label in enumerate(LABELS)},
        },
        "metrics": {
            "before": initial_metrics,
            "after": final_metrics,
            "accuracy_delta": round(final_metrics["accuracy"] - initial_metrics["accuracy"], 6),
            "macro_f1_delta": round(final_metrics["macro_f1"] - initial_metrics["macro_f1"], 6),
        },
        "training": {
            "optimizer_steps": len(losses), "first_loss": round(losses[0], 6),
            "last_loss": round(losses[-1], 6), "all_losses_finite": all(math.isfinite(x) for x in losses),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "elapsed_seconds": report["runtime"]["elapsed_seconds"],
        "before": {"accuracy": initial_metrics["accuracy"], "macro_f1": initial_metrics["macro_f1"]},
        "after": {"accuracy": final_metrics["accuracy"], "macro_f1": final_metrics["macro_f1"]},
        "accuracy_delta": report["metrics"]["accuracy_delta"],
        "macro_f1_delta": report["metrics"]["macro_f1_delta"],
        "checkpoint": str(args.output_dir), "report": str(args.report),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
