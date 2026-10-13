#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.classification.emotion_encoder import (  # noqa: E402
    EMOTION_LABELS,
    build_emotion_examples,
    classification_metrics,
)
from src.common.io import read_jsonl  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def _load_optional_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install the optional training dependencies with "
            "python -m pip install -r requirements-emotion.txt"
        ) from error
    return torch, transformers, (AutoModelForSequenceClassification, AutoTokenizer)


def _encode_dataset(
    tokenizer: Any,
    examples: list[dict[str, Any]],
    label_to_id: dict[str, int],
    max_length: int,
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        [example["text"] for example in examples],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    return [
        {
            **{key: values[index] for key, values in encoded.items()},
            "labels": label_to_id[example["label"]],
        }
        for index, example in enumerate(examples)
    ]


def _collate(tokenizer: Any, torch: Any, features: list[dict[str, Any]]) -> dict[str, Any]:
    labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
    inputs = [{key: value for key, value in feature.items() if key != "labels"} for feature in features]
    batch = tokenizer.pad(inputs, padding=True, return_tensors="pt")
    batch["labels"] = labels
    return batch


def _evaluate(
    model: Any,
    loader: Any,
    device: Any,
    torch: Any,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    expected: list[str] = []
    predicted: list[str] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.argmax(dim=-1).cpu().tolist()
            target_ids = batch["labels"].cpu().tolist()
            predicted.extend(labels[label_id] for label_id in logits)
            expected.extend(labels[label_id] for label_id in target_ids)
    return classification_metrics(expected, predicted, labels)


def _render_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# 中文 encoder 情感分类 baseline",
        "",
        f"**状态：{report['status']}。**",
        "",
        f"模型：[`{report['model']['name']}`]({report['model']['card_url']})，",
        f"revision `{report['model']['revision']}`。",
        f"设备：`{report['device']}`；seed：`{report['seed']}`；",
        f"epoch：{report['config']['epochs']}；max length：{report['config']['max_length']}。",
        f"训练集：{report['data']['train_used']}/{report['data']['train_total']} 个 prediction points；",
        f"验证集：{report['data']['validation_used']}/{report['data']['validation_total']} 个 prediction points。",
        "",
        f"按验证集 Accuracy 选择 epoch {report['selected_epoch']} 的 checkpoint。",
        f"验证 Accuracy：**{metrics['accuracy']:.2%}**；Macro-F1：**{metrics['macro_f1']:.4f}**。",
        (
            f"多数类 baseline Accuracy：{report['baseline_comparison']['majority_accuracy']:.2%}；"
            f"提升 {report['baseline_comparison']['accuracy_delta'] * 100:.2f} 个百分点。"
        ),
        "",
        "| 情感类别 | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in metrics["labels"]:
        value = metrics["per_class"][label]
        lines.append(
            f"| {label} | {value['precision']:.4f} | {value['recall']:.4f} "
            f"| {value['f1']:.4f} | {value['support']} |"
        )
    lines.extend(
        [
            "",
            "数据 SHA-256：",
            "",
            f"- train：`{report['data']['train_sha256']}`",
            f"- val：`{report['data']['validation_sha256']}`",
            "",
            (
                f"本机 checkpoint：`{report['checkpoint']['file']}` "
                f"（{report['checkpoint']['bytes']} bytes，"
                f"SHA-256 `{report['checkpoint']['sha256']}`）。"
            ),
            "混淆矩阵、逐 epoch 指标和完整配置见同名 JSON。模型权重保存在本机忽略目录，不提交到 Git。",
            "",
            "复现命令：",
            "",
            "```bash",
            "python -m pip install -r requirements-emotion.txt",
            "python scripts/train_emotion_encoder.py \\",
            "  --train data/raw/训练-验证-数据集/train/train_public.jsonl \\",
            "  --val data/raw/训练-验证-数据集/val/val_public.jsonl",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch, transformers, model_classes = _load_optional_dependencies()
    AutoModelForSequenceClassification, AutoTokenizer = model_classes
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train_rows = read_jsonl(args.train)
    validation_rows = read_jsonl(args.val)
    train_examples = build_emotion_examples(train_rows)
    validation_examples = build_emotion_examples(validation_rows)

    train_ids = {example["conversation_id"] for example in train_examples}
    validation_ids = {example["conversation_id"] for example in validation_examples}
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(f"train/validation conversation_id overlap: {len(overlap)}")

    train_total = len(train_examples)
    validation_total = len(validation_examples)
    for option_name, value in (
        ("max_train_samples", args.max_train_samples),
        ("max_validation_samples", args.max_validation_samples),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{option_name} must be positive")
    if args.max_train_samples:
        train_examples = train_examples[: args.max_train_samples]
    if args.max_validation_samples:
        validation_examples = validation_examples[: args.max_validation_samples]
    if not train_examples or not validation_examples:
        raise ValueError("training and validation sets must both contain samples")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(torch, args.device)

    model_name = str(config["model_name"])
    model_revision = str(config["model_revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    tokenizer.truncation_side = "left"
    label_to_id = {label: index for index, label in enumerate(EMOTION_LABELS)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=model_revision,
        num_labels=len(EMOTION_LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
        local_files_only=args.local_files_only,
    ).to(device)

    train_encoded = _encode_dataset(
        tokenizer, train_examples, label_to_id, int(config["max_length"])
    )
    validation_encoded = _encode_dataset(
        tokenizer, validation_examples, label_to_id, int(config["max_length"])
    )
    collate = lambda features: _collate(tokenizer, torch, features)
    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_encoded,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_encoded,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=collate,
    )

    epochs = int(config["epochs"])
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * float(config["warmup_ratio"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    def learning_rate_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    selected_epoch = 0
    selected_metrics: dict[str, Any] = {}
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            if step % 100 == 0 or step == len(train_loader):
                mean_loss = total_loss / step
                print(
                    f"epoch={epoch + 1}/{epochs} step={step}/{len(train_loader)} "
                    f"train_loss={mean_loss:.4f} device={device}",
                    flush=True,
                )
        validation_metrics = _evaluate(
            model, validation_loader, device, torch, EMOTION_LABELS
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(total_loss / max(len(train_loader), 1), 6),
                "validation": validation_metrics,
            }
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            selected_epoch = epoch + 1
            selected_metrics = validation_metrics
            model.save_pretrained(args.output_dir)
        print(
            f"epoch={epoch + 1}/{epochs} validation_accuracy="
            f"{validation_metrics['accuracy']:.6f} "
            f"validation_macro_f1={validation_metrics['macro_f1']:.6f}",
            flush=True,
        )

    tokenizer.save_pretrained(args.output_dir)
    resolved_revision = getattr(model.config, "_commit_hash", None) or model_revision
    checkpoint_files = sorted(args.output_dir.glob("*.safetensors"))
    if len(checkpoint_files) != 1:
        raise RuntimeError("expected exactly one safetensors checkpoint")
    checkpoint_path = checkpoint_files[0]
    used_all_data = len(train_examples) == train_total and len(validation_examples) == validation_total
    report = {
        "version": "emotion_encoder_baseline_v1",
        "status": "Integration Gate" if used_all_data else "Exploratory sample only",
        "model": {
            "name": model_name,
            "revision": resolved_revision,
            "card_url": f"https://huggingface.co/{model_name}",
        },
        "device": str(device),
        "seed": seed,
        "config": config,
        "runtime": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "training_seconds": round(time.perf_counter() - started, 3),
        },
        "data": {
            "train_total": train_total,
            "train_used": len(train_examples),
            "validation_total": validation_total,
            "validation_used": len(validation_examples),
            "conversation_id_overlap": 0,
            "train_sha256": _sha256(args.train),
            "validation_sha256": _sha256(args.val),
        },
        "history": history,
        "selected_epoch": selected_epoch,
        "metrics": selected_metrics,
        "checkpoint_directory": _portable_path(args.output_dir),
        "checkpoint": {
            "file": f"{_portable_path(args.output_dir)}/{checkpoint_path.name}",
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256(checkpoint_path),
        },
        "baseline_comparison": _majority_comparison(
            args.majority_report, validation_total, selected_metrics["accuracy"]
        ),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report_md.write_text(_render_report(report), encoding="utf-8")
    print(
        f"PASS: device={device} train={len(train_examples)}/{train_total} "
        f"val={len(validation_examples)}/{validation_total} "
        f"accuracy={report['metrics']['accuracy']:.6f} "
        f"macro_f1={report['metrics']['macro_f1']:.6f}",
        flush=True,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Chinese encoder emotion baseline")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/emotion_encoder.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "weights/emotion_baseline")
    parser.add_argument(
        "--report-json", type=Path, default=PROJECT / "reports/emotion_encoder_baseline.json"
    )
    parser.add_argument(
        "--report-md", type=Path, default=PROJECT / "reports/EMOTION_ENCODER_BASELINE.md"
    )
    parser.add_argument(
        "--majority-report", type=Path, default=PROJECT / "reports/classification_baseline.json"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT.resolve()).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


def _majority_comparison(
    report_path: Path, validation_samples: int, encoder_accuracy: float
) -> dict[str, float]:
    baseline = json.loads(report_path.read_text(encoding="utf-8"))
    if baseline.get("validation_samples") != validation_samples:
        raise ValueError("majority baseline and encoder validation sample counts differ")
    baseline_accuracy = float(baseline["metrics"]["emotion_accuracy"])
    return {
        "majority_accuracy": baseline_accuracy,
        "accuracy_delta": round(encoder_accuracy - baseline_accuracy, 6),
    }


if __name__ == "__main__":
    run(parse_args())
