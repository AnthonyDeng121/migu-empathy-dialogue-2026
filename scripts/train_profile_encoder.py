"""Fine-tune a profile-only encoder using a frozen B1 conversation split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.train_profile_heads import (  # noqa: E402
    EMOTION_LABELS,
    PROFILE_LABELS_BY_DIMENSION,
    profile_metrics,
    profile_targets,
    read_examples,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT.resolve()).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


def partition_by_manifest(
    examples: Sequence[dict[str, Any]], manifest_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and apply the frozen fit/calibration manifest by sample ID."""
    by_id: dict[str, dict[str, Any]] = {}
    for example in examples:
        sample_id = example["sample_id"]
        if sample_id in by_id:
            raise ValueError(f"duplicate training sample ID: {sample_id}")
        by_id[sample_id] = example

    assignments: dict[str, str] = {}
    conversations: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{manifest_path}:{line_number}: malformed JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}:{line_number}: expected an object")
            sample_id = row.get("sample_id")
            conversation_id = row.get("conversation_id")
            subset = row.get("subset")
            if row.get("source_split") != "train" or subset not in {"fit", "calibration"}:
                raise ValueError(f"{manifest_path}:{line_number}: invalid split assignment")
            if not isinstance(sample_id, str) or not isinstance(conversation_id, str):
                raise ValueError(f"{manifest_path}:{line_number}: invalid sample/conversation ID")
            if sample_id in assignments:
                raise ValueError(f"{manifest_path}:{line_number}: duplicate sample ID {sample_id}")
            example = by_id.get(sample_id)
            if example is None or example["conversation_id"] != conversation_id:
                raise ValueError(f"{manifest_path}:{line_number}: sample ID or conversation mismatch")
            assignments[sample_id] = subset
            conversations.setdefault(conversation_id, subset)
            if conversations[conversation_id] != subset:
                raise ValueError("conversation appears in both fit and calibration")

    if set(assignments) != set(by_id):
        raise ValueError("manifest and training sample IDs differ")
    fit = [row for row in examples if assignments[row["sample_id"]] == "fit"]
    calibration = [row for row in examples if assignments[row["sample_id"]] == "calibration"]
    if not fit or not calibration:
        raise ValueError("fit and calibration partitions must both be non-empty")
    return fit, calibration


def exact_thresholds(probabilities: dict[str, Any], targets: dict[str, Any], torch: Any) -> dict[str, float]:
    """Maximize each dimension's exact accuracy on training-only calibration rows.

    Threshold 1.0 is the empty-prediction fallback. Probabilities are clipped
    below 1 so the fallback remains empty and complies with the packaged loader.
    Ties prefer the higher threshold.
    """
    thresholds: dict[str, float] = {}
    for dimension in PROFILE_LABELS_BY_DIMENSION:
        target = targets[dimension].bool()
        best_score = -1.0
        best_threshold = 0.0
        for step in range(101):
            threshold = step / 100
            guess = probabilities[dimension] >= threshold
            score = float((target == guess).all(dim=1).float().mean().item())
            if score > best_score or (score == best_score and threshold > best_threshold):
                best_score = score
                best_threshold = threshold
        thresholds[dimension] = best_threshold
    return thresholds


def _predict(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    *,
    torch: Any,
    device: Any,
    max_length: int,
    batch_size: int,
    name: str,
) -> dict[str, Any]:
    chunks = {dimension: [] for dimension in PROFILE_LABELS_BY_DIMENSION}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [row["text"] for row in batch],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**inputs)
            for dimension in PROFILE_LABELS_BY_DIMENSION:
                probs = torch.sigmoid(logits[dimension]).float().clamp_(max=1.0 - 1e-7)
                chunks[dimension].append(probs.cpu())
            done = min(start + len(batch), len(rows))
            if done == len(rows) or done % 1024 < len(batch):
                print(f"predict {name}: {done}/{len(rows)}", flush=True)
    return {dimension: torch.cat(values) for dimension, values in chunks.items()}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _probability_rows(
    examples: Sequence[dict[str, Any]], probabilities: dict[str, Any],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    values = {dimension: probabilities[dimension].tolist() for dimension in PROFILE_LABELS_BY_DIMENSION}
    for index, row in enumerate(examples):
        dimension_probabilities = {
            dimension: {
                label: float(probability)
                for label, probability in zip(PROFILE_LABELS_BY_DIMENSION[dimension], values[dimension][index])
            }
            for dimension in PROFILE_LABELS_BY_DIMENSION
        }
        profile = {
            dimension: [label for label, probability in dimension_probabilities[dimension].items()
                        if probability >= thresholds[dimension]]
            for dimension in PROFILE_LABELS_BY_DIMENSION
        }
        output.append({
            "id": row["sample_id"],
            "conversation_id": row["conversation_id"],
            "user_profile": profile,
            "profile_probabilities": dimension_probabilities,
        })
    return output


def _mean_micro_f1(metrics: dict[str, Any]) -> float:
    return sum(metrics["per_dimension"][dimension]["micro_f1"]
               for dimension in PROFILE_LABELS_BY_DIMENSION) / len(PROFILE_LABELS_BY_DIMENSION)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=PROJECT / "data/processed/b1_full_v1/classification_train.jsonl")
    parser.add_argument("--val", type=Path, default=PROJECT / "data/processed/b1_full_v1/classification_val.jsonl")
    parser.add_argument("--split-manifest", type=Path, default=PROJECT / "reports/b_t1_train_internal_split.jsonl")
    parser.add_argument("--base-model", type=Path, required=True, help="Local generic pretrained MacBERT snapshot")
    parser.add_argument("--base-model-id", default="hfl/chinese-macbert-base")
    parser.add_argument("--base-model-revision", default="a986e004d2a7f2a1c2f5a3edef4e20604a974ed1")
    parser.add_argument("--emotion-checkpoint", type=Path, default=PROJECT / "weights/emotion_longtail075_b1_20261016")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--positive-weight-exponent", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--max-steps", type=int, help="Bounded smoke run; never a full-training result")
    parser.add_argument("--smoke-samples", type=int, default=16)
    args = parser.parse_args()
    if args.max_length < 1 or args.batch_size < 1 or args.epochs < 1:
        parser.error("max length, batch size and epochs must be positive")
    if not 0 <= args.positive_weight_exponent <= 1:
        parser.error("positive weight exponent must be between 0 and 1")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max steps must be positive")
    if args.device == "cuda":
        parser.error("this local-training entry does not support CUDA in this run")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "mps":
        torch.mps.manual_seed(args.seed)

    train = read_examples(args.train, "train")
    val = read_examples(args.val, "val")
    fit_rows, calibration_rows = partition_by_manifest(train, args.split_manifest)
    if {row["conversation_id"] for row in train} & {row["conversation_id"] for row in val}:
        raise ValueError("B1 train/validation conversations overlap")

    full_run = args.max_steps is None
    if not full_run:
        fit_rows = fit_rows[: max(args.smoke_samples, args.batch_size * args.max_steps)]
        calibration_rows = calibration_rows[: args.smoke_samples]
        val = val[: args.smoke_samples]
    if not fit_rows or not calibration_rows or not val:
        raise ValueError("smoke sample limit leaves an empty fit/calibration/validation split")

    try:
        split_manifest_reference = args.split_manifest.resolve().relative_to(PROJECT.resolve()).as_posix()
    except ValueError:
        split_manifest_reference = args.split_manifest.name
    try:
        emotion_checkpoint_reference = args.emotion_checkpoint.resolve().relative_to(PROJECT.resolve()).as_posix()
    except ValueError:
        emotion_checkpoint_reference = args.emotion_checkpoint.name

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.truncation_side = "left"
    encoder = AutoModel.from_pretrained(args.base_model, local_files_only=True)
    hidden_size = int(encoder.config.hidden_size)

    class ProfileOnlyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.profile_heads = torch.nn.ModuleDict({
                dimension: torch.nn.Linear(hidden_size, len(labels))
                for dimension, labels in PROFILE_LABELS_BY_DIMENSION.items()
            })

        def forward(self, **inputs: Any) -> dict[str, Any]:
            pooled = self.encoder(**inputs, return_dict=True).pooler_output
            return {dimension: head(pooled) for dimension, head in self.profile_heads.items()}

    model = ProfileOnlyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    fit_targets = profile_targets(fit_rows, torch)
    calibration_targets = profile_targets(calibration_rows, torch)
    val_targets = profile_targets(val, torch)
    pos_weights: dict[str, Any] = {}
    for dimension in PROFILE_LABELS_BY_DIMENSION:
        positive = fit_targets[dimension].sum(dim=0)
        negative = len(fit_rows) - positive
        pos_weights[dimension] = (negative / positive.clamp(min=1)).pow(
            args.positive_weight_exponent
        ).clamp(min=1.0, max=8.0).to(device)
    fit_targets = {dimension: target.to(device) for dimension, target in fit_targets.items()}
    losses: list[float] = []
    optimizer_steps = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(fit_rows)).tolist()
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            batch = [fit_rows[index] for index in indices]
            encoded = tokenizer(
                [row["text"] for row in batch], truncation=True,
                max_length=args.max_length, padding=True, return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in encoded.items()}
            targets = {dimension: fit_targets[dimension][indices] for dimension in PROFILE_LABELS_BY_DIMENSION}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs)
            dimension_losses = [
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[dimension], targets[dimension], pos_weight=pos_weights[dimension]
                )
                for dimension in PROFILE_LABELS_BY_DIMENSION
            ]
            loss = sum(dimension_losses) / len(dimension_losses)
            if not torch.isfinite(loss):
                raise RuntimeError("profile-only training produced a non-finite loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm) or float(grad_norm.detach().cpu()) <= 0:
                raise RuntimeError("profile-only training produced invalid gradients")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            optimizer_steps += 1
            if optimizer_steps % 25 == 0 or optimizer_steps == 1:
                print(f"epoch={epoch}/{args.epochs} step={optimizer_steps} loss={losses[-1]:.6f}", flush=True)
            if args.max_steps is not None and optimizer_steps >= args.max_steps:
                break
        if args.max_steps is not None and optimizer_steps >= args.max_steps:
            break

    elapsed = time.perf_counter() - started
    calibration_probabilities = _predict(
        model, tokenizer, calibration_rows, torch=torch, device=device,
        max_length=args.max_length, batch_size=args.batch_size, name="calibration",
    )
    thresholds = exact_thresholds(calibration_probabilities, calibration_targets, torch)
    val_probabilities = _predict(
        model, tokenizer, val, torch=torch, device=device,
        max_length=args.max_length, batch_size=args.batch_size, name="validation",
    )
    val_prediction_rows = _probability_rows(val, val_probabilities, thresholds)
    calibration_prediction_rows = _probability_rows(calibration_rows, calibration_probabilities, thresholds)
    predicted_targets = {
        dimension: val_probabilities[dimension] >= thresholds[dimension]
        for dimension in PROFILE_LABELS_BY_DIMENSION
    }
    calibration_predicted_targets = {
        dimension: calibration_probabilities[dimension] >= thresholds[dimension]
        for dimension in PROFILE_LABELS_BY_DIMENSION
    }
    validation_metrics = profile_metrics(val_targets, predicted_targets)
    calibration_metrics = profile_metrics(calibration_targets, calibration_predicted_targets)
    empty_validation = profile_metrics(
        val_targets, {dimension: torch.zeros_like(val_targets[dimension]) for dimension in PROFILE_LABELS_BY_DIMENSION}
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(args.output_dir / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(
        args.emotion_checkpoint, local_files_only=True
    )
    emotion_labels = [emotion_model.config.id2label[index] for index in range(emotion_model.config.num_labels)]
    if emotion_labels != list(EMOTION_LABELS):
        raise ValueError("emotion checkpoint label order differs from the official schema")
    classifier_state = emotion_model.classifier.state_dict()
    head_state = {
        "emotion_head.weight": classifier_state["weight"].detach().cpu().contiguous(),
        "emotion_head.bias": classifier_state["bias"].detach().cpu().contiguous(),
    }
    for dimension, head in model.profile_heads.items():
        for key, value in head.state_dict().items():
            head_state[f"profile_heads.{dimension}.{key}"] = value.detach().cpu().contiguous()
    save_file(head_state, str(args.output_dir / "heads.safetensors"))
    metadata = {
        "format_version": 2,
        "base_model": args.base_model_id,
        "base_model_revision": args.base_model_revision,
        "profile_training_objective": "mean of three profile-only weighted BCE losses; encoder trainable",
        "emotion_labels": list(EMOTION_LABELS),
        "profile_labels": {dimension: list(labels) for dimension, labels in PROFILE_LABELS_BY_DIMENSION.items()},
        "profile_thresholds": thresholds,
        "profile_threshold_objective": "per-dimension exact match on training-only calibration split",
        "pooling": "BertModel.pooler_output",
        "max_length": args.max_length,
        "emotion_head_role": "format-compatibility only; independent profile mode ignores this emotion output",
        "calibration_split_manifest": split_manifest_reference,
    }
    (args.output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_jsonl(args.predictions, val_prediction_rows)
    calibration_output = args.predictions.with_name("calibration_probabilities.jsonl")
    _write_jsonl(calibration_output, calibration_prediction_rows)

    # Verify the real packaged loader consumes the saved checkpoint and reproduces
    # probabilities for a small fixed validation prefix.
    from src.classification.emotion_profile_checkpoint import EmotionProfileCheckpoint
    packaged = EmotionProfileCheckpoint(args.output_dir, str(device))
    reloaded = packaged.predict_texts([row["text"] for row in val[: min(4, len(val))]])
    for index, result in enumerate(reloaded):
        if result["user_profile"] != val_prediction_rows[index]["user_profile"]:
            raise RuntimeError("packaged checkpoint prediction differs from training output")

    base_weight_path = args.base_model / "pytorch_model.bin"
    if not base_weight_path.is_file():
        base_weight_path = args.base_model / "model.safetensors"
    report = {
        "status": "full profile v1 training" if full_run else "exploratory bounded smoke only",
        "decision": "pending evaluation" if full_run else "not eligible for model selection",
        "device": str(device),
        "runtime": {
            "python": platform.python_version(), "system": platform.system(),
            "torch": str(torch.__version__), "transformers": __import__("transformers").__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "data": {
            "train_sha256": sha256_file(args.train), "validation_sha256": sha256_file(args.val),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "train_rows": len(train), "fit_rows": len(fit_rows),
            "calibration_rows": len(calibration_rows), "validation_rows": len(val),
            "fit_calibration_conversation_overlap": 0,
            "train_validation_conversation_overlap": 0,
            "smoke_subset": not full_run,
        },
        "training": {
            "seed": args.seed, "epochs_requested": args.epochs, "max_length": args.max_length,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "positive_weight_exponent": args.positive_weight_exponent,
            "optimizer_steps": optimizer_steps, "elapsed_seconds": round(elapsed, 3),
            "first_loss": losses[0], "last_loss": losses[-1],
            "all_losses_finite": all(math.isfinite(value) for value in losses),
            "encoder_trainable": True,
            "loss": "mean of three profile-only weighted binary cross-entropies",
        },
        "initialization": {
            "base_model_id": args.base_model_id,
            "base_model_revision": args.base_model_revision,
            "base_model_weights_sha256": sha256_file(base_weight_path),
            "base_model_config_sha256": sha256_file(args.base_model / "config.json"),
            "profile_encoder_initialized_from_task_finetuned_checkpoint": False,
            "emotion_head_source": emotion_checkpoint_reference,
            "emotion_head_ignored_by_independent_profile_mode": True,
        },
        "calibration": {
            "threshold_source": "B T1 frozen training-only conversation-disjoint calibration partition",
            "validation_used_for_threshold_selection": False,
            "thresholds": thresholds,
            "metrics": calibration_metrics,
            "empty_baseline": profile_metrics(
                calibration_targets,
                {dimension: torch.zeros_like(calibration_targets[dimension]) for dimension in PROFILE_LABELS_BY_DIMENSION},
            ),
        },
        "validation": validation_metrics,
        "empty_set_baseline": empty_validation,
        "macro_micro_f1": {
            "validation": _mean_micro_f1(validation_metrics),
            "calibration": _mean_micro_f1(calibration_metrics),
        },
        "prediction_output": _portable_path(args.predictions),
        "calibration_probability_output": _portable_path(calibration_output),
        "checkpoint": _portable_path(args.output_dir),
        "packaged_reload_check": {
            "passed": True,
            "rows_checked": len(reloaded),
            "loader": "src.classification.emotion_profile_checkpoint.EmotionProfileCheckpoint",
        },
    }
    _write_json(args.report, report)
    print(json.dumps({
        "status": report["status"], "steps": optimizer_steps,
        "thresholds": thresholds,
        "validation_macro_exact": validation_metrics["profile_macro_exact_match_accuracy"],
        "empty_macro_exact": empty_validation["profile_macro_exact_match_accuracy"],
        "validation_macro_micro_f1": _mean_micro_f1(validation_metrics),
        "report": _portable_path(args.report),
        "checkpoint": _portable_path(args.output_dir),
        "predictions": _portable_path(args.predictions),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
