#!/usr/bin/env python3
"""Train profile heads on top of the stronger standalone emotion encoder.

The encoder and emotion head stay frozen. This avoids the negative transfer seen
in the first shared-loss experiment while allowing profile labels to use the same
MacBERT representation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT))

from src.data.schema import PROFILE_LABELS  # noqa: E402

EMOTION_LABELS = tuple(sorted({
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
    "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
    "shame", "helplessness",
}))
PROFILE_LABELS_BY_DIMENSION = {
    name: tuple(sorted(labels)) for name, labels in PROFILE_LABELS.items()
}
SPEAKER_NAMES = {"user": "用户", "assistant": "助手"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_examples(path: Path, split: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
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
            if set(row) != required or row["source_split"] != split:
                raise ValueError(f"{path}:{line_number}: invalid B1 fields/split")
            if row["sample_id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate sample id")
            seen.add(row["sample_id"])
            turns = row["context_turns"]
            if not turns or turns[-1]["role"] != "user" or turns[-1]["turn_id"] != row["target_user_turn_id"]:
                raise ValueError(f"{path}:{line_number}: context does not end at target user turn")
            text = "\n".join(
                f"{SPEAKER_NAMES[turn['role']]}：{turn['content']}" for turn in turns
            )
            profile = row["user_profile"]
            if set(profile) != set(PROFILE_LABELS_BY_DIMENSION):
                raise ValueError(f"{path}:{line_number}: invalid profile dimensions")
            for dimension, labels in PROFILE_LABELS_BY_DIMENSION.items():
                if not set(profile[dimension]) <= set(labels):
                    raise ValueError(f"{path}:{line_number}: invalid {dimension} label")
            if row["emotion_label"] not in EMOTION_LABELS:
                raise ValueError(f"{path}:{line_number}: invalid emotion label")
            examples.append({
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
                "text": text,
                "emotion": row["emotion_label"],
                "profile": {key: list(value) for key, value in profile.items()},
            })
    if not examples:
        raise ValueError(f"empty dataset: {path}")
    return examples


def profile_targets(examples: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    return {
        dim: torch.tensor(
            [[int(label in row["profile"][dim]) for label in labels] for row in examples],
            dtype=torch.float32,
        )
        for dim, labels in PROFILE_LABELS_BY_DIMENSION.items()
    }


def profile_metrics(
    expected: dict[str, Any], predicted: dict[str, Any]
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    exacts: list[float] = []
    empty_exacts: list[float] = []
    for dim in PROFILE_LABELS_BY_DIMENSION:
        truth = expected[dim].bool()
        guess = predicted[dim].bool()
        tp = int((truth & guess).sum().item())
        fp = int((~truth & guess).sum().item())
        fn = int((truth & ~guess).sum().item())
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 0.0
        exact = float((truth == guess).all(dim=1).float().mean().item())
        empty = float((truth.sum(dim=1) == 0).float().mean().item())
        details[dim] = {
            "exact_match_accuracy": round(exact, 6),
            "micro_f1": round(f1, 6),
            "gold_positive_count": int(truth.sum().item()),
            "predicted_positive_count": int(guess.sum().item()),
        }
        exacts.append(exact)
        empty_exacts.append(empty)
    return {
        "per_dimension": details,
        "profile_macro_exact_match_accuracy": round(sum(exacts) / len(exacts), 6),
        "empty_set_baseline": {
            "per_dimension_exact_match_accuracy": {
                dim: round(value, 6)
                for dim, value in zip(PROFILE_LABELS_BY_DIMENSION, empty_exacts)
            },
            "profile_macro_exact_match_accuracy": round(sum(empty_exacts) / len(empty_exacts), 6),
        },
    }


def calibrate_thresholds(
    probabilities: dict[str, Any], targets: dict[str, Any], torch: Any
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for dim in PROFILE_LABELS_BY_DIMENSION:
        best_f1 = -1.0
        best_threshold = 0.5
        target = targets[dim].bool()
        probs = probabilities[dim]
        for index in range(5, 96):
            threshold = index / 100
            guess = probs >= threshold
            tp = int((target & guess).sum().item())
            fp = int((~target & guess).sum().item())
            fn = int((target & ~guess).sum().item())
            denom = 2 * tp + fp + fn
            f1 = 2 * tp / denom if denom else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds[dim] = best_threshold
    return thresholds


def apply_thresholds(
    probabilities: dict[str, Any], thresholds: dict[str, float]
) -> dict[str, Any]:
    return {
        dim: probabilities[dim] >= thresholds[dim]
        for dim in PROFILE_LABELS_BY_DIMENSION
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=PROJECT / "data/processed/b1_full_v1/classification_train.jsonl")
    parser.add_argument("--val", type=Path, default=PROJECT / "data/processed/b1_full_v1/classification_val.jsonl")
    parser.add_argument("--emotion-checkpoint", type=Path, default=PROJECT / "weights/emotion_baseline")
    parser.add_argument("--emotion-report", type=Path, default=PROJECT / "reports/emotion_encoder_baseline.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "weights/emotion_profile_balanced_20261015")
    parser.add_argument("--report", type=Path, default=PROJECT / "results/emotion_profile_balanced_20261015.json")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install requirements-emotion.txt before training") from error

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    train = read_examples(args.train, "train")
    val = read_examples(args.val, "val")
    train_conversations = {row["conversation_id"] for row in train}
    val_conversations = {row["conversation_id"] for row in val}
    if train_conversations & val_conversations:
        raise ValueError("B1 train/validation conversations overlap")

    # Keep complete conversations together for a leakage-safe, training-only
    # calibration split. The official validation set remains untouched.
    by_conversation: dict[str, list[int]] = {}
    for index, row in enumerate(train):
        by_conversation.setdefault(row["conversation_id"], []).append(index)
    conv_ids = sorted(by_conversation)
    rng = random.Random(args.seed)
    rng.shuffle(conv_ids)
    calib_count = max(1, round(len(conv_ids) * args.calibration_fraction))
    calibration_conversations = set(conv_ids[:calib_count])
    fit_indices = [i for i, row in enumerate(train) if row["conversation_id"] not in calibration_conversations]
    calib_indices = [i for i, row in enumerate(train) if row["conversation_id"] in calibration_conversations]
    fit_rows = [train[i] for i in fit_indices]
    calib_rows = [train[i] for i in calib_indices]

    torch.manual_seed(args.seed)
    if device == "mps":
        torch.mps.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.emotion_checkpoint, local_files_only=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.emotion_checkpoint, local_files_only=True
    ).to(device)
    model.eval()
    encoder = model.base_model

    def encode(examples: list[dict[str, Any]], name: str) -> Any:
        chunks: list[Any] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(examples), args.batch_size):
                batch = examples[start : start + args.batch_size]
                tokens = tokenizer(
                    [row["text"] for row in batch], truncation=True,
                    max_length=args.max_length, padding=True, return_tensors="pt",
                )
                tokens = {key: value.to(device) for key, value in tokens.items()}
                output = encoder(**tokens, return_dict=True)
                chunks.append(output.pooler_output.detach().cpu())
                done = min(start + len(batch), len(examples))
                if done % 1024 < len(batch) or done == len(examples):
                    print(f"encode {name}: {done}/{len(examples)}", flush=True)
        print(f"encoded {name} in {time.perf_counter() - started:.1f}s", flush=True)
        return torch.cat(chunks, dim=0)

    started = time.perf_counter()
    fit_features = encode(fit_rows, "train-fit")
    calibration_features = encode(calib_rows, "calibration")
    val_features = encode(val, "validation")
    targets_fit = profile_targets(fit_rows, torch)
    targets_calib = profile_targets(calib_rows, torch)
    targets_val = profile_targets(val, torch)

    class ProfileHeads(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = torch.nn.ModuleDict({
                dim: torch.nn.Linear(int(model.config.hidden_size), len(labels))
                for dim, labels in PROFILE_LABELS_BY_DIMENSION.items()
            })

        def forward(self, features: Any) -> dict[str, Any]:
            return {dim: head(features) for dim, head in self.heads.items()}

    heads = ProfileHeads().to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    fit_x = fit_features.to(device)
    calib_x = calibration_features.to(device)
    fit_y = {dim: value.to(device) for dim, value in targets_fit.items()}
    val_x = val_features.to(device)
    val_y = {dim: value.to(device) for dim, value in targets_val.items()}
    pos_weights: dict[str, Any] = {}
    for dim in PROFILE_LABELS_BY_DIMENSION:
        positives = fit_y[dim].sum(dim=0)
        negatives = len(fit_rows) - positives
        pos_weights[dim] = torch.sqrt(negatives / positives.clamp(min=1)).clamp(min=1.0, max=8.0)

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_state: dict[str, Any] | None = None
    best_thresholds: dict[str, float] = {}
    batch_size = 512
    for epoch in range(1, args.epochs + 1):
        heads.train()
        permutation = torch.randperm(len(fit_rows))
        for start in range(0, len(fit_rows), batch_size):
            indices = permutation[start : start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = heads(fit_x[indices])
            losses = [
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[dim], fit_y[dim][indices], pos_weight=pos_weights[dim]
                )
                for dim in PROFILE_LABELS_BY_DIMENSION
            ]
            loss = sum(losses) / len(losses)
            if not torch.isfinite(loss):
                raise RuntimeError("profile training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 5.0)
            optimizer.step()

        heads.eval()
        with torch.inference_mode():
            calibration_probs = {
                dim: torch.sigmoid(x).cpu() for dim, x in heads(calib_x).items()
            }
            calibration_targets_cpu = {
                dim: value.cpu() for dim, value in targets_calib.items()
            }
            thresholds = calibrate_thresholds(
                calibration_probs, calibration_targets_cpu, torch
            )
            calibration_metrics = profile_metrics(
                calibration_targets_cpu, apply_thresholds(calibration_probs, thresholds)
            )
            score = sum(
                calibration_metrics["per_dimension"][dim]["micro_f1"]
                for dim in PROFILE_LABELS_BY_DIMENSION
            ) / len(PROFILE_LABELS_BY_DIMENSION)
            val_probs = {dim: torch.sigmoid(x).cpu() for dim, x in heads(val_x).items()}
            val_targets_cpu = {dim: value.cpu() for dim, value in val_y.items()}
            val_metrics = profile_metrics(
                val_targets_cpu, apply_thresholds(val_probs, thresholds)
            )
        history.append({
            "epoch": epoch,
            "calibration_macro_micro_f1": round(score, 6),
            "calibration_macro_exact_match": calibration_metrics["profile_macro_exact_match_accuracy"],
            "thresholds": thresholds,
            "validation_macro_micro_f1": round(sum(
                val_metrics["per_dimension"][dim]["micro_f1"]
                for dim in PROFILE_LABELS_BY_DIMENSION
            ) / len(PROFILE_LABELS_BY_DIMENSION), 6),
            "validation_macro_exact_match": val_metrics["profile_macro_exact_match_accuracy"],
        })
        print(
            f"epoch={epoch}/{args.epochs} calib_macro_micro_f1={score:.4f} "
            f"calib_exact={calibration_metrics['profile_macro_exact_match_accuracy']:.4f} "
            f"val_macro_micro_f1={history[-1]['validation_macro_micro_f1']:.4f} "
            f"val_exact={val_metrics['profile_macro_exact_match_accuracy']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_thresholds = thresholds
            best_state = {
                key: value.detach().cpu().contiguous().clone()
                for key, value in heads.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no profile checkpoint was selected")
    heads.load_state_dict(best_state)
    heads.eval()
    with torch.inference_mode():
        final_val_probs = {dim: torch.sigmoid(x).cpu() for dim, x in heads(val_x).items()}
        val_targets_cpu = {dim: value.cpu() for dim, value in val_y.items()}
        final_val_metrics = profile_metrics(
            val_targets_cpu, apply_thresholds(final_val_probs, best_thresholds)
        )
    baseline_probs = {
        dim: torch.zeros_like(val_targets_cpu[dim]) for dim in PROFILE_LABELS_BY_DIMENSION
    }
    empty_baseline = profile_metrics(val_targets_cpu, baseline_probs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = args.output_dir / "encoder"
    encoder.save_pretrained(encoder_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    classifier_state = {
        f"emotion_head.{key}": value.detach().cpu().contiguous()
        for key, value in model.classifier.state_dict().items()
    }
    profile_state = {
        f"profile_heads.{key.removeprefix('heads.')}": value
        for key, value in best_state.items()
    }
    from safetensors.torch import save_file
    save_file({**classifier_state, **profile_state}, str(args.output_dir / "heads.safetensors"))
    metadata = {
        "format_version": 2,
        "base_emotion_checkpoint": str(args.emotion_checkpoint),
        "base_emotion_model_sha256": sha256(args.emotion_checkpoint / "model.safetensors"),
        "emotion_labels": list(EMOTION_LABELS),
        "profile_labels": {key: list(value) for key, value in PROFILE_LABELS_BY_DIMENSION.items()},
        "profile_thresholds": best_thresholds,
        "pooling": "BertModel.pooler_output",
        "max_length": args.max_length,
        "calibration_split": "15% of training conversations, held out from head fitting",
    }
    (args.output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    emotion_report = json.loads(args.emotion_report.read_text(encoding="utf-8"))
    emotion_metrics = emotion_report.get("metrics", {})
    if "after" in emotion_metrics:
        emotion_metrics = emotion_metrics["after"]
    report = {
        "status": "profile heads trained; emotion checkpoint preserved",
        "device": device,
        "runtime": {
            "python": __import__("platform").python_version(),
            "torch": str(torch.__version__),
            "transformers": __import__("transformers").__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "method": {
            "encoder": "frozen from standalone emotion checkpoint",
            "emotion_head": "preserved without modification",
            "profile_heads": "three linear heads trained with capped inverse-square-root positive weights",
            "calibration": "training-only 15% conversation-disjoint split; thresholds selected by per-dimension micro-F1",
            "epochs": args.epochs,
            "selected_epoch": max(history, key=lambda row: row["calibration_macro_micro_f1"])["epoch"],
            "learning_rate": args.learning_rate,
            "batch_size": batch_size,
            "seed": args.seed,
        },
        "data": {
            "train_path": str(args.train), "train_sha256": sha256(args.train),
            "train_samples": len(train), "fit_samples": len(fit_rows),
            "calibration_samples": len(calib_rows), "calibration_conversations": calib_count,
            "validation_path": str(args.val), "validation_sha256": sha256(args.val),
            "validation_samples": len(val), "conversation_overlap": 0,
        },
        "emotion_reference_metrics": emotion_metrics,
        "emotion_reference_report": str(args.emotion_report),
        "profile": {
            "thresholds": best_thresholds,
            "validation": final_val_metrics,
            "empty_set_baseline": empty_baseline,
        },
        "history": history,
        "experiment_elapsed_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(args.output_dir),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_epoch": report["method"]["selected_epoch"],
        "emotion_reference_accuracy": report["emotion_reference_metrics"]["accuracy"],
        "emotion_reference_macro_f1": report["emotion_reference_metrics"]["macro_f1"],
        "profile_thresholds": best_thresholds,
        "profile_validation": final_val_metrics,
        "empty_set_baseline": empty_baseline,
        "checkpoint": str(args.output_dir),
        "report": str(args.report),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
