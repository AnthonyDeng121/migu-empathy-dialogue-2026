#!/usr/bin/env python3
"""Calibrate profile thresholds for per-dimension exact match on train-only holdout."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.train_profile_heads import (  # noqa: E402
    PROFILE_LABELS_BY_DIMENSION,
    profile_metrics,
    profile_targets,
    read_examples,
)


def exact_thresholds(probabilities: dict[str, Any], targets: dict[str, Any]) -> dict[str, float]:
    """Choose one threshold per dimension using only the calibration split.

    Ties prefer the higher threshold, which avoids extra predictions when the
    calibration exact-match score is unchanged.
    """
    thresholds: dict[str, float] = {}
    for dim in PROFILE_LABELS_BY_DIMENSION:
        target = targets[dim].bool()
        best_score = -1.0
        best_threshold = 1.01  # explicit all-empty prediction candidate
        for index in range(0, 102):
            threshold = index / 100
            guess = probabilities[dim] >= threshold
            score = float((target == guess).all(dim=1).float().mean().item())
            if score > best_score or (score == best_score and threshold > best_threshold):
                best_score = score
                best_threshold = threshold
        thresholds[dim] = best_threshold
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=PROJECT / "data/processed/b1_full_v1/classification_train.jsonl")
    parser.add_argument("--val", type=Path, default=PROJECT / "data/processed/b1_full_v1/classification_val.jsonl")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    device = args.device
    metadata = json.loads((args.checkpoint / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    max_length = int(metadata["max_length"])
    train = read_examples(args.train, "train")
    val = read_examples(args.val, "val")
    train_conversations = {row["conversation_id"] for row in train}
    val_conversations = {row["conversation_id"] for row in val}
    if train_conversations & val_conversations:
        raise ValueError("B1 train/validation conversations overlap")

    by_conversation: dict[str, list[int]] = {}
    for index, row in enumerate(train):
        by_conversation.setdefault(row["conversation_id"], []).append(index)
    conv_ids = sorted(by_conversation)
    rng = random.Random(args.seed)
    rng.shuffle(conv_ids)
    calib_count = max(1, round(len(conv_ids) * 0.15))
    calibration_conversations = set(conv_ids[:calib_count])
    calibration = [row for row in train if row["conversation_id"] in calibration_conversations]

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    tokenizer.truncation_side = "left"
    encoder = AutoModel.from_pretrained(
        args.checkpoint / "encoder", local_files_only=True
    ).to(device)
    encoder.eval()
    state = load_file(str(args.checkpoint / "heads.safetensors"), device="cpu")

    def probabilities(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
        chunks: list[Any] = []
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                tokens = tokenizer(
                    [row["text"] for row in batch], truncation=True,
                    max_length=max_length, padding=True, return_tensors="pt",
                )
                tokens = {key: value.to(device) for key, value in tokens.items()}
                chunks.append(encoder(**tokens, return_dict=True).pooler_output.detach().cpu())
                done = min(start + len(batch), len(rows))
                if done % 1024 < len(batch) or done == len(rows):
                    print(f"encode {name}: {done}/{len(rows)}", flush=True)
        features = torch.cat(chunks, dim=0)
        return {
            dim: torch.sigmoid(
                features @ state[f"profile_heads.{dim}.weight"].T
                + state[f"profile_heads.{dim}.bias"]
            )
            for dim in PROFILE_LABELS_BY_DIMENSION
        }

    started = time.perf_counter()
    calib_probs = probabilities(calibration, "calibration")
    val_probs = probabilities(val, "validation")
    thresholds = exact_thresholds(calib_probs, profile_targets(calibration, torch))
    val_targets = profile_targets(val, torch)
    val_predicted = {dim: val_probs[dim] >= thresholds[dim] for dim in PROFILE_LABELS_BY_DIMENSION}
    val_metrics = profile_metrics(val_targets, val_predicted)
    baseline = profile_metrics(
        val_targets,
        {dim: torch.zeros_like(val_targets[dim]) for dim in PROFILE_LABELS_BY_DIMENSION},
    )
    calib_targets = profile_targets(calibration, torch)
    calib_metrics = profile_metrics(
        calib_targets,
        {dim: calib_probs[dim] >= thresholds[dim] for dim in PROFILE_LABELS_BY_DIMENSION},
    )
    calib_empty_baseline = profile_metrics(
        calib_targets,
        {dim: torch.zeros_like(calib_targets[dim]) for dim in PROFILE_LABELS_BY_DIMENSION},
    )
    report = {
        "status": "threshold calibration evaluation; checkpoint weights unchanged",
        "method": {
            "objective": "per-dimension exact-match accuracy",
            "threshold_source": "training-only 15% conversation-disjoint calibration split",
            "validation_used_for_selection": False,
            "max_length": max_length,
            "seed": args.seed,
            "thresholds": thresholds,
        },
        "data": {
            "train_sha256": __import__("hashlib").sha256(args.train.read_bytes()).hexdigest(),
            "validation_sha256": __import__("hashlib").sha256(args.val.read_bytes()).hexdigest(),
            "train_samples": len(train),
            "calibration_samples": len(calibration),
            "validation_samples": len(val),
            "conversation_overlap": 0,
        },
        "calibration": calib_metrics,
        "validation": val_metrics,
        "empty_set_baseline": {
            "calibration": calib_empty_baseline,
            "validation": baseline,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(args.checkpoint),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "thresholds": thresholds,
        "calibration_macro_exact": calib_metrics["profile_macro_exact_match_accuracy"],
        "validation_macro_exact": val_metrics["profile_macro_exact_match_accuracy"],
        "empty_baseline_macro_exact": baseline["profile_macro_exact_match_accuracy"],
        "validation_macro_micro_f1": round(sum(
            val_metrics["per_dimension"][dim]["micro_f1"]
            for dim in PROFILE_LABELS_BY_DIMENSION
        ) / len(PROFILE_LABELS_BY_DIMENSION), 6),
        "report": str(args.report),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
