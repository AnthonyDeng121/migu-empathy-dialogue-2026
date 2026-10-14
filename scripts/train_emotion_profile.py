#!/usr/bin/env python3
"""Train a shared MacBERT emotion and three-dimension profile classifier."""
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

from src.classification.emotion_encoder import classification_metrics  # noqa: E402
from src.classification.emotion_profile import (  # noqa: E402
    EMOTION_LABELS,
    PROFILE_LABELS_BY_DIMENSION,
    build_b1_classification_examples,
    decode_emotion_profile_scores,
    profile_metrics,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT).as_posix()
    except ValueError:
        return str(resolved)


def _load_b1_jsonl(
    path: Path, *, split: str, max_samples: int | None, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_samples is not None and max_samples < 1:
        raise ValueError("sample limits must be positive")
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    conversation_ids: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                example = build_b1_classification_examples([row], expected_split=split)[0]
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if example["sample_id"] in sample_ids:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {example['sample_id']}")
            sample_ids.add(example["sample_id"])
            conversation_ids.add(example["conversation_id"])
            count += 1
            if max_samples is None or len(examples) < max_samples:
                examples.append(example)
            else:
                replacement = rng.randrange(count)
                if replacement < max_samples:
                    examples[replacement] = example
    if not examples:
        raise ValueError(f"{path} contains no classification rows")
    return examples, {
        "path": str(path),
        "sha256": _sha256(path),
        "total_samples": count,
        "used_samples": len(examples),
        "conversation_count": len(conversation_ids),
        "sample_ids": sample_ids,
        "conversation_ids": conversation_ids,
    }


def _assert_disjoint(train: dict[str, Any], val: dict[str, Any]) -> None:
    shared_samples = train["sample_ids"] & val["sample_ids"]
    if shared_samples:
        raise ValueError(f"B1 train/val sample_id overlap: {sorted(shared_samples)[:5]}")
    shared_conversations = train["conversation_ids"] & val["conversation_ids"]
    if shared_conversations:
        raise ValueError(f"B1 train/val conversation overlap: {sorted(shared_conversations)[:5]}")


def _make_model(torch: Any, encoder: Any) -> Any:
    nn = torch.nn
    hidden_size = int(encoder.config.hidden_size)

    class EmotionProfileModel(nn.Module):
        def __init__(self, base_encoder: Any) -> None:
            super().__init__()
            self.encoder = base_encoder
            self.dropout = nn.Dropout(0.1)
            self.emotion_head = nn.Linear(hidden_size, len(EMOTION_LABELS))
            self.profile_heads = nn.ModuleDict(
                {
                    dimension: nn.Linear(hidden_size, len(labels))
                    for dimension, labels in PROFILE_LABELS_BY_DIMENSION.items()
                }
            )

        def forward(self, **inputs: Any) -> dict[str, Any]:
            encoded = self.encoder(**inputs, return_dict=True)
            pooled = encoded.last_hidden_state[:, 0, :]
            pooled = self.dropout(pooled)
            return {
                "emotion_logits": self.emotion_head(pooled),
                "profile_logits": {
                    dimension: head(pooled)
                    for dimension, head in self.profile_heads.items()
                },
            }

    return EmotionProfileModel(encoder)


def _load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install the isolated training stack with "
            "python -m pip install -r requirements-emotion-train.txt"
        ) from error
    return torch, (AutoModel, AutoTokenizer), (load_file, save_file)


def _select_device(torch: Any, requested: str) -> Any:
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


def _batch(
    tokenizer: Any, torch: Any, examples: Sequence[dict[str, Any]], max_length: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    encoded = tokenizer(
        [example["text"] for example in examples],
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value for key, value in encoded.items() if key in {"input_ids", "attention_mask", "token_type_ids"}}
    emotion_labels = torch.tensor(
        [EMOTION_LABELS.index(example["emotion_label"]) for example in examples],
        dtype=torch.long,
    )
    profile_labels = {
        dimension: torch.tensor(
            [
                [int(label in example["user_profile"][dimension]) for label in labels]
                for example in examples
            ],
            dtype=torch.float32,
        )
        for dimension, labels in PROFILE_LABELS_BY_DIMENSION.items()
    }
    return inputs, {"emotion": emotion_labels, "profile": profile_labels}


def _batches(
    examples: Sequence[dict[str, Any]], batch_size: int
) -> Sequence[Sequence[dict[str, Any]]]:
    return [examples[start : start + batch_size] for start in range(0, len(examples), batch_size)]


def _evaluate(
    model: Any,
    tokenizer: Any,
    torch: Any,
    examples: Sequence[dict[str, Any]],
    *,
    device: Any,
    batch_size: int,
    max_length: int,
    profile_threshold: float,
) -> dict[str, Any]:
    expected_emotion: list[str] = []
    predicted_emotion: list[str] = []
    expected_profile: list[dict[str, Any]] = []
    predicted_profile: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for group in _batches(examples, batch_size):
            inputs, _ = _batch(tokenizer, torch, group, max_length)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model(**inputs)
            emotion_scores = output["emotion_logits"].detach().cpu().tolist()
            profile_scores = {
                dimension: torch.sigmoid(logits).detach().cpu().tolist()
                for dimension, logits in output["profile_logits"].items()
            }
            for index, example in enumerate(group):
                decoded = decode_emotion_profile_scores(
                    emotion_scores[index],
                    {dimension: scores[index] for dimension, scores in profile_scores.items()},
                    threshold=profile_threshold,
                )
                expected_emotion.append(example["emotion_label"])
                predicted_emotion.append(decoded["emotion_label"])
                expected_profile.append(example["user_profile"])
                predicted_profile.append(decoded["user_profile"])
    return {
        "emotion": classification_metrics(expected_emotion, predicted_emotion, EMOTION_LABELS),
        "profile": profile_metrics(expected_profile, predicted_profile),
    }


def _head_state(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in model.state_dict().items()
        if name.startswith("emotion_head.") or name.startswith("profile_heads.")
    }


def _checkpoint_reload_check(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    AutoModel: Any,
    output_dir: Path,
    examples: Sequence[dict[str, Any]],
    device: Any,
    max_length: int,
    load_file: Any,
) -> bool:
    reload_encoder = AutoModel.from_pretrained(
        output_dir / "encoder", local_files_only=True
    )
    reloaded = _make_model(torch, reload_encoder)
    incompatibility = reloaded.load_state_dict(
        load_file(str(output_dir / "heads.safetensors")), strict=False
    )
    if incompatibility.unexpected_keys or any(
        not key.startswith("encoder.") for key in incompatibility.missing_keys
    ):
        raise RuntimeError("checkpoint head state is incompatible with the model")
    reloaded.to(device)
    reloaded.eval()
    original_device = model.training
    model.eval()
    inputs, _ = _batch(tokenizer, torch, examples[: min(2, len(examples))], max_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        expected = model(**inputs)
        actual = reloaded(**inputs)
    torch.testing.assert_close(
        actual["emotion_logits"], expected["emotion_logits"], rtol=1e-4, atol=1e-4
    )
    for dimension in PROFILE_LABELS_BY_DIMENSION:
        torch.testing.assert_close(
            actual["profile_logits"][dimension],
            expected["profile_logits"][dimension],
            rtol=1e-4,
            atol=1e-4,
        )
    model.train(original_device)
    return True


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _markdown(report: dict[str, Any]) -> str:
    emotion = report["metrics"]["emotion"]
    profiles = report["metrics"]["profile"]
    lines = [
        "# Day 3 C：情感与用户画像分类 dry-run",
        "",
        f"**状态：{report['status']}。** 本次是本机准备/冒烟证据，不代表平台 GPU 全量训练。",
        "",
        f"- 设备：`{report['device']}`；模型：`{report['model']['name']}`@`{report['model']['revision']}`",
        f"- 训练样本：{report['data']['train']['used_samples']}/{report['data']['train']['total_samples']}；",
        f"验证样本：{report['data']['val']['used_samples']}/{report['data']['val']['total_samples']}",
        f"- B1 全行校验：{report['integrity']['train_rows_validated']} train + {report['integrity']['val_rows_validated']} val；",
        f"conversation overlap：{report['integrity']['train_val_conversation_overlap']}",
        f"- 训练步数：{report['training']['optimizer_steps']}；batch size：{report['config']['batch_size']}；",
        f"最大长度：{report['config']['max_length']}；实测耗时：{report['training']['elapsed_seconds']:.2f}s",
        f"- 环境：Python {report['runtime']['python']} / PyTorch {report['runtime']['torch']} / Transformers {report['runtime']['transformers']}",
        f"- 情感验证 Accuracy：{emotion['accuracy']:.4f}；Macro-F1：{emotion['macro_f1']:.4f}",
        f"- 画像三维平均 exact match：{profiles['profile_macro_exact_match_accuracy']:.4f}；",
        f"空集合 baseline：{profiles['empty_set_baseline']['profile_macro_exact_match_accuracy']:.4f}",
        f"- 训练数据 SHA-256：`{report['data']['train']['sha256']}`",
        f"- 验证数据 SHA-256：`{report['data']['val']['sha256']}`",
        f"- checkpoint 目录哈希：`{report['checkpoint']['tree_sha256']}`；",
        f"head 哈希：`{report['checkpoint']['heads_sha256']}`",
        "",
        "| 画像维度 | Exact match | Micro-F1 | 空集合 baseline |",
        "|---|---:|---:|---:|",
    ]
    for dimension, values in profiles["per_dimension"].items():
        empty = profiles["empty_set_baseline"]["per_dimension_exact_match_accuracy"][dimension]
        lines.append(
            f"| {dimension} | {values['exact_match_accuracy']:.4f} | "
            f"{values['micro_f1']:.4f} | {empty:.4f} |"
        )
    if report.get("inference_smoke"):
        inference = report["inference_smoke"]
        lines.extend(
            [
                "",
                f"正式输入接口冒烟：官方 test 输入 {inference['input_rows']} 行；输出 {inference['output_rows']} 行，",
                f"ID 顺序和 `{', '.join(inference['output_fields'])}` 字段检查通过（{inference['device']}）。",
            ]
        )
    lines.extend(
        [
            "",
            "本机 checkpoint 保存并重载后 logits 一致性检查通过。正式训练仍需在大赛 GPU 环境复跑；",
            "官方 vLLM 镜像的依赖兼容性需在该镜像内验证后才能确认。",
            "",
            "## 复现命令",
            "",
            "```bash",
            report["reproduction_command"],
            "```",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/emotion_profile.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "weights/emotion_profile_run")
    parser.add_argument("--report-json", type=Path, default=PROJECT / "reports/c_day3_emotion_profile.json")
    parser.add_argument("--report-md", type=Path, default=PROJECT / "reports/C_DAY3_EMOTION_PROFILE.md")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    train, train_meta = _load_b1_jsonl(
        args.train, split="train", max_samples=args.max_train_samples, seed=seed
    )
    val, val_meta = _load_b1_jsonl(
        args.val, split="val", max_samples=args.max_validation_samples, seed=seed + 1
    )
    _assert_disjoint(train_meta, val_meta)
    train_meta.pop("sample_ids")
    train_meta.pop("conversation_ids")
    val_meta.pop("sample_ids")
    val_meta.pop("conversation_ids")

    torch, (AutoModel, AutoTokenizer), (load_file, save_file) = _load_dependencies()
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    device = _select_device(torch, args.device)
    model_name = str(config["model_name"])
    revision = str(config["model_revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, local_files_only=args.local_files_only
    )
    encoder = AutoModel.from_pretrained(
        model_name, revision=revision, local_files_only=args.local_files_only
    )
    model = _make_model(torch, encoder).to(device)

    batch_size = int(config["batch_size"])
    max_length = int(config["max_length"])
    epochs = int(config["epochs"])
    threshold = float(config["profile_threshold"])
    learning_rate = float(config["learning_rate"])
    if batch_size < 1 or max_length < 2 or epochs < 1 or not 0 <= threshold <= 1:
        raise ValueError("invalid training configuration")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"checkpoint output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    changed_before = {
        name: parameter.detach().to("cpu").clone()
        for name, parameter in model.named_parameters()
        if name.startswith("emotion_head.") or name.startswith("profile_heads.")
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    train_indices = list(range(len(train)))
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    gradient_norms: list[float] = []
    step_durations: list[float] = []
    start_time = time.perf_counter()
    for epoch in range(epochs):
        permutation = torch.randperm(len(train_indices), generator=generator).tolist()
        ordered = [train[index] for index in permutation]
        model.train()
        for examples in _batches(ordered, batch_size):
            step_started = time.perf_counter()
            inputs, labels = _batch(tokenizer, torch, examples, max_length)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            emotion_labels = labels["emotion"].to(device)
            profile_labels = {
                dimension: values.to(device)
                for dimension, values in labels["profile"].items()
            }
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs)
            emotion_loss = torch.nn.functional.cross_entropy(
                output["emotion_logits"], emotion_labels
            )
            profile_losses = [
                torch.nn.functional.binary_cross_entropy_with_logits(
                    output["profile_logits"][dimension], profile_labels[dimension]
                )
                for dimension in PROFILE_LABELS_BY_DIMENSION
            ]
            loss = emotion_loss + sum(profile_losses) / len(profile_losses)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("training produced a non-finite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gradient_value = float(gradient_norm.detach().cpu().item())
            if not math.isfinite(gradient_value) or gradient_value <= 0:
                raise RuntimeError("training produced an invalid gradient norm")
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            gradient_norms.append(gradient_value)
            step_durations.append(time.perf_counter() - step_started)
            if args.max_steps is not None and len(losses) >= args.max_steps:
                break
        if args.max_steps is not None and len(losses) >= args.max_steps:
            break
    elapsed = time.perf_counter() - start_time
    if not losses:
        raise RuntimeError("training completed zero optimizer steps")
    changed_names = [
        name for name, before in changed_before.items()
        if not torch.equal(before, dict(model.named_parameters())[name].detach().to("cpu"))
    ]
    changed_heads = sorted({name.split(".")[0] if name.startswith("emotion_head.") else "profile_heads." + name.split(".")[1] for name in changed_names})
    required_heads = {"emotion_head"} | {
        f"profile_heads.{dimension}" for dimension in PROFILE_LABELS_BY_DIMENSION
    }
    if changed_heads != sorted(required_heads):
        raise RuntimeError(f"not all four classifier heads changed: {changed_heads}")

    metrics = _evaluate(
        model, tokenizer, torch, val, device=device, batch_size=batch_size,
        max_length=max_length, profile_threshold=threshold,
    )
    encoder_dir = output_dir / "encoder"
    encoder.save_pretrained(encoder_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    save_file(_head_state(model), str(output_dir / "heads.safetensors"))
    _write_json(
        output_dir / "checkpoint_metadata.json",
        {
            "format_version": 1,
            "model_name": model_name,
            "model_revision": revision,
            "emotion_labels": list(EMOTION_LABELS),
            "profile_labels": {
                dimension: list(labels)
                for dimension, labels in PROFILE_LABELS_BY_DIMENSION.items()
            },
            "profile_threshold": threshold,
            "max_length": max_length,
        },
    )
    checkpoint_reload_passed = _checkpoint_reload_check(
        model=model, tokenizer=tokenizer, torch=torch, AutoModel=AutoModel,
        output_dir=output_dir, examples=val, device=device, max_length=max_length,
        load_file=load_file,
    )
    expected_steps = math.ceil(train_meta["total_samples"] / batch_size) * epochs
    mean_step = sum(step_durations) / len(step_durations)
    exploratory = any(
        [args.max_train_samples is not None, args.max_validation_samples is not None, args.max_steps is not None]
    )
    command = (
        "python scripts/train_emotion_profile.py "
        f"--train '{_portable_path(args.train)}' --val '{_portable_path(args.val)}' "
        "--config 'configs/emotion_profile.json' --output-dir 'weights/emotion_profile_repro' "
        f"--device {device.type}"
    )
    if args.local_files_only:
        command += " --local-files-only"
    if args.max_train_samples is not None:
        command += f" --max-train-samples {args.max_train_samples}"
    if args.max_validation_samples is not None:
        command += f" --max-validation-samples {args.max_validation_samples}"
    if args.max_steps is not None:
        command += f" --max-steps {args.max_steps}"
    report = {
        "status": "Exploratory sample only" if exploratory else "Full B1 training run",
        "scope_note": "Local MPS/CPU dry-run is not platform RTX 4090 training or official-model-image validation.",
        "device": str(device),
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0] or None,
            "torch": str(torch.__version__),
            "transformers": __import__("transformers").__version__,
            "mps_available": bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ),
        },
        "model": {"name": model_name, "revision": revision},
        "config": {
            "seed": seed, "epochs": epochs, "batch_size": batch_size,
            "max_length": max_length, "learning_rate": learning_rate,
            "profile_threshold": threshold,
            "loss": "emotion cross entropy + mean of three profile BCE losses",
        },
        "data": {"train": train_meta, "val": val_meta},
        "integrity": {
            "train_rows_validated": train_meta["total_samples"],
            "val_rows_validated": val_meta["total_samples"],
            "train_val_sample_id_overlap": 0,
            "train_val_conversation_overlap": 0,
            "all_b1_rows_validated_even_when_sampling": True,
        },
        "training": {
            "optimizer_steps": len(losses), "elapsed_seconds": round(elapsed, 3),
            "mean_step_seconds": round(mean_step, 6),
            "estimated_full_epoch_seconds_from_observed_step": round(mean_step * expected_steps, 3),
            "estimated_steps_per_epoch": expected_steps,
            "first_loss": round(losses[0], 8), "last_loss": round(losses[-1], 8),
            "first_gradient_norm": round(gradient_norms[0], 8),
            "all_losses_finite": all(math.isfinite(value) for value in losses),
            "all_gradient_norms_finite_nonzero": all(math.isfinite(value) and value > 0 for value in gradient_norms),
            "changed_heads": changed_heads,
        },
        "metrics": metrics,
        "checkpoint": {
            "path": _portable_path(output_dir), "heads_sha256": _sha256(output_dir / "heads.safetensors"),
            "tree_sha256": _tree_sha256(output_dir), "bytes": sum(
                child.stat().st_size for child in output_dir.rglob("*") if child.is_file()
            ),
            "reload_logits_match": checkpoint_reload_passed,
            "safetensors_only_for_custom_heads": True,
        },
        "runtime_compatibility": {
            "official_vllm_image_verified": False,
            "torch_owned_by_official_base_image": True,
            "training_dependencies_separated_from_inference": True,
        },
        "reproduction_command": command,
    }
    _write_json(args.report_json, report)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
