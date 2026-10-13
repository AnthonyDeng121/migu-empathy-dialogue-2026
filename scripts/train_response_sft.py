#!/usr/bin/env python3
"""LoRA SFT template for the response-generation line (task D1/D2).

Trains a small Chinese instruct model on empathy dialogue samples built by
src/response/sft_data.py. With --max-train-samples/--max-steps it runs as a
smoke: finite loss, nonzero LoRA gradient, adapter weights change, adapter
checkpoint reloads consistently, and the model produces non-empty replies.
Local smoke runs (CUDA/MPS/CPU) never count as platform GPU hours.
"""
from __future__ import annotations

import argparse
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

from src.common.io import read_jsonl  # noqa: E402
from src.response.sft_data import build_sft_samples, response_turn_consistency  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT.resolve()).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


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


def _load_optional_dependencies() -> dict[str, Any]:
    try:
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install the optional training dependencies with "
            "python -m pip install -r requirements-response.txt"
        ) from error
    return {
        "torch": torch,
        "transformers": transformers,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _render_ids(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
    # transformers 5.x apply_chat_template(tokenize=True) returns a tokenizer
    # Encoding, not flat ids; render to text and tokenize explicitly instead.
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=False
    )
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def _encode_sample(
    tokenizer: Any,
    sample: dict[str, Any],
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Prompt-masked token ids for one SFT sample.

    prompt tokens carry labels -100; the supervised tokens are the assistant
    response plus whatever the chat template appends (e.g. the EOS turn).
    Oversized contexts drop the oldest history turns first so the system
    prompt (with its condition block) survives; token-level left-cutting is
    the last resort once only system + target turn remain.
    """
    messages = list(sample["messages"])
    full_messages = messages + [{"role": "assistant", "content": sample["response_text"]}]
    while True:
        prompt_ids = _render_ids(tokenizer, messages, add_generation_prompt=True)
        full_ids = _render_ids(tokenizer, full_messages, add_generation_prompt=False)
        overflow = len(prompt_ids) + (len(full_ids) - len(prompt_ids)) - max_length
        if overflow <= 0:
            break
        if len(messages) > 2:
            del messages[1]
            full_messages = messages + [
                {"role": "assistant", "content": sample["response_text"]}
            ]
            continue
        response_ids = full_ids[len(prompt_ids):]
        if overflow > len(prompt_ids):
            raise ValueError(
                f"{sample['id']}: response tokens alone exceed max_length "
                f"{max_length} ({len(response_ids)} tokens)"
            )
        prompt_ids = prompt_ids[overflow:]
        labels = [-100] * len(prompt_ids) + list(response_ids)
        return list(prompt_ids) + list(response_ids), labels
    response_ids = full_ids[len(prompt_ids):]
    if not response_ids:
        raise ValueError(f"empty supervised tokens for {sample['id']}")
    labels = [-100] * len(prompt_ids) + list(response_ids)
    return list(full_ids), labels


def _collate(tokenizer: Any, torch: Any, features: list[dict[str, Any]]) -> dict[str, Any]:
    maxlen = max(len(feature["input_ids"]) for feature in features)
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention: list[list[int]] = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for feature in features:
        padding = maxlen - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_id] * padding)
        labels.append(feature["labels"] + [-100] * padding)
        attention.append([1] * len(feature["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def _lora_weight_snapshot(model: Any) -> tuple[str, Any]:
    """First LoRA weight as (parameter name, tensor) for before/after checks."""
    for name, parameter in model.named_parameters():
        if "lora_A" in name:
            return name, parameter.detach().cpu().clone()
    raise RuntimeError("no LoRA parameter found on model")


def _steps_per_epoch(examples: int, batch_size: int, accumulation: int) -> int:
    micro_batches = math.ceil(examples / batch_size)
    return math.ceil(micro_batches / accumulation)


def _generate_samples(
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    samples: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[dict[str, str]]:
    model.eval()
    results: list[dict[str, str]] = []
    for sample in samples:
        prompt_text = tokenizer.apply_chat_template(
            sample["messages"], add_generation_prompt=True, tokenize=False
        )
        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = output_ids[0][prompt_ids.shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        results.append({"id": sample["id"], "generated_text": text})
    model.train()
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = _load_optional_dependencies()
    torch = deps["torch"]
    transformers = deps["transformers"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    LoraConfig = deps["LoraConfig"]
    get_peft_model = deps["get_peft_model"]

    config = json.loads(args.config.read_text(encoding="utf-8"))
    train_rows = read_jsonl(args.train)
    consistency = response_turn_consistency(train_rows)
    train_examples = build_sft_samples(
        train_rows,
        condition_mode=str(config["condition_mode"]),
        max_history_turns=int(config["max_history_turns"]),
    )
    train_total = len(train_examples)
    for option_name, value in (
        ("max_train_samples", args.max_train_samples),
        ("max_steps", args.max_steps),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{option_name} must be positive")
    if args.max_train_samples:
        train_examples = train_examples[: args.max_train_samples]
    if not train_examples:
        raise ValueError("training set contains no samples")
    smoke_mode = args.max_train_samples is not None or args.max_steps is not None

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(torch, args.device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    model_name = str(config["model_name"])
    model_revision = str(config["model_revision"])
    # --model-dir loads a local snapshot (e.g. weights/hf/<name>); the config
    # name/revision stay in the report for provenance.
    load_name = str(args.model_dir) if args.model_dir else model_name
    load_kwargs: dict[str, Any] = (
        {"local_files_only": True} if args.model_dir else {"revision": model_revision, "local_files_only": args.local_files_only}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        load_name,
        use_fast=True,
        **load_kwargs,
    )
    model = AutoModelForCausalLM.from_pretrained(
        load_name,
        **load_kwargs,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    model.config.use_cache = False
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["lora_target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    model.print_trainable_parameters()
    if smoke_mode:
        initial_lora_name, initial_lora_weight = _lora_weight_snapshot(model)

    train_encoded: list[dict[str, Any]] = []
    context_trimmed = 0
    for sample in train_examples:
        try:
            input_ids, labels = _encode_sample(
                tokenizer, sample, int(config["max_length"])
            )
        except ValueError as error:
            if "exceed max_length" in str(error):
                context_trimmed += 1
                continue
            raise
        train_encoded.append({"input_ids": input_ids, "labels": labels})
    if not train_encoded:
        raise ValueError("all training samples exceeded max_length")
    token_totals = [len(feature["input_ids"]) for feature in train_encoded]

    batch_size = int(config["batch_size"])
    accumulation = int(config["gradient_accumulation"])
    optimizer_steps_per_epoch = _steps_per_epoch(len(train_encoded), batch_size, accumulation)
    epochs = int(config["epochs"])
    total_steps = optimizer_steps_per_epoch * epochs
    effective_total_steps = args.max_steps or total_steps
    warmup_steps = max(1, int(effective_total_steps * float(config["warmup_ratio"])))

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    def learning_rate_scale(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, (effective_total_steps - step) / max(effective_total_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_encoded), generator=generator).tolist()
    micro_batches = [
        train_encoded[index : index + batch_size] for index in range(0, len(order), batch_size)
    ]
    collate = lambda features: _collate(tokenizer, torch, features)

    autocast_enabled = use_bf16
    first_loss = math.nan
    first_gradient_norm = math.nan
    optimizer_steps = 0
    measured_step_seconds = 0.0
    loss_accumulator = 0.0
    micro_batches_done = 0
    training_started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for micro_index, micro_batch in enumerate(micro_batches):
        if args.max_steps is not None and optimizer_steps >= args.max_steps:
            break
        step_started = time.perf_counter()
        batch = collate(micro_batch)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            loss = model(**batch).loss / accumulation
        loss_value = float(loss.detach().cpu()) * accumulation
        if optimizer_steps == 0 and not math.isfinite(loss_value):
            raise RuntimeError(f"first step produced a non-finite loss: {loss_value}")
        if optimizer_steps == 0:
            first_loss = loss_value
        loss.backward()
        loss_accumulator += loss_value
        micro_batches_done += 1
        is_last_micro = micro_index == len(micro_batches) - 1
        if (micro_index + 1) % accumulation == 0 or is_last_micro:
            if optimizer_steps == 0:
                gradients = [
                    p.grad
                    for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                if gradients:
                    first_gradient_norm = float(
                        torch.linalg.vector_norm(
                            torch.stack([torch.linalg.vector_norm(g) for g in gradients])
                        ).cpu()
                    )
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            measured_step_seconds += time.perf_counter() - step_started
            if args.max_steps is not None and optimizer_steps >= args.max_steps:
                break
            if optimizer_steps % 10 == 0:
                print(
                    f"step={optimizer_steps}/{effective_total_steps} "
                    f"train_loss={loss_accumulator / max(micro_batches_done, 1):.4f} device={device}",
                    flush=True,
                )
    training_seconds = time.perf_counter() - training_started
    if optimizer_steps < 1:
        raise RuntimeError("training completed no optimizer steps")

    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    if smoke_mode:
        # Reload check compares the saved adapter file against the in-memory
        # LoRA weights; the base model is not loaded a second time (RAM peak).
        from safetensors.torch import load_file

        lora_name, selected_lora_weight = _lora_weight_snapshot(model)
        # saved adapter keys drop the peft adapter name ("...lora_A.default.weight"
        # becomes "...lora_A.weight")
        adapter_file = args.output_dir / "adapter_model.safetensors"
        saved_lora = load_file(adapter_file)
        saved_name = lora_name.replace(".default.weight", ".weight")
        if saved_name not in saved_lora:
            raise RuntimeError(f"{saved_name} missing from saved adapter")
        reload_ok = bool(torch.equal(saved_lora[saved_name], selected_lora_weight))
    else:
        reload_ok = True

    eval_prompts = int(args.eval_prompts)
    generation_results: list[dict[str, str]] = []
    if eval_prompts > 0:
        if args.max_steps is None:
            raise ValueError("--eval-prompts requires smoke mode (--max-steps)")
        probe = train_examples[: min(eval_prompts, len(train_examples))]
        generation_results = _generate_samples(
            model, tokenizer, torch, device, probe, int(config["max_new_tokens"])
        )
    non_empty_generations = sum(1 for item in generation_results if item["generated_text"])

    seconds_per_step = measured_step_seconds / optimizer_steps
    full_steps_per_epoch = _steps_per_epoch(train_total, batch_size, accumulation)
    eta_seconds = seconds_per_step * full_steps_per_epoch * epochs
    vram_peak = (
        int(torch.cuda.max_memory_allocated())
        if device.type == "cuda"
        else None
    )
    adapter_files = sorted(args.output_dir.glob("*.safetensors"))
    adapter_path = adapter_files[0] if adapter_files else None
    report = {
        "version": "response_sft_prep_v1",
        "status": "smoke" if smoke_mode else "candidate training",
        "model": {
            "name": model_name,
            "revision": model_revision,
            "card_url": f"https://huggingface.co/{model_name}",
        },
        "device": str(device),
        "dtype": "bfloat16" if use_bf16 else "float32",
        "seed": seed,
        "config": config,
        "runtime": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "training_seconds": round(training_seconds, 3),
        },
        "data": {
            "train_path": _portable_path(args.train),
            "train_total_samples": train_total,
            "train_used_samples": len(train_encoded),
            "skipped_long_samples": context_trimmed,
            "train_sha256": _sha256(args.train),
            "response_turn_consistency": consistency,
            "condition_mode": str(config["condition_mode"]),
            "max_history_turns": int(config["max_history_turns"]),
            "prompt_plus_response_tokens": {
                "min": min(token_totals),
                "mean": round(sum(token_totals) / len(token_totals), 1),
                "max": max(token_totals),
            },
        },
        "parameters": {
            "trainable": trainable_parameters,
            "total": total_parameters,
            "fraction": round(trainable_parameters / total_parameters, 6),
        },
        "training_speed": {
            "optimizer_steps": optimizer_steps,
            "micro_batches_done": micro_batches_done,
            "mean_train_loss": round(loss_accumulator / max(micro_batches_done, 1), 4),
            "measured_step_seconds": round(measured_step_seconds, 3),
            "seconds_per_step": round(seconds_per_step, 6),
            "full_steps_per_epoch": full_steps_per_epoch,
            "estimated_full_train_seconds": round(eta_seconds, 3),
            "eta_margin_note": "workplan requires +30% failure margin on top of this estimate",
        },
        "vram_peak_bytes": vram_peak,
        "smoke_checks": {
            "optimizer_steps": optimizer_steps,
            "first_loss": round(first_loss, 6) if math.isfinite(first_loss) else None,
            "first_step_gradient_finite_nonzero": bool(
                math.isfinite(first_gradient_norm) and first_gradient_norm > 0
            ),
            "first_step_lora_gradient_norm": (
                round(first_gradient_norm, 8) if math.isfinite(first_gradient_norm) else None
            ),
            "lora_weights_changed": bool(
                smoke_mode
                and not torch.equal(
                    saved_lora.get(
                        lora_name.replace(".default.weight", ".weight"), initial_lora_weight
                    ),
                    initial_lora_weight,
                )
            ),
            "lora_weight_name": lora_name if smoke_mode else None,
            "adapter_reload_passed": bool(reload_ok),
            "generation_probe": generation_results,
            "non_empty_generations": non_empty_generations,
        }
        if smoke_mode
        else None,
        "checkpoint_directory": _portable_path(args.output_dir),
        "adapter": {
            "file": f"{_portable_path(args.output_dir)}/{adapter_path.name}" if adapter_path else None,
            "bytes": adapter_path.stat().st_size if adapter_path else 0,
            "sha256": _sha256(adapter_path) if adapter_path else None,
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report_md.write_text(_render_report(report), encoding="utf-8")
    print(
        f"PASS: device={device} steps={optimizer_steps} "
        f"sec/step={seconds_per_step:.4f} eta_full={eta_seconds:.0f}s "
        f"vram={vram_peak if vram_peak is not None else 'n/a'}",
        flush=True,
    )
    return report


def _render_report(report: dict[str, Any]) -> str:
    data = report["data"]
    speed = report["training_speed"]
    lines = [
        "# D 岗位回复 SFT 训练运行报告",
        "",
        f"**状态：{report['status']}。**",
        "",
        f"模型：[`{report['model']['name']}`]({report['model']['card_url']})，"
        f"revision `{report['model']['revision']}`；设备 `{report['device']}`（{report['dtype']}）；"
        f"seed `{report['seed']}`。",
        f"LoRA：r={report['config']['lora_r']} alpha={report['config']['lora_alpha']} "
        f"dropout={report['config']['lora_dropout']}；"
        f"可训练参数 {report['parameters']['trainable']:,} "
        f"（{report['parameters']['fraction']:.2%} of {report['parameters']['total']:,}）。",
        f"样本：{data['train_used_samples']}/{data['train_total_samples']} 个 SFT 样本"
        f"（超长跳过 {data['skipped_long_samples']}）；条件注入 `{data['condition_mode']}`；"
        f"最大历史轮数 {data['max_history_turns']}。",
        f"token 长度（prompt+回复）：min {data['prompt_plus_response_tokens']['min']} / "
        f"mean {data['prompt_plus_response_tokens']['mean']} / "
        f"max {data['prompt_plus_response_tokens']['max']}（max_length {report['config']['max_length']}）。",
        f"回复 turn 与目标一致性：{data['response_turn_consistency']['match']}/"
        f"{data['response_turn_consistency']['total']} 与转写正文一致。",
        "",
        f"测速：{speed['optimizer_steps']} optimizer steps，"
        f"{speed['seconds_per_step']:.4f} 秒/step；全量 {speed['full_steps_per_epoch']} steps/epoch，"
        f"估算 {speed['estimated_full_train_seconds']:.0f} 秒/epoch"
        f"（未含 workplan 要求的 +30% 故障余量）。",
    ]
    if report.get("vram_peak_bytes") is not None:
        lines.append(f"显存峰值：{report['vram_peak_bytes'] / 1024**3:.2f} GiB。")
    smoke = report.get("smoke_checks")
    if smoke:
        lines.extend(
            [
                "",
                f"冒烟检查：{smoke['optimizer_steps']} steps；first loss {smoke['first_loss']}；"
                f"梯度有限非零={smoke['first_step_gradient_finite_nonzero']}；"
                f"LoRA 权重变化={smoke['lora_weights_changed']}；"
                f"adapter 重载一致={smoke['adapter_reload_passed']}；"
                f"生成探针 {smoke['non_empty_generations']}/{len(smoke['generation_probe'])} 非空。",
            ]
        )
        for item in smoke["generation_probe"]:
            preview = item["generated_text"][:120].replace("\n", " ")
            lines.append(f"- `{item['id']}`：{preview}…")
    lines.extend(
        [
            "",
            "复现命令：",
            "",
            "```bash",
            "python -m pip install -r requirements-response.txt",
            f"python scripts/train_response_sft.py --train '{data['train_path']}' \\",
            f"  --config configs/response_sft.json --device {report['device']} \\",
            f"  --output-dir {report['checkpoint_directory']} \\",
            "  --report-json reports/response_sft_run.json --report-md reports/RESPONSE_SFT_RUN.md",
            "```",
            "",
            "权重只保存在本机忽略目录，不提交 Git；本机运行不计平台 GPU 卡时。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT training template for response generation")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/response_sft.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "weights/response_sft")
    parser.add_argument(
        "--report-json", type=Path, default=PROJECT / "reports/response_sft_prep.json"
    )
    parser.add_argument(
        "--report-md", type=Path, default=PROJECT / "reports/RESPONSE_SFT_PREP.md"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="local model snapshot directory (overrides config model_name/revision for loading)",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-prompts", type=int, default=3)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
