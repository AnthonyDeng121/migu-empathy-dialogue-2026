#!/usr/bin/env python3
"""LoRA SFT template for the response-generation line (task D1/D2).

Trains a small Chinese instruct model on empathy dialogue samples built by
src/response/sft_data.py. With --max-train-samples/--max-steps it runs as a
smoke: finite loss, nonzero LoRA gradient, adapter weights change, the saved
adapter is reloaded offline into a fresh base model and reproduces the same
deterministic generations, and the model produces non-empty replies.
Training runs one shuffled pass per configured epoch; step timing covers a
full gradient-accumulation window with the device synchronized. Local smoke
runs (CUDA/MPS/CPU) never count as platform GPU hours.

--train-format b1 trains on the frozen B1 response split; gold conditions
join from --condition-jsonl (the B1 classification split) by sample_id.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl  # noqa: E402
from src.response.sft_data import (  # noqa: E402
    build_sft_samples,
    build_sft_samples_from_b1,
    response_turn_consistency,
)

# Optimizer steps whose timing is discarded before averaging: the first steps
# absorb cuDNN autotuning and allocator warm-up, so the workplan requires
# measuring after warm-up.
TIMING_WARMUP_STEPS = 3
# Failure margin the training workplan requires on top of any ETA estimate.
ETA_MARGIN = 1.3


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
        import peft
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install the optional training dependencies with "
            "python -m pip install -r requirements-response.txt"
        ) from error
    return {
        "torch": torch,
        "transformers": transformers,
        "peft": peft,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
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


def _lora_state(model: Any) -> dict[str, Any]:
    """All LoRA tensors as {saved-adapter key: cpu tensor}."""
    state: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        state[_normalize_adapter_key(name)] = parameter.detach().cpu().clone()
    if not state:
        raise RuntimeError("no LoRA parameter found on model")
    return state


def _normalize_adapter_key(name: str) -> str:
    # peft names live tensors "...lora_A.default.weight"; saved adapter files
    # drop the adapter name: "...lora_A.weight".
    return name.replace(".default.weight", ".weight")


def _validate_run_limits(
    max_train_samples: int | None,
    max_steps: int | None,
    eval_prompts: int,
) -> None:
    """Reject impossible run limits before any model is loaded."""
    for option_name, value in (
        ("max_train_samples", max_train_samples),
        ("max_steps", max_steps),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{option_name} must be positive")
    if eval_prompts < 0:
        raise ValueError("eval_prompts must be non-negative")


def _epoch_micro_batches(
    sample_count: int, batch_size: int, epochs: int, seed: int
) -> list[list[list[int]]]:
    """Per-epoch micro-batch index plan (pure data, no torch).

    Every epoch draws its own permutation from one seeded RNG stream, so
    multi-epoch training does not replay the file order. Returning index
    lists keeps the shuffle contract unit-testable without the training
    dependencies.
    """
    rng = random.Random(seed)
    plan: list[list[list[int]]] = []
    for _ in range(epochs):
        order = list(range(sample_count))
        rng.shuffle(order)
        plan.append(
            [order[start : start + batch_size] for start in range(0, sample_count, batch_size)]
        )
    return plan


def _synchronize(torch: Any, device: Any) -> None:
    """Block until queued device work finishes so wall-clock timing is real."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


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


def _code_provenance() -> dict[str, Any]:
    """Git commit and dirty state of the code producing this run."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(PROJECT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": bool(status)}


def _reproduction_command(args: argparse.Namespace) -> str:
    """The exact command line that reproduces this run."""
    parts = [
        "python scripts/train_response_sft.py",
        f"--train '{_portable_path(args.train)}'",
        f"--config '{_portable_path(args.config)}'",
        f"--device {args.device}",
        f"--output-dir '{_portable_path(args.output_dir)}'",
        f"--report-json '{_portable_path(args.report_json)}'",
        f"--report-md '{_portable_path(args.report_md)}'",
    ]
    if args.train_format != "raw":
        parts.append(f"--train-format {args.train_format}")
    if args.condition_jsonl is not None:
        parts.append(f"--condition-jsonl '{_portable_path(args.condition_jsonl)}'")
    if args.model_dir:
        parts.append(f"--model-dir '{_portable_path(args.model_dir)}'")
    if args.max_train_samples is not None:
        parts.append(f"--max-train-samples {args.max_train_samples}")
    if args.max_steps is not None:
        parts.append(f"--max-steps {args.max_steps}")
    if args.eval_prompts != 3:
        parts.append(f"--eval-prompts {args.eval_prompts}")
    if args.local_files_only:
        parts.append("--local-files-only")
    return " \\\n  ".join(parts)


def _verify_adapter_reload(
    deps: dict[str, Any],
    load_name: str,
    load_kwargs: dict[str, Any],
    adapter_dir: Path,
    tokenizer: Any,
    device: Any,
    use_bf16: bool,
    probe: list[dict[str, Any]],
    max_new_tokens: int,
    reference_outputs: list[dict[str, str]],
) -> dict[str, Any]:
    """Offline reload check: fresh base model + saved adapter from disk.

    The training model must already be freed. Every LoRA tensor in the saved
    adapter file is compared against the reloaded model, and greedy decoding
    on the probe prompts must reproduce the in-memory model's texts exactly.
    A check that was not executed can never be reported as passed.
    """
    torch = deps["torch"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    PeftModel = deps["PeftModel"]
    from safetensors.torch import load_file

    base = AutoModelForCausalLM.from_pretrained(
        load_name,
        **load_kwargs,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    reloaded = PeftModel.from_pretrained(base, str(adapter_dir), local_files_only=True)
    reloaded.to(device)
    reloaded.eval()

    saved = load_file(adapter_dir / "adapter_model.safetensors")
    live = _lora_state(reloaded)
    missing = sorted(set(saved) - set(live))
    unmatched = sorted(
        key for key in set(saved) & set(live) if not torch.equal(saved[key], live[key])
    )
    weights_match = bool(saved) and not missing and not unmatched

    reloaded_outputs = _generate_samples(
        reloaded, tokenizer, torch, device, probe, max_new_tokens
    )
    reference_by_id = {item["id"]: item["generated_text"] for item in reference_outputs}
    generation_mismatched = [
        item["id"]
        for item in reloaded_outputs
        if reference_by_id.get(item["id"]) != item["generated_text"]
    ]
    generation_match = bool(probe) and not generation_mismatched

    del reloaded, base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "method": (
            "offline PeftModel.from_pretrained on the saved adapter; all LoRA "
            "tensors compared and greedy generations reproduced"
        ),
        "adapter_directory": _portable_path(adapter_dir),
        "lora_tensors_in_file": len(saved),
        "lora_tensors_compared": len(set(saved) & set(live)),
        "lora_weights_match": bool(weights_match),
        "lora_weight_mismatches": unmatched,
        "lora_weight_missing_in_model": missing,
        "generation_prompts": len(probe),
        "generation_match": bool(generation_match),
        "generation_mismatched_ids": generation_mismatched,
        "reloaded_outputs": reloaded_outputs,
        "passed": bool(weights_match and generation_match),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = _load_optional_dependencies()
    torch = deps["torch"]
    transformers = deps["transformers"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    LoraConfig = deps["LoraConfig"]
    PeftModel = deps["PeftModel"]
    get_peft_model = deps["get_peft_model"]

    config = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_run_limits(args.max_train_samples, args.max_steps, args.eval_prompts)
    train_rows = read_jsonl(args.train)
    condition_sha256: str | None = None
    if args.train_format == "b1":
        # Frozen B1 response split: gold conditions live in the classification
        # JSONL and join by sample_id; a partial join is an error, never a
        # silent fallback to the unconditioned prompt.
        condition_rows = None
        if str(config["condition_mode"]) == "gold":
            if args.condition_jsonl is None:
                raise ValueError(
                    "--train-format b1 with condition_mode=gold requires "
                    "--condition-jsonl (the B1 classification split)"
                )
            condition_rows = read_jsonl(args.condition_jsonl)
            condition_sha256 = _sha256(args.condition_jsonl)
        train_examples = build_sft_samples_from_b1(
            train_rows,
            condition_rows=condition_rows,
            condition_mode=str(config["condition_mode"]),
            max_history_turns=int(config["max_history_turns"]),
        )
        consistency = None
    else:
        consistency = response_turn_consistency(train_rows)
        train_examples = build_sft_samples(
            train_rows,
            condition_mode=str(config["condition_mode"]),
            max_history_turns=int(config["max_history_turns"]),
        )
    train_total = len(train_examples)
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
    initial_lora_state = _lora_state(model)

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

    # One shuffled pass per configured epoch; the plan holds indices so the
    # shuffle itself stays unit-testable without torch.
    epoch_plan = _epoch_micro_batches(len(train_encoded), batch_size, epochs, seed)
    collate = lambda features: _collate(tokenizer, torch, features)

    autocast_enabled = use_bf16
    first_loss = math.nan
    first_gradient_norm = math.nan
    optimizer_steps = 0
    measured_window_seconds = 0.0
    timed_optimizer_steps = 0
    loss_accumulator = 0.0
    micro_batches_done = 0
    training_started = time.perf_counter()
    model.train()
    stop = False
    for epoch_index, micro_batches in enumerate(epoch_plan):
        print(
            f"epoch={epoch_index + 1}/{epochs} micro_batches={len(micro_batches)} "
            f"device={device}",
            flush=True,
        )
        optimizer.zero_grad(set_to_none=True)
        for micro_index, indices in enumerate(micro_batches):
            if args.max_steps is not None and optimizer_steps >= args.max_steps:
                stop = True
                break
            # Time a full gradient-accumulation window: the clock starts at the
            # window's first micro batch and stops after the optimizer step,
            # with the device synchronized at both ends. Timing a single micro
            # batch would undercount the window by (accumulation - 1) batches.
            if micro_index % accumulation == 0:
                _synchronize(torch, device)
                window_started = time.perf_counter()
            batch = collate([train_encoded[index] for index in indices])
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
                _synchronize(torch, device)
                window_seconds = time.perf_counter() - window_started
                if optimizer_steps > TIMING_WARMUP_STEPS:
                    measured_window_seconds += window_seconds
                    timed_optimizer_steps += 1
                if args.max_steps is not None and optimizer_steps >= args.max_steps:
                    stop = True
                    break
                if optimizer_steps % 10 == 0:
                    print(
                        f"step={optimizer_steps}/{effective_total_steps} "
                        f"train_loss={loss_accumulator / max(micro_batches_done, 1):.4f} device={device}",
                        flush=True,
                    )
        if stop:
            break
    training_seconds = time.perf_counter() - training_started
    if optimizer_steps < 1:
        raise RuntimeError("training completed no optimizer steps")

    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    # The generation probe runs on the in-memory trained model and doubles as
    # the reference for the offline reload check, which needs at least one
    # prompt. --eval-prompts 0 only shrinks the report probe, never the check.
    eval_prompts = int(args.eval_prompts)
    probe = train_examples[: min(max(eval_prompts, 1), len(train_examples))]
    generation_results = _generate_samples(
        model, tokenizer, torch, device, probe, int(config["max_new_tokens"])
    )
    non_empty_generations = sum(1 for item in generation_results if item["generated_text"])

    # Free the training model before reloading so the second copy of the base
    # model fits in the same VRAM budget.
    del model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reload_check = _verify_adapter_reload(
        deps,
        load_name,
        load_kwargs,
        args.output_dir,
        tokenizer,
        device,
        use_bf16,
        probe,
        int(config["max_new_tokens"]),
        generation_results,
    )
    if not reload_check["passed"]:
        raise RuntimeError(
            "offline adapter reload check failed: "
            f"weights_match={reload_check['lora_weights_match']} "
            f"generation_match={reload_check['generation_match']} "
            f"mismatched_ids={reload_check['generation_mismatched_ids']}"
        )

    from safetensors.torch import load_file

    saved_lora = load_file(args.output_dir / "adapter_model.safetensors")
    lora_weights_changed = any(
        key not in saved_lora or not torch.equal(saved_lora[key], value)
        for key, value in initial_lora_state.items()
    )

    seconds_per_step = (
        measured_window_seconds / timed_optimizer_steps
        if timed_optimizer_steps
        else training_seconds / optimizer_steps
    )
    full_steps_per_epoch = _steps_per_epoch(train_total, batch_size, accumulation)
    eta_seconds = seconds_per_step * full_steps_per_epoch * epochs
    vram_peak = (
        int(torch.cuda.max_memory_allocated())
        if device.type == "cuda"
        else None
    )
    adapter_files = sorted(args.output_dir.glob("*.safetensors"))
    adapter_path = adapter_files[0] if adapter_files else None
    provenance = _code_provenance()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    report = {
        "version": "response_sft_prep_v2",
        "status": "smoke" if smoke_mode else "candidate training",
        "model": {
            "name": model_name,
            "revision": model_revision,
            "card_url": f"https://huggingface.co/{model_name}",
            "loaded_from": _portable_path(Path(load_name)) if args.model_dir else model_name,
        },
        "device": str(device),
        "device_name": device_name,
        "dtype": "bfloat16" if use_bf16 else "float32",
        "seed": seed,
        "config": config,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "peft": str(deps["peft"].__version__),
            "training_seconds": round(training_seconds, 3),
        },
        "reproduction": {
            "command": _reproduction_command(args),
            "code_commit": provenance["commit"],
            "code_dirty": provenance["dirty"],
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "data": {
            "train_path": _portable_path(args.train),
            "train_format": args.train_format,
            "train_total_samples": train_total,
            "train_used_samples": len(train_encoded),
            "skipped_long_samples": context_trimmed,
            "train_sha256": _sha256(args.train),
            "condition_train_sha256": condition_sha256,
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
            "timing_warmup_steps": TIMING_WARMUP_STEPS,
            "timed_optimizer_steps": timed_optimizer_steps,
            "measured_window_seconds": round(measured_window_seconds, 3),
            "seconds_per_step": round(seconds_per_step, 6),
            "seconds_per_step_basis": (
                "mean full gradient-accumulation window after warm-up"
                if timed_optimizer_steps
                else "wall-clock fallback: run shorter than the warm-up skip"
            ),
            "full_steps_per_epoch": full_steps_per_epoch,
            "estimated_full_train_seconds": round(eta_seconds, 3),
            "estimated_full_train_seconds_with_30pct_margin": round(eta_seconds * ETA_MARGIN, 3),
            "eta_note": (
                "small-sample extrapolation, budget reference only; workplan "
                f"requires a +30% failure margin on top of the raw estimate"
            ),
        },
        "vram_peak_bytes": vram_peak,
        "checks": {
            "optimizer_steps": optimizer_steps,
            "first_loss": round(first_loss, 6) if math.isfinite(first_loss) else None,
            "first_step_gradient_finite_nonzero": bool(
                math.isfinite(first_gradient_norm) and first_gradient_norm > 0
            ),
            "first_step_lora_gradient_norm": (
                round(first_gradient_norm, 8) if math.isfinite(first_gradient_norm) else None
            ),
            "lora_tensors": len(initial_lora_state),
            "lora_weights_changed": bool(lora_weights_changed),
            "adapter_reload": reload_check,
            "generation_probe": generation_results,
            "non_empty_generations": non_empty_generations,
        },
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
        f"(+30% margin {eta_seconds * ETA_MARGIN:.0f}s) "
        f"reload={reload_check['passed']} "
        f"vram={vram_peak if vram_peak is not None else 'n/a'}",
        flush=True,
    )
    return report


def _weight_storage_note() -> str:
    """Card-time wording that stays true on local and platform GPUs alike.

    The old wording claimed local runs never count as platform GPU hours,
    which is false once the same script trains on a rented platform card.
    """
    return (
        "权重只保存在忽略目录，不提交 Git；GPU 卡时按 TRAINING_WORKPLAN 与平台账单记录"
        "（本报告不声称运行是否计入平台卡时，以平台账单为准）。"
    )


def _render_report(report: dict[str, Any]) -> str:
    data = report["data"]
    speed = report["training_speed"]
    checks = report["checks"]
    reload = checks["adapter_reload"]
    reproduction = report["reproduction"]
    runtime = report["runtime"]
    lines = [
        "# D 岗位回复 SFT 训练运行报告",
        "",
        f"**状态：{report['status']}。**",
        "",
        f"模型：[`{report['model']['name']}`]({report['model']['card_url']})，"
        f"revision `{report['model']['revision']}`，本次加载自 "
        f"`{report['model']['loaded_from']}`；设备 `{report['device']}`"
        f"（{report['device_name']}，{report['dtype']}）；seed `{report['seed']}`。",
        f"环境：Python {runtime['python']}、torch {runtime['torch']}、"
        f"transformers {runtime['transformers']}、peft {runtime['peft']}。",
        f"LoRA：r={report['config']['lora_r']} alpha={report['config']['lora_alpha']} "
        f"dropout={report['config']['lora_dropout']}；"
        f"可训练参数 {report['parameters']['trainable']:,} "
        f"（{report['parameters']['fraction']:.2%} of {report['parameters']['total']:,}）。",
        f"样本：{data['train_used_samples']}/{data['train_total_samples']} 个 SFT 样本"
        f"（超长跳过 {data['skipped_long_samples']}）；条件注入 `{data['condition_mode']}`；"
        f"最大历史轮数 {data['max_history_turns']}；数据格式 `{data['train_format']}`。",
        f"token 长度（prompt+回复）：min {data['prompt_plus_response_tokens']['min']} / "
        f"mean {data['prompt_plus_response_tokens']['mean']} / "
        f"max {data['prompt_plus_response_tokens']['max']}（max_length {report['config']['max_length']}）。",
    ]
    if data.get("response_turn_consistency"):
        lines.append(
            f"回复 turn 与目标一致性：{data['response_turn_consistency']['match']}/"
            f"{data['response_turn_consistency']['total']} 与转写正文一致。"
        )
    else:
        lines.append(
            "回复 turn 与目标一致性：B1 转换产物已由 prepare_training_data.py 校验"
            "（target.response_text 与助手转写不一致会进 anomalies.jsonl；本批异常均为"
            "邻接/上下文问题，无金标不一致）。"
        )
    lines.extend(
        [
            "",
            f"测速：{speed['optimizer_steps']} 个 optimizer steps，"
            f"其中 {speed['timed_optimizer_steps']} 个计入平均（前 {speed['timing_warmup_steps']} 个 "
            f"warm-up step 不计）；每步为一个完整梯度累积窗口"
            f"（batch {report['config']['batch_size']} × 累积 {report['config']['gradient_accumulation']}），"
            f"计时两端同步设备；{speed['seconds_per_step']:.4f} 秒/step"
            f"（{speed['seconds_per_step_basis']}）。",
            f"全量 {speed['full_steps_per_epoch']} steps/epoch × {report['config']['epochs']} epoch，"
            f"估算 {speed['estimated_full_train_seconds']:.0f} 秒"
            f"（+30% 故障余量 {speed['estimated_full_train_seconds_with_30pct_margin']:.0f} 秒）；"
            f"{speed['eta_note']}。",
        ]
    )
    if report.get("vram_peak_bytes") is not None:
        lines.append(f"显存峰值：{report['vram_peak_bytes'] / 1024**3:.2f} GiB。")
    lines.extend(
        [
            "",
            f"检查：{checks['optimizer_steps']} steps；first loss {checks['first_loss']}；"
            f"梯度有限非零={checks['first_step_gradient_finite_nonzero']}"
            f"（范数 {checks['first_step_lora_gradient_norm']}）；"
            f"LoRA 权重变化={checks['lora_weights_changed']}"
            f"（{checks['lora_tensors']} 个张量）；"
            f"离线重载一致={reload['passed']}；"
            f"生成探针 {checks['non_empty_generations']}/{len(checks['generation_probe'])} 非空。",
            f"重载方式：{reload['method']}；"
            f"比对 {reload['lora_tensors_compared']}/{reload['lora_tensors_in_file']} 个 LoRA 张量"
            f"（weights_match={reload['lora_weights_match']}），"
            f"{reload['generation_prompts']} 条确定性生成复现"
            f"（generation_match={reload['generation_match']}）。",
        ]
    )
    for item in checks["generation_probe"]:
        preview = item["generated_text"][:120].replace("\n", " ")
        lines.append(f"- `{item['id']}`：{preview}…")
    lines.extend(
        [
            "",
            "复现命令：",
            "",
            "```bash",
            "python -m pip install -r requirements-response.txt",
            reproduction["command"],
            "```",
            "",
            f"运行代码：commit `{reproduction['code_commit']}`"
            f"（dirty={reproduction['code_dirty']}）；"
            f"脚本 SHA-256 `{reproduction['script_sha256']}`。",
            "",
            _weight_storage_note(),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT training template for response generation")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-format", choices=("raw", "b1"), default="raw",
                        help="raw = official train rows (turns + prediction_points); "
                             "b1 = frozen B1 response JSONL from prepare_training_data.py --mode full")
    parser.add_argument("--condition-jsonl", type=Path,
                        help="B1 classification JSONL with gold emotion_label/user_profile; "
                             "required for --train-format b1 when the config condition_mode is gold")
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
    parser.add_argument(
        "--eval-prompts",
        type=int,
        default=3,
        help=(
            "greedy generation probes recorded in the report; allowed in smoke "
            "and full runs (0 keeps only the single prompt the offline reload "
            "check needs)"
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
