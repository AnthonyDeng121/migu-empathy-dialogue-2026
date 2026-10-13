#!/usr/bin/env python3
"""Run the (optionally adapter-tuned) generation model on eval input rows.

This is the response-line inference half used for the three-way comparison
(retrieval / original model / fine-tuned model): it only produces
response_text plus timing; emotion and profile come from the classification
line and memory_refs stays []. Throughput on the first N samples feeds the
full-evaluation ETA required by the training workplan.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl, write_jsonl  # noqa: E402
from src.response.sft_data import build_eval_sample  # noqa: E402


def _load_optional_dependencies() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install the optional training dependencies with "
            "python -m pip install -r requirements-response.txt"
        ) from error
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


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
    return torch.device(requested)


def run(args: argparse.Namespace) -> dict[str, Any]:
    deps = _load_optional_dependencies()
    torch = deps["torch"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    PeftModel = deps["PeftModel"]

    rows = read_jsonl(args.input)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("eval input contains no rows")
    device = _device(torch, args.device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        local_files_only=True,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True)
    model.to(device)
    model.eval()

    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    generated_tokens = 0
    started = time.perf_counter()
    for row in rows:
        sample = build_eval_sample(row, condition=None)
        prompt_text = tokenizer.apply_chat_template(
            sample["messages"], add_generation_prompt=True, tokenize=False
        )
        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)
        sample_started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        latencies.append(time.perf_counter() - sample_started)
        generated = output_ids[0][prompt_ids.shape[1]:]
        generated_tokens += int(generated.numel())
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        predictions.append({"id": row["id"], "response_text": text})
    total_seconds = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, predictions)
    report = {
        "version": "response_generation_inference_v1",
        "model_path": str(args.model),
        "adapter_path": str(args.adapter) if args.adapter else None,
        "device": str(device),
        "dtype": "bfloat16" if use_bf16 else "float32",
        "condition_mode": "none",
        "input_path": str(args.input),
        "samples": len(predictions),
        "max_new_tokens": int(args.max_new_tokens),
        "total_seconds": round(total_seconds, 3),
        "mean_seconds_per_sample": round(total_seconds / len(predictions), 4),
        "p50_seconds_per_sample": round(sorted(latencies)[len(latencies) // 2], 4),
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(generated_tokens / total_seconds, 2),
        "full_set_estimate_seconds": round(
            total_seconds / len(predictions) * int(args.full_set_size), 1
        ),
        "full_set_size": int(args.full_set_size),
        "empty_responses": sum(1 for item in predictions if not item["response_text"]),
    }
    args.performance_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"PASS: {len(predictions)} samples in {total_seconds:.1f}s "
        f"({report['mean_seconds_per_sample']:.3f}s/sample, "
        f"{report['tokens_per_second']:.1f} tok/s)",
        flush=True,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses for eval input rows")
    parser.add_argument("--input", type=Path, default=PROJECT / "data/processed/val_eval_input.jsonl")
    parser.add_argument("--model", type=Path, required=True, help="local model snapshot directory")
    parser.add_argument("--adapter", type=Path, help="optional LoRA adapter directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--performance-report", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--full-set-size", type=int, default=2354)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
