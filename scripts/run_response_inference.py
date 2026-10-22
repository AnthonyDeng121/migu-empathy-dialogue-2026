#!/usr/bin/env python3
"""Run the (optionally adapter-tuned) generation model on eval input rows.

This is the response-line inference half used for the three-way comparison
(retrieval / original model / fine-tuned model): it only produces
response_text plus timing; emotion and profile come from the classification
line and memory_refs stays []. Throughput on the first N samples feeds the
full-evaluation ETA required by the training workplan.

--condition-jsonl feeds C's predicted emotion_label/user_profile per sample
id (official submission-schema fields) so the training-time gold conditioning
can be replaced by predicted conditions at inference; every eval id must be
covered, otherwise the conditioned/unconditioned prompts would silently mix.
--condition-fields selects which of the two condition fields enter the prompt:
"both" (default, every recorded run) or "emotion" (cycle-2 emotion-only arm).

Eval rows may be official test rows (id / history) or frozen B1 response
rows (sample_id / context_turns); both resolve to the same prompt structure.
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
from src.response.sft_data import build_eval_sample, eval_row_id  # noqa: E402


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


def _load_conditions(path: Path, fields: str = "both") -> dict[str, dict[str, Any]]:
    """Predicted conditions keyed by sample id (C's classification output).

    fields="both" is the default and reproduces every recorded run:
    emotion_label plus a dict user_profile are both required. fields="emotion"
    is the cycle-2 emotion-only ablation arm: the profile may be absent or
    null and is dropped, so the prompt carries only the 情感状态 line. The
    same C condition file drives both arms.
    """
    if fields not in ("both", "emotion"):
        raise ValueError(f"unknown condition fields: {fields!r}")
    conditions: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if not sample_id:
            raise ValueError(f"condition row without id in {path}")
        if fields == "emotion":
            if "emotion_label" not in row or not isinstance(row["emotion_label"], str):
                raise ValueError(f"condition row {sample_id} lacks a string emotion_label")
            conditions[sample_id] = {"emotion_label": row["emotion_label"]}
            continue
        if "emotion_label" not in row or not isinstance(row.get("user_profile"), dict):
            raise ValueError(
                f"condition row {sample_id} lacks emotion_label/user_profile"
            )
        conditions[sample_id] = {
            "emotion_label": row["emotion_label"],
            "user_profile": row["user_profile"],
        }
    if not conditions:
        raise ValueError(f"condition file contains no rows: {path}")
    return conditions


def condition_coverage(
    rows: list[dict[str, Any]], conditions: dict[str, dict[str, Any]]
) -> str | None:
    """Predicted-condition coverage as "covered/total", or None without conditions.

    Ids resolve through eval_row_id so the official test schema (id) and the
    frozen B1 schema (sample_id) are both counted. Reading row["id"] directly
    raised KeyError on B1 splits while writing the performance report, i.e.
    only after the whole inference had already run.
    """
    if not conditions:
        return None
    covered = sum(1 for row in rows if eval_row_id(row) in conditions)
    return f"{covered}/{len(rows)}"


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
    conditions = (
        _load_conditions(args.condition_jsonl, args.condition_fields)
        if args.condition_jsonl
        else {}
    )
    if conditions:
        missing = [eval_row_id(row) for row in rows if eval_row_id(row) not in conditions]
        if missing:
            raise ValueError(
                f"{len(missing)} eval ids lack a predicted condition "
                f"(first: {missing[:5]}); refusing to mix conditioned and "
                "unconditioned prompts"
            )
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
        row_id = eval_row_id(row)
        sample = build_eval_sample(row, condition=conditions.get(row_id))
        prompt_text = tokenizer.apply_chat_template(
            sample["messages"], add_generation_prompt=True, tokenize=False
        )
        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)
        sample_started = time.perf_counter()
        generate_kwargs: dict[str, Any] = {}
        if args.min_new_tokens:
            generate_kwargs["min_new_tokens"] = int(args.min_new_tokens)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                **generate_kwargs,
            )
        latencies.append(time.perf_counter() - sample_started)
        generated = output_ids[0][prompt_ids.shape[1]:]
        generated_tokens += int(generated.numel())
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        predictions.append({"id": row_id, "response_text": text})
    total_seconds = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, predictions)
    report = {
        "version": "response_generation_inference_v1",
        "model_path": str(args.model),
        "adapter_path": str(args.adapter) if args.adapter else None,
        "device": str(device),
        "dtype": "bfloat16" if use_bf16 else "float32",
        "condition_mode": "predicted" if conditions else "none",
        "condition_source": str(args.condition_jsonl) if args.condition_jsonl else None,
        "condition_fields": args.condition_fields if args.condition_jsonl else None,
        "condition_coverage": condition_coverage(rows, conditions),
        "input_path": str(args.input),
        "samples": len(predictions),
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(args.min_new_tokens),
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
    parser.add_argument(
        "--condition-jsonl",
        type=Path,
        help=(
            "predicted emotion_label/user_profile per sample id (C's output); "
            "when given, every eval id must be covered"
        ),
    )
    parser.add_argument(
        "--condition-fields",
        choices=("both", "emotion"),
        default="both",
        help=(
            "which condition fields to inject from --condition-jsonl; both "
            "(default) reproduces every recorded run, emotion is the cycle-2 "
            "emotion-only ablation arm (profile dropped from the prompt)"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=0,
        help=(
            "floor on generated tokens; the D2 arm emits 32 near-empty replies "
            "(<=6 chars, 1.4%%) that the original model never produces, so this "
            "is the D3 probe for that failure mode"
        ),
    )
    parser.add_argument("--full-set-size", type=int, default=2354)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
