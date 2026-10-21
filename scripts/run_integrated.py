#!/usr/bin/env python3
"""Offline C classifier + D generator, loaded once for the entire input."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.common.io import read_jsonl
from src.data.schema import validate_test_row, validate_prediction


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("classifier_dir", "response_model_dir", "response_adapter_dir"):
        value = Path(config[key])
        config[key] = value if value.is_absolute() else PROJECT / value
    if config.get("device") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if config.get("classifier_format", "joint") not in {"joint", "packaged", "sequence"}:
        raise ValueError("classifier_format must be joint, packaged or sequence")
    if config.get("classifier_format") == "sequence" and config.get("profile_mode") != "empty":
        raise ValueError("sequence classifier requires profile_mode=empty")
    if config.get("profile_mode", "model") not in {"model", "empty"}:
        raise ValueError("profile_mode must be model or empty")
    if type(config.get("condition_response")) is not bool:
        raise ValueError("condition_response must be a boolean")
    if type(config.get("max_new_tokens")) is not int or config["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    return config


def check_files(config):
    from scripts.predict_emotion_profile import _metadata
    checkpoint = config["classifier_dir"]
    if config.get("classifier_format") == "sequence":
        for directory, names in ((checkpoint, ("config.json", "model.safetensors", "tokenizer_config.json")),
                                 (config["response_model_dir"], ("config.json", "model.safetensors", "tokenizer_config.json")),
                                 (config["response_adapter_dir"], ("adapter_config.json", "adapter_model.safetensors"))):
            for name in names:
                if not (directory / name).is_file():
                    raise FileNotFoundError(directory / name)
        return
    if config.get("classifier_format", "joint") == "packaged":
        metadata = json.loads((checkpoint / "checkpoint_metadata.json").read_text(encoding="utf-8"))
        if "profile_thresholds" not in metadata:
            raise ValueError("Packaged checkpoint lacks profile_thresholds")
    else:
        _metadata(checkpoint)
    for path in (checkpoint / "heads.safetensors", checkpoint / "encoder/config.json",
                 config["response_model_dir"] / "config.json",
                 config["response_adapter_dir"] / "adapter_config.json",
                 config["response_adapter_dir"] / "adapter_model.safetensors"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}")
    for directory in (checkpoint / "encoder", config["response_model_dir"]):
        if not any(directory.glob("*.safetensors")) and not any(directory.glob("pytorch_model*.bin")):
            raise FileNotFoundError(f"Missing model weights in {directory}")
    for directory in (checkpoint, config["response_model_dir"]):
        if not (directory / "tokenizer_config.json").is_file():
            raise FileNotFoundError(f"Missing tokenizer_config.json in {directory}")


class Models:
    def __init__(self, config):
        from scripts.predict_emotion_profile import _metadata
        from scripts.train_emotion_profile import _load_dependencies, _make_model, _select_device
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        torch, (AutoModel, AutoTokenizer), (load_file, _) = _load_dependencies()
        self.torch = torch
        self.device = _select_device(torch, config["device"])
        torch.manual_seed(20261014)
        self.config = config
        checkpoint = config["classifier_dir"]
        self.packaged = config.get("classifier_format", "joint") == "packaged"
        self.sequence = config.get("classifier_format") == "sequence"
        if self.sequence:
            from transformers import AutoModelForSequenceClassification
            from src.classification.emotion_encoder import EMOTION_LABELS
            self.c_tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            self.c_tokenizer.truncation_side = "left"
            self.classifier = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(self.device).eval()
            if self.classifier.config.num_labels != len(EMOTION_LABELS):
                raise ValueError("Sequence classifier must have 16 labels")
        elif self.packaged:
            from src.classification.emotion_profile_checkpoint import EmotionProfileCheckpoint
            self.classifier = EmotionProfileCheckpoint(checkpoint, str(self.device))
        else:
            self.metadata = _metadata(checkpoint)
            self.c_tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            self.classifier = _make_model(torch, AutoModel.from_pretrained(checkpoint / "encoder", local_files_only=True))
            incompatible = self.classifier.load_state_dict(load_file(str(checkpoint / "heads.safetensors")), strict=False)
            if incompatible.unexpected_keys or any(not key.startswith("encoder.") for key in incompatible.missing_keys):
                raise ValueError("Classifier heads are incompatible")
            self.classifier.to(self.device).eval()
        self.d_tokenizer = AutoTokenizer.from_pretrained(config["response_model_dir"], local_files_only=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(config["response_model_dir"], local_files_only=True, torch_dtype=dtype)
        self.generator = PeftModel.from_pretrained(base, config["response_adapter_dir"], local_files_only=True).to(self.device).eval()

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def predict(self, row):
        from scripts.predict_emotion_profile import _predict_group
        from src.response.sft_data import build_eval_sample
        if self.sequence:
            from src.classification.emotion_encoder import EMOTION_LABELS, _format_context
            from src.data.schema import PROFILE_LABELS
            text = _format_context(row["history"], row["target_user_turn_id"], row["conversation_id"])
            tokens = self.c_tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                label = int(self.classifier(**tokens).logits.argmax(dim=-1).item())
            classification = {"id": row["id"], "emotion_label": EMOTION_LABELS[label],
                              "user_profile": {key: [] for key in PROFILE_LABELS}}
        elif self.packaged:
            classification = {"id": row["id"], **self.classifier.predict_contexts([row["history"]])[0]}
        else:
            classification = _predict_group([row], self.classifier, self.c_tokenizer, self.torch,
                                            self.device, int(self.metadata["max_length"]), float(self.metadata["profile_threshold"]))[0]
        if self.config.get("profile_mode", "model") == "empty":
            classification["user_profile"] = {key: [] for key in classification["user_profile"]}
        condition = classification if self.config["condition_response"] else None
        sample = build_eval_sample(row, condition=condition)
        prompt = self.d_tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=True)
        inputs = self.d_tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.generator.generate(**inputs, max_new_tokens=self.config["max_new_tokens"], do_sample=False,
                                             use_cache=True, pad_token_id=self.d_tokenizer.pad_token_id if self.d_tokenizer.pad_token_id is not None else self.d_tokenizer.eos_token_id)
        text = self.d_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return {**classification, "response_text": text, "memory_refs": []}


def run(test_file, result_dir, config, factory=Models):
    test_file, result_dir = Path(test_file), Path(result_dir)
    rows = read_jsonl(test_file)
    if not rows:
        raise ValueError("Input contains no rows")
    for row in rows:
        validate_test_row(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate input IDs")
    result_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    models = factory(config)
    models.synchronize()
    loading = time.perf_counter() - started
    latencies = []
    descriptor, temporary = tempfile.mkstemp(dir=result_dir, suffix=".tmp")
    destination = result_dir / "submission.jsonl"
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(rows):
                models.synchronize()
                started = time.perf_counter()
                prediction = models.predict(row)
                validate_prediction(prediction, row["id"])
                models.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                if index < 100:
                    latencies.append(elapsed)
                handle.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
                print(f"[{index + 1}/{len(rows)}] {row['id']}: {elapsed:.3f} ms", flush=True)
        digest = hashlib.sha256(Path(temporary).read_bytes()).hexdigest()
        report = {
            "version": "integrated_transformers_v1", "backend": "transformers",
            "timing_scope": "classification+prompt+tokenize+generate+decode+validate; model loading excluded",
            "complete": len(latencies) == 100, "required_rounds": 100, "rounds": len(latencies),
            "average_latency_ms": statistics.fmean(latencies), "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": sorted(latencies)[math.ceil(len(latencies) * .95) - 1],
            "min_latency_ms": min(latencies), "max_latency_ms": max(latencies), "latencies_ms": latencies,
            "hardware_label": config.get("hardware_label", "unspecified"), "model_loading_seconds": loading,
            "test_file_sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(), "submission_sha256": digest,
            "samples": len(rows), "parse_failure_count": 0, "parse_failure_ids": [],
            "condition_mode": "predicted" if config["condition_response"] else "none",
            "profile_mode": config.get("profile_mode", "model"),
        }
        os.replace(temporary, destination)
        (result_dir / "performance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_file", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/integrated.json")
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    config = load_config(args.config)
    check_files(config)
    run(args.test_file, args.result_dir, config)


if __name__ == "__main__":
    main()
