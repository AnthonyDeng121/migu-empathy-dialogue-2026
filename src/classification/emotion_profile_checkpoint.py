"""Load and run a packaged emotion + profile checkpoint."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from src.data.schema import EMOTIONS, PROFILE_LABELS


def read_packaged_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Validate the existing per-dimension format without importing ML dependencies.

    Metadata label order defines head indices; permutations are valid, duplicates are not.
    """
    metadata = json.loads((Path(checkpoint_dir) / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    emotions = metadata.get("emotion_labels")
    if not isinstance(emotions, list) or len(emotions) != len(EMOTIONS) or set(emotions) != EMOTIONS:
        raise ValueError("checkpoint emotion labels do not match the official schema")
    profiles = metadata.get("profile_labels")
    if not isinstance(profiles, dict) or set(profiles) != set(PROFILE_LABELS):
        raise ValueError("checkpoint profile dimensions do not match the official schema")
    for dim, allowed in PROFILE_LABELS.items():
        labels = profiles[dim]
        if not isinstance(labels, list) or len(labels) != len(allowed) or set(labels) != allowed:
            raise ValueError(f"checkpoint labels do not match the official {dim} schema")
    thresholds = metadata.get("profile_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(PROFILE_LABELS):
        raise ValueError("checkpoint must include one calibrated threshold per dimension")
    for dim, value in thresholds.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid checkpoint profile_thresholds.{dim}; expected a number in [0, 1]")
    if type(metadata.get("max_length")) is not int or metadata["max_length"] < 1:
        raise ValueError("checkpoint max_length must be a positive integer")
    return metadata


def check_packaged_files(checkpoint_dir: str | Path) -> None:
    path = Path(checkpoint_dir)
    read_packaged_metadata(path)
    for name in ("heads.safetensors", "encoder/config.json", "tokenizer_config.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing profile model file: {path / name}")
    encoder = path / "encoder"
    if not any(encoder.glob("*.safetensors")) and not any(encoder.glob("pytorch_model*.bin")):
        raise FileNotFoundError(f"Missing profile encoder weights in {encoder}")


class EmotionProfileCheckpoint:
    """Shared MacBERT encoder with a preserved emotion head and calibrated profile heads."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cpu") -> None:
        try:
            import torch
            from safetensors.torch import load_file
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Install the inference dependencies before loading this checkpoint"
            ) from error

        self._torch = torch
        self.device = torch.device(device)
        self.path = Path(checkpoint_dir)
        self.metadata = read_packaged_metadata(self.path)
        self.emotion_labels = tuple(self.metadata["emotion_labels"])
        self.profile_labels = {dim: tuple(labels) for dim, labels in self.metadata["profile_labels"].items()}
        self.thresholds = {dim: float(value) for dim, value in self.metadata["profile_thresholds"].items()}

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True
        )
        self.tokenizer.truncation_side = "left"
        self.encoder = AutoModel.from_pretrained(
            self.path / "encoder", local_files_only=True
        )
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = torch.nn.Dropout(float(self.encoder.config.hidden_dropout_prob))
        self.emotion_head = torch.nn.Linear(hidden_size, len(self.emotion_labels))
        self.profile_heads = torch.nn.ModuleDict({
            dim: torch.nn.Linear(hidden_size, len(labels))
            for dim, labels in self.profile_labels.items()
        })
        state = load_file(str(self.path / "heads.safetensors"), device="cpu")
        incompatible = self.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys or incompatible.missing_keys:
            raise ValueError(
                "checkpoint head state mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        self.to(self.device)
        self.eval()

    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> Any:
        own_state = {
            **{f"emotion_head.{key}": value for key, value in self.emotion_head.state_dict().items()},
            **{
                f"profile_heads.{key}": value
                for key, value in self.profile_heads.state_dict().items()
            },
        }
        missing = [key for key in own_state if key not in state]
        unexpected = [key for key in state if key not in own_state]
        if strict and (missing or unexpected):
            raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")
        if not missing and not unexpected:
            self.emotion_head.load_state_dict({
                key.removeprefix("emotion_head."): value
                for key, value in state.items() if key.startswith("emotion_head.")
            })
            self.profile_heads.load_state_dict({
                key.removeprefix("profile_heads."): value
                for key, value in state.items() if key.startswith("profile_heads.")
            })
        return type("StateLoadResult", (), {"missing_keys": missing, "unexpected_keys": unexpected})()

    def to(self, device: Any) -> "EmotionProfileCheckpoint":
        self.device = self._torch.device(device)
        self.encoder.to(self.device)
        self.emotion_head.to(self.device)
        self.profile_heads.to(self.device)
        self.dropout.to(self.device)
        return self

    def eval(self) -> "EmotionProfileCheckpoint":
        self.encoder.eval()
        self.emotion_head.eval()
        self.profile_heads.eval()
        self.dropout.eval()
        return self

    def predict_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[dict[str, Any]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        results: list[dict[str, Any]] = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = self.tokenizer(
                    list(texts[start : start + batch_size]),
                    truncation=True,
                    max_length=int(self.metadata["max_length"]),
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                pooled = self.encoder(**encoded, return_dict=True).pooler_output
                emotion_logits = self.emotion_head(self.dropout(pooled))
                profile_logits = {
                    dim: head(pooled) for dim, head in self.profile_heads.items()
                }
                emotion_ids = emotion_logits.argmax(dim=-1).cpu().tolist()
                probabilities = {
                    dim: self._torch.sigmoid(logits).cpu().tolist()
                    for dim, logits in profile_logits.items()
                }
                for row_index, emotion_id in enumerate(emotion_ids):
                    profile = {
                        dim: [
                            label
                            for label, probability in zip(
                                self.profile_labels[dim], probabilities[dim][row_index]
                            )
                            if probability >= self.thresholds[dim]
                        ]
                        for dim in self.profile_labels
                    }
                    results.append({
                        "emotion_label": self.emotion_labels[emotion_id],
                        "user_profile": profile,
                    })
        return results

    def predict_contexts(self, contexts: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
        speaker_names = {"user": "用户", "assistant": "助手"}
        texts = []
        for context in contexts:
            if not context or context[-1].get("role") != "user":
                raise ValueError("each context must end with a user turn")
            texts.append("\n".join(
                f"{speaker_names[turn['role']]}：{turn['content']}" for turn in context
            ))
        return self.predict_texts(texts)
