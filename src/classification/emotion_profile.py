"""B1 adapters and metrics for the joint emotion/profile classifier."""
from __future__ import annotations

import json
import math
from typing import Any, Sequence

from src.data.schema import EMOTIONS, PROFILE_LABELS, validate_classification_labels, validate_test_row

EMOTION_LABELS = tuple(sorted(EMOTIONS))
PROFILE_LABELS_BY_DIMENSION = {
    dimension: tuple(sorted(labels)) for dimension, labels in PROFILE_LABELS.items()
}
_B1_FIELDS = {
    "sample_id", "conversation_id", "target_user_turn_id", "source_split",
    "context_turns", "context_text", "emotion_label", "user_profile",
}
_SPEAKER_NAMES = {"user": "用户", "assistant": "助手"}


def _format_context(turns: list[dict[str, Any]]) -> str:
    return "\n".join(f"{_SPEAKER_NAMES[turn['role']]}：{turn['content']}" for turn in turns)


def _validate_context(
    *, sample_id: str, conversation_id: str, target_turn_id: int,
    context_turns: Any, context_text: Any | None = None,
) -> str:
    test_row = {
        "id": sample_id,
        "conversation_id": conversation_id,
        "target_user_turn_id": target_turn_id,
        "history": context_turns,
    }
    validate_test_row(test_row)
    expected_serialization = "\n".join(
        json.dumps(turn, ensure_ascii=False, separators=(",", ":"))
        for turn in context_turns
    )
    if context_text is not None and context_text != expected_serialization:
        raise ValueError(f"context_text does not match context_turns for sample_id {sample_id}")
    return _format_context(context_turns)


def build_b1_classification_examples(
    rows: Sequence[dict[str, Any]], *, expected_split: str
) -> list[dict[str, Any]]:
    """Validate and adapt B1 classification JSONL rows into model-ready records."""
    if expected_split not in {"train", "val"}:
        raise ValueError("expected_split must be 'train' or 'val'")
    examples: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _B1_FIELDS:
            raise ValueError(f"B1 row {row_index} fields must be exactly {sorted(_B1_FIELDS)}")
        sample_id = row["sample_id"]
        conversation_id = row["conversation_id"]
        target_turn_id = row["target_user_turn_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"B1 row {row_index} has invalid sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"B1 row {row_index} has invalid conversation_id")
        if isinstance(target_turn_id, bool) or not isinstance(target_turn_id, int) or target_turn_id < 1:
            raise ValueError(f"B1 row {row_index} has invalid target_user_turn_id")
        if row["source_split"] != expected_split:
            raise ValueError(
                f"B1 row {sample_id} source_split is {row['source_split']!r}, "
                f"expected {expected_split!r}"
            )
        context = _validate_context(
            sample_id=sample_id,
            conversation_id=conversation_id,
            target_turn_id=target_turn_id,
            context_turns=row["context_turns"],
            context_text=row["context_text"],
        )
        validate_classification_labels(row["emotion_label"], row["user_profile"])
        examples.append(
            {
                "sample_id": sample_id,
                "conversation_id": conversation_id,
                "target_user_turn_id": target_turn_id,
                "source_split": expected_split,
                "text": context,
                "emotion_label": row["emotion_label"],
                "user_profile": {
                    dimension: list(row["user_profile"][dimension])
                    for dimension in PROFILE_LABELS_BY_DIMENSION
                },
            }
        )
    if not examples:
        raise ValueError(f"{expected_split} classification dataset is empty")
    return examples


def build_inference_context(row: dict[str, Any]) -> str:
    """Format one official inference row after enforcing the target-turn cutoff."""
    validate_test_row(row)
    return _format_context(row["history"])


def validate_disjoint_splits(
    train_examples: Sequence[dict[str, Any]], val_examples: Sequence[dict[str, Any]]
) -> None:
    train_ids = {example["sample_id"] for example in train_examples}
    val_ids = {example["sample_id"] for example in val_examples}
    duplicate_ids = train_ids & val_ids
    if duplicate_ids:
        raise ValueError(f"train/val sample_id overlap: {sorted(duplicate_ids)[:5]}")
    train_conversations = {example["conversation_id"] for example in train_examples}
    val_conversations = {example["conversation_id"] for example in val_examples}
    overlap = train_conversations & val_conversations
    if overlap:
        raise ValueError(f"train/val conversation overlap: {sorted(overlap)[:5]}")


def profile_targets(user_profile: dict[str, Sequence[str]]) -> dict[str, list[int]]:
    """Encode each independent profile dimension as a deterministic multi-hot vector."""
    validate_classification_labels("neutral", user_profile)
    return {
        dimension: [int(label in user_profile[dimension]) for label in labels]
        for dimension, labels in PROFILE_LABELS_BY_DIMENSION.items()
    }


def _validated_profile(profile: dict[str, Any]) -> dict[str, list[str]]:
    validate_classification_labels("neutral", profile)
    return {dimension: list(profile[dimension]) for dimension in PROFILE_LABELS_BY_DIMENSION}


def profile_metrics(
    expected: Sequence[dict[str, Any]], predicted: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Compute exact-set accuracy and multi-label micro F1 per profile dimension."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths differ")
    if not expected:
        raise ValueError("cannot evaluate an empty profile set")
    truth = [_validated_profile(profile) for profile in expected]
    guesses = [_validated_profile(profile) for profile in predicted]
    per_dimension: dict[str, dict[str, Any]] = {}
    empty_exact: list[float] = []
    exact_values: list[float] = []
    for dimension, labels in PROFILE_LABELS_BY_DIMENSION.items():
        exact_correct = 0
        true_positive = false_positive = false_negative = 0
        for expected_profile, predicted_profile in zip(truth, guesses):
            expected_set = set(expected_profile[dimension])
            predicted_set = set(predicted_profile[dimension])
            exact_correct += expected_set == predicted_set
            true_positive += len(expected_set & predicted_set)
            false_positive += len(predicted_set - expected_set)
            false_negative += len(expected_set - predicted_set)
        accuracy = exact_correct / len(truth)
        denominator = 2 * true_positive + false_positive + false_negative
        micro_f1 = 2 * true_positive / denominator if denominator else 0.0
        per_dimension[dimension] = {
            "exact_match_accuracy": round(accuracy, 6),
            "micro_f1": round(micro_f1, 6),
            "label_count": len(labels),
            "gold_positive_count": sum(len(profile[dimension]) for profile in truth),
            "predicted_positive_count": sum(len(profile[dimension]) for profile in guesses),
        }
        empty_exact.append(sum(not profile[dimension] for profile in truth) / len(truth))
        exact_values.append(accuracy)
    return {
        "per_dimension": per_dimension,
        "profile_macro_exact_match_accuracy": round(sum(exact_values) / len(exact_values), 6),
        "empty_set_baseline": {
            "per_dimension_exact_match_accuracy": {
                dimension: round(value, 6)
                for dimension, value in zip(PROFILE_LABELS_BY_DIMENSION, empty_exact)
            },
            "profile_macro_exact_match_accuracy": round(sum(empty_exact) / len(empty_exact), 6),
        },
    }


def decode_emotion_profile_scores(
    emotion_scores: Sequence[float],
    profile_scores: dict[str, Sequence[float]],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Decode one emotion-logit vector and three profile probability vectors."""
    if len(emotion_scores) != len(EMOTION_LABELS):
        raise ValueError(f"emotion score count must be {len(EMOTION_LABELS)}")
    if not all(math.isfinite(float(score)) for score in emotion_scores):
        raise ValueError("emotion scores must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("profile threshold must be between 0 and 1")
    if set(profile_scores) != set(PROFILE_LABELS_BY_DIMENSION):
        raise ValueError("profile score dimensions must match the three official dimensions")
    profile: dict[str, list[str]] = {}
    for dimension, labels in PROFILE_LABELS_BY_DIMENSION.items():
        scores = profile_scores[dimension]
        if len(scores) != len(labels):
            raise ValueError(f"profile score count for {dimension} must be {len(labels)}")
        if not all(math.isfinite(float(score)) and 0 <= float(score) <= 1 for score in scores):
            raise ValueError(f"profile probabilities for {dimension} must be finite and within [0, 1]")
        profile[dimension] = [
            label for label, score in zip(labels, scores) if float(score) >= threshold
        ]
    emotion_index = max(range(len(emotion_scores)), key=lambda index: float(emotion_scores[index]))
    return {"emotion_label": EMOTION_LABELS[emotion_index], "user_profile": profile}
