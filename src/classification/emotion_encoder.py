from __future__ import annotations

from typing import Any, Sequence

from src.data.schema import EMOTIONS

EMOTION_LABELS = tuple(sorted(EMOTIONS))
SPEAKER_NAMES = {"user": "用户", "assistant": "助手"}


def build_emotion_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand conversations into emotion examples without exposing future turns."""
    examples: list[dict[str, Any]] = []
    for row_index, conversation in enumerate(rows):
        conversation_id = conversation.get("conversation_id")
        turns = conversation.get("turns")
        points = conversation.get("prediction_points")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"conversation {row_index} must have a conversation_id")
        if not isinstance(turns, list) or not isinstance(points, list):
            raise ValueError(f"conversation {conversation_id} has invalid turns or prediction_points")

        for point_index, point in enumerate(points):
            if not isinstance(point, dict):
                raise ValueError(f"prediction point {point_index} in {conversation_id} must be an object")
            target_turn = point.get("target_user_turn_id")
            target = point.get("target")
            if not isinstance(target_turn, int) or target_turn < 1:
                raise ValueError(f"prediction point {point_index} in {conversation_id} has invalid target turn")
            if not isinstance(target, dict):
                raise ValueError(f"prediction point {point_index} in {conversation_id} must contain a target")
            label = target.get("emotion_label")
            if not isinstance(label, str) or label not in EMOTIONS:
                raise ValueError("invalid emotion_label")
            text = _format_context(turns, target_turn, conversation_id)
            examples.append(
                {
                    "conversation_id": conversation_id,
                    "target_user_turn_id": target_turn,
                    "text": text,
                    "label": label,
                }
            )
    if not examples:
        raise ValueError("dataset contains no labeled prediction points")
    return examples


def _format_context(
    turns: list[dict[str, Any]], target_turn: int, conversation_id: str
) -> str:
    context: list[str] = []
    last_turn_id = 0
    last_context_turn_id = 0
    last_role = ""
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError(f"conversation {conversation_id} has a malformed turn")
        turn_id = turn.get("turn_id")
        if not isinstance(turn_id, int) or turn_id <= last_turn_id:
            raise ValueError(f"conversation {conversation_id} turn ids must increase")
        last_turn_id = turn_id
        if turn_id > target_turn:
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role not in SPEAKER_NAMES or not isinstance(content, str):
            raise ValueError(f"conversation {conversation_id} has an invalid context turn")
        context.append(f"{SPEAKER_NAMES[role]}：{content}")
        last_context_turn_id = turn_id
        last_role = role
    if not context or last_context_turn_id != target_turn or last_role != "user":
        raise ValueError(
            f"conversation {conversation_id} context must end at target_user_turn_id with a user turn"
        )
    return "\n".join(context)


def classification_metrics(
    expected: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    """Return accuracy, fixed-label macro F1, per-class metrics, and confusion counts."""
    label_order = tuple(labels)
    if not label_order or len(label_order) != len(set(label_order)):
        raise ValueError("labels must be a non-empty sequence of unique values")
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths differ")
    if not expected:
        raise ValueError("cannot evaluate an empty validation set")
    unknown = (set(expected) | set(predicted)) - set(label_order)
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")

    label_index = {label: index for index, label in enumerate(label_order)}
    confusion = [[0 for _ in label_order] for _ in label_order]
    correct = 0
    for truth, guess in zip(expected, predicted):
        confusion[label_index[truth]][label_index[guess]] += 1
        correct += truth == guess

    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    for index, label in enumerate(label_order):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted_count = sum(row[index] for row in confusion)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    return {
        "accuracy": round(correct / len(expected), 6),
        "macro_f1": round(sum(f1_values) / len(label_order), 6),
        "labels": list(label_order),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
