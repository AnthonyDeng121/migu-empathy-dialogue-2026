from __future__ import annotations

import re
from typing import Any

EMOTIONS = {
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
    "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
    "shame", "helplessness",
}
PROFILE_LABELS = {
    "personality_traits": {"extroverted", "introverted", "open", "conservative", "high_conscientiousness", "casual", "agreeable", "assertive", "emotionally_stable", "sensitive"},
    "interests": {"study_exam", "programming_technology", "reading_writing", "film_animation", "music", "games", "sports_fitness", "travel_outdoor", "pets", "social", "career_development", "art_design"},
    "style": {"brief", "detailed", "colloquial", "formal", "direct", "indirect", "humorous", "rational", "high_emotional_expression", "low_emotional_expression", "emoji_user"},
}
MEMORY_RE = re.compile(r"^mem_[0-9]{6}$")


def _string(value: Any, name: str, *, nonempty: bool = False) -> None:
    if not isinstance(value, str) or (nonempty and not value):
        raise ValueError(f"{name} must be {'a non-empty ' if nonempty else 'a '}string")


def validate_test_row(row: dict[str, Any]) -> None:
    required = {"id", "conversation_id", "target_user_turn_id", "history"}
    if set(row) != required:
        raise ValueError(f"test row fields must be exactly {sorted(required)}")
    _string(row["id"], "id", nonempty=True)
    _string(row["conversation_id"], "conversation_id", nonempty=True)
    if not isinstance(row["target_user_turn_id"], int) or row["target_user_turn_id"] < 1:
        raise ValueError("target_user_turn_id must be a positive integer")
    if not isinstance(row["history"], list) or not row["history"]:
        raise ValueError("history must be a non-empty array")
    ids = []
    for turn in row["history"]:
        if set(turn) != {"turn_id", "role", "content"}:
            raise ValueError("history turn has unexpected fields")
        if not isinstance(turn["turn_id"], int) or turn["turn_id"] < 1:
            raise ValueError("turn_id must be a positive integer")
        if turn["role"] not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        _string(turn["content"], "content", nonempty=True)
        ids.append(turn["turn_id"])
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("history turn_id values must be unique and ordered")
    if row["target_user_turn_id"] != ids[-1] or row["history"][-1]["role"] != "user":
        raise ValueError("history must end at target_user_turn_id with a user turn")


def validate_classification_labels(emotion_label: Any, profile: Any) -> None:
    if not isinstance(emotion_label, str) or emotion_label not in EMOTIONS:
        raise ValueError("invalid emotion_label")
    if not isinstance(profile, dict) or set(profile) != set(PROFILE_LABELS):
        raise ValueError("user_profile must contain exactly the three official dimensions")
    for dimension, allowed in PROFILE_LABELS.items():
        values = profile[dimension]
        if (
            not isinstance(values, list)
            or len(values) != len(set(values))
            or not set(values) <= allowed
        ):
            raise ValueError(f"invalid user_profile.{dimension}")


def validate_prediction(row: dict[str, Any], expected_id: str | None = None) -> None:
    required = {"id", "response_text", "emotion_label", "user_profile", "memory_refs"}
    if set(row) != required:
        raise ValueError(f"prediction fields must be exactly {sorted(required)}")
    _string(row["id"], "id", nonempty=True)
    if expected_id is not None and row["id"] != expected_id:
        raise ValueError(f"prediction id {row['id']!r} does not match input {expected_id!r}")
    _string(row["response_text"], "response_text", nonempty=True)
    validate_classification_labels(row["emotion_label"], row["user_profile"])
    refs = row["memory_refs"]
    if not isinstance(refs, list) or len(refs) != len(set(refs)) or not all(isinstance(v, str) and MEMORY_RE.fullmatch(v) for v in refs):
        raise ValueError("invalid memory_refs")

