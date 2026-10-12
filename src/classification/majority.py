from __future__ import annotations

from collections import Counter
from typing import Any, Hashable, TypeVar

from src.data.schema import PROFILE_LABELS, validate_classification_labels

Label = TypeVar("Label", bound=Hashable)


def _majority(counter: Counter[Label]) -> Label:
    if not counter:
        raise ValueError("cannot fit a majority baseline without labeled samples")
    largest = max(counter.values())
    return min((label for label, count in counter.items() if count == largest), key=str)


class MajorityClassifier:
    """Predict the most common emotion and complete profile set in training data."""

    def __init__(self, training_rows: list[dict[str, Any]]) -> None:
        emotion_counts: Counter[str] = Counter()
        profile_counts: dict[str, Counter[tuple[str, ...]]] = {
            dimension: Counter() for dimension in PROFILE_LABELS
        }
        samples = 0

        for conversation in training_rows:
            points = conversation.get("prediction_points")
            if not isinstance(points, list):
                raise ValueError("training conversation must contain prediction_points")
            for point in points:
                target = point.get("target") if isinstance(point, dict) else None
                if not isinstance(target, dict):
                    raise ValueError("prediction point must contain a target object")
                if not {"emotion_label", "user_profile"} <= target.keys():
                    raise ValueError("training target must contain emotion_label and user_profile")
                validate_classification_labels(target["emotion_label"], target["user_profile"])
                samples += 1
                emotion_counts[target["emotion_label"]] += 1
                for dimension in PROFILE_LABELS:
                    labels = tuple(sorted(target["user_profile"][dimension]))
                    profile_counts[dimension][labels] += 1

        if samples == 0:
            raise ValueError("training data contains no labeled prediction points")

        self.training_samples = samples
        self.emotion_label = _majority(emotion_counts)
        self.profile = {
            dimension: list(_majority(counts))
            for dimension, counts in profile_counts.items()
        }
        profile_support = {
            dimension: profile_counts[dimension][tuple(labels)]
            for dimension, labels in self.profile.items()
        }
        self.majority_support = {
            "emotion_label": emotion_counts[self.emotion_label],
            "user_profile": profile_support,
        }

    def predict(self) -> dict[str, Any]:
        """Return the constant classification fields for one sample."""
        return {
            "emotion_label": self.emotion_label,
            "user_profile": {
                dimension: list(labels) for dimension, labels in self.profile.items()
            },
        }
