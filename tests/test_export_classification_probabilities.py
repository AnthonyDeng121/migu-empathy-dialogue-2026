from __future__ import annotations

import unittest

from scripts.export_classification_probabilities import (
    emotion_prediction_records,
    profile_prediction_records,
)
from src.data.schema import EMOTIONS, PROFILE_LABELS


class ExportClassificationProbabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [{
            "sample_id": "sample-1",
            "conversation_id": "conversation-1",
            "context_turns": [{"role": "user", "content": "private dialogue"}],
            "emotion_label": "joy",
            "user_profile": {dimension: [] for dimension in PROFILE_LABELS},
        }]

    def test_emotion_record_contains_probabilities_without_dialogue_or_gold(self) -> None:
        labels = sorted(EMOTIONS)
        probabilities = [0.0] * len(labels)
        probabilities[labels.index("anxiety")] = 0.2
        probabilities[labels.index("joy")] = 0.8

        records = emotion_prediction_records(self.rows, [probabilities], labels)

        self.assertEqual(records[0]["id"], "sample-1")
        self.assertEqual(records[0]["emotion_label"], "joy")
        self.assertAlmostEqual(records[0]["emotion_probabilities"]["anxiety"], 0.2)
        self.assertAlmostEqual(records[0]["emotion_probabilities"]["joy"], 0.8)
        self.assertEqual(set(records[0]), {
            "id", "conversation_id", "emotion_label", "emotion_probabilities"
        })

    def test_emotion_record_rejects_probability_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "label count"):
            emotion_prediction_records(self.rows, [[0.2]], sorted(EMOTIONS))

    def test_emotion_record_rejects_duplicate_ids(self) -> None:
        labels = sorted(EMOTIONS)
        probabilities = [0.0] * len(labels)
        probabilities[labels.index("joy")] = 1.0

        with self.assertRaisesRegex(ValueError, "duplicate sample ID"):
            emotion_prediction_records(
                [self.rows[0], self.rows[0]], [probabilities, probabilities], labels
            )

    def test_emotion_record_rejects_invalid_or_unnormalized_probabilities(self) -> None:
        labels = sorted(EMOTIONS)
        with self.assertRaisesRegex(ValueError, "do not sum to one"):
            emotion_prediction_records(self.rows, [[0.0] * len(labels)], labels)

        invalid = [0.0] * len(labels)
        invalid[0] = 1.1
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            emotion_prediction_records(self.rows, [invalid], labels)

    def test_profile_record_uses_frozen_thresholds_and_omits_dialogue(self) -> None:
        labels = {dimension: list(values) for dimension, values in PROFILE_LABELS.items()}
        thresholds = {dimension: 0.5 for dimension in PROFILE_LABELS}
        probabilities = {
            dimension: [[0.6] + [0.4] * (len(labels[dimension]) - 1)]
            for dimension in PROFILE_LABELS
        }

        records = profile_prediction_records(
            self.rows, probabilities, labels, thresholds
        )

        self.assertEqual(records[0]["user_profile"], {
            dimension: values[:1] for dimension, values in labels.items()
        })
        self.assertEqual(records[0]["profile_probabilities"], {
            dimension: dict(zip(labels[dimension], probabilities[dimension][0]))
            for dimension in PROFILE_LABELS
        })
        self.assertNotIn("context_turns", records[0])
        self.assertNotIn("emotion_label", records[0])

    def test_profile_record_rejects_threshold_dimension_mismatch(self) -> None:
        labels = {dimension: list(values)[:1] for dimension, values in PROFILE_LABELS.items()}
        probabilities = {dimension: [[0.5]] for dimension in PROFILE_LABELS}

        with self.assertRaisesRegex(ValueError, "threshold dimensions"):
            profile_prediction_records(self.rows, probabilities, labels, {})

    def test_profile_threshold_includes_probability_equal_to_threshold(self) -> None:
        labels = {dimension: list(values) for dimension, values in PROFILE_LABELS.items()}
        thresholds = {dimension: 0.5 for dimension in PROFILE_LABELS}
        probabilities = {
            dimension: [[0.5] + [0.0] * (len(labels[dimension]) - 1)]
            for dimension in PROFILE_LABELS
        }

        records = profile_prediction_records(
            self.rows, probabilities, labels, thresholds
        )

        self.assertEqual(records[0]["user_profile"], {
            dimension: values[:1] for dimension, values in labels.items()
        })

    def test_profile_record_rejects_probability_count_mismatch(self) -> None:
        labels = {dimension: list(values) for dimension, values in PROFILE_LABELS.items()}
        thresholds = {dimension: 0.5 for dimension in PROFILE_LABELS}
        probabilities = {
            dimension: [[0.5] * len(labels[dimension])]
            for dimension in PROFILE_LABELS
        }
        probabilities["style"] = []

        with self.assertRaisesRegex(ValueError, "probability count differs"):
            profile_prediction_records(
                self.rows, probabilities, labels, thresholds
            )


if __name__ == "__main__":
    unittest.main()
