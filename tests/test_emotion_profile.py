from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.emotion_profile import (  # noqa: E402
    EMOTION_LABELS,
    PROFILE_LABELS_BY_DIMENSION,
    build_b1_classification_examples,
    build_inference_context,
    decode_emotion_profile_scores,
    profile_metrics,
    profile_targets,
    validate_disjoint_splits,
)


def _b1_row(*, split: str = "train", content: str = "今天有点累") -> dict:
    import json

    turns = [
        {"turn_id": 1, "role": "user", "content": content},
        {"turn_id": 2, "role": "assistant", "content": "听起来不容易"},
        {"turn_id": 3, "role": "user", "content": "先听我说就好"},
    ]
    return {
        "sample_id": f"sample-{split}-{content}",
        "conversation_id": f"conversation-{split}-{content}",
        "target_user_turn_id": 3,
        "source_split": split,
        "context_turns": turns,
        "context_text": "\n".join(
            json.dumps(turn, ensure_ascii=False, separators=(",", ":"))
            for turn in turns
        ),
        "emotion_label": "anxiety",
        "user_profile": {
            "personality_traits": ["sensitive"],
            "interests": [],
            "style": ["brief", "direct"],
        },
    }


class EmotionProfileDataTests(unittest.TestCase):
    def test_b1_adapter_uses_b1_fields_and_formats_only_context(self) -> None:
        example = build_b1_classification_examples([_b1_row()], expected_split="train")[0]

        self.assertEqual(example["sample_id"], "sample-train-今天有点累")
        self.assertIn("用户：先听我说就好", example["text"])
        self.assertNotIn("先听我说就好\n助手", example["text"])
        self.assertEqual(example["emotion_label"], "anxiety")
        self.assertEqual(example["user_profile"]["style"], ["brief", "direct"])

    def test_b1_adapter_rejects_wrong_split_or_context_text_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_split"):
            build_b1_classification_examples([_b1_row(split="val")], expected_split="train")

        row = _b1_row()
        row["context_text"] += "\n{}"
        with self.assertRaisesRegex(ValueError, "context_text"):
            build_b1_classification_examples([row], expected_split="train")

    def test_b1_adapter_rejects_duplicate_sample_ids_and_invalid_profile_labels(self) -> None:
        row = _b1_row()
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            build_b1_classification_examples([row, row], expected_split="train")

        row = _b1_row()
        row["user_profile"]["style"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "user_profile.style"):
            build_b1_classification_examples([row], expected_split="train")

    def test_split_integrity_rejects_shared_conversation_and_sample(self) -> None:
        train = build_b1_classification_examples([_b1_row()], expected_split="train")
        val_row = _b1_row(split="val")
        val_row["conversation_id"] = train[0]["conversation_id"]
        val = build_b1_classification_examples([val_row], expected_split="val")

        with self.assertRaisesRegex(ValueError, "conversation overlap"):
            validate_disjoint_splits(train, val)

    def test_inference_context_stops_at_target_user_turn(self) -> None:
        row = {
            "id": "sample-1",
            "conversation_id": "conversation-1",
            "target_user_turn_id": 3,
            "history": [
                {"turn_id": 1, "role": "user", "content": "之前"},
                {"turn_id": 2, "role": "assistant", "content": "回应"},
                {"turn_id": 3, "role": "user", "content": "目标"},
            ],
        }

        text = build_inference_context(row)

        self.assertTrue(text.endswith("用户：目标"))
        self.assertNotIn("回应之后", text)


class EmotionProfileHeadTests(unittest.TestCase):
    def test_profile_targets_are_three_deterministic_multihot_heads(self) -> None:
        encoded = profile_targets(
            {"personality_traits": ["sensitive"], "interests": [], "style": ["direct"]}
        )

        self.assertEqual(set(encoded), set(PROFILE_LABELS_BY_DIMENSION))
        self.assertEqual(len(encoded["personality_traits"]), len(PROFILE_LABELS_BY_DIMENSION["personality_traits"]))
        self.assertEqual(sum(encoded["interests"]), 0)
        self.assertEqual(sum(encoded["personality_traits"]), 1)
        self.assertEqual(sum(encoded["style"]), 1)

    def test_profile_metrics_report_per_dimension_and_empty_set_baseline(self) -> None:
        expected = [
            {"personality_traits": ["introverted"], "interests": [], "style": ["brief", "direct"]}
        ]
        predicted = [
            {"personality_traits": [], "interests": [], "style": ["brief"]}
        ]

        report = profile_metrics(expected, predicted)

        self.assertEqual(report["per_dimension"]["personality_traits"]["exact_match_accuracy"], 0.0)
        self.assertEqual(report["per_dimension"]["interests"]["exact_match_accuracy"], 1.0)
        self.assertEqual(report["empty_set_baseline"]["profile_macro_exact_match_accuracy"], 0.333333)
        self.assertEqual(report["profile_macro_exact_match_accuracy"], 0.333333)
        self.assertEqual(report["per_dimension"]["style"]["micro_f1"], 0.666667)

    def test_profile_metrics_reject_unknown_labels(self) -> None:
        expected = [{"personality_traits": ["invalid"], "interests": [], "style": []}]
        predicted = [{"personality_traits": [], "interests": [], "style": []}]
        with self.assertRaisesRegex(ValueError, "invalid user_profile"):
            profile_metrics(expected, predicted)

    def test_decoder_selects_emotion_and_multilabel_profile_values(self) -> None:
        emotion_scores = [0.0] * len(EMOTION_LABELS)
        emotion_scores[EMOTION_LABELS.index("anxiety")] = 1.0
        profile_scores = {
            dimension: [0.0] * len(labels)
            for dimension, labels in PROFILE_LABELS_BY_DIMENSION.items()
        }
        style_labels = PROFILE_LABELS_BY_DIMENSION["style"]
        profile_scores["style"][style_labels.index("direct")] = 0.9
        profile_scores["style"][style_labels.index("brief")] = 0.5

        result = decode_emotion_profile_scores(emotion_scores, profile_scores)

        self.assertEqual(result["emotion_label"], "anxiety")
        self.assertEqual(result["user_profile"]["style"], ["brief", "direct"])
        self.assertEqual(result["user_profile"]["interests"], [])


if __name__ == "__main__":
    unittest.main()
