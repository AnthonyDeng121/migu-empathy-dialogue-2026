from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_emotion_encoder import _majority_comparison, _portable_path


class EmotionEncoderTests(unittest.TestCase):
    def _api(self) -> tuple[Any, Any]:
        try:
            from src.classification.emotion_encoder import (
                build_emotion_examples,
                classification_metrics,
            )
        except (ImportError, AttributeError) as error:
            self.fail(f"emotion encoder API is not implemented: {error}")
        return build_emotion_examples, classification_metrics

    def test_examples_stop_at_target_user_turn(self) -> None:
        build_emotion_examples, _ = self._api()
        rows = [
            {
                "conversation_id": "conversation-1",
                "turns": [
                    {"turn_id": 1, "role": "user", "content": "我今天很开心"},
                    {"turn_id": 2, "role": "assistant", "content": "听起来不错"},
                    {"turn_id": 3, "role": "user", "content": "现在有点难过"},
                    {"turn_id": 4, "role": "assistant", "content": "后来发生了什么"},
                ],
                "prediction_points": [
                    {
                        "target_user_turn_id": 3,
                        "target": {"emotion_label": "sadness"},
                    }
                ],
            }
        ]

        examples = build_emotion_examples(rows)

        self.assertEqual(len(examples), 1)
        self.assertEqual(
            examples[0]["text"],
            "用户：我今天很开心\n助手：听起来不错\n用户：现在有点难过",
        )
        self.assertEqual(examples[0]["label"], "sadness")
        self.assertEqual(examples[0]["conversation_id"], "conversation-1")

    def test_metrics_use_fixed_label_order_and_count_absent_class_as_zero(self) -> None:
        _, classification_metrics = self._api()

        report = classification_metrics(
            ["joy", "joy", "sadness"],
            ["joy", "neutral", "sadness"],
            ["joy", "sadness", "neutral"],
        )

        self.assertEqual(report["accuracy"], 0.666667)
        self.assertEqual(report["macro_f1"], 0.555556)
        self.assertEqual(report["confusion_matrix"], [[1, 0, 1], [0, 1, 0], [0, 0, 0]])
        self.assertEqual(report["per_class"]["neutral"]["support"], 0)

    def test_examples_reject_unknown_emotion_label(self) -> None:
        build_emotion_examples, _ = self._api()
        rows = [
            {
                "conversation_id": "conversation-1",
                "turns": [{"turn_id": 1, "role": "user", "content": "你好"}],
                "prediction_points": [
                    {"target_user_turn_id": 1, "target": {"emotion_label": "unknown"}}
                ],
            }
        ]

        with self.assertRaisesRegex(ValueError, "invalid emotion_label"):
            build_emotion_examples(rows)

    def test_examples_reject_missing_target_user_turn(self) -> None:
        build_emotion_examples, _ = self._api()
        rows = [
            {
                "conversation_id": "conversation-1",
                "turns": [
                    {"turn_id": 1, "role": "user", "content": "你好"},
                    {"turn_id": 3, "role": "user", "content": "再见"},
                ],
                "prediction_points": [
                    {"target_user_turn_id": 2, "target": {"emotion_label": "neutral"}}
                ],
            }
        ]

        with self.assertRaisesRegex(ValueError, "end at target_user_turn_id"):
            build_emotion_examples(rows)

    def test_report_uses_relative_checkpoint_path_and_compares_same_validation_set(self) -> None:
        self.assertEqual(_portable_path(Path("weights/emotion_baseline")), "weights/emotion_baseline")
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            baseline_path.write_text(
                json.dumps({"validation_samples": 2, "metrics": {"emotion_accuracy": 0.25}}),
                encoding="utf-8",
            )
            comparison = _majority_comparison(baseline_path, 2, 0.5)
            self.assertEqual(comparison, {"majority_accuracy": 0.25, "accuracy_delta": 0.25})
            with self.assertRaisesRegex(ValueError, "sample counts differ"):
                _majority_comparison(baseline_path, 3, 0.5)


if __name__ == "__main__":
    unittest.main()
