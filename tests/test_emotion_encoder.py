from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_emotion_encoder import (
    _majority_comparison,
    _portable_path,
    _render_report,
    _smoke_checks,
    _training_speed_summary,
)


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

    def test_training_speed_summary_projects_full_epoch_from_measured_steps(self) -> None:
        summary = _training_speed_summary(50, 125.0, 18_657, 32, 1)

        self.assertEqual(summary["optimizer_steps"], 50)
        self.assertEqual(summary["full_steps_per_epoch"], 584)
        self.assertEqual(summary["estimated_full_train_seconds"], 1460.0)

    def test_smoke_checks_require_finite_learning_and_reloadable_checkpoint(self) -> None:
        summary = _smoke_checks(
            optimizer_steps=50,
            first_loss=1.2,
            first_gradient_norm=0.8,
            classifier_weights_changed=True,
            checkpoint_reload_passed=True,
        )

        self.assertTrue(summary["finite_loss"])
        self.assertTrue(summary["first_step_gradient_finite_nonzero"])
        self.assertTrue(summary["classifier_weights_changed"])
        self.assertTrue(summary["checkpoint_reload_passed"])
        with self.assertRaisesRegex(RuntimeError, "classifier weights did not change"):
            _smoke_checks(
                optimizer_steps=50,
                first_loss=1.2,
                first_gradient_norm=0.8,
                classifier_weights_changed=False,
                checkpoint_reload_passed=True,
            )

    def test_smoke_report_reproduces_sample_limits_and_uses_new_output_paths(self) -> None:
        report = {
            "status": "Exploratory sample only",
            "model": {"name": "model", "revision": "abc", "card_url": "https://example.com"},
            "device": "mps",
            "seed": 7,
            "config": {"epochs": 1, "max_length": 32},
            "data": {
                "train_path": "data/train.jsonl",
                "validation_path": "data/val.jsonl",
                "train_used": 1600,
                "train_total": 18657,
                "validation_used": 2,
                "validation_total": 2,
                "train_sha256": "train-hash",
                "validation_sha256": "val-hash",
            },
            "selected_epoch": 1,
            "metrics": {
                "accuracy": 0.5,
                "macro_f1": 0.5,
                "labels": ["joy"],
                "per_class": {"joy": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}},
            },
            "training_speed": {
                "optimizer_steps": 50,
                "seconds_per_step": 2.5,
                "estimated_full_train_seconds": 1460.0,
            },
            "baseline_comparison": {"majority_accuracy": 0.25, "accuracy_delta": 0.25},
            "smoke_checks": {"optimizer_steps": 50},
            "checkpoint": {"file": "weights/smoke/model.safetensors", "bytes": 1, "sha256": "hash"},
            "sampling": {"max_train_samples": 1600, "max_validation_samples": None},
        }

        markdown = _render_report(report)

        self.assertIn("--max-train-samples 1600", markdown)
        self.assertIn("--output-dir weights/emotion_encoder_repro", markdown)
        self.assertIn("'data/train.jsonl'", markdown)


if __name__ == "__main__":
    unittest.main()
