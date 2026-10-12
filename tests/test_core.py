from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import validate_prediction, validate_test_row
from src.evaluation.metrics import bleu4, rouge_l


class CoreTests(unittest.TestCase):
    def test_identical_text_metrics(self) -> None:
        self.assertAlmostEqual(bleu4("你好世界", "你好世界"), 1.0)
        self.assertAlmostEqual(rouge_l("你好世界", "你好世界"), 1.0)

    def test_official_sample_id_is_accepted_by_identity(self) -> None:
        sample = {"id": "test1_000001_t5", "conversation_id": "test1_conv_000001", "target_user_turn_id": 1, "history": [{"turn_id": 1, "role": "user", "content": "你好"}]}
        validate_test_row(sample)
        prediction = {"id": sample["id"], "response_text": "你好", "emotion_label": "neutral", "user_profile": {"personality_traits": [], "interests": [], "style": []}, "memory_refs": []}
        validate_prediction(prediction, sample["id"])


if __name__ == "__main__":
    unittest.main()

