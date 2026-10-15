from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_condition_file import condition_rows, contexts_from_rows  # noqa: E402

ROW = {
    "sample_id": 'pp:["val_conv_000001",1,1]',
    "context_turns": [
        {"turn_id": 1, "role": "user", "content": "最近压力很大"},
        {"turn_id": 2, "role": "assistant", "content": "发生了什么？"},
        {"turn_id": 3, "role": "user", "content": "答辩只剩两周"},
    ],
    "target_user_turn_id": "3",
}


class ContextsFromRowsTests(unittest.TestCase):
    def test_keeps_only_role_and_content_in_order(self) -> None:
        contexts = contexts_from_rows([ROW])
        self.assertEqual(len(contexts), 1)
        self.assertEqual(
            contexts[0],
            [
                {"role": "user", "content": "最近压力很大"},
                {"role": "assistant", "content": "发生了什么？"},
                {"role": "user", "content": "答辩只剩两周"},
            ],
        )

    def test_rejects_context_not_ending_in_user_turn(self) -> None:
        # C's predict_contexts raises on this; catching it here keeps the failure
        # at the row instead of halfway through a 2,354-row batch.
        row = dict(ROW, context_turns=ROW["context_turns"][:2])
        with self.assertRaises(ValueError):
            contexts_from_rows([row])

    def test_rejects_missing_or_empty_context(self) -> None:
        for bad in ([], None, "not-a-list"):
            with self.assertRaises(ValueError):
                contexts_from_rows([dict(ROW, context_turns=bad)])


class ConditionRowsTests(unittest.TestCase):
    def test_pairs_ids_with_predictions_in_order(self) -> None:
        rows = [ROW, dict(ROW, sample_id="b")]
        predictions = [
            {"emotion_label": "anxiety", "user_profile": {"style": ["casual"]}},
            {"emotion_label": "joy", "user_profile": {"style": []}},
        ]
        conditions = condition_rows(rows, predictions)
        self.assertEqual([c["id"] for c in conditions], [ROW["sample_id"], "b"])
        self.assertEqual(conditions[0]["emotion_label"], "anxiety")
        self.assertEqual(conditions[0]["user_profile"], {"style": ["casual"]})

    def test_refuses_to_write_a_partial_file(self) -> None:
        # A silent short file would make the missing-condition check in
        # run_response_inference.py fire only after a full inference run.
        with self.assertRaises(ValueError):
            condition_rows([ROW], [])

    def test_rejects_prediction_without_a_condition(self) -> None:
        with self.assertRaises(ValueError):
            condition_rows([ROW], [{"emotion_label": "joy"}])
        with self.assertRaises(ValueError):
            condition_rows([ROW], [{"emotion_label": "joy", "user_profile": []}])


if __name__ == "__main__":
    unittest.main()
