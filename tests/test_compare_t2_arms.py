"""Unit tests for the T2 arm comparison.

The comparison must fail loudly when the two arms do not cover exactly the
same ids as the frozen selection (a silent intersection would hide a broken
arm), must report byte-identity correctly, and must aggregate the local
diagnostic metrics over the ids it actually compared. Synthetic rows only —
the real 160-row arms are exercised by the generation run.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_t2_arms import compare_arms  # noqa: E402

OFF = Path("off.jsonl")
EMOTION = Path("emotion.jsonl")
SELECTION = Path("selection.jsonl")


class CompareArmsTests(unittest.TestCase):
    def test_identical_arms_and_aggregates(self) -> None:
        report = compare_arms(
            [
                {"id": "b", "response_text": "同样的回复"},
                {"id": "a", "response_text": "第一条回复内容"},
            ],
            [
                {"id": "a", "response_text": "第一条回复内容"},
                {"id": "b", "response_text": "另一条金标下的不同回复"},
            ],
            [
                {"sample_id": "a", "response_text": "第一条回复内容"},
                {"sample_id": "b", "response_text": "另一条金标"},
            ],
        )
        self.assertEqual(report["ids"]["compared"], 2)
        self.assertEqual(report["identical_responses"]["count"], 1)
        self.assertAlmostEqual(report["identical_responses"]["rate"], 0.5)
        # arm "a" matches gold exactly: char-level BLEU-4/ROUGE-L are 1.0
        self.assertAlmostEqual(report["per_id"][0]["off"]["bleu4"], 1.0)
        self.assertAlmostEqual(report["per_id"][0]["off"]["rouge_l"], 1.0)
        self.assertEqual(
            [row["id"] for row in report["per_id"]], ["a", "b"]
        )

    def test_length_delta_direction(self) -> None:
        report = compare_arms(
            [{"id": "a", "response_text": "短回复"}],
            [{"id": "a", "response_text": "这是一条明显更长的回复内容"}],
            [{"sample_id": "a", "response_text": "金标回复"}],
        )
        delta = report["length_delta"]
        self.assertGreater(delta["mean"], 0)
        self.assertEqual(delta["longer_under_emotion"], 1)
        self.assertEqual(delta["shorter_under_emotion"], 0)
        self.assertEqual(delta["same_length"], 0)

    def test_divergent_ids_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "only_emotion"):
            compare_arms(
                [{"id": "a", "response_text": "x"}],
                [
                    {"id": "a", "response_text": "x"},
                    {"id": "b", "response_text": "y"},
                ],
                [
                    {"sample_id": "a", "response_text": "x"},
                    {"sample_id": "b", "response_text": "z"},
                ],
            )

    def test_missing_gold_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing_gold"):
            compare_arms(
                [{"id": "a", "response_text": "x"}],
                [{"id": "a", "response_text": "x"}],
                [{"sample_id": "b", "response_text": "z"}],
            )

    def test_both_arms_missing_selection_id_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing_predictions=1"):
            compare_arms(
                [{"id": "a", "response_text": "x"}],
                [{"id": "a", "response_text": "x"}],
                [
                    {"sample_id": "a", "response_text": "x"},
                    {"sample_id": "b", "response_text": "y"},
                ],
            )

    def test_duplicate_id_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_arms(
                [
                    {"id": "a", "response_text": "x"},
                    {"id": "a", "response_text": "y"},
                ],
                [{"id": "a", "response_text": "x"}],
                [{"sample_id": "a", "response_text": "x"}],
            )


if __name__ == "__main__":
    unittest.main()
