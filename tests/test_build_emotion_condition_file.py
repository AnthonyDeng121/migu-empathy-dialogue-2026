from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_emotion_condition_file import (  # noqa: E402
    BATCH_SIZE,
    LABELS,
    MAX_LENGTH,
    SPEAKERS,
    condition_rows,
    format_context_text,
    labels_from_indices,
    read_rows,
)
from src.data.schema import EMOTIONS  # noqa: E402

ROW = {
    "sample_id": 'pp:["val_conv_000001",1,1]',
    "conversation_id": "val_conv_000001",
    "target_user_turn_id": 3,
    "source_split": "val",
    "context_turns": [
        {"turn_id": 1, "role": "user", "content": "最近压力很大"},
        {"turn_id": 2, "role": "assistant", "content": "发生了什么？"},
        {"turn_id": 3, "role": "user", "content": "答辩只剩两周"},
    ],
    "context_text": '{"turn_id": 3, "role": "user"}',
    "emotion_label": "anxiety",
    "user_profile": {"personality_traits": [], "interests": [], "style": []},
}


def _write(rows: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", encoding="utf-8", delete=False
    )
    with handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return Path(handle.name)


class LabelMappingTests(unittest.TestCase):
    def test_labels_are_the_sorted_schema_order(self) -> None:
        self.assertEqual(LABELS, tuple(sorted(EMOTIONS)))
        self.assertEqual(len(LABELS), 16)
        self.assertEqual(LABELS[0], "anger")
        self.assertEqual(LABELS[15], "surprise")

    def test_labels_from_indices_maps_positions(self) -> None:
        self.assertEqual(labels_from_indices([0, 15]), ["anger", "surprise"])
        self.assertEqual(labels_from_indices([]), [])

    def test_labels_from_indices_rejects_out_of_range(self) -> None:
        for bad in (-1, 16, 99):
            with self.assertRaises(ValueError):
                labels_from_indices([bad])


class FormatContextTextTests(unittest.TestCase):
    def test_multi_turn_order_and_fullwidth_colon(self) -> None:
        self.assertEqual(
            format_context_text(ROW["context_turns"]),
            "用户：最近压力很大\n助手：发生了什么？\n用户：答辩只剩两周",
        )

    def test_rejects_unknown_role_or_missing_content(self) -> None:
        with self.assertRaises(ValueError):
            format_context_text([{"turn_id": 1, "role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            format_context_text([{"turn_id": 1, "role": "user"}])


class ReadRowsTests(unittest.TestCase):
    def test_accepts_valid_row_with_text_and_gold(self) -> None:
        rows = read_rows(_write([ROW]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], ROW["sample_id"])
        self.assertEqual(rows[0]["gold"], "anxiety")
        self.assertEqual(
            rows[0]["text"],
            "用户：最近压力很大\n助手：发生了什么？\n用户：答辩只剩两周",
        )

    def test_gold_label_is_optional_but_validated(self) -> None:
        row = {key: value for key, value in ROW.items() if key != "emotion_label"}
        self.assertIsNone(read_rows(_write([row]))[0]["gold"])
        with self.assertRaises(ValueError):
            read_rows(_write([dict(ROW, emotion_label="not-a-label")]))

    def test_rejects_context_not_ending_in_user_turn(self) -> None:
        row = dict(ROW, context_turns=ROW["context_turns"][:2])
        with self.assertRaises(ValueError):
            read_rows(_write([row]))

    def test_rejects_context_not_cut_at_target_user_turn(self) -> None:
        # A context that runs past the target turn would leak future turns.
        row = dict(ROW, target_user_turn_id=1)
        with self.assertRaises(ValueError):
            read_rows(_write([row]))

    def test_rejects_missing_required_fields(self) -> None:
        for field in (
            "sample_id",
            "conversation_id",
            "target_user_turn_id",
            "source_split",
            "context_turns",
        ):
            row = {key: value for key, value in ROW.items() if key != field}
            with self.assertRaises(ValueError):
                read_rows(_write([row]))

    def test_rejects_empty_context_and_malformed_turn(self) -> None:
        with self.assertRaises(ValueError):
            read_rows(_write([dict(ROW, context_turns=[])]))
        with self.assertRaises(ValueError):
            read_rows(_write([dict(ROW, context_turns=[{"turn_id": 1, "role": "user"}])]))

    def test_rejects_duplicate_sample_id(self) -> None:
        with self.assertRaises(ValueError):
            read_rows(_write([ROW, ROW]))

    def test_rejects_empty_file(self) -> None:
        with self.assertRaises(ValueError):
            read_rows(_write([]))


class ConditionRowsTests(unittest.TestCase):
    def test_pairs_ids_with_predictions_in_order(self) -> None:
        rows = read_rows(_write([ROW, dict(ROW, sample_id="b", target_user_turn_id=3)]))
        conditions = condition_rows(rows, ["anxiety", "joy"])
        self.assertEqual(
            conditions,
            [
                {"id": ROW["sample_id"], "emotion_label": "anxiety"},
                {"id": "b", "emotion_label": "joy"},
            ],
        )

    def test_refuses_to_write_a_partial_file(self) -> None:
        rows = read_rows(_write([ROW]))
        with self.assertRaises(ValueError):
            condition_rows(rows, [])
        with self.assertRaises(ValueError):
            condition_rows(rows, ["anxiety", "joy"])

    def test_rejects_prediction_outside_the_schema(self) -> None:
        rows = read_rows(_write([ROW]))
        with self.assertRaises(ValueError):
            condition_rows(rows, ["not-a-label"])


class ConventionTests(unittest.TestCase):
    def test_inference_conventions_match_c_training_script(self) -> None:
        self.assertEqual(SPEAKERS, {"user": "用户", "assistant": "助手"})
        self.assertEqual(MAX_LENGTH, 256)
        self.assertEqual(BATCH_SIZE, 32)


if __name__ == "__main__":
    unittest.main()
