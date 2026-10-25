"""Unit tests for the T2 conditions-off arm extractor.

The off arm must reuse the historical D Day4 v2 full-val predictions rather
than recompute them, so the tests pin the extraction contract: a strict join
in one direction (every subset id must have a prediction), verbatim rows
sorted by id, loud failure on a missing id or a duplicate on either side,
and a manifest that records provenance (source digests and counts) without
touching the data. Synthetic rows only — the real 160-row subset and
2,354-row predictions are exercised by the generation run.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_off_arm_subset import (  # noqa: E402
    extract_off_arm,
    main,
)

SELECTION = Path("selection.jsonl")
PREDICTIONS = Path("predictions.jsonl")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


class ExtractOffArmTests(unittest.TestCase):
    def test_extracts_verbatim_rows_sorted_by_id(self) -> None:
        extracted = extract_off_arm(
            [
                {"sample_id": 'pp:["val_conv_000002",5,1]'},
                {"sample_id": 'pp:["val_conv_000001",1,1]'},
            ],
            [
                {"id": 'pp:["val_conv_000002",5,1]', "response_text": "second"},
                {"id": 'pp:["val_conv_000001",1,1]', "response_text": "first"},
                {"id": 'pp:["val_conv_000003",5,1]', "response_text": "outside"},
            ],
        )
        self.assertEqual(
            extracted,
            [
                {"id": 'pp:["val_conv_000001",1,1]', "response_text": "first"},
                {"id": 'pp:["val_conv_000002",5,1]', "response_text": "second"},
            ],
        )

    def test_subset_ids_outside_predictions_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "no prediction"):
            extract_off_arm(
                [{"sample_id": 'pp:["val_conv_000009",5,1]'}],
                [{"id": 'pp:["val_conv_000001",1,1]', "response_text": "x"}],
            )

    def test_duplicate_subset_id_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            extract_off_arm(
                [
                    {"sample_id": 'pp:["val_conv_000001",1,1]'},
                    {"sample_id": 'pp:["val_conv_000001",1,1]'},
                ],
                [{"id": 'pp:["val_conv_000001",1,1]', "response_text": "x"}],
            )

    def test_duplicate_prediction_id_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            extract_off_arm(
                [{"sample_id": 'pp:["val_conv_000001",1,1]'}],
                [
                    {"id": 'pp:["val_conv_000001",1,1]', "response_text": "x"},
                    {"id": 'pp:["val_conv_000001",1,1]', "response_text": "y"},
                ],
            )

    def test_prediction_without_response_text_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "response_text"):
            extract_off_arm(
                [{"sample_id": 'pp:["val_conv_000001",1,1]'}],
                [{"id": 'pp:["val_conv_000001",1,1]'}],
            )


class MainTests(unittest.TestCase):
    def test_writes_rows_and_manifest_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = root / SELECTION
            predictions = root / PREDICTIONS
            output = root / "off.jsonl"
            manifest = root / "off_manifest.json"
            write_jsonl(
                selection,
                [
                    {"sample_id": 'pp:["val_conv_000001",1,1]'},
                    {"sample_id": 'pp:["val_conv_000002",5,1]'},
                ],
            )
            write_jsonl(
                predictions,
                [
                    {"id": 'pp:["val_conv_000002",5,1]', "response_text": "b"},
                    {"id": 'pp:["val_conv_000001",1,1]', "response_text": "a"},
                    {"id": 'pp:["val_conv_000003",5,1]', "response_text": "c"},
                ],
            )
            self.assertEqual(
                main(
                    [
                        "--selection",
                        str(selection),
                        "--predictions",
                        str(predictions),
                        "--output",
                        str(output),
                        "--manifest",
                        str(manifest),
                    ]
                ),
                0,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in rows], sorted(row["id"] for row in rows))
            self.assertEqual(len(rows), 2)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["arm"], "off")
            self.assertEqual(payload["result"]["record_count"], 2)
            self.assertEqual(payload["result"]["predictions_outside_subset"], 1)
            self.assertEqual(payload["sources"]["predictions"]["record_count"], 3)
            self.assertIn("sha256", payload["sources"]["subset"])
            self.assertIn("extract_off_arm_subset.py", payload["command"])


if __name__ == "__main__":
    unittest.main()
