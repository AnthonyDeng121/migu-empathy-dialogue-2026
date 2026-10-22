from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_response_inference import (  # noqa: E402
    condition_coverage,
    parse_args,
)

B1_ROWS = [
    {"sample_id": 'pp:["val_conv_000001",1,1]', "context_turns": []},
    {"sample_id": 'pp:["val_conv_000001",3,2]', "context_turns": []},
]

OFFICIAL_ROWS = [
    {"id": "eval_000001_t1", "history": []},
    {"id": "eval_000002_t3", "history": []},
]


class ConditionCoverageTests(unittest.TestCase):
    """Coverage must resolve ids the same way the rest of the script does.

    Regression: the coverage expression used row["id"] directly, so a B1
    response split (sample_id) with predicted conditions supplied crashed
    with KeyError('id') while writing the performance report — after the
    whole inference had already run. A review found it because the
    predicted-conditions path had never been executed.
    """

    def test_b1_rows_are_counted_not_crashed(self) -> None:
        conditions = {row["sample_id"]: {} for row in B1_ROWS}
        self.assertEqual(condition_coverage(B1_ROWS, conditions), "2/2")

    def test_official_rows_are_counted(self) -> None:
        conditions = {row["id"]: {} for row in OFFICIAL_ROWS}
        self.assertEqual(condition_coverage(OFFICIAL_ROWS, conditions), "2/2")

    def test_partial_coverage_is_reported(self) -> None:
        conditions = {B1_ROWS[0]["sample_id"]: {}}
        self.assertEqual(condition_coverage(B1_ROWS, conditions), "1/2")

    def test_no_conditions_reports_none(self) -> None:
        self.assertIsNone(condition_coverage(B1_ROWS, {}))


class ConditionFieldsFlagTests(unittest.TestCase):
    """Cycle-2 T1: the off vs emotion-only ablation needs the same condition
    file to drive both arms. ``both`` (the default) must stay byte-identical
    to every recorded run; ``emotion`` drops the profile from the prompt."""

    def _parse(self, argv: list[str]) -> argparse.Namespace:
        saved = sys.argv
        try:
            sys.argv = ["run_response_inference.py", *argv]
            return parse_args()
        finally:
            saved = sys.argv

    def test_defaults_to_both_so_recorded_runs_are_unchanged(self) -> None:
        args = self._parse(["--model", "m", "--output", "o", "--performance-report", "p"])
        self.assertEqual(args.condition_fields, "both")

    def test_emotion_is_parsed_when_given(self) -> None:
        args = self._parse(
            [
                "--model", "m", "--output", "o", "--performance-report", "p",
                "--condition-jsonl", "c.jsonl", "--condition-fields", "emotion",
            ]
        )
        self.assertEqual(args.condition_fields, "emotion")

    def test_unknown_field_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self._parse(
                [
                    "--model", "m", "--output", "o", "--performance-report", "p",
                    "--condition-fields", "profile",
                ]
            )


class MinNewTokensFlagTests(unittest.TestCase):
    """The D3 probe for the 32 near-empty replies needs the floor to actually
    reach generate(), and to be absent by default so every previously recorded
    run stays byte-reproducible."""

    def _parse(self, argv: list[str]) -> argparse.Namespace:
        saved = sys.argv
        try:
            sys.argv = ["run_response_inference.py", *argv]
            return parse_args()
        finally:
            sys.argv = saved

    def test_defaults_to_zero_so_recorded_runs_are_unchanged(self) -> None:
        args = self._parse(["--model", "m", "--output", "o", "--performance-report", "p"])
        self.assertEqual(args.min_new_tokens, 0)

    def test_floor_is_parsed_when_given(self) -> None:
        args = self._parse(
            ["--model", "m", "--output", "o", "--performance-report", "p", "--min-new-tokens", "8"]
        )
        self.assertEqual(args.min_new_tokens, 8)


if __name__ == "__main__":
    unittest.main()
