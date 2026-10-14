from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_response_inference import condition_coverage  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
