from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_long_response_weights import sample_weight  # noqa: E402


class SampleWeightRuleTests(unittest.TestCase):
    """The T4 rule: gold response longer than 200 chars weighs 2.0."""

    def test_threshold_is_strictly_above_200_chars(self) -> None:
        self.assertEqual(sample_weight("回" * 200), 1.0)
        self.assertEqual(sample_weight("回" * 201), 2.0)

    def test_surrounding_whitespace_is_ignored(self) -> None:
        self.assertEqual(sample_weight("  " + "回" * 200 + "\n"), 1.0)
        self.assertEqual(sample_weight("\t" + "回" * 201 + " "), 2.0)


class BuildLongResponseWeightsTests(unittest.TestCase):
    """File-level contract: same rows, same order, weight field added."""

    def _run_main(self, source: Path, output: Path) -> None:
        import scripts.build_long_response_weights as module

        argv = sys.argv
        try:
            sys.argv = [
                "build_long_response_weights.py",
                "--input",
                str(source),
                "--output",
                str(output),
            ]
            module.main()
        finally:
            sys.argv = argv

    def test_writes_weights_in_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "response_train.jsonl"
            rows = [
                {"sample_id": "s1", "response_text": "短回复"},
                {"sample_id": "s2", "response_text": "长" * 250},
                {"sample_id": "s3", "response_text": "中" * 150},
            ]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "weighted.jsonl"
            self._run_main(source, output)
            written = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [row["sample_id"] for row in written], ["s1", "s2", "s3"]
            )
            self.assertEqual(
                [row["sample_weight"] for row in written], [1.0, 2.0, 1.0]
            )
            for original, row in zip(rows, written):
                self.assertEqual(row["response_text"], original["response_text"])
                self.assertEqual(
                    {key: value for key, value in row.items() if key != "sample_weight"},
                    original,
                )

    def test_threshold_boundary_through_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "response_train.jsonl"
            rows = [
                {"sample_id": "exact", "response_text": "回" * 200},
                {"sample_id": "over", "response_text": "回" * 201},
            ]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "weighted.jsonl"
            self._run_main(source, output)
            written = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [row["sample_weight"] for row in written], [1.0, 2.0]
            )


if __name__ == "__main__":
    unittest.main()
