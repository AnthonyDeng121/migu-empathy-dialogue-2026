from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.b_validate_submission import validate_files, validate_rows


def source(key: str) -> dict:
    return {"id": key, "conversation_id": "conv", "target_user_turn_id": 1,
            "history": [{"turn_id": 1, "role": "user", "content": "hello"}]}


def prediction(key: str) -> dict:
    return {"id": key, "response_text": "\u4f60\u597d", "emotion_label": "neutral",
            "user_profile": {"personality_traits": [], "interests": [], "style": []}, "memory_refs": []}


class ValidationTests(unittest.TestCase):
    def check_ids(self, expected: list[str], actual: list[str]) -> dict:
        return validate_rows([source(key) for key in expected], [prediction(key) for key in actual])

    def test_valid_submission(self) -> None:
        self.assertTrue(self.check_ids(["A", "B"], ["A", "B"])["passed"])

    def test_same_row_count_missing_and_extra(self) -> None:
        report = self.check_ids(["A", "B", "C"], ["A", "B", "D"])
        self.assertEqual(report["missing_ids"], ["C"])
        self.assertEqual(report["extra_ids"], ["D"])
        self.assertEqual((report["missing_id_count"], report["extra_id_count"]), (1, 1))
        self.assertFalse(report["passed"])

    def test_missing_id(self) -> None:
        report = self.check_ids(["A", "B"], ["A"])
        self.assertEqual(report["missing_ids"], ["B"])
        self.assertEqual(report["extra_id_count"], 0)
        self.assertFalse(report["passed"])

    def test_extra_id(self) -> None:
        report = self.check_ids(["A"], ["A", "B"])
        self.assertEqual(report["extra_ids"], ["B"])
        self.assertEqual(report["missing_id_count"], 0)
        self.assertFalse(report["passed"])

    def test_duplicates_are_independent_of_completeness(self) -> None:
        report = self.check_ids(["A", "B"], ["A", "A"])
        self.assertEqual(report["duplicate_submission_ids"], ["A"])
        self.assertEqual(report["missing_ids"], ["B"])
        self.assertEqual(self.check_ids(["A", "A"], ["A", "A"])["duplicate_input_ids"], ["A"])

    def test_order_mismatch(self) -> None:
        report = self.check_ids(["A", "B"], ["B", "A"])
        self.assertEqual(report["order_mismatch_count"], 2)
        self.assertEqual(report["missing_id_count"], 0)
        self.assertEqual(report["schema_error_count"], 0)
        self.assertFalse(report["passed"])

    def test_order_checked_with_different_row_counts(self) -> None:
        report = self.check_ids(["A", "B", "C"], ["B", "A"])
        self.assertEqual(report["order_mismatch_count"], 2)
        self.assertEqual(report["missing_ids"], ["C"])

    def test_empty_response_counted_even_with_other_schema_error(self) -> None:
        for text in ("", " \t"):
            bad = prediction("A")
            bad.update(response_text=text, emotion_label="unknown")
            report = validate_rows([source("A")], [bad])
            self.assertEqual(report["empty_response_count"], 1)
            self.assertEqual(report["schema_error_count"], 1)
            self.assertFalse(report["passed"])

    def test_schema_legality(self) -> None:
        for field, value in (("emotion_label", "unknown"), ("user_profile", {}),
                             ("memory_refs", ["bad"]), ("memory_refs", [[1]]), ("id", 7)):
            with self.subTest(field=field, value=value):
                bad = prediction("A")
                bad[field] = value
                report = validate_rows([source("A")], [bad])
                self.assertEqual(report["schema_error_count"], 1)
                self.assertFalse(report["passed"])
        bad = prediction("A")
        bad["extra"] = True
        self.assertFalse(validate_rows([source("A")], [bad])["passed"])

    def test_input_schema_and_empty_input(self) -> None:
        bad = source("A")
        bad["history"] = []
        self.assertEqual(validate_rows([bad], [prediction("A")])["schema_error_count"], 1)
        self.assertFalse(validate_rows([], [])["passed"])

    def test_file_hash_stability_and_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path, actual = Path(directory) / "input.jsonl", Path(directory) / "actual.jsonl"
            source_path.write_text(json.dumps(source("A")) + "\n", encoding="utf-8")
            actual.write_bytes((json.dumps(prediction("A")) + "\r\n").encode())
            first = validate_files(source_path, actual)
            self.assertEqual(first["submission_sha256"], hashlib.sha256(actual.read_bytes()).hexdigest())
            self.assertEqual(first["submission_sha256"], validate_files(source_path, actual)["submission_sha256"])
            actual.write_bytes(actual.read_bytes() + b"\n")
            self.assertNotEqual(first["submission_sha256"], validate_files(source_path, actual)["submission_sha256"])

    def test_read_errors_and_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            a.write_text(json.dumps(source("A")) + "\n\n", encoding="utf-8")
            b.write_text(json.dumps(prediction("A")) + "\n", encoding="utf-8")
            self.assertTrue(validate_files(a, b)["passed"])
            for text in ("{", "[]\n"):
                b.write_text(text, encoding="utf-8")
                self.assertFalse(validate_files(a, b)["passed"])
            self.assertIsNone(validate_files(a, Path(directory) / "missing")["submission_sha256"])

    def test_cli_chinese_summary_report_and_failure_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a, b, output = (Path(directory) / name for name in ("a", "b", "report.json"))
            a.write_text(json.dumps(source("A")) + "\n", encoding="utf-8")
            b.write_text(json.dumps(prediction("A")) + "\n", encoding="utf-8")
            command = [sys.executable, "-B", "-X", "utf8", str(PROJECT / "scripts/b_validate_submission.py"),
                       "--input", str(a), "--submission", str(b), "--report", str(output)]
            result = subprocess.run(command, capture_output=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("\u3010B \u63d0\u4ea4\u7ed3\u679c\u9a8c\u6536\u3011", result.stdout)
            self.assertIn("\u6700\u7ec8\u7ed3\u679c\uff1a\u5df2\u901a\u8fc7", result.stdout)
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["passed"])
            b.write_text(json.dumps(prediction("B")) + "\n", encoding="utf-8")
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


PROJECT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    unittest.main()
