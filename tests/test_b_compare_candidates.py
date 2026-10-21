from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.b_compare_candidates import METRIC_STATUS, compare_candidates, compare_files


def row(key: str) -> dict:
    return {"id": key, "response_text": "\u4f60\u597d", "emotion_label": "neutral",
            "user_profile": {"personality_traits": [], "interests": [], "style": []}, "memory_refs": []}


class ComparisonTests(unittest.TestCase):
    def test_identical_files(self) -> None:
        rows = [row("A"), row("B")]
        report = compare_candidates(rows, copy.deepcopy(rows))
        self.assertEqual(report["common_id_count"], 2)
        self.assertTrue(report["order_identical"])
        self.assertEqual(report["different_sample_count"], 0)
        self.assertEqual(report["difference_examples"], [])
        for counts in report["fields"].values():
            self.assertEqual(counts, {"equal_count": 2, "different_count": 0, "unavailable_count": 0})

    def test_reordered_file_content_is_equal(self) -> None:
        report = compare_candidates([row("A"), row("B")], [row("B"), row("A")])
        self.assertFalse(report["order_identical"])
        self.assertFalse(report["common_order_identical"])
        self.assertEqual(report["fields"]["response_text"]["equal_count"], 2)
        self.assertEqual(report["different_sample_count"], 0)

    def changed_field(self, field: str, value: object) -> None:
        candidate = row("A")
        candidate[field] = value
        report = compare_candidates([row("A")], [candidate])
        self.assertEqual(report["fields"][field]["different_count"], 1)
        example = report["difference_examples"][0]
        self.assertEqual(example["id"], "A")
        self.assertEqual(example["different_fields"], [field])
        self.assertEqual(example["values"][field]["baseline"], row("A")[field])
        self.assertEqual(example["values"][field]["candidate"], value)

    def test_response_change(self) -> None:
        self.changed_field("response_text", "changed")

    def test_emotion_change(self) -> None:
        self.changed_field("emotion_label", "joy")

    def test_profile_change(self) -> None:
        self.changed_field("user_profile", {"personality_traits": [], "interests": [], "style": ["brief"]})

    def test_memory_refs_change(self) -> None:
        self.changed_field("memory_refs", ["mem_000001"])

    def test_baseline_missing_id(self) -> None:
        report = compare_candidates([row("A")], [row("A"), row("B")])
        self.assertEqual(report["candidate_only_ids"], ["B"])
        self.assertEqual(report["baseline_only_ids"], [])
        self.assertEqual(report["common_id_count"], 1)

    def test_candidate_missing_id(self) -> None:
        report = compare_candidates([row("A"), row("B")], [row("A")])
        self.assertEqual(report["baseline_only_ids"], ["B"])
        self.assertEqual(report["candidate_only_ids"], [])
        self.assertFalse(report["order_identical"])
        self.assertTrue(report["common_order_identical"])

    def test_no_overlap(self) -> None:
        report = compare_candidates([row("A")], [row("B")])
        self.assertEqual(report["common_id_count"], 0)
        self.assertEqual(report["fields"]["response_text"]["equal_count"], 0)

    def test_maximum_ten_examples_in_baseline_order(self) -> None:
        rows = [row(str(i)) for i in range(15)]
        candidate = [{**item, "response_text": "changed"} for item in reversed(rows)]
        report = compare_candidates(rows, candidate)
        self.assertEqual(report["different_sample_count"], 15)
        self.assertEqual(report["fields"]["response_text"]["different_count"], 15)
        self.assertEqual([e["id"] for e in report["difference_examples"]], [str(i) for i in range(10)])

    def test_response_only_files_do_not_invent_field_agreement_or_metrics(self) -> None:
        rows = [{"id": "A", "response_text": "text"}]
        report = compare_candidates(rows, rows)
        self.assertEqual(report["fields"]["emotion_label"]["unavailable_count"], 1)
        self.assertEqual(report["fields"]["emotion_label"]["equal_count"], 0)
        self.assertEqual(report["metric_status"], METRIC_STATUS)
        self.assertNotIn("metrics", report)

    def test_missing_field_distinguished_from_null(self) -> None:
        left, right = {"id": "A", "response_text": "text"}, {"id": "A", "response_text": "text", "memory_refs": None}
        example = compare_candidates([left], [right])["difference_examples"][0]
        self.assertFalse(example["values"]["memory_refs"]["baseline_present"])
        self.assertTrue(example["values"]["memory_refs"]["candidate_present"])

    def test_duplicate_or_invalid_ids_fail_instead_of_overwriting(self) -> None:
        for invalid in ([row("A"), row("A")], [row("")], [{"response_text": "x"}], []):
            for baseline, candidate in ((invalid, [row("A")]), ([row("A")], invalid)):
                with self.assertRaises(ValueError):
                    compare_candidates(baseline, candidate)

    def test_sha256_exact_bytes_and_stability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            a.write_bytes((json.dumps(row("A")) + "\n").encode())
            b.write_bytes((json.dumps(row("A")) + "\r\n").encode())
            report = compare_files(a, b)
            self.assertEqual(report["baseline_sha256"], hashlib.sha256(a.read_bytes()).hexdigest())
            self.assertEqual(report["candidate_sha256"], hashlib.sha256(b.read_bytes()).hexdigest())
            self.assertNotEqual(report["baseline_sha256"], report["candidate_sha256"])
            self.assertEqual(report["candidate_sha256"], compare_files(a, b)["candidate_sha256"])
            self.assertEqual(report["fields"]["response_text"]["equal_count"], 1)

    def test_cli_two_paths_summary_and_optional_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a, b, output = (Path(directory) / name for name in ("a", "b", "report.json"))
            for path in (a, b):
                path.write_text(json.dumps(row("A")) + "\n", encoding="utf-8")
            command = [sys.executable, "-B", "-X", "utf8", str(PROJECT / "scripts/b_compare_candidates.py"),
                       "--baseline", str(a), "--candidate", str(b)]
            result = subprocess.run(command, capture_output=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            for text in ("baseline", "candidate", "common IDs", "response_text", "emotion_label", "user_profile", "memory_refs", "SHA-256", METRIC_STATUS):
                self.assertIn(text, result.stdout)
            result = subprocess.run(command + ["--report", str(output)], capture_output=True)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["common_id_count"], 1)
            b.write_text("{", encoding="utf-8")
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
