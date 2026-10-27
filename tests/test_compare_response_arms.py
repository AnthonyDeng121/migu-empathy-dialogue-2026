from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import compare_response_arms as cra
from src.common.io import read_jsonl


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


class ArmMetricsTests(unittest.TestCase):
    def test_perfect_copy_scores_one(self) -> None:
        gold = {"a": "你好世界", "b": "今天天气不错"}
        arm = dict(gold)
        m = cra.arm_metrics(gold, arm)
        self.assertAlmostEqual(m["bleu4_mean"], 1.0)
        self.assertAlmostEqual(m["rouge_l_mean"], 1.0)
        self.assertEqual(m["empty_responses"], 0)

    def test_empty_response_counts(self) -> None:
        gold = {"a": "你好"}
        arm = {"a": "   "}
        m = cra.arm_metrics(gold, arm)
        self.assertEqual(m["empty_responses"], 1)

    def test_paired_direction_and_identity(self) -> None:
        gold = {"a": "参考回复", "b": "另一条参考", "c": "第三条参考"}
        main = {"a": "参考回复", "b": "完全不同的内容", "c": "第三条参考"}
        base = {"a": "别的内容", "b": "另一条参考", "c": "第三条参考"}
        p = cra.paired(gold, main, base)
        self.assertEqual(p["wins"], 1)
        self.assertEqual(p["losses"], 1)
        self.assertEqual(p["ties"], 1)
        self.assertEqual(p["identical_text"], 1)
        self.assertEqual(p["length_longer"], 1)
        self.assertEqual(p["length_shorter"], 0)
        self.assertEqual(p["length_same"], 2)


class LoadArmTests(unittest.TestCase):
    def test_duplicate_id_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arm.jsonl"
            write_jsonl(path, [{"id": "x", "response_text": "a"}, {"id": "x", "response_text": "b"}])
            with self.assertRaises(SystemExit):
                cra.load_arm(path, "t")

    def test_missing_id_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arm.jsonl"
            write_jsonl(path, [{"response_text": "a"}])
            with self.assertRaises(SystemExit):
                cra.load_arm(path, "t")

    def test_read_jsonl_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_jsonl(path, [{"sample_id": "s", "response_text": "t"}])
            rows = read_jsonl(path)
            self.assertEqual(rows[0]["sample_id"], "s")


if __name__ == "__main__":
    unittest.main()
