from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


class AnalyzeResponseErrorsEndToEndTest(unittest.TestCase):
    def test_join_and_head_to_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = [
                {"conversation_id": "val_conv_000001", "target_user_turn_id": 1, "sample_id": 'pp:["val_conv_000001",1,1]', "response_text": "我在听，你慢慢说，不着急。"},
                {"conversation_id": "val_conv_000002", "target_user_turn_id": 3, "sample_id": 'pp:["val_conv_000002",3,2]', "response_text": "听起来这件事让你压力很大，要不要展开说说最担心的部分？"},
            ]
            arm_a = [
                {"id": 'pp:["val_conv_000001",1,1]', "response_text": "我在听，你慢慢说，不着急。"},
                {"id": 'pp:["val_conv_000002",3,2]', "response_text": "完全不同的另一句话，没有任何重合。"},
            ]
            arm_b = [
                {"id": "eval_000001_t1", "response_text": "我在听，你慢慢说，不着急。"},
                {"id": "eval_000002_t3", "response_text": "听起来这件事让你压力很大，要不要展开说说最担心的部分？"},
            ]
            bridge = [
                {"id": "eval_000001_t1", "conversation_id": "val_conv_000001", "target_user_turn_id": 1},
                {"id": "eval_000002_t3", "conversation_id": "val_conv_000002", "target_user_turn_id": 3},
            ]
            write_jsonl(root / "gold.jsonl", gold)
            write_jsonl(root / "arm_a.jsonl", arm_a)
            write_jsonl(root / "arm_b.jsonl", arm_b)
            write_jsonl(root / "bridge.jsonl", bridge)
            output = root / "out.json"
            result = subprocess.run(
                [
                    sys.executable, str(PROJECT / "scripts/analyze_response_errors.py"),
                    "--gold", str(root / "gold.jsonl"),
                    "--arm", f"a={root / 'arm_a.jsonl'}",
                    "--arm", f"b={root / 'arm_b.jsonl'}",
                    "--bridge", str(root / "bridge.jsonl"),
                    "--main", "a", "--base", "b",
                    "--output", str(output),
                ],
                capture_output=True, text=True, cwd=PROJECT,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["samples"], 2)
            self.assertEqual(set(report["arms"]), {"a", "b"})
            # a 第 1 条全中、第 2 条全不中；b 两条都全中 → a 对 b 一胜一负
            head = report["head_to_head"]
            self.assertEqual((head["main"], head["base"], head["wins"], head["losses"]), ("a", "b", 0, 1))

    def test_missing_arm_rows_fail_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "gold.jsonl", [
                {"conversation_id": "c", "target_user_turn_id": 1, "sample_id": 'pp:["c",1,1]', "response_text": "在的。"},
            ])
            write_jsonl(root / "arm.jsonl", [{"id": "eval_x", "response_text": "在的。"}])
            result = subprocess.run(
                [sys.executable, str(PROJECT / "scripts/analyze_response_errors.py"),
                 "--gold", str(root / "gold.jsonl"), "--arm", f"a={root / 'arm.jsonl'}"],
                capture_output=True, text=True, cwd=PROJECT,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("无法解析预测 id", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
