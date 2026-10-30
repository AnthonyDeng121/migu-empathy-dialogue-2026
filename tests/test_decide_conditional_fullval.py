from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.remote import decide_conditional_fullval as decider  # noqa: E402


def _comparison(key: str, rouge: float, off: float, wins: int, losses: int) -> dict:
    return {
        "arms": {key: {"rouge_l_mean": rouge}, "off": {"rouge_l_mean": off}},
        "head_to_head": {"main": key, "base": "off", "wins": wins, "losses": losses},
    }


def _write_comparison(directory: Path, arm: str, **kwargs) -> Path:
    path = directory / f"t5_{arm}_vs_off.json"
    path.write_text(
        json.dumps(_comparison(arm.split("_", 1)[1], **kwargs)), encoding="utf-8"
    )
    return path


def _args(comparisons: list[Path], **overrides) -> argparse.Namespace:
    values = {
        "comparison": list(comparisons),
        "delta_rouge": decider.DEFAULT_DELTA_ROUGE,
        "min_win_margin": decider.DEFAULT_MIN_WIN_MARGIN,
        "t4_adapter": decider.DEFAULT_T4_ADAPTER,
        "d4v2_adapter": decider.DEFAULT_D4V2_ADAPTER,
        "c5_conditions": decider.DEFAULT_C5_CONDITIONS,
        "both_conditions": decider.DEFAULT_BOTH_CONDITIONS,
        "out_dir": Path("results/response_gen"),
        "output": Path("decision.json"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _capture_failure(func) -> tuple[object, str]:
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            func()
    except SystemExit as exc:
        return exc.code, stderr.getvalue()
    raise AssertionError("expected SystemExit")


class PreregisteredBarTests(unittest.TestCase):
    def test_defaults_are_the_preregistered_line(self) -> None:
        self.assertEqual(decider.DEFAULT_DELTA_ROUGE, 0.005)
        self.assertEqual(decider.DEFAULT_MIN_WIN_MARGIN, 20)
        self.assertEqual(decider.DEFAULT_T4_ADAPTER, "weights/t4_longweight_2ep")
        self.assertEqual(
            decider.DEFAULT_D4V2_ADAPTER, "weights/response_sft_d4_2ep"
        )
        self.assertEqual(
            decider.DEFAULT_BOTH_CONDITIONS,
            "results/response_gen/t5_conditions_val_both.jsonl",
        )
        self.assertEqual(
            decider.ARM_ORDER,
            ("t4_emotion", "t4_eprofile", "d4v2_emotion", "d4v2_eprofile"),
        )

    def test_boundary_values_are_inclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            exact = _write_comparison(
                directory, "t4_emotion", rouge=0.5, off=0.4375, wins=60, losses=40
            )
            decision = decider.decide(
                _args([exact], delta_rouge=0.0625, min_win_margin=20)
            )
            self.assertTrue(decision["trigger"])
            self.assertEqual(decision["chosen"]["delta_rouge_l"], 0.0625)
            below = _write_comparison(
                directory, "t4_eprofile", rouge=0.5, off=0.46875, wins=60, losses=40
            )
            decision = decider.decide(
                _args([below], delta_rouge=0.0625, min_win_margin=20)
            )
            self.assertFalse(decision["trigger"])


class ArmIdentityTests(unittest.TestCase):
    def test_arm_from_filename_accepts_newcard_suffix(self) -> None:
        for name in (
            "t5_t4_emotion_vs_off.json",
            "t5_t4_emotion_vs_off_newcard.json",
            "t5_d4v2_eprofile_vs_off.json",
            "t5_d4v2_eprofile_vs_off_newcard.json",
        ):
            with self.subTest(name=name):
                expected = name[len("t5_"):].split("_vs_off")[0]
                self.assertEqual(decider.arm_from_filename(Path(name)), expected)

    def test_arm_from_filename_rejects_unknown_names(self) -> None:
        for name in (
            "t5_t5_emotion_vs_off.json",
            "t5_t4_emotion_vs_base.json",
            "t5_t4_emotion_vs_off_extra.json",
            "t4_emotion_vs_off.json",
        ):
            with self.subTest(name=name):
                code, stderr = _capture_failure(
                    lambda name=name: decider.arm_from_filename(Path(name))
                )
                self.assertEqual(code, 2)
                self.assertIn(name, stderr)


class DecideTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_emotion_arm_triggers_with_c5_conditions(self) -> None:
        path = _write_comparison(
            self.dir, "t4_emotion", rouge=0.42, off=0.40, wins=80, losses=30
        )
        decision = decider.decide(_args([path]))
        self.assertTrue(decision["trigger"])
        chosen = decision["chosen"]
        self.assertEqual(chosen["arm"], "t4_emotion")
        self.assertEqual(chosen["adapter"], "weights/t4_longweight_2ep")
        self.assertEqual(
            chosen["condition_jsonl"], "results/response_gen/c5_conditions_val_full.jsonl"
        )
        self.assertEqual(chosen["condition_fields"], "emotion")
        self.assertEqual(
            chosen["output"],
            "results/response_gen/t5_conditional_t4_emotion_val_newcard.jsonl",
        )
        self.assertEqual(
            chosen["resume_dir"],
            "results/response_gen/t5_conditional_t4_emotion_val_newcard_progress",
        )
        self.assertTrue(chosen["eligible"])
        self.assertEqual(decision["decision_version"], "t5_conditional_fullval_v1")
        self.assertEqual(
            decision["bar"],
            {"delta_rouge_l": 0.005, "min_win_margin": 20},
        )
        self.assertIn("cleared the bar", decision["reason"])

    def test_eprofile_arm_uses_both_conditions_and_d4v2_adapter(self) -> None:
        path = _write_comparison(
            self.dir, "d4v2_eprofile", rouge=0.31, off=0.30, wins=70, losses=20
        )
        decision = decider.decide(_args([path]))
        self.assertTrue(decision["trigger"])
        chosen = decision["chosen"]
        self.assertEqual(chosen["adapter"], "weights/response_sft_d4_2ep")
        self.assertEqual(
            chosen["condition_jsonl"],
            "results/response_gen/t5_conditions_val_both.jsonl",
        )
        self.assertEqual(chosen["condition_fields"], "both")
        self.assertEqual(
            chosen["output"],
            "results/response_gen/t5_conditional_d4v2_eprofile_val_newcard.jsonl",
        )

    def test_below_delta_bar_does_not_trigger(self) -> None:
        path = _write_comparison(
            self.dir, "t4_emotion", rouge=0.4009, off=0.40, wins=90, losses=10
        )
        decision = decider.decide(_args([path]))
        self.assertFalse(decision["trigger"])
        self.assertIsNone(decision["chosen"])
        self.assertFalse(decision["candidates"][0]["eligible"])
        self.assertEqual(decision["reason"], "no arm cleared the bar")

    def test_below_margin_bar_does_not_trigger(self) -> None:
        path = _write_comparison(
            self.dir, "t4_emotion", rouge=0.42, off=0.40, wins=25, losses=10
        )
        decision = decider.decide(_args([path]))
        self.assertFalse(decision["trigger"])
        self.assertEqual(decision["candidates"][0]["win_margin"], 15)
        self.assertEqual(decision["reason"], "no arm cleared the bar")

    def test_tie_break_prefers_delta_then_margin_then_fixed_order(self) -> None:
        equal = _write_comparison(
            self.dir, "t4_eprofile", rouge=0.5, off=0.4375, wins=60, losses=20
        )
        other = _write_comparison(
            self.dir, "t4_emotion", rouge=0.5, off=0.4375, wins=60, losses=20
        )
        decision = decider.decide(_args([equal, other]))
        self.assertEqual(decision["chosen"]["arm"], "t4_emotion")

        higher_margin = _write_comparison(
            self.dir, "d4v2_emotion", rouge=0.5, off=0.4375, wins=80, losses=10
        )
        decision = decider.decide(_args([equal, other, higher_margin]))
        self.assertEqual(decision["chosen"]["arm"], "d4v2_emotion")

        higher_delta = _write_comparison(
            self.dir, "d4v2_eprofile", rouge=0.75, off=0.4375, wins=40, losses=0
        )
        decision = decider.decide(_args([equal, other, higher_delta]))
        self.assertEqual(decision["chosen"]["arm"], "d4v2_eprofile")
        self.assertEqual(
            [item["arm"] for item in decision["candidates"]],
            ["t4_emotion", "t4_eprofile", "d4v2_eprofile"],
        )

    def test_missing_comparison_file_exits_2(self) -> None:
        absent = self.dir / "t5_t4_emotion_vs_off.json"
        code, stderr = _capture_failure(lambda: decider.decide(_args([absent])))
        self.assertEqual(code, 2)
        self.assertIn(str(absent), stderr)
        self.assertIn("缺失", stderr)

    def test_malformed_comparison_exits_2(self) -> None:
        broken = {
            "no_off_arm": {
                "arms": {"emotion": {"rouge_l_mean": 0.4}},
                "head_to_head": {"main": "emotion", "base": "off", "wins": 5, "losses": 1},
            },
            "no_head_to_head": {"arms": {"emotion": {"rouge_l_mean": 0.4}, "off": {"rouge_l_mean": 0.3}}},
            "float_wins": {
                "arms": {"emotion": {"rouge_l_mean": 0.4}, "off": {"rouge_l_mean": 0.3}},
                "head_to_head": {"main": "emotion", "base": "off", "wins": 5.5, "losses": 1},
            },
            "wrong_main": {
                "arms": {"emotion": {"rouge_l_mean": 0.4}, "off": {"rouge_l_mean": 0.3}},
                "head_to_head": {"main": "eprofile", "base": "off", "wins": 5, "losses": 1},
            },
        }
        for name, payload in broken.items():
            with self.subTest(payload=name):
                path = self.dir / f"t5_t4_emotion_vs_off_{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                code, stderr = _capture_failure(lambda: decider.decide(_args([path])))
                self.assertEqual(code, 2)
                self.assertIn(str(path), stderr)

    def test_duplicate_arm_exits_2(self) -> None:
        first = _write_comparison(
            self.dir, "t4_emotion", rouge=0.42, off=0.40, wins=80, losses=30
        )
        second = self.dir / "t5_t4_emotion_vs_off_newcard.json"
        second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
        code, stderr = _capture_failure(lambda: decider.decide(_args([first, second])))
        self.assertEqual(code, 2)
        self.assertIn("t4_emotion", stderr)

    def test_no_comparisons_leaves_trigger_false(self) -> None:
        decision = decider.decide(_args([]))
        self.assertFalse(decision["trigger"])
        self.assertEqual(decision["candidates"], [])
        self.assertEqual(decision["reason"], "no comparison files provided")

    def test_custom_paths_flow_into_candidate(self) -> None:
        path = _write_comparison(
            self.dir, "d4v2_emotion", rouge=0.5, off=0.4375, wins=60, losses=20
        )
        decision = decider.decide(
            _args(
                [path],
                t4_adapter="weights/custom_t4",
                d4v2_adapter="weights/custom_d4v2",
                c5_conditions="reports/custom_c5.jsonl",
                out_dir=Path("results/custom"),
            )
        )
        chosen = decision["chosen"]
        self.assertEqual(chosen["adapter"], "weights/custom_d4v2")
        self.assertEqual(chosen["condition_jsonl"], "reports/custom_c5.jsonl")
        self.assertEqual(chosen["output"], "results/custom/t5_conditional_d4v2_emotion_val_newcard.jsonl")
        self.assertEqual(
            chosen["resume_dir"], "results/custom/t5_conditional_d4v2_emotion_val_newcard_progress"
        )


class MainTests(unittest.TestCase):
    def test_main_writes_decision_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _write_comparison(
                directory, "t4_eprofile", rouge=0.42, off=0.40, wins=80, losses=30
            )
            output = directory / "decision.json"
            argv = [
                "decide_conditional_fullval.py",
                "--comparison",
                str(path),
                "--output",
                str(output),
            ]
            stdout = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                decider.main()
            decision = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(decision["trigger"])
            self.assertEqual(decision["chosen"]["arm"], "t4_eprofile")
            self.assertIn("TRIGGER: t4_eprofile", stdout.getvalue())
            self.assertIn(str(output), stdout.getvalue())

    def test_main_reports_no_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _write_comparison(
                directory, "t4_emotion", rouge=0.40, off=0.40, wins=5, losses=5
            )
            output = directory / "decision.json"
            argv = [
                "decide_conditional_fullval.py",
                "--comparison",
                str(path),
                "--output",
                str(output),
            ]
            stdout = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                decider.main()
            decision = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(decision["trigger"])
            self.assertEqual(decision["reason"], "no arm cleared the bar")
            self.assertIn("NO_TRIGGER", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
