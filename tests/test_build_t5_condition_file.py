from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_t5_condition_file as builder  # noqa: E402

EMOTIONS = [
    {"id": "a", "emotion_label": "anxiety"},
    {"id": "b", "emotion_label": "joy"},
]
PROFILES = [
    {
        "id": "a",
        "conversation_id": "conv_a",
        "user_profile": {
            "personality_traits": ["sensitive"],
            "interests": [],
            "style": ["casual"],
        },
        "profile_probabilities": {"personality_traits": {"sensitive": 0.9}},
    },
    {
        "id": "b",
        "user_profile": {
            "personality_traits": [],
            "interests": ["art_design"],
            "style": [],
        },
    },
]
INPUT = [
    {"sample_id": "b", "conversation_id": "conv_b", "context_turns": []},
    {"sample_id": "a", "conversation_id": "conv_a", "context_turns": []},
]


def _write(directory: Path, name: str, rows: list[dict]) -> Path:
    path = directory / name
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _args(directory: Path, **overrides) -> argparse.Namespace:
    values = {
        "emotions": _write(directory, "emotions.jsonl", EMOTIONS),
        "profiles": _write(directory, "profiles.jsonl", PROFILES),
        "input": _write(directory, "input.jsonl", INPUT),
        "output": directory / "conditions.jsonl",
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


def _run(args: argparse.Namespace) -> str:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        builder.run(args)
    return stdout.getvalue()


class BuildConditionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_three_key_rows_follow_input_order(self) -> None:
        args = _args(self.dir)
        conditions = builder.build_conditions(
            builder.load_input_ids(args.input),
            builder.load_emotions(args.emotions),
            builder.load_profiles(args.profiles),
        )
        self.assertEqual(conditions[0]["id"], "b")
        self.assertEqual(conditions[1]["id"], "a")
        self.assertEqual(
            [sorted(row) for row in conditions],
            [["emotion_label", "id", "user_profile"]] * 2,
        )
        self.assertEqual(conditions[0]["emotion_label"], "joy")
        self.assertEqual(conditions[1]["emotion_label"], "anxiety")
        self.assertEqual(
            conditions[1]["user_profile"],
            {"personality_traits": ["sensitive"], "interests": [], "style": ["casual"]},
        )

    def test_run_writes_file_and_reports_count_and_sha256(self) -> None:
        args = _args(self.dir)
        output = _run(args)
        rows = [
            json.loads(line)
            for line in args.output.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 2)
        self.assertIn("2 conditions", output)
        digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
        self.assertIn(digest, output)
        self.assertEqual(builder.sha256_file(args.output), digest)

    def test_input_row_missing_from_emotions_fails_and_lists_id(self) -> None:
        args = _args(
            self.dir,
            emotions=_write(self.dir, "short_emotions.jsonl", EMOTIONS[:1]),
        )
        code, stderr = _capture_failure(lambda: builder.run(args))
        self.assertEqual(code, 2)
        self.assertIn("b", stderr)
        self.assertIn("情绪来源", stderr)
        self.assertFalse(args.output.exists())

    def test_input_row_missing_from_profiles_fails_and_lists_id(self) -> None:
        args = _args(
            self.dir,
            profiles=_write(self.dir, "short_profiles.jsonl", PROFILES[:1]),
        )
        code, stderr = _capture_failure(lambda: builder.run(args))
        self.assertEqual(code, 2)
        self.assertIn("b", stderr)
        self.assertIn("画像来源", stderr)

    def test_duplicate_ids_rejected_in_every_source(self) -> None:
        cases = {
            "input": {"input": _write(self.dir, "dup_input.jsonl", INPUT + [INPUT[0]])},
            "emotions": {
                "emotions": _write(self.dir, "dup_emotions.jsonl", EMOTIONS + [EMOTIONS[0]])
            },
            "profiles": {
                "profiles": _write(self.dir, "dup_profiles.jsonl", PROFILES + [PROFILES[0]])
            },
        }
        for name, overrides in cases.items():
            with self.subTest(source=name):
                code, stderr = _capture_failure(
                    lambda overrides=overrides: builder.run(_args(self.dir, **overrides))
                )
                self.assertEqual(code, 2)
                self.assertIn("重复", stderr)

    def test_bad_emotion_label_rejected(self) -> None:
        for name, label in (("empty", ""), ("none", None), ("number", 7)):
            with self.subTest(label=name):
                rows = [dict(EMOTIONS[0], emotion_label=label), EMOTIONS[1]]
                args = _args(
                    self.dir,
                    emotions=_write(self.dir, f"bad_label_{name}.jsonl", rows),
                )
                code, stderr = _capture_failure(lambda: builder.run(args))
                self.assertEqual(code, 2)
                self.assertIn("emotion_label", stderr)

    def test_bad_user_profile_rejected(self) -> None:
        broken = {
            "missing_key": {"personality_traits": [], "interests": []},
            "not_a_list": {"personality_traits": "open", "interests": [], "style": []},
            "not_a_dict": ["sensitive"],
        }
        for name, profile in broken.items():
            with self.subTest(profile=name):
                rows = [dict(PROFILES[0], user_profile=profile), PROFILES[1]]
                args = _args(
                    self.dir,
                    profiles=_write(self.dir, f"bad_profile_{name}.jsonl", rows),
                )
                code, stderr = _capture_failure(lambda: builder.run(args))
                self.assertEqual(code, 2)
                self.assertIn("user_profile", stderr)

    def test_missing_source_file_fails(self) -> None:
        args = _args(self.dir, emotions=self.dir / "absent.jsonl")
        code, stderr = _capture_failure(lambda: builder.run(args))
        self.assertEqual(code, 2)
        self.assertIn("absent.jsonl", stderr)

    def test_empty_input_fails(self) -> None:
        args = _args(self.dir, input=_write(self.dir, "empty.jsonl", []))
        code, stderr = _capture_failure(lambda: builder.run(args))
        self.assertEqual(code, 2)
        self.assertIn("输入文件为空", stderr)


if __name__ == "__main__":
    unittest.main()
