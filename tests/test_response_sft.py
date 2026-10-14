from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_response_inference import _load_conditions
from src.response.sft_data import (
    BASE_SYSTEM_PROMPT,
    build_eval_sample,
    build_sft_samples,
    build_sft_samples_from_b1,
    history_messages,
    profile_text,
    response_turn_consistency,
    system_prompt,
)


def make_conversation(conversation_id: str, points: list[dict]) -> dict:
    """Build a conversation row matching train_public.jsonl's shape.

    turns: 1 user / 2 assistant / 3 user / 4 assistant / 5 user.
    """
    turns = [
        {"turn_id": 1, "role": "user", "content": "第一轮用户发言"},
        {"turn_id": 2, "role": "assistant", "content": "第一轮助手回复"},
        {"turn_id": 3, "role": "user", "content": "第二轮用户发言"},
        {"turn_id": 4, "role": "assistant", "content": "第二轮助手回复"},
        {"turn_id": 5, "role": "user", "content": "第三轮用户发言"},
    ]
    return {"conversation_id": conversation_id, "turns": turns, "prediction_points": points}


def make_point(target_user_turn_id: int, response_text: str = "目标回复") -> dict:
    return {
        "target_user_turn_id": target_user_turn_id,
        "assistant_response_turn_id": target_user_turn_id + 1,
        "target": {
            "response_text": response_text,
            "emotion_label": "anxiety",
            "user_profile": {
                "personality_traits": ["sensitive"],
                "interests": ["games"],
                "style": ["rational"],
            },
            "memory_refs": [],
        },
    }


def make_b1_response_row(
    sample_id: str,
    target_user_turn_id: int = 3,
    context_turns: list[dict] | None = None,
    response_text: str = "目标回复",
) -> dict:
    """Build a B1 response JSONL row (frozen split conversion product)."""
    if context_turns is None:
        context_turns = [
            {"turn_id": 1, "role": "user", "content": "第一轮用户发言"},
            {"turn_id": 2, "role": "assistant", "content": "第一轮助手回复"},
            {"turn_id": 3, "role": "user", "content": "第二轮用户发言"},
        ]
    return {
        "sample_id": sample_id,
        "conversation_id": "train_conv_000001",
        "target_user_turn_id": target_user_turn_id,
        "source_split": "train",
        "context_turns": context_turns,
        "response_text": response_text,
        "memory_refs": [],
    }


def make_b1_classification_row(
    sample_id: str,
    emotion_label: str = "anxiety",
    user_profile: dict | None = None,
) -> dict:
    """Build the matching B1 classification row (gold conditions)."""
    return {
        "sample_id": sample_id,
        "conversation_id": "train_conv_000001",
        "target_user_turn_id": 3,
        "source_split": "train",
        "emotion_label": emotion_label,
        "user_profile": user_profile
        if user_profile is not None
        else {
            "personality_traits": ["sensitive"],
            "interests": ["games"],
            "style": ["rational"],
        },
    }


class HistoryMessagesTests(unittest.TestCase):
    def test_cutoff_excludes_response_and_future_turns(self) -> None:
        turns = make_conversation("c", [make_point(3)])["turns"]
        messages = history_messages(turns, 3)
        self.assertEqual(
            [message["content"] for message in messages],
            ["第一轮用户发言", "第一轮助手回复", "第二轮用户发言"],
        )
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "user"])
        for message in messages:
            self.assertNotIn("第二轮助手回复", message["content"])

    def test_last_message_is_target_user_turn(self) -> None:
        turns = make_conversation("c", [make_point(5)])["turns"]
        messages = history_messages(turns, 5)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "第三轮用户发言")

    def test_target_turn_must_be_user(self) -> None:
        turns = make_conversation("c", [make_point(3)])["turns"]
        with self.assertRaises(ValueError):
            history_messages(turns, 2)

    def test_window_keeps_most_recent_turns_including_target(self) -> None:
        turns = make_conversation("c", [make_point(5)])["turns"]
        messages = history_messages(turns, 5, max_history_turns=2)
        self.assertEqual(
            [message["content"] for message in messages],
            ["第二轮助手回复", "第三轮用户发言"],
        )


class ProfileTextTests(unittest.TestCase):
    def test_profile_text_renders_values(self) -> None:
        text = profile_text(
            {"personality_traits": ["sensitive"], "interests": ["games"], "style": ["rational"]}
        )
        self.assertIn("sensitive", text)
        self.assertIn("games", text)
        self.assertIn("rational", text)

    def test_profile_text_empty_lists(self) -> None:
        text = profile_text({"personality_traits": [], "interests": [], "style": []})
        self.assertIn("无", text)


class SystemPromptTests(unittest.TestCase):
    def test_base_prompt_without_condition(self) -> None:
        self.assertEqual(system_prompt(None), BASE_SYSTEM_PROMPT)

    def test_condition_appended(self) -> None:
        condition = {
            "emotion_label": "anxiety",
            "user_profile": {
                "personality_traits": ["sensitive"],
                "interests": [],
                "style": ["rational"],
            },
        }
        prompt = system_prompt(condition)
        self.assertTrue(prompt.startswith(BASE_SYSTEM_PROMPT))
        self.assertIn("anxiety", prompt)
        self.assertIn("sensitive", prompt)


class BuildSftSamplesTests(unittest.TestCase):
    def test_basic_fields_and_order(self) -> None:
        rows = [
            make_conversation("train_conv_000001", [make_point(1), make_point(3)]),
            make_conversation("train_conv_000002", [make_point(1)]),
        ]
        samples = build_sft_samples(rows, condition_mode="gold")
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]["id"], "train_conv_000001_t1")
        self.assertEqual(samples[1]["id"], "train_conv_000001_t3")
        self.assertEqual(samples[2]["id"], "train_conv_000002_t1")
        self.assertEqual(samples[0]["conversation_id"], "train_conv_000001")
        self.assertEqual(samples[0]["response_text"], "目标回复")
        self.assertEqual(samples[0]["target_user_turn_id"], 1)
        self.assertEqual(samples[0]["assistant_response_turn_id"], 2)

    def test_messages_start_with_system_and_end_with_target_user_turn(self) -> None:
        rows = [make_conversation("train_conv_000001", [make_point(3)])]
        sample = build_sft_samples(rows, condition_mode="gold")[0]
        messages = sample["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "第二轮用户发言")

    def test_response_turn_content_never_leaks_into_messages(self) -> None:
        rows = [make_conversation("train_conv_000001", [make_point(3)])]
        sample = build_sft_samples(rows, condition_mode="gold")[0]
        for message in sample["messages"]:
            self.assertNotIn("第二轮助手回复", message["content"])

    def test_condition_mode_none_keeps_base_prompt(self) -> None:
        rows = [make_conversation("train_conv_000001", [make_point(3)])]
        sample = build_sft_samples(rows, condition_mode="none")[0]
        self.assertEqual(sample["messages"][0]["content"], BASE_SYSTEM_PROMPT)

    def test_condition_mode_gold_uses_target_labels(self) -> None:
        rows = [make_conversation("train_conv_000001", [make_point(3)])]
        sample = build_sft_samples(rows, condition_mode="gold")[0]
        system_content = sample["messages"][0]["content"]
        self.assertIn("anxiety", system_content)
        self.assertIn("sensitive", system_content)

    def test_duplicate_target_turn_rejected(self) -> None:
        rows = [make_conversation("train_conv_000001", [make_point(3), make_point(3)])]
        with self.assertRaises(ValueError):
            build_sft_samples(rows, condition_mode="gold")

    def test_empty_response_rejected(self) -> None:
        rows = [make_conversation("train_conv_000001", [make_point(3, response_text="")])]
        with self.assertRaises(ValueError):
            build_sft_samples(rows, condition_mode="gold")

    def test_deterministic_output(self) -> None:
        rows = [make_conversation("train_conv_000001", [make_point(1), make_point(3)])]
        first = build_sft_samples(rows, condition_mode="gold")
        second = build_sft_samples(rows, condition_mode="gold")
        self.assertEqual(first, second)


class BuildEvalSampleTests(unittest.TestCase):
    def test_eval_row_shape(self) -> None:
        row = {
            "id": "eval_000001_t1",
            "conversation_id": "val_conv_000001",
            "target_user_turn_id": 1,
            "history": [{"turn_id": 1, "role": "user", "content": "导师让我大改论文，时间不够。"}],
        }
        sample = build_eval_sample(row)
        self.assertEqual(sample["id"], "eval_000001_t1")
        self.assertEqual(sample["conversation_id"], "val_conv_000001")
        self.assertEqual(sample["messages"][0]["role"], "system")
        self.assertEqual(sample["messages"][-1]["role"], "user")
        self.assertEqual(sample["messages"][-1]["content"], "导师让我大改论文，时间不够。")

    def test_eval_row_with_condition(self) -> None:
        row = {
            "id": "eval_000001_t1",
            "conversation_id": "val_conv_000001",
            "target_user_turn_id": 1,
            "history": [{"turn_id": 1, "role": "user", "content": "有点烦。"}],
        }
        condition = {
            "emotion_label": "anxiety",
            "user_profile": {"personality_traits": [], "interests": [], "style": []},
        }
        sample = build_eval_sample(row, condition=condition)
        self.assertIn("anxiety", sample["messages"][0]["content"])


class PredictedConditionFileTests(unittest.TestCase):
    """C's predicted conditions feed the same prompt structure as gold."""

    def _write(self, rows: list[dict]) -> Path:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        with handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return Path(handle.name)

    def test_conditions_keyed_by_id(self) -> None:
        path = self._write(
            [
                {
                    "id": "eval_000001_t1",
                    "emotion_label": "anxiety",
                    "user_profile": {"personality_traits": [], "interests": [], "style": []},
                }
            ]
        )
        conditions = _load_conditions(path)
        self.assertEqual(conditions["eval_000001_t1"]["emotion_label"], "anxiety")
        self.assertIn("user_profile", conditions["eval_000001_t1"])
        path.unlink()

    def test_row_without_profile_rejected(self) -> None:
        path = self._write([{"id": "eval_000001_t1", "emotion_label": "anxiety"}])
        with self.assertRaises(ValueError):
            _load_conditions(path)
        path.unlink()

    def test_empty_file_rejected(self) -> None:
        path = self._write([])
        with self.assertRaises(ValueError):
            _load_conditions(path)
        path.unlink()


class BuildSftSamplesFromB1Tests(unittest.TestCase):
    """The frozen B1 response JSONL path used for the D2 first round."""

    def test_basic_fields_and_messages(self) -> None:
        rows = [make_b1_response_row('pp:["train_conv_000001",3,1]')]
        samples = build_sft_samples_from_b1(rows, condition_mode="none")
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["id"], 'pp:["train_conv_000001",3,1]')
        self.assertEqual(sample["conversation_id"], "train_conv_000001")
        self.assertEqual(sample["target_user_turn_id"], 3)
        self.assertEqual(sample["response_text"], "目标回复")
        messages = sample["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual([m["role"] for m in messages[1:]], ["user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "第二轮用户发言")

    def test_context_after_target_rejected(self) -> None:
        rows = [
            make_b1_response_row(
                "s1",
                context_turns=[
                    {"turn_id": 1, "role": "user", "content": "一"},
                    {"turn_id": 2, "role": "user", "content": "二"},
                    {"turn_id": 4, "role": "user", "content": "未来轮"},
                ],
            )
        ]
        with self.assertRaises(ValueError):
            build_sft_samples_from_b1(rows, condition_mode="none")

    def test_target_turn_must_be_last_user_turn(self) -> None:
        rows = [
            make_b1_response_row(
                "s1",
                context_turns=[
                    {"turn_id": 1, "role": "user", "content": "一"},
                    {"turn_id": 3, "role": "user", "content": "二"},
                    {"turn_id": 2, "role": "user", "content": "旧轮"},
                ],
            )
        ]
        with self.assertRaises(ValueError):
            build_sft_samples_from_b1(rows, condition_mode="none")

    def test_ascending_turn_ids_required(self) -> None:
        rows = [
            make_b1_response_row(
                "s1",
                context_turns=[
                    {"turn_id": 3, "role": "user", "content": "三"},
                    {"turn_id": 2, "role": "user", "content": "二"},
                ],
            )
        ]
        with self.assertRaises(ValueError):
            build_sft_samples_from_b1(rows, condition_mode="none")

    def test_missing_field_rejected(self) -> None:
        row = make_b1_response_row("s1")
        del row["response_text"]
        with self.assertRaises(ValueError):
            build_sft_samples_from_b1([row], condition_mode="none")

    def test_empty_response_rejected(self) -> None:
        rows = [make_b1_response_row("s1", response_text="  ")]
        with self.assertRaises(ValueError):
            build_sft_samples_from_b1(rows, condition_mode="none")

    def test_duplicate_sample_id_rejected(self) -> None:
        rows = [make_b1_response_row("s1"), make_b1_response_row("s1")]
        with self.assertRaises(ValueError):
            build_sft_samples_from_b1(rows, condition_mode="none")

    def test_condition_mode_none_keeps_base_prompt(self) -> None:
        rows = [make_b1_response_row("s1")]
        sample = build_sft_samples_from_b1(rows, condition_mode="none")[0]
        self.assertEqual(sample["messages"][0]["content"], BASE_SYSTEM_PROMPT)

    def test_gold_joins_classification_rows_by_sample_id(self) -> None:
        rows = [make_b1_response_row('pp:["c",3,1]')]
        conditions = [make_b1_classification_row('pp:["c",3,1]')]
        sample = build_sft_samples_from_b1(rows, condition_rows=conditions, condition_mode="gold")[0]
        system_content = sample["messages"][0]["content"]
        self.assertIn("anxiety", system_content)
        self.assertIn("sensitive", system_content)

    def test_gold_without_condition_rows_rejected(self) -> None:
        rows = [make_b1_response_row("s1")]
        with self.assertRaises(ValueError):
            build_sft_samples_from_b1(rows, condition_rows=None, condition_mode="gold")

    def test_gold_missing_condition_rejected_not_silently_skipped(self) -> None:
        rows = [make_b1_response_row("s1"), make_b1_response_row("s2")]
        conditions = [make_b1_classification_row("s1")]
        with self.assertRaises(ValueError):
            build_sft_samples_from_b1(rows, condition_rows=conditions, condition_mode="gold")

    def test_max_history_turns_window_keeps_target(self) -> None:
        rows = [make_b1_response_row("s1")]
        sample = build_sft_samples_from_b1(rows, condition_mode="none", max_history_turns=2)[0]
        messages = sample["messages"]
        self.assertEqual([m["content"] for m in messages[1:]], ["第一轮助手回复", "第二轮用户发言"])

    def test_deterministic_output(self) -> None:
        rows = [make_b1_response_row("s1")]
        conditions = [make_b1_classification_row("s1")]
        first = build_sft_samples_from_b1(rows, condition_rows=conditions, condition_mode="gold")
        second = build_sft_samples_from_b1(rows, condition_rows=conditions, condition_mode="gold")
        self.assertEqual(first, second)


class BuildEvalSampleContextTurnsTests(unittest.TestCase):
    """B1 response rows double as eval input (context_turns, no history)."""

    def test_context_turns_row_shape(self) -> None:
        row = make_b1_response_row('pp:["c",3,1]')
        sample = build_eval_sample(row)
        self.assertEqual(sample["id"], 'pp:["c",3,1]')
        self.assertEqual(sample["messages"][0]["role"], "system")
        self.assertEqual(sample["messages"][-1]["role"], "user")
        self.assertEqual(sample["messages"][-1]["content"], "第二轮用户发言")

    def test_context_turns_with_condition(self) -> None:
        row = make_b1_response_row("s1")
        condition = {
            "emotion_label": "anxiety",
            "user_profile": {"personality_traits": [], "interests": [], "style": []},
        }
        sample = build_eval_sample(row, condition=condition)
        self.assertIn("anxiety", sample["messages"][0]["content"])

    def test_context_turns_future_turns_excluded(self) -> None:
        # Eval input is filtered at the target turn (the D1 contract): future
        # turns never reach the prompt. The training loader is the strict
        # side and rejects them outright (test_context_after_target_rejected).
        row = make_b1_response_row(
            "s1",
            context_turns=[
                {"turn_id": 1, "role": "user", "content": "一"},
                {"turn_id": 3, "role": "user", "content": "二"},
                {"turn_id": 4, "role": "user", "content": "未来轮"},
            ],
        )
        sample = build_eval_sample(row)
        contents = [message["content"] for message in sample["messages"]]
        self.assertNotIn("未来轮", contents)
        self.assertEqual(contents[-1], "二")

    def test_row_without_history_or_context_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_eval_sample({"id": "s1", "target_user_turn_id": 1})


class ResponseTurnConsistencyTests(unittest.TestCase):
    def test_counts_matches_and_mismatches(self) -> None:
        match = make_point(3, response_text="第二轮助手回复")
        mismatch = make_point(5, response_text="与正文不同的目标回复")
        rows = [make_conversation("train_conv_000001", [match, mismatch])]
        stats = response_turn_consistency(rows)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["match"], 1)
        self.assertEqual(stats["mismatch"], 1)
        self.assertEqual(stats["mismatch_examples"], ["train_conv_000001_t5"])


if __name__ == "__main__":
    unittest.main()
