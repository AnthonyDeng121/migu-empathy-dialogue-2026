from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.response.sft_data import (
    BASE_SYSTEM_PROMPT,
    build_eval_sample,
    build_sft_samples,
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
