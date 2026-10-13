from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import validate_prediction
from src.response.retrieval import Candidate, RetrievalBaseline, char_ngrams, cosine, history_until


def make_conversation(conversation_id: str, target: dict, history: str | None = None) -> dict:
    """Build a minimal train row matching train_public.jsonl's shape."""
    user_turn = history if history is not None else "最近有点烦，聚会之后老是会想起自己当时的表现。"
    return {
        "conversation_id": conversation_id,
        "turns": [
            {"turn_id": 1, "role": "user", "content": user_turn},
            {"turn_id": 2, "role": "assistant", "content": "明白，先不谈怎么办。"},
        ],
        "prediction_points": [
            {
                "target_user_turn_id": 1,
                "assistant_response_turn_id": 2,
                "target": target,
            }
        ],
    }


VALID_TARGET = {
    "response_text": "明白，先不谈怎么办。",
    "emotion_label": "anxiety",
    "user_profile": {"personality_traits": ["sensitive"], "interests": [], "style": ["rational"]},
    "memory_refs": ["mem_000001"],
}


class CharNgramsTests(unittest.TestCase):
    def test_char_ngrams_compact_and_split(self) -> None:
        grams = char_ngrams("你好世界", 3)
        self.assertEqual(grams, {"你好世": 1, "好世界": 1})

    def test_char_ngrams_ignores_whitespace(self) -> None:
        grams = char_ngrams("你好 世界", 3)
        self.assertEqual(grams, {"你好世": 1, "好世界": 1})

    def test_char_ngrams_short_text_returns_whole(self) -> None:
        self.assertEqual(char_ngrams("你好", 3), {"你好": 1})

    def test_char_ngrams_empty_text(self) -> None:
        self.assertEqual(char_ngrams("", 3), {})

    def test_char_ngrams_whitespace_only(self) -> None:
        self.assertEqual(char_ngrams("   ", 3), {})


class CosineTests(unittest.TestCase):
    def test_cosine_identical(self) -> None:
        self.assertAlmostEqual(cosine({"a": 1, "b": 2}, {"a": 1, "b": 2}), 1.0)

    def test_cosine_disjoint(self) -> None:
        self.assertEqual(cosine({"a": 1}, {"b": 1}), 0.0)

    def test_cosine_empty(self) -> None:
        self.assertEqual(cosine({}, {"a": 1}), 0.0)
        self.assertEqual(cosine({"a": 1}, {}), 0.0)


class HistoryUntilTests(unittest.TestCase):
    def test_history_until_cutoff(self) -> None:
        turns = [
            {"turn_id": 1, "role": "user", "content": "第一轮"},
            {"turn_id": 2, "role": "assistant", "content": "第二轮"},
            {"turn_id": 3, "role": "user", "content": "第三轮"},
        ]
        self.assertEqual(history_until(turns, 2), "user:第一轮\nassistant:第二轮")

    def test_history_until_full(self) -> None:
        turns = [
            {"turn_id": 1, "role": "user", "content": "第一轮"},
            {"turn_id": 2, "role": "assistant", "content": "第二轮"},
        ]
        self.assertEqual(history_until(turns, 2), "user:第一轮\nassistant:第二轮")


class RetrievalBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target_a = dict(VALID_TARGET)
        self.target_b = {
            "response_text": "完全不同的回复内容。",
            "emotion_label": "joy",
            "user_profile": {"personality_traits": [], "interests": ["music"], "style": ["brief"]},
            "memory_refs": [],
        }
        self.training = [
            make_conversation("train_conv_a", self.target_a, "最近有点烦，聚会之后老是会想起自己当时的表现。"),
            make_conversation("train_conv_b", self.target_b, "今天在公园听了一下午音乐，心情特别好。"),
        ]

    def test_predict_returns_valid_prediction(self) -> None:
        model = RetrievalBaseline(self.training, ngram=3)
        sample = {
            "id": "test1_000001_t1",
            "conversation_id": "test1_conv_000001",
            "target_user_turn_id": 1,
            "history": [{"turn_id": 1, "role": "user", "content": "最近有点烦，聚会之后老是会想起自己当时的表现。"}],
        }
        prediction = model.predict(sample)
        validate_prediction(prediction, sample["id"])
        self.assertEqual(prediction["memory_refs"], [])
        self.assertEqual(prediction["response_text"], self.target_a["response_text"])
        self.assertEqual(prediction["emotion_label"], self.target_a["emotion_label"])
        self.assertEqual(prediction["user_profile"], self.target_a["user_profile"])

    def test_predict_prefers_most_similar_candidate(self) -> None:
        model = RetrievalBaseline(self.training, ngram=3)
        sample = {
            "id": "test1_000002_t1",
            "conversation_id": "test1_conv_000002",
            "target_user_turn_id": 1,
            "history": [{"turn_id": 1, "role": "user", "content": "今天在公园听了一下午音乐，心情特别好。"}],
        }
        prediction = model.predict(sample)
        self.assertEqual(prediction["response_text"], self.target_b["response_text"])

    def test_predict_cosine_tie_equal_votes_returns_first_candidate(self) -> None:
        """Lock the tie contract: equal-length candidates share only the role prefix,
        cosines tie exactly, their votes tie too, and predict returns the first
        candidate in training order."""
        training = [
            make_conversation("train_conv_a", self.target_a, "甲" * 20),
            make_conversation("train_conv_b", self.target_b, "乙" * 20),
        ]
        model = RetrievalBaseline(training, ngram=3)
        sample = {
            "id": "test1_000003_t1",
            "conversation_id": "test1_conv_000003",
            "target_user_turn_id": 1,
            "history": [{"turn_id": 1, "role": "user", "content": "丙" * 20}],
        }
        prediction = model.predict(sample)
        validate_prediction(prediction, sample["id"])
        self.assertEqual(prediction["response_text"], self.target_a["response_text"])

    def test_predict_cosine_tie_prefers_higher_vote_candidate(self) -> None:
        """Lock the full tie contract: with cosines tied exactly, the candidate ranked
        higher in the pre-screen (more query n-gram votes) wins even though it is later
        in training order. Both candidates match the query with identical dot product
        and identical norm, but candidate 1 shares one more query n-gram (7 votes vs 6)."""
        training = [
            make_conversation("train_conv_a", self.target_a, "abcabc"),
            make_conversation("train_conv_b", self.target_b, "abcdefgh"),
        ]
        model = RetrievalBaseline(training, ngram=3)
        sample = {
            "id": "test1_000005_t1",
            "conversation_id": "test1_conv_000005",
            "target_user_turn_id": 1,
            "history": [{"turn_id": 1, "role": "user", "content": "abcd"}],
        }
        prediction = model.predict(sample)
        validate_prediction(prediction, sample["id"])
        self.assertEqual(prediction["response_text"], self.target_b["response_text"])

    def test_predict_empty_training_library_raises(self) -> None:
        """An empty retrieval library has no candidate to return; predict must fail loudly."""
        model = RetrievalBaseline([], ngram=3)
        sample = {
            "id": "test1_000004_t1",
            "conversation_id": "test1_conv_000004",
            "target_user_turn_id": 1,
            "history": [{"turn_id": 1, "role": "user", "content": "随便说什么"}],
        }
        with self.assertRaises(ValueError):
            model.predict(sample)

    def test_candidate_holds_target_and_features(self) -> None:
        model = RetrievalBaseline(self.training, ngram=3)
        self.assertEqual(len(model.candidates), 2)
        for index, candidate in enumerate(model.candidates):
            self.assertIsInstance(candidate, Candidate)
            self.assertTrue(candidate.features)
            expected = history_until(self.training[index]["turns"], 1)
            self.assertEqual(candidate.text, expected)


if __name__ == "__main__":
    unittest.main()
