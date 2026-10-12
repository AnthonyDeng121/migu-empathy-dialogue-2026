from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


def char_ngrams(text: str, n: int = 3) -> Counter[str]:
    compact = "".join(text.split())
    if len(compact) < n:
        return Counter([compact]) if compact else Counter()
    return Counter(compact[index:index + n] for index in range(len(compact) - n + 1))


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    lnorm = sum(value * value for value in left.values()) ** 0.5
    rnorm = sum(value * value for value in right.values()) ** 0.5
    return dot / (lnorm * rnorm) if lnorm and rnorm else 0.0


def history_until(turns: list[dict[str, Any]], target_turn_id: int) -> str:
    return "\n".join(f"{turn['role']}:{turn['content']}" for turn in turns if turn["turn_id"] <= target_turn_id)


@dataclass(frozen=True)
class Candidate:
    text: str
    target: dict[str, Any]
    features: Counter[str]


class RetrievalBaseline:
    def __init__(self, training_rows: list[dict[str, Any]], ngram: int = 3) -> None:
        self.ngram = ngram
        self.candidates: list[Candidate] = []
        self.postings: dict[str, set[int]] = defaultdict(set)
        for conversation in training_rows:
            for point in conversation["prediction_points"]:
                text = history_until(conversation["turns"], point["target_user_turn_id"])
                features = char_ngrams(text, ngram)
                index = len(self.candidates)
                self.candidates.append(Candidate(text, point["target"], features))
                for token in features:
                    self.postings[token].add(index)

    def predict(self, sample: dict[str, Any], candidate_limit: int = 300) -> dict[str, Any]:
        query_text = history_until(sample["history"], sample["target_user_turn_id"])
        query = char_ngrams(query_text, self.ngram)
        votes: Counter[int] = Counter()
        for token, count in query.items():
            for index in self.postings.get(token, ()):
                votes[index] += count
        pool = [idx for idx, _ in votes.most_common(candidate_limit)] or list(range(min(candidate_limit, len(self.candidates))))
        best = max(pool, key=lambda idx: cosine(query, self.candidates[idx].features))
        target = self.candidates[best].target
        return {
            "id": sample["id"],
            "response_text": target["response_text"],
            "emotion_label": target["emotion_label"],
            "user_profile": target["user_profile"],
            "memory_refs": [],
        }

