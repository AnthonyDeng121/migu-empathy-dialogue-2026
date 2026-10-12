from __future__ import annotations

import math
from collections import Counter
from typing import Iterable


def _chars(text: str) -> list[str]:
    return [char for char in text if not char.isspace()]


def accuracy(gold: Iterable[str], predicted: Iterable[str]) -> float:
    pairs = list(zip(gold, predicted))
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else 0.0


def bleu4(reference: str, hypothesis: str) -> float:
    ref, hyp = _chars(reference), _chars(hypothesis)
    if not hyp:
        return 0.0
    logs = []
    for n in range(1, 5):
        ref_counts = Counter(tuple(ref[i:i+n]) for i in range(max(0, len(ref)-n+1)))
        hyp_counts = Counter(tuple(hyp[i:i+n]) for i in range(max(0, len(hyp)-n+1)))
        total = sum(hyp_counts.values())
        matched = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
        logs.append(math.log((matched + 1) / (total + 1)))
    brevity = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / max(1, len(hyp)))
    return brevity * math.exp(sum(logs) / 4)


def rouge_l(reference: str, hypothesis: str) -> float:
    ref, hyp = _chars(reference), _chars(hypothesis)
    if not ref or not hyp:
        return 0.0
    previous = [0] * (len(hyp) + 1)
    for left in ref:
        current = [0]
        for index, right in enumerate(hyp, 1):
            current.append(previous[index - 1] + 1 if left == right else max(current[-1], previous[index]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(hyp), lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def profile_dimension_accuracy(gold: dict[str, list[str]], predicted: dict[str, list[str]]) -> float:
    dimensions = sorted(gold)
    return sum(set(gold[d]) == set(predicted.get(d, [])) for d in dimensions) / len(dimensions)

