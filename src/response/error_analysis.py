"""回复错误的格式与内容诊断（纯函数，供脚本与单测复用）。"""
from __future__ import annotations

import re
from collections import Counter

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_SENTENCE_SPLIT = re.compile(r"[。！？!?\n]+")
_MARKDOWN_PATTERNS = {
    "code_fence": re.compile(r"```"),
    "bold": re.compile(r"\*\*[^*]+\*\*"),
    "heading": re.compile(r"(?m)^#{1,6}\s"),
    "blockquote": re.compile(r"(?m)^>\s"),
    "list_marker": re.compile(r"(?m)^\s*[-*]\s+\S"),
    "link": re.compile(r"\[[^\]]+\]\([^)]+\)"),
    "table_row": re.compile(r"(?m)^\|.*\|$"),
}


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_CJK.findall(text)) / len(text)


def markdown_artifacts(text: str) -> list[str]:
    """返回文本中出现的 markdown 结构标签列表（按名称排序）。"""
    return sorted(name for name, pattern in _MARKDOWN_PATTERNS.items() if pattern.search(text))


def repeated_sentences(text: str, min_length: int = 6) -> int:
    """统计重复出现的句子数（出现 >=2 次且长度 >= min_length）。"""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) >= min_length]
    counts = Counter(sentences)
    return sum(1 for s, c in counts.items() if c >= 2)


def repeated_char_ngrams(text: str, n: int = 8, min_count: int = 3) -> int:
    """统计自我重复的字符 n-gram 种数（同一 n-gram 出现 >= min_count 次，排除标点-only）。"""
    if len(text) < n:
        return 0
    grams = Counter(text[i : i + n] for i in range(len(text) - n + 1))
    return sum(1 for g, c in grams.items() if c >= min_count and _CJK.search(g))


def format_issues(text: str, near_empty_chars: int = 6) -> list[str]:
    """返回文本的格式问题标签：empty / near_empty / low_cjk / markdown:<name> / repeated_sentence / repeated_ngram。"""
    issues: list[str] = []
    stripped = text.strip()
    if not stripped:
        return ["empty"]
    if len(stripped) <= near_empty_chars:
        issues.append("near_empty")
    if cjk_ratio(stripped) < 0.5:
        issues.append("low_cjk")
    for name in markdown_artifacts(stripped):
        issues.append(f"markdown:{name}")
    if repeated_sentences(stripped) > 0:
        issues.append("repeated_sentence")
    if repeated_char_ngrams(stripped) > 0:
        issues.append("repeated_ngram")
    return issues


def length_bucket(length: int) -> str:
    """按字符数分桶：<=6 / 7-20 / 21-50 / 51-100 / 101-200 / >200。"""
    if length <= 6:
        return "<=6"
    if length <= 20:
        return "7-20"
    if length <= 50:
        return "21-50"
    if length <= 100:
        return "51-100"
    if length <= 200:
        return "101-200"
    return ">200"
