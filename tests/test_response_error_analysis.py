from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.response.error_analysis import (
    cjk_ratio,
    format_issues,
    length_bucket,
    markdown_artifacts,
    repeated_char_ngrams,
    repeated_sentences,
)


class CjkRatioTest(unittest.TestCase):
    def test_empty_is_zero(self) -> None:
        self.assertEqual(cjk_ratio(""), 0.0)

    def test_pure_chinese(self) -> None:
        self.assertEqual(cjk_ratio("我在听着"), 1.0)

    def test_mixed_text_between_zero_and_one(self) -> None:
        ratio = cjk_ratio("你好abc")
        self.assertTrue(0.0 < ratio < 1.0)


class MarkdownArtifactsTest(unittest.TestCase):
    def test_plain_chinese_has_no_artifacts(self) -> None:
        self.assertEqual(markdown_artifacts("我在。你慢慢说，不着急。"), [])

    def test_bold_and_heading_detected(self) -> None:
        found = markdown_artifacts("**重点**\n## 标题\n正文")
        self.assertIn("bold", found)
        self.assertIn("heading", found)

    def test_list_marker_requires_line_start(self) -> None:
        self.assertIn("list_marker", markdown_artifacts("- 第一项\n- 第二项"))
        self.assertNotIn("list_marker", markdown_artifacts("中间-破折号不算列表"))

    def test_code_fence_detected(self) -> None:
        self.assertIn("code_fence", markdown_artifacts("示例：\n```\ncode\n```"))


class RepetitionTest(unittest.TestCase):
    def test_repeated_sentences_counts_duplicates(self) -> None:
        text = "我会一直陪着你。今天过得怎么样？我会一直陪着你。"
        self.assertEqual(repeated_sentences(text), 1)

    def test_repeated_sentences_ignores_short(self) -> None:
        self.assertEqual(repeated_sentences("好。好。好。"), 0)

    def test_repeated_ngrams_detects_loop(self) -> None:
        text = "加油加油加油加油加油加油"
        self.assertGreater(repeated_char_ngrams(text, n=4, min_count=3), 0)

    def test_repeated_ngrams_empty_for_short_text(self) -> None:
        self.assertEqual(repeated_char_ngrams("短", n=8), 0)


class FormatIssuesTest(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(format_issues("   "), ["empty"])

    def test_near_empty(self) -> None:
        self.assertIn("near_empty", format_issues("我在。"))

    def test_low_cjk(self) -> None:
        self.assertIn("low_cjk", format_issues("OK ok ok ok ok ok"))

    def test_markdown_tag_prefixed(self) -> None:
        issues = format_issues("**加粗**开头的一句话，后面继续写很多内容避免触发其他检查。")
        self.assertTrue(any(i.startswith("markdown:") for i in issues))

    def test_clean_long_reply_has_no_issues(self) -> None:
        text = "听起来你最近真的压力很大，论文的事情一直压在心上。要不要跟我说说具体是哪一部分最让你担心？我可以陪你一起梳理一下。"
        self.assertEqual(format_issues(text), [])


class LengthBucketTest(unittest.TestCase):
    def test_boundaries(self) -> None:
        self.assertEqual(length_bucket(6), "<=6")
        self.assertEqual(length_bucket(7), "7-20")
        self.assertEqual(length_bucket(20), "7-20")
        self.assertEqual(length_bucket(21), "21-50")
        self.assertEqual(length_bucket(200), "101-200")
        self.assertEqual(length_bucket(201), ">200")


if __name__ == "__main__":
    unittest.main()
