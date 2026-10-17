from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_response_errors import pp_key


class PpKeyTest(unittest.TestCase):
    def test_target_is_second_element(self) -> None:
        # pp:["conv", target_user_turn_id, ...]——第二个元素才是 target turn
        self.assertEqual(pp_key('pp:["val_conv_000001",3,2]'), ("val_conv_000001", 3))

    def test_single_digit_ids(self) -> None:
        self.assertEqual(pp_key('pp:["val_conv_000002",5,1]'), ("val_conv_000002", 5))

    def test_non_pp_id_returns_none(self) -> None:
        self.assertIsNone(pp_key("eval_000002_t3"))
        self.assertIsNone(pp_key("pp-malformed"))

    def test_roundtrip_with_gold_sample_id(self) -> None:
        sample_id = json.dumps(["val_conv_000010", 12, 4], ensure_ascii=False)
        key = pp_key(f"pp:{sample_id}")
        self.assertEqual(key, ("val_conv_000010", 12))


if __name__ == "__main__":
    unittest.main()
