"""Core validation, metric and conversation aggregation checks using synthetic profiles."""
import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

from scripts.b_profile_eval import evaluate_files, evaluate_rows, index_rows
from src.common.io import write_jsonl
from src.data.schema import PROFILE_LABELS


def profile(**values):
    return {dim: values.get(dim, []) for dim in PROFILE_LABELS}


def fixtures():
    # A has two prediction points; B and C each have one. Never treat A as two conversations.
    gold = [
        {"sample_id": "a1", "conversation_id": "A", "emotion_label": "joy", "history": [{}],
         "user_profile": profile(interests=["music"])},
        {"sample_id": "a2", "conversation_id": "A", "emotion_label": "care", "history": [{}] * 6,
         "user_profile": profile(style=["brief"])},
        {"sample_id": "b1", "conversation_id": "B", "emotion_label": "joy", "history": [{}] * 16,
         "user_profile": profile()},
        {"sample_id": "c1", "conversation_id": "C", "emotion_label": "care", "history": [{}] * 5,
         "user_profile": profile(personality_traits=["open"])},
    ]
    baseline = [{"id": r["sample_id"], "user_profile": profile()} for r in gold]
    candidate = copy.deepcopy(baseline)
    candidate[0]["user_profile"] = copy.deepcopy(gold[0]["user_profile"])
    candidate[1]["user_profile"] = copy.deepcopy(gold[1]["user_profile"])
    candidate[2]["user_profile"] = profile(style=["brief"])
    return gold, baseline, candidate


class ProfileEvalTests(unittest.TestCase):
    def setUp(self):
        self.gold, self.baseline, self.candidate = fixtures()

    def compare(self):
        return evaluate_rows(self.gold, self.candidate, baseline=self.baseline)

    def test_conversation_aggregation_and_equal_weight_delta(self):
        conv = self.compare()["paired"]["conversations"]
        self.assertEqual((conv["total"], conv["wins"], conv["ties"], conv["losses"]), (3, 1, 1, 1))
        self.assertEqual(conv["mean_paired_delta"], 0.0)
        self.assertEqual(conv["win_rate"], 1 / 3)
        a = conv["details"][0]
        self.assertEqual(a["sample_ids"], ["a1", "a2"])
        self.assertEqual(a["prediction_points"], 2)
        self.assertEqual(a["baseline_score"], 2 / 3)
        self.assertEqual(a["candidate_score"], 1)
        self.assertEqual(a["delta"], 1 / 3)

    def test_identical_baseline(self):
        result = evaluate_rows(self.gold, self.baseline, baseline=self.baseline)
        conv = result["paired"]["conversations"]
        self.assertEqual((conv["wins"], conv["ties"], conv["losses"], conv["mean_paired_delta"]), (0, 3, 0, 0))
        self.assertEqual(result["paired"]["samples"]["full_profile"]["identical"], 4)

    def test_improvement_and_regression_direction(self):
        perfect = [{"id": r["sample_id"], "user_profile": r["user_profile"]} for r in self.gold]
        better = evaluate_rows(self.gold, perfect, baseline=self.baseline)["paired"]["conversations"]
        worse = evaluate_rows(self.gold, self.baseline, baseline=perfect)["paired"]["conversations"]
        self.assertEqual((better["wins"], better["ties"], better["losses"]), (2, 1, 0))
        self.assertGreater(better["mean_paired_delta"], 0)
        self.assertEqual(better["mean_paired_delta"], -worse["mean_paired_delta"])

    def test_sample_pair_categories_and_dimensions(self):
        paired = self.compare()["paired"]["samples"]
        self.assertEqual(paired["full_profile"], {
            "identical": 1, "changed": 3, "baseline_correct_candidate_wrong": 1,
            "baseline_wrong_candidate_correct": 2, "both_correct": 0, "both_wrong": 1})
        self.assertEqual(paired["interests"]["baseline_wrong_candidate_correct"], 1)
        self.assertEqual(paired["style"]["baseline_correct_candidate_wrong"], 1)
        self.assertEqual(paired["personality_traits"]["both_correct"], 3)

    def test_reordering_all_inputs_preserves_entire_result(self):
        expected = self.compare()
        for seed in range(5):
            rows = copy.deepcopy([self.gold, self.baseline, self.candidate])
            for data in rows:
                random.Random(seed).shuffle(data)
            self.assertEqual(evaluate_rows(rows[0], rows[2], baseline=rows[1]), expected)

    def test_full_exact_differs_from_macro_exact(self):
        report = evaluate_rows(self.gold, self.baseline)
        metrics = report["summary"]["prediction"]
        self.assertEqual(metrics["full_profile_exact_match"], 1 / 4)
        self.assertEqual(metrics["macro_exact_unrounded"], 9 / 12)
        self.assertEqual(metrics["predicted_all_empty_rate"], 1)
        self.assertEqual(metrics["predicted_any_nonempty_rate"], 0)
        self.assertEqual(metrics["global_micro"], dict(precision=0, recall=0, f1=0))

    def test_prf_support_and_fixed_label_macro(self):
        metrics = self.compare()["summary"]["candidate"]
        style = metrics["per_dimension"]["style"]
        self.assertEqual(style["support"], 1)
        self.assertEqual(style["samples"], 4)
        self.assertEqual(style["micro"], dict(precision=0.5, recall=1, f1=2 / 3))
        self.assertEqual(style["macro"]["f1"], (2 / 3) / 11)
        self.assertEqual(style["gold_nonempty_rate"], 0.25)
        self.assertEqual(style["predicted_nonempty_rate"], 0.5)
        self.assertEqual(style["per_label"]["emoji_user"]["support"], 0)
        self.assertAlmostEqual(metrics["global_micro"]["f1"], 2 / 3)

    def test_empty_behavior_four_cells(self):
        metrics = self.compare()["summary"]["candidate"]
        self.assertEqual(metrics["any_profile_empty_behavior"], {
            "gold_empty_pred_empty": 0, "gold_empty_pred_nonempty": 1,
            "gold_nonempty_pred_empty": 1, "gold_nonempty_pred_nonempty": 2})
        self.assertEqual(metrics["per_dimension"]["style"]["empty_behavior"], {
            "gold_empty_pred_empty": 2, "gold_empty_pred_nonempty": 1,
            "gold_nonempty_pred_empty": 0, "gold_nonempty_pred_nonempty": 1})

    def test_slice_support_and_t1_bucket_boundaries(self):
        slices = self.compare()["slices"]
        for key, count in [("history:short_1_5", 2), ("history:medium_6_15", 1),
                           ("history:long_16_plus", 1), ("gold_profile:empty", 1),
                           ("gold_profile:nonempty", 3), ("emotion:care", 2)]:
            self.assertEqual(slices[key]["samples"], count)
            self.assertIn("paired", slices[key])
        gold = copy.deepcopy(self.gold)
        gold[1]["history"] = [{}] * 15
        self.assertEqual(evaluate_rows(gold, self.baseline)["slices"]["history:medium_6_15"]["samples"], 1)

    def test_missing_and_extra_reported_separately_even_equal_length(self):
        self.candidate[-1]["id"] = "extra"
        with self.assertRaises(ValueError) as caught:
            self.compare()
        report = json.loads(str(caught.exception).split(": ", 1)[1])
        self.assertEqual(report["candidate"]["missing_ids"], ["c1"])
        self.assertEqual(report["candidate"]["extra_ids"], ["extra"])

    def test_duplicate_ids_rejected_in_every_input(self):
        for target in ("gold", "baseline", "candidate"):
            gold, baseline, candidate = fixtures()
            rows = {"gold": gold, "baseline": baseline, "candidate": candidate}[target]
            rows.append(copy.deepcopy(rows[0]))
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "duplicate ID"):
                evaluate_rows(gold, candidate, baseline=baseline)

    def test_invalid_ids_and_conflicting_aliases(self):
        for key in (None, "", " ", 1, True, []):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "invalid sample ID"):
                index_rows([{"id": key, "user_profile": profile()}], "test")
        with self.assertRaisesRegex(ValueError, "conflicting sample_id/id"):
            index_rows([{"id": "x", "sample_id": "y"}], "test")

    def test_missing_dimension_is_not_empty(self):
        del self.candidate[0]["user_profile"]["style"]
        with self.assertRaisesRegex(ValueError, "three official dimensions"):
            self.compare()

    def test_invalid_profiles_and_labels_fail_loudly(self):
        for bad in (None, [], "", {}, profile(style=None), profile(style="brief"),
                    profile(style=[{}]), profile(style=[1]), profile(style=["invalid"]),
                    profile(style=["brief", "brief"]), dict(profile(), unexpected=[])):
            with self.subTest(profile=bad), self.assertRaises(ValueError):
                index_rows([{"id": "x", "user_profile": bad}], "test")

    def test_empty_inputs_fail(self):
        for gold, pred in (([], self.baseline), (self.gold, [])):
            with self.assertRaisesRegex(ValueError, "empty input"):
                evaluate_rows(gold, pred)

    def test_label_order_is_set_equality(self):
        self.gold[0]["user_profile"]["style"] = ["brief", "formal"]
        self.candidate[0]["user_profile"] = copy.deepcopy(self.gold[0]["user_profile"])
        self.candidate[0]["user_profile"]["style"].reverse()
        result = self.compare()
        self.assertEqual(result["summary"]["candidate"]["full_profile_exact_match"], 0.5)

    def test_conversation_required_and_prediction_conflict(self):
        gold = copy.deepcopy(self.gold)
        del gold[0]["conversation_id"]
        with self.assertRaisesRegex(ValueError, "conversation_id required"):
            evaluate_rows(gold, self.baseline)
        for bad in ("other", "", None):
            self.candidate[0]["conversation_id"] = bad
            with self.subTest(cid=bad), self.assertRaisesRegex(ValueError, "conversation_id"):
                self.compare()

    def test_manifest_join_and_unavailable_metadata(self):
        gold = copy.deepcopy(self.gold)
        manifest = []
        buckets = ("short_1_5", "medium_6_15", "long_16_plus", "short_1_5")
        for r, bucket in zip(gold, buckets):
            manifest.append(dict(sample_id=r["sample_id"], conversation_id=r["conversation_id"],
                                 emotion_label=r.pop("emotion_label"), history_bucket=bucket))
            del r["history"]
        self.assertEqual(evaluate_rows(gold, self.candidate, baseline=self.baseline, manifest=manifest), self.compare())
        missing = evaluate_rows(gold, self.baseline)
        self.assertEqual(missing["slices"]["history:unavailable"]["samples"], 4)
        self.assertEqual(missing["slices"]["emotion:unavailable"]["samples"], 4)

    def test_manifest_conflicts_duplicates_extra_and_mask_rejected(self):
        entry = {"sample_id": "a1", "conversation_id": "A"}
        cases = [[dict(entry, conversation_id="wrong")], [dict(entry, emotion_label="care")],
                 [dict(entry, history_bucket="long_16_plus")], [dict(entry, profile_nonempty_mask="000")],
                 [dict(entry, sample_id="extra")], [entry, entry]]
        for manifest in cases:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                evaluate_rows(self.gold, self.baseline, manifest=manifest)

    def test_metadata_invalid_types_rejected(self):
        for field, value in (("history", []), ("history", None), ("emotion_label", []),
                             ("emotion_label", "invalid"), ("history_bucket", [])):
            gold = copy.deepcopy(self.gold)
            gold[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                evaluate_rows(gold, self.baseline)

    def test_files_hash_raw_bytes_and_reorder_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            g, b, c = (root / name for name in ("gold.jsonl", "baseline.jsonl", "candidate.jsonl"))
            for path, rows in ((g, self.gold), (b, self.baseline), (c, self.candidate)):
                write_jsonl(path, rows)
            report = evaluate_files(g, c, baseline=b)
            self.assertEqual(report["inputs"]["prediction"]["sha256"], hashlib.sha256(c.read_bytes()).hexdigest())
            raw = c.read_bytes()
            c.write_bytes(raw.replace(b"\n", b"\r\n"))
            changed = evaluate_files(g, c, baseline=b)
            self.assertNotEqual(report.pop("inputs"), changed.pop("inputs"))
            self.assertEqual(report, changed)
            write_jsonl(c, list(reversed(self.candidate)))
            shuffled = evaluate_files(g, c, baseline=b)
            shuffled.pop("inputs")
            self.assertEqual(report, shuffled)

    def test_file_failures_return_no_metrics_and_keep_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            g, p = Path(directory) / "g.jsonl", Path(directory) / "p.jsonl"
            write_jsonl(g, self.gold)
            for bad in (self.candidate[:-1], self.candidate + self.candidate[:1], [{"id": "a1"}]):
                write_jsonl(p, bad)
                result = evaluate_files(g, p)
                self.assertFalse(result["passed"])
                self.assertNotIn("summary", result)
                self.assertEqual(result["inputs"]["prediction"]["sha256"], hashlib.sha256(p.read_bytes()).hexdigest())
            p.write_text("{broken\n", encoding="utf-8")
            self.assertIn("invalid JSON", evaluate_files(g, p)["error"])


if __name__ == "__main__":
    unittest.main()
