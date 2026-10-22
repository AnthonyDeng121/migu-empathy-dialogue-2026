"""Leakage, manifest selection and historical replay checks for B T1."""
import copy
import unittest

from scripts.b_t1_profile_diagnostics import (
    PROFILE_LABELS_BY_DIMENSION, extended_profile_metrics, internal_split,
    conversation_coverage, dataset_stats, replay, screening_manifest, validate_rows,
)


def row(index, split="val", emotion="anxiety", turns=1):
    history = [{"turn_id": i, "role": "user" if i % 2 else "assistant", "content": "fixture"}
               for i in range(1, turns + 1)]
    import json
    return {"sample_id": f"{split}_{index}", "conversation_id": f"{split}_conv_{index // 2}",
            "target_user_turn_id": turns, "source_split": split, "context_turns": history,
            "context_text": "\n".join(json.dumps(t, ensure_ascii=False, separators=(",", ":")) for t in history),
            "emotion_label": emotion, "user_profile": {d: [] for d in PROFILE_LABELS_BY_DIMENSION}}


class T1Tests(unittest.TestCase):
    def test_conversation_coverage_distinguishes_zero_points_and_all_unusable(self):
        from scripts.prepare_training_data import sample_id
        raw = [{"conversation_id": cid, "prediction_points": [
                    {"target_user_turn_id": i} for i in turns]}
               for cid, turns in (("zero", []), ("bad", [1]), ("mixed", [1, 3]), ("good", [1]))]
        manifest = [
            {"sample_id": sample_id(r["conversation_id"], p, i),
             "conversation_id": r["conversation_id"], "source_split": "train",
             "usable": r["conversation_id"] == "good" or (r["conversation_id"] == "mixed" and i == 2),
             "reasons": ["fixture_anomaly"]}
            for r in raw for i, p in enumerate(r["prediction_points"], 1)]
        samples = [{"sample_id": r["sample_id"], "conversation_id": r["conversation_id"]}
                   for r in manifest if r["usable"]]
        result = conversation_coverage(raw, manifest, samples, "train")
        self.assertEqual(result["raw_conversations"], 4)
        self.assertEqual(result["prediction_point_conversations"], 3)
        self.assertEqual(result["usable_sample_conversations"], 2)
        self.assertEqual(result["no_prediction_point_conversation_ids"], ["zero"])
        self.assertEqual(result["all_prediction_points_unusable_conversation_ids"], ["bad"])
        self.assertEqual(result["other_missing_conversations"], 0)
        self.assertTrue(result["difference_reconciles"])
        self.assertEqual(result["all_prediction_points_unusable_evidence"][0]["excluded_points"][0]["reasons"], ["fixture_anomaly"])
        # Raw record count and unique conversation count must remain distinct.
        duplicate_zero = conversation_coverage(raw + [raw[0]], manifest, samples, "train")
        self.assertEqual(duplicate_zero["raw_records"], 5)
        self.assertEqual(duplicate_zero["raw_conversations"], 4)
        # Missing an actual sample must fail, even when its conversation remains.
        for bad_manifest, bad_samples in ((manifest[:-1], samples), (manifest, samples[:-1]),
                                          (manifest, samples + samples[:1])):
            with self.assertRaises(ValueError):
                conversation_coverage(raw, bad_manifest, bad_samples, "train")

    def test_dataset_schema_names_usable_conversation_coverage(self):
        result = dataset_stats([row(i, "train") for i in range(4)])
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["usable_sample_conversations"], 2)
        self.assertNotIn("conversations", result)

    def test_split_reuses_historical_conversation_algorithm(self):
        import random
        rows = [row(i, "train") for i in range(120)]
        result = internal_split(rows)
        conversations = sorted({r["conversation_id"] for r in rows})
        random.Random(20261015).shuffle(conversations)
        expected = set(conversations[:round(len(conversations) * .15)])
        self.assertEqual({r["conversation_id"] for r in result if r["subset"] == "calibration"}, expected)
        self.assertEqual(result, internal_split(list(reversed(rows))))
        self.assertFalse({r["conversation_id"] for r in result if r["subset"] == "fit"} & expected)
        self.assertTrue(all("context_text" not in r and "context_turns" not in r for r in result))

    def test_validation_cannot_enter_internal_split(self):
        with self.assertRaises(ValueError):
            internal_split([row(1), row(2)])
        with self.assertRaises(ValueError):
            internal_split([row(1, "train")])

    def test_adapter_rejects_missing_null_future_and_leakage(self):
        train, val = [row(0, "train")], [row(1)]
        validate_rows(train, val)
        for mutate in (lambda r: r.pop("emotion_label"),
                       lambda r: r["user_profile"].update(style=None),
                       lambda r: r.update(target_user_turn_id=0),
                       lambda r: r.update(conversation_id=train[0]["conversation_id"])):
            bad = copy.deepcopy(val)
            mutate(bad[0])
            with self.assertRaises((ValueError, TypeError)):
                validate_rows(train, bad)

    def test_screening_deterministic_covers_strata_without_text(self):
        rows = [row(i, emotion=("care" if i < 9 else "anxiety"), turns=(1, 7, 17)[i % 3]) for i in range(200)]
        for i, r in enumerate(rows):
            if i % 2:
                r["user_profile"]["style"] = ["direct"]
        result = screening_manifest(rows)
        self.assertEqual(len(result), 160)
        self.assertEqual(len({r["sample_id"] for r in result}), 160)
        self.assertEqual(sum(r["emotion_label"] == "care" for r in result), 9)
        self.assertEqual({r["history_bucket"] for r in result}, {"short_1_5", "medium_6_15", "long_16_plus"})
        self.assertEqual({r["profile_nonempty_mask"] for r in result}, {"000", "001"})
        self.assertEqual(result, screening_manifest(list(reversed(rows))))
        allowed = {"sample_id", "conversation_id", "source_split", "source_file", "emotion_label", "history_bucket", "profile_nonempty_mask"}
        self.assertTrue(all(set(r) == allowed for r in result))
        for size in (127, 201):
            with self.assertRaises(ValueError):
                screening_manifest(rows, size)

    def test_replay_aligns_ids_and_rejects_missing_duplicate_extra(self):
        rows = [row(i) for i in range(3)]
        predictions = [{"id": r["sample_id"], "emotion_label": r["emotion_label"], "user_profile": r["user_profile"]} for r in rows]
        self.assertEqual(replay(rows, predictions)["emotion"]["accuracy"], 1)
        self.assertEqual(replay(rows, list(reversed(predictions)))["emotion"]["accuracy"], 1)
        for bad in (predictions[:-1], predictions+[predictions[0]], predictions[:-1]+[dict(predictions[0], id="extra")]):
            with self.assertRaises(ValueError):
                replay(rows, bad)

    def test_profile_nonempty_rate_is_not_positive_label_count(self):
        gold = [{d: [] for d in PROFILE_LABELS_BY_DIMENSION} for _ in range(2)]
        pred = copy.deepcopy(gold)
        gold[0]["style"] = ["direct"]
        pred[0]["style"] = ["direct", "rational"]
        result = extended_profile_metrics(gold, pred)
        style = result["per_dimension"]["style"]
        self.assertEqual(style["predicted_nonempty_rate"], .5)
        self.assertEqual(style["predicted_positive_count"], 2)
        self.assertEqual(style["micro"]["precision"], .5)
        self.assertEqual(style["micro"]["recall"], 1)
        self.assertAlmostEqual(style["micro"]["f1"], 2/3)
        self.assertEqual(style["nonempty_prediction_exact_precision"], 0)
        self.assertAlmostEqual(result["macro_exact_unrounded"], 5/6)
        self.assertAlmostEqual(style["macro"]["f1"], 1/len(PROFILE_LABELS_BY_DIMENSION["style"]))


if __name__ == "__main__":
    unittest.main()
