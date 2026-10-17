import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from scripts.run_integrated import run, load_config


class FakeModels:
    def __init__(self, config):
        self.bad = config.get("bad", False)

    def synchronize(self):
        pass

    def predict(self, row):
        return {"id": "wrong" if self.bad else row["id"], "response_text": "我理解你的感受。",
                "emotion_label": "neutral", "user_profile": {"personality_traits": [], "interests": [], "style": []},
                "memory_refs": []}


class IntegratedTests(unittest.TestCase):
    def test_sequence_checkpoint_cannot_silently_supply_model_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = {"classifier_dir": "c", "response_model_dir": "d", "response_adapter_dir": "a",
                      "device": "cuda", "condition_response": False, "max_new_tokens": 256,
                      "classifier_format": "sequence", "profile_mode": "model"}
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires profile_mode=empty"):
                load_config(path)
            config["profile_mode"] = "empty"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(load_config(path)["classifier_format"], "sequence")

    def test_rejects_invalid_modes_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = {"classifier_dir": "c", "response_model_dir": "d", "response_adapter_dir": "a",
                      "device": "cuda", "condition_response": False, "max_new_tokens": 256,
                      "classifier_format": "packaged", "profile_mode": "empty"}
            for key, value in [("device", "auto"), ("profile_mode", "typo"), ("classifier_format", "typo"),
                               ("condition_response", "false"), ("max_new_tokens", 0)]:
                path.write_text(json.dumps({**config, key: value}), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_config(path)

    def prepare(self, directory, count=1):
        path = Path(directory) / "input.jsonl"
        rows = [{"id": f"test1_{i}", "conversation_id": f"c{i}", "target_user_turn_id": 1,
                 "history": [{"turn_id": 1, "role": "user", "content": "你好"}]} for i in range(count)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_full_input_and_first_100_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(directory, 101)
            result = Path(directory) / "result"
            with patch("builtins.print"):
                report = run(path, result, {"condition_response": False}, FakeModels)
            self.assertEqual(report["rounds"], 100)
            self.assertTrue(report["complete"])
            predictions = [json.loads(line) for line in (result / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(predictions), 101)
            self.assertEqual(predictions[-1]["id"], "test1_100")
            self.assertTrue((result / "performance_report.json").exists())
            with self.assertRaises(FileExistsError):
                run(path, result, {"condition_response": False}, FakeModels)

    def test_invalid_prediction_does_not_publish_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(directory)
            result = Path(directory) / "result"
            with self.assertRaises(ValueError):
                run(path, result, {"condition_response": False, "bad": True}, FakeModels)
            self.assertFalse((result / "submission.jsonl").exists())
            self.assertEqual(list(result.iterdir()), [])
