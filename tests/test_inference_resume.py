import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import run_integrated as integrated
from scripts import run_response_inference as response
from src.common.inference_resume import Journal, fingerprint, tree_digest


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.jsonl"
        self.rows = [{"id": f"test_{i}", "conversation_id": f"c{i}", "target_user_turn_id": 1,
                      "history": [{"turn_id": 1, "role": "user", "content": f"你好{i}"}]} for i in range(5)]
        self.input.write_text("".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8")
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "weights.bin").write_bytes(b"weights")
        self.config = {"condition_response": False, "max_new_tokens": 256,
                       "classifier_dir": self.model, "response_model_dir": self.model,
                       "response_adapter_dir": self.model, "device": "cpu"}
        self.cache = self.root / "cache"
        self.result = self.root / "result"
        self.calls = []
        self.fail_at = None
        owner = self

        class FakeModels:
            def __init__(self, config):
                pass

            def synchronize(self):
                pass

            def predict(self, row):
                if row["id"] == owner.fail_at:
                    raise RuntimeError("injected failure")
                owner.calls.append(row["id"])
                return {"id": row["id"], "response_text": "回复" + row["id"], "emotion_label": "neutral",
                        "user_profile": {"personality_traits": [], "interests": [], "style": []}, "memory_refs": []}

        self.factory = FakeModels
        self.addCleanup(patch.stopall)
        patch("builtins.print").start()

    def run_integrated(self, cache=True):
        return integrated.run(self.input, self.result, self.config, self.factory,
                              resume_dir=self.cache if cache else None)

    def interrupt(self):
        self.fail_at = "test_2"
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.run_integrated()
        self.assertFalse((self.result / "submission.jsonl").exists())
        self.fail_at = None

    def test_integrated_failure_resume_matches_full_run_and_preserves_files(self):
        self.result.mkdir()
        (self.result / "keep.txt").write_text("keep")
        self.interrupt()
        report = self.run_integrated()
        self.assertEqual(self.calls, [r["id"] for r in self.rows])
        restored = (self.result / "submission.jsonl").read_bytes()
        self.assertEqual(report["resume"]["reused_samples"], 2)
        self.assertEqual(report["resume"]["new_samples"], 3)
        self.assertFalse(report["complete"])
        self.assertFalse(report["resume"]["cold_start"])
        self.assertEqual(report["rounds"], 5)
        self.assertAlmostEqual(sum(report["latencies_ms"]) / 1000,
                               report["resume"]["cumulative_successful_sample_seconds"])
        self.run_integrated(cache=False)
        self.assertEqual(restored, (self.result / "submission.jsonl").read_bytes())
        self.assertEqual((self.result / "keep.txt").read_text(), "keep")

    def test_partial_final_line_is_recomputed(self):
        self.interrupt()
        with (self.cache / "records.jsonl").open("ab") as handle:
            handle.write(b'{"index":2,"prediction":')
        self.run_integrated()
        self.assertEqual(self.calls, [r["id"] for r in self.rows])

    def test_complete_corrupt_line_is_rejected_before_model_loading(self):
        self.interrupt()
        path = self.cache / "records.jsonl"
        data = path.read_text(encoding="utf-8")
        path.write_text(data.replace('"neutral"', '"joy"', 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_integrated()
        self.assertEqual(len(self.calls), 2)

    def test_changed_input_order_model_tokenizer_threshold_and_generation_are_rejected(self):
        self.interrupt()
        original = self.input.read_bytes()
        mutations = [
            lambda: self.input.write_bytes(original.replace(b"test_4", b"test_x")),
            lambda: self.input.write_bytes(b"\n".join(original.splitlines()[::-1]) + b"\n"),
            lambda: (self.model / "weights.bin").write_bytes(b"changed"),
            lambda: (self.model / "tokenizer.json").write_text("changed"),
            lambda: (self.model / "thresholds.json").write_text("0.7"),
            lambda: self.config.update(max_new_tokens=128),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                mutate()
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    self.run_integrated()
                self.input.write_bytes(original)
                (self.model / "weights.bin").write_bytes(b"weights")
                (self.model / "tokenizer.json").unlink(missing_ok=True)
                (self.model / "thresholds.json").unlink(missing_ok=True)
                self.config["max_new_tokens"] = 256
        self.assertEqual(len(self.calls), 2)

    def test_code_and_environment_changes_are_fingerprinted(self):
        project = self.root / "project"
        (project / "src").mkdir(parents=True)
        code = project / "src" / "module.py"
        code.write_text("a=1")
        before = fingerprint(project, {"test": self.input}, {}, {})
        code.write_text("a=2")
        self.assertNotEqual(before, fingerprint(project, {"test": self.input}, {}, {}))
        code.write_text("a=1")
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "changed"}):
            self.assertNotEqual(before, fingerprint(project, {"test": self.input}, {}, {}))
        if os.name != "nt":
            external = self.root / "external"
            external.mkdir()
            asset = external / "tokenizer.json"
            asset.write_text("first")
            (self.model / "encoder").symlink_to(external, target_is_directory=True)
            before = tree_digest(self.model)
            asset.write_text("changed")
            self.assertNotEqual(before, tree_digest(self.model))
            (external / "cycle").symlink_to(self.model, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Cyclic"):
                tree_digest(self.model)

    def test_concurrent_writer_rejected_and_lock_released_after_failure(self):
        with Journal(self.cache, {"v": 1}, ["a"], response._validate_response):
            with self.assertRaisesRegex(RuntimeError, "active writer"):
                with Journal(self.cache, {"v": 1}, ["a"], response._validate_response):
                    self.fail("concurrent writer allowed")
        with Journal(self.cache, {"v": 1}, ["a"], response._validate_response):
            pass

    def test_process_death_keeps_progress_and_releases_os_lock(self):
        code = """import os, sys
from src.common.inference_resume import Journal
from scripts.run_response_inference import _validate_response
with Journal(sys.argv[1], {'v': 1}, ['a', 'b'], _validate_response) as journal:
    journal.append({'id': 'a', 'response_text': 'ok'}, 0.25, 2)
    os._exit(17)
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.cache)],
                                cwd=integrated.PROJECT, capture_output=True)
        self.assertEqual(result.returncode, 17, result.stderr)
        with Journal(self.cache, {"v": 1}, ["a", "b"], response._validate_response) as journal:
            self.assertEqual(journal.reused, 1)
            journal.append({"id": "b", "response_text": "ok"}, .5, 3)
            self.assertEqual(journal.summary()["cumulative_successful_sample_seconds"], .75)

    def test_first_100_timings_accumulate_across_attempts_and_completed_cache_is_reusable(self):
        self.rows = [{**self.rows[0], "id": f"test_{i}"} for i in range(105)]
        self.input.write_text("".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8")
        self.fail_at = "test_40"
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.run_integrated()
        self.fail_at = None
        report = self.run_integrated()
        self.assertEqual(report["rounds"], 100)
        self.assertFalse(report["complete"])
        self.assertEqual(report["resume"]["reused_samples"], 40)
        self.assertEqual(len(self.calls), 105)
        report = self.run_integrated()
        self.assertEqual(report["resume"]["new_samples"], 0)
        self.assertEqual(len(self.calls), 105)

    def test_bad_order_duplicate_or_invalid_cached_prediction_rejected(self):
        for prediction, index in [({"id": "wrong", "response_text": "ok"}, 0),
                                  ({"id": "a", "response_text": ""}, 0),
                                  ({"id": "a", "response_text": "ok"}, 1)]:
            with self.subTest(prediction=prediction, index=index):
                with Journal(self.cache, {"v": 1}, ["a"], response._validate_response) as journal:
                    with self.assertRaises(ValueError):
                        journal._validate_record({"index": index, "prediction": prediction, "seconds": .1, "tokens": 1})

    def response_args(self):
        return argparse.Namespace(input=self.input, model=self.model, adapter=None,
                                  output=self.root / "response.jsonl", performance_report=self.root / "perf.json",
                                  condition_jsonl=None, condition_fields="both", max_samples=None,
                                  max_new_tokens=256, min_new_tokens=0, full_set_size=2354, device="cpu",
                                  resume_dir=self.cache)

    def response_deps(self):
        owner = self

        class Tensor:
            shape = (1, 2)

            def __init__(self, text):
                self.text = text

            def to(self, device):
                return self

            def __getitem__(self, key):
                return self

            def numel(self):
                return 3

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def apply_chat_template(self, messages, **kwargs):
                return messages[-1]["content"]

            def __call__(self, text, **kwargs):
                return {"input_ids": Tensor(text)}

            def decode(self, generated, **kwargs):
                return "回复" + generated.text

        class Model:
            def to(self, device):
                return self

            def eval(self):
                pass

            def generate(self, input_ids, **kwargs):
                if input_ids.text == owner.fail_at:
                    raise RuntimeError("injected failure")
                owner.calls.append(input_ids.text)
                return [input_ids]

        torch = Mock()
        torch.device.return_value = SimpleNamespace(type="cpu")
        torch.inference_mode.side_effect = nullcontext
        return {"torch": torch, "PeftModel": Mock(),
                "AutoTokenizer": Mock(from_pretrained=Mock(return_value=Tokenizer())),
                "AutoModelForCausalLM": Mock(from_pretrained=Mock(return_value=Model()))}

    def test_response_failure_resume_matches_full_and_condition_changes_rejected(self):
        args = self.response_args()
        with patch.object(response, "_load_optional_dependencies", side_effect=self.response_deps):
            self.fail_at = "你好2"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                response.run(args)
            self.assertFalse(args.output.exists())
            self.fail_at = None
            args.max_new_tokens = 128
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                response.run(args)
            args.max_new_tokens = 256
            report = response.run(args)
            self.assertEqual(self.calls, ["你好" + str(i) for i in range(5)])
            restored = args.output.read_bytes()
            self.assertEqual(report["resume"]["reused_samples"], 2)
            self.assertEqual(report["generated_tokens"], 15)
            self.assertIsNone(report["full_set_estimate_seconds"])
            args.resume_dir = None
            response.run(args)
            self.assertEqual(restored, args.output.read_bytes())

    def test_response_condition_file_content_is_checked(self):
        args = self.response_args()
        args.condition_jsonl = self.root / "conditions.jsonl"
        args.condition_jsonl.write_text("".join(json.dumps({"id": r["id"], "emotion_label": "neutral"}) + "\n" for r in self.rows))
        args.condition_fields = "emotion"
        with patch.object(response, "_load_optional_dependencies", side_effect=self.response_deps):
            response.run(args)
            args.condition_jsonl.write_text(args.condition_jsonl.read_text().replace("neutral", "joy"))
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                response.run(args)


if __name__ == "__main__":
    unittest.main()
