import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.assemble_candidate import assemble_rows, digest, fingerprint, load_inputs, main, portable_path


class CandidateAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [{"id": str(i), "conversation_id": f"c{i}", "target_user_turn_id": 1,
                        "history": [{"turn_id": 1, "role": "user", "content": "hello"}]} for i in range(2)]
        self.responses = [{"id": str(i), "response_text": f"reply {i}\n原文"} for i in range(2)]
        self.emotions = [{"id": str(i), "emotion_label": "joy"} for i in range(2)]
        self.profiles = [{"id": str(i), "user_profile": {"personality_traits": [], "interests": ["music"], "style": []}} for i in range(2)]

    def run_assembly(self):
        return assemble_rows(self.inputs, self.responses, self.emotions, self.profiles)

    def test_shuffle_aligns_by_id_preserves_response_and_uses_predictions(self):
        self.responses.reverse()
        self.emotions.reverse()
        rows, checks = self.run_assembly()
        self.assertEqual([r["id"] for r in rows], ["0", "1"])
        self.assertEqual(rows[0]["response_text"], "reply 0\n原文")
        self.assertEqual(rows[0]["emotion_label"], "joy")
        self.assertEqual(checks["alignment"]["response"]["order_mismatch_count"], 2)
        self.assertTrue(checks["validation"]["passed"])

    def test_classifier_superset_allowed_and_reported(self):
        self.emotions.append({"id": "other", "emotion_label": "joy"})
        _, checks = self.run_assembly()
        self.assertEqual(checks["alignment"]["emotion"]["unused_source_rows"], 1)

    def test_missing_each_source_rejected(self):
        for name in ("responses", "emotions", "profiles"):
            with self.subTest(name=name):
                original = getattr(self, name)
                setattr(self, name, original[:1])
                with self.assertRaises(ValueError):
                    self.run_assembly()
                setattr(self, name, original)

    def test_duplicate_each_source_including_input_rejected(self):
        for name in ("inputs", "responses", "emotions", "profiles"):
            with self.subTest(name=name):
                rows = getattr(self, name)
                rows.append(copy.deepcopy(rows[0]))
                with self.assertRaises(ValueError):
                    self.run_assembly()
                rows.pop()

    def test_extra_response_rejected(self):
        self.responses.append({"id": "other", "response_text": "extra"})
        with self.assertRaises(ValueError):
            self.run_assembly()

    def test_invalid_label_empty_text_and_wrong_conversation_rejected(self):
        for row, key, value in ((self.emotions[0], "emotion_label", "invalid"),
                                (self.responses[0], "response_text", "  "),
                                (self.profiles[0], "conversation_id", "wrong")):
            with self.subTest(key=key):
                original = copy.deepcopy(row)
                row[key] = value
                with self.assertRaises(ValueError):
                    self.run_assembly()
                row.clear()
                row.update(original)

    def test_empty_input_rejected(self):
        self.inputs = []
        with self.assertRaises(ValueError):
            self.run_assembly()

    def test_explicit_response_conversation_mismatch_rejected(self):
        for conversation in ("wrong", None, ""):
            with self.subTest(conversation=conversation):
                self.responses[0]["conversation_id"] = conversation
                with self.assertRaisesRegex(ValueError, "response: conversation mismatch"):
                    self.run_assembly()
        self.responses[0]["conversation_id"] = self.inputs[0]["conversation_id"]
        self.assertTrue(self.run_assembly()[1]["validation"]["passed"])
        del self.responses[0]["conversation_id"]
        self.assertTrue(self.run_assembly()[1]["validation"]["passed"])

    def test_manifest_and_hash_identical_across_output_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rows in (("input", self.inputs), ("responses", self.responses),
                               ("emotions", self.emotions), ("profiles", self.profiles)):
                (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            (root / "report").write_text(json.dumps({"samples": 2, "condition_mode": "none",
                "adapter_path": str(root / "weights/adapter"),
                "output_sha256": digest((root / "responses").read_bytes())}), encoding="utf-8")
            manifests, fingerprints = [], []
            for bundle in (root / "first", root / "other/nested"):
                argv = ["assemble_candidate", "--provenance-root", str(root),
                        "--output", str(bundle / "candidate.jsonl"),
                        "--manifest", str(bundle / "candidate.manifest.json"),
                        "--emotion-model", "weights/emotion", "--profile-model", "empty",
                        "--response-report", str(root / "report")]
                for name in ("input", "responses", "emotions", "profiles"):
                    argv += [f"--{name}", str(root / name)]
                with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    main()
                manifest = bundle / "candidate.manifest.json"
                manifests.append(manifest.read_bytes())
                fingerprints.append(fingerprint(manifest, repo_root=root, artifact_root=bundle))
                data = json.loads(manifests[-1])
                self.assertEqual(data["output"]["path_base"], "artifact")
                self.assertEqual(data["sources"]["input"]["path_base"], "repo")
                self.assertEqual(data["models"]["response"]["adapter"], "weights/adapter")
                self.assertNotIn(root.as_posix(), manifests[-1].decode())
            self.assertEqual(manifests[0], manifests[1])
            self.assertEqual(fingerprints[0], fingerprints[1])

    def test_provenance_rejects_paths_outside_declared_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "outside declared roots"):
                portable_path(root / "elsewhere/file", repo_root=root / "repo", artifact_root=root / "bundle")

    def test_cli_provenance_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = [{"sample_id": r["id"], "conversation_id": r["conversation_id"],
                          "target_user_turn_id": 1, "context_turns": r["history"],
                          "emotion_label": "anxiety"} for r in self.inputs]
            for name, rows in (("input", reference), ("responses", self.responses),
                               ("emotions", self.emotions), ("profiles", self.profiles)):
                (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            self.assertNotIn("emotion_label", load_inputs(root / "input")[0])
            report = {"samples": 2, "condition_mode": "none", "adapter_path": "latest",
                      "output_sha256": digest((root / "responses").read_bytes())}
            report_path = root / "response-report"
            argv = ["assemble_candidate"]
            for name in ("input", "responses", "emotions", "profiles", "response-report", "output", "manifest"):
                argv += [f"--{name}", str(root / name)]
            argv += ["--emotion-model", "c5", "--profile-model", "v1"]
            for change in ({"output_sha256": "wrong"}, {"condition_mode": "emotion"}, {"samples": 1}):
                report_path.write_text(json.dumps(report | change), encoding="utf-8")
                with patch("sys.argv", argv), self.assertRaises(ValueError):
                    main()
                self.assertFalse((root / "output").exists())
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            manifest = json.loads((root / "manifest").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_end_to_end_timing"])
            self.assertEqual(manifest["models"]["response"]["adapter"], "latest")
            self.assertEqual(manifest["output"]["sha256"], digest((root / "output").read_bytes()))
            original = (root / "output").read_bytes()
            with patch("sys.argv", argv), self.assertRaises(ValueError):
                main()
            self.assertEqual((root / "output").read_bytes(), original)
