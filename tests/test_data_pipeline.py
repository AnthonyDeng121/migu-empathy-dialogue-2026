from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_training_data import (
    PROJECT, build_full, build_smoke, check_full_result, expand_point, main,
    write_jsonl as write_training_jsonl,
)
from src.common.io import read_jsonl, write_jsonl


def conversation(cid: str = "train_1") -> dict:
    turns = [
        {"turn_id": 1, "role": "user", "content": "第一问\n第二行"},
        {"turn_id": 2, "role": "assistant", "content": "第一答"},
        {"turn_id": 3, "role": "user", "content": "第二问"},
        {"turn_id": 4, "role": "assistant", "content": "第二答"},
        {"turn_id": 5, "role": "user", "content": "未来问题"},
        {"turn_id": 6, "role": "assistant", "content": "未来答案"},
    ]
    return {
        "conversation_id": cid, "turns": turns,
        "prediction_points": [
            {
                "target_user_turn_id": tid, "assistant_response_turn_id": tid + 1,
                "target": {
                    "response_text": turns[tid]["content"], "emotion_label": "neutral",
                    "user_profile": {"personality_traits": [], "interests": [], "style": []},
                    "memory_refs": ["mem_000001"],
                },
            }
            for tid in (1, 3)
        ],
    }


def create_symlink_or_skip(test: unittest.TestCase, link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as error:
        test.skipTest(f"symlink creation is unavailable: {error}")


class DataPipelineTests(unittest.TestCase):
    def test_multiple_prediction_points_expand(self) -> None:
        source = conversation()
        original = copy.deepcopy(source)
        result = build_smoke([source], [conversation("val_1")])
        self.assertEqual(result["summary"]["prediction_points"], 2)
        self.assertEqual(result["summary"]["classification_samples"], 2)
        self.assertEqual(result["summary"]["response_samples"], 2)
        self.assertEqual(len({m["sample_id"] for m in result["manifest"]}), 2)
        self.assertTrue(all(m["source_split"] == "train" for m in result["manifest"]))
        self.assertEqual(source, original)

    def test_context_ends_at_target_user_and_excludes_response_and_future(self) -> None:
        result = build_smoke([conversation()], [])
        for key in ("classification", "response"):
            first, second = result[key]
            self.assertEqual([t["turn_id"] for t in first["context_turns"]], [1])
            self.assertEqual([t["turn_id"] for t in second["context_turns"]], [1, 2, 3])
            self.assertEqual(second["context_turns"][-1]["role"], "user")
            self.assertEqual(second["context_turns"][-1]["turn_id"], second["target_user_turn_id"])
            self.assertEqual([json.loads(line) for line in second["context_text"].splitlines()], second["context_turns"])
        self.assertFalse(result["summary"]["context_after_target"])
        self.assertFalse(result["summary"]["target_assistant_in_context"])

    def test_repeated_gold_text_in_past_is_not_leakage(self) -> None:
        source = conversation()
        source["turns"][0]["content"] = "第二答"
        result = build_smoke([source], [])
        self.assertEqual(result["summary"]["usable"], 2)
        self.assertTrue(result["summary"]["checks_passed"])

    def test_split_overlap_fails_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, val, output = root / "train.jsonl", root / "val.jsonl", root / "out"
            write_jsonl(train, [conversation("shared")])
            write_jsonl(val, [conversation("shared")])
            captured = io.StringIO()
            with redirect_stdout(captured), redirect_stderr(io.StringIO()):
                code = main(["--train", str(train), "--val", str(val), "--output-dir", str(output)])
            self.assertEqual(code, 1)
            self.assertIn('"train_val_conversation_overlap": 1', captured.getvalue())
            self.assertFalse(output.exists())

    def test_empty_profiles_and_gold_memory_refs_preserved(self) -> None:
        source = conversation()
        source["prediction_points"][1]["target"]["memory_refs"] = []
        result = build_smoke([source], [])
        self.assertEqual(result["classification"][0]["user_profile"], {
            "personality_traits": [], "interests": [], "style": [],
        })
        self.assertEqual(result["response"][0]["memory_refs"], ["mem_000001"])
        self.assertEqual(result["response"][1]["memory_refs"], [])
        self.assertIn("无法进一步验证引用", result["response"][0]["memory_refs_verification"])

    def test_structural_anomalies_remain_in_manifest_only(self) -> None:
        cases = {
            "missing_user": ("target_user_turn_missing", lambda r: r["prediction_points"][0].update(target_user_turn_id=99)),
            "wrong_user_role": ("target_user_turn_not_user", lambda r: r["turns"][0].update(role="assistant")),
            "missing_assistant": ("assistant_target_turn_missing", lambda r: r["prediction_points"][0].update(assistant_response_turn_id=99)),
            "wrong_response": ("assistant_target_gold_response_mismatch", lambda r: r["prediction_points"][0]["target"].update(response_text="错误标签")),
            "duplicate_turn_id": ("unsafe_context_turn_ids", lambda r: r["turns"][1].update(turn_id=1)),
            "unordered_turn_ids": ("unsafe_context_turn_ids", lambda r: r["turns"].reverse()),
            "past_assistant_reference": ("target_assistant_in_context", lambda r: r["prediction_points"][0].update(target_user_turn_id=3)),
        }
        for name, (reason, mutate) in cases.items():
            with self.subTest(name=name):
                source = conversation()
                source["prediction_points"] = source["prediction_points"][:1]
                mutate(source)
                result = build_smoke([source], [])
                self.assertEqual(len(result["manifest"]), 1)
                record = result["manifest"][0]
                self.assertFalse(record["usable"])
                self.assertIn(reason, record["reasons"])
                self.assertFalse(record["classification_included"])
                self.assertFalse(record["response_included"])
                self.assertEqual(result["classification"], [])
                self.assertEqual(result["response"], [])
                self.assertEqual(result["summary"]["excluded"], 1)

    def test_invalid_classification_labels_are_excluded(self) -> None:
        source = conversation()
        source["prediction_points"][0]["target"]["emotion_label"] = "unknown"
        result = build_smoke([source], [])
        self.assertEqual(result["summary"]["usable"], 1)
        self.assertTrue(any(reason.startswith("classification_labels_invalid") for reason in result["manifest"][0]["reasons"]))

    def test_seed_stable_conversation_selection_and_whole_groups(self) -> None:
        train = [conversation(f"train_{i:02}") for i in range(20)]
        first = build_smoke(train, [conversation("val_1")], count=5, seed=20261013)
        second = build_smoke(list(reversed(train)), [], count=5, seed=20261013)
        selected = first["summary"]["selected_conversation_ids"]
        self.assertEqual(selected, second["summary"]["selected_conversation_ids"])
        self.assertEqual(len(selected), 5)
        self.assertEqual(first["summary"]["prediction_points"], 10)
        self.assertNotIn("val_1", selected)
        for cid in selected:
            self.assertEqual(sum(m["conversation_id"] == cid for m in first["manifest"]), 2)
        self.assertEqual(
            [m["sample_id"] for m in first["manifest"]],
            [m["sample_id"] for m in second["manifest"]],
        )

    def test_existing_ids_reused_and_conflicts_fail_even_outside_smoke(self) -> None:
        source = conversation()
        source["prediction_points"][0]["id"] = "official_point_1"
        result = build_smoke([source], [])
        self.assertEqual(result["manifest"][0]["sample_id"], "official_point_1")
        val = conversation("val_1")
        val["prediction_points"][0]["id"] = "official_point_1"
        result = build_smoke([source], [val])
        self.assertFalse(result["summary"]["checks_passed"])
        self.assertEqual(result["summary"]["duplicate_sample_ids"], 1)

    def test_cli_writes_only_three_smoke_files_with_anomaly_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = conversation()
            source["prediction_points"][0]["target"]["response_text"] = "不一致"
            train, val, output = root / "train.jsonl", root / "val.jsonl", root / "out"
            write_jsonl(train, [source])
            write_jsonl(val, [conversation("val_1")])
            with redirect_stdout(io.StringIO()):
                code = main(["--train", str(train), "--val", str(val), "--output-dir", str(output), "--smoke-conversations", "1"])
            self.assertEqual(code, 0)
            self.assertEqual({p.name for p in output.iterdir()}, {
                "classification_smoke.jsonl", "response_smoke.jsonl", "split_manifest_smoke.jsonl",
            })
            self.assertEqual(len(read_jsonl(output / "split_manifest_smoke.jsonl")), 2)
            self.assertEqual(len(read_jsonl(output / "classification_smoke.jsonl")), 1)
            self.assertEqual(len(read_jsonl(output / "response_smoke.jsonl")), 1)

    def test_smoke_rejects_symlinked_output_file_without_overwriting_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, val, output = root / "train.jsonl", root / "val.jsonl", root / "out"
            protected = root / "protected.txt"
            write_jsonl(train, [conversation()])
            write_jsonl(val, [conversation("val_1")])
            output.mkdir()
            protected.write_text("keep me", encoding="utf-8")
            link = output / "classification_smoke.jsonl"
            create_symlink_or_skip(self, link, protected)

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main(["--train", str(train), "--val", str(val), "--output-dir", str(output)])

            self.assertEqual(code, 1)
            self.assertEqual(protected.read_text(encoding="utf-8"), "keep me")
            self.assertTrue((output / "classification_smoke.jsonl").is_symlink())

    def test_jsonl_writer_replaces_symlink_without_overwriting_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / "protected.txt"
            output = root / "classification_train.jsonl"
            protected.write_text("keep me", encoding="utf-8")
            create_symlink_or_skip(self, output, protected)

            write_training_jsonl(output, [{"sample_id": "sample-1"}])

            self.assertEqual(protected.read_text(encoding="utf-8"), "keep me")
            self.assertFalse(output.is_symlink())
            self.assertEqual(read_jsonl(output), [{"sample_id": "sample-1"}])


class FullDataPipelineTests(unittest.TestCase):
    def run_full(self, root: Path, train_rows: list, val_rows: list, output: Path | None = None) -> tuple[int, Path]:
        train, val = root / "train.jsonl", root / "val.jsonl"
        output = output or root / "out"
        write_jsonl(train, train_rows)
        write_jsonl(val, val_rows)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = main([
                "--mode", "full", "--train", str(train), "--val", str(val),
                "--output-dir", str(output),
            ])
        return code, output

    def test_explicit_source_split_in_all_three_records(self) -> None:
        for split in ("train", "val"):
            source = conversation(f"{split}_1")
            records = expand_point(source, source["prediction_points"][0], 1, 1, split)
            self.assertTrue(all(record["source_split"] == split for record in records))
        with self.assertRaises(ValueError):
            expand_point(source, source["prediction_points"][0], 1, 1, "test")

    def test_full_outputs_both_splits_and_manifest_covers_all_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code, output = self.run_full(Path(directory), [conversation()], [conversation("val_1")])
            self.assertEqual(code, 0)
            self.assertEqual({p.name for p in output.iterdir()}, {
                "classification_train.jsonl", "classification_val.jsonl",
                "response_train.jsonl", "response_val.jsonl", "split_manifest.jsonl",
                "anomalies.jsonl", "leakage.json", "manifest.json", "SHA256SUMS",
            })
            ledger = read_jsonl(output / "split_manifest.jsonl")
            self.assertEqual(len(ledger), 4)
            for split in ("train", "val"):
                records = [r for r in ledger if r["source_split"] == split]
                self.assertEqual(len(records), 2)
                for task in ("classification", "response"):
                    samples = read_jsonl(output / f"{task}_{split}.jsonl")
                    self.assertEqual(len(samples), 2)
                    self.assertTrue(all(s["source_split"] == split for s in samples))
                    self.assertEqual([s["sample_id"] for s in samples], [r["sample_id"] for r in records])
            report = json.loads((output / "leakage.json").read_text(encoding="utf-8"))
            self.assertEqual(report["train_prediction_points"], 2)
            self.assertEqual(report["val_prediction_points"], 2)
            self.assertEqual(report["train_conversations"], 1)
            self.assertEqual(report["val_conversations"], 1)
            self.assertTrue(report["checks_passed"])

    def test_full_anomalies_retained_and_excluded_with_actual_reasons(self) -> None:
        train, val = conversation(), conversation("val_1")
        # Nonadjacent future assistant, with a matching gold response: still unsafe.
        train["prediction_points"][0]["assistant_response_turn_id"] = 4
        train["prediction_points"][0]["target"]["response_text"] = "第二答"
        # Assistant in the past is retained as evidence, never emitted as input.
        val["prediction_points"][1]["assistant_response_turn_id"] = 2
        val["prediction_points"][1]["target"]["response_text"] = "第一答"
        with tempfile.TemporaryDirectory() as directory:
            code, output = self.run_full(Path(directory), [train], [val])
            self.assertEqual(code, 0)
            ledger = read_jsonl(output / "split_manifest.jsonl")
            anomalies = read_jsonl(output / "anomalies.jsonl")
            self.assertEqual(len(ledger), 4)
            self.assertEqual(anomalies, [r for r in ledger if not r["usable"]])
            self.assertEqual(len(anomalies), 2)
            for split in ("train", "val"):
                for task in ("classification", "response"):
                    self.assertEqual(len(read_jsonl(output / f"{task}_{split}.jsonl")), 1)
            report = json.loads((output / "leakage.json").read_text(encoding="utf-8"))
            self.assertEqual(report["excluded_train"], 1)
            self.assertEqual(report["excluded_val"], 1)
            self.assertEqual(report["reason_counts"]["assistant_target_not_adjacent_future_turn"], 2)
            self.assertFalse(report["target_assistant_in_context"])
            self.assertTrue(report["checks_passed"])

    def test_full_overlap_fails_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code, output = self.run_full(Path(directory), [conversation("shared")], [conversation("shared")])
            self.assertEqual(code, 1)
            self.assertFalse(output.exists())

    def test_full_duplicate_ids_fail_without_outputs(self) -> None:
        train, val = conversation(), conversation("val_1")
        train["prediction_points"][0]["id"] = "shared_point"
        val["prediction_points"][0]["id"] = "shared_point"
        with tempfile.TemporaryDirectory() as directory:
            code, output = self.run_full(Path(directory), [train], [val])
            self.assertEqual(code, 1)
            self.assertFalse(output.exists())

    def test_hashes_record_counts_and_input_hashes_match_actual_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, output = self.run_full(root, [conversation()], [conversation("val_1")])
            self.assertEqual(code, 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["conversion_version"], "b1_v1")
            self.assertEqual(manifest["anomaly_count"], 0)
            self.assertIn("turn_id", manifest["context_text_spec"])
            self.assertTrue(manifest["generated_at"].endswith("+08:00"))
            for split in ("train", "val"):
                path = root / f"{split}.jsonl"
                self.assertEqual(manifest["inputs"][split]["path"], str(path.resolve()))
                self.assertEqual(manifest["inputs"][split]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            for filename, metadata in manifest["outputs"].items():
                path = output / filename
                self.assertEqual(metadata["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                count = len(read_jsonl(path)) if path.suffix == ".jsonl" else 1
                self.assertEqual(metadata["record_count"], count)
            hashes = {}
            for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                digest, filename = line.split("  ", 1)
                hashes[filename] = digest
                self.assertEqual(digest, hashlib.sha256((output / filename).read_bytes()).hexdigest())
            self.assertEqual(set(hashes), set(manifest["outputs"]) | {"manifest.json"})
            self.assertNotIn("SHA256SUMS", hashes)

    def test_repeated_full_runs_produce_identical_data_and_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_code, first = self.run_full(root, [conversation()], [conversation("val_1")], root / "first")
            second_code, second = self.run_full(root, [conversation()], [conversation("val_1")], root / "second")
            self.assertEqual((first_code, second_code), (0, 0))
            for path in first.iterdir():
                if path.suffix == ".jsonl" or path.name == "leakage.json":
                    self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())

    def test_full_context_leakage_and_count_mismatch_fail_before_writing(self) -> None:
        for defect in ("future", "assistant", "count", "split"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                result = build_full([conversation()], [conversation("val_1")])
                if defect == "count":
                    result["classification_val"].pop()
                elif defect == "split":
                    result["classification_val"][0]["source_split"] = "train"
                else:
                    turn_id = 5 if defect == "future" else 2
                    result["classification_val"][0]["context_turns"].append({
                        "turn_id": turn_id, "role": "assistant", "content": "injected",
                    })
                report = check_full_result(result)
                self.assertFalse(report["checks_passed"])
                if defect == "future":
                    self.assertTrue(report["context_after_target"])
                elif defect == "assistant":
                    self.assertTrue(report["target_assistant_in_context"])
                with patch("scripts.prepare_training_data.build_full", return_value=result):
                    code, output = self.run_full(Path(directory), [conversation()], [conversation("val_1")])
                self.assertEqual(code, 1)
                self.assertFalse(output.exists())

    def test_full_rejects_raw_and_source_overwrite_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "classification_train.jsonl"
            val = root / "val.jsonl"
            write_jsonl(train, [conversation()])
            write_jsonl(val, [conversation("val_1")])
            original = train.read_bytes()
            for output in (root, PROJECT / "data/raw/b1_forbidden_test_output"):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    code = main(["--mode", "full", "--train", str(train), "--val", str(val), "--output-dir", str(output)])
                self.assertEqual(code, 1)
                self.assertEqual(train.read_bytes(), original)
            self.assertFalse((root / "manifest.json").exists())
            self.assertFalse((PROJECT / "data/raw/b1_forbidden_test_output").exists())

    def test_full_rejects_symlinked_output_file_without_overwriting_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            protected = root / "protected.txt"
            output.mkdir()
            protected.write_text("keep me", encoding="utf-8")
            link = output / "classification_train.jsonl"
            create_symlink_or_skip(self, link, protected)

            code, _ = self.run_full(root, [conversation()], [conversation("val_1")], output)

            self.assertEqual(code, 1)
            self.assertEqual(protected.read_text(encoding="utf-8"), "keep me")
            self.assertTrue((output / "classification_train.jsonl").is_symlink())


if __name__ == "__main__":
    unittest.main()
