from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_response_artifact import (  # noqa: E402
    INFERENCE_SCRIPT,
    artifact_digest,
    build_command,
    compare_runs,
    compare_to_recorded,
    enforce_offline,
    sha256_file,
    verify_digest,
)


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "model": Path("/models/Qwen2.5-1.5B-Instruct"),
        "adapter": Path("/models/adapter"),
        "eval_input": Path("/data/response_val.jsonl"),
        "samples": 8,
        "max_new_tokens": 256,
        "device": "auto",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class OfflineEnvTests(unittest.TestCase):
    """Offline must be forced, not inherited: a warm local cache would
    otherwise let a missing artifact pass on this machine and fail in the
    network-less submission image."""

    def test_forces_offline_and_reports_overrides(self) -> None:
        import os

        saved = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        try:
            os.environ["HF_HUB_OFFLINE"] = "0"
            overridden = enforce_offline()
            self.assertEqual(overridden["HF_HUB_OFFLINE"], "0")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_DATASETS_OFFLINE"], "1")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class DigestTests(unittest.TestCase):
    def test_sha256_of_known_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            path.write_bytes(b"migu")
            self.assertEqual(
                sha256_file(path),
                "37aba646ed939c02207663d316b0eff7704878fb2a2319d0455cf24cce633e80",
            )

    def test_digest_is_stable_and_covers_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_model.safetensors").write_bytes(b"weights")
            first = artifact_digest(root)
            self.assertEqual(first["files"], 1)
            self.assertEqual(artifact_digest(root)["digest"], first["digest"])
            # Renaming a file changes the digest: name-only changes are caught.
            (root / "adapter_model.safetensors").rename(root / "renamed.safetensors")
            self.assertNotEqual(artifact_digest(root)["digest"], first["digest"])

    def test_digest_ignores_hidden_cache_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            before = artifact_digest(root)
            cache = root / ".cache"
            cache.mkdir()
            (cache / "metadata.json").write_text("{}", encoding="utf-8")
            self.assertEqual(artifact_digest(root), before)

    def test_digest_detects_changed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            before = artifact_digest(root)
            (root / "config.json").write_text('{"r": 16}', encoding="utf-8")
            self.assertNotEqual(artifact_digest(root)["digest"], before["digest"])

    def test_missing_directory_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            artifact_digest(Path("/nonexistent/artifact"))


class VerifyDigestTests(unittest.TestCase):
    def test_ok_mismatch_and_unpinned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            digest = artifact_digest(root)["digest"]
            self.assertEqual(verify_digest("adapter", root, digest)["status"], "ok")
            self.assertEqual(
                verify_digest("adapter", root, "0" * 64)["status"], "mismatch"
            )
            self.assertEqual(verify_digest("adapter", root, None)["status"], "unpinned")


class CompareRunsTests(unittest.TestCase):
    def test_identical_runs_pass(self) -> None:
        verdict = compare_runs({"a": "x", "b": "y"}, {"b": "y", "a": "x"})
        self.assertTrue(verdict["identical"])
        self.assertEqual(verdict["differing_ids"], [])

    def test_differing_run_is_reported_with_ids(self) -> None:
        verdict = compare_runs({"a": "x", "b": "y"}, {"a": "x", "b": "z"})
        self.assertFalse(verdict["identical"])
        self.assertEqual(verdict["differing_ids"], ["b"])

    def test_missing_id_counts_as_difference(self) -> None:
        verdict = compare_runs({"a": "x"}, {})
        self.assertFalse(verdict["identical"])
        self.assertEqual(verdict["differing_ids"], ["a"])


class CompareToRecordedTests(unittest.TestCase):
    def test_exact_match_and_metrics(self) -> None:
        predictions = {"a": "我听着。", "b": "谢谢你的信任"}
        recorded = {"a": "我听着。", "b": "谢谢你的信任"}
        result = compare_to_recorded(predictions, recorded)
        self.assertEqual(result["exact_matches"], 2)
        self.assertEqual(result["exact_match_rate"], 1.0)
        self.assertEqual(result["mean_bleu4"], 1.0)
        self.assertEqual(result["mean_rouge_l"], 1.0)

    def test_drift_lowers_scores_and_empty_is_counted(self) -> None:
        predictions = {"a": "", "b": "完全不同的另一句话"}
        recorded = {"a": "我听着。", "b": "谢谢你的信任"}
        result = compare_to_recorded(predictions, recorded)
        self.assertEqual(result["exact_matches"], 0)
        self.assertEqual(result["empty_predictions"], 1)
        self.assertLess(result["mean_bleu4"], 1.0)

    def test_ids_absent_from_recorded_are_listed(self) -> None:
        result = compare_to_recorded({"a": "x", "c": "y"}, {"a": "x"})
        self.assertEqual(result["missing_from_recorded"], ["c"])
        self.assertEqual(result["compared"], 1)

    def test_no_overlap_reports_zero_without_crashing(self) -> None:
        result = compare_to_recorded({"a": "x"}, {"b": "y"})
        self.assertEqual(result["compared"], 0)
        self.assertIsNone(result["exact_match_rate"])


class BuildCommandTests(unittest.TestCase):
    def test_command_pins_offline_relevant_flags(self) -> None:
        args = _args()
        command = build_command(args, Path("/tmp/run1.jsonl"), Path("/tmp/run1.json"))
        self.assertEqual(command[1], str(INFERENCE_SCRIPT))
        self.assertIn("--adapter", command)
        self.assertIn(str(Path("/models/adapter")), command)
        self.assertIn("--max-samples", command)
        self.assertIn("8", command)
        self.assertIn("--device", command)
        self.assertIn("auto", command)

    def test_optional_flags_dropped_when_absent(self) -> None:
        args = _args(samples=None, max_new_tokens=None)
        command = build_command(args, Path("/tmp/o.jsonl"), Path("/tmp/o.json"))
        self.assertNotIn("--max-samples", command)
        self.assertNotIn("--max-new-tokens", command)


class KeepRunsTests(unittest.TestCase):
    def test_copies_both_runs_predictions_and_reports(self) -> None:
        from scripts.verify_response_artifact import keep_runs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run1.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
            (root / "run1_report.json").write_text("{}", encoding="utf-8")
            (root / "run2.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
            (root / "run2_report.json").write_text("{}", encoding="utf-8")
            destination = root / "kept"
            copied = keep_runs(root, destination)
            self.assertEqual(len(copied), 4)
            self.assertEqual(
                sorted(path.name for path in destination.iterdir()),
                ["run1.jsonl", "run1_report.json", "run2.jsonl", "run2_report.json"],
            )


class CopyPinnedFilesTests(unittest.TestCase):
    def test_copies_only_digest_covered_files_to_a_new_path(self) -> None:
        from scripts.verify_response_artifact import copy_pinned_files

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model"
            source.mkdir()
            (source / "model.safetensors").write_bytes(b"weights")
            (source / "config.json").write_text("{}", encoding="utf-8")
            (source / ".cache").mkdir()
            (source / ".cache" / "metadata.json").write_text("{}", encoding="utf-8")
            staging = copy_pinned_files(source, Path(directory) / "staged")
            self.assertEqual(
                sorted(path.name for path in staging.iterdir()),
                ["config.json", "model.safetensors"],
            )
            self.assertFalse((staging / ".cache").exists())
            self.assertEqual(
                artifact_digest(staging)["digest"], artifact_digest(source)["digest"]
            )


if __name__ == "__main__":
    unittest.main()
