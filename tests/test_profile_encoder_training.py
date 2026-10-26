from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_profile_encoder import PROJECT, _portable_path, partition_by_manifest


class PortableProfileEncoderReportPathTests(unittest.TestCase):
    def test_keeps_repository_paths_relative(self) -> None:
        path = PROJECT / "reports" / "predictions.jsonl"

        self.assertEqual(_portable_path(path), "reports/predictions.jsonl")

    def test_hides_external_machine_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="private-user-") as temporary:
            path = Path(temporary) / "checkpoint"

            portable = _portable_path(path)

        self.assertEqual(portable, "external/checkpoint")
        self.assertNotIn("private-user-", portable)


class ProfileEncoderTrainingManifestTests(unittest.TestCase):
    def _manifest(self, root: Path, rows: list[dict[str, str]]) -> Path:
        path = root / "split.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    def test_partitions_rows_by_exact_sample_id_and_conversation(self) -> None:
        examples = [
            {"sample_id": "s1", "conversation_id": "c1"},
            {"sample_id": "s2", "conversation_id": "c1"},
            {"sample_id": "s3", "conversation_id": "c2"},
        ]
        manifest_rows = [
            {"sample_id": "s1", "conversation_id": "c1", "source_split": "train", "subset": "fit"},
            {"sample_id": "s2", "conversation_id": "c1", "source_split": "train", "subset": "fit"},
            {"sample_id": "s3", "conversation_id": "c2", "source_split": "train", "subset": "calibration"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            fit, calibration = partition_by_manifest(
                examples, self._manifest(Path(temporary), manifest_rows)
            )

        self.assertEqual([row["sample_id"] for row in fit], ["s1", "s2"])
        self.assertEqual([row["sample_id"] for row in calibration], ["s3"])

    def test_rejects_conversation_split_between_fit_and_calibration(self) -> None:
        examples = [
            {"sample_id": "s1", "conversation_id": "c1"},
            {"sample_id": "s2", "conversation_id": "c1"},
        ]
        manifest_rows = [
            {"sample_id": "s1", "conversation_id": "c1", "source_split": "train", "subset": "fit"},
            {"sample_id": "s2", "conversation_id": "c1", "source_split": "train", "subset": "calibration"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "conversation appears in both"):
                partition_by_manifest(
                    examples, self._manifest(Path(temporary), manifest_rows)
                )

    def test_rejects_incomplete_manifest(self) -> None:
        examples = [
            {"sample_id": "s1", "conversation_id": "c1"},
            {"sample_id": "s2", "conversation_id": "c2"},
        ]
        manifest_rows = [
            {"sample_id": "s1", "conversation_id": "c1", "source_split": "train", "subset": "fit"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "sample IDs differ"):
                partition_by_manifest(
                    examples, self._manifest(Path(temporary), manifest_rows)
                )


if __name__ == "__main__":
    unittest.main()
