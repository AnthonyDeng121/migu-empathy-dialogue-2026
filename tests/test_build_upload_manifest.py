"""Tests for scripts/remote/build_upload_manifest.py.

The manifest is the contract between ``upload_workload.sh`` and the remote
``sha256sum -c``: whatever it lists must be exactly what gets pushed, so a
wrong include/exclude rule shows up hours later as a failed generation run
instead of at upload time. The synthetic tree below therefore contains both
sides -- workload inputs and everything that must stay local.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "remote" / "build_upload_manifest.py"
MANIFEST_NAME = "MANIFEST.sha256"
LINE_RE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")

# Files the default rules must pick up, as repo-relative POSIX paths.
EXPECTED = {
    "README.md",
    "requirements-response.txt",
    "requirements.txt",
    ".github/workflows/ci.yml",
    "configs/c.json",
    "docs/d.md",
    "reports/r.json",
    "scripts/a.py",
    "scripts/remote/b.sh",
    "src/mod.py",
    "tests/t.py",
    "weights/README.md",
    "weights/response_sft_d4_2ep/adapter_config.json",
    "weights/response_sft_d4_2ep/adapter_model.safetensors",
    "weights/t4_longweight_2ep/adapter_config.json",
    "weights/t4_longweight_2ep/adapter_model.safetensors",
    "results/response_gen/keep.jsonl",
    "results/response_gen/keep2.json",
    "results/response_gen/report.json.publication.json",
    "data/processed/b1_full_v1/SHA256SUMS",
    "data/processed/b1_full_v1/manifest.json",
    "data/processed/b1_full_v1/response_val.jsonl",
}

# Files that exist in the tree but must never reach the remote host.
EXCLUDED = {
    "AGENTS.md",
    "MANIFEST.sha256",
    "scripts/__pycache__/a.cpython-312.pyc",
    "src/__pycache__/mod.pyc",
    "__pycache__/root.pyc",
    ".git/config",
    ".hermes-memory/memory.md",
    ".codex_tmp/scratch.txt",
    "results/baseline/old.jsonl",
    "results/response_gen/three_way.log",
    "results/response_gen/t4_val_progress/journal.jsonl",
    "results/response_gen/artifact_verify_day3/evidence.json",
    "weights/hf/Qwen2.5-1.5B-Instruct/model.safetensors",
    "weights/classification/old.safetensors",
    "weights/response_sft_smoke_0p5b/x.json",
    "data/processed/b1_full_v1/response_train.jsonl",
    "data/processed/b1_full_v1/split_manifest.jsonl",
    "data/processed/b1_full_v1/anomalies.jsonl",
    "outside/real.py",
    "scripts/link.py",
    "scripts/linkdir/real.py",
}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _symlink(link: Path, target: Path) -> bool:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:  # pragma: no cover - Windows without developer mode
        return False
    return True


def build_tree(root: Path) -> bool:
    """Create the synthetic repository; report whether symlinks were possible."""
    for rel in EXPECTED:
        _write(root / rel, f"content of {rel}\n")
    for rel in EXCLUDED - {"scripts/link.py", "scripts/linkdir/real.py", "outside/real.py"}:
        _write(root / rel, f"excluded content of {rel}\n")
    # A stale manifest from an earlier run must not list itself.
    _write(root / MANIFEST_NAME, "stale manifest from a previous run\n")
    # Real content that is only reachable through a symlink: neither the
    # linked file nor anything under the linked directory may be hashed.
    _write(root / "outside" / "real.py", "content of outside/real.py\n")
    linked_file = _symlink(root / "scripts" / "link.py", root / "scripts" / "a.py")
    linked_dir = _symlink(root / "scripts" / "linkdir", root / "outside")
    return linked_file and linked_dir


class BuildUploadManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.symlinks_supported = build_tree(self.root)
        self.output = self.root / "out" / MANIFEST_NAME

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_manifest(self, includes: tuple[str, ...] = (), output: Path | None = None):
        target = self.output if output is None else output
        cmd = [sys.executable, str(SCRIPT), "--root", str(self.root), "--output", str(target)]
        for rule in includes:
            cmd += ["--include", rule]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    def read_manifest(self, path: Path | None = None) -> list[str]:
        text = (path or self.output).read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"), "manifest must end with a newline")
        return text.splitlines()

    def test_default_manifest_lists_exactly_the_workload_inputs(self) -> None:
        result = self.run_manifest()
        self.assertEqual(result.returncode, 0, result.stderr)
        listed = {LINE_RE.match(line).group(2) for line in self.read_manifest()}
        self.assertEqual(listed, EXPECTED)
        self.assertEqual(listed & EXCLUDED, set())

    def test_line_format_and_hashes_match_the_files(self) -> None:
        self.run_manifest()
        for line in self.read_manifest():
            match = LINE_RE.match(line)
            self.assertIsNotNone(match, f"malformed manifest line: {line!r}")
            digest, relpath = match.group(1), match.group(2)
            self.assertNotIn("\\", relpath, "relpath must use POSIX separators")
            expected = hashlib.sha256((self.root / relpath).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, f"hash mismatch for {relpath}")

    def test_lines_are_sorted_by_relpath(self) -> None:
        self.run_manifest()
        relpaths = [LINE_RE.match(line).group(2) for line in self.read_manifest()]
        self.assertEqual(relpaths, sorted(relpaths))

    def test_summary_reports_file_count_and_bytes(self) -> None:
        result = self.run_manifest()
        summary = [line for line in result.stdout.splitlines() if line.startswith("PASS ")]
        self.assertEqual(len(summary), 1, result.stdout)
        match = re.search(r"files=(\d+) total_bytes=(\d+)", summary[0])
        self.assertIsNotNone(match, summary[0])
        total = sum((self.root / rel).stat().st_size for rel in EXPECTED)
        self.assertEqual(int(match.group(1)), len(EXPECTED))
        self.assertEqual(int(match.group(2)), total)

    def test_default_output_path_is_root_manifest(self) -> None:
        default = self.root / MANIFEST_NAME
        result = self.run_manifest(output=default)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(default.is_file())
        listed = {LINE_RE.match(line).group(2) for line in self.read_manifest(default)}
        self.assertNotIn(MANIFEST_NAME, listed)

    def test_missing_include_directory_fails_before_writing(self) -> None:
        shutil.rmtree(self.root / "weights" / "t4_longweight_2ep")
        result = self.run_manifest()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("weights/t4_longweight_2ep", result.stderr)
        self.assertFalse(self.output.exists(), "a failed build must not write a manifest")

    def test_missing_named_file_fails(self) -> None:
        (self.root / "data" / "processed" / "b1_full_v1" / "manifest.json").unlink()
        result = self.run_manifest()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("manifest.json", result.stderr)
        self.assertFalse(self.output.exists())

    def test_missing_glob_match_fails(self) -> None:
        for path in self.root.glob("requirements*.txt"):
            path.unlink()
        result = self.run_manifest()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requirements*.txt", result.stderr)

    def test_include_rules_replace_the_defaults(self) -> None:
        result = self.run_manifest(includes=("tree:src", "file:README.md"))
        self.assertEqual(result.returncode, 0, result.stderr)
        listed = {LINE_RE.match(line).group(2) for line in self.read_manifest()}
        self.assertEqual(listed, {"README.md", "src/mod.py"})

    def test_unknown_include_rule_fails(self) -> None:
        result = self.run_manifest(includes=("bogus:scripts",))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bogus:scripts", result.stderr)

    def test_symlinks_are_never_followed(self) -> None:
        if not self.symlinks_supported:
            self.skipTest("symlink creation is not permitted in this environment")
        result = self.run_manifest()
        self.assertEqual(result.returncode, 0, result.stderr)
        listed = {LINE_RE.match(line).group(2) for line in self.read_manifest()}
        self.assertNotIn("scripts/link.py", listed)
        self.assertNotIn("scripts/linkdir/real.py", listed)
        self.assertIn("scripts/a.py", listed)


if __name__ == "__main__":
    unittest.main()
