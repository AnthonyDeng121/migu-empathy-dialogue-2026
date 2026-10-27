from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REMOTE = Path(__file__).resolve().parents[1] / "scripts" / "remote"


def _bash() -> str:
    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - bash ships with Linux/macOS
        raise unittest.SkipTest("bash is unavailable")
    return bash


class RemoteScriptSyntaxTests(unittest.TestCase):
    """CI only lints start.sh; these scripts ship in the same repo."""

    def test_every_script_parses(self) -> None:
        scripts = sorted(REMOTE.glob("*.sh"))
        self.assertTrue(scripts, "no remote scripts found")
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [_bash(), "-n", str(script)], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_scripts_are_invoked_via_bash(self) -> None:
        """Repo convention: scripts ship as 644 and run via `bash script.sh`
        (core.fileMode=false, so the exec bit is never recorded)."""
        for script in sorted(REMOTE.glob("*.sh")):
            with self.subTest(script=script.name):
                text = script.read_text(encoding="utf-8")
                self.assertTrue(text.startswith("#!"), f"{script.name} lacks a shebang")
        supervisor = (REMOTE / "supervisor.sh").read_text(encoding="utf-8")
        self.assertIn('bash "$(dirname "$0")/train_resilient.sh"', supervisor)
        self.assertIn('bash "$(dirname "$0")/watch_and_generate.sh"', supervisor)

    def test_supervisor_relaunches_both_siblings(self) -> None:
        text = (REMOTE / "supervisor.sh").read_text(encoding="utf-8")
        self.assertIn("train_resilient.sh", text)
        self.assertIn("watch_and_generate.sh", text)


class ResilientWrapperTests(unittest.TestCase):
    """The {RESUME} substitution is the whole point of the wrapper."""

    def _run_wrapper(self, with_checkpoint: bool, command: str) -> tuple[str, str]:
        """Run the wrapper in a temp ROOT; return (marker text, wrapper log)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            if with_checkpoint:
                (out / "checkpoints" / "step_000250").mkdir(parents=True)
            env = dict(os.environ)
            env.update(
                ROOT=str(root),
                OUT=str(out),
                TRAIN_CMD=command,
                MAX_ATTEMPTS="1",
                SLEEP_BETWEEN="0",
            )
            result = subprocess.run(
                [_bash(), str(REMOTE / "train_resilient.sh")],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            marker = root / "marker"
            log = root / "train_resilient.log"
            return (
                marker.read_text(encoding="utf-8") if marker.is_file() else "",
                log.read_text(encoding="utf-8") if log.is_file() else "",
            )

    def test_resume_flag_injected_from_latest_checkpoint(self) -> None:
        marker, log = self._run_wrapper(True, "echo ran {RESUME} > marker")
        self.assertIn("--resume-from", marker)
        self.assertIn("step_000250", marker)
        self.assertIn("TRAINING_COMPLETE", log)

    def test_no_resume_flag_without_checkpoint(self) -> None:
        marker, log = self._run_wrapper(False, "echo ran {RESUME} > marker")
        self.assertEqual(marker.strip(), "ran")
        self.assertIn("TRAINING_COMPLETE", log)

    def test_failed_command_is_recorded_not_retried_forever(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "out").mkdir()
            env = dict(os.environ)
            env.update(
                ROOT=str(root),
                OUT=str(root / "out"),
                TRAIN_CMD="exit 3",
                MAX_ATTEMPTS="1",
                SLEEP_BETWEEN="0",
            )
            result = subprocess.run(
                [_bash(), str(REMOTE / "train_resilient.sh")],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log = (root / "train_resilient.log").read_text(encoding="utf-8")
            self.assertIn("NO_CHECKPOINT_YET_FAIL", log)
            self.assertIn("RESILIENT_WRAPPER_DONE attempts=1", log)


if __name__ == "__main__":
    unittest.main()
