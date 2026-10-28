from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
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


class WatchAndGenerateTests(unittest.TestCase):
    """Post-training generation pipeline safety."""

    def test_refuses_generation_when_training_failed(self) -> None:
        """Pipeline exits 1 when wrapper log lacks success markers."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrapper_log = root / "train_resilient.log"
            wrapper_log.write_text("RESILIENT_WRAPPER_DONE attempts=1\n")
            pipeline_log = root / "pipeline.log"
            env = dict(os.environ)
            env.update(ROOT=str(root), GEN_CMDS="echo arm1", LOG=str(pipeline_log))
            result = subprocess.run(
                [_bash(), str(REMOTE / "watch_and_generate.sh")],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            self.assertNotEqual(result.returncode, 0, "should exit 1 on training failure")
            log_text = pipeline_log.read_text()
            self.assertIn("training did not complete successfully", log_text)
            self.assertNotIn("PIPELINE_DONE", log_text)

    def test_proceeds_when_training_completed(self) -> None:
        """Pipeline runs generations when wrapper log has success marker."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrapper_log = root / "train_resilient.log"
            wrapper_log.write_text("TRAINING_COMPLETE\nRESILIENT_WRAPPER_DONE attempts=3\n")
            pipeline_log = root / "pipeline.log"
            out1 = root / "arm1.txt"
            env = dict(os.environ)
            env.update(ROOT=str(root), GEN_CMDS=f"echo arm1 > {out1}", LOG=str(pipeline_log))
            result = subprocess.run(
                [_bash(), str(REMOTE / "watch_and_generate.sh")],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = pipeline_log.read_text()
            self.assertIn("training verified successful", log_text)
            self.assertIn("PIPELINE_DONE", log_text)
            self.assertTrue(out1.exists())
            self.assertEqual(out1.read_text().strip(), "arm1")


@unittest.skipUnless(shutil.which("python3"), "python3 is unavailable")
class SupervisorDoneFilesTests(unittest.TestCase):
    """DONE_FILES gates ALL_DONE.

    A pipeline that dies between the last generation and its post-processing
    (metrics) must be relaunched, not declared finished because the arm
    files happen to be complete.
    """

    def _run_supervisor(self, root: Path, done_files_exist: bool):
        arm = root / "arm_val.jsonl"
        arm.write_text('{"id": "a"}\n' * 3, encoding="utf-8")
        (root / "train_resilient.log").write_text("TRAINING_COMPLETE\n", encoding="utf-8")
        metrics = root / "metrics.json"
        if done_files_exist:
            metrics.write_text("{}\n", encoding="utf-8")
        shim = root / "bin"
        shim.mkdir(exist_ok=True)
        # Fake pgrep (nothing alive) and setsid (records the relaunch instead
        # of spawning one) keep the test from touching real processes.
        (shim / "pgrep").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        (shim / "setsid").write_text(
            '#!/bin/sh\necho relaunch >> "$RELUNCH_LOG"\nexit 0\n', encoding="utf-8"
        )
        for path in shim.iterdir():
            path.chmod(0o755)
        env = dict(os.environ)
        env.update(
            ROOT=str(root),
            ARMS="arm_val.jsonl",
            ARM_LINES="3",
            DONE_FILES="metrics.json",
            PATH=f"{shim}:{env['PATH']}",
            RELUNCH_LOG=str(root / "relaunch.log"),
        )
        return subprocess.run(
            [_bash(), str(REMOTE / "supervisor.sh")],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

    def test_all_done_requires_done_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = self._run_supervisor(root, done_files_exist=False)
            self.assertNotIn("ALL_DONE", missing.stdout)
            self.assertIn("pipeline dead -> relaunch", missing.stdout)
            # The relaunch itself is backgrounded, so allow a moment for the
            # shim to record it.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not (root / "relaunch.log").exists():
                time.sleep(0.1)
            self.assertTrue(
                (root / "relaunch.log").exists(),
                "a finished-but-unprocessed run must trigger a pipeline relaunch",
            )
            (root / "metrics.json").write_text("{}\n", encoding="utf-8")
            complete = self._run_supervisor(root, done_files_exist=True)
            self.assertIn("ALL_DONE", complete.stdout)
            self.assertEqual(complete.returncode, 0, complete.stderr)

    def test_done_files_unset_keeps_old_behaviour(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arm = root / "arm_val.jsonl"
            arm.write_text('{"id": "a"}\n', encoding="utf-8")
            env = dict(os.environ)
            env.update(ROOT=str(root), ARMS="arm_val.jsonl", ARM_LINES="1")
            env.pop("DONE_FILES", None)
            result = subprocess.run(
                [_bash(), str(REMOTE / "supervisor.sh")],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            self.assertIn("ALL_DONE", result.stdout)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
