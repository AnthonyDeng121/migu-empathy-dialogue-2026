from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REMOTE = Path(__file__).resolve().parents[1] / "scripts" / "remote"
RUNNER = REMOTE / "run_workload.sh"
STAGES = REMOTE / "workload_stages.sh"
LAUNCHER = REMOTE / "launch_workload.sh"
SUPERVISOR = REMOTE / "workload_supervisor.sh"
UPLOADER = REMOTE / "upload_workload.sh"
WORKLOAD_SCRIPTS = (RUNNER, STAGES, LAUNCHER, SUPERVISOR)


def _bash() -> str:
    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - bash ships with Linux/macOS
        raise unittest.SkipTest("bash is unavailable")
    return bash


# A stand-in for the GPU interpreter: it answers the CUDA stack probes and the
# repo's generation/analysis entry points, and delegates everything else to the
# real interpreter running this test. No model, no GPU, no network.
FAKE_PYTHON = """#!/bin/sh
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "pip" ]; then
    # Dependency repair: record it instead of touching the real environment,
    # and remember the install so the re-check sees the module.
    echo "fake pip $*" >> "${FAKE_PIP_LOG:-/dev/null}"
    if [ "${FAKE_PIP_RC:-0}" = "0" ] && [ -n "${FAKE_PIP_LOG:-}" ]; then
        : > "${FAKE_PIP_LOG}.installed"
    fi
    exit "${FAKE_PIP_RC:-0}"
fi
if [ "${1:-}" = "-c" ]; then
    case "$2" in
        *cuda.is_available*)
            [ "${FAKE_CUDA_MISSING:-0}" = "1" ] && exit 1
            [ "${FAKE_TORCH_OK:-1}" = "1" ] || exit 1
            exit 0 ;;
        *torch.__version__*) echo "9.9.9+fake" ; exit 0 ;;
        *transformers.__version__*) echo "4.57.6" ; exit 0 ;;
        *peft.__version__*) echo "0.21.2" ; exit 0 ;;
        *accelerate.__version__*) echo "1.15.0" ; exit 0 ;;
        "import torch, transformers") [ "${FAKE_TORCH_OK:-1}" = "1" ] || exit 1 ; exit 0 ;;
        "import torch") [ "${FAKE_TORCH_OK:-1}" = "1" ] || exit 1 ; exit 0 ;;
        "import transformers")
            [ "${FAKE_TRANSFORMERS_MISSING:-0}" = "1" ] || exit 0
            [ -f "${FAKE_PIP_LOG:-/dev/null}.installed" ] && exit 0
            exit 1 ;;
        "import peft")
            [ "${FAKE_PEFT_MISSING:-0}" = "1" ] || exit 0
            [ -f "${FAKE_PIP_LOG:-/dev/null}.installed" ] && exit 0
            exit 1 ;;
        "import accelerate")
            [ "${FAKE_ACCELERATE_MISSING:-0}" = "1" ] || exit 0
            [ -f "${FAKE_PIP_LOG:-/dev/null}.installed" ] && exit 0
            exit 1 ;;
    esac
    exec "$REAL_PYTHON" "$@"
fi
case "${1:-}" in
    *run_response_inference.py)
        out="" ; report="" ; resume=""
        shift
        while [ $# -gt 0 ]; do
            case "$1" in
                --output) out="$2" ; shift 2 ;;
                --performance-report) report="$2" ; shift 2 ;;
                --resume-dir) resume="$2" ; shift 2 ;;
                *) shift ;;
            esac
        done
        [ -n "$out" ] || exit 9
        mkdir -p "$(dirname "$out")"
        i=0
        while [ "$i" -lt "${FAKE_GEN_LINES:-2}" ]; do
            printf '{"id": "s%s", "response_text": "ok"}\\n' "$i"
            i=$((i + 1))
        done > "$out"
        if [ -n "$report" ]; then mkdir -p "$(dirname "$report")" ; printf '{"ok": true}\\n' > "$report" ; fi
        if [ -n "$resume" ]; then mkdir -p "$resume" ; fi
        exit "${FAKE_GEN_RC:-0}" ;;
    *decide_conditional_fullval.py)
        out=""
        shift
        while [ $# -gt 0 ]; do
            case "$1" in
                --output) out="$2" ; shift 2 ;;
                *) shift ;;
            esac
        done
        [ -n "$out" ] || exit 9
        mkdir -p "$(dirname "$out")"
        if [ -n "${FAKE_DECISION_FILE:-}" ] && [ -f "$FAKE_DECISION_FILE" ]; then
            cp "$FAKE_DECISION_FILE" "$out"
        else
            printf '{"trigger": false, "chosen": null}\\n' > "$out"
        fi
        exit 0 ;;
    *.py)
        out=""
        shift
        while [ $# -gt 0 ]; do
            case "$1" in
                --output) out="$2" ; shift 2 ;;
                *) shift ;;
            esac
        done
        if [ -n "$out" ]; then
            mkdir -p "$(dirname "$out")"
            printf '{}\\n' > "$out"
        fi
        exit "${FAKE_METRIC_RC:-0}" ;;
    *)
        exec "$REAL_PYTHON" "$@" ;;
esac
"""


def _write_shim(root: Path, name: str, body: str) -> Path:
    shim = root / "bin"
    shim.mkdir(parents=True, exist_ok=True)
    path = shim / name
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_stages(root: Path, lines: list[str]) -> Path:
    """A stages file whose commands are plain shell, no model involved."""
    path = root / "fake_stages.sh"
    path.write_text(
        "#!/bin/bash\nworkload_stages() {\n    cat <<STAGES\n"
        + "\n".join(lines)
        + "\nSTAGES\n}\n",
        encoding="utf-8",
    )
    return path


def _stage_line(root: Path, name: str, output: str, lines: int = 2) -> str:
    args = " ".join(
        "'" + json.dumps({"id": f"{name}-{i}", "response_text": "ok"}) + "'"
        for i in range(lines)
    )
    return f"{name}|mkdir -p {root}/out && printf '%s\\n' {args} > {root}/{output}"


def _t4_emotion_decision() -> dict:
    """The decider product that triggers the conditional full-val arm."""
    return {
        "trigger": True,
        "chosen": {
            "arm": "t4_emotion",
            "adapter": "weights/t4_longweight_2ep",
            "condition_jsonl": "results/response_gen/c5_conditions_val_full.jsonl",
            "condition_fields": "emotion",
            "output": "out/t5_conditional_t4_emotion_val_newcard.jsonl",
            "resume_dir": "out/t5_conditional_t4_emotion_val_newcard_progress",
        },
    }


class WorkloadScriptSyntaxTests(unittest.TestCase):
    """The workload scripts ship in the same repo as the training ones."""

    def test_every_script_parses(self) -> None:
        for script in WORKLOAD_SCRIPTS:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [_bash(), "-n", str(script)], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_scripts_are_invoked_via_bash(self) -> None:
        for script in WORKLOAD_SCRIPTS:
            with self.subTest(script=script.name):
                self.assertTrue(
                    script.read_text(encoding="utf-8").startswith("#!"),
                    f"{script.name} lacks a shebang",
                )

    def test_runner_sources_the_env_file(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"$ROOT/workload.env"', text)
        self.assertIn("set -u", text)

    def test_cron_line_sources_the_env_file_first(self) -> None:
        """A bare cron environment leaves the supervisor variable-less."""
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(". $ENV_FILE && bash $SUPERVISOR", text)
        self.assertIn("workload_supervisor.sh", text)

    def test_supervisor_pgrep_pattern_cannot_match_itself(self) -> None:
        text = SUPERVISOR.read_text(encoding="utf-8")
        self.assertIn("run_workloa[d].sh", text)
        self.assertNotIn('pgrep -f "run_workload.sh"', text)
        self.assertNotIn("pgrep -f run_workload.sh", text)

    def test_launcher_picks_exactly_one_supervisor_transport(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("cron_mode=", text)
        self.assertIn("apt-get install -y cron", text)
        self.assertIn("while true; do bash", text)

    def test_launcher_probes_the_autodl_conda_paths_in_order(self) -> None:
        """AutoDL keeps conda in /root/miniconda3, off the non-interactive PATH."""
        text = LAUNCHER.read_text(encoding="utf-8")
        conda = text.index('"/opt/conda/bin/python"')
        miniconda = text.index('"/root/miniconda3/bin/python"')
        system = text.index('"python3"')
        self.assertLess(conda, miniconda)
        self.assertLess(miniconda, system)
        self.assertIn("torch.cuda.is_available()", text)

    def test_dependency_repair_uses_the_pinned_versions(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("pip install --no-input", text)
        self.assertIn("transformers==5.17.0 peft==0.21.2 accelerate==1.15.0", text)
        self.assertIn("for module in transformers peft accelerate", text)
        # torch ships with the image (2.5.1, same pin) and is only probed.
        self.assertIn("-c 'import torch'", text)
        self.assertNotIn("pip install torch", text)

    def test_runner_and_supervisor_self_default_their_root(self) -> None:
        """Both must find ROOT from their own location before any guard.

        The launcher starts `bash run_workload.sh` without exporting ROOT and
        the cron-less user-space loop is the only supervisor transport, so a
        guard placed before the self-default kills the whole run (this exact
        ordering bug shipped once).
        """
        for script in (RUNNER, SUPERVISOR):
            text = script.read_text(encoding="utf-8")
            with self.subTest(script=script.name):
                self.assertIn('ROOT="${ROOT:-', text)
                self.assertIn("$SELF_DIR/../..", text)
                # the self-default must precede the hard guard
                self.assertLess(text.index('ROOT="${ROOT:-'), text.index(': "${ROOT:?'))

    def test_user_space_loop_sources_the_env_file(self) -> None:
        """Without cron the detached loop is the only supervisor transport."""
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('. "$1" && bash "$2"', text)

    def test_base_pull_disables_xet_for_the_mirror(self) -> None:
        """hf-mirror does not serve the xet CAS protocol (401 on reconstruction)."""
        text = UPLOADER.read_text(encoding="utf-8")
        self.assertIn("export HF_HUB_DISABLE_XET=1", text)


class WorkloadStagesTests(unittest.TestCase):
    """The default stage list must match the cycle-2 arm matrix."""

    EXPECTED_NAMES = [
        "01_t4_off_selection",
        "02_d4v2_off_selection",
        "03_t4_val",
        "04_d4v2_val",
        "05_t4_emotion_selection",
        "06_t4_eprofile_selection",
        "07_d4v2_emotion_selection",
        "08_d4v2_eprofile_selection",
        "09_m1_t4_arms_selection",
        "10_m2_t4_val_errors",
        "11_m3a_d4v2_emotion_vs_off",
        "12_m3b_d4v2_eprofile_vs_off",
        "13_m3c_t4_emotion_vs_off",
        "14_m3d_t4_eprofile_vs_off",
        "15_t5_decide",
    ]
    GENERATION = EXPECTED_NAMES[:8]

    def _stages(self) -> list[tuple[str, str]]:
        env = dict(os.environ)
        env.update(
            PY="python3",
            GPU="0",
            MODEL_DIR="weights/hf/Qwen2.5-1.5B-Instruct",
            T4_ADAPTER="weights/t4_longweight_2ep",
            D4V2_ADAPTER="weights/response_sft_d4_2ep",
            SEL_INPUT="results/response_gen/t2_selection_input.jsonl",
            VAL_INPUT="data/processed/b1_full_v1/response_val.jsonl",
            C5_CONDITIONS="results/response_gen/c5_conditions_val_full.jsonl",
            BOTH_COND_SEL="results/response_gen/t5_conditions_selection_both.jsonl",
            BOTH_COND_VAL="results/response_gen/t5_conditions_val_both.jsonl",
            OUT_DIR="results/response_gen",
        )
        result = subprocess.run(
            [_bash(), "-c", f'. "{STAGES}" && workload_stages'],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            name, _, command = line.partition("|")
            rows.append((name, command))
        return rows

    def test_stage_names_and_order(self) -> None:
        rows = self._stages()
        self.assertEqual([name for name, _ in rows], self.EXPECTED_NAMES)

    def test_generation_stages_pin_device_and_length(self) -> None:
        rows = dict(self._stages())
        for name in self.GENERATION:
            command = rows[name]
            with self.subTest(stage=name):
                self.assertIn("--device cuda", command)
                self.assertIn("--max-new-tokens 256", command)
                self.assertIn("--performance-report", command)
                self.assertIn("--resume-dir", command)
                self.assertIn("--model weights/hf/Qwen2.5-1.5B-Instruct", command)

    def test_selection_arms_use_the_selection_input(self) -> None:
        rows = dict(self._stages())
        for name in ("01_t4_off_selection", "05_t4_emotion_selection", "07_d4v2_emotion_selection"):
            self.assertIn(
                "--input results/response_gen/t2_selection_input.jsonl", rows[name]
            )
        for name in ("03_t4_val", "04_d4v2_val"):
            self.assertIn(
                "--input data/processed/b1_full_v1/response_val.jsonl", rows[name]
            )

    def test_condition_flags_match_the_condition_files(self) -> None:
        rows = dict(self._stages())
        self.assertIn("--condition-fields emotion", rows["05_t4_emotion_selection"])
        self.assertIn("--condition-fields both", rows["06_t4_eprofile_selection"])
        self.assertIn(
            "--condition-jsonl results/response_gen/c5_conditions_val_full.jsonl",
            rows["05_t4_emotion_selection"],
        )
        self.assertIn(
            "--condition-jsonl results/response_gen/t5_conditions_selection_both.jsonl",
            rows["06_t4_eprofile_selection"],
        )

    def test_metric_stages_are_cpu_only(self) -> None:
        rows = dict(self._stages())
        for name in self.EXPECTED_NAMES[8:]:
            with self.subTest(stage=name):
                self.assertNotIn("--device cuda", rows[name])
                self.assertIn("scripts/", rows[name])

    def test_val_outputs_stay_in_sync_with_the_runner(self) -> None:
        """The conditional analysis hardcodes these two paths."""
        runner = RUNNER.read_text(encoding="utf-8")
        rows = dict(self._stages())
        for name in ("03_t4_val", "04_d4v2_val"):
            command = rows[name]
            output = command.split("--output ", 1)[1].split(" ", 1)[0]
            with self.subTest(stage=name):
                self.assertIn(output.rsplit("/", 1)[-1], runner)

    def test_decider_stage_wires_all_four_comparisons(self) -> None:
        rows = dict(self._stages())
        command = rows["15_t5_decide"]
        self.assertEqual(command.count("--comparison "), 4)
        self.assertIn(
            "--output results/response_gen/t5_conditional_decision.json", command
        )
        for arm in ("d4v2_emotion", "d4v2_eprofile", "t4_emotion", "t4_eprofile"):
            self.assertIn(f"t5_{arm}_vs_off_newcard.json", command)

    def test_no_command_contains_the_separator(self) -> None:
        for name, command in self._stages():
            with self.subTest(stage=name):
                self.assertNotIn("|", command)

    @staticmethod
    def _launcher_list(text: str, name: str) -> list[str]:
        """Values of a `NAME="..."` assignment in the launcher, continuations
        joined and $OUT_DIR expanded the way the launcher itself would."""
        marker = f'{name}="'
        start = text.index(marker) + len(marker)
        end = text.index('"', start)
        block = text[start:end].replace("\\\n", " ").replace("$OUT_DIR", "results/response_gen")
        return [token for token in block.split() if token]

    def test_launcher_audit_lists_match_the_stage_outputs(self) -> None:
        """Every audited path must be produced by some stage.

        A path typo here passes all stages and still fails the final audit,
        which would leave a complete run without WORKLOAD_DONE (this check
        exists because reports/ vs $OUT_DIR for the val error analysis was
        exactly such a typo).
        """
        launcher = LAUNCHER.read_text(encoding="utf-8")
        rows = dict(self._stages())
        outputs = set()
        for command in rows.values():
            for token in command.split("--output ", 1)[1:]:
                outputs.add(token.split(" ", 1)[0])
        for entry in self._launcher_list(launcher, "ARMS_SPEC"):
            path, _, want = entry.rpartition(":")
            with self.subTest(arms_entry=entry):
                self.assertIn(path, outputs)
                self.assertIn(want, ("160", "2354"))
        for path in self._launcher_list(launcher, "DONE_FILES"):
            with self.subTest(done_entry=path):
                self.assertIn(path, outputs)


@unittest.skipUnless(shutil.which("sha256sum"), "sha256sum is unavailable")
class RunWorkloadTests(unittest.TestCase):
    """Preflight, marker idempotency and the artifact audit, on a fake root."""

    def _build_root(self, root: Path, *, stage_lines: list[str], env_extra: dict | None = None) -> Path:
        model_dir = root / "weights" / "hf" / "Qwen2.5-1.5B-Instruct"
        model_dir.mkdir(parents=True)
        model = model_dir / "model.safetensors"
        model.write_bytes(b"not-a-real-3gb-base-model\n")
        model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
        for adapter in ("weights/t4_longweight_2ep", "weights/response_sft_d4_2ep"):
            (root / adapter).mkdir(parents=True)
            (root / adapter / "adapter_model.safetensors").write_bytes(b"adapter-bytes\n")
        (root / "base_ready").write_text(
            f"base_sha256={model_sha}\nbytes={model.stat().st_size}\n"
            f"model={model}\nverified_at=2026-10-29T00:00:00\n",
            encoding="utf-8",
        )
        # The upload manifest covers a couple of uploaded files; its hashes are
        # real so `sha256sum -c` passes on the fake root.
        uploaded = root / "scripts" / "uploaded.py"
        uploaded.parent.mkdir(parents=True, exist_ok=True)
        uploaded.write_text("# uploaded\n", encoding="utf-8")
        manifest_rows = [
            (uploaded, uploaded.relative_to(root).as_posix()),
            (root / "weights" / "t4_longweight_2ep" / "adapter_model.safetensors",
             "weights/t4_longweight_2ep/adapter_model.safetensors"),
        ]
        manifest = "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel}\n" for path, rel in manifest_rows
        )
        (root / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
        for rel in (
            "results/response_gen/t2_selection_input.jsonl",
            "data/processed/b1_full_v1/response_val.jsonl",
        ):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(json.dumps({"id": f"v{i}", "response_text": "gold"}) + "\n" for i in range(2)),
                encoding="utf-8",
            )
        (root / "out" / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
        (root / "out" / "metrics.json").write_text("{}\n", encoding="utf-8")
        # The decider's default product (what stage 15 writes when no arm
        # clears the bar); the conditional tests overwrite it.
        (root / "out" / "t5_conditional_decision.json").write_text(
            json.dumps({"trigger": False, "chosen": None}), encoding="utf-8"
        )

        python_shim = _write_shim(root, "python", FAKE_PYTHON)
        _write_shim(root, "nvidia-smi", "#!/bin/sh\necho 'FakeGPU (test shim)'\nexit 0\n")
        env = {
            "ROOT": str(root),
            "PY": str(python_shim),
            "GPU": "0",
            "MODEL_DIR": "weights/hf/Qwen2.5-1.5B-Instruct",
            "T4_ADAPTER": "weights/t4_longweight_2ep",
            "D4V2_ADAPTER": "weights/response_sft_d4_2ep",
            "SEL_INPUT": "results/response_gen/t2_selection_input.jsonl",
            "VAL_INPUT": "data/processed/b1_full_v1/response_val.jsonl",
            "C5_CONDITIONS": "results/response_gen/c5_conditions_val_full.jsonl",
            "BOTH_COND_SEL": "results/response_gen/t5_conditions_selection_both.jsonl",
            "BOTH_COND_VAL": "results/response_gen/t5_conditions_val_both.jsonl",
            "OUT_DIR": "out",
            "ARMS_SPEC": "out/alpha.jsonl:2 out/beta.jsonl:2",
            "DONE_FILES": "out/metrics.json",
            "STAGES_FILE": str(_fake_stages(root, stage_lines)),
            "BASE_WAIT_MINUTES": "0",
            "BASE_SHA256_EXPECTED": model_sha,
            "BASE_MODEL_SIZE": str(model.stat().st_size),
            "MAX_ATTEMPTS": "6",
        }
        env.update(env_extra or {})
        lines = [f'export {key}="{value}"' for key, value in env.items()]
        (root / "workload.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return root

    def _run(self, root: Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PATH"] = f"{root / 'bin'}{os.pathsep}{env['PATH']}"
        # ROOT is the only value passed in: everything else must come from
        # workload.env, which is exactly what cron and the supervisor do.
        env["ROOT"] = str(root)
        env["REAL_PYTHON"] = sys.executable
        env.update(env_extra or {})
        return subprocess.run(
            [_bash(), str(RUNNER)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(root),
            timeout=300,
        )

    def _status(self, root: Path) -> dict:
        return json.loads((root / "workload_status.json").read_text(encoding="utf-8"))

    def _preflight(self, root: Path) -> dict:
        return json.loads((root / "preflight.json").read_text(encoding="utf-8"))

    def _markers(self, root: Path) -> list[str]:
        return sorted(p.name for p in (root / "stages").glob("*.done"))

    def test_runner_finds_its_root_without_the_environment(self) -> None:
        """The launcher starts `bash run_workload.sh` without exporting ROOT.

        Only the script's own location may decide ROOT here, and the cwd is
        deliberately foreign so a stale environment cannot mask a regression.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_root(
                root,
                stage_lines=[_stage_line(root, "01_alpha", "out/alpha.jsonl")],
                env_extra={"ARMS_SPEC": "out/alpha.jsonl:2"},
            )
            local_runner = root / "scripts" / "remote" / "run_workload.sh"
            local_runner.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(RUNNER, local_runner)
            env = dict(os.environ)
            env["PATH"] = f"{root / 'bin'}{os.pathsep}{env['PATH']}"
            env["REAL_PYTHON"] = sys.executable
            env.pop("ROOT", None)
            result = subprocess.run(
                [_bash(), str(local_runner)],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(root.parent),
                timeout=300,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("set ROOT", result.stdout + result.stderr)
            self.assertEqual(
                self._markers(root), ["01_alpha.done", "16_t5_conditional_val.done"]
            )
            self.assertEqual(self._status(root)["outcome"], "ok")

    def test_stages_are_idempotent_across_relaunches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[
                    _stage_line(root, "01_alpha", "out/alpha.jsonl"),
                    _stage_line(root, "02_beta", "out/beta.jsonl"),
                ],
            )
            first = self._run(root)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertEqual(
                self._markers(root),
                ["01_alpha.done", "02_beta.done", "16_t5_conditional_val.done"],
            )
            self.assertTrue((root / "WORKLOAD_DONE").is_file())
            status = self._status(root)
            self.assertEqual(status["outcome"], "ok")
            self.assertEqual(
                status["stages"],
                {
                    "01_alpha": "done",
                    "02_beta": "done",
                    "16_t5_conditional_val": "skipped",
                },
            )
            self.assertEqual(status["arms"]["out/alpha.jsonl"]["lines"], 2)
            self.assertIn("WORKLOAD_DONE", (root / "workload.log").read_text(encoding="utf-8"))

            second = self._run(root)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            status = self._status(root)
            self.assertEqual(
                status["stages"],
                {
                    "01_alpha": "skipped",
                    "02_beta": "skipped",
                    "16_t5_conditional_val": "skipped",
                },
            )
            # Nothing was regenerated: the stage commands never ran again.
            self.assertNotIn("stage 01_alpha started", (root / "workload.log").read_text(encoding="utf-8").split("workload start")[-1])

    def test_failed_stage_leaves_no_marker_and_records_the_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[
                    _stage_line(root, "01_alpha", "out/alpha.jsonl"),
                    "02_beta|echo boom >&2; exit 1",
                    _stage_line(root, "03_gamma", "out/gamma.jsonl"),
                ],
            )
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((root / "stages" / "01_alpha.done").is_file())
            self.assertFalse((root / "stages" / "02_beta.done").exists())
            self.assertFalse((root / "stages" / "03_gamma.done").exists())
            self.assertFalse((root / "WORKLOAD_DONE").exists())
            status = self._status(root)
            self.assertEqual(status["outcome"], "stage_failed")
            self.assertEqual(status["failed_stage"], "02_beta")
            self.assertEqual(status["stages"]["02_beta"], "failed")
            self.assertIn("boom", (root / "workload.log").read_text(encoding="utf-8"))

    def test_broken_manifest_aborts_before_any_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[
                    _stage_line(root, "01_alpha", "out/alpha.jsonl"),
                    _stage_line(root, "02_beta", "out/beta.jsonl"),
                ],
            )
            (root / "scripts" / "uploaded.py").write_text("# tampered\n", encoding="utf-8")
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list((root / "stages").glob("*.done")), [])
            preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(preflight["outcome"], "fail")
            self.assertIn("manifest", preflight["reason"])
            self.assertEqual(self._status(root)["outcome"], "preflight_failed")

    def test_missing_base_model_with_zero_wait_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[_stage_line(root, "01_alpha", "out/alpha.jsonl")],
            )
            (root / "base_ready").unlink()
            started = time.monotonic()
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertLess(time.monotonic() - started, 60, "must not block on the base wait")
            self.assertEqual(list((root / "stages").glob("*.done")), [])
            preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
            self.assertIn("base model not ready", preflight["reason"])

    def test_base_ready_with_a_wrong_hash_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[_stage_line(root, "01_alpha", "out/alpha.jsonl")],
            )
            (root / "base_ready").write_text("base_sha256=" + "0" * 64 + "\n", encoding="utf-8")
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list((root / "stages").glob("*.done")), [])

    def test_unexpected_base_hash_aborts_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[_stage_line(root, "01_alpha", "out/alpha.jsonl")],
                env_extra={"BASE_SHA256_EXPECTED": "f" * 64},
            )
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list((root / "stages").glob("*.done")), [])
            preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
            self.assertIn("is not the expected", preflight["reason"])

    def test_broken_python_stack_aborts_before_any_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[_stage_line(root, "01_alpha", "out/alpha.jsonl")],
            )
            result = self._run(root, env_extra={"FAKE_TORCH_OK": "0"})
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list((root / "stages").glob("*.done")), [])
            preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
            self.assertIn("cannot import torch", preflight["reason"])

    def test_missing_cuda_aborts_before_any_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[_stage_line(root, "01_alpha", "out/alpha.jsonl")],
            )
            result = self._run(root, env_extra={"FAKE_CUDA_MISSING": "1"})
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list((root / "stages").glob("*.done")), [])
            preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
            self.assertIn("torch.cuda.is_available()", preflight["reason"])

    def test_missing_peft_is_repaired_with_the_pinned_versions(self) -> None:
        extra = {
            "FAKE_PEFT_MISSING": "1",
            "FAKE_PIP_LOG": "pip.log",
            "ARMS_SPEC": "out/alpha.jsonl:2",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[_stage_line(root, "01_alpha", "out/alpha.jsonl")],
                env_extra=extra,
            )
            result = self._run(root, env_extra=extra)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(preflight["deps_repaired"], ["peft"])
            self.assertEqual(preflight["peft"], "0.21.2")
            self.assertEqual(preflight["accelerate"], "1.15.0")
            self.assertEqual(
                preflight["pip_packages"], "transformers==5.17.0 peft==0.21.2 accelerate==1.15.0"
            )
            self.assertTrue((root / "workload_deps_install.log").is_file())
            pip_log = (root / "pip.log").read_text(encoding="utf-8")
            self.assertIn("transformers==5.17.0 peft==0.21.2 accelerate==1.15.0", pip_log)
            # The repair happens inside preflight: the stages still ran.
            self.assertEqual(
                self._markers(root), ["01_alpha.done", "16_t5_conditional_val.done"]
            )

    def test_missing_transformers_is_repaired_with_the_pinned_versions(self) -> None:
        """The SeeTaCloud image ships torch 2.5.1 but no transformers/peft/accelerate."""
        extra = {
            "FAKE_TRANSFORMERS_MISSING": "1",
            "FAKE_PIP_LOG": "pip.log",
            "ARMS_SPEC": "out/alpha.jsonl:2",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[_stage_line(root, "01_alpha", "out/alpha.jsonl")],
                env_extra=extra,
            )
            result = self._run(root, env_extra=extra)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(preflight["deps_repaired"], ["transformers"])
            self.assertEqual(preflight["transformers"], "4.57.6")
            self.assertEqual(preflight["torch"], "9.9.9+fake")

    def test_missing_peft_and_accelerate_are_both_recorded(self) -> None:
        extra = {
            "FAKE_PEFT_MISSING": "1",
            "FAKE_ACCELERATE_MISSING": "1",
            "FAKE_PIP_LOG": "pip.log",
            "ARMS_SPEC": "out/alpha.jsonl:2",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[_stage_line(root, "01_alpha", "out/alpha.jsonl")],
                env_extra=extra,
            )
            result = self._run(root, env_extra=extra)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(preflight["deps_repaired"], ["peft", "accelerate"])

    def test_failed_dependency_repair_aborts_preflight(self) -> None:
        extra = {
            "FAKE_PEFT_MISSING": "1",
            "FAKE_PIP_RC": "1",
            "FAKE_PIP_LOG": "pip.log",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[_stage_line(root, "01_alpha", "out/alpha.jsonl")],
                env_extra=extra,
            )
            result = self._run(root, env_extra=extra)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list((root / "stages").glob("*.done")), [])
            preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
            self.assertIn("pip install", preflight["reason"])

    def test_no_repair_is_recorded_when_the_stack_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[_stage_line(root, "01_alpha", "out/alpha.jsonl")],
                env_extra={"ARMS_SPEC": "out/alpha.jsonl:2"},
            )
            result = self._run(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(preflight["deps_repaired"], [])

    def test_adapter_hash_mismatch_aborts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[_stage_line(root, "01_alpha", "out/alpha.jsonl")],
                env_extra={
                    "ADAPTER_SHA256S": "weights/t4_longweight_2ep:" + "a" * 64
                    + " weights/response_sft_d4_2ep:"
                },
            )
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list((root / "stages").glob("*.done")), [])
            preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
            self.assertIn("adapter sha256 mismatch", preflight["reason"])

    def test_missing_adapter_weights_abort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[_stage_line(root, "01_alpha", "out/alpha.jsonl")],
            )
            (root / "weights" / "response_sft_d4_2ep" / "adapter_model.safetensors").unlink()
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0)
            preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
            self.assertIn("adapter weights missing", preflight["reason"])

    def test_short_arm_is_not_declared_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[
                    _stage_line(root, "01_alpha", "out/alpha.jsonl", lines=1),
                    _stage_line(root, "02_beta", "out/beta.jsonl"),
                ],
            )
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "WORKLOAD_DONE").exists())
            status = self._status(root)
            self.assertEqual(status["outcome"], "arms_incomplete")
            self.assertFalse(status["arms"]["out/alpha.jsonl"]["ok"])
            self.assertEqual(status["arms"]["out/alpha.jsonl"]["expected"], 2)

    def test_conditional_arm_runs_when_the_decider_triggers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[
                    _stage_line(root, "01_alpha", "out/alpha.jsonl"),
                    _stage_line(root, "02_beta", "out/beta.jsonl"),
                ],
            )
            decision = {
                "trigger": True,
                "chosen": {
                    "arm": "t4_emotion",
                    "adapter": "weights/t4_longweight_2ep",
                    "condition_jsonl": "results/response_gen/c5_conditions_val_full.jsonl",
                    "condition_fields": "emotion",
                    "output": "out/t5_conditional_t4_emotion_val_newcard.jsonl",
                    "resume_dir": "out/t5_conditional_t4_emotion_val_newcard_progress",
                },
            }
            (root / "out" / "t5_conditional_decision.json").write_text(
                json.dumps(decision), encoding="utf-8"
            )
            result = self._run(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            marker = root / "stages" / "16_t5_conditional_val.done"
            self.assertTrue(marker.is_file())
            output = root / "out" / "t5_conditional_t4_emotion_val_newcard.jsonl"
            self.assertTrue(output.is_file())
            self.assertTrue(
                (root / "reports" / "t5_conditional_val_error_analysis_newcard.json").is_file()
            )
            self.assertTrue(
                (root / "out" / "t5_conditional_t4_emotion_val_newcard_progress").is_dir()
            )
            status = self._status(root)
            self.assertEqual(status["stages"]["16_t5_conditional_val"], "done")
            self.assertTrue(status["arms"]["out/t5_conditional_t4_emotion_val_newcard.jsonl"]["ok"])
            self.assertTrue(status["done_files"]["reports/t5_conditional_val_error_analysis_newcard.json"])
            log = (root / "workload.log").read_text(encoding="utf-8")
            self.assertIn("--condition-fields emotion", log)
            self.assertIn("--arm conditional=out/t5_conditional_t4_emotion_val_newcard.jsonl", log)
            effective = (root / "workload_arms_effective.txt").read_text(encoding="utf-8")
            self.assertIn("out/t5_conditional_t4_emotion_val_newcard.jsonl:2", effective)

    def test_conditional_arm_is_skipped_without_a_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[
                    _stage_line(root, "01_alpha", "out/alpha.jsonl"),
                    _stage_line(root, "02_beta", "out/beta.jsonl"),
                ],
            )
            result = self._run(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / "stages" / "16_t5_conditional_val.done").is_file())
            self.assertFalse(
                (root / "out" / "t5_conditional_t4_emotion_val_newcard.jsonl").exists()
            )
            log = (root / "workload.log").read_text(encoding="utf-8")
            self.assertIn("decider chose no arm", log)
            self.assertEqual(self._status(root)["stages"]["16_t5_conditional_val"], "skipped")

    def test_conditional_marker_makes_the_arm_restartable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[
                    _stage_line(root, "01_alpha", "out/alpha.jsonl"),
                    _stage_line(root, "02_beta", "out/beta.jsonl"),
                ],
            )
            (root / "out" / "t5_conditional_decision.json").write_text(
                json.dumps(_t4_emotion_decision()), encoding="utf-8"
            )
            self.assertEqual(self._run(root).returncode, 0)
            # The marker freezes the arm that actually ran: a relaunch audits
            # that record, never a decision rewritten after the fact.
            marker = root / "stages" / "16_t5_conditional_val.done"
            self.assertEqual(
                marker.read_text(encoding="utf-8").splitlines(),
                [
                    "out/t5_conditional_t4_emotion_val_newcard.jsonl:2",
                    "reports/t5_conditional_val_error_analysis_newcard.json",
                ],
            )
            # A new decision must not resurrect an arm that already ran.
            (root / "out" / "t5_conditional_decision.json").write_text(
                json.dumps(
                    {
                        "trigger": True,
                        "chosen": {
                            "arm": "d4v2_emotion",
                            "adapter": "weights/response_sft_d4_2ep",
                            "condition_jsonl": "results/response_gen/c5_conditions_val_full.jsonl",
                            "condition_fields": "emotion",
                            "output": "out/t5_conditional_d4v2_emotion_val_newcard.jsonl",
                            "resume_dir": "out/t5_conditional_d4v2_emotion_val_newcard_progress",
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(self._run(root).returncode, 0)
            log = (root / "workload.log").read_text(encoding="utf-8")
            self.assertNotIn("d4v2_emotion ->", log)
            status = self._status(root)
            self.assertEqual(status["stages"]["16_t5_conditional_val"], "skipped")
            self.assertTrue(
                status["arms"]["out/t5_conditional_t4_emotion_val_newcard.jsonl"]["ok"]
            )
            self.assertTrue(
                status["done_files"][
                    "reports/t5_conditional_val_error_analysis_newcard.json"
                ]
            )

    def test_conditional_marker_with_missing_artifacts_refuses_completion(self) -> None:
        """A marker claiming a finished conditional arm must never launder
        missing or short outputs into WORKLOAD_DONE on a relaunch."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[
                    _stage_line(root, "01_alpha", "out/alpha.jsonl"),
                    _stage_line(root, "02_beta", "out/beta.jsonl"),
                ],
            )
            # Static stages and the frozen decision are all in place; only the
            # conditional arm is under test.
            (root / "stages").mkdir(parents=True, exist_ok=True)
            for name in ("01_alpha", "02_beta"):
                (root / "stages" / f"{name}.done").write_text("", encoding="utf-8")
            for name in ("alpha", "beta"):
                (root / "out" / f"{name}.jsonl").write_text(
                    '{"id": "a"}\n{"id": "b"}\n', encoding="utf-8"
                )
            (root / "out" / "t5_conditional_decision.json").write_text(
                json.dumps(_t4_emotion_decision()), encoding="utf-8"
            )
            (root / "stages" / "16_t5_conditional_val.done").write_text(
                "out/t5_conditional_t4_emotion_val_newcard.jsonl:2\n"
                "reports/t5_conditional_val_error_analysis_newcard.json\n",
                encoding="utf-8",
            )
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "WORKLOAD_DONE").exists())
            status = self._status(root)
            self.assertEqual(status["outcome"], "arms_incomplete")
            self.assertIn("conditional arm output missing", status["reason"])
            # A short output is refused by the audit, not waved through.
            (root / "out" / "t5_conditional_t4_emotion_val_newcard.jsonl").write_text(
                '{"id": "s0"}\n', encoding="utf-8"
            )
            (root / "reports").mkdir(parents=True, exist_ok=True)
            (root / "reports" / "t5_conditional_val_error_analysis_newcard.json").write_text(
                "{}\n", encoding="utf-8"
            )
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "WORKLOAD_DONE").exists())
            status = self._status(root)
            self.assertEqual(status["outcome"], "arms_incomplete")
            self.assertFalse(
                status["arms"]["out/t5_conditional_t4_emotion_val_newcard.jsonl"]["ok"]
            )

    def test_conditional_marker_is_written_only_after_the_audit(self) -> None:
        """Generation returning 0 with too few rows must not leave a marker:
        the relaunch has to re-run the arm, and only a verified one may
        complete the workload."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[
                    _stage_line(root, "01_alpha", "out/alpha.jsonl"),
                    _stage_line(root, "02_beta", "out/beta.jsonl"),
                ],
            )
            # Static stages and the frozen decision are all in place; only the
            # conditional arm is under test.
            (root / "stages").mkdir(parents=True, exist_ok=True)
            for name in ("01_alpha", "02_beta"):
                (root / "stages" / f"{name}.done").write_text("", encoding="utf-8")
            for name in ("alpha", "beta"):
                (root / "out" / f"{name}.jsonl").write_text(
                    '{"id": "a"}\n{"id": "b"}\n', encoding="utf-8"
                )
            (root / "out" / "t5_conditional_decision.json").write_text(
                json.dumps(_t4_emotion_decision()), encoding="utf-8"
            )
            short = self._run(root, env_extra={"FAKE_GEN_LINES": "1"})
            self.assertNotEqual(short.returncode, 0)
            self.assertFalse(
                (root / "stages" / "16_t5_conditional_val.done").exists()
            )
            self.assertFalse((root / "WORKLOAD_DONE").exists())
            self.assertEqual(self._status(root)["outcome"], "arms_incomplete")
            # The relaunch re-runs the arm and only then completes.
            again = self._run(root)
            self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
            self.assertTrue((root / "WORKLOAD_DONE").is_file())
            marker = root / "stages" / "16_t5_conditional_val.done"
            self.assertEqual(
                marker.read_text(encoding="utf-8").splitlines()[0],
                "out/t5_conditional_t4_emotion_val_newcard.jsonl:2",
            )
            log = (root / "workload.log").read_text(encoding="utf-8")
            self.assertEqual(log.count("16_t5_conditional_val started"), 2)

    def test_legacy_empty_marker_is_re_audited_from_the_decision(self) -> None:
        """An empty marker is what the previous runner left behind: it proves
        the stage ran, not that the arm is complete, so a relaunch re-derives
        the spec from the frozen decision and audits the row count -- a short
        output must never complete the workload."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root = self._build_root(
                root,
                stage_lines=[
                    _stage_line(root, "01_alpha", "out/alpha.jsonl"),
                    _stage_line(root, "02_beta", "out/beta.jsonl"),
                ],
            )
            # Static stages and the frozen decision are all in place; only the
            # conditional arm is under test.
            (root / "stages").mkdir(parents=True, exist_ok=True)
            for name in ("01_alpha", "02_beta"):
                (root / "stages" / f"{name}.done").write_text("", encoding="utf-8")
            for name in ("alpha", "beta"):
                (root / "out" / f"{name}.jsonl").write_text(
                    '{"id": "a"}\n{"id": "b"}\n', encoding="utf-8"
                )
            (root / "out" / "t5_conditional_decision.json").write_text(
                json.dumps(_t4_emotion_decision()), encoding="utf-8"
            )
            # The old runner's marker: present but empty, no audit lists.
            (root / "stages" / "16_t5_conditional_val.done").write_text(
                "", encoding="utf-8"
            )
            (root / "reports").mkdir(parents=True, exist_ok=True)
            (
                root / "reports" / "t5_conditional_val_error_analysis_newcard.json"
            ).write_text("{}\n", encoding="utf-8")
            cond_out = root / "out" / "t5_conditional_t4_emotion_val_newcard.jsonl"
            # 1 of the 2 val rows: non-empty, but the audit must refuse.
            cond_out.write_text('{"id": "s0"}\n', encoding="utf-8")
            result = self._run(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "WORKLOAD_DONE").exists())
            status = self._status(root)
            self.assertEqual(status["outcome"], "arms_incomplete")
            self.assertFalse(
                status["arms"]["out/t5_conditional_t4_emotion_val_newcard.jsonl"]["ok"]
            )
            # A complete arm passes the same audit and finishes; the arm
            # itself is not re-generated (the marker still skips it).
            cond_out.write_text('{"id": "s0"}\n{"id": "s1"}\n', encoding="utf-8")
            again = self._run(root)
            self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
            self.assertTrue((root / "WORKLOAD_DONE").is_file())
            log = (root / "workload.log").read_text(encoding="utf-8")
            self.assertNotIn("16_t5_conditional_val started", log)


class SupervisorWorkloadTests(unittest.TestCase):
    """ALL_DONE gating, relaunch and the retry budget."""

    def _shims(self, root: Path) -> None:
        # Fake pgrep (nothing alive) and setsid (records the relaunch instead
        # of spawning one) keep the test from touching real processes.
        _write_shim(root, "pgrep", "#!/bin/sh\nexit 1\n")
        _write_shim(root, "setsid", '#!/bin/sh\necho relaunch >> "$RELUNCH_LOG"\nexit 0\n')

    def _run_supervisor(self, root: Path, env_extra: dict | None = None):
        self._shims(root)
        env = dict(os.environ)
        env["PATH"] = f"{root / 'bin'}{os.pathsep}{env['PATH']}"
        env.update(
            ROOT=str(root),
            ARMS_SPEC="out/alpha.jsonl:2",
            DONE_FILES="out/metrics.json",
            RELUNCH_LOG=str(root / "relaunch.log"),
            MAX_ATTEMPTS="2",
        )
        env.update(env_extra or {})
        return subprocess.run(
            [_bash(), str(SUPERVISOR)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(root),
            timeout=120,
        )

    def _complete_arms(self, root: Path) -> None:
        (root / "out").mkdir(parents=True, exist_ok=True)
        (root / "out" / "alpha.jsonl").write_text(
            '{"id": "a"}\n{"id": "b"}\n', encoding="utf-8"
        )
        (root / "out" / "metrics.json").write_text("{}\n", encoding="utf-8")

    def test_all_done_when_arms_and_done_files_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete_arms(root)
            result = self._run_supervisor(root)
            self.assertIn("ALL_DONE", result.stdout)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "relaunch.log").exists())

    def test_supervisor_finds_its_root_without_the_environment(self) -> None:
        """The cron-less loop sources the env file, but the supervisor must not
        depend on that transport: a bare invocation from a foreign cwd
        self-defaults ROOT and reads the audit lists from workload.env."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / "scripts" / "remote" / "workload_supervisor.sh"
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SUPERVISOR, local)
            (root / "workload.env").write_text(
                f'export ROOT="{root}"\n'
                'export ARMS_SPEC="out/alpha.jsonl:2"\n'
                'export DONE_FILES="out/metrics.json"\n'
                'export MAX_ATTEMPTS="2"\n',
                encoding="utf-8",
            )
            self._complete_arms(root)
            self._shims(root)
            env = dict(os.environ)
            env["PATH"] = f"{root / 'bin'}{os.pathsep}{env['PATH']}"
            env["RELUNCH_LOG"] = str(root / "relaunch.log")
            env.pop("ROOT", None)
            env.pop("ARMS_SPEC", None)
            result = subprocess.run(
                [_bash(), str(local)],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(root.parent),
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("set ROOT", result.stdout + result.stderr)
            self.assertIn("ALL_DONE", result.stdout)
            self.assertFalse((root / "relaunch.log").exists())

    def test_incomplete_arms_trigger_one_relaunch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "out").mkdir(parents=True, exist_ok=True)
            (root / "out" / "alpha.jsonl").write_text('{"id": "a"}\n', encoding="utf-8")
            result = self._run_supervisor(root)
            self.assertIn("relaunch attempt 1/2", result.stdout)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not (root / "relaunch.log").exists():
                time.sleep(0.1)
            self.assertTrue(
                (root / "relaunch.log").exists(),
                "an incomplete run must trigger exactly one relaunch",
            )

    def test_budget_exhaustion_writes_gave_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for expected in (1, 2):
                result = self._run_supervisor(root)
                self.assertIn(f"relaunch attempt {expected}/2", result.stdout)
            final = self._run_supervisor(root)
            self.assertIn("GAVE_UP after 2 relaunches", final.stdout)
            self.assertTrue((root / "WORKLOAD_GAVE_UP").is_file())
            before = (root / "relaunch.log").read_text(encoding="utf-8").count("relaunch")
            again = self._run_supervisor(root)
            self.assertIn("GAVE_UP recorded earlier", again.stdout)
            after = (root / "relaunch.log").read_text(encoding="utf-8").count("relaunch")
            self.assertEqual(before, after)

    def test_progress_resets_the_relaunch_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIn("relaunch attempt 1/2", self._run_supervisor(root).stdout)
            self.assertIn("relaunch attempt 2/2", self._run_supervisor(root).stdout)
            # A new stage marker means the last relaunch did real work, so the
            # consecutive-failure budget starts over instead of giving up.
            (root / "stages").mkdir(parents=True, exist_ok=True)
            (root / "stages" / "01_alpha.done").write_text("", encoding="utf-8")
            result = self._run_supervisor(root)
            self.assertIn("relaunch attempt 1/2", result.stdout)
            self.assertFalse((root / "WORKLOAD_GAVE_UP").exists())

    def test_done_marker_alone_declares_all_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "WORKLOAD_DONE").write_text("", encoding="utf-8")
            result = self._run_supervisor(root)
            self.assertIn("ALL_DONE", result.stdout)
            self.assertFalse((root / "relaunch.log").exists())

    def test_effective_arms_spec_extends_the_static_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete_arms(root)
            (root / "workload_arms_effective.txt").write_text(
                "out/alpha.jsonl:2\nout/conditional.jsonl:2\n", encoding="utf-8"
            )
            result = self._run_supervisor(root)
            self.assertNotIn("ALL_DONE", result.stdout)
            self.assertIn("relaunch attempt", result.stdout)
            (root / "out" / "conditional.jsonl").write_text(
                '{"id": "c"}\n{"id": "d"}\n', encoding="utf-8"
            )
            result = self._run_supervisor(root)
            self.assertIn("ALL_DONE", result.stdout)

    def test_triggered_decision_blocks_all_done_until_the_arm_finishes(self) -> None:
        """Static completion must not pass for done while a triggered
        decision still owes a ninth arm: the runner can die between the
        decision and that arm, and the effective lists are only written at
        the very end of a successful pass."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete_arms(root)
            (root / "out" / "t5_conditional_decision.json").write_text(
                json.dumps(
                    {"trigger": True, "chosen": {"output": "out/conditional.jsonl"}}
                ),
                encoding="utf-8",
            )
            result = self._run_supervisor(root, env_extra={"OUT_DIR": "out"})
            self.assertNotIn("ALL_DONE", result.stdout)
            self.assertIn("relaunch attempt", result.stdout)
            self.assertFalse((root / "WORKLOAD_GAVE_UP").exists())
            # The arm finishes: the marker freezes the spec, artifacts land.
            (root / "stages").mkdir(parents=True, exist_ok=True)
            (root / "stages" / "16_t5_conditional_val.done").write_text(
                "out/conditional.jsonl:2\n"
                "reports/t5_conditional_val_error_analysis_newcard.json\n",
                encoding="utf-8",
            )
            (root / "out" / "conditional.jsonl").write_text(
                '{"id": "c"}\n{"id": "d"}\n', encoding="utf-8"
            )
            (root / "reports").mkdir(parents=True, exist_ok=True)
            (
                root / "reports" / "t5_conditional_val_error_analysis_newcard.json"
            ).write_text("{}\n", encoding="utf-8")
            result = self._run_supervisor(root, env_extra={"OUT_DIR": "out"})
            self.assertIn("ALL_DONE", result.stdout)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not (root / "relaunch.log").exists():
                time.sleep(0.1)
            self.assertTrue((root / "relaunch.log").is_file())
            before = (root / "relaunch.log").read_text(encoding="utf-8").count("relaunch")
            result = self._run_supervisor(root, env_extra={"OUT_DIR": "out"})
            self.assertIn("ALL_DONE", result.stdout)
            after = (root / "relaunch.log").read_text(encoding="utf-8").count("relaunch")
            self.assertEqual(before, after, "a finished workload must not relaunch")

    def test_unreadable_decision_blocks_all_done(self) -> None:
        """A decision that exists but cannot be parsed is unknown, not
        "nothing owed": refusing to finish is the only safe reading."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete_arms(root)
            (root / "out" / "t5_conditional_decision.json").write_text(
                "{not json", encoding="utf-8"
            )
            result = self._run_supervisor(root, env_extra={"OUT_DIR": "out"})
            self.assertNotIn("ALL_DONE", result.stdout)
            self.assertIn("relaunch attempt", result.stdout)

    def test_no_trigger_decision_allows_all_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete_arms(root)
            (root / "out" / "t5_conditional_decision.json").write_text(
                json.dumps({"trigger": False, "chosen": None}), encoding="utf-8"
            )
            result = self._run_supervisor(root, env_extra={"OUT_DIR": "out"})
            self.assertIn("ALL_DONE", result.stdout)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_new_format_marker_audits_the_frozen_row_count(self) -> None:
        """The new-format marker freezes "output:expected_lines"; the gate
        must audit that exact count, not merely that the file exists."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete_arms(root)
            (root / "out" / "t5_conditional_decision.json").write_text(
                json.dumps(
                    {"trigger": True, "chosen": {"output": "out/conditional.jsonl"}}
                ),
                encoding="utf-8",
            )
            (root / "stages").mkdir(parents=True, exist_ok=True)
            (root / "stages" / "16_t5_conditional_val.done").write_text(
                "out/conditional.jsonl:2\n"
                "reports/t5_conditional_val_error_analysis_newcard.json\n",
                encoding="utf-8",
            )
            (root / "reports").mkdir(parents=True, exist_ok=True)
            (
                root / "reports" / "t5_conditional_val_error_analysis_newcard.json"
            ).write_text("{}\n", encoding="utf-8")
            # 1 of the frozen 2 rows: non-empty, last line valid JSON --
            # every existence check passes, the row count does not.
            (root / "out" / "conditional.jsonl").write_text(
                '{"id": "c"}\n', encoding="utf-8"
            )
            result = self._run_supervisor(root, env_extra={"OUT_DIR": "out"})
            self.assertNotIn("ALL_DONE", result.stdout)
            self.assertIn("relaunch attempt", result.stdout)
            (root / "out" / "conditional.jsonl").write_text(
                '{"id": "c"}\n{"id": "d"}\n', encoding="utf-8"
            )
            result = self._run_supervisor(root, env_extra={"OUT_DIR": "out"})
            self.assertIn("ALL_DONE", result.stdout)

    def test_legacy_empty_marker_audits_the_val_input_row_count(self) -> None:
        """An empty marker plus a triggered decision and no audit lists is
        exactly the state a crashed old runner leaves behind: the expected
        count is re-derived from VAL_INPUT, and a short output must not
        pass for done."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete_arms(root)
            (root / "data").mkdir(parents=True, exist_ok=True)
            (root / "data" / "val.jsonl").write_text(
                '{"id": "v0"}\n{"id": "v1"}\n', encoding="utf-8"
            )
            (root / "out" / "t5_conditional_decision.json").write_text(
                json.dumps(
                    {"trigger": True, "chosen": {"output": "out/conditional.jsonl"}}
                ),
                encoding="utf-8",
            )
            (root / "stages").mkdir(parents=True, exist_ok=True)
            (root / "stages" / "16_t5_conditional_val.done").write_text(
                "", encoding="utf-8"
            )
            (root / "reports").mkdir(parents=True, exist_ok=True)
            (
                root / "reports" / "t5_conditional_val_error_analysis_newcard.json"
            ).write_text("{}\n", encoding="utf-8")
            env_extra = {"OUT_DIR": "out", "VAL_INPUT": "data/val.jsonl"}
            # 1 of the 2 val rows, and no audit list to fall back on.
            (root / "out" / "conditional.jsonl").write_text(
                '{"id": "c"}\n', encoding="utf-8"
            )
            result = self._run_supervisor(root, env_extra=env_extra)
            self.assertNotIn("ALL_DONE", result.stdout)
            self.assertIn("relaunch attempt", result.stdout)
            self.assertFalse((root / "WORKLOAD_GAVE_UP").exists())
            # The arm is completed: only now may the workload finish.
            (root / "out" / "conditional.jsonl").write_text(
                '{"id": "c"}\n{"id": "d"}\n', encoding="utf-8"
            )
            result = self._run_supervisor(root, env_extra=env_extra)
            self.assertIn("ALL_DONE", result.stdout)

    def test_legacy_marker_without_val_input_fails_closed(self) -> None:
        """Without VAL_INPUT the expected count is unknown, and unknown must
        never pass for done: the runner's own audit stays the only
        authority that may complete the workload."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete_arms(root)
            (root / "out" / "t5_conditional_decision.json").write_text(
                json.dumps(
                    {"trigger": True, "chosen": {"output": "out/conditional.jsonl"}}
                ),
                encoding="utf-8",
            )
            (root / "stages").mkdir(parents=True, exist_ok=True)
            (root / "stages" / "16_t5_conditional_val.done").write_text(
                "", encoding="utf-8"
            )
            (root / "reports").mkdir(parents=True, exist_ok=True)
            (
                root / "reports" / "t5_conditional_val_error_analysis_newcard.json"
            ).write_text("{}\n", encoding="utf-8")
            # A complete output still cannot prove its own row count here.
            (root / "out" / "conditional.jsonl").write_text(
                '{"id": "c"}\n{"id": "d"}\n', encoding="utf-8"
            )
            result = self._run_supervisor(root, env_extra={"OUT_DIR": "out"})
            self.assertNotIn("ALL_DONE", result.stdout)
            self.assertIn("relaunch attempt", result.stdout)

    def test_conditional_paths_stay_in_sync_with_the_runner(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        supervisor = SUPERVISOR.read_text(encoding="utf-8")
        for shared in (
            'COND_STAGE_NAME="16_t5_conditional_val"',
            "reports/t5_conditional_val_error_analysis_newcard.json",
        ):
            self.assertIn(shared, runner)
            self.assertIn(shared, supervisor)


if __name__ == "__main__":
    unittest.main()
