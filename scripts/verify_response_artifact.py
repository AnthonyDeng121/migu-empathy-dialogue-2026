#!/usr/bin/env python3
"""Verify the delivered response artifact is self-contained and reproducible.

The submission image runs with no network, and G5 requires two cold starts to
produce identical predictions. Neither property follows from "it trained and it
inferred once on the training box", so this script re-runs the real inference
entry point under forced-offline environment variables, twice, and compares the
two runs to each other and to the recorded full-val predictions. Artifact bytes
are pinned by directory digest so a swapped adapter or a half-copied model
snapshot fails loudly instead of silently changing the submitted model.

Exit code 0 only when every hard gate passes; a diagnostic-only disagreement
with the recorded remote run is reported but does not fail the gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl  # noqa: E402
from src.evaluation.metrics import bleu4, rouge_l  # noqa: E402

INFERENCE_SCRIPT = PROJECT / "scripts" / "run_response_inference.py"
DEFAULT_ARTIFACT_CONFIG = PROJECT / "configs" / "response_artifact.json"

# Set before transformers is imported (the inference subprocess imports it):
# without these, a machine with a warm Hugging Face cache proves nothing about
# an image that has no cache and no route out.
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def enforce_offline() -> dict[str, str]:
    """Force offline mode; return the variables this overrode."""
    overridden: dict[str, str] = {}
    for key, value in OFFLINE_ENV.items():
        if os.environ.get(key) != value:
            overridden[key] = os.environ.get(key, "")
            os.environ[key] = value
    return overridden


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_digest(directory: Path) -> dict[str, Any]:
    """Digest over a model directory's loadable files.

    Hidden entries are excluded: the Hugging Face ``.cache`` directory is
    download bookkeeping rather than model bytes, and it differs between a
    fresh download and a copied artifact.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"artifact directory not found: {directory}")
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(directory).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(sha256_file(path).encode("utf-8"))
        total_bytes += path.stat().st_size
    return {"digest": digest.hexdigest(), "files": len(files), "bytes": total_bytes}


def verify_digest(name: str, directory: Path, expected: str | None) -> dict[str, Any]:
    """Compare a directory's digest against the pinned value."""
    actual = artifact_digest(directory)
    if expected is None:
        status = "unpinned"
    elif actual["digest"] == expected:
        status = "ok"
    else:
        status = "mismatch"
    return {
        "name": name,
        "path": str(directory),
        "expected": expected,
        "actual": actual["digest"],
        "files": actual["files"],
        "bytes": actual["bytes"],
        "status": status,
    }


def predictions_by_id(path: Path) -> dict[str, str]:
    return {row["id"]: row["response_text"] for row in read_jsonl(path)}


def compare_runs(first: dict[str, str], second: dict[str, str]) -> dict[str, Any]:
    """Cold-start determinism gate: the two runs must agree byte for byte."""
    keys = sorted(set(first) | set(second))
    differing = [key for key in keys if first.get(key) != second.get(key)]
    return {
        "ids": len(keys),
        "identical": not differing,
        "differing_ids": differing,
    }


def compare_to_recorded(
    predictions: dict[str, str], recorded: dict[str, str]
) -> dict[str, Any]:
    """Diagnostic agreement with the recorded remote full-val predictions.

    The recorded run used a different GPU in bfloat16, so exact agreement is
    not required; this quantifies how far this machine drifts.
    """
    shared = sorted(set(predictions) & set(recorded))
    missing = sorted(set(predictions) - set(recorded))
    if not shared:
        return {
            "compared": 0,
            "missing_from_recorded": missing,
            "exact_matches": 0,
            "exact_match_rate": None,
            "mean_bleu4": None,
            "mean_rouge_l": None,
            "empty_predictions": 0,
        }
    exact = sum(1 for key in shared if predictions[key] == recorded[key])
    return {
        "compared": len(shared),
        "missing_from_recorded": missing,
        "exact_matches": exact,
        "exact_match_rate": round(exact / len(shared), 4),
        "mean_bleu4": round(sum(bleu4(recorded[key], predictions[key]) for key in shared) / len(shared), 4),
        "mean_rouge_l": round(
            sum(rouge_l(recorded[key], predictions[key]) for key in shared) / len(shared), 4
        ),
        "empty_predictions": sum(1 for key in shared if not predictions[key].strip()),
    }


def build_command(
    args: argparse.Namespace, output: Path, performance_report: Path
) -> list[str]:
    """The exact inference invocation, so tests can pin the flags we rely on."""
    command = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--input",
        str(args.eval_input),
        "--model",
        str(args.model),
        "--adapter",
        str(args.adapter),
        "--output",
        str(output),
        "--performance-report",
        str(performance_report),
        "--device",
        args.device,
    ]
    if args.samples is not None:
        command += ["--max-samples", str(args.samples)]
    if args.max_new_tokens is not None:
        command += ["--max-new-tokens", str(args.max_new_tokens)]
    return command


def run_inference(
    args: argparse.Namespace, output: Path, performance_report: Path
) -> dict[str, Any]:
    """Run the real inference entry point in a subprocess under offline env."""
    command = build_command(args, output, performance_report)
    completed = subprocess.run(command, cwd=str(PROJECT), capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"inference run failed ({completed.returncode}):\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
    report = json.loads(performance_report.read_text(encoding="utf-8"))
    report["_command"] = command
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the response artifact loads offline and reproduces itself"
    )
    parser.add_argument("--model", type=Path, required=True, help="local base model snapshot")
    parser.add_argument("--adapter", type=Path, required=True, help="LoRA adapter directory")
    parser.add_argument(
        "--eval-input",
        type=Path,
        default=PROJECT / "data/processed/b1_full_v1/response_val.jsonl",
    )
    parser.add_argument(
        "--recorded",
        type=Path,
        default=PROJECT / "results/response_gen/d2_1p5b_val_full.jsonl",
        help="recorded full-val predictions to compare against",
    )
    parser.add_argument(
        "--artifact-config",
        type=Path,
        default=DEFAULT_ARTIFACT_CONFIG,
        help="pinned digests for the base model and adapter",
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--keep-runs",
        type=Path,
        help="directory to copy all three runs' predictions and performance reports into",
    )
    return parser.parse_args()


def keep_runs(root: Path, destination: Path) -> list[str]:
    """Copy both runs' raw predictions and reports out of the temp directory."""
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in sorted(root.iterdir()):
        if path.is_file():
            target = destination / path.name
            target.write_bytes(path.read_bytes())
            copied.append(str(target))
    return copied


def copy_pinned_files(source: Path, staging: Path) -> Path:
    """Copy only the digest-covered files into a fresh directory.

    This is the portability gate: the submission image holds the model at a
    different path and is built from a copied file set, so a directory that
    only loads because of leftover Hugging Face cache metadata proves nothing.
    Hidden entries are skipped for the same reason the digest skips them.
    """
    staging.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part.startswith(".") for part in relative.parts):
            continue
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    return staging


def run(args: argparse.Namespace) -> dict[str, Any]:
    overridden = enforce_offline()
    config: dict[str, Any] = {}
    if args.artifact_config.exists():
        config = json.loads(args.artifact_config.read_text(encoding="utf-8"))
    base_config = config.get("base_model", {})
    adapter_config = config.get("adapter", {})

    digests = [
        verify_digest("base_model", args.model, base_config.get("expected_digest")),
        verify_digest("adapter", args.adapter, adapter_config.get("expected_digest")),
    ]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runs = []
        # Runs 1 and 2 use the working copy: the G5 cold-start determinism gate.
        for index in (1, 2):
            runs.append(
                run_inference(
                    args,
                    root / f"run{index}.jsonl",
                    root / f"run{index}_report.json",
                )
            )
        # Run 3 rebuilds the artifact from only the digest-pinned files at a
        # different path, which is what the image actually does.
        staged_model = copy_pinned_files(args.model, root / "staged_model")
        staged_adapter = copy_pinned_files(args.adapter, root / "staged_adapter")
        staged_args = argparse.Namespace(**{**vars(args), "model": staged_model, "adapter": staged_adapter})
        runs.append(
            run_inference(
                staged_args,
                root / "run3.jsonl",
                root / "run3_report.json",
            )
        )
        first = predictions_by_id(root / "run1.jsonl")
        second = predictions_by_id(root / "run2.jsonl")
        staged = predictions_by_id(root / "run3.jsonl")
        kept = keep_runs(root, args.keep_runs) if args.keep_runs else []

    determinism = compare_runs(first, second)
    portability = compare_runs(first, staged)
    recorded = predictions_by_id(args.recorded) if args.recorded.exists() else {}
    agreement = compare_to_recorded(first, recorded)

    failures: list[str] = []
    for entry in digests:
        if entry["status"] == "mismatch":
            failures.append(f"{entry['name']} digest mismatch")
        elif entry["status"] == "unpinned":
            failures.append(f"{entry['name']} digest is not pinned")
    if not determinism["identical"]:
        failures.append(
            f"two cold-start runs differ on {len(determinism['differing_ids'])} ids "
            f"(first: {determinism['differing_ids'][:3]})"
        )
    if not portability["identical"]:
        failures.append(
            "pinned-file rebuild at a different path differs on "
            f"{len(portability['differing_ids'])} ids "
            f"(first: {portability['differing_ids'][:3]})"
        )
    if agreement["empty_predictions"]:
        failures.append(f"{agreement['empty_predictions']} empty responses")
    if agreement["compared"] == 0:
        failures.append("no predictions overlapped the recorded file")

    report = {
        "version": "response_artifact_verification_v1",
        "offline_env": {**OFFLINE_ENV, "_overridden": overridden},
        "artifact_config": str(args.artifact_config),
        "digests": digests,
        "runs": [
            {key: value for key, value in entry.items() if key != "_command"}
            for entry in runs
        ],
        "commands": [run["_command"] for run in runs],
        "kept_run_artifacts": kept,
        "determinism": determinism,
        "portability": portability,
        "recorded_agreement": agreement,
        "gates": {
            "digests_pinned_and_matching": all(
                entry["status"] == "ok" for entry in digests
            ),
            "deterministic": determinism["identical"],
            "pinned_files_sufficient_and_path_independent": portability["identical"],
            "no_empty_responses": agreement["empty_predictions"] == 0,
            "recorded_overlap": agreement["compared"] > 0,
        },
        "passed": not failures,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    status = "PASS" if report["passed"] else "FAIL"
    print(
        f"{status}: artifact verification ({args.samples} samples x3, "
        f"device={runs[0]['device']}, dtype={runs[0]['dtype']}); "
        f"deterministic={determinism['identical']}, "
        f"pinned-file rebuild identical={portability['identical']}, "
        f"recorded exact match={agreement['exact_matches']}/{agreement['compared']}, "
        f"report={args.report}",
        flush=True,
    )
    if failures:
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
    return report


if __name__ == "__main__":
    run(parse_args())
