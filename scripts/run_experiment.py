#!/usr/bin/env python3
"""Reproduce the retrieval experiment and retain provenance beside its outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/baseline.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    sources = {name: getattr(args, name).resolve() for name in ("train", "val", "config")}
    config = json.loads(sources["config"].read_text(encoding="utf-8"))
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("--output-dir must be empty; use a new directory for each experiment")
    git = lambda *cmd: subprocess.check_output(["git", *cmd], cwd=PROJECT, text=True).strip()
    record = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version, "platform": platform.platform(), "seed": config["seed"],
        "config": config, "limit": args.limit,
        "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()},
        "commands": [], "metric_scope": "character-level local diagnostics, not official scores",
    }
    env = dict(os.environ, PYTHONHASHSEED=str(config["seed"]), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    def save() -> None:
        (out / "experiment.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    def run(script: str, *arguments: object) -> None:
        command = [sys.executable, str(PROJECT / "scripts" / script), *map(str, arguments)]
        record["commands"].append(command)
        save()
        subprocess.run(command, cwd=PROJECT, env=env, check=True)
    save()
    try:
        run("make_eval_input.py", sources["val"], out / "input.jsonl", *(["--limit", args.limit] if args.limit else []))
        run("run_baseline.py", out / "input.jsonl", out, "--train", sources["train"], "--config", sources["config"])
        run("validate_io.py", "--input", out / "input.jsonl", "--submission", out / "submission.jsonl")
        run("evaluate_local.py", "--gold-dataset", sources["val"], "--input", out / "input.jsonl", "--submission", out / "submission.jsonl", "--output", out / "metrics.json")
        record["metrics"] = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        record["outputs_sha256"] = {name: sha256(out / name) for name in ("input.jsonl", "submission.jsonl", "metrics.json", "performance_report.json")}
        record["status"] = "passed"
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
