#!/usr/bin/env python3
"""Prepare an explicit, hashed expert-material context without acceptance records."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MATERIALS = (
    "docs/TECHNICAL_SOLUTION.md", "docs/TRAINING_WORKPLAN.md",
    "reports/EXPERIMENT_TEMPLATE.md", "reports/C_DAY3_EMOTION_PROFILE.md",
    "reports/C_DAY3_RETRAINED_HANDOFF_20261015.md",
    "reports/C_DAY4_PROFILE_UNWEIGHTED.md", "reports/c_day3_retrained_metrics.json",
    "reports/c_day4_profile_unweighted_metrics.json",
    "reports/D_DAY3_RESPONSE_LORA_SFT.md", "reports/D_DAY3_SFT_FULL_1P5B.md",
    "reports/d_day3_sft_full_1p5b.json", "logs/D_DAY3_SFT_FULL_1P5B.txt",
    "logs/D_DAY3_SFT_SMOKE_1P5B.txt", "configs/integrated.json",
    "configs/integrated_artifacts.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh directory for the Docker expert_materials context")
    args = parser.parse_args()
    missing = [name for name in MATERIALS if not (PROJECT / name).is_file()]
    if missing:
        parser.error(f"Missing required materials: {missing}")
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; choose a fresh directory")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    output.mkdir(parents=True)
    hashes = {}
    for name in MATERIALS:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {"source_commit": commit, "files": hashes,
                "candidate": "C retrained emotion; empty profile; D Day3 1.5B LoRA",
                "limitations": ["Official scoring and final image approval remain pending",
                                "C training logs embedded in metrics JSON; D raw logs included",
                                "Acceptance records intentionally excluded"]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(hashes)} materials: {output}")


if __name__ == "__main__":
    main()
