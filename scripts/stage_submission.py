#!/usr/bin/env python3
"""Stage a fresh self-contained participant directory for platform image saving.

Does not replace /root/participant or save/publish a platform image.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.run_integrated import load_config, check_files


def stage(config_path, destination):
    """Copy selected assets and return a portable config; no model inference."""
    config = load_config(config_path)
    check_files(config)
    keys = {"classifier_dir": "weights/emotion_longtail075_b1_20261016",
            "response_model_dir": "weights/hf/Qwen2.5-1.5B-Instruct",
            "response_adapter_dir": "weights/response_sft_d4_2ep"}
    if config.get("profile_mode") == "independent":
        keys["profile_model_dir"] = "weights/profile_encoder_v1_b1_20261026"
    destination = Path(destination).absolute()
    if destination.exists():
        raise ValueError("Destination already exists; choose a fresh directory")
    destination.mkdir(parents=True)
    for name in ("src", "scripts", "configs"):
        shutil.copytree(PROJECT / name, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "cloud_runtime.json"))
    for key, relative in keys.items():
        shutil.copytree(config[key], destination / relative)
        config[key] = relative
    if config.get("profile_mode") != "independent":
        config.pop("profile_model_dir", None)
    (destination / "configs/integrated.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    executable = args.python.absolute()
    if not executable.is_file() or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(executable)):
        parser.error("Provide an existing absolute Linux Python path without shell metacharacters")
    destination = args.destination.absolute()
    stage(args.config, destination)
    compatibility_path = destination / "configs/inference_config.json"
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    compatibility["python_bin"] = str(executable)
    compatibility_path.write_text(json.dumps(compatibility, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(PROJECT / "run_inference.py", destination / "run_inference.py")
    shutil.copyfile(PROJECT / "start.sh", destination / "start_backend.sh")
    wrapper = ('#!/bin/sh\nset -eu\n'
               f'export PYTHON_BIN={executable}\n'
               'export INFERENCE_BACKEND=integrated HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'
               'export TOKENIZERS_PARALLELISM=false\n'
               'SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
               'export INTEGRATED_CONFIG="$SCRIPT_DIR/configs/integrated.json"\n'
               'exec bash "$SCRIPT_DIR/start_backend.sh" "$@"\n')
    (destination / "start.sh").write_text(wrapper, encoding="utf-8")
    (destination / "start.sh").chmod(0o755)
    print(f"Staged participant: {destination}")
    print(f"Python environment must also be retained in image: {executable}")


if __name__ == "__main__":
    main()
