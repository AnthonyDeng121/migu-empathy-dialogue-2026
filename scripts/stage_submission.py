#!/usr/bin/env python3
"""Stage a fresh self-contained participant directory for platform image saving.

Does not replace /root/participant or save/publish a platform image.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    executable = args.python.absolute()
    if not executable.is_file() or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(executable)):
        parser.error("Provide an existing absolute Linux Python path without shell metacharacters")
    model_dirs = {}
    for key, relative in (("classifier_dir", "weights/emotion_profile"),
                          ("response_model_dir", "weights/hf/Qwen2.5-1.5B-Instruct"),
                          ("response_adapter_dir", "weights/response_sft_d2_1p5b")):
        source = Path(config[key])
        source = source if source.is_absolute() else PROJECT / source
        if not source.is_dir():
            parser.error(f"Missing model directory: {source}")
        model_dirs[relative] = source
        config[key] = relative
    destination = args.destination.absolute()
    if destination.exists():
        parser.error("Destination already exists; choose a fresh directory")
    destination.mkdir(parents=True)
    for name in ("src", "scripts", "configs"):
        shutil.copytree(PROJECT / name, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "cloud_runtime.json"))
    for relative, source in model_dirs.items():
        shutil.copytree(source, destination / relative)
    (destination / "configs/integrated.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
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
