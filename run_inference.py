#!/usr/bin/env python3
"""Compatibility entry for the official Transformers launcher.

Always use the validated C classifier and D adapter, including when the
platform retains the official start.sh instead of our deployment wrapper.
"""
import os
import sys
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
os.environ["INTEGRATED_CONFIG"] = str(PROJECT / "configs/integrated.json")
compat = json.loads((PROJECT / "configs/inference_config.json").read_text(encoding="utf-8"))
python = Path(os.environ.get("PYTHON_BIN") or compat.get("python_bin") or sys.executable)
if python.is_file() and Path(sys.executable).absolute() != python.absolute():
    os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
from scripts.run_integrated import main

if __name__ == "__main__":
    main()
