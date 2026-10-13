"""Synthetic end-to-end check; no official data or downloads required."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for split in ("train", "val"):
            row = {"conversation_id": f"synthetic_{split}", "turns": [
                {"turn_id": 1, "role": "user", "content": "今天学习很开心"},
                {"turn_id": 2, "role": "assistant", "content": "很高兴你享受学习的过程"}],
                "prediction_points": [{"target_user_turn_id": 1, "target": {
                    "response_text": "很高兴你享受学习的过程", "emotion_label": "joy",
                    "user_profile": {"personality_traits": [], "interests": [], "style": []}}}]}
            (root / f"{split}.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        outputs = []
        for index in (1, 2):
            out = root / f"run{index}"
            subprocess.run([sys.executable, str(PROJECT / "scripts/run_experiment.py"), "--train", str(root / "train.jsonl"), "--val", str(root / "val.jsonl"), "--output-dir", str(out)], check=True)
            record = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
            if record["status"] != "passed" or record["metrics"]["emotion_accuracy"] != 1.0:
                raise RuntimeError("synthetic experiment failed")
            outputs.append((out / "submission.jsonl").read_bytes())
        if outputs[0] != outputs[1]:
            raise RuntimeError("repeated experiment predictions differ")
        if os.name == "posix":
            shell_out = root / "shell"
            shell_out.mkdir()
            (shell_out / "keep.txt").write_text("keep", encoding="utf-8")
            subprocess.run(["bash", str(PROJECT / "start.sh"), str(root / "run1/input.jsonl"), str(shell_out)],
                           env=dict(os.environ, BASELINE_TRAIN_FILE=str(root / "train.jsonl")), check=True)
            if (shell_out / "submission.jsonl").read_bytes() != outputs[0] or (shell_out / "keep.txt").read_text() != "keep":
                raise RuntimeError("two-argument shell entry or existing result directory failed")
    print("PASS synthetic experiment, schema, metrics and repeatability")


if __name__ == "__main__":
    main()
