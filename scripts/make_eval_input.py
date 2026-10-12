#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert labeled conversations to official-like inference rows")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = []
    for conversation in read_jsonl(args.dataset):
        for point in conversation["prediction_points"]:
            target_id = point["target_user_turn_id"]
            history = [turn for turn in conversation["turns"] if turn["turn_id"] <= target_id]
            rows.append({
                "id": f"eval_{len(rows) + 1:06d}_t{target_id}",
                "conversation_id": conversation["conversation_id"],
                "target_user_turn_id": target_id,
                "history": history,
            })
            if args.limit and len(rows) >= args.limit:
                write_jsonl(args.output, rows)
                print(f"PASS rows={len(rows)}")
                return
    write_jsonl(args.output, rows)
    print(f"PASS rows={len(rows)}")


if __name__ == "__main__":
    main()

