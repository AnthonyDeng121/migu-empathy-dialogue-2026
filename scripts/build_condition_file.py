#!/usr/bin/env python3
"""Turn a packaged C emotion/profile checkpoint into a D condition file.

The D2 candidate was trained with gold emotion/profile conditions, so the
deployment question is whether *predicted* conditions help or hurt. This script
runs C's checkpoint over the frozen B1 eval rows and writes the condition file
that ``scripts/run_response_inference.py --condition-jsonl`` consumes: one row
per prediction point, keyed by the B1 ``sample_id``.

Contexts are taken from B1 ``context_turns`` as-is. B1 already cuts each context
at the target user turn (``context_after_target=false`` in the conversion), which
is what C's ``predict_contexts`` requires from its caller — it does not cut, so
passing uncut conversations would leak future turns. The row count and id order
are asserted against the input so a silent partial run cannot masquerade as a
full one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl, write_jsonl  # noqa: E402


def contexts_from_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Extract the context turn lists C's checkpoint expects, in input order."""
    contexts: list[list[dict[str, Any]]] = []
    for row in rows:
        turns = row.get("context_turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"row {row.get('sample_id')!r} has no context_turns list")
        if turns[-1].get("role") != "user":
            raise ValueError(
                f"row {row.get('sample_id')!r} context does not end with a user turn"
            )
        contexts.append([{"role": t["role"], "content": t["content"]} for t in turns])
    return contexts


def condition_rows(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pair each prediction with its sample id, refusing to drop or reorder any."""
    if len(rows) != len(predictions):
        raise ValueError(
            f"{len(predictions)} predictions for {len(rows)} rows; refusing to "
            "write a partial condition file"
        )
    conditions: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        if "emotion_label" not in prediction or not isinstance(
            prediction.get("user_profile"), dict
        ):
            raise ValueError(f"prediction for {row['sample_id']!r} lacks a condition")
        conditions.append(
            {
                "id": row["sample_id"],
                "emotion_label": prediction["emotion_label"],
                "user_profile": prediction["user_profile"],
            }
        )
    return conditions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a D condition file from a packaged C checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT / "data/processed/b1_full_v1/response_val.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from src.classification.emotion_profile_checkpoint import EmotionProfileCheckpoint

    rows = read_jsonl(args.input)
    contexts = contexts_from_rows(rows)
    model = EmotionProfileCheckpoint(args.checkpoint, device=args.device).eval()
    predictions = model.predict_contexts(contexts)
    conditions = condition_rows(rows, predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, conditions)

    emotion_counts: dict[str, int] = {}
    empty_profiles = 0
    for condition in conditions:
        emotion_counts[condition["emotion_label"]] = (
            emotion_counts.get(condition["emotion_label"], 0) + 1
        )
        if all(not values for values in condition["user_profile"].values()):
            empty_profiles += 1
    summary = {
        "version": "condition_file_v1",
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "rows": len(conditions),
        "empty_profiles": empty_profiles,
        "emotion_label_counts": dict(
            sorted(emotion_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
    print(
        f"PASS: {len(conditions)} conditions -> {args.output} "
        f"({empty_profiles} with an all-empty profile)",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run(parse_args())
