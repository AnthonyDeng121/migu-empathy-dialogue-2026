#!/usr/bin/env python3
"""Compare the T2 condition arms (off vs emotion-only) on the frozen subset.

Cycle-2 task T2 asks whether injecting the predicted emotion condition
changes the generated responses at all, and in which direction. This script
joins the two arm outputs by prediction-point id and reports, per id:

  * whether the two arms produced byte-identical responses;
  * char-level BLEU-4 / ROUGE-L of each arm against the gold response
    (local diagnostic implementations from ``src/evaluation/metrics.py`` --
    the official competition metric is not published, so these numbers are
    never reported as official scores);
  * response length in non-whitespace characters, and the arm-to-arm delta.

The report is written as JSON (machine-checkable, committed as evidence)
and summarised on stdout. Gold responses come from the frozen selection
input; no labels beyond that enter the comparison.
"""
from __future__ import annotations

import argparse
import json
import shlex
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl  # noqa: E402
from src.evaluation.metrics import bleu4, rouge_l  # noqa: E402

DEFAULT_OFF = PROJECT / "results/response_gen/t2_condition_off.jsonl"
DEFAULT_EMOTION = PROJECT / "results/response_gen/t2_emotion_only.jsonl"
DEFAULT_SELECTION = PROJECT / "results/response_gen/t2_selection_input.jsonl"
DEFAULT_OUTPUT = PROJECT / "results/response_gen/t2_arms_comparison.json"

EXAMPLE_LIMIT = 5


def _index(rows: Sequence[Mapping[str, Any]], id_field: str, source: Path) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = row.get(id_field)
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{source}: row without a string {id_field}")
        if row_id in by_id:
            raise ValueError(f"{source}: duplicate {id_field} {row_id!r}")
        by_id[row_id] = row
    return by_id


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _length(text: str) -> int:
    return len("".join(text.split()))


def compare_arms(
    off_rows: Sequence[Mapping[str, Any]],
    emotion_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    off_path: Path = DEFAULT_OFF,
    emotion_path: Path = DEFAULT_EMOTION,
    selection_path: Path = DEFAULT_SELECTION,
) -> dict[str, Any]:
    """Per-id and aggregate comparison of the two arms against gold."""
    off = _index(off_rows, "id", off_path)
    emotion = _index(emotion_rows, "id", emotion_path)
    gold = _index(selection_rows, "sample_id", selection_path)

    ids = sorted(set(off) & set(emotion) & set(gold))
    only_off = sorted(set(off) - set(emotion))
    only_emotion = sorted(set(emotion) - set(off))
    missing_gold = sorted((set(off) | set(emotion)) - set(gold))
    missing_predictions = sorted(set(gold) - (set(off) & set(emotion)))
    if only_off or only_emotion or missing_gold or missing_predictions:
        raise ValueError(
            "arm id sets differ from each other or from the selection: "
            f"only_off={len(only_off)} only_emotion={len(only_emotion)} "
            f"missing_gold={len(missing_gold)} missing_predictions={len(missing_predictions)} (first: "
            f"{(only_off or only_emotion or missing_gold or missing_predictions)[0]!r})"
        )

    per_id: list[dict[str, Any]] = []
    identical = 0
    for row_id in ids:
        off_text = off[row_id]["response_text"]
        emotion_text = emotion[row_id]["response_text"]
        gold_text = gold[row_id]["response_text"]
        same = off_text == emotion_text
        identical += int(same)
        per_id.append(
            {
                "id": row_id,
                "identical": same,
                "off": {
                    "bleu4": bleu4(gold_text, off_text),
                    "rouge_l": rouge_l(gold_text, off_text),
                    "length": _length(off_text),
                },
                "emotion_only": {
                    "bleu4": bleu4(gold_text, emotion_text),
                    "rouge_l": rouge_l(gold_text, emotion_text),
                    "length": _length(emotion_text),
                },
            }
        )

    def aggregate(arm: str) -> dict[str, float]:
        return {
            "bleu4": _mean(row[arm]["bleu4"] for row in per_id),
            "rouge_l": _mean(row[arm]["rouge_l"] for row in per_id),
            "mean_length": _mean(row[arm]["length"] for row in per_id),
            "median_length": statistics.median(row[arm]["length"] for row in per_id),
        }

    length_deltas = [
        row["emotion_only"]["length"] - row["off"]["length"] for row in per_id
    ]
    examples = []
    for row in per_id:
        if row["identical"]:
            continue
        if len(examples) >= EXAMPLE_LIMIT:
            break
        examples.append(
            {
                "id": row["id"],
                "gold": gold[row["id"]]["response_text"],
                "off": off[row["id"]]["response_text"],
                "emotion_only": emotion[row["id"]]["response_text"],
            }
        )

    return {
        "version": "t2_arms_comparison_v1",
        "purpose": (
            "Cycle-2 T2: fixed 160-row stratified subset, D Day4 v2 weights and "
            "decoding held fixed; conditions off vs emotion-only predicted by "
            "C Day5. Local char-level BLEU-4/ROUGE-L are diagnostic only -- the "
            "official competition metric is not published."
        ),
        "ids": {
            "compared": len(ids),
            "only_off": only_off,
            "only_emotion": only_emotion,
            "missing_gold": missing_gold,
        },
        "identical_responses": {
            "count": identical,
            "rate": identical / len(ids) if ids else 0.0,
        },
        "aggregate": {"off": aggregate("off"), "emotion_only": aggregate("emotion_only")},
        "length_delta": {
            "mean": _mean(length_deltas),
            "median": statistics.median(length_deltas) if length_deltas else 0.0,
            "longer_under_emotion": sum(1 for d in length_deltas if d > 0),
            "shorter_under_emotion": sum(1 for d in length_deltas if d < 0),
            "same_length": sum(1 for d in length_deltas if d == 0),
        },
        "divergence_examples": examples,
        "per_id": per_id,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--off", type=Path, default=DEFAULT_OFF)
    parser.add_argument("--emotion", type=Path, default=DEFAULT_EMOTION)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = compare_arms(
        read_jsonl(args.off),
        read_jsonl(args.emotion),
        read_jsonl(args.selection),
        args.off,
        args.emotion,
        args.selection,
    )
    report["command"] = "python " + shlex.join(
        ["scripts/compare_t2_arms.py", *(argv or [])]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    off_agg = report["aggregate"]["off"]
    emo_agg = report["aggregate"]["emotion_only"]
    same = report["identical_responses"]
    print(
        f"compared {report['ids']['compared']} ids; identical responses "
        f"{same['count']}/{report['ids']['compared']} ({same['rate']:.1%})"
    )
    for name, agg in (("off", off_agg), ("emotion_only", emo_agg)):
        print(
            f"{name:>13}: BLEU-4 {agg['bleu4']:.4f}  ROUGE-L {agg['rouge_l']:.4f}  "
            f"mean length {agg['mean_length']:.1f}"
        )
    delta = report["length_delta"]
    print(
        f"length delta (emotion - off): mean {delta['mean']:+.1f}, "
        f"longer {delta['longer_under_emotion']}, shorter {delta['shorter_under_emotion']}, "
        f"same {delta['same_length']}"
    )
    print(f"report -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
