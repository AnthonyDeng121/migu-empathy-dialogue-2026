#!/usr/bin/env python3
"""Compare two existing prediction files by ID; never run inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl  # noqa: E402
from scripts.b_validate_submission import duplicate_ids, row_ids, sha256_file  # noqa: E402

FIELDS = ("response_text", "emotion_label", "user_profile", "memory_refs")
METRIC_STATUS = "\u6307\u6807\u672a\u8ba1\u7b97\uff1a\u672a\u63d0\u4f9b gold/reference"


def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    ids = row_ids(rows)
    if not rows or None in ids:
        raise ValueError(f"{name}: non-empty rows and non-empty string IDs required")
    duplicates = duplicate_ids(ids)
    if duplicates:
        raise ValueError(f"{name}: duplicate IDs: {duplicates}")
    for row in rows:
        if not isinstance(row.get("response_text"), str):
            raise ValueError(f"{name}: {row['id']!r} needs string response_text")
    return {row["id"]: row for row in rows}


def compare_candidates(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare exact JSON field values on common IDs, in baseline order.

    Response-only files are supported. Absent optional fields are reported as
    unavailable, not equal. Array order is significant; object key order is not.
    Content changes do not make comparison execution fail.
    """
    left, right = _index(baseline, "baseline"), _index(candidate, "candidate")
    common = [key for key in left if key in right]
    counts = {field: {"equal_count": 0, "different_count": 0, "unavailable_count": 0} for field in FIELDS}
    examples: list[dict[str, Any]] = []
    different_samples = 0
    for key in common:
        changes = {}
        for field in FIELDS:
            a, b = field in left[key], field in right[key]
            if not (a and b):
                counts[field]["unavailable_count"] += 1
            elif left[key][field] == right[key][field]:
                counts[field]["equal_count"] += 1
            else:
                counts[field]["different_count"] += 1
            if a != b or (a and b and left[key][field] != right[key][field]):
                changes[field] = {
                    "baseline": left[key].get(field), "candidate": right[key].get(field),
                    "baseline_present": a, "candidate_present": b,
                }
        if changes:
            different_samples += 1
            if len(examples) < 10:
                examples.append({"id": key, "different_fields": list(changes), "values": changes})
    return {
        "version": "b_candidate_comparison_v2",
        "baseline_rows": len(baseline), "candidate_rows": len(candidate),
        "common_id_count": len(common),
        "baseline_only_ids": sorted(left.keys() - right.keys()),
        "candidate_only_ids": sorted(right.keys() - left.keys()),
        "order_identical": list(left) == list(right),
        "common_order_identical": common == [key for key in right if key in left],
        "fields": counts, "different_sample_count": different_samples,
        "difference_examples": examples, "metric_status": METRIC_STATUS,
    }


def compare_files(baseline: Path, candidate: Path) -> dict[str, Any]:
    report = compare_candidates(read_jsonl(baseline), read_jsonl(candidate))
    report.update(baseline=str(baseline), candidate=str(candidate),
                  baseline_sha256=sha256_file(baseline), candidate_sha256=sha256_file(candidate))
    return report


def summary(report: dict[str, Any]) -> str:
    lines = [
        "\u3010B \u5019\u9009\u9010\u6761\u6bd4\u8f83\u3011",
        f"baseline \u884c\u6570\uff1a{report['baseline_rows']}",
        f"candidate \u884c\u6570\uff1a{report['candidate_rows']}",
        f"common IDs\uff1a{report['common_id_count']}",
        f"baseline-only IDs\uff1a{json.dumps(report['baseline_only_ids'], ensure_ascii=False)}",
        f"candidate-only IDs\uff1a{json.dumps(report['candidate_only_ids'], ensure_ascii=False)}",
        f"\u987a\u5e8f\u4e00\u81f4\uff1a{report['order_identical']}",
    ]
    for field, counts in report["fields"].items():
        lines.append(f"{field} \u4e00\u81f4\uff1a{counts['equal_count']} / \u4e0d\u4e00\u81f4\uff1a{counts['different_count']} / \u5b57\u6bb5\u7f3a\u5931\u65e0\u6cd5\u6bd4\u8f83\uff1a{counts['unavailable_count']}")
    lines.extend([
        f"baseline SHA-256\uff1a{report['baseline_sha256']}",
        f"candidate SHA-256\uff1a{report['candidate_sha256']}",
        "\u5dee\u5f02\u4f8b\u5b50\uff08\u6700\u591a 10 \u6761\uff09\uff1a" + json.dumps(report['difference_examples'], ensure_ascii=False),
        report["metric_status"],
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="optional JSON report")
    args = parser.parse_args()
    try:
        report = compare_files(args.baseline, args.candidate)
    except (OSError, ValueError, TypeError) as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 1
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
