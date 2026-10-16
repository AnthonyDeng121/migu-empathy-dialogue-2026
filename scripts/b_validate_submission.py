#!/usr/bin/env python3
"""CPU-only submission checks using the repository's JSONL and schema code."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl  # noqa: E402
from src.data.schema import validate_prediction, validate_test_row  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_ids(rows: list[dict[str, Any]]) -> list[str | None]:
    return [row.get("id") if isinstance(row.get("id"), str) and row["id"] else None for row in rows]


def duplicate_ids(ids: list[str | None]) -> list[str]:
    return sorted(key for key, count in Counter(ids).items() if key is not None and count > 1)


def validate_rows(inputs: list[dict[str, Any]], submissions: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for kind, rows, validator in (
        ("input_schema", inputs, validate_test_row),
        ("submission_schema", submissions, validate_prediction),
    ):
        for index, row in enumerate(rows, 1):
            try:
                validator(row)
            except (KeyError, TypeError, ValueError) as exc:
                issues.append({"kind": kind, "row": index, "message": str(exc)})
    expected, actual = row_ids(inputs), row_ids(submissions)
    missing = sorted(set(expected) - set(actual) - {None})
    extra = sorted(set(actual) - set(expected) - {None})
    duplicate_input, duplicate_submission = duplicate_ids(expected), duplicate_ids(actual)
    # Position mismatches are independent of set completeness and row counts.
    mismatches = [
        {"row": index, "input_id": left, "submission_id": right}
        for index, (left, right) in enumerate(zip(expected, actual), 1) if left != right
    ]
    empty_rows = [
        index for index, row in enumerate(submissions, 1)
        if isinstance(row.get("response_text"), str) and not row["response_text"].strip()
    ]
    for kind, failed, message in (
        ("empty_input", not inputs, "input contains no rows"),
        ("row_count", len(inputs) != len(submissions), "input/submission row count mismatch"),
        ("missing_ids", bool(missing), "expected IDs missing from submission"),
        ("extra_ids", bool(extra), "unexpected IDs in submission"),
        ("duplicate_input_id", bool(duplicate_input), "duplicate input IDs"),
        ("duplicate_submission_id", bool(duplicate_submission), "duplicate submission IDs"),
        ("id_alignment", bool(mismatches), "IDs differ at corresponding positions"),
        ("empty_response", bool(empty_rows), "empty or whitespace-only response_text"),
    ):
        if failed:
            issues.append({"kind": kind, "message": message})
    return {
        "version": "b_submission_validation_v2", "passed": not issues,
        "input_rows": len(inputs), "submission_rows": len(submissions),
        "missing_ids": missing, "extra_ids": extra,
        "missing_id_count": len(missing), "extra_id_count": len(extra),
        "duplicate_input_ids": duplicate_input, "duplicate_submission_ids": duplicate_submission,
        "duplicate_submission_id_count": len(duplicate_submission),
        "id_mismatches": mismatches, "order_mismatch_count": len(mismatches),
        "schema_error_count": sum(i["kind"].endswith("_schema") for i in issues),
        "empty_response_rows": empty_rows, "empty_response_count": len(empty_rows),
        "issues": issues,
    }


def validate_files(input_path: Path, submission_path: Path) -> dict[str, Any]:
    rows, errors = [], []
    for kind, path in (("input_read", input_path), ("submission_read", submission_path)):
        try:
            rows.append(read_jsonl(path))
        except (OSError, ValueError) as exc:
            rows.append([])
            errors.append({"kind": kind, "message": str(exc)})
    report = validate_rows(*rows)
    try:
        report["submission_sha256"] = sha256_file(submission_path)
    except OSError as exc:
        report["submission_sha256"] = None
        errors.append({"kind": "submission_hash", "message": str(exc)})
    report.update(input=str(input_path), submission=str(submission_path))
    report["issues"] = errors + report["issues"]
    report["passed"] = not report["issues"]
    return report


def summary(report: dict[str, Any]) -> str:
    return "\n".join([
        "\u3010B \u63d0\u4ea4\u7ed3\u679c\u9a8c\u6536\u3011",
        f"\u9884\u671f\u6837\u672c\u6570\uff1a{report['input_rows']}",
        f"\u5b9e\u9645\u6837\u672c\u6570\uff1a{report['submission_rows']}",
        f"\u91cd\u590d ID\uff1a{report['duplicate_submission_id_count']}",
        f"\u8f93\u5165\u91cd\u590d ID\uff1a{len(report['duplicate_input_ids'])}",
        f"\u7f3a\u5931 ID\uff1a{report['missing_id_count']}",
        f"\u989d\u5916 ID\uff1a{report['extra_id_count']}",
        f"\u987a\u5e8f\u9519\u8bef\uff1a{report['order_mismatch_count']}",
        f"Schema \u9519\u8bef\uff1a{report['schema_error_count']}",
        f"\u7a7a\u56de\u590d\uff1a{report['empty_response_count']}",
        f"\u6587\u4ef6 SHA-256\uff1a{report['submission_sha256']}",
        "\n\u6700\u7ec8\u7ed3\u679c\uff1a" + ("\u5df2\u901a\u8fc7" if report['passed'] else "\u672a\u901a\u8fc7"),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_files(args.input, args.submission)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(summary(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
