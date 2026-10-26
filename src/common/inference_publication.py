"""Publish and verify a prediction/report pair with a final commit marker."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

from src.common.inference_resume import atomic_json, file_digest


def marker_path(report_path):
    report_path = Path(report_path)
    return report_path.with_name(report_path.name + ".publication.json")


def _prediction_ids(data):
    rows = [json.loads(line) for line in data.splitlines()]
    ids = [row["id"] for row in rows]
    if (not ids or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Publication predictions must have nonempty unique IDs")
    return ids


def validate_publication(output_path, report_path, expected_ids=None):
    """Reject pending, legacy, mixed or modified files. Return the bound report.

    Readers use the same bytes for hashes and row counts, and recheck the marker
    after reading to detect a concurrent publication. `complete` in an integrated
    timing report is deliberately unrelated to publication completion.
    """
    output_path, report_path = Path(output_path), Path(report_path)
    marker = marker_path(report_path)
    try:
        before = marker.read_bytes()
        commit = json.loads(before)
        if commit.get("state") != "complete" or commit.get("version") != 1:
            raise ValueError("Publication has no completed commit marker")
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
        publication = report.get("publication", {})
        if (publication.get("id") != commit["id"] or publication.get("state") != "complete"
                or publication.get("version") != 1
                or hashlib.sha256(report_bytes).hexdigest() != commit["report_sha256"]):
            raise ValueError("Publication report does not match commit marker")
        output = output_path.read_bytes()
        ids = _prediction_ids(output)
        output_hash = hashlib.sha256(output).hexdigest()
        if (output_hash != commit["output_sha256"] or output_hash != report["output_sha256"]
                or ("submission_sha256" in report and report["submission_sha256"] != output_hash)):
            raise ValueError("Publication prediction hash mismatch")
        if (type(report["samples"]) is not int or type(commit["samples"]) is not int
                or report["samples"] != len(ids) or commit["samples"] != len(ids)):
            raise ValueError("Publication sample count mismatch")
        if expected_ids is not None and ids != list(expected_ids):
            raise ValueError("Publication input ID order/count mismatch")
        if marker.read_bytes() != before:
            raise ValueError("Publication changed during validation")
        return report
    except (OSError, KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(f"Invalid inference publication: {error}") from error


def publish_prediction_report(staged_output, output_path, report_path, report, expected_ids):
    """Invalidate old certification before replacing predictions; commit last.

    On interruption, leave a pending marker and an explicitly incomplete report
    rather than an old cold-start report certifying new predictions. A subsequent
    recovery can safely republish the complete pair from its verified journal.
    """
    staged_output, output_path, report_path = map(Path, (staged_output, output_path, report_path))
    marker = marker_path(report_path)
    if len({p.resolve() for p in (staged_output, output_path, report_path, marker)}) != 4:
        raise ValueError("Publication staging, prediction, report and marker paths must differ")
    ids = _prediction_ids(staged_output.read_bytes())
    if ids != list(expected_ids) or report["samples"] != len(ids):
        raise ValueError("Publication output IDs/count do not match input and report")
    with staged_output.open("r+b") as handle:
        os.fsync(handle.fileno())
    output_hash = file_digest(staged_output)
    transaction = uuid.uuid4().hex
    publication = {"version": 1, "id": transaction, "state": "complete"}
    report = {**report, "output_sha256": output_hash, "publication": publication}
    if "submission_sha256" in report and report["submission_sha256"] != output_hash:
        raise ValueError("Publication report submission hash mismatch")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending = {**publication, "state": "publishing"}
    atomic_json(marker, pending)
    atomic_json(report_path, {"publication": pending, "complete": False})
    os.replace(staged_output, output_path)
    atomic_json(report_path, report)
    atomic_json(marker, {**publication, "output_sha256": output_hash, "samples": len(ids),
                         "report_sha256": file_digest(report_path)})
    return validate_publication(output_path, report_path, ids)
