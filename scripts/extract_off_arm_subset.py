#!/usr/bin/env python3
"""Extract the "conditions off" arm for the T2 stratified screening subset.

Cycle-2 task T2 compares the latest C Day5 + D Day4 weights under two
conditions -- off (no condition injected, the deployed default) vs
emotion-only -- on the 160-row frozen subset. The off arm needs no new
inference at all: the D Day4 v2 full-val predictions already contain one
response per eval id under exactly the deployed condition (off, empty
profile, max_new_tokens 256), so the arm is extracted by id instead of
recomputed. Recomputing it would burn GPU budget and, worse, risk a second
baseline that no longer matches the historical one.

The join is strict: every subset id must exist in the source predictions,
and duplicate ids on either side are a data bug rather than something to
paper over. Predictions outside the subset are expected -- the source is a
superset -- and are only counted, never copied.

Same inputs, same bytes out: rows are copied verbatim from the source and
written sorted by id, so the extraction is reproducible and auditable
against the manifest digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl, write_jsonl  # noqa: E402

DEFAULT_SELECTION = PROJECT / "results/response_gen/t2_selection_input.jsonl"
DEFAULT_PREDICTIONS = PROJECT / "results/response_gen/d4_2ep_val_full.jsonl"
DEFAULT_OUTPUT = PROJECT / "results/response_gen/t2_condition_off.jsonl"
DEFAULT_MANIFEST = PROJECT / "results/response_gen/t2_condition_off_manifest.json"

ARM_NOTE = (
    "Conditions-off arm for the T2 subset, extracted by id from the D Day4 v2 "
    "full-val predictions (adapter digest b3e9017e..., conditions off, empty "
    "profile, max_new_tokens 256 -- the deployed configuration). Reused, not "
    "recomputed: no GPU was spent and the responses are byte-identical to the "
    "historical baseline."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_by_id(rows: Sequence[Mapping[str, Any]], id_field: str, source: Path) -> dict[str, dict[str, Any]]:
    """Index rows by their id field, refusing duplicates and missing ids."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get(id_field)
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{source}: row without a string {id_field}: {row!r}")
        if row_id in by_id:
            raise ValueError(f"{source}: duplicate {id_field} {row_id!r}")
        by_id[row_id] = dict(row)
    return by_id


def extract_off_arm(
    selection_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    selection_path: Path = DEFAULT_SELECTION,
    predictions_path: Path = DEFAULT_PREDICTIONS,
) -> list[dict[str, Any]]:
    """The off-arm predictions for the subset, sorted by id.

    The selection carries its ids in ``sample_id`` (the response-split
    convention); the predictions carry them in ``id`` (the inference-output
    convention). Both hold the same ``pp:[...]`` prediction-point id, which
    is what the join runs on.
    """
    selection_by_id = _index_by_id(selection_rows, "sample_id", selection_path)
    predictions_by_id = _index_by_id(prediction_rows, "id", predictions_path)
    missing = sorted(set(selection_by_id) - set(predictions_by_id))
    if missing:
        raise ValueError(
            f"{len(missing)} subset ids have no prediction in {predictions_path} "
            f"(first: {missing[0]!r}); the off arm cannot be partially reused"
        )
    extracted = []
    for row_id in sorted(selection_by_id):
        prediction = predictions_by_id[row_id]
        if "response_text" not in prediction:
            raise ValueError(f"prediction {row_id!r} has no response_text")
        extracted.append({"id": row_id, "response_text": prediction["response_text"]})
    return extracted


def build_manifest(
    extracted: Sequence[Mapping[str, Any]],
    selection_path: Path,
    predictions_path: Path,
    selection_row_count: int,
    prediction_row_count: int,
    command: str,
) -> dict[str, Any]:
    selection_ids = sorted({row["id"] for row in extracted})
    return {
        "version": "t2_condition_off_v1",
        "purpose": ARM_NOTE,
        "arm": "off",
        "sources": {
            "subset": {
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
                "record_count": selection_row_count,
            },
            "predictions": {
                "path": str(predictions_path),
                "sha256": sha256_file(predictions_path),
                "record_count": prediction_row_count,
            },
        },
        "result": {
            "record_count": len(extracted),
            "predictions_outside_subset": prediction_row_count - len(extracted),
            "first_id": selection_ids[0] if selection_ids else None,
            "last_id": selection_ids[-1] if selection_ids else None,
        },
        "command": command,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    selection_rows = read_jsonl(args.selection)
    prediction_rows = read_jsonl(args.predictions)
    extracted = extract_off_arm(selection_rows, prediction_rows, args.selection, args.predictions)

    write_jsonl(args.output, extracted)
    manifest = build_manifest(
        extracted,
        args.selection,
        args.predictions,
        len(selection_rows),
        len(prediction_rows),
        "python " + shlex.join(["scripts/extract_off_arm_subset.py", *(argv or [])]),
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"off arm: {len(extracted)} rows extracted from {len(prediction_rows)} "
        f"predictions (subset {len(selection_rows)} rows) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
