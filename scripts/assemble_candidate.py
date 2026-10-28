#!/usr/bin/env python3
"""Assemble an offline candidate from existing predictions without model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.b_validate_submission import duplicate_ids, row_ids, validate_rows
from scripts.extract_off_arm_subset import _index_by_id
from src.common.io import read_jsonl
from src.data.schema import validate_classification_labels, validate_prediction


def assemble_rows(inputs, responses, emotions, profiles):
    """Require exact response coverage; classification sources may be supersets.

    Output follows input order. Source reordering is reported and repaired by ID,
    never by positional zip. All sources, including unused rows, reject bad IDs.
    """
    if not inputs:
        raise ValueError("input contains no rows")
    expected = row_ids(inputs)
    if None in expected or duplicate_ids(expected):
        raise ValueError("input IDs must be nonempty strings and unique")
    target = set(expected)
    indexes, alignment = {}, {}
    for name, rows in (("response", responses), ("emotion", emotions), ("profile", profiles)):
        index = _index_by_id(rows, "id", Path(name))
        missing, extra = sorted(target - index.keys()), sorted(index.keys() - target)
        if missing or (name == "response" and extra):
            raise ValueError(f"{name}: missing={len(missing)} extra={len(extra)}; ID coverage mismatch")
        selected_order = [r["id"] for r in rows if r["id"] in target]
        alignment[name] = {
            "source_rows": len(rows), "selected_rows": len(expected),
            "unused_source_rows": len(extra), "missing_ids": missing,
            "order_mismatch_count": sum(a != b for a, b in zip(expected, selected_order)),
        }
        indexes[name] = index
    result = []
    for source in inputs:
        key = source["id"]
        emotion, profile = indexes["emotion"][key], indexes["profile"][key]
        for name, row in (("emotion", emotion), ("profile", profile)):
            if "conversation_id" in row and row["conversation_id"] != source["conversation_id"]:
                raise ValueError(f"{name}: conversation mismatch for {key}")
        # Read only predicted fields; reference labels are never copied.
        validate_classification_labels(emotion.get("emotion_label"), profile.get("user_profile"))
        response = indexes["response"][key]
        row = {"id": key, "response_text": response["response_text"],
               "emotion_label": emotion["emotion_label"], "user_profile": profile["user_profile"],
               "memory_refs": response.get("memory_refs", [])}
        validate_prediction(row, key)
        if not row["response_text"].strip():
            raise ValueError(f"empty response for {key}")
        result.append(row)
    checks = validate_rows(inputs, result)
    if not checks["passed"]:
        raise ValueError(f"candidate validation failed: {checks['issues']}")
    return result, {"alignment": alignment, "validation": checks,
                    "response_text_unchanged": all(r["response_text"] == indexes["response"][r["id"]]["response_text"] for r in result)}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": path.as_posix(), "sha256": digest(data),
            "lf_sha256": digest(data.replace(b"\r\n", b"\n")), "bytes": len(data)}


def load_inputs(path):
    rows = read_jsonl(path)
    # Accept official inference input or B1 reference; discard gold fields.
    return [{"id": r["id"] if "id" in r else r["sample_id"],
             "conversation_id": r["conversation_id"],
             "target_user_turn_id": r["target_user_turn_id"],
             "history": r["history"] if "history" in r else r["context_turns"]} for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "responses", "emotions", "profiles", "response-report", "output", "manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--emotion-model", required=True)
    parser.add_argument("--profile-model", required=True)
    parser.add_argument("--source-record", type=Path, action="append", default=[])
    args = parser.parse_args()
    source_paths = [args.input, args.responses, args.emotions, args.profiles, args.response_report, *args.source_record]
    destinations = [args.output.resolve(), args.manifest.resolve()]
    if len(set(destinations)) != 2 or set(destinations) & {p.resolve() for p in source_paths}:
        raise ValueError("outputs must be distinct from each other and all inputs")
    for path in (args.output, args.manifest):
        if path.exists() or path.is_symlink():
            raise ValueError(f"refusing to overwrite existing output: {path}")
    before = [fingerprint(p) for p in source_paths]
    inputs = load_inputs(args.input)
    responses = read_jsonl(args.responses)
    report = json.loads(args.response_report.read_text(encoding="utf-8"))
    response_fp = before[1]
    if report.get("output_sha256") not in (response_fp["sha256"], response_fp["lf_sha256"]):
        raise ValueError("response report hash does not match predictions (raw or LF)")
    if report.get("samples") != len(responses) or report.get("condition_mode") != "none":
        raise ValueError("offline field replacement requires matching count and unconditioned responses")
    rows, checks = assemble_rows(inputs, responses, read_jsonl(args.emotions), read_jsonl(args.profiles))
    if before != [fingerprint(p) for p in source_paths]:
        raise ValueError("source files changed during assembly")
    payload = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
    manifest = {
        "version": "offline_candidate_assembly_v1", "artifact_kind": "offline_field_assembly",
        "formal_end_to_end_timing": False, "model_inference_performed": False,
        "response_condition_mode": "none", "classification_used_to_generate_response": False,
        "samples": len(rows), "sources": dict(zip(
            ["input", "response", "emotion", "profile", "response_report", *[f"record_{i}" for i in range(len(args.source_record))]], before)),
        "assembler_sha256": digest(Path(__file__).read_bytes()),
        "models": {"response": {"base": report.get("model_path"), "adapter": report.get("adapter_path")},
                   "emotion": args.emotion_model, "profile": args.profile_model},
        "model_identity_basis": "existing prediction reports and supplied model IDs; no checkpoint reload",
        "field_sources": {"id_and_order": "input", "response_text": "response", "emotion_label": "emotion",
                          "user_profile": "profile", "memory_refs": "response if present, otherwise empty (no memory bank)"},
        "output": {"path": args.output.as_posix(), "sha256": digest(payload)},
        "checks": checks,
    }
    for path in (args.output, args.manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(payload)
    with args.manifest.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"samples": len(rows), "output_sha256": digest(payload), "passed": True}))


if __name__ == "__main__":
    main()
