#!/usr/bin/env python3
"""CPU-only profile summary and ID-aligned conversation paired comparison."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.b_t1_profile_diagnostics import extended_profile_metrics, length_bucket, ratio
from scripts.b_validate_submission import sha256_file
from src.common.io import read_jsonl
from src.data.schema import EMOTIONS, PROFILE_LABELS, validate_classification_labels

BUCKETS = ("short_1_5", "medium_6_15", "long_16_plus")
DEFINITIONS = {
    "exact": "Set equality; full profile requires all three dimensions correct.",
    "macro_exact": "Equal mean of the three dimension exact accuracies (T1 primary).",
    "micro": "Pool label TP/FP/FN within dimension; global_micro pools all dimensions.",
    "label_macro": "Mean over each dimension's fixed official labels, including zero support.",
    "dimension_macro": "Equal mean over three dimensions, separately for micro and label macro PRF.",
    "zero_denominator": 0,
    "support": "samples is row count; support is gold positive label occurrences.",
    "conversation_score": "Correct dimension sets / (3 * prediction points) within each conversation.",
    "paired": "Compare integer correct-dimension totals within the same conversation; equal is tie. "
              "Mean paired delta equally weights conversations, not prediction points.",
    "slices": "History message count includes target user: 1..5 / 6..15 / >=16. "
              "Gold empty means all three dimensions empty. Emotion is reference gold. "
              "Missing metadata is unavailable; manifest joins by exact sample ID only.",
}


def index_rows(rows, name, *, profiles=True):
    """Validate before indexing so duplicate IDs can never be overwritten."""
    if not rows:
        raise ValueError(f"{name}: empty input")
    result = {}
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{name}:{number}: row must be an object")
        key = row.get("sample_id", row.get("id"))
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name}:{number}: invalid sample ID")
        if "sample_id" in row and "id" in row and row["id"] != key:
            raise ValueError(f"{name}:{number}: conflicting sample_id/id")
        if key in result:
            raise ValueError(f"{name}: duplicate ID: {key}")
        if "conversation_id" in row:
            cid = row["conversation_id"]
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError(f"{name}:{key}: invalid conversation_id")
        if profiles:
            profile = row.get("user_profile")
            # Guard unhashable labels before calling the shared set-based validator.
            if isinstance(profile, dict):
                for dim, values in profile.items():
                    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                        raise ValueError(f"{name}:{key}: invalid user_profile.{dim} list/labels")
            try:
                validate_classification_labels("neutral", profile)
            except ValueError as exc:
                raise ValueError(f"{name}:{key}: {exc}") from exc
        result[key] = row
    return result


def coverage(reference, predictions):
    report = {}
    for name, rows in predictions.items():
        missing, extra = sorted(reference.keys() - rows.keys()), sorted(rows.keys() - reference.keys())
        report[name] = {"rows": len(rows), "unique_ids": len(rows), "missing_ids": missing,
                        "extra_ids": extra, "duplicate_ids": [], "complete": not (missing or extra)}
    if any(not v["complete"] for v in report.values()):
        raise ValueError("ID coverage mismatch: " + json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


def reference_metadata(reference, manifest):
    """Join only authoritative metadata; reject conflicts instead of guessing."""
    if manifest.keys() - reference.keys():
        raise ValueError(f"manifest extra IDs: {sorted(manifest.keys() - reference.keys())}")
    metadata = {}
    for key, row in sorted(reference.items()):
        cid = row.get("conversation_id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"reference:{key}: conversation_id required")
        joined = manifest.get(key, {})
        if joined and joined.get("conversation_id") != cid:
            raise ValueError(f"manifest:{key}: conversation_id mismatch")
        histories = [row[field] for field in ("context_turns", "history") if field in row]
        if any(not isinstance(h, list) or not h for h in histories):
            raise ValueError(f"reference:{key}: history must be a non-empty list")
        if len(histories) == 2 and histories[0] != histories[1]:
            raise ValueError(f"reference:{key}: conflicting context_turns/history")
        buckets = [length_bucket(len(histories[0]))] if histories else []
        buckets.extend(r["history_bucket"] for r in (row, joined) if "history_bucket" in r)
        if any(not isinstance(b, str) or b not in BUCKETS for b in buckets) or len(set(buckets)) > 1:
            raise ValueError(f"reference/manifest:{key}: invalid or conflicting history_bucket")
        emotions = [r["emotion_label"] for r in (row, joined) if "emotion_label" in r]
        if any(not isinstance(e, str) or e not in EMOTIONS for e in emotions) or len(set(emotions)) > 1:
            raise ValueError(f"reference/manifest:{key}: invalid or conflicting emotion_label")
        mask = "".join("1" if row["user_profile"][d] else "0" for d in PROFILE_LABELS)
        if "profile_nonempty_mask" in joined and joined["profile_nonempty_mask"] != mask:
            raise ValueError(f"manifest:{key}: profile_nonempty_mask mismatch")
        metadata[key] = {"conversation_id": cid, "history": buckets[0] if buckets else "unavailable",
                         "emotion": emotions[0] if emotions else "unavailable",
                         "gold_profile": "nonempty" if "1" in mask else "empty"}
    return metadata


def empty_behavior(gold, predicted):
    result = dict.fromkeys(("gold_empty_pred_empty", "gold_empty_pred_nonempty",
                            "gold_nonempty_pred_empty", "gold_nonempty_pred_nonempty"), 0)
    for a, b in zip(gold, predicted):
        result[f"gold_{'nonempty' if a else 'empty'}_pred_{'nonempty' if b else 'empty'}"] += 1
    return result


def summarize(reference, predicted):
    """Reuse T1 exact and PRF functions; add support, full exact and empty behavior."""
    gold = [r["user_profile"] for r in reference]
    guesses = [r["user_profile"] for r in predicted]
    result = extended_profile_metrics(gold, guesses)
    n = len(gold)
    for dim in PROFILE_LABELS:
        result["per_dimension"][dim].update(
            samples=n, support=result["per_dimension"][dim]["gold_positive_count"],
            gold_nonempty_rate=ratio(sum(bool(p[dim]) for p in gold), n),
            empty_behavior=empty_behavior([bool(p[dim]) for p in gold], [bool(p[dim]) for p in guesses]))
    nonempty = sum(any(p.values()) for p in guesses)
    result.update(samples=n, unique_conversations=len({r["conversation_id"] for r in reference}),
                  full_profile_exact_match=ratio(sum(all(set(a[d]) == set(b[d]) for d in PROFILE_LABELS)
                                                       for a, b in zip(gold, guesses)), n),
                  predicted_all_empty_rate=ratio(n - nonempty, n),
                  predicted_any_nonempty_rate=ratio(nonempty, n),
                  any_profile_empty_behavior=empty_behavior([any(p.values()) for p in gold],
                                                           [any(p.values()) for p in guesses]))
    return result


def paired_stats(reference, baseline, candidate):
    """Each conversation contributes once; use integer totals for exact tie decisions."""
    def hits(g, p, dims):
        return sum(set(g[d]) == set(p[d]) for d in dims)

    sample = {}
    for name, dims in [("full_profile", tuple(PROFILE_LABELS))] + [(d, (d,)) for d in PROFILE_LABELS]:
        counts = dict.fromkeys(("identical", "changed", "baseline_correct_candidate_wrong",
                               "baseline_wrong_candidate_correct", "both_correct", "both_wrong"), 0)
        for r, a, b in zip(reference, baseline, candidate):
            g, p, q = r["user_profile"], a["user_profile"], b["user_profile"]
            counts["identical" if hits(p, q, dims) == len(dims) else "changed"] += 1
            left, right = hits(g, p, dims) == len(dims), hits(g, q, dims) == len(dims)
            status = ("both_correct" if left and right else "both_wrong" if not left and not right
                      else "baseline_correct_candidate_wrong" if left else "baseline_wrong_candidate_correct")
            counts[status] += 1
        sample[name] = counts
    groups = defaultdict(list)
    for r, a, b in zip(reference, baseline, candidate):
        groups[r["conversation_id"]].append((r, a, b))
    details, deltas = [], []
    for cid, rows in sorted(groups.items()):
        left = sum(hits(r["user_profile"], a["user_profile"], PROFILE_LABELS) for r, a, _ in rows)
        right = sum(hits(r["user_profile"], b["user_profile"], PROFILE_LABELS) for r, _, b in rows)
        denominator = 3 * len(rows)
        delta = Fraction(right - left, denominator)
        deltas.append(delta)
        details.append({"conversation_id": cid, "prediction_points": len(rows),
                        "sample_ids": sorted(r["sample_id"] for r, _, _ in rows),
                        "baseline_score": left / denominator, "candidate_score": right / denominator,
                        "delta": float(delta), "outcome": "win" if right > left else "loss" if right < left else "tie"})
    wins, losses = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
    n = len(details)
    return {"samples": sample, "conversations": {"total": n, "wins": wins, "ties": n - wins - losses,
            "losses": losses, "win_rate": ratio(wins, n), "loss_rate": ratio(losses, n),
            "mean_paired_delta": float(sum(deltas, Fraction()) / n), "details": details}}


def evaluate_rows(reference, prediction, *, baseline=None, manifest=None):
    """Validate all inputs, sort by ID, then evaluate; metadata never comes from predictions."""
    ref = index_rows(reference, "reference")
    pred = index_rows(prediction, "candidate" if baseline is not None else "prediction")
    inputs = {"candidate" if baseline is not None else "prediction": pred}
    if baseline is not None:
        inputs["baseline"] = index_rows(baseline, "baseline")
    id_coverage = coverage(ref, inputs)
    meta = reference_metadata(ref, index_rows(manifest, "manifest", profiles=False) if manifest is not None else {})
    for name, rows in inputs.items():
        for key, row in rows.items():
            if "conversation_id" in row and row["conversation_id"] != meta[key]["conversation_id"]:
                raise ValueError(f"{name}:{key}: conversation_id mismatch with reference")
    keys = sorted(ref)
    gold = [dict(ref[k], sample_id=k) for k in keys]
    aligned = {name: [rows[k] for k in keys] for name, rows in inputs.items()}

    def metrics(indices):
        r = [gold[i] for i in indices]
        p = {name: [rows[i] for i in indices] for name, rows in aligned.items()}
        value = {"samples": len(r), "unique_conversations": len({row["conversation_id"] for row in r}),
                 "summary": {name: summarize(r, rows) for name, rows in p.items()}}
        if baseline is not None:
            value["paired"] = paired_stats(r, p["baseline"], p["candidate"])
        return value

    report = {"version": "b_profile_eval_v1", "passed": True, "definitions": DEFINITIONS,
              "id_coverage": id_coverage, "conversation_mapping_source": "reference.conversation_id",
              **metrics(range(len(keys)))}
    slices = defaultdict(list)
    for i, key in enumerate(keys):
        for field in ("history", "gold_profile", "emotion"):
            slices[f"{field}:{meta[key][field]}"].append(i)
    report["slices"] = {name: metrics(indices) for name, indices in sorted(slices.items())}
    return report


def evaluate_files(reference, prediction, *, baseline=None, manifest=None):
    paths = {"reference": reference, "prediction": prediction}
    if baseline is not None:
        paths["baseline"] = baseline
    if manifest is not None:
        paths["manifest"] = manifest
    fingerprints = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}
    try:
        rows = {name: read_jsonl(path) for name, path in paths.items()}
        report = evaluate_rows(rows["reference"], rows["prediction"], baseline=rows.get("baseline"),
                               manifest=rows.get("manifest"))
    except ValueError as exc:
        report = {"version": "b_profile_eval_v1", "passed": False, "error": str(exc)}
    return dict(report, inputs=fingerprints)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("summary", "compare"):
        command = sub.add_parser(mode)
        command.add_argument("--reference", type=Path, required=True)
        command.add_argument("--prediction" if mode == "summary" else "--candidate", dest="prediction", type=Path, required=True)
        if mode == "compare":
            command.add_argument("--baseline", type=Path, required=True)
        command.add_argument("--manifest", type=Path, help="Optional metadata join; does not select reference rows")
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate_files(args.reference, args.prediction, baseline=getattr(args, "baseline", None), manifest=args.manifest)
    except OSError as exc:
        report = {"version": "b_profile_eval_v1", "passed": False, "error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output), "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
