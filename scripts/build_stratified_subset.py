#!/usr/bin/env python3
"""Freeze the T2 fixed stratified screening subset for the D condition arms.

Cycle-2 task T2 compares the latest C Day5 + D Day4 weights under two
conditions (off vs emotion-only) on a small fixed input before any full-val
run. This script picks that input once and writes it down, so the comparison
cannot drift towards whatever happens to look good: the subset is chosen by
stratification over gold labels, not by any model output.

Stratification (3x2x2 = 12 cells) covers the axes the local adoption rules
demand -- short/mid/long history, empty vs non-empty gold profile, major vs
rare gold emotion:

  * history length: len(context_turns) <= 4 / 5..12 / > 12;
  * gold profile: personality_traits, interests and style all empty / any
    non-empty (an all-empty profile is a legal label, not missing data);
  * gold emotion: major = the five best-supported labels, rare = the rest.

Quotas follow each cell's share of the pool (largest remainder), with a floor
of 2 per non-empty cell so rare corners survive. Each conversation_id
contributes at most one row -- the subset is an eval input choice, not a
training split, but repeating a conversation would spend the budget on one
dialogue and bias any per-conversation reading. Gold labels are used for
stratification only and never enter a model input: the selected rows are
copied verbatim from the response split, which
``scripts/run_response_inference.py`` consumes directly.

Same seed, same inputs, same bytes out: within each cell the candidates are
ordered by sample_id and then shuffled with ``random.Random(seed)``, so the
frozen list is reproducible without depending on file order or wall-clock
state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.io import read_jsonl, write_jsonl  # noqa: E402

LENGTH_SHORT_MAX = 4
LENGTH_MID_MAX = 12
PROFILE_FIELDS = ("personality_traits", "interests", "style")
MAJOR_EMOTION_TOP_K = 5

DEFAULT_TARGET_SIZE = 160
DEFAULT_SEED = 20261022
DEFAULT_MIN_PER_CELL = 2
DEFAULT_RESPONSE_INPUT = PROJECT / "data/processed/b1_full_v1/response_val.jsonl"
DEFAULT_LABEL_INPUT = PROJECT / "data/processed/b1_full_v1/classification_val.jsonl"
DEFAULT_OUTPUT = PROJECT / "results/response_gen/t2_selection_input.jsonl"
DEFAULT_MANIFEST = PROJECT / "results/response_gen/t2_selection_manifest.json"

CellKey = tuple[str, str, str]

LENGTH_ORDER = {"short": 0, "mid": 1, "long": 2}
PROFILE_ORDER = {"empty": 0, "non_empty": 1}
EMOTION_ORDER = {"major": 0, "rare": 1}

GOLD_LABELS_NOTE = (
    "Gold emotion_label/user_profile from the classification split are used "
    "for stratification only and never enter any model input; selected rows "
    "are copied verbatim from the response split."
)
SELECTION_RULES = (
    "Target rows are allocated across the 12 cells in proportion to cell size "
    "(largest remainder, ties broken by larger cell then canonical cell "
    "order), with a floor of 2 per non-empty cell. Within each cell, "
    "candidates are ordered by sample_id and shuffled with random.Random(seed) "
    "before the quota is filled, skipping any conversation_id already used "
    "elsewhere in the selection (at most one row per conversation). If "
    "de-duplication leaves the selection short of the target, a second round "
    "tops it up from the largest cells, again skipping used conversations; if "
    "the distinct conversation pool itself is smaller than the target the "
    "script fails instead of padding. Output rows are sorted by sample_id."
)


def length_bucket(turn_count: int) -> str:
    """Bucket a context by its number of turns (4/5 and 12/13 are boundaries)."""
    if turn_count <= LENGTH_SHORT_MAX:
        return "short"
    if turn_count <= LENGTH_MID_MAX:
        return "mid"
    return "long"


def profile_bucket(user_profile: Any) -> str:
    """Empty means all three profile dimensions are empty lists, a legal label."""
    if not isinstance(user_profile, Mapping):
        return "empty"
    for field in PROFILE_FIELDS:
        if user_profile.get(field):
            return "non_empty"
    return "empty"


def emotion_bucket(emotion_label: Any, major_labels: Any) -> str:
    """Major labels are the best-supported ones; everything else is rare."""
    return "major" if emotion_label in major_labels else "rare"


def cell_key(response_row: Mapping[str, Any], label_row: Mapping[str, Any], major_labels: Any) -> CellKey:
    """The (length, profile, emotion) stratum of one prediction point."""
    turns = response_row.get("context_turns")
    if not isinstance(turns, list):
        raise ValueError(f"row {response_row.get('sample_id')!r} has no context_turns list")
    return (
        length_bucket(len(turns)),
        profile_bucket(label_row.get("user_profile")),
        emotion_bucket(label_row.get("emotion_label"), major_labels),
    )


def canonical_key(key: CellKey) -> tuple[int, int, int]:
    """A fixed cell order so allocations and rounds never depend on dict order."""
    try:
        return (LENGTH_ORDER[key[0]], PROFILE_ORDER[key[1]], EMOTION_ORDER[key[2]])
    except KeyError as error:  # pragma: no cover - guards hand-built cells
        raise ValueError(f"unknown cell bucket in {key!r}") from error


def major_emotion_labels(
    label_rows: Sequence[Mapping[str, Any]], top_k: int = MAJOR_EMOTION_TOP_K
) -> list[str]:
    """The `top_k` best-supported gold emotion labels, ties broken by name."""
    counts: Counter[str] = Counter()
    for row in label_rows:
        label = row.get("emotion_label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"label row {row.get('sample_id')!r} has no emotion_label string")
        counts[label] += 1
    return [label for label, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


def allocate_quotas(
    cell_sizes: Mapping[CellKey, int], target_size: int, min_per_cell: int = DEFAULT_MIN_PER_CELL
) -> dict[CellKey, int]:
    """Largest-remainder allocation of `target_size` across non-empty cells.

    Empty cells get zero. A non-empty cell never drops below `min_per_cell`;
    if the floors alone exceed the target the configuration is impossible and
    the caller is told rather than silently under-filling the pool.
    """
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    quotas: dict[CellKey, int] = {key: 0 for key, size in cell_sizes.items() if size <= 0}
    non_empty = {key: int(size) for key, size in cell_sizes.items() if size > 0}
    if not non_empty:
        raise ValueError("no non-empty cells to allocate")
    if min_per_cell * len(non_empty) > target_size:
        raise ValueError(
            f"floor of {min_per_cell} over {len(non_empty)} non-empty cells needs "
            f"{min_per_cell * len(non_empty)} rows but the target is {target_size}"
        )
    total = sum(non_empty.values())
    ideals = {key: target_size * size / total for key, size in non_empty.items()}
    remainders: list[tuple[float, int, CellKey]] = []
    allocated = 0
    for key, ideal in ideals.items():
        floor = int(ideal // 1)
        base = max(min_per_cell, floor)
        quotas[key] = base
        allocated += base
        if base == floor:  # a floor-lifted cell already got more than its share
            remainders.append((ideal - floor, non_empty[key], key))
    leftover = target_size - allocated
    if leftover > 0:
        if not remainders:
            raise ValueError(
                f"cannot distribute {leftover} leftover rows: every cell sits at the floor"
            )
        remainders.sort(key=lambda item: (-item[0], -item[1], item[2]))
        # One pass suffices: leftover <= number of remainder-bearing cells.
        for _, _, key in remainders[:leftover]:
            quotas[key] += 1
    elif leftover < 0:
        # Floors over-allocated; take the rows back from the cells that were
        # least owed a remainder, never dropping a cell below its floor.
        excess = -leftover
        trim_order = sorted(
            (key for key in non_empty if quotas[key] > min_per_cell),
            key=lambda key: (ideals[key] - int(ideals[key] // 1), non_empty[key], key),
        )
        for key in trim_order:
            while excess > 0 and quotas[key] > min_per_cell:
                quotas[key] -= 1
                excess -= 1
            if excess == 0:
                break
        if excess:
            raise ValueError("cannot trim to the target without breaking the per-cell floor")
    return quotas


def build_cells(
    response_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    major_labels: Any,
) -> dict[CellKey, list[dict[str, Any]]]:
    """Join response rows with their gold labels and bucket both sides.

    The join is strict in both directions: a response row without a gold
    label, a label without a response row, or a duplicated sample_id is a
    data bug, not something to paper over by dropping the row.
    """
    label_by_id: dict[str, Mapping[str, Any]] = {}
    for row in label_rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("label row without a string sample_id")
        if sample_id in label_by_id:
            raise ValueError(f"duplicate label sample_id {sample_id!r}")
        label_by_id[sample_id] = row
    cells: dict[CellKey, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for row in response_rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("response row without a string sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate response sample_id {sample_id!r}")
        seen.add(sample_id)
        conversation_id = row.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"response row {sample_id!r} has no conversation_id string")
        label = label_by_id.get(sample_id)
        if label is None:
            raise ValueError(f"no gold label for response sample_id {sample_id!r}")
        cells.setdefault(cell_key(row, label, major_labels), []).append(dict(row))
    unmatched = sorted(set(label_by_id) - seen)
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} gold label rows have no response row "
            f"(first: {unmatched[0]!r}); the two splits must cover the same ids"
        )
    return cells


@dataclass
class SubsetSelection:
    """A frozen selection plus the numbers needed to audit it."""

    rows: list[dict[str, Any]]
    cell_sizes: dict[CellKey, int]
    quotas: dict[CellKey, int]
    cell_counts: dict[CellKey, int]
    conversation_ids: list[str]

    @property
    def actual_size(self) -> int:
        return len(self.rows)


def _select(
    cells: Mapping[CellKey, Sequence[dict[str, Any]]],
    target_size: int,
    seed: int,
    min_per_cell: int,
) -> SubsetSelection:
    ordered_keys = sorted(cells, key=canonical_key)
    sizes = {key: len(cells[key]) for key in ordered_keys}
    quotas = allocate_quotas(sizes, target_size, min_per_cell)
    conversations = {
        row["conversation_id"] for key in ordered_keys for row in cells[key]
    }
    if len(conversations) < target_size:
        raise ValueError(
            f"only {len(conversations)} distinct conversations are available but the "
            f"target is {target_size}; refusing to pad the subset with duplicate "
            "conversations (check --target against the input pool)"
        )
    rng = random.Random(seed)
    shuffled: dict[CellKey, list[dict[str, Any]]] = {}
    for key in ordered_keys:
        rows = sorted(cells[key], key=lambda row: row["sample_id"])
        rng.shuffle(rows)
        shuffled[key] = rows

    used: set[str] = set()
    picked: list[tuple[CellKey, dict[str, Any]]] = []

    def take(key: CellKey) -> None:
        for row in shuffled[key]:
            conversation_id = row["conversation_id"]
            if conversation_id in used:
                continue
            picked.append((key, row))
            used.add(conversation_id)
            return

    for key in ordered_keys:  # round 1: honour each cell's quota
        for _ in range(quotas[key]):
            if len(picked) >= target_size:
                break
            before = len(picked)
            take(key)
            if len(picked) == before:
                break  # cell ran out of unused conversations

    if len(picked) < target_size:  # round 2: top up from the largest cells
        for key in sorted(ordered_keys, key=lambda item: (-sizes[item], canonical_key(item))):
            while len(picked) < target_size:
                before = len(picked)
                take(key)
                if len(picked) == before:
                    break

    picked.sort(key=lambda item: item[1]["sample_id"])
    rows = [row for _, row in picked]
    counts = {key: 0 for key in ordered_keys}
    for key, _ in picked:
        counts[key] += 1
    return SubsetSelection(
        rows=rows,
        cell_sizes=sizes,
        quotas=quotas,
        cell_counts=counts,
        conversation_ids=sorted(used),
    )


def select_subset(
    cells: Mapping[CellKey, Sequence[dict[str, Any]]],
    target_size: int,
    seed: int,
    min_per_cell: int = DEFAULT_MIN_PER_CELL,
) -> list[dict[str, Any]]:
    """The selected rows alone; see `_select` for the full audited result."""
    return _select(cells, target_size, seed, min_per_cell).rows


def build_subset(
    response_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    target_size: int = DEFAULT_TARGET_SIZE,
    seed: int = DEFAULT_SEED,
    min_per_cell: int = DEFAULT_MIN_PER_CELL,
    major_labels: Sequence[str] | None = None,
) -> SubsetSelection:
    """End-to-end selection: derive major labels, join, stratify, select."""
    if major_labels is None:
        major_labels = major_emotion_labels(label_rows)
    cells = build_cells(response_rows, label_rows, frozenset(major_labels))
    return _select(cells, target_size, seed, min_per_cell)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    selection: SubsetSelection,
    major_labels: Sequence[str],
    rare_labels: Sequence[str],
    target_size: int,
    seed: int,
    min_per_cell: int,
    response_input: Path,
    label_input: Path,
    response_row_count: int,
    label_row_count: int,
    command: str,
) -> dict[str, Any]:
    cells = []
    for key in sorted(selection.cell_sizes, key=canonical_key):
        cells.append(
            {
                "history_length": key[0],
                "gold_profile": key[1],
                "emotion": key[2],
                "size": selection.cell_sizes[key],
                "quota": selection.quotas[key],
                "selected": selection.cell_counts[key],
            }
        )
    return {
        "version": "t2_stratified_subset_v1",
        "purpose": (
            "Fixed stratified screening subset for the cycle-2 D condition arms "
            "(conditions off vs emotion-only), frozen before any new results were seen."
        ),
        "seed": seed,
        "target_size": target_size,
        "actual_size": selection.actual_size,
        "target_reached": selection.actual_size == target_size,
        "min_per_cell": min_per_cell,
        "sources": {
            "response_val": {
                "path": str(response_input),
                "sha256": sha256_file(response_input),
                "record_count": response_row_count,
            },
            "classification_val": {
                "path": str(label_input),
                "sha256": sha256_file(label_input),
                "record_count": label_row_count,
            },
        },
        "stratification": {
            "history_length": {
                "short": f"len(context_turns) <= {LENGTH_SHORT_MAX}",
                "mid": f"{LENGTH_SHORT_MAX + 1} <= len(context_turns) <= {LENGTH_MID_MAX}",
                "long": f"len(context_turns) > {LENGTH_MID_MAX}",
            },
            "gold_profile": {
                "empty": "personality_traits, interests and style are all empty lists",
                "non_empty": "at least one of the three dimensions is non-empty",
            },
            "emotion": {"major": list(major_labels), "rare": list(rare_labels)},
        },
        "gold_labels_note": GOLD_LABELS_NOTE,
        "selection_rules": SELECTION_RULES,
        "cells": cells,
        "conversations": {
            "distinct": len(selection.conversation_ids),
            "rows": selection.actual_size,
            "max_rows_per_conversation": 1,
        },
        "command": command,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the T2 fixed stratified screening subset (D condition arms)."
    )
    parser.add_argument("--response-input", type=Path, default=DEFAULT_RESPONSE_INPUT)
    parser.add_argument("--label-input", type=Path, default=DEFAULT_LABEL_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-per-cell", type=int, default=DEFAULT_MIN_PER_CELL)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    response_rows = read_jsonl(args.response_input)
    label_rows = read_jsonl(args.label_input)
    if not response_rows or not label_rows:
        raise ValueError("both input splits must be non-empty")
    major_labels = major_emotion_labels(label_rows)
    rare_labels = sorted(
        {row["emotion_label"] for row in label_rows} - set(major_labels)
    )
    selection = build_subset(
        response_rows,
        label_rows,
        target_size=args.target,
        seed=args.seed,
        min_per_cell=args.min_per_cell,
        major_labels=major_labels,
    )
    if selection.actual_size < args.target:
        # Unreachable while the conversation pool check passes; recorded rather
        # than silently reported as a full target.
        print(
            f"WARNING: selected {selection.actual_size} of {args.target} rows after "
            "conversation de-duplication; recording the actual size",
            file=sys.stderr,
        )
    command = shlex.join(
        [
            "python",
            "scripts/build_stratified_subset.py",
            "--response-input",
            str(args.response_input),
            "--label-input",
            str(args.label_input),
            "--output",
            str(args.output),
            "--manifest",
            str(args.manifest),
            "--target",
            str(args.target),
            "--seed",
            str(args.seed),
            "--min-per-cell",
            str(args.min_per_cell),
        ]
    )
    manifest = build_manifest(
        selection,
        major_labels=major_labels,
        rare_labels=rare_labels,
        target_size=args.target,
        seed=args.seed,
        min_per_cell=args.min_per_cell,
        response_input=args.response_input,
        label_input=args.label_input,
        response_row_count=len(response_rows),
        label_row_count=len(label_rows),
        command=command,
    )
    write_jsonl(args.output, selection.rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"PASS: {selection.actual_size} rows -> {args.output} "
        f"(manifest: {args.manifest}; {len(selection.conversation_ids)} conversations)",
        flush=True,
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
