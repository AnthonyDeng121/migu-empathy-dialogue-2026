"""Unit tests for the T2 fixed stratified screening subset builder.

The screening subset freezes the ~160-row comparison input for the cycle-2
condition arms (off vs emotion-only) before any new results are seen, so the
tests pin the selection contract itself: bucket boundaries, proportional
largest-remainder allocation with a per-cell floor, one row per conversation,
determinism under a fixed seed, and loud failure when the conversation pool
cannot fill the target. Synthetic rows only — the real 2,354-row val split is
exercised by the generation run, not by these tests.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stratified_subset import (  # noqa: E402
    allocate_quotas,
    build_cells,
    build_subset,
    emotion_bucket,
    length_bucket,
    main,
    major_emotion_labels,
    profile_bucket,
    select_subset,
)

MAJOR = frozenset({"anxiety", "relaxed", "mixed", "helplessness", "sadness"})
EMPTY_PROFILE = {"personality_traits": [], "interests": [], "style": []}


def resp_row(sample_id: str, conversation_id: str, turns: int) -> dict:
    """A minimal B1 response row; only the stratification fields matter here."""
    return {
        "sample_id": sample_id,
        "conversation_id": conversation_id,
        "target_user_turn_id": turns,
        "source_split": "val",
        "context_turns": [
            {
                "turn_id": i + 1,
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"turn-{i}",
            }
            for i in range(turns)
        ],
        "response_text": f"response-{sample_id}",
    }


def label_row(sample_id: str, emotion: str, profile: dict | None = None) -> dict:
    return {
        "sample_id": sample_id,
        "emotion_label": emotion,
        "user_profile": EMPTY_PROFILE if profile is None else profile,
    }


def cell_rows(cell: tuple[str, str, str], count: int, prefix: str) -> list[dict]:
    """`count` response rows that all land in `cell`, one conversation each."""
    length, profile, emotion = cell
    turns = {"short": 3, "mid": 8, "long": 20}[length]
    profile_value = (
        {"personality_traits": ["内向"], "interests": [], "style": []}
        if profile == "non_empty"
        else EMPTY_PROFILE
    )
    emotion_label = "anxiety" if emotion == "major" else "care"
    rows = []
    for index in range(count):
        sample_id = f"{prefix}-{index:04d}"
        rows.append(
            {
                "response": resp_row(sample_id, f"conv-{prefix}-{index:04d}", turns),
                "label": label_row(sample_id, emotion_label, profile_value),
            }
        )
    return rows


def make_inputs(pairs: list[tuple[tuple[str, str, str], int]]) -> tuple[list[dict], list[dict]]:
    response_rows: list[dict] = []
    label_rows: list[dict] = []
    for cell, count in pairs:
        for pair in cell_rows(cell, count, f"{cell[0][:2]}{cell[1][:3]}{cell[2][:2]}"):
            response_rows.append(pair["response"])
            label_rows.append(pair["label"])
    return response_rows, label_rows


def recount(selected: list[dict], label_rows: list[dict]) -> dict:
    """Re-bucket a selection through the same join build_cells performs."""
    by_id = {row["sample_id"]: row for row in label_rows}
    return build_cells(selected, [by_id[row["sample_id"]] for row in selected], MAJOR)


class LengthBucketTests(unittest.TestCase):
    def test_boundaries_four_five_and_twelve_thirteen(self) -> None:
        self.assertEqual(length_bucket(1), "short")
        self.assertEqual(length_bucket(4), "short")
        self.assertEqual(length_bucket(5), "mid")
        self.assertEqual(length_bucket(12), "mid")
        self.assertEqual(length_bucket(13), "long")
        self.assertEqual(length_bucket(49), "long")


class ProfileBucketTests(unittest.TestCase):
    def test_all_empty_lists_is_empty(self) -> None:
        self.assertEqual(profile_bucket(EMPTY_PROFILE), "empty")
        self.assertEqual(profile_bucket({}), "empty")
        self.assertEqual(profile_bucket(None), "empty")

    def test_any_non_empty_list_is_non_empty(self) -> None:
        for field in ("personality_traits", "interests", "style"):
            profile = dict(EMPTY_PROFILE)
            profile[field] = ["音乐"]
            self.assertEqual(profile_bucket(profile), "non_empty", field)


class EmotionBucketTests(unittest.TestCase):
    def test_major_labels_and_rare_rest(self) -> None:
        for label in sorted(MAJOR):
            self.assertEqual(emotion_bucket(label, MAJOR), "major", label)
        for label in ("care", "surprise", "neutral"):
            self.assertEqual(emotion_bucket(label, MAJOR), "rare", label)
        self.assertEqual(emotion_bucket("unseen-label", MAJOR), "rare")


class MajorEmotionLabelsTests(unittest.TestCase):
    def test_top_five_by_support(self) -> None:
        labels = [
            label_row("a", "anxiety"),
            label_row("b", "anxiety"),
            label_row("c", "relaxed"),
            label_row("d", "mixed"),
            label_row("e", "mixed"),
            label_row("f", "care"),
        ]
        self.assertEqual(
            major_emotion_labels(labels),
            ["anxiety", "mixed", "care", "relaxed"],
        )

    def test_support_tie_broken_by_label_name(self) -> None:
        labels = [
            label_row("a", "zeta"),
            label_row("b", "alpha"),
            label_row("c", "mid"),
        ]
        # all three tie at 1; the boundary keeps the lexicographically
        # smaller names so the frozen major list cannot drift with input order.
        self.assertEqual(major_emotion_labels(labels, top_k=2), ["alpha", "mid"])


class AllocateQuotasTests(unittest.TestCase):
    def test_proportional_split_without_remainder(self) -> None:
        quotas = allocate_quotas({("short", "empty", "major"): 100, ("mid", "empty", "major"): 100}, 10)
        self.assertEqual(quotas, {("short", "empty", "major"): 5, ("mid", "empty", "major"): 5})

    def test_largest_remainder_tie_broken_by_size_then_key(self) -> None:
        a, b, c = ("short", "empty", "major"), ("short", "empty", "rare"), ("short", "non_empty", "major")
        quotas = allocate_quotas({a: 5, b: 5, c: 5}, 16)
        self.assertEqual(quotas, {a: 6, b: 5, c: 5})
        self.assertEqual(sum(quotas.values()), 16)

    def test_largest_fractional_part_wins_over_cell_size(self) -> None:
        a, b, c = ("mid", "empty", "major"), ("mid", "empty", "rare"), ("mid", "non_empty", "major")
        quotas = allocate_quotas({a: 8, b: 4, c: 4}, 15)
        # ideals 7.5 / 3.75 / 3.75: floors leave 2, both .75 cells take one.
        self.assertEqual(quotas, {a: 7, b: 4, c: 4})

    def test_empty_cells_get_zero_quota(self) -> None:
        empty_cell = ("long", "non_empty", "rare")
        quotas = allocate_quotas(
            {empty_cell: 0, ("short", "empty", "major"): 50, ("mid", "empty", "major"): 50}, 20
        )
        self.assertEqual(quotas[empty_cell], 0)
        self.assertEqual(sum(quotas.values()), 20)

    def test_non_empty_cell_floor_respected_with_trim(self) -> None:
        small, large = ("short", "empty", "major"), ("short", "empty", "rare")
        quotas = allocate_quotas({small: 1, large: 99}, 10)
        # ideal 0.1 / 9.9: floor lifts the small cell to 2, the large cell
        # gives back the overflow (never below its own floor of 2).
        self.assertEqual(quotas, {small: 2, large: 8})

    def test_sum_equals_target_when_floor_allows(self) -> None:
        sizes = {
            ("short", "empty", "major"): 184,
            ("short", "empty", "rare"): 75,
            ("short", "non_empty", "major"): 143,
            ("mid", "non_empty", "major"): 555,
        }
        self.assertEqual(sum(allocate_quotas(sizes, 160).values()), 160)

    def test_raises_when_per_cell_floor_exceeds_target(self) -> None:
        sizes = {index: 3 for index in range(10)}
        with self.assertRaises(ValueError):
            allocate_quotas(sizes, 10, min_per_cell=2)


class BuildCellsTests(unittest.TestCase):
    def test_joins_labels_and_buckets_rows(self) -> None:
        response_rows = [
            resp_row("s1", "c1", 3),   # short / empty / major
            resp_row("s2", "c2", 8),   # mid / empty / major
            resp_row("s3", "c3", 20),  # long / non_empty / rare
        ]
        label_rows = [
            label_row("s1", "anxiety"),
            label_row("s2", "relaxed"),
            label_row("s3", "care", {"personality_traits": ["内向"], "interests": [], "style": []}),
        ]
        cells = build_cells(response_rows, label_rows, MAJOR)
        self.assertEqual(
            {key: [row["sample_id"] for row in rows] for key, rows in cells.items()},
            {
                ("short", "empty", "major"): ["s1"],
                ("mid", "empty", "major"): ["s2"],
                ("long", "non_empty", "rare"): ["s3"],
            },
        )

    def test_rejects_response_row_without_label(self) -> None:
        with self.assertRaises(ValueError):
            build_cells([resp_row("s1", "c1", 3)], [], MAJOR)

    def test_rejects_label_without_response_row(self) -> None:
        with self.assertRaises(ValueError):
            build_cells([resp_row("s1", "c1", 3)], [label_row("s2", "anxiety")], MAJOR)

    def test_rejects_duplicate_sample_id(self) -> None:
        response_rows = [resp_row("s1", "c1", 3), resp_row("s1", "c2", 3)]
        with self.assertRaises(ValueError):
            build_cells(response_rows, [label_row("s1", "anxiety")], MAJOR)


class SelectSubsetTests(unittest.TestCase):
    def test_deterministic_for_same_seed(self) -> None:
        response_rows, label_rows = make_inputs(
            [(("short", "empty", "major"), 20), (("mid", "non_empty", "rare"), 20)]
        )
        cells = build_cells(response_rows, label_rows, MAJOR)
        first = select_subset(cells, target_size=12, seed=20261022)
        second = select_subset(cells, target_size=12, seed=20261022)
        self.assertEqual([row["sample_id"] for row in first], [row["sample_id"] for row in second])
        self.assertEqual([row["sample_id"] for row in first], sorted(row["sample_id"] for row in first))

    def test_selects_exactly_target_when_conversations_suffice(self) -> None:
        response_rows, label_rows = make_inputs(
            [
                (("short", "empty", "major"), 15),
                (("short", "empty", "rare"), 15),
                (("mid", "non_empty", "major"), 15),
                (("mid", "non_empty", "rare"), 15),
            ]
        )
        cells = build_cells(response_rows, label_rows, MAJOR)
        selected = select_subset(cells, target_size=20, seed=20261022)
        self.assertEqual(len(selected), 20)

    def test_at_most_one_row_per_conversation(self) -> None:
        # 15 conversations x 4 prediction points each; only 15 rows may be taken.
        response_rows, label_rows = [], []
        for conv in range(15):
            for turn in (3, 8, 20, 30):
                sample_id = f"s-{conv:02d}-{turn}"
                response_rows.append(resp_row(sample_id, f"conv-{conv:02d}", turn))
                label_rows.append(label_row(sample_id, "anxiety"))
        cells = build_cells(response_rows, label_rows, MAJOR)
        selected = select_subset(cells, target_size=12, seed=20261022)
        conversations = [row["conversation_id"] for row in selected]
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(set(conversations)), 12)

    def test_cell_counts_match_quotas_when_no_dedup_shortfall(self) -> None:
        pairs = [
            (("short", "empty", "major"), 25),
            (("short", "empty", "rare"), 25),
            (("mid", "empty", "major"), 25),
            (("mid", "empty", "rare"), 25),
        ]
        response_rows, label_rows = make_inputs(pairs)
        cells = build_cells(response_rows, label_rows, MAJOR)
        selected = select_subset(cells, target_size=20, seed=20261022)
        quotas = allocate_quotas({key: len(rows) for key, rows in cells.items()}, 20)
        recount_cells = recount(selected, label_rows)
        for key, quota in quotas.items():
            self.assertEqual(len(recount_cells.get(key, [])), quota, key)
        self.assertEqual(sum(len(rows) for rows in recount_cells.values()), len(selected))

    def test_cell_counts_never_exceed_cell_size(self) -> None:
        # A one-row cell keeps its floor-2 quota but can only contribute its
        # single row; round 2 tops the selection up from the large cell.
        response_rows = [resp_row("tiny", "conv-tiny", 3)]
        label_rows = [label_row("tiny", "anxiety")]
        for index in range(30):
            sample_id = f"big-{index:03d}"
            response_rows.append(resp_row(sample_id, f"conv-big-{index:03d}", 8))
            label_rows.append(label_row(sample_id, "care", {"interests": ["音乐"]}))
        cells = build_cells(response_rows, label_rows, MAJOR)
        selected = select_subset(cells, target_size=10, seed=20261022)
        recount_cells = recount(selected, label_rows)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(recount_cells[("short", "empty", "major")]), 1)
        self.assertEqual(len(recount_cells[("mid", "non_empty", "rare")]), 9)

    def test_raises_when_conversation_pool_below_target(self) -> None:
        response_rows, label_rows = [], []
        for conv in range(3):
            for turn in (3, 8):
                sample_id = f"s-{conv}-{turn}"
                response_rows.append(resp_row(sample_id, f"conv-{conv}", turn))
                label_rows.append(label_row(sample_id, "anxiety"))
        cells = build_cells(response_rows, label_rows, MAJOR)
        with self.assertRaises(ValueError):
            select_subset(cells, target_size=5, seed=20261022)


class BuildSubsetTests(unittest.TestCase):
    def test_end_to_end_contract(self) -> None:
        response_rows, label_rows = make_inputs(
            [
                (("short", "empty", "major"), 10),
                (("short", "non_empty", "rare"), 10),
                (("long", "empty", "major"), 10),
                (("long", "non_empty", "rare"), 10),
            ]
        )
        selection = build_subset(response_rows, label_rows, target_size=16, seed=20261022)
        ids = [row["sample_id"] for row in selection.rows]
        self.assertEqual(selection.actual_size, 16)
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(set(ids)), 16)
        self.assertEqual(len(selection.conversation_ids), 16)
        by_id = {row["sample_id"]: row for row in response_rows}
        for row in selection.rows:
            self.assertEqual(row, by_id[row["sample_id"]])
        self.assertEqual(sum(selection.cell_counts.values()), 16)
        for key, count in selection.cell_counts.items():
            self.assertLessEqual(count, selection.cell_sizes[key], key)


class MainTests(unittest.TestCase):
    def test_writes_selection_and_manifest(self) -> None:
        response_rows, label_rows = make_inputs(
            [
                (("short", "empty", "major"), 12),
                (("mid", "non_empty", "rare"), 12),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            response_path = root / "response_val.jsonl"
            label_path = root / "classification_val.jsonl"
            output_path = root / "out" / "selection.jsonl"
            manifest_path = root / "out" / "manifest.json"
            response_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in response_rows),
                encoding="utf-8",
            )
            label_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in label_rows),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--response-input",
                        str(response_path),
                        "--label-input",
                        str(label_path),
                        "--output",
                        str(output_path),
                        "--manifest",
                        str(manifest_path),
                        "--target",
                        "10",
                        "--seed",
                        "20261022",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertIn("PASS:", stdout.getvalue())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            written = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(written), manifest["actual_size"])
            self.assertEqual(manifest["target_size"], 10)
            self.assertEqual(manifest["seed"], 20261022)
            self.assertEqual(len(manifest["cells"]), 2)
            self.assertIn("gold", manifest["gold_labels_note"].lower())
            self.assertEqual(
                sum(cell["selected"] for cell in manifest["cells"]), manifest["actual_size"]
            )
            ids = [row["sample_id"] for row in written]
            self.assertEqual(ids, sorted(ids))


if __name__ == "__main__":
    unittest.main()
