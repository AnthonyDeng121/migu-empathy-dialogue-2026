from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_response_sft import (
    ETA_MARGIN,
    TIMING_WARMUP_STEPS,
    _epoch_micro_batches,
    _normalize_adapter_key,
    _reproduction_command,
    _steps_per_epoch,
    _validate_run_limits,
    _weight_storage_note,
)

class _Args:
    """Stands in for the parsed CLI arguments of the training script."""
    def __init__(self, **overrides):
        self.train = Path("data/raw/训练-验证-数据集/train/train_public.jsonl")
        self.config = Path("configs/response_sft.json")
        self.device = "cuda"
        self.output_dir = Path("weights/response_sft_smoke_1p5b")
        self.report_json = Path("reports/d_day2_sft_smoke_1p5b.json")
        self.report_md = Path("reports/D_DAY2_SFT_SMOKE_1P5B.md")
        self.model_dir = None
        self.max_train_samples = None
        self.max_steps = None
        self.eval_prompts = 3
        self.local_files_only = False
        self.train_format = "raw"
        self.condition_jsonl = None
        for key, value in overrides.items():
            setattr(self, key, value)


class EpochMicroBatchPlanTests(unittest.TestCase):
    """The shuffle/epoch contract the old single-pass loop got wrong."""

    def test_every_sample_appears_exactly_once_per_epoch(self) -> None:
        plan = _epoch_micro_batches(23, 4, 2, seed=20261013)
        for epoch in plan:
            flat = [index for micro_batch in epoch for index in micro_batch]
            self.assertEqual(sorted(flat), list(range(23)))

    def test_plan_is_deterministic_for_a_seed(self) -> None:
        first = _epoch_micro_batches(32, 8, 3, seed=20261013)
        second = _epoch_micro_batches(32, 8, 3, seed=20261013)
        self.assertEqual(first, second)

    def test_order_is_not_the_file_order(self) -> None:
        plan = _epoch_micro_batches(32, 1, 1, seed=20261013)
        flat = plan[0]
        self.assertNotEqual(flat, [[index] for index in range(32)])
        self.assertEqual(sorted(index[0] for index in flat), list(range(32)))

    def test_epochs_do_not_replay_the_same_order(self) -> None:
        plan = _epoch_micro_batches(32, 1, 3, seed=20261013)
        orders = [[index[0] for index in epoch] for epoch in plan]
        self.assertNotEqual(orders[0], orders[1])
        self.assertNotEqual(orders[1], orders[2])

    def test_last_micro_batch_holds_the_remainder(self) -> None:
        plan = _epoch_micro_batches(10, 4, 1, seed=20261013)
        self.assertEqual([len(micro_batch) for micro_batch in plan[0]], [4, 4, 2])

    def test_single_batch_covers_all_samples(self) -> None:
        plan = _epoch_micro_batches(5, 8, 1, seed=20261013)
        self.assertEqual(len(plan[0]), 1)
        self.assertEqual(sorted(plan[0][0]), [0, 1, 2, 3, 4])


class StepsPerEpochTests(unittest.TestCase):
    def test_full_official_train_set(self) -> None:
        # 18,657 samples, batch 1, accumulation 8 -> 2,333 optimizer steps.
        self.assertEqual(_steps_per_epoch(18657, 1, 8), 2333)

    def test_smoke_subset(self) -> None:
        self.assertEqual(_steps_per_epoch(512, 1, 8), 64)

    def test_remainder_micro_batch_counts_as_a_step(self) -> None:
        # 5 samples, batch 2 -> 3 micro batches; the lone last micro batch
        # still forces an optimizer step, so 2 steps, not 1.
        self.assertEqual(_steps_per_epoch(5, 2, 2), 2)
        self.assertEqual(_steps_per_epoch(5, 2, 1), 3)


class ValidateRunLimitsTests(unittest.TestCase):
    def test_defaults_pass(self) -> None:
        _validate_run_limits(None, None, 3)

    def test_zero_or_negative_limits_rejected(self) -> None:
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                _validate_run_limits(bad, None, 3)
            with self.assertRaises(ValueError):
                _validate_run_limits(None, bad, 3)

    def test_negative_eval_prompts_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _validate_run_limits(None, None, -1)

    def test_full_run_with_probes_allowed(self) -> None:
        # The old code raised here for any full run with the default probes;
        # full runs must be able to finish and write their report.
        _validate_run_limits(None, None, 3)
        _validate_run_limits(None, None, 0)


class NormalizeAdapterKeyTests(unittest.TestCase):
    def test_default_adapter_suffix_dropped(self) -> None:
        name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        self.assertEqual(
            _normalize_adapter_key(name),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        )

    def test_saved_key_unchanged(self) -> None:
        key = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
        self.assertEqual(_normalize_adapter_key(key), key)


class ReproductionCommandTests(unittest.TestCase):
    def test_smoke_command_carries_every_cap(self) -> None:
        command = _reproduction_command(
            _Args(max_train_samples=512, max_steps=53, model_dir=Path("weights/hf/Qwen2.5-1.5B-Instruct"))
        )
        self.assertIn("--max-train-samples 512", command)
        self.assertIn("--max-steps 53", command)
        self.assertIn("--model-dir 'weights/hf/Qwen2.5-1.5B-Instruct'", command)
        self.assertIn("--config 'configs/response_sft.json'", command)

    def test_defaults_omitted(self) -> None:
        command = _reproduction_command(_Args())
        self.assertNotIn("--max-steps", command)
        self.assertNotIn("--eval-prompts", command)
        self.assertIn("--device cuda", command)

    def test_raw_format_omitted_by_default(self) -> None:
        command = _reproduction_command(_Args())
        self.assertNotIn("--train-format", command)
        self.assertNotIn("--condition-jsonl", command)

    def test_b1_command_carries_format_and_condition_file(self) -> None:
        command = _reproduction_command(
            _Args(
                train=Path("data/processed/b1_full_v1/response_train.jsonl"),
                train_format="b1",
                condition_jsonl=Path("data/processed/b1_full_v1/classification_train.jsonl"),
                model_dir=Path("weights/hf/Qwen2.5-1.5B-Instruct"),
            )
        )
        self.assertIn("--train-format b1", command)
        self.assertIn(
            "--condition-jsonl 'data/processed/b1_full_v1/classification_train.jsonl'",
            command,
        )
        self.assertIn("--train 'data/processed/b1_full_v1/response_train.jsonl'", command)

    def test_b1_without_condition_file_when_mode_not_gold(self) -> None:
        # CYCLE2 T3 无条件对照（config condition_mode="none"）跑 B1 数据时
        # 不传 --condition-jsonl；复现命令必须反映这条路径，防止后人照抄
        # gold 臂命令时误带条件文件。test_raw_format_omitted_by_default 只
        # 覆盖 raw 默认路径，b1+none 路径此前无覆盖。
        command = _reproduction_command(
            _Args(
                train=Path("data/processed/b1_full_v1/response_train.jsonl"),
                train_format="b1",
                condition_jsonl=None,
                model_dir=Path("weights/hf/Qwen2.5-1.5B-Instruct"),
            )
        )
        self.assertIn("--train-format b1", command)
        self.assertNotIn("--condition-jsonl", command)
        self.assertIn("--train 'data/processed/b1_full_v1/response_train.jsonl'", command)


class TimingContractTests(unittest.TestCase):
    def test_warmup_steps_are_skipped_and_margin_applied(self) -> None:
        self.assertGreater(TIMING_WARMUP_STEPS, 0)
        self.assertEqual(ETA_MARGIN, 1.3)


class WeightStorageNoteTests(unittest.TestCase):
    """The card-time wording must hold on rented platform GPUs too."""

    def test_note_makes_no_card_time_claim(self) -> None:
        note = _weight_storage_note()
        self.assertIn("不提交 Git", note)
        self.assertNotIn("不计平台", note)
        self.assertNotIn("本机", note)


if __name__ == "__main__":
    unittest.main()


class TwoEpochConfigTests(unittest.TestCase):
    """Day4 单变量候选配置：与 D2 首轮唯一差别是 epochs。"""

    def test_only_epochs_differs_from_first_round(self) -> None:
        import json

        base = json.loads(Path("configs/response_sft.json").read_text(encoding="utf-8"))
        candidate = json.loads(Path("configs/response_sft_2epoch.json").read_text(encoding="utf-8"))
        candidate.pop("_comment", None)
        self.assertEqual(candidate.pop("epochs"), 2)
        self.assertEqual(base.pop("epochs"), 1)
        self.assertEqual(candidate, base)

    def test_epoch_plan_doubles_steps(self) -> None:
        plan_one = _epoch_micro_batches(64, 1, 1, 20261013)
        plan_two = _epoch_micro_batches(64, 1, 2, 20261013)
        self.assertEqual(len(plan_one), 1)
        self.assertEqual(len(plan_two), 2)
        self.assertEqual(len(plan_two[0]), len(plan_two[1]))
        self.assertEqual(_steps_per_epoch(18653, 1, 8), 2332)
