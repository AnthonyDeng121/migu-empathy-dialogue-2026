from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_response_sft import (
    ETA_MARGIN,
    TIMING_WARMUP_STEPS,
    _checkpoint_path,
    _epoch_micro_batches,
    _flatten_epoch_plan,
    _normalize_adapter_key,
    _reproduction_command,
    _resume_start,
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
        self.checkpoint_every = 0
        self.resume_from = None
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


class FlattenEpochPlanTests(unittest.TestCase):
    def test_positions_follow_training_order(self) -> None:
        plan = _epoch_micro_batches(5, 2, 2, seed=20261013)
        positions = _flatten_epoch_plan(plan)
        self.assertEqual(positions, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])

    def test_empty_plan_flattens_to_nothing(self) -> None:
        self.assertEqual(_flatten_epoch_plan([]), [])


class ResumeStartTests(unittest.TestCase):
    """A global micro-batch counter must map back to the exact position."""

    def test_zero_starts_at_the_beginning(self) -> None:
        plan = _epoch_micro_batches(6, 2, 2, seed=20261013)
        self.assertEqual(_resume_start(plan, 0), (0, 0))

    def test_mid_epoch_position(self) -> None:
        plan = _epoch_micro_batches(6, 2, 2, seed=20261013)
        self.assertEqual(_resume_start(plan, 2), (0, 2))

    def test_crosses_into_the_next_epoch(self) -> None:
        plan = _epoch_micro_batches(6, 2, 2, seed=20261013)
        self.assertEqual(_resume_start(plan, 3), (1, 0))
        self.assertEqual(_resume_start(plan, 5), (1, 2))

    def test_finished_checkpoint_refuses_to_resume(self) -> None:
        plan = _epoch_micro_batches(6, 2, 1, seed=20261013)
        with self.assertRaises(ValueError):
            _resume_start(plan, 3)
        with self.assertRaises(ValueError):
            _resume_start(plan, 99)


class CheckpointPathTests(unittest.TestCase):
    def test_step_is_zero_padded_and_namespaced(self) -> None:
        path = _checkpoint_path(Path("weights/run"), 7)
        self.assertEqual(path, Path("weights/run/checkpoints/step_000007"))

    def test_padding_survives_four_digit_steps(self) -> None:
        path = _checkpoint_path(Path("weights/run"), 4664)
        self.assertEqual(path.name, "step_004664")


class CheckpointReproductionTests(unittest.TestCase):
    def test_checkpoint_flags_omitted_by_default(self) -> None:
        command = _reproduction_command(_Args())
        self.assertNotIn("--checkpoint-every", command)
        self.assertNotIn("--resume-from", command)

    def test_checkpoint_every_carried_when_set(self) -> None:
        command = _reproduction_command(_Args(checkpoint_every=250))
        self.assertIn("--checkpoint-every 250", command)

    def test_resume_from_carried_when_set(self) -> None:
        command = _reproduction_command(
            _Args(resume_from=Path("weights/run/checkpoints/step_000250"))
        )
        self.assertIn(
            "--resume-from 'weights/run/checkpoints/step_000250'", command
        )


_TRAINING_DEPS_AVAILABLE = True
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import peft  # noqa: F401
except ImportError:  # pragma: no cover - depends on the interpreter
    _TRAINING_DEPS_AVAILABLE = False

_TINY_MODEL_SNAPSHOT = Path("weights/hf/Qwen2.5-0.5B-Instruct")


def _tiny_model_dir(target: Path, snapshot: Path) -> Path:
    """A 2-layer random Qwen2.5 with the real tokenizer (~10M params)."""
    import torch
    from transformers import AutoConfig, Qwen2ForCausalLM

    config = AutoConfig.from_pretrained(str(snapshot))
    config.num_hidden_layers = 2
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    if getattr(config, "layer_types", None):
        config.layer_types = ["full_attention"] * config.num_hidden_layers
    torch.manual_seed(7)
    model = Qwen2ForCausalLM(config)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ):
        source = snapshot / name
        if source.is_file():
            (target / name).write_bytes(source.read_bytes())
    return target


def _b1_resume_rows(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "sample_id": f"resume_e2e_{index}",
                "conversation_id": "resume_e2e_conv",
                "target_user_turn_id": 2,
                "context_turns": [
                    {"turn_id": 1, "role": "user", "content": f"第 {index} 轮：最近有点累。"},
                    {"turn_id": 2, "role": "user", "content": "请给一句简短温柔的回复。"},
                ],
                "response_text": f"第 {index} 条简短温柔的测试回复，抱抱你。",
            }
        )
    return rows


@unittest.skipUnless(
    _TRAINING_DEPS_AVAILABLE and _TINY_MODEL_SNAPSHOT.is_dir(),
    "training dependencies or the local 0.5B snapshot are unavailable",
)
class CheckpointResumeEndToEndTests(unittest.TestCase):
    """Interrupt-and-resume on a tiny model: the pod-restart recovery path.

    The full suite runs on a bare interpreter (CI installs no training
    dependencies), so this class skips there; it executes wherever torch /
    transformers / peft and the local snapshot exist.
    """

    def setUp(self) -> None:
        import argparse
        import json
        import tempfile

        self._argparse = argparse
        self._json = json
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.root = root
        self.train_path = root / "b1_train.jsonl"
        with self.train_path.open("w", encoding="utf-8") as handle:
            for row in _b1_resume_rows(6):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.config_path = root / "config.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                    "model_revision": "test-tiny",
                    "max_length": 128,
                    "max_new_tokens": 16,
                    "epochs": 2,
                    "batch_size": 1,
                    "gradient_accumulation": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "warmup_ratio": 0.0,
                    "seed": 20261026,
                    "lora_r": 4,
                    "lora_alpha": 8,
                    "lora_dropout": 0.0,
                    "lora_target_modules": ["q_proj", "v_proj"],
                    "gradient_checkpointing": False,
                    "max_history_turns": 2,
                    "condition_mode": "none",
                }
            ),
            encoding="utf-8",
        )
        self.model_dir = _tiny_model_dir(root / "tiny_model", _TINY_MODEL_SNAPSHOT)
        self.output_dir = root / "out"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _args(self, **overrides):
        values = dict(
            train=self.train_path,
            train_format="b1",
            condition_jsonl=None,
            config=self.config_path,
            output_dir=self.output_dir,
            report_json=self.root / "report.json",
            report_md=self.root / "report.md",
            device="cpu",
            model_dir=self.model_dir,
            max_train_samples=6,
            max_steps=None,
            checkpoint_every=0,
            resume_from=None,
            eval_prompts=2,
            local_files_only=True,
        )
        values.update(overrides)
        return self._argparse.Namespace(**values)

    def test_interrupted_run_resumes_from_checkpoint(self) -> None:
        import hashlib

        from scripts.train_response_sft import run

        first = run(self._args(max_steps=2, checkpoint_every=1))
        checkpoints = sorted((self.output_dir / "checkpoints").iterdir())
        self.assertEqual(
            [path.name for path in checkpoints], ["step_000001", "step_000002"]
        )
        for path in checkpoints:
            for required in (
                "progress.json",
                "adapter_model.safetensors",
                "trainer_state.pt",
            ):
                self.assertTrue((path / required).is_file(), f"{path} lacks {required}")
        progress = self._json.loads(
            (checkpoints[1] / "progress.json").read_text(encoding="utf-8")
        )
        self.assertEqual(progress["optimizer_steps"], 2)
        self.assertEqual(progress["micro_batches_done"], 4)
        self.assertEqual(first["checkpointing"]["every_steps"], 1)
        self.assertIsNone(first["resume"])

        second = run(
            self._args(max_steps=4, resume_from=checkpoints[1], checkpoint_every=1)
        )
        self.assertEqual(second["resume"]["skipped_micro_batches"], 4)
        self.assertEqual(second["resume"]["optimizer_steps_at_resume"], 2)
        self.assertEqual(
            second["resume"]["resume_position"], {"epoch": 0, "micro_batch": 4}
        )
        self.assertEqual(second["training_speed"]["optimizer_steps"], 4)
        self.assertEqual(second["training_speed"]["micro_batches_done"], 8)
        self.assertEqual(second["training_speed"]["micro_batches_trained"], 4)
        self.assertTrue(second["checks"]["adapter_reload"]["passed"])
        resumed_names = {Path(p).name for p in second["checkpointing"]["saved"]}
        self.assertEqual(resumed_names, {"step_000003", "step_000004"})

        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        resumed_adapter = sha256(self.output_dir / "adapter_model.safetensors")
        checkpoint_adapter = sha256(checkpoints[1] / "adapter_model.safetensors")
        self.assertNotEqual(resumed_adapter, checkpoint_adapter)

    def test_resume_rejects_a_checkpoint_that_is_not_a_checkpoint(self) -> None:
        from scripts.train_response_sft import run

        empty = self.root / "not_a_checkpoint"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            run(self._args(resume_from=empty))


if __name__ == "__main__":
    unittest.main()
