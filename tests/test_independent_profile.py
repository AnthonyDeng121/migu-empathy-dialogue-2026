import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.run_integrated import Models, check_files, load_config
from src.classification.emotion_profile_checkpoint import read_packaged_metadata
from src.data.schema import EMOTIONS, PROFILE_LABELS


class IndependentProfileTests(unittest.TestCase):
    def metadata(self):
        return {"emotion_labels": sorted(EMOTIONS),
                "profile_labels": {k: sorted(v) for k, v in PROFILE_LABELS.items()},
                "profile_thresholds": {k: 0.5 for k in PROFILE_LABELS}, "max_length": 256}

    def test_config_requires_explicit_directory(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            config = {"classifier_dir": "c", "response_model_dir": "d", "response_adapter_dir": "a",
                      "device": "cpu", "classifier_format": "sequence", "profile_mode": "independent",
                      "condition_response": False, "max_new_tokens": 256}
            for value in (None, "", "   ", 42):
                path.write_text(json.dumps({**config, "profile_model_dir": value}), encoding="utf-8")
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "profile_model_dir"):
                    load_config(path)
            path.write_text(json.dumps({**config, "profile_model_dir": "weights/profile"}), encoding="utf-8")
            self.assertTrue(load_config(path)["profile_model_dir"].is_absolute())

    def test_independent_profile_never_overwrites_emotion_or_id(self):
        model = Models.__new__(Models)
        model.config = {"profile_mode": "independent"}
        profile = {k: [sorted(v)[0]] for k, v in PROFILE_LABELS.items()}
        model.profile_classifier = Mock()
        model.profile_classifier.predict_contexts.return_value = [{"id": "wrong", "emotion_label": "anger", "user_profile": profile}]
        original = {"id": "sample", "emotion_label": "joy", "user_profile": {k: [] for k in PROFILE_LABELS}}
        history = [{"role": "user", "content": "hello"}]
        result = model.apply_profile({"history": history}, original)
        self.assertEqual(result, {**original, "user_profile": profile})
        self.assertEqual(original["user_profile"], {k: [] for k in PROFILE_LABELS})
        model.profile_classifier.predict_contexts.assert_called_once_with([history])
        model.profile_classifier.predict_contexts.side_effect = RuntimeError("broken weights")
        with self.assertRaisesRegex(RuntimeError, "broken weights"):
            model.apply_profile({"history": history}, original)

    def test_empty_mode_does_not_call_profile_model(self):
        model = Models.__new__(Models)
        model.config = {"profile_mode": "empty"}
        model.profile_classifier = Mock()
        result = model.apply_profile({}, {"id": "x", "emotion_label": "joy", "user_profile": {}})
        self.assertEqual(result["user_profile"], {k: [] for k in PROFILE_LABELS})
        model.profile_classifier.predict_contexts.assert_not_called()

    def test_invalid_profile_rejected_before_generation(self):
        model = Models.__new__(Models)
        model.config = {"profile_mode": "independent"}
        model.profile_classifier = Mock()
        for profile in ({}, {k: ["invalid"] for k in PROFILE_LABELS}):
            model.profile_classifier.predict_contexts.return_value = [{"user_profile": profile}]
            with self.assertRaises(ValueError):
                model.apply_profile({"history": []}, {"emotion_label": "joy"})

    def test_metadata_validation_and_file_preflight(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "checkpoint_metadata.json"
            valid = self.metadata()
            path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(read_packaged_metadata(root), valid)
            # A different order is valid: metadata explicitly maps each head index.
            reordered = copy.deepcopy(valid)
            reordered["profile_labels"]["style"].reverse()
            path.write_text(json.dumps(reordered), encoding="utf-8")
            self.assertEqual(read_packaged_metadata(root), reordered)
            bad_values = []
            for key in ("profile_labels", "profile_thresholds", "max_length"):
                bad = copy.deepcopy(valid)
                del bad[key]
                bad_values.append(bad)
            for value in (-0.1, 1.1, float("nan"), True, [0.5]):
                bad = copy.deepcopy(valid)
                bad["profile_thresholds"]["style"] = value
                bad_values.append(bad)
            bad = copy.deepcopy(valid)
            bad["profile_labels"]["style"].append(bad["profile_labels"]["style"][0])
            bad_values.append(bad)
            for bad in bad_values:
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(metadata=bad), self.assertRaises(ValueError):
                    read_packaged_metadata(root)
            path.write_text(json.dumps(valid), encoding="utf-8")
            config = {"profile_mode": "independent", "profile_model_dir": root}
            # Preflight must run even on sequence's early-return route.
            with self.assertRaisesRegex(FileNotFoundError, "heads.safetensors"):
                check_files(config)
            from src.classification.emotion_profile_checkpoint import check_packaged_files
            for name in ("heads.safetensors", "encoder/config.json", "tokenizer_config.json", "encoder/model.safetensors"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            check_packaged_files(root)
            (root / "encoder/model.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "encoder weights"):
                check_packaged_files(root)
    def test_real_predict_path_uses_independent_profile_with_conditions_off(self):
        from contextlib import nullcontext
        model = Models.__new__(Models)
        model.config = {"profile_mode": "independent", "condition_response": False, "max_new_tokens": 256}
        model.sequence = False
        model.packaged = True
        model.device = "cpu"
        model.classifier = Mock()
        model.classifier.predict_contexts.return_value = [{"emotion_label": "joy", "user_profile": {k: [] for k in PROFILE_LABELS}}]
        profile = {k: [sorted(v)[0]] for k, v in PROFILE_LABELS.items()}
        model.profile_classifier = Mock()
        model.profile_classifier.predict_contexts.return_value = [{"emotion_label": "anger", "user_profile": profile}]
        model.torch = Mock()
        model.torch.inference_mode.side_effect = nullcontext
        model.d_tokenizer = Mock(pad_token_id=0)
        tokens = Mock()
        tokens.shape = (1, 2)
        class Inputs(dict):
            def to(self, device):
                return self
        model.d_tokenizer.return_value = Inputs(input_ids=tokens)
        model.generator = Mock()
        model.generator.generate.return_value = [[0, 1, 2]]
        model.d_tokenizer.decode.return_value = "hello"
        row = {"id": "sample", "history": [{"role": "user", "content": "hi"}]}
        for enabled, fields, expected in (
            (False, "both", None),
            (True, "emotion", {"emotion_label": "joy"}),
            (True, "both", {"emotion_label": "joy", "user_profile": profile}),
        ):
            model.config.update(condition_response=enabled, condition_fields=fields)
            with patch("src.response.sft_data.build_eval_sample", return_value={"messages": []}) as build:
                result = model.predict(row)
            build.assert_called_once_with(row, condition=expected)
            self.assertEqual(result["user_profile"], profile)
            from src.response.sft_data import system_prompt
            prompt = system_prompt(expected)
            self.assertEqual("用户画像：" in prompt, enabled and fields == "both")
            self.assertEqual("joy" in prompt, enabled)
        self.assertEqual(result["emotion_label"], "joy")
        self.assertEqual(result["user_profile"], profile)
        self.assertEqual(result["id"], "sample")
        self.assertEqual(model.generator.generate.call_args.kwargs["max_new_tokens"], 256)
        self.assertFalse(model.generator.generate.call_args.kwargs["do_sample"])
