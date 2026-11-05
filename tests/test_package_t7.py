"""Frozen response identity must gate packaging, not just its final manifest."""
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from scripts import package_t7, run_integrated, stage_submission
from scripts.verify_response_artifact import artifact_digest


class T7ResponseIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {
            'classifier_dir': 'weights/emotion_longtail075_b1_20261016',
            'response_model_dir': 'weights/hf/Qwen2.5-1.5B-Instruct',
            'response_adapter_dir': 'weights/response_sft_d4_2ep',
            'classifier_format': 'sequence', 'profile_mode': 'empty',
            'condition_response': False, 'max_new_tokens': 256, 'device': 'cuda'}
        for name in ('src', 'scripts', 'configs'):
            (self.root / name).mkdir()
        specs = {
            'classifier_dir': ('config.json', 'model.safetensors', 'tokenizer_config.json'),
            'response_model_dir': ('model.safetensors', 'config.json', 'generation_config.json',
                                   'tokenizer.json', 'tokenizer_config.json'),
            'response_adapter_dir': ('adapter_model.safetensors', 'adapter_config.json',
                                     'tokenizer.json', 'tokenizer_config.json')}
        for key, names in specs.items():
            directory = self.root / self.config[key]
            directory.mkdir(parents=True)
            for name in names:
                (directory / name).write_bytes(b'{}\n')
        self.targets = [self.root / self.config[key] / name
                        for key in ('response_model_dir', 'response_adapter_dir')
                        for name in specs[key] if not name.endswith('.safetensors')]
        self.weights = [self.root / self.config['response_model_dir'] / 'model.safetensors',
                        self.root / self.config['response_adapter_dir'] / 'adapter_model.safetensors']
        self.weight_bytes = [p.read_bytes() for p in self.weights]
        digest = hashlib.sha256(b'{}\n').hexdigest()
        files = {
            'integrated.json': self.config,
            'inference_config.json': {},
            'integrated_artifacts.json': {'classifier_files': [],
                'response_adapter_sha256': digest, 'response_base_weights_sha256': digest},
            'response_artifact.json': {
                name: {'expected_' + k: v for k, v in artifact_digest(self.root / self.config[key]).items()}
                for name, key in (('base_model', 'response_model_dir'), ('adapter', 'response_adapter_dir'))}}
        for name, data in files.items():
            (self.root / 'configs' / name).write_text(json.dumps(data), encoding='utf-8')
        (self.root / 'start.sh').write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
        (self.root / 'run_inference.py').write_text('# fixture\n', encoding='utf-8')

    def package(self, destination, stage=None):
        with ExitStack() as stack:
            for module in (package_t7, run_integrated, stage_submission):
                stack.enter_context(patch.object(module, 'PROJECT', self.root))
            stack.enter_context(patch('sys.argv', ['package_t7', '--destination', str(destination),
                '--python-bin', '/target/bin/python', '--source-commit', 'a' * 40]))
            if stage:
                stack.enter_context(patch.object(package_t7, 'stage', stage))
            package_t7.main()

    def test_normal_assets_package_successfully(self):
        destination = self.root / 'normal'
        self.package(destination)
        manifest = json.loads((destination / 'candidate_manifest.json').read_text())
        for name, key in (('base_model', 'response_model_dir'), ('adapter', 'response_adapter_dir')):
            self.assertEqual(manifest['frozen_response_assets'][name],
                             artifact_digest(self.root / self.config[key]))

    def test_each_modified_or_deleted_companion_fails_before_staging(self):
        for index, path in enumerate(self.targets):
            original = path.read_bytes()
            for action in ('modify', 'delete'):
                with self.subTest(path=path.relative_to(self.root).as_posix(), action=action):
                    destination = self.root / f'bad-{index}-{action}'
                    try:
                        if action == 'modify':
                            path.write_bytes(b'{"changed": true}\n')
                        else:
                            path.unlink()
                        self.assertEqual([p.read_bytes() for p in self.weights], self.weight_bytes)
                        with self.assertRaisesRegex(ValueError, 'Frozen response .* identity mismatch'):
                            self.package(destination)
                        self.assertFalse(destination.exists())
                    finally:
                        path.write_bytes(original)

    def test_changed_copy_is_rejected_before_manifest_publication(self):
        def damaged_copy(config, destination):
            result = stage_submission.stage(config, destination)
            (destination / self.config['response_model_dir'] / 'tokenizer.json').write_bytes(b'changed')
            return result

        destination = self.root / 'copy-changed'
        with self.assertRaisesRegex(ValueError, 'Frozen response base_model identity mismatch'):
            self.package(destination, damaged_copy)
        self.assertFalse((destination / 'candidate_manifest.json').exists())
