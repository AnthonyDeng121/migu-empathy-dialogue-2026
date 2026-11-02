import json
from pathlib import Path
import subprocess
import sys
import os
import tempfile
import unittest
import zipfile

from src.data.schema import EMOTIONS, PROFILE_LABELS
import scripts.stage_submission as stage_submission


class CloudPackageContractTests(unittest.TestCase):
    def test_empty_independent_profile_path_is_rejected(self):
        resolver = getattr(stage_submission, 'resolve_model_directory', None)
        self.assertTrue(callable(resolver), 'staging must validate configured model directories')
        with self.assertRaisesRegex(ValueError, 'profile_model_dir'):
            resolver('', 'profile_model_dir')

    def test_official_launcher_files_survive_cloud_packaging(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / 'sample.jsonl'
            sample.write_text(json.dumps({'id': 'smoke_1', 'conversation_id': 'c1',
                                         'target_user_turn_id': 1,
                                         'history': [{'turn_id': 1, 'role': 'user', 'content': 'hello'}]}) + '\n')
            subprocess.run([sys.executable, str(project / 'scripts/package_cloud.py'),
                            '--sample-input', str(sample)], check=True, capture_output=True)
        with zipfile.ZipFile(project / 'artifacts/cloud_source.zip') as archive:
            config = json.loads(archive.read('configs/inference_config.json'))
            self.assertEqual(config['backend'], 'transformers')
            self.assertEqual(config['integrated_config'], 'configs/integrated.json')
            self.assertIn('run_inference.py', archive.namelist())
            self.assertIn('configs/integrated.json', archive.namelist())

    def test_official_python_entry_supports_help_without_model_loading(self):
        project = Path(__file__).resolve().parents[1]
        env = {**os.environ, 'PYTHON_BIN': sys.executable}
        result = subprocess.run([sys.executable, str(project / 'run_inference.py'), '--help'],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('test_file', result.stdout)

    @unittest.skipIf(os.name == 'nt', 'Staging validates absolute Linux Python paths')
    def test_staged_official_config_keeps_selected_python(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((project / 'configs/integrated.json').read_text(encoding='utf-8'))
            for key in ('classifier_dir', 'response_model_dir', 'response_adapter_dir'):
                model = root / key
                model.mkdir()
                (model / 'placeholder.json').write_text('{}')
                config[key] = str(model)
            path = root / 'runtime.json'
            path.write_text(json.dumps(config))
            destination = root / 'participant'
            staged = subprocess.run([sys.executable, str(project / 'scripts/stage_submission.py'),
                                     '--config', str(path), '--python', sys.executable,
                                     '--destination', str(destination)], capture_output=True, text=True)
            self.assertEqual(staged.returncode, 0, staged.stderr)
            official = json.loads((destination / 'configs/inference_config.json').read_text())
            self.assertEqual(official['python_bin'], str(Path(sys.executable).absolute()))
            self.assertTrue((destination / 'run_inference.py').is_file())
            self.assertIn('INFERENCE_BACKEND=integrated', (destination / 'start.sh').read_text())
            self.assertFalse((destination / 'weights/profile_encoder_v1_b1_20261026').exists())

    @unittest.skipIf(os.name == 'nt', 'Staging targets a Linux participant directory')
    def test_staged_independent_profile_checkpoint(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((project / 'configs/integrated_profile_v1_off.json').read_text())
            for key in ('classifier_dir', 'response_model_dir', 'response_adapter_dir', 'profile_model_dir'):
                model = root / key
                model.mkdir()
                config[key] = str(model)
            for key in ('classifier_dir', 'response_model_dir'):
                model = Path(config[key])
                (model / 'config.json').write_text('{}')
                (model / 'model.safetensors').write_bytes(b'model')
                (model / 'tokenizer_config.json').write_text('{}')
            adapter = Path(config['response_adapter_dir'])
            (adapter / 'adapter_config.json').write_text('{}')
            (adapter / 'adapter_model.safetensors').write_bytes(b'adapter')
            profile = Path(config['profile_model_dir'])
            (profile / 'heads.safetensors').write_bytes(b'heads')
            (profile / 'tokenizer_config.json').write_text('{}')
            (profile / 'checkpoint_metadata.json').write_text(json.dumps({
                'emotion_labels': sorted(EMOTIONS),
                'profile_labels': {dim: sorted(labels) for dim, labels in PROFILE_LABELS.items()},
                'profile_thresholds': {dim: 0.5 for dim in PROFILE_LABELS},
                'max_length': 256,
            }))
            encoder = profile / 'encoder'
            encoder.mkdir()
            (encoder / 'config.json').write_text(json.dumps({'hidden_size': 2, 'hidden_dropout_prob': 0.1}))
            (encoder / 'model.safetensors').write_bytes(b'encoder')
            path = root / 'runtime.json'
            path.write_text(json.dumps(config))
            destination = root / 'participant'
            staged = subprocess.run([sys.executable, str(project / 'scripts/stage_submission.py'),
                                     '--config', str(path), '--python', sys.executable,
                                     '--destination', str(destination)], capture_output=True, text=True)
            self.assertEqual(staged.returncode, 0, staged.stderr)

            staged = json.loads((destination / 'configs/integrated.json').read_text())
            self.assertEqual(staged['profile_mode'], 'independent')
            self.assertEqual(staged['profile_model_dir'], 'weights/profile_encoder_v1_b1_20261026')
            self.assertEqual((destination / staged['profile_model_dir'] / 'heads.safetensors').read_bytes(), b'heads')

            preflight = subprocess.run([
                sys.executable, str(destination / 'scripts/cloud_preflight.py'),
                '--config', str(destination / 'configs/integrated.json'), '--files-only',
            ], cwd=destination, capture_output=True, text=True)
            self.assertEqual(preflight.returncode, 0, preflight.stdout + preflight.stderr)
            self.assertTrue(json.loads(preflight.stdout)['ready_for_smoke'])

    @unittest.skipIf(os.name == 'nt', 'Staging targets a Linux participant directory')
    def test_python_path_with_shell_special_characters_is_quoted(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((project / 'configs/integrated.json').read_text())
            for key in ('classifier_dir', 'response_model_dir', 'response_adapter_dir'):
                model = root / key
                model.mkdir()
                (model / 'placeholder.json').write_text('{}')
                config[key] = str(model)
            path = root / 'runtime.json'
            path.write_text(json.dumps(config))
            python_path = root / 'python @$(touch INJECTED)'
            python_path.symlink_to(sys.executable)
            destination = root / 'participant'
            staged = subprocess.run([sys.executable, str(project / 'scripts/stage_submission.py'),
                                     '--config', str(path), '--python', str(python_path),
                                     '--destination', str(destination)], capture_output=True, text=True)
            self.assertEqual(staged.returncode, 0, staged.stderr)
            (destination / 'start_backend.sh').write_text('#!/bin/sh\nprintf "%s\\n" "$PYTHON_BIN"\n')
            result = subprocess.run(['bash', str(destination / 'start.sh')], cwd=root,
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), str(python_path))
            self.assertFalse((root / 'INJECTED').exists())
