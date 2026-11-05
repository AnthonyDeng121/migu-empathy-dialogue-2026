"""Portable staging and rejection of corrupted optional profile assets."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from scripts.stage_submission import stage, PROJECT
from scripts.cloud_preflight import verify_hashes
from src.data.schema import EMOTIONS, PROFILE_LABELS

class T6PackagingTests(unittest.TestCase):
    def test_relocated_stage_and_profile_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((PROJECT / 'configs/integrated_profile_v1_off.json').read_text(encoding='utf-8'))
            for key, names in {
                'classifier_dir': ['config.json', 'model.safetensors', 'tokenizer_config.json'],
                'response_model_dir': ['config.json', 'model.safetensors', 'tokenizer_config.json'],
                'response_adapter_dir': ['adapter_config.json', 'adapter_model.safetensors'],
                'profile_model_dir': ['checkpoint_metadata.json', 'heads.safetensors', 'encoder/config.json', 'encoder/model.safetensors', 'tokenizer.json', 'tokenizer_config.json'],
            }.items():
                directory = root / key
                for name in names:
                    target = directory / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text('{}', encoding='utf-8')
                if key == 'profile_model_dir':
                    metadata = {'emotion_labels': sorted(EMOTIONS),
                                'profile_labels': {k: sorted(v) for k, v in PROFILE_LABELS.items()},
                                'profile_thresholds': {k: 0.5 for k in PROFILE_LABELS}, 'max_length': 256}
                    (directory / 'checkpoint_metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
                config[key] = str(directory)
            source = root / 'input.json'
            source.write_text(json.dumps(config), encoding='utf-8')
            destination = root / 'stage'
            portable = stage(source, destination)
            moved = root / 'relocated'
            destination.rename(moved)
            code = 'from scripts.run_integrated import load_config,check_files; from pathlib import Path; check_files(load_config(Path("configs/integrated.json")))'
            subprocess.run([sys.executable, '-c', code], cwd=moved, check=True, capture_output=True)
            resolved = {key: moved / value if key.endswith('_dir') else value for key,value in portable.items()}
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            profile = resolved['profile_model_dir']
            manifest = {'classifier_files': [], 'response_adapter_sha256': digest(resolved['response_adapter_dir'] / 'adapter_model.safetensors'),
                        'profile_files': [{'path': f.relative_to(profile).as_posix(), 'sha256': digest(f)} for f in profile.rglob('*') if f.is_file()]}
            verify_hashes(resolved, manifest)
            if os.name == 'posix':
                cli_stage = root / 'cli-stage'
                subprocess.run([sys.executable, str(PROJECT / 'scripts/stage_submission.py'),
                                '--config', str(source), '--python', sys.executable,
                                '--destination', str(cli_stage)], check=True, capture_output=True)
                (cli_stage / 'configs/integrated_artifacts.json').write_text(json.dumps(manifest), encoding='utf-8')
                cli_stage.rename(root / 'cli-relocated')
                subprocess.run([sys.executable, 'scripts/cloud_preflight.py', '--files-only', '--verify-hashes'],
                               cwd=root / 'cli-relocated', check=True, capture_output=True)
                self.assertTrue((root / 'cli-relocated/start.sh').stat().st_mode & 0o111)
                self.assertTrue((root / 'cli-relocated/run_inference.py').is_file())
            for name in ('checkpoint_metadata.json', 'tokenizer.json', 'heads.safetensors', 'encoder/model.safetensors'):
                path = profile / name
                original = path.read_bytes()
                path.write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    verify_hashes(resolved, manifest)
                path.write_bytes(original)
            (profile / 'tokenizer.json').unlink()
            with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
                verify_hashes(resolved, manifest)
            portable['profile_mode'] = 'empty'
            source.write_text(json.dumps({**config, 'profile_mode':'empty'}), encoding='utf-8')
            empty = root / 'empty'
            result = stage(source, empty)
            self.assertNotIn('profile_model_dir', result)
            self.assertFalse((empty / 'weights/profile_encoder_v1_b1_20261026').exists())
