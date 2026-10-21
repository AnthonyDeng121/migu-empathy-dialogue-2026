import json
from pathlib import Path
import subprocess
import sys
import os
import tempfile
import unittest
import zipfile


class CloudPackageContractTests(unittest.TestCase):
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
            subprocess.run([sys.executable, str(project / 'scripts/stage_submission.py'),
                            '--config', str(path), '--python', sys.executable,
                            '--destination', str(destination)], check=True, capture_output=True)
            official = json.loads((destination / 'configs/inference_config.json').read_text())
            self.assertEqual(official['python_bin'], str(Path(sys.executable).absolute()))
            self.assertTrue((destination / 'run_inference.py').is_file())
            self.assertIn('INFERENCE_BACKEND=integrated', (destination / 'start.sh').read_text())
