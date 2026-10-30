"""Exercise the full T5 bundle writer without weights, GPU or official data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_t5_candidates as builder
from scripts.assemble_candidate import digest
from src.data.schema import EMOTIONS, PROFILE_LABELS


class T5PortableBundleTests(unittest.TestCase):
    def test_complete_bundles_and_linked_hashes_match_across_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(path, value, jsonl=False):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                text = ''.join(json.dumps(row) + '\n' for row in value) if jsonl else json.dumps(value)
                target.write_text(text, encoding='utf-8', newline='\n')
                return target

            config = {'classifier_dir': 'weights/emotion', 'classifier_format': 'sequence',
                      'response_model_dir': 'weights/base', 'response_adapter_dir': 'weights/adapter',
                      'profile_mode': 'empty', 'condition_response': False, 'device': 'cpu', 'max_new_tokens': 1}
            write('configs/integrated.json', config)
            for name in ['configs/integrated_artifacts.json', 'reports/C_CYCLE2_T4_EMOTION_TARGETED.md',
                         'reports/D_CYCLE2_T4_LONG_RESPONSE.md', 'reports/B_T3_INDEPENDENT_REVIEW.md',
                         'reports/c_cycle2_profile_v1_20261026.json', 'results/response_gen/t5_conditional_decision.json']:
                write(name, {})
            rows = [{'id': 'sample', 'conversation_id': 'conversation', 'target_user_turn_id': 1,
                     'history': [{'turn_id': 1, 'role': 'user', 'content': 'hello'}]}]
            source = write('data/processed/b1_full_v1/response_val.jsonl', rows, True)
            response = write('results/response_gen/d4v2_val_newcard.jsonl',
                             [{'id': 'sample', 'conversation_id': 'conversation', 'response_text': 'reply'}], True)
            write('results/response_gen/d4v2_val_newcard_report.json',
                  {'samples': 1, 'condition_mode': 'none', 'model_path': config['response_model_dir'],
                   'adapter_path': config['response_adapter_dir'], 'output_sha256': digest(response.read_bytes())})
            write('results/response_gen/c5_conditions_val_full.jsonl', [{'id': 'sample', 'emotion_label': 'joy'}], True)
            write('reports/c_cycle2_profile_v1_predictions.jsonl',
                  [{'id': 'sample', 'user_profile': {'personality_traits': [], 'interests': ['music'], 'style': []}}], True)
            profile = 'weights/profile_encoder_v1_b1_20261026/'
            write(profile + 'checkpoint_metadata.json',
                  {'emotion_labels': sorted(EMOTIONS), 'profile_labels': {k: sorted(v) for k, v in PROFILE_LABELS.items()},
                   'profile_thresholds': {k: 0.5 for k in PROFILE_LABELS}, 'max_length': 256})
            for name in ['heads.safetensors', 'encoder/config.json', 'encoder/model.safetensors', 'tokenizer_config.json']:
                write(profile + name, {})

            bundles = [root / 'first', root / 'different/depth/second']
            with patch.object(builder, 'ROOT', root), patch.object(builder, 'EXPECTED_INPUT_SHA256', digest(source.read_bytes())):
                for folder in bundles:
                    with patch('sys.argv', ['build_t5_candidates', '--output-dir', str(folder)]), contextlib.redirect_stdout(io.StringIO()):
                        builder.main()
            names = {path.name for path in bundles[0].iterdir()}
            self.assertEqual(len(names), 11)
            self.assertEqual(names, {path.name for path in bundles[1].iterdir()})
            for name in names:
                with self.subTest(file=name):
                    self.assertEqual((bundles[0] / name).read_bytes(), (bundles[1] / name).read_bytes())

            def verify(obj, folder):
                if isinstance(obj, dict):
                    if 'path_base' in obj:
                        self.assertIn(obj['path_base'], ['repo', 'artifact'])
                        self.assertFalse(Path(obj['path']).is_absolute())
                        self.assertNotIn('..', Path(obj['path']).parts)
                        path = (root if obj['path_base'] == 'repo' else folder) / obj['path']
                        if 'sha256' in obj:
                            self.assertEqual(obj['sha256'], digest(path.read_bytes()))
                    for value in obj.values():
                        verify(value, folder)
                elif isinstance(obj, list):
                    for value in obj:
                        verify(value, folder)

            for folder in bundles:
                for name in ['catalog.json', 'baseline.manifest.json', 'profile_v1.manifest.json']:
                    verify(json.loads((folder / name).read_text(encoding='utf-8')), folder)
