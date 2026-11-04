#!/usr/bin/env python3
"""Prepare the frozen T7 candidate locally; does not build or publish an image."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.stage_submission import stage


def verify_response_assets(config, pins):
    """Bind complete current response directories to the existing frozen pins."""
    from scripts.verify_response_artifact import artifact_digest
    verified = {}
    for name, key in (('base_model', 'response_model_dir'), ('adapter', 'response_adapter_dir')):
        expected = pins[name]
        actual = artifact_digest(config[key])
        if any(actual[field] != expected['expected_' + field]
               for field in ('digest', 'files', 'bytes')):
            raise ValueError(f'Frozen response {name} identity mismatch: {config[key]}')
        verified[name] = actual
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--python-bin', required=True, help='Absolute Python path in the target Linux image; not a local interpreter')
    parser.add_argument('--source-commit', help='Full source commit for an exported archive without Git metadata')
    args = parser.parse_args()
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', args.python_bin):
        parser.error('Provide an absolute Linux Python path without shell metacharacters')
    if args.source_commit and not re.fullmatch(r'[a-f0-9]{40}', args.source_commit):
        parser.error('source-commit must be a full 40-character commit SHA')
    source_commit = args.source_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
    source_dirty = None if args.source_commit else bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT, text=True).strip())
    source_config = PROJECT / 'configs/integrated.json'
    config = json.loads(source_config.read_text(encoding='utf-8'))
    expected = {'classifier_dir': 'weights/emotion_longtail075_b1_20261016',
                'response_model_dir': 'weights/hf/Qwen2.5-1.5B-Instruct',
                'response_adapter_dir': 'weights/response_sft_d4_2ep',
                'classifier_format': 'sequence', 'profile_mode': 'empty',
                'condition_response': False, 'max_new_tokens': 256, 'device': 'cuda'}
    if any(config.get(key) != value for key, value in expected.items()):
        parser.error('Default config differs from the frozen T7 selection')
    from scripts.cloud_preflight import verify_hashes
    from scripts.run_integrated import load_config
    resolved = load_config(source_config)
    verify_hashes(resolved, json.loads((PROJECT / 'configs/integrated_artifacts.json').read_text(encoding='utf-8')))
    response_pins = json.loads((PROJECT / 'configs/response_artifact.json').read_text(encoding='utf-8'))
    verify_response_assets(resolved, response_pins)
    destination = args.destination.absolute()
    stage(source_config, destination)
    # Verify the copied bytes too: source changes during copying must not be
    # accepted merely because the resulting package has an internally valid hash.
    staged_response = {key: destination / config[key]
                       for key in ('response_model_dir', 'response_adapter_dir')}
    verified_response = verify_response_assets(staged_response, response_pins)
    for name in ('run_inference.py', 'start.sh'):
        target = 'start_backend.sh' if name == 'start.sh' else name
        (destination / target).write_text((PROJECT / name).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    compat = destination / 'configs/inference_config.json'
    data = json.loads(compat.read_text(encoding='utf-8'))
    data['python_bin'] = args.python_bin
    compat.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8', newline='\n')
    wrapper = ('#!/bin/sh\nset -eu\n'
               f'export PYTHON_BIN={args.python_bin}\n'
               'export INFERENCE_BACKEND=integrated HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'
               'export TOKENIZERS_PARALLELISM=false\n'
               'SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
               'export INTEGRATED_CONFIG="$SCRIPT_DIR/configs/integrated.json"\n'
               'exec bash "$SCRIPT_DIR/start_backend.sh" "$@"\n')
    (destination / 'start.sh').write_text(wrapper, encoding='utf-8', newline='\n')
    (destination / 'start.sh').chmod(0o755)
    (destination / 'configs/integrated.original.json').write_bytes(source_config.read_bytes())
    files = {}
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            files[path.relative_to(destination).as_posix()] = {'bytes': path.stat().st_size, 'sha256': digest}
    manifest = {'version': 'a-t7-candidate-v1',
                'source_commit': source_commit, 'source_dirty': source_dirty,
                'frozen_response_assets': verified_response,
                'selection': expected, 'target_python': args.python_bin,
                'target_python_verified': False, 'official_image_verified': False,
                'historical_reproducibility': 'B T6 provenance caveat retained; not an exact historical regeneration claim',
                'files': files}
    (destination / 'candidate_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'Prepared {len(files)} files: {destination}')
    print('Target Python environment and official GPU/network-isolated entry checks remain required.')


if __name__ == '__main__':
    main()
