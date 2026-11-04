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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--python-bin', required=True, help='Absolute Python path in the target Linux image; not a local interpreter')
    args = parser.parse_args()
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', args.python_bin):
        parser.error('Provide an absolute Linux Python path without shell metacharacters')
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
    verify_hashes(load_config(source_config), json.loads((PROJECT / 'configs/integrated_artifacts.json').read_text(encoding='utf-8')))
    destination = args.destination.absolute()
    stage(source_config, destination)
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
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
                'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT, text=True).strip()),
                'selection': expected, 'target_python': args.python_bin,
                'target_python_verified': False, 'official_image_verified': False,
                'historical_reproducibility': 'B T6 provenance caveat retained; not an exact historical regeneration claim',
                'files': files}
    (destination / 'candidate_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'Prepared {len(files)} files: {destination}')
    print('Target Python environment and official GPU/network-isolated entry checks remain required.')


if __name__ == '__main__':
    main()
