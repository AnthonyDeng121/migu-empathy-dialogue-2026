"""Build A T5 fixed candidates using the latest accepted C/D delivery decisions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.assemble_candidate import fingerprint
from scripts.run_integrated import load_config
from src.common.io import read_jsonl
from src.classification.emotion_profile_checkpoint import check_packaged_files, read_packaged_metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise ValueError('output directory must not exist')
    # Source fingerprints are recorded before creating any output.
    sources = ['configs/integrated_artifacts.json', 'reports/C_CYCLE2_T4_EMOTION_TARGETED.md',
               'reports/D_CYCLE2_T4_LONG_RESPONSE.md', 'reports/B_T3_INDEPENDENT_REVIEW.md',
               'reports/c_cycle2_profile_v1_20261026.json',
               'results/response_gen/t5_conditional_decision.json']
    before = {s: fingerprint(ROOT / s) for s in sources}
    base = json.loads((ROOT / 'configs/integrated.json').read_text(encoding='utf-8'))
    input_path = ROOT / 'data/processed/b1_full_v1/response_val.jsonl'
    input_fp = fingerprint(input_path)
    expected_input = 'ddf76588ee2be4912b7c7bd5e9070ef1a9bc967e59378b911f13cf2cfc3078e2'
    if expected_input not in (input_fp['sha256'], input_fp['lf_sha256']):
        raise ValueError('B1 validation input differs from D frozen input')
    profile_path = ROOT / 'weights/profile_encoder_v1_b1_20261026'
    check_packaged_files(profile_path)
    metadata = read_packaged_metadata(profile_path)
    inputs = read_jsonl(input_path)
    empty = [{'id': r.get('id', r.get('sample_id')), 'user_profile': {
        'personality_traits': [], 'interests': [], 'style': []}} for r in inputs]
    out.mkdir(parents=True)
    empty_path = out / 'empty_profiles.jsonl'
    empty_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in empty), encoding='utf-8')
    candidates = []
    # D4 v2 remains the selected response model; use its latest same-card full predictions.
    for name, profile in [('baseline', 'empty'), ('profile_v1', 'independent')]:
        config = dict(base, condition_response=False, profile_mode=profile)
        if profile == 'independent':
            config['profile_model_dir'] = 'weights/profile_encoder_v1_b1_20261026'
        cfg = out / f'{name}.config.json'
        cfg.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        load_config(cfg)
        command = [sys.executable, 'scripts/assemble_candidate.py',
                   '--input', 'data/processed/b1_full_v1/response_val.jsonl',
                   '--responses', 'results/response_gen/d4v2_val_newcard.jsonl',
                   '--response-report', 'results/response_gen/d4v2_val_newcard_report.json',
                   '--emotions', 'results/response_gen/c5_conditions_val_full.jsonl',
                   '--profiles', str(empty_path if profile == 'empty' else ROOT / 'reports/c_cycle2_profile_v1_predictions.jsonl'),
                   '--emotion-model', config['classifier_dir'],
                   '--profile-model', 'empty' if profile == 'empty' else config['profile_model_dir'],
                   '--output', str(out / f'{name}.jsonl'), '--manifest', str(out / f'{name}.manifest.json')]
        for source in sources:
            command += ['--source-record', source]
        subprocess.run(command, cwd=ROOT, check=True)
        model_paths = {key: {'path': config[key], 'present': (ROOT / config[key]).is_dir()}
                       for key in ('classifier_dir', 'response_model_dir', 'response_adapter_dir', 'profile_model_dir') if key in config}
        candidates.append({'id': name, 'status': 'self_checked_pending_independent_review',
                           'config': fingerprint(cfg), 'prediction': fingerprint(out / f'{name}.jsonl'),
                           'manifest': fingerprint(out / f'{name}.manifest.json'),
                           'local_model_availability': model_paths,
                           'checkpoint_reload_performed': False,
                           'response_condition': 'off', 'samples': len(inputs)})
    deferred_configs = []
    for name, changes, status in [
        ('emotion065_only', {'classifier_dir': 'weights/emotion_longtail065_b1_20261028'}, 'missing_delivery'),
        ('emotion065_profile_v1', {'classifier_dir': 'weights/emotion_longtail065_b1_20261028',
                                  'profile_mode': 'independent',
                                  'profile_model_dir': 'weights/profile_encoder_v1_b1_20261026'}, 'missing_delivery'),
        ('response_t4_only', {'response_adapter_dir': 'weights/t4_longweight_2ep'}, 'no_go'),
    ]:
        config = dict(base, **changes)
        cfg = out / f'{name}.config.json'
        cfg.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        load_config(cfg)
        deferred_configs.append({'id': name, 'status': status, 'config': fingerprint(cfg),
                                 'predictions_assembled': False, 'eligible_for_deployment': False})
    if before != {s: fingerprint(ROOT / s) for s in sources}:
        raise ValueError('source changed during candidate assembly')
    catalog = {'version': 1, 'task': 'A/T5', 'source_main': 'b4c07f7',
               'selected_models': {'emotion': base['classifier_dir'], 'response': base['response_adapter_dir'],
                                   'profile_research_candidate': 'weights/profile_encoder_v1_b1_20261026'},
               'sources': before, 'input': input_fp, 'candidates': candidates, 'deferred_configs': deferred_configs,
               'profile_metadata': {'file': fingerprint(profile_path / 'checkpoint_metadata.json'),
                                    'labels': metadata['profile_labels'], 'thresholds': metadata['profile_thresholds'],
                                    'max_length': metadata['max_length']},
               'deferred': [
                   {'id': 'emotion065_only_and_combined', 'status': 'missing_delivery',
                    'model': 'weights/emotion_longtail065_b1_20261028',
                    'reason': 'C T4 internal candidate only; checkpoint and per-ID predictions not delivered locally; retain Day5 default. Requires reload/label/tokenizer/preprocessing checks before assembly.'},
                   {'id': 'response_only_t4_and_combined', 'status': 'no_go',
                    'reason': 'D T4 longweight failed preregistered gate; retain D4 v2; existing comparison predictions remain in results/response_gen/.'},
                   {'id': 'emotion_or_profile_conditioned', 'status': 'no_go',
                    'reason': 'D T5 conditional decision NO_TRIGGER; do not reuse unconditioned replies as conditioned results.'}],
               'limitations': ['Offline field assembly only; no model reload or formal end-to-end timing.',
                               'Existing full validation set is not a blind test.',
                               'Profile v1 output fields do not condition D; candidate does not change deployment defaults.']}
    (out / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'candidates': len(candidates), 'samples_each': len(inputs), 'catalog': str(out / 'catalog.json')}))

if __name__ == '__main__':
    main()
