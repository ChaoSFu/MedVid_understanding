"""python -m relive.stg_claim_bank_v41 --help"""
import argparse
import json


def main(argv=None):
    p = argparse.ArgumentParser(description='STG batch using unchanged V4.1 single-pass generation.')
    p.add_argument('--mode', required=True, choices=['prepare', 'find-stage1', 'preflight', 'run', 'replay', 'validate'])
    for n in ('data-dir', 'stage1-bank', 'config', 'cache-dir', 'output-dir', 'dataset-json', 'source-index', 'search-root', 'run-dir'):
        p.add_argument('--' + n)
    p.add_argument('--expected-count', type=int, default=780)
    p.add_argument('--limit', type=int); p.add_argument('--max-new-calls', type=int)
    p.add_argument('--on-unplanned-stage1', choices=['question-only', 'skip'], default='question-only')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(argv)
    def need(*names):
        for n in names:
            if not getattr(a, n): p.error('--' + n.replace('_', '-') + ' required')
    try:
        if a.mode == 'prepare':
            need('dataset_json', 'source_index', 'output_dir')
            from .data import prepare
            result = prepare(a.dataset_json, a.source_index, a.output_dir, a.expected_count)
        elif a.mode == 'find-stage1':
            need('data_dir', 'search_root')
            from .data import find_stage1
            print(find_stage1(a.data_dir, a.search_root)); return 0
        elif a.mode == 'validate':
            need('run_dir')
            from relive.stage2_single_pass_v41.storage import verify
            m = verify(a.run_dir); result = {'status': 'PASS', 'sample_count': len(m['selected_keys']), 'synthetic_test': m['synthetic_test'], 'new_model_calls': 0}
        else:
            need('data_dir', 'stage1_bank', 'config', 'cache_dir', 'output_dir')
            if a.mode == 'replay': need('run_dir')
            from .runner import run
            result = run(data_dir=a.data_dir, stage1_bank=a.stage1_bank, config_path=a.config,
                         cache_dir=a.cache_dir, output_dir=a.output_dir, limit=a.limit, max_new_calls=a.max_new_calls,
                         on_unplanned=a.on_unplanned_stage1, resume=a.resume, dry_run=a.mode == 'preflight',
                         replay_from=a.run_dir if a.mode == 'replay' else None)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 2 if result.get('status') == 'PARTIAL' else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'FAIL', 'reason': str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
                          'repair_calls': 0, 'new_verified_count': 0})); return 2


if __name__ == '__main__': raise SystemExit(main())
