"""V4.1 CLI has no C arm, repair flag, or second model stage."""
import argparse
import json


def main(argv=None):
    p = argparse.ArgumentParser(description='V4.1 single-pass A/B0/B1; no repair, images, GT or verification.')
    p.add_argument('--arms', default='A,B0,B1'); p.add_argument('--smoke22', action='store_true')
    for n in ('baseline-v31', 'baseline-b', 'stage1-bank', 'output-dir'): p.add_argument('--' + n, required=True)
    for n in ('config', 'cache-dir', 'model', 'reuse-run'): p.add_argument('--' + n)
    for n in ('dry-run', 'resume', 'replay', 'offline'): p.add_argument('--' + n, action='store_true')
    p.add_argument('--max-new-calls', type=int, default=22)
    a = p.parse_args(argv)
    try:
        from .runner import run
        result = run(stage1_bank=a.stage1_bank, baseline_v31=a.baseline_v31, baseline_b=a.baseline_b,
                     output_dir=a.output_dir, config_path=a.config, cache_dir=a.cache_dir, model=a.model,
                     arms=tuple(a.arms.split(',')), dry_run=a.dry_run, resume=a.resume, replay=a.replay,
                     offline=a.offline, reuse_run=a.reuse_run, max_new_calls=a.max_new_calls)
        print(json.dumps({k: result[k] for k in ('status', 'B1_execution', 'sample_count', 'requirement_count',
              'usage_this_invocation', 'repair_calls', 'critic_calls', 'output_dir')}, ensure_ascii=False))
        return 2 if result['status'] == 'PARTIAL' else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'FAIL', 'reason': str(exc) if isinstance(exc, ValueError) else 'INPUT_OR_FILESYSTEM_ERROR',
                          'repair_calls': 0, 'critic_calls': 0, 'certificate_writes': 0, 'new_verified_count': 0}))
        return 2


if __name__ == '__main__': raise SystemExit(main())
