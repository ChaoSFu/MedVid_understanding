"""Run from relive/: PYTHONPATH=src python -m relive.stage2_lite_ablation."""
import argparse
import json


def main(argv=None):
    p = argparse.ArgumentParser(description='V4-Lite paired unverified planning; no image reads or certificates.')
    p.add_argument('--arms', default='A,B,C')
    p.add_argument('--smoke22', action='store_true', help='Fixed 22 only; no other selection is supported.')
    for name in ('stage1-bank', 'v31-results', 'output-dir'):
        p.add_argument('--' + name, required=True)
    for name in ('config', 'cache-dir', 'model', 'reuse-run'):
        p.add_argument('--' + name)
    for name in ('offline', 'resume', 'replay', 'dry-run', 'skip-repair', 'force-new-run'):
        p.add_argument('--' + name, action='store_true')
    p.add_argument('--max-new-calls', type=int, default=44)
    a = p.parse_args(argv)
    arms = tuple(a.arms.split(','))
    if not set(arms) <= {'A', 'B', 'C'}: p.error('--arms accepts A,B,C only')
    try:
        from .runner import run
        result = run(stage1_bank=a.stage1_bank, v31_results=a.v31_results, output_dir=a.output_dir,
                     config_path=a.config, cache_dir=a.cache_dir, model=a.model, arms=arms,
                     dry_run=a.dry_run, offline=a.offline, replay=a.replay, reuse_run=a.reuse_run,
                     resume=a.resume, skip_repair=a.skip_repair, force_new_run=a.force_new_run,
                     max_new_calls=a.max_new_calls)
        # Full metrics are in the output artifact, keep stdout small.
        print(json.dumps({k: result[k] for k in ('status', 'sample_count', 'requirement_count', 'synthetic_test',
                                               'usage_this_invocation', 'output_dir') if k in result}, ensure_ascii=False))
        return 2 if result['status'] == 'PARTIAL' else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'FAIL', 'error_type': type(exc).__name__,
                          'reason': str(exc) if isinstance(exc, ValueError) else 'INPUT_OR_FILESYSTEM_ERROR',
                          'certificate_writes': 0, 'new_verified_count': 0}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
