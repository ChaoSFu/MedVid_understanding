"""Offline-only CLI. There is intentionally no model/config/cache-write option."""
import argparse
import json

from .offline import RevalidationError, revalidate

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('revalidate')
    r.add_argument('--input', required=True)
    r.add_argument('--output-dir', required=True)
    r.add_argument('--offline', action='store_true', required=True)
    r.add_argument('--no-model-calls', action='store_true', required=True)
    r.add_argument('--v2-metrics')
    r.add_argument('--v2-html')
    r.add_argument('--expected-input-sha256')
    r.add_argument('--cache-root', help='Optional existing V2 cache namespace root, read-only. Missing/mismatched explicit cache fails closed.')
    r.add_argument('--replay-from', help='Recompute offline and require byte-identical derived artifacts.')
    a = p.parse_args()
    try:
        result = revalidate(input_path=a.input, output_dir=a.output_dir, offline=a.offline, no_model_calls=a.no_model_calls,
                            v2_metrics=a.v2_metrics, v2_html=a.v2_html, cache_root=a.cache_root, replay_from=a.replay_from,
                            expected_input_sha256=a.expected_input_sha256)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        code = str(exc).split(':',1)[0] if isinstance(exc, RevalidationError) else type(exc).__name__
        print(json.dumps({'status':'FAIL','error':code,'new_model_calls':0,'certificate_writes':0,'new_verified_count':0}))
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
