"""Offline tools and optional paired inference; no API keys in CLI arguments."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description='Question-only matched baseline vs RAG Stage 1 experiment')
    parser.add_argument('--mode', required=True, choices=['freeze', 'index', 'prepare', 'preflight', 'run', 'replay', 'validate', 'compare', 'mock-smoke'])
    parser.add_argument('--knowledge-json'); parser.add_argument('--corpus-dir'); parser.add_argument('--index-dir')
    parser.add_argument('--dataset-json'); parser.add_argument('--runtime-manifest'); parser.add_argument('--config')
    parser.add_argument('--preflight-dir'); parser.add_argument('--run-dir'); parser.add_argument('--cache-dir')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--allow-pending-knowledge', action='store_true')
    parser.add_argument('--expected-sha256'); parser.add_argument('--expected-count', type=int)
    parser.add_argument('--references-root'); parser.add_argument('--parse-cache')
    parser.add_argument('--per-qa-type', type=int, default=2); parser.add_argument('--max-calls', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    def need(*names):
        for name in names:
            if getattr(args, name) is None: parser.error('--'+name.replace('_', '-')+' is required for '+args.mode)
    try:
        if args.mode == 'freeze':
            need('knowledge_json')
            from .corpus import freeze
            result = freeze(args.knowledge_json, args.output_dir, allow_pending=args.allow_pending_knowledge,
                            expected_sha=args.expected_sha256, expected_count=args.expected_count,
                            references_root=args.references_root, parse_cache=args.parse_cache)
        elif args.mode == 'index':
            need('corpus_dir')
            from .corpus import index
            result = index(args.corpus_dir, args.output_dir)
        elif args.mode == 'prepare':
            need('dataset_json')
            from .protocol import prepare
            result = prepare(args.dataset_json, args.output_dir, args.expected_count if args.expected_count is not None else 6270)
        elif args.mode == 'preflight':
            need('runtime_manifest', 'index_dir', 'config')
            from .runner import preflight
            result = preflight(args.runtime_manifest, args.index_dir, args.config, args.output_dir, args.per_qa_type, args.max_calls)
        elif args.mode in ('run', 'replay'):
            need('preflight_dir', 'index_dir', 'cache_dir')
            if args.mode == 'replay': need('run_dir')
            from .runner import run
            result = run(args.preflight_dir, args.index_dir, args.cache_dir, args.output_dir,
                         resume=args.resume, replay_source=args.run_dir if args.mode == 'replay' else None)
        elif args.mode in ('validate', 'compare'):
            need('run_dir', 'preflight_dir', 'index_dir')
            from .runner import validate, compare
            from .common import new_output, separate_output, write_json
            if args.mode == 'validate':
                separate_output(args.output_dir,args.run_dir,args.preflight_dir,args.index_dir)
                result = validate(args.run_dir, args.preflight_dir, args.index_dir)
                write_json(new_output(args.output_dir)/'validation_report.json', result)
            else: result = compare(args.run_dir, args.preflight_dir, args.index_dir, args.output_dir)
        else:
            from .smoke import mock_smoke
            result = mock_smoke(args.output_dir)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 2 if result.get('engineering_gate') == 'FAIL' else 0
    except Exception as exc:
        # No provider response bodies, prompts or credentials in error output.
        safe = str(exc) if isinstance(exc, ValueError) and str(exc).replace('_', '').isalnum() else type(exc).__name__
        detail = {'status': 'FAIL', 'error': safe, 'mode': args.mode, 'output_dir': args.output_dir,
                  'model_calls_not_automatically_retried': True}
        if isinstance(exc, FileNotFoundError): detail['missing_path'] = str(exc.filename)
        print(json.dumps(detail), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
