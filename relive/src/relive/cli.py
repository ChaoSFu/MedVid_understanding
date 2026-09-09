"""Command-line entrypoints. Evaluation imports GT only in its dedicated handler."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from relive.audits import audit_run
from relive.backends.inspection import inspect_local_hf
from relive.config import load_config
from relive.data.medvidu import PUBLIC_ADAPTER, REPORT_ONLY_ADAPTER, prepare_medvidu
from relive.runner import run


def _emit(value, output: str | None = None) -> None:
    body = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output:
        destination = Path(output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(body, encoding="utf-8")
    sys.stdout.write(body)


def _evaluate(args: argparse.Namespace) -> int:
    # This is intentionally the only runtime command that imports evaluation.
    from relive.evaluation.metrics import evaluate
    _emit(evaluate(args.run_dir, args.ground_truth), args.output)
    return 0


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _emit(run(config, args.runtime, args.output_dir, cache_dir=args.cache_dir,
              max_samples=args.max_samples, phase=args.phase))
    return 0


def _audit(args: argparse.Namespace) -> int:
    outcome = audit_run(args.run_dir)
    _emit(outcome, args.output)
    return 0 if outcome["status"] == "PASS" else 2


def _inspect_local_hf(args: argparse.Namespace) -> int:
    _emit(inspect_local_hf(args.model_path, probe_processor=args.probe_processor,
                           trust_remote_code=args.trust_remote_code), args.output)
    return 0


def _prepare_medvidu(args: argparse.Namespace) -> int:
    outcome = prepare_medvidu(
        args.source_json, args.frame_root, args.output_dir, source_prefix=args.source_prefix,
        adapter=args.adapter, path_audit_scope=args.path_audit_scope, max_samples=args.max_samples,
        public_claim_manifest=args.public_claim_manifest,
    )
    _emit(outcome, args.output)
    return 0 if outcome["status"] == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relive", description="ReliVE-v1 frozen-model evidence verification")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Run bounded GT-isolated runtime samples")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--runtime", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--cache-dir")
    run_parser.add_argument("--max-samples", type=int, default=5)
    run_parser.add_argument("--phase", choices=("smoke", "preflight"), default="smoke",
                            help="smoke permits up to 5 samples; preflight permits up to 10")
    run_parser.set_defaults(handler=_run)
    eval_parser = commands.add_parser("evaluate", help="Independently join optional ground truth")
    eval_parser.add_argument("--run-dir", required=True)
    eval_parser.add_argument("--ground-truth")
    eval_parser.add_argument("--output")
    eval_parser.set_defaults(handler=_evaluate)
    audit_parser = commands.add_parser("audit", help="Audit runtime import boundary and strict sources")
    audit_parser.add_argument("--run-dir", required=True)
    audit_parser.add_argument("--output")
    audit_parser.set_defaults(handler=_audit)
    inspect_parser = commands.add_parser("inspect-local-hf", help="Read checkpoint metadata and GPU compatibility without loading weights")
    inspect_parser.add_argument("--model-path", required=True)
    inspect_parser.add_argument("--probe-processor", action="store_true",
                                help="Instantiate only AutoProcessor with local files; never load model weights")
    inspect_parser.add_argument("--trust-remote-code", action="store_true",
                                help="Permit the explicit processor probe to use checkpoint-provided code")
    inspect_parser.add_argument("--output")
    inspect_parser.set_defaults(handler=_inspect_local_hf)
    prepare_parser = commands.add_parser("prepare-medvidu", help="Audit MedVidU public fields and prepare only explicit public-claim runtime")
    prepare_parser.add_argument("--source-json", required=True)
    prepare_parser.add_argument("--frame-root", required=True)
    prepare_parser.add_argument("--source-prefix", default="/root/data")
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--adapter", choices=(REPORT_ONLY_ADAPTER, PUBLIC_ADAPTER), default=REPORT_ONLY_ADAPTER)
    prepare_parser.add_argument("--path-audit-scope", choices=("selected", "all"), default="selected")
    prepare_parser.add_argument("--max-samples", type=int, default=5)
    prepare_parser.add_argument("--public-claim-manifest")
    prepare_parser.add_argument("--output")
    prepare_parser.set_defaults(handler=_prepare_medvidu)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"ReliVE-v1 error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
