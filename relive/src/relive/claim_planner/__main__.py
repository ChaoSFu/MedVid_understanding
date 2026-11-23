"""Separate Stage 2 V2 entry point; legacy CLI and Stage 1 remain untouched."""
from __future__ import annotations

import argparse
import json

from relive.claim_bank.contracts import ClaimBankError
from relive.config import load_config
from .fixtures import FixtureBackend
from .report import compare
from .runner import generate, validate_run
from .sources import freeze

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['freeze', 'run', 'replay', 'validate', 'compare', 'mock-smoke'], required=True)
    p.add_argument('--runtime-manifest')
    p.add_argument('--stage1-dir')
    p.add_argument('--legacy-diagnostic')
    p.add_argument('--allow-stage2-requirements', action='store_true')
    p.add_argument('--references-dir')
    p.add_argument('--config')
    p.add_argument('--cache-dir')
    p.add_argument('--run-dir')
    p.add_argument('--replay-from')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    def need(*names):
        for name in names:
            if not getattr(a, name):
                p.error('--' + name.replace('_', '-') + ' is required')
    try:
        if a.mode == 'freeze':
            need('runtime_manifest', 'legacy_diagnostic')
            result = freeze(runtime_manifest=a.runtime_manifest, legacy_diagnostic=a.legacy_diagnostic,
                            output_dir=a.output_dir, stage1_dir=a.stage1_dir, allow_stage2_requirements=a.allow_stage2_requirements)
        elif a.mode == 'validate':
            need('references_dir')
            result = validate_run(a.output_dir, a.references_dir)
        elif a.mode == 'compare':
            need('references_dir', 'run_dir')
            result = compare(references_dir=a.references_dir, run_dir=a.run_dir, output_dir=a.output_dir)
        else:
            need('references_dir', 'config', 'cache_dir')
            if a.mode == 'replay':
                need('replay_from')
            elif a.replay_from:
                p.error('--replay-from is allowed only in replay mode')
            backend = None
            if a.mode == 'mock-smoke':
                cfg = load_config(a.config)
                if cfg['backend']['kind'] != 'mock':
                    raise ClaimBankError('MOCK_CONFIG_REQUIRED')
                backend = FixtureBackend(cfg['backend'])
            result = generate(references_dir=a.references_dir, config_path=a.config, cache_dir=a.cache_dir,
                              output_dir=a.output_dir, resume=a.resume, replay_from=a.replay_from,
                              dry_run=a.dry_run, _backend=backend)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        code = str(exc).split(':', 1)[0] if isinstance(exc, ClaimBankError) else type(exc).__name__
        print(json.dumps({'status': 'FAIL', 'error': code, 'new_verified_count': 0}))
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
