#!/usr/bin/env bash
# Single-pass only; old A/B0/C outputs stay read-only.
set -euo pipefail
if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo 'Usage: bash scripts/run_stage2_v41_smoke.sh REFERENCES V31_RUN_OR_DIAGNOSIS V4_B_RUN_OR_PAIRED_JSONL NEW_OUTPUT_ROOT [CONFIG]' >&2
  exit 2
fi
RELIVE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$RELIVE_DIR"
export PYTHONPATH="$RELIVE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
REFS=$1
V31_RESULTS=$2
B0_RUN=$3
V41_ROOT=$4
V41_CONFIG=${5:-"$RELIVE_DIR/configs/qwen35_9b_stage2_v41.yaml"}
test -f "$REFS/manifest.json"
test -e "$V31_RESULTS"
test -e "$B0_RUN"
test -f "$V41_CONFIG"
mkdir -- "$V41_ROOT"
python - "$V41_ROOT/server_variables.sh" "$REFS" "$V31_RESULTS" "$B0_RUN" "$V41_ROOT" "$V41_CONFIG" "$PYTHONPATH" <<'PY'
from pathlib import Path
import shlex, sys
names = ['REFS', 'V31_RESULTS', 'B0_RUN', 'V41_ROOT', 'V41_CONFIG', 'PYTHONPATH']
with Path(sys.argv[1]).open('x') as f:
    for name, value in zip(names, sys.argv[2:]): f.write(f'export {name}={shlex.quote(value)}\n')
PY
python -u -m relive.stage2_single_pass_v41 --smoke22 --arms A,B0,B1 \
  --stage1-bank "$REFS" --baseline-v31 "$V31_RESULTS" --baseline-b "$B0_RUN" \
  --config "$V41_CONFIG" --dry-run --output-dir "$V41_ROOT/preflight"
python -u -m relive.stage2_single_pass_v41 --smoke22 --arms A,B0,B1 \
  --stage1-bank "$REFS" --baseline-v31 "$V31_RESULTS" --baseline-b "$B0_RUN" \
  --config "$V41_CONFIG" --cache-dir "$V41_ROOT/cache" \
  --max-new-calls 22 --output-dir "$V41_ROOT/run"
python -u -m relive.stage2_single_pass_v41 --smoke22 --arms A,B0,B1 \
  --stage1-bank "$REFS" --baseline-v31 "$V31_RESULTS" --baseline-b "$B0_RUN" \
  --config "$V41_CONFIG" --cache-dir "$V41_ROOT/cache" \
  --replay --reuse-run "$V41_ROOT/run" --max-new-calls 0 --output-dir "$V41_ROOT/replay"
python - "$V41_ROOT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
m = json.loads((p / 'run/ablation_A_B0_B1_metrics.json').read_text())
r = json.loads((p / 'replay/cache_replay_audit.json').read_text())
assert m['sample_count'] == 22 and m['requirement_count'] == 100
assert m['normal_generation_calls_this_invocation'] <= 22
assert all(m[k] == 0 for k in ('repair_calls', 'critic_calls', 'image_reads', 'certificate_writes', 'new_verified_count'))
assert m['gt_used'] is False and m['stage1_requirements_unchanged']
assert r['status'] == 'PASS' and r['new_model_calls'] == 0
assert (p / 'run/smoke22_results.jsonl').read_bytes() == (p / 'replay/smoke22_results.jsonl').read_bytes()
print(json.dumps({'engineering_checks': 'PASS', 'B1_execution': m['B1_execution'],
                  'semantic_quality': 'NOT_REVIEWED', 'report': str(p / 'run/ablation_A_B0_B1.html')}))
PY
