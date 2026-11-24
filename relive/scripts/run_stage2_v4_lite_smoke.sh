#!/usr/bin/env bash
# New outputs only. Uses the caller's activated Python environment and GPU choice.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo 'Usage: bash scripts/run_stage2_v4_lite_smoke.sh FROZEN_REFERENCES V31_RUN_OR_DIAGNOSIS NEW_OUTPUT_ROOT [CONFIG]' >&2
  exit 2
fi
RELIVE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$RELIVE_DIR"
export PYTHONPATH="$RELIVE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
REFS=$1
BASELINE=$2
RUN_ROOT=$3
CONFIG=${4:-"$RELIVE_DIR/configs/qwen35_9b_stage2_v4_lite.yaml"}
test -f "$REFS/manifest.json"
test -e "$BASELINE"
test -f "$CONFIG"
# mkdir without -p deliberately refuses an existing root.
mkdir -- "$RUN_ROOT"
python -u -m relive.stage2_lite_ablation --smoke22 \
  --stage1-bank "$REFS" --v31-results "$BASELINE" --config "$CONFIG" \
  --dry-run --output-dir "$RUN_ROOT/preflight"
python -u -m relive.stage2_lite_ablation --smoke22 --arms A,B \
  --stage1-bank "$REFS" --v31-results "$BASELINE" --config "$CONFIG" \
  --cache-dir "$RUN_ROOT/cache" --max-new-calls 22 --output-dir "$RUN_ROOT/B"
python -u -m relive.stage2_lite_ablation --smoke22 --arms A,B,C \
  --stage1-bank "$REFS" --v31-results "$BASELINE" --config "$CONFIG" \
  --cache-dir "$RUN_ROOT/cache" --max-new-calls 22 --reuse-run "$RUN_ROOT/B" \
  --output-dir "$RUN_ROOT/C"
python -u -m relive.stage2_lite_ablation --smoke22 --arms A,B,C \
  --stage1-bank "$REFS" --v31-results "$BASELINE" --config "$CONFIG" \
  --cache-dir "$RUN_ROOT/cache" --max-new-calls 0 --replay --reuse-run "$RUN_ROOT/C" \
  --output-dir "$RUN_ROOT/replay"
python - "$RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
b = json.loads((p / 'B/ablation_metrics.json').read_text())
c = json.loads((p / 'C/ablation_metrics.json').read_text())
r = json.loads((p / 'replay/replay_audit.json').read_text())
assert b['cost']['B']['logical_calls_in_result'] == 22
assert c['B_C_shared_generation']
assert c['cost']['B']['new_calls_this_invocation'] == 0
assert c['repair']['executed_calls'] <= 22
assert r['status'] == 'PASS' and r['new_model_calls'] == 0
assert (p / 'C/paired_results.jsonl').read_bytes() == (p / 'replay/paired_results.jsonl').read_bytes()
assert all(c[k] == 0 for k in ('image_reads', 'certificate_writes', 'new_verified_count'))
print(json.dumps({'engineering_checks': 'PASS', 'synthetic_test': c['synthetic_test'],
                  'semantic_quality': 'NOT_REVIEWED', 'report': str(p / 'C/ablation_comparison.html')}, ensure_ascii=False))
PY
