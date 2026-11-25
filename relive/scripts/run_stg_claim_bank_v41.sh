#!/usr/bin/env bash
# STG single-pass generation. Never reruns Stage 1 or any image/verification task.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo 'Usage: bash scripts/run_stg_claim_bank_v41.sh DATA_DIR NEW_OUTPUT_ROOT [FROZEN_STAGE1_BANK] [CONFIG]' >&2
  exit 2
fi
RELIVE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$RELIVE_DIR"
export PYTHONPATH="$RELIVE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
STG_DATA_DIR=$1
STG_OUT=$2
STG_STAGE1=${3:-}
STG_CONFIG=${4:-"$RELIVE_DIR/configs/qwen35_9b_stg_claim_bank_v41.yaml"}
STG_CACHE="$STG_OUT/cache"
test -f "$STG_DATA_DIR/manifest.json"
test -f "$STG_CONFIG"
if [[ -z "$STG_STAGE1" ]]; then
  if ! STG_STAGE1=$(python -m relive.stg_claim_bank_v41 --mode find-stage1 \
       --data-dir "$STG_DATA_DIR" --search-root "$(dirname -- "$STG_OUT")"); then
    printf '%s\n' "$STG_STAGE1" >&2
    exit 2
  fi
fi
test -f "$STG_STAGE1/bank_manifest.json"
printf 'Using frozen Stage 1: %s\n' "$STG_STAGE1"
mkdir -- "$STG_OUT"
python - "$STG_OUT/server_variables.sh" "$STG_DATA_DIR" "$STG_STAGE1" "$STG_OUT" "$STG_CONFIG" "$STG_CACHE" "$PYTHONPATH" <<'PY'
from pathlib import Path
import shlex, sys
names=['STG_DATA_DIR','STG_STAGE1','STG_OUT','STG_CONFIG','STG_CACHE','PYTHONPATH']
with Path(sys.argv[1]).open('x') as f:
    for name,value in zip(names,sys.argv[2:]): f.write(f'export {name}={shlex.quote(value)}\n')
PY
python -u -m relive.stg_claim_bank_v41 --mode preflight \
  --data-dir "$STG_DATA_DIR" --stage1-bank "$STG_STAGE1" --config "$STG_CONFIG" \
  --cache-dir "$STG_CACHE" --output-dir "$STG_OUT/preflight"
python -u -m relive.stg_claim_bank_v41 --mode run --limit 5 \
  --data-dir "$STG_DATA_DIR" --stage1-bank "$STG_STAGE1" --config "$STG_CONFIG" \
  --cache-dir "$STG_CACHE" --on-unplanned-stage1 question-only --output-dir "$STG_OUT/smoke5"
python -u -m relive.stg_claim_bank_v41 --mode replay --limit 5 --max-new-calls 0 \
  --data-dir "$STG_DATA_DIR" --stage1-bank "$STG_STAGE1" --config "$STG_CONFIG" \
  --cache-dir "$STG_CACHE" --run-dir "$STG_OUT/smoke5" --output-dir "$STG_OUT/smoke5_replay"
python -u -m relive.stg_claim_bank_v41 --mode run \
  --data-dir "$STG_DATA_DIR" --stage1-bank "$STG_STAGE1" --config "$STG_CONFIG" \
  --cache-dir "$STG_CACHE" --on-unplanned-stage1 question-only --output-dir "$STG_OUT/run"
python -u -m relive.stg_claim_bank_v41 --mode replay --max-new-calls 0 \
  --data-dir "$STG_DATA_DIR" --stage1-bank "$STG_STAGE1" --config "$STG_CONFIG" \
  --cache-dir "$STG_CACHE" --run-dir "$STG_OUT/run" --output-dir "$STG_OUT/replay"
python - "$STG_OUT" <<'PY'
from pathlib import Path
import json,sys
p=Path(sys.argv[1]);m=json.loads((p/'run/generation_summary.json').read_text())
r=json.loads((p/'replay/cache_replay_audit.json').read_text())
assert m['sample_count']==780 and m['complete_source_coverage']
assert m['usage_this_invocation']['calls']<=780
assert all(m[k]==0 for k in ('repair_calls','critic_calls','stage1_new_calls','image_reads','certificate_writes','new_verified_count'))
assert r['status']=='PASS' and r['new_model_calls']==0
assert (p/'run/claim_bank.jsonl').read_bytes()==(p/'replay/claim_bank.jsonl').read_bytes()
print(json.dumps({'engineering_checks':'PASS','sample_count':m['sample_count'],'synthetic_test':m['synthetic_test'],
                  'claims':str(p/'run/claims.jsonl'),'semantic_quality':'NOT_REVIEWED'}))
PY
