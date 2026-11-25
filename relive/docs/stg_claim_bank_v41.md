# Full STG claim bank using V4.1

This batch wrapper reuses the **unchanged** V4.1 `prompt_for`, protocol, schema,
normalizer and quality gate. The frozen 22-case ablation entry point is untouched.
No repair, critic, images, grounding, KEEP/DROP/CONTROL, final QA answers, certificate
or VERIFIED creation is added. One Stage-2 model attempt per selected record.

## Uploaded data and identity

`data_json/medvidu_task_subsets_trainval_v1/` contains the 780-record original STG
JSON, its 780-row source-index sidecar, and prepared `stg_v41_inputs/`. TAL and dense
captioning raw subsets are not included in this upload.

The prepared GT-free runtime uses the **original 6270-record dataset SHA and source
indices**, not the STG file's reindexed positions. This preserves original sample
keys and public-record hashes, so the already-executed full Stage-1 bank matches.
Duplicate source `id` values are never used as unique identity. Both raw records
and the mapping are SHA checked; raw records are unchanged, including evaluation
answers/annotations. They are never passed whole to the model. Projection uses
the existing explicit allowlist. Image paths need not exist for this text-only job.

The prepared manifest is portable: its source filenames are siblings of the input
directory, not developer/server absolute paths. Every byte of the uploaded raw
STG file and mapping is checked before execution.

## Frozen Stage 1 and failed upstream plans

Use a completed existing Stage-1 bank containing all 780 original sample keys.
The wrapper validates the old bank and matches question, task, original index,
dataset SHA and public-record hash **per record**. It can reuse a 6270-record bank;
it does not require its runtime file hash to equal the new 780-row subset file.
No Stage-1 model call is made.

Default `--on-unplanned-stage1 question-only`: if an original upstream record has
no valid plan and empty requirements, make the same V4.1 single-pass request with
the original question and unchanged empty requirements. Record
QUESTION_ONLY_UPSTREAM_UNPLANNED and upstream_review_required. This is explicitly
not a successful Stage-1 plan; no replacement requirements are invented. Its
statistics are separate from FROZEN_REQUIREMENTS. All retained hypotheses stay
UNVERIFIED / NOT_SEARCHED and no downstream admission is implied.

Use `--on-unplanned-stage1 skip` to omit inference for those records instead; they
still get structured UPSTREAM_NOT_PLANNED records. Missing source keys or mismatched
questions always fail closed, never silently fall back.

## Server execution

From the `relive/` directory in this branch, with the original model environment:

```bash
source /home/huihui/miniconda3/etc/profile.d/conda.sh
conda activate MedVidU-cu124
export PYTHONPATH="$PWD/src"
STG_DATA_DIR="$PWD/../data_json/medvidu_task_subsets_trainval_v1/stg_v41_inputs"
STG_OUT="/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/stg_claim_bank_v41_$(date +%Y%m%d_%H%M%S)"

nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PYTHONPATH" \
  bash scripts/run_stg_claim_bank_v41.sh "$STG_DATA_DIR" "$STG_OUT" \
  > "$STG_OUT.log" 2>&1 &
echo "PID=$! LOG=$STG_OUT.log ROOT=$STG_OUT"
```

Choose an available GPU before starting. The script searches the output root's
parent for an existing nonsynthetic Stage-1 bank with all 780 matching records.
It never launches Stage 1. If zero or multiple compatible banks are found, it
prints the candidates and stops before inference; explicitly pass the desired
bank as the third script argument. Do not pick a bank merely because its name
resembles stage1_all; per-record binding checks still apply.

The script saves `server_variables.sh`, does zero-call preflight, a five-record
smoke and replay, then full 780-record generation and replay. The full run reuses
the same five cached completions: at most 780 total fresh Stage-2 requests, not 785.
It prints progress every ten records and refuses existing output roots.

Manual commands, also used for resuming interrupted runs:

```bash
source "$STG_OUT/server_variables.sh"

python -m relive.stg_claim_bank_v41 --mode run \
  --data-dir "$STG_DATA_DIR" --stage1-bank "$STG_STAGE1" --config "$STG_CONFIG" \
  --cache-dir "$STG_CACHE" --on-unplanned-stage1 question-only \
  --output-dir "$STG_OUT/run" --resume

python -m relive.stg_claim_bank_v41 --mode replay --max-new-calls 0 \
  --data-dir "$STG_DATA_DIR" --stage1-bank "$STG_STAGE1" --config "$STG_CONFIG" \
  --cache-dir "$STG_CACHE" --run-dir "$STG_OUT/run" --output-dir "$STG_OUT/replay"
```

Completed --resume returns original stored accounting. Exact replay uses the
inference-forbidden ReplayBackend, fails on a cache miss and makes zero new calls.
Interrupted partial publication continues into a new child directory; completed
sealed budget-limited PARTIAL runs require a new output with the same cache.
No historical outputs or raw cache records are overwritten. Technical failures
are cached failures, not automatically retried to manufacture successful claims.

## Outputs and checks

- `claim_bank.jsonl`: one row per selected source record, original question,
  frozen requirements, upstream status, raw response, parsed hypotheses and audit.
- `claims.jsonl`: flattened retained unverified hypotheses with original sample key.
- `generation_failures.jsonl`: records without retained hypotheses or unexecuted plans.
- `generation_summary.json`: counts, conditioning groups, status distribution and usage.
- `cache_replay_audit.json`, `gt_isolation_audit.json`, `manifest.json`, checkpoints.

All records receive an artifact; this does **not** guarantee every question yields
a valid claim. Model/schema failure and no valid claim are legitimate recorded
outcomes. Review the conditioning groups and raw content before using this bank.
Warnings and structural acceptance are never medical truth or semantic approval.

```bash
python -m unittest discover -s tests -p test_stg_claim_bank_v41.py -v
python -m unittest discover -s tests
git diff --check
```

The batch configuration has the same frozen Qwen3.5-9B backend as V4.1; only the
cohort-wide accounting budget changes from 22 to 780. No model weights, generation
settings, prompt or gate are changed. Local testing is synthetic; real STG
generation must run on the GPU server, not be inferred from mock results.
