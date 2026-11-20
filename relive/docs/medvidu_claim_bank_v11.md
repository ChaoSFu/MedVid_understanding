# Claim Bank v1.1: separate planning from observation

This revision addresses the first real Stage-1 smoke's output-contract issues.
It changes prompts, task instructions, parser diagnostics and reporting, not the
model, input whitelist, frame sampler, claim truth status, or scientific task.
All existing v1 outputs and raw cache entries remain immutable.

## Diagnosed failure and revised behavior

The reported v1 smoke had 6 GENERATED, 10 MODEL_OR_SCHEMA_FAILURE and 6
NO_VALID_CLAIMS rows. The displayed TAL response used
`planning_status: GENERATED` in both attempts. The shared v1 output instructions
included Stage-2 enums and a missing-visual-context abstention rule even during
question-only planning. Its region-caption task instruction also asked the
planner to return an observation-stage status. These were conflicting cues.
The 10 schema failures share a coarse error code, so the one displayed example
does not establish identical raw causes for all ten. The empty dense-caption
plan is syntactically valid, but does not show that the video lacks evidence.

V1.1 isolates both output contracts and task instructions. Stage 1 never
receives Stage-2 statuses, provenance fields, or rules requiring observed
pixels. It is explicitly told that missing images are expected and are not a
question-level planning blocker. Candidate events remain alternatives.
Dense-caption alternatives can share requirements to respect the six-item
limit. Region-caption planning describes evidence needs with target reference
as a later prerequisite; the stage-2 target-reference restriction remains.

## Stage-1 model output

```json
{
  "planning_status": "PLANNED",
  "planning_reason": null,
  "requirements": [
    {
      "requirement_id": "R1",
      "description": "Determine whether the queried interaction occurs.",
      "type": "ACTION",
      "evidence_type": "TEMPORAL",
      "importance": "REQUIRED"
    }
  ]
}
```

The only planning_status enums are PLANNED and UNABLE_TO_PLAN. PLANNED requires
1–6 requirements and a null reason. UNABLE_TO_PLAN requires zero requirements
and a nonempty explanation of why the question cannot define a visual evidence
need. Abstention is allowed; the runner does not invent a plan or retry a valid
abstention until it produces requirements. Every abstention carries
PLANNING_ABSTENTION_REVIEW_REQUIRED; obvious image-absence explanations also
carry IMAGE_ABSENCE_NOT_VALID_PLANNING_BLOCKER. This lexical flag is not a proof
that every other abstention explanation is justified.

Sample-level reports include planning_status, planning_reason and
planning_review_flags. generation_status remains the existing batch field:
GENERATED for a parsed nonempty plan, NO_VALID_CLAIMS for a parsed empty plan,
MODEL_OR_SCHEMA_FAILURE for a failed contract. **NO_VALID_CLAIMS at Stage 1
means no requirements were formed; it says nothing about visual claim truth.**
Summary adds requirement_count, planning_status_counts,
planning_abstention_count and planning_review_flag_counts. Failure exports also
carry the planning reason/flags. DRY_RUN planning_status is NOT_EXECUTED.

## Bounded format repair

The strict parser never aliases GENERATED to PLANNED. It reports field paths,
actual values and expected enums/keys. A fresh invalid completion can receive
one repair call with explicit Stage-1 instructions. Both calls remain cached.

If the original requirements list is valid under the requirement schema
(including a valid empty list), the repair must preserve it exactly: no
rewriting, additions, removals or reordering. A changed list fails with
FORMAT_REPAIR_CHANGED_REQUIREMENTS. An invalid or unparseable original list
cannot receive this exact-content guarantee; attempts and diagnostics remain
visible for review. A valid UNABLE_TO_PLAN response is not format-repaired.
The displayed v1 repair that removed a requirement would fail this new gate.
No old response is reparsed or imported into v1.1 to make it successful.

## Versions, cache and historical artifacts

New CLI generation defaults to `--protocol-version v1.1`; explicit selection
is recommended in experiment commands. Original `contracts.py` and v1 model
configuration remain available. Contract, parser and prompt IDs, protocol hash
and implementation hash bind new requests. Stage 2 rejects a Stage-1 bank from
a different protocol. Existing white-listed runtime input can be reused.

`--mode validate` detects the frozen v1/v1.1 protocol from its manifest and
checks it without changing any file. Existing v1 smoke banks still validate.
Resume/replay require the exact implementation hash, as before. To resume or
replay a historical v1 inference run, use its original code commit
`18db229898555f0481302dcf5863dc95bb67abdb` in an independent checkout; do not
resume it with this revised executor. A new v1.1 run needs new output/cache
directories, never the interrupted v1 Stage-2 directory.

## Tests and local synthetic smoke

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_claim_bank*.py' -v
PYTHONPATH=src python -m relive.claim_bank.smoke \
  --protocol-version v1.1 --config configs/claim_bank_mock_v11.yaml \
  --output-dir /tmp/relive_claim_bank_v11_synthetic_NEW
PYTHONPATH=src python -m unittest discover -s tests
git diff --check
```

Tests cover all 11 task prompt boundaries, detailed enum errors, missing reason,
one repair, altered-requirement rejection, abstention review flags, unchanged
v1 contract hash, v1 validation, cross-version rejection, sentinel isolation,
byte-identical sample/flat exports and zero-call replay including repair calls.
Synthetic success validates engineering only. Real planning quality still
requires a fresh GPU smoke and inspection of every abstention/failure.

## Server: repeat the SAME public-order 22 samples

Use the same frozen model configuration as the failed smoke to isolate this
contract change. The configuration below is the originally supplied 9B config;
if the prior smoke used a separately frozen 27B config, retain that exact file
instead. Do not change models while comparing contracts.

```bash
cd /home/huihui/codes/MedVid_understanding_relive_spatial
git pull --ff-only origin feature/relive-v2-protocol-dev-cases
cd relive
source /home/huihui/miniconda3/etc/profile.d/conda.sh
conda activate MedVidU-cu124
export PYTHONPATH=src

RUNTIME=/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/medvidu_claim_bank_v1_20261120_174255/data/runtime_manifest.jsonl
CONFIG=configs/qwen35_9b_claim_bank_v1.yaml
DATA_ROOT=/mnt/hdd3/huihui/hh_datas/MedVidU/valdata
BANK_V11=/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/medvidu_claim_bank_v11_smoke_$(date +%Y%m%d_%H%M%S)
test -f "$RUNTIME"
test -f "$CONFIG"

CUDA_VISIBLE_DEVICES=0 python -m relive.claim_bank --mode stage1 \
  --protocol-version v1.1 --runtime-manifest "$RUNTIME" --config "$CONFIG" \
  --cache-dir "$BANK_V11/cache" --per-qa-type 2 \
  --output-dir "$BANK_V11/stage1_smoke22"

python -m relive.claim_bank --mode validate \
  --output-dir "$BANK_V11/stage1_smoke22"

python -m relive.claim_bank --mode stage1 --protocol-version v1.1 --replay-only \
  --replay-from "$BANK_V11/stage1_smoke22" --runtime-manifest "$RUNTIME" \
  --config "$CONFIG" --cache-dir "$BANK_V11/cache" --per-qa-type 2 \
  --output-dir "$BANK_V11/stage1_replay22"

cat "$BANK_V11/stage1_smoke22/generation_summary.json"
cat "$BANK_V11/stage1_smoke22/generation_failures.jsonl"
cat "$BANK_V11/stage1_replay22/cache_replay_audit.json"
cmp "$BANK_V11/stage1_smoke22/claim_bank.jsonl" \
    "$BANK_V11/stage1_replay22/claim_bank.jsonl"
```

Stop here and inspect planning_status_counts, requirement_count, abstention
reasons, schema diagnostics and raw responses. Validation PASS is structural,
not a guarantee of useful plans. Do not force 22 PLANNED responses by retrying.

Only after planning smoke review, run observation with v1.1 requirements:

```bash
CUDA_VISIBLE_DEVICES=0 python -m relive.claim_bank --mode stage2 \
  --protocol-version v1.1 --runtime-manifest "$RUNTIME" \
  --stage1-dir "$BANK_V11/stage1_smoke22" --config "$CONFIG" \
  --cache-dir "$BANK_V11/cache" --data-root "$DATA_ROOT" \
  --max-frames 12 --per-qa-type 2 --output-dir "$BANK_V11/stage2_smoke22"
```

For full generation, retain the v1 commands' parameters but explicitly add
`--protocol-version v1.1` to both stages, use new directories, and ensure all
Stage-1/Stage-2 references point to v1.1 banks. No full real inference is run as
part of this code revision.
