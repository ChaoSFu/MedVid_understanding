# Stage 2 v1.2: question-relevant hypotheses with bound image references

This is a new observation-stage contract. It consumes the existing frozen
Stage-1 v1.1 bank and never reruns, rewrites or relabels that bank. V1/v1.1
contracts, checkpoints and raw failures remain historical artifacts. There are
no GT, spatial-intervention, verification, calibration or certificate calls.

## Why this revision exists

The complete v1.1 smoke diagnosis contained 22 executed, JSON-parseable
responses: 3 accepted samples and 19 failures. Frame references failed in 14
samples, question quotes in 5, requirement references in 3, one type/evidence
pair in 1, and the overbroad future-action rule in 1 (counts overlap). All 14
frame-reference failures used numbers consistent with local image indices.
That observation motivates a new index contract, not retrospective conversion
of those failed outputs. Three accepted outputs also had lexical atomicity or
verification-language risks. JSON validity is not hypothesis quality.

## Model output and deterministic binding

The model returns `generation_status`, `claims`, `coverage_notes`. Every claim
has exactly `claim_id`, `text`, `type`, `evidence_type`, `requirement_ids`,
`depends_on`, `status`, `generation_basis`, `question_quote`, `image_indices`,
`temporal_role`, `target_alignment`. Claim status must be UNVERIFIED.

`image_indices` are zero-based integers indexing the actual supplied image
list. The prompt lists allowed indices and no longer presents competing source
frame numbers. Original order/path/duplicate-source-list provenance remains in
the frozen sampling artifact. After all hard checks pass, the program adds:

- `frame_orders`: exact source-list orders looked up from the sampler;
- `frame_bindings`: image index, original/resolved path, SHA, source-list
  indices, sampled_video_frames, image size and available timestamp metadata;
- `quality_review_flags`: lexical risk codes, never truth verdicts.

For example, with input images mapped to source orders `[0, 17, 80]`, model
indices `[0, 2]` bind to source orders `[0, 80]`. Negative, Boolean, string,
duplicate or out-of-range indices fail. Old `frame_orders` model outputs are
not silently treated as image indices. Validation recomputes bindings and
rejects altered paths, hashes or derived orders.

VISUAL_HYPOTHESIS requires a null question_quote and at least one supplied
image. QUESTION_HYPOTHESIS requires a nonempty exact substring of the ORIGINAL
question, not a requirement description. Every claim, including SCENE_ONLY,
needs a valid, nonempty requirement_ids list. Fewer than six claims is expected;
irrelevant scene filler must not be attached to arbitrary requirement IDs.
Reference checks cannot prove semantic relevance; that remains a quality limit.

## Time role and task rules

Temporal role is independent of OBJECT/ATTRIBUTE/RELATION/ACTION/HIGH_LEVEL:

| Role | Meaning |
| --- | --- |
| OBSERVED_CONTEXT | Candidate observation within supplied frames, including current actions for next-action questions |
| FUTURE_HYPOTHESIS | Explicitly tentative question-derived future alternative, not an observed future event |
| QUESTION_TIME_QUERY | Claim aligned to a question-specified time; requires a verified public frame-time mapping |

Future wording is checked across every claim type. Explicit future hypotheses
need QUESTION_HYPOTHESIS provenance and prospective language. Current ACTION
observations are allowed; they are not automatically treated as predictions.
Lexical detection cannot recognize every implicit future statement. All v1
timestamps remain unresolved, so assertions of query-time alignment fail;
sampled observations can still be proposed with their limited scope.

No query bbox is supplied for region-captioning. Such samples must retain
UNRESOLVED_TARGET_REFERENCE. Zero claims is valid; optional requirement-linked
scene claims must have target_alignment=UNRESOLVED and must not assert that they
identify the object inside the missing query box. No RC_info is opened.

OBJECT prerequisites can accompany ACTION/HIGH_LEVEL evidence. If a linked
action/high-level requirement has only OBJECT claims, the reduction is rejected.
No claim or dependency becomes a verified premise. Explicit verification
language, chained-action/purpose phrases, boundary-definition meta descriptions,
final numerical assessments and cycles fail. Other lexical inference/sparse
scope risks are review flags. This is not a complete semantic or medical audit.

## Four audit layers

Each new raw completion retains separate schema, binding, task and semantic_risk
results with claim-index paths, expected values and actual values. All evaluable
checks run even when another layer fails. For example, invalid image indices
and quotes no longer hide a missing boxed-target reference. Non-object or
unparseable JSON explicitly marks downstream checks NOT_EVALUABLE.

Hard errors retain the failed raw response and emit no accepted claims for
that sample; there is no favorable-claim filtering. Semantic risk alone can
produce a candidate bank requiring review, never a certificate. Summary adds
failure_reason_counts, semantic_risk_counts (accepted claim flags), and
binding_version. All four layers remain in per-attempt output_contract_audit.

At most one format-only repair is allowed for syntax/shape failures. It cannot
retry a binding/task/quality failure. When original claim texts are identifiable,
every existing declared claim field, proposition, order and count is protected;
semantic edits fail FORMAT_REPAIR_CHANGED_CLAIMS. Unparseable source text cannot
receive an exact proposition-preservation guarantee; both attempts remain
auditable. No hidden labels, GT corrections, parser aliases or infinite retries.

## Upstream compatibility and replay

V1.2 is Stage-2 only. Its protocol explicitly allows the frozen Stage-1 v1.1
protocol SHA. It binds the complete Stage-1 manifest SHA, runtime SHA, per-sample
requirements SHA, Stage-1 sample SHA and original implementation SHA. It does
not require that the Stage-1 implementation match the revised executor. V1
requirements and mismatched runtime manifests are rejected. Frozen v1.1
planning_status/reason/flags propagate unchanged.

Use new output/cache directories. CLI defaults are now Stage 1 v1.1 and Stage 2
v1.2; explicit `--protocol-version` is recommended. Old banks still validate by
their frozen protocol. Historical resume/replay must use the original executor
commit because implementation identity remains part of requests. V1.2 replay
loads no weights, makes zero new calls and compares full sample/flat exports.

## Do not update a checkout running full Stage 1

If Stage 1 is running from an existing server checkout, keep that checkout and
configuration untouched. Fetch code and create an independent checkout:

```bash
cd /home/huihui/codes/MedVid_understanding_relive_spatial
git fetch origin feature/relive-v2-protocol-dev-cases
git worktree add --detach \
  /home/huihui/codes/MedVid_understanding_claimbank_stage2_v12 \
  origin/feature/relive-v2-protocol-dev-cases
cd /home/huihui/codes/MedVid_understanding_claimbank_stage2_v12/relive

source /home/huihui/miniconda3/etc/profile.d/conda.sh
conda activate MedVidU-cu124
export PYTHONPATH=src

RUNTIME=/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/medvidu_claim_bank_v1_20261120_174255/data/runtime_manifest.jsonl
STAGE1_V11=/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/medvidu_claim_bank_v11_smoke_20261120_181359/stage1_smoke22
CONFIG=/home/huihui/codes/MedVid_understanding_relive_spatial/relive/configs/qwen35_9b_claim_bank_v1.yaml
DATA_ROOT=/mnt/hdd3/huihui/hh_datas/MedVidU/valdata
OUT_V12=/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/medvidu_claim_bank_stage2_v12_$(date +%Y%m%d_%H%M%S)

test -f "$RUNTIME"
test -f "$STAGE1_V11/bank_manifest.json"
test -f "$CONFIG"
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv

# If GPU 0 still runs Stage 1, use GPU 1 ONLY after confirming it is available.
GPU=1
CUDA_VISIBLE_DEVICES="$GPU" python -u -m relive.claim_bank --mode stage2 \
  --protocol-version v1.2 --runtime-manifest "$RUNTIME" \
  --stage1-dir "$STAGE1_V11" --config "$CONFIG" \
  --cache-dir "$OUT_V12/cache" --data-root "$DATA_ROOT" \
  --max-frames 12 --per-qa-type 2 --output-dir "$OUT_V12/stage2_smoke22"

python -m relive.claim_bank --mode validate --output-dir "$OUT_V12/stage2_smoke22"

python -m relive.claim_bank --mode stage2 --protocol-version v1.2 --replay-only \
  --replay-from "$OUT_V12/stage2_smoke22" --runtime-manifest "$RUNTIME" \
  --stage1-dir "$STAGE1_V11" --config "$CONFIG" \
  --cache-dir "$OUT_V12/cache" --data-root "$DATA_ROOT" \
  --max-frames 12 --per-qa-type 2 --output-dir "$OUT_V12/stage2_replay22"

cat "$OUT_V12/stage2_smoke22/generation_summary.json"
cat "$OUT_V12/stage2_smoke22/generation_failures.jsonl"
cat "$OUT_V12/stage2_replay22/cache_replay_audit.json"
cmp "$OUT_V12/stage2_smoke22/claim_bank.jsonl" "$OUT_V12/stage2_replay22/claim_bank.jsonl"
```

Keep the exact model configuration used by the prior Stage-2 smoke. The shown
CONFIG is the originally supplied 9B file; if the previous run used a separate
frozen 27B config, use that unchanged file. No model upgrade is part of this
contract experiment. GPU selection remaps the physical device to logical cuda:0.
Do not run concurrently on an occupied GPU solely because weights might fit.

After smoke review and completion of Stage-1 all-sample output, set STAGE1_V11
to that frozen `stage1_all` directory and omit `--per-qa-type 2` for full Stage 2
in a new output directory. Do not rerun Stage 1 or correct its requirements from
GT/downstream outcomes. Inspect upstream requirements for semantic bias as a
separate read-only review.

## Local checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_claim_bank*.py' -v
PYTHONPATH=src python -m relive.claim_bank.smoke --protocol-version v1.2 \
  --config configs/claim_bank_mock_v12.yaml --output-dir /tmp/claim_bank_v12_synthetic_NEW
PYTHONPATH=src python -m unittest discover -s tests
git diff --check
```

All local inference is synthetic. The regression suite covers noncontiguous
source indices, bool/negative/invalid references, both quote provenance rules,
current actions and future roles across all types, hidden task errors, region
abstention, lexical quality checks, immutable Stage-1 reuse, exact replay, bound
reference tampering, and repair budgets/proposition preservation. It does not
establish real VLM quality or clinical correctness.
