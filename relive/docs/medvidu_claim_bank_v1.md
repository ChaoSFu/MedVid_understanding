# MedVidU Question-Conditioned Claim Bank v1

Scientific product: **question-conditioned, observation-informed, unverified
candidate hypotheses**. This module does not verify claims, localize boxes,
perform spatial interventions, emit assessments or final QA answers, or create
certificates. It does not replace the historical RC-answer-derived atomic-claim
pipeline. The new generator never imports that pipeline.

## Boundaries and reuse

The existing `data.medvidu` loader deliberately reports native QA tasks as
unsupported for its legacy claim-verification runner. Claim Bank uses a separate
explicit projection in `claim_bank/data.py`; it reuses `LocalHFBackend`,
`CachedInference`, `Budget`, `ArtifactStore`, strict duplicate-key/nonfinite JSON
parsing, and the existing frozen checkpoint/template checks. No historical
runner, pilot artifact, cache, certificate policy or scientific status is changed.

Source JSON is deserialized once in audit/prepare, but only whitelisted values
are accessed. For a non-human conversation turn, its `value` is never accessed.
Allowed metadata keys are `video_id`, `fps`, `start_frame`, `end_frame`,
`input_video_start_frame`, `input_video_end_frame`, `input_video_start_time`,
`input_video_end_time`. Unknown metadata is discarded. `id` is retained for
traceability only; the unique key binds dataset byte SHA-256 and zero-based
source index. The complete record is never passed to either model stage.

Runtime is closed-field JSONL with a hash-bound `.provenance.json` sidecar.
Runtime mutations, extra annotation fields, duplicate sample keys, wrong
dataset/manifest hashes, and unknown metadata are rejected. The hash of dataset
bytes identifies the container; it is not a source of claim text. The runtime
does not open the original dataset again. Sentinel and access-guard tests cover
assistant turns, annotations and forbidden metadata through both stages,
requests, prompts and exported artifacts. Public questions may legitimately
mention candidate actions or clinical criteria. This audit cannot establish
that an upstream public question was never contaminated before ingestion.

Stage 1 sends exactly question/qa_type and frozen task/output instructions with
zero images. It returns PLANNED (1–6 requirements) or UNABLE_TO_PLAN (zero).
Stage 2 receives requirements, question, public task context and real selected
image paths. The existing adapter opens RGB pixels and calls multimodal
`generate()`. No frame-path text is a substitute for image input. Missing or
undecodable selected images produce MISSING_FRAMES without stage-2 inference;
there is no automatic search for replacement images.

## Contracts and quality checks

Frozen prompts, output instructions, task policies, types and sampler are in
`contracts.py` and exported in `protocol.json`. Backend fingerprints include
checkpoint/template hashes, installed torch/transformers versions, image
preprocessing and generation settings. Cache requests additionally bind public
row hash, requirements, sampler, prompt, implementation and actual image bytes.
The v1 real config pins greedy generation (`do_sample: false`), 2,048 new tokens,
no technical retries, no-thinking template arguments and a **new explicit
262,144-pixel per-image processor cap**. This differs from historical spatial
runs. Probe the processor and confirm no ignored-template warnings on the GPU
host before real generation. The backend's in-process timeout is advisory.

Requirements: `requirement_id`, `description`, `type`, `evidence_type`,
`importance`. Types are OBJECT/ATTRIBUTE/RELATION/ACTION/HIGH_LEVEL; evidence is
SPATIAL/SPATIAL_MULTI_REGION/TEMPORAL/COMPOSITIONAL. IDs are R1–R6.

Claims: `claim_id`, `text`, `type`, `evidence_type`, `requirement_ids`,
`depends_on`, `status`, `generation_basis`, `question_quote`, `frame_orders`,
`target_alignment`. IDs are H1–H6. Status must be exactly UNVERIFIED; a model
emitting a truth verdict fails validation rather than having its verdict silently
rewritten. QUESTION_HYPOTHESIS requires a verbatim question quote. A
VISUAL_HYPOTHESIS must reference supplied frame orders; these are model-supplied
provenance references, **not verified visual grounding**. All dependency edges,
including HIGH_LEVEL nodes, join unverified hypotheses; no edge treats a node as
an established fact.

Closed schemas, count limits, duplicate propositions/IDs, evidence/type
compatibility, references, DAGs, chained-action/purpose lexical risks and task
policies are checked without GT. Visual actions need multiple observations.
Action/high-level requirements cannot be satisfied solely by OBJECT claims.
Next-action ACTION claims must remain question-derived and prospective.
Region-caption samples always retain UNRESOLVED_TARGET_REFERENCE, with optional
scene claims marked UNRESOLVED. No RC_info or GT box is accessed.

These are structural and conservative lexical checks, not proofs of relevance,
single-predicate semantics, medical validity, candidate-versus-observation
correctness or completeness. Semantic human review remains appropriate before
downstream experiments. Failed raw generations remain in the cache; exported
failure records retain reasons and raw-response references in their sample plan.
Only syntax/shape errors allow one format-repair call using the same inputs and
raw completion. Semantic/status/quality violations do not receive semantic
retries. A repair model may still change a proposition; both raw attempts remain
auditable. No reference answer is used to repair or select claims.

## Frames and time

The first source-list occurrence determines order. Exact duplicate source paths
are removed, while all original list indices and paired sampled_video_frames
values remain recorded. Uniform coverage includes endpoints. Temporal or
compositional requirements use uniform groups of adjacent supplied entries;
these entries are **not asserted to be consecutive native frames**. At most
1–16 images are read per sample (default 12); no GT affects sampling.

V1 uses frame order only. `fps` is retained as public metadata but is not taken
as proof of the source-frame clock or its origin. It never divides filename
numbers by sampled fps. All timestamps and seconds coverage remain null with
UNRESOLVED_TIMEBASE. Question query times stay intact in the question, but do
not steer temporal positions until a future version binds a trustworthy public
per-frame time mapping. Sparse observation and unknown time mapping are recorded
for every stage-2 plan. The supplementary-observation interface is reserved with
`enabled=false`, zero rounds and zero remaining budget; v1 makes no hidden
second observation pass.

## Persistence and execution

Every selected source row gets a plan, including failures and zero claims. A
full unfiltered run covers all source rows; filtered runs explicitly report
`complete_source_coverage=false`. `--end-index` is exclusive; `--qa-type` can
repeat. `--per-qa-type 2` freezes the first two source records of each task before
inference, providing a 22-sample engineering subset without outcome selection.

Immutable per-sample checkpoints bind result hashes. `--resume` requires the
same runtime, config, code, selection and sampling settings. Completed samples
are retained; a partially executed sample recovers content-cache attempts.
Frame modifications are detected on resumed stage 2. A completed resume validates
and returns the existing summary. On an interrupted run, summary usage describes
the final invocation; raw cache records retain earlier calls. Do not read this
number as lifetime GPU expenditure across interruptions.

`--replay-only --replay-from RUN` uses the frozen backend fingerprint, loads no
model weights and checks terminal cache presence before every call. Cache misses
stop with exit 2 and zero inference; they never fall back to generation. Core
cache locks/index recovery may touch cache bookkeeping, but no new raw model
response is produced. Successful replay requires byte-identical sample results
and flattened exports. Replay summary/accounting and manifest hashes naturally
differ from fresh. Cache and artifact directories should be local to one run;
do not run multiple writers into the same output directory.

`--dry-run` produces explicitly nonexecuted plans and no model/cache access.
Dry stage-1 plans cannot be used as generated requirements. A mock stage-1 bank
cannot enter a real stage-2 run. Mock artifacts have SYNTHETIC_TEST_ONLY status
and synthetic provenance and belong in dedicated output/cache directories.

Exit 0 means the batch exported successfully, including scientific abstentions
and recorded per-sample failures. It does not mean every claim was generated or
is correct. Structural/hash/configuration/replay errors exit 2. Inspect summary
status counts before promoting an experiment. No certificate is created.

## Files

`claim_bank.jsonl` contains source identity, question, requirements, hypotheses,
generation status, failure reasons, model/prompt identities, raw response refs,
sampling and image hashes. `requirements.jsonl` and `claims.jsonl` are flattened
by `(sample_key, requirement_id/claim_id)`. Additional required exports are
`frame_sampling_manifest.jsonl`, `generation_failures.jsonl`,
`generation_summary.json`, `protocol.json`, `gt_isolation_audit.json`,
`cache_replay_audit.json`. `run_binding.json`, `samples/*.json` and
`bank_manifest.json` provide immutable bindings and validation. Stage 1 emits
the same export family, with an empty claims file and null frame sampling.

Generation statuses: GENERATED, NO_VALID_CLAIMS, INSUFFICIENT_VISUAL_CONTEXT,
UNRESOLVED_TARGET_REFERENCE, MISSING_FRAMES, MODEL_OR_SCHEMA_FAILURE. A region
reference abstention may still carry explicitly unaligned scene hypotheses.

## Local acceptance

Run from `relive`, with the environment's executable named `python`:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_claim_bank.py' -v
PYTHONPATH=src python -m relive.claim_bank.smoke \
  --output-dir /tmp/relive_claim_bank_synthetic_smoke_NEW
PYTHONPATH=src python -m unittest discover -s tests
git diff --check
```

The smoke creates 22 explicitly synthetic samples, two of every qa_type, with
its own toy images. It exercises stage 1, multimodal pixel loading in the mock
dataflow, stage 2, missing target-reference abstention, validation and byte-exact
zero-call replay. It cannot demonstrate real VLM quality or GPU compatibility.

## GPU server commands

No full GPU experiment has been run locally. Execute sequentially and retain
outputs. These commands assume the previously reviewed Qwen checkpoint and
MedVidU image root on the four-A100 server. **The `(2)` source file must exist;
do not substitute another file without explicitly reviewing its dataset SHA.**

```bash
set -euo pipefail
cd /home/huihui/codes/MedVid_understanding_relive_spatial/relive
source /home/huihui/miniconda3/etc/profile.d/conda.sh
conda activate MedVidU-cu124
export PYTHONPATH=src

DATASET='/home/huihui/codes/MedVid_understanding_relive_spatial/data_json/init_datas/medvidu_eccv2026_trainval(2).json'
DATA_ROOT=/mnt/hdd3/huihui/hh_datas/MedVidU/valdata
CONFIG=configs/qwen35_9b_claim_bank_v1.yaml
BANK=/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/medvidu_claim_bank_v1_$(date +%Y%m%d_%H%M%S)
test -f "$DATASET"
test -d "$DATA_ROOT"
mkdir -p "$BANK"
python -m pip freeze > "$BANK/environment.freeze.txt"
git rev-parse HEAD

# 1. Dataset audit, projection and validation: zero model calls/cache/certificates.
python -m relive.claim_bank --mode prepare \
  --dataset-json "$DATASET" --expected-count 6270 \
  --data-root "$DATA_ROOT" --output-dir "$BANK/data"
RUNTIME="$BANK/data/runtime_manifest.jsonl"
cat "$BANK/data/dataset_audit.json"

# Frozen 22-sample preflight: no model load or image access in stage 1.
python -m relive.claim_bank --mode stage1 --dry-run \
  --runtime-manifest "$RUNTIME" --config "$CONFIG" \
  --cache-dir "$BANK/unused_preflight_cache" --per-qa-type 2 \
  --output-dir "$BANK/stage1_preflight22"

# Real text-only engineering smoke: at most 44 calls including format repairs.
CUDA_VISIBLE_DEVICES=0 python -m relive.claim_bank --mode stage1 \
  --runtime-manifest "$RUNTIME" --config "$CONFIG" --model Qwen3.5-9B \
  --cache-dir "$BANK/cache" --per-qa-type 2 \
  --output-dir "$BANK/stage1_smoke22"

# Real multimodal engineering smoke: at most 44 additional calls.
CUDA_VISIBLE_DEVICES=0 python -m relive.claim_bank --mode stage2 \
  --runtime-manifest "$RUNTIME" --stage1-dir "$BANK/stage1_smoke22" \
  --config "$CONFIG" --cache-dir "$BANK/cache" --data-root "$DATA_ROOT" \
  --max-frames 12 --per-qa-type 2 --output-dir "$BANK/stage2_smoke22"
python -m relive.claim_bank --mode validate --output-dir "$BANK/stage2_smoke22"
cat "$BANK/stage2_smoke22/generation_summary.json"

# STOP and inspect smoke failures/coverage/raw outputs before full generation.

# 2. Stage 1 all source samples: text only, <=12,540 logical calls with repair.
CUDA_VISIBLE_DEVICES=0 python -m relive.claim_bank --mode stage1 \
  --runtime-manifest "$RUNTIME" --config "$CONFIG" \
  --cache-dir "$BANK/cache" --output-dir "$BANK/stage1_all"
python -m relive.claim_bank --mode validate --output-dir "$BANK/stage1_all"

# 3. Stage 2 dev subset, exactly first two public source records per qa_type.
# Cached smoke requests may be reused because requirements/pixels are identical.
CUDA_VISIBLE_DEVICES=0 python -m relive.claim_bank --mode stage2 \
  --runtime-manifest "$RUNTIME" --stage1-dir "$BANK/stage1_all" \
  --config "$CONFIG" --cache-dir "$BANK/cache" --data-root "$DATA_ROOT" \
  --max-frames 12 --per-qa-type 2 --output-dir "$BANK/stage2_dev22"
python -m relive.claim_bank --mode validate --output-dir "$BANK/stage2_dev22"

# 4. Stage 2 all source samples: <=12,540 logical calls with repair.
# Each row remains represented even when planning, frames or generation fail.
CUDA_VISIBLE_DEVICES=0 python -m relive.claim_bank --mode stage2 \
  --runtime-manifest "$RUNTIME" --stage1-dir "$BANK/stage1_all" \
  --config "$CONFIG" --cache-dir "$BANK/cache" --data-root "$DATA_ROOT" \
  --max-frames 12 --output-dir "$BANK/stage2_all"

# 5. Claim Bank validation: no model/cache access.
python -m relive.claim_bank --mode validate --output-dir "$BANK/stage2_all"
cat "$BANK/stage2_all/generation_summary.json"

# 6. Exact replay audit of BOTH stages, no weights loaded, zero new calls.
python -m relive.claim_bank --mode stage1 --replay-only \
  --replay-from "$BANK/stage1_all" --runtime-manifest "$RUNTIME" \
  --config "$CONFIG" --cache-dir "$BANK/cache" --output-dir "$BANK/stage1_replay"
python -m relive.claim_bank --mode stage2 --replay-only \
  --replay-from "$BANK/stage2_all" --runtime-manifest "$RUNTIME" \
  --stage1-dir "$BANK/stage1_all" --config "$CONFIG" --cache-dir "$BANK/cache" \
  --data-root "$DATA_ROOT" --max-frames 12 --output-dir "$BANK/stage2_replay"
cat "$BANK/stage1_replay/cache_replay_audit.json"
cat "$BANK/stage2_replay/cache_replay_audit.json"
cmp "$BANK/stage2_all/claim_bank.jsonl" "$BANK/stage2_replay/claim_bank.jsonl"
cmp "$BANK/stage2_all/claims.jsonl" "$BANK/stage2_replay/claims.jsonl"
```

After an interruption, rerun the exact command for that stage/output with
`--resume`; do not change max-frames, config, roots or selection. The budgets
above are ceilings, not promises of successful hypotheses: missing images,
unplannable tasks and format failures reduce real inference calls. New
experiment variants need new output directories and protocol/config bindings.

For multiple roots, use `--path-map /path/to/local.json` instead of (or alongside)
`--data-root`, with an explicit JSON object such as
`{"/root/data/AVOS":"/local/AVOS"}`. Longest exact directory prefix wins;
traversal and mapped symlink escapes are rejected. No recursive filename search
or alternate-image fallback occurs.

## Limitations and next interfaces

No verified public per-frame timebase is available in v1, and supplemental
observation is reserved but disabled. Fine temporal queries, long clips and
hidden region references can therefore remain unresolved. Lexical quality
checks cannot establish full semantic atomicity or distinguish every candidate
action hallucination. Clinical assessment claims remain hypotheses. Runtime
claim statuses never become SUPPORTED/CONTRADICTED/VERIFIED. No grounding,
verification, adaptation, calibration, answer composition or certificate path
is called by this stage. Do not correct this frozen bank with answers or
downstream KEEP/DROP/IoU results.
