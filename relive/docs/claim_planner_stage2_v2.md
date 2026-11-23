# Stage 2 V2: question-driven hypothesis planning

The new, opt-in `relive-claimgen-stage2-v2` plans question-discriminating
**UNVERIFIED hypotheses** from the public question, `qa_type` and existing
Stage 1 requirements. It does not observe pixels, locate evidence, invoke a
verifier, infer a final answer, or create certificates. The original
`relive.claim_bank` entry point, Stage 1 prompt/parser and observation-based
Stage 2 implementations are unchanged.

## Frozen input and isolation

The selection is exactly the 22 source indices supplied for this task:

```text
0 1 1700 1701 2480 2481 2650 2651 2790 2791 3540 3541
4270 4271 4940 4941 5250 5251 5520 5521 6120 6121
```

`freeze` validates the existing runtime provenance and Stage 1 bank, compares
the Stage 1 requirements with the carried copy in the old Stage 2 diagnostic,
and requires exactly 100 unchanged requirements. A mismatch stops; there is
no regeneration or reconciliation. It preserves each question and binds
canonical question/requirement hashes, dataset hash and source record index.
Duplicate original dataset IDs are never used as unique identities.

If the Stage 1 bank is genuinely inaccessible, the explicit
`--allow-stage2-requirements` option admits the frozen carried copy, compared
against the public runtime question/identity. The reference audit records
`FALLBACK_STAGE2_CARRIED_REQUIREMENTS_ONLY` and
`independent_stage1_bank_checked=false`. This is not an independent assertion
that the inaccessible bank was verified. Supplying an accessible Stage 1 bank
always requires exact agreement; fallback cannot override a mismatch.

The freeze command imports old claims/raw responses into a separate
`legacy_comparison.jsonl` quarantine for reports. The generator reads public
reference rows and projects exactly `question`, `qa_type`, `requirements`
into the prompt. It never passes old claims, metadata, original IDs, paths,
source images, GT answers, `RC_info`, `struc_info`, scores or future results.
It does not open the original QA JSON. No sampler runs. Requests explicitly
contain empty `image_paths` and `frame_ids`.

The full source runtime file is hashed and validated without opening its
image paths. Tests inject forbidden-field and old-claim sentinels and fail
on any pixel access. Public questions/frozen requirements may themselves
contain assumptions; provenance isolation is not proof of their correctness.
The planner is instructed to treat Stage 1 assertions as unverified suggestions,
including avoiding invented exclusivity between camera and instrument motion.

## Schema and graph semantics

The closed JSON Schema is
`schemas/claim_planner_stage2_v2.schema.json`; its canonical content is bound
inside `protocol.json`. Runtime validation implements its emitted subset
without adding a JSON Schema dependency. The top-level model object contains
`planning_status`, `entities`, `claims` (0–6), `information_gaps`.

| Field | Contract |
| --- | --- |
| `entities` | IDs plus an exact input quote containing the entity label, or a `$variable` with unresolved identity. Requirement-origin terms remain unverified. |
| `proposition` | Exactly one typed predicate, subject/object entity IDs, one optional qualifier, polarity and bounded scope. Multiple predicates cannot be encoded as a list. |
| `scope.query_time` | Optional question-quoted seconds/range/stride parameters; values must occur in that quoted question text. No frame/time conversion is performed. |
| `answer_relevance` | REQUIRED, OPTIONAL or PREREQUISITE; a model proposal, not a measured relevance score. |
| `verification_purpose` | Why future verification could discriminate answers; semantic adequacy needs review. |
| `evidence_requirement` | SINGLE_FRAME, MULTI_REGION_SINGLE_FRAME, MULTI_FRAME or COMPOSITIONAL; target entity IDs; optional temporal contract with at least two observations. |
| `dependencies` | Typed edges with rationales; no truth propagation. |
| `interpretation_rule` | HIGH_LEVEL requires a question-grounded rule and lower-level non-presence-only premises. Skill interpretation additionally needs a temporal premise. This does not establish clinical validity. |
| `bindings` | `frame`, `timestamp`, `bbox` all null. |
| `status / evidence_status` | Always UNVERIFIED / NOT_SEARCHED. |

Every dependency is stored on the consumer and references its prerequisite
or supporting-evidence claim. `REQUIRES` is an acquisition/verification
prerequisite, not logical implication. `SUPPORTS` is **conditional on future
verification**, never an assertion that its source is established. HIGH_LEVEL
cannot be used as an already-established premise. All edge types are checked
for cycles and dangling references.

`ALTERNATIVE_TO` is represented as one directed edge, never a reciprocal pair.
This first implementation deterministically admits only the same predicate,
arguments and bounded scope with opposite polarity. More general mutually
exclusive alternatives need a future reviewed competition contract; two
different actions are not automatically exclusive.

The internal predicate families are IS_VISIBLE, HAS_ATTRIBUTE,
STATIC_RELATION, CONTACT, MOTION, CHANGE, EVENT and INTERPRETATION. Actions
require a multi-frame/compositional temporal contract; two static objects
alone cannot satisfy a clinical criterion. STG uses parameterized templates,
not one duplicate sentence per frame/time. Missing public RC targets retain
UNRESOLVED_TARGET_REFERENCE and explicit gaps; optional templates use only
unresolved entities, never invented scene observations.

## Partial success and semantic limits

The new parser is strict JSON: fences, extra text, duplicate JSON keys and
malformed roots fail; no fence trimming, bracket repair or old-result salvage.
This version freezes **zero format repairs and zero technical retries**:
one model call maximum per sample, 22 calls maximum for smoke. All raw output
is retained. A schema issue in one claim does not invalidate unrelated claims.

Individual rejected candidates retain their exact original content, schema
errors, explicit contract errors and semantic-risk warnings. Dependency
cycles and dependents of rejected claims are pruned to a fixed point. IDs,
propositions and wording are never rewritten or renumbered. Partial admission
or an incomplete model plan yields PARTIAL_GENERATED. Root ambiguity cannot
be safely salvaged. Zero claims, incomplete plans and missing references are
legitimate outputs; COMPLETE only denotes a finished batch.

Structured predicate/type/argument checks are primary; deterministic checks
also detect explicit action chains, per-frame duplication, global scope,
double negation and certain type/prose mismatches. Lexical hints alone cannot
prove general atomicity. Text/structure equivalence, clinical logic,
paraphrase redundancy, relevance and unsupported specificity are always
REVIEW_REQUIRED. Arbitrary prose/qualifier hallucinations can escape these
checks. A quoted entity is not proof that an English proposition is correct.
This limit is explicit in every graph audit and in the comparison metrics.

## Immutable artifacts and replay

`freeze` writes `frozen_requirements.jsonl`, `legacy_comparison.jsonl`,
`reference_audit.json`, `manifest.json`. `run` writes:

- `stage2_v2_smoke22.jsonl`: all 22 sample graphs, raw outputs and rejections;
- `stage2_v2_smoke22_summary.json`: status/counts and invocation usage;
- `claims.jsonl`: admitted hypotheses only, still UNVERIFIED;
- `protocol.json`, `run_binding.json`, `gt_isolation_audit.json`;
- `stage2_v2_validation.json`, `cache_replay_audit.json`, `manifest.json`;
- immutable, hash-bound `samples/<sample_key>.json` checkpoints.

The shared `CachedInference`/`ArtifactStore` are reused. The new cache namespace
is `<cache-dir>/relive-claimgen-stage2-v2`, isolated from V1. Identity includes
protocol/prompt/model/backend/decoding/config/question/requirements hashes,
sample identity and implementation hash. Output directories must be new;
explicit `--resume` checks the frozen identity and completed checkpoints.
Resume usage counts only work performed in that invocation, and reports
checkpoint reuse separately. A complete resume returns its frozen summary,
not a new execution claim.

Replay restores the frozen backend fingerprint and never loads weights. Any
cache miss aborts before model execution. Sample JSONL and flattened claims
must be byte-identical, with 0 new calls. Cache locks/index bookkeeping may
be touched; replay is not advertised as zero filesystem writes. No historical
cache is cleared. Preflight `--dry-run` opens no backend or cache.

`compare` writes the requested HTML/Markdown report plus metrics, per-case
deltas and an empty human semantic-review queue. Old failed responses are
displayed separately as Rejected Raw Candidates and never reparsed as success.
V1 used pixels; V2 uses text and a different output contract/budget. The report
is not a matched-input performance comparison. Split/merge/deletion reasons
cannot be inferred from independent generations and remain REVIEW_REQUIRED.
Automatic exact text differences, required-ID coverage and structural graph
validity are reported separately from unknown semantic quality. In particular,
"verification contract coverage" means static schema completeness, not proof
that evidence or an executable operator exists.

## GPU server: same 22 samples only

Run from the updated feature checkout. These paths refer to the user's already
completed runs; all three inputs must exist. The diagnostic JSON must be the
full 22-case file, not an abbreviated terminal summary. Do not substitute
another run or regenerate Stage 1 when a check fails.

```bash
cd /home/huihui/codes/MedVid_understanding_relive_spatial
git fetch origin feature/relive-v2-protocol-dev-cases
git status --short
# Only when this checkout is on the same branch and has no conflicting changes:
git merge --ff-only origin/feature/relive-v2-protocol-dev-cases
cd relive
source /home/huihui/miniconda3/etc/profile.d/conda.sh
conda activate MedVidU-cu124
export PYTHONPATH=src

RUNTIME=/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/medvidu_claim_bank_v1_20261120_174255/data/runtime_manifest.jsonl
S1=/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/medvidu_claim_bank_v11_smoke_20261120_181359/stage1_smoke22
DIAG=/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/claim_bank_diagnosis_20261123_092845.json
CONFIG="$PWD/configs/qwen35_9b_claim_planner_stage2_v2.yaml"
V2_ROOT=/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/medvidu_claim_planner_stage2_v2_$(date +%Y%m%d_%H%M%S)
test -f "$RUNTIME" && test -f "$S1/bank_manifest.json" && test -f "$DIAG" && test -f "$CONFIG" || { echo 'STOP: missing frozen input'; return 2 2>/dev/null || exit 2; }
mkdir "$V2_ROOT"
printf '%s\n' "V2_ROOT=$V2_ROOT"

# 1. Freeze unchanged questions and 100 requirements; 0 model calls/cache writes.
python -m relive.claim_planner --mode freeze \
  --runtime-manifest "$RUNTIME" --stage1-dir "$S1" \
  --legacy-diagnostic "$DIAG" --output-dir "$V2_ROOT/references"

# 2. Text-only preflight; 0 model calls/cache writes. Exit 0 = static PASS.
python -m relive.claim_planner --mode run --dry-run \
  --references-dir "$V2_ROOT/references" --config "$CONFIG" \
  --cache-dir "$V2_ROOT/cache" --output-dir "$V2_ROOT/preflight"

# 3. Fresh text-only smoke, at most 22 new calls; no format/semantic retries.
# Choose an available GPU before starting; logical cuda:0 maps to physical GPU 0.
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -u -m relive.claim_planner \
  --mode run --references-dir "$V2_ROOT/references" --config "$CONFIG" \
  --cache-dir "$V2_ROOT/cache" --output-dir "$V2_ROOT/run" \
  > "$V2_ROOT/stage2_v2_smoke.log" 2>&1 &
printf '%s\n' "$!" > "$V2_ROOT/stage2_v2_smoke.pid"
```

Wait for the fresh process to exit and inspect its summary. Do not start replay
while it is writing. A finished run with failed claims is preserved as such.

```bash
cat "$V2_ROOT/stage2_v2_smoke.log"
cat "$V2_ROOT/run/stage2_v2_smoke22_summary.json"

# 4. Independent static validation, exit 0 = artifact/contract PASS, not truth.
python -m relive.claim_planner --mode validate \
  --references-dir "$V2_ROOT/references" --output-dir "$V2_ROOT/run"

# 5. Replay; must have 0 new model calls and exact sample/claim bytes.
python -m relive.claim_planner --mode replay \
  --references-dir "$V2_ROOT/references" --config "$CONFIG" \
  --cache-dir "$V2_ROOT/cache" --replay-from "$V2_ROOT/run" \
  --output-dir "$V2_ROOT/replay"
cat "$V2_ROOT/replay/cache_replay_audit.json"

# 6. Read-only old/new report, no model calls/cache writes.
python -m relive.claim_planner --mode compare \
  --references-dir "$V2_ROOT/references" --run-dir "$V2_ROOT/run" \
  --output-dir "$V2_ROOT/comparison"
```

All CLI modes exit 2 on binding/schema/IO/argument failure and must not be
followed by dependent commands until resolved. A run exits 0 for a complete
22-row export even when some samples/candidates failed; examine summary and
validation rather than interpreting the exit code as semantic success.
The new real config reuses the frozen 9B backend identity, with greedy decoding
and a separately bound 6144-token output budget for the larger graph schema.
It must pass the checkpoint/template checks on the server. Do not change it
after looking at results to selectively rerun difficult cases.

If the old Stage 1 bank is unavailable, omit `--stage1-dir` and explicitly add
`--allow-stage2-requirements` to `freeze`; keep the fallback warning in all
reports. The server should prefer independent bank verification.

Return the run summary, validation, replay audit, graph JSONL and complete
comparison directory. The report exposes a per-case pending semantic review
queue; it does not invent deletion/split/merge explanations or quality scores.

## Local test and fixture-only smoke

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_claim_planner_stage2_v2.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_claim_bank*.py' -v
PYTHONPATH=src python -m unittest discover -s tests
git diff --check
```

After freezing references, `--mode mock-smoke --config
configs/claim_planner_stage2_v2_mock.yaml` exercises the same 22 questions and
100 requirements with explicitly synthetic, task-level transport fixtures.
Replay it using the same mock config, then `compare`. The banner and manifests
say SYNTHETIC FIXTURE ONLY. Such a report proves dataflow/immutability and
rendering, not improvement of Cases 02/04/05/20. Real changes in hypotheses
cannot be reported until the server has produced real V2 outputs.
