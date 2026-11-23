# Stage 2 V3: minimal hypothesis graphs and independent contracts

The product is a **question-driven, unverified hypothesis graph**. V3 never opens
images, searches evidence, grounds boxes, performs KEEP/DROP, verifies facts,
adapts evidence, composes an answer or creates a certificate.

The historical V1/V2/V2.1 modules, prompts and artifacts are preserved. Stage 1
is not rerun. The fixed 22 source indexes and 100 requirements are checked;
question/requirement disagreement stops the affected sample without inference.
The current question has priority over assumptions inside a requirement.

## Boundaries and schema

`python -m relive.hypothesis_graph --help` lists the independent modes.
The closed model JSON Schema is `schemas/hypothesis_graph_stage2_v3.schema.json`.
The versioned prompt is `src/relive/prompts/hypothesis_planner_stage2_v3.txt`.

The model emits `plan_status`, `claims`, `unresolved_references`,
`requirement_issues`, and `uncovered_requirements`. Each of 0–6 hypotheses has
only `claim_id`, `text`, `type`, `role`, `requirement_ids`, `requires`, and
`alternative_to`. Types are OBJECT, ATTRIBUTE, STATIC_RELATION, TEMPORAL_EVENT,
HIGH_LEVEL; roles are TARGET, PREREQUISITE, DISAMBIGUATOR, INTERPRETATION.
The program adds UNVERIFIED / NOT_SEARCHED. No truth flows through graph edges.

REQUIRES points from consumer to prerequisite. It requests prior inspection;
it does not entail the consumer. REQUIRES must be a DAG. ALTERNATIVE_TO is an
explicit contrast link whose same-target/scope mutual exclusivity needs human
review; it is not a prerequisite. Partial retention preserves all remaining
references: descendants/alternatives of rejected nodes are blocked, not rewired.

Execution bindings are excluded from the schema. Numeric instantiations inside
claim text are also refused. Explicit query times are parsed separately from
the public question, never inferred from requirements, IDs or old answers.
Parameterized `t`, `t_i`, `t_j` stay unbound. RC references without a public box
remain UNRESOLVED_TARGET_REFERENCE; an abstract template is not a resolved target.

The validator emits STRUCTURALLY_ACCEPTED, REVIEW_REQUIRED, REJECTED or
BLOCKED_BY_DEPENDENCY **as planning audit states**, separate from visual truth.
Exact normalized duplicates and explicit chains are detectable. Paraphrase
redundancy, atomicity, medical necessity, question relevance and complete
answer discrimination cannot be proved by lexical rules. Semantic metrics stay
REVIEW_REQUIRED with null accuracy values. A legal type alone does not establish
spatial eligibility.

## Frozen references and provenance

`prepare` reads the original V2 completions, verifies the historical rejection,
checks the sealed V2.1 source binding, and optionally validates the original
Stage-1 bank independently. Comparison-only V1/V2/V2.1 claims are physically
quarantined in `comparison_quarantine.jsonl`. The generation code never parses
that file or includes old claims in its requests. Its content hash is checked
as an opaque reference-manifest member. V1 raw attempts are displayed separately
from V1 saved claims.

The public projection has an explicit field allowlist. Prompts contain only
question, qa_type and the unchanged requirement objects. Reference/source
hashes, protocol/prompt/schema/config/implementation hashes and model backend
fingerprint bind generation and cache identity. No GT/evaluation loader is used.

On this workstation the original Stage-1 bank is unavailable: local references
use matching frozen V2/V2.1 copies (and matching V1 copies), explicitly record
`stage1_bank_independently_checked=false`, and cannot publish formal contracts.
The server commands below perform the independent check.
Real generation refuses references without this independent Stage-1 check.

## V2.2 diagnostic: zero calls

On the real saved 84 candidates:

| Diagnostic view | Structural pass | Interpretation |
|---|---:|---|
| A: strip search parameters from a copy | 78 / 84 | Remaining V2 structure only |
| B: retain text/type/requirements/dependencies/status | 84 / 84 | Minimal shape only |
| C: semantic candidate audit | 2 flagged violations; 82 review-required | 0 semantically proven |

A/B are deliberately lossy projections, not repairs or historical re-admission.
All 84 original texts and raw candidates remain in the audit. No diagnostic
candidate is published into V3. The four diagnostic files are separate from
the new generated bank. These counts do not predict real V3 generation quality.

## Runtime, cache and replay

The existing LocalHF/Qwen adapter and CachedInference are reused. The V3 wrapper
calls LocalHF's existing `infer_with_generation_metadata` exactly once per
sample, preserving its decoded response and token/stop metadata in a transport
envelope. No JSON repair, semantic retry, generated-clause rewrite or claim
renumbering occurs. Config uses greedy decoding, max_new_tokens=3072 and zero
retries. The older configs remain unchanged. Raw malformed completions and
technical failures are retained. Missing adapter token counts are null, never
guessed from text. Latency is execution measurement, not semantic quality.

Cache namespace is `<cache-dir>/relive-claimgen-stage2-v3`. New output directories
are mandatory. `--resume` validates input/config/backend identity and immutable
per-sample checkpoints before continuing. Accounting covers **this invocation**;
already checkpointed calls are reported separately. A completed resume returns
the previously stored summary, not a newly executed run.

Replay binds the same config, sources, implementation and backend fingerprint;
it uses a no-inference ReplayBackend and checks terminal cache entries before
access. It never loads model weights. Missing entries fail closed, with no
fallback call. The complete sample JSONL must be byte-identical. Usage summaries
differ in cache hits/new calls/latency as expected. New replay/cache audit files
are permitted; the historical cache and result files are not rewritten.

`mock-smoke` uses generic unresolved event templates based on task category,
not expected claims for the 22 cases. Synthetic flags are sealed and formal
publication is forbidden. The local 22 mock invocations and 22 cache hits prove
data flow only, not clinical correctness or a V3 improvement.

## Independent compiler and review gate

The compiler has no backend, image or certificate API. It deterministically
parses a small controlled-English vocabulary. It never calls another LLM to
recover a proposition. Unknown sentences are UNCOMPILABLE; expanding this
vocabulary later requires a new compiler hash and explicit regression tests.

| Type | Evidence modality | Required semantics |
|---|---|---|
| OBJECT | SINGLE_FRAME | Parseable visible/present target |
| ATTRIBUTE | SINGLE_FRAME | Object plus parseable discriminative state |
| STATIC_RELATION | MULTI_REGION_SINGLE_FRAME | Two entities and relation/contact interface |
| TEMPORAL_EVENT | MULTI_FRAME | Parseable change predicate; at least two observations |
| HIGH_LEVEL | COMPOSITIONAL | Lower-level prerequisites plus explicit reviewed inference requirement |

Presence-only prerequisites cannot establish HIGH_LEVEL interpretation.
No coordinates, source frames or query instantiations are invented:
`candidate_frames=[]`, `spatial_regions=[]`, `time_bindings={}`.

`review-template` emits one row per retained node, bound to the run-manifest,
graph and claim hashes. Humans fill `decision=APPROVE/REJECT/PENDING`,
`reviewer_id`, `rationale`; HIGH_LEVEL also needs `inference_requirement`.
When requirement issues exist, `requirement_issues_resolution` is mandatory.
APPROVE means the planning interpretation is acceptable, **not visually true**.
Do not modify protected text/hash fields. A text change requires a new graph,
never silent rewriting by the compiler.

`compile --dry-run` always writes only diagnostic contracts; the consumable
`evidence_contracts.jsonl` stays empty. `compile --publish --review-jsonl ...`
requires a sealed nonsynthetic executed graph, independent Stage-1 binding,
valid structure, all retained-node approvals, no unresolved references attached
to retained nodes, all nodes compilable and all prerequisites present. Any
blocker yields FORMAL_PUBLICATION_BLOCKED, empty formal JSONL and exit code 2.
No partial official release conceals failed nodes. A no-claim RC case stays
unresolved and contributes no contract. Contracts do not assert whole-question
coverage, nor do approvals establish medical facts.

The downstream Stage-3 interface is ONLY a formally published contract carrying
the unchanged claim/hash, graph/run/review/compiler bindings, modality,
unbound targets and dependency links. A future acquisition module must fill
evidence references in new artifacts and perform its own isolation/admission;
it must reject dry-run, synthetic, UNCOMPILABLE and unresolved-target records.
No Stage-3 executor is implemented here.

## Server: one fixed real smoke, then replay and diagnostic compilation

Use the existing MedVidU-cu124 environment; do not install packages or change the
model/template during the run. Run from a checkout containing this commit.
V2_INPUT is found by full content hash, not guessed from a timestamp. Multiple
matching copies are identical; the lexicographically first path is recorded.
If absent, stop rather than substituting a different experiment.

```bash
source /home/huihui/miniconda3/etc/profile.d/conda.sh
conda activate MedVidU-cu124
cd /home/huihui/codes/MedVid_understanding_relive_spatial/relive
export PYTHONPATH=src

ARTBASE=/mnt/hdd3/huihui/MedVid_understanding/relive_output/real
STAGE1="$ARTBASE/medvidu_claim_bank_v11_smoke_20261120_181359/stage1_smoke22"
LEGACY="$ARTBASE/claim_bank_diagnosis_20261123_092845.json"
CONFIG=configs/qwen35_9b_hypothesis_graph_stage2_v3.yaml
V2_INPUT=$(python - "$ARTBASE" <<'PY'
import hashlib, sys
from pathlib import Path
expected = '93d36bb9063660d48a4689f5631e0e07b0d27375aae521afd492a18f01d0d9f0'
matches = sorted(p for p in Path(sys.argv[1]).rglob('stage2_v2_smoke22.jsonl')
                 if hashlib.sha256(p.read_bytes()).hexdigest() == expected)
if not matches:
    raise SystemExit('Frozen V2 completion file missing; stop.')
print(matches[0])
PY
)
test -n "$V2_INPUT" && test -f "$V2_INPUT" && test -f "$STAGE1/bank_manifest.json" && test -f "$LEGACY" || { echo 'Missing frozen input; stop here.'; false; }

V3_ROOT="$ARTBASE/medvidu_hypothesis_graph_v3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$V3_ROOT"
# Keep a restorable shell state; paths are printed before the real run.
python - "$V3_ROOT/server_variables.sh" "$V3_ROOT" "$STAGE1" "$LEGACY" "$CONFIG" "$V2_INPUT" <<'PY'
import shlex, sys
from pathlib import Path
names = ['V3_ROOT', 'STAGE1', 'LEGACY', 'CONFIG', 'V2_INPUT']
Path(sys.argv[1]).write_text(''.join(f'{k}={shlex.quote(v)}\n' for k,v in zip(names,sys.argv[2:])))
PY

# Zero-call diagnostics and independent frozen-input binding. Stop on errors.
python -m relive.hypothesis_graph --mode diagnostic \
  --v2-input "$V2_INPUT" --output-dir "$V3_ROOT/v2_2_diagnostic" && \
python -m relive.claim_revalidation revalidate \
  --input "$V2_INPUT" --offline --no-model-calls \
  --expected-input-sha256 93d36bb9063660d48a4689f5631e0e07b0d27375aae521afd492a18f01d0d9f0 \
  --output-dir "$V3_ROOT/v21_reference_check" && \
python -m relive.hypothesis_graph --mode prepare \
  --v2-input "$V2_INPUT" --v21-dir "$V3_ROOT/v21_reference_check" \
  --stage1-dir "$STAGE1" --legacy-diagnostic "$LEGACY" \
  --output-dir "$V3_ROOT/references"
```

Check `reference_audit.json`: 22 samples, 100 unchanged requirements, no mismatches,
and `stage1_bank_independently_checked=true`. Otherwise stop and report the
disagreement. The runner also skips any mismatched sample without a request.

```bash
# Fresh real V3: exactly 22 calls when all 22 inputs are bound; no image reads.
CUDA_VISIBLE_DEVICES=0 python -m relive.hypothesis_graph --mode run \
  --references-dir "$V3_ROOT/references" --config "$CONFIG" \
  --cache-dir "$V3_ROOT/cache" --output-dir "$V3_ROOT/run"

# If interrupted, repeat the SAME command with --resume; don't make a new run.
# Replay: 22 cache hits, 0 new calls; no weights loaded.
python -m relive.hypothesis_graph --mode replay \
  --references-dir "$V3_ROOT/references" --config "$CONFIG" \
  --cache-dir "$V3_ROOT/cache" --run-dir "$V3_ROOT/run" \
  --output-dir "$V3_ROOT/replay"

python -m relive.hypothesis_graph --mode validate \
  --references-dir "$V3_ROOT/references" --run-dir "$V3_ROOT/run"

python -m relive.hypothesis_graph --mode compare \
  --references-dir "$V3_ROOT/references" --run-dir "$V3_ROOT/run" \
  --output-dir "$V3_ROOT/comparison"

python -m relive.hypothesis_graph --mode review-template \
  --references-dir "$V3_ROOT/references" --run-dir "$V3_ROOT/run" \
  --output-dir "$V3_ROOT/semantic_review"

python -m relive.hypothesis_graph --mode compile --dry-run \
  --references-dir "$V3_ROOT/references" --run-dir "$V3_ROOT/run" \
  --output-dir "$V3_ROOT/compiler_dry_run"

cat "$V3_ROOT/run/stage2_v3_metrics.json"
cat "$V3_ROOT/replay/stage2_v3_replay_audit.json"
cat "$V3_ROOT/compiler_dry_run/contract_compiler_audit.json"
```

All commands normally exit 0; `COMPLETE` means execution completion, not semantic
acceptance. Validation proves integrity and deterministic structural checks.
Review rejected/raw responses before any additional generation; do not tune
against this frozen smoke and overwrite it. No full Claim Bank or Stage 3 run is
authorized by these commands. The `semantic_review/semantic_review.jsonl` is a
separate human input; preserve the sealed graph and template and save a completed
copy before any future publication request.

Local engineering checks (no GPU):

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_hypothesis_graph_v3.py -v
PYTHONPATH=src python -m unittest discover -s tests
git diff --check
```
