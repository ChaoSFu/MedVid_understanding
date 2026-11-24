# V4.1: single-pass question-driven hypotheses

V4.1 adds a separate `relive.stage2_single_pass_v41` entry point. It accepts A,
B0, B1 only and has no repair/critic option or second inference stage. Legacy
V3.1/V4-Lite code, the old C runner, historical results, tags, Stage 3 and later
systems are unchanged. Do not use the old V4 CLI to run a V4.1 experiment.

## Contract and audit policy

Only the original question, frozen requirements, public task type and lexical
decision targets extracted from the QUESTION enter generation. Historical A/B0
claims are comparison data and never enter the new request. No video, reference
answer, struc_info, RC_info, GT locations, times or scores are read. Hypotheses
always remain UNVERIFIED / NOT_SEARCHED. No certificate or Stage-3 admission is
created. Requirements are fallible guidance; they are neither facts nor ground
truth for checking the generator's medicine.

The prompt prioritizes the question, independently testable predicates, meaningful
observation targets and default-empty edges. It explicitly distinguishes a
declarative UNVERIFIED hypothesis from a verified observation. It gives a complete
seven-field claim INSTANCE plus a complete three-field slot INSTANCE, never a
JSON Schema definition. It requests an internal final check without a reasoning
trace. The frozen LocalHF adapter does not implement a reliable general JSON-schema
decoder, so this uses the documented strict-parser fallback. The model/backend,
checkpoint metadata, chat template and greedy decoding (3072 tokens, thinking
disabled, zero technical retries) are unchanged from V4-Lite. No training occurs.

Ordinary questions allow at most six claims; explicit dense-event lists can raise
the limit to twelve. Every named question decision target must have a hypothesis
or a slot, including those omitted by requirements. Question-only claims may use
`requirement_ids: []`, explicitly review-required; they do not invent IDs or
increase required-ID coverage. No requirement is added, deleted or rewritten.

The two edge arrays are optional in the model schema and mandatory in the canonical
schema. If omitted, deterministic normalization inserts `[]`, meaning **no declared
edge**, with before/after audit. Omitted requires also adds DEPENDENCY_NOT_DECLARED_REVIEW;
textual H references without an edge remain review-required or reject unknown IDs.
No existing edge or text is removed, inferred or rewritten. Null/incorrect edge
types still fail. Only documented ACTION/RELATION enum aliases are normalized.
Canonical raw-shape errors and post-normalization errors are reported separately.

The lightweight gate checks schemas, unique IDs, exact duplicates, references,
cycles, dependency closure, numeric execution bindings, schema echo and RC target
absence. Text warnings are deliberately heuristic, not semantic certifications.
Conjunctions inside a relation need not be independent facts; a generic connective
warning never mandates splitting. Relevance, atomicity, clinical correctness,
dependency necessity and sufficiency remain NOT_REVIEWED. Any retained hypothesis,
even ACCEPTED_STRUCTURAL, is withheld from downstream publication pending review.

## A / B0 / B1 comparison

- A: original V3.1 raw responses and original admission, unchanged.
- B0: original V4-Lite B raw responses and original admission, unchanged. C data
  may be present in an uploaded paired-results file but is never executed or used
  as an input to B1.
- B1: at most one generation request per frozen sample. Technical/model failures
  remain failures, with no retry or model repair.

ALL three arms use the exact same V4.1 deterministic audit on copies. V3.1 root
`unresolved_references` strings are projected into equivalent UNBOUND slot purposes,
with a root-layout audit; its old uncovered-ID metadata is preserved for diagnosis,
not trusted as coverage. Node texts and IDs are unchanged except explicit field
defaults/enum aliases. Historical retained counts remain separately visible.
Recovering missing arrays in B0 is format recovery, never evidence of better B1.

## Input paths

`--stage1-bank`: existing independently checked V3 references directory, or the
original executed Stage-1 bank. Exactly SMOKE22 and 100 requirements are required.
All questions/requirements are hash checked against A AND B0. Local copied
references without independent Stage-1 binding support preflight/mock only.
Never edit a manifest's independent-binding flag to enable real inference.

`--baseline-v31`: original sealed V3.1 run (use its original references directory),
or the complete stage2_v31_smoke22_diagnosis.json.

`--baseline-b`: the completed V4 B or C run directory containing paired_results.jsonl,
or the complete 22-row paired_results.jsonl supplied by the user. A directory is
manifest/hash checked; a paired file must embed matching questions, requirements
and hashes. Original V3.1/V4 validators verify historical admission in memory.
No historical artifact or cache is rewritten. Missing model fingerprints in an
uploaded file are labelled UNKNOWN for environment comparability.

## Tests and local no-model smoke

From `relive/` in a Python environment with the existing Pillow/PyYAML dependencies:

```bash
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -p test_stage2_single_pass_v41.py -v
python -m unittest discover -s tests
git diff --check

python -m relive.stage2_single_pass_v41 --arms A,B0 --smoke22 --offline \
  --stage1-bank "$REFS" --baseline-v31 "$V31_RESULTS" --baseline-b "$B0_RUN" \
  --output-dir "$NEW_BASELINE_AUDIT"

bash scripts/run_stage2_v41_smoke.sh "$REFS" "$V31_RESULTS" "$B0_RUN" \
  "$NEW_MOCK_ROOT" configs/stage2_v41_mock.yaml
```

Mock results are explicitly SYNTHETIC_TEST_ONLY, not a real hypothesis bank or a
medical-quality comparison. Tests cover field defaults and preservation, partial
acceptance, schemas/IDs/types/numeric bindings, DAGs, task policies, sentinel GT
isolation, one call per sample, no repair, failure accounting, budget/resume,
exact replay and complete report generation. Prompt-regression tests do not prove
that a real LLM obeys the prompt; inspect the real outputs and human reviews.

## GPU server: real B1 then exact replay

Use the original model environment and an available GPU. Preserve the actual
V4_ROOT variable from the experiment that produced 90 B0 candidates.

```bash
source /home/huihui/miniconda3/etc/profile.d/conda.sh
conda activate MedVidU-cu124
# cd to relive/ in the codex/relive-stage2-v41 checkout.
export PYTHONPATH="$PWD/src"

V31_ROOT=/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/medvidu_hypothesis_graph_v31_20261124_101317
test -f "$V31_ROOT/server_variables.sh"
source "$V31_ROOT/server_variables.sh"
export PYTHONPATH="$PWD/src"
V31_RESULTS="$V31_ROOT/run"
: "${V4_ROOT:?Restore the completed real V4-Lite experiment directory}"
B0_RUN="$V4_ROOT/B"
test -f "$REFS/manifest.json"
test -f "$V31_RESULTS/manifest.json"
test -f "$B0_RUN/paired_results.jsonl"

V41_ROOT="/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/stage2_v41_$(date +%Y%m%d_%H%M%S)"
V41_CONFIG="$PWD/configs/qwen35_9b_stage2_v41.yaml"
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PYTHONPATH" \
  bash scripts/run_stage2_v41_smoke.sh "$REFS" "$V31_RESULTS" "$B0_RUN" "$V41_ROOT" "$V41_CONFIG" \
  > "$V41_ROOT.log" 2>&1 &
echo "PID=$! LOG=$V41_ROOT.log ROOT=$V41_ROOT"
```

The script creates a NEW output root, saves server_variables.sh for later terminals,
performs zero-call preflight, runs at most 22 new B1 calls, and runs a cache-only
replay with zero new calls. It fails rather than overwrite an existing output root.

Individual commands, for interrupted-run recovery or separate scheduling:

```bash
python -m relive.stage2_single_pass_v41 --arms A,B0,B1 --smoke22 \
  --stage1-bank "$REFS" --baseline-v31 "$V31_RESULTS" --baseline-b "$B0_RUN" \
  --config "$V41_CONFIG" --model Qwen3.5-9B --cache-dir "$V41_ROOT/cache" \
  --max-new-calls 22 --output-dir "$V41_ROOT/run"

python -m relive.stage2_single_pass_v41 --arms A,B0,B1 --smoke22 \
  --stage1-bank "$REFS" --baseline-v31 "$V31_RESULTS" --baseline-b "$B0_RUN" \
  --config "$V41_CONFIG" --cache-dir "$V41_ROOT/cache" --replay \
  --reuse-run "$V41_ROOT/run" --max-new-calls 0 --output-dir "$V41_ROOT/replay"
```

Repeat an interrupted command with `--resume`; saved checkpoints/raw cache prevent
repeating completed calls. A sealed PARTIAL run continues into a new child directory.
A report-publication crash also resumes into a new child, preserving old partial
files. Completed --resume returns original stored usage; it is not replay evidence.
`--offline --reuse-run` regenerates comparisons from saved B1 results, without a
model/cache invocation. Config/source/prompt/code mismatches fail closed.
`--arms C`, `--repair` and budgets above 22 are rejected by the V4.1 entry point.

## Outputs and interpretation

Each run emits the requested manifest, smoke22_results.jsonl, raw_completions/,
normalized_claims.jsonl, quality_gate_audit.jsonl, ablation_A_B0_B1_metrics.json,
HTML/Markdown full-claim comparison, blind claim/graph CSVs with separate mapping,
GT isolation and cache replay audits. Reports show original historical retention
AND common-audit retention; structural status is never a medical truth label.

Human fields are blank, NOT_REVIEWED, accepting PASS/FAIL/UNCERTAIN. Human-Approved
Claim Rate and Graph-Level Sufficiency Rate remain null until real human judgments
exist; invalid dependency edge counts are also unknown. Semantic review is needed
before deciding whether B1 should replace B0 or enter Stage 3.

Costs separately show historical A/B0 and current B1. Frozen backend metadata,
chat-template hash, prompt/protocol hashes, decoded generation parameters, measured
output tokens and latency are saved. Unavailable input-token counts, seed control
and billing costs are not invented. Checkpoint metadata hashes/inventory come from
the existing frozen adapter, not a newly computed full weight-file SHA.

Current local development has no PyTorch/GPU runtime. Real B1 quality, token/latency
results and superiority over B0 remain PENDING until the server experiment and
human review are actually performed. No real result may be substituted with mock.
