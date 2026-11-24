# Stage 2 V4-Lite: paired generation and conditional repair

This opt-in module does not modify V3/V3.1, the Contract Compiler, evidence
acquisition, verification, adaptation, answers, pilot tags or historical outputs.
The model receives only the public question, task type and the unchanged frozen
requirements. Repair additionally receives the CURRENT B graph and its gate
diagnostics. Arm A is quarantined to comparison; its claims never enter B/C
prompts. No images, GT or reference-answer records are opened by this runner.
All retained claims remain UNVERIFIED / NOT_SEARCHED. No certificates are written.

## Architecture and limits

One generation call per sample; one conditional repair call at most. Both use
the existing frozen adapter, GenerationEnvelope, content-addressed CachedInference
and greedy, zero-technical-retry settings. The real config has an identical
backend to V3.1, including max_new_tokens=3072. LocalHF has no reliable general
JSON-schema decoder in this adapter, so the prompt uses field rules plus a small
result instance, followed by strict parsing. We do not claim constrained decoding.

The gate checks closed JSON shape, IDs, references, cycles, numeric execution
bindings, duplicates, missing targets and dependency closure. Only documented
RELATION->STATIC_RELATION and ACTION->TEMPORAL_EVENT enum aliases are normalized,
with an audit; prose is never silently rewritten. The schema safety cap is 12.
Ordinary questions allow 6; explicit dense-event lists may raise that to 12.
The event list is read from the QUESTION, never from benchmark labels/GT.
More than 12 explicit classes cannot be fully represented and must remain gaps.

Evidence slots carry UNBOUND information needs and are not counted as claims.
Missing RC query ROI remains UNRESOLVED_TARGET_REFERENCE, without guessed targets.
Question time parameters stay in the frozen inputs; generator/repair emit symbolic
time only, never execution bindings or invented frame coordinates.

Heuristic flags cover compound text, placeholders, unbounded negatives, continuity,
clinical judgments, entity ambiguity, dependencies and alternatives. Connectives
are review flags, not proof of a semantic violation. ACCEPTED_STRUCTURAL means
only that no implemented structural defect/heuristic flag was found. Relevance,
atomicity, specificity, dependency necessity and medical correctness all remain
NOT_REVIEWED until a human reviews them. Lexical event-name gaps and unknown
dependency semantics alone do not trigger repair. Known structure defects and
localized high-risk text shapes can trigger one patch. UNVERIFIED never does.

Patches support KEEP, SPLIT, MERGE, DROP, DEMOTE_TO_SLOT, FIX_EDGE, FIX_TYPE.
They are transactional, preserve unaffected nodes, and must explicitly update
affected edges. A root replacement needs a recorded reason and an unrepairable
root/empty invalid plan. Patches introducing structural failures or NEW code
flags are rolled back; unresolved old risks are preserved. Code-flag clearance
is not proof of semantic improvement. All proposals, reasons, original/repaired
graphs, operations, costs and rollback reasons are kept in repair_audit.jsonl.

The existing Contract Compiler is intentionally not invoked. Potential spatial
types are counted only as diagnostics: no human semantic admission, compiled
execution contract, spatial experiment admission or truth propagation is implied.

## Inputs and freeze checks

`--stage1-bank` accepts an original frozen Stage-1 bank, or the existing V3
references directory (`manifest.json`, `public_references.jsonl`). The runner
requires exactly the original SMOKE22 indices and 100 requirements, checks every
question/requirement against A, and records their aggregate hash plus input hashes.
An executed, independently bound, nonsynthetic Stage-1 source is required for real
calls. A references directory with `stage1_independently_checked=false` is sufficient
for local preflight/mock only; do not edit that flag. Recreate NEW references from
the original bank using the existing prepare command if needed.

`--v31-results` accepts the original sealed V3.1 run directory or the complete
`stage2_v31_smoke22_diagnosis.json`. A sealed run must use its original references
directory. A diagnosis is checked by replaying the OLD deterministic V3.1 validator
in memory; historical raw/retained/rejected records are preserved. Model/config
comparability is UNKNOWN if the diagnostic lacks a frozen backend fingerprint;
do not assume equal configuration. No old model calls are repeated.

## Local tests and no-model execution

From `relive/`, using an environment with the repository's existing dependencies:

```bash
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -p test_stage2_lite_ablation.py -v
python -m unittest discover -s tests
git diff --check

# Variables must point to existing inputs; no raw QA/answer dataset is needed.
python -m relive.stage2_lite_ablation --smoke22 \
  --stage1-bank "$REFS" --v31-results "$V31_RESULTS" \
  --dry-run --output-dir "$V4_ROOT/preflight"

# Same full B/C/replay flow with the explicitly synthetic adapter.
bash scripts/run_stage2_v4_lite_smoke.sh "$REFS" "$V31_RESULTS" \
  "$NEW_MOCK_ROOT" configs/stage2_v4_lite_mock.yaml
```

Synthetic outputs are labelled SYNTHETIC_TEST_ONLY. They establish engineering
data flow, never real-model quality improvement. Mock completions are generic
test placeholders, not a proposed medical knowledge or hypothesis bank.

## GPU server: execute B, then conditional C, then exact replay

Use the original V3.1 environment and references, and an available GPU. The
following script does not rerun Stage 1 or A. Pass a NEW output root (its parent
must exist). It refuses an existing root and stops on command failure.

```bash
source /home/huihui/miniconda3/etc/profile.d/conda.sh
conda activate MedVidU-cu124
# cd to relive/ in a checkout containing this branch.
export PYTHONPATH="$PWD/src"

# Restore REFS from the ORIGINAL V3.1 server_variables.sh.
V31_ROOT=/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/medvidu_hypothesis_graph_v31_20261124_101317
test -f "$V31_ROOT/server_variables.sh"
source "$V31_ROOT/server_variables.sh"
V31_RESULTS="$V31_ROOT/run"
test -f "$REFS/manifest.json"
test -f "$V31_RESULTS/manifest.json"

V4_ROOT="/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/stage2_v4_lite_$(date +%Y%m%d_%H%M%S)"
CONFIG="$PWD/configs/qwen35_9b_stage2_v4_lite.yaml"

nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PYTHONPATH" \
  bash scripts/run_stage2_v4_lite_smoke.sh "$REFS" "$V31_RESULTS" "$V4_ROOT" "$CONFIG" \
  > "$V4_ROOT.log" 2>&1 &
echo "PID=$! LOG=$V4_ROOT.log ROOT=$V4_ROOT"
```

If your original V3.1 directory has a different name, restore its real path;
never manufacture missing references or regenerate requirements. If the model
path differs, make a new versioned config, document the difference from A, and
pass it explicitly. `--model` is an identity assertion, not a hidden config edit.

The script runs these paired phases (up to 22 + 22 real calls):

```bash
python -m relive.stage2_lite_ablation --arms A,B --smoke22 \
  --stage1-bank "$REFS" --v31-results "$V31_RESULTS" --config "$CONFIG" \
  --cache-dir "$V4_ROOT/cache" --max-new-calls 22 --output-dir "$V4_ROOT/B"

python -m relive.stage2_lite_ablation --arms A,B,C --smoke22 \
  --stage1-bank "$REFS" --v31-results "$V31_RESULTS" --config "$CONFIG" \
  --cache-dir "$V4_ROOT/cache" --reuse-run "$V4_ROOT/B" --max-new-calls 22 \
  --output-dir "$V4_ROOT/C"

python -m relive.stage2_lite_ablation --arms A,B,C --smoke22 \
  --stage1-bank "$REFS" --v31-results "$V31_RESULTS" --config "$CONFIG" \
  --cache-dir "$V4_ROOT/cache" --reuse-run "$V4_ROOT/C" --replay --max-new-calls 0 \
  --output-dir "$V4_ROOT/replay"
```

Replay uses the inference-forbidden ReplayBackend and fails on any cache miss or
changed per-sample B/C result. It does not load model weights. Invocation accounting
is separate from historical completion costs. A B-only run and its C extension
share exactly the same B hashes; C never calls the generator again.

## Resume, offline and immutability

- Interrupted run: repeat its command with `--resume`. Validated write-once B/C
  checkpoints and immutable raw cache prevent repeating completed calls.
- Budget exhausted: exit 2 and seal PARTIAL artifacts. `--resume` creates a new
  `continuations/<time>/` run and reuses prior B/C checkpoints; it never overwrites
  the partial report. It prints the actual new output directory.
- Completed `--resume` returns saved original usage, not evidence of a zero-call
  replay. Use `--replay` for the latter.
- `--offline --reuse-run <run>` rebuilds reports from existing outputs, without
  opening inference/cache. Supply the same config to check frozen identity.
- `--skip-repair` produces B without C. `--force-new-run` creates a timestamped
  child run instead of changing any existing outputs.
- Config, prompts, schema, implementation, source files and public inputs are
  hashed. A changed model/prompt/code cannot silently reuse an old run.

## Report and human evaluation

Every run emits manifest.json, A/B/B-raw/C (requested arms), gate/repair audits,
ablation_metrics.json, replay/GT-isolation audits, full natural-language HTML and
Markdown comparison, claim/graph review CSVs, and deterministic blind claim export
with a separate identity mapping. B-raw is diagnostic only; rejected raw candidates
are available for assessing gate false positives, never downstream publication.

Review CSV decisions are blank with review_status=NOT_REVIEWED. Allowed human
values are PASS / FAIL / UNCERTAIN. Review atomicity, relevance, verifiability,
redundancy, dependency necessity, unsupported specificity, placeholders and graph
answer sufficiency. Do not infer semantic success from automatic risk counts.
Blind export hides arm labels; related B/C rows are paired, not independent trials.

Costs distinguish historical A, B generation, and C incremental repair. Output
tokens and latency are measured only when adapter metadata exists. Input tokens,
API/GPU cost and cost per human-approved claim remain null when unavailable.
LocalHF lacks reliable input-token/billing metadata here; no estimates are invented.

The real B/C experiment and semantic improvement conclusions remain PENDING until
the GPU server run and human review are actually performed.
