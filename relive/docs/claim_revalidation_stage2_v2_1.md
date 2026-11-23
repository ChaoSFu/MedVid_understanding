# Stage 2 V2.1: offline schema/validator recovery

`python -m relive.claim_revalidation revalidate` processes existing V2 raw
completions. It has no generation, backend, model, configuration, image or
cache-write option. `--offline --no-model-calls` are mandatory. It never runs
Stage 1, changes the V2 prompt, acquires evidence, verifies truth or creates
certificates. Historical V2 decisions remain frozen; V2.1 admission is a
separate, explicitly versioned derived result.

## Inputs and immutability

The frozen cohort is exactly source indices
`0,1,1700,1701,2480,2481,2650,2651,2790,2791,3540,3541,4270,4271,4940,4941,5250,5251,5520,5521,6120,6121`.
The runner checks 22 unique sample identities, 100 unchanged requirements,
84 distinct `(sample_key,candidate_index)` records, 0 old admitted claims,
84 old rejections, and the old 18 failure / 4 unresolved-RC distribution.
The frozen V2 validator is rerun locally only to reproduce its historical
decisions. It is not changed. Raw completions are taken from embedded
`raw_response`; no model reconstructs missing data.

Question and requirement canonical hashes must match the original fields.
Every candidate retains its raw text/type/dependencies, raw object, canonical
hashes, source response reference and normalization history. All raw response
strings are copied unchanged. File hashes use SHA-256 over bytes; object/text
hashes use the repository's SHA-256 over canonical JSON serialization.

If an original raw cache JSON is accessible, it is opened read-only. The
content-addressed filename, cache key/identity, question/requirement hashes
and exact raw text must match. A mismatch stops. If the server cache is not
locally available, the receipt says `NOT_AVAILABLE`; embedded raw responses
are sufficient for offline revalidation, but the tool does not claim an
independent cache check. `--cache-root` explicitly supplies the old namespace
root and requires every referenced raw cache record to exist and match.
No `ArtifactStore`, cache lock/index, backend or image reader is opened.

The source JSONL and optional historical metrics/HTML are hashed before and
after processing. Optional baseline artifacts default to the named files next
to the input, if present. A missing explicitly supplied path fails closed.
The new output directory must not exist and cannot be inside a source cache.
No original review, pilot, R0/R1, V2 output or cache is rewritten.

## Recovery policy: representation, not hypothesis rewriting

The machine schema is `schemas/claim_revalidation_stage2_v2_1.schema.json`.
Shape validation and scientific/semantic contract checks are separate layers.

| Topic | Allowed recovery | Still rejected or held for review |
| --- | --- | --- |
| Unknown time | A traceable time object with **all three numeric fields null**, with no explicit question range, becomes `query_time:null`, `kind:SEARCH_INPUT_CLIP`; the scope description remains unchanged. | Model-created `0`, `1.0`, `0.5` etc. are never erased or moved to a sampling policy. Removing them is not proven lossless. |
| Explicit time | Independently parse a unique seconds range/stride from the **original question**. Keep the exact model numeric values. A wrong citation may point to the matching question sentence only when every value agrees. | Missing, wrong, ambiguous, reversed or unsupported time parameters remain failures. Requirement-only times do not authorize a question constraint. No filename/FPS conversion. |
| Static observations | Preserve `min_observations:1` for static OBJECT/ATTRIBUTE/RELATION with a single-frame modality. It is a valid one-observation branch, not a dynamic temporal requirement. | Zero observations fail. Dynamic ACTION and MULTI_FRAME require at least two. A compositional interpretation with one observation needs review. No frame-count inflation or modality downgrade. |
| Entity provenance | An exact quote in the explicitly named frozen requirement can correct QUESTION to REQUIREMENT. A missing pointer may use a unique exact requirement match. QUESTION/UNRESOLVED may retain a valid contextual `requirement_id`. | No automatic medical synonyms, fuzzy matching, plural-to-singular changes, added attributes or external knowledge. Label/quote equivalence not proven => REVIEW_REQUIRED. |
| Entity spelling/references | Separator-only label changes; `$` serialization of an already UNRESOLVED symbol; unique exact declared-label-to-ID references. | No identity resolution, entity insertion or target guessing. Ambiguous aliases stay unresolved. |
| Modality | The static `SPATIAL_MULTI_REGION` alias becomes `MULTI_REGION_SINGLE_FRAME` with the same target set and observation meaning. | No ACTION→SINGLE_FRAME conversion; type/predicate/evidence conflicts remain errors or review. |

`DOMAIN_KNOWLEDGE` is representable, but this offline recovery registers no
external knowledge source. Such an entity cannot be admitted without a real
declared source; no domain information is fetched or invented.

Every applied change records `field`, `before`, `after`, `rule_id`,
`semantic_preserving:true`, and its proof. Unproved changes are not applied.
Natural-language text, claim ID/type, predicates' truth polarity, dependencies,
verification purpose and interpretation-rule text are never rewritten.
Evidence requirements change only for declared equivalent enum/reference
serialization; observation counts and descriptive requirements remain intact.

## Layered validation and admission

Each original candidate has a record even when it fails. Independently
evaluable time, provenance and semantic checks continue after shape failures,
so a schema failure no longer hides downstream problems. The record separates:

- `structural_admission`: shape PASS/FAIL;
- `contract_validity`: PASS/FAIL/NOT_EVALUABLE/REVIEW_REQUIRED/BLOCKED;
- `semantic_review_status`: always REVIEW_REQUIRED for future scientific review;
- `state`: ACCEPTED, REJECTED, REVIEW_REQUIRED, BLOCKED_BY_DEPENDENCY.

General unproven visual truth/relevance warnings do not mean a claim is true.
Detected major ambiguity (unproved entity alias, compound proposition,
unestablished alternatives, continuity/absence burden, or interpretation
premise) blocks export. Explicit contract violations are rejected. Typed
dependencies retain their original direction and rationale. Missing references,
duplicate identities and cycles fail; every dependent of a non-admitted node
is blocked unless it already has its own rejection/review reason. `blocked_by`
is reported in **all** these cases. Edges are never deleted to improve counts,
and REQUIRES is never treated as logical implication.

Only ACCEPTED nodes with admitted dependencies are flattened into
`stage2_v2_1_claims.jsonl`, still UNVERIFIED / NOT_SEARCHED with null bindings.
The audit JSONL contains all other nodes. English atomicity and clinical
meaning are not fully decidable by these deterministic checks; detected
uncertainty remains explicit. No LLM self-scoring is used.

Spatial eligibility is a diagnostic subset of admitted nodes with an actual
SINGLE_FRAME or MULTI_REGION_SINGLE_FRAME contract: OBJECT, ATTRIBUTE, or
RELATION with STATIC_RELATION/CONTACT and no multi-frame requirement. Merely
calling a claim OBJECT does not qualify it. Eligibility is not evidence
grounding, experimental execution, truth verification or certificate admission.

## Outputs

- `stage2_v2_1_revalidated.jsonl`: all 22 samples / 84 candidate audit records;
- `stage2_v2_1_claims.jsonl`: consumable unverified contracts only;
- `stage2_v2_1_metrics.json`: actual counts, types, states, coverage and eligibility;
- `stage2_v2_vs_v2_1.html`: all questions, unchanged requirements, raw/normalized
  candidates, prior/current rejection reasons, contracts, dependencies and changes;
- `stage2_v2_1_rejection_audit.json`, `stage2_v2_1_normalization_audit.json`;
- `stage2_v2_1_summary.md`, `policy.json`, `offline_revalidation_audit.json`;
- `manifest.json`: source/policy/implementation and artifact hashes.

`not_exported_claim_count` includes strict rejection, blocking and pending
review; `candidate_state_counts` reports the four disjoint states. Reason
counts overlap and are not added as candidate counts. Requirement coverage
is a reference-ID count, not proof of semantic answer coverage. The historical
22 fresh calls are sourced from the task's frozen V2 summary; they are not
inferred as fresh calls from cache references. V2.1 performs zero calls.

## Local and GPU-server commands (no GPU used)

From `relive/`, the supplied local files can be processed directly:

```bash
export PYTHONPATH=src
OUT=../outputs/claim_revalidation_stage2_v2_1_$(date +%Y%m%d_%H%M%S)

python -m relive.claim_revalidation revalidate \
  --input ../outputs/stage2_v2_smoke22.jsonl \
  --expected-input-sha256 93d36bb9063660d48a4689f5631e0e07b0d27375aae521afd492a18f01d0d9f0 \
  --v2-metrics ../outputs/stage2_v2_metrics.json \
  --v2-html ../outputs/stage2_v1_vs_v2_smoke22.html \
  --output-dir "$OUT" --offline --no-model-calls

python -m relive.claim_revalidation revalidate \
  --input ../outputs/stage2_v2_smoke22.jsonl \
  --v2-metrics ../outputs/stage2_v2_metrics.json \
  --v2-html ../outputs/stage2_v1_vs_v2_smoke22.html \
  --replay-from "$OUT" --output-dir "${OUT}_replay" \
  --offline --no-model-calls
```

For the server, restore the previous `V2_ROOT` from that run's
`server_variables.sh`. Fetch the feature branch and use a new worktree so the
old V2 checkout is preserved:

```bash
REPO=/home/huihui/codes/MedVid_understanding_relive_spatial
WORKTREE=/home/huihui/codes/MedVid_understanding_claim_revalidation_v21
git -C "$REPO" fetch origin feature/relive-v2-protocol-dev-cases
git -C "$REPO" worktree add --detach "$WORKTREE" origin/feature/relive-v2-protocol-dev-cases
cd "$WORKTREE/relive"
source /home/huihui/miniconda3/etc/profile.d/conda.sh
conda activate MedVidU-cu124
export PYTHONPATH=src

# V2_ROOT must be the saved directory of the original real V2 smoke.
test -n "$V2_ROOT" && test -f "$V2_ROOT/run/stage2_v2_smoke22.jsonl"
V21_ROOT="${V2_ROOT}_offline_v21_$(date +%Y%m%d_%H%M%S)"

python -m relive.claim_revalidation revalidate \
  --input "$V2_ROOT/run/stage2_v2_smoke22.jsonl" \
  --expected-input-sha256 93d36bb9063660d48a4689f5631e0e07b0d27375aae521afd492a18f01d0d9f0 \
  --v2-metrics "$V2_ROOT/comparison/stage2_v2_metrics.json" \
  --v2-html "$V2_ROOT/comparison/stage2_v1_vs_v2_smoke22.html" \
  --output-dir "$V21_ROOT" --offline --no-model-calls

python -m relive.claim_revalidation revalidate \
  --input "$V2_ROOT/run/stage2_v2_smoke22.jsonl" \
  --v2-metrics "$V2_ROOT/comparison/stage2_v2_metrics.json" \
  --v2-html "$V2_ROOT/comparison/stage2_v1_vs_v2_smoke22.html" \
  --replay-from "$V21_ROOT" --output-dir "${V21_ROOT}_replay" \
  --offline --no-model-calls

cat "$V21_ROOT/stage2_v2_1_metrics.json"
cat "${V21_ROOT}_replay/offline_revalidation_audit.json"
```

These commands do not require CUDA or model files. Exit 0 means offline
processing completed, not that every candidate was admitted. Exit 2 means
an input/hash/cache/contract execution error; stop rather than reconstructing
inputs with a model. Existing output directories are never reused. Offline
replay checks byte identity of all principal derived artifacts and uses no
inference cache or new model call. Preserve every failed candidate.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_claim_revalidation_v21.py' -v
PYTHONPATH=src python -m unittest discover -s tests
git diff --check
```

No new planner experiment or evidence-grounding run should start until the
human review of this independent recovery is complete.
