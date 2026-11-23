# Stage 2 V3.1: instance contracts and conservative predicate compilation

V3.1 is an independent opt-in revision, not revalidation or repair of V3 results.
Use the SAME frozen 22 samples and 100 Stage-1 requirements. No images, GT,
verification, adaptation, certificate or answer generation is performed.

## Changes grounded in the failed real V3 smoke

The saved 22 responses contained 6 schema definitions (one also had duplicate
keys), 11 instances incorrectly nested under schema properties, and 5 proper
result instances. These historical responses remain unchanged.

The V3.1 prompt provides field rules and one explicitly fictional result-instance
example. It does not embed a JSON Schema definition. Schema metadata and wrapper
outputs remain rejected, with separate root diagnostic codes. Duplicate JSON
keys, markdown fences and invalid fields remain rejected. There is no automatic
unwrapping, JSON repair, candidate rewriting or format retry.

The model schema has the same seven minimal node fields, types and roles as V3.
It adds no search execution contract. Program-generated statuses remain
UNVERIFIED / NOT_SEARCHED. Only the protocol/schema ID changes.

Numeric inspection distinguishes declared H/R identifiers from numeric execution
bindings. H1 in prose is not a timestamp. Explicit seconds, frames, strides,
coordinates or numeric time instantiations stay forbidden even when a number
was copied from the public question. Such copying is not labelled GT leakage or
an invented time; query instantiation belongs in the separate public metadata.
Other numeric content is review-required, not automatically considered a search
binding. Unknown textual claim references are rejected.

Unresolved RC templates may use generic anaphors such as "the object identified
in H1". Failure to use one preferred phrase is not proof of an identity guess.
All RC templates stay review-required and unresolved, and cannot be compiled
into a target-bound contract. Explicit named presence/attribute target binding
is rejected. The implementation does not claim general semantic identity detection.

Clear predicate/type contradictions, such as direct contact labelled OBJECT,
are rejected without relabelling. Tautologies, prospective interpretation,
efficiency/quality language and contact-to-action leaps receive explicit review
flags. Contact alone is not silently rewritten into cutting or successful action.
Question relevance and medical correctness still require human review.

## Shared validator/compiler phrase analysis

A small deterministic grammar supports selected present/plural/progressive
motion predicates, static contact relations, presence and state expressions.
It extracts exact source phrases and character spans; it does not infer missing
entities or substitute medical synonyms. Attribute values and time parameters
stay textual/unbound. Negation, unresolved pronouns, complex clauses, unknown
predicates and composite arguments abstain from compilation.

This prevents accepting an arbitrary free-text temporal interpretation and then
pretending the compiler understood it. A syntactically valid graph may retain
an unknown predicate as REVIEW_REQUIRED while its contract remains UNCOMPILABLE.
The grammar is intentionally incomplete, not an NLP or medical correctness proof.

HIGH_LEVEL admission is unchanged: lower-level prerequisites plus an explicit
reviewed inference requirement, no presence-only inference, no truth propagation.
Formal publication still needs a frozen nonsynthetic executed graph, original
Stage-1 binding, valid references, all-node semantic approval, all compilable
nodes and no missing compiled prerequisites. Dry-run's formal JSONL stays empty.

## Version and reuse

- Protocol/cache namespace: relive-claimgen-stage2-v3.1.
- CLI: python -m relive.hypothesis_graph_v31.
- Independent prompt, schema, configs, validator, compiler and immutable outputs.
- Frozen V3 references can be reused read-only after their manifest checks.
- Existing LocalHF generation-metadata adapter, content-addressed cache,
  no-inference ReplayBackend, source-binding utilities and schema helpers are reused.
- Small orchestration/graph-validation snapshots remain versioned locally to
  avoid changing frozen V3 behavior. No monkeypatching of V3 module globals.
- Prompt/schema/config/code/backend fingerprints bind cache identity. The code
  hash includes imported frozen helpers that affect request/validation behavior.
- Resume verifies checkpoints; replay makes zero new calls or fails on cache miss.
- A completed --resume returns the stored original summary and its original usage;
  use replay, not that stored fresh summary, to demonstrate zero new calls.
- Model, chat template, generation limit (3072), greedy settings and retry budget
  remain unchanged from V3. No token-level constraint is claimed by this revision.

New runtime files:

- stage2_v3_1_smoke22.jsonl
- stage2_v3_1_metrics.json (includes detailed root/candidate error counts)
- stage2_v3_1_replay_audit.json
- stage2_v3_1_gt_isolation_audit.json
- protocol.json, run_binding.json, samples/, manifest.json

Comparison writes stage2_v3_1_comparison.html and
stage2_v3_1_quality_review.md. Compiler and review-template modes keep their
existing artifact responsibilities in NEW output directories.

## Local tests and zero-call historical diagnostic

From the relive directory in this checkout:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_hypothesis_graph_v31.py -v
PYTHONPATH=src python -m unittest discover -s tests
git diff --check

PYTHONPATH=src python -m relive.hypothesis_graph_v31 \
  --mode diagnose-v3 \
  --diagnosis-json /path/to/stage2_v3_smoke22_diagnosis.json \
  --output-dir /new/path/v3_failure_shape_diagnostic
```

diagnose-v3 only classifies root shapes and preserves raw responses and historical
statuses. It neither unwraps the 44 nested texts nor exports recovered claims.
Seven actual GT-free failed/accepted raw responses are stored as TEST_ONLY
regression fixtures; runtime prompts never read test fixtures or old claims.

Legacy full-suite protocol-dev tests additionally require the existing ignored
outputs/9个 case.json fixture at the REPOSITORY root. In an isolated worktree,
make a local read-only-use copy; never edit the original or commit it for this task.

## Server smoke: reuse references, new output/cache

Run from a checkout containing V3.1 and the existing MedVidU-cu124 environment.
First restore the original V3 server_variables.sh so V3_ROOT is known.
Do not rerun Stage 1 or overwrite the V3 run/replay/cache.

```bash
source /home/huihui/miniconda3/etc/profile.d/conda.sh
conda activate MedVidU-cu124
export PYTHONPATH="$PWD/src"

: "${V3_ROOT:?Restore the original V3 server_variables.sh first}"
REFS="$V3_ROOT/references"
CONFIG="$PWD/configs/qwen35_9b_hypothesis_graph_stage2_v3_1.yaml"
V31_ROOT="/mnt/hdd3/huihui/MedVid_understanding/relive_output/real/medvidu_hypothesis_graph_v31_$(date +%Y%m%d_%H%M%S)"
mkdir "$V31_ROOT"

python - "$V31_ROOT/server_variables.sh" "$REFS" "$CONFIG" "$V31_ROOT" "$PYTHONPATH" <<'PY'
import shlex, sys
from pathlib import Path
names = ['REFS', 'CONFIG', 'V31_ROOT', 'PYTHONPATH']
with Path(sys.argv[1]).open('x') as stream:
    for name, value in zip(names, sys.argv[2:]):
        stream.write(f'export {name}={shlex.quote(value)}\n')
PY

# Fresh V3.1: fixed 22 requests, no repair/retry. Choose an available GPU first.
CUDA_VISIBLE_DEVICES=0 python -m relive.hypothesis_graph_v31 \
  --mode run --references-dir "$REFS" --config "$CONFIG" \
  --cache-dir "$V31_ROOT/cache" --output-dir "$V31_ROOT/run"

# An interruption resumes with the SAME command plus --resume.
python -m relive.hypothesis_graph_v31 \
  --mode replay --references-dir "$REFS" --config "$CONFIG" \
  --cache-dir "$V31_ROOT/cache" --run-dir "$V31_ROOT/run" \
  --output-dir "$V31_ROOT/replay"

python -m relive.hypothesis_graph_v31 \
  --mode validate --references-dir "$REFS" --run-dir "$V31_ROOT/run"

python -m relive.hypothesis_graph_v31 \
  --mode compare --references-dir "$REFS" --run-dir "$V31_ROOT/run" \
  --output-dir "$V31_ROOT/comparison"

python -m relive.hypothesis_graph_v31 \
  --mode review-template --references-dir "$REFS" --run-dir "$V31_ROOT/run" \
  --output-dir "$V31_ROOT/semantic_review"

python -m relive.hypothesis_graph_v31 \
  --mode compile --dry-run --references-dir "$REFS" --run-dir "$V31_ROOT/run" \
  --output-dir "$V31_ROOT/compiler_dry_run"

cat "$V31_ROOT/run/stage2_v3_1_metrics.json"
cat "$V31_ROOT/replay/stage2_v3_1_replay_audit.json"
cat "$V31_ROOT/compiler_dry_run/contract_compiler_audit.json"
```

Optionally add --diagnosis-json to compare to display the frozen V3 raw responses;
all 22 questions and requirements must match exactly before comparison.
Expected replay: 22 cache hits, zero new calls, exact sample JSONL byte equality.
COMPLETE means completion, not scientific/semantic acceptance. Review failures,
root_error_counts, claim risks, and the HTML before any further experiment.
Do not run publish or Stage 3 from these instructions.
