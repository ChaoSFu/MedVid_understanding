# ReliVE-v1 — Reliable Visual Evidence Verification

ReliVE-v1 is a small, training-free research framework for test-time visual evidence verification on a frozen VLM. It changes candidate evidence, claims, support regions, and acquired temporal context; it never trains or updates model parameters. `VERIFIED` means that an Evidence–Claim pair passed its configured protocol. It is not a guarantee of medical truth. KEEP/DROP checks are spatial intervention checks, not causal proof.

The framework separates acquisition from admission: chronological or uniform acquisition assigns an `acquisition_rank`; only verification can admit evidence. It does not use temporal stability, shuffle, or FREEZE sensitivity as admission rules. Synthetic fixtures validate engineering behavior, not the scientific effectiveness of the method.

## Install

Use Python 3.10 or later. From this directory:

```bash
python -m pip install -e .
```

For a source checkout with the bundled dependencies already available, commands can instead use `PYTHONPATH=src python -m relive`.

## Supported tasks and runtime isolation

`claim_verification` is the current ReliVE-v1 core task for any atomic claim
that can be visually verified. It requires a runtime `target_claim`.
`action_qa` remains a generic, experimental internal protocol: it requires
nonempty, pre-inference `required_claims`, which stay frozen for the run. It is
not a MedVidU native task, has no MedVidU adapter, and is not used for the
current MedVidU experiments. `required_for_question` is not an input field, so
a user-supplied `false` cannot be silently rewritten to `true`.

| Task/protocol | Core runner | MedVidU adapter | Current role |
| --- | --- | --- | --- |
| `claim_verification` | supported | `user_claim_verification_v1` | GT-isolated custom MedVidU smoke; not an official MedVidU QA result |
| `action_qa` | supported, experimental | none | internal protocol testing with frozen `required_claims` |
| native MedVidU `qa_type` values | unsupported | none | future task-specific input/output schemas and evaluators |

`SUPPORTED_TASKS` names generic core-runner capability only; it does not assert
dataset-adapter compatibility. MedVidU retains its native `qa_type` in
`source_qa_type`. TAL, STG, next-action, region captioning, dense captioning,
CVS, skill assessment, and video summary are reported as
`UNSUPPORTED_NO_EXPLICIT_ADAPTER`; none is converted to `action_qa`. The STG
adapter only records the inspected schema boundary and does not load official
data or hidden labels.

Runtime input is JSONL with a mandatory hash-bound sidecar named `<runtime>.provenance.json`. Its fields are a closed whitelist, so answer, temporal-span, bounding-box, mask, and other annotation fields are rejected before inference. The sidecar records allowed provenance for question, claims, frames, and metadata. It makes the source path auditable; it does not prove that a curator never saw hidden labels.

Each sample has `sample_id`, `task`, `question`, and `frames`. Each frame has `frame_id`, `path`, and an explicit original `order`; `timestamp` is optional. When a timestamp is present it needs an approved source and source reference. ReliVE never invents FPS, seconds, or temporal ordering. The current minimal implementation accepts decoded frames; direct `video_path` decoding is rejected clearly rather than guessing a decoder or timestamps.

```json
{"sample_id":"case-1","task":"claim_verification","question":"...","frames":[{"frame_id":"f000","path":"frames/000.png","order":0}],"target_claim":{"claim_id":"c1","text":"..."},"metadata":{"question_scope":"single_action"}}
```

The matching sidecar has exactly this shape:

```json
{"schema_version":"relive-runtime-v1","source_kind":"public_runtime","runtime_sha256":"<sha256 of the JSONL bytes>","field_sources":{"question":"public_question","target_claim":"user_query","required_claims":"user_query","frames":"public_frames","metadata":"public_metadata"}}
```

## Mock smoke test

The bundled fixture is clearly synthetic and has no medical claim. Its action
wording is a deterministic protocol-test example, not a claim that ReliVE is
limited to action questions. It includes one declared-exclusive synthetic
contrast fixture so the complete protocol can be exercised deterministically.

```bash
cd relive
PYTHONPATH=src python -m relive run \
  --config configs/mock_smoke.yaml \
  --runtime examples/synthetic_runtime.jsonl \
  --output-dir runs/mock_smoke \
  --max-samples 2

PYTHONPATH=src python -m relive audit --run-dir runs/mock_smoke
PYTHONPATH=src python -m relive evaluate --run-dir runs/mock_smoke
```

The default `smoke` phase permits at most five samples. An explicit `--phase preflight` permits at most ten engineering samples. There is no full benchmark command.

## Configuration and real backends

`configs/mock_smoke.yaml` freezes every test setting. The real template at `configs/real_backend.example.yaml` requires an explicit base URL, endpoint, model identity and revision, credential environment-variable name, generation settings, image ordering, encoding, timeouts, retries, and backend-specific parameters. It never guesses a model path, endpoint, chat template, or sampling setting. Credentials remain in the environment and are neither saved nor logged.

For a configured service, copy the template outside version control, replace all placeholders, set the named credential environment variable, and use a `public_runtime` source. A real smoke run has not been executed by this package unless its run manifest says `synthetic: false` and records the actual backend fingerprint.

## Local Hugging Face checkpoints

`local_hf` is a frozen, in-process Hugging Face backend. It has no Qwen-specific
class list, hand-written chat template, architecture fallback, automatic entry
GPU selection, or automatic `trust_remote_code` setting. First inspect the
checkpoint on the GPU host. The default inspection reads only checkpoint
metadata, lists template candidates by source and SHA-256, checks whether the
declared architecture is available in the installed `transformers`, and reports
CUDA devices. It does not load a processor, model, or weight file.

```bash
cd /home/huihui/codes/MedVid_understanding/relive
PYTHONPATH=src python -m relive inspect-local-hf \
  --model-path /mnt/hdd3/huihui/models/Qwen3.5-9B \
  --output /mnt/hdd3/huihui/MedVid_understanding/relive_output/real/preflight/qwen35_metadata.json
```

After reviewing the static report, an explicit processor-only probe can confirm
the loaded processor class and template hash without loading model weights:

```bash
PYTHONPATH=src python -m relive inspect-local-hf \
  --model-path /mnt/hdd3/huihui/models/Qwen3.5-9B \
  --probe-processor \
  --output /mnt/hdd3/huihui/MedVid_understanding/relive_output/real/preflight/qwen35_processor.json
```

The processor probe uses local files and does not enable checkpoint-provided
code unless `--trust-remote-code` is explicitly passed. If it fails, do not
guess an architecture or template. Resolve the reported compatibility issue
before a model load.

Before choosing `chat_message_layout` or `processor_call_mode`, run the
processor image-contract probe. It creates two non-medical RGB images in memory
and tests all four explicit layout/call-mode candidates through the same PIL
message representation as `LocalHFBackend`; it never opens a model weight,
constructs a model, moves a tensor to CUDA, or calls `generate`. A candidate
passes only if `input_ids`, `pixel_values`, and `image_grid_thw` are present and
the distinct two-image grid and pixel payload preserve a deliberately swapped
image order. It reports facts for review and never picks a configuration.

```bash
PYTHONPATH=src python -m relive inspect-local-hf \
  --model-path /mnt/hdd3/huihui/models/Qwen3.5-9B \
  --probe-processor-images \
  --output /mnt/hdd3/huihui/MedVid_understanding/relive_output/real/preflight/qwen35_processor_image_contract.json
```

Copy one reviewed `PASS` candidate into `local_hf` only after this probe. The
initial configuration must reproduce its `source` frame encoding, null
pixel-limit overrides, reviewed template kwargs, and explicit image order.

For the reviewed `/mnt/hdd3/huihui/models/Qwen3.5-9B` checkpoint, use
[`configs/qwen35_9b_medvidu_claim_smoke_no_thinking.yaml`](configs/qwen35_9b_medvidu_claim_smoke_no_thinking.yaml).
It pins the observed checkpoint/template hashes and the reviewed
`images_then_text` plus `tokenized_chat_template` contract. A processor-only
probe on the same template showed that `enable_thinking: false` is accepted and
changes its rendering from an open `<think>` prefix to a closed empty block.
The original empty-kwargs baseline is retained as
[`configs/qwen35_9b_medvidu_claim_smoke.yaml`](configs/qwen35_9b_medvidu_claim_smoke.yaml)
for provenance only: its one-sample public-claim smoke emitted truncated
natural-language analysis instead of JSON. The no-thinking template argument is
part of the cache identity, so baseline cache entries cannot be reused even when
the same cache directory is retained. Its `revision` label is derived from the
checkpoint metadata hash because no upstream revision was reported by the local
directory.

The semantic verifier uses `relive-semantic-v4`. `SUPPORTED` and
`CONTRADICTED` must each cite one supplied frame through a zero-based image
index. The parser resolves that index to the immutable runtime frame ID before
certification; missing or out-of-range references are technical `PARSE_ERROR`,
never semantic support or contradiction. `INSUFFICIENT` may omit a reference.
The compact index avoids requiring a local model to reproduce long runtime IDs.
The spatial proposer uses `relive-spatial-v2`: it accepts explicitly declared
`normalized_0_1_xyxy` or `normalized_0_1000_xyxy` coordinates and records every
0–1000 to 0–1 conversion in proposal provenance. Semantic, claim, contrast, and spatial
parsers share closed JSON parsing: duplicate keys, `NaN`/`Infinity`, prose, and
undeclared fields fail technically. A complete ` ```json ` fence is normalized
before validation.

Copy [`configs/local_hf.example.yaml`](configs/local_hf.example.yaml) outside
version control and fill every inspected value exactly. `model_class` must be a
literal entry in `config.json`'s `architectures`; `processor_class`, template
source and template hash must match the loaded processor. The backend records
checkpoint metadata identity, classes, template contract, dtype, device map,
input device, generation parameters, and image encoding in the manifest and
cache key. It uses `eval()` and `torch.inference_mode()` and never updates
parameters. In-process `generate` cannot be safely killed by the configured
timeout, so that limitation is recorded in the fingerprint rather than hidden.

## Calibration-only intervention visual sweep

`scripts/calibration_intervention_sweep.py` is a no-model diagnostic for a
previously frozen, human-authored fixed-ROI calibration manifest. It creates
contact sheets for the existing Gaussian intervention at several radii and two
opaque diagnostic operators: ROI mean fill and neutral-gray fill. The tool
reads only the frozen manifest and its declared public frames; it does not load
model weights, open an inference cache, create a runtime, or alter the core
verifier, certificate rules, prompts, thresholds, or benchmark artifacts.

Use the contact sheets only to choose an operator/setting before a *new*
manifest is frozen. A setting must never be selected from Qwen verdicts.
The fixed-ROI calibration runner accepts opaque gray only when that new frozen
manifest binds its exact historical calibration version, RGB value `[127, 127,
127]`, and `human_visual_review_pre_inference` selection basis. The runner maps
that frozen declaration to the same registered core implementation used by the
formal protocol; it does not maintain a second image-transform implementation.

```bash
PYTHONPATH=src python scripts/calibration_intervention_sweep.py \
  --manifest /path/to/calibration_manifest.frozen.json \
  --output-dir /path/to/calibration/intervention_sweep \
  --blur-radii 4,8,16,32
```

## Versioned spatial intervention operators

`spatial.intervention` is a closed, cache-bound protocol declaration with an
operator name, implementation version, and parameters. The historical default
remains `gaussian_blur` / `relive-h4-pure-gaussian-hard-mask-v1`; configs that
only declare `blur_radius` are normalized to that explicit declaration for
backward compatibility. The registered `opaque_gray` /
`relive-opaque-gray-hard-mask-v1` operator accepts only RGB `[127,127,127]`.
Unknown operators, versions, and fill values fail configuration validation.

Opaque occlusion is a VLM sensitivity check. It is not a reconstruction of
anatomy, medical truth, or a causal proof. It leaves certificate admission,
semantic schema, prompts, thresholds, and KEEP/DROP/CONTROL policy conditions
unchanged. The complete operator declaration is recorded in the run manifest,
inference request/cache identity, spatial result, pixel audit, and certificate
provenance; Gaussian and opaque requests therefore cannot share a cache entry.

After a fixed-ROI calibration has been frozen and independently repeated, the
following *formal* vertical slice uses only its public atomic claim and public
frames. It never supplies the calibration's human ROI to the core runner. The
normal spatial proposer supplies the ROI, and the human ROI is compared only
after the run for diagnosis:

```bash
PYTHONPATH=src python scripts/formal_fixed_window_vertical_slice.py \
  --mode preflight \
  --manifest /path/to/calibration_manifest.opaque_gray.frozen.json \
  --config configs/qwen35_9b_medvidu_claim_smoke_no_thinking.yaml \
  --output-dir /path/to/new_formal_vertical_slice

PYTHONPATH=src python scripts/formal_fixed_window_vertical_slice.py \
  --mode run \
  --manifest /path/to/calibration_manifest.opaque_gray.frozen.json \
  --config configs/qwen35_9b_medvidu_claim_smoke_no_thinking.yaml \
  --output-dir /path/to/new_formal_vertical_slice

PYTHONPATH=src python scripts/formal_fixed_window_vertical_slice.py \
  --mode replay \
  --manifest /path/to/calibration_manifest.opaque_gray.frozen.json \
  --config configs/qwen35_9b_medvidu_claim_smoke_no_thinking.yaml \
  --output-dir /path/to/new_formal_vertical_slice
```

`preflight` makes no model call. `run` has a five-call upper bound (original
semantic result, automatic spatial proposal, KEEP, DROP, one matched control),
with later calls conditional on the formal path reaching them. `replay` uses the
same content-addressed cache and should report zero new model calls.


## Phase 3: failure-aware spatial evidence adaptation

Phase 3 is a GT-free, within-candidate R0→R1 engineering loop. It consumes an
immutable Phase 2 `fixed_candidate_pool.jsonl` and a Phase 2.5 diagnostic JSONL,
then admits only rows whose frozen `refinement_eligible` is `true` (the current
engineering cohort has nine rows). It does not call acquisition or alter a
candidate's frames, rank, window parameters, semantic verifier, opaque-gray
operator, or certificate rules.

`SpatialEvidenceAdaptationController` maps only documented failure states to
one versioned R1 prompt: residual support for `DEPENDENCE_UNRESOLVED`, claim
components for sufficiency failures, a tight claim-sufficient region for a
Phase 2.5-confirmed `B_LEGITIMATE_PROTOCOL_UNAVAILABLE` control geometry, and
a visibly grounded region for a confirmed proposal/ROI no-effect failure.
Unresolved control mismatches, original insufficiency, and technical failures
are excluded. The controller parses one integer `[0,1000]` rectangle and has no
certificate authority: every R1 rectangle goes through the existing
ORIGINAL/KEEP/DROP/matched-control pixel and semantic checks and the existing
`build_certificate()` function.

Use new output and cache directories; preflight makes no model call, `run`
uses the frozen local-HF configuration, and `replay` must report zero new
calls. This does not run Phase 4 or a larger experiment.

```bash
cd /home/huihui/codes/MedVid_understanding/relive
OUT=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/phase3_spatial_adaptation_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/phase3_spatial_adaptation.py \
  --mode preflight --config /path/to/frozen_phase2_config.json \
  --runtime /path/to/frozen_five_sample_public_runtime.jsonl \
  --phase2-run-dir /path/to/completed_phase2_run \
  --phase25-diagnostics /path/to/phase2_failure_diagnostics.jsonl \
  --output-dir "$OUT"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/phase3_spatial_adaptation.py \
  --mode run --config /path/to/frozen_phase2_config.json \
  --runtime /path/to/frozen_five_sample_public_runtime.jsonl \
  --phase2-run-dir /path/to/completed_phase2_run \
  --phase25-diagnostics /path/to/phase2_failure_diagnostics.jsonl \
  --output-dir "$OUT"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/phase3_spatial_adaptation.py \
  --mode replay --config /path/to/frozen_phase2_config.json \
  --runtime /path/to/frozen_five_sample_public_runtime.jsonl \
  --phase2-run-dir /path/to/completed_phase2_run \
  --phase25-diagnostics /path/to/phase2_failure_diagnostics.jsonl \
  --output-dir "$OUT"
```

The immutable outputs are `phase3_spatial_adaptation_trace.jsonl`,
`phase3_candidate_transition_summary.json`, `phase3_failure_reason_summary.md`,
and `phase3_audit.json`. Opaque occlusion remains a model-sensitivity
intervention, not medical truth or causal proof.

## Phase 2: fixed sliding-window candidate-pool traversal

Phase 2 adds only a GT-free outer temporal loop. It keeps the spatial proposer,
registered intervention operator, semantic verifier, and certificate builder
unchanged. Before inference, `relive.acquisition.acquire(...,
method="sliding_windows")` emits the entire chronological candidate pool and
binds it to an immutable `fixed_candidate_pool.jsonl` hash. The controller then
runs each candidate once, in rank order, and stops at the first formal
`VERIFIED` certificate. A failed candidate is recorded in
`candidate_traversal_trace.jsonl`; it is never re-proposed, re-sized, or
re-ranked in this phase.

The Phase 2 engineering smoke is deliberately frozen to its confirmed local
settings: window length 3 frames, stride 2 frames, at most 4 candidates, and
the existing retained final short-window behavior. These are smoke parameters,
not a benchmark or publication window protocol. The historical H2/H3/H4 window
code is not imported because its alignment analysis is outside the ReliVE
runtime isolation boundary.

Prepare a new config from the actual five-sample smoke's recorded config on the
GPU host. This preserves its Qwen checkpoint and generation settings verbatim;
only the explicit Phase 2 traversal and its already-calibrated opaque operator
are selected. The `max_spatial_proposals: 4` setting is a traversal resource
budget for one unchanged, one-shot proposal at each of the four frozen
candidates. It does not enable spatial refinement.

```bash
cd /home/huihui/codes/MedVid_understanding/relive

SOURCE_RUN=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/smoke_5_qwen35_gpu0_smoke5_v4_v2_20260910_113056
RUNTIME=/path/to/the_existing_five_sample_public_runtime.jsonl
OUT=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/phase2_fixed_sliding_window_$(date +%Y%m%d_%H%M%S)
PHASE2_CONFIG="${OUT}.config.json"

python - "$SOURCE_RUN/manifest/run.json" "$PHASE2_CONFIG" <<'PY'
import json, sys
source, destination = map(__import__('pathlib').Path, sys.argv[1:])
config = json.loads(source.read_text())['config']
expected = {'method': 'sliding_windows', 'window_size': 3, 'stride': 2, 'max_candidates': 4}
if {key: config['acquisition'][key] for key in expected} != expected:
    raise SystemExit('source smoke config does not match the frozen Phase 2 window protocol')
config['traversal'] = {'mode': 'fixed_sliding_window_pool'}
config['adaptation']['enabled'] = False
config['spatial']['intervention'] = {
    'operator': 'opaque_gray',
    'operator_version': 'relive-opaque-gray-hard-mask-v1',
    'parameters': {'fill_rgb': [127, 127, 127]},
}
config['budget']['max_candidates'] = max(config['budget']['max_candidates'], 4)
config['budget']['max_spatial_proposals'] = 4
destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
PY

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/phase2_fixed_sliding_window_smoke.py \
  --mode preflight --config "$PHASE2_CONFIG" --runtime "$RUNTIME" --output-dir "$OUT"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/phase2_fixed_sliding_window_smoke.py \
  --mode run --config "$PHASE2_CONFIG" --runtime "$RUNTIME" --output-dir "$OUT" \
  --baseline-run "$SOURCE_RUN"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/phase2_fixed_sliding_window_smoke.py \
  --mode replay --config "$PHASE2_CONFIG" --runtime "$RUNTIME" --output-dir "$OUT"
```

The run writes `analysis/failure_reason_distribution.json`,
`analysis/failure_reason_summary.md`, `analysis/phase2_traversal_audit.json`,
and the frozen-pool and trace artifacts. A replay must report
`new_model_calls: 0`; its cache-hit count covers the candidate-level requests
actually replayed. These diagnostics identify temporal acquisition versus
proposal/intervention/verifier bottlenecks. They do not support a benchmark
performance claim and do not change certificate admission.

## MedVidU public-runtime preparation

`prepare-medvidu` does not use the original assistant turn, `struc_info`,
`RC_info`, source `metadata`, answers, boxes, masks, or temporal labels. It
reads only the source ID, exactly one human question, ordered public frame
paths, the paired `sampled_video_frames` vector for length/type validation, the
native `qa_type`, and `dataset_name`. It preserves the input frame-list order,
including duplicate frames, and never converts frame references into FPS or
timestamps.

First run the report-only path audit. It writes a human-question schema report,
a compact public selector index, a human-question selector with only the human
question and verified public first/last frame paths, and a separate GT-isolation
audit. It writes no runtime and makes no model call.

```bash
PYTHONPATH=src python -m relive prepare-medvidu \
  --source-json /home/huihui/codes/MedVid_understanding/data_json/init_datas/medvidu_eccv2026_trainval.json \
  --frame-root /mnt/hdd3/huihui/hh_datas/MedVidU/valdata \
  --source-prefix /root/data \
  --output-dir /mnt/hdd3/huihui/MedVid_understanding/relive_output/real/preparation \
  --adapter report_only --path-audit-scope all --max-samples 1
```

The native MedVidU question types are retained as unsupported in that report:
TAL needs time spans; STG and region-caption tasks need boxes; next-action is a
future prediction; CVS and skill assessment need score vectors; dense captions
need multiple time-bounded events; summaries need multiple claims. None is
silently mapped to `action_qa`. Each future native task needs its own
input/output schema and evaluator; generic core `action_qa` is not a shortcut.

Use `medvidu_public_question_selector.jsonl`, not the source JSON, when
choosing a record for a user-authored visible claim. Its rows contain only
`source_record_index`, public-record identity hashes, native `qa_type`, the
human question, frame count, dataset name when public, and verified first/last
frame paths. It never contains assistant values, annotations, answers, boxes,
masks, or timestamp labels. A claim-verification smoke remains a custom public
claim protocol, never an official MedVidU QA result.

The only runtime-producing MedVidU adapter is
`user_claim_verification_v1`. It emits `task: claim_verification`, requires a
separate user-authored public claim JSONL, and is a custom GT-isolated smoke,
not an official MedVidU QA result. Each line binds an inspected public record
to a user claim:

```json
{"source_record_index":17,"public_record_sha256":"<from medvidu_public_record_index.jsonl>","target_claim":{"claim_id":"claim-17","text":"<user-authored visible claim>"}}
```

No `answer`, `bbox`, `mask`, `time_scope`, unknown field, or hidden-label value
is accepted in this manifest. This adapter is a custom public-claim protocol
smoke, not an official MedVidU QA result. Once a claim manifest exists, prepare
one runtime sample, then run one real smoke:

```bash
PYTHONPATH=src python -m relive prepare-medvidu \
  --source-json /home/huihui/codes/MedVid_understanding/data_json/init_datas/medvidu_eccv2026_trainval.json \
  --frame-root /mnt/hdd3/huihui/hh_datas/MedVidU/valdata \
  --output-dir /mnt/hdd3/huihui/MedVid_understanding/relive_output/real/runtime \
  --adapter user_claim_verification_v1 --max-samples 1 \
  --public-claim-manifest /absolute/path/public_claims.jsonl

PYTHONPATH=src python -m relive run \
  --config /absolute/path/local_hf.yaml \
  --runtime /mnt/hdd3/huihui/MedVid_understanding/relive_output/real/runtime/medvidu_user_claim_verification.runtime.jsonl \
  --output-dir /mnt/hdd3/huihui/MedVid_understanding/relive_output/real/smoke_1 \
  --cache-dir /mnt/hdd3/huihui/MedVid_understanding/relive_output/real/shared_cache \
  --max-samples 1 --phase smoke
```

The runner requires the matching `.gt_isolation_audit.json` beside every real
runtime and binds its runtime SHA-256 into the run manifest. It also rejects a
real run under a `mock` directory or a synthetic run under a `real` directory.
Repeat the same command with a fresh `smoke_5` output directory and
`--max-samples 5` only after the one-sample run succeeds; reuse the same cache
directory to verify zero new calls on a replay of identical inputs.

| Policy | Required checks |
| --- | --- |
| `acquisition_only` | Never verifies; acquisition ablation only |
| `semantic_only` | Original semantic support |
| `semantic_keep_drop` | Semantic plus KEEP/DROP, no controls |
| `semantic_spatial` | Semantic plus KEEP/DROP and all planned matched controls |
| `semantic_contrast_spatial` | Semantic, declared contrast handling, and matched spatial controls |

`semantic_contrast_spatial` is available only for synthetic declared-exclusive
fixtures in this release. Model-generated alternatives remain `UNRESOLVED` and
are diagnostic only: they cannot become a contrast admission check. A real
backend selecting the full contrast policy fails during configuration unless a
task-declared, public-option, or ontology-declared exclusivity source with
provenance is implemented and bound. The reviewed real Qwen smoke configs use
`semantic_spatial`.

The spatial protocol preserves source resolution and emits a pixel audit for
`ORIGINAL`, `KEEP_TARGET`, `DROP_TARGET`, and `DROP_MATCHED_CONTROL`. `ORIGINAL`
must change neither region. KEEP must preserve the ROI and change at least one
pixel outside it across the candidate frames; DROP and matched control must
preserve the exterior and change at least one interior pixel. No pixel-change
ratio threshold is used. If every relevant intervention is a no-op, the runner
returns `INTERVENTION_NO_EFFECT`, makes no variant VLM call, and the certificate
is `UNCERTAIN` rather than crashing the sample. These are intervention responses,
not causal proof.

The fixed configured alternate rectangle is synthetic-fixture-only. It has no
claim-conditioned grounding semantics and cannot admit real evidence. When a
KEEP failure would require a second support region on a real run, ReliVE records
`RE_GROUNDING_NOT_IMPLEMENTED` and remains `UNCERTAIN`; constrained re-proposal
is a later-stage feature.

## Artifacts, cache, and evaluation

Every inference attempt is immutable and content-addressed. Cache identity includes backend fingerprint, prompt contents, ordered frame paths and bytes, preprocessing, claim, region, intervention, and control parameters. Cache hits replay the same logical budget cost while making zero new model calls. A repeated run in the same output directory resumes completed samples; a cache entry with an incomplete raw-attempt chain is not accepted as success.

A new output directory can reuse a cache directory when only certificate policy changes: raw inference remains immutable and certificates are recalculated.

```bash
PYTHONPATH=src python -m relive run --config my_policy.yaml \
  --runtime runtime.jsonl --output-dir runs/policy_b --cache-dir runs/shared_cache
```

Only `relive evaluate` may open a ground-truth file:

```bash
PYTHONPATH=src python -m relive evaluate \
  --run-dir runs/policy_b --ground-truth held_out_gt.jsonl --output evaluation.json
```

Its task-answer exact match is descriptive and explicitly non-official. It also reports strict coverage, fallback rate, certificate/check status distributions, calls, latency, adaptation snapshots, and spatial/temporal alignment only when matching prediction and annotation types exist. Spatial IoU does not establish semantic truth, and time non-overlap does not establish action absence.

Pair `REJECTED` means only that the current Evidence–Claim pair was contradicted.
The runner tries the next candidate when available. Claim-level aggregation is
conservative: an unconflicted `VERIFIED` pair supports a claim; a contradiction
requires a claim with explicit `time_scope.frame_ids` and a rejected certificate
bound to exactly that same scope; a local rejection cannot contradict a
video-global/existential claim; verified and rejected evidence in the same scope
is a conflict. Strict answers cite only `VERIFIED` Evidence–Claim pairs and
abstain on incomplete coverage. `benchmark_forced` is a separate, explicitly
marked fallback output and is excluded from strict metrics.

## Phase 2.5 failure-mode audit

A completed Phase 2 run can be diagnosed without another model call. The audit
reads only completed runtime certificates, control-generation events, pixel
audits, and the existing raw-inference cache. It never opens evaluation or GT
artifacts and cannot change the candidate manifest, cache, operator, verifier,
or certificate policy.

```bash
PYTHONPATH=src python scripts/phase25_failure_mode_audit.py \
  --run-dir /path/to/phase2/run \
  --cache-dir /path/to/phase2/cache \
  --output-dir /path/to/new_phase25_diagnostics
```

It writes immutable `phase2_failure_diagnostics.jsonl`, a JSON summary, and a
Markdown report. A `CONTROL_RESULT_COUNT_MISMATCH` is classified as an
implementation/artifact discrepancy, a documented protocol-unavailable state,
or unresolved. It remains in technical-diagnostic status until resolved; the
audit never performs spatial refinement. `ORIGINAL_INSUFFICIENT` recommends
only temporal reacquisition, while spatial refinement is merely eligible (not
executed) for diagnosed dependence/sufficiency and eligible ROI-geometry
failures.

## Phase 3-v2: refiner contract repair

Phase 3-v2 preserves all Phase 3-v1 artifacts and uses a new cache identity.
It canonicalizes the historical R0 rectangle to integer 0–1000 xyxy before it
is shown to the refiner; input and output therefore share one coordinate
system. The refiner must return a closed object with either a changed proposed
rectangle or explicit `UNRESOLVED`. An unchanged proposal is recorded as
`REFINEMENT_NO_OP`; it never enters the formal intervention or certificate
path. `UNRESOLVED` is also not a certificate outcome.

The required first command is a five-candidate contract preflight selected
only from Phase 3-v1 proposal-contract failures. It calls the refiner but not
the formal verifier. Full v2 execution is rejected unless it has zero parse
failures, zero coordinate violations, and zero silent R0 copies. A legitimate
`UNRESOLVED` is allowed. The nine-candidate run then uses the same frozen
Phase 2/2.5 cohort and performs formal verification only for valid, changed R1
proposals.

```bash
OUT=/path/to/new_phase3_v2_output
PYTHONPATH=src python scripts/phase3_v2_spatial_adaptation.py \
  --mode contract-preflight --config /path/to/frozen_phase2_config.json \
  --runtime /path/to/public_runtime.jsonl --phase2-run-dir /path/to/phase2/run \
  --phase25-diagnostics /path/to/phase25_failure_diagnostics.jsonl \
  --phase3-v1-run-dir /path/to/phase3_v1/run --output-dir "$OUT"

PYTHONPATH=src python scripts/phase3_v2_spatial_adaptation.py \
  --mode run --config /path/to/frozen_phase2_config.json \
  --runtime /path/to/public_runtime.jsonl --phase2-run-dir /path/to/phase2/run \
  --phase25-diagnostics /path/to/phase25_failure_diagnostics.jsonl \
  --phase3-v1-run-dir /path/to/phase3_v1/run --output-dir "$OUT"

PYTHONPATH=src python scripts/phase3_v2_spatial_adaptation.py \
  --mode replay --config /path/to/frozen_phase2_config.json \
  --runtime /path/to/public_runtime.jsonl --phase2-run-dir /path/to/phase2/run \
  --phase25-diagnostics /path/to/phase25_failure_diagnostics.jsonl \
  --phase3-v1-run-dir /path/to/phase3_v1/run --output-dir "$OUT"
```

The v2 transition summary separates proposal-contract outcomes from formal
certificate transitions. `RESIDUAL_SUPPORT_OR_SEMANTIC_INSENSITIVITY` is a
diagnostic label for valid R1 cases with ORIGINAL and DROP both SUPPORTED; it
does not alter any certificate reason or trigger another round.

## Phase 3.5: claim-scope applicability routing

The GT-free `relive-claim-scope-router-v1` runs before spatial refinement and
before certificate outcomes. It accepts only public claim text, `qa_type`, and
closed runtime metadata. It rejects GT, ROI, model-result, and certificate
fields. Its deterministic taxonomy is:

- `LOCAL_ATOMIC`: one localized object, state, contact, or relation; eligible
  for the existing single-ROI KEEP/DROP/matched-control protocol.
- `MULTI_SUPPORT_POSSIBLE`: an existential or count-insensitive claim with
  potentially independent witnesses; defer to a future support-set verifier.
- `GLOBAL_DISTRIBUTED`: scene, procedure, environment, or viewpoint claim;
  defer to a future global/temporal verifier.
- `UNRESOLVED_SCOPE`: abstain from the single-ROI protocol.

This is an applicability decision, never a `VERIFIED` decision. The command
retrospectively audits the frozen Phase 2 development candidates without
reading certificate values for routing, then freezes a fresh, non-overlapping
`LOCAL_ATOMIC` cohort in public-runtime input order. It does not run Phase
3-v3 or any model inference.

```bash
PYTHONPATH=src python scripts/phase35_claim_scope_audit.py \
  --development-runtime /path/to/phase2_development_runtime.jsonl \
  --phase2-run-dir /path/to/phase2/run \
  --phase25-diagnostics /path/to/phase25_failure_diagnostics.jsonl \
  --prospective-runtime /path/to/fresh_public_runtime.jsonl \
  --output-dir /path/to/new_phase35_output \
  --max-prospective-samples 5
```

Both runtime files require their hash-bound GT-isolation audit sidecars. The
prospective runtime must contain only sample IDs absent from the Phase 2
development candidate manifest; otherwise the command stops before producing
an artifact.

### Freezing a fresh public LOCAL_ATOMIC window

For Phase 3.5/3-v3 spatial-mechanism development, use the separate public
window preparer. Its claim input has exactly the three public selector fields:

```json
{"source_record_index":123,"public_record_sha256":"<selector sha256>","target_claim":{"claim_id":"phase35-local-001","text":"The jaws of the forceps are contacting the tissue."}}
```

Its accompanying human-confirmation JSONL binds each `claim_id` to one to three
consecutive **public frame orders** and a short observation note. It has no
bbox, mask, ROI, pixel coordinate, GT, or answer field:

```json
{"claim_id":"phase35-local-001","frozen_frame_orders":[40,41,42],"human_public_visual_confirmation":true,"human_public_visual_confirmation_note":"Forceps jaws visibly contact tissue in the frozen public frames."}
```

```bash
PYTHONPATH=src python scripts/phase35_prepare_fresh_runtime.py \
  --source-json /path/to/medvidu_source.json \
  --frame-root /path/to/public_frames \
  --fresh-claims /path/to/fresh_claims.jsonl \
  --window-confirmations /path/to/frozen_public_windows.jsonl \
  --config /path/to/frozen_phase2_config.json \
  --output-dir /path/to/new_fresh_phase35_runtime
```

The preparer rejects the five Phase 2 development source indices, duplicate
record hashes/claims, non-consecutive or non-public frame orders, non-local
claim scope, and any ROI/GT language in the confirmation artifact. It writes a
closed runtime containing only the frozen public frames, while the human
confirmation remains in the external immutable source manifest. It makes zero
model calls and must finish before any spatial proposal, refinement, or
certificate run.

### Selecting fresh public records for human review

Before authoring a fresh claim, export a deterministic batch of records from a
public selector. The exporter excludes the five Phase 2 development records,
deduplicates public-record hashes and sample IDs, maps only public frames, and
renders **every** public frame in continuous original order across paginated
contact sheets. It writes no claim, visual confirmation, ROI, or model output.

```bash
PYTHONPATH=src python scripts/phase35_export_public_record_candidates.py \
  --public-selector /path/to/medvidu_public_question_selector.jsonl \
  --source-json /path/to/medvidu_source.json \
  --frame-root /path/to/public_frames \
  --output-dir /path/to/new_phase35_public_record_candidates \
  --batch-size 10
```

`fresh_source_record_candidates.jsonl` contains each selected
`source_record_index`, `public_record_sha256`, every original public frame
order, and its contact-sheet page paths. It is a selection aid only: human
review must create the specific `LOCAL_ATOMIC` claim and frozen 1–3 frame
window later, before any model call.
