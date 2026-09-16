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

### Phase 3-v3: fresh `LOCAL_ATOMIC` automatic spatial-certificate slice

`phase35_v3_fresh_spatial_certificate.py` accepts only the immutable Phase 3.5
prospective `LOCAL_ATOMIC` manifest and its frozen public-window runtime. It
creates exactly one candidate containing all and only the frozen one-to-three
public frames for each sample. The ROI is generated by the existing automatic
spatial proposer, then passed unchanged through the registered intervention,
KEEP/DROP/matched-control checks, and existing certificate builder.

The command rejects non-local claim scopes, GT/outcome-shaped manifest fields,
changed runtime/config/manifest bindings, non-consecutive frames, adaptation,
and any operator except the registered `opaque_gray` v1 protocol. Human
confirmation prose and any human ROI are excluded from runtime and model
inputs. Phase 3-v3 does not call the Phase 3 reason-specific refiner; a fresh
cohort has no historical failure outcome to route on before its first formal
certificate attempt.

### Phase 3.6: failure-conditioned automatic ROI re-grounding

`phase36_roi_regrounding.py` is restricted to the two Phase 3-v3 fresh claims
whose ORIGINAL result was `SUPPORTED` and whose frozen artifact identifies an
automatic ROI geometry failure. `phase35-local-001` receives the versioned
`OVERBROAD_OR_MISLOCALIZED_INTERACTION_ROI` prompt; `phase35-local-002`
receives `INCOMPLETE_INTERACTION_EVIDENCE`. The prior R0 is supplied only as
canonical 0–1000 integer xyxy coordinates. Each candidate gets one R1 attempt;
`UNRESOLVED`, invalid, or unchanged R1 proposals never enter formal
intervention. The ORIGINAL-insufficient third claim is recorded as excluded.

A valid changed R1 uses the existing semantic verifier, opaque-gray operator,
matched-control generator, pixel audit, and certificate builder without any
policy, threshold, or manual-ROI change. A retained DROP `SUPPORTED` result is
reported only as `RESIDUAL_SUPPORT_OR_SEMANTIC_INSENSITIVITY`; it does not
trigger another refinement round.

## Phase 3.7 and Phase 4A diagnostic boundary

Phase 3.7 records a frozen residual-support diagnostic and stops before R2,
R3, or multi-ROI admission. Phase 4A-0 has two separate uses: historical audit
remains `HISTORICAL_EXPLORATORY`; development controls require independent human
eligibility review before they can enter `DEVELOPMENT_POSITIVE_CONTROL` mode.
The latter mode accepts only the frozen `ELIGIBLE_FOR_PHASE4A_POSITIVE_CONTROL`
records and never automatically loads historical Phase 3.5 cases.

Phase 4A measures A/B/C forced-choice likelihood under frozen ORIGINAL, KEEP,
DROP, matched-control, full-gray, and mismatched-public inputs. Its finalizer
is read-only and creates descriptive failure-routing fixtures only. Human
eligibility, likelihood diagnostics, certificate admission, and `VERIFIED` are
different stages: neither human eligibility nor a diagnostic pattern creates a
certificate or establishes claim truth.

A cache replay establishes engineering reproducibility. An independent fresh
forward run is needed to evaluate inference stability. Development controls are
for threshold/rule development only; future held-out confirmatory controls are
required for confirmation.

## ReliVE-v2 protocol-development status

ReliVE-v2 currently provides CPU-only, append-only `RequirementSpec` and
`ClaimSpec` contracts, a claim graph, and a failure-routed adaptation controller.
They are future-planning records only: they neither run retrieval or spatial
proposals nor create certificates or answers. Phase 4A routing fixtures remain
diagnostic-only legacy development artifacts. Their old posthoc status is read
as `POSTHOC_VALIDATED_FOR_DIAGNOSTIC_FINALIZATION_ONLY`, never as certificate
verification.

The frozen v2 controller can decide a permitted next action and budget effect,
but it does not execute that action. `STOP_DIAGNOSTIC` and `STOP_ABSTAIN` are
not evidence admission. Future temporal planning, re-grounding, evidence-bank
construction, and final reasoning require separately frozen protocols.

Implemented now:

- immutable `RequirementSpec` contract;
- append-only Claim Graph contract;
- closed failure taxonomy;
- bounded deterministic controller; and
- Phase 4A fixture routing validation.

Not implemented: Task Adapter, hypothesis or observation generator, video index,
temporal pyramid, claim-conditioned retrieval, typed spatial-proposal execution,
SAM 2 propagation, geometry-specific verifier, Evidence Bank, and Final
Reasoner. Controller decision != action success; diagnostic complete != `VERIFIED`; fixture replay != model experiment; and legacy posthoc Phase 4A != a
formal confirmatory control.

## ReliVE-v2 TAL requirement and VideoIndex stages

ReliVE-v2 now implements only this GT-isolated, non-inferential path:

```text
public question → deterministic TAL adapter → frozen RequirementSpec
frozen RequirementSpec → audited selected public media → immutable VideoIndex
```

The TAL adapter retains the complete human-authored question and its SHA-256.
It recognizes one registered terminal query clause and never derives an event
from a procedure description or possible-action list. Unsupported, unknown,
and ambiguous queries remain unresolved.

A `RequirementSpec` describes what the question demands. A VideoIndex binds
selected, decoded public frame bytes and can be created only when a public,
validated seconds timebase is supplied. It does not localize an event.

```text
VideoIndex frozen != event localized
timebase resolved != claim supported
frame audit != VERIFIED
```

Freeze an ordered public cohort, or replace `--max-samples` with a strict,
ordered `--identity-manifest` containing only public identity fields:

```bash
PYTHONPATH=src python scripts/freeze_v2_tal_requirement_selection.py \
  --public-question-selector /path/to/medvidu_public_question_selector.jsonl \
  --output /path/to/tal_requirement_selection.frozen.json \
  --max-samples 5

PYTHONPATH=src python scripts/prepare_v2_tal_requirements.py \
  --public-question-selector /path/to/medvidu_public_question_selector.jsonl \
  --selection-manifest /path/to/tal_requirement_selection.frozen.json \
  --event-ontology configs/v2/tal_event_ontology.yaml \
  --output-dir /path/to/v2_tal_requirements
```

Build a VideoIndex. Without `--public-timestamp-manifest`, the command still
writes the media projection but produces the legitimate
`UNRESOLVED_TIMEBASE` stop state and no VideoIndex rows:

```bash
PYTHONPATH=src python scripts/prepare_v2_tal_video_index.py \
  --requirement-freeze-dir /path/to/v2_tal_requirements \
  --selection-manifest /path/to/tal_requirement_selection.frozen.json \
  --source-json /path/to/medvidu_source.json \
  --frame-root /path/to/public_frames \
  --source-prefix /root/data \
  --timebase-policy configs/v2/tal_timebase_sources.yaml \
  --output-dir /path/to/v2_video_index

PYTHONPATH=src python scripts/validate_v2_tal_video_index.py \
  --output-dir /path/to/v2_video_index --materialize-frames
```

A timestamp JSONL is insufficient on its own. Stage 2 accepts a resolved
seconds timebase only when it is accompanied by
`public_per_frame_timestamps.provenance.json`, which binds its SHA-256, source
file SHA-256, registered source type, adapter version, and explicit time origin.
Use `scripts/audit_v2_tal_timestamp_source.py` (or the equivalent exporter) on
the frozen `v2_public_media_projection.jsonl`. It accepts only decoder PTS with
a frozen decoder-frame map, or an explicit documented source-frame-index/FPS
adapter; without either it writes `UNRESOLVED_TIMEBASE_SOURCE` and no timestamp
manifest. Frame order, filenames, IDs, sampled references, and default FPS are
never treated as seconds. Pass both files to Stage 2:

```bash
PYTHONPATH=src python scripts/prepare_v2_tal_video_index.py \
  # ... frozen Stage 2 inputs ... \
  --public-timestamp-manifest /path/to/public_per_frame_timestamps.jsonl \
  --public-timestamp-provenance /path/to/public_per_frame_timestamps.provenance.json \
  --output-dir /path/to/v2_video_index
```

All timestamp exporter, validator, and README commands use `python`; they never
load a model, cache, backend, GT, certificate builder, or retrieval stage. The freeze opens a mixed source container only to project a
whitelisted public identity, human question, ordered frame references, and
paired sampled-reference shape. It does not access assistant or GT values.

Hypothesis/observation generation, temporal pyramids and retrieval, spatial
grounding, verification, Evidence Bank construction, and Final Reasoner remain
unimplemented.

### NurViD dataset-native timebase

For the narrow, audited `NurViD/frames_2fps/<video_id>/000001.jpg` layout,
ReliVE can derive **clip-relative** seconds from public dataset metadata without
downloading an original YouTube video. This is a
`DATASET_INTERNAL_DERIVED` timebase, not decoder PTS verification. The v1
formula is exact decimal arithmetic:

```text
anchor_source_frame_reference = sampled_video_frames[0]
timestamp_seconds = (source_frame_reference - anchor_source_frame_reference) / 2
```

It accepts only `frames_2fps`, anchors zero seconds to the first presented
logical frame, verifies frame paths, `sampled_video_frames`, `metadata.video_id`,
monotonicity, and a preregistered span/tail-gap bound, and preserves duplicate
logical frames. This does not recover original-video absolute seconds or decoder
PTS; `000001.jpg` is reported only as a legacy diagnostic and is not a gate. It only reads the
allowlisted public metadata fields; assistant values and annotations are not
opened. First run the full-cohort audit and then export one frozen selection:

```bash
PYTHONPATH=src python scripts/export_v2_tal_dataset_native_timebase.py \
  --dataset-json /path/to/medvidu_eccv2026_trainval.json \
  --selection-manifest /path/to/tal_requirement_selection.frozen.json \
  --requirement-freeze-dir /path/to/v2_tal_requirements \
  --media-audit-dir /path/to/previous_v2_video_index \
  --frame-root /path/to/MedVidU/valdata \
  --source-prefix /root/data \
  --frame-bank-layout frames_2fps \
  --audit-output-dir /path/to/nurvid_timebase_audit \
  --output-dir /path/to/nurvid_timestamps
```

On success it writes `public_per_frame_timestamps.jsonl`, its provenance
sidecar, and `v2_tal_timestamp_source_audit.json`. Supply both timestamp files
to `prepare_v2_tal_video_index.py`. Cohort exceptions remain reported. Export is permitted only when the already
frozen selected identity itself passes every clip-local hard condition; an
unsupported layout, selected path/reference mismatch, non-monotonicity, or
span/tail-gap failure emits no timestamp manifest.

### Stage 3A: immutable temporal search plan

Stage 3A consumes only frozen RequirementSpec, selection, VideoIndex, and
timestamp artifacts. It emits no HypothesisClaim or ObservationClaim instance,
does not load a model, and does not score windows. It freezes the contracts for
future claims plus a deterministic 8/16/32-second, 50%-overlap clip-local
temporal pyramid. A right-aligned tail window closes each scale's coverage gap;
the full-clip window is always last. Duplicate logical frames remain in the
index and are represented by an alias map to unique visual frames.

```bash
PYTHONPATH=src python scripts/prepare_v2_tal_temporal_search_plan.py \
  --requirement-freeze-dir /path/to/v2_tal_requirements \
  --selection-manifest /path/to/tal_requirement_selection.frozen.json \
  --video-index-dir /path/to/v2_video_index \
  --public-timestamp-manifest /path/to/public_per_frame_timestamps.jsonl \
  --public-timestamp-provenance /path/to/public_per_frame_timestamps.provenance.json \
  --temporal-policy configs/v2/tal_temporal_pyramid_policy.yaml \
  --output-dir /path/to/v2_tal_temporal_search

PYTHONPATH=src python scripts/validate_v2_tal_temporal_search_plan.py \
  --output-dir /path/to/v2_tal_temporal_search
```

The sole successful terminal state is `FROZEN_PRE_MODEL` with
`READY_FOR_COARSE_HYPOTHESIS_GENERATION`. That readiness is a planning gate,
not a hypothesis, score, localized time interval, certificate, or VERIFIED
result.

### Stage 3B: coarse hypotheses from frozen windows

Stage 3B is the first model-using TAL stage. It may score only the immutable
Stage 3A windows, once each, with the reviewed local-HF forced-choice API. It
uses A/B/C/D next-token likelihoods, computes the frozen
`logP(A)-logsumexp(B,C,D)` margin, and uses that margin only for rank ordering.
The full-clip packet is diagnostic context and is never eligible for a positive
interval. Positive candidates use fixed temporal NMS (IoU >= 0.5), retain at
most five windows, and are accompanied by one `NO_VISIBLE_EVENT` alternative.
Every emitted hypothesis remains `CANDIDATE_UNVERIFIED`.

```bash
PYTHONPATH=src python scripts/prepare_v2_tal_coarse_hypotheses.py \
  --mode prepare --config /path/to/reviewed_qwen_opaque_gray.json \
  --requirement-freeze-dir /path/to/v2_tal_requirements \
  --selection-manifest /path/to/tal_requirement_selection.frozen.json \
  --stage3a-dir /path/to/v2_tal_temporal_search \
  --video-index-dir /path/to/v2_video_index \
  --public-timestamp-manifest /path/to/public_per_frame_timestamps.jsonl \
  --public-timestamp-provenance /path/to/public_per_frame_timestamps.provenance.json \
  --output-dir /path/to/v2_tal_stage3b

PYTHONPATH=src python scripts/prepare_v2_tal_coarse_hypotheses.py \
  --mode preflight --config /path/to/reviewed_qwen_opaque_gray.json \
  --output-dir /path/to/v2_tal_stage3b

PYTHONPATH=src python scripts/prepare_v2_tal_coarse_hypotheses.py \
  --mode run --config /path/to/reviewed_qwen_opaque_gray.json \
  --output-dir /path/to/v2_tal_stage3b

PYTHONPATH=src python scripts/prepare_v2_tal_coarse_hypotheses.py \
  --mode replay --config /path/to/reviewed_qwen_opaque_gray.json \
  --output-dir /path/to/v2_tal_stage3b

PYTHONPATH=src python scripts/prepare_v2_tal_coarse_hypotheses.py \
  --mode validate --output-dir /path/to/v2_tal_stage3b
```

No Stage 3B command creates an ObservationClaim, ROI, boundary refinement,
certificate, VERIFIED result, or final answer.

### Stage 3B: independent fresh-run reproducibility audit

Before Stage 3C, run a second GPU fresh inference in a different output root
and therefore a different empty `cache/` directory.  Do not copy, remove, or
replay the first run's cache.  The zero-model comparator reads only emitted
Stage 3B artifacts, first checks their scientific-input projection and fresh
run counters, then reports either exact numerical/discrete reproduction,
stable discrete results with numerical variation, or instability.  It never
creates claims, certificates, or VERIFIED results.

```bash
# RUN_A is the completed first Stage 3B root.  RUN_B and CMP must be new,
# empty output directories.  The prepare inputs are the same frozen Stage 1--3A
# inputs used for RUN_A.
PYTHONPATH=src python scripts/prepare_v2_tal_coarse_hypotheses.py \
  --mode prepare --config /path/to/reviewed_qwen_opaque_gray.json \
  --requirement-freeze-dir /path/to/v2_tal_requirements \
  --selection-manifest /path/to/tal_requirement_selection.frozen.json \
  --stage3a-dir /path/to/v2_tal_temporal_search \
  --video-index-dir /path/to/v2_video_index \
  --public-timestamp-manifest /path/to/public_per_frame_timestamps.jsonl \
  --public-timestamp-provenance /path/to/public_per_frame_timestamps.provenance.json \
  --output-dir /path/to/RUN_B

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/prepare_v2_tal_coarse_hypotheses.py \
  --mode preflight --config /path/to/reviewed_qwen_opaque_gray.json --output-dir /path/to/RUN_B
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/prepare_v2_tal_coarse_hypotheses.py \
  --mode run --config /path/to/reviewed_qwen_opaque_gray.json --output-dir /path/to/RUN_B
PYTHONPATH=src python scripts/prepare_v2_tal_coarse_hypotheses.py --mode validate --output-dir /path/to/RUN_B

PYTHONPATH=src python scripts/compare_v2_tal_stage3b_independent_runs.py \
  --run-a-dir /path/to/RUN_A --run-b-dir /path/to/RUN_B --output-dir /path/to/CMP
PYTHONPATH=src python scripts/summarize_v2_tal_stage3b_candidates.py --run-dir /path/to/RUN_A
```

The comparator writes `v2_stage3b_independent_repeat_comparison.json`,
`v2_stage3b_independent_repeat_window_differences.jsonl`, and
`v2_stage3b_independent_repeat_audit.json`.  Only a comparison with
`ready_for_stage3c=true` permits planning the next stage.

### Stage 3C: immutable ObservationClaim and fine acquisition planning

Stage 3C is a zero-model freeze.  Given an independently reproducible Stage
3B root, it preserves every parent hypothesis byte-for-byte, creates three
unverified required-observation claims per positive hypothesis, creates only a
global-negative-coverage obligation for `NO_VISIBLE_EVENT`, and freezes 4 s /
2 s fine acquisition windows.  It stops at
`READY_FOR_OBSERVATION_RETRIEVAL` and never reads images, scores a claim, or
creates a certificate.

```bash
PYTHONPATH=src python scripts/prepare_v2_tal_observation_planning.py \
  --mode prepare \
  --requirement-freeze-dir /path/to/v2_tal_requirements \
  --selection-manifest /path/to/tal_requirement_selection.frozen.json \
  --video-index-dir /path/to/v2_video_index \
  --stage3b-dir /path/to/completed_stage3b_run \
  --independent-repeat-comparison /path/to/v2_stage3b_independent_repeat_comparison.json \
  --policy configs/v2/tal_observation_temporal_planning_policy.json \
  --output-dir /path/to/v2_tal_stage3c

PYTHONPATH=src python scripts/prepare_v2_tal_observation_planning.py \
  --mode validate --output-dir /path/to/v2_tal_stage3c
```

### Stage 3D: claim-conditioned observation evidence retrieval

Stage 3D consumes the immutable Stage 3C observation claims, physical windows,
and binding fan-out. It creates one chronological, de-duplicated packet per
physical window (at most eight unique frames) and one cached A/B/C/D likelihood
task per unique `(template, packet, prompt, model, choice-policy)` identity.
Parent hypotheses are not in prompts or cache identity. A temporal observation
with too few distinct timestamps is reported as
`TEMPORAL_PACKET_INSUFFICIENT` without a model call. Results are fanned out to
all original bindings and ranked only within each ObservationClaim; at most
three `CANDIDATE_EVIDENCE_UNVERIFIED` candidates are retained. No threshold,
certificate, status admission, `NO_VISIBLE_EVENT` conclusion, or final TAL
answer is produced.

```bash
PYTHONPATH=src python scripts/prepare_v2_tal_observation_retrieval.py \
  --mode prepare --config /path/to/reviewed_qwen_opaque_gray.json \
  --requirement-freeze-dir /path/to/v2_tal_requirements \
  --selection-manifest /path/to/tal_requirement_selection.frozen.json \
  --video-index-dir /path/to/v2_video_index \
  --stage3c-dir /path/to/v2_tal_stage3c \
  --policy configs/v2/tal_observation_retrieval_policy.json \
  --output-dir /path/to/v2_tal_stage3d

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/prepare_v2_tal_observation_retrieval.py \
  --mode preflight --config /path/to/reviewed_qwen_opaque_gray.json \
  --stage3c-dir /path/to/v2_tal_stage3c --output-dir /path/to/v2_tal_stage3d
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/prepare_v2_tal_observation_retrieval.py \
  --mode run --config /path/to/reviewed_qwen_opaque_gray.json --output-dir /path/to/v2_tal_stage3d
PYTHONPATH=src python scripts/prepare_v2_tal_observation_retrieval.py \
  --mode replay --config /path/to/reviewed_qwen_opaque_gray.json --output-dir /path/to/v2_tal_stage3d
PYTHONPATH=src python scripts/prepare_v2_tal_observation_retrieval.py \
  --mode validate --output-dir /path/to/v2_tal_stage3d
```

The fresh run must have `new_model_calls=planned_model_calls` and
`cache_hits=0`; replay must have `new_model_calls=0` and
`cache_hits=planned_model_calls`. The four scientific result hashes in the
run/replay summaries must match before later temporal evidence composition.

### Stage 3F: typed spatial evidence planning freeze

Stage 3F is a metadata-only planning freeze over the immutable Stage 3C
ObservationClaims, completed Stage 3D retrieval, Stage 3E temporal chains, and
VideoIndex metadata. It writes only ungrounded composite spatial contracts and
anchor schedules; it does not open frame/video bytes, generate an ROI or mask,
call a model, or create a certificate. Each dynamic contract propagates across
its full physical window and declares a future
`COMPOSITE_EVIDENCE_UNION` intervention with a matched control requirement.

```bash
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_evidence_planning.py \
  --mode prepare \
  --stage3c-dir /path/to/v2_tal_stage3c \
  --stage3d-dir /path/to/v2_tal_stage3d \
  --stage3e-dir /path/to/v2_tal_stage3e \
  --video-index-dir /path/to/v2_video_index \
  --policy configs/v2/tal_spatial_evidence_planning_policy.json \
  --output-dir /path/to/v2_tal_stage3f

PYTHONPATH=src python scripts/prepare_v2_tal_spatial_evidence_planning.py \
  --mode validate --output-dir /path/to/v2_tal_stage3f
```

A successful freeze has `stage_status=SPATIAL_EVIDENCE_PLANS_FROZEN_UNGROUNDED`
and `ready_for_stage3g_spatial_grounding=true`; it remains non-evidentiary and
cannot produce `VERIFIED`.

### Stage 3G-A: native VLM role-labelled anchor grounding

Stage 3G-A consumes only a validated Stage 3F freeze and its bound Stage 3C,
3D, 3E, and VideoIndex artifacts. It runs one single-image Qwen grounding call
for every frozen `(grounding task, unique anchor)` pair. The model may return
only role visibility and a relative-1000 bounding box. Grounding candidates are
not ObservationClaim verification, support tubes, masks, evidence, certificates,
or `VERIFIED` results.

```bash
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding.py \
  --mode preflight --config /path/to/reviewed_qwen_config.json \
  --policy configs/v2/tal_spatial_anchor_grounding_policy.json \
  --stage3f-dir /path/to/v2_tal_stage3f --stage3c-dir /path/to/v2_tal_stage3c \
  --stage3d-dir /path/to/v2_tal_stage3d --stage3e-dir /path/to/v2_tal_stage3e \
  --video-index-dir /path/to/v2_video_index --output-dir /path/to/v2_tal_stage3g
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding.py \
  --mode run --config /path/to/reviewed_qwen_config.json \
  --policy configs/v2/tal_spatial_anchor_grounding_policy.json --output-dir /path/to/v2_tal_stage3g
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding.py \
  --mode replay --config /path/to/reviewed_qwen_config.json \
  --policy configs/v2/tal_spatial_anchor_grounding_policy.json --output-dir /path/to/v2_tal_stage3g
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding.py \
  --mode validate --output-dir /path/to/v2_tal_stage3g
```

The review packet is generated for every successful, ambiguous, not-visible,
or parser-failure candidate. It supports later human development audit only.

The Stage 3G-A v2 policy fixes a deterministic `max_new_tokens=512` budget for
its multi-role JSON response. This is bound into the effective model generation
configuration, preflight, cache identity, and result manifest. Earlier runs
with a lower generation budget remain immutable parser-failure artifacts and
must not be reused as cache inputs for v2.

### Stage 3G-A v3: output-contract hardening

Stage 3G-A v3 leaves every v2 run, cache entry, and frozen artifact untouched.  It
uses a separate prompt, parser version, cache namespace (`stage3g_v3`), and
artifact names.  Local-HF provides stopping metadata (`EOS_TOKEN`,
`MAX_NEW_TOKENS`, or `OTHER_STOP`); this is native generation telemetry, not
schema-constrained decoding.  The reviewed backend does not provide reliable
JSON-schema/grammar constrained decoding, so v3 freezes a strengthened prompt
and retains the fail-closed JSON-object parser.  Top-level arrays, bracket
repair, and LLM repair remain prohibited.

First create the read-only v2 diagnostic and bounded engineering smoke manifest.
`V2` is the root containing `stage3g_preflight.json` and `run/` from the
existing 75-call run.  The smoke picks, before a v3 call, one historical
parse/schema-failure anchor for each PRE/ACTION/POST role plus at most one
historical schema-failure case.  It is not a scientific cohort.

```bash
cd /home/huihui/codes/MedVid_understanding/relive
conda activate MedVidU-cu124

V2=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/your_existing_stage3g_v2
DIAG=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/v2_tal_stage3g_v2_diagnostic_$(date +%Y%m%d_%H%M%S)
SMOKE=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/v2_tal_stage3g_v3_smoke_$(date +%Y%m%d_%H%M%S)

PYTHONPATH=src python scripts/diagnose_v2_tal_spatial_anchor_grounding.py \
  --v2-output-dir "$V2" --output-dir "$DIAG"
PYTHONPATH=src python scripts/freeze_v2_tal_stage3g_v3_smoke_selection.py \
  --v2-output-dir "$V2" --output-dir "$SMOKE"
```

For a real engineering smoke, use the frozen selection file as follows.  The
v3 preflight writes its full call plan before model execution.  Run, replay,
and validate use the same new cache.  Review packet overlays are for human
review only: legal boxes do not support an ObservationClaim and do not create a
certificate.

```bash
POLICY=configs/v2/tal_spatial_anchor_grounding_v3_policy.json
SMOKE_OUT=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/v2_tal_stage3g_v3_smoke_run_$(date +%Y%m%d_%H%M%S)

PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding_v3.py \
  --mode preflight --config "$CONFIG" --policy "$POLICY" \
  --stage3f-dir "$STAGE3F" --stage3c-dir "$STAGE3C" --stage3d-dir "$STAGE3D" \
  --stage3e-dir "$STAGE3E" --video-index-dir "$INDEX_OUT" \
  --smoke-selection-manifest "$SMOKE/v2_tal_stage3g_v3_smoke_selection.jsonl" \
  --output-dir "$SMOKE_OUT"
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding_v3.py \
  --mode run --config "$CONFIG" --policy "$POLICY" --output-dir "$SMOKE_OUT"
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding_v3.py \
  --mode replay --config "$CONFIG" --policy "$POLICY" --output-dir "$SMOKE_OUT"
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding_v3.py \
  --mode validate --output-dir "$SMOKE_OUT"
```

For the complete frozen 75-anchor cohort, omit
`--smoke-selection-manifest`.  Make a second distinct output directory and
fresh cache, then compare the two `run/` directories with this zero-model
command:

```bash
FULL_A=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/v2_tal_stage3g_v3_full_a_$(date +%Y%m%d_%H%M%S)
FULL_B=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/v2_tal_stage3g_v3_full_b_$(date +%Y%m%d_%H%M%S)
# Run preflight, run, replay, and validate above for FULL_A and FULL_B, without the smoke option.
CMP=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/v2_tal_stage3g_v3_compare_$(date +%Y%m%d_%H%M%S)
PYTHONPATH=src python scripts/compare_v2_tal_spatial_anchor_grounding_v3_repeats.py \
  --run-a-dir "$FULL_A" --run-b-dir "$FULL_B" --output-dir "$CMP"

cat "$FULL_A/run/v2_tal_stage3g_v3_manifest.json"
cat "$FULL_A/run/v2_tal_stage3g_v3_audit.json"
cat "$FULL_A/run/v2_tal_spatial_anchor_v3_review_packet_index.jsonl"
cat "$CMP/v2_tal_stage3g_v3_independent_repeat_comparison.json"
```

The v3 manifest reports parse/schema compliance, required-role visibility,
legal boxes, actual unlocalized roles, stopping metadata, and fresh/replay
cache counts.  It always reports `certificate_status=NOT_APPLICABLE` and
`new_verified_count=0`.

### Stage 3G-A v3.1: null-bbox output-contract revision

v3.1 is a separate prompt-only contract revision following the immutable v3
smoke observation that two ACTION results used `[null,null,null,null]` for
`NOT_VISIBLE`.  Local-HF native generation metadata is available, but reliable
schema/grammar-constrained decoding is not; the v3.1 policy records
`PROMPT_ONLY_FAIL_CLOSED`.  The parser still rejects every array-shaped null
value and does not import boxes from v3 raw output.

v3.1 smoke reuses the exact four v3 smoke anchor IDs in exactly their frozen
order.  It must run once.  If either ACTION output retains a schema violation,
do not run the complete 75-anchor cohort or change the prompt again in that
run.  The PRE cover frame remains part of the smoke and may correctly yield all
`NOT_VISIBLE` roles.

```bash
cd /home/huihui/codes/MedVid_understanding/relive
conda activate MedVidU-cu124

V3_SMOKE=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/your_v3_smoke_selection_directory
SMOKE31=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/v2_tal_stage3g_v31_smoke_selection_$(date +%Y%m%d_%H%M%S)
OUT31=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/v2_tal_stage3g_v31_smoke_$(date +%Y%m%d_%H%M%S)
POLICY31=configs/v2/tal_spatial_anchor_grounding_v31_policy.json

PYTHONPATH=src python scripts/freeze_v2_tal_stage3g_v31_smoke_selection.py \
  --v3-selection-manifest "$V3_SMOKE/v2_tal_stage3g_v3_smoke_selection.jsonl" \
  --output-dir "$SMOKE31"

PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding_v31.py \
  --mode preflight --config "$CONFIG" --policy "$POLICY31" \
  --stage3f-dir "$STAGE3F" --stage3c-dir "$STAGE3C" --stage3d-dir "$STAGE3D" \
  --stage3e-dir "$STAGE3E" --video-index-dir "$INDEX_OUT" \
  --smoke-selection-manifest "$SMOKE31/v2_tal_stage3g_v31_smoke_selection.jsonl" \
  --output-dir "$OUT31"
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding_v31.py \
  --mode run --config "$CONFIG" --policy "$POLICY31" --output-dir "$OUT31"
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding_v31.py \
  --mode replay --config "$CONFIG" --policy "$POLICY31" --output-dir "$OUT31"
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding_v31.py \
  --mode validate --output-dir "$OUT31"

cat "$OUT31/run/v2_tal_stage3g_v3_1_manifest.json"
cat "$OUT31/run/v2_tal_stage3g_v3_1_audit.json"
cat "$OUT31/run/v2_tal_spatial_anchor_v3_1_review_packet_index.jsonl"
```

Only if the smoke has zero schema violations should the full frozen cohort be
run twice in separate output/cache roots without `--smoke-selection-manifest`.
Then compare the two fresh runs:

```bash
PYTHONPATH=src python scripts/compare_v2_tal_spatial_anchor_grounding_v31_repeats.py \
  --run-a-dir "$FULL_A31" --run-b-dir "$FULL_B31" \
  --output-dir "$COMPARE31"
```

### Stage 3G-A v3.2: token-level constrained JSON contract

v3.2 leaves all v2, v3, and v3.1 source code, caches, and run artifacts
immutable.  It does not trim a trailing character, repair JSON, or import a box
from a rejected response.  Instead, the Local-HF backend binds a versioned
compact JSON prefix grammar to the loaded tokenizer and attaches it to the
actual multimodal `model.generate()` logits path.  Before the object is
complete, EOS is masked; after the one permitted complete object, only EOS is
allowed.  If tokenization exposes no legal successor, EOS is used only to end
that failed attempt and the result records a `TOKEN_CONSTRAINT_FAILURE`; no
text is completed or repaired.

The grammar freezes the task's exact component-role order, three visibility
labels, literal `null` for `NOT_VISIBLE`/`AMBIGUOUS`, and relative-1000 integer
coordinates for `VISIBLE`.  It does not decide visibility or establish that a
box is visually correct.  The independent strict parser/geometry validator
still runs after decoding.  The preflight loads the reviewed Local-HF backend,
audits the actual tokenizer/template binding for every frozen grammar, and
writes those bindings.  The smoke `run` is the required confirmation that the
constraint is active in real multimodal generation.

Use the v3.1 smoke-selection file, not a run result, to preserve the exact four
frozen PRE/ACTION/POST anchor IDs and their order:

```bash
cd /home/huihui/codes/MedVid_understanding/relive
conda activate MedVidU-cu124

POLICY32=configs/v2/tal_spatial_anchor_grounding_v32_policy.json
V31_SELECTION=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/your_v31_smoke_selection
SMOKE32_SELECTION=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/v2_tal_stage3g_v32_smoke_selection_$(date +%Y%m%d_%H%M%S)
OUT32=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/v2_tal_stage3g_v32_smoke_$(date +%Y%m%d_%H%M%S)

PYTHONPATH=src python scripts/freeze_v2_tal_stage3g_v32_smoke_selection.py \
  --v31-selection-manifest "$V31_SELECTION/v2_tal_stage3g_v31_smoke_selection.jsonl" \
  --output-dir "$SMOKE32_SELECTION"

PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding_v32.py \
  --mode preflight --config "$CONFIG" --policy "$POLICY32" \
  --stage3f-dir "$STAGE3F" --stage3c-dir "$STAGE3C" --stage3d-dir "$STAGE3D" \
  --stage3e-dir "$STAGE3E" --video-index-dir "$INDEX_OUT" \
  --smoke-selection-manifest "$SMOKE32_SELECTION/v2_tal_stage3g_v32_smoke_selection.jsonl" \
  --output-dir "$OUT32"
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding_v32.py \
  --mode run --config "$CONFIG" --policy "$POLICY32" --output-dir "$OUT32"
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding_v32.py \
  --mode replay --config "$CONFIG" --policy "$POLICY32" --output-dir "$OUT32"
PYTHONPATH=src python scripts/prepare_v2_tal_spatial_anchor_grounding_v32.py \
  --mode validate --output-dir "$OUT32"

cat "$OUT32/run/v2_tal_stage3g_v3_2_manifest.json"
cat "$OUT32/run/v2_tal_stage3g_v3_2_audit.json"
cat "$OUT32/run/v2_tal_spatial_anchor_groundings_v3_2.jsonl"
cat "$OUT32/run/v2_tal_spatial_anchor_v3_2_review_packet_index.jsonl"
```

Only proceed to the complete 75-anchor cohort if the smoke manifest reports
zero `parse_failure_count`, `schema_violation_count`, and
`constraint_failure_count`.  Do not rerun or modify the smoke in place.  The
PRE cover frame remains a valid diagnostic all-`NOT_VISIBLE` outcome.

```bash
FULL_A32=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/v2_tal_stage3g_v32_full_a_$(date +%Y%m%d_%H%M%S)
FULL_B32=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/v2_tal_stage3g_v32_full_b_$(date +%Y%m%d_%H%M%S)
# For each root: run preflight, run, replay, validate as above, but omit
# --smoke-selection-manifest.  Each root has its own fresh cache.

COMPARE32=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/v2_tal_stage3g_v32_compare_$(date +%Y%m%d_%H%M%S)
PYTHONPATH=src python scripts/compare_v2_tal_spatial_anchor_grounding_v32_repeats.py \
  --run-a-dir "$FULL_A32" --run-b-dir "$FULL_B32" --output-dir "$COMPARE32"
```

The comparison reports raw/parsed/status agreement, role visibility, exact
boxes, and canonical-result equality only.  Every legal box remains subject to
human overlay review; v3.2 creates neither certificates nor `VERIFIED`.
