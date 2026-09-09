# ReliVE-v1 — Reliable Visual Evidence Verification

ReliVE-v1 is a small, training-free research framework for test-time visual evidence verification on a frozen VLM. It changes candidate evidence, claims, support regions, and acquired temporal context; it never trains or updates model parameters. `VERIFIED` means that an Evidence–Claim pair passed its configured protocol. It is not a guarantee of medical truth. KEEP/DROP checks are spatial intervention checks, not causal proof.

The framework separates acquisition from admission: chronological or uniform acquisition assigns an `acquisition_rank`; only verification can admit evidence. It does not use temporal stability, shuffle, or FREEZE sensitivity as admission rules. Synthetic fixtures validate engineering behavior, not the scientific effectiveness of the method.

## Install

Use Python 3.11 or later. From this directory:

```bash
python3.11 -m pip install -e .
```

For a source checkout with the bundled dependencies already available, commands can instead use `PYTHONPATH=src python3.11 -m relive`.

## Supported tasks and runtime isolation

Only `claim_verification` and `action_qa` are implemented. `DVC`, `CVS`, `NAP`, `SA`, `VS`, and `stg` return `UNSUPPORTED`; they are never silently converted to `action_qa`. The STG adapter only records the inspected schema boundary and does not load official STG data or hidden labels.

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

The bundled fixture is clearly synthetic and has no medical claim. It includes one declared-exclusive synthetic contrast fixture so the complete protocol can be exercised deterministically.

```bash
cd relive
PYTHONPATH=src python3.11 -m relive run \
  --config configs/mock_smoke.yaml \
  --runtime examples/synthetic_runtime.jsonl \
  --output-dir runs/mock_smoke \
  --max-samples 2

PYTHONPATH=src python3.11 -m relive audit --run-dir runs/mock_smoke
PYTHONPATH=src python3.11 -m relive evaluate --run-dir runs/mock_smoke
```

The default `smoke` phase permits at most five samples. An explicit `--phase preflight` permits at most ten engineering samples. There is no full benchmark command.

## Configuration and real backends

`configs/mock_smoke.yaml` freezes every test setting. The real template at `configs/real_backend.example.yaml` requires an explicit base URL, endpoint, model identity and revision, credential environment-variable name, generation settings, image ordering, encoding, timeouts, retries, and backend-specific parameters. It never guesses a model path, endpoint, chat template, or sampling setting. Credentials remain in the environment and are neither saved nor logged.

For a configured service, copy the template outside version control, replace all placeholders, set the named credential environment variable, and use a `public_runtime` source. A real smoke run has not been executed by this package unless its run manifest says `synthetic: false` and records the actual backend fingerprint.

| Policy | Required checks |
| --- | --- |
| `acquisition_only` | Never verifies; acquisition ablation only |
| `semantic_only` | Original semantic support |
| `semantic_keep_drop` | Semantic plus KEEP/DROP, no controls |
| `semantic_spatial` | Semantic plus KEEP/DROP and all planned matched controls |
| `semantic_contrast_spatial` | Semantic, declared contrast handling, and matched spatial controls |

The spatial protocol preserves source resolution and emits a pixel audit for `ORIGINAL`, `KEEP_TARGET`, `DROP_TARGET`, and `DROP_MATCHED_CONTROL`. The pure Gaussian-blur/hard-mask operation is extracted from the audited H4 script while leaving its historical implementation untouched. Support regions and target bounding boxes are separate fields. Full-frame regions are retained and marked, not silently discarded.

## Artifacts, cache, and evaluation

Every inference attempt is immutable and content-addressed. Cache identity includes backend fingerprint, prompt contents, ordered frame paths and bytes, preprocessing, claim, region, intervention, and control parameters. Cache hits replay the same logical budget cost while making zero new model calls. A repeated run in the same output directory resumes completed samples; a cache entry with an incomplete raw-attempt chain is not accepted as success.

A new output directory can reuse a cache directory when only certificate policy changes: raw inference remains immutable and certificates are recalculated.

```bash
PYTHONPATH=src python3.11 -m relive run --config my_policy.yaml \
  --runtime runtime.jsonl --output-dir runs/policy_b --cache-dir runs/shared_cache
```

Only `relive evaluate` may open a ground-truth file:

```bash
PYTHONPATH=src python3.11 -m relive evaluate \
  --run-dir runs/policy_b --ground-truth held_out_gt.jsonl --output evaluation.json
```

Its task-answer exact match is descriptive and explicitly non-official. It also reports strict coverage, fallback rate, certificate/check status distributions, calls, latency, adaptation snapshots, and spatial/temporal alignment only when matching prediction and annotation types exist. Spatial IoU does not establish semantic truth, and time non-overlap does not establish action absence.

Strict answers cite only `VERIFIED` Evidence–Claim pairs and abstain on incomplete coverage. `benchmark_forced` is a separate, explicitly marked fallback output and is excluded from strict metrics.
