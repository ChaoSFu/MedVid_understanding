# ReliVE-Spatial: Spatial Evidence Reliability

Version: `relive-spatial-evidence-reliability-v1` (prospectively frozen)

ReliVE-Spatial studies whether a frozen VLM can identify and test spatial evidence for one local
claim in one temporally relevant frame. It does not replace historical ReliVE Phase 2/3 results and
does not perform temporal retrieval.

## Scientific contract

The model is frozen. Only spatial evidence changes. A proposal is tested with the same semantic
verifier under `ORIGINAL`, `KEEP_TARGET`, `DROP_TARGET`, and a matched control when geometry permits.
Opaque gray means unavailable information, never negative evidence.

`KEEP=SUPPORTED` establishes sufficiency. It does not establish dependency. If
`DROP=SUPPORTED`, the region remains valid sufficient evidence and the controller performs exactly
one neutral residual-support search. At most two evidence regions may be admitted.

The deterministic state transitions are:

- Any unparseable or technically failed verifier result → `TECHNICAL_FAILURE`; it must never be
  reported as `BASE_UNSUPPORTED` or used in an oracle feasibility rate.
- `ORIGINAL != SUPPORTED` → `BASE_UNSUPPORTED` → stop.
- `KEEP != SUPPORTED` → one `SUFFICIENCY_REPAIR`; a second repair is forbidden.
- `KEEP = SUPPORTED && DROP != SUPPORTED` → dependency status; if control is geometrically
  unavailable, the explicit downgraded status is `DEPENDENCY_SUPPORTED_NO_CONTROL`.
- A constructed control that is not supported → `INTERVENTION_UNSTABLE` → stop.
- `KEEP = SUPPORTED && DROP = SUPPORTED` → one `RESIDUAL_SUPPORT_SEARCH`.
- A residual proposal must pass `KEEP_RESIDUAL`; then the union must pass `KEEP_UNION` and
  fail support under `DROP_UNION` to yield `EVIDENCE_SET_VERIFIED`.
- Support persisting after `DROP_UNION` yields `DEPENDENCY_UNRESOLVED`, not a hallucination claim.

## Isolation

Dataset preparation writes two physically separate manifests. `runtime_manifest.jsonl` contains a
single frame and a normalized claim and is automatically scanned for spatial-GT fields.
`oracle_manifest.jsonl` contains the original caption and GT box. Runtime code does not import the
oracle runner or evaluation module. Only `run_oracle_feasibility` and `evaluate` accept an oracle
path.

## Commands

All commands run from `relive/` with `PYTHONPATH=src`.

```bash
DATASET=/path/to/medvidu_eccv2026_trainval.json
FRAMES=/path/to/public/frame/root
DATA_OUT=/path/to/artifacts/relive_spatial/dataset

PYTHONPATH=src python -m relive.spatial_reliability.prepare_rc_dataset \
  --source-json "$DATASET" \
  --source-prefix /root/data \
  --frame-root "$FRAMES" \
  --dev-count 20 \
  --output-dir "$DATA_OUT"
```

Oracle feasibility (GT is opened only by this command):

```bash
CONFIG=configs/qwen35_9b_relive_spatial_verifier_v3.yaml
CACHE=/path/to/immutable_relive_spatial_cache
ORACLE_OUT=/path/to/artifacts/relive_spatial/oracle_feasibility

PYTHONPATH=src python -m relive.spatial_reliability.run_oracle_feasibility \
  --runtime-manifest "$DATA_OUT/runtime_manifest.jsonl" \
  --oracle-manifest "$DATA_OUT/oracle_manifest.jsonl" \
  --config "$CONFIG" --cache-dir "$CACHE" --split development \
  --output-dir "$ORACLE_OUT"
```

The spatial semantic verifier v3 is a single-image closed JSON contract. It requires exactly
`status`, a short `observation`, and integer `frame_index: 0`; a single complete JSON Markdown fence
is tolerated, while partial fences, surrounding text, and the legacy `frame_references` field are
rejected. The reviewed Qwen3.5-9B config freezes deterministic
generation at 256 new tokens. Its prompt version and generation settings are part of cache identity,
so verifier-v1/v2 responses are neither reused nor repaired. Oracle CLI status is nonzero when any
verdict has a technical or parse failure, and scientific rates remain unavailable in that case.

Runtime localization uses the separately versioned sufficiency-proposal v3 token-JSON contract.
The actual LocalHF `generate()` logits are masked by a prefix grammar: only a compact `PROPOSED`
object with four integer coordinates, or a compact `UNRESOLVED` object with a null box, can reach
EOS. Markdown fences, aliases, trailing text, semantic verdicts, and the legacy `image_id`/`bbox_2d`
surface have no legal token continuation. The independent parser then rechecks exact fields, reason
codes, coordinate order, and range; grammar validity is not evidence that a localized box is
correct. Premature EOS, token-limit termination, or no legal successor is recorded as a constraint
failure and routed to `TECHNICAL_FAILURE` without repair.

The grammar version, grammar-spec SHA-256, prompt version, checkpoint/generation fingerprint,
tokenizer vocabulary SHA-256, and chat-template SHA-256 are retained in cache/run provenance. The
same contract applies to initial proposal, sufficiency repair, and residual search. Existing v2
development artifacts and caches remain unchanged and are not reinterpreted.

Formal runtime and exact replay:

```bash
RUN_OUT=/path/to/artifacts/relive_spatial/dev/run
REPLAY_OUT=/path/to/artifacts/relive_spatial/dev/replay
RUNTIME_CONFIG=configs/qwen35_9b_relive_spatial_proposal_v3.yaml

PYTHONPATH=src python -m relive.spatial_reliability.run \
  --runtime-manifest "$DATA_OUT/runtime_manifest.jsonl" \
  --config "$RUNTIME_CONFIG" --cache-dir "$CACHE" --mode relive_spatial \
  --execution-mode run --split development --output-dir "$RUN_OUT"

PYTHONPATH=src python -m relive.spatial_reliability.run \
  --runtime-manifest "$DATA_OUT/runtime_manifest.jsonl" \
  --config "$RUNTIME_CONFIG" --cache-dir "$CACHE" --mode relive_spatial \
  --execution-mode replay --split development --output-dir "$REPLAY_OUT"
```

The replay passes only when `cache_audit.json` reports zero new model calls. Baselines use the same
command with `--mode one_shot`, `verify_only`, or `generic_retry` and separate output directories.

Post-hoc GT evaluation:

```bash
PYTHONPATH=src python -m relive.spatial_reliability.evaluate \
  --runtime-results "$RUN_OUT/sample_results.jsonl" \
  --runtime-manifest "$DATA_OUT/runtime_manifest.jsonl" \
  --oracle-manifest "$DATA_OUT/oracle_manifest.jsonl" \
  --output-dir /path/to/artifacts/relive_spatial/evaluation
```

## Limits

The normalizer and scope router are deliberately conservative and may exclude many valid captions.
Region Caption boxes may be sufficient but non-necessary. Matched control geometry does not prove
that a control contains no evidence. These are reported outcomes, not silently tuned away.
