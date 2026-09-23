# ReliVE v2 protocol_dev data contracts

`protocol_dev` is an isolated, full-video development cohort. It is not a
calibration cohort, a blind test, a model input, or a certificate result. The
builder is model-free and never opens a cache, runs R0/R1, or creates a
certificate.

The immutable precondition report binds the accepted differential policy, its
fixture-only acceptance report, and the closed pilot manifest. The pilot tag
`relive-r1-pilot-closed` remains an historical pointer and is never rewritten.

## Stage A: candidate video review

Each line of `protocol_dev_stage_a_candidate_review.jsonl` has this closed
schema:

```json
{"candidate_id":"...","dataset":"...","video_id":"...","source_record_uid":"source_record_...","source_record_canonical_hashes":["<64 hex>"],"source_task":"...","proposed_claim_type":"SPATIAL_RELATION","source_temporal_span":{"start_seconds":0.0,"end_seconds":1.0},"media_path":"/absolute/path","decision":"ADMIT","reviewer_id":"...","rationale":"..."}
```

`source_record_uid` is recomputed from dataset, video ID, sorted full
source-record canonical hashes, and source task. A question, source ID hint,
or video ID alone cannot bind a record. A candidate must be `ADMIT`, `REJECT`,
or `RESERVE`; blank decision, reviewer ID, or rationale preserves only
`PREVIEW_ONLY` status.

The supplied prefilter CSV is a non-authoritative source for creating the
template. It contains no trusted media path, canonical record hash, reviewer
ID, or completed review decision, so it cannot freeze a cohort.

## Stage B: paired case definition

Every admitted video has exactly two records sharing a `pair_id`: `TRUE` and
`FALSE`. The false record may change only `claim.predicate`; subject, object,
qualifiers, entity list, windows, evidence roles, source binding, and media
path must remain equal. The only allowed claim types are
`SPATIAL_RELATION`, `CONTACT_ACTION`, and `POSTCONDITION_PERSISTENCE`.

Stage B retains the human oracle tube/mask only in its oracle output. No human
label, rationale, changed predicate, or oracle field may occur in the runtime
manifest.

## Frozen cohort criteria

A frozen output requires exactly three distinct full videos for each claim
type, nine videos overall, and 18 paired cases. It rejects duplicate source
UIDs, videos, claims, pilot video `NurViD/vR0_BaXYcE4`, missing media, and
video reuse in an explicitly supplied external calibration/blind split
registry. The registry is mandatory for a frozen cohort (an empty strict JSONL
file is a valid declaration that no external split has yet been registered).
A failed condition
produces empty runtime/ground-truth/oracle outputs and a hash-bound
`PREVIEW_ONLY` audit; it never quietly admits a smaller cohort.

## Output separation

- `protocol_dev_runtime_manifest.jsonl`: runtime-visible claim, media, window,
  entity, and evidence-role fields only.
- `protocol_dev_ground_truth_manifest.jsonl`: true/false labels and manual
  pairing metadata only.
- `protocol_dev_oracle_evidence_manifest.jsonl`: human oracle tube/mask only,
  marked `HUMAN_ORACLE` and `automatic_certificate_eligible: false`.
- `protocol_dev_freeze_manifest.json`: hashes, split, policy/precondition
  bindings, counts, and freeze status.
- `protocol_dev_audit_report.json`: all admission checks and any blocking
  reasons.

No output in this stage is an automatic certificate or `VERIFIED` result.

## Server workflow

All commands below use `python` and perform no model, cache, R0/R1, or
certificate operation.

```bash
cd /home/huihui/codes/MedVid_understanding/relive
conda activate MedVidU-cu124

POLICY=configs/v2/differential_evidence_policy.yaml
ACCEPTANCE=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/differential_policy_acceptance_20260923_103409
CLOSURE=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/relive_v2_pilot_closure_20260922_150257/relive_v2_pilot_closure_manifest.json
PREFILTER=/home/huihui/codes/MedVid_understanding/outputs/protocol_dev_prefilter_review.csv
ROOT_OUT=/mnt/hdd/huihui/MedVid_understanding/relive_output/real/protocol_dev_contract_$(date +%Y%m%d_%H%M%S)
```

First freeze the accepted references and create the incomplete Stage-A form:

```bash
PYTHONPATH=src python scripts/freeze_v2_protocol_dev_preconditions.py \
  --policy "$POLICY" --acceptance-dir "$ACCEPTANCE" --pilot-closure "$CLOSURE" \
  --accepted-code-commit e9e2949f5a2050e140a9cd36d52bc09edcb1d907 \
  --output-dir "$ROOT_OUT/preconditions"

PYTHONPATH=src python scripts/prepare_v2_protocol_dev_stage_a.py \
  --mode from-prefilter --prefilter-csv "$PREFILTER" --output-dir "$ROOT_OUT/review_forms"
```

Humans must fill
`$ROOT_OUT/review_forms/protocol_dev_stage_a_candidate_review.jsonl` with
canonical source record hashes, span, resolvable media path, decision, reviewer
ID, and rationale. Compute the source UID in a new derived file; this command
does not fill any human decision. `REJECT` and `RESERVE` are allowed;
only nine `ADMIT` records can be used in a freeze. Generate the two-row-per-
admit Stage-B form in a *new* directory, then humans define paired true/false
claims and oracle evidence there:

```bash
PYTHONPATH=src python scripts/prepare_v2_protocol_dev_stage_a.py \
  --mode canonicalize-source-uids \
  --stage-a "$ROOT_OUT/review_forms/protocol_dev_stage_a_candidate_review.jsonl" \
  --output-dir "$ROOT_OUT/derived_stage_a"

PYTHONPATH=src python scripts/prepare_v2_protocol_dev_stage_a.py \
  --mode stage-b-template \
  --stage-a "$ROOT_OUT/derived_stage_a/protocol_dev_stage_a_candidate_review.derived.jsonl" \
  --output-dir "$ROOT_OUT/stage_b_form"
```

Create `external_video_splits.jsonl` as a strict JSONL registry of already
allocated full videos; each nonempty line is
`{"dataset":"...","video_id":"...","split":"calibration"}` or
`blind_test`. An empty file is an explicit declaration that neither split has
any registered video. Then build and validate. The builder remains
`PREVIEW_ONLY` until all nine videos and 18 cases meet the contract.

```bash
SPLITS="$ROOT_OUT/external_video_splits.jsonl"
: > "$SPLITS"

PYTHONPATH=src python scripts/build_v2_protocol_dev_manifests.py \
  --policy "$POLICY" \
  --precondition-freeze-report "$ROOT_OUT/preconditions/protocol_dev_precondition_freeze_report.json" \
  --stage-a "$ROOT_OUT/derived_stage_a/protocol_dev_stage_a_candidate_review.derived.jsonl" \
  --stage-b "$ROOT_OUT/stage_b_form/protocol_dev_stage_b_case_definition.jsonl" \
  --external-split-registry "$SPLITS" --output-dir "$ROOT_OUT/freeze"

PYTHONPATH=src python scripts/validate_v2_protocol_dev_manifests.py \
  --policy "$POLICY" --freeze-manifest "$ROOT_OUT/freeze/protocol_dev_freeze_manifest.json"
```
