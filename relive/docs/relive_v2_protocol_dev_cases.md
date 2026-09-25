# ReliVE v2 protocol_dev 9-case contracts

Each file in `protocol_dev/cases/` is a portable
`relive-v2-protocol-dev-case-v1` case. It contains no absolute data-root path
and no oracle coordinate. Oracle coordinates, when humans create them, must be
stored outside this directory; a case can reference only an artifact name and
its readiness state. `assert_automatic_certificate_input_safe()` rejects
oracle fields before any automatic certificate path.

The source draft is treated as a human draft, not a runtime input. The current
draft had concatenated top-level objects and a full-width comma. The normalizer
records those facts in every affected case provenance; it does not rewrite the
draft or turn missing review fields into an admission.

```bash
cd /home/huihui/codes/MedVid_understanding/relive
conda activate MedVidU-cu124

# Schema-only: no data root, no model, no cache, no certificate.
PYTHONPATH=src python scripts/audit_v2_protocol_dev_cases.py \
  --cases-dir protocol_dev/cases \
  --output-dir /mnt/hdd/huihui/MedVid_understanding/relive_output/real/protocol_dev_cases_audit_$(date +%Y%m%d_%H%M%S)
```

For an optional existence check, create a local JSON mapping that is not
committed, e.g. `configs/v2/protocol_dev_data_roots.local.json`, from the
example. It maps `*_ROOT` keys to a local dataset root and never changes a case
file.

```bash
PYTHONPATH=src python scripts/audit_v2_protocol_dev_cases.py \
  --cases-dir protocol_dev/cases \
  --oracle-dir /path/to/separately-stored-human-oracle-artifacts \
  --data-roots configs/v2/protocol_dev_data_roots.local.json \
  --output-dir /path/to/new_audit
```

`--strict` exits nonzero whenever any case is `PENDING` or `INVALID`. It is
appropriate only after automated source materialization and frame resolution,
plus human reviewer identities/decisions, predicate components, keyframes, and
externally stored oracle references.

## Completion responsibilities

`reviews/protocol_dev_human_completion_queue.jsonl` contains only fields that
need human judgement: review decision, reviewer ID, rationale, entity/claim
components, exact keyframes, and oracle coordinates. Humans must not write
record hashes, timestamps, frame counts, normalized paths, or manifest hashes.
Those are machine-derived after an input source record is available.

`materialize_v2_protocol_dev_source_records.py` reads a QA JSON or JSONL file,
projects only public binding fields, and emits canonical source-record hashes.
It neither edits a case nor reads non-human conversation values. A missing or
ambiguous selector remains unresolved. `resolve_v2_protocol_dev_frame_paths.py`
derives a relative path pattern only when every selected frame has a unique,
existing numeric filename beneath its configured dataset root; it never guesses
or updates a case.

```bash
PYTHONPATH=src python scripts/materialize_v2_protocol_dev_source_records.py \
  --cases-dir protocol_dev/cases \
  --qa-file /path/to/raw_qa.json \
  --output-dir /path/to/immutable_source_materialization

PYTHONPATH=src python scripts/resolve_v2_protocol_dev_frame_paths.py \
  --cases-dir protocol_dev/cases \
  --data-roots configs/v2/protocol_dev_data_roots.local.json \
  --output-dir /path/to/immutable_frame_resolution
```

## Frame-path binding is fail-closed

`--strict` on the frame resolver writes its inspection artifacts and then exits
nonzero unless every one of the nine cases has an unambiguous, existing frame
pattern. It never changes a canonical case. The derived
`protocol_dev_frame_pattern_resolution.jsonl` is an audit artifact and must
not be edited by hand.

```bash
PYTHONPATH=src python scripts/inspect_v2_protocol_dev_ego_candidates.py \
  --cases-dir protocol_dev/cases \
  --data-roots configs/v2/protocol_dev_data_roots.local.json \
  --render-previews \
  --output-dir /path/to/immutable_ego_candidate_review

PYTHONPATH=src python scripts/prepare_v2_protocol_dev_frame_binding_queue.py \
  --cases-dir protocol_dev/cases \
  --output-dir /path/to/immutable_frame_binding_queue
```

For CoPESD, the source sample-ID bounds are not image-frame numbers. The
timebase audit accepts only JSONL rows with `video_id`, `frame_id`,
`timestamp_seconds`, `relative_path`, `timebase_source`, and a 64-character
`source_reference_sha256`. It then emits the image/time/file mapping for human
keyframe selection.

The checked-in CoPESD public-frame registry records only portable relative
paths and file numbers supplied for `012626`; it deliberately marks timestamp
mapping as unresolved. It contains no assistant response or box data.

```bash
PYTHONPATH=src python scripts/audit_v2_protocol_dev_copesd_timebase.py \
  --cases-dir protocol_dev/cases \
  --data-roots configs/v2/protocol_dev_data_roots.local.json \
  --public-frame-registry protocol_dev/registries/copesd_012626_part0_public_frame_registry.json \
  --output-dir /path/to/immutable_copesd_public_frame_audit
```

## Author-confirmed CoPESD timebase

`registries/copesd_012626_part0_author_confirmed_timebase.jsonl` records the
portable file/time mapping for `012626` after a human confirmed the 1-FPS base
sequence and its 1408-to-26-second anchor. The case input samples only even
base-sequence frame IDs, which is a 0.5-FPS input selection; it does not alter
the source timestamp formula. The mapping's source-reference SHA-256 binds
`reviews/author_confirmed_timebase_constraints.jsonl`.

PD-S-08 therefore has a fixed evidence frame range `1434–1446` and a mapped
seconds interval `52–64`. Exact keyframes remain a human choice only from the
listed even IDs `1434, 1436, 1438, 1440, 1442, 1444, 1446`; path existence and
file SHA-256 must still be verified from `COPESD_ROOT` before promotion.
