# ReliVE-v2 protocol split manifests

`protocol_dev_manifest.jsonl`, `calibration_manifest.jsonl`, and
`blind_test_manifest.jsonl` are intentionally empty immutable-input templates.
Populate each from a separately frozen full-video cohort, never from model
results.  Each JSONL record has exactly:

```json
{"protocol_version":"relive-v2-protocol-first-v1","program_sha256":"<64 hex>","split":"protocol_dev","source_video_sha256":"<64 hex>","frame_sha256s":["<64 hex>"],"mode":"automatic"}
```

The `split` value changes to `calibration` or `blind_test` in the corresponding
file. A source-video hash and every frame hash must occur in exactly one split.
The validator rejects cross-split reuse before any model, cache, or certificate
path can run.

After a completed pilot has an immutable closure manifest, the repository may
be tagged from the reviewed commit, for example:

```bash
git tag -a relive-r1-pilot-closed -m "Close the reviewed-anchor R0/R1 protocol-development pilot"
git push origin relive-r1-pilot-closed
```

Create this tag only after `freeze_relive_v2_pilot_closure.py` returns
`PILOT_CLOSED`; the tag identifies a code revision and never modifies a run
artifact.
