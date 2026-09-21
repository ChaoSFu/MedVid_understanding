
## Stage 3G human overlay adjudication

Use `scripts/adjudicate_v2_tal_human_overlay_reviews.py` after every frozen
anchor-role pair has one human label.  The validator is zero-model and keeps the
raw grounding JSONL unchanged.  `ACCEPTED` means only that a displayed role box
is usable as a spatial-intervention candidate; it does not establish the
observation claim, create a certificate, or create `VERIFIED` evidence.

```bash
PYTHONPATH=src python scripts/adjudicate_v2_tal_human_overlay_reviews.py \
  --raw-anchor-manifest /path/to/v2_tal_spatial_anchor_groundings_v3_2.jsonl \
  --review-jsonl /path/to/v2_tal_spatial_anchor_human_overlay_review.jsonl \
  --output-dir /path/to/v2_stage3g_human_adjudication
```

The output includes the immutable adjudicated anchor manifest, one selected
anchor decision per observation, intervention-ready components with deterministic
matched-control geometry, a second-pass queue, and validation reports.  Entries
not selected as eligible remain `ABSTAIN` / reacquire / relocalize / relational
routes and must not enter a certificate.

The review JSONL has exactly one row per frozen `(anchor_candidate_id, role)`.
Do not change the anchor, role, or `model_visibility`.  For `VISIBLE`, choose
`ACCEPTED`, `BOX_TOO_BROAD`, `BOX_TOO_TIGHT`, `WRONG_OBJECT`,
`INTERFACE_NOT_LOCALIZED`, `SHOULD_BE_NOT_VISIBLE`, or `UNRESOLVED`.  For
`NOT_VISIBLE`, choose only `NOT_VISIBLE_CONFIRMED` or `UNRESOLVED`; for
`AMBIGUOUS`, choose only `AMBIGUOUS_CONFIRMED` or `UNRESOLVED`.  Illegal,
missing, duplicated, or mismatched rows fail closed.  Duplicate model
groundings with differing human labels are retained as warnings in the
second-pass queue, rather than silently reconciled.
