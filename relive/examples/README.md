
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

# Reviewed-anchor formal intervention

`reviewed_anchor_formal_intervention.py` is the only bridge from a completed
Stage 3G human overlay adjudication to the existing `semantic_spatial` protocol.
It never treats `ACCEPTED` as semantic support: review data only admits exactly
five frozen observation anchors to an independent verifier run.  It reuses the
registered opaque-gray operator, core matched control generation, pixel audit,
semantic verifier, cache and certificate builder.

Before preflight, create a human-only duplicate-grounding adjudication template:

```bash
PYTHONPATH=src python scripts/reviewed_anchor_formal_intervention.py \
  --mode prepare-warning-adjudication \
  --output-dir "$OUT" \
  --warning-queue "$ADJ/schema_issue_candidates.jsonl" \
  --warning-template-output "$OUT/warning_adjudication.template.jsonl"
```

For every row, a reviewer must set `decision` to `UNIFY_LABELS`,
`CONTEXT_SPECIFIC_ALLOWED`, or `UNRESOLVED`; non-unresolved decisions require a
reviewer id and rationale.  This does not edit a raw grounding or review label.
An unresolved duplicate warning that touches an eligible required role makes
preflight write `FORMAL_RUN_BLOCKED` and prevents `run`.

The frozen plan has exactly `ORIGINAL`, `KEEP_TARGET`, `DROP_TARGET`, and
`DROP_MATCHED_CONTROL`. `FULL_GRAY` and `MISMATCHED_PUBLIC` are not formal
variants and cannot enter certificate admission. The machine-generated report
is protocol evidence, not a medical truth assertion.
