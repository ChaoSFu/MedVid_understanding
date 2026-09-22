
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

### Explicit canonical labels after `UNIFY_LABELS`

The v1 warning-decision file records a decision and rationale only. It never
parses a phrase such as `Canonical label: ACCEPTED` from reviewer prose. When a
reviewer chooses `UNIFY_LABELS`, create a separate v2 file with one explicit
`resolved_label` per grounding signature, then derive a new review and new
diagnostic-only routes:

```bash
PYTHONPATH=src python scripts/resolve_reviewed_anchor_duplicate_groundings.py \
  --mode prepare \
  --warning-adjudication "$MANUAL/duplicate_grounding_adjudication.jsonl" \
  --output-dir "$RESOLUTION" \
  --template-output "$RESOLUTION/canonical_label_adjudication.template.jsonl"

# A human fills resolved_label in every row of a copied template. No source
# review, raw grounding, or warning file is edited.
PYTHONPATH=src python scripts/resolve_reviewed_anchor_duplicate_groundings.py \
  --mode apply \
  --raw-grounding "$RAW" \
  --human-review "$REVIEW" \
  --warning-queue "$QUEUE" \
  --warning-adjudication "$MANUAL/duplicate_grounding_adjudication.jsonl" \
  --canonical-label-adjudication "$RESOLUTION/canonical_label_adjudication.jsonl" \
  --output-dir "$RESOLUTION/applied"
```

`apply` emits a derived review plus a new `adjudicated_anchor_manifest.jsonl`,
`observation_anchor_decisions.jsonl`, and `eligible_anchor_manifest.jsonl`.
The formal preflight binds all of those files through
`canonical_label_resolution_manifest.json`. If the corrected route no longer
has exactly five eligible observations, preflight fails closed; it does not
replace the missing candidate.

The frozen plan has exactly `ORIGINAL`, `KEEP_TARGET`, `DROP_TARGET`, and
`DROP_MATCHED_CONTROL`. `FULL_GRAY` and `MISMATCHED_PUBLIC` are not formal
variants and cannot enter certificate admission. The machine-generated report
is protocol evidence, not a medical truth assertion.

## Reviewed-anchor R0 audit and one-round R1 recomposition

`reviewed_anchor_spatial_recomposition.py` first audits an immutable R0
five-candidate run *and replay*. It reports each candidate's four semantic
outcomes, pixel-audit status, failure reasons, certificate status, and fresh /
replay call counts. Missing candidates or a replay cache miss stop R1.

`UNIFY_LABELS` is treated conservatively. The historical duplicate-warning
schema has no `resolved_label`; recording that decision alone does not alter a
review, recompute routes, or rebuild the eligible manifest. Such an input is
reported as `DIAGNOSTIC_ONLY_INPUT_ADJUDICATION_NOT_APPLIED` and blocks R1.

For a frozen `ACTION_CORE_PRESSING` R0 failure with `ORIGINAL=SUPPORTED`,
`KEEP_TARGET=INSUFFICIENT`, `DROP_TARGET=SUPPORTED`, and a supported matched
control, R1 performs exactly one `SPATIAL_RECOMPOSE`. Its target is the binary
union of `OPERATOR_HAND`, `BASE_PLATE`, and `HAND_BASE_INTERFACE`; it never
uses an enclosing rectangle. The matched control is a non-overlapping uniform
translation of every component rectangle, preserving the composite geometry.
The opaque-gray operator, semantic verifier, pixel audit, cache and
`semantic_spatial` certificate policy remain unchanged. `NO_R2_AFTER_R1` is a
hard cycle guard.
