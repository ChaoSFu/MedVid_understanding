# Phase 4A-0 development-control cohort: ROI preparation

`phase4a0_development_controls_10.selection.draft.jsonl` transcribes the ten
human-selected claims and one-frame windows from the supplied public-data
worksheet. It contains no ROI, matched control, model output, certificate
output, answer, or ground truth.

Run `scripts/phase4a0_prepare_development_controls.py` on the server before
selecting any ROI. The command validates each source-record identity, maps the
selected public frame, recomputes its SHA-256, and writes:

- one public-frame preview per development case;
- `phase4a0_development_roi_freeze.template.jsonl`, with the exact claim,
  public frame ID/path/SHA-256, and blank target/control ROI fields; and
- a zero-model provenance report.

Fill each target ROI and matched-control ROI only after reviewing those exact
previews. Both are normalized `[x1, y1, x2, y2]` coordinates, measured from
the upper-left image corner. The matched control must have the same width and
height as the target ROI and must not overlap it. Do not modify the claim,
source-record identity, frozen order, public path, or SHA-256.

This cohort intentionally allows multiple development cases from the same
source record. It is therefore not an input to the prospective Phase 3.5
fresh-runtime generator, whose unique-source rule remains unchanged.

Before sending any filled ROI sheet for freezing, reconsider the claim scope:

- `dev-002`, `dev-007`, `dev-008`, `dev-009`, and `dev-010` are the closest to
  single local visual facts.
- `dev-001`, `dev-003`, `dev-004`, `dev-005`, and `dev-006` include a
  holding/position/contact relation. They may fail the independent human
  single-ROI eligibility review even if their pixels are clear. Keep their
  wording unchanged unless you explicitly choose a new visual-only claim
  before the ROI freeze.

Every later Phase 4A-0 result remains diagnostic-only and cannot create a
certificate or VERIFIED outcome.
