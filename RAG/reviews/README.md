# Source-scope review preset

`source_review.assistant_screened.v1.jsonl` contains 46 source-bound decisions:
45 INCLUDE and 1 EXCLUDE. It can be passed directly to `--source-review`.

This preset was prepared by Codex after inspecting the actual PDF front matter,
the Open Surgery page bodies and the complete extracted-text benchmark screen.
The reviewer is explicitly `codex_source_scope_review_v1`, not a human's name.
The paired audit records the exact PDF hash, source-identifier quotation and PDF
page supporting each source assessment. The original source manifest, template
and knowledge-library snapshot are unchanged.

The excluded source is `Laparoscopic instruments – OpenSurgicalVision.pdf`:
it explicitly mixes the project's CVAT labels, synthetic examples and mask
annotation rules. This version does not admit selected paragraphs from it.

INCLUDE means permission for later source-grounded extraction from a general
reference. It does not assert medical correctness, source completeness, legal
clearance, or approval of the generated cards. All generated cards remain
`review_status=pending`; figure interpretation, marketing claims, contextual
clinical recommendations and possible source conflicts still need the existing
extraction/validation gates and subsequent professional review. Navigation,
irrelevant content and duplicate PDFs may contribute zero cards.

On a server worktree already at the previous commit, fetch the updated review
without switching branches or overwriting a prior human file:

```bash
git fetch origin codex/relive-surgical-visual-knowledge-v01
export SOURCE_REVIEW="$RAG_RUN/source_review.assistant_screened.v1.jsonl"
git show FETCH_HEAD:RAG/reviews/source_review.assistant_screened.v1.jsonl > "$SOURCE_REVIEW"
# Persist this non-secret variable for a restarted terminal.
printf 'export SOURCE_REVIEW=%q\n' "$SOURCE_REVIEW" >> "$RAG_RUN/server_variables.sh"
```

Then run the same three-document smoke command with `--source-review
"$SOURCE_REVIEW"`. If a server inspection reports a source-ID, SHA or title
mismatch, stop and inspect the difference; never patch the binding fields to
make the gate pass. The pre-existing human review file remains available.
