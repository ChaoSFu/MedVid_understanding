# Server 24k-token smoke: source and engineering review

This is an assistant source/engineering audit, **not medical approval**.
The machine-readable record is [smoke24k_review.v1.json](smoke24k_review.v1.json).

The three user-supplied artifacts exactly matched the server summary hashes:
library `2db5a39e…`, exclusions `5d7570ed…`, binding `566df88c…`.
The binding's code hash was reproduced from commit
`1558e9863a4048d46214deb140a72a97cd303a48`; all 46 input PDF hashes and the
review-file hash matched. All 51 fact-quote entries in the 17 retained cards
matched their actual PDF blocks. This checks provenance, not medical entailment.

The smoke is still incomplete: the server report had one truncated multi-page
unit. Among 34 rejected candidates, 22 had unsupported additions, 8 were
uncertain, and 4 failed section-locator binding. Many rejected additions were
in guidance/comparators rather than the main source paraphrase. Keep those
records and their reasons; do not delete offending fields to auto-admit cards.

All four titles called `INVENTED_SECTION` actually occur in the PDF. They were
not the supplied unit's section locator. The new reason accurately describes
this as `SECTION_NOT_IN_SUPPLIED_LOCATOR`; historical records remain unchanged.

The retained institutional ESD eligibility card is a **nonvisual scope concern**:
provenance can be correct while the item is unsuitable for a visual-reference
corpus. Model screening of visual relevance is now separate from source
consistency. Neither screen replaces professional review. Likewise, source
approval does not establish a commercial device
performance claim as a clinical outcome or an observed video event.

Extractor/selector v0.2 therefore uses bounded complete paragraphs/pages,
at most four compact cards per unit, optional sparse guidance, and an explicit
visual-relevance gate. No model calls were made during this local review.
No historical card, rejection, PDF, R0/R1 output, cache or certificate was changed.
No historical card is retrospectively admitted by the new policy.
