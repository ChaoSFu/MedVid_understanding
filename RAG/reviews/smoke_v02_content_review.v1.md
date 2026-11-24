# v0.2 server smoke content review

Status: **ENGINEERING_PASS_CONTENT_REVIEW_PENDING**. This is an assistant
provenance/content screen, not professional medical approval.

The input library (`99e70fe5…`), exclusions (`772df283…`) and unit audit
(`2c152a3c…`) matched the server summary. All 46 PDF hashes, 31 unit input
hashes, 47 retained-card admission checks and 92 fact-quote entries matched.
All retained cards remain pending, with reviewer/reviewed_at empty. No model,
certificate or historical artifact was written or reclassified by this audit.

## Candidate accounting

87 generated candidates = 36 initially rejected + 51 initially admitted.
Four more cards were quarantined by the existing normalization heuristic,
leaving 47 retained and 40 rejected/quarantined. Three units generated no card:
institutional eligibility policy, non-discriminative specialty overview, and
image-credit metadata. These are appropriate abstentions, not technical errors.

The same-entity polarity detector reacts to any `not/no/never/without` in a
summary. Its two flagged pairs are **not established medical contradictions**:

- DualKnife J: a device capability description versus a complementary account
  that includes “no longer necessary” for device changing.
- Mayo scissors: a description with “the source gives no blade shape…” versus
  a second source providing straight/curved variants. Missing detail in one
  source is not a denial of detail in another.

The four cards remain quarantined. This review neither auto-merges them nor
restores their admission; the weak lexical flag is a review cue only.

## Priority review

[The queue](smoke_v02_priority_review_queue.v1.jsonl) has 17 entries: 15 retained
cards with review flags, plus the two quarantined pairs. Categories overlap:

| Category | Retained cards | Meaning |
|---|---:|---|
| Source-attributed technique/safety/benefit | 6 | Keep qualifiers; source statements do not prove observed efficacy or safety. |
| Partial source context | 3 | Check original PDF continuation before extending the claim. |
| Weak visual discrimination | 7 | Functional/class knowledge alone may not identify a specific instrument. |

None of these flags is an automatic rejection or a clinical assessment. All
other cards still need normal medical review. Text-only extraction has not
verified figure pixels. The 92 matching quotes establish traceability, not
semantic entailment of every derived suggestion.

The complete machine-readable audit, card list and model-reported omissions
are in [smoke_v02_content_review.v1.json](smoke_v02_content_review.v1.json).
Budget-limited omissions (for example individual instrument entries) and
incomplete source paragraphs remain explicit; coverage is not exhaustive.

## Full-source readiness

An API-free inspection of all 46 PDFs/2,136 pages found 45 reviewed includes,
one reviewed exclude, and no pending source decisions. The initial three
oversized blocks were actually bibliography continuations on pages 21–23 of
one review article. Selector v0.2.1 handles explicit bibliography sections
across pages, while preserving publisher metadata and following medical prose.
The original 31 smoke unit IDs/input hashes remain unchanged. All 31 historical
extraction request keys were reproduced offline from the recorded configuration
and exact current inputs, prompt/schema and adapter. Those cached requests remain
reusable. Changing build code still requires a new output directory.

[Full readiness](full_text_readiness.v021.json) records 1,471 candidate units,
1,451 planned model units, and zero oversized blocks. Excluded/duplicate-source
units are not sent to the model. Before cache reuse and retries, the full text
build would need 1,451–2,902 calls. A 3,500-call ceiling is a bounded execution
choice, not an assertion that every unit will complete or produce knowledge.
Only pending-card generation may proceed; medical approval, image-based
coverage and any trusted retrieval admission remain separate work.
